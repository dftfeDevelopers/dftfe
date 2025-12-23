// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author Gourab Panigrahi
//

#include <MatrixFree.h>

namespace dftfe
{
  template <typename T,
            dftfe::utils::MemorySpace memorySpace,
            unsigned int              nDofsPerDim,
            unsigned int              nQuadPointsPerDim,
            unsigned int              batchSize,
            unsigned int              subBatchSize>
  MatrixFree<T,
             memorySpace,
             nDofsPerDim,
             nQuadPointsPerDim,
             batchSize,
             subBatchSize>::
    MatrixFree(const MPI_Comm                       &mpi_comm,
               std::shared_ptr<dftfe::basis::FEBasisOperations<
                 dataTypes::number,
                 double,
                 dftfe::utils::MemorySpace::HOST>>   basisOperationsPtrHost,
               std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<
                 dftfe::utils::MemorySpace::DEVICE>> BLASWrapperPtr,
               const bool                            useDevice,
               const unsigned int                    operatorID,
               const unsigned int                    quadratureID,
               const unsigned int                    nVectors)
    : mpi_communicator(mpi_comm)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm))
    , this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_comm))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0))
    , d_basisOperationsPtrHost(basisOperationsPtrHost)
    , d_BLASWrapperPtr(BLASWrapperPtr)
    , d_useDevice(useDevice)
    , d_operatorID(operatorID)
    , d_quadratureID(quadratureID)
    , d_nVectors(nVectors)
    , d_nBatch(nVectors / batchSize)
    , d_nDofsPerCell(nDofsPerDim * nDofsPerDim * nDofsPerDim)
    , d_nQuadsPerCell(nQuadPointsPerDim * nQuadPointsPerDim * nQuadPointsPerDim)
  {
    AssertThrow(memorySpace == dftfe::utils::MemorySpace::DEVICE,
                dealii::ExcMessage(
                  "Matrix-Free framework is implemented only on GPUs\n"));

    AssertThrow(
      batchSize % subBatchSize == 0,
      dealii::ExcMessage(
        "Set batchSize as multiple of subBatchSize for real and same as subBatchSize for complex\n"));

    AssertThrow(nVectors % batchSize == 0,
                dealii::ExcMessage("Set nVectors as multiple of batchSize\n"));

    AssertThrow(
      operatorID < 3,
      dealii::ExcMessage(
        "Only Laplace and Helmholtz operators are implemented in Matrix-Free framework\n"));
  }


  template <typename T,
            dftfe::utils::MemorySpace memorySpace,
            unsigned int              nDofsPerDim,
            unsigned int              nQuadPointsPerDim,
            unsigned int              batchSize,
            unsigned int              subBatchSize>
  void
  MatrixFree<T,
             memorySpace,
             nDofsPerDim,
             nQuadPointsPerDim,
             batchSize,
             subBatchSize>::init()
  {
    d_basisOperationsPtrHost->reinit(0, 0, d_quadratureID);
    d_matrixFreeDataPtr = &(d_basisOperationsPtrHost->matrixFreeData());

    auto dofInfo = d_matrixFreeDataPtr->get_dof_info(
      d_basisOperationsPtrHost->d_dofHandlerID);

    auto shapeData =
      d_matrixFreeDataPtr
        ->get_shape_info(d_basisOperationsPtrHost->d_dofHandlerID,
                         d_quadratureID)
        .get_shape_data();
    auto mappingData =
      d_matrixFreeDataPtr->get_mapping_info().cell_data[d_quadratureID];

    d_constraintMatrixPtr =
      (*(d_basisOperationsPtrHost
           ->d_constraintsVector))[d_basisOperationsPtrHost->d_dofHandlerID];

    // Initialize shape and gradient functions
    std::array<double, nDofsPerDim * nQuadPointsPerDim>
      nodalShapeFunctionValuesAtQuadPoints;
    std::array<double, nQuadPointsPerDim * nQuadPointsPerDim>
      quadShapeFunctionGradientsAtQuadPoints;

    for (dftfe::Int iQuad = 0; iQuad < nQuadPointsPerDim; iQuad++)
      quadratureWeights[iQuad] = shapeData.quadrature.weight(iQuad);

    for (dftfe::Int iDoF = 0; iDoF < nDofsPerDim; iDoF++)
      for (dftfe::Int iQuad = 0; iQuad < nQuadPointsPerDim; iQuad++)
        nodalShapeFunctionValuesAtQuadPoints[iQuad + iDoF * nQuadPointsPerDim] =

#if (DEAL_II_VERSION_MAJOR >= 9 && DEAL_II_VERSION_MINOR >= 6)
          shapeData.shape_values[iQuad + iDoF * nQuadPointsPerDim] *
          (d_operatorID < 4 ? std::sqrt(shapeData.quadrature.weight(iQuad)) :
                              1);
#else
          shapeData.shape_values[iQuad + iDoF * nQuadPointsPerDim][0] *
          (d_operatorID < 4 ? std::sqrt(shapeData.quadrature.weight(iQuad)) :
                              1);
#endif

    for (dftfe::Int iQuad2 = 0; iQuad2 < nQuadPointsPerDim; iQuad2++)
      for (dftfe::Int iQuad1 = 0; iQuad1 < nQuadPointsPerDim; iQuad1++)
        quadShapeFunctionGradientsAtQuadPoints[iQuad1 +
                                               iQuad2 * nQuadPointsPerDim] =
#if (DEAL_II_VERSION_MAJOR >= 9 && DEAL_II_VERSION_MINOR >= 6)
          shapeData
            .shape_gradients_collocation[iQuad1 + iQuad2 * nQuadPointsPerDim] *
          (d_operatorID < 4 ? std::sqrt(shapeData.quadrature.weight(iQuad1)) /
                                std::sqrt(shapeData.quadrature.weight(iQuad2)) :
                              1);
#else
          shapeData.shape_gradients_collocation[iQuad1 +
                                                iQuad2 * nQuadPointsPerDim][0] *
          (d_operatorID < 4 ? std::sqrt(shapeData.quadrature.weight(iQuad1)) /
                                std::sqrt(shapeData.quadrature.weight(iQuad2)) :
                              1);
#endif

    for (dftfe::Int iDoF = 0; iDoF < d_dofEDim; iDoF++)
      for (dftfe::Int iQuad = 0; iQuad < d_quadEDim; iQuad++)
        nodalShapeFunctionValuesAtQuadPointsEO[iQuad + iDoF * d_quadEDim] =
          (nodalShapeFunctionValuesAtQuadPoints[iQuad +
                                                iDoF * nQuadPointsPerDim] +
           nodalShapeFunctionValuesAtQuadPoints[iQuad +
                                                (nDofsPerDim - 1 - iDoF) *
                                                  nQuadPointsPerDim]) *
          0.5;

    for (dftfe::Int iDoF = 0; iDoF < d_dofODim; iDoF++)
      for (dftfe::Int iQuad = 0; iQuad < d_quadODim; iQuad++)
        nodalShapeFunctionValuesAtQuadPointsEO[iQuad + iDoF * d_quadODim +
                                               d_quadEDim * d_dofEDim] =
          (nodalShapeFunctionValuesAtQuadPoints[iQuad +
                                                iDoF * nQuadPointsPerDim] -
           nodalShapeFunctionValuesAtQuadPoints[iQuad +
                                                (nDofsPerDim - 1 - iDoF) *
                                                  nQuadPointsPerDim]) *
          0.5;

    for (dftfe::Int iQuad2 = 0; iQuad2 < d_quadEDim; iQuad2++)
      for (dftfe::Int iQuad1 = 0; iQuad1 < d_quadODim; iQuad1++)
        quadShapeFunctionGradientsAtQuadPointsEO[iQuad1 + iQuad2 * d_quadODim] =
          (quadShapeFunctionGradientsAtQuadPoints[iQuad1 +
                                                  iQuad2 * nQuadPointsPerDim] +
           quadShapeFunctionGradientsAtQuadPoints
             [iQuad1 + (nQuadPointsPerDim - 1 - iQuad2) * nQuadPointsPerDim]) *
          0.5;

    for (dftfe::Int iQuad2 = 0; iQuad2 < d_quadODim; iQuad2++)
      for (dftfe::Int iQuad1 = 0; iQuad1 < d_quadEDim; iQuad1++)
        quadShapeFunctionGradientsAtQuadPointsEO[iQuad1 + iQuad2 * d_quadEDim +
                                                 d_quadEDim * d_quadODim] =
          (quadShapeFunctionGradientsAtQuadPoints[iQuad1 +
                                                  iQuad2 * nQuadPointsPerDim] -
           quadShapeFunctionGradientsAtQuadPoints
             [iQuad1 + (nQuadPointsPerDim - 1 - iQuad2) * nQuadPointsPerDim]) *
          0.5;

    // Construct cellIndexToMacroCellSubCellIndexMap
    auto d_nMacroCells = d_matrixFreeDataPtr->n_cell_batches();
    auto cellPtr       = d_matrixFreeDataPtr
                     ->get_dof_handler(d_basisOperationsPtrHost->d_dofHandlerID)
                     .begin_active();
    auto endcPtr = d_matrixFreeDataPtr
                     ->get_dof_handler(d_basisOperationsPtrHost->d_dofHandlerID)
                     .end();

    std::map<dealii::CellId, dftfe::uInt> cellIdToCellIndexMap;
    std::vector<dftfe::Int> cellIndexToMacroCellSubCellIndexMap(d_nCells);

    dftfe::Int iCell = 0;
    for (; cellPtr != endcPtr; cellPtr++)
      if (cellPtr->is_locally_owned())
        {
          cellIdToCellIndexMap[cellPtr->id()] = iCell;
          iCell++;
        }

    iCell = 0;
    for (dftfe::Int iMacroCell = 0; iMacroCell < d_nMacroCells; iMacroCell++)
      {
        const dftfe::Int numberSubCells =
          d_matrixFreeDataPtr->n_active_entries_per_cell_batch(iMacroCell);

        for (dftfe::Int iSubCell = 0; iSubCell < numberSubCells; iSubCell++)
          {
            cellPtr = d_matrixFreeDataPtr->get_cell_iterator(
              iMacroCell, iSubCell, d_basisOperationsPtrHost->d_dofHandlerID);

            dftfe::uInt cellIndex = cellIdToCellIndexMap[cellPtr->id()];
            cellIndexToMacroCellSubCellIndexMap[cellIndex] = iCell;

            iCell++;
          }
      }

    double coeff;

    switch (d_operatorID)
      {
        case Laplace:
          coeff = 1.0 / (4.0 * M_PI);
          break;
        case Helmholtz:
          coeff = 1.0;
          break;
        default:
          coeff = 1.0;
          break;
      }

    // Initialize Jacobian matrix
    constexpr dftfe::Int dim = 3;
    dftfe::utils::MemoryStorage<T, dftfe::utils::MemorySpace::HOST>
      jacobianFactor(dim * dim * d_nCells);

    d_jacobianFactor.resize(jacobianFactor.size());

    auto cellOffsets = mappingData.data_index_offsets;

    for (auto iCellBatch = 0, cellCount = 0;
         iCellBatch < dofInfo.n_vectorization_lanes_filled[2].size();
         iCellBatch++)
      for (auto iCell = 0;
           iCell < dofInfo.n_vectorization_lanes_filled[2][iCellBatch];
           iCell++, cellCount++)
        for (auto d = 0; d < dim; d++)
          for (auto e = 0; e < dim; e++)
            for (auto f = 0; f < dim; f++)
              jacobianFactor[e + d * dim + cellCount * dim * dim] +=
                coeff *
                mappingData.jacobians[0][cellOffsets[iCellBatch]][d][f][iCell] *
                mappingData.jacobians[0][cellOffsets[iCellBatch]][e][f][iCell] *
                mappingData.JxW_values[cellOffsets[iCellBatch]][iCell];

    for (dftfe::uInt iCell = 0; iCell < d_nCells; iCell++)
      for (dftfe::uInt iDim = 0; iDim < dim * dim; iDim++)
        d_jacobianFactor[iDim + iCell * dim * dim] =
          jacobianFactor[iDim + cellIndexToMacroCellSubCellIndexMap[iCell] *
                                  dim * dim];

    // Create matrix-free maps
    dftfe::utils::MemoryStorage<dftfe::Int, dftfe::utils::MemorySpace::HOST>
      singleVectorGlobalToLocalMap(d_nDofsPerCell * d_nCells);
    d_singleVectorGlobalToLocalMap.resize(singleVectorGlobalToLocalMap.size());

    // Construct singleVectorGlobalToLocalMap with matrix-free cell ordering
    for (auto iCell = 0; iCell < d_nCells; ++iCell)
      {
        auto checkExpr = dofInfo.row_starts[iCell].second ==
                           dofInfo.row_starts[iCell + 1].second &&
                         dofInfo.row_starts_plain_indices[iCell] ==
                           dealii::numbers::invalid_unsigned_int;

        auto trueClause =
          dofInfo.dof_indices.data() + dofInfo.row_starts[iCell].first;

        auto falseClause = dofInfo.plain_dof_indices.data() +
                           dofInfo.row_starts_plain_indices[iCell];

        std::transform(checkExpr ? trueClause : falseClause,
                       checkExpr ? trueClause + d_nDofsPerCell :
                                   falseClause + d_nDofsPerCell,
                       singleVectorGlobalToLocalMap.data() +
                         iCell * d_nDofsPerCell,
                       [](unsigned int &v) { return v; });
      }

    // Reorder cell numbering to cell-matrix order
    for (auto iCell = 0; iCell < d_nCells; iCell++)
      for (auto iDof = 0; iDof < d_nDofsPerCell; iDof++)
        d_singleVectorGlobalToLocalMap[iDof + iCell * d_nDofsPerCell] =
          singleVectorGlobalToLocalMap
            [iDof +
             cellIndexToMacroCellSubCellIndexMap[iCell] * d_nDofsPerCell];

    // Initialize constraints
    initConstraints();

    // Initialize member variables
    d_nOwnedDofs    = d_basisOperationsPtrHost->nOwnedDofs();
    d_nRelaventDofs = d_basisOperationsPtrHost->nRelaventDofs();
    d_nCells        = d_basisOperationsPtrHost->nCells();
    d_nGhostDofs    = d_nRelaventDofs - d_nOwnedDofs;

    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      {
        if (d_useDevice)
          {
            d_MatrixFreeDevice = std::make_unique<
              dftfe::MatrixFreeDevice<T,
                                      nDofsPerDim,
                                      nQuadPointsPerDim,
                                      std::is_same_v<T, double> ? 1 : 1>>(
              d_nVectors, d_nCells, d_nOwnedDofs, d_nGhostDofs);

            dftfe::utils::MemoryStorage<T, dftfe::utils::MemorySpace::HOST>
              shapeFunctionValueGradient(
                2 * d_quadEDim * d_dofEDim + 2 * d_quadODim * d_dofODim +
                4 * d_quadEDim * d_quadODim + nQuadPointsPerDim * nDofsPerDim +
                nQuadPointsPerDim);

            for (unsigned int iDoF = 0; iDoF < d_dofEDim; iDoF++)
              for (unsigned int iQuad = 0; iQuad < d_quadEDim; iQuad++)
                {
                  shapeFunctionValueGradient[iQuad + iDoF * d_quadEDim] =
                    nodalShapeFunctionValuesAtQuadPointsEO[iQuad +
                                                           iDoF * d_quadEDim];

                  shapeFunctionValueGradient[iDoF + iQuad * d_dofEDim +
                                             d_quadEDim * d_dofEDim +
                                             d_quadODim * d_dofODim +
                                             2 * d_quadEDim * d_quadODim] =
                    nodalShapeFunctionValuesAtQuadPointsEO[iQuad +
                                                           iDoF * d_quadEDim];
                }

            for (unsigned int iDoF = 0; iDoF < d_dofODim; iDoF++)
              for (unsigned int iQuad = 0; iQuad < d_quadODim; iQuad++)
                {
                  shapeFunctionValueGradient[iQuad + iDoF * d_quadODim +
                                             d_quadEDim * d_dofEDim] =
                    nodalShapeFunctionValuesAtQuadPointsEO
                      [iQuad + iDoF * d_quadODim + d_quadEDim * d_dofEDim];
                  shapeFunctionValueGradient[iDoF + iQuad * d_dofODim +
                                             2 * d_quadEDim * d_dofEDim +
                                             d_quadODim * d_dofODim +
                                             2 * d_quadEDim * d_quadODim] =
                    nodalShapeFunctionValuesAtQuadPointsEO
                      [iQuad + iDoF * d_quadODim + d_quadEDim * d_dofEDim];
                }

            for (unsigned int iQuad1 = 0; iQuad1 < d_quadEDim; iQuad1++)
              for (unsigned int iQuad2 = 0; iQuad2 < d_quadODim; iQuad2++)
                {
                  shapeFunctionValueGradient[iQuad2 + iQuad1 * d_quadODim +
                                             d_quadEDim * d_dofEDim +
                                             d_quadODim * d_dofODim] =
                    quadShapeFunctionGradientsAtQuadPointsEO[iQuad1 +
                                                             iQuad2 *
                                                               d_quadODim];

                  shapeFunctionValueGradient[iQuad1 + iQuad2 * d_quadEDim +
                                             2 * d_quadEDim * d_dofEDim +
                                             2 * d_quadODim * d_dofODim +
                                             2 * d_quadEDim * d_quadODim] =
                    quadShapeFunctionGradientsAtQuadPointsEO[iQuad1 +
                                                             iQuad2 *
                                                               d_quadODim];
                }

            for (unsigned int iQuad1 = 0; iQuad1 < d_quadODim; iQuad1++)
              for (unsigned int iQuad2 = 0; iQuad2 < d_quadEDim; iQuad2++)
                {
                  shapeFunctionValueGradient[iQuad2 + iQuad1 * d_quadEDim +
                                             d_quadEDim * d_dofEDim +
                                             d_quadODim * d_dofODim +
                                             d_quadEDim * d_quadODim] =
                    quadShapeFunctionGradientsAtQuadPointsEO
                      [iQuad1 + iQuad2 * d_quadEDim + d_quadEDim * d_quadODim];

                  shapeFunctionValueGradient[iQuad1 + iQuad2 * d_quadODim +
                                             2 * d_quadEDim * d_dofEDim +
                                             2 * d_quadODim * d_dofODim +
                                             3 * d_quadEDim * d_quadODim] =
                    quadShapeFunctionGradientsAtQuadPointsEO
                      [iQuad1 + iQuad2 * d_quadEDim + d_quadEDim * d_quadODim];
                }

            for (unsigned int iDoF = 0; iDoF < nDofsPerDim; iDoF++)
              for (unsigned int iQuad = 0; iQuad < nQuadPointsPerDim; iQuad++)
                shapeFunctionValueGradient[iQuad + iDoF * nQuadPointsPerDim +
                                           2 * d_quadEDim * d_dofEDim +
                                           2 * d_quadODim * d_dofODim +
                                           4 * d_quadEDim * d_quadODim] =
                  nodalShapeFunctionValuesAtQuadPoints[iQuad +
                                                       iDoF *
                                                         nQuadPointsPerDim];

            for (unsigned int iQuad = 0; iQuad < nQuadPointsPerDim; iQuad++)
              shapeFunctionValueGradient[iQuad + 2 * d_quadEDim * d_dofEDim +
                                         2 * d_quadODim * d_dofODim +
                                         4 * d_quadEDim * d_quadODim +
                                         nQuadPointsPerDim * nDofsPerDim] =
                quadratureWeights[iQuad];

            d_MatrixFreeDevice->init(shapeFunctionValueGradient.data(),
                                     shapeFunctionValueGradient.size(),
                                     d_jacobianFactor,
                                     d_singleVectorGlobalToLocalMap,
                                     d_constrainingNodeBuckets,
                                     d_constrainedNodeBuckets,
                                     d_weightMatrixList);
          }
      }
  }


  template <typename T,
            dftfe::utils::MemorySpace memorySpace,
            unsigned int              nDofsPerDim,
            unsigned int              nQuadPointsPerDim,
            unsigned int              batchSize,
            unsigned int              subBatchSize>
  void
  MatrixFree<T,
             memorySpace,
             nDofsPerDim,
             nQuadPointsPerDim,
             batchSize,
             subBatchSize>::initConstraints()
  {
    // Initialize constraint data structures
    const dealii::IndexSet &locallyOwnedDofs =
      d_matrixFreeDataPtr
        ->get_vector_partitioner(d_basisOperationsPtrHost->d_dofHandlerID)
        ->locally_owned_range();

    setupConstraints(locallyOwnedDofs);

    const dealii::IndexSet &ghostDofs =
      d_matrixFreeDataPtr
        ->get_vector_partitioner(d_basisOperationsPtrHost->d_dofHandlerID)
        ->ghost_indices();

    setupConstraints(ghostDofs);
  }


  template <typename T,
            dftfe::utils::MemorySpace memorySpace,
            unsigned int              nDofsPerDim,
            unsigned int              nQuadPointsPerDim,
            unsigned int              batchSize,
            unsigned int              subBatchSize>
  void
  MatrixFree<T,
             memorySpace,
             nDofsPerDim,
             nQuadPointsPerDim,
             batchSize,
             subBatchSize>::setupConstraints(const dealii::IndexSet &indexSet)
  {
    for (dealii::IndexSet::ElementIterator iter = indexSet.begin();
         iter != indexSet.end();
         iter++)
      if (d_constraintMatrixPtr->is_constrained(*iter))
        {
          bool isConstraintRhsExpandingOutOfIndexSet    = false;
          const dealii::types::global_dof_index lineDof = *iter;
          const std::vector<std::pair<dealii::types::global_dof_index, double>>
            *rowData = d_constraintMatrixPtr->get_constraint_entries(lineDof);

          for (unsigned int j = 0; j < rowData->size(); j++)
            {
              if (!(d_matrixFreeDataPtr
                      ->get_vector_partitioner(
                        d_basisOperationsPtrHost->d_dofHandlerID)
                      ->is_ghost_entry((*rowData)[j].first) ||
                    d_matrixFreeDataPtr
                      ->get_vector_partitioner(
                        d_basisOperationsPtrHost->d_dofHandlerID)
                      ->in_local_range((*rowData)[j].first)))
                {
                  isConstraintRhsExpandingOutOfIndexSet = true;
                  break;
                }
            }

          if (isConstraintRhsExpandingOutOfIndexSet)
            continue;

          std::vector<unsigned int> constrainingData(rowData->size());
          std::vector<T>            weightData(rowData->size());

          for (auto i = 0; i < rowData->size(); i++)
            {
              constrainingData[i] =
                d_matrixFreeDataPtr
                  ->get_vector_partitioner(
                    d_basisOperationsPtrHost->d_dofHandlerID)
                  ->global_to_local((*rowData)[i].first);

              weightData[i] = (*rowData)[i].second;
            }

          bool         constraintExists = false;
          unsigned int constraintIndex  = 0;
          T inhomogenity = d_constraintMatrixPtr->get_inhomogeneity(lineDof);

          for (auto i = 0; i < d_constrainingNodeBuckets.size(); i++)
            if ((d_constrainingNodeBuckets[i] == constrainingData) &&
                (d_inhomogenityList[i] == inhomogenity))
              {
                constraintIndex  = i;
                constraintExists = true;
                break;
              }

          if (constraintExists)
            {
              d_constrainedNodeBuckets[constraintIndex].push_back(
                d_matrixFreeDataPtr
                  ->get_vector_partitioner(
                    d_basisOperationsPtrHost->d_dofHandlerID)
                  ->global_to_local(lineDof));

              d_weightMatrixList[constraintIndex].insert(
                d_weightMatrixList[constraintIndex].end(),
                weightData.begin(),
                weightData.end());
            }
          else
            {
              d_constrainedNodeBuckets.push_back(std::vector<unsigned int>(
                1,
                d_matrixFreeDataPtr
                  ->get_vector_partitioner(
                    d_basisOperationsPtrHost->d_dofHandlerID)
                  ->global_to_local(lineDof)));

              d_weightMatrixList.push_back(weightData);
              d_constrainingNodeBuckets.push_back(constrainingData);
              d_inhomogenityList.push_back(inhomogenity);
            }
        }
  }


  template <typename T,
            dftfe::utils::MemorySpace memorySpace,
            unsigned int              nDofsPerDim,
            unsigned int              nQuadPointsPerDim,
            unsigned int              batchSize,
            unsigned int              subBatchSize>
  inline void
  MatrixFree<T,
             memorySpace,
             nDofsPerDim,
             nQuadPointsPerDim,
             batchSize,
             subBatchSize>::constraintsDistribute(T *src)
  {
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      {
        if (d_useDevice)
          d_MatrixFreeDevice->constraintsDistribute(src);
      }
  }


  template <typename T,
            dftfe::utils::MemorySpace memorySpace,
            unsigned int              nDofsPerDim,
            unsigned int              nQuadPointsPerDim,
            unsigned int              batchSize,
            unsigned int              subBatchSize>
  inline void
  MatrixFree<T,
             memorySpace,
             nDofsPerDim,
             nQuadPointsPerDim,
             batchSize,
             subBatchSize>::constraintsDistributeTranspose(T *dst, T *src)
  {
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      {
        if (d_useDevice)
          d_MatrixFreeDevice->constraintsDistributeTranspose(dst, src);
      }
  }


  template <typename T,
            dftfe::utils::MemorySpace memorySpace,
            unsigned int              nDofsPerDim,
            unsigned int              nQuadPointsPerDim,
            unsigned int              batchSize,
            unsigned int              subBatchSize>
  inline void
  MatrixFree<T,
             memorySpace,
             nDofsPerDim,
             nQuadPointsPerDim,
             batchSize,
             subBatchSize>::constraintsSetZero(T *src)
  {
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      {
        if (d_useDevice)
          d_MatrixFreeDevice->constraintsSetZero(src);
      }
  }


  template <typename T,
            dftfe::utils::MemorySpace memorySpace,
            unsigned int              nDofsPerDim,
            unsigned int              nQuadPointsPerDim,
            unsigned int              batchSize,
            unsigned int              subBatchSize>
  inline void
  MatrixFree<T,
             memorySpace,
             nDofsPerDim,
             nQuadPointsPerDim,
             batchSize,
             subBatchSize>::computeAX(T *dst, T *src)

  {
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      {
        if (d_useDevice)
          {
            switch (d_operatorID)
              {
                case Laplace:
                  d_MatrixFreeDevice->computeLaplaceX(dst, src);
                  break;

                case Helmholtz:
                  d_MatrixFreeDevice->computeLaplaceX(dst, src);
                  break;
              }
          }
      }
  }

#include "MatrixFree.inst.cc"
} // namespace dftfe
