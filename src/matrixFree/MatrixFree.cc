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

/**
 * @author Gourab Panigrahi
 *
 */

#include <MatrixFree.h>

namespace dftfe
{
  template <typename T,
            dftfe::operatorList       operatorID,
            dftfe::utils::MemorySpace memorySpace,
            bool                      isComplex,
            std::uint32_t             nDofsPerDim,
            std::uint32_t             nQuadPointsPerDim,
            std::uint32_t             batchSize,
            std::uint32_t             subBatchSize>
  MatrixFree<T,
             operatorID,
             memorySpace,
             isComplex,
             nDofsPerDim,
             nQuadPointsPerDim,
             batchSize,
             subBatchSize>::
    MatrixFree(const MPI_Comm                          &mpi_comm,
               const dealii::MatrixFree<3, double>     *matrixFreeDataPtr,
               const dealii::AffineConstraints<double> &constraintMatrix,
               std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
                                   BLASWrapperPtr,
               const std::uint32_t dofHandlerID,
               const std::uint32_t quadratureID,
               const dftfe::uInt   nVectors)
    : mpi_communicator(mpi_comm)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm))
    , this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_comm))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0))
    , d_matrixFreeDataPtr(matrixFreeDataPtr)
    , d_constraintMatrixPtr(&constraintMatrix)
    , d_BLASWrapperPtr(BLASWrapperPtr)
    , d_dofHandlerID(dofHandlerID)
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
      operatorID < 2,
      dealii::ExcMessage(
        "Only Laplace and Helmholtz operators are implemented in Matrix-Free framework\n"));
  }


  template <typename T,
            dftfe::operatorList       operatorID,
            dftfe::utils::MemorySpace memorySpace,
            bool                      isComplex,
            std::uint32_t             nDofsPerDim,
            std::uint32_t             nQuadPointsPerDim,
            std::uint32_t             batchSize,
            std::uint32_t             subBatchSize>
  void
  MatrixFree<T,
             operatorID,
             memorySpace,
             isComplex,
             nDofsPerDim,
             nQuadPointsPerDim,
             batchSize,
             subBatchSize>::init()
  {
    d_nCells     = d_matrixFreeDataPtr->n_physical_cells();
    d_nOwnedDofs = d_matrixFreeDataPtr->get_vector_partitioner(d_dofHandlerID)
                     ->locally_owned_size();
    d_nGhostDofs = d_matrixFreeDataPtr->get_vector_partitioner(d_dofHandlerID)
                     ->n_ghost_indices();
    d_nRelaventDofs = d_nOwnedDofs + d_nGhostDofs;

    auto dofInfo = d_matrixFreeDataPtr->get_dof_info(d_dofHandlerID);
    auto shapeData =
      d_matrixFreeDataPtr->get_shape_info(d_dofHandlerID, d_quadratureID)
        .get_shape_data();
    auto mappingData =
      d_matrixFreeDataPtr->get_mapping_info().cell_data[d_quadratureID];

    // Initialize shape and gradient functions
    std::array<double, nDofsPerDim * nQuadPointsPerDim>
      nodalShapeFunctionValuesAtQuadPoints;
    std::array<double, nQuadPointsPerDim * nQuadPointsPerDim>
      quadShapeFunctionGradientsAtQuadPoints;

    for (std::uint32_t iQuad = 0; iQuad < nQuadPointsPerDim; iQuad++)
      quadratureWeights[iQuad] = shapeData.quadrature.weight(iQuad);

    for (std::uint32_t iDoF = 0; iDoF < nDofsPerDim; iDoF++)
      for (std::uint32_t iQuad = 0; iQuad < nQuadPointsPerDim; iQuad++)
        nodalShapeFunctionValuesAtQuadPoints[iQuad + iDoF * nQuadPointsPerDim] =

#if (DEAL_II_VERSION_MAJOR >= 9 && DEAL_II_VERSION_MINOR >= 6)
          shapeData.shape_values[iQuad + iDoF * nQuadPointsPerDim] *
          (operatorID < 3 ? std::sqrt(shapeData.quadrature.weight(iQuad)) : 1);
#else
          shapeData.shape_values[iQuad + iDoF * nQuadPointsPerDim][0] *
          (operatorID < 3 ? std::sqrt(shapeData.quadrature.weight(iQuad)) : 1);
#endif

    for (std::uint32_t iQuad2 = 0; iQuad2 < nQuadPointsPerDim; iQuad2++)
      for (std::uint32_t iQuad1 = 0; iQuad1 < nQuadPointsPerDim; iQuad1++)
        quadShapeFunctionGradientsAtQuadPoints[iQuad1 +
                                               iQuad2 * nQuadPointsPerDim] =
#if (DEAL_II_VERSION_MAJOR >= 9 && DEAL_II_VERSION_MINOR >= 6)
          shapeData
            .shape_gradients_collocation[iQuad1 + iQuad2 * nQuadPointsPerDim] *
          (operatorID < 3 ? std::sqrt(shapeData.quadrature.weight(iQuad1)) /
                              std::sqrt(shapeData.quadrature.weight(iQuad2)) :
                            1);
#else
          shapeData.shape_gradients_collocation[iQuad1 +
                                                iQuad2 * nQuadPointsPerDim][0] *
          (operatorID < 3 ? std::sqrt(shapeData.quadrature.weight(iQuad1)) /
                              std::sqrt(shapeData.quadrature.weight(iQuad2)) :
                            1);
#endif

    for (std::uint32_t iDoF = 0; iDoF < d_dofEDim; iDoF++)
      for (std::uint32_t iQuad = 0; iQuad < d_quadEDim; iQuad++)
        nodalShapeFunctionValuesAtQuadPointsEO[iQuad + iDoF * d_quadEDim] =
          (nodalShapeFunctionValuesAtQuadPoints[iQuad +
                                                iDoF * nQuadPointsPerDim] +
           nodalShapeFunctionValuesAtQuadPoints[iQuad +
                                                (nDofsPerDim - 1 - iDoF) *
                                                  nQuadPointsPerDim]) *
          0.5;

    for (std::uint32_t iDoF = 0; iDoF < d_dofODim; iDoF++)
      for (std::uint32_t iQuad = 0; iQuad < d_quadODim; iQuad++)
        nodalShapeFunctionValuesAtQuadPointsEO[iQuad + iDoF * d_quadODim +
                                               d_quadEDim * d_dofEDim] =
          (nodalShapeFunctionValuesAtQuadPoints[iQuad +
                                                iDoF * nQuadPointsPerDim] -
           nodalShapeFunctionValuesAtQuadPoints[iQuad +
                                                (nDofsPerDim - 1 - iDoF) *
                                                  nQuadPointsPerDim]) *
          0.5;

    for (std::uint32_t iQuad2 = 0; iQuad2 < d_quadEDim; iQuad2++)
      for (std::uint32_t iQuad1 = 0; iQuad1 < d_quadODim; iQuad1++)
        quadShapeFunctionGradientsAtQuadPointsEO[iQuad1 + iQuad2 * d_quadODim] =
          (quadShapeFunctionGradientsAtQuadPoints[iQuad1 +
                                                  iQuad2 * nQuadPointsPerDim] +
           quadShapeFunctionGradientsAtQuadPoints
             [iQuad1 + (nQuadPointsPerDim - 1 - iQuad2) * nQuadPointsPerDim]) *
          0.5;

    for (std::uint32_t iQuad2 = 0; iQuad2 < d_quadODim; iQuad2++)
      for (std::uint32_t iQuad1 = 0; iQuad1 < d_quadEDim; iQuad1++)
        quadShapeFunctionGradientsAtQuadPointsEO[iQuad1 + iQuad2 * d_quadEDim +
                                                 d_quadEDim * d_quadODim] =
          (quadShapeFunctionGradientsAtQuadPoints[iQuad1 +
                                                  iQuad2 * nQuadPointsPerDim] -
           quadShapeFunctionGradientsAtQuadPoints
             [iQuad1 + (nQuadPointsPerDim - 1 - iQuad2) * nQuadPointsPerDim]) *
          0.5;

    // Construct cellIndexToMacroCellSubCellIndexMap
    auto d_nMacroCells = d_matrixFreeDataPtr->n_cell_batches();
    auto cellPtr =
      d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).begin_active();
    auto endcPtr = d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).end();

    std::map<dealii::CellId, dftfe::uInt> cellIdToCellIndexMap;
    std::vector<dftfe::uInt> cellIndexToMacroCellSubCellIndexMap(d_nCells);

    dftfe::uInt iCell = 0;
    for (; cellPtr != endcPtr; cellPtr++)
      if (cellPtr->is_locally_owned())
        {
          cellIdToCellIndexMap[cellPtr->id()] = iCell;
          iCell++;
        }

    iCell = 0;
    for (dftfe::uInt iMacroCell = 0; iMacroCell < d_nMacroCells; iMacroCell++)
      {
        const dftfe::uInt numberSubCells =
          d_matrixFreeDataPtr->n_active_entries_per_cell_batch(iMacroCell);

        for (dftfe::uInt iSubCell = 0; iSubCell < numberSubCells; iSubCell++)
          {
            cellPtr = d_matrixFreeDataPtr->get_cell_iterator(iMacroCell,
                                                             iSubCell,
                                                             d_dofHandlerID);

            dftfe::uInt cellIndex = cellIdToCellIndexMap[cellPtr->id()];
            cellIndexToMacroCellSubCellIndexMap[cellIndex] = iCell;

            iCell++;
          }
      }

    double coeff = 1.0;

    if constexpr (operatorID == operatorList::Laplace)
      coeff = 1.0 / (4.0 * M_PI);

    if constexpr (operatorID == operatorList::Helmholtz)
      coeff = 1.0;

    // Initialize Jacobian matrix
    constexpr dftfe::uInt dim = 3;
    dftfe::utils::MemoryStorage<T, dftfe::utils::MemorySpace::HOST>
      jacobianFactorTemp(dim * dim * d_nCells),
      jacobianFactor(dim * dim * d_nCells);

    auto cellOffsets = mappingData.data_index_offsets;

    for (dftfe::uInt iCellBatch = 0, cellCount = 0;
         iCellBatch < dofInfo.n_vectorization_lanes_filled[2].size();
         iCellBatch++)
      for (dftfe::uInt iCell = 0;
           iCell < dofInfo.n_vectorization_lanes_filled[2][iCellBatch];
           iCell++, cellCount++)
        for (dftfe::uInt k = 0; k < dim; k++)
          for (dftfe::uInt j = 0; j < dim; j++)
            for (dftfe::uInt i = 0; i < dim; i++)
              jacobianFactorTemp[i + j * dim + cellCount * dim * dim] +=
                coeff * mappingData.JxW_values[cellOffsets[iCellBatch]][iCell] *
                mappingData.jacobians[0][cellOffsets[iCellBatch]][k][i][iCell] *
                mappingData.jacobians[0][cellOffsets[iCellBatch]][k][j][iCell];

    for (dftfe::uInt iCell = 0; iCell < d_nCells; iCell++)
      for (dftfe::uInt iDim = 0; iDim < dim * dim; iDim++)
        jacobianFactor[iDim + iCell * dim * dim] =
          jacobianFactorTemp[iDim + cellIndexToMacroCellSubCellIndexMap[iCell] *
                                      dim * dim];

    d_jacobianFactor.resize(jacobianFactor.size());
    d_jacobianFactor.copyFrom(jacobianFactor);

    // Create matrix-free maps
    dftfe::utils::MemoryStorage<dftfe::uInt, dftfe::utils::MemorySpace::HOST>
      singleVectorGlobalToLocalMapTemp(d_nDofsPerCell * d_nCells),
      singleVectorGlobalToLocalMap(d_nDofsPerCell * d_nCells);

    // Construct singleVectorGlobalToLocalMap with matrix-free cell ordering
    for (dftfe::uInt iCell = 0; iCell < d_nCells; iCell++)
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
                       singleVectorGlobalToLocalMapTemp.data() +
                         iCell * d_nDofsPerCell,
                       [](unsigned int &v) {
                         return static_cast<dftfe::uInt>(v);
                       });
      }

    // Reorder cell numbering to cell-matrix order
    for (dftfe::uInt iCell = 0; iCell < d_nCells; iCell++)
      for (dftfe::uInt iDof = 0; iDof < d_nDofsPerCell; iDof++)
        singleVectorGlobalToLocalMap[iDof + iCell * d_nDofsPerCell] =
          singleVectorGlobalToLocalMapTemp
            [iDof +
             cellIndexToMacroCellSubCellIndexMap[iCell] * d_nDofsPerCell];

    d_map.resize(singleVectorGlobalToLocalMap.size());
    d_map.copyFrom(singleVectorGlobalToLocalMap);

    // Initialize constraints
    initConstraints();

    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      {
        dftfe::utils::MemoryStorage<T, dftfe::utils::MemorySpace::HOST>
          shapeFunctionValueGradient(
            2 * d_quadEDim * d_dofEDim + 2 * d_quadODim * d_dofODim +
            4 * d_quadEDim * d_quadODim + nQuadPointsPerDim * nDofsPerDim +
            nQuadPointsPerDim);

        for (std::uint32_t iDoF = 0; iDoF < d_dofEDim; iDoF++)
          for (std::uint32_t iQuad = 0; iQuad < d_quadEDim; iQuad++)
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

        for (std::uint32_t iDoF = 0; iDoF < d_dofODim; iDoF++)
          for (std::uint32_t iQuad = 0; iQuad < d_quadODim; iQuad++)
            {
              shapeFunctionValueGradient[iQuad + iDoF * d_quadODim +
                                         d_quadEDim * d_dofEDim] =
                nodalShapeFunctionValuesAtQuadPointsEO[iQuad +
                                                       iDoF * d_quadODim +
                                                       d_quadEDim * d_dofEDim];
              shapeFunctionValueGradient[iDoF + iQuad * d_dofODim +
                                         2 * d_quadEDim * d_dofEDim +
                                         d_quadODim * d_dofODim +
                                         2 * d_quadEDim * d_quadODim] =
                nodalShapeFunctionValuesAtQuadPointsEO[iQuad +
                                                       iDoF * d_quadODim +
                                                       d_quadEDim * d_dofEDim];
            }

        for (std::uint32_t iQuad2 = 0; iQuad2 < d_quadEDim; iQuad2++)
          for (std::uint32_t iQuad1 = 0; iQuad1 < d_quadODim; iQuad1++)
            {
              shapeFunctionValueGradient[iQuad1 + iQuad2 * d_quadODim +
                                         d_quadEDim * d_dofEDim +
                                         d_quadODim * d_dofODim] =
                quadShapeFunctionGradientsAtQuadPointsEO[iQuad1 +
                                                         iQuad2 * d_quadODim];

              shapeFunctionValueGradient[iQuad2 + iQuad1 * d_quadEDim +
                                         2 * d_quadEDim * d_dofEDim +
                                         2 * d_quadODim * d_dofODim +
                                         2 * d_quadEDim * d_quadODim] =
                quadShapeFunctionGradientsAtQuadPointsEO[iQuad1 +
                                                         iQuad2 * d_quadODim];
            }

        for (std::uint32_t iQuad2 = 0; iQuad2 < d_quadODim; iQuad2++)
          for (std::uint32_t iQuad1 = 0; iQuad1 < d_quadEDim; iQuad1++)
            {
              shapeFunctionValueGradient[iQuad1 + iQuad2 * d_quadEDim +
                                         d_quadEDim * d_dofEDim +
                                         d_quadODim * d_dofODim +
                                         d_quadEDim * d_quadODim] =
                quadShapeFunctionGradientsAtQuadPointsEO
                  [iQuad1 + iQuad2 * d_quadEDim + d_quadEDim * d_quadODim];

              shapeFunctionValueGradient[iQuad2 + iQuad1 * d_quadODim +
                                         2 * d_quadEDim * d_dofEDim +
                                         2 * d_quadODim * d_dofODim +
                                         3 * d_quadEDim * d_quadODim] =
                quadShapeFunctionGradientsAtQuadPointsEO
                  [iQuad1 + iQuad2 * d_quadEDim + d_quadEDim * d_quadODim];
            }

        for (std::uint32_t iDoF = 0; iDoF < nDofsPerDim; iDoF++)
          for (std::uint32_t iQuad = 0; iQuad < nQuadPointsPerDim; iQuad++)
            shapeFunctionValueGradient[iQuad + iDoF * nQuadPointsPerDim +
                                       2 * d_quadEDim * d_dofEDim +
                                       2 * d_quadODim * d_dofODim +
                                       4 * d_quadEDim * d_quadODim] =
              nodalShapeFunctionValuesAtQuadPoints[iQuad +
                                                   iDoF * nQuadPointsPerDim];

        for (std::uint32_t iQuad = 0; iQuad < nQuadPointsPerDim; iQuad++)
          shapeFunctionValueGradient[iQuad + 2 * d_quadEDim * d_dofEDim +
                                     2 * d_quadODim * d_dofODim +
                                     4 * d_quadEDim * d_quadODim +
                                     nQuadPointsPerDim * nDofsPerDim] =
            quadratureWeights[iQuad];

        dftfe::utils::MemoryStorage<dftfe::uInt,
                                    dftfe::utils::MemorySpace::HOST>
          constrainingNodeOffset(d_constrainingNodeBuckets.size() + 1),
          constrainedNodeOffset(d_constrainedNodeBuckets.size() + 1),
          weightMatrixOffset(d_weightMatrixList.size() + 1);

        dftfe::uInt k = 0;

        for (dftfe::uInt i = 0; i < d_constrainingNodeBuckets.size(); i++)
          {
            constrainingNodeOffset[i] = k;
            k += d_constrainingNodeBuckets[i].size();
          }

        constrainingNodeOffset[d_constrainingNodeBuckets.size()] = k;
        d_constrainingNodeBucketsDevice.resize(k);

        for (dftfe::uInt i = 0; i < d_constrainingNodeBuckets.size(); i++)
          dftfe::utils::MemoryTransfer<dftfe::utils::MemorySpace::DEVICE,
                                       dftfe::utils::MemorySpace::HOST>::
            copy(d_constrainingNodeBuckets[i].size(),
                 d_constrainingNodeBucketsDevice.data() +
                   constrainingNodeOffset[i],
                 d_constrainingNodeBuckets[i].data());

        k = 0;

        for (dftfe::uInt i = 0; i < d_constrainedNodeBuckets.size(); i++)
          {
            constrainedNodeOffset[i] = k;
            k += d_constrainedNodeBuckets[i].size();
          }

        constrainedNodeOffset[d_constrainedNodeBuckets.size()] = k;
        d_constrainedNodeBucketsDevice.resize(k);

        for (dftfe::uInt i = 0; i < d_constrainedNodeBuckets.size(); i++)
          dftfe::utils::MemoryTransfer<dftfe::utils::MemorySpace::DEVICE,
                                       dftfe::utils::MemorySpace::HOST>::
            copy(d_constrainedNodeBuckets[i].size(),
                 d_constrainedNodeBucketsDevice.data() +
                   constrainedNodeOffset[i],
                 d_constrainedNodeBuckets[i].data());

        k = 0;

        for (dftfe::uInt i = 0; i < d_weightMatrixList.size(); i++)
          {
            weightMatrixOffset[i] = k;
            k += d_weightMatrixList[i].size();
          }

        weightMatrixOffset[d_weightMatrixList.size()] = k;
        d_weightMatrixListDevice.resize(k);

        for (dftfe::uInt i = 0; i < d_weightMatrixList.size(); i++)
          dftfe::utils::MemoryTransfer<dftfe::utils::MemorySpace::DEVICE,
                                       dftfe::utils::MemorySpace::HOST>::
            copy(d_weightMatrixList[i].size(),
                 d_weightMatrixListDevice.data() + weightMatrixOffset[i],
                 d_weightMatrixList[i].data());

        d_constrainingNodeOffsetDevice.resize(constrainingNodeOffset.size());
        d_constrainingNodeOffsetDevice.copyFrom(constrainingNodeOffset);

        d_constrainedNodeOffsetDevice.resize(constrainedNodeOffset.size());
        d_constrainedNodeOffsetDevice.copyFrom(constrainedNodeOffset);

        d_weightMatrixOffsetDevice.resize(weightMatrixOffset.size());
        d_weightMatrixOffsetDevice.copyFrom(weightMatrixOffset);

        d_inhomogenityListDevice.resize(d_inhomogenityList.size());
        d_inhomogenityListDevice.copyFrom(d_inhomogenityList);

#ifdef DFTFE_WITH_DEVICE_LANG_SYCL

        constexpr std::uint32_t d_maxDofsPerDim = 17;

        shapeBufferDevice.resize(
          (d_maxDofsPerDim * d_maxDofsPerDim * 5 + d_maxDofsPerDim));

        dftfe::utils::MemoryTransfer<dftfe::utils::MemorySpace::DEVICE,
                                     dftfe::utils::MemorySpace::HOST>::
          copy(shapeFunctionValueGradient.size(),
               shapeBufferDevice.data(),
               shapeFunctionValueGradient.data());
#else
        dftfe::MatrixFreeDevice<
          T,
          operatorID,
          nDofsPerDim,
          nQuadPointsPerDim,
          batchSize>::init(shapeFunctionValueGradient.data(),
                           shapeFunctionValueGradient.size());
#endif
      }
  }


  template <typename T,
            dftfe::operatorList       operatorID,
            dftfe::utils::MemorySpace memorySpace,
            bool                      isComplex,
            std::uint32_t             nDofsPerDim,
            std::uint32_t             nQuadPointsPerDim,
            std::uint32_t             batchSize,
            std::uint32_t             subBatchSize>
  void
  MatrixFree<T,
             operatorID,
             memorySpace,
             isComplex,
             nDofsPerDim,
             nQuadPointsPerDim,
             batchSize,
             subBatchSize>::initOperatorCoeffs(T coeffHelmholtz)
  {
    d_coeffHelmholtz = coeffHelmholtz;
  }


  template <typename T,
            dftfe::operatorList       operatorID,
            dftfe::utils::MemorySpace memorySpace,
            bool                      isComplex,
            std::uint32_t             nDofsPerDim,
            std::uint32_t             nQuadPointsPerDim,
            std::uint32_t             batchSize,
            std::uint32_t             subBatchSize>
  void
  MatrixFree<T,
             operatorID,
             memorySpace,
             isComplex,
             nDofsPerDim,
             nQuadPointsPerDim,
             batchSize,
             subBatchSize>::initConstraints()
  {
    // Initialize constraint data structures
    const dealii::IndexSet &locallyOwnedDofs =
      d_matrixFreeDataPtr->get_vector_partitioner(d_dofHandlerID)
        ->locally_owned_range();

    setupConstraints(locallyOwnedDofs);

    const dealii::IndexSet &ghostDofs =
      d_matrixFreeDataPtr->get_vector_partitioner(d_dofHandlerID)
        ->ghost_indices();

    setupConstraints(ghostDofs);
  }


  template <typename T,
            dftfe::operatorList       operatorID,
            dftfe::utils::MemorySpace memorySpace,
            bool                      isComplex,
            std::uint32_t             nDofsPerDim,
            std::uint32_t             nQuadPointsPerDim,
            std::uint32_t             batchSize,
            std::uint32_t             subBatchSize>
  void
  MatrixFree<T,
             operatorID,
             memorySpace,
             isComplex,
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

          for (dftfe::uInt j = 0; j < rowData->size(); j++)
            {
              if (!(d_matrixFreeDataPtr->get_vector_partitioner(d_dofHandlerID)
                      ->is_ghost_entry((*rowData)[j].first) ||
                    d_matrixFreeDataPtr->get_vector_partitioner(d_dofHandlerID)
                      ->in_local_range((*rowData)[j].first)))
                {
                  isConstraintRhsExpandingOutOfIndexSet = true;
                  break;
                }
            }

          if (isConstraintRhsExpandingOutOfIndexSet)
            continue;

          std::vector<dftfe::uInt> constrainingData(rowData->size());
          std::vector<T>           weightData(rowData->size());

          for (auto i = 0; i < rowData->size(); i++)
            {
              constrainingData[i] =
                d_matrixFreeDataPtr->get_vector_partitioner(d_dofHandlerID)
                  ->global_to_local((*rowData)[i].first);

              weightData[i] = (*rowData)[i].second;
            }

          bool        constraintExists = false;
          dftfe::uInt constraintIndex  = 0;
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
                d_matrixFreeDataPtr->get_vector_partitioner(d_dofHandlerID)
                  ->global_to_local(lineDof));

              d_weightMatrixList[constraintIndex].insert(
                d_weightMatrixList[constraintIndex].end(),
                weightData.begin(),
                weightData.end());
            }
          else
            {
              d_constrainedNodeBuckets.push_back(std::vector<dftfe::uInt>(
                1,
                d_matrixFreeDataPtr->get_vector_partitioner(d_dofHandlerID)
                  ->global_to_local(lineDof)));

              d_weightMatrixList.push_back(weightData);
              d_constrainingNodeBuckets.push_back(constrainingData);
              d_inhomogenityList.push_back(inhomogenity);
            }
        }
  }


  template <typename T,
            dftfe::operatorList       operatorID,
            dftfe::utils::MemorySpace memorySpace,
            bool                      isComplex,
            std::uint32_t             nDofsPerDim,
            std::uint32_t             nQuadPointsPerDim,
            std::uint32_t             batchSize,
            std::uint32_t             subBatchSize>
  inline void
  MatrixFree<T,
             operatorID,
             memorySpace,
             isComplex,
             nDofsPerDim,
             nQuadPointsPerDim,
             batchSize,
             subBatchSize>::constraintsDistribute(T *src)
  {
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      {
        if (d_constrainedNodeBucketsDevice.size() == 0)
          return;

        dftfe::MatrixFreeDevice<T,
                                operatorID,
                                nDofsPerDim,
                                nQuadPointsPerDim,
                                batchSize>::
          constraintsDistribute(src,
                                d_constrainingNodeBucketsDevice.data(),
                                d_constrainingNodeOffsetDevice.data(),
                                d_constrainedNodeBucketsDevice.data(),
                                d_constrainedNodeOffsetDevice.data(),
                                d_weightMatrixListDevice.data(),
                                d_weightMatrixOffsetDevice.data(),
                                d_inhomogenityListDevice.data(),
                                d_map.data(),
                                d_inhomogenityListDevice.size(),
                                d_nBatch,
                                d_nOwnedDofs,
                                d_nGhostDofs);
      }
  }


  template <typename T,
            dftfe::operatorList       operatorID,
            dftfe::utils::MemorySpace memorySpace,
            bool                      isComplex,
            std::uint32_t             nDofsPerDim,
            std::uint32_t             nQuadPointsPerDim,
            std::uint32_t             batchSize,
            std::uint32_t             subBatchSize>
  inline void
  MatrixFree<T,
             operatorID,
             memorySpace,
             isComplex,
             nDofsPerDim,
             nQuadPointsPerDim,
             batchSize,
             subBatchSize>::constraintsDistributeTranspose(T *dst, T *src)
  {
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      {
        if (d_constrainedNodeBucketsDevice.size() == 0)
          return;

        dftfe::MatrixFreeDevice<T,
                                operatorID,
                                nDofsPerDim,
                                nQuadPointsPerDim,
                                batchSize>::
          constraintsDistributeTranspose(dst,
                                         src,
                                         d_constrainingNodeBucketsDevice.data(),
                                         d_constrainingNodeOffsetDevice.data(),
                                         d_constrainedNodeBucketsDevice.data(),
                                         d_constrainedNodeOffsetDevice.data(),
                                         d_weightMatrixListDevice.data(),
                                         d_weightMatrixOffsetDevice.data(),
                                         d_map.data(),
                                         d_inhomogenityListDevice.size(),
                                         d_nBatch,
                                         d_nOwnedDofs,
                                         d_nGhostDofs);
      }
  }


  template <typename T,
            dftfe::operatorList       operatorID,
            dftfe::utils::MemorySpace memorySpace,
            bool                      isComplex,
            std::uint32_t             nDofsPerDim,
            std::uint32_t             nQuadPointsPerDim,
            std::uint32_t             batchSize,
            std::uint32_t             subBatchSize>
  inline void
  MatrixFree<T,
             operatorID,
             memorySpace,
             isComplex,
             nDofsPerDim,
             nQuadPointsPerDim,
             batchSize,
             subBatchSize>::computeAX(T *dst, T *src)
  {
    if constexpr (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      {
        if constexpr (operatorID == dftfe::operatorList::Laplace)
          dftfe::MatrixFreeDevice<
            T,
            operatorID,
            nDofsPerDim,
            nQuadPointsPerDim,
            batchSize>::computeLaplaceX(dst,
                                        src,
                                        d_jacobianFactor.data(),
                                        d_map.data(),
                                        shapeBufferDevice.data(),
                                        d_nCells,
                                        d_nBatch);

        if constexpr (operatorID == dftfe::operatorList::Helmholtz)
          dftfe::MatrixFreeDevice<
            T,
            operatorID,
            nDofsPerDim,
            nQuadPointsPerDim,
            batchSize>::computeHelmholtzX(dst,
                                          src,
                                          d_jacobianFactor.data(),
                                          d_map.data(),
                                          shapeBufferDevice.data(),
                                          d_coeffHelmholtz,
                                          d_nCells,
                                          d_nBatch);
      }
  }

#include "MatrixFree.inst.cc"
} // namespace dftfe
