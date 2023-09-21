// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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

#include <FEBasisOperations.h>
namespace dftfe
{
  namespace basis
  {
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::
      FEBasisOperationsBase(
        dealii::MatrixFree<3, ValueTypeBasisData> &matrixFreeData,
        std::vector<const dealii::AffineConstraints<ValueTypeBasisData> *>
          &constraintsVector)
    {
      d_matrixFreeDataPtr = &matrixFreeData;
      d_constraintsVector = &constraintsVector;
      d_dofHandlerID      = 0;
      d_quadratureID      = 0;
      d_nVectors          = 0;
      d_updateFlags       = update_default;
      areAllCellsAffine   = true;
      for (unsigned int iMacroCell = 0;
           iMacroCell < d_matrixFreeDataPtr->n_cell_batches();
           ++iMacroCell)
        {
          areAllCellsAffine =
            areAllCellsAffine &&
            (d_matrixFreeDataPtr->get_mapping_info().get_cell_type(
               iMacroCell) <= dealii::internal::MatrixFreeFunctions::affine);
        }
      areAllCellsCartesian = true;
      for (unsigned int iMacroCell = 0;
           iMacroCell < d_matrixFreeDataPtr->n_cell_batches();
           ++iMacroCell)
        {
          areAllCellsCartesian =
            areAllCellsCartesian &&
            (d_matrixFreeDataPtr->get_mapping_info().get_cell_type(
               iMacroCell) == dealii::internal::MatrixFreeFunctions::cartesian);
        }
      // std::cout << "DEBUG cart " << areAllCellsCartesian << " "
      //           << areAllCellsAffine << std::endl;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::reinit(const unsigned int &blockSize,
                                               const unsigned int &dofHandlerID,
                                               const unsigned int &quadratureID,
                                               const UpdateFlags   updateFlags)
    {
      if ((d_dofHandlerID != dofHandlerID) || (d_updateFlags != updateFlags))
        {
          d_dofHandlerID = dofHandlerID;
          d_quadratureID = quadratureID;
          d_nVectors     = blockSize;
          d_updateFlags  = updateFlags;
          initializeIndexMaps();
          initializeConstraints();
          initializeShapeFunctionAndJacobianData();
          initializeFlattenedIndexMaps();
        }
      else if ((d_quadratureID != quadratureID) && (d_nVectors != blockSize))
        {
          d_quadratureID = quadratureID;
          d_nVectors     = blockSize;
          initializeConstraints();
          initializeShapeFunctionAndJacobianData();
          initializeFlattenedIndexMaps();
        }
      else if (d_quadratureID != quadratureID)
        {
          d_quadratureID = quadratureID;
          initializeShapeFunctionAndJacobianData();
        }
      else if (d_nVectors != blockSize)
        {
          d_nVectors = blockSize;
          initializeConstraints();
          initializeFlattenedIndexMaps();
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::initializeFlattenedIndexMaps()
    {
#if defined(DFTFE_WITH_DEVICE)
      dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                  dftfe::utils::MemorySpace::HOST>
        d_flattenedCellDofIndexToProcessDofIndexMapHost;
      dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                  dftfe::utils::MemorySpace::HOST>
        d_nonAffineReshapeIDsHost;
      if ((memorySpace == dftfe::utils::MemorySpace::DEVICE) &&
          (!areAllCellsAffine))
        {
          d_nonAffineReshapeIDsHost.resize(d_nCells * d_nQuadsPerCell * 3);
          for (unsigned int iCell = 0; iCell < d_nCells; ++iCell)
            {
              for (unsigned int iQuad = 0; iQuad < d_nQuadsPerCell; ++iQuad)
                {
                  for (unsigned int iDim = 0; iDim < 3; ++iDim)
                    {
                      d_nonAffineReshapeIDsHost[iQuad + d_nQuadsPerCell * iDim +
                                                d_nQuadsPerCell * 3 * iCell] =
                        (iDim + 3 * iQuad + d_nQuadsPerCell * 3 * iCell) *
                        d_nVectors;
                    }
                }
            }
        }
      d_nonAffineReshapeIDs.resize(d_nonAffineReshapeIDsHost.size());
      d_nonAffineReshapeIDs.copyFrom(d_nonAffineReshapeIDsHost);
#else
      auto &d_flattenedCellDofIndexToProcessDofIndexMapHost =
        d_flattenedCellDofIndexToProcessDofIndexMap;
#endif
      d_flattenedCellDofIndexToProcessDofIndexMapHost.clear();
      d_flattenedCellDofIndexToProcessDofIndexMapHost.resize(d_nCells *
                                                             d_nDofsPerCell);

      std::transform(d_cellDofIndexToProcessDofIndexMap.begin(),
                     d_cellDofIndexToProcessDofIndexMap.end(),
                     d_flattenedCellDofIndexToProcessDofIndexMapHost.begin(),
                     [&a = this->d_nVectors](auto &c) { return c * a; });
#if defined(DFTFE_WITH_DEVICE)
      d_flattenedCellDofIndexToProcessDofIndexMap.resize(
        d_flattenedCellDofIndexToProcessDofIndexMapHost.size());
      d_flattenedCellDofIndexToProcessDofIndexMap.copyFrom(
        d_flattenedCellDofIndexToProcessDofIndexMapHost);
#endif
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::initializeIndexMaps()
    {
      d_nCells = d_matrixFreeDataPtr->n_physical_cells();
      d_nDofsPerCell =
        d_matrixFreeDataPtr->get_dof_handler(0).get_fe().dofs_per_cell;
      d_cellDofIndexToProcessDofIndexMap.clear();
      d_cellDofIndexToProcessDofIndexMap.resize(d_nCells * d_nDofsPerCell);

      d_cellIndexToCellIdMap.clear();
      d_cellIndexToCellIdMap.resize(d_nCells);

      auto cellPtr = d_matrixFreeDataPtr->get_dof_handler(0).begin_active();
      auto endcPtr = d_matrixFreeDataPtr->get_dof_handler(0).end();

      std::vector<global_size_type>       cellDofIndicesGlobal(d_nDofsPerCell);
      std::map<dealii::CellId, size_type> cellIdToCellIndexMap;

      unsigned int iCell = 0;
      for (; cellPtr != endcPtr; ++cellPtr)
        if (cellPtr->is_locally_owned())
          {
            cellPtr->get_dof_indices(cellDofIndicesGlobal);
            for (unsigned int iDof = 0; iDof < d_nDofsPerCell; ++iDof)
              d_cellDofIndexToProcessDofIndexMap[iCell * d_nDofsPerCell +
                                                 iDof] =
                d_matrixFreeDataPtr->get_vector_partitioner(0)->global_to_local(
                  cellDofIndicesGlobal[iDof]);


            d_cellIndexToCellIdMap[iCell] = cellPtr->id();

            ++iCell;
          }
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::initializeConstraints()
    {
      d_constraintInfo.initialize(d_matrixFreeDataPtr->get_vector_partitioner(
                                    0),
                                  *((*d_constraintsVector)[0]));
      d_constraintInfo.precomputeMaps(
        d_matrixFreeDataPtr->get_vector_partitioner(0)->locally_owned_size() +
          d_matrixFreeDataPtr->get_vector_partitioner(0)->n_ghost_indices(),
        d_nVectors);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::initializeShapeFunctionAndJacobianData()
    {
      const dealii::Quadrature<3> &quadrature =
        d_matrixFreeDataPtr->get_quadrature(d_quadratureID);
      dealii::FEValues<3> fe_values(
        d_matrixFreeDataPtr->get_dof_handler(0).get_fe(),
        quadrature,
        dealii::update_values | dealii::update_gradients |
          dealii::update_jacobians | dealii::update_JxW_values |
          dealii::update_inverse_jacobians);

      d_nQuadsPerCell = quadrature.size();

#if defined(DFTFE_WITH_DEVICE)
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                  dftfe::utils::MemorySpace::HOST>
        d_inverseJacobianDataHost;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                  dftfe::utils::MemorySpace::HOST>
        d_JxWDataHost;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                  dftfe::utils::MemorySpace::HOST>
        d_shapeFunctionDataHost;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                  dftfe::utils::MemorySpace::HOST>
        d_shapeFunctionGradientDataHost;
#else
      auto &d_inverseJacobianDataHost       = d_inverseJacobianData;
      auto &d_JxWDataHost                   = d_JxWData;
      auto &d_shapeFunctionDataHost         = d_shapeFunctionData;
      auto &d_shapeFunctionGradientDataHost = d_shapeFunctionGradientData;
#endif


      d_shapeFunctionDataHost.clear();
      if (d_updateFlags & update_values)
        d_shapeFunctionDataHost.resize(d_nQuadsPerCell * d_nDofsPerCell, 0.0);
      d_shapeFunctionGradientDataHost.clear();
      if (d_updateFlags & update_gradients)
        d_shapeFunctionGradientDataHost.resize(d_nQuadsPerCell *
                                                 d_nDofsPerCell * 3,
                                               0.0);

      d_JxWDataHost.clear();
      if ((d_updateFlags & update_values) || (d_updateFlags & update_gradients))
        d_JxWDataHost.resize(d_nCells * d_nQuadsPerCell);

      d_inverseJacobianDataHost.clear();
      if (d_updateFlags & update_gradients)
        d_inverseJacobianDataHost.resize(areAllCellsCartesian ?
                                           d_nCells * 3 :
                                           (areAllCellsAffine ?
                                              d_nCells * 9 :
                                              d_nCells * 9 * d_nQuadsPerCell));
      const unsigned int nJacobiansPerCell =
        areAllCellsAffine ? 1 : d_nQuadsPerCell;

      auto cellPtr = d_matrixFreeDataPtr->get_dof_handler(0).begin_active();
      auto endcPtr = d_matrixFreeDataPtr->get_dof_handler(0).end();

      unsigned int iCell = 0;
      for (; cellPtr != endcPtr; ++cellPtr)
        if (cellPtr->is_locally_owned())
          {
            fe_values.reinit(cellPtr);
            auto &jacobians        = fe_values.get_jacobians();
            auto &inverseJacobians = fe_values.get_inverse_jacobians();
            if (iCell == 0)
              {
                if (d_updateFlags & update_values)
                  for (unsigned int iNode = 0; iNode < d_nDofsPerCell; ++iNode)
                    for (unsigned int iQuad = 0; iQuad < d_nQuadsPerCell;
                         ++iQuad)
                      d_shapeFunctionDataHost[iQuad * d_nDofsPerCell + iNode] =
                        fe_values.shape_value(iNode, iQuad);


                if (d_updateFlags & update_gradients)
                  for (unsigned int iQuad = 0; iQuad < d_nQuadsPerCell; ++iQuad)
                    for (unsigned int iNode = 0; iNode < d_nDofsPerCell;
                         ++iNode)
                      {
                        const auto &shape_grad_real =
                          fe_values.shape_grad(iNode, iQuad);
                        const auto &shape_grad_reference =
                          apply_transformation(jacobians[iQuad].transpose(),
                                               shape_grad_real);
                        for (unsigned int iDim = 0; iDim < 3; ++iDim)
                          if (areAllCellsAffine)
                            d_shapeFunctionGradientDataHost
                              [d_nQuadsPerCell * d_nDofsPerCell * iDim +
                               d_nDofsPerCell * iQuad + iNode] =
                                shape_grad_reference[iDim];
                          else
                            d_shapeFunctionGradientDataHost
                              [iQuad * d_nDofsPerCell * 3 +
                               d_nDofsPerCell * iDim + iNode] =
                                shape_grad_reference[iDim];
                      }
              }
            for (unsigned int iQuad = 0; iQuad < d_nQuadsPerCell; ++iQuad)
              d_JxWDataHost[iCell * d_nQuadsPerCell + iQuad] =
                fe_values.JxW(iQuad);
            for (unsigned int iQuad = 0; iQuad < nJacobiansPerCell; ++iQuad)
              for (unsigned int iDim = 0; iDim < 3; ++iDim)
                if (areAllCellsCartesian)
                  d_inverseJacobianDataHost[iCell * nJacobiansPerCell * 3 +
                                            iDim * nJacobiansPerCell + iQuad] =
                    inverseJacobians[iQuad][iDim][iDim];
                else
                  for (unsigned int jDim = 0; jDim < 3; ++jDim)
                    d_inverseJacobianDataHost[iCell * nJacobiansPerCell * 9 +
                                              9 * iQuad + jDim * 3 + iDim] =
                      inverseJacobians[iQuad][iDim][jDim];
            ++iCell;
          }

#if defined(DFTFE_WITH_DEVICE)
      d_inverseJacobianData.resize(d_inverseJacobianDataHost.size());
      d_inverseJacobianData.copyFrom(d_inverseJacobianDataHost);
      d_JxWData.resize(d_JxWDataHost.size());
      d_JxWData.copyFrom(d_JxWDataHost);
      d_shapeFunctionData.resize(d_shapeFunctionDataHost.size());
      d_shapeFunctionData.copyFrom(d_shapeFunctionDataHost);
      d_shapeFunctionGradientData.resize(
        d_shapeFunctionGradientDataHost.size());
      d_shapeFunctionGradientData.copyFrom(d_shapeFunctionGradientDataHost);
#endif
    }
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::
      createMultiVector(
        const unsigned int dofHandlerIndex,
        const unsigned int blocksize,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &multiVector) const
    {
      dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
        d_matrixFreeDataPtr->get_vector_partitioner(dofHandlerIndex),
        blocksize,
        multiVector);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::
      distribute(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &multiVector) const
    {
      d_constraintInfo.distribute(multiVector, d_nVectors);
    }


    // template class FEBasisOperations<double,
    //                                  double,
    //                                  dftfe::utils::MemorySpace::HOST>;
    // template class FEBasisOperations<double,
    //                                  double,
    //                                  dftfe::utils::MemorySpace::DEVICE>;
  } // namespace basis
} // namespace dftfe
