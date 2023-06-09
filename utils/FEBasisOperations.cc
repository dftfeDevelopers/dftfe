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
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      FEBasisOperations(
        dealii::MatrixFree<3, ValueTypeBasisData> &matrixFreeData,
        std::vector<const dealii::AffineConstraints<ValueTypeBasisCoeff> *>
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
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      reinit(const unsigned int &blockSize,
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
        }
      else if ((d_quadratureID != quadratureID) && (d_nVectors != blockSize))
        {
          d_quadratureID = quadratureID;
          d_nVectors     = blockSize;
          initializeConstraints();
          initializeShapeFunctionAndJacobianData();
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
        }
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      initializeIndexMaps()
    {
      d_nMacroCells  = d_matrixFreeDataPtr->n_cell_batches();
      d_nCells       = d_matrixFreeDataPtr->n_physical_cells();
      d_nDofsPerCell = d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID)
                         .get_fe()
                         .dofs_per_cell;
      d_cellDofIndexToProcessDofIndexMap.clear();
      d_cellDofIndexToProcessDofIndexMap.resize(d_nCells * d_nDofsPerCell);

      d_cellIndexToCellIdMap.clear();
      d_cellIndexToCellIdMap.resize(d_nCells);

      if (d_updateFlags & update_macrocell_map)
        {
          d_cellIndexToMacroCellSubCellIndexMap.clear();
          d_cellIndexToMacroCellSubCellIndexMap.resize(d_nCells);

          d_macroCellSubCellDofIndexToProcessDofIndexMap.clear();
          d_macroCellSubCellDofIndexToProcessDofIndexMap.resize(d_nCells *
                                                                d_nDofsPerCell);
        }

      auto cellPtr =
        d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).begin_active();
      auto endcPtr = d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).end();

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
                d_matrixFreeDataPtr->get_vector_partitioner(d_dofHandlerID)
                  ->global_to_local(cellDofIndicesGlobal[iDof]);

            if (d_updateFlags & update_macrocell_map)
              cellIdToCellIndexMap[cellPtr->id()] = iCell;

            d_cellIndexToCellIdMap[iCell] = cellPtr->id();

            ++iCell;
          }

      iCell = 0;
      for (unsigned int iMacroCell = 0; iMacroCell < d_nMacroCells;
           ++iMacroCell)
        {
          const unsigned int numberSubCells =
            d_matrixFreeDataPtr->n_components_filled(iMacroCell);
          for (unsigned int iSubCell = 0; iSubCell < numberSubCells; ++iSubCell)
            {
              cellPtr = d_matrixFreeDataPtr->get_cell_iterator(iMacroCell,
                                                               iSubCell,
                                                               d_dofHandlerID);
              size_type cellIndex = cellIdToCellIndexMap[cellPtr->id()];
              d_cellIndexToMacroCellSubCellIndexMap[cellIndex] = iCell;
              std::copy(d_cellDofIndexToProcessDofIndexMap.begin() +
                          cellIndex * d_nDofsPerCell,
                        d_cellDofIndexToProcessDofIndexMap.begin() +
                          (cellIndex + 1) * d_nDofsPerCell,
                        d_macroCellSubCellDofIndexToProcessDofIndexMap.begin() +
                          iCell * d_nDofsPerCell);
              ++iCell;
            }
        }
    }



    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      initializeConstraints()
    {
      d_constraintInfo.initialize(d_matrixFreeDataPtr->get_vector_partitioner(
                                    d_dofHandlerID),
                                  *((*d_constraintsVector)[d_dofHandlerID]));
      d_constraintInfo.precomputeMaps(
        d_matrixFreeDataPtr->get_vector_partitioner(d_dofHandlerID)
            ->locally_owned_size() +
          d_matrixFreeDataPtr->get_vector_partitioner(d_dofHandlerID)
            ->n_ghost_indices(),
        d_nVectors);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      initializeShapeFunctionAndJacobianData()
    {
      const dealii::Quadrature<3> &quadrature =
        d_matrixFreeDataPtr->get_quadrature(d_quadratureID);
      dealii::FEValues<3> fe_values(
        d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).get_fe(),
        quadrature,
        dealii::update_values | dealii::update_gradients |
          dealii::update_jacobians | dealii::update_JxW_values |
          dealii::update_inverse_jacobians);

      d_nQuadsPerCell = quadrature.size();

#if defined(DFTFE_WITH_DEVICE)
      std::map<dealii::CellId,
               dftfe::utils::MemoryStorage<ValueTypeBasisData,
                                           dftfe::utils::MemorySpace::HOST>>
        d_inverseJacobianDataHost;
      std::map<dealii::CellId,
               dftfe::utils::MemoryStorage<ValueTypeBasisData,
                                           dftfe::utils::MemorySpace::HOST>>
        d_JxWDataHost;
      dftfe::utils::MemoryStorage<ValueTypeBasisData,
                                  dftfe::utils::MemorySpace::HOST>
        d_shapeFunctionDataHost;
      dftfe::utils::MemoryStorage<ValueTypeBasisData,
                                  dftfe::utils::MemorySpace::HOST>
        d_shapeFunctionGradientDataHost;
      if (memorySpace == dftfe::utils::MemorySpace::HOST)
        {
          &d_inverseJacobianDataHost       = d_inverseJacobianData;
          &d_JxWDataHost                   = d_JxWData;
          &d_shapeFunctionDataHost         = d_shapeFunctionData;
          &d_shapeFunctionGradientDataHost = d_shapeFunctionGradientData;
        }
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
        d_inverseJacobianDataHost.resize(
          areAllCellsAffine ? d_nCells * 9 : d_nCells * 9 * d_nQuadsPerCell);
      const unsigned int nJacobiansPerCell =
        areAllCellsAffine ? 1 : d_nQuadsPerCell;

      auto cellPtr =
        d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).begin_active();
      auto endcPtr = d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).end();

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
                    for (unsigned int q_point = 0; q_point < d_nQuadsPerCell;
                         ++q_point)
                      d_shapeFunctionDataHost[q_point * d_nDofsPerCell +
                                              iNode] =
                        fe_values.shape_value(iNode, q_point);


                if (d_updateFlags & update_gradients)
                  for (unsigned int q_point = 0; q_point < d_nQuadsPerCell;
                       ++q_point)
                    for (unsigned int iNode = 0; iNode < d_nDofsPerCell;
                         ++iNode)
                      {
                        const auto &shape_grad_real =
                          fe_values.shape_grad(iNode, q_point);
                        const auto &shape_grad_reference =
                          apply_transformation(jacobians[q_point].transpose(),
                                               shape_grad_real);
                        for (unsigned int iDim = 0; iDim < 3; ++iDim)
                          d_shapeFunctionGradientDataHost
                            [d_nQuadsPerCell * d_nDofsPerCell * iDim +
                             d_nDofsPerCell * q_point + iNode] =
                              shape_grad_reference[iDim];
                      }
              }
            for (unsigned int q_point = 0; q_point < d_nQuadsPerCell; ++q_point)
              d_JxWDataHost[iCell * d_nQuadsPerCell + q_point] =
                fe_values.JxW(q_point);
            for (unsigned int q_point = 0; q_point < nJacobiansPerCell;
                 ++q_point)
              for (unsigned int iDim = 0; iDim < 3; ++iDim)
                for (unsigned int jDim = 0; jDim < 3; ++jDim)
                  d_inverseJacobianDataHost[iCell * nJacobiansPerCell * 9 +
                                            q_point * 9 + jDim * 3 + iDim] =
                    inverseJacobians[q_point][jDim][iDim];
          }

#if defined(DFTFE_WITH_DEVICE)
      if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
        {
          d_inverseJacobianData.resize(d_inverseJacobianDataHost.size());
          d_inverseJacobianData.copyFrom(d_inverseJacobianDataHost);
          d_JxWData.resize(d_JxWDataHost.size());
          d_JxWData.copyFrom(d_JxWDataHost);
          d_shapeFunctionData.resize(d_shapeFunctionDataHost.size());
          d_shapeFunctionData.copyFrom(d_shapeFunctionDataHost);
          d_shapeFunctionGradientData.resize(
            d_shapeFunctionGradientDataHost.size());
          d_shapeFunctionGradientData.copyFrom(d_shapeFunctionGradientDataHost);
        }
#endif
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      interpolate(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &nodalData,
        std::map<dealii::CellId,
                 dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                             dftfe::utils::MemorySpace::HOST>>
          *quadratureValues,
        std::map<dealii::CellId,
                 dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                             dftfe::utils::MemorySpace::HOST>>
          *  quadratureGradients,
        bool useMacroCellSubCellOrdering) const
    {
      if (memorySpace == dftfe::utils::MemorySpace::HOST)
        {
          for (unsigned int iCell = 0; iCell < d_nCells; ++iCell)
            {
              dealii::CellId currentCellId = d_cellIndexToCellIdMap[iCell];
              auto &cellQuadratureData     = (*quadratureValues)[currentCellId];
              cellQuadratureData.resize(d_nQuadsPerCell * d_nVectors);

              auto &cellQuadratureGradientData =
                (quadratureGradients != NULL) ?
                  (*quadratureGradients)[currentCellId] :
                  NULL;
              if (quadratureGradients != NULL)
                cellQuadratureGradientData.resize(d_nQuadsPerCell * d_nVectors *
                                                  3);
              interpolateHostKernel(
                nodalData,
                &cellQuadratureData,
                &cellQuadratureGradientData,
                std::pair<unsigned int, unsigned int>(iCell, iCell + 1),
                useMacroCellSubCellOrdering);
            }
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      interpolate(dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                    memorySpace> &nodalData,
                  dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
                    *quadratureValues,
                  dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
                    *  quadratureGradients,
                  bool useMacroCellSubCellOrdering) const
    {
      if (memorySpace == dftfe::utils::MemorySpace::HOST)
        {
          interpolateHostKernel(nodalData,
                                quadratureValues,
                                quadratureGradients,
                                std::pair<unsigned int, unsigned int>(0,
                                                                      d_nCells),
                                useMacroCellSubCellOrdering);
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      integrateWithBasis(
        const std::map<
          dealii::CellId,
          dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                      dftfe::utils::MemorySpace::HOST>>
          &quadratureValues,
        std::map<dealii::CellId,
                 dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                             dftfe::utils::MemorySpace::HOST>>
          *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &  nodalData,
        bool useMacroCellSubCellOrdering) const
    {
      if (memorySpace == dftfe::utils::MemorySpace::HOST)
        {
          for (unsigned int iCell = 0; iCell < d_nCells; ++iCell)
            {
              dealii::CellId currentCellId = d_cellIndexToCellIdMap[iCell];
              auto &cellQuadratureData     = (*quadratureValues)[currentCellId];

              auto &cellQuadratureGradientData =
                (quadratureGradients != NULL) ?
                  (*quadratureGradients)[currentCellId] :
                  NULL;
              integrateWithBasisHostKernel(
                &cellQuadratureData,
                &cellQuadratureGradientData,
                nodalData,
                std::pair<unsigned int, unsigned int>(iCell, iCell + 1),
                useMacroCellSubCellOrdering);
            }
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      integrateWithBasis(
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
          *quadratureValues,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
          *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &  nodalData,
        bool useMacroCellSubCellOrdering) const
    {
      if (memorySpace == dftfe::utils::MemorySpace::HOST)
        {
          integrateWithBasisHostKernel(
            quadratureValues,
            quadratureGradients,
            nodalData,
            std::pair<unsigned int, unsigned int>(0, d_nCells),
            useMacroCellSubCellOrdering);
        }
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      extractToCellNodalData(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &nodalData,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
          *  cellNodalDataPtr,
        bool useMacroCellSubCellOrdering) const
    {
      extractToCellNodalDataHostKernel(
        nodalData,
        cellNodalDataPtr,
        std::pair<unsigned int, unsigned int>(0, d_nCells),
        useMacroCellSubCellOrdering);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      accumulateFromCellNodalData(
        const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace>
          *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &  nodalData,
        bool useMacroCellSubCellOrdering) const
    {
      accumulateFromCellNodalDataHostKernel(
        cellNodalDataPtr,
        nodalData,
        std::pair<unsigned int, unsigned int>(0, d_nCells),
        useMacroCellSubCellOrdering);
    }

  } // namespace basis
} // namespace dftfe
