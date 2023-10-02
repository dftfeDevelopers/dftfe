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
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::init(const unsigned int &dofHandlerID,
                                             const std::vector<unsigned int>
                                               &               quadratureID,
                                             const UpdateFlags updateFlags)
    {
      d_dofHandlerID        = dofHandlerID;
      d_quadratureIDsVector = quadratureID;
      d_updateFlags         = updateFlags;
      initializeIndexMaps();
      initializeConstraints();
      initializeShapeFunctionAndJacobianData();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::reinit(const unsigned int &vecBlockSize,
                                               const unsigned int
                                                 &cellsBlockSize,
                                               const unsigned int &quadratureID)
    {
      d_quadratureID   = quadratureID;
      d_cellsBlockSize = cellsBlockSize;
      if (d_nVectors != vecBlockSize)
        {
          d_nVectors = vecBlockSize;
          initializeFlattenedIndexMaps();
        }
      resizeTempStorage();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    unsigned int
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::nQuadsPerCell() const
    {
      return d_nQuadsPerCell[d_quadratureID];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    unsigned int
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::nDofsPerCell() const
    {
      return d_nDofsPerCell;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    unsigned int
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::nCells() const
    {
      return d_nCells;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    unsigned int
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::nRelaventDofs() const
    {
      return d_localSize;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    unsigned int
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::nOwnedDofs() const
    {
      return d_locallyOwnedSize;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const ValueTypeBasisCoeff *
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::shapeFunctionData(bool transpose) const
    {
      return transpose ? d_shapeFunctionDataTranspose[d_quadratureID].data() :
                         d_shapeFunctionData[d_quadratureID].data();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const ValueTypeBasisCoeff *
    FEBasisOperationsBase<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace>::shapeFunctionGradientData(bool transpose) const
    {
      return transpose ?
               d_shapeFunctionGradientDataTranspose[d_quadratureID].data() :
               d_shapeFunctionGradientData[d_quadratureID].data();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const ValueTypeBasisCoeff *
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::inverseJacobians() const
    {
      return d_inverseJacobianData[areAllCellsAffine ? 0 : d_quadratureID]
        .data();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const ValueTypeBasisCoeff *
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::JxW() const
    {
      return d_inverseJacobianData[areAllCellsAffine ? 0 : d_quadratureID]
        .data();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    unsigned int
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::cellsTypeFlag() const
    {
      return (unsigned int)areAllCellsAffine +
             (unsigned int)areAllCellsCartesian;
    }



    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::resizeTempStorage()
    {
      tempCellNodalData.resize(d_nVectors * d_nDofsPerCell * d_cellsBlockSize);

      if (d_updateFlags & update_gradients)
        tempQuadratureGradientsData.resize(
          areAllCellsCartesian ? 0 :
                                 (d_nVectors * d_nQuadsPerCell[d_quadratureID] *
                                  3 * d_cellsBlockSize));

      if (d_updateFlags & update_gradients)
        tempQuadratureGradientsDataNonAffine.resize(
          areAllCellsAffine ? 0 :
                              (d_nVectors * d_nQuadsPerCell[d_quadratureID] *
                               3 * d_cellsBlockSize));
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
      d_nCells       = d_matrixFreeDataPtr->n_physical_cells();
      d_nDofsPerCell = d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID)
                         .get_fe()
                         .dofs_per_cell;
      d_locallyOwnedSize =
        d_matrixFreeDataPtr->get_vector_partitioner(d_dofHandlerID)
          ->locally_owned_size();
      d_localSize = d_locallyOwnedSize +
                    d_matrixFreeDataPtr->get_vector_partitioner(d_dofHandlerID)
                      ->n_ghost_indices();
      d_cellDofIndexToProcessDofIndexMap.clear();
      d_cellDofIndexToProcessDofIndexMap.resize(d_nCells * d_nDofsPerCell);

      d_cellIndexToCellIdMap.clear();
      d_cellIndexToCellIdMap.resize(d_nCells);

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
                                    d_dofHandlerID),
                                  *((*d_constraintsVector)[d_dofHandlerID]));
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::initializeShapeFunctionAndJacobianData()
    {
      d_nQuadsPerCell.resize(d_quadratureIDsVector.size());
      d_inverseJacobianData.resize(
        areAllCellsAffine ? 1 : d_quadratureIDsVector.size());
      d_JxWData.resize(d_quadratureIDsVector.size());
      if (d_updateFlags & update_values)
        {
          d_shapeFunctionData.resize(d_quadratureIDsVector.size());
          if (d_updateFlags & update_transpose)
            d_shapeFunctionDataTranspose.resize(d_quadratureIDsVector.size());
        }
      if (d_updateFlags & update_gradients)
        {
          d_shapeFunctionGradientDataInternalLayout.resize(
            d_quadratureIDsVector.size());
          d_shapeFunctionGradientData.resize(d_quadratureIDsVector.size());
          if (d_updateFlags & update_transpose)
            d_shapeFunctionGradientDataTranspose.resize(
              d_quadratureIDsVector.size());
        }
      for (unsigned int iQuadID = 0; iQuadID < d_quadratureIDsVector.size();
           ++iQuadID)
        {
          const dealii::Quadrature<3> &quadrature =
            d_matrixFreeDataPtr->get_quadrature(d_quadratureIDsVector[iQuadID]);
          dealii::FEValues<3> fe_values(
            d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).get_fe(),
            quadrature,
            dealii::update_values | dealii::update_gradients |
              dealii::update_jacobians | dealii::update_JxW_values |
              dealii::update_inverse_jacobians);

          d_nQuadsPerCell[iQuadID] = quadrature.size();

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
            d_shapeFunctionDataTransposeHost;
          dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                      dftfe::utils::MemorySpace::HOST>
            d_shapeFunctionGradientDataInternalLayoutHost;
          dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                      dftfe::utils::MemorySpace::HOST>
            d_shapeFunctionGradientDataHost;
          dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                      dftfe::utils::MemorySpace::HOST>
            d_shapeFunctionGradientDataTransposeHost;
#else
          auto &d_inverseJacobianDataHost = d_inverseJacobianData;
          auto &d_JxWDataHost             = d_JxWData;
          auto &d_shapeFunctionDataHost   = d_shapeFunctionData;
          auto &d_shapeFunctionGradientDataInternalLayoutHost =
            d_shapeFunctionGradientDataInternalLayout;
          auto &d_shapeFunctionDataTransposeHost = d_shapeFunctionDataTranspose;
          auto &d_shapeFunctionGradientDataHost  = d_shapeFunctionGradientData;
          auto &d_shapeFunctionGradientDataTransposeHost =
            d_shapeFunctionGradientDataTranspose;
#endif


          d_shapeFunctionDataHost.clear();
          if (d_updateFlags & update_values)
            d_shapeFunctionDataHost.resize(d_nQuadsPerCell[iQuadID] *
                                             d_nDofsPerCell,
                                           0.0);
          d_shapeFunctionDataTransposeHost.clear();
          if ((d_updateFlags & update_values) &&
              (d_updateFlags & update_transpose))
            d_shapeFunctionDataTransposeHost.resize(d_nQuadsPerCell[iQuadID] *
                                                      d_nDofsPerCell,
                                                    0.0);
          d_shapeFunctionGradientDataInternalLayoutHost.clear();
          d_shapeFunctionGradientDataHost.clear();
          d_shapeFunctionGradientDataTransposeHost.clear();
          if (d_updateFlags & update_gradients)
            {
              d_shapeFunctionGradientDataInternalLayoutHost.resize(
                d_nQuadsPerCell[iQuadID] * d_nDofsPerCell * 3, 0.0);
              d_shapeFunctionGradientDataHost.resize(d_nQuadsPerCell[iQuadID] *
                                                       d_nDofsPerCell * 3,
                                                     0.0);
              if (d_updateFlags & update_transpose)
                d_shapeFunctionGradientDataTransposeHost.resize(
                  d_nQuadsPerCell[iQuadID] * d_nDofsPerCell * 3, 0.0);
            }

          d_JxWDataHost.clear();
          if ((d_updateFlags & update_values) ||
              (d_updateFlags & update_gradients))
            d_JxWDataHost.resize(d_nCells * d_nQuadsPerCell[iQuadID]);

          d_inverseJacobianDataHost.clear();
          if (d_updateFlags & update_gradients)
            d_inverseJacobianDataHost.resize(
              areAllCellsCartesian ?
                d_nCells * 3 :
                (areAllCellsAffine ? d_nCells * 9 :
                                     d_nCells * 9 * d_nQuadsPerCell[iQuadID]));
          const unsigned int nJacobiansPerCell =
            areAllCellsAffine ? 1 : d_nQuadsPerCell[iQuadID];

          auto cellPtr =
            d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).begin_active();
          auto endcPtr =
            d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).end();

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
                      {
                        for (unsigned int iNode = 0; iNode < d_nDofsPerCell;
                             ++iNode)
                          for (unsigned int iQuad = 0;
                               iQuad < d_nQuadsPerCell[iQuadID];
                               ++iQuad)
                            d_shapeFunctionDataHost[iQuad * d_nDofsPerCell +
                                                    iNode] =
                              fe_values.shape_value(iNode, iQuad);
                        if (d_updateFlags & update_transpose)
                          for (unsigned int iNode = 0; iNode < d_nDofsPerCell;
                               ++iNode)
                            for (unsigned int iQuad = 0;
                                 iQuad < d_nQuadsPerCell[iQuadID];
                                 ++iQuad)
                              d_shapeFunctionDataTransposeHost
                                [iNode * d_nQuadsPerCell[iQuadID] + iQuad] =
                                  fe_values.shape_value(iNode, iQuad);
                      }


                    if (d_updateFlags & update_gradients)
                      for (unsigned int iQuad = 0;
                           iQuad < d_nQuadsPerCell[iQuadID];
                           ++iQuad)
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
                                d_shapeFunctionGradientDataInternalLayoutHost
                                  [d_nQuadsPerCell[iQuadID] * d_nDofsPerCell *
                                     iDim +
                                   d_nDofsPerCell * iQuad + iNode] =
                                    shape_grad_reference[iDim];
                              else
                                d_shapeFunctionGradientDataInternalLayoutHost
                                  [iQuad * d_nDofsPerCell * 3 +
                                   d_nDofsPerCell * iDim + iNode] =
                                    shape_grad_reference[iDim];


                            for (unsigned int iDim = 0; iDim < 3; ++iDim)
                              d_shapeFunctionGradientDataHost
                                [iDim * d_nQuadsPerCell[iQuadID] *
                                   d_nDofsPerCell +
                                 iQuad * d_nDofsPerCell + iNode] =
                                  shape_grad_reference[iDim];
                            if (d_updateFlags & update_transpose)
                              for (unsigned int iDim = 0; iDim < 3; ++iDim)
                                d_shapeFunctionGradientDataTransposeHost
                                  [iDim * d_nQuadsPerCell[iQuadID] *
                                     d_nDofsPerCell +
                                   iNode * d_nQuadsPerCell[iQuadID] + iQuad] =
                                    shape_grad_reference[iDim];
                          }
                  }
                for (unsigned int iQuad = 0; iQuad < d_nQuadsPerCell[iQuadID];
                     ++iQuad)
                  d_JxWDataHost[iCell * d_nQuadsPerCell[iQuadID] + iQuad] =
                    fe_values.JxW(iQuad);
                for (unsigned int iQuad = 0; iQuad < nJacobiansPerCell; ++iQuad)
                  for (unsigned int iDim = 0; iDim < 3; ++iDim)
                    if (areAllCellsCartesian)
                      d_inverseJacobianDataHost[iCell * nJacobiansPerCell * 3 +
                                                iDim * nJacobiansPerCell +
                                                iQuad] =
                        inverseJacobians[iQuad][iDim][iDim];
                    else
                      for (unsigned int jDim = 0; jDim < 3; ++jDim)
                        d_inverseJacobianDataHost[iCell * nJacobiansPerCell *
                                                    9 +
                                                  9 * iQuad + jDim * 3 + iDim] =
                          inverseJacobians[iQuad][iDim][jDim];
                ++iCell;
              }

#if defined(DFTFE_WITH_DEVICE)
          d_inverseJacobianData[areAllCellsAffine ? 0 : iQuadID].resize(
            d_inverseJacobianDataHost.size());
          d_inverseJacobianData[areAllCellsAffine ? 0 : iQuadID].copyFrom(
            d_inverseJacobianDataHost);
          d_JxWData[iQuadID].resize(d_JxWDataHost.size());
          d_JxWData[iQuadID].copyFrom(d_JxWDataHost);
          d_shapeFunctionData[iQuadID].resize(d_shapeFunctionDataHost.size());
          d_shapeFunctionData[iQuadID].copyFrom(d_shapeFunctionDataHost);
          d_shapeFunctionGradientDataInternalLayout[iQuadID].resize(
            d_shapeFunctionGradientDataInternalLayoutHost.size());
          d_shapeFunctionGradientDataInternalLayout[iQuadID].copyFrom(
            d_shapeFunctionGradientDataInternalLayoutHost);
          d_shapeFunctionDataTranspose[iQuadID].resize(
            d_shapeFunctionDataTransposeHost.size());
          d_shapeFunctionDataTranspose[iQuadID].copyFrom(
            d_shapeFunctionDataTransposeHost);
          d_shapeFunctionGradientData[iQuadID].resize(
            d_shapeFunctionGradientDataHost.size());
          d_shapeFunctionGradientData[iQuadID].copyFrom(
            d_shapeFunctionGradientDataHost);
          d_shapeFunctionGradientDataTranspose[iQuadID].resize(
            d_shapeFunctionGradientDataTransposeHost.size());
          d_shapeFunctionGradientDataTranspose[iQuadID].copyFrom(
            d_shapeFunctionGradientDataTransposeHost);
#endif
        }
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
  } // namespace basis
} // namespace dftfe
