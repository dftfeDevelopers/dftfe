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
          &constraintsVector,
        std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
          BLASWrapperPtr)
    {
      d_matrixFreeDataPtr = &matrixFreeData;
      d_constraintsVector = &constraintsVector;
      d_BLASWrapperPtr    = BLASWrapperPtr;
      d_dofHandlerID      = 0;
      d_nVectors          = 0;
      d_updateFlags       = update_default;
      areAllCellsAffine   = true;
      d_nOMPThreads       = std::stoi(std::getenv("DFTFE_NUM_THREADS"));
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
      initializeMPIPattern();
      initializeConstraints();
      initializeShapeFunctionAndJacobianData();
      if (!std::is_same<ValueTypeBasisCoeff, ValueTypeBasisData>::value)
        initializeShapeFunctionAndJacobianBasisData();
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    template <dftfe::utils::MemorySpace memorySpaceSrc>
    void
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::
      init(const FEBasisOperationsBase<ValueTypeBasisCoeff,
                                       ValueTypeBasisData,
                                       memorySpaceSrc> &basisOperationsSrc)
    {
      d_matrixFreeDataPtr   = basisOperationsSrc.d_matrixFreeDataPtr;
      d_constraintsVector   = basisOperationsSrc.d_constraintsVector;
      areAllCellsAffine     = basisOperationsSrc.areAllCellsAffine;
      d_nOMPThreads         = basisOperationsSrc.d_nOMPThreads;
      areAllCellsCartesian  = basisOperationsSrc.areAllCellsCartesian;
      d_dofHandlerID        = basisOperationsSrc.d_dofHandlerID;
      d_quadratureIDsVector = basisOperationsSrc.d_quadratureIDsVector;
      d_updateFlags         = basisOperationsSrc.d_updateFlags;
      d_nVectors            = basisOperationsSrc.d_nVectors;
      d_nCells              = basisOperationsSrc.d_nCells;
      d_nDofsPerCell        = basisOperationsSrc.d_nDofsPerCell;
      d_locallyOwnedSize    = basisOperationsSrc.d_locallyOwnedSize;
      d_localSize           = basisOperationsSrc.d_localSize;
      d_cellDofIndexToProcessDofIndexMap =
        basisOperationsSrc.d_cellDofIndexToProcessDofIndexMap;
      d_cellIndexToCellIdMap = basisOperationsSrc.d_cellIndexToCellIdMap;
      d_cellIdToCellIndexMap = basisOperationsSrc.d_cellIdToCellIndexMap;
      d_nQuadsPerCell        = basisOperationsSrc.d_nQuadsPerCell;
      initializeMPIPattern();
      initializeConstraints();
      d_nQuadsPerCell.resize(d_quadratureIDsVector.size());
      d_quadPoints.resize(d_quadratureIDsVector.size());
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
          d_inverseJacobianData[areAllCellsAffine ? 0 : iQuadID].resize(
            basisOperationsSrc
              .d_inverseJacobianData[areAllCellsAffine ? 0 : iQuadID]
              .size());
          d_inverseJacobianData[areAllCellsAffine ? 0 : iQuadID].copyFrom(
            basisOperationsSrc
              .d_inverseJacobianData[areAllCellsAffine ? 0 : iQuadID]);
          d_JxWData[iQuadID].resize(
            basisOperationsSrc.d_JxWData[iQuadID].size());
          d_JxWData[iQuadID].copyFrom(basisOperationsSrc.d_JxWData[iQuadID]);
          d_shapeFunctionData[iQuadID].resize(
            basisOperationsSrc.d_shapeFunctionData[iQuadID].size());
          d_shapeFunctionData[iQuadID].copyFrom(
            basisOperationsSrc.d_shapeFunctionData[iQuadID]);
          d_shapeFunctionGradientDataInternalLayout[iQuadID].resize(
            basisOperationsSrc
              .d_shapeFunctionGradientDataInternalLayout[iQuadID]
              .size());
          d_shapeFunctionGradientDataInternalLayout[iQuadID].copyFrom(
            basisOperationsSrc
              .d_shapeFunctionGradientDataInternalLayout[iQuadID]);
          d_shapeFunctionDataTranspose[iQuadID].resize(
            basisOperationsSrc.d_shapeFunctionDataTranspose[iQuadID].size());
          d_shapeFunctionDataTranspose[iQuadID].copyFrom(
            basisOperationsSrc.d_shapeFunctionDataTranspose[iQuadID]);
          d_shapeFunctionGradientData[iQuadID].resize(
            basisOperationsSrc.d_shapeFunctionGradientData[iQuadID].size());
          d_shapeFunctionGradientData[iQuadID].copyFrom(
            basisOperationsSrc.d_shapeFunctionGradientData[iQuadID]);
          d_shapeFunctionGradientDataTranspose[iQuadID].resize(
            basisOperationsSrc.d_shapeFunctionGradientDataTranspose[iQuadID]
              .size());
          d_shapeFunctionGradientDataTranspose[iQuadID].copyFrom(
            basisOperationsSrc.d_shapeFunctionGradientDataTranspose[iQuadID]);
        }
      if (!std::is_same<ValueTypeBasisCoeff, ValueTypeBasisData>::value)
        {
          d_inverseJacobianBasisData.resize(
            areAllCellsAffine ? 1 : d_quadratureIDsVector.size());
          d_JxWBasisData.resize(d_quadratureIDsVector.size());
          if (d_updateFlags & update_values)
            {
              d_shapeFunctionBasisData.resize(d_quadratureIDsVector.size());
              if (d_updateFlags & update_transpose)
                d_shapeFunctionBasisDataTranspose.resize(
                  d_quadratureIDsVector.size());
            }
          if (d_updateFlags & update_gradients)
            {
              d_shapeFunctionGradientBasisData.resize(
                d_quadratureIDsVector.size());
              if (d_updateFlags & update_transpose)
                d_shapeFunctionGradientBasisDataTranspose.resize(
                  d_quadratureIDsVector.size());
            }
          for (unsigned int iQuadID = 0; iQuadID < d_quadratureIDsVector.size();
               ++iQuadID)
            {
              d_inverseJacobianBasisData[areAllCellsAffine ? 0 : iQuadID]
                .resize(
                  basisOperationsSrc
                    .d_inverseJacobianBasisData[areAllCellsAffine ? 0 : iQuadID]
                    .size());
              d_inverseJacobianBasisData[areAllCellsAffine ? 0 : iQuadID]
                .copyFrom(
                  basisOperationsSrc
                    .d_inverseJacobianBasisData[areAllCellsAffine ? 0 :
                                                                    iQuadID]);
              d_JxWBasisData[iQuadID].resize(
                basisOperationsSrc.d_JxWBasisData[iQuadID].size());
              d_JxWBasisData[iQuadID].copyFrom(
                basisOperationsSrc.d_JxWBasisData[iQuadID]);
              d_shapeFunctionBasisData[iQuadID].resize(
                basisOperationsSrc.d_shapeFunctionBasisData[iQuadID].size());
              d_shapeFunctionBasisData[iQuadID].copyFrom(
                basisOperationsSrc.d_shapeFunctionBasisData[iQuadID]);
              d_shapeFunctionBasisDataTranspose[iQuadID].resize(
                basisOperationsSrc.d_shapeFunctionBasisDataTranspose[iQuadID]
                  .size());
              d_shapeFunctionBasisDataTranspose[iQuadID].copyFrom(
                basisOperationsSrc.d_shapeFunctionBasisDataTranspose[iQuadID]);
              d_shapeFunctionGradientBasisData[iQuadID].resize(
                basisOperationsSrc.d_shapeFunctionGradientBasisData[iQuadID]
                  .size());
              d_shapeFunctionGradientBasisData[iQuadID].copyFrom(
                basisOperationsSrc.d_shapeFunctionGradientBasisData[iQuadID]);
              d_shapeFunctionGradientBasisDataTranspose[iQuadID].resize(
                basisOperationsSrc
                  .d_shapeFunctionGradientBasisDataTranspose[iQuadID]
                  .size());
              d_shapeFunctionGradientBasisDataTranspose[iQuadID].copyFrom(
                basisOperationsSrc
                  .d_shapeFunctionGradientBasisDataTranspose[iQuadID]);
            }
        }
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
                                               const unsigned int &quadratureID,
                                               const bool isResizeTempStorage)
    {
      d_quadratureID = quadratureID;
      d_cellsBlockSize =
        cellsBlockSize == 0 ? d_cellsBlockSize : cellsBlockSize;
      if (d_nVectors != vecBlockSize && vecBlockSize != 0)
        {
          d_nVectors = vecBlockSize;
          initializeFlattenedIndexMaps();
        }
      if (isResizeTempStorage)
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
    const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace> &
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::shapeFunctionData(bool transpose) const
    {
      return transpose ? d_shapeFunctionDataTranspose[d_quadratureID] :
                         d_shapeFunctionData[d_quadratureID];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace> &
    FEBasisOperationsBase<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace>::shapeFunctionGradientData(bool transpose) const
    {
      return transpose ? d_shapeFunctionGradientDataTranspose[d_quadratureID] :
                         d_shapeFunctionGradientData[d_quadratureID];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace> &
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::inverseJacobians() const
    {
      return d_inverseJacobianData[areAllCellsAffine ? 0 : d_quadratureID];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData,
                                      dftfe::utils::MemorySpace::HOST> &
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::quadPoints() const
    {
      return d_quadPoints[d_quadratureID];
    }



    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace> &
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::JxW() const
    {
      return d_JxWData[d_quadratureID];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    template <typename A,
              typename B,
              typename std::enable_if_t<std::is_same<A, B>::value, int>>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::JxWBasisData() const
    {
      return d_JxWData[d_quadratureID];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    template <typename A,
              typename B,
              typename std::enable_if_t<!std::is_same<A, B>::value, int>>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::JxWBasisData() const
    {
      return d_JxWBasisData[d_quadratureID];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    template <typename A,
              typename B,
              typename std::enable_if_t<std::is_same<A, B>::value, int>>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::inverseJacobiansBasisData() const
    {
      return d_inverseJacobianData[areAllCellsAffine ? 0 : d_quadratureID];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    template <typename A,
              typename B,
              typename std::enable_if_t<!std::is_same<A, B>::value, int>>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::inverseJacobiansBasisData() const
    {
      return d_inverseJacobianBasisData[areAllCellsAffine ? 0 : d_quadratureID];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    template <typename A,
              typename B,
              typename std::enable_if_t<std::is_same<A, B>::value, int>>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::shapeFunctionBasisData(bool transpose)
      const
    {
      return transpose ? d_shapeFunctionDataTranspose[d_quadratureID] :
                         d_shapeFunctionData[d_quadratureID];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    template <typename A,
              typename B,
              typename std::enable_if_t<!std::is_same<A, B>::value, int>>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::shapeFunctionBasisData(bool transpose)
      const
    {
      return transpose ? d_shapeFunctionBasisDataTranspose[d_quadratureID] :
                         d_shapeFunctionBasisData[d_quadratureID];
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    template <typename A,
              typename B,
              typename std::enable_if_t<std::is_same<A, B>::value, int>>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperationsBase<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace>::shapeFunctionGradientBasisData(bool transpose) const
    {
      return transpose ? d_shapeFunctionGradientDataTranspose[d_quadratureID] :
                         d_shapeFunctionGradientData[d_quadratureID];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    template <typename A,
              typename B,
              typename std::enable_if_t<!std::is_same<A, B>::value, int>>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperationsBase<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace>::shapeFunctionGradientBasisData(bool transpose) const
    {
      return transpose ?
               d_shapeFunctionGradientBasisDataTranspose[d_quadratureID] :
               d_shapeFunctionGradientBasisData[d_quadratureID];
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
    dealii::CellId
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::cellID(const unsigned int iElem) const
    {
      return d_cellIndexToCellIdMap[iElem];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    unsigned int
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::cellIndex(const dealii::CellId cellid)
      const
    {
      return d_cellIdToCellIndexMap.find(cellid)->second;
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
                          memorySpace>::initializeMPIPattern()
    {
      const std::pair<global_size_type, global_size_type> &locallyOwnedRange =
        d_matrixFreeDataPtr->get_vector_partitioner(d_dofHandlerID)
          ->local_range();

      std::vector<global_size_type> ghostIndices;
      (d_matrixFreeDataPtr->get_vector_partitioner(d_dofHandlerID)
         ->ghost_indices())
        .fill_index_vector(ghostIndices);

      mpiPatternP2P =
        std::make_shared<dftfe::utils::mpi::MPIPatternP2P<memorySpace>>(
          locallyOwnedRange,
          ghostIndices,
          d_matrixFreeDataPtr->get_vector_partitioner(d_dofHandlerID)
            ->get_mpi_communicator());
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

      d_cellIdToCellIndexMap.clear();
      auto cellPtr =
        d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).begin_active();
      auto endcPtr = d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).end();

      std::vector<global_size_type> cellDofIndicesGlobal(d_nDofsPerCell);

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


            d_cellIndexToCellIdMap[iCell]         = cellPtr->id();
            d_cellIdToCellIndexMap[cellPtr->id()] = iCell;


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
      d_constraintInfo.clear();
      d_constraintInfo.resize((*d_constraintsVector).size());
      for (unsigned int iConstraint = 0;
           iConstraint < (*d_constraintsVector).size();
           ++iConstraint)
        d_constraintInfo[iConstraint].initialize(
          d_matrixFreeDataPtr->get_vector_partitioner(d_dofHandlerID),
          *((*d_constraintsVector)[iConstraint]));
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
      d_quadPoints.resize(d_quadratureIDsVector.size());
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
            dealii::update_quadrature_points | dealii::update_jacobians |
              dealii::update_JxW_values | dealii::update_inverse_jacobians);
          dealii::FEValues<3> fe_values_reference(
            d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).get_fe(),
            quadrature,
            dealii::update_values | dealii::update_gradients);
          dealii::Triangulation<3> reference_cell;
          dealii::GridGenerator::hyper_cube(reference_cell, 0., 1.);
          fe_values_reference.reinit(reference_cell.begin());

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
          auto &d_inverseJacobianDataHost =
            d_inverseJacobianData[areAllCellsAffine ? 0 : iQuadID];
          auto &d_JxWDataHost           = d_JxWData[iQuadID];
          auto &d_shapeFunctionDataHost = d_shapeFunctionData[iQuadID];
          auto &d_shapeFunctionGradientDataInternalLayoutHost =
            d_shapeFunctionGradientDataInternalLayout[iQuadID];
          auto &d_shapeFunctionDataTransposeHost =
            d_shapeFunctionDataTranspose[iQuadID];
          auto &d_shapeFunctionGradientDataHost =
            d_shapeFunctionGradientData[iQuadID];
          auto &d_shapeFunctionGradientDataTransposeHost =
            d_shapeFunctionGradientDataTranspose[iQuadID];
#endif

          d_quadPoints[iQuadID].clear();
          d_quadPoints[iQuadID].resize(d_nCells * d_nQuadsPerCell[iQuadID] * 3);
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


          if (d_updateFlags & update_values)
            {
              for (unsigned int iQuad = 0; iQuad < d_nQuadsPerCell[iQuadID];
                   ++iQuad)
                for (unsigned int iNode = 0; iNode < d_nDofsPerCell; ++iNode)
                  d_shapeFunctionDataHost[iQuad * d_nDofsPerCell + iNode] =
                    fe_values_reference.shape_value(iNode, iQuad);
              if (d_updateFlags & update_transpose)
                {
                  for (unsigned int iNode = 0; iNode < d_nDofsPerCell; ++iNode)
                    for (unsigned int iQuad = 0;
                         iQuad < d_nQuadsPerCell[iQuadID];
                         ++iQuad)
                      d_shapeFunctionDataTransposeHost
                        [iNode * d_nQuadsPerCell[iQuadID] + iQuad] =
                          fe_values_reference.shape_value(iNode, iQuad);
                }
            }


          if (d_updateFlags & update_gradients)
            {
              for (unsigned int iDim = 0; iDim < 3; ++iDim)
                for (unsigned int iQuad = 0; iQuad < d_nQuadsPerCell[iQuadID];
                     ++iQuad)
                  for (unsigned int iNode = 0; iNode < d_nDofsPerCell; ++iNode)
                    d_shapeFunctionGradientDataHost
                      [iDim * d_nQuadsPerCell[iQuadID] * d_nDofsPerCell +
                       iQuad * d_nDofsPerCell + iNode] =
                        fe_values_reference.shape_grad(iNode, iQuad)[iDim];

              if (areAllCellsAffine)
                d_shapeFunctionGradientDataInternalLayoutHost =
                  d_shapeFunctionGradientDataHost;
              else
                for (unsigned int iDim = 0; iDim < 3; ++iDim)
                  for (unsigned int iQuad = 0; iQuad < d_nQuadsPerCell[iQuadID];
                       ++iQuad)
                    std::memcpy(
                      d_shapeFunctionGradientDataInternalLayoutHost.data() +
                        iQuad * d_nDofsPerCell * 3 + d_nDofsPerCell * iDim,
                      d_shapeFunctionGradientDataHost.data() +
                        iDim * d_nQuadsPerCell[iQuadID] * d_nDofsPerCell +
                        iQuad * d_nDofsPerCell,
                      d_nDofsPerCell * sizeof(ValueTypeBasisCoeff));


              if (d_updateFlags & update_transpose)
                for (unsigned int iDim = 0; iDim < 3; ++iDim)
                  for (unsigned int iNode = 0; iNode < d_nDofsPerCell; ++iNode)
                    for (unsigned int iQuad = 0;
                         iQuad < d_nQuadsPerCell[iQuadID];
                         ++iQuad)
                      d_shapeFunctionGradientDataTransposeHost
                        [iDim * d_nQuadsPerCell[iQuadID] * d_nDofsPerCell +
                         iNode * d_nQuadsPerCell[iQuadID] + iQuad] =
                          fe_values_reference.shape_grad(iNode, iQuad)[iDim];
            }



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
                for (unsigned int iQuad = 0; iQuad < d_nQuadsPerCell[iQuadID];
                     ++iQuad)
                  for (unsigned int iDim = 0; iDim < 3; ++iDim)
                    d_quadPoints[iQuadID][iCell * d_nQuadsPerCell[iQuadID] * 3 +
                                          iQuad * 3 + iDim] =
                      fe_values.quadrature_point(iQuad)[iDim];
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
    FEBasisOperationsBase<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace>::initializeShapeFunctionAndJacobianBasisData()
    {
      d_inverseJacobianBasisData.resize(
        areAllCellsAffine ? 1 : d_quadratureIDsVector.size());
      d_JxWBasisData.resize(d_quadratureIDsVector.size());
      if (d_updateFlags & update_values)
        {
          d_shapeFunctionBasisData.resize(d_quadratureIDsVector.size());
          if (d_updateFlags & update_transpose)
            d_shapeFunctionBasisDataTranspose.resize(
              d_quadratureIDsVector.size());
        }
      if (d_updateFlags & update_gradients)
        {
          d_shapeFunctionGradientBasisData.resize(d_quadratureIDsVector.size());
          if (d_updateFlags & update_transpose)
            d_shapeFunctionGradientBasisDataTranspose.resize(
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
            dealii::update_jacobians | dealii::update_JxW_values |
              dealii::update_inverse_jacobians);

          dealii::FEValues<3> fe_values_reference(
            d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).get_fe(),
            quadrature,
            dealii::update_values | dealii::update_gradients);
          dealii::Triangulation<3> reference_cell;
          dealii::GridGenerator::hyper_cube(reference_cell, 0., 1.);
          fe_values_reference.reinit(reference_cell.begin());
#if defined(DFTFE_WITH_DEVICE)
          dftfe::utils::MemoryStorage<ValueTypeBasisData,
                                      dftfe::utils::MemorySpace::HOST>
            d_inverseJacobianDataHost;
          dftfe::utils::MemoryStorage<ValueTypeBasisData,
                                      dftfe::utils::MemorySpace::HOST>
            d_JxWDataHost;
          dftfe::utils::MemoryStorage<ValueTypeBasisData,
                                      dftfe::utils::MemorySpace::HOST>
            d_shapeFunctionDataHost;
          dftfe::utils::MemoryStorage<ValueTypeBasisData,
                                      dftfe::utils::MemorySpace::HOST>
            d_shapeFunctionDataTransposeHost;
          dftfe::utils::MemoryStorage<ValueTypeBasisData,
                                      dftfe::utils::MemorySpace::HOST>
            d_shapeFunctionGradientDataHost;
          dftfe::utils::MemoryStorage<ValueTypeBasisData,
                                      dftfe::utils::MemorySpace::HOST>
            d_shapeFunctionGradientDataTransposeHost;
#else
          auto &d_inverseJacobianDataHost =
            d_inverseJacobianBasisData[areAllCellsAffine ? 0 : iQuadID];
          auto &d_JxWDataHost           = d_JxWBasisData[iQuadID];
          auto &d_shapeFunctionDataHost = d_shapeFunctionBasisData[iQuadID];
          auto &d_shapeFunctionDataTransposeHost =
            d_shapeFunctionBasisDataTranspose[iQuadID];
          auto &d_shapeFunctionGradientDataHost =
            d_shapeFunctionGradientBasisData[iQuadID];
          auto &d_shapeFunctionGradientDataTransposeHost =
            d_shapeFunctionGradientBasisDataTranspose[iQuadID];
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
          d_shapeFunctionGradientDataHost.clear();
          d_shapeFunctionGradientDataTransposeHost.clear();
          if (d_updateFlags & update_gradients)
            {
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

          if (d_updateFlags & update_values)
            {
              for (unsigned int iNode = 0; iNode < d_nDofsPerCell; ++iNode)
                for (unsigned int iQuad = 0; iQuad < d_nQuadsPerCell[iQuadID];
                     ++iQuad)
                  d_shapeFunctionDataHost[iQuad * d_nDofsPerCell + iNode] =
                    fe_values_reference.shape_value(iNode, iQuad);
              if (d_updateFlags & update_transpose)
                for (unsigned int iNode = 0; iNode < d_nDofsPerCell; ++iNode)
                  for (unsigned int iQuad = 0; iQuad < d_nQuadsPerCell[iQuadID];
                       ++iQuad)
                    d_shapeFunctionDataTransposeHost
                      [iNode * d_nQuadsPerCell[iQuadID] + iQuad] =
                        fe_values_reference.shape_value(iNode, iQuad);
            }


          if (d_updateFlags & update_gradients)
            for (unsigned int iQuad = 0; iQuad < d_nQuadsPerCell[iQuadID];
                 ++iQuad)
              for (unsigned int iNode = 0; iNode < d_nDofsPerCell; ++iNode)
                {
                  const auto &shape_grad_reference =
                    fe_values_reference.shape_grad(iNode, iQuad);

                  for (unsigned int iDim = 0; iDim < 3; ++iDim)
                    d_shapeFunctionGradientDataHost
                      [iDim * d_nQuadsPerCell[iQuadID] * d_nDofsPerCell +
                       iQuad * d_nDofsPerCell + iNode] =
                        shape_grad_reference[iDim];
                  if (d_updateFlags & update_transpose)
                    for (unsigned int iDim = 0; iDim < 3; ++iDim)
                      d_shapeFunctionGradientDataTransposeHost
                        [iDim * d_nQuadsPerCell[iQuadID] * d_nDofsPerCell +
                         iNode * d_nQuadsPerCell[iQuadID] + iQuad] =
                          shape_grad_reference[iDim];
                }


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
          d_inverseJacobianBasisData[areAllCellsAffine ? 0 : iQuadID].resize(
            d_inverseJacobianDataHost.size());
          d_inverseJacobianBasisData[areAllCellsAffine ? 0 : iQuadID].copyFrom(
            d_inverseJacobianDataHost);
          d_JxWBasisData[iQuadID].resize(d_JxWDataHost.size());
          d_JxWBasisData[iQuadID].copyFrom(d_JxWDataHost);
          d_shapeFunctionBasisData[iQuadID].resize(
            d_shapeFunctionDataHost.size());
          d_shapeFunctionBasisData[iQuadID].copyFrom(d_shapeFunctionDataHost);
          d_shapeFunctionBasisDataTranspose[iQuadID].resize(
            d_shapeFunctionDataTransposeHost.size());
          d_shapeFunctionBasisDataTranspose[iQuadID].copyFrom(
            d_shapeFunctionDataTransposeHost);
          d_shapeFunctionGradientBasisData[iQuadID].resize(
            d_shapeFunctionGradientDataHost.size());
          d_shapeFunctionGradientBasisData[iQuadID].copyFrom(
            d_shapeFunctionGradientDataHost);
          d_shapeFunctionGradientBasisDataTranspose[iQuadID].resize(
            d_shapeFunctionGradientDataTransposeHost.size());
          d_shapeFunctionGradientBasisDataTranspose[iQuadID].copyFrom(
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
        const unsigned int blocksize,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &multiVector) const
    {
      multiVector.reinit(mpiPatternP2P, blocksize);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::
      createScratchMultiVectors(const unsigned int vecBlockSize,
                                const unsigned int numMultiVecs) const
    {
      auto iter = scratchMultiVectors.find(vecBlockSize);
      if (iter == scratchMultiVectors.end())
        {
          scratchMultiVectors[vecBlockSize] =
            std::vector<dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                          memorySpace>>(
              numMultiVecs);
          for (unsigned int iVec = 0; iVec < numMultiVecs; ++iVec)
            scratchMultiVectors[vecBlockSize][iVec].reinit(mpiPatternP2P,
                                                           vecBlockSize);
        }
      else
        {
          scratchMultiVectors[vecBlockSize].resize(
            scratchMultiVectors[vecBlockSize].size() + numMultiVecs);
          for (unsigned int iVec = 0;
               iVec < scratchMultiVectors[vecBlockSize].size();
               ++iVec)
            scratchMultiVectors[vecBlockSize][iVec].reinit(mpiPatternP2P,
                                                           vecBlockSize);
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::clearScratchMultiVectors() const
    {
      scratchMultiVectors.clear();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &
    FEBasisOperationsBase<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace>::getMultiVector(const unsigned int vecBlockSize,
                                   const unsigned int index) const
    {
      AssertThrow(scratchMultiVectors.find(vecBlockSize) !=
                    scratchMultiVectors.end(),
                  dealii::ExcMessage(
                    "DFT-FE Error: MultiVector not found in scratch storage."));
      return scratchMultiVectors[vecBlockSize][index];
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::
      distribute(dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                   memorySpace> &multiVector,
                 unsigned int constraintIndex) const
    {
      d_constraintInfo[constraintIndex ==
                           std::numeric_limits<unsigned int>::max() ?
                         d_dofHandlerID :
                         constraintIndex]
        .distribute(multiVector, multiVector.numVectors());
    }


  } // namespace basis
} // namespace dftfe
