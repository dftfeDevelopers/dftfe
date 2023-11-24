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
      areAllCellsAffine   = true;
      d_nOMPThreads       = 1;
      if (const char *penv = std::getenv("DFTFE_NUM_THREADS"))
        d_nOMPThreads = std::stoi(std::string(penv));

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
    FEBasisOperationsBase<
      ValueTypeBasisCoeff,
      ValueTypeBasisData,
      memorySpace>::init(const unsigned int &             dofHandlerID,
                         const std::vector<unsigned int> &quadratureID,
                         const std::vector<UpdateFlags>   updateFlags)
    {
      AssertThrow(
        updateFlags.size() == quadratureID.size(),
        dealii::ExcMessage(
          "DFT-FE Error: Inconsistent size of update flags for FEBasisOperations class."));


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
      d_quadPoints = basisOperationsSrc.d_quadPoints;
      for (unsigned int iQuadIndex = 0;
           iQuadIndex < d_quadratureIDsVector.size();
           ++iQuadIndex)
        {
          unsigned int quadIndex = d_quadratureIDsVector[iQuadIndex];
          if (d_updateFlags[iQuadIndex] & update_inversejacobians)
            {
              d_inverseJacobianData[areAllCellsAffine ? 0 : quadIndex].resize(
                basisOperationsSrc.d_inverseJacobianData
                  .find(areAllCellsAffine ? 0 : quadIndex)
                  ->second.size());
              d_inverseJacobianData[areAllCellsAffine ? 0 : quadIndex].copyFrom(
                basisOperationsSrc.d_inverseJacobianData
                  .find(areAllCellsAffine ? 0 : quadIndex)
                  ->second);
            }
          if (d_updateFlags[iQuadIndex] & update_jxw)
            {
              d_JxWData[quadIndex].resize(
                basisOperationsSrc.d_JxWData.find(quadIndex)->second.size());
              d_JxWData[quadIndex].copyFrom(
                basisOperationsSrc.d_JxWData.find(quadIndex)->second);
            }
          if (d_updateFlags[iQuadIndex] & update_values)
            {
              d_shapeFunctionData[quadIndex].resize(
                basisOperationsSrc.d_shapeFunctionData.find(quadIndex)
                  ->second.size());
              d_shapeFunctionData[quadIndex].copyFrom(
                basisOperationsSrc.d_shapeFunctionData.find(quadIndex)->second);
              if (d_updateFlags[iQuadIndex] & update_transpose)
                {
                  d_shapeFunctionDataTranspose[quadIndex].resize(
                    basisOperationsSrc.d_shapeFunctionDataTranspose
                      .find(quadIndex)
                      ->second.size());
                  d_shapeFunctionDataTranspose[quadIndex].copyFrom(
                    basisOperationsSrc.d_shapeFunctionDataTranspose
                      .find(quadIndex)
                      ->second);
                }
            }
          if (d_updateFlags[iQuadIndex] & update_gradients)
            {
              d_shapeFunctionGradientDataInternalLayout[quadIndex].resize(
                basisOperationsSrc.d_shapeFunctionGradientDataInternalLayout
                  .find(quadIndex)
                  ->second.size());
              d_shapeFunctionGradientDataInternalLayout[quadIndex].copyFrom(
                basisOperationsSrc.d_shapeFunctionGradientDataInternalLayout
                  .find(quadIndex)
                  ->second);
              d_shapeFunctionGradientData[quadIndex].resize(
                basisOperationsSrc.d_shapeFunctionGradientData.find(quadIndex)
                  ->second.size());
              d_shapeFunctionGradientData[quadIndex].copyFrom(
                basisOperationsSrc.d_shapeFunctionGradientData.find(quadIndex)
                  ->second);
              if (d_updateFlags[iQuadIndex] & update_transpose)
                {
                  d_shapeFunctionGradientDataTranspose[quadIndex].resize(
                    basisOperationsSrc.d_shapeFunctionGradientDataTranspose
                      .find(quadIndex)
                      ->second.size());
                  d_shapeFunctionGradientDataTranspose[quadIndex].copyFrom(
                    basisOperationsSrc.d_shapeFunctionGradientDataTranspose
                      .find(quadIndex)
                      ->second);
                }
            }
        }
      if (!std::is_same<ValueTypeBasisCoeff, ValueTypeBasisData>::value)
        for (unsigned int iQuadIndex = 0;
             iQuadIndex < d_quadratureIDsVector.size();
             ++iQuadIndex)
          {
            unsigned int quadIndex = d_quadratureIDsVector[iQuadIndex];
            if (d_updateFlags[iQuadIndex] & update_inversejacobians)
              {
                d_inverseJacobianBasisData[areAllCellsAffine ? 0 : quadIndex]
                  .resize(basisOperationsSrc.d_inverseJacobianBasisData
                            .find(areAllCellsAffine ? 0 : quadIndex)
                            ->second.size());
                d_inverseJacobianBasisData[areAllCellsAffine ? 0 : quadIndex]
                  .copyFrom(basisOperationsSrc.d_inverseJacobianBasisData
                              .find(areAllCellsAffine ? 0 : quadIndex)
                              ->second);
              }
            if (d_updateFlags[iQuadIndex] & update_jxw)
              {
                d_JxWBasisData[quadIndex].resize(
                  basisOperationsSrc.d_JxWBasisData.find(quadIndex)
                    ->second.size());
                d_JxWBasisData[quadIndex].copyFrom(
                  basisOperationsSrc.d_JxWBasisData.find(quadIndex)->second);
              }
            if (d_updateFlags[iQuadIndex] & update_values)
              {
                d_shapeFunctionBasisData[quadIndex].resize(
                  basisOperationsSrc.d_shapeFunctionBasisData.find(quadIndex)
                    ->second.size());
                d_shapeFunctionBasisData[quadIndex].copyFrom(
                  basisOperationsSrc.d_shapeFunctionBasisData.find(quadIndex)
                    ->second);
                if (d_updateFlags[iQuadIndex] & update_transpose)
                  {
                    d_shapeFunctionBasisDataTranspose[quadIndex].resize(
                      basisOperationsSrc.d_shapeFunctionBasisDataTranspose
                        .find(quadIndex)
                        ->second.size());
                    d_shapeFunctionBasisDataTranspose[quadIndex].copyFrom(
                      basisOperationsSrc.d_shapeFunctionBasisDataTranspose
                        .find(quadIndex)
                        ->second);
                  }
              }
            if (d_updateFlags[iQuadIndex] & update_gradients)
              {
                d_shapeFunctionGradientBasisData[quadIndex].resize(
                  basisOperationsSrc.d_shapeFunctionGradientBasisData
                    .find(quadIndex)
                    ->second.size());
                d_shapeFunctionGradientBasisData[quadIndex].copyFrom(
                  basisOperationsSrc.d_shapeFunctionGradientBasisData
                    .find(quadIndex)
                    ->second);
                if (d_updateFlags[iQuadIndex] & update_transpose)
                  {
                    d_shapeFunctionGradientBasisDataTranspose[quadIndex].resize(
                      basisOperationsSrc
                        .d_shapeFunctionGradientBasisDataTranspose
                        .find(quadIndex)
                        ->second.size());
                    d_shapeFunctionGradientBasisDataTranspose[quadIndex]
                      .copyFrom(basisOperationsSrc
                                  .d_shapeFunctionGradientBasisDataTranspose
                                  .find(quadIndex)
                                  ->second);
                  }
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
      auto itr       = std::find(d_quadratureIDsVector.begin(),
                           d_quadratureIDsVector.end(),
                           d_quadratureID);
      AssertThrow(
        itr != d_quadratureIDsVector.end(),
        dealii::ExcMessage(
          "DFT-FE Error: FEBasisOperations Class not initialized with this quadrature Index."));
      d_quadratureIndex = std::distance(d_quadratureIDsVector.begin(), itr);
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
      return d_nQuadsPerCell[d_quadratureIndex];
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
      return transpose ?
               d_shapeFunctionDataTranspose.find(d_quadratureID)->second :
               d_shapeFunctionData.find(d_quadratureID)->second;
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
      return transpose ?
               d_shapeFunctionGradientDataTranspose.find(d_quadratureID)
                 ->second :
               d_shapeFunctionGradientData.find(d_quadratureID)->second;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace> &
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::inverseJacobians() const
    {
      return d_inverseJacobianData.find(areAllCellsAffine ? 0 : d_quadratureID)
        ->second;
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
      return d_quadPoints.find(d_quadratureID)->second;
    }



    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace> &
    FEBasisOperationsBase<ValueTypeBasisCoeff,
                          ValueTypeBasisData,
                          memorySpace>::JxW() const
    {
      return d_JxWData.find(d_quadratureID)->second;
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
      return d_JxWData.find(d_quadratureID)->second;
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
      return d_JxWBasisData.find(d_quadratureID)->second;
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
      return d_inverseJacobianData.find(areAllCellsAffine ? 0 : d_quadratureID)
        ->second;
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
      return d_inverseJacobianBasisData
        .find(areAllCellsAffine ? 0 : d_quadratureID)
        ->second;
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
      return transpose ?
               d_shapeFunctionDataTranspose.find(d_quadratureID)->second :
               d_shapeFunctionData.find(d_quadratureID)->second;
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
      return transpose ?
               d_shapeFunctionBasisDataTranspose.find(d_quadratureID)->second :
               d_shapeFunctionBasisData.find(d_quadratureID)->second;
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
      return transpose ?
               d_shapeFunctionGradientDataTranspose.find(d_quadratureID)
                 ->second :
               d_shapeFunctionGradientData.find(d_quadratureID)->second;
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
               d_shapeFunctionGradientBasisDataTranspose.find(d_quadratureID)
                 ->second :
               d_shapeFunctionGradientBasisData.find(d_quadratureID)->second;
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
      if (d_updateFlags[d_quadratureIndex] & update_gradients)
        tempQuadratureGradientsData.resize(
          areAllCellsCartesian ?
            0 :
            (d_nVectors * d_nQuadsPerCell[d_quadratureIndex] * 3 *
             d_cellsBlockSize));

      if (d_updateFlags[d_quadratureIndex] & update_gradients)
        tempQuadratureGradientsDataNonAffine.resize(
          areAllCellsAffine ? 0 :
                              (d_nVectors * d_nQuadsPerCell[d_quadratureIndex] *
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
          d_matrixFreeDataPtr->get_vector_partitioner(iConstraint),
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
      for (unsigned int iQuadIndex = 0;
           iQuadIndex < d_quadratureIDsVector.size();
           ++iQuadIndex)
        {
          unsigned int quadID = d_quadratureIDsVector[iQuadIndex];
          const dealii::Quadrature<3> &quadrature =
            d_matrixFreeDataPtr->get_quadrature(quadID);
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

          d_nQuadsPerCell[iQuadIndex] = quadrature.size();

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
            d_inverseJacobianData[areAllCellsAffine ? 0 : quadID];
          auto &d_JxWDataHost           = d_JxWData[quadID];
          auto &d_shapeFunctionDataHost = d_shapeFunctionData[quadID];
          auto &d_shapeFunctionGradientDataInternalLayoutHost =
            d_shapeFunctionGradientDataInternalLayout[quadID];
          auto &d_shapeFunctionDataTransposeHost =
            d_shapeFunctionDataTranspose[quadID];
          auto &d_shapeFunctionGradientDataHost =
            d_shapeFunctionGradientData[quadID];
          auto &d_shapeFunctionGradientDataTransposeHost =
            d_shapeFunctionGradientDataTranspose[quadID];
#endif

          if (d_updateFlags[iQuadIndex] & update_quadpoints)
            {
              d_quadPoints[quadID].clear();
              d_quadPoints[quadID].resize(d_nCells *
                                          d_nQuadsPerCell[iQuadIndex] * 3);
            }
          d_shapeFunctionDataHost.clear();
          if (d_updateFlags[iQuadIndex] & update_values)
            d_shapeFunctionDataHost.resize(d_nQuadsPerCell[iQuadIndex] *
                                             d_nDofsPerCell,
                                           0.0);
          d_shapeFunctionDataTransposeHost.clear();
          if ((d_updateFlags[iQuadIndex] & update_values) &&
              (d_updateFlags[iQuadIndex] & update_transpose))
            d_shapeFunctionDataTransposeHost.resize(
              d_nQuadsPerCell[iQuadIndex] * d_nDofsPerCell, 0.0);
          d_shapeFunctionGradientDataInternalLayoutHost.clear();
          d_shapeFunctionGradientDataHost.clear();
          d_shapeFunctionGradientDataTransposeHost.clear();
          if (d_updateFlags[iQuadIndex] & update_gradients)
            {
              d_shapeFunctionGradientDataInternalLayoutHost.resize(
                d_nQuadsPerCell[iQuadIndex] * d_nDofsPerCell * 3, 0.0);
              d_shapeFunctionGradientDataHost.resize(
                d_nQuadsPerCell[iQuadIndex] * d_nDofsPerCell * 3, 0.0);
              if (d_updateFlags[iQuadIndex] & update_transpose)
                d_shapeFunctionGradientDataTransposeHost.resize(
                  d_nQuadsPerCell[iQuadIndex] * d_nDofsPerCell * 3, 0.0);
            }

          d_JxWDataHost.clear();
          if ((d_updateFlags[iQuadIndex] & update_jxw))
            d_JxWDataHost.resize(d_nCells * d_nQuadsPerCell[iQuadIndex]);

          d_inverseJacobianDataHost.clear();
          if (d_updateFlags[iQuadIndex] & update_inversejacobians)
            d_inverseJacobianDataHost.resize(
              areAllCellsCartesian ?
                d_nCells * 3 :
                (areAllCellsAffine ?
                   d_nCells * 9 :
                   d_nCells * 9 * d_nQuadsPerCell[iQuadIndex]));
          const unsigned int nJacobiansPerCell =
            areAllCellsAffine ? 1 : d_nQuadsPerCell[iQuadIndex];


          if (d_updateFlags[iQuadIndex] & update_values)
            {
              for (unsigned int iQuad = 0; iQuad < d_nQuadsPerCell[iQuadIndex];
                   ++iQuad)
                for (unsigned int iNode = 0; iNode < d_nDofsPerCell; ++iNode)
                  d_shapeFunctionDataHost[iQuad * d_nDofsPerCell + iNode] =
                    fe_values_reference.shape_value(iNode, iQuad);
              if (d_updateFlags[iQuadIndex] & update_transpose)
                {
                  for (unsigned int iNode = 0; iNode < d_nDofsPerCell; ++iNode)
                    for (unsigned int iQuad = 0;
                         iQuad < d_nQuadsPerCell[iQuadIndex];
                         ++iQuad)
                      d_shapeFunctionDataTransposeHost
                        [iNode * d_nQuadsPerCell[iQuadIndex] + iQuad] =
                          fe_values_reference.shape_value(iNode, iQuad);
                }
            }


          if (d_updateFlags[iQuadIndex] & update_gradients)
            {
              for (unsigned int iDim = 0; iDim < 3; ++iDim)
                for (unsigned int iQuad = 0;
                     iQuad < d_nQuadsPerCell[iQuadIndex];
                     ++iQuad)
                  for (unsigned int iNode = 0; iNode < d_nDofsPerCell; ++iNode)
                    d_shapeFunctionGradientDataHost
                      [iDim * d_nQuadsPerCell[iQuadIndex] * d_nDofsPerCell +
                       iQuad * d_nDofsPerCell + iNode] =
                        fe_values_reference.shape_grad(iNode, iQuad)[iDim];

              if (areAllCellsAffine)
                d_shapeFunctionGradientDataInternalLayoutHost =
                  d_shapeFunctionGradientDataHost;
              else
                for (unsigned int iDim = 0; iDim < 3; ++iDim)
                  for (unsigned int iQuad = 0;
                       iQuad < d_nQuadsPerCell[iQuadIndex];
                       ++iQuad)
                    std::memcpy(
                      d_shapeFunctionGradientDataInternalLayoutHost.data() +
                        iQuad * d_nDofsPerCell * 3 + d_nDofsPerCell * iDim,
                      d_shapeFunctionGradientDataHost.data() +
                        iDim * d_nQuadsPerCell[iQuadIndex] * d_nDofsPerCell +
                        iQuad * d_nDofsPerCell,
                      d_nDofsPerCell * sizeof(ValueTypeBasisCoeff));


              if (d_updateFlags[iQuadIndex] & update_transpose)
                for (unsigned int iDim = 0; iDim < 3; ++iDim)
                  for (unsigned int iNode = 0; iNode < d_nDofsPerCell; ++iNode)
                    for (unsigned int iQuad = 0;
                         iQuad < d_nQuadsPerCell[iQuadIndex];
                         ++iQuad)
                      d_shapeFunctionGradientDataTransposeHost
                        [iDim * d_nQuadsPerCell[iQuadIndex] * d_nDofsPerCell +
                         iNode * d_nQuadsPerCell[iQuadIndex] + iQuad] =
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
                if (d_updateFlags[iQuadIndex] & update_quadpoints)
                  for (unsigned int iQuad = 0;
                       iQuad < d_nQuadsPerCell[iQuadIndex];
                       ++iQuad)
                    for (unsigned int iDim = 0; iDim < 3; ++iDim)
                      d_quadPoints[quadID]
                                  [iCell * d_nQuadsPerCell[iQuadIndex] * 3 +
                                   iQuad * 3 + iDim] =
                                    fe_values.quadrature_point(iQuad)[iDim];
                if (d_updateFlags[iQuadIndex] & update_jxw)
                  for (unsigned int iQuad = 0;
                       iQuad < d_nQuadsPerCell[iQuadIndex];
                       ++iQuad)
                    d_JxWDataHost[iCell * d_nQuadsPerCell[iQuadIndex] + iQuad] =
                      fe_values.JxW(iQuad);
                if (d_updateFlags[iQuadIndex] & update_inversejacobians)
                  {
                    auto &inverseJacobians = fe_values.get_inverse_jacobians();
                    for (unsigned int iQuad = 0; iQuad < nJacobiansPerCell;
                         ++iQuad)
                      for (unsigned int iDim = 0; iDim < 3; ++iDim)
                        if (areAllCellsCartesian)
                          d_inverseJacobianDataHost[iCell * nJacobiansPerCell *
                                                      3 +
                                                    iDim * nJacobiansPerCell +
                                                    iQuad] =
                            inverseJacobians[iQuad][iDim][iDim];
                        else
                          for (unsigned int jDim = 0; jDim < 3; ++jDim)
                            d_inverseJacobianDataHost[iCell *
                                                        nJacobiansPerCell * 9 +
                                                      9 * iQuad + jDim * 3 +
                                                      iDim] =
                              inverseJacobians[iQuad][iDim][jDim];
                  }
                ++iCell;
              }

#if defined(DFTFE_WITH_DEVICE)
          d_inverseJacobianData[areAllCellsAffine ? 0 : quadID].resize(
            d_inverseJacobianDataHost.size());
          d_inverseJacobianData[areAllCellsAffine ? 0 : quadID].copyFrom(
            d_inverseJacobianDataHost);
          d_JxWData[quadID].resize(d_JxWDataHost.size());
          d_JxWData[quadID].copyFrom(d_JxWDataHost);
          d_shapeFunctionData[quadID].resize(d_shapeFunctionDataHost.size());
          d_shapeFunctionData[quadID].copyFrom(d_shapeFunctionDataHost);
          d_shapeFunctionGradientDataInternalLayout[quadID].resize(
            d_shapeFunctionGradientDataInternalLayoutHost.size());
          d_shapeFunctionGradientDataInternalLayout[quadID].copyFrom(
            d_shapeFunctionGradientDataInternalLayoutHost);
          d_shapeFunctionDataTranspose[quadID].resize(
            d_shapeFunctionDataTransposeHost.size());
          d_shapeFunctionDataTranspose[quadID].copyFrom(
            d_shapeFunctionDataTransposeHost);
          d_shapeFunctionGradientData[quadID].resize(
            d_shapeFunctionGradientDataHost.size());
          d_shapeFunctionGradientData[quadID].copyFrom(
            d_shapeFunctionGradientDataHost);
          d_shapeFunctionGradientDataTranspose[quadID].resize(
            d_shapeFunctionGradientDataTransposeHost.size());
          d_shapeFunctionGradientDataTranspose[quadID].copyFrom(
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
      for (unsigned int iQuadIndex = 0;
           iQuadIndex < d_quadratureIDsVector.size();
           ++iQuadIndex)
        {
          unsigned int quadID = d_quadratureIDsVector[iQuadIndex];
          const dealii::Quadrature<3> &quadrature =
            d_matrixFreeDataPtr->get_quadrature(quadID);
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
            d_inverseJacobianBasisData[areAllCellsAffine ? 0 : quadID];
          auto &d_JxWDataHost           = d_JxWBasisData[quadID];
          auto &d_shapeFunctionDataHost = d_shapeFunctionBasisData[quadID];
          auto &d_shapeFunctionDataTransposeHost =
            d_shapeFunctionBasisDataTranspose[quadID];
          auto &d_shapeFunctionGradientDataHost =
            d_shapeFunctionGradientBasisData[quadID];
          auto &d_shapeFunctionGradientDataTransposeHost =
            d_shapeFunctionGradientBasisDataTranspose[quadID];
#endif


          d_shapeFunctionDataHost.clear();
          if (d_updateFlags[iQuadIndex] & update_values)
            d_shapeFunctionDataHost.resize(d_nQuadsPerCell[iQuadIndex] *
                                             d_nDofsPerCell,
                                           0.0);
          d_shapeFunctionDataTransposeHost.clear();
          if ((d_updateFlags[iQuadIndex] & update_values) &&
              (d_updateFlags[iQuadIndex] & update_transpose))
            d_shapeFunctionDataTransposeHost.resize(
              d_nQuadsPerCell[iQuadIndex] * d_nDofsPerCell, 0.0);
          d_shapeFunctionGradientDataHost.clear();
          d_shapeFunctionGradientDataTransposeHost.clear();
          if (d_updateFlags[iQuadIndex] & update_gradients)
            {
              d_shapeFunctionGradientDataHost.resize(
                d_nQuadsPerCell[iQuadIndex] * d_nDofsPerCell * 3, 0.0);
              if (d_updateFlags[iQuadIndex] & update_transpose)
                d_shapeFunctionGradientDataTransposeHost.resize(
                  d_nQuadsPerCell[iQuadIndex] * d_nDofsPerCell * 3, 0.0);
            }

          d_JxWDataHost.clear();
          if ((d_updateFlags[iQuadIndex] & update_jxw))
            d_JxWDataHost.resize(d_nCells * d_nQuadsPerCell[iQuadIndex]);

          d_inverseJacobianDataHost.clear();
          if (d_updateFlags[iQuadIndex] & update_inversejacobians)
            d_inverseJacobianDataHost.resize(
              areAllCellsCartesian ?
                d_nCells * 3 :
                (areAllCellsAffine ?
                   d_nCells * 9 :
                   d_nCells * 9 * d_nQuadsPerCell[iQuadIndex]));
          const unsigned int nJacobiansPerCell =
            areAllCellsAffine ? 1 : d_nQuadsPerCell[iQuadIndex];

          if (d_updateFlags[iQuadIndex] & update_values)
            {
              for (unsigned int iNode = 0; iNode < d_nDofsPerCell; ++iNode)
                for (unsigned int iQuad = 0;
                     iQuad < d_nQuadsPerCell[iQuadIndex];
                     ++iQuad)
                  d_shapeFunctionDataHost[iQuad * d_nDofsPerCell + iNode] =
                    fe_values_reference.shape_value(iNode, iQuad);
              if (d_updateFlags[iQuadIndex] & update_transpose)
                for (unsigned int iNode = 0; iNode < d_nDofsPerCell; ++iNode)
                  for (unsigned int iQuad = 0;
                       iQuad < d_nQuadsPerCell[iQuadIndex];
                       ++iQuad)
                    d_shapeFunctionDataTransposeHost
                      [iNode * d_nQuadsPerCell[iQuadIndex] + iQuad] =
                        fe_values_reference.shape_value(iNode, iQuad);
            }


          if (d_updateFlags[iQuadIndex] & update_gradients)
            for (unsigned int iQuad = 0; iQuad < d_nQuadsPerCell[iQuadIndex];
                 ++iQuad)
              for (unsigned int iNode = 0; iNode < d_nDofsPerCell; ++iNode)
                {
                  const auto &shape_grad_reference =
                    fe_values_reference.shape_grad(iNode, iQuad);

                  for (unsigned int iDim = 0; iDim < 3; ++iDim)
                    d_shapeFunctionGradientDataHost
                      [iDim * d_nQuadsPerCell[iQuadIndex] * d_nDofsPerCell +
                       iQuad * d_nDofsPerCell + iNode] =
                        shape_grad_reference[iDim];
                  if (d_updateFlags[iQuadIndex] & update_transpose)
                    for (unsigned int iDim = 0; iDim < 3; ++iDim)
                      d_shapeFunctionGradientDataTransposeHost
                        [iDim * d_nQuadsPerCell[iQuadIndex] * d_nDofsPerCell +
                         iNode * d_nQuadsPerCell[iQuadIndex] + iQuad] =
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
                if (d_updateFlags[iQuadIndex] & update_jxw)
                  for (unsigned int iQuad = 0;
                       iQuad < d_nQuadsPerCell[iQuadIndex];
                       ++iQuad)
                    d_JxWDataHost[iCell * d_nQuadsPerCell[iQuadIndex] + iQuad] =
                      fe_values.JxW(iQuad);
                if (d_updateFlags[iQuadIndex] & update_inversejacobians)
                  {
                    auto &inverseJacobians = fe_values.get_inverse_jacobians();
                    for (unsigned int iQuad = 0; iQuad < nJacobiansPerCell;
                         ++iQuad)
                      for (unsigned int iDim = 0; iDim < 3; ++iDim)
                        if (areAllCellsCartesian)
                          d_inverseJacobianDataHost[iCell * nJacobiansPerCell *
                                                      3 +
                                                    iDim * nJacobiansPerCell +
                                                    iQuad] =
                            inverseJacobians[iQuad][iDim][iDim];
                        else
                          for (unsigned int jDim = 0; jDim < 3; ++jDim)
                            d_inverseJacobianDataHost[iCell *
                                                        nJacobiansPerCell * 9 +
                                                      9 * iQuad + jDim * 3 +
                                                      iDim] =
                              inverseJacobians[iQuad][iDim][jDim];
                  }
                ++iCell;
              }

#if defined(DFTFE_WITH_DEVICE)
          d_inverseJacobianBasisData[areAllCellsAffine ? 0 : quadID].resize(
            d_inverseJacobianDataHost.size());
          d_inverseJacobianBasisData[areAllCellsAffine ? 0 : quadID].copyFrom(
            d_inverseJacobianDataHost);
          d_JxWBasisData[quadID].resize(d_JxWDataHost.size());
          d_JxWBasisData[quadID].copyFrom(d_JxWDataHost);
          d_shapeFunctionBasisData[quadID].resize(
            d_shapeFunctionDataHost.size());
          d_shapeFunctionBasisData[quadID].copyFrom(d_shapeFunctionDataHost);
          d_shapeFunctionBasisDataTranspose[quadID].resize(
            d_shapeFunctionDataTransposeHost.size());
          d_shapeFunctionBasisDataTranspose[quadID].copyFrom(
            d_shapeFunctionDataTransposeHost);
          d_shapeFunctionGradientBasisData[quadID].resize(
            d_shapeFunctionGradientDataHost.size());
          d_shapeFunctionGradientBasisData[quadID].copyFrom(
            d_shapeFunctionGradientDataHost);
          d_shapeFunctionGradientBasisDataTranspose[quadID].resize(
            d_shapeFunctionGradientDataTransposeHost.size());
          d_shapeFunctionGradientBasisDataTranspose[quadID].copyFrom(
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
