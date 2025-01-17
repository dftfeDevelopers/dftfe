// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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
#include <FEBasisOperationsKernelsInternal.h>
#include <dftUtils.h>
namespace dftfe
{
  namespace basis
  {
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      FEBasisOperations(
        std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
          BLASWrapperPtr)
    {
      d_BLASWrapperPtr = BLASWrapperPtr;
      d_nOMPThreads    = 1;
      if (const char *penv = std::getenv("DFTFE_NUM_THREADS"))
        d_nOMPThreads = std::stoi(std::string(penv));
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      clear()
    {
      d_constraintInfo.clear();
      d_cellDofIndexToProcessDofIndexMap.clear();
      d_quadPoints.clear();
      d_flattenedCellDofIndexToProcessDofIndexMap.clear();
      d_cellIndexToCellIdMap.clear();
      d_cellIdToCellIndexMap.clear();
      d_inverseJacobianData.clear();
      d_JxWData.clear();
      d_shapeFunctionData.clear();
      d_shapeFunctionGradientDataInternalLayout.clear();
      d_shapeFunctionGradientData.clear();
      d_shapeFunctionDataTranspose.clear();
      d_shapeFunctionGradientDataTranspose.clear();
      d_inverseJacobianBasisData.clear();
      d_JxWBasisData.clear();
      d_shapeFunctionBasisData.clear();
      d_shapeFunctionGradientBasisData.clear();
      d_shapeFunctionBasisDataTranspose.clear();
      d_shapeFunctionGradientBasisDataTranspose.clear();

      d_cellStiffnessMatrixBasisType.clear();
      d_cellStiffnessMatrixCoeffType.clear();
      d_cellMassMatrixBasisType.clear();
      d_cellMassMatrixCoeffType.clear();
      d_cellInverseMassVectorBasisType.clear();
      d_cellInverseMassVectorCoeffType.clear();
      d_cellInverseSqrtMassVectorBasisType.clear();
      d_cellInverseSqrtMassVectorCoeffType.clear();
      d_inverseSqrtMassVectorBasisType.clear();
      d_inverseSqrtMassVectorCoeffType.clear();
      d_sqrtMassVectorBasisType.clear();
      d_sqrtMassVectorCoeffType.clear();
      scratchMultiVectors.clear();
      tempCellNodalData.clear();
      tempQuadratureGradientsData.clear();
      tempQuadratureGradientsDataNonAffine.clear();

      d_cellStiffnessVectorBasisType.clear();
      d_cellInverseStiffnessVectorBasisType.clear();
      d_cellSqrtStiffnessVectorBasisType.clear();
      d_cellInverseSqrtStiffnessVectorBasisType.clear();
      d_inverseSqrtStiffnessVectorBasisType.clear();
      d_sqrtStiffnessVectorBasisType.clear();
      d_inverseStiffnessVectorBasisType.clear();
      d_stiffnessVectorBasisType.clear();

      d_cellInverseStiffnessVectorCoeffType.clear();
      d_cellInverseSqrtStiffnessVectorCoeffType.clear();
      d_cellStiffnessVectorCoeffType.clear();
      d_cellSqrtStiffnessVectorCoeffType.clear();
      d_inverseSqrtStiffnessVectorCoeffType.clear();
      d_sqrtStiffnessVectorCoeffType.clear();
      d_stiffnessVectorCoeffType.clear();
      d_inverseStiffnessVectorCoeffType.clear();

      d_quadratureIDsVector.clear();
      d_nQuadsPerCell.clear();
      d_updateFlags.clear();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
      FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
        init(dealii::MatrixFree<3, ValueTypeBasisData> &matrixFreeData,
             std::vector<const dealii::AffineConstraints<ValueTypeBasisData> *>
               &                              constraintsVector,
             const unsigned int &             dofHandlerID,
             const std::vector<unsigned int> &quadratureID,
             const std::vector<UpdateFlags>   updateFlags)
    {
      d_matrixFreeDataPtr = &matrixFreeData;
      d_constraintsVector = &constraintsVector;
      d_dofHandlerID      = 0;
      d_nVectors          = 0;
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
      initializeConstraints();
      AssertThrow(
        updateFlags.size() == quadratureID.size(),
        dealii::ExcMessage(
          "DFT-FE Error: Inconsistent size of update flags for FEBasisOperations class."));

      d_dofHandlerID        = dofHandlerID;
      d_quadratureIDsVector = quadratureID;
      d_updateFlags         = updateFlags;
      initializeIndexMaps();
      initializeMPIPattern();
      initializeShapeFunctionAndJacobianData();
      if constexpr (!std::is_same<ValueTypeBasisCoeff,
                                  ValueTypeBasisData>::value)
        initializeShapeFunctionAndJacobianBasisData();
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    template <dftfe::utils::MemorySpace memorySpaceSrc>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      init(const FEBasisOperations<ValueTypeBasisCoeff,
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
      d_nQuadsPerCell.resize(d_quadratureIDsVector.size());
      d_quadPoints = basisOperationsSrc.d_quadPoints;
      initializeConstraints();
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
      if constexpr (!std::is_same<ValueTypeBasisCoeff,
                                  ValueTypeBasisData>::value)
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
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      reinit(const unsigned int &vecBlockSize,
             const unsigned int &cellsBlockSize,
             const unsigned int &quadratureID,
             const bool          isResizeTempStorageForInterpolation,
             const bool          isResizeTempStorageForCellMatrices)
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
      resizeTempStorage(isResizeTempStorageForInterpolation,
                        isResizeTempStorageForCellMatrices);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    unsigned int
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      nQuadsPerCell() const
    {
      return d_nQuadsPerCell[d_quadratureIndex];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    unsigned int
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      nVectors() const
    {
      return d_nVectors;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    unsigned int
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      nDofsPerCell() const
    {
      return d_nDofsPerCell;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    unsigned int
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      nCells() const
    {
      return d_nCells;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    unsigned int
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      nRelaventDofs() const
    {
      return d_localSize;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    unsigned int
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      nOwnedDofs() const
    {
      return d_locallyOwnedSize;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      shapeFunctionData(bool transpose) const
    {
      return transpose ?
               d_shapeFunctionDataTranspose.find(d_quadratureID)->second :
               d_shapeFunctionData.find(d_quadratureID)->second;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      shapeFunctionGradientData(bool transpose) const
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
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      inverseJacobians() const
    {
      return d_inverseJacobianData.find(areAllCellsAffine ? 0 : d_quadratureID)
        ->second;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData,
                                      dftfe::utils::MemorySpace::HOST> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      quadPoints() const
    {
      return d_quadPoints.find(d_quadratureID)->second;
    }



    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff, memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      JxW() const
    {
      return d_JxWData.find(d_quadratureID)->second;
    }



    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      cellStiffnessMatrixBasisData() const
    {
      //      std::cout<<" size of stiffness vec =
      //      "<<d_cellStiffnessMatrixBasisType.size()<<"\n";
      return d_cellStiffnessMatrixBasisType;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      cellMassMatrixBasisData() const
    {
      return d_cellMassMatrixBasisType;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      cellInverseSqrtMassVectorBasisData() const
    {
      return d_cellInverseSqrtMassVectorBasisType;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      stiffnessVectorBasisData() const
    {
      return d_stiffnessVectorBasisType;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      inverseStiffnessVectorBasisData() const
    {
      return d_inverseStiffnessVectorBasisType;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      inverseSqrtStiffnessVectorBasisData() const
    {
      return d_inverseSqrtStiffnessVectorBasisType;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      sqrtStiffnessVectorBasisData() const
    {
      return d_sqrtStiffnessVectorBasisType;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      cellInverseMassVectorBasisData() const
    {
      return d_cellInverseMassVectorBasisType;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      cellSqrtMassVectorBasisData() const
    {
      return d_cellSqrtMassVectorBasisType;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      cellMassVectorBasisData() const
    {
      return d_cellMassVectorBasisType;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      inverseSqrtMassVectorBasisData() const
    {
      return d_inverseSqrtMassVectorBasisType;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      sqrtMassVectorBasisData() const
    {
      return d_sqrtMassVectorBasisType;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      inverseMassVectorBasisData() const
    {
      return d_inverseMassVectorBasisType;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      massVectorBasisData() const
    {
      return d_massVectorBasisType;
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    unsigned int
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      cellsTypeFlag() const
    {
      return (unsigned int)areAllCellsAffine +
             (unsigned int)areAllCellsCartesian;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    dealii::CellId
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      cellID(const unsigned int iElem) const
    {
      return d_cellIndexToCellIdMap[iElem];
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    unsigned int
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      cellIndex(const dealii::CellId cellid) const
    {
      return d_cellIdToCellIndexMap.find(cellid)->second;
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dealii::MatrixFree<3, ValueTypeBasisData> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      matrixFreeData() const
    {
      return *d_matrixFreeDataPtr;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    const dealii::DoFHandler<3> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      getDofHandler() const
    {
      return d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID);
    }



    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      resizeTempStorage(const bool isResizeTempStorageForInterpolation,
                        const bool isResizeTempStorageForCellMatrices)
    {
      if (isResizeTempStorageForInterpolation)
        {
          tempCellNodalData.resize(d_nVectors * d_nDofsPerCell *
                                   d_cellsBlockSize);
          if (d_updateFlags[d_quadratureIndex] & update_gradients)
            tempQuadratureGradientsData.resize(
              areAllCellsCartesian ?
                0 :
                (d_nVectors * d_nQuadsPerCell[d_quadratureIndex] * 3 *
                 d_cellsBlockSize));

          if (d_updateFlags[d_quadratureIndex] & update_gradients)
            tempQuadratureGradientsDataNonAffine.resize(
              areAllCellsAffine ?
                0 :
                (d_nVectors * d_nQuadsPerCell[d_quadratureIndex] * 3 *
                 d_cellsBlockSize));
        }
      if (isResizeTempStorageForCellMatrices)
        {
          if (tempCellMatrixBlock.size() !=
              d_nDofsPerCell * d_nDofsPerCell * d_cellsBlockSize)
            tempCellMatrixBlock.resize(d_nDofsPerCell * d_nDofsPerCell *
                                       d_cellsBlockSize);
          if (tempCellValuesBlock.size() != d_nQuadsPerCell[d_quadratureIndex] *
                                              d_nDofsPerCell * d_cellsBlockSize)
            tempCellValuesBlock.resize(d_nQuadsPerCell[d_quadratureIndex] *
                                       d_nDofsPerCell * d_cellsBlockSize);

          if (tempCellValuesBlockCoeff.size() !=
              d_nQuadsPerCell[d_quadratureIndex] * d_nDofsPerCell *
                d_cellsBlockSize)
            {
              tempCellValuesBlockCoeff.resize(
                d_nQuadsPerCell[d_quadratureIndex] * d_nDofsPerCell *
                d_cellsBlockSize);
            }
          if (tempCellGradientsBlock.size() !=
              d_nQuadsPerCell[d_quadratureIndex] * d_nDofsPerCell *
                d_cellsBlockSize * 3)
            tempCellGradientsBlock.resize(d_nQuadsPerCell[d_quadratureIndex] *
                                          d_nDofsPerCell * d_cellsBlockSize *
                                          3);
          if (tempCellGradientsBlock2.size() !=
              d_nQuadsPerCell[d_quadratureIndex] * d_nDofsPerCell *
                d_cellsBlockSize * 3)
            tempCellGradientsBlock2.resize(d_nQuadsPerCell[d_quadratureIndex] *
                                           d_nDofsPerCell * d_cellsBlockSize *
                                           3);
          if (zeroIndexVec.size() != d_cellsBlockSize)
            zeroIndexVec.resize(d_cellsBlockSize, 0);
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      initializeFlattenedIndexMaps()
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
    dftfe::utils::MemoryStorage<dftfe::global_size_type,
                                dftfe::utils::MemorySpace::HOST> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      getFlattenedMapsHost()
    {
      return d_cellDofIndexToProcessDofIndexMap;
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      initializeMPIPattern()
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
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      initializeIndexMaps()
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
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      reinitializeConstraints(
        std::vector<const dealii::AffineConstraints<ValueTypeBasisData> *>
          &constraintsVector)
    {
      d_constraintsVector = &constraintsVector;
      initializeConstraints();
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      initializeConstraints()
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
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      initializeShapeFunctionAndJacobianData()
    {
      d_nQuadsPerCell.resize(d_quadratureIDsVector.size());
      for (unsigned int iQuadIndex = 0;
           iQuadIndex < d_quadratureIDsVector.size();
           ++iQuadIndex)
        {
          unsigned int quadID = d_quadratureIDsVector[iQuadIndex];
          const dealii::Quadrature<3> &quadrature =
            d_matrixFreeDataPtr->get_quadrature(quadID);
          auto dealiiUpdateFlags = dealii::update_default;
          if (d_updateFlags[iQuadIndex] & update_jxw)
            dealiiUpdateFlags = dealiiUpdateFlags | dealii::update_JxW_values;
          if (d_updateFlags[iQuadIndex] & update_inversejacobians)
            dealiiUpdateFlags =
              dealiiUpdateFlags | dealii::update_inverse_jacobians;
          if (d_updateFlags[iQuadIndex] & update_quadpoints)
            dealiiUpdateFlags =
              dealiiUpdateFlags | dealii::update_quadrature_points;
          dealii::FEValues<3> fe_values(
            d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).get_fe(),
            quadrature,
            dealiiUpdateFlags);
          dealiiUpdateFlags = dealii::update_default;
          if (d_updateFlags[iQuadIndex] & update_values)
            dealiiUpdateFlags = dealiiUpdateFlags | dealii::update_values;
          if (d_updateFlags[iQuadIndex] & update_gradients)
            dealiiUpdateFlags = dealiiUpdateFlags | dealii::update_gradients;
          dealii::FEValues<3> fe_values_reference(
            d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).get_fe(),
            quadrature,
            dealiiUpdateFlags);
          if ((d_updateFlags[iQuadIndex] & update_values) |
              (d_updateFlags[iQuadIndex] & update_gradients))
            {
              dealii::Triangulation<3> reference_cell;
              dealii::GridGenerator::hyper_cube(reference_cell, 0., 1.);
              fe_values_reference.reinit(reference_cell.begin());
            }
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

          if (!areAllCellsAffine)
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
          if (d_updateFlags[iQuadIndex] & update_inversejacobians)
            {
              d_inverseJacobianData[areAllCellsAffine ? 0 : quadID].resize(
                d_inverseJacobianDataHost.size());
              d_inverseJacobianData[areAllCellsAffine ? 0 : quadID].copyFrom(
                d_inverseJacobianDataHost);
            }
          if (d_updateFlags[iQuadIndex] & update_jxw)
            {
              d_JxWData[quadID].resize(d_JxWDataHost.size());
              d_JxWData[quadID].copyFrom(d_JxWDataHost);
            }
          if (d_updateFlags[iQuadIndex] & update_values)
            {
              d_shapeFunctionData[quadID].resize(
                d_shapeFunctionDataHost.size());
              d_shapeFunctionData[quadID].copyFrom(d_shapeFunctionDataHost);
              if (d_updateFlags[iQuadIndex] & update_transpose)
                {
                  d_shapeFunctionDataTranspose[quadID].resize(
                    d_shapeFunctionDataTransposeHost.size());
                  d_shapeFunctionDataTranspose[quadID].copyFrom(
                    d_shapeFunctionDataTransposeHost);
                }
            }
          if (d_updateFlags[iQuadIndex] & update_gradients)
            {
              d_shapeFunctionGradientDataInternalLayout[quadID].resize(
                d_shapeFunctionGradientDataInternalLayoutHost.size());
              d_shapeFunctionGradientDataInternalLayout[quadID].copyFrom(
                d_shapeFunctionGradientDataInternalLayoutHost);
              d_shapeFunctionGradientData[quadID].resize(
                d_shapeFunctionGradientDataHost.size());
              d_shapeFunctionGradientData[quadID].copyFrom(
                d_shapeFunctionGradientDataHost);
              if (d_updateFlags[iQuadIndex] & update_transpose)
                {
                  d_shapeFunctionGradientDataTranspose[quadID].resize(
                    d_shapeFunctionGradientDataTransposeHost.size());
                  d_shapeFunctionGradientDataTranspose[quadID].copyFrom(
                    d_shapeFunctionGradientDataTransposeHost);
                }
            }
#endif
          // std::cout<<" quad id = "<<quadID<<" size of sha = "<<
          //  d_shapeFunctionData.find(quadID)->second.size()
          //  <<" size of tra =
          //  "<<d_shapeFunctionDataTranspose.find(quadID)->second.size()<<"\n";
        }
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      initializeShapeFunctionAndJacobianBasisData()
    {
      for (unsigned int iQuadIndex = 0;
           iQuadIndex < d_quadratureIDsVector.size();
           ++iQuadIndex)
        {
          unsigned int quadID = d_quadratureIDsVector[iQuadIndex];
          const dealii::Quadrature<3> &quadrature =
            d_matrixFreeDataPtr->get_quadrature(quadID);
          auto dealiiUpdateFlags = dealii::update_default;
          if (d_updateFlags[iQuadIndex] & update_jxw)
            dealiiUpdateFlags = dealiiUpdateFlags | dealii::update_JxW_values;
          if (d_updateFlags[iQuadIndex] & update_inversejacobians)
            dealiiUpdateFlags =
              dealiiUpdateFlags | dealii::update_inverse_jacobians;
          dealii::FEValues<3> fe_values(
            d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).get_fe(),
            quadrature,
            dealiiUpdateFlags);
          dealiiUpdateFlags = dealii::update_default;
          if (d_updateFlags[iQuadIndex] & update_values)
            dealiiUpdateFlags = dealiiUpdateFlags | dealii::update_values;
          if (d_updateFlags[iQuadIndex] & update_gradients)
            dealiiUpdateFlags = dealiiUpdateFlags | dealii::update_gradients;
          dealii::FEValues<3> fe_values_reference(
            d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).get_fe(),
            quadrature,
            dealiiUpdateFlags);
          if ((d_updateFlags[iQuadIndex] & update_values) |
              (d_updateFlags[iQuadIndex] & update_gradients))
            {
              dealii::Triangulation<3> reference_cell;
              dealii::GridGenerator::hyper_cube(reference_cell, 0., 1.);
              fe_values_reference.reinit(reference_cell.begin());
            }
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

          if (!areAllCellsAffine)
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
          if (d_updateFlags[iQuadIndex] & update_inversejacobians)
            {
              d_inverseJacobianBasisData[areAllCellsAffine ? 0 : quadID].resize(
                d_inverseJacobianDataHost.size());
              d_inverseJacobianBasisData[areAllCellsAffine ? 0 : quadID]
                .copyFrom(d_inverseJacobianDataHost);
            }
          if (d_updateFlags[iQuadIndex] & update_jxw)
            {
              d_JxWBasisData[quadID].resize(d_JxWDataHost.size());
              d_JxWBasisData[quadID].copyFrom(d_JxWDataHost);
            }
          if (d_updateFlags[iQuadIndex] & update_values)
            {
              d_shapeFunctionBasisData[quadID].resize(
                d_shapeFunctionDataHost.size());
              d_shapeFunctionBasisData[quadID].copyFrom(
                d_shapeFunctionDataHost);
              if (d_updateFlags[iQuadIndex] & update_transpose)
                {
                  d_shapeFunctionBasisDataTranspose[quadID].resize(
                    d_shapeFunctionDataTransposeHost.size());
                  d_shapeFunctionBasisDataTranspose[quadID].copyFrom(
                    d_shapeFunctionDataTransposeHost);
                }
            }
          if (d_updateFlags[iQuadIndex] & update_gradients)
            {
              d_shapeFunctionGradientBasisData[quadID].resize(
                d_shapeFunctionGradientDataHost.size());
              d_shapeFunctionGradientBasisData[quadID].copyFrom(
                d_shapeFunctionGradientDataHost);
              if (d_updateFlags[iQuadIndex] & update_transpose)
                {
                  d_shapeFunctionGradientBasisDataTranspose[quadID].resize(
                    d_shapeFunctionGradientDataTransposeHost.size());
                  d_shapeFunctionGradientBasisDataTranspose[quadID].copyFrom(
                    d_shapeFunctionGradientDataTransposeHost);
                }
            }
#endif
        }
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      computeCellStiffnessMatrix(const unsigned int quadratureID,
                                 const unsigned int cellsBlockSize,
                                 const bool         basisType,
                                 const bool         ceoffType)
    {
      reinit(1, cellsBlockSize, quadratureID, false, true);
      if (basisType)
        d_cellStiffnessMatrixBasisType.resize(d_nDofsPerCell * d_nDofsPerCell *
                                              d_nCells);
      if (ceoffType)
        if constexpr (std::is_same<ValueTypeBasisCoeff,
                                   ValueTypeBasisData>::value)
          {
            if (!basisType)
              d_cellStiffnessMatrixBasisType.resize(d_nDofsPerCell *
                                                    d_nDofsPerCell * d_nCells);
          }
        else
          d_cellStiffnessMatrixCoeffType.resize(d_nDofsPerCell *
                                                d_nDofsPerCell * d_nCells);

      unsigned int nQuadsPerCell = this->nQuadsPerCell();
      dftfe::utils::MemoryStorage<ValueTypeBasisData,
                                  dftfe::utils::MemorySpace::HOST>
        d_jacobianFactorHost;

#if defined(DFTFE_WITH_DEVICE)
      dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
        d_jacobianFactor;
#else
      auto &d_jacobianFactor = d_jacobianFactorHost;
#endif
      d_jacobianFactorHost.resize(9 * nQuadsPerCell * d_nCells);

      const dealii::Quadrature<3> &quadrature =
        d_matrixFreeDataPtr->get_quadrature(quadratureID);
      dealii::FEValues<3> fe_values(
        d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).get_fe(),
        quadrature,
        dealii::update_JxW_values | dealii::update_inverse_jacobians);
      auto cellPtr =
        d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).begin_active();
      auto endcPtr = d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).end();
      for (unsigned int iCell = 0; cellPtr != endcPtr; ++cellPtr)
        if (cellPtr->is_locally_owned())
          {
            fe_values.reinit(cellPtr);
            const auto &inverseJacobians = fe_values.get_inverse_jacobians();
            for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
              {
                const auto &inverseJacobianQuad = inverseJacobians[iQuad];
                const auto  jxw                 = fe_values.JxW(iQuad);
                const auto  jacobianFactorPtr   = d_jacobianFactorHost.data() +
                                               iCell * nQuadsPerCell * 9 +
                                               iQuad * 9;
                for (unsigned int jDim = 0; jDim < 3; ++jDim)
                  for (unsigned int iDim = 0; iDim < 3; ++iDim)
                    for (unsigned int kDim = 0; kDim < 3; ++kDim)
                      jacobianFactorPtr[3 * jDim + iDim] +=
                        inverseJacobianQuad[iDim][kDim] *
                        inverseJacobianQuad[jDim][kDim] * jxw;
              }
            ++iCell;
          }
#if defined(DFTFE_WITH_DEVICE)
      d_jacobianFactor.resize(d_jacobianFactorHost.size());
      d_jacobianFactor.copyFrom(d_jacobianFactorHost);
#endif
      if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
        {
          dftfe::basis::FEBasisOperationsKernelsInternal::
            reshapeToNonAffineLayoutHost(
              d_nDofsPerCell,
              nQuadsPerCell,
              1,
              shapeFunctionGradientBasisData().data(),
              tempCellGradientsBlock.data());
        }
      else
        {
          dftfe::basis::FEBasisOperationsKernelsInternal::
            reshapeToNonAffineLayoutDevice(
              d_nDofsPerCell,
              nQuadsPerCell,
              1,
              shapeFunctionGradientBasisData().data(),
              tempCellGradientsBlock.data());
        }
      if (cellsBlockSize > 1)
        d_BLASWrapperPtr->stridedCopyToBlock(nQuadsPerCell * d_nDofsPerCell * 3,
                                             cellsBlockSize - 1,
                                             tempCellGradientsBlock.data(),
                                             tempCellGradientsBlock.data() +
                                               nQuadsPerCell * d_nDofsPerCell *
                                                 3,
                                             zeroIndexVec.data());
      const ValueTypeBasisData scalarCoeffAlpha = ValueTypeBasisData(1.0),
                               scalarCoeffBeta  = ValueTypeBasisData(0.0);

      for (unsigned int iCell = 0; iCell < d_nCells; iCell += cellsBlockSize)
        {
          std::pair<unsigned int, unsigned int> cellRange(
            iCell, std::min(iCell + cellsBlockSize, d_nCells));
          d_BLASWrapperPtr->xgemmStridedBatched(
            'N',
            'N',
            d_nDofsPerCell,
            3,
            3,
            &scalarCoeffAlpha,
            tempCellGradientsBlock.data(),
            d_nDofsPerCell,
            d_nDofsPerCell * 3,
            d_jacobianFactor.data() + 9 * cellRange.first * nQuadsPerCell,
            3,
            9,
            &scalarCoeffBeta,
            tempCellGradientsBlock2.data(),
            d_nDofsPerCell,
            d_nDofsPerCell * 3,
            (cellRange.second - cellRange.first) * nQuadsPerCell);
          d_BLASWrapperPtr->xgemmStridedBatched('N',
                                                'T',
                                                d_nDofsPerCell,
                                                d_nDofsPerCell,
                                                nQuadsPerCell * 3,
                                                &scalarCoeffAlpha,
                                                tempCellGradientsBlock2.data(),
                                                d_nDofsPerCell,
                                                d_nDofsPerCell * nQuadsPerCell *
                                                  3,
                                                tempCellGradientsBlock.data(),
                                                d_nDofsPerCell,
                                                0,
                                                &scalarCoeffBeta,
                                                tempCellMatrixBlock.data(),
                                                d_nDofsPerCell,
                                                d_nDofsPerCell * d_nDofsPerCell,
                                                cellRange.second -
                                                  cellRange.first);
          if (basisType)
            d_cellStiffnessMatrixBasisType.copyFrom(
              tempCellMatrixBlock,
              d_nDofsPerCell * d_nDofsPerCell *
                (cellRange.second - cellRange.first),
              0,
              cellRange.first * d_nDofsPerCell * d_nDofsPerCell);
          if (ceoffType)
            if constexpr (std::is_same<ValueTypeBasisCoeff,
                                       ValueTypeBasisData>::value)
              {
                if (!basisType)
                  d_cellStiffnessMatrixBasisType.copyFrom(
                    tempCellMatrixBlock,
                    d_nDofsPerCell * d_nDofsPerCell *
                      (cellRange.second - cellRange.first),
                    0,
                    cellRange.first * d_nDofsPerCell * d_nDofsPerCell);
              }
            else
              d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                d_nDofsPerCell * d_nDofsPerCell *
                  (cellRange.second - cellRange.first),
                tempCellMatrixBlock.data(),
                d_cellStiffnessMatrixCoeffType.data() +
                  cellRange.first * d_nDofsPerCell * d_nDofsPerCell);
        }
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      computeWeightedCellMassMatrix(
        const std::pair<unsigned int, unsigned int> cellRangeTotal,
        dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &weights,
        dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
          &weightedCellMassMatrix) const
    {
      const unsigned int nCells        = this->nCells();
      const unsigned int nQuadsPerCell = this->nQuadsPerCell();
      const unsigned int nDofsPerCell  = this->nDofsPerCell();

      const double scalarCoeffAlpha = 1.0, scalarCoeffBeta = 0.0;

      for (unsigned int iCell = cellRangeTotal.first;
           iCell < cellRangeTotal.second;
           iCell += d_cellsBlockSize)
        {
          std::pair<unsigned int, unsigned int> cellRange(
            iCell, std::min(iCell + d_cellsBlockSize, cellRangeTotal.second));
          d_BLASWrapperPtr->stridedCopyToBlock(nQuadsPerCell * nDofsPerCell,
                                               (cellRange.second -
                                                cellRange.first),
                                               shapeFunctionBasisData().data(),
                                               tempCellValuesBlock.data(),
                                               zeroIndexVec.data());
          d_BLASWrapperPtr->stridedBlockScale(
            nDofsPerCell,
            nQuadsPerCell * (cellRange.second - cellRange.first),
            scalarCoeffAlpha,
            weights.data() + cellRange.first * nQuadsPerCell,
            tempCellValuesBlock.data());
          d_BLASWrapperPtr->xgemmStridedBatched(
            'N',
            'T',
            nDofsPerCell,
            nDofsPerCell,
            nQuadsPerCell,
            &scalarCoeffAlpha,
            tempCellValuesBlock.data(),
            nDofsPerCell,
            nDofsPerCell * nQuadsPerCell,
            shapeFunctionBasisData().data(),
            nDofsPerCell,
            0,
            &scalarCoeffAlpha,
            weightedCellMassMatrix.data() +
              (cellRange.first - cellRangeTotal.first) * nDofsPerCell *
                nDofsPerCell,

            nDofsPerCell,
            nDofsPerCell * nDofsPerCell,
            cellRange.second - cellRange.first);
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      computeWeightedCellNjGradNiMatrix(
        const std::pair<unsigned int, unsigned int> cellRangeTotal,
        dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &weights,
        dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
          &weightedCellNjGradNiMatrix) const
    {
      const unsigned int nCells        = this->nCells();
      const unsigned int nQuadsPerCell = this->nQuadsPerCell();
      const unsigned int nDofsPerCell  = this->nDofsPerCell();

      if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
        {
          dftfe::basis::FEBasisOperationsKernelsInternal::
            reshapeToNonAffineLayoutHost(
              nDofsPerCell,
              nQuadsPerCell,
              1,
              shapeFunctionGradientBasisData().data(),
              tempCellGradientsBlock.data());
        }
      else
        {
          dftfe::basis::FEBasisOperationsKernelsInternal::
            reshapeToNonAffineLayoutDevice(
              nDofsPerCell,
              nQuadsPerCell,
              1,
              shapeFunctionGradientBasisData().data(),
              tempCellGradientsBlock.data());
        }
      if (d_cellsBlockSize > 1)
        d_BLASWrapperPtr->stridedCopyToBlock(nQuadsPerCell * nDofsPerCell * 3,
                                             d_cellsBlockSize - 1,
                                             tempCellGradientsBlock.data(),
                                             tempCellGradientsBlock.data() +
                                               nQuadsPerCell * nDofsPerCell * 3,
                                             zeroIndexVec.data());
      const ValueTypeBasisData scalarCoeffAlpha = ValueTypeBasisData(1.0),
                               scalarCoeffBeta  = ValueTypeBasisData(0.0);

      for (unsigned int iCell = cellRangeTotal.first;
           iCell < cellRangeTotal.second;
           iCell += d_cellsBlockSize)
        {
          std::pair<unsigned int, unsigned int> cellRange(
            iCell, std::min(iCell + d_cellsBlockSize, cellRangeTotal.second));
          d_BLASWrapperPtr->xgemmStridedBatched(
            'N',
            'N',
            d_nDofsPerCell,
            1,
            3,
            &scalarCoeffAlpha,
            tempCellGradientsBlock.data(),
            d_nDofsPerCell,
            d_nDofsPerCell * 3,
            weights.data() + 3 * cellRange.first * nQuadsPerCell,
            3,
            3,
            &scalarCoeffBeta,
            tempCellValuesBlock.data(),
            d_nDofsPerCell,
            d_nDofsPerCell,
            (cellRange.second - cellRange.first) * nQuadsPerCell);
          d_BLASWrapperPtr->xgemmStridedBatched(
            'N',
            'T',
            nDofsPerCell,
            nDofsPerCell,
            nQuadsPerCell,
            &scalarCoeffAlpha,
            tempCellValuesBlock.data(),
            nDofsPerCell,
            nDofsPerCell * nQuadsPerCell,
            shapeFunctionBasisData().data(),
            nDofsPerCell,
            0,
            &scalarCoeffAlpha,
            weightedCellNjGradNiMatrix.data() +
              (cellRange.first - cellRangeTotal.first) * nDofsPerCell *
                nDofsPerCell,
            nDofsPerCell,
            nDofsPerCell * nDofsPerCell,
            cellRange.second - cellRange.first);
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      computeWeightedCellNjGradNiPlusNiGradNjMatrix(
        const std::pair<unsigned int, unsigned int> cellRangeTotal,
        dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace> &weights,
        dftfe::utils::MemoryStorage<ValueTypeBasisData, memorySpace>
          &weightedCellNjGradNiPlusNiGradNjMatrix) const
    {
      const unsigned int nCells        = this->nCells();
      const unsigned int nQuadsPerCell = this->nQuadsPerCell();
      const unsigned int nDofsPerCell  = this->nDofsPerCell();

      if constexpr (memorySpace == dftfe::utils::MemorySpace::HOST)
        {
          dftfe::basis::FEBasisOperationsKernelsInternal::
            reshapeToNonAffineLayoutHost(
              nDofsPerCell,
              nQuadsPerCell,
              1,
              shapeFunctionGradientBasisData().data(),
              tempCellGradientsBlock.data());
        }
      else
        {
          dftfe::basis::FEBasisOperationsKernelsInternal::
            reshapeToNonAffineLayoutDevice(
              nDofsPerCell,
              nQuadsPerCell,
              1,
              shapeFunctionGradientBasisData().data(),
              tempCellGradientsBlock.data());
        }
      if (d_cellsBlockSize > 1)
        d_BLASWrapperPtr->stridedCopyToBlock(nQuadsPerCell * nDofsPerCell * 3,
                                             d_cellsBlockSize - 1,
                                             tempCellGradientsBlock.data(),
                                             tempCellGradientsBlock.data() +
                                               nQuadsPerCell * nDofsPerCell * 3,
                                             zeroIndexVec.data());
      const ValueTypeBasisData scalarCoeffAlpha = ValueTypeBasisData(1.0),
                               scalarCoeffBeta  = ValueTypeBasisData(0.0);

      for (unsigned int iCell = cellRangeTotal.first;
           iCell < cellRangeTotal.second;
           iCell += d_cellsBlockSize)
        {
          std::pair<unsigned int, unsigned int> cellRange(
            iCell, std::min(iCell + d_cellsBlockSize, cellRangeTotal.second));
          d_BLASWrapperPtr->xgemmStridedBatched(
            'N',
            'N',
            d_nDofsPerCell,
            1,
            3,
            &scalarCoeffAlpha,
            tempCellGradientsBlock.data(),
            d_nDofsPerCell,
            d_nDofsPerCell * 3,
            weights.data() + 3 * cellRange.first * nQuadsPerCell,
            3,
            3,
            &scalarCoeffBeta,
            tempCellValuesBlock.data(),
            d_nDofsPerCell,
            d_nDofsPerCell,
            (cellRange.second - cellRange.first) * nQuadsPerCell);
          d_BLASWrapperPtr->xgemmStridedBatched(
            'N',
            'T',
            nDofsPerCell,
            nDofsPerCell,
            nQuadsPerCell,
            &scalarCoeffAlpha,
            tempCellValuesBlock.data(),
            nDofsPerCell,
            nDofsPerCell * nQuadsPerCell,
            shapeFunctionBasisData().data(),
            nDofsPerCell,
            0,
            &scalarCoeffAlpha,
            weightedCellNjGradNiPlusNiGradNjMatrix.data() +
              (cellRange.first - cellRangeTotal.first) * nDofsPerCell *
                nDofsPerCell,
            nDofsPerCell,
            nDofsPerCell * nDofsPerCell,
            cellRange.second - cellRange.first);
          // FIXME : Can be optimized further, this is just the transpose of the
          // earlier gemm
          d_BLASWrapperPtr->xgemmStridedBatched(
            'N',
            'T',
            nDofsPerCell,
            nDofsPerCell,
            nQuadsPerCell,
            &scalarCoeffAlpha,
            shapeFunctionBasisData().data(),
            nDofsPerCell,
            0,
            tempCellValuesBlock.data(),
            nDofsPerCell,
            nDofsPerCell * nQuadsPerCell,
            &scalarCoeffAlpha,
            weightedCellNjGradNiPlusNiGradNjMatrix.data() +
              (cellRange.first - cellRangeTotal.first) * nDofsPerCell *
                nDofsPerCell,
            nDofsPerCell,
            nDofsPerCell * nDofsPerCell,
            cellRange.second - cellRange.first);
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      computeCellMassMatrix(const unsigned int quadratureID,
                            const unsigned int cellsBlockSize,
                            const bool         basisType,
                            const bool         ceoffType)
    {
      reinit(0, cellsBlockSize, quadratureID, false, true);
      if (basisType)
        d_cellMassMatrixBasisType.resize(d_nDofsPerCell * d_nDofsPerCell *
                                         d_nCells);
      if (ceoffType)
        if constexpr (std::is_same<ValueTypeBasisCoeff,
                                   ValueTypeBasisData>::value)
          {
            if (!basisType)
              d_cellMassMatrixBasisType.resize(d_nDofsPerCell * d_nDofsPerCell *
                                               d_nCells);
          }
        else
          d_cellMassMatrixCoeffType.resize(d_nDofsPerCell * d_nDofsPerCell *
                                           d_nCells);

      unsigned int nQuadsPerCell = this->nQuadsPerCell();


      const ValueTypeBasisData scalarCoeffAlpha = ValueTypeBasisData(1.0),
                               scalarCoeffBeta  = ValueTypeBasisData(0.0);

      for (unsigned int iCell = 0; iCell < d_nCells; iCell += cellsBlockSize)
        {
          std::pair<unsigned int, unsigned int> cellRange(
            iCell, std::min(iCell + cellsBlockSize, d_nCells));
          d_BLASWrapperPtr->stridedCopyToBlock(nQuadsPerCell * d_nDofsPerCell,
                                               (cellRange.second -
                                                cellRange.first),
                                               shapeFunctionBasisData().data(),
                                               tempCellValuesBlock.data(),
                                               zeroIndexVec.data());
          d_BLASWrapperPtr->stridedBlockScale(
            d_nDofsPerCell,
            nQuadsPerCell * (cellRange.second - cellRange.first),
            scalarCoeffAlpha,
            JxWBasisData().data() + cellRange.first * nQuadsPerCell,
            tempCellValuesBlock.data());
          d_BLASWrapperPtr->xgemmStridedBatched('N',
                                                'T',
                                                d_nDofsPerCell,
                                                d_nDofsPerCell,
                                                nQuadsPerCell,
                                                &scalarCoeffAlpha,
                                                tempCellValuesBlock.data(),
                                                d_nDofsPerCell,
                                                d_nDofsPerCell * nQuadsPerCell,
                                                shapeFunctionBasisData().data(),
                                                d_nDofsPerCell,
                                                0,
                                                &scalarCoeffBeta,
                                                tempCellMatrixBlock.data(),
                                                d_nDofsPerCell,
                                                d_nDofsPerCell * d_nDofsPerCell,
                                                cellRange.second -
                                                  cellRange.first);

          if (basisType)
            d_cellMassMatrixBasisType.copyFrom(
              tempCellMatrixBlock,
              d_nDofsPerCell * d_nDofsPerCell *
                (cellRange.second - cellRange.first),
              0,
              cellRange.first * d_nDofsPerCell * d_nDofsPerCell);
          if (ceoffType)
            if constexpr (std::is_same<ValueTypeBasisCoeff,
                                       ValueTypeBasisData>::value)
              {
                if (!basisType)
                  d_cellMassMatrixBasisType.copyFrom(
                    tempCellMatrixBlock,
                    d_nDofsPerCell * d_nDofsPerCell *
                      (cellRange.second - cellRange.first),
                    0,
                    cellRange.first * d_nDofsPerCell * d_nDofsPerCell);
              }
            else
              d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                d_nDofsPerCell * d_nDofsPerCell *
                  (cellRange.second - cellRange.first),
                tempCellMatrixBlock.data(),
                d_cellMassMatrixCoeffType.data() +
                  cellRange.first * d_nDofsPerCell * d_nDofsPerCell);
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      computeInverseSqrtMassVector(const bool basisType, const bool ceoffType)
    {
      distributedCPUVec<double> massVector, sqrtMassVector, invMassVector,
        invSqrtMassVector;
      d_matrixFreeDataPtr->initialize_dof_vector(massVector, d_dofHandlerID);
      sqrtMassVector.reinit(massVector);
      invMassVector.reinit(massVector);
      invSqrtMassVector.reinit(massVector);
      massVector        = 0.0;
      sqrtMassVector    = 0.0;
      invMassVector     = 0.0;
      invSqrtMassVector = 0.0;

      // FIXME : check for roundoff errors
      dealii::QGaussLobatto<3> quadrature(std::cbrt(d_nDofsPerCell));
      unsigned int             nQuadsPerCell = quadrature.size();
      dealii::FEValues<3>      fe_values(
        d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).get_fe(),
        quadrature,
        dealii::update_values | dealii::update_JxW_values);

      dealii::Vector<double> massVectorLocal(d_nDofsPerCell);
      std::vector<dealii::types::global_dof_index> local_dof_indices(
        d_nDofsPerCell);


      //
      // parallel loop over all elements
      //
      typename dealii::DoFHandler<3>::active_cell_iterator
        cell =
          d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).begin_active(),
        endc = d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).end();
      for (; cell != endc; ++cell)
        if (cell->is_locally_owned())
          {
            // compute values for the current element
            fe_values.reinit(cell);
            massVectorLocal = 0.0;
            for (unsigned int iDoF = 0; iDoF < d_nDofsPerCell; ++iDoF)
              for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                massVectorLocal(iDoF) += fe_values.shape_value(iDoF, iQuad) *
                                         fe_values.shape_value(iDoF, iQuad) *
                                         fe_values.JxW(iQuad);

            cell->get_dof_indices(local_dof_indices);
            (*d_constraintsVector)[d_dofHandlerID]->distribute_local_to_global(
              massVectorLocal, local_dof_indices, massVector);
          }

      massVector.compress(dealii::VectorOperation::add);
      massVector.update_ghost_values();


      for (dealii::types::global_dof_index i = 0; i < massVector.size(); ++i)
        if (massVector.in_local_range(i) &&
            !((*d_constraintsVector)[d_dofHandlerID]->is_constrained(i)))
          {
            sqrtMassVector(i) = std::sqrt(massVector(i));
            if (std::abs(massVector(i)) > 1.0e-15)
              {
                invSqrtMassVector(i) = 1.0 / std::sqrt(massVector(i));
                invMassVector(i)     = 1.0 / massVector(i);
              }

            AssertThrow(
              !std::isnan(invMassVector(i)),
              dealii::ExcMessage(
                "Value of inverse square root of mass matrix on the unconstrained node is undefined"));
          }

      invMassVector.compress(dealii::VectorOperation::insert);
      invMassVector.update_ghost_values();
      sqrtMassVector.compress(dealii::VectorOperation::insert);
      sqrtMassVector.update_ghost_values();
      invSqrtMassVector.compress(dealii::VectorOperation::insert);
      invSqrtMassVector.update_ghost_values();

      cell =
        d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).begin_active();
      std::vector<dealii::types::global_dof_index> cell_dof_indices(
        d_nDofsPerCell);

      dftfe::utils::MemoryStorage<ValueTypeBasisData,
                                  dftfe::utils::MemorySpace::HOST>
        cellMassVectorHost, cellInvMassVectorHost, cellInvSqrtMassVectorHost,
        cellSqrtMassVectorHost;
      cellMassVectorHost.resize(d_nCells * d_nDofsPerCell);
      cellInvMassVectorHost.resize(d_nCells * d_nDofsPerCell);
      cellSqrtMassVectorHost.resize(d_nCells * d_nDofsPerCell);
      cellInvSqrtMassVectorHost.resize(d_nCells * d_nDofsPerCell);
      unsigned int iElemCount = 0;
      for (; cell != endc; ++cell)
        {
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(cell_dof_indices);
              for (unsigned int iNode = 0; iNode < d_nDofsPerCell; ++iNode)
                {
                  dealii::types::global_dof_index globalIndex =
                    cell_dof_indices[iNode];
                  if ((*d_constraintsVector)[d_dofHandlerID]->is_constrained(
                        globalIndex))
                    {
                      cellMassVectorHost[iElemCount * d_nDofsPerCell + iNode] =
                        1.0;
                      cellInvMassVectorHost[iElemCount * d_nDofsPerCell +
                                            iNode]     = 1.0;
                      cellSqrtMassVectorHost[iElemCount * d_nDofsPerCell +
                                             iNode]    = 1.0;
                      cellInvSqrtMassVectorHost[iElemCount * d_nDofsPerCell +
                                                iNode] = 1.0;
                    }
                  else
                    {
                      ValueTypeBasisData massVecValue = massVector(globalIndex);
                      cellMassVectorHost[iElemCount * d_nDofsPerCell + iNode] =
                        massVecValue;
                      cellSqrtMassVectorHost[iElemCount * d_nDofsPerCell +
                                             iNode] = std::sqrt(massVecValue);
                      if (std::abs(massVecValue) > 1.0e-15)
                        {
                          cellInvMassVectorHost[iElemCount * d_nDofsPerCell +
                                                iNode] = 1.0 / massVecValue;
                          cellInvSqrtMassVectorHost[iElemCount *
                                                      d_nDofsPerCell +
                                                    iNode] =
                            1.0 / std::sqrt(massVecValue);
                        }
                    }
                }
              ++iElemCount;
            }
        }
      if (basisType)
        {
          d_cellMassVectorBasisType.resize(cellMassVectorHost.size());
          d_cellMassVectorBasisType.copyFrom(cellMassVectorHost);
          d_cellInverseMassVectorBasisType.resize(cellInvMassVectorHost.size());
          d_cellInverseMassVectorBasisType.copyFrom(cellInvMassVectorHost);
          d_cellSqrtMassVectorBasisType.resize(cellSqrtMassVectorHost.size());
          d_cellSqrtMassVectorBasisType.copyFrom(cellSqrtMassVectorHost);
          d_cellInverseSqrtMassVectorBasisType.resize(
            cellInvSqrtMassVectorHost.size());
          d_cellInverseSqrtMassVectorBasisType.copyFrom(
            cellInvSqrtMassVectorHost);

          d_inverseSqrtMassVectorBasisType.resize(
            mpiPatternP2P->localOwnedSize() + mpiPatternP2P->localGhostSize());
          d_inverseSqrtMassVectorBasisType
            .template copyFrom<dftfe::utils::MemorySpace::HOST>(
              invSqrtMassVector.begin(),
              d_inverseSqrtMassVectorBasisType.size(),
              0,
              0);
          d_sqrtMassVectorBasisType.resize(mpiPatternP2P->localOwnedSize() +
                                           mpiPatternP2P->localGhostSize());
          d_sqrtMassVectorBasisType
            .template copyFrom<dftfe::utils::MemorySpace::HOST>(
              sqrtMassVector.begin(), d_sqrtMassVectorBasisType.size(), 0, 0);
          d_inverseMassVectorBasisType.resize(mpiPatternP2P->localOwnedSize() +
                                              mpiPatternP2P->localGhostSize());
          d_inverseMassVectorBasisType
            .template copyFrom<dftfe::utils::MemorySpace::HOST>(
              invMassVector.begin(), d_inverseMassVectorBasisType.size(), 0, 0);
          d_massVectorBasisType.resize(mpiPatternP2P->localOwnedSize() +
                                       mpiPatternP2P->localGhostSize());
          d_massVectorBasisType
            .template copyFrom<dftfe::utils::MemorySpace::HOST>(
              massVector.begin(), d_massVectorBasisType.size(), 0, 0);
        }
      if (ceoffType)
        {
          if (!basisType)
            {
              d_cellMassVectorBasisType.resize(cellMassVectorHost.size());
              d_cellMassVectorBasisType.copyFrom(cellMassVectorHost);
              d_cellInverseMassVectorBasisType.resize(
                cellInvMassVectorHost.size());
              d_cellInverseMassVectorBasisType.copyFrom(cellInvMassVectorHost);
              d_cellSqrtMassVectorBasisType.resize(
                cellSqrtMassVectorHost.size());
              d_cellSqrtMassVectorBasisType.copyFrom(cellSqrtMassVectorHost);
              d_cellInverseSqrtMassVectorBasisType.resize(
                cellInvSqrtMassVectorHost.size());
              d_cellInverseSqrtMassVectorBasisType.copyFrom(
                cellInvSqrtMassVectorHost);

              d_inverseSqrtMassVectorBasisType.resize(
                mpiPatternP2P->localOwnedSize() +
                mpiPatternP2P->localGhostSize());
              d_inverseSqrtMassVectorBasisType
                .template copyFrom<dftfe::utils::MemorySpace::HOST>(
                  invSqrtMassVector.begin(),
                  d_inverseSqrtMassVectorBasisType.size(),
                  0,
                  0);
              d_sqrtMassVectorBasisType.resize(mpiPatternP2P->localOwnedSize() +
                                               mpiPatternP2P->localGhostSize());
              d_sqrtMassVectorBasisType
                .template copyFrom<dftfe::utils::MemorySpace::HOST>(
                  sqrtMassVector.begin(),
                  d_sqrtMassVectorBasisType.size(),
                  0,
                  0);
              d_inverseMassVectorBasisType.resize(
                mpiPatternP2P->localOwnedSize() +
                mpiPatternP2P->localGhostSize());
              d_inverseMassVectorBasisType
                .template copyFrom<dftfe::utils::MemorySpace::HOST>(
                  invMassVector.begin(),
                  d_inverseMassVectorBasisType.size(),
                  0,
                  0);
              d_massVectorBasisType.resize(mpiPatternP2P->localOwnedSize() +
                                           mpiPatternP2P->localGhostSize());
              d_massVectorBasisType
                .template copyFrom<dftfe::utils::MemorySpace::HOST>(
                  massVector.begin(), d_massVectorBasisType.size(), 0, 0);
            }
          if constexpr (!std::is_same<ValueTypeBasisCoeff,
                                      ValueTypeBasisData>::value)
            {
              d_cellInverseMassVectorCoeffType.resize(
                cellInvMassVectorHost.size());
              d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                d_nDofsPerCell * d_nCells,
                d_cellInverseMassVectorBasisType.data(),
                d_cellInverseMassVectorCoeffType.data());
              d_cellInverseSqrtMassVectorCoeffType.resize(
                cellInvSqrtMassVectorHost.size());
              d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                d_nDofsPerCell * d_nCells,
                d_cellInverseSqrtMassVectorBasisType.data(),
                d_cellInverseSqrtMassVectorCoeffType.data());
              d_cellMassVectorCoeffType.resize(cellMassVectorHost.size());
              d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                d_nDofsPerCell * d_nCells,
                d_cellMassVectorBasisType.data(),
                d_cellMassVectorCoeffType.data());
              d_cellSqrtMassVectorCoeffType.resize(
                cellSqrtMassVectorHost.size());
              d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                d_nDofsPerCell * d_nCells,
                d_cellSqrtMassVectorBasisType.data(),
                d_cellSqrtMassVectorCoeffType.data());
              d_inverseSqrtMassVectorCoeffType.resize(
                mpiPatternP2P->localOwnedSize() +
                mpiPatternP2P->localOwnedSize());
              d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                d_inverseSqrtMassVectorCoeffType.size(),
                d_inverseSqrtMassVectorBasisType.data(),
                d_inverseSqrtMassVectorCoeffType.data());
              d_sqrtMassVectorCoeffType.resize(mpiPatternP2P->localOwnedSize() +
                                               mpiPatternP2P->localOwnedSize());
              d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                d_sqrtMassVectorCoeffType.size(),
                d_sqrtMassVectorBasisType.data(),
                d_sqrtMassVectorCoeffType.data());
              d_massVectorCoeffType.resize(mpiPatternP2P->localOwnedSize() +
                                           mpiPatternP2P->localOwnedSize());
              d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                d_massVectorCoeffType.size(),
                d_massVectorBasisType.data(),
                d_massVectorCoeffType.data());
              d_inverseMassVectorCoeffType.resize(
                mpiPatternP2P->localOwnedSize() +
                mpiPatternP2P->localOwnedSize());
              d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                d_inverseMassVectorCoeffType.size(),
                d_inverseMassVectorBasisType.data(),
                d_inverseMassVectorCoeffType.data());
            }
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      computeStiffnessVector(const bool basisType, const bool ceoffType)
    {
      distributedCPUVec<double> stiffnessVector, sqrtStiffnessVector,
        invStiffnessVector, invSqrtStiffnessVector;
      d_matrixFreeDataPtr->initialize_dof_vector(stiffnessVector,
                                                 d_dofHandlerID);
      sqrtStiffnessVector.reinit(stiffnessVector);
      invStiffnessVector.reinit(stiffnessVector);
      invSqrtStiffnessVector.reinit(stiffnessVector);
      stiffnessVector        = 0.0;
      sqrtStiffnessVector    = 0.0;
      invStiffnessVector     = 0.0;
      invSqrtStiffnessVector = 0.0;

      //      std::cout<<" size of stiffnessVector vec =
      //      "<<stiffnessVector.size()<<"\n";
      dealii::types::global_dof_index sizeVectemp = stiffnessVector.size();

      //      std::cout<<" dof handler id = "<<d_dofHandlerID<<"\n";
      // FIXME : check for roundoff errors
      dealii::QGauss<3>   quadrature(std::cbrt(d_nDofsPerCell) + 1);
      unsigned int        nQuadsPerCell = quadrature.size();
      dealii::FEValues<3> fe_values(
        d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).get_fe(),
        quadrature,
        dealii::update_gradients | dealii::update_JxW_values);

      dealii::Vector<double> stiffnessVectorLocal(d_nDofsPerCell);
      std::vector<dealii::types::global_dof_index> local_dof_indices(
        d_nDofsPerCell);

      //      std::cout<<" dofs per cell = "<<d_nDofsPerCell<<"\n";

      //
      // parallel loop over all elements
      //
      typename dealii::DoFHandler<3>::active_cell_iterator
        cell =
          d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).begin_active(),
        endc = d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).end();
      for (; cell != endc; ++cell)
        if (cell->is_locally_owned())
          {
            // compute values for the current element
            fe_values.reinit(cell);
            stiffnessVectorLocal = 0.0;
            for (unsigned int iDoF = 0; iDoF < d_nDofsPerCell; ++iDoF)
              for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
                stiffnessVectorLocal(iDoF) +=
                  fe_values.shape_grad(iDoF, iQuad) *
                  fe_values.shape_grad(iDoF, iQuad) * fe_values.JxW(iQuad);

            cell->get_dof_indices(local_dof_indices);

            //            for ( unsigned int iNode = 0 ;iNode < d_nDofsPerCell;
            //            iNode++)
            //              {
            //                if (local_dof_indices[iNode] > sizeVectemp)
            //                  {
            //                    std::cout<<" global id greater than max size
            //                    \n";
            //                  }
            //
            //              }
            (*d_constraintsVector)[d_dofHandlerID]->distribute_local_to_global(
              stiffnessVectorLocal, local_dof_indices, stiffnessVector);
          }

      stiffnessVector.compress(dealii::VectorOperation::add);
      stiffnessVector.update_ghost_values();


      for (dealii::types::global_dof_index i = 0; i < stiffnessVector.size();
           ++i)
        if (stiffnessVector.in_local_range(i) &&
            !((*d_constraintsVector)[d_dofHandlerID]->is_constrained(i)))
          {
            sqrtStiffnessVector(i) = std::sqrt(stiffnessVector(i));
            if (std::abs(stiffnessVector(i)) > 1.0e-15)
              {
                invSqrtStiffnessVector(i) = 1.0 / std::sqrt(stiffnessVector(i));
                invStiffnessVector(i)     = 1.0 / stiffnessVector(i);
              }

            AssertThrow(
              !std::isnan(invStiffnessVector(i)),
              dealii::ExcMessage(
                "Value of inverse square root of stiffness matrix on the unconstrained node is undefined"));
          }

      invStiffnessVector.compress(dealii::VectorOperation::insert);
      invStiffnessVector.update_ghost_values();
      sqrtStiffnessVector.compress(dealii::VectorOperation::insert);
      sqrtStiffnessVector.update_ghost_values();
      invSqrtStiffnessVector.compress(dealii::VectorOperation::insert);
      invSqrtStiffnessVector.update_ghost_values();

      cell =
        d_matrixFreeDataPtr->get_dof_handler(d_dofHandlerID).begin_active();
      std::vector<dealii::types::global_dof_index> cell_dof_indices(
        d_nDofsPerCell);

      dftfe::utils::MemoryStorage<ValueTypeBasisData,
                                  dftfe::utils::MemorySpace::HOST>
        cellStiffnessVectorHost, cellInvStiffnessVectorHost,
        cellInvSqrtStiffnessVectorHost, cellSqrtStiffnessVectorHost;
      cellStiffnessVectorHost.resize(d_nCells * d_nDofsPerCell);
      cellInvStiffnessVectorHost.resize(d_nCells * d_nDofsPerCell);
      cellSqrtStiffnessVectorHost.resize(d_nCells * d_nDofsPerCell);
      cellInvSqrtStiffnessVectorHost.resize(d_nCells * d_nDofsPerCell);
      unsigned int iElemCount = 0;
      for (; cell != endc; ++cell)
        {
          if (cell->is_locally_owned())
            {
              cell->get_dof_indices(cell_dof_indices);
              for (unsigned int iNode = 0; iNode < d_nDofsPerCell; ++iNode)
                {
                  dealii::types::global_dof_index globalIndex =
                    cell_dof_indices[iNode];
                  if ((*d_constraintsVector)[d_dofHandlerID]->is_constrained(
                        globalIndex))
                    {
                      cellStiffnessVectorHost[iElemCount * d_nDofsPerCell +
                                              iNode]        = 1.0;
                      cellInvStiffnessVectorHost[iElemCount * d_nDofsPerCell +
                                                 iNode]     = 1.0;
                      cellSqrtStiffnessVectorHost[iElemCount * d_nDofsPerCell +
                                                  iNode]    = 1.0;
                      cellInvSqrtStiffnessVectorHost[iElemCount *
                                                       d_nDofsPerCell +
                                                     iNode] = 1.0;
                    }
                  else
                    {
                      ValueTypeBasisData stiffnessVecValue =
                        stiffnessVector(globalIndex);
                      cellStiffnessVectorHost[iElemCount * d_nDofsPerCell +
                                              iNode] = stiffnessVecValue;
                      cellSqrtStiffnessVectorHost[iElemCount * d_nDofsPerCell +
                                                  iNode] =
                        std::sqrt(stiffnessVecValue);
                      if (std::abs(stiffnessVecValue) > 1.0e-15)
                        {
                          cellInvStiffnessVectorHost[iElemCount *
                                                       d_nDofsPerCell +
                                                     iNode] =
                            1.0 / stiffnessVecValue;
                          cellInvSqrtStiffnessVectorHost[iElemCount *
                                                           d_nDofsPerCell +
                                                         iNode] =
                            1.0 / std::sqrt(stiffnessVecValue);
                        }
                    }
                }
              ++iElemCount;
            }
        }
      if (basisType)
        {
          d_cellStiffnessVectorBasisType.resize(cellStiffnessVectorHost.size());
          d_cellStiffnessVectorBasisType.copyFrom(cellStiffnessVectorHost);
          d_cellInverseStiffnessVectorBasisType.resize(
            cellInvStiffnessVectorHost.size());
          d_cellInverseStiffnessVectorBasisType.copyFrom(
            cellInvStiffnessVectorHost);
          d_cellSqrtStiffnessVectorBasisType.resize(
            cellSqrtStiffnessVectorHost.size());
          d_cellSqrtStiffnessVectorBasisType.copyFrom(
            cellSqrtStiffnessVectorHost);
          d_cellInverseSqrtStiffnessVectorBasisType.resize(
            cellInvSqrtStiffnessVectorHost.size());
          d_cellInverseSqrtStiffnessVectorBasisType.copyFrom(
            cellInvSqrtStiffnessVectorHost);

          d_inverseSqrtStiffnessVectorBasisType.resize(
            mpiPatternP2P->localOwnedSize() + mpiPatternP2P->localGhostSize());
          d_inverseSqrtStiffnessVectorBasisType
            .template copyFrom<dftfe::utils::MemorySpace::HOST>(
              invSqrtStiffnessVector.begin(),
              d_inverseSqrtStiffnessVectorBasisType.size(),
              0,
              0);
          d_sqrtStiffnessVectorBasisType.resize(
            mpiPatternP2P->localOwnedSize() + mpiPatternP2P->localGhostSize());
          d_sqrtStiffnessVectorBasisType
            .template copyFrom<dftfe::utils::MemorySpace::HOST>(
              sqrtStiffnessVector.begin(),
              d_sqrtStiffnessVectorBasisType.size(),
              0,
              0);
          d_inverseStiffnessVectorBasisType.resize(
            mpiPatternP2P->localOwnedSize() + mpiPatternP2P->localGhostSize());
          d_inverseStiffnessVectorBasisType
            .template copyFrom<dftfe::utils::MemorySpace::HOST>(
              invStiffnessVector.begin(),
              d_inverseStiffnessVectorBasisType.size(),
              0,
              0);
          d_stiffnessVectorBasisType.resize(mpiPatternP2P->localOwnedSize() +
                                            mpiPatternP2P->localGhostSize());
          d_stiffnessVectorBasisType
            .template copyFrom<dftfe::utils::MemorySpace::HOST>(
              stiffnessVector.begin(), d_stiffnessVectorBasisType.size(), 0, 0);
        }
      if (ceoffType)
        {
          if (!basisType)
            {
              d_cellStiffnessVectorBasisType.resize(
                cellStiffnessVectorHost.size());
              d_cellStiffnessVectorBasisType.copyFrom(cellStiffnessVectorHost);
              d_cellInverseStiffnessVectorBasisType.resize(
                cellInvStiffnessVectorHost.size());
              d_cellInverseStiffnessVectorBasisType.copyFrom(
                cellInvStiffnessVectorHost);
              d_cellSqrtStiffnessVectorBasisType.resize(
                cellSqrtStiffnessVectorHost.size());
              d_cellSqrtStiffnessVectorBasisType.copyFrom(
                cellSqrtStiffnessVectorHost);
              d_cellInverseSqrtStiffnessVectorBasisType.resize(
                cellInvSqrtStiffnessVectorHost.size());
              d_cellInverseSqrtStiffnessVectorBasisType.copyFrom(
                cellInvSqrtStiffnessVectorHost);

              d_inverseSqrtStiffnessVectorBasisType.resize(
                mpiPatternP2P->localOwnedSize() +
                mpiPatternP2P->localGhostSize());
              d_inverseSqrtStiffnessVectorBasisType
                .template copyFrom<dftfe::utils::MemorySpace::HOST>(
                  invSqrtStiffnessVector.begin(),
                  d_inverseSqrtStiffnessVectorBasisType.size(),
                  0,
                  0);
              d_sqrtStiffnessVectorBasisType.resize(
                mpiPatternP2P->localOwnedSize() +
                mpiPatternP2P->localGhostSize());
              d_sqrtStiffnessVectorBasisType
                .template copyFrom<dftfe::utils::MemorySpace::HOST>(
                  sqrtStiffnessVector.begin(),
                  d_sqrtStiffnessVectorBasisType.size(),
                  0,
                  0);
              d_inverseStiffnessVectorBasisType.resize(
                mpiPatternP2P->localOwnedSize() +
                mpiPatternP2P->localGhostSize());
              d_inverseStiffnessVectorBasisType
                .template copyFrom<dftfe::utils::MemorySpace::HOST>(
                  invStiffnessVector.begin(),
                  d_inverseStiffnessVectorBasisType.size(),
                  0,
                  0);
              d_stiffnessVectorBasisType.resize(
                mpiPatternP2P->localOwnedSize() +
                mpiPatternP2P->localGhostSize());
              d_stiffnessVectorBasisType
                .template copyFrom<dftfe::utils::MemorySpace::HOST>(
                  stiffnessVector.begin(),
                  d_stiffnessVectorBasisType.size(),
                  0,
                  0);
            }
          if constexpr (!std::is_same<ValueTypeBasisCoeff,
                                      ValueTypeBasisData>::value)
            {
              d_cellInverseStiffnessVectorCoeffType.resize(
                cellInvStiffnessVectorHost.size());
              d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                d_nDofsPerCell * d_nCells,
                d_cellInverseStiffnessVectorBasisType.data(),
                d_cellInverseStiffnessVectorCoeffType.data());
              d_cellInverseSqrtStiffnessVectorCoeffType.resize(
                cellInvSqrtStiffnessVectorHost.size());
              d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                d_nDofsPerCell * d_nCells,
                d_cellInverseSqrtStiffnessVectorBasisType.data(),
                d_cellInverseSqrtStiffnessVectorCoeffType.data());
              d_cellStiffnessVectorCoeffType.resize(
                cellStiffnessVectorHost.size());
              d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                d_nDofsPerCell * d_nCells,
                d_cellStiffnessVectorBasisType.data(),
                d_cellStiffnessVectorCoeffType.data());
              d_cellSqrtStiffnessVectorCoeffType.resize(
                cellSqrtStiffnessVectorHost.size());
              d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                d_nDofsPerCell * d_nCells,
                d_cellSqrtStiffnessVectorBasisType.data(),
                d_cellSqrtStiffnessVectorCoeffType.data());
              d_inverseSqrtStiffnessVectorCoeffType.resize(
                mpiPatternP2P->localOwnedSize() +
                mpiPatternP2P->localOwnedSize());
              d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                d_inverseSqrtStiffnessVectorCoeffType.size(),
                d_inverseSqrtStiffnessVectorBasisType.data(),
                d_inverseSqrtStiffnessVectorCoeffType.data());
              d_sqrtStiffnessVectorCoeffType.resize(
                mpiPatternP2P->localOwnedSize() +
                mpiPatternP2P->localOwnedSize());
              d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                d_sqrtStiffnessVectorCoeffType.size(),
                d_sqrtStiffnessVectorBasisType.data(),
                d_sqrtStiffnessVectorCoeffType.data());
              d_stiffnessVectorCoeffType.resize(
                mpiPatternP2P->localOwnedSize() +
                mpiPatternP2P->localOwnedSize());
              d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                d_stiffnessVectorCoeffType.size(),
                d_stiffnessVectorBasisType.data(),
                d_stiffnessVectorCoeffType.data());
              d_inverseStiffnessVectorCoeffType.resize(
                mpiPatternP2P->localOwnedSize() +
                mpiPatternP2P->localOwnedSize());
              d_BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
                d_inverseStiffnessVectorCoeffType.size(),
                d_inverseStiffnessVectorBasisType.data(),
                d_inverseStiffnessVectorCoeffType.data());
            }
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
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
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
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
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      createScratchMultiVectorsSinglePrec(const unsigned int vecBlockSize,
                                          const unsigned int numMultiVecs) const
    {
      auto iter = scratchMultiVectorsSinglePrec.find(vecBlockSize);
      if (iter == scratchMultiVectorsSinglePrec.end())
        {
          scratchMultiVectorsSinglePrec[vecBlockSize] =
            std::vector<dftfe::linearAlgebra::MultiVector<
              typename dftfe::dataTypes::singlePrecType<
                ValueTypeBasisCoeff>::type,
              memorySpace>>(numMultiVecs);
          for (unsigned int iVec = 0; iVec < numMultiVecs; ++iVec)
            scratchMultiVectorsSinglePrec[vecBlockSize][iVec].reinit(
              mpiPatternP2P, vecBlockSize);
        }
      else
        {
          scratchMultiVectorsSinglePrec[vecBlockSize].resize(
            scratchMultiVectorsSinglePrec[vecBlockSize].size() + numMultiVecs);
          for (unsigned int iVec = 0;
               iVec < scratchMultiVectorsSinglePrec[vecBlockSize].size();
               ++iVec)
            scratchMultiVectorsSinglePrec[vecBlockSize][iVec].reinit(
              mpiPatternP2P, vecBlockSize);
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      clearScratchMultiVectors() const
    {
      scratchMultiVectors.clear();
      scratchMultiVectorsSinglePrec.clear();
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      getMultiVector(const unsigned int vecBlockSize,
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
    dftfe::linearAlgebra::MultiVector<
      typename dftfe::dataTypes::singlePrecType<ValueTypeBasisCoeff>::type,
      memorySpace> &
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      getMultiVectorSinglePrec(const unsigned int vecBlockSize,
                               const unsigned int index) const
    {
      AssertThrow(scratchMultiVectorsSinglePrec.find(vecBlockSize) !=
                    scratchMultiVectorsSinglePrec.end(),
                  dealii::ExcMessage(
                    "DFT-FE Error: MultiVector not found in scratch storage."));
      return scratchMultiVectorsSinglePrec[vecBlockSize][index];
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      distribute(dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                   memorySpace> &multiVector,
                 unsigned int constraintIndex) const
    {
      d_constraintInfo[constraintIndex ==
                           std::numeric_limits<unsigned int>::max() ?
                         d_dofHandlerID :
                         constraintIndex]
        .distribute(multiVector);
    }

    template class FEBasisOperations<double,
                                     double,
                                     dftfe::utils::MemorySpace::HOST>;

    template void
    FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>::init(
      const FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>
        &basisOperationsSrc);
#if defined(USE_COMPLEX)
    template class FEBasisOperations<std::complex<double>,
                                     double,
                                     dftfe::utils::MemorySpace::HOST>;
#endif
#if defined(DFTFE_WITH_DEVICE)
    template class FEBasisOperations<double,
                                     double,
                                     dftfe::utils::MemorySpace::DEVICE>;
    template void
    FEBasisOperations<double, double, dftfe::utils::MemorySpace::DEVICE>::init(
      const FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>
        &basisOperationsSrc);
#  if defined(USE_COMPLEX)
    template class FEBasisOperations<std::complex<double>,
                                     double,
                                     dftfe::utils::MemorySpace::DEVICE>;
    template void
    FEBasisOperations<std::complex<double>,
                      double,
                      dftfe::utils::MemorySpace::DEVICE>::
      init(const FEBasisOperations<std::complex<double>,
                                   double,
                                   dftfe::utils::MemorySpace::HOST>
             &basisOperationsSrc);
#  endif
#endif
  } // namespace basis
} // namespace dftfe
