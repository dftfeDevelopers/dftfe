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
#include <linearAlgebraOperations.h>
#include <FEBasisOperationsKernelsInternal.h>

namespace dftfe
{
  namespace basis
  {
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      interpolate(dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                    memorySpace> &nodalData,
                  ValueTypeBasisCoeff *quadratureValues,
                  ValueTypeBasisCoeff *quadratureGradients) const
    {
      interpolateKernel(nodalData,
                        quadratureValues,
                        quadratureGradients,
                        std::pair<unsigned int, unsigned int>(0, d_nCells));
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      integrateWithBasis(
        ValueTypeBasisCoeff *quadratureValues,
        ValueTypeBasisCoeff *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &nodalData) const
    {
      integrateWithBasisKernel(quadratureValues,
                               quadratureGradients,
                               nodalData,
                               std::pair<unsigned int, unsigned int>(0,
                                                                     d_nCells));
    }


    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      extractToCellNodalData(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &                  nodalData,
        ValueTypeBasisCoeff *cellNodalDataPtr) const
    {
      extractToCellNodalDataKernel(
        nodalData,
        cellNodalDataPtr,
        std::pair<unsigned int, unsigned int>(0, d_nCells));
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      accumulateFromCellNodalData(
        const ValueTypeBasisCoeff *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &nodalData) const
    {
      accumulateFromCellNodalDataKernel(
        cellNodalDataPtr,
        nodalData,
        std::pair<unsigned int, unsigned int>(0, d_nCells));
    }
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      interpolateKernel(
        const dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                memorySpace> &nodalValues,
        ValueTypeBasisCoeff *                                 quadratureValues,
        ValueTypeBasisCoeff *                       quadratureGradients,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      for (unsigned int iCell = cellRange.first; iCell < cellRange.second;
           iCell += d_cellsBlockSize)
        {
          extractToCellNodalDataKernel(
            nodalValues,
            tempCellNodalData.data(),
            std::pair<unsigned int, unsigned int>(
              iCell, std::min(d_nCells, iCell + d_cellsBlockSize)));
          interpolateKernel(tempCellNodalData.data(),
                            quadratureValues,
                            quadratureGradients,
                            std::pair<unsigned int, unsigned int>(
                              iCell,
                              std::min(d_nCells, iCell + d_cellsBlockSize)));
        }
    }
    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      interpolateKernel(
        const ValueTypeBasisCoeff *                 cellNodalValues,
        ValueTypeBasisCoeff *                       quadratureValues,
        ValueTypeBasisCoeff *                       quadratureGradients,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      const ValueTypeBasisCoeff scalarCoeffAlpha = ValueTypeBasisCoeff(1.0),
                                scalarCoeffBeta  = ValueTypeBasisCoeff(0.0);

      if (quadratureValues != NULL)
        {
          d_BLASWrapperPtr->xgemmStridedBatched(
            'N',
            'N',
            d_nVectors,
            d_nQuadsPerCell[d_quadratureIndex],
            d_nDofsPerCell,
            &scalarCoeffAlpha,
            cellNodalValues,
            d_nVectors,
            d_nVectors * d_nDofsPerCell,
            d_shapeFunctionData.find(d_quadratureID)->second.data(),
            d_nDofsPerCell,
            0,
            &scalarCoeffBeta,
            quadratureValues,
            d_nVectors,
            d_nVectors * d_nQuadsPerCell[d_quadratureIndex],
            cellRange.second - cellRange.first);
        }

      if (quadratureGradients != NULL)
        {
          d_BLASWrapperPtr->xgemmStridedBatched(
            'N',
            'N',
            d_nVectors,
            d_nQuadsPerCell[d_quadratureIndex] * 3,
            d_nDofsPerCell,
            &scalarCoeffAlpha,
            cellNodalValues,
            d_nVectors,
            d_nVectors * d_nDofsPerCell,
            d_shapeFunctionGradientDataInternalLayout.find(d_quadratureID)
              ->second.data(),
            d_nDofsPerCell,
            0,
            &scalarCoeffBeta,
            areAllCellsCartesian ? quadratureGradients :
                                   tempQuadratureGradientsData.data(),
            d_nVectors,
            d_nVectors * d_nQuadsPerCell[d_quadratureIndex] * 3,
            cellRange.second - cellRange.first);
          if (areAllCellsCartesian)
            {
              d_BLASWrapperPtr->stridedBlockScale(
                d_nQuadsPerCell[d_quadratureIndex] * d_nVectors,
                3 * (cellRange.second - cellRange.first),
                ValueTypeBasisCoeff(1.0),
                d_inverseJacobianData.find(0)->second.data() +
                  cellRange.first * 3,
                quadratureGradients);
            }
          else if (areAllCellsAffine)
            {
              d_BLASWrapperPtr->xgemmStridedBatched(
                'N',
                'N',
                d_nQuadsPerCell[d_quadratureIndex] * d_nVectors,
                3,
                3,
                &scalarCoeffAlpha,
                tempQuadratureGradientsData.data(),
                d_nQuadsPerCell[d_quadratureIndex] * d_nVectors,
                d_nQuadsPerCell[d_quadratureIndex] * d_nVectors * 3,
                d_inverseJacobianData.find(0)->second.data() +
                  9 * cellRange.first,
                3,
                9,
                &scalarCoeffBeta,
                quadratureGradients,
                d_nQuadsPerCell[d_quadratureIndex] * d_nVectors,
                d_nVectors * d_nQuadsPerCell[d_quadratureIndex] * 3,
                cellRange.second - cellRange.first);
            }
          else
            {
              d_BLASWrapperPtr->xgemmStridedBatched(
                'N',
                'N',
                d_nVectors,
                3,
                3,
                &scalarCoeffAlpha,
                tempQuadratureGradientsData.data(),
                d_nVectors,
                d_nVectors * 3,
                d_inverseJacobianData.find(d_quadratureID)->second.data() +
                  9 * cellRange.first * d_nQuadsPerCell[d_quadratureIndex],
                3,
                9,
                &scalarCoeffBeta,
                tempQuadratureGradientsDataNonAffine.data(),
                d_nVectors,
                d_nVectors * 3,
                (cellRange.second - cellRange.first) *
                  d_nQuadsPerCell[d_quadratureIndex]);
              if (memorySpace == dftfe::utils::MemorySpace::HOST)
                dftfe::basis::FEBasisOperationsKernelsInternal::
                  reshapeFromNonAffineLayoutHost(
                    d_nVectors,
                    d_nQuadsPerCell[d_quadratureIndex],
                    (cellRange.second - cellRange.first),
                    tempQuadratureGradientsDataNonAffine.data(),
                    quadratureGradients);
              else
                dftfe::basis::FEBasisOperationsKernelsInternal::
                  reshapeFromNonAffineLayoutDevice(
                    d_nVectors,
                    d_nQuadsPerCell[d_quadratureIndex],
                    (cellRange.second - cellRange.first),
                    tempQuadratureGradientsDataNonAffine.data(),
                    quadratureGradients);
            }
        }
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      integrateWithBasisKernel(
        const ValueTypeBasisCoeff *quadratureValues,
        const ValueTypeBasisCoeff *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &                                         nodalData,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      const ValueTypeBasisCoeff scalarCoeffAlpha = ValueTypeBasisCoeff(1.0),
                                scalarCoeffBeta  = ValueTypeBasisCoeff(0.0);
      if (quadratureValues != NULL)
        {
          d_BLASWrapperPtr->xgemmStridedBatched(
            'N',
            'T',
            d_nVectors,
            d_nDofsPerCell,
            d_nQuadsPerCell[d_quadratureIndex],
            &scalarCoeffAlpha,
            quadratureValues,
            d_nVectors,
            d_nVectors * d_nQuadsPerCell[d_quadratureIndex],
            d_shapeFunctionData.find(d_quadratureID)->second.data(),
            d_nQuadsPerCell[d_quadratureIndex],
            0,
            &scalarCoeffBeta,
            tempCellNodalData.data(),
            d_nVectors,
            d_nVectors * d_nDofsPerCell,
            cellRange.second - cellRange.first);
        }
      if (quadratureGradients != NULL)
        {
          if (areAllCellsCartesian)
            {
              tempQuadratureGradientsData.template copyFrom<memorySpace>(
                quadratureGradients,
                3 * d_nQuadsPerCell[d_quadratureIndex] * d_nVectors *
                  (cellRange.second - cellRange.first),
                3 * d_nQuadsPerCell[d_quadratureIndex] * d_nVectors *
                  cellRange.first,
                0);
              d_BLASWrapperPtr->stridedBlockScale(
                d_nQuadsPerCell[d_quadratureIndex] * d_nVectors,
                3 * (cellRange.second - cellRange.first),
                ValueTypeBasisCoeff(1.0),
                d_inverseJacobianData.find(0)->second.data() +
                  cellRange.first * 3,
                tempQuadratureGradientsData.data());
            }
          else if (areAllCellsAffine)
            {
              d_BLASWrapperPtr->xgemmStridedBatched(
                'N',
                'T',
                d_nQuadsPerCell[d_quadratureIndex] * d_nVectors,
                3,
                3,
                &scalarCoeffAlpha,
                quadratureGradients,
                d_nQuadsPerCell[d_quadratureIndex] * d_nVectors,
                d_nQuadsPerCell[d_quadratureIndex] * d_nVectors * 3,
                d_inverseJacobianData.find(0)->second.data() +
                  9 * cellRange.first,
                3,
                9,
                &scalarCoeffBeta,
                tempQuadratureGradientsData.data(),
                d_nQuadsPerCell[d_quadratureIndex] * d_nVectors,
                d_nVectors * d_nQuadsPerCell[d_quadratureIndex] * 3,
                cellRange.second - cellRange.first);
            }
          else
            {
              if (memorySpace == dftfe::utils::MemorySpace::HOST)
                dftfe::basis::FEBasisOperationsKernelsInternal::
                  reshapeToNonAffineLayoutHost(
                    d_nVectors,
                    d_nQuadsPerCell[d_quadratureIndex],
                    (cellRange.second - cellRange.first),
                    quadratureGradients,
                    tempQuadratureGradientsDataNonAffine.data());
              else
                dftfe::basis::FEBasisOperationsKernelsInternal::
                  reshapeToNonAffineLayoutDevice(
                    d_nVectors,
                    d_nQuadsPerCell[d_quadratureIndex],
                    (cellRange.second - cellRange.first),
                    quadratureGradients,
                    tempQuadratureGradientsDataNonAffine.data());
              d_BLASWrapperPtr->xgemmStridedBatched(
                'N',
                'T',
                d_nVectors,
                3,
                3,
                &scalarCoeffAlpha,
                tempQuadratureGradientsDataNonAffine.data(),
                d_nVectors,
                d_nVectors * 3,
                d_inverseJacobianData.find(d_quadratureID)->second.data() +
                  9 * cellRange.first * d_nQuadsPerCell[d_quadratureIndex],
                3,
                9,
                &scalarCoeffBeta,
                tempQuadratureGradientsData.data(),
                d_nVectors,
                d_nVectors * 3,
                (cellRange.second - cellRange.first) *
                  d_nQuadsPerCell[d_quadratureIndex]);
            }
          d_BLASWrapperPtr->xgemmStridedBatched(
            'N',
            'T',
            d_nVectors,
            d_nDofsPerCell,
            d_nQuadsPerCell[d_quadratureIndex] * 3,
            &scalarCoeffAlpha,
            tempQuadratureGradientsData.data(),
            d_nVectors,
            d_nVectors * d_nQuadsPerCell[d_quadratureIndex],
            d_shapeFunctionGradientDataInternalLayout.find(d_quadratureID)
              ->second.data(),
            d_nQuadsPerCell[d_quadratureIndex],
            0,
            &scalarCoeffBeta,
            tempCellNodalData.data(),
            d_nVectors,
            d_nVectors * d_nDofsPerCell,
            cellRange.second - cellRange.first);
        }
      accumulateFromCellNodalDataKernel(tempCellNodalData.data(),
                                        nodalData,
                                        cellRange);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      extractToCellNodalDataKernel(
        const dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                                memorySpace> &nodalData,
        ValueTypeBasisCoeff *                                 cellNodalDataPtr,
        const std::pair<unsigned int, unsigned int>           cellRange) const
    {
      d_BLASWrapperPtr->stridedCopyToBlock(
        d_nVectors,
        (cellRange.second - cellRange.first) * d_nDofsPerCell,
        nodalData.data(),
        cellNodalDataPtr,
        d_flattenedCellDofIndexToProcessDofIndexMap.data() +
          cellRange.first * d_nDofsPerCell);
    }

    template <typename ValueTypeBasisCoeff,
              typename ValueTypeBasisData,
              dftfe::utils::MemorySpace memorySpace>
    void
    FEBasisOperations<ValueTypeBasisCoeff, ValueTypeBasisData, memorySpace>::
      accumulateFromCellNodalDataKernel(
        const ValueTypeBasisCoeff *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff, memorySpace>
          &                                         nodalData,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      d_BLASWrapperPtr->axpyStridedBlockAtomicAdd(
        d_nVectors,
        (cellRange.second - cellRange.first) * d_nDofsPerCell,
        cellNodalDataPtr,
        nodalData.begin(),
        d_flattenedCellDofIndexToProcessDofIndexMap.begin() +
          cellRange.first * d_nDofsPerCell);
    }
    template class FEBasisOperations<dataTypes::number,
                                     double,
                                     dftfe::utils::MemorySpace::HOST>;
#if defined(DFTFE_WITH_DEVICE)
    template class FEBasisOperations<dataTypes::number,
                                     double,
                                     dftfe::utils::MemorySpace::DEVICE>;
#endif

  } // namespace basis
} // namespace dftfe
