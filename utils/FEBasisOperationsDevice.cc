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
#include <deviceKernelsGeneric.h>
#include <DeviceBlasWrapper.h>
#include <DeviceTypeConfig.h>

namespace dftfe
{
  namespace basis
  {
    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::DEVICE>::
      interpolate(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &nodalData,
        std::map<dealii::CellId,
                 dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                             dftfe::utils::MemorySpace::HOST>>
          *quadratureValues,
        std::map<dealii::CellId,
                 dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                             dftfe::utils::MemorySpace::HOST>>
          *quadratureGradients) const
    {
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                  dftfe::utils::MemorySpace::DEVICE>
        quadratureValuesAllCells;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                  dftfe::utils::MemorySpace::DEVICE>
        quadratureGradientsAllCells;
      quadratureValuesAllCells.resize(d_nCells * d_nQuadsPerCell * d_nVectors);
      quadratureGradientsAllCells.resize(d_nCells * 3 * d_nQuadsPerCell *
                                         d_nVectors);

      for (unsigned int iCell = 0; iCell < d_nCells; ++iCell)
        {
          dealii::CellId currentCellId = d_cellIndexToCellIdMap[iCell];
          quadratureValuesAllCells.copyFrom(quadratureValues->at(currentCellId),
                                            d_nQuadsPerCell * d_nVectors,
                                            0,
                                            d_nVectors * d_nQuadsPerCell *
                                              iCell);
          if (quadratureGradients != NULL)
            quadratureGradientsAllCells.copyFrom(
              quadratureGradients->at(currentCellId),
              d_nQuadsPerCell * d_nVectors * 3,
              0,
              d_nVectors * d_nQuadsPerCell * 3 * iCell);
        }
      interpolateKernel(nodalData,
                        &quadratureValuesAllCells,
                        quadratureGradients == NULL ?
                          NULL :
                          &quadratureGradientsAllCells,
                        std::pair<unsigned int, unsigned int>(0, d_nCells));
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::DEVICE>::
      interpolate(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &nodalData,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::DEVICE>
          *quadratureValues,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::DEVICE>
          *quadratureGradients) const
    {
      interpolateKernel(nodalData,
                        quadratureValues,
                        quadratureGradients,
                        std::pair<unsigned int, unsigned int>(0, d_nCells));
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::DEVICE>::
      integrateWithBasis(
        std::map<dealii::CellId,
                 dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                             dftfe::utils::MemorySpace::HOST>>
          *quadratureValues,
        std::map<dealii::CellId,
                 dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                             dftfe::utils::MemorySpace::HOST>>
          *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &nodalData) const
    {
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                  dftfe::utils::MemorySpace::DEVICE>
        quadratureValuesAllCells;
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                  dftfe::utils::MemorySpace::DEVICE>
        quadratureGradientsAllCells;
      quadratureValuesAllCells.resize(d_nCells * d_nQuadsPerCell * d_nVectors);
      quadratureGradientsAllCells.resize(d_nCells * 3 * d_nQuadsPerCell *
                                         d_nVectors);

      for (unsigned int iCell = 0; iCell < d_nCells; ++iCell)
        {
          dealii::CellId currentCellId = d_cellIndexToCellIdMap[iCell];
          quadratureValuesAllCells.copyFrom(quadratureValues->at(currentCellId),
                                            d_nQuadsPerCell * d_nVectors,
                                            0,
                                            d_nVectors * d_nQuadsPerCell *
                                              iCell);
          if (quadratureGradients != NULL)
            quadratureGradientsAllCells.copyFrom(
              quadratureGradients->at(currentCellId),
              d_nQuadsPerCell * d_nVectors * 3,
              0,
              d_nVectors * d_nQuadsPerCell * 3 * iCell);
        }
      integrateWithBasisKernel(
        &quadratureValuesAllCells,
        quadratureGradients == NULL ? NULL : &quadratureGradientsAllCells,
        nodalData,
        std::pair<unsigned int, unsigned int>(0, d_nCells));
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::DEVICE>::
      integrateWithBasis(
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::DEVICE>
          *quadratureValues,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::DEVICE>
          *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &nodalData) const
    {
      integrateWithBasisKernel(quadratureValues,
                               quadratureGradients,
                               nodalData,
                               std::pair<unsigned int, unsigned int>(0,
                                                                     d_nCells));
    }


    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::DEVICE>::
      extractToCellNodalData(
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &nodalData,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::DEVICE>
          *cellNodalDataPtr) const
    {
      extractToCellNodalDataKernel(
        nodalData,
        cellNodalDataPtr,
        std::pair<unsigned int, unsigned int>(0, d_nCells));
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::DEVICE>::
      accumulateFromCellNodalData(
        const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &nodalData) const
    {
      accumulateFromCellNodalDataKernel(
        cellNodalDataPtr,
        nodalData,
        std::pair<unsigned int, unsigned int>(0, d_nCells));
    }



    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::DEVICE>::
      interpolateKernel(
        const dftfe::linearAlgebra::MultiVector<
          ValueTypeBasisCoeff,
          dftfe::utils::MemorySpace::DEVICE> &nodalValues,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::DEVICE>
          *quadratureValues,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::DEVICE>
          *                                         quadratureGradients,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                  dftfe::utils::MemorySpace::DEVICE>
        cellNodalData, tempQuadratureGradientsData,
        tempQuadratureGradientsDataNonAffine;
      cellNodalData.resize(d_nVectors * d_nDofsPerCell *
                           (cellRange.second - cellRange.first));

      if (quadratureGradients != NULL)
        tempQuadratureGradientsData.resize(
          areAllCellsCartesian ? 0 :
                                 (d_nVectors * d_nQuadsPerCell * 3 *
                                  (cellRange.second - cellRange.first)));

      if (quadratureGradients != NULL)
        tempQuadratureGradientsDataNonAffine.resize(
          areAllCellsAffine ? 0 :
                              (d_nVectors * d_nQuadsPerCell * 3 *
                               (cellRange.second - cellRange.first)));

      extractToCellNodalDataKernel(nodalValues, &cellNodalData, cellRange);

      const ValueTypeBasisCoeff scalarCoeffAlpha = ValueTypeBasisCoeff(1.0),
                                scalarCoeffBeta  = ValueTypeBasisCoeff(0.0);

      dftfe::utils::deviceBlasWrapper::gemmStridedBatched(
        *d_deviceBlasHandlePtr,
        dftfe::utils::DEVICEBLAS_OP_N,
        dftfe::utils::DEVICEBLAS_OP_N,
        d_nVectors,
        d_nQuadsPerCell,
        d_nDofsPerCell,
        &scalarCoeffAlpha,
        cellNodalData.data(),
        d_nVectors,
        d_nVectors * d_nDofsPerCell,
        d_shapeFunctionData.data(),
        d_nDofsPerCell,
        0,
        &scalarCoeffBeta,
        quadratureValues->data(),
        d_nVectors,
        d_nVectors * d_nQuadsPerCell,
        cellRange.second - cellRange.first);
      if (quadratureGradients != NULL)
        {
          dftfe::utils::deviceBlasWrapper::gemmStridedBatched(
            *d_deviceBlasHandlePtr,
            dftfe::utils::DEVICEBLAS_OP_N,
            dftfe::utils::DEVICEBLAS_OP_N,
            d_nVectors,
            d_nQuadsPerCell * 3,
            d_nDofsPerCell,
            &scalarCoeffAlpha,
            cellNodalData.data(),
            d_nVectors,
            d_nVectors * d_nDofsPerCell,
            d_shapeFunctionGradientData.data(),
            d_nDofsPerCell,
            0,
            &scalarCoeffBeta,
            areAllCellsCartesian ? quadratureGradients->data() :
                                   tempQuadratureGradientsData.data(),
            d_nVectors,
            d_nVectors * d_nQuadsPerCell * 3,
            cellRange.second - cellRange.first);
          if (areAllCellsCartesian)
            {
              dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
                d_nQuadsPerCell * d_nVectors,
                3 * (cellRange.second - cellRange.first),
                ValueTypeBasisCoeff(1.0),
                d_inverseJacobianData.data() + cellRange.first * 3,
                quadratureGradients->data());
            }
          else if (areAllCellsAffine)
            {
              dftfe::utils::deviceBlasWrapper::gemmStridedBatched(
                *d_deviceBlasHandlePtr,
                dftfe::utils::DEVICEBLAS_OP_N,
                dftfe::utils::DEVICEBLAS_OP_N,
                d_nQuadsPerCell * d_nVectors,
                3,
                3,
                &scalarCoeffAlpha,
                tempQuadratureGradientsData.data(),
                d_nQuadsPerCell * d_nVectors,
                d_nQuadsPerCell * d_nVectors * 3,
                d_inverseJacobianData.data() + 9 * cellRange.first,
                3,
                9,
                &scalarCoeffBeta,
                quadratureGradients->data(),
                d_nQuadsPerCell * d_nVectors,
                d_nVectors * d_nQuadsPerCell * 3,
                cellRange.second - cellRange.first);
            }
          else
            {
              dftfe::utils::deviceBlasWrapper::gemmStridedBatched(
                *d_deviceBlasHandlePtr,
                dftfe::utils::DEVICEBLAS_OP_N,
                dftfe::utils::DEVICEBLAS_OP_N,
                d_nVectors,
                3,
                3,
                &scalarCoeffAlpha,
                tempQuadratureGradientsData.data(),
                d_nVectors,
                d_nVectors * 3,
                d_inverseJacobianData.data() +
                  9 * cellRange.first * d_nQuadsPerCell,
                3,
                9,
                &scalarCoeffBeta,
                tempQuadratureGradientsDataNonAffine.data(),
                d_nVectors,
                d_nVectors * 3,
                (cellRange.second - cellRange.first) * d_nQuadsPerCell);
              dftfe::utils::deviceKernelsGeneric::stridedCopyToBlock(
                d_nVectors,
                (cellRange.second - cellRange.first) * d_nQuadsPerCell * 3,
                tempQuadratureGradientsDataNonAffine.data(),
                quadratureGradients->data(),
                d_nonAffineReshapeIDs.data() +
                  cellRange.first * d_nDofsPerCell);
            }
        }
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::DEVICE>::
      integrateWithBasisKernel(
        const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          *quadratureValues,
        const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          *quadratureGradients,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &                                         nodalData,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {}

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::DEVICE>::
      extractToCellNodalDataKernel(
        const dftfe::linearAlgebra::MultiVector<
          ValueTypeBasisCoeff,
          dftfe::utils::MemorySpace::DEVICE> &nodalData,
        dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                    dftfe::utils::MemorySpace::DEVICE>
          *                                         cellNodalDataPtr,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      dftfe::utils::deviceKernelsGeneric::stridedCopyToBlock(
        d_nVectors,
        (cellRange.second - cellRange.first) * d_nDofsPerCell,
        nodalData.data(),
        cellNodalDataPtr->data(),
        d_flattenedCellDofIndexToProcessDofIndexMap.data() +
          cellRange.first * d_nDofsPerCell);
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::DEVICE>::
      accumulateFromCellNodalDataKernel(
        const dftfe::utils::MemoryStorage<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &                                         nodalData,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      dftfe::utils::deviceKernelsGeneric::axpyStridedBlockAtomicAdd(
        d_nVectors,
        (cellRange.second - cellRange.first) * d_nDofsPerCell,
        cellNodalDataPtr->begin(),
        nodalData.begin(),
        d_flattenedCellDofIndexToProcessDofIndexMap.begin() +
          cellRange.first * d_nDofsPerCell);
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::DEVICE>::
      setDeviceBLASHandle(dftfe::utils::deviceBlasHandle_t *deviceBlasHandlePtr)
    {
      d_deviceBlasHandlePtr = deviceBlasHandlePtr;
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    dftfe::utils::deviceBlasHandle_t &
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::DEVICE>::getDeviceBLASHandle()
    {
      return *d_deviceBlasHandlePtr;
    }

  } // namespace basis
} // namespace dftfe
