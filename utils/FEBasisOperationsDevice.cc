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
#include <DeviceKernelLauncherConstants.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>

namespace dftfe
{
  namespace
  {
    template <typename ValueType1, typename ValueType2>
    __global__ void
    reshapeNonAffineCaseDeviceKernel(const dftfe::size_type numVecs,
                                     const dftfe::size_type numQuads,
                                     const dftfe::size_type numCells,
                                     const ValueType1 *     copyFromVec,
                                     ValueType2 *           copyToVec)
    {
      const dftfe::size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::size_type numberEntries = numQuads * numCells * numVecs * 3;

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::size_type blockIndex  = index / numVecs;
          dftfe::size_type iVec        = index - blockIndex * numVecs;
          dftfe::size_type blockIndex2 = blockIndex / numQuads;
          dftfe::size_type iQuad       = blockIndex - blockIndex2 * numQuads;
          dftfe::size_type iCell       = blockIndex2 / 3;
          dftfe::size_type iDim        = blockIndex2 - iCell * 3;
          dftfe::utils::copyValue(
            copyToVec + index,
            copyFromVec[iVec + iDim * numVecs + iQuad * 3 * numVecs +
                        iCell * 3 * numQuads * numVecs]);
        }
    }
  } // namespace

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
          &                  nodalData,
        ValueTypeBasisCoeff *quadratureValues,
        ValueTypeBasisCoeff *quadratureGradients) const
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
        ValueTypeBasisCoeff *quadratureValues,
        ValueTypeBasisCoeff *quadratureGradients,
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
          &                  nodalData,
        ValueTypeBasisCoeff *cellNodalDataPtr) const
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
        const ValueTypeBasisCoeff *cellNodalDataPtr,
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
          dftfe::utils::MemorySpace::DEVICE> &      nodalValues,
        ValueTypeBasisCoeff *                       quadratureValues,
        ValueTypeBasisCoeff *                       quadratureGradients,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      extractToCellNodalDataKernel(nodalValues,
                                   tempCellNodalData.data(),
                                   cellRange);
      interpolateKernel(tempCellNodalData.data(),
                        quadratureValues,
                        quadratureGradients,
                        cellRange);
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::DEVICE>::
      interpolateKernel(
        const ValueTypeBasisCoeff *                 cellNodalValues,
        ValueTypeBasisCoeff *                       quadratureValues,
        ValueTypeBasisCoeff *                       quadratureGradients,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      const ValueTypeBasisCoeff scalarCoeffAlpha = ValueTypeBasisCoeff(1.0),
                                scalarCoeffBeta  = ValueTypeBasisCoeff(0.0);

      if (quadratureValues != NULL)
        dftfe::utils::deviceBlasWrapper::gemmStridedBatched(
          *d_deviceBlasHandlePtr,
          dftfe::utils::DEVICEBLAS_OP_N,
          dftfe::utils::DEVICEBLAS_OP_N,
          d_nVectors,
          d_nQuadsPerCell[d_quadratureID],
          d_nDofsPerCell,
          &scalarCoeffAlpha,
          cellNodalValues,
          d_nVectors,
          d_nVectors * d_nDofsPerCell,
          d_shapeFunctionData[d_quadratureID].data(),
          d_nDofsPerCell,
          0,
          &scalarCoeffBeta,
          quadratureValues,
          d_nVectors,
          d_nVectors * d_nQuadsPerCell[d_quadratureID],
          cellRange.second - cellRange.first);
      if (quadratureGradients != NULL)
        {
          dftfe::utils::deviceBlasWrapper::gemmStridedBatched(
            *d_deviceBlasHandlePtr,
            dftfe::utils::DEVICEBLAS_OP_N,
            dftfe::utils::DEVICEBLAS_OP_N,
            d_nVectors,
            d_nQuadsPerCell[d_quadratureID] * 3,
            d_nDofsPerCell,
            &scalarCoeffAlpha,
            cellNodalValues,
            d_nVectors,
            d_nVectors * d_nDofsPerCell,
            d_shapeFunctionGradientDataInternalLayout[d_quadratureID].data(),
            d_nDofsPerCell,
            0,
            &scalarCoeffBeta,
            areAllCellsCartesian ? quadratureGradients :
                                   tempQuadratureGradientsData.data(),
            d_nVectors,
            d_nVectors * d_nQuadsPerCell[d_quadratureID] * 3,
            cellRange.second - cellRange.first);
          if (areAllCellsCartesian)
            {
              dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
                d_nQuadsPerCell[d_quadratureID] * d_nVectors,
                3 * (cellRange.second - cellRange.first),
                ValueTypeBasisCoeff(1.0),
                d_inverseJacobianData[0].data() + cellRange.first * 3,
                quadratureGradients);
            }
          else if (areAllCellsAffine)
            {
              dftfe::utils::deviceBlasWrapper::gemmStridedBatched(
                *d_deviceBlasHandlePtr,
                dftfe::utils::DEVICEBLAS_OP_N,
                dftfe::utils::DEVICEBLAS_OP_N,
                d_nQuadsPerCell[d_quadratureID] * d_nVectors,
                3,
                3,
                &scalarCoeffAlpha,
                tempQuadratureGradientsData.data(),
                d_nQuadsPerCell[d_quadratureID] * d_nVectors,
                d_nQuadsPerCell[d_quadratureID] * d_nVectors * 3,
                d_inverseJacobianData[0].data() + 9 * cellRange.first,
                3,
                9,
                &scalarCoeffBeta,
                quadratureGradients,
                d_nQuadsPerCell[d_quadratureID] * d_nVectors,
                d_nVectors * d_nQuadsPerCell[d_quadratureID] * 3,
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
                d_inverseJacobianData[d_quadratureID].data() +
                  9 * cellRange.first * d_nQuadsPerCell[d_quadratureID],
                3,
                9,
                &scalarCoeffBeta,
                tempQuadratureGradientsDataNonAffine.data(),
                d_nVectors,
                d_nVectors * 3,
                (cellRange.second - cellRange.first) *
                  d_nQuadsPerCell[d_quadratureID]);
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
              reshapeNonAffineCaseDeviceKernel<<<
                (d_nVectors * (cellRange.second - cellRange.first) *
                 d_nQuadsPerCell[d_quadratureID] * 3) /
                    dftfe::utils::DEVICE_BLOCK_SIZE +
                  1,
                dftfe::utils::DEVICE_BLOCK_SIZE>>>(
                d_nVectors,
                d_nQuadsPerCell[d_quadratureID],
                (cellRange.second - cellRange.first),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  tempQuadratureGradientsDataNonAffine.data()),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  quadratureGradients));
#elif DFTFE_WITH_DEVICE_LANG_HIP
              hipLaunchKernelGGL(reshapeNonAffineCaseDeviceKernel,
                                 (d_nVectors *
                                  (cellRange.second - cellRange.first) *
                                  d_nQuadsPerCell[d_quadratureID] * 3) /
                                     dftfe::utils::DEVICE_BLOCK_SIZE +
                                   1,
                                 dftfe::utils::DEVICE_BLOCK_SIZE,
                                 0,
                                 0,
                                 d_nVectors,
                                 d_nQuadsPerCell[d_quadratureID],
                                 (cellRange.second - cellRange.first),
                                 dftfe::utils::makeDataTypeDeviceCompatible(
                                   tempQuadratureGradientsDataNonAffine.data()),
                                 dftfe::utils::makeDataTypeDeviceCompatible(
                                   quadratureGradients), );
#endif
            }
        }
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::DEVICE>::
      integrateWithBasisKernel(
        const ValueTypeBasisCoeff *quadratureValues,
        const ValueTypeBasisCoeff *quadratureGradients,
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
          dftfe::utils::MemorySpace::DEVICE> &      nodalData,
        ValueTypeBasisCoeff *                       cellNodalDataPtr,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      dftfe::utils::deviceKernelsGeneric::stridedCopyToBlock(
        d_nVectors,
        (cellRange.second - cellRange.first) * d_nDofsPerCell,
        nodalData.data(),
        cellNodalDataPtr,
        d_flattenedCellDofIndexToProcessDofIndexMap.data() +
          cellRange.first * d_nDofsPerCell);
    }

    template <typename ValueTypeBasisCoeff, typename ValueTypeBasisData>
    void
    FEBasisOperations<ValueTypeBasisCoeff,
                      ValueTypeBasisData,
                      dftfe::utils::MemorySpace::DEVICE>::
      accumulateFromCellNodalDataKernel(
        const ValueTypeBasisCoeff *cellNodalDataPtr,
        dftfe::linearAlgebra::MultiVector<ValueTypeBasisCoeff,
                                          dftfe::utils::MemorySpace::DEVICE>
          &                                         nodalData,
        const std::pair<unsigned int, unsigned int> cellRange) const
    {
      dftfe::utils::deviceKernelsGeneric::axpyStridedBlockAtomicAdd(
        d_nVectors,
        (cellRange.second - cellRange.first) * d_nDofsPerCell,
        cellNodalDataPtr,
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
    template class FEBasisOperations<dataTypes::number,
                                     double,
                                     dftfe::utils::MemorySpace::DEVICE>;
#ifdef USE_COMPLEX
    template class FEBasisOperations<double,
                                     double,
                                     dftfe::utils::MemorySpace::DEVICE>;
#endif

  } // namespace basis
} // namespace dftfe
