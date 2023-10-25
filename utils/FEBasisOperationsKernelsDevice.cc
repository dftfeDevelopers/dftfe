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

#include <FEBasisOperationsKernelsDevice.h>
#include <DeviceAPICalls.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherConstants.h>
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
    namespace FEBasisOperationsKernelsDevice
    {
      template <typename ValueType1, typename ValueType2>
      void
      reshapeNonAffineCase(const dftfe::size_type numVecs,
                           const dftfe::size_type numQuads,
                           const dftfe::size_type numCells,
                           const ValueType1 *     copyFromVec,
                           ValueType2 *           copyToVec)
      {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
        reshapeNonAffineCaseDeviceKernel<<<(numVecs * numCells * numQuads * 3) /
                                               dftfe::utils::DEVICE_BLOCK_SIZE +
                                             1,
                                           dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          numVecs,
          numQuads,
          numCells,
          dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
          dftfe::utils::makeDataTypeDeviceCompatible(copyToVec));
#elif DFTFE_WITH_DEVICE_LANG_HIP
        hipLaunchKernelGGL(
          reshapeNonAffineCaseDeviceKernel,
          (numVecs * numCells * numQuads * 3) /
              dftfe::utils::DEVICE_BLOCK_SIZE +
            1,
          dftfe::utils::DEVICE_BLOCK_SIZE,
          0,
          0,
          numVecs,
          numQuads,
          numCells,
          dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
          dftfe::utils::makeDataTypeDeviceCompatible(copyToVec));
#endif
      }
      template void
      reshapeNonAffineCase(const dftfe::size_type numVecs,
                           const dftfe::size_type numQuads,
                           const dftfe::size_type numCells,
                           const double *         copyFromVec,
                           double *               copyToVec);
      template void
      reshapeNonAffineCase(const dftfe::size_type      numVecs,
                           const dftfe::size_type      numQuads,
                           const dftfe::size_type      numCells,
                           const std::complex<double> *copyFromVec,
                           std::complex<double> *      copyToVec);

    } // namespace FEBasisOperationsKernelsDevice
  }   // namespace basis
} // namespace dftfe
