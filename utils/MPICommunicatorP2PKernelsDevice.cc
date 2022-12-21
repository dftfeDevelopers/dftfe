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
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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


/*
 * @author Sambit Das.
 */

#ifdef DFTFE_WITH_DEVICE
#  include <DeviceKernelLauncherConstants.h>
#  include <DeviceDataTypeOverloads.h>
#  include <MPICommunicatorP2PKernels.h>
#  include <Exceptions.h>
#  include <complex>
#  include <algorithm>
#  include <deviceKernelsGeneric.h>


namespace dftfe
{
  namespace utils
  {
    namespace
    {
      template <typename ValueType>
      __global__ void
      gatherSendBufferDeviceKernel(
        const size_type  totalFlattenedSize,
        const size_type  blockSize,
        const ValueType *dataArray,
        const size_type *ownedLocalIndicesForTargetProcs,
        ValueType *      sendBuffer)
      {
        const size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
        for (size_type i = globalThreadId; i < totalFlattenedSize;
             i += blockDim.x * gridDim.x)
          {
            const size_type blockId      = i / blockSize;
            const size_type intraBlockId = i - blockId * blockSize;

            sendBuffer[i] =
              dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                        intraBlockId];
          }
      }

      __global__ void
      accumAddFromRecvBufferDeviceKernel(
        const size_type  totalFlattenedSize,
        const size_type  blockSize,
        const double *   recvBuffer,
        const size_type *ownedLocalIndicesForTargetProcs,
        double *         dataRealArray,
        double *         dataImagArray)
      {}

      __global__ void
      accumAddFromRecvBufferDeviceKernel(
        const size_type  totalFlattenedSize,
        const size_type  blockSize,
        const float *    recvBuffer,
        const size_type *ownedLocalIndicesForTargetProcs,
        float *          dataRealArray,
        float *          dataImagArray)
      {}

      __global__ void
      accumAddFromRecvBufferDeviceKernel(
        const size_type                          totalFlattenedSize,
        const size_type                          blockSize,
        const dftfe::utils::deviceDoubleComplex *recvBuffer,
        const size_type *ownedLocalIndicesForTargetProcs,
        double *         dataRealArray,
        double *         dataImagArray)
      {
        const size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
        for (size_type i = globalThreadId; i < totalFlattenedSize;
             i += blockDim.x * gridDim.x)
          {
            const size_type blockId      = i / blockSize;
            const size_type intraBlockId = i - blockId * blockSize;

            atomicAdd(&dataRealArray[ownedLocalIndicesForTargetProcs[blockId] *
                                       blockSize +
                                     intraBlockId],
                      dftfe::utils::realPartDevice(recvBuffer[i]));
            atomicAdd(&dataImagArray[ownedLocalIndicesForTargetProcs[blockId] *
                                       blockSize +
                                     intraBlockId],
                      dftfe::utils::imagPartDevice(recvBuffer[i]));
          }
      }

      __global__ void
      accumAddFromRecvBufferDeviceKernel(
        const size_type                         totalFlattenedSize,
        const size_type                         blockSize,
        const dftfe::utils::deviceFloatComplex *recvBuffer,
        const size_type *                       ownedLocalIndicesForTargetProcs,
        float *                                 dataRealArray,
        float *                                 dataImagArray)
      {
        const size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
        for (size_type i = globalThreadId; i < totalFlattenedSize;
             i += blockDim.x * gridDim.x)
          {
            const size_type blockId      = i / blockSize;
            const size_type intraBlockId = i - blockId * blockSize;

            atomicAdd(&dataRealArray[ownedLocalIndicesForTargetProcs[blockId] *
                                       blockSize +
                                     intraBlockId],
                      dftfe::utils::realPartDevice(recvBuffer[i]));
            atomicAdd(&dataImagArray[ownedLocalIndicesForTargetProcs[blockId] *
                                       blockSize +
                                     intraBlockId],
                      dftfe::utils::imagPartDevice(recvBuffer[i]));
          }
      }

      template <typename ValueType>
      __global__ void
      accumAddFromRecvBufferDeviceKernel(
        const size_type  totalFlattenedSize,
        const size_type  blockSize,
        const ValueType *recvBuffer,
        const size_type *ownedLocalIndicesForTargetProcs,
        ValueType *      dataArray)
      {
        const size_type globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
        for (size_type i = globalThreadId; i < totalFlattenedSize;
             i += blockDim.x * gridDim.x)
          {
            const size_type blockId      = i / blockSize;
            const size_type intraBlockId = i - blockId * blockSize;

            atomicAdd(
              &dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                         intraBlockId],
              recvBuffer[i]);
          }
      }
    } // namespace

    template <typename ValueType>
    void
    MPICommunicatorP2PKernels<ValueType, utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<ValueType, utils::MemorySpace::DEVICE> &dataArray,
        const MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        MemoryStorage<ValueType, utils::MemorySpace::DEVICE> &sendBuffer)
    {
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      gatherSendBufferDeviceKernel<<<(ownedLocalIndicesForTargetProcs.size() *
                                      blockSize) /
                                         dftfe::utils::DEVICE_BLOCK_SIZE +
                                       1,
                                     dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        ownedLocalIndicesForTargetProcs.size() * blockSize,
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(dataArray.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          ownedLocalIndicesForTargetProcs.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(sendBuffer.data()));
#  elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        gatherSendBufferDeviceKernel,
        (ownedLocalIndicesForTargetProcs.size() * blockSize) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        ownedLocalIndicesForTargetProcs.size() * blockSize,
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(dataArray.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          ownedLocalIndicesForTargetProcs.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(sendBuffer.data()));
#  endif
    }

    template <>
    void
    MPICommunicatorP2PKernels<std::complex<double>,
                              utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<double>, utils::MemorySpace::DEVICE>
          &recvBuffer,
        const utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        const size_type locallyOwnedSize,
        const size_type ghostSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          &tempDoubleRealDataArray,
        MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          &tempDoubleImagDataArray,
        MemoryStorage<float, dftfe::utils::MemorySpace::DEVICE>
          &tempFloatRealDataArray,
        MemoryStorage<float, dftfe::utils::MemorySpace::DEVICE>
          &tempFloatImagDataArray,
        MemoryStorage<std::complex<double>, dftfe::utils::MemorySpace::DEVICE>
          &dataArray)
    {
      deviceKernelsGeneric::copyComplexArrToRealArrsDevice(
        (locallyOwnedSize + ghostSize) * blockSize,
        dataArray.data(),
        tempDoubleRealDataArray.data(),
        tempDoubleImagDataArray.data());

#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      accumAddFromRecvBufferDeviceKernel<<<
        (ownedLocalIndicesForTargetProcs.size() * blockSize) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        ownedLocalIndicesForTargetProcs.size() * blockSize,
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(recvBuffer.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          ownedLocalIndicesForTargetProcs.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          tempDoubleRealDataArray.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          tempDoubleImagDataArray.data()));
#  elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(accumAddFromRecvBufferDeviceKernel,
                         (ownedLocalIndicesForTargetProcs.size() * blockSize) /
                             dftfe::utils::DEVICE_BLOCK_SIZE +
                           1,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         ownedLocalIndicesForTargetProcs.size() * blockSize,
                         blockSize,
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           recvBuffer.data()),
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           ownedLocalIndicesForTargetProcs.data()),
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           tempDoubleRealDataArray.data()),
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           tempDoubleImagDataArray.data()));
#  endif

      deviceKernelsGeneric::copyRealArrsToComplexArrDevice(
        locallyOwnedSize * blockSize,
        tempDoubleRealDataArray.data(),
        tempDoubleImagDataArray.data(),
        dataArray.data());
    }

    template <>
    void
    MPICommunicatorP2PKernels<std::complex<float>, utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<float>, utils::MemorySpace::DEVICE>
          &recvBuffer,
        const utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        const size_type locallyOwnedSize,
        const size_type ghostSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          &tempDoubleRealDataArray,
        MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          &tempDoubleImagDataArray,
        MemoryStorage<float, dftfe::utils::MemorySpace::DEVICE>
          &tempFloatRealDataArray,
        MemoryStorage<float, dftfe::utils::MemorySpace::DEVICE>
          &tempFloatImagDataArray,
        MemoryStorage<std::complex<float>, dftfe::utils::MemorySpace::DEVICE>
          &dataArray)
    {
      deviceKernelsGeneric::copyComplexArrToRealArrsDevice(
        (locallyOwnedSize + ghostSize) * blockSize,
        dataArray.data(),
        tempFloatRealDataArray.data(),
        tempFloatImagDataArray.data());

#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      accumAddFromRecvBufferDeviceKernel<<<
        (ownedLocalIndicesForTargetProcs.size() * blockSize) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        ownedLocalIndicesForTargetProcs.size() * blockSize,
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(recvBuffer.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          ownedLocalIndicesForTargetProcs.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          tempFloatRealDataArray.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          tempFloatImagDataArray.data()));
#  elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(accumAddFromRecvBufferDeviceKernel,
                         (ownedLocalIndicesForTargetProcs.size() * blockSize) /
                             dftfe::utils::DEVICE_BLOCK_SIZE +
                           1,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         ownedLocalIndicesForTargetProcs.size() * blockSize,
                         blockSize,
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           recvBuffer.data()),
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           ownedLocalIndicesForTargetProcs.data()),
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           tempFloatRealDataArray.data()),
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           tempFloatImagDataArray.data()));
#  endif

      deviceKernelsGeneric::copyRealArrsToComplexArrDevice(
        locallyOwnedSize * blockSize,
        tempFloatRealDataArray.data(),
        tempFloatImagDataArray.data(),
        dataArray.data());
    }

    template <>
    void
    MPICommunicatorP2PKernels<double, utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<double, utils::MemorySpace::DEVICE> &recvBuffer,
        const utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        const size_type locallyOwnedSize,
        const size_type ghostSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          &tempDoubleRealDataArray,
        MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          &tempDoubleImagDataArray,
        MemoryStorage<float, dftfe::utils::MemorySpace::DEVICE>
          &tempFloatRealDataArray,
        MemoryStorage<float, dftfe::utils::MemorySpace::DEVICE>
          &tempFloatImagDataArray,
        MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> &dataArray)
    {
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      accumAddFromRecvBufferDeviceKernel<<<
        (ownedLocalIndicesForTargetProcs.size() * blockSize) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        ownedLocalIndicesForTargetProcs.size() * blockSize,
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(recvBuffer.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          ownedLocalIndicesForTargetProcs.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(dataArray.data()));
#  elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        accumAddFromRecvBufferDeviceKernel,
        (ownedLocalIndicesForTargetProcs.size() * blockSize) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        ownedLocalIndicesForTargetProcs.size() * blockSize,
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(recvBuffer.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          ownedLocalIndicesForTargetProcs.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(dataArray.data()));
#  endif
    }

    template <>
    void
    MPICommunicatorP2PKernels<float, utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<float, utils::MemorySpace::DEVICE> &recvBuffer,
        const utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        const size_type locallyOwnedSize,
        const size_type ghostSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          &tempDoubleRealDataArray,
        MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          &tempDoubleImagDataArray,
        MemoryStorage<float, dftfe::utils::MemorySpace::DEVICE>
          &tempFloatRealDataArray,
        MemoryStorage<float, dftfe::utils::MemorySpace::DEVICE>
          &tempFloatImagDataArray,
        MemoryStorage<float, dftfe::utils::MemorySpace::DEVICE> &dataArray)
    {
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      accumAddFromRecvBufferDeviceKernel<<<
        (ownedLocalIndicesForTargetProcs.size() * blockSize) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        ownedLocalIndicesForTargetProcs.size() * blockSize,
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(recvBuffer.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          ownedLocalIndicesForTargetProcs.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(dataArray.data()));
#  elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        accumAddFromRecvBufferDeviceKernel,
        (ownedLocalIndicesForTargetProcs.size() * blockSize) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        ownedLocalIndicesForTargetProcs.size() * blockSize,
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(recvBuffer.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          ownedLocalIndicesForTargetProcs.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(dataArray.data()));
#  endif
    }

    template class MPICommunicatorP2PKernels<double,
                                             dftfe::utils::MemorySpace::DEVICE>;
    template class MPICommunicatorP2PKernels<float,
                                             dftfe::utils::MemorySpace::DEVICE>;
    template class MPICommunicatorP2PKernels<std::complex<double>,
                                             dftfe::utils::MemorySpace::DEVICE>;
    template class MPICommunicatorP2PKernels<std::complex<float>,
                                             dftfe::utils::MemorySpace::DEVICE>;

  } // namespace utils
} // namespace dftfe
#endif
