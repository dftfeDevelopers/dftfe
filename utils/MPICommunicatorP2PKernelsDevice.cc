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
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
#  include <BLASWrapper.h>

namespace dftfe
{
  namespace utils
  {
    namespace
    {
      template <typename ValueType1, typename ValueType2>
      __global__ void
      gatherSendBufferDeviceKernel(
        const dftfe::uInt  totalFlattenedSize,
        const dftfe::uInt  blockSize,
        const ValueType1  *dataArray,
        const dftfe::uInt *ownedLocalIndicesForTargetProcs,
        ValueType2        *sendBuffer)
      {
        const dftfe::uInt globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt blockId      = i / blockSize;
            const dftfe::uInt intraBlockId = i - blockId * blockSize;

            sendBuffer[i] =
              dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                        intraBlockId];
          }
      }

      template <>
      __global__ void
      gatherSendBufferDeviceKernel(
        const dftfe::uInt                        totalFlattenedSize,
        const dftfe::uInt                        blockSize,
        const dftfe::utils::deviceDoubleComplex *dataArray,
        const dftfe::uInt                *ownedLocalIndicesForTargetProcs,
        dftfe::utils::deviceFloatComplex *sendBuffer)
      {
        const dftfe::uInt globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt blockId      = i / blockSize;
            const dftfe::uInt intraBlockId = i - blockId * blockSize;
            sendBuffer[i].x =
              dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                        intraBlockId]
                .x;
            sendBuffer[i].y =
              dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                        intraBlockId]
                .y;
          }
      }

      template <typename ValueType1, typename ValueType2>
      __global__ void
      accumAddFromRecvBufferDeviceKernel(
        const dftfe::uInt  totalFlattenedSize,
        const dftfe::uInt  blockSize,
        const ValueType1  *recvBuffer,
        const dftfe::uInt *ownedLocalIndicesForTargetProcs,
        ValueType2        *dataArray)
      {
        const dftfe::uInt globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt blockId      = i / blockSize;
            const dftfe::uInt intraBlockId = i - blockId * blockSize;
            const ValueType2  recvVal      = recvBuffer[i];
            atomicAdd(
              &dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                         intraBlockId],
              recvVal);
          }
      }

      template <>
      __global__ void
      accumAddFromRecvBufferDeviceKernel(
        const dftfe::uInt                       totalFlattenedSize,
        const dftfe::uInt                       blockSize,
        const dftfe::utils::deviceFloatComplex *recvBuffer,
        const dftfe::uInt                      *ownedLocalIndicesForTargetProcs,
        dftfe::utils::deviceFloatComplex       *dataArray)
      {
        const dftfe::uInt globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt blockId      = i / blockSize;
            const dftfe::uInt intraBlockId = i - blockId * blockSize;

            atomicAdd(
              &dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                         intraBlockId]
                 .x,
              dftfe::utils::realPartDevice(recvBuffer[i]));
            atomicAdd(
              &dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                         intraBlockId]
                 .y,
              dftfe::utils::imagPartDevice(recvBuffer[i]));
          }
      }

      template <>
      __global__ void
      accumAddFromRecvBufferDeviceKernel(
        const dftfe::uInt                        totalFlattenedSize,
        const dftfe::uInt                        blockSize,
        const dftfe::utils::deviceDoubleComplex *recvBuffer,
        const dftfe::uInt                 *ownedLocalIndicesForTargetProcs,
        dftfe::utils::deviceDoubleComplex *dataArray)
      {
        const dftfe::uInt globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt blockId      = i / blockSize;
            const dftfe::uInt intraBlockId = i - blockId * blockSize;

            atomicAdd(
              &dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                         intraBlockId]
                 .x,
              dftfe::utils::realPartDevice(recvBuffer[i]));
            atomicAdd(
              &dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                         intraBlockId]
                 .y,
              dftfe::utils::imagPartDevice(recvBuffer[i]));
          }
      }

      template <>
      __global__ void
      accumAddFromRecvBufferDeviceKernel(
        const dftfe::uInt                       totalFlattenedSize,
        const dftfe::uInt                       blockSize,
        const dftfe::utils::deviceFloatComplex *recvBuffer,
        const dftfe::uInt                      *ownedLocalIndicesForTargetProcs,
        dftfe::utils::deviceDoubleComplex      *dataArray)
      {
        const dftfe::uInt globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt blockId      = i / blockSize;
            const dftfe::uInt intraBlockId = i - blockId * blockSize;
            const double      recvValReal =
              dftfe::utils::realPartDevice(recvBuffer[i]);
            const double recvValImag =
              dftfe::utils::imagPartDevice(recvBuffer[i]);

            atomicAdd(
              &dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                         intraBlockId]
                 .x,
              recvValReal);
            atomicAdd(
              &dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                         intraBlockId]
                 .y,
              recvValImag);
          }
      }

      template <typename ValueType1, typename ValueType2>
      __global__ void
      accumInsertFromRecvBufferDeviceKernel(
        const dftfe::uInt  totalFlattenedSize,
        const dftfe::uInt  blockSize,
        const ValueType1  *recvBuffer,
        const dftfe::uInt *ownedLocalIndicesForTargetProcs,
        ValueType2        *dataArray)
      {
        const dftfe::uInt globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt blockId      = i / blockSize;
            const dftfe::uInt intraBlockId = i - blockId * blockSize;
            const ValueType2  recvVal      = recvBuffer[i];

            dftfe::utils::copyValue(
              &dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                         intraBlockId],
              recvVal);
          }
      }
      /*
                template <>
                __global__ void
                accumInsertFromRecvBufferDeviceKernel(
                  const dftfe::uInt                         totalFlattenedSize,
                  const dftfe::uInt                         blockSize,
                  const dftfe::utils::deviceFloatComplex *recvBuffer,
                  const dftfe::uInt * ownedLocalIndicesForTargetProcs,
                  dftfe::utils::deviceFloatComplex *      dataArray)
                {
                  const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x +
             threadIdx.x; for (dftfe::uInt i = globalThreadId; i <
         totalFlattenedSize; i += blockDim.x * gridDim.x)
                    {
                      const dftfe::uInt blockId      = i / blockSize;
                      const dftfe::uInt intraBlockId = i - blockId * blockSize;

                      dftfe::utils::copyValue(
                        &dataArray[ownedLocalIndicesForTargetProcs[blockId] *
             blockSize + intraBlockId] .x,
                        dftfe::utils::realPartDevice(recvBuffer[i]));

                      dftfe::utils::copyValue(
                        &dataArray[ownedLocalIndicesForTargetProcs[blockId] *
             blockSize + intraBlockId] .y,
                        dftfe::utils::imagPartDevice(recvBuffer[i]));
                    }
                }

                template <>
                __global__ void
                accumInsertFromRecvBufferDeviceKernel(
                  const dftfe::uInt                          totalFlattenedSize,
                  const dftfe::uInt                          blockSize,
                  const dftfe::utils::deviceDoubleComplex *recvBuffer,
                  const dftfe::uInt * ownedLocalIndicesForTargetProcs,
                  dftfe::utils::deviceDoubleComplex *dataArray)
                {
                  const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x +
             threadIdx.x; for (dftfe::uInt i = globalThreadId; i <
         totalFlattenedSize; i += blockDim.x * gridDim.x)
                    {
                      const dftfe::uInt blockId      = i / blockSize;
                      const dftfe::uInt intraBlockId = i - blockId * blockSize;

                      dftfe::utils::copyValue(
                        &dataArray[ownedLocalIndicesForTargetProcs[blockId] *
             blockSize + intraBlockId] .x,
                        dftfe::utils::realPartDevice(recvBuffer[i]));

                      dftfe::utils::copyValue(
                        &dataArray[ownedLocalIndicesForTargetProcs[blockId] *
             blockSize + intraBlockId] .y,
                        dftfe::utils::imagPartDevice(recvBuffer[i]));
                    }
                }
    */
      template <>
      __global__ void
      accumInsertFromRecvBufferDeviceKernel(
        const dftfe::uInt                       totalFlattenedSize,
        const dftfe::uInt                       blockSize,
        const dftfe::utils::deviceFloatComplex *recvBuffer,
        const dftfe::uInt                      *ownedLocalIndicesForTargetProcs,
        dftfe::utils::deviceDoubleComplex      *dataArray)
      {
        const dftfe::uInt globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
             i += blockDim.x * gridDim.x)
          {
            const dftfe::uInt blockId      = i / blockSize;
            const dftfe::uInt intraBlockId = i - blockId * blockSize;
            const double      recvValReal =
              dftfe::utils::realPartDevice(recvBuffer[i]);
            const double recvValImag =
              dftfe::utils::imagPartDevice(recvBuffer[i]);

            dftfe::utils::copyValue(
              &dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                         intraBlockId]
                 .x,
              recvValReal);

            dftfe::utils::copyValue(
              &dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                         intraBlockId]
                 .y,
              recvValImag);
          }
      }


    } // namespace

    template <typename ValueType>
    template <typename ValueTypeComm>
    void
    MPICommunicatorP2PKernels<ValueType, utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<ValueType, utils::MemorySpace::DEVICE> &dataArray,
        const MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        MemoryStorage<ValueTypeComm, utils::MemorySpace::DEVICE> &sendBuffer,
        dftfe::utils::deviceStream_t deviceCommStream)
    {
      DFTFE_LAUNCH_KERNEL(
        gatherSendBufferDeviceKernel,
        (ownedLocalIndicesForTargetProcs.size() * blockSize) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        deviceCommStream,
        ownedLocalIndicesForTargetProcs.size() * blockSize,
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(dataArray.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          ownedLocalIndicesForTargetProcs.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(sendBuffer.data()));
    }

    template <typename ValueType>
    template <typename ValueTypeComm>
    void
    MPICommunicatorP2PKernels<ValueType, utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<ValueTypeComm, utils::MemorySpace::DEVICE>
          &recvBuffer,
        const utils::MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        const dftfe::uInt locallyOwnedSize,
        const dftfe::uInt ghostSize,
        MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE> &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream)
    {
      DFTFE_LAUNCH_KERNEL(
        accumAddFromRecvBufferDeviceKernel,
        (ownedLocalIndicesForTargetProcs.size() * blockSize) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        deviceCommStream,
        ownedLocalIndicesForTargetProcs.size() * blockSize,
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(recvBuffer.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          ownedLocalIndicesForTargetProcs.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(dataArray.data()));
    }

    template <typename ValueType>
    template <typename ValueTypeComm>
    void
    MPICommunicatorP2PKernels<ValueType, utils::MemorySpace::DEVICE>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<ValueTypeComm, utils::MemorySpace::DEVICE>
          &recvBuffer,
        const utils::MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        const dftfe::uInt locallyOwnedSize,
        const dftfe::uInt ghostSize,
        MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE> &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream)
    {
      DFTFE_LAUNCH_KERNEL(
        accumInsertFromRecvBufferDeviceKernel,
        (ownedLocalIndicesForTargetProcs.size() * blockSize) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        deviceCommStream,
        ownedLocalIndicesForTargetProcs.size() * blockSize,
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(recvBuffer.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          ownedLocalIndicesForTargetProcs.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(dataArray.data()));
    }

    template <typename ValueType>
    template <typename ValueType1, typename ValueType2>
    void
    MPICommunicatorP2PKernels<ValueType, utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const dftfe::uInt            blockSize,
        const ValueType1            *type1Array,
        ValueType2                  *type2Array,
        dftfe::utils::deviceStream_t deviceCommStream)
    {
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
        copyValueType1ArrToValueType2ArrDeviceCall(blockSize,
                                                   type1Array,
                                                   type2Array,
                                                   deviceCommStream);
    }

    template class MPICommunicatorP2PKernels<double,
                                             dftfe::utils::MemorySpace::DEVICE>;
    template class MPICommunicatorP2PKernels<float,
                                             dftfe::utils::MemorySpace::DEVICE>;
    template class MPICommunicatorP2PKernels<std::complex<double>,
                                             dftfe::utils::MemorySpace::DEVICE>;
    template class MPICommunicatorP2PKernels<std::complex<float>,
                                             dftfe::utils::MemorySpace::DEVICE>;
    template void
    MPICommunicatorP2PKernels<double, utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<double, utils::MemorySpace::DEVICE> &dataArray,
        const MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        MemoryStorage<double, utils::MemorySpace::DEVICE> &sendBuffer,
        dftfe::utils::deviceStream_t                       deviceCommStream);

    template void
    MPICommunicatorP2PKernels<double, utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<double, utils::MemorySpace::DEVICE> &dataArray,
        const MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        MemoryStorage<float, utils::MemorySpace::DEVICE> &sendBuffer,
        dftfe::utils::deviceStream_t                      deviceCommStream);

    template void
    MPICommunicatorP2PKernels<float, utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<float, utils::MemorySpace::DEVICE> &dataArray,
        const MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        MemoryStorage<float, utils::MemorySpace::DEVICE> &sendBuffer,
        dftfe::utils::deviceStream_t                      deviceCommStream);

    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<std::complex<double>, utils::MemorySpace::DEVICE>
          &dataArray,
        const MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        MemoryStorage<std::complex<double>, utils::MemorySpace::DEVICE>
                                    &sendBuffer,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<std::complex<double>, utils::MemorySpace::DEVICE>
          &dataArray,
        const MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        MemoryStorage<std::complex<float>, utils::MemorySpace::DEVICE>
                                    &sendBuffer,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<std::complex<float>, utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<std::complex<float>, utils::MemorySpace::DEVICE>
          &dataArray,
        const MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        MemoryStorage<std::complex<float>, utils::MemorySpace::DEVICE>
                                    &sendBuffer,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<double, utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<double, utils::MemorySpace::DEVICE> &recvBuffer,
        const utils::MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        const dftfe::uInt locallyOwnedSize,
        const dftfe::uInt ghostSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<double, utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<float, utils::MemorySpace::DEVICE> &recvBuffer,
        const utils::MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        const dftfe::uInt locallyOwnedSize,
        const dftfe::uInt ghostSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<float, utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<float, utils::MemorySpace::DEVICE> &recvBuffer,
        const utils::MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        const dftfe::uInt locallyOwnedSize,
        const dftfe::uInt ghostSize,
        MemoryStorage<float, dftfe::utils::MemorySpace::DEVICE> &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<double>, utils::MemorySpace::DEVICE>
          &recvBuffer,
        const utils::MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        const dftfe::uInt locallyOwnedSize,
        const dftfe::uInt ghostSize,
        MemoryStorage<std::complex<double>, dftfe::utils::MemorySpace::DEVICE>
                                    &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<float>, utils::MemorySpace::DEVICE>
          &recvBuffer,
        const utils::MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        const dftfe::uInt locallyOwnedSize,
        const dftfe::uInt ghostSize,
        MemoryStorage<std::complex<double>, dftfe::utils::MemorySpace::DEVICE>
                                    &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<std::complex<float>, utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<float>, utils::MemorySpace::DEVICE>
          &recvBuffer,
        const utils::MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        const dftfe::uInt locallyOwnedSize,
        const dftfe::uInt ghostSize,
        MemoryStorage<std::complex<float>, dftfe::utils::MemorySpace::DEVICE>
                                    &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<double, utils::MemorySpace::DEVICE>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<double, utils::MemorySpace::DEVICE> &recvBuffer,
        const utils::MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        const dftfe::uInt locallyOwnedSize,
        const dftfe::uInt ghostSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<double, utils::MemorySpace::DEVICE>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<float, utils::MemorySpace::DEVICE> &recvBuffer,
        const utils::MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        const dftfe::uInt locallyOwnedSize,
        const dftfe::uInt ghostSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<float, utils::MemorySpace::DEVICE>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<float, utils::MemorySpace::DEVICE> &recvBuffer,
        const utils::MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        const dftfe::uInt locallyOwnedSize,
        const dftfe::uInt ghostSize,
        MemoryStorage<float, dftfe::utils::MemorySpace::DEVICE> &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              utils::MemorySpace::DEVICE>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<double>, utils::MemorySpace::DEVICE>
          &recvBuffer,
        const utils::MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        const dftfe::uInt locallyOwnedSize,
        const dftfe::uInt ghostSize,
        MemoryStorage<std::complex<double>, dftfe::utils::MemorySpace::DEVICE>
                                    &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              utils::MemorySpace::DEVICE>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<float>, utils::MemorySpace::DEVICE>
          &recvBuffer,
        const utils::MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        const dftfe::uInt locallyOwnedSize,
        const dftfe::uInt ghostSize,
        MemoryStorage<std::complex<double>, dftfe::utils::MemorySpace::DEVICE>
                                    &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<std::complex<float>, utils::MemorySpace::DEVICE>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<float>, utils::MemorySpace::DEVICE>
          &recvBuffer,
        const utils::MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        const dftfe::uInt locallyOwnedSize,
        const dftfe::uInt ghostSize,
        MemoryStorage<std::complex<float>, dftfe::utils::MemorySpace::DEVICE>
                                    &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const dftfe::uInt            blockSize,
        const double                *type1Array,
        float                       *type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const dftfe::uInt            blockSize,
        const float                 *type1Array,
        double                      *type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const dftfe::uInt            blockSize,
        const std::complex<double>  *type1Array,
        std::complex<float>         *type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const dftfe::uInt            blockSize,
        const std::complex<float>   *type1Array,
        std::complex<double>        *type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);
    template void
    MPICommunicatorP2PKernels<float, dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const dftfe::uInt            blockSize,
        const float                 *type1Array,
        float                       *type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<std::complex<float>,
                              dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const dftfe::uInt            blockSize,
        const std::complex<float>   *type1Array,
        std::complex<float>         *type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);

  } // namespace utils
} // namespace dftfe
#endif
