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

#ifdef DFTFE_WITH_DEVICE_CUDA
#  include <DeviceKernelLauncherConstants.h>
#  include <DeviceDataTypeOverloads.h>
#  include <MPICommunicatorP2PKernels.h>
#  include <Exceptions.h>
#  include <complex>
#  include <algorithm>


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

            dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                      intraBlockId] =
              dftfe::utils::add(
                dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
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
      gatherSendBufferDeviceKernel<<<
        ownedLocalIndicesForTargetProcs.size() / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        ownedLocalIndicesForTargetProcs.size(),
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(dataArray.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          ownedLocalIndicesForTargetProcs.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(sendBuffer.data()));
    }


    template <typename ValueType>
    void
    MPICommunicatorP2PKernels<ValueType, utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<ValueType, utils::MemorySpace::DEVICE> &recvBuffer,
        const utils::MemoryStorage<size_type, utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE> &dataArray)
    {
      accumAddFromRecvBufferDeviceKernel<<<
        ownedLocalIndicesForTargetProcs.size() / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        ownedLocalIndicesForTargetProcs.size(),
        blockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(recvBuffer.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          ownedLocalIndicesForTargetProcs.data()),
        dftfe::utils::makeDataTypeDeviceCompatible(dataArray.data()));
    }


    template class MPICommunicatorP2PKernels<
      double,
      dftfe::utils::MemorySpace::DEVICE>;
    template class MPICommunicatorP2PKernels<
      float,
      dftfe::utils::MemorySpace::DEVICE>;
    template class MPICommunicatorP2PKernels<
      std::complex<double>,
      dftfe::utils::MemorySpace::DEVICE>;
    template class MPICommunicatorP2PKernels<
      std::complex<float>,
      dftfe::utils::MemorySpace::DEVICE>;

  } // namespace utils
} // namespace dftfe
#endif
