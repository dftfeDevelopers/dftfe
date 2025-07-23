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
#  include <DeviceKernelLauncherHelpers.h>
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

      DFTFE_CREATE_KERNEL(
        void,
        gatherSendBufferDeviceKernel,
        {
          for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
               i += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockId      = i / blockSize;
              const dftfe::uInt intraBlockId = i - blockId * blockSize;

              sendBuffer[i] =
                dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                          intraBlockId];
            }
        },
        const dftfe::uInt  totalFlattenedSize,
        const dftfe::uInt  blockSize,
        const ValueType1  *dataArray,
        const dftfe::uInt *ownedLocalIndicesForTargetProcs,
        ValueType2        *sendBuffer);


      template <>

      DFTFE_CREATE_KERNEL(
        void,
        gatherSendBufferDeviceKernel,
        {
          for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
               i += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockId      = i / blockSize;
              const dftfe::uInt intraBlockId = i - blockId * blockSize;
              sendBuffer[i]                  = dftfe::utils::makeComplex(
                (float)dftfe::utils::realPartDevice(
                  dataArray[ownedLocalIndicesForTargetProcs[blockId] *
                              blockSize +
                            intraBlockId]),
                (float)dftfe::utils::imagPartDevice(
                  dataArray[ownedLocalIndicesForTargetProcs[blockId] *
                              blockSize +
                            intraBlockId]));
            }
        },
        const dftfe::uInt                        totalFlattenedSize,
        const dftfe::uInt                        blockSize,
        const dftfe::utils::deviceDoubleComplex *dataArray,
        const dftfe::uInt                *ownedLocalIndicesForTargetProcs,
        dftfe::utils::deviceFloatComplex *sendBuffer);


      template <typename ValueType1, typename ValueType2>

      DFTFE_CREATE_KERNEL(
        void,
        accumAddFromRecvBufferDeviceKernel,
        {
          for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
               i += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockId      = i / blockSize;
              const dftfe::uInt intraBlockId = i - blockId * blockSize;
              const ValueType2  recvVal      = recvBuffer[i];
              dftfe::utils::atomicAddWrapper(
                &dataArray[ownedLocalIndicesForTargetProcs[blockId] *
                             blockSize +
                           intraBlockId],
                recvVal);
            }
        },
        const dftfe::uInt  totalFlattenedSize,
        const dftfe::uInt  blockSize,
        const ValueType1  *recvBuffer,
        const dftfe::uInt *ownedLocalIndicesForTargetProcs,
        ValueType2        *dataArray);


      template <>

      DFTFE_CREATE_KERNEL(
        void,
        accumAddFromRecvBufferDeviceKernel,
        {
          for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
               i += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockId      = i / blockSize;
              const dftfe::uInt intraBlockId = i - blockId * blockSize;

              auto *add_real = reinterpret_cast<float *>(
                &dataArray[ownedLocalIndicesForTargetProcs[blockId] *
                             blockSize +
                           intraBlockId]);
              auto *add_imag = add_real + 1;

              dftfe::utils::atomicAddWrapper(
                add_real, dftfe::utils::realPartDevice(recvBuffer[i]));
              dftfe::utils::atomicAddWrapper(
                add_imag, dftfe::utils::imagPartDevice(recvBuffer[i]));
            }
        },
        const dftfe::uInt                       totalFlattenedSize,
        const dftfe::uInt                       blockSize,
        const dftfe::utils::deviceFloatComplex *recvBuffer,
        const dftfe::uInt                      *ownedLocalIndicesForTargetProcs,
        dftfe::utils::deviceFloatComplex       *dataArray);


      template <>

      DFTFE_CREATE_KERNEL(
        void,
        accumAddFromRecvBufferDeviceKernel,
        {
          for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
               i += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockId      = i / blockSize;
              const dftfe::uInt intraBlockId = i - blockId * blockSize;

              auto *add_real = reinterpret_cast<double *>(
                &dataArray[ownedLocalIndicesForTargetProcs[blockId] *
                             blockSize +
                           intraBlockId]);
              auto *add_imag = add_real + 1;

              dftfe::utils::atomicAddWrapper(
                add_real, dftfe::utils::realPartDevice(recvBuffer[i]));
              dftfe::utils::atomicAddWrapper(
                add_imag, dftfe::utils::imagPartDevice(recvBuffer[i]));
            }
        },
        const dftfe::uInt                        totalFlattenedSize,
        const dftfe::uInt                        blockSize,
        const dftfe::utils::deviceDoubleComplex *recvBuffer,
        const dftfe::uInt                 *ownedLocalIndicesForTargetProcs,
        dftfe::utils::deviceDoubleComplex *dataArray);


      template <>

      DFTFE_CREATE_KERNEL(
        void,
        accumAddFromRecvBufferDeviceKernel,
        {
          for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
               i += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockId      = i / blockSize;
              const dftfe::uInt intraBlockId = i - blockId * blockSize;
              const double      recvValReal =
                dftfe::utils::realPartDevice(recvBuffer[i]);
              const double recvValImag =
                dftfe::utils::imagPartDevice(recvBuffer[i]);

              auto *add_real = reinterpret_cast<double *>(
                &dataArray[ownedLocalIndicesForTargetProcs[blockId] *
                             blockSize +
                           intraBlockId]);
              auto *add_imag = add_real + 1;

              dftfe::utils::atomicAddWrapper(add_real, recvValReal);
              dftfe::utils::atomicAddWrapper(add_imag, recvValImag);
            }
        },
        const dftfe::uInt                       totalFlattenedSize,
        const dftfe::uInt                       blockSize,
        const dftfe::utils::deviceFloatComplex *recvBuffer,
        const dftfe::uInt                      *ownedLocalIndicesForTargetProcs,
        dftfe::utils::deviceDoubleComplex      *dataArray);


      template <typename ValueType1, typename ValueType2>

      DFTFE_CREATE_KERNEL(
        void,
        accumInsertFromRecvBufferDeviceKernel,
        {
          for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
               i += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockId      = i / blockSize;
              const dftfe::uInt intraBlockId = i - blockId * blockSize;
              const ValueType2  recvVal      = recvBuffer[i];

              dftfe::utils::copyValue(
                &dataArray[ownedLocalIndicesForTargetProcs[blockId] *
                             blockSize +
                           intraBlockId],
                recvVal);
            }
        },
        const dftfe::uInt  totalFlattenedSize,
        const dftfe::uInt  blockSize,
        const ValueType1  *recvBuffer,
        const dftfe::uInt *ownedLocalIndicesForTargetProcs,
        ValueType2        *dataArray);

      /*
                template <>

    DFTFE_CREATE_KERNEL(void, accumInsertFromRecvBufferDeviceKernel, {
                  for (dftfe::uInt i = globalThreadId; i <
         totalFlattenedSize; i += nThreadsPerBlock * nThreadBlock)
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
                },
                  const dftfe::uInt                         totalFlattenedSize,
                  const dftfe::uInt                         blockSize,
                  const dftfe::utils::deviceFloatComplex *recvBuffer,
                  const dftfe::uInt * ownedLocalIndicesForTargetProcs,
                  dftfe::utils::deviceFloatComplex *      dataArray);


                template <>

    DFTFE_CREATE_KERNEL(void, accumInsertFromRecvBufferDeviceKernel, {
                  for (dftfe::uInt i = globalThreadId; i <
         totalFlattenedSize; i += nThreadsPerBlock * nThreadBlock)
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
                },
                  const dftfe::uInt                          totalFlattenedSize,
                  const dftfe::uInt                          blockSize,
                  const dftfe::utils::deviceDoubleComplex *recvBuffer,
                  const dftfe::uInt * ownedLocalIndicesForTargetProcs,
                  dftfe::utils::deviceDoubleComplex *dataArray);

    */
      template <>

      DFTFE_CREATE_KERNEL(
        void,
        accumInsertFromRecvBufferDeviceKernel,
        {
          for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
               i += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockId      = i / blockSize;
              const dftfe::uInt intraBlockId = i - blockId * blockSize;
              const double      recvValReal =
                dftfe::utils::realPartDevice(recvBuffer[i]);
              const double recvValImag =
                dftfe::utils::imagPartDevice(recvBuffer[i]);

              dftfe::utils::copyValue(
                reinterpret_cast<double *>(
                  dataArray +
                  ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                  intraBlockId),
                recvValReal);

              dftfe::utils::copyValue(
                reinterpret_cast<double *>(
                  dataArray +
                  ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                  intraBlockId) +
                  1,
                recvValImag);
            }
        },
        const dftfe::uInt                       totalFlattenedSize,
        const dftfe::uInt                       blockSize,
        const dftfe::utils::deviceFloatComplex *recvBuffer,
        const dftfe::uInt                      *ownedLocalIndicesForTargetProcs,
        dftfe::utils::deviceDoubleComplex      *dataArray);



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
      const auto *dataArray_data =
        dftfe::utils::makeDataTypeDeviceCompatible(dataArray.data());
      const auto *ownedLocalIndicesForTargetProcs_data =
        dftfe::utils::makeDataTypeDeviceCompatible(
          ownedLocalIndicesForTargetProcs.data());
      auto *sendBuffer_data =
        dftfe::utils::makeDataTypeDeviceCompatible(sendBuffer.data());
      const dftfe::uInt numIndices =
        ownedLocalIndicesForTargetProcs.size() * blockSize;
      /*DFTFE_LAUNCH_KERNEL(gatherSendBufferDeviceKernel,
                          (numIndices) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          deviceCommStream,
                          numIndices,
                          blockSize,
                          dataArray_data,
                          ownedLocalIndicesForTargetProcs_data,
                          sendBuffer_data);*/
    }

    __global__ void
    accumAddFromRecvBufferHalfPrecDeviceKernel(
      const dftfe::uInt totalFlattenedSize,
      const dftfe::uInt blockSize,
      const typename dftfe::dataTypes::halfPrecType<double>::type *recvBuffer,
      const dftfe::uInt *ownedLocalIndicesForTargetProcs,
      double            *dataArray)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
           i += blockDim.x * gridDim.x)
        {
          const dftfe::uInt blockId      = i / blockSize;
          const dftfe::uInt intraBlockId = i - blockId * blockSize;
          const double      recvVal      = __bfloat162float(
            *(reinterpret_cast<const dftfe::utils::__device_bfloat16 *>(
                recvBuffer) +
              i));
          atomicAdd(
            &dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                       intraBlockId],
            recvVal);
        }
    }


    __global__ void
    accumAddFromRecvBufferHalfPrecDeviceKernel(
      const dftfe::uInt totalFlattenedSize,
      const dftfe::uInt blockSize,
      const typename dftfe::dataTypes::halfPrecType<double>::type *recvBuffer,
      const dftfe::uInt *ownedLocalIndicesForTargetProcs,
      float             *dataArray)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
           i += blockDim.x * gridDim.x)
        {
          const dftfe::uInt blockId      = i / blockSize;
          const dftfe::uInt intraBlockId = i - blockId * blockSize;
          const float       recvVal      = __bfloat162float(
            *(reinterpret_cast<const dftfe::utils::__device_bfloat16 *>(
                recvBuffer) +
              i));
          atomicAdd(
            &dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                       intraBlockId],
            recvVal);
        }
    }


    __global__ void
    accumAddFromRecvBufferHalfPrecDeviceKernel(
      const dftfe::uInt totalFlattenedSize,
      const dftfe::uInt blockSize,
      const typename dftfe::dataTypes::halfPrecType<std::complex<double>>::type
                          *recvBuffer,
      const dftfe::uInt   *ownedLocalIndicesForTargetProcs,
      deviceDoubleComplex *dataArray)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
           i += blockDim.x * gridDim.x)
        {
          const dftfe::uInt blockId      = i / blockSize;
          const dftfe::uInt intraBlockId = i - blockId * blockSize;
          atomicAdd(
            &dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                       intraBlockId]
               .x,
            (double)__bfloat162float(
              (*(reinterpret_cast<const dftfe::utils::__device_bfloat162 *>(
                   recvBuffer) +
                 i))
                .x));
          atomicAdd(
            &dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                       intraBlockId]
               .y,
            (double)__bfloat162float(
              (*(reinterpret_cast<const dftfe::utils::__device_bfloat162 *>(
                   recvBuffer) +
                 i))
                .y));
        }
    }
    __global__ void
    gatherSendBufferHalfPrecDeviceKernel(
      const dftfe::uInt  totalFlattenedSize,
      const dftfe::uInt  blockSize,
      const double      *dataArray,
      const dftfe::uInt *ownedLocalIndicesForTargetProcs,
      typename dftfe::dataTypes::halfPrecType<double>::type *sendBuffer)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
           i += blockDim.x * gridDim.x)
        {
          const dftfe::uInt blockId      = i / blockSize;
          const dftfe::uInt intraBlockId = i - blockId * blockSize;
          *(reinterpret_cast<dftfe::utils::__device_bfloat16 *>(sendBuffer) +
            i) =
            dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                      intraBlockId];
        }
    }

    __global__ void
    gatherSendBufferHalfPrecDeviceKernel(
      const dftfe::uInt  totalFlattenedSize,
      const dftfe::uInt  blockSize,
      const float       *dataArray,
      const dftfe::uInt *ownedLocalIndicesForTargetProcs,
      typename dftfe::dataTypes::halfPrecType<float>::type *sendBuffer)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
           i += blockDim.x * gridDim.x)
        {
          const dftfe::uInt blockId      = i / blockSize;
          const dftfe::uInt intraBlockId = i - blockId * blockSize;
          *(reinterpret_cast<dftfe::utils::__device_bfloat16 *>(sendBuffer) +
            i) =
            dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                      intraBlockId];
        }
    }


    __global__ void
    gatherSendBufferHalfPrecDeviceKernel(
      const dftfe::uInt          totalFlattenedSize,
      const dftfe::uInt          blockSize,
      const deviceDoubleComplex *dataArray,
      const dftfe::uInt         *ownedLocalIndicesForTargetProcs,
      typename dftfe::dataTypes::halfPrecType<std::complex<double>>::type
        *sendBuffer)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
           i += blockDim.x * gridDim.x)
        {
          const dftfe::uInt blockId      = i / blockSize;
          const dftfe::uInt intraBlockId = i - blockId * blockSize;
          (*(reinterpret_cast<dftfe::utils::__device_bfloat162 *>(sendBuffer) +
             i))
            .x =
            dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                      intraBlockId]
              .x;
          (*(reinterpret_cast<dftfe::utils::__device_bfloat162 *>(sendBuffer) +
             i))
            .y =
            dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                      intraBlockId]
              .y;
        }
    }


    __global__ void
    gatherSendBufferHalfPrecDeviceKernel(
      const dftfe::uInt         totalFlattenedSize,
      const dftfe::uInt         blockSize,
      const deviceFloatComplex *dataArray,
      const dftfe::uInt        *ownedLocalIndicesForTargetProcs,
      typename dftfe::dataTypes::halfPrecType<std::complex<float>>::type
        *sendBuffer)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
           i += blockDim.x * gridDim.x)
        {
          const dftfe::uInt blockId      = i / blockSize;
          const dftfe::uInt intraBlockId = i - blockId * blockSize;
          (*(reinterpret_cast<dftfe::utils::__device_bfloat162 *>(sendBuffer) +
             i))
            .x =
            dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                      intraBlockId]
              .x;
          (*(reinterpret_cast<dftfe::utils::__device_bfloat162 *>(sendBuffer) +
             i))
            .y =
            dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                      intraBlockId]
              .y;
        }
    }


    __global__ void
    accumAddFromRecvBufferHalfPrecDeviceKernel(
      const dftfe::uInt totalFlattenedSize,
      const dftfe::uInt blockSize,
      const typename dftfe::dataTypes::halfPrecType<std::complex<float>>::type
                         *recvBuffer,
      const dftfe::uInt  *ownedLocalIndicesForTargetProcs,
      deviceFloatComplex *dataArray)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      for (dftfe::uInt i = globalThreadId; i < totalFlattenedSize;
           i += blockDim.x * gridDim.x)
        {
          const dftfe::uInt blockId      = i / blockSize;
          const dftfe::uInt intraBlockId = i - blockId * blockSize;
          atomicAdd(
            &dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                       intraBlockId]
               .x,
            (float)__bfloat162float(
              (*(reinterpret_cast<const dftfe::utils::__device_bfloat162 *>(
                   recvBuffer) +
                 i))
                .x));
          atomicAdd(
            &dataArray[ownedLocalIndicesForTargetProcs[blockId] * blockSize +
                       intraBlockId]
               .y,
            (float)__bfloat162float(
              (*(reinterpret_cast<const dftfe::utils::__device_bfloat162 *>(
                   recvBuffer) +
                 i))
                .y));
        }
    }



    template <typename ValueType>
    void
    MPICommunicatorP2PKernels<ValueType, utils::MemorySpace::DEVICE>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcsHalfPrec(
        const MemoryStorage<
          typename dftfe::dataTypes::halfPrecType<ValueType>::type,
          utils::MemorySpace::DEVICE> &recvBuffer,
        const utils::MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        const dftfe::uInt locallyOwnedSize,
        const dftfe::uInt ghostSize,
        MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE> &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream)
    {
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      accumAddFromRecvBufferHalfPrecDeviceKernel<<<
        (ownedLocalIndicesForTargetProcs.size() * blockSize) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        deviceCommStream>>>(ownedLocalIndicesForTargetProcs.size() * blockSize,
                            blockSize,
                            recvBuffer.data(),
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              ownedLocalIndicesForTargetProcs.data()),
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              dataArray.data())); // CHECK
/*
#  elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(accumAddFromRecvBufferHalfPrecDeviceKernel,
                         (ownedLocalIndicesForTargetProcs.size() * blockSize) /
                             dftfe::utils::DEVICE_BLOCK_SIZE +
                           1,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         deviceCommStream,
                         ownedLocalIndicesForTargetProcs.size() * blockSize,
                         blockSize,
                         recvBuffer.data(),
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           ownedLocalIndicesForTargetProcs.data()),
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           dataArray.data()));*/
#  endif
    }

    template <typename ValueType>
    void
    MPICommunicatorP2PKernels<ValueType, utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcsHalfPrec(
        const MemoryStorage<ValueType, utils::MemorySpace::DEVICE> &dataArray,
        const MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        MemoryStorage<typename dftfe::dataTypes::halfPrecType<ValueType>::type,
                      utils::MemorySpace::DEVICE> &sendBuffer,
        dftfe::utils::deviceStream_t               deviceCommStream)
    {
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      gatherSendBufferHalfPrecDeviceKernel<<<
        (ownedLocalIndicesForTargetProcs.size() * blockSize) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        deviceCommStream>>>(ownedLocalIndicesForTargetProcs.size() * blockSize,
                            blockSize,
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              dataArray.data()),
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              ownedLocalIndicesForTargetProcs.data()),
                            sendBuffer.data());
      // CHECK
      /*
#  elif DFTFE_WITH_DEVICE_LANG_HIP
    hipLaunchKernelGGL(
      gatherSendBufferHalfPrecDeviceKernel,
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
      sendBuffer.data());*/
#  endif
    }


    __global__ void
    copyHalfPrecArrToValueTypeArrDeviceKernel(
      const dftfe::uInt size,
      const typename dftfe::dataTypes::halfPrecType<double>::type
             *halfPrecArray,
      double *valueTypeArray)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      for (dftfe::uInt i = globalThreadId; i < size;
           i += blockDim.x * gridDim.x)
        valueTypeArray[i] = __bfloat162float(
          *(reinterpret_cast<const dftfe::utils::__device_bfloat16 *>(
              halfPrecArray) +
            i));
    }

    __global__ void
    copyHalfPrecArrToValueTypeArrDeviceKernel(
      const dftfe::uInt                                           size,
      const typename dftfe::dataTypes::halfPrecType<float>::type *halfPrecArray,
      float *valueTypeArray)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      for (dftfe::uInt i = globalThreadId; i < size;
           i += blockDim.x * gridDim.x)
        valueTypeArray[i] = __bfloat162float(
          *(reinterpret_cast<const dftfe::utils::__device_bfloat16 *>(
              halfPrecArray) +
            i));
    }


    __global__ void
    copyHalfPrecArrToValueTypeArrDeviceKernel(
      const dftfe::uInt size,
      const typename dftfe::dataTypes::halfPrecType<std::complex<double>>::type
                          *halfPrecArray,
      deviceDoubleComplex *valueTypeArray)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      for (dftfe::uInt i = globalThreadId; i < size;
           i += blockDim.x * gridDim.x)
        {
          valueTypeArray[i].x = __bfloat162float(
            (*(reinterpret_cast<const dftfe::utils::__device_bfloat162 *>(
                 halfPrecArray) +
               i))
              .x);
          valueTypeArray[i].y = __bfloat162float(
            (*(reinterpret_cast<const dftfe::utils::__device_bfloat162 *>(
                 halfPrecArray) +
               i))
              .y);
        }
    }


    __global__ void
    copyHalfPrecArrToValueTypeArrDeviceKernel(
      const dftfe::uInt size,
      const typename dftfe::dataTypes::halfPrecType<std::complex<float>>::type
                         *halfPrecArray,
      deviceFloatComplex *valueTypeArray)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      for (dftfe::uInt i = globalThreadId; i < size;
           i += blockDim.x * gridDim.x)
        {
          valueTypeArray[i].x = __bfloat162float(
            (*(reinterpret_cast<const dftfe::utils::__device_bfloat162 *>(
                 halfPrecArray) +
               i))
              .x);
          valueTypeArray[i].y = __bfloat162float(
            (*(reinterpret_cast<const dftfe::utils::__device_bfloat162 *>(
                 halfPrecArray) +
               i))
              .y);
        }
    }

    __global__ void
    copyValueTypeArrToHalfPrecArrDeviceKernel(
      const dftfe::uInt                                      size,
      const double                                          *valueTypeArray,
      typename dftfe::dataTypes::halfPrecType<double>::type *halfPrecArray)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      for (dftfe::uInt i = globalThreadId; i < size;
           i += blockDim.x * gridDim.x)
        *(reinterpret_cast<dftfe::utils::__device_bfloat16 *>(halfPrecArray) +
          i) = __float2bfloat16((float)valueTypeArray[i]);
    }

    __global__ void
    copyValueTypeArrToHalfPrecArrDeviceKernel(
      const dftfe::uInt                                     size,
      const float                                          *valueTypeArray,
      typename dftfe::dataTypes::halfPrecType<float>::type *halfPrecArray)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      for (dftfe::uInt i = globalThreadId; i < size;
           i += blockDim.x * gridDim.x)
        *(reinterpret_cast<dftfe::utils::__device_bfloat16 *>(halfPrecArray) +
          i) = __float2bfloat16(valueTypeArray[i]);
    }

    __global__ void
    copyValueTypeArrToHalfPrecArrDeviceKernel(
      const dftfe::uInt          size,
      const deviceDoubleComplex *valueTypeArray,
      typename dftfe::dataTypes::halfPrecType<std::complex<double>>::type
        *halfPrecArray)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      for (dftfe::uInt i = globalThreadId; i < size;
           i += blockDim.x * gridDim.x)
        {
          (*(reinterpret_cast<dftfe::utils::__device_bfloat162 *>(
               halfPrecArray) +
             i))
            .x = __float2bfloat16((double)valueTypeArray[i].x);
          (*(reinterpret_cast<dftfe::utils::__device_bfloat162 *>(
               halfPrecArray) +
             i))
            .y = __float2bfloat16((double)valueTypeArray[i].y);
        }
    }

    __global__ void
    copyValueTypeArrToHalfPrecArrDeviceKernel(
      const dftfe::uInt         size,
      const deviceFloatComplex *valueTypeArray,
      typename dftfe::dataTypes::halfPrecType<std::complex<float>>::type
        *halfPrecArray)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      for (dftfe::uInt i = globalThreadId; i < size;
           i += blockDim.x * gridDim.x)
        {
          (*(reinterpret_cast<dftfe::utils::__device_bfloat162 *>(
               halfPrecArray) +
             i))
            .x = __float2bfloat16(valueTypeArray[i].x);
          (*(reinterpret_cast<dftfe::utils::__device_bfloat162 *>(
               halfPrecArray) +
             i))
            .y = __float2bfloat16(valueTypeArray[i].y);
        }
    }


    template <typename ValueType>
    void
    MPICommunicatorP2PKernels<ValueType, utils::MemorySpace::DEVICE>::
      copyValueTypeArrToHalfPrecArr(
        const dftfe::uInt blockSize,
        const ValueType  *valueTypeArray,
        typename dftfe::dataTypes::halfPrecType<ValueType>::type *halfPrecArray,
        dftfe::utils::deviceStream_t deviceCommStream)
    {
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      copyValueTypeArrToHalfPrecArrDeviceKernel<<<
        (blockSize) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        deviceCommStream>>>(blockSize,
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              valueTypeArray),
                            halfPrecArray); /*
 #  elif DFTFE_WITH_DEVICE_LANG_HIP
       hipLaunchKernelGGL(copyValueTypeArrToHalfPrecArrDeviceKernel,
                          (blockSize) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          0,
                          deviceCommStream,
                          blockSize,
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            valueTypeArray),
                          halfPrecArray);*/
#  endif
    }


    template <typename ValueType>
    void
    MPICommunicatorP2PKernels<ValueType, utils::MemorySpace::DEVICE>::
      copyHalfPrecArrToValueTypeArr(
        const dftfe::uInt blockSize,
        const typename dftfe::dataTypes::halfPrecType<ValueType>::type
                                    *halfPrecArray,
        ValueType                   *valueTypeArray,
        dftfe::utils::deviceStream_t deviceCommStream)
    {
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      copyHalfPrecArrToValueTypeArrDeviceKernel<<<
        (blockSize) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        deviceCommStream>>>(blockSize,
                            halfPrecArray,
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              valueTypeArray));
#  elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(copyHalfPrecArrToValueTypeArrDeviceKernel,
                         (blockSize) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         deviceCommStream,
                         blockSize,
                         halfPrecArray,
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           valueTypeArray));
#  endif
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
      const auto *recvBuffer_data =
        dftfe::utils::makeDataTypeDeviceCompatible(recvBuffer.data());
      const auto *ownedLocalIndicesForTargetProcs_data =
        dftfe::utils::makeDataTypeDeviceCompatible(
          ownedLocalIndicesForTargetProcs.data());
      auto *dataArray_data =
        dftfe::utils::makeDataTypeDeviceCompatible(dataArray.data());
      const dftfe::uInt numIndices =
        ownedLocalIndicesForTargetProcs.size() * blockSize;
      DFTFE_LAUNCH_KERNEL(accumAddFromRecvBufferDeviceKernel,
                          (numIndices) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          deviceCommStream,
                          numIndices,
                          blockSize,
                          recvBuffer_data,
                          ownedLocalIndicesForTargetProcs_data,
                          dataArray_data);
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
      const auto *recvBuffer_data =
        dftfe::utils::makeDataTypeDeviceCompatible(recvBuffer.data());
      const auto *ownedLocalIndicesForTargetProcs_data =
        dftfe::utils::makeDataTypeDeviceCompatible(
          ownedLocalIndicesForTargetProcs.data());
      auto *dataArray_data =
        dftfe::utils::makeDataTypeDeviceCompatible(dataArray.data());
      const dftfe::uInt numIndices =
        ownedLocalIndicesForTargetProcs.size() * blockSize;
      DFTFE_LAUNCH_KERNEL(accumInsertFromRecvBufferDeviceKernel,
                          (numIndices) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          deviceCommStream,
                          numIndices,
                          blockSize,
                          recvBuffer_data,
                          ownedLocalIndicesForTargetProcs_data,
                          dataArray_data);
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
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<double, utils::MemorySpace::DEVICE> &dataArray,
        const MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        MemoryStorage<uint16_t, utils::MemorySpace::DEVICE> &sendBuffer,
        dftfe::utils::deviceStream_t                         deviceCommStream);

    template void
    MPICommunicatorP2PKernels<float, utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<float, utils::MemorySpace::DEVICE> &dataArray,
        const MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        MemoryStorage<uint16_t, utils::MemorySpace::DEVICE> &sendBuffer,
        dftfe::utils::deviceStream_t                         deviceCommStream);


    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              utils::MemorySpace::DEVICE>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<std::complex<double>, utils::MemorySpace::DEVICE>
          &dataArray,
        const MemoryStorage<dftfe::uInt, utils::MemorySpace::DEVICE>
                         &ownedLocalIndicesForTargetProcs,
        const dftfe::uInt blockSize,
        MemoryStorage<std::complex<uint16_t>, utils::MemorySpace::DEVICE>
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
        MemoryStorage<std::complex<uint16_t>, utils::MemorySpace::DEVICE>
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

    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const dftfe::uInt            blockSize,
        const double                *type1Array,
        uint16_t                    *type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const dftfe::uInt            blockSize,
        const uint16_t              *type1Array,
        double                      *type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);


    template void
    MPICommunicatorP2PKernels<float, dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const dftfe::uInt            blockSize,
        const float                 *type1Array,
        uint16_t                    *type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<float, dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const dftfe::uInt            blockSize,
        const uint16_t              *type1Array,
        float                       *type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const dftfe::uInt            blockSize,
        const std::complex<double>  *type1Array,
        std::complex<uint16_t>      *type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const dftfe::uInt             blockSize,
        const std::complex<uint16_t> *type1Array,
        std::complex<double>         *type2Array,
        dftfe::utils::deviceStream_t  deviceCommStream);

    template void
    MPICommunicatorP2PKernels<std::complex<float>,
                              dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const dftfe::uInt            blockSize,
        const std::complex<float>   *type1Array,
        std::complex<uint16_t>      *type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);

    template void
    MPICommunicatorP2PKernels<std::complex<float>,
                              dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(
        const dftfe::uInt             blockSize,
        const std::complex<uint16_t> *type1Array,
        std::complex<float>          *type2Array,
        dftfe::utils::deviceStream_t  deviceCommStream);



  } // namespace utils
} // namespace dftfe
#endif
