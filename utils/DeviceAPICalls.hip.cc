#include "hip/hip_runtime.h"
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


#ifdef DFTFE_WITH_DEVICE_LANG_HIP
#  include <DeviceAPICalls.h>
#  include <stdio.h>
#  include <vector>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <Exceptions.h>
namespace dftfe
{
  namespace utils
  {
    namespace
    {
      template <typename ValueType>
      __global__ void
      setValueKernel(ValueType *devPtr, ValueType value, std::size_t size)
      {
        const std::size_t globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        for (std::size_t i = globalThreadId; i < size;
             i += blockDim.x * gridDim.x)
          {
            devPtr[i] = value;
          }
      }

      __global__ void
      emptyKernel(void)
      {}


    } // namespace

    deviceError_t
    deviceReset()
    {
      deviceError_t err = hipDeviceReset();
      DEVICE_API_CHECK(err);

      hipLaunchKernelGGL(emptyKernel, 1, 1, 0, 0);

      return err;
    }


    deviceError_t
    deviceMemGetInfo(std::size_t *free, std::size_t *total)
    {
      deviceError_t err = hipMemGetInfo(free, total);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    getDeviceCount(int *count)
    {
      deviceError_t err = hipGetDeviceCount(count);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    getDevice(int *deviceId)
    {
      deviceError_t err = hipGetDevice(deviceId);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    setDevice(int deviceId)
    {
      deviceError_t err = hipSetDevice(deviceId);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMalloc(void **devPtr, std::size_t size)
    {
      deviceError_t err = hipMalloc(devPtr, size);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMemset(void *devPtr, int value, std::size_t count)
    {
      deviceError_t err = hipMemset(devPtr, value, count);
      DEVICE_API_CHECK(err);
      return err;
    }

    template <typename ValueType>
    void
    deviceSetValue(ValueType *devPtr, ValueType value, std::size_t size)
    {
      hipLaunchKernelGGL(setValueKernel,
                         size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         makeDataTypeDeviceCompatible(devPtr),
                         makeDataTypeDeviceCompatible(value),
                         size);
    }

    template void
    deviceSetValue(int *devPtr, int value, std::size_t size);

    template void
    deviceSetValue(long int *devPtr, long int value, std::size_t size);

    template void
    deviceSetValue(size_type *devPtr,
                   size_type  value,
                   std::size_t     size);

    template void
    deviceSetValue(global_size_type *devPtr,
                   global_size_type  value,
                   std::size_t            size);

    template void
    deviceSetValue(double *devPtr, double value, std::size_t size);

    template void
    deviceSetValue(float *devPtr, float value, std::size_t size);

    template void
    deviceSetValue(std::complex<float> *devPtr,
                   std::complex<float>  value,
                   std::size_t          size);

    template void
    deviceSetValue(std::complex<double> *devPtr,
                   std::complex<double>  value,
                   std::size_t           size);

    deviceError_t
    deviceFree(void *devPtr)
    {
      deviceError_t err = hipFree(devPtr);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceHostMalloc(void **hostPtr, std::size_t size)
    {
      deviceError_t err =
        hipHostMalloc(hostPtr, size, hipHostMallocNonCoherent);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceHostFree(void *hostPtr)
    {
      deviceError_t err = hipHostFree(hostPtr);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMemcpyD2H(void *dst, const void *src, std::size_t count)
    {
      deviceError_t err = hipMemcpy(dst, src, count, hipMemcpyDeviceToHost);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMemcpyD2D(void *dst, const void *src, std::size_t count)
    {
      deviceError_t err = hipMemcpy(dst, src, count, hipMemcpyDeviceToDevice);
      DEVICE_API_CHECK(err);
      return err;
    }
    deviceError_t
    deviceMemcpyH2D(void *dst, const void *src, std::size_t count)
    {
      deviceError_t err = hipMemcpy(dst, src, count, hipMemcpyHostToDevice);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMemcpyD2H_2D(void *      dst,
                       std::size_t dpitch,
                       const void *src,
                       std::size_t spitch,
                       std::size_t width,
                       std::size_t height)
    {
      deviceError_t err = hipMemcpy2D(
        dst, dpitch, src, spitch, width, height, hipMemcpyDeviceToHost);
      DEVICE_API_CHECK(err);
      return err;
    }


    deviceError_t
    deviceMemcpyD2D_2D(void *      dst,
                       std::size_t dpitch,
                       const void *src,
                       std::size_t spitch,
                       std::size_t width,
                       std::size_t height)
    {
      deviceError_t err = hipMemcpy2D(
        dst, dpitch, src, spitch, width, height, hipMemcpyDeviceToDevice);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMemcpyH2D_2D(void *      dst,
                       std::size_t dpitch,
                       const void *src,
                       std::size_t spitch,
                       std::size_t width,
                       std::size_t height)
    {
      deviceError_t err = hipMemcpy2D(
        dst, dpitch, src, spitch, width, height, hipMemcpyHostToDevice);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceSynchronize()
    {
      deviceError_t err = hipDeviceSynchronize();
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMemcpyAsyncD2H(void *         dst,
                         const void *   src,
                         std::size_t    count,
                         deviceStream_t stream)
    {
      deviceError_t err =
        hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToHost, stream);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMemcpyAsyncD2D(void *         dst,
                         const void *   src,
                         std::size_t    count,
                         deviceStream_t stream)
    {
      deviceError_t err =
        hipMemcpyAsync(dst, src, count, hipMemcpyDeviceToDevice, stream);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceMemcpyAsyncH2D(void *         dst,
                         const void *   src,
                         std::size_t    count,
                         deviceStream_t stream)
    {
      deviceError_t err =
        hipMemcpyAsync(dst, src, count, hipMemcpyHostToDevice, stream);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceStreamCreate(deviceStream_t *pStream)
    {
      deviceError_t err = hipStreamCreate(pStream);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceStreamDestroy(deviceStream_t stream)
    {
      deviceError_t err = hipStreamDestroy(stream);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceStreamSynchronize(deviceStream_t stream)
    {
      deviceError_t err = hipStreamSynchronize(stream);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceEventCreate(deviceEvent_t *pEvent)
    {
      deviceError_t err = hipEventCreate(pEvent);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceEventDestroy(deviceEvent_t event)
    {
      deviceError_t err = hipEventDestroy(event);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceEventRecord(deviceEvent_t event, deviceStream_t stream)
    {
      deviceError_t err = hipEventRecord(event, stream);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceEventSynchronize(deviceEvent_t event)
    {
      deviceError_t err = hipEventSynchronize(event);
      DEVICE_API_CHECK(err);
      return err;
    }

    deviceError_t
    deviceStreamWaitEvent(deviceStream_t stream,
                          deviceEvent_t  event,
                          unsigned int   flags)
    {
      deviceError_t err = hipStreamWaitEvent(stream, event, flags);
      DEVICE_API_CHECK(err);
      return err;
    }

  } // namespace utils
} // namespace dftfe
#endif
