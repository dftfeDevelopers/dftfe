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


#ifdef DFTFE_WITH_DEVICE_CUDA
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
      setValueKernel(ValueType *devPtr, ValueType value, size_type size)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        for (unsigned int i = globalThreadId; i < size;
             i += blockDim.x * gridDim.x)
          {
            devPtr[i] = value;
          }
      }
    } // namespace

    void
    deviceGetDeviceCount(int *count)
    {
      DEVICE_API_CHECK(cudaGetDeviceCount(count));
    }

    void
    deviceGetDevice(int *deviceId)
    {
      DEVICE_API_CHECK(cudaGetDevice(deviceId));
    }

    void
    deviceSetDevice(int deviceId)
    {
      DEVICE_API_CHECK(cudaSetDevice(deviceId));
    }

    void
    deviceMalloc(void **devPtr, size_type size)
    {
      DEVICE_API_CHECK(cudaMalloc(devPtr, size));
    }

    void
    deviceMemset(void *devPtr, size_type count)
    {
      DEVICE_API_CHECK(cudaMemset(devPtr, 0, count));
    }

    template <typename ValueType>
    void
    deviceSetValue(ValueType *devPtr, ValueType value, size_type size)
    {
      setValueKernel<<<size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                       dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        makeDataTypeDeviceCompatible(devPtr),
        makeDataTypeDeviceCompatible(value),
        size);
    }

    template void
    deviceSetValue(size_type *devPtr, size_type value, size_type size);

    template void
    deviceSetValue(int *devPtr, int value, size_type size);

    template void
    deviceSetValue(double *devPtr, double value, size_type size);

    template void
    deviceSetValue(float *devPtr, float value, size_type size);

    template void
    deviceSetValue(std::complex<float> *devPtr,
                   std::complex<float>  value,
                   size_type            size);

    template void
    deviceSetValue(std::complex<double> *devPtr,
                   std::complex<double>  value,
                   size_type             size);

    void
    deviceFree(void *devPtr)
    {
      DEVICE_API_CHECK(cudaFree(devPtr));
    }

    void
    hostPinnedMalloc(void **hostPtr, size_type size)
    {
      DEVICE_API_CHECK(cudaMallocHost(hostPtr, size));
    }

    void
    hostPinnedFree(void *hostPtr)
    {
      DEVICE_API_CHECK(cudaFreeHost(hostPtr));
    }

    void
    deviceMemcpyD2H(void *dst, const void *src, size_type count)
    {
      DEVICE_API_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
    }

    void
    deviceMemcpyD2D(void *dst, const void *src, size_type count)
    {
      DEVICE_API_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToDevice));
    }
    void
    deviceMemcpyH2D(void *dst, const void *src, size_type count)
    {
      DEVICE_API_CHECK(cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice));
    }
  } // namespace utils
} // namespace dftfe
#endif
