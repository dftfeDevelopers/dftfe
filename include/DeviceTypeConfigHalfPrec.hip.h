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
//
#ifndef dftfeDeviceTypeConfigHalfPrec_hiph
#define dftfeDeviceTypeConfigHalfPrec_hiph
#include <complex>
#include <hip/hip_complex.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
namespace dftfe
{
  namespace utils
  {
    typedef __hip_bfloat16  __device_bfloat16;
    typedef __hip_bfloat162 __device_bfloat162;

    __forceinline__ __device__ void
    copyValue(__device_bfloat16 *a, const float b)
    {
      *a = __float2bfloat16(b);
    }

    __forceinline__ __device__ void
    copyValue(__device_bfloat16 *a, const double b)
    {
      *a = __float2bfloat16((float)b);
    }

    __forceinline__ __device__ void
    copyValue(__device_bfloat162 *a, const hipFloatComplex b)
    {
      a->x = __float2bfloat16(b.x);
      a->y = __float2bfloat16(b.y);
    }

    __forceinline__ __device__ void
    copyValue(__device_bfloat162 *a, const hipDoubleComplex b)
    {
      a->x = __float2bfloat16((float)b.x);
      a->y = __float2bfloat16((float)b.y);
    }

    __forceinline__ __device__ void
    copyValue(float *a, const __device_bfloat16 b)
    {
      *a = __bfloat162float(b);
    }

    __forceinline__ __device__ void
    copyValue(double *a, const __device_bfloat16 b)
    {
      *a = (double)__bfloat162float(b);
    }

    __forceinline__ __device__ void
    copyValue(hipFloatComplex *a, const __device_bfloat162 b)

    {
      a->x = __bfloat162float(b.x);
      a->y = __bfloat162float(b.y);
    }

    __forceinline__ __device__ void
    copyValue(hipDoubleComplex *a, const __device_bfloat162 b)
    {
      a->x = (double)__bfloat162float(b.x);
      a->y = (double)__bfloat162float(b.y);
    }

    __forceinline__ __device__ float
    realPartDevice(const __device_bfloat162 a)
    {
      return a.x;
    }

    __forceinline__ __device__ float
    imagPartDevice(const __device_bfloat162 a)
    {
      return a.y;
    }

    // uint16_t saves bits only
    // not for arithmetic operations

    inline __device_bfloat16
    makeDataTypeDeviceCompatible(uint16_t a)
    {
      return __device_bfloat16{__hip_bfloat16_raw{a}};
    }

    inline __device_bfloat16 *
    makeDataTypeDeviceCompatible(uint16_t *a)
    {
      return reinterpret_cast<__device_bfloat16 *>(a);
    }

    inline const __device_bfloat16 *
    makeDataTypeDeviceCompatible(const uint16_t *a)
    {
      return reinterpret_cast<const __device_bfloat16 *>(a);
    }

    inline __device_bfloat162
    makeDataTypeDeviceCompatible(std::complex<uint16_t> a)
    {
      return __device_bfloat162{__device_bfloat16{__hip_bfloat16_raw{a.real()}},
                                __device_bfloat16{
                                  __hip_bfloat16_raw{a.imag()}}};
    }

    inline __device_bfloat162 *
    makeDataTypeDeviceCompatible(std::complex<uint16_t> *a)
    {
      return reinterpret_cast<__device_bfloat162 *>(a);
    }

    inline const __device_bfloat162 *
    makeDataTypeDeviceCompatible(const std::complex<uint16_t> *a)
    {
      return reinterpret_cast<const __device_bfloat162 *>(a);
    }

  } // namespace utils
} // namespace dftfe

#endif // dftfeDeviceTypeConfigHalfPrec_hiph
