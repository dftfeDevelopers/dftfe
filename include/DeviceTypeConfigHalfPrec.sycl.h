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
//
#ifndef dftfeDeviceTypeConfigHalfPrec_syclh
#define dftfeDeviceTypeConfigHalfPrec_syclh

#include <complex>
#include <cstdint>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/bfloat16.hpp>

namespace dftfe
{
  namespace utils
  {
    typedef sycl::ext::oneapi::bfloat16               __device_bfloat16;
    typedef std::complex<sycl::ext::oneapi::bfloat16> __device_bfloat162;

    inline float
    __bfloat162float(__device_bfloat16 bf16)
    {
      uint16_t bits       = sycl::bit_cast<uint16_t>(bf16);
      uint32_t float_bits = (bits << 16);
      return sycl::bit_cast<float>(float_bits);
    }

    inline __device_bfloat16
    __float2bfloat16(float f)
    {
      uint32_t float_bits    = sycl::bit_cast<uint32_t>(f);
      uint32_t rounding_bias = ((float_bits >> 16) & 1) + 0x7FFF;
      uint32_t rounded_bits  = float_bits + rounding_bias;
      uint16_t bf16_bits     = (rounded_bits >> 16) & 0xFFFF;
      return sycl::bit_cast<__device_bfloat16>(bf16_bits);
    }

    inline __device_bfloat16
    __sycl_bfloat16_raw(uint16_t a)
    {
      return sycl::bit_cast<__device_bfloat16>(a);
    }

    inline void
    copyValue(__device_bfloat16 *a, const float b)
    {
      *a = __float2bfloat16(b);
    }

    inline void
    copyValue(__device_bfloat16 *a, const double b)
    {
      *a = __float2bfloat16((float)b);
    }

    inline void
    copyValue(__device_bfloat162 *a, const std::complex<float> &b)
    {
      *a = __device_bfloat162(__float2bfloat16(b.real()),
                              __float2bfloat16(b.imag()));
    }

    inline void
    copyValue(__device_bfloat162 *a, const std::complex<double> &b)
    {
      *a = __device_bfloat162(__float2bfloat16((float)(b.real())),
                              __float2bfloat16((float)(b.imag())));
    }

    inline void
    copyValue(float *a, const __device_bfloat16 &b)
    {
      *a = __bfloat162float(b);
    }

    inline void
    copyValue(double *a, const __device_bfloat16 &b)
    {
      *a = (double)__bfloat162float(b);
    }

    inline void
    copyValue(std::complex<float> *a, const __device_bfloat162 &b)
    {
      *a = std::complex<float>(__bfloat162float(b.real()),
                               __bfloat162float(b.imag()));
    }

    inline void
    copyValue(std::complex<double> *a, const __device_bfloat162 &b)
    {
      *a = std::complex<double>((double)__bfloat162float(b.real()),
                                (double)__bfloat162float(b.imag()));
    }

    inline float
    realPartDevice(const __device_bfloat162 a)
    {
      return a.real();
    }

    inline float
    imagPartDevice(const __device_bfloat162 a)
    {
      return a.imag();
    }

    // uint16_t saves bits only
    // not for arithmetic operations

    inline __device_bfloat16
    makeDataTypeDeviceCompatible(uint16_t a)
    {
      return __device_bfloat16{__sycl_bfloat16_raw(a)};
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
      return __device_bfloat162{__sycl_bfloat16_raw(a.real()),
                                __sycl_bfloat16_raw(a.imag())};
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

#endif // dftfeDeviceTypeConfigHalfPrec_syclh
