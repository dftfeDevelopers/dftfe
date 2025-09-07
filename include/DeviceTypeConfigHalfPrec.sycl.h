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

#include <sycl/ext/oneapi/bfloat16.hpp>
#include <sycl/ext/intel/math/imf_fp_conversions.hpp>
namespace dftfe
{
  namespace utils
  {
    typedef sycl::ext::oneapi::bfloat16               __device_bfloat16;
    typedef std::complex<sycl::ext::oneapi::bfloat16> __device_bfloat162;

    inline void
    copyValue(__device_bfloat16 *a, float b)
    {
      *a = sycl::ext::intel::math::float2bfloat16(b);
    }

    inline void
    copyValue(__device_bfloat16 *a, double b)
    {
      *a = sycl::ext::intel::math::double2bfloat16(b);
    }

    inline void
    copyValue(__device_bfloat162 *a, const std::complex<float> &b)
    {
      *a = __device_bfloat162(sycl::ext::intel::math::float2bfloat16(b.real()),
                              sycl::ext::intel::math::float2bfloat16(b.imag()));
    }

    inline void
    copyValue(__device_bfloat162 *a, const std::complex<double> &b)
    {
      *a =
        __device_bfloat162(sycl::ext::intel::math::double2bfloat16(b.real()),
                           sycl::ext::intel::math::double2bfloat16(b.imag()));
    }

    inline void
    copyValue(float *a, const __device_bfloat16 b)
    {
      *a = sycl::ext::intel::math::bfloat162float(b);
    }

    inline void
    copyValue(double *a, const __device_bfloat16 b)
    {
      *a = (double)sycl::ext::intel::math::bfloat162float(b);
    }

    inline void
    copyValue(std::complex<float> *a, const __device_bfloat162 &b)
    {
      *a =
        std::complex<float>(sycl::ext::intel::math::bfloat162float(b.real()),
                            sycl::ext::intel::math::bfloat162float(b.imag()));
    }

    inline void
    copyValue(std::complex<double> *a, const __device_bfloat162 &b)
    {
      *a =
        std::complex<double>(sycl::ext::intel::math::bfloat162float(b.real()),
                             sycl::ext::intel::math::bfloat162float(b.imag()));
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

    inline __device_bfloat162
    makeDataTypeDeviceCompatible(uint16_t a)
    {
      return __device_bfloat16{sycl::ext::intel::math::ushort_as_bfloat16(a)};
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
      return __device_bfloat162{
        __device_bfloat16{sycl::ext::intel::math::ushort_as_bfloat16(a.real())},
        __device_bfloat16{
          sycl::ext::intel::math::ushort_as_bfloat16(a.imag())}};
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
