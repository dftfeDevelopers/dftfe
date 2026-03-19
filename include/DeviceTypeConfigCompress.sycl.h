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
#ifndef dftfeDeviceTypeConfigCompress_syclh
#define dftfeDeviceTypeConfigCompress_syclh
#include <complex>
#include <cstdint>
namespace dftfe
{
  namespace utils
  {
    struct ComplexU8
    {
      uint8_t real;
      uint8_t imag;

      inline constexpr ComplexU8(uint8_t r = 0, uint8_t i = 0)
        : real(r)
        , imag(i)
      {}
    };

    inline uint8_t
    makeDataTypeDeviceCompatible(uint8_t a)
    {
      return a;
    }

    inline uint8_t *
    makeDataTypeDeviceCompatible(uint8_t *a)
    {
      return a;
    }

    inline const uint8_t *
    makeDataTypeDeviceCompatible(const uint8_t *a)
    {
      return a;
    }

    inline ComplexU8
    makeDataTypeDeviceCompatible(std::complex<uint8_t> a)
    {
      return ComplexU8{static_cast<uint8_t>(a.real()),
                       static_cast<uint8_t>(a.imag())};
    }

    inline ComplexU8 *
    makeDataTypeDeviceCompatible(std::complex<uint8_t> *a)
    {
      return reinterpret_cast<ComplexU8 *>(a);
    }

    inline const ComplexU8 *
    makeDataTypeDeviceCompatible(const std::complex<uint8_t> *a)
    {
      return reinterpret_cast<const ComplexU8 *>(a);
    }
  } // namespace utils
} // namespace dftfe

#endif // dftfeDeviceTypeConfigCompress_syclh
