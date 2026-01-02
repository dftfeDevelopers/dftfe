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
#ifndef dftfeDeviceTypeConfigCompress_cuh
#define dftfeDeviceTypeConfigCompress_cuh
#include <complex>
#include <cuComplex.h>
namespace dftfe
{
  namespace utils
  {
    inline float
    makeDataTypeDeviceCompatible(uint8_t)
    {
      return 0.0f;
    }

    inline float *
    makeDataTypeDeviceCompatible(uint8_t *)
    {
      return nullptr;
    }

    inline const float *
    makeDataTypeDeviceCompatible(const uint8_t *)
    {
      return nullptr;
    }

    inline cuFloatComplex
    makeDataTypeDeviceCompatible(std::complex<uint8_t> a)
    {
      return make_cuFloatComplex(0.0f, 0.0f);
    }

    inline cuFloatComplex *
    makeDataTypeDeviceCompatible(std::complex<uint8_t> *)
    {
      return nullptr;
    }

    inline const cuFloatComplex *
    makeDataTypeDeviceCompatible(const std::complex<uint8_t> *)
    {
      return nullptr;
    }
  } // namespace utils
} // namespace dftfe

#endif // dftfeDeviceTypeConfigCompress_cuh
