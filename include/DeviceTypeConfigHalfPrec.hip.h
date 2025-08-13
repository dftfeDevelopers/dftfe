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
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
namespace dftfe
{
  namespace utils
  {
    typedef __hip_bfloat16   __device_bfloat16;
    typedef __hip_bfloat162  __device_bfloat162;
  } // namespace utils
} // namespace dftfe

#endif // dftfeDeviceTypeConfigHalfPrec_hiph
