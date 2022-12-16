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
#ifndef dftfeDeviceTypeConfig_hiph
#define dftfeDeviceTypeConfig_hiph

#include <hip/hip_complex.h>
#include <hipblas.h>
namespace dftfe
{
  namespace utils
  {
    typedef hipError_t     deviceError_t;
    typedef hipStream_t    deviceStream_t;
    typedef hipEvent_t     deviceEvent_t;
    typedef hipDoubleComplex deviceDoubleComplex;


    // static consts
    static const deviceError_t deviceSuccess = hipSuccess;

    // vendor blas related typedef and static consts
    typedef hipblasHandle_t    deviceBlasHandle_t;
    typedef hipblasOperation_t deviceBlasOperation_t;
    typedef hipblasStatus_t    deviceBlasStatus_t;

    static const hipblasOperation_t DEVICEBLAS_OP_N = HIPBLAS_OP_N;
    static const hipblasOperation_t DEVICEBLAS_OP_T = HIPBLAS_OP_T;
    static const hipblasOperation_t DEVICEBLAS_OP_C = HIPBLAS_OP_C;
  } // namespace utils
} // namespace dftfe

#endif // dftfeDeviceTypeConfig_hiph
