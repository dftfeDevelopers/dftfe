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
#ifndef dftfeDeviceTypeConfig_cuh
#define dftfeDeviceTypeConfig_cuh

#include <cuComplex.h>
#include <cublas_v2.h>
namespace dftfe
{
  namespace utils
  {
    typedef cudaError_t     deviceError_t;
    typedef cudaStream_t    deviceStream_t;
    typedef cudaEvent_t     deviceEvent_t;
    typedef cuDoubleComplex deviceDoubleComplex;
    typedef cuFloatComplex deviceFloatComplex;

    // static consts
    static const deviceError_t deviceSuccess = cudaSuccess;

    // vendor blas related typedef and static consts
    typedef cublasHandle_t    deviceBlasHandle_t;
    typedef cublasOperation_t deviceBlasOperation_t;
    typedef cublasStatus_t    deviceBlasStatus_t;
    typedef cublasMath_t      deviceBlasMath_t;

    static const cublasOperation_t DEVICEBLAS_OP_N = CUBLAS_OP_N;
    static const cublasOperation_t DEVICEBLAS_OP_T = CUBLAS_OP_T;
    static const cublasOperation_t DEVICEBLAS_OP_C = CUBLAS_OP_C;
    static const cublasMath_t      DEVICEBLAS_TF32_TENSOR_OP_MATH =
      CUBLAS_TF32_TENSOR_OP_MATH;
  } // namespace utils
} // namespace dftfe

#endif // dftfeDeviceTypeConfig_cuh
