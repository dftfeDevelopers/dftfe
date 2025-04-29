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
#ifndef dftfeDeviceTypeConfig_hiph
#define dftfeDeviceTypeConfig_hiph
#define HIPBLAS_V2
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <hipblas.h>
namespace dftfe
{
  namespace utils
  {
    typedef hipError_t       deviceError_t;
    typedef hipStream_t      deviceStream_t;
    typedef hipEvent_t       deviceEvent_t;
    typedef hipDoubleComplex deviceDoubleComplex;
    typedef hipFloatComplex  deviceFloatComplex;

    // static consts
    static const deviceError_t deviceSuccess = hipSuccess;

    // vendor blas related typedef and static consts
    typedef hipblasHandle_t      deviceBlasHandle_t;
    typedef hipblasOperation_t   deviceBlasOperation_t;
    typedef hipblasStatus_t      deviceBlasStatus_t;
    typedef hipblasComputeType_t deviceBlasComputeType_t;
    typedef hipDataType          deviceDataType_t;

    static const hipblasOperation_t   DEVICEBLAS_OP_N = HIPBLAS_OP_N;
    static const hipblasOperation_t   DEVICEBLAS_OP_T = HIPBLAS_OP_T;
    static const hipblasOperation_t   DEVICEBLAS_OP_C = HIPBLAS_OP_C;
    static const hipblasComputeType_t DEVICEBLAS_COMPUTE_32F =
      HIPBLAS_COMPUTE_32F;
    static const hipblasComputeType_t DEVICEBLAS_COMPUTE_32F_FAST_TF32 =
      HIPBLAS_COMPUTE_32F_FAST_TF32;
    static const hipblasComputeType_t DEVICEBLAS_COMPUTE_32F_FAST_16BF =
      HIPBLAS_COMPUTE_32F_FAST_16BF;
    static const hipblasComputeType_t DEVICEBLAS_COMPUTE_32F_FAST_16F =
      HIPBLAS_COMPUTE_32F_FAST_16F;
    static const hipblasGemmAlgo_t DEVICEBLAS_GEMM_DEFAULT =
      HIPBLAS_GEMM_DEFAULT;
    static const hipDataType DEVICE_R_64F  = HIP_R_64F;
    static const hipDataType DEVICE_R_32F  = HIP_R_32F;
    static const hipDataType DEVICE_R_16F  = HIP_R_16F;
    static const hipDataType DEVICE_R_16BF = HIP_R_16BF;
    static const hipDataType DEVICE_C_64F  = HIP_C_64F;
    static const hipDataType DEVICE_C_32F  = HIP_C_32F;
    static const hipDataType DEVICE_C_16F  = HIP_C_16F;
    static const hipDataType DEVICE_C_16BF = HIP_C_16BF;
  } // namespace utils
} // namespace dftfe

#endif // dftfeDeviceTypeConfig_hiph
