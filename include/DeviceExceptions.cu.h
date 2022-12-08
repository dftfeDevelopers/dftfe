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
#ifndef dftfeDeviceExceptions_cuh
#define dftfeDeviceExceptions_cuh


#define DEVICE_API_CHECK(cmd)                       \
  do                                                \
    {                                               \
      cudaError_t e = cmd;                          \
      if (e != cudaSuccess)                         \
        {                                           \
          printf("Failed: Cuda error %s:%d '%s'\n", \
                 __FILE__,                          \
                 __LINE__,                          \
                 cudaGetErrorString(e));            \
          exit(EXIT_FAILURE);                       \
        }                                           \
    }                                               \
  while (0)

#define DEVICEBLAS_API_CHECK(expr)                                                   \
  {                                                                                  \
    cublasStatus_t __cublas_error = expr;                                            \
    if ((__cublas_error) != CUBLAS_STATUS_SUCCESS)                                   \
      {                                                                              \
        printf(                                                                      \
          "cuBLAS error on or before line number %d in file: %s. Error code: %d.\n", \
          __LINE__,                                                                  \
          __FILE__,                                                                  \
          __cublas_error);                                                           \
      }                                                                              \
  }


#endif // dftfeDeviceExceptions_cuh
