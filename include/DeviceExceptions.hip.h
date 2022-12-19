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
#ifndef dftfeDeviceExceptions_hiph
#define dftfeDeviceExceptions_hiph


#define DEVICE_API_CHECK(cmd)                      \
  do                                               \
    {                                              \
      hipError_t e = cmd;                          \
      if (e != hipSuccess)                         \
        {                                          \
          printf("Failed: HIP error %s:%d '%s'\n", \
                 __FILE__,                         \
                 __LINE__,                         \
                 hipGetErrorString(e));            \
          exit(EXIT_FAILURE);                      \
        }                                          \
    }                                              \
  while (0)

#define DEVICEBLAS_API_CHECK(expr)                                                    \
  {                                                                                   \
    hipblasStatus_t __hipblas_error = expr;                                           \
    if ((__hipblas_error) != HIPBLAS_STATUS_SUCCESS)                                  \
      {                                                                               \
        printf(                                                                       \
          "hipBLAS error on or before line number %d in file: %s. Error code: %d.\n", \
          __LINE__,                                                                   \
          __FILE__,                                                                   \
          __hipblas_error);                                                           \
      }                                                                               \
  }


#endif // dftfeDeviceExceptions_hiph
