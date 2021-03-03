// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018  The Regents of the University of Michigan and DFT-FE authors.
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

#if defined(DFTFE_WITH_GPU)
#ifndef cudaHelpers_h
#define cudaHelpers_h

#include <cuda_runtime.h>

namespace dftfe
{
#define CUDACHECK(cmd) do {                         \
    cudaError_t e = cmd;                              \
    if( e != cudaSuccess ) {                          \
      printf("Failed: Cuda error %s:%d '%s'\n",       \
          __FILE__,__LINE__,cudaGetErrorString(e));   \
      exit(EXIT_FAILURE);                             \
    }                                                 \
  } while(0)

  void setupGPU();  
}

#endif
#endif
