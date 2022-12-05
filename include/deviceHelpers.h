// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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

#if defined(DFTFE_WITH_DEVICE)
#  ifndef deviceHelpers_h
#    define deviceHelpers_h

#    include <cuda_runtime.h>
#    include <dftfeDataTypes.h>
#    include <MemorySpaceType.h>
#    include <headers.h>
#include <TypeConfig.h>

namespace dftfe
{
#    define cublasCheck(expr)                                                            \
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

  namespace deviceUtils
  {
    void
    setupDevice();

    template <typename NumberTypeComplex, typename NumberTypeReal>
    void
    copyComplexArrToRealArrsDevice(const dftfe::size_type size,
                                   const NumberTypeComplex *        complexArr,
                                   NumberTypeReal *                 realArr,
                                   NumberTypeReal *                 imagArr);


    template <typename NumberTypeComplex, typename NumberTypeReal>
    void
    copyRealArrsToComplexArrDevice(const dftfe::size_type size,
                                   const NumberTypeReal *           realArr,
                                   const NumberTypeReal *           imagArr,
                                   NumberTypeComplex *              complexArr);

    void
    add(double *        y,
        const double *  x,
        const double    alpha,
        const int       size,
        cublasHandle_t &cublasHandle);

    double
    l2_norm(const double *  x,
            const int       size,
            const MPI_Comm &mpi_communicator,
            cublasHandle_t &cublasHandle);

    double
    dot(const double *  x,
        const double *  y,
        const int       size,
        const MPI_Comm &mpi_communicator,
        cublasHandle_t &cublasHandle);

    template <typename NumberType>
    void
    sadd(NumberType *y, NumberType *x, const NumberType beta, const int size);

  } // namespace deviceUtils

} // namespace dftfe

#  endif
#endif
