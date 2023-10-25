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
//
#include <BLASWrapper.h>
#include <linearAlgebraOperations.h>

namespace dftfe
{
  namespace linearAlgebra
  {



    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemm(
      const char *        transA,
      const char *        transB,
      const unsigned int *m,
      const unsigned int *n,
      const unsigned int *k,
      const float *       alpha,
      const float *       A,
      const unsigned int *lda,
      const float *       B,
      const unsigned int *ldb,
      const float *       beta,
      float *             C,
      const unsigned int *ldc) const
    {
      sgemm_(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemm(
      const char *        transA,
      const char *        transB,
      const unsigned int *m,
      const unsigned int *n,
      const unsigned int *k,
      const double *      alpha,
      const double *      A,
      const unsigned int *lda,
      const double *      B,
      const unsigned int *ldb,
      const double *      beta,
      double *            C,
      const unsigned int *ldc) const
    {
      dgemm_(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xscal(
      const unsigned int *n,
      const double *      alpha,
      double *            x,
      const unsigned int *inc) const
    {
      dscal_(n, alpha, x, inc);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xscal(
      const unsigned int *n,
      const float *       alpha,
      float *             x,
      const unsigned int *inc) const
    {
      sscal_(n, alpha, x, inc);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xcopy(
      const unsigned int *n,
      const double *      x,
      const unsigned int *incx,
      double *            y,
      const unsigned int *incy) const
    {
      dcopy_(n, x, incx, y, incy);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xcopy(
      const unsigned int *n,
      const float *       x,
      const unsigned int *incx,
      float *             y,
      const unsigned int *incy) const
    {
      scopy_(n, x, incx, y, incy);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemm(
      const char *               transA,
      const char *               transB,
      const unsigned int *       m,
      const unsigned int *       n,
      const unsigned int *       k,
      const std::complex<float> *alpha,
      const std::complex<float> *A,
      const unsigned int *       lda,
      const std::complex<float> *B,
      const unsigned int *       ldb,
      const std::complex<float> *beta,
      std::complex<float> *      C,
      const unsigned int *       ldc) const
    {
      cgemm_(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemm(
      const char *                transA,
      const char *                transB,
      const unsigned int *        m,
      const unsigned int *        n,
      const unsigned int *        k,
      const std::complex<double> *alpha,
      const std::complex<double> *A,
      const unsigned int *        lda,
      const std::complex<double> *B,
      const unsigned int *        ldb,
      const std::complex<double> *beta,
      std::complex<double> *      C,
      const unsigned int *        ldc) const
    {
      zgemm_(transA, transB, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xscal(
      const unsigned int *        n,
      const std::complex<double> *alpha,
      std::complex<double> *      x,
      const unsigned int *        inc) const
    {
      zscal_(n, alpha, x, inc);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xscal(
      const unsigned int *  n,
      const double *        alpha,
      std::complex<double> *x,
      const unsigned int *  inc) const
    {
      zdscal_(n, alpha, x, inc);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xcopy(
      const unsigned int *        n,
      const std::complex<double> *x,
      const unsigned int *        incx,
      std::complex<double> *      y,
      const unsigned int *        incy) const
    {
      zcopy_(n, x, incx, y, incy);
    }



  } // End of namespace linearAlgebra
} // End of namespace dftfe
