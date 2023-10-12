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

#ifndef BLASWrapperClass_h
#define BLASWrapperClass_h

#include <dftfeDataTypes.h>
#include <MemorySpaceType.h>
#include <headers.h>
#include <complex>
#include <TypeConfig.h>
#include <DeviceTypeConfig.h>


namespace dftfe
{
  namespace linearAlgebra
  {
    template <dftfe::utils::MemorySpace memorySpace>
    class BLASWrapperClassBase
    {
    public:
      // Real-Single Precision GEMM
      void
      xgemm(const char *        transA,
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
            const unsigned int *ldc);
      // Complex-Single Precision GEMM
      void
      xgemm(const char *               transA,
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
            const unsigned int *       ldc);

      // Real-double precison GEMM
      void
      xgemm(const char *        transA,
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
            const unsigned int *ldc);


      // Complex-double precision GEMM
      void
      xgemm(const char *                transA,
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
            const unsigned int *        ldc);

      // Real-Double scaling of Real-vector
      void
      xscal(const unsigned int *n,
            const double *      alpha,
            double *            x,
            const unsigned int *inc);

      // Real-Float scaling of Real-vector
      void
      xscal(const unsigned int *n,
            const float *       alpha,
            float *             x,
            const unsigned int *inc);

      // Complex-double scaling of complex-vector
      void
      xscal(const unsigned int *        n,
            const std::complex<double> *alpha,
            std::complex<double> *      x,
            const unsigned int *        inc);

      // Real-double scaling of complex-vector
      void
      xscal(const unsigned int *  n,
            const double *        alpha,
            std::complex<double> *x,
            const unsigned int *  inc);

      // Real double Norm2
      void
      xnrm2(const unsigned int *n, const double *x, const unsigned int *incx);

      // Real dot product
      void
      xdot(const unsigned int *N,
           const double *      X,
           const unsigned int *INCX,
           const double *      Y,
           const unsigned int *INCY);

      // Real double Ax+y
      void
      xaxpy(const unsigned int *n,
            const double *      alpha,
            double *            x,
            const unsigned int *incx,
            double *            y,
            const unsigned int *incy);

      // Complex double Ax+y
      void
      xaxpy(const unsigned int *        n,
            const std::complex<double> *alpha,
            std::complex<double> *      x,
            const unsigned int *        incx,
            std::complex<double> *      y,
            const unsigned int *        incy);

      // Real copy of double data
      void
      xcopy(const unsigned int *n,
            const double *      x,
            const unsigned int *incx,
            double *            y,
            const unsigned int *incy);

      // Complex double copy of data
      void
      xcopy(const unsigned int *        n,
            const std::complex<double> *x,
            const unsigned int *        incx,
            std::complex<double> *      y,
            const unsigned int *        incy);

      // Real copy of float data
      void
      xcopy(const unsigned int *n,
            const float *       x,
            const unsigned int *incx,
            float *             y,
            const unsigned int *incy);

      // Real double symmetric matrix-vector product
      void
      xsymv(const char *        UPLO,
            const unsigned int *N,
            const double *      alpha,
            const double *      A,
            const unsigned int *LDA,
            const double *      X,
            const unsigned int *INCX,
            const double *      beta,
            double *            C,
            const unsigned int *INCY);

      void
      xgemmBatched(const char *        transA,
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
                   const unsigned int *ldc,
                   const int *         batchCount);

      void
      xgemmBatched(const char *               transA,
                   const char *               transB,
                   const unsigned int *       m,
                   const unsigned int *       n,
                   const unsigned int *       k,
                   const std::complex<double> e *alpha,
                   const std::complex<double> *  A,
                   const unsigned int *          lda,
                   const std::complex<double> *  B,
                   const unsigned int *          ldb,
                   const std::complex<double> *  beta,
                   std::complex<double> e *C,
                   const unsigned int *    ldc,
                   const int *             batchCount);


      void
      xgemmStridedBatched(const char *        transA,
                          const char *        transB,
                          const unsigned int *m,
                          const unsigned int *n,
                          const unsigned int *k,
                          const double *      alpha,
                          const double *      A,
                          const unsigned int *lda,
                          long long int *     strideA,
                          const double *      B,
                          const unsigned int *ldb,
                          long long int *     strideB,
                          const double *      beta,
                          double *            C,
                          const unsigned int *ldc,
                          const int *         batchCount,
                          long long int *     strideC);

      void
      xgemmStridedBatched(const char *                transA,
                          const char *                transB,
                          const unsigned int *        m,
                          const unsigned int *        n,
                          const unsigned int *        k,
                          const std::complex<double> *alpha,
                          const std::complex<double> *A,
                          const unsigned int *        lda,
                          long long int *             strideA,
                          const std::complex<double> *B,
                          const unsigned int *        ldb,
                          long long int *             strideB,
                          const std::complex<double> *beta,
                          std::complex<double> *      C,
                          const unsigned int *        ldc,
                          const int *                 batchCount,
                          long long int *             strideC);

    private:
    };

    class BLASWrapperClass<dftfe::utils::MemorySpace::HOST>
      : public BLASWrapperClassBase<dftfe::utils::MemorySpace::HOST>
    {};

#if defined(DFTFE_WITH_DEVICE)
    class BLASWrapperClass<dftfe::utils::MemorySpace::DEVICE>
      : public BLASWrapperClassBase<dftfe::utils::MemorySpace::DEVICE>
    {
#  ifdef DFTFE_WITH_DEVICE_AMD
      void
      initialize();
#  endif

      deviceBlasStatus_t
      create(deviceBlasHandle_t *pHandle);

      deviceBlasStatus_t
      destroy(deviceBlasHandle_t handle);

      deviceBlasStatus_t
      setStream(deviceBlasHandle_t handle, deviceStream_t stream);

#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      deviceBlasStatus_t
      setMathMode(deviceBlasHandle_t handle, deviceBlasMath_t mathMode);
#  endif
    };

#endif

  } // end of namespace linearAlgebra

} // end of namespace dftfe
#include "../utils/BLASWrapperClass.t.cc"
#include "../utils/BLASWrapperClassHost.t.cc"
#if defined(DFTFE_WITH_DEVICE)
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
#    include "../utils/BLASWrapperClassDevice.t.cu.cc"
#  elif
#    include "../utils/BLASWrapperClassDevice.t.hip.cc"
#  endif
#endif

#endif // BLASWrapperClass_h
