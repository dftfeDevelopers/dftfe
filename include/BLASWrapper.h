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

#ifndef BLASWrapper_h
#define BLASWrapper_h

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
    class BLASWrapper;

    template<>
    class BLASWrapper<dftfe::utils::MemorySpace::HOST>
    {
    public:
      // BLASWrapper()const;
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
            const unsigned int *ldc) const;
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
            const unsigned int *       ldc) const;

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
            const unsigned int *ldc) const;


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
            const unsigned int *        ldc) const;

      // Real-Double scaling of Real-vector
      void
      xscal(const unsigned int *n,
            const double *      alpha,
            double *            x,
            const unsigned int *inc) const;

      // Real-Float scaling of Real-vector
      void
      xscal(const unsigned int *n,
            const float *       alpha,
            float *             x,
            const unsigned int *inc) const;

      // Complex-double scaling of complex-vector
      void
      xscal(const unsigned int *        n,
            const std::complex<double> *alpha,
            std::complex<double> *      x,
            const unsigned int *        inc) const;

      // Real-double scaling of complex-vector
      void
      xscal(const unsigned int *  n,
            const double *        alpha,
            std::complex<double> *x,
            const unsigned int *  inc) const;

      // Real double Norm2
      void
      xnrm2(const unsigned int *n,
            const double *      x,
            const unsigned int *incx) const;

      // Real dot product
      void
      xdot(const unsigned int *N,
           const double *      X,
           const unsigned int *INCX,
           const double *      Y,
           const unsigned int *INCY,
           double * result) const;

      // Real double Ax+y
      void
      xaxpy(const unsigned int *n,
            const double *      alpha,
            double *            x,
            const unsigned int *incx,
            double *            y,
            const unsigned int *incy) const;

      // Complex double Ax+y
      void
      xaxpy(const unsigned int *        n,
            const std::complex<double> *alpha,
            std::complex<double> *      x,
            const unsigned int *        incx,
            std::complex<double> *      y,
            const unsigned int *        incy) const;

      // Real copy of double data
      void
      xcopy(const unsigned int *n,
            const double *      x,
            const unsigned int *incx,
            double *            y,
            const unsigned int *incy) const;

      // Complex double copy of data
      void
      xcopy(const unsigned int *        n,
            const std::complex<double> *x,
            const unsigned int *        incx,
            std::complex<double> *      y,
            const unsigned int *        incy) const;

      // Real copy of float data
      void
      xcopy(const unsigned int *n,
            const float *       x,
            const unsigned int *incx,
            float *             y,
            const unsigned int *incy) const;

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
            const unsigned int *INCY) const;

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
                   const int *         batchCount) const;

      void
      xgemmBatched(const char *                transA,
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
                   const unsigned int *        ldc,
                   const int *                 batchCount) const;


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
                          long long int *     strideC) const;

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
                          long long int *             strideC) const;

    private:
    };
#if defined(DFTFE_WITH_DEVICE)
    template <>
    class BLASWrapper<dftfe::utils::MemorySpace::DEVICE>
    {
    public:
      // BLASWrapper()const;
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
            const unsigned int *ldc) const;
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
            const unsigned int *       ldc) const;

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
            const unsigned int *ldc) const;


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
            const unsigned int *        ldc) const;

      // Real-Double scaling of Real-vector
      void
      xscal(const unsigned int *n,
            const double *      alpha,
            double *            x,
            const unsigned int *inc) const;

      // Real-Float scaling of Real-vector
      void
      xscal(const unsigned int *n,
            const float *       alpha,
            float *             x,
            const unsigned int *inc) const;

      // Complex-double scaling of complex-vector
      void
      xscal(const unsigned int *        n,
            const std::complex<double> *alpha,
            std::complex<double> *      x,
            const unsigned int *        inc) const;

      // Real-double scaling of complex-vector
      void
      xscal(const unsigned int *  n,
            const double *        alpha,
            std::complex<double> *x,
            const unsigned int *  inc) const;

      // Real double Norm2
      void
      xnrm2(const unsigned int *n,
            const double *      x,
            const unsigned int *incx) const;

      // Real dot product
      void
      xdot(const unsigned int *N,
           const double *      X,
           const unsigned int *INCX,
           const double *      Y,
           const unsigned int *INCY,
           double * result) const;

      // Real double Ax+y
      void
      xaxpy(const unsigned int *n,
            const double *      alpha,
            double *            x,
            const unsigned int *incx,
            double *            y,
            const unsigned int *incy) const;

      // Complex double Ax+y
      void
      xaxpy(const unsigned int *        n,
            const std::complex<double> *alpha,
            std::complex<double> *      x,
            const unsigned int *        incx,
            std::complex<double> *      y,
            const unsigned int *        incy) const;

      // Real copy of double data
      void
      xcopy(const unsigned int *n,
            const double *      x,
            const unsigned int *incx,
            double *            y,
            const unsigned int *incy) const;

      // Complex double copy of data
      void
      xcopy(const unsigned int *        n,
            const std::complex<double> *x,
            const unsigned int *        incx,
            std::complex<double> *      y,
            const unsigned int *        incy) const;

      // Real copy of float data
      void
      xcopy(const unsigned int *n,
            const float *       x,
            const unsigned int *incx,
            float *             y,
            const unsigned int *incy) const;

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
            const unsigned int *INCY) const;

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
                   const int *         batchCount) const;

      void
      xgemmBatched(const char *                transA,
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
                   const unsigned int *        ldc,
                   const int *                 batchCount) const;


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
                          long long int *     strideC) const;

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
                          long long int *             strideC) const;

    private:
#  ifdef DFTFE_WITH_DEVICE_AMD
      void
      initialize() ;
#  endif

      /// storage for deviceblas handle
      dftfe::utils::deviceBlasHandle_t d_deviceBlasHandle;


      dftfe::utils::deviceBlasStatus_t
      create() ;

      dftfe::utils::deviceBlasStatus_t
      destroy() ;

      dftfe::utils::deviceBlasStatus_t
      setStream(dftfe::utils::deviceStream_t     stream) ;

#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      dftfe::utils::deviceBlasStatus_t
      setMathMode(dftfe::utils::deviceBlasMath_t   mathMode) ;
#  endif
    };
#endif

  } // end of namespace linearAlgebra

} // end of namespace dftfe
//#include "../utils/BLASWrapper.t.cc"
//#include "../utils/BLASWrapperHost.t.cc"
// #if defined(DFTFE_WITH_DEVICE)
// #  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
// #    include "../utils/BLASWrapperDevice.t.cu.cc"
// #  elif
// #    include "../utils/BLASWrapperDevice.t.hip.cc"
// #  endif
// #endif

#endif // BLASWrapper_h
