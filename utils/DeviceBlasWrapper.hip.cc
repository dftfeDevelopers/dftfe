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


#ifdef DFTFE_WITH_DEVICE_LANG_HIP
#  include <DeviceBlasWrapper.h>
#  include <stdio.h>
#  include <vector>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <Exceptions.h>
#  include <hipblas.h>
#  ifdef DFTFE_WITH_DEVICE_AMD
#    include <rocblas.h>
#  endif
namespace dftfe
{
  namespace utils
  {
    namespace deviceBlasWrapper
    {
      namespace
      {
        inline hipblasDoubleComplex
        makeDataTypeHipBlasCompatible(std::complex<double> a)
        {
          return hipblasDoubleComplex(a.real(), a.imag());
        }

        inline hipblasComplex
        makeDataTypeHipBlasCompatible(std::complex<float> a)
        {
          return hipblasComplex(a.real(), a.imag());
        }

        inline hipblasComplex *
        makeDataTypeHipBlasCompatible(std::complex<float> *a)
        {
          return reinterpret_cast<hipblasComplex *>(a);
        }

        inline const hipblasComplex *
        makeDataTypeHipBlasCompatible(const std::complex<float> *a)
        {
          return reinterpret_cast<const hipblasComplex *>(a);
        }

        inline hipblasDoubleComplex *
        makeDataTypeHipBlasCompatible(std::complex<double> *a)
        {
          return reinterpret_cast<hipblasDoubleComplex *>(a);
        }

        inline const hipblasDoubleComplex *
        makeDataTypeHipBlasCompatible(const std::complex<double> *a)
        {
          return reinterpret_cast<const hipblasDoubleComplex *>(a);
        }

      } // namespace

#  ifdef DFTFE_WITH_DEVICE_AMD
      void
      initialize()
      {
        rocblas_initialize();
      }
#  endif


      deviceBlasStatus_t
      create(deviceBlasHandle_t *pHandle)
      {
        deviceBlasStatus_t status = hipblasCreate(pHandle);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      destroy(deviceBlasHandle_t handle)
      {
        deviceBlasStatus_t status = hipblasDestroy(handle);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      setStream(deviceBlasHandle_t handle, deviceStream_t stream)
      {
        deviceBlasStatus_t status = hipblasSetStream(handle, stream);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      copy(deviceBlasHandle_t handle,
           int                n,
           const double *     x,
           int                incx,
           double *           y,
           int                incy)
      {
        deviceBlasStatus_t status = hipblasDcopy(handle, n, x, incx, y, incy);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      nrm2(deviceBlasHandle_t handle,
           int                n,
           const double *     x,
           int                incx,
           double *           result)
      {
        deviceBlasStatus_t status = hipblasDnrm2(handle, n, x, incx, result);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      dot(deviceBlasHandle_t handle,
          int                n,
          const double *     x,
          int                incx,
          const double *     y,
          int                incy,
          double *           result)
      {
        deviceBlasStatus_t status =
          hipblasDdot(handle, n, x, incx, y, incy, result);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      axpy(deviceBlasHandle_t handle,
           int                n,
           const double *     alpha,
           const double *     x,
           int                incx,
           double *           y,
           int                incy)
      {
        deviceBlasStatus_t status =
          hipblasDaxpy(handle, n, alpha, x, incx, y, incy);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      gemm(deviceBlasHandle_t    handle,
           deviceBlasOperation_t transa,
           deviceBlasOperation_t transb,
           int                   m,
           int                   n,
           int                   k,
           const double *        alpha,
           const double *        A,
           int                   lda,
           const double *        B,
           int                   ldb,
           const double *        beta,
           double *              C,
           int                   ldc)
      {
        deviceBlasStatus_t status = hipblasDgemm(
          handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      gemm(deviceBlasHandle_t    handle,
           deviceBlasOperation_t transa,
           deviceBlasOperation_t transb,
           int                   m,
           int                   n,
           int                   k,
           const float *         alpha,
           const float *         A,
           int                   lda,
           const float *         B,
           int                   ldb,
           const float *         beta,
           float *               C,
           int                   ldc)
      {
        deviceBlasStatus_t status = hipblasSgemm(
          handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      gemm(deviceBlasHandle_t          handle,
           deviceBlasOperation_t       transa,
           deviceBlasOperation_t       transb,
           int                         m,
           int                         n,
           int                         k,
           const std::complex<double> *alpha,
           const std::complex<double> *A,
           int                         lda,
           const std::complex<double> *B,
           int                         ldb,
           const std::complex<double> *beta,
           std::complex<double> *      C,
           int                         ldc)
      {
        deviceBlasStatus_t status =
          hipblasZgemm(handle,
                       transa,
                       transb,
                       m,
                       n,
                       k,
                       makeDataTypeHipBlasCompatible(alpha),
                       makeDataTypeHipBlasCompatible(A),
                       lda,
                       makeDataTypeHipBlasCompatible(B),
                       ldb,
                       makeDataTypeHipBlasCompatible(beta),
                       makeDataTypeHipBlasCompatible(C),
                       ldc);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      gemm(deviceBlasHandle_t         handle,
           deviceBlasOperation_t      transa,
           deviceBlasOperation_t      transb,
           int                        m,
           int                        n,
           int                        k,
           const std::complex<float> *alpha,
           const std::complex<float> *A,
           int                        lda,
           const std::complex<float> *B,
           int                        ldb,
           const std::complex<float> *beta,
           std::complex<float> *      C,
           int                        ldc)
      {
        deviceBlasStatus_t status =
          hipblasCgemm(handle,
                       transa,
                       transb,
                       m,
                       n,
                       k,
                       makeDataTypeHipBlasCompatible(alpha),
                       makeDataTypeHipBlasCompatible(A),
                       lda,
                       makeDataTypeHipBlasCompatible(B),
                       ldb,
                       makeDataTypeHipBlasCompatible(beta),
                       makeDataTypeHipBlasCompatible(C),
                       ldc);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      gemmBatched(deviceBlasHandle_t    handle,
                  deviceBlasOperation_t transa,
                  deviceBlasOperation_t transb,
                  int                   m,
                  int                   n,
                  int                   k,
                  const double *        alpha,
                  const double *        Aarray[],
                  int                   lda,
                  const double *        Barray[],
                  int                   ldb,
                  const double *        beta,
                  double *              Carray[],
                  int                   ldc,
                  int                   batchCount)
      {
        deviceBlasStatus_t status = hipblasDgemmBatched(handle,
                                                        transa,
                                                        transb,
                                                        m,
                                                        n,
                                                        k,
                                                        alpha,
                                                        Aarray,
                                                        lda,
                                                        Barray,
                                                        ldb,
                                                        beta,
                                                        Carray,
                                                        ldc,
                                                        batchCount);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      gemmBatched(deviceBlasHandle_t          handle,
                  deviceBlasOperation_t       transa,
                  deviceBlasOperation_t       transb,
                  int                         m,
                  int                         n,
                  int                         k,
                  const std::complex<double> *alpha,
                  const std::complex<double> *Aarray[],
                  int                         lda,
                  const std::complex<double> *Barray[],
                  int                         ldb,
                  const std::complex<double> *beta,
                  std::complex<double> *      Carray[],
                  int                         ldc,
                  int                         batchCount)
      {
        deviceBlasStatus_t status =
          hipblasZgemmBatched(handle,
                              transa,
                              transb,
                              m,
                              n,
                              k,
                              (const hipblasDoubleComplex *)alpha,
                              (const hipblasDoubleComplex **)Aarray,
                              lda,
                              (const hipblasDoubleComplex **)Barray,
                              ldb,
                              (const hipblasDoubleComplex *)beta,
                              (hipblasDoubleComplex **)Carray,
                              ldc,
                              batchCount);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      gemmStridedBatched(deviceBlasHandle_t    handle,
                         deviceBlasOperation_t transa,
                         deviceBlasOperation_t transb,
                         int                   m,
                         int                   n,
                         int                   k,
                         const double *        alpha,
                         const double *        A,
                         int                   lda,
                         long long int         strideA,
                         const double *        B,
                         int                   ldb,
                         long long int         strideB,
                         const double *        beta,
                         double *              C,
                         int                   ldc,
                         long long int         strideC,
                         int                   batchCount)
      {
        deviceBlasStatus_t status = hipblasDgemmStridedBatched(handle,
                                                               transa,
                                                               transb,
                                                               m,
                                                               n,
                                                               k,
                                                               alpha,
                                                               A,
                                                               lda,
                                                               strideA,
                                                               B,
                                                               ldb,
                                                               strideB,
                                                               beta,
                                                               C,
                                                               ldc,
                                                               strideC,
                                                               batchCount);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      gemmStridedBatched(deviceBlasHandle_t    handle,
                         deviceBlasOperation_t transa,
                         deviceBlasOperation_t transb,
                         int                   m,
                         int                   n,
                         int                   k,
                         const float *         alpha,
                         const float *         A,
                         int                   lda,
                         long long int         strideA,
                         const float *         B,
                         int                   ldb,
                         long long int         strideB,
                         const float *         beta,
                         float *               C,
                         int                   ldc,
                         long long int         strideC,
                         int                   batchCount)
      {
        deviceBlasStatus_t status = hipblasSgemmStridedBatched(handle,
                                                               transa,
                                                               transb,
                                                               m,
                                                               n,
                                                               k,
                                                               alpha,
                                                               A,
                                                               lda,
                                                               strideA,
                                                               B,
                                                               ldb,
                                                               strideB,
                                                               beta,
                                                               C,
                                                               ldc,
                                                               strideC,
                                                               batchCount);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }


      deviceBlasStatus_t
      gemmStridedBatched(deviceBlasHandle_t          handle,
                         deviceBlasOperation_t       transa,
                         deviceBlasOperation_t       transb,
                         int                         m,
                         int                         n,
                         int                         k,
                         const std::complex<double> *alpha,
                         const std::complex<double> *A,
                         int                         lda,
                         long long int               strideA,
                         const std::complex<double> *B,
                         int                         ldb,
                         long long int               strideB,
                         const std::complex<double> *beta,
                         std::complex<double> *      C,
                         int                         ldc,
                         long long int               strideC,
                         int                         batchCount)
      {
        deviceBlasStatus_t status =
          hipblasZgemmStridedBatched(handle,
                                     transa,
                                     transb,
                                     m,
                                     n,
                                     k,
                                     makeDataTypeHipBlasCompatible(alpha),
                                     makeDataTypeHipBlasCompatible(A),
                                     lda,
                                     strideA,
                                     makeDataTypeHipBlasCompatible(B),
                                     ldb,
                                     strideB,
                                     makeDataTypeHipBlasCompatible(beta),
                                     makeDataTypeHipBlasCompatible(C),
                                     ldc,
                                     strideC,
                                     batchCount);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      gemmStridedBatched(deviceBlasHandle_t         handle,
                         deviceBlasOperation_t      transa,
                         deviceBlasOperation_t      transb,
                         int                        m,
                         int                        n,
                         int                        k,
                         const std::complex<float> *alpha,
                         const std::complex<float> *A,
                         int                        lda,
                         long long int              strideA,
                         const std::complex<float> *B,
                         int                        ldb,
                         long long int              strideB,
                         const std::complex<float> *beta,
                         std::complex<float> *      C,
                         int                        ldc,
                         long long int              strideC,
                         int                        batchCount)
      {
        deviceBlasStatus_t status =
          hipblasCgemmStridedBatched(handle,
                                     transa,
                                     transb,
                                     m,
                                     n,
                                     k,
                                     makeDataTypeHipBlasCompatible(alpha),
                                     makeDataTypeHipBlasCompatible(A),
                                     lda,
                                     strideA,
                                     makeDataTypeHipBlasCompatible(B),
                                     ldb,
                                     strideB,
                                     makeDataTypeHipBlasCompatible(beta),
                                     makeDataTypeHipBlasCompatible(C),
                                     ldc,
                                     strideC,
                                     batchCount);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }
      deviceBlasStatus_t
      gemv(deviceBlasHandle_t    handle,
           deviceBlasOperation_t trans,
           int                   m,
           int                   n,
           const double *        alpha,
           const double *        A,
           int                   lda,
           const double *        x,
           int                   incx,
           const double *        beta,
           double *              y,
           int                   incy)
      {
        deviceBlasStatus_t status = hipblasDgemv(
          handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      gemv(deviceBlasHandle_t    handle,
           deviceBlasOperation_t trans,
           int                   m,
           int                   n,
           const float *         alpha,
           const float *         A,
           int                   lda,
           const float *         x,
           int                   incx,
           const float *         beta,
           float *               y,
           int                   incy)
      {
        deviceBlasStatus_t status = hipblasSgemv(
          handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      gemv(deviceBlasHandle_t          handle,
           deviceBlasOperation_t       trans,
           int                         m,
           int                         n,
           const std::complex<double> *alpha,
           const std::complex<double> *A,
           int                         lda,
           const std::complex<double> *x,
           int                         incx,
           const std::complex<double> *beta,
           std::complex<double> *      y,
           int                         incy)
      {
        deviceBlasStatus_t status =
          hipblasZgemv(handle,
                       trans,
                       m,
                       n,
                       makeDataTypeHipBlasCompatible(alpha),
                       makeDataTypeHipBlasCompatible(A),
                       lda,
                       makeDataTypeHipBlasCompatible(x),
                       incx,
                       makeDataTypeHipBlasCompatible(beta),
                       makeDataTypeHipBlasCompatible(y),
                       incy);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      gemv(deviceBlasHandle_t         handle,
           deviceBlasOperation_t      trans,
           int                        m,
           int                        n,
           const std::complex<float> *alpha,
           const std::complex<float> *A,
           int                        lda,
           const std::complex<float> *x,
           int                        incx,
           const std::complex<float> *beta,
           std::complex<float> *      y,
           int                        incy)
      {
        deviceBlasStatus_t status =
          hipblasCgemv(handle,
                       trans,
                       m,
                       n,
                       makeDataTypeHipBlasCompatible(alpha),
                       makeDataTypeHipBlasCompatible(A),
                       lda,
                       makeDataTypeHipBlasCompatible(x),
                       incx,
                       makeDataTypeHipBlasCompatible(beta),
                       makeDataTypeHipBlasCompatible(y),
                       incy);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

    } // namespace deviceBlasWrapper
  }   // namespace utils
} // namespace dftfe
#endif
