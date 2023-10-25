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


#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
#  include <DeviceBlasWrapper.h>
#  include <stdio.h>
#  include <vector>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <Exceptions.h>
#  include <cublas_v2.h>
namespace dftfe
{
  namespace utils
  {
    namespace deviceBlasWrapper
    {
      deviceBlasStatus_t
      create(deviceBlasHandle_t *pHandle)
      {
        deviceBlasStatus_t status = cublasCreate(pHandle);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      destroy(deviceBlasHandle_t handle)
      {
        deviceBlasStatus_t status = cublasDestroy(handle);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      setStream(deviceBlasHandle_t handle, deviceStream_t stream)
      {
        deviceBlasStatus_t status = cublasSetStream(handle, stream);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceBlasStatus_t
      setMathMode(deviceBlasHandle_t handle, deviceBlasMath_t mathMode)
      {
        deviceBlasStatus_t status = cublasSetMathMode(handle, mathMode);
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
        deviceBlasStatus_t status = cublasDcopy(handle, n, x, incx, y, incy);
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
        deviceBlasStatus_t status = cublasDnrm2(handle, n, x, incx, result);
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
          cublasDdot(handle, n, x, incx, y, incy, result);
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
          cublasDaxpy(handle, n, alpha, x, incx, y, incy);
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
        deviceBlasStatus_t status = cublasDgemm(
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
        deviceBlasStatus_t status = cublasSgemm(
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
          cublasZgemm(handle,
                      transa,
                      transb,
                      m,
                      n,
                      k,
                      dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                      dftfe::utils::makeDataTypeDeviceCompatible(A),
                      lda,
                      dftfe::utils::makeDataTypeDeviceCompatible(B),
                      ldb,
                      dftfe::utils::makeDataTypeDeviceCompatible(beta),
                      dftfe::utils::makeDataTypeDeviceCompatible(C),
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
          cublasCgemm(handle,
                      transa,
                      transb,
                      m,
                      n,
                      k,
                      dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                      dftfe::utils::makeDataTypeDeviceCompatible(A),
                      lda,
                      dftfe::utils::makeDataTypeDeviceCompatible(B),
                      ldb,
                      dftfe::utils::makeDataTypeDeviceCompatible(beta),
                      dftfe::utils::makeDataTypeDeviceCompatible(C),
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
        deviceBlasStatus_t status = cublasDgemmBatched(handle,
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
          cublasZgemmBatched(handle,
                             transa,
                             transb,
                             m,
                             n,
                             k,
                             (const deviceDoubleComplex *)alpha,
                             (const deviceDoubleComplex **)Aarray,
                             lda,
                             (const deviceDoubleComplex **)Barray,
                             ldb,
                             (const deviceDoubleComplex *)beta,
                             (deviceDoubleComplex **)Carray,
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
        deviceBlasStatus_t status = cublasDgemmStridedBatched(handle,
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
        deviceBlasStatus_t status = cublasSgemmStridedBatched(handle,
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
        deviceBlasStatus_t status = cublasZgemmStridedBatched(
          handle,
          transa,
          transb,
          m,
          n,
          k,
          dftfe::utils::makeDataTypeDeviceCompatible(alpha),
          dftfe::utils::makeDataTypeDeviceCompatible(A),
          lda,
          strideA,
          dftfe::utils::makeDataTypeDeviceCompatible(B),
          ldb,
          strideB,
          dftfe::utils::makeDataTypeDeviceCompatible(beta),
          dftfe::utils::makeDataTypeDeviceCompatible(C),
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
        deviceBlasStatus_t status = cublasCgemmStridedBatched(
          handle,
          transa,
          transb,
          m,
          n,
          k,
          dftfe::utils::makeDataTypeDeviceCompatible(alpha),
          dftfe::utils::makeDataTypeDeviceCompatible(A),
          lda,
          strideA,
          dftfe::utils::makeDataTypeDeviceCompatible(B),
          ldb,
          strideB,
          dftfe::utils::makeDataTypeDeviceCompatible(beta),
          dftfe::utils::makeDataTypeDeviceCompatible(C),
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
        deviceBlasStatus_t status = cublasDgemv(
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
        deviceBlasStatus_t status = cublasSgemv(
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
          cublasZgemv(handle,
                      trans,
                      m,
                      n,
                      dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                      dftfe::utils::makeDataTypeDeviceCompatible(A),
                      lda,
                      dftfe::utils::makeDataTypeDeviceCompatible(x),
                      incx,
                      dftfe::utils::makeDataTypeDeviceCompatible(beta),
                      dftfe::utils::makeDataTypeDeviceCompatible(y),
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
          cublasCgemv(handle,
                      trans,
                      m,
                      n,
                      dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                      dftfe::utils::makeDataTypeDeviceCompatible(A),
                      lda,
                      dftfe::utils::makeDataTypeDeviceCompatible(x),
                      incx,
                      dftfe::utils::makeDataTypeDeviceCompatible(beta),
                      dftfe::utils::makeDataTypeDeviceCompatible(y),
                      incy);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

    } // namespace deviceBlasWrapper
  }   // namespace utils
} // namespace dftfe
#endif
