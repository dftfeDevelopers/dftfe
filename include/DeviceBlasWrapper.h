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

#ifdef DFTFE_WITH_DEVICE

#  ifndef dftfeDeviceBlasWrapper_H
#    define dftfeDeviceBlasWrapper_H

#    include <complex>
#    include <TypeConfig.h>
#    include <DeviceTypeConfig.h>
namespace dftfe
{
  namespace utils
  {
    namespace deviceBlasWrapper
    {
#    ifdef DFTFE_WITH_DEVICE_AMD
      void
      initialize();
#    endif

      deviceBlasStatus_t
      create(deviceBlasHandle_t *pHandle);

      deviceBlasStatus_t
      destroy(deviceBlasHandle_t handle);

      deviceBlasStatus_t
      setStream(deviceBlasHandle_t handle, deviceStream_t stream);

#    ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      deviceBlasStatus_t
      setMathMode(deviceBlasHandle_t handle, deviceBlasMath_t mathMode);
#    endif

      deviceBlasStatus_t
      copy(deviceBlasHandle_t handle,
           int                n,
           const double *     x,
           int                incx,
           double *           y,
           int                incy);

      deviceBlasStatus_t
      nrm2(deviceBlasHandle_t handle,
           int                n,
           const double *     x,
           int                incx,
           double *           result);

      deviceBlasStatus_t
      dot(deviceBlasHandle_t handle,
          int                n,
          const double *     x,
          int                incx,
          const double *     y,
          int                incy,
          double *           result);

      deviceBlasStatus_t
      axpy(deviceBlasHandle_t handle,
           int                n,
           const double *     alpha,
           const double *     x,
           int                incx,
           double *           y,
           int                incy);

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
           int                   ldc);

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
           int                   ldc);

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
           int                         ldc);

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
           int                        ldc);

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
                  int                   batchCount);

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
                  int                         batchCount);

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
                         int                   batchCount);


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
                         int                   batchCount);

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
                         int                         batchCount);

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
                         int                        batchCount);

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
           int                   incy);

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
           int                   incy);

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
           int                         incy);

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
           int                        incy);


    } // namespace deviceBlasWrapper
  }   // namespace utils
} // namespace dftfe

#  endif // dftfeDeviceBlasWrapper_H
#endif   // DFTFE_WITH_DEVICE
