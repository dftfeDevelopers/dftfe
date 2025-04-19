// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
           dftfe::Int         n,
           const double      *x,
           dftfe::Int         incx,
           double            *y,
           dftfe::Int         incy);

      deviceBlasStatus_t
      nrm2(deviceBlasHandle_t handle,
           dftfe::Int         n,
           const double      *x,
           dftfe::Int         incx,
           double            *result);

      deviceBlasStatus_t
      dot(deviceBlasHandle_t handle,
          dftfe::Int         n,
          const double      *x,
          dftfe::Int         incx,
          const double      *y,
          dftfe::Int         incy,
          double            *result);

      deviceBlasStatus_t
      axpy(deviceBlasHandle_t handle,
           dftfe::Int         n,
           const double      *alpha,
           const double      *x,
           dftfe::Int         incx,
           double            *y,
           dftfe::Int         incy);

      deviceBlasStatus_t
      gemm(deviceBlasHandle_t    handle,
           deviceBlasOperation_t transa,
           deviceBlasOperation_t transb,
           dftfe::Int            m,
           dftfe::Int            n,
           dftfe::Int            k,
           const double         *alpha,
           const double         *A,
           dftfe::Int            lda,
           const double         *B,
           dftfe::Int            ldb,
           const double         *beta,
           double               *C,
           dftfe::Int            ldc);

      deviceBlasStatus_t
      gemm(deviceBlasHandle_t    handle,
           deviceBlasOperation_t transa,
           deviceBlasOperation_t transb,
           dftfe::Int            m,
           dftfe::Int            n,
           dftfe::Int            k,
           const float          *alpha,
           const float          *A,
           dftfe::Int            lda,
           const float          *B,
           dftfe::Int            ldb,
           const float          *beta,
           float                *C,
           dftfe::Int            ldc);

      deviceBlasStatus_t
      gemm(deviceBlasHandle_t          handle,
           deviceBlasOperation_t       transa,
           deviceBlasOperation_t       transb,
           dftfe::Int                  m,
           dftfe::Int                  n,
           dftfe::Int                  k,
           const std::complex<double> *alpha,
           const std::complex<double> *A,
           dftfe::Int                  lda,
           const std::complex<double> *B,
           dftfe::Int                  ldb,
           const std::complex<double> *beta,
           std::complex<double>       *C,
           dftfe::Int                  ldc);

      deviceBlasStatus_t
      gemm(deviceBlasHandle_t         handle,
           deviceBlasOperation_t      transa,
           deviceBlasOperation_t      transb,
           dftfe::Int                 m,
           dftfe::Int                 n,
           dftfe::Int                 k,
           const std::complex<float> *alpha,
           const std::complex<float> *A,
           dftfe::Int                 lda,
           const std::complex<float> *B,
           dftfe::Int                 ldb,
           const std::complex<float> *beta,
           std::complex<float>       *C,
           dftfe::Int                 ldc);

      deviceBlasStatus_t
      gemmBatched(deviceBlasHandle_t    handle,
                  deviceBlasOperation_t transa,
                  deviceBlasOperation_t transb,
                  dftfe::Int            m,
                  dftfe::Int            n,
                  dftfe::Int            k,
                  const double         *alpha,
                  const double         *Aarray[],
                  dftfe::Int            lda,
                  const double         *Barray[],
                  dftfe::Int            ldb,
                  const double         *beta,
                  double               *Carray[],
                  dftfe::Int            ldc,
                  dftfe::Int            batchCount);

      deviceBlasStatus_t
      gemmBatched(deviceBlasHandle_t          handle,
                  deviceBlasOperation_t       transa,
                  deviceBlasOperation_t       transb,
                  dftfe::Int                  m,
                  dftfe::Int                  n,
                  dftfe::Int                  k,
                  const std::complex<double> *alpha,
                  const std::complex<double> *Aarray[],
                  dftfe::Int                  lda,
                  const std::complex<double> *Barray[],
                  dftfe::Int                  ldb,
                  const std::complex<double> *beta,
                  std::complex<double>       *Carray[],
                  dftfe::Int                  ldc,
                  dftfe::Int                  batchCount);

      deviceBlasStatus_t
      gemmStridedBatched(deviceBlasHandle_t    handle,
                         deviceBlasOperation_t transa,
                         deviceBlasOperation_t transb,
                         dftfe::Int            m,
                         dftfe::Int            n,
                         dftfe::Int            k,
                         const double         *alpha,
                         const double         *A,
                         dftfe::Int            lda,
                         long long int         strideA,
                         const double         *B,
                         dftfe::Int            ldb,
                         long long int         strideB,
                         const double         *beta,
                         double               *C,
                         dftfe::Int            ldc,
                         long long int         strideC,
                         dftfe::Int            batchCount);


      deviceBlasStatus_t
      gemmStridedBatched(deviceBlasHandle_t    handle,
                         deviceBlasOperation_t transa,
                         deviceBlasOperation_t transb,
                         dftfe::Int            m,
                         dftfe::Int            n,
                         dftfe::Int            k,
                         const float          *alpha,
                         const float          *A,
                         dftfe::Int            lda,
                         long long int         strideA,
                         const float          *B,
                         dftfe::Int            ldb,
                         long long int         strideB,
                         const float          *beta,
                         float                *C,
                         dftfe::Int            ldc,
                         long long int         strideC,
                         dftfe::Int            batchCount);

      deviceBlasStatus_t
      gemmStridedBatched(deviceBlasHandle_t          handle,
                         deviceBlasOperation_t       transa,
                         deviceBlasOperation_t       transb,
                         dftfe::Int                  m,
                         dftfe::Int                  n,
                         dftfe::Int                  k,
                         const std::complex<double> *alpha,
                         const std::complex<double> *A,
                         dftfe::Int                  lda,
                         long long int               strideA,
                         const std::complex<double> *B,
                         dftfe::Int                  ldb,
                         long long int               strideB,
                         const std::complex<double> *beta,
                         std::complex<double>       *C,
                         dftfe::Int                  ldc,
                         long long int               strideC,
                         dftfe::Int                  batchCount);

      deviceBlasStatus_t
      gemmStridedBatched(deviceBlasHandle_t         handle,
                         deviceBlasOperation_t      transa,
                         deviceBlasOperation_t      transb,
                         dftfe::Int                 m,
                         dftfe::Int                 n,
                         dftfe::Int                 k,
                         const std::complex<float> *alpha,
                         const std::complex<float> *A,
                         dftfe::Int                 lda,
                         long long int              strideA,
                         const std::complex<float> *B,
                         dftfe::Int                 ldb,
                         long long int              strideB,
                         const std::complex<float> *beta,
                         std::complex<float>       *C,
                         dftfe::Int                 ldc,
                         long long int              strideC,
                         dftfe::Int                 batchCount);

      deviceBlasStatus_t
      gemv(deviceBlasHandle_t    handle,
           deviceBlasOperation_t trans,
           dftfe::Int            m,
           dftfe::Int            n,
           const double         *alpha,
           const double         *A,
           dftfe::Int            lda,
           const double         *x,
           dftfe::Int            incx,
           const double         *beta,
           double               *y,
           dftfe::Int            incy);

      deviceBlasStatus_t
      gemv(deviceBlasHandle_t    handle,
           deviceBlasOperation_t trans,
           dftfe::Int            m,
           dftfe::Int            n,
           const float          *alpha,
           const float          *A,
           dftfe::Int            lda,
           const float          *x,
           dftfe::Int            incx,
           const float          *beta,
           float                *y,
           dftfe::Int            incy);

      deviceBlasStatus_t
      gemv(deviceBlasHandle_t          handle,
           deviceBlasOperation_t       trans,
           dftfe::Int                  m,
           dftfe::Int                  n,
           const std::complex<double> *alpha,
           const std::complex<double> *A,
           dftfe::Int                  lda,
           const std::complex<double> *x,
           dftfe::Int                  incx,
           const std::complex<double> *beta,
           std::complex<double>       *y,
           dftfe::Int                  incy);

      deviceBlasStatus_t
      gemv(deviceBlasHandle_t         handle,
           deviceBlasOperation_t      trans,
           dftfe::Int                 m,
           dftfe::Int                 n,
           const std::complex<float> *alpha,
           const std::complex<float> *A,
           dftfe::Int                 lda,
           const std::complex<float> *x,
           dftfe::Int                 incx,
           const std::complex<float> *beta,
           std::complex<float>       *y,
           dftfe::Int                 incy);


    } // namespace deviceBlasWrapper
  }   // namespace utils
} // namespace dftfe

#  endif // dftfeDeviceBlasWrapper_H
#endif   // DFTFE_WITH_DEVICE
