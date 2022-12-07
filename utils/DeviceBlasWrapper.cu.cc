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


#ifdef DFTFE_WITH_DEVICE_CUDA
#  include <DeviceBlasWrapper.h>
#  include <stdio.h>
#  include <vector>
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <Exceptions.h>
#include <cublas_v2.h>
namespace dftfe
{
  namespace utils
  {
    namespace deviceBlasWrapper
    {
      deviceStatus_t
      create(deviceBlasHandle_t   *pHandle)
      {
        deviceStatus_t status=cublasCreate(pHandle);
        DEVICEBLAS_API_CHECK(status);
        return status;        
      }

      deviceStatus_t
      destroy(deviceBlasHandle_t   handle)
      {
        deviceStatus_t status=cublasDestroy(handle);
        DEVICEBLAS_API_CHECK(status);
        return status;        
      }

      deviceStatus_t
      setStream(deviceBlasHandle_t handle, deviceStream_t stream)
      {
        deviceStatus_t status=cublasSetStream(handle,stream);
        DEVICEBLAS_API_CHECK(status);
        return status;        
      }

      deviceStatus_t
      setMathMode(deviceBlasHandle_t handle, deviceMath_t mathMode)
      {
        deviceStatus_t status=cublasSetMathMode(handle,mathMode);
        DEVICEBLAS_API_CHECK(status);
        return status;        
      }

      deviceStatus_t
      copy(deviceBlasHandle_t handle,
           int n,
           const double          *x,
           int incx,
           double                *y,
           int incy) 
      {
        deviceStatus_t status=cublasDcopy(handle,n,x,incx,y,incy);
        DEVICEBLAS_API_CHECK(status);
        return status;        
      }      

      deviceStatus_t
      nrm2(deviceBlasHandle_t handle,
           int n,
           const double          *x,
           int incx,
           double *result)
      {
        deviceStatus_t status=cublasDnrm2(handle,n,x,incx,result);
        DEVICEBLAS_API_CHECK(status);
        return status;        
      } 

      deviceStatus_t
      dot(deviceBlasHandle_t handle,
          int n,
          const double          *x,
          int incx,
          const double          *y,
          int incy,
          double          *result)
      {
        deviceStatus_t status=cublasDdot(handle,n,x,incx,y,incy,result);
        DEVICEBLAS_API_CHECK(status);
        return status;        
      } 

      deviceStatus_t
      axpy(deviceBlasHandle_t handle,
           int n,
           const double          *alpha,
           const double          *x,
           int incx,
           double                *y,
           int incy)
      {
        deviceStatus_t status=cublasDaxpy(handle,n,alpha,x,incx,y,incy);
        DEVICEBLAS_API_CHECK(status);
        return status;        
      }

      deviceStatus_t
      gemm(deviceBlasHandle_t    handle,
                  deviceBlasOperation_t transa,
                  deviceBlasOperation_t transb,
                  int               m,
                  int               n,
                  int               k,
                  const double *    alpha,
                  const double *    A,
                  int               lda,
                  const double *    B,
                  int               ldb,
                  const double *    beta,
                  double *          C,
                  int               ldc)
      {
        deviceStatus_t status= cublasDgemm(
          handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceStatus_t
      gemm(deviceBlasHandle_t    handle,
                  deviceBlasOperation_t transa,
                  deviceBlasOperation_t transb,
                  int               m,
                  int               n,
                  int               k,
                  const float *    alpha,
                  const float *    A,
                  int               lda,
                  const float *    B,
                  int               ldb,
                  const float *    beta,
                  float *          C,
                  int               ldc)
      {
        deviceStatus_t status= cublasSgemm(
          handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceStatus_t
      gemm(deviceBlasHandle_t    handle,
                  deviceBlasOperation_t transa,
                  deviceBlasOperation_t transb,
                  int               m,
                  int               n,
                  int               k,
                  const std::complex<double> *    alpha,
                  const std::complex<double> *    A,
                  int               lda,
                  const std::complex<double> *    B,
                  int               ldb,
                  const std::complex<double> *    beta,
                  std::complex<double> *          C,
                  int               ldc)
      {
        deviceStatus_t status= cublasZgemm(
          handle, transa, transb, m, n, k, dftfe::utils::makeDataTypeDeviceCompatible(alpha), dftfe::utils::makeDataTypeDeviceCompatible(A), lda, dftfe::utils::makeDataTypeDeviceCompatible(B), ldb, dftfe::utils::makeDataTypeDeviceCompatible(beta), dftfe::utils::makeDataTypeDeviceCompatible(C), ldc);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceStatus_t
      gemm(deviceBlasHandle_t    handle,
                  deviceBlasOperation_t transa,
                  deviceBlasOperation_t transb,
                  int               m,
                  int               n,
                  int               k,
                  const std::complex<float> *    alpha,
                  const std::complex<float> *    A,
                  int               lda,
                  const std::complex<float> *    B,
                  int               ldb,
                  const std::complex<float> *    beta,
                  std::complex<float> *          C,
                  int               ldc)
      {
        deviceStatus_t status= cublasCgemm(
          handle, transa, transb, m, n, k, dftfe::utils::makeDataTypeDeviceCompatible(alpha), dftfe::utils::makeDataTypeDeviceCompatible(A), lda, dftfe::utils::makeDataTypeDeviceCompatible(B), ldb, dftfe::utils::makeDataTypeDeviceCompatible(beta), dftfe::utils::makeDataTypeDeviceCompatible(C), ldc);
        DEVICEBLAS_API_CHECK(status);
        return status;
      }

      deviceStatus_t
      gemmBatched(deviceBlasHandle_t    handle,
                     deviceBlasOperation_t transa,
                     deviceBlasOperation_t transb,
                     int               m,
                     int               n,
                     int               k,
                     const double *    alpha,
                     const double *    Aarray[],
                     int               lda,
                     const double *    Barray[],
                     int               ldb,
                     const double *    beta,
                     double *          Carray[],
                     int               ldc,
                     int               batchCount)
      {
        deviceStatus_t status= cublasDgemmBatched(handle,
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

      deviceStatus_t
      gemmBatched(deviceBlasHandle_t    handle,
                     deviceBlasOperation_t transa,
                     deviceBlasOperation_t transb,
                     int               m,
                     int               n,
                     int               k,
                     const std::complex<double> *    alpha,
                     const std::complex<double> *    Aarray[],
                     int               lda,
                     const std::complex<double> *    Barray[],
                     int               ldb,
                     const std::complex<double> *    beta,
                     std::complex<double> *          Carray[],
                     int               ldc,
                     int               batchCount)
      {
        deviceStatus_t status= cublasZgemmBatched(handle,
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

      deviceStatus_t
      gemmStridedBatched(deviceBlasHandle_t         handle,
                            deviceBlasOperation_t      transa,
                            deviceBlasOperation_t      transb,
                            int                    m,
                            int                    n,
                            int                    k,
                            const double *alpha,
                            const double *A,
                            int                    lda,
                            long long int          strideA,
                            const double *B,
                            int                    ldb,
                            long long int          strideB,
                            const double *beta,
                            double *      C,
                            int                    ldc,
                            long long int          strideC,
                            int                    batchCount)
      {
        deviceStatus_t status= cublasDgemmStridedBatched(handle,
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

      deviceStatus_t
      gemmStridedBatched(deviceBlasHandle_t         handle,
                            deviceBlasOperation_t      transa,
                            deviceBlasOperation_t      transb,
                            int                    m,
                            int                    n,
                            int                    k,
                            const float *alpha,
                            const float *A,
                            int                    lda,
                            long long int          strideA,
                            const float *B,
                            int                    ldb,
                            long long int          strideB,
                            const float *beta,
                            float *      C,
                            int                    ldc,
                            long long int          strideC,
                            int                    batchCount)
      {
        deviceStatus_t status= cublasSgemmStridedBatched(handle,
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


      deviceStatus_t
      gemmStridedBatched(deviceBlasHandle_t    handle,
                            deviceBlasOperation_t transa,
                            deviceBlasOperation_t transb,
                            int               m,
                            int               n,
                            int               k,
                            const std::complex<double> *    alpha,
                            const std::complex<double> *    A,
                            int               lda,
                            long long int     strideA,
                            const std::complex<double> *    B,
                            int               ldb,
                            long long int     strideB,
                            const std::complex<double> *    beta,
                            std::complex<double> *          C,
                            int               ldc,
                            long long int     strideC,
                            int               batchCount)
      {
        deviceStatus_t status= cublasZgemmStridedBatched(handle,
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

      deviceStatus_t
      gemmStridedBatched(deviceBlasHandle_t    handle,
                            deviceBlasOperation_t transa,
                            deviceBlasOperation_t transb,
                            int               m,
                            int               n,
                            int               k,
                            const std::complex<float> *    alpha,
                            const std::complex<float> *    A,
                            int               lda,
                            long long int     strideA,
                            const std::complex<float> *    B,
                            int               ldb,
                            long long int     strideB,
                            const std::complex<float> *    beta,
                            std::complex<float> *          C,
                            int               ldc,
                            long long int     strideC,
                            int               batchCount)
      {
        deviceStatus_t status= cublasCgemmStridedBatched(handle,
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

    }//namespace deviceBlasWrapper
  } // namespace utils
} // namespace dftfe
#endif
