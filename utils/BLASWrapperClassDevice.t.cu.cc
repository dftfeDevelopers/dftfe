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

#include <BLASWrapperClass.h>
#include <deviceKernelsGeneric.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherConstants.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>

namespace dftfe
{
  namespace linearAlgebra
  {
    void
    BLASWrapperClass<dftfe::utils::MemorySpace::DEVICE>::xgemm(
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
      const unsigned int *ldc)
    {
      dftfe::utils::deviceBlasStatus_t status = cublasSgemm(d_deviceBlasHandle,
                                              transA,
                                              transB,
                                              m,
                                              n,
                                              k,
                                              alpha,
                                              A,
                                              lda,
                                              B,
                                              ldb,
                                              beta,
                                              C,
                                              ldc);
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapperClass<dftfe::utils::MemorySpace::DEVICE>::xgemm(
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
      const unsigned int *       ldc)
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasCgemm(d_deviceBlasHandle,
                    transA,
                    transB,
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
    }

    void
    BLASWrapperClass<dftfe::utils::MemorySpace::DEVICE>::xgemm(
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
      const unsigned int *ldc)
    {
      dftfe::utils::deviceBlasStatus_t status = cublasDgemm(d_deviceBlasHandle,
                                              transA,
                                              transB,
                                              m,
                                              n,
                                              k,
                                              alpha,
                                              A,
                                              lda,
                                              B,
                                              ldb,
                                              beta,
                                              C,
                                              ldc);
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapperClass<dftfe::utils::MemorySpace::DEVICE>::xgemm(
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
      const unsigned int *        ldc)
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasZgemm(d_deviceBlasHandle,
                    transA,
                    transB,
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
    }
    dftfe::utils::deviceBlasStatus_t
    BLASWrapperClass<dftfe::utils::MemorySpace::DEVICE>::create()
    {
      dftfe::utils::deviceBlasStatus_t status = cublasCreate(d_deviceBlasHandle);
      DEVICEBLAS_API_CHECK(status);
      return status;
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapperClass<dftfe::utils::MemorySpace::DEVICE>::destroy()
    {
      dftfe::utils::deviceBlasStatus_t status = cublasDestroy(d_deviceBlasHandle);
      DEVICEBLAS_API_CHECK(status);
      return status;
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapperClass<dftfe::utils::MemorySpace::DEVICE>::setStream(
      dftfe::utils::deviceStream_t stream)
    {
      dftfe::utils::deviceBlasStatus_t status = cublasSetStream(d_deviceBlasHandle, stream);
      DEVICEBLAS_API_CHECK(status);
      return status;
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapperClass<dftfe::utils::MemorySpace::DEVICE>::setMathMode(
      deviceBlasMath_t mathMode)
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasSetMathMode(d_deviceBlasHandle, mathMode);
      DEVICEBLAS_API_CHECK(status);
      return status;
    }

    void
    BLASWrapperClass<dftfe::utils::MemorySpace::DEVICE>::xaxpy(
      const unsigned int *n,
      const double *      alpha,
      double *            x,
      const unsigned int *incx,
      double *            y,
      const unsigned int *incy)
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasDaxpy(d_deviceBlasHandle, n, alpha, x, incx, y, incy);
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapperClass<dftfe::utils::MemorySpace::DEVICE>::xaxpy(
      const unsigned int *        n,
      const std::complex<double> *alpha,
      std::complex<double> *      x,
      const unsigned int *        incx,
      std::complex<double> *      y,
      const unsigned int *        incy)
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasZaxpy(d_deviceBlasHandle,
                    n,
                    dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                    dftfe::utils::makeDataTypeDeviceCompatible(x),
                    incx,
                    dftfe::utils::makeDataTypeDeviceCompatible(y),
                    incy);
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapperClass<dftfe::utils::MemorySpace::DEVICE>::xdot(const unsigned int *N,
           const double *      X,
           const unsigned int *INCX,
           const double *      Y,
           const unsigned int *INCY)
    {
              dftfe::utils::deviceBlasStatus_t status =
          cublasDdot(d_deviceBlasHandle, n, x, incx, y, incy, result);
        DEVICEBLAS_API_CHECK(status);
    }


  } // End of namespace linearAlgebra
} // End of namespace dftfe
