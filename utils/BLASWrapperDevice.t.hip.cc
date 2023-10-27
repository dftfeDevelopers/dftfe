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
#include <deviceKernelsGeneric.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherConstants.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include "BLASWrapperDeviceKernels.cc"

namespace dftfe
{
  namespace linearAlgebra
  {
#ifdef DFTFE_WITH_DEVICE_AMD
    void
    initialize()
    {
      rocblas_initialize();
    }
#endif
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
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
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
      dftfe::utils::deviceBlasStatus_t status = hipblasSgemm(d_deviceBlasHandle,
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
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
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
      dftfe::utils::deviceBlasStatus_t status =
        hipblasCgemm(d_deviceBlasHandle,
                     transA,
                     transB,
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
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
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
      dftfe::utils::deviceBlasStatus_t status = hipblasDgemm(d_deviceBlasHandle,
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
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
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
      dftfe::utils::deviceBlasStatus_t status =
        hipblasZgemm(d_deviceBlasHandle,
                     transA,
                     transB,
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
    }
    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::create()
    {
      dftfe::utils::deviceBlasStatus_t status =
        hipblasCreate(d_deviceBlasHandle);
      DEVICEBLAS_API_CHECK(status);
      return status;
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::destroy()
    {
      dftfe::utils::deviceBlasStatus_t status =
        hipblasDestroy(d_deviceBlasHandle);
      DEVICEBLAS_API_CHECK(status);
      return status;
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::setStream(
      deviceStream_t stream)
    {
      dftfe::utils::deviceBlasStatus_t status =
        hipblasSetStream(d_deviceBlasHandle, stream);
      DEVICEBLAS_API_CHECK(status);
      return status;
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::setMathMode(
      deviceBlasMath_t mathMode)
    {
      dftfe::utils::deviceBlasStatus_t status =
        hipblasSetMathMode(d_deviceBlasHandle, mathMode);
      DEVICEBLAS_API_CHECK(status);
      return status;
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xaxpy(
      const unsigned int *n,
      const double *      alpha,
      double *            x,
      const unsigned int *incx,
      double *            y,
      const unsigned int *incy)
    {
      dftfe::utils::deviceBlasStatus_t status =
        hipblasDaxpy(d_deviceBlasHandle, n, alpha, x, incx, y, incy);
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xaxpy(
      const unsigned int *        n,
      const std::complex<double> *alpha,
      std::complex<double> *      x,
      const unsigned int *        incx,
      std::complex<double> *      y,
      const unsigned int *        incy)
    {
      dftfe::utils::deviceBlasStatus_t status =
        hipblasZaxpy(d_deviceBlasHandle,
                     n,
                     makeDataTypeHipBlasCompatible(alpha),
                     makeDataTypeHipBlasCompatible(x),
                     incx,
                     makeDataTypeHipBlasCompatible(y),
                     incy);
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const unsigned int *N,
      const double *      X,
      const unsigned int *INCX,
      const double *      Y,
      const unsigned int *INCY)
    {
      dftfe::utils::deviceBlasStatus_t status =
        hipblasDdot(d_deviceBlasHandle, n, x, incx, y, incy, result);
      DEVICEBLAS_API_CHECK(status);
    }



  } // End of namespace linearAlgebra
} // End of namespace dftfe
