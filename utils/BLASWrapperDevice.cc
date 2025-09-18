// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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
#include <DeviceKernelLauncherHelpers.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfigHalfPrec.h>
#ifdef DFTFE_WITH_DEVICE_INTEL
#  include <oneapi/mkl.hpp>
#  include <oneapi/mkl/blas.hpp>
#endif
#ifdef DFTFE_WITH_DEVICE_AMD
#  define HIPBLAS_V2
#  include <rocblas.h>
#  include <hipblas.h>
#  include <hipblas/hipblas-version.h>
#endif
#ifdef DFTFE_WITH_DEVICE_NVIDIA
#  include <cublas_v2.h>
#endif

#include "BLASWrapperDeviceKernels.cc"

#ifdef DFTFE_WITH_DEVICE_NVIDIA
#  ifdef DFTFE_WITH_64BIT_INT
#    define DFTFE_DEVICE_BLAS_INT(type, name) cublas##type##name##_64
#  else
#    define DFTFE_DEVICE_BLAS_INT(type, name) cublas##type##name
#  endif
#  define DFTFE_DEVICE_BLAS(type, name) cublas##type##name
#elif defined(DFTFE_WITH_DEVICE_AMD)
#  ifdef DFTFE_WITH_64BIT_INT
#    define DFTFE_DEVICE_BLAS_INT(type, name) hipblas##type##name##_64
#  else
#    define DFTFE_DEVICE_BLAS_INT(type, name) hipblas##type##name
#  endif
#  define DFTFE_DEVICE_BLAS(type, name) hipblas##type##name
#elif defined(DFTFE_WITH_DEVICE_INTEL)
#  define DFTFE_DEVICE_BLAS_INT(type, name) \
    oneapi::mkl::blas::column_major::name
#else
#  error \
    "No device backend defined (DFTFE_WITH_DEVICE_NVIDIA or DFTFE_WITH_DEVICE_AMD)"
#endif

namespace dftfe
{
  namespace utils
  {
#if hipblasVersionMajor >= 2 || defined(DFTFE_WITH_DEVICE_NVIDIA) || \
  defined(DFTFE_WITH_DEVICE_INTEL)
    template <typename T>
    inline auto
    makeDataTypeDeviceBlasCompatible(T &&x)
      -> decltype(makeDataTypeDeviceCompatible(std::forward<T>(x)))
    {
      return makeDataTypeDeviceCompatible(std::forward<T>(x));
    }

#else
    inline double
    makeDataTypeDeviceBlasCompatible(double a)
    {
      return a;
    }

    inline float
    makeDataTypeDeviceBlasCompatible(float a)
    {
      return a;
    }

    inline float *
    makeDataTypeDeviceBlasCompatible(float *a)
    {
      return reinterpret_cast<float *>(a);
    }

    inline const float *
    makeDataTypeDeviceBlasCompatible(const float *a)
    {
      return reinterpret_cast<const float *>(a);
    }

    inline double *
    makeDataTypeDeviceBlasCompatible(double *a)
    {
      return reinterpret_cast<double *>(a);
    }

    inline const double *
    makeDataTypeDeviceBlasCompatible(const double *a)
    {
      return reinterpret_cast<const double *>(a);
    }
    inline hipblasDoubleComplex
    makeDataTypeDeviceBlasCompatible(std::complex<double> a)
    {
      return hipblasDoubleComplex(a.real(), a.imag());
    }

    inline hipblasComplex
    makeDataTypeDeviceBlasCompatible(std::complex<float> a)
    {
      return hipblasComplex(a.real(), a.imag());
    }

    inline hipblasComplex *
    makeDataTypeDeviceBlasCompatible(std::complex<float> *a)
    {
      return reinterpret_cast<hipblasComplex *>(a);
    }

    inline const hipblasComplex *
    makeDataTypeDeviceBlasCompatible(const std::complex<float> *a)
    {
      return reinterpret_cast<const hipblasComplex *>(a);
    }

    inline hipblasDoubleComplex *
    makeDataTypeDeviceBlasCompatible(std::complex<double> *a)
    {
      return reinterpret_cast<hipblasDoubleComplex *>(a);
    }

    inline const hipblasDoubleComplex *
    makeDataTypeDeviceBlasCompatible(const std::complex<double> *a)
    {
      return reinterpret_cast<const hipblasDoubleComplex *>(a);
    }
#endif
  } // namespace utils

  namespace linearAlgebra
  {
#ifdef DFTFE_WITH_DEVICE_AMD
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::initialize()
    {
      rocblas_initialize();
    }
#endif
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::BLASWrapper()
    {
#ifdef DFTFE_WITH_DEVICE_AMD
      initialize();
#endif

      dftfe::utils::deviceBlasStatus_t status;
      status     = create();
      d_opType   = tensorOpDataType::fp32;
      d_streamId = dftfe::utils::defaultStream;
      status     = setStream(d_streamId);
    }

    dftfe::utils::deviceBlasHandle_t &
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::getDeviceBlasHandle()
    {
      return d_deviceBlasHandle;
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2ArrDeviceCall(
        const dftfe::uInt            size,
        const ValueType1            *valueType1Arr,
        ValueType2                  *valueType2Arr,
        dftfe::utils::deviceStream_t streamId)
    {
      DFTFE_LAUNCH_KERNEL(
        copyValueType1ArrToValueType2ArrDeviceKernel,
        size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        streamId,
        size,
        dftfe::utils::makeDataTypeDeviceCompatible(valueType1Arr),
        dftfe::utils::makeDataTypeDeviceCompatible(valueType2Arr));
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xcopy(
      const dftfe::uInt           n,
      const std::complex<double> *x,
      const dftfe::uInt           incx,
      std::complex<double>       *y,
      const dftfe::uInt           incy)
    {
      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(Z, copy)(
        d_deviceBlasHandle,
        n,
        dftfe::utils::makeDataTypeDeviceBlasCompatible(x),
        incx,
        dftfe::utils::makeDataTypeDeviceBlasCompatible(y),
        incy));
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xcopy(
      const dftfe::uInt          n,
      const std::complex<float> *x,
      const dftfe::uInt          incx,
      std::complex<float>       *y,
      const dftfe::uInt          incy)
    {
      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(C, copy)(
        d_deviceBlasHandle,
        n,
        dftfe::utils::makeDataTypeDeviceBlasCompatible(x),
        incx,
        dftfe::utils::makeDataTypeDeviceBlasCompatible(y),
        incy));
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xcopy(
      const dftfe::uInt n,
      const double     *x,
      const dftfe::uInt incx,
      double           *y,
      const dftfe::uInt incy)
    {
      DEVICEBLAS_API_CHECK(
        DFTFE_DEVICE_BLAS_INT(D,
                              copy)(d_deviceBlasHandle, n, x, incx, y, incy));
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xcopy(
      const dftfe::uInt n,
      const float      *x,
      const dftfe::uInt incx,
      float            *y,
      const dftfe::uInt incy)
    {
      DEVICEBLAS_API_CHECK(
        DFTFE_DEVICE_BLAS_INT(S,
                              copy)(d_deviceBlasHandle, n, x, incx, y, incy));
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(const char transA,
                                                          const char transB,
                                                          const dftfe::uInt m,
                                                          const dftfe::uInt n,
                                                          const dftfe::uInt k,
                                                          const float *alpha,
                                                          const float *A,
                                                          const dftfe::uInt lda,
                                                          const float      *B,
                                                          const dftfe::uInt ldb,
                                                          const float *beta,
                                                          float       *C,
                                                          const dftfe::uInt ldc)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T' || transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else
        {
          throw std::invalid_argument("Incorrect transA in gemm ");
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T' || transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else
        {
          throw std::invalid_argument("Incorrect transB in gemm ");
        }
      dftfe::utils::deviceBlasComputeType_t computeType =
        dftfe::utils::DEVICEBLAS_COMPUTE_32F;
      if (d_opType == tensorOpDataType::tf32)
        computeType = dftfe::utils::DEVICEBLAS_COMPUTE_32F_FAST_TF32;
      else if (d_opType == tensorOpDataType::bf16)
        computeType = dftfe::utils::DEVICEBLAS_COMPUTE_32F_FAST_16BF;
      else if (d_opType == tensorOpDataType::fp16)
        computeType = dftfe::utils::DEVICEBLAS_COMPUTE_32F_FAST_16F;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status =
        DFTFE_DEVICE_BLAS_INT(, GemmEx)(d_deviceBlasHandle,
                                        transa,
                                        transb,
                                        dftfe::Int(m),
                                        dftfe::Int(n),
                                        dftfe::Int(k),
                                        (const void *)alpha,
                                        (const void *)A,
                                        dftfe::utils::DEVICE_R_32F,
                                        dftfe::Int(lda),
                                        (const void *)B,
                                        dftfe::utils::DEVICE_R_32F,
                                        dftfe::Int(ldb),
                                        (const void *)beta,
                                        (void *)C,
                                        dftfe::utils::DEVICE_R_32F,
                                        dftfe::Int(ldc),
                                        computeType,
                                        dftfe::utils::DEVICEBLAS_GEMM_DEFAULT);
      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(, gemm)(d_deviceBlasHandle,
                                                         transa,
                                                         transb,
                                                         dftfe::Int(m),
                                                         dftfe::Int(n),
                                                         dftfe::Int(k),
                                                         alpha,
                                                         A,
                                                         dftfe::Int(lda),
                                                         B,
                                                         dftfe::Int(ldb),
                                                         beta,
                                                         C,
                                                         dftfe::Int(ldc),
                                                         computeType));
#endif
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
      const char                 transA,
      const char                 transB,
      const dftfe::uInt          m,
      const dftfe::uInt          n,
      const dftfe::uInt          k,
      const std::complex<float> *alpha,
      const std::complex<float> *A,
      const dftfe::uInt          lda,
      const std::complex<float> *B,
      const dftfe::uInt          ldb,
      const std::complex<float> *beta,
      std::complex<float>       *C,
      const dftfe::uInt          ldc)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          throw std::invalid_argument("Incorrect transA in gemm ");
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          throw std::invalid_argument("Incorrect transB in gemm ");
        }
      dftfe::utils::deviceBlasComputeType_t computeType =
        dftfe::utils::DEVICEBLAS_COMPUTE_32F;
      if (d_opType == tensorOpDataType::tf32)
        computeType = dftfe::utils::DEVICEBLAS_COMPUTE_32F_FAST_TF32;
      else if (d_opType == tensorOpDataType::bf16)
        computeType = dftfe::utils::DEVICEBLAS_COMPUTE_32F_FAST_16BF;
      else if (d_opType == tensorOpDataType::fp16)
        computeType = dftfe::utils::DEVICEBLAS_COMPUTE_32F_FAST_16F;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status =
        DFTFE_DEVICE_BLAS_INT(, GemmEx)(d_deviceBlasHandle,
                                        transa,
                                        transb,
                                        dftfe::Int(m),
                                        dftfe::Int(n),
                                        dftfe::Int(k),
                                        (const void *)alpha,
                                        (const void *)A,
                                        dftfe::utils::DEVICE_C_32F,
                                        dftfe::Int(lda),
                                        (const void *)B,
                                        dftfe::utils::DEVICE_C_32F,
                                        dftfe::Int(ldb),
                                        (const void *)beta,
                                        (void *)C,
                                        dftfe::utils::DEVICE_C_32F,
                                        dftfe::Int(ldc),
                                        computeType,
                                        dftfe::utils::DEVICEBLAS_GEMM_DEFAULT);

      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(, gemm)(d_deviceBlasHandle,
                                                         transa,
                                                         transb,
                                                         dftfe::Int(m),
                                                         dftfe::Int(n),
                                                         dftfe::Int(k),
                                                         alpha,
                                                         A,
                                                         dftfe::Int(lda),
                                                         B,
                                                         dftfe::Int(ldb),
                                                         beta,
                                                         C,
                                                         dftfe::Int(ldc),
                                                         computeType));
#endif
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(const char transA,
                                                          const char transB,
                                                          const dftfe::uInt m,
                                                          const dftfe::uInt n,
                                                          const dftfe::uInt k,
                                                          const double *alpha,
                                                          const double *A,
                                                          const dftfe::uInt lda,
                                                          const double     *B,
                                                          const dftfe::uInt ldb,
                                                          const double *beta,
                                                          double       *C,
                                                          const dftfe::uInt ldc)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T' || transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          throw std::invalid_argument("Incorrect transA in gemm ");
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T' || transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          throw std::invalid_argument("Incorrect transB in gemm ");
        }
      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(D, gemm)(d_deviceBlasHandle,
                                                          transa,
                                                          transb,
                                                          dftfe::Int(m),
                                                          dftfe::Int(n),
                                                          dftfe::Int(k),
                                                          alpha,
                                                          A,
                                                          dftfe::Int(lda),
                                                          B,
                                                          dftfe::Int(ldb),
                                                          beta,
                                                          C,
                                                          dftfe::Int(ldc)));
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
      const char                  transA,
      const char                  transB,
      const dftfe::uInt           m,
      const dftfe::uInt           n,
      const dftfe::uInt           k,
      const std::complex<double> *alpha,
      const std::complex<double> *A,
      const dftfe::uInt           lda,
      const std::complex<double> *B,
      const dftfe::uInt           ldb,
      const std::complex<double> *beta,
      std::complex<double>       *C,
      const dftfe::uInt           ldc)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          throw std::invalid_argument("Incorrect transA in gemm ");
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          throw std::invalid_argument("Incorrect transB in gemm ");
        }
      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(Z, gemm)(
        d_deviceBlasHandle,
        transa,
        transb,
        dftfe::Int(m),
        dftfe::Int(n),
        dftfe::Int(k),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(alpha),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(A),
        dftfe::Int(lda),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(B),
        dftfe::Int(ldb),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(beta),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(C),
        dftfe::Int(ldc)));
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemv(
      const char        transA,
      const dftfe::uInt m,
      const dftfe::uInt n,
      const double     *alpha,
      const double     *A,
      const dftfe::uInt lda,
      const double     *x,
      const dftfe::uInt incx,
      const double     *beta,
      double           *y,
      const dftfe::uInt incy)
    {
      dftfe::utils::deviceBlasOperation_t transa;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T' || transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else
        {
          throw std::invalid_argument("Incorrect transA in gemv ");
        }
      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(D, gemv)(d_deviceBlasHandle,
                                                          transa,
                                                          dftfe::Int(m),
                                                          dftfe::Int(n),
                                                          alpha,
                                                          A,
                                                          dftfe::Int(lda),
                                                          x,
                                                          dftfe::Int(incx),
                                                          beta,
                                                          y,
                                                          dftfe::Int(incy)));
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemv(
      const char        transA,
      const dftfe::uInt m,
      const dftfe::uInt n,
      const float      *alpha,
      const float      *A,
      const dftfe::uInt lda,
      const float      *x,
      const dftfe::uInt incx,
      const float      *beta,
      float            *y,
      const dftfe::uInt incy)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T' || transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else
        {
          throw std::invalid_argument("Incorrect transA in gemv ");
        }

      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(S, gemv)(d_deviceBlasHandle,
                                                          transa,
                                                          dftfe::Int(m),
                                                          dftfe::Int(n),
                                                          alpha,
                                                          A,
                                                          dftfe::Int(lda),
                                                          x,
                                                          dftfe::Int(incx),
                                                          beta,
                                                          y,
                                                          dftfe::Int(incy)));
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemv(
      const char                  transA,
      const dftfe::uInt           m,
      const dftfe::uInt           n,
      const std::complex<double> *alpha,
      const std::complex<double> *A,
      const dftfe::uInt           lda,
      const std::complex<double> *x,
      const dftfe::uInt           incx,
      const std::complex<double> *beta,
      std::complex<double>       *y,
      const dftfe::uInt           incy)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          throw std::invalid_argument("Incorrect transA in gemv ");
        }

      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(Z, gemv)(
        d_deviceBlasHandle,
        transa,
        dftfe::Int(m),
        dftfe::Int(n),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(alpha),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(A),
        dftfe::Int(lda),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(x),
        dftfe::Int(incx),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(beta),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(y),
        dftfe::Int(incy)));
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemv(
      const char                 transA,
      const dftfe::uInt          m,
      const dftfe::uInt          n,
      const std::complex<float> *alpha,
      const std::complex<float> *A,
      const dftfe::uInt          lda,
      const std::complex<float> *x,
      const dftfe::uInt          incx,
      const std::complex<float> *beta,
      std::complex<float>       *y,
      const dftfe::uInt          incy)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          throw std::invalid_argument("Incorrect transA in gemv ");
        }

      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(C, gemv)(
        d_deviceBlasHandle,
        transa,
        dftfe::Int(m),
        dftfe::Int(n),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(alpha),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(A),
        dftfe::Int(lda),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(x),
        dftfe::Int(incx),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(beta),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(y),
        dftfe::Int(incy)));
    }


    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::create()
    {
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status =
        DFTFE_DEVICE_BLAS(, Create)(&d_deviceBlasHandle);
      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      d_streamId = dftfe::utils::defaultStream;
      d_deviceBlasHandle =
        dftfe::utils::queueRegistry.find(dftfe::utils::defaultStream)->second;
      dftfe::utils::deviceBlasStatus_t status = dftfe::utils::deviceBlasSuccess;
#endif
      return status;
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::destroy()
    {
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status =
        DFTFE_DEVICE_BLAS(, Destroy)(d_deviceBlasHandle);
      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      dftfe::utils::deviceBlasStatus_t status = dftfe::utils::deviceBlasSuccess;
#endif
      return status;
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::setStream(
      dftfe::utils::deviceStream_t streamId)
    {
      d_streamId = streamId;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status =
        DFTFE_DEVICE_BLAS(, SetStream)(d_deviceBlasHandle, d_streamId);
      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      d_deviceBlasHandle = dftfe::utils::queueRegistry.find(streamId)->second;
      dftfe::utils::deviceBlasStatus_t status = dftfe::utils::deviceBlasSuccess;
#endif
      return status;
    }



    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xaxpy(
      const dftfe::uInt n,
      const double     *alpha,
      const double     *x,
      const dftfe::uInt incx,
      double           *y,
      const dftfe::uInt incy)
    {
      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(D, axpy)(d_deviceBlasHandle,
                                                          dftfe::Int(n),
                                                          alpha,
                                                          x,
                                                          dftfe::Int(incx),
                                                          y,
                                                          dftfe::Int(incy)));
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xaxpy(
      const dftfe::uInt           n,
      const std::complex<double> *alpha,
      const std::complex<double> *x,
      const dftfe::uInt           incx,
      std::complex<double>       *y,
      const dftfe::uInt           incy)
    {
      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(Z, axpy)(
        d_deviceBlasHandle,
        dftfe::Int(n),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(alpha),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(x),
        dftfe::Int(incx),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(y),
        dftfe::Int(incy)));
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpby(
      const dftfe::uInt n,
      const ValueType2  alpha,
      const ValueType1 *x,
      const ValueType2  beta,
      ValueType1       *y)
    {
      DFTFE_LAUNCH_KERNEL(axpbyDeviceKernel,
                          n / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          d_streamId,
                          n,
                          dftfe::utils::makeDataTypeDeviceCompatible(x),
                          dftfe::utils::makeDataTypeDeviceCompatible(y),
                          alpha,
                          beta);
    }

    template <typename ValueType0,
              typename ValueType1,
              typename ValueType2,
              typename ValueType3,
              typename ValueType4>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::ApaBD(
      const dftfe::uInt m,
      const dftfe::uInt n,
      const ValueType0  alpha,
      const ValueType1 *A,
      const ValueType2 *B,
      const ValueType3 *D,
      ValueType4       *C)
    {
      DFTFE_LAUNCH_KERNEL(ApaBDDeviceKernel,
                          (n * m / dftfe::utils::DEVICE_BLOCK_SIZE) + 1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          d_streamId,
                          m,
                          n,
                          alpha,
                          dftfe::utils::makeDataTypeDeviceCompatible(A),
                          dftfe::utils::makeDataTypeDeviceCompatible(B),
                          dftfe::utils::makeDataTypeDeviceCompatible(D),
                          dftfe::utils::makeDataTypeDeviceCompatible(C));
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockAxpy(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const ValueType1 *addFromVec,
      const ValueType2 *scalingVector,
      const ValueType2  a,
      ValueType1       *addToVec)
    {
      DFTFE_LAUNCH_KERNEL(
        stridedBlockAxpyDeviceKernel,
        (contiguousBlockSize * numContiguousBlocks) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        d_streamId,
        contiguousBlockSize,
        numContiguousBlocks,
        dftfe::utils::makeDataTypeDeviceCompatible(a),
        dftfe::utils::makeDataTypeDeviceCompatible(scalingVector),
        dftfe::utils::makeDataTypeDeviceCompatible(addFromVec),
        dftfe::utils::makeDataTypeDeviceCompatible(addToVec));
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockAxpBy(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const ValueType1 *addFromVec,
      const ValueType2 *scalingVector,
      const ValueType2  a,
      const ValueType2  b,
      ValueType1       *addToVec)
    {
      DFTFE_LAUNCH_KERNEL(
        stridedBlockAxpByDeviceKernel,
        (contiguousBlockSize * numContiguousBlocks) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        d_streamId,
        contiguousBlockSize,
        numContiguousBlocks,
        dftfe::utils::makeDataTypeDeviceCompatible(a),
        dftfe::utils::makeDataTypeDeviceCompatible(b),
        dftfe::utils::makeDataTypeDeviceCompatible(scalingVector),
        dftfe::utils::makeDataTypeDeviceCompatible(addFromVec),
        dftfe::utils::makeDataTypeDeviceCompatible(addToVec));
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const ValueType   *addFromVec,
      ValueType         *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      DFTFE_LAUNCH_KERNEL(axpyStridedBlockAtomicAddDeviceKernel,
                          (contiguousBlockSize * numContiguousBlocks) /
                              dftfe::utils::DEVICE_BLOCK_SIZE +
                            1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          d_streamId,
                          contiguousBlockSize,
                          numContiguousBlocks,
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            addFromVec),
                          dftfe::utils::makeDataTypeDeviceCompatible(addToVec),
                          addToVecStartingContiguousBlockIds);
    }

    template <typename ValueType1, typename ValueType2, typename ValueType3>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const ValueType1   a,
      const ValueType1  *s,
      const ValueType2  *addFromVec,
      ValueType3        *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      DFTFE_LAUNCH_KERNEL(axpyStridedBlockAtomicAddDeviceKernel,
                          (contiguousBlockSize * numContiguousBlocks) /
                              dftfe::utils::DEVICE_BLOCK_SIZE +
                            1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          d_streamId,
                          contiguousBlockSize,
                          numContiguousBlocks,
                          dftfe::utils::makeDataTypeDeviceCompatible(a),
                          dftfe::utils::makeDataTypeDeviceCompatible(s),
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            addFromVec),
                          dftfe::utils::makeDataTypeDeviceCompatible(addToVec),
                          addToVecStartingContiguousBlockIds);
    }

    template <typename ValueType1, typename ValueType2, typename ValueType3>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const ValueType1   a,
      const ValueType2  *addFromVec,
      ValueType3        *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      DFTFE_LAUNCH_KERNEL(axpyStridedBlockAtomicAddDeviceKernel,
                          (contiguousBlockSize * numContiguousBlocks) /
                              dftfe::utils::DEVICE_BLOCK_SIZE +
                            1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          d_streamId,
                          contiguousBlockSize,
                          numContiguousBlocks,
                          dftfe::utils::makeDataTypeDeviceCompatible(a),
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            addFromVec),
                          dftfe::utils::makeDataTypeDeviceCompatible(addToVec),
                          addToVecStartingContiguousBlockIds);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(const dftfe::uInt N,
                                                         const double     *X,
                                                         const dftfe::uInt INCX,
                                                         const double     *Y,
                                                         const dftfe::uInt INCY,
                                                         double *result)
    {
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status =
        DFTFE_DEVICE_BLAS_INT(D, dot)(d_deviceBlasHandle,
                                      dftfe::Int(N),
                                      X,
                                      dftfe::Int(INCX),
                                      Y,
                                      dftfe::Int(INCY),
                                      result);
      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      double *dev_res = sycl::malloc_device<double>(1, d_deviceBlasHandle);
      if (!dev_res)
        throw std::bad_alloc{};
      dftfe::utils::deviceEvent_t event =
        DFTFE_DEVICE_BLAS_INT(D, dot)(d_deviceBlasHandle,
                                      dftfe::Int(N),
                                      X,
                                      dftfe::Int(INCX),
                                      Y,
                                      dftfe::Int(INCY),
                                      dev_res);
      d_deviceBlasHandle.memcpy(result, dev_res, sizeof(double)).wait();
      sycl::free(dev_res, d_deviceBlasHandle);
      d_deviceBlasHandle.wait();
#endif
    }
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(const dftfe::uInt N,
                                                         const float      *X,
                                                         const dftfe::uInt INCX,
                                                         const float      *Y,
                                                         const dftfe::uInt INCY,
                                                         float *result)
    {
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status =
        DFTFE_DEVICE_BLAS_INT(S, dot)(d_deviceBlasHandle,
                                      dftfe::Int(N),
                                      X,
                                      dftfe::Int(INCX),
                                      Y,
                                      dftfe::Int(INCY),
                                      result);
      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      float *dev_res = sycl::malloc_device<float>(1, d_deviceBlasHandle);
      if (!dev_res)
        throw std::bad_alloc{};
      dftfe::utils::deviceEvent_t event =
        DFTFE_DEVICE_BLAS_INT(S, dot)(d_deviceBlasHandle,
                                      dftfe::Int(N),
                                      X,
                                      dftfe::Int(INCX),
                                      Y,
                                      dftfe::Int(INCY),
                                      dev_res);
      d_deviceBlasHandle.memcpy(result, dev_res, sizeof(float)).wait();
      sycl::free(dev_res, d_deviceBlasHandle);
      d_deviceBlasHandle.wait();
#endif
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const dftfe::uInt N,
      const double     *X,
      const dftfe::uInt INCX,
      const double     *Y,
      const dftfe::uInt INCY,
      const MPI_Comm   &mpi_communicator,
      double           *result)
    {
      double localResult = 0.0;
      *result            = 0.0;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status =
        DFTFE_DEVICE_BLAS_INT(D, dot)(d_deviceBlasHandle,
                                      dftfe::Int(N),
                                      X,
                                      dftfe::Int(INCX),
                                      Y,
                                      dftfe::Int(INCY),
                                      &localResult);
      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      double *dev_res = sycl::malloc_device<double>(1, d_deviceBlasHandle);
      if (!dev_res)
        throw std::bad_alloc{};
      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(D, dot)(d_deviceBlasHandle,
                                                         dftfe::Int(N),
                                                         X,
                                                         dftfe::Int(INCX),
                                                         Y,
                                                         dftfe::Int(INCY),
                                                         dev_res));
      d_deviceBlasHandle.memcpy(&localResult, dev_res, sizeof(double)).wait();
      sycl::free(dev_res, d_deviceBlasHandle);
      d_deviceBlasHandle.wait();
#endif
      MPI_Allreduce(
        &localResult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const dftfe::uInt           N,
      const std::complex<double> *X,
      const dftfe::uInt           INCX,
      const std::complex<double> *Y,
      const dftfe::uInt           INCY,
      std::complex<double>       *result)
    {
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status = DFTFE_DEVICE_BLAS_INT(Z, dotc)(
        d_deviceBlasHandle,
        dftfe::Int(N),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(X),
        dftfe::Int(INCX),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(Y),
        dftfe::Int(INCY),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(result));
      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      std::complex<double> *dev_res =
        sycl::malloc_device<std::complex<double>>(1, d_deviceBlasHandle);
      if (!dev_res)
        throw std::bad_alloc{};
      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(Z, dotc)(
        d_deviceBlasHandle,
        dftfe::Int(N),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(X),
        dftfe::Int(INCX),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(Y),
        dftfe::Int(INCY),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(dev_res)));
      d_deviceBlasHandle.memcpy(result, dev_res, sizeof(std::complex<double>))
        .wait();
      sycl::free(dev_res, d_deviceBlasHandle);
      d_deviceBlasHandle.wait();
#endif
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const dftfe::uInt          N,
      const std::complex<float> *X,
      const dftfe::uInt          INCX,
      const std::complex<float> *Y,
      const dftfe::uInt          INCY,
      std::complex<float>       *result)
    {
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status = DFTFE_DEVICE_BLAS_INT(C, dotc)(
        d_deviceBlasHandle,
        dftfe::Int(N),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(X),
        dftfe::Int(INCX),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(Y),
        dftfe::Int(INCY),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(result));
      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      std::complex<float> *dev_res =
        sycl::malloc_device<std::complex<float>>(1, d_deviceBlasHandle);
      if (!dev_res)
        throw std::bad_alloc{};
      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(C, dotu)(
        d_deviceBlasHandle,
        dftfe::Int(N),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(X),
        dftfe::Int(INCX),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(Y),
        dftfe::Int(INCY),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(dev_res)));
      d_deviceBlasHandle.memcpy(result, dev_res, sizeof(std::complex<float>))
        .wait();
      sycl::free(dev_res, d_deviceBlasHandle);
      d_deviceBlasHandle.wait();
#endif
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const dftfe::uInt           N,
      const std::complex<double> *X,
      const dftfe::uInt           INCX,
      const std::complex<double> *Y,
      const dftfe::uInt           INCY,
      const MPI_Comm             &mpi_communicator,
      std::complex<double>       *result)
    {
      std::complex<double> localResult = 0.0;
      *result                          = 0.0;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status = DFTFE_DEVICE_BLAS_INT(Z, dotc)(
        d_deviceBlasHandle,
        dftfe::Int(N),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(X),
        dftfe::Int(INCX),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(Y),
        dftfe::Int(INCY),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(&localResult));
      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      std::complex<double> *dev_res =
        sycl::malloc_device<std::complex<double>>(1, d_deviceBlasHandle);
      if (!dev_res)
        throw std::bad_alloc{};
      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(Z, dotc)(
        d_deviceBlasHandle,
        dftfe::Int(N),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(X),
        dftfe::Int(INCX),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(Y),
        dftfe::Int(INCY),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(dev_res)));
      d_deviceBlasHandle
        .memcpy(&localResult, dev_res, sizeof(std::complex<double>))
        .wait();
      sycl::free(dev_res, d_deviceBlasHandle);
      d_deviceBlasHandle.wait();
#endif
      MPI_Allreduce(&localResult,
                    result,
                    1,
                    dataTypes::mpi_type_id(result),
                    MPI_SUM,
                    mpi_communicator);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char        transA,
      const char        transB,
      const dftfe::uInt m,
      const dftfe::uInt n,
      const dftfe::uInt k,
      const double     *alpha,
      const double     *A,
      const dftfe::uInt lda,
      long long int     strideA,
      const double     *B,
      const dftfe::uInt ldb,
      long long int     strideB,
      const double     *beta,
      double           *C,
      const dftfe::uInt ldc,
      long long int     strideC,
      const dftfe::Int  batchCount)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status =
        DFTFE_DEVICE_BLAS_INT(D, gemmStridedBatched)(d_deviceBlasHandle,
                                                     transa,
                                                     transb,
                                                     dftfe::Int(m),
                                                     dftfe::Int(n),
                                                     dftfe::Int(k),
                                                     alpha,
                                                     A,
                                                     dftfe::Int(lda),
                                                     strideA,
                                                     B,
                                                     dftfe::Int(ldb),
                                                     strideB,
                                                     beta,
                                                     C,
                                                     dftfe::Int(ldc),
                                                     strideC,
                                                     dftfe::Int(batchCount));
      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      DEVICEBLAS_API_CHECK(
        DFTFE_DEVICE_BLAS_INT(D, gemm_batch)(d_deviceBlasHandle,
                                             transa,
                                             transb,
                                             dftfe::Int(m),
                                             dftfe::Int(n),
                                             dftfe::Int(k),
                                             alpha,
                                             A,
                                             dftfe::Int(lda),
                                             strideA,
                                             B,
                                             dftfe::Int(ldb),
                                             strideB,
                                             beta,
                                             C,
                                             dftfe::Int(ldc),
                                             strideC,
                                             dftfe::Int(batchCount)));
#endif
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char                  transA,
      const char                  transB,
      const dftfe::uInt           m,
      const dftfe::uInt           n,
      const dftfe::uInt           k,
      const std::complex<double> *alpha,
      const std::complex<double> *A,
      const dftfe::uInt           lda,
      long long int               strideA,
      const std::complex<double> *B,
      const dftfe::uInt           ldb,
      long long int               strideB,
      const std::complex<double> *beta,
      std::complex<double>       *C,
      const dftfe::uInt           ldc,
      long long int               strideC,
      const dftfe::Int            batchCount)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status =
        DFTFE_DEVICE_BLAS_INT(Z, gemmStridedBatched)(
          d_deviceBlasHandle,
          transa,
          transb,
          dftfe::Int(m),
          dftfe::Int(n),
          dftfe::Int(k),
          dftfe::utils::makeDataTypeDeviceBlasCompatible(alpha),
          dftfe::utils::makeDataTypeDeviceBlasCompatible(A),
          dftfe::Int(lda),
          strideA,
          dftfe::utils::makeDataTypeDeviceBlasCompatible(B),
          dftfe::Int(ldb),
          strideB,
          dftfe::utils::makeDataTypeDeviceBlasCompatible(beta),
          dftfe::utils::makeDataTypeDeviceBlasCompatible(C),
          dftfe::Int(ldc),
          strideC,
          dftfe::Int(batchCount));
      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(Z, gemm_batch)(
        d_deviceBlasHandle,
        transa,
        transb,
        dftfe::Int(m),
        dftfe::Int(n),
        dftfe::Int(k),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(alpha),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(A),
        dftfe::Int(lda),
        strideA,
        dftfe::utils::makeDataTypeDeviceBlasCompatible(B),
        dftfe::Int(ldb),
        strideB,
        dftfe::utils::makeDataTypeDeviceBlasCompatible(beta),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(C),
        dftfe::Int(ldc),
        strideC,
        dftfe::Int(batchCount)));
#endif
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char        transA,
      const char        transB,
      const dftfe::uInt m,
      const dftfe::uInt n,
      const dftfe::uInt k,
      const float      *alpha,
      const float      *A,
      const dftfe::uInt lda,
      long long int     strideA,
      const float      *B,
      const dftfe::uInt ldb,
      long long int     strideB,
      const float      *beta,
      float            *C,
      const dftfe::uInt ldc,
      long long int     strideC,
      const dftfe::Int  batchCount)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }
      dftfe::utils::deviceBlasComputeType_t computeType =
        dftfe::utils::DEVICEBLAS_COMPUTE_32F;
      if (d_opType == tensorOpDataType::tf32)
        computeType = dftfe::utils::DEVICEBLAS_COMPUTE_32F_FAST_TF32;
      else if (d_opType == tensorOpDataType::bf16)
        computeType = dftfe::utils::DEVICEBLAS_COMPUTE_32F_FAST_16BF;
      else if (d_opType == tensorOpDataType::fp16)
        computeType = dftfe::utils::DEVICEBLAS_COMPUTE_32F_FAST_16F;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status = DFTFE_DEVICE_BLAS_INT(
        , GemmStridedBatchedEx)(d_deviceBlasHandle,
                                transa,
                                transb,
                                dftfe::Int(m),
                                dftfe::Int(n),
                                dftfe::Int(k),
                                (const void *)alpha,
                                (const void *)A,
                                dftfe::utils::DEVICE_R_32F,
                                dftfe::Int(lda),
                                strideA,
                                (const void *)B,
                                dftfe::utils::DEVICE_R_32F,
                                dftfe::Int(ldb),
                                strideB,
                                (const void *)beta,
                                (void *)C,
                                dftfe::utils::DEVICE_R_32F,
                                dftfe::Int(ldc),
                                strideC,
                                dftfe::Int(batchCount),
                                computeType,
                                dftfe::utils::DEVICEBLAS_GEMM_DEFAULT);
      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      DEVICEBLAS_API_CHECK(
        DFTFE_DEVICE_BLAS_INT(S, gemm_batch)(d_deviceBlasHandle,
                                             transa,
                                             transb,
                                             dftfe::Int(m),
                                             dftfe::Int(n),
                                             dftfe::Int(k),
                                             alpha,
                                             A,
                                             dftfe::Int(lda),
                                             strideA,
                                             B,
                                             dftfe::Int(ldb),
                                             strideB,
                                             beta,
                                             C,
                                             dftfe::Int(ldc),
                                             strideC,
                                             dftfe::Int(batchCount),
                                             computeType));
#endif
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char                 transA,
      const char                 transB,
      const dftfe::uInt          m,
      const dftfe::uInt          n,
      const dftfe::uInt          k,
      const std::complex<float> *alpha,
      const std::complex<float> *A,
      const dftfe::uInt          lda,
      long long int              strideA,
      const std::complex<float> *B,
      const dftfe::uInt          ldb,
      long long int              strideB,
      const std::complex<float> *beta,
      std::complex<float>       *C,
      const dftfe::uInt          ldc,
      long long int              strideC,
      const dftfe::Int           batchCount)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasComputeType_t computeType =
        dftfe::utils::DEVICEBLAS_COMPUTE_32F;
      if (d_opType == tensorOpDataType::tf32)
        computeType = dftfe::utils::DEVICEBLAS_COMPUTE_32F_FAST_TF32;
      else if (d_opType == tensorOpDataType::bf16)
        computeType = dftfe::utils::DEVICEBLAS_COMPUTE_32F_FAST_16BF;
      else if (d_opType == tensorOpDataType::fp16)
        computeType = dftfe::utils::DEVICEBLAS_COMPUTE_32F_FAST_16F;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status = DFTFE_DEVICE_BLAS_INT(
        , GemmStridedBatchedEx)(d_deviceBlasHandle,
                                transa,
                                transb,
                                dftfe::Int(m),
                                dftfe::Int(n),
                                dftfe::Int(k),
                                (const void *)alpha,
                                (const void *)A,
                                dftfe::utils::DEVICE_C_32F,
                                dftfe::Int(lda),
                                strideA,
                                (const void *)B,
                                dftfe::utils::DEVICE_C_32F,
                                dftfe::Int(ldb),
                                strideB,
                                (const void *)beta,
                                (void *)C,
                                dftfe::utils::DEVICE_C_32F,
                                dftfe::Int(ldc),
                                strideC,
                                dftfe::Int(batchCount),
                                computeType,
                                dftfe::utils::DEVICEBLAS_GEMM_DEFAULT);
      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      DEVICEBLAS_API_CHECK(
        DFTFE_DEVICE_BLAS_INT(C, gemm_batch)(d_deviceBlasHandle,
                                             transa,
                                             transb,
                                             dftfe::Int(m),
                                             dftfe::Int(n),
                                             dftfe::Int(k),
                                             alpha,
                                             A,
                                             dftfe::Int(lda),
                                             strideA,
                                             B,
                                             dftfe::Int(ldb),
                                             strideB,
                                             beta,
                                             C,
                                             dftfe::Int(ldc),
                                             strideC,
                                             dftfe::Int(batchCount),
                                             computeType));
#endif
    }
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char        transA,
      const char        transB,
      const dftfe::uInt m,
      const dftfe::uInt n,
      const dftfe::uInt k,
      const double     *alpha,
      const double     *A[],
      const dftfe::uInt lda,
      const double     *B[],
      const dftfe::uInt ldb,
      const double     *beta,
      double           *C[],
      const dftfe::uInt ldc,
      const dftfe::Int  batchCount)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status =
        DFTFE_DEVICE_BLAS_INT(D, gemmBatched)(d_deviceBlasHandle,
                                              transa,
                                              transb,
                                              dftfe::Int(m),
                                              dftfe::Int(n),
                                              dftfe::Int(k),
                                              alpha,
                                              A,
                                              dftfe::Int(lda),
                                              B,
                                              dftfe::Int(ldb),
                                              beta,
                                              C,
                                              dftfe::Int(ldc),
                                              dftfe::Int(batchCount));

      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      const std::int64_t m_local          = std::int64_t(m);
      const std::int64_t n_local          = std::int64_t(n);
      const std::int64_t k_local          = std::int64_t(k);
      const std::int64_t lda_local        = std::int64_t(lda);
      const std::int64_t ldb_local        = std::int64_t(ldb);
      const std::int64_t ldc_local        = std::int64_t(ldc);
      const std::int64_t batchCount_local = std::int64_t(batchCount);
      DEVICEBLAS_API_CHECK(
        DFTFE_DEVICE_BLAS_INT(D, gemm_batch)(d_deviceBlasHandle,
                                             &transa,
                                             &transb,
                                             &m_local,
                                             &n_local,
                                             &k_local,
                                             alpha,
                                             A,
                                             &lda_local,
                                             B,
                                             &ldb_local,
                                             beta,
                                             C,
                                             &ldc_local,
                                             1,
                                             &batchCount_local));
#endif
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char                  transA,
      const char                  transB,
      const dftfe::uInt           m,
      const dftfe::uInt           n,
      const dftfe::uInt           k,
      const std::complex<double> *alpha,
      const std::complex<double> *A[],
      const dftfe::uInt           lda,
      const std::complex<double> *B[],
      const dftfe::uInt           ldb,
      const std::complex<double> *beta,
      std::complex<double>       *C[],
      const dftfe::uInt           ldc,
      const dftfe::Int            batchCount)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status = DFTFE_DEVICE_BLAS_INT(
        Z, gemmBatched)(d_deviceBlasHandle,
                        transa,
                        transb,
                        dftfe::Int(m),
                        dftfe::Int(n),
                        dftfe::Int(k),
                        dftfe::utils::makeDataTypeDeviceBlasCompatible(alpha),
                        (const dftfe::utils::deviceDoubleComplex **)A,
                        dftfe::Int(lda),
                        (const dftfe::utils::deviceDoubleComplex **)B,
                        dftfe::Int(ldb),
                        dftfe::utils::makeDataTypeDeviceBlasCompatible(beta),
                        (dftfe::utils::deviceDoubleComplex **)C,
                        dftfe::Int(ldc),
                        dftfe::Int(batchCount));

      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      const std::int64_t m_local          = std::int64_t(m);
      const std::int64_t n_local          = std::int64_t(n);
      const std::int64_t k_local          = std::int64_t(k);
      const std::int64_t lda_local        = std::int64_t(lda);
      const std::int64_t ldb_local        = std::int64_t(ldb);
      const std::int64_t ldc_local        = std::int64_t(ldc);
      const std::int64_t batchCount_local = std::int64_t(batchCount);
      DEVICEBLAS_API_CHECK(
        DFTFE_DEVICE_BLAS_INT(Z, gemm_batch)(d_deviceBlasHandle,
                                             &transa,
                                             &transb,
                                             &m_local,
                                             &n_local,
                                             &k_local,
                                             alpha,
                                             A,
                                             &lda_local,
                                             B,
                                             &ldb_local,
                                             beta,
                                             C,
                                             &ldc_local,
                                             1,
                                             &batchCount_local));
#endif
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char        transA,
      const char        transB,
      const dftfe::uInt m,
      const dftfe::uInt n,
      const dftfe::uInt k,
      const float      *alpha,
      const float      *A[],
      const dftfe::uInt lda,
      const float      *B[],
      const dftfe::uInt ldb,
      const float      *beta,
      float            *C[],
      const dftfe::uInt ldc,
      const dftfe::Int  batchCount)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasComputeType_t computeType =
        dftfe::utils::DEVICEBLAS_COMPUTE_32F;
      if (d_opType == tensorOpDataType::tf32)
        computeType = dftfe::utils::DEVICEBLAS_COMPUTE_32F_FAST_TF32;
      else if (d_opType == tensorOpDataType::bf16)
        computeType = dftfe::utils::DEVICEBLAS_COMPUTE_32F_FAST_16BF;
      else if (d_opType == tensorOpDataType::fp16)
        computeType = dftfe::utils::DEVICEBLAS_COMPUTE_32F_FAST_16F;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status = DFTFE_DEVICE_BLAS_INT(
        , GemmBatchedEx)(d_deviceBlasHandle,
                         transa,
                         transb,
                         dftfe::Int(m),
                         dftfe::Int(n),
                         dftfe::Int(k),
                         (const void *)alpha,
                         (const void **)A,
                         dftfe::utils::DEVICE_R_32F,
                         dftfe::Int(lda),
                         (const void **)B,
                         dftfe::utils::DEVICE_R_32F,
                         dftfe::Int(ldb),
                         (const void *)beta,
                         (void **)C,
                         dftfe::utils::DEVICE_R_32F,
                         dftfe::Int(ldc),
                         dftfe::Int(batchCount),
                         computeType,
                         dftfe::utils::DEVICEBLAS_GEMM_DEFAULT);


      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      const std::int64_t m_local          = std::int64_t(m);
      const std::int64_t n_local          = std::int64_t(n);
      const std::int64_t k_local          = std::int64_t(k);
      const std::int64_t lda_local        = std::int64_t(lda);
      const std::int64_t ldb_local        = std::int64_t(ldb);
      const std::int64_t ldc_local        = std::int64_t(ldc);
      const std::int64_t batchCount_local = std::int64_t(batchCount);
      DEVICEBLAS_API_CHECK(
        DFTFE_DEVICE_BLAS_INT(S, gemm_batch)(d_deviceBlasHandle,
                                             &transa,
                                             &transb,
                                             &m_local,
                                             &n_local,
                                             &k_local,
                                             alpha,
                                             A,
                                             &lda_local,
                                             B,
                                             &ldb_local,
                                             beta,
                                             C,
                                             &ldc_local,
                                             1,
                                             &batchCount_local,
                                             computeType));
#endif
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char                 transA,
      const char                 transB,
      const dftfe::uInt          m,
      const dftfe::uInt          n,
      const dftfe::uInt          k,
      const std::complex<float> *alpha,
      const std::complex<float> *A[],
      const dftfe::uInt          lda,
      const std::complex<float> *B[],
      const dftfe::uInt          ldb,
      const std::complex<float> *beta,
      std::complex<float>       *C[],
      const dftfe::uInt          ldc,
      const dftfe::Int           batchCount)
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasComputeType_t computeType =
        dftfe::utils::DEVICEBLAS_COMPUTE_32F;
      if (d_opType == tensorOpDataType::tf32)
        computeType = dftfe::utils::DEVICEBLAS_COMPUTE_32F_FAST_TF32;
      else if (d_opType == tensorOpDataType::bf16)
        computeType = dftfe::utils::DEVICEBLAS_COMPUTE_32F_FAST_16BF;
      else if (d_opType == tensorOpDataType::fp16)
        computeType = dftfe::utils::DEVICEBLAS_COMPUTE_32F_FAST_16F;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status = DFTFE_DEVICE_BLAS_INT(
        , GemmBatchedEx)(d_deviceBlasHandle,
                         transa,
                         transb,
                         dftfe::Int(m),
                         dftfe::Int(n),
                         dftfe::Int(k),
                         (const void *)alpha,
                         (const void **)A,
                         dftfe::utils::DEVICE_C_32F,
                         dftfe::Int(lda),
                         (const void **)B,
                         dftfe::utils::DEVICE_C_32F,
                         dftfe::Int(ldb),
                         (const void *)beta,
                         (void **)C,
                         dftfe::utils::DEVICE_C_32F,
                         dftfe::Int(ldc),
                         dftfe::Int(batchCount),
                         computeType,
                         dftfe::utils::DEVICEBLAS_GEMM_DEFAULT);

      DEVICEBLAS_API_CHECK(status);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      const std::int64_t m_local          = std::int64_t(m);
      const std::int64_t n_local          = std::int64_t(n);
      const std::int64_t k_local          = std::int64_t(k);
      const std::int64_t lda_local        = std::int64_t(lda);
      const std::int64_t ldb_local        = std::int64_t(ldb);
      const std::int64_t ldc_local        = std::int64_t(ldc);
      const std::int64_t batchCount_local = std::int64_t(batchCount);

      DEVICEBLAS_API_CHECK(
        DFTFE_DEVICE_BLAS_INT(C, gemm_batch)(d_deviceBlasHandle,
                                             &transa,
                                             &transb,
                                             &m_local,
                                             &n_local,
                                             &k_local,
                                             alpha,
                                             A,
                                             &lda_local,
                                             B,
                                             &ldb_local,
                                             beta,
                                             C,
                                             &ldc_local,
                                             1,
                                             &batchCount_local,
                                             computeType));
#endif
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xnrm2(
      const dftfe::uInt           n,
      const std::complex<double> *x,
      const dftfe::uInt           incx,
      const MPI_Comm             &mpi_communicator,
      double                     *result)
    {
      double localResult = 0.0;
      *result            = 0.0;

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status = DFTFE_DEVICE_BLAS_INT(D, znrm2)(
        d_deviceBlasHandle,
        dftfe::Int(n),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(x),
        dftfe::Int(incx),
        &localResult);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      double *dev_res = sycl::malloc_device<double>(1, d_deviceBlasHandle);
      if (!dev_res)
        throw std::bad_alloc{};
      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(D, nrm2)(
        d_deviceBlasHandle,
        dftfe::Int(n),
        dftfe::utils::makeDataTypeDeviceBlasCompatible(x),
        dftfe::Int(incx),
        dev_res));
      d_deviceBlasHandle.memcpy(&localResult, dev_res, sizeof(double)).wait();
      sycl::free(dev_res, d_deviceBlasHandle);
      d_deviceBlasHandle.wait();
#endif
      localResult *= localResult;
      MPI_Allreduce(
        &localResult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
      *result = std::sqrt(*result);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xnrm2(
      const dftfe::uInt n,
      const double     *x,
      const dftfe::uInt incx,
      const MPI_Comm   &mpi_communicator,
      double           *result)
    {
      double localResult = 0.0;
      *result            = 0.0;
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA) || defined(DFTFE_WITH_DEVICE_LANG_HIP)
      dftfe::utils::deviceBlasStatus_t status = DFTFE_DEVICE_BLAS_INT(D, nrm2)(
        d_deviceBlasHandle, dftfe::Int(n), x, dftfe::Int(incx), &localResult);
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
      double *dev_res = sycl::malloc_device<double>(1, d_deviceBlasHandle);
      if (!dev_res)
        throw std::bad_alloc{};
      DEVICEBLAS_API_CHECK(DFTFE_DEVICE_BLAS_INT(D, nrm2)(
        d_deviceBlasHandle, dftfe::Int(n), x, dftfe::Int(incx), dev_res));
      d_deviceBlasHandle.memcpy(&localResult, dev_res, sizeof(double)).wait();
      sycl::free(dev_res, d_deviceBlasHandle);
      d_deviceBlasHandle.wait();
#endif
      localResult *= localResult;
      MPI_Allreduce(
        &localResult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
      *result = std::sqrt(*result);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      ValueType1       *x,
      const ValueType2  alpha,
      const dftfe::uInt n)
    {
      DFTFE_LAUNCH_KERNEL(ascalDeviceKernel,
                          n / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          d_streamId,
                          n,
                          dftfe::utils::makeDataTypeDeviceCompatible(x),
                          dftfe::utils::makeDataTypeDeviceCompatible(alpha));
    }

    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::copyComplexArrToRealArrs(
      const dftfe::uInt       size,
      const ValueTypeComplex *complexArr,
      ValueTypeReal          *realArr,
      ValueTypeReal          *imagArr)
    {
      DFTFE_LAUNCH_KERNEL(copyComplexArrToRealArrsDeviceKernel,
                          size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          d_streamId,
                          size,
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            complexArr),
                          realArr,
                          imagArr);
    }



    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::copyRealArrsToComplexArr(
      const dftfe::uInt    size,
      const ValueTypeReal *realArr,
      const ValueTypeReal *imagArr,
      ValueTypeComplex    *complexArr)
    {
      DFTFE_LAUNCH_KERNEL(copyRealArrsToComplexArrDeviceKernel,
                          size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          d_streamId,
                          size,
                          realArr,
                          imagArr,
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            complexArr));
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(const dftfe::uInt size,
                                       const ValueType1 *valueType1Arr,
                                       ValueType2       *valueType2Arr)
    {
      DFTFE_LAUNCH_KERNEL(
        copyValueType1ArrToValueType2ArrDeviceKernel,
        size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        d_streamId,
        size,
        dftfe::utils::makeDataTypeDeviceCompatible(valueType1Arr),
        dftfe::utils::makeDataTypeDeviceCompatible(valueType2Arr));
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      copyBlockDiagonalValueType1OffDiagonalValueType2FromValueType1Arr(
        const dftfe::uInt B,
        const dftfe::uInt DRem,
        const dftfe::uInt D,
        const ValueType1 *valueType1SrcArray,
        ValueType1       *valueType1DstArray,
        ValueType2       *valueType2DstArray)
    {
      const dftfe::uInt size = D * B;
      DFTFE_LAUNCH_KERNEL(
        copyBlockDiagonalValueType1OffDiagonalValueType2FromValueType1ArrDeviceKernel,
        size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        d_streamId,
        B,
        DRem,
        D,
        dftfe::utils::makeDataTypeDeviceCompatible(valueType1SrcArray),
        dftfe::utils::makeDataTypeDeviceCompatible(valueType1DstArray),
        dftfe::utils::makeDataTypeDeviceCompatible(valueType2DstArray));
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const ValueType1  *copyFromVec,
      ValueType2        *copyToVecBlock,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds)
    {
      DFTFE_LAUNCH_KERNEL(
        stridedCopyToBlockDeviceKernel,
        (contiguousBlockSize * numContiguousBlocks) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        d_streamId,
        contiguousBlockSize,
        numContiguousBlocks,
        dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
        dftfe::utils::makeDataTypeDeviceCompatible(copyToVecBlock),
        copyFromVecStartingContiguousBlockIds);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const dftfe::uInt  startingVecId,
      const ValueType1  *copyFromVec,
      ValueType2        *copyToVecBlock,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds)
    {
      DFTFE_LAUNCH_KERNEL(
        stridedCopyToBlockDeviceKernel,
        (contiguousBlockSize * numContiguousBlocks) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        d_streamId,
        contiguousBlockSize,
        numContiguousBlocks,
        startingVecId,
        dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
        dftfe::utils::makeDataTypeDeviceCompatible(copyToVecBlock),
        copyFromVecStartingContiguousBlockIds);
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleCopy(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const ValueType1   a,
      const ValueType1  *s,
      const ValueType2  *copyFromVec,
      ValueType2        *copyToVecBlock,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds)
    {
      DFTFE_LAUNCH_KERNEL(
        stridedCopyToBlockScaleDeviceKernel,
        (contiguousBlockSize * numContiguousBlocks) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        d_streamId,
        contiguousBlockSize,
        numContiguousBlocks,
        dftfe::utils::makeDataTypeDeviceCompatible(a),
        dftfe::utils::makeDataTypeDeviceCompatible(s),
        dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
        dftfe::utils::makeDataTypeDeviceCompatible(copyToVecBlock),
        copyFromVecStartingContiguousBlockIds);
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const ValueType1  *copyFromVecBlock,
      ValueType2        *copyToVec,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds)
    {
      DFTFE_LAUNCH_KERNEL(stridedCopyFromBlockDeviceKernel,
                          (contiguousBlockSize * numContiguousBlocks) /
                              dftfe::utils::DEVICE_BLOCK_SIZE +
                            1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          d_streamId,
                          contiguousBlockSize,
                          numContiguousBlocks,
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            copyFromVecBlock),
                          dftfe::utils::makeDataTypeDeviceCompatible(copyToVec),
                          copyFromVecStartingContiguousBlockIds);
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      stridedCopyToBlockConstantStride(const dftfe::uInt blockSizeTo,
                                       const dftfe::uInt blockSizeFrom,
                                       const dftfe::uInt numBlocks,
                                       const dftfe::uInt startingId,
                                       const ValueType1 *copyFromVec,
                                       ValueType2       *copyToVec)
    {
      DFTFE_LAUNCH_KERNEL(
        stridedCopyToBlockConstantStrideDeviceKernel,
        (blockSizeTo * numBlocks) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        d_streamId,
        blockSizeTo,
        blockSizeFrom,
        numBlocks,
        startingId,
        dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
        dftfe::utils::makeDataTypeDeviceCompatible(copyToVec));
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
      const dftfe::uInt blockSize,
      const dftfe::uInt strideTo,
      const dftfe::uInt strideFrom,
      const dftfe::uInt numBlocks,
      const dftfe::uInt startingToId,
      const dftfe::uInt startingFromId,
      const ValueType1 *copyFromVec,
      ValueType2       *copyToVec)
    {
      DFTFE_LAUNCH_KERNEL(
        stridedCopyConstantStrideDeviceKernel,
        (blockSize * numBlocks) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        d_streamId,
        blockSize,
        strideTo,
        strideFrom,
        numBlocks,
        startingToId,
        startingFromId,
        dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
        dftfe::utils::makeDataTypeDeviceCompatible(copyToVec));
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      stridedCopyFromBlockConstantStride(const dftfe::uInt blockSizeTo,
                                         const dftfe::uInt blockSizeFrom,
                                         const dftfe::uInt numBlocks,
                                         const dftfe::uInt startingId,
                                         const ValueType1 *copyFromVec,
                                         ValueType2       *copyToVec)
    {
      DFTFE_LAUNCH_KERNEL(
        stridedCopyFromBlockConstantStrideDeviceKernel,
        (blockSizeFrom * numBlocks) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        d_streamId,
        blockSizeTo,
        blockSizeFrom,
        numBlocks,
        startingId,
        dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
        dftfe::utils::makeDataTypeDeviceCompatible(copyToVec));
    }
    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const ValueType1  a,
      const ValueType1 *s,
      ValueType2       *x)
    {
      DFTFE_LAUNCH_KERNEL(stridedBlockScaleDeviceKernel,
                          (contiguousBlockSize * numContiguousBlocks) /
                              dftfe::utils::DEVICE_BLOCK_SIZE +
                            1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          d_streamId,
                          contiguousBlockSize,
                          numContiguousBlocks,
                          dftfe::utils::makeDataTypeDeviceCompatible(a),
                          dftfe::utils::makeDataTypeDeviceCompatible(s),
                          dftfe::utils::makeDataTypeDeviceCompatible(x));
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::hadamardProduct(
      const dftfe::uInt m,
      const ValueType  *X,
      const ValueType  *Y,
      ValueType        *output)
    {
      DFTFE_LAUNCH_KERNEL(hadamardProductKernel,
                          m / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          d_streamId,
                          m,
                          dftfe::utils::makeDataTypeDeviceCompatible(X),
                          dftfe::utils::makeDataTypeDeviceCompatible(Y),
                          dftfe::utils::makeDataTypeDeviceCompatible(output));
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::hadamardProductWithConj(
      const dftfe::uInt m,
      const ValueType  *X,
      const ValueType  *Y,
      ValueType        *output)
    {
      DFTFE_LAUNCH_KERNEL(hadamardProductWithConjKernel,
                          (m) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          d_streamId,
                          m,
                          dftfe::utils::makeDataTypeDeviceCompatible(X),
                          dftfe::utils::makeDataTypeDeviceCompatible(Y),
                          dftfe::utils::makeDataTypeDeviceCompatible(output));
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::addVecOverContinuousIndex(
      const dftfe::uInt numContiguousBlocks,
      const dftfe::uInt contiguousBlockSize,
      const ValueType  *input1,
      const ValueType  *input2,
      ValueType        *output)
    {
      DFTFE_LAUNCH_KERNEL(addVecOverContinuousIndexKernel,
                          (numContiguousBlocks) /
                              dftfe::utils::DEVICE_BLOCK_SIZE +
                            1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          d_streamId,
                          numContiguousBlocks,
                          contiguousBlockSize,
                          dftfe::utils::makeDataTypeDeviceCompatible(input1),
                          dftfe::utils::makeDataTypeDeviceCompatible(input2),
                          dftfe::utils::makeDataTypeDeviceCompatible(output));
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleColumnWise(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const ValueType  *beta,
      ValueType        *x)
    {
      DFTFE_LAUNCH_KERNEL(stridedBlockScaleColumnWiseKernel,
                          (contiguousBlockSize * numContiguousBlocks) /
                              dftfe::utils::DEVICE_BLOCK_SIZE +
                            1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          d_streamId,
                          contiguousBlockSize,
                          numContiguousBlocks,
                          dftfe::utils::makeDataTypeDeviceCompatible(beta),
                          dftfe::utils::makeDataTypeDeviceCompatible(x));
    }
    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      stridedBlockScaleAndAddColumnWise(const dftfe::uInt contiguousBlockSize,
                                        const dftfe::uInt numContiguousBlocks,
                                        const ValueType  *x,
                                        const ValueType  *beta,
                                        ValueType        *y)
    {
      DFTFE_LAUNCH_KERNEL(stridedBlockScaleAndAddColumnWiseKernel,
                          (contiguousBlockSize * numContiguousBlocks) /
                              dftfe::utils::DEVICE_BLOCK_SIZE +
                            1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          d_streamId,
                          contiguousBlockSize,
                          numContiguousBlocks,
                          dftfe::utils::makeDataTypeDeviceCompatible(x),
                          dftfe::utils::makeDataTypeDeviceCompatible(beta),
                          dftfe::utils::makeDataTypeDeviceCompatible(y));
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      stridedBlockScaleAndAddTwoVecColumnWise(
        const dftfe::uInt contiguousBlockSize,
        const dftfe::uInt numContiguousBlocks,
        const ValueType  *x,
        const ValueType  *alpha,
        const ValueType  *y,
        const ValueType  *beta,
        ValueType        *z)
    {
      DFTFE_LAUNCH_KERNEL(stridedBlockScaleAndAddTwoVecColumnWiseKernel,
                          (contiguousBlockSize * numContiguousBlocks) /
                              dftfe::utils::DEVICE_BLOCK_SIZE +
                            1,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          d_streamId,
                          contiguousBlockSize,
                          numContiguousBlocks,
                          dftfe::utils::makeDataTypeDeviceCompatible(x),
                          dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                          dftfe::utils::makeDataTypeDeviceCompatible(y),
                          dftfe::utils::makeDataTypeDeviceCompatible(beta),
                          dftfe::utils::makeDataTypeDeviceCompatible(z));
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::MultiVectorXDot(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const ValueType  *X,
      const ValueType  *Y,
      const ValueType  *onesVec,
      ValueType        *tempVector,
      ValueType        *tempResults,
      ValueType        *result)
    {
      hadamardProductWithConj(contiguousBlockSize * numContiguousBlocks,
                              X,
                              Y,
                              tempVector);

      ValueType   alpha  = 1.0;
      ValueType   beta   = 0.0;
      dftfe::uInt numVec = 1;
      xgemm('N',
            'T',
            numVec,
            contiguousBlockSize,
            numContiguousBlocks,
            &alpha,
            onesVec,
            numVec,
            tempVector,
            contiguousBlockSize,
            &beta,
            tempResults,
            numVec);
      dftfe::utils::deviceMemcpyD2H(dftfe::utils::makeDataTypeDeviceCompatible(
                                      result),
                                    tempResults,
                                    contiguousBlockSize * sizeof(ValueType));
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::MultiVectorXDot(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const ValueType  *X,
      const ValueType  *Y,
      const ValueType  *onesVec,
      ValueType        *tempVector,
      ValueType        *tempResults,
      const MPI_Comm   &mpi_communicator,
      ValueType        *result)

    {
      MultiVectorXDot(contiguousBlockSize,
                      numContiguousBlocks,
                      X,
                      Y,
                      onesVec,
                      tempVector,
                      tempResults,
                      result);

      MPI_Allreduce(MPI_IN_PLACE,
                    &result[0],
                    contiguousBlockSize,
                    dataTypes::mpi_type_id(&result[0]),
                    MPI_SUM,
                    mpi_communicator);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::rightDiagonalScale(
      const dftfe::uInt numberofVectors,
      const dftfe::uInt sizeOfVector,
      ValueType1       *X,
      ValueType2       *D)
    {
      DFTFE_LAUNCH_KERNEL(computeRightDiagonalScaleKernel,
                          (numberofVectors +
                           (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                            dftfe::utils::DEVICE_BLOCK_SIZE * sizeOfVector,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          d_streamId,
                          dftfe::utils::makeDataTypeDeviceCompatible(D),
                          dftfe::utils::makeDataTypeDeviceCompatible(X),
                          numberofVectors,
                          sizeOfVector);
    }

#include "./BLASWrapperDevice.inst.cc"
  } // End of namespace linearAlgebra
} // End of namespace dftfe
