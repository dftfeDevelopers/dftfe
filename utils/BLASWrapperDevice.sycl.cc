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
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/blas.hpp>
#include "BLASWrapperDeviceKernel.cc"
#define DFTFE_WITH_DEVICE_MKL 1
namespace dftfe
{
  namespace linearAlgebra
  {

    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::BLASWrapper()
    {
      create();
    }

    dftfe::utils::deviceStream_t &
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::getDeviceStream()
    {
      return d_streamId;
    }

    dftfe::utils::deviceStream_t &
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::getDeviceBlasHandle()
    {
      return d_deviceBlasHandle;
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::destroy()
    {
      dftfe::utils::deviceBlasStatus_t status = {};
      return status;
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::setStream(
      dftfe::utils::deviceStream_t streamId)
    {
      dftfe::utils::deviceBlasStatus_t status = {};
      d_streamId                              = streamId;
      return status;
    }

    auto exception_handler = [](sycl::exception_list exceptions) {
      for (std::exception_ptr const &e : exceptions)
        {
          try
            {
              std::rethrow_exception(e);
            }
          catch (sycl::exception const &e)
            {
              std::cerr << "Caught asynchronous SYCL exception:" << std::endl
                        << e.what() << std::endl
                        << "Exception caught at file:" << __FILE__
                        << ", line:" << __LINE__ << std::endl;
            }
        }
    };

    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::create()
    {
      d_streamId         = sycl::queue{sycl::gpu_selector_v, exception_handler};
      d_deviceBlasHandle = sycl::queue{sycl::gpu_selector_v, exception_handler};
      return dftfe::utils::deviceBlasSuccess;
    }

    dftfe::utils::deviceBlasStatus_t
    setMathMode(dftfe::utils::deviceBlasMath_t mathMode)
    {
#ifdef DFTFE_WITH_DEVICE_MKL
      if (mathMode == dftfe::utils::DEVICEBLAS_TF32_TENSOR_OP_MATH)
        setenv("MKL_BLAS_COMPUTE_MODE", "FLOAT_TO_TF32", 1);
      else if (mathMode == dftfe::utils::DEVICEBLAS_DEFAULT_MATH)
        setenv("MKL_BLAS_COMPUTE_MODE", "STANDARD", 1);
      else if (mathMode == oneapi::mkl::blas::compute_mode::float_to_bf16)
        setenv("MKL_BLAS_COMPUTE_MODE", "FLOAT_TO_BF16", 1);
      else if (mathMode == oneapi::mkl::blas::compute_mode::float_to_bf16x2)
        setenv("MKL_BLAS_COMPUTE_MODE", "FLOAT_TO_BF16X2", 1);
      else if (mathMode == oneapi::mkl::blas::compute_mode::float_to_bf16x3)
        setenv("MKL_BLAS_COMPUTE_MODE", "FLOAT_TO_BF16X3", 1);
      else if (mathMode == oneapi::mkl::blas::compute_mode::complex_3m)
        setenv("MKL_BLAS_COMPUTE_MODE", "COMPLEX_3M", 1);
#endif
      return dftfe::utils::deviceBlasSuccess;
    }

    template <typename ValueType>
    ValueType *
    device_allocation(dftfe::utils::deviceStream_t d_streamId, unsigned int n)
    {
      ValueType *A_device = sycl::malloc_device<ValueType>(n, d_streamId);
      d_streamId.wait();
      return A_device;
    }

    template <typename ValueType>
    void
    device_copy(dftfe::utils::deviceStream_t d_streamId,
                ValueType                   *A_device,
                const ValueType             *A,
                int                          n)
    {
      dftfe::utils::deviceEvent_t event =
        d_streamId.memcpy(A_device, A, sizeof(ValueType) * n);
      DEVICE_API_CHECK(event);
    }

    template <typename ValueType>
    ValueType *
    device_allocation_copy(dftfe::utils::deviceStream_t d_streamId,
                           const ValueType             *A,
                           int                          n)
    {
      ValueType *A_device = sycl::malloc_device<ValueType>(n, d_streamId);
      dftfe::utils::deviceEvent_t event =
        d_streamId.memcpy(A_device, A, sizeof(ValueType) * n);
      DEVICE_API_CHECK(event);
      return A_device;
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xcopy(
      const unsigned int n,
      const double      *x,
      const unsigned int incx,
      double            *y,
      const unsigned int incy) const
    {
#ifdef DFTFE_WITH_DEVICE_MKL
      dftfe::utils::deviceEvent_t event =
        oneapi::mkl::blas::column_major::copy(d_streamId, n, x, incx, y, incy);
#else
      size_t total_workitems =
        ((n / incx + 1) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          size_t index = ind.get_global_id(0);
          if (incx * index < n)
            y[incy * index] = x[incx * index];
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xcopy(
      const unsigned int          n,
      const std::complex<double> *x,
      const unsigned int          incx,
      std::complex<double>       *y,
      const unsigned int          incy) const
    {
#ifdef DFTFE_WITH_DEVICE_MKL
      dftfe::utils::deviceEvent_t event =
        oneapi::mkl::blas::column_major::copy(d_streamId, n, x, incx, y, incy);
#else
      size_t total_workitems =
        ((n / incx + 1) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          size_t index = ind.get_global_id(0);
          if (incx * index < n)
            {
              y[incy * index].real(x[incx * index].real());
              y[incy * index].imag(x[incx * index].imag());
            }
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xcopy(
      const unsigned int n,
      const float       *x,
      const unsigned int incx,
      float             *y,
      const unsigned int incy) const
    {
#ifdef DFTFE_WITH_DEVICE_MKL
      dftfe::utils::deviceEvent_t event =
        oneapi::mkl::blas::column_major::copy(d_streamId, n, x, incx, y, incy);
#else
      size_t total_workitems =
        ((n / incx + 1) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          size_t index = ind.get_global_id(0);
          if (incx * index < n)
            y[incy * index] = x[incx * index];
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xcopy(
      const unsigned int         n,
      const std::complex<float> *x,
      const unsigned int         incx,
      std::complex<float>       *y,
      const unsigned int         incy) const
    {
#ifdef DFTFE_WITH_DEVICE_MKL
      dftfe::utils::deviceEvent_t event =
        oneapi::mkl::blas::column_major::copy(d_streamId, n, x, incx, y, incy);
#else
      size_t total_workitems =
        ((n / incx + 1) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          size_t index = ind.get_global_id(0);
          if (incx * index < n)
            {
              y[incy * index].real(x[incx * index].real());
              y[incy * index].imag(x[incx * index].imag());
            }
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
      const char                  transA,
      const char                  transB,
      const unsigned int          m,
      const unsigned int          n,
      const unsigned int          k,
      const std::complex<double> *alpha,
      const std::complex<double> *A,
      const unsigned int          lda,
      const std::complex<double> *B,
      const unsigned int          ldb,
      const std::complex<double> *beta,
      std::complex<double>       *C,
      const unsigned int          ldc) const
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
#ifdef DFTFE_WITH_DEVICE_MKL
      dftfe::utils::deviceEvent_t event =
        oneapi::mkl::blas::column_major::gemm(d_streamId,
                                              transa,
                                              transb,
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
#else
      size_t total_workitems = (n / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
                               dftfe::utils::DEVICE_BLOCK_SIZE;
      auto                        alpha_local = alpha[0];
      auto                        beta_local  = beta[0];
      dftfe::utils::deviceEvent_t event       = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
          size_t col = item.get_global_id(0);

          if (col >= n)
            {
              return;
            }

          for (size_t row = 0; row < m; ++row)
            {
              C[col * ldc + row] = beta_local * C[col * ldc + row];

              for (size_t i = 0; i < k; ++i)
                {
                  C[col * ldc + row] +=
                    alpha_local *
                    ((transa == dftfe::utils::DEVICEBLAS_OP_N) ?
                             A[i * lda + row] :
                             A[row * lda + i]) *
                    ((transb == dftfe::utils::DEVICEBLAS_OP_N) ?
                             B[col * ldb + i] :
                             B[i * ldb + col]);
                }
            }
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
      const char         transA,
      const char         transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const double      *alpha,
      const double      *A,
      const unsigned int lda,
      const double      *B,
      const unsigned int ldb,
      const double      *beta,
      double            *C,
      const unsigned int ldc) const
    {
      d_streamId.wait();
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
#ifdef DFTFE_WITH_DEVICE_MKL
      dftfe::utils::deviceEvent_t event =
        oneapi::mkl::blas::column_major::gemm(d_streamId,
                                              transa,
                                              transb,
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
#else
      size_t total_workitems = (n / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
                               dftfe::utils::DEVICE_BLOCK_SIZE;
      auto                        alpha_local = alpha[0];
      auto                        beta_local  = beta[0];
      dftfe::utils::deviceEvent_t event       = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
          size_t col = item.get_global_id(0);

          if (col >= n)
            {
              return;
            }

          for (size_t row = 0; row < m; ++row)
            {
              C[col * ldc + row] = beta_local * C[col * ldc + row];

              for (size_t i = 0; i < k; ++i)
                {
                  C[col * ldc + row] +=
                    alpha_local *
                    ((transa == dftfe::utils::DEVICEBLAS_OP_N) ?
                             A[i * lda + row] :
                             A[row * lda + i]) *
                    ((transb == dftfe::utils::DEVICEBLAS_OP_N) ?
                             B[col * ldb + i] :
                             B[i * ldb + col]);
                }
            }
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
      const char         transA,
      const char         transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const float       *alpha,
      const float       *A,
      const unsigned int lda,
      const float       *B,
      const unsigned int ldb,
      const float       *beta,
      float             *C,
      const unsigned int ldc) const
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
#ifdef DFTFE_WITH_DEVICE_MKL
      dftfe::utils::deviceEvent_t event =
        oneapi::mkl::blas::column_major::gemm(d_streamId,
                                              transa,
                                              transb,
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
#else
      size_t total_workitems = (n / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
                               dftfe::utils::DEVICE_BLOCK_SIZE;
      auto                        alpha_local = alpha[0];
      auto                        beta_local  = beta[0];
      dftfe::utils::deviceEvent_t event       = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
          size_t col = item.get_global_id(0);

          if (col >= n)
            {
              return;
            }

          for (size_t row = 0; row < m; ++row)
            {
              C[col * ldc + row] = beta_local * C[col * ldc + row];

              for (size_t i = 0; i < k; ++i)
                {
                  C[col * ldc + row] +=
                    alpha_local *
                    ((transa == dftfe::utils::DEVICEBLAS_OP_N) ?
                             A[i * lda + row] :
                             A[row * lda + i]) *
                    ((transb == dftfe::utils::DEVICEBLAS_OP_N) ?
                             B[col * ldb + i] :
                             B[i * ldb + col]);
                }
            }
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
      const char                 transA,
      const char                 transB,
      const unsigned int         m,
      const unsigned int         n,
      const unsigned int         k,
      const std::complex<float> *alpha,
      const std::complex<float> *A,
      const unsigned int         lda,
      const std::complex<float> *B,
      const unsigned int         ldb,
      const std::complex<float> *beta,
      std::complex<float>       *C,
      const unsigned int         ldc) const
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
#ifdef DFTFE_WITH_DEVICE_MKL
      dftfe::utils::deviceEvent_t event =
        oneapi::mkl::blas::column_major::gemm(d_streamId,
                                              transa,
                                              transb,
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
#else
      size_t total_workitems = (n / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
                               dftfe::utils::DEVICE_BLOCK_SIZE;
      auto                        alpha_local = alpha[0];
      auto                        beta_local  = beta[0];
      dftfe::utils::deviceEvent_t event       = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
          size_t col = item.get_global_id(0);

          if (col >= n)
            {
              return;
            }

          for (size_t row = 0; row < m; ++row)
            {
              C[col * ldc + row] = beta_local * C[col * ldc + row];

              for (size_t i = 0; i < k; ++i)
                {
                  C[col * ldc + row] +=
                    alpha_local *
                    ((transa == dftfe::utils::DEVICEBLAS_OP_N) ?
                             A[i * lda + row] :
                             A[row * lda + i]) *
                    ((transb == dftfe::utils::DEVICEBLAS_OP_N) ?
                             B[col * ldb + i] :
                             B[i * ldb + col]);
                }
            }
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemv(
      const char         transA,
      const unsigned int m,
      const unsigned int n,
      const double      *alpha,
      const double      *A,
      const unsigned int lda,
      const double      *x,
      const unsigned int incx,
      const double      *beta,
      double            *y,
      const unsigned int incy) const
    {
      dftfe::utils::deviceBlasOperation_t transa;
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
#ifdef DFTFE_WITH_DEVICE_MKL
      dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemv(
        d_streamId, transa, m, n, alpha, A, lda, x, incx, beta, y, incy);
#else
      unsigned int m_local = m, n_local = n;
      if (transa == dftfe::utils::DEVICEBLAS_OP_T)
        {
          m_local = n;
          n_local = m;
        }
      size_t total_workitems = (m_local / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
                               dftfe::utils::DEVICE_BLOCK_SIZE;
      auto                        alpha_local = alpha[0];
      auto                        beta_local  = beta[0];
      dftfe::utils::deviceEvent_t event       = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
          size_t row = item.get_global_id(0);

          if (row * incy < m_local)
            {
              y[row * incy] = beta_local * y[row * incy];
              for (size_t col = 0; col < n_local; ++col)
                {
                  if (col * incx >= n_local)
                    {
                      break;
                    }
                  y[row * incy] += alpha_local * x[col * incx] *
                                   ((transa == dftfe::utils::DEVICEBLAS_OP_N) ?
                                            A[col * lda + row] :
                                            A[row * lda + col]);
                }
            }
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemv(
      const char         transA,
      const unsigned int m,
      const unsigned int n,
      const float       *alpha,
      const float       *A,
      const unsigned int lda,
      const float       *x,
      const unsigned int incx,
      const float       *beta,
      float             *y,
      const unsigned int incy) const
    {
      dftfe::utils::deviceBlasOperation_t transa;
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
#ifdef DFTFE_WITH_DEVICE_MKL
      dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemv(
        d_streamId, transa, m, n, alpha, A, lda, x, incx, beta, y, incy);
#else
      unsigned int m_local = m, n_local = n;
      if (transa == dftfe::utils::DEVICEBLAS_OP_T)
        {
          m_local = n;
          n_local = m;
        }
      size_t total_workitems = (m_local / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
                               dftfe::utils::DEVICE_BLOCK_SIZE;
      auto                        alpha_local = alpha[0];
      auto                        beta_local  = beta[0];
      dftfe::utils::deviceEvent_t event       = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
          size_t row = item.get_global_id(0);

          if (row * incy < m_local)
            {
              y[row * incy] = beta_local * y[row * incy];
              for (size_t col = 0; col < n_local; ++col)
                {
                  if (col * incx >= n_local)
                    {
                      break;
                    }
                  y[row * incy] += alpha_local * x[col * incx] *
                                   ((transa == dftfe::utils::DEVICEBLAS_OP_N) ?
                                            A[col * lda + row] :
                                            A[row * lda + col]);
                }
            }
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemv(
      const char                  transA,
      const unsigned int          m,
      const unsigned int          n,
      const std::complex<double> *alpha,
      const std::complex<double> *A,
      const unsigned int          lda,
      const std::complex<double> *x,
      const unsigned int          incx,
      const std::complex<double> *beta,
      std::complex<double>       *y,
      const unsigned int          incy) const
    {
      dftfe::utils::deviceBlasOperation_t transa;
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
#ifdef DFTFE_WITH_DEVICE_MKL
      dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemv(
        d_streamId, transa, m, n, alpha, A, lda, x, incx, beta, y, incy);
#else
      unsigned int m_local = m, n_local = n;
      if (transa == dftfe::utils::DEVICEBLAS_OP_T)
        {
          m_local = n;
          n_local = m;
        }
      size_t total_workitems = (m_local / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
                               dftfe::utils::DEVICE_BLOCK_SIZE;
      auto                        alpha_local = alpha[0];
      auto                        beta_local  = beta[0];
      dftfe::utils::deviceEvent_t event       = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
          size_t row = item.get_global_id(0);

          if (row * incy < m_local)
            {
              y[row * incy] = beta_local * y[row * incy];
              for (size_t col = 0; col < n_local; ++col)
                {
                  if (col * incx >= n_local)
                    {
                      break;
                    }
                  y[row * incy] += alpha_local * x[col * incx] *
                                   ((transa == dftfe::utils::DEVICEBLAS_OP_N) ?
                                            A[col * lda + row] :
                                            A[row * lda + col]);
                }
            }
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemv(
      const char                 transA,
      const unsigned int         m,
      const unsigned int         n,
      const std::complex<float> *alpha,
      const std::complex<float> *A,
      const unsigned int         lda,
      const std::complex<float> *x,
      const unsigned int         incx,
      const std::complex<float> *beta,
      std::complex<float>       *y,
      const unsigned int         incy) const
    {
      dftfe::utils::deviceBlasOperation_t transa;
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
#ifdef DFTFE_WITH_DEVICE_MKL
      dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::gemv(
        d_streamId, transa, m, n, alpha, A, lda, x, incx, beta, y, incy);
#else
      unsigned int m_local = m, n_local = n;
      if (transa == dftfe::utils::DEVICEBLAS_OP_T)
        {
          m_local = n;
          n_local = m;
        }
      size_t total_workitems = (m_local / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
                               dftfe::utils::DEVICE_BLOCK_SIZE;
      auto                        alpha_local = alpha[0];
      auto                        beta_local  = beta[0];
      dftfe::utils::deviceEvent_t event       = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
          size_t row = item.get_global_id(0);

          if (row * incy < m_local)
            {
              y[row * incy] = beta_local * y[row * incy];
              for (size_t col = 0; col < n_local; ++col)
                {
                  if (col * incx >= n_local)
                    {
                      break;
                    }
                  y[row * incy] += alpha_local * x[col * incx] *
                                   ((transa == dftfe::utils::DEVICEBLAS_OP_N) ?
                                            A[col * lda + row] :
                                            A[row * lda + col]);
                }
            }
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xaxpy(
      const unsigned int n,
      const double      *alpha,
      const double      *x,
      const unsigned int incx,
      double            *y,
      const unsigned int incy) const
    {
#ifdef DFTFE_WITH_DEVICE_MKL
      dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::axpy(
        d_streamId, n, alpha, x, incx, y, incy);
#else
      size_t total_workitems = (n / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
                               dftfe::utils::DEVICE_BLOCK_SIZE;
      auto                        alpha_local = alpha[0];
      dftfe::utils::deviceEvent_t event       = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          size_t index = ind.get_global_id(0);
          if (incx * index < n && incy * index < n)
            {
              y[incy * index] = y[incy * index] + alpha_local * x[incx * index];
            }
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xaxpy(
      const unsigned int          n,
      const std::complex<double> *alpha,
      const std::complex<double> *x,
      const unsigned int          incx,
      std::complex<double>       *y,
      const unsigned int          incy) const
    {
#ifdef DFTFE_WITH_DEVICE_MKL
      dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::axpy(
        d_streamId, n, alpha, x, incx, y, incy);
#else
      size_t total_workitems =
        ((n / incx + 1) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      auto                        alpha_local = alpha[0];
      dftfe::utils::deviceEvent_t event       = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          size_t index = ind.get_global_id(0);
          if (incx * index < n && incy * index < n)
            {
              y[incy * index] = y[incy * index] + alpha_local * x[incx * index];
            }
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::add(double       *y,
                                                        const double *x,
                                                        const double  alpha,
                                                        const unsigned int size)
    {
      xaxpy(size, &alpha, x, 1, y, 1);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const unsigned int n,
      const double      *x,
      const unsigned int incx,
      const double      *y,
      const unsigned int incy,
      double            *result) const
    {
#ifdef DFTFE_WITH_DEVICE_MKL
      dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::dot(
        d_streamId, n, x, incx, y, incy, result);
#else
      size_t total_workitems = (n / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
                               dftfe::utils::DEVICE_BLOCK_SIZE;

      sycl::buffer<double> x_buf(x, sycl::range<1>(n));
      sycl::buffer<double> y_buf(y, sycl::range<1>(n));
      sycl::buffer<double> sum_buf(result, sycl::range<1>(1));

      dftfe::utils::deviceEvent_t event =
        d_streamId.submit([&](sycl::handler &cgh) {
          auto x_acc = x_buf.get_access<sycl::access::mode::read>(cgh);
          auto y_acc = y_buf.get_access<sycl::access::mode::read>(cgh);
          auto sum_acc =
            sum_buf.get_access<sycl::access::mode::read_write>(cgh);

          cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            double val_x = x_acc[i * incx], val_y = y_acc[i * incy];
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
              atomic_sum(sum_acc[0]);
            atomic_sum.fetch_add(val_x * val_y);
          });
        });
      sycl::host_accessor sum_host_acc(sum_buf, sycl::read_only);
      *result = sum_host_acc[0];
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const unsigned int          n,
      const std::complex<double> *x,
      const unsigned int          incx,
      const std::complex<double> *y,
      const unsigned int          incy,
      std::complex<double>       *result) const
    {
#ifdef DFTFE_WITH_DEVICE_MKL
      dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::dotu(
        d_streamId, n, x, incx, y, incy, result);
#else
      size_t total_workitems =
        ((n / incx + 1) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          unsigned int global_id    = ind.get_global_id(0);
          unsigned int n_workgroups = ind.get_group_range(0);
          unsigned int n_workitems  = ind.get_local_range(0);
          for (unsigned int index = global_id; index < n;
               index += n_workgroups * n_workitems)
            {
              if (incx * global_id >= n)
                {
                  break;
                }
              // auto atomic_add_real = sycl::atomic_ref<double,
              // sycl::memory_order::relaxed,
              //                                     sycl::memory_scope::device,
              //                                     sycl::access::address_space::global_space>
              //                                     (reinterpret_cast<double*>(&result[0])[0]);
              // atomic_add_real +=
              // x[incx*global_id].real()*y[incy*global_id].real();

              // auto atomic_add_imag = sycl::atomic_ref<double,
              // sycl::memory_order::relaxed,
              //                                     sycl::memory_scope::device,
              //                                     sycl::access::address_space::global_space>
              //                                     (reinterpret_cast<double*>(&result[0])[1]);
              // atomic_add_imag +=
              // x[incx*global_id].imag()*y[incy*global_id].imag();
            }
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const unsigned int          n,
      const std::complex<double> *x,
      const unsigned int          incx,
      const std::complex<double> *y,
      const unsigned int          incy,
      const MPI_Comm             &mpi_communicator,
      std::complex<double>       *result) const
    {
      std::complex<double> localResult(0.0, 0.0);
      *result = std::complex<double>(0.0, 0.0);
#ifdef DFTFE_WITH_DEVICE_MKL
      std::complex<double> *localResult_device =
        sycl::malloc_device<std::complex<double>>(1, d_streamId);
      d_streamId.memcpy(localResult_device,
                        &localResult,
                        sizeof(std::complex<double>));
      dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::dotu(
        d_streamId, n, x, incx, y, incy, localResult_device);
      d_streamId.wait();
      d_streamId.memcpy(&localResult,
                        localResult_device,
                        sizeof(std::complex<double>));
#else
      size_t total_workitems =
        ((n / incx + 1) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;

      sycl::buffer<std::complex<double>, 1> partial_sums(total_workitems);
      dftfe::utils::deviceEvent_t           event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
          // size_t global_id = item.get_global_id(0);
          // size_t n_workgroups = item.get_group_range(0);
          // size_t n_workitems = item.get_local_range(0);

          // std::complex<double> partial_sum(0.0, 0.0);

          // for (size_t index = global_id; index < n; index += n_workgroups *
          // n_workitems) {
          //     if (index >= n) break;
          //     partial_sum += x[incx * index] * y[incy * index];
          // }

          // // Write partial sum to buffer
          // sycl::atomic_ref<std::complex<double>, sycl::memory_order::relaxed,
          //                 sycl::memory_scope::device,
          //                 sycl::access::address_space::global_space>
          //     (partial_sums[item.get_local_id()]) += partial_sum;
          // auto atomic_add = sycl::atomic_ref<std::complex<double>,
          // sycl::memory_order::relaxed,
          //                                   sycl::memory_scope::device,
          //                                   sycl::access::address_space::global_space>(
          //     reinterpret_cast<double&>(result[0]));
          // atomic_add.fetch_add(partial_sum);
        });

      d_streamId.wait();

      {
        auto host_sums = partial_sums.get_host_access();
        for (size_t i = 0; i < total_workitems; ++i)
          {
            localResult += host_sums[i];
          }
      }
#endif
      DEVICE_API_CHECK(event);
      MPI_Allreduce(&localResult,
                    result,
                    1,
                    dataTypes::mpi_type_id(result),
                    MPI_SUM,
                    mpi_communicator);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const unsigned int n,
      const double      *x,
      const unsigned int incx,
      const double      *y,
      const unsigned int incy,
      const MPI_Comm    &mpi_communicator,
      double            *result) const
    {
      double localResult = 0.0;
      *result            = 0.0;
#ifdef DFTFE_WITH_DEVICE_MKL
      double *localResult_device = sycl::malloc_device<double>(1, d_streamId);
      d_streamId.memcpy(localResult_device, &localResult, sizeof(double));
      dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::dot(
        d_streamId, n, x, incx, y, incy, localResult_device);
      d_streamId.wait();
      d_streamId.memcpy(&localResult, localResult_device, sizeof(double));
#else
      size_t total_workitems = (n / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
                               dftfe::utils::DEVICE_BLOCK_SIZE;

      sycl::buffer<double> x_buf(x, sycl::range<1>(n));
      sycl::buffer<double> y_buf(y, sycl::range<1>(n));
      sycl::buffer<double> sum_buf(&localResult, sycl::range<1>(1));

      dftfe::utils::deviceEvent_t event =
        d_streamId.submit([&](sycl::handler &cgh) {
          auto x_acc = x_buf.get_access<sycl::access::mode::read>(cgh);
          auto y_acc = y_buf.get_access<sycl::access::mode::read>(cgh);
          auto sum_acc =
            sum_buf.get_access<sycl::access::mode::read_write>(cgh);

          cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            double val_x = x_acc[i * incx], val_y = y_acc[i * incy];
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
              atomic_sum(sum_acc[0]);
            atomic_sum.fetch_add(val_x * val_y);
          });
        });
      d_streamId.wait();
      sycl::host_accessor sum_host_acc(sum_buf, sycl::read_only);
      localResult            = sum_host_acc[0];
#endif
      MPI_Allreduce(
        &localResult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xnrm2(
      const unsigned int n,
      const double      *x,
      const unsigned int incx,
      const MPI_Comm    &mpi_communicator,
      double            *result) const
    {
      double localResult = 0.0;
      *result            = 0.0;
#ifdef DFTFE_WITH_DEVICE_MKL
      double *localResult_device = sycl::malloc_device<double>(1, d_streamId);
      d_streamId.memcpy(localResult_device, &localResult, sizeof(double));
      dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::nrm2(
        d_streamId, n, x, incx, localResult_device);
      d_streamId.wait();
      d_streamId.memcpy(&localResult, localResult_device, sizeof(double));
#else
      size_t total_workitems = (n / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
                               dftfe::utils::DEVICE_BLOCK_SIZE;

      sycl::buffer<double> x_buf(x, sycl::range<1>(n));
      sycl::buffer<double> sum_buf(&localResult, sycl::range<1>(1));

      dftfe::utils::deviceEvent_t event =
        d_streamId.submit([&](sycl::handler &cgh) {
          auto x_acc = x_buf.get_access<sycl::access::mode::read>(cgh);
          auto sum_acc =
            sum_buf.get_access<sycl::access::mode::read_write>(cgh);

          cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            double val = x_acc[i * incx];
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
              atomic_sum(sum_acc[0]);
            atomic_sum.fetch_add(val * val);
          });
        });
      d_streamId.wait();
      sycl::host_accessor sum_host_acc(sum_buf, sycl::read_only);
      localResult = sum_host_acc[0];
#endif
      MPI_Allreduce(
        &localResult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
      *result = std::sqrt(*result);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xnrm2(
      const unsigned int          n,
      const std::complex<double> *x,
      const unsigned int          incx,
      const MPI_Comm             &mpi_communicator,
      double                     *result) const
    {
      double localresult = 0.0;
      *result            = 0.0;
#ifdef DFTFE_WITH_DEVICE_MKL
      double *localResult_device = sycl::malloc_device<double>(1, d_streamId);
      d_streamId.memcpy(localResult_device, &localresult, sizeof(double));
      dftfe::utils::deviceEvent_t event = oneapi::mkl::blas::column_major::nrm2(
        d_streamId, n, x, incx, localResult_device);
      d_streamId.wait();
      d_streamId.memcpy(&localresult, localResult_device, sizeof(double));
#else
      size_t total_workitems =
        ((n / incx + 1) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          unsigned int global_id    = ind.get_global_id(0);
          unsigned int n_workgroups = ind.get_group_range(0);
          unsigned int n_workitems  = ind.get_local_range(0);
          for (unsigned int index = global_id; index < n;
               index += n_workgroups * n_workitems)
            {
              if (incx * global_id >= n)
                {
                  break;
                }
              // auto atomic_add_real = sycl::atomic_ref<double,
              // sycl::memory_order::relaxed,
              //                                     sycl::memory_scope::device,
              //                                     sycl::access::address_space::global_space>
              //                                     (reinterpret_cast<double*>(&result)[0]);
              // atomic_add_real +=
              // x[incx*global_id].real()*x[incx*global_id].real();

              // auto atomic_add_imag = sycl::atomic_ref<double,
              // sycl::memory_order::relaxed,
              //                                     sycl::memory_scope::device,
              //                                     sycl::access::address_space::global_space>
              //                                     (reinterpret_cast<double*>(&result)[1]);
              // atomic_add_imag +=
              // x[incx*global_id].imag()*x[incx*global_id].imag();
            }
        });
#endif
      DEVICE_API_CHECK(event);
      MPI_Allreduce(
        &localresult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
      *result = std::sqrt(*result);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      ValueType1        *x,
      const ValueType2   alpha,
      const unsigned int n) const
    {
      const unsigned int incx = 1;
#ifdef DFTFE_WITH_DEVICE_MKL
      dftfe::utils::deviceEvent_t event =
        oneapi::mkl::blas::column_major::scal(d_streamId, n, alpha, x, incx);
#else
      size_t total_workitems = (n / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
                               dftfe::utils::DEVICE_BLOCK_SIZE;
      // auto alpha_local = alpha;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          size_t index = ind.get_global_id(0);
          if (index < n)
            x[index] = x[index] * alpha;
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char         transA,
      const char         transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const double      *alpha,
      const double      *A,
      const unsigned int lda,
      long long int      strideA,
      const double      *B,
      const unsigned int ldb,
      long long int      strideB,
      const double      *beta,
      double            *C,
      const unsigned int ldc,
      long long int      strideC,
      const int          batchCount) const
    {
#ifdef DFTFE_WITH_DEVICE_MKL
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

      dftfe::utils::deviceEvent_t event =
        oneapi::mkl::blas::column_major::gemm_batch(d_streamId,
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
      DEVICE_API_CHECK(event);
#else
      for (int iBatch = 0; iBatch < batchCount; iBatch++)
        {
          xgemm(transA,
                transB,
                m,
                n,
                k,
                alpha,
                A + iBatch * strideA,
                lda,
                B + iBatch * strideB,
                ldb,
                beta,
                C + iBatch * strideC,
                ldc);
        }
#endif
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char                  transA,
      const char                  transB,
      const unsigned int          m,
      const unsigned int          n,
      const unsigned int          k,
      const std::complex<double> *alpha,
      const std::complex<double> *A,
      const unsigned int          lda,
      long long int               strideA,
      const std::complex<double> *B,
      const unsigned int          ldb,
      long long int               strideB,
      const std::complex<double> *beta,
      std::complex<double>       *C,
      const unsigned int          ldc,
      long long int               strideC,
      const int                   batchCount) const
    {
#ifdef DFTFE_WITH_DEVICE_MKL
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
      dftfe::utils::deviceEvent_t event =
        oneapi::mkl::blas::column_major::gemm_batch(d_streamId,
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
      DEVICE_API_CHECK(event);
#else
      for (int iBatch = 0; iBatch < batchCount; iBatch++)
        {
          xgemm(transA,
                transB,
                m,
                n,
                k,
                alpha,
                A + iBatch * strideA,
                lda,
                B + iBatch * strideB,
                ldb,
                beta,
                C + iBatch * strideC,
                ldc);
        }
#endif
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char                 transA,
      const char                 transB,
      const unsigned int         m,
      const unsigned int         n,
      const unsigned int         k,
      const std::complex<float> *alpha,
      const std::complex<float> *A,
      const unsigned int         lda,
      long long int              strideA,
      const std::complex<float> *B,
      const unsigned int         ldb,
      long long int              strideB,
      const std::complex<float> *beta,
      std::complex<float>       *C,
      const unsigned int         ldc,
      long long int              strideC,
      const int                  batchCount) const
    {
#ifdef DFTFE_WITH_DEVICE_MKL
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

      dftfe::utils::deviceEvent_t event =
        oneapi::mkl::blas::column_major::gemm_batch(d_streamId,
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
      DEVICE_API_CHECK(event);
#else
      for (int iBatch = 0; iBatch < batchCount; iBatch++)
        {
          xgemm(transA,
                transB,
                m,
                n,
                k,
                alpha,
                A + iBatch * strideA,
                lda,
                B + iBatch * strideB,
                ldb,
                beta,
                C + iBatch * strideC,
                ldc);
        }
#endif
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char         transA,
      const char         transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const float       *alpha,
      const float       *A,
      const unsigned int lda,
      long long int      strideA,
      const float       *B,
      const unsigned int ldb,
      long long int      strideB,
      const float       *beta,
      float             *C,
      const unsigned int ldc,
      long long int      strideC,
      const int          batchCount) const
    {
#ifdef DFTFE_WITH_DEVICE_MKL
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

      dftfe::utils::deviceEvent_t event =
        oneapi::mkl::blas::column_major::gemm_batch(d_streamId,
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
      DEVICE_API_CHECK(event);
#else
      for (int iBatch = 0; iBatch < batchCount; iBatch++)
        {
          xgemm(transA,
                transB,
                m,
                n,
                k,
                alpha,
                A + iBatch * strideA,
                lda,
                B + iBatch * strideB,
                ldb,
                beta,
                C + iBatch * strideC,
                ldc);
        }
#endif
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char                  transA,
      const char                  transB,
      const unsigned int          m,
      const unsigned int          n,
      const unsigned int          k,
      const std::complex<double> *alpha,
      const std::complex<double> *A[],
      const unsigned int          lda,
      const std::complex<double> *B[],
      const unsigned int          ldb,
      const std::complex<double> *beta,
      std::complex<double>       *C[],
      const unsigned int          ldc,
      const int                   batchCount) const
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

#ifdef DFTFE_WITH_DEVICE_MKL
      const long                  group_size = 1;
      const int                   m_local    = int(m);
      const int                   n_local    = int(n);
      const int                   k_local    = int(k);
      const int                   lda_local  = int(lda);
      const int                   ldb_local  = int(ldb);
      const int                   ldc_local  = int(ldc);
      dftfe::utils::deviceEvent_t event =
        oneapi::mkl::blas::column_major::gemm_batch(d_streamId,
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
                                                    &batchCount);
#else
      auto                        alpha_local = alpha[0];
      auto                        beta_local  = beta[0];
      dftfe::utils::deviceEvent_t event       = d_streamId.parallel_for(
        sycl::nd_range<1>(batchCount, 1), [=](sycl::nd_item<1> item) {
          size_t batch = item.get_global_id(0);

          if (batch < batchCount)
            {
              for (size_t col = 0; col < n; ++col)
                {
                  for (size_t row = 0; row < m; ++row)
                    {
                      C[batch][col * ldc + row] =
                        beta_local * C[batch][col * ldc + row];

                      for (size_t i = 0; i < k; ++i)
                        {
                          C[batch][col * ldc + row] +=
                            alpha_local *
                            ((transa == dftfe::utils::DEVICEBLAS_OP_N) ?
                                     A[batch][i * lda + row] :
                                     A[batch][row * lda + i]) *
                            ((transb == dftfe::utils::DEVICEBLAS_OP_N) ?
                                     B[batch][col * ldb + i] :
                                     B[batch][i * ldb + col]);
                        }
                    }
                }
            }
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char         transA,
      const char         transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const double      *alpha,
      const double      *A[],
      const unsigned int lda,
      const double      *B[],
      const unsigned int ldb,
      const double      *beta,
      double            *C[],
      const unsigned int ldc,
      const int          batchCount) const
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

#ifdef DFTFE_WITH_DEVICE_MKL
      const long                  group_size = 1;
      const int                   m_local    = int(m);
      const int                   n_local    = int(n);
      const int                   k_local    = int(k);
      const int                   lda_local  = int(lda);
      const int                   ldb_local  = int(ldb);
      const int                   ldc_local  = int(ldc);
      dftfe::utils::deviceEvent_t event =
        oneapi::mkl::blas::column_major::gemm_batch(d_streamId,
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
                                                    &batchCount);
#else
      auto                        alpha_local = alpha[0];
      auto                        beta_local  = beta[0];
      dftfe::utils::deviceEvent_t event       = d_streamId.parallel_for(
        sycl::nd_range<1>(batchCount, 1), [=](sycl::nd_item<1> item) {
          size_t batch = item.get_global_id(0);

          if (batch < batchCount)
            {
              for (size_t col = 0; col < n; ++col)
                {
                  for (size_t row = 0; row < m; ++row)
                    {
                      C[batch][col * ldc + row] =
                        beta_local * C[batch][col * ldc + row];

                      for (size_t i = 0; i < k; ++i)
                        {
                          C[batch][col * ldc + row] +=
                            alpha_local *
                            ((transa == dftfe::utils::DEVICEBLAS_OP_N) ?
                                     A[batch][i * lda + row] :
                                     A[batch][row * lda + i]) *
                            ((transb == dftfe::utils::DEVICEBLAS_OP_N) ?
                                     B[batch][col * ldb + i] :
                                     B[batch][i * ldb + col]);
                        }
                    }
                }
            }
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char                 transA,
      const char                 transB,
      const unsigned int         m,
      const unsigned int         n,
      const unsigned int         k,
      const std::complex<float> *alpha,
      const std::complex<float> *A[],
      const unsigned int         lda,
      const std::complex<float> *B[],
      const unsigned int         ldb,
      const std::complex<float> *beta,
      std::complex<float>       *C[],
      const unsigned int         ldc,
      const int                  batchCount) const
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

#ifdef DFTFE_WITH_DEVICE_MKL
      const long                  group_size = 1;
      const int                   m_local    = int(m);
      const int                   n_local    = int(n);
      const int                   k_local    = int(k);
      const int                   lda_local  = int(lda);
      const int                   ldb_local  = int(ldb);
      const int                   ldc_local  = int(ldc);
      dftfe::utils::deviceEvent_t event =
        oneapi::mkl::blas::column_major::gemm_batch(d_streamId,
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
                                                    &batchCount);
#else
      auto                        alpha_local = alpha[0];
      auto                        beta_local  = beta[0];
      dftfe::utils::deviceEvent_t event       = d_streamId.parallel_for(
        sycl::nd_range<1>(batchCount, 1), [=](sycl::nd_item<1> item) {
          size_t batch = item.get_global_id(0);

          if (batch < batchCount)
            {
              for (size_t col = 0; col < n; ++col)
                {
                  for (size_t row = 0; row < m; ++row)
                    {
                      C[batch][col * ldc + row] =
                        beta_local * C[batch][col * ldc + row];

                      for (size_t i = 0; i < k; ++i)
                        {
                          C[batch][col * ldc + row] +=
                            alpha_local *
                            ((transa == dftfe::utils::DEVICEBLAS_OP_N) ?
                                     A[batch][i * lda + row] :
                                     A[batch][row * lda + i]) *
                            ((transb == dftfe::utils::DEVICEBLAS_OP_N) ?
                                     B[batch][col * ldb + i] :
                                     B[batch][i * ldb + col]);
                        }
                    }
                }
            }
        });
#endif
      DEVICE_API_CHECK(event);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char         transA,
      const char         transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const float       *alpha,
      const float       *A[],
      const unsigned int lda,
      const float       *B[],
      const unsigned int ldb,
      const float       *beta,
      float             *C[],
      const unsigned int ldc,
      const int          batchCount) const
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

#ifdef DFTFE_WITH_DEVICE_MKL
      const long                  group_size = 1;
      const int                   m_local    = int(m);
      const int                   n_local    = int(n);
      const int                   k_local    = int(k);
      const int                   lda_local  = int(lda);
      const int                   ldb_local  = int(ldb);
      const int                   ldc_local  = int(ldc);
      dftfe::utils::deviceEvent_t event =
        oneapi::mkl::blas::column_major::gemm_batch(d_streamId,
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
                                                    &batchCount);
#else
      auto                        alpha_local = alpha[0];
      auto                        beta_local  = beta[0];
      dftfe::utils::deviceEvent_t event       = d_streamId.parallel_for(
        sycl::nd_range<1>(batchCount, 1), [=](sycl::nd_item<1> item) {
          size_t batch = item.get_global_id(0);

          if (batch < batchCount)
            {
              for (size_t col = 0; col < n; ++col)
                {
                  for (size_t row = 0; row < m; ++row)
                    {
                      C[batch][col * ldc + row] =
                        beta_local * C[batch][col * ldc + row];

                      for (size_t i = 0; i < k; ++i)
                        {
                          C[batch][col * ldc + row] +=
                            alpha_local *
                            ((transa == dftfe::utils::DEVICEBLAS_OP_N) ?
                                     A[batch][i * lda + row] :
                                     A[batch][row * lda + i]) *
                            ((transb == dftfe::utils::DEVICEBLAS_OP_N) ?
                                     B[batch][col * ldb + i] :
                                     B[batch][i * ldb + col]);
                        }
                    }
                }
            }
        });
#endif
      DEVICE_API_CHECK(event);
    }

    // BlasWrapperDeviceKernels.sycl.cpp kernels used
    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpby(
      const unsigned int n,
      const ValueType2   alpha,
      const ValueType1  *x,
      const ValueType2   beta,
      ValueType1        *y) const
    {
      unsigned int total_workitems = (n / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
                                     dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          dftfe::axpbyDeviceKernel(ind, n, x, y, alpha, beta);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType0,
              typename ValueType1,
              typename ValueType2,
              typename ValueType3,
              typename ValueType4>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::ApaBD(
      const unsigned int m,
      const unsigned int n,
      const ValueType0   alpha,
      const ValueType1  *A,
      const ValueType2  *B,
      const ValueType3  *D,
      ValueType4        *C) const
    {
      unsigned int total_workitems =
        ((n * m / dftfe::utils::DEVICE_BLOCK_SIZE) + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          dftfe::ApaBDDeviceKernel(ind, m, n, alpha, A, B, D, C);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::copyComplexArrToRealArrs(
      const unsigned int      size,
      const ValueTypeComplex *complexArr,
      ValueTypeReal          *realArr,
      ValueTypeReal          *imagArr)
    {
      unsigned int total_workitems =
        (size / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          dftfe::copyComplexArrToRealArrsDeviceKernel(
            ind, size, complexArr, realArr, imagArr);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::copyRealArrsToComplexArr(
      const unsigned int   size,
      const ValueTypeReal *realArr,
      const ValueTypeReal *imagArr,
      ValueTypeComplex    *complexArr)
    {
      unsigned int total_workitems =
        (size / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          dftfe::copyRealArrsToComplexArrDeviceKernel(
            ind, size, realArr, imagArr, complexArr);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(const unsigned int size,
                                       const ValueType1  *valueType1Arr,
                                       ValueType2        *valueType2Arr)
    {
      unsigned int total_workitems =
        (size / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          dftfe::copyValueType1ArrToValueType2ArrDeviceKernel(ind,
                                                              size,
                                                              valueType1Arr,
                                                              valueType2Arr);
        });
      DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const ValueType1   *copyFromVec,
      ValueType2         *copyToVecBlock,
      const unsigned int *copyFromVecStartingContiguousBlockIds)
    {
      unsigned int total_workitems =
        ((numContiguousBlocks * contiguousBlockSize) /
           dftfe::utils::DEVICE_BLOCK_SIZE +
         1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;

      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          dftfe::stridedCopyToBlockDeviceKernel(
            ind,
            contiguousBlockSize,
            numContiguousBlocks,
            copyFromVec,
            copyToVecBlock,
            copyFromVecStartingContiguousBlockIds);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const unsigned int  startingVecId,
      const ValueType1   *copyFromVec,
      ValueType2         *copyToVecBlock,
      const unsigned int *copyFromVecStartingContiguousBlockIds)
    {
      unsigned int total_workitems =
        ((contiguousBlockSize * numContiguousBlocks) /
           dftfe::utils::DEVICE_BLOCK_SIZE +
         1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;

      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          dftfe::stridedCopyToBlockDeviceKernel(
            ind,
            contiguousBlockSize,
            numContiguousBlocks,
            startingVecId,
            dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
            dftfe::utils::makeDataTypeDeviceCompatible(copyToVecBlock),
            copyFromVecStartingContiguousBlockIds);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const ValueType    *addFromVec,
      ValueType          *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds) const
    {
      unsigned int total_workitems =
        ((numContiguousBlocks * contiguousBlockSize) /
           dftfe::utils::DEVICE_BLOCK_SIZE +
         1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          dftfe::axpyStridedBlockAtomicAddDeviceKernel(
            ind,
            contiguousBlockSize,
            numContiguousBlocks,
            addFromVec,
            addToVec,
            addToVecStartingContiguousBlockIds);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2, typename ValueType3>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const ValueType1    a,
      const ValueType1   *s,
      const ValueType2   *addFromVec,
      ValueType3         *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds) const
    {
      unsigned int total_workitems =
        ((numContiguousBlocks * contiguousBlockSize) /
           dftfe::utils::DEVICE_BLOCK_SIZE +
         1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          dftfe::axpyStridedBlockAtomicAddDeviceKernel(
            ind,
            contiguousBlockSize,
            numContiguousBlocks,
            a,
            s,
            addFromVec,
            addToVec,
            addToVecStartingContiguousBlockIds);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2, typename ValueType3>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const ValueType1    a,
      const ValueType2   *addFromVec,
      ValueType3         *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds) const
    {
      unsigned int total_workitems =
        ((contiguousBlockSize * numContiguousBlocks) /
           dftfe::utils::DEVICE_BLOCK_SIZE +
         1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          dftfe::axpyStridedBlockAtomicAddDeviceKernel(
            ind,
            contiguousBlockSize,
            numContiguousBlocks,
            dftfe::utils::makeDataTypeDeviceCompatible(a),
            dftfe::utils::makeDataTypeDeviceCompatible(addFromVec),
            dftfe::utils::makeDataTypeDeviceCompatible(addToVec),
            addToVecStartingContiguousBlockIds);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const ValueType1   *copyFromVecBlock,
      ValueType2         *copyToVec,
      const unsigned int *copyFromVecStartingContiguousBlockIds)
    {
      unsigned int total_workitems =
        ((numContiguousBlocks * contiguousBlockSize) /
           dftfe::utils::DEVICE_BLOCK_SIZE +
         1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          dftfe::stridedCopyFromBlockDeviceKernel(
            ind,
            contiguousBlockSize,
            numContiguousBlocks,
            copyFromVecBlock,
            copyToVec,
            copyFromVecStartingContiguousBlockIds);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      stridedCopyToBlockConstantStride(const unsigned int blockSizeTo,
                                       const unsigned int blockSizeFrom,
                                       const unsigned int numBlocks,
                                       const unsigned int startingId,
                                       const ValueType1  *copyFromVec,
                                       ValueType2        *copyToVec) const
    {
      unsigned int total_workitems =
        ((numBlocks * blockSizeTo) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          dftfe::stridedCopyToBlockConstantStrideDeviceKernel(ind,
                                                              blockSizeTo,
                                                              blockSizeFrom,
                                                              numBlocks,
                                                              startingId,
                                                              copyFromVec,
                                                              copyToVec);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      stridedCopyFromBlockConstantStride(const unsigned int blockSizeTo,
                                         const unsigned int blockSizeFrom,
                                         const unsigned int numBlocks,
                                         const unsigned int startingId,
                                         const ValueType1  *copyFromVec,
                                         ValueType2        *copyToVec)
    {
      unsigned int total_workitems =
        ((numBlocks * blockSizeFrom) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          dftfe::stridedCopyFromBlockConstantStrideDeviceKernel(ind,
                                                                blockSizeTo,
                                                                blockSizeFrom,
                                                                numBlocks,
                                                                startingId,
                                                                copyFromVec,
                                                                copyToVec);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
      const unsigned int blockSize,
      const unsigned int strideTo,
      const unsigned int strideFrom,
      const unsigned int numBlocks,
      const unsigned int startingToId,
      const unsigned int startingFromId,
      const ValueType1  *copyFromVec,
      ValueType2        *copyToVec)
    {
      unsigned int total_workitems =
        ((numBlocks * blockSize) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          dftfe::stridedCopyConstantStrideDeviceKernel(ind,
                                                       blockSize,
                                                       strideTo,
                                                       strideFrom,
                                                       numBlocks,
                                                       startingToId,
                                                       startingFromId,
                                                       copyFromVec,
                                                       copyToVec);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleCopy(
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const ValueType1    a,
      const ValueType1   *s,
      const ValueType2   *copyFromVec,
      ValueType2         *copyToVecBlock,
      const unsigned int *copyFromVecStartingContiguousBlockIds)
    {
      unsigned int total_workitems =
        ((numContiguousBlocks * contiguousBlockSize) /
           dftfe::utils::DEVICE_BLOCK_SIZE +
         1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          dftfe::stridedCopyToBlockScaleDeviceKernel(
            ind,
            contiguousBlockSize,
            numContiguousBlocks,
            a,
            s,
            copyFromVec,
            copyToVecBlock,
            copyFromVecStartingContiguousBlockIds);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const unsigned int contiguousBlockSize,
      const unsigned int numContiguousBlocks,
      const ValueType1   a,
      const ValueType1  *s,
      ValueType2        *x)
    {
      unsigned int total_workitems =
        ((numContiguousBlocks * contiguousBlockSize) /
           dftfe::utils::DEVICE_BLOCK_SIZE +
         1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          dftfe::stridedBlockScaleDeviceKernel(
            ind, contiguousBlockSize, numContiguousBlocks, a, s, x);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2ArrDeviceCall(
        const unsigned int                 size,
        const ValueType1                  *valueType1Arr,
        ValueType2                        *valueType2Arr,
        const dftfe::utils::deviceStream_t streamId)
    {
      unsigned int total_workitems =
        (size / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      sycl::queue                 stream{sycl::gpu_selector_v};
      dftfe::utils::deviceEvent_t event =
        stream.parallel_for(sycl::nd_range<1>(total_workitems,
                                              dftfe::utils::DEVICE_BLOCK_SIZE),
                            [=](sycl::nd_item<1> ind) {
                              copyValueType1ArrToValueType2ArrDeviceKernel(
                                ind, size, valueType1Arr, valueType2Arr);
                            });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::hadamardProduct(
      const unsigned int m,
      const ValueType   *X,
      const ValueType   *Y,
      ValueType         *output) const
    {
      unsigned int total_workitems =
        ((m) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          hadamardProductKernel(ind, m, X, Y, output);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::hadamardProductWithConj(
      const unsigned int m,
      const ValueType   *X,
      const ValueType   *Y,
      ValueType         *output) const
    {
      unsigned int total_workitems =
        ((m) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          hadamardProductWithConjKernel(ind, m, X, Y, output);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::MultiVectorXDot(
      const unsigned int contiguousBlockSize,
      const unsigned int numContiguousBlocks,
      const ValueType   *X,
      const ValueType   *Y,
      const ValueType   *onesVec,
      ValueType         *tempVector,
      ValueType         *tempResults,
      ValueType         *result) const
    {
      hadamardProductWithConj(contiguousBlockSize * numContiguousBlocks,
                              X,
                              Y,
                              tempVector);

      ValueType    alpha  = 1.0;
      ValueType    beta   = 0.0;
      unsigned int numVec = 1;
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
      const unsigned int contiguousBlockSize,
      const unsigned int numContiguousBlocks,
      const ValueType   *X,
      const ValueType   *Y,
      const ValueType   *onesVec,
      ValueType         *tempVector,
      ValueType         *tempResults,
      const MPI_Comm    &mpi_communicator,
      ValueType         *result) const

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

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::addVecOverContinuousIndex(
      const unsigned int numContiguousBlocks,
      const unsigned int contiguousBlockSize,
      const ValueType   *input1,
      const ValueType   *input2,
      ValueType         *output)
    {
      unsigned int total_workitems =
        ((numContiguousBlocks) / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          addVecOverContinuousIndexKernel(ind,
                                          numContiguousBlocks,
                                          contiguousBlockSize,
                                          input1,
                                          input2,
                                          output);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleColumnWise(
      const unsigned int contiguousBlockSize,
      const unsigned int numContiguousBlocks,
      const ValueType   *beta,
      ValueType         *x)
    {
      unsigned int total_workitems =
        ((contiguousBlockSize * numContiguousBlocks) /
           dftfe::utils::DEVICE_BLOCK_SIZE +
         1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          stridedBlockScaleColumnWiseKernel(
            ind, contiguousBlockSize, numContiguousBlocks, beta, x);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      stridedBlockScaleAndAddColumnWise(const unsigned int contiguousBlockSize,
                                        const unsigned int numContiguousBlocks,
                                        const ValueType   *x,
                                        const ValueType   *beta,
                                        ValueType         *y)
    {
      unsigned int total_workitems =
        ((contiguousBlockSize * numContiguousBlocks) /
           dftfe::utils::DEVICE_BLOCK_SIZE +
         1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          stridedBlockScaleAndAddColumnWiseKernel(
            ind, contiguousBlockSize, numContiguousBlocks, x, beta, y);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      stridedBlockScaleAndAddTwoVecColumnWise(
        const unsigned int contiguousBlockSize,
        const unsigned int numContiguousBlocks,
        const ValueType   *x,
        const ValueType   *alpha,
        const ValueType   *y,
        const ValueType   *beta,
        ValueType         *z)
    {
      unsigned int total_workitems =
        ((contiguousBlockSize * numContiguousBlocks) /
           dftfe::utils::DEVICE_BLOCK_SIZE +
         1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          stridedBlockScaleAndAddTwoVecColumnWiseKernel(ind,
                                                        contiguousBlockSize,
                                                        numContiguousBlocks,
                                                        x,
                                                        alpha,
                                                        y,
                                                        beta,
                                                        z);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockAxpy(
      const unsigned int contiguousBlockSize,
      const unsigned int numContiguousBlocks,
      const ValueType1  *addFromVec,
      const ValueType2  *scalingVector,
      const ValueType2   a,
      ValueType1        *addToVec) const
    {
      unsigned int total_workitems =
        ((contiguousBlockSize * numContiguousBlocks) /
           dftfe::utils::DEVICE_BLOCK_SIZE +
         1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          stridedBlockAxpyDeviceKernel(
            ind,
            contiguousBlockSize,
            numContiguousBlocks,
            dftfe::utils::makeDataTypeDeviceCompatible(a),
            dftfe::utils::makeDataTypeDeviceCompatible(scalingVector),
            dftfe::utils::makeDataTypeDeviceCompatible(addFromVec),
            dftfe::utils::makeDataTypeDeviceCompatible(addToVec));
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockAxpBy(
      const unsigned int contiguousBlockSize,
      const unsigned int numContiguousBlocks,
      const ValueType1  *addFromVec,
      const ValueType2  *scalingVector,
      const ValueType2   a,
      const ValueType2   b,
      ValueType1        *addToVec) const
    {
      unsigned int total_workitems =
        ((contiguousBlockSize * numContiguousBlocks) /
           dftfe::utils::DEVICE_BLOCK_SIZE +
         1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          stridedBlockAxpByDeviceKernel(
            ind,
            contiguousBlockSize,
            numContiguousBlocks,
            dftfe::utils::makeDataTypeDeviceCompatible(a),
            dftfe::utils::makeDataTypeDeviceCompatible(b),
            dftfe::utils::makeDataTypeDeviceCompatible(scalingVector),
            dftfe::utils::makeDataTypeDeviceCompatible(addFromVec),
            dftfe::utils::makeDataTypeDeviceCompatible(addToVec));
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::rightDiagonalScale(
      const unsigned int numberofVectors,
      const unsigned int sizeOfVector,
      ValueType1        *X,
      ValueType2        *D)
    {
      unsigned int total_workitems =
        ((numberofVectors + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
         dftfe::utils::DEVICE_BLOCK_SIZE * sizeOfVector) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          computeRightDiagonalScaleKernel(
            ind,
            dftfe::utils::makeDataTypeDeviceCompatible(D),
            dftfe::utils::makeDataTypeDeviceCompatible(X),
            numberofVectors,
            sizeOfVector);
        });

      DEVICE_API_CHECK(event);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      copyBlockDiagonalValueType1OffDiagonalValueType2FromValueType1Arr(
        const unsigned int B,
        const unsigned int DRem,
        const unsigned int D,
        const ValueType1  *valueType1SrcArray,
        ValueType1        *valueType1DstArray,
        ValueType2        *valueType2DstArray)
    {
      const unsigned int size = D * B;
      unsigned int       total_workitems =
        (size / dftfe::utils::DEVICE_BLOCK_SIZE + 1) *
        dftfe::utils::DEVICE_BLOCK_SIZE;
      dftfe::utils::deviceEvent_t event = d_streamId.parallel_for(
        sycl::nd_range<1>(total_workitems, dftfe::utils::DEVICE_BLOCK_SIZE),
        [=](sycl::nd_item<1> ind) {
          copyBlockDiagonalValueType1OffDiagonalValueType2FromValueType1ArrDeviceKernel(
            ind,
            B,
            DRem,
            D,
            dftfe::utils::makeDataTypeDeviceCompatible(valueType1SrcArray),
            dftfe::utils::makeDataTypeDeviceCompatible(valueType1DstArray),
            dftfe::utils::makeDataTypeDeviceCompatible(valueType2DstArray));
        });

      DEVICE_API_CHECK(event);
    }

#include "./BLASWrapperDevice.inst.cc"
  } // namespace linearAlgebra
} // namespace dftfe
