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
#include <hipblas.h>
#ifdef DFTFE_WITH_DEVICE_AMD
#  include <rocblas.h>
#endif
#include "BLASWrapperDeviceKernels.cc"
namespace dftfe
{
  namespace utils
  {
    inline double
    makeDataTypeHipBlasCompatible(double a)
    {
      return a;
    }

    inline float
    makeDataTypeHipBlasCompatible(float a)
    {
      return a;
    }

    inline float *
    makeDataTypeHipBlasCompatible(float *a)
    {
      return reinterpret_cast<float *>(a);
    }

    inline const float *
    makeDataTypeHipBlasCompatible(const float *a)
    {
      return reinterpret_cast<const float *>(a);
    }

    inline double *
    makeDataTypeHipBlasCompatible(double *a)
    {
      return reinterpret_cast<double *>(a);
    }

    inline const double *
    makeDataTypeHipBlasCompatible(const double *a)
    {
      return reinterpret_cast<const double *>(a);
    }


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
      status = create();
      status = setStream(NULL);
    }

    dftfe::utils::deviceBlasHandle_t &
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::getDeviceBlasHandle()
    {
      return d_deviceBlasHandle;
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xcopy(
      const unsigned int          n,
      const std::complex<double> *x,
      const unsigned int          incx,
      std::complex<double> *      y,
      const unsigned int          incy) const
    {
      dftfe::utils::deviceBlasStatus_t status =
        hipblasZcopy(d_deviceBlasHandle,
                     n,
                     dftfe::utils::makeDataTypeHipBlasCompatible(x),
                     incx,
                     dftfe::utils::makeDataTypeHipBlasCompatible(y),
                     incy);
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xcopy(
      const unsigned int         n,
      const std::complex<float> *x,
      const unsigned int         incx,
      std::complex<float> *      y,
      const unsigned int         incy) const
    {
      dftfe::utils::deviceBlasStatus_t status =
        hipblasCcopy(d_deviceBlasHandle,
                     n,
                     dftfe::utils::makeDataTypeHipBlasCompatible(x),
                     incx,
                     dftfe::utils::makeDataTypeHipBlasCompatible(y),
                     incy);
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xcopy(
      const unsigned int n,
      const double *     x,
      const unsigned int incx,
      double *           y,
      const unsigned int incy) const
    {
      dftfe::utils::deviceBlasStatus_t status =
        hipblasDcopy(d_deviceBlasHandle, n, x, incx, y, incy);
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xcopy(
      const unsigned int n,
      const float *      x,
      const unsigned int incx,
      float *            y,
      const unsigned int incy) const
    {
      dftfe::utils::deviceBlasStatus_t status =
        hipblasScopy(d_deviceBlasHandle, n, x, incx, y, incy);
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
      const char         transA,
      const char         transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const float *      alpha,
      const float *      A,
      const unsigned int lda,
      const float *      B,
      const unsigned int ldb,
      const float *      beta,
      float *            C,
      const unsigned int ldc) const
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

      dftfe::utils::deviceBlasStatus_t status = hipblasSgemm(d_deviceBlasHandle,
                                                             transa,
                                                             transb,
                                                             int(m),
                                                             int(n),
                                                             int(k),
                                                             alpha,
                                                             A,
                                                             int(lda),
                                                             B,
                                                             int(ldb),
                                                             beta,
                                                             C,
                                                             int(ldc));
      DEVICEBLAS_API_CHECK(status);
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
      std::complex<float> *      C,
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

      dftfe::utils::deviceBlasStatus_t status =
        hipblasCgemm(d_deviceBlasHandle,
                     transa,
                     transb,
                     int(m),
                     int(n),
                     int(k),
                     dftfe::utils::makeDataTypeHipBlasCompatible(alpha),
                     dftfe::utils::makeDataTypeHipBlasCompatible(A),
                     int(lda),
                     dftfe::utils::makeDataTypeHipBlasCompatible(B),
                     int(ldb),
                     dftfe::utils::makeDataTypeHipBlasCompatible(beta),
                     dftfe::utils::makeDataTypeHipBlasCompatible(C),
                     int(ldc));
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemm(
      const char         transA,
      const char         transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const double *     alpha,
      const double *     A,
      const unsigned int lda,
      const double *     B,
      const unsigned int ldb,
      const double *     beta,
      double *           C,
      const unsigned int ldc) const
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
      dftfe::utils::deviceBlasStatus_t status = hipblasDgemm(d_deviceBlasHandle,
                                                             transa,
                                                             transb,
                                                             int(m),
                                                             int(n),
                                                             int(k),
                                                             alpha,
                                                             A,
                                                             int(lda),
                                                             B,
                                                             int(ldb),
                                                             beta,
                                                             C,
                                                             int(ldc));
      DEVICEBLAS_API_CHECK(status);
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
      std::complex<double> *      C,
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


      dftfe::utils::deviceBlasStatus_t status =
        hipblasZgemm(d_deviceBlasHandle,
                     transa,
                     transb,
                     int(m),
                     int(n),
                     int(k),
                     dftfe::utils::makeDataTypeHipBlasCompatible(alpha),
                     dftfe::utils::makeDataTypeHipBlasCompatible(A),
                     int(lda),
                     dftfe::utils::makeDataTypeHipBlasCompatible(B),
                     int(ldb),
                     dftfe::utils::makeDataTypeHipBlasCompatible(beta),
                     dftfe::utils::makeDataTypeHipBlasCompatible(C),
                     int(ldc));
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemv(
      const char         transA,
      const unsigned int m,
      const unsigned int n,
      const double *     alpha,
      const double *     A,
      const unsigned int lda,
      const double *     x,
      const unsigned int incx,
      const double *     beta,
      double *           y,
      const unsigned int incy) const
    {
      dftfe::utils::deviceBlasOperation_t transa;
      if (transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else
        {
          // Assert Statement
        }
      dftfe::utils::deviceBlasStatus_t status = hipblasDgemv(d_deviceBlasHandle,
                                                             transa,
                                                             int(m),
                                                             int(n),
                                                             alpha,
                                                             A,
                                                             int(lda),
                                                             x,
                                                             int(incx),
                                                             beta,
                                                             y,
                                                             int(incy));
      DEVICEBLAS_API_CHECK(status);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemv(
      const char         transA,
      const unsigned int m,
      const unsigned int n,
      const float *      alpha,
      const float *      A,
      const unsigned int lda,
      const float *      x,
      const unsigned int incx,
      const float *      beta,
      float *            y,
      const unsigned int incy) const
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

      dftfe::utils::deviceBlasStatus_t status = hipblasSgemv(d_deviceBlasHandle,
                                                             transa,
                                                             int(m),
                                                             int(n),
                                                             alpha,
                                                             A,
                                                             int(lda),
                                                             x,
                                                             int(incx),
                                                             beta,
                                                             y,
                                                             int(incy));
      DEVICEBLAS_API_CHECK(status);
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
      std::complex<double> *      y,
      const unsigned int          incy) const
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

      dftfe::utils::deviceBlasStatus_t status =
        hipblasZgemv(d_deviceBlasHandle,
                     transa,
                     int(m),
                     int(n),
                     dftfe::utils::makeDataTypeHipBlasCompatible(alpha),
                     dftfe::utils::makeDataTypeHipBlasCompatible(A),
                     int(lda),
                     dftfe::utils::makeDataTypeHipBlasCompatible(x),
                     int(incx),
                     dftfe::utils::makeDataTypeHipBlasCompatible(beta),
                     dftfe::utils::makeDataTypeHipBlasCompatible(y),
                     int(incy));
      DEVICEBLAS_API_CHECK(status);
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
      std::complex<float> *      y,
      const unsigned int         incy) const
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

      dftfe::utils::deviceBlasStatus_t status =
        hipblasCgemv(d_deviceBlasHandle,
                     transa,
                     int(m),
                     int(n),
                     dftfe::utils::makeDataTypeHipBlasCompatible(alpha),
                     dftfe::utils::makeDataTypeHipBlasCompatible(A),
                     int(lda),
                     dftfe::utils::makeDataTypeHipBlasCompatible(x),
                     int(incx),
                     dftfe::utils::makeDataTypeHipBlasCompatible(beta),
                     dftfe::utils::makeDataTypeHipBlasCompatible(y),
                     int(incy));
      DEVICEBLAS_API_CHECK(status);
    }


    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::create()
    {
      dftfe::utils::deviceBlasStatus_t status =
        hipblasCreate(&d_deviceBlasHandle);
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
      dftfe::utils::deviceStream_t streamId)
    {
      d_streamId = streamId;
      dftfe::utils::deviceBlasStatus_t status =
        hipblasSetStream(d_deviceBlasHandle, d_streamId);
      DEVICEBLAS_API_CHECK(status);
      return status;
    }



    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xaxpy(
      const unsigned int n,
      const double *     alpha,
      const double *     x,
      const unsigned int incx,
      double *           y,
      const unsigned int incy) const
    {
      dftfe::utils::deviceBlasStatus_t status = hipblasDaxpy(
        d_deviceBlasHandle, int(n), alpha, x, int(incx), y, int(incy));
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xaxpy(
      const unsigned int          n,
      const std::complex<double> *alpha,
      const std::complex<double> *x,
      const unsigned int          incx,
      std::complex<double> *      y,
      const unsigned int          incy) const
    {
      dftfe::utils::deviceBlasStatus_t status =
        hipblasZaxpy(d_deviceBlasHandle,
                     int(n),
                     dftfe::utils::makeDataTypeHipBlasCompatible(alpha),
                     dftfe::utils::makeDataTypeHipBlasCompatible(x),
                     int(incx),
                     dftfe::utils::makeDataTypeHipBlasCompatible(y),
                     int(incy));
      DEVICEBLAS_API_CHECK(status);
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpby(
      const unsigned int n,
      const ValueType2   alpha,
      const ValueType1 * x,
      const ValueType2   beta,
      ValueType1 *       y) const
    {
      hipLaunchKernelGGL(axpbyDeviceKernel,n / dftfe::utils::DEVICE_BLOCK_SIZE) +
          1,dftfe::utils::DEVICE_BLOCK_SIZE,0,0,n,
                      dftfe::utils::makeDataTypeDeviceCompatible(x),
                      dftfe::utils::makeDataTypeDeviceCompatible(y),
                      alpha,
                      beta);
    }


    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType *              addFromVec,
      ValueType *                    addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const
    {
      hipLaunchKernelGGL(axpyStridedBlockAtomicAddDeviceKernel,
                         (contiguousBlockSize * numContiguousBlocks) /
                             dftfe::utils::DEVICE_BLOCK_SIZE +
                           1,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         contiguousBlockSize,
                         numContiguousBlocks,
                         dftfe::utils::makeDataTypeDeviceCompatible(addFromVec),
                         dftfe::utils::makeDataTypeDeviceCompatible(addToVec),
                         addToVecStartingContiguousBlockIds);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1               a,
      const ValueType1 *             s,
      const ValueType2 *             addFromVec,
      ValueType2 *                   addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const
    {
      hipLaunchKernelGGL(axpyStridedBlockAtomicAddDeviceKernel,
                         (contiguousBlockSize * numContiguousBlocks) /
                             dftfe::utils::DEVICE_BLOCK_SIZE +
                           1,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         contiguousBlockSize,
                         numContiguousBlocks,
                         dftfe::utils::makeDataTypeDeviceCompatible(a),
                         dftfe::utils::makeDataTypeDeviceCompatible(s),
                         dftfe::utils::makeDataTypeDeviceCompatible(addFromVec),
                         dftfe::utils::makeDataTypeDeviceCompatible(addToVec),
                         addToVecStartingContiguousBlockIds);
    }



    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const unsigned int N,
      const double *     X,
      const unsigned int INCX,
      const double *     Y,
      const unsigned int INCY,
      double *           result) const
    {
      dftfe::utils::deviceBlasStatus_t status = hipblasDdot(
        d_deviceBlasHandle, int(N), X, int(INCX), Y, int(INCY), result);
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const unsigned int N,
      const double *     X,
      const unsigned int INCX,
      const double *     Y,
      const unsigned int INCY,
      const MPI_Comm &   mpi_communicator,
      double *           result) const
    {
      double localResult                      = 0.0;
      *result                                 = 0.0;
      dftfe::utils::deviceBlasStatus_t status = hipblasDdot(
        d_deviceBlasHandle, int(N), X, int(INCX), Y, int(INCY), &localResult);
      DEVICEBLAS_API_CHECK(status);
      MPI_Allreduce(
        &localResult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const unsigned int          N,
      const std::complex<double> *X,
      const unsigned int          INCX,
      const std::complex<double> *Y,
      const unsigned int          INCY,
      std::complex<double> *      result) const
    {
      dftfe::utils::deviceBlasStatus_t status =
        hipblasZdotc(d_deviceBlasHandle,
                     int(N),
                     dftfe::utils::makeDataTypeHipBlasCompatible(X),
                     int(INCX),
                     dftfe::utils::makeDataTypeHipBlasCompatible(Y),
                     int(INCY),
                     dftfe::utils::makeDataTypeHipBlasCompatible(result));
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const unsigned int          N,
      const std::complex<double> *X,
      const unsigned int          INCX,
      const std::complex<double> *Y,
      const unsigned int          INCY,
      const MPI_Comm &            mpi_communicator,
      std::complex<double> *      result) const
    {
      std::complex<double> localResult = 0.0;
      *result                          = 0.0;
      dftfe::utils::deviceBlasStatus_t status =
        hipblasZdotc(d_deviceBlasHandle,
                     int(N),
                     dftfe::utils::makeDataTypeHipBlasCompatible(X),
                     int(INCX),
                     dftfe::utils::makeDataTypeHipBlasCompatible(Y),
                     int(INCY),
                     dftfe::utils::makeDataTypeHipBlasCompatible(&localResult));
      DEVICEBLAS_API_CHECK(status);
      MPI_Allreduce(&localResult,
                    result,
                    1,
                    dataTypes::mpi_type_id(result),
                    MPI_SUM,
                    mpi_communicator);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char         transA,
      const char         transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const double *     alpha,
      const double *     A,
      const unsigned int lda,
      long long int      strideA,
      const double *     B,
      const unsigned int ldb,
      long long int      strideB,
      const double *     beta,
      double *           C,
      const unsigned int ldc,
      long long int      strideC,
      const int          batchCount) const
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

      dftfe::utils::deviceBlasStatus_t status =
        hipblasDgemmStridedBatched(d_deviceBlasHandle,
                                   transa,
                                   transb,
                                   int(m),
                                   int(n),
                                   int(k),
                                   alpha,
                                   A,
                                   int(lda),
                                   strideA,
                                   B,
                                   int(ldb),
                                   strideB,
                                   beta,
                                   C,
                                   int(ldc),
                                   strideC,
                                   int(batchCount));
      DEVICEBLAS_API_CHECK(status);
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
      std::complex<double> *      C,
      const unsigned int          ldc,
      long long int               strideC,
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

      dftfe::utils::deviceBlasStatus_t status = hipblasZgemmStridedBatched(
        d_deviceBlasHandle,
        transa,
        transb,
        int(m),
        int(n),
        int(k),
        dftfe::utils::makeDataTypeHipBlasCompatible(alpha),
        dftfe::utils::makeDataTypeHipBlasCompatible(A),
        int(lda),
        strideA,
        dftfe::utils::makeDataTypeHipBlasCompatible(B),
        int(ldb),
        strideB,
        dftfe::utils::makeDataTypeHipBlasCompatible(beta),
        dftfe::utils::makeDataTypeHipBlasCompatible(C),
        int(ldc),
        strideC,
        int(batchCount));
      DEVICEBLAS_API_CHECK(status);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char         transA,
      const char         transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const float *      alpha,
      const float *      A,
      const unsigned int lda,
      long long int      strideA,
      const float *      B,
      const unsigned int ldb,
      long long int      strideB,
      const float *      beta,
      float *            C,
      const unsigned int ldc,
      long long int      strideC,
      const int          batchCount) const
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

      dftfe::utils::deviceBlasStatus_t status =
        hipblasSgemmStridedBatched(d_deviceBlasHandle,
                                   transa,
                                   transb,
                                   int(m),
                                   int(n),
                                   int(k),
                                   alpha,
                                   A,
                                   int(lda),
                                   strideA,
                                   B,
                                   int(ldb),
                                   strideB,
                                   beta,
                                   C,
                                   int(ldc),
                                   strideC,
                                   int(batchCount));
      DEVICEBLAS_API_CHECK(status);
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
      std::complex<float> *      C,
      const unsigned int         ldc,
      long long int              strideC,
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

      dftfe::utils::deviceBlasStatus_t status = hipblasCgemmStridedBatched(
        d_deviceBlasHandle,
        transa,
        transb,
        int(m),
        int(n),
        int(k),
        dftfe::utils::makeDataTypeHipBlasCompatible(alpha),
        dftfe::utils::makeDataTypeHipBlasCompatible(A),
        int(lda),
        strideA,
        dftfe::utils::makeDataTypeHipBlasCompatible(B),
        int(ldb),
        strideB,
        dftfe::utils::makeDataTypeHipBlasCompatible(beta),
        dftfe::utils::makeDataTypeHipBlasCompatible(C),
        int(ldc),
        strideC,
        int(batchCount));
      DEVICEBLAS_API_CHECK(status);
    }
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char         transA,
      const char         transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const double *     alpha,
      const double *     A[],
      const unsigned int lda,
      const double *     B[],
      const unsigned int ldb,
      const double *     beta,
      double *           C[],
      const unsigned int ldc,
      const int          batchCount) const
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

      dftfe::utils::deviceBlasStatus_t status =
        hipblasDgemmBatched(d_deviceBlasHandle,
                            transa,
                            transb,
                            int(m),
                            int(n),
                            int(k),
                            alpha,
                            A,
                            int(lda),
                            B,
                            int(ldb),
                            beta,
                            C,
                            int(ldc),
                            int(batchCount));

      DEVICEBLAS_API_CHECK(status);
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
      std::complex<double> *      C[],
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

      dftfe::utils::deviceBlasStatus_t status =
        hipblasZgemmBatched(d_deviceBlasHandle,
                            transa,
                            transb,
                            int(m),
                            int(n),
                            int(k),
                            dftfe::utils::makeDataTypeHipBlasCompatible(alpha),
                            (const hipblasDoubleComplex **)A,
                            int(lda),
                            (const hipblasDoubleComplex **)B,
                            int(ldb),
                            dftfe::utils::makeDataTypeHipBlasCompatible(beta),
                            (hipblasDoubleComplex **)C,
                            int(ldc),
                            int(batchCount));

      DEVICEBLAS_API_CHECK(status);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char         transA,
      const char         transB,
      const unsigned int m,
      const unsigned int n,
      const unsigned int k,
      const float *      alpha,
      const float *      A[],
      const unsigned int lda,
      const float *      B[],
      const unsigned int ldb,
      const float *      beta,
      float *            C[],
      const unsigned int ldc,
      const int          batchCount) const
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

      dftfe::utils::deviceBlasStatus_t status =
        hipblasSgemmBatched(d_deviceBlasHandle,
                            transa,
                            transb,
                            int(m),
                            int(n),
                            int(k),
                            alpha,
                            A,
                            int(lda),
                            B,
                            int(ldb),
                            beta,
                            C,
                            int(ldc),
                            int(batchCount));

      DEVICEBLAS_API_CHECK(status);
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
      std::complex<float> *      C[],
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

      dftfe::utils::deviceBlasStatus_t status =
        hipblasCgemmBatched(d_deviceBlasHandle,
                            transa,
                            transb,
                            int(m),
                            int(n),
                            int(k),
                            dftfe::utils::makeDataTypeHipBlasCompatible(alpha),
                            (const hipblasComplex **)A,
                            int(lda),
                            (const hipblasComplex **)B,
                            int(ldb),
                            dftfe::utils::makeDataTypeHipBlasCompatible(beta),
                            (hipblasComplex **)C,
                            int(ldc),
                            int(batchCount));

      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xnrm2(
      const unsigned int          n,
      const std::complex<double> *x,
      const unsigned int          incx,
      const MPI_Comm &            mpi_communicator,
      double *                    result) const
    {
      double localresult = 0.0;
      *result            = 0.0;
      dftfe::utils::deviceBlasStatus_t status =
        hipblasDznrm2(d_deviceBlasHandle,
                      int(n),
                      dftfe::utils::makeDataTypeHipBlasCompatible(x),
                      int(incx),
                      &localresult);
      localresult *= localresult;
      MPI_Allreduce(
        &localresult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
      *result = std::sqrt(*result);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xnrm2(
      const unsigned int n,
      const double *     x,
      const unsigned int incx,
      const MPI_Comm &   mpi_communicator,
      double *           result) const
    {
      double localresult = 0.0;
      *result            = 0.0;
      dftfe::utils::deviceBlasStatus_t status =
        hipblasDnrm2(d_deviceBlasHandle, int(n), x, int(incx), &localresult);
      localresult *= localresult;
      MPI_Allreduce(
        &localresult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
      *result = std::sqrt(*result);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      ValueType1 *           x,
      const ValueType2       alpha,
      const dftfe::size_type n) const
    {
      hipLaunchKernelGGL(ascalDeviceKernel,
                         n / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         n,
                         dftfe::utils::makeDataTypeDeviceCompatible(x),
                         dftfe::utils::makeDataTypeDeviceCompatible(alpha));
    }

    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::copyComplexArrToRealArrs(
      const dftfe::size_type  size,
      const ValueTypeComplex *complexArr,
      ValueTypeReal *         realArr,
      ValueTypeReal *         imagArr)
    {
      hipLaunchKernelGGL(copyComplexArrToRealArrsDeviceKernel,
                         size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         size,
                         dftfe::utils::makeDataTypeDeviceCompatible(complexArr),
                         realArr,
                         imagArr);
    }



    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::copyRealArrsToComplexArr(
      const dftfe::size_type size,
      const ValueTypeReal *  realArr,
      const ValueTypeReal *  imagArr,
      ValueTypeComplex *     complexArr)
    {
      hipLaunchKernelGGL(copyRealArrsToComplexArrDeviceKernel,
                         size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         size,
                         realArr,
                         imagArr,
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           complexArr));
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const ValueType1 *     valueType1Arr,
                                       ValueType2 *           valueType2Arr)
    {
      copyValueType1ArrToValueType2ArrDeviceKernel<<<
        size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        d_streamId>>>(size,
                      dftfe::utils::makeDataTypeDeviceCompatible(valueType1Arr),
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        valueType2Arr));
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1 *             copyFromVec,
      ValueType2 *                   copyToVecBlock,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {
      hipLaunchKernelGGL(
        stridedCopyToBlockDeviceKernel,
        (contiguousBlockSize * numContiguousBlocks) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        contiguousBlockSize,
        numContiguousBlocks,
        dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
        dftfe::utils::makeDataTypeDeviceCompatible(copyToVecBlock),
        copyFromVecStartingContiguousBlockIds);
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleCopy(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1               a,
      const ValueType1 *             s,
      const ValueType2 *             copyFromVec,
      ValueType2 *                   copyToVecBlock,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {
      hipLaunchKernelGGL(
        stridedCopyToBlockScaleDeviceKernel,
        (contiguousBlockSize * numContiguousBlocks) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        contiguousBlockSize,
        numContiguousBlocks,
        dftfe::utils::makeDataTypeDeviceCompatible(a),
        dftfe::utils::makeDataTypeDeviceCompatible(s),
        dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
        dftfe::utils::makeDataTypeDeviceCompatible(copyToVecBlock),
        copyFromVecStartingContiguousBlockIds);
    }
    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleCopy(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const double *                 s,
      const double *                 copyFromVec,
      double *                       copyToVecBlock,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds);
    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleCopy(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const double *                 s,
      const std::complex<double> *   copyFromVec,
      std::complex<double> *         copyToVecBlock,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds);

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::add(
      double *               y,
      const double *         x,
      const double           alpha,
      const dftfe::size_type size)
    {
      xaxpy(size, &alpha, x, 1, y, 1);
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1 *             copyFromVecBlock,
      ValueType2 *                   copyToVec,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {
      hipLaunchKernelGGL(stridedCopyFromBlockDeviceKernel,
                         (contiguousBlockSize * numContiguousBlocks) /
                             dftfe::utils::DEVICE_BLOCK_SIZE +
                           1,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
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
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const ValueType1 *     copyFromVec,
                                       ValueType2 *           copyToVec)
    {
      hipLaunchKernelGGL(
        stridedCopyToBlockConstantStrideDeviceKernel,
        (blockSizeTo * numBlocks) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
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
      const dftfe::size_type blockSize,
      const dftfe::size_type strideTo,
      const dftfe::size_type strideFrom,
      const dftfe::size_type numBlocks,
      const dftfe::size_type startingToId,
      const dftfe::size_type startingFromId,
      const ValueType1 *     copyFromVec,
      ValueType2 *           copyToVec)
    {
      hipLaunchKernelGGL(
        stridedCopyConstantStrideDeviceKernel,
        (blockSize * numBlocks) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
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
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const ValueType1 *     copyFromVec,
                                         ValueType2 *           copyToVec)
    {
      hipLaunchKernelGGL(
        stridedCopyFromBlockConstantStrideDeviceKernel,
        (blockSizeFrom * numBlocks) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
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
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const ValueType1       a,
      const ValueType1 *     s,
      ValueType2 *           x)
    {
      hipLaunchKernelGGL(stridedBlockScaleDeviceKernel,
                         (contiguousBlockSize * numContiguousBlocks) /
                             dftfe::utils::DEVICE_BLOCK_SIZE +
                           1,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         contiguousBlockSize,
                         numContiguousBlocks,
                         dftfe::utils::makeDataTypeDeviceCompatible(a),
                         dftfe::utils::makeDataTypeDeviceCompatible(s),
                         dftfe::utils::makeDataTypeDeviceCompatible(x));
    }
    // for stridedBlockScale
    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const double           a,
      const double *         s,
      double *               x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const float            a,
      const float *          s,
      float *                x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type      contiguousBlockSize,
      const dftfe::size_type      numContiguousBlocks,
      const std::complex<double>  a,
      const std::complex<double> *s,
      std::complex<double> *      x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type     contiguousBlockSize,
      const dftfe::size_type     numContiguousBlocks,
      const std::complex<float>  a,
      const std::complex<float> *s,
      std::complex<float> *      x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const double           a,
      const double *         s,
      float *                x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const float            a,
      const float *          s,
      double *               x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type      contiguousBlockSize,
      const dftfe::size_type      numContiguousBlocks,
      const std::complex<double>  a,
      const std::complex<double> *s,
      std::complex<float> *       x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type     contiguousBlockSize,
      const dftfe::size_type     numContiguousBlocks,
      const std::complex<float>  a,
      const std::complex<float> *s,
      std::complex<double> *     x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const double           a,
      const double *         s,
      std::complex<double> * x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const double           a,
      const double *         s,
      std::complex<float> *  x);


    // for xscal
    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      double *               x,
      const double           a,
      const dftfe::size_type n) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      float *                x,
      const float            a,
      const dftfe::size_type n) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      std::complex<double> *     x,
      const std::complex<double> a,
      const dftfe::size_type     n) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      std::complex<float> *     x,
      const std::complex<float> a,
      const dftfe::size_type    n) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      std::complex<double> * x,
      const double           a,
      const dftfe::size_type n) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double *                 copyFromVec,
      double *                       copyToVecBlock,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double *                 copyFromVec,
      float *                        copyToVecBlock,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const float *                  copyFromVec,
      float *                        copyToVecBlock,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const std::complex<double> *   copyFromVec,
      std::complex<double> *         copyToVecBlock,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const std::complex<double> *   copyFromVec,
      std::complex<float> *          copyToVecBlock,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const std::complex<float> *    copyFromVec,
      std::complex<float> *          copyToVecBlock,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);


    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const double *         valueType1Arr,
                                       std::complex<double> * valueType2Arr);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const double *         valueType1Arr,
                                       std::complex<float> *  valueType2Arr);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const double *         valueType1Arr,
                                       double *               valueType2Arr);
    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const double *         valueType1Arr,
                                       float *                valueType2Arr);
    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const double *         copyFromVec,
                                       double *               copyToVec);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const std::complex<double> *copyFromVec,
                                       std::complex<double> *      copyToVec);


    // axpyStridedBlockAtomicAdd
    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double *                 addFromVec,
      double *                       addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const std::complex<double> *   addFromVec,
      std::complex<double> *         addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const double *                 s,
      const double *                 addFromVec,
      double *                       addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const double *                 s,
      const std::complex<double> *   addFromVec,
      std::complex<double> *         addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;


    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpby(const unsigned int n,
                                                          const double  alpha,
                                                          const double *x,
                                                          const double  beta,
                                                          double *y) const;


    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpby(
      const unsigned int          n,
      const double                alpha,
      const std::complex<double> *x,
      const double                beta,
      std::complex<double> *      y) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::copyRealArrsToComplexArr(
      const dftfe::size_type size,
      const double *         realArr,
      const double *         imagArr,
      std::complex<double> * complexArr);

  } // End of namespace linearAlgebra
} // End of namespace dftfe
