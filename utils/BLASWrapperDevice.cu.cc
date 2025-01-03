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
#include <DeviceKernelLauncherConstants.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <cublas_v2.h>
#include "BLASWrapperDeviceKernels.cc"

namespace dftfe
{
  namespace linearAlgebra
  {
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::BLASWrapper()
    {
      dftfe::utils::deviceBlasStatus_t status;
      status = create();
      status = setStream(NULL);
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
        const dftfe::size_type             size,
        const ValueType1 *                 valueType1Arr,
        ValueType2 *                       valueType2Arr,
        const dftfe::utils::deviceStream_t streamId)
    {
      copyValueType1ArrToValueType2ArrDeviceKernel<<<
        size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        streamId>>>(size,
                    dftfe::utils::makeDataTypeDeviceCompatible(valueType1Arr),
                    dftfe::utils::makeDataTypeDeviceCompatible(valueType2Arr));
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
        cublasZcopy(d_deviceBlasHandle,
                    n,
                    dftfe::utils::makeDataTypeDeviceCompatible(x),
                    incx,
                    dftfe::utils::makeDataTypeDeviceCompatible(y),
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
        cublasCcopy(d_deviceBlasHandle,
                    n,
                    dftfe::utils::makeDataTypeDeviceCompatible(x),
                    incx,
                    dftfe::utils::makeDataTypeDeviceCompatible(y),
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
        cublasDcopy(d_deviceBlasHandle, n, x, incx, y, incy);
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
        cublasScopy(d_deviceBlasHandle, n, x, incx, y, incy);
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
      cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
      if (d_opType == tensorOpDataType::tf32)
        computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
      else if (d_opType == tensorOpDataType::bf16)
        computeType = CUBLAS_COMPUTE_32F_FAST_16BF;
      else if (d_opType == tensorOpDataType::fp16)
        computeType = CUBLAS_COMPUTE_32F_FAST_16F;
      dftfe::utils::deviceBlasStatus_t status =
        cublasGemmEx(d_deviceBlasHandle,
                     transa,
                     transb,
                     int(m),
                     int(n),
                     int(k),
                     (const void *)alpha,
                     (const void *)A,
                     CUDA_R_32F,
                     int(lda),
                     (const void *)B,
                     CUDA_R_32F,
                     int(ldb),
                     (const void *)beta,
                     (void *)C,
                     CUDA_R_32F,
                     int(ldc),
                     computeType,
                     CUBLAS_GEMM_DEFAULT);
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
      cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
      if (d_opType == tensorOpDataType::tf32)
        computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
      else if (d_opType == tensorOpDataType::bf16)
        computeType = CUBLAS_COMPUTE_32F_FAST_16BF;
      else if (d_opType == tensorOpDataType::fp16)
        computeType = CUBLAS_COMPUTE_32F_FAST_16F;
      dftfe::utils::deviceBlasStatus_t status =
        cublasGemmEx(d_deviceBlasHandle,
                     transa,
                     transb,
                     int(m),
                     int(n),
                     int(k),
                     (const void *)alpha,
                     (const void *)A,
                     CUDA_C_32F,
                     int(lda),
                     (const void *)B,
                     CUDA_C_32F,
                     int(ldb),
                     (const void *)beta,
                     (void *)C,
                     CUDA_C_32F,
                     int(ldc),
                     computeType,
                     CUBLAS_GEMM_DEFAULT);

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


      dftfe::utils::deviceBlasStatus_t status = cublasDgemm(d_deviceBlasHandle,
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
        cublasZgemm(d_deviceBlasHandle,
                    transa,
                    transb,
                    int(m),
                    int(n),
                    int(k),
                    dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                    dftfe::utils::makeDataTypeDeviceCompatible(A),
                    int(lda),
                    dftfe::utils::makeDataTypeDeviceCompatible(B),
                    int(ldb),
                    dftfe::utils::makeDataTypeDeviceCompatible(beta),
                    dftfe::utils::makeDataTypeDeviceCompatible(C),
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
      dftfe::utils::deviceBlasStatus_t status = cublasDgemv(d_deviceBlasHandle,
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

      dftfe::utils::deviceBlasStatus_t status = cublasSgemv(d_deviceBlasHandle,
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
        cublasZgemv(d_deviceBlasHandle,
                    transa,
                    int(m),
                    int(n),
                    dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                    dftfe::utils::makeDataTypeDeviceCompatible(A),
                    int(lda),
                    dftfe::utils::makeDataTypeDeviceCompatible(x),
                    int(incx),
                    dftfe::utils::makeDataTypeDeviceCompatible(beta),
                    dftfe::utils::makeDataTypeDeviceCompatible(y),
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
        cublasCgemv(d_deviceBlasHandle,
                    transa,
                    int(m),
                    int(n),
                    dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                    dftfe::utils::makeDataTypeDeviceCompatible(A),
                    int(lda),
                    dftfe::utils::makeDataTypeDeviceCompatible(x),
                    int(incx),
                    dftfe::utils::makeDataTypeDeviceCompatible(beta),
                    dftfe::utils::makeDataTypeDeviceCompatible(y),
                    int(incy));
      DEVICEBLAS_API_CHECK(status);
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::create()
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasCreate(&d_deviceBlasHandle);
      DEVICEBLAS_API_CHECK(status);
      return status;
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::destroy()
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasDestroy(d_deviceBlasHandle);
      DEVICEBLAS_API_CHECK(status);
      return status;
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::setStream(
      dftfe::utils::deviceStream_t streamId)
    {
      d_streamId = streamId;
      dftfe::utils::deviceBlasStatus_t status =
        cublasSetStream(d_deviceBlasHandle, d_streamId);
      DEVICEBLAS_API_CHECK(status);
      return status;
    }

    dftfe::utils::deviceBlasStatus_t
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::setMathMode(
      dftfe::utils::deviceBlasMath_t mathMode)
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasSetMathMode(d_deviceBlasHandle, mathMode);
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
      dftfe::utils::deviceBlasStatus_t status = cublasDaxpy(
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
        cublasZaxpy(d_deviceBlasHandle,
                    int(n),
                    dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                    dftfe::utils::makeDataTypeDeviceCompatible(x),
                    int(incx),
                    dftfe::utils::makeDataTypeDeviceCompatible(y),
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
      axpbyDeviceKernel<<<(n / dftfe::utils::DEVICE_BLOCK_SIZE) + 1,
                          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
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
      const unsigned int m,
      const unsigned int n,
      const ValueType0   alpha,
      const ValueType1 * A,
      const ValueType2 * B,
      const ValueType3 * D,
      ValueType4 *       C) const
    {
      ApaBDDeviceKernel<<<(n * m / dftfe::utils::DEVICE_BLOCK_SIZE) + 1,
                          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        m,
        n,
        alpha,
        dftfe::utils::makeDataTypeDeviceCompatible(A),
        dftfe::utils::makeDataTypeDeviceCompatible(B),
        dftfe::utils::makeDataTypeDeviceCompatible(D),
        dftfe::utils::makeDataTypeDeviceCompatible(C));
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
      axpyStridedBlockAtomicAddDeviceKernel<<<
        (contiguousBlockSize * numContiguousBlocks) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        contiguousBlockSize,
        numContiguousBlocks,
        dftfe::utils::makeDataTypeDeviceCompatible(addFromVec),
        dftfe::utils::makeDataTypeDeviceCompatible(addToVec),
        addToVecStartingContiguousBlockIds);
    }


    template <typename ValueType1, typename ValueType2, typename ValueType3>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1               a,
      const ValueType1 *             s,
      const ValueType2 *             addFromVec,
      ValueType3 *                   addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const
    {
      axpyStridedBlockAtomicAddDeviceKernel<<<
        (contiguousBlockSize * numContiguousBlocks) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
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
      const MPI_Comm &   mpi_communicator,
      double *           result) const
    {
      double localResult                      = 0.0;
      *result                                 = 0.0;
      dftfe::utils::deviceBlasStatus_t status = cublasDdot(
        d_deviceBlasHandle, int(N), X, int(INCX), Y, int(INCY), &localResult);
      DEVICEBLAS_API_CHECK(status);
      MPI_Allreduce(
        &localResult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
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
      dftfe::utils::deviceBlasStatus_t status = cublasDdot(
        d_deviceBlasHandle, int(N), X, int(INCX), Y, int(INCY), result);
      DEVICEBLAS_API_CHECK(status);
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
        cublasZdotc(d_deviceBlasHandle,
                    int(N),
                    dftfe::utils::makeDataTypeDeviceCompatible(X),
                    int(INCX),
                    dftfe::utils::makeDataTypeDeviceCompatible(Y),
                    int(INCY),
                    dftfe::utils::makeDataTypeDeviceCompatible(result));
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
        cublasZdotc(d_deviceBlasHandle,
                    int(N),
                    dftfe::utils::makeDataTypeDeviceCompatible(X),
                    int(INCX),
                    dftfe::utils::makeDataTypeDeviceCompatible(Y),
                    int(INCY),
                    dftfe::utils::makeDataTypeDeviceCompatible(&localResult));
      DEVICEBLAS_API_CHECK(status);
      MPI_Allreduce(&localResult,
                    result,
                    1,
                    dataTypes::mpi_type_id(result),
                    MPI_SUM,
                    mpi_communicator);
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::hadamardProduct(
      const unsigned int m,
      const ValueType *  X,
      const ValueType *  Y,
      ValueType *        output) const
    {
      hadamardProductKernel<<<(m) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                              dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        m,
        dftfe::utils::makeDataTypeDeviceCompatible(X),
        dftfe::utils::makeDataTypeDeviceCompatible(Y),
        dftfe::utils::makeDataTypeDeviceCompatible(output));
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::hadamardProductWithConj(
      const unsigned int m,
      const ValueType *  X,
      const ValueType *  Y,
      ValueType *        output) const
    {
      hadamardProductWithConjKernel<<<(m) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                                      dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        m,
        dftfe::utils::makeDataTypeDeviceCompatible(X),
        dftfe::utils::makeDataTypeDeviceCompatible(Y),
        dftfe::utils::makeDataTypeDeviceCompatible(output));
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::MultiVectorXDot(
      const unsigned int contiguousBlockSize,
      const unsigned int numContiguousBlocks,
      const ValueType *  X,
      const ValueType *  Y,
      const ValueType *  onesVec,
      ValueType *        tempVector,
      ValueType *        tempResults,
      ValueType *        result) const
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
      const ValueType *  X,
      const ValueType *  Y,
      const ValueType *  onesVec,
      ValueType *        tempVector,
      ValueType *        tempResults,
      const MPI_Comm &   mpi_communicator,
      ValueType *        result) const

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
        cublasDgemmStridedBatched(d_deviceBlasHandle,
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

      dftfe::utils::deviceBlasStatus_t status = cublasZgemmStridedBatched(
        d_deviceBlasHandle,
        transa,
        transb,
        int(m),
        int(n),
        int(k),
        dftfe::utils::makeDataTypeDeviceCompatible(alpha),
        dftfe::utils::makeDataTypeDeviceCompatible(A),
        int(lda),
        strideA,
        dftfe::utils::makeDataTypeDeviceCompatible(B),
        int(ldb),
        strideB,
        dftfe::utils::makeDataTypeDeviceCompatible(beta),
        dftfe::utils::makeDataTypeDeviceCompatible(C),
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
      cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
      if (d_opType == tensorOpDataType::tf32)
        computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
      else if (d_opType == tensorOpDataType::bf16)
        computeType = CUBLAS_COMPUTE_32F_FAST_16BF;
      else if (d_opType == tensorOpDataType::fp16)
        computeType = CUBLAS_COMPUTE_32F_FAST_16F;
      dftfe::utils::deviceBlasStatus_t status =
        cublasGemmStridedBatchedEx(d_deviceBlasHandle,
                                   transa,
                                   transb,
                                   int(m),
                                   int(n),
                                   int(k),
                                   (const void *)alpha,
                                   (const void *)A,
                                   CUDA_R_32F,
                                   int(lda),
                                   strideA,
                                   (const void *)B,
                                   CUDA_R_32F,
                                   int(ldb),
                                   strideB,
                                   (const void *)beta,
                                   (void *)C,
                                   CUDA_R_32F,
                                   int(ldc),
                                   strideC,
                                   int(batchCount),
                                   computeType,
                                   CUBLAS_GEMM_DEFAULT);

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

      cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
      if (d_opType == tensorOpDataType::tf32)
        computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
      else if (d_opType == tensorOpDataType::bf16)
        computeType = CUBLAS_COMPUTE_32F_FAST_16BF;
      else if (d_opType == tensorOpDataType::fp16)
        computeType = CUBLAS_COMPUTE_32F_FAST_16F;
      dftfe::utils::deviceBlasStatus_t status =
        cublasGemmStridedBatchedEx(d_deviceBlasHandle,
                                   transa,
                                   transb,
                                   int(m),
                                   int(n),
                                   int(k),
                                   (const void *)alpha,
                                   (const void *)A,
                                   CUDA_C_32F,
                                   int(lda),
                                   strideA,
                                   (const void *)B,
                                   CUDA_C_32F,
                                   int(ldb),
                                   strideB,
                                   (const void *)beta,
                                   (void *)C,
                                   CUDA_C_32F,
                                   int(ldc),
                                   strideC,
                                   int(batchCount),
                                   computeType,
                                   CUBLAS_GEMM_DEFAULT);
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
        cublasDgemmBatched(d_deviceBlasHandle,
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
        cublasZgemmBatched(d_deviceBlasHandle,
                           transa,
                           transb,
                           int(m),
                           int(n),
                           int(k),
                           dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                           (const dftfe::utils::deviceDoubleComplex **)A,
                           int(lda),
                           (const dftfe::utils::deviceDoubleComplex **)B,
                           int(ldb),
                           dftfe::utils::makeDataTypeDeviceCompatible(beta),
                           (dftfe::utils::deviceDoubleComplex **)C,
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

      cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
      if (d_opType == tensorOpDataType::tf32)
        computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
      else if (d_opType == tensorOpDataType::bf16)
        computeType = CUBLAS_COMPUTE_32F_FAST_16BF;
      else if (d_opType == tensorOpDataType::fp16)
        computeType = CUBLAS_COMPUTE_32F_FAST_16F;
      dftfe::utils::deviceBlasStatus_t status =
        cublasGemmBatchedEx(d_deviceBlasHandle,
                            transa,
                            transb,
                            int(m),
                            int(n),
                            int(k),
                            (const void *)alpha,
                            (const void **)A,
                            CUDA_R_32F,
                            int(lda),
                            (const void **)B,
                            CUDA_R_32F,
                            int(ldb),
                            (const void *)beta,
                            (void **)C,
                            CUDA_R_32F,
                            int(ldc),
                            int(batchCount),
                            computeType,
                            CUBLAS_GEMM_DEFAULT);

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
      cublasComputeType_t computeType = CUBLAS_COMPUTE_32F;
      if (d_opType == tensorOpDataType::tf32)
        computeType = CUBLAS_COMPUTE_32F_FAST_TF32;
      else if (d_opType == tensorOpDataType::bf16)
        computeType = CUBLAS_COMPUTE_32F_FAST_16BF;
      else if (d_opType == tensorOpDataType::fp16)
        computeType = CUBLAS_COMPUTE_32F_FAST_16F;
      dftfe::utils::deviceBlasStatus_t status =
        cublasGemmBatchedEx(d_deviceBlasHandle,
                            transa,
                            transb,
                            int(m),
                            int(n),
                            int(k),
                            (const void *)alpha,
                            (const void **)A,
                            CUDA_C_32F,
                            int(lda),
                            (const void **)B,
                            CUDA_C_32F,
                            int(ldb),
                            (const void *)beta,
                            (void **)C,
                            CUDA_C_32F,
                            int(ldc),
                            int(batchCount),
                            computeType,
                            CUBLAS_GEMM_DEFAULT);


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
        cublasDznrm2(d_deviceBlasHandle,
                     int(n),
                     dftfe::utils::makeDataTypeDeviceCompatible(x),
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
        cublasDnrm2(d_deviceBlasHandle, int(n), x, int(incx), &localresult);
      localresult *= localresult;
      MPI_Allreduce(
        &localresult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
      *result = std::sqrt(*result);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::add(
      double *               y,
      const double *         x,
      const double           alpha,
      const dftfe::size_type size)
    {
      xaxpy(size, &alpha, x, 1, y, 1);
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::addVecOverContinuousIndex(
      const dftfe::size_type numContiguousBlocks,
      const dftfe::size_type contiguousBlockSize,
      const ValueType *      input1,
      const ValueType *      input2,
      ValueType *            output)
    {
      addVecOverContinuousIndexKernel<<<
        (numContiguousBlocks) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        numContiguousBlocks,
        contiguousBlockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(input1),
        dftfe::utils::makeDataTypeDeviceCompatible(input2),
        dftfe::utils::makeDataTypeDeviceCompatible(output));
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      ValueType1 *           x,
      const ValueType2       alpha,
      const dftfe::size_type n) const
    {
      ascalDeviceKernel<<<n / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
                          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
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
      copyComplexArrToRealArrsDeviceKernel<<<
        size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
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
      copyRealArrsToComplexArrDeviceKernel<<<
        size / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        size,
        realArr,
        imagArr,
        dftfe::utils::makeDataTypeDeviceCompatible(complexArr));
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
      stridedCopyToBlockDeviceKernel<<<(contiguousBlockSize *
                                        numContiguousBlocks) /
                                           dftfe::utils::DEVICE_BLOCK_SIZE +
                                         1,
                                       dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        contiguousBlockSize,
        numContiguousBlocks,
        dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
        dftfe::utils::makeDataTypeDeviceCompatible(copyToVecBlock),
        copyFromVecStartingContiguousBlockIds);
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
      stridedCopyFromBlockDeviceKernel<<<(contiguousBlockSize *
                                          numContiguousBlocks) /
                                             dftfe::utils::DEVICE_BLOCK_SIZE +
                                           1,
                                         dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        contiguousBlockSize,
        numContiguousBlocks,
        dftfe::utils::makeDataTypeDeviceCompatible(copyFromVecBlock),
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
                                       ValueType2 *           copyToVec) const
    {
      stridedCopyToBlockConstantStrideDeviceKernel<<<
        (blockSizeTo * numBlocks) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
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
      stridedCopyConstantStrideDeviceKernel<<<
        (blockSize * numBlocks) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
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
      stridedCopyFromBlockConstantStrideDeviceKernel<<<
        (blockSizeFrom * numBlocks) / dftfe::utils::DEVICE_BLOCK_SIZE + 1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        blockSizeTo,
        blockSizeFrom,
        numBlocks,
        startingId,
        dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
        dftfe::utils::makeDataTypeDeviceCompatible(copyToVec));
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
      stridedCopyToBlockScaleDeviceKernel<<<
        (contiguousBlockSize * numContiguousBlocks) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
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
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const ValueType1       a,
      const ValueType1 *     s,
      ValueType2 *           x)
    {
      stridedBlockScaleDeviceKernel<<<(contiguousBlockSize *
                                       numContiguousBlocks) /
                                          dftfe::utils::DEVICE_BLOCK_SIZE +
                                        1,
                                      dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        contiguousBlockSize,
        numContiguousBlocks,
        dftfe::utils::makeDataTypeDeviceCompatible(a),
        dftfe::utils::makeDataTypeDeviceCompatible(s),
        dftfe::utils::makeDataTypeDeviceCompatible(x));
    }


    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleColumnWise(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const ValueType *      beta,
      ValueType *            x)
    {
      stridedBlockScaleColumnWiseKernel<<<(contiguousBlockSize *
                                           numContiguousBlocks) /
                                              dftfe::utils::DEVICE_BLOCK_SIZE +
                                            1,
                                          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        contiguousBlockSize,
        numContiguousBlocks,
        dftfe::utils::makeDataTypeDeviceCompatible(beta),
        dftfe::utils::makeDataTypeDeviceCompatible(x));
    }
    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
      stridedBlockScaleAndAddColumnWise(
        const dftfe::size_type contiguousBlockSize,
        const dftfe::size_type numContiguousBlocks,
        const ValueType *      x,
        const ValueType *      beta,
        ValueType *            y)
    {
      stridedBlockScaleAndAddColumnWiseKernel<<<
        (contiguousBlockSize * numContiguousBlocks) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
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
        const dftfe::size_type contiguousBlockSize,
        const dftfe::size_type numContiguousBlocks,
        const ValueType *      x,
        const ValueType *      alpha,
        const ValueType *      y,
        const ValueType *      beta,
        ValueType *            z)
    {
      stridedBlockScaleAndAddTwoVecColumnWiseKernel<<<
        (contiguousBlockSize * numContiguousBlocks) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        contiguousBlockSize,
        numContiguousBlocks,
        dftfe::utils::makeDataTypeDeviceCompatible(x),
        dftfe::utils::makeDataTypeDeviceCompatible(alpha),
        dftfe::utils::makeDataTypeDeviceCompatible(y),
        dftfe::utils::makeDataTypeDeviceCompatible(beta),
        dftfe::utils::makeDataTypeDeviceCompatible(z));
    }
#include "./BLASWrapperDevice.inst.cc"
  } // End of namespace linearAlgebra
} // End of namespace dftfe
