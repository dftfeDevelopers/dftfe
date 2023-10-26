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
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
#  include <BLASWrapper.h>
#  include <deviceKernelsGeneric.h>
#  include <DeviceTypeConfig.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <DeviceAPICalls.h>
#  include <DeviceDataTypeOverloads.h>
#  include <Exceptions.h>
#  include <cublas_v2.h>
namespace dftfe
{
  namespace linearAlgebra
  {
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
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (*transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else
        {
          // Assert Statement
        }
      if (*transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status = cublasSgemm(d_deviceBlasHandle,
                                                            transa,
                                                            transb,
                                                            int(*m),
                                                            int(*n),
                                                            int(*k),
                                                            alpha,
                                                            A,
                                                            int(*lda),
                                                            B,
                                                            int(*ldb),
                                                            beta,
                                                            C,
                                                            int(*ldc));
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
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (*transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (*transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (*transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (*transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status =
        cublasCgemm(d_deviceBlasHandle,
                    transa,
                    transb,
                    int(*m),
                    int(*n),
                    int(*k),
                    dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                    dftfe::utils::makeDataTypeDeviceCompatible(A),
                    int(*lda),
                    dftfe::utils::makeDataTypeDeviceCompatible(B),
                    int(*ldb),
                    dftfe::utils::makeDataTypeDeviceCompatible(beta),
                    dftfe::utils::makeDataTypeDeviceCompatible(C),
                    int(*ldc));
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
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (*transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }
      if (*transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }


      dftfe::utils::deviceBlasStatus_t status = cublasDgemm(d_deviceBlasHandle,
                                                            transa,
                                                            transb,
                                                            int(*m),
                                                            int(*n),
                                                            int(*k),
                                                            alpha,
                                                            A,
                                                            int(*lda),
                                                            B,
                                                            int(*ldb),
                                                            beta,
                                                            C,
                                                            int(*ldc));
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
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (*transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (*transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (*transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (*transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }


      dftfe::utils::deviceBlasStatus_t status =
        cublasZgemm(d_deviceBlasHandle,
                    transa,
                    transb,
                    int(*m),
                    int(*n),
                    int(*k),
                    dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                    dftfe::utils::makeDataTypeDeviceCompatible(A),
                    int(*lda),
                    dftfe::utils::makeDataTypeDeviceCompatible(B),
                    int(*ldb),
                    dftfe::utils::makeDataTypeDeviceCompatible(beta),
                    dftfe::utils::makeDataTypeDeviceCompatible(C),
                    int(*ldc));
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
      dftfe::utils::deviceStream_t stream)
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasSetStream(d_deviceBlasHandle, stream);
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
      const unsigned int *n,
      const double *      alpha,
      double *            x,
      const unsigned int *incx,
      double *            y,
      const unsigned int *incy) const
    {
      dftfe::utils::deviceBlasStatus_t status = cublasDaxpy(
        d_deviceBlasHandle, int(*n), alpha, x, int(*incx), y, int(*incy));
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xaxpy(
      const unsigned int *        n,
      const std::complex<double> *alpha,
      std::complex<double> *      x,
      const unsigned int *        incx,
      std::complex<double> *      y,
      const unsigned int *        incy) const
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasZaxpy(d_deviceBlasHandle,
                    int(*n),
                    dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                    dftfe::utils::makeDataTypeDeviceCompatible(x),
                    int(*incx),
                    dftfe::utils::makeDataTypeDeviceCompatible(y),
                    int(*incy));
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const unsigned int *N,
      const double *      X,
      const unsigned int *INCX,
      const double *      Y,
      const unsigned int *INCY,
      double *            result) const
    {
      dftfe::utils::deviceBlasStatus_t status = cublasDdot(
        d_deviceBlasHandle, int(*N), X, int(*INCX), Y, int(*INCY), result);
      DEVICEBLAS_API_CHECK(status);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xdot(
      const unsigned int *        N,
      const std::complex<double> *X,
      const unsigned int *        INCX,
      const std::complex<double> *Y,
      const unsigned int *        INCY,
      std::complex<double> *      result) const
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasZdotc(d_deviceBlasHandle,
                    int(*N),
                    dftfe::utils::makeDataTypeDeviceCompatible(X),
                    int(*INCX),
                    dftfe::utils::makeDataTypeDeviceCompatible(Y),
                    int(*INCY),
                    dftfe::utils::makeDataTypeDeviceCompatible(result));
      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char *        transA,
      const char *        transB,
      const unsigned int *m,
      const unsigned int *n,
      const unsigned int *k,
      const double *      alpha,
      const double *      A,
      const unsigned int *lda,
      long long int *     strideA,
      const double *      B,
      const unsigned int *ldb,
      long long int *     strideB,
      const double *      beta,
      double *            C,
      const unsigned int *ldc,
      const int *         batchCount,
      long long int *     strideC) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (*transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }
      if (*transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status =
        cublasDgemmStridedBatched(d_deviceBlasHandle,
                                  transa,
                                  transb,
                                  int(*m),
                                  int(*n),
                                  int(*k),
                                  alpha,
                                  A,
                                  int(*lda),
                                  *strideA,
                                  B,
                                  int(*ldb),
                                  *strideB,
                                  beta,
                                  C,
                                  int(*ldc),
                                  *strideC,
                                  int(*batchCount));
      DEVICEBLAS_API_CHECK(status);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char *                transA,
      const char *                transB,
      const unsigned int *        m,
      const unsigned int *        n,
      const unsigned int *        k,
      const std::complex<double> *alpha,
      const std::complex<double> *A,
      const unsigned int *        lda,
      long long int *             strideA,
      const std::complex<double> *B,
      const unsigned int *        ldb,
      long long int *             strideB,
      const std::complex<double> *beta,
      std::complex<double> *      C,
      const unsigned int *        ldc,
      const int *                 batchCount,
      long long int *             strideC) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (*transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (*transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (*transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (*transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status = cublasZgemmStridedBatched(
        d_deviceBlasHandle,
        transa,
        transb,
        int(*m),
        int(*n),
        int(*k),
        dftfe::utils::makeDataTypeDeviceCompatible(alpha),
        dftfe::utils::makeDataTypeDeviceCompatible(A),
        int(*lda),
        *strideA,
        dftfe::utils::makeDataTypeDeviceCompatible(B),
        int(*ldb),
        *strideB,
        dftfe::utils::makeDataTypeDeviceCompatible(beta),
        dftfe::utils::makeDataTypeDeviceCompatible(C),
        int(*ldc),
        *strideC,
        int(*batchCount));
      DEVICEBLAS_API_CHECK(status);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char *        transA,
      const char *        transB,
      const unsigned int *m,
      const unsigned int *n,
      const unsigned int *k,
      const float *       alpha,
      const float *       A,
      const unsigned int *lda,
      long long int *     strideA,
      const float *       B,
      const unsigned int *ldb,
      long long int *     strideB,
      const float *       beta,
      float *             C,
      const unsigned int *ldc,
      const int *         batchCount,
      long long int *     strideC) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (*transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }
      if (*transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status =
        cublasSgemmStridedBatched(d_deviceBlasHandle,
                                  transa,
                                  transb,
                                  int(*m),
                                  int(*n),
                                  int(*k),
                                  alpha,
                                  A,
                                  int(*lda),
                                  *strideA,
                                  B,
                                  int(*ldb),
                                  *strideB,
                                  beta,
                                  C,
                                  int(*ldc),
                                  *strideC,
                                  int(*batchCount));
      DEVICEBLAS_API_CHECK(status);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmStridedBatched(
      const char *               transA,
      const char *               transB,
      const unsigned int *       m,
      const unsigned int *       n,
      const unsigned int *       k,
      const std::complex<float> *alpha,
      const std::complex<float> *A,
      const unsigned int *       lda,
      long long int *            strideA,
      const std::complex<float> *B,
      const unsigned int *       ldb,
      long long int *            strideB,
      const std::complex<float> *beta,
      std::complex<float> *      C,
      const unsigned int *       ldc,
      const int *                batchCount,
      long long int *            strideC) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (*transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (*transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (*transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (*transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status = cublasCgemmStridedBatched(
        d_deviceBlasHandle,
        transa,
        transb,
        int(*m),
        int(*n),
        int(*k),
        dftfe::utils::makeDataTypeDeviceCompatible(alpha),
        dftfe::utils::makeDataTypeDeviceCompatible(A),
        int(*lda),
        *strideA,
        dftfe::utils::makeDataTypeDeviceCompatible(B),
        int(*ldb),
        *strideB,
        dftfe::utils::makeDataTypeDeviceCompatible(beta),
        dftfe::utils::makeDataTypeDeviceCompatible(C),
        int(*ldc),
        *strideC,
        int(*batchCount));
      DEVICEBLAS_API_CHECK(status);
    }
    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char *        transA,
      const char *        transB,
      const unsigned int *m,
      const unsigned int *n,
      const unsigned int *k,
      const double *      alpha,
      const double *      A[],
      const unsigned int *lda,
      const double *      B[],
      const unsigned int *ldb,
      const double *      beta,
      double *            C[],
      const unsigned int *ldc,
      const int *         batchCount) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (*transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }
      if (*transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status =
        cublasDgemmBatched(d_deviceBlasHandle,
                           transa,
                           transb,
                           int(*m),
                           int(*n),
                           int(*k),
                           alpha,
                           A,
                           int(*lda),
                           B,
                           int(*ldb),
                           beta,
                           C,
                           int(*ldc),
                           int(*batchCount));

      DEVICEBLAS_API_CHECK(status);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char *                transA,
      const char *                transB,
      const unsigned int *        m,
      const unsigned int *        n,
      const unsigned int *        k,
      const std::complex<double> *alpha,
      const std::complex<double> *A[],
      const unsigned int *        lda,
      const std::complex<double> *B[],
      const unsigned int *        ldb,
      const std::complex<double> *beta,
      std::complex<double> *      C[],
      const unsigned int *        ldc,
      const int *                 batchCount) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (*transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (*transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (*transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (*transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status =
        cublasZgemmBatched(d_deviceBlasHandle,
                           transa,
                           transb,
                           int(*m),
                           int(*n),
                           int(*k),
                           dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                           (const dftfe::utils::deviceDoubleComplex **)A,
                           int(*lda),
                           (const dftfe::utils::deviceDoubleComplex **)B,
                           int(*ldb),
                           dftfe::utils::makeDataTypeDeviceCompatible(beta),
                           (dftfe::utils::deviceDoubleComplex **)C,
                           int(*ldc),
                           int(*batchCount));

      DEVICEBLAS_API_CHECK(status);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char *        transA,
      const char *        transB,
      const unsigned int *m,
      const unsigned int *n,
      const unsigned int *k,
      const float *       alpha,
      const float *       A[],
      const unsigned int *lda,
      const float *       B[],
      const unsigned int *ldb,
      const float *       beta,
      float *             C[],
      const unsigned int *ldc,
      const int *         batchCount) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (*transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }
      if (*transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;

      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status =
        cublasSgemmBatched(d_deviceBlasHandle,
                           transa,
                           transb,
                           int(*m),
                           int(*n),
                           int(*k),
                           alpha,
                           A,
                           int(*lda),
                           B,
                           int(*ldb),
                           beta,
                           C,
                           int(*ldc),
                           int(*batchCount));

      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xgemmBatched(
      const char *               transA,
      const char *               transB,
      const unsigned int *       m,
      const unsigned int *       n,
      const unsigned int *       k,
      const std::complex<float> *alpha,
      const std::complex<float> *A[],
      const unsigned int *       lda,
      const std::complex<float> *B[],
      const unsigned int *       ldb,
      const std::complex<float> *beta,
      std::complex<float> *      C[],
      const unsigned int *       ldc,
      const int *                batchCount) const
    {
      dftfe::utils::deviceBlasOperation_t transa, transb;
      if (*transA == 'N')
        transa = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transA == 'T')
        transa = dftfe::utils::DEVICEBLAS_OP_T;
      else if (*transA == 'C')
        transa = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }
      if (*transB == 'N')
        transb = dftfe::utils::DEVICEBLAS_OP_N;
      else if (*transB == 'T')
        transb = dftfe::utils::DEVICEBLAS_OP_T;
      else if (*transB == 'C')
        transb = dftfe::utils::DEVICEBLAS_OP_C;
      else
        {
          // Assert Statement
        }

      dftfe::utils::deviceBlasStatus_t status =
        cublasCgemmBatched(d_deviceBlasHandle,
                           transa,
                           transb,
                           int(*m),
                           int(*n),
                           int(*k),
                           dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                           (const dftfe::utils::deviceFloatComplex **)A,
                           int(*lda),
                           (const dftfe::utils::deviceFloatComplex **)B,
                           int(*ldb),
                           dftfe::utils::makeDataTypeDeviceCompatible(beta),
                           (dftfe::utils::deviceFloatComplex **)C,
                           int(*ldc),
                           int(*batchCount));

      DEVICEBLAS_API_CHECK(status);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xnrm2(
      const unsigned int *        n,
      const std::complex<double> *x,
      const unsigned int *        incx,
      const MPI_Comm &            mpi_communicator,
      double *                    result) const
    {
      double                           localresult = 0.0;
      dftfe::utils::deviceBlasStatus_t status =
        cublasDznrm2(d_deviceBlasHandle,
                     int(*n),
                     dftfe::utils::makeDataTypeDeviceCompatible(x),
                     int(*incx),
                     &localresult);
      localresult *= localresult;
      MPI_Allreduce(
        &localresult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xnrm2(
      const unsigned int *n,
      const double *      x,
      const unsigned int *incx,
      const MPI_Comm &    mpi_communicator,
      double *            result) const
    {
      double                           localresult = 0.0;
      dftfe::utils::deviceBlasStatus_t status =
        cublasDnrm2(d_deviceBlasHandle, int(*n), x, int(*incx), &localresult);
      localresult *= localresult;
      MPI_Allreduce(
        &localresult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      const unsigned int *n,
      const double *      alpha,
      double *            x,
      const unsigned int *inc) const
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasDscal(d_deviceBlasHandle, int(*n), alpha, x, int(*inc));
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      const unsigned int *n,
      const float *       alpha,
      float *             x,
      const unsigned int *inc) const
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasSscal(d_deviceBlasHandle, int(*n), alpha, x, int(*inc));
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      const unsigned int *        n,
      const std::complex<double> *alpha,
      std::complex<double> *      x,
      const unsigned int *        inc) const
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasZscal(d_deviceBlasHandle,
                    int(*n),
                    dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                    dftfe::utils::makeDataTypeDeviceCompatible(x),
                    int(*inc));
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
      const unsigned int *       n,
      const std::complex<float> *alpha,
      std::complex<float> *      x,
      const unsigned int *       inc) const
    {
      dftfe::utils::deviceBlasStatus_t status =
        cublasCscal(d_deviceBlasHandle,
                    int(*n),
                    dftfe::utils::makeDataTypeDeviceCompatible(alpha),
                    dftfe::utils::makeDataTypeDeviceCompatible(x),
                    int(*inc));
    }



  } // End of namespace linearAlgebra
} // End of namespace dftfe
#endif
