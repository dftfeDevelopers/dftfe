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
#include <linearAlgebraOperations.h>
#include <dftUtils.h>
namespace dftfe
{
  namespace linearAlgebra
  {
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::BLASWrapper()
    {}

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProduct(
      const unsigned int m,
      const ValueType *  X,
      const ValueType *  Y,
      ValueType *        output) const
    {
      for (unsigned int i = 0; i < m; i++)
        {
          output[i] = X[i] * Y[i];
        }
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProductWithConj(
      const unsigned int m,
      const ValueType *  X,
      const ValueType *  Y,
      ValueType *        output) const
    {
      for (unsigned int i = 0; i < m; i++)
        {
          output[i] = X[i] * Y[i];
        }
    }


    template <>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProductWithConj(
      const unsigned int          m,
      const std::complex<double> *X,
      const std::complex<double> *Y,
      std::complex<double> *      output) const
    {
      for (unsigned int i = 0; i < m; i++)
        {
          output[i] = std::conj(X[i]) * Y[i];
        }
    }

    template <>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProductWithConj(
      const unsigned int         m,
      const std::complex<float> *X,
      const std::complex<float> *Y,
      std::complex<float> *      output) const
    {
      for (unsigned int i = 0; i < m; i++)
        {
          output[i] = std::conj(X[i]) * Y[i];
        }
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemm(
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
      sgemm_(
        &transA, &transB, &m, &n, &k, alpha, A, &lda, B, &ldb, beta, C, &ldc);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemm(
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
      dgemm_(
        &transA, &transB, &m, &n, &k, alpha, A, &lda, B, &ldb, beta, C, &ldc);
    }


    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::addVecOverContinuousIndex(
      const dftfe::size_type numContiguousBlocks,
      const dftfe::size_type contiguousBlockSize,
      const ValueType *      input1,
      const ValueType *      input2,
      ValueType *            output)
    {
      for (dftfe::size_type iIndex = 0; iIndex < numContiguousBlocks; iIndex++)
        {
          for (dftfe::size_type jIndex = 0; jIndex < contiguousBlockSize;
               jIndex++)
            {
              output[iIndex] += input1[iIndex * contiguousBlockSize + jIndex] *
                                input2[iIndex * contiguousBlockSize + jIndex];
            }
        }
    }

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::addVecOverContinuousIndex(
      const dftfe::size_type numContiguousBlocks,
      const dftfe::size_type contiguousBlockSize,
      const double *         input1,
      const double *         input2,
      double *               output);
    template <typename ValueType0,
              typename ValueType1,
              typename ValueType2,
              typename ValueType3,
              typename ValueType4>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::ApaBD(const unsigned int m,
                                                        const unsigned int n,
                                                        const ValueType0  alpha,
                                                        const ValueType1 *A,
                                                        const ValueType2 *B,
                                                        const ValueType3 *D,
                                                        ValueType4 *C) const
    {
      for (unsigned int iRow = 0; iRow < m; ++iRow)
        {
          for (unsigned int iCol = 0; iCol < n; ++iCol)
            {
              C[iCol + n * iRow] =
                A[iCol + n * iRow] + alpha * B[iCol + n * iRow] * D[iCol];
            }
        }
    }

    template <>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::ApaBD(
      const unsigned int          m,
      const unsigned int          n,
      const double                alpha,
      const std::complex<float> * A,
      const std::complex<double> *B,
      const double *              D,
      std::complex<float> *       C) const
    {
      for (unsigned int iRow = 0; iRow < m; ++iRow)
        {
          for (unsigned int iCol = 0; iCol < n; ++iCol)
            {
              C[iCol + n * iRow] = std::complex<double>(A[iCol + n * iRow]) +
                                   alpha * B[iCol + n * iRow] * D[iCol];
            }
        }
    }

    template <>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::ApaBD(
      const unsigned int          m,
      const unsigned int          n,
      const double                alpha,
      const std::complex<float> * A,
      const std::complex<double> *B,
      const double *              D,
      std::complex<double> *      C) const
    {
      for (unsigned int iRow = 0; iRow < m; ++iRow)
        {
          for (unsigned int iCol = 0; iCol < n; ++iCol)
            {
              C[iCol + n * iRow] = std::complex<double>(A[iCol + n * iRow]) +
                                   alpha * B[iCol + n * iRow] * D[iCol];
            }
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xscal(
      ValueType1 *           x,
      const ValueType2       alpha,
      const dftfe::size_type n) const
    {
      std::transform(x, x + n, x, [&alpha](auto &c) { return alpha * c; });
    }
    // for xscal
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xscal(
      double *               x,
      const double           a,
      const dftfe::size_type n) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xscal(
      float *                x,
      const float            a,
      const dftfe::size_type n) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xscal(
      std::complex<double> *     x,
      const std::complex<double> a,
      const dftfe::size_type     n) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xscal(
      std::complex<double> * x,
      const double           a,
      const dftfe::size_type n) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xscal(
      std::complex<float> *     x,
      const std::complex<float> a,
      const dftfe::size_type    n) const;

    // hadamard product
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProduct(
      const unsigned int m,
      const double *     X,
      const double *     Y,
      double *           output) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProduct(
      const unsigned int m,
      const float *      X,
      const float *      Y,
      float *            output) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProduct(
      const unsigned int          m,
      const std::complex<double> *X,
      const std::complex<double> *Y,
      std::complex<double> *      output) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProduct(
      const unsigned int         m,
      const std::complex<float> *X,
      const std::complex<float> *Y,
      std::complex<float> *      output) const;

    // hadamard product with conj
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProductWithConj(
      const unsigned int m,
      const double *     X,
      const double *     Y,
      double *           output) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProductWithConj(
      const unsigned int m,
      const float *      X,
      const float *      Y,
      float *            output) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProductWithConj(
      const unsigned int          m,
      const std::complex<double> *X,
      const std::complex<double> *Y,
      std::complex<double> *      output) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProductWithConj(
      const unsigned int         m,
      const std::complex<float> *X,
      const std::complex<float> *Y,
      std::complex<float> *      output) const;


    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xcopy(
      const unsigned int n,
      const double *     x,
      const unsigned int incx,
      double *           y,
      const unsigned int incy) const
    {
      dcopy_(&n, x, &incx, y, &incy);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xcopy(
      const unsigned int n,
      const float *      x,
      const unsigned int incx,
      float *            y,
      const unsigned int incy) const
    {
      scopy_(&n, x, &incx, y, &incy);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemm(
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
      cgemm_(
        &transA, &transB, &m, &n, &k, alpha, A, &lda, B, &ldb, beta, C, &ldc);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemm(
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
      zgemm_(
        &transA, &transB, &m, &n, &k, alpha, A, &lda, B, &ldb, beta, C, &ldc);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemv(
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
      dgemv_(&transA, &m, &n, alpha, A, &lda, x, &incx, beta, y, &incy);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemv(
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
      sgemv_(&transA, &m, &n, alpha, A, &lda, x, &incx, beta, y, &incy);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemv(
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
      zgemv_(&transA, &m, &n, alpha, A, &lda, x, &incx, beta, y, &incy);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemv(
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
      cgemv_(&transA, &m, &n, alpha, A, &lda, x, &incx, beta, y, &incy);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xcopy(
      const unsigned int          n,
      const std::complex<double> *x,
      const unsigned int          incx,
      std::complex<double> *      y,
      const unsigned int          incy) const
    {
      zcopy_(&n, x, &incx, y, &incy);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xcopy(
      const unsigned int         n,
      const std::complex<float> *x,
      const unsigned int         incx,
      std::complex<float> *      y,
      const unsigned int         incy) const
    {
      ccopy_(&n, x, &incx, y, &incy);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xnrm2(
      const unsigned int n,
      const double *     x,
      const unsigned int incx,
      const MPI_Comm &   mpi_communicator,
      double *           result) const
    {
      double localresult = dnrm2_(&n, x, &incx);
      *result            = 0.0;
      localresult *= localresult;
      MPI_Allreduce(
        &localresult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
      *result = std::sqrt(*result);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xnrm2(
      const unsigned int          n,
      const std::complex<double> *x,
      const unsigned int          incx,
      const MPI_Comm &            mpi_communicator,
      double *                    result) const
    {
      double localresult = dznrm2_(&n, x, &incx);
      *result            = 0.0;
      localresult *= localresult;
      MPI_Allreduce(
        &localresult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
      *result = std::sqrt(*result);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xdot(const unsigned int N,
                                                       const double *     X,
                                                       const unsigned int INCX,
                                                       const double *     Y,
                                                       const unsigned int INCY,
                                                       double *result) const
    {
      *result = ddot_(&N, X, &INCX, Y, &INCY);
    }
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xdot(
      const unsigned int          N,
      const std::complex<double> *X,
      const unsigned int          INCX,
      const std::complex<double> *Y,
      const unsigned int          INCY,
      std::complex<double> *      result) const
    {
      *result = zdotc_(&N, X, &INCX, Y, &INCY);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xdot(
      const unsigned int N,
      const double *     X,
      const unsigned int INCX,
      const double *     Y,
      const unsigned int INCY,
      const MPI_Comm &   mpi_communicator,
      double *           result) const
    {
      double localResult = 0.0;
      *result            = 0.0;
      localResult        = ddot_(&N, X, &INCX, Y, &INCY);
      MPI_Allreduce(&localResult,
                    result,
                    1,
                    dataTypes::mpi_type_id(result),
                    MPI_SUM,
                    mpi_communicator);
    }
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xdot(
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
      localResult =
        std::inner_product(X,
                           X + N,
                           Y,
                           std::complex<double>(0.0),
                           std::plus<>{},
                           [](auto &a, auto &b) { return std::conj(a) * b; });
      MPI_Allreduce(&localResult,
                    result,
                    1,
                    dataTypes::mpi_type_id(result),
                    MPI_SUM,
                    mpi_communicator);
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::MultiVectorXDot(
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
            result,
            numVec);
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::MultiVectorXDot(
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
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xaxpy(
      const unsigned int n,
      const double *     alpha,
      const double *     x,
      const unsigned int incx,
      double *           y,
      const unsigned int incy) const
    {
      daxpy_(&n, alpha, x, &incx, y, &incy);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xaxpy(
      const unsigned int          n,
      const std::complex<double> *alpha,
      const std::complex<double> *x,
      const unsigned int          incx,
      std::complex<double> *      y,
      const unsigned int          incy) const
    {
      zaxpy_(&n, alpha, x, &incx, y, &incy);
    }


    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType *              addFromVec,
      ValueType *                    addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const
    {
      for (unsigned int iBlock = 0; iBlock < numContiguousBlocks; ++iBlock)
        std::transform(addFromVec + iBlock * contiguousBlockSize,
                       addFromVec + (iBlock + 1) * contiguousBlockSize,
                       addToVec + addToVecStartingContiguousBlockIds[iBlock],
                       addToVec + addToVecStartingContiguousBlockIds[iBlock],
                       std::plus<>{});
    }

    template <typename ValueType1, typename ValueType2, typename ValueType3>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1               a,
      const ValueType1 *             s,
      const ValueType2 *             addFromVec,
      ValueType3 *                   addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const
    {
      for (unsigned int iBlock = 0; iBlock < numContiguousBlocks; ++iBlock)
        {
          ValueType1 coeff = a * s[iBlock];
          std::transform(addFromVec + iBlock * contiguousBlockSize,
                         addFromVec + (iBlock + 1) * contiguousBlockSize,
                         addToVec + addToVecStartingContiguousBlockIds[iBlock],
                         addToVec + addToVecStartingContiguousBlockIds[iBlock],
                         [&coeff](auto &p, auto &q) { return p * coeff + q; });
        }
    }

    template <>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const double *                 s,
      const std::complex<float> *    addFromVec,
      std::complex<float> *          addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const
    {
      for (unsigned int iBlock = 0; iBlock < numContiguousBlocks; ++iBlock)
        {
          double coeff = a * s[iBlock];
          std::transform(addFromVec + iBlock * contiguousBlockSize,
                         addFromVec + (iBlock + 1) * contiguousBlockSize,
                         addToVec + addToVecStartingContiguousBlockIds[iBlock],
                         addToVec + addToVecStartingContiguousBlockIds[iBlock],
                         [&coeff](auto &p, auto &q) {
                           return std::complex<double>(p) * coeff +
                                  std::complex<double>(q);
                         });
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpby(const unsigned int n,
                                                        const ValueType2  alpha,
                                                        const ValueType1 *x,
                                                        const ValueType2  beta,
                                                        ValueType1 *y) const
    {
      std::transform(x, x + n, y, y, [&alpha, &beta](auto &p, auto &q) {
        return alpha * p + beta * q;
      });
    }

    template <>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpby(
      const unsigned int         n,
      const double               alpha,
      const std::complex<float> *x,
      const double               beta,
      std::complex<float> *      y) const
    {
      std::transform(x, x + n, y, y, [&alpha, &beta](auto &p, auto &q) {
        return alpha * std::complex<double>(p) + beta * std::complex<double>(q);
      });
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xsymv(
      const char         UPLO,
      const unsigned int N,
      const double *     alpha,
      const double *     A,
      const unsigned int LDA,
      const double *     X,
      const unsigned int INCX,
      const double *     beta,
      double *           C,
      const unsigned int INCY) const
    {
      dsymv_(&UPLO, &N, alpha, A, &LDA, X, &INCX, beta, C, &INCY);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::add(
      double *               y,
      const double *         x,
      const double           alpha,
      const dftfe::size_type size)
    {
      xaxpy(size, &alpha, x, 1, y, 1);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemmBatched(
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
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }



    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemmBatched(
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
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemmStridedBatched(
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
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemmStridedBatched(
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
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemmStridedBatched(
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
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemmStridedBatched(
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
    }

    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::copyComplexArrToRealArrs(
      const dftfe::size_type  size,
      const ValueTypeComplex *complexArr,
      ValueTypeReal *         realArr,
      ValueTypeReal *         imagArr)
    {
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }



    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::copyRealArrsToComplexArr(
      const dftfe::size_type size,
      const ValueTypeReal *  realArr,
      const ValueTypeReal *  imagArr,
      ValueTypeComplex *     complexArr)
    {
      std::transform(realArr,
                     realArr + size,
                     imagArr,
                     complexArr,
                     [](auto &a, auto &b) { return ValueTypeComplex(a, b); });
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const ValueType1 *     valueType1Arr,
                                       ValueType2 *           valueType2Arr)
    {
      for (unsigned int i = 0; i < size; ++i)
        valueType2Arr[i] = valueType1Arr[i];
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyToBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1 *             copyFromVec,
      ValueType2 *                   copyToVecBlock,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {
      for (unsigned int iBlock = 0; iBlock < numContiguousBlocks; ++iBlock)
        {
          xcopy(contiguousBlockSize,
                copyFromVec + copyFromVecStartingContiguousBlockIds[iBlock],
                1,
                copyToVecBlock + iBlock * contiguousBlockSize,
                1);
        }
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyFromBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1 *             copyFromVecBlock,
      ValueType2 *                   copyToVec,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const ValueType1 *     copyFromVec,
                                       ValueType2 *           copyToVec) const
    {
      for (dftfe::size_type iIndex = 0; iIndex < numBlocks; iIndex++)
        {
          for (dftfe::size_type jIndex = 0; jIndex < blockSizeTo; jIndex++)
            {
              copyToVec[iIndex * blockSizeTo + jIndex] =
                copyFromVec[iIndex * blockSizeFrom + startingId + jIndex];
            }
        }
    }

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const double *         copyFromVec,
                                       double *               copyToVec) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const std::complex<double> *copyFromVec,
                                       std::complex<double> *copyToVec) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const float *          copyFromVec,
                                       float *                copyToVec) const;


    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::size_type     blockSizeTo,
                                       const dftfe::size_type     blockSizeFrom,
                                       const dftfe::size_type     numBlocks,
                                       const dftfe::size_type     startingId,
                                       const std::complex<float> *copyFromVec,
                                       std::complex<float> *copyToVec) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const double *         copyFromVec,
                                       float *                copyToVec) const;
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const float *          copyFromVec,
                                       double *               copyToVec) const;
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const std::complex<double> *copyFromVec,
                                       std::complex<float> *copyToVec) const;
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::size_type     blockSizeTo,
                                       const dftfe::size_type     blockSizeFrom,
                                       const dftfe::size_type     numBlocks,
                                       const dftfe::size_type     startingId,
                                       const std::complex<float> *copyFromVec,
                                       std::complex<double> *copyToVec) const;

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyConstantStride(
      const dftfe::size_type blockSize,
      const dftfe::size_type strideTo,
      const dftfe::size_type strideFrom,
      const dftfe::size_type numBlocks,
      const dftfe::size_type startingToId,
      const dftfe::size_type startingFromId,
      const ValueType1 *     copyFromVec,
      ValueType2 *           copyToVec)
    {
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const ValueType1 *     copyFromVec,
                                         ValueType2 *           copyToVec)
    {
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }
    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScaleCopy(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1               a,
      const ValueType1 *             s,
      const ValueType2 *             copyFromVec,
      ValueType2 *                   copyToVecBlock,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {
      for (int iBatch = 0; iBatch < numContiguousBlocks; iBatch++)
        {
          ValueType1 alpha = a * s[iBatch];
          std::transform(copyFromVec +
                           copyFromVecStartingContiguousBlockIds[iBatch],
                         copyFromVec +
                           copyFromVecStartingContiguousBlockIds[iBatch] +
                           contiguousBlockSize,
                         copyToVecBlock + iBatch * contiguousBlockSize,
                         [&alpha](auto &a) { return alpha * a; });
        }
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScaleColumnWise(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const ValueType *      beta,
      ValueType *            x)
    {
      for (unsigned int i = 0; i < numContiguousBlocks; i++)
        {
          for (unsigned int j = 0; j < contiguousBlockSize; j++)
            {
              x[j + i * contiguousBlockSize] *= beta[j];
            }
        }
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddColumnWise(
        const dftfe::size_type contiguousBlockSize,
        const dftfe::size_type numContiguousBlocks,
        const ValueType *      x,
        const ValueType *      beta,
        ValueType *            y)
    {
      for (unsigned int i = 0; i < numContiguousBlocks; i++)
        {
          for (unsigned int j = 0; j < contiguousBlockSize; j++)
            {
              y[j + i * contiguousBlockSize] +=
                beta[j] * x[j + i * contiguousBlockSize];
            }
        }
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddTwoVecColumnWise(
        const dftfe::size_type contiguousBlockSize,
        const dftfe::size_type numContiguousBlocks,
        const ValueType *      x,
        const ValueType *      alpha,
        const ValueType *      y,
        const ValueType *      beta,
        ValueType *            z)
    {
      for (unsigned int i = 0; i < numContiguousBlocks; i++)
        {
          for (unsigned int j = 0; j < contiguousBlockSize; j++)
            {
              z[j + i * contiguousBlockSize] =
                alpha[j] * x[j + i * contiguousBlockSize] +
                beta[j] * y[j + i * contiguousBlockSize];
            }
        }
    }


    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScaleCopy(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const double *                 s,
      const double *                 copyFromVec,
      double *                       copyToVecBlock,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds);
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScaleCopy(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const double *                 s,
      const std::complex<double> *   copyFromVec,
      std::complex<double> *         copyToVecBlock,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds);


    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScaleColumnWise(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const double *         beta,
      double *               x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScaleColumnWise(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const float *          beta,
      float *                x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScaleColumnWise(
      const dftfe::size_type      contiguousBlockSize,
      const dftfe::size_type      numContiguousBlocks,
      const std::complex<double> *beta,
      std::complex<double> *      x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScaleColumnWise(
      const dftfe::size_type     contiguousBlockSize,
      const dftfe::size_type     numContiguousBlocks,
      const std::complex<float> *beta,
      std::complex<float> *      x);

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const ValueType1       a,
      const ValueType1 *     s,
      ValueType2 *           x)
    {
      for (int iBatch = 0; iBatch < numContiguousBlocks; iBatch++)
        {
          ValueType1 alpha = a * s[iBatch];
          xscal(x + iBatch * contiguousBlockSize, alpha, contiguousBlockSize);
        }
    }
    // MultiVectorXDot
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::MultiVectorXDot(
      const unsigned int contiguousBlockSize,
      const unsigned int numContiguousBlocks,
      const double *     X,
      const double *     Y,
      const double *     onesVec,
      double *           tempVector,
      double *           tempResults,
      double *           result) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::MultiVectorXDot(
      const unsigned int contiguousBlockSize,
      const unsigned int numContiguousBlocks,
      const double *     X,
      const double *     Y,
      const double *     onesVec,
      double *           tempVector,
      double *           tempResults,
      const MPI_Comm &   mpi_communicator,
      double *           result) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::MultiVectorXDot(
      const unsigned int          contiguousBlockSize,
      const unsigned int          numContiguousBlocks,
      const std::complex<double> *X,
      const std::complex<double> *Y,
      const std::complex<double> *onesVec,
      std::complex<double> *      tempVector,
      std::complex<double> *      tempResults,
      std::complex<double> *      result) const;
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::MultiVectorXDot(
      const unsigned int          contiguousBlockSize,
      const unsigned int          numContiguousBlocks,
      const std::complex<double> *X,
      const std::complex<double> *Y,
      const std::complex<double> *onesVec,
      std::complex<double> *      tempVector,
      std::complex<double> *      tempResults,
      const MPI_Comm &            mpi_communicator,
      std::complex<double> *      result) const;

    // stridedBlockScale
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const double           a,
      const double *         s,
      double *               x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const float            a,
      const float *          s,
      float *                x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::size_type      contiguousBlockSize,
      const dftfe::size_type      numContiguousBlocks,
      const std::complex<double>  a,
      const std::complex<double> *s,
      std::complex<double> *      x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::size_type     contiguousBlockSize,
      const dftfe::size_type     numContiguousBlocks,
      const std::complex<float>  a,
      const std::complex<float> *s,
      std::complex<float> *      x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const double           a,
      const double *         s,
      float *                x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const float            a,
      const float *          s,
      double *               x);

    // for stridedBlockScaleAndAddColumnWise
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddColumnWise(
        const dftfe::size_type contiguousBlockSize,
        const dftfe::size_type numContiguousBlocks,
        const double *         x,
        const double *         beta,
        double *               y);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddColumnWise(
        const dftfe::size_type contiguousBlockSize,
        const dftfe::size_type numContiguousBlocks,
        const float *          x,
        const float *          beta,
        float *                y);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddColumnWise(
        const dftfe::size_type      contiguousBlockSize,
        const dftfe::size_type      numContiguousBlocks,
        const std::complex<double> *x,
        const std::complex<double> *beta,
        std::complex<double> *      y);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddColumnWise(
        const dftfe::size_type     contiguousBlockSize,
        const dftfe::size_type     numContiguousBlocks,
        const std::complex<float> *x,
        const std::complex<float> *beta,
        std::complex<float> *      y);

    // for stridedBlockScaleAndAddTwoVecColumnWise
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddTwoVecColumnWise(
        const dftfe::size_type contiguousBlockSize,
        const dftfe::size_type numContiguousBlocks,
        const double *         x,
        const double *         alpha,
        const double *         y,
        const double *         beta,
        double *               z);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddTwoVecColumnWise(
        const dftfe::size_type contiguousBlockSize,
        const dftfe::size_type numContiguousBlocks,
        const float *          x,
        const float *          alpha,
        const float *          y,
        const float *          beta,
        float *                z);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddTwoVecColumnWise(
        const dftfe::size_type      contiguousBlockSize,
        const dftfe::size_type      numContiguousBlocks,
        const std::complex<double> *x,
        const std::complex<double> *alpha,
        const std::complex<double> *y,
        const std::complex<double> *beta,
        std::complex<double> *      z);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddTwoVecColumnWise(
        const dftfe::size_type     contiguousBlockSize,
        const dftfe::size_type     numContiguousBlocks,
        const std::complex<float> *x,
        const std::complex<float> *alpha,
        const std::complex<float> *y,
        const std::complex<float> *beta,
        std::complex<float> *      z);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const double           a,
      const double *         s,
      std::complex<double> * x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyToBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double *                 copyFromVec,
      double *                       copyToVecBlock,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyToBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const float *                  copyFromVec,
      float *                        copyToVecBlock,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);


    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyToBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const std::complex<double> *   copyFromVec,
      std::complex<double> *         copyToVecBlock,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyToBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const std::complex<float> *    copyFromVec,
      std::complex<float> *          copyToVecBlock,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

    // template void
    // BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyToBlock(
    //   const dftfe::size_type         contiguousBlockSize,
    //   const dftfe::size_type         numContiguousBlocks,
    //   const std::complex<double> *   copyFromVec,
    //   std::complex<float> *         copyToVecBlock,
    //   const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const double *         valueType1Arr,
                                       std::complex<double> * valueType2Arr);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const double *         valueType1Arr,
                                       double *               valueType2Arr);
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const double *         valueType1Arr,
                                       float *                valueType2Arr);
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      copyValueType1ArrToValueType2Arr(
        const dftfe::size_type      size,
        const std::complex<double> *valueType1Arr,
        std::complex<float> *       valueType2Arr);

    // axpyStridedBlockAtomicAdd
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double *                 addFromVec,
      double *                       addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const std::complex<double> *   addFromVec,
      std::complex<double> *         addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const double *                 s,
      const double *                 addFromVec,
      double *                       addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const double *                 s,
      const std::complex<double> *   addFromVec,
      std::complex<double> *         addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const double *                 s,
      const float *                  addFromVec,
      float *                        addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const double *                 s,
      const std::complex<float> *    addFromVec,
      std::complex<float> *          addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const float                    a,
      const float *                  s,
      const float *                  addFromVec,
      float *                        addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const float                    a,
      const float *                  s,
      const std::complex<float> *    addFromVec,
      std::complex<float> *          addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpby(const unsigned int n,
                                                        const double  alpha,
                                                        const double *x,
                                                        const double  beta,
                                                        double *      y) const;


    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpby(
      const unsigned int          n,
      const double                alpha,
      const std::complex<double> *x,
      const double                beta,
      std::complex<double> *      y) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpby(const unsigned int n,
                                                        const double alpha,
                                                        const float *x,
                                                        const double beta,
                                                        float *      y) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::ApaBD(const unsigned int m,
                                                        const unsigned int n,
                                                        const double  alpha,
                                                        const double *A,
                                                        const double *B,
                                                        const double *D,
                                                        double *      C) const;
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::ApaBD(
      const unsigned int          m,
      const unsigned int          n,
      const double                alpha,
      const std::complex<double> *A,
      const std::complex<double> *B,
      const double *              D,
      std::complex<double> *      C) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::ApaBD(const unsigned int m,
                                                        const unsigned int n,
                                                        const double  alpha,
                                                        const float * A,
                                                        const double *B,
                                                        const double *D,
                                                        float *       C) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::ApaBD(const unsigned int m,
                                                        const unsigned int n,
                                                        const double  alpha,
                                                        const float * A,
                                                        const double *B,
                                                        const double *D,
                                                        double *      C) const;

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::copyRealArrsToComplexArr(
      const dftfe::size_type size,
      const double *         realArr,
      const double *         imagArr,
      std::complex<double> * complexArr);



  } // End of namespace linearAlgebra
} // End of namespace dftfe
