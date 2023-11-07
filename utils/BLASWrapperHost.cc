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
#include <linearAlgebraOperations.h>

namespace dftfe
{
  namespace linearAlgebra
  {
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::BLASWrapper()
    {}


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

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xscal(
      ValueType1 *           x,
      const ValueType2       alpha,
      const dftfe::size_type n) const
    {
      std::transform(x, x + n, x, [&](auto &c) { return alpha * c; });
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
      std::complex<float> *     x,
      const std::complex<float> a,
      const dftfe::size_type    n) const;

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
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xnrm2(
      const unsigned int n,
      const double *     x,
      const unsigned int incx,
      const MPI_Comm &   mpi_communicator,
      double *           result) const
    {
      double localresult = dnrm2_(&n, x, &incx);
      localresult *= localresult;
      MPI_Allreduce(
        &localresult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
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
      localresult *= localresult;
      MPI_Allreduce(
        &localresult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
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
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xaxpy(
      const unsigned int n,
      const double *     alpha,
      double *           x,
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
      std::complex<double> *      x,
      const unsigned int          incx,
      std::complex<double> *      y,
      const unsigned int          incy) const
    {
      zaxpy_(&n, alpha, x, &incx, y, &incy);
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
    {}



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
    {}


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

    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::copyComplexArrToRealArrs(
      const dftfe::size_type  size,
      const ValueTypeComplex *complexArr,
      ValueTypeReal *         realArr,
      ValueTypeReal *         imagArr)
    {}



    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::copyRealArrsToComplexArr(
      const dftfe::size_type size,
      const ValueTypeReal *  realArr,
      const ValueTypeReal *  imagArr,
      ValueTypeComplex *     complexArr)
    {}

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const ValueType1 *     valueType1Arr,
                                       ValueType2 *           valueType2Arr)
    {}

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyToBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1 *             copyFromVec,
      ValueType2 *                   copyToVecBlock,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {}


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyFromBlock(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1 *             copyFromVecBlock,
      ValueType2 *                   copyToVec,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {}


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const ValueType1 *     copyFromVec,
                                       ValueType2 *           copyToVec)
    {}

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
    {}


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const ValueType1 *     copyFromVec,
                                         ValueType2 *           copyToVec)
    {}
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

    // template void
    // BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
    //   const dftfe::size_type      contiguousBlockSize,
    //   const dftfe::size_type      numContiguousBlocks,
    //   const std::complex<double>  a,
    //   const std::complex<double> *s,
    //   std::complex<float> *       x);

    // template void
    // BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
    //   const dftfe::size_type     contiguousBlockSize,
    //   const dftfe::size_type     numContiguousBlocks,
    //   const std::complex<float>  a,
    //   const std::complex<float> *s,
    //   std::complex<double> *     x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const double           a,
      const double *         s,
      std::complex<double> * x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const double           a,
      const double *         s,
      std::complex<float> *  x);

  } // End of namespace linearAlgebra
} // End of namespace dftfe
