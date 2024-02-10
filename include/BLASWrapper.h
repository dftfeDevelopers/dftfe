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
//

#ifndef BLASWrapper_h
#define BLASWrapper_h

#include <dftfeDataTypes.h>
#include <MemorySpaceType.h>
#include <complex>
#include <TypeConfig.h>
#include <DeviceTypeConfig.h>
#include <cmath>


namespace dftfe
{
  namespace linearAlgebra
  {
    template <dftfe::utils::MemorySpace memorySpace>
    class BLASWrapper;

    template <>
    class BLASWrapper<dftfe::utils::MemorySpace::HOST>
    {
    public:
      BLASWrapper();
      // Real-Single Precision GEMM
      void
      xgemm(const char         transA,
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
            const unsigned int ldc) const;
      // Complex-Single Precision GEMM
      void
      xgemm(const char                 transA,
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
            const unsigned int         ldc) const;

      // Real-double precison GEMM
      void
      xgemm(const char         transA,
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
            const unsigned int ldc) const;


      // Complex-double precision GEMM
      void
      xgemm(const char                  transA,
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
            const unsigned int          ldc) const;

      void
      xgemv(const char         transA,
            const unsigned int m,
            const unsigned int n,
            const double *     alpha,
            const double *     A,
            const unsigned int lda,
            const double *     x,
            const unsigned int incx,
            const double *     beta,
            double *           y,
            const unsigned int incy) const;

      void
      xgemv(const char         transA,
            const unsigned int m,
            const unsigned int n,
            const float *      alpha,
            const float *      A,
            const unsigned int lda,
            const float *      x,
            const unsigned int incx,
            const float *      beta,
            float *            y,
            const unsigned int incy) const;

      void
      xgemv(const char                  transA,
            const unsigned int          m,
            const unsigned int          n,
            const std::complex<double> *alpha,
            const std::complex<double> *A,
            const unsigned int          lda,
            const std::complex<double> *x,
            const unsigned int          incx,
            const std::complex<double> *beta,
            std::complex<double> *      y,
            const unsigned int          incy) const;

      void
      xgemv(const char                 transA,
            const unsigned int         m,
            const unsigned int         n,
            const std::complex<float> *alpha,
            const std::complex<float> *A,
            const unsigned int         lda,
            const std::complex<float> *x,
            const unsigned int         incx,
            const std::complex<float> *beta,
            std::complex<float> *      y,
            const unsigned int         incy) const;


      template <typename ValueType1, typename ValueType2>
      void
      xscal(ValueType1 *           x,
            const ValueType2       alpha,
            const dftfe::size_type n) const;

      // Real-Float scaling of Real-vector


      // Real double Norm2
      void
      xnrm2(const unsigned int n,
            const double *     x,
            const unsigned int incx,
            const MPI_Comm &   mpi_communicator,
            double *           result) const;


      // Comples double Norm2
      void
      xnrm2(const unsigned int          n,
            const std::complex<double> *x,
            const unsigned int          incx,
            const MPI_Comm &            mpi_communicator,
            double *                    result) const;
      // Real dot product
      void
      xdot(const unsigned int N,
           const double *     X,
           const unsigned int INCX,
           const double *     Y,
           const unsigned int INCY,
           double *           result) const;
      // Real dot proeuct with all Reduce call
      void
      xdot(const unsigned int N,
           const double *     X,
           const unsigned int INCX,
           const double *     Y,
           const unsigned int INCY,
           const MPI_Comm &   mpi_communicator,
           double *           result) const;

      // Complex dot product
      void
      xdot(const unsigned int          N,
           const std::complex<double> *X,
           const unsigned int          INCX,
           const std::complex<double> *Y,
           const unsigned int          INCY,
           std::complex<double> *      result) const;

      // Real double Ax+y
      void
      xaxpy(const unsigned int n,
            const double *     alpha,
            const double *     x,
            const unsigned int incx,
            double *           y,
            const unsigned int incy) const;

      // Complex double Ax+y
      void
      xaxpy(const unsigned int          n,
            const std::complex<double> *alpha,
            const std::complex<double> *x,
            const unsigned int          incx,
            std::complex<double> *      y,
            const unsigned int          incy) const;

      // Real copy of double data
      void
      xcopy(const unsigned int n,
            const double *     x,
            const unsigned int incx,
            double *           y,
            const unsigned int incy) const;

      // Real copy of double data to float
      void
      xcopy(const unsigned int n,
            double *           x,
            const unsigned int incx,
            float *            y,
            const unsigned int incy) const;

      // Complex double copy of data
      void
      xcopy(const unsigned int          n,
            const std::complex<double> *x,
            const unsigned int          incx,
            std::complex<double> *      y,
            const unsigned int          incy) const;

      // Real copy of float data
      void
      xcopy(const unsigned int n,
            const float *      x,
            const unsigned int incx,
            float *            y,
            const unsigned int incy) const;

      // Complex float copy of data
      void
      xcopy(const unsigned int         n,
            const std::complex<float> *x,
            const unsigned int         incx,
            std::complex<float> *      y,
            const unsigned int         incy) const;

      void
      xcopy(const unsigned int    n,
            std::complex<double> *x,
            const unsigned int    incx,
            std::complex<float> * y,
            const unsigned int    incy) const;

      // Real double symmetric matrix-vector product
      void
      xsymv(const char         UPLO,
            const unsigned int N,
            const double *     alpha,
            const double *     A,
            const unsigned int LDA,
            const double *     X,
            const unsigned int INCX,
            const double *     beta,
            double *           C,
            const unsigned int INCY) const;

      void
      xgemmBatched(const char         transA,
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
                   const int          batchCount) const;

      void
      xgemmBatched(const char                  transA,
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
                   const int                   batchCount) const;


      void
      xgemmBatched(const char         transA,
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
                   const int          batchCount) const;

      void
      xgemmBatched(const char                 transA,
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
                   const int                  batchCount) const;


      void
      xgemmStridedBatched(const char         transA,
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
                          const int          batchCount) const;

      void
      xgemmStridedBatched(const char                  transA,
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
                          const int                   batchCount) const;

      void
      xgemmStridedBatched(const char                 transA,
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
                          const int                  batchCount) const;

      void
      xgemmStridedBatched(const char         transA,
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
                          const int          batchCount) const;

      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyComplexArrToRealArrs(const dftfe::size_type  size,
                               const ValueTypeComplex *complexArr,
                               ValueTypeReal *         realArr,
                               ValueTypeReal *         imagArr);


      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyRealArrsToComplexArr(const dftfe::size_type size,
                               const ValueTypeReal *  realArr,
                               const ValueTypeReal *  imagArr,
                               ValueTypeComplex *     complexArr);

      template <typename ValueType1, typename ValueType2>
      void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const ValueType1 *     valueType1Arr,
                                       ValueType2 *           valueType2Arr);


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const ValueType1 *             copyFromVec,
        ValueType2 *                   copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);



      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const ValueType1 *             copyFromVecBlock,
        ValueType2 *                   copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const ValueType1 *     copyFromVec,
                                       ValueType2 *           copyToVec);


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyConstantStride(const dftfe::size_type blockSize,
                                const dftfe::size_type strideTo,
                                const dftfe::size_type strideFrom,
                                const dftfe::size_type numBlocks,
                                const dftfe::size_type startingToId,
                                const dftfe::size_type startingFromId,
                                const ValueType1 *     copyFromVec,
                                ValueType2 *           copyToVec);


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const ValueType1 *     copyFromVec,
                                         ValueType2 *           copyToVec);

      template <typename ValueType>
      void
      axpyStridedBlockAtomicAdd(const dftfe::size_type contiguousBlockSize,
                                const dftfe::size_type numContiguousBlocks,
                                const ValueType *      addFromVec,
                                ValueType *            addToVec,
                                const dftfe::global_size_type
                                  *addToVecStartingContiguousBlockIds) const;

      template <typename ValueType>
      void
      axpyStridedBlockAtomicAdd(const dftfe::size_type contiguousBlockSize,
                                const dftfe::size_type numContiguousBlocks,
                                const ValueType *      addFromVec,
                                double *               addToVecReal,
                                double *               addToVecImag,
                                const dftfe::global_size_type
                                  *addToVecStartingContiguousBlockIds) const;

      template <typename ValueType1, typename ValueType2>
      void
      stridedBlockScale(const dftfe::size_type contiguousBlockSize,
                        const dftfe::size_type numContiguousBlocks,
                        const ValueType1       a,
                        const ValueType1 *     s,
                        ValueType2 *           x);

      void
      add(double *               y,
          const double *         x,
          const double           alpha,
          const dftfe::size_type size);

      template <typename ValueType>
      void
      sadd(ValueType *            y,
           ValueType *            x,
           const ValueType        beta,
           const dftfe::size_type size);

    private:
    };
#if defined(DFTFE_WITH_DEVICE)
    template <>
    class BLASWrapper<dftfe::utils::MemorySpace::DEVICE>
    {
    public:
      BLASWrapper();
      // Real-Single Precision GEMM
      void
      xgemm(const char         transA,
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
            const unsigned int ldc) const;
      // Complex-Single Precision GEMM
      void
      xgemm(const char                 transA,
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
            const unsigned int         ldc) const;

      // Real-double precison GEMM
      void
      xgemm(const char         transA,
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
            const unsigned int ldc) const;


      // Complex-double precision GEMM
      void
      xgemm(const char                  transA,
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
            const unsigned int          ldc) const;


      void
      xgemv(const char         transA,
            const unsigned int m,
            const unsigned int n,
            const double *     alpha,
            const double *     A,
            const unsigned int lda,
            const double *     x,
            const unsigned int incx,
            const double *     beta,
            double *           y,
            const unsigned int incy) const;

      void
      xgemv(const char         transA,
            const unsigned int m,
            const unsigned int n,
            const float *      alpha,
            const float *      A,
            const unsigned int lda,
            const float *      x,
            const unsigned int incx,
            const float *      beta,
            float *            y,
            const unsigned int incy) const;

      void
      xgemv(const char                  transA,
            const unsigned int          m,
            const unsigned int          n,
            const std::complex<double> *alpha,
            const std::complex<double> *A,
            const unsigned int          lda,
            const std::complex<double> *x,
            const unsigned int          incx,
            const std::complex<double> *beta,
            std::complex<double> *      y,
            const unsigned int          incy) const;

      void
      xgemv(const char                 transA,
            const unsigned int         m,
            const unsigned int         n,
            const std::complex<float> *alpha,
            const std::complex<float> *A,
            const unsigned int         lda,
            const std::complex<float> *x,
            const unsigned int         incx,
            const std::complex<float> *beta,
            std::complex<float> *      y,
            const unsigned int         incy) const;



      template <typename ValueType1, typename ValueType2>
      void
      xscal(ValueType1 *           x,
            const ValueType2       alpha,
            const dftfe::size_type n) const;



      // Real double Norm2
      void
      xnrm2(const unsigned int n,
            const double *     x,
            const unsigned int incx,
            const MPI_Comm &   mpi_communicator,
            double *           result) const;


      // Complex double Norm2
      void
      xnrm2(const unsigned int          n,
            const std::complex<double> *x,
            const unsigned int          incx,
            const MPI_Comm &            mpi_communicator,
            double *                    result) const;

      // Real dot product
      void
      xdot(const unsigned int N,
           const double *     X,
           const unsigned int INCX,
           const double *     Y,
           const unsigned int INCY,
           double *           result) const;

      //
      // Real dot product
      void
      xdot(const unsigned int N,
           const double *     X,
           const unsigned int INCX,
           const double *     Y,
           const unsigned int INCY,
           const MPI_Comm &   mpi_communicator,
           double *           result) const;

      // Complex dot product
      void
      xdot(const unsigned int          N,
           const std::complex<double> *X,
           const unsigned int          INCX,
           const std::complex<double> *Y,
           const unsigned int          INCY,
           std::complex<double> *      result) const;

      // Real double Ax+y
      void
      xaxpy(const unsigned int n,
            const double *     alpha,
            const double *     x,
            const unsigned int incx,
            double *           y,
            const unsigned int incy) const;

      // Complex double Ax+y
      void
      xaxpy(const unsigned int          n,
            const std::complex<double> *alpha,
            const std::complex<double> *x,
            const unsigned int          incx,
            std::complex<double> *      y,
            const unsigned int          incy) const;

      // Real copy of double data
      void
      xcopy(const unsigned int n,
            const double *     x,
            const unsigned int incx,
            double *           y,
            const unsigned int incy) const;

      // Real copy of double data
      void
      xcopy(const unsigned int n,
            double *           x,
            const unsigned int incx,
            float *            y,
            const unsigned int incy) const;

      // Complex double copy of data
      void
      xcopy(const unsigned int          n,
            const std::complex<double> *x,
            const unsigned int          incx,
            std::complex<double> *      y,
            const unsigned int          incy) const;

      // Real copy of float data
      void
      xcopy(const unsigned int n,
            const float *      x,
            const unsigned int incx,
            float *            y,
            const unsigned int incy) const;

      // Complex float copy of data
      void
      xcopy(const unsigned int         n,
            const std::complex<float> *x,
            const unsigned int         incx,
            std::complex<float> *      y,
            const unsigned int         incy) const;

      void
      xcopy(const unsigned int    n,
            std::complex<double> *x,
            const unsigned int    incx,
            std::complex<float> * y,
            const unsigned int    incy) const;

      // Real double symmetric matrix-vector product
      void
      xsymv(const char         UPLO,
            const unsigned int N,
            const double *     alpha,
            const double *     A,
            const unsigned int LDA,
            const double *     X,
            const unsigned int INCX,
            const double *     beta,
            double *           C,
            const unsigned int INCY) const;

      void
      xgemmBatched(const char         transA,
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
                   const int          batchCount) const;

      void
      xgemmBatched(const char                  transA,
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
                   const int                   batchCount) const;

      void
      xgemmBatched(const char         transA,
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
                   const int          batchCount) const;

      void
      xgemmBatched(const char                 transA,
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
                   const int                  batchCount) const;

      void
      xgemmStridedBatched(const char         transA,
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
                          const int          batchCount) const;

      void
      xgemmStridedBatched(const char                  transA,
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
                          const int                   batchCount) const;

      void
      xgemmStridedBatched(const char                 transA,
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
                          const int                  batchCount) const;

      void
      xgemmStridedBatched(const char         transA,
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
                          const int          batchCount) const;

      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyComplexArrToRealArrs(const dftfe::size_type  size,
                               const ValueTypeComplex *complexArr,
                               ValueTypeReal *         realArr,
                               ValueTypeReal *         imagArr);


      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyRealArrsToComplexArr(const dftfe::size_type size,
                               const ValueTypeReal *  realArr,
                               const ValueTypeReal *  imagArr,
                               ValueTypeComplex *     complexArr);

      template <typename ValueType1, typename ValueType2>
      void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const ValueType1 *     valueType1Arr,
                                       ValueType2 *           valueType2Arr);


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const ValueType1 *             copyFromVec,
        ValueType2 *                   copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);



      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const ValueType1 *             copyFromVecBlock,
        ValueType2 *                   copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const ValueType1 *     copyFromVec,
                                       ValueType2 *           copyToVec);


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyConstantStride(const dftfe::size_type blockSize,
                                const dftfe::size_type strideTo,
                                const dftfe::size_type strideFrom,
                                const dftfe::size_type numBlocks,
                                const dftfe::size_type startingToId,
                                const dftfe::size_type startingFromId,
                                const ValueType1 *     copyFromVec,
                                ValueType2 *           copyToVec);


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const ValueType1 *     copyFromVec,
                                         ValueType2 *           copyToVec);

      template <typename ValueType>
      void
      axpyStridedBlockAtomicAdd(const dftfe::size_type contiguousBlockSize,
                                const dftfe::size_type numContiguousBlocks,
                                const ValueType *      addFromVec,
                                ValueType *            addToVec,
                                const dftfe::global_size_type
                                  *addToVecStartingContiguousBlockIds) const;

      template <typename ValueType>
      void
      axpyStridedBlockAtomicAdd(const dftfe::size_type contiguousBlockSize,
                                const dftfe::size_type numContiguousBlocks,
                                const ValueType *      addFromVec,
                                double *               addToVecReal,
                                double *               addToVecImag,
                                const dftfe::global_size_type
                                  *addToVecStartingContiguousBlockIds) const;

      template <typename ValueType1, typename ValueType2>
      void
      stridedBlockScale(const dftfe::size_type contiguousBlockSize,
                        const dftfe::size_type numContiguousBlocks,
                        const ValueType1       a,
                        const ValueType1 *     s,
                        ValueType2 *           x);

      void
      add(double *               y,
          const double *         x,
          const double           alpha,
          const dftfe::size_type size);

      template <typename ValueType>
      void
      sadd(ValueType *            y,
           ValueType *            x,
           const ValueType        beta,
           const dftfe::size_type size);

      dftfe::utils::deviceBlasHandle_t &
      getDeviceBlasHandle();

    private:
#  ifdef DFTFE_WITH_DEVICE_AMD
      void
      initialize();
#  endif

      /// storage for deviceblas handle
      dftfe::utils::deviceBlasHandle_t d_deviceBlasHandle;
      dftfe::utils::deviceStream_t     d_streamId;

      dftfe::utils::deviceBlasStatus_t
      create();

      dftfe::utils::deviceBlasStatus_t
      destroy();

      dftfe::utils::deviceBlasStatus_t
      setStream(dftfe::utils::deviceStream_t streamId);

#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      dftfe::utils::deviceBlasStatus_t
      setMathMode(dftfe::utils::deviceBlasMath_t mathMode);
#  endif
    };
#endif

  } // end of namespace linearAlgebra

} // end of namespace dftfe


#endif // BLASWrapper_h
