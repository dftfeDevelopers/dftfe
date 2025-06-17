// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
#if defined(DFTFE_WITH_DEVICE)
#  include "Exceptions.h"
#endif
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

      template <typename ValueType>
      void
      hadamardProduct(const dftfe::uInt m,
                      const ValueType  *X,
                      const ValueType  *Y,
                      ValueType        *output);

      template <typename ValueType>
      void
      hadamardProductWithConj(const dftfe::uInt m,
                              const ValueType  *X,
                              const ValueType  *Y,
                              ValueType        *output);

      // Real-Single Precision GEMM
      void
      xgemm(const char        transA,
            const char        transB,
            const dftfe::uInt m,
            const dftfe::uInt n,
            const dftfe::uInt k,
            const float      *alpha,
            const float      *A,
            const dftfe::uInt lda,
            const float      *B,
            const dftfe::uInt ldb,
            const float      *beta,
            float            *C,
            const dftfe::uInt ldc);
      // Complex-Single Precision GEMM
      void
      xgemm(const char                 transA,
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
            const dftfe::uInt          ldc);

      // Real-double precison GEMM
      void
      xgemm(const char        transA,
            const char        transB,
            const dftfe::uInt m,
            const dftfe::uInt n,
            const dftfe::uInt k,
            const double     *alpha,
            const double     *A,
            const dftfe::uInt lda,
            const double     *B,
            const dftfe::uInt ldb,
            const double     *beta,
            double           *C,
            const dftfe::uInt ldc);


      // Complex-double precision GEMM
      void
      xgemm(const char                  transA,
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
            const dftfe::uInt           ldc);

      void
      xgemv(const char        transA,
            const dftfe::uInt m,
            const dftfe::uInt n,
            const double     *alpha,
            const double     *A,
            const dftfe::uInt lda,
            const double     *x,
            const dftfe::uInt incx,
            const double     *beta,
            double           *y,
            const dftfe::uInt incy);

      void
      xgemv(const char        transA,
            const dftfe::uInt m,
            const dftfe::uInt n,
            const float      *alpha,
            const float      *A,
            const dftfe::uInt lda,
            const float      *x,
            const dftfe::uInt incx,
            const float      *beta,
            float            *y,
            const dftfe::uInt incy);

      void
      xgemv(const char                  transA,
            const dftfe::uInt           m,
            const dftfe::uInt           n,
            const std::complex<double> *alpha,
            const std::complex<double> *A,
            const dftfe::uInt           lda,
            const std::complex<double> *x,
            const dftfe::uInt           incx,
            const std::complex<double> *beta,
            std::complex<double>       *y,
            const dftfe::uInt           incy);

      void
      xgemv(const char                 transA,
            const dftfe::uInt          m,
            const dftfe::uInt          n,
            const std::complex<float> *alpha,
            const std::complex<float> *A,
            const dftfe::uInt          lda,
            const std::complex<float> *x,
            const dftfe::uInt          incx,
            const std::complex<float> *beta,
            std::complex<float>       *y,
            const dftfe::uInt          incy);


      template <typename ValueType1, typename ValueType2>
      void
      xscal(ValueType1 *x, const ValueType2 alpha, const dftfe::uInt n);

      // Brief
      //      for ( i = 0  i < numContiguousBlocks; i ++)
      //        {
      //          for( j = 0 ; j < contiguousBlockSize; j++)
      //            {
      //              output[j] += input1[i*contiguousBlockSize+j] *
      //              input2[i*contiguousBlockSize+j];
      //            }
      //        }
      template <typename ValueType>
      void
      addVecOverContinuousIndex(const dftfe::uInt numContiguousBlocks,
                                const dftfe::uInt contiguousBlockSize,
                                const ValueType  *input1,
                                const ValueType  *input2,
                                ValueType        *output);

      // Real-Float scaling of Real-vector


      // Real double Norm2
      void
      xnrm2(const dftfe::uInt n,
            const double     *x,
            const dftfe::uInt incx,
            const MPI_Comm   &mpi_communicator,
            double           *result);


      // Comples double Norm2
      void
      xnrm2(const dftfe::uInt           n,
            const std::complex<double> *x,
            const dftfe::uInt           incx,
            const MPI_Comm             &mpi_communicator,
            double                     *result);
      // Real dot product
      void
      xdot(const dftfe::uInt N,
           const double     *X,
           const dftfe::uInt INCX,
           const double     *Y,
           const dftfe::uInt INCY,
           double           *result);
      // Real dot product
      void
      xdot(const dftfe::uInt N,
           const float      *X,
           const dftfe::uInt INCX,
           const float      *Y,
           const dftfe::uInt INCY,
           float            *result);
      // Real dot proeuct with all Reduce call
      void
      xdot(const dftfe::uInt N,
           const double     *X,
           const dftfe::uInt INCX,
           const double     *Y,
           const dftfe::uInt INCY,
           const MPI_Comm   &mpi_communicator,
           double           *result);

      // Complex dot product
      void
      xdot(const dftfe::uInt           N,
           const std::complex<double> *X,
           const dftfe::uInt           INCX,
           const std::complex<double> *Y,
           const dftfe::uInt           INCY,
           std::complex<double>       *result);
      // Complex dot product
      void
      xdot(const dftfe::uInt          N,
           const std::complex<float> *X,
           const dftfe::uInt          INCX,
           const std::complex<float> *Y,
           const dftfe::uInt          INCY,
           std::complex<float>       *result);
      // Complex dot proeuct with all Reduce call
      void
      xdot(const dftfe::uInt           N,
           const std::complex<double> *X,
           const dftfe::uInt           INCX,
           const std::complex<double> *Y,
           const dftfe::uInt           INCY,
           const MPI_Comm             &mpi_communicator,
           std::complex<double>       *result);


      // MultiVector Real dot product
      template <typename ValueType>
      void
      MultiVectorXDot(const dftfe::uInt contiguousBlockSize,
                      const dftfe::uInt numContiguousBlocks,
                      const ValueType  *X,
                      const ValueType  *Y,
                      const ValueType  *onesVec,
                      ValueType        *tempVector,
                      ValueType        *tempResults,
                      ValueType        *result);

      // MultiVector Real dot product with all Reduce call
      template <typename ValueType>
      void
      MultiVectorXDot(const dftfe::uInt contiguousBlockSize,
                      const dftfe::uInt numContiguousBlocks,
                      const ValueType  *X,
                      const ValueType  *Y,
                      const ValueType  *onesVec,
                      ValueType        *tempVector,
                      ValueType        *tempResults,
                      const MPI_Comm   &mpi_communicator,
                      ValueType        *result);


      // Real double Ax+y
      void
      xaxpy(const dftfe::uInt n,
            const double     *alpha,
            const double     *x,
            const dftfe::uInt incx,
            double           *y,
            const dftfe::uInt incy);

      // Complex double Ax+y
      void
      xaxpy(const dftfe::uInt           n,
            const std::complex<double> *alpha,
            const std::complex<double> *x,
            const dftfe::uInt           incx,
            std::complex<double>       *y,
            const dftfe::uInt           incy);

      // Real float Ax+y
      void
      xaxpy(const dftfe::uInt n,
            const float      *alpha,
            const float      *x,
            const dftfe::uInt incx,
            float            *y,
            const dftfe::uInt incy);

      // Complex double Ax+y
      void
      xaxpy(const dftfe::uInt          n,
            const std::complex<float> *alpha,
            const std::complex<float> *x,
            const dftfe::uInt          incx,
            std::complex<float>       *y,
            const dftfe::uInt          incy);

      // Real copy of double data
      void
      xcopy(const dftfe::uInt n,
            const double     *x,
            const dftfe::uInt incx,
            double           *y,
            const dftfe::uInt incy);

      // Complex double copy of data
      void
      xcopy(const dftfe::uInt           n,
            const std::complex<double> *x,
            const dftfe::uInt           incx,
            std::complex<double>       *y,
            const dftfe::uInt           incy);

      // Real copy of float data
      void
      xcopy(const dftfe::uInt n,
            const float      *x,
            const dftfe::uInt incx,
            float            *y,
            const dftfe::uInt incy);

      // Complex float copy of data
      void
      xcopy(const dftfe::uInt          n,
            const std::complex<float> *x,
            const dftfe::uInt          incx,
            std::complex<float>       *y,
            const dftfe::uInt          incy);

      // Real double symmetric matrix-vector product
      void
      xsymv(const char        UPLO,
            const dftfe::uInt N,
            const double     *alpha,
            const double     *A,
            const dftfe::uInt LDA,
            const double     *X,
            const dftfe::uInt INCX,
            const double     *beta,
            double           *C,
            const dftfe::uInt INCY);

      void
      xgemmBatched(const char        transA,
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
                   const dftfe::Int  batchCount);

      void
      xgemmBatched(const char                  transA,
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
                   const dftfe::Int            batchCount);


      void
      xgemmBatched(const char        transA,
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
                   const dftfe::Int  batchCount);

      void
      xgemmBatched(const char                 transA,
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
                   const dftfe::Int           batchCount);


      void
      xgemmStridedBatched(const char        transA,
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
                          const dftfe::Int  batchCount);

      void
      xgemmStridedBatched(const char                  transA,
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
                          const dftfe::Int            batchCount);

      void
      xgemmStridedBatched(const char                 transA,
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
                          const dftfe::Int           batchCount);

      void
      xgemmStridedBatched(const char        transA,
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
                          const dftfe::Int  batchCount);

      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyComplexArrToRealArrs(const dftfe::uInt       size,
                               const ValueTypeComplex *complexArr,
                               ValueTypeReal          *realArr,
                               ValueTypeReal          *imagArr);


      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyRealArrsToComplexArr(const dftfe::uInt    size,
                               const ValueTypeReal *realArr,
                               const ValueTypeReal *imagArr,
                               ValueTypeComplex    *complexArr);

      template <typename ValueType1, typename ValueType2>
      void
      copyValueType1ArrToValueType2Arr(const dftfe::uInt size,
                                       const ValueType1 *valueType1Arr,
                                       ValueType2       *valueType2Arr);


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyToBlock(
        const dftfe::uInt  contiguousBlockSize,
        const dftfe::uInt  numContiguousBlocks,
        const ValueType1  *copyFromVec,
        ValueType2        *copyToVecBlock,
        const dftfe::uInt *copyFromVecStartingContiguousBlockIds);


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyToBlock(
        const dftfe::uInt  contiguousBlockSize,
        const dftfe::uInt  numContiguousBlocks,
        const dftfe::uInt  startingVecId,
        const ValueType1  *copyFromVec,
        ValueType2        *copyToVecBlock,
        const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyFromBlock(
        const dftfe::uInt  contiguousBlockSize,
        const dftfe::uInt  numContiguousBlocks,
        const ValueType1  *copyFromVecBlock,
        ValueType2        *copyToVec,
        const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyToBlockConstantStride(const dftfe::uInt blockSizeTo,
                                       const dftfe::uInt blockSizeFrom,
                                       const dftfe::uInt numBlocks,
                                       const dftfe::uInt startingId,
                                       const ValueType1 *copyFromVec,
                                       ValueType2       *copyToVec);


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyConstantStride(const dftfe::uInt blockSize,
                                const dftfe::uInt strideTo,
                                const dftfe::uInt strideFrom,
                                const dftfe::uInt numBlocks,
                                const dftfe::uInt startingToId,
                                const dftfe::uInt startingFromId,
                                const ValueType1 *copyFromVec,
                                ValueType2       *copyToVec);


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyFromBlockConstantStride(const dftfe::uInt blockSizeTo,
                                         const dftfe::uInt blockSizeFrom,
                                         const dftfe::uInt numBlocks,
                                         const dftfe::uInt startingId,
                                         const ValueType1 *copyFromVec,
                                         ValueType2       *copyToVec);

      template <typename ValueType1, typename ValueType2>
      void
      stridedBlockAxpy(const dftfe::uInt contiguousBlockSize,
                       const dftfe::uInt numContiguousBlocks,
                       const ValueType1 *addFromVec,
                       const ValueType2 *scalingVector,
                       const ValueType2  a,
                       ValueType1       *addToVec);


      template <typename ValueType1, typename ValueType2>
      void
      stridedBlockAxpBy(const dftfe::uInt contiguousBlockSize,
                        const dftfe::uInt numContiguousBlocks,
                        const ValueType1 *addFromVec,
                        const ValueType2 *scalingVector,
                        const ValueType2  a,
                        const ValueType2  b,
                        ValueType1       *addToVec);
      template <typename ValueType1, typename ValueType2>
      void
      axpby(const dftfe::uInt n,
            const ValueType2  alpha,
            const ValueType1 *x,
            const ValueType2  beta,
            ValueType1       *y);
      template <typename ValueType0,
                typename ValueType1,
                typename ValueType2,
                typename ValueType3,
                typename ValueType4>
      void
      ApaBD(const dftfe::uInt m,
            const dftfe::uInt n,
            const ValueType0  alpha,
            const ValueType1 *A,
            const ValueType2 *B,
            const ValueType3 *D,
            ValueType4       *C);

      template <typename ValueType>
      void
      axpyStridedBlockAtomicAdd(
        const dftfe::uInt  contiguousBlockSize,
        const dftfe::uInt  numContiguousBlocks,
        const ValueType   *addFromVec,
        ValueType         *addToVec,
        const dftfe::uInt *addToVecStartingContiguousBlockIds);

      template <typename ValueType1, typename ValueType2, typename ValueType3>
      void
      axpyStridedBlockAtomicAdd(
        const dftfe::uInt  contiguousBlockSize,
        const dftfe::uInt  numContiguousBlocks,
        const ValueType1   a,
        const ValueType1  *s,
        const ValueType2  *addFromVec,
        ValueType3        *addToVec,
        const dftfe::uInt *addToVecStartingContiguousBlockIds);
      template <typename ValueType1, typename ValueType2, typename ValueType3>
      void
      axpyStridedBlockAtomicAdd(
        const dftfe::uInt  contiguousBlockSize,
        const dftfe::uInt  numContiguousBlocks,
        const ValueType1   a,
        const ValueType2  *addFromVec,
        ValueType3        *addToVec,
        const dftfe::uInt *addToVecStartingContiguousBlockIds);

      template <typename ValueType1, typename ValueType2>
      void
      stridedBlockScale(const dftfe::uInt contiguousBlockSize,
                        const dftfe::uInt numContiguousBlocks,
                        const ValueType1  a,
                        const ValueType1 *s,
                        ValueType2       *x);

      template <typename ValueType1, typename ValueType2>
      void
      stridedBlockScaleCopy(
        const dftfe::uInt  contiguousBlockSize,
        const dftfe::uInt  numContiguousBlocks,
        const ValueType1   a,
        const ValueType1  *s,
        const ValueType2  *copyFromVec,
        ValueType2        *copyToVecBlock,
        const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

      template <typename ValueType>
      void
      stridedBlockScaleColumnWise(const dftfe::uInt contiguousBlockSize,
                                  const dftfe::uInt numContiguousBlocks,
                                  const ValueType  *beta,
                                  ValueType        *x);

      template <typename ValueType>
      void
      stridedBlockScaleAndAddColumnWise(const dftfe::uInt contiguousBlockSize,
                                        const dftfe::uInt numContiguousBlocks,
                                        const ValueType  *x,
                                        const ValueType  *beta,
                                        ValueType        *y);

      template <typename ValueType>
      void
      stridedBlockScaleAndAddTwoVecColumnWise(
        const dftfe::uInt contiguousBlockSize,
        const dftfe::uInt numContiguousBlocks,
        const ValueType  *x,
        const ValueType  *alpha,
        const ValueType  *y,
        const ValueType  *beta,
        ValueType        *z);

      template <typename ValueType1, typename ValueType2>
      void
      rightDiagonalScale(const dftfe::uInt numberofVectors,
                         const dftfe::uInt sizeOfVector,
                         ValueType1       *X,
                         ValueType2       *D);

    private:
    };
#if defined(DFTFE_WITH_DEVICE)
#  include "Exceptions.h"
    enum class tensorOpDataType
    {
      fp32,
      tf32,
      bf16,
      fp16
    };

    template <>
    class BLASWrapper<dftfe::utils::MemorySpace::DEVICE>
    {
    public:
      BLASWrapper();

      template <typename ValueType1, typename ValueType2>
      static void
      copyValueType1ArrToValueType2ArrDeviceCall(
        const dftfe::uInt            size,
        const ValueType1            *valueType1Arr,
        ValueType2                  *valueType2Arr,
        dftfe::utils::deviceStream_t streamId = dftfe::utils::defaultStream);

      template <typename ValueType>
      void
      hadamardProduct(const dftfe::uInt m,
                      const ValueType  *X,
                      const ValueType  *Y,
                      ValueType        *output);

      template <typename ValueType>
      void
      hadamardProductWithConj(const dftfe::uInt m,
                              const ValueType  *X,
                              const ValueType  *Y,
                              ValueType        *output);

      // Real-Single Precision GEMM
      void
      xgemm(const char        transA,
            const char        transB,
            const dftfe::uInt m,
            const dftfe::uInt n,
            const dftfe::uInt k,
            const float      *alpha,
            const float      *A,
            const dftfe::uInt lda,
            const float      *B,
            const dftfe::uInt ldb,
            const float      *beta,
            float            *C,
            const dftfe::uInt ldc);
      // Complex-Single Precision GEMM
      void
      xgemm(const char                 transA,
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
            const dftfe::uInt          ldc);

      // Real-double precison GEMM
      void
      xgemm(const char        transA,
            const char        transB,
            const dftfe::uInt m,
            const dftfe::uInt n,
            const dftfe::uInt k,
            const double     *alpha,
            const double     *A,
            const dftfe::uInt lda,
            const double     *B,
            const dftfe::uInt ldb,
            const double     *beta,
            double           *C,
            const dftfe::uInt ldc);


      // Complex-double precision GEMM
      void
      xgemm(const char                  transA,
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
            const dftfe::uInt           ldc);


      void
      xgemv(const char        transA,
            const dftfe::uInt m,
            const dftfe::uInt n,
            const double     *alpha,
            const double     *A,
            const dftfe::uInt lda,
            const double     *x,
            const dftfe::uInt incx,
            const double     *beta,
            double           *y,
            const dftfe::uInt incy);

      void
      xgemv(const char        transA,
            const dftfe::uInt m,
            const dftfe::uInt n,
            const float      *alpha,
            const float      *A,
            const dftfe::uInt lda,
            const float      *x,
            const dftfe::uInt incx,
            const float      *beta,
            float            *y,
            const dftfe::uInt incy);

      void
      xgemv(const char                  transA,
            const dftfe::uInt           m,
            const dftfe::uInt           n,
            const std::complex<double> *alpha,
            const std::complex<double> *A,
            const dftfe::uInt           lda,
            const std::complex<double> *x,
            const dftfe::uInt           incx,
            const std::complex<double> *beta,
            std::complex<double>       *y,
            const dftfe::uInt           incy);

      void
      xgemv(const char                 transA,
            const dftfe::uInt          m,
            const dftfe::uInt          n,
            const std::complex<float> *alpha,
            const std::complex<float> *A,
            const dftfe::uInt          lda,
            const std::complex<float> *x,
            const dftfe::uInt          incx,
            const std::complex<float> *beta,
            std::complex<float>       *y,
            const dftfe::uInt          incy);

      template <typename ValueType>
      void
      addVecOverContinuousIndex(const dftfe::uInt numContiguousBlocks,
                                const dftfe::uInt contiguousBlockSize,
                                const ValueType  *input1,
                                const ValueType  *input2,
                                ValueType        *output);



      template <typename ValueType1, typename ValueType2>
      void
      xscal(ValueType1 *x, const ValueType2 alpha, const dftfe::uInt n);



      // Real double Norm2
      void
      xnrm2(const dftfe::uInt n,
            const double     *x,
            const dftfe::uInt incx,
            const MPI_Comm   &mpi_communicator,
            double           *result);


      // Complex double Norm2
      void
      xnrm2(const dftfe::uInt           n,
            const std::complex<double> *x,
            const dftfe::uInt           incx,
            const MPI_Comm             &mpi_communicator,
            double                     *result);

      // Real dot product
      void
      xdot(const dftfe::uInt N,
           const double     *X,
           const dftfe::uInt INCX,
           const double     *Y,
           const dftfe::uInt INCY,
           double           *result);
      // Real dot product
      void
      xdot(const dftfe::uInt N,
           const float      *X,
           const dftfe::uInt INCX,
           const float      *Y,
           const dftfe::uInt INCY,
           float            *result);
      //
      // Real dot product
      void
      xdot(const dftfe::uInt N,
           const double     *X,
           const dftfe::uInt INCX,
           const double     *Y,
           const dftfe::uInt INCY,
           const MPI_Comm   &mpi_communicator,
           double           *result);

      // Complex dot product
      void
      xdot(const dftfe::uInt           N,
           const std::complex<double> *X,
           const dftfe::uInt           INCX,
           const std::complex<double> *Y,
           const dftfe::uInt           INCY,
           std::complex<double>       *result);
      // Complex dot product
      void
      xdot(const dftfe::uInt          N,
           const std::complex<float> *X,
           const dftfe::uInt          INCX,
           const std::complex<float> *Y,
           const dftfe::uInt          INCY,
           std::complex<float>       *result);
      // Complex dot product
      void
      xdot(const dftfe::uInt           N,
           const std::complex<double> *X,
           const dftfe::uInt           INCX,
           const std::complex<double> *Y,
           const dftfe::uInt           INCY,
           const MPI_Comm             &mpi_communicator,
           std::complex<double>       *result);


      template <typename ValueType>
      void
      MultiVectorXDot(const dftfe::uInt contiguousBlockSize,
                      const dftfe::uInt numContiguousBlocks,
                      const ValueType  *X,
                      const ValueType  *Y,
                      const ValueType  *onesVec,
                      ValueType        *tempVector,
                      ValueType        *tempResults,
                      ValueType        *result);

      template <typename ValueType>
      void
      MultiVectorXDot(const dftfe::uInt contiguousBlockSize,
                      const dftfe::uInt numContiguousBlocks,
                      const ValueType  *X,
                      const ValueType  *Y,
                      const ValueType  *onesVec,
                      ValueType        *tempVector,
                      ValueType        *tempResults,
                      const MPI_Comm   &mpi_communicator,
                      ValueType        *result);

      // Real double Ax+y
      void
      xaxpy(const dftfe::uInt n,
            const double     *alpha,
            const double     *x,
            const dftfe::uInt incx,
            double           *y,
            const dftfe::uInt incy);

      // Complex double Ax+y
      void
      xaxpy(const dftfe::uInt           n,
            const std::complex<double> *alpha,
            const std::complex<double> *x,
            const dftfe::uInt           incx,
            std::complex<double>       *y,
            const dftfe::uInt           incy);

      // Real copy of double data
      void
      xcopy(const dftfe::uInt n,
            const double     *x,
            const dftfe::uInt incx,
            double           *y,
            const dftfe::uInt incy);

      // Complex double copy of data
      void
      xcopy(const dftfe::uInt           n,
            const std::complex<double> *x,
            const dftfe::uInt           incx,
            std::complex<double>       *y,
            const dftfe::uInt           incy);

      // Real copy of float data
      void
      xcopy(const dftfe::uInt n,
            const float      *x,
            const dftfe::uInt incx,
            float            *y,
            const dftfe::uInt incy);

      // Complex float copy of data
      void
      xcopy(const dftfe::uInt          n,
            const std::complex<float> *x,
            const dftfe::uInt          incx,
            std::complex<float>       *y,
            const dftfe::uInt          incy);

      // Real double symmetric matrix-vector product
      void
      xsymv(const char        UPLO,
            const dftfe::uInt N,
            const double     *alpha,
            const double     *A,
            const dftfe::uInt LDA,
            const double     *X,
            const dftfe::uInt INCX,
            const double     *beta,
            double           *C,
            const dftfe::uInt INCY);

      void
      xgemmBatched(const char        transA,
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
                   const dftfe::Int  batchCount);

      void
      xgemmBatched(const char                  transA,
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
                   const dftfe::Int            batchCount);

      void
      xgemmBatched(const char        transA,
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
                   const dftfe::Int  batchCount);

      void
      xgemmBatched(const char                 transA,
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
                   const dftfe::Int           batchCount);

      void
      xgemmStridedBatched(const char        transA,
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
                          const dftfe::Int  batchCount);

      void
      xgemmStridedBatched(const char                  transA,
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
                          const dftfe::Int            batchCount);

      void
      xgemmStridedBatched(const char                 transA,
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
                          const dftfe::Int           batchCount);

      void
      xgemmStridedBatched(const char        transA,
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
                          const dftfe::Int  batchCount);

      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyComplexArrToRealArrs(const dftfe::uInt       size,
                               const ValueTypeComplex *complexArr,
                               ValueTypeReal          *realArr,
                               ValueTypeReal          *imagArr);


      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyRealArrsToComplexArr(const dftfe::uInt    size,
                               const ValueTypeReal *realArr,
                               const ValueTypeReal *imagArr,
                               ValueTypeComplex    *complexArr);

      template <typename ValueType1, typename ValueType2>
      void
      copyValueType1ArrToValueType2Arr(const dftfe::uInt size,
                                       const ValueType1 *valueType1Arr,
                                       ValueType2       *valueType2Arr);


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyToBlock(
        const dftfe::uInt  contiguousBlockSize,
        const dftfe::uInt  numContiguousBlocks,
        const ValueType1  *copyFromVec,
        ValueType2        *copyToVecBlock,
        const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyToBlock(
        const dftfe::uInt  contiguousBlockSize,
        const dftfe::uInt  numContiguousBlocks,
        const dftfe::uInt  startingVecId,
        const ValueType1  *copyFromVec,
        ValueType2        *copyToVecBlock,
        const dftfe::uInt *copyFromVecStartingContiguousBlockIds);


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyFromBlock(
        const dftfe::uInt  contiguousBlockSize,
        const dftfe::uInt  numContiguousBlocks,
        const ValueType1  *copyFromVecBlock,
        ValueType2        *copyToVec,
        const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyToBlockConstantStride(const dftfe::uInt blockSizeTo,
                                       const dftfe::uInt blockSizeFrom,
                                       const dftfe::uInt numBlocks,
                                       const dftfe::uInt startingId,
                                       const ValueType1 *copyFromVec,
                                       ValueType2       *copyToVec);


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyConstantStride(const dftfe::uInt blockSize,
                                const dftfe::uInt strideTo,
                                const dftfe::uInt strideFrom,
                                const dftfe::uInt numBlocks,
                                const dftfe::uInt startingToId,
                                const dftfe::uInt startingFromId,
                                const ValueType1 *copyFromVec,
                                ValueType2       *copyToVec);


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyFromBlockConstantStride(const dftfe::uInt blockSizeTo,
                                         const dftfe::uInt blockSizeFrom,
                                         const dftfe::uInt numBlocks,
                                         const dftfe::uInt startingId,
                                         const ValueType1 *copyFromVec,
                                         ValueType2       *copyToVec);
      template <typename ValueType1, typename ValueType2>
      void
      axpby(const dftfe::uInt n,
            const ValueType2  alpha,
            const ValueType1 *x,
            const ValueType2  beta,
            ValueType1       *y);

      template <typename ValueType1, typename ValueType2>
      void
      stridedBlockAxpy(const dftfe::uInt contiguousBlockSize,
                       const dftfe::uInt numContiguousBlocks,
                       const ValueType1 *addFromVec,
                       const ValueType2 *scalingVector,
                       const ValueType2  a,
                       ValueType1       *addToVec);
      template <typename ValueType1, typename ValueType2>
      void
      stridedBlockAxpBy(const dftfe::uInt contiguousBlockSize,
                        const dftfe::uInt numContiguousBlocks,
                        const ValueType1 *addFromVec,
                        const ValueType2 *scalingVector,
                        const ValueType2  a,
                        const ValueType2  b,
                        ValueType1       *addToVec);

      template <typename ValueType0,
                typename ValueType1,
                typename ValueType2,
                typename ValueType3,
                typename ValueType4>
      void
      ApaBD(const dftfe::uInt m,
            const dftfe::uInt n,
            const ValueType0  alpha,
            const ValueType1 *A,
            const ValueType2 *B,
            const ValueType3 *D,
            ValueType4       *C);


      template <typename ValueType>
      void
      axpyStridedBlockAtomicAdd(
        const dftfe::uInt  contiguousBlockSize,
        const dftfe::uInt  numContiguousBlocks,
        const ValueType   *addFromVec,
        ValueType         *addToVec,
        const dftfe::uInt *addToVecStartingContiguousBlockIds);

      template <typename ValueType1, typename ValueType2, typename ValueType3>
      void
      axpyStridedBlockAtomicAdd(
        const dftfe::uInt  contiguousBlockSize,
        const dftfe::uInt  numContiguousBlocks,
        const ValueType1   a,
        const ValueType1  *s,
        const ValueType2  *addFromVec,
        ValueType3        *addToVec,
        const dftfe::uInt *addToVecStartingContiguousBlockIds);
      template <typename ValueType1, typename ValueType2, typename ValueType3>
      void
      axpyStridedBlockAtomicAdd(
        const dftfe::uInt  contiguousBlockSize,
        const dftfe::uInt  numContiguousBlocks,
        const ValueType1   a,
        const ValueType2  *addFromVec,
        ValueType3        *addToVec,
        const dftfe::uInt *addToVecStartingContiguousBlockIds);

      template <typename ValueType1, typename ValueType2>
      void
      stridedBlockScale(const dftfe::uInt contiguousBlockSize,
                        const dftfe::uInt numContiguousBlocks,
                        const ValueType1  a,
                        const ValueType1 *s,
                        ValueType2       *x);
      template <typename ValueType1, typename ValueType2>
      void
      stridedBlockScaleCopy(
        const dftfe::uInt  contiguousBlockSize,
        const dftfe::uInt  numContiguousBlocks,
        const ValueType1   a,
        const ValueType1  *s,
        const ValueType2  *copyFromVec,
        ValueType2        *copyToVecBlock,
        const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

      template <typename ValueType>
      void
      stridedBlockScaleColumnWise(const dftfe::uInt contiguousBlockSize,
                                  const dftfe::uInt numContiguousBlocks,
                                  const ValueType  *beta,
                                  ValueType        *x);

      template <typename ValueType>
      void
      stridedBlockScaleAndAddColumnWise(const dftfe::uInt contiguousBlockSize,
                                        const dftfe::uInt numContiguousBlocks,
                                        const ValueType  *x,
                                        const ValueType  *beta,
                                        ValueType        *y);

      template <typename ValueType>
      void
      stridedBlockScaleAndAddTwoVecColumnWise(
        const dftfe::uInt contiguousBlockSize,
        const dftfe::uInt numContiguousBlocks,
        const ValueType  *x,
        const ValueType  *alpha,
        const ValueType  *y,
        const ValueType  *beta,
        ValueType        *z);

      template <typename ValueType1, typename ValueType2>
      void
      rightDiagonalScale(const dftfe::uInt numberofVectors,
                         const dftfe::uInt sizeOfVector,
                         ValueType1       *X,
                         ValueType2       *D);

      dftfe::utils::deviceBlasHandle_t &
      getDeviceBlasHandle();


      template <typename ValueType1, typename ValueType2>
      void
      copyBlockDiagonalValueType1OffDiagonalValueType2FromValueType1Arr(
        const dftfe::uInt B,
        const dftfe::uInt DRem,
        const dftfe::uInt D,
        const ValueType1 *valueType1SrcArray,
        ValueType1       *valueType1DstArray,
        ValueType2       *valueType2DstArray);

      void
      setTensorOpDataType(tensorOpDataType opType)
      {
        d_opType = opType;
      }

      static dftfe::utils::deviceBlasStatus_t
      setStream(dftfe::utils::deviceStream_t streamId);

      inline static dftfe::utils::deviceBlasHandle_t d_deviceBlasHandle;
      inline static dftfe::utils::deviceStream_t     d_streamId;

    private:
#  ifdef DFTFE_WITH_DEVICE_AMD
      void
      initialize();
#  endif

      /// storage for deviceblas handle
      tensorOpDataType d_opType;

      dftfe::utils::deviceBlasStatus_t
      create();

      dftfe::utils::deviceBlasStatus_t
      destroy();
    };
#endif

  } // end of namespace linearAlgebra

} // end of namespace dftfe


#endif // BLASWrapper_h
