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
#include <dftfe/BLASWrapper.h>
#include <dftfe/linearAlgebraOperations.h>
#include <dftfe/dftUtils.h>
namespace dftfe
{
  namespace linearAlgebra
  {
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::BLASWrapper()
    {}

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProduct(
      const dftfe::uInt m,
      const ValueType  *X,
      const ValueType  *Y,
      ValueType        *output)
    {
      for (dftfe::uInt i = 0; i < m; i++)
        {
          output[i] = X[i] * Y[i];
        }
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProductWithConj(
      const dftfe::uInt m,
      const ValueType  *X,
      const ValueType  *Y,
      ValueType        *output)
    {
      for (dftfe::uInt i = 0; i < m; i++)
        {
          output[i] = X[i] * Y[i];
        }
    }


    template <>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProductWithConj(
      const dftfe::uInt           m,
      const std::complex<double> *X,
      const std::complex<double> *Y,
      std::complex<double>       *output)
    {
      for (dftfe::uInt i = 0; i < m; i++)
        {
          output[i] = std::conj(X[i]) * Y[i];
        }
    }

    template <>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProductWithConj(
      const dftfe::uInt          m,
      const std::complex<float> *X,
      const std::complex<float> *Y,
      std::complex<float>       *output)
    {
      for (dftfe::uInt i = 0; i < m; i++)
        {
          output[i] = std::conj(X[i]) * Y[i];
        }
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemm(const char transA,
                                                        const char transB,
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
                                                        const dftfe::uInt ldc)
    {
      unsigned int mTmp   = m;
      unsigned int nTmp   = n;
      unsigned int kTmp   = k;
      unsigned int ldaTmp = lda;
      unsigned int ldbTmp = ldb;
      unsigned int ldcTmp = ldc;
      sgemm_(&transA,
             &transB,
             &mTmp,
             &nTmp,
             &kTmp,
             alpha,
             A,
             &ldaTmp,
             B,
             &ldbTmp,
             beta,
             C,
             &ldcTmp);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemm(const char transA,
                                                        const char transB,
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
                                                        const dftfe::uInt ldc)
    {
      unsigned int mTmp   = m;
      unsigned int nTmp   = n;
      unsigned int kTmp   = k;
      unsigned int ldaTmp = lda;
      unsigned int ldbTmp = ldb;
      unsigned int ldcTmp = ldc;
      dgemm_(&transA,
             &transB,
             &mTmp,
             &nTmp,
             &kTmp,
             alpha,
             A,
             &ldaTmp,
             B,
             &ldbTmp,
             beta,
             C,
             &ldcTmp);
    }


    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::addVecOverContinuousIndex(
      const dftfe::uInt numContiguousBlocks,
      const dftfe::uInt contiguousBlockSize,
      const ValueType  *input1,
      const ValueType  *input2,
      ValueType        *output)
    {
      for (dftfe::uInt iIndex = 0; iIndex < numContiguousBlocks; iIndex++)
        {
          for (dftfe::uInt jIndex = 0; jIndex < contiguousBlockSize; jIndex++)
            {
              output[iIndex] += input1[iIndex * contiguousBlockSize + jIndex] *
                                input2[iIndex * contiguousBlockSize + jIndex];
            }
        }
    }

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::addVecOverContinuousIndex(
      const dftfe::uInt numContiguousBlocks,
      const dftfe::uInt contiguousBlockSize,
      const double     *input1,
      const double     *input2,
      double           *output);
    template <typename ValueType0,
              typename ValueType1,
              typename ValueType2,
              typename ValueType3,
              typename ValueType4>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::ApaBD(const dftfe::uInt m,
                                                        const dftfe::uInt n,
                                                        const ValueType0  alpha,
                                                        const ValueType1 *A,
                                                        const ValueType2 *B,
                                                        const ValueType3 *D,
                                                        ValueType4       *C)
    {
      for (dftfe::uInt iRow = 0; iRow < m; ++iRow)
        {
          for (dftfe::uInt iCol = 0; iCol < n; ++iCol)
            {
              C[iCol + n * iRow] =
                A[iCol + n * iRow] + alpha * B[iCol + n * iRow] * D[iCol];
            }
        }
    }

    template <>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::ApaBD(
      const dftfe::uInt           m,
      const dftfe::uInt           n,
      const double                alpha,
      const std::complex<float>  *A,
      const std::complex<double> *B,
      const double               *D,
      std::complex<float>        *C)
    {
      for (dftfe::uInt iRow = 0; iRow < m; ++iRow)
        {
          for (dftfe::uInt iCol = 0; iCol < n; ++iCol)
            {
              C[iCol + n * iRow] = std::complex<double>(A[iCol + n * iRow]) +
                                   alpha * B[iCol + n * iRow] * D[iCol];
            }
        }
    }

    template <>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::ApaBD(
      const dftfe::uInt           m,
      const dftfe::uInt           n,
      const double                alpha,
      const std::complex<float>  *A,
      const std::complex<double> *B,
      const double               *D,
      std::complex<double>       *C)
    {
      for (dftfe::uInt iRow = 0; iRow < m; ++iRow)
        {
          for (dftfe::uInt iCol = 0; iCol < n; ++iCol)
            {
              C[iCol + n * iRow] = std::complex<double>(A[iCol + n * iRow]) +
                                   alpha * B[iCol + n * iRow] * D[iCol];
            }
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xscal(ValueType1       *x,
                                                        const ValueType2  alpha,
                                                        const dftfe::uInt n)
    {
      std::transform(x, x + n, x, [&alpha](auto &c) { return alpha * c; });
    }
    // for xscal
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xscal(double           *x,
                                                        const double      a,
                                                        const dftfe::uInt n);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xscal(float            *x,
                                                        const float       a,
                                                        const dftfe::uInt n);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xscal(
      std::complex<double>      *x,
      const std::complex<double> a,
      const dftfe::uInt          n);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xscal(std::complex<double> *x,
                                                        const double          a,
                                                        const dftfe::uInt n);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xscal(
      std::complex<float>      *x,
      const std::complex<float> a,
      const dftfe::uInt         n);

    // hadamard product
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProduct(
      const dftfe::uInt m,
      const double     *X,
      const double     *Y,
      double           *output);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProduct(
      const dftfe::uInt m,
      const float      *X,
      const float      *Y,
      float            *output);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProduct(
      const dftfe::uInt           m,
      const std::complex<double> *X,
      const std::complex<double> *Y,
      std::complex<double>       *output);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProduct(
      const dftfe::uInt          m,
      const std::complex<float> *X,
      const std::complex<float> *Y,
      std::complex<float>       *output);

    // hadamard product with conj
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProductWithConj(
      const dftfe::uInt m,
      const double     *X,
      const double     *Y,
      double           *output);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProductWithConj(
      const dftfe::uInt m,
      const float      *X,
      const float      *Y,
      float            *output);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProductWithConj(
      const dftfe::uInt           m,
      const std::complex<double> *X,
      const std::complex<double> *Y,
      std::complex<double>       *output);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::hadamardProductWithConj(
      const dftfe::uInt          m,
      const std::complex<float> *X,
      const std::complex<float> *Y,
      std::complex<float>       *output);


    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xcopy(const dftfe::uInt n,
                                                        const double     *x,
                                                        const dftfe::uInt incx,
                                                        double           *y,
                                                        const dftfe::uInt incy)
    {
      unsigned int nTmp    = n;
      unsigned int incxTmp = incx;
      unsigned int incyTmp = incy;
      dcopy_(&nTmp, x, &incxTmp, y, &incyTmp);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xcopy(const dftfe::uInt n,
                                                        const float      *x,
                                                        const dftfe::uInt incx,
                                                        float            *y,
                                                        const dftfe::uInt incy)
    {
      unsigned int nTmp    = n;
      unsigned int incxTmp = incx;
      unsigned int incyTmp = incy;
      scopy_(&nTmp, x, &incxTmp, y, &incyTmp);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemm(
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
      unsigned int mTmp   = m;
      unsigned int nTmp   = n;
      unsigned int kTmp   = k;
      unsigned int ldaTmp = lda;
      unsigned int ldbTmp = ldb;
      unsigned int ldcTmp = ldc;
      cgemm_(&transA,
             &transB,
             &mTmp,
             &nTmp,
             &kTmp,
             alpha,
             A,
             &ldaTmp,
             B,
             &ldbTmp,
             beta,
             C,
             &ldcTmp);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemm(
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
      unsigned int mTmp   = m;
      unsigned int nTmp   = n;
      unsigned int kTmp   = k;
      unsigned int ldaTmp = lda;
      unsigned int ldbTmp = ldb;
      unsigned int ldcTmp = ldc;
      zgemm_(&transA,
             &transB,
             &mTmp,
             &nTmp,
             &kTmp,
             alpha,
             A,
             &ldaTmp,
             B,
             &ldbTmp,
             beta,
             C,
             &ldcTmp);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemv(const char transA,
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
      unsigned int mTmp    = m;
      unsigned int nTmp    = n;
      unsigned int ldaTmp  = lda;
      unsigned int incxTmp = incx;
      unsigned int incyTmp = incy;
      dgemv_(&transA,
             &mTmp,
             &nTmp,
             alpha,
             A,
             &ldaTmp,
             x,
             &incxTmp,
             beta,
             y,
             &incyTmp);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemv(const char transA,
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
      unsigned int mTmp    = m;
      unsigned int nTmp    = n;
      unsigned int ldaTmp  = lda;
      unsigned int incxTmp = incx;
      unsigned int incyTmp = incy;
      sgemv_(&transA,
             &mTmp,
             &nTmp,
             alpha,
             A,
             &ldaTmp,
             x,
             &incxTmp,
             beta,
             y,
             &incyTmp);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemv(
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
      unsigned int mTmp    = m;
      unsigned int nTmp    = n;
      unsigned int ldaTmp  = lda;
      unsigned int incxTmp = incx;
      unsigned int incyTmp = incy;
      zgemv_(&transA,
             &mTmp,
             &nTmp,
             alpha,
             A,
             &ldaTmp,
             x,
             &incxTmp,
             beta,
             y,
             &incyTmp);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemv(
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
      unsigned int mTmp    = m;
      unsigned int nTmp    = n;
      unsigned int ldaTmp  = lda;
      unsigned int incxTmp = incx;
      unsigned int incyTmp = incy;
      cgemv_(&transA,
             &mTmp,
             &nTmp,
             alpha,
             A,
             &ldaTmp,
             x,
             &incxTmp,
             beta,
             y,
             &incyTmp);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xcopy(
      const dftfe::uInt           n,
      const std::complex<double> *x,
      const dftfe::uInt           incx,
      std::complex<double>       *y,
      const dftfe::uInt           incy)
    {
      unsigned int nTmp    = n;
      unsigned int incxTmp = incx;
      unsigned int incyTmp = incy;
      zcopy_(&nTmp, x, &incxTmp, y, &incyTmp);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xcopy(
      const dftfe::uInt          n,
      const std::complex<float> *x,
      const dftfe::uInt          incx,
      std::complex<float>       *y,
      const dftfe::uInt          incy)
    {
      unsigned int nTmp    = n;
      unsigned int incxTmp = incx;
      unsigned int incyTmp = incy;
      ccopy_(&nTmp, x, &incxTmp, y, &incyTmp);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xnrm2(
      const dftfe::uInt n,
      const double     *x,
      const dftfe::uInt incx,
      const MPI_Comm   &mpi_communicator,
      double           *result)
    {
      unsigned int nTmp        = n;
      unsigned int incxTmp     = incx;
      double       localresult = dnrm2_(&nTmp, x, &incxTmp);
      *result                  = 0.0;
      localresult *= localresult;
      MPI_Allreduce(
        &localresult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
      *result = std::sqrt(*result);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xnrm2(
      const dftfe::uInt           n,
      const std::complex<double> *x,
      const dftfe::uInt           incx,
      const MPI_Comm             &mpi_communicator,
      double                     *result)
    {
      unsigned int nTmp        = n;
      unsigned int incxTmp     = incx;
      double       localresult = dznrm2_(&nTmp, x, &incxTmp);
      *result                  = 0.0;
      localresult *= localresult;
      MPI_Allreduce(
        &localresult, result, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
      *result = std::sqrt(*result);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xdot(const dftfe::uInt N,
                                                       const double     *X,
                                                       const dftfe::uInt INCX,
                                                       const double     *Y,
                                                       const dftfe::uInt INCY,
                                                       double           *result)
    {
      unsigned int nTmp    = N;
      unsigned int incxTmp = INCX;
      unsigned int incyTmp = INCY;
      *result              = ddot_(&nTmp, X, &incxTmp, Y, &incyTmp);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xdot(const dftfe::uInt N,
                                                       const float      *X,
                                                       const dftfe::uInt INCX,
                                                       const float      *Y,
                                                       const dftfe::uInt INCY,
                                                       float            *result)
    {
      unsigned int nTmp    = N;
      unsigned int incxTmp = INCX;
      unsigned int incyTmp = INCY;
      *result              = sdot_(&nTmp, X, &incxTmp, Y, &incyTmp);
    }
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xdot(
      const dftfe::uInt           N,
      const std::complex<double> *X,
      const dftfe::uInt           INCX,
      const std::complex<double> *Y,
      const dftfe::uInt           INCY,
      std::complex<double>       *result)
    {
      unsigned int nTmp    = N;
      unsigned int incxTmp = INCX;
      unsigned int incyTmp = INCY;
      *result              = zdotc_(&nTmp, X, &incxTmp, Y, &incyTmp);
    }
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xdot(
      const dftfe::uInt          N,
      const std::complex<float> *X,
      const dftfe::uInt          INCX,
      const std::complex<float> *Y,
      const dftfe::uInt          INCY,
      std::complex<float>       *result)
    {
      unsigned int nTmp    = N;
      unsigned int incxTmp = INCX;
      unsigned int incyTmp = INCY;
      *result              = cdotc_(&nTmp, X, &incxTmp, Y, &incyTmp);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xdot(
      const dftfe::uInt N,
      const double     *X,
      const dftfe::uInt INCX,
      const double     *Y,
      const dftfe::uInt INCY,
      const MPI_Comm   &mpi_communicator,
      double           *result)
    {
      double localResult   = 0.0;
      *result              = 0.0;
      unsigned int nTmp    = N;
      unsigned int incxTmp = INCX;
      unsigned int incyTmp = INCY;
      localResult          = ddot_(&nTmp, X, &incxTmp, Y, &incyTmp);
      MPI_Allreduce(&localResult,
                    result,
                    1,
                    dataTypes::mpi_type_id(result),
                    MPI_SUM,
                    mpi_communicator);
    }
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xdot(
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

      ValueType    alpha                  = 1.0;
      ValueType    beta                   = 0.0;
      unsigned int numVec                 = 1;
      unsigned int contiguousBlockSizeTmp = contiguousBlockSize;
      unsigned int numContiguousBlocksTmp = numContiguousBlocks;
      xgemm('N',
            'T',
            numVec,
            contiguousBlockSizeTmp,
            numContiguousBlocksTmp,
            &alpha,
            onesVec,
            numVec,
            tempVector,
            contiguousBlockSizeTmp,
            &beta,
            result,
            numVec);
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::MultiVectorXDot(
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


    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xaxpy(const dftfe::uInt n,
                                                        const double     *alpha,
                                                        const double     *x,
                                                        const dftfe::uInt incx,
                                                        double           *y,
                                                        const dftfe::uInt incy)
    {
      unsigned int nTmp    = n;
      unsigned int incxTmp = incx;
      unsigned int incyTmp = incy;
      daxpy_(&nTmp, alpha, x, &incxTmp, y, &incyTmp);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xaxpy(const dftfe::uInt n,
                                                        const float      *alpha,
                                                        const float      *x,
                                                        const dftfe::uInt incx,
                                                        float            *y,
                                                        const dftfe::uInt incy)
    {
      unsigned int nTmp    = n;
      unsigned int incxTmp = incx;
      unsigned int incyTmp = incy;
      saxpy_(&nTmp, alpha, x, &incxTmp, y, &incyTmp);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xaxpy(
      const dftfe::uInt           n,
      const std::complex<double> *alpha,
      const std::complex<double> *x,
      const dftfe::uInt           incx,
      std::complex<double>       *y,
      const dftfe::uInt           incy)
    {
      unsigned int nTmp    = n;
      unsigned int incxTmp = incx;
      unsigned int incyTmp = incy;
      zaxpy_(&nTmp, alpha, x, &incxTmp, y, &incyTmp);
    }


    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xaxpy(
      const dftfe::uInt          n,
      const std::complex<float> *alpha,
      const std::complex<float> *x,
      const dftfe::uInt          incx,
      std::complex<float>       *y,
      const dftfe::uInt          incy)
    {
      unsigned int nTmp    = n;
      unsigned int incxTmp = incx;
      unsigned int incyTmp = incy;
      caxpy_(&nTmp, alpha, x, &incxTmp, y, &incyTmp);
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const ValueType   *addFromVec,
      ValueType         *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      for (dftfe::uInt iBlock = 0; iBlock < numContiguousBlocks; ++iBlock)
        std::transform(addFromVec + iBlock * contiguousBlockSize,
                       addFromVec + (iBlock + 1) * contiguousBlockSize,
                       addToVec + addToVecStartingContiguousBlockIds[iBlock],
                       addToVec + addToVecStartingContiguousBlockIds[iBlock],
                       std::plus<>{});
    }

    template <typename ValueType1, typename ValueType2, typename ValueType3>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const ValueType1   a,
      const ValueType1  *s,
      const ValueType2  *addFromVec,
      ValueType3        *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      for (dftfe::uInt iBlock = 0; iBlock < numContiguousBlocks; ++iBlock)
        {
          ValueType1 coeff = a * s[iBlock];
          std::transform(addFromVec + iBlock * contiguousBlockSize,
                         addFromVec + (iBlock + 1) * contiguousBlockSize,
                         addToVec + addToVecStartingContiguousBlockIds[iBlock],
                         addToVec + addToVecStartingContiguousBlockIds[iBlock],
                         [&coeff](auto &p, auto &q) { return p * coeff + q; });
        }
    }

    template <typename ValueType1, typename ValueType2, typename ValueType3>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const ValueType1   a,
      const ValueType2  *addFromVec,
      ValueType3        *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      for (dftfe::uInt iBlock = 0; iBlock < numContiguousBlocks; ++iBlock)
        {
          std::transform(addFromVec + iBlock * contiguousBlockSize,
                         addFromVec + (iBlock + 1) * contiguousBlockSize,
                         addToVec + addToVecStartingContiguousBlockIds[iBlock],
                         addToVec + addToVecStartingContiguousBlockIds[iBlock],
                         [&a](auto &p, auto &q) { return p * a + q; });
        }
    }

    template <>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt          contiguousBlockSize,
      const dftfe::uInt          numContiguousBlocks,
      const double               a,
      const double              *s,
      const std::complex<float> *addFromVec,
      std::complex<float>       *addToVec,
      const dftfe::uInt         *addToVecStartingContiguousBlockIds)
    {
      for (dftfe::uInt iBlock = 0; iBlock < numContiguousBlocks; ++iBlock)
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

    template <>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt          contiguousBlockSize,
      const dftfe::uInt          numContiguousBlocks,
      const double               a,
      const std::complex<float> *addFromVec,
      std::complex<float>       *addToVec,
      const dftfe::uInt         *addToVecStartingContiguousBlockIds)
    {
      for (dftfe::uInt iBlock = 0; iBlock < numContiguousBlocks; ++iBlock)
        {
          std::transform(addFromVec + iBlock * contiguousBlockSize,
                         addFromVec + (iBlock + 1) * contiguousBlockSize,
                         addToVec + addToVecStartingContiguousBlockIds[iBlock],
                         addToVec + addToVecStartingContiguousBlockIds[iBlock],
                         [&a](auto &p, auto &q) {
                           return std::complex<double>(p) * a +
                                  std::complex<double>(q);
                         });
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpby(const dftfe::uInt n,
                                                        const ValueType2  alpha,
                                                        const ValueType1 *x,
                                                        const ValueType2  beta,
                                                        ValueType1       *y)
    {
      std::transform(x, x + n, y, y, [&alpha, &beta](auto &p, auto &q) {
        return alpha * p + beta * q;
      });
    }

    template <>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpby(
      const dftfe::uInt          n,
      const double               alpha,
      const std::complex<float> *x,
      const double               beta,
      std::complex<float>       *y)
    {
      std::transform(x, x + n, y, y, [&alpha, &beta](auto &p, auto &q) {
        return alpha * std::complex<double>(p) + beta * std::complex<double>(q);
      });
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xsymv(const char        UPLO,
                                                        const dftfe::uInt N,
                                                        const double     *alpha,
                                                        const double     *A,
                                                        const dftfe::uInt LDA,
                                                        const double     *X,
                                                        const dftfe::uInt INCX,
                                                        const double     *beta,
                                                        double           *C,
                                                        const dftfe::uInt INCY)
    {
      unsigned int nTmp    = N;
      unsigned int ldaTmp  = LDA;
      unsigned int incxTmp = INCX;
      unsigned int incyTmp = INCY;
      dsymv_(&UPLO, &nTmp, alpha, A, &ldaTmp, X, &incxTmp, beta, C, &incyTmp);
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemmBatched(
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
      for (dftfe::Int iBatch = 0; iBatch < batchCount; iBatch++)
        {
          xgemm(transA,
                transB,
                m,
                n,
                k,
                alpha,
                A[iBatch],
                lda,
                B[iBatch],
                ldb,
                beta,
                C[iBatch],
                ldc);
        }
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemmBatched(
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
      for (dftfe::Int iBatch = 0; iBatch < batchCount; iBatch++)
        {
          xgemm(transA,
                transB,
                m,
                n,
                k,
                alpha,
                A[iBatch],
                lda,
                B[iBatch],
                ldb,
                beta,
                C[iBatch],
                ldc);
        }
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemmBatched(
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
      for (dftfe::Int iBatch = 0; iBatch < batchCount; iBatch++)
        {
          xgemm(transA,
                transB,
                m,
                n,
                k,
                alpha,
                A[iBatch],
                lda,
                B[iBatch],
                ldb,
                beta,
                C[iBatch],
                ldc);
        }
    }
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemmBatched(
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
      for (dftfe::Int iBatch = 0; iBatch < batchCount; iBatch++)
        {
          xgemm(transA,
                transB,
                m,
                n,
                k,
                alpha,
                A[iBatch],
                lda,
                B[iBatch],
                ldb,
                beta,
                C[iBatch],
                ldc);
        }
    }

    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::xgemmStridedBatched(
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
      for (dftfe::Int iBatch = 0; iBatch < batchCount; iBatch++)
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
      for (dftfe::Int iBatch = 0; iBatch < batchCount; iBatch++)
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
      for (dftfe::Int iBatch = 0; iBatch < batchCount; iBatch++)
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
      for (dftfe::Int iBatch = 0; iBatch < batchCount; iBatch++)
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
      const dftfe::uInt       size,
      const ValueTypeComplex *complexArr,
      ValueTypeReal          *realArr,
      ValueTypeReal          *imagArr)
    {
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }



    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::copyRealArrsToComplexArr(
      const dftfe::uInt    size,
      const ValueTypeReal *realArr,
      const ValueTypeReal *imagArr,
      ValueTypeComplex    *complexArr)
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
      copyValueType1ArrToValueType2Arr(const dftfe::uInt size,
                                       const ValueType1 *valueType1Arr,
                                       ValueType2       *valueType2Arr)
    {
      for (dftfe::uInt i = 0; i < size; ++i)
        valueType2Arr[i] = valueType1Arr[i];
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyToBlock(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const ValueType1  *copyFromVec,
      ValueType2        *copyToVecBlock,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds)
    {
      for (dftfe::uInt iBlock = 0; iBlock < numContiguousBlocks; ++iBlock)
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
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyToBlock(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const dftfe::uInt  startingVecId,
      const ValueType1  *copyFromVec,
      ValueType2        *copyToVecBlock,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds)
    {
      for (dftfe::uInt iBlock = 0; iBlock < numContiguousBlocks; ++iBlock)
        {
          xcopy(contiguousBlockSize,
                copyFromVec + copyFromVecStartingContiguousBlockIds[iBlock] +
                  startingVecId,
                1,
                copyToVecBlock + iBlock * contiguousBlockSize,
                1);
        }
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyFromBlock(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const ValueType1  *copyFromVecBlock,
      ValueType2        *copyToVec,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds)
    {
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::uInt blockSizeTo,
                                       const dftfe::uInt blockSizeFrom,
                                       const dftfe::uInt numBlocks,
                                       const dftfe::uInt startingId,
                                       const ValueType1 *copyFromVec,
                                       ValueType2       *copyToVec)
    {
      for (dftfe::uInt iIndex = 0; iIndex < numBlocks; iIndex++)
        {
          for (dftfe::uInt jIndex = 0; jIndex < blockSizeTo; jIndex++)
            {
              copyToVec[iIndex * blockSizeTo + jIndex] =
                copyFromVec[iIndex * blockSizeFrom + startingId + jIndex];
            }
        }
    }

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::uInt blockSizeTo,
                                       const dftfe::uInt blockSizeFrom,
                                       const dftfe::uInt numBlocks,
                                       const dftfe::uInt startingId,
                                       const double     *copyFromVec,
                                       double           *copyToVec);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::uInt blockSizeTo,
                                       const dftfe::uInt blockSizeFrom,
                                       const dftfe::uInt numBlocks,
                                       const dftfe::uInt startingId,
                                       const std::complex<double> *copyFromVec,
                                       std::complex<double>       *copyToVec);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::uInt blockSizeTo,
                                       const dftfe::uInt blockSizeFrom,
                                       const dftfe::uInt numBlocks,
                                       const dftfe::uInt startingId,
                                       const float      *copyFromVec,
                                       float            *copyToVec);


    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::uInt          blockSizeTo,
                                       const dftfe::uInt          blockSizeFrom,
                                       const dftfe::uInt          numBlocks,
                                       const dftfe::uInt          startingId,
                                       const std::complex<float> *copyFromVec,
                                       std::complex<float>       *copyToVec);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::uInt blockSizeTo,
                                       const dftfe::uInt blockSizeFrom,
                                       const dftfe::uInt numBlocks,
                                       const dftfe::uInt startingId,
                                       const double     *copyFromVec,
                                       float            *copyToVec);
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::uInt blockSizeTo,
                                       const dftfe::uInt blockSizeFrom,
                                       const dftfe::uInt numBlocks,
                                       const dftfe::uInt startingId,
                                       const float      *copyFromVec,
                                       double           *copyToVec);
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::uInt blockSizeTo,
                                       const dftfe::uInt blockSizeFrom,
                                       const dftfe::uInt numBlocks,
                                       const dftfe::uInt startingId,
                                       const std::complex<double> *copyFromVec,
                                       std::complex<float>        *copyToVec);
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyToBlockConstantStride(const dftfe::uInt          blockSizeTo,
                                       const dftfe::uInt          blockSizeFrom,
                                       const dftfe::uInt          numBlocks,
                                       const dftfe::uInt          startingId,
                                       const std::complex<float> *copyFromVec,
                                       std::complex<double>      *copyToVec);

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyConstantStride(
      const dftfe::uInt blockSize,
      const dftfe::uInt strideTo,
      const dftfe::uInt strideFrom,
      const dftfe::uInt numBlocks,
      const dftfe::uInt startingToId,
      const dftfe::uInt startingFromId,
      const ValueType1 *copyFromVec,
      ValueType2       *copyToVec)
    {
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }


    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedCopyFromBlockConstantStride(const dftfe::uInt blockSizeTo,
                                         const dftfe::uInt blockSizeFrom,
                                         const dftfe::uInt numBlocks,
                                         const dftfe::uInt startingId,
                                         const ValueType1 *copyFromVec,
                                         ValueType2       *copyToVec)
    {
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
    }
    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScaleCopy(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const ValueType1   a,
      const ValueType1  *s,
      const ValueType2  *copyFromVec,
      ValueType2        *copyToVecBlock,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds)
    {
      for (dftfe::Int iBatch = 0; iBatch < numContiguousBlocks; iBatch++)
        {
          ValueType2 alpha = a * s[iBatch];
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
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const ValueType  *beta,
      ValueType        *x)
    {
      for (dftfe::uInt i = 0; i < numContiguousBlocks; i++)
        {
          for (dftfe::uInt j = 0; j < contiguousBlockSize; j++)
            {
              x[j + i * contiguousBlockSize] *= beta[j];
            }
        }
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddColumnWise(const dftfe::uInt contiguousBlockSize,
                                        const dftfe::uInt numContiguousBlocks,
                                        const ValueType  *x,
                                        const ValueType  *beta,
                                        ValueType        *y)
    {
      for (dftfe::uInt i = 0; i < numContiguousBlocks; i++)
        {
          for (dftfe::uInt j = 0; j < contiguousBlockSize; j++)
            {
              y[j + i * contiguousBlockSize] +=
                beta[j] * x[j + i * contiguousBlockSize];
            }
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockAxpy(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const ValueType1 *addFromVec,
      const ValueType2 *scalingVector,
      const ValueType2  a,
      ValueType1       *addToVec)
    {
      for (dftfe::uInt iBlock = 0; iBlock < numContiguousBlocks; ++iBlock)
        {
          ValueType2 coeff = a * scalingVector[iBlock];
          std::transform(addFromVec + iBlock * contiguousBlockSize,
                         addFromVec + (iBlock + 1) * contiguousBlockSize,
                         addToVec + iBlock * contiguousBlockSize,
                         addToVec + iBlock * contiguousBlockSize,
                         [&coeff](auto &p, auto &q) { return p * coeff + q; });
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockAxpBy(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const ValueType1 *addFromVec,
      const ValueType2 *scalingVector,
      const ValueType2  a,
      const ValueType2  b,
      ValueType1       *addToVec)
    {
      for (dftfe::uInt iBlock = 0; iBlock < numContiguousBlocks; ++iBlock)
        {
          ValueType2 coeff = a * scalingVector[iBlock];
          std::transform(addFromVec + iBlock * contiguousBlockSize,
                         addFromVec + (iBlock + 1) * contiguousBlockSize,
                         addToVec + iBlock * contiguousBlockSize,
                         addToVec + iBlock * contiguousBlockSize,
                         [&coeff, &b](auto &p, auto &q) {
                           return p * coeff + b * q;
                         });
        }
    }

    template <>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockAxpy(
      const dftfe::uInt          contiguousBlockSize,
      const dftfe::uInt          numContiguousBlocks,
      const std::complex<float> *addFromVec,
      const double              *scalingVector,
      const double               a,
      std::complex<float>       *addToVec)
    {
      for (dftfe::uInt iBlock = 0; iBlock < numContiguousBlocks; ++iBlock)
        {
          double coeff = a * scalingVector[iBlock];
          std::transform(addFromVec + iBlock * contiguousBlockSize,
                         addFromVec + (iBlock + 1) * contiguousBlockSize,
                         addToVec + iBlock * contiguousBlockSize,
                         addToVec + iBlock * contiguousBlockSize,
                         [&coeff](auto &p, auto &q) {
                           return std::complex<double>(p) * coeff +
                                  std::complex<double>(q);
                         });
        }
    }

    template <>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockAxpBy(
      const dftfe::uInt          contiguousBlockSize,
      const dftfe::uInt          numContiguousBlocks,
      const std::complex<float> *addFromVec,
      const double              *scalingVector,
      const double               a,
      const double               b,
      std::complex<float>       *addToVec)
    {
      for (dftfe::uInt iBlock = 0; iBlock < numContiguousBlocks; ++iBlock)
        {
          double coeff = a * scalingVector[iBlock];
          std::transform(addFromVec + iBlock * contiguousBlockSize,
                         addFromVec + (iBlock + 1) * contiguousBlockSize,
                         addToVec + iBlock * contiguousBlockSize,
                         addToVec + iBlock * contiguousBlockSize,
                         [&coeff, &b](auto &p, auto &q) {
                           return std::complex<double>(p) * coeff +
                                  std::complex<double>(q) * b;
                         });
        }
    }

    template <typename ValueType>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddTwoVecColumnWise(
        const dftfe::uInt contiguousBlockSize,
        const dftfe::uInt numContiguousBlocks,
        const ValueType  *x,
        const ValueType  *alpha,
        const ValueType  *y,
        const ValueType  *beta,
        ValueType        *z)
    {
      for (dftfe::uInt i = 0; i < numContiguousBlocks; i++)
        {
          for (dftfe::uInt j = 0; j < contiguousBlockSize; j++)
            {
              z[j + i * contiguousBlockSize] =
                alpha[j] * x[j + i * contiguousBlockSize] +
                beta[j] * y[j + i * contiguousBlockSize];
            }
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::rightDiagonalScale(
      const dftfe::uInt numberofVectors,
      const dftfe::uInt sizeOfVector,
      ValueType1       *X,
      ValueType2       *D)
    {
      for (dftfe::uInt iDof = 0; iDof < sizeOfVector; ++iDof)
        for (dftfe::uInt iWave = 0; iWave < numberofVectors; iWave++)
          {
            X[numberofVectors * iDof + iWave] *= ValueType1(D[iWave]);
          }
    }

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::rightDiagonalScale(
      const dftfe::uInt numberofVectors,
      const dftfe::uInt sizeOfVector,
      double           *X,
      double           *D);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::rightDiagonalScale(
      const dftfe::uInt     numberofVectors,
      const dftfe::uInt     sizeOfVector,
      std::complex<double> *X,
      double               *D);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScaleCopy(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double       a,
      const double      *s,
      const double      *copyFromVec,
      double            *copyToVecBlock,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScaleCopy(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double       a,
      const double      *s,
      const float       *copyFromVec,
      float             *copyToVecBlock,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScaleCopy(
      const dftfe::uInt           contiguousBlockSize,
      const dftfe::uInt           numContiguousBlocks,
      const double                a,
      const double               *s,
      const std::complex<double> *copyFromVec,
      std::complex<double>       *copyToVecBlock,
      const dftfe::uInt          *addToVecStartingContiguousBlockIds);
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScaleCopy(
      const dftfe::uInt          contiguousBlockSize,
      const dftfe::uInt          numContiguousBlocks,
      const double               a,
      const double              *s,
      const std::complex<float> *copyFromVec,
      std::complex<float>       *copyToVecBlock,
      const dftfe::uInt         *addToVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScaleColumnWise(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const double     *beta,
      double           *x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScaleColumnWise(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const float      *beta,
      float            *x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScaleColumnWise(
      const dftfe::uInt           contiguousBlockSize,
      const dftfe::uInt           numContiguousBlocks,
      const std::complex<double> *beta,
      std::complex<double>       *x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScaleColumnWise(
      const dftfe::uInt          contiguousBlockSize,
      const dftfe::uInt          numContiguousBlocks,
      const std::complex<float> *beta,
      std::complex<float>       *x);

    template <typename ValueType1, typename ValueType2>
    void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const ValueType1  a,
      const ValueType1 *s,
      ValueType2       *x)
    {
      for (dftfe::Int iBatch = 0; iBatch < numContiguousBlocks; iBatch++)
        {
          ValueType2 alpha = a * s[iBatch];
          xscal(x + iBatch * contiguousBlockSize, alpha, contiguousBlockSize);
        }
    }
    // MultiVectorXDot
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::MultiVectorXDot(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const double     *X,
      const double     *Y,
      const double     *onesVec,
      double           *tempVector,
      double           *tempResults,
      double           *result);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::MultiVectorXDot(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const double     *X,
      const double     *Y,
      const double     *onesVec,
      double           *tempVector,
      double           *tempResults,
      const MPI_Comm   &mpi_communicator,
      double           *result);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::MultiVectorXDot(
      const dftfe::uInt           contiguousBlockSize,
      const dftfe::uInt           numContiguousBlocks,
      const std::complex<double> *X,
      const std::complex<double> *Y,
      const std::complex<double> *onesVec,
      std::complex<double>       *tempVector,
      std::complex<double>       *tempResults,
      std::complex<double>       *result);
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::MultiVectorXDot(
      const dftfe::uInt           contiguousBlockSize,
      const dftfe::uInt           numContiguousBlocks,
      const std::complex<double> *X,
      const std::complex<double> *Y,
      const std::complex<double> *onesVec,
      std::complex<double>       *tempVector,
      std::complex<double>       *tempResults,
      const MPI_Comm             &mpi_communicator,
      std::complex<double>       *result);

    // stridedBlockScale
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const double      a,
      const double     *s,
      double           *x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const float       a,
      const float      *s,
      float            *x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::uInt           contiguousBlockSize,
      const dftfe::uInt           numContiguousBlocks,
      const std::complex<double>  a,
      const std::complex<double> *s,
      std::complex<double>       *x);
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::uInt     contiguousBlockSize,
      const dftfe::uInt     numContiguousBlocks,
      const double          a,
      const double         *s,
      std::complex<double> *x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::uInt    contiguousBlockSize,
      const dftfe::uInt    numContiguousBlocks,
      const double         a,
      const double        *s,
      std::complex<float> *x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::uInt          contiguousBlockSize,
      const dftfe::uInt          numContiguousBlocks,
      const std::complex<float>  a,
      const std::complex<float> *s,
      std::complex<float>       *x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const double      a,
      const double     *s,
      float            *x);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockScale(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const float       a,
      const float      *s,
      double           *x);

    // for stridedBlockScaleAndAddColumnWise
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddColumnWise(const dftfe::uInt contiguousBlockSize,
                                        const dftfe::uInt numContiguousBlocks,
                                        const double     *x,
                                        const double     *beta,
                                        double           *y);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddColumnWise(const dftfe::uInt contiguousBlockSize,
                                        const dftfe::uInt numContiguousBlocks,
                                        const float      *x,
                                        const float      *beta,
                                        float            *y);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddColumnWise(const dftfe::uInt contiguousBlockSize,
                                        const dftfe::uInt numContiguousBlocks,
                                        const std::complex<double> *x,
                                        const std::complex<double> *beta,
                                        std::complex<double>       *y);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddColumnWise(const dftfe::uInt contiguousBlockSize,
                                        const dftfe::uInt numContiguousBlocks,
                                        const std::complex<float> *x,
                                        const std::complex<float> *beta,
                                        std::complex<float>       *y);

    // for stridedBlockScaleAndAddTwoVecColumnWise
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddTwoVecColumnWise(
        const dftfe::uInt contiguousBlockSize,
        const dftfe::uInt numContiguousBlocks,
        const double     *x,
        const double     *alpha,
        const double     *y,
        const double     *beta,
        double           *z);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddTwoVecColumnWise(
        const dftfe::uInt contiguousBlockSize,
        const dftfe::uInt numContiguousBlocks,
        const float      *x,
        const float      *alpha,
        const float      *y,
        const float      *beta,
        float            *z);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddTwoVecColumnWise(
        const dftfe::uInt           contiguousBlockSize,
        const dftfe::uInt           numContiguousBlocks,
        const std::complex<double> *x,
        const std::complex<double> *alpha,
        const std::complex<double> *y,
        const std::complex<double> *beta,
        std::complex<double>       *z);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      stridedBlockScaleAndAddTwoVecColumnWise(
        const dftfe::uInt          contiguousBlockSize,
        const dftfe::uInt          numContiguousBlocks,
        const std::complex<float> *x,
        const std::complex<float> *alpha,
        const std::complex<float> *y,
        const std::complex<float> *beta,
        std::complex<float>       *z);


    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyToBlock(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double      *copyFromVec,
      double            *copyToVecBlock,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyToBlock(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const float       *copyFromVec,
      float             *copyToVecBlock,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds);


    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyToBlock(
      const dftfe::uInt           contiguousBlockSize,
      const dftfe::uInt           numContiguousBlocks,
      const std::complex<double> *copyFromVec,
      std::complex<double>       *copyToVecBlock,
      const dftfe::uInt          *copyFromVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyToBlock(
      const dftfe::uInt          contiguousBlockSize,
      const dftfe::uInt          numContiguousBlocks,
      const std::complex<float> *copyFromVec,
      std::complex<float>       *copyToVecBlock,
      const dftfe::uInt         *copyFromVecStartingContiguousBlockIds);


    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyToBlock(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const dftfe::uInt  startingVecId,
      const double      *copyFromVec,
      double            *copyToVecBlock,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyToBlock(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const dftfe::uInt  startingVecId,
      const float       *copyFromVec,
      float             *copyToVecBlock,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyToBlock(
      const dftfe::uInt           contiguousBlockSize,
      const dftfe::uInt           numContiguousBlocks,
      const dftfe::uInt           startingVecId,
      const std::complex<double> *copyFromVec,
      std::complex<double>       *copyToVecBlock,
      const dftfe::uInt          *copyFromVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyToBlock(
      const dftfe::uInt          contiguousBlockSize,
      const dftfe::uInt          numContiguousBlocks,
      const dftfe::uInt          startingVecId,
      const std::complex<float> *copyFromVec,
      std::complex<float>       *copyToVecBlock,
      const dftfe::uInt         *copyFromVecStartingContiguousBlockIds);

    // template void
    // BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedCopyToBlock(
    //   const dftfe::uInt         contiguousBlockSize,
    //   const dftfe::uInt         numContiguousBlocks,
    //   const std::complex<double> *   copyFromVec,
    //   std::complex<float> *         copyToVecBlock,
    //   const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      copyValueType1ArrToValueType2Arr(const dftfe::uInt     size,
                                       const double         *valueType1Arr,
                                       std::complex<double> *valueType2Arr);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      copyValueType1ArrToValueType2Arr(
        const dftfe::uInt           size,
        const std::complex<double> *valueType1Arr,
        std::complex<double>       *valueType2Arr);
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      copyValueType1ArrToValueType2Arr(const dftfe::uInt size,
                                       const double     *valueType1Arr,
                                       double           *valueType2Arr);
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      copyValueType1ArrToValueType2Arr(const dftfe::uInt size,
                                       const double     *valueType1Arr,
                                       float            *valueType2Arr);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      copyValueType1ArrToValueType2Arr(const dftfe::uInt size,
                                       const float      *valueType1Arr,
                                       double           *valueType2Arr);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::
      copyValueType1ArrToValueType2Arr(
        const dftfe::uInt           size,
        const std::complex<double> *valueType1Arr,
        std::complex<float>        *valueType2Arr);

    // axpyStridedBlockAtomicAdd
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double      *addFromVec,
      double            *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt           contiguousBlockSize,
      const dftfe::uInt           numContiguousBlocks,
      const std::complex<double> *addFromVec,
      std::complex<double>       *addToVec,
      const dftfe::uInt          *addToVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double       a,
      const double      *s,
      const double      *addFromVec,
      double            *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt           contiguousBlockSize,
      const dftfe::uInt           numContiguousBlocks,
      const double                a,
      const double               *s,
      const std::complex<double> *addFromVec,
      std::complex<double>       *addToVec,
      const dftfe::uInt          *addToVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double       a,
      const double      *s,
      const float       *addFromVec,
      float             *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt          contiguousBlockSize,
      const dftfe::uInt          numContiguousBlocks,
      const double               a,
      const double              *s,
      const std::complex<float> *addFromVec,
      std::complex<float>       *addToVec,
      const dftfe::uInt         *addToVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const float        a,
      const float       *s,
      const float       *addFromVec,
      float             *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt          contiguousBlockSize,
      const dftfe::uInt          numContiguousBlocks,
      const float                a,
      const float               *s,
      const std::complex<float> *addFromVec,
      std::complex<float>       *addToVec,
      const dftfe::uInt         *addToVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double       a,
      const double      *addFromVec,
      double            *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt           contiguousBlockSize,
      const dftfe::uInt           numContiguousBlocks,
      const double                a,
      const std::complex<double> *addFromVec,
      std::complex<double>       *addToVec,
      const dftfe::uInt          *addToVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double       a,
      const float       *addFromVec,
      float             *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt          contiguousBlockSize,
      const dftfe::uInt          numContiguousBlocks,
      const double               a,
      const std::complex<float> *addFromVec,
      std::complex<float>       *addToVec,
      const dftfe::uInt         *addToVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const float        a,
      const float       *addFromVec,
      float             *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpyStridedBlockAtomicAdd(
      const dftfe::uInt          contiguousBlockSize,
      const dftfe::uInt          numContiguousBlocks,
      const float                a,
      const std::complex<float> *addFromVec,
      std::complex<float>       *addToVec,
      const dftfe::uInt         *addToVecStartingContiguousBlockIds);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpby(const dftfe::uInt n,
                                                        const double      alpha,
                                                        const double     *x,
                                                        const double      beta,
                                                        double           *y);


    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpby(
      const dftfe::uInt           n,
      const double                alpha,
      const std::complex<double> *x,
      const double                beta,
      std::complex<double>       *y);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::axpby(const dftfe::uInt n,
                                                        const double      alpha,
                                                        const float      *x,
                                                        const double      beta,
                                                        float            *y);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::ApaBD(const dftfe::uInt m,
                                                        const dftfe::uInt n,
                                                        const double      alpha,
                                                        const double     *A,
                                                        const double     *B,
                                                        const double     *D,
                                                        double           *C);
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::ApaBD(
      const dftfe::uInt           m,
      const dftfe::uInt           n,
      const double                alpha,
      const std::complex<double> *A,
      const std::complex<double> *B,
      const double               *D,
      std::complex<double>       *C);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::ApaBD(const dftfe::uInt m,
                                                        const dftfe::uInt n,
                                                        const double      alpha,
                                                        const float      *A,
                                                        const double     *B,
                                                        const double     *D,
                                                        float            *C);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::ApaBD(const dftfe::uInt m,
                                                        const dftfe::uInt n,
                                                        const double      alpha,
                                                        const float      *A,
                                                        const double     *B,
                                                        const double     *D,
                                                        double           *C);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::copyRealArrsToComplexArr(
      const dftfe::uInt     size,
      const double         *realArr,
      const double         *imagArr,
      std::complex<double> *complexArr);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockAxpy(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const double     *addFromVec,
      const double     *scalingVector,
      const double      a,
      double           *addToVec);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockAxpy(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const float      *addFromVec,
      const double     *scalingVector,
      const double      a,
      float            *addToVec);


    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockAxpy(
      const dftfe::uInt           contiguousBlockSize,
      const dftfe::uInt           numContiguousBlocks,
      const std::complex<double> *addFromVec,
      const std::complex<double> *scalingVector,
      const std::complex<double>  a,
      std::complex<double>       *addToVec);
    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockAxpy(
      const dftfe::uInt           contiguousBlockSize,
      const dftfe::uInt           numContiguousBlocks,
      const std::complex<double> *addFromVec,
      const double               *scalingVector,
      const double                a,
      std::complex<double>       *addToVec);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockAxpBy(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const double     *addFromVec,
      const double     *scalingVector,
      const double      a,
      const double      b,
      double           *addToVec);

    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockAxpBy(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const float      *addFromVec,
      const double     *scalingVector,
      const double      a,
      const double      b,
      float            *addToVec);


    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockAxpBy(
      const dftfe::uInt           contiguousBlockSize,
      const dftfe::uInt           numContiguousBlocks,
      const std::complex<double> *addFromVec,
      const std::complex<double> *scalingVector,
      const std::complex<double>  a,
      const std::complex<double>  b,
      std::complex<double>       *addToVec);



    template void
    BLASWrapper<dftfe::utils::MemorySpace::HOST>::stridedBlockAxpBy(
      const dftfe::uInt           contiguousBlockSize,
      const dftfe::uInt           numContiguousBlocks,
      const std::complex<double> *addFromVec,
      const double               *scalingVector,
      const double                a,
      const double                b,
      std::complex<double>       *addToVec);



  } // End of namespace linearAlgebra
} // End of namespace dftfe
