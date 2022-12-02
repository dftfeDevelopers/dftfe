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

#if defined(DFTFE_WITH_DEVICE)
#  ifndef linearAlgebraOperationsDevice_h
#    define linearAlgebraOperationsDevice_h

#    include <cublas_v2.h>
#    include <headers.h>
#    include <operatorDevice.h>
#    include "process_grid.h"
#    include "scalapackWrapper.h"
#    include "elpaScalaManager.h"
#    include "deviceDirectCCLWrapper.h"
#    include "dftParameters.h"

namespace dftfe
{
  extern "C"
  {
    void
    dsyevd_(const char *        jobz,
            const char *        uplo,
            const unsigned int *n,
            double *            A,
            const unsigned int *lda,
            double *            w,
            double *            work,
            const unsigned int *lwork,
            int *               iwork,
            const unsigned int *liwork,
            int *               info);

    void
    zheevd_(const char *          jobz,
            const char *          uplo,
            const unsigned int *  n,
            std::complex<double> *A,
            const unsigned int *  lda,
            double *              w,
            std::complex<double> *work,
            const unsigned int *  lwork,
            double *              rwork,
            const unsigned int *  lrwork,
            int *                 iwork,
            const unsigned int *  liwork,
            int *                 info);
  }

  inline cublasStatus_t
  cublasXgemm(cublasHandle_t    handle,
              cublasOperation_t transa,
              cublasOperation_t transb,
              int               m,
              int               n,
              int               k,
              const double *    alpha,
              const double *    A,
              int               lda,
              const double *    B,
              int               ldb,
              const double *    beta,
              double *          C,
              int               ldc)
  {
    return cublasDgemm(
      handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  inline cublasStatus_t
  cublasXgemm(cublasHandle_t    handle,
              cublasOperation_t transa,
              cublasOperation_t transb,
              int               m,
              int               n,
              int               k,
              const float *     alpha,
              const float *     A,
              int               lda,
              const float *     B,
              int               ldb,
              const float *     beta,
              float *           C,
              int               ldc)
  {
    return cublasSgemm(
      handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  inline cublasStatus_t
  cublasXgemm(cublasHandle_t         handle,
              cublasOperation_t      transa,
              cublasOperation_t      transb,
              int                    m,
              int                    n,
              int                    k,
              const cuDoubleComplex *alpha,
              const cuDoubleComplex *A,
              int                    lda,
              const cuDoubleComplex *B,
              int                    ldb,
              const cuDoubleComplex *beta,
              cuDoubleComplex *      C,
              int                    ldc)
  {
    return cublasZgemm(
      handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  inline cublasStatus_t
  cublasXgemm(cublasHandle_t    handle,
              cublasOperation_t transa,
              cublasOperation_t transb,
              int               m,
              int               n,
              int               k,
              const cuComplex * alpha,
              const cuComplex * A,
              int               lda,
              const cuComplex * B,
              int               ldb,
              const cuComplex * beta,
              cuComplex *       C,
              int               ldc)
  {
    return cublasCgemm(
      handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  inline cublasStatus_t
  cublasXgemmBatched(cublasHandle_t    handle,
                     cublasOperation_t transa,
                     cublasOperation_t transb,
                     int               m,
                     int               n,
                     int               k,
                     const double *    alpha,
                     const double *    Aarray[],
                     int               lda,
                     const double *    Barray[],
                     int               ldb,
                     const double *    beta,
                     double *          Carray[],
                     int               ldc,
                     int               batchCount)
  {
    return cublasDgemmBatched(handle,
                              transa,
                              transb,
                              m,
                              n,
                              k,
                              alpha,
                              Aarray,
                              lda,
                              Barray,
                              ldb,
                              beta,
                              Carray,
                              ldc,
                              batchCount);
  }


  inline cublasStatus_t
  cublasXgemmBatched(cublasHandle_t         handle,
                     cublasOperation_t      transa,
                     cublasOperation_t      transb,
                     int                    m,
                     int                    n,
                     int                    k,
                     const cuDoubleComplex *alpha,
                     const cuDoubleComplex *Aarray[],
                     int                    lda,
                     const cuDoubleComplex *Barray[],
                     int                    ldb,
                     const cuDoubleComplex *beta,
                     cuDoubleComplex *      Carray[],
                     int                    ldc,
                     int                    batchCount)
  {
    return cublasZgemmBatched(handle,
                              transa,
                              transb,
                              m,
                              n,
                              k,
                              alpha,
                              Aarray,
                              lda,
                              Barray,
                              ldb,
                              beta,
                              Carray,
                              ldc,
                              batchCount);
  }

  inline cublasStatus_t
  cublasXgemmStridedBatched(cublasHandle_t    handle,
                            cublasOperation_t transa,
                            cublasOperation_t transb,
                            int               m,
                            int               n,
                            int               k,
                            const double *    alpha,
                            const double *    A,
                            int               lda,
                            long long int     strideA,
                            const double *    B,
                            int               ldb,
                            long long int     strideB,
                            const double *    beta,
                            double *          C,
                            int               ldc,
                            long long int     strideC,
                            int               batchCount)
  {
    return cublasDgemmStridedBatched(handle,
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
  }

  inline cublasStatus_t
  cublasXgemmStridedBatched(cublasHandle_t    handle,
                            cublasOperation_t transa,
                            cublasOperation_t transb,
                            int               m,
                            int               n,
                            int               k,
                            const float *     alpha,
                            const float *     A,
                            int               lda,
                            long long int     strideA,
                            const float *     B,
                            int               ldb,
                            long long int     strideB,
                            const float *     beta,
                            float *           C,
                            int               ldc,
                            long long int     strideC,
                            int               batchCount)
  {
    return cublasSgemmStridedBatched(handle,
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
  }


  inline cublasStatus_t
  cublasXgemmStridedBatched(cublasHandle_t         handle,
                            cublasOperation_t      transa,
                            cublasOperation_t      transb,
                            int                    m,
                            int                    n,
                            int                    k,
                            const cuDoubleComplex *alpha,
                            const cuDoubleComplex *A,
                            int                    lda,
                            long long int          strideA,
                            const cuDoubleComplex *B,
                            int                    ldb,
                            long long int          strideB,
                            const cuDoubleComplex *beta,
                            cuDoubleComplex *      C,
                            int                    ldc,
                            long long int          strideC,
                            int                    batchCount)
  {
    return cublasZgemmStridedBatched(handle,
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
  }


  inline cublasStatus_t
  cublasXgemmStridedBatched(cublasHandle_t        handle,
                            cublasOperation_t     transa,
                            cublasOperation_t     transb,
                            int                   m,
                            int                   n,
                            int                   k,
                            const cuFloatComplex *alpha,
                            const cuFloatComplex *A,
                            int                   lda,
                            long long int         strideA,
                            const cuFloatComplex *B,
                            int                   ldb,
                            long long int         strideB,
                            const cuFloatComplex *beta,
                            cuFloatComplex *      C,
                            int                   ldc,
                            long long int         strideC,
                            int                   batchCount)
  {
    return cublasCgemmStridedBatched(handle,
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
  }

  /**
   *  @brief Contains functions for linear algebra operations on Device
   *
   *  @author Sambit Das
   */
  namespace linearAlgebraOperationsDevice
  {
    /** @brief Computes Sc=X^{T}*Xc.
     *
     *
     */
    void
    fillParallelOverlapMatScalapack(
      const dataTypes::numberDevice *                  X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                               devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar,
      const dftParameters &                            dftParams);



    /** @brief Computes Sc=X^{T}*Xc.
     *
     *
     */
    void
    fillParallelOverlapMatScalapackAsyncComputeCommun(
      const dataTypes::numberDevice *                  X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                               devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar,
      const dftParameters &                            dftParams);



    /** @brief Computes Sc=X^{T}*Xc.
     *
     *
     */
    void
    fillParallelOverlapMatMixedPrecScalapackAsyncComputeCommun(
      const dataTypes::numberDevice *                  X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                               devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar,
      const dftParameters &                            dftParams);



    /** @brief Computes Sc=X^{T}*Xc.
     *
     *
     */
    void
    fillParallelOverlapMatMixedPrecScalapack(
      const dataTypes::numberDevice *                  X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                               devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar,
      const dftParameters &                            dftParams);



    /** @brief CGS orthogonalization
     */
    void
    pseudoGramSchmidtOrthogonalization(elpaScalaManager &       elpaScala,
                                       dataTypes::numberDevice *X,
                                       const unsigned int       M,
                                       const unsigned int       N,
                                       const MPI_Comm &         mpiCommParent,
                                       const MPI_Comm &         mpiCommDomain,
                                       utils::DeviceCCLWrapper &devicecclMpiCommDomain,
                                       const MPI_Comm &  interBandGroupComm,
                                       cublasHandle_t &  handle,
                                       const dftParameters &dftParams,
                                       const bool useMixedPrecOverall = false);

    void
    subspaceRotationScalapack(
      dataTypes::numberDevice *                        X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                               devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters &                            dftParams,
      const bool rotationMatTranspose   = false,
      const bool isRotationMatLowerTria = false);


    void
    subspaceRotationSpectrumSplitScalapack(
      const dataTypes::numberDevice *                  X,
      dataTypes::numberDevice *                        XFrac,
      const unsigned int                               M,
      const unsigned int                               N,
      const unsigned int                               Nfr,
      cublasHandle_t &                                 handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                               devicecclMpiCommDomain,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters &                            dftParams,
      const bool rotationMatTranspose = false);

    void
    subspaceRotationCGSMixedPrecScalapack(
      dataTypes::numberDevice *                        X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                               devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters &                            dftParams,
      const bool rotationMatTranspose = false);


    void
    subspaceRotationRRMixedPrecScalapack(
      dataTypes::numberDevice *                        X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                               devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters &                            dftParams,
      const bool rotationMatTranspose = false);


    void
    rayleighRitz(
      operatorDFTDeviceClass &                           operatorMatrix,
      elpaScalaManager &                                 elpaScala,
      dataTypes::numberDevice *                          X,
      distributedDeviceVec<dataTypes::numberDevice> &    Xb,
      distributedDeviceVec<dataTypes::numberFP32Device> &floatXb,
      distributedDeviceVec<dataTypes::numberDevice> &    HXb,
      distributedDeviceVec<dataTypes::numberDevice> &projectorKetTimesVector,
      const unsigned int                             M,
      const unsigned int                             N,
      const MPI_Comm &                               mpiCommParent,
      const MPI_Comm &                               mpiCommDomain,
      utils::DeviceCCLWrapper &                             devicecclMpiCommDomain,
      const MPI_Comm &                               interBandGroupComm,
      std::vector<double> &                          eigenValues,
      cublasHandle_t &                               handle,
      const dftParameters &                          dftParams,
      const bool useMixedPrecOverall = false);

    void
    rayleighRitzGEP(
      operatorDFTDeviceClass &                           operatorMatrix,
      elpaScalaManager &                                 elpaScala,
      dataTypes::numberDevice *                          X,
      distributedDeviceVec<dataTypes::numberDevice> &    Xb,
      distributedDeviceVec<dataTypes::numberFP32Device> &floatXb,
      distributedDeviceVec<dataTypes::numberDevice> &    HXb,
      distributedDeviceVec<dataTypes::numberDevice> &projectorKetTimesVector,
      const unsigned int                             M,
      const unsigned int                             N,
      const MPI_Comm &                               mpiCommParent,
      const MPI_Comm &                               mpiCommDomain,
      utils::DeviceCCLWrapper &                             devicecclMpiCommDomain,
      const MPI_Comm &                               interBandGroupComm,
      std::vector<double> &                          eigenValues,
      cublasHandle_t &                               handle,
      const dftParameters &                          dftParams,
      const bool useMixedPrecOverall = false);

    void
    rayleighRitzGEPSpectrumSplitDirect(
      operatorDFTDeviceClass &                           operatorMatrix,
      elpaScalaManager &                                 elpaScala,
      dataTypes::numberDevice *                          X,
      dataTypes::numberDevice *                          XFrac,
      distributedDeviceVec<dataTypes::numberDevice> &    Xb,
      distributedDeviceVec<dataTypes::numberFP32Device> &floatXb,
      distributedDeviceVec<dataTypes::numberDevice> &    HXb,
      distributedDeviceVec<dataTypes::numberDevice> &projectorKetTimesVector,
      const unsigned int                             M,
      const unsigned int                             N,
      const unsigned int                             Noc,
      const MPI_Comm &                               mpiCommParent,
      const MPI_Comm &                               mpiCommDomain,
      utils::DeviceCCLWrapper &                             devicecclMpiCommDomain,
      const MPI_Comm &                               interBandGroupComm,
      std::vector<double> &                          eigenValues,
      cublasHandle_t &                               handle,
      const dftParameters &                          dftParams,
      const bool useMixedPrecOverall = false);


    void
    densityMatrixEigenBasisFirstOrderResponse(
      operatorDFTDeviceClass &                           operatorMatrix,
      dataTypes::numberDevice *                          X,
      distributedDeviceVec<dataTypes::numberDevice> &    Xb,
      distributedDeviceVec<dataTypes::numberFP32Device> &floatXb,
      distributedDeviceVec<dataTypes::numberDevice> &    HXb,
      distributedDeviceVec<dataTypes::numberDevice> &projectorKetTimesVector,
      const unsigned int                             M,
      const unsigned int                             N,
      const MPI_Comm &                               mpiCommParent,
      const MPI_Comm &                               mpiCommDomain,
      utils::DeviceCCLWrapper &                             devicecclMpiCommDomain,
      const MPI_Comm &                               interBandGroupComm,
      const std::vector<double> &                    eigenValues,
      const double                                   fermiEnergy,
      std::vector<double> &                          densityMatDerFermiEnergy,
      dftfe::elpaScalaManager &                      elpaScala,
      cublasHandle_t &                               handle,
      const dftParameters &                          dftParams);

    /** @brief Calculates an estimate of lower and upper bounds of a matrix using
     *  k-step Lanczos method.
     *
     *  @param  operatorMatrix An object which has access to the given matrix
     *  @param  vect A dummy vector
     *  @return double An estimate of the upper bound of the given matrix
     */
    std::pair<double, double>
    lanczosLowerUpperBoundEigenSpectrum(
      operatorDFTDeviceClass &                       operatorMatrix,
      distributedDeviceVec<dataTypes::numberDevice> &Xb,
      distributedDeviceVec<dataTypes::numberDevice> &Yb,
      distributedDeviceVec<dataTypes::numberDevice> &projectorKetTimesVector,
      const unsigned int                             blockSize,
      const dftParameters &                          dftParams);


    /** @brief Apply Chebyshev filter to a given subspace
     *
     *  @param[in] operatorMatrix An object which has access to the given matrix
     *  @param[in,out]  X Given subspace as a dealii array representing multiple
     * fields as a flattened array. In-place update of the given subspace.
     *  @param[in]  numberComponents Number of multiple-fields
     *  @param[in]  m Chebyshev polynomial degree
     *  @param[in]  a lower bound of unwanted spectrum
     *  @param[in]  b upper bound of unwanted spectrum
     *  @param[in]  a0 lower bound of wanted spectrum
     */
    void
    chebyshevFilter(
      operatorDFTDeviceClass &operatorMatrix,
      distributedDeviceVec<dataTypes::numberDevice>
        &X, // thrust::device_vector<dataTypes::number> & X,
      distributedDeviceVec<dataTypes::numberDevice> &    Y,
      distributedDeviceVec<dataTypes::numberFP32Device> &Z,
      distributedDeviceVec<dataTypes::numberDevice> &projectorKetTimesVector,
      const unsigned int                             localVectorSize,
      const unsigned int                             numberComponents,
      const unsigned int                             m,
      const double                                   a,
      const double                                   b,
      const double                                   a0,
      const bool                                     mixedPrecOverall,
      const dftParameters &                          dftParams);


    void
    chebyshevFilter(
      operatorDFTDeviceClass &                           operatorMatrix,
      distributedDeviceVec<dataTypes::numberDevice> &    X1,
      distributedDeviceVec<dataTypes::numberDevice> &    Y1,
      distributedDeviceVec<dataTypes::numberFP32Device> &Z,
      distributedDeviceVec<dataTypes::numberDevice> &projectorKetTimesVector1,
      distributedDeviceVec<dataTypes::numberDevice> &X2,
      distributedDeviceVec<dataTypes::numberDevice> &Y2,
      distributedDeviceVec<dataTypes::numberDevice> &projectorKetTimesVector2,
      const unsigned int                             localVectorSize,
      const unsigned int                             numberComponents,
      const unsigned int                             m,
      const double                                   a,
      const double                                   b,
      const double                                   a0,
      const bool                                     mixedPrecOverall,
      const dftParameters &                          dftParams);

    void
    computeEigenResidualNorm(
      operatorDFTDeviceClass &                       operatorMatrix,
      dataTypes::numberDevice *                      X,
      distributedDeviceVec<dataTypes::numberDevice> &Xb,
      distributedDeviceVec<dataTypes::numberDevice> &HXb,
      distributedDeviceVec<dataTypes::numberDevice> &projectorKetTimesVector,
      const unsigned int                             M,
      const unsigned int                             N,
      const std::vector<double> &                    eigenValues,
      const MPI_Comm &                               mpiCommDomain,
      const MPI_Comm &                               interBandGroupComm,
      cublasHandle_t &                               handle,
      std::vector<double> &                          residualNorm,
      const dftParameters &                          dftParams,
      const bool                                     useBandParal = false);
  } // namespace linearAlgebraOperationsDevice
} // namespace dftfe
#  endif
#endif
