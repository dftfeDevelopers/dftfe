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
      const dataTypes::number *                  X,
      const unsigned int                               M,
      const unsigned int                               N,
      dftfe::utils::deviceBlasHandle_t &                                 handle,
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
      const dataTypes::number *                  X,
      const unsigned int                               M,
      const unsigned int                               N,
      dftfe::utils::deviceBlasHandle_t &                                 handle,
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
      const dataTypes::number *                  X,
      const unsigned int                               M,
      const unsigned int                               N,
      dftfe::utils::deviceBlasHandle_t &                                 handle,
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
      const dataTypes::number *                  X,
      const unsigned int                               M,
      const unsigned int                               N,
      dftfe::utils::deviceBlasHandle_t &                                 handle,
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
                                       dataTypes::number *X,
                                       const unsigned int       M,
                                       const unsigned int       N,
                                       const MPI_Comm &         mpiCommParent,
                                       const MPI_Comm &         mpiCommDomain,
                                       utils::DeviceCCLWrapper &devicecclMpiCommDomain,
                                       const MPI_Comm &  interBandGroupComm,
                                       dftfe::utils::deviceBlasHandle_t &  handle,
                                       const dftParameters &dftParams,
                                       const bool useMixedPrecOverall = false);

    void
    subspaceRotationScalapack(
      dataTypes::number *                        X,
      const unsigned int                               M,
      const unsigned int                               N,
      dftfe::utils::deviceBlasHandle_t &                                 handle,
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
      const dataTypes::number *                  X,
      dataTypes::number *                        XFrac,
      const unsigned int                               M,
      const unsigned int                               N,
      const unsigned int                               Nfr,
      dftfe::utils::deviceBlasHandle_t &                                 handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                               devicecclMpiCommDomain,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters &                            dftParams,
      const bool rotationMatTranspose = false);

    void
    subspaceRotationCGSMixedPrecScalapack(
      dataTypes::number *                        X,
      const unsigned int                               M,
      const unsigned int                               N,
      dftfe::utils::deviceBlasHandle_t &                                 handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                               devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters &                            dftParams,
      const bool rotationMatTranspose = false);


    void
    subspaceRotationRRMixedPrecScalapack(
      dataTypes::number *                        X,
      const unsigned int                               M,
      const unsigned int                               N,
      dftfe::utils::deviceBlasHandle_t &                                 handle,
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
      dataTypes::number *                          X,
      distributedDeviceVec<dataTypes::number> &    Xb,
      distributedDeviceVec<dataTypes::numberFP32> &floatXb,
      distributedDeviceVec<dataTypes::number> &    HXb,
      distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
      const unsigned int                             M,
      const unsigned int                             N,
      const MPI_Comm &                               mpiCommParent,
      const MPI_Comm &                               mpiCommDomain,
      utils::DeviceCCLWrapper &                             devicecclMpiCommDomain,
      const MPI_Comm &                               interBandGroupComm,
      std::vector<double> &                          eigenValues,
      dftfe::utils::deviceBlasHandle_t &                               handle,
      const dftParameters &                          dftParams,
      const bool useMixedPrecOverall = false);

    void
    rayleighRitzGEP(
      operatorDFTDeviceClass &                           operatorMatrix,
      elpaScalaManager &                                 elpaScala,
      dataTypes::number *                          X,
      distributedDeviceVec<dataTypes::number> &    Xb,
      distributedDeviceVec<dataTypes::numberFP32> &floatXb,
      distributedDeviceVec<dataTypes::number> &    HXb,
      distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
      const unsigned int                             M,
      const unsigned int                             N,
      const MPI_Comm &                               mpiCommParent,
      const MPI_Comm &                               mpiCommDomain,
      utils::DeviceCCLWrapper &                             devicecclMpiCommDomain,
      const MPI_Comm &                               interBandGroupComm,
      std::vector<double> &                          eigenValues,
      dftfe::utils::deviceBlasHandle_t &                               handle,
      const dftParameters &                          dftParams,
      const bool useMixedPrecOverall = false);

    void
    rayleighRitzGEPSpectrumSplitDirect(
      operatorDFTDeviceClass &                           operatorMatrix,
      elpaScalaManager &                                 elpaScala,
      dataTypes::number *                          X,
      dataTypes::number *                          XFrac,
      distributedDeviceVec<dataTypes::number> &    Xb,
      distributedDeviceVec<dataTypes::numberFP32> &floatXb,
      distributedDeviceVec<dataTypes::number> &    HXb,
      distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
      const unsigned int                             M,
      const unsigned int                             N,
      const unsigned int                             Noc,
      const MPI_Comm &                               mpiCommParent,
      const MPI_Comm &                               mpiCommDomain,
      utils::DeviceCCLWrapper &                             devicecclMpiCommDomain,
      const MPI_Comm &                               interBandGroupComm,
      std::vector<double> &                          eigenValues,
      dftfe::utils::deviceBlasHandle_t &                               handle,
      const dftParameters &                          dftParams,
      const bool useMixedPrecOverall = false);


    void
    densityMatrixEigenBasisFirstOrderResponse(
      operatorDFTDeviceClass &                           operatorMatrix,
      dataTypes::number *                          X,
      distributedDeviceVec<dataTypes::number> &    Xb,
      distributedDeviceVec<dataTypes::numberFP32> &floatXb,
      distributedDeviceVec<dataTypes::number> &    HXb,
      distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
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
      dftfe::utils::deviceBlasHandle_t &                               handle,
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
      distributedDeviceVec<dataTypes::number> &Xb,
      distributedDeviceVec<dataTypes::number> &Yb,
      distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
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
      distributedDeviceVec<dataTypes::number>
        &X, 
      distributedDeviceVec<dataTypes::number> &    Y,
      distributedDeviceVec<dataTypes::numberFP32> &Z,
      distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
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
      distributedDeviceVec<dataTypes::number> &    X1,
      distributedDeviceVec<dataTypes::number> &    Y1,
      distributedDeviceVec<dataTypes::numberFP32> &Z,
      distributedDeviceVec<dataTypes::number> &projectorKetTimesVector1,
      distributedDeviceVec<dataTypes::number> &X2,
      distributedDeviceVec<dataTypes::number> &Y2,
      distributedDeviceVec<dataTypes::number> &projectorKetTimesVector2,
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
      dataTypes::number *                      X,
      distributedDeviceVec<dataTypes::number> &Xb,
      distributedDeviceVec<dataTypes::number> &HXb,
      distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
      const unsigned int                             M,
      const unsigned int                             N,
      const std::vector<double> &                    eigenValues,
      const MPI_Comm &                               mpiCommDomain,
      const MPI_Comm &                               interBandGroupComm,
      dftfe::utils::deviceBlasHandle_t &                               handle,
      std::vector<double> &                          residualNorm,
      const dftParameters &                          dftParams,
      const bool                                     useBandParal = false);
  } // namespace linearAlgebraOperationsDevice
} // namespace dftfe
#  endif
#endif
