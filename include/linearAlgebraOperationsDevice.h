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
#    include <operator.h>
#    include "process_grid.h"
#    include "scalapackWrapper.h"
#    include "elpaScalaManager.h"
#    include "deviceDirectCCLWrapper.h"
#    include "dftParameters.h"
#    include <BLASWrapper.h>

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
    chebyshevFilterOverlapComputeCommunication(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &X1,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &Y1,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &X2,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &Y2,
      const unsigned int                                                    m,
      const double                                                          a,
      const double                                                          b,
      const double                                                          a0);

    void
    chebyshevFilterOverlapComputeCommunicationSinglePrec(
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                                  BLASWrapperPtr,
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &X1,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &Y1,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &X2,
      dftfe::linearAlgebra::MultiVector<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE> &Y2,
      dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32,
                                        dftfe::utils::MemorySpace::DEVICE>
        &X1_SP,
      dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32,
                                        dftfe::utils::MemorySpace::DEVICE>
        &Y1_SP,
      dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32,
                                        dftfe::utils::MemorySpace::DEVICE>
        &X2_SP,
      dftfe::linearAlgebra::MultiVector<dataTypes::numberFP32,
                                        dftfe::utils::MemorySpace::DEVICE>
        &                 Y2_SP,
      std::vector<double> eigenvalues,
      const unsigned int  m,
      const double        a,
      const double        b,
      const double        a0);

    /** @brief Computes Sc=X^{T}*Xc.
     *
     *
     */
    void
    fillParallelOverlapMatScalapack(
      const dataTypes::number *X,
      const unsigned int       M,
      const unsigned int       N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
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
      const dataTypes::number *X,
      const unsigned int       M,
      const unsigned int       N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
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
      const dataTypes::number *X,
      const unsigned int       M,
      const unsigned int       N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar,
      const dftParameters &                            dftParams);

    /** @brief Computes Sc=X^{T}*Xc.
     *
     *
     */
    void
    fillParallelOverlapMatMixedPrecCommunScalapackAsyncComputeCommun(
      const dataTypes::number *X,
      const unsigned int       M,
      const unsigned int       N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
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
      const dataTypes::number *X,
      const unsigned int       M,
      const unsigned int       N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar,
      const dftParameters &                            dftParams);



    /** @brief CGS orthogonalization
     */
    void
    pseudoGramSchmidtOrthogonalization(
      elpaScalaManager &       elpaScala,
      dataTypes::number *      X,
      const unsigned int       M,
      const unsigned int       N,
      const MPI_Comm &         mpiCommParent,
      const MPI_Comm &         mpiCommDomain,
      utils::DeviceCCLWrapper &devicecclMpiCommDomain,
      const MPI_Comm &         interBandGroupComm,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                  BLASWrapperPtr,
      const dftParameters &dftParams,
      const bool           useMixedPrecOverall = false);

    void
    subspaceRotationScalapack(
      dataTypes::number *X,
      const unsigned int M,
      const unsigned int N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters &                            dftParams,
      const bool rotationMatTranspose   = false,
      const bool isRotationMatLowerTria = false);


    void
    subspaceRotationSpectrumSplitScalapack(
      const dataTypes::number *X,
      dataTypes::number *      XFrac,
      const unsigned int       M,
      const unsigned int       N,
      const unsigned int       Nfr,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters &                            dftParams,
      const bool rotationMatTranspose = false);

    void
    subspaceRotationCGSMixedPrecScalapack(
      dataTypes::number *X,
      const unsigned int M,
      const unsigned int N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters &                            dftParams,
      const bool rotationMatTranspose = false);


    void
    subspaceRotationRRMixedPrecScalapack(
      dataTypes::number *X,
      const unsigned int M,
      const unsigned int N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters &                            dftParams,
      const bool rotationMatTranspose = false);


    void
    rayleighRitz(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      elpaScalaManager &                                   elpaScala,
      dataTypes::number *                                  X,
      distributedDeviceVec<dataTypes::number> &            Xb,
      distributedDeviceVec<dataTypes::number> &            HXb,
      const unsigned int                                   M,
      const unsigned int                                   N,
      const MPI_Comm &                                     mpiCommParent,
      const MPI_Comm &                                     mpiCommDomain,
      utils::DeviceCCLWrapper &devicecclMpiCommDomain,
      const MPI_Comm &         interBandGroupComm,
      std::vector<double> &    eigenValues,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                  BLASWrapperPtr,
      const dftParameters &dftParams,
      const bool           useMixedPrecOverall = false);

    void
    rayleighRitzGEP(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      elpaScalaManager &                                   elpaScala,
      dataTypes::number *                                  X,
      distributedDeviceVec<dataTypes::number> &            Xb,
      distributedDeviceVec<dataTypes::number> &            HXb,
      const unsigned int                                   M,
      const unsigned int                                   N,
      const MPI_Comm &                                     mpiCommParent,
      const MPI_Comm &                                     mpiCommDomain,
      utils::DeviceCCLWrapper &devicecclMpiCommDomain,
      const MPI_Comm &         interBandGroupComm,
      std::vector<double> &    eigenValues,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                  BLASWrapperPtr,
      const dftParameters &dftParams,
      const bool           useMixedPrecOverall = false);

    void
    rayleighRitzGEPSpectrumSplitDirect(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      elpaScalaManager &                                   elpaScala,
      dataTypes::number *                                  X,
      dataTypes::number *                                  XFrac,
      distributedDeviceVec<dataTypes::number> &            Xb,
      distributedDeviceVec<dataTypes::number> &            HXb,
      const unsigned int                                   M,
      const unsigned int                                   N,
      const unsigned int                                   Noc,
      const MPI_Comm &                                     mpiCommParent,
      const MPI_Comm &                                     mpiCommDomain,
      utils::DeviceCCLWrapper &devicecclMpiCommDomain,
      const MPI_Comm &         interBandGroupComm,
      std::vector<double> &    eigenValues,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                  BLASWrapperPtr,
      const dftParameters &dftParams,
      const bool           useMixedPrecOverall = false);


    void
    densityMatrixEigenBasisFirstOrderResponse(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      dataTypes::number *                                  X,
      distributedDeviceVec<dataTypes::number> &            Xb,
      distributedDeviceVec<dataTypes::number> &            HXb,
      const unsigned int                                   M,
      const unsigned int                                   N,
      const MPI_Comm &                                     mpiCommParent,
      const MPI_Comm &                                     mpiCommDomain,
      utils::DeviceCCLWrapper &  devicecclMpiCommDomain,
      const MPI_Comm &           interBandGroupComm,
      const std::vector<double> &eigenValues,
      const double               fermiEnergy,
      std::vector<double> &      densityMatDerFermiEnergy,
      dftfe::elpaScalaManager &  elpaScala,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                  BLASWrapperPtr,
      const dftParameters &dftParams);

    void
    computeEigenResidualNorm(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      dataTypes::number *                                  X,
      distributedDeviceVec<dataTypes::number> &            Xb,
      distributedDeviceVec<dataTypes::number> &            HXb,
      const unsigned int                                   M,
      const unsigned int                                   N,
      const std::vector<double> &                          eigenValues,
      const MPI_Comm &                                     mpiCommParent,
      const MPI_Comm &                                     mpiCommDomain,
      const MPI_Comm &                                     interBandGroupComm,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                  BLASWrapperPtr,
      std::vector<double> &residualNorm,
      const dftParameters &dftParams,
      const bool           useBandParal = false);

    void
    XtHX(operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
         const dataTypes::number *                            X,
         distributedDeviceVec<dataTypes::number> &            XBlock,
         distributedDeviceVec<dataTypes::number> &            HXBlock,
         const unsigned int                                   M,
         const unsigned int                                   N,
         std::shared_ptr<
           dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
           &                                              BLASWrapperPtr,
         const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
         dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
         utils::DeviceCCLWrapper &devicecclMpiCommDomain,
         const MPI_Comm &         mpiCommDomain,
         const MPI_Comm &         interBandGroupComm,
         const dftParameters &    dftParams,
         const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);

    void
    XtHXMixedPrecOverlapComputeCommun(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      const dataTypes::number *                            X,
      distributedDeviceVec<dataTypes::number> &            XBlock,
      distributedDeviceVec<dataTypes::number> &            HXBlock,
      const unsigned int                                   M,
      const unsigned int                                   N,
      const unsigned int                                   Noc,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 mpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftParameters &                            dftParams,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);

    void
    XtHXOverlapComputeCommun(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      const dataTypes::number *                            X,
      distributedDeviceVec<dataTypes::number> &            XBlock,
      distributedDeviceVec<dataTypes::number> &            HXBlock,
      const unsigned int                                   M,
      const unsigned int                                   N,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 mpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftParameters &                            dftParams,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);

    void
    XtHXMixedPrecCommunOverlapComputeCommun(
      operatorDFTClass<dftfe::utils::MemorySpace::DEVICE> &operatorMatrix,
      const dataTypes::number *                            X,
      distributedDeviceVec<dataTypes::number> &            XBlock,
      distributedDeviceVec<dataTypes::number> &            HXBlock,
      const unsigned int                                   M,
      const unsigned int                                   N,
      const unsigned int                                   Noc,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                              BLASWrapperPtr,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      utils::DeviceCCLWrapper &                        devicecclMpiCommDomain,
      const MPI_Comm &                                 mpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftParameters &                            dftParams,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);

  } // namespace linearAlgebraOperationsDevice
} // namespace dftfe
#  endif
#endif
