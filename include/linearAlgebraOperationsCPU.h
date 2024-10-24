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


#ifndef linearAlgebraOperationsCPU_h
#define linearAlgebraOperationsCPU_h

#include <elpaScalaManager.h>
#include <headers.h>
#include <operator.h>
#include "process_grid.h"
#include "scalapackWrapper.h"
#include "dftParameters.h"
#include <BLASWrapper.h>
namespace dftfe
{
  //
  // extern declarations for blas-lapack routines
  //



  /**
   *  @brief Contains linear algebra functions used in the implementation of an eigen solver
   *
   *  @author Phani Motamarri, Sambit Das
   */
  namespace linearAlgebraOperations
  {
    /** @brief Orthogonalize given subspace using GramSchmidt orthogonalization
     *
     *  @param[in,out]  X Given subspace as flattened array of multi-vectors.
     *  In-place update of the given subspace
     *  @param[in] numberComponents Number of multiple-fields
     *  @param[in] mpiComm global communicator
     */
    template <typename T>
    void
    gramSchmidtOrthogonalization(T *                X,
                                 const unsigned int numberComponents,
                                 const unsigned int numberDofs,
                                 const MPI_Comm &   mpiComm);


    /** @brief Orthogonalize given subspace using Lowden orthogonalization for double data-type
     *  (serial version using LAPACK)
     *
     *  @param[in,out]  X Given subspace as flattened array of multi-vectors.
     *  In-place update of the given subspace
     *  @param[in] numberComponents Number of multiple-fields
     *  @param[in] mpiComm global communicator
     *  @return flag indicating success/failure. 1 for failure, 0 for success
     */
    unsigned int
    lowdenOrthogonalization(std::vector<dataTypes::number> &X,
                            const unsigned int              numberComponents,
                            const MPI_Comm &                mpiComm,
                            const dftParameters &           dftParams);


    /** @brief Orthogonalize given subspace using Pseudo-Gram-Schmidt orthogonalization
     * (serial version using LAPACK, parallel version using ScaLAPACK)
     *
     *  @param[in,out]  X Given subspace as flattened array of multi-vectors.
     *  In-place update of the given subspace
     *  @param[in] numberComponents Number of multiple-fields
     *  @param[in] mpiCommParent parent communicator
     *  @param[in] interBandGroupComm interpool communicator for parallelization
     * over band groups
     *  @param[in] mpiComm domain decomposition communicator
     *
     *  @return flag indicating success/failure. 1 for failure, 0 for success
     */
    template <typename T>
    unsigned int
    pseudoGramSchmidtOrthogonalization(elpaScalaManager &   elpaScala,
                                       T *                  X,
                                       const unsigned int   numberComponents,
                                       const unsigned int   numberDofs,
                                       const MPI_Comm &     mpiCommParent,
                                       const MPI_Comm &     interBandGroupComm,
                                       const MPI_Comm &     mpiCommDomain,
                                       const bool           useMixedPrec,
                                       const dftParameters &dftParams);


    /** @brief Compute Rayleigh-Ritz projection
     *  (serial version using LAPACK, parallel version using ScaLAPACK)
     *
     *  @param[in] operatorMatrix An object which has access to the given matrix
     *  @param[in,out]  X Given subspace as flattened array of multi-vectors.
     *  In-place rotated subspace
     *  @param[in] numberComponents Number of vectors
     *  @param[in] mpiCommDomain parent communicator
     *  @param[in] interBandGroupComm interpool communicator for parallelization
     * over band groups
     *  @param[in] mpiCommDomain domain decomposition communicator
     *  @param[out] eigenValues of the Projected Hamiltonian
     */
    template <typename T>
    void
    rayleighRitzGEP(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      elpaScalaManager &                                 elpaScala,
      T *                                                X,
      const unsigned int                                 numberComponents,
      const unsigned int                                 numberDofs,
      const MPI_Comm &                                   mpiCommParent,
      const MPI_Comm &                                   interBandGroupComm,
      const MPI_Comm &                                   mpiCommDomain,
      std::vector<double> &                              eigenValues,
      const bool                                         useMixedPrec,
      const dftParameters &                              dftParams);


    /** @brief Compute Rayleigh-Ritz projection
     *  (serial version using LAPACK, parallel version using ScaLAPACK)
     *
     *  @param[in] operatorMatrix An object which has access to the given matrix
     *  @param[in,out]  X Given subspace as flattened array of multi-vectors.
     *  In-place rotated subspace
     *  @param[in] numberComponents Number of vectors
     *  @param[in] mpiCommParent parent mpi communicator
     *  @param[in] interBandGroupComm interpool communicator for parallelization
     * over band groups
     *  @param[in] mpiCommDomain domain decomposition communicator
     *  @param[out] eigenValues of the Projected Hamiltonian
     */
    template <typename T>
    void
    rayleighRitz(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      elpaScalaManager &                                 elpaScala,
      T *                                                X,
      const unsigned int                                 numberComponents,
      const unsigned int                                 numberDofs,
      const MPI_Comm &                                   mpiCommParent,
      const MPI_Comm &                                   interBandGroupComm,
      const MPI_Comm &                                   mpiCommDomain,
      std::vector<double> &                              eigenValues,
      const dftParameters &                              dftParams,
      const bool doCommAfterBandParal = true);

    /** @brief Compute Rayleigh-Ritz projection in case of spectrum split using direct diagonalization
     *  (serial version using LAPACK, parallel version using ScaLAPACK)
     *
     *  @param[in] operatorMatrix An object which has access to the given matrix
     *  @param[in]  X Given subspace as flattened array of multi-vectors.
     *  @param[out] Y rotated subspace of top states
     *  @param[in] numberComponents Number of vectors
     *  @param[in] numberCoreStates Number of core states to be used for
     * spectrum splitting
     *  @param[in] mpiCommParent parent mpi communicator
     *  @param[in] interBandGroupComm interpool communicator for parallelization
     * over band groups
     *  @param[in] mpiCommDomain domain decomposition communicator
     *  @param[out] eigenValues of the Projected Hamiltonian
     */
    template <typename T>
    void
    rayleighRitzGEPSpectrumSplitDirect(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      elpaScalaManager &                                 elpaScala,
      T *                                                X,
      T *                                                Y,
      const unsigned int                                 numberComponents,
      const unsigned int                                 numberDofs,
      const unsigned int                                 numberCoreStates,
      const MPI_Comm &                                   mpiCommParent,
      const MPI_Comm &                                   interBandGroupComm,
      const MPI_Comm &                                   mpiCommDomain,
      const bool                                         useMixedPrec,
      std::vector<double> &                              eigenValues,
      const dftParameters &                              dftParams);


    /** @brief Compute Rayleigh-Ritz projection in case of spectrum split using direct diagonalization
     *  (serial version using LAPACK, parallel version using ScaLAPACK)
     *
     *  @param[in] operatorMatrix An object which has access to the given matrix
     *  @param[in]  X Given subspace as flattened array of multi-vectors.
     *  @param[out] Y rotated subspace of top states
     *  @param[in] numberComponents Number of vectors
     *  @param[in] numberCoreStates Number of core states to be used for
     * spectrum splitting
     *  @param[in] mpiCommParent parent mpi communicator
     *  @param[in] interBandGroupComm interpool communicator for parallelization
     * over band groups
     *  @param[in] mpiCommDomain domain decomposition communicator
     *  @param[out] eigenValues of the Projected Hamiltonian
     */
    template <typename T>
    void
    rayleighRitzSpectrumSplitDirect(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      elpaScalaManager &                                 elpaScala,
      const T *                                          X,
      T *                                                Y,
      const unsigned int                                 numberComponents,
      const unsigned int                                 numberDofs,
      const unsigned int                                 numberCoreStates,
      const MPI_Comm &                                   mpiCommParent,
      const MPI_Comm &                                   interBandGroupComm,
      const MPI_Comm &                                   mpiCommDomain,
      const bool                                         useMixedPrec,
      std::vector<double> &                              eigenValues,
      const dftParameters &                              dftParams);


    /** @brief Compute residual norm associated with eigenValue problem of the given operator
     *
     *  @param[in] operatorMatrix An object which has access to the given matrix
     *  @param[in]  X Given subspace as STL vector of dealii vectors
     *  @param[in]  eigenValues eigenValues of the operator
     *  @param[in]  mpiCommParent parent mpi communicator
     *  @param[in]  mpiCommDomain domain decomposition communicator
     *  @param[out] residualNorms of the eigen Value problem
     */
    template <typename T>
    void
    computeEigenResidualNorm(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      T *                                                X,
      const std::vector<double> &                        eigenValues,
      const unsigned int                                 numberComponents,
      const unsigned int                                 numberDofs,
      const MPI_Comm &                                   mpiCommParent,
      const MPI_Comm &                                   mpiCommDomain,
      const MPI_Comm &                                   interBandGroupComm,
      std::vector<double> &                              residualNorm,
      const dftParameters &                              dftParams);

    /** @brief Compute first order response in density matrix with respect to perturbation in the Hamiltonian.
     * Perturbation is computed in the eigenbasis.
     */
    template <typename T>
    void
    densityMatrixEigenBasisFirstOrderResponse(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      T *                                                X,
      const unsigned int                                 N,
      const unsigned int                                 numberLocalDofs,
      const MPI_Comm &                                   mpiCommParent,
      const MPI_Comm &                                   mpiCommDomain,
      const MPI_Comm &                                   interBandGroupComm,
      const std::vector<double> &                        eigenValues,
      const double                                       fermiEnergy,
      std::vector<double> &densityMatDerFermiEnergy,
      elpaScalaManager &   elpaScala,
      const dftParameters &dftParams);

    /**
     * @brief Compute projection of the operator into a subspace spanned by a given orthogonal basis HProjConj=X^{T}*HConj*XConj
     *
     * @param X Vector of Vectors containing multi-wavefunction fields
     * @param numberComponents number of wavefunctions associated with a given node
     * @param ProjMatrix projected small matrix
     */
    void
    XtHX(operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
         const dataTypes::number *                          X,
         const unsigned int                                 numberComponents,
         const unsigned int                                 numberLocalDofs,
         const MPI_Comm &                                   mpiCommDomain,
         const MPI_Comm &                                   interBandGroupComm,
         const dftParameters &                              dftParams,
         std::vector<dataTypes::number> &                   ProjHam);

    /**
     * @brief Compute projection of the operator into a subspace spanned by a given orthogonal basis HProjConj=X^{T}*HConj*XConj
     *
     * @param X Vector of Vectors containing multi-wavefunction fields
     * @param numberComponents number of wavefunctions associated with a given node
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     */
    void
    XtHX(operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
         const dataTypes::number *                          X,
         const unsigned int                                 numberComponents,
         const unsigned int                                 numberLocalDofs,
         const std::shared_ptr<const dftfe::ProcessGrid> &  processGrid,
         const MPI_Comm &                                   mpiCommDomain,
         const MPI_Comm &                                   interBandGroupComm,
         const dftParameters &                              dftParams,
         dftfe::ScaLAPACKMatrix<dataTypes::number> &        projHamPar,
         const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);


    /**
     * @brief Compute projection of the operator into a subspace spanned by a given orthogonal basis HProjConj=X^{T}*HConj*XConj
     *
     * @param X Vector of Vectors containing multi-wavefunction fields
     * @param totalNumberComponents number of wavefunctions associated with a given node
     * @param singlePrecComponents number of wavecfuntions starting from the first for
     * which the project Hamiltionian block will be computed in single
     * procession. However the cross blocks will still be computed in double
     * precision.
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the operation into the given subspace
     */
    void
    XtHXMixedPrec(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      const dataTypes::number *                          X,
      const unsigned int                                 totalNumberComponents,
      const unsigned int                                 singlePrecComponents,
      const unsigned int                                 numberLocalDofs,
      const std::shared_ptr<const dftfe::ProcessGrid> &  processGrid,
      const MPI_Comm &                                   mpiCommDomain,
      const MPI_Comm &                                   interBandGroupComm,
      const dftParameters &                              dftParams,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &        projHamPar,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);

    /**
     * @brief Computes the projection of Hamiltonian and Overlap with only a single extraction.
     * Single extraction will be beneficial in full M, PAW cases.
     * @param X Vector of Vectors containing multi-wavefunction fields
     * @param numberComponents number of wavefunctions associated with a given node
     * @param numberLocalDofs number of DOFs owned in the current procesor
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the Hamiltonian into the given subspace
     * @param projOverlapPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the Overlap into the given subspace
     */
    void
    XtHXXtOX(operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
             const dataTypes::number *                          X,
             const unsigned int                               numberComponents,
             const unsigned int                               numberLocalDofs,
             const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
             const MPI_Comm &                                 mpiCommDomain,
             const MPI_Comm &                           interBandGroupComm,
             const dftParameters &                      dftParams,
             dftfe::ScaLAPACKMatrix<dataTypes::number> &projHamPar,
             dftfe::ScaLAPACKMatrix<dataTypes::number> &projOverlapPar,
             const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);
    /**
     * @brief Computes the projection of Hamiltonian and Overlap with only a single extraction with mixed precision.
     * Single extraction will be beneficial in full M, PAW cases.
     * THe projected Hamiltonain has full precision along blocks of diagonal and
     * for states greater than Ncore THe projected Overlap will be of full
     * precision along the diagonal.
     * @param X Vector of Vectors containing multi-wavefunction fields
     * @param numberComponents number of wavefunctions associated with a given node
     * @param numberLocalDofs number of DOFs owned in the current procesor
     * @param processGrid two-dimensional processor grid corresponding to the parallel projHamPar
     * @param projHamPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the Hamiltonian into the given subspace
     * @param projOverlapPar parallel ScaLAPACKMatrix which stores the computed projection
     * of the Overlap into the given subspace
     */
    void
    XtHXXtOXMixedPrec(
      operatorDFTClass<dftfe::utils::MemorySpace::HOST> &operatorMatrix,
      const dataTypes::number *                          X,
      const unsigned int                                 totalNumberComponents,
      const unsigned int                                 singlePrecComponents,
      const unsigned int                                 numberLocalDofs,
      const std::shared_ptr<const dftfe::ProcessGrid> &  processGrid,
      const MPI_Comm &                                   mpiCommDomain,
      const MPI_Comm &                                   interBandGroupComm,
      const dftParameters &                              dftParams,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &        projHamPar,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &        projOverlapPar,
      const bool onlyHPrimePartForFirstOrderDensityMatResponse = false);

  } // namespace linearAlgebraOperations

} // namespace dftfe
#endif
