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

#ifndef DFTFE_EXE_HUBBARDCLASS_H
#define DFTFE_EXE_HUBBARDCLASS_H

#include "headers.h"
#include "FEBasisOperations.h"
#include "AtomCenteredSphericalFunctionBase.h"
#include "AtomicCenteredNonLocalOperator.h"
#include "AuxDensityMatrix.h"
#include "AtomPseudoWavefunctions.h"

namespace dftfe
{
  /**
   * @brief This structure provides the
   * relevant information pertaining to hubbard
   * correction such as U value, orbital Ids and so on
   * for each hubbard species.
   */

  struct hubbardSpecies
  {
  public:
    unsigned int              atomicNumber;
    double                    hubbardValue;
    unsigned int              numProj;
    unsigned int              numberSphericalFunc;
    unsigned int              numberSphericalFuncSq;
    double                    initialOccupation;
    std::vector<unsigned int> nQuantumNum;
    std::vector<unsigned int> lQuantumNum;
  };

  /**
   * @brief This enum class provides the
   * tags for the occupation matrices.
   * This will be relevant during mixing.
   */
  enum class HubbardOccFieldType
  {
    In,
    Out,
    Residual
  };


  /**
   * @brief This class provides the Hubbard correction.
   * This class is an object of ExcDFTPluU Class.
   * @author Vishal Subramanian
   */
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  class hubbard
  {
  public:
    /*
     * @brief The constructor of the Hubbard class.
     * This class takes the relevant mpi communicators.
     * param[in] mpi_comm_parent The global mpi communicator
     * param[in] mpi_comm_domain The mpi communicator for domain decomposition
     * param[in] mpi_comm_interPool The mpi communicator for the k point
     * parallelisation
     */
    hubbard(const MPI_Comm &mpi_comm_parent,
            const MPI_Comm &mpi_comm_domain,
            const MPI_Comm &mpi_comm_interPool);

    /*
     * @brief The init function that initialises the relevant data members of the class
     * This class takes the relevant mpi communicators.
     * param[in] basisOperationsMemPtr The basis Operation class templated to
     * memory space param[in] basisOperationsHostPtr The basis Operation class
     * templated to HOST memory space param[in] BLASWrapperMemPtr The Blas
     * wrapper for performing the Blas operations in memory space param[in]
     * BLASWrapperHostPtr The Blas wrapper for performing the Blas operations in
     * HOST memory space param[in] matrixFreeVectorComponent The matrix vector
     * component corresponding to wavefunctions param[in] densityQuadratureId --
     * the d_nlpspQuadratureId. This cant be changed as it has to compatible
     * with oncv class as that is required to for the force class param[in]
     * sparsityPatternQuadratureId - quadrature required for the class atomic
     * non local operator param[in] numberWaveFunctions - total number of
     * wavefunctions param[in] numSpins - number of spins param[in] dftParam -
     * dft parameters for input condition param[in] scratchFolderName -  the
     * path required to read the atomic orbitals param[in] atomLocations- the
     * atomic locations (cartesian) for all the atoms ( including the atoms with
     * no hubbard corrections)
     *
     */
    void
    init(
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<ValueType, double, memorySpace>>
        basisOperationsMemPtr,
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
        basisOperationsHostPtr,
      std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        BLASWrapperMemPtr,
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                              BLASWrapperHostPtr,
      const unsigned int                      matrixFreeVectorComponent,
      const unsigned int                      densityQuadratureId,
      const unsigned int                      sparsityPatternQuadratureId,
      const unsigned int                      numberWaveFunctions,
      const unsigned int                      numSpins,
      const dftParameters &                   dftParam,
      const std::string &                     scratchFolderName,
      const bool                              singlePrecNonLocalOperator,
      const bool                              updateNonlocalSparsity,
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<std::vector<double>> &atomLocationsFrac,
      const std::vector<int> &                imageIds,
      const std::vector<std::vector<double>> &imagePositions,
      const std::vector<double> &             kPointCoordinates,
      const std::vector<double> &             kPointWeights,
      const std::vector<std::vector<double>> &domainBoundaries);

    /*
     * @brief This function computes the hubbard energy
     *
     * E_U = \sum_I  \sum_{\sigma} 0.5* U \Big( \sum_m( n^{\sigma}_{mm}) -
     * \sum_{m,m'}( n^{\sigma}_{mm'} n^{\sigma}_{m'm})\Big)
     *
     */

    void
    computeEnergyFromOccupationMatrix();

    /*
     * @brief This function computes the occupation matrix at the end of the SCF iteration.
     * The output is stored in HubbardOccFieldType::Out
     *  n^{I,\sigma}_{m'm} =  \sum_k \sum_i f_k f_i < \psi^{\sigma}_i| \phi^{I}_m> < \phi^{I}_m' | \psi^{\sigma}_i >
     */
    void
    computeOccupationMatrix(
      const dftfe::utils::MemoryStorage<ValueType, memorySpace> *X,
      const std::vector<std::vector<double>> &orbitalOccupancy);

    /*
     * @brief This function computes the action of the hubbard potential
     *  V_h \psi^{\sigma}_i = \sum_{I} | \phi^{I}_m>  A^{I \sigma}_{m m'} < \phi^{I}_m' | \psi^{\sigma}_i >
     *  Where A is the coupling matrix. In the case of hubbard, it is a dense
     * matrix for each atom I.
     */

    void
    applyPotentialDueToHubbardCorrection(
      const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &src,
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &      dst,
      const unsigned int inputVecSize,
      const double       factor,
      const unsigned int kPointIndex,
      const unsigned int spinIndex);

    /*
     * @brief This function is similar to above
     * but used in HXCheby(). A different function is required from above as the
     * src requires a different initialisation
     *  V_h \psi^{\sigma}_i = \sum_{I} | \phi^{I}_m>  A^{I \sigma}_{m m'} < \phi^{I}_m' | \psi^{\sigma}_i >
     *  Where A is the coupling matrix. In the case of hubbard, it is a dense
     * matrix for each atom I.
     */
    void
    applyPotentialDueToHubbardCorrectionCheby(
      const dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &src,
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace> &      dst,
      const unsigned int inputVecSize,
      const double       factor,
      const unsigned int kPointIndex,
      const unsigned int spinIndex);

    void
    initialiseOperatorActionOnX(unsigned int kPointIndex);

    void
    initialiseFlattenedDataStructure(unsigned int numVectors);

    void
    initialiseCellWaveFunctionPointers(unsigned int numVectors);

    /*
     * @brief Functions that returns the coupling matrix A required in the apply().
     *
     * A^{I \sigma}_{m m'} = 0.5*U (\delta_{m m'} - n^{I \sigma}_{m m'})
     *
     */
    const dftfe::utils::MemoryStorage<ValueType, memorySpace> &
    getCouplingMatrix(unsigned int spinIndex);

    /*
     * @brief Functions that returns the occupation matrices.
     * This is required for the mixing classes
     */
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &
    getOccMatIn();

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &
    getOccMatRes();

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &
    getOccMatOut();

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &
    getHubbMatrixForMixing();

    /*
     * @brief Function that sets the occupation matrix
     * obtained from the output of the mixing class
     */
    void
    setInOccMatrix(
      const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
        &inputOccMatrix);

    /*
     * @brief Returns the Hubbard energy
     */
    double
    getHubbardEnergy();


    /*
     * @brief Returns the Expectation of the Hubbard potential
     * While using band energy approach to compute the total free energy
     * the expectation of the hubbard potential is included in the band energy.
     * Hence it has to be subtracted and hubbard energy has to be added to the
     * free energy.
     */
    double
    getExpectationOfHubbardPotential();


    /*
        * @brief Get the underlying Atomic nonlocal operator

     */
    const std::shared_ptr<
      AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
    getNonLocalOperator();

  private:
    void
    computeCouplingMatrix();

    void
    computeHubbardOccNumberFromCTransOnX(const bool         isOccOut,
                                         const unsigned int vectorBlockSize,
                                         const unsigned int spinIndex,
                                         const unsigned int kpointIndex);

    void
    setInitialOccMatrix();

    void
    createAtomCenteredSphericalFunctionsForProjectors();

    void
    computeResidualOccMat();

    void
    readHubbardInput(const std::vector<std::vector<double>> &atomLocations,
                     const std::vector<int> &                imageIds,
                     const std::vector<std::vector<double>> &imagePositions);


    std::shared_ptr<
      dftfe::basis::FEBasisOperations<ValueType, double, memorySpace>>
      d_BasisOperatorMemPtr;
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
      d_BasisOperatorHostPtr;

    std::shared_ptr<AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
      d_nonLocalOperator;

    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      d_BLASWrapperMemPtr;

    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                           d_BLASWrapperHostPtr;
    std::map<unsigned int, hubbardSpecies> d_hubbardSpeciesData;
    std::map<std::pair<unsigned int, unsigned int>,
             std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicProjectorFnsMap;

    std::shared_ptr<AtomCenteredSphericalFunctionContainer>
      d_atomicProjectorFnsContainer;

    std::vector<double>              d_kPointWeights;
    std::vector<std::vector<double>> d_domainBoundaries;
    const dftParameters *            d_dftParamsPtr;
    std::vector<double>              d_kPointCoordinates;

    unsigned int d_numKPoints;

    double d_atomOrbitalMaxLength;

    const MPI_Comm d_mpi_comm_parent;
    const MPI_Comm d_mpi_comm_domain;
    const MPI_Comm d_mpi_comm_interPool;

    unsigned int n_mpi_processes, this_mpi_process;

    unsigned int              d_numSpins;
    std::vector<unsigned int> d_procLocalAtomId;

    dealii::ConditionalOStream pcout;

    std::vector<double>              d_atomicCoords;
    std::vector<std::vector<double>> d_periodicImagesCoords;
    std::vector<int>                 d_imageIds;
    std::vector<unsigned int>        d_mapAtomToHubbardIds;
    std::vector<unsigned int>        d_mapAtomToAtomicNumber;

    double       d_spinPolarizedFactor;
    unsigned int d_noOfSpin;
    std::string  d_dftfeScratchFolderName;

    std::vector<dftfe::utils::MemoryStorage<ValueType, memorySpace>>
      d_couplingMatrixEntries;

    std::map<
      HubbardOccFieldType,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
                 d_occupationMatrix;
    unsigned int d_noSpecies;

    unsigned int d_densityQuadratureId, d_numberWaveFunctions;

    unsigned int              d_numTotalOccMatrixEntriesPerSpin;
    std::vector<unsigned int> d_OccMatrixEntryStartForAtom;

    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
      d_hubbNonLocalProjectorTimesVectorBlock;
    dftfe::utils::MemoryStorage<ValueType, memorySpace>
      d_cellWaveFunctionMatrixSrc, d_cellWaveFunctionMatrixDst;

    unsigned int d_cellsBlockSizeApply;
    unsigned int d_verbosity;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_hubbOccMatAfterMixing;

    double d_hubbardEnergy;
    double d_expectationOfHubbardPotential;
  };
} // namespace dftfe

#endif // DFTFE_EXE_HUBBARDCLASS_H
