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
    dftfe::uInt              atomicNumber;
    double                   hubbardValue;
    dftfe::uInt              numProj;
    dftfe::uInt              numberSphericalFunc;
    dftfe::uInt              numberSphericalFuncSq;
    double                   initialOccupation;
    std::vector<dftfe::uInt> nQuantumNum;
    std::vector<dftfe::uInt> lQuantumNum;
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
            const MPI_Comm &mpi_comm_interPool,
            const MPI_Comm &mpi_comm_interBandGroup);

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
      const dftfe::uInt                       matrixFreeVectorComponent,
      const dftfe::uInt                       densityQuadratureId,
      const dftfe::uInt                       sparsityPatternQuadratureId,
      const dftfe::uInt                       numberWaveFunctions,
      const dftfe::uInt                       numSpins,
      const dftParameters                    &dftParam,
      const std::string                      &scratchFolderName,
      const bool                              singlePrecNonLocalOperator,
      const bool                              updateNonlocalSparsity,
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<std::vector<double>> &atomLocationsFrac,
      const std::vector<dftfe::Int>          &imageIds,
      const std::vector<std::vector<double>> &imagePositions,
      const std::vector<double>              &kPointCoordinates,
      const std::vector<double>              &kPointWeights,
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
      dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>       &dst,
      const dftfe::uInt inputVecSize,
      const dftfe::uInt kPointIndex,
      const dftfe::uInt spinIndex);

    void
    applyPotentialDueToHubbardCorrection(
      const dftfe::linearAlgebra::MultiVector<
        typename dftfe::dataTypes::singlePrecType<ValueType>::type,
        memorySpace> &src,
      dftfe::linearAlgebra::MultiVector<
        typename dftfe::dataTypes::singlePrecType<ValueType>::type,
        memorySpace>   &dst,
      const dftfe::uInt inputVecSize,
      const dftfe::uInt kPointIndex,
      const dftfe::uInt spinIndex);

    void
    initialiseOperatorActionOnX(dftfe::uInt kPointIndex);

    void
    initialiseFlattenedDataStructure(dftfe::uInt numVectors);


    /*
     * @brief Functions that returns the coupling matrix A required in the apply().
     *
     * A^{I \sigma}_{m m'} = 0.5*U (\delta_{m m'} - n^{I \sigma}_{m m'})
     *
     */
    const dftfe::utils::MemoryStorage<ValueType, memorySpace> &
    getCouplingMatrix(dftfe::uInt spinIndex);

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

    dftfe::uInt
    getTotalNumberOfSphericalFunctionsForAtomId(dftfe::uInt iAtom);

    dftfe::uInt
    getGlobalAtomId(dftfe::uInt iAtom);


    /*
     * @brief Write the hubbard occupation numbers to a
     * file. Used for nscf calculations.
     */
    void
    writeHubbOccToFile();

    /*
     * @brief Read the hubbard occupation numbers from a file.
     * Used for nscf calculations.
     */
    void
    readHubbOccFromFile();


  private:
    std::map<dftfe::uInt, dftfe::uInt> d_mapHubbardAtomToGlobalAtomId;

    dftfe::uInt d_totalNumHubbAtoms;
    void
    computeCouplingMatrix();

    void
    computeHubbardOccNumberFromCTransOnX(const bool        isOccOut,
                                         const dftfe::uInt vectorBlockSize,
                                         const dftfe::uInt spinIndex,
                                         const dftfe::uInt kpointIndex);

    void
    setInitialOccMatrix();

    void
    createAtomCenteredSphericalFunctionsForProjectors();

    void
    computeResidualOccMat();

    void
    readHubbardInput(const std::vector<std::vector<double>> &atomLocations,
                     const std::vector<dftfe::Int>          &imageIds,
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

    std::shared_ptr<
      AtomicCenteredNonLocalOperator<dataTypes::numberFP32, memorySpace>>
      d_nonLocalOperatorSinglePrec;

    bool d_useSinglePrec;

    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      d_BLASWrapperMemPtr;

    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                          d_BLASWrapperHostPtr;
    std::map<dftfe::uInt, hubbardSpecies> d_hubbardSpeciesData;
    std::map<std::pair<dftfe::uInt, dftfe::uInt>,
             std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicProjectorFnsMap;

    std::shared_ptr<AtomCenteredSphericalFunctionContainer>
      d_atomicProjectorFnsContainer;

    std::vector<double>              d_kPointWeights;
    std::vector<std::vector<double>> d_domainBoundaries;
    const dftParameters             *d_dftParamsPtr;
    std::vector<double>              d_kPointCoordinates;

    dftfe::uInt d_numKPoints;

    double d_atomOrbitalMaxLength;

    const MPI_Comm d_mpi_comm_parent;
    const MPI_Comm d_mpi_comm_domain;
    const MPI_Comm d_mpi_comm_interPool;
    const MPI_Comm d_mpi_comm_interBand;

    dftfe::uInt n_mpi_processes, this_mpi_process;

    std::vector<dftfe::uInt> d_bandGroupLowHighPlusOneIndices;

    dftfe::uInt              d_numSpins;
    std::vector<dftfe::uInt> d_procLocalAtomId;

    dealii::ConditionalOStream pcout;

    std::vector<double>              d_atomicCoords;
    std::vector<double>              d_initialAtomicSpin;
    std::vector<std::vector<double>> d_periodicImagesCoords;
    std::vector<dftfe::Int>          d_imageIds;
    std::vector<dftfe::uInt>         d_mapAtomToHubbardIds;
    std::vector<dftfe::uInt>         d_mapAtomToAtomicNumber;

    double      d_spinPolarizedFactor;
    dftfe::uInt d_noOfSpin;
    std::string d_dftfeScratchFolderName;

    std::vector<dftfe::utils::MemoryStorage<ValueType, memorySpace>>
      d_couplingMatrixEntries;

    std::vector<dftfe::utils::MemoryStorage<
      typename dftfe::dataTypes::singlePrecType<ValueType>::type,
      memorySpace>>
      d_couplingMatrixEntriesSinglePrec;
    std::map<
      HubbardOccFieldType,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
                d_occupationMatrix;
    dftfe::uInt d_noSpecies;

    dftfe::uInt d_densityQuadratureId, d_numberWaveFunctions;

    dftfe::uInt              d_numTotalOccMatrixEntriesPerSpin;
    std::vector<dftfe::uInt> d_OccMatrixEntryStartForAtom;

    dftfe::linearAlgebra::MultiVector<ValueType, memorySpace>
      d_hubbNonLocalProjectorTimesVectorBlock;

    dftfe::linearAlgebra::MultiVector<
      typename dftfe::dataTypes::singlePrecType<ValueType>::type,
      memorySpace>
      d_hubbNonLocalProjectorTimesVectorBlockSinglePrec;


    dftfe::uInt d_cellsBlockSizeApply;
    dftfe::uInt d_verbosity;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      d_hubbOccMatAfterMixing;

    double d_hubbardEnergy;
    double d_expectationOfHubbardPotential;

    dftfe::uInt d_maxOccMatSizePerAtom;
  };
} // namespace dftfe

#endif // DFTFE_EXE_HUBBARDCLASS_H
