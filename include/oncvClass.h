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
// @author  Kartick Ramakrishnan, Sambit Das
//

#ifndef DFTFE_ONCVCLASS_H
#define DFTFE_ONCVCLASS_H

#include <pseudopotentialBaseClass.h>
namespace dftfe
{
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  class oncvClass : public pseudopotentialBaseClass<ValueType, memorySpace>
  {
  public:
    oncvClass(const MPI_Comm                           &mpi_comm_parent,
              const std::string                        &scratchFolderName,
              const std::set<dftfe::uInt>              &atomTypes,
              const bool                                floatingNuclearCharges,
              const dftfe::uInt                         nOMPThreads,
              const std::map<dftfe::uInt, dftfe::uInt> &atomAttributes,
              const bool                                reproducibleOutput,
              const dftfe::Int                          verbosity,
              const bool                                useDevice,
              const bool                                memOptMode);
    /**
     * @brief Initialises all the data members with addresses/values to/of dftClass.
     * @param[in] densityQuadratureId quadratureId for density.
     * @param[in] localContributionQuadratureId quadratureId for local/zero
     * potential
     * @param[in] nuclearChargeQuadratureIdElectro quadratureId for nuclear
     * charges
     * @param[in] densityQuadratureIdElectro quadratureId for density in
     * Electrostatics mesh
     * @param[in] excFunctionalPtr address XC functional pointer
     * @param[in] numEigenValues number of eigenvalues
     * @param[in] atomLocations atomic Coordinates
     * @param[in] imageIds image IDs of periodic cell
     * @param[in] periodicCoords coordinates of image atoms
     */

    void
    initialise(
      std::shared_ptr<
        dftfe::basis::
          FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
        basisOperationsHostPtr,
#if defined(DFTFE_WITH_DEVICE)
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<ValueType,
                                        double,
                                        dftfe::utils::MemorySpace::DEVICE>>
        basisOperationsDevicePtr,
#endif
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        BLASWrapperPtrHost,
#if defined(DFTFE_WITH_DEVICE)
      std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        BLASWrapperPtrDevice,
#endif
      dftfe::uInt                              densityQuadratureId,
      dftfe::uInt                              localContributionQuadratureId,
      dftfe::uInt                              sparsityPatternQuadratureId,
      dftfe::uInt                              nlpspQuadratureId,
      dftfe::uInt                              densityQuadratureIdElectro,
      std::shared_ptr<excManager<memorySpace>> excFunctionalPtr,
      const std::vector<std::vector<double>>  &atomLocations,
      dftfe::uInt                              numEigenValues,
      const bool                               singlePrecNonLocalOperator,
      const bool computeSphericalFnTimesXNonLocalOperator = true);

    /**
     * @brief Initialises all the data members with addresses/values to/of dftClass.
     * @param[in] densityQuadratureId quadratureId for density.
     * @param[in] localContributionQuadratureId quadratureId for local/zero
     * potential
     * @param[in] nuclearChargeQuadratureIdElectro quadratureId for nuclear
     * charges
     * @param[in] densityQuadratureIdElectro quadratureId for density in
     * Electrostatics mesh
     * @param[in] bQuadValuesAllAtoms address of nuclear charge field
     * @param[in] excFunctionalPtr address XC functional pointer
     * @param[in] numEigenValues number of eigenvalues
     * @param[in] atomLocations atomic Coordinates
     * @param[in] imageIds image IDs of periodic cell
     * @param[in] periodicCoords coordinates of image atoms
     */
    void
    initialiseNonLocalContribution(
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<dftfe::Int>          &imageIds,
      const std::vector<std::vector<double>> &periodicCoords,
      const std::vector<double>              &kPointWeights,
      const std::vector<double>              &kPointCoordinates,
      const bool                              updateNonlocalSparsity);


    void
    initialiseNonLocalContribution(
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<dftfe::Int>          &imageIds,
      const std::vector<std::vector<double>> &periodicCoords,
      const std::vector<double>              &kPointWeights,
      const std::vector<double>              &kPointCoordinates,
      const bool                              updateNonlocalSparsity,
      const std::map<dftfe::uInt, std::vector<dftfe::Int>> &sparsityPattern,
      const std::vector<std::vector<dealii::CellId>>
        &elementIdsInAtomCompactSupport,
      const std::vector<std::vector<dftfe::uInt>>
                                     &elementIndexesInAtomCompactSupport,
      const std::vector<dftfe::uInt> &atomIdsInCurrentProcess,
      dftfe::uInt                     numberElements);


    /**
     * @brief Initialises local potential
     */
    void
    initLocalPotential();

    void
    getRadialValenceDensity(dftfe::uInt          Znum,
                            double               rad,
                            std::vector<double> &Val);

    double
    getRadialValenceDensity(dftfe::uInt Znum, double rad);

    double
    getRmaxValenceDensity(dftfe::uInt Znum);

    void
    getRadialCoreDensity(dftfe::uInt          Znum,
                         double               rad,
                         std::vector<double> &Val);

    double
    getRadialCoreDensity(dftfe::uInt Znum, double rad);

    double
    getRmaxCoreDensity(dftfe::uInt Znum);

    double
    getRadialLocalPseudo(dftfe::uInt Znum, double rad);

    double
    getRmaxLocalPot(dftfe::uInt Znum);

    bool
    coreNuclearDensityPresent(dftfe::uInt Znum);
    // Returns the number of Projectors for the given atomID in cooridnates List
    dftfe::uInt
    getTotalNumberOfSphericalFunctionsForAtomId(dftfe::uInt atomId);
    // Returns the Total Number of atoms with support in the processor
    dftfe::uInt
    getTotalNumberOfAtomsInCurrentProcessor();
    // Returns the atomID in coordinates list for the iAtom index.
    dftfe::uInt
    getAtomIdInCurrentProcessor(dftfe::uInt iAtom);


    const dftfe::utils::MemoryStorage<ValueType, memorySpace> &
    getCouplingMatrix(
      CouplingType couplingtype = CouplingType::HamiltonianEntries);


    const std::shared_ptr<
      AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
    getNonLocalOperator();

    const dftfe::utils::MemoryStorage<
      typename dftfe::dataTypes::singlePrecType<ValueType>::type,
      memorySpace> &
    getCouplingMatrixSinglePrec(
      CouplingType couplingtype = CouplingType::HamiltonianEntries);


    const std::shared_ptr<AtomicCenteredNonLocalOperator<
      typename dftfe::dataTypes::singlePrecType<ValueType>::type,
      memorySpace>>
    getNonLocalOperatorSinglePrec();

  private:
    /**
     * @brief Converts the periodic image data structure to relevant form for the container class
     * @param[in] atomLocations atomic Coordinates
     * @param[in] imageIds image IDs of periodic cell
     * @param[in] periodicCoords coordinates of image atoms
     * @param[out] imageIdsTemp image IDs of periodic cell
     * @param[out] imageCoordsTemp coordinates of image atoms
     */
    void
    setImageCoordinates(const std::vector<std::vector<double>> &atomLocations,
                        const std::vector<dftfe::Int>          &imageIds,
                        const std::vector<std::vector<double>> &periodicCoords,
                        std::vector<dftfe::uInt>               &imageIdsTemp,
                        std::vector<double> &imageCoordsTemp);
    /**
     * @brief Creating Density splines for all atomTypes
     */
    void
    createAtomCenteredSphericalFunctionsForDensities();

    void
    computeNonlocalPseudoPotentialConstants();
    void
    createAtomCenteredSphericalFunctionsForProjectors();
    void
    createAtomCenteredSphericalFunctionsForLocalPotential();

    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      d_BLASWrapperHostPtr;
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      d_BLASWrapperDevicePtr;
#endif
    std::vector<std::vector<double>> d_nonLocalPseudoPotentialConstants;
    std::map<dftfe::uInt, std::vector<double>>
      d_atomicNonLocalPseudoPotentialConstants;
    dftfe::utils::MemoryStorage<ValueType, memorySpace> d_couplingMatrixEntries;
    dftfe::utils::MemoryStorage<
      typename dftfe::dataTypes::singlePrecType<ValueType>::type,
      memorySpace>
      d_couplingMatrixEntriesSinglePrec;

    bool d_HamiltonianCouplingMatrixEntriesUpdated;
    bool d_HamiltonianCouplingMatrixSinglePrecEntriesUpdated;
    std::vector<std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicWaveFnsVector;
    std::shared_ptr<AtomCenteredSphericalFunctionContainer>
      d_atomicProjectorFnsContainer;
    std::map<std::pair<dftfe::uInt, dftfe::uInt>,
             std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicProjectorFnsMap;

    // parallel communication objects
    const MPI_Comm    d_mpiCommParent;
    const dftfe::uInt d_this_mpi_process;

    // conditional stream object
    dealii::ConditionalOStream               pcout;
    bool                                     d_useDevice;
    bool                                     d_memoryOptMode;
    dftfe::uInt                              d_densityQuadratureId;
    dftfe::uInt                              d_localContributionQuadratureId;
    dftfe::uInt                              d_nuclearChargeQuadratureIdElectro;
    dftfe::uInt                              d_densityQuadratureIdElectro;
    dftfe::uInt                              d_sparsityPatternQuadratureId;
    dftfe::uInt                              d_nlpspQuadratureId;
    std::shared_ptr<excManager<memorySpace>> d_excManagerPtr;
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::HOST>>
      d_BasisOperatorHostPtr;
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<ValueType, double, dftfe::utils::MemorySpace::DEVICE>>
      d_BasisOperatorDevicePtr;
#endif

    std::map<dftfe::uInt, bool>      d_atomTypeCoreFlagMap;
    bool                             d_floatingNuclearCharges;
    bool                             d_singlePrecNonLocalOperator;
    dftfe::Int                       d_verbosity;
    std::vector<std::vector<double>> d_atomLocations;
    std::set<dftfe::uInt>            d_atomTypes;
    std::map<dftfe::uInt, std::vector<dftfe::uInt>> d_atomTypesList;
    std::string                                     d_dftfeScratchFolderName;
    std::vector<dftfe::Int>                         d_imageIds;
    std::vector<std::vector<double>>                d_imagePositions;
    dftfe::uInt                                     d_numEigenValues;
    dftfe::uInt                                     d_nOMPThreads;

    // Creating Object for Atom Centerd Nonlocal Operator
    std::shared_ptr<AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
      d_nonLocalOperator;

    std::shared_ptr<AtomicCenteredNonLocalOperator<
      typename dftfe::dataTypes::singlePrecType<ValueType>::type,
      memorySpace>>
      d_nonLocalOperatorSinglePrec;


    std::vector<std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicProjectorFnsVector;
    std::vector<
      std::map<dftfe::uInt, std::shared_ptr<AtomCenteredSphericalFunctionBase>>>
      d_atomicLocalPotVector;
    std::vector<
      std::map<dftfe::uInt, std::shared_ptr<AtomCenteredSphericalFunctionBase>>>
      d_atomicValenceDensityVector;
    std::vector<
      std::map<dftfe::uInt, std::shared_ptr<AtomCenteredSphericalFunctionBase>>>
         d_atomicCoreDensityVector;
    bool d_reproducible_output;
    /// FIXME: eventually it should be a map of atomic number to struct-
    /// {valence number, mesh input etc}
    std::map<dftfe::uInt, dftfe::uInt> d_atomTypeAtributes;



  }; // end of class
} // end of namespace dftfe
#endif //  DFTFE_PSEUDOPOTENTIALBASE_H
