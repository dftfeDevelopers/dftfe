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

#include "vector"
#include "map"
#include "AtomCenteredSphericalFunctionValenceDensitySpline.h"
#include "AtomCenteredSphericalFunctionCoreDensitySpline.h"
#include "AtomCenteredSphericalFunctionLocalPotentialSpline.h"
#include "AtomCenteredSphericalFunctionProjectorSpline.h"
#include "AtomCenteredSphericalFunctionContainer.h"
#include "AtomicCenteredNonLocalOperator.h"
#include <memory>
#include <MemorySpaceType.h>
#include <headers.h>
#include <TypeConfig.h>
#include <dftUtils.h>
#include "FEBasisOperations.h"
#include <BLASWrapper.h>
#include <xc.h>
#include <excManager.h>
#ifdef _OPENMP
#  include <omp.h>
#else
#  define omp_get_thread_num() 0
#endif
namespace dftfe
{
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  class oncvClass
  {
  public:
    oncvClass(const MPI_Comm &              mpi_comm_parent,
              const std::string &           scratchFolderName,
              const std::set<unsigned int> &atomTypes,
              const bool                    floatingNuclearCharges,
              const unsigned int            nOMPThreads,
              const std::map<unsigned int, unsigned int> &atomAttributes,
              const bool                                  reproducibleOutput,
              const int                                   verbosity,
              const bool                                  useDevice,
              const bool                                  memOptMode);
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
      unsigned int                             densityQuadratureId,
      unsigned int                             localContributionQuadratureId,
      unsigned int                             sparsityPatternQuadratureId,
      unsigned int                             nlpspQuadratureId,
      unsigned int                             densityQuadratureIdElectro,
      std::shared_ptr<excManager<memorySpace>> excFunctionalPtr,
      const std::vector<std::vector<double>> & atomLocations,
      unsigned int                             numEigenValues,
      const bool                               singlePrecNonLocalOperator);

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
      const std::vector<int> &                imageIds,
      const std::vector<std::vector<double>> &periodicCoords,
      const std::vector<double> &             kPointWeights,
      const std::vector<double> &             kPointCoordinates,
      const bool                              updateNonlocalSparsity);


    void
    initialiseNonLocalContribution(
      const std::vector<std::vector<double>> &        atomLocations,
      const std::vector<int> &                        imageIds,
      const std::vector<std::vector<double>> &        periodicCoords,
      const std::vector<double> &                     kPointWeights,
      const std::vector<double> &                     kPointCoordinates,
      const bool                                      updateNonlocalSparsity,
      const std::map<unsigned int, std::vector<int>> &sparsityPattern,
      const std::vector<std::vector<dealii::CellId>>
        &elementIdsInAtomCompactSupport,
      const std::vector<std::vector<unsigned int>>
        &                              elementIndexesInAtomCompactSupport,
      const std::vector<unsigned int> &atomIdsInCurrentProcess,
      unsigned int                     numberElements);


    /**
     * @brief Initialises local potential
     */
    void
    initLocalPotential();

    void
    getRadialValenceDensity(unsigned int         Znum,
                            double               rad,
                            std::vector<double> &Val);

    double
    getRadialValenceDensity(unsigned int Znum, double rad);

    double
    getRmaxValenceDensity(unsigned int Znum);

    void
    getRadialCoreDensity(unsigned int         Znum,
                         double               rad,
                         std::vector<double> &Val);

    double
    getRadialCoreDensity(unsigned int Znum, double rad);

    double
    getRmaxCoreDensity(unsigned int Znum);

    double
    getRadialLocalPseudo(unsigned int Znum, double rad);

    double
    getRmaxLocalPot(unsigned int Znum);

    bool
    coreNuclearDensityPresent(unsigned int Znum);
    // Returns the number of Projectors for the given atomID in cooridnates List
    unsigned int
    getTotalNumberOfSphericalFunctionsForAtomId(unsigned int atomId);
    // Returns the Total Number of atoms with support in the processor
    unsigned int
    getTotalNumberOfAtomsInCurrentProcessor();
    // Returns the atomID in coordinates list for the iAtom index.
    unsigned int
    getAtomIdInCurrentProcessor(unsigned int iAtom);


    const dftfe::utils::MemoryStorage<ValueType, memorySpace> &
    getCouplingMatrix();


    const std::shared_ptr<
      AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
    getNonLocalOperator();

    const dftfe::utils::MemoryStorage<
      typename dftfe::dataTypes::singlePrecType<ValueType>::type,
      memorySpace> &
    getCouplingMatrixSinglePrec();


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
                        const std::vector<int> &                imageIds,
                        const std::vector<std::vector<double>> &periodicCoords,
                        std::vector<unsigned int> &             imageIdsTemp,
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
    std::map<unsigned int, std::vector<double>>
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
    std::map<std::pair<unsigned int, unsigned int>,
             std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicProjectorFnsMap;

    // parallel communication objects
    const MPI_Comm     d_mpiCommParent;
    const unsigned int d_this_mpi_process;

    // conditional stream object
    dealii::ConditionalOStream               pcout;
    bool                                     d_useDevice;
    bool                                     d_memoryOptMode;
    unsigned int                             d_densityQuadratureId;
    unsigned int                             d_localContributionQuadratureId;
    unsigned int                             d_nuclearChargeQuadratureIdElectro;
    unsigned int                             d_densityQuadratureIdElectro;
    unsigned int                             d_sparsityPatternQuadratureId;
    unsigned int                             d_nlpspQuadratureId;
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

    std::map<unsigned int, bool>     d_atomTypeCoreFlagMap;
    bool                             d_floatingNuclearCharges;
    bool                             d_singlePrecNonLocalOperator;
    int                              d_verbosity;
    std::vector<std::vector<double>> d_atomLocations;
    std::set<unsigned int>           d_atomTypes;
    std::map<unsigned int, std::vector<unsigned int>> d_atomTypesList;
    std::string                                       d_dftfeScratchFolderName;
    std::vector<int>                                  d_imageIds;
    std::vector<std::vector<double>>                  d_imagePositions;
    unsigned int                                      d_numEigenValues;
    unsigned int                                      d_nOMPThreads;

    // Creating Object for Atom Centerd Nonlocal Operator
    std::shared_ptr<AtomicCenteredNonLocalOperator<ValueType, memorySpace>>
      d_nonLocalOperator;

    std::shared_ptr<AtomicCenteredNonLocalOperator<
      typename dftfe::dataTypes::singlePrecType<ValueType>::type,
      memorySpace>>
      d_nonLocalOperatorSinglePrec;


    std::vector<std::shared_ptr<AtomCenteredSphericalFunctionBase>>
      d_atomicProjectorFnsVector;
    std::vector<std::map<unsigned int,
                         std::shared_ptr<AtomCenteredSphericalFunctionBase>>>
      d_atomicLocalPotVector;
    std::vector<std::map<unsigned int,
                         std::shared_ptr<AtomCenteredSphericalFunctionBase>>>
      d_atomicValenceDensityVector;
    std::vector<std::map<unsigned int,
                         std::shared_ptr<AtomCenteredSphericalFunctionBase>>>
         d_atomicCoreDensityVector;
    bool d_reproducible_output;
    /// FIXME: eventually it should be a map of atomic number to struct-
    /// {valence number, mesh input etc}
    std::map<unsigned int, unsigned int> d_atomTypeAtributes;



  }; // end of class
} // end of namespace dftfe
#endif //  DFTFE_PSEUDOPOTENTIALBASE_H
