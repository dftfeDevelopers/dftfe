// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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

#ifndef dft_H_
#define dft_H_
#include <constants.h>
#include <constraintMatrixInfo.h>
#include <elpaScalaManager.h>
#include <headers.h>
#include <MemorySpaceType.h>
#include <MemoryStorage.h>
#include <FEBasisOperations.h>
#include <BLASWrapper.h>

#include <complex>
#include <deque>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

#ifdef DFTFE_WITH_DEVICE
#  include <chebyshevOrthogonalizedSubspaceIterationSolverDevice.h>
#  include <constraintMatrixInfoDevice.h>
#  include <kohnShamDFTOperatorDevice.h>
#  include "deviceKernelsGeneric.h"
#  include <poissonSolverProblemDevice.h>
#  include <kerkerSolverProblemDevice.h>
#  include <linearSolverCGDevice.h>
#  include <deviceDirectCCLWrapper.h>
#endif

#include <chebyshevOrthogonalizedSubspaceIterationSolver.h>
#include <dealiiLinearSolver.h>
#include <dftParameters.h>
#include <eigenSolver.h>
#include <interpolation.h>
#include <kerkerSolverProblem.h>
#include <kohnShamDFTOperator.h>
#include <meshMovementAffineTransform.h>
#include <meshMovementGaussian.h>
#include <poissonSolverProblem.h>
#include <triangulationManager.h>
#include <vselfBinsManager.h>
#include <excManager.h>
#include <dftd.h>
#include <force.h>
#include "dftBase.h"
#ifdef USE_PETSC
#  include <petsc.h>

#  include <slepceps.h>
#endif

#include <mixingClass.h>


namespace dftfe
{
  //
  // Initialize Namespace
  //



#ifndef DOXYGEN_SHOULD_SKIP_THIS

  struct orbital
  {
    unsigned int                atomID;
    unsigned int                waveID;
    unsigned int                Z, n, l;
    int                         m;
    alglib::spline1dinterpolant psi;
  };

  /* code that must be skipped by Doxygen */
  // forward declarations
  template <unsigned int T1, unsigned int T2>
  class symmetryClass;
  template <unsigned int T1, unsigned int T2>
  class forceClass;

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

  /**
   * @brief This class is the primary interface location of all other parts of the DFT-FE code
   * for all steps involved in obtaining the Kohn-Sham DFT ground-state
   * solution.
   *
   * @author Shiva Rudraraju, Phani Motamarri, Sambit Das
   */
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  class dftClass : public dftBase
  {
    friend class kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>;

#ifdef DFTFE_WITH_DEVICE
    friend class kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>;
#endif

    friend class forceClass<FEOrder, FEOrderElectro>;

    friend class symmetryClass<FEOrder, FEOrderElectro>;

  public:
    /**
     * @brief dftClass constructor
     *
     *  @param[in] mpi_comm_parent parent communicator
     *  @param[in] mpi_comm_domain  mpi_communicator for domain decomposition
     * parallelization
     *  @param[in] interpoolcomm  mpi_communicator for parallelization over k
     * points
     *  @param[in] interBandGroupComm  mpi_communicator for parallelization over
     * bands
     *  @param[in] scratchFolderName  scratch folder name
     *  @param[in] dftParams  dftParameters object containg parameter values
     * parsed from an input parameter file in dftfeWrapper class
     */
    dftClass(const MPI_Comm &   mpiCommParent,
             const MPI_Comm &   mpi_comm_domain,
             const MPI_Comm &   interpoolcomm,
             const MPI_Comm &   interBandGroupComm,
             const std::string &scratchFolderName,
             dftParameters &    dftParams);

    /**
     * @brief dftClass destructor
     */
    ~dftClass();

    /**
     * @brief atomic system pre-processing steps.
     *
     * Reads the coordinates of the atoms.
     * If periodic calculation, reads fractional coordinates of atoms in the
     * unit-cell, lattice vectors, kPoint quadrature rules to be used and also
     * generates image atoms. Also determines orbital-ordering
     */
    void
    set();

    /**
     * @brief Does KSDFT problem pre-processing steps including mesh generation calls.
     */
    void
    init();

    /**
     * @brief Does KSDFT problem pre-processing steps but without remeshing.
     */
    void
    initNoRemesh(const bool updateImagesAndKPointsAndVselfBins = true,
                 const bool checkSmearedChargeWidthsForOverlap = true,
                 const bool useSingleAtomSolutionOverride      = false,
                 const bool isMeshDeformed                     = false);



    /**
     * @brief FIXME: legacy call, move to main.cc
     */
    void
    run();

    /**
     * @brief Writes inital density and mesh to file.
     */
    void
    writeMesh();

    /**
     * @brief compute approximation to ground-state without solving the SCF iteration
     */
    void
    solveNoSCF();
    /**
     * @brief Kohn-Sham ground-state solve using SCF iteration
     *
     * @return tuple of boolean flag on whether scf converged,
     *  and L2 norm of residual electron-density of the last SCF iteration step
     *
     */
    std::tuple<bool, double>
    solve(const bool computeForces                 = true,
          const bool computestress                 = true,
          const bool restartGroundStateCalcFromChk = false);

    void
    computeStress();

    void
    trivialSolveForStress();


    void
    computeOutputDensityDirectionalDerivative(
      const distributedCPUVec<double> &v,
      const distributedCPUVec<double> &vSpin0,
      const distributedCPUVec<double> &vSpin1,
      distributedCPUVec<double> &      fv,
      distributedCPUVec<double> &      fvSpin0,
      distributedCPUVec<double> &      fvSpin1);

    /**
     * @brief Copies the density stored in std::map<dealii:cellId, std::vector<double>> into a flattened std::vector<double> format
     *
     *
     */
    void
    copyDensityToVector(
      const std::shared_ptr<std::map<dealii::CellId, std::vector<double>>>
        &                  rhoValues,
      std::vector<double> &rhoValuesVector);

    /**
     * @brief Copies the density stored in  a flattened std::vector<double> format to std::map<dealii:cellId, std::vector<double>>
     *
     */
    void
    copyDensityFromVector(
      const std::vector<double> &rhoValuesVector,
      std::shared_ptr<std::map<dealii::CellId, std::vector<double>>>
        &rhoValues);

    /**
     * @brief Copies the gradient of density stored in a std::map<dealii:cellId, std::vector<double>>
     * into a flattened std::vector<double> format
     *
     *
     */
    void
    copyGradDensityToVector(
      const std::shared_ptr<std::map<dealii::CellId, std::vector<double>>>
        &                  gradRhoValues,
      std::vector<double> &gradRhoValuesVector);

    /**
     * @brief Copies the gradient of density stored in  a flattened std::vector<double> format
     * to std::map<dealii:cellId, std::vector<double>>
     *
     */
    void
    copyGradDensityFromVector(
      const std::vector<double> &gradRhoValuesVector,
      std::shared_ptr<std::map<dealii::CellId, std::vector<double>>>
        &gradRhoValues);

    /**
     * @brief Computes the total density from the spin polarised densities
     *
     */
    void
    computeTotalDensityFromSpinPolarised(
      const std::shared_ptr<std::map<dealii::CellId, std::vector<double>>>
        &rhoSpinValues,
      std::shared_ptr<std::map<dealii::CellId, std::vector<double>>>
        &rhoValues);

    /**
     * @brief Computes the total gradient of density from the spin polarised densities
     *
     */
    void
    computeTotalGradDensityFromSpinPolarised(
      const std::shared_ptr<std::map<dealii::CellId, std::vector<double>>>
        &gradRhoSpinValues,
      std::shared_ptr<std::map<dealii::CellId, std::vector<double>>>
        &gradRhoValues);

    /**
     * @brief Computes the JxW values used in the \int \rho
     *
     */
    void
    computeJxWForRho(std::vector<double> &vecJxW);
    void
    initializeKohnShamDFTOperator(const bool initializeCublas = true);


    void
    reInitializeKohnShamDFTOperator();


    void
    finalizeKohnShamDFTOperator();


    double
    getInternalEnergy() const;

    double
    getEntropicEnergy() const;

    double
    getFreeEnergy() const;

    distributedCPUVec<double>
    getRhoNodalOut() const;

    distributedCPUVec<double>
    getRhoNodalSplitOut() const;

    double
    getTotalChargeforRhoSplit();

    void
    resetRhoNodalIn(distributedCPUVec<double> &OutDensity);

    virtual void
    resetRhoNodalSplitIn(distributedCPUVec<double> &OutDensity);

    /**
     * @brief Number of Kohn-Sham eigen values to be computed
     */
    unsigned int d_numEigenValues;

    /**
     * @brief Number of Kohn-Sham eigen values to be computed in the Rayleigh-Ritz step
     * after spectrum splitting.
     */
    unsigned int d_numEigenValuesRR;

    /**
     * @brief Number of random wavefunctions
     */
    unsigned int d_nonAtomicWaveFunctions;

    void
    readkPointData();

    /**
     *@brief Get local dofs global indices real
     */
    const std::vector<dealii::types::global_dof_index> &
    getLocalDofIndicesReal() const;

    /**
     *@brief Get local dofs global indices imag
     */
    const std::vector<dealii::types::global_dof_index> &
    getLocalDofIndicesImag() const;

    /**
     *@brief Get local dofs local proc indices real
     */
    const std::vector<dealii::types::global_dof_index> &
    getLocalProcDofIndicesReal() const;

    /**
     *@brief Get local dofs local proc indices imag
     */
    const std::vector<dealii::types::global_dof_index> &
    getLocalProcDofIndicesImag() const;

    /**
     *@brief Get matrix free data object
     */
    const dealii::MatrixFree<3, double> &
    getMatrixFreeData() const;


    /** @brief Updates atom positions, remeshes/moves mesh and calls appropriate reinits.
     *
     *  Function to update the atom positions and mesh based on the provided
     * displacement input. Depending on the maximum displacement magnitude this
     * function decides wether to do auto remeshing or move mesh using Gaussian
     * functions. Additionaly this function also wraps the atom position across
     * the periodic boundary if the atom moves across it beyond a certain
     * magnitude. In case of floating atoms, only the atomic positions are
     * updated keeping the mesh fixed. This function also calls initNoRemesh to
     * reinitialize all the required FEM and KSDFT objects.
     *
     *  @param[in] globalAtomsDisplacements vector containing the displacements
     * (from current position) of all atoms (global).
     *  @return void.
     */
    void
    updateAtomPositionsAndMoveMesh(
      const std::vector<dealii::Tensor<1, 3, double>> &globalAtomsDisplacements,
      const double maxJacobianRatioFactor         = 1.25,
      const bool   useSingleAtomSolutionsOverride = false);


    /**
     * @brief writes the current domain bounding vectors and atom coordinates to files, which are required for
     * geometry relaxation restart

     */
    void
    writeDomainAndAtomCoordinates();

    /**
     * @brief writes the current domain bounding vectors and atom coordinates to files for
     * structural optimization and dynamics restarts. The coordinates are stored
     * as: 1. fractional for semi-periodic/periodic 2. Cartesian for
     * non-periodic.
     * @param[in] Path The folder path to store the atom coordinates required
     * during restart.
     */
    void
    writeDomainAndAtomCoordinates(const std::string Path) const;

    /**
     * @brief writes atomistics data for subsequent post-processing. Related to
     * WRITE STRUCTURE ENERGY FORCES DATA POST PROCESS input parameter.
     * @param[in] Path The folder path to store the atomistics data.
     */
    void
    writeStructureEnergyForcesDataPostProcess(const std::string Path) const;

    /**
     * @brief writes quadrature grid information and associated spin-up
     * and spin-down electron-density for post-processing
     * @param[in] Path The folder path to store the atomistics data.
     */
    virtual void
    writeGSElectronDensity(const std::string Path) const;


    /**
     * @brief Gets the current atom Locations in cartesian form
     * (origin at center of domain) from dftClass
     */
    std::vector<std::vector<double>>
    getAtomLocationsCart() const;

    /**
     * @brief Gets the current atom Locations in fractional form
     * from dftClass (only applicable for periodic and semi-periodic BCs)
     */
    std::vector<std::vector<double>>
    getAtomLocationsFrac() const;



    /**
     * @brief Gets the current cell lattice vectors
     *
     *  @return std::vector<std::vector<double>> 3 \times 3 matrix,lattice[i][j]
     *  corresponds to jth component of ith lattice vector
     */
    std::vector<std::vector<double>>
    getCell() const;

    /**
     * @brief Gets the current cell volume
     *
     */
    double
    getCellVolume() const;

    /**
     * @brief Gets the current atom types from dftClass
     */
    std::set<unsigned int>
    getAtomTypes() const;

    /**
     * @brief Gets the current atomic forces from dftClass
     */
    std::vector<double>
    getForceonAtoms() const;

    /**
     * @brief Gets the current cell stress from dftClass
     */
    dealii::Tensor<2, 3, double>
    getCellStress() const;

    /**
     * @brief Get reference to dftParameters object
     */
    dftParameters &
    getParametersObject() const;

  private:
    /**
     * @brief generate image charges and update k point cartesian coordinates based
     * on current lattice vectors
     */
    void
    initImageChargesUpdateKPoints(bool flag = true);


    /**
     *@brief project ground state electron density from previous mesh into
     * the new mesh to be used as initial guess for the new ground state solve
     */
    void
    projectPreviousGroundStateRho();

    /**
     *@brief save triangulation information and rho quadrature data to checkpoint file for restarts
     */
    void
    saveTriaInfoAndRhoNodalData();

    /**
     *@brief load triangulation information rho quadrature data from checkpoint file for restarted run
     */
    void
    loadTriaInfoAndRhoNodalData();

    void
    generateMPGrid();
    void
    writeMesh(std::string meshFileName);

    /// creates datastructures related to periodic image charges
    void
    generateImageCharges(const double                      pspCutOff,
                         std::vector<int> &                imageIds,
                         std::vector<double> &             imageCharges,
                         std::vector<std::vector<double>> &imagePositions);

    void
    createMasterChargeIdToImageIdMaps(
      const double                            pspCutOff,
      const std::vector<int> &                imageIds,
      const std::vector<std::vector<double>> &imagePositions,
      std::vector<std::vector<int>> &         globalChargeIdToImageIdMap);

    void
    determineOrbitalFilling();

    //
    // generate mesh using a-posteriori error estimates
    //
    void
    aposterioriMeshGenerate();
    dataTypes::number
    computeTraceXtHX(unsigned int numberWaveFunctionsEstimate);
    double
    computeTraceXtKX(unsigned int numberWaveFunctionsEstimate);


    /**
     *@brief  moves the triangulation vertices using Gaussians such that the all atoms are on triangulation vertices
     */
    void moveMeshToAtoms(dealii::Triangulation<3, 3> &triangulationMove,
                         dealii::Triangulation<3, 3> &triangulationSerial,
                         bool                         reuseFlag      = false,
                         bool                         moveSubdivided = false);

    /**
     *@brief  a
     */
    void
    calculateSmearedChargeWidths();

    /**
     *@brief  a
     */
    void
    calculateNearestAtomDistances();

    /**
     * Initializes the guess of electron-density and single-atom wavefunctions
     * on the mesh, maps finite-element nodes to given atomic positions,
     * initializes pseudopotential files and exchange-correlation functionals to
     * be used based on user-choice. In periodic problems, periodic faces are
     * mapped here. Further finite-element nodes to be pinned for solving the
     * Poisson problem electro-static potential is set here
     */
    void initUnmovedTriangulation(
      dealii::parallel::distributed::Triangulation<3> &triangulation);
    void
    initBoundaryConditions(const bool meshOnlyDeformed                 = false,
                           const bool vselfPerturbationUpdateForStress = false);
    void
    initElectronicFields();
    void
    initPseudoPotentialAll(const bool updateNonlocalSparsity = true);

    /**
     * create a dofHandler containing finite-element interpolating polynomial
     * twice of the original polynomial required for Kerker mixing and
     * initialize various objects related to this refined dofHandler
     */
    void createpRefinedDofHandler(
      dealii::parallel::distributed::Triangulation<3> &triangulation);
    void
    initpRefinedObjects(const bool meshOnlyDeformed,
                        const bool vselfPerturbationUpdateForStress = false);

    /**
     *@brief interpolate rho nodal data to quadrature data using FEEvaluation
     *
     *@param[in] matrixFreeData matrix free data object
     *@param[in] nodalField nodal data to be interpolated
     *@param[out] quadratureValueData to be computed at quadrature points
     *@param[out] quadratureGradValueData to be computed at quadrature points
     *@param[in] isEvaluateGradData denotes a flag to evaluate gradients or not
     */
    void interpolateRhoNodalDataToQuadratureDataGeneral(
      dealii::MatrixFree<3, double> &                matrixFreeData,
      const unsigned int                             dofHandlerId,
      const unsigned int                             quadratureId,
      const distributedCPUVec<double> &              nodalField,
      std::map<dealii::CellId, std::vector<double>> &quadratureValueData,
      std::map<dealii::CellId, std::vector<double>> &quadratureGradValueData,
      std::map<dealii::CellId, std::vector<double>> &quadratureHessianValueData,
      const bool                                     isEvaluateGradData = false,
      const bool isEvaluateHessianData = false);

    /**
     *@brief interpolate spin rho nodal data to quadrature data using FEEvaluation
     *
     */
    void interpolateRhoSpinNodalDataToQuadratureDataGeneral(
      dealii::MatrixFree<3, double> &                matrixFreeData,
      const unsigned int                             dofHandlerId,
      const unsigned int                             quadratureId,
      const distributedCPUVec<double> &              nodalFieldSpin0,
      const distributedCPUVec<double> &              nodalFieldSpin1,
      std::map<dealii::CellId, std::vector<double>> &quadratureValueData,
      std::map<dealii::CellId, std::vector<double>> &quadratureGradValueData,
      std::map<dealii::CellId, std::vector<double>> &quadratureHessianValueData,
      const bool                                     isEvaluateGradData = false,
      const bool isEvaluateHessianData = false);

    /**
     *@brief interpolate nodal data to quadrature data using FEEvaluation
     *
     *@param[in] matrixFreeData matrix free data object
     *@param[in] nodalField nodal data to be interpolated
     *@param[out] quadratureValueData to be computed at quadrature points
     *@param[out] quadratureGradValueData to be computed at quadrature points
     *@param[in] isEvaluateGradData denotes a flag to evaluate gradients or not
     */
    void interpolateElectroNodalDataToQuadratureDataGeneral(
      dealii::MatrixFree<3, double> &                matrixFreeData,
      const unsigned int                             dofHandlerId,
      const unsigned int                             quadratureId,
      const distributedCPUVec<double> &              nodalField,
      std::map<dealii::CellId, std::vector<double>> &quadratureValueData,
      std::map<dealii::CellId, std::vector<double>> &quadratureGradValueData,
      const bool isEvaluateGradData = false);


    /**
     *@brief interpolate nodal data to quadrature data using FEEvaluation
     *
     *@param[in] matrixFreeData matrix free data object
     *@param[in] nodalField nodal data to be interpolated
     *@param[in] matrix free dofHandler id
     *@param[in] matrix free quadrature id
     *@param[out] quadratureValueData to be computed at quadrature points
     *@param[out] quadratureGradValueData to be computed at quadrature points
     *@param[in] isEvaluateGradData denotes a flag to evaluate gradients or not
     */
    void interpolateRhoNodalDataToQuadratureDataLpsp(
      dealii::MatrixFree<3, double> &                matrixFreeData,
      const unsigned int                             dofHandlerId,
      const unsigned int                             quadratureId,
      const distributedCPUVec<double> &              nodalField,
      std::map<dealii::CellId, std::vector<double>> &quadratureValueData,
      std::map<dealii::CellId, std::vector<double>> &quadratureGradValueData,
      const bool                                     isEvaluateGradData);

    /**
     *@brief add atomic densities at quadrature points
     *
     */
    void
    addAtomicRhoQuadValuesGradients(
      std::map<dealii::CellId, std::vector<double>> &quadratureValueData,
      std::map<dealii::CellId, std::vector<double>> &quadratureGradValueData,
      const bool isConsiderGradData = false);

    /**
     *@brief subtract atomic densities at quadrature points
     *
     */
    void
    subtractAtomicRhoQuadValuesGradients(
      std::map<dealii::CellId, std::vector<double>> &quadratureValueData,
      std::map<dealii::CellId, std::vector<double>> &quadratureGradValueData,
      const bool isConsiderGradData = false);


    /**
     *@brief Finds the global dof ids of the nodes containing atoms.
     *
     * @param[in] dofHandler
     * @param[out] atomNodeIdToChargeValueMap local map of global dof id to atom
     *charge id
     */
    void
    locateAtomCoreNodes(const dealii::DoFHandler<3> &_dofHandler,
                        std::map<dealii::types::global_dof_index, double>
                          &atomNodeIdToChargeValueMap);

    /**
     *@brief Sets homogeneous dirichlet boundary conditions on a node farthest from
     * all atoms (pinned node). This is only done in case of periodic boundary
     *conditions to get an unique solution to the total electrostatic potential
     *problem.
     *
     * @param[in] dofHandler
     * @param[in] constraintMatrixBase base dealii::AffineConstraints<double>
     *object
     * @param[out] constraintMatrix dealii::AffineConstraints<double> object
     *with homogeneous Dirichlet boundary condition entries added
     */
    void
    locatePeriodicPinnedNodes(
      const dealii::DoFHandler<3> &            _dofHandler,
      const dealii::AffineConstraints<double> &constraintMatrixBase,
      dealii::AffineConstraints<double> &      constraintMatrix);

    void
    initAtomicRho();

    double d_atomicRhoScalingFac;

    void
    initRho();
    void
    initCoreRho();
    void
    computeRhoInitialGuessFromPSI(
      std::vector<std::vector<distributedCPUVec<double>>> eigenVectors);
    void
    clearRhoData();

    /**
     *@brief computes nodal electron-density from cell quadrature data using project function of dealii
     */
    void
    computeNodalRhoFromQuadData();


    /**
     *@brief computes density nodal data from wavefunctions
     */
    void
    computeRhoNodalFromPSI(
#ifdef DFTFE_WITH_DEVICE
      kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
        &kohnShamDFTEigenOperator,
#endif
      kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
        &  kohnShamDFTEigenOperatorCPU,
      bool isConsiderSpectrumSplitting);


    void
    computeRhoNodalFirstOrderResponseFromPSIAndPSIPrime(
#ifdef DFTFE_WITH_DEVICE
      kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
        &kohnShamDFTEigenOperatorDevice,
#endif
      kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
        &                        kohnShamDFTEigenOperatorCPU,
      distributedCPUVec<double> &fv,
      distributedCPUVec<double> &fvSpin0,
      distributedCPUVec<double> &fvSpin1);

    void
    noRemeshRhoDataInit();
    void
    readPSI();
    void
    readPSIRadialValues();
    void
    loadPSIFiles(unsigned int  Z,
                 unsigned int  n,
                 unsigned int  l,
                 unsigned int &flag);
    void
    initLocalPseudoPotential(
      const dealii::DoFHandler<3> &            _dofHandler,
      const unsigned int                       lpspQuadratureId,
      const dealii::MatrixFree<3, double> &    _matrix_free_data,
      const unsigned int                       _phiExtDofHandlerIndex,
      const dealii::AffineConstraints<double> &phiExtConstraintMatrix,
      const std::map<dealii::types::global_dof_index, dealii::Point<3>>
        &                                              supportPoints,
      const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinManager,
      distributedCPUVec<double> &                      phiExt,
      std::map<dealii::CellId, std::vector<double>> &  _pseudoValues,
      std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
        &_pseudoValuesAtoms);
    void
    initNonLocalPseudoPotential_OV();
    void
    computeSparseStructureNonLocalProjectors_OV();


    /**
     *@brief Sets homegeneous dirichlet boundary conditions for total potential constraints on
     * non-periodic boundary (boundary id==0).
     *
     * @param[in] dofHandler
     * @param[out] constraintMatrix dealii::AffineConstraints<double> object
     *with homogeneous Dirichlet boundary condition entries added
     */
    void
    applyHomogeneousDirichletBC(
      const dealii::DoFHandler<3> &            _dofHandler,
      const dealii::AffineConstraints<double> &onlyHangingNodeConstraints,
      dealii::AffineConstraints<double> &      constraintMatrix);

    void
    computeElementalOVProjectorKets();

    /**
     *@brief Computes total charge by integrating the electron-density
     */
    double
    totalCharge(const dealii::DoFHandler<3> &    dofHandlerOfField,
                const distributedCPUVec<double> &rhoNodalField,
                std::map<dealii::CellId, std::vector<double>> &rhoQuadValues);


    double
    totalCharge(const dealii::DoFHandler<3> &    dofHandlerOfField,
                const distributedCPUVec<double> &rhoNodalField);


    double
    totalCharge(
      const dealii::DoFHandler<3> &                        dofHandlerOfField,
      const std::map<dealii::CellId, std::vector<double>> *rhoQuadValues);


    double
    totalCharge(const dealii::MatrixFree<3, double> &matrixFreeDataObject,
                const distributedCPUVec<double> &    rhoNodalField);


    void
    dipole(const dealii::DoFHandler<3> &dofHandlerOfField,
           const std::map<dealii::CellId, std::vector<double>> *rhoQuadValues,
           bool                                                 centerofCharge);

    double
    rhofieldl2Norm(const dealii::MatrixFree<3, double> &matrixFreeDataObject,
                   const distributedCPUVec<double> &    rhoNodalField,
                   const unsigned int                   dofHandlerId,
                   const unsigned int                   quadratureId);

    double
    rhofieldInnerProduct(
      const dealii::MatrixFree<3, double> &matrixFreeDataObject,
      const distributedCPUVec<double> &    rhoNodalField1,
      const distributedCPUVec<double> &    rhoNodalField2,
      const unsigned int                   dofHandlerId,
      const unsigned int                   quadratureId);


    double
    fieldGradl2Norm(const dealii::MatrixFree<3, double> &matrixFreeDataObject,
                    const distributedCPUVec<double> &    field);

    /**
     *@brief l2 projection
     */
    void
    l2ProjectionQuadToNodal(
      const dealii::MatrixFree<3, double> &                matrixFreeDataObject,
      const dealii::AffineConstraints<double> &            constraintMatrix,
      const unsigned int                                   dofHandlerId,
      const unsigned int                                   quadratureId,
      const std::map<dealii::CellId, std::vector<double>> &quadratureValueData,
      distributedCPUVec<double> &                          nodalField);

    /**
     *@brief l2 projection
     */
    void
    l2ProjectionQuadDensityMinusAtomicDensity(
      const dealii::MatrixFree<3, double> &                matrixFreeDataObject,
      const dealii::AffineConstraints<double> &            constraintMatrix,
      const unsigned int                                   dofHandlerId,
      const unsigned int                                   quadratureId,
      const std::map<dealii::CellId, std::vector<double>> &quadratureValueData,
      distributedCPUVec<double> &                          nodalField);

    /**
     *@brief Computes net magnetization from the difference of local spin densities
     */
    double
    totalMagnetization(
      const std::map<dealii::CellId, std::vector<double>> *rhoQuadValues);

    /**
     *@brief normalize the input electron density
     */
    void
    normalizeRhoInQuadValues();

    /**
     *@brief normalize the output electron density in each scf
     */
    void
    normalizeRhoOutQuadValues();

    /**
     *@brief normalize the electron density
     */
    void
    normalizeAtomicRhoQuadValues();

    /**
     *@brief Computes output electron-density from wavefunctions
     */
    void
    compute_rhoOut(
#ifdef DFTFE_WITH_DEVICE
      kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
        &kohnShamDFTEigenOperator,
#endif
      kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
        &        kohnShamDFTEigenOperatorCPU,
      const bool isConsiderSpectrumSplitting,
      const bool isGroundState = false);


    void
    popOutRhoInRhoOutVals();

    /**
     *@brief Mixing schemes for mixing electron-density
     */
    double
    mixing_simple();

    double
    mixing_simple_spinPolarized();

    double
    nodalDensity_mixing_simple_kerker(
#ifdef DFTFE_WITH_DEVICE
      kerkerSolverProblemDevice<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>
        &                   kerkerPreconditionedResidualSolverProblemDevice,
      linearSolverCGDevice &CGSolverDevice,
#endif
      kerkerSolverProblem<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>
        &                 kerkerPreconditionedResidualSolverProblem,
      dealiiLinearSolver &CGSolver);

    double
    nodalDensity_mixing_anderson_kerker(
#ifdef DFTFE_WITH_DEVICE
      kerkerSolverProblemDevice<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>
        &                   kerkerPreconditionedResidualSolverProblemDevice,
      linearSolverCGDevice &CGSolverDevice,
#endif
      kerkerSolverProblem<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>
        &                 kerkerPreconditionedResidualSolverProblem,
      dealiiLinearSolver &CGSolver);

    double
    lowrankApproxScfDielectricMatrixInv(const unsigned int scfIter);

    double
    lowrankApproxScfDielectricMatrixInvSpinPolarized(
      const unsigned int scfIter);

    /**
     * Re solves the all electrostatics on a h refined mesh, and computes
     * the corresponding energy. This function
     * is called after reaching the ground state electron density. Currently the
     * h refinement is hardcoded to a one subdivison of carser mesh
     * FIXME: The function is not yet extened to the case when point group
     * symmetry is used. However, it works for time reversal symmetry.
     *
     */
    void
    computeElectrostaticEnergyHRefined(
#ifdef DFTFE_WITH_DEVICE
      kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
        &kohnShamDFTEigenOperator
#endif
    );

    /**
     *@brief Computes Fermi-energy obtained by imposing constraint on the number of electrons
     */
    void
    compute_fermienergy(
      const std::vector<std::vector<double>> &eigenValuesInput,
      const double                            numElectronsInput);
    /**
     *@brief Computes Fermi-energy obtained by imposing separate constraints on the number of spin-up and spin-down electrons
     */
    void
    compute_fermienergy_constraintMagnetization(
      const std::vector<std::vector<double>> &eigenValuesInput);

    /**
     *@brief compute density of states and local density of states
     */
    void
    compute_tdos(const std::vector<std::vector<double>> &eigenValuesInput,
                 const unsigned int                      highestStateOfInterest,
                 const std::string &                     fileName);

    void
    compute_ldos(const std::vector<std::vector<double>> &eigenValuesInput,
                 const std::string &                     fileName);

    void
    compute_pdos(const std::vector<std::vector<double>> &eigenValuesInput,
                 const std::string &                     fileName);


    /**
     *@brief compute localization length
     */
    void
    compute_localizationLength(const std::string &locLengthFileName);

    /**
     *@brief write wavefunction solution fields
     */
    void
    outputWfc();

    /**
     *@brief write electron density solution fields
     */
    void
    outputDensity();

    /**
     *@brief write the KS eigen values for given BZ sampling/path
     */
    void
    writeBands();

    /**
     *@brief Computes the volume of the domain
     */
    double
    computeVolume(const dealii::DoFHandler<3> &_dofHandler);

    /**
     *@brief Deforms the domain by the given deformation gradient and reinitializes the
     * dftClass datastructures.
     */
    void
    deformDomain(const dealii::Tensor<2, 3, double> &deformationGradient,
                 const bool vselfPerturbationUpdateForStress = false,
                 const bool useSingleAtomSolutionsOverride   = false,
                 const bool print                            = true);

    /**
     *@brief Computes inner Product and Y = alpha*X + Y for complex vectors used during
     * periodic boundary conditions
     */

#ifdef USE_COMPLEX
    std::complex<double>
    innerProduct(distributedCPUVec<double> &a, distributedCPUVec<double> &b);

    void
    alphaTimesXPlusY(std::complex<double>       alpha,
                     distributedCPUVec<double> &x,
                     distributedCPUVec<double> &y);

#endif
    /**
     *@brief Sets dirichlet boundary conditions for total potential constraints on
     * non-periodic boundary (boundary id==0). Currently setting homogeneous bc
     *
     */
    void
    applyPeriodicBCHigherOrderNodes();



    excManager *         d_excManagerPtr;
    dispersionCorrection d_dispersionCorr;

    /**
     * stores required data for Kohn-Sham problem
     */
    unsigned int numElectrons, numElectronsUp, numElectronsDown, numLevels;
    std::set<unsigned int> atomTypes;

    /// FIXME: eventually it should be a map of atomic number to struct-
    /// {valence number, mesh input etc}
    std::map<unsigned int, unsigned int> d_atomTypeAtributes;

    /// FIXME: remove atom type atributes from atomLocations
    std::vector<std::vector<double>> atomLocations, atomLocationsFractional,
      d_reciprocalLatticeVectors, d_domainBoundingVectors;
    std::vector<std::vector<double>> d_atomLocationsAutoMesh;
    std::vector<std::vector<double>> d_imagePositionsAutoMesh;

    /// Gaussian displacements of atoms read from file
    std::vector<dealii::Tensor<1, 3, double>> d_atomsDisplacementsGaussianRead;

    ///
    std::vector<double> d_netFloatingDispSinceLastBinsUpdate;

    ///
    std::vector<double> d_netFloatingDispSinceLastCheckForSmearedChargeOverlaps;

    bool d_isAtomsGaussianDisplacementsReadFromFile = false;

    /// Gaussian generator parameter for force computation and Gaussian
    /// deformation of atoms and FEM mesh Gaussian generator: Gamma(r)=
    /// exp(-(r/d_gaussianConstant)^2) Stored for all domain atoms
    std::vector<double> d_gaussianConstantsForce;

    /// Gaussian constants for automesh mesh movement stored for all domain
    /// atoms
    std::vector<double> d_gaussianConstantsAutoMesh;

    /// composite generator flat top widths for all domain atoms
    std::vector<double> d_generatorFlatTopWidths;

    /// flat top widths for all domain atoms in case of automesh mesh movement
    /// composite gaussian
    std::vector<double> d_flatTopWidthsAutoMeshMove;

    /// smeared charge widths for all domain atoms
    std::vector<double> d_smearedChargeWidths;

    /// smeared charge normalization scaling for all domain atoms
    std::vector<double> d_smearedChargeScaling;

    /// nearest atom ids for all domain atoms
    std::vector<unsigned int> d_nearestAtomIds;

    /// nearest atom distances for all domain atoms
    std::vector<double> d_nearestAtomDistances;

    ///
    double d_minDist;

    /// vector of lendth number of periodic image charges with corresponding
    /// master chargeIds
    std::vector<int> d_imageIds;
    // std::vector<int> d_imageIdsAutoMesh;


    /// vector of length number of periodic image charges with corresponding
    /// charge values
    std::vector<double> d_imageCharges;

    /// vector of length number of periodic image charges with corresponding
    /// positions in cartesian coordinates
    std::vector<std::vector<double>> d_imagePositions;

    /// globalChargeId to ImageChargeId Map
    std::vector<std::vector<int>> d_globalChargeIdToImageIdMap;

    /// vector of lendth number of periodic image charges with corresponding
    /// master chargeIds , generated with a truncated pspCutoff
    std::vector<int> d_imageIdsTrunc;

    /// vector of length number of periodic image charges with corresponding
    /// charge values , generated with a truncated pspCutoff
    std::vector<double> d_imageChargesTrunc;

    /// vector of length number of periodic image charges with corresponding
    /// positions in cartesian coordinates, generated with a truncated pspCutOff
    std::vector<std::vector<double>> d_imagePositionsTrunc;

    /// globalChargeId to ImageChargeId Map generated with a truncated pspCutOff
    std::vector<std::vector<int>> d_globalChargeIdToImageIdMapTrunc;

    /// distance from the domain till which periodic images will be considered
    double d_pspCutOff = 15.0;

    /// distance from the domain till which periodic images will be considered
    const double d_pspCutOffTrunc = 15.0;

    /// cut-off distance from atom till which non-local projectors are
    /// non-trivial
    double d_nlPSPCutOff = 8.0;

    /// non-intersecting smeared charges of all atoms at quad points
    std::map<dealii::CellId, std::vector<double>> d_bQuadValuesAllAtoms;

    /// non-intersecting smeared charge gradients of all atoms at quad points
    std::map<dealii::CellId, std::vector<double>> d_gradbQuadValuesAllAtoms;

    /// non-intersecting smeared charges atom ids of all atoms at quad points
    std::map<dealii::CellId, std::vector<int>> d_bQuadAtomIdsAllAtoms;

    /// non-intersecting smeared charges atom ids of all atoms (with image atom
    /// ids separately accounted) at quad points
    std::map<dealii::CellId, std::vector<int>> d_bQuadAtomIdsAllAtomsImages;

    /// map of cell and non-trivial global atom ids (no images) for smeared
    /// charges for each bin
    std::map<dealii::CellId, std::vector<unsigned int>>
      d_bCellNonTrivialAtomIds;

    /// map of cell and non-trivial global atom ids (no images) for smeared
    /// charge for each bin
    std::vector<std::map<dealii::CellId, std::vector<unsigned int>>>
      d_bCellNonTrivialAtomIdsBins;

    /// map of cell and non-trivial global atom and image ids for smeared
    /// charges for each bin
    std::map<dealii::CellId, std::vector<unsigned int>>
      d_bCellNonTrivialAtomImageIds;

    /// map of cell and non-trivial global atom and image ids for smeared charge
    /// for each bin
    std::vector<std::map<dealii::CellId, std::vector<unsigned int>>>
      d_bCellNonTrivialAtomImageIdsBins;

    /// minimum smeared charge width
    const double d_smearedChargeWidthMin = 0.4;

    std::vector<orbital> waveFunctionsVector;
    std::map<unsigned int,
             std::map<unsigned int,
                      std::map<unsigned int, alglib::spline1dinterpolant>>>
      radValues;
    std::map<unsigned int,
             std::map<unsigned int, std::map<unsigned int, double>>>
      outerValues;

    /**
     * meshGenerator based object
     */
    triangulationManager d_mesh;

    double       d_autoMeshMaxJacobianRatio;
    unsigned int d_autoMesh;


    /// affine transformation object
    meshMovementAffineTransform d_affineTransformMesh;

    /// meshMovementGaussianClass object
    meshMovementGaussianClass d_gaussianMovePar;

    std::vector<dealii::Tensor<1, 3, double>>
                                  d_gaussianMovementAtomsNetDisplacements;
    std::vector<dealii::Point<3>> d_controlPointLocationsCurrentMove;

    /// volume of the domain
    double d_domainVolume;

    /// init wfc trunctation radius
    double d_wfcInitTruncation = 5.0;

    /**
     * dealii based FE data structres
     */
    dealii::FESystem<3>   FE, FEEigen;
    dealii::DoFHandler<3> dofHandler, dofHandlerEigen, d_dofHandlerPRefined,
      d_dofHandlerRhoNodal;
    unsigned int d_eigenDofHandlerIndex, d_phiExtDofHandlerIndexElectro,
      d_forceDofHandlerIndex;
    unsigned int                  d_densityDofHandlerIndex;
    unsigned int                  d_densityDofHandlerIndexElectro;
    unsigned int                  d_nonPeriodicDensityDofHandlerIndexElectro;
    unsigned int                  d_baseDofHandlerIndexElectro;
    unsigned int                  d_forceDofHandlerIndexElectro;
    unsigned int                  d_smearedChargeQuadratureIdElectro;
    unsigned int                  d_nlpspQuadratureId;
    unsigned int                  d_lpspQuadratureId;
    unsigned int                  d_lpspQuadratureIdElectro;
    unsigned int                  d_gllQuadratureId;
    unsigned int                  d_phiTotDofHandlerIndexElectro;
    unsigned int                  d_phiTotAXQuadratureIdElectro;
    unsigned int                  d_helmholtzDofHandlerIndexElectro;
    unsigned int                  d_binsStartDofHandlerIndexElectro;
    unsigned int                  d_densityQuadratureId;
    unsigned int                  d_densityQuadratureIdElectro;
    dealii::MatrixFree<3, double> matrix_free_data, d_matrixFreeDataPRefined;
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      basisOperationsPtrHost;
#if defined(DFTFE_WITH_DEVICE)
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::DEVICE>>
      basisOperationsPtrDevice;
#endif

    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      d_BLASWrapperPtrHost;


    std::shared_ptr<
#if defined(DFTFE_WITH_DEVICE)
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
#else
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
#endif
      d_BLASWrapperPtr;

    std::map<dealii::types::global_dof_index, dealii::Point<3>> d_supportPoints,
      d_supportPointsPRefined, d_supportPointsEigen;
    std::vector<const dealii::AffineConstraints<double> *> d_constraintsVector;
    std::vector<const dealii::AffineConstraints<double> *>
      d_constraintsVectorElectro;

    /**
     * parallel objects
     */
    const MPI_Comm mpi_communicator;
#if defined(DFTFE_WITH_DEVICE)
    utils::DeviceCCLWrapper *d_devicecclMpiCommDomainPtr;
#endif
    const MPI_Comm     d_mpiCommParent;
    const MPI_Comm     interpoolcomm;
    const MPI_Comm     interBandGroupComm;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    dealii::IndexSet   locally_owned_dofs, locally_owned_dofsEigen;
    dealii::IndexSet   locally_relevant_dofs, locally_relevant_dofsEigen,
      d_locallyRelevantDofsPRefined, d_locallyRelevantDofsRhoNodal;
    std::vector<dealii::types::global_dof_index> local_dof_indicesReal,
      local_dof_indicesImag;
    std::vector<dealii::types::global_dof_index> localProc_dof_indicesReal,
      localProc_dof_indicesImag;
    std::vector<bool> selectedDofsHanging;

    forceClass<FEOrder, FEOrderElectro> *   forcePtr;
    symmetryClass<FEOrder, FEOrderElectro> *symmetryPtr;

    elpaScalaManager *d_elpaScala;

    poissonSolverProblem<FEOrder, FEOrderElectro> d_phiTotalSolverProblem;
#ifdef DFTFE_WITH_DEVICE
    poissonSolverProblemDevice<FEOrder, FEOrderElectro>
      d_phiTotalSolverProblemDevice;
#endif

    bool d_kohnShamDFTOperatorsInitialized;

    kohnShamDFTOperatorClass<FEOrder, FEOrderElectro> *d_kohnShamDFTOperatorPtr;
#ifdef DFTFE_WITH_DEVICE
    kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
      *d_kohnShamDFTOperatorDevicePtr;
#endif

    const std::string d_dftfeScratchFolderName;

    /**
     * chebyshev subspace iteration solver objects
     *
     */
    chebyshevOrthogonalizedSubspaceIterationSolver d_subspaceIterationSolver;
#ifdef DFTFE_WITH_DEVICE
    chebyshevOrthogonalizedSubspaceIterationSolverDevice
      d_subspaceIterationSolverDevice;
#endif

    /**
     * constraint Matrices
     */

    /**
     *object which is used to store dealii constraint matrix information
     *using STL vectors. The relevant dealii constraint matrix
     *has hanging node constraints and periodic constraints(for periodic
     *problems) used in eigen solve
     */
    dftUtils::constraintMatrixInfo constraintsNoneEigenDataInfo;


    /**
     *object which is used to store dealii constraint matrix information
     *using STL vectors. The relevant dealii constraint matrix
     *has hanging node constraints used in Poisson problem solution
     *
     */
    dftUtils::constraintMatrixInfo constraintsNoneDataInfo;


#ifdef DFTFE_WITH_DEVICE
    dftUtils::constraintMatrixInfoDevice d_constraintsNoneDataInfoDevice;
#endif


    dealii::AffineConstraints<double> constraintsNone, constraintsNoneEigen,
      d_noConstraints;

    dealii::AffineConstraints<double> d_constraintsForTotalPotentialElectro;

    dealii::AffineConstraints<double> d_constraintsForHelmholtzRhoNodal;

    dealii::AffineConstraints<double> d_constraintsPRefined;

    dealii::AffineConstraints<double> d_constraintsPRefinedOnlyHanging;

    dealii::AffineConstraints<double> d_constraintsRhoNodal;

    dealii::AffineConstraints<double> d_constraintsRhoNodalOnlyHanging;

    dftUtils::constraintMatrixInfo d_constraintsRhoNodalInfo;

    /**
     * data storage for Kohn-Sham wavefunctions
     */
    std::vector<std::vector<double>> eigenValues;

    std::vector<std::vector<double>> d_densityMatDerFermiEnergy;

    /// Spectrum split higher eigenvalues computed in Rayleigh-Ritz step
    std::vector<std::vector<double>> eigenValuesRRSplit;

    /**
     * The indexing of d_eigenVectorsFlattenedHost and
     * d_eigenVectorsFlattenedDevice [kPoint * numSpinComponents *
     * numLocallyOwnedNodes * numWaveFunctions + iSpin * numLocallyOwnedNodes *
     * numWaveFunctions + iNode * numWaveFunctions + iWaveFunction]
     */
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST>
      d_eigenVectorsFlattenedHost;

    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST>
      d_eigenVectorsRotFracDensityFlattenedHost;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::HOST>
      d_eigenVectorsDensityMatrixPrimeHost;

    /// device eigenvectors
#ifdef DFTFE_WITH_DEVICE
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_eigenVectorsFlattenedDevice;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_eigenVectorsRotFracFlattenedDevice;
    dftfe::utils::MemoryStorage<dataTypes::number,
                                dftfe::utils::MemorySpace::DEVICE>
      d_eigenVectorsDensityMatrixPrimeFlattenedDevice;
#endif

    /// parallel message stream
    dealii::ConditionalOStream pcout;

    /// compute-time logger
    dealii::TimerOutput computing_timer;
    dealii::TimerOutput computingTimerStandard;

    /// A plain global timer to track only the total elapsed time after every
    /// ground-state solve
    dealii::Timer d_globalTimer;

    // dft related objects
    std::shared_ptr<std::map<dealii::CellId, std::vector<double>>> rhoInValues,
      rhoOutValues, rhoInValuesSpinPolarized, rhoOutValuesSpinPolarized;

    std::map<dealii::CellId, std::vector<double>> d_phiInValues, d_phiOutValues;

    MixingScheme d_mixingScheme;

    distributedCPUVec<double> d_rhoInNodalValuesRead, d_rhoInNodalValues,
      d_rhoOutNodalValues, d_rhoOutNodalValuesSplit, d_preCondResidualVector,
      d_rhoNodalFieldRefined, d_rhoOutNodalValuesDistributed;
    std::deque<distributedCPUVec<double>> d_rhoInNodalVals, d_rhoOutNodalVals;

    distributedCPUVec<double> d_rhoInSpin0NodalValues;
    distributedCPUVec<double> d_rhoInSpin1NodalValues;

    distributedCPUVec<double> d_rhoInSpin0NodalValuesRead;
    distributedCPUVec<double> d_rhoInSpin1NodalValuesRead;

    distributedCPUVec<double> d_rhoOutSpin0NodalValues,
      d_rhoOutSpin1NodalValues;

    std::deque<distributedCPUVec<double>> d_rhoInSpin0NodalVals,
      d_rhoOutSpin0NodalVals;
    std::deque<distributedCPUVec<double>> d_rhoInSpin1NodalVals,
      d_rhoOutSpin1NodalVals;

    std::map<dealii::CellId, std::vector<double>> d_rhoOutValuesLpspQuad,
      d_rhoInValuesLpspQuad, d_gradRhoOutValuesLpspQuad,
      d_gradRhoInValuesLpspQuad;

    /// for low rank jacobian inverse approximation
    std::deque<distributedCPUVec<double>> d_vcontainerVals;
    std::deque<distributedCPUVec<double>> d_fvcontainerVals;
    std::deque<distributedCPUVec<double>> d_vSpin0containerVals;
    std::deque<distributedCPUVec<double>> d_fvSpin0containerVals;
    std::deque<distributedCPUVec<double>> d_vSpin1containerVals;
    std::deque<distributedCPUVec<double>> d_fvSpin1containerVals;
    distributedCPUVec<double>             d_residualPredicted;
    unsigned int                          d_rankCurrentLRD;
    double                                d_relativeErrorJacInvApproxPrevScfLRD;
    double                                d_residualNormPredicted;
    bool                                  d_tolReached;

    /// for xl-bomd
    std::map<dealii::CellId, std::vector<double>> d_rhoAtomsValues,
      d_gradRhoAtomsValues, d_hessianRhoAtomsValues;
    std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      d_rhoAtomsValuesSeparate, d_gradRhoAtomsValuesSeparate,
      d_hessianRhoAtomsValuesSeparate;

    std::shared_ptr<std::map<dealii::CellId, std::vector<double>>>
      gradRhoInValues, gradRhoInValuesSpinPolarized;
    std::shared_ptr<std::map<dealii::CellId, std::vector<double>>>
      gradRhoOutValues, gradRhoOutValuesSpinPolarized;


    // storage for total electrostatic potential solution vector corresponding
    // to input scf electron density
    distributedCPUVec<double> d_phiTotRhoIn;

    // storage for total electrostatic potential solution vector corresponding
    // to output scf electron density
    distributedCPUVec<double> d_phiTotRhoOut;

    // storage for sum of nuclear electrostatic potential
    distributedCPUVec<double> d_phiExt;

    // storage for projection of rho cell quadrature data to nodal field
    distributedCPUVec<double> d_rhoNodalField;

    // storage for projection of rho cell quadrature data to nodal field
    distributedCPUVec<double> d_rhoNodalFieldSpin0;

    // storage for projection of rho cell quadrature data to nodal field
    distributedCPUVec<double> d_rhoNodalFieldSpin1;

    // storage of densities for xl-bomd
    std::deque<distributedCPUVec<double>> d_groundStateDensityHistory;

    std::map<dealii::CellId, std::vector<double>> d_pseudoVLoc;

    /// Internal data:: map for cell id to Vpseudo local of individual atoms.
    /// Only for atoms whose psp tail intersects the local domain.
    std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      d_pseudoVLocAtoms;


    std::vector<std::vector<double>> d_localVselfs;

    // nonlocal pseudopotential related objects used only for pseudopotential
    // calculation
    std::map<dealii::CellId, std::vector<double>> d_rhoCore;

    std::map<dealii::CellId, std::vector<double>> d_gradRhoCore;

    std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      d_gradRhoCoreAtoms;

    std::map<dealii::CellId, std::vector<double>> d_hessianRhoCore;

    std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      d_hessianRhoCoreAtoms;

    //
    // Store the map between the "pseudo" wave function Id and the function Id
    // details (i.e., global splineId, l quantum number, m quantum number)
    //
    std::vector<std::vector<int>> d_pseudoWaveFunctionIdToFunctionIdDetails;

    //
    // Store the map between the "pseudo" potential Id and the function Id
    // details (i.e., global splineId, l quantum number)
    //
    std::vector<std::vector<int>> d_deltaVlIdToFunctionIdDetails;

    //
    // vector to store the number of pseudowave functions/pseudo potentials
    // associated with an atom (global nonlocal psp atom id)
    //
    std::vector<int> d_numberPseudoAtomicWaveFunctions;
    std::vector<int> d_numberPseudoPotentials;
    std::vector<int> d_nonLocalAtomGlobalChargeIds;

    //
    // matrices denoting the sparsity of nonlocal projectors and elemental
    // projector matrices
    //
    std::map<unsigned int, std::vector<int>> d_sparsityPattern;
    std::vector<std::vector<dealii::DoFHandler<3>::active_cell_iterator>>
                                           d_elementIteratorsInAtomCompactSupport;
    std::vector<std::vector<unsigned int>> d_elementIdsInAtomCompactSupport;
    std::vector<std::vector<dealii::DoFHandler<3>::active_cell_iterator>>
                                  d_elementOneFieldIteratorsInAtomCompactSupport;
    std::vector<std::vector<int>> d_nonLocalAtomIdsInElement;
    std::vector<unsigned int>     d_nonLocalAtomIdsInCurrentProcess;
    dealii::IndexSet              d_locallyOwnedProjectorIdsCurrentProcess;
    dealii::IndexSet              d_ghostProjectorIdsCurrentProcess;
    std::map<std::pair<unsigned int, unsigned int>, unsigned int>
      d_projectorIdsNumberingMapCurrentProcess;
#ifdef USE_COMPLEX
    std::vector<std::vector<std::vector<std::complex<double>>>>
      d_nonLocalProjectorElementMatricesConjugate,
      d_nonLocalProjectorElementMatricesTranspose;


    std::vector<distributedCPUVec<std::complex<double>>>
      d_projectorKetTimesVectorPar;

    /// parallel vector used in nonLocalHamiltionian times wavefunction vector
    /// computation pre-initialization of the parallel layout is more efficient
    /// than creating the parallel layout for every nonLocalHamiltionan times
    /// wavefunction computation
    distributedCPUMultiVec<std::complex<double>>
      d_projectorKetTimesVectorParFlattened;
#else
    std::vector<std::vector<std::vector<double>>>
      d_nonLocalProjectorElementMatricesConjugate,
      d_nonLocalProjectorElementMatricesTranspose;


    std::vector<distributedCPUVec<double>> d_projectorKetTimesVectorPar;

    /// parallel vector used in nonLocalHamiltionian times wavefunction vector
    /// computation pre-initialization of the parallel layout is more efficient
    /// than creating the parallel layout for every nonLocalHamiltionan times
    /// wavefunction computation
    distributedCPUMultiVec<double> d_projectorKetTimesVectorParFlattened;
#endif

    //
    // storage for nonlocal pseudopotential constants
    //
    std::vector<std::vector<double>> d_nonLocalPseudoPotentialConstants;

    //
    // spline vector for data corresponding to each spline of pseudo
    // wavefunctions
    //
    std::vector<alglib::spline1dinterpolant> d_pseudoWaveFunctionSplines;

    //
    // spline vector for data corresponding to each spline of delta Vl
    //
    std::vector<alglib::spline1dinterpolant> d_deltaVlSplines;

    /* Flattened Storage for precomputed nonlocal pseudopotential quadrature
     * data. This is to speedup the configurational force computation. Data
     * format: vector(numNonLocalAtomsCurrentProcess with non-zero compact
     * support, vector(number pseudo wave
     * functions,map<cellid,num_quad_points*2>)). Refer to
     * (https://link.aps.org/doi/10.1103/PhysRevB.97.165132) for details of the
     * expression of the configurational force terms for the norm-conserving
     * Troullier-Martins pseudopotential in the Kleinman-Bylander form. The same
     * expressions also extend to the Optimized Norm-Conserving Vanderbilt
     * (ONCV) pseudopotentials.
     */
    std::vector<dataTypes::number> d_nonLocalPSP_ZetalmDeltaVl;


    /* Flattened Storage for precomputed nonlocal pseudopotential quadrature
     * data. This is to speedup the configurational stress computation. Data
     * format: vector(numNonLocalAtomsCurrentProcess with non-zero compact
     * support, vector(number pseudo wave
     * functions,map<cellid,num_quad_points*num_k_points*3*2>)). Refer to
     * (https://link.aps.org/doi/10.1103/PhysRevB.97.165132) for details of the
     * expression of the configurational force terms for the norm-conserving
     * Troullier-Martins pseudopotential in the Kleinman-Bylander form. The same
     * expressions also extend to the Optimized Norm-Conserving Vanderbilt
     * (ONCV) pseudopotentials.
     */
    std::vector<dataTypes::number>
      d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms;


    /// map from cell number to set of non local atom ids (local numbering)
    std::map<unsigned int, std::vector<unsigned int>>
      d_cellIdToNonlocalAtomIdsLocalCompactSupportMap;

    /// vector of size num physical cells
    std::vector<unsigned int> d_nonTrivialPseudoWfcsPerCellZetaDeltaVQuads;

    /// vector of size num physical cell with starting index for each cell for
    /// the above array
    std::vector<unsigned int>
      d_nonTrivialPseudoWfcsCellStartIndexZetaDeltaVQuads;

    std::vector<unsigned int> d_nonTrivialAllCellsPseudoWfcIdToElemIdMap;

    /// map from local nonlocal atomid to vector over cells
    std::map<unsigned int, std::vector<unsigned int>>
      d_atomIdToNonTrivialPseudoWfcsCellStartIndexZetaDeltaVQuads;

    unsigned int d_sumNonTrivialPseudoWfcsOverAllCellsZetaDeltaVQuads;


    std::vector<unsigned int> d_projecterKetTimesFlattenedVectorLocalIds;
    std::vector<std::vector<unsigned int>> d_projectorKetTimesVectorLocalIds;
    //
    // vector of outermost Points for various radial Data
    //
    std::vector<double> d_outerMostPointPseudoWaveFunctionsData;
    std::vector<double> d_outerMostPointPseudoPotData;
    std::vector<double> d_outerMostPointPseudoProjectorData;

    /// map of atom node number and atomic weight
    std::map<dealii::types::global_dof_index, double> d_atomNodeIdToChargeMap;

    /// vselfBinsManager object
    vselfBinsManager<FEOrder, FEOrderElectro> d_vselfBinsManager;

    /// Gateaux derivative of vself field with respect to affine strain tensor
    /// components using central finite difference. This is used for cell stress
    /// computation
    std::vector<distributedCPUVec<double>> d_vselfFieldGateauxDerStrainFDBins;

    /// Compute Gateaux derivative of vself field in bins with respect to affine
    /// strain tensor components
    void
    computeVselfFieldGateauxDerFD(
#ifdef DFTFE_WITH_DEVICE
      kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
        &kohnShamDFTEigenOperatorDevice
#endif
    );

    /// dftParameters object
    dftParameters *d_dftParamsPtr;

    /// kPoint cartesian coordinates
    std::vector<double> d_kPointCoordinates;

    /// k point crystal coordinates
    std::vector<double> kPointReducedCoordinates;

    /// k point weights
    std::vector<double> d_kPointWeights;

    /// closest tria vertex
    std::vector<dealii::Point<3>> d_closestTriaVertexToAtomsLocation;
    std::vector<dealii::Tensor<1, 3, double>> d_dispClosestTriaVerticesToAtoms;

    /// global k index of lower bound of the local k point set
    unsigned int lowerBoundKindex = 0;
    /**
     * Recomputes the k point cartesian coordinates from the crystal k point
     * coordinates and the current lattice vectors, which can change in each
     * ground state solve dutring cell optimization.
     */
    void
    recomputeKPointCoordinates();

    /// fermi energy
    double fermiEnergy, fermiEnergyUp, fermiEnergyDown, d_groundStateEnergy;

    double d_freeEnergyInitial;

    double d_freeEnergy;

    /// entropic energy
    double d_entropicEnergy;

    // chebyshev filter variables and functions
    // int numPass ; // number of filter passes

    std::vector<double> a0;
    std::vector<double> bLow;

    /// stores flag for first ever call to chebyshev filtering for a given FEM
    /// mesh vector for each k point and spin
    std::vector<bool> d_isFirstFilteringCall;

    std::vector<double> d_upperBoundUnwantedSpectrumValues;

    distributedCPUVec<double> d_tempEigenVec;

    bool d_isRestartGroundStateCalcFromChk;

    /**
     * @ nscf variables
     */
    bool scfConverged;
    void
    nscf(
      kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
        &                                             kohnShamDFTEigenOperator,
      chebyshevOrthogonalizedSubspaceIterationSolver &subspaceIterationSolver);
    void
    initnscf(
      kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
        &                                            kohnShamDFTEigenOperator,
      poissonSolverProblem<FEOrder, FEOrderElectro> &phiTotalSolverProblem,
      dealiiLinearSolver &                           CGSolver);

    /**
     * @brief compute the maximum of the residual norm of the highest occupied state among all k points
     */
    double
    computeMaximumHighestOccupiedStateResidualNorm(
      const std::vector<std::vector<double>>
        &residualNormWaveFunctionsAllkPoints,
      const std::vector<std::vector<double>> &eigenValuesAllkPoints,
      const double                            _fermiEnergy);


    /**
     * @brief compute the maximum of the residual norm of the highest state of interest among all k points
     */
    double
    computeMaximumHighestOccupiedStateResidualNorm(
      const std::vector<std::vector<double>>
        &residualNormWaveFunctionsAllkPoints,
      const std::vector<std::vector<double>> &eigenValuesAllkPoints,
      const unsigned int                      highestState);


    void
    kohnShamEigenSpaceCompute(
      const unsigned int s,
      const unsigned int kPointIndex,
      kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
        &                                             kohnShamDFTEigenOperator,
      elpaScalaManager &                              elpaScala,
      chebyshevOrthogonalizedSubspaceIterationSolver &subspaceIterationSolver,
      std::vector<double> &                           residualNormWaveFunctions,
      const bool                                      computeResidual,
      const bool                                      isSpectrumSplit = false,
      const bool                                      useMixedPrec    = false,
      const bool                                      isFirstScf      = false);


#ifdef DFTFE_WITH_DEVICE
    void
    kohnShamEigenSpaceCompute(
      const unsigned int s,
      const unsigned int kPointIndex,
      kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
        &               kohnShamDFTEigenOperator,
      elpaScalaManager &elpaScala,
      chebyshevOrthogonalizedSubspaceIterationSolverDevice
        &                  subspaceIterationSolverDevice,
      std::vector<double> &residualNormWaveFunctions,
      const bool           computeResidual,
      const unsigned int   numberRayleighRitzAvoidancePasses = 0,
      const bool           isSpectrumSplit                   = false,
      const bool           useMixedPrec                      = false,
      const bool           isFirstScf                        = false);
#endif


#ifdef DFTFE_WITH_DEVICE
    void
    kohnShamEigenSpaceFirstOrderDensityMatResponse(
      const unsigned int s,
      const unsigned int kPointIndex,
      kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
        &               kohnShamDFTEigenOperator,
      elpaScalaManager &elpaScala,
      chebyshevOrthogonalizedSubspaceIterationSolverDevice
        &subspaceIterationSolverDevice);

#endif

    void
    kohnShamEigenSpaceFirstOrderDensityMatResponse(
      const unsigned int s,
      const unsigned int kPointIndex,
      kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
        &               kohnShamDFTEigenOperator,
      elpaScalaManager &elpaScala);

    void
    kohnShamEigenSpaceComputeNSCF(
      const unsigned int spinType,
      const unsigned int kPointIndex,
      kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
        &                                             kohnShamDFTEigenOperator,
      chebyshevOrthogonalizedSubspaceIterationSolver &subspaceIterationSolver,
      std::vector<double> &                           residualNormWaveFunctions,
      unsigned int                                    ipass);
  };

} // namespace dftfe

#endif
