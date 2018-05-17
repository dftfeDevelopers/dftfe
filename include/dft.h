// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das
//

#ifndef dft_H_
#define dft_H_
#include <iostream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <complex>
#include <deque>

#include <headers.h>
#include <constants.h>
#include <constraintMatrixInfo.h>

#include <eigen.h>
#include <symmetry.h>
#include <meshMovementAffineTransform.h>
#include <meshMovementGaussian.h>
#include <eigenSolver.h>
#include <chebyshevOrthogonalizedSubspaceIterationSolver.h>
#include <vselfBinsManager.h>
#include <dftParameters.h>
#include <triangulationManager.h>

#include <interpolation.h>
#include <xc.h>
#include <petsc.h>
#include <slepceps.h>
#include <spglib.h>

namespace dftfe {

  //
  //Initialize Namespace
  //
  using namespace dealii;


  //forward declarations
  template <unsigned int T> class poissonClass;
  template <unsigned int T> class eigenClass;
  template <unsigned int T> class forceClass;
  template <unsigned int T> class symmetryClass;
  template <unsigned int T> class forceClass;
  template <unsigned int T> class geoOptIon;
  template <unsigned int T> class geoOptCell;

  //
  //
  //
  struct orbital
  {
    unsigned int atomID;
    unsigned int Z, n, l;
    int m;
    alglib::spline1dinterpolant* psi;
  };

  //
  //dft class for initializing mesh, setting up guesses for initial electron-density and wavefunctions,
  //solving individual vSelf problem after setting up bins, initializing pseudopotentials. Also
  //has member functions which sets up the process of SCF iteration including mixing of the electron-density
  template <unsigned int FEOrder>
    class dftClass
    {

      template <unsigned int T>
	friend class eigenClass;

      template <unsigned int T>
	friend class forceClass;

      template <unsigned int T>
	friend class geoOptIon;

      template <unsigned int T>
	friend class geoOptCell;

      template <unsigned int T>
	friend class symmetryClass;

    public:

      /**
       * dftClass constructor
       */
      dftClass(const MPI_Comm &mpi_comm_replica, const MPI_Comm &interpoolcomm);
      /**
       * dftClass destructor
       */
      ~dftClass();
      /**
       * Reads the coordinates of the atoms.
       * If periodic calculation, reads fractional coordinates of atoms in the unit-cell,
       * lattice vectors, kPoint quadrature rules to be used and also generates image atoms.
       * Also determines orbital-ordering
       */
      void set();
      /**
       * Does required pre-processing steps including mesh generation calls.
       */
      void init(const bool usePreviousGroundStateRho=false);
      /**
       * Does required pre-processing steps but without remeshing.
       */
      void initNoRemesh();

      /**
       * Selects between only electronic field relaxation or combined electronic and geometry relxation
       */
      void run();

      /**
       *  Kohn-Sham ground solve using SCF iteration
       */
      void solve();

      /**
       * Number of Kohn-Sham eigen values to be computed
       */
      unsigned int numEigenValues;

      void readkPointData();

      /**
       *Get local dofs global indices real
       */
      const std::vector<dealii::types::global_dof_index> & getLocalDofIndicesReal() const;

      /**
       *Get local dofs global indices imag
       */
      const std::vector<dealii::types::global_dof_index> & getLocalDofIndicesImag() const;

      /**
       *Get local dofs local proc indices real
       */
      const std::vector<dealii::types::global_dof_index> & getLocalProcDofIndicesReal() const;

      /**
       *Get local dofs local proc indices imag
       */
      const std::vector<dealii::types::global_dof_index> & getLocalProcDofIndicesImag() const;

      /**
       *Get dealii constraint matrix involving periodic constraints and hanging node constraints in periodic
       *case else only hanging node constraints in non-periodic case
       */
      const ConstraintMatrix & getConstraintMatrixEigen() const;

      /**
       *Get overloaded constraint matrix information involving periodic constraints and hanging node constraints in periodic
       *case else only hanging node constraints in non-periodic case (data stored in STL format)
       */
      const dftUtils::constraintMatrixInfo & getConstraintMatrixEigenDataInfo() const;


      /**
       *Get matrix free data object
       */
      const MatrixFree<3,double> & getMatrixFreeData() const;


      /** @brief Updates atom positions, remeshes/moves mesh and calls appropriate reinits.
       *
       *  Function to update the atom positions and mesh based on the provided displacement input.
       *  Depending on the maximum displacement magnitude this function decides wether to do auto remeshing
       *  or move mesh using Gaussian functions. Additionaly this function also wraps the atom position across the
       *  periodic boundary if the atom moves across it.
       *
       *  @param globalAtomsDisplacements vector containing the displacements (from current position) of all atoms (global).
       *  @return void.
       */
      void updateAtomPositionsAndMoveMesh(const std::vector<Point<3> > & globalAtomsDisplacements);

      /**
       * writes the current domain bounding vectors and atom coordinates to files, which are required for
       * geometry relaxation restart
       */
      void writeDomainAndAtomCoordinates() const;


    private:

      /**
       * generate image charges and update k point cartesian coordinates based
       * on current lattice vectors
       */
      void initImageChargesUpdateKPoints();


      /**
       * project ground state electron density from previous mesh into
       * the new mesh to be used as initial guess for the new ground state solve
       */
      void projectPreviousGroundStateRho();

      /**
       * save triangulation information and rho quadrature data to checkpoint file for restarts
       */
      void saveTriaInfoAndRhoData();

      /**
       * load triangulation information rho quadrature data from checkpoint file for restarted run
       */
      void loadTriaInfoAndRhoData();

      void generateMPGrid();
      void writeMesh(std::string meshFileName);
      void generateImageCharges();
      void determineOrbitalFilling();


      /**
       * moves the triangulation vertices using Gaussians such that the all atoms are on triangulation vertices
       */
      void moveMeshToAtoms(const Triangulation<3,3> & triangulationMove);

      /**
       * Initializes the guess of electron-density and single-atom wavefunctions on the mesh,
       * maps finite-element nodes to given atomic positions,
       * initializes pseudopotential files and exchange-correlation functionals to be used
       * based on user-choice.
       * In periodic problems, periodic faces are mapped here. Further finite-element nodes
       * to be pinned for solving the Poisson problem electro-static potential is set here
       */
      void initUnmovedTriangulation(const parallel::distributed::Triangulation<3> & triangulation);
      void initBoundaryConditions();
      void initElectronicFields(bool usePreviousGroundStateRho=false);
      void initPseudoPotentialAll();

     /**
       * Finds the global dof ids of the nodes containing atoms.
       *
       * @param dofHandler[in]
       * @param atomNodeIdToChargeValueMap[out] local map of global dof id to atom charge id
       */
      void locateAtomCoreNodes(const dealii::DoFHandler<3> & _dofHandler,
	                       std::map<dealii::types::global_dof_index, double> & atomNodeIdToChargeValueMap);

     /**
       * Sets homogeneous dirichlet boundary conditions on a node farthest from
       * all atoms (pinned node). This is only done in case of periodic boundary conditions
       * to get an unique solution to the total electrostatic potential problem.
       *
       * @param dofHandler[in]
       * @param constraintMatrixBase[in] base ConstraintMatrix object
       * @param constraintMatrix[out] ConstraintMatrix object with homogeneous
       * Dirichlet boundary condition entries added
       */
      void locatePeriodicPinnedNodes(const dealii::DoFHandler<3> & _dofHandler,
	                             const dealii::ConstraintMatrix & constraintMatrixBase,
	                             dealii::ConstraintMatrix & constraintMatrix);
      void initRho();
      void computeRhoInitialGuessFromPSI();
      void clearRhoData();
      void noRemeshRhoDataInit();
      void readPSI();
      void readPSIRadialValues();
      void loadPSIFiles(unsigned int Z, unsigned int n, unsigned int l, unsigned int & flag);
      void initLocalPseudoPotential();
      void initNonLocalPseudoPotential();
      void initNonLocalPseudoPotential_OV();
      void computeSparseStructureNonLocalProjectors();
      void computeSparseStructureNonLocalProjectors_OV();
      void computeElementalProjectorKets();


      /**
       * Sets homegeneous dirichlet boundary conditions for total potential constraints on
       * non-periodic boundary (boundary id==0).
       *
       * @param dofHandler[in]
       * @param constraintMatrix[out] ConstraintMatrix object with homogeneous
       * Dirichlet boundary condition entries added
       */
      void applyHomogeneousDirichletBC(const dealii::DoFHandler<3> & _dofHandler,
	                               dealii::ConstraintMatrix & constraintMatrix);

      void computeElementalOVProjectorKets();

      /**
       * Computes total charge by integrating the electron-density
       */
      double totalCharge(const std::map<dealii::CellId, std::vector<double> > *rhoQuadValues);

      /**
       * normalize the electron density
       */
      void normalizeRho();

      /**
       * Computes output electron-density from wavefunctions
       */
      void compute_rhoOut();

      /**
       * Mixing schemes for mixing electron-density
       */
      double mixing_simple();
      double mixing_anderson();
      double mixing_simple_spinPolarized();
      double mixing_anderson_spinPolarized();

      /**
       * Re solves the all electrostatics on a p refined mesh, and computes
       * the corresponding energy. This function
       * is called after reaching the ground state electron density. Currently the p refinement
       * is hardcoded to FEOrder+2.
       * FIXME: The function is not yet extened to the case when point group symmetry is used.
       * However, it works for time reversal symmetry.
       *
       */
      void computeElectrostaticEnergyPRefined();

      /**
       * Computes Fermi-energy obtained by imposing constraint on the number of electrons
       */
      void compute_fermienergy();

      void output();


      /**
       * Computes the volume of the domain
       */
      double computeVolume(const dealii::DoFHandler<3> & _dofHandler);

      /**
       * Deforms the domain by the given deformation gradient and reinitializes the
       * dftClass datastructures.
       */
      void deformDomain(const Tensor<2,3,double> & deformationGradient);

      /**
       * Computes inner Product and Y = alpha*X + Y for complex vectors used during
       * periodic boundary conditions
       */

#ifdef USE_COMPLEX
      std::complex<double> innerProduct(vectorType & a,
					vectorType & b);

      void alphaTimesXPlusY(std::complex<double>   alpha,
			    vectorType           & x,
			    vectorType           & y);

#endif
      /**
       * Sets dirichlet boundary conditions for total potential constraints on
       * non-periodic boundary (boundary id==0). Currently setting homogeneous bc
       *
       */
      void applyPeriodicBCHigherOrderNodes();

      /// objects for various exchange-correlations (from libxc package)
      xc_func_type funcX, funcC;

      /**
       * stores required data for Kohn-Sham problem
       */
      unsigned int numElectrons, numLevels;
      std::set<unsigned int> atomTypes;
      std::vector<std::vector<double> > atomLocations,atomLocationsFractional,d_reciprocalLatticeVectors, d_imagePositions, d_domainBoundingVectors;

      std::vector<int> d_imageIds;
      std::vector<double> d_imageCharges;
      std::vector<orbital> waveFunctionsVector;
      std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, alglib::spline1dinterpolant*> > > radValues;
      std::map<unsigned int, std::map<unsigned int, std::map <unsigned int, double> > >outerValues;

      /**
       * meshGenerator based object
       */
      triangulationManager d_mesh;

      /// affine transformation object
      meshMovementAffineTransform d_affineTransformMesh;

      /// meshMovementGaussianClass object
      meshMovementGaussianClass d_gaussianMovePar;

      /// volume of the domain
      double d_domainVolume;

      /**
       * dealii based FE data structres
       */
      FESystem<3>        FE, FEEigen;
      DoFHandler<3>      dofHandler, dofHandlerEigen;
      unsigned int       eigenDofHandlerIndex,phiExtDofHandlerIndex,phiTotDofHandlerIndex,forceDofHandlerIndex;
      MatrixFree<3,double> matrix_free_data;
      std::map<types::global_dof_index, Point<3> > d_supportPoints, d_supportPointsEigen;
      std::vector<const ConstraintMatrix * > d_constraintsVector;

      /**
       * parallel objects
       */
      MPI_Comm   mpi_communicator, interpoolcomm;
      const unsigned int n_mpi_processes;
      const unsigned int this_mpi_process;
      IndexSet   locally_owned_dofs, locally_owned_dofsEigen;
      IndexSet   locally_relevant_dofs, locally_relevant_dofsEigen;
      std::vector<dealii::types::global_dof_index> local_dof_indicesReal, local_dof_indicesImag;
      std::vector<dealii::types::global_dof_index> localProc_dof_indicesReal,localProc_dof_indicesImag;
      std::vector<bool> selectedDofsHanging;



      forceClass<FEOrder> * forcePtr;
      symmetryClass<FEOrder> * symmetryPtr;
      geoOptIon<FEOrder> * geoOptIonPtr;
      geoOptCell<FEOrder> * geoOptCellPtr;

      /**
       * constraint Matrices
       */

      /**
       *object which is used to store dealii constraint matrix information
       *using STL vectors. The relevant dealii constraint matrix
       *has hanging node constraints and periodic constraints(for periodic problems)
       *used in eigen solve
       */
      dftUtils::constraintMatrixInfo constraintsNoneEigenDataInfo;

      /**
       *object which is used to store dealii constraint matrix information
       *using STL vectors. The relevant dealii constraint matrix
       *has hanging node constraints used in Poisson problem solution
       *
       */
      dftUtils::constraintMatrixInfo constraintsNoneDataInfo;

      ConstraintMatrix constraintsNone, constraintsNoneEigen, d_constraintsForTotalPotential, d_noConstraints, d_noConstraintsEigen;


      /**
       * data storage for Kohn-Sham wavefunctions
       */
      std::vector<std::vector<double> > eigenValues;
      std::vector<std::vector<vectorType> > eigenVectors;

      /// parallel message stream
      ConditionalOStream  pcout;

      /// compute-time logger
      TimerOutput computing_timer;

      //dft related objects
      std::map<dealii::CellId, std::vector<double> > *rhoInValues, *rhoOutValues, *rhoInValuesSpinPolarized, *rhoOutValuesSpinPolarized;
      std::deque<std::map<dealii::CellId,std::vector<double> >> rhoInVals, rhoOutVals, rhoInValsSpinPolarized, rhoOutValsSpinPolarized;


      std::map<dealii::CellId, std::vector<double> > * gradRhoInValues, *gradRhoInValuesSpinPolarized;
      std::map<dealii::CellId, std::vector<double> > * gradRhoOutValues, *gradRhoOutValuesSpinPolarized;
      std::deque<std::map<dealii::CellId,std::vector<double> >> gradRhoInVals,gradRhoInValsSpinPolarized,gradRhoOutVals, gradRhoOutValsSpinPolarized;

      // storage for total electrostatic potential solution vector corresponding to input scf electron density
      vectorType d_phiTotRhoIn;

      // storage for total electrostatic potential solution vector corresponding to output scf electron density
      vectorType d_phiTotRhoOut;

      // storage for sum of nuclear electrostatic potential
      vectorType d_phiExt;

      double d_pspTail = 8.0;
      std::map<dealii::CellId, std::vector<double> > pseudoValues;
      std::vector<std::vector<double> > d_localVselfs;

      //nonlocal pseudopotential related objects used only for pseudopotential calculation

      //
      // Store the map between the "pseudo" wave function Id and the function Id details (i.e., global splineId, l quantum number, m quantum number)
      //
      std::vector<std::vector<int> > d_pseudoWaveFunctionIdToFunctionIdDetails;

      //
      // Store the map between the "pseudo" potential Id and the function Id details (i.e., global splineId, l quantum number)
      //
      std::vector<std::vector<int> > d_deltaVlIdToFunctionIdDetails;

      //
      // vector to store the number of pseudowave functions/pseudo potentials associated with an atom
      //
      std::vector<int> d_numberPseudoAtomicWaveFunctions;
      std::vector<int> d_numberPseudoPotentials;
      std::vector<int> d_nonLocalAtomGlobalChargeIds;

      //
      //matrices denoting the sparsity of nonlocal projectors and elemental projector matrices
      //
      std::vector<std::vector<int> > d_sparsityPattern;
      std::vector<std::vector<DoFHandler<3>::active_cell_iterator> > d_elementIteratorsInAtomCompactSupport;
      std::vector<std::vector<unsigned int> > d_elementIdsInAtomCompactSupport;
      std::vector<std::vector<DoFHandler<3>::active_cell_iterator> > d_elementOneFieldIteratorsInAtomCompactSupport;
      std::vector<std::vector<int> > d_nonLocalAtomIdsInElement;
      std::vector<unsigned int> d_nonLocalAtomIdsInCurrentProcess;
      IndexSet d_locallyOwnedProjectorIdsCurrentProcess;
      IndexSet d_ghostProjectorIdsCurrentProcess;
      std::map<std::pair<unsigned int,unsigned int>, unsigned int> d_projectorIdsNumberingMapCurrentProcess;
#ifdef USE_COMPLEX
      std::vector<std::vector<std::vector<std::vector<std::complex<double> > > > > d_nonLocalProjectorElementMatrices,d_nonLocalProjectorElementMatricesConjugate;
      std::vector<dealii::parallel::distributed::Vector<std::complex<double> > > d_projectorKetTimesVectorPar;

      /// parallel vector used in nonLocalHamiltionian times wavefunction vector computation
      /// pre-initialization of the parallel layout is more efficient than creating the parallel
      /// layout for every nonLocalHamiltionan times wavefunction computation
      dealii::parallel::distributed::Vector<std::complex<double> >  d_projectorKetTimesVectorParFlattened;
#else
      std::vector<std::vector<std::vector<std::vector<double> > > > d_nonLocalProjectorElementMatrices,d_nonLocalProjectorElementMatricesConjugate;
      std::vector<dealii::parallel::distributed::Vector<double> > d_projectorKetTimesVectorPar;

      /// parallel vector used in nonLocalHamiltionian times wavefunction vector computation
      /// pre-initialization of the parallel layout is more efficient than creating the parallel
      /// layout for every nonLocalHamiltionan times wavefunction computation
      dealii::parallel::distributed::Vector<double> d_projectorKetTimesVectorParFlattened;
#endif

      //
      //storage for nonlocal pseudopotential constants
      //
      std::vector<std::vector<double> > d_nonLocalPseudoPotentialConstants;
      std::vector<std::vector<std::vector<double> >> d_nonLocalPseudoPotentialConstants_OV;

      //
      //globalChargeId to ImageChargeId Map
      //
      std::vector<std::vector<int> > d_globalChargeIdToImageIdMap;

      //
      // spline vector for data corresponding to each spline of pseudo wavefunctions
      //
      std::vector<alglib::spline1dinterpolant> d_pseudoWaveFunctionSplines;

      //
      // spline vector for data corresponding to each spline of delta Vl
      //
      std::vector<alglib::spline1dinterpolant> d_deltaVlSplines;

      //
      //vector of outermost Points for various radial Data
      //
      std::vector<double> d_outerMostPointPseudoWaveFunctionsData;
      std::vector<double> d_outerMostPointPseudoPotData;

      /// map of atom node number and atomic weight
      std::map<dealii::types::global_dof_index, double> d_atomNodeIdToChargeMap;

      /// vselfBinsManager object
      vselfBinsManager<FEOrder> d_vselfBinsManager;

      /// kPoint cartesian coordinates
      std::vector<double> d_kPointCoordinates;

      /// k point crystal coordinates
      std::vector<double> kPointReducedCoordinates;

      /// k point weights
      std::vector<double> d_kPointWeights;

      /**
       * Recomputes the k point cartesian coordinates from the crystal k point coordinates
       * and the current lattice vectors, which can change in each ground state solve when
       * isCellOpt is true
       */
      void recomputeKPointCoordinates();

      /// fermi energy
      double fermiEnergy;

      //chebyshev filter variables and functions
      //int numPass ; // number of filter passes

      std::vector<double> a0;
      std::vector<double> bLow;
      vectorType vChebyshev;

      /**
       * @brief compute the maximum of the residual norm of the highest occupied state among all k points
       */
      double computeMaximumHighestOccupiedStateResidualNorm(const std::vector<std::vector<double> > & residualNormWaveFunctionsAllkPoints,
							    const std::vector<std::vector<double> > & eigenValuesAllkPoints,
							    const double _fermiEnergy);


      void kohnShamEigenSpaceCompute(const unsigned int s,
				     const unsigned int kPointIndex,
				     eigenClass<FEOrder> & kohnShamDFTEigenOperator,
				     chebyshevOrthogonalizedSubspaceIterationSolver & subspaceIterationSolver,
				     std::vector<double> & residualNormWaveFunctions);

      void computeResidualNorm(const std::vector<double> & eigenValuesTemp,
			       eigenClass<FEOrder> & kohnShamDFTEigenOperator,
			       std::vector<vectorType> & X,
			       std::vector<double> & residualNorm) const;


    };

}

#endif
