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
// @author Shiva Rudraraju (2016), Phani Motamarri (2016)
//

#ifndef dft_H_
#define dft_H_
#include <iostream>
#include <iomanip> 
#include <numeric>
#include <sstream>
#include <complex>
#include <deque>



#include "headers.h"
#include "constants.h"

#include "poisson.h"
#include "eigen.h"
#include "symmetry.h"
#include "meshMovementAffineTransform.h"

#include <interpolation.h> 
#include <xc.h>
#include <petsc.h>
#include <slepceps.h>

#include "dftParameters.h"
#include "constraintMatrixInfo.h"
#include "meshGenerator.h"
#include <spglib.h>

//
//Initialize Namespace
//
using namespace dealii;

typedef dealii::parallel::distributed::Vector<double> vectorType;
//forward declarations
template <unsigned int T> class poissonClass;
template <unsigned int T> class eigenClass; 
template <unsigned int T> class forceClass;  
template <unsigned int T> class symmetryClass;
template <unsigned int T> class forceClass; 
template <unsigned int T> class geoOptIon;
template <unsigned int T> class geoOptCell; 

//
//extern declarations for blas-lapack routines
//
extern "C"{
  void dgemv_(char* TRANS, const int* M, const int* N, double* alpha, double* A, const int* LDA, double* X, const int* INCX, double* beta, double* C, const int* INCY);
  void dgesv_( int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, int* info );
  void dscal_(int *n, double *alpha, double *x, int *incx);
  void daxpy_(int *n, double *alpha, double *x, int *incx, double *y, int *incy);
  void dgemm_(const char* transA, const char* transB, const int *m, const int *n, const int *k, const double *alpha, const double *A, const int *lda, const double *B, const int *ldb, const double *beta, double *C, const int *ldc);
  void dsyevd_(char* jobz, char* uplo, int* n, double* A, int *lda, double* w, double* work, int* lwork, int* iwork, int* liwork, int* info);
  void zgemm_(const char* transA, const char* transB, const int *m, const int *n, const int *k, const std::complex<double> *alpha, const std::complex<double> *A, const int *lda, const std::complex<double> *B, const int *ldb, const std::complex<double> *beta, std::complex<double> *C, const int *ldc);
  void zheevd_(char *jobz, char *uplo,int *n,std::complex<double> *A,int *lda,double *w,std::complex<double> *work,int *lwork,double *rwork,int *lrwork,int *iwork,int *liwork,int *info);
  void zdotc_(std::complex<double> *C,int *N,const std::complex<double> *X,int *INCX,const std::complex<double> *Y,int *INCY);
  void zaxpy_(int *n,std::complex<double> *alpha,std::complex<double> *x,int *incx,std::complex<double> *y,int *incy);
}

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
  friend class poissonClass;

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
  dftClass(MPI_Comm &mpi_comm_replica, MPI_Comm &interpoolcomm);
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
  void init();   
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

 private:

  /**
   * generate image charges and update k point cartesian coordinates based
   * on current lattice vectors
   */
  void initImageChargesUpdateKPoints();

  void generateMPGrid();
  void writeMesh(std::string meshFileName);
  void generateImageCharges();
  void determineOrbitalFilling();

 
  /**
   * moves the triangulation vertices using Gaussians such that the all atoms are on triangulation vertices
   */
  void moveMeshToAtoms(Triangulation<3,3> & triangulationMove,bool reuse=false);  

  /**
   * Initializes the guess of electron-density and single-atom wavefunctions on the mesh,
   * maps finite-element nodes to given atomic positions,
   * initializes pseudopotential files and exchange-correlation functionals to be used
   * based on user-choice. 
   * In periodic problems, periodic faces are mapped here. Further finite-element nodes
   * to be pinned for solving the Poisson problem electro-static potential is set here
   */
  void initUnmovedTriangulation(parallel::distributed::Triangulation<3> & triangulation);
  void initBoundaryConditions();
  void initElectronicFields();
  void initPseudoPotentialAll();
  void locateAtomCoreNodes();
  void locatePeriodicPinnedNodes();
  void initRho();
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
   * Sets dirichlet boundary conditions for total potential constraints on 
   * non-periodic boundary (boundary id==0). Currently setting homogeneous bc
   *
   */
  void applyTotalPotentialDirichletBC();

  void computeElementalOVProjectorKets();


  /**
   * Categorizes atoms into bins based on self-potential ball radius around each atom such 
   * that no two atoms in each bin has overlapping balls
   * and finally solves the self-potentials in each bin one-by-one.
   */
  void createAtomBins(std::vector<ConstraintMatrix * > & constraintsVector);
  void createAtomBinsExtraSanityCheck();
  void solveVself();
  
  /**
   * Computes total charge by integrating the electron-density
   */
  double totalCharge(std::map<dealii::CellId, std::vector<double> > *);

  /**
   * normalized the charge density
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
   * Computes ground-state energy in a given SCF iteration,
   * computes repulsive energy explicity for a non-periodic system
   */
  double compute_energy(const bool print);
  double compute_energy_spinPolarized(const bool print);
  double repulsiveEnergy();

  /**
   * Computes Fermi-energy obtained by imposing constraint on the number of electrons
   */
  void compute_fermienergy();

  void output();


  /**
   * Computes the volume of the domain
   */
  void computeVolume();

  /**
   * Deforms the domain by the given deformation gradient and reinitializes the
   * dftClass datastructures.
   */
  void deformDomain(const Tensor<2,3,double> & deformationGradient);  

  /**
   * Computes inner Product and Y = alpha*X + Y for complex vectors used during
   * periodic boundary conditions
   */
#ifdef ENABLE_PERIODIC_BC
  std::complex<double> innerProduct(vectorType & a,
				    vectorType & b);

  void alphaTimesXPlusY(std::complex<double>   alpha,
			vectorType           & x,
			vectorType           & y);

    
  /**
   * Sets dirichlet boundary conditions for total potential constraints on 
   * non-periodic boundary (boundary id==0). Currently setting homogeneous bc
   *
   */
  void applyPeriodicBCHigherOrderNodes();
#endif

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
  std::vector<Point<3>> closestTriaVertexToAtomsLocation;

  std::vector<Tensor<1,3,double> > distanceClosestTriaVerticesToAtoms;
  std::vector<Tensor<1,3,double> > dispClosestTriaVerticesToAtoms;


  /**
   * meshGenerator based object
   */
  meshGeneratorClass d_mesh;

  /// affine transformation object
  meshMovementAffineTransform d_affineTransformMesh;

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
  std::vector< ConstraintMatrix * > d_constraintsVector; 
  
  /**
   * parallel objects
   */
  MPI_Comm   mpi_communicator, interpoolcomm;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  IndexSet   locally_owned_dofs, locally_owned_dofsEigen;
  IndexSet   locally_relevant_dofs, locally_relevant_dofsEigen;
  std::vector<unsigned int> local_dof_indicesReal, local_dof_indicesImag;
  std::vector<unsigned int> localProc_dof_indicesReal,localProc_dof_indicesImag;
  std::vector<bool> selectedDofsHanging;

  poissonClass<FEOrder> * poissonPtr;
  eigenClass<FEOrder> * eigenPtr;
  forceClass<FEOrder> * forcePtr;
  symmetryClass<FEOrder> * symmetryPtr;
  geoOptIon<FEOrder> * geoOptIonPtr;
  geoOptCell<FEOrder> * geoOptCellPtr;  

  /**
   * constraint Matrices
   */
  ConstraintMatrix constraintsNone, constraintsNoneEigen, d_constraintsForTotalPotential, d_constraintsPeriodicWithDirichlet, d_noConstraints, d_noConstraintsEigen;
 
  /// vector of constraint matrices for vself bins
  std::vector<ConstraintMatrix> d_vselfBinConstraintMatrices;

  /**
   * data storage for Kohn-Sham wavefunctions
   */
  std::vector<std::vector<double> > eigenValues, eigenValuesTemp; 
  std::vector<std::vector<vectorType> > eigenVectors;

  /**
   * storage for constraintMatrices in terms of arrays (STL)
   */
 dftUtils::constraintMatrixInfo constraintsNoneEigenDataInfo; 



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
  std::vector<std::vector<DoFHandler<3>::active_cell_iterator> > d_elementOneFieldIteratorsInAtomCompactSupport;
  std::vector<std::vector<int> > d_nonLocalAtomIdsInElement;
  std::vector<unsigned int> d_nonLocalAtomIdsInCurrentProcess;
  IndexSet d_locallyOwnedProjectorIdsCurrentProcess;
  IndexSet d_ghostProjectorIdsCurrentProcess;
  std::map<std::pair<unsigned int,unsigned int>, unsigned int> d_projectorIdsNumberingMapCurrentProcess;
#ifdef ENABLE_PERIODIC_BC
  std::vector<std::vector<std::vector<std::vector<std::complex<double> > > > > d_nonLocalProjectorElementMatrices;
  std::vector<dealii::parallel::distributed::Vector<std::complex<double> > > d_projectorKetTimesVectorPar;  
#else
  std::vector<std::vector<std::vector<std::vector<double> > > > d_nonLocalProjectorElementMatrices;
  std::vector<dealii::parallel::distributed::Vector<double> > d_projectorKetTimesVectorPar;  
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

  //map of atom node number and atomic weight
  std::map<unsigned int, double> atoms;
  std::vector<std::map<unsigned int, double> > d_atomsInBin;
  
  //map of binIds and atomIds in it and other bin related information
  std::map<int,std::set<int> > d_bins;
  std::vector<std::vector<int> > d_imageIdsInBins;
  std::vector<std::map<dealii::types::global_dof_index, int> > d_boundaryFlag;
  std::vector<std::map<dealii::types::global_dof_index, double> > d_vselfBinField;
  std::vector<std::map<dealii::types::global_dof_index, int> > d_closestAtomBin;
  std::vector<vectorType> d_vselfFieldBins;//required for configurational force
  
  /// kPoint cartesian coordinates 
  std::vector<double> d_kPointCoordinates;

  /// k point crystal coordinates
  std::vector<double> kPointReducedCoordinates;

  /// k point weights
  std::vector<double> d_kPointWeights;

  /// total number of k points
  int d_maxkPoints;

  /// current k point index during the ground state solve
  int d_kPointIndex;
  
  /** Recomputes the k point cartesian coordinates from the crystal k point coordinates
   * and the current lattice vectors, which can change in each ground state solve when
   * isCellOpt is true
   */
  void recomputeKPointCoordinates();

  /// integralRhoOut to store number of electrons
  double integralRhoValue;
  
  /// fermi energy
  double fermiEnergy;

  //chebyshev filter variables and functions
  //int numPass ; // number of filter passes
  double bUp;
  std::vector<double> a0;
  std::vector<double> bLow;
  vectorType vChebyshev, v0Chebyshev, fChebyshev;

  
  void chebyshevSolver(const unsigned int s);
  void computeResidualNorm(std::vector<vectorType>& X);
  std::vector<std::vector<double> > d_tempResidualNormWaveFunctions;
  double computeMaximumHighestOccupiedStateResidualNorm();

  double upperBound();

  void gramSchmidt(std::vector<vectorType>& X);

  void chebyshevFilter(std::vector<vectorType>& X, 
		       const unsigned int m, 
		       const double a, 
		       const double b, 
		       const double a0);  

  void rayleighRitz(const unsigned int spinType, 
		    std::vector<vectorType>& X);

  void scale(const vectorType & diagonal,
	     const unsigned int spinType);

};

#endif
