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
#include "poisson.h"
#include "eigen.h"

#include <interpolation.h> 
#include <xc.h>
#include <petsc.h>
#include <slepceps.h>

//
//Initialize Namespace
//
using namespace dealii;

//
//extern declarations for blas-lapack routines
//
extern "C"{
  void dgemv_(char* TRANS, const int* M, const int* N, double* alpha, double* A, const int* LDA, double* X, const int* INCX, double* beta, double* C, const int* INCY);
  void dgesv_( int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, int* info );
  void dscal_(int *n, double *alpha, double *x, int *incx);
  void daxpy_(int *n, double *alpha, double *x, int *incx, double *y, int *incy);
  void dgemm_(char* transA, char* transB, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);
  void dsyevd_(char* jobz, char* uplo, int* n, double* A, int *lda, double* w, double* work, int* lwork, int* iwork, int* liwork, int* info);
  void zgemm_(char* transA, char* transB, int *m, int *n, int *k, std::complex<double> *alpha, std::complex<double> *A, int *lda, std::complex<double> *B, int *ldb, std::complex<double> *beta, std::complex<double> *C, int *ldc);
  void zheevd_(char *jobz, char *uplo,int *n,std::complex<double> *A,int *lda,double *w,std::complex<double> *work,int *lwork,double *rwork,int *lrwork,int *iwork,int *liwork,int *info);
  void zdotc_(std::complex<double> *C,int *N,const std::complex<double> *X,int *INCX,const std::complex<double> *Y,int *INCY);
  void zaxpy_(int *n,std::complex<double> *alpha,std::complex<double> *x,int *incx,std::complex<double> *y,int *incy);
}

//
//objects for various exchange-correlations (from libxc package)
//
xc_func_type funcX, funcC;

//
//Boltzmann constant
//
const double kb = 3.166811429e-06;

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

 public:

  /**
   * dftClass constructor
   */
  dftClass();

  /**
   * Sets up Kohn-Sham SCF iteration after the required pre-processing steps
   */
  void run();

  /**
   * Number of Kohn-Sham eigen values to be computed
   */
  unsigned int numEigenValues;

 private:

  /**
   * Reads the coordinates of the atoms.
   * If periodic calculation, reads fractional coordinates of atoms in the unit-cell,
   * lattice vectors, kPoint quadrature rules to be used and also generates image atoms.
   * Also determines orbital-ordering
   */
  void set();
  void readkPointData();
  void generateImageCharges();
  void determineOrbitalFilling();

  /**
   * Initializes the finite-element mesh
   */
  void mesh();

  /**
   * Initializes the guess of electron-density and single-atom wavefunctions on the mesh,
   * maps finite-element nodes to given atomic positions,
   * initializes pseudopotential files and exchange-correlation functionals to be used
   * based on user-choice. 
   * In periodic problems, periodic faces are mapped here. Further finite-element nodes
   * to be pinned for solving the Poisson problem electro-static potential is set here
   */
  void init();
  void locateAtomCoreNodes();
  void locatePeriodicPinnedNodes();
  void initRho();
  void readPSI();
  void readPSIRadialValues();
  void loadPSIFiles(unsigned int Z, unsigned int n, unsigned int l, unsigned int & flag);
  void initLocalPseudoPotential();
  void initNonLocalPseudoPotential();
  void computeSparseStructureNonLocalProjectors();
  void computeElementalProjectorKets();

  /**
   * Categorizes atoms into bins based on self-potential ball radius around each atom such 
   * that no two atoms in each bin has overlapping balls
   * and finally solves the self-potentials in each bin one-by-one.
   */
  void createAtomBins(std::vector<const ConstraintMatrix * > & constraintsVector);
  void solveVself();
  
  /**
   * Computes total charge by integrating the electron-density
   */
  double totalCharge(std::map<dealii::CellId, std::vector<double> > *);
  
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
  void compute_energy();
  void compute_energy_spinPolarized();
  double repulsiveEnergy();

  /**
   * Computes Fermi-energy obtained by imposing constraint on the number of electrons
   */
  void compute_fermienergy();

  void output();

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
#endif
  
  /**
   * stores required data for Kohn-Sham problem
   */
  unsigned int numElectrons, numLevels;
  std::set<unsigned int> atomTypes;
  std::vector<std::vector<double> > atomLocations,d_latticeVectors,d_imagePositions;
  std::vector<int> d_imageIds;
  std::vector<double> d_imageCharges;
  std::vector<orbital> waveFunctionsVector;
  std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, alglib::spline1dinterpolant*> > > radValues;
  std::map<unsigned int, std::map<unsigned int, std::map <unsigned int, double> > >outerValues;
  
  /**
   * dealii based FE data structres
   */
  parallel::distributed::Triangulation<3> triangulation;
  FESystem<3>        FE, FEEigen;
  DoFHandler<3>      dofHandler, dofHandlerEigen;
  unsigned int       eigenDofHandlerIndex,phiExtDofHandlerIndex;
  MatrixFree<3,double> matrix_free_data;
  std::map<types::global_dof_index, Point<3> > d_supportPoints, d_supportPointsEigen;
  std::vector< const ConstraintMatrix * > d_constraintsVector; 
  
  /**
   * parallel objects
   */
  MPI_Comm   mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  IndexSet   locally_owned_dofs, locally_owned_dofsEigen;
  IndexSet   locally_relevant_dofs, locally_relevant_dofsEigen;
  std::vector<unsigned int> local_dof_indicesReal, local_dof_indicesImag;
  std::vector<unsigned int> localProc_dof_indicesReal,localProc_dof_indicesImag;


  poissonClass<FEOrder> poisson;
  eigenClass<FEOrder> eigen;
  ConstraintMatrix constraintsNone, constraintsNoneEigen, d_constraintsForTotalPotential, d_constraintsPeriodicWithDirichlet, d_noConstraints; 
  std::vector<std::vector<double> > eigenValues, eigenValuesTemp; 
  std::vector<std::vector<parallel::distributed::Vector<double>*> > eigenVectors;
  std::vector<std::vector<parallel::distributed::Vector<double>*> > eigenVectorsOrig;


  //parallel message stream
  ConditionalOStream  pcout;  
  
  //compute-time logger
  TimerOutput computing_timer;
  
  //dft related objects
  std::map<dealii::CellId, std::vector<double> > *rhoInValues, *rhoOutValues, *rhoInValuesSpinPolarized, *rhoOutValuesSpinPolarized;
  std::deque<std::map<dealii::CellId,std::vector<double> >*> rhoInVals, rhoOutVals, rhoInValsSpinPolarized, rhoOutValsSpinPolarized;


  std::map<dealii::CellId, std::vector<double> > *gradRhoInValues, *gradRhoInValuesSpinPolarized;
  std::map<dealii::CellId, std::vector<double> > *gradRhoOutValues, *gradRhoOutValuesSpinPolarized;
  std::deque<std::map<dealii::CellId,std::vector<double> >*> gradRhoInVals,gradRhoInValsSpinPolarized,gradRhoOutVals, gradRhoOutValsSpinPolarized; 



  std::map<dealii::CellId, std::vector<double> > *pseudoValues;
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
#ifdef ENABLE_PERIODIC_BC
  std::vector<std::vector<std::vector<std::vector<std::complex<double> > > > > d_nonLocalProjectorElementMatrices;
#else
  std::vector<std::vector<std::vector<std::vector<double> > > > d_nonLocalProjectorElementMatrices;
#endif

  //
  //storage for nonlocal pseudopotential constants
  //
  std::vector<std::vector<double> > d_nonLocalPseudoPotentialConstants;

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
  std::vector<std::map<dealii::types::global_dof_index, int> > d_vselfBinField;

  //
  //kPointCoordinates
  //
  std::vector<double> d_kPointCoordinates;
  std::vector<double> d_kPointWeights;
  int d_maxkPoints;
  int d_kPointIndex;
  
  //integralRhoOut to store number of electrons
  double integralRhoValue;
  
  //fermi energy
  double fermiEnergy;

  //chebyshev filter variables and functions
  //int numPass ; // number of filter passes
  double bUp;// bLow, a0;
  std::vector<double> a0;
  std::vector<double> bLow;
  vectorType vChebyshev, v0Chebyshev, fChebyshev, aj[5];
  std::vector<parallel::distributed::Vector<double>*> PSI, tempPSI, tempPSI2, tempPSI3, tempPSI4;
  void chebyshevSolver(unsigned int s);
  void computeResidualNorm(std::vector<vectorType*>& X);
  double upperBound();
  void gramSchmidt(std::vector<vectorType*>& X);
  void chebyshevFilter(std::vector<vectorType*>& X, unsigned int m, double a, double b, double a0);  
  void rayleighRitz(unsigned int s, std::vector<vectorType*>& X);
};

#endif
