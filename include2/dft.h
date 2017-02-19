#ifndef dft_H_
#define dft_H_
#include <iostream>
#include <iomanip> 
#include <numeric>
#include <sstream>
#include <complex>
#include "headers.h"
#include "poisson.h"
#include "eigen.h"
#include "/nfs/mcfs_comp/home/rudraa/software/alglib/cpp/src/interpolation.h"
#include "/nfs/mcfs_comp/home/rudraa/software/libxc/libxc-2.2.0/installDir/include/xc.h"
#ifdef ENABLE_PERIODIC_BC
#include "/home/vikramg/DFT-FE-softwares/softwareCentos/petsc/intel_petsc3.7.5_complex/include/petsc.h"
#include "/home/vikramg/DFT-FE-softwares/softwareCentos/slepc/intel_slepc3.7.3_complex/include/slepceps.h"
#else
#include "/home/vikramg/DFT-FE-softwares/softwareCentos/petsc/intel_petsc3.7.5_double_elemental/include/petsc.h"
#include "/home/vikramg/DFT-FE-softwares/softwareCentos/slepc/intel_slepc3.7.3_double_elemental/include/slepceps.h"
#endif

//std::cout << std::setprecision(18) << std::scientific;

//Initialize Namespace
using namespace dealii;
//blas-lapack routines
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

xc_func_type funcX, funcC;

//
const double kb = 3.166811429e-06;

//
struct orbital{
  unsigned int atomID;
  unsigned int Z, n, l;
  int m;
  alglib::spline1dinterpolant* psi;
};

//Define dft class
class dftClass{
  friend class poissonClass;
  friend class eigenClass;  
 public:
  dftClass();
  void run();
  std::map<unsigned int, unsigned int> additionalWaveFunctions;
  std::map<unsigned int, unsigned int> numberAtomicWaveFunctions;
  unsigned int numEigenValues;
 private:
  void set();
  unsigned int numElectrons, numBaseLevels, numLevels;
  std::set<unsigned int> atomTypes;
  std::vector<std::vector<double> > atomLocations,d_latticeVectors,d_imagePositions;
  std::vector<int> d_imageIds;
  std::vector<double> d_imageCharges;
  std::vector<orbital> waveFunctionsVector;
  std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, alglib::spline1dinterpolant*> > > radValues;
  std::map<unsigned int, std::map<unsigned int, std::map <unsigned int, double> > >outerValues;

  void mesh();
  void init();
  void solveVself();
  void initRho();
  void initLocalPseudoPotential();
  void initNonLocalPseudoPotential();
  void computeSparseStructureNonLocalProjectors();
  void computeElementalProjectorKets();
  double totalCharge(std::map<dealii::CellId, std::vector<double> > *);
  void locateAtomCoreNodes();
  void locatePeriodicPinnedNodes();
  void createAtomBins(std::vector<const ConstraintMatrix * > & constraintsVector);
  double mixing_simple();
  double mixing_anderson();
  void compute_energy();
  void compute_fermienergy();
  double repulsiveEnergy();
  void compute_rhoOut();
  void readPSIRadialValues();
  void readPSI();
  void determineOrbitalFilling();
  void generateImageCharges();
  void readkPointData();
  void loadPSIFiles(unsigned int Z, unsigned int n, unsigned int l, unsigned int & flag);

#ifdef ENABLE_PERIODIC_BC  
  std::complex<double> innerProduct(vectorType & a,
				    vectorType & b);

  void alphaTimesXPlusY(std::complex<double>   alpha,
			vectorType           & x,
			vectorType           & y);
#endif

  
  

  //FE data structres
  parallel::distributed::Triangulation<3> triangulation;
  FESystem<3>        FE, FEEigen;
  DoFHandler<3>      dofHandler, dofHandlerEigen;
  unsigned int       eigenDofHandlerIndex;
  MatrixFree<3,double> matrix_free_data;
  std::map<types::global_dof_index, Point<3> > d_supportPoints, d_supportPointsEigen;
  std::vector< const ConstraintMatrix * > d_constraintsVector; 
  
  //parallel objects
  MPI_Comm   mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  IndexSet   locally_owned_dofs, locally_owned_dofsEigen;
  IndexSet   locally_relevant_dofs, locally_relevant_dofsEigen;
  std::vector<unsigned int> local_dof_indicesReal, local_dof_indicesImag;
  poissonClass poisson;
  eigenClass eigen;
  ConstraintMatrix constraintsNone, constraintsNoneEigen, d_constraintsForTotalPotential, d_constraintsPeriodicWithDirichlet; 
  std::vector<std::vector<double> > eigenValues;
  std::vector<std::vector<parallel::distributed::Vector<double>*> > eigenVectors;
  std::vector<std::vector<parallel::distributed::Vector<double>*> > eigenVectorsOrig;
  //unsigned int numEigenValues;

  //parallel message stream
  ConditionalOStream  pcout;  
  
  //compute-time logger
  TimerOutput computing_timer;
  
  //dft related objects
  std::map<dealii::CellId, std::vector<double> > *rhoInValues, *rhoOutValues;
  std::vector<std::map<dealii::CellId,std::vector<double> >*> rhoInVals, rhoOutVals;

#ifdef xc_id
  #if xc_id == 4
  std::map<dealii::CellId, std::vector<double> > *gradRhoXInValues,  *gradRhoYInValues,  *gradRhoZInValues;
  std::map<dealii::CellId, std::vector<double> > *gradRhoXOutValues, *gradRhoYOutValues, *gradRhoZOutValues;
  std::vector<std::map<dealii::CellId,std::vector<double> >*> gradRhoXInVals, gradRhoYInVals, gradRhoZInVals; 
  std::vector<std::map<dealii::CellId,std::vector<double> >*> gradRhoXOutVals, gradRhoYOutVals, gradRhoZOutVals;
  #endif
#endif


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

  //
  //kPointCoordinates
  //
  std::vector<double> d_kPointCoordinates;
  std::vector<double> d_kPointWeights;
  int d_maxkPoints;
  int d_kPointIndex;

  //fermi energy
  double fermiEnergy;

  //chebyshev filter variables and functions
  double bUp;// bLow, a0;
  std::vector<double> a0;
  std::vector<double> bLow;
  vectorType vChebyshev, v0Chebyshev, fChebyshev, aj[5];
  std::vector<parallel::distributed::Vector<double>*> PSI, tempPSI, tempPSI2, tempPSI3, tempPSI4;
  void chebyshevSolver();
  double upperBound();
  void gramSchmidt(std::vector<vectorType*>& X);
  void chebyshevFilter(std::vector<vectorType*>& X, unsigned int m, double a, double b, double a0);  
  void rayleighRitz(std::vector<vectorType*>& X);
};

#endif
