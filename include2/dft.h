#ifndef dft_H_
#define dft_H_
#include <iostream>
#include <iomanip> 
#include <numeric>
#include <sstream> 
#include "headers.h"
#include "poisson.h"
#include "eigen.h"
#include "/nfs/mcfs_comp/home/rudraa/software/alglib/cpp/src/interpolation.h"
#include "/nfs/mcfs_comp/home/rudraa/software/libxc/libxc-2.2.0/installDir/include/xc.h"


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
 private:
  void set();
  unsigned int numElectrons, numBaseLevels, numLevels;
  std::set<unsigned int> atomTypes;
  std::vector<std::vector<double> > atomLocations;
  std::vector<orbital> waveFunctionsVector;
  std::map<unsigned int, std::map<unsigned int, std::map<unsigned int, alglib::spline1dinterpolant*> > > radValues;
  std::map<unsigned int, std::map<unsigned int, std::map <unsigned int, double> > >outerValues;

  void mesh();
  void init();
  void initRho();
  double totalCharge();
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
  void loadPSIFiles(unsigned int Z, unsigned int n, unsigned int l);

  //FE data structres
  parallel::distributed::Triangulation<3> triangulation;
  FE_Q<3>            FE;
  DoFHandler<3>      dofHandler;
  MatrixFree<3,double> matrix_free_data;
  std::map<types::global_dof_index, Point<3> > d_supportPoints; 
  std::vector< const ConstraintMatrix * > d_constraintsVector; 
  
  //parallel objects
  MPI_Comm   mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  IndexSet   locally_owned_dofs;
  IndexSet   locally_relevant_dofs;

  poissonClass poisson;
  eigenClass eigen;
  ConstraintMatrix constraintsNone, d_constraintsForTotalPotential; 
  std::vector<double> eigenValues;
  std::vector<parallel::distributed::Vector<double>*> eigenVectors;
  unsigned int numEigenValues;

  //parallel message stream
  ConditionalOStream  pcout;  
  
  //compute-time logger
  TimerOutput computing_timer;
  
  //dft related objects
  std::map<dealii::CellId, std::vector<double> > *rhoInValues, *rhoOutValues;
  std::vector<std::map<dealii::CellId,std::vector<double> >*> rhoInVals, rhoOutVals;

  //map of atom node number and atomic weight
  std::map<unsigned int, double> atoms;
  std::vector<std::map<unsigned int, double> > d_atomsInBin;
  
  //map of binIds and atomIds in it and other bin related information
  std::map<int,std::set<int> > d_bins;
  std::vector<std::vector<int> > d_imageIdsInBins;
  std::vector<std::map<dealii::types::global_dof_index, int> > d_boundaryFlag;

  //fermi energy
  double fermiEnergy;

  //chebyshev filter variables and functions
  double bUp, bLow, a0;
  vectorType vChebyshev, v0Chebyshev, fChebyshev, aj[5];
  std::vector<parallel::distributed::Vector<double>*> PSI, tempPSI, tempPSI2, tempPSI3;
  void chebyshevSolver();
  double upperBound();
  void gramSchmidt(std::vector<vectorType*>& X);
  void chebyshevFilter(std::vector<vectorType*>& X, unsigned int m, double a, double b, double a0);  
  void rayleighRitz(std::vector<vectorType*>& X);
};

#endif
