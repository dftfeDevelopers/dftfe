#ifndef dft_H_
#define dft_H_
#include "headers.h"
#include "poisson.h"
#include "eigen.h"
//include alglib
//#include "/nfs/mcfs_home/rudraa/Public/alglib/cpp/src/interpolation.h"
//#include "/nfs/mcfs_home/rudraa/Public/libxc/libxc-2.2.0/installDir/include/xc.h"
#include "/opt/software/numerics/alglib/cpp/src/interpolation.h"
#include "/opt/software/numerics/libxc-2.2.2/installDir/include/xc.h"

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

//Define dft class
class dftClass{
  friend class poissonClass;
  friend class eigenClass;  
 public:
  dftClass();
  void run();
  Table<2,double> atomLocations;
  std::map<unsigned int, std::string> initialGuessFiles;

 private:
  void mesh();
  void init();
  void initRho();
  double totalCharge();
  void locateAtomCoreNodes();
  double mixing_simple();
  double mixing_anderson();
  void compute_energy();
  void compute_fermienergy();
  double repulsiveEnergy();
  void compute_rhoOut();
  
  //FE data structres
  parallel::distributed::Triangulation<3> triangulation;
  FE_Q<3>            FE;
  DoFHandler<3>      dofHandler;
  MatrixFree<3,double> matrix_free_data;
  
  //parallel objects
  MPI_Comm   mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  IndexSet   locally_owned_dofs;
  IndexSet   locally_relevant_dofs;

  poissonClass poisson;
  eigenClass eigen;
  ConstraintMatrix constraintsNone;
  std::vector<double> eigenValues;
  std::vector<parallel::distributed::Vector<double>*> eigenVectors;

  //parallel message stream
  ConditionalOStream  pcout;  
  
  //compute-time logger
  TimerOutput computing_timer;
  
  //dft related objects
  std::map<dealii::CellId, std::vector<double> > *rhoInValues, *rhoOutValues;
  std::vector<std::map<dealii::CellId,std::vector<double> >*> rhoInVals, rhoOutVals;
  //map of atom node number and atomic weight
  std::map<unsigned int, double> atoms; 
  //fermi energy
  double fermiEnergy;
  //spectrum bounds for chebyshev filter
  double bUp, bLow, a0;
};

#endif
