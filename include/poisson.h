#ifndef poisson_H_
#define poisson_H_
#include "headers.h"

typedef double dataType;
typedef dealii::parallel::distributed::Vector<double> vectorType;

class dftClass;

//Define poisson class
class poissonClass
{
  friend class dftClass; 
public:
  poissonClass(dftClass* _dftPtr);
  void computeLocalJacobians();
private:
  void init ();
  void computeRHS(const dealii::Table<2,double>* rhoValues);
  void solve(const dealii::Table<2,double>* rhoValues=0);

  //pointer to dft class
  dftClass* dftPtr;

  //FE data structres
  dealii::FE_Q<3>   FE;
  dealii::Table<3,dataType>   localJacobians;

  //constraints
  dealii::ConstraintMatrix  constraintsNone, constraintsZero, constraints1byR;

  //data structures
  vectorType rhs, Ax;
  vectorType jacobianDiagonal;
  vectorType phiTotRhoIn, phiTotRhoOut, phiExt;
 
  //parallel objects
  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  dealii::ConditionalOStream   pcout;

  //compute-time logger
  dealii::TimerOutput computing_timer;
};

#endif
