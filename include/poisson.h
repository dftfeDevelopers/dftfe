#ifndef poisson_H_
#define poisson_H_
#include "headers.h"

typedef double dataType;
typedef dealii::parallel::distributed::Vector<double> vectorType;

class dft;

//Define poisson class
template <int dim>
class poisson
{
  friend class dft; 
public:
  poisson(dft* _dftPtr);
  void computeLocalJacobians();
private:
  void init ();
  
  //pointer to dft class
  dft* dftPtr;

  //FE data structres
  dealii::FE_Q<dim>   FE;
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
