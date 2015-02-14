#ifndef eigen_H_
#define eigen_H_
#include "headers.h"
#include "dft.h"

//Initialize Namespace
using namespace dealii;

//Define eigen class
template <int dim>
class eigen
{
  friend class dft;
public:
  eigen(DoFHandler<dim>* _dofHandler);
  void solve(PETScWrappers::MPI::Vector& solution, 
	     PETScWrappers::MPI::SparseMatrix& massMatrix,
	     PETScWrappers::MPI::SparseMatrix& hamiltonianMatrix,
	     PETScWrappers::MPI::Vector& massVector, 
	     ConstraintMatrix& constraints,
	     Table<2,double>* rhoValues);
 private:
  void init ();
  void assemble(PETScWrappers::MPI::Vector& solution, 
		PETScWrappers::MPI::SparseMatrix& massMatrix,
		PETScWrappers::MPI::SparseMatrix& hamiltonianMatrix,
		PETScWrappers::MPI::Vector& massVector, 
		ConstraintMatrix& constraints,
		Table<2,double>* rhoValues);

  //FE data structres
  FE_Q<dim>          FE;
  DoFHandler<dim>*    dofHandler;

  //parallel objects
  MPI_Comm mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  ConditionalOStream   pcout;
  IndexSet   locally_owned_dofs;
  IndexSet   locally_relevant_dofs;

  //compute-time logger
  TimerOutput computing_timer;
};

#endif
