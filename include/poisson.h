#ifndef poisson_H_
#define poisson_H_
#include "headers.h"
#include "dft.h"

//Initialize Namespace
using namespace dealii;

//Define poisson class
template <int dim>
class poisson
{
  friend class dft; 
public:
  poisson(DoFHandler<dim>* _dofHandler);
  void solve(PETScWrappers::MPI::Vector& solution, 
	     PETScWrappers::MPI::Vector& residual,
	     PETScWrappers::MPI::SparseMatrix& jacobian,
	     ConstraintMatrix& constraints,
	     std::vector<unsigned int>& originIDs,
	     Table<2,double>* rhoValues
	     );
private:
  void init ();
  void assemble(PETScWrappers::MPI::Vector& solution, 
		PETScWrappers::MPI::Vector& residual,
		PETScWrappers::MPI::SparseMatrix& jacobian,
		ConstraintMatrix& constraints,
		std::vector<unsigned int>& originIDs,
		Table<2,double>* rhoValues
		);

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
