#ifndef poisson_H_
#define poisson_H_
#include "headers.h"

//Initialize Namespace
using namespace dealii;

//Define poisson class
template <int dim>
class poisson
{
public:
  poisson(DoFHandler<dim>* _dofHandler);
  void solve();
private:
  void init ();
  void assemble();
  PETScWrappers::MPI::SparseMatrix jacobian;
  PETScWrappers::MPI::Vector       solution, residual;

  //FE data structres
  FE_Q<dim>          FE;
  ConstraintMatrix   constraints;
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
