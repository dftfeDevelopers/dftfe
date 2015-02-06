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
  void solve();
private:
  void init ();
  void assemble();
  PETScWrappers::MPI::SparseMatrix jacobianPhiTot, jacobianPhiExt;
  PETScWrappers::MPI::Vector       phiTotRhoIn, phiTotRhoOut, phiExtRhoOut;
  PETScWrappers::MPI::Vector       rhsPhiTot, rhsPhiExt;

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
