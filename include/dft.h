#ifndef dft_H_
#define dft_H_
#include "headers.h"

//Initialize Namespace
using namespace dealii;

//Define dft class
class dft{
  //  friend class poissonProblem;
  //  friend class eigenProblem;
 public:
  dft();
  void run();
  
 private:
  void mesh();
  void init();
  
  //FE data structres
  parallel::distributed::Triangulation<3> triangulation;
  FE_Q<3>            FE;
  DoFHandler<3>      dofHandler;

  //parallel objects
  MPI_Comm   mpi_communicator;
  const unsigned int n_mpi_processes;
  const unsigned int this_mpi_process;
  IndexSet   locally_owned_dofs;
  IndexSet   locally_relevant_dofs;

  //parallel message stream
  ConditionalOStream  pcout;  
  
  //compute-time logger
  TimerOutput computing_timer;
};

#endif
