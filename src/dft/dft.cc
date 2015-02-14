//Include header files
#include "../../include/headers.h"
#include "../../include/dft.h"
#include "../../utils/fileReaders.cc"
#include "../poisson/poisson.cc"
#include "../eigen/eigen.cc"
#include "mesh.cc"
#include "boundary.cc"
#include "init.cc"
#include "energy.cc"
#include "charge.cc"
#include "density.cc"
#include "locatenodes.cc"
#include "mixingschemes.cc"
 
//dft constructor
dft::dft():
  triangulation (MPI_COMM_WORLD,
		 typename Triangulation<3>::MeshSmoothing
		 (Triangulation<3>::smoothing_on_refinement |
		  Triangulation<3>::smoothing_on_coarsening)),
  FE (QGaussLobatto<1>(FEOrder+1)),
  dofHandler (triangulation),
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times),
  poissonObject(&dofHandler),
  eigenObject(&dofHandler),
  denSpline(numAtomTypes)
{}

//dft run
void dft::run ()
{
  computing_timer.enter_section("total time"); 
  pcout << "number of MPI processes: "
	<< Utilities::MPI::n_mpi_processes(mpi_communicator)
	<< std::endl;

  //generate/read mesh
  mesh();
  
  //initialize
  computing_timer.enter_section("dft setup"); 
  init();
  initRho();
  locateAtomCoreNodes();
  computing_timer.exit_section("dft setup"); 
  
  //solve
  computing_timer.enter_section("dft solve"); 
  poissonObject.solve(phiTotRhoIn, residual, jacobian, constraintsZero, rhoInValues);
  eigenObject.solve(phiTotRhoIn, massMatrix, hamiltonianMatrix, massVector, constraintsNone, rhoInValues, eigenValues, eigenVectors);
  //mixing_simple();
  computing_timer.exit_section("dft solve"); 

  computing_timer.exit_section("total time"); 
}

