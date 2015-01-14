//Include header files
#include "../include/headers.h"
#include "../include/dft.h"
#include "poisson.cc"
//#include "eigen.cc"

//dft constructor
dft::dft():
  triangulation (MPI_COMM_WORLD,
		 typename Triangulation<dim>::MeshSmoothing
		 (Triangulation<dim>::smoothing_on_refinement |
		  Triangulation<dim>::smoothing_on_coarsening)),
  FE (QGaussLobatto<1>(FEOrder+1)),
  dofHandler (triangulation),
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times)
{}

void dft::init(){
  pcout << "number of MPI processes: "
	<< Utilities::MPI::n_mpi_processes(mpi_communicator)
	<< std::endl;

  //initialize FE objects
  dofHandler.distribute_dofs (FE);
  locally_owned_dofs = dofHandler.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dofHandler, locally_relevant_dofs);
  pcout << "number of elements: "
	<< triangulation.n_global_active_cells()
	<< std::endl
	<< "number of degrees of freedom: " 
	<< dofHandler.n_dofs() 
	<< std::endl;
}

//Generate triangulation.
void dft::mesh()
{
  //GridGenerator::hyper_cube (triangulation, -1, 1);
  //triangulation.refine_global (6);
  //Read mesh written out in UCD format
  static const Point<3> center = Point<3>();
  static const HyperBallBoundary<3, 3> boundary(center,radius);
  GridIn<3> gridin;
  gridin.attach_triangulation(triangulation);
  //Read mesh in UCD format generated from Cubit
  std::ifstream f("meshFiles/ucd.inp");
  gridin.read_ucd(f);
  triangulation.set_boundary(0, boundary);
  triangulation.refine_global (n_refinement_steps);
  pcout << "Number of active cells: "
	<< triangulation.n_active_cells()
	<< std::endl;
  // Output the total number of cells.
  pcout << "Total number of cells: "
	<< triangulation.n_cells()
	<< std::endl;
}

void dft::run ()
{
  mesh();
  init();
  //initialize poisson, eigen objects
  poisson<3> poissonObject(&dofHandler);
  poissonObject.solve();
  //eigenProblem eigen(this);
  //Initialize mesh, setup, locate origin.
}
