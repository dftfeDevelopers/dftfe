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
  eigenObject(&dofHandler)
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
  //phiExt with nuclear charge
  poissonObject.solve(phiExt, residual, jacobian, constraints1byR, atoms);
  
  //Begin SCF iteration
  unsigned int scfIter=0;
  double norm=1.0;
  while ((norm>1.0e-13) && (scfIter<11)){
    if(this_mpi_process==0) printf("\n\nBegin SCF Iteration:%u\n", scfIter+1);
    //Mixing scheme
    if (scfIter>0){
      if (scfIter==1) norm=mixing_simple();
      else norm=mixing_anderson();
      if(this_mpi_process==0) printf("Mixing Scheme: iter:%u, norm:%12.6e\n", scfIter+1, norm);
    }
    //phiTot with rhoIn
    poissonObject.solve(phiTotRhoIn, residual, jacobian, constraintsZero, atoms, rhoInValues);
    //eigen solve
    eigenObject.solve(phiTotRhoIn, massMatrix, hamiltonianMatrix, massVector, constraintsNone, rhoInValues, eigenValues, eigenVectors, scfIter);
    //fermi energy
    compute_fermienergy();
    //rhoOut
    compute_rhoOut();
    //phiTot with rhoOut
    poissonObject.solve(phiTotRhoOut, residual, jacobian, constraintsZero, atoms, rhoOutValues);
    //energy
    compute_energy();
    pcout<<"SCF iteration: " << scfIter+1 << " complete\n";
    scfIter++;
  }
  computing_timer.exit_section("dft solve"); 
  //
  computing_timer.exit_section("total time"); 
}

