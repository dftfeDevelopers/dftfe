//Include header files
#include "../../include/headers.h"
#include "../../include/dft.h"
#include "../../utils/fileReaders.cc"
#include "../poisson/poisson.cc"
#include "../eigen/eigen.cc"
#include "mesh.cc"
#include "init.cc"
#include "energy.cc"
#include "charge.cc"
#include "density.cc"
#include "locatenodes.cc"
#include "mixingschemes.cc"
 
//dft constructor
dftClass::dftClass():
  triangulation (MPI_COMM_WORLD),
  FE (QGaussLobatto<1>(FEOrder+1)),
  dofHandler (triangulation),
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times),
  poisson(this),
  eigen(this),
  bLow(0.0),
  a0(lowerEndWantedSpectrum)
{
  //set size of eigenvalues and eigenvectors data structures
  eigenValues.resize(numEigenValues);
  for (unsigned int i=0; i<numEigenValues; ++i){
    vectorType* temp=new vectorType;
    eigenVectors.push_back(temp);
  } 
}

//dft run
void dftClass::run ()
{
  pcout << "number of MPI processes: "
	<< Utilities::MPI::n_mpi_processes(mpi_communicator)
	<< std::endl;
  //generate/read mesh
  mesh();
  //initialize
  init();
  poisson.solve();
  //
  eigen.computeLocalHamiltonians(rhoInValues, poisson.phiTotRhoOut);
  eigen.HX(eigen.HXvalue, eigenVectors);
  eigen.XHX(eigenVectors);

  /*
  //solve
  computing_timer.enter_section("dft solve"); 
  //phiExt with nuclear charge
  poisson.solve(phiExt, residual, jacobian, constraints1byR, atoms);
  
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
    poisson.solve(phiTotRhoIn, residual, jacobian, constraintsZero, atoms, rhoInValues);
    //eigen solve
    eigenObject.solve(phiTotRhoIn, massMatrix, hamiltonianMatrix, massVector, constraintsNone, rhoInValues, eigenValues, eigenVectors, scfIter);
    //fermi energy
    compute_fermienergy();
    //rhoOut
    compute_rhoOut();
    //phiTot with rhoOut
    poisson.solve(phiTotRhoOut, residual, jacobian, constraintsZero, atoms, rhoOutValues);
    //energy
    compute_energy();
    pcout<<"SCF iteration: " << scfIter+1 << " complete\n";
    scfIter++;

    //fill eigen spectrum bounds for chebyshev filter (bLow, a0)
    a0=eigenValues[0]; bLow=*(--eigenValues.end());
  }
  computing_timer.exit_section("dft solve"); 
  //
  */
}

