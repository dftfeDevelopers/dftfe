//Define parameters
//const double radius=20.0;
//testing for one C atom
const unsigned int FEOrder=4;
const unsigned int n_refinement_steps=4;
const unsigned int numElectrons=6;
const double lowerEndWantedSpectrum=-10.0;
const unsigned int chebyshevOrder=2000; 
const unsigned int numSCFIterations=2;

//solver paramteters 
const unsigned int maxLinearSolverIterations=5000;
const double relLinearSolverTolerance=1.0e-12; 

//Define constants
const double TVal=500.0;
const double kb = 3.166811429e-06;

//Mesh information
#define meshFileName "../../../data/meshes/carbon1Atom/mesh.inp"
#define coordinatesFile "../../../data/meshes/carbon1Atom/coordinates.inp" 

//dft header
#include "../../../src2/dft/dft.cc"


int main (int argc, char *argv[]){
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
  deallog.depth_console(0);
  {
    dftClass problem;
    problem.additionalWaveFunctions[6]=2;
    problem.run();
  }
  return 0;
}

