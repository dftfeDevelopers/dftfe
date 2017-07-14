//Define parameters
const unsigned int FEOrder=4;
const unsigned int n_refinement_steps=0;
const double lowerEndWantedSpectrum=-10.0;
const unsigned int chebyshevOrder=100; 
const unsigned int numSCFIterations=25;

//solver paramteters 
const unsigned int maxLinearSolverIterations=5000;
const double relLinearSolverTolerance=1.0e-12; 

//Define constants
const double TVal=500.0;

//Mesh information
#define meshFileName "../../../../data/meshes/AllElectron/Methane/mesh.inp"
#define coordinatesFile "../../../../data/meshes/AllElectron/Methane/coordinates.inp" 

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


