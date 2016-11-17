//
//Define parameters
//
const double radiusAtomBall = 0.5;

//
//test case for carbon-monoxide
//
const unsigned int FEOrder=4;
const unsigned int n_refinement_steps=0;
const double lowerEndWantedSpectrum=-18.0;
const unsigned int chebyshevOrder=50; 
const unsigned int numSCFIterations=20;
const bool isPseudopotential = false;
//solver paramteters 
const unsigned int maxLinearSolverIterations=5000;
const double relLinearSolverTolerance=1.0e-14; 


//Define constants
const double TVal=500.0;

//Mesh information
#define meshFileName "../../../data/meshes/allElectron/carbonMonoxide/meshRefined.inp"
#define coordinatesFile "../../../data/meshes/allElectron/carbonMonoxide/coordinates.inp" 

//dft header
#include "../../../src2/dft/dft.cc"


int main (int argc, char *argv[]){
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
  deallog.depth_console(0);
  {
    // set stdout precision
    //
    std::cout << std::scientific << std::setprecision(18);
    dftClass problem;
    problem.additionalWaveFunctions[6]=2;
    problem.additionalWaveFunctions[8]=1;
    problem.run();
  }
  return 0;
}

