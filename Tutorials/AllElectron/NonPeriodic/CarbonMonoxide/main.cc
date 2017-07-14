//
//Define parameters
//
const double radiusAtomBall = 3.0;

//
//test case for carbon-monoxide
//
const unsigned int FEOrder=1;
const unsigned int n_refinement_steps=0;
const double lowerEndWantedSpectrum=-18.0;
const unsigned int chebyshevOrder=20; 
const unsigned int numSCFIterations=50;
const bool isPseudopotential = false;

//solver parameters 
const unsigned int maxLinearSolverIterations=5000;
const double relLinearSolverTolerance=1.0e-14; 
const double selfConsistentSolverTolerance   = 1.0e-11;

//Define constants
const double TVal=500.0;

//Mesh information
#define meshFileName "../../../data/meshes/allElectron/carbonMonoxide/mesh.inp"
#define coordinatesFile "../../../data/meshes/allElectron/carbonMonoxide/coordinates.inp" 
#define kPointDataFile "../../../data/kPointList/GammaPoint.inp"

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
    //problem.additionalWaveFunctions[6]=5;
    //problem.additionalWaveFunctions[8]=1;
    problem.numberAtomicWaveFunctions[6] = 5;
    problem.numberAtomicWaveFunctions[8] = 5;
    problem.numEigenValues = 10;
    problem.run();
  }
  return 0;
}

