//Define parameters
#define ENABLE_PERIODIC_BC

//testing for one C atom in periodic domain
const unsigned int FEOrder=1; 
const unsigned int n_refinement_steps=4;
const double lowerEndWantedSpectrum=-10.0;
const unsigned int chebyshevOrder=2000; 
const unsigned int numSCFIterations=15;
const bool isPseudopotential = false;
//solver paramteters 
const unsigned int maxLinearSolverIterations=5000;
const double relLinearSolverTolerance=1.0e-14; 

//Define constants
const double TVal=500.0;

//Mesh information
#define meshFileName "../../../data/meshes/allElectron/carbon1Atom/meshPeriodic.inp"
#define coordinatesFile "../../../data/meshes/allElectron/carbon1Atom/coordinates.inp" 
  
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
    problem.run();
  }
  return 0;
}

