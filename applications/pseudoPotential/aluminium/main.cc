//
//Define parameters
//
const double radiusAtomBall           = 3.0;
const unsigned int FEOrder            = 1;
const unsigned int n_refinement_steps = 0;
const double lowerEndWantedSpectrum   = -0.3;
const unsigned int chebyshevOrder     = 1500; 
const unsigned int numSCFIterations   = 1;
const bool isPseudopotential          = true;
const double nlpTolerance = 1.0e-08;

//
//Solver parameters 
//
const unsigned int maxLinearSolverIterations = 5000;
const double relLinearSolverTolerance        = 1.0e-12; 
const double selfConsistentSolverTolerance   = 1.0e-11;

//
//Define constants
//
const double TVal = 500.0;

//
//Mesh information
//
#define meshFileName "../../../data/meshes/pseudoPotential/aluminum1Atom/mesh.inp"
#define coordinatesFile "../../../data/meshes/pseudoPotential/aluminum1Atom/coordinates.inp" 

//
//dft header
//
#include "../../../src2/dft/dft.cc"

int main (int argc, char *argv[]){
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
  deallog.depth_console(0);
  {
    //
    // set stdout precision
    //
    std::cout << std::scientific << std::setprecision(18);
    dftClass problem;
    problem.numberAtomicWaveFunctions[13] = 9;
    problem.numEigenValues = 9;
    problem.run();
  }
  return 0;
}


