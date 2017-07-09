//Define parameters
//const double radius=20.0;
//testing for one C atom
const double radiusAtomBall = 4.0;
const unsigned int FEOrder  = 4;
const unsigned int n_refinement_steps = 4;
const int periodic_x = 0; //For non-orthogonal unit-cells, this is a-direction
const int periodic_y = 0; //For non-orthogonal unit-cells, this is b-direction
const int periodic_z = 0; //For non-orthogonal unit-cells, this is c-direction


const double lowerEndWantedSpectrum   = -10.0;
const unsigned int chebyshevOrder     = 2000; 
const unsigned int numSCFIterations   = 20;
const unsigned int maxLinearSolverIterations = 5000;
const double relLinearSolverTolerance        = 1.0e-14; 
const double selfConsistentSolverTolerance   = 1.0e-11;


const bool isPseudopotential = false;


//
//Define constants
//
const double TVal = 500.0;

//
//Mesh information
//
#define meshFileName "../../../data/meshes/allElectron/carbon1Atom/mesh.inp"
#define coordinatesFile "../../../data/meshes/allElectron/carbon1Atom/coordinates.inp" 

#define xc_id 1

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
    problem.numberAtomicWaveFunctions[6] = 5;
    problem.numEigenValues = 5;
    problem.run();
  }
  return 0;
}

