//
//Define parameters
//
#define ENABLE_PERIODIC_BC

//
//Testing for one C atom simple cubic with periodic boundary conditions
//
const double radiusAtomBall           = 3.0;
const unsigned int FEOrder            = 1; 
const unsigned int n_refinement_steps = 4;
const int periodic_x = 1; //For non-orthogonal unit-cells, this is a-direction
const int periodic_y = 1; //For non-orthogonal unit-cells, this is b-direction
const int periodic_z = 1; //For non-orthogonal unit-cells, this is c-direction

//
//Solver parameters
//
const double lowerEndWantedSpectrum = -10.0;
const unsigned int chebyshevOrder   = 50;
const unsigned int numSCFIterations = 40;
const unsigned int maxLinearSolverIterations = 5000;
const double relLinearSolverTolerance        = 1.0e-14; 

//
//decide whether it is all-electron and pseudopotential
//
const bool isPseudopotential = false;

//
//Define constants
//
const double TVal=500.0;

//
//Mesh information
//
#define meshFileName "../../../data/meshes/allElectron/PeriodicSystems/simplecubic/carbon/meshPeriodic.inp"
//#define coordinatesFile "../../../data/meshes/allElectron/PeriodicSystems/simplecubic/carbon/coordinatesCenter.inp"
#define coordinatesFile "../../../data/meshes/allElectron/PeriodicSystems/simplecubic/carbon/coordinatesCorner.inp"
#define latticeVectorsFile "../../../data/meshes/allElectron/PeriodicSystems/simplecubic/carbon/latticeVectors.inp"

/*#define meshFileName "../../../data/meshes/allElectron/PeriodicSystems/bcc/carbon/meshPeriodic.inp"
#define coordinatesFile "../../../data/meshes/allElectron/PeriodicSystems/bcc/carbon/coordinatesPeriodic.inp"
#define latticeVectorsFile "../../../data/meshes/allElectron/PeriodicSystems/bcc/carbon/latticeVectors.inp"*/

//  
//dft header
// 
#include "../../../src2/dft/dft.cc"

int main (int argc, char *argv[]){
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
  deallog.depth_console(0);
  {
    // set stdout precision
    //
    std::cout << std::scientific << std::setprecision(18);
    dftClass problem;
    problem.numberAtomicWaveFunctions[6]=5;
    problem.numEigenValues = 10;
    problem.run();
  }
  return 0;
}

