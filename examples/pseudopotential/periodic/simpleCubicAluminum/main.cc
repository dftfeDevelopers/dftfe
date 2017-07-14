//
//Define parameters
//
#define ENABLE_PERIODIC_BC

//
//Testing for one C atom simple cubic with periodic boundary conditions
//
const double radiusAtomBall           = 3.0;
const unsigned int FEOrder            = 4; 
const unsigned int n_refinement_steps = 4;
const int periodic_x = 1; //For non-orthogonal unit-cells, this is a-direction
const int periodic_y = 1; //For non-orthogonal unit-cells, this is b-direction
const int periodic_z = 1; //For non-orthogonal unit-cells, this is c-direction


//
//Solver parameters
//
const double lowerEndWantedSpectrum = -0.3;
const unsigned int chebyshevOrder   = 50;
const unsigned int numSCFIterations = 50;
const unsigned int maxLinearSolverIterations = 5000;
const double relLinearSolverTolerance        = 1.0e-14; 
const double selfConsistentSolverTolerance   = 1.0e-11;

//
//decide whether it is all-electron and pseudopotential
//
const bool isPseudopotential   = true;
const double nlpTolerance      = 1.0e-08;


//
//Define constants
//
const double TVal=500.0;

//
//exchange correlation functional choice 
//

/*xc_id = 0 (No exchange correlation)
  xc_id = 1 (LDA: Perdew Zunger Ceperley Alder correlation with Slater Exchange[PRB. 23, 5048 (1981)])
  xc_id = 2 (LDA: Perdew-Wang 92 functional with Slater Exchange [PRB. 45, 13244 (1992)])
  xc_id = 3 (LDA: Vosko, Wilk & Nusair with Slater Exchange[Can. J. Phys. 58, 1200 (1980)])
  xc_id = 4 (GGA: Perdew-Burke-Ernzerhof functional [PRL. 77, 3865 (1996)])*/
#define xc_id 4

//
//Mesh information
//
#define meshFileName "../../../data/meshes/pseudoPotential/PeriodicSystems/simplecubic/aluminum/meshPeriodicsd1.inp"
//#define meshFileName "../../../data/meshes/pseudoPotential/PeriodicSystems/DebugMeshes/mesh.inp"
#define coordinatesFile "../../../data/meshes/pseudoPotential/PeriodicSystems/simplecubic/aluminum/coordinatesCenter.inp"
//#define coordinatesFile "../../../data/meshes/pseudoPotential/PeriodicSystems/simplecubic/aluminum/coordinatesCorner.inp"
#define latticeVectorsFile "../../../data/meshes/pseudoPotential/PeriodicSystems/simplecubic/aluminum/latticeVectors.inp"
#define kPointDataFile "../../../data/kPointList/OnekPoint.inp"
//#define kPointDataFile "../../../data/kPointList/2x2x2kPointRule.inp"
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
    problem.numberAtomicWaveFunctions[13]=9;
    problem.numEigenValues = 12;
    problem.run();
  }
  return 0;
}

