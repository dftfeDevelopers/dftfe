//FE and mesh parameters
const unsigned int FEOrder=4;
const unsigned int additionalSubDivisions=0;
//Chebyshev filter parameters
const double lowerEndWantedSpectrum=-10.0;
const unsigned int chebyshevOrder=2000; 
const unsigned int numSCFIterations=2;
//Poisson solver parameters 
const unsigned int maxLinearSolverIterations=5000;
const double relLinearSolverTolerance=1.0e-12; 

//constants
const double TVal=500.0;
const double kb = 3.166811429e-06;

//Mesh information
#define coordsFileName "../../../data/meshes/silicon18Atom/coordinates.inp"
#define meshFileName   "../../../data/meshes/silicon18Atom/mesh.inp"


//dft header
#include "../../../src2/dft/dft.cc"

//main
int main (int argc, char *argv[]){
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
  deallog.depth_console(0);
  {
    dftClass problem;
    //add additional levels, if needed
    problem.additionalLevels[14]=1;
    //run
    problem.run();
  }
  return 0;
}

