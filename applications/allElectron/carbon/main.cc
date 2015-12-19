//Define parameters
//const double radius=20.0;
//testing for one C atom
const unsigned int FEOrder=4;
const unsigned int n_refinement_steps=4;
const unsigned int numElectrons=6;
const unsigned int numEigenValues=numElectrons/2+2;
const double lowerEndWantedSpectrum=-10.0;
const unsigned int chebyshevOrder=2000; 
const unsigned int numSCFIterations=25;

//solver paramteters 
const unsigned int maxLinearSolverIterations=5000;
const double relLinearSolverTolerance=1.0e-12; 

//Define constants
const double TVal=500.0;
const double kb = 3.166811429e-06;

//Mesh information
//#define meshFileName "../../../data/mesh/singleAtom.inp"

//dft header
#include "../../../src/dft/dft.cc"

//Atom locations
void getAtomicLocations(dealii::Table<2,double>& atoms, std::map<unsigned int, std::string>& initialGuessFiles){
  atoms.reinit(1,4);
  atoms(0,0)=6.0; atoms(0,1)=0.0; atoms(0,2)=0.0; atoms(0,3)=0.0;  
  //initial guess for rho
  //initialGuessFiles.clear();
  initialGuessFiles[6]=std::string("../../../data/rhoInitialGuess/rho_C");
}

int main (int argc, char *argv[]){
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
  deallog.depth_console(0);
  {
    dftClass problem;
    getAtomicLocations(problem.atomLocations, problem.initialGuessFiles);
    problem.run();
  }
  return 0;
}

