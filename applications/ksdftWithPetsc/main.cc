//Define parameters
//const double radius=20.0;
unsigned int FEOrder=4;
unsigned int n_refinement_steps=0;
unsigned int numElectrons=10;
unsigned int numEigenValues=numElectrons/2;
unsigned int quadratureRule=5;

//solver paramteters 
unsigned int maxLinearSolverIterations=5000;
double relLinearSolverTolerance=1.0e-12; 

//Define constants
double TVal=100.0;
double kb = 3.166811429e-06;

//Mesh information
#define meshFileName "../../data/mesh/CH4.inp"

//dft header
#include "../../src/dft/dft.cc"

//Atom locations
void getAtomicLocations(dealii::Table<2,double>& atoms, std::map<unsigned int, std::string>& initialGuessFiles){
  atoms.reinit(5,4);
  atoms(0,0)=6.0; atoms(0,1)=0.0; atoms(0,2)=0.0; atoms(0,3)=0.0;  
  atoms(1,0)=1.0; atoms(1,1)=1.2; atoms(1,2)=1.2; atoms(1,3)=1.2;  
  atoms(2,0)=1.0; atoms(2,1)=-1.2; atoms(2,2)=-1.2; atoms(2,3)=1.2;  
  atoms(3,0)=1.0; atoms(3,1)=1.2; atoms(3,2)=-1.2; atoms(3,3)=-1.2;  
  atoms(4,0)=1.0; atoms(4,1)=-1.2; atoms(4,2)=1.2; atoms(4,3)=-1.2;  
  //initial guess for rho
  //initialGuessFiles.clear();
  initialGuessFiles[6]=std::string("../../data/rhoInitialGuess/rho_C");
  initialGuessFiles[1]=std::string("../../data/rhoInitialGuess/rho_H");
}

int main (int argc, char *argv[]){
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
  deallog.depth_console(0);
  {
    dft problem;
    getAtomicLocations(problem.atomLocations, problem.initialGuessFiles);
    problem.run();
  }
  return 0;
}

