//Define parameters
const double radius=20.0;
unsigned int FEOrder=4;
unsigned int n_refinement_steps=0;
unsigned int numElectrons=6;
unsigned int numEigenValues=numElectrons/2+2;
unsigned int quadratureRule=5;
const unsigned int numAtomTypes=1;
double atomCharge=6.0;
char rhoFileName[100]="rhoInitialGuess/rho_C";

//solver paramteters 
unsigned int maxLinearSolverIterations=5000;
double relLinearSolverTolerance=1.0e-12; 

//Define constants
double TVal=100.0;
double kb = 3.166811429e-06;
#include "../../src/dft/dft.cc"

//Atom locations
void getAtomicLocations(dealii::Table<2,double>& atoms){
  atoms.reinit(1,4);
  atoms(0,0)=6.0; atoms(0,1)=0.0; atoms(0,2)=0.0; atoms(0,3)=0.0;  
}

int main (int argc, char *argv[]){
  Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
  deallog.depth_console(0);
  {
    dft problem;
    getAtomicLocations(problem.atomLocations);
    problem.run();
  }
  return 0;
}

