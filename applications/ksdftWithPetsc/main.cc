//Define parameters
const unsigned int dim=3;
const double radius=20.0;
unsigned int FEOrder=1;
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

#include "../../src/dft.cc"

int main (int argc, char *argv[])
{
    Utilities::MPI::MPI_InitFinalize mpi_initialization (argc, argv);
    deallog.depth_console(0);
    {
      dft problem;
      problem.run();
    }
    return 0;
}
