//Define parameters
const unsigned int dim=3;
const double radius=20.0;
unsigned int degree=2;
unsigned int FEOrder=1;
unsigned int n_refinement_steps=0;
unsigned int noOfEigenValues=1;
unsigned int quadratureRule=5;
unsigned int numAtomTypes=1;
double atomCharge=6.0;
char rhoFileName[100]="rhoInitialGuess/rho_C";

//solver paramteters
unsigned int maxLinearSolverIterations=5000;
double relLinearSolverTolerance=1.0e-12; 

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
