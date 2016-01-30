#include <iostream>
#include <fstream>
#include <string>

//Define parameters
//const double radius=20.0;
//testing for one C atom
const unsigned int FEOrder=4;
const unsigned int n_refinement_steps=4;
const unsigned int numElectrons=6;
const unsigned int numEigenValues=numElectrons/2+2;
const double lowerEndWantedSpectrum=-10.0;
const unsigned int chebyshevOrder=2000; 
const unsigned int numSCFIterations=2;

//solver paramteters 
const unsigned int maxLinearSolverIterations=5000;
const double relLinearSolverTolerance=1.0e-12; 

//Define constants
const double TVal=500.0;
const double kb = 3.166811429e-06;

//Mesh information
#define meshFileName "../../../data/meshes/carbon1Atom/mesh.inp"
#define coordinatesFile "../../../data/meshes/carbon1Atom/coordinates.inp" 

//dft header
#include "../../../src2/dft/dft.cc"

//Atom locations
void getAtomicLocations(dealii::Table<2,double>& atoms, std::map<unsigned int, std::string>& initialGuessFiles){
  //open coordinates file
  std::ifstream coordsFile(coordinatesFile);
  if(coordsFile.fail()) {
    std::cerr<<"Error opening coordinatesFile\n";
    exit(-1);
  }
  //number of atoms
  unsigned int numAtoms;
  coordsFile >> numAtoms;
  std::cout << numAtoms << "\n";
  atoms.reinit(numAtoms,4);
  //
  unsigned int Z, atom=0;
  double x,y,z;
  while (!coordsFile.eof()) {
    coordsFile >> Z;
    coordsFile >> x; coordsFile >> y; coordsFile >> z;
    if (atom<numAtoms){
      atoms(atom,0)=(double) Z;
      atoms(atom,1)= x; atoms(atom,2)= y; atoms(atom,3)= z;
      atom++;
    }
    else{
      std::cerr << "atoms>=numAtoms\n"; exit(-1);
    }
    std::cout << Z << ", " << x << ", " << y << ", " << z << "\n";
  }
  //initial guess for rho
  for (unsigned int i=0; i<numAtoms; i++){
    Z=atoms(i,0);
    char densityFile[256];
    sprintf(densityFile, "../../../data/electronicStructure/z%u/density.inp", Z);
    initialGuessFiles[Z]=std::string(densityFile);
    std::cout << "initial density: " << initialGuessFiles[Z] << "\n";
  }
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

