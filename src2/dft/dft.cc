//Include header files
#include "../../include2/headers.h"
#include "../../include2/dft.h"
#include "../../utils/fileReaders.cc"
#include "../poisson/poisson.cc"
#include "../eigen/eigen.cc"
#include "mesh.cc"
#include "init.cc"
#include "psiInitialGuess.cc"
#include "energy.cc"
#include "charge.cc"
#include "density.cc"
#include "locatenodes.cc"
#include "createBins.cc"
#include "mixingschemes.cc"
#include "chebyshev.cc"
#include "solveVself.cc"
#ifdef ENABLE_PERIODIC_BC
#include "generateImageCharges.cc"
#endif
 
//dft constructor
dftClass::dftClass():
  triangulation (MPI_COMM_WORLD),
  FE (QGaussLobatto<1>(FEOrder+1)),
    dofHandler (triangulation),
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  poisson(this),
  eigen(this),
  numElectrons(0),
  numBaseLevels(0),
  numLevels(0),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times),
  bLow(0.0),
  a0(lowerEndWantedSpectrum)
{

}

void convertToCellCenteredCartesianCoordinates(std::vector<std::vector<double> > & atomLocations,
					       std::vector<std::vector<double> > & latticeVectors)
{
  std::vector<double> cartX(atomLocations.size(),0.0);
  std::vector<double> cartY(atomLocations.size(),0.0);
  std::vector<double> cartZ(atomLocations.size(),0.0);

  //
  //convert fractional atomic coordinates to cartesian coordinates
  //
  for(int i = 0; i < atomLocations.size(); ++i)
    {
      cartX[i] = atomLocations[i][2]*latticeVectors[0][0] + atomLocations[i][3]*latticeVectors[1][0] + atomLocations[i][4]*latticeVectors[2][0];
      cartY[i] = atomLocations[i][2]*latticeVectors[0][1] + atomLocations[i][3]*latticeVectors[1][1] + atomLocations[i][4]*latticeVectors[2][1];
      cartZ[i] = atomLocations[i][2]*latticeVectors[0][2] + atomLocations[i][3]*latticeVectors[1][2] + atomLocations[i][4]*latticeVectors[2][2];
    }

  //
  //define cell centroid (confirm whether it will work for non-orthogonal lattice vectors)
  //
  double cellCentroidX = 0.5*(latticeVectors[0][0] + latticeVectors[1][0] + latticeVectors[2][0]);
  double cellCentroidY = 0.5*(latticeVectors[0][1] + latticeVectors[1][1] + latticeVectors[2][1]);
  double cellCentroidZ = 0.5*(latticeVectors[0][2] + latticeVectors[1][2] + latticeVectors[2][2]);

  for(int i = 0; i < atomLocations.size(); ++i)
    {
      atomLocations[i][2] = cartX[i] - cellCentroidX;
      atomLocations[i][3] = cartY[i] - cellCentroidY;
      atomLocations[i][4] = cartZ[i] - cellCentroidZ;
    }
}


void dftClass::set(){
  //
  //read coordinates
  //
  unsigned int numberColumnsCoordinatesFile = 5;

#ifdef ENABLE_PERIODIC_BC

  //
  //read fractionalCoordinates of atoms in periodic case
  //
  readFile(numberColumnsCoordinatesFile, atomLocations, coordinatesFile);
  pcout << "Number of Atoms: " << atomLocations.size() << "\n";

  //
  //find unique atom types
  //
  for (std::vector<std::vector<double> >::iterator it=atomLocations.begin(); it<atomLocations.end(); it++)
    {
      atomTypes.insert((unsigned int)((*it)[0]));
    }

  //
  //print fractional coordinates
  //
  for(int i = 0; i < atomLocations.size(); ++i)
    {
      pcout<<"Fractional Coordinates: "<<atomLocations[i][2]<<" "<<atomLocations[i][3]<<" "<<atomLocations[i][4]<<"\n";
    }

  //
  //read lattice Vectors
  //
  unsigned int numberColumnsLatticeVectorsFile = 3;
  readFile(numberColumnsLatticeVectorsFile,d_latticeVectors,latticeVectorsFile);
  for(int i = 0; i < d_latticeVectors.size(); ++i)
    {
      pcout<<"Lattice Vectors: "<<d_latticeVectors[i][0]<<" "<<d_latticeVectors[i][1]<<" "<<d_latticeVectors[i][2]<<"\n";
    }

  //
  //generate Image charges
  //
  generateImageCharges();


  //
  //find cell-centered cartesian coordinates
  //
  convertToCellCenteredCartesianCoordinates(atomLocations,
					    d_latticeVectors);

  //
  //print cartesian coordinates
  //
  for(int i = 0; i < atomLocations.size(); ++i)
    {
      pcout<<"Cartesian Coordinates: "<<atomLocations[i][2]<<" "<<atomLocations[i][3]<<" "<<atomLocations[i][4]<<"\n";
    }

#else
  readFile(numberColumnsCoordinatesFile, atomLocations, coordinatesFile);
  pcout << "number of atoms: " << atomLocations.size() << "\n";

  //
  //find unique atom types
  //
  for (std::vector<std::vector<double> >::iterator it=atomLocations.begin(); it<atomLocations.end(); it++){
    atomTypes.insert((unsigned int)((*it)[0]));
  }

  //
  //print cartesian coordinates
  //
  for(int i = 0; i < atomLocations.size(); ++i)
    {
      pcout<<"Cartesian Coordinates: "<<atomLocations[i][2]<<" "<<atomLocations[i][3]<<" "<<atomLocations[i][4]<<"\n";
    }
#endif

  /*readFile(numberColumnsCoordinatesFile, atomLocations, coordinatesFile);
  pcout << "number of atoms: " << atomLocations.size() << "\n";
  //find unique atom types
  for (std::vector<std::vector<double> >::iterator it=atomLocations.begin(); it<atomLocations.end(); it++){
    atomTypes.insert((unsigned int)((*it)[0]));
    }*/
 
  pcout << "number of atoms types: " << atomTypes.size() << "\n";


  
  //estimate total number of wave functions
  determineOrbitalFilling();  
  //numEigenValues=waveFunctionsVector.size();
  pcout << "num of eigen values: " << numEigenValues << std::endl; 
  //set size of eigenvalues and eigenvectors data structures
  eigenValues.resize(numEigenValues);
  for (unsigned int i=0; i<numEigenValues; ++i){
    eigenVectors.push_back(new vectorType);
    PSI.push_back(new vectorType);
    tempPSI.push_back(new vectorType);
    tempPSI2.push_back(new vectorType);
    tempPSI3.push_back(new vectorType);
  } 
}

//dft run
void dftClass::run ()
{
  pcout << "number of MPI processes: "
	<< Utilities::MPI::n_mpi_processes(mpi_communicator)
	<< std::endl;

  //
  //read coordinates file 
  //
  set();
  
  //generate mesh
  //if meshFile provided, pass to mesh()
  mesh();

  //
  //initialize
  //
  init();

  //
  //solve vself
  //
  solveVself();
 
  //
  //solve
  //
  computing_timer.enter_section("dft solve"); 

  //
  //Begin SCF iteration
  //
  unsigned int scfIter=0;
  double norm=1.0;
  while ((norm > 1.0e-13) && (scfIter < numSCFIterations))
    {
      if(this_mpi_process==0) printf("\n\nBegin SCF Iteration:%u\n", scfIter+1);
      //Mixing scheme
      if(scfIter > 0)
	{
	  if (scfIter==1) norm=mixing_simple();
	  else norm=mixing_anderson();
	  if(this_mpi_process==0) printf("Mixing Scheme: iter:%u, norm:%12.6e\n", scfIter+1, norm);
	}
      //phiTot with rhoIn
      int constraintMatrixId = 1;
      poisson.solve(poisson.phiTotRhoIn,constraintMatrixId,rhoInValues);
      std::cout<<"L2 Norm of Phi out Tot L2  : "<<poisson.phiTotRhoIn.l2_norm()<<std::endl;
      std::cout<<"L2 Norm of Phi out Tot Linf: "<<poisson.phiTotRhoIn.linfty_norm()<<std::endl;
      //visualise
      DataOut<3> data_out;
      data_out.attach_dof_handler (dofHandler);
      data_out.add_data_vector (poisson.phiTotRhoIn, "solution");
      data_out.build_patches (4);
      std::ofstream output ("poisson.vtu");
      data_out.write_vtu (output);
      //eigen solve
      eigen.computeVEff(rhoInValues, poisson.phiTotRhoIn); 
      chebyshevSolver();
      //fermi energy
      compute_fermienergy();
      //rhoOut
      compute_rhoOut();
      //compute integral rhoOut
      double integralRhoOut=totalCharge(rhoOutValues);
      char buffer[100];
      sprintf(buffer, "Number of Electrons: %18.16e \n", integralRhoOut);
      pcout << buffer;
      //phiTot with rhoOut
      poisson.solve(poisson.phiTotRhoOut,constraintMatrixId, rhoOutValues);
      pcout<<"L2 Norm of Phi out Tot L2  : "<<poisson.phiTotRhoOut.l2_norm()<<std::endl;
      pcout<<"L2 Norm of Phi out Tot Linf: "<<poisson.phiTotRhoOut.linfty_norm()<<std::endl;
      //energy
      compute_energy();
      pcout<<"SCF iteration: " << scfIter+1 << " complete\n";
      scfIter++;
    }
  computing_timer.exit_section("dft solve"); 
}

