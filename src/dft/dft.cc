// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author Shiva Rudraraju (2016), Phani Motamarri (2018)
//

//Include header files

#include "../../include/dft.h"
#include "../../include/eigen.h"
#include "../../include/poisson.h"
#include "../../include/force.h"
#include "../../include/meshMovementGaussian.h"
#include "../../include/fileReaders.h"


//Include cc files
#include "moveMeshToAtoms.cc"
#include "initUnmovedTriangulation.cc"
#include "initBoundaryConditions.cc"
#include "initElectronicFields.cc"


#include "psiInitialGuess.cc"
#include "energy.cc"
#include "charge.cc"
#include "density.cc"

#include "mixingschemes.cc"
#include "chebyshev.cc"
#include "solveVself.cc"

#include <complex>
#include <cmath>
#include <algorithm>

#ifdef ENABLE_PERIODIC_BC
#include "generateImageCharges.cc"
#endif



//
//dft constructor
//
template<unsigned int FEOrder>
dftClass<FEOrder>::dftClass():
  FE (FE_Q<3>(QGaussLobatto<1>(C_num1DQuad<FEOrder>())), 1),
#ifdef ENABLE_PERIODIC_BC
  FEEigen (FE_Q<3>(QGaussLobatto<1>(C_num1DQuad<FEOrder>())), 2),
#else
  FEEigen (FE_Q<3>(QGaussLobatto<1>(C_num1DQuad<FEOrder>())), 1),
#endif
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  numElectrons(0),
  numLevels(0),
  d_maxkPoints(1),
  integralRhoValue(0),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times)
{
  poissonPtr= new poissonClass<FEOrder>(this);
  eigenPtr= new eigenClass<FEOrder>(this);
  forcePtr= new forceClass<FEOrder>(this);
  //
  // initialize PETSc
  //
  PetscErrorCode petscError = SlepcInitialize(NULL,
					      NULL,
					      NULL,
					      NULL);


}

template<unsigned int FEOrder>
dftClass<FEOrder>::~dftClass()
{
    delete poissonPtr;
    delete eigenPtr;
    matrix_free_data.clear();
    delete forcePtr;
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

template<unsigned int FEOrder>
void dftClass<FEOrder>::set()
{
  //
  //read coordinates
  //
  unsigned int numberColumnsCoordinatesFile = 5;

#ifdef ENABLE_PERIODIC_BC

  //
  //read fractionalCoordinates of atoms in periodic case
  //
  dftUtils::readFile(numberColumnsCoordinatesFile, atomLocations, dftParameters::coordinatesFile);
  pcout << "number of atoms: " << atomLocations.size() << "\n";

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
      pcout<<"fractional coordinates of atom: "<<atomLocations[i][2]<<" "<<atomLocations[i][3]<<" "<<atomLocations[i][4]<<"\n";
    }

  //
  //read lattice Vectors
  //
  unsigned int numberColumnsLatticeVectorsFile = 3;
  dftUtils::readFile(numberColumnsLatticeVectorsFile,d_latticeVectors,dftParameters::latticeVectorsFile);
  for(int i = 0; i < d_latticeVectors.size(); ++i)
    {
      pcout<<"lattice vectors: "<<d_latticeVectors[i][0]<<" "<<d_latticeVectors[i][1]<<" "<<d_latticeVectors[i][2]<<"\n";
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
      pcout<<"Cartesian coordinates of atoms: "<<atomLocations[i][2]<<" "<<atomLocations[i][3]<<" "<<atomLocations[i][4]<<"\n";
    }

  //
  //create domain bounding vectors
  //
  d_domainBoundingVectors = d_latticeVectors;

#else
  dftUtils::readFile(numberColumnsCoordinatesFile, atomLocations, dftParameters::coordinatesFile);
  pcout << "number of atoms: " << atomLocations.size() << "\n";

  //
  //find unique atom types
  //
  for (std::vector<std::vector<double> >::iterator it=atomLocations.begin(); it<atomLocations.end(); it++)
    {
      atomTypes.insert((unsigned int)((*it)[0]));
    }

  //
  //print cartesian coordinates
  //
  for(int i = 0; i < atomLocations.size(); ++i)
    {
      pcout<<"Cartesian coordinates of atoms: "<<atomLocations[i][2]<<" "<<atomLocations[i][3]<<" "<<atomLocations[i][4]<<"\n";
    }


  std::vector<double> domainVector;
  domainVector.push_back(dftParameters::domainSizeX);domainVector.push_back(0.0);domainVector.push_back(0.0);
  d_domainBoundingVectors.push_back(domainVector);
  domainVector.clear();
  domainVector.push_back(0.0);domainVector.push_back(dftParameters::domainSizeY);domainVector.push_back(0.0);
  d_domainBoundingVectors.push_back(domainVector);
  domainVector.clear();
  domainVector.push_back(0.0);domainVector.push_back(0.0);domainVector.push_back(dftParameters::domainSizeZ);
  d_domainBoundingVectors.push_back(domainVector);
#endif

  pcout << "number of atoms types: " << atomTypes.size() << "\n";

  //
  //create domain bounding vectors
  //

  
  //estimate total number of wave functions
  determineOrbitalFilling();  

  pcout << "number of eigen values: " << numEigenValues << std::endl; 

  //
  //read kPoint data
  //
#ifdef ENABLE_PERIODIC_BC
  readkPointData();
#else
  d_maxkPoints = 1;
  d_kPointCoordinates.resize(3*d_maxkPoints,0.0);
  d_kPointWeights.resize(d_maxkPoints,1.0);
#endif

  pcout<<"actual k-Point-coordinates and weights: "<<std::endl;
  for(int i = 0; i < d_maxkPoints; ++i)
    {
      pcout<<d_kPointCoordinates[3*i + 0]<<" "<<d_kPointCoordinates[3*i + 1]<<" "<<d_kPointCoordinates[3*i + 2]<<" "<<d_kPointWeights[i]<<std::endl;
    } 
  
  //set size of eigenvalues and eigenvectors data structures
  eigenValues.resize(d_maxkPoints);
  a0.resize(d_maxkPoints,dftParameters::lowerEndWantedSpectrum);
  bLow.resize(d_maxkPoints,0.0);
  eigenVectors.resize(d_maxkPoints);
  eigenVectorsOrig.resize(d_maxkPoints);

  for(unsigned int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
    {
      eigenValues[kPoint].resize(numEigenValues);  
      for (unsigned int i=0; i<numEigenValues; ++i)
	{
	  eigenVectors[kPoint].push_back(new vectorType);
	  eigenVectorsOrig[kPoint].push_back(new vectorType);
	}
    }

  for (unsigned int i=0; i<numEigenValues; ++i){
    PSI.push_back(new vectorType);
    tempPSI.push_back(new vectorType);
    tempPSI2.push_back(new vectorType);
    tempPSI3.push_back(new vectorType);
    tempPSI4.push_back(new vectorType);
  } 
}

//dft run
template<unsigned int FEOrder>
void dftClass<FEOrder>::run ()
{
  pcout << std::endl << "number of MPI processes: "
	<< Utilities::MPI::n_mpi_processes(mpi_communicator)
	<< std::endl;

  //
  //read coordinates file 
  //
  set();

  
  //generate mesh
  //mesh();

  //
  //generate mesh (both parallel and serial)
  //
  d_mesh.generateSerialAndParallelMesh(atomLocations,
				       d_imagePositions,
				       d_domainBoundingVectors);


  //
  //get access to triangulation objects from meshGenerator class
  //
  parallel::distributed::Triangulation<3> & triangulationPar = d_mesh.getParallelMesh();
  Triangulation<3,3> & triangulationSer = d_mesh.getSerialMesh();
 
  //
  //initialize dofHandlers and hanging-node constraints and periodic constraints on the unmoved Mesh
  //
  initUnmovedTriangulation(triangulationPar);

  //
  //move triangulation to have atoms on triangulation vertices
  //
  moveMeshToAtoms(triangulationPar);
  moveMeshToAtoms(triangulationSer,true);//can only be called after calling moveMeshToAtoms(triangulationPar)


  //
  //initialize dirichlet BCs for total potential and vSelf poisson solutions
  //
  initBoundaryConditions();

  //
  //initialize guesses for electron-density and wavefunctions
  //
  initElectronicFields();
  
  //
  //initialize local pseudopotential
  //
  if(dftParameters::isPseudopotential)
  {
      initLocalPseudoPotential();
      initNonLocalPseudoPotential();
      computeSparseStructureNonLocalProjectors();
      computeElementalProjectorKets();
      forcePtr->initPseudoData();
  }   
  //
  //solve vself
  //
  solveVself();

  //
  //solve
  //
  computing_timer.enter_section("solve"); 

  
  //
  //Begin SCF iteration
  //
  unsigned int scfIter=0;
  double norm = 1.0;
  char buffer[100];

  while ((norm > dftParameters::selfConsistentSolverTolerance) && (scfIter < dftParameters::numSCFIterations))
    {
      sprintf(buffer, "\n\n**** Begin Self-Consistent-Field Iteration: %u ****\n", scfIter+1); pcout << buffer;
      //Mixing scheme
      if(scfIter > 0)
	{
	  if (scfIter==1) norm = mixing_simple();
	  else norm = sqrt(mixing_anderson());
	  sprintf(buffer, "Anderson Mixing: L2 norm of electron-density difference: %12.6e\n\n", norm); pcout << buffer;
	  poissonPtr->phiTotRhoIn = poissonPtr->phiTotRhoOut;
	}
      //phiTot with rhoIn

      //parallel loop over all elements

      int constraintMatrixId = phiTotDofHandlerIndex;
      sprintf(buffer, "Poisson solve for total electrostatic potential (rhoIn+b):\n"); pcout << buffer; 
      poissonPtr->solve(poissonPtr->phiTotRhoIn,constraintMatrixId, rhoInValues);
      //pcout<<"L-2 Norm of Phi-in   : "<<poissonPtr->phiTotRhoIn.l2_norm()<<std::endl;
      //pcout<<"L-inf Norm of Phi-in : "<<poissonPtr->phiTotRhoIn.linfty_norm()<<std::endl;

     
      //eigen solve

      if(dftParameters::xc_id < 4)
	{
	  if(dftParameters::isPseudopotential)
	    eigenPtr->computeVEff(rhoInValues, poissonPtr->phiTotRhoIn, poissonPtr->phiExt, pseudoValues);
	  else
	    eigenPtr->computeVEff(rhoInValues, poissonPtr->phiTotRhoIn, poissonPtr->phiExt); 
	}
      else if (dftParameters::xc_id == 4)
	{
	  if(dftParameters::isPseudopotential)
	    eigenPtr->computeVEff(rhoInValues, gradRhoInValues, poissonPtr->phiTotRhoIn, poissonPtr->phiExt, pseudoValues);
	  else
	    eigenPtr->computeVEff(rhoInValues, gradRhoInValues, poissonPtr->phiTotRhoIn, poissonPtr->phiExt);
	}
 


      for(int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
	{
	  d_kPointIndex = kPoint;
	  chebyshevSolver();
	}
      
      //fermi energy
      compute_fermienergy();

      //rhoOut
      compute_rhoOut();
      
      //compute integral rhoOut
      integralRhoValue=totalCharge(rhoOutValues);

      //phiTot with rhoOut
      sprintf(buffer, "Poisson solve for total electrostatic potential (rhoOut+b):\n"); pcout << buffer; 
      poissonPtr->solve(poissonPtr->phiTotRhoOut,constraintMatrixId, rhoOutValues);
      //pcout<<"L-2 Norm of Phi-out   :"<<poissonPtr->phiTotRhoOut.l2_norm()<<std::endl;
      //pcout<<"L-inf Norm of Phi-out :"<<poissonPtr->phiTotRhoOut.linfty_norm()<<std::endl;
      //energy
      compute_energy();
      pcout<<"SCF iteration " << scfIter+1 << " complete\n";
      
      //output wave functions
      output();
      
      //
      scfIter++;
    }
  computing_timer.enter_section("configurational force computation"); 
  forcePtr->computeAtomsForces();
  forcePtr->printAtomsForces();
  computing_timer.exit_section("configurational force computation");  
  computing_timer.exit_section("solve"); 
}

//Output
template <unsigned int FEOrder>
void dftClass<FEOrder>::output () {
  //only generate wave function output for serial runs
  if (n_mpi_processes>1) return;
  //
  DataOut<3> data_outEigen;
  data_outEigen.attach_dof_handler (dofHandlerEigen);
  for (unsigned int i=0; i<eigenVectors[0].size(); ++i){
    char buffer[100]; sprintf(buffer,"eigen%u", i);  
    data_outEigen.add_data_vector (*eigenVectors[0][i], buffer);
  }
  data_outEigen.build_patches (C_num1DQuad<FEOrder>());
  std::ofstream output ("eigen.vtu");
  data_outEigen.write_vtu (output);
  //Doesn't work with mvapich2_ib mpi libraries
  //data_outEigen.write_vtu_in_parallel(std::string("eigen.vtu").c_str(),mpi_communicator);
}

template class dftClass<1>;
template class dftClass<2>;
template class dftClass<3>;
template class dftClass<4>;
template class dftClass<5>;
template class dftClass<6>;
template class dftClass<7>;
template class dftClass<8>;
template class dftClass<9>;
template class dftClass<10>;
template class dftClass<11>;
template class dftClass<12>;
