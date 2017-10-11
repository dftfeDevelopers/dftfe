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
// @author Shiva Rudraraju (2016), Phani Motamarri (2016)
//

//Include header files
#include "../../include/headers.h"
#include "../../include/dft.h"
#include "../../utils/fileReaders.cc"
#include "../poisson/poisson.cc"
#include "../eigen/eigen.cc"
#include "mesh.cc"
#include "init.cc"
#include "psiInitialGuess.cc"
#include "energy.cc"
#include "charge.cc"
#include "density.cc"
#include "symmetrizeRho.cc"
#include "locatenodes.cc"
#include "createBins.cc"
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
  triangulation (MPI_COMM_WORLD),
  FE (FE_Q<3>(QGaussLobatto<1>(FEOrder+1)), 1),
#ifdef ENABLE_PERIODIC_BC
  FEEigen (FE_Q<3>(QGaussLobatto<1>(FEOrder+1)), 2),
#else
  FEEigen (FE_Q<3>(QGaussLobatto<1>(FEOrder+1)), 1),
#endif
  dofHandler (triangulation),
  dofHandlerEigen (triangulation),
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  poisson(this),
  eigen(this),
  numElectrons(0),
  numLevels(0),
  d_maxkPoints(1),
  integralRhoValue(0),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times)
{

  //
  // initialize PETSc
  //
  PetscErrorCode petscError = SlepcInitialize(NULL,
					      NULL,
					      NULL,
					      NULL);


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
  readFile(numberColumnsCoordinatesFile, atomLocations, coordinatesFile);
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
  readFile(numberColumnsLatticeVectorsFile,d_latticeVectors,latticeVectorsFile);
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
      pcout<<"Cartesian coordinates of atoms: "<<atomLocations[i][2]<<" "<<atomLocations[i][3]<<" "<<atomLocations[i][4]<<"\n";
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

  pcout << "number of eigen values: " << numEigenValues << std::endl; 

  //
  //read kPoint data
  //
#ifdef ENABLE_PERIODIC_BC
  //readkPointData();
   generateMPGrid();
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
  eigenValuesTemp.resize(d_maxkPoints);
  a0.resize((spinPolarized+1)*d_maxkPoints,lowerEndWantedSpectrum);
  bLow.resize((spinPolarized+1)*d_maxkPoints,0.0);
  eigenVectors.resize((1+spinPolarized)*d_maxkPoints);
  eigenVectorsOrig.resize((1+spinPolarized)*d_maxkPoints);
  //
  //char buffer[100];
  //sprintf(buffer, "%s:%10u\n", "check 0", eigenVectors.size());
  //pcout << buffer;
  //
  for(unsigned int kPoint = 0; kPoint < (1+spinPolarized)*d_maxkPoints; ++kPoint)
    {
      //for (unsigned int j=0; j<(spinPolarized+1); ++j) // for spin
       //{
        for (unsigned int i=0; i<numEigenValues; ++i)
	  {
	    eigenVectors[kPoint].push_back(new vectorType);
	    eigenVectorsOrig[kPoint].push_back(new vectorType);
	  }
       //}
    }
   for(unsigned int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
    {
      eigenValues[kPoint].resize((spinPolarized+1)*numEigenValues);  
      eigenValuesTemp[kPoint].resize(numEigenValues); 
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
  //if meshFile provided, pass to mesh()
  mesh();


  //
  //initialize
  //
  init();

#ifdef ENABLE_PERIODIC_BC
  displayQuadPoints() ;
#endif
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
  char buffer[100];
  double norm = 1.0;
  while ((norm > selfConsistentSolverTolerance) && (scfIter < numSCFIterations))
    {
      sprintf(buffer, "\n\n**** Begin Self-Consistent-Field Iteration: %u ****\n", scfIter+1); pcout << buffer;
      //Mixing scheme
      if(scfIter > 0)
	{
	  if (scfIter==1)
              {
		if (spinPolarized==1)
                  {
		    //for (unsigned int s=0; s<2; ++s)
		       norm = mixing_simple_spinPolarized(); 
		  }
		else
	          norm = mixing_simple();
	      }
	  else 
             {
		if (spinPolarized==1)
		  {
		    //for (unsigned int s=0; s<2; ++s)
		       norm = sqrt(mixing_anderson_spinPolarized());
		  } 
		else
	          norm = sqrt(mixing_anderson());		
	      }
	  sprintf(buffer, "Anderson Mixing: L2 norm of electron-density difference: %12.6e\n\n", norm); pcout << buffer;
	  poisson.phiTotRhoIn = poisson.phiTotRhoOut;
	}
      //phiTot with rhoIn

      //parallel loop over all elements

      int constraintMatrixId = 1;
      sprintf(buffer, "Poisson solve for total electrostatic potential (rhoIn+b):\n"); pcout << buffer; 
      poisson.solve(poisson.phiTotRhoIn,constraintMatrixId, rhoInValues);
      //pcout<<"L-2 Norm of Phi-in   : "<<poisson.phiTotRhoIn.l2_norm()<<std::endl;
      //pcout<<"L-inf Norm of Phi-in : "<<poisson.phiTotRhoIn.linfty_norm()<<std::endl;

     
      //eigen solve
      if (spinPolarized==1)
	{
	  for(unsigned int s=0; s<2; ++s)
	      {
	       if(xc_id < 4) 
	        {
		  if(isPseudopotential)
		    eigen.computeVEffSpinPolarized(rhoInValuesSpinPolarized, poisson.phiTotRhoIn, poisson.phiExt, s, pseudoValues);
		  else
		    eigen.computeVEffSpinPolarized(rhoInValuesSpinPolarized, poisson.phiTotRhoIn, poisson.phiExt, s);
                }
	       else if (xc_id == 4)
	        {
	          if(isPseudopotential)
		    eigen.computeVEffSpinPolarized(rhoInValuesSpinPolarized, gradRhoInValuesSpinPolarized, poisson.phiTotRhoIn, poisson.phiExt, s, pseudoValues);
	          else
		    eigen.computeVEffSpinPolarized(rhoInValuesSpinPolarized, gradRhoInValuesSpinPolarized, poisson.phiTotRhoIn, poisson.phiExt, s);
	        }
	      for (int kPoint = 0; kPoint < d_maxkPoints; ++kPoint) 
	        {
	          d_kPointIndex = kPoint;
	          char buffer[100];
	          for(int j = 0; j < numPass; ++j)
	            { 
		       sprintf(buffer, "%s:%3u%s:%3u\n", "Beginning Chebyshev filter pass ", j+1, " for spin ", s+1);
		       pcout << buffer;
		       chebyshevSolver(s);
	            }
	        }
	    }
        }
      else
        {
	  if(xc_id < 4)
	      {
	      if(isPseudopotential)
		eigen.computeVEff(rhoInValues, poisson.phiTotRhoIn, poisson.phiExt, pseudoValues);
	      else
		eigen.computeVEff(rhoInValues, poisson.phiTotRhoIn, poisson.phiExt); 
	      }
	  else if (xc_id == 4)
	     {
	      if(isPseudopotential)
		eigen.computeVEff(rhoInValues, gradRhoInValues, poisson.phiTotRhoIn, poisson.phiExt, pseudoValues);
	      else
		eigen.computeVEff(rhoInValues, gradRhoInValues, poisson.phiTotRhoIn, poisson.phiExt);
	     } 
        
	  for (int kPoint = 0; kPoint < d_maxkPoints; ++kPoint) 
	    {
	      d_kPointIndex = kPoint;
	      char buffer[100];
	      for(int j = 0; j < numPass; ++j)
	      { 
		    sprintf(buffer, "%s:%3u\n", "Beginning Chebyshev filter pass ", j+1);
		    pcout << buffer;
		    chebyshevSolver(0);
	      }
	    }
	}
 
       //fermi energy
        compute_fermienergy();
	//rhoOut
#ifdef ENABLE_PERIODIC_BC
   if (useSymm)
	computeAndSymmetrize_rhoOut();
   else
       compute_rhoOut();
#else
   compute_rhoOut();
#endif
      
      //compute integral rhoOut
      integralRhoValue=totalCharge(rhoOutValues);

      //phiTot with rhoOut
      sprintf(buffer, "Poisson solve for total electrostatic potential (rhoOut+b):\n"); pcout << buffer; 
      poisson.solve(poisson.phiTotRhoOut,constraintMatrixId, rhoOutValues);
      //pcout<<"L-2 Norm of Phi-out   :"<<poisson.phiTotRhoOut.l2_norm()<<std::endl;
      //pcout<<"L-inf Norm of Phi-out :"<<poisson.phiTotRhoOut.linfty_norm()<<std::endl;
      //energy
      if (spinPolarized==1)
         compute_energy_spinPolarized();
      else
     	 compute_energy () ;
      pcout<<"SCF iteration " << scfIter+1 << " complete\n";
      
      //output wave functions
      output();
      
      //
      scfIter++;
    }
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
  data_outEigen.build_patches (FEOrder+1);
  std::ofstream output ("eigen.vtu");
  data_outEigen.write_vtu (output);
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
