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
#include "../../include/symmetry.h"
#include "../../include/geoOptIon.h"
#include "../../include/meshMovementGaussian.h"
#include "../../include/fileReaders.h"
#include "../../include/dftParameters.h"


//Include cc files
#include "moveMeshToAtoms.cc"
#include "initUnmovedTriangulation.cc"
#include "initBoundaryConditions.cc"
#include "initElectronicFields.cc"
#include "initPseudo.cc"
#include "initPseudo-OV.cc"
#include "initRho.cc"


#include "psiInitialGuess.cc"
#include "energy.cc"
#include "charge.cc"
#include "density.cc"
#include "nscf.cc"
//#include "polarization.cc"
//#include "symmetrizeRho.cc"

#include "mixingschemes.cc"
#include "chebyshev.cc"
#include "solveVself.cc"

#include <complex>
#include <cmath>
#include <algorithm>
#include "linalg.h"
#include "stdafx.h"
#ifdef ENABLE_PERIODIC_BC
#include "generateImageCharges.cc"
//#include "initGroupSymmetry.cc"
#endif


using namespace dftParameters ;

//
//dft constructor
//
template<unsigned int FEOrder>
dftClass<FEOrder>::dftClass(MPI_Comm &mpi_comm_replica, MPI_Comm &interpoolcomm):
  FE (FE_Q<3>(QGaussLobatto<1>(C_num1DQuad<FEOrder>())), 1),
#ifdef ENABLE_PERIODIC_BC
  FEEigen (FE_Q<3>(QGaussLobatto<1>(C_num1DQuad<FEOrder>())), 2),
#else
  FEEigen (FE_Q<3>(QGaussLobatto<1>(C_num1DQuad<FEOrder>())), 1),
#endif
  mpi_communicator (mpi_comm_replica),
  interpoolcomm (interpoolcomm),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  numElectrons(0),
  numLevels(0),
  d_maxkPoints(1),
  integralRhoValue(0),
  d_mesh(mpi_comm_replica),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times)
{
  poissonPtr= new poissonClass<FEOrder>(this, mpi_comm_replica);
  eigenPtr= new eigenClass<FEOrder>(this, mpi_comm_replica);
  forcePtr= new forceClass<FEOrder>(this, mpi_comm_replica);
  symmetryPtr= new symmetryClass<FEOrder>(this, mpi_comm_replica, interpoolcomm);
  geoOptIonPtr= new geoOptIon<FEOrder>(this, mpi_comm_replica);
  //
  // initialize PETSc
  //
  PetscErrorCode petscError = SlepcInitialize(NULL,
					      NULL,
					      NULL,
					      NULL);

  pseudoValues=NULL;
}

template<unsigned int FEOrder>
dftClass<FEOrder>::~dftClass()
{
    delete poissonPtr;
    delete eigenPtr;
    delete symmetryPtr;
    matrix_free_data.clear();
    delete forcePtr;
    delete geoOptIonPtr;
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
  pcout << std::endl << "number of MPI processes: "
	<< Utilities::MPI::n_mpi_processes(mpi_communicator)
	<< std::endl;       
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
  atomLocationsFractional.resize(atomLocations.size()) ;
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
      atomLocationsFractional[i] = atomLocations[i] ;
      pcout<<"fractional coordinates of atom: "<<atomLocationsFractional[i][2]<<" "<<atomLocationsFractional[i][3]<<" "<<atomLocationsFractional[i][4]<<"\n";
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
  //atomLocationsFractional = atomLocations;
  /*for(int i = 0; i < atomLocationsFractional.size(); ++i)
    {
      atomLocationsFractional[i][2] = atomLocationsFractional[i][2] - 0.5;
      atomLocationsFractional[i][3] = atomLocationsFractional[i][3] - 0.5;
      atomLocationsFractional[i][4] = atomLocationsFractional[i][4] - 0.5;
    }*/ 
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
  //readkPointData();
   generateMPGrid();
   if (useSymm)
      symmetryPtr->test_spg_get_ir_reciprocal_mesh() ;
#else
  d_maxkPoints = 1;
  d_kPointCoordinates.resize(3*d_maxkPoints,0.0);
  d_kPointWeights.resize(d_maxkPoints,1.0);
#endif
  char buffer[100];
  pcout<<"actual k-Point-coordinates and weights: "<<std::endl;
  for(int i = 0; i < d_maxkPoints; ++i){
    sprintf(buffer, "  %5u:  %12.5f  %12.5f %12.5f %12.5f\n", i, d_kPointCoordinates[3*i+0], d_kPointCoordinates[3*i+1], d_kPointCoordinates[3*i+2],d_kPointWeights[i]);
    pcout << buffer;
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
  } 
}


//dft init
template<unsigned int FEOrder>
void dftClass<FEOrder>::init ()
{
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
  parallel::distributed::Triangulation<3> & triangulationSer = d_mesh.getSerialMesh();
  writeMesh("meshInitial");
  //
  //initialize dofHandlers and hanging-node constraints and periodic constraints on the unmoved Mesh
  //
  initUnmovedTriangulation(triangulationPar);
#ifdef ENABLE_PERIODIC_BC
 if (useSymm)
    symmetryPtr->initSymmetry() ;
#endif
  //
  //move triangulation to have atoms on triangulation vertices
  //
  //pcout << " check 0.11 : " << std::endl ;
  
  moveMeshToAtoms(triangulationPar);
  //moveMeshToAtoms(triangulationSer,true);//can only be called after calling moveMeshToAtoms(triangulationPar)


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
   if(isPseudopotential)
    {
      initLocalPseudoPotential();
      //
      if (pseudoProjector==2)
         initNonLocalPseudoPotential_OV() ;
      else
         initNonLocalPseudoPotential();	
      //
      //
      if (pseudoProjector==2){
         computeSparseStructureNonLocalProjectors_OV();
         computeElementalOVProjectorKets();
	}
      else{
	 computeSparseStructureNonLocalProjectors();
         computeElementalProjectorKets();
	}
     
      forcePtr->initPseudoData();
	
    }
   pcout << " check 0.5 : " << std::endl ;
  
}

//dft run
template<unsigned int FEOrder>
void dftClass<FEOrder>::run()
{
  solve();
  if (dftParameters::isIonOpt)
  {
    geoOptIonPtr->init();
    geoOptIonPtr->run();
  }
}

//dft solve
template<unsigned int FEOrder>
void dftClass<FEOrder>::solve()
{  

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
      if (spinPolarized==1)
	{
	  for(unsigned int s=0; s<2; ++s)
	      {
	       if(dftParameters::xc_id < 4) 
	        {
		  if(dftParameters::isPseudopotential)
		    eigenPtr->computeVEffSpinPolarized(rhoInValuesSpinPolarized, poissonPtr->phiTotRhoIn, poissonPtr->phiExt, s, pseudoValues);
		  else
		    eigenPtr->computeVEffSpinPolarized(rhoInValuesSpinPolarized, poissonPtr->phiTotRhoIn, poissonPtr->phiExt, s);
                }
	       else if (dftParameters::xc_id == 4)
	        {
	          if(dftParameters::isPseudopotential)
		    eigenPtr->computeVEffSpinPolarized(rhoInValuesSpinPolarized, gradRhoInValuesSpinPolarized, poissonPtr->phiTotRhoIn, poissonPtr->phiExt, s, pseudoValues);
	          else
		    eigenPtr->computeVEffSpinPolarized(rhoInValuesSpinPolarized, gradRhoInValuesSpinPolarized, poissonPtr->phiTotRhoIn, poissonPtr->phiExt, s);
	        }
	      for (int kPoint = 0; kPoint < d_maxkPoints; ++kPoint) 
	        {
	          d_kPointIndex = kPoint;
	          char buffer[100];
	          for(int j = 0; j < dftParameters::numPass; ++j)
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
        
	  for (int kPoint = 0; kPoint < d_maxkPoints; ++kPoint) 
	    {
	      d_kPointIndex = kPoint;
	      for(int j = 0; j < dftParameters::numPass; ++j)
	      { 
		    sprintf(buffer, "%s:%3u\n", "Beginning Chebyshev filter pass ", j+1);
		    pcout << buffer;
		    chebyshevSolver(0);
	      }
	    }
	  if (norm>1e-2)
	  {
	      //fermi energy
	      compute_fermienergy();
	      //maximum of the residual norm of the highest occupied state among all k points
	      double maxRes = computeMaximumHighestOccupiedStateResidualNorm();
	      pcout << "Maximum residual norm of the highest occupied state: "<< maxRes << std::endl;
	      //if the maximum residual norm of the highest occupied state is greater than 1e-2 (heuristic) 
	      // do one more pass of chebysev filter. This improves the scf convergence performance. Currently this
	      // approach is not implemented for spin-polarization case
	      if (maxRes>1e-2)
	      {
		  for (int kPoint = 0; kPoint < d_maxkPoints; ++kPoint) 
		    {
		      d_kPointIndex = kPoint;
		      sprintf(buffer, "%s:%3u\n", "Beginning Chebyshev filter pass ", dftParameters::numPass+1);
		      pcout << buffer;
		      chebyshevSolver(0);
		    }	      
	      }
	  }

	}
 
       //fermi energy
       compute_fermienergy();
	//rhoOut
   computing_timer.enter_section("compute rho"); 
#ifdef ENABLE_PERIODIC_BC
   if (useSymm){
	symmetryPtr->computeLocalrhoOut();
	symmetryPtr->computeAndSymmetrize_rhoOut();
    }
   else
       compute_rhoOut();
#else
   compute_rhoOut();
#endif
    computing_timer.exit_section("compute rho"); 
      
      //compute integral rhoOut
      integralRhoValue=totalCharge(rhoOutValues);

      //phiTot with rhoOut
      sprintf(buffer, "Poisson solve for total electrostatic potential (rhoOut+b):\n"); pcout << buffer; 
      poissonPtr->solve(poissonPtr->phiTotRhoOut,constraintMatrixId, rhoOutValues);
      //pcout<<"L-2 Norm of Phi-out   :"<<poissonPtr->phiTotRhoOut.l2_norm()<<std::endl;
      //pcout<<"L-inf Norm of Phi-out :"<<poissonPtr->phiTotRhoOut.linfty_norm()<<std::endl;

      //energy
      if (spinPolarized==1)
         compute_energy_spinPolarized();
      else
     	 compute_energy () ;
      pcout<<"SCF iteration " << scfIter+1 << " complete\n";
      
      //output wave functions
      //output();
      
      //
      scfIter++;
    }
 computing_timer.exit_section("solve");
#ifdef ENABLE_PERIODIC_BC
 if (useSymm)
    symmetryPtr->clearMaps() ;
#endif
//
 computing_timer.enter_section(" pp "); 
#ifdef ENABLE_PERIODIC_BC
  if ((Utilities::MPI::this_mpi_process(interpoolcomm))==0){
     pcout<<"Beginning nscf calculation "<<std::endl;
     readkPointData() ;
     char buffer[100];
     pcout<<"actual k-Point-coordinates and weights: "<<std::endl;
     for(int i = 0; i < d_maxkPoints; ++i){
       sprintf(buffer, "  %5u:  %12.5f  %12.5f %12.5f %12.5f\n", i, d_kPointCoordinates[3*i+0], d_kPointCoordinates[3*i+1], d_kPointCoordinates[3*i+2],d_kPointWeights[i]);
       pcout << buffer;
     }   
     //
     nscf() ;
     //compute_polarization() ;
  }
#endif
 computing_timer.exit_section(" pp "); 
  //
  MPI_Barrier(interpoolcomm) ;
  computing_timer.enter_section("configurational force computation"); 
  forcePtr->computeAtomsForces();
  forcePtr->printAtomsForces();
  computing_timer.exit_section("configurational force computation");  
   
}

//Output
template <unsigned int FEOrder>
void dftClass<FEOrder>::output() 
{
  //
  //only generate wave function output for serial runs
  //
  DataOut<3> data_outEigen;
  data_outEigen.attach_dof_handler (dofHandlerEigen);
  for(unsigned int i=0; i<eigenVectors[0].size(); ++i)
    {
      char buffer[100]; sprintf(buffer,"eigen%u", i);  
      data_outEigen.add_data_vector (*eigenVectors[0][i], buffer);
    }
  data_outEigen.build_patches (C_num1DQuad<FEOrder>());

  std::ofstream output ("eigen.vtu");
  //data_outEigen.write_vtu (output);
  //Doesn't work with mvapich2_ib mpi libraries
  data_outEigen.write_vtu_in_parallel(std::string("eigen.vtu").c_str(),mpi_communicator);

  //
  //write the electron-density
  //

  //
  //access quadrature rules and mapping data
  //
  QGauss<3>  quadrature_formula(FEOrder+1);
  const unsigned int n_q_points = quadrature_formula.size();
  MappingQ1<3,3> mapping;
  struct quadDensityData { double density; };

  //
  //create electron-density quadrature data using "CellDataStorage" class of dealii
  //
  CellDataStorage<typename DoFHandler<3>::active_cell_iterator,quadDensityData> rhoQuadData;
  

  rhoQuadData.initialize(dofHandler.begin_active(),
			 dofHandler.end(),
			 n_q_points);
  //
  //copy rhoValues into CellDataStorage container
  //
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();

  for(; cell!=endc; ++cell) 
    {
      if(cell->is_locally_owned())
	{
	  const std::vector<std::shared_ptr<quadDensityData> > rhoQuadPointVector = rhoQuadData.get_data(cell);
	  for(unsigned int q = 0; q < n_q_points; ++q)
	    {
	      rhoQuadPointVector[q]->density = (*rhoOutValues)[cell->id()][q];
	    }
	}
    }

  //
  //project and create a nodal field of the same mesh from the quadrature data (L2 projection from quad points to nodes)
  //

  //
  //create a new nodal field
  //
  vectorType rhoNodalField;
  matrix_free_data.initialize_dof_vector(rhoNodalField);

  VectorTools::project<3,parallel::distributed::Vector<double>>(mapping,
								dofHandler,
								constraintsNone,
								quadrature_formula,
								[&](const typename DoFHandler<3>::active_cell_iterator & cell , const unsigned int q) -> double {return rhoQuadData.get_data(cell)[q]->density;},
								rhoNodalField);

  rhoNodalField.update_ghost_values();

  //
  //only generate output for electron-density
  //
  DataOut<3> dataOutRho;
  dataOutRho.attach_dof_handler(dofHandler);
  char buffer[100]; sprintf(buffer,"rhoField");  
  dataOutRho.add_data_vector(rhoNodalField, buffer);
  dataOutRho.build_patches(C_num1DQuad<FEOrder>());
  //data_outEigen.write_vtu (output);
  //Doesn't work with mvapich2_ib mpi libraries
  dataOutRho.write_vtu_in_parallel(std::string("rhoField.vtu").c_str(),mpi_communicator);

}

template <unsigned int FEOrder>
void dftClass<FEOrder>::writeMesh(std::string meshFileName)
 {
      FESystem<3> FETemp(FE_Q<3>(QGaussLobatto<1>(2)), 1);
      DoFHandler<3> dofHandlerTemp; dofHandlerTemp.initialize(d_mesh.getSerialMesh(),FETemp);		
      dofHandlerTemp.distribute_dofs(FETemp);
      DataOut<3> data_out;
      data_out.attach_dof_handler(dofHandlerTemp);
      data_out.build_patches ();
      meshFileName+=".vtu";
      std::ofstream output(meshFileName);
      data_out.write_vtu (output);
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
