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
// @author Shiva Rudraraju (2016), Phani Motamarri (2018), Sambit Das (2018)
//

//Include header files

#include <dft.h>
#include <eigen.h>
#include <poisson.h>
#include <force.h>
#include <symmetry.h>
#include <geoOptIon.h>
#include <geoOptCell.h>
#include <meshMovementGaussian.h>
#include <meshMovementAffineTransform.h>
#include <fileReaders.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <interpolateFieldsFromPreviousMesh.h>


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
#include "rhoDataUtils.cc"
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
#endif


//
//dft constructor
//
template<unsigned int FEOrder>
dftClass<FEOrder>::dftClass(MPI_Comm &mpi_comm_replica, MPI_Comm &interpoolcomm):
  FE (FE_Q<3>(QGaussLobatto<1>(FEOrder+1)), 1),
#ifdef ENABLE_PERIODIC_BC
  FEEigen (FE_Q<3>(QGaussLobatto<1>(FEOrder+1)), 2),
#else
  FEEigen (FE_Q<3>(QGaussLobatto<1>(FEOrder+1)), 1),
#endif
  mpi_communicator (mpi_comm_replica),
  interpoolcomm (interpoolcomm),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_comm_replica)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_comm_replica)),
  numElectrons(0),
  numLevels(0),
  d_maxkPoints(1),
  integralRhoValue(0),
  d_mesh(mpi_comm_replica,interpoolcomm),
  d_affineTransformMesh(mpi_comm_replica),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times)
{
  poissonPtr= new poissonClass<FEOrder>(this, mpi_comm_replica);
  eigenPtr= new eigenClass<FEOrder>(this, mpi_comm_replica);
  forcePtr= new forceClass<FEOrder>(this, mpi_comm_replica);
  symmetryPtr= new symmetryClass<FEOrder>(this, mpi_comm_replica, interpoolcomm);
  geoOptIonPtr= new geoOptIon<FEOrder>(this, mpi_comm_replica);
#ifdef ENABLE_PERIODIC_BC
  geoOptCellPtr= new geoOptCell<FEOrder>(this, mpi_comm_replica);
#endif
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
#ifdef ENABLE_PERIODIC_BC
  delete geoOptCellPtr;
#endif
}

namespace internaldft
{

  void convertToCellCenteredCartesianCoordinates(std::vector<std::vector<double> > & atomLocations,
						 const std::vector<std::vector<double> > & latticeVectors)
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
}

template<unsigned int FEOrder>
void dftClass<FEOrder>::computeVolume()
{
  d_domainVolume=0;
  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (FE, quadrature, update_JxW_values);

  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  for (; cell!=endc; ++cell)
    {
      if (cell->is_locally_owned())
	{
	  fe_values.reinit (cell);
	  for (unsigned int q_point = 0; q_point < quadrature.size(); ++q_point)
	    {
	      d_domainVolume+=fe_values.JxW (q_point);
	    }
	}
    }
  d_domainVolume= Utilities::MPI::sum(d_domainVolume, mpi_communicator);
  pcout<< "Volume of the domain (Bohr^3): "<< d_domainVolume<<std::endl;
}

template<unsigned int FEOrder>
void dftClass<FEOrder>::set()
{
  if (dftParameters::verbosity==2)
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
    }
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
#endif

  //
  //read domain bounding Vectors
  //
  unsigned int numberColumnsLatticeVectorsFile = 3;
  dftUtils::readFile(numberColumnsLatticeVectorsFile,d_domainBoundingVectors,dftParameters::domainBoundingVectorsFile);

  pcout << "number of atoms types: " << atomTypes.size() << "\n";

  //estimate total number of wave functions
  determineOrbitalFilling();

#ifdef ENABLE_PERIODIC_BC
  if (dftParameters::isIonForce || dftParameters::isCellStress)
    AssertThrow(!dftParameters::useSymm,ExcMessage("USE GROUP SYMMETRY must be set to false if either ION FORCE or CELL STRESS is set to true. This functionality will be added in a future release"));
  //readkPointData();
  generateMPGrid();
  //if (useSymm)
  //symmetryPtr->test_spg_get_ir_reciprocal_mesh() ;
#else
  d_maxkPoints = 1;
  d_kPointCoordinates.resize(3*d_maxkPoints,0.0);
  d_kPointWeights.resize(d_maxkPoints,1.0);
#endif

  //set size of eigenvalues and eigenvectors data structures
  eigenValues.resize(d_maxkPoints);
  eigenValuesTemp.resize(d_maxkPoints);

  a0.resize((dftParameters::spinPolarized+1)*d_maxkPoints,dftParameters::lowerEndWantedSpectrum);
  bLow.resize((dftParameters::spinPolarized+1)*d_maxkPoints,0.0);
  eigenVectors.resize((1+dftParameters::spinPolarized)*d_maxkPoints);

  for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_maxkPoints; ++kPoint)
    eigenVectors[kPoint].resize(numEigenValues);

  for(unsigned int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
    {
      eigenValues[kPoint].resize((dftParameters::spinPolarized+1)*numEigenValues);
      eigenValuesTemp[kPoint].resize(numEigenValues);
    }

}

//dft pseudopotential init
template<unsigned int FEOrder>
void dftClass<FEOrder>::initPseudoPotentialAll()
{
  if(dftParameters::isPseudopotential)
    {
      TimerOutput::Scope scope (computing_timer, "psp init");
      pcout<<std::endl<<"Pseuodopotential initalization...."<<std::endl;
      initLocalPseudoPotential();
      //
      if(dftParameters::pseudoProjector == 2)
	initNonLocalPseudoPotential_OV();
      else
	initNonLocalPseudoPotential();
      //
      //
      if(dftParameters::pseudoProjector == 2)
	{
	  computeSparseStructureNonLocalProjectors_OV();
	  computeElementalOVProjectorKets();
	}
      else
	{
	  computeSparseStructureNonLocalProjectors();
	  computeElementalProjectorKets();
	}

      forcePtr->initPseudoData();
    }
}


// generate image charges and update k point cartesian coordinates based on current lattice vectors
template<unsigned int FEOrder>
void dftClass<FEOrder>::initImageChargesUpdateKPoints()
{

  pcout<<"-----------Domain bounding vectors (lattice vectors in fully periodic case)-------------"<<std::endl;
  for(int i = 0; i < d_domainBoundingVectors.size(); ++i)
    {
      pcout<<"v"<< i+1<<" : "<< d_domainBoundingVectors[i][0]<<" "<<d_domainBoundingVectors[i][1]<<" "<<d_domainBoundingVectors[i][2]<<std::endl;
    }
  pcout<<"-----------------------------------------------------------------------------------------"<<std::endl;
#ifdef ENABLE_PERIODIC_BC
  pcout<<"-----Fractional coordinates of atoms------ "<<std::endl;
  for(unsigned int i = 0; i < atomLocations.size(); ++i)
    {
      atomLocations[i] = atomLocationsFractional[i] ;
      pcout<<"AtomId "<<i <<":  "<<atomLocationsFractional[i][2]<<" "<<atomLocationsFractional[i][3]<<" "<<atomLocationsFractional[i][4]<<"\n";
    }
  pcout<<"-----------------------------------------------------------------------------------------"<<std::endl;
  generateImageCharges();

  internaldft::convertToCellCenteredCartesianCoordinates(atomLocations,
					                 d_domainBoundingVectors);
  recomputeKPointCoordinates();

  if (dftParameters::verbosity==2)
    {
      //FIXME: Print all k points across all pools
      pcout<<"-------------------k points cartesian coordinates and weights-----------------------------"<<std::endl;
      for(unsigned int i = 0; i < d_maxkPoints; ++i)
	{
	  pcout<<" ["<< d_kPointCoordinates[3*i+0] <<", "<< d_kPointCoordinates[3*i+1]<<", "<< d_kPointCoordinates[3*i+2]<<"] "<<d_kPointWeights[i]<<std::endl;
	}
      pcout<<"-----------------------------------------------------------------------------------------"<<std::endl;
    }
#else
  //
  //print cartesian coordinates
  //
  pcout<<"------------Cartesian coordinates of atoms (origin at center of domain)------------------"<<std::endl;
  for(unsigned int i = 0; i < atomLocations.size(); ++i)
    {
      pcout<<"AtomId "<<i <<":  "<<atomLocations[i][2]<<" "<<atomLocations[i][3]<<" "<<atomLocations[i][4]<<"\n";
    }
  pcout<<"-----------------------------------------------------------------------------------------"<<std::endl;
#endif
}

//dft init
template<unsigned int FEOrder>
void dftClass<FEOrder>::init (const bool usePreviousGroundStateFields)
{

  initImageChargesUpdateKPoints();

  computing_timer.enter_section("mesh generation");
  //
  //generate mesh (both parallel and serial)
  //
  d_mesh.generateSerialUnmovedAndParallelMovedUnmovedMesh(atomLocations,
				                          d_imagePositions,
				                          d_domainBoundingVectors);
  computing_timer.exit_section("mesh generation");


  //
  //get access to triangulation objects from meshGenerator class
  //
  const parallel::distributed::Triangulation<3> & triangulationPar = d_mesh.getParallelMeshMoved();

  //initialize affine transformation object (must be done on unmoved triangulation)
  d_affineTransformMesh.init(d_mesh.getParallelMeshMoved(),d_domainBoundingVectors);

  //
  //initialize dofHandlers and hanging-node constraints and periodic constraints on the unmoved Mesh
  //
  initUnmovedTriangulation(triangulationPar);
#ifdef ENABLE_PERIODIC_BC
  if (dftParameters::useSymm)
    symmetryPtr->initSymmetry() ;
#endif
  //
  //move triangulation to have atoms on triangulation vertices
  //

  moveMeshToAtoms(triangulationPar);

  //
  //initialize dirichlet BCs for total potential and vSelf poisson solutions
  //
  initBoundaryConditions();

  //
  //initialize guesses for electron-density and wavefunctions
  //
  initElectronicFields(usePreviousGroundStateFields);

  //
  //store constraintEigen Matrix entries into STL vector
  //
  constraintsNoneEigenDataInfo.initialize(vChebyshev.get_partitioner(),
					  constraintsNoneEigen);

  //
  //initialize pseudopotential data for both local and nonlocal part
  //
  initPseudoPotentialAll();
}

template<unsigned int FEOrder>
void dftClass<FEOrder>::initNoRemesh()
{
  initImageChargesUpdateKPoints();

  //
  //reinitialize dirichlet BCs for total potential and vSelf poisson solutions
  //
  initBoundaryConditions();

  //rho init (use previous ground state electron density)
  //
  noRemeshRhoDataInit();

  //
  //reinitialize pseudopotential related data structures
  //
  initPseudoPotentialAll();
}

//
// deform domain and call appropriate reinits
//
template<unsigned int FEOrder>
void dftClass<FEOrder>::deformDomain(const Tensor<2,3,double> & deformationGradient)
{
  d_affineTransformMesh.initMoved(d_domainBoundingVectors);
  d_affineTransformMesh.transform(deformationGradient);

  dftUtils::transformDomainBoundingVectors(d_domainBoundingVectors,deformationGradient);

  initNoRemesh();
}

//
//dft run
//
template<unsigned int FEOrder>
void dftClass<FEOrder>::run()
{
  solve();

  if (dftParameters::isIonOpt && !dftParameters::isCellOpt)
    {
      geoOptIonPtr->init();
      geoOptIonPtr->run();
    }
  else if (!dftParameters::isIonOpt && dftParameters::isCellOpt)
    {
#ifdef ENABLE_PERIODIC_BC
      geoOptCellPtr->init();
      geoOptCellPtr->run();
#else
      AssertThrow(false,ExcMessage("CELL OPT cannot be set to true for fully non-periodic domain."));
#endif
    }
  else if (dftParameters::isIonOpt && dftParameters::isCellOpt)
    {
#ifdef ENABLE_PERIODIC_BC
      //first relax ion positions in the starting cell configuration
      geoOptIonPtr->init();
      geoOptIonPtr->run();

      //start cell relaxation, where for each cell relaxation update the ion positions are again relaxed
      geoOptCellPtr->init();
      geoOptCellPtr->run();
#else
      AssertThrow(false,ExcMessage("CELL OPT cannot be set to true for fully non-periodic domain."));
#endif
    }
}

//
//dft solve
//
template<unsigned int FEOrder>
void dftClass<FEOrder>::solve()
{

  //
  //solve vself
  //
  computing_timer.enter_section("vself solve");
  solveVself();
  computing_timer.exit_section("vself solve");

  //
  //solve
  //
  computing_timer.enter_section("scf solve");


  //
  //Begin SCF iteration
  //
  unsigned int scfIter=0;
  double norm = 1.0;


  pcout<<std::endl;
  if (dftParameters::verbosity==0)
    pcout<<"Starting SCF iteration...."<<std::endl;
  while ((norm > dftParameters::selfConsistentSolverTolerance) && (scfIter < dftParameters::numSCFIterations))
    {
      if (dftParameters::verbosity>=1)
        pcout<<"************************Begin Self-Consistent-Field Iteration: "<<std::setw(2)<<scfIter+1<<" ***********************"<<std::endl;
      //
      //Mixing scheme
      //
      if(scfIter > 0)
	{
	  if (scfIter==1)
	    {
	      if (dftParameters::spinPolarized==1)
		{
		  norm = mixing_simple_spinPolarized();
		}
	      else
		norm = mixing_simple();
	    }
	  else
	    {
	      if (dftParameters::spinPolarized==1)
		{
		  norm = sqrt(mixing_anderson_spinPolarized());
		}
	      else
		norm = sqrt(mixing_anderson());
	    }

	  if (dftParameters::verbosity>=1)
	    pcout<<"Anderson Mixing: L2 norm of electron-density difference: "<< norm<< std::endl;

	  poissonPtr->phiTotRhoIn = poissonPtr->phiTotRhoOut;
	}

      //
      //phiTot with rhoIn
      //
      int constraintMatrixId = phiTotDofHandlerIndex;
      if (dftParameters::verbosity==2)
        pcout<< std::endl<<"Poisson solve for total electrostatic potential (rhoIn+b): ";
      computing_timer.enter_section("phiTot solve");
      poissonPtr->solve(poissonPtr->phiTotRhoIn,constraintMatrixId, rhoInValues);
      computing_timer.exit_section("phiTot solve");

      //
      //eigen solve
      //
      if (dftParameters::spinPolarized==1)
	{
	  for(unsigned int s=0; s<2; ++s)
	    {
	      if(dftParameters::xc_id < 4)
	        {
		  eigenPtr->computeVEffSpinPolarized(rhoInValuesSpinPolarized, poissonPtr->phiTotRhoIn, poissonPtr->phiExt, s, pseudoValues);
                }
	      else if (dftParameters::xc_id == 4)
	        {
		  eigenPtr->computeVEffSpinPolarized(rhoInValuesSpinPolarized, gradRhoInValuesSpinPolarized, poissonPtr->phiTotRhoIn, poissonPtr->phiExt, s, pseudoValues);
	        }
	      for (int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
	        {
	          d_kPointIndex = kPoint;
	          for(int j = 0; j < dftParameters::numPass; ++j)
	            {
		      if (dftParameters::verbosity==2)
			pcout<<"Beginning Chebyshev filter pass "<< j+1<< " for spin "<< s+1<<std::endl;

		      chebyshevSolver(s);
	            }
	        }
	    }

	  //
	  //fermi energy
	  //
          compute_fermienergy();
        }
      else
        {
	  if(dftParameters::xc_id < 4)
	    {
	      eigenPtr->computeVEff(rhoInValues, poissonPtr->phiTotRhoIn, poissonPtr->phiExt, pseudoValues);
	    }
	  else if (dftParameters::xc_id == 4)
	    {
	      eigenPtr->computeVEff(rhoInValues, gradRhoInValues, poissonPtr->phiTotRhoIn, poissonPtr->phiExt, pseudoValues);
	    }

	  for (int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
	    {
	      d_kPointIndex = kPoint;
	      for(int j = 0; j < dftParameters::numPass; ++j)
		{
		  if (dftParameters::verbosity==2)
		    pcout<< "Beginning Chebyshev filter pass "<< j+1<<std::endl;

		  chebyshevSolver(0);
		}
	    }

	  //
	  //fermi energy
	  //
	  compute_fermienergy();

	  //
	  //maximum of the residual norm of the state closest to and below the Fermi level among all k points
	  //
	  double maxRes = computeMaximumHighestOccupiedStateResidualNorm();
	  if (dftParameters::verbosity==2)
	    pcout << "Maximum residual norm of the state closest to and below Fermi level: "<< maxRes << std::endl;

	  //if the residual norm is greater than 1e-1 (heuristic)
	  // do more passes of chebysev filter till the check passes.
	  // This improves the scf convergence performance. Currently this
	  // approach is not implemented for spin-polarization case
	  int count=1;
	  while (maxRes>1e-1)
	    {
	      for (int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
		{
		  d_kPointIndex = kPoint;
		  if (dftParameters::verbosity==2)
		    pcout<< "Beginning Chebyshev filter pass "<< dftParameters::numPass+count<<std::endl;

		  chebyshevSolver(0);
		}
	      count++;
	      compute_fermienergy();
	      maxRes = computeMaximumHighestOccupiedStateResidualNorm();
	      if (dftParameters::verbosity==2)
	        pcout << "Maximum residual norm of the state closest to and below Fermi level: "<< maxRes << std::endl;
	    }

	}
      computing_timer.enter_section("compute rho");
#ifdef ENABLE_PERIODIC_BC
      if(dftParameters::useSymm){
	symmetryPtr->computeLocalrhoOut();
	symmetryPtr->computeAndSymmetrize_rhoOut();
      }
      else
	compute_rhoOut();
#else
      compute_rhoOut();
#endif
      computing_timer.exit_section("compute rho");

      //
      //compute integral rhoOut
      //
      integralRhoValue=totalCharge(rhoOutValues);

      //
      //phiTot with rhoOut
      //
      if(dftParameters::verbosity==2)
	pcout<< std::endl<<"Poisson solve for total electrostatic potential (rhoOut+b): ";

      computing_timer.enter_section("phiTot solve");
      poissonPtr->solve(poissonPtr->phiTotRhoOut,constraintMatrixId, rhoOutValues);
      computing_timer.exit_section("phiTot solve");


      const double totalEnergy = dftParameters::spinPolarized==1 ?
	compute_energy_spinPolarized(dftParameters::verbosity==2) :
	compute_energy(dftParameters::verbosity==2);
      if (dftParameters::verbosity==1)
	{
	  pcout<<"Total energy  : " << totalEnergy << std::endl;
	}

      if (dftParameters::verbosity>=1)
        pcout<<"***********************Self-Consistent-Field Iteration: "<<std::setw(2)<<scfIter+1<<" complete**********************"<<std::endl<<std::endl;

      //output wave functions
      //output();

      //
      scfIter++;
    }

  if(scfIter==dftParameters::numSCFIterations)
    pcout<< "SCF iteration did not converge to the specified tolerance after: "<<scfIter<<" iterations."<<std::endl;
  else
    pcout<< "SCF iteration converged to the specified tolerance after: "<<scfIter<<" iterations."<<std::endl;

  //
  // compute and print ground state energy or energy after max scf iterations
  //
  if (dftParameters::spinPolarized==1)
    compute_energy_spinPolarized(true);
  else
    compute_energy (true);

  computing_timer.exit_section("scf solve");

  MPI_Barrier(interpoolcomm) ;
  if (dftParameters::isIonForce)
    {
      computing_timer.enter_section("ion force");
      forcePtr->computeAtomsForces();
      forcePtr->printAtomsForces();
      computing_timer.exit_section("ion force");
    }
#ifdef ENABLE_PERIODIC_BC
  if (dftParameters::isCellStress)
    {
      computing_timer.enter_section("cell stress");
      forcePtr->computeStress();
      forcePtr->printStress();
      computing_timer.exit_section("cell stress");
    }
#endif
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
      data_outEigen.add_data_vector (eigenVectors[0][i], buffer);
    }
  data_outEigen.build_patches (C_num1DQuad<FEOrder>());

  std::ofstream output ("eigen.vtu");
  //data_outEigen.write_vtu (output);
  //Doesn't work with mvapich2_ib mpi libraries
  data_outEigen.write_vtu_in_parallel(std::string("eigen.vtu").c_str(),mpi_communicator);

  //
  //write the electron-density
  //
  computeGroundStateRhoNodalField();

  //
  //only generate output for electron-density
  //
  DataOut<3> dataOutRho;
  dataOutRho.attach_dof_handler(dofHandler);
  char buffer[100]; sprintf(buffer,"rhoField");
  dataOutRho.add_data_vector(d_rhoNodalFieldGroundState, buffer);
  dataOutRho.build_patches(C_num1DQuad<FEOrder>());
  //data_outEigen.write_vtu (output);
  //Doesn't work with mvapich2_ib mpi libraries
  dataOutRho.write_vtu_in_parallel(std::string("rhoField.vtu").c_str(),mpi_communicator);

}

template <unsigned int FEOrder>
void dftClass<FEOrder>::writeMesh(std::string meshFileName)
{
  DataOut<3> data_out;
  data_out.attach_dof_handler(dofHandler);
  data_out.build_patches ();
  meshFileName+=".vtu";
  data_out.write_vtu_in_parallel(meshFileName.c_str(),mpi_communicator);
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
