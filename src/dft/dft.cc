// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das
//

//Include header files
#include <dft.h>
#include <eigen.h>
#include <force.h>
#include <poissonSolverProblem.h>
#include <dealiiLinearSolver.h>
#include <energyCalculator.h>
#include <symmetry.h>
#include <geoOptIon.h>
#include <geoOptCell.h>
#include <meshMovementGaussian.h>
#include <meshMovementAffineTransform.h>
#include <fileReaders.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <chebyshevOrthogonalizedSubspaceIterationSolver.h>
#include <complex>
#include <cmath>
#include <algorithm>
#include "linalg.h"
#include "stdafx.h"
#include <fstream>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/normal_distribution.hpp>
#include <interpolateFieldsFromPreviousMesh.h>
#include <linearAlgebraOperations.h>
#include <vectorUtilities.h>


namespace dftfe {

  //Include cc files
#include "pseudoUtils.cc"
#include "moveMeshToAtoms.cc"
#include "initUnmovedTriangulation.cc"
#include "initBoundaryConditions.cc"
#include "initElectronicFields.cc"
#include "initPseudo.cc"
#include "initPseudo-OV.cc"
#include "initRho.cc"
#include "publicMethods.cc"
#include "generateImageCharges.cc"
#include "psiInitialGuess.cc"
#include "fermiEnergy.cc"
#include "charge.cc"
#include "density.cc"
#include "mixingschemes.cc"
#include "kohnShamEigenSolve.cc"
#include "restart.cc"
//#include "electrostaticPRefinedEnergy.cc"
#include "moveAtoms.cc"

  //
  //dft constructor
  //
  template<unsigned int FEOrder>
  dftClass<FEOrder>::dftClass(const MPI_Comm &mpi_comm_replica,const MPI_Comm &_interpoolcomm):
    FE (FE_Q<3>(QGaussLobatto<1>(FEOrder+1)), 1),
#ifdef USE_COMPLEX
    FEEigen (FE_Q<3>(QGaussLobatto<1>(FEOrder+1)), 2),
#else
    FEEigen (FE_Q<3>(QGaussLobatto<1>(FEOrder+1)), 1),
#endif
    mpi_communicator (mpi_comm_replica),
    interpoolcomm (_interpoolcomm),
    n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_comm_replica)),
    this_mpi_process (Utilities::MPI::this_mpi_process(mpi_comm_replica)),
    numElectrons(0),
    numLevels(0),
    d_mesh(mpi_comm_replica,_interpoolcomm),
    d_affineTransformMesh(mpi_comm_replica),
    d_gaussianMovePar(mpi_comm_replica),
    d_vselfBinsManager(mpi_comm_replica),
    pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
    computing_timer (pcout,
		     dftParameters::reproducible_output ? TimerOutput::never : TimerOutput::summary,
		     TimerOutput::wall_times)
  {
    forcePtr= new forceClass<FEOrder>(this, mpi_comm_replica);
    symmetryPtr= new symmetryClass<FEOrder>(this, mpi_comm_replica, _interpoolcomm);
    geoOptIonPtr= new geoOptIon<FEOrder>(this, mpi_comm_replica);

#ifdef USE_COMPLEX
    geoOptCellPtr= new geoOptCell<FEOrder>(this, mpi_comm_replica);
#endif
  }

  template<unsigned int FEOrder>
  dftClass<FEOrder>::~dftClass()
  {
    delete symmetryPtr;
    matrix_free_data.clear();
    delete forcePtr;
    delete geoOptIonPtr;
#ifdef USE_COMPLEX
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
  double dftClass<FEOrder>::computeVolume(const dealii::DoFHandler<3> & _dofHandler)
  {
    double domainVolume=0;
    QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
    FEValues<3> fe_values (_dofHandler.get_fe(), quadrature, update_JxW_values);

    typename DoFHandler<3>::active_cell_iterator cell = _dofHandler.begin_active(), endc = _dofHandler.end();
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
	{
	  fe_values.reinit (cell);
	  for (unsigned int q_point = 0; q_point < quadrature.size(); ++q_point)
	    domainVolume+=fe_values.JxW (q_point);
	}

    domainVolume= Utilities::MPI::sum(domainVolume, mpi_communicator);
    if (dftParameters::verbosity>=1)
      pcout<< "Volume of the domain (Bohr^3): "<< domainVolume<<std::endl;
    return domainVolume;
  }

  template<unsigned int FEOrder>
  void dftClass<FEOrder>::set()
  {
    if (dftParameters::verbosity>=2)
      pcout << std::endl << "number of MPI processes: "
	    << Utilities::MPI::n_mpi_processes(mpi_communicator)
	    << std::endl;
    //
    //read coordinates
    //
    unsigned int numberColumnsCoordinatesFile = 5;

    if (dftParameters::periodicX || dftParameters::periodicY || dftParameters::periodicZ)
      {
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
      }
    else
      {
	dftUtils::readFile(numberColumnsCoordinatesFile, atomLocations, dftParameters::coordinatesFile);
	pcout << "number of atoms: " << atomLocations.size() << "\n";

	//
	//find unique atom types
	//
	for (std::vector<std::vector<double> >::iterator it=atomLocations.begin(); it<atomLocations.end(); it++)
	  {
	    atomTypes.insert((unsigned int)((*it)[0]));
	  }
      }

    //
    //read domain bounding Vectors
    //
    unsigned int numberColumnsLatticeVectorsFile = 3;
    dftUtils::readFile(numberColumnsLatticeVectorsFile,d_domainBoundingVectors,dftParameters::domainBoundingVectorsFile);

    pcout << "number of atoms types: " << atomTypes.size() << "\n";

    //determine number of electrons
    for(unsigned int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
    {
      const unsigned int Z = atomLocations[iAtom][0];
      const unsigned int valenceZ = atomLocations[iAtom][1];

      if(dftParameters::isPseudopotential)
	  numElectrons += valenceZ;
      else
	  numElectrons += Z;
    }

    //estimate total number of wave functions from atomic orbital filling
    if (dftParameters::startingWFCType=="ATOMIC")
      determineOrbitalFilling();

#ifdef USE_COMPLEX
    generateMPGrid();
#else
    d_kPointCoordinates.resize(3,0.0);
    d_kPointWeights.resize(1,1.0);
#endif

    //set size of eigenvalues and eigenvectors data structures
    eigenValues.resize(d_kPointWeights.size());

    a0.resize((dftParameters::spinPolarized+1)*d_kPointWeights.size(),dftParameters::lowerEndWantedSpectrum);
    bLow.resize((dftParameters::spinPolarized+1)*d_kPointWeights.size(),0.0);
    d_eigenVectorsFlattened.resize((1+dftParameters::spinPolarized)*d_kPointWeights.size());

    for(unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
        eigenValues[kPoint].resize((dftParameters::spinPolarized+1)*numEigenValues);
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

    if (dftParameters::periodicX || dftParameters::periodicY || dftParameters::periodicZ)
      {
	pcout<<"-----Fractional coordinates of atoms------ "<<std::endl;
	for(unsigned int i = 0; i < atomLocations.size(); ++i)
	  {
	    atomLocations[i] = atomLocationsFractional[i] ;
	    pcout<<"AtomId "<<i <<":  "<<atomLocationsFractional[i][2]<<" "<<atomLocationsFractional[i][3]<<" "<<atomLocationsFractional[i][4]<<"\n";
	  }
	pcout<<"-----------------------------------------------------------------------------------------"<<std::endl;
	//sanity check on fractional coordinates
	std::vector<bool> periodicBc(3,false);
	periodicBc[0]=dftParameters::periodicX;periodicBc[1]=dftParameters::periodicY;periodicBc[2]=dftParameters::periodicZ;
        const double tol=1e-6;
  	for(unsigned int i = 0; i < atomLocationsFractional.size(); ++i)
	  {
	    for(unsigned int idim = 0; idim < 3; ++idim)
	      if (periodicBc[idim])
	        AssertThrow(atomLocationsFractional[i][2+idim]>-tol && atomLocationsFractional[i][2+idim]<1.0+tol,ExcMessage("DFT-FE Error: periodic direction fractional coordinates doesn't lie in [0,1]. Please check input fractional coordinates, or if this is an ionic relaxation step, please check the corresponding algorithm."));
	  }

	generateImageCharges();

	internaldft::convertToCellCenteredCartesianCoordinates(atomLocations,
							       d_domainBoundingVectors);
#ifdef USE_COMPLEX
	recomputeKPointCoordinates();
#endif
	if (dftParameters::verbosity>=2)
	  {
	    //FIXME: Print all k points across all pools
	    pcout<<"-------------------k points cartesian coordinates and weights-----------------------------"<<std::endl;
	    for(unsigned int i = 0; i < d_kPointWeights.size(); ++i)
	      {
		pcout<<" ["<< d_kPointCoordinates[3*i+0] <<", "<< d_kPointCoordinates[3*i+1]<<", "<< d_kPointCoordinates[3*i+2]<<"] "<<d_kPointWeights[i]<<std::endl;
	      }
	    pcout<<"-----------------------------------------------------------------------------------------"<<std::endl;
	  }
      }
    else
      {
	//
	//print cartesian coordinates
	//
	pcout<<"------------Cartesian coordinates of atoms (origin at center of domain)------------------"<<std::endl;
	for(unsigned int i = 0; i < atomLocations.size(); ++i)
	  {
	    pcout<<"AtomId "<<i <<":  "<<atomLocations[i][2]<<" "<<atomLocations[i][3]<<" "<<atomLocations[i][4]<<"\n";
	  }
	pcout<<"-----------------------------------------------------------------------------------------"<<std::endl;
      }
  }

  //dft init
  template<unsigned int FEOrder>
  void dftClass<FEOrder>::init (const unsigned int usePreviousGroundStateFields)
  {

    initImageChargesUpdateKPoints();

    computing_timer.enter_section("mesh generation");
    //
    //generate mesh (both parallel and serial)
    //
    if (dftParameters::chkType==2 && dftParameters::restartFromChk)
      {
	d_mesh.generateCoarseMeshesForRestart(atomLocations,
					      d_imagePositions,
					      d_domainBoundingVectors,
					      dftParameters::useSymm);
	loadTriaInfoAndRhoData();
      }
    else
      d_mesh.generateSerialUnmovedAndParallelMovedUnmovedMesh(atomLocations,
							      d_imagePositions,
							      d_domainBoundingVectors,
							      dftParameters::useSymm);
    computing_timer.exit_section("mesh generation");


    //
    //get access to triangulation objects from meshGenerator class
    //
    const parallel::distributed::Triangulation<3> & triangulationPar = d_mesh.getParallelMeshMoved();

    //
    //initialize dofHandlers and hanging-node constraints and periodic constraints on the unmoved Mesh
    //
    initUnmovedTriangulation(triangulationPar);
#ifdef USE_COMPLEX
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
#ifdef USE_COMPLEX
	geoOptCellPtr->init();
	geoOptCellPtr->run();
#else
	AssertThrow(false,ExcMessage("CELL OPT cannot be set to true for fully non-periodic domain."));
#endif
      }
    else if (dftParameters::isIonOpt && dftParameters::isCellOpt)
      {
#ifdef USE_COMPLEX
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
    //solve vself in bins
    //
    computing_timer.enter_section("vself solve");


    d_vselfBinsManager.solveVselfInBins(matrix_free_data,
					2,
					d_phiExt,
					d_noConstraints,
					d_localVselfs);

    computing_timer.exit_section("vself solve");

    energyCalculator energyCalc(mpi_communicator, interpoolcomm);



    //set up poisson solver
    dealiiLinearSolver dealiiCGSolver(mpi_communicator, dealiiLinearSolver::CG);
    poissonSolverProblem<FEOrder> phiTotalSolverProblem(mpi_communicator);


    //
    //create eigenClass object
    //
    eigenClass<FEOrder> kohnShamDFTEigenOperator(this,mpi_communicator);
    kohnShamDFTEigenOperator.init();


    //
    //create eigen solver object
    //
    chebyshevOrthogonalizedSubspaceIterationSolver subspaceIterationSolver(dftParameters::lowerEndWantedSpectrum,
									   0.0);


    //
    //precompute shapeFunctions and shapeFunctionGradients and shapeFunctionGradientIntegrals
    //
    computing_timer.enter_section("shapefunction data");
    kohnShamDFTEigenOperator.preComputeShapeFunctionGradientIntegrals();
    computing_timer.exit_section("shapefunction data");


    //
    //solve
    //
    computing_timer.enter_section("scf solve");


    //
    //Begin SCF iteration
    //
    unsigned int scfIter=0;
    double norm = 1.0;
    //CAUTION: Choosing a looser tolerance might lead to failed tests
    const double adaptiveChebysevFilterPassesTol = dftParameters::chebyshevTolerance;


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
	if(scfIter > 0 && !(dftParameters::restartFromChk && dftParameters::chkType==2))
	  {
	    if (scfIter==1)
	      {
		if (dftParameters::spinPolarized==1)
		  {
		    norm = mixing_simple_spinPolarized();
		  }
		else
		  norm = mixing_simple();

		if (dftParameters::verbosity>=1)
		  pcout<<"Simple mixing: L2 norm of electron-density difference: "<< norm<< std::endl;
	      }
	    else
	      {
		if (dftParameters::spinPolarized==1)
		  {
		    norm = sqrt(mixing_anderson_spinPolarized());
		  }
		else
		  norm = sqrt(mixing_anderson());

		if (dftParameters::verbosity>=1)
		  pcout<<"Anderson mixing: L2 norm of electron-density difference: "<< norm<< std::endl;
	      }

	    d_phiTotRhoIn = d_phiTotRhoOut;
	  }
	else if (dftParameters::restartFromChk && dftParameters::chkType==2)
	  {
	    if (dftParameters::spinPolarized==1)
	      {
		norm = sqrt(mixing_anderson_spinPolarized());
	      }
	    else
	      norm = sqrt(mixing_anderson());

	    if (dftParameters::verbosity>=1)
	      pcout<<"Anderson Mixing: L2 norm of electron-density difference: "<< norm<< std::endl;

	    d_phiTotRhoIn = d_phiTotRhoOut;
	  }

	//
	//phiTot with rhoIn
	//
	if (dftParameters::verbosity>=2)
	  pcout<< std::endl<<"Poisson solve for total electrostatic potential (rhoIn+b): ";
	computing_timer.enter_section("phiTot solve");

	if (scfIter>0)
	  phiTotalSolverProblem.reinit(matrix_free_data,
				       d_phiTotRhoIn,
				       *d_constraintsVector[phiTotDofHandlerIndex],
				       phiTotDofHandlerIndex,
				       d_atomNodeIdToChargeMap,
				       *rhoInValues,
				       false);
	else
	  phiTotalSolverProblem.reinit(matrix_free_data,
				       d_phiTotRhoIn,
				       *d_constraintsVector[phiTotDofHandlerIndex],
				       phiTotDofHandlerIndex,
				       d_atomNodeIdToChargeMap,
				       *rhoInValues);

	dealiiCGSolver.solve(phiTotalSolverProblem,

			     dftParameters::relLinearSolverTolerance,
			     dftParameters::maxLinearSolverIterations,
			     dftParameters::verbosity);

	computing_timer.exit_section("phiTot solve");





	//
	//eigen solve
	//
	if (dftParameters::spinPolarized==1)
	  {

	    std::vector<std::vector<std::vector<double> > > eigenValuesSpins(2,
									     std::vector<std::vector<double> >(d_kPointWeights.size(),
													       std::vector<double>(numEigenValues)));

	    std::vector<std::vector<std::vector<double>>> residualNormWaveFunctionsAllkPointsSpins(2,
												   std::vector<std::vector<double> >(d_kPointWeights.size(),
																     std::vector<double>(numEigenValues)));

	    for(unsigned int s=0; s<2; ++s)
	      {
		if(dftParameters::xc_id < 4)
		  {
		    computing_timer.enter_section("VEff Computation");
		    kohnShamDFTEigenOperator.computeVEffSpinPolarized(rhoInValuesSpinPolarized, d_phiTotRhoIn, d_phiExt, s, pseudoValues);
		    computing_timer.exit_section("VEff Computation");
		  }
		else if (dftParameters::xc_id == 4)
		  {
		    computing_timer.enter_section("VEff Computation");
		    kohnShamDFTEigenOperator.computeVEffSpinPolarized(rhoInValuesSpinPolarized, gradRhoInValuesSpinPolarized, d_phiTotRhoIn, d_phiExt, s, pseudoValues);
		    computing_timer.exit_section("VEff Computation");
		  }
		for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
		  {
		    kohnShamDFTEigenOperator.reinitkPointIndex(kPoint);


		    computing_timer.enter_section("Hamiltonian Matrix Computation");
		    kohnShamDFTEigenOperator.computeHamiltonianMatrix(kPoint);
		    computing_timer.exit_section("Hamiltonian Matrix Computation");


		    for(unsigned int j = 0; j < dftParameters::numPass; ++j)
		      {
			if (dftParameters::verbosity>=2)
			  pcout<<"Beginning Chebyshev filter pass "<< j+1<< " for spin "<< s+1<<std::endl;

			kohnShamEigenSpaceCompute(s,
						  kPoint,
						  kohnShamDFTEigenOperator,
						  subspaceIterationSolver,
						  residualNormWaveFunctionsAllkPointsSpins[s][kPoint]);
		      }
		  }
	      }

	    for(unsigned int s=0; s<2; ++s)
	      for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
		for (unsigned int i = 0; i<numEigenValues; ++i)
		  eigenValuesSpins[s][kPoint][i]=eigenValues[kPoint][numEigenValues*s+i];
	    //
	    //fermi energy
	    //
	    compute_fermienergy();

	    //maximum of the residual norm of the state closest to and below the Fermi level among all k points,
	    //and also the maximum between the two spins
	    double maxRes =std::max(computeMaximumHighestOccupiedStateResidualNorm
				    (residualNormWaveFunctionsAllkPointsSpins[0],
				     eigenValuesSpins[0],
				     fermiEnergy),
				    computeMaximumHighestOccupiedStateResidualNorm
				    (residualNormWaveFunctionsAllkPointsSpins[1],
				     eigenValuesSpins[1],
				     fermiEnergy));

	    if (dftParameters::verbosity>=2)
	      pcout << "Maximum residual norm of the state closest to and below Fermi level: "<< maxRes << std::endl;

	    //if the residual norm is greater than adaptiveChebysevFilterPassesTol (a heuristic value)
	    // do more passes of chebysev filter till the check passes.
	    // This improves the scf convergence performance.
	    unsigned int count=1;
	    const double filterPassTol=(scfIter==0
		                       && dftParameters::restartFromChk
				       && dftParameters::chkType==2)? 1.0e-4
		                       :adaptiveChebysevFilterPassesTol;
	    while (maxRes>filterPassTol && count<20)
	      {
		for(unsigned int s=0; s<2; ++s)
		  {
		    if(dftParameters::xc_id < 4)
		      {
			computing_timer.enter_section("VEff Computation");
			kohnShamDFTEigenOperator.computeVEffSpinPolarized(rhoInValuesSpinPolarized, d_phiTotRhoIn, d_phiExt, s, pseudoValues);
			computing_timer.exit_section("VEff Computation");
		      }
		    else if (dftParameters::xc_id == 4)
		      {
			computing_timer.enter_section("VEff Computation");
			kohnShamDFTEigenOperator.computeVEffSpinPolarized(rhoInValuesSpinPolarized, gradRhoInValuesSpinPolarized, d_phiTotRhoIn, d_phiExt, s, pseudoValues);
			computing_timer.exit_section("VEff Computation");
		      }

		    for(unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
		      {
			kohnShamDFTEigenOperator.reinitkPointIndex(kPoint);
			if (dftParameters::verbosity>=2)
			  pcout<< "Beginning Chebyshev filter pass "<< dftParameters::numPass+count<< " for spin "<< s+1<<std::endl;;

			computing_timer.enter_section("Hamiltonian Matrix Computation");
			kohnShamDFTEigenOperator.computeHamiltonianMatrix(kPoint);
			computing_timer.exit_section("Hamiltonian Matrix Computation");

			kohnShamEigenSpaceCompute(s,
						  kPoint,
						  kohnShamDFTEigenOperator,
						  subspaceIterationSolver,
						  residualNormWaveFunctionsAllkPointsSpins[s][kPoint]);

		      }
		  }
		count++;
		for(unsigned int s=0; s<2; ++s)
		  for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
		    for (unsigned int i = 0; i<numEigenValues; ++i)
		      eigenValuesSpins[s][kPoint][i]=eigenValues[kPoint][numEigenValues*s+i];

		compute_fermienergy();
		maxRes =std::max(computeMaximumHighestOccupiedStateResidualNorm
				 (residualNormWaveFunctionsAllkPointsSpins[0],
				  eigenValuesSpins[0],
				  fermiEnergy),
				 computeMaximumHighestOccupiedStateResidualNorm
				 (residualNormWaveFunctionsAllkPointsSpins[1],
				  eigenValuesSpins[1],
				  fermiEnergy));
		if (dftParameters::verbosity>=2)
		  pcout << "Maximum residual norm of the state closest to and below Fermi level: "<< maxRes << std::endl;
	      }
	  }
	else
	  {

	    std::vector<std::vector<double>> residualNormWaveFunctionsAllkPoints;
	    residualNormWaveFunctionsAllkPoints.resize(d_kPointWeights.size());
	    for(unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
	      residualNormWaveFunctionsAllkPoints[kPoint].resize(numEigenValues);

	    if(dftParameters::xc_id < 4)
	      {
		computing_timer.enter_section("VEff Computation");
		kohnShamDFTEigenOperator.computeVEff(rhoInValues, d_phiTotRhoIn, d_phiExt, pseudoValues);
		computing_timer.exit_section("VEff Computation");
	      }
	    else if (dftParameters::xc_id == 4)
	      {
		computing_timer.enter_section("VEff Computation");
		kohnShamDFTEigenOperator.computeVEff(rhoInValues, gradRhoInValues, d_phiTotRhoIn, d_phiExt, pseudoValues);
		computing_timer.exit_section("VEff Computation");
	      }

	    for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
	      {
		kohnShamDFTEigenOperator.reinitkPointIndex(kPoint);

		computing_timer.enter_section("Hamiltonian Matrix Computation");
		kohnShamDFTEigenOperator.computeHamiltonianMatrix(kPoint);
		computing_timer.exit_section("Hamiltonian Matrix Computation");

		for(unsigned int j = 0; j < dftParameters::numPass; ++j)
		  {
		    if (dftParameters::verbosity>=2)
		      pcout<< "Beginning Chebyshev filter pass "<< j+1<<std::endl;


		    kohnShamEigenSpaceCompute(0,
					      kPoint,
					      kohnShamDFTEigenOperator,
					      subspaceIterationSolver,
					      residualNormWaveFunctionsAllkPoints[kPoint]);

		  }
	      }

	    //
	    //fermi energy
	    //
	    compute_fermienergy();

	    //
	    //maximum of the residual norm of the state closest to and below the Fermi level among all k points
	    //
	    double maxRes = computeMaximumHighestOccupiedStateResidualNorm
	      (residualNormWaveFunctionsAllkPoints,
	       eigenValues,
	       fermiEnergy);
	    if (dftParameters::verbosity>=2)
	      pcout << "Maximum residual norm of the state closest to and below Fermi level: "<< maxRes << std::endl;

	    //if the residual norm is greater than adaptiveChebysevFilterPassesTol (a heuristic value)
	    // do more passes of chebysev filter till the check passes.
	    // This improves the scf convergence performance.
	    unsigned int count=1;
	    const double filterPassTol=(scfIter==0
		                       && dftParameters::restartFromChk
				       && dftParameters::chkType==2)? 1.0e-4
		                       :adaptiveChebysevFilterPassesTol;
	    while (maxRes>filterPassTol && count<20)
	      {

		for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
		  {
		    kohnShamDFTEigenOperator.reinitkPointIndex(kPoint);
		    if (dftParameters::verbosity>=2)
		      pcout<< "Beginning Chebyshev filter pass "<< dftParameters::numPass+count<<std::endl;

		    computing_timer.enter_section("Hamiltonian Matrix Computation");
		    kohnShamDFTEigenOperator.computeHamiltonianMatrix(kPoint);
		    computing_timer.exit_section("Hamiltonian Matrix Computation");

		    kohnShamEigenSpaceCompute(0,
					      kPoint,
					      kohnShamDFTEigenOperator,
					      subspaceIterationSolver,
					      residualNormWaveFunctionsAllkPoints[kPoint]);
		  }
		count++;
		compute_fermienergy();
		maxRes = computeMaximumHighestOccupiedStateResidualNorm
		  (residualNormWaveFunctionsAllkPoints,
		   eigenValues,
		   fermiEnergy);
		if (dftParameters::verbosity>=2)
		  pcout << "Maximum residual norm of the state closest to and below Fermi level: "<< maxRes << std::endl;
	      }

	  }
	computing_timer.enter_section("compute rho");
#ifdef USE_COMPLEX
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
	const double integralRhoValue=totalCharge(rhoOutValues);

	if (dftParameters::verbosity>=2){
	  pcout<< std::endl<<"number of electrons: "<< integralRhoValue<<std::endl;
	  if (dftParameters::spinPolarized==1)
		pcout<< std::endl<<"net magnetization: "<< totalMagnetization(rhoOutValuesSpinPolarized) << std::endl;
	}
	//
	//phiTot with rhoOut
	//
	if(dftParameters::verbosity>=2)
	  pcout<< std::endl<<"Poisson solve for total electrostatic potential (rhoOut+b): ";

	computing_timer.enter_section("phiTot solve");


	phiTotalSolverProblem.reinit(matrix_free_data,
				     d_phiTotRhoOut,
				     *d_constraintsVector[phiTotDofHandlerIndex],
				     phiTotDofHandlerIndex,
				     d_atomNodeIdToChargeMap,
				     *rhoOutValues,
				     false);


	dealiiCGSolver.solve(phiTotalSolverProblem,
			     dftParameters::relLinearSolverTolerance,
			     dftParameters::maxLinearSolverIterations,
			     dftParameters::verbosity);

	computing_timer.exit_section("phiTot solve");

	QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
	const double totalEnergy = dftParameters::spinPolarized==0 ?
	  energyCalc.computeEnergy(dofHandler,
				   dofHandler,
				   quadrature,
				   quadrature,
				   eigenValues,
				   d_kPointWeights,
				   fermiEnergy,
				   funcX,
				   funcC,
				   d_phiTotRhoIn,
				   d_phiTotRhoOut,
				   *rhoInValues,
				   *rhoOutValues,
				   *rhoOutValues,
				   *gradRhoInValues,
				   *gradRhoOutValues,
				   d_localVselfs,
				   d_atomNodeIdToChargeMap,
				   atomLocations.size(),
				   dftParameters::verbosity>=2) :
	  energyCalc.computeEnergySpinPolarized(dofHandler,
						dofHandler,
						quadrature,
						quadrature,
						eigenValues,
						d_kPointWeights,
						fermiEnergy,
						funcX,
						funcC,
						d_phiTotRhoIn,
						d_phiTotRhoOut,
						*rhoInValues,
						*rhoOutValues,
						*rhoOutValues,
						*gradRhoInValues,
						*gradRhoOutValues,
						*rhoInValuesSpinPolarized,
						*rhoOutValuesSpinPolarized,
						*gradRhoInValuesSpinPolarized,
						*gradRhoOutValuesSpinPolarized,
						d_localVselfs,
						d_atomNodeIdToChargeMap,
						atomLocations.size(),
						dftParameters::verbosity>=2);
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

	if (dftParameters::chkType==2)
	  saveTriaInfoAndRhoData();
      }

    if(scfIter==dftParameters::numSCFIterations)
      pcout<< "SCF iteration did not converge to the specified tolerance after: "<<scfIter<<" iterations."<<std::endl;
    else
      pcout<< "SCF iteration converged to the specified tolerance after: "<<scfIter<<" iterations."<<std::endl;


    //
    // compute and print ground state energy or energy after max scf iterations
    //
    QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
    const double totalEnergy = dftParameters::spinPolarized==0 ?
      energyCalc.computeEnergy(dofHandler,
			       dofHandler,
			       quadrature,
			       quadrature,
			       eigenValues,
			       d_kPointWeights,
			       fermiEnergy,
			       funcX,
			       funcC,
			       d_phiTotRhoIn,
			       d_phiTotRhoOut,
			       *rhoInValues,
			       *rhoOutValues,
			       *rhoOutValues,
			       *gradRhoInValues,
			       *gradRhoOutValues,
			       d_localVselfs,
			       d_atomNodeIdToChargeMap,
			       atomLocations.size(),
			       true) :
      energyCalc.computeEnergySpinPolarized(dofHandler,
					    dofHandler,
					    quadrature,
					    quadrature,
					    eigenValues,
					    d_kPointWeights,
					    fermiEnergy,
					    funcX,
					    funcC,
					    d_phiTotRhoIn,
					    d_phiTotRhoOut,
					    *rhoInValues,
					    *rhoOutValues,
					    *rhoOutValues,
					    *gradRhoInValues,
					    *gradRhoOutValues,
					    *rhoInValuesSpinPolarized,
					    *rhoOutValuesSpinPolarized,
					    *gradRhoInValuesSpinPolarized,
					    *gradRhoOutValuesSpinPolarized,
					    d_localVselfs,
					    d_atomNodeIdToChargeMap,
					    atomLocations.size(),
					    true);

    computing_timer.exit_section("scf solve");

    MPI_Barrier(interpoolcomm) ;
    if (dftParameters::isIonForce)
      {
	computing_timer.enter_section("ion force");
	forcePtr->computeAtomsForces();
	forcePtr->printAtomsForces();
	computing_timer.exit_section("ion force");
      }
#ifdef USE_COMPLEX
    if (dftParameters::isCellStress)
      {
	computing_timer.enter_section("cell stress");
	forcePtr->computeStress();
	forcePtr->printStress();
	computing_timer.exit_section("cell stress");
      }
#endif

    //if (dftParameters::electrostaticsPRefinement)
    //  computeElectrostaticEnergyPRefined();

    if (dftParameters::writeSolutionFields)
      output();
  }

  //Output
  template <unsigned int FEOrder>
  void dftClass<FEOrder>::output()
  {
    DataOut<3> data_outEigen;
    data_outEigen.attach_dof_handler (dofHandlerEigen);
    std::vector<vectorType> tempVec(1);
    tempVec[0].reinit(d_tempEigenVec);
    for(unsigned int i=0; i<numEigenValues; ++i)
      {
	char buffer[100]; sprintf(buffer,"eigen%u", i);
#ifdef USE_COMPLEX
        vectorTools::copyFlattenedDealiiVecToSingleCompVec
		 (d_eigenVectorsFlattened[0],
		  numEigenValues,
		  std::make_pair(i,i+1),
		  localProc_dof_indicesReal,
		  localProc_dof_indicesImag,
		  tempVec);
#else
        vectorTools::copyFlattenedDealiiVecToSingleCompVec
		 (d_eigenVectorsFlattened[0],
		  numEigenValues,
		  std::make_pair(i,i+1),
		  tempVec);
#endif
	data_outEigen.add_data_vector (d_tempEigenVec, buffer);
      }
    data_outEigen.build_patches (C_num1DQuad<FEOrder>());

    std::ofstream output ("eigen.vtu");
    dftUtils::writeDataVTUParallelLowestPoolId(data_outEigen,
					       mpi_communicator,
					       interpoolcomm,
					       std::string("eigen"));

    //
    //compute nodal electron-density from quad data
    //
    dealii::parallel::distributed::Vector<double>  rhoNodalField;
    matrix_free_data.initialize_dof_vector(rhoNodalField);
    rhoNodalField=0;
    dealii::VectorTools::project<3,dealii::parallel::distributed::Vector<double>> (dealii::MappingQ1<3,3>(),
										   dofHandler,
										   constraintsNone,
										   QGauss<3>(C_num1DQuad<FEOrder>()),
										   [&](const typename dealii::DoFHandler<3>::active_cell_iterator & cell , const unsigned int q) -> double {return (*rhoOutValues).find(cell->id())->second[q];},
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
    dftUtils::writeDataVTUParallelLowestPoolId(dataOutRho,
					       mpi_communicator,
					       interpoolcomm,
					       std::string("rhoField"));

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
}

