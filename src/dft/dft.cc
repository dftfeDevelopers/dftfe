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
#include <kohnShamDFTOperator.h>
#include <force.h>
#include <poissonSolverProblem.h>
#include <dealiiLinearSolver.h>
#include <energyCalculator.h>
#include <densityCalculator.h>
#include <symmetry.h>
#include <geoOptIon.h>
#include <geoOptCell.h>
#include <molecularDynamics.h>
#include <meshMovementGaussian.h>
#include <meshMovementAffineTransform.h>
#include <fileReaders.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <chebyshevOrthogonalizedSubspaceIterationSolver.h>
#include <complex>
#include <cmath>
#include <algorithm>
#include <linalg.h>
#include <interpolateFieldsFromPreviousMesh.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsInternal.h>
#include <vectorUtilities.h>
#include <pseudoConverter.h>
//#include <stdafx.h>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/normal_distribution.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <sys/stat.h>

#include <spglib.h>
#include <stdafx.h>

#ifdef DFTFE_WITH_GPU
#include <densityCalculatorCUDA.h>
#include <linearAlgebraOperationsCUDA.h>
#endif

#ifdef DFTFE_WITH_ELPA
extern "C"
{
#include <elpa.hh>
}
#endif


namespace dftfe {

	//Include cc files
#include "pseudoUtils.cc"
#include "moveMeshToAtoms.cc"
#include "initUnmovedTriangulation.cc"
#include "initBoundaryConditions.cc"
#include "initElectronicFields.cc"
#include "initPseudo.cc"
#include "initPseudo-OV.cc"
#include "femUtilityFunctions.cc"
#include "initRho.cc"
#include "initCoreRho.cc" 
#include "atomicRho.cc"    
#include "dos.cc"
#include "localizationLength.cc"
#include "publicMethods.cc"
#include "generateImageCharges.cc"
#include "psiInitialGuess.cc"
#include "fermiEnergy.cc"
#include "charge.cc"
#include "density.cc"
#include "mixingschemes.cc"
#include "nodalDensityMixingSchemes.cc"
#include "pRefinedDoFHandler.cc"
#include "kohnShamEigenSolve.cc"
#include "moveAtoms.cc"
#include "restart.cc"
#include "nscf.cc"
#include "electrostaticHRefinedEnergy.cc"

	//
	//dft constructor
	//
	template<unsigned int FEOrder,unsigned int FEOrderElectro>
		dftClass<FEOrder,FEOrderElectro>::dftClass(const MPI_Comm & mpi_comm_replica,
				const MPI_Comm &_interpoolcomm,
				const MPI_Comm & _interBandGroupComm):
			FE (FE_Q<3>(QGaussLobatto<1>(FEOrder+1)), 1),
#ifdef USE_COMPLEX
			FEEigen (FE_Q<3>(QGaussLobatto<1>(FEOrder+1)), 2),
#else
			FEEigen (FE_Q<3>(QGaussLobatto<1>(FEOrder+1)), 1),
#endif
			mpi_communicator (mpi_comm_replica),
			interpoolcomm (_interpoolcomm),
			interBandGroupComm(_interBandGroupComm),
			n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_comm_replica)),
			this_mpi_process (Utilities::MPI::this_mpi_process(mpi_comm_replica)),
			numElectrons(0),
			numLevels(0),
			d_autoMesh(1),
			d_mesh(mpi_comm_replica,_interpoolcomm,_interBandGroupComm,FEOrder),
			d_affineTransformMesh(mpi_comm_replica),
			d_gaussianMovePar(mpi_comm_replica),
			d_vselfBinsManager(mpi_comm_replica),
			pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
			d_elpaScala(mpi_comm_replica),
			computing_timer (mpi_comm_replica,
					pcout,
					dftParameters::reproducible_output
					|| dftParameters::verbosity<4? TimerOutput::never : TimerOutput::summary,
					TimerOutput::wall_times),
			computingTimerStandard(mpi_comm_replica,
					pcout,
					dftParameters::reproducible_output
					|| dftParameters::verbosity<1? TimerOutput::never : TimerOutput::every_call_and_summary,
					TimerOutput::wall_times),
	    d_subspaceIterationSolver(mpi_comm_replica,
					0.0,
					0.0,
          0.0)
#ifdef DFTFE_WITH_GPU
           ,
			d_subspaceIterationSolverCUDA(mpi_comm_replica,
					0.0,
					0.0,
          0.0)
#endif          
			{
				forcePtr= new forceClass<FEOrder,FEOrderElectro>(this, mpi_comm_replica);
				symmetryPtr= new symmetryClass<FEOrder,FEOrderElectro>(this, mpi_comm_replica, _interpoolcomm);
				geoOptIonPtr= new geoOptIon<FEOrder,FEOrderElectro>(this, mpi_comm_replica);

				geoOptCellPtr= new geoOptCell<FEOrder,FEOrderElectro>(this, mpi_comm_replica);
				d_mdPtr= new molecularDynamics<FEOrder,FEOrderElectro>(this, mpi_comm_replica);

				d_isRestartGroundStateCalcFromChk=false;
#ifdef DFTFE_WITH_ELPA
				int error;

				if (elpa_init(ELPA_API_VERSION) != ELPA_OK) {
					fprintf(stderr, "Error: ELPA API version not supported. Use API version 20181113.");
					exit(1);
				}
#endif

#if defined(DFTFE_WITH_GPU)      
        d_gpucclMpiCommDomainPtr= new GPUCCLWrapper;
        if (dftParameters::useGPUDirectAllReduce)
          d_gpucclMpiCommDomainPtr->init(mpi_comm_replica);
#endif 
        d_pspCutOff=dftParameters::reproducible_output?30.0:(std::max(dftParameters::pspCutoffImageCharges,d_pspCutOffTrunc));
			}

	template<unsigned int FEOrder,unsigned int FEOrderElectro>
		dftClass<FEOrder,FEOrderElectro>::~dftClass()
		{
			delete symmetryPtr;
			matrix_free_data.clear();
			delete forcePtr;
			delete geoOptIonPtr;
			delete geoOptCellPtr;

#ifdef DFTFE_WITH_ELPA
			if (dftParameters::useELPA)
				d_elpaScala.elpaDeallocateHandles(d_numEigenValues,
						d_numEigenValuesRR);

			int error;
			elpa_uninit(& error);
			AssertThrow(error == ELPA_OK,
					dealii::ExcMessage("DFT-FE Error: elpa error."));
#endif
    
#if defined(DFTFE_WITH_GPU)      
      delete d_gpucclMpiCommDomainPtr;
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

	template<unsigned int FEOrder,unsigned int FEOrderElectro>
		double dftClass<FEOrder,FEOrderElectro>::computeVolume(const dealii::DoFHandler<3> & _dofHandler)
		{
			double domainVolume=0;
			const Quadrature<3> &  quadrature=matrix_free_data.get_quadrature(d_densityQuadratureId);
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

	template<unsigned int FEOrder,unsigned int FEOrderElectro>
		void dftClass<FEOrder,FEOrderElectro>::set()
		{
			computingTimerStandard.enter_section("Atomic system initialization");
			if (dftParameters::verbosity>=4)
				dftUtils::printCurrentMemoryUsage(mpi_communicator,
						"Entered call to set");
			//
			//read coordinates
			//
			unsigned int numberColumnsCoordinatesFile =dftParameters::useMeshSizesFromAtomsFile?7:5;

			if (dftParameters::periodicX || dftParameters::periodicY || dftParameters::periodicZ)
			{
				//
				//read fractionalCoordinates of atoms in periodic case
				//
				dftUtils::readFile(numberColumnsCoordinatesFile, atomLocations, dftParameters::coordinatesFile);
				AssertThrow(dftParameters::natoms==atomLocations.size(),ExcMessage("DFT-FE Error: The number atoms"
							"read from the atomic coordinates file (input through ATOMIC COORDINATES FILE) doesn't"
							"match the NATOMS input. Please check your atomic coordinates file. Sometimes an extra"
							"blank row at the end can cause this issue too."));
				pcout << "number of atoms: " << atomLocations.size() << "\n";
				atomLocationsFractional.resize(atomLocations.size()) ;
				//
				//find unique atom types
				//
				for (std::vector<std::vector<double> >::iterator it=atomLocations.begin(); it<atomLocations.end(); it++)
				{
					atomTypes.insert((unsigned int)((*it)[0]));

					if (!dftParameters::isPseudopotential)
						AssertThrow((*it)[0]<=50,ExcMessage("DFT-FE Error: One of the atomic numbers exceeds 50."
									"Currently, for all-electron calculations we have single atom wavefunction and electron-density"
									"initial guess data till atomic number 50 only. Data for the remaining atomic numbers will be"
									"added in the next release. In the mean time, you could also contact the developers of DFT-FE, who can provide"
									"you the data for the single atom wavefunction and electron-density data for"
									"atomic numbers beyond 50."));
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

				AssertThrow(dftParameters::natoms==atomLocations.size(),ExcMessage("DFT-FE Error: The number atoms"
							"read from the atomic coordinates file (input through ATOMIC COORDINATES FILE) doesn't"
							"match the NATOMS input. Please check your atomic coordinates file. Sometimes an extra"
							"blank row at the end can cause this issue too."));
				pcout << "number of atoms: " << atomLocations.size() << "\n";

				//
				//find unique atom types
				//
				for (std::vector<std::vector<double> >::iterator it=atomLocations.begin(); it<atomLocations.end(); it++)
				{
					atomTypes.insert((unsigned int)((*it)[0]));

					if (!dftParameters::isPseudopotential)
						AssertThrow((*it)[0]<=50,ExcMessage("DFT-FE Error: One of the atomic numbers exceeds 50."
									"Currently, for all-electron calculations we have single atom wavefunction and electron-density"
									"initial guess data till atomic number 50 only. Data for the remaining atomic numbers will be"
									"added in the next release. You could also contact the developers of DFT-FE, who can provide"
									"you with the code to generate the single atom wavefunction and electron-density data for"
									"atomic numbers beyond 50."));
				}
			}

			//
			//read Gaussian atomic displacements
			//
			std::vector<std::vector<double> > atomsDisplacementsGaussian;
			d_atomsDisplacementsGaussianRead.resize(atomLocations.size(),Tensor<1,3,double>());
			d_gaussianMovementAtomsNetDisplacements.resize(atomLocations.size(),Tensor<1,3,double>());
			if (dftParameters::coordinatesGaussianDispFile!="")
			{
				dftUtils::readFile(3,
						atomsDisplacementsGaussian,
						dftParameters::coordinatesGaussianDispFile);

				for(int i = 0; i < atomsDisplacementsGaussian.size(); ++i)
					for(int j = 0; j < 3; ++j)
						d_atomsDisplacementsGaussianRead[i][j] = atomsDisplacementsGaussian[i][j];

				d_isAtomsGaussianDisplacementsReadFromFile=true;
			}

			//
			//read domain bounding Vectors
			//
			unsigned int numberColumnsLatticeVectorsFile = 3;
			dftUtils::readFile(numberColumnsLatticeVectorsFile,d_domainBoundingVectors,dftParameters::domainBoundingVectorsFile);

			AssertThrow(d_domainBoundingVectors.size()==3,ExcMessage("DFT-FE Error: The number of domain bounding"
						"vectors read from input file (input through DOMAIN VECTORS FILE) should be 3. Please check"
						"your domain vectors file. Sometimes an extra blank row at the end can cause this issue too."));

			//
			//evaluate cross product of
			//
			std::vector<double> cross;
			dftUtils::cross_product(d_domainBoundingVectors[0],
					d_domainBoundingVectors[1],
					cross);

			double scalarConst = d_domainBoundingVectors[2][0]*cross[0] + d_domainBoundingVectors[2][1]*cross[1] + d_domainBoundingVectors[2][2]*cross[2];
			AssertThrow(scalarConst>0,ExcMessage("DFT-FE Error: Domain bounding vectors or lattice vectors read from"
						"input file (input through DOMAIN VECTORS FILE) should form a right-handed coordinate system."
						"Please check your domain vectors file. This is usually fixed by changing the order of the"
						"vectors in the domain vectors file."));

			pcout << "number of atoms types: " << atomTypes.size() << "\n";


			//
			//determine number of electrons
			//
			for(unsigned int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
			{
				const unsigned int Z = atomLocations[iAtom][0];
				const unsigned int valenceZ = atomLocations[iAtom][1];

				if(dftParameters::isPseudopotential)
					numElectrons += valenceZ;
				else
					numElectrons += Z;
			}

			if(dftParameters::numberEigenValues <= numElectrons/2.0)
			{
				if(dftParameters::verbosity >= 1)
				{
					pcout <<" Warning: User has requested the number of Kohn-Sham wavefunctions to be less than or"
						"equal to half the number of electrons in the system. Setting the Kohn-Sham wavefunctions"
						"to half the number of electrons with a 20 percent buffer to avoid convergence issues in"
						"SCF iterations"<<std::endl;
				}
				d_numEigenValues = (numElectrons/2.0) + std::max(0.2*(numElectrons/2.0),20.0);

				// start with 17% buffer to leave room for additional modifications due to block size restrictions
#ifdef DFTFE_WITH_GPU
				if (dftParameters::useGPU && dftParameters::autoGPUBlockSizes)
					d_numEigenValues = (numElectrons/2.0) + std::max(0.17*(numElectrons/2.0),20.0);
#endif

				if(dftParameters::verbosity >= 1)
				{
					pcout <<" Setting the number of Kohn-Sham wave functions to be "<<d_numEigenValues<<std::endl;
				}
			}

			if (dftParameters::algoType=="FAST")
				dftParameters::numCoreWfcRR=0.93*numElectrons/2.0;


#ifdef DFTFE_WITH_GPU
			if (dftParameters::useGPU && dftParameters::autoGPUBlockSizes)
			{

				const unsigned int numberBandGroups=
					dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);


				d_numEigenValues=std::ceil(d_numEigenValues/(numberBandGroups*1.0))*numberBandGroups;

				AssertThrow((d_numEigenValues%numberBandGroups==0 || d_numEigenValues/numberBandGroups==0)
						,ExcMessage("DFT-FE Error: TOTAL NUMBER OF KOHN-SHAM WAVEFUNCTIONS must be exactly divisible by NPBAND for GPU run."));

				const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
				std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
				dftUtils::createBandParallelizationIndices(interBandGroupComm,
						d_numEigenValues,
						bandGroupLowHighPlusOneIndices);

				const unsigned int eigenvaluesInBandGroup=bandGroupLowHighPlusOneIndices[1];

				if (eigenvaluesInBandGroup<=200)
				{
					dftParameters::chebyWfcBlockSize=eigenvaluesInBandGroup;
					dftParameters::wfcBlockSize=eigenvaluesInBandGroup;
				}
				else if (eigenvaluesInBandGroup<=600)
				{
					std::vector<int> temp1(4,0);
					std::vector<int> temp2(4,0);
					temp1[0]=std::ceil(eigenvaluesInBandGroup/150.0)*150.0*numberBandGroups;
					temp1[1]=std::ceil(eigenvaluesInBandGroup/160.0)*160.0*numberBandGroups;
					temp1[2]=std::ceil(eigenvaluesInBandGroup/170.0)*170.0*numberBandGroups;
					temp1[3]=std::ceil(eigenvaluesInBandGroup/180.0)*180.0*numberBandGroups;

					temp2[0]=150;
					temp2[1]=160;
					temp2[2]=170;
					temp2[3]=180;

					int minElementIndex = std::min_element(temp1.begin(),temp1.end()) - temp1.begin();
					int minElement = *std::min_element(temp1.begin(), temp1.end());

					d_numEigenValues=minElement; 
					dftParameters::chebyWfcBlockSize=temp2[minElementIndex];
					dftParameters::wfcBlockSize=temp2[minElementIndex];
				}
				else if (eigenvaluesInBandGroup<=2000)
				{
					std::vector<int> temp1(4,0);
					std::vector<int> temp2(4,0);
					temp1[0]=std::ceil(eigenvaluesInBandGroup/160.0)*160.0*numberBandGroups;
					temp1[1]=std::ceil(eigenvaluesInBandGroup/180.0)*180.0*numberBandGroups;
					temp1[2]=std::ceil(eigenvaluesInBandGroup/200.0)*200.0*numberBandGroups;
					temp1[3]=std::ceil(eigenvaluesInBandGroup/220.0)*220.0*numberBandGroups;

					temp2[0]=160;
					temp2[1]=180;
					temp2[2]=200;
					temp2[3]=220;

					int minElementIndex = std::min_element(temp1.begin(),temp1.end()) - temp1.begin();
					int minElement = *std::min_element(temp1.begin(), temp1.end());

					d_numEigenValues=minElement;                            
					dftParameters::chebyWfcBlockSize=temp2[minElementIndex];
					dftParameters::wfcBlockSize=temp2[minElementIndex];
				}
				else
				{
					std::vector<int> temp1(4,0);
					std::vector<int> temp2(4,0);
					temp1[0]=std::ceil(eigenvaluesInBandGroup/360.0)*360.0*numberBandGroups;
					temp1[1]=std::ceil(eigenvaluesInBandGroup/380.0)*380.0*numberBandGroups;
					temp1[2]=std::ceil(eigenvaluesInBandGroup/400.0)*400.0*numberBandGroups;
					temp1[3]=std::ceil(eigenvaluesInBandGroup/440.0)*440.0*numberBandGroups;

					temp2[0]=360;
					temp2[1]=380;
					temp2[2]=400;
					temp2[3]=440;

					int minElementIndex = std::min_element(temp1.begin(),temp1.end()) - temp1.begin();
					int minElement = *std::min_element(temp1.begin(), temp1.end());

					d_numEigenValues=minElement;
					dftParameters::chebyWfcBlockSize=numberBandGroups>1?temp2[minElementIndex]:temp2[minElementIndex]/2;
					dftParameters::wfcBlockSize=temp2[minElementIndex];
				}

				if (dftParameters::algoType=="FAST")
					dftParameters::numCoreWfcRR=std::floor(dftParameters::numCoreWfcRR/dftParameters::wfcBlockSize)*dftParameters::wfcBlockSize;

				if(dftParameters::verbosity >= 1)
				{
					pcout <<" Setting the number of Kohn-Sham wave functions for GPU run to be: "<<d_numEigenValues<<std::endl;
					pcout <<" Setting CHEBY WFC BLOCK SIZE for GPU run to be "<<dftParameters::chebyWfcBlockSize<<std::endl;
					pcout <<" Setting WFC BLOCK SIZE for GPU run to be "<<dftParameters::wfcBlockSize<<std::endl;
					if (dftParameters::algoType=="FAST")
						pcout <<" Setting SPECTRUM SPLIT CORE EIGENSTATES for GPU run to be "<<dftParameters::numCoreWfcRR<<std::endl;
				}
			}
#endif

			if (dftParameters::constraintMagnetization)
			{
				numElectronsUp = std::ceil(static_cast<double>(numElectrons)/2.0);
				numElectronsDown = numElectrons - numElectronsUp;
				//
				int netMagnetization = std::round(2.0 * static_cast<double>(numElectrons) * dftParameters::start_magnetization ) ;
				//
				while ( (numElectronsUp-numElectronsDown) < std::abs(netMagnetization))
				{
					numElectronsDown -=1 ;
					numElectronsUp +=1 ;
				}
				//
				if(dftParameters::verbosity >= 1)
				{
					pcout <<" Number of spin up electrons "<<numElectronsUp<<std::endl;
					pcout <<" Number of spin down electrons "<<numElectronsDown<<std::endl;
				}
			}

			//estimate total number of wave functions from atomic orbital filling
			if (dftParameters::startingWFCType=="ATOMIC")
				determineOrbitalFilling();

			AssertThrow(dftParameters::numCoreWfcRR<=d_numEigenValues
					,ExcMessage("DFT-FE Error: Incorrect input value used- SPECTRUM SPLIT CORE EIGENSTATES should be less than the total number of wavefunctions."));
			d_numEigenValuesRR=d_numEigenValues-dftParameters::numCoreWfcRR;


#ifdef USE_COMPLEX
			generateMPGrid();
#else
			d_kPointCoordinates.resize(3,0.0);
			d_kPointWeights.resize(1,1.0);
#endif

			//set size of eigenvalues and eigenvectors data structures
			eigenValues.resize(d_kPointWeights.size());
			eigenValuesRRSplit.resize(d_kPointWeights.size());

      a0.clear();
      bLow.clear();

			a0.resize((dftParameters::spinPolarized+1)*d_kPointWeights.size(),0.0);
			bLow.resize((dftParameters::spinPolarized+1)*d_kPointWeights.size(),0.0);

      d_upperBoundUnwantedSpectrumValues.clear();
      d_upperBoundUnwantedSpectrumValues.resize((dftParameters::spinPolarized+1)*d_kPointWeights.size(),0.0);

			d_eigenVectorsFlattenedSTL.resize((1+dftParameters::spinPolarized)*d_kPointWeights.size());
			d_eigenVectorsRotFracDensityFlattenedSTL.resize((1+dftParameters::spinPolarized)*d_kPointWeights.size());

			for(unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
			{
				eigenValues[kPoint].resize((dftParameters::spinPolarized+1)*d_numEigenValues);
				eigenValuesRRSplit[kPoint].resize((dftParameters::spinPolarized+1)*d_numEigenValuesRR);
			}

			//convert pseudopotential files in upf format to dftfe format
			if(dftParameters::verbosity>=1)
			{
				pcout<<std::endl<<"Reading Pseudo-potential data for each atom from the list given in : " <<dftParameters::pseudoPotentialFile<<std::endl;
			}

      int nlccFlag=0;
			if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && dftParameters::isPseudopotential == true)
				nlccFlag=pseudoUtils::convert(dftParameters::pseudoPotentialFile);

      nlccFlag = Utilities::MPI::sum(nlccFlag, MPI_COMM_WORLD);

			if(nlccFlag > 0 && dftParameters::isPseudopotential == true)
          dftParameters::nonLinearCoreCorrection = true;

			if(dftParameters::verbosity>=1)
        if(dftParameters::nonLinearCoreCorrection == true)
          pcout<<"Atleast one atom has pseudopotential with nonlinear core correction"<<std::endl;

			d_elpaScala.processGridOptionalELPASetup(d_numEigenValues,
					d_numEigenValuesRR);

			MPI_Barrier(MPI_COMM_WORLD);
			computingTimerStandard.exit_section("Atomic system initialization");
		}

	//dft pseudopotential init
	template<unsigned int FEOrder,unsigned int FEOrderElectro>
		void dftClass<FEOrder,FEOrderElectro>::initPseudoPotentialAll(const bool updateNonlocalSparsity)
		{
			if(dftParameters::isPseudopotential)
			{
				TimerOutput::Scope scope (computing_timer, "psp init");
				pcout<<std::endl<<"Pseudopotential initalization...."<<std::endl;
				const Quadrature<3> &  quadrature=matrix_free_data.get_quadrature(d_densityQuadratureId);

        double init_core;
        MPI_Barrier(MPI_COMM_WORLD);
        init_core = MPI_Wtime();

        if(dftParameters::nonLinearCoreCorrection == true)
				  initCoreRho();

        MPI_Barrier(MPI_COMM_WORLD);
        init_core = MPI_Wtime() - init_core;
        if (dftParameters::verbosity>=1)
          pcout<<"initPseudoPotentialAll: Time taken for initializing core density for non-linear core correction: "<<init_core<<std::endl;


				if (updateNonlocalSparsity)
				{
					double init_nonlocal1;
					MPI_Barrier(MPI_COMM_WORLD);
					init_nonlocal1 = MPI_Wtime();

					computeSparseStructureNonLocalProjectors_OV();

					MPI_Barrier(MPI_COMM_WORLD);
					init_nonlocal1 = MPI_Wtime() - init_nonlocal1;
					if (dftParameters::verbosity>=1)
						pcout<<"initPseudoPotentialAll: Time taken for computeSparseStructureNonLocalProjectors_OV: "<<init_nonlocal1<<std::endl;
				}

				double init_nonlocal2;
				MPI_Barrier(MPI_COMM_WORLD);
				init_nonlocal2 = MPI_Wtime();

				computeElementalOVProjectorKets();

				//forcePtr->initPseudoData();

				MPI_Barrier(MPI_COMM_WORLD);
				init_nonlocal2 = MPI_Wtime() - init_nonlocal2;
				if (dftParameters::verbosity>=1)
					pcout<<"initPseudoPotentialAll: Time taken for non local psp init: "<<init_nonlocal2<<std::endl;
			}
		}


	// generate image charges and update k point cartesian coordinates based on current lattice vectors
	template<unsigned int FEOrder,unsigned int FEOrderElectro>
		void dftClass<FEOrder,FEOrderElectro>::initImageChargesUpdateKPoints(bool flag)
		{
			TimerOutput::Scope scope (computing_timer, "image charges and k point generation");
			pcout<<"-----------Simulation Domain bounding vectors (lattice vectors in fully periodic case)-------------"<<std::endl;
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

				if(flag)
				{
					for(unsigned int i = 0; i < atomLocationsFractional.size(); ++i)
					{
						for(unsigned int idim = 0; idim < 3; ++idim)
						{
							if (periodicBc[idim])
								AssertThrow(atomLocationsFractional[i][2+idim]>-tol && atomLocationsFractional[i][2+idim]<1.0+tol,ExcMessage("DFT-FE Error: periodic direction fractional coordinates doesn't lie in [0,1]. Please check input"
											"fractional coordinates, or if this is an ionic relaxation step, please check the corresponding"
											"algorithm."));
							if (!periodicBc[idim])
								AssertThrow(atomLocationsFractional[i][2+idim]>tol && atomLocationsFractional[i][2+idim]<1.0-tol,ExcMessage("DFT-FE Error: non-periodic direction fractional coordinates doesn't lie in (0,1). Please check"
											"input fractional coordinates, or if this is an ionic relaxation step, please check the"
											"corresponding algorithm."));
						}
					}
				}

				generateImageCharges(d_pspCutOff,
						d_imageIds,
						d_imageCharges,
						d_imagePositions,
						d_globalChargeIdToImageIdMap);

				generateImageCharges(d_pspCutOffTrunc,
						d_imageIdsTrunc,
						d_imageChargesTrunc,
						d_imagePositionsTrunc,
						d_globalChargeIdToImageIdMapTrunc);

				if ((dftParameters::verbosity>=4 || dftParameters::reproducible_output))
					pcout<<"Number Image Charges  "<<d_imageIds.size()<<std::endl;

				internaldft::convertToCellCenteredCartesianCoordinates(atomLocations,
						d_domainBoundingVectors);
#ifdef USE_COMPLEX
				recomputeKPointCoordinates();
#endif
				if (dftParameters::verbosity>=4)
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

				//
				//redundant call (check later)
				//
				generateImageCharges(d_pspCutOff,
						d_imageIds,
						d_imageCharges,
						d_imagePositions,
						d_globalChargeIdToImageIdMap);

				generateImageCharges(d_pspCutOffTrunc,
						d_imageIdsTrunc,
						d_imageChargesTrunc,
						d_imagePositionsTrunc,
						d_globalChargeIdToImageIdMapTrunc);
			}
		}

	//dft init
	template<unsigned int FEOrder,unsigned int FEOrderElectro>
		void dftClass<FEOrder,FEOrderElectro>::init (const unsigned int usePreviousGroundStateFields)
		{
			computingTimerStandard.enter_section("KSDFT problem initialization");

			if (dftParameters::verbosity>=4)
				dftUtils::printCurrentMemoryUsage(mpi_communicator,
						"Entering init");

			initImageChargesUpdateKPoints();

      calculateNearestAtomDistances();

			computing_timer.enter_section("mesh generation");
			//
			//generate mesh (both parallel and serial)
			//while parallel meshes are always generated, serial meshes are only generated
			//for following three cases: symmetrization is on, ionic optimization is on as well
			//as reuse wfcs and density from previous ionic step is on, or if serial constraints
			//generation is on.
			//
			if ((dftParameters::chkType==2 || dftParameters::chkType==3)  && (dftParameters::restartFromChk || dftParameters::restartMdFromChk))
			{
				d_mesh.generateCoarseMeshesForRestart(atomLocations,
						d_imagePositionsTrunc,
						d_imageIdsTrunc,
            d_nearestAtomDistances,
						d_domainBoundingVectors,
						dftParameters::useSymm || dftParameters::createConstraintsFromSerialDofhandler);

				if (dftParameters::chkType==2)
					loadTriaInfoAndRhoData();
				else if (dftParameters::chkType==3)
					loadTriaInfoAndRhoNodalData(); 
			}
			else
			{
				d_mesh.generateSerialUnmovedAndParallelMovedUnmovedMesh(atomLocations,
						d_imagePositionsTrunc,
						d_imageIdsTrunc,
            d_nearestAtomDistances,
						d_domainBoundingVectors,
						dftParameters::useSymm || dftParameters::createConstraintsFromSerialDofhandler,
						dftParameters::electrostaticsHRefinement);

			}
			computing_timer.exit_section("mesh generation");

			if (dftParameters::verbosity>=4)
				dftUtils::printCurrentMemoryUsage(mpi_communicator,
						"Mesh generation completed");
			//
			//get access to triangulation objects from meshGenerator class
			//
			parallel::distributed::Triangulation<3> & triangulationPar = d_mesh.getParallelMeshMoved();

			//
			//initialize dofHandlers and hanging-node constraints and periodic constraints on the unmoved Mesh
			//
			initUnmovedTriangulation(triangulationPar);

			if (dftParameters::verbosity>=4)
				dftUtils::printCurrentMemoryUsage(mpi_communicator,
						"initUnmovedTriangulation completed");
#ifdef USE_COMPLEX
			if (dftParameters::useSymm)
				symmetryPtr->initSymmetry() ;
#endif




			//
			//move triangulation to have atoms on triangulation vertices
			//
      if (!dftParameters::floatingNuclearCharges)
        moveMeshToAtoms(triangulationPar,
            d_mesh.getSerialMeshUnmoved());


      if (dftParameters::smearedNuclearCharges)
         calculateSmearedChargeWidths();

			if (dftParameters::verbosity>=4)
				dftUtils::printCurrentMemoryUsage(mpi_communicator,
						"moveMeshToAtoms completed");
			//
			//initialize dirichlet BCs for total potential and vSelf poisson solutions
			//
			initBoundaryConditions();


			if (dftParameters::verbosity>=4)
				dftUtils::printCurrentMemoryUsage(mpi_communicator,
						"initBoundaryConditions completed");
			//
			//initialize guesses for electron-density and wavefunctions
			//
			initElectronicFields();

			if (dftParameters::chkType==3 && dftParameters::restartFromChk)
			{
				if (!d_isAtomsGaussianDisplacementsReadFromFile)
				{
					for (unsigned int i = 0; i < d_rhoInNodalValues.local_size(); i++)
						d_rhoInNodalValues.local_element(i)=d_rhoInNodalValuesRead.local_element(i);

					d_rhoInNodalValues.update_ghost_values();
					interpolateRhoNodalDataToQuadratureDataGeneral(d_matrixFreeDataPRefined,
              d_densityDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
							d_rhoInNodalValues,
							*(rhoInValues),
							*(gradRhoInValues),
							*(gradRhoInValues),
							dftParameters::xcFamilyType=="GGA");

					normalizeRhoInQuadValues();
				}

				d_isRestartGroundStateCalcFromChk=true;
			}

			if (dftParameters::verbosity>=4)
				dftUtils::printCurrentMemoryUsage(mpi_communicator,
						"initElectronicFields completed");
			//
			//initialize pseudopotential data for both local and nonlocal part
			//
			initPseudoPotentialAll();

			if (dftParameters::verbosity>=4)
				dftUtils::printCurrentMemoryUsage(mpi_communicator,
						"initPseudopotential completed");

			//
			//Apply Gaussian displacments to atoms and mesh if input gaussian displacments
			//are read from file. When restarting a relaxation, this must be done only once
			//at the begining- this is why the flag is to false after the Gaussian movement.
			//The last flag to updateAtomPositionsAndMoveMesh is set to true to force use of
			//single atom solutions.
			//
			if (d_isAtomsGaussianDisplacementsReadFromFile)
			{
				updateAtomPositionsAndMoveMesh(d_atomsDisplacementsGaussianRead,
						1e+4,
						true,
						false);
				d_isAtomsGaussianDisplacementsReadFromFile=false;

				if (dftParameters::chkType==3 && dftParameters::restartFromChk)
				{
					for (unsigned int i = 0; i < d_rhoInNodalValues.local_size(); i++)
						d_rhoInNodalValues.local_element(i)=d_rhoInNodalValuesRead.local_element(i);

					d_rhoInNodalValues.update_ghost_values();
					interpolateRhoNodalDataToQuadratureDataGeneral(d_matrixFreeDataPRefined,
              d_densityDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
							d_rhoInNodalValues,
							*(rhoInValues),
							*(gradRhoInValues),
							*(gradRhoInValues),
							dftParameters::xcFamilyType=="GGA");

					normalizeRhoInQuadValues();
				}
			}

      d_isFirstFilteringCall.clear();
      d_isFirstFilteringCall.resize((dftParameters::spinPolarized+1)*d_kPointWeights.size(),true);

			computingTimerStandard.exit_section("KSDFT problem initialization");
		}

	template<unsigned int FEOrder,unsigned int FEOrderElectro>
		void dftClass<FEOrder,FEOrderElectro>::initNoRemesh(const bool updateImagesAndKPointsAndVselfBins,
        const bool updateSmearedChargeWidths,
				const bool useSingleAtomSolutionOverride,
				const bool useAtomicRhoSplitDensityUpdateForGeoOpt)
		{
			computingTimerStandard.enter_section("KSDFT problem initialization");
			if(updateImagesAndKPointsAndVselfBins)
      {
				initImageChargesUpdateKPoints();

        if (updateSmearedChargeWidths)
        {
          calculateNearestAtomDistances(); 

          if (dftParameters::smearedNuclearCharges)
            calculateSmearedChargeWidths();
        }
      }

			//
			//reinitialize dirichlet BCs for total potential and vSelf poisson solutions
			//
			double init_bc;
			MPI_Barrier(MPI_COMM_WORLD);
			init_bc = MPI_Wtime();


      // false option reinitializes vself bins from scratch wheras true option only updates the boundary conditions
      const bool updateOnlyBinsBc=!updateImagesAndKPointsAndVselfBins;
      initBoundaryConditions(updateOnlyBinsBc);

			MPI_Barrier(MPI_COMM_WORLD);
			init_bc = MPI_Wtime() - init_bc;
			if (dftParameters::verbosity>=1)
				pcout<<"updateAtomPositionsAndMoveMesh: Time taken for initBoundaryConditions: "<<init_bc<<std::endl;

			double init_rho;
			MPI_Barrier(MPI_COMM_WORLD);
			init_rho = MPI_Wtime();

			if (useSingleAtomSolutionOverride)
			{
				readPSI();
				initRho();
			}
			else
			{
				//
				//rho init (use previous ground state electron density)
				//
				//if(dftParameters::mixingMethod != "ANDERSON_WITH_KERKER")
				//   solveNoSCF();

				noRemeshRhoDataInit();

        if (dftParameters::isIonOpt)
        {
          if (!dftParameters::reuseWfcGeoOpt)
            readPSI();

          if (dftParameters::reuseDensityGeoOpt && useAtomicRhoSplitDensityUpdateForGeoOpt && dftParameters::spinPolarized!=1)
          {
            d_rhoOutNodalValuesSplit.add(-totalCharge(d_matrixFreeDataPRefined,
                d_rhoOutNodalValuesSplit)/d_domainVolume);

            initAtomicRho();

            interpolateRhoNodalDataToQuadratureDataGeneral(d_matrixFreeDataPRefined,
                d_densityDofHandlerIndexElectro,
                d_densityQuadratureIdElectro,
                d_rhoOutNodalValuesSplit,
                *(rhoInValues),
                *(gradRhoInValues),
                *(gradRhoInValues),
                dftParameters::xcFamilyType=="GGA");

            addAtomicRhoQuadValuesGradients(*(rhoInValues),
                                            *(gradRhoInValues),
                                            dftParameters::xcFamilyType=="GGA");

            normalizeRhoInQuadValues();

            l2ProjectionQuadToNodal(d_matrixFreeDataPRefined,
                d_constraintsRhoNodal,
                d_densityDofHandlerIndexElectro,
                d_densityQuadratureIdElectro,
                *rhoInValues,
                d_rhoInNodalValues);

            d_rhoInNodalValues.update_ghost_values();
          }
          else
          {
            initRho();
          }
        }
			}

			MPI_Barrier(MPI_COMM_WORLD);
			init_rho = MPI_Wtime() - init_rho;
			if (dftParameters::verbosity>=1)
				pcout<<"updateAtomPositionsAndMoveMesh: Time taken for initRho: "<<init_rho<<std::endl;

			//
			//reinitialize pseudopotential related data structures
			//
			double init_pseudo;
			MPI_Barrier(MPI_COMM_WORLD);
			init_pseudo = MPI_Wtime();

			initPseudoPotentialAll(dftParameters::floatingNuclearCharges?true:false);

			MPI_Barrier(MPI_COMM_WORLD);
			init_pseudo = MPI_Wtime() - init_pseudo;
			if (dftParameters::verbosity>=1)
				pcout<<"Time taken for initPseudoPotentialAll: "<<init_pseudo<<std::endl;

      d_isFirstFilteringCall.clear();
      d_isFirstFilteringCall.resize((dftParameters::spinPolarized+1)*d_kPointWeights.size(),true);

			computingTimerStandard.exit_section("KSDFT problem initialization");
		}

	//
	// deform domain and call appropriate reinits
	//
	template<unsigned int FEOrder,unsigned int FEOrderElectro>
		void dftClass<FEOrder,FEOrderElectro>::deformDomain(const Tensor<2,3,double> & deformationGradient)
		{
			d_affineTransformMesh.initMoved(d_domainBoundingVectors);
			d_affineTransformMesh.transform(deformationGradient);

			dftUtils::transformDomainBoundingVectors(d_domainBoundingVectors,deformationGradient);

			initNoRemesh(true,true,false,false);
		}


	//
	//generate a-posteriori mesh
	//
	template<unsigned int FEOrder,unsigned int FEOrderElectro>
		void dftClass<FEOrder,FEOrderElectro>::aposterioriMeshGenerate()
		{
			//
			//get access to triangulation objects from meshGenerator class
			//
			parallel::distributed::Triangulation<3> & triangulationPar = d_mesh.getParallelMeshMoved();
			unsigned int numberLevelRefinements = dftParameters::numLevels;
			unsigned int numberWaveFunctionsErrorEstimate = dftParameters::numberWaveFunctionsForEstimate;
			bool refineFlag = true;
			unsigned int countLevel = 0;
			double traceXtKX = computeTraceXtKX(numberWaveFunctionsErrorEstimate);
			double traceXtKXPrev = traceXtKX;

			while(refineFlag)
			{
				if(numberLevelRefinements > 0)
				{
					distributedCPUVec<double> tempVec;
					matrix_free_data.initialize_dof_vector(tempVec);

					std::vector<distributedCPUVec<double> > eigenVectorsArray(numberWaveFunctionsErrorEstimate);

					for(unsigned int i = 0; i < numberWaveFunctionsErrorEstimate; ++i)
						eigenVectorsArray[i].reinit(tempVec);


					vectorTools::copyFlattenedSTLVecToSingleCompVec(d_eigenVectorsFlattenedSTL[0],
							d_numEigenValues,
							std::make_pair(0,numberWaveFunctionsErrorEstimate),
							eigenVectorsArray);


					for(unsigned int i= 0; i < numberWaveFunctionsErrorEstimate; ++i)
					{
						constraintsNone.distribute(eigenVectorsArray[i]);
						eigenVectorsArray[i].update_ghost_values();
					}


					d_mesh.generateAutomaticMeshApriori(dofHandler,
							triangulationPar,
							eigenVectorsArray,
							FEOrder,
							dftParameters::electrostaticsHRefinement);

				}


				//
				//initialize dofHandlers of refined mesh and move triangulation
				//
				initUnmovedTriangulation(triangulationPar);
				moveMeshToAtoms(triangulationPar,
						d_mesh.getSerialMeshUnmoved());
				initBoundaryConditions();
				initElectronicFields();
				initPseudoPotentialAll();

				//
				//compute Tr(XtHX) for each level of mesh
				//
				//dataTypes::number traceXtHX = computeTraceXtHX(numberWaveFunctionsErrorEstimate);
				//pcout<<" Tr(XtHX) value for Level: "<<countLevel<<" "<<traceXtHX<<std::endl;

				//
				//compute Tr(XtKX) for each level of mesh
				//
				traceXtKX = computeTraceXtKX(numberWaveFunctionsErrorEstimate);
				if(dftParameters::verbosity>0)
					pcout<<" Tr(XtKX) value for Level: "<<countLevel<<" "<<traceXtKX<<std::endl;

				//compute change in traceXtKX
				double deltaKinetic = std::abs(traceXtKX - traceXtKXPrev)/atomLocations.size();

				//reset traceXtkXPrev to traceXtKX
				traceXtKXPrev = traceXtKX;

				//
				//set refineFlag
				//
				countLevel += 1;
				if(countLevel >= numberLevelRefinements || deltaKinetic <= dftParameters::toleranceKinetic)
					refineFlag = false;

			}

		}


	//
	//dft run
	//
	template<unsigned int FEOrder,unsigned int FEOrderElectro>
		void dftClass<FEOrder,FEOrderElectro>::run()
		{
			if(dftParameters::meshAdaption)
				aposterioriMeshGenerate();

			if (dftParameters::isBOMD)
			{
				d_mdPtr->run();
			}
			else
			{
				if (!(dftParameters::chkType==1  && dftParameters::restartFromChk && dftParameters::ionOptSolver == "CGPRP"))
				{
					solve(false,
							true,
							false,
							d_isRestartGroundStateCalcFromChk);
				}

				d_isRestartGroundStateCalcFromChk=false;
				if (dftParameters::isIonOpt && !dftParameters::isCellOpt)
				{
					d_atomLocationsInitial = atomLocations;
					d_freeEnergyInitial = d_freeEnergy;

					geoOptIonPtr->init();
					geoOptIonPtr->run();
				}
				else if (!dftParameters::isIonOpt && dftParameters::isCellOpt)
				{
					d_atomLocationsInitial = atomLocations;
					d_freeEnergyInitial = d_freeEnergy;

					geoOptCellPtr->init();
					geoOptCellPtr->run();
				}
				else if (dftParameters::isIonOpt && dftParameters::isCellOpt)
				{
					d_atomLocationsInitial = atomLocations;
					d_freeEnergyInitial = d_freeEnergy;

					//first relax ion positions in the starting cell configuration
					geoOptIonPtr->init();
					geoOptIonPtr->run();

					//start cell relaxation, where for each cell relaxation update the ion positions are again relaxed
					geoOptCellPtr->init();
					geoOptCellPtr->run();
				}
			}

			if(dftParameters::writeDosFile)
				compute_tdos(eigenValues,
						"dosData.out");

			if(dftParameters::writeLdosFile)
				compute_ldos(eigenValues,
						"ldosData.out");

			if(dftParameters::writePdosFile)
				compute_pdos(eigenValues,
						"pdosData");

			if(dftParameters::writeLocalizationLengths)
				compute_localizationLength("localizationLengths.out");


			if (dftParameters::verbosity>=1)
				pcout << std::endl<< "------------------DFT-FE ground-state solve completed---------------------------"<<std::endl;
		}


	//
	//initialize
	//
	template<unsigned int FEOrder,unsigned int FEOrderElectro>
		void dftClass<FEOrder,FEOrderElectro>::initializeKohnShamDFTOperator(kohnShamDFTOperatorClass<FEOrder,FEOrderElectro> & kohnShamDFTEigenOperator
#ifdef DFTFE_WITH_GPU
				,
				kohnShamDFTOperatorCUDAClass<FEOrder,FEOrderElectro> & kohnShamDFTEigenOperatorCUDA
#endif
				,
				const bool initializeCUDAScala)
		{
			if (!dftParameters::useGPU)
			{
				kohnShamDFTEigenOperator.init();

				kohnShamDFTEigenOperator.processGridOptionalELPASetup(d_numEigenValues,
						d_numEigenValuesRR);
			}

#ifdef DFTFE_WITH_GPU
			if (dftParameters::useGPU)
			{
				kohnShamDFTEigenOperatorCUDA.init();

				if (initializeCUDAScala)
				{
					kohnShamDFTEigenOperatorCUDA.createCublasHandle();

					kohnShamDFTEigenOperatorCUDA.processGridSetup(d_numEigenValues,
							d_numEigenValuesRR);
				}

				AssertThrow((d_numEigenValues%dftParameters::chebyWfcBlockSize==0 || d_numEigenValues/dftParameters::chebyWfcBlockSize==0)
						,ExcMessage("DFT-FE Error: total number wavefunctions must be exactly divisible by cheby wfc block size for GPU run."));


				AssertThrow((d_numEigenValues%dftParameters::wfcBlockSize==0 || d_numEigenValues/dftParameters::wfcBlockSize==0)
						,ExcMessage("DFT-FE Error: total number wavefunctions must be exactly divisible by wfc block size for GPU run."));

				AssertThrow((dftParameters::wfcBlockSize%dftParameters::chebyWfcBlockSize==0 && dftParameters::wfcBlockSize/dftParameters::chebyWfcBlockSize>=0)
						,ExcMessage("DFT-FE Error: wfc block size must be exactly divisible by cheby wfc block size and also larger for GPU run."));

				if (d_numEigenValuesRR!=d_numEigenValues)
					AssertThrow((d_numEigenValuesRR%dftParameters::wfcBlockSize==0 || d_numEigenValuesRR/dftParameters::wfcBlockSize==0)
							,ExcMessage("DFT-FE Error: total number RR wavefunctions must be exactly divisible by wfc block size for GPU run."));

				AssertThrow((dftParameters::mixedPrecXtHXFracStates%dftParameters::wfcBlockSize==0
							|| dftParameters::mixedPrecXtHXFracStates/dftParameters::wfcBlockSize==0)
						,ExcMessage("DFT-FE Error: MIXED PREC XTHX FRAC STATES must be exactly divisible by WFC BLOCK SIZE for GPU run."));

				//band group parallelization data structures
				const unsigned int numberBandGroups=
					dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);

				AssertThrow((d_numEigenValues%numberBandGroups==0 || d_numEigenValues/numberBandGroups==0)
						,ExcMessage("DFT-FE Error: TOTAL NUMBER OF KOHN-SHAM WAVEFUNCTIONS must be exactly divisible by NPBAND for GPU run."));

				const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
				std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
				dftUtils::createBandParallelizationIndices(interBandGroupComm,
						d_numEigenValues,
						bandGroupLowHighPlusOneIndices);

				AssertThrow((bandGroupLowHighPlusOneIndices[1]%dftParameters::chebyWfcBlockSize==0)
						,ExcMessage("DFT-FE Error: band parallelization group size must be exactly divisible by CHEBY WFC BLOCK SIZE for GPU run."));

				AssertThrow((bandGroupLowHighPlusOneIndices[1]%dftParameters::wfcBlockSize==0)
						,ExcMessage("DFT-FE Error: band parallelization group size must be exactly divisible by WFC BLOCK SIZE for GPU run."));

				kohnShamDFTEigenOperatorCUDA.reinit(std::min(dftParameters::chebyWfcBlockSize,
							d_numEigenValues),
						true);
			}
#endif
		}


	//
	//re-initialize (significantly cheaper than initialize)
	//
	template<unsigned int FEOrder,unsigned int FEOrderElectro>
		void dftClass<FEOrder,FEOrderElectro>::reInitializeKohnShamDFTOperator(kohnShamDFTOperatorClass<FEOrder,FEOrderElectro> & kohnShamDFTEigenOperator
#ifdef DFTFE_WITH_GPU
				,
				kohnShamDFTOperatorCUDAClass<FEOrder,FEOrderElectro> & kohnShamDFTEigenOperatorCUDA
#endif
				)
		{
			if (!dftParameters::useGPU)
			{
				kohnShamDFTEigenOperator.init();
			}

#ifdef DFTFE_WITH_GPU
			if (dftParameters::useGPU)
			{
				kohnShamDFTEigenOperatorCUDA.init();

				kohnShamDFTEigenOperatorCUDA.reinitNoRemesh(std::min(dftParameters::chebyWfcBlockSize,
							d_numEigenValues));
			}
#endif
		}

	//
	//finalize
	//
	template<unsigned int FEOrder,unsigned int FEOrderElectro>
		void dftClass<FEOrder,FEOrderElectro>::finalizeKohnShamDFTOperator(kohnShamDFTOperatorClass<FEOrder,FEOrderElectro> & kohnShamDFTEigenOperator
#ifdef DFTFE_WITH_GPU
				,
				kohnShamDFTOperatorCUDAClass<FEOrder,FEOrderElectro> & kohnShamDFTEigenOperatorCUDA
#endif
				)
		{
#ifdef DFTFE_WITH_GPU
			if (dftParameters::useGPU)
				kohnShamDFTEigenOperatorCUDA.destroyCublasHandle();
#endif

#ifdef DFTFE_WITH_ELPA
			if (dftParameters::useELPA && !dftParameters::useGPU)
				kohnShamDFTEigenOperator.elpaDeallocateHandles(d_numEigenValues,
						d_numEigenValuesRR);
#endif
		}

	//
	//dft solve
	//
	template<unsigned int FEOrder,unsigned int FEOrderElectro>
		void dftClass<FEOrder,FEOrderElectro>::computeDensityPerturbation(const bool kohnShamDFTOperatorsInitialized)
		{
			kohnShamDFTOperatorClass<FEOrder,FEOrderElectro> kohnShamDFTEigenOperator(this,mpi_communicator);
#ifdef DFTFE_WITH_GPU
			kohnShamDFTOperatorCUDAClass<FEOrder,FEOrderElectro> kohnShamDFTEigenOperatorCUDA(this,mpi_communicator);
#endif

			const Quadrature<3> &  quadrature=matrix_free_data.get_quadrature(d_densityQuadratureId);

			//computingTimerStandard.enter_section("Total scf solve");
			computingTimerStandard.enter_section("Kohn-sham dft operator init");
			energyCalculator energyCalc(mpi_communicator, interpoolcomm,interBandGroupComm);


      std::vector<std::vector<dataTypes::number> > eigenVectorsFlattenedSTLTemp=d_eigenVectorsFlattenedSTL;

			//set up linear solver
			dealiiLinearSolver dealiiCGSolver(mpi_communicator, dealiiLinearSolver::CG);

			//set up solver functions for Poisson
			poissonSolverProblem<FEOrder,FEOrderElectro> phiTotalSolverProblem(mpi_communicator);

			if (!kohnShamDFTOperatorsInitialized || true)
				initializeKohnShamDFTOperator(kohnShamDFTEigenOperator
#ifdef DFTFE_WITH_GPU
						,
						kohnShamDFTEigenOperatorCUDA
#endif
						);
			else
				reInitializeKohnShamDFTOperator(kohnShamDFTEigenOperator
#ifdef DFTFE_WITH_GPU
						,
						kohnShamDFTEigenOperatorCUDA
#endif
						);

			//
			//precompute shapeFunctions and shapeFunctionGradients and shapeFunctionGradientIntegrals
			//
			computing_timer.enter_section("shapefunction data");
			if (!dftParameters::useGPU)
				kohnShamDFTEigenOperator.preComputeShapeFunctionGradientIntegrals(d_lpspQuadratureId);
#ifdef DFTFE_WITH_GPU
			if (dftParameters::useGPU)
				kohnShamDFTEigenOperatorCUDA.preComputeShapeFunctionGradientIntegrals(d_lpspQuadratureId);
#endif
			computing_timer.exit_section("shapefunction data");

			if (dftParameters::verbosity>=4)
				dftUtils::printCurrentMemoryUsage(mpi_communicator,
						"Precompute shapefunction grad integrals, just before starting scf solve");

			if (dftParameters::verbosity>=4)
				dftUtils::printCurrentMemoryUsage(mpi_communicator,
						"Kohn-sham dft operator init called");

			computingTimerStandard.exit_section("Kohn-sham dft operator init");



			computingTimerStandard.enter_section("Density perturbation computation");
			computing_timer.enter_section("Density perturbation computation");

      phiTotalSolverProblem.reinit(d_matrixFreeDataPRefined,
          d_phiTotRhoIn,
          *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
          d_phiTotDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          d_phiTotAXQuadratureIdElectro,
          d_atomNodeIdToChargeMap,
          d_bQuadValuesAllAtoms,
          d_smearedChargeQuadratureIdElectro,
          *rhoInValues,
          true,
          dftParameters::periodicX && dftParameters::periodicY && dftParameters::periodicZ && !dftParameters::pinnedNodeForPBC,
          dftParameters::smearedNuclearCharges,
          true,
          false,
          0,
          true,
          false);


			dealiiCGSolver.solve(phiTotalSolverProblem,
					dftParameters::absLinearSolverTolerance,
					dftParameters::maxLinearSolverIterations,
					dftParameters::verbosity);

			//check integral phi equals 0
      /*
			if(dftParameters::periodicX && dftParameters::periodicY && dftParameters::periodicZ && !dftParameters::pinnedNodeForPBC)
			{
				if (dftParameters::verbosity>=2)
					pcout<<"Value of integPhiIn: "<<totalCharge(d_dofHandlerPRefined,d_phiTotRhoIn)<<std::endl;
			}
      */

      std::map<dealii::CellId,std::vector<double> > dummy;
      interpolateElectroNodalDataToQuadratureDataGeneral(d_matrixFreeDataPRefined,
          d_phiTotDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          d_phiTotRhoIn,
          d_phiInValues,
          dummy);

			{
				if(dftParameters::xcFamilyType=="LDA")
				{
					computing_timer.enter_section("VEff Computation");
#ifdef DFTFE_WITH_GPU
					if (dftParameters::useGPU)
						kohnShamDFTEigenOperatorCUDA.computeVEff(rhoInValues, d_phiInValues,d_pseudoVLoc, d_rhoCore, d_lpspQuadratureId);
#endif
					if (!dftParameters::useGPU)
						kohnShamDFTEigenOperator.computeVEff(rhoInValues, d_phiInValues, d_pseudoVLoc, d_rhoCore, d_lpspQuadratureId);
					computing_timer.exit_section("VEff Computation");
				}
				else if (dftParameters::xcFamilyType=="GGA")
				{
					computing_timer.enter_section("VEff Computation");
#ifdef DFTFE_WITH_GPU
					if (dftParameters::useGPU)
						kohnShamDFTEigenOperatorCUDA.computeVEff(rhoInValues, gradRhoInValues, d_phiInValues, d_pseudoVLoc, d_rhoCore, d_gradRhoCore, d_lpspQuadratureId);
#endif
					if (!dftParameters::useGPU)
						kohnShamDFTEigenOperator.computeVEff(rhoInValues, gradRhoInValues, d_phiInValues, d_pseudoVLoc, d_rhoCore, d_gradRhoCore, d_lpspQuadratureId);
					computing_timer.exit_section("VEff Computation");
				}

				for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
				{
#ifdef DFTFE_WITH_GPU
					if (dftParameters::useGPU)
						kohnShamDFTEigenOperatorCUDA.reinitkPointSpinIndex(kPoint,0);
#endif
					if (!dftParameters::useGPU)
						kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint,0);

					computing_timer.enter_section("Hamiltonian Matrix Computation");
#ifdef DFTFE_WITH_GPU
					if (dftParameters::useGPU)
						kohnShamDFTEigenOperatorCUDA.computeHamiltonianMatrix(kPoint,0);
#endif
					if (!dftParameters::useGPU)
						kohnShamDFTEigenOperator.computeHamiltonianMatrix(kPoint,0);
					computing_timer.exit_section("Hamiltonian Matrix Computation");

					if (dftParameters::verbosity>=4)
						dftUtils::printCurrentMemoryUsage(mpi_communicator,
								"Hamiltonian Matrix computed");
#ifdef DFTFE_WITH_GPU
          if (dftParameters::useGPU)
            kohnShamEigenSpaceOnlyRRCompute(0,
                kPoint,
                kohnShamDFTEigenOperatorCUDA,
                d_elpaScala,
                d_subspaceIterationSolverCUDA,
                true,
                true);
          else
            kohnShamEigenSpaceOnlyRRCompute(0,
                kPoint,
                kohnShamDFTEigenOperator,
                d_elpaScala,
                d_subspaceIterationSolver,
                true,
                true);            
#else  
          kohnShamEigenSpaceOnlyRRCompute(0,
              kPoint,
              kohnShamDFTEigenOperator,
              d_elpaScala,
              d_subspaceIterationSolver,
              true,
              true);              
#endif
				}

				//
				//fermi energy
				//
				if (dftParameters::constraintMagnetization)
					compute_fermienergy_constraintMagnetization(eigenValues) ;
				else
					compute_fermienergy(eigenValues,
							numElectrons);


				if(dftParameters::verbosity>=1)
				{
					pcout  << "Fermi Energy computed: "<<fermiEnergy<<std::endl;
				} 
			}
			computing_timer.enter_section("compute rho");


#ifdef DFTFE_WITH_GPU
			compute_rhoOut(kohnShamDFTEigenOperatorCUDA,
					true,
					true);
#else
			compute_rhoOut(true,
					true);
#endif
			computing_timer.exit_section("compute rho");


			computing_timer.exit_section("Density perturbation computation");
			computingTimerStandard.exit_section("Density perturbation computation");

      d_eigenVectorsFlattenedSTL=eigenVectorsFlattenedSTLTemp;

			if (!kohnShamDFTOperatorsInitialized || true)
				finalizeKohnShamDFTOperator(kohnShamDFTEigenOperator
#ifdef DFTFE_WITH_GPU
						,
						kohnShamDFTEigenOperatorCUDA
#endif
						);
		}


	//
	//dft solve
	//
	template<unsigned int FEOrder,unsigned int FEOrderElectro>
		void dftClass<FEOrder,FEOrderElectro>::solve(const bool kohnShamDFTOperatorsInitialized,
				const bool computeForces,
				const bool solveLinearizedKS,
				const bool isRestartGroundStateCalcFromChk)
		{
			kohnShamDFTOperatorClass<FEOrder,FEOrderElectro> kohnShamDFTEigenOperator(this,mpi_communicator);
#ifdef DFTFE_WITH_GPU
			kohnShamDFTOperatorCUDAClass<FEOrder,FEOrderElectro> kohnShamDFTEigenOperatorCUDA(this,mpi_communicator);
#endif

		  const Quadrature<3> &  quadrature=matrix_free_data.get_quadrature(d_densityQuadratureId);

			//computingTimerStandard.enter_section("Total scf solve");
			computingTimerStandard.enter_section("Kohn-sham dft operator init");
			energyCalculator energyCalc(mpi_communicator, interpoolcomm,interBandGroupComm);
			DensityCalculator<FEOrder,FEOrderElectro> densityCalc;



			//set up linear solver
			dealiiLinearSolver dealiiCGSolver(mpi_communicator, dealiiLinearSolver::CG);

			//set up solver functions for Poisson
			poissonSolverProblem<FEOrder,FEOrderElectro> phiTotalSolverProblem(mpi_communicator);
 
			//
			//set up solver functions for Helmholtz to be used only when Kerker mixing is on
			//use higher polynomial order dofHandler
			//
			kerkerSolverProblem<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()> kerkerPreconditionedResidualSolverProblem(mpi_communicator);
			if(dftParameters::mixingMethod=="ANDERSON_WITH_KERKER")
				kerkerPreconditionedResidualSolverProblem.init(d_matrixFreeDataPRefined,
						d_constraintsForHelmholtzRhoNodal,
						d_preCondResidualVector,
						dftParameters::kerkerParameter,
            d_helmholtzDofHandlerIndexElectro,
            d_densityQuadratureIdElectro);

			if (!kohnShamDFTOperatorsInitialized || true)
				initializeKohnShamDFTOperator(kohnShamDFTEigenOperator
#ifdef DFTFE_WITH_GPU
						,
						kohnShamDFTEigenOperatorCUDA
#endif
						);
			else
				reInitializeKohnShamDFTOperator(kohnShamDFTEigenOperator
#ifdef DFTFE_WITH_GPU
						,
						kohnShamDFTEigenOperatorCUDA
#endif
						);

			//
			//precompute shapeFunctions and shapeFunctionGradients and shapeFunctionGradientIntegrals
			//
			computing_timer.enter_section("shapefunction data");
			if (!dftParameters::useGPU)
				kohnShamDFTEigenOperator.preComputeShapeFunctionGradientIntegrals(d_lpspQuadratureId);
#ifdef DFTFE_WITH_GPU
			if (dftParameters::useGPU)
				kohnShamDFTEigenOperatorCUDA.preComputeShapeFunctionGradientIntegrals(d_lpspQuadratureId);
#endif
			computing_timer.exit_section("shapefunction data");


			if (dftParameters::verbosity>=4)
				dftUtils::printCurrentMemoryUsage(mpi_communicator,
						"Precompute shapefunction grad integrals, just before starting scf solve");

			if (dftParameters::verbosity>=4)
				dftUtils::printCurrentMemoryUsage(mpi_communicator,
						"Kohn-sham dft operator init called");

			computingTimerStandard.exit_section("Kohn-sham dft operator init");

			//
			//solve vself in bins
			//
      computing_timer.enter_section("Nuclear self-potential solve");
      computingTimerStandard.enter_section("Nuclear self-potential solve");
#ifdef DFTFE_WITH_GPU
      if (dftParameters::useGPU)
        d_vselfBinsManager.solveVselfInBinsGPU(d_matrixFreeDataPRefined,
            d_baseDofHandlerIndexElectro,
            d_phiTotAXQuadratureIdElectro,
            d_binsStartDofHandlerIndexElectro,
            kohnShamDFTEigenOperatorCUDA,
            d_constraintsPRefined,
            d_imagePositionsTrunc,
            d_imageIdsTrunc,
            d_imageChargesTrunc,
            d_localVselfs,
            d_bQuadValuesAllAtoms,
            d_bQuadAtomIdsAllAtoms,
            d_bQuadAtomIdsAllAtomsImages,
            d_bCellNonTrivialAtomIds,
            d_bCellNonTrivialAtomIdsBins,
            d_smearedChargeWidths,
            d_smearedChargeScaling,
            d_smearedChargeQuadratureIdElectro,
            dftParameters::smearedNuclearCharges);
      else
        d_vselfBinsManager.solveVselfInBins(d_matrixFreeDataPRefined,
            d_binsStartDofHandlerIndexElectro,
            d_phiTotAXQuadratureIdElectro,
            d_constraintsPRefined,
            d_imagePositionsTrunc,
            d_imageIdsTrunc,
            d_imageChargesTrunc,
            d_localVselfs,
            d_bQuadValuesAllAtoms,
            d_bQuadAtomIdsAllAtoms,
            d_bQuadAtomIdsAllAtomsImages,
            d_bCellNonTrivialAtomIds,
            d_bCellNonTrivialAtomIdsBins,
            d_smearedChargeWidths,
            d_smearedChargeScaling,
            d_smearedChargeQuadratureIdElectro,
            dftParameters::smearedNuclearCharges);
#else
      d_vselfBinsManager.solveVselfInBins(d_matrixFreeDataPRefined,
          d_binsStartDofHandlerIndexElectro,
          d_phiTotAXQuadratureIdElectro,
          d_constraintsPRefined,
          d_imagePositionsTrunc,
          d_imageIdsTrunc,
          d_imageChargesTrunc,
          d_localVselfs,
          d_bQuadValuesAllAtoms,
          d_bQuadAtomIdsAllAtoms,
          d_bQuadAtomIdsAllAtomsImages,
          d_bCellNonTrivialAtomIds,
          d_bCellNonTrivialAtomIdsBins,
          d_smearedChargeWidths,
          d_smearedChargeScaling,
          d_smearedChargeQuadratureIdElectro,
          dftParameters::smearedNuclearCharges);
#endif
      computingTimerStandard.exit_section("Nuclear self-potential solve");
      computing_timer.exit_section("Nuclear self-potential solve");

			if((dftParameters::isPseudopotential || dftParameters::smearedNuclearCharges))
			{
        computingTimerStandard.enter_section("Init local PSP");
				initLocalPseudoPotential(d_dofHandlerPRefined,
					  d_lpspQuadratureIdElectro,
						d_matrixFreeDataPRefined,
						d_phiExtDofHandlerIndexElectro,
						d_constraintsPRefinedOnlyHanging,
						d_supportPointsPRefined,
						d_vselfBinsManager,
            d_phiExt,
						d_pseudoVLoc,
						d_pseudoVLocAtoms);

        computingTimerStandard.exit_section("Init local PSP");
			}


			computingTimerStandard.enter_section("Total scf solve");

			//
			//solve
			//
			computing_timer.enter_section("scf solve");

			double firstScfChebyTol=dftParameters::mixingMethod=="ANDERSON_WITH_KERKER"?1e-2:2e-2;

			if (dftParameters::isBOMD && dftParameters::isXLBOMD && solveLinearizedKS)
				firstScfChebyTol=dftParameters::chebyshevFilterTolXLBOMD;
			else if (dftParameters::isBOMD)
				firstScfChebyTol=dftParameters::chebyshevTolerance>1e-4?1e-4:dftParameters::chebyshevTolerance;
			else if (dftParameters::isIonOpt || dftParameters::isCellOpt)
				firstScfChebyTol=dftParameters::chebyshevTolerance>1e-3?1e-3:dftParameters::chebyshevTolerance;

			//
			//Begin SCF iteration
			//
			unsigned int scfIter=0;
			double norm = 1.0;
			//CAUTION: Choosing a looser tolerance might lead to failed tests
			const double adaptiveChebysevFilterPassesTol = dftParameters::chebyshevTolerance;
			bool scfConverged=false;
			pcout<<std::endl;
			if (dftParameters::verbosity==0)
				pcout<<"Starting SCF iterations...."<<std::endl;
			while ((norm > dftParameters::selfConsistentSolverTolerance) && (scfIter < dftParameters::numSCFIterations))
			{

				dealii::Timer local_timer(MPI_COMM_WORLD,true);
				if (dftParameters::verbosity>=1)
					pcout<<"************************Begin Self-Consistent-Field Iteration: "<<std::setw(2)<<scfIter+1<<" ***********************"<<std::endl;
				//
				//Mixing scheme
				//
				computing_timer.enter_section("density mixing");
				if(scfIter > 0 && !(isRestartGroundStateCalcFromChk && dftParameters::chkType==2))
				{
					if (scfIter==1)
					{
						if (dftParameters::spinPolarized==1)
						{
							norm = sqrt(mixing_simple_spinPolarized());
						}
						else
						{
							if(dftParameters::mixingMethod=="ANDERSON_WITH_KERKER")
								norm = sqrt(nodalDensity_mixing_simple(kerkerPreconditionedResidualSolverProblem,
											dealiiCGSolver));
							else
								norm = sqrt(mixing_simple());
						}

						if (dftParameters::verbosity>=1)
							pcout<<"Simple mixing, L2 norm of electron-density difference: "<< norm<< std::endl;
					}
					else
					{
						if (dftParameters::spinPolarized==1)
						{
							if (dftParameters::mixingMethod=="ANDERSON" )
								norm = sqrt(mixing_anderson_spinPolarized());
							else if (dftParameters::mixingMethod=="BROYDEN" )
								norm = sqrt(mixing_broyden_spinPolarized());
							else if (dftParameters::mixingMethod=="ANDERSON_WITH_KERKER")
								AssertThrow(false,ExcMessage("Kerker is not implemented for spin-polarized problems yet"));
						}
						else
						{
							if(dftParameters::mixingMethod=="ANDERSON")
								norm = sqrt(mixing_anderson());
							else if(dftParameters::mixingMethod=="BROYDEN")
								norm = sqrt(mixing_broyden());
							else if(dftParameters::mixingMethod=="ANDERSON_WITH_KERKER")
								norm = sqrt(nodalDensity_mixing_anderson(kerkerPreconditionedResidualSolverProblem,
											dealiiCGSolver));
						}

						if (dftParameters::verbosity>=1)
							pcout<<"Anderson mixing, L2 norm of electron-density difference: "<< norm<< std::endl;
					}

					if (dftParameters::computeEnergyEverySCF && d_numEigenValuesRR==d_numEigenValues)
						d_phiTotRhoIn = d_phiTotRhoOut;
				}
				else if (isRestartGroundStateCalcFromChk && dftParameters::chkType==2)
				{
					if (dftParameters::spinPolarized==1)
					{
						if (dftParameters::mixingMethod=="ANDERSON")
							norm = sqrt(mixing_anderson_spinPolarized());
						else if (dftParameters::mixingMethod=="BROYDEN")
							norm = sqrt(mixing_broyden_spinPolarized());
						else if (dftParameters::mixingMethod=="ANDERSON_WITH_KERKER")
							AssertThrow(false,ExcMessage("Kerker is not implemented for spin-polarized problems"));
					}
					else
						if(dftParameters::mixingMethod.compare("ANDERSON_WITH_KERKER"))
							norm = sqrt(nodalDensity_mixing_anderson(kerkerPreconditionedResidualSolverProblem,
										dealiiCGSolver));
						else if (dftParameters::mixingMethod=="ANDERSON")
							norm = sqrt(mixing_anderson());
						else if (dftParameters::mixingMethod=="BROYDEN")
							norm = sqrt(mixing_broyden());

					if (dftParameters::verbosity>=1)
						pcout<<"Anderson Mixing, L2 norm of electron-density difference: "<< norm<< std::endl;
					if (dftParameters::computeEnergyEverySCF && d_numEigenValuesRR==d_numEigenValues)
						d_phiTotRhoIn = d_phiTotRhoOut;
				}
				computing_timer.exit_section("density mixing");

				if (!(norm > dftParameters::selfConsistentSolverTolerance))
					scfConverged=true;
				//
				//phiTot with rhoIn
				//
				if (dftParameters::verbosity>=2)
					pcout<< std::endl<<"Poisson solve for total electrostatic potential (rhoIn+b): ";
				

				if (scfIter>0)
					phiTotalSolverProblem.reinit(d_matrixFreeDataPRefined,
							d_phiTotRhoIn,
							*d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
							d_phiTotDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
              d_phiTotAXQuadratureIdElectro,
							d_atomNodeIdToChargeMap,
							d_bQuadValuesAllAtoms,
              d_smearedChargeQuadratureIdElectro,
							*rhoInValues,
							false,
							false,
							dftParameters::smearedNuclearCharges,
              true,
              false,
              0,
              false,
              true);          
				else
					phiTotalSolverProblem.reinit(d_matrixFreeDataPRefined,
							d_phiTotRhoIn,
							*d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
							d_phiTotDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
              d_phiTotAXQuadratureIdElectro,
							d_atomNodeIdToChargeMap,
							d_bQuadValuesAllAtoms,
              d_smearedChargeQuadratureIdElectro,
							*rhoInValues,
							true,
							dftParameters::periodicX && dftParameters::periodicY && dftParameters::periodicZ && !dftParameters::pinnedNodeForPBC,
							dftParameters::smearedNuclearCharges,
              true,
              false,
              0,
              true,
              false);

				computing_timer.enter_section("phiTot solve");

				dealiiCGSolver.solve(phiTotalSolverProblem,
						dftParameters::absLinearSolverTolerance,
						dftParameters::maxLinearSolverIterations,
						dftParameters::verbosity);

        std::map<dealii::CellId,std::vector<double> > dummy;
        interpolateElectroNodalDataToQuadratureDataGeneral(d_matrixFreeDataPRefined,
            d_phiTotDofHandlerIndexElectro,
            d_densityQuadratureIdElectro,
            d_phiTotRhoIn,
            d_phiInValues,
            dummy);

				//
				//impose integral phi equals 0
				//
        /*
				if(dftParameters::periodicX && dftParameters::periodicY && dftParameters::periodicZ && !dftParameters::pinnedNodeForPBC)
				{
					if (dftParameters::verbosity>=2)
						pcout<<"Value of integPhiIn: "<<totalCharge(d_dofHandlerPRefined,d_phiTotRhoIn)<<std::endl;
				}
        */

				computing_timer.exit_section("phiTot solve");

				unsigned int numberChebyshevSolvePasses=0;
				//
				//eigen solve
				//
				if (dftParameters::spinPolarized==1)
				{

					std::vector<std::vector<std::vector<double> > >
						eigenValuesSpins(2,
								std::vector<std::vector<double> >(d_kPointWeights.size(),
									std::vector<double>((scfIter<dftParameters::spectrumSplitStartingScfIter || scfConverged)?
										d_numEigenValues:d_numEigenValuesRR)));

					std::vector<std::vector<std::vector<double>>>
						residualNormWaveFunctionsAllkPointsSpins
						(2,
						 std::vector<std::vector<double> >(d_kPointWeights.size(),
							 std::vector<double>((scfIter<dftParameters::spectrumSplitStartingScfIter || scfConverged)?
								 d_numEigenValues:d_numEigenValuesRR)));

					for(unsigned int s=0; s<2; ++s)
					{
						if(dftParameters::xcFamilyType=="LDA")
						{
							computing_timer.enter_section("VEff Computation");
#ifdef DFTFE_WITH_GPU
							if (dftParameters::useGPU)
								kohnShamDFTEigenOperatorCUDA.computeVEffSpinPolarized(rhoInValuesSpinPolarized, d_phiInValues, s, d_pseudoVLoc, d_rhoCore, d_lpspQuadratureId);
#endif
							if (!dftParameters::useGPU)
								kohnShamDFTEigenOperator.computeVEffSpinPolarized(rhoInValuesSpinPolarized, d_phiInValues, s, d_pseudoVLoc, d_rhoCore, d_lpspQuadratureId);
							computing_timer.exit_section("VEff Computation");
						}
						else if (dftParameters::xcFamilyType=="GGA")
						{
							computing_timer.enter_section("VEff Computation");
#ifdef DFTFE_WITH_GPU
							if (dftParameters::useGPU)
								kohnShamDFTEigenOperatorCUDA.computeVEffSpinPolarized(rhoInValuesSpinPolarized, gradRhoInValuesSpinPolarized, d_phiInValues, s, d_pseudoVLoc, d_rhoCore, d_gradRhoCore, d_lpspQuadratureId);
#endif
							if (!dftParameters::useGPU)
								kohnShamDFTEigenOperator.computeVEffSpinPolarized(rhoInValuesSpinPolarized, gradRhoInValuesSpinPolarized, d_phiInValues, s, d_pseudoVLoc, d_rhoCore, d_gradRhoCore, d_lpspQuadratureId);
							computing_timer.exit_section("VEff Computation");
						}
						for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
						{
#ifdef DFTFE_WITH_GPU
							if (dftParameters::useGPU)
								kohnShamDFTEigenOperatorCUDA.reinitkPointSpinIndex(kPoint,s);
#endif
							if (!dftParameters::useGPU)
								kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint,s);

							computing_timer.enter_section("Hamiltonian Matrix Computation");
#ifdef DFTFE_WITH_GPU
							if (dftParameters::useGPU)
								kohnShamDFTEigenOperatorCUDA.computeHamiltonianMatrix(kPoint,s);
#endif
							if (!dftParameters::useGPU)
								kohnShamDFTEigenOperator.computeHamiltonianMatrix(kPoint,s);
							computing_timer.exit_section("Hamiltonian Matrix Computation");

							if (dftParameters::verbosity>=4)
								dftUtils::printCurrentMemoryUsage(mpi_communicator,
										"Hamiltonian Matrix computed");

							for(unsigned int j = 0; j < 1; ++j)
							{
								if (dftParameters::verbosity>=2)
								{
									if (dftParameters::numberPassesRRSkippedXLBOMD>0)
										pcout<<"Beginning no RR XL-BOMD Chebyshev filter passes with total such passes: "<< dftParameters::numberPassesRRSkippedXLBOMD<< " for spin "<< s+1<<std::endl;
									else
										pcout<<"Beginning Chebyshev filter pass "<< j+1<< " for spin "<< s+1<<std::endl;
								}

#ifdef DFTFE_WITH_GPU
								if (dftParameters::useGPU)
									kohnShamEigenSpaceCompute(s,
											kPoint,
											kohnShamDFTEigenOperatorCUDA,
											d_elpaScala,
											d_subspaceIterationSolverCUDA,
											residualNormWaveFunctionsAllkPointsSpins[s][kPoint],
                      (scfIter==0 || allowMultipleFilteringPassesAfterFirstScf)?true:false,
											dftParameters::numberPassesRRSkippedXLBOMD,
											(scfIter<dftParameters::spectrumSplitStartingScfIter || scfConverged)?false:true,
											scfConverged?false:true,
											scfIter==0);
#endif
								if (!dftParameters::useGPU)
									kohnShamEigenSpaceCompute(s,
											kPoint,
											kohnShamDFTEigenOperator,
											d_elpaScala,
											d_subspaceIterationSolver,
											residualNormWaveFunctionsAllkPointsSpins[s][kPoint],
                      (scfIter==0 || allowMultipleFilteringPassesAfterFirstScf)?true:false,                      
											(scfIter<dftParameters::spectrumSplitStartingScfIter || scfConverged)?false:true,
											scfConverged?false:true,
											scfIter==0);
							}
						}
					}

					if (!(dftParameters::numberPassesRRSkippedXLBOMD>0))
					{
						for(unsigned int s=0; s<2; ++s)
							for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
							{
								if (scfIter<dftParameters::spectrumSplitStartingScfIter || scfConverged)
									for (unsigned int i = 0; i<d_numEigenValues; ++i)
										eigenValuesSpins[s][kPoint][i]=eigenValues[kPoint][d_numEigenValues*s+i];
								else
									for (unsigned int i = 0; i<d_numEigenValuesRR; ++i)
										eigenValuesSpins[s][kPoint][i]=eigenValuesRRSplit[kPoint][d_numEigenValuesRR*s+i];
							}
						//
						//fermi energy
						//
						if (dftParameters::constraintMagnetization)
							compute_fermienergy_constraintMagnetization(eigenValues) ;
						else
							compute_fermienergy(eigenValues,
									numElectrons);
					}

					unsigned int count=(dftParameters::numberPassesRRSkippedXLBOMD>0)?numberPassesRRSkippedXLBOMD:1;

					if (!scfConverged && (scfIter==0 || allowMultipleFilteringPassesAfterFirstScf))
					{

						//maximum of the residual norm of the state closest to and below the Fermi level among all k points,
						//and also the maximum between the two spins
						double maxRes =(dftParameters::numberPassesRRSkippedXLBOMD>0)?1e+6:std::max(computeMaximumHighestOccupiedStateResidualNorm
								(residualNormWaveFunctionsAllkPointsSpins[0],
								 eigenValuesSpins[0],
								 fermiEnergy),
								computeMaximumHighestOccupiedStateResidualNorm
								(residualNormWaveFunctionsAllkPointsSpins[1],
								 eigenValuesSpins[1],
								 fermiEnergy));

						if (dftParameters::verbosity>=2 && !(dftParameters::numberPassesRRSkippedXLBOMD>0))
						{
							pcout << "Maximum residual norm of the state closest to and below Fermi level: "<< maxRes << std::endl;
						}

						//if the residual norm is greater than adaptiveChebysevFilterPassesTol (a heuristic value)
						// do more passes of chebysev filter till the check passes.
						// This improves the scf convergence performance.

						const double filterPassTol=(scfIter==0
								&& isRestartGroundStateCalcFromChk
								&& (dftParameters::chkType==2 || dftParameters::chkType==3))? 1.0e-8
							:((scfIter==0 && adaptiveChebysevFilterPassesTol>firstScfChebyTol)?firstScfChebyTol:adaptiveChebysevFilterPassesTol);
						while (maxRes>filterPassTol && count<100)
						{
							for(unsigned int s=0; s<2; ++s)
							{
								for(unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
								{
									if (dftParameters::verbosity>=2)
										pcout<< "Beginning Chebyshev filter pass "<< 1+count<< " for spin "<< s+1<<std::endl;;

#ifdef DFTFE_WITH_GPU
									if (dftParameters::useGPU)
										kohnShamDFTEigenOperatorCUDA.reinitkPointSpinIndex(kPoint,s);
#endif
									if (!dftParameters::useGPU)
										kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint,s);

#ifdef DFTFE_WITH_GPU
									if (dftParameters::useGPU)
										kohnShamEigenSpaceCompute(s,
												kPoint,
												kohnShamDFTEigenOperatorCUDA,
												d_elpaScala,
												d_subspaceIterationSolverCUDA,
												residualNormWaveFunctionsAllkPointsSpins[s][kPoint],
                        true,
												0, 
												(scfIter<dftParameters::spectrumSplitStartingScfIter)?false:true,
												true,
												scfIter==0);
#endif
									if (!dftParameters::useGPU)
										kohnShamEigenSpaceCompute(s,
												kPoint,
												kohnShamDFTEigenOperator,
												d_elpaScala,
												d_subspaceIterationSolver,
												residualNormWaveFunctionsAllkPointsSpins[s][kPoint],
                        true,
												(scfIter<dftParameters::spectrumSplitStartingScfIter)?false:true,
												true,
												scfIter==0);

								}
							}

							for(unsigned int s=0; s<2; ++s)
								for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
								{
									if (scfIter<dftParameters::spectrumSplitStartingScfIter || scfConverged)
										for (unsigned int i = 0; i<d_numEigenValues; ++i)
											eigenValuesSpins[s][kPoint][i]=eigenValues[kPoint][d_numEigenValues*s+i];
									else
										for (unsigned int i = 0; i<d_numEigenValuesRR; ++i)
											eigenValuesSpins[s][kPoint][i]=eigenValuesRRSplit[kPoint][d_numEigenValuesRR*s+i];
								}
							//
							if (dftParameters::constraintMagnetization)
								compute_fermienergy_constraintMagnetization(eigenValues) ;
							else
								compute_fermienergy(eigenValues,
										numElectrons);
							//
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

						count++;
					}

					if(dftParameters::verbosity>=1)
					{
						pcout  << "Fermi Energy computed: "<<fermiEnergy<<std::endl;
					}

					numberChebyshevSolvePasses=count;
				}
				else
				{

					std::vector<std::vector<double>> residualNormWaveFunctionsAllkPoints;
					residualNormWaveFunctionsAllkPoints.resize(d_kPointWeights.size());
					for(unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
						residualNormWaveFunctionsAllkPoints[kPoint].resize((scfIter<dftParameters::spectrumSplitStartingScfIter || scfConverged)?d_numEigenValues:d_numEigenValuesRR);

					if(dftParameters::xcFamilyType=="LDA")
					{
						computing_timer.enter_section("VEff Computation");
#ifdef DFTFE_WITH_GPU
						if (dftParameters::useGPU)
							kohnShamDFTEigenOperatorCUDA.computeVEff(rhoInValues, d_phiInValues, d_pseudoVLoc, d_rhoCore, d_lpspQuadratureId);
#endif
						if (!dftParameters::useGPU)
							kohnShamDFTEigenOperator.computeVEff(rhoInValues, d_phiInValues, d_pseudoVLoc, d_rhoCore, d_lpspQuadratureId);
						computing_timer.exit_section("VEff Computation");
					}
					else if (dftParameters::xcFamilyType=="GGA")
					{
						computing_timer.enter_section("VEff Computation");
#ifdef DFTFE_WITH_GPU
						if (dftParameters::useGPU)
							kohnShamDFTEigenOperatorCUDA.computeVEff(rhoInValues, gradRhoInValues, d_phiInValues, d_pseudoVLoc, d_rhoCore, d_gradRhoCore, d_lpspQuadratureId);
#endif
						if (!dftParameters::useGPU)
							kohnShamDFTEigenOperator.computeVEff(rhoInValues, gradRhoInValues, d_phiInValues, d_pseudoVLoc, d_rhoCore, d_gradRhoCore, d_lpspQuadratureId);
						computing_timer.exit_section("VEff Computation");
					}

					for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
					{
#ifdef DFTFE_WITH_GPU
						if (dftParameters::useGPU)
							kohnShamDFTEigenOperatorCUDA.reinitkPointSpinIndex(kPoint,0);
#endif
						if (!dftParameters::useGPU)
							kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint,0);

						computing_timer.enter_section("Hamiltonian Matrix Computation");
#ifdef DFTFE_WITH_GPU
						if (dftParameters::useGPU)
							kohnShamDFTEigenOperatorCUDA.computeHamiltonianMatrix(kPoint,0);
#endif
						if (!dftParameters::useGPU)
							kohnShamDFTEigenOperator.computeHamiltonianMatrix(kPoint,0);
						computing_timer.exit_section("Hamiltonian Matrix Computation");

						if (dftParameters::verbosity>=4)
							dftUtils::printCurrentMemoryUsage(mpi_communicator,
									"Hamiltonian Matrix computed");

						for(unsigned int j = 0; j < 1; ++j)
						{
							if (dftParameters::verbosity>=2)
							{
								if (dftParameters::numberPassesRRSkippedXLBOMD>0)
									pcout<<"Beginning no RR XL-BOMD Chebyshev filter passes with total such passes: "<< dftParameters::numberPassesRRSkippedXLBOMD<<std::endl;
								else
									pcout<< "Beginning Chebyshev filter pass "<< j+1<<std::endl;
							}


#ifdef DFTFE_WITH_GPU
							if (dftParameters::useGPU)
								kohnShamEigenSpaceCompute(0,
										kPoint,
										kohnShamDFTEigenOperatorCUDA,
										d_elpaScala,
										d_subspaceIterationSolverCUDA,
										residualNormWaveFunctionsAllkPoints[kPoint],
                    (scfIter==0 || allowMultipleFilteringPassesAfterFirstScf)?true:false,                    
										dftParameters::numberPassesRRSkippedXLBOMD,
										(scfIter<dftParameters::spectrumSplitStartingScfIter || scfConverged)?false:true,
										scfConverged?false:true,
										scfIter==0);
#endif
							if (!dftParameters::useGPU)
								kohnShamEigenSpaceCompute(0,
										kPoint,
										kohnShamDFTEigenOperator,
										d_elpaScala,
										d_subspaceIterationSolver,
										residualNormWaveFunctionsAllkPoints[kPoint],
                    (scfIter==0 || allowMultipleFilteringPassesAfterFirstScf)?true:false,                      
										(scfIter<dftParameters::spectrumSplitStartingScfIter || scfConverged)?false:true,
										scfConverged?false:true,
										scfIter==0);
						}
					}

					if (!(dftParameters::numberPassesRRSkippedXLBOMD>0))
					{
						//
						//fermi energy
						//
						if (dftParameters::constraintMagnetization)
							compute_fermienergy_constraintMagnetization(eigenValues) ;
						else
							compute_fermienergy(eigenValues,
									numElectrons);
					}

					unsigned int count=(dftParameters::numberPassesRRSkippedXLBOMD>0)?dftParameters::numberPassesRRSkippedXLBOMD:1;

					if (!scfConverged && (scfIter==0 || allowMultipleFilteringPassesAfterFirstScf))
					{
						//
						//maximum of the residual norm of the state closest to and below the Fermi level among all k points
						//
						double maxRes = (dftParameters::numberPassesRRSkippedXLBOMD>0)?1e+6:computeMaximumHighestOccupiedStateResidualNorm
							(residualNormWaveFunctionsAllkPoints,
							 (scfIter<dftParameters::spectrumSplitStartingScfIter)?eigenValues:eigenValuesRRSplit,
							 fermiEnergy);
						if (dftParameters::verbosity>=2 && !(dftParameters::numberPassesRRSkippedXLBOMD>0))
							pcout << "Maximum residual norm of the state closest to and below Fermi level: "<< maxRes << std::endl;

						//if the residual norm is greater than adaptiveChebysevFilterPassesTol (a heuristic value)
						// do more passes of chebysev filter till the check passes.
						// This improves the scf convergence performance.

						const double filterPassTol=(scfIter==0
								&& isRestartGroundStateCalcFromChk
								&& (dftParameters::chkType==2 || dftParameters::chkType==3))? 1.0e-8
							:((scfIter==0 && adaptiveChebysevFilterPassesTol>firstScfChebyTol)?firstScfChebyTol:adaptiveChebysevFilterPassesTol);
						while (maxRes>filterPassTol && count<100)
						{

							for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
							{
								if (dftParameters::verbosity>=2)
									pcout<< "Beginning Chebyshev filter pass "<< 1+count<<std::endl;

#ifdef DFTFE_WITH_GPU
								if (dftParameters::useGPU)
									kohnShamDFTEigenOperatorCUDA.reinitkPointSpinIndex(kPoint,0);
#endif
								if (!dftParameters::useGPU)
									kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint,0);

#ifdef DFTFE_WITH_GPU
								if (dftParameters::useGPU)
									kohnShamEigenSpaceCompute(0,
											kPoint,
											kohnShamDFTEigenOperatorCUDA,
											d_elpaScala,
											d_subspaceIterationSolverCUDA,
											residualNormWaveFunctionsAllkPoints[kPoint],
                      true,
											0,
											(scfIter<dftParameters::spectrumSplitStartingScfIter)?false:true,
											true,
											scfIter==0);

#endif
								if (!dftParameters::useGPU)
									kohnShamEigenSpaceCompute(0,
											kPoint,
											kohnShamDFTEigenOperator,
											d_elpaScala,
											d_subspaceIterationSolver,
											residualNormWaveFunctionsAllkPoints[kPoint],
                      true,
											(scfIter<dftParameters::spectrumSplitStartingScfIter)?false:true,
											true,
											scfIter==0);

							}

							//
							if (dftParameters::constraintMagnetization)
								compute_fermienergy_constraintMagnetization(eigenValues) ;
							else
								compute_fermienergy(eigenValues,
										numElectrons);
							//
							maxRes = computeMaximumHighestOccupiedStateResidualNorm
								(residualNormWaveFunctionsAllkPoints,
								 (scfIter<dftParameters::spectrumSplitStartingScfIter || scfConverged)?eigenValues:eigenValuesRRSplit,
								 fermiEnergy);
							if (dftParameters::verbosity>=2)
								pcout << "Maximum residual norm of the state closest to and below Fermi level: "<< maxRes << std::endl;

							count++;
						}
					}

					numberChebyshevSolvePasses=count;

					if(dftParameters::verbosity>=1)
					{
						pcout  << "Fermi Energy computed: "<<fermiEnergy<<std::endl;
					}
				}
				computing_timer.enter_section("compute rho");
#ifdef USE_COMPLEX
				if(dftParameters::useSymm)
        {
					symmetryPtr->computeLocalrhoOut();
					symmetryPtr->computeAndSymmetrize_rhoOut();
         
          std::function<double(const typename dealii::DoFHandler<3>::active_cell_iterator & cell ,
              const unsigned int q)> funcRho =
            [&](const typename dealii::DoFHandler<3>::active_cell_iterator & cell ,
                const unsigned int q)
            {return (*rhoOutValues).find(cell->id())->second[q];};
          dealii::VectorTools::project<3,distributedCPUVec<double>> (dealii::MappingQ1<3,3>(),
              d_dofHandlerRhoNodal,
              d_constraintsRhoNodal,
              d_matrixFreeDataPRefined.get_quadrature(d_densityQuadratureIdElectro),
              funcRho,
              d_rhoOutNodalValues);
          d_rhoOutNodalValues.update_ghost_values();

          interpolateRhoNodalDataToQuadratureDataLpsp(d_matrixFreeDataPRefined,
              d_densityDofHandlerIndexElectro,
              d_lpspQuadratureIdElectro,
              d_rhoOutNodalValues,
              d_rhoOutValuesLpspQuad,
              d_gradRhoOutValuesLpspQuad,
              true);           
				}
				else
					compute_rhoOut((scfIter<dftParameters::spectrumSplitStartingScfIter || scfConverged)?false:true,
							scfConverged || (scfIter == (dftParameters::numSCFIterations-1)) || solveLinearizedKS);
#else

#ifdef DFTFE_WITH_GPU
				compute_rhoOut(kohnShamDFTEigenOperatorCUDA,
						(scfIter<dftParameters::spectrumSplitStartingScfIter || scfConverged)?false:true,
						scfConverged || (scfIter == (dftParameters::numSCFIterations-1)) || solveLinearizedKS);
#else
				compute_rhoOut((scfIter<dftParameters::spectrumSplitStartingScfIter || scfConverged)?false:true,
						scfConverged || (scfIter == (dftParameters::numSCFIterations-1)) || solveLinearizedKS);
#endif
#endif
				computing_timer.exit_section("compute rho");

				//
				//compute integral rhoOut
				//
				const double integralRhoValue=totalCharge(d_dofHandlerPRefined,
						rhoOutValues);

				if (dftParameters::verbosity>=2){
					pcout<< std::endl<<"number of electrons: "<< integralRhoValue<<std::endl;
					if (dftParameters::spinPolarized==1)
						pcout<< std::endl<<"net magnetization: "<< totalMagnetization(rhoOutValuesSpinPolarized) << std::endl;
				}
				//
				//phiTot with rhoOut
				//
				if (dftParameters::computeEnergyEverySCF && d_numEigenValuesRR==d_numEigenValues)
				{
					if(dftParameters::verbosity>=2)
						pcout<< std::endl<<"Poisson solve for total electrostatic potential (rhoOut+b): ";

					computing_timer.enter_section("phiTot solve");

					phiTotalSolverProblem.reinit(d_matrixFreeDataPRefined,
							d_phiTotRhoOut,
							*d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
							d_phiTotDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
              d_phiTotAXQuadratureIdElectro,
							d_atomNodeIdToChargeMap,
							d_bQuadValuesAllAtoms,
              d_smearedChargeQuadratureIdElectro,
							*rhoOutValues,
							false,
							false,
							dftParameters::smearedNuclearCharges,
              true,
              false,
              0,
              false,
              true);              


					dealiiCGSolver.solve(phiTotalSolverProblem,
							dftParameters::absLinearSolverTolerance,
							dftParameters::maxLinearSolverIterations,
							dftParameters::verbosity);


					//
					//impose integral phi equals 0
					//
          /*
					if(dftParameters::periodicX && dftParameters::periodicY && dftParameters::periodicZ && !dftParameters::pinnedNodeForPBC)
					{
						if(dftParameters::verbosity>=2)
							pcout<<"Value of integPhiOut: "<<totalCharge(d_dofHandlerPRefined,d_phiTotRhoOut);
					}
          */

					computing_timer.exit_section("phiTot solve");

				  const Quadrature<3> &  quadrature=matrix_free_data.get_quadrature(d_densityQuadratureId);
					const double totalEnergy = dftParameters::spinPolarized==0 ?
						energyCalc.computeEnergy(d_dofHandlerPRefined,
								dofHandler,
								quadrature,
								quadrature,
							  d_matrixFreeDataPRefined.get_quadrature(d_smearedChargeQuadratureIdElectro),
                d_matrixFreeDataPRefined.get_quadrature(d_lpspQuadratureIdElectro),
								eigenValues,
								d_kPointWeights,
								fermiEnergy,
								funcX,
								funcC,
								d_phiInValues,
								d_phiTotRhoOut,
								*rhoInValues,
								*rhoOutValues,
                d_rhoOutValuesLpspQuad,
								*rhoOutValues,
                d_rhoOutValuesLpspQuad,
								*gradRhoInValues,
								*gradRhoOutValues,
						   	d_rhoCore,
							  d_gradRhoCore,	                
								d_bQuadValuesAllAtoms,
								d_localVselfs,
								d_pseudoVLoc,
								d_pseudoVLoc,
								d_atomNodeIdToChargeMap,
								atomLocations.size(),
								lowerBoundKindex,
								0,
								dftParameters::verbosity>=2,
								dftParameters::smearedNuclearCharges) :
									energyCalc.computeEnergySpinPolarized(d_dofHandlerPRefined,
									    dofHandler,
											quadrature,
											quadrature,
                      d_matrixFreeDataPRefined.get_quadrature(d_smearedChargeQuadratureIdElectro),
                      d_matrixFreeDataPRefined.get_quadrature(d_lpspQuadratureIdElectro),                    
											eigenValues,
											d_kPointWeights,
											fermiEnergy,
											fermiEnergyUp,
											fermiEnergyDown,
											funcX,
											funcC,
											d_phiInValues,
											d_phiTotRhoOut,
											*rhoInValues,
											*rhoOutValues,
                      d_rhoOutValuesLpspQuad,
											*rhoOutValues,
                      d_rhoOutValuesLpspQuad,
											*gradRhoInValues,
											*gradRhoOutValues,
											*rhoInValuesSpinPolarized,
											*rhoOutValuesSpinPolarized,
											*gradRhoInValuesSpinPolarized,
											*gradRhoOutValuesSpinPolarized,
											d_bQuadValuesAllAtoms,
											d_localVselfs,
											d_pseudoVLoc,
											d_pseudoVLoc,
											d_atomNodeIdToChargeMap,
											atomLocations.size(),
											lowerBoundKindex,
											0,
											dftParameters::verbosity>=2,
											dftParameters::smearedNuclearCharges);
					if (dftParameters::verbosity==1)
						pcout<<"Total energy  : " << totalEnergy << std::endl;
				}
				else
				{
					if (d_numEigenValuesRR!=d_numEigenValues && dftParameters::computeEnergyEverySCF && dftParameters::verbosity>=1)
						pcout<<"DFT-FE Message: energy computation is not performed at the end of each scf iteration step\n"<<"if SPECTRUM SPLIT CORE EIGENSTATES is set to a non-zero value."<< std::endl;
				}

				if (dftParameters::verbosity>=1)
					pcout<<"***********************Self-Consistent-Field Iteration: "<<std::setw(2)<<scfIter+1<<" complete**********************"<<std::endl;

				local_timer.stop();
				if (dftParameters::verbosity>=1)
					pcout << "Wall time for the above scf iteration: " << local_timer.wall_time() << " seconds\n"<<
						"Number of Chebyshev filtered subspace iterations: "<< numberChebyshevSolvePasses<<std::endl<<std::endl;
				//
				scfIter++;

				if (dftParameters::chkType==2 && scfIter%10 == 0)
					saveTriaInfoAndRhoData();


				if (dftParameters::isBOMD && dftParameters::isXLBOMD && solveLinearizedKS)
					break;
			}

			if (!(dftParameters::isBOMD && dftParameters::isXLBOMD && solveLinearizedKS))
			{
				if(scfIter==dftParameters::numSCFIterations)
					pcout<<"DFT-FE Warning: SCF iterations did not converge to the specified tolerance after: "<<scfIter<<" iterations."<<std::endl;
				else
					pcout<<"SCF iterations converged to the specified tolerance after: "<<scfIter<<" iterations."<<std::endl;
			}

			if ((!dftParameters::computeEnergyEverySCF || d_numEigenValuesRR!=d_numEigenValues)
					&& !(dftParameters::isBOMD && dftParameters::isXLBOMD && solveLinearizedKS))
			{
				if(dftParameters::verbosity>=2)
					pcout<< std::endl<<"Poisson solve for total electrostatic potential (rhoOut+b): ";

				computing_timer.enter_section("phiTot solve");

				phiTotalSolverProblem.reinit(d_matrixFreeDataPRefined,
						d_phiTotRhoOut,
						*d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
						d_phiTotDofHandlerIndexElectro,
            d_densityQuadratureIdElectro,
            d_phiTotAXQuadratureIdElectro,
						d_atomNodeIdToChargeMap,
						d_bQuadValuesAllAtoms,
            d_smearedChargeQuadratureIdElectro,
						*rhoOutValues,
						false,
						false,
						dftParameters::smearedNuclearCharges,
            true,
            false,
            0,
            false,
            true);            


				dealiiCGSolver.solve(phiTotalSolverProblem,
						dftParameters::absLinearSolverTolerance,
						dftParameters::maxLinearSolverIterations,
						dftParameters::verbosity);

				computing_timer.exit_section("phiTot solve");
			}

			distributedCPUVec<double> phiRhoMinusApproxRho;
			phiRhoMinusApproxRho.reinit(d_phiTotRhoIn); 
			if (dftParameters::isBOMD && dftParameters::isXLBOMD && solveLinearizedKS && computeForces)
			{
				if(dftParameters::verbosity>=2)
					pcout<< std::endl<<"Poisson solve for (rho_min-n): ";

				computing_timer.enter_section("Poisson solve for (rho_min-approx_rho)");

				std::map<dealii::CellId, std::vector<double> > rhoMinMinusApproxRho;
				std::map<dealii::CellId, std::vector<double> > dummy;
				DoFHandler<3>::active_cell_iterator
					cell = d_matrixFreeDataPRefined.get_dof_handler(d_phiTotDofHandlerIndexElectro).begin_active(),
					     endc = d_matrixFreeDataPRefined.get_dof_handler(d_phiTotDofHandlerIndexElectro).end();
				for (; cell!=endc; ++cell) 
					if (cell->is_locally_owned())
					{
						std::vector<double> & temp= rhoMinMinusApproxRho[cell->id()];
						const std::vector<double> & rhoOut=(*rhoOutValues).find(cell->id())->second;
						const std::vector<double> & rhoIn=(*rhoInValues).find(cell->id())->second;
						temp.resize(d_matrixFreeDataPRefined.get_quadrature(d_densityQuadratureIdElectro).size());
						for (unsigned int q_point=0; q_point<temp.size(); ++q_point)
							temp[q_point]=rhoOut[q_point]-rhoIn[q_point];

					}

				phiTotalSolverProblem.reinit(d_matrixFreeDataPRefined,
						phiRhoMinusApproxRho,
						*d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
						d_phiTotDofHandlerIndexElectro,
            d_densityQuadratureIdElectro,
            d_phiTotAXQuadratureIdElectro,
						std::map<dealii::types::global_dof_index, double>(),
						dummy,
            d_smearedChargeQuadratureIdElectro,
						rhoMinMinusApproxRho,
						false,
            false);

        phiRhoMinusApproxRho=0;
				dealiiCGSolver.solve(phiTotalSolverProblem,
						dftParameters::absLinearSolverTolerance,
						dftParameters::maxLinearSolverIterations,
						dftParameters::verbosity);

				computing_timer.exit_section("Poisson solve for (rho_min-approx_rho)");
			}

			//
			// compute and print ground state energy or energy after max scf iterations
			//
			if (!(dftParameters::isBOMD && dftParameters::isXLBOMD && solveLinearizedKS))
			{
				const double totalEnergy = dftParameters::spinPolarized==0 ?
					energyCalc.computeEnergy(d_dofHandlerPRefined,
							dofHandler,
							quadrature,
							quadrature,
							d_matrixFreeDataPRefined.get_quadrature(d_smearedChargeQuadratureIdElectro),
              d_matrixFreeDataPRefined.get_quadrature(d_lpspQuadratureIdElectro),
							eigenValues,
							d_kPointWeights,
							fermiEnergy,
							funcX,
							funcC,
							d_phiInValues,
							d_phiTotRhoOut,
							*rhoInValues,
							*rhoOutValues,
              d_rhoOutValuesLpspQuad,
							*rhoOutValues,
              d_rhoOutValuesLpspQuad,
							*gradRhoInValues,
							*gradRhoOutValues,
              d_rhoCore,
							d_gradRhoCore,	 
							d_bQuadValuesAllAtoms,
							d_localVselfs,
							d_pseudoVLoc,
							d_pseudoVLoc,
							d_atomNodeIdToChargeMap,
							atomLocations.size(),
							lowerBoundKindex,
							1,
							true,
							dftParameters::smearedNuclearCharges) :
								energyCalc.computeEnergySpinPolarized(d_dofHandlerPRefined,
										dofHandler,
										quadrature,
										quadrature,
						      	d_matrixFreeDataPRefined.get_quadrature(d_smearedChargeQuadratureIdElectro),
                    d_matrixFreeDataPRefined.get_quadrature(d_lpspQuadratureIdElectro),
										eigenValues,
										d_kPointWeights,
										fermiEnergy,
										fermiEnergyUp,
										fermiEnergyDown,
										funcX,
										funcC,
										d_phiInValues,
										d_phiTotRhoOut,
										*rhoInValues,
										*rhoOutValues,
                    d_rhoOutValuesLpspQuad,
										*rhoOutValues,
                    d_rhoOutValuesLpspQuad,
										*gradRhoInValues,
										*gradRhoOutValues,
										*rhoInValuesSpinPolarized,
										*rhoOutValuesSpinPolarized,
										*gradRhoInValuesSpinPolarized,
										*gradRhoOutValuesSpinPolarized,
										d_bQuadValuesAllAtoms,
										d_localVselfs,
										d_pseudoVLoc,
										d_pseudoVLoc,
										d_atomNodeIdToChargeMap,
										atomLocations.size(),
										lowerBoundKindex,
										1,
										true,
										dftParameters::smearedNuclearCharges);

				d_groundStateEnergy = totalEnergy;
			}

			MPI_Barrier(interpoolcomm);

		  d_entropicEnergy=energyCalc.computeEntropicEnergy(eigenValues,
						d_kPointWeights,
						fermiEnergy,
						fermiEnergyUp,
						fermiEnergyDown,
						dftParameters::spinPolarized==1,
						dftParameters::constraintMagnetization,
						dftParameters::TVal);

      if (dftParameters::verbosity>=1)
         pcout<<"Total entropic energy: "<<d_entropicEnergy<<std::endl;    
      
			if (dftParameters::isBOMD && dftParameters::isXLBOMD && solveLinearizedKS)
			{
				d_shadowPotentialEnergy =
					energyCalc.computeShadowPotentialEnergyExtendedLagrangian(d_dofHandlerPRefined,
							dofHandler,
							quadrature,
							d_matrixFreeDataPRefined.get_quadrature(d_smearedChargeQuadratureIdElectro),
							eigenValues,
							d_kPointWeights,
							fermiEnergy,
							funcX,
							funcC,
							d_phiInValues,
							d_phiTotRhoIn,
							*rhoInValues,
							*gradRhoInValues,
              d_rhoCore,
							d_gradRhoCore,	              
							d_bQuadValuesAllAtoms,
							d_localVselfs,
							d_atomNodeIdToChargeMap,
							atomLocations.size(),
							lowerBoundKindex,
							dftParameters::smearedNuclearCharges);
			}

      d_freeEnergy=((dftParameters::isXLBOMD && solveLinearizedKS)?d_shadowPotentialEnergy:d_groundStateEnergy)-d_entropicEnergy;    

      if (dftParameters::verbosity>=1)
         pcout<<"Total free energy: "<<d_freeEnergy<<std::endl; 

			//This step is required for interpolating rho from current mesh to the new
			//mesh in case of atomic relaxation
			//computeNodalRhoFromQuadData();

			computing_timer.exit_section("scf solve");
			computingTimerStandard.exit_section("Total scf solve");

			if (dftParameters::chkType==3 && !(dftParameters::isBOMD && dftParameters::isXLBOMD))
			{
				writeDomainAndAtomCoordinates();
				saveTriaInfoAndRhoNodalData();
			}

#ifdef DFTFE_WITH_GPU
			if (dftParameters::useGPU && (dftParameters::isCellStress || dftParameters::spinPolarized==1 || dftParameters::writeWfcSolutionFields || dftParameters::writeLdosFile || dftParameters::writePdosFile))
				for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
				{
					vectorToolsCUDA::copyCUDAVecToHostVec(d_eigenVectorsFlattenedCUDA.begin()+kPoint*d_eigenVectorsFlattenedSTL[0].size(),
							&d_eigenVectorsFlattenedSTL[kPoint][0],
							d_eigenVectorsFlattenedSTL[kPoint].size());
				}
#endif

			const unsigned int numberBandGroups=
				dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);

			const unsigned int localVectorSize = d_eigenVectorsFlattenedSTL[0].size()/d_numEigenValues;


#ifndef USE_COMPLEX
			if (numberBandGroups>1)
				for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
					MPI_Allreduce(MPI_IN_PLACE,
							&d_eigenVectorsFlattenedSTL[kPoint][0],
							localVectorSize*d_numEigenValues,
							dataTypes::mpi_type_id(&d_eigenVectorsFlattenedSTL[kPoint][0]),
							MPI_SUM,
							interBandGroupComm);
#endif

			if(dftParameters::isCellStress)
			{
				//
				//Create the full dealii partitioned array
				//
				d_eigenVectorsFlattened.resize((1+dftParameters::spinPolarized)*d_kPointWeights.size());

				for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
				{
					vectorTools::createDealiiVector<dataTypes::number>(matrix_free_data.get_vector_partitioner(),
							d_numEigenValues,
							d_eigenVectorsFlattened[kPoint]);


					d_eigenVectorsFlattened[kPoint] = dataTypes::number(0.0);

				}


				Assert(d_eigenVectorsFlattened[0].local_size()==d_eigenVectorsFlattenedSTL[0].size(),
						dealii::ExcMessage("Incorrect local sizes of STL and dealii arrays"));

				constraintsNoneDataInfo.precomputeMaps(matrix_free_data.get_vector_partitioner(),
						d_eigenVectorsFlattened[0].get_partitioner(),
						d_numEigenValues);

				for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
				{
					for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
					{
						for(unsigned int iWave = 0; iWave < d_numEigenValues; ++iWave)
						{
							d_eigenVectorsFlattened[kPoint].local_element(iNode*d_numEigenValues+iWave)
								= d_eigenVectorsFlattenedSTL[kPoint][iNode*d_numEigenValues+iWave];
						}
					}

					constraintsNoneDataInfo.distribute(d_eigenVectorsFlattened[kPoint],
							d_numEigenValues);

				}

				/*
				   for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
				   {
				   d_eigenVectorsFlattenedSTL[kPoint].clear();
				   std::vector<dataTypes::number>().swap(d_eigenVectorsFlattenedSTL[kPoint]);
				   }
				 */
			}

			if (dftParameters::isIonForce)
			{
				if(dftParameters::selfConsistentSolverTolerance>1e-4 && dftParameters::verbosity>=1)
					pcout<<"DFT-FE Warning: Ion force accuracy may be affected for the given scf iteration solve tolerance: "<<dftParameters::selfConsistentSolverTolerance<<", recommended to use TOLERANCE below 1e-4."<<std::endl;

				computing_timer.enter_section("Ion force computation");
				computingTimerStandard.enter_section("Ion force computation");
				if (computeForces)
				{
					if (dftParameters::isBOMD && dftParameters::isXLBOMD && solveLinearizedKS)
						forcePtr->computeAtomsForces(matrix_free_data,
#ifdef DFTFE_WITH_GPU
								kohnShamDFTEigenOperatorCUDA,
#endif
								d_eigenDofHandlerIndex,
                d_smearedChargeQuadratureIdElectro,
                d_lpspQuadratureIdElectro,
								d_matrixFreeDataPRefined,
								d_phiTotDofHandlerIndexElectro,
								d_phiTotRhoIn,
								*rhoInValues,
								*gradRhoInValues,
                d_gradRhoOutValuesLpspQuad,
								*rhoInValues,
                d_rhoOutValuesLpspQuad,
								*gradRhoInValues,
                d_gradRhoOutValuesLpspQuad,
								d_rhoCore,
                d_gradRhoCore, 
                d_hessianRhoCore,
                d_gradRhoCoreAtoms,
                d_hessianRhoCoreAtoms,	                
								d_pseudoVLoc,
								d_pseudoVLocAtoms,
								d_constraintsPRefined,
								d_vselfBinsManager,
								*rhoOutValues,
								*gradRhoOutValues,
								phiRhoMinusApproxRho,
								true);
					else
						forcePtr->computeAtomsForces(matrix_free_data,
#ifdef DFTFE_WITH_GPU
								kohnShamDFTEigenOperatorCUDA,
#endif
								d_eigenDofHandlerIndex,
                d_smearedChargeQuadratureIdElectro,
                d_lpspQuadratureIdElectro,
								d_matrixFreeDataPRefined,
								d_phiTotDofHandlerIndexElectro,
								d_phiTotRhoOut,
								*rhoOutValues,
								*gradRhoOutValues,
                d_gradRhoOutValuesLpspQuad,
								*rhoOutValues,
                d_rhoOutValuesLpspQuad,
								*gradRhoOutValues,
                d_gradRhoOutValuesLpspQuad,
								d_rhoCore,
                d_gradRhoCore, 
                d_hessianRhoCore,
                d_gradRhoCoreAtoms,
                d_hessianRhoCoreAtoms,                
								d_pseudoVLoc,
								d_pseudoVLocAtoms,
								d_constraintsPRefined,
								d_vselfBinsManager,
								*rhoOutValues,
								*gradRhoOutValues,
								d_phiTotRhoIn);
					forcePtr->printAtomsForces();
				}
				computingTimerStandard.exit_section("Ion force computation");
				computing_timer.exit_section("Ion force computation");
			}
			
      if (dftParameters::isCellStress)
			{
				if(dftParameters::selfConsistentSolverTolerance>1e-4 && dftParameters::verbosity>=1)
					pcout<<"DFT-FE Warning: Cell stress accuracy may be affected for the given scf iteration solve tolerance: "<<dftParameters::selfConsistentSolverTolerance<<", recommended to use TOLERANCE below 1e-4."<<std::endl;

				computing_timer.enter_section("Cell stress computation");
				computingTimerStandard.enter_section("Cell stress computation");
				if (computeForces)
				{
					forcePtr->computeStress(matrix_free_data,
							d_eigenDofHandlerIndex,
              d_smearedChargeQuadratureIdElectro,
              d_lpspQuadratureIdElectro,
							d_matrixFreeDataPRefined,
							d_phiTotDofHandlerIndexElectro,
							d_phiTotRhoOut,
              *rhoOutValues,
              *gradRhoOutValues,
              d_gradRhoOutValuesLpspQuad,
              *rhoOutValues,
              d_rhoOutValuesLpspQuad,
              *gradRhoOutValues,
              d_gradRhoOutValuesLpspQuad,
              d_pseudoVLoc,
              d_pseudoVLocAtoms,
							d_rhoCore,
              d_gradRhoCore, 
              d_hessianRhoCore,  
              d_gradRhoCoreAtoms,
              d_hessianRhoCoreAtoms,                
						  d_constraintsPRefined,
							d_vselfBinsManager);
					forcePtr->printStress();
				}
				computingTimerStandard.exit_section("Cell stress computation");
				computing_timer.exit_section("Cell stress computation");
			}

			if(dftParameters::electrostaticsHRefinement)
				computeElectrostaticEnergyHRefined(
#ifdef DFTFE_WITH_GPU
						kohnShamDFTEigenOperatorCUDA
#endif
      );

			if (dftParameters::writeWfcSolutionFields)
				outputWfc();

			if (dftParameters::writeDensitySolutionFields)
				outputDensity();


#ifdef USE_COMPLEX
			if( !(dftParameters::kPointDataFile == "") )
			{
				readkPointData();
				initnscf(kohnShamDFTEigenOperator, phiTotalSolverProblem,dealiiCGSolver) ;
				nscf(kohnShamDFTEigenOperator,d_subspaceIterationSolver) ;
				writeBands() ;
			}
#endif

			if (!kohnShamDFTOperatorsInitialized || true)
				finalizeKohnShamDFTOperator(kohnShamDFTEigenOperator
#ifdef DFTFE_WITH_GPU
						,
						kohnShamDFTEigenOperatorCUDA
#endif
						);
		}

	//Output wfc
	template <unsigned int FEOrder,unsigned int FEOrderElectro>
		void dftClass<FEOrder,FEOrderElectro>::outputWfc()
		{

			//
			//identify the index which is close to Fermi Energy
			//
			int indexFermiEnergy = -1.0;
			for(int spinType = 0; spinType < 1+dftParameters::spinPolarized; ++spinType)
			{
				for(int i = 0; i < d_numEigenValues; ++i)
				{
					if(eigenValues[0][spinType*d_numEigenValues + i] >= fermiEnergy)
					{
						if(i > indexFermiEnergy)
						{
							indexFermiEnergy = i;
							break;
						}
					}
				}
			}

			//
			//create a range of wavefunctions to output the wavefunction files
			//
			int startingRange = indexFermiEnergy - 4;
			int endingRange = indexFermiEnergy + 4;

			int startingRangeSpin = startingRange;

			for(int spinType = 0; spinType < 1+dftParameters::spinPolarized; ++spinType)
			{
				for(int i = indexFermiEnergy-5; i > 0; --i)
				{
					if(std::abs(eigenValues[0][spinType*d_numEigenValues + (indexFermiEnergy-4)] - eigenValues[0][spinType*d_numEigenValues + i]) <= 5e-04)
					{
						if(spinType == 0)
							startingRange -= 1;
						else
							startingRangeSpin -= 1;
					}

				}
			}


			if(startingRangeSpin < startingRange)
				startingRange = startingRangeSpin;

			int numStatesOutput = (endingRange - startingRange) + 1;


			DataOut<3> data_outEigen;
			data_outEigen.attach_dof_handler(dofHandlerEigen);

			std::vector<distributedCPUVec<double>> tempVec(1);
			tempVec[0].reinit(d_tempEigenVec);

			std::vector<distributedCPUVec<double>> visualizeWaveFunctions(d_kPointWeights.size()*(1+dftParameters::spinPolarized)*numStatesOutput);

			unsigned int count = 0;
			for(unsigned int s = 0; s < 1+dftParameters::spinPolarized; ++s)
				for(unsigned int k = 0; k < d_kPointWeights.size(); ++k)
					for(unsigned int i = startingRange; i < endingRange; ++i)
					{

#ifdef USE_COMPLEX
						vectorTools::copyFlattenedSTLVecToSingleCompVec(d_eigenVectorsFlattenedSTL[k*(1+dftParameters::spinPolarized)+s],
								d_numEigenValues,
								std::make_pair(i,i+1),
								localProc_dof_indicesReal,
								localProc_dof_indicesImag,
								tempVec);
#else
						vectorTools::copyFlattenedSTLVecToSingleCompVec(d_eigenVectorsFlattenedSTL[k*(1+dftParameters::spinPolarized)+s],
								d_numEigenValues,
								std::make_pair(i,i+1),
								tempVec);
#endif

						constraintsNoneEigenDataInfo.distribute(tempVec[0]);
						visualizeWaveFunctions[count] = tempVec[0];

						if (dftParameters::spinPolarized==1)
							data_outEigen.add_data_vector(visualizeWaveFunctions[count],"wfc_spin"+std::to_string(s)+"_kpoint"+std::to_string(k)+"_"+std::to_string(i));
						else
							data_outEigen.add_data_vector(visualizeWaveFunctions[count],"wfc_kpoint"+std::to_string(k)+"_"+std::to_string(i));

						count += 1;


					}

			data_outEigen.build_patches(FEOrder);

			std::string tempFolder = "waveFunctionOutputFolder";
			mkdir(tempFolder.c_str(),ACCESSPERMS);

			dftUtils::writeDataVTUParallelLowestPoolId(dofHandlerEigen,
					data_outEigen,
					mpi_communicator,
					interpoolcomm,
					interBandGroupComm,
					tempFolder,
					"wfcOutput");
			//"wfcOutput_"+std::to_string(k)+"_"+std::to_string(i));


		}


	//Output density
	template <unsigned int FEOrder,unsigned int FEOrderElectro>
		void dftClass<FEOrder,FEOrderElectro>::outputDensity()
		{
			//
			//compute nodal electron-density from quad data
			//
			distributedCPUVec<double>  rhoNodalField;
			d_matrixFreeDataPRefined.initialize_dof_vector(rhoNodalField,d_densityDofHandlerIndexElectro);
			rhoNodalField=0;
			std::function<double(const typename dealii::DoFHandler<3>::active_cell_iterator & cell ,
					const unsigned int q)> funcRho =
				[&](const typename dealii::DoFHandler<3>::active_cell_iterator & cell ,
						const unsigned int q)
				{return (*rhoOutValues).find(cell->id())->second[q];};
			dealii::VectorTools::project<3,distributedCPUVec<double>> (dealii::MappingQ1<3,3>(),
					d_dofHandlerRhoNodal,
					d_constraintsRhoNodal,
					d_matrixFreeDataPRefined.get_quadrature(d_densityQuadratureIdElectro),
					funcRho,
					rhoNodalField);
			rhoNodalField.update_ghost_values();

			distributedCPUVec<double>  rhoNodalFieldSpin0;
			distributedCPUVec<double>  rhoNodalFieldSpin1;
			if (dftParameters::spinPolarized==1)
			{
		  	rhoNodalFieldSpin0.reinit(rhoNodalField);        
				rhoNodalFieldSpin0=0;
				std::function<double(const typename dealii::DoFHandler<3>::active_cell_iterator & cell ,
						const unsigned int q)> funcRhoSpin0 =
					[&](const typename dealii::DoFHandler<3>::active_cell_iterator & cell ,
							const unsigned int q)
					{return (*rhoOutValuesSpinPolarized).find(cell->id())->second[2*q];};
				dealii::VectorTools::project<3,distributedCPUVec<double>> (dealii::MappingQ1<3,3>(),
            d_dofHandlerRhoNodal,
            d_constraintsRhoNodal,
            d_matrixFreeDataPRefined.get_quadrature(d_densityQuadratureIdElectro),
						funcRhoSpin0,
						rhoNodalFieldSpin0);
				rhoNodalFieldSpin0.update_ghost_values();


		  	rhoNodalFieldSpin1.reinit(rhoNodalField);   
				rhoNodalFieldSpin1=0;
				std::function<double(const typename dealii::DoFHandler<3>::active_cell_iterator & cell ,
						const unsigned int q)> funcRhoSpin1 =
					[&](const typename dealii::DoFHandler<3>::active_cell_iterator & cell ,
							const unsigned int q)
					{return (*rhoOutValuesSpinPolarized).find(cell->id())->second[2*q+1];};
				dealii::VectorTools::project<3,distributedCPUVec<double>> (dealii::MappingQ1<3,3>(),
            d_dofHandlerRhoNodal,
            d_constraintsRhoNodal,
            d_matrixFreeDataPRefined.get_quadrature(d_densityQuadratureIdElectro),
						funcRhoSpin1,
						rhoNodalFieldSpin1);
				rhoNodalFieldSpin1.update_ghost_values();
			}

			//
			//only generate output for electron-density
			//
			DataOut<3> dataOutRho;
		  dataOutRho.attach_dof_handler(d_dofHandlerRhoNodal);
			dataOutRho.add_data_vector(rhoNodalField, std::string("density"));
			if (dftParameters::spinPolarized==1)
			{
				dataOutRho.add_data_vector(rhoNodalFieldSpin0, std::string("density_0"));
				dataOutRho.add_data_vector(rhoNodalFieldSpin1, std::string("density_1"));
			}
			dataOutRho.build_patches(FEOrder);

			std::string tempFolder = "densityOutputFolder";
			mkdir(tempFolder.c_str(),ACCESSPERMS);

			dftUtils::writeDataVTUParallelLowestPoolId(d_dofHandlerRhoNodal,
					dataOutRho,
					mpi_communicator,
					interpoolcomm,
					interBandGroupComm,
					tempFolder,
					"densityOutput");

		}

	template <unsigned int FEOrder,unsigned int FEOrderElectro>
		void dftClass<FEOrder,FEOrderElectro>::writeBands()
		{
			int numkPoints = (1+dftParameters::spinPolarized)*d_kPointWeights.size();
			std::vector<double> eigenValuesFlattened ;
			//
			for (unsigned int kPoint = 0; kPoint < numkPoints; ++kPoint)
				for(unsigned int iWave = 0; iWave < d_numEigenValues; ++iWave)
					eigenValuesFlattened.push_back(eigenValues[kPoint][iWave]) ;
			//
			//
			//
			int totkPoints = Utilities::MPI::sum(numkPoints, interpoolcomm);
			std::vector<int> numkPointsArray(dftParameters::npool), mpi_offsets(dftParameters::npool, 0);
			std::vector<double> eigenValuesFlattenedGlobal(totkPoints*d_numEigenValues,0.0);
			//
			MPI_Gather(&numkPoints,1,MPI_INT, &(numkPointsArray[0]),1, MPI_INT,0,interpoolcomm);
			//
			numkPointsArray[0] = d_numEigenValues * numkPointsArray[0] ;
			for (unsigned int ipool=1; ipool < dftParameters::npool ; ++ipool)
			{
				numkPointsArray[ipool] =  d_numEigenValues * numkPointsArray[ipool] ;
				mpi_offsets[ipool] = mpi_offsets[ipool-1] + numkPointsArray[ipool -1] ;
			}
			//
			MPI_Gatherv(&(eigenValuesFlattened[0]), numkPoints*d_numEigenValues, MPI_DOUBLE, &(eigenValuesFlattenedGlobal[0]),&(numkPointsArray[0]), &(mpi_offsets[0]), MPI_DOUBLE, 0 ,interpoolcomm);
			//
			if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) ==0)
			{
				FILE * pFile;
				pFile = fopen ("bands.out","w");
				fprintf (pFile, "%d %d\n", totkPoints, d_numEigenValues );
				for (unsigned int kPoint = 0; kPoint < totkPoints/(1+dftParameters::spinPolarized); ++kPoint)
				{
					for(unsigned int iWave = 0; iWave < d_numEigenValues; ++iWave)
					{
						if (dftParameters::spinPolarized)
							fprintf (pFile, "%d  %d   %g   %g\n",  kPoint, iWave, eigenValuesFlattenedGlobal[2*kPoint*d_numEigenValues+iWave], eigenValuesFlattenedGlobal[(2*kPoint+1)*d_numEigenValues+iWave]);
						else
							fprintf (pFile, "%d  %d %g\n",  kPoint, iWave, eigenValuesFlattenedGlobal[kPoint*d_numEigenValues+iWave]);
					}
				}
			}
			MPI_Barrier(MPI_COMM_WORLD);
			//
		}

#include "dft.inst.cc"
}


