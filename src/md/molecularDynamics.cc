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
// @author Sambit Das
//

#include <molecularDynamics.h>
#include <force.h>
#include <dft.h>
#include <fileReaders.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <boost/random.hpp>
#include <boost/generator_iterator.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random/normal_distribution.hpp>

namespace dftfe {

//
//constructor
//
template<unsigned int FEOrder>
molecularDynamics<FEOrder>::molecularDynamics(dftClass<FEOrder>* _dftPtr,const MPI_Comm &mpi_comm_replica):
  dftPtr(_dftPtr),
  mpi_communicator (mpi_comm_replica),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_comm_replica)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_comm_replica)),
  pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
{

}



template<unsigned int FEOrder>
void molecularDynamics<FEOrder>::run()
{

	//********************* Molecular dynamics simulation*****************************//

        pcout<<"---------------Starting Molecular dynamics simulation---------------------"<<std::endl;
        const unsigned int numberGlobalCharges=dftPtr->atomLocations.size();
        //https://lammps.sandia.gov/doc/units.html
	const double initialTemperature = 300;//K
	const unsigned int restartFlag =dftParameters::restartFromChk?1:0; //1; //0;//1;
	int startingTimeStep = 0; //625;//450;// 50;//300; //0;// 300;

	double massAtomAl = 26.982;//mass proton is chosen 1 **49611.513**
	double massAtomMg= 24.305;
	const double timeStep = 0.05; //0.5 femtoseconds
	const unsigned int numberTimeSteps = 2;

        //https://physics.nist.gov/cuu/Constants/Table/allascii.txt
	const double kb = 8.617333262e-05;//eV/K **3.166811429e-6**;
	const double initialVelocityDeviation = sqrt(kb*initialTemperature/massAtomAl);
	const double haPerBohrToeVPerAng = 27.211386245988/0.529177210903;
	const double haToeV = 27.211386245988;
	const double bohrToAng = 0.529177210903;
	const double AngTobohr = 1.0/bohrToAng;

        std::vector<double> massAtoms(numberGlobalCharges);
	for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
	{
           const double charge= dftPtr->atomLocations[iCharge][0];
	   if (std::fabs(charge-12.0)<1e-8)
	       massAtoms[iCharge]=massAtomMg;
	   else if(std::fabs(charge-13.0)<1e-8)
	       massAtoms[iCharge]=massAtomAl;
	   else
	       AssertThrow(false,ExcMessage("Currently md capability is hardcoded for systems with Al and Mg atom types only."));
	}

	std::vector<dealii::Tensor<1,3,double> > displacements(numberGlobalCharges);
	std::vector<double> velocity(3*numberGlobalCharges,0.0);
        std::vector<double> velocityMid(3*numberGlobalCharges,0.0);
	std::vector<double> force(3*numberGlobalCharges,0.0);
	std::vector<double> acceleration(3*numberGlobalCharges,0.0);
	std::vector<double> accelerationNew(3*numberGlobalCharges,0.0);

	std::vector<double> internalEnergyVector(numberTimeSteps,0.0);
	std::vector<double> entropicEnergyVector(numberTimeSteps,0.0);
	std::vector<double> kineticEnergyVector(numberTimeSteps,0.0);
        std::vector<double> totalEnergyVector(numberTimeSteps,0.0);

        const unsigned int kmax=6;
        const double k=1.82;
        const double alpha=0.018;
        const double c0=-6.0;
        const double c1=14.0;
        const double c2=-8.0;
        const double c3=-3.0;
        const double c4=4.0;
        const double c5=-1.0;
        const double diracDeltaKernelConstant=-0.5;
        std::deque<vectorType> approxDensityContainer;
        vectorType shadowKSRhoMin;

	double kineticEnergy = 0.0;

	boost::mt19937 rng(std::time(0));
	//boost::mt19937 rng;
	boost::normal_distribution<> gaussianDist(0.0,initialVelocityDeviation);
	boost::variate_generator<boost::mt19937&,boost::normal_distribution<> > generator(rng,gaussianDist);
	double averageKineticEnergy;
	double temperatureFromVelocities;

	if(restartFlag == 0)
	  {
	   // dftParameters::TVal=initialTemperature;
	    dftPtr->solve();
            const std::vector<double> forceOnAtoms= dftPtr->forcePtr->getAtomsForces();

            // Set approximate electron density to the exact ground state electron density in the 0th step
            if (dftParameters::isXLBOMD)
            {
                   shadowKSRhoMin=dftPtr->d_rhoOutNodalValues;
                   dftPtr->d_constraintsPRefined.distribute(shadowKSRhoMin);
                   shadowKSRhoMin.update_ghost_values();

                   approxDensityContainer.push_back(dftPtr->d_rhoOutNodalValues);
                   dftPtr->d_constraintsPRefined.distribute(approxDensityContainer.back());
                   approxDensityContainer.back().update_ghost_values();

	           //normalize approxDensityVec
	           const double charge = dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,
						     approxDensityContainer.back());
	           pcout<<"Total Charge before Normalizing approxDensityVec:  "<<charge<<std::endl;
	       
		   const double scalingFactor = ((double)dftPtr->numElectrons)/charge;

		   //scale nodal vector with scalingFactor
		   approxDensityContainer.back() *= scalingFactor;
		   pcout<<"Total Charge after Normalizing approxDensityVec:  "<<dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,approxDensityContainer.back())<<std::endl;
            }

            const bool testing=false;
            if (testing)
            {
		   dftPtr->interpolateNodalDataToQuadratureData(dftPtr->d_matrixFreeDataPRefined,
						        approxDensityContainer.back(),
						        *(dftPtr->rhoInValues),
						        *(dftPtr->gradRhoInValues),
						         dftParameters::xc_id == 4);
		    dftPtr->solve(true,
				  true);
                    pcout<<"Shadow potential Energy check: "<<dftPtr->d_shadowPotentialEnergy<<std::endl;


		    exit(0);
            }

	    if(this_mpi_process == 0)
	      {
		for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
		  {
		    velocity[3*iCharge + 0] = generator();
		    velocity[3*iCharge + 1] = generator();
		    velocity[3*iCharge + 2] = generator();
		  }
	      }

	   for (unsigned int i=0; i< numberGlobalCharges*3; ++i)
	   {
                  velocity[i]=dealii::Utilities::MPI::sum(velocity[i], mpi_communicator);

	   }

	    //compute initial acceleration and initial kinEnergy
	    for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
	      {
		acceleration[3*iCharge + 0] = -(forceOnAtoms[3*iCharge + 0]*haPerBohrToeVPerAng)/massAtoms[iCharge];
		acceleration[3*iCharge + 1] = -(forceOnAtoms[3*iCharge + 1]*haPerBohrToeVPerAng)/massAtoms[iCharge];
		acceleration[3*iCharge + 2] = -(forceOnAtoms[3*iCharge + 2]*haPerBohrToeVPerAng)/massAtoms[iCharge];
		kineticEnergy += 0.5*massAtoms[iCharge]*(velocity[3*iCharge + 0]*velocity[3*iCharge + 0] + velocity[3*iCharge + 1]*velocity[3*iCharge + 1] + velocity[3*iCharge + 2]*velocity[3*iCharge + 2]);
	      }

	  }
	else
	  {
	    //read into acceleration vector
	    std::string fileName = "acceleration.chk"; //accelerationData1x1x1
	    std::vector<std::vector<double> > fileAccData;

	    dftUtils::readFile(3,
		               fileAccData,
		               fileName);

	    std::string fileName1 = "velocity.chk"; //velocityData1x1x1
	    std::vector<std::vector<double> > fileVelData;

	    dftUtils::readFile(3,
		               fileVelData,
		               fileName1);

	    for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
	      {
		acceleration[3*iCharge + 0] = fileAccData[iCharge][0];
		acceleration[3*iCharge + 1] = fileAccData[iCharge][1];
		acceleration[3*iCharge + 2] = fileAccData[iCharge][2];
	      }

	    for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
	      {
		velocity[3*iCharge + 0] = fileVelData[iCharge][0];
		velocity[3*iCharge + 1] = fileVelData[iCharge][1];
		velocity[3*iCharge + 2] = fileVelData[iCharge][2];
	      }

	    std::vector<std::vector<double> > fileTemperatueData;
	    std::vector<std::vector<double> > timeIndexData;

	    dftUtils::readFile(1,
		               fileTemperatueData,
		               "temperature.chk");

	    dftUtils::readFile(1,
		               timeIndexData,
		               "time.chk");

	    temperatureFromVelocities=fileTemperatueData[0][0];


	    startingTimeStep=timeIndexData[0][0];


            pcout<<" Ending time step read from file: "<<startingTimeStep<<std::endl;
	    pcout<<" Temperature read from file: "<<temperatureFromVelocities<<std::endl;
		/*
		pcout<<"Acceleration  of charges read from file: "<<std::endl;
		for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
		  pcout<<iCharge<<" "<<acceleration[3*iCharge + 0]<<" "<<acceleration[3*iCharge + 1]<<" "<<acceleration[3*iCharge + 2]<<std::endl;

		pcout<<"Velocity  of charges read from file: "<<std::endl;
		for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
		  pcout<<iCharge<<" "<<velocity[3*iCharge + 0]<<" "<<velocity[3*iCharge + 1]<<" "<<velocity[3*iCharge + 2]<<std::endl;
		 */
	  }





	if(restartFlag == 0)
	  {
	    //
	    //compute average velocity
	    //
	    double avgVelx = 0.0,avgVely = 0.0,avgVelz = 0.0;
	    for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
	      {
		avgVelx += velocity[3*iCharge + 0];
		avgVely += velocity[3*iCharge + 1];
		avgVelz += velocity[3*iCharge + 2];
	      }

	    avgVelx = avgVelx/numberGlobalCharges;
	    avgVely = avgVely/numberGlobalCharges;
	    avgVelz = avgVelz/numberGlobalCharges;

	    //
	    //recompute initial velocities by subtracting out center of mass velocity
	    //
	    for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
	      {
		velocity[3*iCharge + 0] -= avgVelx;
		velocity[3*iCharge + 1] -= avgVely;
		velocity[3*iCharge + 2] -= avgVelz;
	      }

	    //recompute kinetic energy again
	    kineticEnergy = 0.0;
	    for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
	      {
		kineticEnergy += 0.5*massAtoms[iCharge]*(velocity[3*iCharge + 0]*velocity[3*iCharge + 0] + velocity[3*iCharge + 1]*velocity[3*iCharge + 1] + velocity[3*iCharge + 2]*velocity[3*iCharge + 2]);
	      }

	    averageKineticEnergy = kineticEnergy/(3*numberGlobalCharges);
	    temperatureFromVelocities = averageKineticEnergy*2/kb;

	    pcout<<"Temperature computed from Velocities: "<<temperatureFromVelocities<<std::endl;

	    double gamma = sqrt(initialTemperature/temperatureFromVelocities);

	    for(int i = 0; i < 3*numberGlobalCharges; ++i)
	      {
		velocity[i] = gamma*velocity[i];
	      }

	    //recompute kinetic energy again
	    kineticEnergy = 0.0;
	    for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
	      {
		kineticEnergy += 0.5*massAtoms[iCharge]*(velocity[3*iCharge + 0]*velocity[3*iCharge + 0] + velocity[3*iCharge + 1]*velocity[3*iCharge + 1] + velocity[3*iCharge + 2]*velocity[3*iCharge + 2]);
	      }

	    averageKineticEnergy = kineticEnergy/(3*numberGlobalCharges);
	    temperatureFromVelocities = averageKineticEnergy*2/kb;

            pcout<<"Temperature computed from Scaled Velocities: "<<temperatureFromVelocities<<std::endl;

	    kineticEnergyVector[0] = kineticEnergy/haToeV;
	    internalEnergyVector[0] = dftPtr->d_groundStateEnergy;
	    entropicEnergyVector[0] = dftPtr->d_entropicEnergy;
            totalEnergyVector[0]=kineticEnergyVector[0]+internalEnergyVector[0]-entropicEnergyVector[0];

            pcout<<" Initial Velocity Deviation: "<<initialVelocityDeviation<<std::endl;
	    pcout<<" Kinetic Energy in Ha at timeIndex 0 "<<kineticEnergyVector[0]<<std::endl;
	    pcout<<" Internal Energy in Ha at timeIndex 0 "<<internalEnergyVector[0]<<std::endl;
	    pcout<<" Entropic Energy in Ha at timeIndex 0 "<<entropicEnergyVector[0]<<std::endl;
            pcout<<" Total Energy in Ha at timeIndex 0 "<<totalEnergyVector[0]<<std::endl;

	  }

	//
	//start the MD simulation
	//
	for(int timeIndex = startingTimeStep+1; timeIndex < numberTimeSteps; ++timeIndex)
	  {
	    //
	    //compute average velocity
	    //
	    double avgVelx = 0.0,avgVely = 0.0,avgVelz = 0.0;
	    for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
	      {
		avgVelx += velocity[3*iCharge + 0];
		avgVely += velocity[3*iCharge + 1];
		avgVelz += velocity[3*iCharge + 2];
	      }

	    avgVelx = avgVelx/numberGlobalCharges;
	    avgVely = avgVely/numberGlobalCharges;
	    avgVelz = avgVelz/numberGlobalCharges;

	    pcout<<"Velocity and acceleration of atoms at current timeStep "<<timeIndex<<std::endl;
	    pcout<<"Velocity of center of mass: "<<avgVelx<<" "<<avgVely<<" "<<avgVelz<<" "<<std::endl;
		/*
		pcout<<"Velocity of charges: "<<std::endl;
		for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
		  pcout<<iCharge<<" "<<velocity[3*iCharge + 0]<<" "<<velocity[3*iCharge + 1]<<" "<<velocity[3*iCharge + 2]<<std::endl;

		pcout<<"Acceleration of charges: "<<std::endl;
		for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
		  pcout<<iCharge<<" "<<acceleration[3*iCharge + 0]<<" "<<acceleration[3*iCharge + 1]<<" "<<acceleration[3*iCharge + 2]<<std::endl;
		  */

	    //
	    //compute new positions and displacements from t to t + \Delta t using combined leap-frog and velocity Verlet scheme
	    //
	    for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
	      {
	        velocityMid[3*iCharge+0] = velocity[3*iCharge + 0] + 0.5*acceleration[3*iCharge+ 0]*timeStep;
		velocityMid[3*iCharge+1] = velocity[3*iCharge + 1] + 0.5*acceleration[3*iCharge+ 1]*timeStep;
		velocityMid[3*iCharge+2] = velocity[3*iCharge + 2] + 0.5*acceleration[3*iCharge+ 2]*timeStep;

		displacements[iCharge][0] = velocityMid[3*iCharge + 0]*timeStep;
		displacements[iCharge][1] = velocityMid[3*iCharge + 1]*timeStep;
		displacements[iCharge][2] = velocityMid[3*iCharge + 2]*timeStep;
	      }

	    //
	    //compute forces for atoms at t + \Delta t
	    //

	    //convert displacements into atomic units
	    for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
	      {
		displacements[iCharge][0] = displacements[iCharge][0]*AngTobohr;
		displacements[iCharge][1] = displacements[iCharge][1]*AngTobohr;
		displacements[iCharge][2] = displacements[iCharge][2]*AngTobohr;
	      }

	    if(dftParameters::verbosity>=5)
	      {
		pcout<<"Displacement of atoms at current timeStep "<<timeIndex<<std::endl;
		for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
		  {
		    pcout<<"Charge Id: "<<iCharge<<" "<<displacements[iCharge][0]<<" "<<displacements[iCharge][1]<<" "<<displacements[iCharge][2]<<std::endl;
		  }
	      }

	    //
	    //first move the mesh to current positions
	    //
	    dftPtr->updateAtomPositionsAndMoveMesh(displacements,dftParameters::maxJacobianRatioFactorForMD,(timeIndex ==startingTimeStep+1 && restartFlag==1)?true:false);

	    if(dftParameters::verbosity>=5)
	      {
		pcout<<"New atomic positions on the Mesh: "<<std::endl;
		for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
		  {
		    pcout<<"Charge Id: "<<iCharge<<" "<<dftPtr->atomLocations[iCharge][2]<<" "<<dftPtr->atomLocations[iCharge][3]<<" "<<dftPtr->atomLocations[iCharge][4]<<std::endl;
		  }
	      }

            if (dftParameters::isXLBOMD)
            {

                    if (dftPtr->d_autoMesh==1)
                    {
                        pcout<<".............Auto meshing step: interpolation of approximate density container to new mesh.............."<<std::endl;
                        //interpolate all the vectors in the approximate density container on the current mesh
                        std::vector<vectorType* > fieldsPtrsPrevious(approxDensityContainer.size());
                        std::vector<vectorType* > fieldsPtrsCurrent(approxDensityContainer.size());
                        std::vector<vectorType> fieldsCurrent(approxDensityContainer.size());

                        for (unsigned int i = 0; i < approxDensityContainer.size(); i++)
                        {
                            fieldsPtrsPrevious[i]=&approxDensityContainer[i];
                            dftPtr->d_matrixFreeDataPRefined.initialize_dof_vector(fieldsCurrent[i]);
                            fieldsPtrsCurrent[i]=&fieldsCurrent[i];
                        } 

                        dealii::FESystem<3> fe(dealii::FESystem<3>(dftPtr->d_matrixFreeDataPRefined.get_dof_handler().get_fe(),1));
                        dftPtr->interpolateFieldsFromPrevToCurrentMesh(fieldsPtrsPrevious,
                                                                       fieldsPtrsCurrent,
                                                                       fe,
                                                                       fe,
                                                                       dftPtr->d_constraintsPRefined);    

                        for (unsigned int i = 0; i < approxDensityContainer.size(); i++)
                        {
                           approxDensityContainer[i]=fieldsCurrent[i];
                           approxDensityContainer[i].update_ghost_values();
                        }                     

                        //re-normalize
                        for (unsigned int i = 0; i < approxDensityContainer.size(); i++)
                        {
	                   const double charge = dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,
						     approxDensityContainer[i]);
	                   pcout<<"Total Charge before Normalizing interpolated field:  "<<charge<<std::endl;
	       
	 	           const double scalingFactor = ((double)dftPtr->numElectrons)/charge;

		           approxDensityContainer[i] *= scalingFactor;
		           pcout<<"Total Charge after Normalizing interpolated field:  "<<dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,approxDensityContainer[i])<<std::endl;
                        }

                        dftPtr->updatePrevMeshDataStructures();
                        pcout<<".............Auto meshing step: interpolation and re-normalization completed.............."<<std::endl;
                    }


                    vectorType approxDensityNext;
                    approxDensityNext.reinit(approxDensityContainer.back()); 

                    const unsigned int containerSizeCurrent=approxDensityContainer.size();
                    vectorType & approxDensityTimeT=approxDensityContainer[containerSizeCurrent-1];
                    vectorType & approxDensityTimeTMinusDeltat=containerSizeCurrent>1?
                                                               approxDensityContainer[containerSizeCurrent-2]
                                                               :approxDensityTimeT;
                   
                    const unsigned int local_size = approxDensityNext.local_size();

                    for (unsigned int i = 0; i < local_size; i++)
                        approxDensityNext.local_element(i)=approxDensityTimeT.local_element(i)*2.0
                                                          -approxDensityTimeTMinusDeltat.local_element(i)
                                                          -(shadowKSRhoMin.local_element(i)-approxDensityTimeT.local_element(i))*k*diracDeltaKernelConstant;
 
 
                    if (approxDensityContainer.size()==kmax)
                    {
                        for (unsigned int i = 0; i < local_size; i++)
				approxDensityNext.local_element(i)+=alpha*(c0*approxDensityContainer[containerSizeCurrent-1].local_element(i)
							 +c1*approxDensityContainer[containerSizeCurrent-2].local_element(i)
							 +c2*approxDensityContainer[containerSizeCurrent-3].local_element(i)
							 +c3*approxDensityContainer[containerSizeCurrent-4].local_element(i)
							 +c4*approxDensityContainer[containerSizeCurrent-5].local_element(i)
							 +c5*approxDensityContainer[containerSizeCurrent-6].local_element(i));
                        approxDensityContainer.pop_front();
                    }
                    approxDensityContainer.push_back(approxDensityNext);
                    approxDensityContainer.back().update_ghost_values();                    
 
	            //normalize approxDensityVec
	            const double charge = dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,
						     approxDensityContainer.back());
	            pcout<<"Total Charge before Normalizing new approxDensityVec:  "<<charge<<std::endl;
	       
		    const double scalingFactor = ((double)dftPtr->numElectrons)/charge;

		    //scale nodal vector with scalingFactor
		    approxDensityContainer.back() *= scalingFactor;
		    pcout<<"Total Charge after Normalizing new approxDensityVec:  "<<dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,approxDensityContainer.back())<<std::endl;

		    dftPtr->interpolateNodalDataToQuadratureData(dftPtr->d_matrixFreeDataPRefined,
						        approxDensityContainer.back(),
						        *(dftPtr->rhoInValues),
						        *(dftPtr->gradRhoInValues),
						         dftParameters::xc_id == 4);
                     
                    dftPtr->normalizeRho(); 
		    //
		    //do an scf calculation
		    //
		    dftPtr->solve(true,true);

                    shadowKSRhoMin=dftPtr->d_rhoOutNodalValues;
                    dftPtr->d_constraintsPRefined.distribute(shadowKSRhoMin);
                    shadowKSRhoMin.update_ghost_values();
            }
            else
            {
		    //
		    //do an scf calculation
		    //
		    dftPtr->solve();
            }

	    //
	    //get force field using Gaussians
	    //
	    const std::vector<double> forceOnAtoms= dftPtr->forcePtr->getAtomsForces();

	    //
	    //compute acceleration at t + delta t
	    //
	    for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
	      {
		accelerationNew[3*iCharge + 0] = -(forceOnAtoms[3*iCharge + 0]*haPerBohrToeVPerAng)/massAtoms[iCharge];
		accelerationNew[3*iCharge + 1] = -(forceOnAtoms[3*iCharge + 1]*haPerBohrToeVPerAng)/massAtoms[iCharge];
		accelerationNew[3*iCharge + 2] = -(forceOnAtoms[3*iCharge + 2]*haPerBohrToeVPerAng)/massAtoms[iCharge];
	      }

            //compute velocity at t + delta t 
	    for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
	      {
		velocity[3*iCharge + 0] = velocityMid[3*iCharge + 0] + 0.5*accelerationNew[3*iCharge + 0]*timeStep;
		velocity[3*iCharge + 1] = velocityMid[3*iCharge + 1] + 0.5*accelerationNew[3*iCharge + 1]*timeStep;
		velocity[3*iCharge + 2] = velocityMid[3*iCharge + 2] + 0.5*accelerationNew[3*iCharge + 2]*timeStep;
	      }

	    kineticEnergy = 0.0;
	    for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
	      {
		kineticEnergy += 0.5*massAtoms[iCharge]*(velocity[3*iCharge + 0]*velocity[3*iCharge + 0] + velocity[3*iCharge + 1]*velocity[3*iCharge + 1] + velocity[3*iCharge + 2]*velocity[3*iCharge + 2]);
	      }

	    averageKineticEnergy = kineticEnergy/(3*numberGlobalCharges);
	    temperatureFromVelocities = averageKineticEnergy*2/kb;
	    kineticEnergyVector[timeIndex-startingTimeStep] = kineticEnergy/haToeV;
	    internalEnergyVector[timeIndex-startingTimeStep] = dftParameters::isXLBOMD?
                                                               dftPtr->d_shadowPotentialEnergy
                                                               :dftPtr->d_groundStateEnergy;
	    entropicEnergyVector[timeIndex-startingTimeStep] = dftPtr->d_entropicEnergy;
            totalEnergyVector[timeIndex-startingTimeStep] = kineticEnergyVector[timeIndex-startingTimeStep] +internalEnergyVector[timeIndex-startingTimeStep] -entropicEnergyVector[timeIndex-startingTimeStep];

	    //
	    //reset acceleration at time t based on the acceleration at previous time step
	    //
	    acceleration = accelerationNew;

	    pcout<<" Temperature from velocities: "<<timeIndex<<" "<<temperatureFromVelocities<<std::endl;
	    pcout<<" Kinetic Energy in Ha at timeIndex "<<timeIndex<<" "<<kineticEnergyVector[timeIndex-startingTimeStep]<<std::endl;
	    pcout<<" Internal Energy in Ha at timeIndex "<<timeIndex<<" "<<internalEnergyVector[timeIndex-startingTimeStep]<<std::endl;
	    pcout<<" Entropic Energy in Ha at timeIndex "<<timeIndex<<" "<<entropicEnergyVector[timeIndex-startingTimeStep]<<std::endl;
            pcout<<" Total Energy in Ha at timeIndex "<<timeIndex<<" "<<totalEnergyVector[timeIndex-startingTimeStep]<<std::endl;

	     std::vector<std::vector<double> > data1(timeIndex+1,std::vector<double>(1,0.0));
             std::vector<std::vector<double> > data2(timeIndex+1,std::vector<double>(1,0.0));
             std::vector<std::vector<double> > data3(timeIndex+1,std::vector<double>(1,0.0));
             std::vector<std::vector<double> > data4(timeIndex+1,std::vector<double>(1,0.0));

	    if (restartFlag==1)
	    {
		std::vector<std::vector<double> > kineticEnergyData;
		dftUtils::readFile(1,
				   kineticEnergyData,
				   "KeEngMd");

		std::vector<std::vector<double> > internalEnergyData;
		dftUtils::readFile(1,
				   internalEnergyData,
				   "IntEngMd");

		std::vector<std::vector<double> > entropicEnergyData;
		dftUtils::readFile(1,
				   entropicEnergyData,
				   "EntEngMd");

		std::vector<std::vector<double> > totalEnergyData;
		dftUtils::readFile(1,
				   totalEnergyData,
				   "TotEngMd");



		 for(int i = 0; i <= startingTimeStep; ++i)
		 {
		     data1[i][0]=kineticEnergyData[i][0];
		     data2[i][0]=internalEnergyData[i][0];
		     data3[i][0]=entropicEnergyData[i][0];
                     data4[i][0]=totalEnergyData[i][0];
		 }
	     }
	     else
	     {
		 data1[0][0]=kineticEnergyVector[0];
		 data2[0][0]=internalEnergyVector[0];
		 data3[0][0]=entropicEnergyVector[0];
                 data4[0][0]=totalEnergyVector[0];
	     }

	     for(int i = startingTimeStep+1; i <= timeIndex; ++i)
	     {
		 data1[i][0]=kineticEnergyVector[i-startingTimeStep];
		 data2[i][0]=internalEnergyVector[i-startingTimeStep];
		 data3[i][0]=entropicEnergyVector[i-startingTimeStep];
                 data4[i][0]=totalEnergyVector[i-startingTimeStep];
	     }
             MPI_Barrier(MPI_COMM_WORLD);
	     dftUtils::writeDataIntoFile(data1,
			       "KeEngMd");


	     dftUtils::writeDataIntoFile(data2,
			       "IntEngMd");


	     dftUtils::writeDataIntoFile(data3,
			       "EntEngMd");

	     dftUtils::writeDataIntoFile(data4,
			       "TotEngMd");


	    ///write velocity and acceleration data
	    std::vector<std::vector<double> > fileAccData(numberGlobalCharges,std::vector<double>(3,0.0));
	    std::vector<std::vector<double> > fileVelData(numberGlobalCharges,std::vector<double>(3,0.0));
	    std::vector<std::vector<double> > fileTemperatureData(1,std::vector<double>(1,0.0));
	    std::vector<std::vector<double> > timeIndexData(1,std::vector<double>(1,0.0));


	    for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
	      {
		fileAccData[iCharge][0]=accelerationNew[3*iCharge + 0];
		fileAccData[iCharge][1]=accelerationNew[3*iCharge + 1];
		fileAccData[iCharge][2]=accelerationNew[3*iCharge + 2];
	      }

	    for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
	      {
		fileVelData[iCharge][0]=velocity[3*iCharge + 0];
		fileVelData[iCharge][1]=velocity[3*iCharge + 1];
		fileVelData[iCharge][2]=velocity[3*iCharge + 2];
	      }
	    dftUtils::writeDataIntoFile(fileAccData,
			       "acceleration.chk");
	    dftUtils::writeDataIntoFile(fileVelData,
			       "velocity.chk");

	    fileTemperatureData[0][0]=temperatureFromVelocities;
	    dftUtils::writeDataIntoFile(fileTemperatureData,
			               "temperature.chk");

	    timeIndexData[0][0]=timeIndex;
	    dftUtils::writeDataIntoFile(timeIndexData,
			               "time.chk");

            if (dftParameters::chkType==1)
	       dftPtr->writeDomainAndAtomCoordinates();

	  }



}




template class molecularDynamics<1>;
template class molecularDynamics<2>;
template class molecularDynamics<3>;
template class molecularDynamics<4>;
template class molecularDynamics<5>;
template class molecularDynamics<6>;
template class molecularDynamics<7>;
template class molecularDynamics<8>;
template class molecularDynamics<9>;
template class molecularDynamics<10>;
template class molecularDynamics<11>;
template class molecularDynamics<12>;

}
