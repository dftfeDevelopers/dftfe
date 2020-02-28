// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2020 The Regents of the University of Michigan and DFT-FE authors.
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

        namespace internalmd
	{


	  extern "C"{
	    //
	    // lapack Ax=b
	    //
	    void dgesv_(int *N, int * NRHS, double* A, int * LDA, int* IPIV,
			double *B, int * LDB, int *INFO);

	  }

	  std::vector<double> getFractionalCoordinates(const std::vector<double> & latticeVectors,
						       const Point<3> & point,                                                                        const Point<3> & corner)
	  {
	    //
	    // recenter vertex about corner
	    //
	    std::vector<double> recenteredPoint(3);
	    for(unsigned int i = 0; i < 3; ++i)
	      recenteredPoint[i] = point[i]-corner[i];

	    std::vector<double> latticeVectorsDup = latticeVectors;

	    //
	    // to get the fractionalCoords, solve a linear
	    // system of equations
	    //
	    int N = 3;
	    int NRHS = 1;
	    int LDA = 3;
	    int IPIV[3];
	    int info;

	    dgesv_(&N, &NRHS, &latticeVectorsDup[0], &LDA, &IPIV[0], &recenteredPoint[0], &LDA,&info);
	    AssertThrow(info == 0, ExcMessage("LU solve in finding fractional coordinates failed."));
	    return recenteredPoint;
	  }

	  std::vector<double> wrapAtomsAcrossPeriodicBc(const Point<3> & cellCenteredCoord,
							const Point<3> & corner,
							const std::vector<double> & latticeVectors,
							const std::vector<bool> & periodicBc)
	  {
	    const double tol=1e-8;
	    std::vector<double> fracCoord= getFractionalCoordinates(latticeVectors,
								    cellCenteredCoord,                                                                                                corner);


	    //if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
	    //std::cout<<"Fractional Coordinates before wrapping: "<<fracCoord[0]<<" "<<fracCoord[1]<<" "<<fracCoord[2]<<std::endl;


	    //wrap fractional coordinate
	    for(unsigned int i = 0; i < 3; ++i)
	      {
		if (periodicBc[i])
		  {
		    if (fracCoord[i]<-tol)
		      fracCoord[i]+=1.0;
		    else if (fracCoord[i]>1.0+tol)
		      fracCoord[i]-=1.0;

		    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
		     std::cout<<fracCoord[i]<<" ";

		    AssertThrow(fracCoord[i]>-2.0*tol && fracCoord[i]<1.0+2.0*tol,ExcMessage("Moved atom position doesnt't lie inside the cell after wrapping across periodic boundary"));
		  }
	      }

	    //if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
	    //std::cout<<std::endl;

	    return fracCoord;
	  }

	}


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
	const double initialTemperature = dftParameters::startingTempBOMDNVE;//K
	const unsigned int restartFlag =dftParameters::restartFromChk?1:0; //1; //0;//1;
	int startingTimeStep = 0; //625;//450;// 50;//300; //0;// 300;

	double massAtomAl = 26.982;//mass proton is chosen 1 **49611.513**
	double massAtomMg= 24.305;
	const double timeStep = dftParameters::timeStepBOMD/10.0; //0.5 femtoseconds
	const unsigned int numberTimeSteps = dftParameters::numberStepsBOMD;

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
	   else if(std::fabs(charge-49.0)<1e-8)
	       massAtoms[iCharge]=114.82;
	   else if(std::fabs(charge-15.0)<1e-8)
	       massAtoms[iCharge]=30.97376;
	   else if(std::fabs(charge-8.0)<1e-8)
	       massAtoms[iCharge]=15.9994;
	   else if(std::fabs(charge-1.0)<1e-8)
	       massAtoms[iCharge]=1.00797;
	   else
	       AssertThrow(false,ExcMessage("Currently md capability is hardcoded for systems with Al, Mg, In, P, O, and H atom types only."));
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
        std::vector<double> rmsErrorRhoVector(numberTimeSteps,0.0);
        std::vector<double> rmsErrorGradRhoVector(numberTimeSteps,0.0);

        const unsigned int kmax=8;
        const double k=1.86;
        const double alpha=0.0016;
        const double c0=-36.0;
        const double c1=99.0;
        const double c2=-88.0;
        const double c3=11.0;
        const double c4=32.0;
        const double c5=-25.0;
        const double c6=8.0;
        const double c7=-1.0;
        const double diracDeltaKernelConstant=-dftParameters::diracDeltaKernelScalingConstant;
        std::deque<vectorType> approxDensityContainer(kmax);
        vectorType shadowKSRhoMin;
        vectorType atomicRho;
        vectorType approxDensityNext;
        vectorType rhoErrorVec;

	double kineticEnergy = 0.0;

	//boost::mt19937 rng(std::time(0));
	boost::mt19937 rng;
	boost::normal_distribution<> gaussianDist(0.0,initialVelocityDeviation);
	boost::variate_generator<boost::mt19937&,boost::normal_distribution<> > generator(rng,gaussianDist);
	double averageKineticEnergy;
	double temperatureFromVelocities;

	if(restartFlag == 0)
	  {
	    if (dftParameters::autoMeshStepInterpolateBOMD)
	       dftPtr->updatePrevMeshDataStructures();

	    dftPtr->solve();
            const std::vector<double> forceOnAtoms= dftPtr->forcePtr->getAtomsForces();

            dftPtr->d_matrixFreeDataPRefined.initialize_dof_vector(atomicRho);
            dftPtr->initAtomicRho(atomicRho);
            shadowKSRhoMin=dftPtr->d_rhoOutNodalValues;
            if (dftParameters::useAtomicRhoXLBOMD)
               shadowKSRhoMin-=atomicRho;

            shadowKSRhoMin.update_ghost_values();

	    //normalize shadowKSRhoMin
	    double charge = dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,
	 		                 shadowKSRhoMin);

            if (dftParameters::useAtomicRhoXLBOMD)
              shadowKSRhoMin.add(-charge/dftPtr->d_domainVolume);
            else
              shadowKSRhoMin*=((double)dftPtr->numElectrons)/charge;

            // Set approximate electron density to the exact ground state electron density in the 0th step
            if (dftParameters::isXLBOMD)
            {
                   for (unsigned int i=0; i<approxDensityContainer.size();++i)
                   {
                      approxDensityContainer[i]=shadowKSRhoMin;
                      approxDensityContainer[i].update_ghost_values();
                   }
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
		acceleration[3*iCharge + 0] = -(forceOnAtoms[3*iCharge + 0]*haPerBohrToeVPerAng)/massAtoms[iCharge]*0.0;
		acceleration[3*iCharge + 1] = -(forceOnAtoms[3*iCharge + 1]*haPerBohrToeVPerAng)/massAtoms[iCharge]*0.0;
		acceleration[3*iCharge + 2] = -(forceOnAtoms[3*iCharge + 2]*haPerBohrToeVPerAng)/massAtoms[iCharge]*0.0;
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
            rmsErrorRhoVector[0]=0.0;
            rmsErrorGradRhoVector[0]=0.0;

            pcout<<" Initial Velocity Deviation: "<<initialVelocityDeviation<<std::endl;
	    pcout<<" Kinetic Energy in Ha at timeIndex 0 "<<kineticEnergyVector[0]<<std::endl;
	    pcout<<" Internal Energy in Ha at timeIndex 0 "<<internalEnergyVector[0]<<std::endl;
	    pcout<<" Entropic Energy in Ha at timeIndex 0 "<<entropicEnergyVector[0]<<std::endl;
            pcout<<" Total Energy in Ha at timeIndex 0 "<<totalEnergyVector[0]<<std::endl;
            if (dftParameters::isXLBOMD)
            {
              pcout<<" RMS error in rho in a.u. at timeIndex 0 "<<rmsErrorRhoVector[0]<<std::endl;
              pcout<<" RMS error in grad rho in a.u. at timeIndex 0 "<<rmsErrorGradRhoVector[0]<<std::endl;
            }
	}

        double internalEnergyAccumulatedCorrection=0.0;
        double entropicEnergyAccumulatedCorrection=0.0;
	//
	//start the MD simulation
	//
	for(int timeIndex = startingTimeStep+1; timeIndex < numberTimeSteps; ++timeIndex)
	  {
             double step_time;
             MPI_Barrier(MPI_COMM_WORLD);
             step_time = MPI_Wtime();
            
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

            double update_time;
            MPI_Barrier(MPI_COMM_WORLD);
            update_time = MPI_Wtime(); 
	    //
	    //first move the mesh to current positions
	    //
	    dftPtr->updateAtomPositionsAndMoveMesh(displacements,
						   dftParameters::maxJacobianRatioFactorForMD,
						   (timeIndex ==startingTimeStep+1 && restartFlag==1)?true:false);

            /*
	    if (d_isAtomsGaussianDisplacementsReadFromFile)
	    {
		dftPtr->updateAtomPositionsAndMoveMesh(dftPtr->d_atomsDisplacementsGaussianRead,1e+4,true);
		dftPtr->d_isAtomsGaussianDisplacementsReadFromFile=false;
	    }
            */

	    if(dftParameters::verbosity>=5)
	      {
		pcout<<"New atomic positions on the Mesh: "<<std::endl;
		for(int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
		  {
		    pcout<<"Charge Id: "<<iCharge<<" "<<dftPtr->atomLocations[iCharge][2]<<" "<<dftPtr->atomLocations[iCharge][3]<<" "<<dftPtr->atomLocations[iCharge][4]<<std::endl;
		  }
	      }

            if (dftPtr->d_autoMesh==1)
                dftPtr->d_matrixFreeDataPRefined.initialize_dof_vector(atomicRho);
            dftPtr->initAtomicRho(atomicRho);
       
            MPI_Barrier(MPI_COMM_WORLD);
            update_time = MPI_Wtime() - update_time;
            if (dftParameters::verbosity>=1)
                pcout<<"Time taken for updateAtomPositionsAndMoveMesh and initAtomicRho: "<<update_time<<std::endl;
 
            double rmsErrorRho=0.0;
            double rmsErrorGradRho=0.0;
            if (dftParameters::isXLBOMD)
            {
                    if (dftPtr->d_autoMesh==1 && dftParameters::autoMeshStepInterpolateBOMD)
                    {
                        pcout<<".............Auto meshing step: interpolation of approximate density container to new mesh.............."<<std::endl;
                        //interpolate all the vectors in the approximate density container on the current mesh
                        std::vector<vectorType* > fieldsPtrsPrevious(approxDensityContainer.size()+1);
                        std::vector<vectorType* > fieldsPtrsCurrent(approxDensityContainer.size()+1);
                        std::vector<vectorType> fieldsCurrent(approxDensityContainer.size()+1);

                        for (unsigned int i = 0; i < approxDensityContainer.size(); i++)
                        {
                            fieldsPtrsPrevious[i]=&approxDensityContainer[i];
                            dftPtr->d_matrixFreeDataPRefined.initialize_dof_vector(fieldsCurrent[i]);
                            fieldsPtrsCurrent[i]=&fieldsCurrent[i];
                        }
                        fieldsPtrsPrevious[approxDensityContainer.size()]=&shadowKSRhoMin;
                        dftPtr->d_matrixFreeDataPRefined.initialize_dof_vector(fieldsCurrent[approxDensityContainer.size()]);
                        fieldsPtrsCurrent[approxDensityContainer.size()]=&fieldsCurrent[approxDensityContainer.size()]; 

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
                        
                        shadowKSRhoMin=fieldsCurrent[approxDensityContainer.size()];
                        shadowKSRhoMin.update_ghost_values();

                        //re-normalize
                        for (unsigned int i = 0; i < approxDensityContainer.size()+1; i++)
                        {
	                   const double charge = dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,
						     i<approxDensityContainer.size()?approxDensityContainer[i]:shadowKSRhoMin);
	                   pcout<<"Total Charge before Normalizing interpolated field:  "<<charge<<std::endl;
	       
	 	           const double scalingFactor = ((double)dftPtr->numElectrons)/charge;

                           if (i<approxDensityContainer.size())
		              approxDensityContainer[i] *= scalingFactor;
                           else
                              shadowKSRhoMin *=scalingFactor;
		           pcout<<"Total Charge after Normalizing interpolated field:  "<<dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,
                                                                                         i<approxDensityContainer.size()?
                                                                                         approxDensityContainer[i]:shadowKSRhoMin)<<std::endl;
                        }

                        dftPtr->updatePrevMeshDataStructures();
                        pcout<<".............Auto meshing step: interpolation and re-normalization completed.............."<<std::endl;
                    }
                    else if (dftPtr->d_autoMesh==1)
                    {
                        dftPtr->solve();
                        
			shadowKSRhoMin=dftPtr->d_rhoOutNodalValues;
			if (dftParameters::useAtomicRhoXLBOMD)
			   shadowKSRhoMin-=atomicRho;

			shadowKSRhoMin.update_ghost_values();

			//normalize shadowKSRhoMin
			double charge = dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,
							       shadowKSRhoMin);
			if (dftParameters::useAtomicRhoXLBOMD)
		  	    shadowKSRhoMin.add(-charge/dftPtr->d_domainVolume);
                        else
                            shadowKSRhoMin *= ((double)dftPtr->numElectrons)/charge;  

			for (unsigned int i=0; i<approxDensityContainer.size();++i)
			{
			      approxDensityContainer[i]=shadowKSRhoMin;
			      approxDensityContainer[i].update_ghost_values();
			}
                    }
                    else
		    {
			double xlbomdpre_time;
			MPI_Barrier(MPI_COMM_WORLD);
			xlbomdpre_time = MPI_Wtime(); 

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
								 +c5*approxDensityContainer[containerSizeCurrent-6].local_element(i)
								 +c6*approxDensityContainer[containerSizeCurrent-7].local_element(i)
								 +c7*approxDensityContainer[containerSizeCurrent-8].local_element(i));
				approxDensityContainer.pop_front();
			}
			approxDensityContainer.push_back(approxDensityNext);
			approxDensityContainer.back().update_ghost_values();                    
	 
			//normalize approxDensityVec
			double charge = dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,
							     approxDensityContainer.back());
			pcout<<"Total Charge before Normalizing new approxDensityVec:  "<<charge<<std::endl;
		       
   		        if (dftParameters::useAtomicRhoXLBOMD)
 			   approxDensityContainer.back().add(-charge/dftPtr->d_domainVolume);
                        else
                           approxDensityContainer.back() *= ((double)dftPtr->numElectrons)/charge;
			pcout<<"Total Charge after Normalizing new approxDensityVec:  "<<dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,approxDensityContainer.back())<<std::endl;

                        dftPtr->d_rhoInNodalValues=approxDensityContainer.back();
                        if (dftParameters::useAtomicRhoXLBOMD)
                          dftPtr->d_rhoInNodalValues+=atomicRho;
                         
                        dftPtr->d_rhoInNodalValues.update_ghost_values();
			dftPtr->interpolateNodalDataToQuadratureData(dftPtr->d_matrixFreeDataPRefined,
								dftPtr->d_rhoInNodalValues,
								*(dftPtr->rhoInValues),
								*(dftPtr->gradRhoInValues),
                                                                *(dftPtr->gradRhoInValues),
								 dftParameters::xc_id == 4);		
	     
			dftPtr->normalizeRho();

			MPI_Barrier(MPI_COMM_WORLD);
			xlbomdpre_time = MPI_Wtime() - xlbomdpre_time;
			if (dftParameters::verbosity>=1)
			   pcout<<"Time taken for xlbomd preinitializations: "<<xlbomdpre_time<<std::endl;

                        double shadowsolve_time;
                        MPI_Barrier(MPI_COMM_WORLD);
                        shadowsolve_time = MPI_Wtime(); 

			//do an scf calculation
			dftPtr->solve(true,true);

			MPI_Barrier(MPI_COMM_WORLD);
			shadowsolve_time = MPI_Wtime() - shadowsolve_time;
			if (dftParameters::verbosity>=1)
			   pcout<<"Time taken for xlbomd shadow potential solve: "<<shadowsolve_time<<std::endl;

			double xlbomdpost_time;
			MPI_Barrier(MPI_COMM_WORLD);
			xlbomdpost_time = MPI_Wtime(); 

			shadowKSRhoMin=dftPtr->d_rhoOutNodalValues;
			if (dftParameters::useAtomicRhoXLBOMD)
			   shadowKSRhoMin-=atomicRho;

			shadowKSRhoMin.update_ghost_values();

			//normalize shadowKSRhoMin
			charge = dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,
					       shadowKSRhoMin);
			if (dftParameters::useAtomicRhoXLBOMD)
		  	    shadowKSRhoMin.add(-charge/dftPtr->d_domainVolume);
                        else
                            shadowKSRhoMin *= ((double)dftPtr->numElectrons)/charge; 


			rhoErrorVec=approxDensityContainer.back();
			rhoErrorVec-=shadowKSRhoMin;
			rmsErrorRho=std::sqrt(dftPtr->fieldl2Norm(dftPtr->d_matrixFreeDataPRefined,rhoErrorVec)/dftPtr->d_domainVolume);
			rmsErrorGradRho=std::sqrt(dftPtr->fieldGradl2Norm(dftPtr->d_matrixFreeDataPRefined,rhoErrorVec)/dftPtr->d_domainVolume);
			MPI_Barrier(MPI_COMM_WORLD);
			xlbomdpost_time = MPI_Wtime() - xlbomdpost_time;
			if (dftParameters::verbosity>=1)
			   pcout<<"Time taken for xlbomd post solve operations: "<<xlbomdpost_time<<std::endl; 
		    }
            }
            else
            {
                    if (!dftPtr->d_autoMesh==1)
                       if (!(timeIndex ==startingTimeStep+1 && restartFlag==1))
                       {
                            dftPtr->d_rhoInNodalValues=shadowKSRhoMin;
                            if (dftParameters::useAtomicRhoXLBOMD)
                               dftPtr->d_rhoInNodalValues+=atomicRho;
                         
                            dftPtr->d_rhoInNodalValues.update_ghost_values();
			    dftPtr->interpolateNodalDataToQuadratureData(dftPtr->d_matrixFreeDataPRefined,
								dftPtr->d_rhoInNodalValues,
								*(dftPtr->rhoInValues),
								*(dftPtr->gradRhoInValues),
                                                                *(dftPtr->gradRhoInValues),
								 dftParameters::xc_id == 4);	
                            dftPtr->normalizeRho();
                       }

		    //
		    //do an scf calculation
		    //
		    dftPtr->solve();

	            shadowKSRhoMin=dftPtr->d_rhoOutNodalValues;
		    if (dftParameters::useAtomicRhoXLBOMD)
		        shadowKSRhoMin-=atomicRho;

		    shadowKSRhoMin.update_ghost_values();

		    //normalize shadowKSRhoMin
		    double charge = dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,
					       shadowKSRhoMin);
			
                    if (dftParameters::useAtomicRhoXLBOMD)
		        shadowKSRhoMin.add(-charge/dftPtr->d_domainVolume);
                    else
                        shadowKSRhoMin *= ((double)dftPtr->numElectrons)/charge;  
            }

            double bomdpost_time;
	    MPI_Barrier(MPI_COMM_WORLD);
	    bomdpost_time = MPI_Wtime();

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
	    internalEnergyVector[timeIndex-startingTimeStep] = (dftParameters::isXLBOMD && dftPtr->d_autoMesh!=1)?
                                                               dftPtr->d_shadowPotentialEnergy
                                                               :dftPtr->d_groundStateEnergy-internalEnergyAccumulatedCorrection;
	    entropicEnergyVector[timeIndex-startingTimeStep] = dftPtr->d_entropicEnergy-entropicEnergyAccumulatedCorrection;
            totalEnergyVector[timeIndex-startingTimeStep] = kineticEnergyVector[timeIndex-startingTimeStep] +internalEnergyVector[timeIndex-startingTimeStep] -entropicEnergyVector[timeIndex-startingTimeStep];
            rmsErrorRhoVector[timeIndex-startingTimeStep] = rmsErrorRho;
            rmsErrorGradRhoVector[timeIndex-startingTimeStep] = rmsErrorGradRho;

	    //
	    //reset acceleration at time t based on the acceleration at previous time step
	    //
	    acceleration = accelerationNew;

	    pcout<<" Temperature from velocities: "<<timeIndex<<" "<<temperatureFromVelocities<<std::endl;
	    pcout<<" Kinetic Energy in Ha at timeIndex "<<timeIndex<<" "<<kineticEnergyVector[timeIndex-startingTimeStep]<<std::endl;
	    pcout<<" Internal Energy in Ha at timeIndex "<<timeIndex<<" "<<internalEnergyVector[timeIndex-startingTimeStep]<<std::endl;
	    pcout<<" Entropic Energy in Ha at timeIndex "<<timeIndex<<" "<<entropicEnergyVector[timeIndex-startingTimeStep]<<std::endl;
            pcout<<" Total Energy in Ha at timeIndex "<<timeIndex<<" "<<totalEnergyVector[timeIndex-startingTimeStep]<<std::endl;
            if (dftParameters::isXLBOMD)
            {
              pcout<<" RMS error in rho in a.u. at timeIndex "<<timeIndex<<" "<<rmsErrorRhoVector[timeIndex-startingTimeStep]<<std::endl;
              pcout<<" RMS error in grad rho in a.u. at timeIndex "<<timeIndex<<" "<<rmsErrorGradRhoVector[timeIndex-startingTimeStep]<<std::endl;
            }

	     std::vector<std::vector<double> > data1(timeIndex+1,std::vector<double>(1,0.0));
             std::vector<std::vector<double> > data2(timeIndex+1,std::vector<double>(1,0.0));
             std::vector<std::vector<double> > data3(timeIndex+1,std::vector<double>(1,0.0));
             std::vector<std::vector<double> > data4(timeIndex+1,std::vector<double>(1,0.0));
             std::vector<std::vector<double> > data5(timeIndex+1,std::vector<double>(1,0.0));
             std::vector<std::vector<double> > data6(timeIndex+1,std::vector<double>(1,0.0));

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


		std::vector<std::vector<double> > rmsErrorRhoData;
                std::vector<std::vector<double> > rmsErrorGradRhoData;
                if (dftParameters::isXLBOMD)
                {
			dftUtils::readFile(1,
					   totalEnergyData,
					   "RMSErrorRhoMd");

			dftUtils::readFile(1,
					   totalEnergyData,
					   "RMSErrorGradRhoMd");
                }
		for(int i = 0; i <= startingTimeStep; ++i)
		{
		     data1[i][0]=kineticEnergyData[i][0];
		     data2[i][0]=internalEnergyData[i][0];
		     data3[i][0]=entropicEnergyData[i][0];
                     data4[i][0]=totalEnergyData[i][0];
                     if (dftParameters::isXLBOMD)
                     {
                        data5[i][0]=rmsErrorRhoData[i][0];
                        data6[i][0]=rmsErrorGradRhoData[i][0];
                     }
		}
	     }
	     else
	     {
		 data1[0][0]=kineticEnergyVector[0];
		 data2[0][0]=internalEnergyVector[0];
		 data3[0][0]=entropicEnergyVector[0];
                 data4[0][0]=totalEnergyVector[0];
                 if (dftParameters::isXLBOMD)
                 {
                     data5[0][0]=rmsErrorRhoVector[0];
                     data6[0][0]=rmsErrorGradRhoVector[0];
                 }
	     }

	     for(int i = startingTimeStep+1; i <= timeIndex; ++i)
	     {
		 data1[i][0]=kineticEnergyVector[i-startingTimeStep];
		 data2[i][0]=internalEnergyVector[i-startingTimeStep];
		 data3[i][0]=entropicEnergyVector[i-startingTimeStep];
                 data4[i][0]=totalEnergyVector[i-startingTimeStep];
                 if (dftParameters::isXLBOMD)
                 {
                     data5[i][0]=rmsErrorRhoVector[i-startingTimeStep];
                     data6[i][0]=rmsErrorGradRhoVector[i-startingTimeStep];
                 }
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

             if (dftParameters::isXLBOMD)
             {
  	         dftUtils::writeDataIntoFile(data5,
			         "RMSErrorRhoMd");
  	         dftUtils::writeDataIntoFile(data6,
			         "RMSErrorGradRhoMd");
             }

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


	    MPI_Barrier(MPI_COMM_WORLD);
	    bomdpost_time = MPI_Wtime() - bomdpost_time;
	    if (dftParameters::verbosity>=1)
	      pcout<<"Time taken for bomd post solve operations: "<<bomdpost_time<<std::endl; 

            MPI_Barrier(MPI_COMM_WORLD);
            step_time = MPI_Wtime() - step_time;
            if (dftParameters::verbosity>=1)
                pcout<<"Time taken for md step: "<<step_time<<std::endl;
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
