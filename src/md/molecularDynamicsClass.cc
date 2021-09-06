#include <boost/generator_iterator.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <dft.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include <force.h>
#include <vector>
#include <molecularDynamicsClass.h>

#ifdef DFTFE_WITH_GPU
#  include <vectorUtilitiesCUDA.h>
#endif

namespace dftfe
{




    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    molecularDynamicsClass<FEOrder, FEOrderElectro>::molecularDynamicsClass(dftClass<FEOrder, FEOrderElectro> *_dftPtr,
    const MPI_Comm &                   mpi_comm_replica)
    : dftPtr(_dftPtr)
    , mpi_communicator(mpi_comm_replica)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_comm_replica))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_comm_replica))
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
      {
        startingTimeStep        = 0;
        TimeIndex               = 0;
        timeStep                =
                dftParameters::timeStepBOMD*0.09822694541304546435; // Conversion factor from femteseconds:
                              // 0.09822694541304546435 based on NIST constants
        numberofSteps           =
                dftParameters::numberStepsBOMD;        
        
         startingTemperature     = 
                dftParameters::startingTempBOMD;
         thermostatTimeConstant  = 
                dftParameters::thermostatTimeConstantBOMD;
         thermostatType          =
                dftParameters::tempControllerTypeBOMD;
        numberGlobalCharges      = 
                dftParameters::natoms; 
        pcout << "----------------------Starting Initialization of BOMD-------------------------" << std::endl;        
        pcout << "Starting Temperature from Input "
              << startingTemperature << std::endl; 
       } 

        
    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    void
    molecularDynamicsClass<FEOrder, FEOrderElectro>::runMD()
    {
 
        std::vector<double> massAtoms(numberGlobalCharges);
        //
        // read atomic masses in amu
        //
        std::vector<std::vector<double>> atomTypesMasses;
        dftUtils::readFile(2, atomTypesMasses, dftParameters::atomicMassesFile);
        AssertThrow(dftPtr->atomTypes.size() == atomTypesMasses.size(),
                ExcMessage("DFT-FE Error: check ATOM MASSES FILE"));

        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
        {
          bool isFound = false;
            for (int jatomtype = 0; jatomtype < atomTypesMasses.size(); ++jatomtype)
              {
                const double charge = dftPtr->atomLocations[iCharge][0];
                if (std::fabs(charge - atomTypesMasses[jatomtype][0]) < 1e-8)
                  {
                    massAtoms[iCharge] = atomTypesMasses[jatomtype][1];
                    isFound            = true;
                  }
              }

          AssertThrow(isFound,
                    ExcMessage("DFT-FE Error: check ATOM MASSES FILE"));
        }

        std::vector<dealii::Tensor<1, 3, double>> displacements(
        numberGlobalCharges);
        std::vector<double> velocity(3 * numberGlobalCharges, 0.0);
        std::vector<double> force(3 * numberGlobalCharges, 0.0);
        std::vector<double> InternalEnergyVector(numberofSteps, 0.0);
        std::vector<double> EntropicEnergyVector(numberofSteps, 0.0);
        std::vector<double> KineticEnergyVector(numberofSteps, 0.0);
        std::vector<double> TotalEnergyVector(numberofSteps, 0.0);   
        double totMass = 0.0; 
        double velocityDistribution;            
        restartFlag = ((dftParameters::chkType == 1 || dftParameters::chkType == 3) &&
       dftParameters::restartMdFromChk) ?        1 :        0; // 1; //0;//1;

//--------------------Starting Initialization ----------------------------------------------//
        double KineticEnergy=0.0 , TemperatureFromVelocities = 0.0;
        double Px=0.0, Py=0.0 , Pz = 0.0;
        //Initialise Velocity
        if (this_mpi_process == 0)
          { 
            for(int jatomtype = 0; jatomtype < atomTypesMasses.size(); ++jatomtype)
              {
                    double Mass = atomTypesMasses[jatomtype][1];
                    velocityDistribution = sqrt(kB*startingTemperature/Mass);
               
                    pcout << "Initialising Velocities of species no: "<<jatomtype<<" mass in amu: "<<Mass
                          <<"Velocity Deviation"<<velocityDistribution<<std::endl;                    
                    boost::mt19937               rng;
                    boost::normal_distribution<> gaussianDist(0.0, velocityDistribution);
                    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<>>
                            generator(rng, gaussianDist);
                for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
                  { 

                    if (std::fabs(dftPtr->atomLocations[iCharge][0] - atomTypesMasses[jatomtype][0]) < 1e-8)
                    {
                      velocity[3 * iCharge + 0] = generator();
                      velocity[3 * iCharge + 1] = generator();
                      velocity[3 * iCharge + 2] = generator();
                    }                  
                  }
              }   
          }

            for (unsigned int i = 0; i < numberGlobalCharges * 3; ++i)
            {
              velocity[i] =
              dealii::Utilities::MPI::sum(velocity[i], mpi_communicator);
            }



        // compute KEinetic Energy and COM vecloity
        totMass = 0.0;
        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            totMass += massAtoms[iCharge];
            Px += massAtoms[iCharge]*velocity[3*iCharge+0];
            Py += massAtoms[iCharge]*velocity[3*iCharge+1];
            Pz += massAtoms[iCharge]*velocity[3*iCharge+2];          
          }
        //Correcting for COM velocity to be 0  

        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            velocity[3*iCharge+0] = (massAtoms[iCharge]*velocity[3*iCharge+0] - Px/numberGlobalCharges)/ massAtoms[iCharge];   
            velocity[3*iCharge+1] = (massAtoms[iCharge]*velocity[3*iCharge+1] - Py/numberGlobalCharges)/ massAtoms[iCharge];
            velocity[3*iCharge+2] = (massAtoms[iCharge]*velocity[3*iCharge+2] - Pz/numberGlobalCharges)/ massAtoms[iCharge];


            KineticEnergy +=
              0.5 * massAtoms[iCharge] *
              (velocity[3 * iCharge + 0] * velocity[3 * iCharge + 0] +
               velocity[3 * iCharge + 1] * velocity[3 * iCharge + 1] +
               velocity[3 * iCharge + 2] * velocity[3 * iCharge + 2]);   

          
          }
          TemperatureFromVelocities = 2.0/3.0/double(numberGlobalCharges-1)*KineticEnergy/(kB);  
          pcout << "Temperature computed from Velocities: "
              << TemperatureFromVelocities << std::endl;       

          // Correcting velocity to match init Temperature
        double gamma = sqrt( startingTemperature / TemperatureFromVelocities);

        for (int i = 0; i < 3 * numberGlobalCharges; ++i)
          {
            velocity[i] = gamma * velocity[i];
          }


          KineticEnergy = 0.0;
          for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            KineticEnergy +=
              0.5 * massAtoms[iCharge] *
              (velocity[3 * iCharge + 0] * velocity[3 * iCharge + 0] +
               velocity[3 * iCharge + 1] * velocity[3 * iCharge + 1] +
               velocity[3 * iCharge + 2] * velocity[3 * iCharge + 2]);
          }                   
          TemperatureFromVelocities = 2.0/3.0/double(numberGlobalCharges-1)*KineticEnergy/(kB);  
 

        dftPtr->solve(true, false, false, false);
        KineticEnergyVector[0]  = KineticEnergy / haToeV;
        InternalEnergyVector[0] = dftPtr->d_groundStateEnergy;
        EntropicEnergyVector[0] = dftPtr->d_entropicEnergy;
        TotalEnergyVector[0]    = KineticEnergyVector[0] +
                               InternalEnergyVector[0] -
                               EntropicEnergyVector[0];
        if (dftParameters::verbosity >= 1)
          {
            pcout << "Velocity of atoms " << std::endl;
            for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
              {
                pcout << "Charge Id: " << iCharge << " "
                      << velocity[3*iCharge+0] << " "
                      << velocity[3*iCharge+1] << " "
                      << velocity[3*iCharge+2] << std::endl;
              }         
          
          }            
       
       
        pcout << "---------------MD 0th STEP------------------ " <<  std::endl; 
        pcout << " Kinetic Energy in Ha at timeIndex 0 "
              << KineticEnergyVector[0] << std::endl;
        pcout << " Internal Energy in Ha at timeIndex 0 "
              << InternalEnergyVector[0] << std::endl;
        pcout << " Entropic Energy in Ha at timeIndex 0 "
              << EntropicEnergyVector[0] << std::endl;
        pcout << " Total Energy in Ha at timeIndex 0 "
              << TotalEnergyVector[0]  << std::endl;
        pcout << " Temperature from Velocities"
              << TemperatureFromVelocities  << std::endl;              

        MPI_Barrier(MPI_COMM_WORLD);
//--------------------Completed Initialization ----------------------------------------------//

//--------------------Choosing Ensemble ----------------------------------------------//
        if( thermostatType =="NO_CONTROL")
            {
                mdNVE(KineticEnergyVector,InternalEnergyVector,EntropicEnergyVector,TotalEnergyVector, displacements, velocity, force,massAtoms );
            }
        else if( thermostatType =="RESCALE")
                {
                  mdNVTrescaleThermostat(KineticEnergyVector,InternalEnergyVector,EntropicEnergyVector,TotalEnergyVector, displacements, velocity, force,massAtoms );      
                }    
        else if( thermostatType =="NOSE_HOVER_CHAINS")
                { 
                    mdNVTnosehoverchainsThermostat(KineticEnergyVector,InternalEnergyVector,EntropicEnergyVector,TotalEnergyVector, displacements, velocity, force,massAtoms );
                  
                }

        pcout << "MD run completed" << std::endl;    
    
    }

    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    void
    molecularDynamicsClass<FEOrder, FEOrderElectro>::mdNVE(std::vector<double> &KineticEnergyVector ,
                                                           std::vector<double> &InternalEnergyVector ,
                                                            std::vector<double> &EntropicEnergyVector , 
                                                            std::vector<double> &TotalEnergyVector ,
                                                            std::vector<dealii::Tensor<1, 3, double>> &displacements ,
                                                            std::vector<double> &velocity ,
                                                            std::vector<double> &force, 
                                                            std::vector<double> atomMass  )
    {
      pcout << "---------------MDNVE() called ------------------ " <<  std::endl;

        
      
        double KineticEnergy;
        double TemperatureFromVelocities;
        for(TimeIndex=1; TimeIndex< numberofSteps;TimeIndex++)
        {       
            double step_time;
            MPI_Barrier(MPI_COMM_WORLD);
            step_time = MPI_Wtime();   

            velocityVerlet(velocity, displacements,atomMass,KineticEnergy, force);
            MPI_Barrier(MPI_COMM_WORLD);
   
            KineticEnergyVector[TimeIndex]  = KineticEnergy / haToeV;
            InternalEnergyVector[TimeIndex] = dftPtr->d_groundStateEnergy;
            EntropicEnergyVector[TimeIndex] = dftPtr->d_entropicEnergy;
            TotalEnergyVector[TimeIndex]    = KineticEnergyVector[TimeIndex] +
                               InternalEnergyVector[TimeIndex] -
                               EntropicEnergyVector[TimeIndex]; 
            TemperatureFromVelocities = 2.0/3.0/double(numberGlobalCharges-1)*KineticEnergy/(kB);                              

            //Based on verbose print required MD details...
            MPI_Barrier(MPI_COMM_WORLD);
            step_time = MPI_Wtime() - step_time;
            if (dftParameters::verbosity >= 1)
            { 
              pcout << "---------------MD STEP: "<<TimeIndex<<" ------------------ " <<  std::endl;
              pcout << "Time taken for md step: " << step_time << std::endl;
               pcout << " Temperature from velocities: " << TimeIndex << " "
              << TemperatureFromVelocities << std::endl;
              pcout << " Kinetic Energy in Ha at timeIndex " << TimeIndex << " "
              << KineticEnergyVector[TimeIndex] << std::endl;
              pcout << " Internal Energy in Ha at timeIndex " << TimeIndex<< " "
              << InternalEnergyVector[TimeIndex]
              << std::endl;
              pcout << " Entropic Energy in Ha at timeIndex " << TimeIndex << " "
              << EntropicEnergyVector[TimeIndex]
              << std::endl;
              pcout << " Total Energy in Ha at timeIndex " << TimeIndex << " "
              << TotalEnergyVector[TimeIndex] << std::endl;
            }


        }

    }

    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    void
    molecularDynamicsClass<FEOrder, FEOrderElectro>::mdNVTrescaleThermostat(std::vector<double> &KineticEnergyVector ,
                                                           std::vector<double> &InternalEnergyVector ,
                                                            std::vector<double> &EntropicEnergyVector , 
                                                            std::vector<double> &TotalEnergyVector ,
                                                            std::vector<dealii::Tensor<1, 3, double>> &displacements ,
                                                            std::vector<double> &velocity ,
                                                            std::vector<double> &force, 
                                                            std::vector<double> atomMass  )
    {
      
      
        double KineticEnergy;
        double TemperatureFromVelocities;
        for(TimeIndex=1; TimeIndex< numberofSteps;TimeIndex++)
        {       
            double step_time;
            MPI_Barrier(MPI_COMM_WORLD);
            step_time = MPI_Wtime();   

            velocityVerlet(velocity, displacements,atomMass,KineticEnergy, force);
            //MPI_Barrier(MPI_COMM_WORLD);
            TemperatureFromVelocities = 2.0/3.0/double(numberGlobalCharges-1)*KineticEnergy/(kB);
            if(TimeIndex%thermostatTimeConstant==0)
            {
              RescaleVelocities(velocity,KineticEnergy,atomMass,TemperatureFromVelocities );
            }

            MPI_Barrier(MPI_COMM_WORLD);   
            KineticEnergyVector[TimeIndex]  = KineticEnergy / haToeV;
            InternalEnergyVector[TimeIndex] = dftPtr->d_groundStateEnergy;
            EntropicEnergyVector[TimeIndex] = dftPtr->d_entropicEnergy;
            TotalEnergyVector[TimeIndex]    = KineticEnergyVector[TimeIndex] +
                               InternalEnergyVector[TimeIndex] -
                               EntropicEnergyVector[TimeIndex]; 
            TemperatureFromVelocities = 2.0/3.0/double(numberGlobalCharges-1)*KineticEnergy/(kB);                              

            //Based on verbose print required MD details...
            MPI_Barrier(MPI_COMM_WORLD);
            step_time = MPI_Wtime() - step_time;
            if (dftParameters::verbosity >= 1)
            { 
              pcout << "---------------MD STEP: "<<TimeIndex<<" ------------------ " <<  std::endl;
              pcout << "Time taken for md step: " << step_time << std::endl;
               pcout << " Temperature from velocities: " << TimeIndex << " "
              << TemperatureFromVelocities << std::endl;
              pcout << " Kinetic Energy in Ha at timeIndex " << TimeIndex << " "
              << KineticEnergyVector[TimeIndex] << std::endl;
              pcout << " Internal Energy in Ha at timeIndex " << TimeIndex<< " "
              << InternalEnergyVector[TimeIndex]
              << std::endl;
              pcout << " Entropic Energy in Ha at timeIndex " << TimeIndex << " "
              << EntropicEnergyVector[TimeIndex]
              << std::endl;
              pcout << " Total Energy in Ha at timeIndex " << TimeIndex << " "
              << TotalEnergyVector[TimeIndex] << std::endl;
            }


        }

    }
    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    void
    molecularDynamicsClass<FEOrder, FEOrderElectro>::mdNVTnosehoverchainsThermostat(std::vector<double> &KineticEnergyVector ,
                                                           std::vector<double> &InternalEnergyVector ,
                                                            std::vector<double> &EntropicEnergyVector , 
                                                            std::vector<double> &TotalEnergyVector ,
                                                            std::vector<dealii::Tensor<1, 3, double>> &displacements ,
                                                            std::vector<double> &velocity ,
                                                            std::vector<double> &force, 
                                                            std::vector<double> atomMass  )
    {


        
      
        double KineticEnergy;
        double TemperatureFromVelocities;
        TemperatureFromVelocities = 2.0/3.0/double(numberGlobalCharges-1)*KineticEnergyVector[0]/(kB); 
        std::vector<double> ThermostatMass(2,0.0);
        std::vector<double> Thermostatvelocity(2,0.0);
        std::vector<double> Thermostatposition(2,0.0);      
        thermostatTimeConstant*=timeStep;  
        ThermostatMass.at(0) = 3*(numberGlobalCharges-1)*kB*startingTemperature*(thermostatTimeConstant*thermostatTimeConstant);
        ThermostatMass.at(1) = kB*startingTemperature*(thermostatTimeConstant*thermostatTimeConstant);
        for(TimeIndex=1; TimeIndex< numberofSteps;TimeIndex++)
        {       
            double step_time;
            MPI_Barrier(MPI_COMM_WORLD);
            step_time = MPI_Wtime();   
            NoseHoverChains(velocity, Thermostatvelocity,Thermostatposition, ThermostatMass,KineticEnergyVector[TimeIndex-1]*haToeV,TemperatureFromVelocities );
            velocityVerlet(velocity, displacements,atomMass,KineticEnergy, force);
            TemperatureFromVelocities = 2.0/3.0/double(numberGlobalCharges-1)*KineticEnergy/(kB); 
            NoseHoverChains(velocity, Thermostatvelocity,Thermostatposition, ThermostatMass,KineticEnergy,TemperatureFromVelocities );          
            KineticEnergy = 0.0;
            for(int iCharge=0; iCharge < numberGlobalCharges; iCharge++)
              {
                KineticEnergy +=0.5*atomMass[iCharge]*(velocity[3*iCharge+0]*velocity[3*iCharge+0] + 
                                                      velocity[3*iCharge+1]*velocity[3*iCharge+1] +
                                                      velocity[3*iCharge+2]*velocity[3*iCharge+2]);
              }


            MPI_Barrier(MPI_COMM_WORLD);   
            KineticEnergyVector[TimeIndex]  = KineticEnergy / haToeV;
            InternalEnergyVector[TimeIndex] = dftPtr->d_groundStateEnergy;
            EntropicEnergyVector[TimeIndex] = dftPtr->d_entropicEnergy;
            TotalEnergyVector[TimeIndex]    = KineticEnergyVector[TimeIndex] +
                               InternalEnergyVector[TimeIndex] -
                               EntropicEnergyVector[TimeIndex]; 
            TemperatureFromVelocities = 2.0/3.0/double(numberGlobalCharges-1)*KineticEnergy/(kB);                              

            //Based on verbose print required MD details...
            MPI_Barrier(MPI_COMM_WORLD);
            step_time = MPI_Wtime() - step_time;
            if (dftParameters::verbosity >= 1)
            { 
              pcout << "---------------MD STEP: "<<TimeIndex<<" ------------------ " <<  std::endl;
              pcout << "Time taken for md step: " << step_time << std::endl;
               pcout << " Temperature from velocities: " << TimeIndex << " "
              << TemperatureFromVelocities << std::endl;
              pcout << " Kinetic Energy in Ha at timeIndex " << TimeIndex << " "
              << KineticEnergyVector[TimeIndex] << std::endl;
              pcout << " Internal Energy in Ha at timeIndex " << TimeIndex<< " "
              << InternalEnergyVector[TimeIndex]
              << std::endl;
              pcout << " Entropic Energy in Ha at timeIndex " << TimeIndex << " "
              << EntropicEnergyVector[TimeIndex]
              << std::endl;
              pcout << " Total Energy in Ha at timeIndex " << TimeIndex << " "
              << TotalEnergyVector[TimeIndex] << std::endl;
            }


        }

    }

    
    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    void    
    molecularDynamicsClass<FEOrder, FEOrderElectro>::velocityVerlet(std::vector<double> &v , std::vector<dealii::Tensor<1, 3, double>> &r, 
                                                                    std::vector<double> atomMass, double &KE, std::vector<double> &forceOnAtoms  )
    {    
        
        int i;
        double totalKE;
        KE = 0.0;
        double dt = timeStep;
        double dt_2 = dt/2;
        forceOnAtoms =dftPtr->forcePtr->getAtomsForces();        
        for(i=0; i < numberGlobalCharges; i++)
            {   
                /*Computing New position as taylor expansion about r(t) O(dt^3) */
                r[i][0] = ( dt*v[3*i+0] - dt*dt_2*forceOnAtoms[3*i+0]/atomMass[i]* haPerBohrToeVPerAng)* AngTobohr; //New position of x cordinate
                r[i][1] = ( dt*v[3*i+1] - dt*dt_2*forceOnAtoms[3*i+1]/atomMass[i]* haPerBohrToeVPerAng)* AngTobohr; // New Position of Y cordinate
                r[i][2] = ( dt*v[3*i+2] - dt*dt_2*forceOnAtoms[3*i+2]/atomMass[i]* haPerBohrToeVPerAng)* AngTobohr; // New POsition of Z cordinate


                /* Computing velocity from v(t) to v(t+dt/2) */
                v[3*i+0] = v[3*i+0] - forceOnAtoms[3*i+0]/atomMass[i]*dt_2* haPerBohrToeVPerAng;
                v[3*i+1] = v[3*i+1] - forceOnAtoms[3*i+1]/atomMass[i]*dt_2* haPerBohrToeVPerAng;
                v[3*i+2] = v[3*i+2] - forceOnAtoms[3*i+2]/atomMass[i]*dt_2* haPerBohrToeVPerAng;
            }
        double update_time;
        MPI_Barrier(MPI_COMM_WORLD);
        update_time = MPI_Wtime();
        //
        // first move the mesh to current positions
        //
        dftPtr->updateAtomPositionsAndMoveMesh(
          r,
          dftParameters::maxJacobianRatioFactorForMD,
          (TimeIndex == startingTimeStep + 1 && restartFlag == 1) ? true :
                                                                    false);


        if (dftParameters::verbosity >= 5)
          {
            pcout << "New atomic positions on the Mesh: " << std::endl;
            for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
              {
                pcout << "Charge Id: " << iCharge << " "
                      << r[iCharge][0] << " "
                      << r[iCharge][0] << " "
                      << r[iCharge][0] << std::endl;
              }
          }
        if (dftParameters::verbosity >= 1)
          {
            pcout << "Displacement of atoms from previous time step " << std::endl;
            for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
              {
                pcout << "Charge Id: " << iCharge << " "
                      << dftPtr->atomLocations[iCharge][2] << " "
                      << dftPtr->atomLocations[iCharge][3] << " "
                      << dftPtr->atomLocations[iCharge][4] << std::endl;
              }         
          
          }          

        MPI_Barrier(MPI_COMM_WORLD);

        update_time = MPI_Wtime() - update_time;
        if (dftParameters::verbosity >= 1)
          pcout << "Time taken for updateAtomPositionsAndMoveMesh: "
                << update_time << std::endl;
        dftPtr->solve(true, false, false, false); 
        forceOnAtoms =dftPtr->forcePtr->getAtomsForces();
        //Call Force
        totalKE = 0.0;
        /* Second half of velocty verlet */
        for(i=0; i < numberGlobalCharges; i++)
            {
                v[3*i+0] = v[3*i+0] - forceOnAtoms[3*i+0]/atomMass[i]*dt_2* haPerBohrToeVPerAng;
                v[3*i+1] = v[3*i+1] - forceOnAtoms[3*i+1]/atomMass[i]*dt_2* haPerBohrToeVPerAng;
                v[3*i+2] = v[3*i+2] - forceOnAtoms[3*i+2]/atomMass[i]*dt_2* haPerBohrToeVPerAng;
                totalKE += 0.5*atomMass[i]*(v[3*i+0]*v[3*i+0]+v[3*i+1]*v[3*i+1] + v[3*i+2]*v[3*i+2]);
            }
        //Save KE
        if (dftParameters::verbosity >= 1)
          {
            pcout << "Velocity of atoms " << std::endl;
            for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
              {
                pcout << "Charge Id: " << iCharge << " "
                      << v[3*iCharge+0] << " "
                      << v[3*iCharge+1] << " "
                      << v[3*iCharge+2] << std::endl;
              }         
          
          }      
      
      
        KE = totalKE;

    }  
    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    void    
    molecularDynamicsClass<FEOrder, FEOrderElectro>::RescaleVelocities(std::vector<double> &v,double &KE,
                                                                        std::vector<double> M, double Temperature)
    {
      pcout << "Rescale Thermostat: Before rescaling temperature= "<<Temperature<<" K" <<  std::endl;
      assert(Temperature==0);// Determine Exit sequence ..
      KE = 0.0;
      for(int iCharge=0; iCharge < numberGlobalCharges; iCharge++)
        {
          v[3*iCharge+0] = v[3*iCharge+0]*sqrt(startingTemperature/Temperature);
          v[3*iCharge+1] = v[3*iCharge+1]*sqrt(startingTemperature/Temperature);
          v[3*iCharge+2] = v[3*iCharge+2]*sqrt(startingTemperature/Temperature);
          KE += 0.5*M[iCharge]*(v[3*iCharge+0]*v[3*iCharge+0]+v[3*iCharge+1]*v[3*iCharge+1] + v[3*iCharge+2]*v[3*iCharge+2]);
        }

    }

    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    void
    molecularDynamicsClass<FEOrder, FEOrderElectro>::NoseHoverChains(std::vector<double> &v, std::vector<double> &v_e, 
                                                                    std::vector<double> &e, std::vector<double> Q, 
                                                                      double KE, double  Temperature)
    {
      double G1, G2, s;
      double L = 3*numberGlobalCharges;
      /* Start Chain 1*/
      G2 = (Q[0]*v_e[0]*v_e[0] - kB*Temperature)/Q[1]; 
      v_e[1]=v_e[1]+G2*timeStep/4;
      v_e[0]=v_e[0]*std::exp(-v_e[1]*timeStep/8);
      G1 = (2*KE-L*kB*Temperature)/Q[0];
      v_e[0] = v_e[0]+G1*timeStep/4;
      v_e[0]=v_e[0]*std::exp(-v_e[1]*timeStep/8);
      e[0] = e[0] + v_e[0]*timeStep/2;
      e[1] = e[1] + v_e[1]*timeStep/2;
      s = std::exp(-v_e[0]*timeStep/2);

      for(int iCharge = 0; iCharge < numberGlobalCharges; iCharge++)
        {
        v[3*iCharge+0] = s*v[3*iCharge+0];
        v[3*iCharge+1] = s*v[3*iCharge+1];
        v[3*iCharge+2] = s*v[3*iCharge+2];
        }
      KE = KE*s*s;
      v_e[0]=v_e[0]*std::exp(-v_e[1]*timeStep/8);
      G1 = (2*KE-L*kB*Temperature)/Q[0];
      v_e[0] = v_e[0]+G1*Temperature/4;
      v_e[0]=v_e[0]*std::exp(-v_e[1]*Temperature/8);
      G2 = (Q[0]*v_e[0]*v_e[0] - kB*Temperature)/Q[1]; 
      v_e[1]=v_e[1]+G2*timeStep/4;
      /* End Chain 1*/
    }

  
    
    
#include "mdClass.inst.cc"
}//nsmespace dftfe
 




