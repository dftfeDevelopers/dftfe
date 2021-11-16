#include <boost/generator_iterator.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>

#include <dft.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include <force.h>
#include <vector>
#include <cmath>
#include <ctime>
#include <molecularDynamicsClass.h>



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
        std::vector<std::vector<double>> atomLocations;
         dftPtr->getAtomLocationsfromdftptr(atomLocations); 
        std::set<unsigned int>   atomTypes;
         dftPtr->getAtomTypesfromdftptr(atomTypes);     
        AssertThrow(atomTypes.size() == atomTypesMasses.size(),
                ExcMessage("DFT-FE Error: check ATOM MASSES FILE"));

        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
        {
          bool isFound = false;
            for (int jatomtype = 0; jatomtype < atomTypesMasses.size(); ++jatomtype)
              {
                const double charge = atomLocations[iCharge][0];
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
      if (restartFlag == 0)
      {  //--------------------Starting Initialization ----------------------------------------------//
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

                    if (std::fabs(atomLocations[iCharge][0] - atomTypesMasses[jatomtype][0]) < 1e-8)
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
         /* pcout << "Temperature computed from Velocities: "
              << TemperatureFromVelocities << std::endl;   
              */    

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
        InternalEnergyVector[0] = dftPtr->GroundStateEnergyvalue;
        EntropicEnergyVector[0] = dftPtr->EntropicEnergyvalue;
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
        pcout << " Temperature from Velocities"
              << TemperatureFromVelocities  << std::endl;
        pcout << " Kinetic Energy in Ha at timeIndex 0 "
              << KineticEnergyVector[0] << std::endl;
        pcout << " Internal Energy in Ha at timeIndex 0 "
              << InternalEnergyVector[0] << std::endl;
        pcout << " Entropic Energy in Ha at timeIndex 0 "
              << EntropicEnergyVector[0] << std::endl;
        pcout << " Total Energy in Ha at timeIndex 0 "
              << TotalEnergyVector[0]  << std::endl;
              

        MPI_Barrier(MPI_COMM_WORLD);
        
        //--------------------Completed Initialization ----------------------------------------------//
      }

      else if(restartFlag == 1)
      {
       std::vector<std::vector<double>> t1,t2;
        int time1, time2;

        dftUtils::readFile(1, t1, "time.chk");
        dftUtils::readFile(1, t2, "time.chk.old");
        time1 = t1[0][0];
        time2 = t2[0][0];        
        if(time1 == time2)
        {
          pcout<<"---Reading Inputs from Restart File---" <<std::endl;
          InitialiseFromRestartFile(velocity, force, KineticEnergyVector , InternalEnergyVector , TotalEnergyVector);
          
        }
        else if(time1 != time2 && dftParameters::UserRestart == 1)
        {
          pcout<<"---Reading Inputs from Restart File Specified by user---" <<std::endl;
          InitialiseFromRestartFile(velocity, force, KineticEnergyVector , InternalEnergyVector , TotalEnergyVector);
          
        }
        else if(time1 != time2 && dftParameters::UserRestart == 0)
        {
          pcout<<"---Kindly specify the restart files in the input parameter file --- "<<std::endl;
          AssertThrow(time1 == time2,
           ExcMessage("DFT-FE Error: Restart Files are corrupted"));           
        }
      }  

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
        else if( thermostatType == "CSVR")
                {
                  mdNVTsvrThermostat(KineticEnergyVector,InternalEnergyVector,EntropicEnergyVector,TotalEnergyVector, displacements, velocity, force,massAtoms ); 
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
        for(TimeIndex=startingTimeStep+1; TimeIndex < startingTimeStep+numberofSteps;TimeIndex++)
        {       
            double step_time;
            MPI_Barrier(MPI_COMM_WORLD);
            step_time = MPI_Wtime();   

            velocityVerlet(velocity, displacements,atomMass,KineticEnergy, force );
            MPI_Barrier(MPI_COMM_WORLD);
   
            KineticEnergyVector[TimeIndex-startingTimeStep]  = KineticEnergy / haToeV;
            InternalEnergyVector[TimeIndex-startingTimeStep] = dftPtr->GroundStateEnergyvalue;
            EntropicEnergyVector[TimeIndex-startingTimeStep] = dftPtr->EntropicEnergyvalue;
            TotalEnergyVector[TimeIndex-startingTimeStep]    = KineticEnergyVector[TimeIndex-startingTimeStep] +
                               InternalEnergyVector[TimeIndex-startingTimeStep] -
                               EntropicEnergyVector[TimeIndex-startingTimeStep]; 
            TemperatureFromVelocities = 2.0/3.0/double(numberGlobalCharges-1)*KineticEnergy/(kB);                              

            //Based on verbose print required MD details...
            MPI_Barrier(MPI_COMM_WORLD);
            step_time = MPI_Wtime() - step_time;
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
            if (dftParameters::verbosity >= 1)
            { 
              pcout << "---------------MD STEP: "<<TimeIndex<<" ------------------ " <<  std::endl;
              pcout << "Time taken for md step: " << step_time << std::endl;
               pcout << " Temperature from velocities: " << TimeIndex << " "
              << TemperatureFromVelocities << std::endl;
              pcout << " Kinetic Energy in Ha at timeIndex " << TimeIndex << " "
              << KineticEnergyVector[TimeIndex-startingTimeStep] << std::endl;
              pcout << " Internal Energy in Ha at timeIndex " << TimeIndex<< " "
              << InternalEnergyVector[TimeIndex-startingTimeStep]
              << std::endl;
              pcout << " Entropic Energy in Ha at timeIndex " << TimeIndex << " "
              << EntropicEnergyVector[TimeIndex-startingTimeStep]
              << std::endl;
              pcout << " Total Energy in Ha at timeIndex " << TimeIndex << " "
              << TotalEnergyVector[TimeIndex-startingTimeStep] << std::endl;
              writeRestartFile(velocity,force,KineticEnergyVector,InternalEnergyVector,TotalEnergyVector,TimeIndex);
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
      
      pcout << "---------------mdNVTrescaleThermostat() called ------------------ " <<  std::endl;
        double KineticEnergy;
        double TemperatureFromVelocities;
        for(TimeIndex=startingTimeStep+1; TimeIndex< startingTimeStep+numberofSteps;TimeIndex++)
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
            KineticEnergyVector[TimeIndex-startingTimeStep]  = KineticEnergy / haToeV;
            InternalEnergyVector[TimeIndex-startingTimeStep] = dftPtr->GroundStateEnergyvalue;
            EntropicEnergyVector[TimeIndex-startingTimeStep] = dftPtr->EntropicEnergyvalue;
            TotalEnergyVector[TimeIndex-startingTimeStep]    = KineticEnergyVector[TimeIndex-startingTimeStep] +
                               InternalEnergyVector[TimeIndex-startingTimeStep] -
                               EntropicEnergyVector[TimeIndex-startingTimeStep]; 
            TemperatureFromVelocities = 2.0/3.0/double(numberGlobalCharges-1)*KineticEnergy/(kB);                              

            //Based on verbose print required MD details...
            MPI_Barrier(MPI_COMM_WORLD);
            step_time = MPI_Wtime() - step_time;
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
            if (dftParameters::verbosity >= 1)
            { 
              pcout << "---------------MD STEP: "<<TimeIndex<<" ------------------ " <<  std::endl;
              pcout << "Time taken for md step: " << step_time << std::endl;
               pcout << " Temperature from velocities: " << TimeIndex << " "
              << TemperatureFromVelocities << std::endl;
              pcout << " Kinetic Energy in Ha at timeIndex " << TimeIndex << " "
              << KineticEnergyVector[TimeIndex-startingTimeStep] << std::endl;
              pcout << " Internal Energy in Ha at timeIndex " << TimeIndex<< " "
              << InternalEnergyVector[TimeIndex-startingTimeStep]
              << std::endl;
              pcout << " Entropic Energy in Ha at timeIndex " << TimeIndex << " "
              << EntropicEnergyVector[TimeIndex-startingTimeStep]
              << std::endl;
              pcout << " Total Energy in Ha at timeIndex " << TimeIndex << " "
              << TotalEnergyVector[TimeIndex-startingTimeStep] << std::endl;
              writeRestartFile(velocity,force,KineticEnergyVector,InternalEnergyVector,TotalEnergyVector,TimeIndex);
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


      pcout << "--------------mdNVTnosehoverchainsThermostat() called ------------------ " <<  std::endl;  
      
        double KineticEnergy;
        double TemperatureFromVelocities;
        double nhctimeconstant;
        TemperatureFromVelocities = 2.0/3.0/double(numberGlobalCharges-1)*KineticEnergyVector[0]/(kB); 
        std::vector<double> ThermostatMass(2,0.0);
        std::vector<double> Thermostatvelocity(2,0.0);
        std::vector<double> Thermostatposition(2,0.0);  
        std::vector<double> NoseHoverExtendedLagrangianvector(numberofSteps,0.0);    
        nhctimeconstant = thermostatTimeConstant*timeStep;  
        if(restartFlag == 0)
        {
          ThermostatMass[0] = 3*(numberGlobalCharges-1)*kB*startingTemperature*(nhctimeconstant*nhctimeconstant);
          ThermostatMass[1] = kB*startingTemperature*(nhctimeconstant*nhctimeconstant);
          pcout << "Time Step " <<timeStep<<" Q2: "<<ThermostatMass[1]<<"Time Constant"<<nhctimeconstant<<"no. of atoms"<< numberGlobalCharges<<"Starting Temp: "<<
                startingTemperature<<"---"<<3*(numberGlobalCharges-1)*kB*startingTemperature*(nhctimeconstant*nhctimeconstant)<<std::endl;
        }
        else
        {
          InitialiseFromRestartNHCFile(Thermostatvelocity,Thermostatposition, ThermostatMass );
        }
        
         for(TimeIndex=startingTimeStep+1; TimeIndex < startingTimeStep+numberofSteps;TimeIndex++)
        {       
            double step_time;
            MPI_Barrier(MPI_COMM_WORLD);
            step_time = MPI_Wtime();   
            NoseHoverChains(velocity, Thermostatvelocity,Thermostatposition, ThermostatMass,KineticEnergyVector[TimeIndex-1-startingTimeStep]*haToeV,startingTemperature);
            velocityVerlet(velocity, displacements,atomMass,KineticEnergy, force);
          //  TemperatureFromVelocities = 2.0/3.0/double(numberGlobalCharges-1)*KineticEnergy/(kB); 
            NoseHoverChains(velocity, Thermostatvelocity,Thermostatposition, ThermostatMass,KineticEnergy,startingTemperature);          
            KineticEnergy = 0.0;
            for(int iCharge=0; iCharge < numberGlobalCharges; iCharge++)
              {
                KineticEnergy +=0.5*atomMass[iCharge]*(velocity[3*iCharge+0]*velocity[3*iCharge+0] + 
                                                      velocity[3*iCharge+1]*velocity[3*iCharge+1] +
                                                      velocity[3*iCharge+2]*velocity[3*iCharge+2]);
              }


            MPI_Barrier(MPI_COMM_WORLD);   
            KineticEnergyVector[TimeIndex-startingTimeStep]  = KineticEnergy / haToeV;
            InternalEnergyVector[TimeIndex-startingTimeStep] = dftPtr->GroundStateEnergyvalue;
            EntropicEnergyVector[TimeIndex-startingTimeStep] = dftPtr->EntropicEnergyvalue;
            TotalEnergyVector[TimeIndex-startingTimeStep]    = KineticEnergyVector[TimeIndex-startingTimeStep] +
                               InternalEnergyVector[TimeIndex-startingTimeStep] -
                               EntropicEnergyVector[TimeIndex-startingTimeStep]; 
            TemperatureFromVelocities = 2.0/3.0/double(numberGlobalCharges-1)*KineticEnergy/(kB);  
            NoseHoverExtendedLagrangianvector[TimeIndex-startingTimeStep] = NoseHoverExtendedLagrangian(Thermostatvelocity,Thermostatposition,ThermostatMass,
                                                KineticEnergyVector[TimeIndex-startingTimeStep],TotalEnergyVector[TimeIndex-startingTimeStep],TemperatureFromVelocities);                            

            //Based on verbose print required MD details...
            MPI_Barrier(MPI_COMM_WORLD);
            step_time = MPI_Wtime() - step_time;
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
            if (dftParameters::verbosity >= 1)
            { 
              pcout << "---------------MD STEP: "<<TimeIndex<<" ------------------ " <<  std::endl;
              pcout << "Time taken for md step: " << step_time << std::endl;
               pcout << " Temperature from velocities: " << TimeIndex << " "
              << TemperatureFromVelocities << std::endl;
              pcout << " Kinetic Energy in Ha at timeIndex " << TimeIndex << " "
              << KineticEnergyVector[TimeIndex-startingTimeStep] << std::endl;
              pcout << " Internal Energy in Ha at timeIndex " << TimeIndex<< " "
              << InternalEnergyVector[TimeIndex-startingTimeStep]
              << std::endl;
              pcout << " Entropic Energy in Ha at timeIndex " << TimeIndex << " "
              << EntropicEnergyVector[TimeIndex-startingTimeStep]
              << std::endl;
              pcout << " Total Energy in Ha at timeIndex " << TimeIndex << " "
              << TotalEnergyVector[TimeIndex-startingTimeStep] << std::endl;
              pcout << "Nose Hover Extended Lagrangian  in Ha at timeIndex " << TimeIndex << " "
              << NoseHoverExtendedLagrangianvector[TimeIndex-startingTimeStep] << std::endl;              
              writeRestartFile(velocity,force,KineticEnergyVector,InternalEnergyVector,TotalEnergyVector,TimeIndex);
              writeRestartNHCfile(Thermostatvelocity,Thermostatposition, ThermostatMass );
            }


        }

    }

    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    void
    molecularDynamicsClass<FEOrder, FEOrderElectro>::mdNVTsvrThermostat(std::vector<double> &KineticEnergyVector ,
                                                           std::vector<double> &InternalEnergyVector ,
                                                            std::vector<double> &EntropicEnergyVector , 
                                                            std::vector<double> &TotalEnergyVector ,
                                                            std::vector<dealii::Tensor<1, 3, double>> &displacements ,
                                                            std::vector<double> &velocity ,
                                                            std::vector<double> &force, 
                                                            std::vector<double> atomMass  )
    {
      
      pcout << "---------------mdNVTsvrThermostat() called ------------------ " <<  std::endl;
        double KineticEnergy;
        double TemperatureFromVelocities;
        double KEref = 3.0/2.0*double(numberGlobalCharges-1)*kB*startingTemperature;
        
        for(TimeIndex=startingTimeStep+1; TimeIndex< startingTimeStep+numberofSteps;TimeIndex++)
        {       
            double step_time;

            MPI_Barrier(MPI_COMM_WORLD);
            step_time = MPI_Wtime();   

            velocityVerlet(velocity, displacements,atomMass,KineticEnergy, force);
            //MPI_Barrier(MPI_COMM_WORLD);

            svr(velocity,KineticEnergy,KEref);
            TemperatureFromVelocities = 2.0/3.0/double(numberGlobalCharges-1)*KineticEnergy/(kB);

            MPI_Barrier(MPI_COMM_WORLD);   
            KineticEnergyVector[TimeIndex-startingTimeStep]  = KineticEnergy / haToeV;
            InternalEnergyVector[TimeIndex-startingTimeStep] = dftPtr->GroundStateEnergyvalue;
            EntropicEnergyVector[TimeIndex-startingTimeStep] = dftPtr->EntropicEnergyvalue;
            TotalEnergyVector[TimeIndex-startingTimeStep]    = KineticEnergyVector[TimeIndex-startingTimeStep] +
                               InternalEnergyVector[TimeIndex-startingTimeStep] -
                               EntropicEnergyVector[TimeIndex-startingTimeStep]; 
            TemperatureFromVelocities = 2.0/3.0/double(numberGlobalCharges-1)*KineticEnergy/(kB);                              

            //Based on verbose print required MD details...
            MPI_Barrier(MPI_COMM_WORLD);
            step_time = MPI_Wtime() - step_time;
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
            if (dftParameters::verbosity >= 1)
            { 
              pcout << "---------------MD STEP: "<<TimeIndex<<" ------------------ " <<  std::endl;
              pcout << "Time taken for md step: " << step_time << std::endl;
               pcout << " Temperature from velocities: " << TimeIndex << " "
              << TemperatureFromVelocities << std::endl;
              pcout << " Kinetic Energy in Ha at timeIndex " << TimeIndex << " "
              << KineticEnergyVector[TimeIndex-startingTimeStep] << std::endl;
              pcout << " Internal Energy in Ha at timeIndex " << TimeIndex<< " "
              << InternalEnergyVector[TimeIndex-startingTimeStep]
              << std::endl;
              pcout << " Entropic Energy in Ha at timeIndex " << TimeIndex << " "
              << EntropicEnergyVector[TimeIndex-startingTimeStep]
              << std::endl;
              pcout << " Total Energy in Ha at timeIndex " << TimeIndex << " "
              << TotalEnergyVector[TimeIndex-startingTimeStep] << std::endl;
              writeRestartFile(velocity,force,KineticEnergyVector,InternalEnergyVector,TotalEnergyVector,TimeIndex);
            }


        }

    }



    
    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    void    
    molecularDynamicsClass<FEOrder, FEOrderElectro>::velocityVerlet(std::vector<double> &v , std::vector<dealii::Tensor<1, 3, double>> &r, 
                                                                    std::vector<double> atomMass, double &KE, std::vector<double> &forceOnAtoms
                                                                     )
    {    
        
        int i;
        double totalKE;
        KE = 0.0;
        double dt = timeStep;
        double dt_2 = dt/2;
              
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
        /*dftPtr->updateAtomPositionsAndMoveMesh(
          r,
          dftParameters::maxJacobianRatioFactorForMD,
          (TimeIndex == startingTimeStep + 1 && restartFlag == 1) ? true :
                                                                    false);*/

        dftPtr->updateAtomPositionsAndMoveMesh(
          r,
          dftParameters::maxJacobianRatioFactorForMD,false);

        if (dftParameters::verbosity >= 1)
          { std::vector<std::vector<double>> atomLocations;
             dftPtr->getAtomLocationsfromdftptr(atomLocations);  
            pcout << "Updated Positions of atoms time step " << std::endl;
            for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
              {
                pcout << "Charge Id: " << iCharge << " "
                      << atomLocations[iCharge][2] << " "
                      << atomLocations[iCharge][3] << " "
                      << atomLocations[iCharge][4] << std::endl;
              }         
          
          }          
        
        MPI_Barrier(MPI_COMM_WORLD);

        update_time = MPI_Wtime() - update_time;
        writeTotalDisplacementFile(r);
        if (dftParameters::verbosity >= 1)
          pcout << "Time taken for updateAtomPositionsAndMoveMesh: "
                << update_time << std::endl;
        dftPtr->solve(true, false, false, false); 
         dftPtr->getForceonAtomsfromdftptr(forceOnAtoms);
         
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
      
      
      
        KE = totalKE;

    }  
    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    void    
    molecularDynamicsClass<FEOrder, FEOrderElectro>::RescaleVelocities(std::vector<double> &v,double &KE,
                                                                        std::vector<double> M, double Temperature)
    {
      pcout << "Rescale Thermostat: Before rescaling temperature= "<<Temperature<<" K" <<  std::endl;
      AssertThrow(std::fabs(Temperature - 0.0) > 0.00001,
           ExcMessage("DFT-FE Error: Temperature reached O K"));      // Determine Exit sequence ..
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
      double L = 3*(numberGlobalCharges-1);
      /* Start Chain 1*/
      G2 = (Q[0]*v_e[0]*v_e[0] - kB*Temperature)/Q[1]; 
      //pcout << "v_e[0]:"<<v_e[0]<<std::endl;
      v_e[1]=v_e[1]+G2*timeStep/4;
      //pcout << "v_e[1]:"<<v_e[1]<<std::endl;
      v_e[0]=v_e[0]*std::exp(-v_e[1]*timeStep/8);
      //pcout << "v_e[0]*std::exp(-v_e[1]*timeStep/8):"<<v_e[0]<<std::endl;
      G1 = (2*KE-L*kB*Temperature)/Q[0];
      v_e[0] = v_e[0]+G1*timeStep/4;
      //pcout << "v_e[0]+G1*timeStep/4:"<<v_e[0]<<std::endl;
      v_e[0]= v_e[0]*std::exp(-v_e[1]*timeStep/8);
      //pcout << "v_e[0]*std::exp(-v_e[1]*timeStep/8):"<<v_e[0]<<std::endl;
      e[0] = e[0] + v_e[0]*timeStep/2;
      e[1] = e[1] + v_e[1]*timeStep/2;
      s = std::exp(-v_e[0]*timeStep/2);
      //pcout << "G2"<<G2<<" v_e1"<<v_e[1]<<"Temp"<<Temperature<<"Q[1] "<<Q[1]<<"v_e[0] "<<v_e[0]<<"G1 "<<G1<< "Exponent: "<<std::exp(1)<< std::endl;
      for(int iCharge = 0; iCharge < numberGlobalCharges; iCharge++)
        {
        v[3*iCharge+0] = s*v[3*iCharge+0];
        v[3*iCharge+1] = s*v[3*iCharge+1];
        v[3*iCharge+2] = s*v[3*iCharge+2];
        }
      KE = KE*s*s;
      v_e[0]=v_e[0]*std::exp(-v_e[1]*timeStep/8);
      G1 = (2*KE-L*kB*Temperature)/Q[0];
      v_e[0] = v_e[0]+G1*timeStep/4;
      v_e[0]=v_e[0]*std::exp(-v_e[1]*timeStep/8);
      G2 = (Q[0]*v_e[0]*v_e[0] - kB*Temperature)/Q[1]; 
      v_e[1]=v_e[1]+G2*timeStep/4;
      /* End Chain 1*/
      
    }

    
    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    void    
    molecularDynamicsClass<FEOrder, FEOrderElectro>::svr(std::vector<double> &v,double &KE, double KEref)
    {
        double alphasq;
        unsigned int Nf = 3*(numberGlobalCharges-1);
        double R1, Rsum;
        R1 = 0.0;
        Rsum = 0.0;
        if (this_mpi_process == 0)
        {
          std::time_t now = std::time(0);
          boost::random::mt19937 gen{static_cast<std::uint32_t>(now)};
          boost::normal_distribution<> gaussianDist(0.0, 1.0);
          boost::variate_generator<boost::mt19937 & , boost::normal_distribution<>>
                  generator(gen, gaussianDist);

          R1 = generator();   
          /*    
          if((Nf-1)%2 == 0)
            {
              boost::gamma_distribution<> my_gamma((Nf-1)/2,1);
              boost::variate_generator<boost::mt19937 &, boost::gamma_distribution<>>
                  generator_gamma(gen, my_gamma);              
              Rsum = generator_gamma();
            }
          else
            {
              Rsum = generator();
              Rsum = Rsum*Rsum;
              boost::gamma_distribution<> my_gamma((Nf-2)/2,1);
              boost::variate_generator<boost::mt19937 &, boost::gamma_distribution<>>
                  generator_gamma(gen, my_gamma);               
              Rsum += generator_gamma();
            }  
             */
           double temp;
           for (int dof = 1; dof < Nf; dof++)
           {
             temp = generator();
             Rsum = Rsum + temp*temp;
           } 
           Rsum = R1*R1 + Rsum;
           
      //Transfer data to all mpi procs    
        }
        R1 = dealii::Utilities::MPI::sum(R1, mpi_communicator);
        Rsum = dealii::Utilities::MPI::sum(Rsum, mpi_communicator);
        alphasq = 0.0;
        alphasq = alphasq+ std::exp(-1/double(thermostatTimeConstant));
        alphasq = alphasq+ (KEref/Nf/KE)*(1-std::exp(-1/double(thermostatTimeConstant)))*(R1*R1 + Rsum);
        alphasq = alphasq + 2*std::exp(-1/2/double(thermostatTimeConstant))*std::sqrt(KEref/Nf/KE*(1-std::exp(-1/double(thermostatTimeConstant))))*R1;
        pcout<<"*** R1: "<<R1<<" Rsum : "<<Rsum<<" alphasq "<<alphasq<<"exp ()"<<std::exp(-1/double(thermostatTimeConstant))<<" timeconstant "<<thermostatTimeConstant<< std::endl;    
        KE      = alphasq*KE;
        double alpha = std::sqrt(alphasq);
        for(int iCharge=0; iCharge<numberGlobalCharges; iCharge++)
        {
          v[3*iCharge+0] = alpha*v[3*iCharge+0];
          v[3*iCharge+1] = alpha*v[3*iCharge+1];
          v[3*iCharge+2] = alpha*v[3*iCharge+2];
        }
    }    
    
    
    
    
    
    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    void
    molecularDynamicsClass<FEOrder, FEOrderElectro>::writeRestartFile(std::vector<double> velocity , std::vector<double> force , std::vector<double> KineticEnergyVector ,
                                                                      std::vector<double> InternalEnergyVector , std::vector<double> TotalEnergyVector, int time )

   {
     //Writes the restart files for velocities and positions
    std::vector<std::vector<double>> fileForceData(numberGlobalCharges,
                                                  std::vector<double>(3,0.0));

    std::vector<std::vector<double>> fileVelocityData(numberGlobalCharges,
                                                  std::vector<double>(3,0.0)); 
    std::vector<std::vector<double>> timeIndexData(1, std::vector<double>(1, 0));
    std::vector<std::vector<double>> KEData(1, std::vector<double>(1, 0.0));
    std::vector<std::vector<double>> IEData(1, std::vector<double>(1, 0.0));
    std::vector<std::vector<double>> TEData(1, std::vector<double>(1, 0.0));

    
    timeIndexData[0][0] = double(time);  
    dftUtils::writeDataIntoFile(timeIndexData, "time.chk");  
    KEData[0][0]    = KineticEnergyVector[time-startingTimeStep];
    IEData[0][0] = InternalEnergyVector[time - startingTimeStep];
    TEData[0][0] = TotalEnergyVector[time - startingTimeStep];  
    dftUtils::writeDataIntoFile(KEData, "KineticEnergy.chk");
    dftUtils::writeDataIntoFile(IEData, "InternalEnergy.chk");
    dftUtils::writeDataIntoFile(TEData, "TotalEnergy.chk");                                        

    for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
      {
        fileForceData[iCharge][0] = force[3 * iCharge + 0];
        fileForceData[iCharge][1] = force[3 * iCharge + 1];
        fileForceData[iCharge][2] = force[3 * iCharge + 2];
      }  
    for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
      {
        fileVelocityData[iCharge][0] = velocity[3 * iCharge + 0];
        fileVelocityData[iCharge][1] = velocity[3 * iCharge + 1];
        fileVelocityData[iCharge][2] = velocity[3 * iCharge + 2];
      }      

    dftPtr->writeDomainAndAtomCoordinates(); 
    if(time > 1)
    pcout << "#RESTART NOTE: Positions:-" << " Positions of TimeStep:"<<time<<" present in file atomsFracCoordCurrent.chk"<< std::endl
          <<" Positions of TimeStep: "<<time-1<<"present in file atomsFracCoordCurrent.chk.old #"<< std::endl;
    dftUtils::writeDataIntoFile(fileVelocityData, "velocity.chk");
    if(time > 1)
    pcout << "#RESTART NOTE: Velocity:-" << " Velocity of TimeStep:"<<time<<" present in file velocity.chk"<< std::endl
          <<" Velocity of TimeStep: "<<time-1<<"present in file velocity.chk.old #"<< std::endl;    
    dftUtils::writeDataIntoFile(fileForceData, "force.chk"); 
    if(time > 1)
    pcout << "#RESTART NOTE: Force:-" << " Force of TimeStep:"<<time<<" present in file force.chk"<< std::endl
          <<" Velocity of TimeStep: "<<time-1<<"present in file force.chk.old #"<< std::endl;     
    MPI_Barrier(MPI_COMM_WORLD);
    dftUtils::writeDataIntoFile(timeIndexData, "time.chk"); //old time == new time then restart files were successfully saved
    pcout << "#RESTART NOTE: restart files for TimeStep: "<<time<<" successfully created #"<< std::endl;     


   }                                                                    
  
    
    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    void
    molecularDynamicsClass<FEOrder, FEOrderElectro>::InitialiseFromRestartFile(std::vector<double> &velocity ,
                                                            std::vector<double> &force, std::vector<double> &KE, std::vector<double> &IE, std::vector<double> &TE  )
    {
        //Initialise Position
      if (dftParameters::verbosity >= 1)
        { std::vector<std::vector<double>> atomLocations;
            dftPtr->getAtomLocationsfromdftptr(atomLocations);  
           pcout << "Atom Locations from Restart " << std::endl;
           for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
             {
               pcout << "Charge Id: " << iCharge << " "
                     << atomLocations[iCharge][2] << " "
                     << atomLocations[iCharge][3] << " "
                     << atomLocations[iCharge][4] << std::endl;
             }         
          
         }
        std::vector<std::vector<double>> t1, KE0, IE0, TE0;
        std::string s1 = dftParameters::PositionRestartFile;
        AssertThrow(s1.compare("atomsFracCoordCurrent.chk")==0 ,
           ExcMessage("DFT-FE Error: Position Restart file not atomsFracCoordCurrent.chk"));
        dftUtils::readFile(1, t1, "time.chk");        
        startingTimeStep = t1[0][0];
        std::string fileName1 = dftParameters::VelocityRestartFile; 
        std::vector<std::vector<double>> fileVelData;
        dftUtils::readFile(3, fileVelData, fileName1);
        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            velocity[3 * iCharge + 0] = fileVelData[iCharge][0];
            velocity[3 * iCharge + 1] = fileVelData[iCharge][1];
            velocity[3 * iCharge + 2] = fileVelData[iCharge][2];
          }   
        std::string fileName2 = dftParameters::ForceRestartFile; 
        std::vector<std::vector<double>> fileForceData;
        dftUtils::readFile(3, fileForceData, fileName2);
        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            force[3 * iCharge + 0] = fileForceData[iCharge][0];
            force[3 * iCharge + 1] = fileForceData[iCharge][1];
            force[3 * iCharge + 2] = fileForceData[iCharge][2];
          }                

      
      dftUtils::readFile(1, KE0, "KineticEnergy.chk");
      dftUtils::readFile(1, IE0, "InternalEnergy.chk");
      dftUtils::readFile(1, TE0, "TotalEnergy.chk");
      KE[0] = KE0[0][0];
      IE[0] = IE0[0][0];
      TE[0] = TE0[0][0];
      dftPtr->solve(true, false, false, false); 
      dftPtr->getForceonAtomsfromdftptr(force);


    }                                                        

    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    void
    molecularDynamicsClass<FEOrder, FEOrderElectro>:: InitialiseFromRestartNHCFile(std::vector<double> &v_e ,
                                                            std::vector<double> &e, std::vector<double> &Q )

  {
    std::vector<std::vector<double>> NHCData;
    dftUtils::readFile(3, NHCData, "NHCThermostat.chk");
    Q[0] = NHCData[0][0];
    Q[1] = NHCData[1][0];
    e[0] = NHCData[0][1];
    e[1] = NHCData[1][1];
    v_e[0] = NHCData[0][2];
    v_e[1] = NHCData[1][2];
    pcout<<"Nose Hover Chains Thermostat configuration read from Restart file "<<std::endl;

  }                                                          


   template <unsigned int FEOrder, unsigned int FEOrderElectro>
   void
    molecularDynamicsClass<FEOrder, FEOrderElectro>:: writeRestartNHCfile(std::vector<double> v_e ,
                                                            std::vector<double> e, std::vector<double> Q )

   {
    std::vector<std::vector<double>> fileNHCData(2,std::vector<double>(3,0.0)); 
    fileNHCData[0][0] = Q[0] ;
    fileNHCData[0][1] = e[0] ;
    fileNHCData[0][2] = v_e[0] ;
    fileNHCData[1][0] = Q[1] ;
    fileNHCData[1][1] =  e[1];
    fileNHCData[1][2] =  v_e[1];    
    dftUtils::writeDataIntoFile(fileNHCData, "NHCThermostat.chk");        


   }                                                         
    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    void    
    molecularDynamicsClass<FEOrder, FEOrderElectro>::writeTotalDisplacementFile( std::vector<dealii::Tensor<1, 3, double>> r)
    {
      std::vector<std::vector<double>>fileDisplacementData(numberGlobalCharges,std::vector<double>(3,0.0)); 
      dftUtils::readFile(3, fileDisplacementData, "Displacement.chk");
      for(int iCharge = 0; iCharge <numberGlobalCharges; iCharge++)
        {
            fileDisplacementData[iCharge][0] = fileDisplacementData[iCharge][0]+ r[iCharge][0];
            fileDisplacementData[iCharge][1] = fileDisplacementData[iCharge][1]+ r[iCharge][1];
            fileDisplacementData[iCharge][2] = fileDisplacementData[iCharge][2]+ r[iCharge][2];
        } 
       dftUtils::writeDataIntoFile(fileDisplacementData, "Displacement.chk");  

    }
    template <unsigned int FEOrder, unsigned int FEOrderElectro>
    double
    molecularDynamicsClass<FEOrder, FEOrderElectro>:: NoseHoverExtendedLagrangian(std::vector<double> thermovelocity ,
     std::vector<double> thermoposition   , std::vector<double> thermomass , double PE, double KE, double T)
  {
    double Hnose = 0.0;
    Hnose = (0.5*thermomass[0]*thermovelocity[0]*thermovelocity[0]+ 0.5*thermomass[1]*thermovelocity[1]*thermovelocity[1]
           + 3*numberGlobalCharges*T*kB*thermoposition[0] +kB*T*thermoposition[1] )/haToeV + KE + PE;
    return(Hnose);
  
  }







#include "mdClass.inst.cc"
}//nsmespace dftfe
 




