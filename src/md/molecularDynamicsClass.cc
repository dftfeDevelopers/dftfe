#include <boost/generator_iterator.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <headers.h>
#include <dft.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include <force.h>
#include <vector>
#include <cmath>
#include <ctime>
#include <molecularDynamicsClass.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <sys/stat.h>
namespace dftfe
{
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  molecularDynamicsClass<FEOrder, FEOrderElectro>::molecularDynamicsClass(
    dftClass<FEOrder, FEOrderElectro> *_dftPtr,
    const MPI_Comm &                   mpi_comm_replica,
    const MPI_Comm &                   interpoolcomm,
    const MPI_Comm &                   interBandGroupComm)
    : dftPtr(_dftPtr)
    , d_mpi_communicator(mpi_comm_replica)
    , d_interpoolcomm(interpoolcomm)
    , d_interBandGroupComm(interBandGroupComm)
    , d_this_mpi_process(Utilities::MPI::this_mpi_process(mpi_comm_replica))
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(d_mpi_communicator) == 0 &&
             Utilities::MPI::this_mpi_process(d_interpoolcomm) == 0 &&
             Utilities::MPI::this_mpi_process(d_interBandGroupComm) == 0))
  {
    MPI_Barrier(d_mpi_communicator);
    MPI_Barrier(d_interBandGroupComm);
    MPI_Barrier(d_interpoolcomm);
    d_MDstartWallTime  = MPI_Wtime();
    d_TimeIndex        = 0;
    d_startingTimeStep = checkRestart();
    d_TimeStep =
      dftParameters::timeStepBOMD *
      0.09822694541304546435; // Conversion factor from femteseconds:
                              // 0.09822694541304546435 based on NIST constants
    d_numberofSteps = dftParameters::numberStepsBOMD;

    d_startingTemperature    = dftParameters::startingTempBOMD;
    d_ThermostatTimeConstant = dftParameters::thermostatTimeConstantBOMD;
    d_ThermostatType         = dftParameters::tempControllerTypeBOMD;
    d_numberGlobalCharges    = dftParameters::natoms;

    d_MaxWallTime = dftParameters::MaxWallTime;
    pcout
      << "----------------------Starting Initialization of BOMD-------------------------"
      << std::endl;
    pcout << "Starting Temperature from Input " << d_startingTemperature
          << std::endl;
    std::vector<std::vector<double>> temp_domainBoundingVectors;
    dftUtils::readFile(3,
                       temp_domainBoundingVectors,
                       dftParameters::domainBoundingVectorsFile);

    for (int i = 0; i < 3; i++)
      {
        double temp =
          temp_domainBoundingVectors[i][0] * temp_domainBoundingVectors[i][0] +
          temp_domainBoundingVectors[i][1] * temp_domainBoundingVectors[i][1] +
          temp_domainBoundingVectors[i][2] * temp_domainBoundingVectors[i][2];
        d_domainLength.push_back(pow(temp, 0.5));
      }
    if (dftParameters::verbosity > 1)
      {
        pcout << "--$ Domain Length$ --" << std::endl;
        pcout << "Lx:= " << d_domainLength[0] << " Ly:=" << d_domainLength[1]
              << " Lz:=" << d_domainLength[2] << std::endl;
      }
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  molecularDynamicsClass<FEOrder, FEOrderElectro>::runMD()
  {
    std::vector<double> massAtoms(d_numberGlobalCharges);
    //
    // read atomic masses in amu
    //
    std::vector<std::vector<double>> atomTypesMasses;
    dftUtils::readFile(2, atomTypesMasses, dftParameters::atomicMassesFile);
    std::vector<std::vector<double>> atomLocations;
    atomLocations = dftPtr->getAtomLocations();
    std::set<unsigned int> atomTypes;
    atomTypes = dftPtr->getAtomTypes();
    AssertThrow(atomTypes.size() == atomTypesMasses.size(),
                ExcMessage("DFT-FE Error: check ATOM MASSES FILE"));

    for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
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
      d_numberGlobalCharges);
    std::vector<double> velocity(3 * d_numberGlobalCharges, 0.0);
    std::vector<double> force(3 * d_numberGlobalCharges, 0.0);
    std::vector<double> InternalEnergyVector(d_numberofSteps, 0.0);
    std::vector<double> EntropicEnergyVector(d_numberofSteps, 0.0);
    std::vector<double> KineticEnergyVector(d_numberofSteps, 0.0);
    std::vector<double> TotalEnergyVector(d_numberofSteps, 0.0);
    double              totMass = 0.0;
    double              velocityDistribution;
    /*  d_restartFlag = ((dftParameters::chkType == 1 || dftParameters::chkType
     == 3) && dftParameters::restartMdFromChk) ?        1 :        0; // 1;
     //0;//1;*/
    d_restartFlag = dftParameters::restartMdFromChk ? 1 : 0;
    pcout << "RestartFlag: " << d_restartFlag << std::endl;
    if (d_restartFlag == 0)
      {
        std::string tempfolder = "mdRestart";
        mkdir(tempfolder.c_str(), ACCESSPERMS);
        double KineticEnergy = 0.0, TemperatureFromVelocities = 0.0,
               GroundStateEnergyvalue = 0.0, EntropicEnergyvalue = 0.0;

        dftUtils::readFile(5,
                           d_atomFractionalunwrapped,
                           dftParameters::coordinatesFile);
        std::vector<std::vector<double>> fileDisplacementData;
        std::vector<double>              initDisp(0.0, 3);
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
          {
            fileDisplacementData.push_back(initDisp);
          }


        dftUtils::writeDataIntoFile(fileDisplacementData, "Displacement.chk");
        //--------------------Starting Initialization
        //----------------------------------------------//

        double Px = 0.0, Py = 0.0, Pz = 0.0;
        // Initialise Velocity
        if (Utilities::MPI::this_mpi_process(d_mpi_communicator) == 0 &&
            Utilities::MPI::this_mpi_process(d_interpoolcomm) == 0 &&
            Utilities::MPI::this_mpi_process(d_interBandGroupComm) == 0)
          {
            for (int jatomtype = 0; jatomtype < atomTypesMasses.size();
                 ++jatomtype)
              {
                double Mass          = atomTypesMasses[jatomtype][1];
                velocityDistribution = sqrt(kB * d_startingTemperature / Mass);

                pcout << "Initialising Velocities of species no: " << jatomtype
                      << " mass in amu: " << Mass << "Velocity Deviation"
                      << velocityDistribution << std::endl;
                boost::mt19937               rng{0};
                boost::normal_distribution<> gaussianDist(0.0,
                                                          velocityDistribution);
                boost::variate_generator<boost::mt19937 &,
                                         boost::normal_distribution<>>
                  generator(rng, gaussianDist);

                for (int iCharge = 0; iCharge < d_numberGlobalCharges;
                     ++iCharge)
                  {
                    if (std::fabs(atomLocations[iCharge][0] -
                                  atomTypesMasses[jatomtype][0]) < 1e-8)
                      {
                        velocity[3 * iCharge + 0] = generator();
                        velocity[3 * iCharge + 1] = generator();
                        velocity[3 * iCharge + 2] = generator();
                      }
                  }
              }
          }

        for (unsigned int i = 0; i < d_numberGlobalCharges * 3; ++i)
          {
            velocity[i] =
              dealii::Utilities::MPI::sum(velocity[i], d_mpi_communicator);
          }



        // compute KEinetic Energy and COM vecloity
        totMass = 0.0;
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
          {
            totMass += massAtoms[iCharge];
            Px += massAtoms[iCharge] * velocity[3 * iCharge + 0];
            Py += massAtoms[iCharge] * velocity[3 * iCharge + 1];
            Pz += massAtoms[iCharge] * velocity[3 * iCharge + 2];
          }
        // Correcting for COM velocity to be 0

        for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
          {
            velocity[3 * iCharge + 0] =
              (massAtoms[iCharge] * velocity[3 * iCharge + 0] -
               Px / d_numberGlobalCharges) /
              massAtoms[iCharge];
            velocity[3 * iCharge + 1] =
              (massAtoms[iCharge] * velocity[3 * iCharge + 1] -
               Py / d_numberGlobalCharges) /
              massAtoms[iCharge];
            velocity[3 * iCharge + 2] =
              (massAtoms[iCharge] * velocity[3 * iCharge + 2] -
               Pz / d_numberGlobalCharges) /
              massAtoms[iCharge];


            KineticEnergy +=
              0.5 * massAtoms[iCharge] *
              (velocity[3 * iCharge + 0] * velocity[3 * iCharge + 0] +
               velocity[3 * iCharge + 1] * velocity[3 * iCharge + 1] +
               velocity[3 * iCharge + 2] * velocity[3 * iCharge + 2]);
          }
        TemperatureFromVelocities =
          2.0 / 3.0 / (d_numberGlobalCharges - 1) * KineticEnergy / (kB);
        /* pcout << "Temperature computed from Velocities: "
             << TemperatureFromVelocities << std::endl;
             */

        // Correcting velocity to match init Temperature
        double gamma = sqrt(d_startingTemperature / TemperatureFromVelocities);

        for (int i = 0; i < 3 * d_numberGlobalCharges; ++i)
          {
            velocity[i] = gamma * velocity[i];
          }


        KineticEnergy = 0.0;
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
          {
            KineticEnergy +=
              0.5 * massAtoms[iCharge] *
              (velocity[3 * iCharge + 0] * velocity[3 * iCharge + 0] +
               velocity[3 * iCharge + 1] * velocity[3 * iCharge + 1] +
               velocity[3 * iCharge + 2] * velocity[3 * iCharge + 2]);
          }
        TemperatureFromVelocities =
          2.0 / 3.0 / double(d_numberGlobalCharges - 1) * KineticEnergy / (kB);


        dftPtr->solve(true, false, false, false);
        force     = dftPtr->getForceonAtoms();
        double dt = d_TimeStep;
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
          {
            displacements[iCharge][0] =
              (dt * velocity[3 * iCharge + 0] -
               dt * dt / 2 * force[3 * iCharge + 0] / massAtoms[iCharge] *
                 haPerBohrToeVPerAng) *
              AngTobohr;
            displacements[iCharge][1] =
              (dt * velocity[3 * iCharge + 1] -
               dt * dt / 2 * force[3 * iCharge + 1] / massAtoms[iCharge] *
                 haPerBohrToeVPerAng) *
              AngTobohr;
            displacements[iCharge][2] =
              (dt * velocity[3 * iCharge + 2] -
               dt * dt / 2 * force[3 * iCharge + 2] / massAtoms[iCharge] *
                 haPerBohrToeVPerAng) *
              AngTobohr;
          }
        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
        GroundStateEnergyvalue  = dftPtr->getInternalEnergy();
        EntropicEnergyvalue     = dftPtr->getEntropicEnergy();
        KineticEnergyVector[0]  = KineticEnergy / haToeV;
        InternalEnergyVector[0] = GroundStateEnergyvalue;
        EntropicEnergyVector[0] = EntropicEnergyvalue;
        TotalEnergyVector[0]    = KineticEnergyVector[0] +
                               InternalEnergyVector[0] -
                               EntropicEnergyVector[0];
        if (dftParameters::verbosity >= 1)
          {
            pcout << "Velocity of atoms " << std::endl;
            for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
              {
                pcout << "Charge Id: " << iCharge << " "
                      << velocity[3 * iCharge + 0] << " "
                      << velocity[3 * iCharge + 1] << " "
                      << velocity[3 * iCharge + 2] << std::endl;
              }
          }

        if (dftParameters::verbosity >= 0 && !dftParameters::reproducible_output)
          {
            pcout << "---------------MD " << d_startingTimeStep
                  << "th STEP------------------ " << std::endl;
            pcout << " Temperature from velocities: " <<TemperatureFromVelocities << std::endl;
            pcout << " Kinetic Energy in Ha at timeIndex  "
                  << d_startingTimeStep << KineticEnergyVector[0] << std::endl;
            pcout << " Internal Energy in Ha at timeIndex  "
                  << d_startingTimeStep << InternalEnergyVector[0] << std::endl;
            pcout << " Entropic Energy in Ha at timeIndex  "
                  << d_startingTimeStep << EntropicEnergyVector[0] << std::endl;
            pcout << " Total Energy in Ha at timeIndex " <<TotalEnergyVector[0] << std::endl;
          }
        else if(dftParameters::verbosity >= 0 && dftParameters::reproducible_output ) 
          {
            pcout << "---------------MD " << d_startingTimeStep
                  << "th STEP------------------ " << std::endl;
            pcout << " Temperature from velocities: " << std::setprecision(2)
                  << TemperatureFromVelocities << std::endl;
            pcout << " Total Energy in Ha at timeIndex " <<std::setprecision(5)
                  << TotalEnergyVector[0] << std::endl;            
          } 

        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);

        //--------------------Completed Initialization
        //----------------------------------------------//
      }

    else if (d_restartFlag == 1)
      {
        if (Utilities::MPI::this_mpi_process(d_mpi_communicator) == 0 &&
            Utilities::MPI::this_mpi_process(d_interpoolcomm) == 0 &&
            Utilities::MPI::this_mpi_process(d_interBandGroupComm) == 0)
          {
            int           error;
            std::string   file1 = "TotalDisplacement.chk";
            std::ifstream readFile1(file1.c_str());
            if (!readFile1.fail())
              {
                error = remove(file1.c_str());
                AssertThrow(error == 0,
                            dealii::ExcMessage(std::string(
                              "Unable to remove file: " + file1 +
                              ", although it seems to exist. " +
                              "The error code is " +
                              dealii::Utilities::to_string(error) + ".")));
              }
            pcout << "Removed File: " << file1 << std::endl;
            std::string   file2 = "Displacement.chk";
            std::ifstream readFile2(file2.c_str());
            if (!readFile2.fail())
              {
                error = remove(file2.c_str());
                AssertThrow(error == 0,
                            dealii::ExcMessage(std::string(
                              "Unable to remove file: " + file2 +
                              ", although it seems to exist. " +
                              "The error code is " +
                              dealii::Utilities::to_string(error) + ".")));
              }
            pcout << "Removed File: " << file2 << std::endl;
            std::string   file3 = "/mdRestart/NHCThermostat.chk";
            std::ifstream readFile3(file3.c_str());
            if (!readFile3.fail() && d_ThermostatType == "NOSE_HOVER_CHAINS")
              {
                error = remove(file3.c_str());
                AssertThrow(error == 0,
                            dealii::ExcMessage(std::string(
                              "Unable to remove file: " + file3 +
                              ", although it seems to exist. " +
                              "The error code is " +
                              dealii::Utilities::to_string(error) + ".")));
              }
            pcout << "Removed File: " << file3 << std::endl;
          }

        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
        InitialiseFromRestartFile(displacements,
                                  velocity,
                                  force,
                                  KineticEnergyVector,
                                  InternalEnergyVector,
                                  TotalEnergyVector);
        if (dftParameters::verbosity > 1 && !dftParameters::reproducible_output)
          {
            pcout << "-- Starting Unwrapped Coordinates: --" << std::endl;
            for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
              pcout << d_atomFractionalunwrapped[iCharge][0] << " "
                    << d_atomFractionalunwrapped[iCharge][1] << " "
                    << d_atomFractionalunwrapped[iCharge][2] << " "
                    << d_atomFractionalunwrapped[iCharge][3] << " "
                    << d_atomFractionalunwrapped[iCharge][4] << std::endl;
          }
      }

    //--------------------Choosing Ensemble
    //----------------------------------------------//
    if (d_ThermostatType == "NO_CONTROL")
      {
        mdNVE(KineticEnergyVector,
              InternalEnergyVector,
              EntropicEnergyVector,
              TotalEnergyVector,
              displacements,
              velocity,
              force,
              massAtoms);
      }
    else if (d_ThermostatType == "RESCALE")
      {
        mdNVTrescaleThermostat(KineticEnergyVector,
                               InternalEnergyVector,
                               EntropicEnergyVector,
                               TotalEnergyVector,
                               displacements,
                               velocity,
                               force,
                               massAtoms);
      }
    else if (d_ThermostatType == "NOSE_HOVER_CHAINS")
      {
        mdNVTnosehoverchainsThermostat(KineticEnergyVector,
                                       InternalEnergyVector,
                                       EntropicEnergyVector,
                                       TotalEnergyVector,
                                       displacements,
                                       velocity,
                                       force,
                                       massAtoms);
      }
    else if (d_ThermostatType == "CSVR")
      {
        mdNVTsvrThermostat(KineticEnergyVector,
                           InternalEnergyVector,
                           EntropicEnergyVector,
                           TotalEnergyVector,
                           displacements,
                           velocity,
                           force,
                           massAtoms);
      }

    pcout << "MD run completed" << std::endl;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  molecularDynamicsClass<FEOrder, FEOrderElectro>::mdNVE(
    std::vector<double> &                      KineticEnergyVector,
    std::vector<double> &                      InternalEnergyVector,
    std::vector<double> &                      EntropicEnergyVector,
    std::vector<double> &                      TotalEnergyVector,
    std::vector<dealii::Tensor<1, 3, double>> &displacements,
    std::vector<double> &                      velocity,
    std::vector<double> &                      force,
    const std::vector<double> &                atomMass)
  {
    pcout << "---------------MDNVE() called------------------ " << std::endl;



    double KineticEnergy, GroundStateEnergyvalue, EntropicEnergyvalue;
    double TemperatureFromVelocities;
    for (d_TimeIndex = d_startingTimeStep + 1;
         d_TimeIndex < d_startingTimeStep + d_numberofSteps;
         d_TimeIndex++)
      {
        double step_time, curr_time;
        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
        step_time     = MPI_Wtime();
        KineticEnergy = velocityVerlet(
          velocity, displacements, atomMass, KineticEnergy, force);
        GroundStateEnergyvalue = dftPtr->getInternalEnergy();
        EntropicEnergyvalue    = dftPtr->getEntropicEnergy();
        KineticEnergyVector[d_TimeIndex - d_startingTimeStep] =
          KineticEnergy / haToeV;
        InternalEnergyVector[d_TimeIndex - d_startingTimeStep] =
          GroundStateEnergyvalue;
        EntropicEnergyVector[d_TimeIndex - d_startingTimeStep] =
          EntropicEnergyvalue;
        TotalEnergyVector[d_TimeIndex - d_startingTimeStep] =
          KineticEnergyVector[d_TimeIndex - d_startingTimeStep] +
          InternalEnergyVector[d_TimeIndex - d_startingTimeStep] -
          EntropicEnergyVector[d_TimeIndex - d_startingTimeStep];
        TemperatureFromVelocities =
          2.0 / 3.0 / double(d_numberGlobalCharges - 1) * KineticEnergy / (kB);

        // Based on verbose print required MD details...
        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
        step_time = MPI_Wtime() - step_time;
        if (dftParameters::verbosity >= 1)
          {
            pcout << "Velocity of atoms " << std::endl;
            for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
              {
                pcout << "Charge Id: " << iCharge << " "
                      << velocity[3 * iCharge + 0] << " "
                      << velocity[3 * iCharge + 1] << " "
                      << velocity[3 * iCharge + 2] << std::endl;
              }
          }
        // Printing COM velocity
        double COM = 0.0;
        double vx  = 0.0;
        double vy  = 0.0;
        double vz  = 0.0;
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
          {
            vx += atomMass[iCharge] * velocity[3 * iCharge + 0];
            vy += atomMass[iCharge] * velocity[3 * iCharge + 1];
            vz += atomMass[iCharge] * velocity[3 * iCharge + 2];
            COM += atomMass[iCharge];
          }
        vx /= COM;
        vy /= COM;
        vz /= COM;
        // pcout<<" The Center of Mass Velocity from NVE: "<<vx<<" "<<vy<<"
        // "<<vz<<std::endl;
<<<<<<< HEAD
        if (dftParameters::verbosity >= 0 && !dftParameters::reproducible_output)
          {
            pcout << "---------------MD STEP: " << d_TimeIndex
                  << " ------------------ " << std::endl;     
            pcout << "Time taken for md step: " << step_time << std::endl;
=======
        if (dftParameters::verbosity >= 1)
          {
            pcout << "---------------MD STEP: " << d_TimeIndex
                  << " ------------------ " << std::endl;
            if (!dftParameters::reproducible_output)
              pcout << "Time taken for md step: " << step_time << std::endl;
>>>>>>> 19e807581d8cdd011d08ecd93433dec293b296ee
            pcout << " Temperature from velocities: " << d_TimeIndex << " "
                  << TemperatureFromVelocities << std::endl;
            pcout << " Kinetic Energy in Ha at timeIndex " << d_TimeIndex << " "
                  << KineticEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
            pcout << " Internal Energy in Ha at timeIndex " << d_TimeIndex
                  << " "
                  << InternalEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
            pcout << " Entropic Energy in Ha at timeIndex " << d_TimeIndex
                  << " "
                  << EntropicEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
            pcout << " Total Energy in Ha at timeIndex " << d_TimeIndex << " "
                  << TotalEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
          }
        else if(dftParameters::verbosity >= 0 && dftParameters::reproducible_output ) 
          {
            pcout << "---------------MD " << d_TimeIndex
                  << "th STEP------------------ " << std::endl;
            pcout << " Temperature from velocities: " << std::setprecision(2)
                  << TemperatureFromVelocities << std::endl;
            pcout << " Total Energy in Ha at timeIndex " <<std::setprecision(5)
                  << TotalEnergyVector[d_TimeIndex - d_startingTimeStep] << std::endl;            
          }           
        writeRestartFile(displacements,
                         velocity,
                         force,
                         KineticEnergyVector,
                         InternalEnergyVector,
                         TotalEnergyVector,
                         d_TimeIndex);
        writeTotalDisplacementFile(displacements, d_TimeIndex);

        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
        curr_time = MPI_Wtime() - d_MDstartWallTime;
        if (!dftParameters::reproducible_output)
          pcout << "*****Time Completed till NOW: " << curr_time << std::endl;
        AssertThrow((d_MaxWallTime - (curr_time + 1.05 * step_time)) > 1.0,
                    ExcMessage(
                      "DFT-FE Exit: Max Wall Time exceeded User Limit"));
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  molecularDynamicsClass<FEOrder, FEOrderElectro>::mdNVTrescaleThermostat(
    std::vector<double> &                      KineticEnergyVector,
    std::vector<double> &                      InternalEnergyVector,
    std::vector<double> &                      EntropicEnergyVector,
    std::vector<double> &                      TotalEnergyVector,
    std::vector<dealii::Tensor<1, 3, double>> &displacements,
    std::vector<double> &                      velocity,
    std::vector<double> &                      force,
    const std::vector<double> &                atomMass)
  {
    pcout
      << "---------------mdNVTrescaleThermostat() called ------------------ "
      << std::endl;
    double KineticEnergy, EntropicEnergyvalue, GroundStateEnergyvalue;
    double TemperatureFromVelocities;
    for (d_TimeIndex = d_startingTimeStep + 1;
         d_TimeIndex < d_startingTimeStep + d_numberofSteps;
         d_TimeIndex++)
      {
        double step_time, curr_time;
        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
        step_time = MPI_Wtime();


        KineticEnergy = velocityVerlet(
          velocity, displacements, atomMass, KineticEnergy, force);
        TemperatureFromVelocities =
          2.0 / 3.0 / double(d_numberGlobalCharges - 1) * KineticEnergy / (kB);
        if (d_TimeIndex % d_ThermostatTimeConstant == 0)
          {
            KineticEnergy = RescaleVelocities(velocity,
                                              KineticEnergy,
                                              atomMass,
                                              TemperatureFromVelocities);
          }

        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
        GroundStateEnergyvalue = dftPtr->getInternalEnergy();
        EntropicEnergyvalue    = dftPtr->getEntropicEnergy();
        KineticEnergyVector[d_TimeIndex - d_startingTimeStep] =
          KineticEnergy / haToeV;
        InternalEnergyVector[d_TimeIndex - d_startingTimeStep] =
          GroundStateEnergyvalue;
        EntropicEnergyVector[d_TimeIndex - d_startingTimeStep] =
          EntropicEnergyvalue;
        TotalEnergyVector[d_TimeIndex - d_startingTimeStep] =
          KineticEnergyVector[d_TimeIndex - d_startingTimeStep] +
          InternalEnergyVector[d_TimeIndex - d_startingTimeStep] -
          EntropicEnergyVector[d_TimeIndex - d_startingTimeStep];
        TemperatureFromVelocities =
          2.0 / 3.0 / double(d_numberGlobalCharges - 1) * KineticEnergy / (kB);

        // Based on verbose print required MD details...
        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
        step_time = MPI_Wtime() - step_time;
        if (dftParameters::verbosity >= 1)
          {
            pcout << "Velocity of atoms " << std::endl;
            for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
              {
                pcout << "Charge Id: " << iCharge << " "
                      << velocity[3 * iCharge + 0] << " "
                      << velocity[3 * iCharge + 1] << " "
                      << velocity[3 * iCharge + 2] << std::endl;
              }
          }
        // Printing COM velocity
        double COM = 0.0;
        double vx  = 0.0;
        double vy  = 0.0;
        double vz  = 0.0;
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
          {
            vx += atomMass[iCharge] * velocity[3 * iCharge + 0];
            vy += atomMass[iCharge] * velocity[3 * iCharge + 1];
            vz += atomMass[iCharge] * velocity[3 * iCharge + 2];
            COM += atomMass[iCharge];
          }
        vx /= COM;
        vy /= COM;
        vz /= COM;
        // pcout<<" The Center of Mass Velocity from Rescale Thermostat:
        // "<<vx<<" "<<vy<<" "<<vz<<std::endl;

        if (dftParameters::verbosity >= 0 && !dftParameters::reproducible_output)
          {
            pcout << "---------------MD STEP: " << d_TimeIndex
<<<<<<< HEAD
                  << " ------------------ " << std::endl;      
            pcout << "Time taken for md step: " << step_time << std::endl;
=======
                  << " ------------------ " << std::endl;
            if (!dftParameters::reproducible_output)
              pcout << "Time taken for md step: " << step_time << std::endl;
>>>>>>> 19e807581d8cdd011d08ecd93433dec293b296ee
            pcout << " Temperature from velocities: " << d_TimeIndex << " "
                  << TemperatureFromVelocities << std::endl;
            pcout << " Kinetic Energy in Ha at timeIndex " << d_TimeIndex << " "
                  << KineticEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
            pcout << " Internal Energy in Ha at timeIndex " << d_TimeIndex
                  << " "
                  << InternalEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
            pcout << " Entropic Energy in Ha at timeIndex " << d_TimeIndex
                  << " "
                  << EntropicEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
            pcout << " Total Energy in Ha at timeIndex " << d_TimeIndex << " "
                  << TotalEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
          }
        else if(dftParameters::verbosity >= 0 && dftParameters::reproducible_output ) 
          {
            pcout << "---------------MD " << d_TimeIndex
                  << "th STEP------------------ " << std::endl;
            pcout << " Temperature from velocities: " << std::setprecision(2)
                  << TemperatureFromVelocities << std::endl;
            pcout << " Total Energy in Ha at timeIndex " <<std::setprecision(5)
                  << TotalEnergyVector[d_TimeIndex - d_startingTimeStep] << std::endl;            
          }           
        writeRestartFile(displacements,
                         velocity,
                         force,
                         KineticEnergyVector,
                         InternalEnergyVector,
                         TotalEnergyVector,
                         d_TimeIndex);
        writeTotalDisplacementFile(displacements, d_TimeIndex);

        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
        curr_time = MPI_Wtime() - d_MDstartWallTime;
        if (!dftParameters::reproducible_output)
          pcout << "*****Time Completed till NOW: " << curr_time << std::endl;
        AssertThrow((d_MaxWallTime - (curr_time + 1.05 * step_time)) > 1.0,
                    ExcMessage(
                      "DFT-FE Exit: Max Wall Time exceeded User Limit"));
      }
  }
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  molecularDynamicsClass<FEOrder, FEOrderElectro>::
    mdNVTnosehoverchainsThermostat(
      std::vector<double> &                      KineticEnergyVector,
      std::vector<double> &                      InternalEnergyVector,
      std::vector<double> &                      EntropicEnergyVector,
      std::vector<double> &                      TotalEnergyVector,
      std::vector<dealii::Tensor<1, 3, double>> &displacements,
      std::vector<double> &                      velocity,
      std::vector<double> &                      force,
      const std::vector<double> &                atomMass)
  {
    pcout
      << "--------------mdNVTnosehoverchainsThermostat() called ------------------ "
      << std::endl;

    double KineticEnergy, GroundStateEnergyvalue, EntropicEnergyvalue;
    double TemperatureFromVelocities;
    double nhctimeconstant;
    TemperatureFromVelocities = 2.0 / 3.0 / double(d_numberGlobalCharges - 1) *
                                KineticEnergyVector[0] / (kB);
    std::vector<double> ThermostatMass(2, 0.0);
    std::vector<double> Thermostatvelocity(2, 0.0);
    std::vector<double> Thermostatposition(2, 0.0);
    std::vector<double> NoseHoverExtendedLagrangianvector(d_numberofSteps, 0.0);
    nhctimeconstant = d_ThermostatTimeConstant * d_TimeStep;
    if (d_restartFlag == 0)
      {
        ThermostatMass[0] = 3 * (d_numberGlobalCharges - 1) * kB *
                            d_startingTemperature *
                            (nhctimeconstant * nhctimeconstant);
        ThermostatMass[1] =
          kB * d_startingTemperature * (nhctimeconstant * nhctimeconstant);
        /*pcout << "Time Step " <<timeStep<<" Q2: "<<ThermostatMass[1]<<"Time
           Constant"<<nhctimeconstant<<"no. of atoms"<<
           d_numberGlobalCharges<<"Starting Temp: "<<
              d_startingTemperature<<"---"<<3*(d_numberGlobalCharges-1)*kB*d_startingTemperature*(nhctimeconstant*nhctimeconstant)<<std::endl;*/
      }
    else
      {
        InitialiseFromRestartNHCFile(Thermostatvelocity,
                                     Thermostatposition,
                                     ThermostatMass);
      }

    for (d_TimeIndex = d_startingTimeStep + 1;
         d_TimeIndex < d_startingTimeStep + d_numberofSteps;
         d_TimeIndex++)
      {
        double step_time, curr_time;
        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
        step_time = MPI_Wtime();
        NoseHoverChains(
          velocity,
          Thermostatvelocity,
          Thermostatposition,
          ThermostatMass,
          KineticEnergyVector[d_TimeIndex - 1 - d_startingTimeStep] * haToeV,
          d_startingTemperature);

        KineticEnergy = velocityVerlet(
          velocity, displacements, atomMass, KineticEnergy, force);

        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);

        NoseHoverChains(velocity,
                        Thermostatvelocity,
                        Thermostatposition,
                        ThermostatMass,
                        KineticEnergy,
                        d_startingTemperature);
        KineticEnergy = 0.0;
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
          {
            KineticEnergy +=
              0.5 * atomMass[iCharge] *
              (velocity[3 * iCharge + 0] * velocity[3 * iCharge + 0] +
               velocity[3 * iCharge + 1] * velocity[3 * iCharge + 1] +
               velocity[3 * iCharge + 2] * velocity[3 * iCharge + 2]);
          }


        GroundStateEnergyvalue = dftPtr->getInternalEnergy();
        EntropicEnergyvalue    = dftPtr->getEntropicEnergy();
        KineticEnergyVector[d_TimeIndex - d_startingTimeStep] =
          KineticEnergy / haToeV;
        InternalEnergyVector[d_TimeIndex - d_startingTimeStep] =
          GroundStateEnergyvalue;
        EntropicEnergyVector[d_TimeIndex - d_startingTimeStep] =
          EntropicEnergyvalue;
        TotalEnergyVector[d_TimeIndex - d_startingTimeStep] =
          KineticEnergyVector[d_TimeIndex - d_startingTimeStep] +
          InternalEnergyVector[d_TimeIndex - d_startingTimeStep] -
          EntropicEnergyVector[d_TimeIndex - d_startingTimeStep];
        TemperatureFromVelocities =
          2.0 / 3.0 / double(d_numberGlobalCharges - 1) * KineticEnergy / (kB);
        NoseHoverExtendedLagrangianvector[d_TimeIndex - d_startingTimeStep] =
          NoseHoverExtendedLagrangian(
            Thermostatvelocity,
            Thermostatposition,
            ThermostatMass,
            KineticEnergyVector[d_TimeIndex - d_startingTimeStep],
            TotalEnergyVector[d_TimeIndex - d_startingTimeStep],
            TemperatureFromVelocities);

        // Based on verbose print required MD details...

        if (dftParameters::verbosity >= 1)
          {
            pcout << "Velocity of atoms " << std::endl;
            for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
              {
                pcout << "Charge Id: " << iCharge << " "
                      << velocity[3 * iCharge + 0] << " "
                      << velocity[3 * iCharge + 1] << " "
                      << velocity[3 * iCharge + 2] << std::endl;
              }
          }
        // Printing COM velocity
        double COM = 0.0;
        double vx  = 0.0;
        double vy  = 0.0;
        double vz  = 0.0;
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
          {
            vx += atomMass[iCharge] * velocity[3 * iCharge + 0];
            vy += atomMass[iCharge] * velocity[3 * iCharge + 1];
            vz += atomMass[iCharge] * velocity[3 * iCharge + 2];
            COM += atomMass[iCharge];
          }
        vx /= COM;
        vy /= COM;
        vz /= COM;
        // pcout<<" The Center of Mass Velocity from NHC: "<<vx<<" "<<vy<<"
        // "<<vz<<std::endl;


        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
        step_time = MPI_Wtime() - step_time;


        if (dftParameters::verbosity >= 0 && !dftParameters::reproducible_output )
          {
            pcout << "---------------MD STEP: " << d_TimeIndex
<<<<<<< HEAD
                  << " ------------------ " << std::endl;       
            pcout << "Time taken for md step: " << step_time << std::endl;
=======
                  << " ------------------ " << std::endl;
            if (!dftParameters::reproducible_output)
              pcout << "Time taken for md step: " << step_time << std::endl;
>>>>>>> 19e807581d8cdd011d08ecd93433dec293b296ee
            pcout << " Temperature from velocities: " << d_TimeIndex << " "
                  << TemperatureFromVelocities << std::endl;
            pcout << " Kinetic Energy in Ha at timeIndex " << d_TimeIndex << " "
                  << KineticEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
            pcout << " Internal Energy in Ha at timeIndex " << d_TimeIndex
                  << " "
                  << InternalEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
            pcout << " Entropic Energy in Ha at timeIndex " << d_TimeIndex
                  << " "
                  << EntropicEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
            pcout << " Total Energy in Ha at timeIndex " << d_TimeIndex << " "
                  << TotalEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
            pcout << "Nose Hover Extended Lagrangian  in Ha at timeIndex "
                  << d_TimeIndex << " "
                  << NoseHoverExtendedLagrangianvector[d_TimeIndex -
                                                       d_startingTimeStep]
                  << std::endl;
          }
        else if(dftParameters::verbosity >= 0 && dftParameters::reproducible_output ) 
          {
            pcout << "---------------MD " << d_TimeIndex
                  << "th STEP------------------ " << std::endl;
            pcout << " Temperature from velocities: " << std::setprecision(2)
                  << TemperatureFromVelocities << std::endl;
            pcout << " Total Energy in Ha at timeIndex " <<std::setprecision(5)
                  << TotalEnergyVector[d_TimeIndex - d_startingTimeStep] << std::endl;            
          } 
        writeRestartFile(displacements,
                         velocity,
                         force,
                         KineticEnergyVector,
                         InternalEnergyVector,
                         TotalEnergyVector,
                         d_TimeIndex);
        writeRestartNHCfile(Thermostatvelocity,
                            Thermostatposition,
                            ThermostatMass,
                            d_TimeIndex);
        writeTotalDisplacementFile(displacements, d_TimeIndex);

        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
        curr_time = MPI_Wtime() - d_MDstartWallTime;
        //  pcout<<"*****Time Completed till NOW: "<<curr_time<<std::endl;
        AssertThrow(
          (d_MaxWallTime - (curr_time + 1.05 * step_time)) > 1.0,
          ExcMessage(
            "DFT-FE Exit: Max Wall Time exceeded User Limit")); // Determine
                                                                // Exit sequence
                                                                // ..
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  molecularDynamicsClass<FEOrder, FEOrderElectro>::mdNVTsvrThermostat(
    std::vector<double> &                      KineticEnergyVector,
    std::vector<double> &                      InternalEnergyVector,
    std::vector<double> &                      EntropicEnergyVector,
    std::vector<double> &                      TotalEnergyVector,
    std::vector<dealii::Tensor<1, 3, double>> &displacements,
    std::vector<double> &                      velocity,
    std::vector<double> &                      force,
    const std::vector<double> &                atomMass)
  {
    pcout << "---------------mdNVTsvrThermostat() called ------------------ "
          << std::endl;
    double KineticEnergy, GroundStateEnergyvalue, EntropicEnergyvalue;
    double TemperatureFromVelocities;
    double KEref = 3.0 / 2.0 * double(d_numberGlobalCharges - 1) * kB *
                   d_startingTemperature;

    for (d_TimeIndex = d_startingTimeStep + 1;
         d_TimeIndex < d_startingTimeStep + d_numberofSteps;
         d_TimeIndex++)
      {
        double step_time, curr_time;

        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
        step_time = MPI_Wtime();


        KineticEnergy = velocityVerlet(
          velocity, displacements, atomMass, KineticEnergy, force);


        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);

        KineticEnergy = svr(velocity, KineticEnergy, KEref);
        TemperatureFromVelocities =
          2.0 / 3.0 / double(d_numberGlobalCharges - 1) * KineticEnergy / (kB);
        GroundStateEnergyvalue = dftPtr->getInternalEnergy();
        EntropicEnergyvalue    = dftPtr->getEntropicEnergy();
        KineticEnergyVector[d_TimeIndex - d_startingTimeStep] =
          KineticEnergy / haToeV;
        InternalEnergyVector[d_TimeIndex - d_startingTimeStep] =
          GroundStateEnergyvalue;
        EntropicEnergyVector[d_TimeIndex - d_startingTimeStep] =
          EntropicEnergyvalue;
        TotalEnergyVector[d_TimeIndex - d_startingTimeStep] =
          KineticEnergyVector[d_TimeIndex - d_startingTimeStep] +
          InternalEnergyVector[d_TimeIndex - d_startingTimeStep] -
          EntropicEnergyVector[d_TimeIndex - d_startingTimeStep];
        TemperatureFromVelocities =
          2.0 / 3.0 / double(d_numberGlobalCharges - 1) * KineticEnergy / (kB);

        // Based on verbose print required MD details...
        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
        step_time = MPI_Wtime() - step_time;
        if (dftParameters::verbosity >= 1)
          {
            pcout << "Velocity of atoms " << std::endl;
            for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
              {
                pcout << "Charge Id: " << iCharge << " "
                      << velocity[3 * iCharge + 0] << " "
                      << velocity[3 * iCharge + 1] << " "
                      << velocity[3 * iCharge + 2] << std::endl;
              }
          }
        // Printing COM velocity
        double COM = 0.0;
        double vx  = 0.0;
        double vy  = 0.0;
        double vz  = 0.0;
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
          {
            vx += atomMass[iCharge] * velocity[3 * iCharge + 0];
            vy += atomMass[iCharge] * velocity[3 * iCharge + 1];
            vz += atomMass[iCharge] * velocity[3 * iCharge + 2];
            COM += atomMass[iCharge];
          }
        vx /= COM;
        vy /= COM;
        vz /= COM;
        // pcout<<" The Center of Mass Velocity from CSVR: "<<vx<<" "<<vy<<"
        // "<<vz<<std::endl;
        if (dftParameters::verbosity >= 0 && !dftParameters::reproducible_output )
          {
            pcout << "---------------MD STEP: " << d_TimeIndex
<<<<<<< HEAD
                  << " ------------------ " << std::endl;       
            pcout << "Time taken for md step: " << step_time << std::endl;
=======
                  << " ------------------ " << std::endl;
            if (!dftParameters::reproducible_output)
              pcout << "Time taken for md step: " << step_time << std::endl;
>>>>>>> 19e807581d8cdd011d08ecd93433dec293b296ee
            pcout << " Temperature from velocities: " << d_TimeIndex << " "
                  << TemperatureFromVelocities << std::endl;
            pcout << " Kinetic Energy in Ha at timeIndex " << d_TimeIndex << " "
                  << KineticEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
            pcout << " Internal Energy in Ha at timeIndex " << d_TimeIndex
                  << " "
                  << InternalEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
            pcout << " Entropic Energy in Ha at timeIndex " << d_TimeIndex
                  << " "
                  << EntropicEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
            pcout << " Total Energy in Ha at timeIndex " << d_TimeIndex << " "
                  << TotalEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
          }
        else if(dftParameters::verbosity >= 0 && dftParameters::reproducible_output ) 
          {
            pcout << "---------------MD " << d_TimeIndex
                  << "th STEP------------------ " << std::endl;
            pcout << " Temperature from velocities: " << std::setprecision(2)
                  << TemperatureFromVelocities << std::endl;
            pcout << " Total Energy in Ha at timeIndex " <<std::setprecision(5)
                  << TotalEnergyVector[d_TimeIndex - d_startingTimeStep] << std::endl;            
          } 
        writeRestartFile(displacements,
                         velocity,
                         force,
                         KineticEnergyVector,
                         InternalEnergyVector,
                         TotalEnergyVector,
                         d_TimeIndex);
        writeTotalDisplacementFile(displacements, d_TimeIndex);

        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
        curr_time = MPI_Wtime() - d_MDstartWallTime;
        if (!dftParameters::reproducible_output)
          pcout << "*****Time Completed till NOW: " << curr_time << std::endl;
        AssertThrow((d_MaxWallTime - (curr_time + 1.05 * step_time)) > 1.0,
                    ExcMessage(
                      "DFT-FE Exit: Max Wall Time exceeded User Limit"));
      }
  }



  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  molecularDynamicsClass<FEOrder, FEOrderElectro>::velocityVerlet(
    std::vector<double> &                      v,
    std::vector<dealii::Tensor<1, 3, double>> &r,
    const std::vector<double> &                atomMass,
    double                                     KE,
    std::vector<double> &                      forceOnAtoms)
  {
    int    i;
    double totalKE;
    KE                       = 0.0;
    double              dt   = d_TimeStep;
    double              dt_2 = dt / 2;
    double              COMM = 0.0;
    double              COMx = 0.0;
    double              COMy = 0.0;
    double              COMz = 0.0;
    std::vector<double> rloc(3 * d_numberGlobalCharges, 0.0);
    if (Utilities::MPI::this_mpi_process(d_mpi_communicator) == 0 &&
        Utilities::MPI::this_mpi_process(d_interpoolcomm) == 0 &&
        Utilities::MPI::this_mpi_process(d_interBandGroupComm) == 0)
      {
        for (i = 0; i < d_numberGlobalCharges; i++)
          {
            /*Computing New position as taylor expansion about r(t) O(dt^3) */
            rloc[3 * i + 0] =
              (dt * v[3 * i + 0] - dt * dt_2 * forceOnAtoms[3 * i + 0] /
                                     atomMass[i] * haPerBohrToeVPerAng) *
              AngTobohr; // New position of x cordinate
            rloc[3 * i + 1] =
              (dt * v[3 * i + 1] - dt * dt_2 * forceOnAtoms[3 * i + 1] /
                                     atomMass[i] * haPerBohrToeVPerAng) *
              AngTobohr; // New Position of Y cordinate
            rloc[3 * i + 2] =
              (dt * v[3 * i + 2] - dt * dt_2 * forceOnAtoms[3 * i + 2] /
                                     atomMass[i] * haPerBohrToeVPerAng) *
              AngTobohr; // New POsition of Z cordinate



            /* Computing velocity from v(t) to v(t+dt/2) */

            v[3 * i + 0] = v[3 * i + 0] - forceOnAtoms[3 * i + 0] /
                                            atomMass[i] * dt_2 *
                                            haPerBohrToeVPerAng;
            v[3 * i + 1] = v[3 * i + 1] - forceOnAtoms[3 * i + 1] /
                                            atomMass[i] * dt_2 *
                                            haPerBohrToeVPerAng;
            v[3 * i + 2] = v[3 * i + 2] - forceOnAtoms[3 * i + 2] /
                                            atomMass[i] * dt_2 *
                                            haPerBohrToeVPerAng;



            COMM += atomMass[i];
            COMx += atomMass[i] * r[i][0];
            COMy += atomMass[i] * r[i][1];
            COMz += atomMass[i] * r[i][2];
          }



        COMx /= COMM;
        COMy /= COMM;
        COMz /= COMM;
      }
    MPI_Bcast(
      &(v[0]), 3 * d_numberGlobalCharges, MPI_DOUBLE, 0, d_mpi_communicator);
    MPI_Bcast(
      &(v[0]), 3 * d_numberGlobalCharges, MPI_DOUBLE, 0, d_interBandGroupComm);
    MPI_Bcast(
      &(v[0]), 3 * d_numberGlobalCharges, MPI_DOUBLE, 0, d_interpoolcomm);
    if (Utilities::MPI::this_mpi_process(d_mpi_communicator) == 0 &&
        Utilities::MPI::this_mpi_process(d_interpoolcomm) == 0 &&
        Utilities::MPI::this_mpi_process(d_interBandGroupComm) == 0)
      {
        for (i = 0; i < d_numberGlobalCharges; i++)
          {
            rloc[3 * i + 0] -= COMx;
            rloc[3 * i + 1] -= COMy;
            rloc[3 * i + 2] -= COMz;
          }
      }

    MPI_Bcast(
      &(rloc[0]), 3 * d_numberGlobalCharges, MPI_DOUBLE, 0, d_mpi_communicator);
    MPI_Bcast(&(rloc[0]),
              3 * d_numberGlobalCharges,
              MPI_DOUBLE,
              0,
              d_interBandGroupComm);
    MPI_Bcast(
      &(rloc[0]), 3 * d_numberGlobalCharges, MPI_DOUBLE, 0, d_interpoolcomm);
    for (unsigned int i = 0; i < d_numberGlobalCharges; ++i)
      {
        for (unsigned int j = 0; j < 3; ++j)
          {
            r[i][j] = rloc[i * 3 + j];
          }
      }

    double update_time;

    update_time = MPI_Wtime();

    dftPtr->updateAtomPositionsAndMoveMesh(
      r, dftParameters::maxJacobianRatioFactorForMD, false);

    if (dftParameters::verbosity >= 1)
      {
        std::vector<std::vector<double>> atomLocations;
        atomLocations = dftPtr->getAtomLocations();
        pcout << "Displacement  " << std::endl;
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
          {
            if (atomLocations[iCharge][0] == dftParameters::MDTrack)
              {
                pcout << "###Charge Id: " << iCharge << " " << r[iCharge][0]
                      << " " << r[iCharge][1] << " " << r[iCharge][2]
                      << std::endl;
              }
            else
              {
                pcout << "Charge Id: " << iCharge << " " << r[iCharge][0] << " "
                      << r[iCharge][1] << " " << r[iCharge][2] << std::endl;
              }
          }
      }
    for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
      {
        d_atomFractionalunwrapped[iCharge][2] =
          d_atomFractionalunwrapped[iCharge][2] +
          r[iCharge][0] / d_domainLength[0];
        d_atomFractionalunwrapped[iCharge][3] =
          d_atomFractionalunwrapped[iCharge][3] +
          r[iCharge][1] / d_domainLength[0];
        d_atomFractionalunwrapped[iCharge][4] =
          d_atomFractionalunwrapped[iCharge][4] +
          r[iCharge][2] / d_domainLength[0];
      }
    if (dftParameters::verbosity > 1 && !dftParameters::reproducible_output)
      {
        pcout << "---- Updated Unwrapped Coordinates: -----" << std::endl;
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
          {
            if (d_atomFractionalunwrapped[iCharge][0] == dftParameters::MDTrack)
              {
                pcout << "$$$ Charge No. " << iCharge << " "
                      << d_atomFractionalunwrapped[iCharge][2] << " "
                      << d_atomFractionalunwrapped[iCharge][3] << " "
                      << d_atomFractionalunwrapped[iCharge][4] << std::endl;
              }
            else
              {
                pcout << "Charge No. " << iCharge << " "
                      << d_atomFractionalunwrapped[iCharge][2] << " "
                      << d_atomFractionalunwrapped[iCharge][3] << " "
                      << d_atomFractionalunwrapped[iCharge][4] << std::endl;
              }
          }
      }



    MPI_Barrier(d_mpi_communicator);
    MPI_Barrier(d_interBandGroupComm);
    MPI_Barrier(d_interpoolcomm);

    update_time = MPI_Wtime() - update_time;

    if (dftParameters::verbosity >= 1)
      pcout << "Time taken for updateAtomPositionsAndMoveMesh: " << update_time
            << std::endl;
    dftPtr->solve(true, false, false, false);
    forceOnAtoms = dftPtr->getForceonAtoms();
    // Call Force
    totalKE = 0.0;
    /* Second half of velocty verlet */
    if (Utilities::MPI::this_mpi_process(d_mpi_communicator) == 0 &&
        Utilities::MPI::this_mpi_process(d_interpoolcomm) == 0 &&
        Utilities::MPI::this_mpi_process(d_interBandGroupComm) == 0)
      {
        for (i = 0; i < d_numberGlobalCharges; i++)
          {
            v[3 * i + 0] = v[3 * i + 0] - forceOnAtoms[3 * i + 0] /
                                            atomMass[i] * dt_2 *
                                            haPerBohrToeVPerAng;
            v[3 * i + 1] = v[3 * i + 1] - forceOnAtoms[3 * i + 1] /
                                            atomMass[i] * dt_2 *
                                            haPerBohrToeVPerAng;
            v[3 * i + 2] = v[3 * i + 2] - forceOnAtoms[3 * i + 2] /
                                            atomMass[i] * dt_2 *
                                            haPerBohrToeVPerAng;



            totalKE +=
              0.5 * atomMass[i] *
              (v[3 * i + 0] * v[3 * i + 0] + v[3 * i + 1] * v[3 * i + 1] +
               v[3 * i + 2] * v[3 * i + 2]);
          }
      }
    MPI_Bcast(
      &(v[0]), 3 * d_numberGlobalCharges, MPI_DOUBLE, 0, d_mpi_communicator);
    MPI_Bcast(
      &(v[0]), 3 * d_numberGlobalCharges, MPI_DOUBLE, 0, d_interBandGroupComm);
    MPI_Bcast(
      &(v[0]), 3 * d_numberGlobalCharges, MPI_DOUBLE, 0, d_interpoolcomm);
    // Printing COM velocity
    double COM = 0.0;
    double vx  = 0.0;
    double vy  = 0.0;
    double vz  = 0.0;
    COMx       = 0.0;
    COMy       = 0.0;
    COMz       = 0.0;
    for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
      {
        vx += atomMass[iCharge] * v[3 * iCharge + 0];
        vy += atomMass[iCharge] * v[3 * iCharge + 1];
        vz += atomMass[iCharge] * v[3 * iCharge + 2];
        COM += atomMass[iCharge];
        COMx += atomMass[iCharge] * r[iCharge][0];
        COMy += atomMass[iCharge] * r[iCharge][1];
        COMz += atomMass[iCharge] * r[iCharge][2];
        // totalKE +=
        // 0.5*atomMass[i]*(v[3*iCharge+0]*v[3*iCharge+0]+v[3*iCharge+1]*v[3*iCharge+1]
        // + v[3*iCharge+2]*v[3*iCharge+2]);
      }
    vx /= COM;
    vy /= COM;
    vz /= COM;
    COMx /= COM;
    COMy /= COM;
    COMz /= COM;
    // pcout<<" The Center of Mass Velocity from Velocity Verlet: "<<vx<<"
    // "<<vy<<" "<<vz<<std::endl; pcout<<" The Center of Mass Position from
    // Velocity Verlet: "<<COMx<<" "<<COMy<<" "<<COMz<<std::endl;
    KE = totalKE;
    return KE;
  }



  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  molecularDynamicsClass<FEOrder, FEOrderElectro>::RescaleVelocities(
    std::vector<double> &      v,
    double                     KE,
    const std::vector<double> &M,
    double                     Temperature)
  {
    pcout << "Rescale Thermostat: Before rescaling temperature= " << Temperature
          << " K" << std::endl;
    AssertThrow(
      std::fabs(Temperature - 0.0) > 0.00001,
      ExcMessage(
        "DFT-FE Error: Temperature reached O K")); // Determine Exit sequence ..
    KE = 0.0;
    for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
      {
        v[3 * iCharge + 0] =
          v[3 * iCharge + 0] * sqrt(d_startingTemperature / Temperature);
        v[3 * iCharge + 1] =
          v[3 * iCharge + 1] * sqrt(d_startingTemperature / Temperature);
        v[3 * iCharge + 2] =
          v[3 * iCharge + 2] * sqrt(d_startingTemperature / Temperature);
        KE += 0.5 * M[iCharge] *
              (v[3 * iCharge + 0] * v[3 * iCharge + 0] +
               v[3 * iCharge + 1] * v[3 * iCharge + 1] +
               v[3 * iCharge + 2] * v[3 * iCharge + 2]);
      }
    return KE;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  molecularDynamicsClass<FEOrder, FEOrderElectro>::NoseHoverChains(
    std::vector<double> &v,
    std::vector<double> &v_e,
    std::vector<double> &e,
    std::vector<double>  Q,
    double               KE,
    double               Temperature)
  {
    double G1, G2, s;
    double L = 3 * (d_numberGlobalCharges - 1);
    /* Start Chain 1*/
    G2 = (Q[0] * v_e[0] * v_e[0] - kB * Temperature) / Q[1];
    // pcout << "v_e[0]:"<<v_e[0]<<std::endl;
    v_e[1] = v_e[1] + G2 * d_TimeStep / 4;
    // pcout << "v_e[1]:"<<v_e[1]<<std::endl;
    v_e[0] = v_e[0] * std::exp(-v_e[1] * d_TimeStep / 8);
    // pcout << "v_e[0]*std::exp(-v_e[1]*timeStep/8):"<<v_e[0]<<std::endl;
    G1     = (2 * KE - L * kB * Temperature) / Q[0];
    v_e[0] = v_e[0] + G1 * d_TimeStep / 4;
    // pcout << "v_e[0]+G1*timeStep/4:"<<v_e[0]<<std::endl;
    v_e[0] = v_e[0] * std::exp(-v_e[1] * d_TimeStep / 8);
    // pcout << "v_e[0]*std::exp(-v_e[1]*timeStep/8):"<<v_e[0]<<std::endl;
    e[0] = e[0] + v_e[0] * d_TimeStep / 2;
    e[1] = e[1] + v_e[1] * d_TimeStep / 2;
    s    = std::exp(-v_e[0] * d_TimeStep / 2);
    // pcout << "G2"<<G2<<" v_e1"<<v_e[1]<<"//"<<Temperature<<"Q[1]
    // "<<Q[1]<<"v_e[0] "<<v_e[0]<<"G1 "<<G1<< "Exponent: "<<std::exp(1)<<
    // std::endl;
    for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
      {
        v[3 * iCharge + 0] = s * v[3 * iCharge + 0];
        v[3 * iCharge + 1] = s * v[3 * iCharge + 1];
        v[3 * iCharge + 2] = s * v[3 * iCharge + 2];
      }
    KE     = KE * s * s;
    v_e[0] = v_e[0] * std::exp(-v_e[1] * d_TimeStep / 8);
    G1     = (2 * KE - L * kB * Temperature) / Q[0];
    v_e[0] = v_e[0] + G1 * d_TimeStep / 4;
    v_e[0] = v_e[0] * std::exp(-v_e[1] * d_TimeStep / 8);
    G2     = (Q[0] * v_e[0] * v_e[0] - kB * Temperature) / Q[1];
    v_e[1] = v_e[1] + G2 * d_TimeStep / 4;
    /* End Chain 1*/
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  molecularDynamicsClass<FEOrder, FEOrderElectro>::svr(std::vector<double> &v,
                                                       double               KE,
                                                       double KEref)
  {
    double       alphasq;
    unsigned int Nf = 3 * (d_numberGlobalCharges - 1);
    double       R1, Rsum;
    R1   = 0.0;
    Rsum = 0.0;
    if (Utilities::MPI::this_mpi_process(d_mpi_communicator) == 0 &&
        Utilities::MPI::this_mpi_process(d_interpoolcomm) == 0 &&
        Utilities::MPI::this_mpi_process(d_interBandGroupComm) == 0)
      {
        std::time_t                  now = std::time(0);
        boost::random::mt19937       gen{dftParameters::reproducible_output ? 0 : static_cast<std::uint32_t>(now)};
        boost::normal_distribution<> gaussianDist(0.0, 1.0);
        boost::variate_generator<boost::mt19937 &, boost::normal_distribution<>>
          generator(gen, gaussianDist);

        R1 = generator();
        /*
        if((Nf-1)%2 == 0)
          {
            boost::gamma_distribution<> my_gamma((Nf-1)/2,1);
            boost::variate_generator<boost::mt19937 &,
        boost::gamma_distribution<>> generator_gamma(gen, my_gamma); Rsum =
        generator_gamma();
          }
        else
          {
            Rsum = generator();
            Rsum = Rsum*Rsum;
            boost::gamma_distribution<> my_gamma((Nf-2)/2,1);
            boost::variate_generator<boost::mt19937 &,
        boost::gamma_distribution<>> generator_gamma(gen, my_gamma); Rsum +=
        generator_gamma();
          }
           */
        double temp;
        for (int dof = 1; dof < Nf; dof++)
          {
            temp = generator();
            Rsum = Rsum + temp * temp;
          }
        Rsum = R1 * R1 + Rsum;

        // Transfer data to all mpi procs
      }
    R1      = dealii::Utilities::MPI::sum(R1, d_mpi_communicator);
    Rsum    = dealii::Utilities::MPI::sum(Rsum, d_mpi_communicator);
    alphasq = 0.0;
    alphasq = alphasq + std::exp(-1 / double(d_ThermostatTimeConstant));
    alphasq =
      alphasq + (KEref / Nf / KE) *
                  (1 - std::exp(-1 / double(d_ThermostatTimeConstant))) *
                  (R1 * R1 + Rsum);
    alphasq =
      alphasq +
      2 * std::exp(-1 / 2 / double(d_ThermostatTimeConstant)) *
        std::sqrt(KEref / Nf / KE *
                  (1 - std::exp(-1 / double(d_ThermostatTimeConstant)))) *
        R1;
    // pcout<<"*** R1: "<<R1<<" Rsum : "<<Rsum<<" alphasq "<<alphasq<<"exp
    // ()"<<std::exp(-1/double(d_ThermostatTimeConstant))<<" timeconstant
    // "<<d_ThermostatTimeConstant<< std::endl;
    KE           = alphasq * KE;
    double alpha = std::sqrt(alphasq);
    for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
      {
        v[3 * iCharge + 0] = alpha * v[3 * iCharge + 0];
        v[3 * iCharge + 1] = alpha * v[3 * iCharge + 1];
        v[3 * iCharge + 2] = alpha * v[3 * iCharge + 2];
      }
    return KE;
  }



  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  molecularDynamicsClass<FEOrder, FEOrderElectro>::writeRestartFile(
    const std::vector<dealii::Tensor<1, 3, double>> &disp,
    const std::vector<double> &                      velocity,
    const std::vector<double> &                      force,
    const std::vector<double> &                      KineticEnergyVector,
    const std::vector<double> &                      InternalEnergyVector,
    const std::vector<double> &                      TotalEnergyVector,
    int                                              time)

  {
    if (dftParameters::reproducible_output == false)
      {
        // Writes the restart files for velocities and positions
        std::vector<std::vector<double>> fileForceData(
          d_numberGlobalCharges, std::vector<double>(3, 0.0));
        std::vector<std::vector<double>> fileDispData(d_numberGlobalCharges,
                                                      std::vector<double>(3,
                                                                          0.0));
        std::vector<std::vector<double>> fileVelocityData(
          d_numberGlobalCharges, std::vector<double>(3, 0.0));
        std::vector<std::vector<double>> timeIndexData(1,
                                                       std::vector<double>(1,
                                                                           0));
        std::vector<std::vector<double>> KEData(1, std::vector<double>(1, 0.0));
        std::vector<std::vector<double>> IEData(1, std::vector<double>(1, 0.0));
        std::vector<std::vector<double>> TEData(1, std::vector<double>(1, 0.0));


        timeIndexData[0][0]    = double(time);
        std::string Folder     = "mdRestart/Step";
        std::string tempfolder = Folder + std::to_string(time);
        mkdir(tempfolder.c_str(), ACCESSPERMS);
        Folder                 = "mdRestart";
        std::string newFolder3 = Folder + "/" + "time.chk";
        dftUtils::writeDataIntoFile(timeIndexData, newFolder3);
        KEData[0][0] = KineticEnergyVector[time - d_startingTimeStep];
        IEData[0][0] = InternalEnergyVector[time - d_startingTimeStep];
        TEData[0][0] = TotalEnergyVector[time - d_startingTimeStep];
        std::string newFolder4 = tempfolder + "/" + "KineticEnergy.chk";
        dftUtils::writeDataIntoFile(KEData, newFolder4);
        std::string newFolder5 = tempfolder + "/" + "InternalEnergy.chk";
        dftUtils::writeDataIntoFile(IEData, newFolder5);
        std::string newFolder6 = tempfolder + "/" + "TotalEnergy.chk";
        dftUtils::writeDataIntoFile(TEData, newFolder6);

        for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
          {
            fileForceData[iCharge][0] = force[3 * iCharge + 0];
            fileForceData[iCharge][1] = force[3 * iCharge + 1];
            fileForceData[iCharge][2] = force[3 * iCharge + 2];
          }
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
          {
            fileDispData[iCharge][0] = disp[iCharge][0];
            fileDispData[iCharge][1] = disp[iCharge][1];
            fileDispData[iCharge][2] = disp[iCharge][2];
          }
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
          {
            fileVelocityData[iCharge][0] = velocity[3 * iCharge + 0];
            fileVelocityData[iCharge][1] = velocity[3 * iCharge + 1];
            fileVelocityData[iCharge][2] = velocity[3 * iCharge + 2];
          }
        std::string cordFolder = tempfolder + "/";
        dftPtr->MDwriteDomainAndAtomCoordinates(cordFolder);
        if (time > 1)
          pcout << "#RESTART NOTE: Positions:-"
                << " Positions of TimeStep: " << time
                << " present in file atomsFracCoordCurrent.chk" << std::endl
                << " Positions of TimeStep: " << time - 1
                << " present in file atomsFracCoordCurrent.chk.old #"
                << std::endl;
        std::string newFolder1 = tempfolder + "/" + "velocity.chk";
        dftUtils::writeDataIntoFile(fileVelocityData, newFolder1);
        if (time > 1)
          pcout << "#RESTART NOTE: Velocity:-"
                << " Velocity of TimeStep: " << time
                << " present in file velocity.chk" << std::endl
                << " Velocity of TimeStep: " << time - 1
                << " present in file velocity.chk.old #" << std::endl;
        std::string newFolder2 = tempfolder + "/" + "force.chk";
        dftUtils::writeDataIntoFile(fileForceData, newFolder2);
        if (time > 1)
          pcout << "#RESTART NOTE: Force:-"
                << " Force of TimeStep: " << time
                << " present in file force.chk" << std::endl
                << " Forces of TimeStep: " << time - 1
                << " present in file force.chk.old #" << std::endl;
        std::string newFolder22 = tempfolder + "/" + "StepDisplacement.chk";
        dftUtils::writeDataIntoFile(fileDispData, newFolder22);
        if (time > 1)
          pcout << "#RESTART NOTE: Step Displacement:-"
                << " Step Displacements of TimeStep: " << time
                << " present in file StepDisplacement.chk" << std::endl
                << " Step Displacements of TimeStep: " << time - 1
                << " present in file StepDisplacement.chk.old #" << std::endl;
        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);


        // std::string newFolder3 = tempfolder + "/" + "time.chk";
        dftUtils::writeDataIntoFile(
          timeIndexData, newFolder3); // old time == new time then restart files
                                      // were successfully saved
        pcout << "#RESTART NOTE: restart files for TimeStep: " << time
              << " successfully created #" << std::endl;
        std::string newFolder0 =
          tempfolder + "/" + "UnwrappedFractionalCoordinates.chk";
        dftUtils::writeDataIntoFile(d_atomFractionalunwrapped, newFolder0);
      }
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
    molecularDynamicsClass<FEOrder, FEOrderElectro>::InitialiseFromRestartFile(
      std::vector<dealii::Tensor<1, 3, double>> &disp,
      std::vector<double> &                      velocity,
      std::vector<double> &                      force,
      std::vector<double> &                      KE,
      std::vector<double> &                      IE,
      std::vector<double> &                      TE)
  {
    // Initialise Position
    if (dftParameters::verbosity >= 1)
      {
        std::vector<std::vector<double>> atomLocations;
        atomLocations = dftPtr->getAtomLocations();
        pcout << "Atom Locations from Restart " << std::endl;
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
          {
            pcout << "Charge Id: " << iCharge << " "
                  << atomLocations[iCharge][2] << " "
                  << atomLocations[iCharge][3] << " "
                  << atomLocations[iCharge][4] << std::endl;
          }
      }
    std::string Folder = "mdRestart";

    std::vector<std::vector<double>> t1, KE0, IE0, TE0;
    std::string                      tempfolder =
      Folder + "/Step" + std::to_string(d_startingTimeStep);
    std::string newFolder0 =
      tempfolder + "/" + "UnwrappedFractionalCoordinates.chk";
    dftUtils::readFile(5, d_atomFractionalunwrapped, newFolder0);
    std::string                      fileName1  = "velocity.chk";
    std::string                      newFolder1 = tempfolder + "/" + fileName1;
    std::vector<std::vector<double>> fileVelData;
    dftUtils::readFile(3, fileVelData, newFolder1);
    for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
      {
        velocity[3 * iCharge + 0] = fileVelData[iCharge][0];
        velocity[3 * iCharge + 1] = fileVelData[iCharge][1];
        velocity[3 * iCharge + 2] = fileVelData[iCharge][2];
      }

    std::string                      fileName2  = "StepDisplacement.chk";
    std::string                      newFolder2 = tempfolder + "/" + fileName2;
    std::vector<std::vector<double>> fileDispData;
    dftUtils::readFile(3, fileDispData, newFolder2);
    for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
      {
        disp[iCharge][0] = fileDispData[iCharge][0];
        disp[iCharge][1] = fileDispData[iCharge][1];
        disp[iCharge][2] = fileDispData[iCharge][2];
      }


    std::string newFolder4 = tempfolder + "/" + "KineticEnergy.chk";
    dftUtils::readFile(1, KE0, newFolder4);
    std::string newFolder5 = tempfolder + "/" + "KineticEnergy.chk";
    dftUtils::readFile(1, IE0, newFolder5);
    std::string newFolder6 = tempfolder + "/" + "KineticEnergy.chk";
    dftUtils::readFile(1, TE0, newFolder6);
    KE[0] = KE0[0][0];
    IE[0] = IE0[0][0];
    TE[0] = TE0[0][0];


    dftPtr->solve(true, false, false, false);
    force = dftPtr->getForceonAtoms();

    if (Utilities::MPI::this_mpi_process(d_mpi_communicator) == 0 &&
        Utilities::MPI::this_mpi_process(d_interpoolcomm) == 0 &&
        Utilities::MPI::this_mpi_process(d_interBandGroupComm) == 0)
      {
        std::string oldFolder1 = "./mdRestart/Step";
        oldFolder1 = oldFolder1 + std::to_string(d_startingTimeStep) +
                     "/TotalDisplacement.chk";
        std::string oldFolder2 = "./mdRestart/Step";
        oldFolder2 =
          oldFolder2 + std::to_string(d_startingTimeStep) + "/Displacement.chk";

        dftUtils::copyFile(oldFolder1, ".");
        dftUtils::copyFile(oldFolder2, ".");
      }
    MPI_Barrier(d_mpi_communicator);
    MPI_Barrier(d_interBandGroupComm);
    MPI_Barrier(d_interpoolcomm);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  molecularDynamicsClass<FEOrder, FEOrderElectro>::InitialiseFromRestartNHCFile(
    std::vector<double> &v_e,
    std::vector<double> &e,
    std::vector<double> &Q)

  {
    if (dftParameters::reproducible_output == false)
      {
        std::vector<std::vector<double>> NHCData;
        std::string                      tempfolder = "mdRestart";
        if (Utilities::MPI::this_mpi_process(d_mpi_communicator) == 0 &&
            Utilities::MPI::this_mpi_process(d_interpoolcomm) == 0 &&
            Utilities::MPI::this_mpi_process(d_interBandGroupComm) == 0)
          {
            std::string oldFolder1 = "./mdRestart/Step";
            oldFolder1 = oldFolder1 + std::to_string(d_startingTimeStep) +
                         "/NHCThermostat.chk";
            dftUtils::copyFile(oldFolder1, "./mdRestart/.");
          }
        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
        std::string fileName  = "NHCThermostat.chk";
        std::string newFolder = tempfolder + "/" + fileName;
        dftUtils::readFile(3, NHCData, newFolder);
        Q[0]   = NHCData[0][0];
        Q[1]   = NHCData[1][0];
        e[0]   = NHCData[0][1];
        e[1]   = NHCData[1][1];
        v_e[0] = NHCData[0][2];
        v_e[1] = NHCData[1][2];
        pcout
          << "Nose Hover Chains Thermostat configuration read from Restart file "
          << std::endl;
      }
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  molecularDynamicsClass<FEOrder, FEOrderElectro>::writeRestartNHCfile(
    const std::vector<double> &v_e,
    const std::vector<double> &e,
    const std::vector<double> &Q,
    int                        time)

  {
    if (dftParameters::reproducible_output == false)
      {
        std::vector<std::vector<double>> fileNHCData(2,
                                                     std::vector<double>(3,
                                                                         0.0));
        fileNHCData[0][0]      = Q[0];
        fileNHCData[0][1]      = e[0];
        fileNHCData[0][2]      = v_e[0];
        fileNHCData[1][0]      = Q[1];
        fileNHCData[1][1]      = e[1];
        fileNHCData[1][2]      = v_e[1];
        std::string tempfolder = "./mdRestart";
        std::string newFolder =
          std::string(tempfolder + "/" + "NHCThermostat.chk");
        dftUtils::writeDataIntoFile(fileNHCData, newFolder);
        if (Utilities::MPI::this_mpi_process(d_mpi_communicator) == 0 &&
            Utilities::MPI::this_mpi_process(d_interpoolcomm) == 0 &&
            Utilities::MPI::this_mpi_process(d_interBandGroupComm) == 0)
          {
            std::string oldpath = newFolder;
            std::string newpath = "./mdRestart/Step";
            newpath             = newpath + std::to_string(time) + "/.";
            dftUtils::copyFile(oldpath, newpath);
          }
        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
      }
  }
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  molecularDynamicsClass<FEOrder, FEOrderElectro>::writeTotalDisplacementFile(
    const std::vector<dealii::Tensor<1, 3, double>> &r,
    int                                              time)
  {
    if (dftParameters::reproducible_output == false)
      {
        std::vector<std::vector<double>> fileDisplacementData;
        dftUtils::readFile(3, fileDisplacementData, "Displacement.chk");
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
          {
            fileDisplacementData[iCharge][0] =
              fileDisplacementData[iCharge][0] + r[iCharge][0];
            fileDisplacementData[iCharge][1] =
              fileDisplacementData[iCharge][1] + r[iCharge][1];
            fileDisplacementData[iCharge][2] =
              fileDisplacementData[iCharge][2] + r[iCharge][2];
          }
        dftUtils::writeDataIntoFile(fileDisplacementData, "Displacement.chk");

        if (Utilities::MPI::this_mpi_process(d_mpi_communicator) == 0 &&
            Utilities::MPI::this_mpi_process(d_interpoolcomm) == 0 &&
            Utilities::MPI::this_mpi_process(d_interBandGroupComm) == 0)
          {
            std::ofstream outfile;
            outfile.open("TotalDisplacement.chk", std::ios_base::app);
            std::vector<std::vector<double>> atomLocations;
            atomLocations = dftPtr->getAtomLocations();
            for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
              {
                outfile << atomLocations[iCharge][0] << "  "
                        << atomLocations[iCharge][1] << std::setprecision(16)
                        << "  " << fileDisplacementData[iCharge][0] << "  "
                        << fileDisplacementData[iCharge][1] << "  "
                        << fileDisplacementData[iCharge][2] << std::endl;
                /* temp[0][0] = atomLocations[iCharge][0];
                temp[0][1] = atomLocations[iCharge][1];
                temp[0][2] = fileDisplacementData[iCharge][0];
                temp[0][3] = fileDisplacementData[iCharge][1];
                temp[0][4] = fileDisplacementData[iCharge][2];*/
              }
            outfile.close();
          }
        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
        if (Utilities::MPI::this_mpi_process(d_mpi_communicator) == 0 &&
            Utilities::MPI::this_mpi_process(d_interpoolcomm) == 0 &&
            Utilities::MPI::this_mpi_process(d_interBandGroupComm) == 0)
          {
            std::string oldpath = "TotalDisplacement.chk";
            std::string newpath = "./mdRestart/Step";
            newpath             = newpath + std::to_string(time) + "/.";
            dftUtils::copyFile(oldpath, newpath);
            std::string oldpath2 = "Displacement.chk";
            std::string newpath2 = "./mdRestart/Step";
            newpath2             = newpath2 + std::to_string(time) + "/.";
            dftUtils::copyFile(oldpath2, newpath2);
          }
        MPI_Barrier(d_mpi_communicator);
        MPI_Barrier(d_interBandGroupComm);
        MPI_Barrier(d_interpoolcomm);
      }
  }
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  molecularDynamicsClass<FEOrder, FEOrderElectro>::NoseHoverExtendedLagrangian(
    const std::vector<double> &thermovelocity,
    const std::vector<double> &thermoposition,
    const std::vector<double> &thermomass,
    double                     PE,
    double                     KE,
    double                     T)
  {
    double Hnose = 0.0;
    Hnose = (0.5 * thermomass[0] * thermovelocity[0] * thermovelocity[0] +
             0.5 * thermomass[1] * thermovelocity[1] * thermovelocity[1] +
             3 * (d_numberGlobalCharges - 1) * T * kB * thermoposition[0] +
             kB * T * thermoposition[1]) /
              haToeV +
            KE + PE;
    return (Hnose);
  }
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  int
  molecularDynamicsClass<FEOrder, FEOrderElectro>::checkRestart()
  {
    int time1 = 0;

    if (dftfe::dftParameters::restartMdFromChk)
      {
        std::vector<std::vector<double>> t1;
        pcout << " MD is in Restart Mode" << std::endl;

        dftfe::dftUtils::readFile(1, t1, "mdRestart/time.chk");
        time1                  = t1[0][0];
        std::string tempfolder = "mdRestart/Step";
        bool        flag       = false;
        std::string path2      = tempfolder + std::to_string(time1);
        pcout << "Looking for files of TimeStep " << time1 << " at: " << path2
              << std::endl;
        while (!flag && time1 > 1)
          {
            std::string   path  = tempfolder + std::to_string(time1);
            std::string   file1 = path + "/atomsFracCoordCurrent.chk";
            std::string   file2 = path + "/velocity.chk";
            std::string   file3 = path + "/NHCThermostat.chk";
            std::ifstream readFile1(file1.c_str());
            std::ifstream readFile2(file2.c_str());
            std::ifstream readFile3(file3.c_str());
            pcout << " Restart folders:"
                  << (!readFile1.fail() && !readFile2.fail()) << std::endl;
            bool NHCflag = true;
            if (dftfe::dftParameters::tempControllerTypeBOMD ==
                "NOSE_HOVER_CHAINS")
              {
                NHCflag = false;
                if (!readFile3.fail())
                  NHCflag = true;
              }
            if (!readFile1.fail() && !readFile2.fail() && NHCflag)
              {
                flag                                  = true;
                dftfe::dftParameters::coordinatesFile = file1;
                pcout << " Restart files are found in: " << path << std::endl;
              }

            else
              pcout << "----Error opening restart files present in: " << path
                    << std::endl
                    << "Switching to time: " << --time1 << " ----" << std::endl;
          }
      }
    return (time1);
  }



#include "mdClass.inst.cc"
} // namespace dftfe
