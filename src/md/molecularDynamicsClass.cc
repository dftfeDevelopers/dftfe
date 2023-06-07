// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
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
// @author Kartick Ramakrishnan
//



#include <boost/generator_iterator.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <headers.h>
#include <dftUtils.h>
#include <fileReaders.h>
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
  molecularDynamicsClass::molecularDynamicsClass(
    const std::string parameter_file,
    const std::string restartFilesPath,
    const MPI_Comm &  mpi_comm_parent,
    const bool        restart,
    const int         verbosity)
    : d_mpiCommParent(mpi_comm_parent)
    , d_this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , d_restartFilesPath(restartFilesPath)
    , d_verbosity(verbosity)
  {
    MPI_Barrier(d_mpiCommParent);
    d_MDstartWallTime = MPI_Wtime();
    d_TimeIndex       = 0;
    d_restartFlag     = restart ? 1 : 0;
    if (d_restartFlag == 0)
      {
        d_startingTimeStep = 0;
        if (d_this_mpi_process == 0)
          if (d_restartFilesPath != ".")
            {
              mkdir(d_restartFilesPath.c_str(), ACCESSPERMS);
            }
        d_dftfeWrapper =
          std::make_unique<dftfe::dftfeWrapper>(parameter_file,
                                                d_mpiCommParent,
                                                true,
                                                true,
                                                "MD",
                                                d_restartFilesPath,
                                                d_verbosity);
      }
    else
      {
        std::string coordinatesFile, domainVectorsFile;
        bool        scfRestart;
        d_startingTimeStep =
          checkRestart(coordinatesFile, domainVectorsFile, scfRestart);
        pcout << "scfRestartFlag: " << scfRestart << std::endl;
        d_dftfeWrapper =
          std::make_unique<dftfe::dftfeWrapper>(parameter_file,
                                                coordinatesFile,
                                                domainVectorsFile,
                                                d_mpiCommParent,
                                                true,
                                                true,
                                                "MD",
                                                d_restartFilesPath,
                                                d_verbosity,
                                                scfRestart);
      }

    d_restartFilesPath = d_restartFilesPath + "/mdRestart";
    set();
  }



  void
  molecularDynamicsClass::set()
  {
    d_dftPtr = d_dftfeWrapper->getDftfeBasePtr();
    d_TimeStep =
      d_dftPtr->getParametersObject().timeStepBOMD *
      0.09822694541304546435; // Conversion factor from femteseconds:
                              // 0.09822694541304546435 based on NIST constants
    d_numberofSteps       = d_dftPtr->getParametersObject().numberStepsBOMD;
    d_startingTemperature = d_dftPtr->getParametersObject().startingTempBOMD;
    d_ThermostatTimeConstant =
      d_dftPtr->getParametersObject().thermostatTimeConstantBOMD;
    d_ThermostatType = d_dftPtr->getParametersObject().tempControllerTypeBOMD;
    d_numberGlobalCharges = d_dftPtr->getParametersObject().natoms;
    d_MaxWallTime         = d_dftPtr->getParametersObject().MaxWallTime;
    pcout
      << "----------------------Starting Initialization of BOMD-------------------------"
      << std::endl;
    pcout << "Starting Temperature from Input " << d_startingTemperature
          << std::endl;
    std::vector<std::vector<double>> temp_domainBoundingVectors;
    dftUtils::readFile(
      3,
      temp_domainBoundingVectors,
      d_dftPtr->getParametersObject().domainBoundingVectorsFile);

    for (int i = 0; i < 3; i++)
      {
        double temp =
          temp_domainBoundingVectors[i][0] * temp_domainBoundingVectors[i][0] +
          temp_domainBoundingVectors[i][1] * temp_domainBoundingVectors[i][1] +
          temp_domainBoundingVectors[i][2] * temp_domainBoundingVectors[i][2];
        d_domainLength.push_back(pow(temp, 0.5));
      }
    if (d_dftPtr->getParametersObject().verbosity > 1)
      {
        pcout << "--$ Domain Length$ --" << std::endl;
        pcout << "Lx:= " << d_domainLength[0] << " Ly:=" << d_domainLength[1]
              << " Lz:=" << d_domainLength[2] << std::endl;
      }
  }



  int
  molecularDynamicsClass::runMD()
  {
    std::vector<double> massAtoms(d_numberGlobalCharges);
    //
    // read atomic masses in amu
    //
    std::vector<std::vector<double>> atomTypesMasses;
    dftUtils::readFile(2,
                       atomTypesMasses,
                       d_dftPtr->getParametersObject().atomicMassesFile);
    std::vector<std::vector<double>> atomLocations;
    atomLocations = d_dftPtr->getAtomLocationsCart();
    std::set<unsigned int> atomTypes;
    atomTypes = d_dftPtr->getAtomTypes();
    AssertThrow(atomTypes.size() == atomTypesMasses.size(),
                dealii::ExcMessage("DFT-FE Error: check ATOM MASSES FILE"));

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
                    dealii::ExcMessage("DFT-FE Error: check ATOM MASSES FILE"));
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
    // d_restartFlag = d_dftPtr->getParametersObject().restartMdFromChk ? 1 : 0;
    pcout << "RestartFlag: " << d_restartFlag << std::endl;
    if (d_restartFlag == 0)
      {
        if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
          mkdir((d_restartFilesPath).c_str(), ACCESSPERMS);

        double KineticEnergy = 0.0, TemperatureFromVelocities = 0.0,
               GroundStateEnergyvalue = 0.0, EntropicEnergyvalue = 0.0;

        dftUtils::readFile(5,
                           d_atomFractionalunwrapped,
                           d_dftPtr->getParametersObject().coordinatesFile);
        std::vector<std::vector<double>> fileDisplacementData;
        std::vector<double>              initDisp(0.0, 3);
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
          {
            fileDisplacementData.push_back(initDisp);
          }
        std::string Folder = d_restartFilesPath + "/Step0";
        mkdir(Folder.c_str(), ACCESSPERMS);
        dftUtils::writeDataIntoFile(fileDisplacementData,
                                    Folder + "/Displacement.chk",
                                    d_mpiCommParent);
        dftUtils::writeDataIntoFile(fileDisplacementData,
                                    Folder + "/TotalDisplacement.chk",
                                    d_mpiCommParent);
        //--------------------Starting Initialization
        //----------------------------------------------//

        double Px = 0.0, Py = 0.0, Pz = 0.0;
        // Initialise Velocity
        if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
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
              dealii::Utilities::MPI::sum(velocity[i], d_mpiCommParent);
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


        d_dftPtr->solve(true,
                        false,
                        d_dftPtr->getParametersObject().loadRhoData);
        force = d_dftPtr->getForceonAtoms();
        if (d_dftPtr->getParametersObject().extrapolateDensity == 1 &&
            d_dftPtr->getParametersObject().spinPolarized != 1)
          DensityExtrapolation(0);
        else if (d_dftPtr->getParametersObject().extrapolateDensity == 2 &&
                 d_dftPtr->getParametersObject().spinPolarized != 1)
          DensitySplitExtrapolation(0);
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
        MPI_Barrier(d_mpiCommParent);
        GroundStateEnergyvalue  = d_dftPtr->getInternalEnergy();
        EntropicEnergyvalue     = d_dftPtr->getEntropicEnergy();
        KineticEnergyVector[0]  = KineticEnergy / haToeV;
        InternalEnergyVector[0] = GroundStateEnergyvalue;
        EntropicEnergyVector[0] = EntropicEnergyvalue;
        TotalEnergyVector[0]    = KineticEnergyVector[0] +
                               InternalEnergyVector[0] -
                               EntropicEnergyVector[0];
        if (d_dftPtr->getParametersObject().verbosity >= 1)
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

        if (d_dftPtr->getParametersObject().verbosity >= 0 &&
            !d_dftPtr->getParametersObject().reproducible_output)
          {
            pcout << "---------------MD STEP" << d_startingTimeStep
                  << "------------------ " << std::endl;
            pcout << " Temperature from velocities: "
                  << TemperatureFromVelocities << std::endl;
            pcout << " Kinetic Energy in Ha at timeIndex  "
                  << d_startingTimeStep << KineticEnergyVector[0] << std::endl;
            pcout << " Internal Energy in Ha at timeIndex  "
                  << d_startingTimeStep << InternalEnergyVector[0] << std::endl;
            pcout << " Entropic Energy in Ha at timeIndex  "
                  << d_startingTimeStep << EntropicEnergyVector[0] << std::endl;
            pcout << " Total Energy in Ha at timeIndex " << TotalEnergyVector[0]
                  << std::endl;
          }
        else if (d_dftPtr->getParametersObject().verbosity >= 0 &&
                 d_dftPtr->getParametersObject().reproducible_output)
          {
            pcout << "---------------MD STEP " << d_startingTimeStep
                  << " ------------------ " << std::endl;
            pcout << " Temperature from velocities: " << std::setprecision(2)
                  << TemperatureFromVelocities << std::endl;
            pcout << " Total Energy in Ha at timeIndex " << std::setprecision(5)
                  << TotalEnergyVector[0] << std::endl;
          }

        MPI_Barrier(d_mpiCommParent);

        //--------------------Completed Initialization
        //----------------------------------------------//
      }

    else if (d_restartFlag == 1)
      {
        /*if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
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

          }

        MPI_Barrier(d_mpiCommParent);*/
        InitialiseFromRestartFile(displacements,
                                  velocity,
                                  force,
                                  KineticEnergyVector,
                                  InternalEnergyVector,
                                  TotalEnergyVector);
        if (d_dftPtr->getParametersObject().verbosity > 1 &&
            !d_dftPtr->getParametersObject().reproducible_output)
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
    int status;
    if (d_ThermostatType == "NO_CONTROL")
      {
        status = mdNVE(KineticEnergyVector,
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
        status = mdNVTrescaleThermostat(KineticEnergyVector,
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
        status = mdNVTnosehoverchainsThermostat(KineticEnergyVector,
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
        status = mdNVTsvrThermostat(KineticEnergyVector,
                                    InternalEnergyVector,
                                    EntropicEnergyVector,
                                    TotalEnergyVector,
                                    displacements,
                                    velocity,
                                    force,
                                    massAtoms);
      }

    if (status == 0)
      pcout << "---MD run completed successfully---" << std::endl;
    else if (status == 1)
      pcout << "---MD run exited: Wall Time Exceeded---" << std::endl;
    return (status);
  }


  int
  molecularDynamicsClass::mdNVE(
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
        MPI_Barrier(d_mpiCommParent);
        step_time = MPI_Wtime();
        KineticEnergy =
          velocityVerlet(velocity, displacements, atomMass, force);
        GroundStateEnergyvalue = d_dftPtr->getInternalEnergy();
        EntropicEnergyvalue    = d_dftPtr->getEntropicEnergy();
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
        MPI_Barrier(d_mpiCommParent);
        step_time = MPI_Wtime() - step_time;
        if (d_dftPtr->getParametersObject().verbosity >= 1)
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
        if (d_dftPtr->getParametersObject().verbosity >= 0 &&
            !d_dftPtr->getParametersObject().reproducible_output)
          {
            pcout << "---------------MD STEP: " << d_TimeIndex
                  << " ------------------ " << std::endl;
            pcout << "Time taken for md step: " << step_time << std::endl;
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
        else if (d_dftPtr->getParametersObject().verbosity >= 0 &&
                 d_dftPtr->getParametersObject().reproducible_output)
          {
            pcout << "---------------MD STEP " << d_TimeIndex
                  << " ------------------ " << std::endl;
            pcout << " Temperature from velocities: " << std::setprecision(2)
                  << TemperatureFromVelocities << std::endl;
            pcout << " Total Energy in Ha at timeIndex " << std::setprecision(5)
                  << TotalEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
          }
        writeRestartFile(displacements,
                         velocity,
                         force,
                         KineticEnergyVector,
                         InternalEnergyVector,
                         TotalEnergyVector,
                         d_TimeIndex);
        writeTotalDisplacementFile(displacements, d_TimeIndex);

        MPI_Barrier(d_mpiCommParent);
        curr_time = MPI_Wtime() - d_MDstartWallTime;
        if (!d_dftPtr->getParametersObject().reproducible_output)
          pcout << "*****Time Completed till NOW: " << curr_time << std::endl;
        if (d_MaxWallTime - (curr_time + 1.05 * step_time) < 0)
          {
            return (1);
          }
      }
    return (0);
  }


  int
  molecularDynamicsClass::mdNVTrescaleThermostat(
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
        MPI_Barrier(d_mpiCommParent);
        step_time = MPI_Wtime();


        KineticEnergy =
          velocityVerlet(velocity, displacements, atomMass, force);
        TemperatureFromVelocities =
          2.0 / 3.0 / double(d_numberGlobalCharges - 1) * KineticEnergy / (kB);
        if (d_TimeIndex % d_ThermostatTimeConstant == 0)
          {
            KineticEnergy =
              RescaleVelocities(velocity, atomMass, TemperatureFromVelocities);
          }

        MPI_Barrier(d_mpiCommParent);
        GroundStateEnergyvalue = d_dftPtr->getInternalEnergy();
        EntropicEnergyvalue    = d_dftPtr->getEntropicEnergy();
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
        MPI_Barrier(d_mpiCommParent);
        step_time = MPI_Wtime() - step_time;
        if (d_dftPtr->getParametersObject().verbosity >= 1)
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

        if (d_dftPtr->getParametersObject().verbosity >= 0 &&
            !d_dftPtr->getParametersObject().reproducible_output)
          {
            pcout << "---------------MD STEP: " << d_TimeIndex
                  << " ------------------ " << std::endl;
            pcout << "Time taken for md step: " << step_time << std::endl;
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
        else if (d_dftPtr->getParametersObject().verbosity >= 0 &&
                 d_dftPtr->getParametersObject().reproducible_output)
          {
            pcout << "---------------MD STEP " << d_TimeIndex
                  << " ------------------ " << std::endl;
            pcout << " Temperature from velocities: " << std::setprecision(2)
                  << TemperatureFromVelocities << std::endl;
            pcout << " Total Energy in Ha at timeIndex " << std::setprecision(5)
                  << TotalEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
          }
        writeRestartFile(displacements,
                         velocity,
                         force,
                         KineticEnergyVector,
                         InternalEnergyVector,
                         TotalEnergyVector,
                         d_TimeIndex);
        writeTotalDisplacementFile(displacements, d_TimeIndex);

        MPI_Barrier(d_mpiCommParent);
        curr_time = MPI_Wtime() - d_MDstartWallTime;
        if (!d_dftPtr->getParametersObject().reproducible_output)
          pcout << "*****Time Completed till NOW: " << curr_time << std::endl;
        if (d_MaxWallTime - (curr_time + 1.05 * step_time) < 0)
          {
            return (1);
          }
      }
    return (0);
  }

  int
  molecularDynamicsClass::mdNVTnosehoverchainsThermostat(
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
        MPI_Barrier(d_mpiCommParent);
        step_time = MPI_Wtime();
        NoseHoverChains(
          velocity,
          Thermostatvelocity,
          Thermostatposition,
          ThermostatMass,
          KineticEnergyVector[d_TimeIndex - 1 - d_startingTimeStep] * haToeV,
          d_startingTemperature);

        KineticEnergy =
          velocityVerlet(velocity, displacements, atomMass, force);

        MPI_Barrier(d_mpiCommParent);

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


        GroundStateEnergyvalue = d_dftPtr->getInternalEnergy();
        EntropicEnergyvalue    = d_dftPtr->getEntropicEnergy();
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

        if (d_dftPtr->getParametersObject().verbosity >= 1)
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


        MPI_Barrier(d_mpiCommParent);
        step_time = MPI_Wtime() - step_time;


        if (d_dftPtr->getParametersObject().verbosity >= 0 &&
            !d_dftPtr->getParametersObject().reproducible_output)
          {
            pcout << "---------------MD STEP: " << d_TimeIndex
                  << " ------------------ " << std::endl;
            pcout << "Time taken for md step: " << step_time << std::endl;
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
        else if (d_dftPtr->getParametersObject().verbosity >= 0 &&
                 d_dftPtr->getParametersObject().reproducible_output)
          {
            pcout << "---------------MD STEP " << d_TimeIndex
                  << " ------------------ " << std::endl;
            pcout << " Temperature from velocities: " << std::setprecision(2)
                  << TemperatureFromVelocities << std::endl;
            pcout << " Total Energy in Ha at timeIndex " << std::setprecision(5)
                  << TotalEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
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

        MPI_Barrier(d_mpiCommParent);
        curr_time = MPI_Wtime() - d_MDstartWallTime;
        //  pcout<<"*****Time Completed till NOW: "<<curr_time<<std::endl;
        if (d_MaxWallTime - (curr_time + 1.05 * step_time) < 0)
          {
            return (1);
          }
      }
    return (0);
  }


  int
  molecularDynamicsClass::mdNVTsvrThermostat(
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

        MPI_Barrier(d_mpiCommParent);
        step_time = MPI_Wtime();


        KineticEnergy =
          velocityVerlet(velocity, displacements, atomMass, force);


        MPI_Barrier(d_mpiCommParent);

        KineticEnergy = svr(velocity, KineticEnergy, KEref);
        TemperatureFromVelocities =
          2.0 / 3.0 / double(d_numberGlobalCharges - 1) * KineticEnergy / (kB);
        GroundStateEnergyvalue = d_dftPtr->getInternalEnergy();
        EntropicEnergyvalue    = d_dftPtr->getEntropicEnergy();
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
        MPI_Barrier(d_mpiCommParent);
        step_time = MPI_Wtime() - step_time;
        if (d_dftPtr->getParametersObject().verbosity >= 1)
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
        if (d_dftPtr->getParametersObject().verbosity >= 0 &&
            !d_dftPtr->getParametersObject().reproducible_output)
          {
            pcout << "---------------MD STEP: " << d_TimeIndex
                  << " ------------------ " << std::endl;
            pcout << "Time taken for md step: " << step_time << std::endl;
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
        else if (d_dftPtr->getParametersObject().verbosity >= 0 &&
                 d_dftPtr->getParametersObject().reproducible_output)
          {
            pcout << "---------------MD STEP " << d_TimeIndex
                  << " ------------------ " << std::endl;
            pcout << " Temperature from velocities: " << std::setprecision(2)
                  << TemperatureFromVelocities << std::endl;
            pcout << " Total Energy in Ha at timeIndex " << std::setprecision(5)
                  << TotalEnergyVector[d_TimeIndex - d_startingTimeStep]
                  << std::endl;
          }
        writeRestartFile(displacements,
                         velocity,
                         force,
                         KineticEnergyVector,
                         InternalEnergyVector,
                         TotalEnergyVector,
                         d_TimeIndex);
        writeTotalDisplacementFile(displacements, d_TimeIndex);

        MPI_Barrier(d_mpiCommParent);
        curr_time = MPI_Wtime() - d_MDstartWallTime;
        if (!d_dftPtr->getParametersObject().reproducible_output)
          pcout << "*****Time Completed till NOW: " << curr_time << std::endl;
        if (d_MaxWallTime - (curr_time + 1.05 * step_time) < 0)
          {
            return (1);
          }
      }
    return (0);
  }



  double
  molecularDynamicsClass::velocityVerlet(
    std::vector<double> &                      v,
    std::vector<dealii::Tensor<1, 3, double>> &r,
    const std::vector<double> &                atomMass,
    std::vector<double> &                      forceOnAtoms)
  {
    int                 i;
    double              totalKE;
    double              KE   = 0.0;
    double              dt   = d_TimeStep;
    double              dt_2 = dt / 2;
    double              COMM = 0.0;
    double              COMx = 0.0;
    double              COMy = 0.0;
    double              COMz = 0.0;
    std::vector<double> rloc(3 * d_numberGlobalCharges, 0.0);
    if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
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
      &(v[0]), 3 * d_numberGlobalCharges, MPI_DOUBLE, 0, d_mpiCommParent);
    if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
      {
        for (i = 0; i < d_numberGlobalCharges; i++)
          {
            rloc[3 * i + 0] -= COMx;
            rloc[3 * i + 1] -= COMy;
            rloc[3 * i + 2] -= COMz;
          }
      }

    MPI_Bcast(
      &(rloc[0]), 3 * d_numberGlobalCharges, MPI_DOUBLE, 0, d_mpiCommParent);

    for (unsigned int i = 0; i < d_numberGlobalCharges; ++i)
      {
        for (unsigned int j = 0; j < 3; ++j)
          {
            r[i][j] = rloc[i * 3 + j];
          }
      }

    double update_time;

    update_time = MPI_Wtime();

    d_dftPtr->updateAtomPositionsAndMoveMesh(
      r, d_dftPtr->getParametersObject().maxJacobianRatioFactorForMD, false);

    if (d_dftPtr->getParametersObject().verbosity >= 1)
      {
        std::vector<std::vector<double>> atomLocations;
        atomLocations = d_dftPtr->getAtomLocationsCart();
        pcout << "Displacement  " << std::endl;
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
          {
            if (atomLocations[iCharge][0] ==
                d_dftPtr->getParametersObject().MDTrack)
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
    if (d_dftPtr->getParametersObject().verbosity > 1 &&
        !d_dftPtr->getParametersObject().reproducible_output)
      {
        pcout << "---- Updated Unwrapped Coordinates: -----" << std::endl;
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
          {
            if (d_atomFractionalunwrapped[iCharge][0] ==
                d_dftPtr->getParametersObject().MDTrack)
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



    MPI_Barrier(d_mpiCommParent);

    update_time = MPI_Wtime() - update_time;

    if (d_dftPtr->getParametersObject().verbosity >= 1)
      pcout << "Time taken for updateAtomPositionsAndMoveMesh: " << update_time
            << std::endl;
    d_dftPtr->solve(true, false);
    forceOnAtoms = d_dftPtr->getForceonAtoms();
    if (d_dftPtr->getParametersObject().extrapolateDensity == 1 &&
        d_dftPtr->getParametersObject().spinPolarized != 1)
      DensityExtrapolation(d_TimeIndex - d_startingTimeStep);
    else if (d_dftPtr->getParametersObject().extrapolateDensity == 2 &&
             d_dftPtr->getParametersObject().spinPolarized != 1)
      DensitySplitExtrapolation(d_TimeIndex - d_startingTimeStep);
    // Call Force
    totalKE = 0.0;
    /* Second half of velocty verlet */
    if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
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
      &(v[0]), 3 * d_numberGlobalCharges, MPI_DOUBLE, 0, d_mpiCommParent);
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



  double
  molecularDynamicsClass::RescaleVelocities(std::vector<double> &      v,
                                            const std::vector<double> &M,
                                            double Temperature)
  {
    pcout << "Rescale Thermostat: Before rescaling temperature= " << Temperature
          << " K" << std::endl;
    AssertThrow(
      std::fabs(Temperature - 0.0) > 0.00001,
      dealii::ExcMessage(
        "DFT-FE Error: Temperature reached O K")); // Determine Exit sequence ..
    double KE = 0.0;
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


  void
  molecularDynamicsClass::NoseHoverChains(std::vector<double> &v,
                                          std::vector<double> &v_e,
                                          std::vector<double> &e,
                                          std::vector<double>  Q,
                                          double               KE,
                                          double               Temperature)
  {
    double G1, G2, s;
    double L = 3 * (d_numberGlobalCharges - 1);
    /* Start Chain 1*/
    G2     = (Q[0] * v_e[0] * v_e[0] - kB * Temperature) / Q[1];
    v_e[1] = v_e[1] + G2 * d_TimeStep / 4;
    v_e[0] = v_e[0] * std::exp(-v_e[1] * d_TimeStep / 8);
    G1     = (2 * KE - L * kB * Temperature) / Q[0];
    v_e[0] = v_e[0] + G1 * d_TimeStep / 4;
    v_e[0] = v_e[0] * std::exp(-v_e[1] * d_TimeStep / 8);
    e[0]   = e[0] + v_e[0] * d_TimeStep / 2;
    e[1]   = e[1] + v_e[1] * d_TimeStep / 2;
    s      = std::exp(-v_e[0] * d_TimeStep / 2);

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



  double
  molecularDynamicsClass::svr(std::vector<double> &v, double KE, double KEref)
  {
    double       alphasq;
    unsigned int Nf = 3 * (d_numberGlobalCharges - 1);
    double       R1, Rsum;
    R1   = 0.0;
    Rsum = 0.0;
    if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
      {
        std::time_t            now = std::time(0);
        boost::random::mt19937 gen{
          d_dftPtr->getParametersObject().reproducible_output ?
            0 :
            static_cast<std::uint32_t>(now)};
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
    R1      = dealii::Utilities::MPI::sum(R1, d_mpiCommParent);
    Rsum    = dealii::Utilities::MPI::sum(Rsum, d_mpiCommParent);
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



  void
  molecularDynamicsClass::writeRestartFile(
    const std::vector<dealii::Tensor<1, 3, double>> &disp,
    const std::vector<double> &                      velocity,
    const std::vector<double> &                      force,
    const std::vector<double> &                      KineticEnergyVector,
    const std::vector<double> &                      InternalEnergyVector,
    const std::vector<double> &                      TotalEnergyVector,
    int                                              time)

  {
    // Writes the restart files for velocities and positions
    std::vector<std::vector<double>> fileForceData(d_numberGlobalCharges,
                                                   std::vector<double>(3, 0.0));
    std::vector<std::vector<double>> fileDispData(d_numberGlobalCharges,
                                                  std::vector<double>(3, 0.0));
    std::vector<std::vector<double>> fileVelocityData(d_numberGlobalCharges,
                                                      std::vector<double>(3,
                                                                          0.0));
    std::vector<std::vector<double>> timeIndexData(1,
                                                   std::vector<double>(1, 0));
    std::vector<std::vector<double>> KEData(1, std::vector<double>(1, 0.0));
    std::vector<std::vector<double>> IEData(1, std::vector<double>(1, 0.0));
    std::vector<std::vector<double>> TEData(1, std::vector<double>(1, 0.0));

    std::vector<std::vector<double>> mdData(5, std::vector<double>(3, 0.0));
    if (d_ThermostatType == "NO_CONTROL")
      mdData[0][0] = 0.0;
    else if (d_ThermostatType == "RESCALE")
      mdData[0][0] = 1.0;
    else if (d_ThermostatType == "NOSE_HOVER_CHAINS")
      mdData[0][0] = 2.0;
    else if (d_ThermostatType == "CSVR")
      mdData[0][0] = 3.0;
    mdData[1][0]           = d_numberGlobalCharges;
    mdData[2][0]           = d_TimeStep;
    mdData[3][0]           = d_TimeIndex;
    mdData[4][0]           = d_dftPtr->getParametersObject().periodicX ? 1 : 0;
    mdData[4][1]           = d_dftPtr->getParametersObject().periodicY ? 1 : 0;
    mdData[4][2]           = d_dftPtr->getParametersObject().periodicZ ? 1 : 0;
    timeIndexData[0][0]    = double(time);
    std::string Folder     = d_restartFilesPath + "/Step";
    std::string tempfolder = Folder + std::to_string(time);
    if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
      mkdir(tempfolder.c_str(), ACCESSPERMS);
    Folder                  = d_restartFilesPath;
    std::string newFolder3  = Folder + "/" + "time.chk";
    std::string newFolder_0 = Folder + "/" + "moleculardynamics.dat";
    dftUtils::writeDataIntoFile(mdData, newFolder_0, d_mpiCommParent);
    dftUtils::writeDataIntoFile(timeIndexData, newFolder3, d_mpiCommParent);
    KEData[0][0]           = KineticEnergyVector[time - d_startingTimeStep];
    IEData[0][0]           = InternalEnergyVector[time - d_startingTimeStep];
    TEData[0][0]           = TotalEnergyVector[time - d_startingTimeStep];
    std::string newFolder4 = tempfolder + "/" + "KineticEnergy.chk";
    dftUtils::writeDataIntoFile(KEData, newFolder4, d_mpiCommParent);
    std::string newFolder5 = tempfolder + "/" + "InternalEnergy.chk";
    dftUtils::writeDataIntoFile(IEData, newFolder5, d_mpiCommParent);
    std::string newFolder6 = tempfolder + "/" + "TotalEnergy.chk";
    dftUtils::writeDataIntoFile(TEData, newFolder6, d_mpiCommParent);

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
    d_dftPtr->writeDomainAndAtomCoordinates(cordFolder);
    if (time > 1)
      {
        if (d_dftPtr->getParametersObject().reproducible_output == false)
          pcout << "#RESTART NOTE: Positions:-"
                << " Positions of TimeStep: " << time
                << " present in file atomsFracCoordCurrent.chk" << std::endl
                << " Positions of TimeStep: " << time - 1
                << " present in file atomsFracCoordCurrent.chk.old #"
                << std::endl;
        std::string newFolder1 = tempfolder + "/" + "velocity.chk";
        dftUtils::writeDataIntoFile(fileVelocityData,
                                    newFolder1,
                                    d_mpiCommParent);


        if (d_dftPtr->getParametersObject().reproducible_output == false)
          pcout << "#RESTART NOTE: Velocity:-"
                << " Velocity of TimeStep: " << time
                << " present in file velocity.chk" << std::endl
                << " Velocity of TimeStep: " << time - 1
                << " present in file velocity.chk.old #" << std::endl;
        std::string newFolder2 = tempfolder + "/" + "force.chk";
        dftUtils::writeDataIntoFile(fileForceData, newFolder2, d_mpiCommParent);
        if (d_dftPtr->getParametersObject().reproducible_output == false)
          pcout << "#RESTART NOTE: Force:-"
                << " Force of TimeStep: " << time
                << " present in file force.chk" << std::endl
                << " Forces of TimeStep: " << time - 1
                << " present in file force.chk.old #" << std::endl;
        std::string newFolder22 = tempfolder + "/" + "StepDisplacement.chk";
        dftUtils::writeDataIntoFile(fileDispData, newFolder22, d_mpiCommParent);
        if (d_dftPtr->getParametersObject().reproducible_output == false)
          pcout << "#RESTART NOTE: Step Displacement:-"
                << " Step Displacements of TimeStep: " << time
                << " present in file StepDisplacement.chk" << std::endl
                << " Step Displacements of TimeStep: " << time - 1
                << " present in file StepDisplacement.chk.old #" << std::endl;
        MPI_Barrier(d_mpiCommParent);
        if (d_dftPtr->getParametersObject().reproducible_output == false)
          pcout << "#RESTART NOTE: restart files for TimeStep: " << time
                << " successfully created #" << std::endl;
        std::string newFolder0 =
          tempfolder + "/" + "UnwrappedFractionalCoordinates.chk";
        dftUtils::writeDataIntoFile(d_atomFractionalunwrapped,
                                    newFolder0,
                                    d_mpiCommParent);

        // std::string newFolder3 = tempfolder + "/" + "time.chk";
        dftUtils::writeDataIntoFile(
          timeIndexData,
          newFolder3,
          d_mpiCommParent); // old time == new time then restart files
                            // were successfully saved
      }
  }



  void molecularDynamicsClass::InitialiseFromRestartFile(
    std::vector<dealii::Tensor<1, 3, double>> &disp,
    std::vector<double> &                      velocity,
    std::vector<double> &                      force,
    std::vector<double> &                      KE,
    std::vector<double> &                      IE,
    std::vector<double> &                      TE)
  {
    // Initialise Position
    if (d_dftPtr->getParametersObject().verbosity >= 1)
      {
        std::vector<std::vector<double>> atomLocations;
        atomLocations = d_dftPtr->getAtomLocationsCart();
        pcout << "Cartesian Atom Locations from Restart " << std::endl;
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; ++iCharge)
          {
            pcout << "Charge Id: " << iCharge << " "
                  << atomLocations[iCharge][2] << " "
                  << atomLocations[iCharge][3] << " "
                  << atomLocations[iCharge][4] << std::endl;
          }
      }
    std::string Folder = d_restartFilesPath;

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

    d_dftPtr->solve(true, false, d_dftPtr->getParametersObject().loadRhoData);
    force = d_dftPtr->getForceonAtoms();

    if (d_dftPtr->getParametersObject().extrapolateDensity == 1 &&
        d_dftPtr->getParametersObject().spinPolarized != 1)
      DensityExtrapolation(0);
    else if (d_dftPtr->getParametersObject().extrapolateDensity == 2 &&
             d_dftPtr->getParametersObject().spinPolarized != 1)
      DensitySplitExtrapolation(0);
    /*if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
      {
        std::string oldFolder1 = d_restartFilesPath + "/Step";
        oldFolder1 = oldFolder1 + std::to_string(d_startingTimeStep) +
                     "/TotalDisplacement.chk";
        std::string oldFolder2 = d_restartFilesPath + "/Step";
        oldFolder2 =
          oldFolder2 + std::to_string(d_startingTimeStep) + "/Displacement.chk";

        dftUtils::copyFile(oldFolder1, ".");
        dftUtils::copyFile(oldFolder2, ".");
      }
    MPI_Barrier(d_mpiCommParent); */
  }


  void
  molecularDynamicsClass::InitialiseFromRestartNHCFile(std::vector<double> &v_e,
                                                       std::vector<double> &e,
                                                       std::vector<double> &Q)

  {
    if (d_dftPtr->getParametersObject().reproducible_output == false)
      {
        std::vector<std::vector<double>> NHCData;
        std::string                      tempfolder = "mdRestart";
        if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
          {
            std::string oldFolder1 = "./mdRestart/Step";
            oldFolder1 = oldFolder1 + std::to_string(d_startingTimeStep) +
                         "/NHCThermostat.chk";
            dftUtils::copyFile(oldFolder1, "./mdRestart/.");
          }
        MPI_Barrier(d_mpiCommParent);
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



  void
  molecularDynamicsClass::writeRestartNHCfile(const std::vector<double> &v_e,
                                              const std::vector<double> &e,
                                              const std::vector<double> &Q,
                                              const int                  time)

  {
    if (d_dftPtr->getParametersObject().reproducible_output == false)
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
        dftUtils::writeDataIntoFile(fileNHCData, newFolder, d_mpiCommParent);
        if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
          {
            std::string oldpath = newFolder;
            std::string newpath = "./mdRestart/Step";
            newpath             = newpath + std::to_string(time) + "/.";
            dftUtils::copyFile(oldpath, newpath);
          }
        MPI_Barrier(d_mpiCommParent);
      }
  }

  void
  molecularDynamicsClass::writeTotalDisplacementFile(
    const std::vector<dealii::Tensor<1, 3, double>> &r,
    int                                              time)
  {
    if (d_dftPtr->getParametersObject().reproducible_output == false)
      {
        std::string prevPath =
          d_restartFilesPath + "/Step" + std::to_string(time - 1) + "/";
        std::string currPath =
          d_restartFilesPath + "/Step" + std::to_string(time) + "/";
        std::vector<std::vector<double>> fileDisplacementData;
        dftUtils::readFile(3,
                           fileDisplacementData,
                           prevPath + "Displacement.chk");
        for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
          {
            fileDisplacementData[iCharge][0] =
              fileDisplacementData[iCharge][0] + r[iCharge][0];
            fileDisplacementData[iCharge][1] =
              fileDisplacementData[iCharge][1] + r[iCharge][1];
            fileDisplacementData[iCharge][2] =
              fileDisplacementData[iCharge][2] + r[iCharge][2];
          }
        dftUtils::writeDataIntoFile(fileDisplacementData,
                                    currPath + "Displacement.chk",
                                    d_mpiCommParent);

        if (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
          {
            std::ofstream outfile;
            dftUtils::copyFile(prevPath + "TotalDisplacement.chk",
                               currPath + "TotalDisplacement.chk");
            outfile.open(currPath + "TotalDisplacement.chk",
                         std::ios_base::app);
            std::vector<std::vector<double>> atomLocations;
            atomLocations = d_dftPtr->getAtomLocationsCart();
            for (int iCharge = 0; iCharge < d_numberGlobalCharges; iCharge++)
              {
                outfile << atomLocations[iCharge][0] << "  "
                        << atomLocations[iCharge][1] << std::setprecision(16)
                        << "  " << fileDisplacementData[iCharge][0] << "  "
                        << fileDisplacementData[iCharge][1] << "  "
                        << fileDisplacementData[iCharge][2] << std::endl;
              }
            outfile.close();
          }
        MPI_Barrier(d_mpiCommParent);
      }
  }

  double
  molecularDynamicsClass::NoseHoverExtendedLagrangian(
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

  int
  molecularDynamicsClass::checkRestart(std::string &coordinatesFile,
                                       std::string &domainVectorsFile,
                                       bool &       scfRestart)
  {
    int time1 = 0;

    if (d_restartFlag == 1)
      {
        std::vector<std::vector<double>> t1, mdData;
        pcout << " MD is in Restart Mode" << std::endl;
        dftfe::dftUtils::readFile(
          3, mdData, d_restartFilesPath + "/mdRestart/moleculardynamics.dat");
        dftfe::dftUtils::readFile(1,
                                  t1,
                                  d_restartFilesPath + "/mdRestart/time.chk");
        time1                  = t1[0][0];
        std::string tempfolder = d_restartFilesPath + "/mdRestart/Step";
        bool        flag       = false;
        std::string path2      = tempfolder + std::to_string(time1);
        pcout << "Looking for files of TimeStep " << time1 << " at: " << path2
              << std::endl;
        while (!flag && time1 > 1)
          {
            std::string path = tempfolder + std::to_string(time1);
            std::string file1;
            if (mdData[4][0] == 1 || mdData[4][1] == 1 || mdData[4][2] == 1)
              file1 = path + "/atomsFracCoordCurrent.chk";
            else
              file1 = path + "/atomsCartCoordCurrent.chk";
            std::string   file2 = path + "/velocity.chk";
            std::string   file3 = path + "/NHCThermostat.chk";
            std::string   file4 = path + "/domainBoundingVectorsCurrent.chk";
            std::ifstream readFile1(file1.c_str());
            std::ifstream readFile2(file2.c_str());
            std::ifstream readFile3(file3.c_str());
            pcout << "Starting files search" << std::endl;
            bool NHCflag = true;
            if (mdData[0][0] == 2.0)
              {
                NHCflag = false;
                if (!readFile3.fail())
                  NHCflag = true;
              }
            pcout << "Finishing files search" << std::endl;
            if (!readFile1.fail() && !readFile2.fail() && NHCflag)
              {
                flag              = true;
                coordinatesFile   = file1;
                domainVectorsFile = file4;

                pcout << " Restart files are found in: " << path << std::endl;
                break;
              }

            else
              {
                pcout << "----Error opening restart files present in: " << path
                      << std::endl
                      << "Switching to time: " << --time1 << " ----"
                      << std::endl;
              }
          }
        if (time1 == t1[0][0])
          scfRestart = true;
        else
          scfRestart = false;
      }

    return (time1);
  }
  void
  molecularDynamicsClass::DensityExtrapolation(int TimeStep)
  {
    if (TimeStep == 0)
      d_extrapDensity_tmin2 = d_dftPtr->getRhoNodalOut();
    else if (TimeStep == 1)
      d_extrapDensity_tmin1 = d_dftPtr->getRhoNodalOut();
    else
      d_extrapDensity_t0 = d_dftPtr->getRhoNodalOut();


    if (TimeStep >= 2)
      {
        double A, B, C;
        // Compute Extrapolated Density
        // for loop
        pcout << "Using Extrapolated Density for init" << std::endl;
        d_extrapDensity_tp1.reinit(d_extrapDensity_t0);
        for (int i = 0; i < d_extrapDensity_t0.local_size(); i++)
          {
            C = d_extrapDensity_t0.local_element(i);
            B = 0.5 * (3 * d_extrapDensity_t0.local_element(i) +
                       d_extrapDensity_tmin2.local_element(i) -
                       4 * d_extrapDensity_tmin1.local_element(i));
            A = 0.5 * (d_extrapDensity_tmin2.local_element(i) -
                       2 * d_extrapDensity_tmin1.local_element(i) +
                       d_extrapDensity_t0.local_element(i));
            d_extrapDensity_tp1.local_element(i) = A + B + C;
            if (d_extrapDensity_tp1.local_element(i) < 0)
              d_extrapDensity_tp1.local_element(i) = 0.0;
            // pcout<<"Current Denisty New Density at "<<i<<"
            // "<<d_extrapDensity_0.local_element(i)<<" ->
            // "<<d_OutDensity.local_element(i)<<std::endl;
          }
        // Changing the Densities
        d_extrapDensity_tmin2 = d_extrapDensity_tmin1;
        d_extrapDensity_tmin1 = d_extrapDensity_t0;
        // Send OutDensity
        d_extrapDensity_tp1.update_ghost_values();
        d_dftPtr->resetRhoNodalIn(d_extrapDensity_tp1);
      }
  }
  void
  molecularDynamicsClass::DensitySplitExtrapolation(int TimeStep)
  {
    if (TimeStep == 0)
      {
        d_extrapDensity_tmin2 = d_dftPtr->getRhoNodalSplitOut();
        // d_extrapDensity_tmin2.add(dftPtr->getTotalChargeforRhoSplit());
      }
    else if (TimeStep == 1)
      {
        d_extrapDensity_tmin1 = d_dftPtr->getRhoNodalSplitOut();
        // d_extrapDensity_tmin1.add(dftPtr->getTotalChargeforRhoSplit());
      }
    else
      {
        d_extrapDensity_t0 = d_dftPtr->getRhoNodalSplitOut();
        // d_extrapDensity_t0.add(dftPtr->getTotalChargeforRhoSplit());
      }

    if (TimeStep >= 2)
      {
        double A, B, C;
        // Compute Extrapolated Density
        // for loop
        pcout << "Using Split Extrapolated Density for initialization"
              << std::endl;
        d_extrapDensity_tp1.reinit(d_extrapDensity_t0);
        for (int i = 0; i < d_extrapDensity_t0.local_size(); i++)
          {
            C = d_extrapDensity_t0.local_element(i);
            B = 0.5 * (3 * d_extrapDensity_t0.local_element(i) +
                       d_extrapDensity_tmin2.local_element(i) -
                       4 * d_extrapDensity_tmin1.local_element(i));
            A = 0.5 * (d_extrapDensity_tmin2.local_element(i) -
                       2 * d_extrapDensity_tmin1.local_element(i) +
                       d_extrapDensity_t0.local_element(i));
            d_extrapDensity_tp1.local_element(i) = A + B + C;
            // pcout<<"Current Denisty New Density at "<<i<<"
            // "<<d_extrapDensity_t0.local_element(i)<<" -> "<<
            // d_extrapDensity_tp1.local_element(i)<<std::endl;
          }
        // Changing the Densities
        d_extrapDensity_tmin2 = d_extrapDensity_tmin1;
        d_extrapDensity_tmin1 = d_extrapDensity_t0;
        // Send OutDensity
        d_extrapDensity_tp1.update_ghost_values();
        d_dftPtr->resetRhoNodalSplitIn(d_extrapDensity_tp1);
      }
  }



} // namespace dftfe
