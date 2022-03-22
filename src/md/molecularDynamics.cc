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
// @author Sambit Das
//

#include <boost/generator_iterator.hpp>
#include <boost/math/distributions/normal.hpp>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>

#include <dft.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include <force.h>
#include <molecularDynamics.h>

namespace dftfe
{
  namespace internalmd
  {
    extern "C"
    {
      //
      // lapack Ax=b
      //
      void
      dgesv_(int *   N,
             int *   NRHS,
             double *A,
             int *   LDA,
             int *   IPIV,
             double *B,
             int *   LDB,
             int *   INFO);
    }

    std::vector<double>
    getFractionalCoordinates(const std::vector<double> &latticeVectors,
                             const Point<3> &           point,
                             const Point<3> &           corner)
    {
      //
      // recenter vertex about corner
      //
      std::vector<double> recenteredPoint(3);
      for (unsigned int i = 0; i < 3; ++i)
        recenteredPoint[i] = point[i] - corner[i];

      std::vector<double> latticeVectorsDup = latticeVectors;

      //
      // to get the fractionalCoords, solve a linear
      // system of equations
      //
      int N    = 3;
      int NRHS = 1;
      int LDA  = 3;
      int IPIV[3];
      int info;

      dgesv_(&N,
             &NRHS,
             &latticeVectorsDup[0],
             &LDA,
             &IPIV[0],
             &recenteredPoint[0],
             &LDA,
             &info);
      AssertThrow(info == 0,
                  ExcMessage(
                    "LU solve in finding fractional coordinates failed."));
      return recenteredPoint;
    }

    void
    recursiveKernelApply(
      const std::vector<distributedCPUVec<double>> &ucontainer,
      const std::vector<distributedCPUVec<double>> &vcontainer,
      const double                                  k0,
      const distributedCPUVec<double> &             x,
      const unsigned int                            id,
      distributedCPUVec<double> &                   y)
    {
      if (id == 0)
        {
          for (unsigned int i = 0; i < x.local_size(); i++)
            y.local_element(i) = k0 * x.local_element(i);
        }
      else
        {
          distributedCPUVec<double> kminus1x, kminus1u;
          kminus1x.reinit(y);
          kminus1u.reinit(y);

          recursiveKernelApply(ucontainer, vcontainer, k0, x, id - 1, kminus1x);

          recursiveKernelApply(
            ucontainer, vcontainer, k0, ucontainer[id - 1], id - 1, kminus1u);

          const double vTkminus1x = vcontainer[id - 1] * kminus1x;
          const double vTkminus1u = vcontainer[id - 1] * kminus1u;
          for (unsigned int i = 0; i < x.local_size(); i++)
            y.local_element(i) =
              kminus1x.local_element(i) -
              kminus1u.local_element(i) * vTkminus1x / (1.0 + vTkminus1u);
        }
    }


    void
    recursiveJApply(const std::vector<distributedCPUVec<double>> &ucontainer,
                    const std::vector<distributedCPUVec<double>> &vcontainer,
                    const double                                  c,
                    const distributedCPUVec<double> &             x,
                    const unsigned int                            id,
                    distributedCPUVec<double> &                   y)
    {
      if (id == 0)
        {
          for (unsigned int i = 0; i < x.local_size(); i++)
            y.local_element(i) = -c * x.local_element(i);
        }
      else
        {
          distributedCPUVec<double> jminus1x;
          jminus1x.reinit(y);

          recursiveJApply(ucontainer, vcontainer, c, x, id - 1, jminus1x);

          const double vTx = vcontainer[id - 1] * x;
          for (unsigned int i = 0; i < x.local_size(); i++)
            y.local_element(i) = jminus1x.local_element(i) +
                                 ucontainer[id - 1].local_element(i) * vTx;
        }
    }

  } // namespace internalmd


  //
  // constructor
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  molecularDynamics<FEOrder, FEOrderElectro>::molecularDynamics(
    dftClass<FEOrder, FEOrderElectro> *_dftPtr,
    const MPI_Comm &                   mpi_comm_replica)
    : dftPtr(_dftPtr)
    , mpi_communicator(mpi_comm_replica)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_comm_replica))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_comm_replica))
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
  {}



  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  molecularDynamics<FEOrder, FEOrderElectro>::run()
  {
    //********************* Molecular dynamics
    // simulation*****************************//

    pcout
      << "---------------Starting Molecular dynamics simulation---------------------"
      << std::endl;
    const unsigned int numberGlobalCharges = dftPtr->atomLocations.size();
    // https://lammps.sandia.gov/doc/units.html (We are using metal units)
    const double initialTemperature = dftParameters::startingTempBOMDNVE; // K
    const unsigned int restartFlag =
      ((dftParameters::chkType == 1 || dftParameters::chkType == 3) &&
       dftParameters::restartMdFromChk) ?
        1 :
        0; // 1; //0;//1;
    bool xlbomdHistoryRestart =
      (dftParameters::chkType == 3 && dftParameters::restartMdFromChk) ? true :
                                                                         false;
    int startingTimeStep = 0; // 625;//450;// 50;//300; //0;// 300;

    const double timeStep =
      dftParameters::timeStepBOMD *
      0.09822694541304546435; // Conversion factor from femteseconds:
                              // 0.09822694541304546435 based on NIST constants
    const unsigned int numberTimeSteps = dftParameters::numberStepsBOMD;

    // https://physics.nist.gov/cuu/Constants/Table/allascii.txt
    const double kb = 8.617333262e-05; // eV/K **3.166811429e-6**;
    const double initialVelocityDeviation =
      sqrt(kb * initialTemperature / 27.0);
    const double haPerBohrToeVPerAng = 27.211386245988 / 0.529177210903;
    const double haToeV              = 27.211386245988;
    const double bohrToAng           = 0.529177210903;
    const double AngTobohr           = 1.0 / bohrToAng;

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
    std::vector<double> velocityMid(3 * numberGlobalCharges, 0.0);
    std::vector<double> force(3 * numberGlobalCharges, 0.0);
    std::vector<double> acceleration(3 * numberGlobalCharges, 0.0);
    std::vector<double> accelerationNew(3 * numberGlobalCharges, 0.0);

    std::vector<double> internalEnergyVector(numberTimeSteps, 0.0);
    std::vector<double> entropicEnergyVector(numberTimeSteps, 0.0);
    std::vector<double> kineticEnergyVector(numberTimeSteps, 0.0);
    std::vector<double> totalEnergyVector(numberTimeSteps, 0.0);
    std::vector<double> rmsErrorRhoVector(numberTimeSteps, 0.0);
    std::vector<double> rmsErrorGradRhoVector(numberTimeSteps, 0.0);

    const unsigned int kmax = dftParameters::kmaxXLBOMD;
    const unsigned int fullScfSolvesBeforeStartingXLBOMD = kmax;
    double             k;
    double             alpha;
    double             c0;
    double             c1;
    double             c2;
    double             c3;
    double             c4;
    double             c5;
    double             c6;
    double             c7;

    if (kmax == 1)
      {
        k     = 2.00;
        alpha = 0.0;
      }
    else if (kmax == 4)
      {
        k     = 1.69;
        alpha = 0.150;
        c0    = -2.0;
        c1    = 3.0;
        c2    = 0.0;
        c3    = -1.0;
      }
    else if (kmax == 6)
      {
        k     = 1.82;
        alpha = 0.018;
        c0    = -6.0;
        c1    = 14.0;
        c2    = -8.0;
        c3    = -3.0;
        c4    = 4.0;
        c5    = -1.0;
      }
    else if (kmax == 8)
      {
        k     = 1.86;
        alpha = 0.0016;
        c0    = -36.0;
        c1    = 99.0;
        c2    = -88.0;
        c3    = 11.0;
        c4    = 32.0;
        c5    = -25.0;
        c6    = 8.0;
        c7    = -1.0;
      }

    const double diracDeltaKernelConstant =
      -dftParameters::diracDeltaKernelScalingConstant;
    const double k0kernelconstant =
      1.0 / dftParameters::diracDeltaKernelScalingConstant - 1.0; // 20.0;//20.0
    std::deque<distributedCPUVec<double>>  approxDensityContainer;
    distributedCPUVec<double>              shadowKSRhoMin;
    distributedCPUVec<double>              approxDensityNext;
    distributedCPUVec<double>              rhoErrorVec;
    std::vector<distributedCPUVec<double>> ucontainer(
      dftParameters::kernelUpdateRankXLBOMD); // for low rank update kernel
    std::vector<distributedCPUVec<double>> vcontainer(
      dftParameters::kernelUpdateRankXLBOMD); // for low rank update kernel
    distributedCPUVec<double> jv;
    distributedCPUVec<double>
                              peturbedApproxDensity; // for low rank update kernel
    distributedCPUVec<double> temp1Vec;              // for central difference
    distributedCPUVec<double> temp2Vec;              // for central difference
    distributedCPUVec<double> kernelAction;          //

    double kineticEnergy = 0.0;

    // boost::mt19937 rng(std::time(0));
    boost::mt19937               rng;
    boost::normal_distribution<> gaussianDist(0.0, initialVelocityDeviation);
    boost::variate_generator<boost::mt19937 &, boost::normal_distribution<>>
           generator(rng, gaussianDist);
    double averageKineticEnergy;
    double temperatureFromVelocities;
    double totalEnergyStartingTimeStep = 0.0;

    dealii::IndexSet locally_relevant_dofs_;
    dealii::DoFTools::extract_locally_relevant_dofs(
      dftPtr->d_dofHandlerRhoNodal, locally_relevant_dofs_);

    const dealii::IndexSet &locally_owned_dofs_ =
      dftPtr->d_dofHandlerRhoNodal.locally_owned_dofs();
    dealii::IndexSet ghost_indices_ = locally_relevant_dofs_;
    ghost_indices_.subtract_set(locally_owned_dofs_);

    distributedCPUVec<double> tempVecRestart =
      distributedCPUVec<double>(locally_owned_dofs_,
                                ghost_indices_,
                                mpi_communicator);

    double temp1p;

    if (restartFlag == 0)
      {
        dftPtr->solve(true,
                      false,
                      false,
                      dftPtr->d_isRestartGroundStateCalcFromChk);
        dftPtr->d_isRestartGroundStateCalcFromChk = false;
        const std::vector<double> forceOnAtoms =
          dftPtr->forcePtr->getAtomsForces();

        dftPtr->initAtomicRho();
        shadowKSRhoMin = dftPtr->d_rhoOutNodalValues;
        if (dftParameters::useAtomicRhoXLBOMD)
          {
            dftPtr->l2ProjectionQuadDensityMinusAtomicDensity(
              dftPtr->d_matrixFreeDataPRefined,
              dftPtr->d_constraintsRhoNodal,
              dftPtr->d_densityDofHandlerIndexElectro,
              dftPtr->d_densityQuadratureIdElectro,
              *(dftPtr->rhoOutValues),
              shadowKSRhoMin);
          }

        shadowKSRhoMin.update_ghost_values();

        // Set approximate electron density to the exact ground state electron
        // density in the 0th step
        if (dftParameters::isXLBOMD)
          {
            approxDensityContainer.push_back(shadowKSRhoMin);
            approxDensityContainer.back().update_ghost_values();
          }

        if (dftParameters::chkType == 3 && !xlbomdHistoryRestart)
          if (approxDensityContainer.size() == kmax)
            {
              dftPtr->d_groundStateDensityHistory.clear();
              dftPtr->d_groundStateDensityHistory.resize(kmax);
              for (unsigned int i = 0; i < kmax; i++)
                {
                  dftPtr->d_groundStateDensityHistory[i].reinit(tempVecRestart);

                  for (unsigned int idof = 0;
                       idof < approxDensityContainer[i].local_size();
                       idof++)
                    dftPtr->d_groundStateDensityHistory[i].local_element(idof) =
                      approxDensityContainer[i].local_element(idof);

                  dftPtr->d_groundStateDensityHistory[i].update_ghost_values();
                }
            }

        if (this_mpi_process == 0)
          {
            for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
              {
                velocity[3 * iCharge + 0] = generator();
                velocity[3 * iCharge + 1] = generator();
                velocity[3 * iCharge + 2] = generator();
              }
          }

        for (unsigned int i = 0; i < numberGlobalCharges * 3; ++i)
          {
            velocity[i] =
              dealii::Utilities::MPI::sum(velocity[i], mpi_communicator);
          }

        // compute initial acceleration and initial kinEnergy
        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            acceleration[3 * iCharge + 0] =
              -(forceOnAtoms[3 * iCharge + 0] * haPerBohrToeVPerAng) /
              massAtoms[iCharge] * 0.0;
            acceleration[3 * iCharge + 1] =
              -(forceOnAtoms[3 * iCharge + 1] * haPerBohrToeVPerAng) /
              massAtoms[iCharge] * 0.0;
            acceleration[3 * iCharge + 2] =
              -(forceOnAtoms[3 * iCharge + 2] * haPerBohrToeVPerAng) /
              massAtoms[iCharge] * 0.0;
            kineticEnergy +=
              0.5 * massAtoms[iCharge] *
              (velocity[3 * iCharge + 0] * velocity[3 * iCharge + 0] +
               velocity[3 * iCharge + 1] * velocity[3 * iCharge + 1] +
               velocity[3 * iCharge + 2] * velocity[3 * iCharge + 2]);
          }
      }
    else
      {
        // read into acceleration vector
        std::string fileName = "acceleration.chk"; // accelerationData1x1x1
        std::vector<std::vector<double>> fileAccData;

        dftUtils::readFile(3, fileAccData, fileName);

        std::string fileName1 = "velocity.chk"; // velocityData1x1x1
        std::vector<std::vector<double>> fileVelData;

        dftUtils::readFile(3, fileVelData, fileName1);

        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            acceleration[3 * iCharge + 0] = fileAccData[iCharge][0];
            acceleration[3 * iCharge + 1] = fileAccData[iCharge][1];
            acceleration[3 * iCharge + 2] = fileAccData[iCharge][2];
          }

        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            velocity[3 * iCharge + 0] = fileVelData[iCharge][0];
            velocity[3 * iCharge + 1] = fileVelData[iCharge][1];
            velocity[3 * iCharge + 2] = fileVelData[iCharge][2];
          }

        std::vector<std::vector<double>> fileTemperatueData;
        std::vector<std::vector<double>> timeIndexData;

        dftUtils::readFile(1, fileTemperatueData, "temperature.chk");

        dftUtils::readFile(1, timeIndexData, "time.chk");


        temperatureFromVelocities = fileTemperatueData[0][0];


        startingTimeStep = timeIndexData[0][0];


        std::vector<std::vector<double>> totalEnergyData;
        dftUtils::readFile(1, totalEnergyData, "TotEngMd");

        totalEnergyStartingTimeStep = totalEnergyData[startingTimeStep][0];

        pcout << " Ending time step read from file: " << startingTimeStep
              << std::endl;
        pcout << " Temperature read from file: " << temperatureFromVelocities
              << std::endl;

        if (xlbomdHistoryRestart)
          {
            approxDensityContainer.resize(kmax);
            for (unsigned int i = 0; i < kmax; i++)
              {
                if (i == 0)
                  dftPtr->d_matrixFreeDataPRefined.initialize_dof_vector(
                    approxDensityContainer[i],
                    dftPtr->d_densityDofHandlerIndexElectro);
                else
                  approxDensityContainer[i].reinit(approxDensityContainer[0]);

                for (unsigned int idof = 0;
                     idof < approxDensityContainer[i].local_size();
                     idof++)
                  approxDensityContainer[i].local_element(idof) =
                    dftPtr->d_groundStateDensityHistory[i].local_element(idof);

                approxDensityContainer[i].update_ghost_values();

                /*
                   if (dftParameters::verbosity>=4)
                   pcout<<"l2 norm of approxDensityVec read:  "<<i<< "  "<<
                approxDensityContainer[i].l2_norm()<<std::endl;

                //normalize approxDensityVec
                double charge =
                dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,
                approxDensityContainer[i]);


                if (dftParameters::verbosity>=1)
                pcout<<"Total Charge approxDensityVec:  "<<charge<<std::endl;
                 */
              }
            shadowKSRhoMin = approxDensityContainer.back();
            shadowKSRhoMin.update_ghost_values();
          }
      }



    if (restartFlag == 0)
      {
        //
        // compute average velocity
        //
        double avgVelx = 0.0, avgVely = 0.0, avgVelz = 0.0;
        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            avgVelx += velocity[3 * iCharge + 0];
            avgVely += velocity[3 * iCharge + 1];
            avgVelz += velocity[3 * iCharge + 2];
          }

        avgVelx = avgVelx / numberGlobalCharges;
        avgVely = avgVely / numberGlobalCharges;
        avgVelz = avgVelz / numberGlobalCharges;

        //
        // recompute initial velocities by subtracting out center of mass
        // velocity
        //
        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            velocity[3 * iCharge + 0] -= avgVelx;
            velocity[3 * iCharge + 1] -= avgVely;
            velocity[3 * iCharge + 2] -= avgVelz;
          }

        // recompute kinetic energy again
        kineticEnergy = 0.0;
        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            kineticEnergy +=
              0.5 * massAtoms[iCharge] *
              (velocity[3 * iCharge + 0] * velocity[3 * iCharge + 0] +
               velocity[3 * iCharge + 1] * velocity[3 * iCharge + 1] +
               velocity[3 * iCharge + 2] * velocity[3 * iCharge + 2]);
          }

        averageKineticEnergy      = kineticEnergy / (3 * numberGlobalCharges);
        temperatureFromVelocities = averageKineticEnergy * 2 / kb;

        pcout << "Temperature computed from Velocities: "
              << temperatureFromVelocities << std::endl;

        double gamma = sqrt(initialTemperature / temperatureFromVelocities);

        for (int i = 0; i < 3 * numberGlobalCharges; ++i)
          {
            velocity[i] = gamma * velocity[i];
          }

        // recompute kinetic energy again
        kineticEnergy = 0.0;
        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            kineticEnergy +=
              0.5 * massAtoms[iCharge] *
              (velocity[3 * iCharge + 0] * velocity[3 * iCharge + 0] +
               velocity[3 * iCharge + 1] * velocity[3 * iCharge + 1] +
               velocity[3 * iCharge + 2] * velocity[3 * iCharge + 2]);
          }

        averageKineticEnergy      = kineticEnergy / (3 * numberGlobalCharges);
        temperatureFromVelocities = averageKineticEnergy * 2 / kb;

        pcout << "Temperature computed from Scaled Velocities: "
              << temperatureFromVelocities << std::endl;

        kineticEnergyVector[0]  = kineticEnergy / haToeV;
        internalEnergyVector[0] = dftPtr->d_groundStateEnergy;
        entropicEnergyVector[0] = dftPtr->d_entropicEnergy;
        totalEnergyVector[0]    = kineticEnergyVector[0] +
                               internalEnergyVector[0] -
                               entropicEnergyVector[0];
        rmsErrorRhoVector[0]     = 0.0;
        rmsErrorGradRhoVector[0] = 0.0;

        pcout << " Initial Velocity Deviation: " << initialVelocityDeviation
              << std::endl;
        pcout << " Kinetic Energy in Ha at timeIndex 0 "
              << kineticEnergyVector[0] << std::endl;
        pcout << " Internal Energy in Ha at timeIndex 0 "
              << internalEnergyVector[0] << std::endl;
        pcout << " Entropic Energy in Ha at timeIndex 0 "
              << entropicEnergyVector[0] << std::endl;
        pcout << " Total Energy in Ha at timeIndex 0 " << totalEnergyVector[0]
              << std::endl;
        if (dftParameters::isXLBOMD)
          {
            pcout << " RMS error in rho in a.u. at timeIndex 0 "
                  << rmsErrorRhoVector[0] << std::endl;
            pcout << " RMS error in grad rho in a.u. at timeIndex 0 "
                  << rmsErrorGradRhoVector[0] << std::endl;
          }
      }

    bool isFirstXLBOMDStep = true;
    //
    // start the MD simulation
    //
    for (int timeIndex = startingTimeStep + 1; timeIndex < numberTimeSteps;
         ++timeIndex)
      {
        double step_time;
        MPI_Barrier(MPI_COMM_WORLD);
        step_time = MPI_Wtime();

        //
        // compute average velocity
        //
        double avgVelx = 0.0, avgVely = 0.0, avgVelz = 0.0;
        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            avgVelx += velocity[3 * iCharge + 0];
            avgVely += velocity[3 * iCharge + 1];
            avgVelz += velocity[3 * iCharge + 2];
          }

        avgVelx = avgVelx / numberGlobalCharges;
        avgVely = avgVely / numberGlobalCharges;
        avgVelz = avgVelz / numberGlobalCharges;

        pcout << "Velocity and acceleration of atoms at current timeStep "
              << timeIndex << std::endl;
        pcout << "Velocity of center of mass: " << avgVelx << " " << avgVely
              << " " << avgVelz << " " << std::endl;

        //
        // compute new positions and displacements from t to t + \Delta t using
        // combined leap-frog and velocity Verlet scheme
        //
        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            velocityMid[3 * iCharge + 0] =
              velocity[3 * iCharge + 0] +
              0.5 * acceleration[3 * iCharge + 0] * timeStep;
            velocityMid[3 * iCharge + 1] =
              velocity[3 * iCharge + 1] +
              0.5 * acceleration[3 * iCharge + 1] * timeStep;
            velocityMid[3 * iCharge + 2] =
              velocity[3 * iCharge + 2] +
              0.5 * acceleration[3 * iCharge + 2] * timeStep;

            displacements[iCharge][0] = velocityMid[3 * iCharge + 0] * timeStep;
            displacements[iCharge][1] = velocityMid[3 * iCharge + 1] * timeStep;
            displacements[iCharge][2] = velocityMid[3 * iCharge + 2] * timeStep;
          }

        //
        // compute forces for atoms at t + \Delta t
        //

        // convert displacements into atomic units
        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            displacements[iCharge][0] = displacements[iCharge][0] * AngTobohr;
            displacements[iCharge][1] = displacements[iCharge][1] * AngTobohr;
            displacements[iCharge][2] = displacements[iCharge][2] * AngTobohr;
          }

        if (dftParameters::verbosity >= 5)
          {
            pcout << "Displacement of atoms at current timeStep " << timeIndex
                  << std::endl;
            for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
              {
                pcout << "Charge Id: " << iCharge << " "
                      << displacements[iCharge][0] << " "
                      << displacements[iCharge][1] << " "
                      << displacements[iCharge][2] << std::endl;
              }
          }

        double update_time;
        MPI_Barrier(MPI_COMM_WORLD);
        update_time = MPI_Wtime();
        //
        // first move the mesh to current positions
        //
        dftPtr->updateAtomPositionsAndMoveMesh(
          displacements,
          dftParameters::maxJacobianRatioFactorForMD,
          (timeIndex == startingTimeStep + 1 && restartFlag == 1) ? true :
                                                                    false);


        if (dftParameters::verbosity >= 5)
          {
            pcout << "New atomic positions on the Mesh: " << std::endl;
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



        double atomicrho_time;
        MPI_Barrier(MPI_COMM_WORLD);
        atomicrho_time = MPI_Wtime();

        dftPtr->initAtomicRho();

        MPI_Barrier(MPI_COMM_WORLD);

        atomicrho_time = MPI_Wtime() - atomicrho_time;
        if (dftParameters::verbosity >= 1)
          pcout << "Time taken for initAtomicRho: " << atomicrho_time
                << std::endl;


        double rmsErrorRho     = 0.0;
        double rmsErrorGradRho = 0.0;
        double shadowPotentialInternalEnergy;
        double entropicEnergyShadowPotential;

        // if ((timeIndex == (startingTimeStep+1) && restartFlag==1))
        //	isFirstXLBOMDStep=true;

        bool isXlBOMDStep        = false;
        bool writeDensityHistory = false;
        if (dftParameters::isXLBOMD)
          {
            if (((timeIndex == (startingTimeStep + 1) && restartFlag == 1) ||
                 (timeIndex < (fullScfSolvesBeforeStartingXLBOMD))) &&
                !xlbomdHistoryRestart)
              {
                if (!(timeIndex == startingTimeStep + 1 && restartFlag == 1))
                  {
                    dftPtr->interpolateRhoNodalDataToQuadratureDataGeneral(
                      dftPtr->d_matrixFreeDataPRefined,
                      dftPtr->d_densityDofHandlerIndexElectro,
                      dftPtr->d_densityQuadratureIdElectro,
                      shadowKSRhoMin,
                      *(dftPtr->rhoInValues),
                      *(dftPtr->gradRhoInValues),
                      *(dftPtr->gradRhoInValues),
                      dftParameters::xcFamilyType == "GGA");

                    if (dftParameters::useAtomicRhoXLBOMD)
                      dftPtr->addAtomicRhoQuadValuesGradients(
                        *(dftPtr->rhoInValues),
                        *(dftPtr->gradRhoInValues),
                        dftParameters::xcFamilyType == "GGA");

                    dftPtr->normalizeRhoInQuadValues();

                    dftPtr->l2ProjectionQuadToNodal(
                      dftPtr->d_matrixFreeDataPRefined,
                      dftPtr->d_constraintsRhoNodal,
                      dftPtr->d_densityDofHandlerIndexElectro,
                      dftPtr->d_densityQuadratureIdElectro,
                      *(dftPtr->rhoInValues),
                      dftPtr->d_rhoInNodalValues);

                    dftPtr->d_rhoInNodalValues.update_ghost_values();
                  }

                dftPtr->solve();

                shadowKSRhoMin = dftPtr->d_rhoOutNodalValues;
                if (dftParameters::useAtomicRhoXLBOMD)
                  {
                    dftPtr->l2ProjectionQuadDensityMinusAtomicDensity(
                      dftPtr->d_matrixFreeDataPRefined,
                      dftPtr->d_constraintsRhoNodal,
                      dftPtr->d_densityDofHandlerIndexElectro,
                      dftPtr->d_densityQuadratureIdElectro,
                      *(dftPtr->rhoOutValues),
                      shadowKSRhoMin);
                  }
                shadowKSRhoMin.update_ghost_values();

                approxDensityContainer.push_back(shadowKSRhoMin);
                approxDensityContainer.back().update_ghost_values();

                if (dftParameters::chkType == 3 && !xlbomdHistoryRestart)
                  if (approxDensityContainer.size() == kmax)
                    {
                      dftPtr->d_groundStateDensityHistory.clear();
                      dftPtr->d_groundStateDensityHistory.resize(kmax);
                      for (unsigned int i = 0; i < kmax; i++)
                        {
                          // if (dftParameters::verbosity>=4)
                          //   pcout<<"l2 norm of approxDensityVec to be
                          //   written:  "<<i<< "  "<<
                          //   approxDensityContainer[i].l2_norm()<<std::endl;

                          dftPtr->d_groundStateDensityHistory[i].reinit(
                            tempVecRestart);

                          for (unsigned int idof = 0;
                               idof < approxDensityContainer[i].local_size();
                               idof++)
                            dftPtr->d_groundStateDensityHistory[i]
                              .local_element(idof) =
                              approxDensityContainer[i].local_element(idof);

                          dftPtr->d_groundStateDensityHistory[i]
                            .update_ghost_values();
                        }
                      writeDensityHistory = true;
                    }
              }
            else
              {
                isXlBOMDStep = true;
                double xlbomdpre_time;
                MPI_Barrier(MPI_COMM_WORLD);
                xlbomdpre_time = MPI_Wtime();

                approxDensityNext.reinit(approxDensityContainer.back());

                const unsigned int containerSizeCurrent =
                  approxDensityContainer.size();
                distributedCPUVec<double> &approxDensityTimeT =
                  approxDensityContainer[containerSizeCurrent - 1];
                distributedCPUVec<double> &approxDensityTimeTMinusDeltat =
                  containerSizeCurrent > 1 ?
                    approxDensityContainer[containerSizeCurrent - 2] :
                    approxDensityTimeT;

                const unsigned int local_size = approxDensityNext.local_size();


                if (dftParameters::kernelUpdateRankXLBOMD > 0 &&
                    !isFirstXLBOMDStep)
                  {
                    for (unsigned int i = 0; i < local_size; i++)
                      approxDensityNext.local_element(i) =
                        approxDensityTimeT.local_element(i) * 2.0 -
                        approxDensityTimeTMinusDeltat.local_element(i) -
                        k * (kernelAction.local_element(i));
                  }
                else
                  {
                    for (unsigned int i = 0; i < local_size; i++)
                      approxDensityNext.local_element(i) =
                        approxDensityTimeT.local_element(i) * 2.0 -
                        approxDensityTimeTMinusDeltat.local_element(i) -
                        (shadowKSRhoMin.local_element(i) -
                         approxDensityTimeT.local_element(i)) *
                          k * diracDeltaKernelConstant;
                  }

                if (approxDensityContainer.size() == kmax)
                  {
                    if (kmax == 1)
                      {
                        for (unsigned int i = 0; i < local_size; i++)
                          approxDensityNext.local_element(i) +=
                            alpha *
                            (c0 *
                             approxDensityContainer[containerSizeCurrent - 1]
                               .local_element(i));
                      }
                    if (kmax == 4)
                      {
                        for (unsigned int i = 0; i < local_size; i++)
                          approxDensityNext.local_element(i) +=
                            alpha *
                            (c0 *
                               approxDensityContainer[containerSizeCurrent - 1]
                                 .local_element(i) +
                             c1 *
                               approxDensityContainer[containerSizeCurrent - 2]
                                 .local_element(i) +
                             c2 *
                               approxDensityContainer[containerSizeCurrent - 3]
                                 .local_element(i) +
                             c3 *
                               approxDensityContainer[containerSizeCurrent - 4]
                                 .local_element(i));
                      }
                    else if (kmax == 6)
                      {
                        for (unsigned int i = 0; i < local_size; i++)
                          approxDensityNext.local_element(i) +=
                            alpha *
                            (c0 *
                               approxDensityContainer[containerSizeCurrent - 1]
                                 .local_element(i) +
                             c1 *
                               approxDensityContainer[containerSizeCurrent - 2]
                                 .local_element(i) +
                             c2 *
                               approxDensityContainer[containerSizeCurrent - 3]
                                 .local_element(i) +
                             c3 *
                               approxDensityContainer[containerSizeCurrent - 4]
                                 .local_element(i) +
                             c4 *
                               approxDensityContainer[containerSizeCurrent - 5]
                                 .local_element(i) +
                             c5 *
                               approxDensityContainer[containerSizeCurrent - 6]
                                 .local_element(i));
                      }
                    else if (kmax == 8)
                      {
                        for (unsigned int i = 0; i < local_size; i++)
                          approxDensityNext.local_element(i) +=
                            alpha *
                            (c0 *
                               approxDensityContainer[containerSizeCurrent - 1]
                                 .local_element(i) +
                             c1 *
                               approxDensityContainer[containerSizeCurrent - 2]
                                 .local_element(i) +
                             c2 *
                               approxDensityContainer[containerSizeCurrent - 3]
                                 .local_element(i) +
                             c3 *
                               approxDensityContainer[containerSizeCurrent - 4]
                                 .local_element(i) +
                             c4 *
                               approxDensityContainer[containerSizeCurrent - 5]
                                 .local_element(i) +
                             c5 *
                               approxDensityContainer[containerSizeCurrent - 6]
                                 .local_element(i) +
                             c6 *
                               approxDensityContainer[containerSizeCurrent - 7]
                                 .local_element(i) +
                             c7 *
                               approxDensityContainer[containerSizeCurrent - 8]
                                 .local_element(i));
                      }
                    approxDensityContainer.pop_front();
                  }
                approxDensityContainer.push_back(approxDensityNext);
                approxDensityContainer.back().update_ghost_values();

                // normalize approxDensityVec
                double charge =
                  dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,
                                      approxDensityContainer.back());


                if (dftParameters::verbosity >= 1)
                  pcout
                    << "Total Charge before Normalizing new approxDensityVec:  "
                    << charge << std::endl;

                if (dftParameters::useAtomicRhoXLBOMD)
                  approxDensityContainer.back().add(-charge /
                                                    dftPtr->d_domainVolume);
                else
                  approxDensityContainer.back() *=
                    ((double)dftPtr->numElectrons) / charge;


                if (dftParameters::verbosity >= 1)
                  pcout
                    << "Total Charge after Normalizing new approxDensityVec:  "
                    << dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,
                                           approxDensityContainer.back())
                    << std::endl;

                approxDensityContainer.back().update_ghost_values();
                dftPtr->interpolateRhoNodalDataToQuadratureDataGeneral(
                  dftPtr->d_matrixFreeDataPRefined,
                  dftPtr->d_densityDofHandlerIndexElectro,
                  dftPtr->d_densityQuadratureIdElectro,
                  approxDensityContainer.back(),
                  *(dftPtr->rhoInValues),
                  *(dftPtr->gradRhoInValues),
                  *(dftPtr->gradRhoInValues),
                  dftParameters::xcFamilyType == "GGA");

                if (dftParameters::useAtomicRhoXLBOMD)
                  dftPtr->addAtomicRhoQuadValuesGradients(
                    *(dftPtr->rhoInValues),
                    *(dftPtr->gradRhoInValues),
                    dftParameters::xcFamilyType == "GGA");

                dftPtr->normalizeRhoInQuadValues();

                MPI_Barrier(MPI_COMM_WORLD);
                xlbomdpre_time = MPI_Wtime() - xlbomdpre_time;
                if (dftParameters::verbosity >= 1)
                  pcout << "Time taken for xlbomd preinitializations: "
                        << xlbomdpre_time << std::endl;

                double shadowsolve_time;
                MPI_Barrier(MPI_COMM_WORLD);
                shadowsolve_time = MPI_Wtime();

                if (dftParameters::verbosity >= 1)
                  pcout
                    << "----------Start shadow potential energy solve with approx density= n-------------"
                    << std::endl;

                if (isFirstXLBOMDStep && xlbomdHistoryRestart)
                  {
                    temp1p = dftParameters::chebyshevFilterTolXLBOMD;
                    dftParameters::chebyshevFilterTolXLBOMD =
                      dftParameters::xlbomdRestartChebyTol;
                  }

                dftPtr->solve(true, false, true, false);

                // Finite difference check for shadow potential forces
                if (false)
                  {
                    for (int iCharge = 0; iCharge < numberGlobalCharges;
                         ++iCharge)
                      {
                        if (iCharge == 0)
                          {
                            displacements[iCharge][0] = 1e-3;
                            displacements[iCharge][1] = 0.0;
                            displacements[iCharge][2] = 0.0;
                          }
                        else
                          {
                            displacements[iCharge][0] = 0.0;
                            displacements[iCharge][1] = 0.0;
                            displacements[iCharge][2] = 0.0;
                          }
                      }

                    dftPtr->updateAtomPositionsAndMoveMesh(
                      displacements,
                      dftParameters::maxJacobianRatioFactorForMD,
                      (timeIndex == startingTimeStep + 1 && restartFlag == 1) ?
                        true :
                        false);

                    dftPtr->initAtomicRho();

                    approxDensityContainer.back().update_ghost_values();

                    dftPtr->interpolateRhoNodalDataToQuadratureDataGeneral(
                      dftPtr->d_matrixFreeDataPRefined,
                      dftPtr->d_densityDofHandlerIndexElectro,
                      dftPtr->d_densityQuadratureIdElectro,
                      approxDensityContainer.back(),
                      *(dftPtr->rhoInValues),
                      *(dftPtr->gradRhoInValues),
                      *(dftPtr->gradRhoInValues),
                      dftParameters::xcFamilyType == "GGA");

                    if (dftParameters::useAtomicRhoXLBOMD)
                      dftPtr->addAtomicRhoQuadValuesGradients(
                        *(dftPtr->rhoInValues),
                        *(dftPtr->gradRhoInValues),
                        dftParameters::xcFamilyType == "GGA");

                    dftPtr->normalizeRhoInQuadValues();

                    dftPtr->solve(true, false, true, false);

                    for (int iCharge = 0; iCharge < numberGlobalCharges;
                         ++iCharge)
                      {
                        if (iCharge == 0)
                          {
                            displacements[iCharge][0] = -2e-3;
                            displacements[iCharge][1] = 0.0;
                            displacements[iCharge][2] = 0.0;
                          }
                        else
                          {
                            displacements[iCharge][0] = 0.0;
                            displacements[iCharge][1] = 0.0;
                            displacements[iCharge][2] = 0.0;
                          }
                      }

                    dftPtr->updateAtomPositionsAndMoveMesh(
                      displacements,
                      dftParameters::maxJacobianRatioFactorForMD,
                      (timeIndex == startingTimeStep + 1 && restartFlag == 1) ?
                        true :
                        false);

                    dftPtr->initAtomicRho();

                    approxDensityContainer.back().update_ghost_values();

                    dftPtr->interpolateRhoNodalDataToQuadratureDataGeneral(
                      dftPtr->d_matrixFreeDataPRefined,
                      dftPtr->d_densityDofHandlerIndexElectro,
                      dftPtr->d_densityQuadratureIdElectro,
                      approxDensityContainer.back(),
                      *(dftPtr->rhoInValues),
                      *(dftPtr->gradRhoInValues),
                      *(dftPtr->gradRhoInValues),
                      dftParameters::xcFamilyType == "GGA");

                    if (dftParameters::useAtomicRhoXLBOMD)
                      dftPtr->addAtomicRhoQuadValuesGradients(
                        *(dftPtr->rhoInValues),
                        *(dftPtr->gradRhoInValues),
                        dftParameters::xcFamilyType == "GGA");

                    dftPtr->normalizeRhoInQuadValues();

                    dftPtr->solve(true, false, true, false);
                  }


                if (isFirstXLBOMDStep && xlbomdHistoryRestart)
                  {
                    dftParameters::chebyshevFilterTolXLBOMD = temp1p;
                  }

                if (dftParameters::verbosity >= 1)
                  pcout
                    << "----------End shadow potential energy solve with approx density= n-------------"
                    << std::endl;

                shadowPotentialInternalEnergy = dftPtr->d_shadowPotentialEnergy;
                entropicEnergyShadowPotential = dftPtr->d_entropicEnergy;

                MPI_Barrier(MPI_COMM_WORLD);
                shadowsolve_time = MPI_Wtime() - shadowsolve_time;
                if (dftParameters::verbosity >= 1)
                  pcout << "Time taken for xlbomd shadow potential solve: "
                        << shadowsolve_time << std::endl;

                double xlbomdpost_time;
                MPI_Barrier(MPI_COMM_WORLD);
                xlbomdpost_time = MPI_Wtime();

                if (dftParameters::useAtomicRhoXLBOMD)
                  {
                    dftPtr->l2ProjectionQuadDensityMinusAtomicDensity(
                      dftPtr->d_matrixFreeDataPRefined,
                      dftPtr->d_constraintsRhoNodal,
                      dftPtr->d_densityDofHandlerIndexElectro,
                      dftPtr->d_densityQuadratureIdElectro,
                      *(dftPtr->rhoOutValues),
                      shadowKSRhoMin);
                  }
                else
                  shadowKSRhoMin = dftPtr->d_rhoOutNodalValues;

                shadowKSRhoMin.update_ghost_values();

                rhoErrorVec = shadowKSRhoMin;
                rhoErrorVec -= approxDensityContainer.back();

                if (dftParameters::kernelUpdateRankXLBOMD > 0)
                  {
                    temp1p = dftParameters::chebyshevFilterTolXLBOMD;
                    dftParameters::chebyshevFilterTolXLBOMD =
                      dftParameters::chebyshevFilterTolXLBOMDRankUpdates;

                    const double deltalambda =
                      dftParameters::xlbomdKernelRankUpdateFDParameter *
                      std::sqrt(dftPtr->rhofieldl2Norm(
                                  dftPtr->d_matrixFreeDataPRefined,
                                  approxDensityContainer.back(),
                                  dftPtr->d_densityDofHandlerIndexElectro,
                                  dftPtr->d_densityQuadratureIdElectro) /
                                dftPtr->d_domainVolume);

                    if (dftParameters::verbosity >= 1)
                      pcout << "deltalambda: " << deltalambda << std::endl;

                    const double k0 = -1.0 / (k0kernelconstant + 1.0);

                    jv.reinit(rhoErrorVec);
                    kernelAction.reinit(rhoErrorVec);

                    distributedCPUVec<double> compvec;
                    compvec.reinit(rhoErrorVec);
                    for (unsigned int irank = 0;
                         irank < dftParameters::kernelUpdateRankXLBOMD;
                         irank++)
                      {
                        vcontainer[irank].reinit(rhoErrorVec);
                        internalmd::recursiveKernelApply(ucontainer,
                                                         vcontainer,
                                                         k0,
                                                         rhoErrorVec,
                                                         irank,
                                                         vcontainer[irank]);

                        compvec = 0;
                        for (int jrank = 0; jrank < irank; jrank++)
                          {
                            const double tTvj =
                              vcontainer[irank] * vcontainer[jrank];
                            compvec.add(tTvj, vcontainer[jrank]);
                          }
                        vcontainer[irank] -= compvec;

                        vcontainer[irank] *= 1.0 / vcontainer[irank].l2_norm();

                        if (dftParameters::verbosity >= 1)
                          pcout << " Vector norm of v:  "
                                << vcontainer[irank].l2_norm()
                                << ", for rank: " << irank + 1 << std::endl;

                        peturbedApproxDensity = approxDensityContainer.back();
                        peturbedApproxDensity.add(deltalambda,
                                                  vcontainer[irank]);

                        peturbedApproxDensity.update_ghost_values();

                        // normalize peturbedApproxDensity
                        charge =
                          dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,
                                              peturbedApproxDensity);


                        if (dftParameters::verbosity >= 1)
                          pcout
                            << "Total Charge before Normalizing peturbedApproxDensity:  "
                            << charge << std::endl;

                        if (dftParameters::useAtomicRhoXLBOMD)
                          peturbedApproxDensity.add(-charge /
                                                    dftPtr->d_domainVolume);
                        else
                          peturbedApproxDensity *=
                            ((double)dftPtr->numElectrons) / charge;

                        peturbedApproxDensity.update_ghost_values();
                        dftPtr->interpolateRhoNodalDataToQuadratureDataGeneral(
                          dftPtr->d_matrixFreeDataPRefined,
                          dftPtr->d_densityDofHandlerIndexElectro,
                          dftPtr->d_densityQuadratureIdElectro,
                          peturbedApproxDensity,
                          *(dftPtr->rhoInValues),
                          *(dftPtr->gradRhoInValues),
                          *(dftPtr->gradRhoInValues),
                          dftParameters::xcFamilyType == "GGA");

                        if (dftParameters::useAtomicRhoXLBOMD)
                          dftPtr->addAtomicRhoQuadValuesGradients(
                            *(dftPtr->rhoInValues),
                            *(dftPtr->gradRhoInValues),
                            dftParameters::xcFamilyType == "GGA");

                        dftPtr->normalizeRhoInQuadValues();

                        if (dftParameters::verbosity >= 1)
                          pcout
                            << "----------Start density perturbation solve with approx density= n+lamda*v1-------------"
                            << std::endl;


                        if (dftParameters::
                              useDensityMatrixPerturbationRankUpdates)
                          dftPtr->computeDensityPerturbation();
                        else
                          dftPtr->solve(false, false, true, false);

                        if (dftParameters::verbosity >= 1)
                          pcout
                            << "----------End density perturbation solve with approx density= n+lamda*v1-------------"
                            << std::endl;

                        temp1Vec.reinit(shadowKSRhoMin);
                        if (dftParameters::useAtomicRhoXLBOMD)
                          {
                            dftPtr->l2ProjectionQuadDensityMinusAtomicDensity(
                              dftPtr->d_matrixFreeDataPRefined,
                              dftPtr->d_constraintsRhoNodal,
                              dftPtr->d_densityDofHandlerIndexElectro,
                              dftPtr->d_densityQuadratureIdElectro,
                              *(dftPtr->rhoOutValues),
                              temp1Vec);
                          }
                        else
                          temp1Vec = dftPtr->d_rhoOutNodalValues;

                        peturbedApproxDensity = approxDensityContainer.back();
                        peturbedApproxDensity.add(-deltalambda,
                                                  vcontainer[irank]);

                        peturbedApproxDensity.update_ghost_values();

                        // normalize peturbedApproxDensity
                        charge =
                          dftPtr->totalCharge(dftPtr->d_matrixFreeDataPRefined,
                                              peturbedApproxDensity);


                        if (dftParameters::verbosity >= 1)
                          pcout
                            << "Total Charge before Normalizing peturbedApproxDensity:  "
                            << charge << std::endl;

                        if (dftParameters::useAtomicRhoXLBOMD)
                          peturbedApproxDensity.add(-charge /
                                                    dftPtr->d_domainVolume);
                        else
                          peturbedApproxDensity *=
                            ((double)dftPtr->numElectrons) / charge;

                        peturbedApproxDensity.update_ghost_values();
                        dftPtr->interpolateRhoNodalDataToQuadratureDataGeneral(
                          dftPtr->d_matrixFreeDataPRefined,
                          dftPtr->d_densityDofHandlerIndexElectro,
                          dftPtr->d_densityQuadratureIdElectro,
                          peturbedApproxDensity,
                          *(dftPtr->rhoInValues),
                          *(dftPtr->gradRhoInValues),
                          *(dftPtr->gradRhoInValues),
                          dftParameters::xcFamilyType == "GGA");

                        if (dftParameters::useAtomicRhoXLBOMD)
                          dftPtr->addAtomicRhoQuadValuesGradients(
                            *(dftPtr->rhoInValues),
                            *(dftPtr->gradRhoInValues),
                            dftParameters::xcFamilyType == "GGA");


                        dftPtr->normalizeRhoInQuadValues();

                        if (dftParameters::verbosity >= 1)
                          pcout
                            << "----------Start density perturbation solve with approx density= n-lamda*v1-------------"
                            << std::endl;


                        if (dftParameters::
                              useDensityMatrixPerturbationRankUpdates)
                          dftPtr->computeDensityPerturbation();
                        else
                          dftPtr->solve(false, false, true, false);

                        if (dftParameters::verbosity >= 1)
                          pcout
                            << "----------End density perturbation solve with approx density= n-lamda*v1-------------"
                            << std::endl;

                        temp2Vec.reinit(shadowKSRhoMin);
                        if (dftParameters::useAtomicRhoXLBOMD)
                          {
                            dftPtr->l2ProjectionQuadDensityMinusAtomicDensity(
                              dftPtr->d_matrixFreeDataPRefined,
                              dftPtr->d_constraintsRhoNodal,
                              dftPtr->d_densityDofHandlerIndexElectro,
                              dftPtr->d_densityQuadratureIdElectro,
                              *(dftPtr->rhoOutValues),
                              temp2Vec);
                          }
                        else
                          temp2Vec = dftPtr->d_rhoOutNodalValues;

                        ucontainer[irank].reinit(shadowKSRhoMin);
                        ucontainer[irank] = 0;
                        for (unsigned int i = 0; i < local_size; i++)
                          ucontainer[irank].local_element(i) =
                            (temp1Vec.local_element(i) -
                             temp2Vec.local_element(i)) /
                            (2.0 * deltalambda);

                        if (dftParameters::verbosity >= 1)
                          pcout
                            << " Vector norm of response (delta rho_min[n+delta_lambda*v1]/ delta_lambda):  "
                            << ucontainer[irank].l2_norm()
                            << " for kernel rank: " << irank + 1 << std::endl;

                        internalmd::recursiveJApply(ucontainer,
                                                    vcontainer,
                                                    k0kernelconstant,
                                                    vcontainer[irank],
                                                    irank,
                                                    jv);

                        ucontainer[irank] -= jv;
                      }

                    internalmd::recursiveKernelApply(ucontainer,
                                                     vcontainer,
                                                     k0,
                                                     rhoErrorVec,
                                                     ucontainer.size(),
                                                     kernelAction);

                    dftParameters::chebyshevFilterTolXLBOMD = temp1p;
                  }

                rmsErrorRho =
                  std::sqrt(dftPtr->rhofieldl2Norm(
                              dftPtr->d_matrixFreeDataPRefined,
                              rhoErrorVec,
                              dftPtr->d_densityDofHandlerIndexElectro,
                              dftPtr->d_densityQuadratureIdElectro) /
                            dftPtr->d_domainVolume);
                rmsErrorGradRho = std::sqrt(
                  dftPtr->fieldGradl2Norm(dftPtr->d_matrixFreeDataPRefined,
                                          rhoErrorVec) /
                  dftPtr->d_domainVolume);
                MPI_Barrier(MPI_COMM_WORLD);
                xlbomdpost_time = MPI_Wtime() - xlbomdpost_time;
                if (dftParameters::verbosity >= 1)
                  pcout << "Time taken for xlbomd post solve operations: "
                        << xlbomdpost_time << std::endl;



                if (dftParameters::chkType == 3 && !xlbomdHistoryRestart)
                  {
                    dftPtr->d_groundStateDensityHistory.pop_front();

                    dftPtr->d_groundStateDensityHistory.push_back(
                      tempVecRestart);

                    for (unsigned int idof = 0;
                         idof < approxDensityContainer[kmax - 1].local_size();
                         idof++)
                      dftPtr->d_groundStateDensityHistory[kmax - 1]
                        .local_element(idof) =
                        approxDensityContainer[kmax - 1].local_element(idof);

                    dftPtr->d_groundStateDensityHistory[kmax - 1]
                      .update_ghost_values();

                    writeDensityHistory = true;
                  }

                isFirstXLBOMDStep = false;
              }
          }
        else
          {
            if (!(timeIndex == startingTimeStep + 1 && restartFlag == 1))
              {
                dftPtr->interpolateRhoNodalDataToQuadratureDataGeneral(
                  dftPtr->d_matrixFreeDataPRefined,
                  dftPtr->d_densityDofHandlerIndexElectro,
                  dftPtr->d_densityQuadratureIdElectro,
                  shadowKSRhoMin,
                  *(dftPtr->rhoInValues),
                  *(dftPtr->gradRhoInValues),
                  *(dftPtr->gradRhoInValues),
                  dftParameters::xcFamilyType == "GGA");

                if (dftParameters::useAtomicRhoXLBOMD)
                  dftPtr->addAtomicRhoQuadValuesGradients(
                    *(dftPtr->rhoInValues),
                    *(dftPtr->gradRhoInValues),
                    dftParameters::xcFamilyType == "GGA");

                dftPtr->normalizeRhoInQuadValues();

                dftPtr->l2ProjectionQuadToNodal(
                  dftPtr->d_matrixFreeDataPRefined,
                  dftPtr->d_constraintsRhoNodal,
                  dftPtr->d_densityDofHandlerIndexElectro,
                  dftPtr->d_densityQuadratureIdElectro,
                  *(dftPtr->rhoInValues),
                  dftPtr->d_rhoInNodalValues);

                dftPtr->d_rhoInNodalValues.update_ghost_values();
              }

            //
            // do an scf calculation
            //
            dftPtr->solve();

            shadowKSRhoMin = dftPtr->d_rhoOutNodalValues;
            if (dftParameters::useAtomicRhoXLBOMD)
              {
                dftPtr->l2ProjectionQuadDensityMinusAtomicDensity(
                  dftPtr->d_matrixFreeDataPRefined,
                  dftPtr->d_constraintsRhoNodal,
                  dftPtr->d_densityDofHandlerIndexElectro,
                  dftPtr->d_densityQuadratureIdElectro,
                  *(dftPtr->rhoOutValues),
                  shadowKSRhoMin);
              }

            shadowKSRhoMin.update_ghost_values();
          }

        double bomdpost_time;
        MPI_Barrier(MPI_COMM_WORLD);
        bomdpost_time = MPI_Wtime();

        //
        // get force field using Gaussians
        //
        const std::vector<double> forceOnAtoms =
          dftPtr->forcePtr->getAtomsForces();

        //
        // compute acceleration at t + delta t
        //
        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            accelerationNew[3 * iCharge + 0] =
              -(forceOnAtoms[3 * iCharge + 0] * haPerBohrToeVPerAng) /
              massAtoms[iCharge];
            accelerationNew[3 * iCharge + 1] =
              -(forceOnAtoms[3 * iCharge + 1] * haPerBohrToeVPerAng) /
              massAtoms[iCharge];
            accelerationNew[3 * iCharge + 2] =
              -(forceOnAtoms[3 * iCharge + 2] * haPerBohrToeVPerAng) /
              massAtoms[iCharge];
          }

        // compute velocity at t + delta t
        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            velocity[3 * iCharge + 0] =
              velocityMid[3 * iCharge + 0] +
              0.5 * accelerationNew[3 * iCharge + 0] * timeStep;
            velocity[3 * iCharge + 1] =
              velocityMid[3 * iCharge + 1] +
              0.5 * accelerationNew[3 * iCharge + 1] * timeStep;
            velocity[3 * iCharge + 2] =
              velocityMid[3 * iCharge + 2] +
              0.5 * accelerationNew[3 * iCharge + 2] * timeStep;
          }

        kineticEnergy = 0.0;
        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            kineticEnergy +=
              0.5 * massAtoms[iCharge] *
              (velocity[3 * iCharge + 0] * velocity[3 * iCharge + 0] +
               velocity[3 * iCharge + 1] * velocity[3 * iCharge + 1] +
               velocity[3 * iCharge + 2] * velocity[3 * iCharge + 2]);
          }

        averageKineticEnergy      = kineticEnergy / (3 * numberGlobalCharges);
        temperatureFromVelocities = averageKineticEnergy * 2 / kb;
        kineticEnergyVector[timeIndex - startingTimeStep] =
          kineticEnergy / haToeV;
        internalEnergyVector[timeIndex - startingTimeStep] =
          isXlBOMDStep ? shadowPotentialInternalEnergy :
                         dftPtr->d_groundStateEnergy;
        entropicEnergyVector[timeIndex - startingTimeStep] =
          isXlBOMDStep ? entropicEnergyShadowPotential :
                         dftPtr->d_entropicEnergy;
        totalEnergyVector[timeIndex - startingTimeStep] =
          kineticEnergyVector[timeIndex - startingTimeStep] +
          internalEnergyVector[timeIndex - startingTimeStep] -
          entropicEnergyVector[timeIndex - startingTimeStep];

        rmsErrorRhoVector[timeIndex - startingTimeStep]     = rmsErrorRho;
        rmsErrorGradRhoVector[timeIndex - startingTimeStep] = rmsErrorGradRho;

        //
        // reset acceleration at time t based on the acceleration at previous
        // time step
        //
        acceleration = accelerationNew;

        pcout << " Temperature from velocities: " << timeIndex << " "
              << temperatureFromVelocities << std::endl;
        pcout << " Kinetic Energy in Ha at timeIndex " << timeIndex << " "
              << kineticEnergyVector[timeIndex - startingTimeStep] << std::endl;
        pcout << " Internal Energy in Ha at timeIndex " << timeIndex << " "
              << internalEnergyVector[timeIndex - startingTimeStep]
              << std::endl;
        pcout << " Entropic Energy in Ha at timeIndex " << timeIndex << " "
              << entropicEnergyVector[timeIndex - startingTimeStep]
              << std::endl;
        pcout << " Total Energy in Ha at timeIndex " << timeIndex << " "
              << totalEnergyVector[timeIndex - startingTimeStep] << std::endl;
        if (dftParameters::isXLBOMD)
          {
            pcout << " RMS error in rho in a.u. at timeIndex " << timeIndex
                  << " " << rmsErrorRhoVector[timeIndex - startingTimeStep]
                  << std::endl;
            pcout << " RMS error in grad rho in a.u. at timeIndex " << timeIndex
                  << " " << rmsErrorGradRhoVector[timeIndex - startingTimeStep]
                  << std::endl;
          }

        std::vector<std::vector<double>> data1(timeIndex + 1,
                                               std::vector<double>(1, 0.0));
        std::vector<std::vector<double>> data2(timeIndex + 1,
                                               std::vector<double>(1, 0.0));
        std::vector<std::vector<double>> data3(timeIndex + 1,
                                               std::vector<double>(1, 0.0));
        std::vector<std::vector<double>> data4(timeIndex + 1,
                                               std::vector<double>(1, 0.0));
        std::vector<std::vector<double>> data5(timeIndex + 1,
                                               std::vector<double>(1, 0.0));
        std::vector<std::vector<double>> data6(timeIndex + 1,
                                               std::vector<double>(1, 0.0));

        if (restartFlag == 1)
          {
            std::vector<std::vector<double>> kineticEnergyData;
            dftUtils::readFile(1, kineticEnergyData, "KeEngMd");

            std::vector<std::vector<double>> internalEnergyData;
            dftUtils::readFile(1, internalEnergyData, "IntEngMd");

            std::vector<std::vector<double>> entropicEnergyData;
            dftUtils::readFile(1, entropicEnergyData, "EntEngMd");

            std::vector<std::vector<double>> totalEnergyData;
            dftUtils::readFile(1, totalEnergyData, "TotEngMd");


            std::vector<std::vector<double>> rmsErrorRhoData;
            std::vector<std::vector<double>> rmsErrorGradRhoData;
            if (dftParameters::isXLBOMD)
              {
                dftUtils::readFile(1, rmsErrorRhoData, "RMSErrorRhoMd");

                dftUtils::readFile(1, rmsErrorGradRhoData, "RMSErrorGradRhoMd");
              }
            for (int i = 0; i <= startingTimeStep; ++i)
              {
                data1[i][0] = kineticEnergyData[i][0];
                data2[i][0] = internalEnergyData[i][0];
                data3[i][0] = entropicEnergyData[i][0];
                data4[i][0] = totalEnergyData[i][0];
                if (dftParameters::isXLBOMD)
                  {
                    data5[i][0] = rmsErrorRhoData[i][0];
                    data6[i][0] = rmsErrorGradRhoData[i][0];
                  }
              }
          }
        else
          {
            data1[0][0] = kineticEnergyVector[0];
            data2[0][0] = internalEnergyVector[0];
            data3[0][0] = entropicEnergyVector[0];
            data4[0][0] = totalEnergyVector[0];
            if (dftParameters::isXLBOMD)
              {
                data5[0][0] = rmsErrorRhoVector[0];
                data6[0][0] = rmsErrorGradRhoVector[0];
              }
          }

        for (int i = startingTimeStep + 1; i <= timeIndex; ++i)
          {
            data1[i][0] = kineticEnergyVector[i - startingTimeStep];
            data2[i][0] = internalEnergyVector[i - startingTimeStep];
            data3[i][0] = entropicEnergyVector[i - startingTimeStep];
            data4[i][0] = totalEnergyVector[i - startingTimeStep];
            if (dftParameters::isXLBOMD)
              {
                data5[i][0] = rmsErrorRhoVector[i - startingTimeStep];
                data6[i][0] = rmsErrorGradRhoVector[i - startingTimeStep];
              }
          }
        MPI_Barrier(MPI_COMM_WORLD);
        dftUtils::writeDataIntoFile(data1, "KeEngMd");


        dftUtils::writeDataIntoFile(data2, "IntEngMd");


        dftUtils::writeDataIntoFile(data3, "EntEngMd");

        dftUtils::writeDataIntoFile(data4, "TotEngMd");

        if (dftParameters::isXLBOMD)
          {
            dftUtils::writeDataIntoFile(data5, "RMSErrorRhoMd");
            dftUtils::writeDataIntoFile(data6, "RMSErrorGradRhoMd");
          }

        /// write velocity and acceleration data
        std::vector<std::vector<double>> fileAccData(numberGlobalCharges,
                                                     std::vector<double>(3,
                                                                         0.0));
        std::vector<std::vector<double>> fileVelData(numberGlobalCharges,
                                                     std::vector<double>(3,
                                                                         0.0));
        std::vector<std::vector<double>> fileTemperatureData(
          1, std::vector<double>(1, 0.0));
        std::vector<std::vector<double>> timeIndexData(
          1, std::vector<double>(1, 0.0));

        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            fileAccData[iCharge][0] = accelerationNew[3 * iCharge + 0];
            fileAccData[iCharge][1] = accelerationNew[3 * iCharge + 1];
            fileAccData[iCharge][2] = accelerationNew[3 * iCharge + 2];
          }

        for (int iCharge = 0; iCharge < numberGlobalCharges; ++iCharge)
          {
            fileVelData[iCharge][0] = velocity[3 * iCharge + 0];
            fileVelData[iCharge][1] = velocity[3 * iCharge + 1];
            fileVelData[iCharge][2] = velocity[3 * iCharge + 2];
          }
        dftUtils::writeDataIntoFile(fileAccData, "acceleration.chk");
        dftUtils::writeDataIntoFile(fileVelData, "velocity.chk");

        fileTemperatureData[0][0] = temperatureFromVelocities;
        dftUtils::writeDataIntoFile(fileTemperatureData, "temperature.chk");

        timeIndexData[0][0] = timeIndex;
        dftUtils::writeDataIntoFile(timeIndexData, "time.chk");

        if (dftParameters::chkType >= 1)
          dftPtr->writeDomainAndAtomCoordinates();


        MPI_Barrier(MPI_COMM_WORLD);
        bomdpost_time = MPI_Wtime() - bomdpost_time;
        if (dftParameters::verbosity >= 1)
          pcout << "Time taken for bomd post solve operations: "
                << bomdpost_time << std::endl;

        MPI_Barrier(MPI_COMM_WORLD);
        step_time = MPI_Wtime() - step_time;
        if (dftParameters::verbosity >= 1)
          pcout << "Time taken for md step: " << step_time << std::endl;

        if (dftParameters::chkType == 3 && writeDensityHistory &&
            !xlbomdHistoryRestart && !dftParameters::restartFromChk)
          {
            dftPtr->saveTriaInfoAndRhoNodalData();
          }

        if (timeIndex == (fullScfSolvesBeforeStartingXLBOMD) &&
            xlbomdHistoryRestart)
          xlbomdHistoryRestart = false;
      }
  }

#include "md.inst.cc"
} // namespace dftfe
