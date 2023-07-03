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
// @author Sambit Das and Phani Motamarri
//

#include <cgPRPNonLinearSolver.h>
#include <BFGSNonLinearSolver.h>
#include <LBFGSNonLinearSolver.h>
#include <dft.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include <force.h>
#include <geoOptIon.h>
#include <sys/stat.h>


namespace dftfe
{
  //
  // constructor
  //

  geoOptIon::geoOptIon(dftBase *       dftPtr,
                       const MPI_Comm &mpi_comm_parent,
                       const bool      restart)
    : d_dftPtr(dftPtr)
    , mpi_communicator(mpi_comm_parent)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_parent))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , d_isRestart(restart)
  {
    d_isScfRestart = d_dftPtr->getParametersObject().loadRhoData;
  }

  //
  //

  void
  geoOptIon::init(const std::string &restartPath)
  {
    d_restartPath   = restartPath + "/ionRelax";
    d_solverRestart = d_isRestart;
    if (d_dftPtr->getParametersObject().ionOptSolver == "BFGS")
      d_solver = 0;
    else if (d_dftPtr->getParametersObject().ionOptSolver == "LBFGS")
      d_solver = 1;
    else if (d_dftPtr->getParametersObject().ionOptSolver == "CGPRP")
      d_solver = 2;
    const int numberGlobalAtoms = d_dftPtr->getAtomLocationsCart().size();
    if (d_dftPtr->getParametersObject().ionRelaxFlagsFile != "")
      {
        std::vector<std::vector<int>>    tempRelaxFlagsData;
        std::vector<std::vector<double>> tempForceData;
        dftUtils::readRelaxationFlagsFile(
          6,
          tempRelaxFlagsData,
          tempForceData,
          d_dftPtr->getParametersObject().ionRelaxFlagsFile);
        AssertThrow(tempRelaxFlagsData.size() == numberGlobalAtoms,
                    dealii::ExcMessage(
                      "Incorrect number of entries in relaxationFlags file"));
        d_relaxationFlags.clear();
        d_externalForceOnAtom.clear();
        for (unsigned int i = 0; i < numberGlobalAtoms; ++i)
          {
            for (unsigned int j = 0; j < 3; ++j)
              {
                d_relaxationFlags.push_back(tempRelaxFlagsData[i][j]);
                d_externalForceOnAtom.push_back(tempForceData[i][j]);
              }
          }
      }
    else
      {
        d_relaxationFlags.clear();
        d_externalForceOnAtom.clear();
        for (unsigned int i = 0; i < numberGlobalAtoms; ++i)
          {
            for (unsigned int j = 0; j < 3; ++j)
              {
                d_relaxationFlags.push_back(1.0);
                d_externalForceOnAtom.push_back(0.0);
              }
          }
      }
    if (d_isRestart)
      {
        std::vector<std::vector<double>> tmp, ionOptData;
        dftUtils::readFile(1, ionOptData, d_restartPath + "/ionOpt.dat");
        dftUtils::readFile(1, tmp, d_restartPath + "/step.chk");
        int  solver            = ionOptData[0][0];
        bool usePreconditioner = ionOptData[1][0] > 1e-6;
        d_totalUpdateCalls     = tmp[0][0];
        tmp.clear();
        dftUtils::readFile(1,
                           tmp,
                           d_restartPath + "/step" +
                             std::to_string(d_totalUpdateCalls) +
                             "/maxForce.chk");
        d_maximumAtomForceToBeRelaxed = tmp[0][0];
        d_relaxationFlags.resize(numberGlobalAtoms * 3);
        bool relaxationFlagsMatch = true;
        for (unsigned int i = 0; i < numberGlobalAtoms; ++i)
          {
            for (unsigned int j = 0; j < 3; ++j)
              {
                relaxationFlagsMatch = (d_relaxationFlags[i * 3 + j] ==
                                        (int)ionOptData[i * 3 + j + 2][0]) &&
                                       relaxationFlagsMatch;
              }
          }
        if (solver != d_solver ||
            usePreconditioner !=
              d_dftPtr->getParametersObject().usePreconditioner)
          pcout
            << "Solver has changed since last save, the newly set solver will start from scratch."
            << std::endl;
        if (!relaxationFlagsMatch)
          pcout
            << "Relaxations flags have changed since last save, the solver will be reset to work with the new flags."
            << std::endl;
        d_solverRestart = (relaxationFlagsMatch) && (solver == d_solver) &&
                          (usePreconditioner ==
                           d_dftPtr->getParametersObject().usePreconditioner);
        d_solverRestartPath =
          d_restartPath + "/step" + std::to_string(d_totalUpdateCalls);
        if (!d_solverRestart)
          {
            d_dftPtr->solve(true, false);

            if (d_dftPtr->getParametersObject()
                  .writeStructreEnergyForcesFileForPostProcess)
              {
                std::string fileName =
                  "structureEnergyForcesGSData_ionRelaxStep" +
                  std::to_string(d_totalUpdateCalls) + ".txt";
                d_dftPtr->writeStructureEnergyForcesDataPostProcess(fileName);
              }
          }
      }
    else
      {
        d_totalUpdateCalls = 0;
        if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
          mkdir(d_restartPath.c_str(), ACCESSPERMS);
        std::vector<std::vector<double>> ionOptData(2 + numberGlobalAtoms * 3,
                                                    std::vector<double>(1,
                                                                        0.0));
        ionOptData[0][0] = d_solver;
        ionOptData[1][0] =
          d_dftPtr->getParametersObject().usePreconditioner ? 1 : 0;
        for (unsigned int i = 0; i < numberGlobalAtoms; ++i)
          {
            for (unsigned int j = 0; j < 3; ++j)
              {
                ionOptData[i * 3 + j + 2][0] = d_relaxationFlags[i * 3 + j];
              }
          }
        if (!d_dftPtr->getParametersObject().reproducible_output)
          dftUtils::writeDataIntoFile(ionOptData,
                                      d_restartPath + "/ionOpt.dat",
                                      mpi_communicator);
      }
    if (d_solver == 0)
      d_nonLinearSolverPtr = std::make_unique<BFGSNonLinearSolver>(
        d_dftPtr->getParametersObject().usePreconditioner,
        d_dftPtr->getParametersObject().bfgsStepMethod == "RFO",
        d_dftPtr->getParametersObject().maxOptIter,
        d_dftPtr->getParametersObject().verbosity,
        mpi_communicator,
        d_dftPtr->getParametersObject().maxIonUpdateStep);
    else if (d_solver == 1)
      d_nonLinearSolverPtr = std::make_unique<LBFGSNonLinearSolver>(
        d_dftPtr->getParametersObject().usePreconditioner,
        d_dftPtr->getParametersObject().maxIonUpdateStep,
        d_dftPtr->getParametersObject().maxOptIter,
        d_dftPtr->getParametersObject().lbfgsNumPastSteps,
        d_dftPtr->getParametersObject().verbosity,
        mpi_communicator);
    else
      d_nonLinearSolverPtr = std::make_unique<cgPRPNonLinearSolver>(
        d_dftPtr->getParametersObject().maxOptIter,
        d_dftPtr->getParametersObject().verbosity,
        mpi_communicator,
        1e-4,
        d_dftPtr->getParametersObject().maxLineSearchIterCGPRP,
        0.8,
        d_dftPtr->getParametersObject().maxIonUpdateStep);
    // print relaxation flags
    if (d_dftPtr->getParametersObject().verbosity >= 1)
      {
        pcout << " --------------Ion force relaxation flags----------------"
              << std::endl;
        for (unsigned int i = 0; i < numberGlobalAtoms; ++i)
          {
            pcout << d_relaxationFlags[i * 3] << "  "
                  << d_relaxationFlags[i * 3 + 1] << "  "
                  << d_relaxationFlags[i * 3 + 2] << std::endl;
          }
        pcout << " --------------------------------------------------------"
              << std::endl;
      }
    if (d_dftPtr->getParametersObject().verbosity >= 1)
      {
        if (d_solver == 0)
          {
            pcout << "   ---Non-linear BFGS Parameters-----------  "
                  << std::endl;
            pcout << "      stopping tol: "
                  << d_dftPtr->getParametersObject().forceRelaxTol << std::endl;

            pcout << "      maxIter: "
                  << d_dftPtr->getParametersObject().maxOptIter << std::endl;

            pcout << "      preconditioner: "
                  << d_dftPtr->getParametersObject().usePreconditioner
                  << std::endl;

            pcout << "      step method: "
                  << d_dftPtr->getParametersObject().bfgsStepMethod
                  << std::endl;

            pcout << "      maxiumum step length: "
                  << d_dftPtr->getParametersObject().maxIonUpdateStep
                  << std::endl;


            pcout << "   -----------------------------------------  "
                  << std::endl;
          }
        if (d_solver == 1)
          {
            pcout << "   ---Non-linear LBFGS Parameters----------  "
                  << std::endl;
            pcout << "      stopping tol: "
                  << d_dftPtr->getParametersObject().forceRelaxTol << std::endl;
            pcout << "      maxIter: "
                  << d_dftPtr->getParametersObject().maxOptIter << std::endl;
            pcout << "      preconditioner: "
                  << d_dftPtr->getParametersObject().usePreconditioner
                  << std::endl;
            pcout << "      lbfgs history: "
                  << d_dftPtr->getParametersObject().lbfgsNumPastSteps
                  << std::endl;
            pcout << "      maxiumum step length: "
                  << d_dftPtr->getParametersObject().maxIonUpdateStep
                  << std::endl;
            pcout << "   -----------------------------------------  "
                  << std::endl;
          }
        if (d_solver == 2)
          {
            pcout << "   ---Non-linear CG Parameters--------------  "
                  << std::endl;
            pcout << "      stopping tol: "
                  << d_dftPtr->getParametersObject().forceRelaxTol << std::endl;
            pcout << "      maxIter: "
                  << d_dftPtr->getParametersObject().maxOptIter << std::endl;
            pcout << "      lineSearch tol: " << 1e-4 << std::endl;
            pcout << "      lineSearch maxIter: "
                  << d_dftPtr->getParametersObject().maxLineSearchIterCGPRP
                  << std::endl;
            pcout << "      lineSearch damping parameter: " << 0.8 << std::endl;
            pcout << "      maxiumum step length: "
                  << d_dftPtr->getParametersObject().maxIonUpdateStep
                  << std::endl;
            pcout << "   -----------------------------------------  "
                  << std::endl;
          }
        if (d_isRestart)
          {
            if (d_solver == 2)
              {
                pcout
                  << " Re starting Ion force relaxation using nonlinear CG solver... "
                  << std::endl;
              }
            else if (d_solver == 0)
              {
                pcout
                  << " Re starting Ion force relaxation using nonlinear BFGS solver... "
                  << std::endl;
              }
            else if (d_solver == 1)
              {
                pcout
                  << " Re starting Ion force relaxation using nonlinear LBFGS solver... "
                  << std::endl;
              }
          }
        else
          {
            if (d_solver == 2)
              {
                pcout
                  << " Starting Ion force relaxation using nonlinear CG solver... "
                  << std::endl;
              }
            else if (d_solver == 0)
              {
                pcout
                  << " Starting Ion force relaxation using nonlinear BFGS solver... "
                  << std::endl;
              }
            else if (d_solver == 1)
              {
                pcout
                  << " Starting Ion force relaxation using nonlinear LBFGS solver... "
                  << std::endl;
              }
          }
      }
    d_isRestart = false;
  }


  int
  geoOptIon::run()
  {
    if (getNumberUnknowns() > 0)
      {
        nonLinearSolver::ReturnValueType solverReturn =
          d_nonLinearSolverPtr->solve(*this,
                                      d_solverRestartPath + "/ionRelax.chk",
                                      d_solverRestart);

        if (solverReturn == nonLinearSolver::SUCCESS &&
            d_dftPtr->getParametersObject()
              .writeStructreEnergyForcesFileForPostProcess)
          {
            std::string fileName = "structureEnergyForcesGSDataIonRelaxed.txt";
            d_dftPtr->writeStructureEnergyForcesDataPostProcess(fileName);
          }


        if (solverReturn == nonLinearSolver::SUCCESS &&
            d_dftPtr->getParametersObject().verbosity >= 1)
          {
            pcout
              << " ...Ion force relaxation completed as maximum force magnitude is less than FORCE TOL: "
              << d_dftPtr->getParametersObject().forceRelaxTol
              << ", total number of ion position updates: "
              << d_totalUpdateCalls << std::endl;

            pcout
              << "-------------------------------Final Relaxed structure-----------------------------"
              << std::endl;
            pcout
              << "-----------------------------------------------------------------------------------"
              << std::endl;
            pcout
              << "-----------Simulation Domain bounding vectors (lattice vectors in fully periodic case)-------------"
              << std::endl;
            for (int i = 0; i < d_dftPtr->getCell().size(); ++i)
              {
                pcout << "v" << i + 1 << " : " << d_dftPtr->getCell()[i][0]
                      << " " << d_dftPtr->getCell()[i][1] << " "
                      << d_dftPtr->getCell()[i][2] << std::endl;
              }
            pcout
              << "-----------------------------------------------------------------------------------------"
              << std::endl;

            if (d_dftPtr->getParametersObject().periodicX ||
                d_dftPtr->getParametersObject().periodicY ||
                d_dftPtr->getParametersObject().periodicZ)
              {
                pcout
                  << "-------------------Fractional coordinates of atoms----------------------"
                  << std::endl;
                for (unsigned int i = 0;
                     i < d_dftPtr->getAtomLocationsCart().size();
                     ++i)
                  pcout << (unsigned int)d_dftPtr->getAtomLocationsFrac()[i][0]
                        << " "
                        << (unsigned int)d_dftPtr->getAtomLocationsFrac()[i][1]
                        << " " << d_dftPtr->getAtomLocationsFrac()[i][2] << " "
                        << d_dftPtr->getAtomLocationsFrac()[i][3] << " "
                        << d_dftPtr->getAtomLocationsFrac()[i][4] << "\n";
                pcout
                  << "-----------------------------------------------------------------------------------------"
                  << std::endl;
              }
            else
              {
                //
                // print cartesian coordinates
                //
                pcout
                  << "------------Cartesian coordinates of atoms (origin at center of domain)------------------"
                  << std::endl;
                for (unsigned int i = 0;
                     i < d_dftPtr->getAtomLocationsCart().size();
                     ++i)
                  {
                    pcout
                      << (unsigned int)d_dftPtr->getAtomLocationsCart()[i][0]
                      << " "
                      << (unsigned int)d_dftPtr->getAtomLocationsCart()[i][1]
                      << " " << d_dftPtr->getAtomLocationsCart()[i][2] << " "
                      << d_dftPtr->getAtomLocationsCart()[i][3] << " "
                      << d_dftPtr->getAtomLocationsCart()[i][4] << "\n";
                  }
                pcout
                  << "-----------------------------------------------------------------------------------------"
                  << std::endl;
              }
            pcout
              << "-----------------------------------------------------------------------------------"
              << std::endl;

            d_dftPtr->writeDomainAndAtomCoordinates("./");
          }
        else if (solverReturn == nonLinearSolver::FAILURE)
          {
            pcout << " ...Ion force relaxation failed " << std::endl;
            d_totalUpdateCalls = -1;
          }
        else if (solverReturn == nonLinearSolver::MAX_ITER_REACHED)
          {
            pcout << " ...Maximum iterations reached " << std::endl;
            d_totalUpdateCalls = -2;
          }
      }

    return d_totalUpdateCalls;
  }



  unsigned int
  geoOptIon::getNumberUnknowns() const
  {
    return std::accumulate(d_relaxationFlags.begin(),
                           d_relaxationFlags.end(),
                           0);
  }


  void
  geoOptIon::value(std::vector<double> &functionValue)
  {
    // AssertThrow(false,dftUtils::ExcNotImplementedYet());
    functionValue.clear();

    // Relative to initial free energy supressed in case of CGPRP
    // as that would not work in case of restarted CGPRP
    functionValue.push_back(d_dftPtr->getInternalEnergy());
  }


  void
  geoOptIon::gradient(std::vector<double> &gradient)
  {
    gradient.clear();
    const int numberGlobalAtoms = d_dftPtr->getAtomLocationsCart().size();
    const std::vector<double> tempGradient = d_dftPtr->getForceonAtoms();
    AssertThrow(tempGradient.size() == numberGlobalAtoms * 3,
                dealii::ExcMessage("Atom forces have wrong size"));
    for (unsigned int i = 0; i < numberGlobalAtoms; ++i)
      {
        for (unsigned int j = 0; j < 3; ++j)
          {
            if (d_relaxationFlags[3 * i + j] == 1)
              {
                gradient.push_back(tempGradient[3 * i + j] -
                                   d_externalForceOnAtom[3 * i + j]);
              }
          }
      }

    d_maximumAtomForceToBeRelaxed = -1.0;

    for (unsigned int i = 0; i < gradient.size(); ++i)
      {
        const double temp = std::sqrt(gradient[i] * gradient[i]);
        if (temp > d_maximumAtomForceToBeRelaxed)
          d_maximumAtomForceToBeRelaxed = temp;
      }
  }


  void
  geoOptIon::precondition(std::vector<double> &      s,
                          const std::vector<double> &gradient)
  {
    if (d_solverRestart)
      {
        std::vector<std::vector<double>> preconData;
        dftUtils::readFile(1,
                           preconData,
                           d_restartPath + "/preconditioner.dat");
        AssertThrow(preconData.size() ==
                      getNumberUnknowns() * getNumberUnknowns(),
                    dealii::ExcMessage(
                      "Incorrect preconditioner size in preconditioner.dat"));
        s.clear();
        s.resize(getNumberUnknowns() * getNumberUnknowns(), 0.0);
        for (int i = 0; i < preconData.size(); ++i)
          {
            s[i] = preconData[i][0];
          }
        d_solverRestart = false;
      }
    else
      {
        const int numberGlobalAtoms = d_dftPtr->getAtomLocationsCart().size();
        std::vector<std::vector<double>> NNdistances(numberGlobalAtoms);
        double                           rNN = 0;
        for (int i = 0; i < numberGlobalAtoms; ++i)
          {
            double riMin = 0;
            for (int j = 0; j < numberGlobalAtoms; ++j)
              {
                double rij = 0;
                for (int k = 2; k < 5; ++k)
                  {
                    rij += (d_dftPtr->getAtomLocationsCart()[i][k] -
                            d_dftPtr->getAtomLocationsCart()[j][k]) *
                           (d_dftPtr->getAtomLocationsCart()[i][k] -
                            d_dftPtr->getAtomLocationsCart()[j][k]);
                  }
                rij = std::sqrt(rij);
                if ((riMin > rij && i != j) || j == 0)
                  {
                    riMin = rij;
                  }
              }
            if (rNN < riMin)
              {
                rNN = riMin;
              }
          }
        double rCut = 2 * rNN;
        if (d_dftPtr->getParametersObject().verbosity >= 2)
          pcout << "Cutoff radius for preconditoner:" << rCut << std::endl;
        std::vector<double> L(numberGlobalAtoms * numberGlobalAtoms, 0.0);
        for (int i = 0; i < numberGlobalAtoms; ++i)
          {
            for (int j = i + 1; j < numberGlobalAtoms; ++j)
              {
                double rij = 0;
                for (int k = 2; k < 5; ++k)
                  {
                    rij += (d_dftPtr->getAtomLocationsCart()[i][k] -
                            d_dftPtr->getAtomLocationsCart()[j][k]) *
                           (d_dftPtr->getAtomLocationsCart()[i][k] -
                            d_dftPtr->getAtomLocationsCart()[j][k]);
                  }
                rij = std::sqrt(rij);
                if (rij < rCut)
                  {
                    L[i * numberGlobalAtoms + j] =
                      -std::exp(-3.0 * (rij / rNN - 1));
                    L[j * numberGlobalAtoms + i] = L[i * numberGlobalAtoms + j];
                  }
              }
          }
        for (int i = 0; i < numberGlobalAtoms; ++i)
          {
            for (int j = 0; j < numberGlobalAtoms; ++j)
              {
                if (i != j)
                  {
                    L[i * numberGlobalAtoms + i] -=
                      L[i * numberGlobalAtoms + j];
                  }
              }
            L[i * numberGlobalAtoms + i] += 0.1;
          }

        s.clear();
        s.resize(getNumberUnknowns() * getNumberUnknowns(), 0.0);
        int icount = 0;
        for (auto i = 0; i < numberGlobalAtoms; ++i)
          {
            for (auto k = 0; k < 3; ++k)
              {
                if (d_relaxationFlags[i * 3 + k] == 1)
                  {
                    int jcount = 0;
                    for (auto j = 0; j < numberGlobalAtoms; ++j)
                      {
                        for (auto l = 0; l < 3; ++l)
                          {
                            if (d_relaxationFlags[j * 3 + l] == 1)
                              {
                                s[icount * getNumberUnknowns() + jcount] =
                                  k == l ? L[i * numberGlobalAtoms + j] : 0.0;
                                ++jcount;
                              }
                          }
                      }
                    ++icount;
                  }
              }
          }
        std::vector<std::vector<double>> preconData(getNumberUnknowns() *
                                                      getNumberUnknowns(),
                                                    std::vector<double>(1, 0));
        for (int i = 0; i < preconData.size(); ++i)
          {
            preconData[i][0] = s[i];
          }
        if (!d_dftPtr->getParametersObject().reproducible_output)
          dftUtils::writeDataIntoFile(preconData,
                                      d_restartPath + "/preconditioner.dat",
                                      mpi_communicator);
      }
  }


  void
  geoOptIon::update(const std::vector<double> &solution,
                    const bool                 computeForces,
                    const bool useSingleAtomSolutionsInitialGuess)
  {
    const unsigned int numberGlobalAtoms =
      d_dftPtr->getAtomLocationsCart().size();
    std::vector<dealii::Tensor<1, 3, double>> globalAtomsDisplacements(
      numberGlobalAtoms);
    int count = 0;
    for (unsigned int i = 0; i < numberGlobalAtoms; ++i)
      {
        for (unsigned int j = 0; j < 3; ++j)
          {
            globalAtomsDisplacements[i][j] = 0.0;
            if (this_mpi_process == 0)
              {
                if (d_relaxationFlags[3 * i + j] == 1)
                  {
                    globalAtomsDisplacements[i][j] = solution[count];
                    count++;
                  }
              }
          }

        MPI_Bcast(&(globalAtomsDisplacements[i][0]),
                  3,
                  MPI_DOUBLE,
                  0,
                  mpi_communicator);
      }

    if (d_dftPtr->getParametersObject().verbosity >= 1)
      pcout << "  Maximum force component to be relaxed: "
            << d_maximumAtomForceToBeRelaxed << std::endl;

    double factor;
    if (d_maximumAtomForceToBeRelaxed >= 1e-03)
      factor = 2.00;
    else if (d_maximumAtomForceToBeRelaxed < 1e-03 &&
             d_maximumAtomForceToBeRelaxed >= 1e-04)
      factor = 1.25;
    else if (d_maximumAtomForceToBeRelaxed < 1e-04)
      factor = 1.15;

    d_dftPtr->updateAtomPositionsAndMoveMesh(
      globalAtomsDisplacements,
      factor,
      useSingleAtomSolutionsInitialGuess && !d_isScfRestart);


    /*if(d_maximumAtomForceToBeRelaxed >= 1e-02)
      d_dftPtr->getParametersObject().selfConsistentSolverTolerance = 1e-03;
      else if(d_maximumAtomForceToBeRelaxed >= 1e-03)
      d_dftPtr->getParametersObject().selfConsistentSolverTolerance = 1e-04;
      else if(d_maximumAtomForceToBeRelaxed >= 1e-04)
      d_dftPtr->getParametersObject().selfConsistentSolverTolerance = 1e-05;
      else if(d_maximumAtomForceToBeRelaxed >= 1e-05)
      d_dftPtr->getParametersObject().selfConsistentSolverTolerance = 5e-06;*/

    d_dftPtr->solve(computeForces, false, d_isScfRestart);
    d_isScfRestart = false;
    d_totalUpdateCalls += 1;

    if (d_dftPtr->getParametersObject()
          .writeStructreEnergyForcesFileForPostProcess)
      {
        std::string fileName = "structureEnergyForcesGSData_ionRelaxStep" +
                               std::to_string(d_totalUpdateCalls) + ".txt";
        d_dftPtr->writeStructureEnergyForcesDataPostProcess(fileName);
      }
  }


  void
  geoOptIon::save()
  {
    if (!d_dftPtr->getParametersObject().reproducible_output)
      {
        std::string savePath =
          d_restartPath + "/step" + std::to_string(d_totalUpdateCalls);
        if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
          mkdir(savePath.c_str(), ACCESSPERMS);
        const int numberGlobalAtoms = d_dftPtr->getAtomLocationsCart().size();
        const std::vector<double> tempGradient = d_dftPtr->getForceonAtoms();
        std::vector<std::vector<double>> forceData(1,
                                                   std::vector<double>(1, 0.0));
        forceData[0][0] = d_maximumAtomForceToBeRelaxed;
        dftUtils::writeDataIntoFile(forceData,
                                    savePath + "/maxForce.chk",
                                    mpi_communicator);
        d_dftPtr->writeDomainAndAtomCoordinates(savePath + "/");
        d_nonLinearSolverPtr->save(savePath + "/ionRelax.chk");
        std::vector<std::vector<double>> tmpData(1,
                                                 std::vector<double>(1, 0.0));
        tmpData[0][0] = d_totalUpdateCalls;
        dftUtils::writeDataIntoFile(tmpData,
                                    d_restartPath + "/step.chk",
                                    mpi_communicator);
      }
  }

  bool
  geoOptIon::isConverged() const
  {
    bool      converged              = true;
    const int numberGlobalAtoms      = d_dftPtr->getAtomLocationsCart().size();
    std::vector<double> tempGradient = d_dftPtr->getForceonAtoms();
    if (tempGradient.size() != numberGlobalAtoms * 3)
      {
        return false;
      }
    for (unsigned int i = 0; i < numberGlobalAtoms; ++i)
      {
        for (unsigned int j = 0; j < 3; ++j)
          {
            if (d_relaxationFlags[3 * i + j] == 1)
              {
                converged =
                  converged && (std::abs(tempGradient[3 * i + j] -
                                         d_externalForceOnAtom[3 * i + j]) <
                                d_dftPtr->getParametersObject().forceRelaxTol);
              }
          }
      }
    return converged;
  }

  const MPI_Comm &
  geoOptIon::getMPICommunicator()
  {
    return mpi_communicator;
  }


  void
  geoOptIon::solution(std::vector<double> &solution)
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }


  std::vector<unsigned int>
  geoOptIon::getUnknownCountFlag() const
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }

} // namespace dftfe
