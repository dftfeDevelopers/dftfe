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

#include <cgPRPNonLinearSolver.h>
#include <BFGSNonLinearSolver.h>
#include <LBFGSNonLinearSolver.h>
#include <dft.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include <force.h>
#include <geoOptCell.h>
#include <sys/stat.h>

namespace dftfe
{
  //
  // constructor
  //

  geoOptCell::geoOptCell(dftBase *       dftPtr,
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
  geoOptCell::init(const std::string &restartPath)
  {
    d_restartPath   = restartPath + "/cellRelax";
    d_solverRestart = d_isRestart;
    if (d_dftPtr->getParametersObject().cellOptSolver == "BFGS")
      d_solver = 0;
    else if (d_dftPtr->getParametersObject().cellOptSolver == "LBFGS")
      d_solver = 1;
    else if (d_dftPtr->getParametersObject().cellOptSolver == "CGPRP")
      d_solver = 2;
    // initialize d_strainEpsilon to identity
    d_strainEpsilon = 0;
    for (unsigned int i = 0; i < 3; ++i)
      d_strainEpsilon[i][i] = 1.0;

    d_domainVolumeInitial = d_dftPtr->getCellVolume();

    // strain tensor is a symmetric second order with six independent components
    d_relaxationFlags.clear();
    d_relaxationFlags.resize(6, 0);

    if (d_dftPtr->getParametersObject().cellConstraintType ==
        1) //(isotropic shape fixed isotropic volume optimization)
      {
        d_relaxationFlags[0] = 1; //(epsilon_11+epsilon22+epsilon_33)/3
      }
    else if (d_dftPtr->getParametersObject().cellConstraintType ==
             2) //(volume fixed shape optimization)
      {
        d_relaxationFlags[1] = 1; // epsilon_12
        d_relaxationFlags[2] = 1; // epsilon_13
        d_relaxationFlags[4] = 1; // epsilon_23
      }
    else if (d_dftPtr->getParametersObject().cellConstraintType ==
             3) // (relax only cell component v1_x)
      {
        d_relaxationFlags[0] = 1; // epsilon_11
      }
    else if (d_dftPtr->getParametersObject().cellConstraintType ==
             4) // (relax only cell component v2_y)
      {
        d_relaxationFlags[3] = 1; // epsilon_22
      }
    else if (d_dftPtr->getParametersObject().cellConstraintType ==
             5) // (relax only cell component v3_z)
      {
        d_relaxationFlags[5] = 1; // epsilon_33
      }
    else if (d_dftPtr->getParametersObject().cellConstraintType ==
             6) // (relax only cell components v2_y and v3_z)
      {
        d_relaxationFlags[3] = 1; // epsilon_22
        d_relaxationFlags[5] = 1; // epsilon_33
      }
    else if (d_dftPtr->getParametersObject().cellConstraintType ==
             7) // (relax only cell components v1_x and v3_z)
      {
        d_relaxationFlags[0] = 1; // epsilon_11
        d_relaxationFlags[5] = 1; // epsilon_33
      }
    else if (d_dftPtr->getParametersObject().cellConstraintType ==
             8) // (relax only cell components v1_x and v2_y)
      {
        d_relaxationFlags[0] = 1; // epsilon_11
        d_relaxationFlags[3] = 1; // epsilon_22
      }
    else if (d_dftPtr->getParametersObject().cellConstraintType ==
             9) //(relax v1_x, v2_y and v3_z)
      {
        d_relaxationFlags[0] = 1; // epsilon_11
        d_relaxationFlags[3] = 1; // epsilon_22
        d_relaxationFlags[5] = 1; // epsilon_33
      }
    else if (d_dftPtr->getParametersObject().cellConstraintType ==
             10) //(2D only x and y components relaxed)
      {
        d_relaxationFlags[0] = 1; // epsilon_11
        d_relaxationFlags[1] = 1; // epsilon_12
        d_relaxationFlags[3] = 1; // epsilon_22
      }
    else if (d_dftPtr->getParametersObject().cellConstraintType ==
             11) //(2D only x and y shape components- inplane area fixed)
      {
        d_relaxationFlags[1] = 1; // epsilon_12
      }
    else if (d_dftPtr->getParametersObject().cellConstraintType ==
             12) // (all cell components relaxed)
      {
        // all six epsilon components
        d_relaxationFlags[0] = 1;
        d_relaxationFlags[1] = 1;
        d_relaxationFlags[2] = 1;
        d_relaxationFlags[3] = 1;
        d_relaxationFlags[4] = 1;
        d_relaxationFlags[5] = 1;
      }
    else if (d_dftPtr->getParametersObject().cellConstraintType ==
             13) //(automatically decides constraints based on boundary
                 // conditions)
      {
        d_relaxationFlags[0] = 1;
        d_relaxationFlags[1] = 1;
        d_relaxationFlags[2] = 1;
        d_relaxationFlags[3] = 1;
        d_relaxationFlags[4] = 1;
        d_relaxationFlags[5] = 1;

        if (!d_dftPtr->getParametersObject().periodicX)
          {
            d_relaxationFlags[0] = 0; // epsilon_11
            d_relaxationFlags[1] = 0; // epsilon_12
            d_relaxationFlags[2] = 0; // epsilon_13
          }

        if (!d_dftPtr->getParametersObject().periodicY)
          {
            d_relaxationFlags[1] = 0; // epsilon_12
            d_relaxationFlags[3] = 0; // epsilon_22
            d_relaxationFlags[4] = 0; // epsilon_23
          }

        if (!d_dftPtr->getParametersObject().periodicZ)
          {
            d_relaxationFlags[2] = 0; // epsilon_13
            d_relaxationFlags[4] = 0; // epsilon_23
            d_relaxationFlags[5] = 0; // epislon_33
          }
      }
    else
      {
        AssertThrow(
          false,
          dealii::ExcMessage(
            "The given value for CELL CONSTRAINT TYPE doesn't match with any available options (1-13)."));
      }
    if (d_isRestart)
      {
        std::vector<std::vector<double>> tmp, cellOptData;
        dftUtils::readFile(1, cellOptData, d_restartPath + "/cellOpt.dat");
        dftUtils::readFile(1, tmp, d_restartPath + "/step.chk");
        int solver             = cellOptData[0][0];
        int cellConstraintType = cellOptData[1][0];
        d_domainVolumeInitial  = cellOptData[2][0];
        d_totalUpdateCalls     = tmp[0][0];
        if (solver != d_solver)
          pcout
            << "Solver has changed since last save, the newly set solver will start from scratch."
            << std::endl;
        if (cellConstraintType !=
            d_dftPtr->getParametersObject().cellConstraintType)
          pcout
            << "Cell constraints have changed since last save, the solver will be reset to work with the new constraints."
            << std::endl;
        d_solverRestart =
          (cellConstraintType ==
           d_dftPtr->getParametersObject().cellConstraintType) &&
          (solver == d_solver);
        d_solverRestartPath =
          d_restartPath + "/step" + std::to_string(d_totalUpdateCalls);
        if (!d_solverRestart)
          {
            d_dftPtr->solve(true, true);

            if (d_dftPtr->getParametersObject()
                  .writeStructreEnergyForcesFileForPostProcess)
              {
                std::string fileName =
                  "structureEnergyForcesGSData_cellRelaxStep" +
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
        std::vector<std::vector<double>> cellOptData(3,
                                                     std::vector<double>(1,
                                                                         0.0));
        cellOptData[0][0] = d_solver;
        cellOptData[1][0] = d_dftPtr->getParametersObject().cellConstraintType;
        cellOptData[2][0] = d_domainVolumeInitial;
        if (!d_dftPtr->getParametersObject().reproducible_output)
          dftUtils::writeDataIntoFile(cellOptData,
                                      d_restartPath + "/cellOpt.dat",
                                      mpi_communicator);
      }
    if (d_solver == 0)
      d_nonLinearSolverPtr = std::make_unique<BFGSNonLinearSolver>(
        d_dftPtr->getParametersObject().usePreconditioner,
        d_dftPtr->getParametersObject().bfgsStepMethod == "RFO",
        d_dftPtr->getParametersObject().maxOptIter,
        d_dftPtr->getParametersObject().verbosity,
        mpi_communicator,
        d_dftPtr->getParametersObject().maxCellUpdateStep);
    else if (d_solver == 1)
      d_nonLinearSolverPtr = std::make_unique<LBFGSNonLinearSolver>(
        d_dftPtr->getParametersObject().usePreconditioner,
        d_dftPtr->getParametersObject().maxCellUpdateStep,
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
        d_dftPtr->getParametersObject().maxCellUpdateStep);

    if (d_dftPtr->getParametersObject().verbosity >= 2)
      {
        pcout << " --------------Cell relaxation flags----------------"
              << std::endl;
        pcout << " [0,0] " << d_relaxationFlags[0] << ", [0,1] "
              << d_relaxationFlags[1] << " [0,2] " << d_relaxationFlags[2]
              << ", [1,1] " << d_relaxationFlags[3] << ", [1,2] "
              << d_relaxationFlags[4] << ", [2,2] " << d_relaxationFlags[5]
              << std::endl;
        pcout << " --------------------------------------------------"
              << std::endl;
      }
    if (d_dftPtr->getParametersObject().verbosity >= 1)
      {
        if (d_solver == 0)
          {
            pcout << "   ---Non-linear BFGS Parameters-----------  "
                  << std::endl;
            pcout << "      stopping tol: "
                  << d_dftPtr->getParametersObject().stressRelaxTol
                  << std::endl;

            pcout << "      maxIter: "
                  << d_dftPtr->getParametersObject().maxOptIter << std::endl;

            pcout << "      preconditioner: "
                  << d_dftPtr->getParametersObject().usePreconditioner
                  << std::endl;

            pcout << "      step method: "
                  << d_dftPtr->getParametersObject().bfgsStepMethod
                  << std::endl;

            pcout << "      maxiumum step length: "
                  << d_dftPtr->getParametersObject().maxCellUpdateStep
                  << std::endl;


            pcout << "   -----------------------------------------  "
                  << std::endl;
          }
        if (d_solver == 1)
          {
            pcout << "   ---Non-linear LBFGS Parameters----------  "
                  << std::endl;
            pcout << "      stopping tol: "
                  << d_dftPtr->getParametersObject().stressRelaxTol
                  << std::endl;
            pcout << "      maxIter: "
                  << d_dftPtr->getParametersObject().maxOptIter << std::endl;
            pcout << "      preconditioner: "
                  << d_dftPtr->getParametersObject().usePreconditioner
                  << std::endl;
            pcout << "      lbfgs history: "
                  << d_dftPtr->getParametersObject().lbfgsNumPastSteps
                  << std::endl;
            pcout << "      maxiumum step length: "
                  << d_dftPtr->getParametersObject().maxCellUpdateStep
                  << std::endl;
            pcout << "   -----------------------------------------  "
                  << std::endl;
          }
        if (d_solver == 2)
          {
            pcout << "   ---Non-linear CG Parameters--------------  "
                  << std::endl;
            pcout << "      stopping tol: "
                  << d_dftPtr->getParametersObject().stressRelaxTol
                  << std::endl;
            pcout << "      maxIter: "
                  << d_dftPtr->getParametersObject().maxOptIter << std::endl;
            pcout << "      lineSearch tol: " << 1e-4 << std::endl;
            pcout << "      lineSearch maxIter: "
                  << d_dftPtr->getParametersObject().maxLineSearchIterCGPRP
                  << std::endl;
            pcout << "      lineSearch damping parameter: " << 0.8 << std::endl;
            pcout << "      maxiumum step length: "
                  << d_dftPtr->getParametersObject().maxCellUpdateStep
                  << std::endl;
            pcout << "   -----------------------------------------  "
                  << std::endl;
          }
        if (d_isRestart)
          {
            if (d_solver == 2)
              {
                pcout
                  << " Re starting Cell relaxation using nonlinear CG solver... "
                  << std::endl;
              }
            else if (d_solver == 0)
              {
                pcout
                  << " Re starting Cell relaxation using nonlinear BFGS solver... "
                  << std::endl;
              }
            else if (d_solver == 1)
              {
                pcout
                  << " Re starting Cell relaxation using nonlinear LBFGS solver... "
                  << std::endl;
              }
          }
        else
          {
            if (d_solver == 2)
              {
                pcout
                  << " Starting Cell relaxation using nonlinear CG solver... "
                  << std::endl;
              }
            else if (d_solver == 0)
              {
                pcout
                  << " Starting Cell relaxation using nonlinear BFGS solver... "
                  << std::endl;
              }
            else if (d_solver == 1)
              {
                pcout
                  << " Starting Cell relaxation using nonlinear LBFGS solver... "
                  << std::endl;
              }
          }
      }
    d_isRestart = false;
  }


  int
  geoOptCell::run()
  {
    if (getNumberUnknowns() > 0)
      {
        nonLinearSolver::ReturnValueType solverReturn =
          d_nonLinearSolverPtr->solve(*this,
                                      d_solverRestartPath + "/cellRelax.chk",
                                      d_solverRestart);


        if (solverReturn == nonLinearSolver::SUCCESS &&
            d_dftPtr->getParametersObject()
              .writeStructreEnergyForcesFileForPostProcess)
          {
            std::string fileName = "structureEnergyForcesGSDataCellRelaxed.txt";
            d_dftPtr->writeStructureEnergyForcesDataPostProcess(fileName);
          }


        if (solverReturn == nonLinearSolver::SUCCESS &&
            d_dftPtr->getParametersObject().verbosity >= 1)
          {
            pcout
              << " ...Cell stress relaxation completed as maximum stress magnitude is less than STRESS TOL: "
              << d_dftPtr->getParametersObject().stressRelaxTol
              << ", total number of cell geometry updates: "
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
                  << "------------------Fractional coordinates of atoms--------------------"
                  << std::endl;
                for (unsigned int i = 0;
                     i < d_dftPtr->getAtomLocationsCart().size();
                     ++i)
                  pcout << "AtomId " << i << ":  "
                        << d_dftPtr->getAtomLocationsFrac()[i][2] << " "
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
                    pcout << "AtomId " << i << ":  "
                          << d_dftPtr->getAtomLocationsCart()[i][2] << " "
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
        else if (solverReturn == nonLinearSolver::MAX_ITER_REACHED)
          {
            pcout << " ...Maximum iterations reached " << std::endl;
            d_totalUpdateCalls = -2;
          }
        else if (solverReturn == nonLinearSolver::FAILURE)
          {
            pcout << " ...Cell stress relaxation failed " << std::endl;
            d_totalUpdateCalls = -1;
          }
      }

    return d_totalUpdateCalls;
  }



  unsigned int
  geoOptCell::getNumberUnknowns() const
  {
    return std::accumulate(d_relaxationFlags.begin(),
                           d_relaxationFlags.end(),
                           0);
  }


  void
  geoOptCell::value(std::vector<double> &functionValue)
  {
    // AssertThrow(false,dftUtils::ExcNotImplementedYet());
    functionValue.clear();
    functionValue.push_back(d_dftPtr->getInternalEnergy());
  }


  void
  geoOptCell::gradient(std::vector<double> &gradient)
  {
    gradient.clear();
    const dealii::Tensor<2, 3, double> tempGradient =
      d_dftPtr->getCellVolume() *
      (d_dftPtr->getCellStress() * invert(d_strainEpsilon)) /
      d_domainVolumeInitial;

    if (d_relaxationFlags[0] == 1)
      gradient.push_back(tempGradient[0][0]);
    if (d_relaxationFlags[1] == 1)
      gradient.push_back(tempGradient[0][1]);
    if (d_relaxationFlags[2] == 1)
      gradient.push_back(tempGradient[0][2]);
    if (d_relaxationFlags[3] == 1)
      gradient.push_back(tempGradient[1][1]);
    if (d_relaxationFlags[4] == 1)
      gradient.push_back(tempGradient[1][2]);
    if (d_relaxationFlags[5] == 1)
      gradient.push_back(tempGradient[2][2]);

    if (d_dftPtr->getParametersObject().cellConstraintType ==
        1) // isotropic (shape fixed isotropic volume optimization)
      {
        gradient[0] =
          (tempGradient[0][0] + tempGradient[1][1] + tempGradient[2][2]) / 3.0;
      }
  }



  void
  geoOptCell::precondition(std::vector<double> &      s,
                           const std::vector<double> &gradient)
  {
    s.resize(getNumberUnknowns() * getNumberUnknowns(), 0.0);
    for (auto i = 0; i < getNumberUnknowns(); ++i)
      {
        s[i + i * getNumberUnknowns()] = 1.0;
      }
  }


  void
  geoOptCell::update(const std::vector<double> &solution,
                     const bool                 computeStress,
                     const bool useSingleAtomSolutionsInitialGuess)
  {
    std::vector<double> bcastSolution(solution.size());
    for (unsigned int i = 0; i < solution.size(); ++i)
      {
        bcastSolution[i] = solution[i];
      }

    // for synchronization
    MPI_Bcast(&(bcastSolution[0]),
              bcastSolution.size(),
              MPI_DOUBLE,
              0,
              mpi_communicator);

    dealii::Tensor<2, 3, double> strainEpsilonNew = d_strainEpsilon;

    unsigned int count = 0;
    if (d_relaxationFlags[0] == 1)
      {
        strainEpsilonNew[0][0] += bcastSolution[count];
        count++;
      }
    if (d_relaxationFlags[1] == 1)
      {
        strainEpsilonNew[0][1] += bcastSolution[count];
        strainEpsilonNew[1][0] += bcastSolution[count];
        count++;
      }
    if (d_relaxationFlags[2] == 1)
      {
        strainEpsilonNew[0][2] += bcastSolution[count];
        strainEpsilonNew[2][0] += bcastSolution[count];
        count++;
      }
    if (d_relaxationFlags[3] == 1)
      {
        strainEpsilonNew[1][1] += bcastSolution[count];
        count++;
      }
    if (d_relaxationFlags[4] == 1)
      {
        strainEpsilonNew[1][2] += bcastSolution[count];
        strainEpsilonNew[2][1] += bcastSolution[count];
        count++;
      }
    if (d_relaxationFlags[5] == 1)
      {
        strainEpsilonNew[2][2] += bcastSolution[count];
        count++;
      }


    if (d_dftPtr->getParametersObject().cellConstraintType ==
        1) // isotropic (shape fixed isotropic volume optimization)
      {
        strainEpsilonNew[1][1] = strainEpsilonNew[0][0];
        strainEpsilonNew[2][2] = strainEpsilonNew[0][0];
      }

    // To transform the domain under the strain we have to first do a inverse
    // transformation to bring the domain back to the unstrained state.
    dealii::Tensor<2, 3, double> deformationGradient =
      strainEpsilonNew * invert(d_strainEpsilon);
    d_strainEpsilon = strainEpsilonNew;

    // deform fem mesh and reinit
    d_dftPtr->deformDomain(deformationGradient,
                           false,
                           useSingleAtomSolutionsInitialGuess &&
                             !d_isScfRestart);


    d_dftPtr->solve(true, computeStress, d_isScfRestart);
    d_isScfRestart = false;
    d_totalUpdateCalls += 1;

    if (d_dftPtr->getParametersObject()
          .writeStructreEnergyForcesFileForPostProcess)
      {
        std::string fileName = "structureEnergyForcesGSData_cellRelaxStep" +
                               std::to_string(d_totalUpdateCalls) + ".txt";
        d_dftPtr->writeStructureEnergyForcesDataPostProcess(fileName);
      }
  }


  void
  geoOptCell::save()
  {
    if (!d_dftPtr->getParametersObject().reproducible_output)
      {
        std::vector<std::vector<double>> tmpData(1,
                                                 std::vector<double>(1, 0.0));
        std::string                      savePath =
          d_restartPath + "/step" + std::to_string(d_totalUpdateCalls) + "/";
        if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
          mkdir(savePath.c_str(), ACCESSPERMS);
        const dealii::Tensor<2, 3, double> tempGradient =
          d_dftPtr->getCellStress();
        d_dftPtr->writeDomainAndAtomCoordinates(savePath);
        d_nonLinearSolverPtr->save(savePath + "/cellRelax.chk");
        tmpData[0][0] = d_totalUpdateCalls;
        dftUtils::writeDataIntoFile(tmpData,
                                    d_restartPath + "/step.chk",
                                    mpi_communicator);
      }
  }

  bool
  geoOptCell::isConverged() const
  {
    bool                         converged    = true;
    dealii::Tensor<2, 3, double> tempGradient = d_dftPtr->getCellStress();
    if (tempGradient.norm() == 0)
      {
        return false;
      }
    std::vector<double> stress;
    if (d_relaxationFlags[0] == 1)
      stress.push_back(tempGradient[0][0]);
    if (d_relaxationFlags[1] == 1)
      stress.push_back(tempGradient[0][1]);
    if (d_relaxationFlags[2] == 1)
      stress.push_back(tempGradient[0][2]);
    if (d_relaxationFlags[3] == 1)
      stress.push_back(tempGradient[1][1]);
    if (d_relaxationFlags[4] == 1)
      stress.push_back(tempGradient[1][2]);
    if (d_relaxationFlags[5] == 1)
      stress.push_back(tempGradient[2][2]);

    if (d_dftPtr->getParametersObject().cellConstraintType ==
        1) // isotropic (shape fixed isotropic volume optimization)
      {
        stress[0] =
          (tempGradient[0][0] + tempGradient[1][1] + tempGradient[2][2]) / 3.0;
      }
    for (int i = 0; i < stress.size(); ++i)
      converged = converged && (std::abs(stress[i]) <
                                d_dftPtr->getParametersObject().stressRelaxTol);
    return converged;
  }

  const MPI_Comm &
  geoOptCell::getMPICommunicator()
  {
    return mpi_communicator;
  }


  void
  geoOptCell::solution(std::vector<double> &solution)
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }


  std::vector<unsigned int>
  geoOptCell::getUnknownCountFlag() const
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }

} // namespace dftfe
