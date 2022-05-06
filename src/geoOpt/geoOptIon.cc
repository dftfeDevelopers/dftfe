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
#include <cg_descent_wrapper.h>
#include <dft.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include <force.h>
#include <geoOptIon.h>


namespace dftfe
{
  //
  // constructor
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  geoOptIon<FEOrder, FEOrderElectro>::geoOptIon(
    dftClass<FEOrder, FEOrderElectro> *_dftPtr,
    const MPI_Comm &                   mpi_comm_parent)
    : dftPtr(_dftPtr)
    , mpi_communicator(mpi_comm_parent)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_comm_parent))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_comm_parent))
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0 &&
             !_dftPtr->getParametersObject().reproducible_output))
  {}

  //
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  geoOptIon<FEOrder, FEOrderElectro>::init()
  {
    const int numberGlobalAtoms = dftPtr->atomLocations.size();
    if (dftPtr->getParametersObject().ionRelaxFlagsFile != "")
      {
        std::vector<std::vector<int>>    tempRelaxFlagsData;
        std::vector<std::vector<double>> tempForceData;
        dftUtils::readRelaxationFlagsFile(
          6,
          tempRelaxFlagsData,
          tempForceData,
          dftPtr->getParametersObject().ionRelaxFlagsFile);
        AssertThrow(tempRelaxFlagsData.size() == numberGlobalAtoms,
                    ExcMessage(
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
        // print relaxation flags
        pcout << " --------------Ion force relaxation flags----------------"
              << std::endl;
        for (unsigned int i = 0; i < numberGlobalAtoms; ++i)
          {
            pcout << tempRelaxFlagsData[i][0] << "  "
                  << tempRelaxFlagsData[i][1] << "  "
                  << tempRelaxFlagsData[i][2] << std::endl;
          }
        pcout << " --------------------------------------------------"
              << std::endl;
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
        // print relaxation flags
        pcout << " --------------Ion force relaxation flags----------------"
              << std::endl;
        for (unsigned int i = 0; i < numberGlobalAtoms; ++i)
          {
            pcout << 1.0 << "  " << 1.0 << "  " << 1.0 << std::endl;
          }
        pcout << " --------------------------------------------------"
              << std::endl;
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  int
  geoOptIon<FEOrder, FEOrderElectro>::run()
  {
    const double tol =
      dftPtr->getParametersObject().forceRelaxTol; //(units: Hatree/Bohr)
    const unsigned int maxIter = 300;
    const double       lineSearchTol =
      1e-4; // Dummy parameter for CGPRP, the actual stopping criteria are the
            // Wolfe conditions and maxLineSearchIter
    const double       lineSearchDampingParameter = 0.8;
    const unsigned int maxLineSearchIter =
      dftPtr->getParametersObject().maxLineSearchIterCGPRP;
    const double       maxDisplacmentInAnyComponent = 0.5; // Bohr
    const unsigned int debugLevel =
      Utilities::MPI::this_mpi_process(mpi_communicator) == 0 ?
        dftPtr->getParametersObject().verbosity :
        0;

    d_totalUpdateCalls = 0;
    cgPRPNonLinearSolver cgSolver(tol,
                                  maxIter,
                                  debugLevel,
                                  mpi_communicator,
                                  lineSearchTol,
                                  maxLineSearchIter,
                                  lineSearchDampingParameter,
                                  maxDisplacmentInAnyComponent);
    BFGSNonLinearSolver  bfgsSolver(tol, maxIter, debugLevel, mpi_communicator);
    LBFGSNonLinearSolver lbfgsSolver(
      false, tol, maxIter, 2, debugLevel, mpi_communicator);

    CGDescent cg_descent(tol, maxIter);


    if (dftPtr->getParametersObject().chkType >= 1 &&
        dftPtr->getParametersObject().restartFromChk)
      pcout << "Re starting Ion force relaxation using nonlinear CG solver... "
            << std::endl;
    else
      pcout << "Starting Ion force relaxation using nonlinear CG solver... "
            << std::endl;
    if (dftPtr->getParametersObject().verbosity >= 2)
      {
        pcout << "   ---Non-linear CG Parameters--------------  " << std::endl;
        pcout << "      stopping tol: " << tol << std::endl;
        pcout << "      maxIter: " << maxIter << std::endl;
        pcout << "      lineSearch tol: " << lineSearchTol << std::endl;
        pcout << "      lineSearch maxIter: " << maxLineSearchIter << std::endl;
        pcout << "      lineSearch damping parameter: "
              << lineSearchDampingParameter << std::endl;
        pcout << "   ------------------------------  " << std::endl;
      }

    if (getNumberUnknowns() > 0)
      {
        nonLinearSolver::ReturnValueType cgReturn = nonLinearSolver::FAILURE;
        bool                             cgSuccess;

        if (dftPtr->getParametersObject().chkType >= 1 &&
            dftPtr->getParametersObject().restartFromChk &&
            dftPtr->getParametersObject().ionOptSolver == "CGPRP")
          cgReturn = cgSolver.solve(*this, std::string("ionRelaxCG.chk"), true);
        else if (dftPtr->getParametersObject().chkType >= 1 &&
                 !dftPtr->getParametersObject().restartFromChk &&
                 dftPtr->getParametersObject().ionOptSolver == "CGPRP")
          cgReturn = cgSolver.solve(*this, std::string("ionRelaxCG.chk"));
        else if (dftPtr->getParametersObject().ionOptSolver == "CGPRP")
          cgReturn = cgSolver.solve(*this);
        else if (dftPtr->getParametersObject().ionOptSolver == "LBFGS")
          {
            cg_descent.set_step(0.8);
            cg_descent.set_lbfgs(true);
            if (this_mpi_process == 0)
              cg_descent.set_PrintLevel(2);

            unsigned int memory =
              std::min((unsigned int)100, getNumberUnknowns());
            if (memory <= 2)
              memory = 0;
            cg_descent.set_memory(memory);
            cgSuccess = cg_descent.run(*this);
          }
        else if (dftPtr->getParametersObject().ionOptSolver == "BFGS")
          {
            cgReturn = bfgsSolver.solve(*this);
          }
        else if (dftPtr->getParametersObject().ionOptSolver == "LBFGSv2")
          {
            cgReturn = lbfgsSolver.solve(*this);
          }
        else
          {
            cg_descent.set_step(0.8);
            if (this_mpi_process == 0)
              cg_descent.set_PrintLevel(2);
            cg_descent.set_AWolfe(true);

            unsigned int memory =
              std::min((unsigned int)100, getNumberUnknowns());
            if (memory <= 2)
              memory = 0;
            cg_descent.set_memory(memory);
            cgSuccess = cg_descent.run(*this);
          }

        if (cgReturn == nonLinearSolver::SUCCESS || cgSuccess)
          {
            pcout
              << " ...Ion force relaxation completed as maximum force magnitude is less than FORCE TOL: "
              << dftPtr->getParametersObject().forceRelaxTol
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
            for (int i = 0; i < dftPtr->d_domainBoundingVectors.size(); ++i)
              {
                pcout << "v" << i + 1 << " : "
                      << dftPtr->d_domainBoundingVectors[i][0] << " "
                      << dftPtr->d_domainBoundingVectors[i][1] << " "
                      << dftPtr->d_domainBoundingVectors[i][2] << std::endl;
              }
            pcout
              << "-----------------------------------------------------------------------------------------"
              << std::endl;

            if (dftPtr->getParametersObject().periodicX ||
                dftPtr->getParametersObject().periodicY ||
                dftPtr->getParametersObject().periodicZ)
              {
                pcout
                  << "-------------------Fractional coordinates of atoms----------------------"
                  << std::endl;
                for (unsigned int i = 0; i < dftPtr->atomLocations.size(); ++i)
                  pcout << (unsigned int)dftPtr->atomLocationsFractional[i][0]
                        << " "
                        << (unsigned int)dftPtr->atomLocationsFractional[i][1]
                        << " " << dftPtr->atomLocationsFractional[i][2] << " "
                        << dftPtr->atomLocationsFractional[i][3] << " "
                        << dftPtr->atomLocationsFractional[i][4] << "\n";
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
                for (unsigned int i = 0; i < dftPtr->atomLocations.size(); ++i)
                  {
                    pcout << (unsigned int)dftPtr->atomLocations[i][0] << " "
                          << (unsigned int)dftPtr->atomLocations[i][1] << " "
                          << dftPtr->atomLocations[i][2] << " "
                          << dftPtr->atomLocations[i][3] << " "
                          << dftPtr->atomLocations[i][4] << "\n";
                  }
                pcout
                  << "-----------------------------------------------------------------------------------------"
                  << std::endl;
              }
            pcout
              << "-----------------------------------------------------------------------------------"
              << std::endl;

            dftPtr->writeDomainAndAtomCoordinates();
          }
        else if (cgReturn == nonLinearSolver::FAILURE || !cgSuccess)
          {
            pcout << " ...Ion force relaxation failed " << std::endl;
          }
        else if (cgReturn == nonLinearSolver::MAX_ITER_REACHED)
          {
            pcout << " ...Maximum iterations reached " << std::endl;
          }
      }

    return d_totalUpdateCalls;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  unsigned int
  geoOptIon<FEOrder, FEOrderElectro>::getNumberUnknowns() const
  {
    unsigned int count = 0;
    for (unsigned int i = 0; i < d_relaxationFlags.size(); ++i)
      {
        count += d_relaxationFlags[i];
      }
    return count;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  geoOptIon<FEOrder, FEOrderElectro>::value(std::vector<double> &functionValue)
  {
    // AssertThrow(false,dftUtils::ExcNotImplementedYet());
    functionValue.clear();

    // Relative to initial free energy supressed in case of CGPRP
    // as that would not work in case of restarted CGPRP
    functionValue.push_back(dftPtr->d_groundStateEnergy);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  geoOptIon<FEOrder, FEOrderElectro>::gradient(std::vector<double> &gradient)
  {
    gradient.clear();
    const int                 numberGlobalAtoms = dftPtr->atomLocations.size();
    const std::vector<double> tempGradient = dftPtr->forcePtr->getAtomsForces();
    AssertThrow(tempGradient.size() == numberGlobalAtoms * 3,
                ExcMessage("Atom forces have wrong size"));
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

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  geoOptIon<FEOrder, FEOrderElectro>::precondition(
    std::vector<double> &      s,
    const std::vector<double> &gradient) const
  {
    const int numberGlobalAtoms = dftPtr->atomLocations.size();
    double    rCutNNList        = 1.0;
    std::vector<std::vector<double>> NNdistances(numberGlobalAtoms);
    bool                             allAtomsHaveNN = false;
    /*while (!allAtomsHaveNN)
      {
        for (int i = 0; i < numberGlobalAtoms; ++i)
          {
            for (int j = i + 1; j < numberGlobalAtoms; ++j)
              {
                double rij = 0;
                for (int k = 2; k < 5; ++k)
                  {
                    rij += (dftPtr->atomLocations[i][k] -
                            dftPtr->atomLocations[j][k]) *
                           (dftPtr->atomLocations[i][k] -
                            dftPtr->atomLocations[j][k]);
                  }
                rij = std::sqrt(rij);
                if (rij <= rCutNNList)
                  {
                    NNdistances[i].push_back(rij);
                    NNdistances[j].push_back(rij);
                  }
              }
          }
        allAtomsHaveNN = true;
        for (int i = 0; i < numberGlobalAtoms; ++i)
          {
            if (NNdistances[i].size() == 0)
              {
                allAtomsHaveNN = false;
                rCutNNList *= (1.0 + std::sqrt(5.0)) / 2.0;
              }
          }
      }
    double rNN = 0;
    for (int i = 0; i < numberGlobalAtoms; ++i)
      {
        double rijMin = NNdistances[i][0];
        for (int j = 1; j < NNdistances[i].size(); ++j)
          {
            rijMin = rijMin < NNdistances[i][j] ? rijMin : NNdistances[i][j];
          }
        rNN = rNN > rijMin ? rNN : rijMin;
      }*/
    double rNN = 0;
    for (int i = 0; i < numberGlobalAtoms; ++i)
      {
        double riMin = 0;
        for (int j = 0; j < numberGlobalAtoms; ++j)
          {
            double rij = 0;
            for (int k = 2; k < 5; ++k)
              {
                rij +=
                  (dftPtr->atomLocations[i][k] - dftPtr->atomLocations[j][k]) *
                  (dftPtr->atomLocations[i][k] - dftPtr->atomLocations[j][k]);
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
    pcout << "DEBUG rcut" << rCut << std::endl;
    std::vector<double> L(numberGlobalAtoms * numberGlobalAtoms, 0.0);
    for (int i = 0; i < numberGlobalAtoms; ++i)
      {
        for (int j = i + 1; j < numberGlobalAtoms; ++j)
          {
            double rij = 0;
            for (int k = 2; k < 5; ++k)
              {
                rij +=
                  (dftPtr->atomLocations[i][k] - dftPtr->atomLocations[j][k]) *
                  (dftPtr->atomLocations[i][k] - dftPtr->atomLocations[j][k]);
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
                L[i * numberGlobalAtoms + i] -= L[i * numberGlobalAtoms + j];
              }
          }
        L[i * numberGlobalAtoms + i] += 0.1;
      }
    pcout << "DEBUG Laplacian" << std::endl;
    for (auto i = 0; i < numberGlobalAtoms; ++i)
      {
        for (auto j = 0; j < numberGlobalAtoms; ++j)
          {
            pcout << L[i * numberGlobalAtoms + j] << "  ";
          }
        pcout << std::endl;
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
                              k == l ? 0.0102908099018465 *
                                         L[i * numberGlobalAtoms + j] :
                                       0.0;
                            ++jcount;
                          }
                      }
                  }
                ++icount;
              }
          }
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  geoOptIon<FEOrder, FEOrderElectro>::update(
    const std::vector<double> &solution,
    const bool                 computeForces,
    const bool                 useSingleAtomSolutionsInitialGuess)
  {
    const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
    std::vector<Tensor<1, 3, double>> globalAtomsDisplacements(
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

    if (dftPtr->getParametersObject().verbosity >= 1)
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

    dftPtr->updateAtomPositionsAndMoveMesh(globalAtomsDisplacements,
                                           factor,
                                           useSingleAtomSolutionsInitialGuess);
    d_totalUpdateCalls += 1;


    /*if(d_maximumAtomForceToBeRelaxed >= 1e-02)
      dftPtr->getParametersObject().selfConsistentSolverTolerance = 1e-03;
      else if(d_maximumAtomForceToBeRelaxed >= 1e-03)
      dftPtr->getParametersObject().selfConsistentSolverTolerance = 1e-04;
      else if(d_maximumAtomForceToBeRelaxed >= 1e-04)
      dftPtr->getParametersObject().selfConsistentSolverTolerance = 1e-05;
      else if(d_maximumAtomForceToBeRelaxed >= 1e-05)
      dftPtr->getParametersObject().selfConsistentSolverTolerance = 5e-06;*/

    dftPtr->solve(computeForces, false);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  geoOptIon<FEOrder, FEOrderElectro>::save()
  {
    dftPtr->writeDomainAndAtomCoordinates();
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  const MPI_Comm &
  geoOptIon<FEOrder, FEOrderElectro>::getMPICommunicator()
  {
    return mpi_communicator;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  geoOptIon<FEOrder, FEOrderElectro>::solution(std::vector<double> &solution)
  {
    // AssertThrow(false,dftUtils::ExcNotImplementedYet());
    solution.clear();
    const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
    for (unsigned int i = 0; i < numberGlobalAtoms; ++i)
      {
        for (unsigned int j = 0; j < 3; ++j)
          {
            if (d_relaxationFlags[3 * i + j] == 1)
              {
                solution.push_back(dftPtr->atomLocations[i][j + 2] -
                                   dftPtr->d_atomLocationsInitial[i][j + 2]);
              }
          }
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  std::vector<unsigned int>
  geoOptIon<FEOrder, FEOrderElectro>::getUnknownCountFlag() const
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }

#include "geoOptIon.inst.cc"
} // namespace dftfe
