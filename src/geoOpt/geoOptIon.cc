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
#include <cg_descent_wrapper.h>
#include <dft.h>
#include <dftParameters.h>
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
    const MPI_Comm &                   mpi_comm_replica)
    : dftPtr(_dftPtr)
    , mpi_communicator(mpi_comm_replica)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_comm_replica))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_comm_replica))
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
             !dftParameters::reproducible_output))
  {}

  //
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  geoOptIon<FEOrder, FEOrderElectro>::init()
  {
    const int numberGlobalAtoms = dftPtr->atomLocations.size();
    if (dftParameters::ionRelaxFlagsFile != "")
      {
        std::vector<std::vector<int>>    tempRelaxFlagsData;
        std::vector<std::vector<double>> tempForceData;
        dftUtils::readRelaxationFlagsFile(6,
                                          tempRelaxFlagsData,
                                          tempForceData,
                                          dftParameters::ionRelaxFlagsFile);
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
    const double tol = dftParameters::forceRelaxTol; //(units: Hatree/Bohr)
    const unsigned int maxIter = 300;
    const double       lineSearchTol =
      1e-4; // Dummy parameter for CGPRP, the actual stopping criteria are the
            // Wolfe conditions and maxLineSearchIter
    const double       lineSearchDampingParameter = 0.8;
    const unsigned int maxLineSearchIter =
      dftParameters::maxLineSearchIterCGPRP;
    const double       maxDisplacmentInAnyComponent = 0.5; // Bohr
    const unsigned int debugLevel =
      Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 ?
        dftParameters::verbosity :
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

    CGDescent cg_descent(tol, maxIter);


    if (dftParameters::chkType >= 1 && dftParameters::restartFromChk)
      pcout << "Re starting Ion force relaxation using nonlinear CG solver... "
            << std::endl;
    else
      pcout << "Starting Ion force relaxation using nonlinear CG solver... "
            << std::endl;
    if (dftParameters::verbosity >= 2)
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

        if (dftParameters::chkType >= 1 && dftParameters::restartFromChk &&
            dftParameters::ionOptSolver == "CGPRP")
          cgReturn = cgSolver.solve(*this, std::string("ionRelaxCG.chk"), true);
        else if (dftParameters::chkType >= 1 &&
                 !dftParameters::restartFromChk &&
                 dftParameters::ionOptSolver == "CGPRP")
          cgReturn = cgSolver.solve(*this, std::string("ionRelaxCG.chk"));
        else if (dftParameters::ionOptSolver == "CGPRP")
          cgReturn = cgSolver.solve(*this);
        else if (dftParameters::ionOptSolver == "LBFGS")
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
              << dftParameters::forceRelaxTol
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

            if (dftParameters::periodicX || dftParameters::periodicY ||
                dftParameters::periodicZ)
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
    functionValue.push_back(dftPtr->d_freeEnergy -
                            ((dftParameters::ionOptSolver == "CGPRP") ?
                               0.0 :
                               dftPtr->d_freeEnergyInitial));
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
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
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

        MPI_Bcast(
          &(globalAtomsDisplacements[i][0]), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
      }

    if (dftParameters::verbosity >= 1)
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
      dftParameters::selfConsistentSolverTolerance = 1e-03;
      else if(d_maximumAtomForceToBeRelaxed >= 1e-03)
      dftParameters::selfConsistentSolverTolerance = 1e-04;
      else if(d_maximumAtomForceToBeRelaxed >= 1e-04)
      dftParameters::selfConsistentSolverTolerance = 1e-05;
      else if(d_maximumAtomForceToBeRelaxed >= 1e-05)
      dftParameters::selfConsistentSolverTolerance = 5e-06;*/

    dftPtr->solve(computeForces, false);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  geoOptIon<FEOrder, FEOrderElectro>::save()
  {
    dftPtr->writeDomainAndAtomCoordinates();
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
