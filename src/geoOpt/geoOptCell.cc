// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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
#include <dft.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include <force.h>
#include <geoOptCell.h>
#include <geoOptIon.h>

namespace dftfe
{
  //
  // constructor
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  geoOptCell<FEOrder, FEOrderElectro>::geoOptCell(
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
  geoOptCell<FEOrder, FEOrderElectro>::init()
  {
    // initialize d_strainEpsilon to identity
    d_strainEpsilon = 0;
    for (unsigned int i = 0; i < 3; ++i)
      d_strainEpsilon[i][i] = 1.0;

    // strain tensor is a symmetric second order with six independent components
    d_relaxationFlags.clear();
    d_relaxationFlags.resize(6, 0);

    if (dftParameters::cellConstraintType ==
        1) //(isotropic shape fixed isotropic volume optimization)
      {
        d_relaxationFlags[0] = 1; //(epsilon_11+epsilon22+epsilon_33)/3
      }
    else if (dftParameters::cellConstraintType ==
             2) //(volume fixed shape optimization)
      {
        d_relaxationFlags[1] = 1; // epsilon_12
        d_relaxationFlags[2] = 1; // epsilon_13
        d_relaxationFlags[4] = 1; // epsilon_23
      }
    else if (dftParameters::cellConstraintType ==
             3) // (relax only cell component v1_x)
      {
        d_relaxationFlags[0] = 1; // epsilon_11
      }
    else if (dftParameters::cellConstraintType ==
             4) // (relax only cell component v2_y)
      {
        d_relaxationFlags[3] = 1; // epsilon_22
      }
    else if (dftParameters::cellConstraintType ==
             5) // (relax only cell component v3_z)
      {
        d_relaxationFlags[5] = 1; // epsilon_33
      }
    else if (dftParameters::cellConstraintType ==
             6) // (relax only cell components v2_y and v3_z)
      {
        d_relaxationFlags[3] = 1; // epsilon_22
        d_relaxationFlags[5] = 1; // epsilon_33
      }
    else if (dftParameters::cellConstraintType ==
             7) // (relax only cell components v1_x and v3_z)
      {
        d_relaxationFlags[0] = 1; // epsilon_11
        d_relaxationFlags[5] = 1; // epsilon_33
      }
    else if (dftParameters::cellConstraintType ==
             8) // (relax only cell components v1_x and v2_y)
      {
        d_relaxationFlags[0] = 1; // epsilon_11
        d_relaxationFlags[3] = 1; // epsilon_22
      }
    else if (dftParameters::cellConstraintType ==
             9) //(relax v1_x, v2_y and v3_z)
      {
        d_relaxationFlags[0] = 1; // epsilon_11
        d_relaxationFlags[3] = 1; // epsilon_22
        d_relaxationFlags[5] = 1; // epsilon_33
      }
    else if (dftParameters::cellConstraintType ==
             10) //(2D only x and y components relaxed)
      {
        d_relaxationFlags[0] = 1; // epsilon_11
        d_relaxationFlags[1] = 1; // epsilon_12
        d_relaxationFlags[3] = 1; // epsilon_22
      }
    else if (dftParameters::cellConstraintType ==
             11) //(2D only x and y shape components- inplane area fixed)
      {
        d_relaxationFlags[1] = 1; // epsilon_12
      }
    else if (dftParameters::cellConstraintType ==
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
    else if (dftParameters::cellConstraintType ==
             13) //(automatically decides constraints based on boundary
                 // conditions)
      {
        d_relaxationFlags[0] = 1;
        d_relaxationFlags[1] = 1;
        d_relaxationFlags[2] = 1;
        d_relaxationFlags[3] = 1;
        d_relaxationFlags[4] = 1;
        d_relaxationFlags[5] = 1;

        if (!dftParameters::periodicX)
          {
            d_relaxationFlags[0] = 0; // epsilon_11
            d_relaxationFlags[1] = 0; // epsilon_12
            d_relaxationFlags[2] = 0; // epsilon_13
          }

        if (!dftParameters::periodicY)
          {
            d_relaxationFlags[1] = 0; // epsilon_12
            d_relaxationFlags[3] = 0; // epsilon_22
            d_relaxationFlags[4] = 0; // epsilon_23
          }

        if (!dftParameters::periodicZ)
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
          ExcMessage(
            "The given value for CELL CONSTRAINT TYPE doesn't match with any available options (1-13)."));
      }

    if (dftParameters::verbosity >= 2)
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
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  int
  geoOptCell<FEOrder, FEOrderElectro>::run()
  {
    const double       tol     = dftParameters::stressRelaxTol;
    const unsigned int maxIter = 100;
    const double       lineSearchTol =
      tol * 2.0; // Dummy parameter for CGPRP, the actual stopping criteria are
                 // the Wolfe conditions and maxLineSearchIter
    const double       lineSearchDampingParameter = 0.5;
    const unsigned int maxLineSearchIter =
      dftParameters::maxLineSearchIterCGPRP;
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
                                  lineSearchDampingParameter);

    if (dftParameters::chkType >= 1 && dftParameters::restartFromChk)
      pcout
        << " Re starting Cell stress relaxation using nonlinear CG solver... "
        << std::endl;
    else
      pcout << " Starting Cell stress relaxation using nonlinear CG solver... "
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

        if (dftParameters::chkType >= 1 && dftParameters::restartFromChk)
          cgReturn =
            cgSolver.solve(*this, std::string("cellRelaxCG.chk"), true);
        else if (dftParameters::chkType >= 1 && !dftParameters::restartFromChk)
          cgReturn = cgSolver.solve(*this, std::string("cellRelaxCG.chk"));
        else
          cgReturn = cgSolver.solve(*this);

        if (cgReturn == nonLinearSolver::SUCCESS)
          {
            pcout
              << " ...Cell stress relaxation completed as maximum stress magnitude is less than STRESS TOL: "
              << dftParameters::stressRelaxTol
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
                  << "------------------Fractional coordinates of atoms--------------------"
                  << std::endl;
                for (unsigned int i = 0; i < dftPtr->atomLocations.size(); ++i)
                  pcout << "AtomId " << i << ":  "
                        << dftPtr->atomLocationsFractional[i][2] << " "
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
                    pcout << "AtomId " << i << ":  "
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
        else if (cgReturn == nonLinearSolver::MAX_ITER_REACHED)
          {
            pcout << " ...Maximum iterations reached " << std::endl;
          }
        else if (cgReturn == nonLinearSolver::FAILURE)
          {
            pcout << " ...Cell stress relaxation failed " << std::endl;
          }
      }

    return d_totalUpdateCalls;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  unsigned int
  geoOptCell<FEOrder, FEOrderElectro>::getNumberUnknowns() const
  {
    return std::accumulate(d_relaxationFlags.begin(),
                           d_relaxationFlags.end(),
                           0);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  geoOptCell<FEOrder, FEOrderElectro>::value(std::vector<double> &functionValue)
  {
    // AssertThrow(false,dftUtils::ExcNotImplementedYet());
    functionValue.clear();
    functionValue.push_back(dftPtr->d_freeEnergy / dftPtr->d_domainVolume);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  geoOptCell<FEOrder, FEOrderElectro>::gradient(std::vector<double> &gradient)
  {
    gradient.clear();
    const Tensor<2, 3, double> tempGradient = dftPtr->forcePtr->getStress();

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

    if (dftParameters::cellConstraintType ==
        1) // isotropic (shape fixed isotropic volume optimization)
      {
        gradient[0] =
          (tempGradient[0][0] + tempGradient[1][1] + tempGradient[2][2]) / 3.0;
      }
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  geoOptCell<FEOrder, FEOrderElectro>::precondition(
    std::vector<double> &      s,
    const std::vector<double> &gradient) const
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  geoOptCell<FEOrder, FEOrderElectro>::update(
    const std::vector<double> &solution,
    const bool                 computeStress,
    const bool                 useSingleAtomSolutionsInitialGuess)
  {
    std::vector<double> bcastSolution(solution.size());
    for (unsigned int i = 0; i < solution.size(); ++i)
      {
        bcastSolution[i] = solution[i];
      }

    // for synchronization
    MPI_Bcast(
      &(bcastSolution[0]), bcastSolution.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    Tensor<2, 3, double> strainEpsilonNew = d_strainEpsilon;

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


    if (dftParameters::cellConstraintType ==
        1) // isotropic (shape fixed isotropic volume optimization)
      {
        strainEpsilonNew[1][1] = strainEpsilonNew[0][0];
        strainEpsilonNew[2][2] = strainEpsilonNew[0][0];
      }

    // To transform the domain under the strain we have to first do a inverse
    // transformation to bring the domain back to the unstrained state.
    Tensor<2, 3, double> deformationGradient =
      strainEpsilonNew * invert(d_strainEpsilon);
    d_strainEpsilon = strainEpsilonNew;

    // deform fem mesh and reinit
    d_totalUpdateCalls += 1;
    dftPtr->deformDomain(deformationGradient);


    dftPtr->solve(true, computeStress);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  geoOptCell<FEOrder, FEOrderElectro>::save()
  {
    dftPtr->writeDomainAndAtomCoordinates();
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  geoOptCell<FEOrder, FEOrderElectro>::solution(std::vector<double> &solution)
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  std::vector<unsigned int>
  geoOptCell<FEOrder, FEOrderElectro>::getUnknownCountFlag() const
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }

#include "geoOptCell.inst.cc"
} // namespace dftfe
