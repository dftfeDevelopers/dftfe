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
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das
//

// Include header files
#include <chebyshevOrthogonalizedSubspaceIterationSolver.h>
#include <dealiiLinearSolver.h>
#include <densityCalculatorCPU.h>
#include <densityFirstOrderResponseCalculator.h>
#include <dft.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <energyCalculator.h>
#include <fileReaders.h>
#include <force.h>
#include <kohnShamDFTOperator.h>
#include <linalg.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsInternal.h>
#include <meshMovementAffineTransform.h>
#include <meshMovementGaussian.h>
#include "molecularDynamicsClass.h"
#include <poissonSolverProblem.h>
#include <pseudoConverter.h>
#include <pseudoUtils.h>
#include <symmetry.h>
#include <vectorUtilities.h>
#include <MemoryTransfer.h>

#include <algorithm>
#include <cmath>
#include <complex>
//#include <stdafx.h>
#include <boost/math/distributions/normal.hpp>
#include <boost/math/special_functions/spherical_harmonic.hpp>
#include <boost/random/normal_distribution.hpp>

#include <spglib.h>
#include <stdafx.h>
#include <sys/stat.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <chrono>
#include <sys/time.h>
#include <ctime>

#ifdef DFTFE_WITH_DEVICE
#  include <densityCalculatorDevice.h>
#  include <linearAlgebraOperationsDevice.h>
#endif

#include <elpa/elpa.h>


namespace dftfe
{
  // Include cc files
#include "atomicRho.cc"
#include "charge.cc"
#include "density.cc"
#include "dos.cc"
#include "electrostaticHRefinedEnergy.cc"
#include "femUtilityFunctions.cc"
#include "fermiEnergy.cc"
#include "generateImageCharges.cc"
#include "initBoundaryConditions.cc"
#include "initCoreRho.cc"
#include "initElectronicFields.cc"
#include "initPseudo-OV.cc"
#include "initPseudoLocal.cc"
#include "initRho.cc"
#include "initUnmovedTriangulation.cc"
#include "kohnShamEigenSolve.cc"
#include "localizationLength.cc"
#include "mixingschemes.cc"
#include "moveAtoms.cc"
#include "moveMeshToAtoms.cc"
#include "nodalDensityMixingSchemes.cc"
#include "nscf.cc"
#include "pRefinedDoFHandler.cc"
#include "psiInitialGuess.cc"
#include "publicMethods.cc"
#include "restart.cc"
#include "lowrankApproxScfDielectricMatrixInv.cc"
#include "lowrankApproxScfDielectricMatrixInvSpinPolarized.cc"
#include "computeOutputDensityDirectionalDerivative.cc"

  //
  // dft constructor
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  dftClass<FEOrder, FEOrderElectro>::dftClass(
    const MPI_Comm &   mpi_comm_parent,
    const MPI_Comm &   mpi_comm_domain,
    const MPI_Comm &   _interpoolcomm,
    const MPI_Comm &   _interBandGroupComm,
    const std::string &scratchFolderName,
    dftParameters &    dftParams)
    : FE(FE_Q<3>(QGaussLobatto<1>(FEOrder + 1)), 1)
    ,
#ifdef USE_COMPLEX
    FEEigen(FE_Q<3>(QGaussLobatto<1>(FEOrder + 1)), 2)
    ,
#else
    FEEigen(FE_Q<3>(QGaussLobatto<1>(FEOrder + 1)), 1)
    ,
#endif
    mpi_communicator(mpi_comm_domain)
    , d_mpiCommParent(mpi_comm_parent)
    , interpoolcomm(_interpoolcomm)
    , interBandGroupComm(_interBandGroupComm)
    , d_dftfeScratchFolderName(scratchFolderName)
    , d_dftParamsPtr(&dftParams)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , numElectrons(0)
    , numLevels(0)
    , d_autoMesh(1)
    , d_mesh(mpi_comm_parent,
             mpi_comm_domain,
             _interpoolcomm,
             _interBandGroupComm,
             FEOrder,
             dftParams)
    , d_affineTransformMesh(mpi_comm_parent, mpi_comm_domain, dftParams)
    , d_gaussianMovePar(mpi_comm_parent, mpi_comm_domain, dftParams)
    , d_vselfBinsManager(mpi_comm_parent,
                         mpi_comm_domain,
                         _interpoolcomm,
                         dftParams)
    , d_dispersionCorr(mpi_comm_parent,
                       mpi_comm_domain,
                       _interpoolcomm,
                       _interBandGroupComm,
                       dftParams)
    , pcout(std::cout,
            (Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0) &&
              dftParams.verbosity >= 0)
    , d_kohnShamDFTOperatorsInitialized(false)
    , computing_timer(mpi_comm_domain,
                      pcout,
                      dftParams.reproducible_output || dftParams.verbosity < 4 ?
                        TimerOutput::never :
                        TimerOutput::summary,
                      TimerOutput::wall_times)
    , computingTimerStandard(mpi_comm_domain,
                             pcout,
                             dftParams.reproducible_output ||
                                 dftParams.verbosity < 1 ?
                               TimerOutput::never :
                               TimerOutput::every_call_and_summary,
                             TimerOutput::wall_times)
    , d_subspaceIterationSolver(mpi_comm_parent,
                                mpi_comm_domain,
                                0.0,
                                0.0,
                                0.0,
                                dftParams)
#ifdef DFTFE_WITH_DEVICE
    , d_subspaceIterationSolverDevice(mpi_comm_parent,
                                      mpi_comm_domain,
                                      0.0,
                                      0.0,
                                      0.0,
                                      dftParams)
    , d_phiTotalSolverProblemDevice(mpi_comm_domain)
#endif
    , d_phiTotalSolverProblem(mpi_comm_domain)
  {
    d_elpaScala = new dftfe::elpaScalaManager(mpi_comm_domain);

    forcePtr    = new forceClass<FEOrder, FEOrderElectro>(this,
                                                       mpi_comm_parent,
                                                       mpi_comm_domain,
                                                       dftParams);
    symmetryPtr = new symmetryClass<FEOrder, FEOrderElectro>(this,
                                                             mpi_comm_parent,
                                                             mpi_comm_domain,
                                                             _interpoolcomm);

    d_isRestartGroundStateCalcFromChk = false;

#if defined(DFTFE_WITH_DEVICE)
    d_devicecclMpiCommDomainPtr = new utils::DeviceCCLWrapper;
    if (d_dftParamsPtr->useDeviceDirectAllReduce)
      d_devicecclMpiCommDomainPtr->init(mpi_comm_domain);
#endif
    d_pspCutOff =
      d_dftParamsPtr->reproducible_output ?
        30.0 :
        (std::max(d_dftParamsPtr->pspCutoffImageCharges, d_pspCutOffTrunc));
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  dftClass<FEOrder, FEOrderElectro>::~dftClass()
  {
    finalizeKohnShamDFTOperator();
    delete symmetryPtr;
    matrix_free_data.clear();
    delete forcePtr;
#if defined(DFTFE_WITH_DEVICE)
    delete d_devicecclMpiCommDomainPtr;
#endif

    d_elpaScala->elpaDeallocateHandles(*d_dftParamsPtr);
    delete d_elpaScala;

    xc_func_end(&funcX);
    xc_func_end(&funcC);
    delete excFunctionalPtr;
  }

  namespace internaldft
  {
    void
    convertToCellCenteredCartesianCoordinates(
      std::vector<std::vector<double>> &      atomLocations,
      const std::vector<std::vector<double>> &latticeVectors)
    {
      std::vector<double> cartX(atomLocations.size(), 0.0);
      std::vector<double> cartY(atomLocations.size(), 0.0);
      std::vector<double> cartZ(atomLocations.size(), 0.0);

      //
      // convert fractional atomic coordinates to cartesian coordinates
      //
      for (int i = 0; i < atomLocations.size(); ++i)
        {
          cartX[i] = atomLocations[i][2] * latticeVectors[0][0] +
                     atomLocations[i][3] * latticeVectors[1][0] +
                     atomLocations[i][4] * latticeVectors[2][0];
          cartY[i] = atomLocations[i][2] * latticeVectors[0][1] +
                     atomLocations[i][3] * latticeVectors[1][1] +
                     atomLocations[i][4] * latticeVectors[2][1];
          cartZ[i] = atomLocations[i][2] * latticeVectors[0][2] +
                     atomLocations[i][3] * latticeVectors[1][2] +
                     atomLocations[i][4] * latticeVectors[2][2];
        }

      //
      // define cell centroid (confirm whether it will work for non-orthogonal
      // lattice vectors)
      //
      double cellCentroidX =
        0.5 *
        (latticeVectors[0][0] + latticeVectors[1][0] + latticeVectors[2][0]);
      double cellCentroidY =
        0.5 *
        (latticeVectors[0][1] + latticeVectors[1][1] + latticeVectors[2][1]);
      double cellCentroidZ =
        0.5 *
        (latticeVectors[0][2] + latticeVectors[1][2] + latticeVectors[2][2]);

      for (int i = 0; i < atomLocations.size(); ++i)
        {
          atomLocations[i][2] = cartX[i] - cellCentroidX;
          atomLocations[i][3] = cartY[i] - cellCentroidY;
          atomLocations[i][4] = cartZ[i] - cellCentroidZ;
        }
    }
  } // namespace internaldft

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::computeVolume(
    const dealii::DoFHandler<3> &_dofHandler)
  {
    double               domainVolume = 0;
    const Quadrature<3> &quadrature =
      matrix_free_data.get_quadrature(d_densityQuadratureId);
    FEValues<3> fe_values(_dofHandler.get_fe(), quadrature, update_JxW_values);

    typename DoFHandler<3>::active_cell_iterator cell =
                                                   _dofHandler.begin_active(),
                                                 endc = _dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          for (unsigned int q_point = 0; q_point < quadrature.size(); ++q_point)
            domainVolume += fe_values.JxW(q_point);
        }

    domainVolume = Utilities::MPI::sum(domainVolume, mpi_communicator);
    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "Volume of the domain (Bohr^3): " << domainVolume << std::endl;
    return domainVolume;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::set()
  {
    computingTimerStandard.enter_subsection("Atomic system initialization");
    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "Entered call to set");

    d_numEigenValues = d_dftParamsPtr->numberEigenValues;

    //
    // read coordinates
    //
    unsigned int numberColumnsCoordinatesFile =
      d_dftParamsPtr->useMeshSizesFromAtomsFile ? 7 : 5;

    if (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
        d_dftParamsPtr->periodicZ)
      {
        //
        // read fractionalCoordinates of atoms in periodic case
        //
        dftUtils::readFile(numberColumnsCoordinatesFile,
                           atomLocations,
                           d_dftParamsPtr->coordinatesFile);
        AssertThrow(
          d_dftParamsPtr->natoms == atomLocations.size(),
          ExcMessage(
            "DFT-FE Error: The number atoms"
            "read from the atomic coordinates file (input through ATOMIC COORDINATES FILE) doesn't"
            "match the NATOMS input. Please check your atomic coordinates file. Sometimes an extra"
            "blank row at the end can cause this issue too."));
        pcout << "number of atoms: " << atomLocations.size() << "\n";
        atomLocationsFractional.resize(atomLocations.size());
        //
        // find unique atom types
        //
        for (std::vector<std::vector<double>>::iterator it =
               atomLocations.begin();
             it < atomLocations.end();
             it++)
          {
            atomTypes.insert((unsigned int)((*it)[0]));
            d_atomTypeAtributes[(unsigned int)((*it)[0])] =
              (unsigned int)((*it)[1]);

            if (!d_dftParamsPtr->isPseudopotential)
              AssertThrow(
                (*it)[0] <= 50,
                ExcMessage(
                  "DFT-FE Error: One of the atomic numbers exceeds 50."
                  "Currently, for all-electron calculations we have single atom wavefunction and electron-density"
                  "initial guess data till atomic number 50 only. Data for the remaining atomic numbers will be"
                  "added in the next release. In the mean time, you could also contact the developers of DFT-FE, who can provide"
                  "you the data for the single atom wavefunction and electron-density data for"
                  "atomic numbers beyond 50."));
          }

        //
        // print fractional coordinates
        //
        for (int i = 0; i < atomLocations.size(); ++i)
          {
            atomLocationsFractional[i] = atomLocations[i];
          }
      }
    else
      {
        dftUtils::readFile(numberColumnsCoordinatesFile,
                           atomLocations,
                           d_dftParamsPtr->coordinatesFile);

        AssertThrow(
          d_dftParamsPtr->natoms == atomLocations.size(),
          ExcMessage(
            "DFT-FE Error: The number atoms"
            "read from the atomic coordinates file (input through ATOMIC COORDINATES FILE) doesn't"
            "match the NATOMS input. Please check your atomic coordinates file. Sometimes an extra"
            "blank row at the end can cause this issue too."));
        pcout << "number of atoms: " << atomLocations.size() << "\n";

        //
        // find unique atom types
        //
        for (std::vector<std::vector<double>>::iterator it =
               atomLocations.begin();
             it < atomLocations.end();
             it++)
          {
            atomTypes.insert((unsigned int)((*it)[0]));
            d_atomTypeAtributes[(unsigned int)((*it)[0])] =
              (unsigned int)((*it)[1]);

            if (!d_dftParamsPtr->isPseudopotential)
              AssertThrow(
                (*it)[0] <= 50,
                ExcMessage(
                  "DFT-FE Error: One of the atomic numbers exceeds 50."
                  "Currently, for all-electron calculations we have single atom wavefunction and electron-density"
                  "initial guess data till atomic number 50 only. Data for the remaining atomic numbers will be"
                  "added in the next release. You could also contact the developers of DFT-FE, who can provide"
                  "you with the code to generate the single atom wavefunction and electron-density data for"
                  "atomic numbers beyond 50."));
          }
      }

    //
    // read Gaussian atomic displacements
    //
    std::vector<std::vector<double>> atomsDisplacementsGaussian;
    d_atomsDisplacementsGaussianRead.resize(atomLocations.size(),
                                            Tensor<1, 3, double>());
    d_gaussianMovementAtomsNetDisplacements.resize(atomLocations.size(),
                                                   Tensor<1, 3, double>());
    if (d_dftParamsPtr->coordinatesGaussianDispFile != "")
      {
        dftUtils::readFile(3,
                           atomsDisplacementsGaussian,
                           d_dftParamsPtr->coordinatesGaussianDispFile);

        for (int i = 0; i < atomsDisplacementsGaussian.size(); ++i)
          for (int j = 0; j < 3; ++j)
            d_atomsDisplacementsGaussianRead[i][j] =
              atomsDisplacementsGaussian[i][j];

        d_isAtomsGaussianDisplacementsReadFromFile = true;
      }

    //
    // read domain bounding Vectors
    //
    unsigned int numberColumnsLatticeVectorsFile = 3;
    dftUtils::readFile(numberColumnsLatticeVectorsFile,
                       d_domainBoundingVectors,
                       d_dftParamsPtr->domainBoundingVectorsFile);

    AssertThrow(
      d_domainBoundingVectors.size() == 3,
      ExcMessage(
        "DFT-FE Error: The number of domain bounding"
        "vectors read from input file (input through DOMAIN VECTORS FILE) should be 3. Please check"
        "your domain vectors file. Sometimes an extra blank row at the end can cause this issue too."));

    //
    // evaluate cross product of
    //
    std::vector<double> cross;
    dftUtils::cross_product(d_domainBoundingVectors[0],
                            d_domainBoundingVectors[1],
                            cross);

    double scalarConst = d_domainBoundingVectors[2][0] * cross[0] +
                         d_domainBoundingVectors[2][1] * cross[1] +
                         d_domainBoundingVectors[2][2] * cross[2];
    AssertThrow(
      scalarConst > 0,
      ExcMessage(
        "DFT-FE Error: Domain bounding vectors or lattice vectors read from"
        "input file (input through DOMAIN VECTORS FILE) should form a right-handed coordinate system."
        "Please check your domain vectors file. This is usually fixed by changing the order of the"
        "vectors in the domain vectors file."));

    pcout << "number of atoms types: " << atomTypes.size() << "\n";


    //
    // determine number of electrons
    //
    for (unsigned int iAtom = 0; iAtom < atomLocations.size(); iAtom++)
      {
        const unsigned int Z        = atomLocations[iAtom][0];
        const unsigned int valenceZ = atomLocations[iAtom][1];

        if (d_dftParamsPtr->isPseudopotential)
          numElectrons += valenceZ;
        else
          numElectrons += Z;
      }

    if (d_dftParamsPtr->numberEigenValues <= numElectrons / 2.0 ||
        d_dftParamsPtr->numberEigenValues == 0)
      {
        if (d_dftParamsPtr->verbosity >= 1)
          {
            pcout
              << " Warning: User has requested the number of Kohn-Sham wavefunctions to be less than or"
                 "equal to half the number of electrons in the system. Setting the Kohn-Sham wavefunctions"
                 "to half the number of electrons with a 20 percent buffer to avoid convergence issues in"
                 "SCF iterations"
              << std::endl;
          }
        d_numEigenValues =
          (numElectrons / 2.0) +
          std::max((d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND" ?
                      0.22 :
                      0.2) *
                     (numElectrons / 2.0),
                   20.0);

        // start with 17-20% buffer to leave room for additional modifications
        // due to block size restrictions
#ifdef DFTFE_WITH_DEVICE
        if (d_dftParamsPtr->useDevice && d_dftParamsPtr->autoDeviceBlockSizes)
          d_numEigenValues =
            (numElectrons / 2.0) + std::max((d_dftParamsPtr->mixingMethod ==
                                                 "LOW_RANK_DIELECM_PRECOND" ?
                                               0.2 :
                                               0.17) *
                                              (numElectrons / 2.0),
                                            20.0);
#endif

        if (d_dftParamsPtr->verbosity >= 1)
          {
            pcout << " Setting the number of Kohn-Sham wave functions to be "
                  << d_numEigenValues << std::endl;
          }
      }

    if (d_dftParamsPtr->algoType == "FAST")
      {
        if (d_dftParamsPtr->TVal < 1000)
          {
            d_dftParamsPtr->numCoreWfcRR = 0.8 * numElectrons / 2.0;
            pcout << " Setting SPECTRUM SPLIT CORE EIGENSTATES to be "
                  << d_dftParamsPtr->numCoreWfcRR << std::endl;
          }
      }


#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice && d_dftParamsPtr->autoDeviceBlockSizes)
      {
        const unsigned int numberBandGroups =
          dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);


        d_numEigenValues =
          std::ceil(d_numEigenValues / (numberBandGroups * 1.0)) *
          numberBandGroups;

        AssertThrow(
          (d_numEigenValues % numberBandGroups == 0 ||
           d_numEigenValues / numberBandGroups == 0),
          ExcMessage(
            "DFT-FE Error: TOTAL NUMBER OF KOHN-SHAM WAVEFUNCTIONS must be exactly divisible by NPBAND for Device run."));

        const unsigned int bandGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
        std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
        dftUtils::createBandParallelizationIndices(
          interBandGroupComm, d_numEigenValues, bandGroupLowHighPlusOneIndices);

        const unsigned int eigenvaluesInBandGroup =
          bandGroupLowHighPlusOneIndices[1];

        if (eigenvaluesInBandGroup <= 100)
          {
            d_dftParamsPtr->chebyWfcBlockSize = eigenvaluesInBandGroup;
            d_dftParamsPtr->wfcBlockSize      = eigenvaluesInBandGroup;
          }
        else if (eigenvaluesInBandGroup <= 600)
          {
            std::vector<int> temp1(4, 0);
            std::vector<int> temp2(4, 0);
            temp1[0] = std::ceil(eigenvaluesInBandGroup / 90.0) * 90.0 *
                       numberBandGroups;
            temp1[1] = std::ceil(eigenvaluesInBandGroup / 100.0) * 100.0 *
                       numberBandGroups;
            temp1[2] = std::ceil(eigenvaluesInBandGroup / 110.0) * 110.0 *
                       numberBandGroups;
            temp1[3] = std::ceil(eigenvaluesInBandGroup / 120.0) * 120.0 *
                       numberBandGroups;

            temp2[0] = 90;
            temp2[1] = 100;
            temp2[2] = 110;
            temp2[3] = 120;

            int minElementIndex =
              std::min_element(temp1.begin(), temp1.end()) - temp1.begin();
            int minElement = *std::min_element(temp1.begin(), temp1.end());

            d_numEigenValues                  = minElement;
            d_dftParamsPtr->chebyWfcBlockSize = temp2[minElementIndex];
            d_dftParamsPtr->wfcBlockSize      = temp2[minElementIndex];
          }
        else if (eigenvaluesInBandGroup <= 1000)
          {
            std::vector<int> temp1(4, 0);
            std::vector<int> temp2(4, 0);
            temp1[0] = std::ceil(eigenvaluesInBandGroup / 150.0) * 150.0 *
                       numberBandGroups;
            temp1[1] = std::ceil(eigenvaluesInBandGroup / 160.0) * 160.0 *
                       numberBandGroups;
            temp1[2] = std::ceil(eigenvaluesInBandGroup / 170.0) * 170.0 *
                       numberBandGroups;
            temp1[3] = std::ceil(eigenvaluesInBandGroup / 180.0) * 180.0 *
                       numberBandGroups;

            temp2[0] = 150;
            temp2[1] = 160;
            temp2[2] = 170;
            temp2[3] = 180;

            int minElementIndex =
              std::min_element(temp1.begin(), temp1.end()) - temp1.begin();
            int minElement = *std::min_element(temp1.begin(), temp1.end());

            d_numEigenValues                  = minElement;
            d_dftParamsPtr->chebyWfcBlockSize = temp2[minElementIndex] / 2;
            d_dftParamsPtr->wfcBlockSize      = temp2[minElementIndex];
          }
        else if (eigenvaluesInBandGroup <= 2000)
          {
            std::vector<int> temp1(4, 0);
            std::vector<int> temp2(4, 0);
            temp1[0] = std::ceil(eigenvaluesInBandGroup / 200.0) * 200.0 *
                       numberBandGroups;
            temp1[1] = std::ceil(eigenvaluesInBandGroup / 220.0) * 220.0 *
                       numberBandGroups;
            temp1[2] = std::ceil(eigenvaluesInBandGroup / 240.0) * 240.0 *
                       numberBandGroups;
            temp1[3] = std::ceil(eigenvaluesInBandGroup / 260.0) * 260.0 *
                       numberBandGroups;

            temp2[0] = 200;
            temp2[1] = 220;
            temp2[2] = 240;
            temp2[3] = 260;

            int minElementIndex =
              std::min_element(temp1.begin(), temp1.end()) - temp1.begin();
            int minElement = *std::min_element(temp1.begin(), temp1.end());

            d_numEigenValues                  = minElement;
            d_dftParamsPtr->chebyWfcBlockSize = temp2[minElementIndex] / 2;
            d_dftParamsPtr->wfcBlockSize      = temp2[minElementIndex];
          }
        else
          {
            std::vector<int> temp1(4, 0);
            std::vector<int> temp2(4, 0);
            temp1[0] = std::ceil(eigenvaluesInBandGroup / 360.0) * 360.0 *
                       numberBandGroups;
            temp1[1] = std::ceil(eigenvaluesInBandGroup / 380.0) * 380.0 *
                       numberBandGroups;
            temp1[2] = std::ceil(eigenvaluesInBandGroup / 400.0) * 400.0 *
                       numberBandGroups;
            temp1[3] = std::ceil(eigenvaluesInBandGroup / 440.0) * 440.0 *
                       numberBandGroups;

            temp2[0] = 360;
            temp2[1] = 380;
            temp2[2] = 400;
            temp2[3] = 440;

            int minElementIndex =
              std::min_element(temp1.begin(), temp1.end()) - temp1.begin();
            int minElement = *std::min_element(temp1.begin(), temp1.end());

            d_numEigenValues                  = minElement;
            d_dftParamsPtr->chebyWfcBlockSize = temp2[minElementIndex] / 2;
            d_dftParamsPtr->wfcBlockSize      = temp2[minElementIndex];
          }

        if (d_dftParamsPtr->algoType == "FAST")
          d_dftParamsPtr->numCoreWfcRR =
            std::floor(d_dftParamsPtr->numCoreWfcRR /
                       d_dftParamsPtr->wfcBlockSize) *
            d_dftParamsPtr->wfcBlockSize;

        if (d_dftParamsPtr->verbosity >= 1)
          {
            pcout
              << " Setting the number of Kohn-Sham wave functions for Device run to be: "
              << d_numEigenValues << std::endl;
            pcout << " Setting CHEBY WFC BLOCK SIZE for Device run to be "
                  << d_dftParamsPtr->chebyWfcBlockSize << std::endl;
            pcout << " Setting WFC BLOCK SIZE for Device run to be "
                  << d_dftParamsPtr->wfcBlockSize << std::endl;
            if (d_dftParamsPtr->algoType == "FAST")
              pcout
                << " Setting SPECTRUM SPLIT CORE EIGENSTATES for Device run to be "
                << d_dftParamsPtr->numCoreWfcRR << std::endl;
          }
      }
#endif

    if (d_dftParamsPtr->constraintMagnetization)
      {
        numElectronsUp   = std::ceil(static_cast<double>(numElectrons) / 2.0);
        numElectronsDown = numElectrons - numElectronsUp;
        //
        int netMagnetization =
          std::round(2.0 * static_cast<double>(numElectrons) *
                     d_dftParamsPtr->start_magnetization);
        //
        while ((numElectronsUp - numElectronsDown) < std::abs(netMagnetization))
          {
            numElectronsDown -= 1;
            numElectronsUp += 1;
          }
        //
        if (d_dftParamsPtr->verbosity >= 1)
          {
            pcout << " Number of spin up electrons " << numElectronsUp
                  << std::endl;
            pcout << " Number of spin down electrons " << numElectronsDown
                  << std::endl;
          }
      }

    // estimate total number of wave functions from atomic orbital filling
    if (d_dftParamsPtr->startingWFCType == "ATOMIC")
      determineOrbitalFilling();

    AssertThrow(
      d_dftParamsPtr->numCoreWfcRR <= d_numEigenValues,
      ExcMessage(
        "DFT-FE Error: Incorrect input value used- SPECTRUM SPLIT CORE EIGENSTATES should be less than the total number of wavefunctions."));
    d_numEigenValuesRR = d_numEigenValues - d_dftParamsPtr->numCoreWfcRR;


#ifdef USE_COMPLEX
    generateMPGrid();
#else
    d_kPointCoordinates.resize(3, 0.0);
    d_kPointWeights.resize(1, 1.0);
#endif

    // set size of eigenvalues and eigenvectors data structures
    eigenValues.resize(d_kPointWeights.size());
    eigenValuesRRSplit.resize(d_kPointWeights.size());

    if (d_dftParamsPtr->mixingMethod == "LOW_RANK_DIELECM_PRECOND")
      d_densityMatDerFermiEnergy.resize((d_dftParamsPtr->spinPolarized + 1) *
                                        d_kPointWeights.size());

    a0.clear();
    bLow.clear();

    a0.resize((d_dftParamsPtr->spinPolarized + 1) * d_kPointWeights.size(),
              0.0);
    bLow.resize((d_dftParamsPtr->spinPolarized + 1) * d_kPointWeights.size(),
                0.0);

    d_upperBoundUnwantedSpectrumValues.clear();
    d_upperBoundUnwantedSpectrumValues.resize(
      (d_dftParamsPtr->spinPolarized + 1) * d_kPointWeights.size(), 0.0);

    d_eigenVectorsFlattenedSTL.resize((1 + d_dftParamsPtr->spinPolarized) *
                                      d_kPointWeights.size());
    d_eigenVectorsRotFracDensityFlattenedSTL.resize(
      (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size());

    for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
      {
        eigenValues[kPoint].resize((d_dftParamsPtr->spinPolarized + 1) *
                                   d_numEigenValues);
        eigenValuesRRSplit[kPoint].resize((d_dftParamsPtr->spinPolarized + 1) *
                                          d_numEigenValuesRR);
      }

    // convert pseudopotential files in upf format to dftfe format
    if (d_dftParamsPtr->verbosity >= 1)
      {
        pcout
          << std::endl
          << "Reading Pseudo-potential data for each atom from the list given in : "
          << d_dftParamsPtr->pseudoPotentialFile << std::endl;
      }

    int nlccFlag = 0;
    if (Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0 &&
        d_dftParamsPtr->isPseudopotential == true)
      nlccFlag = pseudoUtils::convert(d_dftParamsPtr->pseudoPotentialFile,
                                      d_dftfeScratchFolderName,
                                      d_dftParamsPtr->verbosity,
                                      d_dftParamsPtr->natomTypes,
                                      d_dftParamsPtr->pseudoTestsFlag);

    nlccFlag = Utilities::MPI::sum(nlccFlag, d_mpiCommParent);

    if (nlccFlag > 0 && d_dftParamsPtr->isPseudopotential == true)
      d_dftParamsPtr->nonLinearCoreCorrection = true;

    if (d_dftParamsPtr->verbosity >= 1)
      if (d_dftParamsPtr->nonLinearCoreCorrection == true)
        pcout
          << "Atleast one atom has pseudopotential with nonlinear core correction"
          << std::endl;

    d_elpaScala->processGridELPASetup(d_numEigenValues,
                                      d_numEigenValuesRR,
                                      *d_dftParamsPtr);

    MPI_Barrier(d_mpiCommParent);
    computingTimerStandard.leave_subsection("Atomic system initialization");
  }

  // dft pseudopotential init
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::initPseudoPotentialAll(
    const bool updateNonlocalSparsity)
  {
    if (d_dftParamsPtr->isPseudopotential)
      {
        TimerOutput::Scope scope(computing_timer, "psp init");
        pcout << std::endl << "Pseudopotential initalization...." << std::endl;
        const Quadrature<3> &quadrature =
          matrix_free_data.get_quadrature(d_densityQuadratureId);

        double init_core;
        MPI_Barrier(d_mpiCommParent);
        init_core = MPI_Wtime();

        if (d_dftParamsPtr->nonLinearCoreCorrection == true)
          initCoreRho();

        MPI_Barrier(d_mpiCommParent);
        init_core = MPI_Wtime() - init_core;
        if (d_dftParamsPtr->verbosity >= 2)
          pcout
            << "initPseudoPotentialAll: Time taken for initializing core density for non-linear core correction: "
            << init_core << std::endl;


        if (updateNonlocalSparsity)
          {
            double init_nonlocal1;
            MPI_Barrier(d_mpiCommParent);
            init_nonlocal1 = MPI_Wtime();

            computeSparseStructureNonLocalProjectors_OV();

            MPI_Barrier(d_mpiCommParent);
            init_nonlocal1 = MPI_Wtime() - init_nonlocal1;
            if (d_dftParamsPtr->verbosity >= 2)
              pcout
                << "initPseudoPotentialAll: Time taken for computeSparseStructureNonLocalProjectors_OV: "
                << init_nonlocal1 << std::endl;
          }

        double init_nonlocal2;
        MPI_Barrier(d_mpiCommParent);
        init_nonlocal2 = MPI_Wtime();


        computeElementalOVProjectorKets();

        // forcePtr->initPseudoData();

        MPI_Barrier(d_mpiCommParent);
        init_nonlocal2 = MPI_Wtime() - init_nonlocal2;
        if (d_dftParamsPtr->verbosity >= 2)
          pcout << "initPseudoPotentialAll: Time taken for non local psp init: "
                << init_nonlocal2 << std::endl;
      }
  }


  // generate image charges and update k point cartesian coordinates based on
  // current lattice vectors
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::initImageChargesUpdateKPoints(bool flag)
  {
    TimerOutput::Scope scope(computing_timer,
                             "image charges and k point generation");
    pcout
      << "-----------Simulation Domain bounding vectors (lattice vectors in fully periodic case)-------------"
      << std::endl;
    for (int i = 0; i < d_domainBoundingVectors.size(); ++i)
      {
        pcout << "v" << i + 1 << " : " << d_domainBoundingVectors[i][0] << " "
              << d_domainBoundingVectors[i][1] << " "
              << d_domainBoundingVectors[i][2] << std::endl;
      }
    pcout
      << "-----------------------------------------------------------------------------------------"
      << std::endl;

    if (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
        d_dftParamsPtr->periodicZ)
      {
        pcout << "-----Fractional coordinates of atoms------ " << std::endl;
        for (unsigned int i = 0; i < atomLocations.size(); ++i)
          {
            atomLocations[i] = atomLocationsFractional[i];
            pcout << "AtomId " << i << ":  " << atomLocationsFractional[i][2]
                  << " " << atomLocationsFractional[i][3] << " "
                  << atomLocationsFractional[i][4] << "\n";
          }
        pcout
          << "-----------------------------------------------------------------------------------------"
          << std::endl;
        // sanity check on fractional coordinates
        std::vector<bool> periodicBc(3, false);
        periodicBc[0]    = d_dftParamsPtr->periodicX;
        periodicBc[1]    = d_dftParamsPtr->periodicY;
        periodicBc[2]    = d_dftParamsPtr->periodicZ;
        const double tol = 1e-6;

        if (flag)
          {
            for (unsigned int i = 0; i < atomLocationsFractional.size(); ++i)
              {
                for (unsigned int idim = 0; idim < 3; ++idim)
                  {
                    if (periodicBc[idim])
                      AssertThrow(
                        atomLocationsFractional[i][2 + idim] > -tol &&
                          atomLocationsFractional[i][2 + idim] < 1.0 + tol,
                        ExcMessage(
                          "DFT-FE Error: periodic direction fractional coordinates doesn't lie in [0,1]. Please check input"
                          "fractional coordinates, or if this is an ionic relaxation step, please check the corresponding"
                          "algorithm."));
                    if (!periodicBc[idim])
                      AssertThrow(
                        atomLocationsFractional[i][2 + idim] > tol &&
                          atomLocationsFractional[i][2 + idim] < 1.0 - tol,
                        ExcMessage(
                          "DFT-FE Error: non-periodic direction fractional coordinates doesn't lie in (0,1). Please check"
                          "input fractional coordinates, or if this is an ionic relaxation step, please check the"
                          "corresponding algorithm."));
                  }
              }
          }

        generateImageCharges(d_pspCutOff,
                             d_imageIds,
                             d_imageCharges,
                             d_imagePositions);

        generateImageCharges(d_pspCutOffTrunc,
                             d_imageIdsTrunc,
                             d_imageChargesTrunc,
                             d_imagePositionsTrunc);

        if ((d_dftParamsPtr->verbosity >= 4 ||
             d_dftParamsPtr->reproducible_output))
          pcout << "Number Image Charges  " << d_imageIds.size() << std::endl;

        internaldft::convertToCellCenteredCartesianCoordinates(
          atomLocations, d_domainBoundingVectors);
#ifdef USE_COMPLEX
        recomputeKPointCoordinates();
#endif
        if (d_dftParamsPtr->verbosity >= 4)
          {
            // FIXME: Print all k points across all pools
            pcout
              << "-------------------k points cartesian coordinates and weights-----------------------------"
              << std::endl;
            for (unsigned int i = 0; i < d_kPointWeights.size(); ++i)
              {
                pcout << " [" << d_kPointCoordinates[3 * i + 0] << ", "
                      << d_kPointCoordinates[3 * i + 1] << ", "
                      << d_kPointCoordinates[3 * i + 2] << "] "
                      << d_kPointWeights[i] << std::endl;
              }
            pcout
              << "-----------------------------------------------------------------------------------------"
              << std::endl;
          }
      }
    else
      {
        //
        // print cartesian coordinates
        //
        pcout
          << "------------Cartesian coordinates of atoms (origin at center of domain)------------------"
          << std::endl;
        for (unsigned int i = 0; i < atomLocations.size(); ++i)
          {
            pcout << "AtomId " << i << ":  " << atomLocations[i][2] << " "
                  << atomLocations[i][3] << " " << atomLocations[i][4] << "\n";
          }
        pcout
          << "-----------------------------------------------------------------------------------------"
          << std::endl;

        //
        // redundant call (check later)
        //
        generateImageCharges(d_pspCutOff,
                             d_imageIds,
                             d_imageCharges,
                             d_imagePositions);

        generateImageCharges(d_pspCutOffTrunc,
                             d_imageIdsTrunc,
                             d_imageChargesTrunc,
                             d_imagePositionsTrunc);
      }
  }

  // dft init
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::init()
  {
    computingTimerStandard.enter_subsection("KSDFT problem initialization");

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator, "Entering init");

    initImageChargesUpdateKPoints();

    calculateNearestAtomDistances();

    computing_timer.enter_subsection("mesh generation");
    //
    // generate mesh (both parallel and serial)
    // while parallel meshes are always generated, serial meshes are only
    // generated for following three cases: symmetrization is on, ionic
    // optimization is on as well as reuse wfcs and density from previous ionic
    // step is on, or if serial constraints generation is on.
    //
    if (d_dftParamsPtr->loadRhoData)
      {
        d_mesh.generateCoarseMeshesForRestart(
          atomLocations,
          d_imagePositionsTrunc,
          d_imageIdsTrunc,
          d_nearestAtomDistances,
          d_domainBoundingVectors,
          d_dftParamsPtr->useSymm ||
            d_dftParamsPtr->createConstraintsFromSerialDofhandler);

        loadTriaInfoAndRhoNodalData();
      }
    else
      {
        d_mesh.generateSerialUnmovedAndParallelMovedUnmovedMesh(
          atomLocations,
          d_imagePositionsTrunc,
          d_imageIdsTrunc,
          d_nearestAtomDistances,
          d_domainBoundingVectors,
          d_dftParamsPtr->useSymm ||
            d_dftParamsPtr->createConstraintsFromSerialDofhandler,
          d_dftParamsPtr->electrostaticsHRefinement);
      }
    computing_timer.leave_subsection("mesh generation");

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "Mesh generation completed");
    //
    // get access to triangulation objects from meshGenerator class
    //
    parallel::distributed::Triangulation<3> &triangulationPar =
      d_mesh.getParallelMeshMoved();

    //
    // initialize dofHandlers and hanging-node constraints and periodic
    // constraints on the unmoved Mesh
    //
    initUnmovedTriangulation(triangulationPar);

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "initUnmovedTriangulation completed");
#ifdef USE_COMPLEX
    if (d_dftParamsPtr->useSymm)
      symmetryPtr->initSymmetry();
#endif



    //
    // move triangulation to have atoms on triangulation vertices
    //
    if (!d_dftParamsPtr->floatingNuclearCharges)
      moveMeshToAtoms(triangulationPar, d_mesh.getSerialMeshUnmoved());


    if (d_dftParamsPtr->smearedNuclearCharges)
      calculateSmearedChargeWidths();

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "moveMeshToAtoms completed");
    //
    // initialize dirichlet BCs for total potential and vSelf poisson solutions
    //
    initBoundaryConditions();


    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "initBoundaryConditions completed");
    //
    // initialize guesses for electron-density and wavefunctions
    //
    initElectronicFields();

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "initElectronicFields completed");
    //
    // initialize pseudopotential data for both local and nonlocal part
    //
    initPseudoPotentialAll();

    if (d_dftParamsPtr->verbosity >= 4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                        "initPseudopotential completed");

    //
    // Apply Gaussian displacments to atoms and mesh if input gaussian
    // displacments are read from file. When restarting a relaxation, this must
    // be done only once at the begining- this is why the flag is to false after
    // the Gaussian movement. The last flag to updateAtomPositionsAndMoveMesh is
    // set to true to force use of single atom solutions.
    //
    if (d_isAtomsGaussianDisplacementsReadFromFile)
      {
        updateAtomPositionsAndMoveMesh(d_atomsDisplacementsGaussianRead,
                                       1e+4,
                                       true);
        d_isAtomsGaussianDisplacementsReadFromFile = false;
      }

    if (d_dftParamsPtr->loadRhoData)
      {
        if (d_dftParamsPtr->verbosity >= 1)
          pcout
            << "Overwriting input density data to SCF solve with data read from restart file.."
            << std::endl;

        // Note: d_rhoInNodalValuesRead is not compatible with
        // d_matrixFreeDataPRefined
        for (unsigned int i = 0; i < d_rhoInNodalValues.local_size(); i++)
          d_rhoInNodalValues.local_element(i) =
            d_rhoInNodalValuesRead.local_element(i);

        d_rhoInNodalValues.update_ghost_values();
        interpolateRhoNodalDataToQuadratureDataGeneral(
          d_matrixFreeDataPRefined,
          d_densityDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          d_rhoInNodalValues,
          *(rhoInValues),
          *(gradRhoInValues),
          *(gradRhoInValues),
          excFunctionalPtr->getDensityBasedFamilyType() ==
            densityFamilyType::GGA);

        if (d_dftParamsPtr->spinPolarized == 1)
          {
            d_rhoInSpin0NodalValues = 0;
            d_rhoInSpin1NodalValues = 0;
            for (unsigned int i = 0; i < d_rhoInSpin0NodalValues.local_size();
                 i++)
              {
                d_rhoInSpin0NodalValues.local_element(i) =
                  d_rhoInSpin0NodalValuesRead.local_element(i);
                d_rhoInSpin1NodalValues.local_element(i) =
                  d_rhoInSpin1NodalValuesRead.local_element(i);
              }

            d_rhoInSpin0NodalValues.update_ghost_values();
            d_rhoInSpin1NodalValues.update_ghost_values();
            interpolateRhoSpinNodalDataToQuadratureDataGeneral(
              d_matrixFreeDataPRefined,
              d_densityDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
              d_rhoInSpin0NodalValues,
              d_rhoInSpin1NodalValues,
              *rhoInValuesSpinPolarized,
              *gradRhoInValuesSpinPolarized,
              *gradRhoInValuesSpinPolarized,
              excFunctionalPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA);
          }
        if ((d_dftParamsPtr->solverMode == "GEOOPT"))
          {
            d_rhoOutNodalValues = d_rhoInNodalValues;
            d_rhoOutNodalValues.update_ghost_values();
            rhoOutVals.push_back(*(rhoInValues));
            rhoOutValues = &(rhoOutVals.back());

            if (excFunctionalPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA)
              {
                gradRhoOutVals.push_back(*(gradRhoInValues));
                gradRhoOutValues = &(gradRhoOutVals.back());
              }

            if (d_dftParamsPtr->spinPolarized == 1)
              {
                rhoOutValsSpinPolarized.push_back(*rhoInValuesSpinPolarized);
                rhoOutValuesSpinPolarized = &(rhoOutValsSpinPolarized.back());
              }

            if (excFunctionalPtr->getDensityBasedFamilyType() ==
                  densityFamilyType::GGA &&
                d_dftParamsPtr->spinPolarized == 1)
              {
                gradRhoOutValsSpinPolarized.push_back(
                  *gradRhoInValuesSpinPolarized);
                gradRhoOutValuesSpinPolarized =
                  &(gradRhoOutValsSpinPolarized.back());
              }
          }

        d_isRestartGroundStateCalcFromChk = true;
      }

    d_isFirstFilteringCall.clear();
    d_isFirstFilteringCall.resize((d_dftParamsPtr->spinPolarized + 1) *
                                    d_kPointWeights.size(),
                                  true);

    initializeKohnShamDFTOperator();

    d_netFloatingDispSinceLastCheckForSmearedChargeOverlaps.clear();
    d_netFloatingDispSinceLastCheckForSmearedChargeOverlaps.resize(
      atomLocations.size() * 3, 0.0);

    computingTimerStandard.leave_subsection("KSDFT problem initialization");
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::initNoRemesh(
    const bool updateImagesAndKPointsAndVselfBins,
    const bool checkSmearedChargeWidthsForOverlap,
    const bool useSingleAtomSolutionOverride,
    const bool isMeshDeformed)
  {
    computingTimerStandard.enter_subsection("KSDFT problem initialization");
    if (updateImagesAndKPointsAndVselfBins)
      {
        initImageChargesUpdateKPoints();
      }

    if (checkSmearedChargeWidthsForOverlap)
      {
        calculateNearestAtomDistances();

        if (d_dftParamsPtr->smearedNuclearCharges)
          calculateSmearedChargeWidths();

        d_netFloatingDispSinceLastCheckForSmearedChargeOverlaps.clear();
        d_netFloatingDispSinceLastCheckForSmearedChargeOverlaps.resize(
          atomLocations.size() * 3, 0.0);
      }

    //
    // reinitialize dirichlet BCs for total potential and vSelf poisson
    // solutions
    //
    double init_bc;
    MPI_Barrier(d_mpiCommParent);
    init_bc = MPI_Wtime();


    // false option reinitializes vself bins from scratch wheras true option
    // only updates the boundary conditions
    const bool updateOnlyBinsBc = !updateImagesAndKPointsAndVselfBins;
    initBoundaryConditions(updateOnlyBinsBc);

    MPI_Barrier(d_mpiCommParent);
    init_bc = MPI_Wtime() - init_bc;
    if (d_dftParamsPtr->verbosity >= 2)
      pcout
        << "updateAtomPositionsAndMoveMesh: Time taken for initBoundaryConditions: "
        << init_bc << std::endl;

    double init_rho;
    MPI_Barrier(d_mpiCommParent);
    init_rho = MPI_Wtime();

    if (useSingleAtomSolutionOverride)
      {
        readPSI();
        initRho();
      }
    else
      {
        //
        // rho init (use previous ground state electron density)
        //
        // if(d_dftParamsPtr->mixingMethod != "ANDERSON_WITH_KERKER")
        //   solveNoSCF();

        if (!d_dftParamsPtr->reuseWfcGeoOpt)
          readPSI();

        noRemeshRhoDataInit();

        if (d_dftParamsPtr->reuseDensityGeoOpt >= 1 &&
            d_dftParamsPtr->solverMode == "GEOOPT")
          {
            if (d_dftParamsPtr->reuseDensityGeoOpt == 2 &&
                d_dftParamsPtr->spinPolarized != 1)
              {
                d_rhoOutNodalValuesSplit.add(
                  -totalCharge(d_matrixFreeDataPRefined,
                               d_rhoOutNodalValuesSplit) /
                  d_domainVolume);

                initAtomicRho();

                interpolateRhoNodalDataToQuadratureDataGeneral(
                  d_matrixFreeDataPRefined,
                  d_densityDofHandlerIndexElectro,
                  d_densityQuadratureIdElectro,
                  d_rhoOutNodalValuesSplit,
                  *(rhoInValues),
                  *(gradRhoInValues),
                  *(gradRhoInValues),
                  excFunctionalPtr->getDensityBasedFamilyType() ==
                    densityFamilyType::GGA);

                addAtomicRhoQuadValuesGradients(
                  *(rhoInValues),
                  *(gradRhoInValues),
                  excFunctionalPtr->getDensityBasedFamilyType() ==
                    densityFamilyType::GGA);

                normalizeRhoInQuadValues();

                l2ProjectionQuadToNodal(d_matrixFreeDataPRefined,
                                        d_constraintsRhoNodal,
                                        d_densityDofHandlerIndexElectro,
                                        d_densityQuadratureIdElectro,
                                        *rhoInValues,
                                        d_rhoInNodalValues);

                d_rhoInNodalValues.update_ghost_values();
              }
          }

        else if (d_dftParamsPtr->extrapolateDensity == 1 &&
                 d_dftParamsPtr->spinPolarized != 1 &&
                 d_dftParamsPtr->solverMode == "MD")
          {
            interpolateRhoNodalDataToQuadratureDataGeneral(
              d_matrixFreeDataPRefined,
              d_densityDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
              d_rhoOutNodalValues,
              *(rhoInValues),
              *(gradRhoInValues),
              *(gradRhoInValues),
              excFunctionalPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA);

            normalizeRhoInQuadValues();

            l2ProjectionQuadToNodal(d_matrixFreeDataPRefined,
                                    d_constraintsRhoNodal,
                                    d_densityDofHandlerIndexElectro,
                                    d_densityQuadratureIdElectro,
                                    *rhoInValues,
                                    d_rhoInNodalValues);

            d_rhoInNodalValues.update_ghost_values();
          }
        else if (d_dftParamsPtr->extrapolateDensity == 2 &&
                 d_dftParamsPtr->spinPolarized != 1 &&
                 d_dftParamsPtr->solverMode == "MD")
          {
            initAtomicRho();
            interpolateRhoNodalDataToQuadratureDataGeneral(
              d_matrixFreeDataPRefined,
              d_densityDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
              d_rhoOutNodalValuesSplit,
              *(rhoInValues),
              *(gradRhoInValues),
              *(gradRhoInValues),
              excFunctionalPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA);

            addAtomicRhoQuadValuesGradients(
              *(rhoInValues),
              *(gradRhoInValues),
              excFunctionalPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA);

            normalizeRhoInQuadValues();

            l2ProjectionQuadToNodal(d_matrixFreeDataPRefined,
                                    d_constraintsRhoNodal,
                                    d_densityDofHandlerIndexElectro,
                                    d_densityQuadratureIdElectro,
                                    *rhoInValues,
                                    d_rhoInNodalValues);

            d_rhoInNodalValues.update_ghost_values();
          }
        else
          {
            initRho();
          }
      }

    MPI_Barrier(d_mpiCommParent);
    init_rho = MPI_Wtime() - init_rho;
    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "updateAtomPositionsAndMoveMesh: Time taken for initRho: "
            << init_rho << std::endl;

    //
    // reinitialize pseudopotential related data structures
    //
    double init_pseudo;
    MPI_Barrier(d_mpiCommParent);
    init_pseudo = MPI_Wtime();

    initPseudoPotentialAll(d_dftParamsPtr->floatingNuclearCharges ? true :
                                                                    false);

    MPI_Barrier(d_mpiCommParent);
    init_pseudo = MPI_Wtime() - init_pseudo;
    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "Time taken for initPseudoPotentialAll: " << init_pseudo
            << std::endl;

    d_isFirstFilteringCall.clear();
    d_isFirstFilteringCall.resize((d_dftParamsPtr->spinPolarized + 1) *
                                    d_kPointWeights.size(),
                                  true);

    double init_ksoperator;
    MPI_Barrier(d_mpiCommParent);
    init_ksoperator = MPI_Wtime();

    if (isMeshDeformed)
      initializeKohnShamDFTOperator();
    else
      reInitializeKohnShamDFTOperator();

    init_ksoperator = MPI_Wtime() - init_ksoperator;
    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "Time taken for kohnShamDFTOperator class reinitialization: "
            << init_ksoperator << std::endl;

    computingTimerStandard.leave_subsection("KSDFT problem initialization");
  }

  //
  // deform domain and call appropriate reinits
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::deformDomain(
    const Tensor<2, 3, double> &deformationGradient,
    const bool                  vselfPerturbationUpdateForStress,
    const bool                  useSingleAtomSolutionsOverride,
    const bool                  print)
  {
    d_affineTransformMesh.initMoved(d_domainBoundingVectors);
    d_affineTransformMesh.transform(deformationGradient);

    dftUtils::transformDomainBoundingVectors(d_domainBoundingVectors,
                                             deformationGradient);

    if (print)
      {
        pcout
          << "-----------Simulation Domain bounding vectors (lattice vectors in fully periodic case)-------------"
          << std::endl;
        for (int i = 0; i < d_domainBoundingVectors.size(); ++i)
          {
            pcout << "v" << i + 1 << " : " << d_domainBoundingVectors[i][0]
                  << " " << d_domainBoundingVectors[i][1] << " "
                  << d_domainBoundingVectors[i][2] << std::endl;
          }
        pcout
          << "-----------------------------------------------------------------------------------------"
          << std::endl;
      }

#ifdef USE_COMPLEX
    if (!vselfPerturbationUpdateForStress)
      recomputeKPointCoordinates();
#endif

    // update atomic and image positions without any wrapping across periodic
    // boundary
    std::vector<Tensor<1, 3, double>> imageDisplacements(
      d_imagePositions.size());
    std::vector<Tensor<1, 3, double>> imageDisplacementsTrunc(
      d_imagePositionsTrunc.size());

    for (int iImage = 0; iImage < d_imagePositions.size(); ++iImage)
      {
        Point<3>           imageCoor;
        const unsigned int imageChargeId = d_imageIds[iImage];
        imageCoor[0]                     = d_imagePositions[iImage][0];
        imageCoor[1]                     = d_imagePositions[iImage][1];
        imageCoor[2]                     = d_imagePositions[iImage][2];

        Point<3> atomCoor;
        atomCoor[0] = atomLocations[imageChargeId][2];
        atomCoor[1] = atomLocations[imageChargeId][3];
        atomCoor[2] = atomLocations[imageChargeId][4];

        imageDisplacements[iImage] = imageCoor - atomCoor;
      }

    for (int iImage = 0; iImage < d_imagePositionsTrunc.size(); ++iImage)
      {
        Point<3>           imageCoor;
        const unsigned int imageChargeId = d_imageIdsTrunc[iImage];
        imageCoor[0]                     = d_imagePositionsTrunc[iImage][0];
        imageCoor[1]                     = d_imagePositionsTrunc[iImage][1];
        imageCoor[2]                     = d_imagePositionsTrunc[iImage][2];

        Point<3> atomCoor;
        atomCoor[0] = atomLocations[imageChargeId][2];
        atomCoor[1] = atomLocations[imageChargeId][3];
        atomCoor[2] = atomLocations[imageChargeId][4];

        imageDisplacementsTrunc[iImage] = imageCoor - atomCoor;
      }

    for (unsigned int i = 0; i < atomLocations.size(); ++i)
      atomLocations[i] = atomLocationsFractional[i];

    if (print)
      {
        pcout << "-----Fractional coordinates of atoms------ " << std::endl;
        for (unsigned int i = 0; i < atomLocations.size(); ++i)
          {
            pcout << "AtomId " << i << ":  " << atomLocationsFractional[i][2]
                  << " " << atomLocationsFractional[i][3] << " "
                  << atomLocationsFractional[i][4] << "\n";
          }
        pcout
          << "-----------------------------------------------------------------------------------------"
          << std::endl;
      }

    internaldft::convertToCellCenteredCartesianCoordinates(
      atomLocations, d_domainBoundingVectors);


    for (int iImage = 0; iImage < d_imagePositions.size(); ++iImage)
      {
        const unsigned int imageChargeId = d_imageIds[iImage];

        Point<3> atomCoor;
        atomCoor[0] = atomLocations[imageChargeId][2];
        atomCoor[1] = atomLocations[imageChargeId][3];
        atomCoor[2] = atomLocations[imageChargeId][4];

        imageDisplacements[iImage] =
          deformationGradient * imageDisplacements[iImage];

        d_imagePositions[iImage][0] =
          atomCoor[0] + imageDisplacements[iImage][0];
        d_imagePositions[iImage][1] =
          atomCoor[1] + imageDisplacements[iImage][1];
        d_imagePositions[iImage][2] =
          atomCoor[2] + imageDisplacements[iImage][2];
      }

    for (int iImage = 0; iImage < d_imagePositionsTrunc.size(); ++iImage)
      {
        const unsigned int imageChargeId = d_imageIdsTrunc[iImage];

        Point<3> atomCoor;
        atomCoor[0] = atomLocations[imageChargeId][2];
        atomCoor[1] = atomLocations[imageChargeId][3];
        atomCoor[2] = atomLocations[imageChargeId][4];

        imageDisplacementsTrunc[iImage] =
          deformationGradient * imageDisplacementsTrunc[iImage];

        d_imagePositionsTrunc[iImage][0] =
          atomCoor[0] + imageDisplacementsTrunc[iImage][0];
        d_imagePositionsTrunc[iImage][1] =
          atomCoor[1] + imageDisplacementsTrunc[iImage][1];
        d_imagePositionsTrunc[iImage][2] =
          atomCoor[2] + imageDisplacementsTrunc[iImage][2];
      }

    if (vselfPerturbationUpdateForStress)
      {
        //
        // reinitialize dirichlet BCs for total potential and vSelf poisson
        // solutions
        //
        double init_bc;
        MPI_Barrier(d_mpiCommParent);
        init_bc = MPI_Wtime();


        // first true option only updates the boundary conditions
        // second true option signals update is only for vself perturbation
        initBoundaryConditions(true, true);

        MPI_Barrier(d_mpiCommParent);
        init_bc = MPI_Wtime() - init_bc;
        if (d_dftParamsPtr->verbosity >= 2)
          pcout
            << "updateAtomPositionsAndMoveMesh: Time taken for initBoundaryConditions: "
            << init_bc << std::endl;
      }
    else
      {
        initNoRemesh(false, true, useSingleAtomSolutionsOverride, true);
      }
  }


  //
  // generate a-posteriori mesh
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::aposterioriMeshGenerate()
  {
    //
    // get access to triangulation objects from meshGenerator class
    //
    parallel::distributed::Triangulation<3> &triangulationPar =
      d_mesh.getParallelMeshMoved();
    unsigned int numberLevelRefinements = d_dftParamsPtr->numLevels;
    unsigned int numberWaveFunctionsErrorEstimate =
      d_dftParamsPtr->numberWaveFunctionsForEstimate;
    bool         refineFlag = true;
    unsigned int countLevel = 0;
    double       traceXtKX = computeTraceXtKX(numberWaveFunctionsErrorEstimate);
    double       traceXtKXPrev = traceXtKX;

    while (refineFlag)
      {
        if (numberLevelRefinements > 0)
          {
            distributedCPUVec<double> tempVec;
            matrix_free_data.initialize_dof_vector(tempVec);

            std::vector<distributedCPUVec<double>> eigenVectorsArray(
              numberWaveFunctionsErrorEstimate);

            for (unsigned int i = 0; i < numberWaveFunctionsErrorEstimate; ++i)
              eigenVectorsArray[i].reinit(tempVec);


            vectorTools::copyFlattenedSTLVecToSingleCompVec(
              d_eigenVectorsFlattenedSTL[0],
              d_numEigenValues,
              std::make_pair(0, numberWaveFunctionsErrorEstimate),
              eigenVectorsArray);


            for (unsigned int i = 0; i < numberWaveFunctionsErrorEstimate; ++i)
              {
                constraintsNone.distribute(eigenVectorsArray[i]);
                eigenVectorsArray[i].update_ghost_values();
              }


            d_mesh.generateAutomaticMeshApriori(
              dofHandler,
              triangulationPar,
              eigenVectorsArray,
              FEOrder,
              d_dftParamsPtr->electrostaticsHRefinement);
          }


        //
        // initialize dofHandlers of refined mesh and move triangulation
        //
        initUnmovedTriangulation(triangulationPar);
        moveMeshToAtoms(triangulationPar, d_mesh.getSerialMeshUnmoved());
        initBoundaryConditions();
        initElectronicFields();
        initPseudoPotentialAll();

        //
        // compute Tr(XtHX) for each level of mesh
        //
        // dataTypes::number traceXtHX =
        // computeTraceXtHX(numberWaveFunctionsErrorEstimate); pcout<<" Tr(XtHX)
        // value for Level: "<<countLevel<<" "<<traceXtHX<<std::endl;

        //
        // compute Tr(XtKX) for each level of mesh
        //
        traceXtKX = computeTraceXtKX(numberWaveFunctionsErrorEstimate);
        if (d_dftParamsPtr->verbosity > 0)
          pcout << " Tr(XtKX) value for Level: " << countLevel << " "
                << traceXtKX << std::endl;

        // compute change in traceXtKX
        double deltaKinetic =
          std::abs(traceXtKX - traceXtKXPrev) / atomLocations.size();

        // reset traceXtkXPrev to traceXtKX
        traceXtKXPrev = traceXtKX;

        //
        // set refineFlag
        //
        countLevel += 1;
        if (countLevel >= numberLevelRefinements ||
            deltaKinetic <= d_dftParamsPtr->toleranceKinetic)
          refineFlag = false;
      }
  }


  //
  // dft run
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::run()
  {
    if (d_dftParamsPtr->meshAdaption)
      aposterioriMeshGenerate();

    if (d_dftParamsPtr->restartFolder != "." && d_dftParamsPtr->saveRhoData &&
        Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
      {
        mkdir(d_dftParamsPtr->restartFolder.c_str(), ACCESSPERMS);
      }

    solve(true, true, d_isRestartGroundStateCalcFromChk);

    if (d_dftParamsPtr->writeWfcSolutionFields)
      outputWfc();

    if (d_dftParamsPtr->writeDensitySolutionFields)
      outputDensity();

    if (d_dftParamsPtr->writeDosFile)
      compute_tdos(eigenValues, "dosData.out");

    if (d_dftParamsPtr->writeLdosFile)
      compute_ldos(eigenValues, "ldosData.out");

    if (d_dftParamsPtr->writePdosFile)
      compute_pdos(eigenValues, "pdosData");

    if (d_dftParamsPtr->writeLocalizationLengths)
      compute_localizationLength("localizationLengths.out");

    /*if (d_dftParamsPtr->computeDipoleMoment)
      {
        dipole(d_dofHandlerPRefined, rhoOutValues, false);
        dipole(d_dofHandlerPRefined, rhoOutValues, true);
      } */

    if (d_dftParamsPtr->verbosity >= 1)
      pcout
        << std::endl
        << "------------------DFT-FE ground-state solve completed---------------------------"
        << std::endl;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::trivialSolveForStress()
  {
    initBoundaryConditions(false);
    noRemeshRhoDataInit();
    solve(false, true);
  }


  //
  // initialize
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::initializeKohnShamDFTOperator(
    const bool initializeCublas)
  {
    TimerOutput::Scope scope(computing_timer, "kohnShamDFTOperator init");
    double             init_ksoperator;
    MPI_Barrier(d_mpiCommParent);
    init_ksoperator = MPI_Wtime();

    if (d_kohnShamDFTOperatorsInitialized)
      finalizeKohnShamDFTOperator();

    d_kohnShamDFTOperatorPtr =
      new kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>(this,
                                                            d_mpiCommParent,
                                                            mpi_communicator);

#ifdef DFTFE_WITH_DEVICE
    d_kohnShamDFTOperatorDevicePtr =
      new kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>(
        this, d_mpiCommParent, mpi_communicator);
#endif

    kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
      &kohnShamDFTEigenOperator = *d_kohnShamDFTOperatorPtr;
#ifdef DFTFE_WITH_DEVICE
    kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
      &kohnShamDFTEigenOperatorDevice = *d_kohnShamDFTOperatorDevicePtr;
#endif

    if (!d_dftParamsPtr->useDevice)
      {
        kohnShamDFTEigenOperator.init();
      }

#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice)
      {
        kohnShamDFTEigenOperatorDevice.init();

        if (initializeCublas)
          {
            kohnShamDFTEigenOperatorDevice.createDeviceBlasHandle();
          }

        AssertThrow(
          (d_numEigenValues % d_dftParamsPtr->chebyWfcBlockSize == 0 ||
           d_numEigenValues / d_dftParamsPtr->chebyWfcBlockSize == 0),
          ExcMessage(
            "DFT-FE Error: total number wavefunctions must be exactly divisible by cheby wfc block size for Device run."));


        AssertThrow(
          (d_numEigenValues % d_dftParamsPtr->wfcBlockSize == 0 ||
           d_numEigenValues / d_dftParamsPtr->wfcBlockSize == 0),
          ExcMessage(
            "DFT-FE Error: total number wavefunctions must be exactly divisible by wfc block size for Device run."));

        AssertThrow(
          (d_dftParamsPtr->wfcBlockSize % d_dftParamsPtr->chebyWfcBlockSize ==
             0 &&
           d_dftParamsPtr->wfcBlockSize / d_dftParamsPtr->chebyWfcBlockSize >=
             0),
          ExcMessage(
            "DFT-FE Error: wfc block size must be exactly divisible by cheby wfc block size and also larger for Device run."));

        if (d_numEigenValuesRR != d_numEigenValues)
          AssertThrow(
            (d_numEigenValuesRR % d_dftParamsPtr->wfcBlockSize == 0 ||
             d_numEigenValuesRR / d_dftParamsPtr->wfcBlockSize == 0),
            ExcMessage(
              "DFT-FE Error: total number RR wavefunctions must be exactly divisible by wfc block size for Device run."));

        // band group parallelization data structures
        const unsigned int numberBandGroups =
          dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);

        AssertThrow(
          (d_numEigenValues % numberBandGroups == 0 ||
           d_numEigenValues / numberBandGroups == 0),
          ExcMessage(
            "DFT-FE Error: TOTAL NUMBER OF KOHN-SHAM WAVEFUNCTIONS must be exactly divisible by NPBAND for Device run."));

        const unsigned int bandGroupTaskId =
          dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
        std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
        dftUtils::createBandParallelizationIndices(
          interBandGroupComm, d_numEigenValues, bandGroupLowHighPlusOneIndices);

        AssertThrow(
          (bandGroupLowHighPlusOneIndices[1] %
             d_dftParamsPtr->chebyWfcBlockSize ==
           0),
          ExcMessage(
            "DFT-FE Error: band parallelization group size must be exactly divisible by CHEBY WFC BLOCK SIZE for Device run."));

        AssertThrow(
          (bandGroupLowHighPlusOneIndices[1] % d_dftParamsPtr->wfcBlockSize ==
           0),
          ExcMessage(
            "DFT-FE Error: band parallelization group size must be exactly divisible by WFC BLOCK SIZE for Device run."));

        kohnShamDFTEigenOperatorDevice.reinit(
          std::min(d_dftParamsPtr->chebyWfcBlockSize, d_numEigenValues), true);
      }
#endif

    if (!d_dftParamsPtr->useDevice)
      kohnShamDFTEigenOperator.preComputeShapeFunctionGradientIntegrals(
        d_lpspQuadratureId);
#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice)
      kohnShamDFTEigenOperatorDevice.preComputeShapeFunctionGradientIntegrals(
        d_lpspQuadratureId);
#endif

    d_kohnShamDFTOperatorsInitialized = true;

    MPI_Barrier(d_mpiCommParent);
    init_ksoperator = MPI_Wtime() - init_ksoperator;
    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "init: Time taken for kohnShamDFTOperator class initialization: "
            << init_ksoperator << std::endl;
  }


  //
  // re-initialize (significantly cheaper than initialize)
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::reInitializeKohnShamDFTOperator()
  {
    if (!d_dftParamsPtr->useDevice)
      d_kohnShamDFTOperatorPtr->resetExtPotHamFlag();

#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice)
      {
        d_kohnShamDFTOperatorDevicePtr->resetExtPotHamFlag();

        d_kohnShamDFTOperatorDevicePtr->reinit(
          std::min(d_dftParamsPtr->chebyWfcBlockSize, d_numEigenValues), true);
      }
#endif
  }

  //
  // finalize
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::finalizeKohnShamDFTOperator()
  {
    if (d_kohnShamDFTOperatorsInitialized)
      {
#ifdef DFTFE_WITH_DEVICE
        if (d_dftParamsPtr->useDevice)
          d_kohnShamDFTOperatorDevicePtr->destroyDeviceBlasHandle();
#endif

        delete d_kohnShamDFTOperatorPtr;
#ifdef DFTFE_WITH_DEVICE
        delete d_kohnShamDFTOperatorDevicePtr;
#endif
      }
  }

  //
  // dft solve
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  std::tuple<bool, double>
  dftClass<FEOrder, FEOrderElectro>::solve(
    const bool computeForces,
    const bool computestress,
    const bool isRestartGroundStateCalcFromChk)
  {
    kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
      &kohnShamDFTEigenOperator = *d_kohnShamDFTOperatorPtr;
#ifdef DFTFE_WITH_DEVICE
    kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
      &kohnShamDFTEigenOperatorDevice = *d_kohnShamDFTOperatorDevicePtr;
#endif

    const Quadrature<3> &quadrature =
      matrix_free_data.get_quadrature(d_densityQuadratureId);

    // computingTimerStandard.enter_subsection("Total scf solve");
    energyCalculator energyCalc(d_mpiCommParent,
                                mpi_communicator,
                                interpoolcomm,
                                interBandGroupComm,
                                *d_dftParamsPtr);


    // set up linear solver
    dealiiLinearSolver CGSolver(d_mpiCommParent,
                                mpi_communicator,
                                dealiiLinearSolver::CG);

    // set up linear solver Device
#ifdef DFTFE_WITH_DEVICE
    linearSolverCGDevice CGSolverDevice(d_mpiCommParent,
                                        mpi_communicator,
                                        linearSolverCGDevice::CG);
#endif

    //
    // set up solver functions for Helmholtz to be used only when Kerker mixing
    // is on use higher polynomial order dofHandler
    //
    kerkerSolverProblem<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>
      kerkerPreconditionedResidualSolverProblem(d_mpiCommParent,
                                                mpi_communicator);

    // set up solver functions for Helmholtz Device
#ifdef DFTFE_WITH_DEVICE
    kerkerSolverProblemDevice<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>
      kerkerPreconditionedResidualSolverProblemDevice(d_mpiCommParent,
                                                      mpi_communicator);
#endif

    if (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER")
      {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
        if (d_dftParamsPtr->useDevice and
            d_dftParamsPtr->floatingNuclearCharges)
#else
        if (false)
#endif
          {
#ifdef DFTFE_WITH_DEVICE
            kerkerPreconditionedResidualSolverProblemDevice.init(
              d_matrixFreeDataPRefined,
              d_constraintsForHelmholtzRhoNodal,
              d_preCondResidualVector,
              d_dftParamsPtr->kerkerParameter,
              d_helmholtzDofHandlerIndexElectro,
              d_densityQuadratureIdElectro);
#endif
          }
        else
          kerkerPreconditionedResidualSolverProblem.init(
            d_matrixFreeDataPRefined,
            d_constraintsForHelmholtzRhoNodal,
            d_preCondResidualVector,
            d_dftParamsPtr->kerkerParameter,
            d_helmholtzDofHandlerIndexElectro,
            d_densityQuadratureIdElectro);
      }

    // FIXME: Check if this call can be removed
    d_phiTotalSolverProblem.clear();

    //
    // solve vself in bins
    //
    computing_timer.enter_subsection("Nuclear self-potential solve");
    computingTimerStandard.enter_subsection("Nuclear self-potential solve");
#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice)
      d_vselfBinsManager.solveVselfInBinsDevice(
        d_matrixFreeDataPRefined,
        d_baseDofHandlerIndexElectro,
        d_phiTotAXQuadratureIdElectro,
        d_binsStartDofHandlerIndexElectro,
        kohnShamDFTEigenOperatorDevice,
        d_constraintsPRefined,
        d_imagePositionsTrunc,
        d_imageIdsTrunc,
        d_imageChargesTrunc,
        d_localVselfs,
        d_bQuadValuesAllAtoms,
        d_bQuadAtomIdsAllAtoms,
        d_bQuadAtomIdsAllAtomsImages,
        d_bCellNonTrivialAtomIds,
        d_bCellNonTrivialAtomIdsBins,
        d_bCellNonTrivialAtomImageIds,
        d_bCellNonTrivialAtomImageIdsBins,
        d_smearedChargeWidths,
        d_smearedChargeScaling,
        d_smearedChargeQuadratureIdElectro,
        d_dftParamsPtr->smearedNuclearCharges);
    else
      d_vselfBinsManager.solveVselfInBins(
        d_matrixFreeDataPRefined,
        d_binsStartDofHandlerIndexElectro,
        d_phiTotAXQuadratureIdElectro,
        d_constraintsPRefined,
        d_imagePositionsTrunc,
        d_imageIdsTrunc,
        d_imageChargesTrunc,
        d_localVselfs,
        d_bQuadValuesAllAtoms,
        d_bQuadAtomIdsAllAtoms,
        d_bQuadAtomIdsAllAtomsImages,
        d_bCellNonTrivialAtomIds,
        d_bCellNonTrivialAtomIdsBins,
        d_bCellNonTrivialAtomImageIds,
        d_bCellNonTrivialAtomImageIdsBins,
        d_smearedChargeWidths,
        d_smearedChargeScaling,
        d_smearedChargeQuadratureIdElectro,
        d_dftParamsPtr->smearedNuclearCharges);
#else
    d_vselfBinsManager.solveVselfInBins(d_matrixFreeDataPRefined,
                                        d_binsStartDofHandlerIndexElectro,
                                        d_phiTotAXQuadratureIdElectro,
                                        d_constraintsPRefined,
                                        d_imagePositionsTrunc,
                                        d_imageIdsTrunc,
                                        d_imageChargesTrunc,
                                        d_localVselfs,
                                        d_bQuadValuesAllAtoms,
                                        d_bQuadAtomIdsAllAtoms,
                                        d_bQuadAtomIdsAllAtomsImages,
                                        d_bCellNonTrivialAtomIds,
                                        d_bCellNonTrivialAtomIdsBins,
                                        d_bCellNonTrivialAtomImageIds,
                                        d_bCellNonTrivialAtomImageIdsBins,
                                        d_smearedChargeWidths,
                                        d_smearedChargeScaling,
                                        d_smearedChargeQuadratureIdElectro,
                                        d_dftParamsPtr->smearedNuclearCharges);
#endif
    computingTimerStandard.leave_subsection("Nuclear self-potential solve");
    computing_timer.leave_subsection("Nuclear self-potential solve");

    if ((d_dftParamsPtr->isPseudopotential ||
         d_dftParamsPtr->smearedNuclearCharges))
      {
        computingTimerStandard.enter_subsection("Init local PSP");
        initLocalPseudoPotential(d_dofHandlerPRefined,
                                 d_lpspQuadratureIdElectro,
                                 d_matrixFreeDataPRefined,
                                 d_phiExtDofHandlerIndexElectro,
                                 d_constraintsPRefinedOnlyHanging,
                                 d_supportPointsPRefined,
                                 d_vselfBinsManager,
                                 d_phiExt,
                                 d_pseudoVLoc,
                                 d_pseudoVLocAtoms);

        computingTimerStandard.leave_subsection("Init local PSP");
      }


    computingTimerStandard.enter_subsection("Total scf solve");

    //
    // solve
    //
    computing_timer.enter_subsection("scf solve");

    double firstScfChebyTol =
      d_dftParamsPtr->restrictToOnePass ?
        1e+4 :
        (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER" ? 1e-2 : 2e-2);


    if (d_dftParamsPtr->solverMode == "MD")
      firstScfChebyTol = d_dftParamsPtr->chebyshevTolerance > 1e-4 ?
                           1e-4 :
                           d_dftParamsPtr->chebyshevTolerance;
    else if (d_dftParamsPtr->solverMode == "GEOOPT")
      firstScfChebyTol = d_dftParamsPtr->chebyshevTolerance > 1e-3 ?
                           1e-3 :
                           d_dftParamsPtr->chebyshevTolerance;

    //
    // Begin SCF iteration
    //
    unsigned int scfIter                  = 0;
    double       norm                     = 1.0;
    d_rankCurrentLRD                      = 0;
    d_relativeErrorJacInvApproxPrevScfLRD = 100.0;
    // CAUTION: Choosing a looser tolerance might lead to failed tests
    const double adaptiveChebysevFilterPassesTol =
      d_dftParamsPtr->chebyshevTolerance;
    bool scfConverged = false;
    pcout << std::endl;
    if (d_dftParamsPtr->verbosity == 0)
      pcout << "Starting SCF iterations...." << std::endl;
    while ((norm > d_dftParamsPtr->selfConsistentSolverTolerance) &&
           (scfIter < d_dftParamsPtr->numSCFIterations))
      {
        dealii::Timer local_timer(d_mpiCommParent, true);
        if (d_dftParamsPtr->verbosity >= 1)
          pcout
            << "************************Begin Self-Consistent-Field Iteration: "
            << std::setw(2) << scfIter + 1 << " ***********************"
            << std::endl;
        //
        // Mixing scheme
        //
        computing_timer.enter_subsection("density mixing");
        if (scfIter > 0)
          {
            if (scfIter == 1)
              {
                if (d_dftParamsPtr->spinPolarized == 1)
                  {
                    if (d_dftParamsPtr->mixingMethod ==
                        "LOW_RANK_DIELECM_PRECOND")
                      norm = lowrankApproxScfDielectricMatrixInvSpinPolarized(
                        scfIter);
                    else
                      norm = mixing_simple_spinPolarized();
                  }
                else
                  {
                    if (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER")
                      {
                        norm = nodalDensity_mixing_simple_kerker(
#ifdef DFTFE_WITH_DEVICE
                          kerkerPreconditionedResidualSolverProblemDevice,
                          CGSolverDevice,
#endif
                          kerkerPreconditionedResidualSolverProblem,
                          CGSolver);
                      }
                    else if (d_dftParamsPtr->mixingMethod ==
                             "LOW_RANK_DIELECM_PRECOND")
                      norm = lowrankApproxScfDielectricMatrixInv(scfIter);
                    else
                      norm = mixing_simple();
                  }

                if (d_dftParamsPtr->verbosity >= 1)
                  {
                    pcout << d_dftParamsPtr->mixingMethod
                          << " mixing, L2 norm of electron-density difference: "
                          << norm << std::endl;
                  }
              }
            else
              {
                if (d_dftParamsPtr->spinPolarized == 1)
                  {
                    if (d_dftParamsPtr->mixingMethod == "ANDERSON")
                      norm = mixing_anderson_spinPolarized();
                    else if (d_dftParamsPtr->mixingMethod == "BROYDEN")
                      norm = mixing_broyden_spinPolarized();
                    else if (d_dftParamsPtr->mixingMethod ==
                             "LOW_RANK_DIELECM_PRECOND")
                      norm = lowrankApproxScfDielectricMatrixInvSpinPolarized(
                        scfIter);
                    else if (d_dftParamsPtr->mixingMethod ==
                             "ANDERSON_WITH_KERKER")
                      AssertThrow(
                        false,
                        ExcMessage(
                          "Kerker is not implemented for spin-polarized problems yet"));
                  }
                else
                  {
                    if (d_dftParamsPtr->mixingMethod == "ANDERSON")
                      norm = mixing_anderson();
                    else if (d_dftParamsPtr->mixingMethod == "BROYDEN")
                      norm = mixing_broyden();
                    else if (d_dftParamsPtr->mixingMethod ==
                             "ANDERSON_WITH_KERKER")
                      {
                        norm = nodalDensity_mixing_anderson_kerker(
#ifdef DFTFE_WITH_DEVICE
                          kerkerPreconditionedResidualSolverProblemDevice,
                          CGSolverDevice,
#endif
                          kerkerPreconditionedResidualSolverProblem,
                          CGSolver);
                      }
                    else if (d_dftParamsPtr->mixingMethod ==
                             "LOW_RANK_DIELECM_PRECOND")
                      norm = lowrankApproxScfDielectricMatrixInv(scfIter);
                  }

                if (d_dftParamsPtr->verbosity >= 1)
                  pcout << d_dftParamsPtr->mixingMethod
                        << " mixing, L2 norm of electron-density difference: "
                        << norm << std::endl;
              }

            if (d_dftParamsPtr->computeEnergyEverySCF &&
                d_numEigenValuesRR == d_numEigenValues)
              d_phiTotRhoIn = d_phiTotRhoOut;
          }
        computing_timer.leave_subsection("density mixing");

        if (!(norm > d_dftParamsPtr->selfConsistentSolverTolerance))
          scfConverged = true;
        //
        // phiTot with rhoIn
        //
        if (d_dftParamsPtr->verbosity >= 2)
          pcout
            << std::endl
            << "Poisson solve for total electrostatic potential (rhoIn+b): ";

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
        if (d_dftParamsPtr->useDevice and
            d_dftParamsPtr->floatingNuclearCharges and
            not d_dftParamsPtr->pinnedNodeForPBC)
#else
        if (false)
#endif
          {
#ifdef DFTFE_WITH_DEVICE
            if (scfIter > 0)
              d_phiTotalSolverProblemDevice.reinit(
                d_matrixFreeDataPRefined,
                d_phiTotRhoIn,
                *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                d_phiTotDofHandlerIndexElectro,
                d_densityQuadratureIdElectro,
                d_phiTotAXQuadratureIdElectro,
                d_atomNodeIdToChargeMap,
                d_bQuadValuesAllAtoms,
                d_smearedChargeQuadratureIdElectro,
                *rhoInValues,
                kohnShamDFTEigenOperatorDevice.getDeviceBlasHandle(),
                false,
                false,
                d_dftParamsPtr->smearedNuclearCharges,
                true,
                false,
                0,
                false,
                true);
            else
              {
                d_phiTotalSolverProblemDevice.reinit(
                  d_matrixFreeDataPRefined,
                  d_phiTotRhoIn,
                  *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                  d_phiTotDofHandlerIndexElectro,
                  d_densityQuadratureIdElectro,
                  d_phiTotAXQuadratureIdElectro,
                  d_atomNodeIdToChargeMap,
                  d_bQuadValuesAllAtoms,
                  d_smearedChargeQuadratureIdElectro,
                  *rhoInValues,
                  kohnShamDFTEigenOperatorDevice.getDeviceBlasHandle(),
                  true,
                  d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
                    d_dftParamsPtr->periodicZ &&
                    !d_dftParamsPtr->pinnedNodeForPBC,
                  d_dftParamsPtr->smearedNuclearCharges,
                  true,
                  false,
                  0,
                  true,
                  false);
              }
#endif
          }
        else
          {
            if (scfIter > 0)
              d_phiTotalSolverProblem.reinit(
                d_matrixFreeDataPRefined,
                d_phiTotRhoIn,
                *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                d_phiTotDofHandlerIndexElectro,
                d_densityQuadratureIdElectro,
                d_phiTotAXQuadratureIdElectro,
                d_atomNodeIdToChargeMap,
                d_bQuadValuesAllAtoms,
                d_smearedChargeQuadratureIdElectro,
                *rhoInValues,
                false,
                false,
                d_dftParamsPtr->smearedNuclearCharges,
                true,
                false,
                0,
                false,
                true);
            else
              d_phiTotalSolverProblem.reinit(
                d_matrixFreeDataPRefined,
                d_phiTotRhoIn,
                *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                d_phiTotDofHandlerIndexElectro,
                d_densityQuadratureIdElectro,
                d_phiTotAXQuadratureIdElectro,
                d_atomNodeIdToChargeMap,
                d_bQuadValuesAllAtoms,
                d_smearedChargeQuadratureIdElectro,
                *rhoInValues,
                true,
                d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
                  d_dftParamsPtr->periodicZ &&
                  !d_dftParamsPtr->pinnedNodeForPBC,
                d_dftParamsPtr->smearedNuclearCharges,
                true,
                false,
                0,
                true,
                false);
          }

        computing_timer.enter_subsection("phiTot solve");

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
        if (d_dftParamsPtr->useDevice and
            d_dftParamsPtr->floatingNuclearCharges and
            not d_dftParamsPtr->pinnedNodeForPBC)
#else
        if (false)
#endif
          {
#ifdef DFTFE_WITH_DEVICE
            CGSolverDevice.solve(
              d_phiTotalSolverProblemDevice,
              d_dftParamsPtr->relLinearSolverTolerance,
              d_dftParamsPtr->maxLinearSolverIterations,
              kohnShamDFTEigenOperatorDevice.getDeviceBlasHandle(),
              d_dftParamsPtr->verbosity);
#endif
          }
        else
          {
            CGSolver.solve(d_phiTotalSolverProblem,
                           d_dftParamsPtr->relLinearSolverTolerance,
                           d_dftParamsPtr->maxLinearSolverIterations,
                           d_dftParamsPtr->verbosity);
          }

        d_phiTotRhoIn.update_ghost_values();

        std::map<dealii::CellId, std::vector<double>> dummy;
        interpolateElectroNodalDataToQuadratureDataGeneral(
          d_matrixFreeDataPRefined,
          d_phiTotDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          d_phiTotRhoIn,
          d_phiInValues,
          dummy);

        //
        // impose integral phi equals 0
        //
        /*
        if(d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
        d_dftParamsPtr->periodicZ && !d_dftParamsPtr->pinnedNodeForPBC)
        {
          if (d_dftParamsPtr->verbosity>=2)
            pcout<<"Value of integPhiIn:
        "<<totalCharge(d_dofHandlerPRefined,d_phiTotRhoIn)<<std::endl;
        }
        */

        computing_timer.leave_subsection("phiTot solve");

        unsigned int numberChebyshevSolvePasses = 0;
        //
        // eigen solve
        //
        if (d_dftParamsPtr->spinPolarized == 1)
          {
            std::vector<std::vector<std::vector<double>>> eigenValuesSpins(
              2,
              std::vector<std::vector<double>>(
                d_kPointWeights.size(),
                std::vector<double>(
                  (scfIter < d_dftParamsPtr->spectrumSplitStartingScfIter ||
                   scfConverged) ?
                    d_numEigenValues :
                    d_numEigenValuesRR)));

            std::vector<std::vector<std::vector<double>>>
              residualNormWaveFunctionsAllkPointsSpins(
                2,
                std::vector<std::vector<double>>(
                  d_kPointWeights.size(),
                  std::vector<double>(
                    (scfIter < d_dftParamsPtr->spectrumSplitStartingScfIter ||
                     scfConverged) ?
                      d_numEigenValues :
                      d_numEigenValuesRR)));

            for (unsigned int s = 0; s < 2; ++s)
              {
                if (excFunctionalPtr->getDensityBasedFamilyType() ==
                    densityFamilyType::LDA)
                  {
                    computing_timer.enter_subsection("VEff Computation");
#ifdef DFTFE_WITH_DEVICE
                    if (d_dftParamsPtr->useDevice)
                      kohnShamDFTEigenOperatorDevice.computeVEffSpinPolarized(
                        rhoInValuesSpinPolarized,
                        d_phiInValues,
                        s,
                        d_pseudoVLoc,
                        d_rhoCore,
                        d_lpspQuadratureId);
#endif
                    if (!d_dftParamsPtr->useDevice)
                      kohnShamDFTEigenOperator.computeVEffSpinPolarized(
                        rhoInValuesSpinPolarized,
                        d_phiInValues,
                        s,
                        d_pseudoVLoc,
                        d_rhoCore,
                        d_lpspQuadratureId);
                    computing_timer.leave_subsection("VEff Computation");
                  }
                else if (excFunctionalPtr->getDensityBasedFamilyType() ==
                         densityFamilyType::GGA)
                  {
                    computing_timer.enter_subsection("VEff Computation");
#ifdef DFTFE_WITH_DEVICE
                    if (d_dftParamsPtr->useDevice)
                      kohnShamDFTEigenOperatorDevice.computeVEffSpinPolarized(
                        rhoInValuesSpinPolarized,
                        gradRhoInValuesSpinPolarized,
                        d_phiInValues,
                        s,
                        d_pseudoVLoc,
                        d_rhoCore,
                        d_gradRhoCore,
                        d_lpspQuadratureId);
#endif
                    if (!d_dftParamsPtr->useDevice)
                      kohnShamDFTEigenOperator.computeVEffSpinPolarized(
                        rhoInValuesSpinPolarized,
                        gradRhoInValuesSpinPolarized,
                        d_phiInValues,
                        s,
                        d_pseudoVLoc,
                        d_rhoCore,
                        d_gradRhoCore,
                        d_lpspQuadratureId);
                    computing_timer.leave_subsection("VEff Computation");
                  }

#ifdef DFTFE_WITH_DEVICE
                if (d_dftParamsPtr->useDevice)
                  {
                    computing_timer.enter_subsection(
                      "Hamiltonian Matrix Computation");
                    kohnShamDFTEigenOperatorDevice
                      .computeHamiltonianMatricesAllkpt(s);
                    computing_timer.leave_subsection(
                      "Hamiltonian Matrix Computation");
                  }
#endif


                for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size();
                     ++kPoint)
                  {
#ifdef DFTFE_WITH_DEVICE
                    if (d_dftParamsPtr->useDevice)
                      kohnShamDFTEigenOperatorDevice.reinitkPointSpinIndex(
                        kPoint, s);
#endif
                    if (!d_dftParamsPtr->useDevice)
                      kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint, s);



                    if (!d_dftParamsPtr->useDevice)
                      {
                        computing_timer.enter_subsection(
                          "Hamiltonian Matrix Computation");
                        kohnShamDFTEigenOperator.computeHamiltonianMatrix(
                          kPoint, s);
                        computing_timer.leave_subsection(
                          "Hamiltonian Matrix Computation");
                      }


                    for (unsigned int j = 0; j < 1; ++j)
                      {
                        if (d_dftParamsPtr->verbosity >= 2)
                          {
                            pcout << "Beginning Chebyshev filter pass " << j + 1
                                  << " for spin " << s + 1 << std::endl;
                          }

#ifdef DFTFE_WITH_DEVICE
                        if (d_dftParamsPtr->useDevice)
                          kohnShamEigenSpaceCompute(
                            s,
                            kPoint,
                            kohnShamDFTEigenOperatorDevice,
                            *d_elpaScala,
                            d_subspaceIterationSolverDevice,
                            residualNormWaveFunctionsAllkPointsSpins[s][kPoint],
                            (scfIter == 0 ||
                             d_dftParamsPtr
                               ->allowMultipleFilteringPassesAfterFirstScf) ?
                              true :
                              false,
                            0,
                            (scfIter <
                               d_dftParamsPtr->spectrumSplitStartingScfIter ||
                             scfConverged) ?
                              false :
                              true,
                            scfConverged ? false : true,
                            scfIter == 0);
#endif
                        if (!d_dftParamsPtr->useDevice)
                          kohnShamEigenSpaceCompute(
                            s,
                            kPoint,
                            kohnShamDFTEigenOperator,
                            *d_elpaScala,
                            d_subspaceIterationSolver,
                            residualNormWaveFunctionsAllkPointsSpins[s][kPoint],
                            (scfIter == 0 ||
                             d_dftParamsPtr
                               ->allowMultipleFilteringPassesAfterFirstScf) ?
                              true :
                              false,
                            (scfIter <
                               d_dftParamsPtr->spectrumSplitStartingScfIter ||
                             scfConverged) ?
                              false :
                              true,
                            scfConverged ? false : true,
                            scfIter == 0);
                      }
                  }
              }


            for (unsigned int s = 0; s < 2; ++s)
              for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size();
                   ++kPoint)
                {
                  if (scfIter < d_dftParamsPtr->spectrumSplitStartingScfIter ||
                      scfConverged)
                    for (unsigned int i = 0; i < d_numEigenValues; ++i)
                      eigenValuesSpins[s][kPoint][i] =
                        eigenValues[kPoint][d_numEigenValues * s + i];
                  else
                    for (unsigned int i = 0; i < d_numEigenValuesRR; ++i)
                      eigenValuesSpins[s][kPoint][i] =
                        eigenValuesRRSplit[kPoint][d_numEigenValuesRR * s + i];
                }
            //
            // fermi energy
            //
            if (d_dftParamsPtr->constraintMagnetization)
              compute_fermienergy_constraintMagnetization(eigenValues);
            else
              compute_fermienergy(eigenValues, numElectrons);

            unsigned int count = 1;

            if (!scfConverged &&
                (scfIter == 0 ||
                 d_dftParamsPtr->allowMultipleFilteringPassesAfterFirstScf))
              {
                // maximum of the residual norm of the state closest to and
                // below the Fermi level among all k points, and also the
                // maximum between the two spins
                double maxRes =
                  std::max(computeMaximumHighestOccupiedStateResidualNorm(
                             residualNormWaveFunctionsAllkPointsSpins[0],
                             eigenValuesSpins[0],
                             fermiEnergy),
                           computeMaximumHighestOccupiedStateResidualNorm(
                             residualNormWaveFunctionsAllkPointsSpins[1],
                             eigenValuesSpins[1],
                             fermiEnergy));

                if (d_dftParamsPtr->verbosity >= 2)
                  {
                    pcout
                      << "Maximum residual norm of the state closest to and below Fermi level: "
                      << maxRes << std::endl;
                  }

                // if the residual norm is greater than
                // adaptiveChebysevFilterPassesTol (a heuristic value)
                // do more passes of chebysev filter till the check passes.
                // This improves the scf convergence performance.

                const double filterPassTol =
                  (scfIter == 0 && isRestartGroundStateCalcFromChk) ?
                    1.0e-8 :
                    ((scfIter == 0 &&
                      adaptiveChebysevFilterPassesTol > firstScfChebyTol) ?
                       firstScfChebyTol :
                       adaptiveChebysevFilterPassesTol);
                while (maxRes > filterPassTol && count < 100)
                  {
                    for (unsigned int s = 0; s < 2; ++s)
                      {
                        for (unsigned int kPoint = 0;
                             kPoint < d_kPointWeights.size();
                             ++kPoint)
                          {
                            if (d_dftParamsPtr->verbosity >= 2)
                              pcout << "Beginning Chebyshev filter pass "
                                    << 1 + count << " for spin " << s + 1
                                    << std::endl;
                            ;

#ifdef DFTFE_WITH_DEVICE
                            if (d_dftParamsPtr->useDevice)
                              kohnShamDFTEigenOperatorDevice
                                .reinitkPointSpinIndex(kPoint, s);
#endif
                            if (!d_dftParamsPtr->useDevice)
                              kohnShamDFTEigenOperator.reinitkPointSpinIndex(
                                kPoint, s);

#ifdef DFTFE_WITH_DEVICE
                            if (d_dftParamsPtr->useDevice)
                              kohnShamEigenSpaceCompute(
                                s,
                                kPoint,
                                kohnShamDFTEigenOperatorDevice,
                                *d_elpaScala,
                                d_subspaceIterationSolverDevice,
                                residualNormWaveFunctionsAllkPointsSpins
                                  [s][kPoint],
                                true,
                                0,
                                (scfIter <
                                 d_dftParamsPtr->spectrumSplitStartingScfIter) ?
                                  false :
                                  true,
                                true,
                                scfIter == 0);
#endif
                            if (!d_dftParamsPtr->useDevice)
                              kohnShamEigenSpaceCompute(
                                s,
                                kPoint,
                                kohnShamDFTEigenOperator,
                                *d_elpaScala,
                                d_subspaceIterationSolver,
                                residualNormWaveFunctionsAllkPointsSpins
                                  [s][kPoint],
                                true,
                                (scfIter <
                                 d_dftParamsPtr->spectrumSplitStartingScfIter) ?
                                  false :
                                  true,
                                true,
                                scfIter == 0);
                          }
                      }

                    for (unsigned int s = 0; s < 2; ++s)
                      for (unsigned int kPoint = 0;
                           kPoint < d_kPointWeights.size();
                           ++kPoint)
                        {
                          if (scfIter <
                                d_dftParamsPtr->spectrumSplitStartingScfIter ||
                              scfConverged)
                            for (unsigned int i = 0; i < d_numEigenValues; ++i)
                              eigenValuesSpins[s][kPoint][i] =
                                eigenValues[kPoint][d_numEigenValues * s + i];
                          else
                            for (unsigned int i = 0; i < d_numEigenValuesRR;
                                 ++i)
                              eigenValuesSpins[s][kPoint][i] =
                                eigenValuesRRSplit[kPoint]
                                                  [d_numEigenValuesRR * s + i];
                        }
                    //
                    if (d_dftParamsPtr->constraintMagnetization)
                      compute_fermienergy_constraintMagnetization(eigenValues);
                    else
                      compute_fermienergy(eigenValues, numElectrons);
                    //
                    maxRes =
                      std::max(computeMaximumHighestOccupiedStateResidualNorm(
                                 residualNormWaveFunctionsAllkPointsSpins[0],
                                 eigenValuesSpins[0],
                                 fermiEnergy),
                               computeMaximumHighestOccupiedStateResidualNorm(
                                 residualNormWaveFunctionsAllkPointsSpins[1],
                                 eigenValuesSpins[1],
                                 fermiEnergy));
                    if (d_dftParamsPtr->verbosity >= 2)
                      pcout
                        << "Maximum residual norm of the state closest to and below Fermi level: "
                        << maxRes << std::endl;
                    count++;
                  }
              }

            if (d_dftParamsPtr->verbosity >= 1)
              {
                pcout << "Fermi Energy computed: " << fermiEnergy << std::endl;
              }

            numberChebyshevSolvePasses = count;
          }
        else
          {
            std::vector<std::vector<double>>
              residualNormWaveFunctionsAllkPoints;
            residualNormWaveFunctionsAllkPoints.resize(d_kPointWeights.size());
            for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size();
                 ++kPoint)
              residualNormWaveFunctionsAllkPoints[kPoint].resize(
                (scfIter < d_dftParamsPtr->spectrumSplitStartingScfIter ||
                 scfConverged) ?
                  d_numEigenValues :
                  d_numEigenValuesRR);

            if (excFunctionalPtr->getDensityBasedFamilyType() ==
                densityFamilyType::LDA)
              {
                computing_timer.enter_subsection("VEff Computation");
#ifdef DFTFE_WITH_DEVICE
                if (d_dftParamsPtr->useDevice)
                  kohnShamDFTEigenOperatorDevice.computeVEff(
                    rhoInValues,
                    d_phiInValues,
                    d_pseudoVLoc,
                    d_rhoCore,
                    d_lpspQuadratureId);
#endif
                if (!d_dftParamsPtr->useDevice)
                  kohnShamDFTEigenOperator.computeVEff(rhoInValues,
                                                       d_phiInValues,
                                                       d_pseudoVLoc,
                                                       d_rhoCore,
                                                       d_lpspQuadratureId);
                computing_timer.leave_subsection("VEff Computation");
              }
            else if (excFunctionalPtr->getDensityBasedFamilyType() ==
                     densityFamilyType::GGA)
              {
                computing_timer.enter_subsection("VEff Computation");
#ifdef DFTFE_WITH_DEVICE
                if (d_dftParamsPtr->useDevice)
                  kohnShamDFTEigenOperatorDevice.computeVEff(
                    rhoInValues,
                    gradRhoInValues,
                    d_phiInValues,
                    d_pseudoVLoc,
                    d_rhoCore,
                    d_gradRhoCore,
                    d_lpspQuadratureId);
#endif
                if (!d_dftParamsPtr->useDevice)
                  kohnShamDFTEigenOperator.computeVEff(rhoInValues,
                                                       gradRhoInValues,
                                                       d_phiInValues,
                                                       d_pseudoVLoc,
                                                       d_rhoCore,
                                                       d_gradRhoCore,
                                                       d_lpspQuadratureId);
                computing_timer.leave_subsection("VEff Computation");
              }

#ifdef DFTFE_WITH_DEVICE
            if (d_dftParamsPtr->useDevice)
              {
                computing_timer.enter_subsection(
                  "Hamiltonian Matrix Computation");
                kohnShamDFTEigenOperatorDevice.computeHamiltonianMatricesAllkpt(
                  0);
                computing_timer.leave_subsection(
                  "Hamiltonian Matrix Computation");
              }
#endif

            for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size();
                 ++kPoint)
              {
#ifdef DFTFE_WITH_DEVICE
                if (d_dftParamsPtr->useDevice)
                  kohnShamDFTEigenOperatorDevice.reinitkPointSpinIndex(kPoint,
                                                                       0);
#endif
                if (!d_dftParamsPtr->useDevice)
                  kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint, 0);


                if (!d_dftParamsPtr->useDevice)
                  {
                    computing_timer.enter_subsection(
                      "Hamiltonian Matrix Computation");
                    kohnShamDFTEigenOperator.computeHamiltonianMatrix(kPoint,
                                                                      0);
                    computing_timer.leave_subsection(
                      "Hamiltonian Matrix Computation");
                  }


                for (unsigned int j = 0; j < 1; ++j)
                  {
                    if (d_dftParamsPtr->verbosity >= 2)
                      {
                        pcout << "Beginning Chebyshev filter pass " << j + 1
                              << std::endl;
                      }


#ifdef DFTFE_WITH_DEVICE
                    if (d_dftParamsPtr->useDevice)
                      kohnShamEigenSpaceCompute(
                        0,
                        kPoint,
                        kohnShamDFTEigenOperatorDevice,
                        *d_elpaScala,
                        d_subspaceIterationSolverDevice,
                        residualNormWaveFunctionsAllkPoints[kPoint],
                        (scfIter == 0 ||
                         d_dftParamsPtr
                           ->allowMultipleFilteringPassesAfterFirstScf) ?
                          true :
                          false,
                        0,
                        (scfIter <
                           d_dftParamsPtr->spectrumSplitStartingScfIter ||
                         scfConverged) ?
                          false :
                          true,
                        scfConverged ? false : true,
                        scfIter == 0);
#endif
                    if (!d_dftParamsPtr->useDevice)
                      kohnShamEigenSpaceCompute(
                        0,
                        kPoint,
                        kohnShamDFTEigenOperator,
                        *d_elpaScala,
                        d_subspaceIterationSolver,
                        residualNormWaveFunctionsAllkPoints[kPoint],
                        (scfIter == 0 ||
                         d_dftParamsPtr
                           ->allowMultipleFilteringPassesAfterFirstScf) ?
                          true :
                          false,
                        (scfIter <
                           d_dftParamsPtr->spectrumSplitStartingScfIter ||
                         scfConverged) ?
                          false :
                          true,
                        scfConverged ? false : true,
                        scfIter == 0);
                  }
              }


            //
            // fermi energy
            //
            if (d_dftParamsPtr->constraintMagnetization)
              compute_fermienergy_constraintMagnetization(eigenValues);
            else
              compute_fermienergy(eigenValues, numElectrons);

            unsigned int count = 1;

            if (!scfConverged &&
                (scfIter == 0 ||
                 d_dftParamsPtr->allowMultipleFilteringPassesAfterFirstScf))
              {
                //
                // maximum of the residual norm of the state closest to and
                // below the Fermi level among all k points
                //
                double maxRes = computeMaximumHighestOccupiedStateResidualNorm(
                  residualNormWaveFunctionsAllkPoints,
                  (scfIter < d_dftParamsPtr->spectrumSplitStartingScfIter) ?
                    eigenValues :
                    eigenValuesRRSplit,
                  fermiEnergy);
                if (d_dftParamsPtr->verbosity >= 2)
                  pcout
                    << "Maximum residual norm of the state closest to and below Fermi level: "
                    << maxRes << std::endl;

                // if the residual norm is greater than
                // adaptiveChebysevFilterPassesTol (a heuristic value)
                // do more passes of chebysev filter till the check passes.
                // This improves the scf convergence performance.

                const double filterPassTol =
                  (scfIter == 0 && isRestartGroundStateCalcFromChk) ?
                    1.0e-8 :
                    ((scfIter == 0 &&
                      adaptiveChebysevFilterPassesTol > firstScfChebyTol) ?
                       firstScfChebyTol :
                       adaptiveChebysevFilterPassesTol);
                while (maxRes > filterPassTol && count < 100)
                  {
                    for (unsigned int kPoint = 0;
                         kPoint < d_kPointWeights.size();
                         ++kPoint)
                      {
                        if (d_dftParamsPtr->verbosity >= 2)
                          pcout << "Beginning Chebyshev filter pass "
                                << 1 + count << std::endl;

#ifdef DFTFE_WITH_DEVICE
                        if (d_dftParamsPtr->useDevice)
                          kohnShamDFTEigenOperatorDevice.reinitkPointSpinIndex(
                            kPoint, 0);
#endif
                        if (!d_dftParamsPtr->useDevice)
                          kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint,
                                                                         0);

#ifdef DFTFE_WITH_DEVICE
                        if (d_dftParamsPtr->useDevice)
                          kohnShamEigenSpaceCompute(
                            0,
                            kPoint,
                            kohnShamDFTEigenOperatorDevice,
                            *d_elpaScala,
                            d_subspaceIterationSolverDevice,
                            residualNormWaveFunctionsAllkPoints[kPoint],
                            true,
                            0,
                            (scfIter <
                             d_dftParamsPtr->spectrumSplitStartingScfIter) ?
                              false :
                              true,
                            true,
                            scfIter == 0);

#endif
                        if (!d_dftParamsPtr->useDevice)
                          kohnShamEigenSpaceCompute(
                            0,
                            kPoint,
                            kohnShamDFTEigenOperator,
                            *d_elpaScala,
                            d_subspaceIterationSolver,
                            residualNormWaveFunctionsAllkPoints[kPoint],
                            true,
                            (scfIter <
                             d_dftParamsPtr->spectrumSplitStartingScfIter) ?
                              false :
                              true,
                            true,
                            scfIter == 0);
                      }

                    //
                    if (d_dftParamsPtr->constraintMagnetization)
                      compute_fermienergy_constraintMagnetization(eigenValues);
                    else
                      compute_fermienergy(eigenValues, numElectrons);
                    //
                    maxRes = computeMaximumHighestOccupiedStateResidualNorm(
                      residualNormWaveFunctionsAllkPoints,
                      (scfIter < d_dftParamsPtr->spectrumSplitStartingScfIter ||
                       scfConverged) ?
                        eigenValues :
                        eigenValuesRRSplit,
                      fermiEnergy);
                    if (d_dftParamsPtr->verbosity >= 2)
                      pcout
                        << "Maximum residual norm of the state closest to and below Fermi level: "
                        << maxRes << std::endl;

                    count++;
                  }
              }

            numberChebyshevSolvePasses = count;

            if (d_dftParamsPtr->verbosity >= 1)
              {
                pcout << "Fermi Energy computed: " << fermiEnergy << std::endl;
              }
          }
        computing_timer.enter_subsection("compute rho");
        if (d_dftParamsPtr->useSymm)
          {
#ifdef USE_COMPLEX
            symmetryPtr->computeLocalrhoOut();
            symmetryPtr->computeAndSymmetrize_rhoOut();

            std::function<double(
              const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
              const unsigned int                                          q)>
              funcRho =
                [&](const typename dealii::DoFHandler<3>::active_cell_iterator
                      &                cell,
                    const unsigned int q) {
                  return (*rhoOutValues).find(cell->id())->second[q];
                };
            dealii::VectorTools::project<3, distributedCPUVec<double>>(
              dealii::MappingQ1<3, 3>(),
              d_dofHandlerRhoNodal,
              d_constraintsRhoNodal,
              d_matrixFreeDataPRefined.get_quadrature(
                d_densityQuadratureIdElectro),
              funcRho,
              d_rhoOutNodalValues);
            d_rhoOutNodalValues.update_ghost_values();

            interpolateRhoNodalDataToQuadratureDataLpsp(
              d_matrixFreeDataPRefined,
              d_densityDofHandlerIndexElectro,
              d_lpspQuadratureIdElectro,
              d_rhoOutNodalValues,
              d_rhoOutValuesLpspQuad,
              d_gradRhoOutValuesLpspQuad,
              true);
#endif
          }
        else
          {
#ifdef DFTFE_WITH_DEVICE
            compute_rhoOut(
              kohnShamDFTEigenOperatorDevice,
              kohnShamDFTEigenOperator,
              (scfIter < d_dftParamsPtr->spectrumSplitStartingScfIter ||
               scfConverged) ?
                false :
                true,
              scfConverged ||
                (scfIter == (d_dftParamsPtr->numSCFIterations - 1)));
#else
            compute_rhoOut(
              kohnShamDFTEigenOperator,
              (scfIter < d_dftParamsPtr->spectrumSplitStartingScfIter ||
               scfConverged) ?
                false :
                true,
              scfConverged ||
                (scfIter == (d_dftParamsPtr->numSCFIterations - 1)));
#endif
          }
        computing_timer.leave_subsection("compute rho");

        //
        // compute integral rhoOut
        //
        const double integralRhoValue =
          totalCharge(d_dofHandlerPRefined, rhoOutValues);

        if (d_dftParamsPtr->verbosity >= 2)
          {
            pcout << std::endl
                  << "number of electrons: " << integralRhoValue << std::endl;
          }

        if (d_dftParamsPtr->verbosity >= 1 &&
            d_dftParamsPtr->spinPolarized == 1)
          pcout << std::endl
                << "net magnetization: "
                << totalMagnetization(rhoOutValuesSpinPolarized) << std::endl;

        //
        // phiTot with rhoOut
        //
        if (d_dftParamsPtr->computeEnergyEverySCF &&
            d_numEigenValuesRR == d_numEigenValues)
          {
            if (d_dftParamsPtr->verbosity >= 2)
              pcout
                << std::endl
                << "Poisson solve for total electrostatic potential (rhoOut+b): ";

            computing_timer.enter_subsection("phiTot solve");

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
            if (d_dftParamsPtr->useDevice and
                d_dftParamsPtr->floatingNuclearCharges and
                not d_dftParamsPtr->pinnedNodeForPBC)
#else
            if (false)
#endif
              {
#ifdef DFTFE_WITH_DEVICE
                d_phiTotalSolverProblemDevice.reinit(
                  d_matrixFreeDataPRefined,
                  d_phiTotRhoOut,
                  *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                  d_phiTotDofHandlerIndexElectro,
                  d_densityQuadratureIdElectro,
                  d_phiTotAXQuadratureIdElectro,
                  d_atomNodeIdToChargeMap,
                  d_bQuadValuesAllAtoms,
                  d_smearedChargeQuadratureIdElectro,
                  *rhoOutValues,
                  kohnShamDFTEigenOperatorDevice.getDeviceBlasHandle(),
                  false,
                  false,
                  d_dftParamsPtr->smearedNuclearCharges,
                  true,
                  false,
                  0,
                  false,
                  true);

                CGSolverDevice.solve(
                  d_phiTotalSolverProblemDevice,
                  d_dftParamsPtr->relLinearSolverTolerance,
                  d_dftParamsPtr->maxLinearSolverIterations,
                  kohnShamDFTEigenOperatorDevice.getDeviceBlasHandle(),
                  d_dftParamsPtr->verbosity);
#endif
              }
            else
              {
                d_phiTotalSolverProblem.reinit(
                  d_matrixFreeDataPRefined,
                  d_phiTotRhoOut,
                  *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
                  d_phiTotDofHandlerIndexElectro,
                  d_densityQuadratureIdElectro,
                  d_phiTotAXQuadratureIdElectro,
                  d_atomNodeIdToChargeMap,
                  d_bQuadValuesAllAtoms,
                  d_smearedChargeQuadratureIdElectro,
                  *rhoOutValues,
                  false,
                  false,
                  d_dftParamsPtr->smearedNuclearCharges,
                  true,
                  false,
                  0,
                  false,
                  true);

                CGSolver.solve(d_phiTotalSolverProblem,
                               d_dftParamsPtr->relLinearSolverTolerance,
                               d_dftParamsPtr->maxLinearSolverIterations,
                               d_dftParamsPtr->verbosity);
              }

            //
            // impose integral phi equals 0
            //
            /*
            if(d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
            d_dftParamsPtr->periodicZ && !d_dftParamsPtr->pinnedNodeForPBC)
            {
              if(d_dftParamsPtr->verbosity>=2)
                pcout<<"Value of integPhiOut:
            "<<totalCharge(d_dofHandlerPRefined,d_phiTotRhoOut);
            }
            */

            computing_timer.leave_subsection("phiTot solve");

            const Quadrature<3> &quadrature =
              matrix_free_data.get_quadrature(d_densityQuadratureId);
            d_dispersionCorr.computeDispresionCorrection(
              atomLocations, d_domainBoundingVectors);
            const double totalEnergy =
              d_dftParamsPtr->spinPolarized == 0 ?
                energyCalc.computeEnergy(
                  d_dofHandlerPRefined,
                  dofHandler,
                  quadrature,
                  quadrature,
                  d_matrixFreeDataPRefined.get_quadrature(
                    d_smearedChargeQuadratureIdElectro),
                  d_matrixFreeDataPRefined.get_quadrature(
                    d_lpspQuadratureIdElectro),
                  eigenValues,
                  d_kPointWeights,
                  fermiEnergy,
                  excFunctionalPtr,
                  d_dispersionCorr,
                  d_phiInValues,
                  d_phiTotRhoOut,
                  *rhoInValues,
                  *rhoOutValues,
                  d_rhoOutValuesLpspQuad,
                  *rhoOutValues,
                  d_rhoOutValuesLpspQuad,
                  *gradRhoInValues,
                  *gradRhoOutValues,
                  d_rhoCore,
                  d_gradRhoCore,
                  d_bQuadValuesAllAtoms,
                  d_bCellNonTrivialAtomIds,
                  d_localVselfs,
                  d_pseudoVLoc,
                  d_pseudoVLoc,
                  d_atomNodeIdToChargeMap,
                  atomLocations.size(),
                  lowerBoundKindex,
                  0,
                  d_dftParamsPtr->verbosity >= 2,
                  d_dftParamsPtr->smearedNuclearCharges) :
                energyCalc.computeEnergySpinPolarized(
                  d_dofHandlerPRefined,
                  dofHandler,
                  quadrature,
                  quadrature,
                  d_matrixFreeDataPRefined.get_quadrature(
                    d_smearedChargeQuadratureIdElectro),
                  d_matrixFreeDataPRefined.get_quadrature(
                    d_lpspQuadratureIdElectro),
                  eigenValues,
                  d_kPointWeights,
                  fermiEnergy,
                  fermiEnergyUp,
                  fermiEnergyDown,
                  excFunctionalPtr,
                  d_dispersionCorr,
                  d_phiInValues,
                  d_phiTotRhoOut,
                  *rhoInValues,
                  *rhoOutValues,
                  d_rhoOutValuesLpspQuad,
                  *rhoOutValues,
                  d_rhoOutValuesLpspQuad,
                  *gradRhoInValues,
                  *gradRhoOutValues,
                  *rhoInValuesSpinPolarized,
                  *rhoOutValuesSpinPolarized,
                  *gradRhoInValuesSpinPolarized,
                  *gradRhoOutValuesSpinPolarized,
                  d_rhoCore,
                  d_gradRhoCore,
                  d_bQuadValuesAllAtoms,
                  d_bCellNonTrivialAtomIds,
                  d_localVselfs,
                  d_pseudoVLoc,
                  d_pseudoVLoc,
                  d_atomNodeIdToChargeMap,
                  atomLocations.size(),
                  lowerBoundKindex,
                  0,
                  d_dftParamsPtr->verbosity >= 2,
                  d_dftParamsPtr->smearedNuclearCharges);
            if (d_dftParamsPtr->verbosity == 1)
              pcout << "Total energy  : " << totalEnergy << std::endl;
          }
        else
          {
            if (d_numEigenValuesRR != d_numEigenValues &&
                d_dftParamsPtr->computeEnergyEverySCF &&
                d_dftParamsPtr->verbosity >= 1)
              pcout
                << "DFT-FE Message: energy computation is not performed at the end of each scf iteration step\n"
                << "if SPECTRUM SPLIT CORE EIGENSTATES is set to a non-zero value."
                << std::endl;
          }

        if (d_dftParamsPtr->verbosity >= 1)
          pcout << "***********************Self-Consistent-Field Iteration: "
                << std::setw(2) << scfIter + 1
                << " complete**********************" << std::endl;

        local_timer.stop();
        if (d_dftParamsPtr->verbosity >= 1)
          pcout << "Wall time for the above scf iteration: "
                << local_timer.wall_time() << " seconds\n"
                << "Number of Chebyshev filtered subspace iterations: "
                << numberChebyshevSolvePasses << std::endl
                << std::endl;
        //
        scfIter++;

        if (d_dftParamsPtr->saveRhoData && scfIter % 10 == 0 &&
            d_dftParamsPtr->solverMode == "GS")
          saveTriaInfoAndRhoNodalData();
      }

    if (d_dftParamsPtr->saveRhoData &&
        !(d_dftParamsPtr->solverMode == "GS" && scfIter % 10 == 0))
      saveTriaInfoAndRhoNodalData();


    if (scfIter == d_dftParamsPtr->numSCFIterations)
      {
        if (Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
          std::cout
            << "DFT-FE Warning: SCF iterations did not converge to the specified tolerance after: "
            << scfIter << " iterations." << std::endl;
      }
    else
      pcout << "SCF iterations converged to the specified tolerance after: "
            << scfIter << " iterations." << std::endl;

    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);

    const unsigned int localVectorSize =
      d_eigenVectorsFlattenedSTL[0].size() / d_numEigenValues;

    if (numberBandGroups > 1 && !d_dftParamsPtr->useDevice)
      {
        MPI_Barrier(interBandGroupComm);
        const unsigned int blockSize =
          d_dftParamsPtr->mpiAllReduceMessageBlockSizeMB * 1e+6 /
          sizeof(dataTypes::number);
        for (unsigned int kPoint = 0;
             kPoint <
             (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size();
             ++kPoint)
          for (unsigned int i = 0; i < d_numEigenValues * localVectorSize;
               i += blockSize)
            {
              const unsigned int currentBlockSize =
                std::min(blockSize, d_numEigenValues * localVectorSize - i);
              MPI_Allreduce(MPI_IN_PLACE,
                            &d_eigenVectorsFlattenedSTL[kPoint][0] + i,
                            currentBlockSize,
                            dataTypes::mpi_type_id(
                              &d_eigenVectorsFlattenedSTL[kPoint][0]),
                            MPI_SUM,
                            interBandGroupComm);
            }
      }

    if ((!d_dftParamsPtr->computeEnergyEverySCF ||
         d_numEigenValuesRR != d_numEigenValues))
      {
        if (d_dftParamsPtr->verbosity >= 2)
          pcout
            << std::endl
            << "Poisson solve for total electrostatic potential (rhoOut+b): ";

        computing_timer.enter_subsection("phiTot solve");

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
        if (d_dftParamsPtr->useDevice and
            d_dftParamsPtr->floatingNuclearCharges and
            not d_dftParamsPtr->pinnedNodeForPBC)
#else
        if (false)
#endif
          {
#ifdef DFTFE_WITH_DEVICE
            d_phiTotalSolverProblemDevice.reinit(
              d_matrixFreeDataPRefined,
              d_phiTotRhoOut,
              *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
              d_phiTotDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
              d_phiTotAXQuadratureIdElectro,
              d_atomNodeIdToChargeMap,
              d_bQuadValuesAllAtoms,
              d_smearedChargeQuadratureIdElectro,
              *rhoOutValues,
              kohnShamDFTEigenOperatorDevice.getDeviceBlasHandle(),
              false,
              false,
              d_dftParamsPtr->smearedNuclearCharges,
              true,
              false,
              0,
              false,
              true);

            CGSolverDevice.solve(
              d_phiTotalSolverProblemDevice,
              d_dftParamsPtr->relLinearSolverTolerance,
              d_dftParamsPtr->maxLinearSolverIterations,
              kohnShamDFTEigenOperatorDevice.getDeviceBlasHandle(),
              d_dftParamsPtr->verbosity);
#endif
          }
        else
          {
            d_phiTotalSolverProblem.reinit(
              d_matrixFreeDataPRefined,
              d_phiTotRhoOut,
              *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
              d_phiTotDofHandlerIndexElectro,
              d_densityQuadratureIdElectro,
              d_phiTotAXQuadratureIdElectro,
              d_atomNodeIdToChargeMap,
              d_bQuadValuesAllAtoms,
              d_smearedChargeQuadratureIdElectro,
              *rhoOutValues,
              false,
              false,
              d_dftParamsPtr->smearedNuclearCharges,
              true,
              false,
              0,
              false,
              true);

            CGSolver.solve(d_phiTotalSolverProblem,
                           d_dftParamsPtr->relLinearSolverTolerance,
                           d_dftParamsPtr->maxLinearSolverIterations,
                           d_dftParamsPtr->verbosity);
          }

        computing_timer.leave_subsection("phiTot solve");
      }


    //
    // compute and print ground state energy or energy after max scf
    // iterations
    //
    d_dispersionCorr.computeDispresionCorrection(atomLocations,
                                                 d_domainBoundingVectors);
    const double totalEnergy =
      d_dftParamsPtr->spinPolarized == 0 ?
        energyCalc.computeEnergy(d_dofHandlerPRefined,
                                 dofHandler,
                                 quadrature,
                                 quadrature,
                                 d_matrixFreeDataPRefined.get_quadrature(
                                   d_smearedChargeQuadratureIdElectro),
                                 d_matrixFreeDataPRefined.get_quadrature(
                                   d_lpspQuadratureIdElectro),
                                 eigenValues,
                                 d_kPointWeights,
                                 fermiEnergy,
                                 excFunctionalPtr,
                                 d_dispersionCorr,
                                 d_phiInValues,
                                 d_phiTotRhoOut,
                                 *rhoInValues,
                                 *rhoOutValues,
                                 d_rhoOutValuesLpspQuad,
                                 *rhoOutValues,
                                 d_rhoOutValuesLpspQuad,
                                 *gradRhoInValues,
                                 *gradRhoOutValues,
                                 d_rhoCore,
                                 d_gradRhoCore,
                                 d_bQuadValuesAllAtoms,
                                 d_bCellNonTrivialAtomIds,
                                 d_localVselfs,
                                 d_pseudoVLoc,
                                 d_pseudoVLoc,
                                 d_atomNodeIdToChargeMap,
                                 atomLocations.size(),
                                 lowerBoundKindex,
                                 1,
                                 d_dftParamsPtr->verbosity >= 0 ? true : false,
                                 d_dftParamsPtr->smearedNuclearCharges) :
        energyCalc.computeEnergySpinPolarized(
          d_dofHandlerPRefined,
          dofHandler,
          quadrature,
          quadrature,
          d_matrixFreeDataPRefined.get_quadrature(
            d_smearedChargeQuadratureIdElectro),
          d_matrixFreeDataPRefined.get_quadrature(d_lpspQuadratureIdElectro),
          eigenValues,
          d_kPointWeights,
          fermiEnergy,
          fermiEnergyUp,
          fermiEnergyDown,
          excFunctionalPtr,
          d_dispersionCorr,
          d_phiInValues,
          d_phiTotRhoOut,
          *rhoInValues,
          *rhoOutValues,
          d_rhoOutValuesLpspQuad,
          *rhoOutValues,
          d_rhoOutValuesLpspQuad,
          *gradRhoInValues,
          *gradRhoOutValues,
          *rhoInValuesSpinPolarized,
          *rhoOutValuesSpinPolarized,
          *gradRhoInValuesSpinPolarized,
          *gradRhoOutValuesSpinPolarized,
          d_rhoCore,
          d_gradRhoCore,
          d_bQuadValuesAllAtoms,
          d_bCellNonTrivialAtomIds,
          d_localVselfs,
          d_pseudoVLoc,
          d_pseudoVLoc,
          d_atomNodeIdToChargeMap,
          atomLocations.size(),
          lowerBoundKindex,
          1,
          d_dftParamsPtr->verbosity >= 0 ? true : false,
          d_dftParamsPtr->smearedNuclearCharges);

    d_groundStateEnergy = totalEnergy;

    MPI_Barrier(interpoolcomm);

    d_entropicEnergy =
      energyCalc.computeEntropicEnergy(eigenValues,
                                       d_kPointWeights,
                                       fermiEnergy,
                                       fermiEnergyUp,
                                       fermiEnergyDown,
                                       d_dftParamsPtr->spinPolarized == 1,
                                       d_dftParamsPtr->constraintMagnetization,
                                       d_dftParamsPtr->TVal);

    if (d_dftParamsPtr->verbosity >= 1)
      pcout << "Total entropic energy: " << d_entropicEnergy << std::endl;


    d_freeEnergy = d_groundStateEnergy - d_entropicEnergy;

    if (d_dftParamsPtr->verbosity >= 1)
      pcout << "Total free energy: " << d_freeEnergy << std::endl;

    // This step is required for interpolating rho from current mesh to the
    // new mesh in case of atomic relaxation
    // computeNodalRhoFromQuadData();

    computing_timer.leave_subsection("scf solve");
    computingTimerStandard.leave_subsection("Total scf solve");


#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice &&
        (d_dftParamsPtr->writeWfcSolutionFields ||
         d_dftParamsPtr->writeLdosFile || d_dftParamsPtr->writePdosFile))
      for (unsigned int kPoint = 0;
           kPoint <
           (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size();
           ++kPoint)
        {
          d_eigenVectorsFlattenedDevice.copyTo<dftfe::utils::MemorySpace::HOST>(
            &d_eigenVectorsFlattenedSTL[kPoint][0],
            d_eigenVectorsFlattenedSTL[kPoint].size(),
            (kPoint * d_eigenVectorsFlattenedSTL[0].size()),
            0);
        }
#endif


    if (d_dftParamsPtr->isIonForce)
      {
        if (d_dftParamsPtr->selfConsistentSolverTolerance > 1e-4 &&
            d_dftParamsPtr->verbosity >= 1)
          pcout
            << "DFT-FE Warning: Ion force accuracy may be affected for the given scf iteration solve tolerance: "
            << d_dftParamsPtr->selfConsistentSolverTolerance
            << ", recommended to use TOLERANCE below 1e-4." << std::endl;

        if (computeForces)
          {
            computing_timer.enter_subsection("Ion force computation");
            computingTimerStandard.enter_subsection("Ion force computation");
            forcePtr->computeAtomsForces(matrix_free_data,
#ifdef DFTFE_WITH_DEVICE
                                         kohnShamDFTEigenOperatorDevice,
#endif
                                         kohnShamDFTEigenOperator,
                                         d_dispersionCorr,
                                         d_eigenDofHandlerIndex,
                                         d_smearedChargeQuadratureIdElectro,
                                         d_lpspQuadratureIdElectro,
                                         d_matrixFreeDataPRefined,
                                         d_phiTotDofHandlerIndexElectro,
                                         d_phiTotRhoOut,
                                         *rhoOutValues,
                                         *gradRhoOutValues,
                                         d_gradRhoOutValuesLpspQuad,
                                         *rhoOutValues,
                                         d_rhoOutValuesLpspQuad,
                                         *gradRhoOutValues,
                                         d_gradRhoOutValuesLpspQuad,
                                         d_rhoCore,
                                         d_gradRhoCore,
                                         d_hessianRhoCore,
                                         d_gradRhoCoreAtoms,
                                         d_hessianRhoCoreAtoms,
                                         d_pseudoVLoc,
                                         d_pseudoVLocAtoms,
                                         d_constraintsPRefined,
                                         d_vselfBinsManager,
                                         *rhoOutValues,
                                         *gradRhoOutValues,
                                         d_phiTotRhoIn);
            if (d_dftParamsPtr->verbosity >= 0)
              forcePtr->printAtomsForces();
            computingTimerStandard.leave_subsection("Ion force computation");
            computing_timer.leave_subsection("Ion force computation");
          }
      }

    if (d_dftParamsPtr->isCellStress)
      {
        if (d_dftParamsPtr->selfConsistentSolverTolerance > 1e-4 &&
            d_dftParamsPtr->verbosity >= 1)
          pcout
            << "DFT-FE Warning: Cell stress accuracy may be affected for the given scf iteration solve tolerance: "
            << d_dftParamsPtr->selfConsistentSolverTolerance
            << ", recommended to use TOLERANCE below 1e-4." << std::endl;

        if (computestress)
          {
            computing_timer.enter_subsection("Cell stress computation");
            computingTimerStandard.enter_subsection("Cell stress computation");
            computeStress();
            computingTimerStandard.leave_subsection("Cell stress computation");
            computing_timer.leave_subsection("Cell stress computation");
          }
      }

    if (d_dftParamsPtr->electrostaticsHRefinement)
      computeElectrostaticEnergyHRefined(
#ifdef DFTFE_WITH_DEVICE
        kohnShamDFTEigenOperatorDevice
#endif
      );

#ifdef USE_COMPLEX
    if (!(d_dftParamsPtr->kPointDataFile == ""))
      {
        readkPointData();
        initnscf(kohnShamDFTEigenOperator, d_phiTotalSolverProblem, CGSolver);
        nscf(kohnShamDFTEigenOperator, d_subspaceIterationSolver);
        writeBands();
      }
#endif
    return std::make_tuple(scfConverged, norm);
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::computeStress()
  {
    kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>
      &kohnShamDFTEigenOperator = *d_kohnShamDFTOperatorPtr;
#ifdef DFTFE_WITH_DEVICE
    kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
      &kohnShamDFTEigenOperatorDevice = *d_kohnShamDFTOperatorDevicePtr;
#endif

    if (d_dftParamsPtr->isPseudopotential ||
        d_dftParamsPtr->smearedNuclearCharges)
      {
        computeVselfFieldGateauxDerFD(
#ifdef DFTFE_WITH_DEVICE
          kohnShamDFTEigenOperatorDevice
#endif
        );
      }

    forcePtr->computeStress(matrix_free_data,
#ifdef DFTFE_WITH_DEVICE
                            kohnShamDFTEigenOperatorDevice,
#endif
                            kohnShamDFTEigenOperator,
                            d_dispersionCorr,
                            d_eigenDofHandlerIndex,
                            d_smearedChargeQuadratureIdElectro,
                            d_lpspQuadratureIdElectro,
                            d_matrixFreeDataPRefined,
                            d_phiTotDofHandlerIndexElectro,
                            d_phiTotRhoOut,
                            *rhoOutValues,
                            *gradRhoOutValues,
                            d_gradRhoOutValuesLpspQuad,
                            *rhoOutValues,
                            d_rhoOutValuesLpspQuad,
                            *gradRhoOutValues,
                            d_gradRhoOutValuesLpspQuad,
                            d_pseudoVLoc,
                            d_pseudoVLocAtoms,
                            d_rhoCore,
                            d_gradRhoCore,
                            d_hessianRhoCore,
                            d_gradRhoCoreAtoms,
                            d_hessianRhoCoreAtoms,
                            d_constraintsPRefined,
                            d_vselfBinsManager);
    if (d_dftParamsPtr->verbosity >= 0)
      forcePtr->printStress();
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::computeVselfFieldGateauxDerFD(
#ifdef DFTFE_WITH_DEVICE
    kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>
      &kohnShamDFTEigenOperatorDevice
#endif
  )
  {
    d_vselfFieldGateauxDerStrainFDBins.clear();
    d_vselfFieldGateauxDerStrainFDBins.resize(
      (d_vselfBinsManager.getVselfFieldBins()).size() * 6);

    Tensor<2, 3, double> identityTensor;
    Tensor<2, 3, double> deformationGradientPerturb1;
    Tensor<2, 3, double> deformationGradientPerturb2;

    // initialize to indentity tensors
    for (unsigned int idim = 0; idim < 3; idim++)
      for (unsigned int jdim = 0; jdim < 3; jdim++)
        {
          if (idim == jdim)
            {
              identityTensor[idim][jdim]              = 1.0;
              deformationGradientPerturb1[idim][jdim] = 1.0;
              deformationGradientPerturb2[idim][jdim] = 1.0;
            }
          else
            {
              identityTensor[idim][jdim]              = 0.0;
              deformationGradientPerturb1[idim][jdim] = 0.0;
              deformationGradientPerturb2[idim][jdim] = 0.0;
            }
        }

    const double fdparam          = 1e-5;
    unsigned int flattenedIdCount = 0;
    for (unsigned int idim = 0; idim < 3; ++idim)
      for (unsigned int jdim = 0; jdim <= idim; jdim++)
        {
          deformationGradientPerturb1 = identityTensor;
          if (idim == jdim)
            {
              deformationGradientPerturb1[idim][jdim] = 1.0 + fdparam;
            }
          else
            {
              deformationGradientPerturb1[idim][jdim] = fdparam;
              deformationGradientPerturb1[jdim][idim] = fdparam;
            }

          deformDomain(deformationGradientPerturb1 *
                         invert(deformationGradientPerturb2),
                       true,
                       false,
                       d_dftParamsPtr->verbosity >= 4 ? true : false);

#ifdef DFTFE_WITH_DEVICE
          if (d_dftParamsPtr->useDevice)
            kohnShamDFTEigenOperatorDevice
              .preComputeShapeFunctionGradientIntegrals(d_lpspQuadratureId,
                                                        true);
#endif

          computing_timer.enter_subsection(
            "Nuclear self-potential perturbation solve");

          d_vselfBinsManager.solveVselfInBinsPerturbedDomain(
            d_matrixFreeDataPRefined,
            d_baseDofHandlerIndexElectro,
            d_phiTotAXQuadratureIdElectro,
            d_binsStartDofHandlerIndexElectro,
#ifdef DFTFE_WITH_DEVICE
            kohnShamDFTEigenOperatorDevice,
#endif
            d_constraintsPRefined,
            d_imagePositionsTrunc,
            d_imageIdsTrunc,
            d_imageChargesTrunc,
            d_smearedChargeWidths,
            d_smearedChargeQuadratureIdElectro,
            d_dftParamsPtr->smearedNuclearCharges);

          computing_timer.leave_subsection(
            "Nuclear self-potential perturbation solve");

          for (unsigned int ibin = 0;
               ibin < d_vselfBinsManager.getVselfFieldBins().size();
               ibin++)
            d_vselfFieldGateauxDerStrainFDBins[6 * ibin + flattenedIdCount] =
              (d_vselfBinsManager.getPerturbedVselfFieldBins())[ibin];

          deformationGradientPerturb2 = identityTensor;
          if (idim == jdim)
            {
              deformationGradientPerturb2[idim][jdim] = 1.0 - fdparam;
            }
          else
            {
              deformationGradientPerturb2[idim][jdim] = -fdparam;
              deformationGradientPerturb2[jdim][idim] = -fdparam;
            }

          deformDomain(deformationGradientPerturb2 *
                         invert(deformationGradientPerturb1),
                       true,
                       false,
                       d_dftParamsPtr->verbosity >= 4 ? true : false);

#ifdef DFTFE_WITH_DEVICE
          if (d_dftParamsPtr->useDevice)
            kohnShamDFTEigenOperatorDevice
              .preComputeShapeFunctionGradientIntegrals(d_lpspQuadratureId,
                                                        true);
#endif

          computing_timer.enter_subsection(
            "Nuclear self-potential perturbation solve");

          d_vselfBinsManager.solveVselfInBinsPerturbedDomain(
            d_matrixFreeDataPRefined,
            d_baseDofHandlerIndexElectro,
            d_phiTotAXQuadratureIdElectro,
            d_binsStartDofHandlerIndexElectro,
#ifdef DFTFE_WITH_DEVICE
            kohnShamDFTEigenOperatorDevice,
#endif
            d_constraintsPRefined,
            d_imagePositionsTrunc,
            d_imageIdsTrunc,
            d_imageChargesTrunc,
            d_smearedChargeWidths,
            d_smearedChargeQuadratureIdElectro,
            d_dftParamsPtr->smearedNuclearCharges);

          computing_timer.leave_subsection(
            "Nuclear self-potential perturbation solve");

          for (unsigned int ibin = 0;
               ibin < d_vselfBinsManager.getVselfFieldBins().size();
               ibin++)
            d_vselfFieldGateauxDerStrainFDBins[6 * ibin + flattenedIdCount] -=
              (d_vselfBinsManager.getPerturbedVselfFieldBins())[ibin];

          const double fac =
            (idim == jdim) ? (1.0 / 2.0 / fdparam) : (1.0 / 4.0 / fdparam);
          for (unsigned int ibin = 0;
               ibin < d_vselfBinsManager.getVselfFieldBins().size();
               ibin++)
            d_vselfFieldGateauxDerStrainFDBins[6 * ibin + flattenedIdCount] *=
              fac;

          flattenedIdCount++;
        }

    // reset
    deformDomain(invert(deformationGradientPerturb2),
                 true,
                 false,
                 d_dftParamsPtr->verbosity >= 4 ? true : false);

#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice)
      kohnShamDFTEigenOperatorDevice.preComputeShapeFunctionGradientIntegrals(
        d_lpspQuadratureId, true);
#endif
  }

  // Output wfc
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::outputWfc()
  {
    //
    // identify the index which is close to Fermi Energy
    //
    int indexFermiEnergy = -1.0;
    for (int spinType = 0; spinType < 1 + d_dftParamsPtr->spinPolarized;
         ++spinType)
      {
        for (int i = 0; i < d_numEigenValues; ++i)
          {
            if (eigenValues[0][spinType * d_numEigenValues + i] >= fermiEnergy)
              {
                if (i > indexFermiEnergy)
                  {
                    indexFermiEnergy = i;
                    break;
                  }
              }
          }
      }

    //
    // create a range of wavefunctions to output the wavefunction files
    //
    int startingRange = 0;
    int endingRange   = d_numEigenValues;

    /*
    int startingRange = indexFermiEnergy - 4;
    int endingRange   = indexFermiEnergy + 4;

    int startingRangeSpin = startingRange;

    for (int spinType = 0; spinType < 1 + d_dftParamsPtr->spinPolarized;
         ++spinType)
      {
        for (int i = indexFermiEnergy - 5; i > 0; --i)
          {
            if (std::abs(eigenValues[0][spinType * d_numEigenValues +
                                        (indexFermiEnergy - 4)] -
                         eigenValues[0][spinType * d_numEigenValues + i]) <=
                5e-04)
              {
                if (spinType == 0)
                  startingRange -= 1;
                else
                  startingRangeSpin -= 1;
              }
          }
      }


    if (startingRangeSpin < startingRange)
      startingRange = startingRangeSpin;
    */
    int numStatesOutput = (endingRange - startingRange) + 1;


    DataOut<3> data_outEigen;
    data_outEigen.attach_dof_handler(dofHandlerEigen);

    std::vector<distributedCPUVec<double>> tempVec(1);
    tempVec[0].reinit(d_tempEigenVec);

    std::vector<distributedCPUVec<double>> visualizeWaveFunctions(
      d_kPointWeights.size() * (1 + d_dftParamsPtr->spinPolarized) *
      numStatesOutput);

    unsigned int count = 0;
    for (unsigned int s = 0; s < 1 + d_dftParamsPtr->spinPolarized; ++s)
      for (unsigned int k = 0; k < d_kPointWeights.size(); ++k)
        for (unsigned int i = startingRange; i < endingRange; ++i)
          {
#ifdef USE_COMPLEX
            vectorTools::copyFlattenedSTLVecToSingleCompVec(
              d_eigenVectorsFlattenedSTL[k *
                                           (1 + d_dftParamsPtr->spinPolarized) +
                                         s],
              d_numEigenValues,
              std::make_pair(i, i + 1),
              localProc_dof_indicesReal,
              localProc_dof_indicesImag,
              tempVec);
#else
            vectorTools::copyFlattenedSTLVecToSingleCompVec(
              d_eigenVectorsFlattenedSTL[k *
                                           (1 + d_dftParamsPtr->spinPolarized) +
                                         s],
              d_numEigenValues,
              std::make_pair(i, i + 1),
              tempVec);
#endif

            constraintsNoneEigenDataInfo.distribute(tempVec[0]);
            visualizeWaveFunctions[count] = tempVec[0];

            if (d_dftParamsPtr->spinPolarized == 1)
              data_outEigen.add_data_vector(visualizeWaveFunctions[count],
                                            "wfc_spin" + std::to_string(s) +
                                              "_kpoint" + std::to_string(k) +
                                              "_" + std::to_string(i));
            else
              data_outEigen.add_data_vector(visualizeWaveFunctions[count],
                                            "wfc_kpoint" + std::to_string(k) +
                                              "_" + std::to_string(i));

            count += 1;
          }

    data_outEigen.build_patches(FEOrder);

    std::string tempFolder = "waveFunctionOutputFolder";
    mkdir(tempFolder.c_str(), ACCESSPERMS);

    dftUtils::writeDataVTUParallelLowestPoolId(dofHandlerEigen,
                                               data_outEigen,
                                               d_mpiCommParent,
                                               mpi_communicator,
                                               interpoolcomm,
                                               interBandGroupComm,
                                               tempFolder,
                                               "wfcOutput");
    //"wfcOutput_"+std::to_string(k)+"_"+std::to_string(i));
  }


  // Output density
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::outputDensity()
  {
    //
    // compute nodal electron-density from quad data
    //
    distributedCPUVec<double> rhoNodalField;
    d_matrixFreeDataPRefined.initialize_dof_vector(
      rhoNodalField, d_densityDofHandlerIndexElectro);
    rhoNodalField = 0;
    std::function<
      double(const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
             const unsigned int                                          q)>
      funcRho =
        [&](const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
            const unsigned int                                          q) {
          return (*rhoOutValues).find(cell->id())->second[q];
        };
    dealii::VectorTools::project<3, distributedCPUVec<double>>(
      dealii::MappingQ1<3, 3>(),
      d_dofHandlerRhoNodal,
      d_constraintsRhoNodal,
      d_matrixFreeDataPRefined.get_quadrature(d_densityQuadratureIdElectro),
      funcRho,
      rhoNodalField);
    rhoNodalField.update_ghost_values();

    distributedCPUVec<double> rhoNodalFieldSpin0;
    distributedCPUVec<double> rhoNodalFieldSpin1;
    if (d_dftParamsPtr->spinPolarized == 1)
      {
        rhoNodalFieldSpin0.reinit(rhoNodalField);
        rhoNodalFieldSpin0 = 0;
        std::function<double(
          const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
          const unsigned int                                          q)>
          funcRhoSpin0 = [&](const typename dealii::DoFHandler<
                               3>::active_cell_iterator &cell,
                             const unsigned int          q) {
            return (*rhoOutValuesSpinPolarized).find(cell->id())->second[2 * q];
          };
        dealii::VectorTools::project<3, distributedCPUVec<double>>(
          dealii::MappingQ1<3, 3>(),
          d_dofHandlerRhoNodal,
          d_constraintsRhoNodal,
          d_matrixFreeDataPRefined.get_quadrature(d_densityQuadratureIdElectro),
          funcRhoSpin0,
          rhoNodalFieldSpin0);
        rhoNodalFieldSpin0.update_ghost_values();


        rhoNodalFieldSpin1.reinit(rhoNodalField);
        rhoNodalFieldSpin1 = 0;
        std::function<double(
          const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
          const unsigned int                                          q)>
          funcRhoSpin1 =
            [&](
              const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
              const unsigned int                                          q) {
              return (*rhoOutValuesSpinPolarized)
                .find(cell->id())
                ->second[2 * q + 1];
            };
        dealii::VectorTools::project<3, distributedCPUVec<double>>(
          dealii::MappingQ1<3, 3>(),
          d_dofHandlerRhoNodal,
          d_constraintsRhoNodal,
          d_matrixFreeDataPRefined.get_quadrature(d_densityQuadratureIdElectro),
          funcRhoSpin1,
          rhoNodalFieldSpin1);
        rhoNodalFieldSpin1.update_ghost_values();
      }

    //
    // only generate output for electron-density
    //
    DataOut<3> dataOutRho;
    dataOutRho.attach_dof_handler(d_dofHandlerRhoNodal);
    dataOutRho.add_data_vector(rhoNodalField, std::string("density"));
    if (d_dftParamsPtr->spinPolarized == 1)
      {
        dataOutRho.add_data_vector(rhoNodalFieldSpin0,
                                   std::string("density_0"));
        dataOutRho.add_data_vector(rhoNodalFieldSpin1,
                                   std::string("density_1"));
      }
    dataOutRho.build_patches(FEOrder);

    std::string tempFolder = "densityOutputFolder";
    mkdir(tempFolder.c_str(), ACCESSPERMS);

    dftUtils::writeDataVTUParallelLowestPoolId(d_dofHandlerRhoNodal,
                                               dataOutRho,
                                               d_mpiCommParent,
                                               mpi_communicator,
                                               interpoolcomm,
                                               interBandGroupComm,
                                               tempFolder,
                                               "densityOutput");
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::writeBands()
  {
    int numkPoints =
      (1 + d_dftParamsPtr->spinPolarized) * d_kPointWeights.size();
    std::vector<double> eigenValuesFlattened;
    //
    for (unsigned int kPoint = 0; kPoint < numkPoints; ++kPoint)
      for (unsigned int iWave = 0; iWave < d_numEigenValues; ++iWave)
        eigenValuesFlattened.push_back(eigenValues[kPoint][iWave]);
    //
    //
    //
    int totkPoints = Utilities::MPI::sum(numkPoints, interpoolcomm);
    std::vector<int> numkPointsArray(d_dftParamsPtr->npool),
      mpi_offsets(d_dftParamsPtr->npool, 0);
    std::vector<double> eigenValuesFlattenedGlobal(totkPoints *
                                                     d_numEigenValues,
                                                   0.0);
    //
    MPI_Gather(&numkPoints,
               1,
               MPI_INT,
               &(numkPointsArray[0]),
               1,
               MPI_INT,
               0,
               interpoolcomm);
    //
    numkPointsArray[0] = d_numEigenValues * numkPointsArray[0];
    for (unsigned int ipool = 1; ipool < d_dftParamsPtr->npool; ++ipool)
      {
        numkPointsArray[ipool] = d_numEigenValues * numkPointsArray[ipool];
        mpi_offsets[ipool] =
          mpi_offsets[ipool - 1] + numkPointsArray[ipool - 1];
      }
    //
    MPI_Gatherv(&(eigenValuesFlattened[0]),
                numkPoints * d_numEigenValues,
                MPI_DOUBLE,
                &(eigenValuesFlattenedGlobal[0]),
                &(numkPointsArray[0]),
                &(mpi_offsets[0]),
                MPI_DOUBLE,
                0,
                interpoolcomm);
    //
    if (Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0)
      {
        FILE *pFile;
        pFile = fopen("bands.out", "w");
        fprintf(pFile, "%d %d\n", totkPoints, d_numEigenValues);
        for (unsigned int kPoint = 0;
             kPoint < totkPoints / (1 + d_dftParamsPtr->spinPolarized);
             ++kPoint)
          {
            for (unsigned int iWave = 0; iWave < d_numEigenValues; ++iWave)
              {
                if (d_dftParamsPtr->spinPolarized)
                  fprintf(
                    pFile,
                    "%d  %d   %g   %g\n",
                    kPoint,
                    iWave,
                    eigenValuesFlattenedGlobal[2 * kPoint * d_numEigenValues +
                                               iWave],
                    eigenValuesFlattenedGlobal[(2 * kPoint + 1) *
                                                 d_numEigenValues +
                                               iWave]);
                else
                  fprintf(pFile,
                          "%d  %d %g\n",
                          kPoint,
                          iWave,
                          eigenValuesFlattenedGlobal[kPoint * d_numEigenValues +
                                                     iWave]);
              }
          }
      }
    MPI_Barrier(d_mpiCommParent);
    //
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  std::vector<std::vector<double>>
  dftClass<FEOrder, FEOrderElectro>::getAtomLocationsCart() const
  {
    return atomLocations;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  std::vector<std::vector<double>>
  dftClass<FEOrder, FEOrderElectro>::getAtomLocationsFrac() const
  {
    return atomLocationsFractional;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  std::vector<std::vector<double>>
  dftClass<FEOrder, FEOrderElectro>::getCell() const
  {
    return d_domainBoundingVectors;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::getCellVolume() const
  {
    return d_domainVolume;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  std::set<unsigned int>
  dftClass<FEOrder, FEOrderElectro>::getAtomTypes() const
  {
    return atomTypes;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  std::vector<double>
  dftClass<FEOrder, FEOrderElectro>::getForceonAtoms() const
  {
    return (forcePtr->getAtomsForces());
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  Tensor<2, 3, double>
  dftClass<FEOrder, FEOrderElectro>::getCellStress() const
  {
    return (forcePtr->getStress());
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  dftParameters &
  dftClass<FEOrder, FEOrderElectro>::getParametersObject() const
  {
    return (*d_dftParamsPtr);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::getInternalEnergy() const
  {
    return d_groundStateEnergy;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::getEntropicEnergy() const
  {
    return d_entropicEnergy;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::getFreeEnergy() const
  {
    return d_freeEnergy;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  distributedCPUVec<double>
  dftClass<FEOrder, FEOrderElectro>::getRhoNodalOut() const
  {
    return d_rhoOutNodalValues;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  distributedCPUVec<double>
  dftClass<FEOrder, FEOrderElectro>::getRhoNodalSplitOut() const
  {
    return d_rhoOutNodalValuesSplit;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::getTotalChargeforRhoSplit()
  {
    double temp =
      (-totalCharge(d_matrixFreeDataPRefined, d_rhoOutNodalValuesSplit) /
       d_domainVolume);
    return (temp);
  }



  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::resetRhoNodalIn(
    distributedCPUVec<double> &OutDensity)
  {
    d_rhoOutNodalValues = OutDensity;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::resetRhoNodalSplitIn(
    distributedCPUVec<double> &OutDensity)
  {
    d_rhoOutNodalValuesSplit = OutDensity;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::writeMesh()
  {
    //
    // compute nodal electron-density from quad data
    //
    distributedCPUVec<double> rhoNodalField;
    d_matrixFreeDataPRefined.initialize_dof_vector(
      rhoNodalField, d_densityDofHandlerIndexElectro);
    rhoNodalField = 0;
    std::function<
      double(const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
             const unsigned int                                          q)>
      funcRho =
        [&](const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
            const unsigned int                                          q) {
          return (*rhoInValues).find(cell->id())->second[q];
        };
    dealii::VectorTools::project<3, distributedCPUVec<double>>(
      dealii::MappingQ1<3, 3>(),
      d_dofHandlerRhoNodal,
      d_constraintsRhoNodal,
      d_matrixFreeDataPRefined.get_quadrature(d_densityQuadratureIdElectro),
      funcRho,
      rhoNodalField);
    rhoNodalField.update_ghost_values();



    //
    // only generate output for electron-density
    //
    DataOut<3> dataOutRho;
    dataOutRho.attach_dof_handler(d_dofHandlerRhoNodal);
    dataOutRho.add_data_vector(rhoNodalField, std::string("density"));

    dataOutRho.build_patches(FEOrder);

    std::string tempFolder = "meshOutputFolder";
    mkdir(tempFolder.c_str(), ACCESSPERMS);

    dftUtils::writeDataVTUParallelLowestPoolId(d_dofHandlerRhoNodal,
                                               dataOutRho,
                                               d_mpiCommParent,
                                               mpi_communicator,
                                               interpoolcomm,
                                               interBandGroupComm,
                                               tempFolder,
                                               "intialDensityOutput");



    if (d_dftParamsPtr->verbosity >= 1)
      pcout
        << std::endl
        << "------------------DFT-FE mesh file creation completed---------------------------"
        << std::endl;
  }


#include "dft.inst.cc"
} // namespace dftfe
