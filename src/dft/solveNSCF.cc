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
// @author Phani Motamarri
//


//
// dft NSCF solve (non-selfconsistent solution of DFT eigenvalue problem to
// compute
// eigenvalues, eigenfunctions and ground-state energy
// using the self-consistent Hamiltonian)
//
#include <dft.h>
#include <energyCalculator.h>
namespace dftfe
{
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::solveNoSCF()
  {
    KohnShamHamiltonianOperator<memorySpace> &kohnShamDFTEigenOperator =
      *d_kohnShamDFTOperatorPtr;

    const dealii::Quadrature<3> &quadrature =
      matrix_free_data.get_quadrature(d_densityQuadratureId);

    // computingTimerStandard.enter_subsection("Total scf solve");
    energyCalculator<memorySpace> energyCalc(d_mpiCommParent,
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
                                        linearSolverCGDevice::CG,
                                        d_BLASWrapperPtr);
#endif



    // FIXME: Check if this call can be removed
    d_phiTotalSolverProblem.clear();

    //
    // solve vself in bins
    //
    computing_timer.enter_subsection("Nuclear self-potential solve");
    computingTimerStandard.enter_subsection("Nuclear self-potential solve");
#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice and d_dftParamsPtr->vselfGPU)
      d_vselfBinsManager.solveVselfInBinsDevice(
        d_basisOperationsPtrElectroHost,
        d_baseDofHandlerIndexElectro,
        d_phiTotAXQuadratureIdElectro,
        d_binsStartDofHandlerIndexElectro,
        FEOrder == FEOrderElectro ?
          d_basisOperationsPtrDevice->cellStiffnessMatrixBasisData() :
          d_basisOperationsPtrElectroDevice->cellStiffnessMatrixBasisData(),
        d_BLASWrapperPtr,
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
        d_basisOperationsPtrElectroHost,
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
    d_vselfBinsManager.solveVselfInBins(d_basisOperationsPtrElectroHost,
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
        kohnShamDFTEigenOperator.computeVEffExternalPotCorr(d_pseudoVLoc);
        computingTimerStandard.leave_subsection("Init local PSP");
      }


    computingTimerStandard.enter_subsection("Total nscf solve");

    //
    // solve
    //
    computing_timer.enter_subsection("nscf solve");

    double chebyTol;
    chebyTol = d_dftParamsPtr->chebyshevTolerance == 0.0 ?
                 1e-08 :
                 d_dftParamsPtr->chebyshevTolerance;



    if (d_dftParamsPtr->verbosity >= 0)
      pcout << "Starting NSCF iteration...." << std::endl;

    dealii::Timer local_timer(d_mpiCommParent, true);


    //
    // phiTot with rhoIn
    //
    if (d_dftParamsPtr->verbosity >= 2)
      pcout << std::endl
            << "Poisson solve for total electrostatic potential (rhoIn+b): ";

    if (d_dftParamsPtr->multipoleBoundaryConditions)
      {
        computing_timer.enter_subsection("Update inhomogenous BC");
        computeMultipoleMoments(d_basisOperationsPtrElectroHost,
                                d_densityQuadratureIdElectro,
                                d_densityInQuadValues[0],
                                &d_bQuadValuesAllAtoms);
        updatePRefinedConstraints();
        computing_timer.leave_subsection("Update inhomogenous BC");
      }

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      densityInQuadValuesCopy = d_densityInQuadValues[0];
    if (std::abs(d_dftParamsPtr->netCharge) > 1e-12 and
        (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
         d_dftParamsPtr->periodicZ))
      {
        double *tempvec = densityInQuadValuesCopy.data();
        for (unsigned int iquad = 0; iquad < densityInQuadValuesCopy.size();
             iquad++)
          tempvec[iquad] += -d_dftParamsPtr->netCharge / d_domainVolume;
      }


    if (d_dftParamsPtr->useDevice and d_dftParamsPtr->poissonGPU and
        d_dftParamsPtr->floatingNuclearCharges and
        not d_dftParamsPtr->pinnedNodeForPBC)
      {
#ifdef DFTFE_WITH_DEVICE
        d_phiTotalSolverProblemDevice.reinit(
          d_basisOperationsPtrElectroHost,
          d_phiTotRhoIn,
          *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
          d_phiTotDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          d_phiTotAXQuadratureIdElectro,
          d_atomNodeIdToChargeMap,
          d_bQuadValuesAllAtoms,
          d_smearedChargeQuadratureIdElectro,
          densityInQuadValuesCopy,
          d_BLASWrapperPtr,
          true,
          d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
            d_dftParamsPtr->periodicZ && !d_dftParamsPtr->pinnedNodeForPBC,
          d_dftParamsPtr->smearedNuclearCharges,
          true,
          false,
          0,
          true,
          false,
          d_dftParamsPtr->multipoleBoundaryConditions);

#endif
      }
    else
      {
        d_phiTotalSolverProblem.reinit(
          d_basisOperationsPtrElectroHost,
          d_phiTotRhoIn,
          *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
          d_phiTotDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          d_phiTotAXQuadratureIdElectro,
          d_atomNodeIdToChargeMap,
          d_bQuadValuesAllAtoms,
          d_smearedChargeQuadratureIdElectro,
          densityInQuadValuesCopy,
          true,
          d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
            d_dftParamsPtr->periodicZ && !d_dftParamsPtr->pinnedNodeForPBC,
          d_dftParamsPtr->smearedNuclearCharges,
          true,
          false,
          0,
          true,
          false,
          d_dftParamsPtr->multipoleBoundaryConditions);
      }

    computing_timer.enter_subsection("phiTot solve");

    if (d_dftParamsPtr->useDevice and d_dftParamsPtr->poissonGPU and
        d_dftParamsPtr->floatingNuclearCharges and
        not d_dftParamsPtr->pinnedNodeForPBC)
      {
#ifdef DFTFE_WITH_DEVICE
        CGSolverDevice.solve(d_phiTotalSolverProblemDevice,
                             d_dftParamsPtr->absLinearSolverTolerance,
                             d_dftParamsPtr->maxLinearSolverIterations,
                             d_dftParamsPtr->verbosity);
#endif
      }
    else
      {
        CGSolver.solve(d_phiTotalSolverProblem,
                       d_dftParamsPtr->absLinearSolverTolerance,
                       d_dftParamsPtr->maxLinearSolverIterations,
                       d_dftParamsPtr->verbosity);
      }

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> dummy;
    interpolateElectroNodalDataToQuadratureDataGeneral(
      d_basisOperationsPtrElectroHost,
      d_phiTotDofHandlerIndexElectro,
      d_densityQuadratureIdElectro,
      d_phiTotRhoIn,
      d_phiInQuadValues,
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
    // if writeBandsFile == true, get the fermi energy from the fermiEnergy.out
    if (d_dftParamsPtr->writeBandsFile)
      {
        std::ifstream file("fermiEnergy.out");
        std::string   line;

        if (file.is_open())
          {
            if (d_dftParamsPtr->constraintMagnetization)
              {
                std::vector<double> temp;
                while (getline(file, line))
                  {
                    if (!line.empty())
                      {
                        std::istringstream iss(line);
                        double             temp1;
                        while (iss >> temp1)
                          {
                            temp.push_back(temp1);
                          }
                      }
                  }
                fermiEnergy     = temp[0];
                fermiEnergyUp   = temp[1];
                fermiEnergyDown = temp[2];
              }
            else
              {
                getline(file, line);
                std::istringstream iss(line);
                iss >> fermiEnergy;
              }
          }
        else
          {
            pcout
              << "Unable to open file fermiEnergy.out. Check if it is present.";
          }
      }

    if (d_dftParamsPtr->spinPolarized == 1)
      {
        std::vector<std::vector<std::vector<double>>> eigenValuesSpins(
          2,
          std::vector<std::vector<double>>(
            d_kPointWeights.size(), std::vector<double>((d_numEigenValues))));

        std::vector<std::vector<std::vector<double>>>
          residualNormWaveFunctionsAllkPointsSpins(
            2,
            std::vector<std::vector<double>>(
              d_kPointWeights.size(), std::vector<double>(d_numEigenValues)));

        updateAuxDensityXCMatrix(d_densityInQuadValues,
                                 d_gradDensityInQuadValues,
                                 d_rhoCore,
                                 d_gradRhoCore,
                                 getEigenVectors(),
                                 eigenValues,
                                 fermiEnergy,
                                 fermiEnergyUp,
                                 fermiEnergyDown,
                                 d_auxDensityMatrixXCInPtr);


        for (unsigned int s = 0; s < 2; ++s)
          {
            computing_timer.enter_subsection("VEff Computation");


            kohnShamDFTEigenOperator.computeVEff(d_auxDensityMatrixXCInPtr,
                                                 d_phiInQuadValues,
                                                 s);

            computing_timer.leave_subsection("VEff Computation");


            for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size();
                 ++kPoint)
              {
                kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint, s);


                computing_timer.enter_subsection(
                  "Hamiltonian Matrix Computation");
                kohnShamDFTEigenOperator.computeCellHamiltonianMatrix();
                computing_timer.leave_subsection(
                  "Hamiltonian Matrix Computation");


                for (unsigned int j = 0; j < 1; ++j)
                  {
                    if (d_dftParamsPtr->verbosity >= 2)
                      {
                        pcout << "Beginning Chebyshev filter pass " << j + 1
                              << " for spin " << s + 1 << std::endl;
                      }

#ifdef DFTFE_WITH_DEVICE
                    if constexpr (dftfe::utils::MemorySpace::DEVICE ==
                                  memorySpace)
                      kohnShamEigenSpaceCompute(
                        s,
                        kPoint,
                        kohnShamDFTEigenOperator,
                        *d_elpaScala,
                        d_subspaceIterationSolverDevice,
                        residualNormWaveFunctionsAllkPointsSpins[s][kPoint],
                        true,
                        0,
                        false,
                        false,
                        true);
#endif
                    if constexpr (dftfe::utils::MemorySpace::HOST ==
                                  memorySpace)
                      kohnShamEigenSpaceCompute(
                        s,
                        kPoint,
                        kohnShamDFTEigenOperator,
                        *d_elpaScala,
                        d_subspaceIterationSolver,
                        residualNormWaveFunctionsAllkPointsSpins[s][kPoint],
                        true,
                        false,
                        false,
                        true);
                  }
              }
          }


        for (unsigned int s = 0; s < 2; ++s)
          for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size();
               ++kPoint)
            {
              for (unsigned int i = 0; i < d_numEigenValues; ++i)
                eigenValuesSpins[s][kPoint][i] =
                  eigenValues[kPoint][d_numEigenValues * s + i];
            }
        //
        // fermi energy
        //
        if (!(d_dftParamsPtr->writeBandsFile))
          {
            if (d_dftParamsPtr->constraintMagnetization)
              compute_fermienergy_constraintMagnetization(eigenValues);
            else
              compute_fermienergy(eigenValues, numElectrons);
          }

        unsigned int count = 1;


        // maximum of the residual norm of the state closest to and
        // below the Fermi level among all k points, and also the
        // maximum between the two spins
        double maxRes = 0.0;
        if (d_dftParamsPtr->highestStateOfInterestForChebFiltering == 0)
          {
            maxRes = std::max(computeMaximumHighestOccupiedStateResidualNorm(
                                residualNormWaveFunctionsAllkPointsSpins[0],
                                eigenValuesSpins[0],
                                fermiEnergy),
                              computeMaximumHighestOccupiedStateResidualNorm(
                                residualNormWaveFunctionsAllkPointsSpins[1],
                                eigenValuesSpins[1],
                                fermiEnergy));
          }
        else
          {
            maxRes = std::max(
              computeMaximumHighestOccupiedStateResidualNorm(
                residualNormWaveFunctionsAllkPointsSpins[0],
                eigenValuesSpins[0],
                d_dftParamsPtr->highestStateOfInterestForChebFiltering),
              computeMaximumHighestOccupiedStateResidualNorm(
                residualNormWaveFunctionsAllkPointsSpins[1],
                eigenValuesSpins[1],
                d_dftParamsPtr->highestStateOfInterestForChebFiltering));
          }


        if (d_dftParamsPtr->verbosity >= 2)
          {
            if (d_dftParamsPtr->highestStateOfInterestForChebFiltering == 0)
              {
                pcout
                  << "Maximum residual norm of the state closest to and below Fermi level: "
                  << maxRes << std::endl;
              }
            else
              {
                pcout
                  << "Maximum residual norm of the highest state of interest : "
                  << maxRes << std::endl;
              }
          }

        // if the residual norm is greater than
        // adaptiveChebysevFilterPassesTol (a heuristic value)
        // do more passes of chebysev filter till the check passes.
        // This improves the scf convergence performance.


        while (maxRes > chebyTol && count < 100)
          {
            for (unsigned int s = 0; s < 2; ++s)
              {
                if (d_dftParamsPtr->memOptMode)
                  {
                    computing_timer.enter_subsection("VEff Computation");
                    kohnShamDFTEigenOperator.computeVEff(
                      d_auxDensityMatrixXCInPtr, d_phiInQuadValues, s);

                    computing_timer.leave_subsection("VEff Computation");
                  }
                for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size();
                     ++kPoint)
                  {
                    if (d_dftParamsPtr->verbosity >= 2)
                      pcout << "Beginning Chebyshev filter pass " << 1 + count
                            << " for spin " << s + 1 << std::endl;

                    kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint, s);
                    if (d_dftParamsPtr->memOptMode)
                      {
                        computing_timer.enter_subsection(
                          "Hamiltonian Matrix Computation");
                        kohnShamDFTEigenOperator.computeCellHamiltonianMatrix();
                        computing_timer.leave_subsection(
                          "Hamiltonian Matrix Computation");
                      }

#ifdef DFTFE_WITH_DEVICE
                    if constexpr (dftfe::utils::MemorySpace::DEVICE ==
                                  memorySpace)
                      kohnShamEigenSpaceCompute(
                        s,
                        kPoint,
                        kohnShamDFTEigenOperator,
                        *d_elpaScala,
                        d_subspaceIterationSolverDevice,
                        residualNormWaveFunctionsAllkPointsSpins[s][kPoint],
                        true,
                        0,
                        false,
                        false,
                        true);
#endif
                    if constexpr (dftfe::utils::MemorySpace::HOST ==
                                  memorySpace)
                      kohnShamEigenSpaceCompute(
                        s,
                        kPoint,
                        kohnShamDFTEigenOperator,
                        *d_elpaScala,
                        d_subspaceIterationSolver,
                        residualNormWaveFunctionsAllkPointsSpins[s][kPoint],
                        true,
                        false,
                        false,
                        true);
                  }
              }

            for (unsigned int s = 0; s < 2; ++s)
              for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size();
                   ++kPoint)
                {
                  for (unsigned int i = 0; i < d_numEigenValues; ++i)
                    eigenValuesSpins[s][kPoint][i] =
                      eigenValues[kPoint][d_numEigenValues * s + i];
                }
            if (!(d_dftParamsPtr->writeBandsFile))
              {
                if (d_dftParamsPtr->constraintMagnetization)
                  compute_fermienergy_constraintMagnetization(eigenValues);
                else
                  compute_fermienergy(eigenValues, numElectrons);
              }

            if (d_dftParamsPtr->highestStateOfInterestForChebFiltering == 0)
              {
                maxRes =
                  std::max(computeMaximumHighestOccupiedStateResidualNorm(
                             residualNormWaveFunctionsAllkPointsSpins[0],
                             eigenValuesSpins[0],
                             fermiEnergy),
                           computeMaximumHighestOccupiedStateResidualNorm(
                             residualNormWaveFunctionsAllkPointsSpins[1],
                             eigenValuesSpins[1],
                             fermiEnergy));
              }
            else
              {
                maxRes = std::max(
                  computeMaximumHighestOccupiedStateResidualNorm(
                    residualNormWaveFunctionsAllkPointsSpins[0],
                    eigenValuesSpins[0],
                    d_dftParamsPtr->highestStateOfInterestForChebFiltering),
                  computeMaximumHighestOccupiedStateResidualNorm(
                    residualNormWaveFunctionsAllkPointsSpins[1],
                    eigenValuesSpins[1],
                    d_dftParamsPtr->highestStateOfInterestForChebFiltering));
              }

            if (d_dftParamsPtr->verbosity >= 2)
              {
                if (d_dftParamsPtr->highestStateOfInterestForChebFiltering == 0)
                  {
                    pcout
                      << "Maximum residual norm of the state closest to and below Fermi level: "
                      << maxRes << std::endl;
                  }
                else
                  {
                    pcout
                      << "Maximum residual norm of the highest state of interest : "
                      << maxRes << std::endl;
                  }
              }
            count++;
          }


        if (d_dftParamsPtr->verbosity >= 1)
          {
            pcout << "Fermi Energy computed: " << fermiEnergy << std::endl;
          }

        numberChebyshevSolvePasses = count;
      }
    else
      {
        std::vector<std::vector<double>> residualNormWaveFunctionsAllkPoints;
        residualNormWaveFunctionsAllkPoints.resize(d_kPointWeights.size());
        for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
          residualNormWaveFunctionsAllkPoints[kPoint].resize(d_numEigenValues);

        updateAuxDensityXCMatrix(d_densityInQuadValues,
                                 d_gradDensityInQuadValues,
                                 d_rhoCore,
                                 d_gradRhoCore,
                                 getEigenVectors(),
                                 eigenValues,
                                 fermiEnergy,
                                 fermiEnergyUp,
                                 fermiEnergyDown,
                                 d_auxDensityMatrixXCInPtr);

        computing_timer.enter_subsection("VEff Computation");
        kohnShamDFTEigenOperator.computeVEff(d_auxDensityMatrixXCInPtr,
                                             d_phiInQuadValues);
        computing_timer.leave_subsection("VEff Computation");

        for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
          {
            kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint, 0);


            computing_timer.enter_subsection("Hamiltonian Matrix Computation");
            kohnShamDFTEigenOperator.computeCellHamiltonianMatrix();
            computing_timer.leave_subsection("Hamiltonian Matrix Computation");


            for (unsigned int j = 0; j < 1; ++j)
              {
                if (d_dftParamsPtr->verbosity >= 2)
                  {
                    pcout << "Beginning Chebyshev filter pass " << j + 1
                          << std::endl;
                  }


#ifdef DFTFE_WITH_DEVICE
                if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
                  kohnShamEigenSpaceCompute(
                    0,
                    kPoint,
                    kohnShamDFTEigenOperator,
                    *d_elpaScala,
                    d_subspaceIterationSolverDevice,
                    residualNormWaveFunctionsAllkPoints[kPoint],
                    true,
                    0,
                    false,
                    false,
                    true);
#endif
                if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
                  kohnShamEigenSpaceCompute(
                    0,
                    kPoint,
                    kohnShamDFTEigenOperator,
                    *d_elpaScala,
                    d_subspaceIterationSolver,
                    residualNormWaveFunctionsAllkPoints[kPoint],
                    true,
                    false,
                    false,
                    true);
              }
          }


        //
        // fermi energy
        //
        if (!(d_dftParamsPtr->writeBandsFile))
          {
            if (d_dftParamsPtr->constraintMagnetization)
              compute_fermienergy_constraintMagnetization(eigenValues);
            else
              compute_fermienergy(eigenValues, numElectrons);
          }

        unsigned int count = 1;

        //
        // maximum of the residual norm of the state closest to and
        // below the Fermi level among all k points
        //
        double maxRes = 0.0;

        if (d_dftParamsPtr->highestStateOfInterestForChebFiltering == 0)
          {
            maxRes = computeMaximumHighestOccupiedStateResidualNorm(
              residualNormWaveFunctionsAllkPoints, eigenValues, fermiEnergy);
          }
        else
          {
            maxRes = computeMaximumHighestOccupiedStateResidualNorm(
              residualNormWaveFunctionsAllkPoints,
              eigenValues,
              d_dftParamsPtr->highestStateOfInterestForChebFiltering);
          }

        if (d_dftParamsPtr->verbosity >= 2)
          {
            if (d_dftParamsPtr->highestStateOfInterestForChebFiltering == 0)
              {
                pcout
                  << "Maximum residual norm of the state closest to and below Fermi level: "
                  << maxRes << std::endl;
              }
            else
              {
                pcout
                  << "Maximum residual norm of the highest state of interest : "
                  << maxRes << std::endl;
              }
          }

        // if the residual norm is greater than
        // adaptiveChebysevFilterPassesTol (a heuristic value)
        // do more passes of chebysev filter till the check passes.
        // This improves the scf convergence performance.


        while (maxRes > chebyTol && count < 100)
          {
            for (unsigned int kPoint = 0; kPoint < d_kPointWeights.size();
                 ++kPoint)
              {
                if (d_dftParamsPtr->verbosity >= 2)
                  pcout << "Beginning Chebyshev filter pass " << 1 + count
                        << std::endl;

                kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint, 0);
                if (d_dftParamsPtr->memOptMode && d_kPointWeights.size() > 0)
                  {
                    computing_timer.enter_subsection(
                      "Hamiltonian Matrix Computation");
                    kohnShamDFTEigenOperator.computeCellHamiltonianMatrix();
                    computing_timer.leave_subsection(
                      "Hamiltonian Matrix Computation");
                  }

#ifdef DFTFE_WITH_DEVICE
                if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
                  kohnShamEigenSpaceCompute(
                    0,
                    kPoint,
                    kohnShamDFTEigenOperator,
                    *d_elpaScala,
                    d_subspaceIterationSolverDevice,
                    residualNormWaveFunctionsAllkPoints[kPoint],
                    true,
                    0,
                    false,
                    true,
                    true);

#endif
                if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
                  kohnShamEigenSpaceCompute(
                    0,
                    kPoint,
                    kohnShamDFTEigenOperator,
                    *d_elpaScala,
                    d_subspaceIterationSolver,
                    residualNormWaveFunctionsAllkPoints[kPoint],
                    true,
                    false,
                    true,
                    true);
              }

            // //
            if (!(d_dftParamsPtr->writeBandsFile))
              {
                if (d_dftParamsPtr->constraintMagnetization)
                  compute_fermienergy_constraintMagnetization(eigenValues);
                else
                  compute_fermienergy(eigenValues, numElectrons);
              }
            //
            if (d_dftParamsPtr->highestStateOfInterestForChebFiltering == 0)
              {
                maxRes = computeMaximumHighestOccupiedStateResidualNorm(
                  residualNormWaveFunctionsAllkPoints,
                  eigenValues,
                  fermiEnergy);
              }
            else
              {
                maxRes = computeMaximumHighestOccupiedStateResidualNorm(
                  residualNormWaveFunctionsAllkPoints,
                  eigenValues,
                  d_dftParamsPtr->highestStateOfInterestForChebFiltering);
              }
            if (d_dftParamsPtr->verbosity >= 2)
              {
                if (d_dftParamsPtr->highestStateOfInterestForChebFiltering == 0)
                  {
                    pcout
                      << "Maximum residual norm of the state closest to and below Fermi level: "
                      << maxRes << std::endl;
                  }
                else
                  {
                    pcout
                      << "Maximum residual norm of the highest state of interest : "
                      << maxRes << std::endl;
                  }
              }

            count++;
          }


        numberChebyshevSolvePasses = count;

        if (d_dftParamsPtr->verbosity == 0 &&
            d_dftParamsPtr->reproducible_output)
          {
            pcout << "Fermi Energy computed: " << std::fixed
                  << std::setprecision(8) << fermiEnergy << std::endl;
          }
        else
          {
            pcout << "Fermi Energy computed: " << fermiEnergy << std::endl;
          }
      }
    computing_timer.enter_subsection("compute rho");

    compute_rhoOut(false, true);

    computing_timer.leave_subsection("compute rho");

    updateAuxDensityXCMatrix(d_densityOutQuadValues,
                             d_gradDensityOutQuadValues,
                             d_rhoCore,
                             d_gradRhoCore,
                             getEigenVectors(),
                             eigenValues,
                             fermiEnergy,
                             fermiEnergyUp,
                             fermiEnergyDown,
                             d_auxDensityMatrixXCOutPtr);

    //
    // compute integral rhoOut
    //
    const double integralRhoValue =
      totalCharge(d_dofHandlerPRefined, d_densityOutQuadValues[0]);

    if (d_dftParamsPtr->verbosity >= 2)
      {
        pcout << std::endl
              << "number of electrons: " << integralRhoValue << std::endl;
      }

    if (d_dftParamsPtr->verbosity >= 1 && d_dftParamsPtr->spinPolarized == 1)
      totalMagnetization(d_densityOutQuadValues[1]);


    local_timer.stop();
    if (d_dftParamsPtr->verbosity >= 1)
      pcout << "Wall time for the above scf iteration: "
            << local_timer.wall_time() << " seconds\n"
            << "Number of Chebyshev filtered subspace iterations: "
            << numberChebyshevSolvePasses << std::endl
            << std::endl;

    //
    // phiTot with rhoOut
    //

    if (d_dftParamsPtr->verbosity >= 2)
      pcout << std::endl
            << "Poisson solve for total electrostatic potential (rhoOut+b): ";

    computing_timer.enter_subsection("phiTot solve");

    if (d_dftParamsPtr->multipoleBoundaryConditions)
      {
        computing_timer.enter_subsection("Update inhomogenous BC");
        computeMultipoleMoments(d_basisOperationsPtrElectroHost,
                                d_densityQuadratureIdElectro,
                                d_densityOutQuadValues[0],
                                &d_bQuadValuesAllAtoms);
        updatePRefinedConstraints();
        computing_timer.leave_subsection("Update inhomogenous BC");
      }

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      densityOutQuadValuesCopy = d_densityOutQuadValues[0];
    if (std::abs(d_dftParamsPtr->netCharge) > 1e-12 and
        (d_dftParamsPtr->periodicX || d_dftParamsPtr->periodicY ||
         d_dftParamsPtr->periodicZ))
      {
        double *tempvec = densityOutQuadValuesCopy.data();
        for (unsigned int iquad = 0; iquad < densityOutQuadValuesCopy.size();
             iquad++)
          tempvec[iquad] += -d_dftParamsPtr->netCharge / d_domainVolume;
      }


    if (d_dftParamsPtr->useDevice and d_dftParamsPtr->poissonGPU and
        d_dftParamsPtr->floatingNuclearCharges and
        not d_dftParamsPtr->pinnedNodeForPBC)
      {
#ifdef DFTFE_WITH_DEVICE
        d_phiTotalSolverProblemDevice.reinit(
          d_basisOperationsPtrElectroHost,
          d_phiTotRhoOut,
          *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
          d_phiTotDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          d_phiTotAXQuadratureIdElectro,
          d_atomNodeIdToChargeMap,
          d_bQuadValuesAllAtoms,
          d_smearedChargeQuadratureIdElectro,
          densityOutQuadValuesCopy,
          d_BLASWrapperPtr,
          false,
          false,
          d_dftParamsPtr->smearedNuclearCharges,
          true,
          false,
          0,
          false,
          true,
          d_dftParamsPtr->multipoleBoundaryConditions);

        CGSolverDevice.solve(d_phiTotalSolverProblemDevice,
                             d_dftParamsPtr->absLinearSolverTolerance,
                             d_dftParamsPtr->maxLinearSolverIterations,
                             d_dftParamsPtr->verbosity);
#endif
      }
    else
      {
        d_phiTotalSolverProblem.reinit(
          d_basisOperationsPtrElectroHost,
          d_phiTotRhoOut,
          *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
          d_phiTotDofHandlerIndexElectro,
          d_densityQuadratureIdElectro,
          d_phiTotAXQuadratureIdElectro,
          d_atomNodeIdToChargeMap,
          d_bQuadValuesAllAtoms,
          d_smearedChargeQuadratureIdElectro,
          densityOutQuadValuesCopy,
          false,
          false,
          d_dftParamsPtr->smearedNuclearCharges,
          true,
          false,
          0,
          false,
          true,
          d_dftParamsPtr->multipoleBoundaryConditions);

        CGSolver.solve(d_phiTotalSolverProblem,
                       d_dftParamsPtr->absLinearSolverTolerance,
                       d_dftParamsPtr->maxLinearSolverIterations,
                       d_dftParamsPtr->verbosity);
      }

    interpolateElectroNodalDataToQuadratureDataGeneral(
      d_basisOperationsPtrElectroHost,
      d_phiTotDofHandlerIndexElectro,
      d_densityQuadratureIdElectro,
      d_phiTotRhoOut,
      d_phiOutQuadValues,
      dummy);

    computing_timer.leave_subsection("phiTot solve");

    // const Quadrature<3> &quadrature =
    // matrix_free_data.get_quadrature(d_densityQuadratureId);
    d_dispersionCorr.computeDispresionCorrection(atomLocations,
                                                 d_domainBoundingVectors);


    computeFractionalOccupancies();

    d_excManagerPtr->getExcSSDFunctionalObj()
      ->updateWaveFunctionDependentFuncDerWrtPsi(d_auxDensityMatrixXCOutPtr,
                                                 d_kPointWeights);

    d_excManagerPtr->getExcSSDFunctionalObj()
      ->computeWaveFunctionDependentExcEnergy(d_auxDensityMatrixXCOutPtr,
                                              d_kPointWeights);

    const double totalEnergy = energyCalc.computeEnergy(
      d_basisOperationsPtrHost,
      d_basisOperationsPtrElectroHost,
      d_densityQuadratureId,
      d_densityQuadratureIdElectro,
      d_smearedChargeQuadratureIdElectro,
      d_lpspQuadratureIdElectro,
      eigenValues,
      d_kPointWeights,
      fermiEnergy,
      d_dftParamsPtr->spinPolarized == 0 ? fermiEnergy : fermiEnergyUp,
      d_dftParamsPtr->spinPolarized == 0 ? fermiEnergy : fermiEnergyDown,
      d_excManagerPtr,
      d_dispersionCorr,
      d_phiInQuadValues,
      d_phiOutQuadValues,
      d_phiTotRhoOut,
      d_densityInQuadValues,
      d_densityOutQuadValues,
      d_gradDensityOutQuadValues,
      d_densityTotalOutValuesLpspQuad,
      d_auxDensityMatrixXCInPtr,
      d_auxDensityMatrixXCOutPtr,
      d_bQuadValuesAllAtoms,
      d_bCellNonTrivialAtomIds,
      d_localVselfs,
      d_pseudoVLoc,
      d_atomNodeIdToChargeMap,
      atomLocations.size(),
      lowerBoundKindex,
      0,
      d_dftParamsPtr->verbosity >= 0 ? true : false,
      d_dftParamsPtr->smearedNuclearCharges);

    if (d_dftParamsPtr->verbosity <= 1)
      pcout << "Total energy  : " << totalEnergy << std::endl;

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

    if (d_dftParamsPtr->verbosity >= 0)
      pcout << "Total entropic energy: " << d_entropicEnergy << std::endl;


    d_freeEnergy = d_groundStateEnergy - d_entropicEnergy;

    if (d_dftParamsPtr->verbosity >= 0)
      pcout << "Total free energy: " << d_freeEnergy << std::endl;



    computing_timer.leave_subsection("nscf solve");
    computingTimerStandard.leave_subsection("Total nscf solve");


#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice &&
        (d_dftParamsPtr->writeWfcSolutionFields ||
         d_dftParamsPtr->writeLdosFile || d_dftParamsPtr->writePdosFile))
      d_eigenVectorsFlattenedDevice.copyTo(d_eigenVectorsFlattenedHost);
#endif

    //#ifdef USE_COMPLEX
    //  if (!(d_dftParamsPtr->kPointDataFile == ""))
    //  {
    //  readkPointData();
    // initnscf(kohnShamDFTEigenOperator, d_phiTotalSolverProblem, CGSolver);
    // nscf(kohnShamDFTEigenOperator, d_subspaceIterationSolver);
    // writeBands();
    //}
    //#endif
  }
#include "dft.inst.cc"
} // namespace dftfe
