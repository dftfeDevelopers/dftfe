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
// @author Phani Motamarri, Sambit Das
//


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
              d_dftParamsPtr->absLinearSolverTolerance,
              d_dftParamsPtr->maxLinearSolverIterations,
              kohnShamDFTEigenOperatorDevice.getDeviceBlasHandle(),
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
                  d_dftParamsPtr->absLinearSolverTolerance,
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
                               d_dftParamsPtr->absLinearSolverTolerance,
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
              d_dftParamsPtr->absLinearSolverTolerance,
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
                           d_dftParamsPtr->absLinearSolverTolerance,
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

