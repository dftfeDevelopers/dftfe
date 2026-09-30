// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
// @author Nikhil Kodali
//


//
// dft bands solve (non-selfconsistent solution of DFT eigenvalue problem to
// compute
// eigenvalues, eigenfunctions
// using the self-consistent Hamiltonian)
//
#include <dft.h>
namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::solveBands()
  {
    KohnShamDFTBaseOperator<memorySpace> &kohnShamDFTEigenOperator =
      *d_kohnShamDFTOperatorPtr;

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
        d_dftParamsPtr->finiteElementPolynomialOrder ==
            d_dftParamsPtr->finiteElementPolynomialOrderElectrostatics ?
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


    computingTimerStandard.enter_subsection("Total bands solve");

    //
    // solve
    //
    computing_timer.enter_subsection("bands solve");

    double chebyTol;
    chebyTol = d_dftParamsPtr->chebyshevTolerance == 0.0 ?
                 1e-08 :
                 d_dftParamsPtr->chebyshevTolerance;



    if (d_dftParamsPtr->verbosity >= 0)
      pcout << "Starting NSCF iteration for bands...." << std::endl;

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
        for (dftfe::uInt iquad = 0; iquad < densityInQuadValuesCopy.size();
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
          true);

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
          true);
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
    d_basisOperationsPtrElectroHost->interpolate(d_phiTotRhoIn,
                                                 d_phiTotDofHandlerIndexElectro,
                                                 d_densityQuadratureIdElectro,
                                                 d_phiInQuadValues,
                                                 dummy,
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

    dftfe::uInt numberChebyshevSolvePasses = 0;
    //
    // eigen solve
    //
    // get the fermi energy from the fermiEnergy.out
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
        pcout << "Unable to open file fermiEnergy.out. Check if it is present.";
      }

    std::vector<double> residualNormWaveFunctions(d_numEigenValues);

    updateAuxDensityXCMatrix(d_densityInQuadValues,
                             d_gradDensityInQuadValues,
                             d_tauInQuadValues,
                             d_rhoCore,
                             d_gradRhoCore,
                             getEigenVectors(),
                             eigenValues,
                             fermiEnergy,
                             fermiEnergyUp,
                             fermiEnergyDown,
                             d_auxDensityMatrixXCInPtr);

    const dftfe::uInt maxPasses = 100;


    double maxRes = 1.0;

    for (dftfe::uInt s = 0; s < d_dftParamsPtr->spinPolarized + 1; ++s)
      {
        computing_timer.enter_subsection("VEff Computation");
        kohnShamDFTEigenOperator.computeVEff(d_auxDensityMatrixXCInPtr,
                                             d_phiInQuadValues,
                                             s);

        computing_timer.leave_subsection("VEff Computation");
        for (dftfe::uInt kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
          {
            dftfe::uInt count = 0;
            while (count == 0 || maxRes > chebyTol)
              {
                if (d_dftParamsPtr->verbosity >= 2)
                  pcout << "Beginning Chebyshev filter pass " << 1 + count
                        << " for spin " << s + 1 << std::endl;

                kohnShamDFTEigenOperator.reinitkPointSpinIndex(kPoint, s);
                if (d_dftParamsPtr->memOptMode || count == 0)
                  {
                    computing_timer.enter_subsection(
                      "Hamiltonian Matrix Computation");
                    kohnShamDFTEigenOperator.computeCellHamiltonianMatrix();
                    computing_timer.leave_subsection(
                      "Hamiltonian Matrix Computation");
                  }

#ifdef DFTFE_WITH_DEVICE
                if constexpr (dftfe::utils::MemorySpace::DEVICE == memorySpace)
                  kohnShamEigenSpaceCompute(s,
                                            kPoint,
                                            kohnShamDFTEigenOperator,
                                            *d_elpaScala,
                                            d_subspaceIterationSolverDevice,
                                            residualNormWaveFunctions,
                                            true,
                                            0,
                                            false,
                                            true);
#endif
                if constexpr (dftfe::utils::MemorySpace::HOST == memorySpace)
                  kohnShamEigenSpaceCompute(s,
                                            kPoint,
                                            kohnShamDFTEigenOperator,
                                            *d_elpaScala,
                                            d_subspaceIterationSolver,
                                            residualNormWaveFunctions,
                                            true,
                                            false,
                                            true);
                maxRes = *std::max_element(
                  residualNormWaveFunctions.begin(),
                  residualNormWaveFunctions.begin() +
                    d_dftParamsPtr->highestStateOfInterestForChebFiltering);
                if (d_dftParamsPtr->verbosity >= 1)
                  pcout
                    << "Maximum residual norm of the highest state of interest : "
                    << maxRes << std::endl;
                ++count;
              }
          }
      }

    if (d_dftParamsPtr->verbosity >= 1)
      {
        pcout << "Fermi Energy computed: " << fermiEnergy << std::endl;
      }
    local_timer.stop();
    if (d_dftParamsPtr->verbosity >= 1)
      pcout << "Wall time for the above scf iteration: "
            << local_timer.wall_time() << " seconds\n"
            << "Number of Chebyshev filtered subspace iterations: "
            << numberChebyshevSolvePasses << std::endl
            << std::endl;



    computing_timer.leave_subsection("bands solve");
    computingTimerStandard.leave_subsection("Total bands solve");


#ifdef DFTFE_WITH_DEVICE
    if (d_dftParamsPtr->useDevice && (d_dftParamsPtr->writeWfcSolutionFields))
      d_eigenVectorsFlattenedDevice.copyTo(d_eigenVectorsFlattenedHost);
#endif
  }
#include "dft.inst.cc"
} // namespace dftfe
