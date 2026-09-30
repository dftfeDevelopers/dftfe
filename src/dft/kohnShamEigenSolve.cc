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
// @author  Phani Motamarri, Sambit Das
//
#include <complex>
#include <vector>
#include <dft.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsCPU.h>
namespace dftfe
{
  namespace internal
  {
    void
    pointWiseScaleWithDiagonal(const double      *diagonal,
                               const dftfe::uInt  numberFields,
                               const dftfe::uInt  numberDofs,
                               dataTypes::number *fieldsArrayFlattened)
    {
      const unsigned int inc             = 1;
      unsigned int       numberFieldsTmp = numberFields;
      for (dftfe::uInt i = 0; i < numberDofs; ++i)
        {
#ifdef USE_COMPLEX
          double scalingCoeff = diagonal[i];
          zdscal_(&numberFieldsTmp,
                  &scalingCoeff,
                  &fieldsArrayFlattened[i * numberFields],
                  &inc);
#else
          double scalingCoeff = diagonal[i];
          dscal_(&numberFieldsTmp,
                 &scalingCoeff,
                 &fieldsArrayFlattened[i * numberFields],
                 &inc);
#endif
        }
    }
  } // namespace internal

  //
  template <dftfe::utils::MemorySpace memorySpace>
  dataTypes::number
  dftClass<memorySpace>::computeTraceXtHX(
    dftfe::uInt numberWaveFunctionsEstimate)
  {
    // //
    // // set up poisson solver
    // //
    // dealiiLinearSolver                            CGSolver(d_mpiCommParent,
    //                             mpi_communicator,
    //                             dealiiLinearSolver::CG);
    // poissonSolverProblem<FEOrder, FEOrderElectro> phiTotalSolverProblem(
    //   mpi_communicator);

    // //
    // // solve for vself and compute Tr(XtHX)
    // //
    // d_vselfBinsManager.solveVselfInBins(d_basisOperationsPtrElectroHost,
    //                                     d_binsStartDofHandlerIndexElectro,
    //                                     d_phiTotAXQuadratureIdElectro,
    //                                     d_constraintsPRefined,
    //                                     d_imagePositionsTrunc,
    //                                     d_imageIdsTrunc,
    //                                     d_imageChargesTrunc,
    //                                     d_localVselfs,
    //                                     d_bQuadValuesAllAtoms,
    //                                     d_bQuadAtomIdsAllAtomsImages,
    //                                     d_bQuadAtomIdsAllAtoms,
    //                                     d_bCellNonTrivialAtomIds,
    //                                     d_bCellNonTrivialAtomIdsBins,
    //                                     d_bCellNonTrivialAtomImageIds,
    //                                     d_bCellNonTrivialAtomImageIdsBins,
    //                                     d_smearedChargeWidths,
    //                                     d_smearedChargeScaling,
    //                                     d_smearedChargeQuadratureIdElectro);

    // //
    // // solve for potential corresponding to initial electron-density
    // //
    // phiTotalSolverProblem.reinit(
    //   d_basisOperationsPtrElectroHost,
    //   d_phiTotRhoIn,
    //   *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
    //   d_phiTotDofHandlerIndexElectro,
    //   d_phiTotAXQuadratureIdElectro,
    //   d_densityQuadratureIdElectro,
    //   d_atomNodeIdToChargeMap,
    //   d_bQuadValuesAllAtoms,
    //   d_smearedChargeQuadratureIdElectro,
    //   d_densityInQuadValues[0],
    //   true,
    //   d_dftParamsPtr->periodicX && d_dftParamsPtr->periodicY &&
    //     d_dftParamsPtr->periodicZ && !d_dftParamsPtr->pinnedNodeForPBC,
    //   d_dftParamsPtr->smearedNuclearCharges);

    // dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
    //   phiInValues;

    // CGSolver.solve(phiTotalSolverProblem,
    //                d_dftParamsPtr->absLinearSolverTolerance,
    //                d_dftParamsPtr->maxLinearSolverIterations,
    //                d_dftParamsPtr->verbosity);

    // dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
    // dummy; interpolateElectroNodalDataToQuadratureDataGeneral(
    //   d_basisOperationsPtrElectroHost,
    //   d_phiTotDofHandlerIndexElectro,
    //   d_densityQuadratureIdElectro,
    //   d_phiTotRhoIn,
    //   phiInValues,
    //   dummy);

    // //
    // // create kohnShamDFTOperatorClass object
    // //
    // kohnShamDFTOperatorClass<memorySpace>
    //   kohnShamDFTEigenOperator(this, d_mpiCommParent, mpi_communicator);
    // kohnShamDFTEigenOperator.init();

    // //
    // // precompute shapeFunctions and shapeFunctionGradients and
    // // shapeFunctionGradientIntegrals
    // //
    // kohnShamDFTEigenOperator.preComputeShapeFunctionGradientIntegrals(
    //   d_lpspQuadratureId);

    // //
    // // compute Veff
    // //
    //    bool isGradDensityDataDependent = false;
    //    if (d_excManagerPtr->getXCPrimaryVariable() ==
    //    XCPrimaryVariable::DENSITY)
    //      {
    //        isGradDensityDataDependent =
    //        (d_excManagerPtr->getExcDensityObj()->getDensityBasedFamilyType()
    //        == densityFamilyType::GGA) ;
    //      }
    //    else if (d_excManagerPtr->getXCPrimaryVariable() ==
    //    XCPrimaryVariable::SSDETERMINANT)
    //      {
    //        isGradDensityDataDependent =
    //        (d_excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType()
    //        == densityFamilyType::GGA) ;
    //      }
    // if (!isGradDensityDataDependent)
    //   {
    //     kohnShamDFTEigenOperator.computeVEff(d_densityInQuadValues,
    //                                          phiInValues,
    //                                          d_pseudoVLoc,
    //                                          d_rhoCore,
    //                                          d_lpspQuadratureId);
    //   }
    // else if (isGradDensityDataDependent)
    //   {
    //     kohnShamDFTEigenOperator.computeVEff(d_densityInQuadValues,
    //                                          d_gradDensityInQuadValues,
    //                                          phiInValues,
    //                                          d_pseudoVLoc,
    //                                          d_rhoCore,
    //                                          d_gradRhoCore,
    //                                          d_lpspQuadratureId);
    //   }

    // //
    // // compute Hamiltonian matrix
    // //
    // kohnShamDFTEigenOperator.computeHamiltonianMatrix(0, 0);

    // //
    // // scale the eigenVectors (initial guess of single atom wavefunctions or
    // // previous guess) to convert into Lowden Orthonormalized FE basis
    // multiply
    // // by M^{1/2}
    // internal::pointWiseScaleWithDiagonal(
    //   kohnShamDFTEigenOperator.d_sqrtMassVector,
    //   d_numEigenValues,
    //   matrix_free_data.get_vector_partitioner()->locally_owned_size(),
    //   d_eigenVectorsFlattenedHost.data());


    // //
    // // compute projected Hamiltonian
    // //
    // std::vector<dataTypes::number> ProjHam;

    // dftfe::linearAlgebraOperations::XtHX(
    //   kohnShamDFTEigenOperator,
    //   d_eigenVectorsFlattenedHost.data(),
    //   d_numEigenValues,
    //   matrix_free_data.get_vector_partitioner()->locally_owned_size(),
    //   mpi_communicator,
    //   interBandGroupComm,
    //   *d_dftParamsPtr,
    //   ProjHam);

    // //
    // // scale the eigenVectors with M^{-1/2} to represent the wavefunctions in
    // // the usual FE basis
    // //
    // internal::pointWiseScaleWithDiagonal(
    //   kohnShamDFTEigenOperator.getInverseSqrtMassVector().data(),
    //   d_numEigenValues,
    //   matrix_free_data.get_vector_partitioner()->locally_owned_size(),
    //   d_eigenVectorsFlattenedHost.data());


    // dataTypes::number trXtHX = 0.0;
    // for (dftfe::uInt i = 0; i < numberWaveFunctionsEstimate; ++i)
    //   {
    //     trXtHX += ProjHam[d_numEigenValues * i + i];
    //   }

    // return trXtHX;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<memorySpace>::computeTraceXtKX(
    dftfe::uInt numberWaveFunctionsEstimate)
  {
    //     //
    //     // create kohnShamDFTOperatorClass object
    //     //
    //     kohnShamDFTOperatorClass<memorySpace>
    //       kohnShamDFTEigenOperator(this, d_mpiCommParent, mpi_communicator);
    //     kohnShamDFTEigenOperator.init();

    //     //
    //     // precompute shapeFunctions and shapeFunctionGradients and
    //     // shapeFunctionGradientIntegrals
    //     //
    //     kohnShamDFTEigenOperator.preComputeShapeFunctionGradientIntegrals(
    //       d_lpspQuadratureId);


    //     //
    //     // compute Hamiltonian matrix
    //     //
    //     kohnShamDFTEigenOperator.computeKineticMatrix();

    //     //
    //     // scale the eigenVectors (initial guess of single atom wavefunctions
    //     or
    //     // previous guess) to convert into Lowden Orthonormalized FE basis
    //     multiply
    //     // by M^{1/2}
    //     internal::pointWiseScaleWithDiagonal(
    //       kohnShamDFTEigenOperator.d_sqrtMassVector,
    //       d_numEigenValues,
    //       matrix_free_data.get_vector_partitioner()->locally_owned_size(),
    //       d_eigenVectorsFlattenedHost.data());


    //     //
    //     // orthogonalize the vectors
    //     //
    //     linearAlgebraOperations::gramSchmidtOrthogonalization(
    //       d_eigenVectorsFlattenedHost.data(),
    //       d_numEigenValues,
    //       matrix_free_data.get_vector_partitioner()->locally_owned_size(),
    //       mpi_communicator);

    //     //
    //     // compute projected Hamiltonian
    //     //
    //     std::vector<dataTypes::number> ProjHam;

    //     dftfe::linearAlgebraOperations::XtHX(
    //       kohnShamDFTEigenOperator,
    //       d_eigenVectorsFlattenedHost.data(),
    //       d_numEigenValues,
    //       matrix_free_data.get_vector_partitioner()->locally_owned_size(),
    //       mpi_communicator,
    //       interBandGroupComm,
    //       *d_dftParamsPtr,
    //       ProjHam);

    //     //
    //     // scale the eigenVectors with M^{-1/2} to represent the
    //     wavefunctions in
    //     // the usual FE basis
    //     //
    //     internal::pointWiseScaleWithDiagonal(
    //       kohnShamDFTEigenOperator.getInverseSqrtMassVector().data(),
    //       d_numEigenValues,
    //       matrix_free_data.get_vector_partitioner()->locally_owned_size(),
    //       d_eigenVectorsFlattenedHost.data());

    //     double trXtKX = 0.0;
    // #ifdef USE_COMPLEX
    //     trXtKX = 0.0;
    // #else
    //     for (dftfe::uInt i = 0; i < numberWaveFunctionsEstimate; ++i)
    //       {
    //         trXtKX += ProjHam[d_numEigenValues * i + i];
    //       }
    // #endif

    //     return trXtKX;
  }



  // chebyshev solver
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::kohnShamEigenSpaceCompute(
    const dftfe::uInt spinType,
    const dftfe::uInt kPointIndex,
    KohnShamDFTBaseOperator<dftfe::utils::MemorySpace::HOST>
                                                   &kohnShamDFTEigenOperator,
    elpaScalaManager                               &elpaScala,
    chebyshevOrthogonalizedSubspaceIterationSolver &subspaceIterationSolver,
    std::vector<double>                            &residualNormWaveFunctions,
    const bool                                      computeResidual,
    const bool                                      useMixedPrec,
    const bool                                      isFirstScf)
  {
    computing_timer.enter_subsection("Chebyshev solve");


    if (d_dftParamsPtr->verbosity >= 2)
      {
        pcout << "kPoint: " << kPointIndex << std::endl;
        if (d_dftParamsPtr->spinPolarized == 1)
          pcout << "spin: " << spinType + 1 << std::endl;
      }

    std::vector<double> eigenValuesTemp(d_numEigenValues, 0.0);

    for (dftfe::uInt i = 0; i < d_numEigenValues; i++)
      {
        eigenValuesTemp[i] =
          eigenValues[kPointIndex][spinType * d_numEigenValues + i];
      }

    if (d_isFirstFilteringCall[(1 + d_dftParamsPtr->spinPolarized) *
                                 kPointIndex +
                               spinType])
      {
        computing_timer.enter_subsection("Lanczos k-step Upper Bound");

        std::pair<double, double> bounds = linearAlgebraOperations::
          generalisedLanczosLowerUpperBoundEigenSpectrum(
            d_BLASWrapperPtrHost,
            kohnShamDFTEigenOperator,
            kohnShamDFTEigenOperator.getScratchFEMultivector(1, 0),
            kohnShamDFTEigenOperator.getScratchFEMultivector(1, 1),
            kohnShamDFTEigenOperator.getScratchFEMultivector(1, 2),
            kohnShamDFTEigenOperator.getScratchFEMultivector(1, 3),
            *d_dftParamsPtr);

        const double upperBoundUnwantedSpectrum = bounds.second;
        const double lowerBoundWantedSpectrum   = bounds.first;

        a0[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType] =
          lowerBoundWantedSpectrum;
        computing_timer.leave_subsection("Lanczos k-step Upper Bound");

        d_upperBoundUnwantedSpectrumValues[(1 + d_dftParamsPtr->spinPolarized) *
                                             kPointIndex +
                                           spinType] =
          upperBoundUnwantedSpectrum;

        subspaceIterationSolver.reinitSpectrumBounds(
          lowerBoundWantedSpectrum,
          lowerBoundWantedSpectrum +
            (upperBoundUnwantedSpectrum - lowerBoundWantedSpectrum) /
              kohnShamDFTEigenOperator.getScratchFEMultivector(1, 0)
                .globalSize() *
              d_numEigenValues *
              (d_dftParamsPtr->reproducible_output ? 10.0 : 200.0),
          upperBoundUnwantedSpectrum);
      }
    else
      {
        if (!d_dftParamsPtr->reuseLanczosUpperBoundFromFirstCall)
          {
            computing_timer.enter_subsection("Lanczos k-step Upper Bound");

            std::pair<double, double> bounds = linearAlgebraOperations::
              generalisedLanczosLowerUpperBoundEigenSpectrum(
                d_BLASWrapperPtrHost,
                kohnShamDFTEigenOperator,
                kohnShamDFTEigenOperator.getScratchFEMultivector(1, 0),
                kohnShamDFTEigenOperator.getScratchFEMultivector(1, 1),
                kohnShamDFTEigenOperator.getScratchFEMultivector(1, 2),
                kohnShamDFTEigenOperator.getScratchFEMultivector(1, 3),
                *d_dftParamsPtr);

            d_upperBoundUnwantedSpectrumValues
              [(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType] =
                bounds.second;
            computing_timer.leave_subsection("Lanczos k-step Upper Bound");
          }

        subspaceIterationSolver.reinitSpectrumBounds(
          a0[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType],
          bLow[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType],
          d_upperBoundUnwantedSpectrumValues
            [(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType]);
      }
    const dftfe::uInt wfcStartIndex =
      d_dftParamsPtr->solverMode == "BANDS" ?
        0 :
        ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
          d_numEigenValues *
          matrix_free_data.get_vector_partitioner()->locally_owned_size();
    subspaceIterationSolver.solve(
      kohnShamDFTEigenOperator,
      d_BLASWrapperPtrHost,
      elpaScala,
      d_eigenVectorsFlattenedHost.data() + wfcStartIndex,
      d_numEigenValues,
      matrix_free_data.get_vector_partitioner()->locally_owned_size(),
      eigenValuesTemp,
      residualNormWaveFunctions,
      interBandGroupComm,
      mpi_communicator,
      d_isFirstFilteringCall[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex +
                             spinType],
      computeResidual,
      useMixedPrec,
      isFirstScf);

    //
    // copy the eigenValues and corresponding residual norms back to data
    // members
    //


    for (dftfe::uInt i = 0; i < d_numEigenValues; i++)
      {
        if (d_dftParamsPtr->verbosity >= 4)
          pcout << "eigen value " << std::setw(3) << i << ": "
                << eigenValuesTemp[i] << std::endl;

        eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
          eigenValuesTemp[i];
      }

    if (d_dftParamsPtr->verbosity >= 4)
      pcout << std::endl;



    bLow[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType] =
      eigenValuesTemp.back();
    d_isFirstFilteringCall[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex +
                           spinType] = false;

    a0[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType] =
      eigenValuesTemp[0];


    computing_timer.leave_subsection("Chebyshev solve");
  }

#ifdef DFTFE_WITH_DEVICE
  // chebyshev solver
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::kohnShamEigenSpaceCompute(
    const dftfe::uInt spinType,
    const dftfe::uInt kPointIndex,
    KohnShamDFTBaseOperator<dftfe::utils::MemorySpace::DEVICE>
                     &kohnShamDFTEigenOperator,
    elpaScalaManager &elpaScala,
    chebyshevOrthogonalizedSubspaceIterationSolverDevice
                        &subspaceIterationSolverDevice,
    std::vector<double> &residualNormWaveFunctions,
    const bool           computeResidual,
    const dftfe::uInt    numberRayleighRitzAvoidancePasses,
    const bool           useMixedPrec,
    const bool           isFirstScf)
  {
    if (d_dftParamsPtr->verbosity >= 2)
      {
        pcout << "kPoint: " << kPointIndex << std::endl;
        if (d_dftParamsPtr->spinPolarized == 1)
          pcout << "spin: " << spinType + 1 << std::endl;
      }
    std::vector<double> eigenValuesTemp(d_numEigenValues, 0.0);
    if (d_dftParamsPtr->useSinglePrecCheby ||
        d_dftParamsPtr->useReformulatedChFSI)
      for (dftfe::uInt i = 0; i < d_numEigenValues; i++)
        {
          eigenValuesTemp[i] =
            eigenValues[kPointIndex][spinType * d_numEigenValues + i];
        }

    subspaceIterationSolverDevice.reinitSpectrumBounds(
      a0[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType],
      bLow[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType],
      d_upperBoundUnwantedSpectrumValues[(1 + d_dftParamsPtr->spinPolarized) *
                                           kPointIndex +
                                         spinType]);


    const dftfe::uInt wfcStartIndex =
      d_dftParamsPtr->solverMode == "BANDS" ?
        0 :
        ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
          d_numEigenValues *
          matrix_free_data.get_vector_partitioner()->locally_owned_size();

    d_upperBoundUnwantedSpectrumValues[(1 + d_dftParamsPtr->spinPolarized) *
                                         kPointIndex +
                                       spinType] =
      subspaceIterationSolverDevice.solve(
        kohnShamDFTEigenOperator,
        d_BLASWrapperPtr,
        elpaScala,
        d_eigenVectorsFlattenedDevice.begin() + wfcStartIndex,
        d_numEigenValues *
          matrix_free_data.get_vector_partitioner()->locally_owned_size(),
        d_numEigenValues,
        eigenValuesTemp,
        residualNormWaveFunctions,
        *d_devicecclMpiCommDomainPtr,
        interBandGroupComm,
        d_isFirstFilteringCall[(1 + d_dftParamsPtr->spinPolarized) *
                                 kPointIndex +
                               spinType],
        computeResidual,
        useMixedPrec,
        isFirstScf);



    //
    // copy the eigenValues and corresponding residual norms back to data
    // members
    //
    for (dftfe::uInt i = 0; i < d_numEigenValues; i++)
      {
        if (d_dftParamsPtr->verbosity >= 4)
          pcout << "eigen value " << std::setw(3) << i << ": "
                << eigenValuesTemp[i] << std::endl;

        eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
          eigenValuesTemp[i];
      }

    if (d_dftParamsPtr->verbosity >= 4)
      pcout << std::endl;


    bLow[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType] =
      eigenValuesTemp.back();
    d_isFirstFilteringCall[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex +
                           spinType] = false;

    a0[(1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType] =
      eigenValuesTemp[0];
  }
#endif


  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::kohnShamEigenSpaceFirstOrderDensityMatResponse(
    const dftfe::uInt spinType,
    const dftfe::uInt kPointIndex,
    KohnShamDFTBaseOperator<dftfe::utils::MemorySpace::HOST>
                     &kohnShamDFTEigenOperator,
    elpaScalaManager &elpaScala)
  {
    if (d_dftParamsPtr->verbosity >= 2)
      {
        pcout << "kPoint: " << kPointIndex << std::endl;
        if (d_dftParamsPtr->spinPolarized == 1)
          pcout << "spin: " << spinType + 1 << std::endl;
      }


    //
    // scale the eigenVectors to convert into Lowden Orthonormalized FE basis
    // multiply by M^{1/2}
    // internal::pointWiseScaleWithDiagonal(
    //   kohnShamDFTEigenOperator.getSqrtMassVector().data(),
    //   d_numEigenValues,
    //   matrix_free_data.get_vector_partitioner()->locally_owned_size(),
    //   d_eigenVectorsDensityMatrixPrimeHost.data() +
    //     ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
    //       d_numEigenValues *
    //       matrix_free_data.get_vector_partitioner()->locally_owned_size());

    std::vector<double> eigenValuesTemp(d_numEigenValues, 0.0);
    for (dftfe::uInt i = 0; i < d_numEigenValues; i++)
      {
        eigenValuesTemp[i] =
          eigenValues[kPointIndex][spinType * d_numEigenValues + i];
      }

    double fermiEnergyInput = fermiEnergy;
    if (d_dftParamsPtr->constraintMagnetization)
      fermiEnergyInput = spinType == 0 ? fermiEnergyUp : fermiEnergyDown;

    linearAlgebraOperations::densityMatrixEigenBasisFirstOrderResponse(
      kohnShamDFTEigenOperator,
      d_BLASWrapperPtrHost,
      d_eigenVectorsDensityMatrixPrimeHost.data() +
        ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
          d_numEigenValues *
          matrix_free_data.get_vector_partitioner()->locally_owned_size(),
      d_numEigenValues,
      matrix_free_data.get_vector_partitioner()->locally_owned_size(),
      d_mpiCommParent,
      kohnShamDFTEigenOperator.getMPICommunicatorDomain(),
      interBandGroupComm,
      eigenValuesTemp,
      fermiEnergyInput,
      d_densityMatDerFermiEnergy[(1 + d_dftParamsPtr->spinPolarized) *
                                   kPointIndex +
                                 spinType],
      elpaScala,
      *d_dftParamsPtr);


    //
    // scale the eigenVectors with M^{-1/2} to represent the wavefunctions in
    // the usual FE basis
    //
    // internal::pointWiseScaleWithDiagonal(
    //   kohnShamDFTEigenOperator.getInverseSqrtMassVector().data(),
    //   d_numEigenValues,
    //   matrix_free_data.get_vector_partitioner()->locally_owned_size(),
    //   d_eigenVectorsDensityMatrixPrimeHost.data() +
    //     ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
    //       d_numEigenValues *
    //       matrix_free_data.get_vector_partitioner()->locally_owned_size());
  }

#ifdef DFTFE_WITH_DEVICE
  // chebyshev solver
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::kohnShamEigenSpaceFirstOrderDensityMatResponse(
    const dftfe::uInt spinType,
    const dftfe::uInt kPointIndex,
    KohnShamDFTBaseOperator<dftfe::utils::MemorySpace::DEVICE>
                     &kohnShamDFTEigenOperator,
    elpaScalaManager &elpaScala,
    chebyshevOrthogonalizedSubspaceIterationSolverDevice
      &subspaceIterationSolverDevice)
  {
    if (d_dftParamsPtr->verbosity >= 2)
      {
        pcout << "kPoint: " << kPointIndex << std::endl;
        if (d_dftParamsPtr->spinPolarized == 1)
          pcout << "spin: " << spinType + 1 << std::endl;
      }

    std::vector<double> eigenValuesTemp(d_numEigenValues, 0.0);
    for (dftfe::uInt i = 0; i < d_numEigenValues; i++)
      {
        eigenValuesTemp[i] =
          eigenValues[kPointIndex][spinType * d_numEigenValues + i];
      }


    double fermiEnergyInput = fermiEnergy;
    if (d_dftParamsPtr->constraintMagnetization)
      fermiEnergyInput = spinType == 0 ? fermiEnergyUp : fermiEnergyDown;

    subspaceIterationSolverDevice.densityMatrixEigenBasisFirstOrderResponse(
      kohnShamDFTEigenOperator,
      d_BLASWrapperPtr,
      d_eigenVectorsDensityMatrixPrimeFlattenedDevice.begin() +
        ((1 + d_dftParamsPtr->spinPolarized) * kPointIndex + spinType) *
          d_numEigenValues *
          matrix_free_data.get_vector_partitioner()->locally_owned_size(),
      d_numEigenValues *
        matrix_free_data.get_vector_partitioner()->locally_owned_size(),
      d_numEigenValues,
      eigenValuesTemp,
      fermiEnergyInput,
      d_densityMatDerFermiEnergy[(1 + d_dftParamsPtr->spinPolarized) *
                                   kPointIndex +
                                 spinType],
      *d_devicecclMpiCommDomainPtr,
      interBandGroupComm,
      elpaScala);
  }
#endif

  // compute the maximum of the residual norm of the highest state of interest
  // across all K points
  template <dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<memorySpace>::computeMaximumHighestOccupiedStateResidualNorm(
    const std::vector<std::vector<double>> &residualNormWaveFunctionsAllkPoints,
    const std::vector<std::vector<double>> &eigenValuesAllkPoints,
    const dftfe::uInt                       highestState,
    std::vector<double>                    &maxResidualsAllkPoints)
  {
    double maxHighestOccupiedStateResNorm = -1e+6;
    maxResidualsAllkPoints.clear();
    maxResidualsAllkPoints.resize(eigenValuesAllkPoints.size());
    for (dftfe::Int kPoint = 0; kPoint < eigenValuesAllkPoints.size(); ++kPoint)
      {
        maxResidualsAllkPoints[kPoint] = *std::max_element(
          residualNormWaveFunctionsAllkPoints[kPoint].begin(),
          residualNormWaveFunctionsAllkPoints[kPoint].begin() + highestState);
      }
    maxHighestOccupiedStateResNorm =
      *std::max_element(maxResidualsAllkPoints.begin(),
                        maxResidualsAllkPoints.end());
    maxHighestOccupiedStateResNorm =
      dealii::Utilities::MPI::max(maxHighestOccupiedStateResNorm,
                                  interpoolcomm);
    return maxHighestOccupiedStateResNorm;
  }
  // compute the maximum of the residual norm of the highest occupied state
  // among all k points
  template <dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<memorySpace>::computeMaximumHighestOccupiedStateResidualNorm(
    const std::vector<std::vector<double>> &residualNormWaveFunctionsAllkPoints,
    const std::vector<std::vector<double>> &eigenValuesAllkPoints,
    const double                            fermiEnergy,
    std::vector<double>                    &maxResidualsAllkPoints)
  {
    double maxHighestOccupiedStateResNorm = -1e+6;
    maxResidualsAllkPoints.clear();
    maxResidualsAllkPoints.resize(eigenValuesAllkPoints.size(), -1e+6);
    for (dftfe::Int kPoint = 0; kPoint < eigenValuesAllkPoints.size(); ++kPoint)
      {
        dftfe::uInt highestOccupiedState = 0;

        for (dftfe::uInt i = 0; i < eigenValuesAllkPoints[kPoint].size(); i++)
          {
            if (d_partialOccupancies[kPoint][i] > 1e-3)
              highestOccupiedState = i;
          }

        for (dftfe::uInt i = 0; i <= highestOccupiedState; i++)
          {
            if (residualNormWaveFunctionsAllkPoints[kPoint][i] >
                maxResidualsAllkPoints[kPoint])
              {
                maxResidualsAllkPoints[kPoint] =
                  residualNormWaveFunctionsAllkPoints[kPoint][i];
              }
          }
      }
    maxHighestOccupiedStateResNorm =
      *std::max_element(maxResidualsAllkPoints.begin(),
                        maxResidualsAllkPoints.end());
    maxHighestOccupiedStateResNorm =
      dealii::Utilities::MPI::max(maxHighestOccupiedStateResNorm,
                                  interpoolcomm);
    return maxHighestOccupiedStateResNorm;
  }
#include "dft.inst.cc"
} // namespace dftfe
