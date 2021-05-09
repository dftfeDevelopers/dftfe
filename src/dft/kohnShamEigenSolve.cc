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
// @author  Phani Motamarri, Sambit Das
//
#include <complex>
#include <vector>

namespace internal
{
  void
  pointWiseScaleWithDiagonal(
    const distributedCPUVec<double> &       diagonal,
    std::vector<distributedCPUVec<double>> &fieldArray,
    dftUtils::constraintMatrixInfo &        constraintsNoneEigenDataInfo)
  {
    for (unsigned int i = 0; i < fieldArray.size(); ++i)
      {
        auto &vec = fieldArray[i];
        vec.scale(diagonal);
        constraintsNoneEigenDataInfo.distribute(vec);
        vec.update_ghost_values();
      }
  }


  void
  pointWiseScaleWithDiagonal(
    const distributedCPUVec<double> &diagonal,
    const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
      &                             singleComponentPartitioner,
    const unsigned int              numberFields,
    std::vector<dataTypes::number> &fieldsArrayFlattened)
  {
    const unsigned int numberDofs = fieldsArrayFlattened.size() / numberFields;
    const unsigned int inc        = 1;

    for (unsigned int i = 0; i < numberDofs; ++i)
      {
#ifdef USE_COMPLEX
        double scalingCoeff = diagonal.local_element(i);
        zdscal_(&numberFields,
                &scalingCoeff,
                &fieldsArrayFlattened[i * numberFields],
                &inc);
#else
        double scalingCoeff = diagonal.local_element(i);
        dscal_(&numberFields,
               &scalingCoeff,
               &fieldsArrayFlattened[i * numberFields],
               &inc);
#endif
      }
  }
} // namespace internal

//
template <unsigned int FEOrder, unsigned int FEOrderElectro>
dataTypes::number
dftClass<FEOrder, FEOrderElectro>::computeTraceXtHX(
  unsigned int numberWaveFunctionsEstimate)
{
  //
  // set up poisson solver
  //
  dealiiLinearSolver dealiiCGSolver(mpi_communicator, dealiiLinearSolver::CG);
  poissonSolverProblem<FEOrder, FEOrderElectro> phiTotalSolverProblem(
    mpi_communicator);

  //
  // solve for vself and compute Tr(XtHX)
  //
  d_vselfBinsManager.solveVselfInBins(d_matrixFreeDataPRefined,
                                      d_binsStartDofHandlerIndexElectro,
                                      d_phiTotAXQuadratureIdElectro,
                                      d_constraintsPRefined,
                                      d_imagePositionsTrunc,
                                      d_imageIdsTrunc,
                                      d_imageChargesTrunc,
                                      d_localVselfs,
                                      d_bQuadValuesAllAtoms,
                                      d_bQuadAtomIdsAllAtomsImages,
                                      d_bQuadAtomIdsAllAtoms,
                                      d_bCellNonTrivialAtomIds,
                                      d_bCellNonTrivialAtomIdsBins,
                                      d_bCellNonTrivialAtomImageIds,
                                      d_bCellNonTrivialAtomImageIdsBins,
                                      d_smearedChargeWidths,
                                      d_smearedChargeScaling,
                                      d_smearedChargeQuadratureIdElectro);

  //
  // solve for potential corresponding to initial electron-density
  //
  phiTotalSolverProblem.reinit(
    d_matrixFreeDataPRefined,
    d_phiTotRhoIn,
    *d_constraintsVectorElectro[d_phiTotDofHandlerIndexElectro],
    d_phiTotDofHandlerIndexElectro,
    d_phiTotAXQuadratureIdElectro,
    d_densityQuadratureIdElectro,
    d_atomNodeIdToChargeMap,
    d_bQuadValuesAllAtoms,
    d_smearedChargeQuadratureIdElectro,
    *rhoInValues,
    true,
    dftParameters::periodicX && dftParameters::periodicY &&
      dftParameters::periodicZ && !dftParameters::pinnedNodeForPBC,
    dftParameters::smearedNuclearCharges);

  std::map<dealii::CellId, std::vector<double>> phiInValues;

  dealiiCGSolver.solve(phiTotalSolverProblem,
                       dftParameters::absLinearSolverTolerance,
                       dftParameters::maxLinearSolverIterations,
                       dftParameters::verbosity);

  std::map<dealii::CellId, std::vector<double>> dummy;
  interpolateRhoNodalDataToQuadratureDataGeneral(d_matrixFreeDataPRefined,
                                                 d_phiTotDofHandlerIndexElectro,
                                                 d_densityQuadratureIdElectro,
                                                 d_phiTotRhoIn,
                                                 phiInValues,
                                                 dummy,
                                                 dummy);

  //
  // create kohnShamDFTOperatorClass object
  //
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro> kohnShamDFTEigenOperator(
    this, mpi_communicator);
  kohnShamDFTEigenOperator.init();

  //
  // precompute shapeFunctions and shapeFunctionGradients and
  // shapeFunctionGradientIntegrals
  //
  kohnShamDFTEigenOperator.preComputeShapeFunctionGradientIntegrals(
    d_lpspQuadratureId);

  //
  // compute Veff
  //
  if (dftParameters::xcFamilyType == "LDA")
    {
      kohnShamDFTEigenOperator.computeVEff(
        rhoInValues, phiInValues, d_pseudoVLoc, d_rhoCore, d_lpspQuadratureId);
    }
  else if (dftParameters::xcFamilyType == "GGA")
    {
      kohnShamDFTEigenOperator.computeVEff(rhoInValues,
                                           gradRhoInValues,
                                           phiInValues,
                                           d_pseudoVLoc,
                                           d_rhoCore,
                                           d_gradRhoCore,
                                           d_lpspQuadratureId);
    }

  //
  // compute Hamiltonian matrix
  //
  kohnShamDFTEigenOperator.computeHamiltonianMatrix(0, 0);

  //
  // scale the eigenVectors (initial guess of single atom wavefunctions or
  // previous guess) to convert into Lowden Orthonormalized FE basis multiply by
  // M^{1/2}
  internal::pointWiseScaleWithDiagonal(
    kohnShamDFTEigenOperator.d_sqrtMassVector,
    matrix_free_data.get_vector_partitioner(),
    d_numEigenValues,
    d_eigenVectorsFlattenedSTL[0]);


  //
  // compute projected Hamiltonian
  //
  std::vector<dataTypes::number> ProjHam;

  kohnShamDFTEigenOperator.XtHX(d_eigenVectorsFlattenedSTL[0],
                                d_numEigenValues,
                                ProjHam);

  //
  // scale the eigenVectors with M^{-1/2} to represent the wavefunctions in the
  // usual FE basis
  //
  internal::pointWiseScaleWithDiagonal(
    kohnShamDFTEigenOperator.d_invSqrtMassVector,
    matrix_free_data.get_vector_partitioner(),
    d_numEigenValues,
    d_eigenVectorsFlattenedSTL[0]);


  dataTypes::number trXtHX = 0.0;
  for (unsigned int i = 0; i < numberWaveFunctionsEstimate; ++i)
    {
      trXtHX += ProjHam[d_numEigenValues * i + i];
    }

  return trXtHX;
}

template <unsigned int FEOrder, unsigned int FEOrderElectro>
double
dftClass<FEOrder, FEOrderElectro>::computeTraceXtKX(
  unsigned int numberWaveFunctionsEstimate)
{
  //
  // create kohnShamDFTOperatorClass object
  //
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro> kohnShamDFTEigenOperator(
    this, mpi_communicator);
  kohnShamDFTEigenOperator.init();

  //
  // precompute shapeFunctions and shapeFunctionGradients and
  // shapeFunctionGradientIntegrals
  //
  kohnShamDFTEigenOperator.preComputeShapeFunctionGradientIntegrals(
    d_lpspQuadratureId);


  //
  // compute Hamiltonian matrix
  //
  kohnShamDFTEigenOperator.computeKineticMatrix();

  //
  // scale the eigenVectors (initial guess of single atom wavefunctions or
  // previous guess) to convert into Lowden Orthonormalized FE basis multiply by
  // M^{1/2}
  internal::pointWiseScaleWithDiagonal(
    kohnShamDFTEigenOperator.d_sqrtMassVector,
    matrix_free_data.get_vector_partitioner(),
    d_numEigenValues,
    d_eigenVectorsFlattenedSTL[0]);


  //
  // orthogonalize the vectors
  //
  linearAlgebraOperations::gramSchmidtOrthogonalization(
    d_eigenVectorsFlattenedSTL[0], d_numEigenValues, mpi_communicator);

  //
  // compute projected Hamiltonian
  //
  std::vector<dataTypes::number> ProjHam;

  kohnShamDFTEigenOperator.XtHX(d_eigenVectorsFlattenedSTL[0],
                                d_numEigenValues,
                                ProjHam);

  //
  // scale the eigenVectors with M^{-1/2} to represent the wavefunctions in the
  // usual FE basis
  //
  internal::pointWiseScaleWithDiagonal(
    kohnShamDFTEigenOperator.d_invSqrtMassVector,
    matrix_free_data.get_vector_partitioner(),
    d_numEigenValues,
    d_eigenVectorsFlattenedSTL[0]);

  double trXtKX = 0.0;
#ifdef USE_COMPLEX
  trXtKX = 0.0;
#else
  for (unsigned int i = 0; i < numberWaveFunctionsEstimate; ++i)
    {
      trXtKX += ProjHam[d_numEigenValues * i + i];
    }
#endif

  return trXtKX;
}


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::solveNoSCF()
{
  //
  // create kohnShamDFTOperatorClass object
  //
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro> kohnShamDFTEigenOperator(
    this, mpi_communicator);
  kohnShamDFTEigenOperator.init();

  for (unsigned int spinType = 0; spinType < (1 + dftParameters::spinPolarized);
       ++spinType)
    {
      //
      // scale the eigenVectors (initial guess of single atom wavefunctions or
      // previous guess) to convert into Lowden Orthonormalized FE basis
      // multiply by M^{1/2}
      for (unsigned int kPointIndex = 0; kPointIndex < d_kPointWeights.size();
           ++kPointIndex)
        {
          internal::pointWiseScaleWithDiagonal(
            kohnShamDFTEigenOperator.d_sqrtMassVector,
            matrix_free_data.get_vector_partitioner(),
            d_numEigenValues,
            d_eigenVectorsFlattenedSTL[(1 + dftParameters::spinPolarized) *
                                         kPointIndex +
                                       spinType]);
        }


      if (dftParameters::verbosity >= 2)
        pcout
          << "Re-orthonormalizing before solving for ground-state after Gaussian Movement of Mesh "
          << std::endl;
      //
      // orthogonalize the vectors
      //
      for (unsigned int kPointIndex = 0; kPointIndex < d_kPointWeights.size();
           ++kPointIndex)
        {
          const unsigned int flag =
            linearAlgebraOperations::pseudoGramSchmidtOrthogonalization(
              d_elpaScala,
              d_eigenVectorsFlattenedSTL[(1 + dftParameters::spinPolarized) *
                                           kPointIndex +
                                         spinType],
              d_numEigenValues,
              interBandGroupComm,
              mpi_communicator,
              false);
        }


      //
      // scale the eigenVectors with M^{-1/2} to represent the wavefunctions in
      // the usual FE basis
      //
      for (unsigned int kPointIndex = 0; kPointIndex < d_kPointWeights.size();
           ++kPointIndex)
        {
          internal::pointWiseScaleWithDiagonal(
            kohnShamDFTEigenOperator.d_invSqrtMassVector,
            matrix_free_data.get_vector_partitioner(),
            d_numEigenValues,
            d_eigenVectorsFlattenedSTL[(1 + dftParameters::spinPolarized) *
                                         kPointIndex +
                                       spinType]);
        }
    }

  computeRhoFromPSICPU(
    d_eigenVectorsFlattenedSTL,
    d_eigenVectorsRotFracDensityFlattenedSTL,
    d_numEigenValues,
    d_numEigenValuesRR,
    d_eigenVectorsFlattenedSTL[0].size() / d_numEigenValues,
    eigenValues,
    fermiEnergy,
    fermiEnergyUp,
    fermiEnergyDown,
    kohnShamDFTEigenOperator,
    dofHandler,
    matrix_free_data.n_physical_cells(),
    matrix_free_data.get_dofs_per_cell(d_densityDofHandlerIndex),
    matrix_free_data.get_quadrature(d_densityQuadratureId).size(),
    d_kPointWeights,
    rhoOutValues,
    gradRhoOutValues,
    rhoOutValuesSpinPolarized,
    gradRhoOutValuesSpinPolarized,
    dftParameters::xcFamilyType == "GGA",
    interpoolcomm,
    interBandGroupComm,
    false,
    false);
}

// chebyshev solver
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::kohnShamEigenSpaceCompute(
  const unsigned int                                 spinType,
  const unsigned int                                 kPointIndex,
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro> &kohnShamDFTEigenOperator,
  elpaScalaManager &                                 elpaScala,
  chebyshevOrthogonalizedSubspaceIterationSolver &   subspaceIterationSolver,
  std::vector<double> &                              residualNormWaveFunctions,
  const bool                                         computeResidual,
  const bool                                         isSpectrumSplit,
  const bool                                         useMixedPrec,
  const bool                                         isFirstScf)
{
  computing_timer.enter_section("Chebyshev solve");

  if (dftParameters::verbosity >= 2)
    {
      pcout << "kPoint: " << kPointIndex << std::endl;
      if (dftParameters::spinPolarized == 1)
        pcout << "spin: " << spinType + 1 << std::endl;
    }


  //
  // scale the eigenVectors (initial guess of single atom wavefunctions or
  // previous guess) to convert into Lowden Orthonormalized FE basis multiply by
  // M^{1/2}
  internal::pointWiseScaleWithDiagonal(
    kohnShamDFTEigenOperator.d_sqrtMassVector,
    matrix_free_data.get_vector_partitioner(),
    d_numEigenValues,
    d_eigenVectorsFlattenedSTL[(1 + dftParameters::spinPolarized) *
                                 kPointIndex +
                               spinType]);

  std::vector<double> eigenValuesTemp(isSpectrumSplit ? d_numEigenValuesRR :
                                                        d_numEigenValues,
                                      0.0);

  if (d_isFirstFilteringCall[(1 + dftParameters::spinPolarized) * kPointIndex +
                             spinType])
    {
      distributedCPUVec<dataTypes::number> vecForLanczos;
      kohnShamDFTEigenOperator.reinit(1, vecForLanczos, true);

      computing_timer.enter_section("Lanczos k-step Upper Bound");
      std::pair<double, double> bounds =
        linearAlgebraOperations::lanczosLowerUpperBoundEigenSpectrum(
          kohnShamDFTEigenOperator, vecForLanczos);
      const double upperBoundUnwantedSpectrum = bounds.second;
      const double lowerBoundWantedSpectrum   = bounds.first;
      a0[(1 + dftParameters::spinPolarized) * kPointIndex + spinType] =
        lowerBoundWantedSpectrum;
      computing_timer.exit_section("Lanczos k-step Upper Bound");

      d_upperBoundUnwantedSpectrumValues[(1 + dftParameters::spinPolarized) *
                                           kPointIndex +
                                         spinType] = upperBoundUnwantedSpectrum;

      subspaceIterationSolver.reinitSpectrumBounds(
        lowerBoundWantedSpectrum,
        lowerBoundWantedSpectrum +
          (upperBoundUnwantedSpectrum - lowerBoundWantedSpectrum) /
            vecForLanczos.size() * d_numEigenValues *
            (dftParameters::reproducible_output ? 10.0 : 200.0),
        upperBoundUnwantedSpectrum);
    }
  else
    {
      if (!dftParameters::reuseLanczosUpperBoundFromFirstCall)
        {
          computing_timer.enter_section("Lanczos k-step Upper Bound");
          distributedCPUVec<dataTypes::number> vecForLanczos;
          kohnShamDFTEigenOperator.reinit(1, vecForLanczos, true);
          std::pair<double, double> bounds =
            linearAlgebraOperations::lanczosLowerUpperBoundEigenSpectrum(
              kohnShamDFTEigenOperator, vecForLanczos);
          d_upperBoundUnwantedSpectrumValues
            [(1 + dftParameters::spinPolarized) * kPointIndex + spinType] =
              bounds.second;
          computing_timer.exit_section("Lanczos k-step Upper Bound");
        }

      subspaceIterationSolver.reinitSpectrumBounds(
        a0[(1 + dftParameters::spinPolarized) * kPointIndex + spinType],
        bLow[(1 + dftParameters::spinPolarized) * kPointIndex + spinType],
        d_upperBoundUnwantedSpectrumValues[(1 + dftParameters::spinPolarized) *
                                             kPointIndex +
                                           spinType]);
    }

  subspaceIterationSolver.solve(
    kohnShamDFTEigenOperator,
    elpaScala,
    d_eigenVectorsFlattenedSTL[(1 + dftParameters::spinPolarized) *
                                 kPointIndex +
                               spinType],
    d_eigenVectorsRotFracDensityFlattenedSTL
      [(1 + dftParameters::spinPolarized) * kPointIndex + spinType],
    d_tempEigenVec,
    d_numEigenValues,
    eigenValuesTemp,
    residualNormWaveFunctions,
    interBandGroupComm,
    computeResidual,
    useMixedPrec,
    isFirstScf);

  //
  // scale the eigenVectors with M^{-1/2} to represent the wavefunctions in the
  // usual FE basis
  //
  internal::pointWiseScaleWithDiagonal(
    kohnShamDFTEigenOperator.d_invSqrtMassVector,
    matrix_free_data.get_vector_partitioner(),
    d_numEigenValues,
    d_eigenVectorsFlattenedSTL[(1 + dftParameters::spinPolarized) *
                                 kPointIndex +
                               spinType]);

  if (isSpectrumSplit && d_numEigenValuesRR != d_numEigenValues)
    {
      internal::pointWiseScaleWithDiagonal(
        kohnShamDFTEigenOperator.d_invSqrtMassVector,
        matrix_free_data.get_vector_partitioner(),
        d_numEigenValuesRR,
        d_eigenVectorsRotFracDensityFlattenedSTL
          [(1 + dftParameters::spinPolarized) * kPointIndex + spinType]);
    }

  //
  // copy the eigenValues and corresponding residual norms back to data members
  //
  if (isSpectrumSplit)
    {
      for (unsigned int i = 0; i < d_numEigenValuesRR; i++)
        {
          if (dftParameters::verbosity >= 4 &&
              d_numEigenValues == d_numEigenValuesRR)
            pcout << "eigen value " << std::setw(3) << i << ": "
                  << eigenValuesTemp[i] << std::endl;
          else if (dftParameters::verbosity >= 4 &&
                   d_numEigenValues != d_numEigenValuesRR)
            pcout << "valence eigen value " << std::setw(3) << i << ": "
                  << eigenValuesTemp[i] << std::endl;

          eigenValuesRRSplit[kPointIndex][spinType * d_numEigenValuesRR + i] =
            eigenValuesTemp[i];
        }

      for (unsigned int i = 0; i < d_numEigenValues; i++)
        {
          if (i >= (d_numEigenValues - d_numEigenValuesRR))
            eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
              eigenValuesTemp[i - (d_numEigenValues - d_numEigenValuesRR)];
          else
            eigenValues[kPointIndex][spinType * d_numEigenValues + i] = -100.0;
        }
    }
  else
    {
      for (unsigned int i = 0; i < d_numEigenValues; i++)
        {
          if (dftParameters::verbosity >= 4)
            pcout << "eigen value " << std::setw(3) << i << ": "
                  << eigenValuesTemp[i] << std::endl;

          eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
            eigenValuesTemp[i];
        }
    }

  if (dftParameters::verbosity >= 4)
    pcout << std::endl;


  // set a0 and bLow
  /* a0[(1+dftParameters::spinPolarized)*kPointIndex+spinType]=isSpectrumSplit?
     dftParameters::lowerEndWantedSpectrum
     :eigenValuesTemp[0];*/


  bLow[(1 + dftParameters::spinPolarized) * kPointIndex + spinType] =
    eigenValuesTemp.back();
  d_isFirstFilteringCall[(1 + dftParameters::spinPolarized) * kPointIndex +
                         spinType] = false;

  if (!isSpectrumSplit)
    {
      a0[(1 + dftParameters::spinPolarized) * kPointIndex + spinType] =
        eigenValuesTemp[0];
    }

  computing_timer.exit_section("Chebyshev solve");
}

#ifdef DFTFE_WITH_GPU
// chebyshev solver
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::kohnShamEigenSpaceCompute(
  const unsigned int spinType,
  const unsigned int kPointIndex,
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>
    &               kohnShamDFTEigenOperator,
  elpaScalaManager &elpaScala,
  chebyshevOrthogonalizedSubspaceIterationSolverCUDA
    &                  subspaceIterationSolverCUDA,
  std::vector<double> &residualNormWaveFunctions,
  const bool           computeResidual,
  const unsigned int   numberRayleighRitzAvoidanceXLBOMDPasses,
  const bool           isSpectrumSplit,
  const bool           useMixedPrec,
  const bool           isFirstScf)
{
  computing_timer.enter_section("Chebyshev solve CUDA");

  if (dftParameters::verbosity >= 2)
    {
      pcout << "kPoint: " << kPointIndex << std::endl;
      if (dftParameters::spinPolarized == 1)
        pcout << "spin: " << spinType + 1 << std::endl;
    }

  std::vector<double> eigenValuesTemp(isSpectrumSplit ? d_numEigenValuesRR :
                                                        d_numEigenValues,
                                      0.0);
  std::vector<double> eigenValuesDummy(isSpectrumSplit ? d_numEigenValuesRR :
                                                         d_numEigenValues,
                                       0.0);

  subspaceIterationSolverCUDA.reinitSpectrumBounds(
    a0[(1 + dftParameters::spinPolarized) * kPointIndex + spinType],
    bLow[(1 + dftParameters::spinPolarized) * kPointIndex + spinType],
    d_upperBoundUnwantedSpectrumValues[(1 + dftParameters::spinPolarized) *
                                         kPointIndex +
                                       spinType]);

  const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();
  std::shared_ptr<const dftfe::ProcessGrid> processGrid =
    elpaScala.getProcessGridDftfeScalaWrapper();

  dftfe::ScaLAPACKMatrix<double> projHamPar(d_numEigenValues,
                                            processGrid,
                                            rowsBlockSize);


  dftfe::ScaLAPACKMatrix<double> overlapMatPar(d_numEigenValues,
                                               processGrid,
                                               rowsBlockSize);


  if (numberRayleighRitzAvoidanceXLBOMDPasses > 0)
    {
      subspaceIterationSolverCUDA.solveNoRR(
        kohnShamDFTEigenOperator,
        d_eigenVectorsFlattenedCUDA.begin() +
          ((1 + dftParameters::spinPolarized) * kPointIndex + spinType) *
            d_eigenVectorsFlattenedSTL[0].size(),
        d_eigenVectorsFlattenedSTL[0].size(),
        d_tempEigenVec,
        d_numEigenValues,
        eigenValuesDummy,
        *d_gpucclMpiCommDomainPtr,
        interBandGroupComm,
        projHamPar,
        overlapMatPar,
        processGrid,
        numberRayleighRitzAvoidanceXLBOMDPasses,
        useMixedPrec);
    }
  else
    {
#  ifdef DFTFE_WITH_ELPA
      if (dftParameters::useELPA)
        {
          d_upperBoundUnwantedSpectrumValues
            [(1 + dftParameters::spinPolarized) * kPointIndex + spinType] =
              subspaceIterationSolverCUDA.solve(
                kohnShamDFTEigenOperator,
                d_eigenVectorsFlattenedCUDA.begin() +
                  ((1 + dftParameters::spinPolarized) * kPointIndex +
                   spinType) *
                    d_eigenVectorsFlattenedSTL[0].size(),
                d_eigenVectorsRotFracFlattenedCUDA.begin() +
                  ((1 + dftParameters::spinPolarized) * kPointIndex +
                   spinType) *
                    d_eigenVectorsRotFracDensityFlattenedSTL[0].size(),
                d_eigenVectorsFlattenedSTL[0].size(),
                d_tempEigenVec,
                d_numEigenValues,
                eigenValuesDummy,
                residualNormWaveFunctions,
                *d_gpucclMpiCommDomainPtr,
                interBandGroupComm,
                projHamPar,
                overlapMatPar,
                processGrid,
                d_isFirstFilteringCall[(1 + dftParameters::spinPolarized) *
                                         kPointIndex +
                                       spinType],
                computeResidual,
                useMixedPrec,
                isFirstScf,
                true,
                false);
          MPI_Barrier(MPI_COMM_WORLD);
          double time = MPI_Wtime();

          if (isSpectrumSplit && d_numEigenValuesRR != d_numEigenValues)
            {
              linearAlgebraOperations::elpaPartialDiagonalizationGEP(
                elpaScala,
                d_numEigenValues,
                d_numEigenValues - d_numEigenValuesRR,
                elpaScala.getMPICommunicator(),
                eigenValuesTemp,
                projHamPar,
                overlapMatPar,
                processGrid);
            }
          else
            {
              linearAlgebraOperations::elpaDiagonalizationGEP(
                elpaScala,
                d_numEigenValues,
                elpaScala.getMPICommunicator(),
                eigenValuesTemp,
                projHamPar,
                overlapMatPar,
                processGrid);
            }

          MPI_Barrier(MPI_COMM_WORLD);
          time = MPI_Wtime() - time;
          if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
              dftParameters::verbosity >= 2)
            if (isSpectrumSplit && d_numEigenValuesRR != d_numEigenValues)
              std::cout << "Time for ELPA partial eigen decomp, RR step: "
                        << time << std::endl;
            else
              std::cout << "Time for ELPA eigen decomp, RR step: " << time
                        << std::endl;


          d_upperBoundUnwantedSpectrumValues
            [(1 + dftParameters::spinPolarized) * kPointIndex + spinType] =
              subspaceIterationSolverCUDA.solve(
                kohnShamDFTEigenOperator,
                d_eigenVectorsFlattenedCUDA.begin() +
                  ((1 + dftParameters::spinPolarized) * kPointIndex +
                   spinType) *
                    d_eigenVectorsFlattenedSTL[0].size(),
                d_eigenVectorsRotFracFlattenedCUDA.begin() +
                  ((1 + dftParameters::spinPolarized) * kPointIndex +
                   spinType) *
                    d_eigenVectorsRotFracDensityFlattenedSTL[0].size(),
                d_eigenVectorsFlattenedSTL[0].size(),
                d_tempEigenVec,
                d_numEigenValues,
                eigenValuesTemp,
                residualNormWaveFunctions,
                *d_gpucclMpiCommDomainPtr,
                interBandGroupComm,
                projHamPar,
                overlapMatPar,
                processGrid,
                d_isFirstFilteringCall[(1 + dftParameters::spinPolarized) *
                                         kPointIndex +
                                       spinType],
                computeResidual,
                useMixedPrec,
                isFirstScf,
                false,
                true);
        }
      else
        {
          d_upperBoundUnwantedSpectrumValues
            [(1 + dftParameters::spinPolarized) * kPointIndex + spinType] =
              subspaceIterationSolverCUDA.solve(
                kohnShamDFTEigenOperator,
                d_eigenVectorsFlattenedCUDA.begin() +
                  ((1 + dftParameters::spinPolarized) * kPointIndex +
                   spinType) *
                    d_eigenVectorsFlattenedSTL[0].size(),
                d_eigenVectorsRotFracFlattenedCUDA.begin() +
                  ((1 + dftParameters::spinPolarized) * kPointIndex +
                   spinType) *
                    d_eigenVectorsRotFracDensityFlattenedSTL[0].size(),
                d_eigenVectorsFlattenedSTL[0].size(),
                d_tempEigenVec,
                d_numEigenValues,
                eigenValuesTemp,
                residualNormWaveFunctions,
                *d_gpucclMpiCommDomainPtr,
                interBandGroupComm,
                projHamPar,
                overlapMatPar,
                processGrid,
                d_isFirstFilteringCall[(1 + dftParameters::spinPolarized) *
                                         kPointIndex +
                                       spinType],
                computeResidual,
                useMixedPrec,
                isFirstScf);
        }
#  else
      d_upperBoundUnwantedSpectrumValues[(1 + dftParameters::spinPolarized) *
                                           kPointIndex +
                                         spinType] =
        subspaceIterationSolverCUDA.solve(
          kohnShamDFTEigenOperator,
          d_eigenVectorsFlattenedCUDA.begin() +
            ((1 + dftParameters::spinPolarized) * kPointIndex + spinType) *
              d_eigenVectorsFlattenedSTL[0].size(),
          d_eigenVectorsRotFracFlattenedCUDA.begin() +
            ((1 + dftParameters::spinPolarized) * kPointIndex + spinType) *
              d_eigenVectorsRotFracDensityFlattenedSTL[0].size(),
          d_eigenVectorsFlattenedSTL[0].size(),
          d_tempEigenVec,
          d_numEigenValues,
          eigenValuesTemp,
          residualNormWaveFunctions,
          *d_gpucclMpiCommDomainPtr,
          interBandGroupComm,
          projHamPar,
          overlapMatPar,
          processGrid,
          d_isFirstFilteringCall[(1 + dftParameters::spinPolarized) *
                                   kPointIndex +
                                 spinType],
          computeResidual,
          useMixedPrec,
          isFirstScf);
#  endif


      //
      // copy the eigenValues and corresponding residual norms back to data
      // members
      //
      if (isSpectrumSplit)
        {
          for (unsigned int i = 0; i < d_numEigenValuesRR; i++)
            {
              if (dftParameters::verbosity >= 5 &&
                  d_numEigenValues == d_numEigenValuesRR)
                pcout << "eigen value " << std::setw(3) << i << ": "
                      << eigenValuesTemp[i] << std::endl;
              else if (dftParameters::verbosity >= 5 &&
                       d_numEigenValues != d_numEigenValuesRR)
                pcout << "valence eigen value " << std::setw(3) << i << ": "
                      << eigenValuesTemp[i] << std::endl;

              eigenValuesRRSplit[kPointIndex][spinType * d_numEigenValuesRR +
                                              i] = eigenValuesTemp[i];
            }

          for (unsigned int i = 0; i < d_numEigenValues; i++)
            {
              if (i >= (d_numEigenValues - d_numEigenValuesRR))
                eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
                  eigenValuesTemp[i - (d_numEigenValues - d_numEigenValuesRR)];
              else
                eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
                  -100.0;
            }
        }
      else
        {
          for (unsigned int i = 0; i < d_numEigenValues; i++)
            {
              if (dftParameters::verbosity >= 5)
                pcout << "eigen value " << std::setw(3) << i << ": "
                      << eigenValuesTemp[i] << std::endl;

              eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
                eigenValuesTemp[i];
            }
        }

      if (dftParameters::verbosity >= 4)
        pcout << std::endl;


      bLow[(1 + dftParameters::spinPolarized) * kPointIndex + spinType] =
        eigenValuesTemp.back();
      d_isFirstFilteringCall[(1 + dftParameters::spinPolarized) * kPointIndex +
                             spinType] = false;
      if (!isSpectrumSplit)
        {
          a0[(1 + dftParameters::spinPolarized) * kPointIndex + spinType] =
            eigenValuesTemp[0];
        }
    }
  computing_timer.exit_section("Chebyshev solve CUDA");
}
#endif


#ifdef DFTFE_WITH_GPU
// chebyshev solver
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::kohnShamEigenSpaceOnlyRRCompute(
  const unsigned int spinType,
  const unsigned int kPointIndex,
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>
    &               kohnShamDFTEigenOperator,
  elpaScalaManager &elpaScala,
  chebyshevOrthogonalizedSubspaceIterationSolverCUDA
    &        subspaceIterationSolverCUDA,
  const bool isSpectrumSplit,
  const bool useMixedPrec)
{
  if (dftParameters::verbosity >= 2)
    {
      pcout << "kPoint: " << kPointIndex << std::endl;
      if (dftParameters::spinPolarized == 1)
        pcout << "spin: " << spinType + 1 << std::endl;
    }

  std::vector<double> eigenValuesTemp(isSpectrumSplit ? d_numEigenValuesRR :
                                                        d_numEigenValues,
                                      0.0);
  std::vector<double> eigenValuesDummy(isSpectrumSplit ? d_numEigenValuesRR :
                                                         d_numEigenValues,
                                       0.0);


  const unsigned int rowsBlockSize = elpaScala.getScalapackBlockSize();
  std::shared_ptr<const dftfe::ProcessGrid> processGrid =
    elpaScala.getProcessGridDftfeScalaWrapper();

  dftfe::ScaLAPACKMatrix<double> projHamPar(d_numEigenValues,
                                            processGrid,
                                            rowsBlockSize);


  dftfe::ScaLAPACKMatrix<double> overlapMatPar(d_numEigenValues,
                                               processGrid,
                                               rowsBlockSize);


#  ifdef DFTFE_WITH_ELPA
  if (dftParameters::useELPA)
    {
      subspaceIterationSolverCUDA.onlyRR(
        kohnShamDFTEigenOperator,
        d_eigenVectorsFlattenedCUDA.begin() +
          ((1 + dftParameters::spinPolarized) * kPointIndex + spinType) *
            d_eigenVectorsFlattenedSTL[0].size(),
        d_eigenVectorsRotFracFlattenedCUDA.begin() +
          ((1 + dftParameters::spinPolarized) * kPointIndex + spinType) *
            d_eigenVectorsRotFracDensityFlattenedSTL[0].size(),
        d_eigenVectorsFlattenedSTL[0].size(),
        d_tempEigenVec,
        d_numEigenValues,
        eigenValuesDummy,
        *d_gpucclMpiCommDomainPtr,
        interBandGroupComm,
        projHamPar,
        overlapMatPar,
        processGrid,
        useMixedPrec,
        true,
        false);
      MPI_Barrier(MPI_COMM_WORLD);
      double time = MPI_Wtime();

      if (isSpectrumSplit && d_numEigenValuesRR != d_numEigenValues)
        {
          linearAlgebraOperations::elpaPartialDiagonalization(
            elpaScala,
            d_numEigenValues,
            d_numEigenValues - d_numEigenValuesRR,
            elpaScala.getMPICommunicator(),
            eigenValuesTemp,
            projHamPar,
            processGrid);
        }
      else
        {
          linearAlgebraOperations::elpaDiagonalization(
            elpaScala,
            d_numEigenValues,
            elpaScala.getMPICommunicator(),
            eigenValuesTemp,
            projHamPar,
            processGrid);
        }

      MPI_Barrier(MPI_COMM_WORLD);
      time = MPI_Wtime() - time;
      if (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 &&
          dftParameters::verbosity >= 2)
        if (isSpectrumSplit && d_numEigenValuesRR != d_numEigenValues)
          std::cout << "Time for ELPA partial eigen decomp, RR step: " << time
                    << std::endl;
        else
          std::cout << "Time for ELPA eigen decomp, RR step: " << time
                    << std::endl;


      subspaceIterationSolverCUDA.onlyRR(
        kohnShamDFTEigenOperator,
        d_eigenVectorsFlattenedCUDA.begin() +
          ((1 + dftParameters::spinPolarized) * kPointIndex + spinType) *
            d_eigenVectorsFlattenedSTL[0].size(),
        d_eigenVectorsRotFracFlattenedCUDA.begin() +
          ((1 + dftParameters::spinPolarized) * kPointIndex + spinType) *
            d_eigenVectorsRotFracDensityFlattenedSTL[0].size(),
        d_eigenVectorsFlattenedSTL[0].size(),
        d_tempEigenVec,
        d_numEigenValues,
        eigenValuesTemp,
        *d_gpucclMpiCommDomainPtr,
        interBandGroupComm,
        projHamPar,
        overlapMatPar,
        processGrid,
        useMixedPrec,
        false,
        true);
    }
  else
    {
      subspaceIterationSolverCUDA.onlyRR(
        kohnShamDFTEigenOperator,
        d_eigenVectorsFlattenedCUDA.begin() +
          ((1 + dftParameters::spinPolarized) * kPointIndex + spinType) *
            d_eigenVectorsFlattenedSTL[0].size(),
        d_eigenVectorsRotFracFlattenedCUDA.begin() +
          ((1 + dftParameters::spinPolarized) * kPointIndex + spinType) *
            d_eigenVectorsRotFracDensityFlattenedSTL[0].size(),
        d_eigenVectorsFlattenedSTL[0].size(),
        d_tempEigenVec,
        d_numEigenValues,
        eigenValuesTemp,
        *d_gpucclMpiCommDomainPtr,
        interBandGroupComm,
        projHamPar,
        overlapMatPar,
        processGrid,
        useMixedPrec);
    }
#  else
  subspaceIterationSolverCUDA.onlyRR(
    kohnShamDFTEigenOperator,
    d_eigenVectorsFlattenedCUDA.begin() +
      ((1 + dftParameters::spinPolarized) * kPointIndex + spinType) *
        d_eigenVectorsFlattenedSTL[0].size(),
    d_eigenVectorsRotFracFlattenedCUDA.begin() +
      ((1 + dftParameters::spinPolarized) * kPointIndex + spinType) *
        d_eigenVectorsRotFracDensityFlattenedSTL[0].size(),
    d_eigenVectorsFlattenedSTL[0].size(),
    d_tempEigenVec,
    d_numEigenValues,
    eigenValuesTemp,
    *d_gpucclMpiCommDomainPtr,
    interBandGroupComm,
    projHamPar,
    overlapMatPar,
    processGrid,
    useMixedPrec);
#  endif


  //
  // copy the eigenValues and corresponding residual norms back to data members
  //
  if (isSpectrumSplit)
    {
      for (unsigned int i = 0; i < d_numEigenValuesRR; i++)
        {
          if (dftParameters::verbosity >= 5 &&
              d_numEigenValues == d_numEigenValuesRR)
            pcout << "eigen value " << std::setw(3) << i << ": "
                  << eigenValuesTemp[i] << std::endl;
          else if (dftParameters::verbosity >= 5 &&
                   d_numEigenValues != d_numEigenValuesRR)
            pcout << "valence eigen value " << std::setw(3) << i << ": "
                  << eigenValuesTemp[i] << std::endl;

          eigenValuesRRSplit[kPointIndex][spinType * d_numEigenValuesRR + i] =
            eigenValuesTemp[i];
        }

      for (unsigned int i = 0; i < d_numEigenValues; i++)
        {
          if (i >= (d_numEigenValues - d_numEigenValuesRR))
            eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
              eigenValuesTemp[i - (d_numEigenValues - d_numEigenValuesRR)];
          else
            eigenValues[kPointIndex][spinType * d_numEigenValues + i] = -100.0;
        }
    }
  else
    {
      for (unsigned int i = 0; i < d_numEigenValues; i++)
        {
          if (dftParameters::verbosity >= 5)
            pcout << "eigen value " << std::setw(3) << i << ": "
                  << eigenValuesTemp[i] << std::endl;

          eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
            eigenValuesTemp[i];
        }
    }

  if (dftParameters::verbosity >= 4)
    pcout << std::endl;
}
#endif


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::kohnShamEigenSpaceOnlyRRCompute(
  const unsigned int                                 spinType,
  const unsigned int                                 kPointIndex,
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro> &kohnShamDFTEigenOperator,
  elpaScalaManager &                                 elpaScala,
  chebyshevOrthogonalizedSubspaceIterationSolver &   subspaceIterationSolver,
  const bool                                         isSpectrumSplit,
  const bool                                         useMixedPrec)
{
  if (dftParameters::verbosity >= 2)
    {
      pcout << "kPoint: " << kPointIndex << std::endl;
      if (dftParameters::spinPolarized == 1)
        pcout << "spin: " << spinType + 1 << std::endl;
    }


  //
  // scale the eigenVectors to convert into Lowden Orthonormalized FE basis
  // multiply by M^{1/2}
  internal::pointWiseScaleWithDiagonal(
    kohnShamDFTEigenOperator.d_sqrtMassVector,
    matrix_free_data.get_vector_partitioner(),
    d_numEigenValues,
    d_eigenVectorsFlattenedSTL[(1 + dftParameters::spinPolarized) *
                                 kPointIndex +
                               spinType]);

  std::vector<double> eigenValuesTemp(isSpectrumSplit ? d_numEigenValuesRR :
                                                        d_numEigenValues,
                                      0.0);


  subspaceIterationSolver.onlyRR(
    kohnShamDFTEigenOperator,
    elpaScala,
    d_eigenVectorsFlattenedSTL[(1 + dftParameters::spinPolarized) *
                                 kPointIndex +
                               spinType],
    d_eigenVectorsRotFracDensityFlattenedSTL
      [(1 + dftParameters::spinPolarized) * kPointIndex + spinType],
    d_tempEigenVec,
    d_numEigenValues,
    eigenValuesTemp,
    interBandGroupComm,
    useMixedPrec);

  //
  // scale the eigenVectors with M^{-1/2} to represent the wavefunctions in the
  // usual FE basis
  //
  internal::pointWiseScaleWithDiagonal(
    kohnShamDFTEigenOperator.d_invSqrtMassVector,
    matrix_free_data.get_vector_partitioner(),
    d_numEigenValues,
    d_eigenVectorsFlattenedSTL[(1 + dftParameters::spinPolarized) *
                                 kPointIndex +
                               spinType]);

  if (isSpectrumSplit && d_numEigenValuesRR != d_numEigenValues)
    {
      internal::pointWiseScaleWithDiagonal(
        kohnShamDFTEigenOperator.d_invSqrtMassVector,
        matrix_free_data.get_vector_partitioner(),
        d_numEigenValuesRR,
        d_eigenVectorsRotFracDensityFlattenedSTL
          [(1 + dftParameters::spinPolarized) * kPointIndex + spinType]);
    }

  //
  // copy the eigenValues and corresponding residual norms back to data members
  //
  if (isSpectrumSplit)
    {
      for (unsigned int i = 0; i < d_numEigenValuesRR; i++)
        {
          if (dftParameters::verbosity >= 4 &&
              d_numEigenValues == d_numEigenValuesRR)
            pcout << "eigen value " << std::setw(3) << i << ": "
                  << eigenValuesTemp[i] << std::endl;
          else if (dftParameters::verbosity >= 4 &&
                   d_numEigenValues != d_numEigenValuesRR)
            pcout << "valence eigen value " << std::setw(3) << i << ": "
                  << eigenValuesTemp[i] << std::endl;

          eigenValuesRRSplit[kPointIndex][spinType * d_numEigenValuesRR + i] =
            eigenValuesTemp[i];
        }

      for (unsigned int i = 0; i < d_numEigenValues; i++)
        {
          if (i >= (d_numEigenValues - d_numEigenValuesRR))
            eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
              eigenValuesTemp[i - (d_numEigenValues - d_numEigenValuesRR)];
          else
            eigenValues[kPointIndex][spinType * d_numEigenValues + i] = -100.0;
        }
    }
  else
    {
      for (unsigned int i = 0; i < d_numEigenValues; i++)
        {
          if (dftParameters::verbosity >= 4)
            pcout << "eigen value " << std::setw(3) << i << ": "
                  << eigenValuesTemp[i] << std::endl;

          eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
            eigenValuesTemp[i];
        }
    }

  if (dftParameters::verbosity >= 4)
    pcout << std::endl;
}


// chebyshev solver
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::kohnShamEigenSpaceComputeNSCF(
  const unsigned int                                 spinType,
  const unsigned int                                 kPointIndex,
  kohnShamDFTOperatorClass<FEOrder, FEOrderElectro> &kohnShamDFTEigenOperator,
  chebyshevOrthogonalizedSubspaceIterationSolver &   subspaceIterationSolver,
  std::vector<double> &                              residualNormWaveFunctions,
  unsigned int                                       ipass)
{
  computing_timer.enter_section("Chebyshev solve");

  if (dftParameters::verbosity == 2)
    {
      pcout << "kPoint: " << kPointIndex << std::endl;
      pcout << "spin: " << spinType + 1 << std::endl;
    }

  //
  // scale the eigenVectors (initial guess of single atom wavefunctions or
  // previous guess) to convert into Lowden Orthonormalized FE basis multiply by
  // M^{1/2}
  if (ipass == 1)
    internal::pointWiseScaleWithDiagonal(
      kohnShamDFTEigenOperator.d_invSqrtMassVector,
      matrix_free_data.get_vector_partitioner(),
      d_numEigenValues,
      d_eigenVectorsFlattenedSTL[(1 + dftParameters::spinPolarized) *
                                   kPointIndex +
                                 spinType]);


  std::vector<double> eigenValuesTemp(d_numEigenValues, 0.0);

  if (d_isFirstFilteringCall[(1 + dftParameters::spinPolarized) * kPointIndex +
                             spinType])
    {
      distributedCPUVec<dataTypes::number> vecForLanczos;
      kohnShamDFTEigenOperator.reinit(1, vecForLanczos, true);

      computing_timer.enter_section("Lanczos k-step Upper Bound");
      std::pair<double, double> bounds =
        linearAlgebraOperations::lanczosLowerUpperBoundEigenSpectrum(
          kohnShamDFTEigenOperator, vecForLanczos);
      const double upperBoundUnwantedSpectrum = bounds.second;
      const double lowerBoundWantedSpectrum   = bounds.first;
      a0[(1 + dftParameters::spinPolarized) * kPointIndex + spinType] =
        lowerBoundWantedSpectrum;
      computing_timer.exit_section("Lanczos k-step Upper Bound");

      subspaceIterationSolver.reinitSpectrumBounds(
        lowerBoundWantedSpectrum,
        lowerBoundWantedSpectrum +
          (upperBoundUnwantedSpectrum - lowerBoundWantedSpectrum) /
            vecForLanczos.size() * d_numEigenValues *
            (dftParameters::reproducible_output ? 10.0 : 200.0),
        upperBoundUnwantedSpectrum);
    }
  else
    {
      computing_timer.enter_section("Lanczos k-step Upper Bound");
      distributedCPUVec<dataTypes::number> vecForLanczos;
      kohnShamDFTEigenOperator.reinit(1, vecForLanczos, true);
      std::pair<double, double> bounds =
        linearAlgebraOperations::lanczosLowerUpperBoundEigenSpectrum(
          kohnShamDFTEigenOperator, vecForLanczos);
      const double upperBoundUnwantedSpectrum = bounds.second;
      computing_timer.exit_section("Lanczos k-step Upper Bound");

      subspaceIterationSolver.reinitSpectrumBounds(
        a0[(1 + dftParameters::spinPolarized) * kPointIndex + spinType],
        bLow[(1 + dftParameters::spinPolarized) * kPointIndex + spinType],
        upperBoundUnwantedSpectrum);
    }


  subspaceIterationSolver.solve(
    kohnShamDFTEigenOperator,
    d_elpaScala,
    d_eigenVectorsFlattenedSTL[(1 + dftParameters::spinPolarized) *
                                 kPointIndex +
                               spinType],
    d_eigenVectorsFlattenedSTL[(1 + dftParameters::spinPolarized) *
                                 kPointIndex +
                               spinType],
    d_tempEigenVec,
    d_numEigenValues,
    eigenValuesTemp,
    residualNormWaveFunctions,
    interBandGroupComm,
    true,
    false);

  if (dftParameters::verbosity >= 4)
    {
#ifdef USE_PETSC
      PetscLogDouble bytes;
      PetscMemoryGetCurrentUsage(&bytes);
      FILE *       dummy;
      unsigned int this_mpi_process =
        dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
      PetscSynchronizedPrintf(
        mpi_communicator,
        "[%d] Memory after recreating STL vector and exiting from subspaceIteration solver  %e\n",
        this_mpi_process,
        bytes);
      PetscSynchronizedFlush(mpi_communicator, dummy);
#endif
    }



  //
  // copy the eigenValues and corresponding residual norms back to data members
  //
  for (unsigned int i = 0; i < d_numEigenValues; i++)
    {
      // if(dftParameters::verbosity==2)
      //    pcout<<"eigen value "<< std::setw(3) <<i <<": "<<eigenValuesTemp[i]
      //    <<std::endl;

      eigenValues[kPointIndex][spinType * d_numEigenValues + i] =
        eigenValuesTemp[i];
    }

  // if (dftParameters::verbosity==2)
  //   pcout <<std::endl;


  // set a0 and bLow
  a0[(1 + dftParameters::spinPolarized) * kPointIndex + spinType] =
    eigenValuesTemp[0];
  bLow[(1 + dftParameters::spinPolarized) * kPointIndex + spinType] =
    eigenValuesTemp.back();
  d_isFirstFilteringCall[(1 + dftParameters::spinPolarized) * kPointIndex +
                         spinType] = false;
  //


  computing_timer.exit_section("Chebyshev solve");
}


// compute the maximum of the residual norm of the highest occupied state among
// all k points
template <unsigned int FEOrder, unsigned int FEOrderElectro>
double
dftClass<FEOrder, FEOrderElectro>::
  computeMaximumHighestOccupiedStateResidualNorm(
    const std::vector<std::vector<double>> &residualNormWaveFunctionsAllkPoints,
    const std::vector<std::vector<double>> &eigenValuesAllkPoints,
    const double                            fermiEnergy)
{
  double maxHighestOccupiedStateResNorm = -1e+6;
  for (int kPoint = 0; kPoint < eigenValuesAllkPoints.size(); ++kPoint)
    {
      unsigned int highestOccupiedState = 0;

      for (unsigned int i = 0; i < eigenValuesAllkPoints[kPoint].size(); i++)
        {
          const double factor =
            (eigenValuesAllkPoints[kPoint][i] - fermiEnergy) /
            (C_kb * dftParameters::TVal);
          if (factor < 0)
            highestOccupiedState = i;
        }

      if (residualNormWaveFunctionsAllkPoints[kPoint][highestOccupiedState] >
          maxHighestOccupiedStateResNorm)
        {
          maxHighestOccupiedStateResNorm =
            residualNormWaveFunctionsAllkPoints[kPoint][highestOccupiedState];
        }
    }
  maxHighestOccupiedStateResNorm =
    Utilities::MPI::max(maxHighestOccupiedStateResNorm, interpoolcomm);
  return maxHighestOccupiedStateResNorm;
}
