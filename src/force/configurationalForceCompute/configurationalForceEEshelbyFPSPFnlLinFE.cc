// ---------------------------------------------------------------------
//
// Copyright (c) 2017-18 The Regents of the University of Michigan and DFT-FE
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

//
// compute configurational force contribution from all terms except the nuclear
// self energy
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::
  computeConfigurationalForceEEshelbyTensorFPSPFnlLinFE(
    const MatrixFree<3, double> &matrixFreeData,
#ifdef DFTFE_WITH_GPU
    kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>
      &kohnShamDFTEigenOperatorGPU,
#endif
    kohnShamDFTOperatorClass<FEOrder, FEOrderElectro> &kohnShamDFTEigenOperator,
    const unsigned int                                 eigenDofHandlerIndex,
    const unsigned int               smearedChargeQuadratureId,
    const unsigned int               lpspQuadratureIdElectro,
    const MatrixFree<3, double> &    matrixFreeDataElectro,
    const unsigned int               phiTotDofHandlerIndexElectro,
    const distributedCPUVec<double> &phiTotRhoOutElectro,
    const std::map<dealii::CellId, std::vector<double>> &rhoOutValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoOutValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoOutValuesLpsp,
    const std::map<dealii::CellId, std::vector<double>> &rhoOutValuesElectro,
    const std::map<dealii::CellId, std::vector<double>>
      &rhoOutValuesElectroLpsp,
    const std::map<dealii::CellId, std::vector<double>>
      &gradRhoOutValuesElectro,
    const std::map<dealii::CellId, std::vector<double>>
      &gradRhoOutValuesElectroLpsp,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &hessianRhoCoreValues,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &gradRhoCoreAtoms,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &                                                  hessianRhoCoreAtoms,
    const std::map<dealii::CellId, std::vector<double>> &pseudoVLocElectro,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &                                              pseudoVLocAtomsElectro,
    const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro,
    const std::map<dealii::CellId, std::vector<double>> &shadowKSRhoMinValues,
    const std::map<dealii::CellId, std::vector<double>>
      &                              shadowKSGradRhoMinValues,
    const distributedCPUVec<double> &phiRhoMinusApproxRho,
    const bool                       shadowPotentialForce)
{
  int this_process;
  MPI_Comm_rank(d_mpiCommParent, &this_process);
  MPI_Barrier(d_mpiCommParent);
  double forcetotal_time = MPI_Wtime();

  MPI_Barrier(d_mpiCommParent);
  double init_time = MPI_Wtime();

  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  std::map<unsigned int, std::vector<double>> forceContributionFnlGammaAtoms;

  const bool isPseudopotential = d_dftParams.isPseudopotential;

  FEEvaluation<3,
               1,
               C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
               3>
    forceEval(matrixFreeData,
              d_forceDofHandlerIndex,
              dftPtr->d_densityQuadratureId);
  FEEvaluation<3, 1, C_num1DQuadNLPSP<FEOrder>() * C_numCopies1DQuadNLPSP(), 3>
    forceEvalNLP(matrixFreeData,
                 d_forceDofHandlerIndex,
                 dftPtr->d_nlpspQuadratureId);


  std::map<unsigned int, std::vector<double>>
    forceContributionShadowLocalGammaAtoms;

  const unsigned int numQuadPoints    = forceEval.n_q_points;
  const unsigned int numQuadPointsNLP = forceEvalNLP.n_q_points;

  const unsigned int numEigenVectors = dftPtr->d_numEigenValues;
  const unsigned int numKPoints      = dftPtr->d_kPointWeights.size();
  DoFHandler<3>::active_cell_iterator   subCellPtr;
  Tensor<1, 2, VectorizedArray<double>> zeroTensor1;
  zeroTensor1[0] = make_vectorized_array(0.0);
  zeroTensor1[1] = make_vectorized_array(0.0);
  Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>> zeroTensor2;
  Tensor<1, 3, VectorizedArray<double>>               zeroTensor3;
  Tensor<2, 3, VectorizedArray<double>>               zeroTensor4;
  for (unsigned int idim = 0; idim < 3; idim++)
    {
      zeroTensor2[0][idim] = make_vectorized_array(0.0);
      zeroTensor2[1][idim] = make_vectorized_array(0.0);
      zeroTensor3[idim]    = make_vectorized_array(0.0);
    }
  for (unsigned int idim = 0; idim < 3; idim++)
    {
      for (unsigned int jdim = 0; jdim < 3; jdim++)
        {
          zeroTensor4[idim][jdim] = make_vectorized_array(0.0);
        }
    }

  const double spinPolarizedFactor =
    (d_dftParams.spinPolarized == 1) ? 0.5 : 1.0;
  const VectorizedArray<double> spinPolarizedFactorVect =
    (d_dftParams.spinPolarized == 1) ? make_vectorized_array(0.5) :
                                       make_vectorized_array(1.0);

  const unsigned int numPhysicalCells = matrixFreeData.n_physical_cells();

  std::map<dealii::CellId, unsigned int> cellIdToCellNumberMap;

  DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(),
                                      endc = dftPtr->dofHandler.end();
  unsigned int iElem                       = 0;
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      {
        cellIdToCellNumberMap[cell->id()] = iElem;
        iElem++;
      }



  // band group parallelization data structures
  const unsigned int numberBandGroups =
    dealii::Utilities::MPI::n_mpi_processes(dftPtr->interBandGroupComm);
  const unsigned int bandGroupTaskId =
    dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm);
  std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
  dftUtils::createBandParallelizationIndices(dftPtr->interBandGroupComm,
                                             numEigenVectors,
                                             bandGroupLowHighPlusOneIndices);


  const unsigned int localVectorSize =
    dftPtr->d_eigenVectorsFlattenedSTL[0].size() / numEigenVectors;

  const unsigned int numMacroCells = matrixFreeData.n_macro_cells();

// FIXME: This check is no longer needed
#ifdef DFTFE_WITH_GPU
  AssertThrow(
    numMacroCells == numPhysicalCells,
    ExcMessage(
      "DFT-FE Error: dealii for GPU DFT-FE must be compiled without any vectorization enabled."));
#endif


  std::vector<std::vector<double>> partialOccupancies(
    numKPoints,
    std::vector<double>((1 + d_dftParams.spinPolarized) * numEigenVectors,
                        0.0));
  for (unsigned int spinIndex = 0; spinIndex < (1 + d_dftParams.spinPolarized);
       ++spinIndex)
    for (unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
      for (unsigned int iWave = 0; iWave < numEigenVectors; ++iWave)
        {
          const double eigenValue =
            dftPtr->eigenValues[kPoint][numEigenVectors * spinIndex + iWave];
          partialOccupancies[kPoint][numEigenVectors * spinIndex + iWave] =
            dftUtils::getPartialOccupancy(eigenValue,
                                          dftPtr->fermiEnergy,
                                          C_kb,
                                          d_dftParams.TVal);

          if (d_dftParams.constraintMagnetization)
            {
              partialOccupancies[kPoint][numEigenVectors * spinIndex + iWave] =
                1.0;
              if (spinIndex == 0)
                {
                  if (eigenValue > dftPtr->fermiEnergyUp)
                    partialOccupancies[kPoint][numEigenVectors * spinIndex +
                                               iWave] = 0.0;
                }
              else if (spinIndex == 1)
                {
                  if (eigenValue > dftPtr->fermiEnergyDown)
                    partialOccupancies[kPoint][numEigenVectors * spinIndex +
                                               iWave] = 0.0;
                }
            }
        }

  MPI_Barrier(d_mpiCommParent);
  init_time = MPI_Wtime() - init_time;

  for (unsigned int spinIndex = 0; spinIndex < (1 + d_dftParams.spinPolarized);
       ++spinIndex)
    {
      std::vector<double> elocWfcEshelbyTensorQuadValuesH(
        numKPoints * numPhysicalCells * numQuadPoints * 9, 0.0);

      std::vector<dataTypes::number>
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened(
          numKPoints *
            dftPtr->d_sumNonTrivialPseudoWfcsOverAllCellsZetaDeltaVQuads *
            numQuadPointsNLP * 3,
          dataTypes::number(0.0));

#ifdef USE_COMPLEX
      std::vector<dataTypes::number>
        projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened(
          numKPoints *
            dftPtr->d_sumNonTrivialPseudoWfcsOverAllCellsZetaDeltaVQuads *
            numQuadPointsNLP,
          dataTypes::number(0.0));
#endif



#if defined(DFTFE_WITH_GPU)
      if (d_dftParams.useGPU)
        {
          MPI_Barrier(d_mpiCommParent);
          double gpu_time = MPI_Wtime();

          forceCUDA::wfcContractionsForceKernelsAllH(
            kohnShamDFTEigenOperatorGPU,
            dftPtr->d_eigenVectorsFlattenedCUDA.begin(),
            d_dftParams.spinPolarized,
            spinIndex,
            dftPtr->eigenValues,
            partialOccupancies,
            dftPtr->d_kPointCoordinates,
            &dftPtr->d_nonTrivialAllCellsPseudoWfcIdToElemIdMap[0],
            &dftPtr->d_projecterKetTimesFlattenedVectorLocalIds[0],
            localVectorSize,
            numEigenVectors,
            numPhysicalCells,
            numQuadPoints,
            numQuadPointsNLP,
            dftPtr->matrix_free_data.get_dofs_per_cell(
              dftPtr->d_densityDofHandlerIndex),
            dftPtr->d_sumNonTrivialPseudoWfcsOverAllCellsZetaDeltaVQuads,
            &elocWfcEshelbyTensorQuadValuesH[0],
            &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
              [0],
#  ifdef USE_COMPLEX
            &projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened
              [0],
#  endif
            d_mpiCommParent,
            dftPtr->interBandGroupComm,
            isPseudopotential,
            d_dftParams.floatingNuclearCharges,
            false,
            d_dftParams);

          MPI_Barrier(d_mpiCommParent);
          gpu_time = MPI_Wtime() - gpu_time;

          if (this_process == 0 && d_dftParams.verbosity >= 4)
            std::cout << "Time for wfc contractions in forces: " << gpu_time
                      << std::endl;
        }
      else
#endif
        {
          MPI_Barrier(d_mpiCommParent);
          double host_time = MPI_Wtime();

          force::wfcContractionsForceKernelsAllH(
            kohnShamDFTEigenOperator,
            dftPtr->d_eigenVectorsFlattenedSTL,
            d_dftParams.spinPolarized,
            spinIndex,
            dftPtr->eigenValues,
            partialOccupancies,
            dftPtr->d_kPointCoordinates,
            &dftPtr->d_nonTrivialAllCellsPseudoWfcIdToElemIdMap[0],
            &dftPtr->d_projecterKetTimesFlattenedVectorLocalIds[0],
            localVectorSize,
            numEigenVectors,
            numPhysicalCells,
            numQuadPoints,
            numQuadPointsNLP,
            dftPtr->matrix_free_data.get_dofs_per_cell(
              dftPtr->d_densityDofHandlerIndex),
            dftPtr->d_sumNonTrivialPseudoWfcsOverAllCellsZetaDeltaVQuads,
            &elocWfcEshelbyTensorQuadValuesH[0],
            &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
              [0],
#ifdef USE_COMPLEX
            &projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened
              [0],
#endif
            d_mpiCommParent,
            dftPtr->interBandGroupComm,
            isPseudopotential,
            d_dftParams.floatingNuclearCharges,
            false,
            d_dftParams);

          MPI_Barrier(d_mpiCommParent);
          host_time = MPI_Wtime() - host_time;

          if (this_process == 0 && d_dftParams.verbosity >= 4)
            std::cout << "Time for wfc contractions in forces: " << host_time
                      << std::endl;
        }

      dataTypes::number check1 =
        std::accumulate(elocWfcEshelbyTensorQuadValuesH.begin(),
                        elocWfcEshelbyTensorQuadValuesH.end(),
                        dataTypes::number(0.0));
      std::cout << "check1: " << check1 << std::endl;



      // dataTypes::number
      // check2=std::accumulate(projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened.begin(),projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened.end(),dataTypes::number(0.0));
      // std::cout<<"check2: "<<check2<<std::endl;

      // double
      // check2=std::accumulate(dftPtr->d_nonTrivialAllCellsPseudoWfcIdToElemIdMap.begin(),dftPtr->d_nonTrivialAllCellsPseudoWfcIdToElemIdMap.end(),0.0);
      // std::cout<<"check2: "<<check2<<std::endl;
      // std::cout<<"check3:
      // "<<dftPtr->d_sumNonTrivialPseudoWfcsOverAllCellsZetaDeltaVQuads
      // <<std::endl;

      if (!d_dftParams.floatingNuclearCharges)
        {
          dealii::AlignedVector<Tensor<2, 3, VectorizedArray<double>>> EQuad(
            numQuadPoints, zeroTensor4);
          for (unsigned int cell = 0; cell < matrixFreeData.n_macro_cells();
               ++cell)
            {
              forceEval.reinit(cell);

              std::fill(EQuad.begin(), EQuad.end(), zeroTensor4);

              const unsigned int numSubCells =
                matrixFreeData.n_components_filled(cell);

              for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                   ++iSubCell)
                {
                  subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
                  const unsigned int physicalCellId =
                    cellIdToCellNumberMap[subCellPtr->id()];
                  for (unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
                    {
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          const unsigned int id =
                            kPoint * numPhysicalCells * numQuadPoints * 9 +
                            physicalCellId * numQuadPoints * 9 + q * 9;
                          EQuad[q][0][0][iSubCell] +=
                            dftPtr->d_kPointWeights[kPoint] *
                            elocWfcEshelbyTensorQuadValuesH[id + 0];
                          EQuad[q][0][1][iSubCell] +=
                            dftPtr->d_kPointWeights[kPoint] *
                            elocWfcEshelbyTensorQuadValuesH[id + 1];
                          EQuad[q][0][2][iSubCell] +=
                            dftPtr->d_kPointWeights[kPoint] *
                            elocWfcEshelbyTensorQuadValuesH[id + 2];
                          EQuad[q][1][0][iSubCell] +=
                            dftPtr->d_kPointWeights[kPoint] *
                            elocWfcEshelbyTensorQuadValuesH[id + 3];
                          EQuad[q][1][1][iSubCell] +=
                            dftPtr->d_kPointWeights[kPoint] *
                            elocWfcEshelbyTensorQuadValuesH[id + 4];
                          EQuad[q][1][2][iSubCell] +=
                            dftPtr->d_kPointWeights[kPoint] *
                            elocWfcEshelbyTensorQuadValuesH[id + 5];
                          EQuad[q][2][0][iSubCell] +=
                            dftPtr->d_kPointWeights[kPoint] *
                            elocWfcEshelbyTensorQuadValuesH[id + 6];
                          EQuad[q][2][1][iSubCell] +=
                            dftPtr->d_kPointWeights[kPoint] *
                            elocWfcEshelbyTensorQuadValuesH[id + 7];
                          EQuad[q][2][2][iSubCell] +=
                            dftPtr->d_kPointWeights[kPoint] *
                            elocWfcEshelbyTensorQuadValuesH[id + 8];
                        } // quad loop
                    }     // kpoint loop
                }         // subcell loop


              for (unsigned int q = 0; q < numQuadPoints; ++q)
                forceEval.submit_gradient(spinPolarizedFactorVect * EQuad[q],
                                          q);

              forceEval.integrate(false, true);
#ifdef USE_COMPLEX
              forceEval.distribute_local_to_global(
                d_configForceVectorLinFEKPoints);
#else
              forceEval.distribute_local_to_global(d_configForceVectorLinFE);
#endif
            } // cell loop
        }

      if (isPseudopotential)
        {
          dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
            FVectQuads(numQuadPointsNLP, zeroTensor3);
          for (unsigned int cell = 0; cell < matrixFreeData.n_macro_cells();
               ++cell)
            {
              forceEvalNLP.reinit(cell);

              // compute FnlGammaAtoms  (contibution due to Gamma(Rj))
              FnlGammaAtomsElementalContribution(
                forceContributionFnlGammaAtoms,
                matrixFreeData,
                forceEvalNLP,
                cell,
                cellIdToCellNumberMap,
#ifdef USE_COMPLEX
                projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened,
#endif
                dftPtr->d_nonLocalPSP_ZetalmDeltaVl,
                projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened);


              if (!d_dftParams.floatingNuclearCharges)
                {
                  FnlGammaxElementalContribution(
                    FVectQuads,
                    matrixFreeData,
                    numQuadPointsNLP,
                    cell,
                    cellIdToCellNumberMap,
                    dftPtr->d_nonLocalPSP_ZetalmDeltaVl,
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened);

                  for (unsigned int q = 0; q < numQuadPointsNLP; ++q)
                    forceEvalNLP.submit_value(spinPolarizedFactorVect *
                                                FVectQuads[q],
                                              q);

                  forceEvalNLP.integrate(true, false);

#ifdef USE_COMPLEX
                  forceEvalNLP.distribute_local_to_global(
                    d_configForceVectorLinFEKPoints);
#else
                  forceEvalNLP.distribute_local_to_global(
                    d_configForceVectorLinFE);
#endif
                } // no floating charges check
            }     // macro cell loop
        }         // pseudopotential check
    }             // spin index

  // add global Fnl contribution due to Gamma(Rj) to the configurational force
  // vector
  if (isPseudopotential)
    {
      if (d_dftParams.spinPolarized == 1)
        for (auto &iter : forceContributionFnlGammaAtoms)
          {
            std::vector<double> &fnlvec = iter.second;
            for (unsigned int i = 0; i < fnlvec.size(); i++)
              fnlvec[i] *= spinPolarizedFactor;
          }

      if (d_dftParams.floatingNuclearCharges)
        {
#ifdef USE_COMPLEX
          accumulateForceContributionGammaAtomsFloating(
            forceContributionFnlGammaAtoms, d_forceAtomsFloatingKPoints);
#else
          accumulateForceContributionGammaAtomsFloating(
            forceContributionFnlGammaAtoms, d_forceAtomsFloating);
#endif
        }
      else
        distributeForceContributionFnlGammaAtoms(
          forceContributionFnlGammaAtoms);
    }


  MPI_Barrier(d_mpiCommParent);
  double enowfc_time = MPI_Wtime();

  /////////// Compute contribution independent of wavefunctions
  ////////////////////
  if (bandGroupTaskId == 0)
    {
      // kpoint group parallelization data structures
      const unsigned int numberKptGroups =
        dealii::Utilities::MPI::n_mpi_processes(dftPtr->interpoolcomm);

      const unsigned int kptGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(dftPtr->interpoolcomm);
      std::vector<int> kptGroupLowHighPlusOneIndices;

      if (numMacroCells > 0)
        dftUtils::createKpointParallelizationIndices(
          dftPtr->interpoolcomm, numMacroCells, kptGroupLowHighPlusOneIndices);

      if (d_dftParams.spinPolarized == 1)
        {
          dealii::AlignedVector<VectorizedArray<double>> rhoXCQuadsVect(
            numQuadPoints, make_vectorized_array(0.0));
          dealii::AlignedVector<VectorizedArray<double>> phiTotRhoOutQuads(
            numQuadPoints, make_vectorized_array(0.0));
          dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
            gradRhoSpin0QuadsVect(numQuadPoints, zeroTensor3);
          dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
            gradRhoSpin1QuadsVect(numQuadPoints, zeroTensor3);
          dealii::AlignedVector<Tensor<2, 3, VectorizedArray<double>>>
            hessianRhoSpin0Quads(numQuadPoints, zeroTensor4);
          dealii::AlignedVector<Tensor<2, 3, VectorizedArray<double>>>
                                                         hessianRhoSpin1Quads(numQuadPoints, zeroTensor4);
          dealii::AlignedVector<VectorizedArray<double>> excQuads(
            numQuadPoints, make_vectorized_array(0.0));
          dealii::AlignedVector<VectorizedArray<double>> vxcRhoOutSpin0Quads(
            numQuadPoints, make_vectorized_array(0.0));
          dealii::AlignedVector<VectorizedArray<double>> vxcRhoOutSpin1Quads(
            numQuadPoints, make_vectorized_array(0.0));
          dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
            derExchCorrEnergyWithGradRhoOutSpin0Quads(numQuadPoints,
                                                      zeroTensor3);
          dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
            derExchCorrEnergyWithGradRhoOutSpin1Quads(numQuadPoints,
                                                      zeroTensor3);
          dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
            gradRhoCoreQuads(numQuadPoints, zeroTensor3);
          dealii::AlignedVector<Tensor<2, 3, VectorizedArray<double>>>
            hessianRhoCoreQuads(numQuadPoints, zeroTensor4);
          std::map<unsigned int, std::vector<double>>
            forceContributionNonlinearCoreCorrectionGammaAtoms;

          for (unsigned int cell = 0; cell < matrixFreeData.n_macro_cells();
               ++cell)
            {
              if (cell <
                    kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId + 1] &&
                  cell >= kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId])
                {
                  forceEval.reinit(cell);

                  std::fill(rhoXCQuadsVect.begin(),
                            rhoXCQuadsVect.end(),
                            make_vectorized_array(0.0));
                  std::fill(phiTotRhoOutQuads.begin(),
                            phiTotRhoOutQuads.end(),
                            make_vectorized_array(0.0));
                  std::fill(gradRhoSpin0QuadsVect.begin(),
                            gradRhoSpin0QuadsVect.end(),
                            zeroTensor3);
                  std::fill(gradRhoSpin1QuadsVect.begin(),
                            gradRhoSpin1QuadsVect.end(),
                            zeroTensor3);
                  std::fill(hessianRhoSpin0Quads.begin(),
                            hessianRhoSpin0Quads.end(),
                            zeroTensor4);
                  std::fill(hessianRhoSpin1Quads.begin(),
                            hessianRhoSpin1Quads.end(),
                            zeroTensor4);
                  std::fill(excQuads.begin(),
                            excQuads.end(),
                            make_vectorized_array(0.0));
                  std::fill(vxcRhoOutSpin0Quads.begin(),
                            vxcRhoOutSpin0Quads.end(),
                            make_vectorized_array(0.0));
                  std::fill(vxcRhoOutSpin1Quads.begin(),
                            vxcRhoOutSpin1Quads.end(),
                            make_vectorized_array(0.0));
                  std::fill(derExchCorrEnergyWithGradRhoOutSpin0Quads.begin(),
                            derExchCorrEnergyWithGradRhoOutSpin0Quads.end(),
                            zeroTensor3);
                  std::fill(derExchCorrEnergyWithGradRhoOutSpin1Quads.begin(),
                            derExchCorrEnergyWithGradRhoOutSpin1Quads.end(),
                            zeroTensor3);
                  std::fill(gradRhoCoreQuads.begin(),
                            gradRhoCoreQuads.end(),
                            zeroTensor3);
                  std::fill(hessianRhoCoreQuads.begin(),
                            hessianRhoCoreQuads.end(),
                            zeroTensor4);

                  const unsigned int numSubCells =
                    matrixFreeData.n_components_filled(cell);
                  // For LDA
                  std::vector<double> exchValRhoOut(numQuadPoints);
                  std::vector<double> corrValRhoOut(numQuadPoints);
                  std::vector<double> exchPotValRhoOut(2 * numQuadPoints);
                  std::vector<double> corrPotValRhoOut(2 * numQuadPoints);
                  std::vector<double> rhoOutQuadsXC(2 * numQuadPoints);

                  //
                  // For GGA
                  std::vector<double> sigmaValRhoOut(3 * numQuadPoints);
                  std::vector<double> derExchEnergyWithDensityValRhoOut(
                    2 * numQuadPoints),
                    derCorrEnergyWithDensityValRhoOut(2 * numQuadPoints),
                    derExchEnergyWithSigmaRhoOut(3 * numQuadPoints),
                    derCorrEnergyWithSigmaRhoOut(3 * numQuadPoints);
                  std::vector<Tensor<1, 3, double>> gradRhoOutQuadsXCSpin0(
                    numQuadPoints);
                  std::vector<Tensor<1, 3, double>> gradRhoOutQuadsXCSpin1(
                    numQuadPoints);

                  //
                  for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                       ++iSubCell)
                    {
                      subCellPtr =
                        matrixFreeData.get_cell_iterator(cell, iSubCell);
                      dealii::CellId subCellId = subCellPtr->id();

                      const std::vector<double> &temp =
                        (*dftPtr->rhoOutValues).find(subCellId)->second;
                      const std::vector<double> &temp1 =
                        (*dftPtr->rhoOutValuesSpinPolarized)
                          .find(subCellId)
                          ->second;

                      rhoOutQuadsXC = temp1;
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          rhoXCQuadsVect[q][iSubCell] = temp[q];
                        }

                      if (d_dftParams.nonLinearCoreCorrection)
                        {
                          const std::vector<double> &temp2 =
                            rhoCoreValues.find(subCellId)->second;
                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            {
                              rhoOutQuadsXC[2 * q + 0] += temp2[q] / 2.0;
                              rhoOutQuadsXC[2 * q + 1] += temp2[q] / 2.0;
                              rhoXCQuadsVect[q][iSubCell] += temp2[q];
                            }
                        }

                      if (dftPtr->excFunctionalPtr
                            ->getDensityBasedFamilyType() ==
                          densityFamilyType::GGA)
                        {
                          const std::vector<double> &temp3 =
                            (*dftPtr->gradRhoOutValuesSpinPolarized)
                              .find(subCellId)
                              ->second;
                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            for (unsigned int idim = 0; idim < 3; idim++)
                              {
                                gradRhoOutQuadsXCSpin0[q][idim] =
                                  temp3[6 * q + idim];
                                gradRhoOutQuadsXCSpin1[q][idim] =
                                  temp3[6 * q + 3 + idim];
                                gradRhoSpin0QuadsVect[q][idim][iSubCell] =
                                  temp3[6 * q + idim];
                                gradRhoSpin1QuadsVect[q][idim][iSubCell] =
                                  temp3[6 * q + 3 + idim];
                              }

                          if (d_dftParams.nonLinearCoreCorrection)
                            {
                              const std::vector<double> &temp4 =
                                gradRhoCoreValues.find(subCellId)->second;
                              for (unsigned int q = 0; q < numQuadPoints; ++q)
                                for (unsigned int idim = 0; idim < 3; idim++)
                                  {
                                    gradRhoOutQuadsXCSpin0[q][idim] +=
                                      temp4[3 * q + idim] / 2.0;
                                    gradRhoOutQuadsXCSpin1[q][idim] +=
                                      temp4[3 * q + idim] / 2.0;
                                  }
                            }
                        }

                      if (dftPtr->excFunctionalPtr
                            ->getDensityBasedFamilyType() ==
                          densityFamilyType::GGA)
                        {
                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            {
                              sigmaValRhoOut[3 * q + 0] =
                                scalar_product(gradRhoOutQuadsXCSpin0[q],
                                               gradRhoOutQuadsXCSpin0[q]);
                              sigmaValRhoOut[3 * q + 1] =
                                scalar_product(gradRhoOutQuadsXCSpin0[q],
                                               gradRhoOutQuadsXCSpin1[q]);
                              sigmaValRhoOut[3 * q + 2] =
                                scalar_product(gradRhoOutQuadsXCSpin1[q],
                                               gradRhoOutQuadsXCSpin1[q]);
                            }

                          std::map<rhoDataAttributes,
                                   const std::vector<double> *>
                            rhoOutData;

                          std::map<VeffOutputDataAttributes,
                                   std::vector<double> *>
                            outputDerExchangeEnergy;
                          std::map<VeffOutputDataAttributes,
                                   std::vector<double> *>
                            outputDerCorrEnergy;

                          rhoOutData[rhoDataAttributes::values] =
                            &rhoOutQuadsXC;
                          rhoOutData[rhoDataAttributes::sigmaGradValue] =
                            &sigmaValRhoOut;

                          outputDerExchangeEnergy
                            [VeffOutputDataAttributes::derEnergyWithDensity] =
                              &derExchEnergyWithDensityValRhoOut;
                          outputDerExchangeEnergy
                            [VeffOutputDataAttributes::
                               derEnergyWithSigmaGradDensity] =
                              &derExchEnergyWithSigmaRhoOut;

                          outputDerCorrEnergy
                            [VeffOutputDataAttributes::derEnergyWithDensity] =
                              &derCorrEnergyWithDensityValRhoOut;
                          outputDerCorrEnergy[VeffOutputDataAttributes::
                                                derEnergyWithSigmaGradDensity] =
                            &derCorrEnergyWithSigmaRhoOut;

                          dftPtr->excFunctionalPtr
                            ->computeDensityBasedEnergyDensity(numQuadPoints,
                                                               rhoOutData,
                                                               exchValRhoOut,
                                                               corrValRhoOut);

                          dftPtr->excFunctionalPtr->computeDensityBasedVxc(
                            numQuadPoints,
                            rhoOutData,
                            outputDerExchangeEnergy,
                            outputDerCorrEnergy);


                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            {
                              excQuads[q][iSubCell] =
                                exchValRhoOut[q] + corrValRhoOut[q];
                              vxcRhoOutSpin0Quads[q][iSubCell] =
                                derExchEnergyWithDensityValRhoOut[2 * q] +
                                derCorrEnergyWithDensityValRhoOut[2 * q];
                              vxcRhoOutSpin1Quads[q][iSubCell] =
                                derExchEnergyWithDensityValRhoOut[2 * q + 1] +
                                derCorrEnergyWithDensityValRhoOut[2 * q + 1];
                              for (unsigned int idim = 0; idim < 3; idim++)
                                {
                                  derExchCorrEnergyWithGradRhoOutSpin0Quads
                                    [q][idim][iSubCell] =
                                      2.0 *
                                      (derExchEnergyWithSigmaRhoOut[3 * q + 0] +
                                       derCorrEnergyWithSigmaRhoOut[3 * q +
                                                                    0]) *
                                      gradRhoOutQuadsXCSpin0[q][idim];
                                  derExchCorrEnergyWithGradRhoOutSpin0Quads
                                    [q][idim][iSubCell] +=
                                    (derExchEnergyWithSigmaRhoOut[3 * q + 1] +
                                     derCorrEnergyWithSigmaRhoOut[3 * q + 1]) *
                                    gradRhoOutQuadsXCSpin1[q][idim];

                                  derExchCorrEnergyWithGradRhoOutSpin1Quads
                                    [q][idim][iSubCell] +=
                                    2.0 *
                                    (derExchEnergyWithSigmaRhoOut[3 * q + 2] +
                                     derCorrEnergyWithSigmaRhoOut[3 * q + 2]) *
                                    gradRhoOutQuadsXCSpin1[q][idim];
                                  derExchCorrEnergyWithGradRhoOutSpin1Quads
                                    [q][idim][iSubCell] +=
                                    (derExchEnergyWithSigmaRhoOut[3 * q + 1] +
                                     derCorrEnergyWithSigmaRhoOut[3 * q + 1]) *
                                    gradRhoOutQuadsXCSpin0[q][idim];
                                }
                            }
                        }
                      else if (dftPtr->excFunctionalPtr
                                 ->getDensityBasedFamilyType() ==
                               densityFamilyType::LDA)
                        {
                          std::map<rhoDataAttributes,
                                   const std::vector<double> *>
                            rhoOutData;

                          std::map<VeffOutputDataAttributes,
                                   std::vector<double> *>
                            outputDerExchangeEnergy;
                          std::map<VeffOutputDataAttributes,
                                   std::vector<double> *>
                            outputDerCorrEnergy;

                          rhoOutData[rhoDataAttributes::values] =
                            &rhoOutQuadsXC;


                          outputDerExchangeEnergy
                            [VeffOutputDataAttributes::derEnergyWithDensity] =
                              &exchPotValRhoOut;

                          outputDerCorrEnergy
                            [VeffOutputDataAttributes::derEnergyWithDensity] =
                              &corrPotValRhoOut;

                          dftPtr->excFunctionalPtr
                            ->computeDensityBasedEnergyDensity(numQuadPoints,
                                                               rhoOutData,
                                                               exchValRhoOut,
                                                               corrValRhoOut);

                          dftPtr->excFunctionalPtr->computeDensityBasedVxc(
                            numQuadPoints,
                            rhoOutData,
                            outputDerExchangeEnergy,
                            outputDerCorrEnergy);

                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            {
                              excQuads[q][iSubCell] =
                                exchValRhoOut[q] + corrValRhoOut[q];
                              vxcRhoOutSpin0Quads[q][iSubCell] =
                                exchPotValRhoOut[2 * q] +
                                corrPotValRhoOut[2 * q];
                              vxcRhoOutSpin1Quads[q][iSubCell] =
                                exchPotValRhoOut[2 * q + 1] +
                                corrPotValRhoOut[2 * q + 1];
                            }
                        }

                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          if (d_dftParams.nonLinearCoreCorrection == true)
                            {
                              const std::vector<double> &temp1 =
                                gradRhoCoreValues.find(subCellId)->second;
                              for (unsigned int q = 0; q < numQuadPoints; ++q)
                                for (unsigned int idim = 0; idim < 3; idim++)
                                  gradRhoCoreQuads[q][idim][iSubCell] =
                                    temp1[3 * q + idim] / 2.0;

                              if (dftPtr->excFunctionalPtr
                                    ->getDensityBasedFamilyType() ==
                                  densityFamilyType::GGA)
                                {
                                  const std::vector<double> &temp2 =
                                    hessianRhoCoreValues.find(subCellId)
                                      ->second;
                                  for (unsigned int q = 0; q < numQuadPoints;
                                       ++q)
                                    for (unsigned int idim = 0; idim < 3;
                                         ++idim)
                                      for (unsigned int jdim = 0; jdim < 3;
                                           ++jdim)
                                        hessianRhoCoreQuads
                                          [q][idim][jdim][iSubCell] =
                                            temp2[9 * q + 3 * idim + jdim] /
                                            2.0;
                                }
                            }
                        }

                    } // subcell loop

                  if (d_dftParams.nonLinearCoreCorrection)
                    {
                      FNonlinearCoreCorrectionGammaAtomsElementalContributionSpinPolarized(
                        forceContributionNonlinearCoreCorrectionGammaAtoms,
                        forceEval,
                        matrixFreeData,
                        cell,
                        vxcRhoOutSpin0Quads,
                        vxcRhoOutSpin1Quads,
                        derExchCorrEnergyWithGradRhoOutSpin0Quads,
                        derExchCorrEnergyWithGradRhoOutSpin1Quads,
                        gradRhoCoreAtoms,
                        hessianRhoCoreAtoms,
                        dftPtr->excFunctionalPtr->getDensityBasedFamilyType() ==
                          densityFamilyType::GGA);
                    }

                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      const VectorizedArray<double> phiTot_q =
                        phiTotRhoOutQuads[q];

                      Tensor<2, 3, VectorizedArray<double>> E =
                        eshelbyTensorSP::getELocXcEshelbyTensor(
                          rhoXCQuadsVect[q],
                          gradRhoSpin0QuadsVect[q],
                          gradRhoSpin1QuadsVect[q],
                          excQuads[q],
                          derExchCorrEnergyWithGradRhoOutSpin0Quads[q],
                          derExchCorrEnergyWithGradRhoOutSpin1Quads[q]);

                      Tensor<1, 3, VectorizedArray<double>> F = zeroTensor3;

                      if (d_dftParams.nonLinearCoreCorrection)
                        F += eshelbyTensorSP::getFNonlinearCoreCorrection(
                          vxcRhoOutSpin0Quads[q],
                          vxcRhoOutSpin1Quads[q],
                          derExchCorrEnergyWithGradRhoOutSpin0Quads[q],
                          derExchCorrEnergyWithGradRhoOutSpin1Quads[q],
                          gradRhoCoreQuads[q],
                          hessianRhoCoreQuads[q],
                          dftPtr->excFunctionalPtr
                              ->getDensityBasedFamilyType() ==
                            densityFamilyType::GGA);

                      forceEval.submit_value(F, q);
                      forceEval.submit_gradient(E, q);
                    } // quad point loop

                  forceEval.integrate(true, true);
                  forceEval.distribute_local_to_global(
                    d_configForceVectorLinFE); // also takes care of constraints
                }                              // kpt paral
            }                                  // cell loop

          if (d_dftParams.nonLinearCoreCorrection)
            {
              if (d_dftParams.floatingNuclearCharges)
                accumulateForceContributionGammaAtomsFloating(
                  forceContributionNonlinearCoreCorrectionGammaAtoms,
                  d_forceAtomsFloating);
              else
                distributeForceContributionFPSPLocalGammaAtoms(
                  forceContributionNonlinearCoreCorrectionGammaAtoms,
                  d_atomsForceDofs,
                  d_constraintsNoneForce,
                  d_configForceVectorLinFE);
            }
        }
      else
        {
          dealii::AlignedVector<VectorizedArray<double>> rhoQuads(
            numQuadPoints, make_vectorized_array(0.0));
          dealii::AlignedVector<VectorizedArray<double>> rhoXCQuads(
            numQuadPoints, make_vectorized_array(0.0));
          dealii::AlignedVector<VectorizedArray<double>> phiTotRhoOutQuads(
            numQuadPoints, make_vectorized_array(0.0));
          dealii::AlignedVector<VectorizedArray<double>>
            derVxcWithRhoTimesRhoDiffQuads(numQuadPoints,
                                           make_vectorized_array(0.0));

          dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
            gradRhoQuads(numQuadPoints, zeroTensor3);
          // dealii::AlignedVector<Tensor<1,3,VectorizedArray<double> > >
          // gradRhoAtomsQuads(numQuadPoints,zeroTensor3);
          dealii::AlignedVector<Tensor<2, 3, VectorizedArray<double>>>
            hessianRhoQuads(numQuadPoints, zeroTensor4);
          // dealii::AlignedVector<Tensor<2,3,VectorizedArray<double> > >
          // hessianRhoAtomsQuads(numQuadPoints,zeroTensor4);
          dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
            gradRhoCoreQuads(numQuadPoints, zeroTensor3);
          dealii::AlignedVector<Tensor<2, 3, VectorizedArray<double>>>
                                                         hessianRhoCoreQuads(numQuadPoints, zeroTensor4);
          dealii::AlignedVector<VectorizedArray<double>> excQuads(
            numQuadPoints, make_vectorized_array(0.0));
          dealii::AlignedVector<VectorizedArray<double>> vxcRhoOutQuads(
            numQuadPoints, make_vectorized_array(0.0));
          dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
            derExchCorrEnergyWithGradRhoOutQuads(numQuadPoints, zeroTensor3);
          dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
                                                         derVxcWithGradRhoQuads(numQuadPoints, zeroTensor3);
          dealii::AlignedVector<VectorizedArray<double>> derVxcWithRhoQuads(
            numQuadPoints, make_vectorized_array(0.0));
          dealii::AlignedVector<Tensor<2, 3, VectorizedArray<double>>>
            der2ExcWithGradRhoQuads(numQuadPoints, zeroTensor4);
          std::map<unsigned int, std::vector<double>>
            forceContributionGradRhoNonlinearCoreCorrectionGammaAtoms;
          std::map<unsigned int, std::vector<double>>
            forceContributionHessianRhoNonlinearCoreCorrectionGammaAtoms;

          for (unsigned int cell = 0; cell < matrixFreeData.n_macro_cells();
               ++cell)
            {
              if (cell <
                    kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId + 1] &&
                  cell >= kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId])
                {
                  forceEval.reinit(cell);

                  std::fill(rhoQuads.begin(),
                            rhoQuads.end(),
                            make_vectorized_array(0.0));
                  std::fill(rhoXCQuads.begin(),
                            rhoXCQuads.end(),
                            make_vectorized_array(0.0));
                  std::fill(phiTotRhoOutQuads.begin(),
                            phiTotRhoOutQuads.end(),
                            make_vectorized_array(0.0));
                  std::fill(derVxcWithRhoTimesRhoDiffQuads.begin(),
                            derVxcWithRhoTimesRhoDiffQuads.end(),
                            make_vectorized_array(0.0));
                  std::fill(gradRhoQuads.begin(),
                            gradRhoQuads.end(),
                            zeroTensor3);
                  // std::fill(gradRhoAtomsQuads.begin(),gradRhoAtomsQuads.end(),zeroTensor3);
                  std::fill(hessianRhoQuads.begin(),
                            hessianRhoQuads.end(),
                            zeroTensor4);
                  // std::fill(hessianRhoAtomsQuads.begin(),hessianRhoAtomsQuads.end(),zeroTensor4);
                  std::fill(gradRhoCoreQuads.begin(),
                            gradRhoCoreQuads.end(),
                            zeroTensor3);
                  std::fill(hessianRhoCoreQuads.begin(),
                            hessianRhoCoreQuads.end(),
                            zeroTensor4);
                  std::fill(excQuads.begin(),
                            excQuads.end(),
                            make_vectorized_array(0.0));
                  std::fill(vxcRhoOutQuads.begin(),
                            vxcRhoOutQuads.end(),
                            make_vectorized_array(0.0));
                  std::fill(derExchCorrEnergyWithGradRhoOutQuads.begin(),
                            derExchCorrEnergyWithGradRhoOutQuads.end(),
                            zeroTensor3);
                  std::fill(derVxcWithGradRhoQuads.begin(),
                            derVxcWithGradRhoQuads.end(),
                            zeroTensor3);
                  std::fill(derVxcWithRhoQuads.begin(),
                            derVxcWithRhoQuads.end(),
                            make_vectorized_array(0.0));
                  std::fill(der2ExcWithGradRhoQuads.begin(),
                            der2ExcWithGradRhoQuads.end(),
                            zeroTensor4);

                  const unsigned int numSubCells =
                    matrixFreeData.n_components_filled(cell);
                  // For LDA
                  std::vector<double> exchValRhoOut(numQuadPoints);
                  std::vector<double> corrValRhoOut(numQuadPoints);
                  std::vector<double> exchPotValRhoOut(numQuadPoints);
                  std::vector<double> corrPotValRhoOut(numQuadPoints);
                  std::vector<double> rhoOutQuadsXC(numQuadPoints);

                  //
                  // For GGA
                  std::vector<double> sigmaValRhoOut(numQuadPoints);
                  std::vector<double> derExchEnergyWithDensityValRhoOut(
                    numQuadPoints),
                    derCorrEnergyWithDensityValRhoOut(numQuadPoints),
                    derExchEnergyWithSigmaRhoOut(numQuadPoints),
                    derCorrEnergyWithSigmaRhoOut(numQuadPoints);
                  std::vector<Tensor<1, 3, double>> gradRhoOutQuadsXC(
                    numQuadPoints);
                  std::vector<double> derVxWithSigmaRhoOut(numQuadPoints);
                  std::vector<double> derVcWithSigmaRhoOut(numQuadPoints);
                  std::vector<double> der2ExWithSigmaRhoOut(numQuadPoints);
                  std::vector<double> der2EcWithSigmaRhoOut(numQuadPoints);
                  std::vector<double> derVxWithRhoOut(numQuadPoints);
                  std::vector<double> derVcWithRhoOut(numQuadPoints);

                  //
                  for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                       ++iSubCell)
                    {
                      subCellPtr =
                        matrixFreeData.get_cell_iterator(cell, iSubCell);
                      dealii::CellId subCellId = subCellPtr->id();

                      const std::vector<double> &temp1 =
                        rhoOutValues.find(subCellId)->second;
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          rhoOutQuadsXC[q]        = temp1[q];
                          rhoQuads[q][iSubCell]   = temp1[q];
                          rhoXCQuads[q][iSubCell] = temp1[q];
                        }

                      if (d_dftParams.nonLinearCoreCorrection)
                        {
                          const std::vector<double> &temp2 =
                            rhoCoreValues.find(subCellId)->second;
                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            {
                              rhoOutQuadsXC[q] += temp2[q];
                              rhoXCQuads[q][iSubCell] += temp2[q];
                            }
                        }

                      if (dftPtr->excFunctionalPtr
                            ->getDensityBasedFamilyType() ==
                          densityFamilyType::GGA)
                        {
                          const std::vector<double> &temp3 =
                            gradRhoOutValues.find(subCellId)->second;
                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            for (unsigned int idim = 0; idim < 3; idim++)
                              {
                                gradRhoOutQuadsXC[q][idim] =
                                  temp3[3 * q + idim];
                                gradRhoQuads[q][idim][iSubCell] =
                                  temp3[3 * q + idim];
                              }

                          if (d_dftParams.nonLinearCoreCorrection)
                            {
                              const std::vector<double> &temp4 =
                                gradRhoCoreValues.find(subCellId)->second;
                              for (unsigned int q = 0; q < numQuadPoints; ++q)
                                {
                                  gradRhoOutQuadsXC[q][0] += temp4[3 * q + 0];
                                  gradRhoOutQuadsXC[q][1] += temp4[3 * q + 1];
                                  gradRhoOutQuadsXC[q][2] += temp4[3 * q + 2];
                                }
                            }
                        }

                      if (dftPtr->excFunctionalPtr
                            ->getDensityBasedFamilyType() ==
                          densityFamilyType::GGA)
                        {
                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            sigmaValRhoOut[q] =
                              gradRhoOutQuadsXC[q].norm_square();

                          std::map<rhoDataAttributes,
                                   const std::vector<double> *>
                            rhoOutData;

                          std::map<VeffOutputDataAttributes,
                                   std::vector<double> *>
                            outputDerExchangeEnergy;
                          std::map<VeffOutputDataAttributes,
                                   std::vector<double> *>
                            outputDerCorrEnergy;

                          rhoOutData[rhoDataAttributes::values] =
                            &rhoOutQuadsXC;
                          rhoOutData[rhoDataAttributes::sigmaGradValue] =
                            &sigmaValRhoOut;

                          outputDerExchangeEnergy
                            [VeffOutputDataAttributes::derEnergyWithDensity] =
                              &derExchEnergyWithDensityValRhoOut;
                          outputDerExchangeEnergy
                            [VeffOutputDataAttributes::
                               derEnergyWithSigmaGradDensity] =
                              &derExchEnergyWithSigmaRhoOut;

                          outputDerCorrEnergy
                            [VeffOutputDataAttributes::derEnergyWithDensity] =
                              &derCorrEnergyWithDensityValRhoOut;
                          outputDerCorrEnergy[VeffOutputDataAttributes::
                                                derEnergyWithSigmaGradDensity] =
                            &derCorrEnergyWithSigmaRhoOut;

                          dftPtr->excFunctionalPtr
                            ->computeDensityBasedEnergyDensity(numQuadPoints,
                                                               rhoOutData,
                                                               exchValRhoOut,
                                                               corrValRhoOut);

                          dftPtr->excFunctionalPtr->computeDensityBasedVxc(
                            numQuadPoints,
                            rhoOutData,
                            outputDerExchangeEnergy,
                            outputDerCorrEnergy);



                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            {
                              excQuads[q][iSubCell] =
                                exchValRhoOut[q] + corrValRhoOut[q];
                              vxcRhoOutQuads[q][iSubCell] =
                                derExchEnergyWithDensityValRhoOut[q] +
                                derCorrEnergyWithDensityValRhoOut[q];

                              for (unsigned int idim = 0; idim < 3; idim++)
                                {
                                  derExchCorrEnergyWithGradRhoOutQuads
                                    [q][idim][iSubCell] =
                                      2.0 *
                                      (derExchEnergyWithSigmaRhoOut[q] +
                                       derCorrEnergyWithSigmaRhoOut[q]) *
                                      gradRhoOutQuadsXC[q][idim];
                                }
                            }
                        }
                      else if (dftPtr->excFunctionalPtr
                                 ->getDensityBasedFamilyType() ==
                               densityFamilyType::LDA)
                        {
                          std::map<rhoDataAttributes,
                                   const std::vector<double> *>
                            rhoOutData;

                          std::map<VeffOutputDataAttributes,
                                   std::vector<double> *>
                            outputDerExchangeEnergy;
                          std::map<VeffOutputDataAttributes,
                                   std::vector<double> *>
                            outputDerCorrEnergy;

                          rhoOutData[rhoDataAttributes::values] =
                            &rhoOutQuadsXC;

                          outputDerExchangeEnergy
                            [VeffOutputDataAttributes::derEnergyWithDensity] =
                              &exchPotValRhoOut;

                          outputDerCorrEnergy
                            [VeffOutputDataAttributes::derEnergyWithDensity] =
                              &corrPotValRhoOut;

                          dftPtr->excFunctionalPtr
                            ->computeDensityBasedEnergyDensity(numQuadPoints,
                                                               rhoOutData,
                                                               exchValRhoOut,
                                                               corrValRhoOut);

                          dftPtr->excFunctionalPtr->computeDensityBasedVxc(
                            numQuadPoints,
                            rhoOutData,
                            outputDerExchangeEnergy,
                            outputDerCorrEnergy);



                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            {
                              excQuads[q][iSubCell] =
                                exchValRhoOut[q] + corrValRhoOut[q];
                              vxcRhoOutQuads[q][iSubCell] =
                                exchPotValRhoOut[q] + corrPotValRhoOut[q];
                            }
                        }

                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          if (d_dftParams.nonLinearCoreCorrection == true)
                            {
                              const std::vector<double> &temp1 =
                                gradRhoCoreValues.find(subCellId)->second;
                              for (unsigned int q = 0; q < numQuadPoints; ++q)
                                for (unsigned int idim = 0; idim < 3; idim++)
                                  gradRhoCoreQuads[q][idim][iSubCell] =
                                    temp1[3 * q + idim];

                              if (dftPtr->excFunctionalPtr
                                    ->getDensityBasedFamilyType() ==
                                  densityFamilyType::GGA)
                                {
                                  const std::vector<double> &temp2 =
                                    hessianRhoCoreValues.find(subCellId)
                                      ->second;
                                  for (unsigned int q = 0; q < numQuadPoints;
                                       ++q)
                                    for (unsigned int idim = 0; idim < 3;
                                         ++idim)
                                      for (unsigned int jdim = 0; jdim < 3;
                                           ++jdim)
                                        hessianRhoCoreQuads
                                          [q][idim][jdim][iSubCell] =
                                            temp2[9 * q + 3 * idim + jdim];
                                }
                            }
                        }
                    } // subcell loop

                  if (d_dftParams.nonLinearCoreCorrection)
                    {
                      FNonlinearCoreCorrectionGammaAtomsElementalContribution(
                        forceContributionGradRhoNonlinearCoreCorrectionGammaAtoms,
                        forceEval,
                        matrixFreeData,
                        cell,
                        vxcRhoOutQuads,
                        gradRhoCoreAtoms);


                      if (dftPtr->excFunctionalPtr
                            ->getDensityBasedFamilyType() ==
                          densityFamilyType::GGA)
                        FNonlinearCoreCorrectionGammaAtomsElementalContribution(
                          forceContributionHessianRhoNonlinearCoreCorrectionGammaAtoms,
                          forceEval,
                          matrixFreeData,
                          cell,
                          derExchCorrEnergyWithGradRhoOutQuads,
                          hessianRhoCoreAtoms);
                    }


                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      const VectorizedArray<double> phiTot_q =
                        phiTotRhoOutQuads[q];

                      Tensor<2, 3, VectorizedArray<double>> E =
                        eshelbyTensor::getELocXcEshelbyTensor(
                          rhoXCQuads[q],
                          gradRhoQuads[q],
                          excQuads[q],
                          derExchCorrEnergyWithGradRhoOutQuads[q]);


                      Tensor<1, 3, VectorizedArray<double>> F = zeroTensor3;


                      if (d_dftParams.nonLinearCoreCorrection)
                        {
                          F += eshelbyTensor::getFNonlinearCoreCorrection(
                            vxcRhoOutQuads[q], gradRhoCoreQuads[q]);

                          if (dftPtr->excFunctionalPtr
                                ->getDensityBasedFamilyType() ==
                              densityFamilyType::GGA)
                            F += eshelbyTensor::getFNonlinearCoreCorrection(
                              derExchCorrEnergyWithGradRhoOutQuads[q],
                              hessianRhoCoreQuads[q]);
                        }

                      forceEval.submit_value(F, q);
                      forceEval.submit_gradient(E, q);
                    } // quad point loop


                  forceEval.integrate(true, true);
                  forceEval.distribute_local_to_global(
                    d_configForceVectorLinFE); // also takes care of constraints
                }                              // kpt paral
            }                                  // cell loop


          if (d_dftParams.nonLinearCoreCorrection)
            {
              if (d_dftParams.floatingNuclearCharges)
                {
                  accumulateForceContributionGammaAtomsFloating(
                    forceContributionGradRhoNonlinearCoreCorrectionGammaAtoms,
                    d_forceAtomsFloating);

                  if (dftPtr->excFunctionalPtr->getDensityBasedFamilyType() ==
                      densityFamilyType::GGA)
                    accumulateForceContributionGammaAtomsFloating(
                      forceContributionHessianRhoNonlinearCoreCorrectionGammaAtoms,
                      d_forceAtomsFloating);
                }
              else
                {
                  distributeForceContributionFPSPLocalGammaAtoms(
                    forceContributionGradRhoNonlinearCoreCorrectionGammaAtoms,
                    d_atomsForceDofs,
                    d_constraintsNoneForce,
                    d_configForceVectorLinFE);

                  if (dftPtr->excFunctionalPtr->getDensityBasedFamilyType() ==
                      densityFamilyType::GGA)
                    distributeForceContributionFPSPLocalGammaAtoms(
                      forceContributionHessianRhoNonlinearCoreCorrectionGammaAtoms,
                      d_atomsForceDofs,
                      d_constraintsNoneForce,
                      d_configForceVectorLinFE);
                }
            }
        }

      ////Add electrostatic configurational force contribution////////////////
      computeConfigurationalForceEEshelbyEElectroPhiTot(
        matrixFreeDataElectro,
        phiTotDofHandlerIndexElectro,
        smearedChargeQuadratureId,
        lpspQuadratureIdElectro,
        phiTotRhoOutElectro,
        rhoOutValuesElectro,
        rhoOutValuesElectroLpsp,
        gradRhoOutValuesElectro,
        gradRhoOutValuesElectroLpsp,
        pseudoVLocElectro,
        pseudoVLocAtomsElectro,
        vselfBinsManagerElectro,
        shadowKSRhoMinValues,
        phiRhoMinusApproxRho,
        shadowPotentialForce);
    }

  MPI_Barrier(d_mpiCommParent);
  enowfc_time = MPI_Wtime() - enowfc_time;

  forcetotal_time = MPI_Wtime() - forcetotal_time;

  if (this_process == 0 && d_dftParams.verbosity >= 4)
    std::cout
      << "Total time for configurational force computation except Eself contribution: "
      << forcetotal_time << std::endl;

  if (d_dftParams.verbosity >= 4)
    {
      pcout << " Time taken for initialization in force: " << init_time
            << std::endl;
      pcout << " Time taken for non wfc in force: " << enowfc_time << std::endl;
    }
}

template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::
  computeConfigurationalForceEEshelbyEElectroPhiTot(
    const MatrixFree<3, double> &    matrixFreeDataElectro,
    const unsigned int               phiTotDofHandlerIndexElectro,
    const unsigned int               smearedChargeQuadratureId,
    const unsigned int               lpspQuadratureIdElectro,
    const distributedCPUVec<double> &phiTotRhoOutElectro,
    const std::map<dealii::CellId, std::vector<double>> &rhoOutValuesElectro,
    const std::map<dealii::CellId, std::vector<double>>
      &rhoOutValuesElectroLpsp,
    const std::map<dealii::CellId, std::vector<double>>
      &gradRhoOutValuesElectro,
    const std::map<dealii::CellId, std::vector<double>>
      &gradRhoOutValuesElectroLpsp,
    const std::map<dealii::CellId, std::vector<double>> &pseudoVLocElectro,
    const std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
      &                                              pseudoVLocAtomsElectro,
    const vselfBinsManager<FEOrder, FEOrderElectro> &vselfBinsManagerElectro,
    const std::map<dealii::CellId, std::vector<double>> &shadowKSRhoMinValues,
    const distributedCPUVec<double> &phiRhoMinusApproxRhoElectro,
    const bool                       shadowPotentialForce)
{
  FEEvaluation<3,
               1,
               C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
               3>
    forceEvalElectro(matrixFreeDataElectro,
                     d_forceDofHandlerIndexElectro,
                     dftPtr->d_densityQuadratureIdElectro);

  FEEvaluation<3,
               FEOrderElectro,
               C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
               1>
    phiTotEvalElectro(matrixFreeDataElectro,
                      phiTotDofHandlerIndexElectro,
                      dftPtr->d_densityQuadratureIdElectro);

  FEEvaluation<3, -1> phiTotEvalSmearedCharge(matrixFreeDataElectro,
                                              phiTotDofHandlerIndexElectro,
                                              smearedChargeQuadratureId);

  FEEvaluation<3, -1, 1, 3> forceEvalSmearedCharge(
    matrixFreeDataElectro,
    d_forceDofHandlerIndexElectro,
    smearedChargeQuadratureId);

  FEEvaluation<3,
               1,
               C_num1DQuadLPSP<FEOrderElectro>() * C_numCopies1DQuadLPSP(),
               3>
    forceEvalElectroLpsp(matrixFreeDataElectro,
                         d_forceDofHandlerIndexElectro,
                         lpspQuadratureIdElectro);

  std::map<unsigned int, std::vector<double>>
    forceContributionFPSPLocalGammaAtoms;
  std::map<unsigned int, std::vector<double>>
    forceContributionSmearedChargesGammaAtoms;
  std::map<unsigned int, std::vector<double>>
    forceContributionShadowPotentialElectroGammaAtoms;

  const unsigned int numQuadPoints         = forceEvalElectro.n_q_points;
  const unsigned int numQuadPointsSmearedb = forceEvalSmearedCharge.n_q_points;
  const unsigned int numQuadPointsLpsp     = forceEvalElectroLpsp.n_q_points;

  AssertThrow(
    matrixFreeDataElectro.get_quadrature(smearedChargeQuadratureId).size() ==
      numQuadPointsSmearedb,
    dealii::ExcMessage(
      "DFT-FE Error: mismatch in quadrature rule usage in force computation."));

  AssertThrow(
    matrixFreeDataElectro.get_quadrature(lpspQuadratureIdElectro).size() ==
      numQuadPointsLpsp,
    dealii::ExcMessage(
      "DFT-FE Error: mismatch in quadrature rule usage in force computation."));

  if (gradRhoOutValuesElectroLpsp.size() != 0)
    AssertThrow(
      gradRhoOutValuesElectroLpsp.begin()->second.size() ==
        3 * numQuadPointsLpsp,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in force computation."));

  DoFHandler<3>::active_cell_iterator subCellPtr;

  FEValues<3> feVselfValuesElectro(
    matrixFreeDataElectro.get_dof_handler(phiTotDofHandlerIndexElectro)
      .get_fe(),
    matrixFreeDataElectro.get_quadrature(lpspQuadratureIdElectro),
    d_dftParams.floatingNuclearCharges && d_dftParams.smearedNuclearCharges ?
      (update_values | update_quadrature_points) :
      (update_values | update_gradients | update_quadrature_points));

  QIterated<3 - 1> faceQuadrature(QGauss<1>(C_num1DQuadLPSP<FEOrderElectro>()),
                                  C_numCopies1DQuadLPSP());
  FEFaceValues<3>  feFaceValuesElectro(dftPtr->d_dofHandlerRhoNodal.get_fe(),
                                      faceQuadrature,
                                      update_values | update_JxW_values |
                                        update_normal_vectors |
                                        update_quadrature_points);

  Tensor<1, 3, VectorizedArray<double>> zeroTensor;
  for (unsigned int idim = 0; idim < 3; idim++)
    {
      zeroTensor[idim] = make_vectorized_array(0.0);
    }

  Tensor<2, 3, VectorizedArray<double>> zeroTensor2;
  for (unsigned int idim = 0; idim < 3; idim++)
    for (unsigned int jdim = 0; jdim < 3; jdim++)
      {
        zeroTensor2[idim][jdim] = make_vectorized_array(0.0);
      }

  // In case of shadow potential force the standard rhoQuads input is the rhoIn
  // (or rhoTilde in the XLBOMD formulation) except for the external potential
  // correction term where the rhoQuads is the variationally optimized one
  // (rhOut)

  dealii::AlignedVector<VectorizedArray<double>> rhoQuadsElectro(
    numQuadPoints, make_vectorized_array(0.0));
  dealii::AlignedVector<VectorizedArray<double>> rhoQuadsElectroLpsp(
    numQuadPointsLpsp, make_vectorized_array(0.0));
  dealii::AlignedVector<VectorizedArray<double>> smearedbQuads(
    numQuadPointsSmearedb, make_vectorized_array(0.0));
  dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
    gradPhiTotSmearedChargeQuads(numQuadPointsSmearedb, zeroTensor);
  dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
                                                 gradPhiTotPlusPhiRhoMinusApproxRhoSmearedChargeQuads(numQuadPointsSmearedb,
                                                         zeroTensor);
  dealii::AlignedVector<VectorizedArray<double>> shadowKSRhoMinQuadsElectro(
    numQuadPoints, make_vectorized_array(0.0));
  dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
    gradPhiRhoMinusApproxRhoQuadsElectro(numQuadPoints, zeroTensor);
  dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
    gradRhoQuadsElectro(numQuadPoints, zeroTensor);
  dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
    gradRhoQuadsElectroLpsp(numQuadPointsLpsp, zeroTensor);
  // dealii::AlignedVector<Tensor<1,3,VectorizedArray<double> > >
  // gradRhoAtomsQuadsElectro(numQuadPoints,zeroTensor);
  dealii::AlignedVector<VectorizedArray<double>> pseudoVLocQuadsElectro(
    numQuadPointsLpsp, make_vectorized_array(0.0));

  // kpoint group parallelization data structures
  const unsigned int numberKptGroups =
    dealii::Utilities::MPI::n_mpi_processes(dftPtr->interpoolcomm);

  const unsigned int kptGroupTaskId =
    dealii::Utilities::MPI::this_mpi_process(dftPtr->interpoolcomm);
  std::vector<int> kptGroupLowHighPlusOneIndices;

  if (matrixFreeDataElectro.n_macro_cells() > 0)
    dftUtils::createKpointParallelizationIndices(
      dftPtr->interpoolcomm,
      matrixFreeDataElectro.n_macro_cells(),
      kptGroupLowHighPlusOneIndices);

  for (unsigned int cell = 0; cell < matrixFreeDataElectro.n_macro_cells();
       ++cell)
    {
      if (cell < kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId + 1] &&
          cell >= kptGroupLowHighPlusOneIndices[2 * kptGroupTaskId])
        {
          std::set<unsigned int> nonTrivialSmearedChargeAtomIdsMacroCell;

          const unsigned int numSubCells =
            matrixFreeDataElectro.n_components_filled(cell);
          if (d_dftParams.smearedNuclearCharges)
            for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
              {
                subCellPtr =
                  matrixFreeDataElectro.get_cell_iterator(cell, iSubCell);
                dealii::CellId                   subCellId = subCellPtr->id();
                const std::vector<unsigned int> &temp =
                  dftPtr->d_bCellNonTrivialAtomIds.find(subCellId)->second;
                for (int i = 0; i < temp.size(); i++)
                  nonTrivialSmearedChargeAtomIdsMacroCell.insert(temp[i]);
              }

          forceEvalElectro.reinit(cell);
          forceEvalElectroLpsp.reinit(cell);

          phiTotEvalElectro.reinit(cell);
          phiTotEvalElectro.read_dof_values_plain(phiTotRhoOutElectro);
          phiTotEvalElectro.evaluate(true, true);

          if (d_dftParams.smearedNuclearCharges &&
              nonTrivialSmearedChargeAtomIdsMacroCell.size() > 0)
            {
              forceEvalSmearedCharge.reinit(cell);
              phiTotEvalSmearedCharge.reinit(cell);
            }

          std::fill(rhoQuadsElectro.begin(),
                    rhoQuadsElectro.end(),
                    make_vectorized_array(0.0));
          std::fill(rhoQuadsElectroLpsp.begin(),
                    rhoQuadsElectroLpsp.end(),
                    make_vectorized_array(0.0));
          std::fill(smearedbQuads.begin(),
                    smearedbQuads.end(),
                    make_vectorized_array(0.0));
          std::fill(gradPhiTotSmearedChargeQuads.begin(),
                    gradPhiTotSmearedChargeQuads.end(),
                    zeroTensor);
          if (shadowPotentialForce)
            {
              std::fill(shadowKSRhoMinQuadsElectro.begin(),
                        shadowKSRhoMinQuadsElectro.end(),
                        make_vectorized_array(0.0));
              std::fill(gradPhiRhoMinusApproxRhoQuadsElectro.begin(),
                        gradPhiRhoMinusApproxRhoQuadsElectro.end(),
                        zeroTensor);
              std::fill(
                gradPhiTotPlusPhiRhoMinusApproxRhoSmearedChargeQuads.begin(),
                gradPhiTotPlusPhiRhoMinusApproxRhoSmearedChargeQuads.end(),
                zeroTensor);
            }
          std::fill(gradRhoQuadsElectro.begin(),
                    gradRhoQuadsElectro.end(),
                    zeroTensor);
          std::fill(gradRhoQuadsElectroLpsp.begin(),
                    gradRhoQuadsElectroLpsp.end(),
                    zeroTensor);
          // std::fill(gradRhoAtomsQuadsElectro.begin(),gradRhoAtomsQuadsElectro.end(),zeroTensor);
          std::fill(pseudoVLocQuadsElectro.begin(),
                    pseudoVLocQuadsElectro.end(),
                    make_vectorized_array(0.0));

          for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
            {
              subCellPtr =
                matrixFreeDataElectro.get_cell_iterator(cell, iSubCell);
              dealii::CellId subCellId = subCellPtr->id();
              for (unsigned int q = 0; q < numQuadPoints; ++q)
                {
                  rhoQuadsElectro[q][iSubCell] =
                    rhoOutValuesElectro.find(subCellId)->second[q];
                  if (shadowPotentialForce)
                    {
                      // shadowKSRhoMinQuadsElectro[q][iSubCell]=shadowKSRhoMinValues.find(subCellId)->second[q];

                      // shadowKSRhoMinMinusRhoQuadsElectro[q][iSubCell]=shadowKSRhoMinQuadsElectro[q][iSubCell]-rhoQuadsElectro[q][iSubCell];

                      // gradRhoAtomsQuadsElectro[q][0][iSubCell]=dftPtr->d_gradRhoAtomsValues.find(subCellId)->second[3*q+0];
                      // gradRhoAtomsQuadsElectro[q][1][iSubCell]=dftPtr->d_gradRhoAtomsValues.find(subCellId)->second[3*q+1];
                      // gradRhoAtomsQuadsElectro[q][2][iSubCell]=dftPtr->d_gradRhoAtomsValues.find(subCellId)->second[3*q+2];
                    }
                }

              if (d_dftParams.isPseudopotential ||
                  d_dftParams.smearedNuclearCharges)
                {
                  const std::vector<double> &tempPseudoVal =
                    pseudoVLocElectro.find(subCellId)->second;
                  const std::vector<double> &tempLpspRhoVal =
                    rhoOutValuesElectroLpsp.find(subCellId)->second;
                  const std::vector<double> &tempLpspGradRhoVal =
                    gradRhoOutValuesElectroLpsp.find(subCellId)->second;
                  for (unsigned int q = 0; q < numQuadPointsLpsp; ++q)
                    {
                      pseudoVLocQuadsElectro[q][iSubCell] = tempPseudoVal[q];
                      rhoQuadsElectroLpsp[q][iSubCell]    = tempLpspRhoVal[q];
                      gradRhoQuadsElectroLpsp[q][0][iSubCell] =
                        tempLpspGradRhoVal[3 * q + 0];
                      gradRhoQuadsElectroLpsp[q][1][iSubCell] =
                        tempLpspGradRhoVal[3 * q + 1];
                      gradRhoQuadsElectroLpsp[q][2][iSubCell] =
                        tempLpspGradRhoVal[3 * q + 2];
                    }
                }

              if (d_dftParams.smearedNuclearCharges &&
                  nonTrivialSmearedChargeAtomIdsMacroCell.size() > 0)
                {
                  const std::vector<double> &bQuadValuesCell =
                    dftPtr->d_bQuadValuesAllAtoms.find(subCellId)->second;
                  for (unsigned int q = 0; q < numQuadPointsSmearedb; ++q)
                    {
                      smearedbQuads[q][iSubCell] = bQuadValuesCell[q];
                    }
                }
            }

          if (d_dftParams.isPseudopotential ||
              d_dftParams.smearedNuclearCharges)
            {
              FPSPLocalGammaAtomsElementalContribution(
                forceContributionFPSPLocalGammaAtoms,
                feVselfValuesElectro,
                feFaceValuesElectro,
                forceEvalElectroLpsp,
                matrixFreeDataElectro,
                phiTotDofHandlerIndexElectro,
                cell,
                rhoQuadsElectroLpsp,
                gradRhoQuadsElectroLpsp,
                pseudoVLocAtomsElectro,
                vselfBinsManagerElectro,
                d_cellsVselfBallsClosestAtomIdDofHandlerElectro);
            }

          for (unsigned int q = 0; q < numQuadPoints; ++q)
            {
              VectorizedArray<double> phiTotElectro_q =
                phiTotEvalElectro.get_value(q);
              VectorizedArray<double> phiExtElectro_q =
                make_vectorized_array(0.0);
              Tensor<1, 3, VectorizedArray<double>> gradPhiTotElectro_q =
                phiTotEvalElectro.get_gradient(q);

              Tensor<2, 3, VectorizedArray<double>> E =
                eshelbyTensor::getEElectroEshelbyTensor(phiTotElectro_q,
                                                        gradPhiTotElectro_q,
                                                        rhoQuadsElectro[q]);

              Tensor<1, 3, VectorizedArray<double>> F = zeroTensor;


              forceEvalElectro.submit_value(F, q);
              forceEvalElectro.submit_gradient(E, q);
            }

          if (d_dftParams.isPseudopotential ||
              d_dftParams.smearedNuclearCharges)
            for (unsigned int q = 0; q < numQuadPointsLpsp; ++q)
              {
                VectorizedArray<double> phiExtElectro_q =
                  make_vectorized_array(0.0);

                Tensor<2, 3, VectorizedArray<double>> E = zeroTensor2;

                Tensor<1, 3, VectorizedArray<double>> F            = zeroTensor;
                Tensor<1, 3, VectorizedArray<double>> gradPhiExt_q = zeroTensor;
                F -= gradRhoQuadsElectroLpsp[q] * pseudoVLocQuadsElectro[q];

                forceEvalElectroLpsp.submit_value(F, q);
                forceEvalElectroLpsp.submit_gradient(E, q);
              }

          forceEvalElectro.integrate(true, true);
          forceEvalElectro.distribute_local_to_global(
            d_configForceVectorLinFEElectro);

          if (d_dftParams.isPseudopotential ||
              d_dftParams.smearedNuclearCharges)
            {
              forceEvalElectroLpsp.integrate(true, true);
              forceEvalElectroLpsp.distribute_local_to_global(
                d_configForceVectorLinFEElectro);
            }

          if (d_dftParams.smearedNuclearCharges &&
              nonTrivialSmearedChargeAtomIdsMacroCell.size() > 0)
            {
              phiTotEvalSmearedCharge.read_dof_values_plain(
                phiTotRhoOutElectro);
              phiTotEvalSmearedCharge.evaluate(false, true);

              for (unsigned int q = 0; q < numQuadPointsSmearedb; ++q)
                {
                  gradPhiTotSmearedChargeQuads[q] =
                    phiTotEvalSmearedCharge.get_gradient(q);

                  Tensor<1, 3, VectorizedArray<double>> F = zeroTensor;
                  F = -gradPhiTotSmearedChargeQuads[q] * smearedbQuads[q];

                  forceEvalSmearedCharge.submit_value(F, q);
                }


              forceEvalSmearedCharge.integrate(true, false);
              forceEvalSmearedCharge.distribute_local_to_global(
                d_configForceVectorLinFEElectro);

              FPhiTotSmearedChargesGammaAtomsElementalContribution(
                forceContributionSmearedChargesGammaAtoms,
                forceEvalSmearedCharge,
                matrixFreeDataElectro,
                cell,
                gradPhiTotSmearedChargeQuads,
                std::vector<unsigned int>(
                  nonTrivialSmearedChargeAtomIdsMacroCell.begin(),
                  nonTrivialSmearedChargeAtomIdsMacroCell.end()),
                dftPtr->d_bQuadAtomIdsAllAtoms,
                smearedbQuads);
            }
        } // kpt paral
    }     // macro cell loop

  // add global FPSPLocal contribution due to Gamma(Rj) to the configurational
  // force vector
  if (d_dftParams.isPseudopotential || d_dftParams.smearedNuclearCharges)
    {
      if (d_dftParams.floatingNuclearCharges)
        {
          accumulateForceContributionGammaAtomsFloating(
            forceContributionFPSPLocalGammaAtoms, d_forceAtomsFloating);
        }
      else
        distributeForceContributionFPSPLocalGammaAtoms(
          forceContributionFPSPLocalGammaAtoms,
          d_atomsForceDofsElectro,
          d_constraintsNoneForceElectro,
          d_configForceVectorLinFEElectro);
    }

  if (d_dftParams.smearedNuclearCharges)
    {
      if (d_dftParams.floatingNuclearCharges)
        {
          accumulateForceContributionGammaAtomsFloating(
            forceContributionSmearedChargesGammaAtoms, d_forceAtomsFloating);
        }
      else
        distributeForceContributionFPSPLocalGammaAtoms(
          forceContributionSmearedChargesGammaAtoms,
          d_atomsForceDofsElectro,
          d_constraintsNoneForceElectro,
          d_configForceVectorLinFEElectro);
    }
}
