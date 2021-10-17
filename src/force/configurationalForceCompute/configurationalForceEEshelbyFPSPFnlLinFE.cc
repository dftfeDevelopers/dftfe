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
      &kohnShamDFTEigenOperator,
#endif
    const unsigned int               eigenDofHandlerIndex,
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
  MPI_Comm_rank(MPI_COMM_WORLD, &this_process);
  MPI_Barrier(MPI_COMM_WORLD);
  double forcetotal_time = MPI_Wtime();

  MPI_Barrier(MPI_COMM_WORLD);
  double init_time = MPI_Wtime();

  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  std::map<unsigned int, std::vector<double>> forceContributionFnlGammaAtoms;

  const bool isPseudopotential = dftParameters::isPseudopotential;

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

#ifdef USE_COMPLEX
  FEEvaluation<3,
               FEOrder,
               C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
               2>
    psiEval(matrixFreeData,
            eigenDofHandlerIndex,
            dftPtr->d_densityQuadratureId);

  FEEvaluation<3,
               FEOrder,
               C_num1DQuadNLPSP<FEOrder>() * C_numCopies1DQuadNLPSP(),
               2>
    psiEvalNLP(matrixFreeData,
               eigenDofHandlerIndex,
               dftPtr->d_nlpspQuadratureId);
#else
  FEEvaluation<3,
               FEOrder,
               C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
               1>
    psiEval(matrixFreeData,
            eigenDofHandlerIndex,
            dftPtr->d_densityQuadratureId);

  FEEvaluation<3,
               FEOrder,
               C_num1DQuadNLPSP<FEOrder>() * C_numCopies1DQuadNLPSP(),
               1>
    psiEvalNLP(matrixFreeData,
               eigenDofHandlerIndex,
               dftPtr->d_nlpspQuadratureId);
#endif


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
  Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>> zeroTensor3Complex;
  Tensor<2, 3, VectorizedArray<double>>               zeroTensor4;
  for (unsigned int idim = 0; idim < 3; idim++)
    {
      zeroTensor2[0][idim]        = make_vectorized_array(0.0);
      zeroTensor2[1][idim]        = make_vectorized_array(0.0);
      zeroTensor3[idim]           = make_vectorized_array(0.0);
      zeroTensor3Complex[0][idim] = make_vectorized_array(0.0);
      zeroTensor3Complex[1][idim] = make_vectorized_array(0.0);
    }
  for (unsigned int idim = 0; idim < 3; idim++)
    {
      for (unsigned int jdim = 0; jdim < 3; jdim++)
        {
          zeroTensor4[idim][jdim] = make_vectorized_array(0.0);
        }
    }

  const double spinPolarizedFactor =
    (dftParameters::spinPolarized == 1) ? 0.5 : 1.0;
  const VectorizedArray<double> spinPolarizedFactorVect =
    (dftParameters::spinPolarized == 1) ? make_vectorized_array(0.5) :
                                          make_vectorized_array(1.0);

  std::map<unsigned int, std::vector<unsigned int>>
    macroIdToNonlocalAtomsSetMap;
  for (unsigned int cell = 0; cell < matrixFreeData.n_macro_cells(); ++cell)
    {
      const unsigned int numSubCells = matrixFreeData.n_components_filled(cell);
      std::set<unsigned int> mergedSet;
      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        {
          subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
          dealii::CellId subCellId = subCellPtr->id();

          std::set<unsigned int> s;
          std::set_union(
            mergedSet.begin(),
            mergedSet.end(),
            dftPtr->d_cellIdToNonlocalAtomIdsLocalCompactSupportMap[subCellId]
              .begin(),
            dftPtr->d_cellIdToNonlocalAtomIdsLocalCompactSupportMap[subCellId]
              .end(),
            std::inserter(s, s.begin()));
          mergedSet = s;
        }
      macroIdToNonlocalAtomsSetMap[cell] =
        std::vector<unsigned int>(mergedSet.begin(), mergedSet.end());
    }

  std::vector<unsigned int> nonlocalPseudoWfcsAccum(
    dftPtr->d_nonLocalAtomIdsInCurrentProcess.size());
  std::vector<unsigned int> numPseudoWfcsAtom(
    dftPtr->d_nonLocalAtomIdsInCurrentProcess.size());
  std::vector<std::vector<unsigned int>> projectorKetTimesVectorLocalIds(
    dftPtr->d_nonLocalAtomIdsInCurrentProcess.size());
  unsigned int numPseudo = 0;
  for (unsigned int iAtom = 0;
       iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
       ++iAtom)
    {
      const unsigned int atomId =
        dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      nonlocalPseudoWfcsAccum[iAtom] = numPseudo;
      numPseudo += dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      numPseudoWfcsAtom[iAtom] =
        dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];

      for (unsigned int ipsp = 0;
           ipsp < dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
           ++ipsp)
        projectorKetTimesVectorLocalIds[iAtom].push_back(
          dftPtr->d_projectorKetTimesVectorPar[0]
            .get_partitioner()
            ->global_to_local(
              dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(
                atomId, ipsp)]));
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

  const unsigned int blockSize = std::min(dftParameters::chebyWfcBlockSize,
                                          bandGroupLowHighPlusOneIndices[1]);

  const unsigned int localVectorSize =
    dftPtr->d_eigenVectorsFlattenedSTL[0].size() / numEigenVectors;
  std::vector<std::vector<distributedCPUVec<double>>> eigenVectors(
    dftPtr->d_kPointWeights.size());
  std::vector<distributedCPUVec<dataTypes::number>> eigenVectorsFlattenedBlock(
    dftPtr->d_kPointWeights.size());

  const unsigned int numMacroCells    = matrixFreeData.n_macro_cells();
  const unsigned int numPhysicalCells = matrixFreeData.n_physical_cells();


#if defined(DFTFE_WITH_GPU) && !defined(USE_COMPLEX)
  AssertThrow(
    numMacroCells == numPhysicalCells,
    ExcMessage(
      "DFT-FE Error: dealii for GPU DFT-FE must be compiled without any vectorization enabled."));

  // create map between macro cell id and normal cell id
  std::vector<unsigned int> normalCellIdToMacroCellIdMap(numPhysicalCells);
  std::vector<unsigned int> macroCellIdToNormalCellIdMap(numPhysicalCells);

  typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;
  unsigned int                                         iElemNormal = 0;
  for (const auto &cell :
       matrixFreeData.get_dof_handler().active_cell_iterators())
    {
      if (cell->is_locally_owned())
        {
          bool         isFound        = false;
          unsigned int iElemMacroCell = 0;
          for (unsigned int iMacroCell = 0; iMacroCell < numMacroCells;
               ++iMacroCell)
            {
              const unsigned int n_sub_cells =
                matrixFreeData.n_components_filled(iMacroCell);
              for (unsigned int iCell = 0; iCell < n_sub_cells; ++iCell)
                {
                  cellPtr = matrixFreeData.get_cell_iterator(iMacroCell, iCell);
                  if (cell->id() == cellPtr->id())
                    {
                      normalCellIdToMacroCellIdMap[iElemNormal] =
                        iElemMacroCell;
                      macroCellIdToNormalCellIdMap[iElemMacroCell] =
                        iElemNormal;
                      isFound = true;
                      break;
                    }
                  iElemMacroCell++;
                }

              if (isFound)
                break;
            }
          iElemNormal++;
        }
    }

  std::vector<unsigned int> nonTrivialNonLocalIdsAllCells;
  std::vector<unsigned int> nonTrivialIdToElemIdMap;
  std::vector<unsigned int> nonTrivialIdToAllPseudoWfcIdMap;
  std::vector<unsigned int> projecterKetTimesFlattenedVectorLocalIds;
  if (isPseudopotential)
    {
      for (unsigned int ielem = 0; ielem < numPhysicalCells; ++ielem)
        {
          const unsigned int numNonLocalAtomsCurrentProc =
            dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();

          const unsigned int macroCellId = normalCellIdToMacroCellIdMap[ielem];
          for (unsigned int iatom = 0; iatom < numNonLocalAtomsCurrentProc;
               ++iatom)
            {
              bool               isNonTrivial = false;
              const unsigned int macroCellId =
                normalCellIdToMacroCellIdMap[ielem];
              for (unsigned int i = 0;
                   i < macroIdToNonlocalAtomsSetMap[macroCellId].size();
                   i++)
                if (macroIdToNonlocalAtomsSetMap[macroCellId][i] == iatom)
                  {
                    isNonTrivial = true;
                    break;
                  }
              if (isNonTrivial)
                {
                  const int globalAtomId =
                    dftPtr->d_nonLocalAtomIdsInCurrentProcess[iatom];
                  const unsigned int numberSingleAtomPseudoWaveFunctions =
                    numPseudoWfcsAtom[iatom];
                  for (unsigned int ipsp = 0;
                       ipsp < numberSingleAtomPseudoWaveFunctions;
                       ++ipsp)
                    {
                      nonTrivialNonLocalIdsAllCells.push_back(iatom);
                      nonTrivialIdToElemIdMap.push_back(ielem);
                      nonTrivialIdToAllPseudoWfcIdMap.push_back(
                        nonlocalPseudoWfcsAccum[iatom] + ipsp);
                      // const unsigned int
                      // id=dftPtr->d_projectorKetTimesVectorPar[0].get_partitioner()->global_to_local(dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(globalAtomId,ipsp)]);
                      projecterKetTimesFlattenedVectorLocalIds.push_back(
                        projectorKetTimesVectorLocalIds[iatom][ipsp]);
                    }
                }
            }
        }
    }
#endif


#ifdef USE_COMPLEX
  // vector of k points times quadPoints times macrocells, nonlocal atom id,
  // pseudo wave
  // FIXME: flatten nonlocal atomid id and pseudo wave and k point
  dealii::AlignedVector<dealii::AlignedVector<
    dealii::AlignedVector<Tensor<1, 2, VectorizedArray<double>>>>>
    ZetaDeltaVQuads;
#else
  // FIXME: flatten nonlocal atom id and pseudo wave
  // vector of quadPoints times macrocells, nonlocal atom id, pseudo wave
  dealii::AlignedVector<
    dealii::AlignedVector<dealii::AlignedVector<VectorizedArray<double>>>>
    ZetaDeltaVQuads;
#endif


  if (isPseudopotential)
    {
      ZetaDeltaVQuads.resize(numKPoints * numMacroCells * numQuadPointsNLP);

      for (unsigned int q = 0; q < ZetaDeltaVQuads.size(); ++q)
        {
          ZetaDeltaVQuads[q].resize(dftPtr->d_nonLocalPSP_ZetalmDeltaVl.size());

          for (unsigned int i = 0;
               i < dftPtr->d_nonLocalPSP_ZetalmDeltaVl.size();
               ++i)
            {
              const int numberPseudoWaveFunctions =
                dftPtr->d_nonLocalPSP_ZetalmDeltaVl[i].size();
#ifdef USE_COMPLEX
              ZetaDeltaVQuads[q][i].resize(numberPseudoWaveFunctions,
                                           zeroTensor1);
#else
              ZetaDeltaVQuads[q][i].resize(numberPseudoWaveFunctions,
                                           make_vectorized_array(0.0));
#endif
            }
        }

      for (unsigned int ikPoint = 0; ikPoint < numKPoints; ++ikPoint)
        {
          for (unsigned int cell = 0; cell < matrixFreeData.n_macro_cells();
               ++cell)
            {
              const unsigned int numSubCells =
                matrixFreeData.n_components_filled(cell);
              for (unsigned int iSubCell = 0; iSubCell < numSubCells;
                   ++iSubCell)
                {
                  subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
                  dealii::CellId subCellId = subCellPtr->id();

                  for (unsigned int q = 0; q < numQuadPointsNLP; ++q)
                    {
                      for (unsigned int i = 0;
                           i < dftPtr->d_nonLocalPSP_ZetalmDeltaVl.size();
                           ++i)
                        {
                          const int numberPseudoWaveFunctions =
                            dftPtr->d_nonLocalPSP_ZetalmDeltaVl[i].size();
                          for (unsigned int iPseudoWave = 0;
                               iPseudoWave < numberPseudoWaveFunctions;
                               ++iPseudoWave)
                            {
                              if (dftPtr
                                    ->d_nonLocalPSP_ZetalmDeltaVl[i]
                                                                 [iPseudoWave]
                                    .find(subCellId) !=
                                  dftPtr
                                    ->d_nonLocalPSP_ZetalmDeltaVl[i]
                                                                 [iPseudoWave]
                                    .end())
                                {
#ifdef USE_COMPLEX

                                  ZetaDeltaVQuads
                                    [ikPoint * matrixFreeData.n_macro_cells() *
                                       numQuadPointsNLP +
                                     cell * numQuadPointsNLP + q][i]
                                    [iPseudoWave][0][iSubCell] =
                                      dftPtr->d_nonLocalPSP_ZetalmDeltaVl
                                        [i][iPseudoWave][subCellId]
                                        [ikPoint * numQuadPointsNLP * 2 +
                                         q * 2 + 0];
                                  ZetaDeltaVQuads
                                    [ikPoint * matrixFreeData.n_macro_cells() *
                                       numQuadPointsNLP +
                                     cell * numQuadPointsNLP + q][i]
                                    [iPseudoWave][1][iSubCell] =
                                      dftPtr->d_nonLocalPSP_ZetalmDeltaVl
                                        [i][iPseudoWave][subCellId]
                                        [ikPoint * numQuadPointsNLP * 2 +
                                         q * 2 + 1];
#else
                                  ZetaDeltaVQuads[cell * numQuadPointsNLP +
                                                  q][i][iPseudoWave][iSubCell] =
                                    dftPtr->d_nonLocalPSP_ZetalmDeltaVl
                                      [i][iPseudoWave][subCellId][q];
#endif
                                } // non-trivial cellId check
                            }     // iPseudoWave loop
                        }         // i loop
                    }             // q loop
                }                 // subcell loop
            }                     // macrocell loop
        }                         // kpoint loop
    }

  std::vector<std::vector<double>> partialOccupancies(
    numKPoints,
    std::vector<double>((1 + dftParameters::spinPolarized) * numEigenVectors,
                        0.0));
  for (unsigned int spinIndex = 0;
       spinIndex < (1 + dftParameters::spinPolarized);
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
                                          dftParameters::TVal);

          if (dftParameters::constraintMagnetization)
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

  MPI_Barrier(MPI_COMM_WORLD);
  init_time = MPI_Wtime() - init_time;

  for (unsigned int spinIndex = 0;
       spinIndex < (1 + dftParameters::spinPolarized);
       ++spinIndex)
    {
#if defined(DFTFE_WITH_GPU) && !defined(USE_COMPLEX)
      std::vector<double>
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened(
          nonTrivialNonLocalIdsAllCells.size() * numQuadPointsNLP * 3, 0.0);
      std::vector<double> elocWfcEshelbyTensorQuadValuesH(numPhysicalCells *
                                                            numQuadPoints * 6,
                                                          0.0);
#endif
      std::vector<std::vector<std::vector<dataTypes::number>>>
        projectorKetTimesPsiTimesVTimesPartOcc(numKPoints);
#ifdef USE_COMPLEX
      dealii::AlignedVector<dealii::AlignedVector<
        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>>
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuads(
          numKPoints * numMacroCells * numQuadPointsNLP,
          dealii::AlignedVector<
            Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>(
            numPseudo, zeroTensor3Complex));

      dealii::AlignedVector<
        dealii::AlignedVector<Tensor<1, 2, VectorizedArray<double>>>>
        projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuads(
          numKPoints * numMacroCells * numQuadPointsNLP,
          dealii::AlignedVector<Tensor<1, 2, VectorizedArray<double>>>(
            numPseudo, zeroTensor1));

#else
      dealii::AlignedVector<
        dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>>
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuads(
          numMacroCells * numQuadPointsNLP,
          dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>(
            numPseudo, zeroTensor3));
#endif

#if defined(DFTFE_WITH_GPU) && !defined(USE_COMPLEX)
      if (dftParameters::useGPU)
        {
          MPI_Barrier(MPI_COMM_WORLD);
          double gpu_time = MPI_Wtime();

          forceCUDA::gpuPortedForceKernelsAllH(
            kohnShamDFTEigenOperator,
            dftPtr->d_eigenVectorsFlattenedCUDA.begin() +
              spinIndex * localVectorSize * numEigenVectors,
            &dftPtr->eigenValues[0][spinIndex * numEigenVectors],
            &partialOccupancies[0][spinIndex * numEigenVectors],
            &nonTrivialIdToElemIdMap[0],
            &projecterKetTimesFlattenedVectorLocalIds[0],
            numEigenVectors,
            numPhysicalCells,
            numQuadPoints,
            numQuadPointsNLP,
            dftPtr->matrix_free_data.get_dofs_per_cell(
              dftPtr->d_densityDofHandlerIndex),
            nonTrivialNonLocalIdsAllCells.size(),
            &elocWfcEshelbyTensorQuadValuesH[0],
            &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
              [0],
            dftPtr->interBandGroupComm,
            isPseudopotential);

          MPI_Barrier(MPI_COMM_WORLD);
          gpu_time = MPI_Wtime() - gpu_time;

          if (this_process == 0 && dftParameters::verbosity >= 4)
            std::cout << "Time for gpuPortedForceKernelsAllH: " << gpu_time
                      << std::endl;
        }
      else
#endif
        {
          for (unsigned int ivec = 0; ivec < numEigenVectors; ivec += blockSize)
            {
              const unsigned int currentBlockSize =
                std::min(blockSize, numEigenVectors - ivec);

              if (currentBlockSize != blockSize || ivec == 0)
                {
                  for (unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
                    {
                      eigenVectors[kPoint].resize(currentBlockSize);
                      for (unsigned int i = 0; i < currentBlockSize; ++i)
                        eigenVectors[kPoint][i].reinit(dftPtr->d_tempEigenVec);


                      vectorTools::createDealiiVector<dataTypes::number>(
                        dftPtr->matrix_free_data.get_vector_partitioner(
                          dftPtr->d_densityDofHandlerIndex),
                        currentBlockSize,
                        eigenVectorsFlattenedBlock[kPoint]);
                      eigenVectorsFlattenedBlock[kPoint] =
                        dataTypes::number(0.0);
                    }

                  dftPtr->constraintsNoneDataInfo.precomputeMaps(
                    dftPtr->matrix_free_data.get_vector_partitioner(
                      dftPtr->d_densityDofHandlerIndex),
                    eigenVectorsFlattenedBlock[0].get_partitioner(),
                    currentBlockSize);
                }

              if ((ivec + currentBlockSize) <=
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                  (ivec + currentBlockSize) >
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  std::vector<std::vector<double>> blockedEigenValues(
                    dftPtr->d_kPointWeights.size(),
                    std::vector<double>(currentBlockSize, 0.0));
                  std::vector<std::vector<double>> blockedPartialOccupancies(
                    dftPtr->d_kPointWeights.size(),
                    std::vector<double>(currentBlockSize, 0.0));
                  for (unsigned int kPoint = 0;
                       kPoint < dftPtr->d_kPointWeights.size();
                       ++kPoint)
                    for (unsigned int iWave = 0; iWave < currentBlockSize;
                         ++iWave)
                      {
                        blockedEigenValues[kPoint][iWave] =
                          dftPtr
                            ->eigenValues[kPoint][numEigenVectors * spinIndex +
                                                  ivec + iWave];
                        blockedPartialOccupancies[kPoint][iWave] =
                          partialOccupancies[kPoint]
                                            [numEigenVectors * spinIndex +
                                             ivec + iWave];
                      }

                  for (unsigned int kPoint = 0;
                       kPoint < dftPtr->d_kPointWeights.size();
                       ++kPoint)
                    {
                      for (unsigned int iNode = 0; iNode < localVectorSize;
                           ++iNode)
                        for (unsigned int iWave = 0; iWave < currentBlockSize;
                             ++iWave)
                          eigenVectorsFlattenedBlock[kPoint].local_element(
                            iNode * currentBlockSize + iWave) =
                            dftPtr->d_eigenVectorsFlattenedSTL
                              [(dftParameters::spinPolarized + 1) * kPoint +
                               spinIndex]
                              [iNode * numEigenVectors + ivec + iWave];

                      dftPtr->constraintsNoneDataInfo.distribute(
                        eigenVectorsFlattenedBlock[kPoint], currentBlockSize);
                      eigenVectorsFlattenedBlock[kPoint].update_ghost_values();

#ifdef USE_COMPLEX
                      vectorTools::copyFlattenedDealiiVecToSingleCompVec(
                        eigenVectorsFlattenedBlock[kPoint],
                        currentBlockSize,
                        std::make_pair(0, currentBlockSize),
                        dftPtr->localProc_dof_indicesReal,
                        dftPtr->localProc_dof_indicesImag,
                        eigenVectors[kPoint],
                        false);

                      // FIXME: The underlying call to update_ghost_values
                      // is required because currently localProc_dof_indicesReal
                      // and localProc_dof_indicesImag are only available for
                      // locally owned nodes. Once they are also made available
                      // for ghost nodes- use true for the last argument in
                      // copyFlattenedDealiiVecToSingleCompVec(..) above and
                      // supress underlying call.
                      for (unsigned int i = 0; i < currentBlockSize; ++i)
                        eigenVectors[kPoint][i].update_ghost_values();
#else
                    vectorTools::copyFlattenedDealiiVecToSingleCompVec(
                      eigenVectorsFlattenedBlock[kPoint],
                      currentBlockSize,
                      std::make_pair(0, currentBlockSize),
                      eigenVectors[kPoint],
                      true);

#endif
                    }

                  if (isPseudopotential)
                    for (unsigned int ikPoint = 0; ikPoint < numKPoints;
                         ++ikPoint)
                      {
                        computeNonLocalProjectorKetTimesPsiTimesVFlattened(
                          eigenVectorsFlattenedBlock[ikPoint],
                          currentBlockSize,
                          projectorKetTimesPsiTimesVTimesPartOcc[ikPoint],
                          ikPoint,
                          blockedPartialOccupancies[ikPoint]);
                      }

                  for (unsigned int cell = 0;
                       cell < matrixFreeData.n_macro_cells();
                       ++cell)
                    {
                      forceEval.reinit(cell);

                      psiEval.reinit(cell);

                      psiEvalNLP.reinit(cell);

                      const unsigned int numSubCells =
                        matrixFreeData.n_components_filled(cell);
#ifdef USE_COMPLEX
                      dealii::AlignedVector<
                        Tensor<1, 2, VectorizedArray<double>>>
                        psiQuads(numQuadPoints * currentBlockSize * numKPoints,
                                 zeroTensor1);
                      dealii::AlignedVector<
                        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>
                        gradPsiQuads(numQuadPoints * currentBlockSize *
                                       numKPoints,
                                     zeroTensor2);
#else
                    dealii::AlignedVector<VectorizedArray<double>> psiQuads(
                      numQuadPoints * currentBlockSize,
                      make_vectorized_array(0.0));
                    dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
                                                                   gradPsiQuads(numQuadPoints * currentBlockSize,
                                   zeroTensor3);
#endif

                      for (unsigned int ikPoint = 0; ikPoint < numKPoints;
                           ++ikPoint)
                        for (unsigned int iEigenVec = 0;
                             iEigenVec < currentBlockSize;
                             ++iEigenVec)
                          {
                            psiEval.read_dof_values_plain(
                              eigenVectors[ikPoint][iEigenVec]);
                            psiEval.evaluate(true, true);

                            for (unsigned int q = 0; q < numQuadPoints; ++q)
                              {
                                const unsigned int id =
                                  q * currentBlockSize * numKPoints +
                                  currentBlockSize * ikPoint + iEigenVec;
                                psiQuads[id]     = psiEval.get_value(q);
                                gradPsiQuads[id] = psiEval.get_gradient(q);
                              } // quad point loop
                          }     // eigenvector loop

#ifdef USE_COMPLEX
                      dealii::AlignedVector<
                        Tensor<1, 2, VectorizedArray<double>>>
                        psiQuadsNLP;
                      dealii::AlignedVector<
                        Tensor<1, 2, Tensor<1, 3, VectorizedArray<double>>>>
                        gradPsiQuadsNLP;
#else
                    dealii::AlignedVector<VectorizedArray<double>> psiQuadsNLP;
                    dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
                      gradPsiQuadsNLP;
#endif

                      if (isPseudopotential)
                        {
#ifdef USE_COMPLEX
                          psiQuadsNLP.resize(numQuadPointsNLP *
                                               currentBlockSize * numKPoints,
                                             zeroTensor1);
                          gradPsiQuadsNLP.resize(numQuadPointsNLP *
                                                   currentBlockSize *
                                                   numKPoints,
                                                 zeroTensor2);
#else
                        psiQuadsNLP.resize(numQuadPointsNLP * currentBlockSize,
                                           make_vectorized_array(0.0));
                        gradPsiQuadsNLP.resize(numQuadPointsNLP *
                                                 currentBlockSize * numKPoints,
                                               zeroTensor3);
#endif

                          for (unsigned int ikPoint = 0; ikPoint < numKPoints;
                               ++ikPoint)
                            for (unsigned int iEigenVec = 0;
                                 iEigenVec < currentBlockSize;
                                 ++iEigenVec)
                              {
                                psiEvalNLP.read_dof_values_plain(
                                  eigenVectors[ikPoint][iEigenVec]);
                                psiEvalNLP.evaluate(true, true);

                                for (unsigned int q = 0; q < numQuadPointsNLP;
                                     ++q)
                                  {
                                    const unsigned int id =
                                      ikPoint * numQuadPointsNLP *
                                        currentBlockSize +
                                      q * currentBlockSize + iEigenVec;
                                    psiQuadsNLP[id] = psiEvalNLP.get_value(q);
                                    gradPsiQuadsNLP[id] =
                                      psiEvalNLP.get_gradient(q);
                                  } // quad point loop
                              }     // eigenvector loop
                        }

                      const unsigned int numNonLocalAtomsCurrentProc =
                        projectorKetTimesPsiTimesVTimesPartOcc[0].size();
                      std::vector<bool> isAtomInCell(
                        numNonLocalAtomsCurrentProc, false);
                      if (isPseudopotential)
                        {
                          std::vector<unsigned int> nonTrivialNonLocalIds;
                          for (unsigned int iatom = 0;
                               iatom < numNonLocalAtomsCurrentProc;
                               ++iatom)
                            {
                              for (unsigned int i = 0;
                                   i <
                                   macroIdToNonlocalAtomsSetMap[cell].size();
                                   i++)
                                if (macroIdToNonlocalAtomsSetMap[cell][i] ==
                                    iatom)
                                  {
                                    isAtomInCell[iatom] = true;
                                    nonTrivialNonLocalIds.push_back(iatom);
                                    break;
                                  }
                            }

                          for (unsigned int kPoint = 0; kPoint < numKPoints;
                               ++kPoint)
                            {
                              for (unsigned int q = 0; q < numQuadPointsNLP;
                                   ++q)
                                {
#ifdef USE_COMPLEX
                                  dealii::AlignedVector<Tensor<
                                    1,
                                    2,
                                    VectorizedArray<
                                      double>>> &tempContractPketPsiWithPsi =
                                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuads
                                      [kPoint * matrixFreeData.n_macro_cells() *
                                         numQuadPointsNLP +
                                       cell * numQuadPointsNLP + q];

                                  dealii::AlignedVector<Tensor<
                                    1,
                                    2,
                                    Tensor<1, 3, VectorizedArray<double>>>>
                                    &tempContractPketPsiWithGradPsi =
                                      projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuads
                                        [kPoint *
                                           matrixFreeData.n_macro_cells() *
                                           numQuadPointsNLP +
                                         cell * numQuadPointsNLP + q];
#else
                                dealii::AlignedVector<Tensor<
                                  1,
                                  3,
                                  VectorizedArray<
                                    double>>> &tempContractPketPsiWithGradPsi =
                                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuads
                                    [kPoint * matrixFreeData.n_macro_cells() *
                                       numQuadPointsNLP +
                                     cell * numQuadPointsNLP + q];
#endif
                                  for (unsigned int i = 0;
                                       i < nonTrivialNonLocalIds.size();
                                       ++i)
                                    {
                                      const unsigned int iatom =
                                        nonTrivialNonLocalIds[i];
                                      const unsigned int
                                        numberSingleAtomPseudoWaveFunctions =
                                          numPseudoWfcsAtom[iatom];
                                      const unsigned int startingId =
                                        nonlocalPseudoWfcsAccum[iatom];
                                      const std::vector<dataTypes::number>
                                        &pketPsi =
                                          projectorKetTimesPsiTimesVTimesPartOcc
                                            [kPoint][iatom];
                                      for (unsigned int ipsp = 0;
                                           ipsp <
                                           numberSingleAtomPseudoWaveFunctions;
                                           ++ipsp)
                                        {
                                          for (unsigned int iEigenVec = 0;
                                               iEigenVec < currentBlockSize;
                                               ++iEigenVec)
                                            {
#ifdef USE_COMPLEX
                                              VectorizedArray<double> psiReal =
                                                psiQuadsNLP[kPoint *
                                                              numQuadPointsNLP *
                                                              currentBlockSize +
                                                            q *
                                                              currentBlockSize +
                                                            iEigenVec][0];
                                              VectorizedArray<double> psiImag =
                                                psiQuadsNLP[kPoint *
                                                              numQuadPointsNLP *
                                                              currentBlockSize +
                                                            q *
                                                              currentBlockSize +
                                                            iEigenVec][1];

                                              const Tensor<
                                                1,
                                                3,
                                                VectorizedArray<double>>
                                                &gradPsiReal = gradPsiQuadsNLP
                                                  [kPoint * numQuadPointsNLP *
                                                     currentBlockSize +
                                                   q * currentBlockSize +
                                                   iEigenVec][0];
                                              const Tensor<
                                                1,
                                                3,
                                                VectorizedArray<double>>
                                                &gradPsiImag = gradPsiQuadsNLP
                                                  [kPoint * numQuadPointsNLP *
                                                     currentBlockSize +
                                                   q * currentBlockSize +
                                                   iEigenVec][1];


                                              const VectorizedArray<double>
                                                pketPsiReal =
                                                  make_vectorized_array(
                                                    pketPsi[ipsp *
                                                              currentBlockSize +
                                                            iEigenVec]
                                                      .real());
                                              const VectorizedArray<double>
                                                pketPsiImag =
                                                  make_vectorized_array(
                                                    pketPsi[ipsp *
                                                              currentBlockSize +
                                                            iEigenVec]
                                                      .imag());


                                              tempContractPketPsiWithGradPsi
                                                [startingId + ipsp][0] +=
                                                gradPsiReal * pketPsiReal +
                                                gradPsiImag * pketPsiImag;
                                              tempContractPketPsiWithGradPsi
                                                [startingId + ipsp][1] +=
                                                gradPsiReal * pketPsiImag -
                                                gradPsiImag * pketPsiReal;

                                              tempContractPketPsiWithPsi
                                                [startingId + ipsp][0] +=
                                                psiReal * pketPsiReal +
                                                psiImag * pketPsiImag;
                                              tempContractPketPsiWithPsi
                                                [startingId + ipsp][1] +=
                                                psiReal * pketPsiImag -
                                                psiImag * pketPsiReal;


#else
                                            tempContractPketPsiWithGradPsi
                                              [startingId + ipsp] +=
                                              gradPsiQuadsNLP
                                                [q * currentBlockSize +
                                                 iEigenVec] *
                                              make_vectorized_array(
                                                pketPsi[ipsp *
                                                          currentBlockSize +
                                                        iEigenVec]);
#endif
                                            } // eigenfunction loop
                                        }     // pseudowfc loop
                                    }         // non trivial non local atom ids
                                }             // nlp quad loop
                            }                 // k point loop
                        }                     // is pseudopotential check


                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
#ifdef USE_COMPLEX
                          Tensor<2, 3, VectorizedArray<double>> E =
                            spinPolarizedFactorVect *
                            eshelbyTensor::
                              getELocWfcEshelbyTensorPeriodicKPoints(
                                psiQuads.begin() +
                                  q * currentBlockSize * numKPoints,
                                gradPsiQuads.begin() +
                                  q * currentBlockSize * numKPoints,
                                dftPtr->d_kPointCoordinates,
                                dftPtr->d_kPointWeights,
                                blockedEigenValues,
                                dftPtr->fermiEnergy,
                                dftParameters::TVal);
#else
                        Tensor<2, 3, VectorizedArray<double>> E =
                          spinPolarizedFactorVect *
                          eshelbyTensor::getELocWfcEshelbyTensorNonPeriodic(
                            psiQuads.begin() + q * currentBlockSize,
                            gradPsiQuads.begin() + q * currentBlockSize,
                            blockedEigenValues[0],
                            blockedPartialOccupancies[0]);
#endif
                          forceEval.submit_gradient(E, q);
                        } // quad point loop


                      forceEval.integrate(false, true);
#ifdef USE_COMPLEX
                      forceEval.distribute_local_to_global(
                        d_configForceVectorLinFEKPoints);
#else
                    forceEval.distribute_local_to_global(
                      d_configForceVectorLinFE); // also takes care of
                                                 // constraints
#endif
                    } // macro cell loop
                }     // band parallelization loop
            }         // wavefunction block loop
        }

#if defined(DFTFE_WITH_GPU) && !defined(USE_COMPLEX)
      if (dftParameters::useGPU)
        {
          for (unsigned int cell = 0; cell < matrixFreeData.n_macro_cells();
               ++cell)
            {
              forceEval.reinit(cell);

              for (unsigned int q = 0; q < numQuadPoints; ++q)
                {
                  Tensor<2, 3, VectorizedArray<double>> E;
                  const unsigned int                    physicalCellId =
                    macroCellIdToNormalCellIdMap[cell];
                  const unsigned int id = physicalCellId * numQuadPoints + q;
                  E[0][0]               = make_vectorized_array(
                    elocWfcEshelbyTensorQuadValuesH[id * 6 + 0]);
                  E[1][0] = make_vectorized_array(
                    elocWfcEshelbyTensorQuadValuesH[id * 6 + 1]);
                  E[1][1] = make_vectorized_array(
                    elocWfcEshelbyTensorQuadValuesH[id * 6 + 2]);
                  E[2][0] = make_vectorized_array(
                    elocWfcEshelbyTensorQuadValuesH[id * 6 + 3]);
                  E[2][1] = make_vectorized_array(
                    elocWfcEshelbyTensorQuadValuesH[id * 6 + 4]);
                  E[2][2] = make_vectorized_array(
                    elocWfcEshelbyTensorQuadValuesH[id * 6 + 5]);
                  E[0][1] = E[1][0];
                  E[0][2] = E[2][0];
                  E[1][2] = E[2][1];
                  forceEval.submit_gradient(spinPolarizedFactorVect * E, q);

                } // quad point loop
              forceEval.integrate(false, true);
              forceEval.distribute_local_to_global(d_configForceVectorLinFE);
            }

          if (isPseudopotential)
            for (unsigned int i = 0; i < nonTrivialNonLocalIdsAllCells.size();
                 ++i)
              {
                const unsigned int cell =
                  normalCellIdToMacroCellIdMap[nonTrivialIdToElemIdMap[i]];
                const unsigned int id = nonTrivialIdToAllPseudoWfcIdMap[i];
                for (unsigned int q = 0; q < numQuadPointsNLP; ++q)
                  {
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuads
                      [cell * numQuadPointsNLP + q][id]
                      [0] = make_vectorized_array(
                        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                          [i * numQuadPointsNLP * 3 + 3 * q + 0]);
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuads
                      [cell * numQuadPointsNLP + q][id]
                      [1] = make_vectorized_array(
                        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                          [i * numQuadPointsNLP * 3 + 3 * q + 1]);
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuads
                      [cell * numQuadPointsNLP + q][id]
                      [2] = make_vectorized_array(
                        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                          [i * numQuadPointsNLP * 3 + 3 * q + 2]);
                  }
              }
        }
#endif

      if (isPseudopotential)
        for (unsigned int cell = 0; cell < matrixFreeData.n_macro_cells();
             ++cell)
          {
            forceEvalNLP.reinit(cell);

            const unsigned int numNonLocalAtomsCurrentProc =
              dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
            std::vector<bool> isAtomInCell(numNonLocalAtomsCurrentProc, false);

            std::vector<unsigned int> nonTrivialNonLocalIds;
            for (unsigned int iatom = 0; iatom < numNonLocalAtomsCurrentProc;
                 ++iatom)
              {
                for (unsigned int i = 0;
                     i < macroIdToNonlocalAtomsSetMap[cell].size();
                     i++)
                  if (macroIdToNonlocalAtomsSetMap[cell][i] == iatom)
                    {
                      isAtomInCell[iatom] = true;
                      nonTrivialNonLocalIds.push_back(iatom);
                      break;
                    }
              }

            // compute FnlGammaAtoms  (contibution due to Gamma(Rj))
            Tensor<1, 3, VectorizedArray<double>> kcoord;
            for (unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
              {
#ifdef USE_COMPLEX
                kcoord[0] = make_vectorized_array(
                  dftPtr->d_kPointCoordinates[kPoint * 3 + 0]);
                kcoord[1] = make_vectorized_array(
                  dftPtr->d_kPointCoordinates[kPoint * 3 + 1]);
                kcoord[2] = make_vectorized_array(
                  dftPtr->d_kPointCoordinates[kPoint * 3 + 2]);
#endif
                FnlGammaAtomsElementalContribution(
                  forceContributionFnlGammaAtoms,
                  forceEval,
                  forceEvalNLP,
                  matrixFreeData.n_macro_cells(),
                  cell,
#ifdef USE_COMPLEX
                  kPoint,
                  ZetaDeltaVQuads,
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuads,
                  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuads,
                  kcoord,
#else
                  ZetaDeltaVQuads,
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuads,
#endif
                  isAtomInCell,
                  nonlocalPseudoWfcsAccum);
              }



#ifdef USE_COMPLEX
            for (unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
              {
                for (unsigned int q = 0; q < numQuadPointsNLP; ++q)
                  {
                    Tensor<1, 3, VectorizedArray<double>> F =
                      make_vectorized_array(dftPtr->d_kPointWeights[kPoint]) *
                      spinPolarizedFactorVect *
                      eshelbyTensor::getFnl(
                        ZetaDeltaVQuads[kPoint *
                                          matrixFreeData.n_macro_cells() *
                                          numQuadPointsNLP +
                                        cell * numQuadPointsNLP + q],
                        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuads
                          [kPoint * matrixFreeData.n_macro_cells() *
                             numQuadPointsNLP +
                           cell * numQuadPointsNLP + q],
                        isAtomInCell,
                        nonlocalPseudoWfcsAccum);

                    forceEvalNLP.submit_value(F, q);
                  } // nonlocal psp quad points loop


                forceEvalNLP.integrate(true, false);

                forceEvalNLP.distribute_local_to_global(
                  d_configForceVectorLinFEKPoints);
              } // k point loop
#else

            for (unsigned int q = 0; q < numQuadPointsNLP; ++q)
              {
                Tensor<1, 3, VectorizedArray<double>> F =
                  spinPolarizedFactorVect *
                  eshelbyTensor::getFnl(
                    ZetaDeltaVQuads[cell * numQuadPointsNLP + q],
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuads
                      [cell * numQuadPointsNLP + q],
                    isAtomInCell,
                    nonlocalPseudoWfcsAccum);

                forceEvalNLP.submit_value(F, q);
              } // nonlocal psp quad points loop


            forceEvalNLP.integrate(true, false);

            forceEvalNLP.distribute_local_to_global(d_configForceVectorLinFE);

#endif
          } // macro cell loop
    }       // spin index

  // add global Fnl contribution due to Gamma(Rj) to the configurational force
  // vector
  if (isPseudopotential)
    {
      if (dftParameters::spinPolarized == 1)
        for (auto &iter : forceContributionFnlGammaAtoms)
          {
            std::vector<double> &fnlvec = iter.second;
            for (unsigned int i = 0; i < fnlvec.size(); i++)
              fnlvec[i] *= spinPolarizedFactor;
          }

      if (dftParameters::floatingNuclearCharges)
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


  MPI_Barrier(MPI_COMM_WORLD);
  double enowfc_time = MPI_Wtime();

  /////////// Compute contribution independent of wavefunctions
  ////////////////////
  if (bandGroupTaskId == 0)
    {
      if (dftParameters::spinPolarized == 1)
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
                  subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
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

                  if (dftParameters::nonLinearCoreCorrection)
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

                  if (dftParameters::xcFamilyType == "GGA")
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

                      if (dftParameters::nonLinearCoreCorrection)
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

                  if (dftParameters::xcFamilyType == "GGA")
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

                      xc_gga_exc_vxc(&(dftPtr->funcX),
                                     numQuadPoints,
                                     &(rhoOutQuadsXC[0]),
                                     &sigmaValRhoOut[0],
                                     &exchValRhoOut[0],
                                     &derExchEnergyWithDensityValRhoOut[0],
                                     &derExchEnergyWithSigmaRhoOut[0]);
                      xc_gga_exc_vxc(&(dftPtr->funcC),
                                     numQuadPoints,
                                     &(rhoOutQuadsXC[0]),
                                     &sigmaValRhoOut[0],
                                     &corrValRhoOut[0],
                                     &derCorrEnergyWithDensityValRhoOut[0],
                                     &derCorrEnergyWithSigmaRhoOut[0]);

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
                                   derCorrEnergyWithSigmaRhoOut[3 * q + 0]) *
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
                  else
                    {
                      xc_lda_exc(&(dftPtr->funcX),
                                 numQuadPoints,
                                 &(rhoOutQuadsXC[0]),
                                 &exchValRhoOut[0]);
                      xc_lda_exc(&(dftPtr->funcC),
                                 numQuadPoints,
                                 &(rhoOutQuadsXC[0]),
                                 &corrValRhoOut[0]);
                      xc_lda_vxc(&(dftPtr->funcX),
                                 numQuadPoints,
                                 &(rhoOutQuadsXC[0]),
                                 &exchPotValRhoOut[0]);
                      xc_lda_vxc(&(dftPtr->funcC),
                                 numQuadPoints,
                                 &(rhoOutQuadsXC[0]),
                                 &corrPotValRhoOut[0]);
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          excQuads[q][iSubCell] =
                            exchValRhoOut[q] + corrValRhoOut[q];
                          vxcRhoOutSpin0Quads[q][iSubCell] =
                            exchPotValRhoOut[2 * q] + corrPotValRhoOut[2 * q];
                          vxcRhoOutSpin1Quads[q][iSubCell] =
                            exchPotValRhoOut[2 * q + 1] +
                            corrPotValRhoOut[2 * q + 1];
                        }
                    }

                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      if (dftParameters::nonLinearCoreCorrection == true)
                        {
                          const std::vector<double> &temp1 =
                            gradRhoCoreValues.find(subCellId)->second;
                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            for (unsigned int idim = 0; idim < 3; idim++)
                              gradRhoCoreQuads[q][idim][iSubCell] =
                                temp1[3 * q + idim] / 2.0;

                          if (dftParameters::xcFamilyType == "GGA")
                            {
                              const std::vector<double> &temp2 =
                                hessianRhoCoreValues.find(subCellId)->second;
                              for (unsigned int q = 0; q < numQuadPoints; ++q)
                                for (unsigned int idim = 0; idim < 3; ++idim)
                                  for (unsigned int jdim = 0; jdim < 3; ++jdim)
                                    hessianRhoCoreQuads
                                      [q][idim][jdim][iSubCell] =
                                        temp2[9 * q + 3 * idim + jdim] / 2.0;
                            }
                        }
                    }

                } // subcell loop

              if (dftParameters::nonLinearCoreCorrection)
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
                    dftParameters::xcFamilyType == "GGA");
                }

              for (unsigned int q = 0; q < numQuadPoints; ++q)
                {
                  const VectorizedArray<double> phiTot_q = phiTotRhoOutQuads[q];

                  Tensor<2, 3, VectorizedArray<double>> E =
                    eshelbyTensorSP::getELocXcEshelbyTensor(
                      rhoXCQuadsVect[q],
                      gradRhoSpin0QuadsVect[q],
                      gradRhoSpin1QuadsVect[q],
                      excQuads[q],
                      derExchCorrEnergyWithGradRhoOutSpin0Quads[q],
                      derExchCorrEnergyWithGradRhoOutSpin1Quads[q]);

                  Tensor<1, 3, VectorizedArray<double>> F = zeroTensor3;

                  if (dftParameters::nonLinearCoreCorrection)
                    F += eshelbyTensorSP::getFNonlinearCoreCorrection(
                      vxcRhoOutSpin0Quads[q],
                      vxcRhoOutSpin1Quads[q],
                      derExchCorrEnergyWithGradRhoOutSpin0Quads[q],
                      derExchCorrEnergyWithGradRhoOutSpin1Quads[q],
                      gradRhoCoreQuads[q],
                      hessianRhoCoreQuads[q],
                      dftParameters::xcFamilyType == "GGA");

                  forceEval.submit_value(F, q);
                  forceEval.submit_gradient(E, q);
                } // quad point loop

              forceEval.integrate(true, true);
              forceEval.distribute_local_to_global(
                d_configForceVectorLinFE); // also takes care of constraints
            }

          if (dftParameters::nonLinearCoreCorrection)
            {
              if (dftParameters::floatingNuclearCharges)
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
          dealii::AlignedVector<VectorizedArray<double>>
            shadowKSRhoMinMinusRhoQuads(numQuadPoints,
                                        make_vectorized_array(0.0));
          dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>>
            shadowKSGradRhoMinMinusGradRhoQuads(numQuadPoints, zeroTensor3);
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
              std::fill(shadowKSRhoMinMinusRhoQuads.begin(),
                        shadowKSRhoMinMinusRhoQuads.end(),
                        make_vectorized_array(0.0));
              std::fill(shadowKSGradRhoMinMinusGradRhoQuads.begin(),
                        shadowKSGradRhoMinMinusGradRhoQuads.end(),
                        zeroTensor3);
              std::fill(gradRhoQuads.begin(), gradRhoQuads.end(), zeroTensor3);
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
                  subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
                  dealii::CellId subCellId = subCellPtr->id();

                  const std::vector<double> &temp1 =
                    rhoOutValues.find(subCellId)->second;
                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      rhoOutQuadsXC[q]        = temp1[q];
                      rhoQuads[q][iSubCell]   = temp1[q];
                      rhoXCQuads[q][iSubCell] = temp1[q];
                    }

                  if (dftParameters::nonLinearCoreCorrection)
                    {
                      const std::vector<double> &temp2 =
                        rhoCoreValues.find(subCellId)->second;
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          rhoOutQuadsXC[q] += temp2[q];
                          rhoXCQuads[q][iSubCell] += temp2[q];
                        }
                    }

                  if (dftParameters::xcFamilyType == "GGA")
                    {
                      const std::vector<double> &temp3 =
                        gradRhoOutValues.find(subCellId)->second;
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        for (unsigned int idim = 0; idim < 3; idim++)
                          {
                            gradRhoOutQuadsXC[q][idim] = temp3[3 * q + idim];
                            gradRhoQuads[q][idim][iSubCell] =
                              temp3[3 * q + idim];
                          }

                      if (dftParameters::nonLinearCoreCorrection)
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

                  if (dftParameters::xcFamilyType == "GGA")
                    {
                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        sigmaValRhoOut[q] = gradRhoOutQuadsXC[q].norm_square();

                      xc_gga_exc_vxc(&(dftPtr->funcX),
                                     numQuadPoints,
                                     &(rhoOutQuadsXC[0]),
                                     &sigmaValRhoOut[0],
                                     &exchValRhoOut[0],
                                     &derExchEnergyWithDensityValRhoOut[0],
                                     &derExchEnergyWithSigmaRhoOut[0]);
                      xc_gga_exc_vxc(&(dftPtr->funcC),
                                     numQuadPoints,
                                     &(rhoOutQuadsXC[0]),
                                     &sigmaValRhoOut[0],
                                     &corrValRhoOut[0],
                                     &derCorrEnergyWithDensityValRhoOut[0],
                                     &derCorrEnergyWithSigmaRhoOut[0]);
                      if (shadowPotentialForce)
                        {
                          xc_gga_fxc(&(dftPtr->funcX),
                                     numQuadPoints,
                                     &(rhoOutQuadsXC[0]),
                                     &sigmaValRhoOut[0],
                                     &derVxWithRhoOut[0],
                                     &derVxWithSigmaRhoOut[0],
                                     &der2ExWithSigmaRhoOut[0]);
                          xc_gga_fxc(&(dftPtr->funcC),
                                     numQuadPoints,
                                     &(rhoOutQuadsXC[0]),
                                     &sigmaValRhoOut[0],
                                     &derVcWithRhoOut[0],
                                     &derVcWithSigmaRhoOut[0],
                                     &der2EcWithSigmaRhoOut[0]);
                        }

                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          excQuads[q][iSubCell] =
                            exchValRhoOut[q] + corrValRhoOut[q];
                          vxcRhoOutQuads[q][iSubCell] =
                            derExchEnergyWithDensityValRhoOut[q] +
                            derCorrEnergyWithDensityValRhoOut[q];

                          if (shadowPotentialForce)
                            derVxcWithRhoQuads[q][iSubCell] =
                              derVxWithRhoOut[q] + derVcWithRhoOut[q];

                          for (unsigned int idim = 0; idim < 3; idim++)
                            {
                              derExchCorrEnergyWithGradRhoOutQuads
                                [q][idim][iSubCell] =
                                  2.0 *
                                  (derExchEnergyWithSigmaRhoOut[q] +
                                   derCorrEnergyWithSigmaRhoOut[q]) *
                                  gradRhoOutQuadsXC[q][idim];

                              if (shadowPotentialForce)
                                {
                                  derVxcWithGradRhoQuads[q][idim][iSubCell] =
                                    2.0 *
                                    (derVxWithSigmaRhoOut[q] +
                                     derVcWithSigmaRhoOut[q]) *
                                    gradRhoOutQuadsXC[q][idim];

                                  for (unsigned int jdim = 0; jdim < 3; jdim++)
                                    {
                                      if (idim == jdim)
                                        der2ExcWithGradRhoQuads
                                          [q][idim][idim][iSubCell] =
                                            2.0 *
                                              (derExchEnergyWithSigmaRhoOut[q] +
                                               derCorrEnergyWithSigmaRhoOut
                                                 [q]) +
                                            4.0 *
                                              (der2ExWithSigmaRhoOut[q] +
                                               der2EcWithSigmaRhoOut[q]) *
                                              gradRhoOutQuadsXC[q][idim] *
                                              gradRhoOutQuadsXC[q][idim];
                                      else
                                        der2ExcWithGradRhoQuads
                                          [q][idim][jdim][iSubCell] =
                                            4.0 *
                                            (der2ExWithSigmaRhoOut[q] +
                                             der2EcWithSigmaRhoOut[q]) *
                                            gradRhoOutQuadsXC[q][idim] *
                                            gradRhoOutQuadsXC[q][jdim];
                                    }
                                }
                            }
                        }
                    }
                  else
                    {
                      xc_lda_exc(&(dftPtr->funcX),
                                 numQuadPoints,
                                 &(rhoOutQuadsXC[0]),
                                 &exchValRhoOut[0]);
                      xc_lda_exc(&(dftPtr->funcC),
                                 numQuadPoints,
                                 &(rhoOutQuadsXC[0]),
                                 &corrValRhoOut[0]);
                      xc_lda_vxc(&(dftPtr->funcX),
                                 numQuadPoints,
                                 &(rhoOutQuadsXC[0]),
                                 &exchPotValRhoOut[0]);
                      xc_lda_vxc(&(dftPtr->funcC),
                                 numQuadPoints,
                                 &(rhoOutQuadsXC[0]),
                                 &corrPotValRhoOut[0]);

                      if (shadowPotentialForce)
                        {
                          xc_lda_fxc(&(dftPtr->funcX),
                                     numQuadPoints,
                                     &(rhoOutQuadsXC[0]),
                                     &derVxWithRhoOut[0]);
                          xc_lda_fxc(&(dftPtr->funcC),
                                     numQuadPoints,
                                     &(rhoOutQuadsXC[0]),
                                     &derVcWithRhoOut[0]);
                        }

                      for (unsigned int q = 0; q < numQuadPoints; ++q)
                        {
                          excQuads[q][iSubCell] =
                            exchValRhoOut[q] + corrValRhoOut[q];
                          vxcRhoOutQuads[q][iSubCell] =
                            exchPotValRhoOut[q] + corrPotValRhoOut[q];

                          if (shadowPotentialForce)
                            derVxcWithRhoQuads[q][iSubCell] =
                              derVxWithRhoOut[q] + derVcWithRhoOut[q];
                        }
                    }

                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      if (dftParameters::nonLinearCoreCorrection == true)
                        {
                          const std::vector<double> &temp1 =
                            gradRhoCoreValues.find(subCellId)->second;
                          for (unsigned int q = 0; q < numQuadPoints; ++q)
                            for (unsigned int idim = 0; idim < 3; idim++)
                              gradRhoCoreQuads[q][idim][iSubCell] =
                                temp1[3 * q + idim];

                          if (dftParameters::xcFamilyType == "GGA")
                            {
                              const std::vector<double> &temp2 =
                                hessianRhoCoreValues.find(subCellId)->second;
                              for (unsigned int q = 0; q < numQuadPoints; ++q)
                                for (unsigned int idim = 0; idim < 3; ++idim)
                                  for (unsigned int jdim = 0; jdim < 3; ++jdim)
                                    hessianRhoCoreQuads[q][idim][jdim]
                                                       [iSubCell] =
                                                         temp2[9 * q +
                                                               3 * idim + jdim];
                            }
                        }

                      if (shadowPotentialForce)
                        {
                          shadowKSRhoMinMinusRhoQuads[q][iSubCell] =
                            shadowKSRhoMinValues.find(subCellId)->second[q] -
                            rhoQuads[q][iSubCell];

                          // for (unsigned int idim=0; idim<3; idim++)
                          //	gradRhoAtomsQuads[q][idim][iSubCell]=dftPtr->d_gradRhoAtomsValues.find(subCellId)->second[3*q+idim];

                          if (dftParameters::xcFamilyType == "GGA")
                            {
                              /*
                              for (unsigned int idim=0; idim<3; idim++)
                                for (unsigned int jdim=0; jdim<3; jdim++)
                                  hessianRhoAtomsQuads[q][idim][jdim][iSubCell]=dftPtr->d_hessianRhoAtomsValues.find(subCellId)->second[9*q+3*idim+jdim];
                              */
                              for (unsigned int idim = 0; idim < 3; idim++)
                                shadowKSGradRhoMinMinusGradRhoQuads
                                  [q][idim][iSubCell] =
                                    shadowKSGradRhoMinValues.find(subCellId)
                                      ->second[3 * q + idim] -
                                    gradRhoQuads[q][idim][iSubCell];
                            }
                        }
                    }
                } // subcell loop

              if (dftParameters::nonLinearCoreCorrection)
                {
                  FNonlinearCoreCorrectionGammaAtomsElementalContribution(
                    forceContributionGradRhoNonlinearCoreCorrectionGammaAtoms,
                    forceEval,
                    matrixFreeData,
                    cell,
                    vxcRhoOutQuads,
                    gradRhoCoreAtoms);


                  if (dftParameters::xcFamilyType == "GGA")
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
                  const VectorizedArray<double> phiTot_q = phiTotRhoOutQuads[q];

                  if (shadowPotentialForce)
                    {
                      derVxcWithRhoTimesRhoDiffQuads[q] =
                        derVxcWithRhoQuads[q] * shadowKSRhoMinMinusRhoQuads[q];
                      // phiRhoMinMinusApproxRhoQuads[q]=
                      // phiTotEval2.get_value(q);
                    }


                  Tensor<2, 3, VectorizedArray<double>> E =
                    eshelbyTensor::getELocXcEshelbyTensor(
                      rhoXCQuads[q],
                      gradRhoQuads[q],
                      excQuads[q],
                      derExchCorrEnergyWithGradRhoOutQuads[q]);

                  /*
                  if (shadowPotentialForce)
                    E+=eshelbyTensor::getShadowPotentialForceRhoDiffXcEshelbyTensor
                      (shadowKSRhoMinMinusRhoQuads[q],
                       shadowKSGradRhoMinMinusGradRhoQuads[q],
                       gradRhoQuads[q],
                       vxcRhoOutQuads[q],
                       derVxcWithGradRhoOutQuads[q],
                       derExchCorrEnergyWithGradRhoOutQuads[q],
                       der2ExcWithGradRhoOutQuads[q]);
                  */

                  Tensor<1, 3, VectorizedArray<double>> F = zeroTensor3;

                  if (shadowPotentialForce && dftParameters::useAtomicRhoXLBOMD)
                    {
                      /*
                      F+=gradRhoAtomsQuads[q]*(derVxcWithRhoOutTimesRhoDiffQuads[q]+phiRhoMinMinusApproxRhoQuads[q]);

                      if(dftParameters::xcFamilyType=="GGA")
                      {
                        F+=shadowKSGradRhoMinMinusGradRhoQuads[q]*der2ExcWithGradRhoOutQuads[q]*hessianRhoAtomsQuads[q];
                        F+=shadowKSGradRhoMinMinusGradRhoQuads[q]*outer_product(derVxcWithGradRhoOutQuads[q],gradRhoAtomsQuads[q]);
                        F+=shadowKSRhoMinMinusRhoQuads[q]*derVxcWithGradRhoOutQuads[q]*hessianRhoAtomsQuads[q];
                      }
                      */
                    }

                  if (dftParameters::nonLinearCoreCorrection)
                    {
                      F += eshelbyTensor::getFNonlinearCoreCorrection(
                        vxcRhoOutQuads[q], gradRhoCoreQuads[q]);

                      if (dftParameters::xcFamilyType == "GGA")
                        F += eshelbyTensor::getFNonlinearCoreCorrection(
                          derExchCorrEnergyWithGradRhoOutQuads[q],
                          hessianRhoCoreQuads[q]);
                    }

                  forceEval.submit_value(F, q);
                  forceEval.submit_gradient(E, q);
                } // quad point loop


              if (shadowPotentialForce &&
                  (dftParameters::useAtomicRhoXLBOMD ||
                   dftParameters::nonLinearCoreCorrection))
                {
                  FShadowLocalGammaAtomsElementalContributionElectronic(
                    forceContributionShadowLocalGammaAtoms,
                    forceEval,
                    matrixFreeData,
                    cell,
                    derVxcWithRhoTimesRhoDiffQuads,
                    dftPtr->d_gradRhoAtomsValuesSeparate,
                    der2ExcWithGradRhoQuads,
                    derVxcWithGradRhoQuads,
                    shadowKSRhoMinMinusRhoQuads,
                    shadowKSGradRhoMinMinusGradRhoQuads,
                    dftPtr->d_hessianRhoAtomsValuesSeparate,
                    gradRhoCoreAtoms,
                    hessianRhoCoreAtoms,
                    dftParameters::useAtomicRhoXLBOMD,
                    dftParameters::xcFamilyType == "GGA",
                    dftParameters::nonLinearCoreCorrection);
                }

              forceEval.integrate(true, true);
              forceEval.distribute_local_to_global(
                d_configForceVectorLinFE); // also takes care of constraints
            }                              // cell loop


          if (dftParameters::nonLinearCoreCorrection)
            {
              if (dftParameters::floatingNuclearCharges)
                {
                  accumulateForceContributionGammaAtomsFloating(
                    forceContributionGradRhoNonlinearCoreCorrectionGammaAtoms,
                    d_forceAtomsFloating);

                  if (dftParameters::xcFamilyType == "GGA")
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

                  if (dftParameters::xcFamilyType == "GGA")
                    distributeForceContributionFPSPLocalGammaAtoms(
                      forceContributionHessianRhoNonlinearCoreCorrectionGammaAtoms,
                      d_atomsForceDofs,
                      d_constraintsNoneForce,
                      d_configForceVectorLinFE);
                }
            }

          if (shadowPotentialForce && (dftParameters::useAtomicRhoXLBOMD ||
                                       dftParameters::nonLinearCoreCorrection))
            {
              if (dftParameters::floatingNuclearCharges)
                {
                  accumulateForceContributionGammaAtomsFloating(
                    forceContributionShadowLocalGammaAtoms,
                    d_forceAtomsFloating);
                }
              else
                distributeForceContributionFPSPLocalGammaAtoms(
                  forceContributionShadowLocalGammaAtoms,
                  d_atomsForceDofs,
                  d_constraintsNoneForce,
                  d_configForceVectorLinFE);
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

  MPI_Barrier(MPI_COMM_WORLD);
  enowfc_time = MPI_Wtime() - enowfc_time;

  forcetotal_time = MPI_Wtime() - forcetotal_time;

  if (this_process == 0 && dftParameters::verbosity >= 4)
    std::cout
      << "Total time for configurational force computation except Eself contribution: "
      << forcetotal_time << std::endl;

  if (dftParameters::verbosity >= 4)
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
    update_values | update_gradients | update_quadrature_points);

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

  for (unsigned int cell = 0; cell < matrixFreeDataElectro.n_macro_cells();
       ++cell)
    {
      std::set<unsigned int> nonTrivialSmearedChargeAtomIdsMacroCell;

      const unsigned int numSubCells =
        matrixFreeDataElectro.n_components_filled(cell);
      if (dftParameters::smearedNuclearCharges)
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

      if (dftParameters::smearedNuclearCharges &&
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
          subCellPtr = matrixFreeDataElectro.get_cell_iterator(cell, iSubCell);
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

          if (dftParameters::isPseudopotential ||
              dftParameters::smearedNuclearCharges)
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

          if (dftParameters::smearedNuclearCharges &&
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

      if (dftParameters::isPseudopotential ||
          dftParameters::smearedNuclearCharges)
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
          VectorizedArray<double> phiExtElectro_q = make_vectorized_array(0.0);
          Tensor<1, 3, VectorizedArray<double>> gradPhiTotElectro_q =
            phiTotEvalElectro.get_gradient(q);

          Tensor<2, 3, VectorizedArray<double>> E =
            eshelbyTensor::getEElectroEshelbyTensor(phiTotElectro_q,
                                                    gradPhiTotElectro_q,
                                                    rhoQuadsElectro[q]);
          if (shadowPotentialForce)
            {
              /*
              VectorizedArray<double>
              identityTensorFactor=shadowKSRhoMinMinusRhoQuadsElectro[q]*phiTotElectro_q;

              E[0][0]+=identityTensorFactor;
              E[1][1]+=identityTensorFactor;
              E[2][2]+=identityTensorFactor;
              */
            }

          Tensor<1, 3, VectorizedArray<double>> F = zeroTensor;

          if (shadowPotentialForce)
            {
              /*
              VectorizedArray<double> phiRhoMinusApproxRho_q
              =phiTotEvalElectro2.get_value(q);
              Tensor<1,3,VectorizedArray<double> >
              gradPhiRhoMinusApproxRho_q =phiTotEvalElectro2.get_gradient(q);
              VectorizedArray<double>
              identityTensorFactor=make_vectorized_array(-1.0/(4.0*M_PI))*scalar_product(gradPhiRhoMinusApproxRho_q,gradPhiTotElectro_q)+phiRhoMinusApproxRho_q*rhoQuadsElectro[q];
              E+=
              (outer_product(gradPhiRhoMinusApproxRho_q,gradPhiTotElectro_q)+outer_product(gradPhiTotElectro_q,gradPhiRhoMinusApproxRho_q))*make_vectorized_array(1.0/(4.0*M_PI));
              E[0][0]+=identityTensorFactor;
              E[1][1]+=identityTensorFactor;
              E[2][2]+=identityTensorFactor;
              */
            }

          forceEvalElectro.submit_value(F, q);
          forceEvalElectro.submit_gradient(E, q);
        }

      if (dftParameters::isPseudopotential ||
          dftParameters::smearedNuclearCharges)
        for (unsigned int q = 0; q < numQuadPointsLpsp; ++q)
          {
            VectorizedArray<double> phiExtElectro_q =
              make_vectorized_array(0.0);

            Tensor<2, 3, VectorizedArray<double>> E = zeroTensor2;

            // FIXME: quadrature mismatch
            /*
            if (shadowPotentialForce)
              E+=eshelbyTensor::getELocPspEshelbyTensor
                (shadowKSRhoMinMinusRhoQuadsElectro[q],
                 pseudoVLocQuadsElectro[q],
                 phiExtElectro_q);
            */

            Tensor<1, 3, VectorizedArray<double>> F            = zeroTensor;
            Tensor<1, 3, VectorizedArray<double>> gradPhiExt_q = zeroTensor;
            F -= gradRhoQuadsElectroLpsp[q] * pseudoVLocQuadsElectro[q];

            // FIXME: quadrature mismatch
            /*
            if (shadowPotentialForce)
              F+=eshelbyTensor::getFPSPLocal
                (shadowKSRhoMinMinusRhoQuadsElectro[q],
                 gradPseudoVLocQuadsElectro[q],
                 gradPhiExt_q);
            */

            forceEvalElectroLpsp.submit_value(F, q);
            forceEvalElectroLpsp.submit_gradient(E, q);
          }

      forceEvalElectro.integrate(true, true);
      forceEvalElectro.distribute_local_to_global(
        d_configForceVectorLinFEElectro);

      if (dftParameters::isPseudopotential ||
          dftParameters::smearedNuclearCharges)
        {
          forceEvalElectroLpsp.integrate(true, true);
          forceEvalElectroLpsp.distribute_local_to_global(
            d_configForceVectorLinFEElectro);
        }

      if (dftParameters::smearedNuclearCharges &&
          nonTrivialSmearedChargeAtomIdsMacroCell.size() > 0)
        {
          phiTotEvalSmearedCharge.read_dof_values_plain(phiTotRhoOutElectro);
          phiTotEvalSmearedCharge.evaluate(false, true);

          for (unsigned int q = 0; q < numQuadPointsSmearedb; ++q)
            {
              gradPhiTotSmearedChargeQuads[q] =
                phiTotEvalSmearedCharge.get_gradient(q);

              Tensor<1, 3, VectorizedArray<double>> F = zeroTensor;
              F = -gradPhiTotSmearedChargeQuads[q] * smearedbQuads[q];

              forceEvalSmearedCharge.submit_value(F, q);
            }

          if (shadowPotentialForce)
            {
              phiTotEvalSmearedCharge.read_dof_values_plain(
                phiRhoMinusApproxRhoElectro);
              phiTotEvalSmearedCharge.evaluate(false, true);

              for (unsigned int q = 0; q < numQuadPointsSmearedb; ++q)
                {
                  gradPhiTotPlusPhiRhoMinusApproxRhoSmearedChargeQuads[q] =
                    phiTotEvalSmearedCharge.get_gradient(q) +
                    gradPhiTotSmearedChargeQuads[q];
                }
            }

          forceEvalSmearedCharge.integrate(true, false);
          forceEvalSmearedCharge.distribute_local_to_global(
            d_configForceVectorLinFEElectro);

          FPhiTotSmearedChargesGammaAtomsElementalContribution(
            forceContributionSmearedChargesGammaAtoms,
            forceEvalSmearedCharge,
            matrixFreeDataElectro,
            cell,
            shadowPotentialForce ?
              gradPhiTotPlusPhiRhoMinusApproxRhoSmearedChargeQuads :
              gradPhiTotSmearedChargeQuads,
            std::vector<unsigned int>(
              nonTrivialSmearedChargeAtomIdsMacroCell.begin(),
              nonTrivialSmearedChargeAtomIdsMacroCell.end()),
            dftPtr->d_bQuadAtomIdsAllAtoms,
            smearedbQuads);
        }

      if (shadowPotentialForce && dftParameters::useAtomicRhoXLBOMD)
        {
          phiTotEvalElectro.read_dof_values_plain(phiRhoMinusApproxRhoElectro);
          phiTotEvalElectro.evaluate(false, true);

          for (unsigned int q = 0; q < numQuadPoints; ++q)
            {
              gradPhiRhoMinusApproxRhoQuadsElectro[q] =
                phiTotEvalElectro.get_gradient(q);
            }

          FShadowLocalGammaAtomsElementalContributionElectrostatic(
            forceContributionShadowPotentialElectroGammaAtoms,
            forceEvalElectro,
            matrixFreeDataElectro,
            cell,
            gradPhiRhoMinusApproxRhoQuadsElectro,
            dftPtr->d_rhoAtomsValuesSeparate);
        }

    } // macro cell loop

  // add global FPSPLocal contribution due to Gamma(Rj) to the configurational
  // force vector
  if (dftParameters::isPseudopotential || dftParameters::smearedNuclearCharges)
    {
      if (dftParameters::floatingNuclearCharges)
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

  if (dftParameters::smearedNuclearCharges)
    {
      if (dftParameters::floatingNuclearCharges)
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

  if (shadowPotentialForce && dftParameters::useAtomicRhoXLBOMD)
    {
      if (dftParameters::floatingNuclearCharges)
        {
          accumulateForceContributionGammaAtomsFloating(
            forceContributionShadowPotentialElectroGammaAtoms,
            d_forceAtomsFloating);
        }
      else
        distributeForceContributionFPSPLocalGammaAtoms(
          forceContributionShadowPotentialElectroGammaAtoms,
          d_atomsForceDofsElectro,
          d_constraintsNoneForceElectro,
          d_configForceVectorLinFEElectro);
    }
}
