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
// @author Sambit Das
//
//(locally used function) compute Fnl contibution due to Gamma(Rj) for given set
// of cells
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::FnlGammaAtomsElementalContribution(
  std::map<unsigned int, std::vector<double>> &forceContributionFnlGammaAtoms,
  const MatrixFree<3, double> &                matrixFreeData,
  FEEvaluation<3, 1, C_num1DQuadNLPSP<FEOrder>() * C_numCopies1DQuadNLPSP(), 3>
    &                                           forceEvalNLP,
  const unsigned int                            cell,
  const std::map<dealii::CellId, unsigned int> &cellIdToCellNumberMap,
#ifdef USE_COMPLEX
  const std::vector<dataTypes::number>
    &projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened,
#endif
  const std::vector<dataTypes::number> &zetaDeltaVQuadsFlattened,
  const std::vector<dataTypes::number>
    &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened)
{
  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numSubCells   = matrixFreeData.n_components_filled(cell);
  const unsigned int numQuadPoints = forceEvalNLP.n_q_points;

  const unsigned int numNonLocalAtomsCurrentProcess =
    dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
  DoFHandler<3>::active_cell_iterator subCellPtr;

  Tensor<1, 3, VectorizedArray<double>> zeroTensor3;
  for (unsigned int idim = 0; idim < 3; idim++)
    {
      zeroTensor3[idim] = make_vectorized_array(0.0);
    }

  dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>> FVectQuads(
    numQuadPoints, zeroTensor3);

  for (int iAtom = 0; iAtom < numNonLocalAtomsCurrentProcess; ++iAtom)
    {
      //
      // get the global charge Id of the current nonlocal atom
      //
      const int nonLocalAtomId =
        dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const int globalChargeIdNonLocalAtom =
        dftPtr->d_nonLocalAtomGlobalChargeIds[nonLocalAtomId];


      // if map entry corresponding to current nonlocal atom id is empty,
      // initialize it to zero
      if (forceContributionFnlGammaAtoms.find(globalChargeIdNonLocalAtom) ==
          forceContributionFnlGammaAtoms.end())
        forceContributionFnlGammaAtoms[globalChargeIdNonLocalAtom] =
          std::vector<double>(3, 0.0);

      std::fill(FVectQuads.begin(), FVectQuads.end(), zeroTensor3);

      bool isPseudoWfcsAtomInMacroCell = false;
      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        {
          subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
          bool               isPseudoWfcsAtomInCell = false;
          const unsigned int elementId =
            cellIdToCellNumberMap.find(subCellPtr->id())->second;
          for (unsigned int i = 0;
               i <
               dftPtr
                 ->d_cellIdToNonlocalAtomIdsLocalCompactSupportMap[elementId]
                 .size();
               i++)
            if (dftPtr
                  ->d_cellIdToNonlocalAtomIdsLocalCompactSupportMap[elementId]
                                                                   [i] == iAtom)
              {
                isPseudoWfcsAtomInCell      = true;
                isPseudoWfcsAtomInMacroCell = true;
                break;
              }

          if (isPseudoWfcsAtomInCell)
            {
              for (unsigned int kPoint = 0;
                   kPoint < dftPtr->d_kPointWeights.size();
                   ++kPoint)
                {
                  std::vector<double> kcoord(3, 0.0);
                  kcoord[0] = dftPtr->d_kPointCoordinates[kPoint * 3 + 0];
                  kcoord[1] = dftPtr->d_kPointCoordinates[kPoint * 3 + 1];
                  kcoord[2] = dftPtr->d_kPointCoordinates[kPoint * 3 + 2];

                  const unsigned int startingPseudoWfcIdFlattened =
                    kPoint *
                      dftPtr
                        ->d_sumNonTrivialPseudoWfcsOverAllCellsZetaDeltaVQuads *
                      numQuadPoints +
                    dftPtr->d_nonTrivialPseudoWfcsCellStartIndexZetaDeltaVQuads
                        [elementId] *
                      numQuadPoints +
                    dftPtr
                        ->d_atomIdToNonTrivialPseudoWfcsCellStartIndexZetaDeltaVQuads
                          [iAtom][elementId] *
                      numQuadPoints;

                  const unsigned int numberPseudoWaveFunctions =
                    dftPtr->d_numberPseudoAtomicWaveFunctions[nonLocalAtomId];
                  // std::cout<<startingPseudoWfcIdFlattened <<std::endl;
                  std::vector<dataTypes::number> temp2(3);
                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      std::vector<dataTypes::number> F(3,
                                                       dataTypes::number(0.0));

                      for (unsigned int iPseudoWave = 0;
                           iPseudoWave < numberPseudoWaveFunctions;
                           ++iPseudoWave)
                        {
                          const dataTypes::number temp1 =
                            zetaDeltaVQuadsFlattened
                              [startingPseudoWfcIdFlattened +
                               iPseudoWave * numQuadPoints + q];
                          temp2[0] =
                            projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                              [startingPseudoWfcIdFlattened * 3 +
                               iPseudoWave * numQuadPoints * 3 + q * 3 + 0];
                          temp2[1] =
                            projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                              [startingPseudoWfcIdFlattened * 3 +
                               iPseudoWave * numQuadPoints * 3 + q * 3 + 1];
                          temp2[2] =
                            projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                              [startingPseudoWfcIdFlattened * 3 +
                               iPseudoWave * numQuadPoints * 3 + q * 3 + 2];
#ifdef USE_COMPLEX
                          const dataTypes::number temp3 =
                            projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened
                              [startingPseudoWfcIdFlattened +
                               iPseudoWave * numQuadPoints + q];
                          F[0] +=
                            2.0 * (temp1 * temp2[0] +
                                   temp1 * dataTypes::number(0.0, 1.0) * temp3 *
                                     dataTypes::number(kcoord[0]));
                          F[1] +=
                            2.0 * (temp1 * temp2[1] +
                                   temp1 * dataTypes::number(0.0, 1.0) * temp3 *
                                     dataTypes::number(kcoord[1]));
                          F[2] +=
                            2.0 * (temp1 * temp2[2] +
                                   temp1 * dataTypes::number(0.0, 1.0) * temp3 *
                                     dataTypes::number(kcoord[2]));
#else
                          F[0] += 2.0 * (temp1 * temp2[0]);
                          F[1] += 2.0 * (temp1 * temp2[1]);
                          F[2] += 2.0 * (temp1 * temp2[2]);
#endif
                        } // pseudowavefunctions loop

                      FVectQuads[q][0][iSubCell] +=
                        dftPtr->d_kPointWeights[kPoint] * 2.0 * dftfe::utils::realPart(F[0]);
                      FVectQuads[q][1][iSubCell] +=
                        dftPtr->d_kPointWeights[kPoint] * 2.0 * dftfe::utils::realPart(F[1]);
                      FVectQuads[q][2][iSubCell] +=
                        dftPtr->d_kPointWeights[kPoint] * 2.0 * dftfe::utils::realPart(F[2]);

                      // std::cout<<F[0] <<std::endl;
                      // std::cout<<F[1] <<std::endl;
                      // std::cout<<F[2] <<std::endl;
                    } // quad-loop
                }     // kpoint loop
            }         // non-trivial cell check
        }             // subcell loop

      if (isPseudoWfcsAtomInMacroCell)
        {
          for (unsigned int q = 0; q < numQuadPoints; ++q)
            forceEvalNLP.submit_value(FVectQuads[q], q);

          const Tensor<1, 3, VectorizedArray<double>>
            forceContributionFnlGammaiAtomCells =
              forceEvalNLP.integrate_value();

          for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
            for (unsigned int idim = 0; idim < 3; idim++)
              forceContributionFnlGammaAtoms[globalChargeIdNonLocalAtom]
                                            [idim] +=
                forceContributionFnlGammaiAtomCells[idim][iSubCell];
        }
    } // iAtom loop
}


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void forceClass<FEOrder, FEOrderElectro>::FnlGammaxElementalContribution(
  dealii::AlignedVector<Tensor<1, 3, VectorizedArray<double>>> &FVectQuads,
  const MatrixFree<3, double> &                                 matrixFreeData,
  const unsigned int                                            numQuadPoints,
  const unsigned int                                            cell,
  const std::map<dealii::CellId, unsigned int> &cellIdToCellNumberMap,
  const std::vector<dataTypes::number> &        zetaDeltaVQuadsFlattened,
  const std::vector<dataTypes::number>
    &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened)
{
  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numSubCells = matrixFreeData.n_components_filled(cell);

  const unsigned int numNonLocalAtomsCurrentProcess =
    dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
  DoFHandler<3>::active_cell_iterator subCellPtr;

  Tensor<1, 3, VectorizedArray<double>> zeroTensor3;
  for (unsigned int idim = 0; idim < 3; idim++)
    {
      zeroTensor3[idim] = make_vectorized_array(0.0);
    }
  std::fill(FVectQuads.begin(), FVectQuads.end(), zeroTensor3);

  for (int iAtom = 0; iAtom < numNonLocalAtomsCurrentProcess; ++iAtom)
    {
      //
      // get the global charge Id of the current nonlocal atom
      //
      const int nonLocalAtomId =
        dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const int globalChargeIdNonLocalAtom =
        dftPtr->d_nonLocalAtomGlobalChargeIds[nonLocalAtomId];



      for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
        {
          subCellPtr = matrixFreeData.get_cell_iterator(cell, iSubCell);
          bool               isPseudoWfcsAtomInCell = false;
          const unsigned int elementId =
            cellIdToCellNumberMap.find(subCellPtr->id())->second;
          for (unsigned int i = 0;
               i <
               dftPtr
                 ->d_cellIdToNonlocalAtomIdsLocalCompactSupportMap[elementId]
                 .size();
               i++)
            if (dftPtr
                  ->d_cellIdToNonlocalAtomIdsLocalCompactSupportMap[elementId]
                                                                   [i] == iAtom)
              {
                isPseudoWfcsAtomInCell = true;
                break;
              }

          if (isPseudoWfcsAtomInCell)
            {
              for (unsigned int kPoint = 0;
                   kPoint < dftPtr->d_kPointWeights.size();
                   ++kPoint)
                {
                  const unsigned int startingPseudoWfcIdFlattened =
                    kPoint *
                      dftPtr
                        ->d_sumNonTrivialPseudoWfcsOverAllCellsZetaDeltaVQuads *
                      numQuadPoints +
                    dftPtr->d_nonTrivialPseudoWfcsCellStartIndexZetaDeltaVQuads
                        [elementId] *
                      numQuadPoints +
                    dftPtr
                        ->d_atomIdToNonTrivialPseudoWfcsCellStartIndexZetaDeltaVQuads
                          [iAtom][elementId] *
                      numQuadPoints;

                  const unsigned int numberPseudoWaveFunctions =
                    dftPtr->d_numberPseudoAtomicWaveFunctions[nonLocalAtomId];
                  std::vector<dataTypes::number> temp2(3);
                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      std::vector<dataTypes::number> F(3,
                                                       dataTypes::number(0.0));

                      for (unsigned int iPseudoWave = 0;
                           iPseudoWave < numberPseudoWaveFunctions;
                           ++iPseudoWave)
                        {
                          const dataTypes::number temp1 =
                            zetaDeltaVQuadsFlattened
                              [startingPseudoWfcIdFlattened +
                               iPseudoWave * numQuadPoints + q];
                          temp2[0] =
                            projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                              [startingPseudoWfcIdFlattened * 3 +
                               iPseudoWave * numQuadPoints * 3 + q * 3 + 0];
                          temp2[1] =
                            projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                              [startingPseudoWfcIdFlattened * 3 +
                               iPseudoWave * numQuadPoints * 3 + q * 3 + 1];
                          temp2[2] =
                            projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                              [startingPseudoWfcIdFlattened * 3 +
                               iPseudoWave * numQuadPoints * 3 + q * 3 + 2];
                          F[0] -= 2.0 * (temp1 * temp2[0]);
                          F[1] -= 2.0 * (temp1 * temp2[1]);
                          F[2] -= 2.0 * (temp1 * temp2[2]);
                        } // pseudowavefunctions loop

                      FVectQuads[q][0][iSubCell] +=
                        dftPtr->d_kPointWeights[kPoint] * 2.0 * dftfe::utils::realPart(F[0]);
                      FVectQuads[q][1][iSubCell] +=
                        dftPtr->d_kPointWeights[kPoint] * 2.0 * dftfe::utils::realPart(F[1]);
                      FVectQuads[q][2][iSubCell] +=
                        dftPtr->d_kPointWeights[kPoint] * 2.0 * dftfe::utils::realPart(F[2]);
                    } // quad-loop
                }     // kpoint loop
            }         // non-trivial cell check
        }             // subcell loop
    }                 // iAtom loop
}

//(locally used function) accumulate and distribute Fnl contibution due to
// Gamma(Rj)
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
forceClass<FEOrder, FEOrderElectro>::distributeForceContributionFnlGammaAtoms(
  const std::map<unsigned int, std::vector<double>>
    &forceContributionFnlGammaAtoms)
{
  for (unsigned int iAtom = 0; iAtom < dftPtr->atomLocations.size(); iAtom++)
    {
      bool doesAtomIdExistOnLocallyOwnedNode = false;
      if (d_atomsForceDofs.find(
            std::pair<unsigned int, unsigned int>(iAtom, 0)) !=
          d_atomsForceDofs.end())
        doesAtomIdExistOnLocallyOwnedNode = true;

      std::vector<double> forceContributionFnlGammaiAtomGlobal(3);
      std::vector<double> forceContributionFnlGammaiAtomLocal(3, 0.0);

      if (forceContributionFnlGammaAtoms.find(iAtom) !=
          forceContributionFnlGammaAtoms.end())
        forceContributionFnlGammaiAtomLocal =
          forceContributionFnlGammaAtoms.find(iAtom)->second;
      // accumulate value
      MPI_Allreduce(&(forceContributionFnlGammaiAtomLocal[0]),
                    &(forceContributionFnlGammaiAtomGlobal[0]),
                    3,
                    MPI_DOUBLE,
                    MPI_SUM,
                    mpi_communicator);

      if (doesAtomIdExistOnLocallyOwnedNode)
        {
          std::vector<dealii::types::global_dof_index> forceLocalDofIndices(3);
          for (unsigned int idim = 0; idim < 3; idim++)
            forceLocalDofIndices[idim] =
              d_atomsForceDofs[std::pair<unsigned int, unsigned int>(iAtom,
                                                                     idim)];
#ifdef USE_COMPLEX
          d_constraintsNoneForce.distribute_local_to_global(
            forceContributionFnlGammaiAtomGlobal,
            forceLocalDofIndices,
            d_configForceVectorLinFEKPoints);
#else
          d_constraintsNoneForce.distribute_local_to_global(
            forceContributionFnlGammaiAtomGlobal,
            forceLocalDofIndices,
            d_configForceVectorLinFE);
#endif
        }
    }
}
