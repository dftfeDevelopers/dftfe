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

#include <force.h>
#include <dft.h>

namespace dftfe
{

template <unsigned int FEOrder, unsigned int FEOrderElectro>
void forceClass<FEOrder, FEOrderElectro>::stressEnlElementalContribution(
  dealii::Tensor<2, 3, double> &                        stressContribution,
  const dealii::MatrixFree<3, double> &                 matrixFreeData,
  const unsigned int                            numQuadPoints,
  const std::vector<double> &                   jxwQuadsSubCells,
  const unsigned int                            cell,
  const std::map<dealii::CellId, unsigned int> &cellIdToCellNumberMap,
  const std::vector<dataTypes::number> &zetalmDeltaVlProductDistImageAtoms,
#ifdef USE_COMPLEX
  const std::vector<dataTypes::number>
    &projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened,
#endif
  const std::vector<dataTypes::number>
    &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened,
  const bool isSpinPolarized)
{
  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numSubCells = matrixFreeData.n_components_filled(cell);

  const double spinPolarizedFactor = isSpinPolarized ? 0.5 : 1.0;

  const unsigned int numNonLocalAtomsCurrentProcess =
    dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
  dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

  dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor3;
  for (unsigned int idim = 0; idim < 3; idim++)
    {
      zeroTensor3[idim] = dealii::make_vectorized_array(0.0);
    }

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
                  std::vector<dataTypes::number> temp1(3);
                  std::vector<dataTypes::number> temp2(3);
                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      // row major storage
                      std::vector<dataTypes::number> E(9,
                                                       dataTypes::number(0.0));

                      for (unsigned int iPseudoWave = 0;
                           iPseudoWave < numberPseudoWaveFunctions;
                           ++iPseudoWave)
                        {
                          temp1[0] = zetalmDeltaVlProductDistImageAtoms
                            [startingPseudoWfcIdFlattened * 3 +
                             iPseudoWave * numQuadPoints * 3 + q * 3 + 0];
                          temp1[1] = zetalmDeltaVlProductDistImageAtoms
                            [startingPseudoWfcIdFlattened * 3 +
                             iPseudoWave * numQuadPoints * 3 + q * 3 + 1];
                          temp1[2] = zetalmDeltaVlProductDistImageAtoms
                            [startingPseudoWfcIdFlattened * 3 +
                             iPseudoWave * numQuadPoints * 3 + q * 3 + 2];

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
                          for (unsigned int idim = 0; idim < 3; ++idim)
                            for (unsigned int jdim = 0; jdim < 3; ++jdim)
                              E[idim * 3 + jdim] -=
                                2.0 * (temp2[idim] * temp1[jdim] -
                                       dataTypes::number(0.0, 1.0) * temp3 *
                                         temp1[idim] *
                                         dataTypes::number(kcoord[jdim]));
#else
                          for (unsigned int idim = 0; idim < 3; ++idim)
                            for (unsigned int jdim = 0; jdim < 3; ++jdim)
                              E[idim * 3 + jdim] -=
                                2.0 * (temp2[idim] * temp1[jdim]);
#endif
                        } // pseudowavefunctions loop

                      const double factor =
                        spinPolarizedFactor * dftPtr->d_kPointWeights[kPoint] *
                        jxwQuadsSubCells[iSubCell * numQuadPoints + q];

                      for (unsigned int idim = 0; idim < 3; ++idim)
                        for (unsigned int jdim = 0; jdim < 3; ++jdim)
                          stressContribution[idim][jdim] +=
                            factor * 2.0 *
                            dftfe::utils::realPart(E[idim * 3 + jdim]);
                    } // quad-loop
                }     // kpoint loop
            }         // non-trivial cell check
        }             // subcell loop
    }                 // iAtom loop
}
#include "../force.inst.cc"
} // namespace dftfe
