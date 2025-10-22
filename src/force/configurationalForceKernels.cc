// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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


#include <configurationalForceKernels.h>

DFTFE_CREATE_KERNEL(
  void,
  computeELocWfcEshelbyTensorContributions,
  {
    const dftfe::uInt numberEntries = numContiguousBlocks * contiguousBlockSize;

    for (dftfe::uInt index = globalThreadId; index < numberEntries;
         index += nThreadsPerBlock * nThreadBlock)
      {
        const dftfe::uInt blockIndex = index / contiguousBlockSize;
        const dftfe::uInt intraBlockIndex =
          index - blockIndex * contiguousBlockSize;
        const dftfe::uInt blockIndex2  = blockIndex / 9;
        const dftfe::uInt eshelbyIndex = blockIndex - 9 * blockIndex2;
        const dftfe::uInt cellIndex    = blockIndex2 / nQuadsPerCell;
        const dftfe::uInt quadId = blockIndex2 - cellIndex * nQuadsPerCell;
        const dftfe::uInt tempIndex =
          (cellIndex)*nQuadsPerCell * contiguousBlockSize +
          quadId * contiguousBlockSize + intraBlockIndex;
        const dftfe::uInt tempIndex2 =
          (cellIndex)*nQuadsPerCell * contiguousBlockSize * 3 +
          quadId * contiguousBlockSize + intraBlockIndex;
        const dftfe::uInt tempIndex3 = (cellIndex)*nQuadsPerCell + quadId;

        const double psi      = psiQuadValues[tempIndex];
        const double gradPsiX = gradPsiQuadValues[tempIndex2];
        const double gradPsiY =
          gradPsiQuadValues[tempIndex2 + nQuadsPerCell * contiguousBlockSize];
        const double gradPsiZ =
          gradPsiQuadValues[tempIndex2 +
                            2 * nQuadsPerCell * contiguousBlockSize];
        const double eigenValue    = eigenValues[intraBlockIndex];
        const double partOcc       = partialOccupancies[intraBlockIndex];
        double       pdexcTauValue = 0.0;
        if (isTauMGGA)
          pdexcTauValue = pdexTauLocallyOwnedCellsBlock[tempIndex3] +
                          pdecTauLocallyOwnedCellsBlock[tempIndex3];

        const double identityFactor =
          0.5 * partOcc *
            (gradPsiX * gradPsiX + gradPsiY * gradPsiY + gradPsiZ * gradPsiZ) -
          partOcc * eigenValue * psi * psi;

        if (eshelbyIndex == 0)
          eshelbyTensor[index] =
            -partOcc * (1 + pdexcTauValue) * gradPsiX * gradPsiX +
            identityFactor;
        else if (eshelbyIndex == 1)
          eshelbyTensor[index] =
            -partOcc * (1 + pdexcTauValue) * gradPsiX * gradPsiY;
        else if (eshelbyIndex == 2)
          eshelbyTensor[index] =
            -partOcc * (1 + pdexcTauValue) * gradPsiX * gradPsiZ;
        else if (eshelbyIndex == 3)
          eshelbyTensor[index] =
            -partOcc * (1 + pdexcTauValue) * gradPsiY * gradPsiX;
        else if (eshelbyIndex == 4)
          eshelbyTensor[index] =
            -partOcc * (1 + pdexcTauValue) * gradPsiY * gradPsiY +
            identityFactor;
        else if (eshelbyIndex == 5)
          eshelbyTensor[index] =
            -partOcc * (1 + pdexcTauValue) * gradPsiY * gradPsiZ;
        else if (eshelbyIndex == 6)
          eshelbyTensor[index] =
            -partOcc * (1 + pdexcTauValue) * gradPsiZ * gradPsiX;
        else if (eshelbyIndex == 7)
          eshelbyTensor[index] =
            -partOcc * (1 + pdexcTauValue) * gradPsiZ * gradPsiY;
        else if (eshelbyIndex == 8)
          eshelbyTensor[index] =
            -partOcc * (1 + pdexcTauValue) * gradPsiZ * gradPsiZ +
            identityFactor;
      }
  },
  const dftfe::uInt contiguousBlockSize,
  const dftfe::uInt numContiguousBlocks,
  const dftfe::uInt nQuadsPerCell,
  const double     *psiQuadValues,
  const double     *gradPsiQuadValues,
  const double     *eigenValues,
  const double     *partialOccupancies,
  double           *eshelbyTensor,
  const bool        isTauMGGA,
  double           *pdexTauLocallyOwnedCellsBlock,
  double           *pdecTauLocallyOwnedCellsBlock);

DFTFE_CREATE_KERNEL(
  void,
  computeELocWfcEshelbyTensorContributions,
  {
    const dftfe::uInt numberEntries = numContiguousBlocks * contiguousBlockSize;

    for (dftfe::uInt index = globalThreadId; index < numberEntries;
         index += nThreadsPerBlock * nThreadBlock)
      {
        const dftfe::uInt blockIndex = index / contiguousBlockSize;
        const dftfe::uInt intraBlockIndex =
          index - blockIndex * contiguousBlockSize;
        const dftfe::uInt blockIndex2  = blockIndex / 9;
        const dftfe::uInt eshelbyIndex = blockIndex - 9 * blockIndex2;
        const dftfe::uInt cellIndex    = blockIndex2 / nQuadsPerCell;
        const dftfe::uInt quadId = blockIndex2 - cellIndex * nQuadsPerCell;
        const dftfe::uInt tempIndex =
          (cellIndex)*nQuadsPerCell * contiguousBlockSize +
          quadId * contiguousBlockSize + intraBlockIndex;
        const dftfe::uInt tempIndex2 =
          (cellIndex)*nQuadsPerCell * contiguousBlockSize * 3 +
          quadId * contiguousBlockSize + intraBlockIndex;
        const dftfe::uInt tempIndex3 = (cellIndex)*nQuadsPerCell + quadId;

        const dftfe::utils::deviceDoubleComplex psi = psiQuadValues[tempIndex];
        const dftfe::utils::deviceDoubleComplex psiConj =
          dftfe::utils::conj(psiQuadValues[tempIndex]);
        const dftfe::utils::deviceDoubleComplex gradPsiX =
          gradPsiQuadValues[tempIndex2];
        const dftfe::utils::deviceDoubleComplex gradPsiY =
          gradPsiQuadValues[tempIndex2 + nQuadsPerCell * contiguousBlockSize];
        const dftfe::utils::deviceDoubleComplex gradPsiZ =
          gradPsiQuadValues[tempIndex2 +
                            2 * nQuadsPerCell * contiguousBlockSize];
        const dftfe::utils::deviceDoubleComplex gradPsiXConj =
          dftfe::utils::conj(gradPsiQuadValues[tempIndex2]);
        const dftfe::utils::deviceDoubleComplex gradPsiYConj =
          dftfe::utils::conj(
            gradPsiQuadValues[tempIndex2 +
                              nQuadsPerCell * contiguousBlockSize]);
        const dftfe::utils::deviceDoubleComplex gradPsiZConj =
          dftfe::utils::conj(
            gradPsiQuadValues[tempIndex2 +
                              2 * nQuadsPerCell * contiguousBlockSize]);
        const double eigenValue    = eigenValues[intraBlockIndex];
        const double partOcc       = partialOccupancies[intraBlockIndex];
        double       pdexcTauValue = 0.0;
        if (isTauMGGA)
          pdexcTauValue = pdexTauLocallyOwnedCellsBlock[tempIndex3] +
                          pdecTauLocallyOwnedCellsBlock[tempIndex3];

        const double identityFactor =
          0.5 * partOcc *
          ((dftfe::utils::realPartDevice(
              dftfe::utils::mult(gradPsiXConj, gradPsiX)) +
            dftfe::utils::realPartDevice(
              dftfe::utils::mult(gradPsiYConj, gradPsiY)) +
            dftfe::utils::realPartDevice(
              dftfe::utils::mult(gradPsiZConj, gradPsiZ))) +
           2.0 * (kcoordx * dftfe::utils::imagPartDevice(
                              dftfe::utils::mult(psiConj, gradPsiX)) +
                  kcoordy * dftfe::utils::imagPartDevice(
                              dftfe::utils::mult(psiConj, gradPsiY)) +
                  kcoordz * dftfe::utils::imagPartDevice(
                              dftfe::utils::mult(psiConj, gradPsiZ))) +
           (kcoordx * kcoordx + kcoordy * kcoordy + kcoordz * kcoordz -
            2.0 * eigenValue) *
             dftfe::utils::realPartDevice(dftfe::utils::mult(psiConj, psi)));
        if (addEk)
          {
            if (eshelbyIndex == 0)
              eshelbyTensor[index] =
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(gradPsiXConj, gradPsiX)) +
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiX)) *
                  kcoordx -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiX)) *
                  kcoordx -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(psiConj, psi)) *
                  kcoordx * kcoordx +
                identityFactor;
            else if (eshelbyIndex == 1)
              eshelbyTensor[index] =
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(gradPsiXConj, gradPsiY)) +
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiX)) *
                  kcoordy -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiY)) *
                  kcoordx -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(psiConj, psi)) *
                  kcoordx * kcoordy;
            else if (eshelbyIndex == 2)
              eshelbyTensor[index] =
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(gradPsiXConj, gradPsiZ)) +
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiX)) *
                  kcoordz -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiZ)) *
                  kcoordx -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(psiConj, psi)) *
                  kcoordx * kcoordz;
            else if (eshelbyIndex == 3)
              eshelbyTensor[index] =
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(gradPsiYConj, gradPsiX)) +
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiY)) *
                  kcoordx -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiX)) *
                  kcoordy -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(psiConj, psi)) *
                  kcoordy * kcoordx;
            else if (eshelbyIndex == 4)
              eshelbyTensor[index] =
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(gradPsiYConj, gradPsiY)) +
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiY)) *
                  kcoordy -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiY)) *
                  kcoordy -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(psiConj, psi)) *
                  kcoordy * kcoordy +
                identityFactor;
            else if (eshelbyIndex == 5)
              eshelbyTensor[index] =
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(gradPsiYConj, gradPsiZ)) -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiY)) *
                  kcoordz -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiZ)) *
                  kcoordy -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(psiConj, psi)) *
                  kcoordy * kcoordz;
            else if (eshelbyIndex == 6)
              eshelbyTensor[index] =
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(gradPsiZConj, gradPsiX)) +
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiZ)) *
                  kcoordx -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiX)) *
                  kcoordz -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(psiConj, psi)) *
                  kcoordz * kcoordx;
            else if (eshelbyIndex == 7)
              eshelbyTensor[index] =
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(gradPsiZConj, gradPsiY)) -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiZ)) *
                  kcoordy -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiY)) *
                  kcoordz -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(psiConj, psi)) *
                  kcoordz * kcoordy;
            else if (eshelbyIndex == 8)
              eshelbyTensor[index] =
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(gradPsiZConj, gradPsiZ)) +
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiZ)) *
                  kcoordz -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiZ)) *
                  kcoordz -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(psiConj, psi)) *
                  kcoordz * kcoordz +
                identityFactor;
          }
        else
          {
            if (eshelbyIndex == 0)
              eshelbyTensor[index] =
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(gradPsiXConj, gradPsiX)) -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiX)) *
                  kcoordx +
                identityFactor;
            else if (eshelbyIndex == 1)
              eshelbyTensor[index] =
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(gradPsiXConj, gradPsiY)) -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiX)) *
                  kcoordy;
            else if (eshelbyIndex == 2)
              eshelbyTensor[index] =
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(gradPsiXConj, gradPsiZ)) -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiX)) *
                  kcoordz;
            else if (eshelbyIndex == 3)
              eshelbyTensor[index] =
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(gradPsiYConj, gradPsiX)) -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiY)) *
                  kcoordx;
            else if (eshelbyIndex == 4)
              eshelbyTensor[index] =
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(gradPsiYConj, gradPsiY)) -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiY)) *
                  kcoordy +
                identityFactor;
            else if (eshelbyIndex == 5)
              eshelbyTensor[index] =
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(gradPsiYConj, gradPsiZ)) -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiY)) *
                  kcoordz;
            else if (eshelbyIndex == 6)
              eshelbyTensor[index] =
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(gradPsiZConj, gradPsiX)) -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiZ)) *
                  kcoordx;
            else if (eshelbyIndex == 7)
              eshelbyTensor[index] =
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(gradPsiZConj, gradPsiY)) -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiZ)) *
                  kcoordy;
            else if (eshelbyIndex == 8)
              eshelbyTensor[index] =
                -partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::realPartDevice(
                    dftfe::utils::mult(gradPsiZConj, gradPsiZ)) -
                partOcc * (1 + pdexcTauValue) *
                  dftfe::utils::imagPartDevice(
                    dftfe::utils::mult(psiConj, gradPsiZ)) *
                  kcoordz +
                identityFactor;
          }
      }
  },
  const dftfe::uInt                        contiguousBlockSize,
  const dftfe::uInt                        numContiguousBlocks,
  const dftfe::uInt                        nQuadsPerCell,
  const dftfe::utils::deviceDoubleComplex *psiQuadValues,
  const dftfe::utils::deviceDoubleComplex *gradPsiQuadValues,
  const double                            *eigenValues,
  const double                            *partialOccupancies,
  const double                             kcoordx,
  const double                             kcoordy,
  const double                             kcoordz,
  double                                  *eshelbyTensor,
  const bool                               isTauMGGA,
  double                                  *pdexTauLocallyOwnedCellsBlock,
  double                                  *pdecTauLocallyOwnedCellsBlock,
  const bool                               addEk);

DFTFE_CREATE_KERNEL(
  void,
  nlpWfcContractionContributionDeviceKernel,
  {
    const dftfe::uInt numberEntries =
      totalNonTrivialPseudoWfcs * numQuadsNLP * numPsi;

    for (dftfe::uInt index = globalThreadId; index < numberEntries;
         index += nThreadsPerBlock * nThreadBlock)
      {
        const dftfe::uInt blockIndex  = index / numPsi;
        const dftfe::uInt wfcId       = index - blockIndex * numPsi;
        dftfe::uInt       pseudoWfcId = blockIndex / numQuadsNLP;
        const dftfe::uInt quadId      = blockIndex - pseudoWfcId * numQuadsNLP;
        pseudoWfcId += startingId;
        nlpContractionContribution[index] =
          gradPsiOrPsiQuadValuesNLP[nonTrivialIdToElemIdMap[pseudoWfcId] *
                                      numQuadsNLP * numPsi +
                                    quadId * numPsi + wfcId] *
          projectorKetTimesVectorPar
            [projecterKetTimesFlattenedVectorLocalIds[pseudoWfcId] * numPsi +
             wfcId];
      }
  },
  const dftfe::uInt  numPsi,
  const dftfe::uInt  numQuadsNLP,
  const dftfe::uInt  totalNonTrivialPseudoWfcs,
  const dftfe::uInt  startingId,
  const double      *projectorKetTimesVectorPar,
  const double      *gradPsiOrPsiQuadValuesNLP,
  const dftfe::uInt *nonTrivialIdToElemIdMap,
  const dftfe::uInt *projecterKetTimesFlattenedVectorLocalIds,
  double            *nlpContractionContribution);



DFTFE_CREATE_KERNEL(
  void,
  nlpWfcContractionContributionDeviceKernel,
  {
    const dftfe::uInt numberEntries =
      totalNonTrivialPseudoWfcs * numQuadsNLP * numPsi;

    for (dftfe::uInt index = globalThreadId; index < numberEntries;
         index += nThreadsPerBlock * nThreadBlock)
      {
        const dftfe::uInt blockIndex  = index / numPsi;
        const dftfe::uInt wfcId       = index - blockIndex * numPsi;
        dftfe::uInt       pseudoWfcId = blockIndex / numQuadsNLP;
        const dftfe::uInt quadId      = blockIndex - pseudoWfcId * numQuadsNLP;
        pseudoWfcId += startingId;

        const dftfe::utils::deviceDoubleComplex temp = dftfe::utils::mult(
          dftfe::utils::conj(
            gradPsiOrPsiQuadValuesNLP[nonTrivialIdToElemIdMap[pseudoWfcId] *
                                        numQuadsNLP * numPsi +
                                      quadId * numPsi + wfcId]),
          projectorKetTimesVectorPar
            [projecterKetTimesFlattenedVectorLocalIds[pseudoWfcId] * numPsi +
             wfcId]);
        nlpContractionContribution[index] =
          dftfe::utils::makeComplex(dftfe::utils::realPartDevice(temp),
                                    dftfe::utils::imagPartDevice(temp));
      }
  },
  const dftfe::uInt                        numPsi,
  const dftfe::uInt                        numQuadsNLP,
  const dftfe::uInt                        totalNonTrivialPseudoWfcs,
  const dftfe::uInt                        startingId,
  const dftfe::utils::deviceDoubleComplex *projectorKetTimesVectorPar,
  const dftfe::utils::deviceDoubleComplex *gradPsiOrPsiQuadValuesNLP,
  const dftfe::uInt                       *nonTrivialIdToElemIdMap,
  const dftfe::uInt                 *projecterKetTimesFlattenedVectorLocalIds,
  dftfe::utils::deviceDoubleComplex *nlpContractionContribution);

namespace dftfe
{

  void
  computeWavefuncEshelbyContributionLocal(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                             &BLASWrapperPtr,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
    const std::pair<dftfe::uInt, dftfe::uInt> vecRange,
    const dftfe::uInt                         nQuadsPerCell,
    const double                              kcoordx,
    const double                              kcoordy,
    const double                              kcoordz,
    double                                   *partialOccupVec,
    double                                   *eigenValuesVec,
    dataTypes::number                        *wfcQuadPointData,
    dataTypes::number                        *gradWfcQuadPointData,
    double                                   *eshelbyContributions,
    double                                   *eshelbyTensor,
    const bool                                floatingNuclearCharges,
    const bool                                isTauMGGA,
    double                                   *pdexTauLocallyOwnedCellsBlock,
    double                                   *pdecTauLocallyOwnedCellsBlock,
    const bool                                computeForce,
    const bool                                computeStress)
  {
    const dftfe::uInt cellsBlockSize = cellRange.second - cellRange.first;
    const dftfe::uInt wfcBlockSize   = vecRange.second - vecRange.first;

    DFTFE_LAUNCH_KERNEL(
      computeELocWfcEshelbyTensorContributions,
      (wfcBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
        dftfe::utils::DEVICE_BLOCK_SIZE * cellsBlockSize * nQuadsPerCell * 9,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      dftfe::utils::defaultStream,
      wfcBlockSize,
      cellsBlockSize * nQuadsPerCell * 9,
      nQuadsPerCell,
      dftfe::utils::makeDataTypeDeviceCompatible(wfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(gradWfcQuadPointData),
      eigenValuesVec,
      partialOccupVec,
#ifdef USE_COMPLEX
      kcoordx,
      kcoordy,
      kcoordz,
#endif
      eshelbyContributions,
      isTauMGGA,
      pdexTauLocallyOwnedCellsBlock,
      pdecTauLocallyOwnedCellsBlock
#ifdef USE_COMPLEX
      ,
      computeStress
#endif
    );
    const double scalarCoeffAlphaEshelby = 1.0;
    const double scalarCoeffBetaEshelby  = 0.0;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      onesVec(wfcBlockSize, 1.0);

    BLASWrapperPtr->xgemm('N',
                          'N',
                          1,
                          cellsBlockSize * nQuadsPerCell * 9,
                          wfcBlockSize,
                          &scalarCoeffAlphaEshelby,
                          onesVec.data(),
                          1,
                          eshelbyContributions,
                          wfcBlockSize,
                          &scalarCoeffBetaEshelby,
                          eshelbyTensor,
                          1);
  }

  void
  nlpWfcContractionContribution(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                            &BLASWrapperPtr,
    const dftfe::uInt        wfcBlockSize,
    const dftfe::uInt        blockSizeNlp,
    const dftfe::uInt        numQuadsNLP,
    const dftfe::uInt        startingIdNlp,
    const dataTypes::number *projectorKetTimesVectorPar,
    const dataTypes::number *gradPsiOrPsiQuadValuesNLP,
    const dftfe::uInt       *nonTrivialIdToElemIdMap,
    const dftfe::uInt       *projecterKetTimesFlattenedVectorLocalIds,
    dataTypes::number       *nlpContractionContribution)
  {
    DFTFE_LAUNCH_KERNEL(
      nlpWfcContractionContributionDeviceKernel,
      (wfcBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
        dftfe::utils::DEVICE_BLOCK_SIZE * numQuadsNLP * blockSizeNlp,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      dftfe::utils::defaultStream,
      wfcBlockSize,
      numQuadsNLP,
      blockSizeNlp,
      startingIdNlp,
      dftfe::utils::makeDataTypeDeviceCompatible(projectorKetTimesVectorPar),
      dftfe::utils::makeDataTypeDeviceCompatible(gradPsiOrPsiQuadValuesNLP),
      nonTrivialIdToElemIdMap,
      projecterKetTimesFlattenedVectorLocalIds,
      dftfe::utils::makeDataTypeDeviceCompatible(nlpContractionContribution));
  }

} // namespace dftfe
