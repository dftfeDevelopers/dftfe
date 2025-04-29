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
// @author Sambit Das
//

// source file for force related computations
#if defined(DFTFE_WITH_DEVICE)
#  include "dftfeDataTypes.h"
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceAPICalls.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <forceWfcContractionsDeviceKernels.h>

namespace dftfe
{
  namespace forceDeviceKernels
  {
    namespace
    {
      __global__ void
      computeELocWfcEshelbyTensorContributions(
        const dftfe::uInt contiguousBlockSize,
        const dftfe::uInt numContiguousBlocks,
        const dftfe::uInt numQuads,
        const double     *psiQuadValues,
        const double     *gradPsiQuadValues,
        const double     *eigenValues,
        const double     *partialOccupancies,
        double           *eshelbyTensor)
      {
        const dftfe::uInt globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const dftfe::uInt blockIndex = index / contiguousBlockSize;
            const dftfe::uInt intraBlockIndex =
              index - blockIndex * contiguousBlockSize;
            const dftfe::uInt blockIndex2  = blockIndex / 9;
            const dftfe::uInt eshelbyIndex = blockIndex - 9 * blockIndex2;
            const dftfe::uInt cellIndex    = blockIndex2 / numQuads;
            const dftfe::uInt quadId       = blockIndex2 - cellIndex * numQuads;
            const dftfe::uInt tempIndex =
              (cellIndex)*numQuads * contiguousBlockSize +
              quadId * contiguousBlockSize + intraBlockIndex;
            const dftfe::uInt tempIndex2 =
              (cellIndex)*numQuads * contiguousBlockSize * 3 +
              quadId * contiguousBlockSize + intraBlockIndex;
            const double psi      = psiQuadValues[tempIndex];
            const double gradPsiX = gradPsiQuadValues[tempIndex2];
            const double gradPsiY =
              gradPsiQuadValues[tempIndex2 + numQuads * contiguousBlockSize];
            const double gradPsiZ =
              gradPsiQuadValues[tempIndex2 +
                                2 * numQuads * contiguousBlockSize];
            const double eigenValue = eigenValues[intraBlockIndex];
            const double partOcc    = partialOccupancies[intraBlockIndex];

            const double identityFactor =
              partOcc * (gradPsiX * gradPsiX + gradPsiY * gradPsiY +
                         gradPsiZ * gradPsiZ) -
              2.0 * partOcc * eigenValue * psi * psi;

            if (eshelbyIndex == 0)
              eshelbyTensor[index] =
                -2.0 * partOcc * gradPsiX * gradPsiX + identityFactor;
            else if (eshelbyIndex == 1)
              eshelbyTensor[index] = -2.0 * partOcc * gradPsiX * gradPsiY;
            else if (eshelbyIndex == 2)
              eshelbyTensor[index] = -2.0 * partOcc * gradPsiX * gradPsiZ;
            else if (eshelbyIndex == 3)
              eshelbyTensor[index] = -2.0 * partOcc * gradPsiY * gradPsiX;
            else if (eshelbyIndex == 4)
              eshelbyTensor[index] =
                -2.0 * partOcc * gradPsiY * gradPsiY + identityFactor;
            else if (eshelbyIndex == 5)
              eshelbyTensor[index] = -2.0 * partOcc * gradPsiY * gradPsiZ;
            else if (eshelbyIndex == 6)
              eshelbyTensor[index] = -2.0 * partOcc * gradPsiZ * gradPsiX;
            else if (eshelbyIndex == 7)
              eshelbyTensor[index] = -2.0 * partOcc * gradPsiZ * gradPsiY;
            else if (eshelbyIndex == 8)
              eshelbyTensor[index] =
                -2.0 * partOcc * gradPsiZ * gradPsiZ + identityFactor;
          }
      }


      __global__ void
      computeELocWfcEshelbyTensorContributions(
        const dftfe::uInt                        contiguousBlockSize,
        const dftfe::uInt                        numContiguousBlocks,
        const dftfe::uInt                        numQuads,
        const dftfe::utils::deviceDoubleComplex *psiQuadValues,
        const dftfe::utils::deviceDoubleComplex *gradPsiQuadValues,
        const double                            *eigenValues,
        const double                            *partialOccupancies,
        const double                             kcoordx,
        const double                             kcoordy,
        const double                             kcoordz,
        double                                  *eshelbyTensor,
        const bool                               addEk)
      {
        const dftfe::uInt globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const dftfe::uInt blockIndex = index / contiguousBlockSize;
            const dftfe::uInt intraBlockIndex =
              index - blockIndex * contiguousBlockSize;
            const dftfe::uInt blockIndex2  = blockIndex / 9;
            const dftfe::uInt eshelbyIndex = blockIndex - 9 * blockIndex2;
            const dftfe::uInt cellIndex    = blockIndex2 / numQuads;
            const dftfe::uInt quadId       = blockIndex2 - cellIndex * numQuads;
            const dftfe::uInt tempIndex =
              (cellIndex)*numQuads * contiguousBlockSize +
              quadId * contiguousBlockSize + intraBlockIndex;
            const dftfe::uInt tempIndex2 =
              (cellIndex)*numQuads * contiguousBlockSize * 3 +
              quadId * contiguousBlockSize + intraBlockIndex;
            const dftfe::utils::deviceDoubleComplex psi =
              psiQuadValues[tempIndex];
            const dftfe::utils::deviceDoubleComplex psiConj =
              dftfe::utils::conj(psiQuadValues[tempIndex]);
            const dftfe::utils::deviceDoubleComplex gradPsiX =
              gradPsiQuadValues[tempIndex2];
            const dftfe::utils::deviceDoubleComplex gradPsiY =
              gradPsiQuadValues[tempIndex2 + numQuads * contiguousBlockSize];
            const dftfe::utils::deviceDoubleComplex gradPsiZ =
              gradPsiQuadValues[tempIndex2 +
                                2 * numQuads * contiguousBlockSize];
            const dftfe::utils::deviceDoubleComplex gradPsiXConj =
              dftfe::utils::conj(gradPsiQuadValues[tempIndex2]);
            const dftfe::utils::deviceDoubleComplex gradPsiYConj =
              dftfe::utils::conj(
                gradPsiQuadValues[tempIndex2 + numQuads * contiguousBlockSize]);
            const dftfe::utils::deviceDoubleComplex gradPsiZConj =
              dftfe::utils::conj(
                gradPsiQuadValues[tempIndex2 +
                                  2 * numQuads * contiguousBlockSize]);
            const double eigenValue = eigenValues[intraBlockIndex];
            const double partOcc    = partialOccupancies[intraBlockIndex];

            const double identityFactor =
              partOcc *
              ((dftfe::utils::mult(gradPsiXConj, gradPsiX).x +
                dftfe::utils::mult(gradPsiYConj, gradPsiY).x +
                dftfe::utils::mult(gradPsiZConj, gradPsiZ).x) +
               2.0 * (kcoordx * dftfe::utils::mult(psiConj, gradPsiX).y +
                      kcoordy * dftfe::utils::mult(psiConj, gradPsiY).y +
                      kcoordz * dftfe::utils::mult(psiConj, gradPsiZ).y) +
               (kcoordx * kcoordx + kcoordy * kcoordy + kcoordz * kcoordz -
                2.0 * eigenValue) *
                 dftfe::utils::mult(psiConj, psi).x);
            if (addEk)
              {
                if (eshelbyIndex == 0)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiXConj, gradPsiX).x +
                    -2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiX).y *
                      kcoordx -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiX).y *
                      kcoordx -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, psi).x *
                      kcoordx * kcoordx +
                    identityFactor;
                else if (eshelbyIndex == 1)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiXConj, gradPsiY).x +
                    -2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiX).y *
                      kcoordy -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiY).y *
                      kcoordx -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, psi).x *
                      kcoordx * kcoordy;
                else if (eshelbyIndex == 2)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiXConj, gradPsiZ).x +
                    -2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiX).y *
                      kcoordz -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiZ).y *
                      kcoordx -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, psi).x *
                      kcoordx * kcoordz;
                else if (eshelbyIndex == 3)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiYConj, gradPsiX).x +
                    -2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiY).y *
                      kcoordx -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiX).y *
                      kcoordy -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, psi).x *
                      kcoordy * kcoordx;
                else if (eshelbyIndex == 4)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiYConj, gradPsiY).x +
                    -2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiY).y *
                      kcoordy -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiY).y *
                      kcoordy -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, psi).x *
                      kcoordy * kcoordy +
                    identityFactor;
                else if (eshelbyIndex == 5)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiYConj, gradPsiZ).x -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiY).y *
                      kcoordz -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiZ).y *
                      kcoordy -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, psi).x *
                      kcoordy * kcoordz;
                else if (eshelbyIndex == 6)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiZConj, gradPsiX).x +
                    -2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiZ).y *
                      kcoordx -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiX).y *
                      kcoordz -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, psi).x *
                      kcoordz * kcoordx;
                else if (eshelbyIndex == 7)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiZConj, gradPsiY).x -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiZ).y *
                      kcoordy -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiY).y *
                      kcoordz -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, psi).x *
                      kcoordz * kcoordy;
                else if (eshelbyIndex == 8)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiZConj, gradPsiZ).x +
                    -2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiZ).y *
                      kcoordz -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiZ).y *
                      kcoordz -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, psi).x *
                      kcoordz * kcoordz +
                    identityFactor;
              }
            else
              {
                if (eshelbyIndex == 0)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiXConj, gradPsiX).x -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiX).y *
                      kcoordx +
                    identityFactor;
                else if (eshelbyIndex == 1)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiXConj, gradPsiY).x -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiX).y *
                      kcoordy;
                else if (eshelbyIndex == 2)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiXConj, gradPsiZ).x -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiX).y *
                      kcoordz;
                else if (eshelbyIndex == 3)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiYConj, gradPsiX).x -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiY).y *
                      kcoordx;
                else if (eshelbyIndex == 4)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiYConj, gradPsiY).x -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiY).y *
                      kcoordy +
                    identityFactor;
                else if (eshelbyIndex == 5)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiYConj, gradPsiZ).x -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiY).y *
                      kcoordz;
                else if (eshelbyIndex == 6)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiZConj, gradPsiX).x -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiZ).y *
                      kcoordx;
                else if (eshelbyIndex == 7)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiZConj, gradPsiY).x -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiZ).y *
                      kcoordy;
                else if (eshelbyIndex == 8)
                  eshelbyTensor[index] =
                    -2.0 * partOcc *
                      dftfe::utils::mult(gradPsiZConj, gradPsiZ).x -
                    2.0 * partOcc * dftfe::utils::mult(psiConj, gradPsiZ).y *
                      kcoordz +
                    identityFactor;
              }
          }
      }


      __global__ void
      nlpContractionContributionPsiIndexDeviceKernel(
        const dftfe::uInt  numPsi,
        const dftfe::uInt  numQuadsNLP,
        const dftfe::uInt  totalNonTrivialPseudoWfcs,
        const dftfe::uInt  startingId,
        const double      *projectorKetTimesVectorPar,
        const double      *gradPsiOrPsiQuadValuesNLP,
        const double      *partialOccupancies,
        const dftfe::uInt *nonTrivialIdToElemIdMap,
        const dftfe::uInt *projecterKetTimesFlattenedVectorLocalIds,
        double            *nlpContractionContribution)
      {
        const dftfe::uInt globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dftfe::uInt numberEntries =
          totalNonTrivialPseudoWfcs * numQuadsNLP * numPsi;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const dftfe::uInt blockIndex  = index / numPsi;
            const dftfe::uInt wfcId       = index - blockIndex * numPsi;
            dftfe::uInt       pseudoWfcId = blockIndex / numQuadsNLP;
            const dftfe::uInt quadId = blockIndex - pseudoWfcId * numQuadsNLP;
            pseudoWfcId += startingId;
            nlpContractionContribution[index] =
              partialOccupancies[wfcId] *
              gradPsiOrPsiQuadValuesNLP[nonTrivialIdToElemIdMap[pseudoWfcId] *
                                          numQuadsNLP * numPsi +
                                        quadId * numPsi + wfcId] *
              projectorKetTimesVectorPar
                [projecterKetTimesFlattenedVectorLocalIds[pseudoWfcId] *
                   numPsi +
                 wfcId];
          }
      }

      __global__ void
      nlpContractionContributionPsiIndexDeviceKernel(
        const dftfe::uInt                        numPsi,
        const dftfe::uInt                        numQuadsNLP,
        const dftfe::uInt                        totalNonTrivialPseudoWfcs,
        const dftfe::uInt                        startingId,
        const dftfe::utils::deviceDoubleComplex *projectorKetTimesVectorPar,
        const dftfe::utils::deviceDoubleComplex *gradPsiOrPsiQuadValuesNLP,
        const double                            *partialOccupancies,
        const dftfe::uInt                       *nonTrivialIdToElemIdMap,
        const dftfe::uInt *projecterKetTimesFlattenedVectorLocalIds,
        dftfe::utils::deviceDoubleComplex *nlpContractionContribution)
      {
        const dftfe::uInt globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dftfe::uInt numberEntries =
          totalNonTrivialPseudoWfcs * numQuadsNLP * numPsi;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const dftfe::uInt blockIndex  = index / numPsi;
            const dftfe::uInt wfcId       = index - blockIndex * numPsi;
            dftfe::uInt       pseudoWfcId = blockIndex / numQuadsNLP;
            const dftfe::uInt quadId = blockIndex - pseudoWfcId * numQuadsNLP;
            pseudoWfcId += startingId;

            const dftfe::utils::deviceDoubleComplex temp = dftfe::utils::mult(
              dftfe::utils::conj(
                gradPsiOrPsiQuadValuesNLP[nonTrivialIdToElemIdMap[pseudoWfcId] *
                                            numQuadsNLP * numPsi +
                                          quadId * numPsi + wfcId]),
              projectorKetTimesVectorPar
                [projecterKetTimesFlattenedVectorLocalIds[pseudoWfcId] *
                   numPsi +
                 wfcId]);
            nlpContractionContribution[index] =
              dftfe::utils::makeComplex(partialOccupancies[wfcId] * temp.x,
                                        partialOccupancies[wfcId] * temp.y);
          }
      }

    } // namespace

    template <typename ValueType>
    void
    nlpContractionContributionPsiIndex(
      const dftfe::uInt  wfcBlockSize,
      const dftfe::uInt  blockSizeNlp,
      const dftfe::uInt  numQuadsNLP,
      const dftfe::uInt  startingIdNlp,
      const ValueType   *projectorKetTimesVectorPar,
      const ValueType   *gradPsiOrPsiQuadValuesNLP,
      const double      *partialOccupancies,
      const dftfe::uInt *nonTrivialIdToElemIdMap,
      const dftfe::uInt *projecterKetTimesFlattenedVectorLocalIds,
      ValueType         *nlpContractionContribution)
    {
      DFTFE_LAUNCH_KERNEL(
        nlpContractionContributionPsiIndexDeviceKernel,
        (wfcBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE * numQuadsNLP * blockSizeNlp,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        wfcBlockSize,
        numQuadsNLP,
        blockSizeNlp,
        startingIdNlp,
        dftfe::utils::makeDataTypeDeviceCompatible(projectorKetTimesVectorPar),
        dftfe::utils::makeDataTypeDeviceCompatible(gradPsiOrPsiQuadValuesNLP),
        partialOccupancies,
        nonTrivialIdToElemIdMap,
        projecterKetTimesFlattenedVectorLocalIds,
        dftfe::utils::makeDataTypeDeviceCompatible(nlpContractionContribution));
    }


    template <typename ValueType>
    void
    computeELocWfcEshelbyTensorContributions(const dftfe::uInt wfcBlockSize,
                                             const dftfe::uInt cellsBlockSize,
                                             const dftfe::uInt numQuads,
                                             const ValueType  *psiQuadValues,
                                             const ValueType *gradPsiQuadValues,
                                             const double    *eigenValues,
                                             const double *partialOccupancies,
#  ifdef USE_COMPLEX
                                             const double kcoordx,
                                             const double kcoordy,
                                             const double kcoordz,
#  endif
                                             double *eshelbyTensorContributions
#  ifdef USE_COMPLEX
                                             ,
                                             const bool addEk
#  endif
    )
    {
      DFTFE_LAUNCH_KERNEL(
        computeELocWfcEshelbyTensorContributions,
        (wfcBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE * cellsBlockSize * numQuads * 9,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        wfcBlockSize,
        cellsBlockSize * numQuads * 9,
        numQuads,
        dftfe::utils::makeDataTypeDeviceCompatible(psiQuadValues),
        dftfe::utils::makeDataTypeDeviceCompatible(gradPsiQuadValues),
        eigenValues,
        partialOccupancies,
#  ifdef USE_COMPLEX
        kcoordx,
        kcoordy,
        kcoordz,
#  endif
        eshelbyTensorContributions
#  ifdef USE_COMPLEX
        ,
        addEk
#  endif
      );
    }

    template void
    nlpContractionContributionPsiIndex(
      const dftfe::uInt        wfcBlockSize,
      const dftfe::uInt        blockSizeNlp,
      const dftfe::uInt        numQuadsNLP,
      const dftfe::uInt        startingIdNlp,
      const dataTypes::number *projectorKetTimesVectorPar,
      const dataTypes::number *gradPsiOrPsiQuadValuesNLP,
      const double            *partialOccupancies,
      const dftfe::uInt       *nonTrivialIdToElemIdMap,
      const dftfe::uInt       *projecterKetTimesFlattenedVectorLocalIds,
      dataTypes::number       *nlpContractionContribution);

    template void
    computeELocWfcEshelbyTensorContributions(
      const dftfe::uInt        wfcBlockSize,
      const dftfe::uInt        cellsBlockSize,
      const dftfe::uInt        numQuads,
      const dataTypes::number *psiQuadValues,
      const dataTypes::number *gradPsiQuadValues,
      const double            *eigenValues,
      const double            *partialOccupancies,
#  ifdef USE_COMPLEX
      const double kcoordx,
      const double kcoordy,
      const double kcoordz,
#  endif
      double *eshelbyTensor
#  ifdef USE_COMPLEX
      ,
      const bool addEk
#  endif
    );

  } // namespace forceDeviceKernels
} // namespace dftfe
#endif
