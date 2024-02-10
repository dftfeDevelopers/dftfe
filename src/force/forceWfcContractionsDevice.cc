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

// source file for force related computations
#include "constants.h"
#include "dftUtils.h"
#include "forceWfcContractionsDevice.h"
#include "vectorUtilities.h"
#include "deviceKernelsGeneric.h"
#include <MemoryStorage.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceAPICalls.h>
#include <DeviceKernelLauncherConstants.h>
#include <DeviceBlasWrapper.h>



namespace dftfe
{
  namespace forceDevice
  {
    namespace
    {
      __global__ void
      computeELocWfcEshelbyTensorContributions(
        const unsigned int contiguousBlockSize,
        const unsigned int numContiguousBlocks,
        const unsigned int numQuads,
        const double *     psiQuadValues,
        const double *     gradPsiQuadValues,
        const double *     eigenValues,
        const double *     partialOccupancies,
        double *           eshelbyTensor)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex = index / contiguousBlockSize;
            const unsigned int intraBlockIndex =
              index - blockIndex * contiguousBlockSize;
            const unsigned int blockIndex2  = blockIndex / 9;
            const unsigned int eshelbyIndex = blockIndex - 9 * blockIndex2;
            const unsigned int cellIndex    = blockIndex2 / numQuads;
            const unsigned int quadId = blockIndex2 - cellIndex * numQuads;
            const unsigned int tempIndex =
              (cellIndex)*numQuads * contiguousBlockSize +
              quadId * contiguousBlockSize + intraBlockIndex;
            const unsigned int tempIndex2 =
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
        const unsigned int                       contiguousBlockSize,
        const unsigned int                       numContiguousBlocks,
        const unsigned int                       numQuads,
        const dftfe::utils::deviceDoubleComplex *psiQuadValues,
        const dftfe::utils::deviceDoubleComplex *gradPsiQuadValues,
        const double *                           eigenValues,
        const double *                           partialOccupancies,
        const double                             kcoordx,
        const double                             kcoordy,
        const double                             kcoordz,
        double *                                 eshelbyTensor,
        const bool                               addEk)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex = index / contiguousBlockSize;
            const unsigned int intraBlockIndex =
              index - blockIndex * contiguousBlockSize;
            const unsigned int blockIndex2  = blockIndex / 9;
            const unsigned int eshelbyIndex = blockIndex - 9 * blockIndex2;
            const unsigned int cellIndex    = blockIndex2 / numQuads;
            const unsigned int quadId = blockIndex2 - cellIndex * numQuads;
            const unsigned int tempIndex =
              (cellIndex)*numQuads * contiguousBlockSize +
              quadId * contiguousBlockSize + intraBlockIndex;
            const unsigned int tempIndex2 =
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
        const unsigned int  numPsi,
        const unsigned int  numQuadsNLP,
        const unsigned int  totalNonTrivialPseudoWfcs,
        const unsigned int  startingId,
        const double *      projectorKetTimesVectorPar,
        const double *      gradPsiOrPsiQuadValuesNLP,
        const double *      partialOccupancies,
        const unsigned int *nonTrivialIdToElemIdMap,
        const unsigned int *projecterKetTimesFlattenedVectorLocalIds,
        double *            nlpContractionContribution)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries =
          totalNonTrivialPseudoWfcs * numQuadsNLP * numPsi;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex  = index / numPsi;
            const unsigned int wfcId       = index - blockIndex * numPsi;
            unsigned int       pseudoWfcId = blockIndex / numQuadsNLP;
            const unsigned int quadId = blockIndex - pseudoWfcId * numQuadsNLP;
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
        const unsigned int                       numPsi,
        const unsigned int                       numQuadsNLP,
        const unsigned int                       totalNonTrivialPseudoWfcs,
        const unsigned int                       startingId,
        const dftfe::utils::deviceDoubleComplex *projectorKetTimesVectorPar,
        const dftfe::utils::deviceDoubleComplex *gradPsiOrPsiQuadValuesNLP,
        const double *                           partialOccupancies,
        const unsigned int *                     nonTrivialIdToElemIdMap,
        const unsigned int *projecterKetTimesFlattenedVectorLocalIds,
        dftfe::utils::deviceDoubleComplex *nlpContractionContribution)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries =
          totalNonTrivialPseudoWfcs * numQuadsNLP * numPsi;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex  = index / numPsi;
            const unsigned int wfcId       = index - blockIndex * numPsi;
            unsigned int       pseudoWfcId = blockIndex / numQuadsNLP;
            const unsigned int quadId = blockIndex - pseudoWfcId * numQuadsNLP;
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

      void
      interpolatePsiComputeELocWfcEshelbyTensorD(
        std::shared_ptr<
          dftfe::basis::FEBasisOperations<dataTypes::number,
                                          double,
                                          dftfe::utils::MemorySpace::DEVICE>>
          &basisOperationsPtr,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
          &                                      BLASWrapperPtr,
        operatorDFTDeviceClass &                 operatorMatrix,
        distributedDeviceVec<dataTypes::number> &Xb,
        const unsigned int                       BVec,
        const unsigned int                       numCells,
        const unsigned int                       numQuads,
        const unsigned int                       numQuadsNLP,
        const unsigned int                       numNodesPerElement,
        const dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::DEVICE>
          &eigenValuesD,
        const dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::DEVICE>
          &partialOccupanciesD,
#ifdef USE_COMPLEX
        const double kcoordx,
        const double kcoordy,
        const double kcoordz,
#endif
        const dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::DEVICE>
          &                onesVecD,
        const unsigned int cellsBlockSize,
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::DEVICE>
          &psiQuadsFlatD,
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::DEVICE>
          &gradPsiQuadsFlatD,
#ifdef USE_COMPLEX
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::DEVICE>
          &psiQuadsNLPD,
#endif
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::DEVICE>
          &gradPsiQuadsNLPFlatD,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          &eshelbyTensorContributionsD,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          &        eshelbyTensorQuadValuesD,
        const bool isPsp,
        const bool isFloatingChargeForces,
        const bool addEk)
      {
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::DEVICE>
          &cellWaveFunctionMatrix = operatorMatrix.getCellWaveFunctionMatrix();

        const int blockSize    = cellsBlockSize;
        const int numberBlocks = numCells / blockSize;
        const int remBlockSize = numCells - numberBlocks * blockSize;

        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::DEVICE>
          shapeFunctionValuesNLPReferenceD(numQuadsNLP * numNodesPerElement,
                                           dataTypes::number(0.0));

        dftfe::utils::deviceKernelsGeneric::copyValueType1ArrToValueType2Arr(
          numQuadsNLP * numNodesPerElement,
          (operatorMatrix.getShapeFunctionValuesNLPTransposed()).begin(),
          shapeFunctionValuesNLPReferenceD.begin());

        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          shapeFunctionGradientValuesNLPReferenceD(blockSize * numQuadsNLP * 3 *
                                                     numNodesPerElement,
                                                   0.0);
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          shapeFunctionGradientValuesNLPD(blockSize * numQuadsNLP * 3 *
                                            numNodesPerElement,
                                          0.0);
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::DEVICE>
          shapeFunctionGradientValuesNLPDCopy(blockSize * numQuadsNLP * 3 *
                                                numNodesPerElement,
                                              dataTypes::number(0.0));

        for (unsigned int i = 0; i < blockSize; i++)
          shapeFunctionGradientValuesNLPReferenceD.copyFrom(
            operatorMatrix.getShapeFunctionGradientValuesNLPTransposed(),
            (operatorMatrix.getShapeFunctionGradientValuesNLPTransposed())
              .size(),
            0,
            i * numQuadsNLP * 3 * numNodesPerElement);
        basisOperationsPtr->reinit(BVec, cellsBlockSize, 0);
        basisOperationsPtr->extractToCellNodalDataKernel(
          Xb,
          cellWaveFunctionMatrix.data(),
          std::pair<unsigned int, unsigned int>(0, numCells));


        for (int iblock = 0; iblock < (numberBlocks + 1); iblock++)
          {
            const int currentBlockSize =
              (iblock == numberBlocks) ? remBlockSize : blockSize;
            const int startingId = iblock * blockSize;

            if (currentBlockSize > 0)
              {
                const dataTypes::number scalarCoeffAlpha =
                  dataTypes::number(1.0);
                const dataTypes::number scalarCoeffBeta =
                  dataTypes::number(0.0);
                const double scalarCoeffAlphaReal = 1.0;
                const double scalarCoeffBetaReal  = 0.0;

                int strideA = BVec * numNodesPerElement;
                int strideB = 0;
                int strideC = BVec * numQuads;

                if (!isFloatingChargeForces)
                  {
                    basisOperationsPtr->interpolateKernel(
                      cellWaveFunctionMatrix.data() +
                        startingId * numNodesPerElement * BVec,
                      psiQuadsFlatD.data(),
                      gradPsiQuadsFlatD.begin(),
                      std::pair<unsigned int, unsigned int>(
                        startingId, startingId + currentBlockSize));
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
                    computeELocWfcEshelbyTensorContributions<<<
                      (BVec + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                        dftfe::utils::DEVICE_BLOCK_SIZE * currentBlockSize *
                        numQuads * 9,
                      dftfe::utils::DEVICE_BLOCK_SIZE>>>(
                      BVec,
                      currentBlockSize * numQuads * 9,
                      numQuads,
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        psiQuadsFlatD.begin()),
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        gradPsiQuadsFlatD.begin()),
                      eigenValuesD.begin(),
                      partialOccupanciesD.begin(),
#  ifdef USE_COMPLEX
                      kcoordx,
                      kcoordy,
                      kcoordz,
#  endif
                      eshelbyTensorContributionsD.begin()
#  ifdef USE_COMPLEX
                        ,
                      addEk
#  endif
                    );
#elif DFTFE_WITH_DEVICE_LANG_HIP
                    hipLaunchKernelGGL(
                      computeELocWfcEshelbyTensorContributions,
                      (BVec + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                        dftfe::utils::DEVICE_BLOCK_SIZE * currentBlockSize *
                        numQuads * 9,
                      dftfe::utils::DEVICE_BLOCK_SIZE,
                      0,
                      0,
                      BVec,
                      currentBlockSize * numQuads * 9,
                      numQuads,
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        psiQuadsFlatD.begin()),
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        gradPsiQuadsFlatD.begin()),
                      eigenValuesD.begin(),
                      partialOccupanciesD.begin(),
#  ifdef USE_COMPLEX
                      kcoordx,
                      kcoordy,
                      kcoordz,
#  endif
                      eshelbyTensorContributionsD.begin()
#  ifdef USE_COMPLEX
                        ,
                      addEk
#  endif
                    );
#endif

                    const double scalarCoeffAlphaEshelby = 1.0;
                    const double scalarCoeffBetaEshelby  = 1.0;



                    dftfe::utils::deviceBlasWrapper::gemm(
                      BLASWrapperPtr->getDeviceBlasHandle(),
                      dftfe::utils::DEVICEBLAS_OP_N,
                      dftfe::utils::DEVICEBLAS_OP_N,
                      1,
                      currentBlockSize * numQuads * 9,
                      BVec,
                      &scalarCoeffAlphaEshelby,
                      onesVecD.begin(),
                      1,
                      eshelbyTensorContributionsD.begin(),
                      BVec,
                      &scalarCoeffBetaEshelby,
                      eshelbyTensorQuadValuesD.begin() +
                        startingId * numQuads * 9,
                      1);
                  }

                if (isPsp)
                  {
#ifdef USE_COMPLEX
                    const int strideCNLP = BVec * numQuadsNLP;
                    const int strideBNLP = 0;

                    dftfe::utils::deviceBlasWrapper::gemmStridedBatched(
                      BLASWrapperPtr->getDeviceBlasHandle(),
                      dftfe::utils::DEVICEBLAS_OP_N,
                      dftfe::utils::DEVICEBLAS_OP_N,
                      BVec,
                      numQuadsNLP,
                      numNodesPerElement,
                      &scalarCoeffAlpha,
                      cellWaveFunctionMatrix.begin() +
                        startingId * numNodesPerElement * BVec,
                      BVec,
                      strideA,
                      shapeFunctionValuesNLPReferenceD.begin(),
                      numNodesPerElement,
                      strideBNLP,
                      &scalarCoeffBeta,
                      psiQuadsNLPD.begin() + startingId * numQuadsNLP * BVec,
                      BVec,
                      strideCNLP,
                      currentBlockSize);
#endif

                    // shapeGradRef^T*invJacobian^T
                    dftfe::utils::deviceBlasWrapper::gemmStridedBatched(
                      BLASWrapperPtr->getDeviceBlasHandle(),
                      dftfe::utils::DEVICEBLAS_OP_N,
                      dftfe::utils::DEVICEBLAS_OP_N,
                      numNodesPerElement,
                      3,
                      3,
                      &scalarCoeffAlphaReal,
                      shapeFunctionGradientValuesNLPReferenceD.begin(),
                      numNodesPerElement,
                      numNodesPerElement * 3,
                      (operatorMatrix.getInverseJacobiansNLP()).begin() +
                        startingId * numQuadsNLP * 3 * 3,
                      3,
                      3 * 3,
                      &scalarCoeffBetaReal,
                      shapeFunctionGradientValuesNLPD.begin(),
                      numNodesPerElement,
                      numNodesPerElement * 3,
                      currentBlockSize * numQuadsNLP);

                    dftfe::utils::deviceKernelsGeneric::
                      copyValueType1ArrToValueType2Arr(
                        currentBlockSize * numQuadsNLP * numNodesPerElement * 3,
                        shapeFunctionGradientValuesNLPD.begin(),
                        shapeFunctionGradientValuesNLPDCopy.begin());

                    const int strideCNLPGrad = BVec * 3 * numQuadsNLP;
                    const int strideBNLPGrad =
                      numNodesPerElement * 3 * numQuadsNLP;

                    dftfe::utils::deviceBlasWrapper::gemmStridedBatched(
                      BLASWrapperPtr->getDeviceBlasHandle(),
                      dftfe::utils::DEVICEBLAS_OP_N,
                      dftfe::utils::DEVICEBLAS_OP_N,
                      BVec,
                      3 * numQuadsNLP,
                      numNodesPerElement,
                      &scalarCoeffAlpha,
                      cellWaveFunctionMatrix.begin() +
                        startingId * numNodesPerElement * BVec,
                      BVec,
                      strideA,
                      shapeFunctionGradientValuesNLPDCopy.begin(),
                      numNodesPerElement,
                      strideBNLPGrad,
                      &scalarCoeffBeta,
                      gradPsiQuadsNLPFlatD.begin() +
                        startingId * numQuadsNLP * 3 * BVec,
                      BVec,
                      strideCNLPGrad,
                      currentBlockSize);
                  }
              }
          }
      }

      void
      nlpPsiContractionD(
        operatorDFTDeviceClass &operatorMatrix,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
          &BLASWrapperPtr,
#ifdef USE_COMPLEX
        const dftfe::utils::MemoryStorage<dataTypes::number,
                                          dftfe::utils::MemorySpace::DEVICE>
          &psiQuadsNLPD,
#endif
        const dftfe::utils::MemoryStorage<dataTypes::number,
                                          dftfe::utils::MemorySpace::DEVICE>
          &gradPsiQuadsNLPD,
        const dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::DEVICE>
          &partialOccupanciesD,
        const dftfe::utils::MemoryStorage<dataTypes::number,
                                          dftfe::utils::MemorySpace::DEVICE>
          &                      onesVecDNLP,
        const dataTypes::number *projectorKetTimesVectorParFlattenedD,
        const dftfe::utils::MemoryStorage<unsigned int,
                                          dftfe::utils::MemorySpace::DEVICE>
          &nonTrivialIdToElemIdMapD,
        const dftfe::utils::MemoryStorage<unsigned int,
                                          dftfe::utils::MemorySpace::DEVICE>
          &                projecterKetTimesFlattenedVectorLocalIdsD,
        const unsigned int numCells,
        const unsigned int numQuadsNLP,
        const unsigned int numPsi,
        const unsigned int totalNonTrivialPseudoWfcs,
        const unsigned int innerBlockSizeEnlp,
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::DEVICE>
          &nlpContractionContributionD,
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::DEVICE> &
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedDBlock,
        dataTypes::number *
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
#ifdef USE_COMPLEX
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::DEVICE> &
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedDBlock,
        dataTypes::number
          *projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
#endif
        dataTypes::number *
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp)
      {
        const int blockSizeNlp    = innerBlockSizeEnlp;
        const int numberBlocksNlp = totalNonTrivialPseudoWfcs / blockSizeNlp;
        const int remBlockSizeNlp =
          totalNonTrivialPseudoWfcs - numberBlocksNlp * blockSizeNlp;

        dataTypes::number scalarCoeffAlphaNlp = dataTypes::number(1.0);
        dataTypes::number scalarCoeffBetaNlp  = dataTypes::number(0.0);

        for (int iblocknlp = 0; iblocknlp < (numberBlocksNlp + 1); iblocknlp++)
          {
            const int currentBlockSizeNlp =
              (iblocknlp == numberBlocksNlp) ? remBlockSizeNlp : blockSizeNlp;
            const int startingIdNlp = iblocknlp * blockSizeNlp;
            if (currentBlockSizeNlp > 0)
              {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
                nlpContractionContributionPsiIndexDeviceKernel<<<
                  (numPsi + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                    dftfe::utils::DEVICE_BLOCK_SIZE * numQuadsNLP * 3 *
                    currentBlockSizeNlp,
                  dftfe::utils::DEVICE_BLOCK_SIZE>>>(
                  numPsi,
                  numQuadsNLP * 3,
                  currentBlockSizeNlp,
                  startingIdNlp,
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    projectorKetTimesVectorParFlattenedD),
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    gradPsiQuadsNLPD.begin()),
                  partialOccupanciesD.begin(),
                  nonTrivialIdToElemIdMapD.begin(),
                  projecterKetTimesFlattenedVectorLocalIdsD.begin(),
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    nlpContractionContributionD.begin()));
#elif DFTFE_WITH_DEVICE_LANG_HIP
                hipLaunchKernelGGL(
                  nlpContractionContributionPsiIndexDeviceKernel,
                  (numPsi + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                    dftfe::utils::DEVICE_BLOCK_SIZE * numQuadsNLP * 3 *
                    currentBlockSizeNlp,
                  dftfe::utils::DEVICE_BLOCK_SIZE,
                  0,
                  0,
                  numPsi,
                  numQuadsNLP * 3,
                  currentBlockSizeNlp,
                  startingIdNlp,
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    projectorKetTimesVectorParFlattenedD),
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    gradPsiQuadsNLPD.begin()),
                  partialOccupanciesD.begin(),
                  nonTrivialIdToElemIdMapD.begin(),
                  projecterKetTimesFlattenedVectorLocalIdsD.begin(),
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    nlpContractionContributionD.begin()));
#endif

                dftfe::utils::deviceBlasWrapper::gemm(
                  BLASWrapperPtr->getDeviceBlasHandle(),
                  dftfe::utils::DEVICEBLAS_OP_N,
                  dftfe::utils::DEVICEBLAS_OP_N,
                  1,
                  currentBlockSizeNlp * numQuadsNLP * 3,
                  numPsi,
                  &scalarCoeffAlphaNlp,
                  onesVecDNLP.begin(),
                  1,
                  nlpContractionContributionD.begin(),
                  numPsi,
                  &scalarCoeffBetaNlp,
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedDBlock
                    .begin(),
                  1);

                dftfe::utils::deviceMemcpyD2H(
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp),
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedDBlock
                      .begin()),
                  currentBlockSizeNlp * numQuadsNLP * 3 *
                    sizeof(dataTypes::number));

                for (unsigned int i = 0;
                     i < currentBlockSizeNlp * numQuadsNLP * 3;
                     i++)
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH
                    [startingIdNlp * numQuadsNLP * 3 + i] +=
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp
                      [i];
#ifdef USE_COMPLEX
#  ifdef DFTFE_WITH_DEVICE_LANG_CUDA
                nlpContractionContributionPsiIndexDeviceKernel<<<
                  (numPsi + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                    dftfe::utils::DEVICE_BLOCK_SIZE * numQuadsNLP *
                    currentBlockSizeNlp,
                  dftfe::utils::DEVICE_BLOCK_SIZE>>>(
                  numPsi,
                  numQuadsNLP,
                  currentBlockSizeNlp,
                  startingIdNlp,
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    projectorKetTimesVectorParFlattenedD),
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    psiQuadsNLPD.begin()),
                  partialOccupanciesD.begin(),
                  nonTrivialIdToElemIdMapD.begin(),
                  projecterKetTimesFlattenedVectorLocalIdsD.begin(),
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    nlpContractionContributionD.begin()));
#  elif DFTFE_WITH_DEVICE_LANG_HIP
                hipLaunchKernelGGL(
                  nlpContractionContributionPsiIndexDeviceKernel,
                  (numPsi + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                    dftfe::utils::DEVICE_BLOCK_SIZE * numQuadsNLP *
                    currentBlockSizeNlp,
                  dftfe::utils::DEVICE_BLOCK_SIZE,
                  0,
                  0,
                  numPsi,
                  numQuadsNLP,
                  currentBlockSizeNlp,
                  startingIdNlp,
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    projectorKetTimesVectorParFlattenedD),
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    psiQuadsNLPD.begin()),
                  partialOccupanciesD.begin(),
                  nonTrivialIdToElemIdMapD.begin(),
                  projecterKetTimesFlattenedVectorLocalIdsD.begin(),
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    nlpContractionContributionD.begin()));
#  endif

                dftfe::utils::deviceBlasWrapper::gemm(
                  BLASWrapperPtr->getDeviceBlasHandle(),
                  dftfe::utils::DEVICEBLAS_OP_N,
                  dftfe::utils::DEVICEBLAS_OP_N,
                  1,
                  currentBlockSizeNlp * numQuadsNLP,
                  numPsi,
                  &scalarCoeffAlphaNlp,
                  onesVecDNLP.begin(),
                  1,
                  nlpContractionContributionD.begin(),
                  numPsi,
                  &scalarCoeffBetaNlp,
                  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedDBlock
                    .begin(),
                  1);


                dftfe::utils::deviceMemcpyD2H(
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp),
                  dftfe::utils::makeDataTypeDeviceCompatible(
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedDBlock
                      .begin()),
                  currentBlockSizeNlp * numQuadsNLP *
                    sizeof(dataTypes::number));

                for (unsigned int i = 0; i < currentBlockSizeNlp * numQuadsNLP;
                     i++)
                  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH
                    [startingIdNlp * numQuadsNLP + i] +=
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp
                      [i];
#endif
              }
          }
      }


      void
      devicePortedForceKernelsAllD(
        std::shared_ptr<
          dftfe::basis::FEBasisOperations<dataTypes::number,
                                          double,
                                          dftfe::utils::MemorySpace::DEVICE>>
          &basisOperationsPtr,
        const std::shared_ptr<
          dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
          &                                      BLASWrapperPtr,
        operatorDFTDeviceClass &                 operatorMatrix,
        distributedDeviceVec<dataTypes::number> &deviceFlattenedArrayBlock,
        distributedDeviceVec<dataTypes::number> &projectorKetTimesVectorD,
        const dataTypes::number *                X,
        const dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::DEVICE>
          &eigenValuesD,
        const dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::DEVICE>
          &partialOccupanciesD,
#ifdef USE_COMPLEX
        const double kcoordx,
        const double kcoordy,
        const double kcoordz,
#endif
        const dftfe::utils::MemoryStorage<double,
                                          dftfe::utils::MemorySpace::DEVICE>
          &onesVecD,
        const dftfe::utils::MemoryStorage<dataTypes::number,
                                          dftfe::utils::MemorySpace::DEVICE>
          &onesVecDNLP,
        const dftfe::utils::MemoryStorage<unsigned int,
                                          dftfe::utils::MemorySpace::DEVICE>
          &nonTrivialIdToElemIdMapD,
        const dftfe::utils::MemoryStorage<unsigned int,
                                          dftfe::utils::MemorySpace::DEVICE>
          &                projecterKetTimesFlattenedVectorLocalIdsD,
        const unsigned int startingVecId,
        const unsigned int N,
        const unsigned int numPsi,
        const unsigned int numCells,
        const unsigned int numQuads,
        const unsigned int numQuadsNLP,
        const unsigned int numNodesPerElement,
        const unsigned int totalNonTrivialPseudoWfcs,
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::DEVICE>
          &psiQuadsFlatD,
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::DEVICE>
          &gradPsiQuadsFlatD,
#ifdef USE_COMPLEX
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::DEVICE>
          &psiQuadsNLPD,
#endif
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::DEVICE>
          &gradPsiQuadsNLPFlatD,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          &eshelbyTensorContributionsD,
        dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
          &eshelbyTensorQuadValuesD,
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::DEVICE>
          &nlpContractionContributionD,
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::DEVICE> &
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedDBlock,
        dataTypes::number *
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
#ifdef USE_COMPLEX
        dftfe::utils::MemoryStorage<dataTypes::number,
                                    dftfe::utils::MemorySpace::DEVICE> &
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedDBlock,
        dataTypes::number
          *projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
#endif
        dataTypes::number *
                           projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp,
        const unsigned int cellsBlockSize,
        const unsigned int innerBlockSizeEnlp,
        const bool         isPsp,
        const bool         isFloatingChargeForces,
        const bool         addEk)
      {
        // int this_process;
        // MPI_Comm_rank(d_mpiCommParent, &this_process);

        dftfe::utils::deviceKernelsGeneric::stridedCopyToBlockConstantStride(
          numPsi,
          N,
          basisOperationsPtr->nOwnedDofs(),
          startingVecId,
          X,
          deviceFlattenedArrayBlock.begin());
        deviceFlattenedArrayBlock.updateGhostValues();

        (operatorMatrix.getOverloadedConstraintMatrix())
          ->distribute(deviceFlattenedArrayBlock, numPsi);


        // dftfe::utils::deviceSynchronize();
        // MPI_Barrier(d_mpiCommParent);
        // double kernel1_time = MPI_Wtime();

        interpolatePsiComputeELocWfcEshelbyTensorD(basisOperationsPtr,
                                                   BLASWrapperPtr,
                                                   operatorMatrix,
                                                   deviceFlattenedArrayBlock,
                                                   numPsi,
                                                   numCells,
                                                   numQuads,
                                                   numQuadsNLP,
                                                   numNodesPerElement,
                                                   eigenValuesD,
                                                   partialOccupanciesD,
#ifdef USE_COMPLEX
                                                   kcoordx,
                                                   kcoordy,
                                                   kcoordz,
#endif
                                                   onesVecD,
                                                   cellsBlockSize,
                                                   psiQuadsFlatD,
                                                   gradPsiQuadsFlatD,
#ifdef USE_COMPLEX
                                                   psiQuadsNLPD,
#endif
                                                   gradPsiQuadsNLPFlatD,
                                                   eshelbyTensorContributionsD,
                                                   eshelbyTensorQuadValuesD,
                                                   isPsp,
                                                   isFloatingChargeForces,
                                                   addEk);

        // dftfe::utils::deviceSynchronize();
        // MPI_Barrier(d_mpiCommParent);
        // kernel1_time = MPI_Wtime() - kernel1_time;

        // if (this_process==0 && dftParameters::verbosity>=5)
        //	 std::cout<<"Time for
        // interpolatePsiComputeELocWfcEshelbyTensorD inside blocked
        // loop: "<<kernel1_time<<std::endl;

        if (isPsp)
          {
            // dftfe::utils::deviceSynchronize();
            // MPI_Barrier(d_mpiCommParent);
            // double kernel2_time = MPI_Wtime();

            operatorMatrix.computeNonLocalProjectorKetTimesXTimesV(
              deviceFlattenedArrayBlock.begin(),
              projectorKetTimesVectorD,
              numPsi);

            // dftfe::utils::deviceSynchronize();
            // MPI_Barrier(d_mpiCommParent);
            // kernel2_time = MPI_Wtime() - kernel2_time;

            // if (this_process==0 && dftParameters::verbosity>=5)
            //  std::cout<<"Time for computeNonLocalProjectorKetTimesXTimesV
            //  inside blocked loop: "<<kernel2_time<<std::endl;

            // dftfe::utils::deviceSynchronize();
            // MPI_Barrier(d_mpiCommParent);
            // double kernel3_time = MPI_Wtime();

            if (totalNonTrivialPseudoWfcs > 0)
              {
                nlpPsiContractionD(
                  operatorMatrix,
                  BLASWrapperPtr,
#ifdef USE_COMPLEX
                  psiQuadsNLPD,
#endif
                  gradPsiQuadsNLPFlatD,
                  partialOccupanciesD,
                  onesVecDNLP,
                  projectorKetTimesVectorD.begin(),
                  nonTrivialIdToElemIdMapD,
                  projecterKetTimesFlattenedVectorLocalIdsD,
                  numCells,
                  numQuadsNLP,
                  numPsi,
                  totalNonTrivialPseudoWfcs,
                  innerBlockSizeEnlp,
                  nlpContractionContributionD,
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedDBlock,
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
#ifdef USE_COMPLEX
                  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedDBlock,
                  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
#endif
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp);
              }

            // dftfe::utils::deviceSynchronize();
            // MPI_Barrier(d_mpiCommParent);
            // kernel3_time = MPI_Wtime() - kernel3_time;

            // if (this_process==0 && dftParameters::verbosity>=5)
            //	 std::cout<<"Time for nlpPsiContractionD inside blocked loop:
            //"<<kernel3_time<<std::endl;
          }
      }

    } // namespace

    void
    wfcContractionsForceKernelsAllH(
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::DEVICE>>
        &basisOperationsPtr,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &                                     BLASWrapperPtr,
      operatorDFTDeviceClass &                operatorMatrix,
      const dataTypes::number *               X,
      const unsigned int                      spinPolarizedFlag,
      const unsigned int                      spinIndex,
      const std::vector<std::vector<double>> &eigenValuesH,
      const std::vector<std::vector<double>> &partialOccupanciesH,
      const std::vector<double> &             kPointCoordinates,
      const unsigned int *                    nonTrivialIdToElemIdMapH,
      const unsigned int *projecterKetTimesFlattenedVectorLocalIdsH,
      const unsigned int  MLoc,
      const unsigned int  N,
      const unsigned int  numCells,
      const unsigned int  numQuads,
      const unsigned int  numQuadsNLP,
      const unsigned int  numNodesPerElement,
      const unsigned int  totalNonTrivialPseudoWfcs,
      double *            eshelbyTensorQuadValuesH,
      dataTypes::number *
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
#ifdef USE_COMPLEX
      dataTypes::number
        *projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
#endif
      const MPI_Comm &     mpiCommParent,
      const MPI_Comm &     interBandGroupComm,
      const bool           isPsp,
      const bool           isFloatingChargeForces,
      const bool           addEk,
      const dftParameters &dftParams)
    {
      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const unsigned int blockSize =
        std::min(dftParams.chebyWfcBlockSize,
                 bandGroupLowHighPlusOneIndices[1]);

      // int this_process;
      // MPI_Comm_rank(mpiCommParent, &this_process);
      // dftfe::utils::deviceSynchronize();
      // MPI_Barrier(mpiCommParent);
      // double device_time = MPI_Wtime();

      distributedDeviceVec<dataTypes::number> &deviceFlattenedArrayBlock =
        operatorMatrix.getParallelChebyBlockVectorDevice();

      distributedDeviceVec<dataTypes::number> &projectorKetTimesVectorD =
        operatorMatrix.getParallelProjectorKetTimesBlockVectorDevice();

      // dftfe::utils::deviceSynchronize();
      // MPI_Barrier(mpiCommParent);
      // device_time = MPI_Wtime() - device_time;

      // if (this_process == 0 && dftParams.verbosity >= 2)
      //  std::cout
      //    << "Time for creating device parallel vectors for force computation:
      //    "
      //    << device_time << std::endl;

      // device_time = MPI_Wtime();

      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        eigenValuesD(blockSize, 0.0);
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        partialOccupanciesD(blockSize, 0.0);
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        elocWfcEshelbyTensorQuadValuesD(numCells * numQuads * 9, 0.0);

      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        onesVecD(blockSize, 1.0);
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        onesVecDNLP(blockSize, dataTypes::number(1.0));

      const unsigned int cellsBlockSize = std::min((unsigned int)10, numCells);

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        psiQuadsFlatD(cellsBlockSize * numQuads * blockSize,
                      dataTypes::number(0.0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        gradPsiQuadsFlatD(cellsBlockSize * numQuads * blockSize * 3,
                          dataTypes::number(0.0));
#ifdef USE_COMPLEX
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        psiQuadsNLPD(numCells * numQuadsNLP * blockSize,
                     dataTypes::number(0.0));
#endif

      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        gradPsiQuadsNLPFlatD(numCells * numQuadsNLP * 3 * blockSize,
                             dataTypes::number(0.0));

      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        eshelbyTensorContributionsD(cellsBlockSize * numQuads * blockSize * 9,
                                    0.0);

      const unsigned int innerBlockSizeEnlp =
        std::min((unsigned int)10, totalNonTrivialPseudoWfcs);
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        nlpContractionContributionD(innerBlockSizeEnlp * numQuadsNLP * 3 *
                                      blockSize,
                                    dataTypes::number(0.0));
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedDBlock;
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedDBlock;
      dftfe::utils::MemoryStorage<unsigned int,
                                  dftfe::utils::MemorySpace::DEVICE>
        projecterKetTimesFlattenedVectorLocalIdsD;
      dftfe::utils::MemoryStorage<unsigned int,
                                  dftfe::utils::MemorySpace::DEVICE>
        nonTrivialIdToElemIdMapD;
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST_PINNED>
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp;
      if (totalNonTrivialPseudoWfcs > 0)
        {
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedDBlock
            .resize(innerBlockSizeEnlp * numQuadsNLP * 3,
                    dataTypes::number(0.0));
#ifdef USE_COMPLEX
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedDBlock
            .resize(innerBlockSizeEnlp * numQuadsNLP, dataTypes::number(0.0));
#endif
          projecterKetTimesFlattenedVectorLocalIdsD.resize(
            totalNonTrivialPseudoWfcs, 0.0);
          nonTrivialIdToElemIdMapD.resize(totalNonTrivialPseudoWfcs, 0);



          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp
            .resize(innerBlockSizeEnlp * numQuadsNLP * 3, 0);


          dftfe::utils::deviceMemcpyH2D(nonTrivialIdToElemIdMapD.begin(),
                                        nonTrivialIdToElemIdMapH,
                                        totalNonTrivialPseudoWfcs *
                                          sizeof(unsigned int));


          dftfe::utils::deviceMemcpyH2D(
            projecterKetTimesFlattenedVectorLocalIdsD.begin(),
            projecterKetTimesFlattenedVectorLocalIdsH,
            totalNonTrivialPseudoWfcs * sizeof(unsigned int));
        }

      const unsigned numKPoints = kPointCoordinates.size() / 3;
      for (unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
        {
          elocWfcEshelbyTensorQuadValuesD.setValue(0);
          // spin index update is not required
          operatorMatrix.reinitkPointSpinIndex(kPoint, 0);

          const double kcoordx = kPointCoordinates[kPoint * 3 + 0];
          const double kcoordy = kPointCoordinates[kPoint * 3 + 1];
          const double kcoordz = kPointCoordinates[kPoint * 3 + 2];

          if (totalNonTrivialPseudoWfcs > 0)
            {
              std::fill(
                projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH +
                  kPoint * totalNonTrivialPseudoWfcs * numQuadsNLP * 3,
                projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH +
                  (kPoint + 1) * totalNonTrivialPseudoWfcs * numQuadsNLP * 3,
                dataTypes::number(0.0));

#ifdef USE_COMPLEX
              std::fill(
                projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH +
                  kPoint * totalNonTrivialPseudoWfcs * numQuadsNLP,
                projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH +
                  (kPoint + 1) * totalNonTrivialPseudoWfcs * numQuadsNLP,
                dataTypes::number(0.0));
#endif
            }

          for (unsigned int ivec = 0; ivec < N; ivec += blockSize)
            {
              if ((ivec + blockSize) <=
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                  (ivec + blockSize) >
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  std::vector<double> blockedEigenValues(blockSize, 0.0);
                  std::vector<double> blockedPartialOccupancies(blockSize, 0.0);
                  for (unsigned int iWave = 0; iWave < blockSize; ++iWave)
                    {
                      blockedEigenValues[iWave] =
                        eigenValuesH[kPoint][spinIndex * N + ivec + iWave];
                      blockedPartialOccupancies[iWave] =
                        partialOccupanciesH[kPoint]
                                           [spinIndex * N + ivec + iWave];
                    }

                  dftfe::utils::deviceMemcpyH2D(eigenValuesD.begin(),
                                                &blockedEigenValues[0],
                                                blockSize * sizeof(double));

                  dftfe::utils::deviceMemcpyH2D(partialOccupanciesD.begin(),
                                                &blockedPartialOccupancies[0],
                                                blockSize * sizeof(double));

                  // dftfe::utils::deviceSynchronize();
                  // MPI_Barrier(d_mpiCommParent);
                  // double kernel_time = MPI_Wtime();

                  devicePortedForceKernelsAllD(
                    basisOperationsPtr,
                    BLASWrapperPtr,
                    operatorMatrix,
                    deviceFlattenedArrayBlock,
                    projectorKetTimesVectorD,
                    X +
                      ((1 + spinPolarizedFlag) * kPoint + spinIndex) * MLoc * N,
                    eigenValuesD,
                    partialOccupanciesD,
#ifdef USE_COMPLEX
                    kcoordx,
                    kcoordy,
                    kcoordz,
#endif
                    onesVecD,
                    onesVecDNLP,
                    nonTrivialIdToElemIdMapD,
                    projecterKetTimesFlattenedVectorLocalIdsD,
                    ivec,
                    N,
                    blockSize,
                    numCells,
                    numQuads,
                    numQuadsNLP,
                    numNodesPerElement,
                    totalNonTrivialPseudoWfcs,
                    psiQuadsFlatD,
                    gradPsiQuadsFlatD,
#ifdef USE_COMPLEX
                    psiQuadsNLPD,
#endif
                    gradPsiQuadsNLPFlatD,
                    eshelbyTensorContributionsD,
                    elocWfcEshelbyTensorQuadValuesD,
                    nlpContractionContributionD,
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedDBlock,
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH +
                      kPoint * totalNonTrivialPseudoWfcs * numQuadsNLP * 3,
#ifdef USE_COMPLEX
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedDBlock,
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH +
                      kPoint * totalNonTrivialPseudoWfcs * numQuadsNLP,
#endif
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp
                      .begin(),
                    cellsBlockSize,
                    innerBlockSizeEnlp,
                    isPsp,
                    isFloatingChargeForces,
                    addEk);

                  // dftfe::utils::deviceSynchronize();
                  // MPI_Barrier(d_mpiCommParent);
                  // kernel_time = MPI_Wtime() - kernel_time;

                  // if (this_process==0 && dftParameters::verbosity>=5)
                  //   std::cout<<"Time for force kernels all insided block
                  //   loop:
                  //   "<<kernel_time<<std::endl;
                } // band parallelization
            }     // ivec loop

          dftfe::utils::deviceMemcpyD2H(eshelbyTensorQuadValuesH +
                                          kPoint * numCells * numQuads * 9,
                                        elocWfcEshelbyTensorQuadValuesD.begin(),
                                        numCells * numQuads * 9 *
                                          sizeof(double));
        } // k point loop

      // dftfe::utils::deviceSynchronize();
      // MPI_Barrier(mpiCommParent);
      // device_time = MPI_Wtime() - device_time;

      // if (this_process == 0 && dftParams.verbosity >= 1)
      //  std::cout << "Time taken for all device kernels force computation: "
      //            << device_time << std::endl;
    }

  } // namespace forceDevice
} // namespace dftfe
