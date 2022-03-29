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
#include "dftParameters.h"
#include "dftUtils.h"
#include "forceCUDA.h"
#include "vectorUtilities.h"
#include "cudaHelpers.h"
#include "linearAlgebraOperationsCUDA.h"

namespace dftfe
{
  namespace forceCUDA
  {
    namespace
    {
      template <typename numberType>
      __global__ void
      stridedCopyToBlockKernel(const unsigned int BVec,
                               const numberType * xVec,
                               const unsigned int M,
                               const unsigned int N,
                               numberType *       yVec,
                               const unsigned int startingXVecId)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries = M * BVec;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            unsigned int blockIndex      = index / BVec;
            unsigned int intraBlockIndex = index - blockIndex * BVec;
            yVec[index] =
              xVec[blockIndex * N + startingXVecId + intraBlockIndex];
          }
      }


      template <typename numberType>
      __global__ void
      copyCUDAKernel(const unsigned int contiguousBlockSize,
                     const unsigned int numContiguousBlocks,
                     const numberType * copyFromVec,
                     numberType *       copyToVec,
                     const dealii::types::global_dof_index
                       *copyFromVecStartingContiguousBlockIds)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            unsigned int blockIndex = index / contiguousBlockSize;
            unsigned int intraBlockIndex =
              index - blockIndex * contiguousBlockSize;
            copyToVec[index] =
              copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex];
          }
      }


      __global__ void
      computeELocWfcEshelbyTensorContributions(
        const unsigned int contiguousBlockSize,
        const unsigned int numContiguousBlocks,
        const unsigned int numQuads,
        const double *     psiQuadValues,
        const double *     gradPsiQuadValuesX,
        const double *     gradPsiQuadValuesY,
        const double *     gradPsiQuadValuesZ,
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
            const double psi        = psiQuadValues[tempIndex];
            const double gradPsiX   = gradPsiQuadValuesX[tempIndex];
            const double gradPsiY   = gradPsiQuadValuesY[tempIndex];
            const double gradPsiZ   = gradPsiQuadValuesZ[tempIndex];
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
        const unsigned int     contiguousBlockSize,
        const unsigned int     numContiguousBlocks,
        const unsigned int     numQuads,
        const cuDoubleComplex *psiQuadValues,
        const cuDoubleComplex *gradPsiQuadValuesX,
        const cuDoubleComplex *gradPsiQuadValuesY,
        const cuDoubleComplex *gradPsiQuadValuesZ,
        const double *         eigenValues,
        const double *         partialOccupancies,
        const double           kcoordx,
        const double           kcoordy,
        const double           kcoordz,
        double *               eshelbyTensor,
        const bool             addEk)
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
            const cuDoubleComplex psi      = psiQuadValues[tempIndex];
            const cuDoubleComplex psiConj  = cuConj(psiQuadValues[tempIndex]);
            const cuDoubleComplex gradPsiX = gradPsiQuadValuesX[tempIndex];
            const cuDoubleComplex gradPsiY = gradPsiQuadValuesY[tempIndex];
            const cuDoubleComplex gradPsiZ = gradPsiQuadValuesZ[tempIndex];
            const cuDoubleComplex gradPsiXConj =
              cuConj(gradPsiQuadValuesX[tempIndex]);
            const cuDoubleComplex gradPsiYConj =
              cuConj(gradPsiQuadValuesY[tempIndex]);
            const cuDoubleComplex gradPsiZConj =
              cuConj(gradPsiQuadValuesZ[tempIndex]);
            const double eigenValue = eigenValues[intraBlockIndex];
            const double partOcc    = partialOccupancies[intraBlockIndex];

            const double identityFactor =
              partOcc * ((cuCmul(gradPsiXConj, gradPsiX).x +
                          cuCmul(gradPsiYConj, gradPsiY).x +
                          cuCmul(gradPsiZConj, gradPsiZ).x) +
                         2.0 * (kcoordx * cuCmul(psiConj, gradPsiX).y +
                                kcoordy * cuCmul(psiConj, gradPsiY).y +
                                kcoordz * cuCmul(psiConj, gradPsiZ).y) +
                         (kcoordx * kcoordx + kcoordy * kcoordy +
                          kcoordz * kcoordz - 2.0 * eigenValue) *
                           cuCmul(psiConj, psi).x);
            if (addEk)
              {
                if (eshelbyIndex == 0)
                  eshelbyTensor[index] =
                    -2.0 * partOcc * cuCmul(gradPsiXConj, gradPsiX).x +
                    -2.0 * partOcc * cuCmul(psiConj, gradPsiX).y * kcoordx -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiX).y * kcoordx -
                    2.0 * partOcc * cuCmul(psiConj, psi).x * kcoordx * kcoordx +
                    identityFactor;
                else if (eshelbyIndex == 1)
                  eshelbyTensor[index] =
                    -2.0 * partOcc * cuCmul(gradPsiXConj, gradPsiY).x +
                    -2.0 * partOcc * cuCmul(psiConj, gradPsiX).y * kcoordy -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiY).y * kcoordx -
                    2.0 * partOcc * cuCmul(psiConj, psi).x * kcoordx * kcoordy;
                else if (eshelbyIndex == 2)
                  eshelbyTensor[index] =
                    -2.0 * partOcc * cuCmul(gradPsiXConj, gradPsiZ).x +
                    -2.0 * partOcc * cuCmul(psiConj, gradPsiX).y * kcoordz -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiZ).y * kcoordx -
                    2.0 * partOcc * cuCmul(psiConj, psi).x * kcoordx * kcoordz;
                else if (eshelbyIndex == 3)
                  eshelbyTensor[index] =
                    -2.0 * partOcc * cuCmul(gradPsiYConj, gradPsiX).x +
                    -2.0 * partOcc * cuCmul(psiConj, gradPsiY).y * kcoordx -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiX).y * kcoordy -
                    2.0 * partOcc * cuCmul(psiConj, psi).x * kcoordy * kcoordx;
                else if (eshelbyIndex == 4)
                  eshelbyTensor[index] =
                    -2.0 * partOcc * cuCmul(gradPsiYConj, gradPsiY).x +
                    -2.0 * partOcc * cuCmul(psiConj, gradPsiY).y * kcoordy -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiY).y * kcoordy -
                    2.0 * partOcc * cuCmul(psiConj, psi).x * kcoordy * kcoordy +
                    identityFactor;
                else if (eshelbyIndex == 5)
                  eshelbyTensor[index] =
                    -2.0 * partOcc * cuCmul(gradPsiYConj, gradPsiZ).x -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiY).y * kcoordz -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiZ).y * kcoordy -
                    2.0 * partOcc * cuCmul(psiConj, psi).x * kcoordy * kcoordz;
                else if (eshelbyIndex == 6)
                  eshelbyTensor[index] =
                    -2.0 * partOcc * cuCmul(gradPsiZConj, gradPsiX).x +
                    -2.0 * partOcc * cuCmul(psiConj, gradPsiZ).y * kcoordx -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiX).y * kcoordz -
                    2.0 * partOcc * cuCmul(psiConj, psi).x * kcoordz * kcoordx;
                else if (eshelbyIndex == 7)
                  eshelbyTensor[index] =
                    -2.0 * partOcc * cuCmul(gradPsiZConj, gradPsiY).x -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiZ).y * kcoordy -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiY).y * kcoordz -
                    2.0 * partOcc * cuCmul(psiConj, psi).x * kcoordz * kcoordy;
                else if (eshelbyIndex == 8)
                  eshelbyTensor[index] =
                    -2.0 * partOcc * cuCmul(gradPsiZConj, gradPsiZ).x +
                    -2.0 * partOcc * cuCmul(psiConj, gradPsiZ).y * kcoordz -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiZ).y * kcoordz -
                    2.0 * partOcc * cuCmul(psiConj, psi).x * kcoordz * kcoordz +
                    identityFactor;
              }
            else
              {
                if (eshelbyIndex == 0)
                  eshelbyTensor[index] =
                    -2.0 * partOcc * cuCmul(gradPsiXConj, gradPsiX).x -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiX).y * kcoordx +
                    identityFactor;
                else if (eshelbyIndex == 1)
                  eshelbyTensor[index] =
                    -2.0 * partOcc * cuCmul(gradPsiXConj, gradPsiY).x -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiX).y * kcoordy;
                else if (eshelbyIndex == 2)
                  eshelbyTensor[index] =
                    -2.0 * partOcc * cuCmul(gradPsiXConj, gradPsiZ).x -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiX).y * kcoordz;
                else if (eshelbyIndex == 3)
                  eshelbyTensor[index] =
                    -2.0 * partOcc * cuCmul(gradPsiYConj, gradPsiX).x -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiY).y * kcoordx;
                else if (eshelbyIndex == 4)
                  eshelbyTensor[index] =
                    -2.0 * partOcc * cuCmul(gradPsiYConj, gradPsiY).x -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiY).y * kcoordy +
                    identityFactor;
                else if (eshelbyIndex == 5)
                  eshelbyTensor[index] =
                    -2.0 * partOcc * cuCmul(gradPsiYConj, gradPsiZ).x -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiY).y * kcoordz;
                else if (eshelbyIndex == 6)
                  eshelbyTensor[index] =
                    -2.0 * partOcc * cuCmul(gradPsiZConj, gradPsiX).x -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiZ).y * kcoordx;
                else if (eshelbyIndex == 7)
                  eshelbyTensor[index] =
                    -2.0 * partOcc * cuCmul(gradPsiZConj, gradPsiY).x -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiZ).y * kcoordy;
                else if (eshelbyIndex == 8)
                  eshelbyTensor[index] =
                    -2.0 * partOcc * cuCmul(gradPsiZConj, gradPsiZ).x -
                    2.0 * partOcc * cuCmul(psiConj, gradPsiZ).y * kcoordz +
                    identityFactor;
              }
          }
      }


      __global__ void
      nlpContractionContributionPsiIndexCUDAKernel(
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
      nlpContractionContributionPsiIndexCUDAKernel(
        const unsigned int     numPsi,
        const unsigned int     numQuadsNLP,
        const unsigned int     totalNonTrivialPseudoWfcs,
        const unsigned int     startingId,
        const cuDoubleComplex *projectorKetTimesVectorPar,
        const cuDoubleComplex *gradPsiOrPsiQuadValuesNLP,
        const double *         partialOccupancies,
        const unsigned int *   nonTrivialIdToElemIdMap,
        const unsigned int *   projecterKetTimesFlattenedVectorLocalIds,
        cuDoubleComplex *      nlpContractionContribution)
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

            const cuDoubleComplex temp = cuCmul(
              cuConj(
                gradPsiOrPsiQuadValuesNLP[nonTrivialIdToElemIdMap[pseudoWfcId] *
                                            numQuadsNLP * numPsi +
                                          quadId * numPsi + wfcId]),
              projectorKetTimesVectorPar
                [projecterKetTimesFlattenedVectorLocalIds[pseudoWfcId] *
                   numPsi +
                 wfcId]);
            nlpContractionContribution[index] =
              make_cuDoubleComplex(partialOccupancies[wfcId] * temp.x,
                                   partialOccupancies[wfcId] * temp.y);
          }
      }

      __global__ void
      copyCUDAKernel(const unsigned int size,
                     const double *     copyFromVec,
                     double *           copyToVec)
      {
        for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
             index < size;
             index += blockDim.x * gridDim.x)
          copyToVec[index] = copyFromVec[index];
      }

      __global__ void
      copyCUDAKernel(const unsigned int size,
                     const double *     copyFromVec,
                     cuDoubleComplex *  copyToVec)
      {
        for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
             index < size;
             index += blockDim.x * gridDim.x)
          {
            copyToVec[index] = make_cuDoubleComplex(copyFromVec[index], 0.0);
          }
      }

      void
      copyDoubleToNumber(const double *     copyFromVec,
                         const unsigned int size,
                         double *           copyToVec)
      {
        copyCUDAKernel<<<(size + 255) / 256, 256>>>(size,
                                                    copyFromVec,
                                                    copyToVec);
      }

      void
      copyDoubleToNumber(const double *     copyFromVec,
                         const unsigned int size,
                         cuDoubleComplex *  copyToVec)
      {
        copyCUDAKernel<<<(size + 255) / 256, 256>>>(size,
                                                    copyFromVec,
                                                    copyToVec);
      }

      void
      interpolatePsiComputeELocWfcEshelbyTensorD(
        operatorDFTCUDAClass &                   operatorMatrix,
        distributedGPUVec<dataTypes::numberGPU> &Xb,
        const unsigned int                       BVec,
        const unsigned int                       numCells,
        const unsigned int                       numQuads,
        const unsigned int                       numQuadsNLP,
        const unsigned int                       numNodesPerElement,
        const thrust::device_vector<double> &    eigenValuesD,
        const thrust::device_vector<double> &    partialOccupanciesD,
#ifdef USE_COMPLEX
        const double kcoordx,
        const double kcoordy,
        const double kcoordz,
#endif
        const thrust::device_vector<double> &              onesVecD,
        const unsigned int                                 cellsBlockSize,
        thrust::device_vector<dataTypes::numberThrustGPU> &psiQuadsFlatD,
        thrust::device_vector<dataTypes::numberThrustGPU> &gradPsiQuadsXFlatD,
        thrust::device_vector<dataTypes::numberThrustGPU> &gradPsiQuadsYFlatD,
        thrust::device_vector<dataTypes::numberThrustGPU> &gradPsiQuadsZFlatD,
#ifdef USE_COMPLEX
        thrust::device_vector<dataTypes::numberThrustGPU> &psiQuadsNLPD,
#endif
        thrust::device_vector<dataTypes::numberThrustGPU> &gradPsiQuadsNLPFlatD,
        thrust::device_vector<double> &eshelbyTensorContributionsD,
        thrust::device_vector<double> &eshelbyTensorQuadValuesD,
        const bool                     isPsp,
        const bool                     isFloatingChargeForces,
        const bool                     addEk)
      {
        thrust::device_vector<dataTypes::numberThrustGPU>
          &cellWaveFunctionMatrix = operatorMatrix.getCellWaveFunctionMatrix();

        copyCUDAKernel<<<(BVec + 255) / 256 * numCells * numNodesPerElement,
                         256>>>(
          BVec,
          numCells * numNodesPerElement,
          Xb.begin(),
          reinterpret_cast<dataTypes::numberGPU *>(
            thrust::raw_pointer_cast(&cellWaveFunctionMatrix[0])),
          thrust::raw_pointer_cast(
            &(operatorMatrix.getFlattenedArrayCellLocalProcIndexIdMap())[0]));

        const int blockSize    = cellsBlockSize;
        const int numberBlocks = numCells / blockSize;
        const int remBlockSize = numCells - numberBlocks * blockSize;

        thrust::device_vector<dataTypes::numberThrustGPU>
          shapeFunctionValuesReferenceD(numQuads * numNodesPerElement,
                                        dataTypes::numberThrustGPU(0.0));
        thrust::device_vector<dataTypes::numberThrustGPU>
          shapeFunctionValuesNLPReferenceD(numQuadsNLP * numNodesPerElement,
                                           dataTypes::numberThrustGPU(0.0));

        copyDoubleToNumber(
          thrust::raw_pointer_cast(
            &(operatorMatrix.getShapeFunctionValuesInverted())[0]),
          numQuads * numNodesPerElement,
          reinterpret_cast<dataTypes::numberGPU *>(
            thrust::raw_pointer_cast(&shapeFunctionValuesReferenceD[0])));

        copyDoubleToNumber(
          thrust::raw_pointer_cast(
            &(operatorMatrix.getShapeFunctionValuesNLPInverted())[0]),
          numQuadsNLP * numNodesPerElement,
          reinterpret_cast<dataTypes::numberGPU *>(
            thrust::raw_pointer_cast(&shapeFunctionValuesNLPReferenceD[0])));

        thrust::device_vector<dataTypes::numberThrustGPU>
          shapeFunctionGradientValuesXInvertedDevice(
            blockSize * numQuads * numNodesPerElement,
            dataTypes::numberThrustGPU(0.0));

        thrust::device_vector<dataTypes::numberThrustGPU>
          shapeFunctionGradientValuesYInvertedDevice(
            blockSize * numQuads * numNodesPerElement,
            dataTypes::numberThrustGPU(0.0));

        thrust::device_vector<dataTypes::numberThrustGPU>
          shapeFunctionGradientValuesZInvertedDevice(
            blockSize * numQuads * numNodesPerElement,
            dataTypes::numberThrustGPU(0.0));

        thrust::device_vector<double> shapeFunctionGradientValuesNLPReferenceD(
          blockSize * numQuadsNLP * 3 * numNodesPerElement, 0.0);
        thrust::device_vector<double> shapeFunctionGradientValuesNLPD(
          blockSize * numQuadsNLP * 3 * numNodesPerElement, 0.0);
        thrust::device_vector<dataTypes::numberThrustGPU>
          shapeFunctionGradientValuesNLPDCopy(blockSize * numQuadsNLP * 3 *
                                                numNodesPerElement,
                                              dataTypes::numberThrustGPU(0.0));

        for (unsigned int i = 0; i < blockSize; i++)
          thrust::copy(
            operatorMatrix.getShapeFunctionGradientValuesNLPInverted().begin(),
            operatorMatrix.getShapeFunctionGradientValuesNLPInverted().end(),
            shapeFunctionGradientValuesNLPReferenceD.begin() +
              i * numQuadsNLP * 3 * numNodesPerElement);



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
                    dftfe::cublasXgemmStridedBatched(
                      operatorMatrix.getCublasHandle(),
                      CUBLAS_OP_N,
                      CUBLAS_OP_N,
                      BVec,
                      numQuads,
                      numNodesPerElement,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        &scalarCoeffAlpha),
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(
                          &cellWaveFunctionMatrix[startingId *
                                                  numNodesPerElement * BVec])),
                      BVec,
                      strideA,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(
                          &shapeFunctionValuesReferenceD[0])),
                      numNodesPerElement,
                      strideB,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        &scalarCoeffBeta),
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&psiQuadsFlatD[0])),
                      BVec,
                      strideC,
                      currentBlockSize);

                    strideB = numNodesPerElement * numQuads;

                    copyDoubleToNumber(
                      thrust::raw_pointer_cast(
                        &(operatorMatrix
                            .getShapeFunctionGradientValuesXInverted())
                          [startingId * numQuads * numNodesPerElement]),
                      currentBlockSize * numQuads * numNodesPerElement,
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(
                          &shapeFunctionGradientValuesXInvertedDevice[0])));

                    dftfe::cublasXgemmStridedBatched(
                      operatorMatrix.getCublasHandle(),
                      CUBLAS_OP_N,
                      CUBLAS_OP_N,
                      BVec,
                      numQuads,
                      numNodesPerElement,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        &scalarCoeffAlpha),
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(
                          &cellWaveFunctionMatrix[startingId *
                                                  numNodesPerElement * BVec])),
                      BVec,
                      strideA,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(
                          &shapeFunctionGradientValuesXInvertedDevice[0])),
                      numNodesPerElement,
                      strideB,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        &scalarCoeffBeta),
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&gradPsiQuadsXFlatD[0])),
                      BVec,
                      strideC,
                      currentBlockSize);

                    copyDoubleToNumber(
                      thrust::raw_pointer_cast(
                        &(operatorMatrix
                            .getShapeFunctionGradientValuesYInverted())
                          [startingId * numQuads * numNodesPerElement]),
                      currentBlockSize * numQuads * numNodesPerElement,
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(
                          &shapeFunctionGradientValuesYInvertedDevice[0])));

                    dftfe::cublasXgemmStridedBatched(
                      operatorMatrix.getCublasHandle(),
                      CUBLAS_OP_N,
                      CUBLAS_OP_N,
                      BVec,
                      numQuads,
                      numNodesPerElement,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        &scalarCoeffAlpha),
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(
                          &cellWaveFunctionMatrix[startingId *
                                                  numNodesPerElement * BVec])),
                      BVec,
                      strideA,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(
                          &shapeFunctionGradientValuesYInvertedDevice[0])),
                      numNodesPerElement,
                      strideB,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        &scalarCoeffBeta),
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&gradPsiQuadsYFlatD[0])),
                      BVec,
                      strideC,
                      currentBlockSize);

                    copyDoubleToNumber(
                      thrust::raw_pointer_cast(
                        &(operatorMatrix
                            .getShapeFunctionGradientValuesZInverted())
                          [startingId * numQuads * numNodesPerElement]),
                      currentBlockSize * numQuads * numNodesPerElement,
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(
                          &shapeFunctionGradientValuesZInvertedDevice[0])));

                    dftfe::cublasXgemmStridedBatched(
                      operatorMatrix.getCublasHandle(),
                      CUBLAS_OP_N,
                      CUBLAS_OP_N,
                      BVec,
                      numQuads,
                      numNodesPerElement,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        &scalarCoeffAlpha),
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(
                          &cellWaveFunctionMatrix[startingId *
                                                  numNodesPerElement * BVec])),
                      BVec,
                      strideA,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(
                          &shapeFunctionGradientValuesZInvertedDevice[0])),
                      numNodesPerElement,
                      strideB,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        &scalarCoeffBeta),
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&gradPsiQuadsZFlatD[0])),
                      BVec,
                      strideC,
                      currentBlockSize);


                    computeELocWfcEshelbyTensorContributions<<<
                      (BVec + 255) / 256 * currentBlockSize * numQuads * 9,
                      256>>>(
                      BVec,
                      currentBlockSize * numQuads * 9,
                      numQuads,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&psiQuadsFlatD[0])),
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&gradPsiQuadsXFlatD[0])),
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&gradPsiQuadsYFlatD[0])),
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&gradPsiQuadsZFlatD[0])),
                      thrust::raw_pointer_cast(&eigenValuesD[0]),
                      thrust::raw_pointer_cast(&partialOccupanciesD[0]),
#ifdef USE_COMPLEX
                      kcoordx,
                      kcoordy,
                      kcoordz,
#endif
                      thrust::raw_pointer_cast(&eshelbyTensorContributionsD[0])
#ifdef USE_COMPLEX
                        ,
                      addEk
#endif
                    );

                    const double scalarCoeffAlphaEshelby = 1.0;
                    const double scalarCoeffBetaEshelby  = 1.0;



                    cublasDgemm(
                      operatorMatrix.getCublasHandle(),
                      CUBLAS_OP_N,
                      CUBLAS_OP_N,
                      1,
                      currentBlockSize * numQuads * 9,
                      BVec,
                      &scalarCoeffAlphaEshelby,
                      thrust::raw_pointer_cast(&onesVecD[0]),
                      1,
                      thrust::raw_pointer_cast(&eshelbyTensorContributionsD[0]),
                      BVec,
                      &scalarCoeffBetaEshelby,
                      thrust::raw_pointer_cast(
                        &eshelbyTensorQuadValuesD[startingId * numQuads * 9]),
                      1);
                  }

                if (isPsp)
                  {
#ifdef USE_COMPLEX
                    const int strideCNLP = BVec * numQuadsNLP;
                    const int strideBNLP = 0;

                    dftfe::cublasXgemmStridedBatched(
                      operatorMatrix.getCublasHandle(),
                      CUBLAS_OP_N,
                      CUBLAS_OP_N,
                      BVec,
                      numQuadsNLP,
                      numNodesPerElement,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        &scalarCoeffAlpha),
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(
                          &cellWaveFunctionMatrix[startingId *
                                                  numNodesPerElement * BVec])),
                      BVec,
                      strideA,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(
                          &(shapeFunctionValuesNLPReferenceD[0]))),
                      numNodesPerElement,
                      strideBNLP,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        &scalarCoeffBeta),
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(
                          &psiQuadsNLPD[startingId * numQuadsNLP * BVec])),
                      BVec,
                      strideCNLP,
                      currentBlockSize);
#endif

                    // shapeGradRef^T*invJacobian^T
                    cublasDgemmStridedBatched(
                      operatorMatrix.getCublasHandle(),
                      CUBLAS_OP_N,
                      CUBLAS_OP_N,
                      numNodesPerElement,
                      3,
                      3,
                      &scalarCoeffAlphaReal,
                      thrust::raw_pointer_cast(
                        &shapeFunctionGradientValuesNLPReferenceD[0]),
                      numNodesPerElement,
                      numNodesPerElement * 3,
                      thrust::raw_pointer_cast(
                        &(operatorMatrix
                            .getInverseJacobiansNLP())[startingId *
                                                       numQuadsNLP * 3 * 3]),
                      3,
                      3 * 3,
                      &scalarCoeffBetaReal,
                      thrust::raw_pointer_cast(
                        &shapeFunctionGradientValuesNLPD[0]),
                      numNodesPerElement,
                      numNodesPerElement * 3,
                      currentBlockSize * numQuadsNLP);

                    copyDoubleToNumber(
                      thrust::raw_pointer_cast(
                        &shapeFunctionGradientValuesNLPD[0]),
                      currentBlockSize * numQuadsNLP * numNodesPerElement * 3,
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(
                          &shapeFunctionGradientValuesNLPDCopy[0])));

                    const int strideCNLPGrad = BVec * 3 * numQuadsNLP;
                    const int strideBNLPGrad =
                      numNodesPerElement * 3 * numQuadsNLP;

                    dftfe::cublasXgemmStridedBatched(
                      operatorMatrix.getCublasHandle(),
                      CUBLAS_OP_N,
                      CUBLAS_OP_N,
                      BVec,
                      3 * numQuadsNLP,
                      numNodesPerElement,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        &scalarCoeffAlpha),
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(
                          &cellWaveFunctionMatrix[startingId *
                                                  numNodesPerElement * BVec])),
                      BVec,
                      strideA,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(
                          &shapeFunctionGradientValuesNLPDCopy[0])),
                      numNodesPerElement,
                      strideBNLPGrad,
                      reinterpret_cast<const dataTypes::numberGPU *>(
                        &scalarCoeffBeta),
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(
                          &gradPsiQuadsNLPFlatD[startingId * numQuadsNLP * 3 *
                                                BVec])),
                      BVec,
                      strideCNLPGrad,
                      currentBlockSize);
                  }
              }
          }
      }

      void
      nlpPsiContractionD(
        operatorDFTCUDAClass &operatorMatrix,
#ifdef USE_COMPLEX
        const thrust::device_vector<dataTypes::numberThrustGPU> &psiQuadsNLPD,
#endif
        const thrust::device_vector<dataTypes::numberThrustGPU>
          &                                  gradPsiQuadsNLPD,
        const thrust::device_vector<double> &partialOccupanciesD,
        const thrust::device_vector<dataTypes::numberThrustGPU> &onesVecDNLP,
        const dataTypes::numberGPU *projectorKetTimesVectorParFlattenedD,
        const thrust::device_vector<unsigned int> &nonTrivialIdToElemIdMapD,
        const thrust::device_vector<unsigned int>
          &                projecterKetTimesFlattenedVectorLocalIdsD,
        const unsigned int numCells,
        const unsigned int numQuadsNLP,
        const unsigned int numPsi,
        const unsigned int totalNonTrivialPseudoWfcs,
        const unsigned int innerBlockSizeEnlp,
        thrust::device_vector<dataTypes::numberThrustGPU>
          &nlpContractionContributionD,
        thrust::device_vector<dataTypes::numberThrustGPU> &
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedDBlock,
        dataTypes::number *
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
#ifdef USE_COMPLEX
        thrust::device_vector<dataTypes::numberThrustGPU> &
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
                nlpContractionContributionPsiIndexCUDAKernel<<<
                  (numPsi + 255) / 256 * numQuadsNLP * 3 * currentBlockSizeNlp,
                  256>>>(numPsi,
                         numQuadsNLP * 3,
                         currentBlockSizeNlp,
                         startingIdNlp,
                         projectorKetTimesVectorParFlattenedD,
                         reinterpret_cast<const dataTypes::numberGPU *>(
                           thrust::raw_pointer_cast(&gradPsiQuadsNLPD[0])),
                         thrust::raw_pointer_cast(&partialOccupanciesD[0]),
                         thrust::raw_pointer_cast(&nonTrivialIdToElemIdMapD[0]),
                         thrust::raw_pointer_cast(
                           &projecterKetTimesFlattenedVectorLocalIdsD[0]),
                         reinterpret_cast<dataTypes::numberGPU *>(
                           thrust::raw_pointer_cast(
                             &nlpContractionContributionD[0])));

                dftfe::cublasXgemm(
                  operatorMatrix.getCublasHandle(),
                  CUBLAS_OP_N,
                  CUBLAS_OP_N,
                  1,
                  currentBlockSizeNlp * numQuadsNLP * 3,
                  numPsi,
                  reinterpret_cast<const dataTypes::numberGPU *>(
                    &scalarCoeffAlphaNlp),
                  reinterpret_cast<const dataTypes::numberGPU *>(
                    thrust::raw_pointer_cast(&onesVecDNLP[0])),
                  1,
                  reinterpret_cast<const dataTypes::numberGPU *>(
                    thrust::raw_pointer_cast(&nlpContractionContributionD[0])),
                  numPsi,
                  reinterpret_cast<const dataTypes::numberGPU *>(
                    &scalarCoeffBetaNlp),
                  reinterpret_cast<
                    dataTypes::numberGPU *>(thrust::raw_pointer_cast(
                    &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedDBlock
                      [0])),
                  1);

                cudaMemcpy(
                  reinterpret_cast<dataTypes::numberGPU *>(
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp),
                  reinterpret_cast<
                    const dataTypes::numberGPU *>(thrust::raw_pointer_cast(
                    &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedDBlock
                      [0])),
                  currentBlockSizeNlp * numQuadsNLP * 3 *
                    sizeof(dataTypes::numberGPU),
                  cudaMemcpyDeviceToHost);

                for (unsigned int i = 0;
                     i < currentBlockSizeNlp * numQuadsNLP * 3;
                     i++)
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH
                    [startingIdNlp * numQuadsNLP * 3 + i] +=
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp
                      [i];
#ifdef USE_COMPLEX
                nlpContractionContributionPsiIndexCUDAKernel<<<
                  (numPsi + 255) / 256 * numQuadsNLP * currentBlockSizeNlp,
                  256>>>(numPsi,
                         numQuadsNLP,
                         currentBlockSizeNlp,
                         startingIdNlp,
                         projectorKetTimesVectorParFlattenedD,
                         reinterpret_cast<const dataTypes::numberGPU *>(
                           thrust::raw_pointer_cast(&psiQuadsNLPD[0])),
                         thrust::raw_pointer_cast(&partialOccupanciesD[0]),
                         thrust::raw_pointer_cast(&nonTrivialIdToElemIdMapD[0]),
                         thrust::raw_pointer_cast(
                           &projecterKetTimesFlattenedVectorLocalIdsD[0]),
                         reinterpret_cast<dataTypes::numberGPU *>(
                           thrust::raw_pointer_cast(
                             &nlpContractionContributionD[0])));

                dftfe::cublasXgemm(
                  operatorMatrix.getCublasHandle(),
                  CUBLAS_OP_N,
                  CUBLAS_OP_N,
                  1,
                  currentBlockSizeNlp * numQuadsNLP,
                  numPsi,
                  reinterpret_cast<const dataTypes::numberGPU *>(
                    &scalarCoeffAlphaNlp),
                  reinterpret_cast<const dataTypes::numberGPU *>(
                    thrust::raw_pointer_cast(&onesVecDNLP[0])),
                  1,
                  reinterpret_cast<const dataTypes::numberGPU *>(
                    thrust::raw_pointer_cast(&nlpContractionContributionD[0])),
                  numPsi,
                  reinterpret_cast<const dataTypes::numberGPU *>(
                    &scalarCoeffBetaNlp),
                  reinterpret_cast<
                    dataTypes::numberGPU *>(thrust::raw_pointer_cast(
                    &projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedDBlock
                      [0])),
                  1);

                cudaMemcpy(
                  reinterpret_cast<dataTypes::numberGPU *>(
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp),
                  reinterpret_cast<
                    const dataTypes::numberGPU *>(thrust::raw_pointer_cast(
                    &projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedDBlock
                      [0])),
                  currentBlockSizeNlp * numQuadsNLP *
                    sizeof(dataTypes::numberGPU),
                  cudaMemcpyDeviceToHost);

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
      gpuPortedForceKernelsAllD(
        operatorDFTCUDAClass &                   operatorMatrix,
        distributedGPUVec<dataTypes::numberGPU> &cudaFlattenedArrayBlock,
        distributedGPUVec<dataTypes::numberGPU> &projectorKetTimesVectorD,
        const dataTypes::numberGPU *             X,
        const thrust::device_vector<double> &    eigenValuesD,
        const thrust::device_vector<double> &    partialOccupanciesD,
#ifdef USE_COMPLEX
        const double kcoordx,
        const double kcoordy,
        const double kcoordz,
#endif
        const thrust::device_vector<double> &                    onesVecD,
        const thrust::device_vector<dataTypes::numberThrustGPU> &onesVecDNLP,
        const thrust::device_vector<unsigned int> &nonTrivialIdToElemIdMapD,
        const thrust::device_vector<unsigned int>
          &                projecterKetTimesFlattenedVectorLocalIdsD,
        const unsigned int startingVecId,
        const unsigned int N,
        const unsigned int numPsi,
        const unsigned int numCells,
        const unsigned int numQuads,
        const unsigned int numQuadsNLP,
        const unsigned int numNodesPerElement,
        const unsigned int totalNonTrivialPseudoWfcs,
        thrust::device_vector<dataTypes::numberThrustGPU> &psiQuadsFlatD,
        thrust::device_vector<dataTypes::numberThrustGPU> &gradPsiQuadsXFlatD,
        thrust::device_vector<dataTypes::numberThrustGPU> &gradPsiQuadsYFlatD,
        thrust::device_vector<dataTypes::numberThrustGPU> &gradPsiQuadsZFlatD,
#ifdef USE_COMPLEX
        thrust::device_vector<dataTypes::numberThrustGPU> &psiQuadsNLPD,
#endif
        thrust::device_vector<dataTypes::numberThrustGPU> &gradPsiQuadsNLPFlatD,
        thrust::device_vector<double> &eshelbyTensorContributionsD,
        thrust::device_vector<double> &eshelbyTensorQuadValuesD,
        thrust::device_vector<dataTypes::numberThrustGPU>
          &nlpContractionContributionD,
        thrust::device_vector<dataTypes::numberThrustGPU> &
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedDBlock,
        dataTypes::number *
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
#ifdef USE_COMPLEX
        thrust::device_vector<dataTypes::numberThrustGPU> &
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
        //int this_process;
        //MPI_Comm_rank(d_mpiCommParent, &this_process);

        const unsigned int M = operatorMatrix.getMatrixFreeData()
                                 ->get_vector_partitioner()
                                 ->local_size();
        stridedCopyToBlockKernel<<<(numPsi + 255) / 256 * M, 256>>>(
          numPsi, X, M, N, cudaFlattenedArrayBlock.begin(), startingVecId);
        cudaFlattenedArrayBlock.updateGhostValues();

        (operatorMatrix.getOverloadedConstraintMatrix())
          ->distribute(cudaFlattenedArrayBlock, numPsi);


        // cudaDeviceSynchronize();
        // MPI_Barrier(d_mpiCommParent);
        // double kernel1_time = MPI_Wtime();

        interpolatePsiComputeELocWfcEshelbyTensorD(operatorMatrix,
                                                   cudaFlattenedArrayBlock,
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
                                                   gradPsiQuadsXFlatD,
                                                   gradPsiQuadsYFlatD,
                                                   gradPsiQuadsZFlatD,
#ifdef USE_COMPLEX
                                                   psiQuadsNLPD,
#endif
                                                   gradPsiQuadsNLPFlatD,
                                                   eshelbyTensorContributionsD,
                                                   eshelbyTensorQuadValuesD,
                                                   isPsp,
                                                   isFloatingChargeForces,
                                                   addEk);

        // cudaDeviceSynchronize();
        // MPI_Barrier(d_mpiCommParent);
        // kernel1_time = MPI_Wtime() - kernel1_time;

        // if (this_process==0 && dftParameters::verbosity>=5)
        //	 std::cout<<"Time for
        // interpolatePsiComputeELocWfcEshelbyTensorD inside blocked
        // loop: "<<kernel1_time<<std::endl;

        if (isPsp)
          {
            // cudaDeviceSynchronize();
            // MPI_Barrier(d_mpiCommParent);
            // double kernel2_time = MPI_Wtime();

            operatorMatrix.computeNonLocalProjectorKetTimesXTimesV(
              cudaFlattenedArrayBlock.begin(),
              projectorKetTimesVectorD,
              numPsi);

            // cudaDeviceSynchronize();
            // MPI_Barrier(d_mpiCommParent);
            // kernel2_time = MPI_Wtime() - kernel2_time;

            // if (this_process==0 && dftParameters::verbosity>=5)
            //  std::cout<<"Time for computeNonLocalProjectorKetTimesXTimesV
            //  inside blocked loop: "<<kernel2_time<<std::endl;

            // cudaDeviceSynchronize();
            // MPI_Barrier(d_mpiCommParent);
            // double kernel3_time = MPI_Wtime();

            if (totalNonTrivialPseudoWfcs > 0)
              {
                nlpPsiContractionD(
                  operatorMatrix,
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

            // cudaDeviceSynchronize();
            // MPI_Barrier(d_mpiCommParent);
            // kernel3_time = MPI_Wtime() - kernel3_time;

            // if (this_process==0 && dftParameters::verbosity>=5)
            //	 std::cout<<"Time for nlpPsiContractionD inside blocked loop:
            //"<<kernel3_time<<std::endl;
          }
      }

    } // namespace

    void
    gpuPortedForceKernelsAllH(
      operatorDFTCUDAClass &      operatorMatrix,
      const dataTypes::numberGPU *X,
      const double *              eigenValuesH,
      const double *              partialOccupanciesH,
#ifdef USE_COMPLEX
      const double kcoordx,
      const double kcoordy,
      const double kcoordz,
#endif
      const unsigned int *nonTrivialIdToElemIdMapH,
      const unsigned int *projecterKetTimesFlattenedVectorLocalIdsH,
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
      const MPI_Comm &mpiCommParent,  
      const MPI_Comm &interBandGroupComm,
      const bool      isPsp,
      const bool      isFloatingChargeForces,
      const bool      addEk)
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
        std::min(dftParameters::chebyWfcBlockSize,
                 bandGroupLowHighPlusOneIndices[1]);

      int this_process;
      MPI_Comm_rank(mpiCommParent, &this_process);
      cudaDeviceSynchronize();
      MPI_Barrier(mpiCommParent);
      double gpu_time = MPI_Wtime();

      distributedGPUVec<dataTypes::numberGPU> &cudaFlattenedArrayBlock =
        operatorMatrix.getParallelChebyBlockVectorDevice();

      distributedGPUVec<dataTypes::numberGPU> &projectorKetTimesVectorD =
        operatorMatrix.getParallelProjectorKetTimesBlockVectorDevice();

      cudaDeviceSynchronize();
      MPI_Barrier(mpiCommParent);
      gpu_time = MPI_Wtime() - gpu_time;

      if (this_process == 0 && dftParameters::verbosity >= 2)
        std::cout
          << "Time for creating cuda parallel vectors for force computation: "
          << gpu_time << std::endl;

      gpu_time = MPI_Wtime();

      thrust::device_vector<double> eigenValuesD(blockSize, 0.0);
      thrust::device_vector<double> partialOccupanciesD(blockSize, 0.0);
      thrust::device_vector<double> elocWfcEshelbyTensorQuadValuesD(
        numCells * numQuads * 9, 0.0);

      thrust::device_vector<double> onesVecD(blockSize, 1.0);
      thrust::device_vector<dataTypes::numberThrustGPU> onesVecDNLP(
        blockSize, dataTypes::numberThrustGPU(1.0));

      const unsigned int cellsBlockSize = std::min((unsigned int)10, numCells);

      thrust::device_vector<dataTypes::numberThrustGPU> psiQuadsFlatD(
        cellsBlockSize * numQuads * blockSize, dataTypes::numberThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberThrustGPU> gradPsiQuadsXFlatD(
        cellsBlockSize * numQuads * blockSize, dataTypes::numberThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberThrustGPU> gradPsiQuadsYFlatD(
        cellsBlockSize * numQuads * blockSize, dataTypes::numberThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberThrustGPU> gradPsiQuadsZFlatD(
        cellsBlockSize * numQuads * blockSize, dataTypes::numberThrustGPU(0.0));
#ifdef USE_COMPLEX
      thrust::device_vector<dataTypes::numberThrustGPU> psiQuadsNLPD(
        numCells * numQuadsNLP * blockSize, dataTypes::numberThrustGPU(0.0));
#endif

      thrust::device_vector<dataTypes::numberThrustGPU> gradPsiQuadsNLPFlatD(
        numCells * numQuadsNLP * 3 * blockSize,
        dataTypes::numberThrustGPU(0.0));

      thrust::device_vector<double> eshelbyTensorContributionsD(
        cellsBlockSize * numQuads * blockSize * 9, 0.0);

      const unsigned int innerBlockSizeEnlp =
        std::min((unsigned int)10, totalNonTrivialPseudoWfcs);
      thrust::device_vector<dataTypes::numberThrustGPU>
        nlpContractionContributionD(innerBlockSizeEnlp * numQuadsNLP * 3 *
                                      blockSize,
                                    dataTypes::numberThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberThrustGPU>
        projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedDBlock;
      thrust::device_vector<dataTypes::numberThrustGPU>
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedDBlock;
      thrust::device_vector<unsigned int>
                                          projecterKetTimesFlattenedVectorLocalIdsD;
      thrust::device_vector<unsigned int> nonTrivialIdToElemIdMapD;
      dataTypes::number *
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp;
      if (totalNonTrivialPseudoWfcs > 0)
        {
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedDBlock
            .resize(innerBlockSizeEnlp * numQuadsNLP * 3,
                    dataTypes::numberThrustGPU(0.0));
#ifdef USE_COMPLEX
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedDBlock
            .resize(innerBlockSizeEnlp * numQuadsNLP,
                    dataTypes::numberThrustGPU(0.0));
#endif
          projecterKetTimesFlattenedVectorLocalIdsD.resize(
            totalNonTrivialPseudoWfcs, 0.0);
          nonTrivialIdToElemIdMapD.resize(totalNonTrivialPseudoWfcs, 0);


          CUDACHECK(cudaMallocHost(
            (void *
               *)&projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp,
            innerBlockSizeEnlp * numQuadsNLP * 3 *
              sizeof(dataTypes::numberGPU)));

          std::fill(
            projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
            projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH +
              totalNonTrivialPseudoWfcs * numQuadsNLP * 3,
            dataTypes::number(0.0));

#ifdef USE_COMPLEX
          std::fill(
            projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
            projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH +
              totalNonTrivialPseudoWfcs * numQuadsNLP,
            dataTypes::number(0.0));
#endif

          cudaMemcpy(thrust::raw_pointer_cast(&nonTrivialIdToElemIdMapD[0]),
                     nonTrivialIdToElemIdMapH,
                     totalNonTrivialPseudoWfcs * sizeof(unsigned int),
                     cudaMemcpyHostToDevice);


          cudaMemcpy(thrust::raw_pointer_cast(
                       &projecterKetTimesFlattenedVectorLocalIdsD[0]),
                     projecterKetTimesFlattenedVectorLocalIdsH,
                     totalNonTrivialPseudoWfcs * sizeof(unsigned int),
                     cudaMemcpyHostToDevice);
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
                  blockedEigenValues[iWave] = eigenValuesH[ivec + iWave];
                  blockedPartialOccupancies[iWave] =
                    partialOccupanciesH[ivec + iWave];
                }

              cudaMemcpy(thrust::raw_pointer_cast(&eigenValuesD[0]),
                         &blockedEigenValues[0],
                         blockSize * sizeof(double),
                         cudaMemcpyHostToDevice);

              cudaMemcpy(thrust::raw_pointer_cast(&partialOccupanciesD[0]),
                         &blockedPartialOccupancies[0],
                         blockSize * sizeof(double),
                         cudaMemcpyHostToDevice);

              // cudaDeviceSynchronize();
              // MPI_Barrier(d_mpiCommParent);
              // double kernel_time = MPI_Wtime();

              gpuPortedForceKernelsAllD(
                operatorMatrix,
                cudaFlattenedArrayBlock,
                projectorKetTimesVectorD,
                X,
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
                gradPsiQuadsXFlatD,
                gradPsiQuadsYFlatD,
                gradPsiQuadsZFlatD,
#ifdef USE_COMPLEX
                psiQuadsNLPD,
#endif
                gradPsiQuadsNLPFlatD,
                eshelbyTensorContributionsD,
                elocWfcEshelbyTensorQuadValuesD,
                nlpContractionContributionD,
                projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedDBlock,
                projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
#ifdef USE_COMPLEX
                projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedDBlock,
                projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
#endif
                projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp,
                cellsBlockSize,
                innerBlockSizeEnlp,
                isPsp,
                isFloatingChargeForces,
                addEk);

              // cudaDeviceSynchronize();
              // MPI_Barrier(d_mpiCommParent);
              // kernel_time = MPI_Wtime() - kernel_time;

              // if (this_process==0 && dftParameters::verbosity>=5)
              //   std::cout<<"Time for force kernels all insided block loop:
              //   "<<kernel_time<<std::endl;
            } // band parallelization
        }     // ivec loop

      cudaMemcpy(eshelbyTensorQuadValuesH,
                 thrust::raw_pointer_cast(&elocWfcEshelbyTensorQuadValuesD[0]),
                 numCells * numQuads * 9 * sizeof(double),
                 cudaMemcpyDeviceToHost);

      if (totalNonTrivialPseudoWfcs > 0)
        CUDACHECK(cudaFreeHost(
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp));

      cudaDeviceSynchronize();
      MPI_Barrier(mpiCommParent);
      gpu_time = MPI_Wtime() - gpu_time;

      if (this_process == 0 && dftParameters::verbosity >= 1)
        std::cout << "Time taken for all gpu kernels force computation: "
                  << gpu_time << std::endl;
    }

  } // namespace forceCUDA
} // namespace dftfe
