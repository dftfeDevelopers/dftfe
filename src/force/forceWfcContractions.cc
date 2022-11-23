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
#include "forceWfcContractions.h"
#include "vectorUtilities.h"
#include "linearAlgebraOperations.h"

namespace dftfe
{
  namespace force
  {

    void
    wfcContractionsForceKernelsAllH(
      operatorDFTClass &      operatorMatrix,
      const std::vector<std::vector<dataTypes::number>> & X,
      const unsigned int spinPolarizedFlag, 
      const unsigned int spinIndex,
      const std::vector<std::vector<double>>  & eigenValuesH,
      const std::vector<std::vector<double>> &  partialOccupanciesH,
      const std::vector<double> & kPointCoordinates,   
      const unsigned int *nonTrivialIdToElemIdMapH,
      const unsigned int *projecterKetTimesFlattenedVectorLocalIdsH,
      const unsigned int MLoc,
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

      distributedGPUVec<dataTypes::numberGPU> &cudaFlattenedArrayBlock =
        operatorMatrix.getParallelChebyBlockVectorDevice();

      distributedGPUVec<dataTypes::numberGPU> &projectorKetTimesVectorD =
        operatorMatrix.getParallelProjectorKetTimesBlockVectorDevice();


      std::vector<double> eigenValuesD(blockSize, 0.0);
      std::vector<double> partialOccupanciesD(blockSize, 0.0);
      std::vector<double> elocWfcEshelbyTensorQuadValuesD(
        numCells * numQuads * 9, 0.0);

      std::vector<double> onesVecD(blockSize, 1.0);
      std::vector<dataTypes::number> onesVecDNLP(
        blockSize, dataTypes::number(1.0));

      const unsigned int cellsBlockSize = std::min((unsigned int)10, numCells);

      std::vector<dataTypes::number> psiQuadsFlatD(
        cellsBlockSize * numQuads * blockSize, dataTypes::number(0.0));
      std::vector<dataTypes::number> gradPsiQuadsXFlatD(
        cellsBlockSize * numQuads * blockSize, dataTypes::number(0.0));
      std::vector<dataTypes::number> gradPsiQuadsYFlatD(
        cellsBlockSize * numQuads * blockSize, dataTypes::number(0.0));
      std::vector<dataTypes::number> gradPsiQuadsZFlatD(
        cellsBlockSize * numQuads * blockSize, dataTypes::number(0.0));
#ifdef USE_COMPLEX
      std::vector<dataTypes::number> psiQuadsNLPD(
        numCells * numQuadsNLP * blockSize, dataTypes::number(0.0));
#endif

      std::vector<dataTypes::number> gradPsiQuadsNLPFlatD(
        numCells * numQuadsNLP * 3 * blockSize,
        dataTypes::number(0.0));

      std::vector<double> eshelbyTensorContributionsD(
        cellsBlockSize * numQuads * blockSize * 9, 0.0);

      const unsigned int innerBlockSizeEnlp =
        std::min((unsigned int)10, totalNonTrivialPseudoWfcs);
      std::vector<dataTypes::number>
        nlpContractionContributionD(innerBlockSizeEnlp * numQuadsNLP * 3 *
                                      blockSize,
                                    dataTypes::number(0.0));
      std::vector<dataTypes::number>
        projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedDBlock;
      std::vector<dataTypes::number>
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedDBlock;
      std::vector<unsigned int>
                                          projecterKetTimesFlattenedVectorLocalIdsD;
      std::vector<unsigned int> nonTrivialIdToElemIdMapD;
      dataTypes::number *
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp;
      if (totalNonTrivialPseudoWfcs > 0)
        {
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedDBlock
            .resize(innerBlockSizeEnlp * numQuadsNLP * 3,
                    dataTypes::number(0.0));
#ifdef USE_COMPLEX
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedDBlock
            .resize(innerBlockSizeEnlp * numQuadsNLP,
                    dataTypes::number(0.0));
#endif
          projecterKetTimesFlattenedVectorLocalIdsD.resize(
            totalNonTrivialPseudoWfcs, 0.0);
          nonTrivialIdToElemIdMapD.resize(totalNonTrivialPseudoWfcs, 0);


          CUDACHECK(cudaMallocHost(
            (void *
               *)&projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp,
            innerBlockSizeEnlp * numQuadsNLP * 3 *
              sizeof(dataTypes::numberGPU)));


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

        const unsigned numKPoints=kPointCoordinates.size()/3;
        for (unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
          {

          thrust::fill(elocWfcEshelbyTensorQuadValuesD.begin(),elocWfcEshelbyTensorQuadValuesD.end(),0.);
          //spin index update is not required
          operatorMatrix.reinitkPointSpinIndex(kPoint, 0);

          const double  kcoordx = kPointCoordinates[kPoint * 3 + 0];
          const double  kcoordy = kPointCoordinates[kPoint * 3 + 1];
          const double  kcoordz = kPointCoordinates[kPoint * 3 + 2];

          if (totalNonTrivialPseudoWfcs > 0)
            {

              std::fill(
                projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH
                +kPoint * totalNonTrivialPseudoWfcs*numQuadsNLP * 3,
                projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH
                + (kPoint+1) * totalNonTrivialPseudoWfcs*numQuadsNLP * 3,
                dataTypes::number(0.0));

#ifdef USE_COMPLEX
              std::fill(
                projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH
                +kPoint * totalNonTrivialPseudoWfcs*numQuadsNLP,
                projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH 
                +(kPoint+1) * totalNonTrivialPseudoWfcs*numQuadsNLP,
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
                      blockedEigenValues[iWave] = eigenValuesH[kPoint][spinIndex *N+ivec + iWave];
                      blockedPartialOccupancies[iWave] =
                        partialOccupanciesH[kPoint][spinIndex *N+ivec + iWave];
                    }

                  cudaMemcpy(thrust::raw_pointer_cast(&eigenValuesD[0]),
                             &blockedEigenValues[0],
                             blockSize * sizeof(double),
                             cudaMemcpyHostToDevice);

                  cudaMemcpy(thrust::raw_pointer_cast(&partialOccupanciesD[0]),
                             &blockedPartialOccupancies[0],
                             blockSize * sizeof(double),
                             cudaMemcpyHostToDevice);


                  gpuPortedForceKernelsAllD(
                    operatorMatrix,
                    cudaFlattenedArrayBlock,
                    projectorKetTimesVectorD,
                    X+((1 +spinPolarizedFlag) * kPoint + spinIndex) *MLoc * N,
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
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH+kPoint * totalNonTrivialPseudoWfcs*numQuadsNLP * 3,
#ifdef USE_COMPLEX
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedDBlock,
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH+kPoint * totalNonTrivialPseudoWfcs*numQuadsNLP,
#endif
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp,
                    cellsBlockSize,
                    innerBlockSizeEnlp,
                    isPsp,
                    isFloatingChargeForces,
                    addEk);

                } // band parallelization
            }     // ivec loop

          cudaMemcpy(eshelbyTensorQuadValuesH+kPoint * numCells *numQuads * 9,
                     thrust::raw_pointer_cast(&elocWfcEshelbyTensorQuadValuesD[0]),
                     numCells * numQuads * 9 * sizeof(double),
                     cudaMemcpyDeviceToHost);
      }//k point loop

      if (totalNonTrivialPseudoWfcs > 0)
        CUDACHECK(cudaFreeHost(
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp));

    }

  } // namespace force
} // namespace dftfe
