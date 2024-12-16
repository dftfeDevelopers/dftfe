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
#include "constants.h"
#include "dftUtils.h"
#include "forceWfcContractions.h"
#include "vectorUtilities.h"
#include <MemoryStorage.h>
#include <MemoryTransfer.h>
#if defined(DFTFE_WITH_DEVICE)
#  include <DeviceDataTypeOverloads.h>
#  include <DeviceAPICalls.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <forceWfcContractionsDeviceKernels.h>
#endif


namespace dftfe
{
  namespace force
  {
    namespace
    {
      template <dftfe::utils::MemorySpace memorySpace>
      void
      interpolatePsiGradPsiNlpQuads(
        std::shared_ptr<dftfe::basis::FEBasisOperations<dataTypes::number,
                                                        double,
                                                        memorySpace>>
          &                basisOperationsPtr,
        const unsigned int nlpspQuadratureId,
        const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
          &BLASWrapperPtr,
        dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &Xb,
        const unsigned int                                                 BVec,
        const unsigned int numCells,
        const unsigned int cellsBlockSize,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &psiQuadsNLP,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &gradPsiQuadsNLP)
      {
        const int blockSize    = cellsBlockSize;
        const int numberBlocks = numCells / blockSize;
        const int remBlockSize = numCells - numberBlocks * blockSize;


        basisOperationsPtr->reinit(BVec, cellsBlockSize, nlpspQuadratureId);


        for (int iblock = 0; iblock < (numberBlocks + 1); iblock++)
          {
            const int currentBlockSize =
              (iblock == numberBlocks) ? remBlockSize : blockSize;
            const int startingId = iblock * blockSize;

            if (currentBlockSize > 0)
              {
                basisOperationsPtr->interpolateKernel(
                  Xb,
                  psiQuadsNLP.data() +
                    startingId * basisOperationsPtr->nQuadsPerCell() * BVec,
                  gradPsiQuadsNLP.data() +
                    startingId * 3 * basisOperationsPtr->nQuadsPerCell() * BVec,
                  std::pair<unsigned int, unsigned int>(startingId,
                                                        startingId +
                                                          currentBlockSize));
              }
          }
      }

      template <dftfe::utils::MemorySpace memorySpace>
      void
      computeELocWfcEshelbyTensor(
        std::shared_ptr<dftfe::basis::FEBasisOperations<dataTypes::number,
                                                        double,
                                                        memorySpace>>
          &                basisOperationsPtr,
        const unsigned int densityQuadratureId,
        const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
          &BLASWrapperPtr,
        dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace> &Xb,
        const unsigned int                                                 BVec,
        const unsigned int                                      numCells,
        const unsigned int                                      numQuads,
        const dftfe::utils::MemoryStorage<double, memorySpace> &eigenValues,
        const dftfe::utils::MemoryStorage<double, memorySpace>
          &          partialOccupancies,
        const double kcoordx,
        const double kcoordy,
        const double kcoordz,
        const dftfe::utils::MemoryStorage<double, memorySpace> &onesVec,
        const unsigned int                                      cellsBlockSize,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &psiQuadsFlat,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &gradPsiQuadsFlat,
        dftfe::utils::MemoryStorage<double, memorySpace>
          &eshelbyTensorContributions,
        dftfe::utils::MemoryStorage<double, memorySpace>
          &        eshelbyTensorQuadValues,
        const bool isFloatingChargeForces,
        const bool addEk)
      {
        const int blockSize    = cellsBlockSize;
        const int numberBlocks = numCells / blockSize;
        const int remBlockSize = numCells - numberBlocks * blockSize;


        basisOperationsPtr->reinit(BVec, cellsBlockSize, densityQuadratureId);


        for (int iblock = 0; iblock < (numberBlocks + 1); iblock++)
          {
            const int currentBlockSize =
              (iblock == numberBlocks) ? remBlockSize : blockSize;
            const int startingId = iblock * blockSize;

            if (currentBlockSize > 0)
              {
                if (!isFloatingChargeForces)
                  {
                    basisOperationsPtr->interpolateKernel(
                      Xb,
                      psiQuadsFlat.data(),
                      gradPsiQuadsFlat.data(),
                      std::pair<unsigned int, unsigned int>(
                        startingId, startingId + currentBlockSize));

                    if (memorySpace == dftfe::utils::MemorySpace::HOST)
                      {
                        std::vector<double> kcoord(3, 0);
                        kcoord[0]           = kcoordx;
                        kcoord[1]           = kcoordy;
                        kcoord[2]           = kcoordz;
                        const double absksq = kcoord[0] * kcoord[0] +
                                              kcoord[1] * kcoord[1] +
                                              kcoord[2] * kcoord[2];
                        for (unsigned int j = 0; j < currentBlockSize; j++)
                          for (unsigned int iquad = 0; iquad < numQuads;
                               iquad++)
                            for (unsigned int iwfc = 0; iwfc < BVec; iwfc++)
                              {
                                const dataTypes::number psiQuad =
                                  psiQuadsFlat.data()[j * numQuads * BVec +
                                                      iquad * BVec + iwfc];
                                const double partOcc = partialOccupancies[iwfc];
                                const double eigenValue = eigenValues[iwfc];

                                std::vector<dataTypes::number> gradPsiQuad(3);
                                gradPsiQuad[0] =
                                  gradPsiQuadsFlat
                                    .data()[j * 3 * numQuads * BVec +
                                            iquad * BVec + iwfc];
                                gradPsiQuad[1] =
                                  gradPsiQuadsFlat
                                    .data()[j * 3 * numQuads * BVec +
                                            numQuads * BVec + iquad * BVec +
                                            iwfc];

                                gradPsiQuad[2] =
                                  gradPsiQuadsFlat
                                    .data()[j * 3 * numQuads * BVec +
                                            2 * numQuads * BVec + iquad * BVec +
                                            iwfc];

                                const double identityFactor =
                                  partOcc *
                                    dftfe::utils::realPart(
                                      (dftfe::utils::complexConj(
                                         gradPsiQuad[0]) *
                                         gradPsiQuad[0] +
                                       dftfe::utils::complexConj(
                                         gradPsiQuad[1]) *
                                         gradPsiQuad[1] +
                                       dftfe::utils::complexConj(
                                         gradPsiQuad[2]) *
                                         gradPsiQuad[2] +
                                       dataTypes::number(absksq -
                                                         2.0 * eigenValue) *
                                         dftfe::utils::complexConj(psiQuad) *
                                         psiQuad)) +
                                  2.0 * partOcc *
                                    dftfe::utils::imagPart(
                                      dftfe::utils::complexConj(psiQuad) *
                                      (kcoord[0] * gradPsiQuad[0] +
                                       kcoord[1] * gradPsiQuad[1] +
                                       kcoord[2] * gradPsiQuad[2]));
                                for (unsigned int idim = 0; idim < 3; idim++)
                                  for (unsigned int jdim = 0; jdim < 3; jdim++)
                                    {
                                      eshelbyTensorContributions
                                        [j * numQuads * 9 * BVec +
                                         iquad * 9 * BVec + idim * 3 * BVec +
                                         jdim * BVec + iwfc] =
                                          -partOcc *
                                            dftfe::utils::realPart(
                                              dftfe::utils::complexConj(
                                                gradPsiQuad[idim]) *
                                                gradPsiQuad[jdim] +
                                              gradPsiQuad[idim] *
                                                dftfe::utils::complexConj(
                                                  gradPsiQuad[jdim])) -
                                          2.0 * partOcc *
                                            dftfe::utils::imagPart(
                                              dftfe::utils::complexConj(
                                                psiQuad) *
                                              (gradPsiQuad[idim] *
                                               kcoord[jdim]));

                                      if (idim == jdim)
                                        eshelbyTensorContributions
                                          [j * numQuads * 9 * BVec +
                                           iquad * 9 * BVec + idim * 3 * BVec +
                                           jdim * BVec + iwfc] +=
                                          identityFactor;
                                    }
#ifdef USE_COMPLEX
                                if (addEk)
                                  {
                                    for (unsigned int idim = 0; idim < 3;
                                         idim++)
                                      for (unsigned int jdim = 0; jdim < 3;
                                           jdim++)
                                        {
                                          eshelbyTensorContributions
                                            [j * numQuads * 9 * BVec +
                                             iquad * 9 * BVec +
                                             idim * 3 * BVec + jdim * BVec +
                                             iwfc] +=
                                            -2.0 * partOcc *
                                              dftfe::utils::imagPart(
                                                dftfe::utils::complexConj(
                                                  psiQuad) *
                                                (kcoord[idim] *
                                                 gradPsiQuad[jdim])) -
                                            2.0 * partOcc *
                                              dftfe::utils::realPart(
                                                kcoord[idim] * kcoord[jdim] *
                                                dftfe::utils::complexConj(
                                                  psiQuad) *
                                                psiQuad);
                                        }
                                  }
#endif
                              }
                      }
#if defined(DFTFE_WITH_DEVICE)
                    else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
                      forceDeviceKernels::
                        computeELocWfcEshelbyTensorContributions(
                          BVec,
                          currentBlockSize,
                          numQuads,
                          psiQuadsFlat.data(),
                          gradPsiQuadsFlat.data(),
                          eigenValues.data(),
                          partialOccupancies.data(),
#  ifdef USE_COMPLEX
                          kcoordx,
                          kcoordy,
                          kcoordz,
#  endif
                          eshelbyTensorContributions.data()
#  ifdef USE_COMPLEX
                            ,
                          addEk
#  endif
                        );
#endif

                    const double scalarCoeffAlphaEshelby = 1.0;
                    const double scalarCoeffBetaEshelby  = 1.0;

                    BLASWrapperPtr->xgemm('N',
                                          'N',
                                          1,
                                          currentBlockSize * numQuads * 9,
                                          BVec,
                                          &scalarCoeffAlphaEshelby,
                                          onesVec.data(),
                                          1,
                                          eshelbyTensorContributions.data(),
                                          BVec,
                                          &scalarCoeffBetaEshelby,
                                          eshelbyTensorQuadValues.data() +
                                            startingId * numQuads * 9,
                                          1);
                  }
              }
          }
      }

      template <dftfe::utils::MemorySpace memorySpace>
      void
      nlpPsiContraction(
        const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
          &BLASWrapperPtr,
        const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &psiQuadsNLP,
        const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &gradPsiQuadsNLP,
        const dftfe::utils::MemoryStorage<double, memorySpace>
          &partialOccupancies,
        const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &                      onesVecNLP,
        const dataTypes::number *projectorKetTimesVectorParFlattened,
        const dftfe::utils::MemoryStorage<unsigned int, memorySpace>
          &nonTrivialIdToElemIdMap,
        const dftfe::utils::MemoryStorage<unsigned int, memorySpace>
          &                projecterKetTimesFlattenedVectorLocalIds,
        const unsigned int numCells,
        const unsigned int numQuadsNLP,
        const unsigned int numPsi,
        const unsigned int totalNonTrivialPseudoWfcs,
        const unsigned int innerBlockSizeEnlp,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &nlpContractionContribution,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> &
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlock,
        dataTypes::number *
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
#ifdef USE_COMPLEX
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> &
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock,
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
                if (memorySpace == dftfe::utils::MemorySpace::HOST)
                  {
                    for (unsigned int ipseudowfc = 0;
                         ipseudowfc < currentBlockSizeNlp;
                         ipseudowfc++)
                      for (unsigned int iquad = 0; iquad < (numQuadsNLP * 3);
                           iquad++)
                        for (unsigned int iwfc = 0; iwfc < numPsi; iwfc++)
                          {
                            nlpContractionContribution[ipseudowfc *
                                                         numQuadsNLP * 3 *
                                                         numPsi +
                                                       iquad * numPsi + iwfc] =
                              partialOccupancies.data()[iwfc] *
                              dftfe::utils::complexConj(
                                gradPsiQuadsNLP
                                  .data()[nonTrivialIdToElemIdMap.data()
                                              [startingIdNlp + ipseudowfc] *
                                            3 * numQuadsNLP * numPsi +
                                          iquad * numPsi + iwfc]) *
                              projectorKetTimesVectorParFlattened
                                [projecterKetTimesFlattenedVectorLocalIds
                                     .data()[startingIdNlp + ipseudowfc] *
                                   numPsi +
                                 iwfc];
                          }
                  }
#if defined(DFTFE_WITH_DEVICE)
                else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
                  forceDeviceKernels::nlpContractionContributionPsiIndex(
                    numPsi,
                    currentBlockSizeNlp,
                    numQuadsNLP * 3,
                    startingIdNlp,
                    projectorKetTimesVectorParFlattened,
                    gradPsiQuadsNLP.data(),
                    partialOccupancies.data(),
                    nonTrivialIdToElemIdMap.data(),
                    projecterKetTimesFlattenedVectorLocalIds.data(),
                    nlpContractionContribution.data());
#endif



                BLASWrapperPtr->xgemm(
                  'N',
                  'N',
                  1,
                  currentBlockSizeNlp * 3 * numQuadsNLP,
                  numPsi,
                  &scalarCoeffAlphaNlp,
                  onesVecNLP.data(),
                  1,
                  nlpContractionContribution.data(),
                  numPsi,
                  &scalarCoeffBetaNlp,
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlock
                    .data(),
                  1);

                dftfe::utils::MemoryTransfer<dftfe::utils::MemorySpace::HOST,
                                             memorySpace>::
                  copy(
                    currentBlockSizeNlp * 3 * numQuadsNLP,
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp,
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlock
                      .data());


                for (unsigned int i = 0;
                     i < currentBlockSizeNlp * 3 * numQuadsNLP;
                     i++)
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH
                    [startingIdNlp * 3 * numQuadsNLP + i] +=
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp
                      [i];
#ifdef USE_COMPLEX
                if (memorySpace == dftfe::utils::MemorySpace::HOST)
                  {
                    for (unsigned int ipseudowfc = 0;
                         ipseudowfc < currentBlockSizeNlp;
                         ipseudowfc++)
                      for (unsigned int iquad = 0; iquad < numQuadsNLP; iquad++)
                        for (unsigned int iwfc = 0; iwfc < numPsi; iwfc++)
                          nlpContractionContribution[ipseudowfc * numQuadsNLP *
                                                       numPsi +
                                                     iquad * numPsi + iwfc] =
                            partialOccupancies.data()[iwfc] *
                            dftfe::utils::complexConj(
                              psiQuadsNLP.data()[nonTrivialIdToElemIdMap
                                                     .data()[startingIdNlp +
                                                             ipseudowfc] *
                                                   numQuadsNLP * numPsi +
                                                 iquad * numPsi + iwfc]) *
                            projectorKetTimesVectorParFlattened
                              [projecterKetTimesFlattenedVectorLocalIds
                                   .data()[startingIdNlp + ipseudowfc] *
                                 numPsi +
                               iwfc];
                  }
#  if defined(DFTFE_WITH_DEVICE)
                else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
                  forceDeviceKernels::nlpContractionContributionPsiIndex(
                    numPsi,
                    currentBlockSizeNlp,
                    numQuadsNLP,
                    startingIdNlp,
                    projectorKetTimesVectorParFlattened,
                    psiQuadsNLP.data(),
                    partialOccupancies.data(),
                    nonTrivialIdToElemIdMap.data(),
                    projecterKetTimesFlattenedVectorLocalIds.data(),
                    nlpContractionContribution.data());
#  endif


                BLASWrapperPtr->xgemm(
                  'N',
                  'N',
                  1,
                  currentBlockSizeNlp * numQuadsNLP,
                  numPsi,
                  &scalarCoeffAlphaNlp,
                  onesVecNLP.data(),
                  1,
                  nlpContractionContribution.data(),
                  numPsi,
                  &scalarCoeffBetaNlp,
                  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock
                    .data(),
                  1);


                dftfe::utils::MemoryTransfer<dftfe::utils::MemorySpace::HOST,
                                             memorySpace>::
                  copy(
                    currentBlockSizeNlp * numQuadsNLP,
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp,
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock
                      .data());


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


      template <dftfe::utils::MemorySpace memorySpace>
      void
      forceKernelsAll(
        std::shared_ptr<dftfe::basis::FEBasisOperations<dataTypes::number,
                                                        double,
                                                        memorySpace>>
          &                basisOperationsPtr,
        const unsigned int densityQuadratureId,
        const unsigned int nlpspQuadratureId,
        const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
          &BLASWrapperPtr,
        std::shared_ptr<dftfe::oncvClass<dataTypes::number, memorySpace>>
          oncvClassPtr,
        std::shared_ptr<hubbard<dataTypes::number, memorySpace>>
                           hubbardClassPtr,
        const unsigned int kPointIndex,
        const unsigned int spinIndex,
        dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
          &flattenedArrayBlock,
        dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
          &projectorKetTimesVector,
        dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
          &                      projectorKetTimesVectorHubbard,
        const dataTypes::number *X,
        const dftfe::utils::MemoryStorage<double, memorySpace> &eigenValues,
        const dftfe::utils::MemoryStorage<double, memorySpace>
          &          partialOccupancies,
        const double kcoordx,
        const double kcoordy,
        const double kcoordz,
        const dftfe::utils::MemoryStorage<double, memorySpace> &onesVec,
        const dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &onesVecNLP,
        const dftfe::utils::MemoryStorage<unsigned int, memorySpace>
          &nonTrivialIdToElemIdMap,
        const dftfe::utils::MemoryStorage<unsigned int, memorySpace>
          &projecterKetTimesFlattenedVectorLocalIds,
        const dftfe::utils::MemoryStorage<unsigned int, memorySpace>
          &nonTrivialIdToElemIdMapHubbard,
        const dftfe::utils::MemoryStorage<unsigned int, memorySpace>
          &                projecterKetTimesFlattenedVectorLocalIdsHubbard,
        const unsigned int startingVecId,
        const unsigned int N,
        const unsigned int numPsi,
        const unsigned int numCells,
        const unsigned int numQuads,
        const unsigned int numQuadsNLP,
        const unsigned int totalNonTrivialPseudoWfcs,
        const unsigned int totalNonTrivialHubbardProjectors,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &psiQuadsFlat,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &gradPsiQuadsFlat,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &psiQuadsNLP,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &gradPsiQuadsNLP,
        dftfe::utils::MemoryStorage<double, memorySpace>
          &eshelbyTensorContributions,
        dftfe::utils::MemoryStorage<double, memorySpace>
          &eshelbyTensorQuadValues,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &nlpContractionContribution,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
          &contractionContributionHubbard,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> &
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlock,
        dataTypes::number *
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> &
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlockHubbard,
        dataTypes::number *
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHHubbard,
#ifdef USE_COMPLEX
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> &
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock,
        dataTypes::number
          *projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
        dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> &
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlockHubbard,
        dataTypes::number *
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHHubbard,
#endif
        dataTypes::number *
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp,
        dataTypes::number *
                           projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTempHubbard,
        const unsigned int cellsBlockSize,
        const unsigned int innerBlockSizeEnlp,
        const bool         isPsp,
        const unsigned int innerBlockSizeHubbard,
        const bool         useHubbard,
        const bool         isFloatingChargeForces,
        const bool         addEk)
      {
        // int this_process;
        // MPI_Comm_rank(d_mpiCommParent, &this_process);



        if (memorySpace == dftfe::utils::MemorySpace::HOST)
          for (unsigned int iNode = 0; iNode < basisOperationsPtr->nOwnedDofs();
               ++iNode)
            std::memcpy(flattenedArrayBlock.data() + iNode * numPsi,
                        X + iNode * N + startingVecId,
                        numPsi * sizeof(dataTypes::number));
#if defined(DFTFE_WITH_DEVICE)
        else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          BLASWrapperPtr->stridedCopyToBlockConstantStride(
            numPsi,
            N,
            basisOperationsPtr->nOwnedDofs(),
            startingVecId,
            X,
            flattenedArrayBlock.data());
#endif


        flattenedArrayBlock.updateGhostValues();
        basisOperationsPtr->distribute(flattenedArrayBlock);


        // dftfe::utils::deviceSynchronize();
        // MPI_Barrier(d_mpiCommParent);
        // double kernel1_time = MPI_Wtime();

        computeELocWfcEshelbyTensor(basisOperationsPtr,
                                    densityQuadratureId,
                                    BLASWrapperPtr,
                                    flattenedArrayBlock,
                                    numPsi,
                                    numCells,
                                    numQuads,
                                    eigenValues,
                                    partialOccupancies,
                                    kcoordx,
                                    kcoordy,
                                    kcoordz,
                                    onesVec,
                                    cellsBlockSize,
                                    psiQuadsFlat,
                                    gradPsiQuadsFlat,
                                    eshelbyTensorContributions,
                                    eshelbyTensorQuadValues,
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

            // oncvClassPtr->getNonLocalOperator()->initialiseOperatorActionOnX(
            //   kPointIndex);

            oncvClassPtr->getNonLocalOperator()->applyVCconjtransOnX(
              flattenedArrayBlock,
              kPointIndex,
              CouplingStructure::diagonal,
              oncvClassPtr->getCouplingMatrix(),
              projectorKetTimesVector);
            /*

                        dftfe::utils::MemoryStorage<dataTypes::number,
                                                    dftfe::utils::MemorySpace::HOST>
                          projectorKetTimesVectorHostData;

                        projectorKetTimesVectorHostData.resize(projectorKetTimesVector.getData().size());

                        projectorKetTimesVectorHostData.copyFrom(projectorKetTimesVector.getData());

                        double projectorKetTimesVectorHostDataNorm = 0.0;

                        for( unsigned int i = 0; i <
               projectorKetTimesVectorHostData.size(); i++)
                          {
                            projectorKetTimesVectorHostDataNorm +=
               projectorKetTimesVectorHostData.data()[i]*
                                                                   projectorKetTimesVectorHostData.data()[i];
                          }

                        std::cout<<" projectorKetTimesVectorHostDataNorm =
               "<<projectorKetTimesVectorHostDataNorm<<"\n";
            */
          }

        if (useHubbard)
          {
            flattenedArrayBlock.updateGhostValues();
            basisOperationsPtr->distribute(flattenedArrayBlock);

            hubbardClassPtr->getNonLocalOperator()->applyVCconjtransOnX(
              flattenedArrayBlock,
              kPointIndex,
              CouplingStructure::dense,
              hubbardClassPtr->getCouplingMatrix(spinIndex),
              projectorKetTimesVectorHubbard);

            /*
                  dftfe::utils::MemoryStorage<dataTypes::number,
                                              dftfe::utils::MemorySpace::HOST>
                    projectorKetTimesVectorHubbardHostData;

                  projectorKetTimesVectorHubbardHostData.resize(projectorKetTimesVectorHubbard.getData().size());

                  projectorKetTimesVectorHubbardHostData.copyFrom(projectorKetTimesVectorHubbard.getData());

                  double projectorKetTimesVectorHubbardHostDataNorm = 0.0;

                  for( unsigned int i = 0; i <
          projectorKetTimesVectorHubbardHostData.size(); i++)
                    {
          std::cout<<" i = "<<i<<" HubbardHostData =
          "<<projectorKetTimesVectorHubbardHostData.data()[i]<<"\n";
                      projectorKetTimesVectorHubbardHostDataNorm +=
          projectorKetTimesVectorHubbardHostData.data()[i]*
                                                                    projectorKetTimesVectorHubbardHostData.data()[i];
                    }

                  std::cout<<" projectorKetTimesVectorHubbardHostDataNorm =
          "<<projectorKetTimesVectorHubbardHostDataNorm<<"\n";
      */
          }

        // dftfe::utils::deviceSynchronize();
        // MPI_Barrier(d_mpiCommParent);
        // kernel2_time = MPI_Wtime() - kernel2_time;


        // dftfe::utils::deviceSynchronize();
        // MPI_Barrier(d_mpiCommParent);
        // double kernel3_time = MPI_Wtime();

        if (isPsp || useHubbard)
          {
            interpolatePsiGradPsiNlpQuads(basisOperationsPtr,
                                          nlpspQuadratureId,
                                          BLASWrapperPtr,
                                          flattenedArrayBlock,
                                          numPsi,
                                          numCells,
                                          cellsBlockSize,
                                          psiQuadsNLP,
                                          gradPsiQuadsNLP);
          }



        if (totalNonTrivialPseudoWfcs > 0)
          {
            nlpPsiContraction(
              BLASWrapperPtr,
              psiQuadsNLP,
              gradPsiQuadsNLP,
              partialOccupancies,
              onesVecNLP,
              projectorKetTimesVector.data(),
              nonTrivialIdToElemIdMap,
              projecterKetTimesFlattenedVectorLocalIds,
              numCells,
              numQuadsNLP,
              numPsi,
              totalNonTrivialPseudoWfcs,
              innerBlockSizeEnlp,
              nlpContractionContribution,
              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlock,
              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
#ifdef USE_COMPLEX
              projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock,
              projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
#endif
              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp);
          }

        if (totalNonTrivialHubbardProjectors > 0)
          {
            nlpPsiContraction(
              BLASWrapperPtr,
              psiQuadsNLP,
              gradPsiQuadsNLP,
              partialOccupancies,
              onesVecNLP,
              projectorKetTimesVectorHubbard.data(),
              nonTrivialIdToElemIdMapHubbard,
              projecterKetTimesFlattenedVectorLocalIdsHubbard,
              numCells,
              numQuadsNLP,
              numPsi,
              totalNonTrivialHubbardProjectors,
              innerBlockSizeHubbard,
              contractionContributionHubbard,
              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlockHubbard,
              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHHubbard,
#ifdef USE_COMPLEX
              projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlockHubbard,
              projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHHubbard,
#endif
              projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTempHubbard);
          }

        // dftfe::utils::deviceSynchronize();
        // MPI_Barrier(d_mpiCommParent);
        // kernel3_time = MPI_Wtime() - kernel3_time;

        // if (this_process==0 && dftParameters::verbosity>=5)
        //	 std::cout<<"Time for nlpPsiContractionD inside blocked loop:
        //"<<kernel3_time<<std::endl;
      }

    } // namespace

    template <dftfe::utils::MemorySpace memorySpace>
    void
    wfcContractionsForceKernelsAllH(
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number, double, memorySpace>>
        &                basisOperationsPtr,
      const unsigned int densityQuadratureId,
      const unsigned int nlpspQuadratureId,
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        &BLASWrapperPtr,
      std::shared_ptr<dftfe::oncvClass<dataTypes::number, memorySpace>>
                                                               oncvClassPtr,
      std::shared_ptr<hubbard<dataTypes::number, memorySpace>> hubbardClassPtr,
      const bool                                               useHubbard,
      const dataTypes::number *                                X,
      const unsigned int                      spinPolarizedFlag,
      const unsigned int                      spinIndex,
      const std::vector<std::vector<double>> &eigenValuesH,
      const std::vector<std::vector<double>> &partialOccupanciesH,
      const std::vector<double> &             kPointCoordinates,
      const unsigned int                      MLoc,
      const unsigned int                      N,
      const unsigned int                      numCells,
      const unsigned int                      numQuads,
      const unsigned int                      numQuadsNLP,
      double *                                eshelbyTensorQuadValuesH,
      dataTypes::number *
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
      dataTypes::number *
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHHubbard,
#ifdef USE_COMPLEX
      dataTypes::number
        *projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
      dataTypes::number *
        projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHHubbard,
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


      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
        *flattenedArrayBlockPtr;

      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
        projectorKetTimesVector;

      dftfe::linearAlgebra::MultiVector<dataTypes::number, memorySpace>
        projectorKetTimesVectorHubbard;

      // dftfe::utils::deviceSynchronize();
      // MPI_Barrier(mpiCommParent);
      // device_time = MPI_Wtime() - device_time;

      // if (this_process == 0 && dftParams.verbosity >= 2)
      //  std::cout
      //    << "Time for creating device parallel vectors for force computation:
      //    "
      //    << device_time << std::endl;

      // device_time = MPI_Wtime();

      dftfe::utils::MemoryStorage<double, memorySpace> eigenValues(blockSize,
                                                                   0.0);
      dftfe::utils::MemoryStorage<double, memorySpace> partialOccupancies(
        blockSize, 0.0);
      dftfe::utils::MemoryStorage<double, memorySpace>
        elocWfcEshelbyTensorQuadValues(numCells * numQuads * 9, 0.0);

      dftfe::utils::MemoryStorage<double, memorySpace> onesVec(blockSize, 1.0);
      dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> onesVecNLP(
        blockSize, dataTypes::number(1.0));

      const unsigned int cellsBlockSize = std::min((unsigned int)10, numCells);

      dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> psiQuadsFlat(
        cellsBlockSize * numQuads * blockSize, dataTypes::number(0.0));
      dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
                                                                  gradPsiQuadsFlat(cellsBlockSize * numQuads * blockSize * 3,
                         dataTypes::number(0.0));
      dftfe::utils::MemoryStorage<dataTypes::number, memorySpace> psiQuadsNLP(
        numCells * numQuadsNLP * blockSize, dataTypes::number(0.0));

      dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
        gradPsiQuadsNLPFlat(numCells * numQuadsNLP * 3 * blockSize,
                            dataTypes::number(0.0));

      dftfe::utils::MemoryStorage<double, memorySpace>
        eshelbyTensorContributions(cellsBlockSize * numQuads * blockSize * 9,
                                   0.0);

      const unsigned int totalNonTrivialPseudoWfcs =
        isPsp ? oncvClassPtr->getNonLocalOperator()
                  ->getTotalNonTrivialSphericalFnsOverAllCells() :
                0;

      const unsigned int innerBlockSizeEnlp =
        std::min((unsigned int)10, totalNonTrivialPseudoWfcs);
      dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
        nlpContractionContribution(innerBlockSizeEnlp * numQuadsNLP * 3 *
                                     blockSize,
                                   dataTypes::number(0.0));
      dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
        projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock;
      dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlock;
      dftfe::utils::MemoryStorage<unsigned int, memorySpace>
        projecterKetTimesFlattenedVectorLocalIds;
      dftfe::utils::MemoryStorage<unsigned int, memorySpace>
        nonTrivialIdToElemIdMap;
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST>
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp;
      if (totalNonTrivialPseudoWfcs > 0)
        {
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlock
            .resize(innerBlockSizeEnlp * numQuadsNLP * 3,
                    dataTypes::number(0.0));
#ifdef USE_COMPLEX
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock
            .resize(innerBlockSizeEnlp * numQuadsNLP, dataTypes::number(0.0));
#endif
          projecterKetTimesFlattenedVectorLocalIds.resize(
            totalNonTrivialPseudoWfcs, 0.0);
          nonTrivialIdToElemIdMap.resize(totalNonTrivialPseudoWfcs, 0);



          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp
            .resize(innerBlockSizeEnlp * numQuadsNLP * 3, 0);

          dftfe::utils::
            MemoryTransfer<memorySpace, dftfe::utils::MemorySpace::HOST>::copy(
              totalNonTrivialPseudoWfcs,
              nonTrivialIdToElemIdMap.data(),
              &(oncvClassPtr->getNonLocalOperator()
                  ->getNonTrivialAllCellsSphericalFnAlphaToElemIdMap()[0]));

          /*
          dftfe::utils::deviceMemcpyH2D(nonTrivialIdToElemIdMapD.data(),
                                        nonTrivialIdToElemIdMapH,
                                        totalNonTrivialPseudoWfcs *
                                          sizeof(unsigned int));
          */

          dftfe::utils::
            MemoryTransfer<memorySpace, dftfe::utils::MemorySpace::HOST>::copy(
              totalNonTrivialPseudoWfcs,
              projecterKetTimesFlattenedVectorLocalIds.data(),
              &(oncvClassPtr->getNonLocalOperator()
                  ->getSphericalFnTimesVectorFlattenedVectorLocalIds()[0]));

          /*
          dftfe::utils::deviceMemcpyH2D(
            projecterKetTimesFlattenedVectorLocalIdsD.data(),
            projecterKetTimesFlattenedVectorLocalIdsH,
            totalNonTrivialPseudoWfcs * sizeof(unsigned int));
          */
        }


      const unsigned int totalNonTrivialHubbardProjectors =
        useHubbard ? hubbardClassPtr->getNonLocalOperator()
                       ->getTotalNonTrivialSphericalFnsOverAllCells() :
                     0;


      const unsigned int innerBlockSizeHubbard =
        std::min((unsigned int)10, totalNonTrivialHubbardProjectors);
      dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
        contractionContributionHubbard(innerBlockSizeHubbard * numQuadsNLP * 3 *
                                         blockSize,
                                       dataTypes::number(0.0));
      dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
        projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlockHubbard;
      dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlockHubbard;
      dftfe::utils::MemoryStorage<unsigned int, memorySpace>
        projecterKetTimesFlattenedVectorLocalIdsHubbard;
      dftfe::utils::MemoryStorage<unsigned int, memorySpace>
        nonTrivialIdToElemIdMapHubbard;
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::HOST>
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTempHubbard;

      if (totalNonTrivialHubbardProjectors > 0)
        {
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlockHubbard
            .resize(innerBlockSizeHubbard * numQuadsNLP * 3,
                    dataTypes::number(0.0));
#ifdef USE_COMPLEX
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlockHubbard
            .resize(innerBlockSizeHubbard * numQuadsNLP,
                    dataTypes::number(0.0));
#endif
          projecterKetTimesFlattenedVectorLocalIdsHubbard.resize(
            totalNonTrivialHubbardProjectors, 0.0);
          nonTrivialIdToElemIdMapHubbard.resize(
            totalNonTrivialHubbardProjectors, 0);



          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTempHubbard
            .resize(innerBlockSizeHubbard * numQuadsNLP * 3, 0);

          dftfe::utils::
            MemoryTransfer<memorySpace, dftfe::utils::MemorySpace::HOST>::copy(
              totalNonTrivialHubbardProjectors,
              nonTrivialIdToElemIdMapHubbard.data(),
              &(hubbardClassPtr->getNonLocalOperator()
                  ->getNonTrivialAllCellsSphericalFnAlphaToElemIdMap()[0]));


          dftfe::utils::
            MemoryTransfer<memorySpace, dftfe::utils::MemorySpace::HOST>::copy(
              totalNonTrivialHubbardProjectors,
              projecterKetTimesFlattenedVectorLocalIdsHubbard.data(),
              &(hubbardClassPtr->getNonLocalOperator()
                  ->getSphericalFnTimesVectorFlattenedVectorLocalIds()[0]));
        }

      const unsigned numKPoints = kPointCoordinates.size() / 3;
      for (unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
        {
          elocWfcEshelbyTensorQuadValues.setValue(0);
          // spin index update is not required

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

          if (totalNonTrivialHubbardProjectors > 0)
            {
              std::fill(
                projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHHubbard +
                  kPoint * totalNonTrivialHubbardProjectors * numQuadsNLP * 3,
                projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHHubbard +
                  (kPoint + 1) * totalNonTrivialHubbardProjectors *
                    numQuadsNLP * 3,
                dataTypes::number(0.0));

#ifdef USE_COMPLEX
              std::fill(
                projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHHubbard +
                  kPoint * totalNonTrivialHubbardProjectors * numQuadsNLP,
                projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHHubbard +
                  (kPoint + 1) * totalNonTrivialHubbardProjectors * numQuadsNLP,
                dataTypes::number(0.0));
#endif
            }

          for (unsigned int ivec = 0; ivec < N; ivec += blockSize)
            {
              const unsigned int currentBlockSize =
                std::min(blockSize, N - ivec);

              flattenedArrayBlockPtr =
                &(basisOperationsPtr->getMultiVector(currentBlockSize, 0));

              if (isPsp)
                oncvClassPtr->getNonLocalOperator()
                  ->initialiseFlattenedDataStructure(currentBlockSize,
                                                     projectorKetTimesVector);

              if (useHubbard)
                hubbardClassPtr->getNonLocalOperator()
                  ->initialiseFlattenedDataStructure(
                    currentBlockSize, projectorKetTimesVectorHubbard);


              if ((ivec + currentBlockSize) <=
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                  (ivec + currentBlockSize) >
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  std::vector<double> blockedEigenValues(currentBlockSize, 0.0);
                  std::vector<double> blockedPartialOccupancies(
                    currentBlockSize, 0.0);
                  for (unsigned int iWave = 0; iWave < currentBlockSize;
                       ++iWave)
                    {
                      blockedEigenValues[iWave] =
                        eigenValuesH[kPoint][spinIndex * N + ivec + iWave];
                      blockedPartialOccupancies[iWave] =
                        partialOccupanciesH[kPoint]
                                           [spinIndex * N + ivec + iWave];
                    }


                  dftfe::utils::MemoryTransfer<
                    memorySpace,
                    dftfe::utils::MemorySpace::HOST>::
                    copy(currentBlockSize,
                         eigenValues.data(),
                         &blockedEigenValues[0]);

                  dftfe::utils::MemoryTransfer<
                    memorySpace,
                    dftfe::utils::MemorySpace::HOST>::
                    copy(currentBlockSize,
                         partialOccupancies.data(),
                         &blockedPartialOccupancies[0]);

                  /*
                  dftfe::utils::deviceMemcpyH2D(eigenValuesD.data(),
                                                &blockedEigenValues[0],
                                                blockSize * sizeof(double));

                  dftfe::utils::deviceMemcpyH2D(partialOccupanciesD.data(),
                                                &blockedPartialOccupancies[0],
                                                blockSize * sizeof(double));
                  */

                  // dftfe::utils::deviceSynchronize();
                  // MPI_Barrier(d_mpiCommParent);
                  // double kernel_time = MPI_Wtime();

                  forceKernelsAll(
                    basisOperationsPtr,
                    densityQuadratureId,
                    nlpspQuadratureId,
                    BLASWrapperPtr,
                    oncvClassPtr,
                    hubbardClassPtr,
                    kPoint,
                    spinIndex,
                    *flattenedArrayBlockPtr,
                    projectorKetTimesVector,
                    projectorKetTimesVectorHubbard,
                    X +
                      ((1 + spinPolarizedFlag) * kPoint + spinIndex) * MLoc * N,
                    eigenValues,
                    partialOccupancies,
                    kcoordx,
                    kcoordy,
                    kcoordz,
                    onesVec,
                    onesVecNLP,
                    nonTrivialIdToElemIdMap,
                    projecterKetTimesFlattenedVectorLocalIds,
                    nonTrivialIdToElemIdMapHubbard,
                    projecterKetTimesFlattenedVectorLocalIdsHubbard,
                    ivec,
                    N,
                    currentBlockSize,
                    numCells,
                    numQuads,
                    numQuadsNLP,
                    totalNonTrivialPseudoWfcs,
                    totalNonTrivialHubbardProjectors,
                    psiQuadsFlat,
                    gradPsiQuadsFlat,
                    psiQuadsNLP,
                    gradPsiQuadsNLPFlat,
                    eshelbyTensorContributions,
                    elocWfcEshelbyTensorQuadValues,
                    nlpContractionContribution,
                    contractionContributionHubbard,
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlock,
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH +
                      kPoint * totalNonTrivialPseudoWfcs * numQuadsNLP * 3,
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlockHubbard,
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHHubbard +
                      kPoint * totalNonTrivialHubbardProjectors * numQuadsNLP *
                        3,
#ifdef USE_COMPLEX
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock,
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH +
                      kPoint * totalNonTrivialPseudoWfcs * numQuadsNLP,
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlockHubbard,
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHHubbard +
                      kPoint * totalNonTrivialHubbardProjectors * numQuadsNLP,
#endif
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp
                      .data(),
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTempHubbard
                      .data(),
                    cellsBlockSize,
                    innerBlockSizeEnlp,
                    isPsp,
                    innerBlockSizeHubbard,
                    useHubbard,
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

          dftfe::utils::
            MemoryTransfer<dftfe::utils::MemorySpace::HOST, memorySpace>::copy(
              numCells * numQuads * 9,
              eshelbyTensorQuadValuesH + kPoint * numCells * numQuads * 9,
              elocWfcEshelbyTensorQuadValues.data());
          /*
          dftfe::utils::deviceMemcpyD2H(eshelbyTensorQuadValuesH +
                                          kPoint * numCells * numQuads * 9,
                                        elocWfcEshelbyTensorQuadValuesD.data(),
                                        numCells * numQuads * 9 *
                                          sizeof(double));
          */
        } // k point loop

      // dftfe::utils::deviceSynchronize();
      // MPI_Barrier(mpiCommParent);
      // device_time = MPI_Wtime() - device_time;

      // if (this_process == 0 && dftParams.verbosity >= 1)
      //  std::cout << "Time taken for all device kernels force computation: "
      //            << device_time << std::endl;
    }


#if defined(DFTFE_WITH_DEVICE)
    template void
    wfcContractionsForceKernelsAllH(
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::DEVICE>>
        &                basisOperationsPtr,
      const unsigned int densityQuadratureId,
      const unsigned int nlpspQuadratureId,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
        &BLASWrapperPtr,
      std::shared_ptr<
        dftfe::oncvClass<dataTypes::number, dftfe::utils::MemorySpace::DEVICE>>
        oncvClassPtr,
      std::shared_ptr<
        hubbard<dataTypes::number, dftfe::utils::MemorySpace::DEVICE>>
                                              hubbardClassPtr,
      const bool                              useHubbard,
      const dataTypes::number *               X,
      const unsigned int                      spinPolarizedFlag,
      const unsigned int                      spinIndex,
      const std::vector<std::vector<double>> &eigenValuesH,
      const std::vector<std::vector<double>> &partialOccupanciesH,
      const std::vector<double> &             kPointCoordinates,
      const unsigned int                      MLoc,
      const unsigned int                      N,
      const unsigned int                      numCells,
      const unsigned int                      numQuads,
      const unsigned int                      numQuadsNLP,
      double *                                eshelbyTensorQuadValuesH,
      dataTypes::number *
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
      dataTypes::number *
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHHubbard,
#  ifdef USE_COMPLEX
      dataTypes::number
        *projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
      dataTypes::number *
        projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHHubbard,
#  endif
      const MPI_Comm &     mpiCommParent,
      const MPI_Comm &     interBandGroupComm,
      const bool           isPsp,
      const bool           isFloatingChargeForces,
      const bool           addEk,
      const dftParameters &dftParams);
#endif

    template void
    wfcContractionsForceKernelsAllH(
      std::shared_ptr<
        dftfe::basis::FEBasisOperations<dataTypes::number,
                                        double,
                                        dftfe::utils::MemorySpace::HOST>>
        &                basisOperationsPtr,
      const unsigned int densityQuadratureId,
      const unsigned int nlpspQuadratureId,
      const std::shared_ptr<
        dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
        &BLASWrapperPtr,
      std::shared_ptr<
        dftfe::oncvClass<dataTypes::number, dftfe::utils::MemorySpace::HOST>>
                                                                oncvClassPtr,
      std::shared_ptr<hubbard<dataTypes::number,
                              dftfe::utils::MemorySpace::HOST>> hubbardClassPtr,
      const bool                                                useHubbard,
      const dataTypes::number *                                 X,
      const unsigned int                      spinPolarizedFlag,
      const unsigned int                      spinIndex,
      const std::vector<std::vector<double>> &eigenValuesH,
      const std::vector<std::vector<double>> &partialOccupanciesH,
      const std::vector<double> &             kPointCoordinates,
      const unsigned int                      MLoc,
      const unsigned int                      N,
      const unsigned int                      numCells,
      const unsigned int                      numQuads,
      const unsigned int                      numQuadsNLP,
      double *                                eshelbyTensorQuadValuesH,
      dataTypes::number *
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
      dataTypes::number *
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHHubbard,
#ifdef USE_COMPLEX
      dataTypes::number
        *projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
      dataTypes::number *
        projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHHubbard,
#endif
      const MPI_Comm &     mpiCommParent,
      const MPI_Comm &     interBandGroupComm,
      const bool           isPsp,
      const bool           isFloatingChargeForces,
      const bool           addEk,
      const dftParameters &dftParams);

  } // namespace force
} // namespace dftfe
