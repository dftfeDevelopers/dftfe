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
#include <DataTypeOverloads.h>

namespace dftfe
{
  namespace force
  {
    namespace
    {
      void
      interpolatePsiComputeELocWfcEshelbyTensor(
        operatorDFTClass &                         operatorMatrix,
        distributedCPUMultiVec<dataTypes::number> &Xb,
        const unsigned int                         BVec,
        const unsigned int                         numCells,
        const unsigned int                         numQuads,
        const unsigned int                         numQuadsNLP,
        const unsigned int                         numNodesPerElement,
        const std::vector<double> &                eigenValues,
        const std::vector<double> &                partialOccupancies,
        const std::vector<double> &                kcoord,
        const std::vector<double> &                onesVec,
        const unsigned int                         cellsBlockSize,
        std::vector<dataTypes::number> &           psiQuadsFlat,
        std::vector<dataTypes::number> &           gradPsiQuadsXFlat,
        std::vector<dataTypes::number> &           gradPsiQuadsYFlat,
        std::vector<dataTypes::number> &           gradPsiQuadsZFlat,
#ifdef USE_COMPLEX
        std::vector<dataTypes::number> &psiQuadsNLP,
#endif
        std::vector<dataTypes::number> &gradPsiQuadsNLPFlat,
        std::vector<double> &           eshelbyTensorContributions,
        std::vector<double> &           eshelbyTensorQuadValues,
        const bool                      isPsp,
        const bool                      isFloatingChargeForces,
        const bool                      addEk)
      {
        std::vector<dataTypes::number> cellWaveFunctionMatrix(
          numCells * numNodesPerElement * BVec, dataTypes::number(0.0));

        for (int icell = 0; icell < numCells; icell++)
          {
            const unsigned int inc = 1;
            for (unsigned int iNode = 0; iNode < numNodesPerElement; ++iNode)
              {
                dftfe::xcopy(
                  &BVec,
                  Xb.data() +
                    operatorMatrix.getFlattenedArrayCellLocalProcIndexIdMap()
                      [icell * numNodesPerElement + iNode],
                  &inc,
                  &cellWaveFunctionMatrix[icell * numNodesPerElement * BVec +
                                          iNode * BVec],
                  &inc);
              }
          }

        const unsigned int blockSize    = cellsBlockSize;
        const unsigned int numberBlocks = numCells / blockSize;
        const unsigned int remBlockSize = numCells - numberBlocks * blockSize;

        std::vector<dataTypes::number> shapeFunctionValuesReference(
          numQuads * numNodesPerElement, dataTypes::number(0.0));
        std::vector<dataTypes::number> shapeFunctionValuesNLPReference(
          numQuadsNLP * numNodesPerElement, dataTypes::number(0.0));

        for (unsigned int i = 0; i < numQuads * numNodesPerElement; ++i)
          shapeFunctionValuesReference[i] = dataTypes::number(
            (operatorMatrix.getShapeFunctionValuesDensityTransposed())[i]);

        for (unsigned int i = 0; i < numQuadsNLP * numNodesPerElement; ++i)
          shapeFunctionValuesNLPReference[i] = dataTypes::number(
            (operatorMatrix.getShapeFunctionValuesNLPTransposed())[i]);

        std::vector<dataTypes::number> shapeFunctionGradientValuesXTransposed(
          blockSize * numQuads * numNodesPerElement, dataTypes::number(0.0));

        std::vector<dataTypes::number> shapeFunctionGradientValuesYTransposed(
          blockSize * numQuads * numNodesPerElement, dataTypes::number(0.0));

        std::vector<dataTypes::number> shapeFunctionGradientValuesZTransposed(
          blockSize * numQuads * numNodesPerElement, dataTypes::number(0.0));

        std::vector<double> shapeFunctionGradientValuesNLPReference(
          blockSize * numQuadsNLP * 3 * numNodesPerElement, 0.0);
        std::vector<double> shapeFunctionGradientValuesNLP(
          blockSize * numQuadsNLP * 3 * numNodesPerElement, 0.0);
        std::vector<dataTypes::number> shapeFunctionGradientValuesNLPCopy(
          blockSize * numQuadsNLP * 3 * numNodesPerElement,
          dataTypes::number(0.0));

        for (unsigned int i = 0; i < blockSize; i++)
          for (unsigned int j = 0; j < (numQuadsNLP * 3 * numNodesPerElement);
               ++j)
            shapeFunctionGradientValuesNLPReference[i * numQuadsNLP * 3 *
                                                      numNodesPerElement +
                                                    j] =
              (operatorMatrix.getShapeFunctionGradientValuesNLPTransposed())[j];

        for (int iblock = 0; iblock < (numberBlocks + 1); iblock++)
          {
            const unsigned int currentBlockSize =
              (iblock == numberBlocks) ? remBlockSize : blockSize;
            const unsigned int startingId = iblock * blockSize;

            if (currentBlockSize > 0)
              {
                const dataTypes::number scalarCoeffAlpha =
                  dataTypes::number(1.0);
                const dataTypes::number scalarCoeffBeta =
                  dataTypes::number(0.0);
                const double scalarCoeffAlphaReal = 1.0;
                const double scalarCoeffBetaReal  = 0.0;

                const char transA = 'N', transB = 'N';

                int strideA = BVec * numNodesPerElement;
                int strideB = 0;
                int strideC = BVec * numQuads;

                if (!isFloatingChargeForces)
                  {
                    for (int j = 0; j < currentBlockSize; j++)
                      dftfe::xgemm(
                        &transA,
                        &transB,
                        &BVec,
                        &numQuads,
                        &numNodesPerElement,
                        &scalarCoeffAlpha,
                        &cellWaveFunctionMatrix[startingId *
                                                  numNodesPerElement * BVec +
                                                j * strideA],
                        &BVec,
                        &shapeFunctionValuesReference[0],
                        &numNodesPerElement,
                        &scalarCoeffBeta,
                        &psiQuadsFlat[j * strideC],
                        &BVec);

                    strideB = numNodesPerElement * numQuads;

                    for (unsigned int i = 0; i < currentBlockSize * numQuads;
                         ++i)
                      for (unsigned int j = 0; j < numNodesPerElement; ++j)
                        shapeFunctionGradientValuesXTransposed
                          [i * numNodesPerElement + j] = dataTypes::number(
                            (operatorMatrix
                               .getShapeFunctionGradValuesDensityGaussQuad())
                              [startingId * numQuads * 3 * numNodesPerElement +
                               i * 3 * numNodesPerElement + j]);


                    for (int j = 0; j < currentBlockSize; j++)
                      dftfe::xgemm(
                        &transA,
                        &transB,
                        &BVec,
                        &numQuads,
                        &numNodesPerElement,
                        &scalarCoeffAlpha,
                        &cellWaveFunctionMatrix[startingId *
                                                  numNodesPerElement * BVec +
                                                j * strideA],
                        &BVec,
                        &shapeFunctionGradientValuesXTransposed[j * strideB],
                        &numNodesPerElement,
                        &scalarCoeffBeta,
                        &gradPsiQuadsXFlat[j * strideC],
                        &BVec);


                    for (unsigned int i = 0; i < currentBlockSize * numQuads;
                         ++i)
                      for (unsigned int j = 0; j < numNodesPerElement; ++j)
                        shapeFunctionGradientValuesYTransposed
                          [i * numNodesPerElement + j] = dataTypes::number(
                            (operatorMatrix
                               .getShapeFunctionGradValuesDensityGaussQuad())
                              [startingId * numQuads * 3 * numNodesPerElement +
                               i * 3 * numNodesPerElement + numNodesPerElement +
                               j]);

                    for (int j = 0; j < currentBlockSize; j++)
                      dftfe::xgemm(
                        &transA,
                        &transB,
                        &BVec,
                        &numQuads,
                        &numNodesPerElement,
                        &scalarCoeffAlpha,
                        &cellWaveFunctionMatrix[startingId *
                                                  numNodesPerElement * BVec +
                                                j * strideA],
                        &BVec,
                        &shapeFunctionGradientValuesYTransposed[j * strideB],
                        &numNodesPerElement,
                        &scalarCoeffBeta,
                        &gradPsiQuadsYFlat[j * strideC],
                        &BVec);

                    for (unsigned int i = 0; i < currentBlockSize * numQuads;
                         ++i)
                      for (unsigned int j = 0; j < numNodesPerElement; ++j)
                        shapeFunctionGradientValuesZTransposed
                          [i * numNodesPerElement + j] = dataTypes::number(
                            (operatorMatrix
                               .getShapeFunctionGradValuesDensityGaussQuad())
                              [startingId * numQuads * 3 * numNodesPerElement +
                               i * 3 * numNodesPerElement +
                               2 * numNodesPerElement + j]);

                    for (int j = 0; j < currentBlockSize; j++)
                      dftfe::xgemm(
                        &transA,
                        &transB,
                        &BVec,
                        &numQuads,
                        &numNodesPerElement,
                        &scalarCoeffAlpha,
                        &cellWaveFunctionMatrix[startingId *
                                                  numNodesPerElement * BVec +
                                                j * strideA],
                        &BVec,
                        &shapeFunctionGradientValuesZTransposed[j * strideB],
                        &numNodesPerElement,
                        &scalarCoeffBeta,
                        &gradPsiQuadsZFlat[j * strideC],
                        &BVec);

                    const double absksq = kcoord[0] * kcoord[0] +
                                          kcoord[1] * kcoord[1] +
                                          kcoord[2] * kcoord[2];
                    for (unsigned int j = 0; j < currentBlockSize; j++)
                      for (unsigned int iquad = 0; iquad < numQuads; iquad++)
                        for (unsigned int iwfc = 0; iwfc < BVec; iwfc++)
                          {
                            const dataTypes::number psiQuad =
                              psiQuadsFlat[j * numQuads * BVec + iquad * BVec +
                                           iwfc];
                            const double partOcc    = partialOccupancies[iwfc];
                            const double eigenValue = eigenValues[iwfc];

                            std::vector<dataTypes::number> gradPsiQuad(3);
                            gradPsiQuad[0] =
                              gradPsiQuadsXFlat[j * numQuads * BVec +
                                                iquad * BVec + iwfc];
                            gradPsiQuad[1] =
                              gradPsiQuadsYFlat[j * numQuads * BVec +
                                                iquad * BVec + iwfc];

                            gradPsiQuad[2] =
                              gradPsiQuadsZFlat[j * numQuads * BVec +
                                                iquad * BVec + iwfc];

                            const double identityFactor =
                              partOcc *
                                dftfe::utils::realPart((
                                  dftfe::utils::complexConj(gradPsiQuad[0]) *
                                    gradPsiQuad[0] +
                                  dftfe::utils::complexConj(gradPsiQuad[1]) *
                                    gradPsiQuad[1] +
                                  dftfe::utils::complexConj(gradPsiQuad[2]) *
                                    gradPsiQuad[2] +
                                  dataTypes::number(absksq - 2.0 * eigenValue) *
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
                                      -partOcc * dftfe::utils::realPart(
                                                   dftfe::utils::complexConj(
                                                     gradPsiQuad[idim]) *
                                                     gradPsiQuad[jdim] +
                                                   gradPsiQuad[idim] *
                                                     dftfe::utils::complexConj(
                                                       gradPsiQuad[jdim])) -
                                      2.0 * partOcc *
                                        dftfe::utils::imagPart(
                                          dftfe::utils::complexConj(psiQuad) *
                                          (gradPsiQuad[idim] * kcoord[jdim]));

                                  if (idim == jdim)
                                    eshelbyTensorContributions
                                      [j * numQuads * 9 * BVec +
                                       iquad * 9 * BVec + idim * 3 * BVec +
                                       jdim * BVec + iwfc] += identityFactor;
                                }
#ifdef USE_COMPLEX
                            if (addEk)
                              {
                                for (unsigned int idim = 0; idim < 3; idim++)
                                  for (unsigned int jdim = 0; jdim < 3; jdim++)
                                    {
                                      eshelbyTensorContributions
                                        [j * numQuads * 9 * BVec +
                                         iquad * 9 * BVec + idim * 3 * BVec +
                                         jdim * BVec + iwfc] +=
                                        -2.0 * partOcc *
                                          dftfe::utils::imagPart(
                                            dftfe::utils::complexConj(psiQuad) *
                                            (kcoord[idim] *
                                             gradPsiQuad[jdim])) -
                                        2.0 * partOcc *
                                          dftfe::utils::realPart(
                                            kcoord[idim] * kcoord[jdim] *
                                            dftfe::utils::complexConj(psiQuad) *
                                            psiQuad);
                                    }
                              }
#endif
                          }

                    const double       scalarCoeffAlphaEshelby = 1.0;
                    const double       scalarCoeffBetaEshelby  = 1.0;
                    const unsigned int m                       = 1;
                    const unsigned int n = currentBlockSize * numQuads * 9;
                    const unsigned int k = BVec;
                    dftfe::xgemm(
                      &transA,
                      &transB,
                      &m,
                      &n,
                      &k,
                      &scalarCoeffAlphaEshelby,
                      &onesVec[0],
                      &m,
                      &eshelbyTensorContributions[0],
                      &k,
                      &scalarCoeffBetaEshelby,
                      &eshelbyTensorQuadValues[startingId * numQuads * 9],
                      &m);
                  }

                if (isPsp)
                  {
#ifdef USE_COMPLEX
                    const unsigned int strideCNLP = BVec * numQuadsNLP;
                    const unsigned int strideBNLP = 0;
                    for (int j = 0; j < currentBlockSize; j++)
                      dftfe::xgemm(
                        &transA,
                        &transB,
                        &BVec,
                        &numQuadsNLP,
                        &numNodesPerElement,
                        &scalarCoeffAlpha,
                        &cellWaveFunctionMatrix[startingId *
                                                  numNodesPerElement * BVec +
                                                j * strideA],
                        &BVec,
                        &shapeFunctionValuesNLPReference[0],
                        &numNodesPerElement,
                        &scalarCoeffBeta,
                        &psiQuadsNLP[startingId * numQuadsNLP * BVec +
                                     j * strideCNLP],
                        &BVec);

#endif

                    const unsigned int three = 3;
                    // shapeGradRef^T*invJacobian^T
                    for (int j = 0; j < currentBlockSize * numQuadsNLP; j++)
                      dftfe::xgemm(
                        &transA,
                        &transB,
                        &numNodesPerElement,
                        &three,
                        &three,
                        &scalarCoeffAlphaReal,
                        &shapeFunctionGradientValuesNLPReference
                          [0 + j * numNodesPerElement * 3],
                        &numNodesPerElement,
                        &(operatorMatrix.getInverseJacobiansNLP())
                          [startingId * numQuadsNLP * 3 * 3 + j * 3 * 3],
                        &three,
                        &scalarCoeffBetaReal,
                        &shapeFunctionGradientValuesNLP[j * numNodesPerElement *
                                                        3],
                        &numNodesPerElement);


                    for (unsigned int i = 0;
                         i < currentBlockSize * numQuadsNLP *
                               numNodesPerElement * 3;
                         ++i)
                      shapeFunctionGradientValuesNLPCopy[i] =
                        dataTypes::number(shapeFunctionGradientValuesNLP[i]);



                    const unsigned int strideCNLPGrad = BVec * 3 * numQuadsNLP;
                    const unsigned int strideBNLPGrad =
                      numNodesPerElement * 3 * numQuadsNLP;
                    const unsigned int n = 3 * numQuadsNLP;
                    for (int j = 0; j < currentBlockSize; j++)
                      dftfe::xgemm(
                        &transA,
                        &transB,
                        &BVec,
                        &n,
                        &numNodesPerElement,
                        &scalarCoeffAlpha,
                        &cellWaveFunctionMatrix[startingId *
                                                  numNodesPerElement * BVec +
                                                j * strideA],
                        &BVec,
                        &shapeFunctionGradientValuesNLPCopy[j * strideBNLPGrad],
                        &numNodesPerElement,
                        &scalarCoeffBeta,
                        &gradPsiQuadsNLPFlat[startingId * numQuadsNLP * 3 *
                                               BVec +
                                             j * strideCNLPGrad],
                        &BVec);
                  }
              }
          }
      }


      void
      nlpPsiContraction(
        operatorDFTClass &operatorMatrix,
#ifdef USE_COMPLEX
        const std::vector<dataTypes::number> &psiQuadsNLP,
#endif
        const std::vector<dataTypes::number> &gradPsiQuadsNLP,
        const std::vector<double> &           partialOccupancies,
        const std::vector<dataTypes::number> &onesVecNLP,
        const dataTypes::number *projectorKetTimesVectorParFlattened,
        const unsigned int *     nonTrivialIdToElemIdMap,
        const unsigned int *     projecterKetTimesFlattenedVectorLocalIds,
        const unsigned int       numCells,
        const unsigned int       numQuadsNLP,
        const unsigned int       numPsi,
        const unsigned int       totalNonTrivialPseudoWfcs,
        const unsigned int       innerBlockSizeEnlp,
        std::vector<dataTypes::number> &nlpContractionContribution,
        std::vector<dataTypes::number> &
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlock,
        dataTypes::number *
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
#ifdef USE_COMPLEX
        ,
        std::vector<dataTypes::number> &
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock,
        dataTypes::number
          *projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened
#endif
      )
      {
        const unsigned int blockSizeNlp = innerBlockSizeEnlp;
        const unsigned int numberBlocksNlp =
          totalNonTrivialPseudoWfcs / blockSizeNlp;
        const unsigned int remBlockSizeNlp =
          totalNonTrivialPseudoWfcs - numberBlocksNlp * blockSizeNlp;

        dataTypes::number scalarCoeffAlphaNlp = dataTypes::number(1.0);
        dataTypes::number scalarCoeffBetaNlp  = dataTypes::number(0.0);


        const char transA = 'N', transB = 'N';

        for (int iblocknlp = 0; iblocknlp < (numberBlocksNlp + 1); iblocknlp++)
          {
            const unsigned int currentBlockSizeNlp =
              (iblocknlp == numberBlocksNlp) ? remBlockSizeNlp : blockSizeNlp;
            const unsigned int startingIdNlp = iblocknlp * blockSizeNlp;
            if (currentBlockSizeNlp > 0)
              {
                for (unsigned int ipseudowfc = 0;
                     ipseudowfc < currentBlockSizeNlp;
                     ipseudowfc++)
                  for (unsigned int iquad = 0; iquad < (numQuadsNLP * 3);
                       iquad++)
                    for (unsigned int iwfc = 0; iwfc < numPsi; iwfc++)
                      {
                        nlpContractionContribution[ipseudowfc * numQuadsNLP *
                                                     3 * numPsi +
                                                   iquad * numPsi + iwfc] =
                          partialOccupancies[iwfc] *
                          dftfe::utils::complexConj(
                            gradPsiQuadsNLP[nonTrivialIdToElemIdMap
                                                [startingIdNlp + ipseudowfc] *
                                              numQuadsNLP * 3 * numPsi +
                                            iquad * numPsi + iwfc]) *
                          projectorKetTimesVectorParFlattened
                            [projecterKetTimesFlattenedVectorLocalIds
                                 [startingIdNlp + ipseudowfc] *
                               numPsi +
                             iwfc];
                      }

                unsigned int m = 1;
                unsigned int n = currentBlockSizeNlp * numQuadsNLP * 3;
                unsigned int k = numPsi;
                dftfe::xgemm(
                  &transA,
                  &transB,
                  &m,
                  &n,
                  &k,
                  &scalarCoeffAlphaNlp,
                  &onesVecNLP[0],
                  &m,
                  &nlpContractionContribution[0],
                  &k,
                  &scalarCoeffBetaNlp,
                  &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlock
                    [0],
                  &m);


                for (unsigned int i = 0;
                     i < (currentBlockSizeNlp * numQuadsNLP * 3);
                     i++)
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
                    [startingIdNlp * numQuadsNLP * 3 + i] +=
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlock
                      [i];
#ifdef USE_COMPLEX
                for (unsigned int ipseudowfc = 0;
                     ipseudowfc < currentBlockSizeNlp;
                     ipseudowfc++)
                  for (unsigned int iquad = 0; iquad < numQuadsNLP; iquad++)
                    for (unsigned int iwfc = 0; iwfc < numPsi; iwfc++)
                      nlpContractionContribution[ipseudowfc * numQuadsNLP *
                                                   numPsi +
                                                 iquad * numPsi + iwfc] =
                        partialOccupancies[iwfc] *
                        dftfe::utils::complexConj(
                          psiQuadsNLP[nonTrivialIdToElemIdMap[startingIdNlp +
                                                              ipseudowfc] *
                                        numQuadsNLP * numPsi +
                                      iquad * numPsi + iwfc]) *
                        projectorKetTimesVectorParFlattened
                          [projecterKetTimesFlattenedVectorLocalIds
                               [startingIdNlp + ipseudowfc] *
                             numPsi +
                           iwfc];

                n = currentBlockSizeNlp * numQuadsNLP;
                dftfe::xgemm(
                  &transA,
                  &transB,
                  &m,
                  &n,
                  &k,
                  &scalarCoeffAlphaNlp,
                  &onesVecNLP[0],
                  &m,
                  &nlpContractionContribution[0],
                  &k,
                  &scalarCoeffBetaNlp,
                  &projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock
                    [0],
                  &m);

                for (unsigned int i = 0; i < currentBlockSizeNlp * numQuadsNLP;
                     i++)
                  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened
                    [startingIdNlp * numQuadsNLP + i] +=
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock
                      [i];
#endif
              }
          }
      }


      void
      computeBlockContribution(
        operatorDFTClass &                         operatorMatrix,
        distributedCPUMultiVec<dataTypes::number> &flattenedArrayBlock,
        distributedCPUMultiVec<dataTypes::number> &projectorKetTimesVector,
        const dataTypes::number *                  X,
        const std::vector<double> &                eigenValues,
        const std::vector<double> &                partialOccupancies,
        const std::vector<double> &                kcoord,
        const std::vector<double> &                onesVec,
        const std::vector<dataTypes::number> &     onesVecNLP,
        const unsigned int *                       nonTrivialIdToElemIdMap,
        const unsigned int *projecterKetTimesFlattenedVectorLocalIds,
        const unsigned int  startingVecId,
        const unsigned int  M,
        const unsigned int  N,
        const unsigned int  numPsi,
        const unsigned int  numCells,
        const unsigned int  numQuads,
        const unsigned int  numQuadsNLP,
        const unsigned int  numNodesPerElement,
        const unsigned int  totalNonTrivialPseudoWfcs,
        std::vector<dataTypes::number> &psiQuadsFlat,
        std::vector<dataTypes::number> &gradPsiQuadsXFlat,
        std::vector<dataTypes::number> &gradPsiQuadsYFlat,
        std::vector<dataTypes::number> &gradPsiQuadsZFlat,
#ifdef USE_COMPLEX
        std::vector<dataTypes::number> &psiQuadsNLP,
#endif
        std::vector<dataTypes::number> &gradPsiQuadsNLPFlat,
        std::vector<double> &           eshelbyTensorContributions,
        std::vector<double> &           eshelbyTensorQuadValues,
        std::vector<dataTypes::number> &nlpContractionContribution,
        std::vector<dataTypes::number> &
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlock,
        dataTypes::number *
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened,
#ifdef USE_COMPLEX
        std::vector<dataTypes::number> &
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock,
        dataTypes::number
          *projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened,
#endif
        const unsigned int cellsBlockSize,
        const unsigned int innerBlockSizeEnlp,
        const bool         isPsp,
        const bool         isFloatingChargeForces,
        const bool         addEk)
      {
        for (unsigned int iNode = 0; iNode < M; ++iNode)
          for (unsigned int iWave = 0; iWave < numPsi; ++iWave)
            flattenedArrayBlock.data()[iNode * numPsi + iWave] =
              X[iNode * N + startingVecId + iWave];

        (operatorMatrix.getOverloadedConstraintMatrix())
          ->distribute(flattenedArrayBlock, numPsi);

        interpolatePsiComputeELocWfcEshelbyTensor(operatorMatrix,
                                                  flattenedArrayBlock,
                                                  numPsi,
                                                  numCells,
                                                  numQuads,
                                                  numQuadsNLP,
                                                  numNodesPerElement,
                                                  eigenValues,
                                                  partialOccupancies,
                                                  kcoord,
                                                  onesVec,
                                                  cellsBlockSize,
                                                  psiQuadsFlat,
                                                  gradPsiQuadsXFlat,
                                                  gradPsiQuadsYFlat,
                                                  gradPsiQuadsZFlat,
#ifdef USE_COMPLEX
                                                  psiQuadsNLP,
#endif
                                                  gradPsiQuadsNLPFlat,
                                                  eshelbyTensorContributions,
                                                  eshelbyTensorQuadValues,
                                                  isPsp,
                                                  isFloatingChargeForces,
                                                  addEk);

        if (isPsp)
          {
            operatorMatrix.computeNonLocalProjectorKetTimesXTimesV(
              flattenedArrayBlock, projectorKetTimesVector, numPsi);

            if (totalNonTrivialPseudoWfcs > 0)
              {
                nlpPsiContraction(
                  operatorMatrix,
#ifdef USE_COMPLEX
                  psiQuadsNLP,
#endif
                  gradPsiQuadsNLPFlat,
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
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened
#ifdef USE_COMPLEX
                  ,
                  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock,
                  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened
#endif
                );
              }
          }
      }

    } // namespace

    void
    wfcContractionsForceKernelsAllH(
      operatorDFTClass &                      operatorMatrix,
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
        std::min((unsigned int)2, bandGroupLowHighPlusOneIndices[1]);


      distributedCPUMultiVec<dataTypes::number> flattenedArrayBlock;


      distributedCPUMultiVec<dataTypes::number> &projectorKetTimesVector =
        operatorMatrix.getParallelProjectorKetTimesBlockVector();


      std::vector<double> elocWfcEshelbyTensorQuadValuesH(numCells * numQuads *
                                                            9,
                                                          0.0);

      std::vector<double>            onesVecH(blockSize, 1.0);
      std::vector<dataTypes::number> onesVecHNLP(blockSize,
                                                 dataTypes::number(1.0));

      const unsigned int cellsBlockSize = std::min((unsigned int)1, numCells);

      std::vector<dataTypes::number> psiQuadsFlatH(cellsBlockSize * numQuads *
                                                     blockSize,
                                                   dataTypes::number(0.0));
      std::vector<dataTypes::number> gradPsiQuadsXFlatH(cellsBlockSize *
                                                          numQuads * blockSize,
                                                        dataTypes::number(0.0));
      std::vector<dataTypes::number> gradPsiQuadsYFlatH(cellsBlockSize *
                                                          numQuads * blockSize,
                                                        dataTypes::number(0.0));
      std::vector<dataTypes::number> gradPsiQuadsZFlatH(cellsBlockSize *
                                                          numQuads * blockSize,
                                                        dataTypes::number(0.0));
#ifdef USE_COMPLEX
      std::vector<dataTypes::number> psiQuadsNLPH(numCells * numQuadsNLP *
                                                    blockSize,
                                                  dataTypes::number(0.0));
#endif

      std::vector<dataTypes::number> gradPsiQuadsNLPFlatH(
        numCells * numQuadsNLP * 3 * blockSize, dataTypes::number(0.0));

      std::vector<double> eshelbyTensorContributionsH(
        cellsBlockSize * numQuads * blockSize * 9, 0.0);

      const unsigned int innerBlockSizeEnlp =
        std::min((unsigned int)10, totalNonTrivialPseudoWfcs);
      std::vector<dataTypes::number> nlpContractionContributionH(
        innerBlockSizeEnlp * numQuadsNLP * 3 * blockSize,
        dataTypes::number(0.0));
      std::vector<dataTypes::number>
        projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHBlock;
      std::vector<dataTypes::number>
        projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHBlock;
      if (totalNonTrivialPseudoWfcs > 0)
        {
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHBlock
            .resize(innerBlockSizeEnlp * numQuadsNLP * 3,
                    dataTypes::number(0.0));
#ifdef USE_COMPLEX
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHBlock
            .resize(innerBlockSizeEnlp * numQuadsNLP, dataTypes::number(0.0));
#endif
        }

      const unsigned int numKPoints = kPointCoordinates.size() / 3;
      for (unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
        {
          std::fill(elocWfcEshelbyTensorQuadValuesH.begin(),
                    elocWfcEshelbyTensorQuadValuesH.end(),
                    0.);
          // spin index update is not required
          operatorMatrix.reinitkPointSpinIndex(kPoint, 0);
          std::vector<double> kcoord(3, 0.0);
          kcoord[0] = kPointCoordinates[kPoint * 3 + 0];
          kcoord[1] = kPointCoordinates[kPoint * 3 + 1];
          kcoord[2] = kPointCoordinates[kPoint * 3 + 2];

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

          for (unsigned int jvec = 0; jvec < N; jvec += blockSize)
            {
              const unsigned int currentBlockSize =
                std::min(blockSize, N - jvec);

              if (currentBlockSize != blockSize || jvec == 0)
                operatorMatrix.reinit(currentBlockSize,
                                      flattenedArrayBlock,
                                      true);

              if ((jvec + currentBlockSize) <=
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                  (jvec + currentBlockSize) >
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  std::vector<double> blockedEigenValues(currentBlockSize, 0.0);
                  std::vector<double> blockedPartialOccupancies(
                    currentBlockSize, 0.0);
                  for (unsigned int iWave = 0; iWave < currentBlockSize;
                       ++iWave)
                    {
                      blockedEigenValues[iWave] =
                        eigenValuesH[kPoint][spinIndex * N + jvec + iWave];
                      blockedPartialOccupancies[iWave] =
                        partialOccupanciesH[kPoint]
                                           [spinIndex * N + jvec + iWave];
                    }


                  computeBlockContribution(
                    operatorMatrix,
                    flattenedArrayBlock,
                    projectorKetTimesVector,
                    X +
                      ((1 + spinPolarizedFlag) * kPoint + spinIndex) * MLoc * N,
                    blockedEigenValues,
                    blockedPartialOccupancies,
                    kcoord,
                    onesVecH,
                    onesVecHNLP,
                    nonTrivialIdToElemIdMapH,
                    projecterKetTimesFlattenedVectorLocalIdsH,
                    jvec,
                    MLoc,
                    N,
                    currentBlockSize,
                    numCells,
                    numQuads,
                    numQuadsNLP,
                    numNodesPerElement,
                    totalNonTrivialPseudoWfcs,
                    psiQuadsFlatH,
                    gradPsiQuadsXFlatH,
                    gradPsiQuadsYFlatH,
                    gradPsiQuadsZFlatH,
#ifdef USE_COMPLEX
                    psiQuadsNLPH,
#endif
                    gradPsiQuadsNLPFlatH,
                    eshelbyTensorContributionsH,
                    elocWfcEshelbyTensorQuadValuesH,
                    nlpContractionContributionH,
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHBlock,
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH +
                      kPoint * totalNonTrivialPseudoWfcs * numQuadsNLP * 3,
#ifdef USE_COMPLEX
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedHBlock,
                    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH +
                      kPoint * totalNonTrivialPseudoWfcs * numQuadsNLP,
#endif
                    cellsBlockSize,
                    innerBlockSizeEnlp,
                    isPsp,
                    isFloatingChargeForces,
                    addEk);

                } // band parallelization
            }     // ivec loop

          for (unsigned int icopy = 0; (icopy < numCells * numQuads * 9);
               icopy++)
            eshelbyTensorQuadValuesH[kPoint * numCells * numQuads * 9 + icopy] =
              elocWfcEshelbyTensorQuadValuesH[icopy];

        } // k point loop
    }

  } // namespace force
} // namespace dftfe
