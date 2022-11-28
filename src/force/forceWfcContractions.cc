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
    namespace
    {

      double
      realPart(const double x)
      {
        return x;
      }

      double
      realPart(const std::complex<double> x)
      {
        return x.real();
      }

      double
      complexConj(const double x)
      {
        return x;
      }

      std::complex<double>
      complexConj(const std::complex<double> x)
      {
        return std::conj(x);
      }

      void
      interpolatePsiComputeELocWfcEshelbyTensorD(
        operatorDFTClass &                   operatorMatrix,
        distributedCPUVec<dataTypes::number> &Xb,
        const unsigned int                       BVec,
        const unsigned int                       numCells,
        const unsigned int                       numQuads,
        const unsigned int                       numQuadsNLP,
        const unsigned int                       numNodesPerElement,
        const std::vector<double> &    eigenValues,
        const std::vector<double> &    partialOccupancies,
#ifdef USE_COMPLEX
        const double kcoordx,
        const double kcoordy,
        const double kcoordz,
#endif
        const std::vector<double> &              onesVec,
        const unsigned int                                 cellsBlockSize,
        std::vector<dataTypes::number> &psiQuadsFlat,
        std::vector<dataTypes::number> &gradPsiQuadsXFlat,
        std::vector<dataTypes::number> &gradPsiQuadsYFlat,
        std::vector<dataTypes::number> &gradPsiQuadsZFlat,
#ifdef USE_COMPLEX
        std::vector<dataTypes::number> &psiQuadsNLP,
#endif
        std::vector<dataTypes::number> &gradPsiQuadsNLPFlat,
        std::vector<double> &eshelbyTensorContributions,
        std::vector<double> &eshelbyTensorQuadValues,
        const bool                     isPsp,
        const bool                     isFloatingChargeForces,
        const bool                     addEk)
      {
        std::vector<double> cellWaveFunctionMatrix(numNodesPerElement * BVec,0.0);

        for (int icell = 0; icell < numCells; icell++)
          {
            const unsigned int inc = 1;
            for (unsigned int iNode = 0; iNode < numNodesPerElement;
                 ++iNode)
              {
                dftfe::xcopy(
                  &BVec,
                  Xb.begin() +
                    operatorMatrix
                      .getFlattenedArrayCellLocalProcIndexIdMap()
                        [icell * numNodesPerElement + iNode],
                  &inc,
                  &cellWaveFunctionMatrix[BVec * iNode],
                  &inc);
              }
          }

        const unsigned int blockSize    = cellsBlockSize;
        const unsigned int numberBlocks = numCells / blockSize;
        const unsigned int remBlockSize = numCells - numberBlocks * blockSize;

        std::vector<dataTypes::number>
          shapeFunctionValuesReference(numQuads * numNodesPerElement,
                                        dataTypes::number(0.0));
        std::vector<dataTypes::number>
          shapeFunctionValuesNLPReference(numQuadsNLP * numNodesPerElement,
                                           dataTypes::number(0.0));

        for (unsigned int i = 0;i < numQuads * numNodesPerElement;++i)
          shapeFunctionValuesReference[i]=dataTypes::number((operatorMatrix.getShapeFunctionValuesDensityTransposed())[i]);

        for (unsigned int i = 0;i < numQuadsNLP * numNodesPerElement;++i)
          shapeFunctionValuesNLPReference[i]=dataTypes::number((operatorMatrix.getShapeFunctionValuesNLPTransposed())[i]);

        std::vector<dataTypes::number>
          shapeFunctionGradientValuesXTransposed(
            blockSize * numQuads * numNodesPerElement,
            dataTypes::number(0.0));

        std::vector<dataTypes::number>
          shapeFunctionGradientValuesYTransposed(
            blockSize * numQuads * numNodesPerElement,
            dataTypes::number(0.0));

        std::vector<dataTypes::number>
          shapeFunctionGradientValuesZTransposed(
            blockSize * numQuads * numNodesPerElement,
            dataTypes::number(0.0));

        std::vector<double> shapeFunctionGradientValuesNLPReference(
          blockSize * numQuadsNLP * 3 * numNodesPerElement, 0.0);
        std::vector<double> shapeFunctionGradientValuesNLP(
          blockSize * numQuadsNLP * 3 * numNodesPerElement, 0.0);
        std::vector<dataTypes::number>
          shapeFunctionGradientValuesNLPCopy(blockSize * numQuadsNLP * 3 *
                                                numNodesPerElement,
                                              dataTypes::number(0.0));

        for (unsigned int i = 0; i < blockSize; i++)
          std::copy(
            operatorMatrix.getShapeFunctionGradientValuesNLPTransposed().begin(),
            operatorMatrix.getShapeFunctionGradientValuesNLPTransposed().end(),
            shapeFunctionGradientValuesNLPReference.begin() +
              i * numQuadsNLP * 3 * numNodesPerElement);



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
                                                  numNodesPerElement * BVec+j*strideA],
                        &BVec,
                        &shapeFunctionValuesReference[0],
                        &numNodesPerElement,
                        &scalarCoeffBeta,
                        &psiQuadsFlat[j*strideC],
                        &BVec);

                    strideB = numNodesPerElement * numQuads;

                    for (unsigned int i = 0;i < currentBlockSize * numQuads * numNodesPerElement;++i)
                       shapeFunctionGradientValuesXTransposed[i]=dataTypes::number((operatorMatrix
                            .getShapeFunctionGradientValuesXDensityTransposed())
                          [startingId * numQuads * numNodesPerElement+i]);


                    for (int j = 0; j < currentBlockSize; j++)
                      dftfe::xgemm(
                        &transA,
                        &transB,
                        &BVec,
                        &numQuads,
                        &numNodesPerElement,
                        &scalarCoeffAlpha,
                        &cellWaveFunctionMatrix[startingId *
                                                  numNodesPerElement * BVec+j*strideA],
                        &BVec,
                        &shapeFunctionGradientValuesXTransposed[j*strideB],
                        &numNodesPerElement,
                        &scalarCoeffBeta,
                        &gradPsiQuadsXFlat[j*strideC],
                        &BVec);


                    for (unsigned int i = 0;i < currentBlockSize * numQuads * numNodesPerElement;++i)
                       shapeFunctionGradientValuesYTransposed[i]=dataTypes::number((operatorMatrix
                            .getShapeFunctionGradientValuesYDensityTransposed())
                          [startingId * numQuads * numNodesPerElement+i]);

                    for (int j = 0; j < currentBlockSize; j++)
                      dftfe::xgemm(
                        &transA,
                        &transB,
                        &BVec,
                        &numQuads,
                        &numNodesPerElement,
                        &scalarCoeffAlpha,
                        &cellWaveFunctionMatrix[startingId *
                                                  numNodesPerElement * BVec+j*strideA],
                        &BVec,
                        &shapeFunctionGradientValuesYTransposed[j*strideB],
                        &numNodesPerElement,
                        &scalarCoeffBeta,
                        &gradPsiQuadsYFlat[j*strideC],
                        &BVec);

                    for (unsigned int i = 0;i < currentBlockSize * numQuads * numNodesPerElement;++i)
                       shapeFunctionGradientValuesZTransposed[i]=dataTypes::number((operatorMatrix
                            .getShapeFunctionGradientValuesZDensityTransposed())
                          [startingId * numQuads * numNodesPerElement+i]);

                    for (int j = 0; j < currentBlockSize; j++)
                      dftfe::xgemm(
                        &transA,
                        &transB,
                        &BVec,
                        &numQuads,
                        &numNodesPerElement,
                        &scalarCoeffAlpha,
                        &cellWaveFunctionMatrix[startingId *
                                                  numNodesPerElement * BVec+j*strideA],
                        &BVec,
                        &shapeFunctionGradientValuesZTransposed[j*strideB],
                        &numNodesPerElement,
                        &scalarCoeffBeta,
                        &gradPsiQuadsZFlat[j*strideC],
                        &BVec);


                   for (unsigned int j=0;j<currentBlockSize; j++)
                      for (unsigned int iquad=0;iquad<numQuads; iquad++)
                         for (unsigned int iwfc=0;iwfc<BVec; iwfc++)
                         {
#ifdef USE_COMPLEX
                           if(addEk)
                           {
                             for (unsigned int idim=0; idim<3; idim++)
                               for (unsigned int jdim=0; jdim<3; jdim++)
                                  eshelbyTensorContributions[startingId * numQuads * 9] =
                                    -2.0 * partOcc * cuCmul(gradPsiXConj, gradPsiX).x +
                                    -2.0 * partOcc * cuCmul(psiConj, gradPsiX).y * kcoordx -
                                    2.0 * partOcc * cuCmul(psiConj, gradPsiX).y * kcoordx -
                                    2.0 * partOcc * cuCmul(psiConj, psi).x * kcoordx * kcoordx +
                                    identityFactor;                             
                           }
                           else
#endif                       
                           {
                           }
                         }

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
                    const unsigned int m=1;
                    const unsigned int n=currentBlockSize * numQuads * 9;
                    const unsigned int k=BVec;
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
                                                  numNodesPerElement * BVec+j*strideA],
                        &BVec,
                        &shapeFunctionValuesNLPReference[0],
                        &numNodesPerElement,
                        &scalarCoeffBeta,
                        &psiQuadsNLP[startingId * numQuadsNLP * BVec + j*strideCNLP],
                        &BVec);

#endif
                    
                    const unsigned int three=3;
                    // shapeGradRef^T*invJacobian^T
                    for (int j = 0; j < currentBlockSize*numQuadsNLP; j++)
                      dftfe::xgemm(
                        &transA,
                        &transB,
                        &numNodesPerElement,
                        &three,
                        &three,
                        &scalarCoeffAlphaReal,
                        &shapeFunctionGradientValuesNLPReference[0+j*numNodesPerElement * 3],
                        &numNodesPerElement,
                        &(operatorMatrix
                            .getInverseJacobiansNLP())[startingId *
                                                       numQuadsNLP * 3 * 3+j*3*3],
                        &three,
                        &scalarCoeffBetaReal,
                        &shapeFunctionGradientValuesNLP[j*numNodesPerElement * 3],
                        &numNodesPerElement);


                    for (unsigned int i = 0;i < currentBlockSize * numQuadsNLP * numNodesPerElement * 3;++i)
                      shapeFunctionGradientValuesNLPCopy[i]=dataTypes::number(shapeFunctionGradientValuesNLP[i]);



                    const unsigned int strideCNLPGrad = BVec * 3 * numQuadsNLP;
                    const unsigned int strideBNLPGrad =
                      numNodesPerElement * 3 * numQuadsNLP;
                    const unsigned int n=3 * numQuadsNLP;
                    for (int j = 0; j < currentBlockSize; j++)
                      dftfe::xgemm(
                        &transA,
                        &transB,
                        &BVec,
                        &n,
                        &numNodesPerElement,
                        &scalarCoeffAlpha,
                        &cellWaveFunctionMatrix[startingId *
                                                  numNodesPerElement * BVec+j*strideA],
                        &BVec,
                        &shapeFunctionGradientValuesNLPCopy[j*strideBNLPGrad],
                        &numNodesPerElement,
                        &scalarCoeffBeta,
                        &gradPsiQuadsNLPFlatD[startingId * numQuadsNLP * 3 *
                                                BVec+j*strideCNLPGrad],
                        &BVec);

                  }
              }
          }
      }


      void
      nlpPsiContraction(
        operatorDFTClass &operatorMatrix,
#ifdef USE_COMPLEX
        const std::vector<dataType::number> &psiQuadsNLP,
#endif
        const std::vector<dataType::number>
          &                                  gradPsiQuadsNLP,
        const std::vector<double> &partialOccupancies,
        const std::vector<dataType::number> &onesVecNLP,
        const dataTypes::number *projectorKetTimesVectorParFlattened,
        const std::vector<unsigned int> &nonTrivialIdToElemIdMap,
        const std::vector<unsigned int>
          &                projecterKetTimesFlattenedVectorLocalIds,
        const unsigned int numCells,
        const unsigned int numQuadsNLP,
        const unsigned int numPsi,
        const unsigned int totalNonTrivialPseudoWfcs,
        const unsigned int innerBlockSizeEnlp,
        std::vector<dataType::number>
          &nlpContractionContribution,
        std::vector<dataType::number> &
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlock,
        dataTypes::number *
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened,
#ifdef USE_COMPLEX
        std::vector<dataType::number> &
          projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock,
        dataTypes::number
          *projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened,
#endif
          )
      {
        const unsigned int blockSizeNlp    = innerBlockSizeEnlp;
        const unsigned int numberBlocksNlp = totalNonTrivialPseudoWfcs / blockSizeNlp;
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

               for (unsigned int ipseudowfc=0;ipseudowfc<currentBlockSize; ipseudowfc++)
                  for (unsigned int iquad=0;iquad<numQuadsNLP*3; iquad++)
                     for (unsigned int iwfc=0;iwfc<numPsi; iwfc++)
                       nlpContractionContributionD[ipseudowfc*numQuadsNLP*3*numPsi+iquad*numPsi+iwfc]=complexConj(gradPsiQuadsNLP[nonTrivialIdToElemIdMapD[ipseudowfc]*numQuadsNLP*3*numPsi+iquad*numPsi+iwfc])*projectorKetTimesVectorParFlattened[projecterKetTimesFlattenedVectorLocalIds[ipseudowfc] *numPsi +iwfc];

                unsigned int m=1;
                unsigned int n=currentBlockSizeNlp * numQuadsNLP * 3;
                unsigned int k=numPsi;
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
                  &projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlock[0],
                  &one);


                for (unsigned int i = 0;
                     i < currentBlockSizeNlp * numQuadsNLP * 3;
                     i++)
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH
                    [startingIdNlp * numQuadsNLP * 3 + i] +=
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedHPinnedTemp
                      [i];
#ifdef USE_COMPLEX
               for (unsigned int ipseudowfc=0;ipseudowfc<currentBlockSize; ipseudowfc++)
                  for (unsigned int iquad=0;iquad<numQuadsNLP; iquad++)
                     for (unsigned int iwfc=0;iwfc<numPsi; iwfc++)
                       nlpContractionContributionD[ipseudowfc*numQuadsNLP*numPsi+iquad*numPsi+iwfc]=complexConj(psiQuadsNLP[nonTrivialIdToElemIdMapD[ipseudowfc]*numQuadsNLP*numPsi+iquad*numPsi+iwfc])*projectorKetTimesVectorParFlattened[projecterKetTimesFlattenedVectorLocalIds[ipseudowfc] *numPsi +iwfc];

                n=currentBlockSizeNlp * numQuadsNLP;
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
                  &projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedBlock[0],
                  &one);

                for (unsigned int i = 0; i < currentBlockSizeNlp * numQuadsNLP;
                     i++)
                  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH
                    [startingIdNlp * numQuadsNLP + i] +=
                    projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH
                      [i];
#endif
              }
          }
      }


     void
      computeBlockContribution(
        operatorDFTClass &                   operatorMatrix,
        distributedCPUVec<dataTypes::number> &flattenedArrayBlock,
        distributedCPUVec<dataTypes::number> &projectorKetTimesVector,
        const dataTypes::number *             X,
        const std::vector<double> &    eigenValues,
        const std::vector<double> &    partialOccupancies,
#ifdef USE_COMPLEX
        const double kcoordx,
        const double kcoordy,
        const double kcoordz,
#endif
        const std::vector<double> &                    onesVec,
        const std::vector<dataType::number> &onesVecNLP,
        const std::vector<unsigned int> &nonTrivialIdToElemIdMap,
        const std::vector<unsigned int>
          &                projecterKetTimesFlattenedVectorLocalIds,
        const unsigned int startingVecId,
        const unsigned int M,
        const unsigned int N,
        const unsigned int numPsi,
        const unsigned int numCells,
        const unsigned int numQuads,
        const unsigned int numQuadsNLP,
        const unsigned int numNodesPerElement,
        const unsigned int totalNonTrivialPseudoWfcs,
        std::vector<dataType::number> &psiQuadsFlat,
        std::vector<dataType::number> &gradPsiQuadsXFlat,
        std::vector<dataType::number> &gradPsiQuadsYFlat,
        std::vector<dataType::number> &gradPsiQuadsZFlat,
#ifdef USE_COMPLEX
        std::vector<dataType::number> &psiQuadsNLP,
#endif
        std::vector<dataType::number> &gradPsiQuadsNLPFlat,
        std::vector<double> &eshelbyTensorContributions,
        std::vector<double> &eshelbyTensorQuadValues,
        std::vector<dataType::number>
          &nlpContractionContributionD,
        std::vector<dataType::number> &
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedBlock,
        dataTypes::number *
          projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattened,
#ifdef USE_COMPLEX
        std::vector<dataType::number> &
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
            flattenedArrayBlock.local_element(iNode * numPsi +
                                                          iWave) =
              eigenVectorsFlattened[iNode * totalNumberWaveFunctions +
                                    startingVecId + iWave];

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
#ifdef USE_COMPLEX
                                                   kcoordx,
                                                   kcoordy,
                                                   kcoordz,
#endif
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
              cudaFlattenedArrayBlock.begin(),
              projectorKetTimesVectorD,
              numPsi);

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
                  projectorKetTimesVector.begin(),
                  nonTrivialIdToElemIdMap,
                  projecterKetTimesFlattenedVectorLocalIds,
                  numCells,
                  numQuadsNLP,
                  numPsi,
                  totalNonTrivialPseudoWfcs,
                  innerBlockSizeEnlp,
                  nlpContractionContribution,
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedDBlock,
                  projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
#ifdef USE_COMPLEX
                  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedDBlock,
                  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH
#endif
                  );
              }


          }
      }
      
    }

    void
    wfcContractionsForceKernelsAllH(
      operatorDFTClass &                                 operatorMatrix,
      const std::vector<std::vector<dataTypes::number>> &X,
      const unsigned int                                 spinPolarizedFlag,
      const unsigned int                                 spinIndex,
      const std::vector<std::vector<double>> &           eigenValuesH,
      const std::vector<std::vector<double>> &           partialOccupanciesH,
      const std::vector<double> &                        kPointCoordinates,
      const unsigned int *nonTrivialIdToElemIdMapH,
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


      distributedCPUVec<dataTypes::number> &flattenedArrayBlock;
  

      distributedCPUVec<dataTypes::number> &projectorKetTimesVector=operatorMatrix.getParallelProjectorKetTimesBlockVector();


      std::vector<double> elocWfcEshelbyTensorQuadValuesH(numCells * numQuads *
                                                            9,
                                                          0.0);

      std::vector<double>            onesVecH(blockSize, 1.0);
      std::vector<dataTypes::number> onesVecHNLP(blockSize,
                                                 dataTypes::number(1.0));

      const unsigned int cellsBlockSize = std::min((unsigned int)10, numCells);

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

      const unsigned numKPoints = kPointCoordinates.size() / 3;
      for (unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
        {
          std::fill::fill(elocWfcEshelbyTensorQuadValuesH.begin(),
                       elocWfcEshelbyTensorQuadValuesH.end(),
                       0.);
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

            for (unsigned int jvec = 0; jvec <N;
                 jvec += blockSize)
              {
                const unsigned int currentBlockSize =
                  std::min(blockSize, N - jvec);

                if (currentBlockSize != BVec || jvec == 0)
                  operatorMatrix.reinit(currentBlockSize,
                                        flattenedArrayBlock,
                                        true);

                if ((jvec + currentBlockSize) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + currentBlockSize) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {

                  std::vector<double> blockedEigenValues(currentBlockSize, 0.0);
                  std::vector<double> blockedPartialOccupancies(currentBlockSize, 0.0);
                  for (unsigned int iWave = 0; iWave < currentBlockSize; ++iWave)
                    {
                      blockedEigenValues[iWave] =
                        eigenValuesH[kPoint][spinIndex * N + ivec + iWave];
                      blockedPartialOccupancies[iWave] =
                        partialOccupanciesH[kPoint]
                                           [spinIndex * N + ivec + iWave];
                    }


                  computeBlockContribution(
                    operatorMatrix,
                    flattenedArrayBlock,
                    projectorKetTimesVector,
                    &X[(1 + spinPolarizedFlag) * kPoint + spinIndex][0],
                    blockedEigenValues,
                    blockedPartialOccupancies,
#ifdef USE_COMPLEX
                    kcoordx,
                    kcoordy,
                    kcoordz,
#endif
                    onesVecH,
                    onesVecHNLP,
                    nonTrivialIdToElemIdMapH,
                    projecterKetTimesFlattenedVectorLocalIdsH,
                    ivec,
                    M,
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

        } // k point loop

    }

  } // namespace force
} // namespace dftfe
