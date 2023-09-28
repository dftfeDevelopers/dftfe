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

// source file for electron density related computations
#include <constants.h>
#include <densityCalculatorCPU.h>
#include <dftUtils.h>
#include <vectorUtilities.h>
#include <linearAlgebraOperations.h>
#include <DataTypeOverloads.h>

namespace dftfe
{
  template <typename T>
  void
  computeRhoFromPSICPU(
    const std::vector<std::vector<T>> &            X,
    const std::vector<std::vector<T>> &            XFrac,
    const unsigned int                             totalNumWaveFunctions,
    const unsigned int                             Nfr,
    const unsigned int                             numLocalDofs,
    const std::vector<std::vector<double>> &       eigenValues,
    const double                                   fermiEnergy,
    const double                                   fermiEnergyUp,
    const double                                   fermiEnergyDown,
    operatorDFTClass &                             operatorMatrix,
    const dealii::DoFHandler<3> &                  dofHandler,
    const unsigned int                             totalLocallyOwnedCells,
    const unsigned int                             numNodesPerElement,
    const unsigned int                             numQuadPoints,
    const std::vector<double> &                    kPointWeights,
    std::map<dealii::CellId, std::vector<double>> *rhoValues,
    std::map<dealii::CellId, std::vector<double>> *gradRhoValues,
    std::map<dealii::CellId, std::vector<double>> *rhoValuesSpinPolarized,
    std::map<dealii::CellId, std::vector<double>> *gradRhoValuesSpinPolarized,
    const bool                                     isEvaluateGradRho,
    const MPI_Comm &                               mpiCommParent,
    const MPI_Comm &                               interpoolcomm,
    const MPI_Comm &                               interBandGroupComm,
    const dftParameters &                          dftParams,
    const bool                                     spectrumSplit,
    const bool                                     useFEOrderRhoPlusOneGLQuad)
  {
    int this_process;
    MPI_Comm_rank(mpiCommParent, &this_process);
    MPI_Barrier(mpiCommParent);
    double cpu_time = MPI_Wtime();

    // band group parallelization data structures
    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(interBandGroupComm,
                                               totalNumWaveFunctions,
                                               bandGroupLowHighPlusOneIndices);

    const unsigned int BVec =
      std::min(dftParams.chebyWfcBlockSize, bandGroupLowHighPlusOneIndices[1]);

    const double spinPolarizedFactor =
      (dftParams.spinPolarized == 1) ? 1.0 : 2.0;


    std::vector<T> wfcQuads(numQuadPoints * BVec, T(0.0));

    std::vector<T> gradWfcQuads(numQuadPoints * 3 * BVec, T(0.0));

    std::vector<T>     shapeFunctionValues(numQuadPoints * numNodesPerElement,
                                       T(0.0));
    std::vector<T>     shapeFunctionGradValues(numQuadPoints * 3 *
                                             numNodesPerElement,
                                           T(0.0));
    const unsigned int numQuadPointsTimes3 = numQuadPoints * 3;

    if (useFEOrderRhoPlusOneGLQuad)
      {
        for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
          for (unsigned int iNode = 0; iNode < numNodesPerElement; ++iNode)
            shapeFunctionValues[iquad * numNodesPerElement + iNode] =
              T(operatorMatrix.getShapeFunctionValuesDensityGaussLobattoQuad()
                  [iquad * numNodesPerElement + iNode]);
      }
    else
      {
        for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
          for (unsigned int iNode = 0; iNode < numNodesPerElement; ++iNode)
            shapeFunctionValues[iquad * numNodesPerElement + iNode] =
              T(operatorMatrix.getShapeFunctionValuesDensityGaussQuad()
                  [iquad * numNodesPerElement + iNode]);
      }

    std::vector<double> partialOccupVecTimesKptWeight(BVec, 0.0);


    dftfe::distributedCPUMultiVec<T> flattenedArrayBlock;

    std::vector<T> cellWaveFunctionMatrix(numNodesPerElement * BVec, T(0.0));


    // set density to zero
    typename dealii::DoFHandler<3>::active_cell_iterator cell =
      dofHandler.begin_active();
    typename dealii::DoFHandler<3>::active_cell_iterator endc =
      dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          const dealii::CellId cellid = cell->id();


          std::fill((*rhoValues)[cellid].begin(),
                    (*rhoValues)[cellid].end(),
                    0.0);
          if (isEvaluateGradRho)
            std::fill((*gradRhoValues)[cellid].begin(),
                      (*gradRhoValues)[cellid].end(),
                      0.0);

          if (dftParams.spinPolarized == 1)
            {
              std::fill((*rhoValuesSpinPolarized)[cellid].begin(),
                        (*rhoValuesSpinPolarized)[cellid].end(),
                        0.0);
              if (isEvaluateGradRho)
                std::fill((*gradRhoValuesSpinPolarized)[cellid].begin(),
                          (*gradRhoValuesSpinPolarized)[cellid].end(),
                          0.0);
            }
        }

    std::vector<double> rhoValuesFlattened(totalLocallyOwnedCells *
                                             numQuadPoints,
                                           0.0);
    std::vector<double> gradRhoValuesFlattened(totalLocallyOwnedCells *
                                                 numQuadPoints * 3,
                                               0.0);
    std::vector<double> rhoValuesSpinPolarizedFlattened(totalLocallyOwnedCells *
                                                          numQuadPoints * 2,
                                                        0.0);
    std::vector<double> gradRhoValuesSpinPolarizedFlattened(
      totalLocallyOwnedCells * numQuadPoints * 6, 0.0);


    for (unsigned int spinIndex = 0; spinIndex < (1 + dftParams.spinPolarized);
         ++spinIndex)
      {
        for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
          {
            std::vector<double> rhoContribution(totalLocallyOwnedCells *
                                                  numQuadPoints,
                                                0.0);

            std::vector<double> gradRhoXContribution(
              isEvaluateGradRho ? (totalLocallyOwnedCells * numQuadPoints) : 1,
              0.0);
            std::vector<double> gradRhoYContribution(
              isEvaluateGradRho ? (totalLocallyOwnedCells * numQuadPoints) : 1,
              0.0);
            std::vector<double> gradRhoZContribution(
              isEvaluateGradRho ? (totalLocallyOwnedCells * numQuadPoints) : 1,
              0.0);

            const std::vector<T> &XCurrentKPoint =
              X[(dftParams.spinPolarized + 1) * kPoint + spinIndex];
            const std::vector<T> &XFracCurrentKPoint =
              XFrac[(dftParams.spinPolarized + 1) * kPoint + spinIndex];

            for (unsigned int jvec = 0; jvec < totalNumWaveFunctions;
                 jvec += BVec)
              {
                const unsigned int currentBlockSize =
                  std::min(BVec, totalNumWaveFunctions - jvec);

                if (currentBlockSize != BVec || jvec == 0)
                  operatorMatrix.reinit(currentBlockSize,
                                        flattenedArrayBlock,
                                        true);

                if ((jvec + currentBlockSize) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + currentBlockSize) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    if (spectrumSplit)
                      {
                        std::fill(partialOccupVecTimesKptWeight.begin(),
                                  partialOccupVecTimesKptWeight.end(),
                                  kPointWeights[kPoint] * spinPolarizedFactor);
                      }
                    else
                      {
                        if (dftParams.constraintMagnetization)
                          {
                            const double fermiEnergyConstraintMag =
                              spinIndex == 0 ? fermiEnergyUp : fermiEnergyDown;
                            for (unsigned int iEigenVec = 0;
                                 iEigenVec < currentBlockSize;
                                 ++iEigenVec)
                              {
                                if (eigenValues[kPoint][totalNumWaveFunctions *
                                                          spinIndex +
                                                        jvec + iEigenVec] >
                                    fermiEnergyConstraintMag)
                                  partialOccupVecTimesKptWeight[iEigenVec] =
                                    0.0;
                                else
                                  partialOccupVecTimesKptWeight[iEigenVec] =
                                    kPointWeights[kPoint] * spinPolarizedFactor;
                              }
                          }
                        else
                          {
                            for (unsigned int iEigenVec = 0;
                                 iEigenVec < currentBlockSize;
                                 ++iEigenVec)
                              {
                                partialOccupVecTimesKptWeight[iEigenVec] =
                                  dftUtils::getPartialOccupancy(
                                    eigenValues[kPoint][totalNumWaveFunctions *
                                                          spinIndex +
                                                        jvec + iEigenVec],
                                    fermiEnergy,
                                    C_kb,
                                    dftParams.TVal) *
                                  kPointWeights[kPoint] * spinPolarizedFactor;
                              }
                          }
                      }


                    for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                      for (unsigned int iWave = 0; iWave < currentBlockSize;
                           ++iWave)
                        flattenedArrayBlock
                          .data()[iNode * currentBlockSize + iWave] =
                          XCurrentKPoint[iNode * totalNumWaveFunctions + jvec +
                                         iWave];


                    (operatorMatrix.getOverloadedConstraintMatrix())
                      ->distribute(flattenedArrayBlock, currentBlockSize);

                    for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
                      {
                        const unsigned int inc = 1;
                        for (unsigned int iNode = 0; iNode < numNodesPerElement;
                             ++iNode)
                          {
                            xcopy(
                              &currentBlockSize,
                              flattenedArrayBlock.data() +
                                operatorMatrix
                                  .getFlattenedArrayCellLocalProcIndexIdMap()
                                    [icell * numNodesPerElement + iNode],
                              &inc,
                              &cellWaveFunctionMatrix[currentBlockSize * iNode],
                              &inc);
                          }


                        const T scalarCoeffAlpha = T(1.0),
                                scalarCoeffBeta  = T(0.0);
                        const char transA = 'N', transB = 'N';

                        xgemm(&transA,
                              &transB,
                              &currentBlockSize,
                              &numQuadPoints,
                              &numNodesPerElement,
                              &scalarCoeffAlpha,
                              &cellWaveFunctionMatrix[0],
                              &currentBlockSize,
                              &shapeFunctionValues[0],
                              &numNodesPerElement,
                              &scalarCoeffBeta,
                              &wfcQuads[0],
                              &currentBlockSize);

                        for (unsigned int iquad = 0; iquad < numQuadPoints;
                             ++iquad)
                          for (unsigned int iWave = 0; iWave < currentBlockSize;
                               ++iWave)
                            rhoContribution[icell * numQuadPoints + iquad] +=
                              partialOccupVecTimesKptWeight[iWave] *
                              std::abs(
                                wfcQuads[iquad * currentBlockSize + iWave]) *
                              std::abs(
                                wfcQuads[iquad * currentBlockSize + iWave]);

                        if (isEvaluateGradRho)
                          {
                            for (unsigned int i = 0;
                                 i < numNodesPerElement * 3 * numQuadPoints;
                                 ++i)
                              {
                                shapeFunctionGradValues[i] = T(
                                  operatorMatrix
                                    .getShapeFunctionGradValuesDensityGaussQuad()
                                      [icell * numNodesPerElement * 3 *
                                         numQuadPoints +
                                       i]);
                              }

                            xgemm(&transA,
                                  &transB,
                                  &currentBlockSize,
                                  &numQuadPointsTimes3,
                                  &numNodesPerElement,
                                  &scalarCoeffAlpha,
                                  &cellWaveFunctionMatrix[0],
                                  &currentBlockSize,
                                  &shapeFunctionGradValues[0],
                                  &numNodesPerElement,
                                  &scalarCoeffBeta,
                                  &gradWfcQuads[0],
                                  &currentBlockSize);

                            for (unsigned int iquad = 0; iquad < numQuadPoints;
                                 ++iquad)
                              for (unsigned int iWave = 0;
                                   iWave < currentBlockSize;
                                   ++iWave)
                                {
                                  const T wfcQuadVal =
                                    dftfe::utils::complexConj(
                                      wfcQuads[iquad * currentBlockSize +
                                               iWave]);
                                  const T temp1 =
                                    wfcQuadVal *
                                    gradWfcQuads[iquad * 3 * currentBlockSize +
                                                 iWave];
                                  gradRhoXContribution[icell * numQuadPoints +
                                                       iquad] +=
                                    2.0 * partialOccupVecTimesKptWeight[iWave] *
                                    dftfe::utils::realPart(temp1);
                                }

                            for (unsigned int iquad = 0; iquad < numQuadPoints;
                                 ++iquad)
                              for (unsigned int iWave = 0;
                                   iWave < currentBlockSize;
                                   ++iWave)
                                {
                                  const T wfcQuadVal =
                                    dftfe::utils::complexConj(
                                      wfcQuads[iquad * currentBlockSize +
                                               iWave]);
                                  const T temp1 =
                                    wfcQuadVal *
                                    gradWfcQuads[iquad * 3 * currentBlockSize +
                                                 currentBlockSize + iWave];
                                  gradRhoYContribution[icell * numQuadPoints +
                                                       iquad] +=
                                    2.0 * partialOccupVecTimesKptWeight[iWave] *
                                    dftfe::utils::realPart(temp1);
                                }

                            for (unsigned int iquad = 0; iquad < numQuadPoints;
                                 ++iquad)
                              for (unsigned int iWave = 0;
                                   iWave < currentBlockSize;
                                   ++iWave)
                                {
                                  const T wfcQuadVal =
                                    dftfe::utils::complexConj(
                                      wfcQuads[iquad * currentBlockSize +
                                               iWave]);
                                  const T temp1 =
                                    wfcQuadVal *
                                    gradWfcQuads[iquad * 3 * currentBlockSize +
                                                 2 * currentBlockSize + iWave];
                                  gradRhoZContribution[icell * numQuadPoints +
                                                       iquad] +=
                                    2.0 * partialOccupVecTimesKptWeight[iWave] *
                                    dftfe::utils::realPart(temp1);
                                }
                          }

                      } // cells loop
                  }     // band parallelizatoin check
              }         // wave function block loop

            if (spectrumSplit)
              for (unsigned int jvec = 0; jvec < Nfr; jvec += BVec)
                {
                  const unsigned int currentBlockSize =
                    std::min(BVec, Nfr - jvec);

                  if (currentBlockSize != BVec || jvec == 0)
                    operatorMatrix.reinit(currentBlockSize,
                                          flattenedArrayBlock,
                                          true);

                  if ((jvec + totalNumWaveFunctions - Nfr + currentBlockSize) <=
                        bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId +
                                                       1] &&
                      (jvec + totalNumWaveFunctions - Nfr + currentBlockSize) >
                        bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                    {
                      if (dftParams.constraintMagnetization)
                        {
                          const double fermiEnergyConstraintMag =
                            spinIndex == 0 ? fermiEnergyUp : fermiEnergyDown;
                          for (unsigned int iEigenVec = 0;
                               iEigenVec < currentBlockSize;
                               ++iEigenVec)
                            {
                              if (eigenValues[kPoint]
                                             [totalNumWaveFunctions *
                                                spinIndex +
                                              (totalNumWaveFunctions - Nfr) +
                                              jvec + iEigenVec] >
                                  fermiEnergyConstraintMag)
                                partialOccupVecTimesKptWeight[iEigenVec] =
                                  -kPointWeights[kPoint] * spinPolarizedFactor;
                              else
                                partialOccupVecTimesKptWeight[iEigenVec] = 0.0;
                            }
                        }
                      else
                        {
                          for (unsigned int iEigenVec = 0;
                               iEigenVec < currentBlockSize;
                               ++iEigenVec)
                            {
                              partialOccupVecTimesKptWeight[iEigenVec] =
                                (dftUtils::getPartialOccupancy(
                                   eigenValues[kPoint]
                                              [totalNumWaveFunctions *
                                                 spinIndex +
                                               (totalNumWaveFunctions - Nfr) +
                                               jvec + iEigenVec],
                                   fermiEnergy,
                                   C_kb,
                                   dftParams.TVal) -
                                 1.0) *
                                kPointWeights[kPoint] * spinPolarizedFactor;
                            }
                        }


                      for (unsigned int iNode = 0; iNode < numLocalDofs;
                           ++iNode)
                        for (unsigned int iWave = 0; iWave < currentBlockSize;
                             ++iWave)
                          flattenedArrayBlock
                            .data()[iNode * currentBlockSize + iWave] =
                            XFracCurrentKPoint[iNode * Nfr + jvec + iWave];

                      (operatorMatrix.getOverloadedConstraintMatrix())
                        ->distribute(flattenedArrayBlock, currentBlockSize);

                      for (int icell = 0; icell < totalLocallyOwnedCells;
                           icell++)
                        {
                          const unsigned int inc = 1;
                          for (unsigned int iNode = 0;
                               iNode < numNodesPerElement;
                               ++iNode)
                            {
                              xcopy(
                                &currentBlockSize,
                                flattenedArrayBlock.data() +
                                  operatorMatrix
                                    .getFlattenedArrayCellLocalProcIndexIdMap()
                                      [icell * numNodesPerElement + iNode],
                                &inc,
                                &cellWaveFunctionMatrix[currentBlockSize *
                                                        iNode],
                                &inc);
                            }


                          const T scalarCoeffAlpha = T(1.0),
                                  scalarCoeffBeta  = T(0.0);
                          const char transA = 'N', transB = 'N';

                          xgemm(&transA,
                                &transB,
                                &currentBlockSize,
                                &numQuadPoints,
                                &numNodesPerElement,
                                &scalarCoeffAlpha,
                                &cellWaveFunctionMatrix[0],
                                &currentBlockSize,
                                &shapeFunctionValues[0],
                                &numNodesPerElement,
                                &scalarCoeffBeta,
                                &wfcQuads[0],
                                &currentBlockSize);

                          for (unsigned int iquad = 0; iquad < numQuadPoints;
                               ++iquad)
                            for (unsigned int iWave = 0;
                                 iWave < currentBlockSize;
                                 ++iWave)
                              rhoContribution[icell * numQuadPoints + iquad] +=
                                partialOccupVecTimesKptWeight[iWave] *
                                std::abs(
                                  wfcQuads[iquad * currentBlockSize + iWave]) *
                                std::abs(
                                  wfcQuads[iquad * currentBlockSize + iWave]);

                          if (isEvaluateGradRho)
                            {
                              for (unsigned int i = 0;
                                   i < numNodesPerElement * 3 * numQuadPoints;
                                   ++i)
                                {
                                  shapeFunctionGradValues[i] = T(
                                    operatorMatrix
                                      .getShapeFunctionGradValuesDensityGaussQuad()
                                        [icell * numNodesPerElement * 3 *
                                           numQuadPoints +
                                         i]);
                                }

                              xgemm(&transA,
                                    &transB,
                                    &currentBlockSize,
                                    &numQuadPointsTimes3,
                                    &numNodesPerElement,
                                    &scalarCoeffAlpha,
                                    &cellWaveFunctionMatrix[0],
                                    &currentBlockSize,
                                    &shapeFunctionGradValues[0],
                                    &numNodesPerElement,
                                    &scalarCoeffBeta,
                                    &gradWfcQuads[0],
                                    &currentBlockSize);

                              for (unsigned int iquad = 0;
                                   iquad < numQuadPoints;
                                   ++iquad)
                                for (unsigned int iWave = 0;
                                     iWave < currentBlockSize;
                                     ++iWave)
                                  {
                                    const T wfcQuadVal =
                                      dftfe::utils::complexConj(
                                        wfcQuads[iquad * currentBlockSize +
                                                 iWave]);
                                    const T temp1 =
                                      wfcQuadVal *
                                      gradWfcQuads[iquad * 3 *
                                                     currentBlockSize +
                                                   iWave];
                                    gradRhoXContribution[icell * numQuadPoints +
                                                         iquad] +=
                                      2.0 *
                                      partialOccupVecTimesKptWeight[iWave] *
                                      dftfe::utils::realPart(temp1);
                                  }

                              for (unsigned int iquad = 0;
                                   iquad < numQuadPoints;
                                   ++iquad)
                                for (unsigned int iWave = 0;
                                     iWave < currentBlockSize;
                                     ++iWave)
                                  {
                                    const T wfcQuadVal =
                                      dftfe::utils::complexConj(
                                        wfcQuads[iquad * currentBlockSize +
                                                 iWave]);
                                    const T temp1 =
                                      wfcQuadVal *
                                      gradWfcQuads[iquad * 3 *
                                                     currentBlockSize +
                                                   currentBlockSize + iWave];
                                    gradRhoYContribution[icell * numQuadPoints +
                                                         iquad] +=
                                      2.0 *
                                      partialOccupVecTimesKptWeight[iWave] *
                                      dftfe::utils::realPart(temp1);
                                  }

                              for (unsigned int iquad = 0;
                                   iquad < numQuadPoints;
                                   ++iquad)
                                for (unsigned int iWave = 0;
                                     iWave < currentBlockSize;
                                     ++iWave)
                                  {
                                    const T wfcQuadVal =
                                      dftfe::utils::complexConj(
                                        wfcQuads[iquad * currentBlockSize +
                                                 iWave]);
                                    const T temp1 =
                                      wfcQuadVal *
                                      gradWfcQuads[iquad * 3 *
                                                     currentBlockSize +
                                                   2 * currentBlockSize +
                                                   iWave];
                                    gradRhoZContribution[icell * numQuadPoints +
                                                         iquad] +=
                                      2.0 *
                                      partialOccupVecTimesKptWeight[iWave] *
                                      dftfe::utils::realPart(temp1);
                                  }
                            }

                        } // cells loop
                    }
                }

            for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
              for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
                {
                  rhoValuesFlattened[icell * numQuadPoints + iquad] +=
                    rhoContribution[icell * numQuadPoints + iquad];
                }

            if (isEvaluateGradRho)
              for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
                for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
                  {
                    gradRhoValuesFlattened[icell * numQuadPoints * 3 +
                                           3 * iquad + 0] +=
                      gradRhoXContribution[icell * numQuadPoints + iquad];
                    gradRhoValuesFlattened[icell * numQuadPoints * 3 +
                                           3 * iquad + 1] +=
                      gradRhoYContribution[icell * numQuadPoints + iquad];
                    gradRhoValuesFlattened[icell * numQuadPoints * 3 +
                                           3 * iquad + 2] +=
                      gradRhoZContribution[icell * numQuadPoints + iquad];
                  }
            if (dftParams.spinPolarized == 1)
              {
                for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
                  for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
                    {
                      rhoValuesSpinPolarizedFlattened[icell * numQuadPoints *
                                                        2 +
                                                      iquad * 2 + spinIndex] +=
                        rhoContribution[icell * numQuadPoints + iquad];
                    }

                if (isEvaluateGradRho)
                  for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
                    for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
                      {
                        gradRhoValuesSpinPolarizedFlattened
                          [icell * numQuadPoints * 6 + iquad * 6 +
                           spinIndex * 3] +=
                          gradRhoXContribution[icell * numQuadPoints + iquad];
                        gradRhoValuesSpinPolarizedFlattened
                          [icell * numQuadPoints * 6 + iquad * 6 +
                           spinIndex * 3 + 1] +=
                          gradRhoYContribution[icell * numQuadPoints + iquad];
                        gradRhoValuesSpinPolarizedFlattened
                          [icell * numQuadPoints * 6 + iquad * 6 +
                           spinIndex * 3 + 2] +=
                          gradRhoZContribution[icell * numQuadPoints + iquad];
                      }
              }

          } // kpoint loop
      }     // spin index loop


    // gather density from all inter communicators
    if (dealii::Utilities::MPI::n_mpi_processes(interpoolcomm) > 1)
      {
        dealii::Utilities::MPI::sum(rhoValuesFlattened,
                                    interpoolcomm,
                                    rhoValuesFlattened);

        if (isEvaluateGradRho)
          dealii::Utilities::MPI::sum(gradRhoValuesFlattened,
                                      interpoolcomm,
                                      gradRhoValuesFlattened);



        if (dftParams.spinPolarized == 1)
          {
            dealii::Utilities::MPI::sum(rhoValuesSpinPolarizedFlattened,
                                        interpoolcomm,
                                        rhoValuesSpinPolarizedFlattened);

            if (isEvaluateGradRho)
              dealii::Utilities::MPI::sum(gradRhoValuesSpinPolarizedFlattened,
                                          interpoolcomm,
                                          gradRhoValuesSpinPolarizedFlattened);
          }
      }

    if (dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm) > 1)
      {
        dealii::Utilities::MPI::sum(rhoValuesFlattened,
                                    interBandGroupComm,
                                    rhoValuesFlattened);

        if (isEvaluateGradRho)
          dealii::Utilities::MPI::sum(gradRhoValuesFlattened,
                                      interBandGroupComm,
                                      gradRhoValuesFlattened);


        if (dftParams.spinPolarized == 1)
          {
            dealii::Utilities::MPI::sum(rhoValuesSpinPolarizedFlattened,
                                        interBandGroupComm,
                                        rhoValuesSpinPolarizedFlattened);

            if (isEvaluateGradRho)
              dealii::Utilities::MPI::sum(gradRhoValuesSpinPolarizedFlattened,
                                          interBandGroupComm,
                                          gradRhoValuesSpinPolarizedFlattened);
          }
      }


    unsigned int iElem = 0;
    cell               = dofHandler.begin_active();
    endc               = dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          const dealii::CellId cellid = cell->id();

          std::vector<double>  dummy(1);
          std::vector<double> &tempRhoQuads = (*rhoValues)[cellid];
          std::vector<double> &tempGradRhoQuads =
            isEvaluateGradRho ? (*gradRhoValues)[cellid] : dummy;

          std::vector<double> &tempRhoQuadsSP =
            (dftParams.spinPolarized == 1) ? (*rhoValuesSpinPolarized)[cellid] :
                                             dummy;
          std::vector<double> &tempGradRhoQuadsSP =
            ((dftParams.spinPolarized == 1) && isEvaluateGradRho) ?
              (*gradRhoValuesSpinPolarized)[cellid] :
              dummy;

          if (dftParams.spinPolarized == 1)
            {
              for (unsigned int q = 0; q < numQuadPoints; ++q)
                {
                  tempRhoQuadsSP[2 * q + 0] =
                    rhoValuesSpinPolarizedFlattened[iElem * numQuadPoints * 2 +
                                                    q * 2 + 0];

                  tempRhoQuadsSP[2 * q + 1] =
                    rhoValuesSpinPolarizedFlattened[iElem * numQuadPoints * 2 +
                                                    q * 2 + 1];
                }

              if (isEvaluateGradRho)
                for (unsigned int q = 0; q < numQuadPoints; ++q)
                  {
                    tempGradRhoQuadsSP[6 * q + 0] =
                      gradRhoValuesSpinPolarizedFlattened[iElem *
                                                            numQuadPoints * 6 +
                                                          6 * q];
                    tempGradRhoQuadsSP[6 * q + 1] =
                      gradRhoValuesSpinPolarizedFlattened[iElem *
                                                            numQuadPoints * 6 +
                                                          6 * q + 1];
                    tempGradRhoQuadsSP[6 * q + 2] =
                      gradRhoValuesSpinPolarizedFlattened[iElem *
                                                            numQuadPoints * 6 +
                                                          6 * q + 2];
                    tempGradRhoQuadsSP[6 * q + 3] =
                      gradRhoValuesSpinPolarizedFlattened[iElem *
                                                            numQuadPoints * 6 +
                                                          6 * q + 3];
                    tempGradRhoQuadsSP[6 * q + 4] =
                      gradRhoValuesSpinPolarizedFlattened[iElem *
                                                            numQuadPoints * 6 +
                                                          6 * q + 4];
                    tempGradRhoQuadsSP[6 * q + 5] =
                      gradRhoValuesSpinPolarizedFlattened[iElem *
                                                            numQuadPoints * 6 +
                                                          6 * q + 5];
                  }
            }

          for (unsigned int q = 0; q < numQuadPoints; ++q)
            tempRhoQuads[q] = rhoValuesFlattened[iElem * numQuadPoints + q];


          if (isEvaluateGradRho)
            for (unsigned int q = 0; q < numQuadPoints; ++q)
              {
                tempGradRhoQuads[3 * q] =
                  gradRhoValuesFlattened[iElem * numQuadPoints * 3 + q * 3];
                tempGradRhoQuads[3 * q + 1] =
                  gradRhoValuesFlattened[iElem * numQuadPoints * 3 + q * 3 + 1];
                tempGradRhoQuads[3 * q + 2] =
                  gradRhoValuesFlattened[iElem * numQuadPoints * 3 + q * 3 + 2];
              }
          iElem++;
        }

    MPI_Barrier(mpiCommParent);
    cpu_time = MPI_Wtime() - cpu_time;

    if (this_process == 0 && dftParams.verbosity >= 2)
      std::cout << "Time for compute rho on CPU: " << cpu_time << std::endl;
  }

  template void
  computeRhoFromPSICPU(
    const std::vector<std::vector<dataTypes::number>> &X,
    const std::vector<std::vector<dataTypes::number>> &XFrac,
    const unsigned int                                 totalNumWaveFunctions,
    const unsigned int                                 Nfr,
    const unsigned int                                 numLocalDofs,
    const std::vector<std::vector<double>> &           eigenValues,
    const double                                       fermiEnergy,
    const double                                       fermiEnergyUp,
    const double                                       fermiEnergyDown,
    operatorDFTClass &                                 operatorMatrix,
    const dealii::DoFHandler<3> &                      dofHandler,
    const unsigned int                                 totalLocallyOwnedCells,
    const unsigned int                                 numNodesPerElement,
    const unsigned int                                 numQuadPoints,
    const std::vector<double> &                        kPointWeights,
    std::map<dealii::CellId, std::vector<double>> *    rhoValues,
    std::map<dealii::CellId, std::vector<double>> *    gradRhoValues,
    std::map<dealii::CellId, std::vector<double>> *    rhoValuesSpinPolarized,
    std::map<dealii::CellId, std::vector<double>> *gradRhoValuesSpinPolarized,
    const bool                                     isEvaluateGradRho,
    const MPI_Comm &                               mpiCommParent,
    const MPI_Comm &                               interpoolcomm,
    const MPI_Comm &                               interBandGroupComm,
    const dftParameters &                          dftParams,
    const bool                                     spectrumSplit,
    const bool                                     useFEOrderRhoPlusOneGLQuad);
} // namespace dftfe
