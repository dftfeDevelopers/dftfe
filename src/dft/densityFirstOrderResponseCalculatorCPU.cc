// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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
#include <densityFirstOrderResponseCalculator.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <vectorUtilities.h>
#include <linearAlgebraOperations.h>
#include <DataTypeOverloads.h>

namespace dftfe
{
  template <typename T>
  void
  computeRhoFirstOrderResponseCPU(
    const T *                                      X,
    const T *                                      XPrime,
    const std::vector<std::vector<double>> &       densityMatDerFermiEnergy,
    const unsigned int                             totalNumWaveFunctions,
    const unsigned int                             numLocalDofs,
    operatorDFTClass &                             operatorMatrix,
    const unsigned int                             matrixFreeDofhandlerIndex,
    const dealii::DoFHandler<3> &                  dofHandler,
    const unsigned int                             totalLocallyOwnedCells,
    const unsigned int                             numNodesPerElement,
    const unsigned int                             numQuadPoints,
    const std::vector<double> &                    kPointWeights,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesHam,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesFermiEnergy,
    std::map<dealii::CellId, std::vector<double>>
      &rhoResponseValuesHamSpinPolarized,
    std::map<dealii::CellId, std::vector<double>>
      &                  rhoResponseValuesFermiEnergySpinPolarized,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams)
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
    std::vector<T> wfcPrimeQuads(numQuadPoints * BVec, T(0.0));

    std::vector<T>     shapeFunctionValues(numQuadPoints * numNodesPerElement,
                                       T(0.0));
    const unsigned int numQuadPointsTimes3 = numQuadPoints * 3;

    for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
      for (unsigned int iNode = 0; iNode < numNodesPerElement; ++iNode)
        shapeFunctionValues[iquad * numNodesPerElement + iNode] =
          T(operatorMatrix.getShapeFunctionValuesDensityGaussLobattoQuad()
              [iquad * numNodesPerElement + iNode]);


    dftfe::distributedCPUVec<T> flattenedArrayBlock1, flattenedArrayBlock2;

    std::vector<T> cellWaveFunctionMatrix(numNodesPerElement * BVec, T(0.0));

    std::vector<T> cellWaveFunctionPrimeMatrix(numNodesPerElement * BVec,
                                               T(0.0));

    // set density to zero
    typename dealii::DoFHandler<3>::active_cell_iterator cell =
      dofHandler.begin_active();
    typename dealii::DoFHandler<3>::active_cell_iterator endc =
      dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          const dealii::CellId cellid = cell->id();

          std::fill((rhoResponseValuesHam)[cellid].begin(),
                    (rhoResponseValuesHam)[cellid].end(),
                    0.0);
          std::fill((rhoResponseValuesFermiEnergy)[cellid].begin(),
                    (rhoResponseValuesFermiEnergy)[cellid].end(),
                    0.0);

          if (dftParams.spinPolarized == 1)
            {
              std::fill((rhoResponseValuesHamSpinPolarized)[cellid].begin(),
                        (rhoResponseValuesHamSpinPolarized)[cellid].end(),
                        0.0);
              std::fill(
                (rhoResponseValuesFermiEnergySpinPolarized)[cellid].begin(),
                (rhoResponseValuesFermiEnergySpinPolarized)[cellid].end(),
                0.0);
            }
        }


    std::vector<double> rhoResponseHam(totalLocallyOwnedCells * numQuadPoints,
                                       0.0);
    std::vector<double> rhoResponseFermiEnergy(totalLocallyOwnedCells *
                                                 numQuadPoints,
                                               0.0);

    std::vector<double> rhoResponseHamSpinPolarized(totalLocallyOwnedCells *
                                                      numQuadPoints * 2,
                                                    0.0);
    std::vector<double> rhoResponseFermiEnergySpinPolarized(
      totalLocallyOwnedCells * numQuadPoints * 2, 0.0);

    for (unsigned int spinIndex = 0; spinIndex < (1 + dftParams.spinPolarized);
         ++spinIndex)
      {
        std::vector<double> rhoResponseContributionHam(totalLocallyOwnedCells *
                                                         numQuadPoints,
                                                       0.0);
        std::vector<double> rhoResponseContributionFermiEnergy(
          totalLocallyOwnedCells * numQuadPoints, 0.0);

        for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
          {
            const T *XCurrentKPoint =
              X + ((dftParams.spinPolarized + 1) * kPoint + spinIndex) *
                    numLocalDofs * totalNumWaveFunctions;

            const T *XPrimeCurrentKPoint =
              XPrime + ((dftParams.spinPolarized + 1) * kPoint + spinIndex) *
                         numLocalDofs * totalNumWaveFunctions;

            const std::vector<double> &densityMatDerFermiEnergyVec =
              densityMatDerFermiEnergy[(dftParams.spinPolarized + 1) * kPoint +
                                       spinIndex];

            for (unsigned int jvec = 0; jvec < totalNumWaveFunctions;
                 jvec += BVec)
              {
                const unsigned int currentBlockSize =
                  std::min(BVec, totalNumWaveFunctions - jvec);

                if (currentBlockSize != BVec || jvec == 0)
                  {
                    operatorMatrix.reinit(currentBlockSize,
                                          flattenedArrayBlock1,
                                          true);
                    flattenedArrayBlock2.reinit(flattenedArrayBlock1);
                  }

                if ((jvec + currentBlockSize) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + currentBlockSize) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                      for (unsigned int iWave = 0; iWave < currentBlockSize;
                           ++iWave)
                        flattenedArrayBlock1.local_element(
                          iNode * currentBlockSize + iWave) =
                          XCurrentKPoint[iNode * totalNumWaveFunctions + jvec +
                                         iWave];


                    (operatorMatrix.getOverloadedConstraintMatrix())
                      ->distribute(flattenedArrayBlock1, currentBlockSize);


                    for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                      for (unsigned int iWave = 0; iWave < currentBlockSize;
                           ++iWave)
                        flattenedArrayBlock2.local_element(
                          iNode * currentBlockSize + iWave) =
                          XPrimeCurrentKPoint[iNode * totalNumWaveFunctions +
                                              jvec + iWave];


                    (operatorMatrix.getOverloadedConstraintMatrix())
                      ->distribute(flattenedArrayBlock2, currentBlockSize);


                    for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
                      {
                        const unsigned int inc = 1;
                        for (unsigned int iNode = 0; iNode < numNodesPerElement;
                             ++iNode)
                          {
                            xcopy(
                              &currentBlockSize,
                              flattenedArrayBlock1.begin() +
                                operatorMatrix
                                  .getFlattenedArrayCellLocalProcIndexIdMap()
                                    [icell * numNodesPerElement + iNode],
                              &inc,
                              &cellWaveFunctionMatrix[currentBlockSize * iNode],
                              &inc);

                            xcopy(
                              &currentBlockSize,
                              flattenedArrayBlock2.begin() +
                                operatorMatrix
                                  .getFlattenedArrayCellLocalProcIndexIdMap()
                                    [icell * numNodesPerElement + iNode],
                              &inc,
                              &cellWaveFunctionPrimeMatrix[currentBlockSize *
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

                        xgemm(&transA,
                              &transB,
                              &currentBlockSize,
                              &numQuadPoints,
                              &numNodesPerElement,
                              &scalarCoeffAlpha,
                              &cellWaveFunctionPrimeMatrix[0],
                              &currentBlockSize,
                              &shapeFunctionValues[0],
                              &numNodesPerElement,
                              &scalarCoeffBeta,
                              &wfcPrimeQuads[0],
                              &currentBlockSize);

                        for (unsigned int iquad = 0; iquad < numQuadPoints;
                             ++iquad)
                          for (unsigned int iWave = 0; iWave < currentBlockSize;
                               ++iWave)
                            {
                              rhoResponseContributionHam[icell * numQuadPoints +
                                                         iquad] +=
                                kPointWeights[kPoint] * spinPolarizedFactor *
                                dftfe::utils::realPart(
                                  wfcQuads[iquad * currentBlockSize + iWave] *
                                  dftfe::utils::complexConj(
                                    wfcPrimeQuads[iquad * currentBlockSize +
                                                  iWave]));

                              rhoResponseContributionFermiEnergy
                                [icell * numQuadPoints + iquad] +=
                                kPointWeights[kPoint] * spinPolarizedFactor *
                                densityMatDerFermiEnergyVec[jvec + iWave] *
                                dftfe::utils::realPart(
                                  wfcQuads[iquad * currentBlockSize + iWave] *
                                  dftfe::utils::complexConj(
                                    wfcQuads[iquad * currentBlockSize +
                                             iWave]));
                            }

                      } // cells loop
                  }     // band parallelizatoin check
              }         // wave function block loop
          }             // kpoint loop


        for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
          for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
            {
              rhoResponseHam[icell * numQuadPoints + iquad] +=
                rhoResponseContributionHam[icell * numQuadPoints + iquad];
              rhoResponseFermiEnergy[icell * numQuadPoints + iquad] +=
                rhoResponseContributionFermiEnergy[icell * numQuadPoints +
                                                   iquad];
            }

        if (dftParams.spinPolarized == 1)
          {
            for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
              for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
                {
                  rhoResponseHamSpinPolarized[icell * numQuadPoints * 2 +
                                              2 * iquad + spinIndex] =
                    rhoResponseContributionHam[icell * numQuadPoints + iquad];
                  rhoResponseFermiEnergySpinPolarized[icell * numQuadPoints *
                                                        2 +
                                                      2 * iquad + spinIndex] =
                    rhoResponseContributionFermiEnergy[icell * numQuadPoints +
                                                       iquad];
                }
          }
      } // spin index loop

    // gather density response from all inter communicators
    dealii::Utilities::MPI::sum(rhoResponseHam,
                                interBandGroupComm,
                                rhoResponseHam);

    dealii::Utilities::MPI::sum(rhoResponseHam, interpoolcomm, rhoResponseHam);

    dealii::Utilities::MPI::sum(rhoResponseFermiEnergy,
                                interBandGroupComm,
                                rhoResponseFermiEnergy);

    dealii::Utilities::MPI::sum(rhoResponseFermiEnergy,
                                interpoolcomm,
                                rhoResponseFermiEnergy);

    if (dftParams.spinPolarized == 1)
      {
        dealii::Utilities::MPI::sum(rhoResponseHamSpinPolarized,
                                    interBandGroupComm,
                                    rhoResponseHamSpinPolarized);

        dealii::Utilities::MPI::sum(rhoResponseHamSpinPolarized,
                                    interpoolcomm,
                                    rhoResponseHamSpinPolarized);

        dealii::Utilities::MPI::sum(rhoResponseFermiEnergySpinPolarized,
                                    interBandGroupComm,
                                    rhoResponseFermiEnergySpinPolarized);

        dealii::Utilities::MPI::sum(rhoResponseFermiEnergySpinPolarized,
                                    interpoolcomm,
                                    rhoResponseFermiEnergySpinPolarized);
      }


    unsigned int iElem = 0;
    cell               = dofHandler.begin_active();
    endc               = dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          const dealii::CellId cellid = cell->id();

          std::vector<double> &temp1Quads = (rhoResponseValuesHam)[cellid];

          std::vector<double> &temp2Quads =
            (rhoResponseValuesFermiEnergy)[cellid];

          for (unsigned int q = 0; q < numQuadPoints; ++q)
            {
              temp1Quads[q] = rhoResponseHam[iElem * numQuadPoints + q];
              temp2Quads[q] = rhoResponseFermiEnergy[iElem * numQuadPoints + q];
            }

          if (dftParams.spinPolarized == 1)
            {
              std::vector<double> &temp3Quads =
                (rhoResponseValuesHamSpinPolarized)[cellid];

              std::vector<double> &temp4Quads =
                (rhoResponseValuesFermiEnergySpinPolarized)[cellid];

              for (unsigned int q = 0; q < numQuadPoints; ++q)
                {
                  temp3Quads[2 * q + 0] =
                    rhoResponseHamSpinPolarized[iElem * numQuadPoints * 2 +
                                                2 * q + 0];
                  temp3Quads[2 * q + 1] =
                    rhoResponseHamSpinPolarized[iElem * numQuadPoints * 2 +
                                                2 * q + 1];
                  temp4Quads[2 * q + 0] =
                    rhoResponseFermiEnergySpinPolarized[iElem * numQuadPoints *
                                                          2 +
                                                        2 * q + 0];
                  temp4Quads[2 * q + 1] =
                    rhoResponseFermiEnergySpinPolarized[iElem * numQuadPoints *
                                                          2 +
                                                        2 * q + 1];
                }
            }

          iElem++;
        }


    MPI_Barrier(mpiCommParent);
    cpu_time = MPI_Wtime() - cpu_time;

    if (this_process == 0 && dftParams.verbosity >= 2)
      std::cout << "Time for compute rhoprime on CPU: " << cpu_time
                << std::endl;
  }


  template <typename T, typename TLowPrec>
  void
  computeRhoFirstOrderResponseCPUMixedPrec(
    const T *                                      X,
    const T *                                      XPrime,
    const std::vector<std::vector<double>> &       densityMatDerFermiEnergy,
    const unsigned int                             totalNumWaveFunctions,
    const unsigned int                             numLocalDofs,
    operatorDFTClass &                             operatorMatrix,
    const unsigned int                             matrixFreeDofhandlerIndex,
    const dealii::DoFHandler<3> &                  dofHandler,
    const unsigned int                             totalLocallyOwnedCells,
    const unsigned int                             numNodesPerElement,
    const unsigned int                             numQuadPoints,
    const std::vector<double> &                    kPointWeights,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesHam,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesFermiEnergy,
    std::map<dealii::CellId, std::vector<double>>
      &rhoResponseValuesHamSpinPolarized,
    std::map<dealii::CellId, std::vector<double>>
      &                  rhoResponseValuesFermiEnergySpinPolarized,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams)
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

    std::vector<TLowPrec> wfcQuads(numQuadPoints * BVec, TLowPrec(0.0));
    std::vector<TLowPrec> wfcPrimeQuads(numQuadPoints * BVec, TLowPrec(0.0));

    std::vector<TLowPrec> shapeFunctionValues(numQuadPoints *
                                                numNodesPerElement,
                                              TLowPrec(0.0));
    const unsigned int    numQuadPointsTimes3 = numQuadPoints * 3;

    for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
      for (unsigned int iNode = 0; iNode < numNodesPerElement; ++iNode)
        shapeFunctionValues[iquad * numNodesPerElement + iNode] = TLowPrec(
          operatorMatrix.getShapeFunctionValuesDensityGaussLobattoQuad()
            [iquad * numNodesPerElement + iNode]);


    dftfe::distributedCPUVec<T> flattenedArrayBlock1, flattenedArrayBlock2;

    std::vector<TLowPrec> cellWaveFunctionMatrix(numNodesPerElement * BVec,
                                                 TLowPrec(0.0));

    std::vector<TLowPrec> cellWaveFunctionPrimeMatrix(numNodesPerElement * BVec,
                                                      TLowPrec(0.0));

    // set density to zero
    typename dealii::DoFHandler<3>::active_cell_iterator cell =
      dofHandler.begin_active();
    typename dealii::DoFHandler<3>::active_cell_iterator endc =
      dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          const dealii::CellId cellid = cell->id();

          std::fill((rhoResponseValuesHam)[cellid].begin(),
                    (rhoResponseValuesHam)[cellid].end(),
                    0.0);
          std::fill((rhoResponseValuesFermiEnergy)[cellid].begin(),
                    (rhoResponseValuesFermiEnergy)[cellid].end(),
                    0.0);

          if (dftParams.spinPolarized == 1)
            {
              std::fill((rhoResponseValuesHamSpinPolarized)[cellid].begin(),
                        (rhoResponseValuesHamSpinPolarized)[cellid].end(),
                        0.0);
              std::fill(
                (rhoResponseValuesFermiEnergySpinPolarized)[cellid].begin(),
                (rhoResponseValuesFermiEnergySpinPolarized)[cellid].end(),
                0.0);
            }
        }


    std::vector<double> rhoResponseHam(totalLocallyOwnedCells * numQuadPoints,
                                       0.0);
    std::vector<double> rhoResponseFermiEnergy(totalLocallyOwnedCells *
                                                 numQuadPoints,
                                               0.0);

    std::vector<double> rhoResponseHamSpinPolarized(totalLocallyOwnedCells *
                                                      numQuadPoints * 2,
                                                    0.0);
    std::vector<double> rhoResponseFermiEnergySpinPolarized(
      totalLocallyOwnedCells * numQuadPoints * 2, 0.0);

    const std::vector<dealii::types::global_dof_index> &indexMap =
      operatorMatrix.getFlattenedArrayCellLocalProcIndexIdMap();

    for (unsigned int spinIndex = 0; spinIndex < (1 + dftParams.spinPolarized);
         ++spinIndex)
      {
        std::vector<double> rhoResponseContributionHam(totalLocallyOwnedCells *
                                                         numQuadPoints,
                                                       0.0);
        std::vector<double> rhoResponseContributionFermiEnergy(
          totalLocallyOwnedCells * numQuadPoints, 0.0);

        for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
          {
            const T *XCurrentKPoint =
              X + ((dftParams.spinPolarized + 1) * kPoint + spinIndex) *
                    numLocalDofs * totalNumWaveFunctions;

            const T *XPrimeCurrentKPoint =
              XPrime + ((dftParams.spinPolarized + 1) * kPoint + spinIndex) *
                         numLocalDofs * totalNumWaveFunctions;

            const std::vector<double> &densityMatDerFermiEnergyVec =
              densityMatDerFermiEnergy[(dftParams.spinPolarized + 1) * kPoint +
                                       spinIndex];

            for (unsigned int jvec = 0; jvec < totalNumWaveFunctions;
                 jvec += BVec)
              {
                const unsigned int currentBlockSize =
                  std::min(BVec, totalNumWaveFunctions - jvec);

                if (currentBlockSize != BVec || jvec == 0)
                  {
                    operatorMatrix.reinit(currentBlockSize,
                                          flattenedArrayBlock1,
                                          true);
                    flattenedArrayBlock2.reinit(flattenedArrayBlock1);
                  }

                if ((jvec + currentBlockSize) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + currentBlockSize) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                      for (unsigned int iWave = 0; iWave < currentBlockSize;
                           ++iWave)
                        flattenedArrayBlock1.local_element(
                          iNode * currentBlockSize + iWave) =
                          XCurrentKPoint[iNode * totalNumWaveFunctions + jvec +
                                         iWave];


                    (operatorMatrix.getOverloadedConstraintMatrix())
                      ->distribute(flattenedArrayBlock1, currentBlockSize);


                    for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                      for (unsigned int iWave = 0; iWave < currentBlockSize;
                           ++iWave)
                        flattenedArrayBlock2.local_element(
                          iNode * currentBlockSize + iWave) =
                          XPrimeCurrentKPoint[iNode * totalNumWaveFunctions +
                                              jvec + iWave];


                    (operatorMatrix.getOverloadedConstraintMatrix())
                      ->distribute(flattenedArrayBlock2, currentBlockSize);


                    for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
                      {
                        for (unsigned int iNode = 0; iNode < numNodesPerElement;
                             ++iNode)
                          {
                            const unsigned startIndex =
                              indexMap[icell * numNodesPerElement + iNode];
                            for (unsigned int iwave = 0;
                                 iwave < currentBlockSize;
                                 iwave++)
                              {
                                cellWaveFunctionMatrix[currentBlockSize *
                                                         iNode +
                                                       iwave] =
                                  *(flattenedArrayBlock1.begin() + startIndex +
                                    iwave);
                              }

                            for (unsigned int iwave = 0;
                                 iwave < currentBlockSize;
                                 iwave++)
                              {
                                cellWaveFunctionPrimeMatrix[currentBlockSize *
                                                              iNode +
                                                            iwave] =
                                  *(flattenedArrayBlock2.begin() + startIndex +
                                    iwave);
                              }
                          }


                        const TLowPrec scalarCoeffAlpha = TLowPrec(1.0),
                                       scalarCoeffBeta  = TLowPrec(0.0);
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

                        xgemm(&transA,
                              &transB,
                              &currentBlockSize,
                              &numQuadPoints,
                              &numNodesPerElement,
                              &scalarCoeffAlpha,
                              &cellWaveFunctionPrimeMatrix[0],
                              &currentBlockSize,
                              &shapeFunctionValues[0],
                              &numNodesPerElement,
                              &scalarCoeffBeta,
                              &wfcPrimeQuads[0],
                              &currentBlockSize);

                        for (unsigned int iquad = 0; iquad < numQuadPoints;
                             ++iquad)
                          for (unsigned int iWave = 0; iWave < currentBlockSize;
                               ++iWave)
                            {
                              rhoResponseContributionHam[icell * numQuadPoints +
                                                         iquad] +=
                                kPointWeights[kPoint] * spinPolarizedFactor *
                                dftfe::utils::realPart(
                                  wfcQuads[iquad * currentBlockSize + iWave] *
                                  dftfe::utils::complexConj(
                                    wfcPrimeQuads[iquad * currentBlockSize +
                                                  iWave]));

                              rhoResponseContributionFermiEnergy
                                [icell * numQuadPoints + iquad] +=
                                kPointWeights[kPoint] * spinPolarizedFactor *
                                densityMatDerFermiEnergyVec[jvec + iWave] *
                                dftfe::utils::realPart(
                                  wfcQuads[iquad * currentBlockSize + iWave] *
                                  dftfe::utils::complexConj(
                                    wfcQuads[iquad * currentBlockSize +
                                             iWave]));
                            }

                      } // cells loop
                  }     // band parallelizatoin check
              }         // wave function block loop
          }             // kpoint loop


        for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
          for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
            {
              rhoResponseHam[icell * numQuadPoints + iquad] +=
                rhoResponseContributionHam[icell * numQuadPoints + iquad];
              rhoResponseFermiEnergy[icell * numQuadPoints + iquad] +=
                rhoResponseContributionFermiEnergy[icell * numQuadPoints +
                                                   iquad];
            }

        if (dftParams.spinPolarized == 1)
          {
            for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
              for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
                {
                  rhoResponseHamSpinPolarized[icell * numQuadPoints * 2 +
                                              2 * iquad + spinIndex] =
                    rhoResponseContributionHam[icell * numQuadPoints + iquad];
                  rhoResponseFermiEnergySpinPolarized[icell * numQuadPoints *
                                                        2 +
                                                      2 * iquad + spinIndex] =
                    rhoResponseContributionFermiEnergy[icell * numQuadPoints +
                                                       iquad];
                }
          }
      } // spin index loop

    // gather density response from all inter communicators
    dealii::Utilities::MPI::sum(rhoResponseHam,
                                interBandGroupComm,
                                rhoResponseHam);

    dealii::Utilities::MPI::sum(rhoResponseHam, interpoolcomm, rhoResponseHam);

    dealii::Utilities::MPI::sum(rhoResponseFermiEnergy,
                                interBandGroupComm,
                                rhoResponseFermiEnergy);

    dealii::Utilities::MPI::sum(rhoResponseFermiEnergy,
                                interpoolcomm,
                                rhoResponseFermiEnergy);

    if (dftParams.spinPolarized == 1)
      {
        dealii::Utilities::MPI::sum(rhoResponseHamSpinPolarized,
                                    interBandGroupComm,
                                    rhoResponseHamSpinPolarized);

        dealii::Utilities::MPI::sum(rhoResponseHamSpinPolarized,
                                    interpoolcomm,
                                    rhoResponseHamSpinPolarized);

        dealii::Utilities::MPI::sum(rhoResponseFermiEnergySpinPolarized,
                                    interBandGroupComm,
                                    rhoResponseFermiEnergySpinPolarized);

        dealii::Utilities::MPI::sum(rhoResponseFermiEnergySpinPolarized,
                                    interpoolcomm,
                                    rhoResponseFermiEnergySpinPolarized);
      }


    unsigned int iElem = 0;
    cell               = dofHandler.begin_active();
    endc               = dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          const dealii::CellId cellid = cell->id();

          std::vector<double> &temp1Quads = (rhoResponseValuesHam)[cellid];

          std::vector<double> &temp2Quads =
            (rhoResponseValuesFermiEnergy)[cellid];

          for (unsigned int q = 0; q < numQuadPoints; ++q)
            {
              temp1Quads[q] = rhoResponseHam[iElem * numQuadPoints + q];
              temp2Quads[q] = rhoResponseFermiEnergy[iElem * numQuadPoints + q];
            }

          if (dftParams.spinPolarized == 1)
            {
              std::vector<double> &temp3Quads =
                (rhoResponseValuesHamSpinPolarized)[cellid];

              std::vector<double> &temp4Quads =
                (rhoResponseValuesFermiEnergySpinPolarized)[cellid];

              for (unsigned int q = 0; q < numQuadPoints; ++q)
                {
                  temp3Quads[2 * q + 0] =
                    rhoResponseHamSpinPolarized[iElem * numQuadPoints * 2 +
                                                2 * q + 0];
                  temp3Quads[2 * q + 1] =
                    rhoResponseHamSpinPolarized[iElem * numQuadPoints * 2 +
                                                2 * q + 1];
                  temp4Quads[2 * q + 0] =
                    rhoResponseFermiEnergySpinPolarized[iElem * numQuadPoints *
                                                          2 +
                                                        2 * q + 0];
                  temp4Quads[2 * q + 1] =
                    rhoResponseFermiEnergySpinPolarized[iElem * numQuadPoints *
                                                          2 +
                                                        2 * q + 1];
                }
            }

          iElem++;
        }


    MPI_Barrier(mpiCommParent);
    cpu_time = MPI_Wtime() - cpu_time;

    if (this_process == 0 && dftParams.verbosity >= 2)
      std::cout << "Time for compute rhoprime on CPU: " << cpu_time
                << std::endl;
  }


  template void
  computeRhoFirstOrderResponseCPU(
    const dataTypes::number *                      X,
    const dataTypes::number *                      XPrime,
    const std::vector<std::vector<double>> &       densityMatDerFermiEnergy,
    const unsigned int                             totalNumWaveFunctions,
    const unsigned int                             numLocalDofs,
    operatorDFTClass &                             operatorMatrix,
    const unsigned int                             matrixFreeDofhandlerIndex,
    const dealii::DoFHandler<3> &                  dofHandler,
    const unsigned int                             totalLocallyOwnedCells,
    const unsigned int                             numberNodesPerElement,
    const unsigned int                             numQuadPoints,
    const std::vector<double> &                    kPointWeights,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesHam,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesFermiEnergy,
    std::map<dealii::CellId, std::vector<double>>
      &rhoResponseValuesHamSpinPolarized,
    std::map<dealii::CellId, std::vector<double>>
      &                  rhoResponseValuesFermiEnergySpinPolarized,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams);

  template void
  computeRhoFirstOrderResponseCPUMixedPrec<dataTypes::number,
                                           dataTypes::numberFP32>(
    const dataTypes::number *                      X,
    const dataTypes::number *                      XPrime,
    const std::vector<std::vector<double>> &       densityMatDerFermiEnergy,
    const unsigned int                             totalNumWaveFunctions,
    const unsigned int                             numLocalDofs,
    operatorDFTClass &                             operatorMatrix,
    const unsigned int                             matrixFreeDofhandlerIndex,
    const dealii::DoFHandler<3> &                  dofHandler,
    const unsigned int                             totalLocallyOwnedCells,
    const unsigned int                             numberNodesPerElement,
    const unsigned int                             numQuadPoints,
    const std::vector<double> &                    kPointWeights,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesHam,
    std::map<dealii::CellId, std::vector<double>> &rhoResponseValuesFermiEnergy,
    std::map<dealii::CellId, std::vector<double>>
      &rhoResponseValuesHamSpinPolarized,
    std::map<dealii::CellId, std::vector<double>>
      &                  rhoResponseValuesFermiEnergySpinPolarized,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams);

} // namespace dftfe
