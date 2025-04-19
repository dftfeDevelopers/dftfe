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

// source file for electron density related computations
#include <constants.h>
#include <densityCalculator.h>
#include <dftUtils.h>
#include <vectorUtilities.h>
#include <MemoryStorage.h>


namespace dftfe
{
  template <typename NumberType, dftfe::utils::MemorySpace memorySpace>
  void
  computeRhoFromPSI(
    const dftfe::utils::MemoryStorage<NumberType, memorySpace> *X,
    const dftfe::uInt                       totalNumWaveFunctions,
    const std::vector<std::vector<double>> &partialOccupancies,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<NumberType, double, memorySpace>>
      &basisOperationsPtr,
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
                              &BLASWrapperPtr,
    const dftfe::uInt          matrixFreeDofhandlerIndex,
    const dftfe::uInt          quadratureIndex,
    const std::vector<double> &kPointCoords,
    const std::vector<double> &kPointWeights,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &densityValues,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &gradDensityValues,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
                        &tauValues,
    const bool           isEvaluateGradRho,
    const bool           isEvaluateTau,
    const MPI_Comm      &mpiCommParent,
    const MPI_Comm      &interpoolcomm,
    const MPI_Comm      &interBandGroupComm,
    const dftParameters &dftParams)
  {
    int this_process;
    MPI_Comm_rank(mpiCommParent, &this_process);
#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(mpiCommParent);
    double            computeRho_time        = MPI_Wtime();
    const dftfe::uInt numKPoints             = kPointWeights.size();
    const dftfe::uInt numLocalDofs           = basisOperationsPtr->nOwnedDofs();
    const dftfe::uInt totalLocallyOwnedCells = basisOperationsPtr->nCells();
    const dftfe::uInt numNodesPerElement = basisOperationsPtr->nDofsPerCell();
    // band group parallelization data structures
    const dftfe::uInt numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
    const dftfe::uInt bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
    std::vector<dftfe::uInt> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(interBandGroupComm,
                                               totalNumWaveFunctions,
                                               bandGroupLowHighPlusOneIndices);

    const dftfe::uInt BVec =
      std::min(dftParams.chebyWfcBlockSize, bandGroupLowHighPlusOneIndices[1]);

    const double spinPolarizedFactor =
      (dftParams.spinPolarized == 1) ? 1.0 : 2.0;
    const dftfe::uInt numSpinComponents =
      (dftParams.spinPolarized == 1) ? 2 : 1;

    const NumberType zero                    = 0;
    const NumberType scalarCoeffAlphaRho     = 1.0;
    const NumberType scalarCoeffBetaRho      = 1.0;
    const NumberType scalarCoeffAlphaGradRho = 1.0;
    const NumberType scalarCoeffBetaGradRho  = 1.0;

    const dftfe::uInt cellsBlockSize =
      memorySpace == dftfe::utils::MemorySpace::DEVICE ? 50 : 1;
    const dftfe::uInt numCellBlocks = totalLocallyOwnedCells / cellsBlockSize;
    const dftfe::uInt remCellBlockSize =
      totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;
    basisOperationsPtr->reinit(BVec, cellsBlockSize, quadratureIndex);
    const dftfe::uInt numQuadPoints = basisOperationsPtr->nQuadsPerCell();

    dftfe::utils::MemoryStorage<NumberType, memorySpace> wfcQuadPointData;
    dftfe::utils::MemoryStorage<NumberType, memorySpace> gradWfcQuadPointData;
    dftfe::utils::MemoryStorage<double, memorySpace>     rhoWfcContributions;
    dftfe::utils::MemoryStorage<double, memorySpace>     tauWfcContributions;
    dftfe::utils::MemoryStorage<double, memorySpace> gradRhoWfcContributions;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      rhoHost;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      gradRhoHost;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      tauHost;
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, memorySpace> rho;
    dftfe::utils::MemoryStorage<double, memorySpace> gradRho;
    dftfe::utils::MemoryStorage<double, memorySpace> tau;
#else
    auto &rho             = rhoHost;
    auto &gradRho         = gradRhoHost;
    auto &tau             = tauHost;
#endif

    rho.resize(totalLocallyOwnedCells * numQuadPoints * numSpinComponents, 0.0);
    wfcQuadPointData.resize(cellsBlockSize * numQuadPoints * BVec, zero);

    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      rhoWfcContributions.resize(cellsBlockSize * numQuadPoints * BVec, 0.0);
    if (isEvaluateGradRho)
      {
        gradRho.resize(totalLocallyOwnedCells * numQuadPoints * 3 *
                         numSpinComponents,
                       0.0);
        gradWfcQuadPointData.resize(cellsBlockSize * numQuadPoints * BVec * 3,
                                    zero);
        if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          gradRhoWfcContributions.resize(cellsBlockSize * numQuadPoints * BVec *
                                           3,
                                         0.0);
      }

    if (isEvaluateTau)
      {
        tau.resize(totalLocallyOwnedCells * numQuadPoints * numSpinComponents,
                   0.0);
        if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          tauWfcContributions.resize(cellsBlockSize * numQuadPoints * BVec,
                                     0.0);
      }

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      partialOccupVecHost(BVec, 0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      kCoordHost(3, 0.0);
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, memorySpace> partialOccupVec(
      partialOccupVecHost.size());
    dftfe::utils::MemoryStorage<double, memorySpace> kCoord(kCoordHost.size());
#else
    auto &partialOccupVec = partialOccupVecHost;
    auto &kCoord          = kCoordHost;
#endif

    dftfe::linearAlgebra::MultiVector<NumberType, memorySpace>
      *flattenedArrayBlock;

    for (dftfe::uInt kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
      {
        kCoordHost[0] = kPointCoords[3 * kPoint + 0];
        kCoordHost[1] = kPointCoords[3 * kPoint + 1];
        kCoordHost[2] = kPointCoords[3 * kPoint + 2];

        for (dftfe::uInt spinIndex = 0; spinIndex < numSpinComponents;
             ++spinIndex)
          {
            wfcQuadPointData.setValue(zero);
            gradWfcQuadPointData.setValue(zero);
            rhoWfcContributions.setValue(0.0);
            gradRhoWfcContributions.setValue(0.0);
            tauWfcContributions.setValue(0.0);
            for (dftfe::uInt jvec = 0; jvec < totalNumWaveFunctions;
                 jvec += BVec)
              {
                const dftfe::uInt currentBlockSize =
                  std::min(BVec, totalNumWaveFunctions - jvec);
                flattenedArrayBlock =
                  &(basisOperationsPtr->getMultiVector(currentBlockSize, 0));

                if ((jvec + currentBlockSize) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + currentBlockSize) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    for (dftfe::uInt iEigenVec = 0;
                         iEigenVec < currentBlockSize;
                         ++iEigenVec)
                      *(partialOccupVecHost.begin() + iEigenVec) =
                        partialOccupancies[kPoint]
                                          [totalNumWaveFunctions * spinIndex +
                                           jvec + iEigenVec] *
                        kPointWeights[kPoint] * spinPolarizedFactor;

#if defined(DFTFE_WITH_DEVICE)
                    partialOccupVec.copyFrom(partialOccupVecHost);
                    kCoord.copyFrom(kCoordHost);
#endif
                    if (memorySpace == dftfe::utils::MemorySpace::HOST)
                      for (dftfe::uInt iNode = 0; iNode < numLocalDofs; ++iNode)
                        std::memcpy(flattenedArrayBlock->data() +
                                      iNode * currentBlockSize,
                                    X->data() +
                                      numLocalDofs * totalNumWaveFunctions *
                                        (numSpinComponents * kPoint +
                                         spinIndex) +
                                      iNode * totalNumWaveFunctions + jvec,
                                    currentBlockSize * sizeof(NumberType));
#if defined(DFTFE_WITH_DEVICE)
                    else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
                      BLASWrapperPtr->stridedCopyToBlockConstantStride(
                        currentBlockSize,
                        totalNumWaveFunctions,
                        numLocalDofs,
                        jvec,
                        X->data() + numLocalDofs * totalNumWaveFunctions *
                                      (numSpinComponents * kPoint + spinIndex),
                        flattenedArrayBlock->data());
#endif

                    basisOperationsPtr->reinit(currentBlockSize,
                                               cellsBlockSize,
                                               quadratureIndex,
                                               false);


                    flattenedArrayBlock->updateGhostValues();
                    basisOperationsPtr->distribute(*(flattenedArrayBlock));

                    for (dftfe::Int iblock = 0; iblock < (numCellBlocks + 1);
                         iblock++)
                      {
                        const dftfe::uInt currentCellsBlockSize =
                          (iblock == numCellBlocks) ? remCellBlockSize :
                                                      cellsBlockSize;
                        if (currentCellsBlockSize > 0)
                          {
                            const dftfe::uInt startingCellId =
                              iblock * cellsBlockSize;

                            basisOperationsPtr->interpolateKernel(
                              *(flattenedArrayBlock),
                              wfcQuadPointData.data(),
                              isEvaluateGradRho ? gradWfcQuadPointData.data() :
                                                  NULL,
                              std::pair<dftfe::uInt, dftfe::uInt>(
                                startingCellId,
                                startingCellId + currentCellsBlockSize));

                            computeRhoGradRhoFromInterpolatedValues(
                              BLASWrapperPtr,
                              std::pair<dftfe::uInt, dftfe::uInt>(
                                startingCellId,
                                startingCellId + currentCellsBlockSize),
                              std::pair<dftfe::uInt, dftfe::uInt>(
                                jvec, jvec + currentBlockSize),
                              numQuadPoints,
                              partialOccupVec.data(),
                              wfcQuadPointData.data(),
                              gradWfcQuadPointData.data(),
                              rhoWfcContributions.data(),
                              gradRhoWfcContributions.data(),
                              rho.data() + spinIndex * totalLocallyOwnedCells *
                                             numQuadPoints,
                              gradRho.data() + spinIndex *
                                                 totalLocallyOwnedCells *
                                                 numQuadPoints * 3,
                              isEvaluateGradRho);

                            if (isEvaluateTau)
                              {
                                computeTauFromInterpolatedValues(
                                  BLASWrapperPtr,
                                  std::pair<dftfe::uInt, dftfe::uInt>(
                                    startingCellId,
                                    startingCellId + currentCellsBlockSize),
                                  std::pair<dftfe::uInt, dftfe::uInt>(
                                    jvec, jvec + currentBlockSize),
                                  numQuadPoints,
                                  partialOccupVec.data(),
                                  kCoord.data(),
                                  wfcQuadPointData.data(),
                                  gradWfcQuadPointData.data(),
                                  tauWfcContributions.data(),
                                  tau.data() + spinIndex *
                                                 totalLocallyOwnedCells *
                                                 numQuadPoints);
                              }
                          } // non-trivial cell block check
                      }     // cells block loop
                  }
              }
          }
      }
#if defined(DFTFE_WITH_DEVICE)
    rhoHost.resize(rho.size());

    rhoHost.copyFrom(rho);

    if (isEvaluateGradRho)
      {
        gradRhoHost.resize(gradRho.size());
        gradRhoHost.copyFrom(gradRho);
      }
    if (isEvaluateTau)
      {
        tauHost.resize(tau.size());
        tauHost.copyFrom(tau);
      }

#endif

    int size;
    MPI_Comm_size(interpoolcomm, &size);
    if (size > 1)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      rhoHost.data(),
                      totalLocallyOwnedCells * numQuadPoints *
                        numSpinComponents,
                      dataTypes::mpi_type_id(rhoHost.data()),
                      MPI_SUM,
                      interpoolcomm);
        if (isEvaluateGradRho)
          MPI_Allreduce(MPI_IN_PLACE,
                        gradRhoHost.data(),
                        totalLocallyOwnedCells * numQuadPoints *
                          numSpinComponents * 3,
                        dataTypes::mpi_type_id(gradRhoHost.data()),
                        MPI_SUM,
                        interpoolcomm);
        if (isEvaluateTau)
          MPI_Allreduce(MPI_IN_PLACE,
                        tauHost.data(),
                        totalLocallyOwnedCells * numQuadPoints *
                          numSpinComponents,
                        dataTypes::mpi_type_id(tauHost.data()),
                        MPI_SUM,
                        interpoolcomm);
      }

    MPI_Comm_size(interBandGroupComm, &size);
    if (size > 1)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      rhoHost.data(),
                      totalLocallyOwnedCells * numQuadPoints *
                        numSpinComponents,
                      dataTypes::mpi_type_id(rhoHost.data()),
                      MPI_SUM,
                      interBandGroupComm);
        if (isEvaluateGradRho)
          MPI_Allreduce(MPI_IN_PLACE,
                        gradRhoHost.data(),
                        totalLocallyOwnedCells * numQuadPoints *
                          numSpinComponents * 3,
                        dataTypes::mpi_type_id(gradRhoHost.data()),
                        MPI_SUM,
                        interBandGroupComm);

        if (isEvaluateTau)
          MPI_Allreduce(MPI_IN_PLACE,
                        tauHost.data(),
                        totalLocallyOwnedCells * numQuadPoints *
                          numSpinComponents,
                        dataTypes::mpi_type_id(tauHost.data()),
                        MPI_SUM,
                        interBandGroupComm);
      }

    if (dftParams.spinPolarized == 1)
      {
        densityValues[0].resize(totalLocallyOwnedCells * numQuadPoints);
        densityValues[1].resize(totalLocallyOwnedCells * numQuadPoints);
        std::transform(rhoHost.begin(),
                       rhoHost.begin() + totalLocallyOwnedCells * numQuadPoints,
                       rhoHost.begin() + totalLocallyOwnedCells * numQuadPoints,
                       densityValues[0].begin(),
                       std::plus<>{});
        std::transform(rhoHost.begin(),
                       rhoHost.begin() + totalLocallyOwnedCells * numQuadPoints,
                       rhoHost.begin() + totalLocallyOwnedCells * numQuadPoints,
                       densityValues[1].begin(),
                       std::minus<>{});
        if (isEvaluateGradRho)
          {
            gradDensityValues[0].resize(3 * totalLocallyOwnedCells *
                                        numQuadPoints);
            gradDensityValues[1].resize(3 * totalLocallyOwnedCells *
                                        numQuadPoints);
            std::transform(gradRhoHost.begin(),
                           gradRhoHost.begin() +
                             3 * totalLocallyOwnedCells * numQuadPoints,
                           gradRhoHost.begin() +
                             3 * totalLocallyOwnedCells * numQuadPoints,
                           gradDensityValues[0].begin(),
                           std::plus<>{});
            std::transform(gradRhoHost.begin(),
                           gradRhoHost.begin() +
                             3 * totalLocallyOwnedCells * numQuadPoints,
                           gradRhoHost.begin() +
                             3 * totalLocallyOwnedCells * numQuadPoints,
                           gradDensityValues[1].begin(),
                           std::minus<>{});
          }

        if (isEvaluateTau)
          {
            tauValues[0].resize(totalLocallyOwnedCells * numQuadPoints);
            tauValues[1].resize(totalLocallyOwnedCells * numQuadPoints);
            std::transform(tauHost.begin(),
                           tauHost.begin() +
                             totalLocallyOwnedCells * numQuadPoints,
                           tauHost.begin() +
                             totalLocallyOwnedCells * numQuadPoints,
                           tauValues[0].begin(),
                           std::plus<>{});
            std::transform(tauHost.begin(),
                           tauHost.begin() +
                             totalLocallyOwnedCells * numQuadPoints,
                           tauHost.begin() +
                             totalLocallyOwnedCells * numQuadPoints,
                           tauValues[1].begin(),
                           std::minus<>{});
          }
      }
    else
      {
        densityValues[0] = rhoHost;
        if (isEvaluateGradRho)
          gradDensityValues[0] = gradRhoHost;
        if (isEvaluateTau)
          {
            tauValues[0] = tauHost;
          }
      }

#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(mpiCommParent);
    computeRho_time = MPI_Wtime() - computeRho_time;

    if (this_process == 0 && dftParams.verbosity >= 2)
      if (memorySpace == dftfe::utils::MemorySpace::HOST)
        std::cout << "Time for compute rho on CPU: " << computeRho_time
                  << std::endl;
      else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
        std::cout << "Time for compute rho on Device: " << computeRho_time
                  << std::endl;
  }


  template <typename NumberType>
  void
  computeRhoGradRhoFromInterpolatedValues(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                             &BLASWrapperPtr,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
    const std::pair<dftfe::uInt, dftfe::uInt> vecRange,
    const dftfe::uInt                         nQuadsPerCell,
    double                                   *partialOccupVec,
    NumberType                               *wfcQuadPointData,
    NumberType                               *gradWfcQuadPointData,
    double                                   *rhoCellsWfcContributions,
    double                                   *gradRhoCellsWfcContributions,
    double                                   *rho,
    double                                   *gradRho,
    const bool                                isEvaluateGradRho)
  {
    const dftfe::uInt cellsBlockSize   = cellRange.second - cellRange.first;
    const dftfe::uInt vectorsBlockSize = vecRange.second - vecRange.first;
    for (dftfe::uInt iCell = cellRange.first; iCell < cellRange.second; ++iCell)
      for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
        for (dftfe::uInt iWave = 0; iWave < vecRange.second - vecRange.first;
             ++iWave)
          {
            const NumberType psi =
              wfcQuadPointData[(iCell - cellRange.first) * nQuadsPerCell *
                                 vectorsBlockSize +
                               iQuad * vectorsBlockSize + iWave];
            rho[iCell * nQuadsPerCell + iQuad] +=
              partialOccupVec[iWave] * std::abs(psi) * std::abs(psi);
            if (isEvaluateGradRho)
              {
                gradRho[iCell * nQuadsPerCell * 3 + 3 * iQuad] +=
                  2 * partialOccupVec[iWave] *
                  dftfe::utils::realPart(
                    dftfe::utils::complexConj(psi) *
                    gradWfcQuadPointData[(iCell - cellRange.first) *
                                           nQuadsPerCell * vectorsBlockSize *
                                           3 +
                                         iQuad * vectorsBlockSize + iWave]);
                gradRho[iCell * nQuadsPerCell * 3 + 3 * iQuad + 1] +=
                  2 * partialOccupVec[iWave] *
                  dftfe::utils::realPart(
                    dftfe::utils::complexConj(psi) *
                    gradWfcQuadPointData[(iCell - cellRange.first) *
                                           nQuadsPerCell * vectorsBlockSize *
                                           3 +
                                         nQuadsPerCell * vectorsBlockSize +
                                         iQuad * vectorsBlockSize + iWave]);
                gradRho[iCell * nQuadsPerCell * 3 + 3 * iQuad + 2] +=
                  2 * partialOccupVec[iWave] *
                  dftfe::utils::realPart(
                    dftfe::utils::complexConj(psi) *
                    gradWfcQuadPointData[(iCell - cellRange.first) *
                                           nQuadsPerCell * vectorsBlockSize *
                                           3 +
                                         2 * nQuadsPerCell * vectorsBlockSize +
                                         iQuad * vectorsBlockSize + iWave]);
              }
          }
  }

  template <typename NumberType>
  void
  computeTauFromInterpolatedValues(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                                             &BLASWrapperPtr,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
    const std::pair<dftfe::uInt, dftfe::uInt> vecRange,
    const dftfe::uInt                         nQuadsPerCell,
    double                                   *partialOccupVec,
    double                                   *kCoord,
    NumberType                               *wfcQuadPointData,
    NumberType                               *gradWfcQuadPointData,
    double *kineticEnergyDensityCellsWfcContributions,
    double *tau)
  {
    const dftfe::uInt cellsBlockSize   = cellRange.second - cellRange.first;
    const dftfe::uInt vectorsBlockSize = vecRange.second - vecRange.first;

    const double kPointCoordSq =
      kCoord[0] * kCoord[0] + kCoord[1] * kCoord[1] + kCoord[2] * kCoord[2];
    for (dftfe::uInt iCell = cellRange.first; iCell < cellRange.second; ++iCell)
      for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
        for (dftfe::uInt iWave = 0; iWave < vecRange.second - vecRange.first;
             ++iWave)
          {
            NumberType dirValGradPsi;
            double     sumDirValGradPsi = 0.0;
            NumberType tempImag         = 0.0;

            const NumberType psi =
              wfcQuadPointData[(iCell - cellRange.first) * nQuadsPerCell *
                                 vectorsBlockSize +
                               iQuad * vectorsBlockSize + iWave];
            for (dftfe::Int dirIdx = 0; dirIdx < 3; ++dirIdx)
              {
                dirValGradPsi =
                  gradWfcQuadPointData[(iCell - cellRange.first) *
                                         nQuadsPerCell * vectorsBlockSize * 3 +
                                       dirIdx * nQuadsPerCell *
                                         vectorsBlockSize +
                                       iQuad * vectorsBlockSize + iWave];
                sumDirValGradPsi +=
                  std::abs(dirValGradPsi) * std::abs(dirValGradPsi);

                tempImag += kCoord[dirIdx] * dirValGradPsi;
              }

            tau[iCell * nQuadsPerCell + iQuad] +=
              0.5 * partialOccupVec[iWave] * sumDirValGradPsi;

            if (std::is_same<dftfe::dataTypes::number,
                             std::complex<double>>::value)
              {
                tau[iCell * nQuadsPerCell + iQuad] +=
                  0.5 * partialOccupVec[iWave] * kPointCoordSq * std::abs(psi) *
                  std::abs(psi);

                tau[iCell * nQuadsPerCell + iQuad] +=
                  partialOccupVec[iWave] *
                  dftfe::utils::imagPart(tempImag *
                                         dftfe::utils::complexConj(psi));
              }
          }
  }

#if defined(DFTFE_WITH_DEVICE)
  template void
  computeRhoFromPSI(
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::DEVICE> *X,
    const dftfe::uInt                       totalNumWaveFunctions,
    const std::vector<std::vector<double>> &partialOccupancies,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::DEVICE>>
      &basisOperationsPtrDevice,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                              &BLASWrapperPtr,
    const dftfe::uInt          matrixFreeDofhandlerIndex,
    const dftfe::uInt          quadratureIndex,
    const std::vector<double> &kPointCoords,
    const std::vector<double> &kPointWeights,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &densityValues,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &gradDensityValues,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
                        &tauValues,
    const bool           isEvaluateGradRho,
    const bool           isEvaluateTau,
    const MPI_Comm      &mpiCommParent,
    const MPI_Comm      &interpoolcomm,
    const MPI_Comm      &interBandGroupComm,
    const dftParameters &dftParams);

#endif

  template void
  computeRhoFromPSI(
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::HOST> *X,
    const dftfe::uInt                       totalNumWaveFunctions,
    const std::vector<std::vector<double>> &partialOccupancies,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      &basisOperationsPtr,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
                              &BLASWrapperPtr,
    const dftfe::uInt          matrixFreeDofhandlerIndex,
    const dftfe::uInt          quadratureIndex,
    const std::vector<double> &kPointCoords,
    const std::vector<double> &kPointWeights,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &densityValues,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &gradDensityValues,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
                        &tauValues,
    const bool           isEvaluateGradRho,
    const bool           isEvaluateTau,
    const MPI_Comm      &mpiCommParent,
    const MPI_Comm      &interpoolcomm,
    const MPI_Comm      &interBandGroupComm,
    const dftParameters &dftParams);

} // namespace dftfe
