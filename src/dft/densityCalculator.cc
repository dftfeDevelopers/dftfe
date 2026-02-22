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
    const dftfe::uInt          tempQuadratureIndex,
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
      (dftParams.spinPolarized == 1 || dftParams.noncolin || dftParams.hasSOC) ?
        1.0 :
        2.0;
    const dftfe::uInt numSpinComponents =
      (dftParams.spinPolarized == 1) ? 2 : 1;
    const dftfe::uInt numRhoComponents =
      dftParams.noncolin ? 4 : numSpinComponents;

    const dftfe::uInt numWfnSpinors =
      (dftParams.noncolin || dftParams.hasSOC) ? 2 : 1;

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
    const dftfe::uInt numQuadsQuadratureIndex =
      basisOperationsPtr->d_matrixFreeDataPtr->get_quadrature(quadratureIndex)
        .size();
    const dftfe::uInt numQuadsTempQuadratureIndex =
      basisOperationsPtr->d_matrixFreeDataPtr
        ->get_quadrature(tempQuadratureIndex)
        .size();

    const bool useTempQuadrature =
      numQuadsQuadratureIndex > numQuadsTempQuadratureIndex;
    basisOperationsPtr->reinit(BVec * numWfnSpinors,
                               cellsBlockSize,
                               useTempQuadrature ? tempQuadratureIndex :
                                                   quadratureIndex);
    const dftfe::uInt numQuadPoints = basisOperationsPtr->nQuadsPerCell();

    dftfe::utils::MemoryStorage<NumberType, memorySpace> wfcQuadPointData;
    dftfe::utils::MemoryStorage<NumberType, memorySpace> gradWfcQuadPointData;
    dftfe::utils::MemoryStorage<double, memorySpace>     rhoWfcContributions;
    dftfe::utils::MemoryStorage<double, memorySpace>     tauWfcContributions;
    dftfe::utils::MemoryStorage<double, memorySpace> gradRhoWfcContributions;

    dftfe::utils::MemoryStorage<double, memorySpace> rho;
    dftfe::utils::MemoryStorage<double, memorySpace> gradRho;
    dftfe::utils::MemoryStorage<double, memorySpace> tau;

    rho.resize(totalLocallyOwnedCells * numQuadPoints * numRhoComponents, 0.0);
    wfcQuadPointData.resize(cellsBlockSize * numQuadPoints * BVec *
                              numWfnSpinors,
                            zero);

    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      rhoWfcContributions.resize(cellsBlockSize * numQuadPoints * BVec *
                                   numRhoComponents,
                                 0.0);
    if (isEvaluateGradRho)
      {
        gradRho.resize(totalLocallyOwnedCells * numQuadPoints * 3 *
                         numRhoComponents,
                       0.0);
        gradWfcQuadPointData.resize(cellsBlockSize * numQuadPoints * BVec * 3 *
                                      numWfnSpinors,
                                    zero);
        if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          gradRhoWfcContributions.resize(cellsBlockSize * numQuadPoints * BVec *
                                           3 * numRhoComponents,
                                         0.0);
      }

    if (isEvaluateTau)
      {
        tau.resize(totalLocallyOwnedCells * numQuadPoints * numRhoComponents,
                   0.0);
        if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          tauWfcContributions.resize(cellsBlockSize * numQuadPoints * BVec *
                                       numRhoComponents,
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
                flattenedArrayBlock = &(basisOperationsPtr->getMultiVector(
                  currentBlockSize * numWfnSpinors, 0));

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
                      for (dftfe::uInt iNode = 0;
                           iNode < numLocalDofs * numWfnSpinors;
                           ++iNode)
                        std::memcpy(flattenedArrayBlock->data() +
                                      iNode * currentBlockSize,
                                    X->data() +
                                      numLocalDofs * totalNumWaveFunctions *
                                        numWfnSpinors *
                                        (numSpinComponents * kPoint +
                                         spinIndex) +
                                      iNode * totalNumWaveFunctions + jvec,
                                    currentBlockSize * sizeof(NumberType));
#if defined(DFTFE_WITH_DEVICE)
                    else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
                      BLASWrapperPtr->stridedCopyToBlockConstantStride(
                        currentBlockSize,
                        totalNumWaveFunctions,
                        numLocalDofs * numWfnSpinors,
                        jvec,
                        X->data() + numLocalDofs * numWfnSpinors *
                                      totalNumWaveFunctions *
                                      (numSpinComponents * kPoint + spinIndex),
                        flattenedArrayBlock->data());
#endif

                    basisOperationsPtr->reinit(currentBlockSize * numWfnSpinors,
                                               cellsBlockSize,
                                               useTempQuadrature ?
                                                 tempQuadratureIndex :
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
                              totalLocallyOwnedCells,
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
                              isEvaluateGradRho,
                              dftParams.noncolin,
                              dftParams.hasSOC);

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
                                                 numQuadPoints,
                                  dftParams.noncolin,
                                  dftParams.hasSOC);
                              }
                          } // non-trivial cell block check
                      }     // cells block loop
                  }
              } // wfc loop
          }     // spin loop
      }         // kpt loop


    dftfe::utils::MemoryStorage<double, memorySpace> rhoRefinedStorage;
    dftfe::utils::MemoryStorage<double, memorySpace> gradRhoRefinedStorage;
    dftfe::utils::MemoryStorage<double, memorySpace> tauRefinedStorage;

    if (useTempQuadrature)
      {
        rhoRefinedStorage.resize(totalLocallyOwnedCells *
                                   numQuadsQuadratureIndex * numRhoComponents,
                                 0.0);
        if (isEvaluateGradRho)
          {
            gradRhoRefinedStorage.resize(totalLocallyOwnedCells *
                                           numQuadsQuadratureIndex * 3 *
                                           numRhoComponents,
                                         0.0);
          }
        if (isEvaluateTau)
          {
            tauRefinedStorage.resize(totalLocallyOwnedCells *
                                       numQuadsQuadratureIndex *
                                       numRhoComponents,
                                     0.0);
          }
      }
    auto &rhoRefined     = useTempQuadrature ? rhoRefinedStorage : rho;
    auto &gradRhoRefined = useTempQuadrature ? gradRhoRefinedStorage : gradRho;
    auto &tauRefined     = useTempQuadrature ? tauRefinedStorage : tau;

    if (useTempQuadrature)
      {
        basisOperationsPtr->reinit(BVec, cellsBlockSize, quadratureIndex);
        for (dftfe::uInt spinIndex = 0; spinIndex < numRhoComponents;
             ++spinIndex)
          {
            basisOperationsPtr->interpolateQ1ToQ2(
              rho.data() + spinIndex * totalLocallyOwnedCells * numQuadPoints,
              tempQuadratureIndex,
              quadratureIndex,
              rhoRefined.data() +
                spinIndex * totalLocallyOwnedCells * numQuadsQuadratureIndex,
              1);
            if (isEvaluateGradRho)
              basisOperationsPtr->interpolateQ1ToQ2(
                gradRho.data() +
                  spinIndex * totalLocallyOwnedCells * numQuadPoints * 3,
                tempQuadratureIndex,
                quadratureIndex,
                gradRhoRefined.data() + spinIndex * totalLocallyOwnedCells *
                                          numQuadsQuadratureIndex * 3,
                3);
            if (isEvaluateTau)
              basisOperationsPtr->interpolateQ1ToQ2(
                tau.data() + spinIndex * totalLocallyOwnedCells * numQuadPoints,
                tempQuadratureIndex,
                quadratureIndex,
                tauRefined.data() +
                  spinIndex * totalLocallyOwnedCells * numQuadsQuadratureIndex,
                1);
          }
      }

#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      rhoHost;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      gradRhoHost;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      tauHost;

    rhoHost.resize(rhoRefined.size());
    rhoHost.copyFrom(rhoRefined);

    if (isEvaluateGradRho)
      {
        gradRhoHost.resize(gradRhoRefined.size());
        gradRhoHost.copyFrom(gradRhoRefined);
      }
    if (isEvaluateTau)
      {
        tauHost.resize(tauRefined.size());
        tauHost.copyFrom(tauRefined);
      }
#else
    auto &rhoHost         = rhoRefined;
    auto &gradRhoHost     = gradRhoRefined;
    auto &tauHost         = tauRefined;
#endif

    int size;
    MPI_Comm_size(interpoolcomm, &size);
    if (size > 1)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      rhoHost.data(),
                      rhoHost.size(),
                      dataTypes::mpi_type_id(rhoHost.data()),
                      MPI_SUM,
                      interpoolcomm);
        if (isEvaluateGradRho)
          MPI_Allreduce(MPI_IN_PLACE,
                        gradRhoHost.data(),
                        gradRhoHost.size(),
                        dataTypes::mpi_type_id(gradRhoHost.data()),
                        MPI_SUM,
                        interpoolcomm);
        if (isEvaluateTau)
          MPI_Allreduce(MPI_IN_PLACE,
                        tauHost.data(),
                        tauHost.size(),
                        dataTypes::mpi_type_id(tauHost.data()),
                        MPI_SUM,
                        interpoolcomm);
      }

    MPI_Comm_size(interBandGroupComm, &size);
    if (size > 1)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      rhoHost.data(),
                      rhoHost.size(),
                      dataTypes::mpi_type_id(rhoHost.data()),
                      MPI_SUM,
                      interBandGroupComm);
        if (isEvaluateGradRho)
          MPI_Allreduce(MPI_IN_PLACE,
                        gradRhoHost.data(),
                        gradRhoHost.size(),
                        dataTypes::mpi_type_id(gradRhoHost.data()),
                        MPI_SUM,
                        interBandGroupComm);

        if (isEvaluateTau)
          MPI_Allreduce(MPI_IN_PLACE,
                        tauHost.data(),
                        tauHost.size(),
                        dataTypes::mpi_type_id(tauHost.data()),
                        MPI_SUM,
                        interBandGroupComm);
      }

    if (dftParams.spinPolarized == 1)
      {
        densityValues[0].resize(totalLocallyOwnedCells *
                                numQuadsQuadratureIndex);
        densityValues[1].resize(totalLocallyOwnedCells *
                                numQuadsQuadratureIndex);
        std::transform(rhoHost.begin(),
                       rhoHost.begin() +
                         totalLocallyOwnedCells * numQuadsQuadratureIndex,
                       rhoHost.begin() +
                         totalLocallyOwnedCells * numQuadsQuadratureIndex,
                       densityValues[0].begin(),
                       std::plus<>{});
        std::transform(rhoHost.begin(),
                       rhoHost.begin() +
                         totalLocallyOwnedCells * numQuadsQuadratureIndex,
                       rhoHost.begin() +
                         totalLocallyOwnedCells * numQuadsQuadratureIndex,
                       densityValues[1].begin(),
                       std::minus<>{});
        if (isEvaluateGradRho)
          {
            gradDensityValues[0].resize(3 * totalLocallyOwnedCells *
                                        numQuadsQuadratureIndex);
            gradDensityValues[1].resize(3 * totalLocallyOwnedCells *
                                        numQuadsQuadratureIndex);
            std::transform(gradRhoHost.begin(),
                           gradRhoHost.begin() + 3 * totalLocallyOwnedCells *
                                                   numQuadsQuadratureIndex,
                           gradRhoHost.begin() + 3 * totalLocallyOwnedCells *
                                                   numQuadsQuadratureIndex,
                           gradDensityValues[0].begin(),
                           std::plus<>{});
            std::transform(gradRhoHost.begin(),
                           gradRhoHost.begin() + 3 * totalLocallyOwnedCells *
                                                   numQuadsQuadratureIndex,
                           gradRhoHost.begin() + 3 * totalLocallyOwnedCells *
                                                   numQuadsQuadratureIndex,
                           gradDensityValues[1].begin(),
                           std::minus<>{});
          }

        if (isEvaluateTau)
          {
            tauValues[0].resize(totalLocallyOwnedCells *
                                numQuadsQuadratureIndex);
            tauValues[1].resize(totalLocallyOwnedCells *
                                numQuadsQuadratureIndex);
            std::transform(tauHost.begin(),
                           tauHost.begin() +
                             totalLocallyOwnedCells * numQuadsQuadratureIndex,
                           tauHost.begin() +
                             totalLocallyOwnedCells * numQuadsQuadratureIndex,
                           tauValues[0].begin(),
                           std::plus<>{});
            std::transform(tauHost.begin(),
                           tauHost.begin() +
                             totalLocallyOwnedCells * numQuadsQuadratureIndex,
                           tauHost.begin() +
                             totalLocallyOwnedCells * numQuadsQuadratureIndex,
                           tauValues[1].begin(),
                           std::minus<>{});
          }
      }
    else if (dftParams.noncolin)
      {
        for (dftfe::uInt iComp = 0; iComp < 4; ++iComp)
          {
            densityValues[iComp].resize(totalLocallyOwnedCells * numQuadPoints);
            std::memcpy(densityValues[iComp].begin(),
                        rhoHost.begin() +
                          iComp * totalLocallyOwnedCells * numQuadPoints,
                        totalLocallyOwnedCells * numQuadPoints *
                          sizeof(double));
          }
        if (isEvaluateGradRho)
          {
            for (dftfe::uInt iComp = 0; iComp < 4; ++iComp)
              {
                gradDensityValues[iComp].resize(3 * totalLocallyOwnedCells *
                                                numQuadPoints);
                std::memcpy(gradDensityValues[iComp].begin(),
                            gradRhoHost.begin() + iComp * 3 *
                                                    totalLocallyOwnedCells *
                                                    numQuadPoints,
                            totalLocallyOwnedCells * numQuadPoints * 3 *
                              sizeof(double));
              }
          }
        if (isEvaluateTau)
          for (dftfe::uInt iComp = 0; iComp < 4; ++iComp)
            {
              tauValues[iComp].resize(totalLocallyOwnedCells * numQuadPoints);
              std::memcpy(tauValues[iComp].begin(),
                          tauHost.begin() +
                            iComp * totalLocallyOwnedCells * numQuadPoints,
                          totalLocallyOwnedCells * numQuadPoints *
                            sizeof(double));
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
    const dftfe::uInt                         nCells,
    double                                   *partialOccupVec,
    NumberType                               *wfcQuadPointData,
    NumberType                               *gradWfcQuadPointData,
    double                                   *rhoCellsWfcContributions,
    double                                   *gradRhoCellsWfcContributions,
    double                                   *rho,
    double                                   *gradRho,
    const bool                                isEvaluateGradRho,
    const bool                                isNonCollin,
    const bool                                hasSOC)
  {
    const dftfe::uInt cellsBlockSize   = cellRange.second - cellRange.first;
    const dftfe::uInt vectorsBlockSize = vecRange.second - vecRange.first;
    if (isNonCollin || hasSOC)
      for (dftfe::uInt iCell = cellRange.first; iCell < cellRange.second;
           ++iCell)
        for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
          for (dftfe::uInt iWave = 0; iWave < vecRange.second - vecRange.first;
               ++iWave)
            {
              const NumberType psiUp =
                wfcQuadPointData[(iCell - cellRange.first) * nQuadsPerCell *
                                   vectorsBlockSize * 2 +
                                 iQuad * vectorsBlockSize * 2 + iWave];
              const NumberType psiDown =
                wfcQuadPointData[(iCell - cellRange.first) * nQuadsPerCell *
                                   vectorsBlockSize * 2 +
                                 iQuad * vectorsBlockSize * 2 +
                                 vectorsBlockSize + iWave];
              rho[0 * nCells * nQuadsPerCell + iCell * nQuadsPerCell + iQuad] +=
                partialOccupVec[iWave] *
                (std::abs(psiUp * psiUp) + std::abs(psiDown * psiDown));
              if (isNonCollin)
                {
                  rho[1 * nCells * nQuadsPerCell + iCell * nQuadsPerCell +
                      iQuad] +=
                    partialOccupVec[iWave] *
                    (std::abs(psiUp * psiUp) - std::abs(psiDown * psiDown));
                  rho[2 * nCells * nQuadsPerCell + iCell * nQuadsPerCell +
                      iQuad] += partialOccupVec[iWave] * 2.0 *
                                dftfe::utils::imagPart(
                                  dftfe::utils::complexConj(psiUp) * psiDown);
                  rho[3 * nCells * nQuadsPerCell + iCell * nQuadsPerCell +
                      iQuad] += partialOccupVec[iWave] * 2.0 *
                                dftfe::utils::realPart(
                                  dftfe::utils::complexConj(psiUp) * psiDown);
                }
              if (isEvaluateGradRho)
                {
                  for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                    {
                      const NumberType gradPsiUp = gradWfcQuadPointData
                        [(iCell - cellRange.first) * nQuadsPerCell *
                           vectorsBlockSize * 3 * 2 +
                         iDim * nQuadsPerCell * vectorsBlockSize * 2 +
                         iQuad * vectorsBlockSize * 2 + iWave];
                      const NumberType gradPsiDown =
                        gradWfcQuadPointData[(iCell - cellRange.first) *
                                               nQuadsPerCell *
                                               vectorsBlockSize * 3 * 2 +
                                             iDim * nQuadsPerCell *
                                               vectorsBlockSize * 2 +
                                             iQuad * vectorsBlockSize * 2 +
                                             vectorsBlockSize + iWave];
                      gradRho[0 * nCells * nQuadsPerCell * 3 +
                              iCell * nQuadsPerCell * 3 + 3 * iQuad + iDim] +=
                        2.0 * partialOccupVec[iWave] *
                        dftfe::utils::realPart(
                          dftfe::utils::complexConj(psiUp) * gradPsiUp +
                          dftfe::utils::complexConj(psiDown) * gradPsiDown);
                      if (isNonCollin)
                        {
                          gradRho[1 * nCells * nQuadsPerCell * 3 +
                                  iCell * nQuadsPerCell * 3 + 3 * iQuad +
                                  iDim] +=
                            2.0 * partialOccupVec[iWave] *
                            dftfe::utils::realPart(
                              dftfe::utils::complexConj(psiUp) * gradPsiUp -
                              dftfe::utils::complexConj(psiDown) * gradPsiDown);
                          gradRho[2 * nCells * nQuadsPerCell * 3 +
                                  iCell * nQuadsPerCell * 3 + 3 * iQuad +
                                  iDim] +=
                            2.0 * partialOccupVec[iWave] *
                            dftfe::utils::imagPart(
                              dftfe::utils::complexConj(gradPsiUp) * psiDown +
                              dftfe::utils::complexConj(psiUp) * gradPsiDown);
                          gradRho[3 * nCells * nQuadsPerCell * 3 +
                                  iCell * nQuadsPerCell * 3 + 3 * iQuad +
                                  iDim] +=
                            2.0 * partialOccupVec[iWave] *
                            dftfe::utils::realPart(
                              dftfe::utils::complexConj(gradPsiUp) * psiDown +
                              dftfe::utils::complexConj(psiUp) * gradPsiDown);
                        }
                    }
                }
            }
    else
      for (dftfe::uInt iCell = cellRange.first; iCell < cellRange.second;
           ++iCell)
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
                                           2 * nQuadsPerCell *
                                             vectorsBlockSize +
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
    double    *kineticEnergyDensityCellsWfcContributions,
    double    *tau,
    const bool isNonCollin,
    const bool hasSOC)
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
    const dftfe::uInt          tempQuadratureIndex,
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
    const dftfe::uInt          tempQuadratureIndex,
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
