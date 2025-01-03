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
#include <DataTypeOverloads.h>
#include <linearAlgebraOperationsDevice.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherConstants.h>


namespace dftfe
{
  template <typename NumberType, dftfe::utils::MemorySpace memorySpace>
  void
  computeRhoFromPSI(
    const dftfe::utils::MemoryStorage<NumberType, memorySpace> *X,
    const dftfe::utils::MemoryStorage<NumberType, memorySpace> *XFrac,
    const unsigned int                      totalNumWaveFunctions,
    const unsigned int                      Nfr,
    const std::vector<std::vector<double>> &eigenValues,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<NumberType, double, memorySpace>>
      &basisOperationsPtr,
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      &                        BLASWrapperPtr,
    const unsigned int         matrixFreeDofhandlerIndex,
    const unsigned int         quadratureIndex,
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
      &                  tauValues,
    const bool           isEvaluateGradRho,
    const bool           isEvaluateTau,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams,
    const bool           spectrumSplit)
  {
    int this_process;
    MPI_Comm_rank(mpiCommParent, &this_process);
#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(mpiCommParent); // why is the barrier required here?

    double             computeRho_time = MPI_Wtime();
    const unsigned int numKPoints      = kPointWeights.size();
    const unsigned int numLocalDofs    = basisOperationsPtr->nOwnedDofs();
    const unsigned int totalLocallyOwnedCells = basisOperationsPtr->nCells();
    const unsigned int numNodesPerElement = basisOperationsPtr->nDofsPerCell();
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
      std::min(dftParams.chebyWfcBlockSize,
               bandGroupLowHighPlusOneIndices[1]); // have to understand this

    const double spinPolarizedFactor =
      (dftParams.spinPolarized == 1) ? 1.0 : 2.0;
    const unsigned int numSpinComponents =
      (dftParams.spinPolarized == 1) ? 2 : 1;

    const NumberType zero                    = 0;
    const NumberType scalarCoeffAlphaRho     = 1.0;
    const NumberType scalarCoeffBetaRho      = 1.0;
    const NumberType scalarCoeffAlphaGradRho = 1.0;
    const NumberType scalarCoeffBetaGradRho  = 1.0;

    const unsigned int cellsBlockSize =
      memorySpace == dftfe::utils::MemorySpace::DEVICE ? 50 : 1;
    const unsigned int numCellBlocks = totalLocallyOwnedCells / cellsBlockSize;
    const unsigned int remCellBlockSize =
      totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;
    basisOperationsPtr->reinit(BVec,
                               cellsBlockSize,
                               quadratureIndex); // why this is required ?
    const unsigned int numQuadPoints = basisOperationsPtr->nQuadsPerCell();

    dftfe::utils::MemoryStorage<NumberType, memorySpace> wfcQuadPointData;
    dftfe::utils::MemoryStorage<NumberType, memorySpace> gradWfcQuadPointData;
    dftfe::utils::MemoryStorage<double, memorySpace>     rhoWfcContributions;
    dftfe::utils::MemoryStorage<double, memorySpace>
                                                     tauWfcContributions; // why this is required ?
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
    auto &rho                         = rhoHost;
    auto &gradRho                     = gradRhoHost;
    auto &tau                         = tauHost;
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
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, memorySpace> partialOccupVec(
      partialOccupVecHost.size());
#else
    auto &partialOccupVec             = partialOccupVecHost;
#endif

    dftfe::linearAlgebra::MultiVector<NumberType, memorySpace>
      *flattenedArrayBlock;

    for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
      {
        std::vector<double> kCoord = {kPointCoords[3 * kPoint + 0],
                                      kPointCoords[3 * kPoint + 1],
                                      kPointCoords[3 * kPoint + 2]};
        for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
             ++spinIndex)
          {
            wfcQuadPointData.setValue(zero);
            gradWfcQuadPointData.setValue(zero);
            rhoWfcContributions.setValue(0.0);
            gradRhoWfcContributions.setValue(0.0);
            tauWfcContributions.setValue(0.0);
            for (unsigned int jvec = 0; jvec < totalNumWaveFunctions;
                 jvec += BVec)
              {
                const unsigned int currentBlockSize =
                  std::min(BVec, totalNumWaveFunctions - jvec);
                flattenedArrayBlock =
                  &(basisOperationsPtr->getMultiVector(currentBlockSize, 0));

                if ((jvec + currentBlockSize) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + currentBlockSize) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    if (spectrumSplit)
                      {
                        partialOccupVecHost.setValue(kPointWeights[kPoint] *
                                                     spinPolarizedFactor);
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
                                  *(partialOccupVecHost.begin() + iEigenVec) =
                                    0;
                                else
                                  *(partialOccupVecHost.begin() + iEigenVec) =
                                    kPointWeights[kPoint] * spinPolarizedFactor;
                              }
                          }
                        else
                          {
                            for (unsigned int iEigenVec = 0;
                                 iEigenVec < currentBlockSize;
                                 ++iEigenVec)
                              {
                                *(partialOccupVecHost.begin() + iEigenVec) =
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
#if defined(DFTFE_WITH_DEVICE)
                    partialOccupVec.copyFrom(partialOccupVecHost);
#endif
                    if (memorySpace == dftfe::utils::MemorySpace::HOST)
                      for (unsigned int iNode = 0; iNode < numLocalDofs;
                           ++iNode)
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

                    for (int iblock = 0; iblock < (numCellBlocks + 1); iblock++)
                      {
                        const unsigned int currentCellsBlockSize =
                          (iblock == numCellBlocks) ? remCellBlockSize :
                                                      cellsBlockSize;
                        if (currentCellsBlockSize > 0)
                          {
                            const unsigned int startingCellId =
                              iblock * cellsBlockSize;

                            // Since we have the data at the nodal points
                            // this function interpolates to the quad points
                            // the what about the inerpolation error?
                            basisOperationsPtr->interpolateKernel(
                              *(flattenedArrayBlock),
                              wfcQuadPointData.data(),
                              isEvaluateGradRho ? gradWfcQuadPointData.data() :
                                                  NULL,
                              std::pair<unsigned int, unsigned int>(
                                startingCellId,
                                startingCellId + currentCellsBlockSize));

                            computeRhoGradRhoFromInterpolatedValues(
                              basisOperationsPtr,
                              BLASWrapperPtr,
                              std::pair<unsigned int, unsigned int>(
                                startingCellId,
                                startingCellId + currentCellsBlockSize),
                              std::pair<unsigned int, unsigned int>(
                                jvec, jvec + currentBlockSize),
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
                                  basisOperationsPtr,
                                  BLASWrapperPtr,
                                  std::pair<unsigned int, unsigned int>(
                                    startingCellId,
                                    startingCellId + currentCellsBlockSize),
                                  std::pair<unsigned int, unsigned int>(
                                    jvec, jvec + currentBlockSize),
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

            if (spectrumSplit)
              for (unsigned int jvec = 0; jvec < Nfr; jvec += BVec)
                {
                  const unsigned int currentBlockSize =
                    std::min(BVec, Nfr - jvec);
                  flattenedArrayBlock =
                    &(basisOperationsPtr->getMultiVector(currentBlockSize, 0));
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
                                *(partialOccupVecHost.begin() + iEigenVec) =
                                  -kPointWeights[kPoint] * spinPolarizedFactor;
                              else
                                *(partialOccupVecHost.begin() + iEigenVec) = 0;
                            }
                        }
                      else
                        {
                          for (unsigned int iEigenVec = 0;
                               iEigenVec < currentBlockSize;
                               ++iEigenVec)
                            {
                              *(partialOccupVecHost.begin() + iEigenVec) =
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

#if defined(DFTFE_WITH_DEVICE)
                      partialOccupVec.resize(partialOccupVecHost.size());
                      partialOccupVec.copyFrom(partialOccupVecHost);
#endif
                      if (memorySpace == dftfe::utils::MemorySpace::HOST)
                        for (unsigned int iNode = 0; iNode < numLocalDofs;
                             ++iNode)
                          std::memcpy(flattenedArrayBlock->data() +
                                        iNode * currentBlockSize,
                                      XFrac->data() +
                                        numLocalDofs * Nfr *
                                          (numSpinComponents * kPoint +
                                           spinIndex) +
                                        iNode * Nfr + jvec,
                                      currentBlockSize * sizeof(NumberType));
#if defined(DFTFE_WITH_DEVICE)
                      else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
                        BLASWrapperPtr->stridedCopyToBlockConstantStride(
                          currentBlockSize,
                          Nfr,
                          numLocalDofs,
                          jvec,
                          XFrac->data() +
                            numLocalDofs * Nfr *
                              (numSpinComponents * kPoint + spinIndex),
                          flattenedArrayBlock->data());
#endif
                      basisOperationsPtr->reinit(currentBlockSize,
                                                 cellsBlockSize,
                                                 quadratureIndex,
                                                 false);


                      flattenedArrayBlock->updateGhostValues();
                      basisOperationsPtr->distribute(*(flattenedArrayBlock));

                      for (int iblock = 0; iblock < (numCellBlocks + 1);
                           iblock++)
                        {
                          const unsigned int currentCellsBlockSize =
                            (iblock == numCellBlocks) ? remCellBlockSize :
                                                        cellsBlockSize;
                          if (currentCellsBlockSize > 0)
                            {
                              const unsigned int startingCellId =
                                iblock * cellsBlockSize;
                              basisOperationsPtr->interpolateKernel(
                                *(flattenedArrayBlock),
                                wfcQuadPointData.data(),
                                isEvaluateGradRho ?
                                  gradWfcQuadPointData.data() :
                                  NULL,
                                std::pair<unsigned int, unsigned int>(
                                  startingCellId,
                                  startingCellId + currentCellsBlockSize));

                              computeRhoGradRhoFromInterpolatedValues(
                                basisOperationsPtr,
                                BLASWrapperPtr,
                                std::pair<unsigned int, unsigned int>(
                                  startingCellId,
                                  startingCellId + currentCellsBlockSize),
                                std::pair<unsigned int, unsigned int>(
                                  jvec, jvec + currentBlockSize),
                                partialOccupVec.data(),
                                wfcQuadPointData.data(),
                                gradWfcQuadPointData.data(),
                                rhoWfcContributions.data(),
                                gradRhoWfcContributions.data(),
                                rho.data() + spinIndex *
                                               totalLocallyOwnedCells *
                                               numQuadPoints,
                                gradRho.data() + spinIndex *
                                                   totalLocallyOwnedCells *
                                                   numQuadPoints * 3,
                                isEvaluateGradRho);

                              if (isEvaluateTau)
                                {
                                  std::string errMsg =
                                    "spectrum split for tau computation is not implemented yet in SCAN.";
                                  dftfe::utils::throwException(false, errMsg);
                                }
                            } // non-tivial cells block
                        }     // cells block loop
                    }
                } // spectrum split block
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
        std::string errMsg =
          "Device implementation for tau computation is not implemented yet in SCAN.";
        dftfe::utils::throwException(false, errMsg);
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

  template <typename NumberType, dftfe::utils::MemorySpace memorySpace>
  void
  computeInitTauFromPSI(
    const dftfe::utils::MemoryStorage<NumberType, memorySpace> *X,
    const dftfe::utils::MemoryStorage<NumberType, memorySpace> *XFrac,
    const unsigned int                                          numElectrons,
    const unsigned int                      totalNumWaveFunctions,
    const unsigned int                      Nfr,
    const std::vector<std::vector<double>> &eigenValues,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<NumberType, double, memorySpace>>
      &basisOperationsPtr,
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      &                        BLASWrapperPtr,
    const unsigned int         matrixFreeDofhandlerIndex,
    const unsigned int         quadratureIndex,
    const std::vector<double> &kPointCoords,
    const std::vector<double> &kPointWeights,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &                  tauValues,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams,
    const bool           spectrumSplit)
  {
    int this_process;
    MPI_Comm_rank(mpiCommParent, &this_process);
#if defined(DFTFE_WITH_DEVICE)
    std::string errMsg = "GPU case is not implemented yet in SCAN.";
    dftfe::utils::throwException(false, errMsg);
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(mpiCommParent);
    double             computeTau_time = MPI_Wtime();
    const unsigned int numKPoints      = kPointWeights.size();
    const unsigned int numLocalDofs    = basisOperationsPtr->nOwnedDofs();
    const unsigned int totalLocallyOwnedCells = basisOperationsPtr->nCells();
    const unsigned int numNodesPerElement = basisOperationsPtr->nDofsPerCell();
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
    const unsigned int numSpinComponents =
      (dftParams.spinPolarized == 1) ? 2 : 1;

    const NumberType zero = 0;

    const unsigned int cellsBlockSize =
      memorySpace == dftfe::utils::MemorySpace::DEVICE ? 50 : 1;
    const unsigned int numCellBlocks = totalLocallyOwnedCells / cellsBlockSize;
    const unsigned int remCellBlockSize =
      totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;
    basisOperationsPtr->reinit(BVec, cellsBlockSize, quadratureIndex);
    const unsigned int numQuadPoints = basisOperationsPtr->nQuadsPerCell();

    dftfe::utils::MemoryStorage<NumberType, memorySpace> wfcQuadPointData;
    dftfe::utils::MemoryStorage<NumberType, memorySpace> gradWfcQuadPointData;
    dftfe::utils::MemoryStorage<double, memorySpace>     tauWfcContributions;


    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      tauHost;
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, memorySpace> tau;
#else
    auto &tau                         = tauHost;
#endif

    wfcQuadPointData.resize(cellsBlockSize * numQuadPoints * BVec, zero);

    gradWfcQuadPointData.resize(cellsBlockSize * numQuadPoints * BVec * 3,
                                zero);
    tau.resize(totalLocallyOwnedCells * numQuadPoints * numSpinComponents, 0.0);
    tauValues.resize(numSpinComponents);
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      {
        tauWfcContributions.resize(cellsBlockSize * numQuadPoints * BVec, 0.0);
      }

    for (unsigned int iComp = 0; iComp < tauValues.size(); ++iComp)
      {
        tauValues[iComp].resize(totalLocallyOwnedCells * numQuadPoints);
      }

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      partialOccupVecHost(BVec, 0.0);

    std::vector<double> initTauOccvec(numSpinComponents * totalNumWaveFunctions,
                                      0.0);

    for (int i = 0; i < numElectrons / 2; ++i)
      {
        initTauOccvec[i] = double(1.0);
      }

    if (numElectrons % 2 != 0)
      {
        initTauOccvec[int(numElectrons / 2)] = double(0.5);
      }

    if (numSpinComponents == 2)
      {
        for (int i = totalNumWaveFunctions;
             i < totalNumWaveFunctions + numElectrons / 2;
             ++i)
          {
            initTauOccvec[i] = double(1.0);
          }
        if (numElectrons % 2 != 0)
          {
            initTauOccvec[totalNumWaveFunctions + int(numElectrons / 2)] =
              double(0.5);
          }
      }

#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, memorySpace> partialOccupVec(
      partialOccupVecHost.size());
#else
    auto &partialOccupVec             = partialOccupVecHost;
#endif

    dftfe::linearAlgebra::MultiVector<NumberType, memorySpace>
      *flattenedArrayBlock;

    for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
      {
        std::vector<double> kCoord = {kPointCoords[3 * kPoint + 0],
                                      kPointCoords[3 * kPoint + 1],
                                      kPointCoords[3 * kPoint + 2]};
        for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
             ++spinIndex)
          {
            wfcQuadPointData.setValue(zero);
            gradWfcQuadPointData.setValue(zero);
            tauWfcContributions.setValue(0.0);
            for (unsigned int jvec = 0; jvec < totalNumWaveFunctions;
                 jvec += BVec)
              {
                const unsigned int currentBlockSize =
                  std::min(BVec, totalNumWaveFunctions - jvec);
                flattenedArrayBlock =
                  &(basisOperationsPtr->getMultiVector(currentBlockSize, 0));

                if ((jvec + currentBlockSize) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + currentBlockSize) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    if (spectrumSplit)
                      {
                        partialOccupVecHost.setValue(kPointWeights[kPoint] *
                                                     spinPolarizedFactor);
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
                                  *(partialOccupVecHost.begin() + iEigenVec) =
                                    0;
                                else
                                  *(partialOccupVecHost.begin() + iEigenVec) =
                                    kPointWeights[kPoint] * spinPolarizedFactor;
                              }
                          }
                        else
                          {
                            for (unsigned int iEigenVec = 0;
                                 iEigenVec < currentBlockSize;
                                 ++iEigenVec)
                              {
                                *(partialOccupVecHost.begin() + iEigenVec) =
                                  initTauOccvec[jvec + iEigenVec] *
                                  kPointWeights[kPoint] * spinPolarizedFactor;
                              }
                          }
                      }
#if defined(DFTFE_WITH_DEVICE)
                    partialOccupVec.copyFrom(partialOccupVecHost);
#endif
                    if (memorySpace == dftfe::utils::MemorySpace::HOST)
                      for (unsigned int iNode = 0; iNode < numLocalDofs;
                           ++iNode)
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
                      dftfe::utils::deviceKernelsGeneric::
                        stridedCopyToBlockConstantStride(
                          currentBlockSize,
                          totalNumWaveFunctions,
                          numLocalDofs,
                          jvec,
                          X->data() +
                            numLocalDofs * totalNumWaveFunctions *
                              (numSpinComponents * kPoint + spinIndex),
                          flattenedArrayBlock->data());
#endif


                    basisOperationsPtr->reinit(currentBlockSize,
                                               cellsBlockSize,
                                               quadratureIndex,
                                               false);


                    flattenedArrayBlock->updateGhostValues();
                    basisOperationsPtr->distribute(*(flattenedArrayBlock));

                    for (int iblock = 0; iblock < (numCellBlocks + 1); iblock++)
                      {
                        const unsigned int currentCellsBlockSize =
                          (iblock == numCellBlocks) ? remCellBlockSize :
                                                      cellsBlockSize;
                        if (currentCellsBlockSize > 0)
                          {
                            const unsigned int startingCellId =
                              iblock * cellsBlockSize;

                            basisOperationsPtr->interpolateKernel(
                              *(flattenedArrayBlock),
                              wfcQuadPointData.data(),
                              gradWfcQuadPointData.data(),
                              std::pair<unsigned int, unsigned int>(
                                startingCellId,
                                startingCellId + currentCellsBlockSize));

                            computeTauFromInterpolatedValues(
                              basisOperationsPtr,
                              BLASWrapperPtr,
                              std::pair<unsigned int, unsigned int>(
                                startingCellId,
                                startingCellId + currentCellsBlockSize),
                              std::pair<unsigned int, unsigned int>(
                                jvec, jvec + currentBlockSize),
                              partialOccupVec.data(),
                              kCoord.data(),
                              wfcQuadPointData.data(),
                              gradWfcQuadPointData.data(),
                              tauWfcContributions.data(),
                              tau.data() + spinIndex * totalLocallyOwnedCells *
                                             numQuadPoints);

                          } // non-trivial cell block check
                      }     // cells block loop
                  }
              }

            if (spectrumSplit)
              {
                std::string errMsg =
                  "spectrum split for tau computation is not implemented yet in SCAN.";
                dftfe::utils::throwException(false, errMsg);
              } // spectrum split block
          }
      }
#if defined(DFTFE_WITH_DEVICE)
    tauHost.resize(tau.size());

    tauHost.copyFrom(tau);

#endif

    int size;
    MPI_Comm_size(interpoolcomm, &size);
    if (size > 1)
      {
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
                      tauHost.data(),
                      totalLocallyOwnedCells * numQuadPoints *
                        numSpinComponents,
                      dataTypes::mpi_type_id(tauHost.data()),
                      MPI_SUM,
                      interpoolcomm);
      }

    if (dftParams.spinPolarized == 1)
      {
        tauValues[0].resize(totalLocallyOwnedCells * numQuadPoints);
        tauValues[1].resize(totalLocallyOwnedCells * numQuadPoints);
        std::transform(tauHost.begin(),
                       tauHost.begin() + totalLocallyOwnedCells * numQuadPoints,
                       tauHost.begin() + totalLocallyOwnedCells * numQuadPoints,
                       tauValues[0].begin(),
                       std::plus<>{});
        std::transform(tauHost.begin(),
                       tauHost.begin() + totalLocallyOwnedCells * numQuadPoints,
                       tauHost.begin() + totalLocallyOwnedCells * numQuadPoints,
                       tauValues[1].begin(),
                       std::minus<>{});
      }
    else
      {
        tauValues[0] = tauHost;
      }

#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif

    MPI_Barrier(mpiCommParent);
    computeTau_time = MPI_Wtime() - computeTau_time;

    if (this_process == 0 && dftParams.verbosity >= 2)
      if (memorySpace == dftfe::utils::MemorySpace::HOST)
        std::cout << "Time for compute initial guess of tau on CPU: "
                  << computeTau_time << std::endl;
      else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
        std::cout << "Time for compute initial guess of tau on Device: "
                  << computeTau_time << std::endl;
  }

  template <typename NumberType>
  void
  computeRhoGradRhoFromInterpolatedValues(
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<NumberType, double, dftfe::utils::MemorySpace::HOST>>
      &basisOperationsPtr,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      &                                         BLASWrapperPtr,
    const std::pair<unsigned int, unsigned int> cellRange,
    const std::pair<unsigned int, unsigned int> vecRange,
    double *                                    partialOccupVec,
    NumberType *                                wfcQuadPointData,
    NumberType *                                gradWfcQuadPointData,
    double *                                    rhoCellsWfcContributions,
    double *                                    gradRhoCellsWfcContributions,
    double *                                    rho,
    double *                                    gradRho,
    const bool                                  isEvaluateGradRho)
  {
    const unsigned int cellsBlockSize   = cellRange.second - cellRange.first;
    const unsigned int vectorsBlockSize = vecRange.second - vecRange.first;
    const unsigned int nQuadsPerCell    = basisOperationsPtr->nQuadsPerCell();
    const unsigned int nCells           = basisOperationsPtr->nCells();
    for (unsigned int iCell = cellRange.first; iCell < cellRange.second;
         ++iCell)
      for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
        for (unsigned int iWave = 0; iWave < vecRange.second - vecRange.first;
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

  // to  edit for device kernels as well
  template <typename NumberType>
  void
  computeTauFromInterpolatedValues(
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<NumberType, double, dftfe::utils::MemorySpace::HOST>>
      &basisOperationsPtr,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      &                                         BLASWrapperPtr,
    const std::pair<unsigned int, unsigned int> cellRange,
    const std::pair<unsigned int, unsigned int> vecRange,
    double *                                    partialOccupVec,
    double *                                    kCoord,
    NumberType *                                wfcQuadPointData,
    NumberType *                                gradWfcQuadPointData,
    double *kineticEnergyDensityCellsWfcContributions,
    double *tau)
  {
    const unsigned int cellsBlockSize   = cellRange.second - cellRange.first;
    const unsigned int vectorsBlockSize = vecRange.second - vecRange.first;
    const unsigned int nQuadsPerCell    = basisOperationsPtr->nQuadsPerCell();
    const unsigned int nCells           = basisOperationsPtr->nCells();

    const double kPointCoordSq =
      kCoord[0] * kCoord[0] + kCoord[1] * kCoord[1] + kCoord[2] * kCoord[2];
    for (unsigned int iCell = cellRange.first; iCell < cellRange.second;
         ++iCell)
      for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
        for (unsigned int iWave = 0; iWave < vecRange.second - vecRange.first;
             ++iWave)
          {
            NumberType dirValGradPsi;
            double     sumDirValGradPsi = 0.0;
            NumberType tempImag         = 0.0;

            const NumberType psi =
              wfcQuadPointData[(iCell - cellRange.first) * nQuadsPerCell *
                                 vectorsBlockSize +
                               iQuad * vectorsBlockSize + iWave];
            for (int dirIdx = 0; dirIdx < 3; ++dirIdx)
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
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::DEVICE> *XFrac,
    const unsigned int                      totalNumWaveFunctions,
    const unsigned int                      Nfr,
    const std::vector<std::vector<double>> &eigenValues,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::DEVICE>>
      &basisOperationsPtrDevice,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      &                        BLASWrapperPtr,
    const unsigned int         matrixFreeDofhandlerIndex,
    const unsigned int         quadratureIndex,
    const std::vector<double> &kPointWeights,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &densityValues,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &                  gradDensityValues,
    const bool           isEvaluateGradRho,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams,
    const bool           spectrumSplit);
#endif

  template void
  computeRhoFromPSI(
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::HOST> *X,
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::HOST> *XFrac,
    const unsigned int                      totalNumWaveFunctions,
    const unsigned int                      Nfr,
    const std::vector<std::vector<double>> &eigenValues,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      &basisOperationsPtr,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      &                        BLASWrapperPtr,
    const unsigned int         matrixFreeDofhandlerIndex,
    const unsigned int         quadratureIndex,
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
      &                  tauValues,
    const bool           isEvaluateGradRho,
    const bool           isEvaluateTau,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams,
    const bool           spectrumSplit);

  template void
  computeInitTauFromPSI(
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::HOST> *X,
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::HOST> *XFrac,
    const unsigned int                      numElectrons,
    const unsigned int                      totalNumWaveFunctions,
    const unsigned int                      Nfr,
    const std::vector<std::vector<double>> &eigenValues,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      &basisOperationsPtr,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      &                        BLASWrapperPtr,
    const unsigned int         matrixFreeDofhandlerIndex,
    const unsigned int         quadratureIndex,
    const std::vector<double> &kPointCoords,
    const std::vector<double> &kPointWeights,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &                  tauValues,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams,
    const bool           spectrumSplit);
} // namespace dftfe
