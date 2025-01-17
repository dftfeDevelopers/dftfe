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
#include <densityFirstOrderResponseCalculator.h>
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
  computeRhoFirstOrderResponse(
    const dftfe::utils::MemoryStorage<NumberType, memorySpace> &X,
    const dftfe::utils::MemoryStorage<NumberType, memorySpace> &XPrime,
    const unsigned int                      totalNumWaveFunctions,
    const std::vector<std::vector<double>> &densityMatDerFermiEnergy,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<NumberType, double, memorySpace>>
      &basisOperationsPtr,
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      &                        BLASWrapperPtr,
    const unsigned int         matrixFreeDofhandlerIndex,
    const unsigned int         quadratureIndex,
    const std::vector<double> &kPointWeights,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &rhoResponseValuesHam,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &                  rhoResponseValuesFermiEnergy,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams)
  {
    int this_process;
    MPI_Comm_rank(mpiCommParent, &this_process);
#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(mpiCommParent);
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
      std::min(dftParams.chebyWfcBlockSize, bandGroupLowHighPlusOneIndices[1]);

    const double spinPolarizedFactor =
      (dftParams.spinPolarized == 1) ? 1.0 : 2.0;
    const unsigned int numSpinComponents =
      (dftParams.spinPolarized == 1) ? 2 : 1;

    const NumberType zero                = 0;
    const NumberType scalarCoeffAlphaRho = 1.0;
    const NumberType scalarCoeffBetaRho  = 1.0;

    const unsigned int cellsBlockSize =
      memorySpace == dftfe::utils::MemorySpace::DEVICE ? 50 : 1;
    const unsigned int numCellBlocks = totalLocallyOwnedCells / cellsBlockSize;
    const unsigned int remCellBlockSize =
      totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;
    basisOperationsPtr->reinit(BVec, cellsBlockSize, quadratureIndex);
    const unsigned int numQuadPoints = basisOperationsPtr->nQuadsPerCell();

    dftfe::utils::MemoryStorage<NumberType, memorySpace> wfcQuadPointData;
    dftfe::utils::MemoryStorage<NumberType, memorySpace> wfcPrimeQuadPointData;
    dftfe::utils::MemoryStorage<double, memorySpace>
      rhoResponseHamWfcContributions;
    dftfe::utils::MemoryStorage<double, memorySpace>
      rhoResponseFermiEnergyWfcContributions;
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      rhoResponseHamHost;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      rhoResponseFermiEnergyHost;
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, memorySpace> rhoResponseHam;
    dftfe::utils::MemoryStorage<double, memorySpace> rhoResponseFermiEnergy;
#else
    auto &rhoResponseHam         = rhoResponseHamHost;
    auto &rhoResponseFermiEnergy = rhoResponseFermiEnergyHost;
#endif

    rhoResponseHam.resize(totalLocallyOwnedCells * numQuadPoints *
                            numSpinComponents,
                          0.0);
    rhoResponseFermiEnergy.resize(totalLocallyOwnedCells * numQuadPoints *
                                    numSpinComponents,
                                  0.0);
    wfcQuadPointData.resize(cellsBlockSize * numQuadPoints * BVec, zero);

    wfcPrimeQuadPointData.resize(cellsBlockSize * numQuadPoints * BVec, zero);

    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      {
        rhoResponseHamWfcContributions.resize(cellsBlockSize * numQuadPoints *
                                                BVec,
                                              0.0);

        rhoResponseFermiEnergyWfcContributions.resize(cellsBlockSize *
                                                        numQuadPoints * BVec,
                                                      0.0);
      }


    dftfe::utils::MemoryStorage<double, memorySpace> onesVec(
      BVec, spinPolarizedFactor);

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      partialOccupPrimeVecHost(BVec, 0.0);
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, memorySpace> partialOccupPrimeVec;
    partialOccupPrimeVec.resize(partialOccupPrimeVecHost.size());
#else
    auto &partialOccupPrimeVec   = partialOccupPrimeVecHost;
#endif

    std::vector<dftfe::linearAlgebra::MultiVector<NumberType, memorySpace> *>
      flattenedArrayBlock(2);

    for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
      for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
           ++spinIndex)
        {
          wfcQuadPointData.setValue(zero);
          wfcPrimeQuadPointData.setValue(zero);
          rhoResponseHamWfcContributions.setValue(0.0);
          rhoResponseFermiEnergyWfcContributions.setValue(0.0);
          for (unsigned int jvec = 0; jvec < totalNumWaveFunctions;
               jvec += BVec)
            {
              const unsigned int currentBlockSize =
                std::min(BVec, totalNumWaveFunctions - jvec);
              for (unsigned int icomp = 0; icomp < flattenedArrayBlock.size();
                   ++icomp)
                flattenedArrayBlock[icomp] = &(
                  basisOperationsPtr->getMultiVector(currentBlockSize, icomp));

              if ((jvec + currentBlockSize) <=
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                  (jvec + currentBlockSize) >
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  for (unsigned int iEigenVec = 0; iEigenVec < currentBlockSize;
                       ++iEigenVec)
                    {
                      *(partialOccupPrimeVecHost.begin() + iEigenVec) =
                        densityMatDerFermiEnergy[numSpinComponents * kPoint +
                                                 spinIndex][jvec + iEigenVec] *
                        kPointWeights[kPoint] * spinPolarizedFactor;
                    }
#if defined(DFTFE_WITH_DEVICE)
                  partialOccupPrimeVec.copyFrom(partialOccupPrimeVecHost);
#endif
                  if (memorySpace == dftfe::utils::MemorySpace::HOST)
                    for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                      std::memcpy(flattenedArrayBlock[0]->data() +
                                    iNode * currentBlockSize,
                                  X.data() +
                                    numLocalDofs * totalNumWaveFunctions *
                                      (numSpinComponents * kPoint + spinIndex) +
                                    iNode * totalNumWaveFunctions + jvec,
                                  currentBlockSize * sizeof(NumberType));
#if defined(DFTFE_WITH_DEVICE)
                  else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
                    BLASWrapperPtr->stridedCopyToBlockConstantStride(
                      currentBlockSize,
                      totalNumWaveFunctions,
                      numLocalDofs,
                      jvec,
                      X.data() + numLocalDofs * totalNumWaveFunctions *
                                   (numSpinComponents * kPoint + spinIndex),
                      flattenedArrayBlock[0]->data());
#endif


                  if (memorySpace == dftfe::utils::MemorySpace::HOST)
                    for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                      std::memcpy(flattenedArrayBlock[1]->data() +
                                    iNode * currentBlockSize,
                                  XPrime.data() +
                                    numLocalDofs * totalNumWaveFunctions *
                                      (numSpinComponents * kPoint + spinIndex) +
                                    iNode * totalNumWaveFunctions + jvec,
                                  currentBlockSize * sizeof(NumberType));
#if defined(DFTFE_WITH_DEVICE)
                  else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
                    BLASWrapperPtr->stridedCopyToBlockConstantStride(
                      currentBlockSize,
                      totalNumWaveFunctions,
                      numLocalDofs,
                      jvec,
                      XPrime.data() +
                        numLocalDofs * totalNumWaveFunctions *
                          (numSpinComponents * kPoint + spinIndex),
                      flattenedArrayBlock[1]->data());
#endif

                  basisOperationsPtr->reinit(currentBlockSize,
                                             cellsBlockSize,
                                             quadratureIndex,
                                             false);


                  for (unsigned int icomp = 0;
                       icomp < flattenedArrayBlock.size();
                       ++icomp)
                    {
                      flattenedArrayBlock[icomp]->updateGhostValues();
                      basisOperationsPtr->distribute(
                        *(flattenedArrayBlock[icomp]));
                    }

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
                            *(flattenedArrayBlock[0]),
                            wfcQuadPointData.data(),
                            NULL,
                            std::pair<unsigned int, unsigned int>(
                              startingCellId,
                              startingCellId + currentCellsBlockSize));

                          basisOperationsPtr->interpolateKernel(
                            *(flattenedArrayBlock[1]),
                            wfcPrimeQuadPointData.data(),
                            NULL,
                            std::pair<unsigned int, unsigned int>(
                              startingCellId,
                              startingCellId + currentCellsBlockSize));


                          computeRhoResponseFromInterpolatedValues(
                            basisOperationsPtr,
                            BLASWrapperPtr,
                            std::pair<unsigned int, unsigned int>(
                              startingCellId,
                              startingCellId + currentCellsBlockSize),
                            std::pair<unsigned int, unsigned int>(
                              jvec, jvec + currentBlockSize),
                            onesVec.data(),
                            partialOccupPrimeVec.data(),
                            wfcQuadPointData.data(),
                            wfcPrimeQuadPointData.data(),
                            rhoResponseHamWfcContributions.data(),
                            rhoResponseFermiEnergyWfcContributions.data(),
                            rhoResponseHam.data() + spinIndex *
                                                      totalLocallyOwnedCells *
                                                      numQuadPoints,
                            rhoResponseFermiEnergy.data() +
                              spinIndex * totalLocallyOwnedCells *
                                numQuadPoints);
                        } // non-trivial cell block check
                    }     // cells block loop
                }
            }
        }
#if defined(DFTFE_WITH_DEVICE)
    rhoResponseHamHost.resize(rhoResponseHam.size());

    rhoResponseHamHost.copyFrom(rhoResponseHam);

    rhoResponseFermiEnergyHost.resize(rhoResponseFermiEnergy.size());

    rhoResponseFermiEnergyHost.copyFrom(rhoResponseFermiEnergy);
#endif

    int size;
    MPI_Comm_size(interpoolcomm, &size);
    if (size > 1)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      rhoResponseHamHost.data(),
                      totalLocallyOwnedCells * numQuadPoints *
                        numSpinComponents,
                      dataTypes::mpi_type_id(rhoResponseHamHost.data()),
                      MPI_SUM,
                      interpoolcomm);

        MPI_Allreduce(MPI_IN_PLACE,
                      rhoResponseFermiEnergyHost.data(),
                      totalLocallyOwnedCells * numQuadPoints *
                        numSpinComponents,
                      dataTypes::mpi_type_id(rhoResponseFermiEnergyHost.data()),
                      MPI_SUM,
                      interpoolcomm);
      }
    MPI_Comm_size(interBandGroupComm, &size);
    if (size > 1)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      rhoResponseHamHost.data(),
                      totalLocallyOwnedCells * numQuadPoints *
                        numSpinComponents,
                      dataTypes::mpi_type_id(rhoResponseHamHost.data()),
                      MPI_SUM,
                      interBandGroupComm);

        MPI_Allreduce(MPI_IN_PLACE,
                      rhoResponseFermiEnergyHost.data(),
                      totalLocallyOwnedCells * numQuadPoints *
                        numSpinComponents,
                      dataTypes::mpi_type_id(rhoResponseFermiEnergyHost.data()),
                      MPI_SUM,
                      interBandGroupComm);
      }

    if (dftParams.spinPolarized == 1)
      {
        rhoResponseValuesHam[0].resize(totalLocallyOwnedCells * numQuadPoints);
        rhoResponseValuesHam[1].resize(totalLocallyOwnedCells * numQuadPoints);
        std::transform(rhoResponseHamHost.begin(),
                       rhoResponseHamHost.begin() +
                         totalLocallyOwnedCells * numQuadPoints,
                       rhoResponseHamHost.begin() +
                         totalLocallyOwnedCells * numQuadPoints,
                       rhoResponseValuesHam[0].begin(),
                       std::plus<>{});
        std::transform(rhoResponseHamHost.begin(),
                       rhoResponseHamHost.begin() +
                         totalLocallyOwnedCells * numQuadPoints,
                       rhoResponseHamHost.begin() +
                         totalLocallyOwnedCells * numQuadPoints,
                       rhoResponseValuesHam[1].begin(),
                       std::minus<>{});

        rhoResponseValuesFermiEnergy[0].resize(totalLocallyOwnedCells *
                                               numQuadPoints);
        rhoResponseValuesFermiEnergy[1].resize(totalLocallyOwnedCells *
                                               numQuadPoints);
        std::transform(rhoResponseFermiEnergyHost.begin(),
                       rhoResponseFermiEnergyHost.begin() +
                         totalLocallyOwnedCells * numQuadPoints,
                       rhoResponseFermiEnergyHost.begin() +
                         totalLocallyOwnedCells * numQuadPoints,
                       rhoResponseValuesFermiEnergy[0].begin(),
                       std::plus<>{});
        std::transform(rhoResponseFermiEnergyHost.begin(),
                       rhoResponseFermiEnergyHost.begin() +
                         totalLocallyOwnedCells * numQuadPoints,
                       rhoResponseFermiEnergyHost.begin() +
                         totalLocallyOwnedCells * numQuadPoints,
                       rhoResponseValuesFermiEnergy[1].begin(),
                       std::minus<>{});
      }
    else
      {
        rhoResponseValuesHam[0]         = rhoResponseHamHost;
        rhoResponseValuesFermiEnergy[0] = rhoResponseFermiEnergyHost;
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
  computeRhoResponseFromInterpolatedValues(
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<NumberType, double, dftfe::utils::MemorySpace::HOST>>
      &basisOperationsPtr,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      &                                         BLASWrapperPtr,
    const std::pair<unsigned int, unsigned int> cellRange,
    const std::pair<unsigned int, unsigned int> vecRange,
    double *                                    onesVec,
    double *                                    partialOccupVecPrime,
    NumberType *                                wfcQuadPointData,
    NumberType *                                wfcPrimeQuadPointData,
    double *rhoResponseHamCellsWfcContributions,
    double *rhoResponseFermiEnergyCellsWfcContributions,
    double *rhoResponseHam,
    double *rhoResponseFermiEnergy)
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
            const NumberType psiPrime =
              wfcPrimeQuadPointData[(iCell - cellRange.first) * nQuadsPerCell *
                                      vectorsBlockSize +
                                    iQuad * vectorsBlockSize + iWave];
            rhoResponseHam[iCell * nQuadsPerCell + iQuad] +=
              onesVec[iWave] *
              dftfe::utils::realPart(psi * dftfe::utils::complexConj(psiPrime));

            rhoResponseFermiEnergy[iCell * nQuadsPerCell + iQuad] +=
              partialOccupVecPrime[iWave] *
              dftfe::utils::realPart(psi * dftfe::utils::complexConj(psi));
          }
  }
#if defined(DFTFE_WITH_DEVICE)
  template void
  computeRhoFirstOrderResponse(
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::DEVICE> &X,
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::DEVICE>
      &                                     XPrime,
    const unsigned int                      totalNumWaveFunctions,
    const std::vector<std::vector<double>> &densityMatDerFermiEnergy,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::DEVICE>>
      &basisOperationsPtr,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      &                        BLASWrapperPtr,
    const unsigned int         matrixFreeDofhandlerIndex,
    const unsigned int         quadratureIndex,
    const std::vector<double> &kPointWeights,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &rhoResponseValuesHam,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &                  rhoResponseValuesFermiEnergy,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams);
#endif

  template void
  computeRhoFirstOrderResponse(
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::HOST> &X,
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::HOST> &XPrime,
    const unsigned int                      totalNumWaveFunctions,
    const std::vector<std::vector<double>> &densityMatDerFermiEnergy,
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
    const std::vector<double> &kPointWeights,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &rhoResponseValuesHam,
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      &                  rhoResponseValuesFermiEnergy,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams);
} // namespace dftfe
