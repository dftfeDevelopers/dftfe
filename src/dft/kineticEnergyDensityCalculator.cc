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
// @author Sambit Das, Vishal Subramanian
//

// source file for electron density related computations
#include <constants.h>
#include <kineticEnergyDensityCalculator.h>
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
  computeKineticEnergyDensity(
    const dftfe::linearAlgebra::BLASWrapper<memorySpace> &      BLASWrapperPtr,
    const dftfe::utils::MemoryStorage<NumberType, memorySpace> *X,
    const unsigned int                      totalNumWaveFunctions,
    const std::vector<std::vector<double>> &eigenValues,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<NumberType, double, memorySpace>>
      &                        basisOperationsPtr,
    const unsigned int         quadratureIndex,
    const std::vector<double> &kPointCoords,
    const std::vector<double> &kPointWeights,
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                  kineticEnergyDensityValues,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const MPI_Comm &     mpiCommDomain,
    const dftParameters &dftParams)
  {
    int this_process;
    MPI_Comm_rank(mpiCommParent, &this_process);
#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(mpiCommParent);
    double             computeKed_time = MPI_Wtime();
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
    const NumberType scalarCoeffAlphaKed = 1.0;
    const NumberType scalarCoeffBetaKed  = 1.0;

    const unsigned int cellsBlockSize =
      memorySpace == dftfe::utils::MemorySpace::DEVICE ? 50 : 1;
    const unsigned int numCellBlocks = totalLocallyOwnedCells / cellsBlockSize;
    const unsigned int remCellBlockSize =
      totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;
    basisOperationsPtr->reinit(BVec, cellsBlockSize, quadratureIndex);
    const unsigned int numQuadPoints = basisOperationsPtr->nQuadsPerCell();

    std::vector<dftfe::utils::MemoryStorage<NumberType, memorySpace>>
      wfcQuadPointData(numSpinComponents);
    std::vector<dftfe::utils::MemoryStorage<NumberType, memorySpace>>
      gradWfcQuadPointData(numSpinComponents);
    std::vector<dftfe::utils::MemoryStorage<double, memorySpace>>
      kedWfcContributions(numSpinComponents);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      kedHost;

#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, memorySpace> ked;
#else
    auto &ked             = kedHost;
#endif

    ked.resize(totalLocallyOwnedCells * numQuadPoints * numSpinComponents, 0.0);
    for (unsigned int spinIndex = 0; spinIndex < numSpinComponents; ++spinIndex)
      {
        wfcQuadPointData[spinIndex].resize(cellsBlockSize * numQuadPoints *
                                             BVec,
                                           zero);

        if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          kedWfcContributions[spinIndex].resize(cellsBlockSize * numQuadPoints *
                                                  BVec,
                                                0.0);
      }
    for (unsigned int spinIndex = 0; spinIndex < numSpinComponents; ++spinIndex)
      {
        gradWfcQuadPointData[spinIndex].resize(cellsBlockSize * numQuadPoints *
                                                 BVec * 3,
                                               zero);
      }


    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
    partialOccupVecHost(
      numSpinComponents,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>(
        BVec, 0.0));
#if defined(DFTFE_WITH_DEVICE)
    std::vector<dftfe::utils::MemoryStorage<double, memorySpace>>
      partialOccupVec(numSpinComponents);
    for (unsigned int spinIndex = 0; spinIndex < numSpinComponents; ++spinIndex)
      partialOccupVec[spinIndex].resize(partialOccupVecHost[spinIndex].size());
#else
    auto &partialOccupVec = partialOccupVecHost;
#endif

    std::vector<dftfe::linearAlgebra::MultiVector<NumberType, memorySpace> *>
      flattenedArrayBlock(numSpinComponents);

    for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
      {
        std::vector<double> kcoord(3, 0);
        kcoord[0] = kPointCoords[3 * kPoint + 0];
        kcoord[1] = kPointCoords[3 * kPoint + 1];
        kcoord[2] = kPointCoords[3 * kPoint + 2];
        for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
             ++spinIndex)
          {
            wfcQuadPointData[spinIndex].setValue(zero);
            gradWfcQuadPointData[spinIndex].setValue(zero);
            kedWfcContributions[spinIndex].setValue(0.0);
          }
        for (unsigned int jvec = 0; jvec < totalNumWaveFunctions; jvec += BVec)
          {
            const unsigned int currentBlockSize =
              std::min(BVec, totalNumWaveFunctions - jvec);
            for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
                 ++spinIndex)
              flattenedArrayBlock[spinIndex] =
                &(basisOperationsPtr->getMultiVector(currentBlockSize,
                                                     spinIndex));

            if ((jvec + currentBlockSize) <=
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                (jvec + currentBlockSize) >
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
              {
                for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
                     ++spinIndex)
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
                                           [totalNumWaveFunctions * spinIndex +
                                            jvec + iEigenVec] >
                                fermiEnergyConstraintMag)
                              *(partialOccupVecHost[spinIndex].begin() +
                                iEigenVec) = 0;
                            else
                              *(partialOccupVecHost[spinIndex].begin() +
                                iEigenVec) =
                                kPointWeights[kPoint] * spinPolarizedFactor;
                          }
                      }
                    else
                      {
                        for (unsigned int iEigenVec = 0;
                             iEigenVec < currentBlockSize;
                             ++iEigenVec)
                          {
                            *(partialOccupVecHost[spinIndex].begin() +
                              iEigenVec) =
                              dftUtils::getPartialOccupancy(
                                eigenValues[kPoint]
                                           [totalNumWaveFunctions * spinIndex +
                                            jvec + iEigenVec],
                                fermiEnergy,
                                C_kb,
                                dftParams.TVal) *
                              kPointWeights[kPoint] * spinPolarizedFactor;
                          }
                      }
                  }
#if defined(DFTFE_WITH_DEVICE)
                for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
                     ++spinIndex)
                  partialOccupVec[spinIndex].copyFrom(
                    partialOccupVecHost[spinIndex]);
#endif
                for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
                     ++spinIndex)
                  if (memorySpace == dftfe::utils::MemorySpace::HOST)
                    for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                      std::memcpy(flattenedArrayBlock[spinIndex]->data() +
                                    iNode * currentBlockSize,
                                  X->data() +
                                    numLocalDofs * totalNumWaveFunctions *
                                      (numSpinComponents * kPoint + spinIndex) +
                                    iNode * totalNumWaveFunctions + jvec,
                                  currentBlockSize * sizeof(NumberType));
#if defined(DFTFE_WITH_DEVICE)
                  else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
                    BLASWrapperPtr.stridedCopyToBlockConstantStride(
                      currentBlockSize,
                      totalNumWaveFunctions,
                      numLocalDofs,
                      jvec,
                      X->data() + numLocalDofs * totalNumWaveFunctions *
                                    (numSpinComponents * kPoint + spinIndex),
                      flattenedArrayBlock[spinIndex]->data());
#endif


                basisOperationsPtr->reinit(currentBlockSize,
                                           cellsBlockSize,
                                           quadratureIndex,
                                           true);


                for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
                     ++spinIndex)
                  {
                    flattenedArrayBlock[spinIndex]->updateGhostValues();
                    basisOperationsPtr->distribute(
                      *(flattenedArrayBlock[spinIndex]));
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

                        for (unsigned int spinIndex = 0;
                             spinIndex < numSpinComponents;
                             ++spinIndex)
                          basisOperationsPtr->interpolateKernel(
                            *(flattenedArrayBlock[spinIndex]),
                            wfcQuadPointData[spinIndex].data(),
                            gradWfcQuadPointData[spinIndex].data(),
                            std::pair<unsigned int, unsigned int>(
                              startingCellId,
                              startingCellId + currentCellsBlockSize));

                        for (unsigned int spinIndex = 0;
                             spinIndex < numSpinComponents;
                             ++spinIndex)
                          computeKineticEnergyDensityFromInterpolatedValues(
                            BLASWrapperPtr,
                            basisOperationsPtr,
                            std::pair<unsigned int, unsigned int>(
                              startingCellId,
                              startingCellId + currentCellsBlockSize),
                            std::pair<unsigned int, unsigned int>(
                              jvec, jvec + currentBlockSize),
                            partialOccupVec[spinIndex].data(),
                            &kcoord[0],
                            wfcQuadPointData[spinIndex].data(),
                            gradWfcQuadPointData[spinIndex].data(),
                            kedWfcContributions[spinIndex].data(),
                            ked.data() + spinIndex * totalLocallyOwnedCells *
                                           numQuadPoints,
                            mpiCommDomain);
                      } // non-trivial cell block check
                  }     // cells block loop
              }
          } // jvec loop

      } // kpoint loop
#if defined(DFTFE_WITH_DEVICE)
    kedHost.resize(ked.size());

    kedHost.copyFrom(ked);

#endif

    int size;
    MPI_Comm_size(interpoolcomm, &size);
    if (size > 1)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      kedHost.data(),
                      totalLocallyOwnedCells * numQuadPoints *
                        numSpinComponents,
                      dataTypes::mpi_type_id(kedHost.data()),
                      MPI_SUM,
                      interpoolcomm);
      }
    MPI_Comm_size(interBandGroupComm, &size);
    if (size > 1)
      {
        MPI_Allreduce(MPI_IN_PLACE,
                      kedHost.data(),
                      totalLocallyOwnedCells * numQuadPoints *
                        numSpinComponents,
                      dataTypes::mpi_type_id(kedHost.data()),
                      MPI_SUM,
                      interBandGroupComm);
      }

    kineticEnergyDensityValues.resize(totalLocallyOwnedCells * numQuadPoints);
    std::fill(kineticEnergyDensityValues.begin(),
              kineticEnergyDensityValues.end(),
              0.0);
    for (unsigned int iElem = 0; iElem < totalLocallyOwnedCells; ++iElem)
      {
        if (dftParams.spinPolarized == 1)
          {
            for (unsigned int q = 0; q < numQuadPoints; ++q)
              {
                const double ked0 = kedHost[iElem * numQuadPoints + q];
                const double ked1 =
                  kedHost[totalLocallyOwnedCells * numQuadPoints +
                          iElem * numQuadPoints + q];
                kineticEnergyDensityValues[iElem * numQuadPoints + q] =
                  ked0 + ked1;
              }
          }
        else
          std::memcpy(kineticEnergyDensityValues.data() + iElem * numQuadPoints,
                      kedHost.data() + iElem * numQuadPoints,
                      numQuadPoints * sizeof(double));
      }
#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(mpiCommParent);
    computeKed_time = MPI_Wtime() - computeKed_time;

    if (this_process == 0 && dftParams.verbosity >= 2)
      if (memorySpace == dftfe::utils::MemorySpace::HOST)
        std::cout << "Time for compute ked on CPU: " << computeKed_time
                  << std::endl;
      else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
        std::cout << "Time for compute ked on Device: " << computeKed_time
                  << std::endl;
  }
  template <typename NumberType>
  void
  computeKineticEnergyDensityFromInterpolatedValues(
    const dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>
      &BLASWrapperPtr,
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<NumberType, double, dftfe::utils::MemorySpace::HOST>>
      &                                         basisOperationsPtr,
    const std::pair<unsigned int, unsigned int> cellRange,
    const std::pair<unsigned int, unsigned int> vecRange,
    double *                                    partialOccupVec,
    double *                                    kcoord,
    NumberType *                                wfcQuadPointData,
    NumberType *                                gradWfcQuadPointData,
    double *                                    kineticCellsWfcContributions,
    double *                                    kineticEnergyDensity,
    const MPI_Comm &                            mpiCommDomain)
  {
    const unsigned int cellsBlockSize   = cellRange.second - cellRange.first;
    const unsigned int vectorsBlockSize = vecRange.second - vecRange.first;
    const unsigned int nQuadsPerCell    = basisOperationsPtr->nQuadsPerCell();
    const unsigned int nCells           = basisOperationsPtr->nCells();
    const double       kcoordSq =
      kcoord[0] * kcoord[0] + kcoord[1] * kcoord[1] + kcoord[2] * kcoord[2];
    for (unsigned int iCell = cellRange.first; iCell < cellRange.second;
         ++iCell)
      for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
        for (unsigned int iWave = 0; iWave < vecRange.second - vecRange.first;
             ++iWave)
          {
            kineticEnergyDensity[iCell * nQuadsPerCell + iQuad] +=
              0.5 * partialOccupVec[iWave] *
              dftfe::utils::realPart(
                dftfe::utils::complexConj(
                  gradWfcQuadPointData[(iCell - cellRange.first) *
                                         nQuadsPerCell * vectorsBlockSize * 3 +
                                       iQuad * vectorsBlockSize + iWave]) *
                gradWfcQuadPointData[(iCell - cellRange.first) * nQuadsPerCell *
                                       vectorsBlockSize * 3 +
                                     iQuad * vectorsBlockSize + iWave]);
            kineticEnergyDensity[iCell * nQuadsPerCell + iQuad] +=
              0.5 * partialOccupVec[iWave] *
              dftfe::utils::realPart(
                dftfe::utils::complexConj(
                  gradWfcQuadPointData[(iCell - cellRange.first) *
                                         nQuadsPerCell * vectorsBlockSize * 3 +
                                       nQuadsPerCell * vectorsBlockSize +
                                       iQuad * vectorsBlockSize + iWave]) *
                gradWfcQuadPointData[(iCell - cellRange.first) * nQuadsPerCell *
                                       vectorsBlockSize * 3 +
                                     nQuadsPerCell * vectorsBlockSize +
                                     iQuad * vectorsBlockSize + iWave]);
            kineticEnergyDensity[iCell * nQuadsPerCell + iQuad] +=
              0.5 * partialOccupVec[iWave] *
              dftfe::utils::realPart(
                dftfe::utils::complexConj(
                  gradWfcQuadPointData[(iCell - cellRange.first) *
                                         nQuadsPerCell * vectorsBlockSize * 3 +
                                       2 * nQuadsPerCell * vectorsBlockSize +
                                       iQuad * vectorsBlockSize + iWave]) *
                gradWfcQuadPointData[(iCell - cellRange.first) * nQuadsPerCell *
                                       vectorsBlockSize * 3 +
                                     2 * nQuadsPerCell * vectorsBlockSize +
                                     iQuad * vectorsBlockSize + iWave]);

            kineticEnergyDensity[iCell * nQuadsPerCell + iQuad] +=
              0.5 * partialOccupVec[iWave] * kcoordSq *
              dftfe::utils::realPart(
                dftfe::utils::complexConj(
                  wfcQuadPointData[(iCell - cellRange.first) * nQuadsPerCell *
                                     vectorsBlockSize +
                                   iQuad * vectorsBlockSize + iWave]) *
                wfcQuadPointData[(iCell - cellRange.first) * nQuadsPerCell *
                                   vectorsBlockSize +
                                 iQuad * vectorsBlockSize + iWave]);



            kineticEnergyDensity[iCell * nQuadsPerCell + iQuad] +=
              kcoord[0] * partialOccupVec[iWave] *
              dftfe::utils::imagPart(
                dftfe::utils::complexConj(
                  wfcQuadPointData[(iCell - cellRange.first) * nQuadsPerCell *
                                     vectorsBlockSize +
                                   iQuad * vectorsBlockSize + iWave]) *
                gradWfcQuadPointData[(iCell - cellRange.first) * nQuadsPerCell *
                                       vectorsBlockSize * 3 +
                                     0 * nQuadsPerCell * vectorsBlockSize +
                                     iQuad * vectorsBlockSize + iWave]);

            kineticEnergyDensity[iCell * nQuadsPerCell + iQuad] +=
              kcoord[1] * partialOccupVec[iWave] *
              dftfe::utils::imagPart(
                dftfe::utils::complexConj(
                  wfcQuadPointData[(iCell - cellRange.first) * nQuadsPerCell *
                                     vectorsBlockSize +
                                   iQuad * vectorsBlockSize + iWave]) *
                gradWfcQuadPointData[(iCell - cellRange.first) * nQuadsPerCell *
                                       vectorsBlockSize * 3 +
                                     1 * nQuadsPerCell * vectorsBlockSize +
                                     iQuad * vectorsBlockSize + iWave]);

            kineticEnergyDensity[iCell * nQuadsPerCell + iQuad] +=
              kcoord[2] * partialOccupVec[iWave] *
              dftfe::utils::imagPart(
                dftfe::utils::complexConj(
                  wfcQuadPointData[(iCell - cellRange.first) * nQuadsPerCell *
                                     vectorsBlockSize +
                                   iQuad * vectorsBlockSize + iWave]) *
                gradWfcQuadPointData[(iCell - cellRange.first) * nQuadsPerCell *
                                       vectorsBlockSize * 3 +
                                     2 * nQuadsPerCell * vectorsBlockSize +
                                     iQuad * vectorsBlockSize + iWave]);
          }
  }
#if defined(DFTFE_WITH_DEVICE)
  template void
  computeKineticEnergyDensity(
    const dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>
      &BLASWrapperPtr,
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::DEVICE> *X,
    const unsigned int                      totalNumWaveFunctions,
    const std::vector<std::vector<double>> &eigenValues,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::DEVICE>>
      &                        basisOperationsPtr,
    const unsigned int         quadratureIndex,
    const std::vector<double> &kPointCoords,
    const std::vector<double> &kPointWeights,
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                  kineticEnergyDensityValues,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const MPI_Comm &     mpiCommDomain,
    const dftParameters &dftParams);
#endif

  template void
  computeKineticEnergyDensity(
    const dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>
      &BLASWrapperPtr,
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::HOST> *X,
    const unsigned int                      totalNumWaveFunctions,
    const std::vector<std::vector<double>> &eigenValues,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      &                        basisOperationsPtr,
    const unsigned int         quadratureIndex,
    const std::vector<double> &kPointCoords,
    const std::vector<double> &kPointWeights,
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                  kineticEnergyDensityValues,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const MPI_Comm &     mpiCommDomain,
    const dftParameters &dftParams);
} // namespace dftfe
