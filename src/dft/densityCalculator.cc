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
#include <densityCalculator.h>
#include <dftUtils.h>
#include <vectorUtilities.h>
#include <MemoryStorage.h>
#include <DataTypeOverloads.h>
#include <deviceKernelsGeneric.h>
#include <linearAlgebraOperationsDevice.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherConstants.h>
#include <DeviceBlasWrapper.h>

namespace dftfe
{
  template <typename NumberType, dftfe::utils::MemorySpace memorySpace>
  void
  computeRhoFromPSI(
    const dftfe::utils::MemoryStorage<NumberType, memorySpace> *X,
    const dftfe::utils::MemoryStorage<NumberType, memorySpace> *XFrac,
    const unsigned int                      totalNumWaveFunctions,
    const unsigned int                      Nfr,
    const unsigned int                      numLocalDofs,
    const std::vector<std::vector<double>> &eigenValues,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    std::unique_ptr<
      dftfe::basis::FEBasisOperations<NumberType, double, memorySpace>>
      &                                            basisOperationsPtr,
    const unsigned int                             matrixFreeDofhandlerIndex,
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
    const bool                                     use2pPlusOneGLQuad)
  {
    int this_process;
    MPI_Comm_rank(mpiCommParent, &this_process);
    dftfe::utils::deviceSynchronize();
    MPI_Barrier(mpiCommParent);
    double             computeRho_time = MPI_Wtime();
    const unsigned int numKPoints      = kPointWeights.size();

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

    std::vector<dftfe::utils::MemoryStorage<NumberType, memorySpace>>
      wfcQuadPointData(numSpinComponents);
    std::vector<dftfe::utils::MemoryStorage<NumberType, memorySpace>>
      gradWfcQuadPointData(numSpinComponents);
    std::vector<dftfe::utils::MemoryStorage<double, memorySpace>>
      rhoWfcContributions(numSpinComponents);
    std::vector<dftfe::utils::MemoryStorage<double, memorySpace>>
      gradRhoWfcContributions(numSpinComponents);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      rhoHost;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      gradRhoHost;
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, memorySpace> rho;
    dftfe::utils::MemoryStorage<double, memorySpace> gradRho;
#else
    auto &rho             = rhoHost;
    auto &gradRho         = gradRhoHost;
#endif

    rho.resize(totalLocallyOwnedCells * numQuadPoints * numSpinComponents, 0.0);
    for (unsigned int spinIndex = 0; spinIndex < numSpinComponents; ++spinIndex)
      {
        wfcQuadPointData[spinIndex].resize(cellsBlockSize * numQuadPoints *
                                             BVec,
                                           zero);

        if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
          rhoWfcContributions[spinIndex].resize(cellsBlockSize * numQuadPoints *
                                                  BVec,
                                                0.0);
      }
    if (isEvaluateGradRho)
      {
        gradRho.resize(totalLocallyOwnedCells * numQuadPoints * 3 *
                         numSpinComponents,
                       0.0);
        for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
             ++spinIndex)
          {
            gradWfcQuadPointData[spinIndex].resize(cellsBlockSize *
                                                     numQuadPoints * BVec * 3,
                                                   zero);
            if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
              gradRhoWfcContributions[spinIndex].resize(
                cellsBlockSize * numQuadPoints * BVec * 3, 0.0);
          }
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

    std::vector<dftfe::linearAlgebra::MultiVector<NumberType, memorySpace>>
      flattenedArrayBlock(numSpinComponents);
    for (unsigned int spinIndex = 0; spinIndex < numSpinComponents; ++spinIndex)
      basisOperationsPtr->createMultiVector(matrixFreeDofhandlerIndex,
                                            BVec,
                                            flattenedArrayBlock[spinIndex]);

    dftfe::utils::MemoryStorage<NumberType, memorySpace> cellWaveFunctionMatrix(
      cellsBlockSize * numNodesPerElement * BVec);

    for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
      {
        for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
             ++spinIndex)
          {
            wfcQuadPointData[spinIndex].setValue(zero);
            gradWfcQuadPointData[spinIndex].setValue(zero);
            rhoWfcContributions[spinIndex].setValue(0.0);
            gradRhoWfcContributions[spinIndex].setValue(0.0);
          }
        for (unsigned int jvec = 0; jvec < totalNumWaveFunctions; jvec += BVec)
          {
            const unsigned int currentBlockSize =
              std::min(BVec, totalNumWaveFunctions - jvec);
            for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
                 ++spinIndex)
              if (currentBlockSize !=
                  flattenedArrayBlock[spinIndex].numVectors())
                basisOperationsPtr->createMultiVector(
                  matrixFreeDofhandlerIndex,
                  currentBlockSize,
                  flattenedArrayBlock[spinIndex]);
            if ((jvec + currentBlockSize) <=
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                (jvec + currentBlockSize) >
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
              {
                for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
                     ++spinIndex)
                  if (spectrumSplit)
                    {
                      partialOccupVecHost[spinIndex].setValue(
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
                for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
                     ++spinIndex)
                  partialOccupVec[spinIndex].copyFrom(
                    partialOccupVecHost[spinIndex]);
#endif
                for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
                     ++spinIndex)
                  if (memorySpace == dftfe::utils::MemorySpace::HOST)
                    for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                  std::memcpy(flattenedArrayBlock[spinIndex]
                          .data()+iNode * currentBlockSize,X->data()+numLocalDofs * totalNumWaveFunctions *
                                        (numSpinComponents * kPoint +
                                         spinIndex) +
                                      iNode * totalNumWaveFunctions + jvec,currentBlockSize*sizeof(NumberType));
#if defined(DFTFE_WITH_DEVICE)
                  else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
                    dftfe::utils::deviceKernelsGeneric::
                      stridedCopyToBlockConstantStride(
                        currentBlockSize,
                        totalNumWaveFunctions,
                        numLocalDofs,
                        jvec,
                        X->data() + numLocalDofs * totalNumWaveFunctions *
                                      (numSpinComponents * kPoint + spinIndex),
                        flattenedArrayBlock[spinIndex].begin());
#endif


                const unsigned int d_quadratureIndex =
                  use2pPlusOneGLQuad ? 2 : 0;
                dftfe::basis::UpdateFlags updateFlags =
                  dftfe::basis::update_values | dftfe::basis::update_gradients;
                basisOperationsPtr->reinit(currentBlockSize,
                                           cellsBlockSize,
                                           matrixFreeDofhandlerIndex,
                                           d_quadratureIndex,
                                           updateFlags);


                for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
                     ++spinIndex)
                  {
                    flattenedArrayBlock[spinIndex].updateGhostValues();
                    basisOperationsPtr->distribute(
                      flattenedArrayBlock[spinIndex]);
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
                            flattenedArrayBlock[spinIndex],
                            wfcQuadPointData[spinIndex].data(),
                            isEvaluateGradRho ?
                              gradWfcQuadPointData[spinIndex].data() :
                              NULL,
                            std::pair<unsigned int, unsigned int>(
                              startingCellId,
                              startingCellId + currentCellsBlockSize));

                        for (unsigned int spinIndex = 0;
                             spinIndex < numSpinComponents;
                             ++spinIndex)
                          computeRhoGradRhoFromInterpolatedValues(
                            basisOperationsPtr,
                            std::pair<unsigned int, unsigned int>(
                              startingCellId,
                              startingCellId + currentCellsBlockSize),
                            std::pair<unsigned int, unsigned int>(
                              jvec, jvec + currentBlockSize),
                            partialOccupVec[spinIndex].data(),
                            wfcQuadPointData[spinIndex].data(),
                            gradWfcQuadPointData[spinIndex].data(),
                            rhoWfcContributions[spinIndex].data(),
                            gradRhoWfcContributions[spinIndex].data(),
                            rho.data() + spinIndex * totalLocallyOwnedCells *
                                           numQuadPoints,
                            gradRho.data() + spinIndex *
                                               totalLocallyOwnedCells *
                                               numQuadPoints * 3,
                            isEvaluateGradRho);
                      } // non-trivial cell block check
                  }     // cells block loop
              }
          }

        if (spectrumSplit)
          for (unsigned int jvec = 0; jvec < Nfr; jvec += BVec)
            {
              const unsigned int currentBlockSize = std::min(BVec, Nfr - jvec);
              for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
                   ++spinIndex)
                if (currentBlockSize !=
                    flattenedArrayBlock[spinIndex].numVectors())
                  basisOperationsPtr->createMultiVector(
                    matrixFreeDofhandlerIndex,
                    currentBlockSize,
                    flattenedArrayBlock[spinIndex]);
              if ((jvec + totalNumWaveFunctions - Nfr + currentBlockSize) <=
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                  (jvec + totalNumWaveFunctions - Nfr + currentBlockSize) >
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  for (unsigned int spinIndex = 0;
                       spinIndex < numSpinComponents;
                       ++spinIndex)
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
                                            (totalNumWaveFunctions - Nfr) +
                                            jvec + iEigenVec] >
                                fermiEnergyConstraintMag)
                              *(partialOccupVecHost[spinIndex].begin() +
                                iEigenVec) =
                                -kPointWeights[kPoint] * spinPolarizedFactor;
                            else
                              *(partialOccupVecHost[spinIndex].begin() +
                                iEigenVec) = 0;
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
                              (dftUtils::getPartialOccupancy(
                                 eigenValues[kPoint]
                                            [totalNumWaveFunctions * spinIndex +
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
                  for (unsigned int spinIndex = 0;
                       spinIndex < numSpinComponents;
                       ++spinIndex)
                    {
                      partialOccupVec[spinIndex].resize(
                        partialOccupVecHost[spinIndex].size());
                      partialOccupVec[spinIndex].copyFrom(
                        partialOccupVecHost[spinIndex]);
                    }
#endif
                  for (unsigned int spinIndex = 0;
                       spinIndex < numSpinComponents;
                       ++spinIndex)
                    if (memorySpace == dftfe::utils::MemorySpace::HOST)
                      for (unsigned int iNode = 0; iNode < numLocalDofs;
                           ++iNode)
                           std::memcpy(flattenedArrayBlock[spinIndex]
                            .data()+iNode * currentBlockSize,XFrac->data()+numLocalDofs * Nfr *
                                              (numSpinComponents * kPoint +
                                               spinIndex) +
                                            iNode * Nfr + jvec,currentBlockSize*sizeof(NumberType));
                        // for (unsigned int iWave = 0; iWave < currentBlockSize;
                        //      ++iWave)
                        //   flattenedArrayBlock[spinIndex]
                        //     .data()[iNode * currentBlockSize + iWave] =
                        //     (XFrac->data())[numLocalDofs * Nfr *
                        //                       (numSpinComponents * kPoint +
                        //                        spinIndex) +
                        //                     iNode * Nfr + jvec + iWave];
#if defined(DFTFE_WITH_DEVICE)
                    else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
                      dftfe::utils::deviceKernelsGeneric::
                        stridedCopyToBlockConstantStride(
                          currentBlockSize,
                          Nfr,
                          numLocalDofs,
                          jvec,
                          XFrac->data() +
                            numLocalDofs * Nfr *
                              (numSpinComponents * kPoint + spinIndex),
                          flattenedArrayBlock[spinIndex].begin());
#endif
                  const unsigned int d_quadratureIndex =
                    use2pPlusOneGLQuad ? 2 : 0;
                  dftfe::basis::UpdateFlags updateFlags =
                    dftfe::basis::update_values |
                    dftfe::basis::update_gradients;
                  basisOperationsPtr->reinit(currentBlockSize,
                                             cellsBlockSize,
                                             matrixFreeDofhandlerIndex,
                                             d_quadratureIndex,
                                             updateFlags);


                  for (unsigned int spinIndex = 0;
                       spinIndex < numSpinComponents;
                       ++spinIndex)
                    {
                      flattenedArrayBlock[spinIndex].updateGhostValues();
                      basisOperationsPtr->distribute(
                        flattenedArrayBlock[spinIndex]);
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
                              flattenedArrayBlock[spinIndex],
                              wfcQuadPointData[spinIndex].data(),
                              isEvaluateGradRho ?
                                gradWfcQuadPointData[spinIndex].data() :
                                NULL,
                              std::pair<unsigned int, unsigned int>(
                                startingCellId,
                                startingCellId + currentCellsBlockSize));

                          for (unsigned int spinIndex = 0;
                               spinIndex < numSpinComponents;
                               ++spinIndex)
                            computeRhoGradRhoFromInterpolatedValues(
                              basisOperationsPtr,
                              std::pair<unsigned int, unsigned int>(
                                startingCellId,
                                startingCellId + currentCellsBlockSize),
                              std::pair<unsigned int, unsigned int>(
                                jvec, jvec + currentBlockSize),
                              partialOccupVec[spinIndex].data(),
                              wfcQuadPointData[spinIndex].data(),
                              gradWfcQuadPointData[spinIndex].data(),
                              rhoWfcContributions[spinIndex].data(),
                              gradRhoWfcContributions[spinIndex].data(),
                              rho.data() + spinIndex * totalLocallyOwnedCells *
                                             numQuadPoints,
                              gradRho.data() + spinIndex *
                                                 totalLocallyOwnedCells *
                                                 numQuadPoints * 3,
                              isEvaluateGradRho);
                        } // non-tivial cells block
                    }     // cells block loop
                }
            } // spectrum split block
      }
#if defined(DFTFE_WITH_DEVICE)
    rhoHost.resize(rho.size());
    rhoHost.copyFrom(rho);
    if (isEvaluateGradRho)
      {
        gradRhoHost.resize(gradRho.size());
        gradRhoHost.copyFrom(gradRho);
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
      }
    dftfe::utils::deviceSynchronize();
    MPI_Barrier(mpiCommParent);
    double computeRho_time2 = MPI_Wtime();

    unsigned int iElem = 0;
    auto         cell  = dofHandler.begin_active();
    auto         endc  = dofHandler.end();
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
                  const double rho0 = rhoHost[iElem * numQuadPoints + q];
                  const double rho1 = rhoHost[totalLocallyOwnedCells * numQuadPoints +
                            iElem * numQuadPoints + q];
                  tempRhoQuadsSP[2 * q + 0] = rho0;

                  tempRhoQuadsSP[2 * q + 1] = rho1;
                  tempRhoQuads[q] = rho0+rho1;
                }

              if (isEvaluateGradRho)
                for (unsigned int q = 0; q < numQuadPoints; ++q)
                  {
                    const double gradRho0x = gradRhoHost[iElem * numQuadPoints * 3 + 3*q];
                    const double gradRho0y = gradRhoHost[iElem * numQuadPoints * 3 + 3*q+1];
                    const double gradRho0z = gradRhoHost[iElem * numQuadPoints * 3 + 3*q+2];
                    const double gradRho1x = gradRhoHost[totalLocallyOwnedCells * numQuadPoints * 3+iElem * numQuadPoints * 3 + 3*q];
                    const double gradRho1y = gradRhoHost[totalLocallyOwnedCells * numQuadPoints * 3+iElem * numQuadPoints * 3 + 3*q+1];
                    const double gradRho1z = gradRhoHost[totalLocallyOwnedCells * numQuadPoints * 3+iElem * numQuadPoints * 3 + 3*q+2];
                    tempGradRhoQuadsSP[6 * q + 0] = gradRho0x;
                    tempGradRhoQuadsSP[6 * q + 1] = gradRho0y;
                    tempGradRhoQuadsSP[6 * q + 2] = gradRho0z;
                    tempGradRhoQuadsSP[6 * q + 3] = gradRho1x;
                    tempGradRhoQuadsSP[6 * q + 4] = gradRho1y;
                    tempGradRhoQuadsSP[6 * q + 5] = gradRho1z;
                    tempGradRhoQuads[3 * q] = gradRho0x + gradRho1x;
                    tempGradRhoQuads[3 * q + 1] = gradRho0y + gradRho1y;
                    tempGradRhoQuads[3 * q + 2] = gradRho0z + gradRho1z;
                  }
            }
          else
            {
              std::memcpy(tempRhoQuads.data(),
                          rhoHost.data() + iElem * numQuadPoints,
                          numQuadPoints * sizeof(double));

              if (isEvaluateGradRho)
                std::memcpy(tempGradRhoQuads.data(),
                            gradRhoHost.data() + iElem * numQuadPoints * 3,
                            3 * numQuadPoints * sizeof(double));
            }
          iElem++;
        }
    dftfe::utils::deviceSynchronize();
    MPI_Barrier(mpiCommParent);
    computeRho_time  = MPI_Wtime() - computeRho_time;
    computeRho_time2 = MPI_Wtime() - computeRho_time2;

    if (this_process == 0 && dftParams.verbosity >= 2)
      if (memorySpace == dftfe::utils::MemorySpace::HOST)
        std::cout << "Time for compute rho on CPU: " << computeRho_time
                  << std::endl;
      else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
        std::cout << "Time for compute rho on Device: " << computeRho_time
                  << " " << computeRho_time2 << std::endl;
  }
  template <typename NumberType>
  void
  computeRhoGradRhoFromInterpolatedValues(
    std::unique_ptr<
      dftfe::basis::
        FEBasisOperations<NumberType, double, dftfe::utils::MemorySpace::HOST>>
      &                                         basisOperationsPtr,
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
    const unsigned int nQuadsPerCell    = basisOperationsPtr->d_nQuadsPerCell;
    const unsigned int nCells           = basisOperationsPtr->d_nCells;
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
#if defined(DFTFE_WITH_DEVICE)
  template void
  computeRhoFromPSI(
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::DEVICE> *X,
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::DEVICE> *XFrac,
    const unsigned int                      totalNumWaveFunctions,
    const unsigned int                      Nfr,
    const unsigned int                      numLocalDofs,
    const std::vector<std::vector<double>> &eigenValues,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    std::unique_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::DEVICE>>
      &                                            basisOperationsPtrDevice,
    const unsigned int                             matrixFreeDofhandlerIndex,
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
    const bool                                     use2pPlusOneGLQuad);
#endif

  template void
  computeRhoFromPSI(
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::HOST> *X,
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::HOST> *XFrac,
    const unsigned int                      totalNumWaveFunctions,
    const unsigned int                      Nfr,
    const unsigned int                      numLocalDofs,
    const std::vector<std::vector<double>> &eigenValues,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    std::unique_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::HOST>>
      &                                            basisOperationsPtr,
    const unsigned int                             matrixFreeDofhandlerIndex,
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
    const bool                                     use2pPlusOneGLQuad);
} // namespace dftfe
