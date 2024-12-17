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

#include <constants.h>
#include <computeAuxProjectedDensityMatrixFromPSI.h>
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
  computeAuxProjectedDensityMatrixFromPSI(
    const dftfe::utils::MemoryStorage<NumberType, memorySpace> &X,
    const unsigned int                      totalNumWaveFunctions,
    const std::vector<std::vector<double>> &eigenValues,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<NumberType, double, memorySpace>>
      &basisOperationsPtr,
    std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      &                            BLASWrapperPtr,
    const unsigned int             matrixFreeDofhandlerIndex,
    const unsigned int             quadratureIndex,
    const std::vector<double> &    kPointWeights,
    AuxDensityMatrix<memorySpace> &auxDensityMatrixRepresentation,
    const MPI_Comm &               mpiCommParent,
    const MPI_Comm &               domainComm,
    const MPI_Comm &               interpoolcomm,
    const MPI_Comm &               interBandGroupComm,
    const dftParameters &          dftParams)
  {
    int this_process;
    MPI_Comm_rank(mpiCommParent, &this_process);
#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(mpiCommParent);
    double             project_time = MPI_Wtime();
    const unsigned int numKPoints   = kPointWeights.size();
    const unsigned int numLocalDofs = basisOperationsPtr->nOwnedDofs();
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
    dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::HOST>
      wfcQuadPointDataHost;


    wfcQuadPointData.resize(cellsBlockSize * numQuadPoints * BVec, zero);
    wfcQuadPointDataHost.resize(cellsBlockSize * numQuadPoints * BVec, zero);

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      partialOccupVecHost(BVec, 0.0);
#if defined(DFTFE_WITH_DEVICE)
    dftfe::utils::MemoryStorage<double, memorySpace> partialOccupVec(
      partialOccupVecHost.size());
#else
    auto &partialOccupVec = partialOccupVecHost;
#endif

    dftfe::linearAlgebra::MultiVector<NumberType, memorySpace>
      *flattenedArrayBlock;

    basisOperationsPtr->reinit(BVec, cellsBlockSize, quadratureIndex, false);

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      allQuadPointsHost = basisOperationsPtr->quadPoints();

    dftfe::utils::MemoryStorage<NumberType, memorySpace>
      allQuadWeightsMemorySpace = basisOperationsPtr->JxW();


    dftfe::utils::MemoryStorage<NumberType, memorySpace> allQuadWeightsHost;
    allQuadWeightsHost.copyFrom(allQuadWeightsMemorySpace);

    //
    // compute S matrix of aux basis
    //
    for (int iblock = 0; iblock < (numCellBlocks + 1); iblock++)
      {
        const unsigned int currentCellsBlockSize =
          (iblock == numCellBlocks) ? remCellBlockSize : cellsBlockSize;
        if (currentCellsBlockSize > 0)
          {
            const unsigned int startingCellId = iblock * cellsBlockSize;

            std::vector<double> quadPointsBatch(currentCellsBlockSize *
                                                numQuadPoints * 3);
            std::vector<double> quadWeightsBatch(currentCellsBlockSize *
                                                 numQuadPoints);
            for (unsigned int iQuad = 0;
                 iQuad < currentCellsBlockSize * numQuadPoints;
                 ++iQuad)
              {
                for (unsigned int idim = 0; idim < 3; ++idim)
                  quadPointsBatch[3 * iQuad + idim] =
                    allQuadPointsHost[startingCellId * numQuadPoints * 3 +
                                      3 * iQuad + idim];
                quadWeightsBatch[iQuad] = std::real(
                  allQuadWeightsHost[startingCellId * numQuadPoints + iQuad]);
              }

            auxDensityMatrixRepresentation.evalOverlapMatrixStart(
              quadPointsBatch, quadWeightsBatch);
          } // non-trivial cell block check
      }     // cells block loop

    auxDensityMatrixRepresentation.evalOverlapMatrixEnd(domainComm);

    std::unordered_map<std::string, std::vector<NumberType>>
      densityMatrixProjectionInputsDataType;
    std::unordered_map<std::string, std::vector<double>>
                             densityMatrixProjectionInputsRealType;
    std::vector<NumberType> &wfcQuadPointDataBatchHost =
      densityMatrixProjectionInputsDataType["psiFunc"];
    std::vector<double> &quadPointsBatch =
      densityMatrixProjectionInputsRealType["quadpts"];
    std::vector<double> &quadWeightsBatch =
      densityMatrixProjectionInputsRealType["quadWt"];
    std::vector<double> &fValuesBatch =
      densityMatrixProjectionInputsRealType["fValues"];

    for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
      for (unsigned int spinIndex = 0; spinIndex < numSpinComponents;
           ++spinIndex)
        {
          wfcQuadPointData.setValue(zero);
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
                            *(partialOccupVecHost.begin() + iEigenVec) = 0;
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
                              eigenValues[kPoint]
                                         [totalNumWaveFunctions * spinIndex +
                                          jvec + iEigenVec],
                              fermiEnergy,
                              C_kb,
                              dftParams.TVal) *
                            kPointWeights[kPoint] * spinPolarizedFactor;
                        }
                    }
#if defined(DFTFE_WITH_DEVICE)
                  partialOccupVec.copyFrom(partialOccupVecHost);
#endif
                  partialOccupVecHost.copyTo(fValuesBatch);
                  if (memorySpace == dftfe::utils::MemorySpace::HOST)
                    for (unsigned int iNode = 0; iNode < numLocalDofs; ++iNode)
                      std::memcpy(flattenedArrayBlock->data() +
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


                          quadPointsBatch.resize(currentCellsBlockSize *
                                                 numQuadPoints * 3);
                          quadWeightsBatch.resize(currentCellsBlockSize *
                                                  numQuadPoints);
                          for (unsigned int iQuad = 0;
                               iQuad < currentCellsBlockSize * numQuadPoints;
                               ++iQuad)
                            {
                              for (unsigned int idim = 0; idim < 3; ++idim)
                                quadPointsBatch[3 * iQuad + idim] =
                                  allQuadPointsHost[startingCellId *
                                                      numQuadPoints * 3 +
                                                    3 * iQuad + idim];
                              quadWeightsBatch[iQuad] =
                                std::real(allQuadWeightsHost[startingCellId *
                                                               numQuadPoints +
                                                             iQuad]);
                            }

                          basisOperationsPtr->interpolateKernel(
                            *(flattenedArrayBlock),
                            wfcQuadPointData.data(),
                            NULL,
                            std::pair<unsigned int, unsigned int>(
                              startingCellId,
                              startingCellId + currentCellsBlockSize));


                          wfcQuadPointDataBatchHost.resize(
                            currentCellsBlockSize * numQuadPoints *
                            currentBlockSize);
                          wfcQuadPointData.copyTo(wfcQuadPointDataBatchHost);

                          auxDensityMatrixRepresentation
                            .projectDensityMatrixStart(
                              densityMatrixProjectionInputsDataType,
                              densityMatrixProjectionInputsRealType,
                              spinIndex);

                        } // non-trivial cell block check
                    }     // cells block loop
                }
            }
        } // spin loop



    int size;
    MPI_Comm_size(interpoolcomm, &size);
    if (size > 1)
      {
        auxDensityMatrixRepresentation.projectDensityMatrixEnd(interpoolcomm);
      }
    MPI_Comm_size(interBandGroupComm, &size);
    if (size > 1)
      {
        auxDensityMatrixRepresentation.projectDensityMatrixEnd(
          interBandGroupComm);
      }
    auxDensityMatrixRepresentation.projectDensityMatrixEnd(domainComm);

#if defined(DFTFE_WITH_DEVICE)
    if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
      dftfe::utils::deviceSynchronize();
#endif
    MPI_Barrier(mpiCommParent);
    project_time = MPI_Wtime() - project_time;

    if (this_process == 0 && dftParams.verbosity >= 2)
      if (memorySpace == dftfe::utils::MemorySpace::HOST)
        std::cout << "Time for project on CPU: " << project_time << std::endl;
      else if (memorySpace == dftfe::utils::MemorySpace::DEVICE)
        std::cout << "Time for project on Device: " << project_time
                  << std::endl;
  }


#if defined(DFTFE_WITH_DEVICE)
  template void
  computeAuxProjectedDensityMatrixFromPSI(
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::DEVICE> &X,
    const unsigned int                      totalNumWaveFunctions,
    const std::vector<std::vector<double>> &eigenValues,
    const double                            fermiEnergy,
    const double                            fermiEnergyUp,
    const double                            fermiEnergyDown,
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
    AuxDensityMatrix<dftfe::utils::MemorySpace::DEVICE>
      &                  auxDensityMatrixRepresentation,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     domainComm,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams);
#endif
  template void
  computeAuxProjectedDensityMatrixFromPSI(
    const dftfe::utils::MemoryStorage<dataTypes::number,
                                      dftfe::utils::MemorySpace::HOST> &X,
    const unsigned int                      totalNumWaveFunctions,
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
    const std::vector<double> &kPointWeights,
    AuxDensityMatrix<dftfe::utils::MemorySpace::HOST>
      &                  auxDensityMatrixRepresentation,
    const MPI_Comm &     mpiCommParent,
    const MPI_Comm &     domainComm,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interBandGroupComm,
    const dftParameters &dftParams);
} // namespace dftfe
