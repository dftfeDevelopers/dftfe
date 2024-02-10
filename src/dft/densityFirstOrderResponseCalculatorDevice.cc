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

#include <constants.h>
#include <densityFirstOrderResponseCalculator.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <deviceKernelsGeneric.h>
#include <linearAlgebraOperationsDevice.h>
#include <MemoryStorage.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DataTypeOverloads.h>
#include <DeviceKernelLauncherConstants.h>
#include <DeviceBlasWrapper.h>

namespace dftfe
{
  namespace
  {
    __global__ void
    computeRhoResponseFromInterpolatedValues(const unsigned int numberEntries,
                                             double *           XQuads,
                                             double *           XPrimeQuads)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const double psi      = XQuads[index];
          const double psiPrime = XPrimeQuads[index];
          XPrimeQuads[index]    = psi * psiPrime;
          XQuads[index]         = psi * psi;
        }
    }

    __global__ void
    computeRhoResponseFromInterpolatedValues(
      const unsigned int                 numberEntries,
      dftfe::utils::deviceDoubleComplex *XQuads,
      dftfe::utils::deviceDoubleComplex *XPrimeQuads)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::utils::deviceDoubleComplex psi      = XQuads[index];
          const dftfe::utils::deviceDoubleComplex psiPrime = XPrimeQuads[index];
          dftfe::utils::copyValue(XPrimeQuads + index,
                                  psi.x * psiPrime.x + psi.y * psiPrime.y);
          dftfe::utils::copyValue(XQuads + index,
                                  psi.x * psi.x + psi.y * psi.y);
        }
    }

    __global__ void
    computeRhoResponseFromInterpolatedValues(const unsigned int numberEntries,
                                             float *            XQuads,
                                             float *            XPrimeQuads)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const float psi      = XQuads[index];
          const float psiPrime = XPrimeQuads[index];
          XPrimeQuads[index]   = psi * psiPrime;
          XQuads[index]        = psi * psi;
        }
    }

    __global__ void
    computeRhoResponseFromInterpolatedValues(
      const unsigned int                numberEntries,
      dftfe::utils::deviceFloatComplex *XQuads,
      dftfe::utils::deviceFloatComplex *XPrimeQuads)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::utils::deviceFloatComplex psi      = XQuads[index];
          const dftfe::utils::deviceFloatComplex psiPrime = XPrimeQuads[index];
          dftfe::utils::copyValue(XPrimeQuads + index,
                                  psi.x * psiPrime.x + psi.y * psiPrime.y);
          dftfe::utils::copyValue(XQuads + index,
                                  psi.x * psi.x + psi.y * psi.y);
        }
    }
  } // namespace

  template <typename NumberType, typename NumberTypeLowPrec>
  void
  computeRhoFirstOrderResponseDevice(
    const NumberType *                             X,
    const NumberType *                             XPrime,
    const std::vector<std::vector<double>> &       densityMatDerFermiEnergy,
    const unsigned int                             totalNumWaveFunctions,
    const unsigned int                             numLocalDofs,
    operatorDFTDeviceClass &                       operatorMatrix,
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
    const dftParameters &dftParams,
    const std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      BLASWrapperPtr)
  {
    int this_process;
    MPI_Comm_rank(mpiCommParent, &this_process);
    dftfe::utils::deviceSynchronize();
    MPI_Barrier(mpiCommParent);
    double             device_time = MPI_Wtime();
    const unsigned int numKPoints  = kPointWeights.size();

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
      std::min(dftParams.chebyWfcBlockSize, totalNumWaveFunctions);

    const double spinPolarizedFactor =
      (dftParams.spinPolarized == 1) ? 1.0 : 2.0;

    const NumberTypeLowPrec zero                = 0;
    const NumberTypeLowPrec one                 = 1.0;
    const NumberTypeLowPrec scalarCoeffAlphaRho = 1.0;
    const NumberTypeLowPrec scalarCoeffBetaRho  = 1.0;

    const unsigned int cellsBlockSize = 50;
    const unsigned int numCellBlocks  = totalLocallyOwnedCells / cellsBlockSize;
    const unsigned int remCellBlockSize =
      totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;

    dftfe::utils::MemoryStorage<NumberTypeLowPrec,
                                dftfe::utils::MemorySpace::DEVICE>
      rhoResponseContributionHamDevice(totalLocallyOwnedCells * numQuadPoints,
                                       zero);

    dftfe::utils::MemoryStorage<NumberTypeLowPrec,
                                dftfe::utils::MemorySpace::DEVICE>
      rhoResponseContributionFermiEnergyDevice(totalLocallyOwnedCells *
                                                 numQuadPoints,
                                               zero);

    dftfe::utils::MemoryStorage<NumberTypeLowPrec,
                                dftfe::utils::MemorySpace::HOST>
      rhoResponseContributionHamHost(totalLocallyOwnedCells * numQuadPoints,
                                     zero);

    dftfe::utils::MemoryStorage<NumberTypeLowPrec,
                                dftfe::utils::MemorySpace::HOST>
      rhoResponseContributionFermiEnergyHost(totalLocallyOwnedCells *
                                               numQuadPoints,
                                             zero);

    std::vector<double> rhoResponseValuesHamFlattenedHost(
      totalLocallyOwnedCells * numQuadPoints, 0.0);
    std::vector<double> rhoResponseValuesFermiEnergyFlattenedHost(
      totalLocallyOwnedCells * numQuadPoints, 0.0);

    std::vector<double> rhoResponseValuesSpinPolarizedHamFlattenedHost(
      totalLocallyOwnedCells * numQuadPoints * 2, 0.0);
    std::vector<double> rhoResponseValuesSpinPolarizedFermiEnergyFlattenedHost(
      totalLocallyOwnedCells * numQuadPoints * 2, 0.0);

    dftfe::utils::MemoryStorage<NumberTypeLowPrec,
                                dftfe::utils::MemorySpace::DEVICE>
      XQuadsDevice(cellsBlockSize * numQuadPoints * BVec, zero);

    dftfe::utils::MemoryStorage<NumberTypeLowPrec,
                                dftfe::utils::MemorySpace::DEVICE>
      XPrimeQuadsDevice(cellsBlockSize * numQuadPoints * BVec, zero);
    dftfe::utils::MemoryStorage<NumberTypeLowPrec,
                                dftfe::utils::MemorySpace::DEVICE>
      onesVecDevice(BVec, one);

    dftfe::utils::MemoryStorage<NumberTypeLowPrec,
                                dftfe::utils::MemorySpace::HOST>
      densityMatDerFermiEnergyVec(BVec, zero);
    dftfe::utils::MemoryStorage<NumberTypeLowPrec,
                                dftfe::utils::MemorySpace::DEVICE>
      densityMatDerFermiEnergyVecDevice(BVec, zero);

    distributedDeviceVec<NumberType> &deviceFlattenedArrayXBlock =
      operatorMatrix.getParallelChebyBlockVectorDevice();

    distributedDeviceVec<NumberType> &deviceFlattenedArrayXPrimeBlock =
      operatorMatrix.getParallelChebyBlockVector2Device();


    dftfe::utils::MemoryStorage<NumberTypeLowPrec,
                                dftfe::utils::MemorySpace::DEVICE>
      cellWaveFunctionMatrix(cellsBlockSize * numNodesPerElement * BVec, zero);

    dftfe::utils::MemoryStorage<NumberTypeLowPrec,
                                dftfe::utils::MemorySpace::DEVICE>
      shapeFunctionValuesTransposedDevice(numNodesPerElement * numQuadPoints,
                                          zero);

    shapeFunctionValuesTransposedDevice.setValue(zero);


    BLASWrapperPtr->copyValueType1ArrToValueType2Arr(
      numNodesPerElement * numQuadPoints,
      (operatorMatrix.getShapeFunctionValuesTransposed(true)).begin(),
      shapeFunctionValuesTransposedDevice.begin());

    for (unsigned int spinIndex = 0; spinIndex < (1 + dftParams.spinPolarized);
         ++spinIndex)
      {
        for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
          {
            rhoResponseContributionHamDevice.setValue(zero);
            rhoResponseContributionFermiEnergyDevice.setValue(zero);

            for (unsigned int jvec = 0; jvec < totalNumWaveFunctions;
                 jvec += BVec)
              {
                if ((jvec + BVec) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + BVec) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    for (unsigned int iEigenVec = 0; iEigenVec < BVec;
                         ++iEigenVec)
                      {
                        *(densityMatDerFermiEnergyVec.begin() + iEigenVec) =
                          densityMatDerFermiEnergy
                            [(dftParams.spinPolarized + 1) * kPoint + spinIndex]
                            [jvec + iEigenVec];
                      }

                    densityMatDerFermiEnergyVec
                      .template copyTo<dftfe::utils::MemorySpace::DEVICE>(
                        densityMatDerFermiEnergyVecDevice);

                    BLASWrapperPtr->stridedCopyToBlockConstantStride(
                      BVec,
                      totalNumWaveFunctions,
                      numLocalDofs,
                      jvec,
                      X +
                        numLocalDofs * totalNumWaveFunctions *
                          ((dftParams.spinPolarized + 1) * kPoint + spinIndex),
                      deviceFlattenedArrayXBlock.begin());

                    deviceFlattenedArrayXBlock.updateGhostValues();

                    (operatorMatrix.getOverloadedConstraintMatrix())
                      ->distribute(deviceFlattenedArrayXBlock, BVec);


                    BLASWrapperPtr->stridedCopyToBlockConstantStride(
                      BVec,
                      totalNumWaveFunctions,
                      numLocalDofs,
                      jvec,
                      XPrime +
                        numLocalDofs * totalNumWaveFunctions *
                          ((dftParams.spinPolarized + 1) * kPoint + spinIndex),
                      deviceFlattenedArrayXPrimeBlock.begin());

                    deviceFlattenedArrayXPrimeBlock.updateGhostValues();

                    (operatorMatrix.getOverloadedConstraintMatrix())
                      ->distribute(deviceFlattenedArrayXPrimeBlock, BVec);


                    for (int iblock = 0; iblock < (numCellBlocks + 1); iblock++)
                      {
                        const unsigned int currentCellsBlockSize =
                          (iblock == numCellBlocks) ? remCellBlockSize :
                                                      cellsBlockSize;
                        if (currentCellsBlockSize > 0)
                          {
                            const unsigned int startingCellId =
                              iblock * cellsBlockSize;



                            BLASWrapperPtr->stridedCopyToBlock(
                              BVec,
                              currentCellsBlockSize * numNodesPerElement,
                              deviceFlattenedArrayXBlock.begin(),
                              cellWaveFunctionMatrix.begin(),
                              (operatorMatrix
                                 .getFlattenedArrayCellLocalProcIndexIdMap())
                                  .begin() +
                                startingCellId * numNodesPerElement);

                            NumberTypeLowPrec scalarCoeffAlpha = 1.0;
                            NumberTypeLowPrec scalarCoeffBeta  = 0.0;
                            int strideA = BVec * numNodesPerElement;
                            int strideB = 0;
                            int strideC = BVec * numQuadPoints;



                            BLASWrapperPtr->xgemmStridedBatched(
                              'N',
                              'N',
                              BVec,
                              numQuadPoints,
                              numNodesPerElement,
                              &scalarCoeffAlpha,
                              cellWaveFunctionMatrix.begin(),
                              BVec,
                              strideA,
                              shapeFunctionValuesTransposedDevice.begin(),
                              numNodesPerElement,
                              strideB,
                              &scalarCoeffBeta,
                              XQuadsDevice.begin(),
                              BVec,
                              strideC,
                              currentCellsBlockSize);


                            BLASWrapperPtr->stridedCopyToBlock(
                              BVec,
                              currentCellsBlockSize * numNodesPerElement,
                              deviceFlattenedArrayXPrimeBlock.begin(),
                              cellWaveFunctionMatrix.begin(),
                              (operatorMatrix
                                 .getFlattenedArrayCellLocalProcIndexIdMap())
                                  .begin() +
                                startingCellId * numNodesPerElement);

                            BLASWrapperPtr->xgemmStridedBatched(
                              'N',
                              'N',
                              BVec,
                              numQuadPoints,
                              numNodesPerElement,
                              &scalarCoeffAlpha,
                              cellWaveFunctionMatrix.begin(),
                              BVec,
                              strideA,
                              shapeFunctionValuesTransposedDevice.begin(),
                              numNodesPerElement,
                              strideB,
                              &scalarCoeffBeta,
                              XPrimeQuadsDevice.begin(),
                              BVec,
                              strideC,
                              currentCellsBlockSize);


#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
                            computeRhoResponseFromInterpolatedValues<<<
                              (BVec + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                dftfe::utils::DEVICE_BLOCK_SIZE *
                                numQuadPoints * currentCellsBlockSize,
                              dftfe::utils::DEVICE_BLOCK_SIZE>>>(
                              BVec * numQuadPoints * currentCellsBlockSize,
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                XQuadsDevice.begin()),
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                XPrimeQuadsDevice.begin()));
#elif DFTFE_WITH_DEVICE_LANG_HIP
                            hipLaunchKernelGGL(
                              computeRhoResponseFromInterpolatedValues,
                              (BVec + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                dftfe::utils::DEVICE_BLOCK_SIZE *
                                numQuadPoints * currentCellsBlockSize,
                              dftfe::utils::DEVICE_BLOCK_SIZE,
                              0,
                              0,
                              BVec * numQuadPoints * currentCellsBlockSize,
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                XQuadsDevice.begin()),
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                XPrimeQuadsDevice.begin()));
#endif

                            BLASWrapperPtr->xgemm(
                              'N',
                              'N',
                              1,
                              currentCellsBlockSize * numQuadPoints,
                              BVec,
                              &scalarCoeffAlphaRho,
                              onesVecDevice.begin(),
                              1,
                              XPrimeQuadsDevice.begin(),
                              BVec,
                              &scalarCoeffBetaRho,
                              rhoResponseContributionHamDevice.begin() +
                                startingCellId * numQuadPoints,
                              1);

                            BLASWrapperPtr->xgemm(
                              'N',
                              'N',
                              1,
                              currentCellsBlockSize * numQuadPoints,
                              BVec,
                              &scalarCoeffAlphaRho,
                              densityMatDerFermiEnergyVecDevice.begin(),
                              1,
                              XQuadsDevice.begin(),
                              BVec,
                              &scalarCoeffBetaRho,
                              rhoResponseContributionFermiEnergyDevice.begin() +
                                startingCellId * numQuadPoints,
                              1);

                          } // non-trivial cell block check
                      }     // cells block loop
                  }         // band parallelizatoin check
              }             // wave function block loop


            // do memcopy to host
            rhoResponseContributionHamDevice
              .template copyTo<dftfe::utils::MemorySpace::HOST>(
                rhoResponseContributionHamHost.begin(),
                totalLocallyOwnedCells * numQuadPoints,
                0,
                0);

            rhoResponseContributionFermiEnergyDevice
              .template copyTo<dftfe::utils::MemorySpace::HOST>(
                rhoResponseContributionFermiEnergyHost.begin(),
                totalLocallyOwnedCells * numQuadPoints,
                0,
                0);

            for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
              for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
                {
                  rhoResponseValuesHamFlattenedHost[icell * numQuadPoints +
                                                    iquad] +=
                    kPointWeights[kPoint] * spinPolarizedFactor *
                    dftfe::utils::realPart(
                      *(rhoResponseContributionHamHost.begin() +
                        icell * numQuadPoints + iquad));

                  rhoResponseValuesFermiEnergyFlattenedHost[icell *
                                                              numQuadPoints +
                                                            iquad] +=
                    kPointWeights[kPoint] * spinPolarizedFactor *
                    dftfe::utils::realPart(
                      *(rhoResponseContributionFermiEnergyHost.begin() +
                        icell * numQuadPoints + iquad));
                }


            if (dftParams.spinPolarized == 1)
              {
                for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
                  for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
                    {
                      rhoResponseValuesSpinPolarizedHamFlattenedHost
                        [icell * numQuadPoints * 2 + iquad * 2 + spinIndex] +=
                        kPointWeights[kPoint] *
                        dftfe::utils::realPart(
                          *(rhoResponseContributionHamHost.begin() +
                            icell * numQuadPoints + iquad));

                      rhoResponseValuesSpinPolarizedFermiEnergyFlattenedHost
                        [icell * numQuadPoints * 2 + iquad * 2 + spinIndex] +=
                        kPointWeights[kPoint] *
                        dftfe::utils::realPart(
                          *(rhoResponseContributionFermiEnergyHost.begin() +
                            icell * numQuadPoints + iquad));
                    }
              }


          } // kpoint loop
      }     // spin index loop

    // gather density from all inter communicators
    if (dealii::Utilities::MPI::n_mpi_processes(interpoolcomm) > 1)
      {
        dealii::Utilities::MPI::sum(rhoResponseValuesHamFlattenedHost,
                                    interpoolcomm,
                                    rhoResponseValuesHamFlattenedHost);

        dealii::Utilities::MPI::sum(rhoResponseValuesFermiEnergyFlattenedHost,
                                    interpoolcomm,
                                    rhoResponseValuesFermiEnergyFlattenedHost);

        if (dftParams.spinPolarized == 1)
          {
            dealii::Utilities::MPI::sum(
              rhoResponseValuesSpinPolarizedHamFlattenedHost,
              interpoolcomm,
              rhoResponseValuesSpinPolarizedHamFlattenedHost);

            dealii::Utilities::MPI::sum(
              rhoResponseValuesSpinPolarizedFermiEnergyFlattenedHost,
              interpoolcomm,
              rhoResponseValuesSpinPolarizedFermiEnergyFlattenedHost);
          }
      }

    if (dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm) > 1)
      {
        dealii::Utilities::MPI::sum(rhoResponseValuesHamFlattenedHost,
                                    interBandGroupComm,
                                    rhoResponseValuesHamFlattenedHost);

        dealii::Utilities::MPI::sum(rhoResponseValuesFermiEnergyFlattenedHost,
                                    interBandGroupComm,
                                    rhoResponseValuesFermiEnergyFlattenedHost);

        if (dftParams.spinPolarized == 1)
          {
            dealii::Utilities::MPI::sum(
              rhoResponseValuesSpinPolarizedHamFlattenedHost,
              interBandGroupComm,
              rhoResponseValuesSpinPolarizedHamFlattenedHost);

            dealii::Utilities::MPI::sum(
              rhoResponseValuesSpinPolarizedFermiEnergyFlattenedHost,
              interBandGroupComm,
              rhoResponseValuesSpinPolarizedFermiEnergyFlattenedHost);
          }
      }

    unsigned int                                         iElem = 0;
    typename dealii::DoFHandler<3>::active_cell_iterator cell =
      dofHandler.begin_active();
    typename dealii::DoFHandler<3>::active_cell_iterator endc =
      dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          const dealii::CellId cellid = cell->id();

          std::vector<double> &temp1Quads = (rhoResponseValuesHam)[cellid];
          std::vector<double> &temp2Quads =
            (rhoResponseValuesFermiEnergy)[cellid];
          for (unsigned int q = 0; q < numQuadPoints; ++q)
            {
              temp1Quads[q] =
                rhoResponseValuesHamFlattenedHost[iElem * numQuadPoints + q];
              temp2Quads[q] =
                rhoResponseValuesFermiEnergyFlattenedHost[iElem *
                                                            numQuadPoints +
                                                          q];
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
                    rhoResponseValuesSpinPolarizedHamFlattenedHost
                      [iElem * numQuadPoints * 2 + 2 * q + 0];
                  temp3Quads[2 * q + 1] =
                    rhoResponseValuesSpinPolarizedHamFlattenedHost
                      [iElem * numQuadPoints * 2 + 2 * q + 1];
                  temp4Quads[2 * q + 0] =
                    rhoResponseValuesSpinPolarizedFermiEnergyFlattenedHost
                      [iElem * numQuadPoints * 2 + 2 * q + 0];
                  temp4Quads[2 * q + 1] =
                    rhoResponseValuesSpinPolarizedFermiEnergyFlattenedHost
                      [iElem * numQuadPoints * 2 + 2 * q + 1];
                }
            }

          iElem++;
        }

    dftfe::utils::deviceSynchronize();
    MPI_Barrier(mpiCommParent);
    device_time = MPI_Wtime() - device_time;

    if (this_process == 0 && dftParams.verbosity >= 2)
      std::cout << "Time for compute rhoprime on Device: " << device_time
                << std::endl;
  }

  template void
  computeRhoFirstOrderResponseDevice<dataTypes::number, dataTypes::number>(
    const dataTypes::number *                      X,
    const dataTypes::number *                      XPrime,
    const std::vector<std::vector<double>> &       densityMatDerFermiEnergy,
    const unsigned int                             totalNumWaveFunctions,
    const unsigned int                             numLocalDofs,
    operatorDFTDeviceClass &                       operatorMatrix,
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
    const dftParameters &dftParams,
    const std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      BLASWrapperPtr);

  template void
  computeRhoFirstOrderResponseDevice<dataTypes::number, dataTypes::numberFP32>(
    const dataTypes::number *                      X,
    const dataTypes::number *                      XPrime,
    const std::vector<std::vector<double>> &       densityMatDerFermiEnergy,
    const unsigned int                             totalNumWaveFunctions,
    const unsigned int                             numLocalDofs,
    operatorDFTDeviceClass &                       operatorMatrix,
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
    const dftParameters &dftParams,
    const std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      BLASWrapperPtr);
} // namespace dftfe
