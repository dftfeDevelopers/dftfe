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
#include <densityCalculatorDevice.h>
#include <dftUtils.h>
#include <vectorUtilities.h>
#include <deviceKernelsGeneric.h>
#include <linearAlgebraOperationsDevice.h>
#include <MemoryStorage.h>
#include <DataTypeOverloads.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherConstants.h>
#include <DeviceBlasWrapper.h>

namespace dftfe
{
  namespace Device
  {
    namespace
    {
      __global__ void
      computeRhoGradRhoFromInterpolatedValues(
        const unsigned int numberEntries,
        double *           rhoCellsWfcContributions,
        double *           gradRhoCellsWfcContributionsX,
        double *           gradRhoCellsWfcContributionsY,
        double *           gradRhoCellsWfcContributionsZ,
        const bool         isEvaluateGradRho)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const double psi                = rhoCellsWfcContributions[index];
            rhoCellsWfcContributions[index] = psi * psi;

            if (isEvaluateGradRho)
              {
                const double gradPsiX = gradRhoCellsWfcContributionsX[index];
                gradRhoCellsWfcContributionsX[index] = 2.0 * psi * gradPsiX;

                const double gradPsiY = gradRhoCellsWfcContributionsY[index];
                gradRhoCellsWfcContributionsY[index] = 2.0 * psi * gradPsiY;

                const double gradPsiZ = gradRhoCellsWfcContributionsZ[index];
                gradRhoCellsWfcContributionsZ[index] = 2.0 * psi * gradPsiZ;
              }
          }
      }

      __global__ void
      computeRhoGradRhoFromInterpolatedValues(
        const unsigned int                 numberEntries,
        dftfe::utils::deviceDoubleComplex *rhoCellsWfcContributions,
        dftfe::utils::deviceDoubleComplex *gradRhoCellsWfcContributionsX,
        dftfe::utils::deviceDoubleComplex *gradRhoCellsWfcContributionsY,
        dftfe::utils::deviceDoubleComplex *gradRhoCellsWfcContributionsZ,
        const bool                         isEvaluateGradRho)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const dftfe::utils::deviceDoubleComplex psi =
              rhoCellsWfcContributions[index];
            rhoCellsWfcContributions[index] =
              dftfe::utils::makeComplex(psi.x * psi.x + psi.y * psi.y, 0.0);

            if (isEvaluateGradRho)
              {
                const dftfe::utils::deviceDoubleComplex gradPsiX =
                  gradRhoCellsWfcContributionsX[index];
                gradRhoCellsWfcContributionsX[index] =
                  dftfe::utils::makeComplex(2.0 * (psi.x * gradPsiX.x +
                                                   psi.y * gradPsiX.y),
                                            0.0);

                const dftfe::utils::deviceDoubleComplex gradPsiY =
                  gradRhoCellsWfcContributionsY[index];
                gradRhoCellsWfcContributionsY[index] =
                  dftfe::utils::makeComplex(2.0 * (psi.x * gradPsiY.x +
                                                   psi.y * gradPsiY.y),
                                            0.0);

                const dftfe::utils::deviceDoubleComplex gradPsiZ =
                  gradRhoCellsWfcContributionsZ[index];
                gradRhoCellsWfcContributionsZ[index] =
                  dftfe::utils::makeComplex(2.0 * (psi.x * gradPsiZ.x +
                                                   psi.y * gradPsiZ.y),
                                            0.0);
              }
          }
      }
    } // namespace

    template <typename NumberType>
    void
    computeRhoFromPSI(
      const NumberType *                      X,
      const NumberType *                      XFrac,
      const unsigned int                      totalNumWaveFunctions,
      const unsigned int                      Nfr,
      const unsigned int                      numLocalDofs,
      const std::vector<std::vector<double>> &eigenValues,
      const double                            fermiEnergy,
      const double                            fermiEnergyUp,
      const double                            fermiEnergyDown,
      operatorDFTDeviceClass &                operatorMatrix,
      std::unique_ptr<
        dftfe::basis::FEBasisOperations<NumberType,
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
      const bool                                     use2pPlusOneGLQuad)
    {
      if (use2pPlusOneGLQuad)
        AssertThrow(!isEvaluateGradRho, dftUtils::ExcNotImplementedYet());

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
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm,
        totalNumWaveFunctions,
        bandGroupLowHighPlusOneIndices);

      const unsigned int BVec =
        std::min(dftParams.chebyWfcBlockSize, totalNumWaveFunctions);

      const double spinPolarizedFactor =
        (dftParams.spinPolarized == 1) ? 1.0 : 2.0;

      const NumberType zero                    = 0;
      const NumberType scalarCoeffAlphaRho     = 1.0;
      const NumberType scalarCoeffBetaRho      = 1.0;
      const NumberType scalarCoeffAlphaGradRho = 1.0;
      const NumberType scalarCoeffBetaGradRho  = 1.0;

      const unsigned int cellsBlockSize = 50;
      const unsigned int numCellBlocks =
        totalLocallyOwnedCells / cellsBlockSize;
      const unsigned int remCellBlockSize =
        totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;

      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::DEVICE>
        rhoDevice(totalLocallyOwnedCells * numQuadPoints, zero);
      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::DEVICE>
        rhoWfcContributionsDevice(cellsBlockSize * numQuadPoints * BVec, zero);

      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::DEVICE>
        gradRhoDeviceX(isEvaluateGradRho ?
                         (totalLocallyOwnedCells * numQuadPoints) :
                         1,
                       zero);
      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::DEVICE>
        gradRhoDeviceY(isEvaluateGradRho ?
                         (totalLocallyOwnedCells * numQuadPoints) :
                         1,
                       zero);
      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::DEVICE>
        gradRhoDeviceZ(isEvaluateGradRho ?
                         (totalLocallyOwnedCells * numQuadPoints) :
                         1,
                       zero);
      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::DEVICE>
        gradRhoWfcContributionsDeviceX(
          isEvaluateGradRho ? (cellsBlockSize * numQuadPoints * BVec) : 1,
          zero);
      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::DEVICE>
        gradRhoWfcContributionsDeviceY(
          isEvaluateGradRho ? (cellsBlockSize * numQuadPoints * BVec) : 1,
          zero);
      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::DEVICE>
        gradRhoWfcContributionsDeviceZ(
          isEvaluateGradRho ? (cellsBlockSize * numQuadPoints * BVec) : 1,
          zero);

      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::HOST>
        rhoHost;
      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::HOST>
        gradRhoHostX;
      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::HOST>
        gradRhoHostY;
      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::HOST>
        gradRhoHostZ;

      rhoHost.resize(totalLocallyOwnedCells * numQuadPoints, zero);

      if (isEvaluateGradRho)
        {
          gradRhoHostX.resize(totalLocallyOwnedCells * numQuadPoints, zero);

          gradRhoHostY.resize(totalLocallyOwnedCells * numQuadPoints, zero);
          gradRhoHostZ.resize(totalLocallyOwnedCells * numQuadPoints, zero);
        }


      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::DEVICE>
        shapeFunctionValuesTransposedDevice(numNodesPerElement * numQuadPoints,
                                            zero);

      shapeFunctionValuesTransposedDevice.setValue(zero);


      dftfe::utils::deviceKernelsGeneric::copyValueType1ArrToValueType2Arr(
        numNodesPerElement * numQuadPoints,
        (operatorMatrix.getShapeFunctionValuesTransposed(use2pPlusOneGLQuad))
          .begin(),
        shapeFunctionValuesTransposedDevice.begin());

      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::DEVICE>
        shapeFunctionGradientValuesXTransposedDevice;
      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::DEVICE>
        shapeFunctionGradientValuesYTransposedDevice;
      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::DEVICE>
        shapeFunctionGradientValuesZTransposedDevice;

      if (isEvaluateGradRho)
        {
          shapeFunctionGradientValuesXTransposedDevice.resize(
            cellsBlockSize * numNodesPerElement * numQuadPoints, 0);
          shapeFunctionGradientValuesXTransposedDevice.setValue(0);

          shapeFunctionGradientValuesYTransposedDevice.resize(
            cellsBlockSize * numNodesPerElement * numQuadPoints, 0);
          shapeFunctionGradientValuesYTransposedDevice.setValue(0);

          shapeFunctionGradientValuesZTransposedDevice.resize(
            cellsBlockSize * numNodesPerElement * numQuadPoints, 0);
          shapeFunctionGradientValuesZTransposedDevice.setValue(0);
        }

      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::HOST>
        partialOccupVec(BVec, zero);
      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::DEVICE>
        partialOccupVecDevice(BVec, zero);

      distributedDeviceVec<NumberType> &deviceFlattenedArrayBlock =
        operatorMatrix.getParallelChebyBlockVectorDevice();
      dftfe::utils::MemoryStorage<NumberType, dftfe::utils::MemorySpace::DEVICE>
        &cellWaveFunctionMatrixMV = operatorMatrix.getCellWaveFunctionMatrix();
      NumberType *cellWaveFunctionMatrix = (cellWaveFunctionMatrixMV).begin();

      typename dealii::DoFHandler<3>::active_cell_iterator cell =
        dofHandler.begin_active();
      typename dealii::DoFHandler<3>::active_cell_iterator endc =
        dofHandler.end();

      std::vector<double> rhoValuesFlattened(totalLocallyOwnedCells *
                                               numQuadPoints,
                                             0.0);
      std::vector<double> gradRhoValuesFlattened(totalLocallyOwnedCells *
                                                   numQuadPoints * 3,
                                                 0.0);
      std::vector<double> rhoValuesSpinPolarizedFlattened(
        totalLocallyOwnedCells * numQuadPoints * 2, 0.0);
      std::vector<double> gradRhoValuesSpinPolarizedFlattened(
        totalLocallyOwnedCells * numQuadPoints * 6, 0.0);

      for (unsigned int spinIndex = 0;
           spinIndex < (1 + dftParams.spinPolarized);
           ++spinIndex)
        {
          for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
            {
              rhoDevice.setValue(zero);
              rhoWfcContributionsDevice.setValue(zero);
              gradRhoDeviceX.setValue(zero);
              gradRhoDeviceY.setValue(zero);
              gradRhoDeviceZ.setValue(zero);
              gradRhoWfcContributionsDeviceX.setValue(zero);
              gradRhoWfcContributionsDeviceY.setValue(zero);
              gradRhoWfcContributionsDeviceZ.setValue(zero);

              for (unsigned int jvec = 0; jvec < totalNumWaveFunctions;
                   jvec += BVec)
                {
                  if ((jvec + BVec) <=
                        bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId +
                                                       1] &&
                      (jvec + BVec) >
                        bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                    {
                      if (spectrumSplit)
                        {
                          partialOccupVecDevice.setValue(kPointWeights[kPoint] *
                                                         spinPolarizedFactor);
                        }
                      else
                        {
                          if (dftParams.constraintMagnetization)
                            {
                              const double fermiEnergyConstraintMag =
                                spinIndex == 0 ? fermiEnergyUp :
                                                 fermiEnergyDown;
                              for (unsigned int iEigenVec = 0; iEigenVec < BVec;
                                   ++iEigenVec)
                                {
                                  if (eigenValues[kPoint]
                                                 [totalNumWaveFunctions *
                                                    spinIndex +
                                                  jvec + iEigenVec] >
                                      fermiEnergyConstraintMag)
                                    *(partialOccupVec.begin() + iEigenVec) = 0;
                                  else
                                    *(partialOccupVec.begin() + iEigenVec) =
                                      kPointWeights[kPoint] *
                                      spinPolarizedFactor;
                                }
                            }
                          else
                            {
                              for (unsigned int iEigenVec = 0; iEigenVec < BVec;
                                   ++iEigenVec)
                                {
                                  *(partialOccupVec.begin() + iEigenVec) =
                                    dftUtils::getPartialOccupancy(
                                      eigenValues[kPoint]
                                                 [totalNumWaveFunctions *
                                                    spinIndex +
                                                  jvec + iEigenVec],
                                      fermiEnergy,
                                      C_kb,
                                      dftParams.TVal) *
                                    kPointWeights[kPoint] * spinPolarizedFactor;
                                }
                            }

                          partialOccupVec
                            .template copyTo<dftfe::utils::MemorySpace::DEVICE>(
                              partialOccupVecDevice);
                        }

                      dftfe::utils::deviceKernelsGeneric::
                        stridedCopyToBlockConstantStride(
                          BVec,
                          totalNumWaveFunctions,
                          numLocalDofs,
                          jvec,
                          X + numLocalDofs * totalNumWaveFunctions *
                                ((dftParams.spinPolarized + 1) * kPoint +
                                 spinIndex),
                          deviceFlattenedArrayBlock.begin());

                      const unsigned int d_eigenDofHandlerIndex = 1;
                      const unsigned int d_quadratureIndex =
                        use2pPlusOneGLQuad ? 2 : 0;
                      dftfe::basis::UpdateFlags updateFlags =
                        dftfe::basis::update_values |
                        dftfe::basis::update_gradients;
                      basisOperationsPtrDevice->reinit(BVec,
                                                       0,
                                                       d_quadratureIndex,
                                                       updateFlags);


                      deviceFlattenedArrayBlock.updateGhostValues();

                      (operatorMatrix.getOverloadedConstraintMatrix())
                        ->distribute(deviceFlattenedArrayBlock, BVec);

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

                              dftfe::utils::deviceKernelsGeneric::
                                stridedCopyToBlock(
                                  BVec,
                                  currentCellsBlockSize * numNodesPerElement,
                                  deviceFlattenedArrayBlock.begin(),
                                  cellWaveFunctionMatrix,
                                  (operatorMatrix
                                     .getFlattenedArrayCellLocalProcIndexIdMap())
                                      .begin() +
                                    startingCellId * numNodesPerElement);
                              // basisOperationsPtrDevice
                              //   ->extractToCellNodalDataKernel(
                              //     deviceFlattenedArrayBlock,
                              //     &cellWaveFunctionMatrixMV,
                              //     std::pair<unsigned int, unsigned int>(
                              //       startingCellId,
                              //       startingCellId + currentCellsBlockSize));

                              NumberType scalarCoeffAlpha = 1.0;
                              NumberType scalarCoeffBeta  = 0;
                              int        strideA = BVec * numNodesPerElement;
                              int        strideB = 0;
                              int        strideC = BVec * numQuadPoints;

                              // dftfe::utils::deviceBlasWrapper::
                              //   gemmStridedBatched(
                              //     operatorMatrix.getDeviceBlasHandle(),
                              //     dftfe::utils::DEVICEBLAS_OP_N,
                              //     dftfe::utils::DEVICEBLAS_OP_N,
                              //     BVec,
                              //     numQuadPoints,
                              //     numNodesPerElement,
                              //     &scalarCoeffAlpha,
                              //     cellWaveFunctionMatrixMV.data(),
                              //     BVec,
                              //     strideA,
                              //     shapeFunctionValuesTransposedDevice.begin(),
                              //     numNodesPerElement,
                              //     strideB,
                              //     &scalarCoeffBeta,
                              //     rhoWfcContributionsDevice.begin(),
                              //     BVec,
                              //     strideC,
                              //     currentCellsBlockSize);
                              basisOperationsPtrDevice->interpolateKernel(
                                deviceFlattenedArrayBlock,
                                &rhoWfcContributionsDevice,
                                NULL,
                                std::pair<unsigned int, unsigned int>(
                                  startingCellId,
                                  startingCellId + currentCellsBlockSize));


                              if (isEvaluateGradRho)
                                {
                                  strideB = numNodesPerElement * numQuadPoints;


                                  dftfe::utils::deviceKernelsGeneric::
                                    copyValueType1ArrToValueType2Arr(
                                      currentCellsBlockSize *
                                        numNodesPerElement * numQuadPoints,
                                      (operatorMatrix
                                         .getShapeFunctionGradientValuesXTransposed())
                                          .begin() +
                                        startingCellId * numNodesPerElement *
                                          numQuadPoints,
                                      shapeFunctionGradientValuesXTransposedDevice
                                        .begin());

                                  dftfe::utils::deviceKernelsGeneric::
                                    copyValueType1ArrToValueType2Arr(
                                      currentCellsBlockSize *
                                        numNodesPerElement * numQuadPoints,
                                      (operatorMatrix
                                         .getShapeFunctionGradientValuesYTransposed())
                                          .begin() +
                                        startingCellId * numNodesPerElement *
                                          numQuadPoints,
                                      shapeFunctionGradientValuesYTransposedDevice
                                        .begin());

                                  dftfe::utils::deviceKernelsGeneric::
                                    copyValueType1ArrToValueType2Arr(
                                      currentCellsBlockSize *
                                        numNodesPerElement * numQuadPoints,
                                      (operatorMatrix
                                         .getShapeFunctionGradientValuesZTransposed())
                                          .begin() +
                                        startingCellId * numNodesPerElement *
                                          numQuadPoints,
                                      shapeFunctionGradientValuesZTransposedDevice
                                        .begin());

                                  dftfe::utils::deviceBlasWrapper::
                                    gemmStridedBatched(
                                      operatorMatrix.getDeviceBlasHandle(),
                                      dftfe::utils::DEVICEBLAS_OP_N,
                                      dftfe::utils::DEVICEBLAS_OP_N,
                                      BVec,
                                      numQuadPoints,
                                      numNodesPerElement,
                                      &scalarCoeffAlpha,
                                      cellWaveFunctionMatrix,
                                      BVec,
                                      strideA,
                                      shapeFunctionGradientValuesXTransposedDevice
                                        .begin(),
                                      numNodesPerElement,
                                      strideB,
                                      &scalarCoeffBeta,
                                      gradRhoWfcContributionsDeviceX.begin(),
                                      BVec,
                                      strideC,
                                      currentCellsBlockSize);


                                  dftfe::utils::deviceBlasWrapper::
                                    gemmStridedBatched(
                                      operatorMatrix.getDeviceBlasHandle(),
                                      dftfe::utils::DEVICEBLAS_OP_N,
                                      dftfe::utils::DEVICEBLAS_OP_N,
                                      BVec,
                                      numQuadPoints,
                                      numNodesPerElement,
                                      &scalarCoeffAlpha,
                                      cellWaveFunctionMatrix,
                                      BVec,
                                      strideA,
                                      shapeFunctionGradientValuesYTransposedDevice
                                        .begin(),
                                      numNodesPerElement,
                                      strideB,
                                      &scalarCoeffBeta,
                                      gradRhoWfcContributionsDeviceY.begin(),
                                      BVec,
                                      strideC,
                                      currentCellsBlockSize);

                                  dftfe::utils::deviceBlasWrapper::
                                    gemmStridedBatched(
                                      operatorMatrix.getDeviceBlasHandle(),
                                      dftfe::utils::DEVICEBLAS_OP_N,
                                      dftfe::utils::DEVICEBLAS_OP_N,
                                      BVec,
                                      numQuadPoints,
                                      numNodesPerElement,
                                      &scalarCoeffAlpha,
                                      cellWaveFunctionMatrix,
                                      BVec,
                                      strideA,
                                      shapeFunctionGradientValuesZTransposedDevice
                                        .begin(),
                                      numNodesPerElement,
                                      strideB,
                                      &scalarCoeffBeta,
                                      gradRhoWfcContributionsDeviceZ.begin(),
                                      BVec,
                                      strideC,
                                      currentCellsBlockSize);
                                }


#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
                              computeRhoGradRhoFromInterpolatedValues<<<
                                (BVec + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                  dftfe::utils::DEVICE_BLOCK_SIZE *
                                  numQuadPoints * currentCellsBlockSize,
                                dftfe::utils::DEVICE_BLOCK_SIZE>>>(
                                currentCellsBlockSize * numQuadPoints * BVec,
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  rhoWfcContributionsDevice.begin()),
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  gradRhoWfcContributionsDeviceX.begin()),
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  gradRhoWfcContributionsDeviceY.begin()),
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  gradRhoWfcContributionsDeviceZ.begin()),
                                isEvaluateGradRho);
#elif DFTFE_WITH_DEVICE_LANG_HIP
                              hipLaunchKernelGGL(
                                computeRhoGradRhoFromInterpolatedValues,
                                (BVec + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                  dftfe::utils::DEVICE_BLOCK_SIZE *
                                  numQuadPoints * currentCellsBlockSize,
                                dftfe::utils::DEVICE_BLOCK_SIZE,
                                0,
                                0,
                                currentCellsBlockSize * numQuadPoints * BVec,
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  rhoWfcContributionsDevice.begin()),
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  gradRhoWfcContributionsDeviceX.begin()),
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  gradRhoWfcContributionsDeviceY.begin()),
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  gradRhoWfcContributionsDeviceZ.begin()),
                                isEvaluateGradRho);
#endif

                              dftfe::utils::deviceBlasWrapper::gemm(
                                operatorMatrix.getDeviceBlasHandle(),
                                dftfe::utils::DEVICEBLAS_OP_N,
                                dftfe::utils::DEVICEBLAS_OP_N,
                                1,
                                currentCellsBlockSize * numQuadPoints,
                                BVec,
                                &scalarCoeffAlphaRho,
                                partialOccupVecDevice.begin(),
                                1,
                                rhoWfcContributionsDevice.begin(),
                                BVec,
                                &scalarCoeffBetaRho,
                                rhoDevice.begin() +
                                  startingCellId * numQuadPoints,
                                1);


                              if (isEvaluateGradRho)
                                {
                                  dftfe::utils::deviceBlasWrapper::gemm(
                                    operatorMatrix.getDeviceBlasHandle(),
                                    dftfe::utils::DEVICEBLAS_OP_N,
                                    dftfe::utils::DEVICEBLAS_OP_N,
                                    1,
                                    currentCellsBlockSize * numQuadPoints,
                                    BVec,
                                    &scalarCoeffAlphaGradRho,
                                    partialOccupVecDevice.begin(),
                                    1,
                                    gradRhoWfcContributionsDeviceX.begin(),
                                    BVec,
                                    &scalarCoeffBetaGradRho,
                                    gradRhoDeviceX.begin() +
                                      startingCellId * numQuadPoints,
                                    1);


                                  dftfe::utils::deviceBlasWrapper::gemm(
                                    operatorMatrix.getDeviceBlasHandle(),
                                    dftfe::utils::DEVICEBLAS_OP_N,
                                    dftfe::utils::DEVICEBLAS_OP_N,
                                    1,
                                    currentCellsBlockSize * numQuadPoints,
                                    BVec,
                                    &scalarCoeffAlphaGradRho,
                                    partialOccupVecDevice.begin(),
                                    1,
                                    gradRhoWfcContributionsDeviceY.begin(),
                                    BVec,
                                    &scalarCoeffBetaGradRho,
                                    gradRhoDeviceY.begin() +
                                      startingCellId * numQuadPoints,
                                    1);

                                  dftfe::utils::deviceBlasWrapper::gemm(
                                    operatorMatrix.getDeviceBlasHandle(),
                                    dftfe::utils::DEVICEBLAS_OP_N,
                                    dftfe::utils::DEVICEBLAS_OP_N,
                                    1,
                                    currentCellsBlockSize * numQuadPoints,
                                    BVec,
                                    &scalarCoeffAlphaGradRho,
                                    partialOccupVecDevice.begin(),
                                    1,
                                    gradRhoWfcContributionsDeviceZ.begin(),
                                    BVec,
                                    &scalarCoeffBetaGradRho,
                                    gradRhoDeviceZ.begin() +
                                      startingCellId * numQuadPoints,
                                    1);
                                }
                            } // non-trivial cell block check
                        }     // cells block loop
                    }         // band parallelizatoin check
                }             // wave function block loop

              if (spectrumSplit)
                for (unsigned int jvec = 0; jvec < Nfr; jvec += BVec)
                  if ((jvec + totalNumWaveFunctions - Nfr + BVec) <=
                        bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId +
                                                       1] &&
                      (jvec + totalNumWaveFunctions - Nfr + BVec) >
                        bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                    {
                      if (dftParams.constraintMagnetization)
                        {
                          const double fermiEnergyConstraintMag =
                            spinIndex == 0 ? fermiEnergyUp : fermiEnergyDown;
                          for (unsigned int iEigenVec = 0; iEigenVec < BVec;
                               ++iEigenVec)
                            {
                              if (eigenValues[kPoint]
                                             [totalNumWaveFunctions *
                                                spinIndex +
                                              (totalNumWaveFunctions - Nfr) +
                                              jvec + iEigenVec] >
                                  fermiEnergyConstraintMag)
                                *(partialOccupVec.begin() + iEigenVec) =
                                  -kPointWeights[kPoint] * spinPolarizedFactor;
                              else
                                *(partialOccupVec.begin() + iEigenVec) = 0;
                            }
                        }
                      else
                        {
                          for (unsigned int iEigenVec = 0; iEigenVec < BVec;
                               ++iEigenVec)
                            {
                              *(partialOccupVec.begin() + iEigenVec) =
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

                      partialOccupVec
                        .template copyTo<dftfe::utils::MemorySpace::DEVICE>(
                          partialOccupVecDevice);


                      dftfe::utils::deviceKernelsGeneric::
                        stridedCopyToBlockConstantStride(
                          BVec,
                          Nfr,
                          numLocalDofs,
                          jvec,
                          XFrac + numLocalDofs * Nfr *
                                    ((dftParams.spinPolarized + 1) * kPoint +
                                     spinIndex),
                          deviceFlattenedArrayBlock.begin());

                      deviceFlattenedArrayBlock.updateGhostValues();

                      (operatorMatrix.getOverloadedConstraintMatrix())
                        ->distribute(deviceFlattenedArrayBlock, BVec);

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

                              dftfe::utils::deviceKernelsGeneric::
                                stridedCopyToBlock(
                                  BVec,
                                  currentCellsBlockSize * numNodesPerElement,
                                  deviceFlattenedArrayBlock.begin(),
                                  cellWaveFunctionMatrix,
                                  (operatorMatrix
                                     .getFlattenedArrayCellLocalProcIndexIdMap())
                                      .begin() +
                                    startingCellId * numNodesPerElement);

                              NumberType scalarCoeffAlpha = 1.0;
                              NumberType scalarCoeffBeta  = 0;
                              int        strideA = BVec * numNodesPerElement;
                              int        strideB = 0;
                              int        strideC = BVec * numQuadPoints;


                              dftfe::utils::deviceBlasWrapper::
                                gemmStridedBatched(
                                  operatorMatrix.getDeviceBlasHandle(),
                                  dftfe::utils::DEVICEBLAS_OP_N,
                                  dftfe::utils::DEVICEBLAS_OP_N,
                                  BVec,
                                  numQuadPoints,
                                  numNodesPerElement,
                                  &scalarCoeffAlpha,
                                  cellWaveFunctionMatrix,
                                  BVec,
                                  strideA,
                                  shapeFunctionValuesTransposedDevice.begin(),
                                  numNodesPerElement,
                                  strideB,
                                  &scalarCoeffBeta,
                                  rhoWfcContributionsDevice.begin(),
                                  BVec,
                                  strideC,
                                  currentCellsBlockSize);



                              if (isEvaluateGradRho)
                                {
                                  strideB = numNodesPerElement * numQuadPoints;

                                  dftfe::utils::deviceKernelsGeneric::
                                    copyValueType1ArrToValueType2Arr(
                                      currentCellsBlockSize *
                                        numNodesPerElement * numQuadPoints,
                                      (operatorMatrix
                                         .getShapeFunctionGradientValuesXTransposed())
                                          .begin() +
                                        startingCellId * numNodesPerElement *
                                          numQuadPoints,
                                      shapeFunctionGradientValuesXTransposedDevice
                                        .begin());

                                  dftfe::utils::deviceKernelsGeneric::
                                    copyValueType1ArrToValueType2Arr(
                                      currentCellsBlockSize *
                                        numNodesPerElement * numQuadPoints,
                                      (operatorMatrix
                                         .getShapeFunctionGradientValuesYTransposed())
                                          .begin() +
                                        startingCellId * numNodesPerElement *
                                          numQuadPoints,
                                      shapeFunctionGradientValuesYTransposedDevice
                                        .begin());

                                  dftfe::utils::deviceKernelsGeneric::
                                    copyValueType1ArrToValueType2Arr(
                                      currentCellsBlockSize *
                                        numNodesPerElement * numQuadPoints,
                                      (operatorMatrix
                                         .getShapeFunctionGradientValuesZTransposed())
                                          .begin() +
                                        startingCellId * numNodesPerElement *
                                          numQuadPoints,
                                      shapeFunctionGradientValuesZTransposedDevice
                                        .begin());

                                  dftfe::utils::deviceBlasWrapper::
                                    gemmStridedBatched(
                                      operatorMatrix.getDeviceBlasHandle(),
                                      dftfe::utils::DEVICEBLAS_OP_N,
                                      dftfe::utils::DEVICEBLAS_OP_N,
                                      BVec,
                                      numQuadPoints,
                                      numNodesPerElement,
                                      &scalarCoeffAlpha,
                                      cellWaveFunctionMatrix,
                                      BVec,
                                      strideA,
                                      shapeFunctionGradientValuesXTransposedDevice
                                        .begin(),
                                      numNodesPerElement,
                                      strideB,
                                      &scalarCoeffBeta,
                                      gradRhoWfcContributionsDeviceX.begin(),
                                      BVec,
                                      strideC,
                                      currentCellsBlockSize);


                                  dftfe::utils::deviceBlasWrapper::
                                    gemmStridedBatched(
                                      operatorMatrix.getDeviceBlasHandle(),
                                      dftfe::utils::DEVICEBLAS_OP_N,
                                      dftfe::utils::DEVICEBLAS_OP_N,
                                      BVec,
                                      numQuadPoints,
                                      numNodesPerElement,
                                      &scalarCoeffAlpha,
                                      cellWaveFunctionMatrix,
                                      BVec,
                                      strideA,
                                      shapeFunctionGradientValuesYTransposedDevice
                                        .begin(),
                                      numNodesPerElement,
                                      strideB,
                                      &scalarCoeffBeta,
                                      gradRhoWfcContributionsDeviceY.begin(),
                                      BVec,
                                      strideC,
                                      currentCellsBlockSize);

                                  dftfe::utils::deviceBlasWrapper::
                                    gemmStridedBatched(
                                      operatorMatrix.getDeviceBlasHandle(),
                                      dftfe::utils::DEVICEBLAS_OP_N,
                                      dftfe::utils::DEVICEBLAS_OP_N,
                                      BVec,
                                      numQuadPoints,
                                      numNodesPerElement,
                                      &scalarCoeffAlpha,
                                      cellWaveFunctionMatrix,
                                      BVec,
                                      strideA,
                                      shapeFunctionGradientValuesZTransposedDevice
                                        .begin(),
                                      numNodesPerElement,
                                      strideB,
                                      &scalarCoeffBeta,
                                      gradRhoWfcContributionsDeviceZ.begin(),
                                      BVec,
                                      strideC,
                                      currentCellsBlockSize);
                                }


#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
                              computeRhoGradRhoFromInterpolatedValues<<<
                                (BVec + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                  dftfe::utils::DEVICE_BLOCK_SIZE *
                                  numQuadPoints * currentCellsBlockSize,
                                dftfe::utils::DEVICE_BLOCK_SIZE>>>(
                                currentCellsBlockSize * numQuadPoints * BVec,
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  rhoWfcContributionsDevice.begin()),
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  gradRhoWfcContributionsDeviceX.begin()),
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  gradRhoWfcContributionsDeviceY.begin()),
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  gradRhoWfcContributionsDeviceZ.begin()),
                                isEvaluateGradRho);
#elif DFTFE_WITH_DEVICE_LANG_HIP
                              hipLaunchKernelGGL(
                                computeRhoGradRhoFromInterpolatedValues,
                                (BVec + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                  dftfe::utils::DEVICE_BLOCK_SIZE *
                                  numQuadPoints * currentCellsBlockSize,
                                dftfe::utils::DEVICE_BLOCK_SIZE,
                                0,
                                0,
                                currentCellsBlockSize * numQuadPoints * BVec,
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  rhoWfcContributionsDevice.begin()),
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  gradRhoWfcContributionsDeviceX.begin()),
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  gradRhoWfcContributionsDeviceY.begin()),
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  gradRhoWfcContributionsDeviceZ.begin()),
                                isEvaluateGradRho);
#endif

                              dftfe::utils::deviceBlasWrapper::gemm(
                                operatorMatrix.getDeviceBlasHandle(),
                                dftfe::utils::DEVICEBLAS_OP_N,
                                dftfe::utils::DEVICEBLAS_OP_N,
                                1,
                                currentCellsBlockSize * numQuadPoints,
                                BVec,
                                &scalarCoeffAlphaRho,
                                partialOccupVecDevice.begin(),
                                1,
                                rhoWfcContributionsDevice.begin(),
                                BVec,
                                &scalarCoeffBetaRho,
                                rhoDevice.begin() +
                                  startingCellId * numQuadPoints,
                                1);


                              if (isEvaluateGradRho)
                                {
                                  dftfe::utils::deviceBlasWrapper::gemm(
                                    operatorMatrix.getDeviceBlasHandle(),
                                    dftfe::utils::DEVICEBLAS_OP_N,
                                    dftfe::utils::DEVICEBLAS_OP_N,
                                    1,
                                    currentCellsBlockSize * numQuadPoints,
                                    BVec,
                                    &scalarCoeffAlphaGradRho,
                                    partialOccupVecDevice.begin(),
                                    1,
                                    gradRhoWfcContributionsDeviceX.begin(),
                                    BVec,
                                    &scalarCoeffBetaGradRho,
                                    gradRhoDeviceX.begin() +
                                      startingCellId * numQuadPoints,
                                    1);


                                  dftfe::utils::deviceBlasWrapper::gemm(
                                    operatorMatrix.getDeviceBlasHandle(),
                                    dftfe::utils::DEVICEBLAS_OP_N,
                                    dftfe::utils::DEVICEBLAS_OP_N,
                                    1,
                                    currentCellsBlockSize * numQuadPoints,
                                    BVec,
                                    &scalarCoeffAlphaGradRho,
                                    partialOccupVecDevice.begin(),
                                    1,
                                    gradRhoWfcContributionsDeviceY.begin(),
                                    BVec,
                                    &scalarCoeffBetaGradRho,
                                    gradRhoDeviceY.begin() +
                                      startingCellId * numQuadPoints,
                                    1);

                                  dftfe::utils::deviceBlasWrapper::gemm(
                                    operatorMatrix.getDeviceBlasHandle(),
                                    dftfe::utils::DEVICEBLAS_OP_N,
                                    dftfe::utils::DEVICEBLAS_OP_N,
                                    1,
                                    currentCellsBlockSize * numQuadPoints,
                                    BVec,
                                    &scalarCoeffAlphaGradRho,
                                    partialOccupVecDevice.begin(),
                                    1,
                                    gradRhoWfcContributionsDeviceZ.begin(),
                                    BVec,
                                    &scalarCoeffBetaGradRho,
                                    gradRhoDeviceZ.begin() +
                                      startingCellId * numQuadPoints,
                                    1);
                                }
                            } // non-tivial cells block
                        }     // cells block loop
                    }         // spectrum split block


              // do memcopy to host
              rhoDevice.template copyTo<dftfe::utils::MemorySpace::HOST>(
                rhoHost.begin(), totalLocallyOwnedCells * numQuadPoints, 0, 0);

              if (isEvaluateGradRho)
                {
                  gradRhoDeviceX
                    .template copyTo<dftfe::utils::MemorySpace::HOST>(
                      gradRhoHostX.begin(),
                      totalLocallyOwnedCells * numQuadPoints,
                      0,
                      0);

                  gradRhoDeviceY
                    .template copyTo<dftfe::utils::MemorySpace::HOST>(
                      gradRhoHostY.begin(),
                      totalLocallyOwnedCells * numQuadPoints,
                      0,
                      0);

                  gradRhoDeviceZ
                    .template copyTo<dftfe::utils::MemorySpace::HOST>(
                      gradRhoHostZ.begin(),
                      totalLocallyOwnedCells * numQuadPoints,
                      0,
                      0);
                }

              for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
                for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
                  {
                    rhoValuesFlattened[icell * numQuadPoints + iquad] +=
                      dftfe::utils::realPart(
                        *(rhoHost.begin() + icell * numQuadPoints + iquad));
                  }

              if (isEvaluateGradRho)
                for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
                  for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
                    {
                      gradRhoValuesFlattened[icell * numQuadPoints * 3 +
                                             3 * iquad + 0] +=
                        dftfe::utils::realPart(*(gradRhoHostX.begin() +
                                                 icell * numQuadPoints +
                                                 iquad));
                      gradRhoValuesFlattened[icell * numQuadPoints * 3 +
                                             3 * iquad + 1] +=
                        dftfe::utils::realPart(*(gradRhoHostY.begin() +
                                                 icell * numQuadPoints +
                                                 iquad));
                      gradRhoValuesFlattened[icell * numQuadPoints * 3 +
                                             3 * iquad + 2] +=
                        dftfe::utils::realPart(*(gradRhoHostZ.begin() +
                                                 icell * numQuadPoints +
                                                 iquad));
                    }
              if (dftParams.spinPolarized == 1)
                {
                  for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
                    for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
                      {
                        rhoValuesSpinPolarizedFlattened
                          [icell * numQuadPoints * 2 + iquad * 2 + spinIndex] +=
                          dftfe::utils::realPart(
                            *(rhoHost.begin() + icell * numQuadPoints + iquad));
                      }

                  if (isEvaluateGradRho)
                    for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
                      for (unsigned int iquad = 0; iquad < numQuadPoints;
                           ++iquad)
                        {
                          gradRhoValuesSpinPolarizedFlattened
                            [icell * numQuadPoints * 6 + iquad * 6 +
                             spinIndex * 3] +=
                            dftfe::utils::realPart(*(gradRhoHostX.begin() +
                                                     icell * numQuadPoints +
                                                     iquad));
                          gradRhoValuesSpinPolarizedFlattened
                            [icell * numQuadPoints * 6 + iquad * 6 +
                             spinIndex * 3 + 1] +=
                            dftfe::utils::realPart(*(gradRhoHostY.begin() +
                                                     icell * numQuadPoints +
                                                     iquad));
                          gradRhoValuesSpinPolarizedFlattened
                            [icell * numQuadPoints * 6 + iquad * 6 +
                             spinIndex * 3 + 2] +=
                            dftfe::utils::realPart(*(gradRhoHostZ.begin() +
                                                     icell * numQuadPoints +
                                                     iquad));
                        }
                }
            } // kpoint loop
        }     // spin index


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
                dealii::Utilities::MPI::sum(
                  gradRhoValuesSpinPolarizedFlattened,
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
                dealii::Utilities::MPI::sum(
                  gradRhoValuesSpinPolarizedFlattened,
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
              (dftParams.spinPolarized == 1) ?
                (*rhoValuesSpinPolarized)[cellid] :
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
                      rhoValuesSpinPolarizedFlattened[iElem * numQuadPoints *
                                                        2 +
                                                      q * 2 + 0];

                    tempRhoQuadsSP[2 * q + 1] =
                      rhoValuesSpinPolarizedFlattened[iElem * numQuadPoints *
                                                        2 +
                                                      q * 2 + 1];
                  }

                if (isEvaluateGradRho)
                  for (unsigned int q = 0; q < numQuadPoints; ++q)
                    {
                      tempGradRhoQuadsSP[6 * q + 0] =
                        gradRhoValuesSpinPolarizedFlattened
                          [iElem * numQuadPoints * 6 + 6 * q];
                      tempGradRhoQuadsSP[6 * q + 1] =
                        gradRhoValuesSpinPolarizedFlattened
                          [iElem * numQuadPoints * 6 + 6 * q + 1];
                      tempGradRhoQuadsSP[6 * q + 2] =
                        gradRhoValuesSpinPolarizedFlattened
                          [iElem * numQuadPoints * 6 + 6 * q + 2];
                      tempGradRhoQuadsSP[6 * q + 3] =
                        gradRhoValuesSpinPolarizedFlattened
                          [iElem * numQuadPoints * 6 + 6 * q + 3];
                      tempGradRhoQuadsSP[6 * q + 4] =
                        gradRhoValuesSpinPolarizedFlattened
                          [iElem * numQuadPoints * 6 + 6 * q + 4];
                      tempGradRhoQuadsSP[6 * q + 5] =
                        gradRhoValuesSpinPolarizedFlattened
                          [iElem * numQuadPoints * 6 + 6 * q + 5];
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
                    gradRhoValuesFlattened[iElem * numQuadPoints * 3 + q * 3 +
                                           1];
                  tempGradRhoQuads[3 * q + 2] =
                    gradRhoValuesFlattened[iElem * numQuadPoints * 3 + q * 3 +
                                           2];
                }
            iElem++;
          }

      dftfe::utils::deviceSynchronize();
      MPI_Barrier(mpiCommParent);
      device_time = MPI_Wtime() - device_time;

      if (this_process == 0 && dftParams.verbosity >= 2)
        std::cout << "Time for compute rho on Device: " << device_time
                  << std::endl;
    }

    template void
    computeRhoFromPSI(
      const dataTypes::number *               X,
      const dataTypes::number *               XFrac,
      const unsigned int                      totalNumWaveFunctions,
      const unsigned int                      Nfr,
      const unsigned int                      numLocalDofs,
      const std::vector<std::vector<double>> &eigenValues,
      const double                            fermiEnergy,
      const double                            fermiEnergyUp,
      const double                            fermiEnergyDown,
      operatorDFTDeviceClass &                operatorMatrix,
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
  } // namespace Device
} // namespace dftfe
