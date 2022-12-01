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
#include <cuComplex.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include "deviceHelpers.h"
#include "linearAlgebraOperationsDevice.h"


namespace dftfe
{
  namespace Device
  {
    namespace
    {
      template <typename NumberType>
      __global__ void
      stridedCopyToBlockKernel(const unsigned int BVec,
                               const NumberType * xVec,
                               const unsigned int M,
                               const unsigned int N,
                               NumberType *       yVec,
                               const unsigned int startingXVecId)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries = M * BVec;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            unsigned int blockIndex      = index / BVec;
            unsigned int intraBlockIndex = index - blockIndex * BVec;
            yVec[index] =
              xVec[blockIndex * N + startingXVecId + intraBlockIndex];
          }
      }

      template <typename NumberType>
      __global__ void
      copyGlobalToCellDeviceKernel(const unsigned int contiguousBlockSize,
                                   const unsigned int numContiguousBlocks,
                                   const NumberType * copyFromVec,
                                   NumberType *       copyToVec,
                                   const dealii::types::global_dof_index
                                     *copyFromVecStartingContiguousBlockIds)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            unsigned int blockIndex = index / contiguousBlockSize;
            unsigned int intraBlockIndex =
              index - blockIndex * contiguousBlockSize;
            copyToVec[index] =
              copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex];
          }
      }


      __global__ void
      copyDeviceKernel(const unsigned int size,
                       const double *     copyFromVec,
                       double *           copyToVec)
      {
        for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
             index < size;
             index += blockDim.x * gridDim.x)
          copyToVec[index] = copyFromVec[index];
      }

      __global__ void
      copyDeviceKernel(const unsigned int size,
                       const double *     copyFromVec,
                       cuDoubleComplex *  copyToVec)
      {
        for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
             index < size;
             index += blockDim.x * gridDim.x)
          {
            copyToVec[index] = make_cuDoubleComplex(copyFromVec[index], 0.0);
          }
      }

      void
      copyDoubleToNumber(const double *     copyFromVec,
                         const unsigned int size,
                         double *           copyToVec)
      {
        copyDeviceKernel<<<(size + 255) / 256, 256>>>(size,
                                                      copyFromVec,
                                                      copyToVec);
      }

      void
      copyDoubleToNumber(const double *     copyFromVec,
                         const unsigned int size,
                         cuDoubleComplex *  copyToVec)
      {
        copyDeviceKernel<<<(size + 255) / 256, 256>>>(size,
                                                      copyFromVec,
                                                      copyToVec);
      }

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
        const unsigned int numberEntries,
        cuDoubleComplex *  rhoCellsWfcContributions,
        cuDoubleComplex *  gradRhoCellsWfcContributionsX,
        cuDoubleComplex *  gradRhoCellsWfcContributionsY,
        cuDoubleComplex *  gradRhoCellsWfcContributionsZ,
        const bool         isEvaluateGradRho)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const cuDoubleComplex psi = rhoCellsWfcContributions[index];
            rhoCellsWfcContributions[index] =
              make_cuDoubleComplex(psi.x * psi.x + psi.y * psi.y, 0.0);

            if (isEvaluateGradRho)
              {
                const cuDoubleComplex gradPsiX =
                  gradRhoCellsWfcContributionsX[index];
                gradRhoCellsWfcContributionsX[index] = make_cuDoubleComplex(
                  2.0 * (psi.x * gradPsiX.x + psi.y * gradPsiX.y), 0.0);

                const cuDoubleComplex gradPsiY =
                  gradRhoCellsWfcContributionsY[index];
                gradRhoCellsWfcContributionsY[index] = make_cuDoubleComplex(
                  2.0 * (psi.x * gradPsiY.x + psi.y * gradPsiY.y), 0.0);

                const cuDoubleComplex gradPsiZ =
                  gradRhoCellsWfcContributionsZ[index];
                gradRhoCellsWfcContributionsZ[index] = make_cuDoubleComplex(
                  2.0 * (psi.x * gradPsiZ.x + psi.y * gradPsiZ.y), 0.0);
              }
          }
      }
    } // namespace

    template <typename NumberType>
    void
    computeRhoFromPSI(
      const NumberType *                             X,
      const NumberType *                             XFrac,
      const unsigned int                             totalNumWaveFunctions,
      const unsigned int                             Nfr,
      const unsigned int                             numLocalDofs,
      const std::vector<std::vector<double>> &       eigenValues,
      const double                                   fermiEnergy,
      const double                                   fermiEnergyUp,
      const double                                   fermiEnergyDown,
      operatorDFTDeviceClass &                       operatorMatrix,
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
      cudaDeviceSynchronize();
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

      const NumberType zero = deviceUtils::makeNumberFromReal<NumberType>(0.0);
      const NumberType scalarCoeffAlphaRho =
        deviceUtils::makeNumberFromReal<NumberType>(1.0);
      const NumberType scalarCoeffBetaRho =
        deviceUtils::makeNumberFromReal<NumberType>(1.0);
      const NumberType scalarCoeffAlphaGradRho =
        deviceUtils::makeNumberFromReal<NumberType>(1.0);
      const NumberType scalarCoeffBetaGradRho =
        deviceUtils::makeNumberFromReal<NumberType>(1.0);

      const unsigned int cellsBlockSize = 50;
      const unsigned int numCellBlocks =
        totalLocallyOwnedCells / cellsBlockSize;
      const unsigned int remCellBlockSize =
        totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;

      deviceUtils::Vector<NumberType, dftfe::MemorySpace::Device> rhoDevice(
        totalLocallyOwnedCells * numQuadPoints, zero);
      deviceUtils::Vector<NumberType, dftfe::MemorySpace::Device>
        rhoWfcContributionsDevice(cellsBlockSize * numQuadPoints * BVec, zero);

      deviceUtils::Vector<NumberType, dftfe::MemorySpace::Device>
        gradRhoDeviceX(isEvaluateGradRho ?
                         (totalLocallyOwnedCells * numQuadPoints) :
                         1,
                       zero);
      deviceUtils::Vector<NumberType, dftfe::MemorySpace::Device>
        gradRhoDeviceY(isEvaluateGradRho ?
                         (totalLocallyOwnedCells * numQuadPoints) :
                         1,
                       zero);
      deviceUtils::Vector<NumberType, dftfe::MemorySpace::Device>
        gradRhoDeviceZ(isEvaluateGradRho ?
                         (totalLocallyOwnedCells * numQuadPoints) :
                         1,
                       zero);
      deviceUtils::Vector<NumberType, dftfe::MemorySpace::Device>
        gradRhoWfcContributionsDeviceX(
          isEvaluateGradRho ? (cellsBlockSize * numQuadPoints * BVec) : 1,
          zero);
      deviceUtils::Vector<NumberType, dftfe::MemorySpace::Device>
        gradRhoWfcContributionsDeviceY(
          isEvaluateGradRho ? (cellsBlockSize * numQuadPoints * BVec) : 1,
          zero);
      deviceUtils::Vector<NumberType, dftfe::MemorySpace::Device>
        gradRhoWfcContributionsDeviceZ(
          isEvaluateGradRho ? (cellsBlockSize * numQuadPoints * BVec) : 1,
          zero);

      deviceUtils::Vector<NumberType, dftfe::MemorySpace::Host> rhoHost;
      deviceUtils::Vector<NumberType, dftfe::MemorySpace::Host> gradRhoHostX;
      deviceUtils::Vector<NumberType, dftfe::MemorySpace::Host> gradRhoHostY;
      deviceUtils::Vector<NumberType, dftfe::MemorySpace::Host> gradRhoHostZ;

      rhoHost.resize(totalLocallyOwnedCells * numQuadPoints, zero);

      if (isEvaluateGradRho)
        {
          gradRhoHostX.resize(totalLocallyOwnedCells * numQuadPoints, zero);

          gradRhoHostY.resize(totalLocallyOwnedCells * numQuadPoints, zero);
          gradRhoHostZ.resize(totalLocallyOwnedCells * numQuadPoints, zero);
        }


      NumberType *shapeFunctionValuesTransposedDevice;

      DeviceCHECK(
        cudaMalloc((void **)&shapeFunctionValuesTransposedDevice,
                   numNodesPerElement * numQuadPoints * sizeof(NumberType)));
      DeviceCHECK(
        cudaMemset(shapeFunctionValuesTransposedDevice,
                   0,
                   numNodesPerElement * numQuadPoints * sizeof(NumberType)));

      copyDoubleToNumber(thrust::raw_pointer_cast(
                           &(operatorMatrix.getShapeFunctionValuesTransposed(
                             use2pPlusOneGLQuad)[0])),
                         numNodesPerElement * numQuadPoints,
                         shapeFunctionValuesTransposedDevice);


      NumberType *shapeFunctionGradientValuesXTransposedDevice;
      NumberType *shapeFunctionGradientValuesYTransposedDevice;
      NumberType *shapeFunctionGradientValuesZTransposedDevice;

      if (isEvaluateGradRho)
        {
          DeviceCHECK(
            cudaMalloc((void **)&shapeFunctionGradientValuesXTransposedDevice,
                       cellsBlockSize * numNodesPerElement * numQuadPoints *
                         sizeof(NumberType)));
          DeviceCHECK(cudaMemset(shapeFunctionGradientValuesXTransposedDevice,
                                 0,
                                 cellsBlockSize * numNodesPerElement *
                                   numQuadPoints * sizeof(NumberType)));

          DeviceCHECK(
            cudaMalloc((void **)&shapeFunctionGradientValuesYTransposedDevice,
                       cellsBlockSize * numNodesPerElement * numQuadPoints *
                         sizeof(NumberType)));
          DeviceCHECK(cudaMemset(shapeFunctionGradientValuesYTransposedDevice,
                                 0,
                                 cellsBlockSize * numNodesPerElement *
                                   numQuadPoints * sizeof(NumberType)));

          DeviceCHECK(
            cudaMalloc((void **)&shapeFunctionGradientValuesZTransposedDevice,
                       cellsBlockSize * numNodesPerElement * numQuadPoints *
                         sizeof(NumberType)));
          DeviceCHECK(cudaMemset(shapeFunctionGradientValuesZTransposedDevice,
                                 0,
                                 cellsBlockSize * numNodesPerElement *
                                   numQuadPoints * sizeof(NumberType)));
        }

      deviceUtils::Vector<NumberType, dftfe::MemorySpace::Host> partialOccupVec(
        BVec, zero);
      deviceUtils::Vector<NumberType, dftfe::MemorySpace::Device>
        partialOccupVecDevice(BVec, zero);

      distributedDeviceVec<NumberType> &deviceFlattenedArrayBlock =
        operatorMatrix.getParallelChebyBlockVectorDevice();

      const unsigned int numGhosts =
        deviceFlattenedArrayBlock.ghostFlattenedSize();

      NumberType *cellWaveFunctionMatrix =
        reinterpret_cast<NumberType *>(thrust::raw_pointer_cast(
          &operatorMatrix.getCellWaveFunctionMatrix()[0]));

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
              rhoDevice.set(zero);
              rhoWfcContributionsDevice.set(zero);
              gradRhoDeviceX.set(zero);
              gradRhoDeviceY.set(zero);
              gradRhoDeviceZ.set(zero);
              gradRhoWfcContributionsDeviceX.set(zero);
              gradRhoWfcContributionsDeviceY.set(zero);
              gradRhoWfcContributionsDeviceZ.set(zero);

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
                          partialOccupVecDevice.set(
                            deviceUtils::makeNumberFromReal<NumberType>(
                              kPointWeights[kPoint] * spinPolarizedFactor));
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
                                    *(partialOccupVec.begin() + iEigenVec) =
                                      deviceUtils::makeNumberFromReal<
                                        NumberType>(0.0);
                                  else
                                    *(partialOccupVec.begin() + iEigenVec) =
                                      deviceUtils::makeNumberFromReal<
                                        NumberType>(kPointWeights[kPoint] *
                                                    spinPolarizedFactor);
                                }
                            }
                          else
                            {
                              for (unsigned int iEigenVec = 0; iEigenVec < BVec;
                                   ++iEigenVec)
                                {
                                  *(partialOccupVec.begin() + iEigenVec) =
                                    deviceUtils::makeNumberFromReal<NumberType>(
                                      dftUtils::getPartialOccupancy(
                                        eigenValues[kPoint]
                                                   [totalNumWaveFunctions *
                                                      spinIndex +
                                                    jvec + iEigenVec],
                                        fermiEnergy,
                                        C_kb,
                                        dftParams.TVal) *
                                      kPointWeights[kPoint] *
                                      spinPolarizedFactor);
                                }
                            }

                          // partialOccupVecDevice = partialOccupVec;

                          deviceUtils::copyHostVecToDeviceVec(
                            partialOccupVec.begin(),
                            partialOccupVecDevice.begin(),
                            partialOccupVecDevice.size());
                        }

                      stridedCopyToBlockKernel<<<
                        (BVec + 255) / 256 * numLocalDofs,
                        256>>>(BVec,
                               X + numLocalDofs * totalNumWaveFunctions *
                                     ((dftParams.spinPolarized + 1) * kPoint +
                                      spinIndex),
                               numLocalDofs,
                               totalNumWaveFunctions,
                               deviceFlattenedArrayBlock.begin(),
                               jvec);


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

                              copyGlobalToCellDeviceKernel<<<
                                (BVec + 255) / 256 * currentCellsBlockSize *
                                  numNodesPerElement,
                                256>>>(
                                BVec,
                                currentCellsBlockSize * numNodesPerElement,
                                deviceFlattenedArrayBlock.begin(),
                                cellWaveFunctionMatrix,
                                thrust::raw_pointer_cast(&(
                                  operatorMatrix
                                    .getFlattenedArrayCellLocalProcIndexIdMap()
                                      [startingCellId * numNodesPerElement])));

                              NumberType scalarCoeffAlpha =
                                deviceUtils::makeNumberFromReal<NumberType>(
                                  1.0);
                              NumberType scalarCoeffBeta =
                                deviceUtils::makeNumberFromReal<NumberType>(
                                  0.0);
                              int strideA = BVec * numNodesPerElement;
                              int strideB = 0;
                              int strideC = BVec * numQuadPoints;

                              cublasXgemmStridedBatched(
                                operatorMatrix.getCublasHandle(),
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                BVec,
                                numQuadPoints,
                                numNodesPerElement,
                                &scalarCoeffAlpha,
                                cellWaveFunctionMatrix,
                                BVec,
                                strideA,
                                shapeFunctionValuesTransposedDevice,
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

                                  copyDoubleToNumber(
                                    thrust::raw_pointer_cast(&(
                                      operatorMatrix
                                        .getShapeFunctionGradientValuesXTransposed()
                                          [startingCellId * numNodesPerElement *
                                           numQuadPoints])),
                                    currentCellsBlockSize * numNodesPerElement *
                                      numQuadPoints,
                                    shapeFunctionGradientValuesXTransposedDevice);

                                  copyDoubleToNumber(
                                    thrust::raw_pointer_cast(&(
                                      operatorMatrix
                                        .getShapeFunctionGradientValuesYTransposed()
                                          [startingCellId * numNodesPerElement *
                                           numQuadPoints])),
                                    currentCellsBlockSize * numNodesPerElement *
                                      numQuadPoints,
                                    shapeFunctionGradientValuesYTransposedDevice);

                                  copyDoubleToNumber(
                                    thrust::raw_pointer_cast(&(
                                      operatorMatrix
                                        .getShapeFunctionGradientValuesZTransposed()
                                          [startingCellId * numNodesPerElement *
                                           numQuadPoints])),
                                    currentCellsBlockSize * numNodesPerElement *
                                      numQuadPoints,
                                    shapeFunctionGradientValuesZTransposedDevice);

                                  cublasXgemmStridedBatched(
                                    operatorMatrix.getCublasHandle(),
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    BVec,
                                    numQuadPoints,
                                    numNodesPerElement,
                                    &scalarCoeffAlpha,
                                    cellWaveFunctionMatrix,
                                    BVec,
                                    strideA,
                                    shapeFunctionGradientValuesXTransposedDevice,
                                    numNodesPerElement,
                                    strideB,
                                    &scalarCoeffBeta,
                                    gradRhoWfcContributionsDeviceX.begin(),
                                    BVec,
                                    strideC,
                                    currentCellsBlockSize);


                                  cublasXgemmStridedBatched(
                                    operatorMatrix.getCublasHandle(),
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    BVec,
                                    numQuadPoints,
                                    numNodesPerElement,
                                    &scalarCoeffAlpha,
                                    cellWaveFunctionMatrix,
                                    BVec,
                                    strideA,
                                    shapeFunctionGradientValuesYTransposedDevice,
                                    numNodesPerElement,
                                    strideB,
                                    &scalarCoeffBeta,
                                    gradRhoWfcContributionsDeviceY.begin(),
                                    BVec,
                                    strideC,
                                    currentCellsBlockSize);

                                  cublasXgemmStridedBatched(
                                    operatorMatrix.getCublasHandle(),
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    BVec,
                                    numQuadPoints,
                                    numNodesPerElement,
                                    &scalarCoeffAlpha,
                                    cellWaveFunctionMatrix,
                                    BVec,
                                    strideA,
                                    shapeFunctionGradientValuesZTransposedDevice,
                                    numNodesPerElement,
                                    strideB,
                                    &scalarCoeffBeta,
                                    gradRhoWfcContributionsDeviceZ.begin(),
                                    BVec,
                                    strideC,
                                    currentCellsBlockSize);
                                }



                              computeRhoGradRhoFromInterpolatedValues<<<
                                (BVec + 255) / 256 * numQuadPoints *
                                  currentCellsBlockSize,
                                256>>>(currentCellsBlockSize * numQuadPoints *
                                         BVec,
                                       rhoWfcContributionsDevice.begin(),
                                       gradRhoWfcContributionsDeviceX.begin(),
                                       gradRhoWfcContributionsDeviceY.begin(),
                                       gradRhoWfcContributionsDeviceZ.begin(),
                                       isEvaluateGradRho);


                              cublasXgemm(operatorMatrix.getCublasHandle(),
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
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
                                  cublasXgemm(
                                    operatorMatrix.getCublasHandle(),
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
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


                                  cublasXgemm(
                                    operatorMatrix.getCublasHandle(),
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
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

                                  cublasXgemm(
                                    operatorMatrix.getCublasHandle(),
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
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
                                  deviceUtils::makeNumberFromReal<NumberType>(
                                    -kPointWeights[kPoint] *
                                    spinPolarizedFactor);
                              else
                                *(partialOccupVec.begin() + iEigenVec) =
                                  deviceUtils::makeNumberFromReal<NumberType>(
                                    0.0);
                            }
                        }
                      else
                        {
                          for (unsigned int iEigenVec = 0; iEigenVec < BVec;
                               ++iEigenVec)
                            {
                              *(partialOccupVec.begin() + iEigenVec) =
                                deviceUtils::makeNumberFromReal<NumberType>(
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
                                  kPointWeights[kPoint] * spinPolarizedFactor);
                            }
                        }

                      // partialOccupVecDevice = partialOccupVec;
                      deviceUtils::copyHostVecToDeviceVec(
                        partialOccupVec.begin(),
                        partialOccupVecDevice.begin(),
                        partialOccupVecDevice.size());

                      stridedCopyToBlockKernel<<<(BVec + 255) / 256 *
                                                   numLocalDofs,
                                                 256>>>(
                        BVec,
                        XFrac + numLocalDofs * Nfr *
                                  ((dftParams.spinPolarized + 1) * kPoint +
                                   spinIndex),
                        numLocalDofs,
                        Nfr,
                        deviceFlattenedArrayBlock.begin(),
                        jvec);


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

                              copyGlobalToCellDeviceKernel<<<
                                (BVec + 255) / 256 * currentCellsBlockSize *
                                  numNodesPerElement,
                                256>>>(
                                BVec,
                                currentCellsBlockSize * numNodesPerElement,
                                deviceFlattenedArrayBlock.begin(),
                                cellWaveFunctionMatrix,
                                thrust::raw_pointer_cast(&(
                                  operatorMatrix
                                    .getFlattenedArrayCellLocalProcIndexIdMap()
                                      [startingCellId * numNodesPerElement])));

                              NumberType scalarCoeffAlpha =
                                deviceUtils::makeNumberFromReal<NumberType>(
                                  1.0);
                              NumberType scalarCoeffBeta =
                                deviceUtils::makeNumberFromReal<NumberType>(
                                  0.0);
                              int strideA = BVec * numNodesPerElement;
                              int strideB = 0;
                              int strideC = BVec * numQuadPoints;


                              cublasXgemmStridedBatched(
                                operatorMatrix.getCublasHandle(),
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                BVec,
                                numQuadPoints,
                                numNodesPerElement,
                                &scalarCoeffAlpha,
                                cellWaveFunctionMatrix,
                                BVec,
                                strideA,
                                shapeFunctionValuesTransposedDevice,
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

                                  copyDoubleToNumber(
                                    thrust::raw_pointer_cast(&(
                                      operatorMatrix
                                        .getShapeFunctionGradientValuesXTransposed()
                                          [startingCellId * numNodesPerElement *
                                           numQuadPoints])),
                                    currentCellsBlockSize * numNodesPerElement *
                                      numQuadPoints,
                                    shapeFunctionGradientValuesXTransposedDevice);

                                  copyDoubleToNumber(
                                    thrust::raw_pointer_cast(&(
                                      operatorMatrix
                                        .getShapeFunctionGradientValuesYTransposed()
                                          [startingCellId * numNodesPerElement *
                                           numQuadPoints])),
                                    currentCellsBlockSize * numNodesPerElement *
                                      numQuadPoints,
                                    shapeFunctionGradientValuesYTransposedDevice);

                                  copyDoubleToNumber(
                                    thrust::raw_pointer_cast(&(
                                      operatorMatrix
                                        .getShapeFunctionGradientValuesZTransposed()
                                          [startingCellId * numNodesPerElement *
                                           numQuadPoints])),
                                    currentCellsBlockSize * numNodesPerElement *
                                      numQuadPoints,
                                    shapeFunctionGradientValuesZTransposedDevice);

                                  cublasXgemmStridedBatched(
                                    operatorMatrix.getCublasHandle(),
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    BVec,
                                    numQuadPoints,
                                    numNodesPerElement,
                                    &scalarCoeffAlpha,
                                    cellWaveFunctionMatrix,
                                    BVec,
                                    strideA,
                                    shapeFunctionGradientValuesXTransposedDevice,
                                    numNodesPerElement,
                                    strideB,
                                    &scalarCoeffBeta,
                                    gradRhoWfcContributionsDeviceX.begin(),
                                    BVec,
                                    strideC,
                                    currentCellsBlockSize);


                                  cublasXgemmStridedBatched(
                                    operatorMatrix.getCublasHandle(),
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    BVec,
                                    numQuadPoints,
                                    numNodesPerElement,
                                    &scalarCoeffAlpha,
                                    cellWaveFunctionMatrix,
                                    BVec,
                                    strideA,
                                    shapeFunctionGradientValuesYTransposedDevice,
                                    numNodesPerElement,
                                    strideB,
                                    &scalarCoeffBeta,
                                    gradRhoWfcContributionsDeviceY.begin(),
                                    BVec,
                                    strideC,
                                    currentCellsBlockSize);

                                  cublasXgemmStridedBatched(
                                    operatorMatrix.getCublasHandle(),
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    BVec,
                                    numQuadPoints,
                                    numNodesPerElement,
                                    &scalarCoeffAlpha,
                                    cellWaveFunctionMatrix,
                                    BVec,
                                    strideA,
                                    shapeFunctionGradientValuesZTransposedDevice,
                                    numNodesPerElement,
                                    strideB,
                                    &scalarCoeffBeta,
                                    gradRhoWfcContributionsDeviceZ.begin(),
                                    BVec,
                                    strideC,
                                    currentCellsBlockSize);
                                }



                              computeRhoGradRhoFromInterpolatedValues<<<
                                (BVec + 255) / 256 * numQuadPoints *
                                  currentCellsBlockSize,
                                256>>>(currentCellsBlockSize * numQuadPoints *
                                         BVec,
                                       rhoWfcContributionsDevice.begin(),
                                       gradRhoWfcContributionsDeviceX.begin(),
                                       gradRhoWfcContributionsDeviceY.begin(),
                                       gradRhoWfcContributionsDeviceZ.begin(),
                                       isEvaluateGradRho);


                              cublasXgemm(operatorMatrix.getCublasHandle(),
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
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
                                  cublasXgemm(
                                    operatorMatrix.getCublasHandle(),
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
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


                                  cublasXgemm(
                                    operatorMatrix.getCublasHandle(),
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
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

                                  cublasXgemm(
                                    operatorMatrix.getCublasHandle(),
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
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


              // do cuda memcopy to host
              deviceUtils::copyDeviceVecToHostVec(rhoDevice.begin(),
                                                  rhoHost.begin(),
                                                  totalLocallyOwnedCells *
                                                    numQuadPoints);

              if (isEvaluateGradRho)
                {
                  deviceUtils::copyDeviceVecToHostVec(gradRhoDeviceX.begin(),
                                                      gradRhoHostX.begin(),
                                                      totalLocallyOwnedCells *
                                                        numQuadPoints);

                  deviceUtils::copyDeviceVecToHostVec(gradRhoDeviceY.begin(),
                                                      gradRhoHostY.begin(),
                                                      totalLocallyOwnedCells *
                                                        numQuadPoints);

                  deviceUtils::copyDeviceVecToHostVec(gradRhoDeviceZ.begin(),
                                                      gradRhoHostZ.begin(),
                                                      totalLocallyOwnedCells *
                                                        numQuadPoints);
                }

              for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
                for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
                  {
                    rhoValuesFlattened[icell * numQuadPoints + iquad] +=
                      deviceUtils::makeRealFromNumber(
                        *(rhoHost.begin() + icell * numQuadPoints + iquad));
                  }

              if (isEvaluateGradRho)
                for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
                  for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
                    {
                      gradRhoValuesFlattened[icell * numQuadPoints * 3 +
                                             3 * iquad + 0] +=
                        deviceUtils::makeRealFromNumber(
                          *(gradRhoHostX.begin() + icell * numQuadPoints +
                            iquad));
                      gradRhoValuesFlattened[icell * numQuadPoints * 3 +
                                             3 * iquad + 1] +=
                        deviceUtils::makeRealFromNumber(
                          *(gradRhoHostY.begin() + icell * numQuadPoints +
                            iquad));
                      gradRhoValuesFlattened[icell * numQuadPoints * 3 +
                                             3 * iquad + 2] +=
                        deviceUtils::makeRealFromNumber(
                          *(gradRhoHostZ.begin() + icell * numQuadPoints +
                            iquad));
                    }
              if (dftParams.spinPolarized == 1)
                {
                  for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
                    for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
                      {
                        rhoValuesSpinPolarizedFlattened
                          [icell * numQuadPoints * 2 + iquad * 2 + spinIndex] +=
                          deviceUtils::makeRealFromNumber(
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
                            deviceUtils::makeRealFromNumber(
                              *(gradRhoHostX.begin() + icell * numQuadPoints +
                                iquad));
                          gradRhoValuesSpinPolarizedFlattened
                            [icell * numQuadPoints * 6 + iquad * 6 +
                             spinIndex * 3 + 1] +=
                            deviceUtils::makeRealFromNumber(
                              *(gradRhoHostY.begin() + icell * numQuadPoints +
                                iquad));
                          gradRhoValuesSpinPolarizedFlattened
                            [icell * numQuadPoints * 6 + iquad * 6 +
                             spinIndex * 3 + 2] +=
                            deviceUtils::makeRealFromNumber(
                              *(gradRhoHostZ.begin() + icell * numQuadPoints +
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



      DeviceCHECK(cudaFree(shapeFunctionValuesTransposedDevice));

      if (isEvaluateGradRho)
        {
          DeviceCHECK(cudaFree(shapeFunctionGradientValuesXTransposedDevice));
          DeviceCHECK(cudaFree(shapeFunctionGradientValuesYTransposedDevice));
          DeviceCHECK(cudaFree(shapeFunctionGradientValuesZTransposedDevice));
        }

      cudaDeviceSynchronize();
      MPI_Barrier(mpiCommParent);
      device_time = MPI_Wtime() - device_time;

      if (this_process == 0 && dftParams.verbosity >= 2)
        std::cout << "Time for compute rho on Device: " << device_time
                  << std::endl;
    }

    template void
    computeRhoFromPSI(
      const dataTypes::numberDevice *                X,
      const dataTypes::numberDevice *                XFrac,
      const unsigned int                             totalNumWaveFunctions,
      const unsigned int                             Nfr,
      const unsigned int                             numLocalDofs,
      const std::vector<std::vector<double>> &       eigenValues,
      const double                                   fermiEnergy,
      const double                                   fermiEnergyUp,
      const double                                   fermiEnergyDown,
      operatorDFTDeviceClass &                       operatorMatrix,
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
