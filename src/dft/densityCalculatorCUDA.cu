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

// source file for electron density related computations
#include <constants.h>
#include <densityCalculatorCUDA.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <vectorUtilities.h>
#include <cuComplex.h>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include "cudaHelpers.h"
#include "linearAlgebraOperationsCUDA.h"


namespace dftfe
{
  namespace CUDA
  {
    namespace
    {
      void
      sumRhoData(
        const dealii::DoFHandler<3> &                  dofHandler,
        std::map<dealii::CellId, std::vector<double>> *rhoValues,
        std::map<dealii::CellId, std::vector<double>> *gradRhoValues,
        std::map<dealii::CellId, std::vector<double>> *rhoValuesSpinPolarized,
        std::map<dealii::CellId, std::vector<double>>
          *             gradRhoValuesSpinPolarized,
        const bool      isGradRhoDataPresent,
        const MPI_Comm &interComm)
      {
        typename dealii::DoFHandler<3>::active_cell_iterator
          cell = dofHandler.begin_active(),
          endc = dofHandler.end();

        // gather density from inter communicator
        if (dealii::Utilities::MPI::n_mpi_processes(interComm) > 1)
          for (; cell != endc; ++cell)
            if (cell->is_locally_owned())
              {
                const dealii::CellId cellId = cell->id();

                dealii::Utilities::MPI::sum((*rhoValues)[cellId],
                                            interComm,
                                            (*rhoValues)[cellId]);
                if (isGradRhoDataPresent)
                  dealii::Utilities::MPI::sum((*gradRhoValues)[cellId],
                                              interComm,
                                              (*gradRhoValues)[cellId]);

                if (dftParameters::spinPolarized == 1)
                  {
                    dealii::Utilities::MPI::sum(
                      (*rhoValuesSpinPolarized)[cellId],
                      interComm,
                      (*rhoValuesSpinPolarized)[cellId]);
                    if (isGradRhoDataPresent)
                      dealii::Utilities::MPI::sum(
                        (*gradRhoValuesSpinPolarized)[cellId],
                        interComm,
                        (*gradRhoValuesSpinPolarized)[cellId]);
                  }
              }
      }

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
      copyGlobalToCellCUDAKernel(const unsigned int contiguousBlockSize,
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
      copyCUDAKernel(const unsigned int size,
                     const double *     copyFromVec,
                     double *           copyToVec)
      {
        for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
             index < size;
             index += blockDim.x * gridDim.x)
          copyToVec[index] = copyFromVec[index];
      }

      __global__ void
      copyCUDAKernel(const unsigned int size,
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
        copyCUDAKernel<<<(size + 255) / 256, 256>>>(size,
                                                    copyFromVec,
                                                    copyToVec);
      }

      void
      copyDoubleToNumber(const double *     copyFromVec,
                         const unsigned int size,
                         cuDoubleComplex *  copyToVec)
      {
        copyCUDAKernel<<<(size + 255) / 256, 256>>>(size,
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
                  2.0 * (psi.x * gradPsiX.x - psi.y * gradPsiX.y), 0.0);

                const cuDoubleComplex gradPsiY =
                  gradRhoCellsWfcContributionsY[index];
                gradRhoCellsWfcContributionsY[index] = make_cuDoubleComplex(
                  2.0 * (psi.x * gradPsiY.x - psi.y * gradPsiY.y), 0.0);

                const cuDoubleComplex gradPsiZ =
                  gradRhoCellsWfcContributionsZ[index];
                gradRhoCellsWfcContributionsZ[index] = make_cuDoubleComplex(
                  2.0 * (psi.x * gradPsiZ.x - psi.y * gradPsiZ.y), 0.0);
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
      operatorDFTCUDAClass &                         operatorMatrix,
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
      const MPI_Comm &                               interpoolcomm,
      const MPI_Comm &                               interBandGroupComm,
      const bool                                     spectrumSplit,
      const bool                                     use2pPlusOneGLQuad)
    {
      if (use2pPlusOneGLQuad)
        AssertThrow(!isEvaluateGradRho, dftUtils::ExcNotImplementedYet());

      int this_process;
      MPI_Comm_rank(MPI_COMM_WORLD, &this_process);
      cudaDeviceSynchronize();
      MPI_Barrier(MPI_COMM_WORLD);
      double             gpu_time   = MPI_Wtime();
      const unsigned int numKPoints = kPointWeights.size();

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
        std::min(dftParameters::chebyWfcBlockSize, totalNumWaveFunctions);

      const double spinPolarizedFactor =
        (dftParameters::spinPolarized == 1) ? 1.0 : 2.0;

      const NumberType zero = cudaUtils::makeNumberFromReal<NumberType>(0.0);
      const NumberType scalarCoeffAlphaRho =
        cudaUtils::makeNumberFromReal<NumberType>(1.0);
      const NumberType scalarCoeffBetaRho =
        cudaUtils::makeNumberFromReal<NumberType>(1.0);
      const NumberType scalarCoeffAlphaGradRho =
        cudaUtils::makeNumberFromReal<NumberType>(1.0);
      const NumberType scalarCoeffBetaGradRho =
        cudaUtils::makeNumberFromReal<NumberType>(1.0);

      const unsigned int cellsBlockSize = 50;
      const unsigned int numCellBlocks =
        totalLocallyOwnedCells / cellsBlockSize;
      const unsigned int remCellBlockSize =
        totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;

      cudaUtils::Vector<NumberType, dftfe::MemorySpace::GPU> rhoDevice(
        totalLocallyOwnedCells * numQuadPoints, zero);
      cudaUtils::Vector<NumberType, dftfe::MemorySpace::GPU>
        rhoWfcContributionsDevice(cellsBlockSize * numQuadPoints * BVec, zero);

      cudaUtils::Vector<NumberType, dftfe::MemorySpace::GPU> gradRhoDeviceX(
        isEvaluateGradRho ? (totalLocallyOwnedCells * numQuadPoints) : 1, zero);
      cudaUtils::Vector<NumberType, dftfe::MemorySpace::GPU> gradRhoDeviceY(
        isEvaluateGradRho ? (totalLocallyOwnedCells * numQuadPoints) : 1, zero);
      cudaUtils::Vector<NumberType, dftfe::MemorySpace::GPU> gradRhoDeviceZ(
        isEvaluateGradRho ? (totalLocallyOwnedCells * numQuadPoints) : 1, zero);
      cudaUtils::Vector<NumberType, dftfe::MemorySpace::GPU>
        gradRhoWfcContributionsDeviceX(
          isEvaluateGradRho ? (cellsBlockSize * numQuadPoints * BVec) : 1,
          zero);
      cudaUtils::Vector<NumberType, dftfe::MemorySpace::GPU>
        gradRhoWfcContributionsDeviceY(
          isEvaluateGradRho ? (cellsBlockSize * numQuadPoints * BVec) : 1,
          zero);
      cudaUtils::Vector<NumberType, dftfe::MemorySpace::GPU>
        gradRhoWfcContributionsDeviceZ(
          isEvaluateGradRho ? (cellsBlockSize * numQuadPoints * BVec) : 1,
          zero);

      NumberType *shapeFunctionValuesInvertedDevice;

      cudaMalloc((void **)&shapeFunctionValuesInvertedDevice,
                 numNodesPerElement * numQuadPoints * sizeof(NumberType));
      cudaMemset(shapeFunctionValuesInvertedDevice,
                 0,
                 numNodesPerElement * numQuadPoints * sizeof(NumberType));

      copyDoubleToNumber(thrust::raw_pointer_cast(
                           &(operatorMatrix.getShapeFunctionValuesInverted(
                             use2pPlusOneGLQuad)[0])),
                         numNodesPerElement * numQuadPoints,
                         shapeFunctionValuesInvertedDevice);


      NumberType *shapeFunctionGradientValuesXInvertedDevice;
      NumberType *shapeFunctionGradientValuesYInvertedDevice;
      NumberType *shapeFunctionGradientValuesZInvertedDevice;

      if (isEvaluateGradRho)
        {
          cudaMalloc((void **)&shapeFunctionGradientValuesXInvertedDevice,
                     cellsBlockSize * numNodesPerElement * numQuadPoints *
                       sizeof(NumberType));
          cudaMemset(shapeFunctionGradientValuesXInvertedDevice,
                     0,
                     cellsBlockSize * numNodesPerElement * numQuadPoints *
                       sizeof(NumberType));

          cudaMalloc((void **)&shapeFunctionGradientValuesYInvertedDevice,
                     cellsBlockSize * numNodesPerElement * numQuadPoints *
                       sizeof(NumberType));
          cudaMemset(shapeFunctionGradientValuesYInvertedDevice,
                     0,
                     cellsBlockSize * numNodesPerElement * numQuadPoints *
                       sizeof(NumberType));

          cudaMalloc((void **)&shapeFunctionGradientValuesZInvertedDevice,
                     cellsBlockSize * numNodesPerElement * numQuadPoints *
                       sizeof(NumberType));
          cudaMemset(shapeFunctionGradientValuesZInvertedDevice,
                     0,
                     cellsBlockSize * numNodesPerElement * numQuadPoints *
                       sizeof(NumberType));
        }

      cudaUtils::Vector<NumberType, dftfe::MemorySpace::Host> partialOccupVec(
        BVec, zero);
      cudaUtils::Vector<NumberType, dftfe::MemorySpace::GPU>
        partialOccupVecDevice(BVec, zero);

      distributedGPUVec<NumberType> &cudaFlattenedArrayBlock =
        operatorMatrix.getParallelChebyBlockVectorDevice();

      const unsigned int numGhosts =
        cudaFlattenedArrayBlock.ghostFlattenedSize();

      NumberType *cellWaveFunctionMatrix =
        reinterpret_cast<NumberType *>(thrust::raw_pointer_cast(
          &operatorMatrix.getCellWaveFunctionMatrix()[0]));

      // set density to zero
      typename dealii::DoFHandler<3>::active_cell_iterator cell =
        dofHandler.begin_active();
      typename dealii::DoFHandler<3>::active_cell_iterator endc =
        dofHandler.end();
      for (; cell != endc; ++cell)
        if (cell->is_locally_owned())
          {
            const dealii::CellId cellid = cell->id();

            std::fill((*rhoValues)[cellid].begin(),
                      (*rhoValues)[cellid].end(),
                      0.0);
            if (isEvaluateGradRho)
              std::fill((*gradRhoValues)[cellid].begin(),
                        (*gradRhoValues)[cellid].end(),
                        0.0);

            if (dftParameters::spinPolarized == 1)
              {
                std::fill((*rhoValuesSpinPolarized)[cellid].begin(),
                          (*rhoValuesSpinPolarized)[cellid].end(),
                          0.0);
                if (isEvaluateGradRho)
                  std::fill((*gradRhoValuesSpinPolarized)[cellid].begin(),
                            (*gradRhoValuesSpinPolarized)[cellid].end(),
                            0.0);
              }
          }

      for (unsigned int spinIndex = 0;
           spinIndex < (1 + dftParameters::spinPolarized);
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
                            cudaUtils::makeNumberFromReal<NumberType>(
                              kPointWeights[kPoint] * spinPolarizedFactor));
                        }
                      else
                        {
                          if (dftParameters::constraintMagnetization)
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
                                      cudaUtils::makeNumberFromReal<NumberType>(
                                        0.0);
                                  else
                                    *(partialOccupVec.begin() + iEigenVec) =
                                      cudaUtils::makeNumberFromReal<NumberType>(
                                        kPointWeights[kPoint] *
                                        spinPolarizedFactor);
                                }
                            }
                          else
                            {
                              for (unsigned int iEigenVec = 0; iEigenVec < BVec;
                                   ++iEigenVec)
                                {
                                  *(partialOccupVec.begin() + iEigenVec) =
                                    cudaUtils::makeNumberFromReal<NumberType>(
                                      dftUtils::getPartialOccupancy(
                                        eigenValues[kPoint]
                                                   [totalNumWaveFunctions *
                                                      spinIndex +
                                                    jvec + iEigenVec],
                                        fermiEnergy,
                                        C_kb,
                                        dftParameters::TVal) *
                                      kPointWeights[kPoint] *
                                      spinPolarizedFactor);
                                }
                            }

                          // partialOccupVecDevice = partialOccupVec;

                          cudaUtils::copyHostVecToCUDAVec(
                            partialOccupVec.begin(),
                            partialOccupVecDevice.begin(),
                            partialOccupVecDevice.size());
                        }

                      stridedCopyToBlockKernel<<<(BVec + 255) / 256 *
                                                   numLocalDofs,
                                                 256>>>(
                        BVec,
                        X + numLocalDofs * totalNumWaveFunctions *
                              ((dftParameters::spinPolarized + 1) * kPoint +
                               spinIndex),
                        numLocalDofs,
                        totalNumWaveFunctions,
                        cudaFlattenedArrayBlock.begin(),
                        jvec);


                      cudaFlattenedArrayBlock.updateGhostValues();

                      (operatorMatrix.getOverloadedConstraintMatrix())
                        ->distribute(cudaFlattenedArrayBlock, BVec);

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

                              copyGlobalToCellCUDAKernel<<<
                                (BVec + 255) / 256 * currentCellsBlockSize *
                                  numNodesPerElement,
                                256>>>(
                                BVec,
                                currentCellsBlockSize * numNodesPerElement,
                                cudaFlattenedArrayBlock.begin(),
                                cellWaveFunctionMatrix,
                                thrust::raw_pointer_cast(&(
                                  operatorMatrix
                                    .getFlattenedArrayCellLocalProcIndexIdMap()
                                      [startingCellId * numNodesPerElement])));

                              NumberType scalarCoeffAlpha =
                                cudaUtils::makeNumberFromReal<NumberType>(1.0);
                              NumberType scalarCoeffBeta =
                                cudaUtils::makeNumberFromReal<NumberType>(0.0);
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
                                shapeFunctionValuesInvertedDevice,
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
                                        .getShapeFunctionGradientValuesXInverted()
                                          [startingCellId * numNodesPerElement *
                                           numQuadPoints])),
                                    currentCellsBlockSize * numNodesPerElement *
                                      numQuadPoints,
                                    shapeFunctionGradientValuesXInvertedDevice);

                                  copyDoubleToNumber(
                                    thrust::raw_pointer_cast(&(
                                      operatorMatrix
                                        .getShapeFunctionGradientValuesYInverted()
                                          [startingCellId * numNodesPerElement *
                                           numQuadPoints])),
                                    currentCellsBlockSize * numNodesPerElement *
                                      numQuadPoints,
                                    shapeFunctionGradientValuesYInvertedDevice);

                                  copyDoubleToNumber(
                                    thrust::raw_pointer_cast(&(
                                      operatorMatrix
                                        .getShapeFunctionGradientValuesZInverted()
                                          [startingCellId * numNodesPerElement *
                                           numQuadPoints])),
                                    currentCellsBlockSize * numNodesPerElement *
                                      numQuadPoints,
                                    shapeFunctionGradientValuesZInvertedDevice);

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
                                    shapeFunctionGradientValuesXInvertedDevice,
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
                                    shapeFunctionGradientValuesYInvertedDevice,
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
                                    shapeFunctionGradientValuesZInvertedDevice,
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
                      if (dftParameters::constraintMagnetization)
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
                                  cudaUtils::makeNumberFromReal<NumberType>(
                                    -kPointWeights[kPoint] *
                                    spinPolarizedFactor);
                              else
                                *(partialOccupVec.begin() + iEigenVec) =
                                  cudaUtils::makeNumberFromReal<NumberType>(
                                    0.0);
                            }
                        }
                      else
                        {
                          for (unsigned int iEigenVec = 0; iEigenVec < BVec;
                               ++iEigenVec)
                            {
                              *(partialOccupVec.begin() + iEigenVec) =
                                cudaUtils::makeNumberFromReal<NumberType>(
                                  (dftUtils::getPartialOccupancy(
                                     eigenValues[kPoint]
                                                [totalNumWaveFunctions *
                                                   spinIndex +
                                                 (totalNumWaveFunctions - Nfr) +
                                                 jvec + iEigenVec],
                                     fermiEnergy,
                                     C_kb,
                                     dftParameters::TVal) -
                                   1.0) *
                                  kPointWeights[kPoint] * spinPolarizedFactor);
                            }
                        }

                      // partialOccupVecDevice = partialOccupVec;
                      cudaUtils::copyHostVecToCUDAVec(
                        partialOccupVec.begin(),
                        partialOccupVecDevice.begin(),
                        partialOccupVecDevice.size());

                      stridedCopyToBlockKernel<<<(BVec + 255) / 256 *
                                                   numLocalDofs,
                                                 256>>>(
                        BVec,
                        XFrac + numLocalDofs * Nfr *
                                  ((dftParameters::spinPolarized + 1) * kPoint +
                                   spinIndex),
                        numLocalDofs,
                        Nfr,
                        cudaFlattenedArrayBlock.begin(),
                        jvec);


                      cudaFlattenedArrayBlock.updateGhostValues();

                      (operatorMatrix.getOverloadedConstraintMatrix())
                        ->distribute(cudaFlattenedArrayBlock, BVec);

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

                              copyGlobalToCellCUDAKernel<<<
                                (BVec + 255) / 256 * currentCellsBlockSize *
                                  numNodesPerElement,
                                256>>>(
                                BVec,
                                currentCellsBlockSize * numNodesPerElement,
                                cudaFlattenedArrayBlock.begin(),
                                cellWaveFunctionMatrix,
                                thrust::raw_pointer_cast(&(
                                  operatorMatrix
                                    .getFlattenedArrayCellLocalProcIndexIdMap()
                                      [startingCellId * numNodesPerElement])));

                              NumberType scalarCoeffAlpha =
                                cudaUtils::makeNumberFromReal<NumberType>(1.0);
                              NumberType scalarCoeffBeta =
                                cudaUtils::makeNumberFromReal<NumberType>(0.0);
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
                                shapeFunctionValuesInvertedDevice,
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
                                        .getShapeFunctionGradientValuesXInverted()
                                          [startingCellId * numNodesPerElement *
                                           numQuadPoints])),
                                    currentCellsBlockSize * numNodesPerElement *
                                      numQuadPoints,
                                    shapeFunctionGradientValuesXInvertedDevice);

                                  copyDoubleToNumber(
                                    thrust::raw_pointer_cast(&(
                                      operatorMatrix
                                        .getShapeFunctionGradientValuesYInverted()
                                          [startingCellId * numNodesPerElement *
                                           numQuadPoints])),
                                    currentCellsBlockSize * numNodesPerElement *
                                      numQuadPoints,
                                    shapeFunctionGradientValuesYInvertedDevice);

                                  copyDoubleToNumber(
                                    thrust::raw_pointer_cast(&(
                                      operatorMatrix
                                        .getShapeFunctionGradientValuesZInverted()
                                          [startingCellId * numNodesPerElement *
                                           numQuadPoints])),
                                    currentCellsBlockSize * numNodesPerElement *
                                      numQuadPoints,
                                    shapeFunctionGradientValuesZInvertedDevice);

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
                                    shapeFunctionGradientValuesXInvertedDevice,
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
                                    shapeFunctionGradientValuesYInvertedDevice,
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
                                    shapeFunctionGradientValuesZInvertedDevice,
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
              cudaUtils::Vector<NumberType, dftfe::MemorySpace::Host> rhoHost;
              cudaUtils::Vector<NumberType, dftfe::MemorySpace::Host>
                gradRhoHostX;
              cudaUtils::Vector<NumberType, dftfe::MemorySpace::Host>
                gradRhoHostY;
              cudaUtils::Vector<NumberType, dftfe::MemorySpace::Host>
                gradRhoHostZ;

              rhoHost.resize(totalLocallyOwnedCells * numQuadPoints, zero);
              cudaUtils::copyCUDAVecToHostVec(rhoDevice.begin(),
                                              rhoHost.begin(),
                                              totalLocallyOwnedCells *
                                                numQuadPoints);

              if (isEvaluateGradRho)
                {
                  gradRhoHostX.resize(totalLocallyOwnedCells * numQuadPoints,
                                      zero);
                  cudaUtils::copyCUDAVecToHostVec(gradRhoDeviceX.begin(),
                                                  gradRhoHostX.begin(),
                                                  totalLocallyOwnedCells *
                                                    numQuadPoints);

                  gradRhoHostY.resize(totalLocallyOwnedCells * numQuadPoints,
                                      zero);
                  cudaUtils::copyCUDAVecToHostVec(gradRhoDeviceY.begin(),
                                                  gradRhoHostY.begin(),
                                                  totalLocallyOwnedCells *
                                                    numQuadPoints);

                  gradRhoHostZ.resize(totalLocallyOwnedCells * numQuadPoints,
                                      zero);
                  cudaUtils::copyCUDAVecToHostVec(gradRhoDeviceZ.begin(),
                                                  gradRhoHostZ.begin(),
                                                  totalLocallyOwnedCells *
                                                    numQuadPoints);
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
                      (dftParameters::spinPolarized == 1) ?
                        (*rhoValuesSpinPolarized)[cellid] :
                        dummy;
                    std::vector<double> &tempGradRhoQuadsSP =
                      ((dftParameters::spinPolarized == 1) &&
                       isEvaluateGradRho) ?
                        (*gradRhoValuesSpinPolarized)[cellid] :
                        dummy;

                    for (unsigned int q = 0; q < numQuadPoints; ++q)
                      {
                        if (dftParameters::spinPolarized == 1)
                          {
                            tempRhoQuadsSP[2 * q + spinIndex] +=
                              cudaUtils::makeRealFromNumber(
                                *(rhoHost.begin() + iElem * numQuadPoints + q));

                            if (isEvaluateGradRho)
                              {
                                tempGradRhoQuadsSP[6 * q + spinIndex * 3] +=
                                  cudaUtils::makeRealFromNumber(
                                    *(gradRhoHostX.begin() +
                                      iElem * numQuadPoints + q));
                                tempGradRhoQuadsSP[6 * q + 1 + spinIndex * 3] +=
                                  cudaUtils::makeRealFromNumber(
                                    *(gradRhoHostY.begin() +
                                      iElem * numQuadPoints + q));
                                tempGradRhoQuadsSP[6 * q + 2 + spinIndex * 3] +=
                                  cudaUtils::makeRealFromNumber(
                                    *(gradRhoHostZ.begin() +
                                      iElem * numQuadPoints + q));
                              }
                          }

                        tempRhoQuads[q] += cudaUtils::makeRealFromNumber(
                          *(rhoHost.begin() + iElem * numQuadPoints + q));


                        if (isEvaluateGradRho)
                          {
                            tempGradRhoQuads[3 * q] +=
                              cudaUtils::makeRealFromNumber(
                                *(gradRhoHostX.begin() + iElem * numQuadPoints +
                                  q));
                            tempGradRhoQuads[3 * q + 1] +=
                              cudaUtils::makeRealFromNumber(
                                *(gradRhoHostY.begin() + iElem * numQuadPoints +
                                  q));
                            tempGradRhoQuads[3 * q + 2] +=
                              cudaUtils::makeRealFromNumber(
                                *(gradRhoHostZ.begin() + iElem * numQuadPoints +
                                  q));
                          }
                      }
                    iElem++;
                  }
            } // kpoint loop
        }     // spin index



      // gather density from all inter communicators
      sumRhoData(dofHandler,
                 rhoValues,
                 gradRhoValues,
                 rhoValuesSpinPolarized,
                 gradRhoValuesSpinPolarized,
                 isEvaluateGradRho,
                 interBandGroupComm);

      sumRhoData(dofHandler,
                 rhoValues,
                 gradRhoValues,
                 rhoValuesSpinPolarized,
                 gradRhoValuesSpinPolarized,
                 isEvaluateGradRho,
                 interpoolcomm);

      cudaFree(shapeFunctionValuesInvertedDevice);

      if (isEvaluateGradRho)
        {
          cudaFree(shapeFunctionGradientValuesXInvertedDevice);
          cudaFree(shapeFunctionGradientValuesYInvertedDevice);
          cudaFree(shapeFunctionGradientValuesZInvertedDevice);
        }

      cudaDeviceSynchronize();
      MPI_Barrier(MPI_COMM_WORLD);
      gpu_time = MPI_Wtime() - gpu_time;

      if (this_process == 0 && dftParameters::verbosity >= 2)
        std::cout << "Time for compute rho on GPU: " << gpu_time << std::endl;
    }

    template void
    computeRhoFromPSI(
      const dataTypes::numberGPU *                   X,
      const dataTypes::numberGPU *                   XFrac,
      const unsigned int                             totalNumWaveFunctions,
      const unsigned int                             Nfr,
      const unsigned int                             numLocalDofs,
      const std::vector<std::vector<double>> &       eigenValues,
      const double                                   fermiEnergy,
      const double                                   fermiEnergyUp,
      const double                                   fermiEnergyDown,
      operatorDFTCUDAClass &                         operatorMatrix,
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
      const MPI_Comm &                               interpoolcomm,
      const MPI_Comm &                               interBandGroupComm,
      const bool                                     spectrumSplit,
      const bool                                     use2pPlusOneGLQuad);
  } // namespace CUDA
} // namespace dftfe
