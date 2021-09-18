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

      __global__ void
      stridedCopyToBlockKernel(const unsigned int BVec,
                               const double *     xVec,
                               const unsigned int M,
                               const unsigned int N,
                               double *           yVec,
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


      __global__ void
      copyCUDAKernel(const unsigned int contiguousBlockSize,
                     const unsigned int numContiguousBlocks,
                     const double *     copyFromVec,
                     double *           copyToVec,
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


    } // namespace

    void
    computeRhoFromPSI(
      const double *                                 X,
      const double *                                 XFrac,
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
#ifdef USE_COMPLEX
      AssertThrow(false, dftUtils::ExcNotImplementedYet());
#else
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

      const double scalarCoeffAlphaRho     = 1.0;
      const double scalarCoeffBetaRho      = 1.0;
      const double scalarCoeffAlphaGradRho = 1.0;
      const double scalarCoeffBetaGradRho  = 1.0;

      const unsigned int cellsBlockSize = 50;
      const unsigned int numCellBlocks =
        totalLocallyOwnedCells / cellsBlockSize;
      const unsigned int remCellBlockSize =
        totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;

      thrust::device_vector<double> rhoDevice(totalLocallyOwnedCells *
                                                numQuadPoints,
                                              0.0);
      thrust::device_vector<double> rhoWfcContributionsDevice(
        cellsBlockSize * numQuadPoints * BVec, 0.0);

      thrust::device_vector<double> gradRhoDeviceX(
        isEvaluateGradRho ? (totalLocallyOwnedCells * numQuadPoints) : 1, 0.0);
      thrust::device_vector<double> gradRhoDeviceY(
        isEvaluateGradRho ? (totalLocallyOwnedCells * numQuadPoints) : 1, 0.0);
      thrust::device_vector<double> gradRhoDeviceZ(
        isEvaluateGradRho ? (totalLocallyOwnedCells * numQuadPoints) : 1, 0.0);
      thrust::device_vector<double> gradRhoWfcContributionsDeviceX(
        isEvaluateGradRho ? (cellsBlockSize * numQuadPoints * BVec) : 1, 0.0);
      thrust::device_vector<double> gradRhoWfcContributionsDeviceY(
        isEvaluateGradRho ? (cellsBlockSize * numQuadPoints * BVec) : 1, 0.0);
      thrust::device_vector<double> gradRhoWfcContributionsDeviceZ(
        isEvaluateGradRho ? (cellsBlockSize * numQuadPoints * BVec) : 1, 0.0);

      std::vector<double>           partialOccupVec(BVec, 0.0);
      thrust::device_vector<double> partialOccupVecDevice(BVec, 0.0);

      // distributedGPUVec<double> & cudaFlattenedArrayBlock =
      // operatorMatrix.getBlockCUDADealiiVector();

      distributedGPUVec<double> &cudaFlattenedArrayBlock =
        operatorMatrix.getParallelChebyBlockVectorDevice();

      const unsigned int numGhosts =
        cudaFlattenedArrayBlock.ghostFlattenedSize();

      thrust::device_vector<double> &cellWaveFunctionMatrix =
        operatorMatrix.getCellWaveFunctionMatrix();

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
          thrust::fill(rhoDevice.begin(), rhoDevice.end(), 0.0);
          thrust::fill(rhoWfcContributionsDevice.begin(),
                       rhoWfcContributionsDevice.end(),
                       0.0);
          thrust::fill(gradRhoDeviceX.begin(), gradRhoDeviceX.end(), 0.0);
          thrust::fill(gradRhoDeviceY.begin(), gradRhoDeviceY.end(), 0.0);
          thrust::fill(gradRhoDeviceZ.begin(), gradRhoDeviceZ.end(), 0.0);
          thrust::fill(gradRhoWfcContributionsDeviceX.begin(),
                       gradRhoWfcContributionsDeviceX.end(),
                       0.0);
          thrust::fill(gradRhoWfcContributionsDeviceY.begin(),
                       gradRhoWfcContributionsDeviceY.end(),
                       0.0);
          thrust::fill(gradRhoWfcContributionsDeviceZ.begin(),
                       gradRhoWfcContributionsDeviceZ.end(),
                       0.0);

          for (unsigned int jvec = 0; jvec < totalNumWaveFunctions;
               jvec += BVec)
            {
              if ((jvec + BVec) <=
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                  (jvec + BVec) >
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  if (spectrumSplit)
                    {
                      thrust::fill(partialOccupVecDevice.begin(),
                                   partialOccupVecDevice.end(),
                                   spinPolarizedFactor);
                    }
                  else
                    {
                      if (dftParameters::constraintMagnetization)
                        {
                          const double fermiEnergyConstraintMag =
                            spinIndex == 0 ? fermiEnergyUp : fermiEnergyDown;
                          for (unsigned int iEigenVec = 0; iEigenVec < BVec;
                               ++iEigenVec)
                            {
                              if (eigenValues[0][totalNumWaveFunctions *
                                                   spinIndex +
                                                 jvec + iEigenVec] >
                                  fermiEnergyConstraintMag)
                                partialOccupVec[iEigenVec] = 0.0;
                              else
                                partialOccupVec[iEigenVec] =
                                  spinPolarizedFactor;
                            }
                        }
                      else
                        {
                          for (unsigned int iEigenVec = 0; iEigenVec < BVec;
                               ++iEigenVec)
                            {
                              partialOccupVec[iEigenVec] =
                                dftUtils::getPartialOccupancy(
                                  eigenValues[0][totalNumWaveFunctions *
                                                   spinIndex +
                                                 jvec + iEigenVec],
                                  fermiEnergy,
                                  C_kb,
                                  dftParameters::TVal) *
                                spinPolarizedFactor;
                            }
                        }

                      partialOccupVecDevice = partialOccupVec;
                    }

                  stridedCopyToBlockKernel<<<(BVec + 255) / 256 * numLocalDofs,
                                             256>>>(
                    BVec,
                    X + numLocalDofs * totalNumWaveFunctions * spinIndex,
                    numLocalDofs,
                    totalNumWaveFunctions,
                    cudaFlattenedArrayBlock.begin(),
                    jvec);


                  cudaFlattenedArrayBlock.updateGhostValues();

                  (operatorMatrix.getOverloadedConstraintMatrix())
                    ->distribute(cudaFlattenedArrayBlock, BVec);

                  for (int iblock = 0; iblock < (numCellBlocks + 1); iblock++)
                    {
                      const unsigned int currentCellsBlockSize =
                        (iblock == numCellBlocks) ? remCellBlockSize :
                                                    cellsBlockSize;
                      if (currentCellsBlockSize > 0)
                        {
                          const unsigned int startingCellId =
                            iblock * cellsBlockSize;

                          copyCUDAKernel<<<(BVec + 255) / 256 *
                                             currentCellsBlockSize *
                                             numNodesPerElement,
                                           256>>>(
                            BVec,
                            currentCellsBlockSize * numNodesPerElement,
                            cudaFlattenedArrayBlock.begin(),
                            thrust::raw_pointer_cast(
                              &cellWaveFunctionMatrix[0]),
                            thrust::raw_pointer_cast(
                              &(operatorMatrix
                                  .getFlattenedArrayCellLocalProcIndexIdMap()
                                    [startingCellId * numNodesPerElement])));

                          double scalarCoeffAlpha = 1.0, scalarCoeffBeta = 0.0;
                          int    strideA = BVec * numNodesPerElement;
                          int    strideB = 0;
                          int    strideC = BVec * numQuadPoints;


                          cublasDgemmStridedBatched(
                            operatorMatrix.getCublasHandle(),
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            BVec,
                            numQuadPoints,
                            numNodesPerElement,
                            &scalarCoeffAlpha,
                            thrust::raw_pointer_cast(
                              &cellWaveFunctionMatrix[0]),
                            BVec,
                            strideA,
                            thrust::raw_pointer_cast(
                              &(operatorMatrix.getShapeFunctionValuesInverted(
                                use2pPlusOneGLQuad)[0])),
                            numNodesPerElement,
                            strideB,
                            &scalarCoeffBeta,
                            thrust::raw_pointer_cast(
                              &rhoWfcContributionsDevice[0]),
                            BVec,
                            strideC,
                            currentCellsBlockSize);



                          if (isEvaluateGradRho)
                            {
                              strideB = numNodesPerElement * numQuadPoints;

                              cublasDgemmStridedBatched(
                                operatorMatrix.getCublasHandle(),
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                BVec,
                                numQuadPoints,
                                numNodesPerElement,
                                &scalarCoeffAlpha,
                                thrust::raw_pointer_cast(
                                  &cellWaveFunctionMatrix[0]),
                                BVec,
                                strideA,
                                thrust::raw_pointer_cast(
                                  &(operatorMatrix
                                      .getShapeFunctionGradientValuesXInverted()
                                        [startingCellId * numNodesPerElement *
                                         numQuadPoints])),
                                numNodesPerElement,
                                strideB,
                                &scalarCoeffBeta,
                                thrust::raw_pointer_cast(
                                  &gradRhoWfcContributionsDeviceX[0]),
                                BVec,
                                strideC,
                                currentCellsBlockSize);


                              cublasDgemmStridedBatched(
                                operatorMatrix.getCublasHandle(),
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                BVec,
                                numQuadPoints,
                                numNodesPerElement,
                                &scalarCoeffAlpha,
                                thrust::raw_pointer_cast(
                                  &cellWaveFunctionMatrix[0]),
                                BVec,
                                strideA,
                                thrust::raw_pointer_cast(
                                  &(operatorMatrix
                                      .getShapeFunctionGradientValuesYInverted()
                                        [startingCellId * numNodesPerElement *
                                         numQuadPoints])),
                                numNodesPerElement,
                                strideB,
                                &scalarCoeffBeta,
                                thrust::raw_pointer_cast(
                                  &gradRhoWfcContributionsDeviceY[0]),
                                BVec,
                                strideC,
                                currentCellsBlockSize);

                              cublasDgemmStridedBatched(
                                operatorMatrix.getCublasHandle(),
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                BVec,
                                numQuadPoints,
                                numNodesPerElement,
                                &scalarCoeffAlpha,
                                thrust::raw_pointer_cast(
                                  &cellWaveFunctionMatrix[0]),
                                BVec,
                                strideA,
                                thrust::raw_pointer_cast(
                                  &(operatorMatrix
                                      .getShapeFunctionGradientValuesZInverted()
                                        [startingCellId * numNodesPerElement *
                                         numQuadPoints])),
                                numNodesPerElement,
                                strideB,
                                &scalarCoeffBeta,
                                thrust::raw_pointer_cast(
                                  &gradRhoWfcContributionsDeviceZ[0]),
                                BVec,
                                strideC,
                                currentCellsBlockSize);
                            }



                          computeRhoGradRhoFromInterpolatedValues<<<
                            (BVec + 255) / 256 * numQuadPoints *
                              currentCellsBlockSize,
                            256>>>(currentCellsBlockSize * numQuadPoints * BVec,
                                   thrust::raw_pointer_cast(
                                     &rhoWfcContributionsDevice[0]),
                                   thrust::raw_pointer_cast(
                                     &gradRhoWfcContributionsDeviceX[0]),
                                   thrust::raw_pointer_cast(
                                     &gradRhoWfcContributionsDeviceY[0]),
                                   thrust::raw_pointer_cast(
                                     &gradRhoWfcContributionsDeviceZ[0]),
                                   isEvaluateGradRho);


                          cublasDgemm(
                            operatorMatrix.getCublasHandle(),
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            1,
                            currentCellsBlockSize * numQuadPoints,
                            BVec,
                            &scalarCoeffAlphaRho,
                            thrust::raw_pointer_cast(&partialOccupVecDevice[0]),
                            1,
                            thrust::raw_pointer_cast(
                              &rhoWfcContributionsDevice[0]),
                            BVec,
                            &scalarCoeffBetaRho,
                            thrust::raw_pointer_cast(
                              &rhoDevice[startingCellId * numQuadPoints]),
                            1);


                          if (isEvaluateGradRho)
                            {
                              cublasDgemm(operatorMatrix.getCublasHandle(),
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          1,
                                          currentCellsBlockSize * numQuadPoints,
                                          BVec,
                                          &scalarCoeffAlphaGradRho,
                                          thrust::raw_pointer_cast(
                                            &partialOccupVecDevice[0]),
                                          1,
                                          thrust::raw_pointer_cast(
                                            &gradRhoWfcContributionsDeviceX[0]),
                                          BVec,
                                          &scalarCoeffBetaGradRho,
                                          thrust::raw_pointer_cast(
                                            &gradRhoDeviceX[startingCellId *
                                                            numQuadPoints]),
                                          1);


                              cublasDgemm(operatorMatrix.getCublasHandle(),
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          1,
                                          currentCellsBlockSize * numQuadPoints,
                                          BVec,
                                          &scalarCoeffAlphaGradRho,
                                          thrust::raw_pointer_cast(
                                            &partialOccupVecDevice[0]),
                                          1,
                                          thrust::raw_pointer_cast(
                                            &gradRhoWfcContributionsDeviceY[0]),
                                          BVec,
                                          &scalarCoeffBetaGradRho,
                                          thrust::raw_pointer_cast(
                                            &gradRhoDeviceY[startingCellId *
                                                            numQuadPoints]),
                                          1);

                              cublasDgemm(operatorMatrix.getCublasHandle(),
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          1,
                                          currentCellsBlockSize * numQuadPoints,
                                          BVec,
                                          &scalarCoeffAlphaGradRho,
                                          thrust::raw_pointer_cast(
                                            &partialOccupVecDevice[0]),
                                          1,
                                          thrust::raw_pointer_cast(
                                            &gradRhoWfcContributionsDeviceZ[0]),
                                          BVec,
                                          &scalarCoeffBetaGradRho,
                                          thrust::raw_pointer_cast(
                                            &gradRhoDeviceZ[startingCellId *
                                                            numQuadPoints]),
                                          1);
                            }
                        } // non-trivial cell block check
                    }     // cells block loop
                }         // band parallelizatoin check
            }             // wave function block loop

          if (spectrumSplit)
            for (unsigned int jvec = 0; jvec < Nfr; jvec += BVec)
              if ((jvec + totalNumWaveFunctions - Nfr + BVec) <=
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
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
                          if (eigenValues[0][totalNumWaveFunctions * spinIndex +
                                             (totalNumWaveFunctions - Nfr) +
                                             jvec + iEigenVec] >
                              fermiEnergyConstraintMag)
                            partialOccupVec[iEigenVec] = spinPolarizedFactor;
                          else
                            partialOccupVec[iEigenVec] = 0.0;
                        }
                    }
                  else
                    {
                      for (unsigned int iEigenVec = 0; iEigenVec < BVec;
                           ++iEigenVec)
                        {
                          partialOccupVec[iEigenVec] =
                            (dftUtils::getPartialOccupancy(
                               eigenValues[0]
                                          [totalNumWaveFunctions * spinIndex +
                                           (totalNumWaveFunctions - Nfr) +
                                           jvec + iEigenVec],
                               fermiEnergy,
                               C_kb,
                               dftParameters::TVal) -
                             1.0) *
                            spinPolarizedFactor;
                        }
                    }

                  partialOccupVecDevice = partialOccupVec;

                  stridedCopyToBlockKernel<<<(BVec + 255) / 256 * numLocalDofs,
                                             256>>>(
                    BVec,
                    XFrac + numLocalDofs * Nfr * spinIndex,
                    numLocalDofs,
                    Nfr,
                    cudaFlattenedArrayBlock.begin(),
                    jvec);


                  cudaFlattenedArrayBlock.updateGhostValues();

                  (operatorMatrix.getOverloadedConstraintMatrix())
                    ->distribute(cudaFlattenedArrayBlock, BVec);

                  for (int iblock = 0; iblock < (numCellBlocks + 1); iblock++)
                    {
                      const unsigned int currentCellsBlockSize =
                        (iblock == numCellBlocks) ? remCellBlockSize :
                                                    cellsBlockSize;
                      if (currentCellsBlockSize > 0)
                        {
                          const unsigned int startingCellId =
                            iblock * cellsBlockSize;

                          copyCUDAKernel<<<(BVec + 255) / 256 *
                                             currentCellsBlockSize *
                                             numNodesPerElement,
                                           256>>>(
                            BVec,
                            currentCellsBlockSize * numNodesPerElement,
                            cudaFlattenedArrayBlock.begin(),
                            thrust::raw_pointer_cast(
                              &cellWaveFunctionMatrix[0]),
                            thrust::raw_pointer_cast(
                              &(operatorMatrix
                                  .getFlattenedArrayCellLocalProcIndexIdMap()
                                    [startingCellId * numNodesPerElement])));

                          double scalarCoeffAlpha = 1.0, scalarCoeffBeta = 0.0;
                          int    strideA = BVec * numNodesPerElement;
                          int    strideB = 0;
                          int    strideC = BVec * numQuadPoints;


                          cublasDgemmStridedBatched(
                            operatorMatrix.getCublasHandle(),
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            BVec,
                            numQuadPoints,
                            numNodesPerElement,
                            &scalarCoeffAlpha,
                            thrust::raw_pointer_cast(
                              &cellWaveFunctionMatrix[0]),
                            BVec,
                            strideA,
                            thrust::raw_pointer_cast(
                              &(operatorMatrix.getShapeFunctionValuesInverted(
                                use2pPlusOneGLQuad)[0])),
                            numNodesPerElement,
                            strideB,
                            &scalarCoeffBeta,
                            thrust::raw_pointer_cast(
                              &rhoWfcContributionsDevice[0]),
                            BVec,
                            strideC,
                            currentCellsBlockSize);



                          if (isEvaluateGradRho)
                            {
                              strideB = numNodesPerElement * numQuadPoints;

                              cublasDgemmStridedBatched(
                                operatorMatrix.getCublasHandle(),
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                BVec,
                                numQuadPoints,
                                numNodesPerElement,
                                &scalarCoeffAlpha,
                                thrust::raw_pointer_cast(
                                  &cellWaveFunctionMatrix[0]),
                                BVec,
                                strideA,
                                thrust::raw_pointer_cast(
                                  &(operatorMatrix
                                      .getShapeFunctionGradientValuesXInverted()
                                        [startingCellId * numNodesPerElement *
                                         numQuadPoints])),
                                numNodesPerElement,
                                strideB,
                                &scalarCoeffBeta,
                                thrust::raw_pointer_cast(
                                  &gradRhoWfcContributionsDeviceX[0]),
                                BVec,
                                strideC,
                                currentCellsBlockSize);


                              cublasDgemmStridedBatched(
                                operatorMatrix.getCublasHandle(),
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                BVec,
                                numQuadPoints,
                                numNodesPerElement,
                                &scalarCoeffAlpha,
                                thrust::raw_pointer_cast(
                                  &cellWaveFunctionMatrix[0]),
                                BVec,
                                strideA,
                                thrust::raw_pointer_cast(
                                  &(operatorMatrix
                                      .getShapeFunctionGradientValuesYInverted()
                                        [startingCellId * numNodesPerElement *
                                         numQuadPoints])),
                                numNodesPerElement,
                                strideB,
                                &scalarCoeffBeta,
                                thrust::raw_pointer_cast(
                                  &gradRhoWfcContributionsDeviceY[0]),
                                BVec,
                                strideC,
                                currentCellsBlockSize);

                              cublasDgemmStridedBatched(
                                operatorMatrix.getCublasHandle(),
                                CUBLAS_OP_N,
                                CUBLAS_OP_N,
                                BVec,
                                numQuadPoints,
                                numNodesPerElement,
                                &scalarCoeffAlpha,
                                thrust::raw_pointer_cast(
                                  &cellWaveFunctionMatrix[0]),
                                BVec,
                                strideA,
                                thrust::raw_pointer_cast(
                                  &(operatorMatrix
                                      .getShapeFunctionGradientValuesZInverted()
                                        [startingCellId * numNodesPerElement *
                                         numQuadPoints])),
                                numNodesPerElement,
                                strideB,
                                &scalarCoeffBeta,
                                thrust::raw_pointer_cast(
                                  &gradRhoWfcContributionsDeviceZ[0]),
                                BVec,
                                strideC,
                                currentCellsBlockSize);
                            }



                          computeRhoGradRhoFromInterpolatedValues<<<
                            (BVec + 255) / 256 * numQuadPoints *
                              currentCellsBlockSize,
                            256>>>(currentCellsBlockSize * numQuadPoints * BVec,
                                   thrust::raw_pointer_cast(
                                     &rhoWfcContributionsDevice[0]),
                                   thrust::raw_pointer_cast(
                                     &gradRhoWfcContributionsDeviceX[0]),
                                   thrust::raw_pointer_cast(
                                     &gradRhoWfcContributionsDeviceY[0]),
                                   thrust::raw_pointer_cast(
                                     &gradRhoWfcContributionsDeviceZ[0]),
                                   isEvaluateGradRho);


                          cublasDgemm(
                            operatorMatrix.getCublasHandle(),
                            CUBLAS_OP_N,
                            CUBLAS_OP_N,
                            1,
                            currentCellsBlockSize * numQuadPoints,
                            BVec,
                            &scalarCoeffAlphaRho,
                            thrust::raw_pointer_cast(&partialOccupVecDevice[0]),
                            1,
                            thrust::raw_pointer_cast(
                              &rhoWfcContributionsDevice[0]),
                            BVec,
                            &scalarCoeffBetaRho,
                            thrust::raw_pointer_cast(
                              &rhoDevice[startingCellId * numQuadPoints]),
                            1);


                          if (isEvaluateGradRho)
                            {
                              cublasDgemm(operatorMatrix.getCublasHandle(),
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          1,
                                          currentCellsBlockSize * numQuadPoints,
                                          BVec,
                                          &scalarCoeffAlphaGradRho,
                                          thrust::raw_pointer_cast(
                                            &partialOccupVecDevice[0]),
                                          1,
                                          thrust::raw_pointer_cast(
                                            &gradRhoWfcContributionsDeviceX[0]),
                                          BVec,
                                          &scalarCoeffBetaGradRho,
                                          thrust::raw_pointer_cast(
                                            &gradRhoDeviceX[startingCellId *
                                                            numQuadPoints]),
                                          1);


                              cublasDgemm(operatorMatrix.getCublasHandle(),
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          1,
                                          currentCellsBlockSize * numQuadPoints,
                                          BVec,
                                          &scalarCoeffAlphaGradRho,
                                          thrust::raw_pointer_cast(
                                            &partialOccupVecDevice[0]),
                                          1,
                                          thrust::raw_pointer_cast(
                                            &gradRhoWfcContributionsDeviceY[0]),
                                          BVec,
                                          &scalarCoeffBetaGradRho,
                                          thrust::raw_pointer_cast(
                                            &gradRhoDeviceY[startingCellId *
                                                            numQuadPoints]),
                                          1);

                              cublasDgemm(operatorMatrix.getCublasHandle(),
                                          CUBLAS_OP_N,
                                          CUBLAS_OP_N,
                                          1,
                                          currentCellsBlockSize * numQuadPoints,
                                          BVec,
                                          &scalarCoeffAlphaGradRho,
                                          thrust::raw_pointer_cast(
                                            &partialOccupVecDevice[0]),
                                          1,
                                          thrust::raw_pointer_cast(
                                            &gradRhoWfcContributionsDeviceZ[0]),
                                          BVec,
                                          &scalarCoeffBetaGradRho,
                                          thrust::raw_pointer_cast(
                                            &gradRhoDeviceZ[startingCellId *
                                                            numQuadPoints]),
                                          1);
                            }
                        } // non-tivial cells block
                    }     // cells block loop
                }         // spectrum split block


          // do cuda memcopy to host
          std::vector<double> rhoHost;
          std::vector<double> gradRhoHostX;
          std::vector<double> gradRhoHostY;
          std::vector<double> gradRhoHostZ;

          rhoHost.resize(totalLocallyOwnedCells * numQuadPoints, 0.0);
          cudaMemcpy(&rhoHost[0],
                     thrust::raw_pointer_cast(&rhoDevice[0]),
                     totalLocallyOwnedCells * numQuadPoints * sizeof(double),
                     cudaMemcpyDeviceToHost);

          if (isEvaluateGradRho)
            {
              gradRhoHostX.resize(totalLocallyOwnedCells * numQuadPoints, 0.0);
              cudaMemcpy(&gradRhoHostX[0],
                         thrust::raw_pointer_cast(&gradRhoDeviceX[0]),
                         totalLocallyOwnedCells * numQuadPoints *
                           sizeof(double),
                         cudaMemcpyDeviceToHost);

              gradRhoHostY.resize(totalLocallyOwnedCells * numQuadPoints, 0.0);
              cudaMemcpy(&gradRhoHostY[0],
                         thrust::raw_pointer_cast(&gradRhoDeviceY[0]),
                         totalLocallyOwnedCells * numQuadPoints *
                           sizeof(double),
                         cudaMemcpyDeviceToHost);

              gradRhoHostZ.resize(totalLocallyOwnedCells * numQuadPoints, 0.0);
              cudaMemcpy(&gradRhoHostZ[0],
                         thrust::raw_pointer_cast(&gradRhoDeviceZ[0]),
                         totalLocallyOwnedCells * numQuadPoints *
                           sizeof(double),
                         cudaMemcpyDeviceToHost);
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
                  ((dftParameters::spinPolarized == 1) && isEvaluateGradRho) ?
                    (*gradRhoValuesSpinPolarized)[cellid] :
                    dummy;

                for (unsigned int q = 0; q < numQuadPoints; ++q)
                  {
                    if (dftParameters::spinPolarized == 1)
                      {
                        tempRhoQuadsSP[2 * q + spinIndex] +=
                          rhoHost[iElem * numQuadPoints + q];

                        if (isEvaluateGradRho)
                          {
                            tempGradRhoQuadsSP[6 * q + spinIndex * 3] +=
                              gradRhoHostX[iElem * numQuadPoints + q];
                            tempGradRhoQuadsSP[6 * q + 1 + spinIndex * 3] +=
                              gradRhoHostY[iElem * numQuadPoints + q];
                            tempGradRhoQuadsSP[6 * q + 2 + spinIndex * 3] +=
                              gradRhoHostZ[iElem * numQuadPoints + q];
                          }
                      }

                    tempRhoQuads[q] += rhoHost[iElem * numQuadPoints + q];


                    if (isEvaluateGradRho)
                      {
                        tempGradRhoQuads[3 * q] +=
                          gradRhoHostX[iElem * numQuadPoints + q];
                        tempGradRhoQuads[3 * q + 1] +=
                          gradRhoHostY[iElem * numQuadPoints + q];
                        tempGradRhoQuads[3 * q + 2] +=
                          gradRhoHostZ[iElem * numQuadPoints + q];
                      }
                  }
                iElem++;
              }
        } // spin index



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

      cudaDeviceSynchronize();
      MPI_Barrier(MPI_COMM_WORLD);
      gpu_time = MPI_Wtime() - gpu_time;

      if (this_process == 0 && dftParameters::verbosity >= 2)
        std::cout << "Time for compute rho on GPU: " << gpu_time << std::endl;
#endif
    }

  } // namespace CUDA
} // namespace dftfe
