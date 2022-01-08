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
#include "cudaHelpers.h"
#include <cuComplex.h>
#include "linearAlgebraOperationsCUDA.h"

namespace dftfe
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
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries  = M * BVec;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          unsigned int blockIndex      = index / BVec;
          unsigned int intraBlockIndex = index - blockIndex * BVec;
          yVec[index] = xVec[blockIndex * N + startingXVecId + intraBlockIndex];
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
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
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
      copyCUDAKernel<<<(size + 255) / 256, 256>>>(size, copyFromVec, copyToVec);
    }

    void
    copyDoubleToNumber(const double *     copyFromVec,
                       const unsigned int size,
                       cuDoubleComplex *  copyToVec)
    {
      copyCUDAKernel<<<(size + 255) / 256, 256>>>(size, copyFromVec, copyToVec);
    }


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
    computeRhoResponseFromInterpolatedValues(const unsigned int numberEntries,
                                             cuDoubleComplex *  XQuads,
                                             cuDoubleComplex *  XPrimeQuads)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const cuDoubleComplex psi      = XQuads[index];
          const cuDoubleComplex psiPrime = XPrimeQuads[index];
          XPrimeQuads[index] =
            make_cuDoubleComplex(psi.x * psiPrime.x + psi.y * psiPrime.y, 0.0);
          XQuads[index] =
            make_cuDoubleComplex(psi.x * psi.x + psi.y * psi.y, 0.0);
        }
    }
  } // namespace

  template <typename NumberType>
  void
  computeRhoFirstOrderResponseGPU(
    const NumberType *                             X,
    const NumberType *                             XPrime,
    const std::vector<std::vector<double>> &       densityMatDerFermiEnergy,
    const unsigned int                             totalNumWaveFunctions,
    const unsigned int                             numLocalDofs,
    operatorDFTCUDAClass &                         operatorMatrix,
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
      &             rhoResponseValuesFermiEnergySpinPolarized,
    const MPI_Comm &interpoolcomm,
    const MPI_Comm &interBandGroupComm)
  {
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
    dftUtils::createBandParallelizationIndices(interBandGroupComm,
                                               totalNumWaveFunctions,
                                               bandGroupLowHighPlusOneIndices);

    const unsigned int BVec =
      std::min(dftParameters::chebyWfcBlockSize, totalNumWaveFunctions);

    const double spinPolarizedFactor =
      (dftParameters::spinPolarized == 1) ? 1.0 : 2.0;

    const NumberType zero = cudaUtils::makeNumberFromReal<NumberType>(0.0);
    const NumberType one  = cudaUtils::makeNumberFromReal<NumberType>(1.0);
    const NumberType scalarCoeffAlphaRho =
      cudaUtils::makeNumberFromReal<NumberType>(1.0);
    const NumberType scalarCoeffBetaRho =
      cudaUtils::makeNumberFromReal<NumberType>(1.0);

    const unsigned int cellsBlockSize = 50;
    const unsigned int numCellBlocks  = totalLocallyOwnedCells / cellsBlockSize;
    const unsigned int remCellBlockSize =
      totalLocallyOwnedCells - numCellBlocks * cellsBlockSize;

    cudaUtils::Vector<NumberType, dftfe::MemorySpace::GPU>
      rhoResponseContributionHamDevice(totalLocallyOwnedCells * numQuadPoints,
                                       zero);

    cudaUtils::Vector<NumberType, dftfe::MemorySpace::GPU>
      rhoResponseContributionFermiEnergyDevice(totalLocallyOwnedCells *
                                                 numQuadPoints,
                                               zero);

    cudaUtils::Vector<NumberType, dftfe::MemorySpace::Host>
      rhoResponseContributionHamHost(totalLocallyOwnedCells * numQuadPoints,
                                     zero);

    cudaUtils::Vector<NumberType, dftfe::MemorySpace::Host>
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

    cudaUtils::Vector<NumberType, dftfe::MemorySpace::GPU> XQuadsDevice(
      cellsBlockSize * numQuadPoints * BVec, zero);

    cudaUtils::Vector<NumberType, dftfe::MemorySpace::GPU> XPrimeQuadsDevice(
      cellsBlockSize * numQuadPoints * BVec, zero);
    cudaUtils::Vector<NumberType, dftfe::MemorySpace::GPU> onesVecDevice(BVec,
                                                                         one);

    cudaUtils::Vector<NumberType, dftfe::MemorySpace::Host>
      densityMatDerFermiEnergyVec(BVec, zero);
    cudaUtils::Vector<NumberType, dftfe::MemorySpace::GPU>
      densityMatDerFermiEnergyVecDevice(BVec, zero);

    distributedGPUVec<NumberType> &cudaFlattenedArrayXBlock =
      operatorMatrix.getParallelChebyBlockVectorDevice();

    distributedGPUVec<NumberType> cudaFlattenedArrayXPrimeBlock;
    cudaFlattenedArrayXPrimeBlock.reinit(cudaFlattenedArrayXBlock);

    const unsigned int numGhosts =
      cudaFlattenedArrayXBlock.ghostFlattenedSize();

    NumberType *cellWaveFunctionMatrix = reinterpret_cast<NumberType *>(
      thrust::raw_pointer_cast(&operatorMatrix.getCellWaveFunctionMatrix()[0]));

    NumberType *shapeFunctionValuesInvertedDevice;

    CUDACHECK(
      cudaMalloc((void **)&shapeFunctionValuesInvertedDevice,
                 numNodesPerElement * numQuadPoints * sizeof(NumberType)));
    CUDACHECK(
      cudaMemset(shapeFunctionValuesInvertedDevice,
                 0,
                 numNodesPerElement * numQuadPoints * sizeof(NumberType)));

    copyDoubleToNumber(thrust::raw_pointer_cast(
                         &(operatorMatrix.getShapeFunctionValuesInverted(
                           true)[0])),
                       numNodesPerElement * numQuadPoints,
                       shapeFunctionValuesInvertedDevice);

    for (unsigned int spinIndex = 0;
         spinIndex < (1 + dftParameters::spinPolarized);
         ++spinIndex)
      {
        for (unsigned int kPoint = 0; kPoint < kPointWeights.size(); ++kPoint)
          {
            rhoResponseContributionHamDevice.set(zero);
            rhoResponseContributionFermiEnergyDevice.set(zero);

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
                          cudaUtils::makeNumberFromReal<NumberType>(
                            densityMatDerFermiEnergy
                              [(dftParameters::spinPolarized + 1) * kPoint +
                               spinIndex][jvec + iEigenVec]);
                      }

                    cudaUtils::copyHostVecToCUDAVec(
                      densityMatDerFermiEnergyVec.begin(),
                      densityMatDerFermiEnergyVecDevice.begin(),
                      densityMatDerFermiEnergyVecDevice.size());

                    stridedCopyToBlockKernel<<<(BVec + 255) / 256 *
                                                 numLocalDofs,
                                               256>>>(
                      BVec,
                      X + numLocalDofs * totalNumWaveFunctions *
                            ((dftParameters::spinPolarized + 1) * kPoint +
                             spinIndex),
                      numLocalDofs,
                      totalNumWaveFunctions,
                      cudaFlattenedArrayXBlock.begin(),
                      jvec);


                    cudaFlattenedArrayXBlock.updateGhostValues();

                    (operatorMatrix.getOverloadedConstraintMatrix())
                      ->distribute(cudaFlattenedArrayXBlock, BVec);

                    stridedCopyToBlockKernel<<<(BVec + 255) / 256 *
                                                 numLocalDofs,
                                               256>>>(
                      BVec,
                      XPrime + numLocalDofs * totalNumWaveFunctions *
                                 ((dftParameters::spinPolarized + 1) * kPoint +
                                  spinIndex),
                      numLocalDofs,
                      totalNumWaveFunctions,
                      cudaFlattenedArrayXPrimeBlock.begin(),
                      jvec);


                    cudaFlattenedArrayXPrimeBlock.updateGhostValues();

                    (operatorMatrix.getOverloadedConstraintMatrix())
                      ->distribute(cudaFlattenedArrayXPrimeBlock, BVec);


                    for (int iblock = 0; iblock < (numCellBlocks + 1); iblock++)
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
                              cudaFlattenedArrayXBlock.begin(),
                              thrust::raw_pointer_cast(
                                &cellWaveFunctionMatrix[0]),
                              thrust::raw_pointer_cast(
                                &(operatorMatrix
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
                              thrust::raw_pointer_cast(
                                &cellWaveFunctionMatrix[0]),
                              BVec,
                              strideA,
                              shapeFunctionValuesInvertedDevice,
                              numNodesPerElement,
                              strideB,
                              &scalarCoeffBeta,
                              XQuadsDevice.begin(),
                              BVec,
                              strideC,
                              currentCellsBlockSize);

                            copyGlobalToCellCUDAKernel<<<
                              (BVec + 255) / 256 * currentCellsBlockSize *
                                numNodesPerElement,
                              256>>>(
                              BVec,
                              currentCellsBlockSize * numNodesPerElement,
                              cudaFlattenedArrayXPrimeBlock.begin(),
                              thrust::raw_pointer_cast(
                                &cellWaveFunctionMatrix[0]),
                              thrust::raw_pointer_cast(
                                &(operatorMatrix
                                    .getFlattenedArrayCellLocalProcIndexIdMap()
                                      [startingCellId * numNodesPerElement])));


                            cublasXgemmStridedBatched(
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
                              shapeFunctionValuesInvertedDevice,
                              numNodesPerElement,
                              strideB,
                              &scalarCoeffBeta,
                              XPrimeQuadsDevice.begin(),
                              BVec,
                              strideC,
                              currentCellsBlockSize);


                            computeRhoResponseFromInterpolatedValues<<<
                              (BVec + 255) / 256 * numQuadPoints *
                                currentCellsBlockSize,
                              256>>>(BVec * numQuadPoints *
                                       currentCellsBlockSize,
                                     XQuadsDevice.begin(),
                                     XPrimeQuadsDevice.begin());

                            cublasXgemm(
                              operatorMatrix.getCublasHandle(),
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
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

                            cublasXgemm(
                              operatorMatrix.getCublasHandle(),
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
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


            // do cuda memcopy to host
            cudaUtils::copyCUDAVecToHostVec(
              rhoResponseContributionHamDevice.begin(),
              rhoResponseContributionHamHost.begin(),
              totalLocallyOwnedCells * numQuadPoints);

            cudaUtils::copyCUDAVecToHostVec(
              rhoResponseContributionFermiEnergyDevice.begin(),
              rhoResponseContributionFermiEnergyHost.begin(),
              totalLocallyOwnedCells * numQuadPoints);

            for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
              for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
                {
                  rhoResponseValuesHamFlattenedHost[icell * numQuadPoints +
                                                    iquad] +=
                    kPointWeights[kPoint] * spinPolarizedFactor *
                    cudaUtils::makeRealFromNumber(
                      *(rhoResponseContributionHamHost.begin() +
                        icell * numQuadPoints + iquad));

                  rhoResponseValuesFermiEnergyFlattenedHost[icell *
                                                              numQuadPoints +
                                                            iquad] +=
                    kPointWeights[kPoint] * spinPolarizedFactor *
                    cudaUtils::makeRealFromNumber(
                      *(rhoResponseContributionFermiEnergyHost.begin() +
                        icell * numQuadPoints + iquad));
                }


            if (dftParameters::spinPolarized == 1)
              {
                for (int icell = 0; icell < totalLocallyOwnedCells; icell++)
                  for (unsigned int iquad = 0; iquad < numQuadPoints; ++iquad)
                    {
                      rhoResponseValuesSpinPolarizedHamFlattenedHost
                        [icell * numQuadPoints * 2 + iquad * 2 + spinIndex] +=
                        kPointWeights[kPoint] *
                        cudaUtils::makeRealFromNumber(
                          *(rhoResponseContributionHamHost.begin() +
                            icell * numQuadPoints + iquad));

                      rhoResponseValuesSpinPolarizedFermiEnergyFlattenedHost
                        [icell * numQuadPoints * 2 + iquad * 2 + spinIndex] +=
                        kPointWeights[kPoint] *
                        cudaUtils::makeRealFromNumber(
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

        if (dftParameters::spinPolarized == 1)
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

        if (dftParameters::spinPolarized == 1)
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

          if (dftParameters::spinPolarized == 1)
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

    CUDACHECK(cudaFree(shapeFunctionValuesInvertedDevice));
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    gpu_time = MPI_Wtime() - gpu_time;

    if (this_process == 0 && dftParameters::verbosity >= 2)
      std::cout << "Time for compute rhoprime on GPU: " << gpu_time
                << std::endl;
  }

  template void
  computeRhoFirstOrderResponseGPU(
    const dataTypes::numberGPU *                   X,
    const dataTypes::numberGPU *                   XPrime,
    const std::vector<std::vector<double>> &       densityMatDerFermiEnergy,
    const unsigned int                             totalNumWaveFunctions,
    const unsigned int                             numLocalDofs,
    operatorDFTCUDAClass &                         operatorMatrix,
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
      &             rhoResponseValuesFermiEnergySpinPolarized,
    const MPI_Comm &interpoolcomm,
    const MPI_Comm &interBandGroupComm);
} // namespace dftfe
