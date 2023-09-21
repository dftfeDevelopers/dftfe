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
#include <DataTypeOverloads.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherConstants.h>

namespace dftfe
{
  namespace
  {
    __global__ void
    computeRhoGradRhoFromInterpolatedValues(
      const unsigned int numberEntries,
      const unsigned int numCells,
      double *           wfcContributions,
      double *           gradwfcContributions,
      double *           rhoCellsWfcContributions,
      double *           gradRhoCellsWfcContributions,
      const bool         isEvaluateGradRho)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numEntriesPerCell = numberEntries / numCells;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const double psi                = wfcContributions[index];
          rhoCellsWfcContributions[index] = psi * psi;

          if (isEvaluateGradRho)
            {
              unsigned int iCell          = index / numCells;
              unsigned int intraCellIndex = index - iCell * numEntriesPerCell;
              const double gradPsiX =
                gradwfcContributions[intraCellIndex +
                                     numEntriesPerCell * 3 * iCell];
              gradRhoCellsWfcContributions[index] = 2.0 * psi * gradPsiX;

              const double gradPsiY =
                gradwfcContributions[intraCellIndex + numEntriesPerCell +
                                     numEntriesPerCell * 3 * iCell];
              gradRhoCellsWfcContributions[index + numberEntries] =
                2.0 * psi * gradPsiY;

              const double gradPsiZ =
                gradwfcContributions[intraCellIndex + 2 * numEntriesPerCell +
                                     numEntriesPerCell * 3 * iCell];
              gradRhoCellsWfcContributions[index + 2 * numberEntries] =
                2.0 * psi * gradPsiZ;
            }
        }
    }

    __global__ void
    computeRhoGradRhoFromInterpolatedValues(
      const unsigned int                 numberEntries,
      const unsigned int                 numCells,
      dftfe::utils::deviceDoubleComplex *wfcContributions,
      dftfe::utils::deviceDoubleComplex *gradwfcContributions,
      double *                           rhoCellsWfcContributions,
      double *                           gradRhoCellsWfcContributions,
      const bool                         isEvaluateGradRho)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numEntriesPerCell = numberEntries / numCells;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::utils::deviceDoubleComplex psi = wfcContributions[index];
          rhoCellsWfcContributions[index] = psi.x * psi.x + psi.y * psi.y;

          if (isEvaluateGradRho)
            {
              unsigned int iCell          = index / numCells;
              unsigned int intraCellIndex = index - iCell * numEntriesPerCell;
              const dftfe::utils::deviceDoubleComplex gradPsiX =
                gradwfcContributions[intraCellIndex +
                                     numEntriesPerCell * 3 * iCell];
              gradRhoCellsWfcContributions[index] =
                2.0 * (psi.x * gradPsiX.x + psi.y * gradPsiX.y);

              const dftfe::utils::deviceDoubleComplex gradPsiY =
                gradwfcContributions[intraCellIndex + numEntriesPerCell +
                                     numEntriesPerCell * 3 * iCell];
              gradRhoCellsWfcContributions[index + numberEntries] =
                2.0 * (psi.x * gradPsiY.x + psi.y * gradPsiY.y);

              const dftfe::utils::deviceDoubleComplex gradPsiZ =
                gradwfcContributions[intraCellIndex + 2 * numEntriesPerCell +
                                     numEntriesPerCell * 3 * iCell];
              gradRhoCellsWfcContributions[index + 2 * numberEntries] =
                2.0 * (psi.x * gradPsiZ.x + psi.y * gradPsiZ.y);
            }
        }
    }
  } // namespace
  template <typename NumberType>
  void
  computeRhoGradRhoFromInterpolatedValues(
    std::unique_ptr<
      dftfe::basis::FEBasisOperations<NumberType,
                                      double,
                                      dftfe::utils::MemorySpace::DEVICE>>
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
    const double       scalarCoeffAlphaRho     = 1.0;
    const double       scalarCoeffBetaRho      = 1.0;
    const double       scalarCoeffAlphaGradRho = 1.0;
    const double       scalarCoeffBetaGradRho  = 1.0;
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    computeRhoGradRhoFromInterpolatedValues<<<
      (vectorsBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
        dftfe::utils::DEVICE_BLOCK_SIZE * nQuadsPerCell * cellsBlockSize,
      dftfe::utils::DEVICE_BLOCK_SIZE>>>(
      cellsBlockSize * nQuadsPerCell * vectorsBlockSize,
      cellsBlockSize,
      dftfe::utils::makeDataTypeDeviceCompatible(wfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(gradWfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(rhoCellsWfcContributions),
      dftfe::utils::makeDataTypeDeviceCompatible(gradRhoCellsWfcContributions),
      isEvaluateGradRho);
#elif DFTFE_WITH_DEVICE_LANG_HIP
    hipLaunchKernelGGL(
      computeRhoGradRhoFromInterpolatedValues,
      (vectorsBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
        dftfe::utils::DEVICE_BLOCK_SIZE * nQuadsPerCell * cellsBlockSize,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      0,
      0,
      cellsBlockSize * nQuadsPerCell * vectorsBlockSize,
      cellsBlockSize,
      dftfe::utils::makeDataTypeDeviceCompatible(wfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(gradWfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(rhoCellsWfcContributions),
      dftfe::utils::makeDataTypeDeviceCompatible(gradRhoCellsWfcContributions),
      isEvaluateGradRho);
#endif
    dftfe::utils::deviceBlasWrapper::gemm(
      basisOperationsPtr->getDeviceBLASHandle(),
      dftfe::utils::DEVICEBLAS_OP_N,
      dftfe::utils::DEVICEBLAS_OP_N,
      1,
      cellsBlockSize * nQuadsPerCell,
      vectorsBlockSize,
      &scalarCoeffAlphaRho,
      partialOccupVec,
      1,
      rhoCellsWfcContributions,
      vectorsBlockSize,
      &scalarCoeffBetaRho,
      rho + cellRange.first * nQuadsPerCell,
      1);


    if (isEvaluateGradRho)
      {
        dftfe::utils::deviceBlasWrapper::gemm(
          basisOperationsPtr->getDeviceBLASHandle(),
          dftfe::utils::DEVICEBLAS_OP_N,
          dftfe::utils::DEVICEBLAS_OP_N,
          1,
          cellsBlockSize * nQuadsPerCell * 3,
          vectorsBlockSize,
          &scalarCoeffAlphaGradRho,
          partialOccupVec,
          1,
          gradRhoCellsWfcContributions,
          vectorsBlockSize,
          &scalarCoeffBetaGradRho,
          gradRho + cellRange.first * nQuadsPerCell,
          1);
      }
  }
  template void
  computeRhoGradRhoFromInterpolatedValues(
    std::unique_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::DEVICE>>
      &                                         basisOperationsPtr,
    const std::pair<unsigned int, unsigned int> cellRange,
    const std::pair<unsigned int, unsigned int> vecRange,
    double *                                    partialOccupVec,
    dataTypes::number *                         wfcQuadPointData,
    dataTypes::number *                         gradWfcQuadPointData,
    double *                                    rhoCellsWfcContributions,
    double *                                    gradRhoCellsWfcContributions,
    double *                                    rho,
    double *                                    gradRho,
    const bool                                  isEvaluateGradRho);

} // namespace dftfe
