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
      const unsigned int numVectors,
      const unsigned int numCells,
      const unsigned int nQuadsPerCell,
      double *           wfcContributions,
      double *           gradwfcContributions,
      double *           rhoCellsWfcContributions,
      double *           gradRhoCellsWfcContributions,
      const bool         isEvaluateGradRho)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numEntriesPerCell = numVectors * nQuadsPerCell;
      const unsigned int numberEntries     = numEntriesPerCell * numCells;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const double psi                = wfcContributions[index];
          rhoCellsWfcContributions[index] = psi * psi;

          if (isEvaluateGradRho)
            {
              unsigned int iCell          = index / numEntriesPerCell;
              unsigned int intraCellIndex = index - iCell * numEntriesPerCell;
              unsigned int iQuad          = intraCellIndex / numVectors;
              unsigned int iVec           = intraCellIndex - iQuad * numVectors;
              const double gradPsiX = //[iVec * numCells * numVectors + + 0]
                gradwfcContributions[intraCellIndex +
                                     numEntriesPerCell * 3 * iCell];
              gradRhoCellsWfcContributions[iVec + 3 * iQuad * numVectors +
                                           numEntriesPerCell * 3 * iCell] =
                2.0 * psi * gradPsiX;

              const double gradPsiY =
                gradwfcContributions[intraCellIndex + numEntriesPerCell +
                                     numEntriesPerCell * 3 * iCell];
              gradRhoCellsWfcContributions[iVec + numVectors +
                                           3 * iQuad * numVectors +
                                           numEntriesPerCell * 3 * iCell] =
                2.0 * psi * gradPsiY;

              const double gradPsiZ =
                gradwfcContributions[intraCellIndex + 2 * numEntriesPerCell +
                                     numEntriesPerCell * 3 * iCell];
              gradRhoCellsWfcContributions[iVec + 2 * numVectors +
                                           3 * iQuad * numVectors +
                                           numEntriesPerCell * 3 * iCell] =
                2.0 * psi * gradPsiZ;
            }
        }
    }

    __global__ void
    computeRhoGradRhoFromInterpolatedValues(
      const unsigned int                 numVectors,
      const unsigned int                 numCells,
      const unsigned int                 nQuadsPerCell,
      dftfe::utils::deviceDoubleComplex *wfcContributions,
      dftfe::utils::deviceDoubleComplex *gradwfcContributions,
      double *                           rhoCellsWfcContributions,
      double *                           gradRhoCellsWfcContributions,
      const bool                         isEvaluateGradRho)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numEntriesPerCell = numVectors * nQuadsPerCell;
      const unsigned int numberEntries     = numEntriesPerCell * numCells;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::utils::deviceDoubleComplex psi = wfcContributions[index];
          rhoCellsWfcContributions[index] = psi.x * psi.x + psi.y * psi.y;

          if (isEvaluateGradRho)
            {
              unsigned int iCell          = index / numEntriesPerCell;
              unsigned int intraCellIndex = index - iCell * numEntriesPerCell;
              unsigned int iQuad          = intraCellIndex / numVectors;
              unsigned int iVec           = intraCellIndex - iQuad * numVectors;
              const dftfe::utils::deviceDoubleComplex gradPsiX =
                gradwfcContributions[intraCellIndex +
                                     numEntriesPerCell * 3 * iCell];
              gradRhoCellsWfcContributions[iVec + 3 * iQuad * numVectors +
                                           numEntriesPerCell * 3 * iCell] =
                2.0 * (psi.x * gradPsiX.x + psi.y * gradPsiX.y);

              const dftfe::utils::deviceDoubleComplex gradPsiY =
                gradwfcContributions[intraCellIndex + numEntriesPerCell +
                                     numEntriesPerCell * 3 * iCell];
              gradRhoCellsWfcContributions[iVec + numVectors +
                                           3 * iQuad * numVectors +
                                           numEntriesPerCell * 3 * iCell] =
                2.0 * (psi.x * gradPsiY.x + psi.y * gradPsiY.y);

              const dftfe::utils::deviceDoubleComplex gradPsiZ =
                gradwfcContributions[intraCellIndex + 2 * numEntriesPerCell +
                                     numEntriesPerCell * 3 * iCell];
              gradRhoCellsWfcContributions[iVec + 2 * numVectors +
                                           3 * iQuad * numVectors +
                                           numEntriesPerCell * 3 * iCell] =
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
    const unsigned int nQuadsPerCell    = basisOperationsPtr->nQuadsPerCell();
    const unsigned int nCells           = basisOperationsPtr->nCells();
    const double       scalarCoeffAlphaRho     = 1.0;
    const double       scalarCoeffBetaRho      = 1.0;
    const double       scalarCoeffAlphaGradRho = 1.0;
    const double       scalarCoeffBetaGradRho  = 1.0;
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    computeRhoGradRhoFromInterpolatedValues<<<
      (vectorsBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
        dftfe::utils::DEVICE_BLOCK_SIZE * nQuadsPerCell * cellsBlockSize,
      dftfe::utils::DEVICE_BLOCK_SIZE>>>(
      vectorsBlockSize,
      cellsBlockSize,
      nQuadsPerCell,
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
      nQuadsPerCell,
      dftfe::utils::makeDataTypeDeviceCompatible(wfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(gradWfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(rhoCellsWfcContributions),
      dftfe::utils::makeDataTypeDeviceCompatible(gradRhoCellsWfcContributions),
      isEvaluateGradRho);
#endif
    dftfe::utils::deviceBlasWrapper::gemv(
      basisOperationsPtr->getDeviceBLASHandle(),
      dftfe::utils::DEVICEBLAS_OP_T,
      vectorsBlockSize,
      cellsBlockSize * nQuadsPerCell,
      &scalarCoeffAlphaRho,
      rhoCellsWfcContributions,
      vectorsBlockSize,
      partialOccupVec,
      1,
      &scalarCoeffBetaRho,
      rho + cellRange.first * nQuadsPerCell,
      1);


    if (isEvaluateGradRho)
      {
        dftfe::utils::deviceBlasWrapper::gemv(
          basisOperationsPtr->getDeviceBLASHandle(),
          dftfe::utils::DEVICEBLAS_OP_T,
          vectorsBlockSize,
          cellsBlockSize * nQuadsPerCell * 3,
          &scalarCoeffAlphaGradRho,
          gradRhoCellsWfcContributions,
          vectorsBlockSize,
          partialOccupVec,
          1,
          &scalarCoeffBetaGradRho,
          gradRho + cellRange.first * nQuadsPerCell * 3,
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
