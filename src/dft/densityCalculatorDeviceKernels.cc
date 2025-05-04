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

// source file for electron density related computations
#include "densityCalculatorDeviceKernels.h"

namespace dftfe
{
  namespace
  {
    __global__ void
    computeRhoGradRhoFromInterpolatedValues(
      const dftfe::uInt numVectors,
      const dftfe::uInt numCells,
      const dftfe::uInt nQuadsPerCell,
      double           *wfcContributions,
      double           *gradwfcContributions,
      double           *rhoCellsWfcContributions,
      double           *gradRhoCellsWfcContributions,
      const bool        isEvaluateGradRho)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numEntriesPerCell = numVectors * nQuadsPerCell;
      const dftfe::uInt numberEntries     = numEntriesPerCell * numCells;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const double psi                = wfcContributions[index];
          rhoCellsWfcContributions[index] = psi * psi;

          if (isEvaluateGradRho)
            {
              dftfe::uInt  iCell          = index / numEntriesPerCell;
              dftfe::uInt  intraCellIndex = index - iCell * numEntriesPerCell;
              dftfe::uInt  iQuad          = intraCellIndex / numVectors;
              dftfe::uInt  iVec           = intraCellIndex - iQuad * numVectors;
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
      const dftfe::uInt                  numVectors,
      const dftfe::uInt                  numCells,
      const dftfe::uInt                  nQuadsPerCell,
      dftfe::utils::deviceDoubleComplex *wfcContributions,
      dftfe::utils::deviceDoubleComplex *gradwfcContributions,
      double                            *rhoCellsWfcContributions,
      double                            *gradRhoCellsWfcContributions,
      const bool                         isEvaluateGradRho)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numEntriesPerCell = numVectors * nQuadsPerCell;
      const dftfe::uInt numberEntries     = numEntriesPerCell * numCells;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::utils::deviceDoubleComplex psi = wfcContributions[index];
          rhoCellsWfcContributions[index] = psi.x * psi.x + psi.y * psi.y;

          if (isEvaluateGradRho)
            {
              dftfe::uInt iCell          = index / numEntriesPerCell;
              dftfe::uInt intraCellIndex = index - iCell * numEntriesPerCell;
              dftfe::uInt iQuad          = intraCellIndex / numVectors;
              dftfe::uInt iVec           = intraCellIndex - iQuad * numVectors;
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

    __global__ void
    computeTauFromInterpolatedValues(const dftfe::uInt numVectors,
                                     const dftfe::uInt numCells,
                                     const dftfe::uInt nQuadsPerCell,
                                     double           *wfcContributions,
                                     double           *gradwfcContributions,
                                     double           *kCoord,
                                     double           *tauCellsWfcContributions)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numEntriesPerCell = numVectors * nQuadsPerCell;
      const dftfe::uInt numberEntries     = numEntriesPerCell * numCells;
      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const double psi = wfcContributions[index];

          dftfe::uInt iCell          = index / numEntriesPerCell;
          dftfe::uInt intraCellIndex = index - iCell * numEntriesPerCell;
          dftfe::uInt iQuad          = intraCellIndex / numVectors;
          dftfe::uInt iVec           = intraCellIndex - iQuad * numVectors;

          double gradPsiDirVal;
          tauCellsWfcContributions[index] = 0.0;
          for (dftfe::uInt dirIdx = 0; dirIdx < 3; dirIdx++)
            {
              gradPsiDirVal =
                gradwfcContributions[intraCellIndex +
                                     dirIdx * numEntriesPerCell +
                                     numEntriesPerCell * 3 * iCell];

              tauCellsWfcContributions[index] += gradPsiDirVal * gradPsiDirVal;
            }
          tauCellsWfcContributions[index] =
            0.5 * tauCellsWfcContributions[index];
        }
    }


    __global__ void
    computeTauFromInterpolatedValues(
      const dftfe::uInt                  numVectors,
      const dftfe::uInt                  numCells,
      const dftfe::uInt                  nQuadsPerCell,
      dftfe::utils::deviceDoubleComplex *wfcContributions,
      dftfe::utils::deviceDoubleComplex *gradwfcContributions,
      double                            *kCoord,
      double                            *tauCellsWfcContributions)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numEntriesPerCell = numVectors * nQuadsPerCell;
      const dftfe::uInt numberEntries     = numEntriesPerCell * numCells;
      const double      kPointCoordSq =
        kCoord[0] * kCoord[0] + kCoord[1] * kCoord[1] + kCoord[2] * kCoord[2];

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::utils::deviceDoubleComplex psi = wfcContributions[index];

          dftfe::uInt iCell          = index / numEntriesPerCell;
          dftfe::uInt intraCellIndex = index - iCell * numEntriesPerCell;
          dftfe::uInt iQuad          = intraCellIndex / numVectors;
          dftfe::uInt iVec           = intraCellIndex - iQuad * numVectors;

          dftfe::utils::deviceDoubleComplex tempImag;

          tempImag.x = 0.0;
          tempImag.y = 0.0;

          dftfe::utils::deviceDoubleComplex gradPsiDirVal;
          tauCellsWfcContributions[index] = 0.0;
          for (dftfe::uInt dirIdx = 0; dirIdx < 3; dirIdx++)
            {
              gradPsiDirVal =
                gradwfcContributions[intraCellIndex +
                                     dirIdx * numEntriesPerCell +
                                     numEntriesPerCell * 3 * iCell];

              tauCellsWfcContributions[index] +=
                gradPsiDirVal.x * gradPsiDirVal.x +
                gradPsiDirVal.y * gradPsiDirVal.y;

              tempImag.x += kCoord[dirIdx] * gradPsiDirVal.x;
              tempImag.y += kCoord[dirIdx] * gradPsiDirVal.y;
            }

          tauCellsWfcContributions[index] =
            0.5 * tauCellsWfcContributions[index];
          tauCellsWfcContributions[index] +=
            0.5 * kPointCoordSq * (psi.x * psi.x + psi.y * psi.y);
          tauCellsWfcContributions[index] +=
            psi.x * tempImag.y - psi.y * tempImag.x;
        }
    }

  } // namespace
  template <typename NumberType>
  void
  computeRhoGradRhoFromInterpolatedValues(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                             &BLASWrapperPtr,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
    const std::pair<dftfe::uInt, dftfe::uInt> vecRange,
    const dftfe::uInt                         nQuadsPerCell,
    double                                   *partialOccupVec,
    NumberType                               *wfcQuadPointData,
    NumberType                               *gradWfcQuadPointData,
    double                                   *rhoCellsWfcContributions,
    double                                   *gradRhoCellsWfcContributions,
    double                                   *rho,
    double                                   *gradRho,
    const bool                                isEvaluateGradRho)
  {
    const dftfe::uInt cellsBlockSize      = cellRange.second - cellRange.first;
    const dftfe::uInt vectorsBlockSize    = vecRange.second - vecRange.first;
    const double      scalarCoeffAlphaRho = 1.0;
    const double      scalarCoeffBetaRho  = 1.0;
    const double      scalarCoeffAlphaGradRho = 1.0;
    const double      scalarCoeffBetaGradRho  = 1.0;
    DFTFE_LAUNCH_KERNEL(
      computeRhoGradRhoFromInterpolatedValues,
      (vectorsBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
        dftfe::utils::DEVICE_BLOCK_SIZE * nQuadsPerCell * cellsBlockSize,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      0,
      0,
      vectorsBlockSize,
      cellsBlockSize,
      nQuadsPerCell,
      dftfe::utils::makeDataTypeDeviceCompatible(wfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(gradWfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(rhoCellsWfcContributions),
      dftfe::utils::makeDataTypeDeviceCompatible(gradRhoCellsWfcContributions),
      isEvaluateGradRho);
    BLASWrapperPtr->xgemv('T',
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
        BLASWrapperPtr->xgemv('T',
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

  template <typename NumberType>
  void
  computeTauFromInterpolatedValues(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                             &BLASWrapperPtr,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
    const std::pair<dftfe::uInt, dftfe::uInt> vecRange,
    const dftfe::uInt                         nQuadsPerCell,
    double                                   *partialOccupVec,
    double                                   *kCoord,
    NumberType                               *wfcQuadPointData,
    NumberType                               *gradWfcQuadPointData,
    double *kineticEnergyDensityCellsWfcContributions,
    double *tau)
  {
    const dftfe::uInt cellsBlockSize   = cellRange.second - cellRange.first;
    const dftfe::uInt vectorsBlockSize = vecRange.second - vecRange.first;
    const double      scalarCoeffAlpha = 1.0;
    const double      scalarCoeffBeta  = 1.0;

    DFTFE_LAUNCH_KERNEL(
      computeTauFromInterpolatedValues,
      (vectorsBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
        dftfe::utils::DEVICE_BLOCK_SIZE * nQuadsPerCell * cellsBlockSize,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      0,
      0,
      vectorsBlockSize,
      cellsBlockSize,
      nQuadsPerCell,
      dftfe::utils::makeDataTypeDeviceCompatible(wfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(gradWfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(kCoord),
      dftfe::utils::makeDataTypeDeviceCompatible(
        kineticEnergyDensityCellsWfcContributions));

    BLASWrapperPtr->xgemv('T',
                          vectorsBlockSize,
                          cellsBlockSize * nQuadsPerCell,
                          &scalarCoeffAlpha,
                          kineticEnergyDensityCellsWfcContributions,
                          vectorsBlockSize,
                          partialOccupVec,
                          1,
                          &scalarCoeffBeta,
                          tau + cellRange.first * nQuadsPerCell,
                          1);
  }

  template void
  computeRhoGradRhoFromInterpolatedValues(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                             &BLASWrapperPtr,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
    const std::pair<dftfe::uInt, dftfe::uInt> vecRange,
    const dftfe::uInt                         nQuadsPerCell,
    double                                   *partialOccupVec,
    dataTypes::number                        *wfcQuadPointData,
    dataTypes::number                        *gradWfcQuadPointData,
    double                                   *rhoCellsWfcContributions,
    double                                   *gradRhoCellsWfcContributions,
    double                                   *rho,
    double                                   *gradRho,
    const bool                                isEvaluateGradRho);

  template void
  computeTauFromInterpolatedValues(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                             &BLASWrapperPtr,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
    const std::pair<dftfe::uInt, dftfe::uInt> vecRange,
    const dftfe::uInt                         nQuadsPerCell,
    double                                   *partialOccupVec,
    double                                   *kCoord,
    dataTypes::number                        *wfcQuadPointData,
    dataTypes::number                        *gradWfcQuadPointData,
    double *kineticEnergyDensityCellsWfcContributions,
    double *tau);
} // namespace dftfe
