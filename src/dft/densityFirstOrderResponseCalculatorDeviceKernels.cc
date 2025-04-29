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
    computeRhoResponseFromInterpolatedValues(
      const dftfe::uInt numVectors,
      const dftfe::uInt numCells,
      const dftfe::uInt nQuadsPerCell,
      const double     *wfc,
      const double     *wfcPrime,
      double           *rhoResponseHamCellsWfcContributions,
      double           *rhoResponseFermiEnergyCellsWfcContributions)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numEntriesPerCell = numVectors * nQuadsPerCell;
      const dftfe::uInt numberEntries     = numEntriesPerCell * numCells;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const double psi                                   = wfc[index];
          const double psiPrime                              = wfcPrime[index];
          rhoResponseFermiEnergyCellsWfcContributions[index] = psi * psi;
          rhoResponseHamCellsWfcContributions[index]         = psi * psiPrime;
        }
    }

    __global__ void
    computeRhoResponseFromInterpolatedValues(
      const dftfe::uInt                        numVectors,
      const dftfe::uInt                        numCells,
      const dftfe::uInt                        nQuadsPerCell,
      const dftfe::utils::deviceDoubleComplex *wfc,
      const dftfe::utils::deviceDoubleComplex *wfcPrime,
      double *rhoResponseHamCellsWfcContributions,
      double *rhoResponseFermiEnergyCellsWfcContributions)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numEntriesPerCell = numVectors * nQuadsPerCell;
      const dftfe::uInt numberEntries     = numEntriesPerCell * numCells;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::utils::deviceDoubleComplex psi      = wfc[index];
          const dftfe::utils::deviceDoubleComplex psiPrime = wfcPrime[index];
          rhoResponseFermiEnergyCellsWfcContributions[index] =
            psi.x * psi.x + psi.y * psi.y;
          rhoResponseHamCellsWfcContributions[index] =
            psi.x * psiPrime.x + psi.y * psiPrime.y;
        }
    }
  } // namespace
  template <typename NumberType>
  void
  computeRhoResponseFromInterpolatedValues(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                             &BLASWrapperPtr,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
    const std::pair<dftfe::uInt, dftfe::uInt> vecRange,
    const dftfe::uInt                         nQuadsPerCell,
    double                                   *onesVec,
    double                                   *partialOccupPrimeVec,
    NumberType                               *wfcQuadPointData,
    NumberType                               *wfcPrimeQuadPointData,
    double *rhoResponseHamCellsWfcContributions,
    double *rhoResponseFermiEnergyCellsWfcContributions,
    double *rhoResponseHam,
    double *rhoResponseFermiEnergy)
  {
    const dftfe::uInt cellsBlockSize      = cellRange.second - cellRange.first;
    const dftfe::uInt vectorsBlockSize    = vecRange.second - vecRange.first;
    const double      scalarCoeffAlphaRho = 1.0;
    const double      scalarCoeffBetaRho  = 1.0;
    DFTFE_LAUNCH_KERNEL(
      computeRhoResponseFromInterpolatedValues,
      (vectorsBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
        dftfe::utils::DEVICE_BLOCK_SIZE * nQuadsPerCell * cellsBlockSize,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      0,
      0,
      vectorsBlockSize,
      cellsBlockSize,
      nQuadsPerCell,
      dftfe::utils::makeDataTypeDeviceCompatible(wfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(wfcPrimeQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(
        rhoResponseHamCellsWfcContributions),
      dftfe::utils::makeDataTypeDeviceCompatible(
        rhoResponseFermiEnergyCellsWfcContributions));
    BLASWrapperPtr->xgemv('T',
                          vectorsBlockSize,
                          cellsBlockSize * nQuadsPerCell,
                          &scalarCoeffAlphaRho,
                          rhoResponseHamCellsWfcContributions,
                          vectorsBlockSize,
                          onesVec,
                          1,
                          &scalarCoeffBetaRho,
                          rhoResponseHam + cellRange.first * nQuadsPerCell,
                          1);

    BLASWrapperPtr->xgemv('T',
                          vectorsBlockSize,
                          cellsBlockSize * nQuadsPerCell,
                          &scalarCoeffAlphaRho,
                          rhoResponseFermiEnergyCellsWfcContributions,
                          vectorsBlockSize,
                          partialOccupPrimeVec,
                          1,
                          &scalarCoeffBetaRho,
                          rhoResponseFermiEnergy +
                            cellRange.first * nQuadsPerCell,
                          1);
  }
  template void
  computeRhoResponseFromInterpolatedValues(
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                                             &BLASWrapperPtr,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
    const std::pair<dftfe::uInt, dftfe::uInt> vecRange,
    const dftfe::uInt                         nQuadsPerCell,
    double                                   *onesVec,
    double                                   *partialOccupVec,
    dataTypes::number                        *wfcQuadPointData,
    dataTypes::number                        *wfcPrimeQuadPointData,
    double *rhoResponseHamCellsWfcContributions,
    double *rhoResponseFermiEnergyCellsWfcContributions,
    double *rhoResponseHam,
    double *rhoResponseFermiEnergy);

} // namespace dftfe
