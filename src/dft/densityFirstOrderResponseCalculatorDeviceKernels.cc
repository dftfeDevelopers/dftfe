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
#include <densityFirstOrderResponseCalculator.h>
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
    computeRhoResponseFromInterpolatedValues(
      const unsigned int numVectors,
      const unsigned int numCells,
      const unsigned int nQuadsPerCell,
      double *           wfcContributions,
      double *           wfcPrimeContributions,
      double *           rhoResponseHamCellsWfcContributions,
      double *           rhoResponseFermiEnergyCellsWfcContributions)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numEntriesPerCell = numVectors * nQuadsPerCell;
      const unsigned int numberEntries     = numEntriesPerCell * numCells;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const double psi                = wfcContributions[index];
          const double psiPrime           = wfcContributions[index];
          rhoResponseFermiEnergyCellsWfcContributions[index] = psi * psi;
          rhoResponseHamCellsWfcContributions[index] = psi * psiPrime;

        }
    }

    __global__ void
    computeRhoResponseFromInterpolatedValues(
      const unsigned int                 numVectors,
      const unsigned int                 numCells,
      const unsigned int                 nQuadsPerCell,
      dftfe::utils::deviceDoubleComplex *wfcContributions,
      dftfe::utils::deviceDoubleComplex *wfcPrimeContributions,
      double *                           rhoResponseHamCellsWfcContributions,
      double *                           rhoResponseFermiEnergyCellsWfcContributions)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numEntriesPerCell = numVectors * nQuadsPerCell;
      const unsigned int numberEntries     = numEntriesPerCell * numCells;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::utils::deviceDoubleComplex psi = wfcContributions[index];
          const dftfe::utils::deviceDoubleComplex psiPrime = wfcPrimeContributions[index];          
          rhoResponseFermiEnergyCellsWfcContributions[index] = psi.x * psi.x + psi.y * psi.y;
          rhoResponseHamCellsWfcContributions[index] = psi.x * psiPrime.x + psi.y * psiPrime.y;          

        }
    }
  } // namespace
  template <typename NumberType>
  void
  computeRhoResponseFromInterpolatedValues(
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<NumberType,
                                      double,
                                      dftfe::utils::MemorySpace::DEVICE>>
      &basisOperationsPtr,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      &                                         BLASWrapperPtr,
    const std::pair<unsigned int, unsigned int> cellRange,
    const std::pair<unsigned int, unsigned int> vecRange,
    double *                                    onesVec,
    double *                                    partialOccupPrimeVec,
    NumberType *                                wfcQuadPointData,
    NumberType *                                wfcPrimeQuadPointData,
    double *                                    rhoResponseHamCellsWfcContributions,
    double *                                    rhoResponseFermiEnergyCellsWfcContributions,
    double *                                    rhoResponseHam,
    double *                                    rhoResponseFermiEnergy)
  {
    const unsigned int cellsBlockSize   = cellRange.second - cellRange.first;
    const unsigned int vectorsBlockSize = vecRange.second - vecRange.first;
    const unsigned int nQuadsPerCell    = basisOperationsPtr->nQuadsPerCell();
    const unsigned int nCells           = basisOperationsPtr->nCells();
    const double       scalarCoeffAlphaRho     = 1.0;
    const double       scalarCoeffBetaRho      = 1.0;
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    computeRhoResponseFromInterpolatedValues<<<
      (vectorsBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
        dftfe::utils::DEVICE_BLOCK_SIZE * nQuadsPerCell * cellsBlockSize,
      dftfe::utils::DEVICE_BLOCK_SIZE>>>(
      vectorsBlockSize,
      cellsBlockSize,
      nQuadsPerCell,
      dftfe::utils::makeDataTypeDeviceCompatible(wfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(wfcPrimeQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(rhoResponseHamCellsWfcContributions),
      dftfe::utils::makeDataTypeDeviceCompatible(rhoResponseFermiEnergyCellsWfcContributions));
#elif DFTFE_WITH_DEVICE_LANG_HIP
    hipLaunchKernelGGL(
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
      dftfe::utils::makeDataTypeDeviceCompatible(rhoResponseHamCellsWfcContributions),
      dftfe::utils::makeDataTypeDeviceCompatible(rhoResponseFermiEnergyCellsWfcContributions));
#endif
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
                          rhoResponseFermiEnergy + cellRange.first * nQuadsPerCell,
                          1);

  }
  template void
  computeRhoResponseFromInterpolatedValues(
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::DEVICE>>
      &basisOperationsPtr,
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      &                                         BLASWrapperPtr,
    const std::pair<unsigned int, unsigned int> cellRange,
    const std::pair<unsigned int, unsigned int> vecRange,
    double *                                    partialOccupVec,
    dataTypes::number *                         wfcQuadPointData,
    dataTypes::number *                         wfcPrimeQuadPointData,
    double *                                    rhoResponseHamCellsWfcContributions,
    double *                                    rhoResponseFermiEnergyCellsWfcContributions,
    double *                                    rhoResponseHam,
    double *                                    rhoResponseFermiEnergy);

} // namespace dftfe
