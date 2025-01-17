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
// @author Sambit Das, Vishal Subramanian
//

// source file for electron density related computations
#include <constants.h>
#include <kineticEnergyDensityCalculator.h>
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
    computeKedGradKedFromInterpolatedValues(const unsigned int numVectors,
                                            const unsigned int numCells,
                                            const unsigned int nQuadsPerCell,
                                            const double       kCoordSq,
                                            double *           kCoord,
                                            double *           wfcContributions,
                                            double *gradwfcContributions,
                                            double *kedCellsWfcContributions)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numEntriesPerCell = numVectors * nQuadsPerCell;
      const unsigned int numberEntries     = numEntriesPerCell * numCells;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const double psi = wfcContributions[index];

          unsigned int iCell          = index / numEntriesPerCell;
          unsigned int intraCellIndex = index - iCell * numEntriesPerCell;
          unsigned int iQuad          = intraCellIndex / numVectors;
          unsigned int iVec           = intraCellIndex - iQuad * numVectors;
          const double gradPsiX       = //[iVec * numCells * numVectors + + 0]
            gradwfcContributions[intraCellIndex +
                                 numEntriesPerCell * 3 * iCell];

          kedCellsWfcContributions[index] = 0.5 * gradPsiX * gradPsiX;

          const double gradPsiY =
            gradwfcContributions[intraCellIndex + numEntriesPerCell +
                                 numEntriesPerCell * 3 * iCell];
          kedCellsWfcContributions[index] += 0.5 * gradPsiY * gradPsiY;

          const double gradPsiZ =
            gradwfcContributions[intraCellIndex + 2 * numEntriesPerCell +
                                 numEntriesPerCell * 3 * iCell];
          kedCellsWfcContributions[index] += 0.5 * gradPsiZ * gradPsiZ;
        }
    }

    __global__ void
    computeKedGradKedFromInterpolatedValues(
      const unsigned int                 numVectors,
      const unsigned int                 numCells,
      const unsigned int                 nQuadsPerCell,
      const double                       kCoordSq,
      double *                           kCoord,
      dftfe::utils::deviceDoubleComplex *wfcContributions,
      dftfe::utils::deviceDoubleComplex *gradwfcContributions,
      double *                           kedCellsWfcContributions)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numEntriesPerCell = numVectors * nQuadsPerCell;
      const unsigned int numberEntries     = numEntriesPerCell * numCells;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::utils::deviceDoubleComplex psi = wfcContributions[index];
          kedCellsWfcContributions[index] =
            kCoordSq * (psi.x * psi.x + psi.y * psi.y);

          unsigned int iCell          = index / numEntriesPerCell;
          unsigned int intraCellIndex = index - iCell * numEntriesPerCell;
          unsigned int iQuad          = intraCellIndex / numVectors;
          unsigned int iVec           = intraCellIndex - iQuad * numVectors;
          const dftfe::utils::deviceDoubleComplex gradPsiX =
            gradwfcContributions[intraCellIndex +
                                 numEntriesPerCell * 3 * iCell];
          kedCellsWfcContributions[index] +=
            0.5 * (gradPsiX.x * gradPsiX.x + gradPsiX.y * gradPsiX.y);

          const dftfe::utils::deviceDoubleComplex gradPsiY =
            gradwfcContributions[intraCellIndex + numEntriesPerCell +
                                 numEntriesPerCell * 3 * iCell];
          kedCellsWfcContributions[index] +=
            0.5 * (gradPsiY.x * gradPsiY.x + gradPsiY.y * gradPsiY.y);

          const dftfe::utils::deviceDoubleComplex gradPsiZ =
            gradwfcContributions[intraCellIndex + 2 * numEntriesPerCell +
                                 numEntriesPerCell * 3 * iCell];
          kedCellsWfcContributions[index] +=
            0.5 * (gradPsiZ.x * gradPsiZ.x + gradPsiZ.y * gradPsiZ.y);

          kedCellsWfcContributions[index] +=
            kCoord[0] * (psi.x * gradPsiX.y - psi.y * gradPsiX.x);

          kedCellsWfcContributions[index] +=
            kCoord[1] * (psi.x * gradPsiY.y - psi.y * gradPsiY.x);

          kedCellsWfcContributions[index] +=
            kCoord[2] * (psi.x * gradPsiZ.y - psi.y * gradPsiZ.x);
        }
    }
  } // namespace
  template <typename NumberType>
  void
  computeKineticEnergyDensityFromInterpolatedValues(
    const dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>
      &BLASWrapperPtr,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<NumberType,
                                      double,
                                      dftfe::utils::MemorySpace::DEVICE>>
      &                                         basisOperationsPtr,
    const std::pair<unsigned int, unsigned int> cellRange,
    const std::pair<unsigned int, unsigned int> vecRange,
    double *                                    partialOccupVec,
    double *                                    kcoord,
    NumberType *                                wfcQuadPointData,
    NumberType *                                gradWfcQuadPointData,
    double *        kineticEnergyDensityCellsWfcContributions,
    double *        kineticEnergyDensity,
    const MPI_Comm &mpiCommDomain)
  {
    const unsigned int cellsBlockSize   = cellRange.second - cellRange.first;
    const unsigned int vectorsBlockSize = vecRange.second - vecRange.first;
    const unsigned int nQuadsPerCell    = basisOperationsPtr->nQuadsPerCell();
    const unsigned int nCells           = basisOperationsPtr->nCells();
    const double       scalarCoeffAlphaKed = 1.0;
    const double       scalarCoeffBetaKed  = 1.0;
    const double       kcoordSq =
      kcoord[0] * kcoord[0] + kcoord[1] * kcoord[1] + kcoord[2] * kcoord[2];

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
                        kCoordDevice(3);
    std::vector<double> kCoordStdVec(3);
    kCoordStdVec[0] = kcoord[0];
    kCoordStdVec[1] = kcoord[1];
    kCoordStdVec[2] = kcoord[2];
    kCoordDevice.copyFrom(kCoordStdVec);
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    computeKedGradKedFromInterpolatedValues<<<
      (vectorsBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
        dftfe::utils::DEVICE_BLOCK_SIZE * nQuadsPerCell * cellsBlockSize,
      dftfe::utils::DEVICE_BLOCK_SIZE>>>(
      vectorsBlockSize,
      cellsBlockSize,
      nQuadsPerCell,
      kcoordSq,
      dftfe::utils::makeDataTypeDeviceCompatible(kCoordDevice.data()),
      dftfe::utils::makeDataTypeDeviceCompatible(wfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(gradWfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(
        kineticEnergyDensityCellsWfcContributions));
#elif DFTFE_WITH_DEVICE_LANG_HIP
    hipLaunchKernelGGL(
      computeKedGradKedFromInterpolatedValues,
      (vectorsBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
        dftfe::utils::DEVICE_BLOCK_SIZE * nQuadsPerCell * cellsBlockSize,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      0,
      0,
      vectorsBlockSize,
      cellsBlockSize,
      nQuadsPerCell,
      kcoordSq,
      dftfe::utils::makeDataTypeDeviceCompatible(kCoordDevice.data()),
      dftfe::utils::makeDataTypeDeviceCompatible(wfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(gradWfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(
        kineticEnergyDensityCellsWfcContributions));
#endif
    BLASWrapperPtr.xgemm('T',
                         'N',
                         cellsBlockSize * nQuadsPerCell,
                         1,
                         vectorsBlockSize,
                         &scalarCoeffAlphaKed,
                         kineticEnergyDensityCellsWfcContributions,
                         vectorsBlockSize,
                         partialOccupVec,
                         vectorsBlockSize,
                         &scalarCoeffBetaKed,
                         kineticEnergyDensity + cellRange.first * nQuadsPerCell,
                         cellsBlockSize * nQuadsPerCell);
  }
  template void
  computeKineticEnergyDensityFromInterpolatedValues(
    const dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>
      &BLASWrapperPtr,
    std::shared_ptr<
      dftfe::basis::FEBasisOperations<dataTypes::number,
                                      double,
                                      dftfe::utils::MemorySpace::DEVICE>>
      &                                         basisOperationsPtr,
    const std::pair<unsigned int, unsigned int> cellRange,
    const std::pair<unsigned int, unsigned int> vecRange,
    double *                                    partialOccupVec,
    double *                                    kcoord,
    dataTypes::number *                         wfcQuadPointData,
    dataTypes::number *                         gradWfcQuadPointData,
    double *        kineticEnergyCellsWfcContributions,
    double *        kineticEnergyDensity,
    const MPI_Comm &mpiCommDomain);

} // namespace dftfe
