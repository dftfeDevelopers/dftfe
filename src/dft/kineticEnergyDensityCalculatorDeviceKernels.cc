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
#include "densityCalculatorDeviceKernels.h"
#include "MemoryStorage.h"
namespace dftfe
{
  namespace
  {

    DFTFE_CREATE_KERNEL(
      void,
      computeKedGradKedFromInterpolatedValues,
      {
        const dftfe::uInt numEntriesPerCell = numVectors * nQuadsPerCell;
        const dftfe::uInt numberEntries     = numEntriesPerCell * numCells;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            const double psi = wfcContributions[index];

            dftfe::uInt  iCell          = index / numEntriesPerCell;
            dftfe::uInt  intraCellIndex = index - iCell * numEntriesPerCell;
            dftfe::uInt  iQuad          = intraCellIndex / numVectors;
            dftfe::uInt  iVec           = intraCellIndex - iQuad * numVectors;
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
      },
      const dftfe::uInt numVectors,
      const dftfe::uInt numCells,
      const dftfe::uInt nQuadsPerCell,
      const double      kCoordSq,
      double           *kCoord,
      double           *wfcContributions,
      double           *gradwfcContributions,
      double           *kedCellsWfcContributions);



    DFTFE_CREATE_KERNEL(
      void,
      computeKedGradKedFromInterpolatedValues,
      {
        const dftfe::uInt numEntriesPerCell = numVectors * nQuadsPerCell;
        const dftfe::uInt numberEntries     = numEntriesPerCell * numCells;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            const dftfe::utils::deviceDoubleComplex psi =
              wfcContributions[index];
            kedCellsWfcContributions[index] =
              kCoordSq * (dftfe::utils::realPartDevice(psi) *
                            dftfe::utils::realPartDevice(psi) +
                          dftfe::utils::imagPartDevice(psi) *
                            dftfe::utils::imagPartDevice(psi));

            dftfe::uInt iCell          = index / numEntriesPerCell;
            dftfe::uInt intraCellIndex = index - iCell * numEntriesPerCell;
            dftfe::uInt iQuad          = intraCellIndex / numVectors;
            dftfe::uInt iVec           = intraCellIndex - iQuad * numVectors;
            const dftfe::utils::deviceDoubleComplex gradPsiX =
              gradwfcContributions[intraCellIndex +
                                   numEntriesPerCell * 3 * iCell];
            kedCellsWfcContributions[index] +=
              0.5 * (dftfe::utils::realPartDevice(gradPsiX) *
                       dftfe::utils::realPartDevice(gradPsiX) +
                     dftfe::utils::imagPartDevice(gradPsiX) *
                       dftfe::utils::imagPartDevice(gradPsiX));

            const dftfe::utils::deviceDoubleComplex gradPsiY =
              gradwfcContributions[intraCellIndex + numEntriesPerCell +
                                   numEntriesPerCell * 3 * iCell];
            kedCellsWfcContributions[index] +=
              0.5 * (dftfe::utils::realPartDevice(gradPsiY) *
                       dftfe::utils::realPartDevice(gradPsiY) +
                     dftfe::utils::imagPartDevice(gradPsiY) *
                       dftfe::utils::imagPartDevice(gradPsiY));

            const dftfe::utils::deviceDoubleComplex gradPsiZ =
              gradwfcContributions[intraCellIndex + 2 * numEntriesPerCell +
                                   numEntriesPerCell * 3 * iCell];
            kedCellsWfcContributions[index] +=
              0.5 * (dftfe::utils::realPartDevice(gradPsiZ) *
                       dftfe::utils::realPartDevice(gradPsiZ) +
                     dftfe::utils::imagPartDevice(gradPsiZ) *
                       dftfe::utils::imagPartDevice(gradPsiZ));

            kedCellsWfcContributions[index] +=
              kCoord[0] * (dftfe::utils::realPartDevice(psi) *
                             dftfe::utils::imagPartDevice(gradPsiX) -
                           dftfe::utils::imagPartDevice(psi) *
                             dftfe::utils::realPartDevice(gradPsiX));

            kedCellsWfcContributions[index] +=
              kCoord[1] * (dftfe::utils::realPartDevice(psi) *
                             dftfe::utils::imagPartDevice(gradPsiY) -
                           dftfe::utils::imagPartDevice(psi) *
                             dftfe::utils::realPartDevice(gradPsiY));

            kedCellsWfcContributions[index] +=
              kCoord[2] * (dftfe::utils::realPartDevice(psi) *
                             dftfe::utils::imagPartDevice(gradPsiZ) -
                           dftfe::utils::imagPartDevice(psi) *
                             dftfe::utils::realPartDevice(gradPsiZ));
          }
      },
      const dftfe::uInt                  numVectors,
      const dftfe::uInt                  numCells,
      const dftfe::uInt                  nQuadsPerCell,
      const double                       kCoordSq,
      double                            *kCoord,
      dftfe::utils::deviceDoubleComplex *wfcContributions,
      dftfe::utils::deviceDoubleComplex *gradwfcContributions,
      double                            *kedCellsWfcContributions);

  } // namespace
  template <typename NumberType>
  void
  computeKineticEnergyDensityFromInterpolatedValues(
    dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>
                                             &BLASWrapperPtr,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
    const std::pair<dftfe::uInt, dftfe::uInt> vecRange,
    const dftfe::uInt                         nQuadsPerCell,
    double                                   *partialOccupVec,
    double                                   *kcoord,
    NumberType                               *wfcQuadPointData,
    NumberType                               *gradWfcQuadPointData,
    double         *kineticEnergyDensityCellsWfcContributions,
    double         *kineticEnergyDensity,
    const MPI_Comm &mpiCommDomain)
  {
    const dftfe::uInt cellsBlockSize      = cellRange.second - cellRange.first;
    const dftfe::uInt vectorsBlockSize    = vecRange.second - vecRange.first;
    const double      scalarCoeffAlphaKed = 1.0;
    const double      scalarCoeffBetaKed  = 1.0;
    const double      kcoordSq =
      kcoord[0] * kcoord[0] + kcoord[1] * kcoord[1] + kcoord[2] * kcoord[2];

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
                        kCoordDevice(3);
    std::vector<double> kCoordStdVec(3);
    kCoordStdVec[0] = kcoord[0];
    kCoordStdVec[1] = kcoord[1];
    kCoordStdVec[2] = kcoord[2];
    kCoordDevice.copyFrom(kCoordStdVec);
    auto kCoordDevice_data =
      dftfe::utils::makeDataTypeDeviceCompatible(kCoordDevice.data());
    DFTFE_LAUNCH_KERNEL(
      computeKedGradKedFromInterpolatedValues,
      (vectorsBlockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
        dftfe::utils::DEVICE_BLOCK_SIZE * nQuadsPerCell * cellsBlockSize,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      dftfe::utils::defaultStream,
      vectorsBlockSize,
      cellsBlockSize,
      nQuadsPerCell,
      kcoordSq,
      kCoordDevice_data,
      dftfe::utils::makeDataTypeDeviceCompatible(wfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(gradWfcQuadPointData),
      dftfe::utils::makeDataTypeDeviceCompatible(
        kineticEnergyDensityCellsWfcContributions));
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
    dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>
                                             &BLASWrapperPtr,
    const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
    const std::pair<dftfe::uInt, dftfe::uInt> vecRange,
    const dftfe::uInt                         nQuadsPerCell,
    double                                   *partialOccupVec,
    double                                   *kcoord,
    dataTypes::number                        *wfcQuadPointData,
    dataTypes::number                        *gradWfcQuadPointData,
    double         *kineticEnergyCellsWfcContributions,
    double         *kineticEnergyDensity,
    const MPI_Comm &mpiCommDomain);

} // namespace dftfe
