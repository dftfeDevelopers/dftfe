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
// @author Kartick Ramakrishnan, Nikhil Kodali
//
#include <KohnShamDFTOperatorKernels.h>
#include <deviceKernelsGeneric.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherHelpers.h>
#include <BLASWrapper.h>
namespace dftfe
{
  namespace
  {
    DFTFE_CREATE_KERNEL(
      void,
      computeCellHamiltonianMatrixNonCollinearFromBlocksDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numCells * nDofsPerCell * nDofsPerCell;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            const dftfe::uInt jDoF   = index % nDofsPerCell;
            const dftfe::uInt iBlock = index / nDofsPerCell;
            const dftfe::uInt iDoF   = iBlock % nDofsPerCell;
            const dftfe::uInt iCell  = cellStartIndex + iBlock / nDofsPerCell;
            const dftfe::uInt iCellBlock = iBlock / nDofsPerCell;
            const double      H_realIJ =
              tempHamMatrixRealBlock[jDoF + nDofsPerCell * iDoF +
                                     iCellBlock * nDofsPerCell * nDofsPerCell];
            const double H_imagIJ =
              tempHamMatrixImagBlock[jDoF + nDofsPerCell * iDoF +
                                     iCellBlock * nDofsPerCell * nDofsPerCell];
            const double H_bzIJ =
              tempHamMatrixBZBlockNonCollin[jDoF + nDofsPerCell * iDoF +
                                            iCellBlock * nDofsPerCell *
                                              nDofsPerCell];
            const double H_byIJ =
              tempHamMatrixBYBlockNonCollin[jDoF + nDofsPerCell * iDoF +
                                            iCellBlock * nDofsPerCell *
                                              nDofsPerCell];
            const double H_bxIJ =
              tempHamMatrixBXBlockNonCollin[jDoF + nDofsPerCell * iDoF +
                                            iCellBlock * nDofsPerCell *
                                              nDofsPerCell];
            cellHamiltonianMatrix[iCell * nDofsPerCell * nDofsPerCell * 4 +
                                  2 * nDofsPerCell * (2 * iDoF + 1) + 2 * jDoF +
                                  1] =
              dftfe::utils::makeComplex(H_realIJ - H_bzIJ, H_imagIJ);
            cellHamiltonianMatrix[iCell * nDofsPerCell * nDofsPerCell * 4 +
                                  2 * nDofsPerCell * (2 * iDoF) + 2 * jDoF] =
              dftfe::utils::makeComplex(H_realIJ + H_bzIJ, H_imagIJ);
            cellHamiltonianMatrix[iCell * nDofsPerCell * nDofsPerCell * 4 +
                                  2 * nDofsPerCell * (2 * iDoF + 1) +
                                  2 * jDoF] =
              dftfe::utils::makeComplex(H_bxIJ, H_byIJ);
            cellHamiltonianMatrix[iCell * nDofsPerCell * nDofsPerCell * 4 +
                                  2 * nDofsPerCell * (2 * iDoF) + 2 * jDoF +
                                  1] =
              dftfe::utils::makeComplex(H_bxIJ, -H_byIJ);
          }
      },
      const dftfe::uInt                  numCells,
      const dftfe::uInt                  nDofsPerCell,
      const dftfe::uInt                  cellStartIndex,
      const double                      *tempHamMatrixRealBlock,
      const double                      *tempHamMatrixImagBlock,
      const double                      *tempHamMatrixBZBlockNonCollin,
      const double                      *tempHamMatrixBYBlockNonCollin,
      const double                      *tempHamMatrixBXBlockNonCollin,
      dftfe::utils::deviceDoubleComplex *cellHamiltonianMatrix);
  } // namespace

  namespace internal
  {

    template <>
    void
    computeVeffJxWEntries(
      const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
      const dftfe::uInt                         numQuadsPerCell,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &phiVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &pdecVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &pdexVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &jxwVector,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        &VeffJxW)
    {
      // Not yet implemented
    }
    template <>
    void
    computeInvJacderExcWithSigmaTimesGradRhoJxWEntries(
      const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
      const dftfe::uInt                         numQuadsPerCell,
      const dftfe::Int                          spinIndex,
      const dftfe::Int                          cellsTypeFlag,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &pdecVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &pdexVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &jxwVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &invJacobianEntries,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &gradientRhoSpinIndex,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &gradientRhoOtherSpinIndex,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        &invJacderExcWithSigmaTimesGradRhoJxW)
    {
      // Not yet implemented
    }

    template <>
    void
    computeHalfInvJacinvJacderExcWithTauJxWEntries(
      const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
      const dftfe::uInt                         numQuadsPerCell,
      const dftfe::Int                          cellsTypeFlag,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &pdecVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &pdexVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &jxwVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &invJacobianEntries,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        &invJacinvJacderExcWithTauJxW)
    {
      // Not yet implemented
    }

    template <>
    void
    computeKPointDependenderExcWithTauJxWEntries(
      const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
      const dftfe::uInt                         numQuadsPerCell,
      const dftfe::Int                          cellsTypeFlag,
      const dftfe::uInt                         offset,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &kPointCoordinate,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &pdecVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &pdexVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &jxwVector,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &invJacobianEntries,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        &halfKSquareTimesDerExcwithTauJxW,
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
        &invJacKpointTimesderExcwithTauJxW)
    {
      // Not yet implemented
    }
    template <>
    void
    computeCellHamiltonianMatrixNonCollinearFromBlocks(
      const std::pair<dftfe::uInt, dftfe::uInt> cellRange,
      const dftfe::uInt                         nDofsPerCell,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &tempHamMatrixRealBlock,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &tempHamMatrixImagBlock,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &tempHamMatrixBZBlockNonCollin,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &tempHamMatrixBYBlockNonCollin,
      const dftfe::utils::MemoryStorage<double,
                                        dftfe::utils::MemorySpace::DEVICE>
        &tempHamMatrixBXBlockNonCollin,
      dftfe::utils::MemoryStorage<std::complex<double>,
                                  dftfe::utils::MemorySpace::DEVICE>
        &cellHamiltonianMatrix)
    {
      const dftfe::uInt nCells       = cellRange.second - cellRange.first;
      auto tempHamMatrixRealBlockPtr = tempHamMatrixRealBlock.data();
      auto tempHamMatrixImagBlockPtr = tempHamMatrixImagBlock.data();
      auto tempHamMatrixBZBlockNonCollinPtr =
        tempHamMatrixBZBlockNonCollin.data();
      auto tempHamMatrixBYBlockNonCollinPtr =
        tempHamMatrixBYBlockNonCollin.data();
      auto tempHamMatrixBXBlockNonCollinPtr =
        tempHamMatrixBXBlockNonCollin.data();
      auto cellHamiltonianMatrixPtr = cellHamiltonianMatrix.data();
      DFTFE_LAUNCH_KERNEL(
        computeCellHamiltonianMatrixNonCollinearFromBlocksDeviceKernel,
        (nCells * nDofsPerCell * nDofsPerCell) /
            dftfe::utils::DEVICE_BLOCK_SIZE +
          1,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::defaultStream,
        nCells,
        nDofsPerCell,
        cellRange.first,
        dftfe::utils::makeDataTypeDeviceCompatible(tempHamMatrixRealBlockPtr),
        dftfe::utils::makeDataTypeDeviceCompatible(tempHamMatrixImagBlockPtr),
        dftfe::utils::makeDataTypeDeviceCompatible(
          tempHamMatrixBZBlockNonCollinPtr),
        dftfe::utils::makeDataTypeDeviceCompatible(
          tempHamMatrixBYBlockNonCollinPtr),
        dftfe::utils::makeDataTypeDeviceCompatible(
          tempHamMatrixBXBlockNonCollinPtr),
        dftfe::utils::makeDataTypeDeviceCompatible(cellHamiltonianMatrixPtr));
    }
  }; // namespace internal
} // namespace dftfe
