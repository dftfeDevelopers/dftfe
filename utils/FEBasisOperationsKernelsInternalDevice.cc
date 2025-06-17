// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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

#include <FEBasisOperationsKernelsInternal.h>


namespace dftfe
{
  namespace
  {
    template <typename ValueType>

    DFTFE_CREATE_KERNEL(
      void,
      reshapeFromNonAffineDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numQuads * numCells * numVecs * numDims;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt blockIndex  = index / numVecs;
            dftfe::uInt iVec        = index - blockIndex * numVecs;
            dftfe::uInt blockIndex2 = blockIndex / numQuads;
            dftfe::uInt iQuad       = blockIndex - blockIndex2 * numQuads;
            dftfe::uInt iCell       = blockIndex2 / numDims;
            dftfe::uInt iDim        = blockIndex2 - iCell * numDims;
            dftfe::utils::copyValue(
              copyToVec + index,
              copyFromVec[iVec + iDim * numVecs + iQuad * numDims * numVecs +
                          iCell * numDims * numQuads * numVecs]);
          }
      },
      const dftfe::uInt numVecs,
      const dftfe::uInt numQuads,
      const dftfe::uInt numDims,
      const dftfe::uInt numCells,
      const ValueType  *copyFromVec,
      ValueType        *copyToVec);

    template <typename ValueType>

    DFTFE_CREATE_KERNEL(
      void,
      reshapeToNonAffineDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numQuads * numCells * numVecs * numDims;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt blockIndex  = index / numVecs;
            dftfe::uInt iVec        = index - blockIndex * numVecs;
            dftfe::uInt blockIndex2 = blockIndex / numQuads;
            dftfe::uInt iQuad       = blockIndex - blockIndex2 * numQuads;
            dftfe::uInt iCell       = blockIndex2 / numDims;
            dftfe::uInt iDim        = blockIndex2 - iCell * numDims;
            dftfe::utils::copyValue(copyToVec + iVec + iDim * numVecs +
                                      iQuad * numDims * numVecs +
                                      iCell * numDims * numQuads * numVecs,
                                    copyFromVec[index]);
          }
      },
      const dftfe::uInt numVecs,
      const dftfe::uInt numQuads,
      const dftfe::uInt numDims,
      const dftfe::uInt numCells,
      const ValueType  *copyFromVec,
      ValueType        *copyToVec);

  } // namespace
  namespace basis
  {
    namespace FEBasisOperationsKernelsInternal
    {
      template <typename ValueType>

      void
      reshapeFromNonAffineLayoutDevice(const dftfe::uInt numVecs,
                                       const dftfe::uInt numQuads,
                                       const dftfe::uInt numDims,
                                       const dftfe::uInt numCells,
                                       const ValueType  *copyFromVec,
                                       ValueType        *copyToVec)
      {
        DFTFE_LAUNCH_KERNEL(
          reshapeFromNonAffineDeviceKernel,
          (numVecs * numCells * numQuads * numDims) /
              dftfe::utils::DEVICE_BLOCK_SIZE +
            1,
          dftfe::utils::DEVICE_BLOCK_SIZE,
          dftfe::utils::defaultStream,
          numVecs,
          numQuads,
          numDims,
          numCells,
          dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
          dftfe::utils::makeDataTypeDeviceCompatible(copyToVec));
      }
      template <typename ValueType>
      void
      reshapeToNonAffineLayoutDevice(const dftfe::uInt numVecs,
                                     const dftfe::uInt numQuads,
                                     const dftfe::uInt numDims,
                                     const dftfe::uInt numCells,
                                     const ValueType  *copyFromVec,
                                     ValueType        *copyToVec)
      {
        DFTFE_LAUNCH_KERNEL(
          reshapeToNonAffineDeviceKernel,
          (numVecs * numCells * numQuads * numDims) /
              dftfe::utils::DEVICE_BLOCK_SIZE +
            1,
          dftfe::utils::DEVICE_BLOCK_SIZE,
          dftfe::utils::defaultStream,
          numVecs,
          numQuads,
          numDims,
          numCells,
          dftfe::utils::makeDataTypeDeviceCompatible(copyFromVec),
          dftfe::utils::makeDataTypeDeviceCompatible(copyToVec));
      }
      template void
      reshapeFromNonAffineLayoutDevice(const dftfe::uInt numVecs,
                                       const dftfe::uInt numQuads,
                                       const dftfe::uInt numDims,
                                       const dftfe::uInt numCells,
                                       const double     *copyFromVec,
                                       double           *copyToVec);
      template void
      reshapeFromNonAffineLayoutDevice(const dftfe::uInt           numVecs,
                                       const dftfe::uInt           numQuads,
                                       const dftfe::uInt           numDims,
                                       const dftfe::uInt           numCells,
                                       const std::complex<double> *copyFromVec,
                                       std::complex<double>       *copyToVec);

      template void
      reshapeToNonAffineLayoutDevice(const dftfe::uInt numVecs,
                                     const dftfe::uInt numQuads,
                                     const dftfe::uInt numDims,
                                     const dftfe::uInt numCells,
                                     const double     *copyFromVec,
                                     double           *copyToVec);
      template void
      reshapeToNonAffineLayoutDevice(const dftfe::uInt           numVecs,
                                     const dftfe::uInt           numQuads,
                                     const dftfe::uInt           numDims,
                                     const dftfe::uInt           numCells,
                                     const std::complex<double> *copyFromVec,
                                     std::complex<double>       *copyToVec);

    } // namespace FEBasisOperationsKernelsInternal
  }   // namespace basis
} // namespace dftfe
