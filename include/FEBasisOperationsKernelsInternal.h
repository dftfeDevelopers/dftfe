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

#ifndef dftfeFEBasisOperationsKernelsInternal_h
#define dftfeFEBasisOperationsKernelsInternal_h

#ifdef DFTFE_WITH_DEVICE
#  include <TypeConfig.h>
#  include <DeviceAPICalls.h>
#  include <DeviceTypeConfig.h>
#  include <DeviceKernelLauncherConstants.h>
#  include <DeviceDataTypeOverloads.h>
#  include <BLASWrapper.h>
namespace dftfe
{
  namespace basis
  {
    namespace FEBasisOperationsKernelsInternal
    {
      /**
       * @brief rehsape gradient data from [iCell * 3 * d_nQuadsPerCell * d_nVectors + iQuad * 3 * d_nVectors + iDim * d_nVectors + iVec] to [iCell * 3 * d_nQuadsPerCell * d_nVectors + iDim *
       * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[in] numVecs number of vectors.
       * @param[in] numQuads number of quadrature points per cell.
       * @param[in] numCells number of locally owned cells.
       * @param[in] copyFromVec source data pointer.
       * @param[out] copyToVec destination data pointer.
       */
      template <typename ValueType>
      void
      reshapeFromNonAffineLayoutDevice(const dftfe::size_type numVecs,
                                       const dftfe::size_type numQuads,
                                       const dftfe::size_type numCells,
                                       const ValueType *      copyFromVec,
                                       ValueType *            copyToVec);

      template <typename ValueType>
      void
      reshapeFromNonAffineLayoutHost(const dftfe::size_type numVecs,
                                     const dftfe::size_type numQuads,
                                     const dftfe::size_type numCells,
                                     const ValueType *      copyFromVec,
                                     ValueType *            copyToVec);

      /**
       * @brief rehsape gradient data to [iCell * 3 * d_nQuadsPerCell * d_nVectors + iQuad * 3 * d_nVectors + iDim * d_nVectors + iVec] from [iCell * 3 * d_nQuadsPerCell * d_nVectors + iDim *
       * d_nQuadsPerCell * d_nVectors + iQuad * d_nVectors + iVec].
       * @param[in] numVecs number of vectors.
       * @param[in] numQuads number of quadrature points per cell.
       * @param[in] numCells number of locally owned cells.
       * @param[in] copyFromVec source data pointer.
       * @param[out] copyToVec destination data pointer.
       */
      template <typename ValueType>
      void
      reshapeToNonAffineLayoutDevice(const dftfe::size_type numVecs,
                                     const dftfe::size_type numQuads,
                                     const dftfe::size_type numCells,
                                     const ValueType *      copyFromVec,
                                     ValueType *            copyToVec);

      template <typename ValueType>
      void
      reshapeToNonAffineLayoutHost(const dftfe::size_type numVecs,
                                   const dftfe::size_type numQuads,
                                   const dftfe::size_type numCells,
                                   const ValueType *      copyFromVec,
                                   ValueType *            copyToVec);

    } // namespace FEBasisOperationsKernelsInternal
  }   // namespace basis
} // namespace dftfe

#endif // DFTFE_WITH_DEVICE
#endif // dftfeFEBasisOperationsKernelsInternal_h
