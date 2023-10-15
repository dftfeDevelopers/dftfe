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

#ifndef dftfeFEBasisOperationsKernelsDevice_h
#define dftfeFEBasisOperationsKernelsDevice_h

#ifdef DFTFE_WITH_DEVICE
#  include <TypeConfig.h>

namespace dftfe
{
  namespace basis
  {
    namespace FEBasisOperationsKernelsDevice
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
      template <typename ValueType1, typename ValueType2>
      void
      reshapeNonAffineCase(const dftfe::size_type numVecs,
                           const dftfe::size_type numQuads,
                           const dftfe::size_type numCells,
                           const ValueType1 *     copyFromVec,
                           ValueType2 *           copyToVec);


    }; // namespace FEBasisOperationsKernelsDevice
  }    // namespace basis
} // namespace dftfe

#endif // DFTFE_WITH_DEVICE
#endif // dftfeFEBasisOperationsKernelsDevice_h
