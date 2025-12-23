// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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


#ifndef matrixFreeDevice_H_
#define matrixFreeDevice_H_

#include <MemoryStorage.h>

namespace dftfe
{
  template <typename T,
            unsigned int nDofsPerDim,
            unsigned int nQuadPointsPerDim,
            unsigned int batchSize>
  class MatrixFreeDevice
  {
  public:
    MatrixFreeDevice(const unsigned int nVectors,
                     const unsigned int nCells,
                     const unsigned int nOwnedDofs,
                     const unsigned int nGhostDofs);

    void
    init(T           *constMemDataHost,
         unsigned int constMemDataSize,
         dftfe::utils::MemoryStorage<T, dftfe::utils::MemorySpace::HOST>
           &jacobianFactor,
         dftfe::utils::MemoryStorage<unsigned int,
                                     dftfe::utils::MemorySpace::HOST> &map,
         std::vector<std::vector<unsigned int>> &constrainingNodeBuckets,
         std::vector<std::vector<unsigned int>> &constrainedNodeBuckets,
         std::vector<std::vector<T>>            &weightMatrixList);

    inline void
    computeLaplaceX(T *dst, T *src);

    inline void
    constraintsDistribute(T *src);

    inline void
    constraintsDistributeTranspose(T *dst, T *src);

    inline void
    constraintsSetZero(T *src);

#ifdef DFTFE_WITH_DEVICE
    dftfe::utils::MemoryStorage<T, dftfe::utils::MemorySpace::DEVICE>
      d_jacobianFactor, d_cellInverseMassVector, d_cellInverseSqrtMassVector,
      d_weightMatrixList, d_inhomogenityList;

    dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::DEVICE>
      d_map, d_ghostMap, d_constrainingNodeBuckets, d_constrainedNodeBuckets,
      d_constrainingNodeOffset, d_constrainedNodeOffset, d_weightMatrixOffset;
#endif

  private:
    const unsigned int d_nVectors, d_nBatch, d_nCells, d_nOwnedDofs,
      d_nGhostDofs;
  };

} // namespace dftfe
#endif // matrixFreeDevice_H_
