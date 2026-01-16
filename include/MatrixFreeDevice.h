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

/**
 * @author Gourab Panigrahi
 *
 */

#ifndef matrixFreeDevice_H_
#define matrixFreeDevice_H_
#include <cstdint>
#include <DeviceTypeConfig.h>

namespace dftfe
{
  template <typename T,
            std::uint32_t nDofsPerDim,
            std::uint32_t nQuadPointsPerDim,
            std::uint32_t batchSize>
  struct MatrixFreeDevice
  {
    static inline void
    init(T *constMemDataHost, std::size_t constMemDataSize);

    static inline void
    computeLaplaceX(T             *dst,
                    T             *src,
                    T             *jacobianFactor,
                    std::uint32_t *map,
                    std::uint32_t  nCells,
                    std::uint32_t  nBatch);

    static inline void
    constraintsDistribute(T                   *src,
                          const std::uint32_t *constrainingNodeBuckets,
                          const std::uint32_t *constrainingNodeOffset,
                          const std::uint32_t *constrainedNodeBuckets,
                          const std::uint32_t *constrainedNodeOffset,
                          const T             *weightMatrixList,
                          const std::uint32_t *weightMatrixOffset,
                          const T             *inhomogenityList,
                          const std::uint32_t *ghostMap,
                          const std::uint32_t  inhomogenityListSize,
                          const std::uint32_t  nBatch,
                          const std::uint32_t  nOwnedDofs,
                          const std::uint32_t  nGhostDofs);

    static inline void
    constraintsDistributeTranspose(T                   *dst,
                                   T                   *src,
                                   const std::uint32_t *constrainingNodeBuckets,
                                   const std::uint32_t *constrainingNodeOffset,
                                   const std::uint32_t *constrainedNodeBuckets,
                                   const std::uint32_t *constrainedNodeOffset,
                                   const T             *weightMatrixList,
                                   const std::uint32_t *weightMatrixOffset,
                                   const std::uint32_t *ghostMap,
                                   const std::uint32_t  inhomogenityListSize,
                                   const std::uint32_t  nBatch,
                                   const std::uint32_t  nOwnedDofs,
                                   const std::uint32_t  nGhostDofs);
  };

} // namespace dftfe
#endif // matrixFreeDevice_H_
