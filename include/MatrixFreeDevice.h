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
  namespace MatrixFreeInternal
  {
    template <typename T,
              unsigned int nDofsPerDim,
              unsigned int nQuadPointsPerDim,
              unsigned int batchSize>
    void
    init(T *constMemDataHost, std::size_t constMemDataSize);

    template <typename T,
              unsigned int nDofsPerDim,
              unsigned int nQuadPointsPerDim,
              unsigned int batchSize>
    inline void
    computeLaplaceX(T           *dst,
                    T           *src,
                    T           *jacobianFactor,
                    dftfe::uInt *map,
                    dftfe::uInt  nCells,
                    dftfe::uInt  nBatch);

    template <typename T,
              unsigned int nDofsPerDim,
              unsigned int nQuadPointsPerDim,
              unsigned int batchSize>
    inline void
    constraintsDistribute(T *src);

    template <typename T,
              unsigned int nDofsPerDim,
              unsigned int nQuadPointsPerDim,
              unsigned int batchSize>
    inline void
    constraintsDistributeTranspose(T *dst, T *src);

    template <typename T,
              unsigned int nDofsPerDim,
              unsigned int nQuadPointsPerDim,
              unsigned int batchSize>
    inline void
    constraintsSetZero(T *src);

  } // namespace MatrixFreeInternal
} // namespace dftfe
#endif // matrixFreeDevice_H_
