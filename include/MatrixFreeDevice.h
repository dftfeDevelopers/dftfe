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
#include <TypeConfig.h>

namespace dftfe
{
  template <typename T,
            dftfe::uInt nDofsPerDim,
            dftfe::uInt nQuadPointsPerDim,
            dftfe::uInt batchSize>
  struct MatrixFreeDevice
  {
    static inline void
    init(T *constMemDataHost, std::size_t constMemDataSize);

    static inline void
    computeLaplaceX(T           *dst,
                    T           *src,
                    T           *jacobianFactor,
                    dftfe::uInt *map,
                    dftfe::uInt  nCells,
                    dftfe::uInt  nBatch);

    static inline void
    constraintsDistribute(T *src);

    static inline void
    constraintsDistributeTranspose(T *dst, T *src);

    static inline void
    constraintsSetZero(T *src);
  };

} // namespace dftfe
#endif // matrixFreeDevice_H_
