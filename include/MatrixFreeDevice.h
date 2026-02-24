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
#include <stdexcept>
#include <TypeConfig.h>
#include <MemoryStorage.h>
#include <DeviceTypeConfig.h>
#include <DeviceExceptions.h>
#include <DeviceKernelLauncherHelpers.h>

namespace dftfe
{
  // List of operators
  enum operatorList
  {
    Laplace   = 0,
    Helmholtz = 1,
    LDA       = 2,
    GGA       = 3,
    Count     = 4
  };

  /**
   * @brief MatrixFreeDevice class template. template parameter nDofsPerDim
   * is the finite element polynomial order. nQuadPointsPerDim is the order of
   * the Gauss quadrature rule. batchSize is the size of batch tuned to hardware
   *
   * @author Gourab Panigrahi
   *
   */
  template <typename T,
            dftfe::operatorList operatorID,
            std::uint32_t       nDofsPerDim,
            std::uint32_t       nQuadPointsPerDim,
            std::uint32_t       batchSize>
  struct MatrixFreeDevice
  {
    static inline void
    init(T *constMemDataHost, std::size_t constMemDataSize);

    static inline void
    computeLaplaceX(T           *dst,
                    T           *src,
                    T           *jacobianFactor,
                    dftfe::uInt *map,
                    T           *shapeBuffer,
                    dftfe::uInt  nCells,
                    dftfe::uInt  nBatch);

    static inline void
    computeHelmholtzX(T           *dst,
                      T           *src,
                      T           *jacobianFactor,
                      dftfe::uInt *map,
                      T           *shapeBuffer,
                      T            coeffHelmholtz,
                      dftfe::uInt  nCells,
                      dftfe::uInt  nBatch);

    static inline void
    constraintsDistribute(T                 *src,
                          const dftfe::uInt *constrainingNodeBuckets,
                          const dftfe::uInt *constrainingNodeOffset,
                          const dftfe::uInt *constrainedNodeBuckets,
                          const dftfe::uInt *constrainedNodeOffset,
                          const T           *weightMatrixList,
                          const dftfe::uInt *weightMatrixOffset,
                          const T           *inhomogenityList,
                          const dftfe::uInt *ghostMap,
                          const dftfe::uInt  inhomogenityListSize,
                          const dftfe::uInt  nBatch,
                          const dftfe::uInt  nOwnedDofs,
                          const dftfe::uInt  nGhostDofs);

    static inline void
    constraintsDistributeTranspose(T                 *dst,
                                   T                 *src,
                                   const dftfe::uInt *constrainingNodeBuckets,
                                   const dftfe::uInt *constrainingNodeOffset,
                                   const dftfe::uInt *constrainedNodeBuckets,
                                   const dftfe::uInt *constrainedNodeOffset,
                                   const T           *weightMatrixList,
                                   const dftfe::uInt *weightMatrixOffset,
                                   const dftfe::uInt *ghostMap,
                                   const dftfe::uInt  inhomogenityListSize,
                                   const dftfe::uInt  nBatch,
                                   const dftfe::uInt  nOwnedDofs,
                                   const dftfe::uInt  nGhostDofs);
  };

} // namespace dftfe
#endif // matrixFreeDevice_H_
