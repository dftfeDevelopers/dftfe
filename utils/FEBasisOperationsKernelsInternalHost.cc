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
#include <TypeConfig.h>
#include <complex>
#include <vector>
#include <cstring>
#include <algorithm>

namespace dftfe
{
  namespace basis
  {
    namespace FEBasisOperationsKernelsInternal
    {
      template <typename ValueType>
      void
      reshapeFromNonAffineLayoutHost(const dftfe::uInt numVecs,
                                     const dftfe::uInt numQuads,
                                     const dftfe::uInt numCells,
                                     const ValueType  *copyFromVec,
                                     ValueType        *copyToVec)
      {
        for (dftfe::uInt iCell = 0; iCell < numCells; ++iCell)
          for (dftfe::uInt iQuad = 0; iQuad < numQuads; ++iQuad)
            for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
              std::memcpy(copyToVec + numVecs * 3 * numQuads * iCell +
                            numVecs * numQuads * iDim + numVecs * iQuad,
                          copyFromVec + numVecs * 3 * numQuads * iCell +
                            numVecs * 3 * iQuad + numVecs * iDim,
                          numVecs * sizeof(ValueType));
      }
      template <typename ValueType>
      void
      reshapeToNonAffineLayoutHost(const dftfe::uInt numVecs,
                                   const dftfe::uInt numQuads,
                                   const dftfe::uInt numCells,
                                   const ValueType  *copyFromVec,
                                   ValueType        *copyToVec)
      {
        for (dftfe::uInt iCell = 0; iCell < numCells; ++iCell)
          for (dftfe::uInt iQuad = 0; iQuad < numQuads; ++iQuad)
            for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
              std::memcpy(copyToVec + numVecs * 3 * numQuads * iCell +
                            numVecs * 3 * iQuad + numVecs * iDim,
                          copyFromVec + numVecs * 3 * numQuads * iCell +
                            numVecs * numQuads * iDim + numVecs * iQuad,
                          numVecs * sizeof(ValueType));
      }
      template void
      reshapeFromNonAffineLayoutHost(const dftfe::uInt numVecs,
                                     const dftfe::uInt numQuads,
                                     const dftfe::uInt numCells,
                                     const double     *copyFromVec,
                                     double           *copyToVec);
      template void
      reshapeFromNonAffineLayoutHost(const dftfe::uInt           numVecs,
                                     const dftfe::uInt           numQuads,
                                     const dftfe::uInt           numCells,
                                     const std::complex<double> *copyFromVec,
                                     std::complex<double>       *copyToVec);

      template void
      reshapeToNonAffineLayoutHost(const dftfe::uInt numVecs,
                                   const dftfe::uInt numQuads,
                                   const dftfe::uInt numCells,
                                   const double     *copyFromVec,
                                   double           *copyToVec);
      template void
      reshapeToNonAffineLayoutHost(const dftfe::uInt           numVecs,
                                   const dftfe::uInt           numQuads,
                                   const dftfe::uInt           numCells,
                                   const std::complex<double> *copyFromVec,
                                   std::complex<double>       *copyToVec);

    } // namespace FEBasisOperationsKernelsInternal
  }   // namespace basis
} // namespace dftfe
