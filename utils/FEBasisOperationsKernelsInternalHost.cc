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
                                     const dftfe::uInt numDims,
                                     const dftfe::uInt numCells,
                                     const ValueType  *copyFromVec,
                                     ValueType        *copyToVec)
      {
        for (dftfe::uInt iCell = 0; iCell < numCells; ++iCell)
          for (dftfe::uInt iQuad = 0; iQuad < numQuads; ++iQuad)
            for (dftfe::uInt iDim = 0; iDim < numDims; ++iDim)
              std::memcpy(copyToVec + numVecs * numDims * numQuads * iCell +
                            numVecs * numQuads * iDim + numVecs * iQuad,
                          copyFromVec + numVecs * numDims * numQuads * iCell +
                            numVecs * numDims * iQuad + numVecs * iDim,
                          numVecs * sizeof(ValueType));
      }
      template <typename ValueType>
      void
      reshapeToNonAffineLayoutHost(const dftfe::uInt numVecs,
                                   const dftfe::uInt numQuads,
                                   const dftfe::uInt numDims,
                                   const dftfe::uInt numCells,
                                   const ValueType  *copyFromVec,
                                   ValueType        *copyToVec)
      {
        for (dftfe::uInt iCell = 0; iCell < numCells; ++iCell)
          for (dftfe::uInt iQuad = 0; iQuad < numQuads; ++iQuad)
            for (dftfe::uInt iDim = 0; iDim < numDims; ++iDim)
              std::memcpy(copyToVec + numVecs * numDims * numQuads * iCell +
                            numVecs * numDims * iQuad + numVecs * iDim,
                          copyFromVec + numVecs * numDims * numQuads * iCell +
                            numVecs * numQuads * iDim + numVecs * iQuad,
                          numVecs * sizeof(ValueType));
      }
      template <typename ValueType>
      void
      scaleQuadratureDataWithDiagonalJacobianHost(
        const dftfe::uInt  numberOfElements,
        const dftfe::uInt  nDoFsPerCell,
        const dftfe::uInt  nQuadsPerCell,
        const ValueType   *inverseJacobiansEntries,
        const ValueType   *gradientDataBlockCoeff,
        ValueType         *gradientData,
        const dftfe::uInt *cellIndices)
      {
        for (dftfe::uInt iCell = 0; iCell < numberOfElements; iCell++)
          {
            dftfe::uInt cellIndex  = cellIndices[iCell];
            dftfe::uInt cellOffset = cellIndex * 3;
            for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; iQuad++)
              {
                for (dftfe::uInt iDof = 0; iDof < nDoFsPerCell; iDof++)
                  {
                    for (dftfe::Int iDim = 0; iDim < 3; iDim++)
                      {
                        ValueType alpha =
                          inverseJacobiansEntries[cellOffset + iDim];
                        gradientData[iDof + iDim * nDoFsPerCell +
                                     iQuad * 3 * nDoFsPerCell +
                                     iCell * nQuadsPerCell * nDoFsPerCell * 3] =
                          gradientDataBlockCoeff[iDof + iDim * nDoFsPerCell +
                                                 iQuad * 3 * nDoFsPerCell] *
                          alpha;
                      }
                  }
              }
          }
      }
      template void
      reshapeFromNonAffineLayoutHost(const dftfe::uInt numVecs,
                                     const dftfe::uInt numQuads,
                                     const dftfe::uInt numDims,
                                     const dftfe::uInt numCells,
                                     const double     *copyFromVec,
                                     double           *copyToVec);
      template void
      reshapeFromNonAffineLayoutHost(const dftfe::uInt           numVecs,
                                     const dftfe::uInt           numQuads,
                                     const dftfe::uInt           numDims,
                                     const dftfe::uInt           numCells,
                                     const std::complex<double> *copyFromVec,
                                     std::complex<double>       *copyToVec);

      template void
      reshapeToNonAffineLayoutHost(const dftfe::uInt numVecs,
                                   const dftfe::uInt numQuads,
                                   const dftfe::uInt numDims,
                                   const dftfe::uInt numCells,
                                   const double     *copyFromVec,
                                   double           *copyToVec);
      template void
      reshapeToNonAffineLayoutHost(const dftfe::uInt           numVecs,
                                   const dftfe::uInt           numQuads,
                                   const dftfe::uInt           numDims,
                                   const dftfe::uInt           numCells,
                                   const std::complex<double> *copyFromVec,
                                   std::complex<double>       *copyToVec);


      template void
      scaleQuadratureDataWithDiagonalJacobianHost(
        const dftfe::uInt  numberOfElements,
        const dftfe::uInt  nDoFsPerCell,
        const dftfe::uInt  nQuadsPerCell,
        const double      *inverseJacobiansEntries,
        const double      *gradientDataBlockCoeff,
        double            *gradientData,
        const dftfe::uInt *cellIndices);

      template void
      scaleQuadratureDataWithDiagonalJacobianHost(
        const dftfe::uInt           numberOfElements,
        const dftfe::uInt           nDoFsPerCell,
        const dftfe::uInt           nQuadsPerCell,
        const std::complex<double> *inverseJacobiansEntries,
        const std::complex<double> *gradientDataBlockCoeff,
        std::complex<double>       *gradientData,
        const dftfe::uInt          *cellIndices);

    } // namespace FEBasisOperationsKernelsInternal
  }   // namespace basis
} // namespace dftfe
