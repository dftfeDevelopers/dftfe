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

/*
 * @author Vishal Subramanian, Bikash Kanungo
 */

#include "InterpolateFromCellToLocalPoints.h"
namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  InterpolateFromCellToLocalPoints<memorySpace>::
    InterpolateFromCellToLocalPoints(
      const std::shared_ptr<const dftfe::utils::FECell<3>> &srcCell,
      unsigned int                                          numNodes)
  {
    d_srcCell  = srcCell;
    d_numNodes = numNodes;
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  InterpolateFromCellToLocalPoints<memorySpace>::
    setRealCoordinatesOfLocalPoints(unsigned int        numPoints,
                                    std::vector<double> coordinates)
  {
    d_numPoints = numPoints;
    d_shapeValuesHost.resize(numPoints * d_numNodes);
    d_srcCell->getShapeFuncValues(
      numPoints, coordinates, d_shapeValuesHost, 0, d_numNodes);
    d_shapeValuesMemSpace.resize(numPoints * d_numNodes);
    d_shapeValuesMemSpace.copyFrom(d_shapeValuesHost);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  InterpolateFromCellToLocalPoints<memorySpace>::interpolate(
    const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      &                      BLASWrapperPtr,
    unsigned int             numberOfVectors,
    const dataTypes::number *parentNodalMemSpacePtr,
    dataTypes::number *      outputMemSpacePtr)
  {
    const dataTypes::number scalarCoeffAlpha = 1.0;
    const dataTypes::number scalarCoeffBeta  = 0.0;
    const char              transA = 'N', transB = 'N';
    const unsigned int      inc = 1;

    BLASWrapperPtr->xgemm(transA,
                          transB,
                          numberOfVectors,
                          d_numPoints,
                          d_numNodes,
                          &scalarCoeffAlpha,
                          parentNodalMemSpacePtr,
                          numberOfVectors,
                          d_shapeValuesMemSpace.data(),
                          d_numNodes,
                          &scalarCoeffBeta,
                          outputMemSpacePtr,
                          numberOfVectors);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  InterpolateFromCellToLocalPoints<memorySpace>::interpolate(
    const std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::HOST>>
      &                                   BLASWrapperPtr,
    unsigned int                          numberOfVectors,
    const std::vector<dataTypes::number> &parentNodalHost,
    std::vector<dataTypes::number> &      outputHost)
  {
    const dataTypes::number scalarCoeffAlpha = 1.0;
    const dataTypes::number scalarCoeffBeta  = 0.0;
    const char              transA = 'N', transB = 'N';
    const unsigned int      inc = 1;

    BLASWrapperPtr->xgemm(transA,
                          transB,
                          numberOfVectors,
                          d_numPoints,
                          d_numNodes,
                          &scalarCoeffAlpha,
                          &parentNodalHost[0],
                          numberOfVectors,
                          d_shapeValuesHost.data(),
                          d_numNodes,
                          &scalarCoeffBeta,
                          &outputHost[0],
                          numberOfVectors);
  }

  template class InterpolateFromCellToLocalPoints<
    dftfe::utils::MemorySpace::HOST>;


#ifdef DFTFE_WITH_DEVICE

  template class InterpolateFromCellToLocalPoints<
    dftfe::utils::MemorySpace::DEVICE>;
#endif

} // namespace dftfe
