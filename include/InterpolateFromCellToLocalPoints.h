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

#ifndef DFTFE_EXE_INTERPOLATEFROMCELLTOLOCALPOINTS_H
#define DFTFE_EXE_INTERPOLATEFROMCELLTOLOCALPOINTS_H


#include "BLASWrapper.h"
#include "Cell.h"
#include "dftUtils.h"
#include "FECell.h"


namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  class InterpolateFromCellToLocalPoints
  {
  public:
    InterpolateFromCellToLocalPoints(
      const std::shared_ptr<const dftfe::utils::FECell<3>> &srcCell,
      unsigned int                                          numNodes);

    void
    setRealCoordinatesOfLocalPoints(unsigned int        numPoints,
                                    std::vector<double> coordinates);

    void
    interpolate(
      const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
        &                      BLASWrapperPtr,
      unsigned int             numberOfVectors,
      const dataTypes::number *parentNodalMemSpacePtr,
      dataTypes::number *      outputMemSpacePtr);

    void
    interpolate(const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<
                  dftfe::utils::MemorySpace::HOST>> & BLASWrapperPtr,
                unsigned int                          numberOfVectors,
                const std::vector<dataTypes::number> &parentNodalHost,
                std::vector<dataTypes::number> &      outputHost);

  private:
    std::shared_ptr<const dftfe::utils::FECell<3>> d_srcCell;
    unsigned int                                   d_numNodes, d_numPoints;
    dftfe::utils::MemoryStorage<dataTypes::number, memorySpace>
                                   d_shapeValuesMemSpace;
    std::vector<dataTypes::number> d_shapeValuesHost;

    const std::shared_ptr<dftfe::linearAlgebra::BLASWrapper<memorySpace>>
      d_BLASWrapperPtr;
  };

} // namespace dftfe
#endif // DFTFE_EXE_INTERPOLATEFROMCELLTOLOCALPOINTS_H
