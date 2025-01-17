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
 * @author Bikash Kanungo, Vishal Subramanian
 */

#ifndef dftfeMapPointsToCells_h
#define dftfeMapPointsToCells_h

#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/range/adaptors.hpp>
#include "RTreeBox.h"
#include "RTreePoint.h"

#include <TypeConfig.h>
#include <Cell.h>

namespace dftfe
{
  namespace utils
  {
    /**
     * @brief This class takes in a bunch of points and finds the cell (provided as input)
     * it lies in. In case the points dont lie in any of the cells, it sends the
     * points to other processors. Similarly it receives points from other
     * processors and then checks if any of them lies within its cell. It
     * provides the real coordinates of points in each cell
     *
     * @author Vishal Subramanian, Bikash Kanungo
     */
    template <size_type dim, size_type M>
    class MapPointsToCells
    {
    public:
      MapPointsToCells(const unsigned int verbosity, const MPI_Comm &mpiComm);

      /**
       * @brief The init().
       * @param[in] srcCells The cells assigned to this processor
       * @param[in] targetPts The points assigned to this processor
       * @param[out] mapCellsToRealCoordinates The Real coordinates of the
       * points found in each cell.
       * @param[in] locallyOwnedRange The locally owned range for the target
       * points
       * @param[out] ghostGlobalIds The global Ids of the points assigned to
       * other processors but found within the cells assigned to this processor.
       * @param[in] paramCoordsTol Tol used to determine if the point is inside
       * a cell
       *
       * @author Vishal Subramanian, Bikash Kanungo
       */
      void
      init(std::vector<std::shared_ptr<const Cell<dim>>> srcCells,
           const std::vector<std::vector<double>> &      targetPts,
           std::vector<std::vector<double>> &   mapCellsToRealCoordinates,
           std::vector<std::vector<size_type>> &mapCellLocalToProcLocal,
           std::pair<global_size_type, global_size_type> &locallyOwnedRange,
           std::vector<global_size_type> &                ghostGlobalIds,
           const double                                   paramCoordsTol);


    private:
      const MPI_Comm d_mpiComm;
      int            d_numMPIRank;
      int            d_thisRank;
      unsigned int   d_verbosity;

    }; // end of class MapPointsToCells
  }    // end of namespace utils
} // end of namespace dftfe

#include "../src/TransferBetweenMeshes/MapPointsToCells.t.cc"
#endif // dftfeMapPointsToCells_h
