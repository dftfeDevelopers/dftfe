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

#ifndef dftfeRTreePoint_h
#define dftfeRTreePoint_h

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/geometries.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/range/adaptors.hpp>


#include <TypeConfig.h>
#include <Cell.h>

#include "headers.h"

namespace dftfe
{
  namespace utils
  {
    namespace
    {
      namespace BA   = boost::adaptors;
      namespace BG   = boost::geometry;
      namespace BGI  = boost::geometry::index;
      namespace BGIA = boost::geometry::index::adaptors;
    } // namespace

    /** @brief A class template to perform RTreePoint based searching on
     * overlap of boxes
     *
     * @tparam dim Dimension of the box
     * @param M maximum allowable nodes in a branch of RTreePoint (i.e., maximum number of child nodes a parent node can have)
     */
    template <size_type dim, size_type M>
    class RTreePoint
    {
    public:
      using BPoint       = BG::model::point<double, dim, BG::cs::cartesian>;
      using BBox         = BG::model::box<BPoint>;
      using BPointI      = std::pair<BPoint, size_type>;
      using BRTreePointI = BGI::rtree<BPointI, BGI::quadratic<M>>;
      /**
       * @brief Constructor
       *
       *
       */
      RTreePoint(const std::vector<std::vector<double>> &srcPts);

      std::vector<size_type>
      getPointIdsInsideBox(const std::vector<double> &lowerLeft,
                           const std::vector<double> &upperRight);

    private:
      //
      // boost rtree obj
      //
      std::shared_ptr<BRTreePointI> d_rtreePtr;

    }; // end of class RTreePoint
  }    // end of namespace utils
} // end of namespace dftfe
#include <../utils/RTreePoint.t.cc>
#endif // dftfeRTreePoint_h
