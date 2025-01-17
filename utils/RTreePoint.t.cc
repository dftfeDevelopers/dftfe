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

#include "Exceptions.h"
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

      template <size_type dim>
      void
      assignValueToBoostPoint(
        boost::geometry::model::point<double, dim, BG::cs::cartesian> &p,
        const std::vector<double> &                                    vec)
      {
        throwException(
          false,
          "Dim of the Boost point is not extended to the current value \n");
      }

      template <>
      void
      assignValueToBoostPoint<3>(
        boost::geometry::model::point<double, 3, BG::cs::cartesian> &p,
        const std::vector<double> &                                  vec)
      {
        boost::geometry::assign_values(p, vec[0], vec[1], vec[2]);
      }

      template <>
      void
      assignValueToBoostPoint<2>(
        boost::geometry::model::point<double, 2, BG::cs::cartesian> &p,
        const std::vector<double> &                                  vec)
      {
        boost::geometry::assign_values(p, vec[0], vec[1]);
      }

      template <>
      void
      assignValueToBoostPoint<1>(
        boost::geometry::model::point<double, 1, BG::cs::cartesian> &p,
        const std::vector<double> &                                  vec)
      {
        p.set<0>(vec[0]);
      }
    } // namespace

    namespace RTreePointInternal
    {
      template <size_type dim>
      BG::model::box<BG::model::point<double, dim, BG::cs::cartesian>>
      convertToBBox(const std::vector<double> &ll,
                    const std::vector<double> &ur)
      {
        BG::model::point<double, dim, BG::cs::cartesian> pointLowerLeft;
        BG::model::point<double, dim, BG::cs::cartesian> pointUpperRight;

        assignValueToBoostPoint<dim>(pointLowerLeft, ll);
        assignValueToBoostPoint<dim>(pointUpperRight, ur);
        return BG::model::box<BG::model::point<double, dim, BG::cs::cartesian>>(
          pointLowerLeft, pointUpperRight);
      }
    } // namespace RTreePointInternal

    template <size_type dim, size_type M>
    RTreePoint<dim, M>::RTreePoint(
      const std::vector<std::vector<double>> &srcPts)
    {
      const size_type nPts = srcPts.size();
      std::vector<
        std::pair<BG::model::point<double, dim, BG::cs::cartesian>, size_type>>
        srcPtsI(nPts);
      for (size_type i = 0; i < nPts; ++i)
        {
          boost::geometry::model::point<double, dim, BG::cs::cartesian> p;
          assignValueToBoostPoint<dim>(p, srcPts[i]);
          srcPtsI[i] = std::make_pair(p, i);
        }

      d_rtreePtr = std::make_shared<BGI::rtree<
        std::pair<BG::model::point<double, dim, BG::cs::cartesian>, size_type>,
        BGI::quadratic<M>>>(srcPtsI.begin(), srcPtsI.end());
    }

    template <size_type dim, size_type M>
    std::vector<size_type>
    RTreePoint<dim, M>::getPointIdsInsideBox(
      const std::vector<double> &lowerLeft,
      const std::vector<double> &upperRight)
    {
      std::vector<
        std::pair<BG::model::point<double, dim, BG::cs::cartesian>, size_type>>
                             foundPointI(0);
      std::vector<size_type> pointIds(0);
      BG::model::box<BG::model::point<double, dim, BG::cs::cartesian>> bbox =
        RTreePointInternal::convertToBBox<dim>(lowerLeft, upperRight);
      d_rtreePtr->query(BGI::covered_by(bbox), std::back_inserter(foundPointI));
      const size_type nPointsInside = foundPointI.size();
      for (size_type j = 0; j < nPointsInside; ++j)
        pointIds.push_back(foundPointI[j].second);

      return pointIds;
    }
  } // end of namespace utils
} // end of namespace dftfe
