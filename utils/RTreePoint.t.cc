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

      template <dftfe::uInt dim>
      void
      assignValueToBoostPoint(
        boost::geometry::model::point<double, dim, BG::cs::cartesian> &p,
        const std::vector<double>                                     &vec)
      {
        throwException(
          false,
          "Dim of the Boost point is not extended to the current value \n");
      }

      template <>
      void
      assignValueToBoostPoint<3>(
        boost::geometry::model::point<double, 3, BG::cs::cartesian> &p,
        const std::vector<double>                                   &vec)
      {
        boost::geometry::assign_values(p, vec[0], vec[1], vec[2]);
      }

      template <>
      void
      assignValueToBoostPoint<2>(
        boost::geometry::model::point<double, 2, BG::cs::cartesian> &p,
        const std::vector<double>                                   &vec)
      {
        boost::geometry::assign_values(p, vec[0], vec[1]);
      }

      template <>
      void
      assignValueToBoostPoint<1>(
        boost::geometry::model::point<double, 1, BG::cs::cartesian> &p,
        const std::vector<double>                                   &vec)
      {
        p.set<0>(vec[0]);
      }
    } // namespace

    namespace RTreePointInternal
    {
      template <dftfe::uInt dim>
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

    template <dftfe::uInt dim, dftfe::uInt M>
    RTreePoint<dim, M>::RTreePoint(
      const std::vector<std::vector<double>> &srcPts)
    {
      const dftfe::uInt nPts = srcPts.size();
      std::vector<std::pair<BG::model::point<double, dim, BG::cs::cartesian>,
                            dftfe::uInt>>
        srcPtsI(nPts);
      for (dftfe::uInt i = 0; i < nPts; ++i)
        {
          boost::geometry::model::point<double, dim, BG::cs::cartesian> p;
          assignValueToBoostPoint<dim>(p, srcPts[i]);
          srcPtsI[i] = std::make_pair(p, i);
        }

      d_rtreePtr = std::make_shared<
        BGI::rtree<std::pair<BG::model::point<double, dim, BG::cs::cartesian>,
                             dftfe::uInt>,
                   BGI::quadratic<M>>>(srcPtsI.begin(), srcPtsI.end());
    }

    template <dftfe::uInt dim, dftfe::uInt M>
    std::vector<dftfe::uInt>
    RTreePoint<dim, M>::getPointIdsInsideBox(
      const std::vector<double> &lowerLeft,
      const std::vector<double> &upperRight)
    {
      std::vector<std::pair<BG::model::point<double, dim, BG::cs::cartesian>,
                            dftfe::uInt>>
                               foundPointI(0);
      std::vector<dftfe::uInt> pointIds(0);
      BG::model::box<BG::model::point<double, dim, BG::cs::cartesian>> bbox =
        RTreePointInternal::convertToBBox<dim>(lowerLeft, upperRight);
      d_rtreePtr->query(BGI::covered_by(bbox), std::back_inserter(foundPointI));
      const dftfe::uInt nPointsInside = foundPointI.size();
      for (dftfe::uInt j = 0; j < nPointsInside; ++j)
        pointIds.push_back(foundPointI[j].second);

      return pointIds;
    }

    template <dftfe::uInt dim, dftfe::uInt M>
    std::vector<dftfe::uInt>
    RTreePoint<dim, M>::getPointIdsNearInputPoint(
      const std::vector<double> &inputPoint,
      dftfe::uInt                nNearestNeighbours)
    {
      BG::model::point<double, dim, BG::cs::cartesian> bgInputPoint;
      assignValueToBoostPoint<dim>(bgInputPoint, inputPoint);

      std::vector<std::pair<BG::model::point<double, dim, BG::cs::cartesian>,
                            dftfe::uInt>>
        result;

      result.resize(0);
      d_rtreePtr->query(BGI::nearest(bgInputPoint, nNearestNeighbours),
                        std::back_inserter(result));

      std::vector<dftfe::uInt> outputVec;
      outputVec.resize(result.size());
      for (dftfe::uInt j = 0; j < result.size(); j++)
        {
          outputVec[j] = result[j].second;
        }
      return outputVec;
    }
  } // end of namespace utils
} // end of namespace dftfe
