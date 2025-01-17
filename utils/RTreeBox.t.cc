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

namespace dftfe
{
  namespace utils
  {
    namespace RTreeBoxInternal
    {
      template <size_type dim>
      BG::model::box<BG::model::point<double, dim, BG::cs::cartesian>>
      convertToBBox(
        const std::pair<std::vector<double>, std::vector<double>> &boundingBox)
      {
        using BPoint = BG::model::point<double, dim, BG::cs::cartesian>;
        using BBox   = BG::model::box<BPoint>;
        using BBoxI  = std::pair<BBox, size_type>;

        BPoint pointLowerLeft;
        BPoint pointUpperRight;
        for (size_type k = 0; k < dim; ++k)
          {
            pointLowerLeft.set<k>(boundingBox.first[k]);
            pointUpperRight.set<k>(boundingBox.second[k]);
          }
        return BBox(pointLowerLeft, pointUpperRight);
      }
    } // namespace RTreeBoxInternal

    template <size_type dim, size_type M>
    RTreeBox<dim, M>::RTreeBox(
      std::vector<std::shared_ptr<const Cell<dim>>> sourceCells)
    {
      using BPoint     = BG::model::point<double, dim, BG::cs::cartesian>;
      using BBox       = BG::model::box<BPoint>;
      using BBoxI      = std::pair<BBox, size_type>;
      using BRTreeBoxI = BGI::rtree<BBoxI, BGI::quadratic<M>>;


      const size_type    nCells = sourceCells.size();
      std::vector<BBoxI> sourceCellsBoundingBoxes(nCells);
      for (size_type i = 0; i < nCells; ++i)
        {
          std::pair<std::vector<double>, std::vector<double>> boundingBox =
            sourceCells[i]->getBoundingBox();
          BBox bbox = RTreeBoxInternal::convertToBBox<dim>(boundingBox);
          sourceCellsBoundingBoxes[i] = std::make_pair(bbox, i);
        }

      d_rtree = BRTreeBoxI(sourceCellsBoundingBoxes.begin(),
                           sourceCellsBoundingBoxes.end());
    }

    template <size_type dim, size_type M>
    std::vector<std::vector<size_type>>
    RTreeBox<dim, M>::getOverlappingCellIds(
      std::vector<std::shared_ptr<const Cell<dim>>> queryCells)
    {
      using BPoint     = BG::model::point<double, dim, BG::cs::cartesian>;
      using BBox       = BG::model::box<BPoint>;
      using BBoxI      = std::pair<BBox, size_type>;
      using BRTreeBoxI = BGI::rtree<BBoxI, BGI::quadratic<M>>;


      const size_type                     nQCells = queryCells.size();
      std::vector<std::vector<size_type>> cellIds(nQCells,
                                                  std::vector<size_type>(0));
      for (size_type i = 0; i < nQCells; ++i)
        {
          std::vector<BBoxI> overlappingBBoxI(0);
          std::pair<std::vector<double>, std::vector<double>> boundingBox =
            queryCells[i]->getBoundingBox();
          BBox bbox = RTreeBoxInternal::convertToBBox<dim>(boundingBox);
          d_rtree.query(BGI::intersects(bbox),
                        std::back_inserter(overlappingBBoxI));
          const size_type nOverlappingBBox = overlappingBBoxI.size();
          for (size_type j = 0; j < nOverlappingBBox; ++j)
            cellIds[i].push_back(overlappingBBoxI[j].second);
        }

      return cellIds;
    }
  } // end of namespace utils
} // end of namespace dftfe
