

namespace dftfe
{
  namespace utils
  {
    template <dftfe::uInt dim>
    FECell<dim>::FECell(
      typename dealii::DoFHandler<dim>::active_cell_iterator dealiiFECellIter,
      const dealii::FiniteElement<dim, dim>                 &fe)
      : d_mappingQ1()
      , d_feCell(fe)
    {
      this->reinit(dealiiFECellIter);
    }

    template <dftfe::uInt dim>
    void
    FECell<dim>::reinit(
      typename dealii::DoFHandler<dim>::active_cell_iterator dealiiFECellIter)
    {
      d_dealiiFECellIter = dealiiFECellIter;

      auto bb = this->getBoundingBox();
      d_lowerLeft.resize(dim, 0.0);
      d_upperRight.resize(dim, 0.0);
      for (dftfe::uInt j = 0; j < dim; j++)
        {
          d_lowerLeft[j]  = bb.first[j];
          d_upperRight[j] = bb.second[j];
        }
    }

    template <dftfe::uInt dim>
    void
    FECell<dim>::getVertices(std::vector<std::vector<double>> &points) const
    {
      const dftfe::uInt nVertices =
        dealii::GeometryInfo<dim>::vertices_per_cell;
      points.resize(nVertices, std::vector<double>(dim));
      std::vector<dealii::Point<dim, double>> pointsDealii;
      pointsDealii.resize(nVertices);
      for (dftfe::uInt iVertex = 0; iVertex < nVertices; iVertex++)
        {
          pointsDealii[iVertex] = d_dealiiFECellIter->vertex(iVertex);
          for (dftfe::uInt j = 0; j < dim; j++)
            {
              points[iVertex][j] = pointsDealii[iVertex][j];
            }
        }
    }

    template <dftfe::uInt dim>
    void
    FECell<dim>::getVertex(dftfe::uInt i, std::vector<double> &point) const
    {
      point.resize(dim);
      dealii::Point<dim, double> pointDealii = d_dealiiFECellIter->vertex(i);
      for (dftfe::uInt j = 0; j < dim; j++)
        {
          point[j] = pointDealii[j];
        }
    }

    template <dftfe::uInt dim>
    std::pair<std::vector<double>, std::vector<double>>
    FECell<dim>::getBoundingBox() const
    {
      std::vector<double>      ll(dim, 0.0), ur(dim, 0.0);
      dealii::BoundingBox<dim> bb = d_dealiiFECellIter->bounding_box();
      auto                     dealiiPointsPair = bb.get_boundary_points();
      for (dftfe::uInt j = 0; j < dim; j++)
        {
          ll[j] = (dealiiPointsPair.first)[j];
          ur[j] = (dealiiPointsPair.second)[j];
        }
      auto returnVal = make_pair(ll, ur);
      return returnVal;
    }

    template <dftfe::uInt dim>
    bool
    FECell<dim>::isPointInside(const std::vector<double> &point,
                               const double               tol) const
    {
      bool                returnVal  = true;
      std::vector<double> paramPoint = this->getParametricPoint(point);
      for (dftfe::uInt j = 0; j < dim; j++)
        {
          if ((paramPoint[j] < -tol) || (paramPoint[j] > 1.0 + tol))
            {
              returnVal = false;
            }
        }
      return returnVal;
    }

    template <dftfe::uInt dim>
    std::vector<double>
    FECell<dim>::getParametricPoint(const std::vector<double> &realPoint) const
    {
      dealii::Point<dim, double> pointRealDealii;
      for (dftfe::uInt j = 0; j < dim; j++)
        {
          pointRealDealii[j] = realPoint[j];
        }
      dealii::Point<dim, double> pointParamDealii =
        d_mappingQ1.transform_real_to_unit_cell(d_dealiiFECellIter,
                                                pointRealDealii);

      std::vector<double> pointParam(dim, 0.0);
      for (dftfe::uInt j = 0; j < dim; j++)
        {
          pointParam[j] = pointParamDealii[j];
        }

      return pointParam;
    }

    template <dftfe::uInt dim>
    std::vector<double>
    FECell<dim>::getParametricPointForAllPoints(
      dftfe::uInt                numPoints,
      const std::vector<double> &realPoint) const
    {
      dealii::Point<dim, double> pointRealDealii;
      std::vector<double>        pointParam(dim * numPoints, 0.0);
      for (dftfe::uInt iPoint = 0; iPoint < numPoints; iPoint++)
        {
          for (dftfe::uInt j = 0; j < dim; j++)
            {
              pointRealDealii[j] = realPoint[iPoint * dim + j];
            }
          dealii::Point<dim, double> pointParamDealii =
            d_mappingQ1.transform_real_to_unit_cell(d_dealiiFECellIter,
                                                    pointRealDealii);
          for (dftfe::uInt j = 0; j < dim; j++)
            {
              pointParam[iPoint * dim + j] = pointParamDealii[j];
            }
        }

      return pointParam;
    }

    template <dftfe::uInt dim>
    void
    FECell<dim>::getShapeFuncValuesFromParametricPoints(
      dftfe::uInt                     numPointsInCell,
      const std::vector<double>      &parametricPoints,
      std::vector<dataTypes::number> &shapeFuncValues,
      dftfe::uInt                     cellShapeFuncStartIndex,
      dftfe::uInt                     numDofsPerElement) const
    {
      for (dftfe::uInt iPoint = 0; iPoint < numPointsInCell; iPoint++)
        {
          dealii::Point<dim, double> pointParamDealii(
            parametricPoints[dim * iPoint + 0],
            parametricPoints[dim * iPoint + 1],
            parametricPoints[dim * iPoint + 2]);

          for (dftfe::uInt iNode = 0; iNode < numDofsPerElement; iNode++)
            {
              shapeFuncValues[cellShapeFuncStartIndex + iNode +
                              iPoint * numDofsPerElement] =
                d_feCell.shape_value(iNode, pointParamDealii);
            }
        }
    }

    template <dftfe::uInt dim>
    void
    FECell<dim>::getShapeFuncValues(
      dftfe::uInt                     numPointsInCell,
      const std::vector<double>      &coordinatesOfPointsInCell,
      std::vector<dataTypes::number> &shapeFuncValues,
      dftfe::uInt                     cellShapeFuncStartIndex,
      dftfe::uInt                     numDofsPerElement) const
    {
      for (dftfe::uInt iPoint = 0; iPoint < numPointsInCell; iPoint++)
        {
          dealii::Point<dim, double> realCoord(
            coordinatesOfPointsInCell[dim * iPoint + 0],
            coordinatesOfPointsInCell[dim * iPoint + 1],
            coordinatesOfPointsInCell[dim * iPoint + 2]);

          dealii::Point<dim, double> pointParamDealii =
            d_mappingQ1.transform_real_to_unit_cell(d_dealiiFECellIter,
                                                    realCoord);

          AssertThrow((pointParamDealii[0] > -1e-7) &&
                        (pointParamDealii[0] < 1 + 1e-7),
                      dealii::ExcMessage("param point x coord is -ve\n"));
          AssertThrow((pointParamDealii[1] > -1e-7) &&
                        (pointParamDealii[1] < 1 + 1e-7),
                      dealii::ExcMessage("param point y coord is -ve\n"));
          AssertThrow((pointParamDealii[2] > -1e-7) &&
                        (pointParamDealii[2] < 1 + 1e-7),
                      dealii::ExcMessage("param point z coord is -ve\n"));

          for (dftfe::uInt iNode = 0; iNode < numDofsPerElement; iNode++)
            {
              shapeFuncValues[cellShapeFuncStartIndex + iNode +
                              iPoint * numDofsPerElement] =
                d_feCell.shape_value(iNode, pointParamDealii);
            }

          double shapeValForNode = 0.0;
          for (dftfe::uInt iNode = 0; iNode < numDofsPerElement; iNode++)
            {
              shapeValForNode += realPart(
                complexConj(shapeFuncValues[cellShapeFuncStartIndex + iNode +
                                            iPoint * numDofsPerElement]) *
                shapeFuncValues[cellShapeFuncStartIndex + iNode +
                                iPoint * numDofsPerElement]);
            }
          if (std::abs(shapeValForNode) < 1e-3)
            {
              std::cout << " All shape func values are zero for a point \n";
            }
        }
    }
  } // end of namespace utils

} // end of namespace dftfe
