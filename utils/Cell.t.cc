

namespace dftfe
{
  namespace utils
  {
    template <dftfe::uInt dim>
    Cell<dim>::Cell(const std::vector<double> &ll,
                    const std::vector<double> &ur)
    {
      d_lowerLeft.resize(dim, 0.0);
      d_upperRight.resize(dim, 0.0);
      for (dftfe::uInt j = 0; j < dim; j++)
        {
          d_lowerLeft[j]  = ll[j];
          d_upperRight[j] = ur[j];
        }
    }

    template <dftfe::uInt dim>
    Cell<dim>::Cell()
    {
      d_lowerLeft.resize(dim, 0.0);
      d_upperRight.resize(dim, 0.0);
    }

    template <dftfe::uInt dim>
    std::pair<std::vector<double>, std::vector<double>>
    Cell<dim>::getBoundingBox() const
    {
      auto pp = std::make_pair(d_lowerLeft, d_upperRight);

      return pp;
    }

    template <dftfe::uInt dim>
    bool
    Cell<dim>::isPointInside(const std::vector<double> &point,
                             const double               tol) const
    {
      bool returnVal = true;
      for (dftfe::uInt j = 0; j < dim; j++)
        {
          if ((point[j] < d_lowerLeft[j] - tol) ||
              (point[j] > d_upperRight[j] + tol))
            {
              returnVal = false;
            }
        }
      return returnVal;
    }

    template <dftfe::uInt dim>
    void
    Cell<dim>::getVertices(std::vector<std::vector<double>> &points) const
    {
      dftfe::uInt numPoints = std::pow(2, dim);
      points.resize(numPoints, std::vector<double>(dim, 0.0));

      for (dftfe::uInt iPoint = 0; iPoint < numPoints; iPoint++)
        {
          getVertex(iPoint, points[iPoint]);
        }
    }

    template <dftfe::uInt dim>
    void
    Cell<dim>::getVertex(dftfe::uInt i, std::vector<double> &point) const
    {
      point.resize(dim, 0.0);
      for (dftfe::uInt iDim = 0; iDim < dim; iDim++)
        {
          dftfe::uInt denom      = std::pow(2, iDim);
          dftfe::uInt coordIndex = i / denom;
          dftfe::uInt coord      = coordIndex % 2;
          if (coord == 1)
            point[iDim] = d_upperRight[iDim];
          else
            point[iDim] = d_lowerLeft[iDim];
        }
    }

    template <dftfe::uInt dim>
    std::vector<double>
    Cell<dim>::getParametricPoint(const std::vector<double> &realPoint) const
    {
      std::vector<double> pointParam(dim, 0.0);
      for (dftfe::uInt j = 0; j < dim; j++)
        {
          pointParam[j] = (realPoint[j] - d_lowerLeft[j]) /
                          (d_upperRight[j] - d_lowerLeft[j]);
        }

      return pointParam;
    }

    //    template <dftfe::uInt dim>
    //    void
    //    Cell<dim>::getShapeFuncValues(dftfe::uInt numPointsInCell,
    //                                  const std::vector<double>
    //                                  &coordinatesOfPointsInCell,
    //                                  std::vector<dataTypes::number>
    //                                  &shapeFuncValues, dftfe::uInt
    //                                  cellShapeFuncStartIndex, dftfe::uInt
    //                                  numDofsPerElement) const
    //    {
    //      AssertThrow(false,
    //                  dealii::ExcMessage("getting shape function values is not
    //                  possible for Cell\n"));
    //      exit(0);
    //    }

  } // end of namespace utils

} // end of namespace dftfe
