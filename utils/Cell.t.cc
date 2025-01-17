

namespace dftfe
{
  namespace utils
  {
    template <unsigned int dim>
    Cell<dim>::Cell(const std::vector<double> &ll,
                    const std::vector<double> &ur)
    {
      d_lowerLeft.resize(dim, 0.0);
      d_upperRight.resize(dim, 0.0);
      for (unsigned int j = 0; j < dim; j++)
        {
          d_lowerLeft[j]  = ll[j];
          d_upperRight[j] = ur[j];
        }
    }

    template <unsigned int dim>
    Cell<dim>::Cell()
    {
      d_lowerLeft.resize(dim, 0.0);
      d_upperRight.resize(dim, 0.0);
    }

    template <unsigned int dim>
    std::pair<std::vector<double>, std::vector<double>>
    Cell<dim>::getBoundingBox() const
    {
      auto pp = std::make_pair(d_lowerLeft, d_upperRight);

      return pp;
    }

    template <unsigned int dim>
    bool
    Cell<dim>::isPointInside(const std::vector<double> &point,
                             const double               tol) const
    {
      bool returnVal = true;
      for (unsigned int j = 0; j < dim; j++)
        {
          if ((point[j] < d_lowerLeft[j] - tol) ||
              (point[j] > d_upperRight[j] + tol))
            {
              returnVal = false;
            }
        }
      return returnVal;
    }

    template <unsigned int dim>
    void
    Cell<dim>::getVertices(std::vector<std::vector<double>> &points) const
    {
      size_type numPoints = std::pow(2, dim);
      points.resize(numPoints, std::vector<double>(dim, 0.0));

      for (size_type iPoint = 0; iPoint < numPoints; iPoint++)
        {
          getVertex(iPoint, points[iPoint]);
        }
    }

    template <unsigned int dim>
    void
    Cell<dim>::getVertex(size_type i, std::vector<double> &point) const
    {
      point.resize(dim, 0.0);
      for (size_type iDim = 0; iDim < dim; iDim++)
        {
          size_type denom      = std::pow(2, iDim);
          size_type coordIndex = i / denom;
          size_type coord      = coordIndex % 2;
          if (coord == 1)
            point[iDim] = d_upperRight[iDim];
          else
            point[iDim] = d_lowerLeft[iDim];
        }
    }

    template <unsigned int dim>
    std::vector<double>
    Cell<dim>::getParametricPoint(const std::vector<double> &realPoint) const
    {
      std::vector<double> pointParam(dim, 0.0);
      for (unsigned int j = 0; j < dim; j++)
        {
          pointParam[j] = (realPoint[j] - d_lowerLeft[j]) /
                          (d_upperRight[j] - d_lowerLeft[j]);
        }

      return pointParam;
    }

    //    template <unsigned int dim>
    //    void
    //    Cell<dim>::getShapeFuncValues(unsigned int numPointsInCell,
    //                                  const std::vector<double>
    //                                  &coordinatesOfPointsInCell,
    //                                  std::vector<dataTypes::number>
    //                                  &shapeFuncValues, unsigned int
    //                                  cellShapeFuncStartIndex, unsigned int
    //                                  numDofsPerElement) const
    //    {
    //      AssertThrow(false,
    //                  dealii::ExcMessage("getting shape function values is not
    //                  possible for Cell\n"));
    //      exit(0);
    //    }

  } // end of namespace utils

} // end of namespace dftfe
