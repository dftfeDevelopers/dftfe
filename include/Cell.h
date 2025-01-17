/******************************************************************************
 * Copyright (c) 2021.                                                        *
 * The Regents of the University of Michigan and DFT-EFE developers.          *
 *                                                                            *
 * This file is part of the DFT-EFE code.                                     *
 *                                                                            *
 * DFT-EFE is free software: you can redistribute it and/or modify            *
 *   it under the terms of the Lesser GNU General Public License as           *
 *   published by the Free Software Foundation, either version 3 of           *
 *   the License, or (at your option) any later version.                      *
 *                                                                            *
 * DFT-EFE is distributed in the hope that it will be useful, but             *
 *   WITHOUT ANY WARRANTY; without even the implied warranty                  *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                     *
 *   See the Lesser GNU General Public License for more details.              *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public           *
 *   License at the top level of DFT-EFE distribution.  If not, see           *
 *   <https://www.gnu.org/licenses/>.                                         *
 ******************************************************************************/

/*
 * @author Vishal Subramanian, Bikash Kanungo
 */



#ifndef DFTFE_CELL_H
#define DFTFE_CELL_H

#include "headers.h"

namespace dftfe
{
  namespace utils
  {
    /**
     * @brief This class provides the interface that will be required while interpolating a nodal
     * data to arbitrary set of points.
     *
     * @author Vishal Subramanian, Bikash Kanungo
     */
    template <size_type dim>
    class Cell
    {
    public:
      Cell();

      Cell(const std::vector<double> &ll, const std::vector<double> &ur);

      virtual std::pair<std::vector<double>, std::vector<double>>
      getBoundingBox() const;

      virtual bool
      isPointInside(const std::vector<double> &point, const double tol) const;

      virtual void
      getVertices(std::vector<std::vector<double>> &points) const;

      virtual void
      getVertex(size_type i, std::vector<double> &point) const;

      virtual std::vector<double>
      getParametricPoint(const std::vector<double> &realPoint) const;

      virtual void
      getShapeFuncValues(unsigned int               numPointsInCell,
                         const std::vector<double> &coordinatesOfPointsInCell,
                         std::vector<dataTypes::number> &shapeFuncValues,
                         unsigned int cellShapeFuncStartIndex,
                         unsigned int numDofsPerElement) const = 0;

    private:
      std::vector<double> d_lowerLeft, d_upperRight;
    }; // end of class CellBase
  }    // end of namespace utils
} // end of namespace dftfe

#include "../utils/Cell.t.cc"

#endif // DFTFE_CELL_H
