// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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

#include "AtomCenteredSphericalFunctionProjectorSpline.h"
#include "vector"
namespace dftfe
{
  AtomCenteredSphericalFunctionProjectorSpline::
    AtomCenteredSphericalFunctionProjectorSpline(std::string  filename,
                                                 unsigned int l,
                                                 int          radialPower,
                                                 int          colIndex,
                                                 int          totalColSize,
                                                 double       truncationTol,
                                                 bool         consider0thEntry)
  {
    d_lQuantumNumber = l;
    std::vector<std::vector<double>> radialFunctionData(0);
    dftUtils::readFile(totalColSize, radialFunctionData, filename);
    d_DataPresent = true;
    d_cutOff      = 0.0;
    d_rMin        = 0.0;


    unsigned int        numRows = radialFunctionData.size() - 1;
    std::vector<double> xData(numRows), yData(numRows);

    unsigned int maxRowId = 0;

    for (unsigned int irow = 0; irow < numRows; ++irow)
      {
        xData[irow] = radialFunctionData[irow][0];
        if (radialPower == 0)
          yData[irow] = radialFunctionData[irow][colIndex];
        else
          {
            if (xData[irow] != 0)
              yData[irow] = radialFunctionData[irow][colIndex] *
                            pow(xData[irow], radialPower);
            else
              consider0thEntry = true;
          }
        if (std::abs(yData[irow]) > truncationTol)
          maxRowId = irow;
      }


    if (!consider0thEntry)
      yData[0] = yData[1];

    alglib::real_1d_array x;
    x.setcontent(numRows, &xData[0]);
    alglib::real_1d_array y;
    y.setcontent(numRows, &yData[0]);
    alglib::ae_int_t natural_bound_type_L = 1;
    alglib::ae_int_t natural_bound_type_R = 1;
    spline1dbuildcubic(x,
                       y,
                       numRows,
                       natural_bound_type_L,
                       0.0,
                       natural_bound_type_R,
                       0.0,
                       d_radialSplineObject);
    d_cutOff = xData[std::min(maxRowId + 10, numRows - 1)];
    d_rMin   = xData[0];
  }


} // end of namespace dftfe
