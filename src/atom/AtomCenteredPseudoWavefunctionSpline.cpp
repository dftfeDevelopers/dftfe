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

#include "AtomCenteredPseudoWavefunctionSpline.h"
#include "vector"
namespace dftfe
{
  AtomCenteredPseudoWavefunctionSpline::AtomCenteredPseudoWavefunctionSpline(
    std::string  filename,
    unsigned int l,
    double       cutoff,
    double       truncationTol)
  {
    d_lQuantumNumber = l;
    std::vector<std::vector<double>> radialFunctionData(0);
    dftUtils::readFile(2, radialFunctionData, filename);
    d_DataPresent = true;
    d_cutOff      = 0.0;
    d_rMin        = 0.0;


    unsigned int numRows = radialFunctionData.size() - 1;

    std::vector<double> xData(numRows), yData(numRows);

    unsigned int maxRowId = 0;
    for (unsigned int irow = 0; irow < numRows; ++irow)
      {
        xData[irow] = radialFunctionData[irow][0];
        // the input phi data is multiplied by radius and hence has to
        // be deleted
        yData[irow] = radialFunctionData[irow][1] / xData[irow];
        if (std::abs(yData[irow]) > truncationTol)
          maxRowId = irow;
      }

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

    unsigned int maxRowIndex = std::min(maxRowId + 10, numRows - 1);
    if (cutoff < 1e-3)
      {
        d_cutOff = xData[maxRowIndex];
      }
    else
      {
        d_cutOff = cutoff;
      }
    d_rMin = xData[0];
  }


} // end of namespace dftfe
