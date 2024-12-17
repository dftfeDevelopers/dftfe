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

#include "AtomCenteredSphericalFunctionLocalPotentialSpline.h"
#include "vector"
namespace dftfe
{
  AtomCenteredSphericalFunctionLocalPotentialSpline::
    AtomCenteredSphericalFunctionLocalPotentialSpline(std::string filename,
                                                      double      atomAttribute,
                                                      double      truncationTol,
                                                      double maxAllowedTail)
  {
    d_lQuantumNumber = 0;
    std::vector<std::vector<double>> radialFunctionData(0);
    unsigned int                     fileReadFlag =
      dftUtils::readPsiFile(2, radialFunctionData, filename);
    d_DataPresent = fileReadFlag;
    d_cutOff      = 0.0;
    d_rMin        = 0.0;
    if (fileReadFlag)
      {
        unsigned int        numRows = radialFunctionData.size() - 1;
        std::vector<double> xData(numRows), yData(numRows);

        unsigned int maxRowId = 0;
        for (unsigned int irow = 0; irow < numRows; ++irow)
          {
            xData[irow] = radialFunctionData[irow][0];
            yData[irow] = radialFunctionData[irow][1];

            if (irow > 0 && xData[irow] < maxAllowedTail)
              {
                if (std::abs(yData[irow] - (-(atomAttribute) / xData[irow])) >
                    truncationTol)
                  maxRowId = irow;
              }
          }
        d_cutOff = xData[maxRowId];
        d_rMin   = xData[0];
        // interpolate pseudopotentials
        alglib::real_1d_array x;
        x.setcontent(numRows, &xData[0]);
        alglib::real_1d_array y;
        y.setcontent(numRows, &yData[0]);
        alglib::ae_int_t bound_type_l = 0;
        alglib::ae_int_t bound_type_r = 1;
        const double     slopeL =
          (radialFunctionData[1][1] - radialFunctionData[0][1]) /
          (radialFunctionData[1][0] - radialFunctionData[0][0]);
        const double slopeR = -radialFunctionData[numRows - 1][1] /
                              radialFunctionData[numRows - 1][0];
        spline1dbuildcubic(x,
                           y,
                           numRows,
                           bound_type_l,
                           slopeL,
                           bound_type_r,
                           slopeR,
                           d_radialSplineObject);
      }
  }

} // end of namespace dftfe
