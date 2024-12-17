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

#include "AtomCenteredSphericalFunctionSpline.h"
#include "vector"
namespace dftfe
{
  double
  AtomCenteredSphericalFunctionSpline::getRadialValue(double r) const
  {
    if (r >= d_cutOff)
      return 0.0;

    if (r <= d_rMin)
      r = d_rMin;

    double v = alglib::spline1dcalc(d_radialSplineObject, r);
    return v;
  }

  std::vector<double>
  AtomCenteredSphericalFunctionSpline::getDerivativeValue(double r) const
  {
    std::vector<double> Value(3, 0.0);
    if (r >= d_cutOff)
      return Value;

    if (r <= d_rMin)
      r = d_rMin;
    alglib::spline1ddiff(d_radialSplineObject, r, Value[0], Value[1], Value[2]);

    return Value;
  }

  double
  AtomCenteredSphericalFunctionSpline::getrMinVal() const
  {
    return d_rMin;
  }
} // end of namespace dftfe
