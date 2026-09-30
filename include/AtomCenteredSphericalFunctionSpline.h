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
// @author  Vishal Subramanian, Kartick Ramakrishnan, Sambit Das
//

#ifndef DFTFE_ATOMCENTEREDSPHERICALFUNCTIONSPLINE_H
#define DFTFE_ATOMCENTEREDSPHERICALFUNCTIONSPLINE_H

#include "AtomCenteredSphericalFunctionBase.h"
#include "string"
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <vector>
#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <fileReaders.h>
#include <dftUtils.h>
#include <interpolation.h>


namespace dftfe
{
  class AtomCenteredSphericalFunctionSpline
    : public AtomCenteredSphericalFunctionBase
  {
  public:
    double
    getRadialValue(double r) const override;

    std::vector<double>
    getDerivativeValue(double r) const override;

    double
    getrMinVal() const;

  protected:
    double d_rMin;

    alglib::spline1dinterpolant d_radialSplineObject;
  };

} // end of namespace dftfe

#endif // DFTFE_ATOMCENTEREDSPHERICALFUNCTIONSPLINE_H
