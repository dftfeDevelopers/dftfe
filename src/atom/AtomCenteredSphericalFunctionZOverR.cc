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

#include "AtomCenteredSphericalFunctionZOverR.h"
#include "vector"
namespace dftfe
{
  AtomCenteredSphericalFunctionOneOverR::AtomCenteredSphericalFunctionSpline(
    double       Zval,
    double       Rtail,
    unsigned int l)
  {
    d_Zval           = Zval;
    d_cutOff         = Rtail;
    d_lQuantumNumber = l;
  }



  double
  AtomCenteredSphericalFunctionOneOverR::getRadialValue(double r) const
  {
    if (r >= d_cutOff)
      return 0.0;

    if (r <= d_rMin)
      r = d_rMin;

    double v = d_Zval / r;

    return v;
  }

  std::vector<double>
  AtomCenteredSphericalFunctionOneOverR::getDerivativeValue(double r) const
  {
    std::vector<double> Value(3, 0.0);

    return Value;
  }

  double
  AtomCenteredSphericalFunctionOneOverR::getrMinVal() const
  {
    return d_rMin;
  }
} // end of namespace dftfe
