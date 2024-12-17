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

#include "AtomCenteredSphericalFunctionSinc.h"
#include "vector"
namespace dftfe
{
  AtomCenteredSphericalFunctionSinc::AtomCenteredSphericalFunctionSinc(
    double       RcParameter,
    double       RmaxParameter,
    unsigned int lParameter)
  {
    d_lQuantumNumber = lParameter;
    d_Rc             = RcParameter;
    d_cutOff         = RmaxParameter;
    using namespace boost::math::quadrature;
    auto f1 = [&](const double &x) {
      if (std::fabs(x) <= 1E-12)
        return (d_lQuantumNumber == 0 ? 1.0 : 0.0);
      else if (x > std::min(d_Rc, d_cutOff))
        return 0.0;
      else
        {
          double sincValue = boost::math::sinc_pi(x / d_Rc * M_PI);
          double Value     = 0.0;
          Value = pow(x, 2 * d_lQuantumNumber + 2) * pow(sincValue, 2);
          return Value;
        }
    };
    d_NormalizationConstant =
      gauss_kronrod<double, 61>::integrate(f1, 0.0, d_cutOff, 15, 1e-12);
    d_rMinVal = getRadialValue(0.0);
  }

  double
  AtomCenteredSphericalFunctionSinc::getRadialValue(double r) const
  {
    if (r >= d_cutOff)
      return 0.0;
    double sincValue = boost::math::sinc_pi(r / d_Rc * M_PI);
    double Value     = r > d_Rc ? 0.0 : sincValue * sincValue;
    if (d_lQuantumNumber > 0)
      Value *= pow(r, d_lQuantumNumber);
    Value /= d_NormalizationConstant;
    return Value;
  }

  std::vector<double>
  AtomCenteredSphericalFunctionSinc::getDerivativeValue(double r) const
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }

  double
  AtomCenteredSphericalFunctionSinc::getrMinVal() const
  {
    return d_rMinVal;
  }
} // end of namespace dftfe
