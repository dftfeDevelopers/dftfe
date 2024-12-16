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

#include "AtomCenteredSphericalFunctionBase.h"
#include <dftUtils.h>

namespace dftfe
{
  unsigned int
  AtomCenteredSphericalFunctionBase::getQuantumNumberl() const
  {
    return d_lQuantumNumber;
  }

  double
  AtomCenteredSphericalFunctionBase::getRadialCutOff() const
  {
    return d_cutOff;
  }

  double
  AtomCenteredSphericalFunctionBase::getIntegralValue() const
  {
    using namespace boost::math::quadrature;

    auto Integrate = [&](const double &t) {
      double px = getRadialValue(t);
      return ((px)*pow(t, 2));
    };
    double TotalVal = gauss_kronrod<double, 61>::integrate(
      Integrate, 0, getRadialCutOff(), 15, 1e-12);
    return (TotalVal * sqrt(4 * M_PI));
  }
  bool
  AtomCenteredSphericalFunctionBase::isDataPresent() const
  {
    return (d_DataPresent);
  }

} // namespace dftfe
