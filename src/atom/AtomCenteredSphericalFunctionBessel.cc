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

#include "AtomCenteredSphericalFunctionBessel.h"
#include "vector"
namespace dftfe
{
  AtomCenteredSphericalFunctionBessel::AtomCenteredSphericalFunctionBessel(
    double       RcParameter,
    double       RmaxParameter,
    unsigned int lParameter)
  {
    d_lQuantumNumber = lParameter;
    d_Rc             = RcParameter;
    d_cutOff         = RmaxParameter;
    using namespace boost::math::quadrature;
    AssertThrow(
      d_lQuantumNumber <= 2,
      dealii::ExcMessage(
        "DFT-FE Error:  Bessel functions only ti;; LQuantumNo 2 is defined"));
    std::vector<double> q1 = {3.141592653589793 / d_Rc,
                              4.493409457909095 / d_Rc,
                              5.76345919689455 / d_Rc};
    std::vector<double> q2 = {6.283185307179586 / d_Rc,
                              7.7252518369375 / d_Rc,
                              9.095011330476355 / d_Rc};

    auto f1 = [&](const double &x) {
      if (std::fabs(x) <= 1E-12)
        return (d_lQuantumNumber == 0 ? 1.0 : 0.0);
      else if (x > std::min(d_Rc, d_cutOff))
        return 0.0;
      else
        {
          // double alpha = -q1[d_lQuantumNumber] / q2[d_lQuantumNumber] *
          //                (std::sph_bessel(d_lQuantumNumber,
          //                q1[d_lQuantumNumber] * d_Rc)) /
          //                (std::sph_bessel(d_lQuantumNumber,
          //                q2[d_lQuantumNumber] * d_Rc));
          // double Value =
          //   x > d_Rc ? 0.0 :
          //            (std::sph_bessel(d_lQuantumNumber, q1[d_lQuantumNumber]
          //            * x) +
          //             alpha * (std::sph_bessel(d_lQuantumNumber,
          //             q2[d_lQuantumNumber] * x)));
          // return Value;
        }
    };
    d_NormalizationConstant =
      gauss_kronrod<double, 61>::integrate(f1, 0.0, d_cutOff, 15, 1e-12);
    d_rMinVal = getRadialValue(0.0);
  }

  double
  AtomCenteredSphericalFunctionBessel::getRadialValue(double r) const
  {
    double              Value = 0.0;
    std::vector<double> q1    = {3.141592653589793 / d_Rc,
                              4.493409457909095 / d_Rc,
                              5.76345919689455 / d_Rc};
    std::vector<double> q2    = {6.283185307179586 / d_Rc,
                              7.7252518369375 / d_Rc,
                              9.095011330476355 / d_Rc};
    // double alpha = -q1[d_lQuantumNumber] / q2[d_lQuantumNumber] *
    //                (std::sph_bessel(d_lQuantumNumber, q1[d_lQuantumNumber] *
    //                d_Rc)) / (std::sph_bessel(d_lQuantumNumber,
    //                q2[d_lQuantumNumber] * d_Rc));
    // Value =
    //  r > d_Rc ? 0.0 :
    //            (std::sph_bessel(d_lQuantumNumber, q1[d_lQuantumNumber] * r) +
    //             alpha * (std::sph_bessel(d_lQuantumNumber,
    //             q2[d_lQuantumNumber] * r)));
    return Value;
  }
  std::vector<double>
  AtomCenteredSphericalFunctionBessel::getDerivativeValue(double r) const
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }

  double
  AtomCenteredSphericalFunctionBessel::getrMinVal() const
  {
    return d_rMinVal;
  }
} // end of namespace dftfe
