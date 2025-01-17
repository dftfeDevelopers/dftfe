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

#ifndef DFTFE_ATOMCENTEREDSPHERICALFUNCTIONGAUSSIAN_H
#define DFTFE_ATOMCENTEREDSPHERICALFUNCTIONGAUSSIAN_H

#include "AtomCenteredSphericalFunctionBase.h"

#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <dftUtils.h>
#include <vector>
#include <cmath>

namespace dftfe
{
  class AtomCenteredSphericalFunctionGaussian
    : public AtomCenteredSphericalFunctionBase
  {
  public:
    AtomCenteredSphericalFunctionGaussian(double       RcParameter,
                                          double       RmaxParameter,
                                          unsigned int lParameter);

    double
    getRadialValue(double r) const override;

    unsigned int
    getQuantumNumbern() const;

    double
    getrMinVal() const;

    std::vector<double>
    getDerivativeValue(double r) const override;

  private:
    double d_rMinVal;
    double d_NormalizationConstant;
    double d_Rc;
  };

} // end of namespace dftfe

#endif // DFTFE_ATOMCENTEREDSPHERICALFUNCTIONGAUSSIAN_H
