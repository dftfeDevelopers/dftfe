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

#ifndef DFTFE_ATOMCENTEREDPSEUDOWAVEFUNCTIONSPLINE_H
#define DFTFE_ATOMCENTEREDPSEUDOWAVEFUNCTIONSPLINE_H

#include "AtomCenteredSphericalFunctionSpline.h"
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
  class AtomCenteredPseudoWavefunctionSpline
    : public AtomCenteredSphericalFunctionSpline
  {
  public:
    /**
     * @brief Creates splines for radial-Local Potential from file by applying suitable BC on spline and determining the cutOff Radius
     * @param[in] filename the location of file containing the data
     * @param[in] l quantumNumber-l
     * @param[in] cutoff  the distance beyond which the radial function is cut
     * off. If it is set to less than 1e-3 then truncationTol is considered,
     * otherwise cutoff takes precedence.
     * @param[in] truncationTol the minimum function value after which the
     * function is truncated.
     */
    AtomCenteredPseudoWavefunctionSpline(std::string  filename,
                                         unsigned int l,
                                         double       cutoff,
                                         double       truncationTol = 1E-10);
  };

} // end of namespace dftfe

#endif // DFTFE_ATOMCENTEREDPSEUDOWAVEFUNCTIONSPLINE_H
