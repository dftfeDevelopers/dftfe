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
#ifndef DFTFE_SPHERICALHARMONICUTILS_H
#define DFTFE_SPHERICALHARMONICUTILS_H
#include <dftfe/AtomCenteredSphericalFunctionDeviceUtils.h>
#include <interpolation.h>
namespace dftfe
{
  namespace sphericalHarmonicUtils
  {
    inline void
    getRadialFunctionVal(const double                       radialCoordinate,
                         double                            &splineVal,
                         const alglib::spline1dinterpolant *spline)
    {
      splineVal = alglib::spline1dcalc(*spline, radialCoordinate);
      return;
    }

    inline void
    getSphericalHarmonicVal(const double     theta,
                            const double     phi,
                            const dftfe::Int l,
                            const dftfe::Int m,
                            double          &sphericalHarmonicVal)
    {
      if (m < 0)
        sphericalHarmonicVal =
          std::sqrt(2.0) * boost::math::spherical_harmonic_i(l, -m, theta, phi);

      else if (m == 0)
        sphericalHarmonicVal =
          boost::math::spherical_harmonic_r(l, m, theta, phi);

      else if (m > 0)
        sphericalHarmonicVal =
          std::sqrt(2.0) * boost::math::spherical_harmonic_r(l, m, theta, phi);

      return;
    }

    /**
     * @brief atomCenteredSphericalFunctionDeviceUtils::cartesianToSpherical
     * of (x[0], x[1], x[2]).
     */
    inline void
    convertCartesianToSpherical(double *x,
                                double &r,
                                double &theta,
                                double &phi)
    {
      atomCenteredSphericalFunctionDeviceUtils::cartesianToSpherical(
        x[0], x[1], x[2], r, theta, phi);
    }
  } // end of namespace sphericalHarmonicUtils
} // end of namespace dftfe
#endif // DFTFE_SPHERICALHARMONICUTILS_H
