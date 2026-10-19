// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2026 The Regents of the University of Michigan and DFT-FE
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
// Host+device utility functions for the GPU-accelerated PAW/pseudopotential
// initialization paths: real spherical harmonics (matching the boost-based
// host conventions of sphericalHarmonicUtils) and cartesian-to-spherical
// conversion (replicating the host tolerance branches).
//

#ifndef DFTFE_ATOMCENTEREDSPHERICALFUNCTIONDEVICEUTILS_H
#define DFTFE_ATOMCENTEREDSPHERICALFUNCTIONDEVICEUTILS_H

#include <dftfe/splineDeviceEval.h>
#include <dftfe/TypeConfig.h>
#include <cmath>

namespace dftfe
{
  namespace atomCenteredSphericalFunctionDeviceUtils
  {
    /**
     * @brief Associated Legendre polynomial P_l^m(x), m >= 0, including the
     * Condon-Shortley phase (boost convention).
     */
    DFTFE_HOST_DEVICE_INLINE double
    associatedLegendreCS(const dftfe::Int l, const dftfe::Int m, const double x)
    {
      // P_m^m = (-1)^m (2m-1)!! (1-x^2)^{m/2}
      double pmm = 1.0;
      if (m > 0)
        {
          const double somx2 = std::sqrt((1.0 - x) * (1.0 + x));
          double       fact  = 1.0;
          for (dftfe::Int i = 1; i <= m; i++)
            {
              pmm *= -fact * somx2;
              fact += 2.0;
            }
        }
      if (l == m)
        return pmm;
      // P_{m+1}^m = x (2m+1) P_m^m
      double pmmp1 = x * (2.0 * m + 1.0) * pmm;
      if (l == m + 1)
        return pmmp1;
      double pll = 0.0;
      for (dftfe::Int ll = m + 2; ll <= l; ll++)
        {
          pll =
            (x * (2.0 * ll - 1.0) * pmmp1 - (ll + m - 1.0) * pmm) / (ll - m);
          pmm   = pmmp1;
          pmmp1 = pll;
        }
      return pll;
    }

    /**
     * @brief Real spherical harmonic in the convention of
     * sphericalHarmonicUtils::getSphericalHarmonicVal:
     * @code
     *   m == 0 : Y = N(l,0) P_l^0(cos t)
     *   m  > 0 : Y = sqrt(2) N(l,m)  P_l^m(cos t)  cos(m p)
     *   m  < 0 : Y = sqrt(2) N(l,|m|) P_l^|m|(cos t) sin(|m| p)
     * @endcode
     * with P_l^m from associatedLegendreCS and
     * N(l,m) = sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!).
     */
    DFTFE_HOST_DEVICE_INLINE double
    realSphericalHarmonic(const dftfe::Int l,
                          const dftfe::Int m,
                          const double     theta,
                          const double     phi)
    {
      const dftfe::Int absm = m < 0 ? -m : m;
      const double     x    = std::cos(theta);
      // N(l,m) = sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!)
      double factRatio = 1.0; // (l-|m|)!/(l+|m|)!
      for (dftfe::Int i = l - absm + 1; i <= l + absm; i++)
        factRatio /= static_cast<double>(i);
      const double norm = std::sqrt(
        (2.0 * l + 1.0) * 0.07957747154594766788444188168626 * factRatio);
      const double plm = associatedLegendreCS(l, absm, x);
      if (m == 0)
        return norm * plm;
      const double sqrt2 = 1.4142135623730950488016887242097;
      if (m > 0)
        return sqrt2 * norm * plm * std::cos(absm * phi);
      return sqrt2 * norm * plm * std::sin(absm * phi);
    }

    /**
     * @brief Spherical coordinates (r, theta, phi) of the point
     * (x0, x1, x2). An undetermined angle is set to zero: theta and phi
     * when r is within 1e-12 of zero, and phi when theta is within 1e-12 of
     * 0 or pi, where its value does not affect a spherical harmonic.
     */
    DFTFE_HOST_DEVICE_INLINE void
    cartesianToSpherical(const double x0,
                         const double x1,
                         const double x2,
                         double      &r,
                         double      &theta,
                         double      &phi)
    {
      const double tolerance = 1e-12;
      const double pi        = 3.14159265358979323846;
      r                      = std::sqrt(x0 * x0 + x1 * x1 + x2 * x2);
      if (std::fabs(r - 0.0) <= tolerance)
        {
          theta = 0.0;
          phi   = 0.0;
        }
      else
        {
          theta = std::acos(x2 / r);
          if (std::fabs(theta - 0.0) >= tolerance &&
              std::fabs(theta - pi) >= tolerance)
            phi = std::atan2(x1, x0);
          else
            phi = 0.0;
        }
    }
  } // namespace atomCenteredSphericalFunctionDeviceUtils
} // namespace dftfe

#endif // DFTFE_ATOMCENTEREDSPHERICALFUNCTIONDEVICEUTILS_H
