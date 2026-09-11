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
// host conventions of sphericalHarmonicUtils), cartesian-to-spherical
// conversion (replicating the host tolerance branches),
// and a stateless counter-based random number generator (splitmix64) that
// produces identical streams on NVIDIA, AMD and Intel devices and is
// independent of the MPI/GPU decomposition.
//

#ifndef DFTFE_ATOMCENTEREDSPHERICALFUNCTIONDEVICEUTILS_H
#define DFTFE_ATOMCENTEREDSPHERICALFUNCTIONDEVICEUTILS_H

#include <dftfe/splineDeviceEval.h> // for DFTFE_HOST_DEVICE_INLINE
#include <cmath>

namespace dftfe
{
  namespace atomCenteredSphericalFunctionDeviceUtils
  {
    // ---------------- real spherical harmonics ----------------
    // Host convention (sphericalHarmonicUtils::getSphericalHarmonicVal):
    //   m == 0 : Y = N(l,0) P_l^0(cos t)
    //   m  > 0 : Y = sqrt(2) N(l,m)  P_l^m(cos t)  cos(m p)
    //   m  < 0 : Y = sqrt(2) N(l,|m|) P_l^|m|(cos t) sin(|m| p)
    // with P_l^m the associated Legendre polynomial INCLUDING the
    // Condon-Shortley phase (boost convention) and
    // N(l,m) = sqrt((2l+1)/(4 pi) (l-m)!/(l+m)!).

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

    // Replicates sphericalHarmonicUtils::convertCartesianToSpherical
    // including its 1e-12 tolerance branches.
    DFTFE_HOST_DEVICE_INLINE void
    cartesianToSpherical(const double x0,
                         const double x1,
                         const double x2,
                         double      &r,
                         double      &theta,
                         double      &phi)
    {
      const double tolerance = 1e-12;
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
              std::fabs(theta - M_PI) >= tolerance)
            phi = std::atan2(x1, x0);
          else
            phi = 0.0;
        }
    }

    // ---------------- portable counter-based RNG ----------------
    // Seed of the wavefunction initial guess. The host and device paths must
    // share it or the two would produce different guesses.
    constexpr unsigned long long randomWavefunctionSeed = 137137ULL;

    // The generator below deliberately uses unsigned long long rather than
    // dftfe::uInt: dftfe::uInt is 32 bit unless USE_64BIT_INT is set, and a
    // narrower mixer would change the random stream between builds.
    //
    // splitmix64: high-quality 64-bit mixer; identical results on all
    // device vendors (pure integer arithmetic) and independent of the
    // parallel decomposition when keyed on global indices.
    DFTFE_HOST_DEVICE_INLINE unsigned long long
    splitmix64(unsigned long long z)
    {
      z += 0x9E3779B97F4A7C15ULL;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      return z ^ (z >> 31);
    }

    // uniform double in [-0.5, 0.5) from a set of integer keys
    DFTFE_HOST_DEVICE_INLINE double
    uniformRandomCentered(const unsigned long long seed,
                          const unsigned long long key0,
                          const unsigned long long key1,
                          const unsigned long long key2)
    {
      unsigned long long h = splitmix64(seed ^ splitmix64(key0));
      h                    = splitmix64(h ^ splitmix64(key1));
      h                    = splitmix64(h ^ splitmix64(key2));
      // top 53 bits -> double in [0,1)
      const double u = static_cast<double>(h >> 11) * 1.1102230246251565e-16;
      return u - 0.5;
    }
  } // namespace atomCenteredSphericalFunctionDeviceUtils
} // namespace dftfe

#endif // DFTFE_ATOMCENTEREDSPHERICALFUNCTIONDEVICEUTILS_H
