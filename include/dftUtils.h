// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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



#ifndef dftUtils_H_
#define dftUtils_H_

#include <headers.h>
#include <mpi.h>

#include <map>

namespace dftfe
{
  /**
   *  @brief Contains repeatedly used functions in the KSDFT calculations
   *
   *  @author Sambit Das, Krishnendu Ghosh, Phani Motamarri
   */

  namespace dftUtils
  {
    extern "C"
    {
      //
      // lapack Ax=b
      //
      void
      dgesv_(int    *N,
             int    *NRHS,
             double *A,
             int    *LDA,
             int    *IPIV,
             double *B,
             int    *LDB,
             int    *INFO);
    }

    inline double
    smearedCharge(double r, double rc)
    {
      if (r > rc)
        return 0.0;

      const double rmrc = r - rc;
      const double r2   = r * r;
      const double rc2  = rc * rc;
      const double rc4  = rc2 * rc2;
      const double rc8  = rc4 * rc4;

      return -21.0 * rmrc * rmrc * rmrc * (6.0 * r2 + 3.0 * r * rc + rc2) /
             (5.0 * M_PI * rc8);
    }

    inline double
    smearedChargeDr(double r, double rc)
    {
      if (r > rc)
        return 0.0;

      const double rmrc  = r - rc;
      const double rmrc2 = rmrc * rmrc;
      const double r2    = r * r;
      const double rc2   = rc * rc;
      const double rc4   = rc2 * rc2;
      const double rc8   = rc4 * rc4;

      return (-63.0 * rmrc2 * (6.0 * r2 + 3.0 * r * rc + rc2) -
              63.0 * rmrc2 * rmrc * (4.0 * r + rc)) /
             (5.0 * M_PI * rc8);
    }

    inline double
    smearedPot(double r, double rc)
    {
      if (r > rc)
        return 1.0 / r;

      const double r2  = r * r;
      const double r4  = r2 * r2;
      const double r5  = r4 * r;
      const double r6  = r5 * r;
      const double r7  = r6 * r;
      const double rc2 = rc * rc;
      const double rc4 = rc2 * rc2;
      const double rc5 = rc4 * rc;
      const double rc7 = rc5 * rc2;
      const double rc8 = rc4 * rc4;

      return (9.0 * r7 - 30.0 * r6 * rc + 28.0 * r5 * rc2 - 14.0 * r2 * rc5 +
              12.0 * rc7) /
             (5.0 * rc8);
    }

    // derivative w.r.t r
    inline double
    smearedPotDr(double r, double rc)
    {
      if (r > rc)
        return -1.0 / (r * r);

      const double r2  = r * r;
      const double r4  = r2 * r2;
      const double r5  = r4 * r;
      const double r6  = r5 * r;
      const double rc2 = rc * rc;
      const double rc4 = rc2 * rc2;
      const double rc5 = rc4 * rc;
      const double rc8 = rc4 * rc4;

      return (63.0 * r6 - 180.0 * r5 * rc + 140.0 * r4 * rc2 - 28.0 * r * rc5) /
             (5.0 * rc8);
    }


    inline double
    smearedPairInteractionEqualRadius(const double radius,
                                      const double separation)
    {
      if (radius <= 0.0)
        return 0.0;

      if (separation <= 1.0e-14)
        return (31924.0 / 17875.0) / radius;

      if (separation >= 2.0 * radius)
        return 1.0 / separation;

      const double s = separation / radius;
      double       p;
      if (s < 1.0)
        {
          p = 81.0;
          p = p * s - 360.0;
          p = p * s + 1035.0;
          p = p * s - 4200.0;
          p = p * s + 10920.0;
          p = p * s - 6552.0;
          p = p * s - 30030.0;
          p = p * s + 62920.0;
          p = p * s;
          p = p * s - 102960.0;
          p = p * s;
          p = p * s + 196560.0;
          p = p * s;
          p = p * s - 299600.0;
          p = p * s;
          p = p * s + 383088.0;
          return p / 214500.0 / radius;
        }

      p = 27.0;
      p = p * s - 360.0;
      p = p * s + 1845.0;
      p = p * s - 4200.0;
      p = p * s + 3640.0;
      p = p * s - 6552.0;
      p = p * s + 30030.0;
      p = p * s + 62920.0;
      p = p * s - 772200.0;
      p = p * s + 2471040.0;
      p = p * s - 4420416.0;
      p = p * s + 5241600.0;
      p = p * s - 4542720.0;
      p = p * s + 2867200.0;
      p = p * s - 944640.0;
      p = p * s - 178176.0;
      p = p * s - 19940.0;
      return -p / (214500.0 * s * radius);
    }

    inline double
    smearedPairInteractionDerEqualRadius(const double radius,
                                         const double separation)
    {
      if (radius <= 0.0)
        return 0.0;

      if (separation <= 1.0e-12)
        return 0.0;

      if (separation >= 2.0 * radius)
        return -1.0 / (separation * separation);

      const double s = separation / radius;
      double       p;
      if (s < 1.0)
        {
          p = 243.0;
          p = p * s - 1008.0;
          p = p * s + 2691.0;
          p = p * s - 10080.0;
          p = p * s + 24024.0;
          p = p * s - 13104.0;
          p = p * s - 54054.0;
          p = p * s + 100672.0;
          p = p * s;
          p = p * s - 123552.0;
          p = p * s;
          p = p * s + 157248.0;
          p = p * s;
          p = p * s - 119840.0;
          return s * p / 42900.0 / (radius * radius);
        }

      p = 81.0;
      p = p * s - 1008.0;
      p = p * s + 4797.0;
      p = p * s - 10080.0;
      p = p * s + 8008.0;
      p = p * s - 13104.0;
      p = p * s + 54054.0;
      p = p * s + 100672.0;
      p = p * s - 1081080.0;
      p = p * s + 2965248.0;
      p = p * s - 4420416.0;
      p = p * s + 4193280.0;
      p = p * s - 2725632.0;
      p = p * s + 1146880.0;
      p = p * s - 188928.0;
      p = p * s;
      p = p * s + 3988.0;
      return -p / (42900.0 * s * s * radius * radius);
    }


    inline void
    addUniqueBreakpoint(std::vector<double> &breakpoints,
                        const double         value,
                        const double         lower,
                        const double         upper)
    {
      if (value <= lower || value >= upper)
        return;

      const double tolerance =
        1.0e-12 * std::max(1.0, std::max(std::abs(lower), std::abs(upper)));
      for (const double breakpoint : breakpoints)
        if (std::abs(value - breakpoint) <= tolerance)
          return;

      breakpoints.push_back(value);
    }


    struct SmearedPairRadiusCoefficients
    {
      explicit SmearedPairRadiusCoefficients(const double radius)
      {
        rgCharge.fill(0.0);
        r2gCharge.fill(0.0);
        primitiveInside.fill(0.0);
        primitiveOutside.fill(0.0);
        primitiveDerivativeInside.fill(0.0);
        primitiveDerivativeOutside.fill(0.0);
        potential.fill(0.0);

        const double r2 = radius * radius;
        const double r3 = r2 * radius;
        const double r6 = r3 * r3;
        const double r7 = r6 * radius;
        const double r8 = r7 * radius;

        rgCharge[1] = 21.0 / (5.0 * M_PI * r3);
        rgCharge[4] = -42.0 / (M_PI * r6);
        rgCharge[5] = 63.0 / (M_PI * r7);
        rgCharge[6] = -126.0 / (5.0 * M_PI * r8);

        r2gCharge[2] = 21.0 / (5.0 * M_PI * r3);
        r2gCharge[5] = -42.0 / (M_PI * r6);
        r2gCharge[6] = 63.0 / (M_PI * r7);
        r2gCharge[7] = -126.0 / (5.0 * M_PI * r8);

        primitiveInside[2] = 6.0 / (5.0 * radius);
        primitiveInside[4] = -7.0 / (10.0 * r3);
        primitiveInside[7] = 4.0 / (5.0 * r6);
        primitiveInside[8] = -3.0 / (4.0 * r7);
        primitiveInside[9] = 1.0 / (5.0 * r8);

        primitiveOutside[0] = -0.25 * radius;
        primitiveOutside[1] = 1.0;

        primitiveDerivativeInside[1] = 12.0 / (5.0 * radius);
        primitiveDerivativeInside[3] = -14.0 / (5.0 * r3);
        primitiveDerivativeInside[6] = 28.0 / (5.0 * r6);
        primitiveDerivativeInside[7] = -6.0 / r7;
        primitiveDerivativeInside[8] = 9.0 / (5.0 * r8);

        primitiveDerivativeOutside[0] = 1.0;

        potential[0] = 12.0 / (5.0 * radius);
        potential[2] = -14.0 / (5.0 * r3);
        potential[5] = 28.0 / (5.0 * r6);
        potential[6] = -6.0 / r7;
        potential[7] = 9.0 / (5.0 * r8);
      }

      std::array<double, 10> rgCharge;
      std::array<double, 10> r2gCharge;
      std::array<double, 10> primitiveInside;
      std::array<double, 10> primitiveOutside;
      std::array<double, 10> primitiveDerivativeInside;
      std::array<double, 10> primitiveDerivativeOutside;
      std::array<double, 10> potential;
    };

    inline const SmearedPairRadiusCoefficients &
    smearedPairGetRadiusCoefficients(const double radius)
    {
      thread_local std::map<double, SmearedPairRadiusCoefficients> cache;
      const auto iterator = cache.find(radius);
      if (iterator != cache.end())
        return iterator->second;

      const auto insertion =
        cache.emplace(radius, SmearedPairRadiusCoefficients(radius));
      return insertion.first->second;
    }

    inline double
    smearedPairShiftedPolynomialIntegral(
      const std::array<double, 10> &radialCoefficients,
      const std::array<double, 10> &shiftedCoefficients,
      const double                  shift,
      const double                  radialSign,
      const double                  lower,
      const double                  upper)
    {
      if (upper <= lower)
        return 0.0;

      static constexpr double binomial[10][10] = {
        {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0, 3.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0, 4.0, 6.0, 4.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {1.0, 5.0, 10.0, 10.0, 5.0, 1.0, 0.0, 0.0, 0.0, 0.0},
        {1.0, 6.0, 15.0, 20.0, 15.0, 6.0, 1.0, 0.0, 0.0, 0.0},
        {1.0, 7.0, 21.0, 35.0, 35.0, 21.0, 7.0, 1.0, 0.0, 0.0},
        {1.0, 8.0, 28.0, 56.0, 70.0, 56.0, 28.0, 8.0, 1.0, 0.0},
        {1.0, 9.0, 36.0, 84.0, 126.0, 126.0, 84.0, 36.0, 9.0, 1.0}};

      std::array<double, 17> lowerPowers;
      std::array<double, 17> upperPowers;
      lowerPowers[0] = 1.0;
      upperPowers[0] = 1.0;
      for (dftfe::uInt i = 1; i < lowerPowers.size(); ++i)
        {
          lowerPowers[i] = lowerPowers[i - 1] * lower;
          upperPowers[i] = upperPowers[i - 1] * upper;
        }

      std::array<double, 10> shiftPowers;
      shiftPowers[0] = 1.0;
      for (dftfe::uInt i = 1; i < shiftPowers.size(); ++i)
        shiftPowers[i] = shiftPowers[i - 1] * shift;

      double integral = 0.0;
      for (dftfe::uInt radialPower = 0; radialPower < radialCoefficients.size();
           ++radialPower)
        if (radialCoefficients[radialPower] != 0.0)
          for (dftfe::uInt shiftedPower = 0;
               shiftedPower < shiftedCoefficients.size();
               ++shiftedPower)
            if (shiftedCoefficients[shiftedPower] != 0.0)
              for (dftfe::uInt k = 0; k <= shiftedPower; ++k)
                {
                  const dftfe::uInt power = radialPower + k;
                  const double signFactor = (k % 2 == 0) ? 1.0 : radialSign;
                  integral +=
                    radialCoefficients[radialPower] *
                    shiftedCoefficients[shiftedPower] *
                    binomial[shiftedPower][k] * shiftPowers[shiftedPower - k] *
                    signFactor *
                    (upperPowers[power + 1] - lowerPowers[power + 1]) /
                    static_cast<double>(power + 1);
                }
      return integral;
    }

    inline double
    smearedPairInteractionAnalyticUnequalRadius(const double radiusA,
                                                const double radiusB,
                                                const double separation)
    {
      const SmearedPairRadiusCoefficients &coefficientsA =
        smearedPairGetRadiusCoefficients(radiusA);
      const SmearedPairRadiusCoefficients &coefficientsB =
        smearedPairGetRadiusCoefficients(radiusB);

      if (separation <= 1.0e-8 * std::max(1.0, std::max(radiusA, radiusB)))
        {
          std::array<double, 17> radiusPowers;
          radiusPowers[0] = 1.0;
          for (dftfe::uInt i = 1; i < radiusPowers.size(); ++i)
            radiusPowers[i] = radiusPowers[i - 1] * radiusA;

          double centeredIntegral = 0.0;
          for (dftfe::uInt i = 0; i < coefficientsA.r2gCharge.size(); ++i)
            if (coefficientsA.r2gCharge[i] != 0.0)
              for (dftfe::uInt j = 0; j < coefficientsB.potential.size(); ++j)
                if (coefficientsB.potential[j] != 0.0)
                  {
                    const dftfe::uInt power = i + j;
                    centeredIntegral +=
                      coefficientsA.r2gCharge[i] * coefficientsB.potential[j] *
                      radiusPowers[power + 1] / static_cast<double>(power + 1);
                  }
          return 4.0 * M_PI * centeredIntegral;
        }

      std::vector<double> breakpoints;
      breakpoints.reserve(5);
      breakpoints.push_back(0.0);
      addUniqueBreakpoint(breakpoints,
                          std::abs(separation - radiusB),
                          0.0,
                          radiusA);
      addUniqueBreakpoint(breakpoints, separation, 0.0, radiusA);
      addUniqueBreakpoint(breakpoints, separation + radiusB, 0.0, radiusA);
      breakpoints.push_back(radiusA);
      std::sort(breakpoints.begin(), breakpoints.end());

      double primitiveIntegral = 0.0;
      for (dftfe::uInt iInterval = 0; iInterval + 1 < breakpoints.size();
           ++iInterval)
        {
          const double lower = breakpoints[iInterval];
          const double upper = breakpoints[iInterval + 1];
          if (upper <= lower)
            continue;

          const double                  midpoint = 0.5 * (lower + upper);
          const std::array<double, 10> &upperCoefficients =
            separation + midpoint < radiusB ? coefficientsB.primitiveInside :
                                              coefficientsB.primitiveOutside;
          const std::array<double, 10> &lowerCoefficients =
            std::abs(separation - midpoint) < radiusB ?
              coefficientsB.primitiveInside :
              coefficientsB.primitiveOutside;

          primitiveIntegral +=
            smearedPairShiftedPolynomialIntegral(coefficientsA.rgCharge,
                                                 upperCoefficients,
                                                 separation,
                                                 1.0,
                                                 lower,
                                                 upper);

          if (midpoint < separation)
            primitiveIntegral -=
              smearedPairShiftedPolynomialIntegral(coefficientsA.rgCharge,
                                                   lowerCoefficients,
                                                   separation,
                                                   -1.0,
                                                   lower,
                                                   upper);
          else
            primitiveIntegral -=
              smearedPairShiftedPolynomialIntegral(coefficientsA.rgCharge,
                                                   lowerCoefficients,
                                                   -separation,
                                                   1.0,
                                                   lower,
                                                   upper);
        }

      return 2.0 * M_PI * primitiveIntegral / separation;
    }

    inline double
    smearedPairInteractionDerAnalyticUnequalRadius(const double radiusA,
                                                   const double radiusB,
                                                   const double separation)
    {
      if (separation <= 1.0e-8 * std::max(1.0, std::max(radiusA, radiusB)))
        return 0.0;

      const SmearedPairRadiusCoefficients &coefficientsA =
        smearedPairGetRadiusCoefficients(radiusA);
      const SmearedPairRadiusCoefficients &coefficientsB =
        smearedPairGetRadiusCoefficients(radiusB);

      std::vector<double> breakpoints;
      breakpoints.reserve(5);
      breakpoints.push_back(0.0);
      addUniqueBreakpoint(breakpoints,
                          std::abs(separation - radiusB),
                          0.0,
                          radiusA);
      addUniqueBreakpoint(breakpoints, separation, 0.0, radiusA);
      addUniqueBreakpoint(breakpoints, separation + radiusB, 0.0, radiusA);
      breakpoints.push_back(radiusA);
      std::sort(breakpoints.begin(), breakpoints.end());

      double primitiveIntegral           = 0.0;
      double primitiveDerivativeIntegral = 0.0;
      for (dftfe::uInt iInterval = 0; iInterval + 1 < breakpoints.size();
           ++iInterval)
        {
          const double lower = breakpoints[iInterval];
          const double upper = breakpoints[iInterval + 1];
          if (upper <= lower)
            continue;

          const double midpoint    = 0.5 * (lower + upper);
          const bool   upperInside = separation + midpoint < radiusB;
          const bool   lowerInside = std::abs(separation - midpoint) < radiusB;
          const std::array<double, 10> &upperCoefficients =
            upperInside ? coefficientsB.primitiveInside :
                          coefficientsB.primitiveOutside;
          const std::array<double, 10> &lowerCoefficients =
            lowerInside ? coefficientsB.primitiveInside :
                          coefficientsB.primitiveOutside;
          const std::array<double, 10> &upperDerivativeCoefficients =
            upperInside ? coefficientsB.primitiveDerivativeInside :
                          coefficientsB.primitiveDerivativeOutside;
          const std::array<double, 10> &lowerDerivativeCoefficients =
            lowerInside ? coefficientsB.primitiveDerivativeInside :
                          coefficientsB.primitiveDerivativeOutside;

          primitiveIntegral +=
            smearedPairShiftedPolynomialIntegral(coefficientsA.rgCharge,
                                                 upperCoefficients,
                                                 separation,
                                                 1.0,
                                                 lower,
                                                 upper);
          primitiveDerivativeIntegral +=
            smearedPairShiftedPolynomialIntegral(coefficientsA.rgCharge,
                                                 upperDerivativeCoefficients,
                                                 separation,
                                                 1.0,
                                                 lower,
                                                 upper);

          if (midpoint < separation)
            {
              primitiveIntegral -=
                smearedPairShiftedPolynomialIntegral(coefficientsA.rgCharge,
                                                     lowerCoefficients,
                                                     separation,
                                                     -1.0,
                                                     lower,
                                                     upper);
              primitiveDerivativeIntegral -=
                smearedPairShiftedPolynomialIntegral(
                  coefficientsA.rgCharge,
                  lowerDerivativeCoefficients,
                  separation,
                  -1.0,
                  lower,
                  upper);
            }
          else
            {
              primitiveIntegral -=
                smearedPairShiftedPolynomialIntegral(coefficientsA.rgCharge,
                                                     lowerCoefficients,
                                                     -separation,
                                                     1.0,
                                                     lower,
                                                     upper);
              primitiveDerivativeIntegral +=
                smearedPairShiftedPolynomialIntegral(
                  coefficientsA.rgCharge,
                  lowerDerivativeCoefficients,
                  -separation,
                  1.0,
                  lower,
                  upper);
            }
        }

      return 2.0 * M_PI *
             (primitiveDerivativeIntegral / separation -
              primitiveIntegral / (separation * separation));
    }

    inline double
    smearedPairInteraction(const double radiusA,
                           const double radiusB,
                           const double separation)
    {
      if (radiusA <= 0.0 || radiusB <= 0.0)
        return 0.0;

      if (radiusA > radiusB)
        return smearedPairInteraction(radiusB, radiusA, separation);

      const double radiusScale = std::max(radiusA, radiusB);
      if (std::abs(radiusA - radiusB) <= 1.0e-13 * std::max(1.0, radiusScale))
        return smearedPairInteractionEqualRadius(0.5 * (radiusA + radiusB),
                                                 separation);

      if (separation >= radiusA + radiusB)
        return 1.0 / separation;

      return smearedPairInteractionAnalyticUnequalRadius(radiusA,
                                                         radiusB,
                                                         separation);
    }

    inline double
    smearedPairInteractionDer(const double radiusA,
                              const double radiusB,
                              const double separation)
    {
      if (radiusA <= 0.0 || radiusB <= 0.0)
        return 0.0;

      if (radiusA > radiusB)
        return smearedPairInteractionDer(radiusB, radiusA, separation);

      if (separation <= 1.0e-12)
        return 0.0;

      const double radiusScale = std::max(radiusA, radiusB);
      if (std::abs(radiusA - radiusB) <= 1.0e-13 * std::max(1.0, radiusScale))
        return smearedPairInteractionDerEqualRadius(0.5 * (radiusA + radiusB),
                                                    separation);

      if (separation >= radiusA + radiusB)
        return -1.0 / (separation * separation);

      return smearedPairInteractionDerAnalyticUnequalRadius(radiusA,
                                                            radiusB,
                                                            separation);
    }

    inline double
    smearedPairInteractionDifference(const double broadRadiusA,
                                     const double broadRadiusB,
                                     const double referenceRadiusA,
                                     const double referenceRadiusB,
                                     const double separation)
    {
      if (broadRadiusA <= 0.0 || broadRadiusB <= 0.0 ||
          referenceRadiusA <= 0.0 || referenceRadiusB <= 0.0)
        return 0.0;

      if (separation >= std::max(broadRadiusA + broadRadiusB,
                                 referenceRadiusA + referenceRadiusB))
        return 0.0;

      const double broad =
        smearedPairInteraction(broadRadiusA, broadRadiusB, separation);
      const double maxReferenceRadius =
        std::max(referenceRadiusA, referenceRadiusB);
      const double reference =
        std::abs(referenceRadiusA - referenceRadiusB) <=
            1.0e-13 * std::max(1.0, maxReferenceRadius) ?
          smearedPairInteractionEqualRadius(0.5 * (referenceRadiusA +
                                                   referenceRadiusB),
                                            separation) :
          smearedPairInteraction(referenceRadiusA,
                                 referenceRadiusB,
                                 separation);
      return broad - reference;
    }

    inline double
    smearedPairInteractionDerDifference(const double broadRadiusA,
                                        const double broadRadiusB,
                                        const double referenceRadiusA,
                                        const double referenceRadiusB,
                                        const double separation)
    {
      if (broadRadiusA <= 0.0 || broadRadiusB <= 0.0 ||
          referenceRadiusA <= 0.0 || referenceRadiusB <= 0.0)
        return 0.0;

      if (separation <= 1.0e-12)
        return 0.0;

      if (separation >= std::max(broadRadiusA + broadRadiusB,
                                 referenceRadiusA + referenceRadiusB))
        return 0.0;

      const double maxBroad = std::max(broadRadiusA, broadRadiusB);
      const double maxRef   = std::max(referenceRadiusA, referenceRadiusB);
      if (std::abs(broadRadiusA - referenceRadiusA) <=
            1.0e-13 * std::max(1.0, std::max(maxBroad, maxRef)) &&
          std::abs(broadRadiusB - referenceRadiusB) <=
            1.0e-13 * std::max(1.0, std::max(maxBroad, maxRef)))
        return 0.0;

      const double broad =
        smearedPairInteractionDer(broadRadiusA, broadRadiusB, separation);
      const double reference =
        std::abs(referenceRadiusA - referenceRadiusB) <=
            1.0e-13 * std::max(1.0, maxRef) ?
          smearedPairInteractionDerEqualRadius(0.5 * (referenceRadiusA +
                                                      referenceRadiusB),
                                               separation) :
          smearedPairInteractionDer(referenceRadiusA,
                                    referenceRadiusB,
                                    separation);
      return broad - reference;
    }



    inline std::vector<double>
    getFractionalCoordinates(
      const std::vector<double> &latticeVectorsFlattened,
      const std::vector<double> &coordWithRespectToCellCorner)
    {
      std::vector<double> latticeVectorsDup = latticeVectorsFlattened;
      std::vector<double> coordDup          = coordWithRespectToCellCorner;
      //
      // to get the fractionalCoords, solve a linear
      // system of equations
      //
      int N    = 3;
      int NRHS = 1;
      int LDA  = 3;
      int IPIV[3];
      int info;

      dgesv_(&N,
             &NRHS,
             &latticeVectorsDup[0],
             &LDA,
             &IPIV[0],
             &coordDup[0],
             &LDA,
             &info);
      AssertThrow(info == 0,
                  dealii::ExcMessage(
                    "LU solve in finding fractional coordinates failed."));
      return coordDup;
    }

    /** @brief Calculates value of composite generator
     *
     */
    double
    getCompositeGeneratorVal(const double rc,
                             const double r,
                             const double a0,
                             const double power);

    /** @brief Create bounding box around a sphere.
     *
     *  @param  sphere center
     *  @param  sphere radius
     *  @return bounding box
     */
    dealii::BoundingBox<3>
    createBoundingBoxForSphere(const dealii::Point<3> &center,
                               const double            sphereRadius);

    /** @brief Calculates partial occupancy of the atomic orbital using
     *  Fermi-Dirac smearing.
     *
     *  @param  eigenValue
     *  @param  fermiEnergy
     *  @param  kb Boltzmann constant
     *  @param  T smearing temperature
     *  @return double The partial occupancy of the orbital
     */
    double
    getPartialOccupancy(const double eigenValue,
                        const double fermiEnergy,
                        const double kb,
                        const double T);

    /** @brief Calculates the derivative of the partial occupancy of the atomic orbital
     * with respect to (x=eigenvalue-fermiEnergy) using Fermi-Dirac smearing.
     *
     *  @param  eigenValue
     *  @param  fermiEnergy
     *  @param  kb Boltzmann constant
     *  @param  T smearing temperature
     *  @return double The partial occupancy derivative of the orbital
     */
    double
    getPartialOccupancyDer(const double eigenValue,
                           const double fermiEnergy,
                           const double kb,
                           const double T);

    /** @brief Calculates cross product of two vectors
     *
     *  @param  a first vector
     *  @param  b second vector
     *  @param  crossProductVector cross product of a and b
     *  @return void
     */
    void
    cross_product(const std::vector<double> &a,
                  const std::vector<double> &b,
                  std::vector<double>       &crossProductVector);


    /** @brief Applies an affine transformation to the domain bounding vectors
     *
     *  @param  d_domainBoundingVectors the bounding vectors of the domain given as a 2d array
     *  @param  deformationGradient
     *  @return void.
     */
    void
    transformDomainBoundingVectors(
      std::vector<std::vector<double>>   &domainBoundingVectors,
      const dealii::Tensor<2, 3, double> &deformationGradient);

    /** @brief Writes to vtu file only from the lowest pool id
     *
     *  @param  dataOut  DataOut class object
     *  @param  mpiCommParent parent mpi communicator
     *  @param  mpiCommDomain mpi communicator of domain decomposition inside each pool
     *  @param  interpoolcomm  mpi communicator across k point pools
     *  @param  interBandGroupComm  mpi communicator across band groups
     *  @param  fileName
     */
    void
    writeDataVTUParallelLowestPoolId(const dealii::DoFHandler<3> &dofHandler,
                                     const dealii::DataOut<3>    &dataOut,
                                     const MPI_Comm              &mpiCommParent,
                                     const MPI_Comm              &mpiCommDomain,
                                     const MPI_Comm              &interpoolcomm,
                                     const MPI_Comm    &interBandGroupComm,
                                     const std::string &folderName,
                                     const std::string &fileName);

    /** @brief Create index vector which is used for band parallelization
     *
     *  @[in]param  interBandGroupComm  mpi communicator across band groups
     *  @[in]param  numBands
     *  @[out]param bandGroupLowHighPlusOneIndices
     */
    void
    createBandParallelizationIndices(
      const MPI_Comm           &interBandGroupComm,
      const dftfe::uInt         numBands,
      std::vector<dftfe::uInt> &bandGroupLowHighPlusOneIndices);

    void
    createKpointParallelizationIndices(
      const MPI_Comm          &interKptPoolComm,
      const dftfe::Int         numberIndices,
      std::vector<dftfe::Int> &kptGroupLowHighPlusOneIndices);


    /** @brief Wrapper to print current memory usage (prints only the maximum across mpiComm)
     * using PetscMemoryGetCurrentUsage
     *
     *  @[in]param mpiComm  mpi communicator across which the memory printing
     * will be synchronized
     *  @[in]param message message to be printed alongwith the memory usage
     */
    void
    printCurrentMemoryUsage(const MPI_Comm &mpiComm, const std::string message);

    /**
     * A class to split the given communicator into a number of pools
     */
    class Pool
    {
    public:
      Pool(const MPI_Comm   &mpi_communicator,
           const dftfe::uInt n_pools,
           const dftfe::Int  verbosity);

      /**
       * @brief get the communicator across the processor groups
       */
      MPI_Comm &
      get_interpool_comm();

      /**
       * @brief get the communicator associated with processor group
       */
      MPI_Comm &
      get_intrapool_comm();

    private:
      MPI_Comm interpoolcomm;
      MPI_Comm intrapoolcomm;
    };

    /// Exception handler for not implemented functionality
    DeclExceptionMsg(
      ExcNotImplementedYet,
      "This functionality is not implemented yet or not needed to be implemented.");

    /// Exception handler for DFT-FE internal error
    DeclExceptionMsg(ExcInternalError, "DFT-FE internal error.");
  } // namespace dftUtils

} // namespace dftfe
#endif
