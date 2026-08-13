// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// ---------------------------------------------------------------------

#ifndef DFTFE_ANALYTICSMEAREDLOADMANAGER_H
#define DFTFE_ANALYTICSMEAREDLOADMANAGER_H

#include <dftfe/headers.h>
#include <dftfe/MemorySpaceType.h>

#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <vector>

namespace dftfe
{
  class dftParameters;
  template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
  class oncvClass;

  /** Owns geometry-dependent data and analytic pair kernels for the
   *  ANALYTIC_SMEARED_LOAD nuclear-charge pathway.
   */
  template <dftfe::utils::MemorySpace memorySpace>
  class analyticSmearedLoadManager
  {
  public:
    /** Rebuild all ASL data after atom positions, cell, or mesh change. */
    void
    initialize(
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<dftfe::Int>          &imageIds,
      const std::vector<double>              &imageCharges,
      const std::vector<std::vector<double>> &imagePositions,
      const std::vector<std::vector<double>> &meshSizes,
      const std::vector<std::vector<double>> &domainBoundingVectors,
      const double                            minDist,
      const double                            pspCutOffTrunc,
      const std::shared_ptr<oncvClass<dataTypes::number, memorySpace>>
                                                &oncvClassPtr,
      const dealii::DoFHandler<3>             &dofHandlerPRefined,
      const dealii::MatrixFree<3, double>     &matrixFreeDataPRefined,
      const dftfe::uInt                        smearedChargeQuadratureId,
      const MPI_Comm                          &mpiCommunicator,
      const dftParameters                     &dftParams,
      dealii::ConditionalOStream              &pcout);

    const std::vector<double> &
    smearedChargeWidths() const;

    const std::vector<double> &
    smearedChargeScaling() const;

    /** Radial derivative of the broad-minus-reference pair correction. */
    double
    pairInteractionDerivativeDifference(const dftfe::uInt atomIdA,
                                        const dftfe::uInt atomIdB,
                                        const double separation) const;

    const std::vector<std::vector<double>> &
    localVselfs() const;

    const std::map<dealii::CellId, std::vector<double>> &
    bQuadValuesAllAtoms() const;

    std::map<dealii::CellId, std::vector<double>> &
    bQuadValuesAllAtoms();

    const std::map<dealii::CellId, std::vector<dftfe::uInt>> &
    bCellNonTrivialAtomIds() const;

    const std::map<dealii::CellId, std::vector<dftfe::uInt>> &
    bCellNonTrivialAtomImageIds() const;

  private:
    static inline double
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

    static inline double
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


      static inline void
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
        static constexpr std::size_t nCoefficients = 10;

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

        std::array<double, nCoefficients> rgCharge;
        std::array<double, nCoefficients> r2gCharge;
        std::array<double, nCoefficients> primitiveInside;
        std::array<double, nCoefficients> primitiveOutside;
        std::array<double, nCoefficients> primitiveDerivativeInside;
        std::array<double, nCoefficients> primitiveDerivativeOutside;
        std::array<double, nCoefficients> potential;
      };

      static inline double
      smearedPairShiftedPolynomialIntegral(
        const std::array<double, SmearedPairRadiusCoefficients::nCoefficients>
          &radialCoefficients,
        const std::array<double, SmearedPairRadiusCoefficients::nCoefficients>
          &shiftedCoefficients,
        const double shift,
        const double radialSign,
        const double lower,
        const double upper)
      {
        if (upper <= lower)
          return 0.0;

        static constexpr double
          binomial[SmearedPairRadiusCoefficients::nCoefficients]
                  [SmearedPairRadiusCoefficients::nCoefficients] = {
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

        constexpr std::size_t nPowers =
          2 * SmearedPairRadiusCoefficients::nCoefficients;
        std::array<double, nPowers> lowerPowers;
        std::array<double, nPowers> upperPowers;
        lowerPowers[0] = 1.0;
        upperPowers[0] = 1.0;
        for (dftfe::uInt i = 1; i < lowerPowers.size(); ++i)
          {
            lowerPowers[i] = lowerPowers[i - 1] * lower;
            upperPowers[i] = upperPowers[i - 1] * upper;
          }

        std::array<double, SmearedPairRadiusCoefficients::nCoefficients>
          shiftPowers;
        shiftPowers[0] = 1.0;
        for (dftfe::uInt i = 1; i < shiftPowers.size(); ++i)
          shiftPowers[i] = shiftPowers[i - 1] * shift;

        double integral = 0.0;
        for (dftfe::uInt radialPower = 0;
             radialPower < radialCoefficients.size();
             ++radialPower)
          if (radialCoefficients[radialPower] != 0.0)
            for (dftfe::uInt shiftedPower = 0;
                 shiftedPower < shiftedCoefficients.size();
                 ++shiftedPower)
              if (shiftedCoefficients[shiftedPower] != 0.0)
                for (dftfe::uInt k = 0; k <= shiftedPower; ++k)
                  {
                    const dftfe::uInt power = radialPower + k;
                    const double signFactor =
                      (k % 2 == 0) ? 1.0 : radialSign;
                    integral +=
                      radialCoefficients[radialPower] *
                      shiftedCoefficients[shiftedPower] *
                      binomial[shiftedPower][k] *
                      shiftPowers[shiftedPower - k] * signFactor *
                      (upperPowers[power + 1] - lowerPowers[power + 1]) /
                      static_cast<double>(power + 1);
                  }
        return integral;
      }

    static inline double
    smearedPairInteractionAnalyticUnequalRadius(
      const double radiusA,
      const double radiusB,
      const double separation,
      const SmearedPairRadiusCoefficients &coefficientsA,
      const SmearedPairRadiusCoefficients &coefficientsB)
    {
      if (separation <= 1.0e-8 * std::max(1.0, std::max(radiusA, radiusB)))
        {
          constexpr std::size_t nPowers =
            2 * SmearedPairRadiusCoefficients::nCoefficients;
          std::array<double, nPowers> radiusPowers;
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
      addUniqueBreakpoint(
        breakpoints, separation + radiusB, 0.0, radiusA);
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

          const double midpoint = 0.5 * (lower + upper);
          const auto  &upperCoefficients =
            separation + midpoint < radiusB ? coefficientsB.primitiveInside :
                                              coefficientsB.primitiveOutside;
          const auto &lowerCoefficients =
            std::abs(separation - midpoint) < radiusB ?
              coefficientsB.primitiveInside :
              coefficientsB.primitiveOutside;

          primitiveIntegral +=
            smearedPairShiftedPolynomialIntegral(
              coefficientsA.rgCharge,
              upperCoefficients,
              separation,
              1.0,
              lower,
              upper);

          if (midpoint < separation)
            primitiveIntegral -=
              smearedPairShiftedPolynomialIntegral(
                coefficientsA.rgCharge,
                lowerCoefficients,
                separation,
                -1.0,
                lower,
                upper);
          else
            primitiveIntegral -=
              smearedPairShiftedPolynomialIntegral(
                coefficientsA.rgCharge,
                lowerCoefficients,
                -separation,
                1.0,
                lower,
                upper);
        }

      return 2.0 * M_PI * primitiveIntegral / separation;
    }

    static inline double
    smearedPairInteractionAnalyticUnequalRadius(const double radiusA,
                                                const double radiusB,
                                                const double separation)
    {
      const SmearedPairRadiusCoefficients coefficientsA(radiusA);
      const SmearedPairRadiusCoefficients coefficientsB(radiusB);
      return smearedPairInteractionAnalyticUnequalRadius(radiusA,
                                                         radiusB,
                                                         separation,
                                                         coefficientsA,
                                                         coefficientsB);
    }

    static inline double
    smearedPairInteractionDerAnalyticUnequalRadius(
      const double radiusA,
      const double radiusB,
      const double separation,
      const SmearedPairRadiusCoefficients &coefficientsA,
      const SmearedPairRadiusCoefficients &coefficientsB)
    {
      if (separation <= 1.0e-8 * std::max(1.0, std::max(radiusA, radiusB)))
        return 0.0;

      std::vector<double> breakpoints;
      breakpoints.reserve(5);
      breakpoints.push_back(0.0);
      addUniqueBreakpoint(breakpoints,
                                    std::abs(separation - radiusB),
                                    0.0,
                                    radiusA);
      addUniqueBreakpoint(breakpoints, separation, 0.0, radiusA);
      addUniqueBreakpoint(
        breakpoints, separation + radiusB, 0.0, radiusA);
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
          const auto  &upperCoefficients =
            upperInside ? coefficientsB.primitiveInside :
                          coefficientsB.primitiveOutside;
          const auto &lowerCoefficients =
            lowerInside ? coefficientsB.primitiveInside :
                          coefficientsB.primitiveOutside;
          const auto &upperDerivativeCoefficients =
            upperInside ? coefficientsB.primitiveDerivativeInside :
                          coefficientsB.primitiveDerivativeOutside;
          const auto &lowerDerivativeCoefficients =
            lowerInside ? coefficientsB.primitiveDerivativeInside :
                          coefficientsB.primitiveDerivativeOutside;

          primitiveIntegral +=
            smearedPairShiftedPolynomialIntegral(
              coefficientsA.rgCharge,
              upperCoefficients,
              separation,
              1.0,
              lower,
              upper);
          primitiveDerivativeIntegral +=
            smearedPairShiftedPolynomialIntegral(
              coefficientsA.rgCharge,
              upperDerivativeCoefficients,
              separation,
              1.0,
              lower,
              upper);

          if (midpoint < separation)
            {
              primitiveIntegral -=
                smearedPairShiftedPolynomialIntegral(
                  coefficientsA.rgCharge,
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
                smearedPairShiftedPolynomialIntegral(
                  coefficientsA.rgCharge,
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

    static inline double
    smearedPairInteractionDerAnalyticUnequalRadius(const double radiusA,
                                                   const double radiusB,
                                                   const double separation)
    {
      const SmearedPairRadiusCoefficients coefficientsA(radiusA);
      const SmearedPairRadiusCoefficients coefficientsB(radiusB);
      return smearedPairInteractionDerAnalyticUnequalRadius(radiusA,
                                                            radiusB,
                                                            separation,
                                                            coefficientsA,
                                                            coefficientsB);
    }

    static inline double
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

      const SmearedPairRadiusCoefficients coefficientsA(radiusA);
      const SmearedPairRadiusCoefficients coefficientsB(radiusB);
      return smearedPairInteractionAnalyticUnequalRadius(radiusA,
                                                         radiusB,
                                                         separation,
                                                         coefficientsA,
                                                         coefficientsB);
    }

    static inline double
    smearedPairInteraction(
      const double                         radiusA,
      const double                         radiusB,
      const double                         separation,
      const SmearedPairRadiusCoefficients &coefficientsA,
      const SmearedPairRadiusCoefficients &coefficientsB)
    {
      if (radiusA <= 0.0 || radiusB <= 0.0)
        return 0.0;

      if (radiusA > radiusB)
        return smearedPairInteraction(radiusB,
                                      radiusA,
                                      separation,
                                      coefficientsB,
                                      coefficientsA);

      const double radiusScale = std::max(radiusA, radiusB);
      if (std::abs(radiusA - radiusB) <= 1.0e-13 * std::max(1.0, radiusScale))
        return smearedPairInteractionEqualRadius(0.5 * (radiusA + radiusB),
                                                 separation);

      if (separation >= radiusA + radiusB)
        return 1.0 / separation;

      return smearedPairInteractionAnalyticUnequalRadius(radiusA,
                                                         radiusB,
                                                         separation,
                                                         coefficientsA,
                                                         coefficientsB);
    }

    static inline double
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

      const SmearedPairRadiusCoefficients coefficientsA(radiusA);
      const SmearedPairRadiusCoefficients coefficientsB(radiusB);
      return smearedPairInteractionDerAnalyticUnequalRadius(radiusA,
                                                            radiusB,
                                                            separation,
                                                            coefficientsA,
                                                            coefficientsB);
    }

    static inline double
    smearedPairInteractionDer(
      const double                         radiusA,
      const double                         radiusB,
      const double                         separation,
      const SmearedPairRadiusCoefficients &coefficientsA,
      const SmearedPairRadiusCoefficients &coefficientsB)
    {
      if (radiusA <= 0.0 || radiusB <= 0.0)
        return 0.0;

      if (radiusA > radiusB)
        return smearedPairInteractionDer(radiusB,
                                         radiusA,
                                         separation,
                                         coefficientsB,
                                         coefficientsA);

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
                                                            separation,
                                                            coefficientsA,
                                                            coefficientsB);
    }

    static inline double
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
      const double reference = smearedPairInteraction(
        referenceRadiusA, referenceRadiusB, separation);
      return broad - reference;
    }

    static inline double
    smearedPairInteractionDifference(
      const double                         broadRadiusA,
      const double                         broadRadiusB,
      const double                         referenceRadiusA,
      const double                         referenceRadiusB,
      const double                         separation,
      const SmearedPairRadiusCoefficients &broadCoefficientsA,
      const SmearedPairRadiusCoefficients &broadCoefficientsB)
    {
      if (broadRadiusA <= 0.0 || broadRadiusB <= 0.0 ||
          referenceRadiusA <= 0.0 || referenceRadiusB <= 0.0)
        return 0.0;

      if (separation >= std::max(broadRadiusA + broadRadiusB,
                                 referenceRadiusA + referenceRadiusB))
        return 0.0;

      const double broad = smearedPairInteraction(broadRadiusA,
                                                  broadRadiusB,
                                                  separation,
                                                  broadCoefficientsA,
                                                  broadCoefficientsB);
      const double reference = smearedPairInteraction(
        referenceRadiusA, referenceRadiusB, separation);
      return broad - reference;
    }

    static inline double
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
      const double reference = smearedPairInteractionDer(
        referenceRadiusA, referenceRadiusB, separation);
      return broad - reference;
    }

    static inline double
    smearedPairInteractionDerDifference(
      const double                         broadRadiusA,
      const double                         broadRadiusB,
      const double                         referenceRadiusA,
      const double                         referenceRadiusB,
      const double                         separation,
      const SmearedPairRadiusCoefficients &broadCoefficientsA,
      const SmearedPairRadiusCoefficients &broadCoefficientsB)
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

      const double broad = smearedPairInteractionDer(broadRadiusA,
                                                     broadRadiusB,
                                                     separation,
                                                     broadCoefficientsA,
                                                     broadCoefficientsB);
      const double reference = smearedPairInteractionDer(
        referenceRadiusA, referenceRadiusB, separation);
      return broad - reference;
    }




    std::vector<double> d_smearedChargeWidths;
    std::vector<double> d_smearedChargeScaling;
    double              d_referenceSmearedChargeWidth = 0.0;
    std::vector<SmearedPairRadiusCoefficients>
      d_smearedPairRadiusCoefficients;
    std::vector<std::vector<double>> d_localVselfs;

    std::map<dealii::CellId, std::vector<dftfe::uInt>>
      d_physicalCandidatesByCell;
    std::map<dealii::CellId, std::vector<dftfe::uInt>>
      d_imageCandidatesByCell;
    std::map<dealii::CellId, std::vector<double>> d_bQuadValuesAllAtoms;
    std::map<dealii::CellId, std::vector<dftfe::uInt>>
      d_bCellNonTrivialAtomIds;
    std::map<dealii::CellId, std::vector<dftfe::uInt>>
      d_bCellNonTrivialAtomImageIds;
  };
} // namespace dftfe

#endif
