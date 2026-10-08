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

#ifndef atomicInitKernels_H_
#define atomicInitKernels_H_

#include <dftfe/MemorySpaceType.h>
#include <dftfe/TypeConfig.h>
#include <dftfe/splineDeviceEval.h>

namespace dftfe
{
  namespace atomicInitKernels
  {
    /**
     * @brief Closed form of dftUtils::smearedPot.
     */
    DFTFE_HOST_DEVICE_INLINE double
    smearedPot(const double r, const double rc)
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

    /**
     * @brief Radial atomic or core density, its gradient and its hessian at
     * one quadrature point for one center. Shared by the host and device
     * specializations of atomicRadialDensityQuadTotal and
     * atomicRadialDensityPairDerivatives, so the two routes evaluate one body.
     *
     *   d    = x_q - R,    r = |d|
     *   if floatingNuclearCharges and r < 1e-4:
     *        r = 1e-4, d_i = 1e-4 / sqrt(3)
     *   (v, v', v'') = r < splineCutoff ? spline(max(r, rMin)) : (0, 0, 0)
     *   *valueOut     = v
     *   gradOut[i]    = v' d_i / r
     *   hessOut[3i+j] = (v'' - v'/r) (d_i/r)(d_j/r) + delta_ij v'/r
     *
     * which matches AtomCenteredSphericalFunctionSpline::getDerivativeValue.
     * Each output is written only when non-null. Returns whether r lies
     * within the species cutoff.
     */
    DFTFE_HOST_DEVICE_INLINE bool
    radialDensityQuadPoint(const double      *quadPoint,
                           const double      *center,
                           const dftfe::uInt  splineId,
                           const bool         floatingNuclearCharges,
                           const double      *knotXAll,
                           const double      *knotYAll,
                           const double      *coefBAll,
                           const double      *coefCAll,
                           const double      *coefDAll,
                           const dftfe::uInt *splineOffsets,
                           const dftfe::uInt *splineNKnots,
                           const double      *splineGPParams,
                           const double      *splineCutoff,
                           const double      *splineRMin,
                           double            *valueOut,
                           double            *gradOut,
                           double            *hessOut)
    {
      double diff[3];
      diff[0] = quadPoint[0] - center[0];
      diff[1] = quadPoint[1] - center[1];
      diff[2] = quadPoint[2] - center[2];
      double r =
        std::sqrt(diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]);

      if (floatingNuclearCharges && r < 1.0e-4)
        {
          r       = 1.0e-4;
          diff[0] = (1.0e-4) / std::sqrt(3.0);
          diff[1] = (1.0e-4) / std::sqrt(3.0);
          diff[2] = (1.0e-4) / std::sqrt(3.0);
        }

      const double cutoff = splineCutoff[splineId];
      double       v      = 0.0;
      double       d1     = 0.0;
      double       d2     = 0.0;
      if (r < cutoff)
        {
          const splineEval::SplineTableView t =
            splineEval::splineView(knotXAll,
                                   knotYAll,
                                   coefBAll,
                                   coefCAll,
                                   coefDAll,
                                   splineOffsets,
                                   splineNKnots,
                                   splineGPParams,
                                   splineId);
          const double rMin  = splineRMin[splineId];
          const double rEval = r <= rMin ? rMin : r;
          v                  = splineEval::splineEvalValue(t, rEval);
          d1                 = splineEval::splineEvalDerivative(t, rEval, 1);
          if (hessOut != nullptr)
            d2 = splineEval::splineEvalDerivative(t, rEval, 2);
        }

      if (valueOut != nullptr)
        *valueOut = v;

      if (gradOut != nullptr)
        for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
          gradOut[iDim] = d1 * diff[iDim] / r;

      if (hessOut != nullptr)
        {
          const double radial = d2 - d1 / r;
          for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
            for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
              {
                double temp = radial * (diff[iDim] / r) * (diff[jDim] / r);
                if (iDim == jDim)
                  temp += d1 / r;
                hessOut[3 * iDim + jDim] = temp;
              }
        }

      return r <= cutoff;
    }

    /**
     * @brief Free-atom density, magnetization and, when gradRhoOut is
     * non-null, their gradients at one point -- the quadrature-point initial
     * guess of initRho. Shared by the host and device specializations of
     * atomicDensityQuadSuperposition.
     *
     * The centers are looped in the order given. Each contributes, with v and
     * v' from radialDensityQuadPoint without the floating-charge clamp,
     *
     *   t = centerRhoFactor v,           rho += t,      m_k += centerMag_k t
     *
     * and, unless floatingNuclearCharges and r < 1e-3,
     *
     *   g = centerRhoFactor v' d/r,      gradRho += g,  gradM_k += centerMag_k
     * g
     *
     * rhoOut receives |rho| and gradRhoOut gradRho where |rho| > 1e-8, zero
     * elsewhere. With applyConstraintMag, magZOut is totMagnetization |rho|
     * and gradMagZOut totMagnetization times the unmasked gradRho. The magZ
     * outputs are written for numMagComponents >= 1, magY and magX for
     * numMagComponents == 3. Outputs are indexed by pointIndex.
     */
    DFTFE_HOST_DEVICE_INLINE void
    densitySuperpositionQuadPoint(const dftfe::uInt  pointIndex,
                                  const double      *pointCoords,
                                  const dftfe::uInt  numCenters,
                                  const dftfe::uInt  numMagComponents,
                                  const bool         floatingNuclearCharges,
                                  const bool         applyConstraintMag,
                                  const double       totMagnetization,
                                  const double      *centerCoords,
                                  const dftfe::uInt *centerSplineId,
                                  const double      *centerRhoFactor,
                                  const double      *centerMagZFactor,
                                  const double      *centerMagYFactor,
                                  const double      *centerMagXFactor,
                                  const double      *knotXAll,
                                  const double      *knotYAll,
                                  const double      *coefBAll,
                                  const double      *coefCAll,
                                  const double      *coefDAll,
                                  const dftfe::uInt *splineOffsets,
                                  const dftfe::uInt *splineNKnots,
                                  const double      *splineGPParams,
                                  const double      *splineCutoff,
                                  const double      *splineRMin,
                                  double            *rhoOut,
                                  double            *magZOut,
                                  double            *magYOut,
                                  double            *magXOut,
                                  double            *gradRhoOut,
                                  double            *gradMagZOut,
                                  double            *gradMagYOut,
                                  double            *gradMagXOut)
    {
      const double *point = pointCoords + pointIndex * 3;
      double        rho   = 0.0;
      double        magZ  = 0.0;
      double        magY  = 0.0;
      double        magX  = 0.0;
      double        gradRho[3];
      double        gradMagZ[3];
      double        gradMagY[3];
      double        gradMagX[3];
      for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
        {
          gradRho[iDim]  = 0.0;
          gradMagZ[iDim] = 0.0;
          gradMagY[iDim] = 0.0;
          gradMagX[iDim] = 0.0;
        }

      for (dftfe::uInt iCenter = 0; iCenter < numCenters; ++iCenter)
        {
          const double *center   = centerCoords + iCenter * 3;
          bool          withGrad = gradRhoOut != nullptr;
          if (withGrad && floatingNuclearCharges)
            {
              const double dx = point[0] - center[0];
              const double dy = point[1] - center[1];
              const double dz = point[2] - center[2];
              withGrad = std::sqrt(dx * dx + dy * dy + dz * dz) >= 1.0e-3;
            }
          double v;
          double g[3];
          radialDensityQuadPoint(point,
                                 center,
                                 centerSplineId[iCenter],
                                 false,
                                 knotXAll,
                                 knotYAll,
                                 coefBAll,
                                 coefCAll,
                                 coefDAll,
                                 splineOffsets,
                                 splineNKnots,
                                 splineGPParams,
                                 splineCutoff,
                                 splineRMin,
                                 &v,
                                 withGrad ? g : nullptr,
                                 nullptr);
          const double t = centerRhoFactor[iCenter] * v;
          rho += t;
          magZ += centerMagZFactor[iCenter] * t;
          magY += centerMagYFactor[iCenter] * t;
          magX += centerMagXFactor[iCenter] * t;
          if (withGrad)
            for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
              {
                const double gi = centerRhoFactor[iCenter] * g[iDim];
                gradRho[iDim] += gi;
                gradMagZ[iDim] += centerMagZFactor[iCenter] * gi;
                gradMagY[iDim] += centerMagYFactor[iCenter] * gi;
                gradMagX[iDim] += centerMagXFactor[iCenter] * gi;
              }
        }

      const double rhoAbs = rho < 0.0 ? -rho : rho;
      rhoOut[pointIndex]  = rhoAbs;
      if (numMagComponents >= 1)
        magZOut[pointIndex] =
          applyConstraintMag ? totMagnetization * rhoAbs : magZ;
      if (numMagComponents == 3)
        {
          magYOut[pointIndex] = magY;
          magXOut[pointIndex] = magX;
        }

      if (gradRhoOut != nullptr)
        {
          const double rhoMask = rhoAbs > 1.0e-8 ? 1.0 : 0.0;
          for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
            {
              gradRhoOut[pointIndex * 3 + iDim] = rhoMask * gradRho[iDim];
              if (numMagComponents >= 1)
                gradMagZOut[pointIndex * 3 + iDim] =
                  applyConstraintMag ? totMagnetization * gradRho[iDim] :
                                       gradMagZ[iDim];
              if (numMagComponents == 3)
                {
                  gradMagYOut[pointIndex * 3 + iDim] = gradMagY[iDim];
                  gradMagXOut[pointIndex * 3 + iDim] = gradMagX[iDim];
                }
            }
        }
    }

    /**
     * @brief Free-atom density and magnetization superposed at every point.
     *
     * For each center with r <= centerCutoff, adds
     * centerRhoFactor * spline(r) to rhoOut and the same times
     * centerMag*Factor to the magnetization outputs. rhoOut receives the
     * absolute value. magZOut is written for numMagComponents >= 1,
     * magYOut/magXOut additionally for numMagComponents == 3.
     */
    template <dftfe::utils::MemorySpace memorySpace>
    void
    atomicDensitySuperposition(const dftfe::uInt  numPoints,
                               const dftfe::uInt  numCenters,
                               const dftfe::uInt  numMagComponents,
                               const double      *pointCoords,
                               const double      *centerCoords,
                               const dftfe::uInt *centerSplineId,
                               const double      *centerCutoff,
                               const double      *centerRhoFactor,
                               const double      *centerMagZFactor,
                               const double      *centerMagYFactor,
                               const double      *centerMagXFactor,
                               const double      *knotXAll,
                               const double      *knotYAll,
                               const double      *coefBAll,
                               const double      *coefCAll,
                               const double      *coefDAll,
                               const dftfe::uInt *splineOffsets,
                               const dftfe::uInt *splineNKnots,
                               const double      *splineGPParams,
                               const double      *splineCutoff,
                               const double      *splineRMin,
                               double            *rhoOut,
                               double            *magZOut,
                               double            *magYOut,
                               double            *magXOut);

    /**
     * @brief Atomic-orbital initial guess: radial spline times real spherical
     * harmonic, superposed over each orbital's centers.
     *
     * valueOut is indexed [iPoint * numOrbitals + iOrbital]; contributions
     * are scaled by 1/sqrt(2) for the noncollinear case.
     */
    template <dftfe::utils::MemorySpace memorySpace>
    void
    atomicWavefunctionSuperposition(const dftfe::uInt  numPoints,
                                    const dftfe::uInt  numOrbitals,
                                    const bool         isNoncolin,
                                    const double       truncationRadius,
                                    const double      *pointCoords,
                                    const dftfe::uInt *orbitalCenterStart,
                                    const double      *centerCoords,
                                    const dftfe::Int  *orbitalL,
                                    const dftfe::Int  *orbitalM,
                                    const dftfe::uInt *orbitalSplineId,
                                    const double      *knotXAll,
                                    const double      *knotYAll,
                                    const double      *coefBAll,
                                    const double      *coefCAll,
                                    const double      *coefDAll,
                                    const dftfe::uInt *splineOffsets,
                                    const dftfe::uInt *splineNKnots,
                                    const double      *splineGPParams,
                                    double            *valueOut);
    /**
     * @brief Local pseudopotential value at every quadrature point of every
     * (atom-or-image, cell) pair.
     *
     * Reproduces the host loop of initLocalPseudoPotential step 3:
     *
     *   r  = |x_q - R_pair|
     *   v  = r <  splineCutoff[species] ? spline(max(r, rMin))
     *                                   : -Z_pair / r
     *
     * The spline branch matches AtomCenteredSphericalFunctionSpline::
     * getRadialValue, which returns 0 at r >= cutoff and clamps r to rMin
     * below; the host takes the analytic tail for r > cutoff and, at exactly
     * r == cutoff, the spline path returning 0. pairHasSupport[iPair] is set
     * when any quadrature point of the pair lies within cutOffForPsp, which
     * is what decides whether the host stores the pair at all.
     */
    template <dftfe::utils::MemorySpace memorySpace>
    void
    localPseudoPotentialValues(const dftfe::uInt  numPairs,
                               const dftfe::uInt  numQuadPoints,
                               const double       cutOffForPsp,
                               const double      *quadCoords,
                               const double      *pairCenters,
                               const double      *pairCharges,
                               const dftfe::uInt *pairCellSlot,
                               const dftfe::uInt *pairSplineId,
                               const double      *knotXAll,
                               const double      *knotYAll,
                               const double      *coefBAll,
                               const double      *coefCAll,
                               const double      *coefDAll,
                               const dftfe::uInt *splineOffsets,
                               const dftfe::uInt *splineNKnots,
                               const double      *splineGPParams,
                               const double      *splineCutoff,
                               const double      *splineRMin,
                               double            *valueOut,
                               dftfe::uInt       *pairHasSupport);
    /**
     * @brief Total local pseudopotential accumulated over every center, at
     * every quadrature point of every cell -- step 2 of
     * initLocalPseudoPotential.
     *
     * One thread per (cell, quadrature point); the loop over the centers is
     * inside the kernel, so nothing per-(center, point) is ever stored. For
     * each center:
     *
     *   r  = |x_q - R_i|
     *   v  = (r <= maxTail && r <= tailCutoff_i) ? spline_i(max(r, rMin))
     *                                           : -Z_i / r
     *   if analyticSmearedCharge:
     *      v -= (r > w_i) ? -Z_i/r : -Z_i * smearedPot(r, w_i)
     *
     * reproducing the host branch structure and its exact arithmetic, and
     * accumulating v into the output for that quadrature point.
     */
    template <dftfe::utils::MemorySpace memorySpace>
    void
    localPseudoPotentialTotal(const dftfe::uInt  numCells,
                              const dftfe::uInt  numQuadPoints,
                              const dftfe::uInt  numCenters,
                              const bool         analyticSmearedCharge,
                              const double       maxTail,
                              const double      *quadCoords,
                              const double      *centerCoords,
                              const double      *centerCharges,
                              const double      *centerTailCutoff,
                              const double      *centerSmearedWidth,
                              const dftfe::uInt *centerSplineId,
                              const double      *knotXAll,
                              const double      *knotYAll,
                              const double      *coefBAll,
                              const double      *coefCAll,
                              const double      *coefDAll,
                              const dftfe::uInt *splineOffsets,
                              const dftfe::uInt *splineNKnots,
                              const double      *splineGPParams,
                              const double      *splineCutoff,
                              const double      *splineRMin,
                              double            *valueOut);


    /**
     * @brief Atomic or core density and its gradient, summed over every
     * center, at every quadrature point of every cell -- the totals of
     * initAtomicRho and initCoreRho.
     *
     * One thread per (cell, quadrature point); the loop over the centers is
     * inside the kernel, in the order given, so nothing per-(center, point)
     * is stored. Each center contributes radialDensityQuadPoint; centers
     * beyond the cutoff contribute exactly zero, so the sum equals the
     * host's over the centers it does not reject.
     */
    template <dftfe::utils::MemorySpace memorySpace>
    void
    atomicRadialDensityQuadTotal(const dftfe::uInt  numCells,
                                 const dftfe::uInt  numQuadPoints,
                                 const dftfe::uInt  numCenters,
                                 const bool         floatingNuclearCharges,
                                 const double      *quadCoords,
                                 const double      *centerCoords,
                                 const dftfe::uInt *centerSplineId,
                                 const double      *knotXAll,
                                 const double      *knotYAll,
                                 const double      *coefBAll,
                                 const double      *coefCAll,
                                 const double      *coefDAll,
                                 const dftfe::uInt *splineOffsets,
                                 const dftfe::uInt *splineNKnots,
                                 const double      *splineGPParams,
                                 const double      *splineCutoff,
                                 const double      *splineRMin,
                                 double            *valueOut,
                                 double            *gradOut);

    /**
     * @brief Core-density gradient and, when hessOut is non-null, hessian at
     * every quadrature point of every (atom-or-image, cell) pair -- the
     * per-atom maps of initCoreRho read by the force and stress.
     *
     * One thread or iteration per (pair, quadrature point), each evaluating
     * radialDensityQuadPoint. pairHasSupport[iPair] is set when any
     * quadrature point of the pair lies within the species cutoff, which is
     * what decides whether the caller stores the pair at all.
     */
    template <dftfe::utils::MemorySpace memorySpace>
    void
    atomicRadialDensityPairDerivatives(const dftfe::uInt numPairs,
                                       const dftfe::uInt numQuadPoints,
                                       const bool        floatingNuclearCharges,
                                       const double     *quadCoords,
                                       const double     *pairCenters,
                                       const dftfe::uInt *pairCellSlot,
                                       const dftfe::uInt *pairSplineId,
                                       const double      *knotXAll,
                                       const double      *knotYAll,
                                       const double      *coefBAll,
                                       const double      *coefCAll,
                                       const double      *coefDAll,
                                       const dftfe::uInt *splineOffsets,
                                       const dftfe::uInt *splineNKnots,
                                       const double      *splineGPParams,
                                       const double      *splineCutoff,
                                       const double      *splineRMin,
                                       double            *gradOut,
                                       double            *hessOut,
                                       dftfe::uInt       *pairHasSupport);

    /**
     * @brief densitySuperpositionQuadPoint at every point -- the
     * quadrature-point initial guess of initRho, used when the density is
     * mixed at quadrature points.
     */
    template <dftfe::utils::MemorySpace memorySpace>
    void
    atomicDensityQuadSuperposition(const dftfe::uInt  numPoints,
                                   const dftfe::uInt  numCenters,
                                   const dftfe::uInt  numMagComponents,
                                   const bool         floatingNuclearCharges,
                                   const bool         applyConstraintMag,
                                   const double       totMagnetization,
                                   const double      *pointCoords,
                                   const double      *centerCoords,
                                   const dftfe::uInt *centerSplineId,
                                   const double      *centerRhoFactor,
                                   const double      *centerMagZFactor,
                                   const double      *centerMagYFactor,
                                   const double      *centerMagXFactor,
                                   const double      *knotXAll,
                                   const double      *knotYAll,
                                   const double      *coefBAll,
                                   const double      *coefCAll,
                                   const double      *coefDAll,
                                   const dftfe::uInt *splineOffsets,
                                   const dftfe::uInt *splineNKnots,
                                   const double      *splineGPParams,
                                   const double      *splineCutoff,
                                   const double      *splineRMin,
                                   double            *rhoOut,
                                   double            *magZOut,
                                   double            *magYOut,
                                   double            *magXOut,
                                   double            *gradRhoOut,
                                   double            *gradMagZOut,
                                   double            *gradMagYOut,
                                   double            *gradMagXOut);

    // The HOST and DEVICE specialisations are defined in
    // src/dft/atomicInitKernels.cc and src/dft/atomicInitDeviceKernels.cc.
    // They must be declared before any use, otherwise a caller would
    // trigger implicit instantiation of the primary and the program would
    // be ill-formed.
    template <>
    void
    atomicDensitySuperposition<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt  numPoints,
      const dftfe::uInt  numCenters,
      const dftfe::uInt  numMagComponents,
      const double      *pointCoords,
      const double      *centerCoords,
      const dftfe::uInt *centerSplineId,
      const double      *centerCutoff,
      const double      *centerRhoFactor,
      const double      *centerMagZFactor,
      const double      *centerMagYFactor,
      const double      *centerMagXFactor,
      const double      *knotXAll,
      const double      *knotYAll,
      const double      *coefBAll,
      const double      *coefCAll,
      const double      *coefDAll,
      const dftfe::uInt *splineOffsets,
      const dftfe::uInt *splineNKnots,
      const double      *splineGPParams,
      const double      *splineCutoff,
      const double      *splineRMin,
      double            *rhoOut,
      double            *magZOut,
      double            *magYOut,
      double            *magXOut);

    template <>
    void
    atomicDensitySuperposition<dftfe::utils::MemorySpace::DEVICE>(
      const dftfe::uInt  numPoints,
      const dftfe::uInt  numCenters,
      const dftfe::uInt  numMagComponents,
      const double      *pointCoords,
      const double      *centerCoords,
      const dftfe::uInt *centerSplineId,
      const double      *centerCutoff,
      const double      *centerRhoFactor,
      const double      *centerMagZFactor,
      const double      *centerMagYFactor,
      const double      *centerMagXFactor,
      const double      *knotXAll,
      const double      *knotYAll,
      const double      *coefBAll,
      const double      *coefCAll,
      const double      *coefDAll,
      const dftfe::uInt *splineOffsets,
      const dftfe::uInt *splineNKnots,
      const double      *splineGPParams,
      const double      *splineCutoff,
      const double      *splineRMin,
      double            *rhoOut,
      double            *magZOut,
      double            *magYOut,
      double            *magXOut);

    template <>
    void
    atomicWavefunctionSuperposition<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt  numPoints,
      const dftfe::uInt  numOrbitals,
      const bool         isNoncolin,
      const double       truncationRadius,
      const double      *pointCoords,
      const dftfe::uInt *orbitalCenterStart,
      const double      *centerCoords,
      const dftfe::Int  *orbitalL,
      const dftfe::Int  *orbitalM,
      const dftfe::uInt *orbitalSplineId,
      const double      *knotXAll,
      const double      *knotYAll,
      const double      *coefBAll,
      const double      *coefCAll,
      const double      *coefDAll,
      const dftfe::uInt *splineOffsets,
      const dftfe::uInt *splineNKnots,
      const double      *splineGPParams,
      double            *valueOut);

    template <>
    void
    atomicWavefunctionSuperposition<dftfe::utils::MemorySpace::DEVICE>(
      const dftfe::uInt  numPoints,
      const dftfe::uInt  numOrbitals,
      const bool         isNoncolin,
      const double       truncationRadius,
      const double      *pointCoords,
      const dftfe::uInt *orbitalCenterStart,
      const double      *centerCoords,
      const dftfe::Int  *orbitalL,
      const dftfe::Int  *orbitalM,
      const dftfe::uInt *orbitalSplineId,
      const double      *knotXAll,
      const double      *knotYAll,
      const double      *coefBAll,
      const double      *coefCAll,
      const double      *coefDAll,
      const dftfe::uInt *splineOffsets,
      const dftfe::uInt *splineNKnots,
      const double      *splineGPParams,
      double            *valueOut);

    template <>
    void
    atomicRadialDensityQuadTotal<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt  numCells,
      const dftfe::uInt  numQuadPoints,
      const dftfe::uInt  numCenters,
      const bool         floatingNuclearCharges,
      const double      *quadCoords,
      const double      *centerCoords,
      const dftfe::uInt *centerSplineId,
      const double      *knotXAll,
      const double      *knotYAll,
      const double      *coefBAll,
      const double      *coefCAll,
      const double      *coefDAll,
      const dftfe::uInt *splineOffsets,
      const dftfe::uInt *splineNKnots,
      const double      *splineGPParams,
      const double      *splineCutoff,
      const double      *splineRMin,
      double            *valueOut,
      double            *gradOut);

    template <>
    void
    atomicRadialDensityPairDerivatives<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt  numPairs,
      const dftfe::uInt  numQuadPoints,
      const bool         floatingNuclearCharges,
      const double      *quadCoords,
      const double      *pairCenters,
      const dftfe::uInt *pairCellSlot,
      const dftfe::uInt *pairSplineId,
      const double      *knotXAll,
      const double      *knotYAll,
      const double      *coefBAll,
      const double      *coefCAll,
      const double      *coefDAll,
      const dftfe::uInt *splineOffsets,
      const dftfe::uInt *splineNKnots,
      const double      *splineGPParams,
      const double      *splineCutoff,
      const double      *splineRMin,
      double            *gradOut,
      double            *hessOut,
      dftfe::uInt       *pairHasSupport);

    template <>
    void
    atomicRadialDensityQuadTotal<dftfe::utils::MemorySpace::DEVICE>(
      const dftfe::uInt  numCells,
      const dftfe::uInt  numQuadPoints,
      const dftfe::uInt  numCenters,
      const bool         floatingNuclearCharges,
      const double      *quadCoords,
      const double      *centerCoords,
      const dftfe::uInt *centerSplineId,
      const double      *knotXAll,
      const double      *knotYAll,
      const double      *coefBAll,
      const double      *coefCAll,
      const double      *coefDAll,
      const dftfe::uInt *splineOffsets,
      const dftfe::uInt *splineNKnots,
      const double      *splineGPParams,
      const double      *splineCutoff,
      const double      *splineRMin,
      double            *valueOut,
      double            *gradOut);

    template <>
    void
    atomicRadialDensityPairDerivatives<dftfe::utils::MemorySpace::DEVICE>(
      const dftfe::uInt  numPairs,
      const dftfe::uInt  numQuadPoints,
      const bool         floatingNuclearCharges,
      const double      *quadCoords,
      const double      *pairCenters,
      const dftfe::uInt *pairCellSlot,
      const dftfe::uInt *pairSplineId,
      const double      *knotXAll,
      const double      *knotYAll,
      const double      *coefBAll,
      const double      *coefCAll,
      const double      *coefDAll,
      const dftfe::uInt *splineOffsets,
      const dftfe::uInt *splineNKnots,
      const double      *splineGPParams,
      const double      *splineCutoff,
      const double      *splineRMin,
      double            *gradOut,
      double            *hessOut,
      dftfe::uInt       *pairHasSupport);

    template <>
    void
    atomicDensityQuadSuperposition<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt  numPoints,
      const dftfe::uInt  numCenters,
      const dftfe::uInt  numMagComponents,
      const bool         floatingNuclearCharges,
      const bool         applyConstraintMag,
      const double       totMagnetization,
      const double      *pointCoords,
      const double      *centerCoords,
      const dftfe::uInt *centerSplineId,
      const double      *centerRhoFactor,
      const double      *centerMagZFactor,
      const double      *centerMagYFactor,
      const double      *centerMagXFactor,
      const double      *knotXAll,
      const double      *knotYAll,
      const double      *coefBAll,
      const double      *coefCAll,
      const double      *coefDAll,
      const dftfe::uInt *splineOffsets,
      const dftfe::uInt *splineNKnots,
      const double      *splineGPParams,
      const double      *splineCutoff,
      const double      *splineRMin,
      double            *rhoOut,
      double            *magZOut,
      double            *magYOut,
      double            *magXOut,
      double            *gradRhoOut,
      double            *gradMagZOut,
      double            *gradMagYOut,
      double            *gradMagXOut);

    template <>
    void
    atomicDensityQuadSuperposition<dftfe::utils::MemorySpace::DEVICE>(
      const dftfe::uInt  numPoints,
      const dftfe::uInt  numCenters,
      const dftfe::uInt  numMagComponents,
      const bool         floatingNuclearCharges,
      const bool         applyConstraintMag,
      const double       totMagnetization,
      const double      *pointCoords,
      const double      *centerCoords,
      const dftfe::uInt *centerSplineId,
      const double      *centerRhoFactor,
      const double      *centerMagZFactor,
      const double      *centerMagYFactor,
      const double      *centerMagXFactor,
      const double      *knotXAll,
      const double      *knotYAll,
      const double      *coefBAll,
      const double      *coefCAll,
      const double      *coefDAll,
      const dftfe::uInt *splineOffsets,
      const dftfe::uInt *splineNKnots,
      const double      *splineGPParams,
      const double      *splineCutoff,
      const double      *splineRMin,
      double            *rhoOut,
      double            *magZOut,
      double            *magYOut,
      double            *magXOut,
      double            *gradRhoOut,
      double            *gradMagZOut,
      double            *gradMagYOut,
      double            *gradMagXOut);

    template <>
    void
    localPseudoPotentialValues<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt  numPairs,
      const dftfe::uInt  numQuadPoints,
      const double       cutOffForPsp,
      const double      *quadCoords,
      const double      *pairCenters,
      const double      *pairCharges,
      const dftfe::uInt *pairCellSlot,
      const dftfe::uInt *pairSplineId,
      const double      *knotXAll,
      const double      *knotYAll,
      const double      *coefBAll,
      const double      *coefCAll,
      const double      *coefDAll,
      const dftfe::uInt *splineOffsets,
      const dftfe::uInt *splineNKnots,
      const double      *splineGPParams,
      const double      *splineCutoff,
      const double      *splineRMin,
      double            *valueOut,
      dftfe::uInt       *pairHasSupport);

    template <>
    void
    localPseudoPotentialValues<dftfe::utils::MemorySpace::DEVICE>(
      const dftfe::uInt  numPairs,
      const dftfe::uInt  numQuadPoints,
      const double       cutOffForPsp,
      const double      *quadCoords,
      const double      *pairCenters,
      const double      *pairCharges,
      const dftfe::uInt *pairCellSlot,
      const dftfe::uInt *pairSplineId,
      const double      *knotXAll,
      const double      *knotYAll,
      const double      *coefBAll,
      const double      *coefCAll,
      const double      *coefDAll,
      const dftfe::uInt *splineOffsets,
      const dftfe::uInt *splineNKnots,
      const double      *splineGPParams,
      const double      *splineCutoff,
      const double      *splineRMin,
      double            *valueOut,
      dftfe::uInt       *pairHasSupport);

    template <>
    void
    localPseudoPotentialTotal<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt  numCells,
      const dftfe::uInt  numQuadPoints,
      const dftfe::uInt  numCenters,
      const bool         analyticSmearedCharge,
      const double       maxTail,
      const double      *quadCoords,
      const double      *centerCoords,
      const double      *centerCharges,
      const double      *centerTailCutoff,
      const double      *centerSmearedWidth,
      const dftfe::uInt *centerSplineId,
      const double      *knotXAll,
      const double      *knotYAll,
      const double      *coefBAll,
      const double      *coefCAll,
      const double      *coefDAll,
      const dftfe::uInt *splineOffsets,
      const dftfe::uInt *splineNKnots,
      const double      *splineGPParams,
      const double      *splineCutoff,
      const double      *splineRMin,
      double            *valueOut);

    template <>
    void
    localPseudoPotentialTotal<dftfe::utils::MemorySpace::DEVICE>(
      const dftfe::uInt  numCells,
      const dftfe::uInt  numQuadPoints,
      const dftfe::uInt  numCenters,
      const bool         analyticSmearedCharge,
      const double       maxTail,
      const double      *quadCoords,
      const double      *centerCoords,
      const double      *centerCharges,
      const double      *centerTailCutoff,
      const double      *centerSmearedWidth,
      const dftfe::uInt *centerSplineId,
      const double      *knotXAll,
      const double      *knotYAll,
      const double      *coefBAll,
      const double      *coefCAll,
      const double      *coefDAll,
      const dftfe::uInt *splineOffsets,
      const dftfe::uInt *splineNKnots,
      const double      *splineGPParams,
      const double      *splineCutoff,
      const double      *splineRMin,
      double            *valueOut);

  } // namespace atomicInitKernels
} // namespace dftfe
#endif
