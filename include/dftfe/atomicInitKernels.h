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
