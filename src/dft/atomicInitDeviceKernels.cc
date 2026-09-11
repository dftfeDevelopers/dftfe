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

#include <dftfe/atomicInitKernels.h>
#include <dftfe/AtomCenteredSphericalFunctionDeviceUtils.h>
#include <dftfe/BLASWrapper.h>
#include <dftfe/DeviceAPICalls.h>
#include <dftfe/DeviceDataTypeOverloads.h>
#include <dftfe/DeviceTypeConfig.h>
#include <dftfe/DeviceKernelLauncherHelpers.h>

namespace dftfe
{
  namespace atomicInitKernels
  {
    DFTFE_CREATE_KERNEL(
      void,
      localPseudoPotentialValuesKernel,
      {
        const dftfe::uInt numberEntries = numPairs * numQuadPoints;
        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            const dftfe::uInt iPair    = index / numQuadPoints;
            const dftfe::uInt iQuad    = index % numQuadPoints;
            const dftfe::uInt cellSlot = pairCellSlot[iPair];
            const double     *qp =
              quadCoords + (cellSlot * numQuadPoints + iQuad) * 3;
            const double dx = qp[0] - pairCenters[iPair * 3 + 0];
            const double dy = qp[1] - pairCenters[iPair * 3 + 1];
            const double dz = qp[2] - pairCenters[iPair * 3 + 2];
            const double r  = std::sqrt(dx * dx + dy * dy + dz * dz);

            const dftfe::uInt splineId = pairSplineId[iPair];
            const double      cutoff   = splineCutoff[splineId];
            double            v;
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
              }
            else
              {
                // the host takes the analytic tail for r > cutoff, and the
                // spline path (which returns 0 at r >= cutoff) at r == cutoff
                v = r > cutoff ? -pairCharges[iPair] / r : 0.0;
              }
            valueOut[index] = v;
            if (r <= cutOffForPsp)
              pairHasSupport[iPair] = 1;
          }
      },
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
      dftfe::uInt       *pairHasSupport)
    {
      if (numPairs == 0 || numQuadPoints == 0)
        return;
      const dftfe::uInt numberEntries = numPairs * numQuadPoints;
      DFTFE_LAUNCH_KERNEL(
        localPseudoPotentialValuesKernel,
        (dftfe::utils::DEVICE_BLOCK_SIZE + numberEntries) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::linearAlgebra::BLASWrapper<
          dftfe::utils::MemorySpace::DEVICE>::d_streamId,
        numPairs,
        numQuadPoints,
        cutOffForPsp,
        dftfe::utils::makeDataTypeDeviceCompatible(quadCoords),
        dftfe::utils::makeDataTypeDeviceCompatible(pairCenters),
        dftfe::utils::makeDataTypeDeviceCompatible(pairCharges),
        dftfe::utils::makeDataTypeDeviceCompatible(pairCellSlot),
        dftfe::utils::makeDataTypeDeviceCompatible(pairSplineId),
        dftfe::utils::makeDataTypeDeviceCompatible(knotXAll),
        dftfe::utils::makeDataTypeDeviceCompatible(knotYAll),
        dftfe::utils::makeDataTypeDeviceCompatible(coefBAll),
        dftfe::utils::makeDataTypeDeviceCompatible(coefCAll),
        dftfe::utils::makeDataTypeDeviceCompatible(coefDAll),
        dftfe::utils::makeDataTypeDeviceCompatible(splineOffsets),
        dftfe::utils::makeDataTypeDeviceCompatible(splineNKnots),
        dftfe::utils::makeDataTypeDeviceCompatible(splineGPParams),
        dftfe::utils::makeDataTypeDeviceCompatible(splineCutoff),
        dftfe::utils::makeDataTypeDeviceCompatible(splineRMin),
        dftfe::utils::makeDataTypeDeviceCompatible(valueOut),
        dftfe::utils::makeDataTypeDeviceCompatible(pairHasSupport));
    }

    DFTFE_CREATE_KERNEL(
      void,
      localPseudoPotentialTotalKernel,
      {
        const dftfe::uInt numberEntries = numCells * numQuadPoints;
        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            const double *qp  = quadCoords + index * 3;
            const double  qx  = qp[0];
            const double  qy  = qp[1];
            const double  qz  = qp[2];
            double        acc = 0.0;
            for (dftfe::uInt iAtom = 0; iAtom < numCenters; ++iAtom)
              {
                const double dx     = qx - centerCoords[iAtom * 3 + 0];
                const double dy     = qy - centerCoords[iAtom * 3 + 1];
                const double dz     = qz - centerCoords[iAtom * 3 + 2];
                const double r      = std::sqrt(dx * dx + dy * dy + dz * dz);
                const double rInv   = 1.0 / r;
                const double charge = centerCharges[iAtom];

                double v;
                if (r <= maxTail && r <= centerTailCutoff[iAtom])
                  {
                    // getRadialValue: 0 at r >= cutoff, clamp to rMin below
                    const dftfe::uInt sid    = centerSplineId[iAtom];
                    const double      cutoff = splineCutoff[sid];
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
                                                 sid);
                        const double rMin  = splineRMin[sid];
                        const double rEval = r <= rMin ? rMin : r;
                        v = splineEval::splineEvalValue(t, rEval);
                      }
                    else
                      v = 0.0;
                  }
                else
                  v = -charge * rInv;

                if (analyticSmearedCharge)
                  {
                    const double w = centerSmearedWidth[iAtom];
                    v -= r > w ? -charge / r : -charge * smearedPot(r, w);
                  }
                acc += v;
              }
            valueOut[index] = acc;
          }
      },
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
      double            *valueOut)
    {
      if (numCells == 0 || numQuadPoints == 0 || numCenters == 0)
        return;
      const dftfe::uInt numberEntries = numCells * numQuadPoints;
      DFTFE_LAUNCH_KERNEL(
        localPseudoPotentialTotalKernel,
        (dftfe::utils::DEVICE_BLOCK_SIZE + numberEntries) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::linearAlgebra::BLASWrapper<
          dftfe::utils::MemorySpace::DEVICE>::d_streamId,
        numCells,
        numQuadPoints,
        numCenters,
        analyticSmearedCharge,
        maxTail,
        dftfe::utils::makeDataTypeDeviceCompatible(quadCoords),
        dftfe::utils::makeDataTypeDeviceCompatible(centerCoords),
        dftfe::utils::makeDataTypeDeviceCompatible(centerCharges),
        dftfe::utils::makeDataTypeDeviceCompatible(centerTailCutoff),
        dftfe::utils::makeDataTypeDeviceCompatible(centerSmearedWidth),
        dftfe::utils::makeDataTypeDeviceCompatible(centerSplineId),
        dftfe::utils::makeDataTypeDeviceCompatible(knotXAll),
        dftfe::utils::makeDataTypeDeviceCompatible(knotYAll),
        dftfe::utils::makeDataTypeDeviceCompatible(coefBAll),
        dftfe::utils::makeDataTypeDeviceCompatible(coefCAll),
        dftfe::utils::makeDataTypeDeviceCompatible(coefDAll),
        dftfe::utils::makeDataTypeDeviceCompatible(splineOffsets),
        dftfe::utils::makeDataTypeDeviceCompatible(splineNKnots),
        dftfe::utils::makeDataTypeDeviceCompatible(splineGPParams),
        dftfe::utils::makeDataTypeDeviceCompatible(splineCutoff),
        dftfe::utils::makeDataTypeDeviceCompatible(splineRMin),
        dftfe::utils::makeDataTypeDeviceCompatible(valueOut));
    }

    DFTFE_CREATE_KERNEL(
      void,
      atomicDensitySuperpositionKernel,
      {
        for (dftfe::uInt index = globalThreadId; index < numPoints;
             index += nThreadsPerBlock * nThreadBlock)
          {
            const double *pp = pointCoords + index * 3;
            const double  px = pp[0];
            const double  py = pp[1];
            const double  pz = pp[2];

            double rho  = 0.0;
            double magZ = 0.0;
            double magY = 0.0;
            double magX = 0.0;

            for (dftfe::uInt iAtom = 0; iAtom < numCenters; ++iAtom)
              {
                const double dx = px - centerCoords[iAtom * 3 + 0];
                const double dy = py - centerCoords[iAtom * 3 + 1];
                const double dz = pz - centerCoords[iAtom * 3 + 2];
                const double r  = std::sqrt(dx * dx + dy * dy + dz * dz);

                if (r <= centerCutoff[iAtom])
                  {
                    // getRadialValue: 0 at r >= cutoff, clamp to rMin below
                    const dftfe::uInt sid    = centerSplineId[iAtom];
                    const double      cutoff = splineCutoff[sid];
                    double            val    = 0.0;
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
                                                 sid);
                        const double rMin  = splineRMin[sid];
                        const double rEval = r <= rMin ? rMin : r;
                        val = splineEval::splineEvalValue(t, rEval);
                      }
                    const double tempRhoValue = centerRhoFactor[iAtom] * val;
                    rho += tempRhoValue;
                    magZ += centerMagZFactor[iAtom] * tempRhoValue;
                    magY += centerMagYFactor[iAtom] * tempRhoValue;
                    magX += centerMagXFactor[iAtom] * tempRhoValue;
                  }
              }

            rhoOut[index] = rho < 0.0 ? -rho : rho;
            if (numMagComponents >= 1)
              magZOut[index] = magZ;
            if (numMagComponents == 3)
              {
                magYOut[index] = magY;
                magXOut[index] = magX;
              }
          }
      },
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
      double            *magXOut)
    {
      if (numPoints == 0 || numCenters == 0)
        return;
      DFTFE_LAUNCH_KERNEL(
        atomicDensitySuperpositionKernel,
        (dftfe::utils::DEVICE_BLOCK_SIZE + numPoints) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::linearAlgebra::BLASWrapper<
          dftfe::utils::MemorySpace::DEVICE>::d_streamId,
        numPoints,
        numCenters,
        numMagComponents,
        dftfe::utils::makeDataTypeDeviceCompatible(pointCoords),
        dftfe::utils::makeDataTypeDeviceCompatible(centerCoords),
        dftfe::utils::makeDataTypeDeviceCompatible(centerSplineId),
        dftfe::utils::makeDataTypeDeviceCompatible(centerCutoff),
        dftfe::utils::makeDataTypeDeviceCompatible(centerRhoFactor),
        dftfe::utils::makeDataTypeDeviceCompatible(centerMagZFactor),
        dftfe::utils::makeDataTypeDeviceCompatible(centerMagYFactor),
        dftfe::utils::makeDataTypeDeviceCompatible(centerMagXFactor),
        dftfe::utils::makeDataTypeDeviceCompatible(knotXAll),
        dftfe::utils::makeDataTypeDeviceCompatible(knotYAll),
        dftfe::utils::makeDataTypeDeviceCompatible(coefBAll),
        dftfe::utils::makeDataTypeDeviceCompatible(coefCAll),
        dftfe::utils::makeDataTypeDeviceCompatible(coefDAll),
        dftfe::utils::makeDataTypeDeviceCompatible(splineOffsets),
        dftfe::utils::makeDataTypeDeviceCompatible(splineNKnots),
        dftfe::utils::makeDataTypeDeviceCompatible(splineGPParams),
        dftfe::utils::makeDataTypeDeviceCompatible(splineCutoff),
        dftfe::utils::makeDataTypeDeviceCompatible(splineRMin),
        dftfe::utils::makeDataTypeDeviceCompatible(rhoOut),
        dftfe::utils::makeDataTypeDeviceCompatible(magZOut),
        dftfe::utils::makeDataTypeDeviceCompatible(magYOut),
        dftfe::utils::makeDataTypeDeviceCompatible(magXOut));
    }

    DFTFE_CREATE_KERNEL(
      void,
      atomicWavefunctionSuperpositionKernel,
      {
        const dftfe::uInt numberEntries = numPoints * numOrbitals;
        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            const dftfe::uInt iPoint = index / numOrbitals;
            const dftfe::uInt iOrb   = index - iPoint * numOrbitals;

            const double *pp = pointCoords + iPoint * 3;
            const double  px = pp[0];
            const double  py = pp[1];
            const double  pz = pp[2];

            const int                         l   = orbitalL[iOrb];
            const int                         m   = orbitalM[iOrb];
            const dftfe::uInt                 sid = orbitalSplineId[iOrb];
            const splineEval::SplineTableView t =
              splineEval::splineView(knotXAll,
                                     knotYAll,
                                     coefBAll,
                                     coefCAll,
                                     coefDAll,
                                     splineOffsets,
                                     splineNKnots,
                                     splineGPParams,
                                     sid);

            double acc = 0.0;
            for (dftfe::uInt iAtom = orbitalCenterStart[iOrb];
                 iAtom < orbitalCenterStart[iOrb + 1];
                 ++iAtom)
              {
                const double x = px - centerCoords[iAtom * 3 + 0];
                const double y = py - centerCoords[iAtom * 3 + 1];
                const double z = pz - centerCoords[iAtom * 3 + 2];
                const double r = std::sqrt(x * x + y * y + z * z);

                if (r > truncationRadius)
                  continue;

                double theta = std::acos(z / r);
                double phi   = std::atan2(y, x);
                if (r == 0.0)
                  {
                    theta = 0.0;
                    phi   = 0.0;
                  }

                // raw spline evaluation: the host calls alglib::spline1dcalc
                // directly here, without the zero-beyond-cutoff and rMin
                // clamping that AtomCenteredSphericalFunctionSpline applies
                const double R = splineEval::splineEvalValue(t, r);
                const double Y = atomCenteredSphericalFunctionDeviceUtils::
                  realSphericalHarmonic(l, m, theta, phi);

                acc += isNoncolin ? R * Y / 1.4142135623730950488016887242097 :
                                    R * Y;
              }
            valueOut[index] = acc;
          }
      },
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
      double            *valueOut)
    {
      if (numPoints == 0 || numOrbitals == 0)
        return;
      const dftfe::uInt numberEntries = numPoints * numOrbitals;
      DFTFE_LAUNCH_KERNEL(
        atomicWavefunctionSuperpositionKernel,
        (dftfe::utils::DEVICE_BLOCK_SIZE + numberEntries) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::linearAlgebra::BLASWrapper<
          dftfe::utils::MemorySpace::DEVICE>::d_streamId,
        numPoints,
        numOrbitals,
        isNoncolin,
        truncationRadius,
        dftfe::utils::makeDataTypeDeviceCompatible(pointCoords),
        dftfe::utils::makeDataTypeDeviceCompatible(orbitalCenterStart),
        dftfe::utils::makeDataTypeDeviceCompatible(centerCoords),
        dftfe::utils::makeDataTypeDeviceCompatible(orbitalL),
        dftfe::utils::makeDataTypeDeviceCompatible(orbitalM),
        dftfe::utils::makeDataTypeDeviceCompatible(orbitalSplineId),
        dftfe::utils::makeDataTypeDeviceCompatible(knotXAll),
        dftfe::utils::makeDataTypeDeviceCompatible(knotYAll),
        dftfe::utils::makeDataTypeDeviceCompatible(coefBAll),
        dftfe::utils::makeDataTypeDeviceCompatible(coefCAll),
        dftfe::utils::makeDataTypeDeviceCompatible(coefDAll),
        dftfe::utils::makeDataTypeDeviceCompatible(splineOffsets),
        dftfe::utils::makeDataTypeDeviceCompatible(splineNKnots),
        dftfe::utils::makeDataTypeDeviceCompatible(splineGPParams),
        dftfe::utils::makeDataTypeDeviceCompatible(valueOut));
    }
  } // namespace atomicInitKernels
} // namespace dftfe
