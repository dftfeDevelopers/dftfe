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
#include <cstdlib>
#include <string>
#include <dftfe/AtomCenteredSphericalFunctionDeviceUtils.h>
#include <dftfe/splineDeviceEval.h>
#include <cmath>

namespace dftfe
{
  namespace atomicInitKernels
  {
    namespace
    {
      // host threading is governed by DFTFE_NUM_THREADS, as in dftClass and
      // FEBasisOperations
      int
      hostThreadCount()
      {
        static const int nThreads = []() -> int {
          int n = 1;
#ifdef _OPENMP
          if (const char *penv = std::getenv("DFTFE_NUM_THREADS"))
            {
              try
                {
                  n = std::stoi(std::string(penv));
                }
              catch (...)
                {
                  n = 1;
                }
            }
#endif
          return n < 1 ? 1 : n;
        }();
        return nThreads;
      }
    } // namespace

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
      double            *magXOut)
    {
#pragma omp        parallel for num_threads(hostThreadCount())
      for (dftfe::uInt index = 0; index < numPoints; ++index)
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
    }

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
      double            *valueOut)
    {
      const dftfe::uInt numberEntries = numPoints * numOrbitals;
#pragma omp parallel for num_threads(hostThreadCount())
      for (dftfe::uInt index = 0; index < numberEntries; ++index)
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

              const double R = splineEval::splineEvalValue(t, r);
              const double Y =
                atomCenteredSphericalFunctionDeviceUtils::realSphericalHarmonic(
                  l, m, theta, phi);

              acc +=
                isNoncolin ? R * Y / 1.4142135623730950488016887242097 : R * Y;
            }
          valueOut[index] = acc;
        }
    }
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
      dftfe::uInt       *pairHasSupport)
    {
      if (numPairs == 0)
        return;

      const dftfe::uInt numberEntries = numPairs * numQuadPoints;
#pragma omp parallel for num_threads(hostThreadCount())
      for (dftfe::uInt index = 0; index < numberEntries; ++index)
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
            {
              // quadrature points of one pair share this flag
#pragma omp atomic write
              pairHasSupport[iPair] = 1;
            }
        }
    }

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
      double            *valueOut)
    {
      if (numCells == 0)
        return;

      const dftfe::uInt numberEntries = numCells * numQuadPoints;
#pragma omp parallel for num_threads(hostThreadCount())
      for (dftfe::uInt index = 0; index < numberEntries; ++index)
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
    }

  } // namespace atomicInitKernels
} // namespace dftfe
