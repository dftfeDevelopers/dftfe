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
// Host+device evaluation of piecewise-cubic radial splines on coefficient
// tables unpacked from the alglib spline1dinterpolant objects that DFT-FE
// already builds, so device evaluations reproduce the existing host
// spline1dcalc results. The knot-interval lookup is closed-form on
// geometric-progression radial meshes (JTH r(i) = a*(q^i - 1)/(q - 1)) and
// falls back to binary search otherwise.
//
// The host+device inline evaluation scheme and the closed-form
// geometric-progression interval lookup are derived from the DFT-EFE code,
// src/utils/SplineKernels.h (@author Avirup Sircar). The DFT-EFE copyright
// notice of that file is reproduced below:
//
/******************************************************************************
 * Copyright (c) 2021.                                                        *
 * The Regents of the University of Michigan and DFT-EFE developers.          *
 *                                                                            *
 * This file is part of the DFT-EFE code.                                     *
 *                                                                            *
 * DFT-EFE is free software: you can redistribute it and/or modify            *
 *   it under the terms of the Lesser GNU General Public License as           *
 *   published by the Free Software Foundation, either version 3 of           *
 *   the License, or (at your option) any later version.                      *
 *                                                                            *
 * DFT-EFE is distributed in the hope that it will be useful, but             *
 *   WITHOUT ANY WARRANTY; without even the implied warranty                  *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                     *
 *   See the Lesser GNU General Public License for more details.              *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public           *
 *   License at the top level of DFT-EFE distribution.  If not, see           *
 *   <https://www.gnu.org/licenses/>.                                         *
 ******************************************************************************/
//

#ifndef DFTFE_SPLINEDEVICEEVAL_H
#define DFTFE_SPLINEDEVICEEVAL_H

#include <cmath>
#include <vector>

// Keyed on the compiler-mode macros, which are defined only while the
// device compiler is actually compiling this translation unit. The HIP
// platform macros cannot be used here: config.h defines them for every
// unit of a HIP build, including the host-only units that the plain C++
// compiler builds, which would then see the device attributes.
#if defined(__CUDACC__) || defined(__HIPCC__)
#  define DFTFE_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#  define DFTFE_HOST_DEVICE_INLINE inline
#endif

namespace dftfe
{
  namespace splineEval
  {
    /**
     * @brief POD view of one radial spline's coefficient tables, passable
     * by value into kernels. The piecewise cubic on interval i is
     * y(x) = knotY[i] + h*(coefB[i] + h*(coefC[i] + h*coefD[i])),
     * h = x - knotX[i], matching alglib's spline1dunpack convention, so
     * evaluation agrees with the host spline1dcalc to machine precision.
     * Points outside [knotX[0], knotX[nKnots-1]] use the boundary interval
     * polynomial (alglib extrapolation behavior); callers are expected to
     * apply radial cutoffs before evaluation as the host code does.
     */
    struct SplineTableView
    {
      const double *knotX; // nKnots entries
      const double *knotY; // nKnots-1 entries (per interval)
      const double *coefB; // nKnots-1
      const double *coefC; // nKnots-1
      const double *coefD; // nKnots-1
      dftfe::uInt   nKnots;
      // geometric-progression mesh parameters (isGPGrid):
      // knotX[i] = gpA * (pow(gpQ, i) - 1) / (gpQ - 1), gpQ > 1
      bool   isGPGrid;
      double gpA;
      double gpQ;
      double invLogGpQ; // 1/log(gpQ), precomputed
    };

    /**
     * @brief View onto one radial spline inside the packed tables.
     *
     * splineGPParams is 4-wide per function: {isGPGrid, a, q, 1/log(q)}.
     */
    DFTFE_HOST_DEVICE_INLINE SplineTableView
    splineView(const double      *knotXAll,
               const double      *knotYAll,
               const double      *coefBAll,
               const double      *coefCAll,
               const double      *coefDAll,
               const dftfe::uInt *splineOffsets,
               const dftfe::uInt *splineNKnots,
               const double      *splineGPParams,
               const dftfe::uInt  fnId)
    {
      SplineTableView   t;
      const dftfe::uInt off = splineOffsets[fnId];
      t.knotX               = knotXAll + off;
      t.knotY               = knotYAll + off;
      t.coefB               = coefBAll + off;
      t.coefC               = coefCAll + off;
      t.coefD               = coefDAll + off;
      t.nKnots              = splineNKnots[fnId];
      t.isGPGrid            = splineGPParams[fnId * 4 + 0] > 0.5;
      t.gpA                 = splineGPParams[fnId * 4 + 1];
      t.gpQ                 = splineGPParams[fnId * 4 + 2];
      t.invLogGpQ           = splineGPParams[fnId * 4 + 3];
      return t;
    }

    DFTFE_HOST_DEVICE_INLINE dftfe::uInt
    splineFindInterval(const SplineTableView &t, const double x)
    {
      const dftfe::uInt nIntervals = t.nKnots - 1;
      if (x <= t.knotX[0])
        return 0;
      if (x >= t.knotX[t.nKnots - 1])
        return nIntervals - 1;
      dftfe::uInt idx;
      if (t.isGPGrid)
        {
          const double arg = x * (t.gpQ - 1.0) / t.gpA + 1.0;
          double       nd  = std::log(arg) * t.invLogGpQ;
          idx              = nd > 0.0 ? static_cast<dftfe::uInt>(nd) : 0u;
          if (idx > nIntervals - 1)
            idx = nIntervals - 1;
          // nudge against floating-point drift of the closed form
          while (idx > 0 && x < t.knotX[idx])
            idx--;
          while (idx + 1 < t.nKnots - 1 && x >= t.knotX[idx + 1])
            idx++;
        }
      else
        {
          dftfe::uInt lo = 0;
          dftfe::uInt hi = t.nKnots - 1;
          while (hi - lo > 1)
            {
              const dftfe::uInt mid = (lo + hi) / 2;
              if (t.knotX[mid] <= x)
                lo = mid;
              else
                hi = mid;
            }
          idx = lo;
        }
      return idx;
    }

    DFTFE_HOST_DEVICE_INLINE double
    splineEvalValue(const SplineTableView &t, const double x)
    {
      const dftfe::uInt i = splineFindInterval(t, x);
      const double      h = x - t.knotX[i];
      return t.knotY[i] + h * (t.coefB[i] + h * (t.coefC[i] + h * t.coefD[i]));
    }

    // order = 1 or 2
    DFTFE_HOST_DEVICE_INLINE double
    splineEvalDerivative(const SplineTableView &t,
                         const double           x,
                         const int              order)
    {
      const dftfe::uInt i = splineFindInterval(t, x);
      const double      h = x - t.knotX[i];
      if (order == 1)
        return t.coefB[i] + h * (2.0 * t.coefC[i] + 3.0 * h * t.coefD[i]);
      return 2.0 * t.coefC[i] + 6.0 * h * t.coefD[i];
    }

    /**
     * @brief Host-side flattened coefficient table of one alglib spline,
     * ready for upload to a device MemoryStorage. Filled by
     * splineHostTableUtils::unpackAlglibSpline() (host-only header
     * splineHostTableUtils.h) or directly.
     */
    struct HostSplineTable
    {
      std::vector<double> knotX;
      std::vector<double> knotY;
      std::vector<double> coefB;
      std::vector<double> coefC;
      std::vector<double> coefD;
      bool                isGPGrid  = false;
      double              gpA       = 0.0;
      double              gpQ       = 0.0;
      double              invLogGpQ = 0.0;
    };

    /**
     * @brief Detects whether the knots follow a geometric-progression mesh
     * knotX[i] = a*(q^i - 1)/(q - 1) with q > 1 (JTH radial meshes) and
     * fills the closed-form lookup parameters.
     */
    inline void
    detectGeometricGrid(HostSplineTable &table)
    {
      table.isGPGrid      = false;
      const size_t nKnots = table.knotX.size();
      if (nKnots < 4)
        return;
      const double dx0 = table.knotX[1] - table.knotX[0];
      if (dx0 <= 0.0)
        return;
      const double q = (table.knotX[2] - table.knotX[1]) / dx0;
      if (q <= 1.0 + 1e-10)
        return;
      // verify the ratio holds across the mesh
      for (size_t i = 1; i + 1 < nKnots; i++)
        {
          const double dxi  = table.knotX[i + 1] - table.knotX[i];
          const double dxim = table.knotX[i] - table.knotX[i - 1];
          if (std::abs(dxi / dxim - q) > 1e-6 * q)
            return;
        }
      table.isGPGrid  = true;
      table.gpQ       = q;
      table.gpA       = dx0; // x_i = dx0*(q^i - 1)/(q - 1) since x_0 = 0
      table.invLogGpQ = 1.0 / std::log(q);
      // closed form assumes the mesh starts at zero; otherwise fall back
      if (std::abs(table.knotX[0]) > 1e-14)
        table.isGPGrid = false;
    }
  } // namespace splineEval
} // namespace dftfe

#endif // DFTFE_SPLINEDEVICEEVAL_H
