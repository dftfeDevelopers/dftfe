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
// HOST-ONLY utilities to unpack alglib radial splines into the flat
// coefficient tables consumed by the kernels (splineDeviceEval.h) and to
// concatenate a group of such tables into contiguous host arrays.
// AtomTypeSplineTables (atomTypeSplineTables.h) stages the result into the
// memory space the kernels run in. Not to be included from
// device-compiled translation units.
//

#ifndef DFTFE_SPLINEHOSTTABLEUTILS_H
#define DFTFE_SPLINEHOSTTABLEUTILS_H

#include <dftfe/splineDeviceEval.h>
#include <dftfe/TypeConfig.h>
#include <interpolation.h>

namespace dftfe
{
  namespace splineHostTableUtils
  {
    /**
     * @brief Unpacks an alglib cubic spline into per-interval coefficient
     * tables (alglib convention: y = c0 + c1*h + c2*h^2 + c3*h^3 on
     * [x_i, x_{i+1}]) and detects the geometric-progression closed-form
     * lookup parameters.
     */
    inline void
    unpackAlglibSpline(const alglib::spline1dinterpolant &spline,
                       splineEval::HostSplineTable       &table)
    {
      alglib::ae_int_t      n;
      alglib::real_2d_array tbl;
      alglib::spline1dunpack(const_cast<alglib::spline1dinterpolant &>(spline),
                             n,
                             tbl);
      const dftfe::uInt nIntervals = static_cast<dftfe::uInt>(n) - 1;
      table.knotX.resize(nIntervals + 1);
      table.knotY.resize(nIntervals);
      table.coefB.resize(nIntervals);
      table.coefC.resize(nIntervals);
      table.coefD.resize(nIntervals);
      for (dftfe::uInt i = 0; i < nIntervals; i++)
        {
          table.knotX[i] = tbl[i][0];
          table.knotY[i] = tbl[i][2];
          table.coefB[i] = tbl[i][3];
          table.coefC[i] = tbl[i][4];
          table.coefD[i] = tbl[i][5];
        }
      table.knotX[nIntervals] = tbl[nIntervals - 1][1];
      splineEval::detectGeometricGrid(table);
    }

    /**
     * @brief Builder that concatenates several HostSplineTables and
     * uploads them (plus per-function metadata) as device arrays in the
     * layout expected by the atomCenteredSphericalFunctionDeviceKernels
     * launchers. addSpline() returns the function id to be referenced by
     * pairSplineId / speciesRadialFnStart arrays.
     *
     * Note: knotY/coefB/coefC/coefD are per-interval arrays but are padded
     * to nKnots entries so a single offsets array serves all five.
     */
    struct SplineTableGroup
    {
      std::vector<double>      knotXAll, knotYAll, coefBAll, coefCAll, coefDAll;
      std::vector<dftfe::uInt> offsets, nKnots;
      std::vector<double>      gpParams; // [fn*4]: isGP, a, q, invLogQ
      std::vector<double>      cutoffs;
      std::vector<double>      rMins;
      std::vector<double>      evalParams; // [fn*2]: mode, y0

      /**
       * @brief scale multiplies all value coefficients (used to fold
       * constant factors like 1/sqrt(4 pi) into the tables); rMin
       * replicates the AtomCenteredSphericalFunctionSpline clamp.
       *
       * evalMode selects the getRadialValue semantics replicated by the
       * consuming kernel: 0 = AtomCenteredSphericalFunctionSpline
       * (clamp r to rMin below, evaluate spline of f(r)); 1 =
       * AtomCenteredSphericalFunctionPAWProjectorSpline2 (spline of
       * r*f(r): y0 within 1e-8 of rMin, spline(r)/r otherwise, no
       * clamp). y0 is only used by mode 1 and is scaled like the
       * coefficients.
       */
      dftfe::uInt
      addSpline(const alglib::spline1dinterpolant &spline,
                const double                       cutoff,
                const double                       rMin,
                const double                       scale    = 1.0,
                const dftfe::uInt                  evalMode = 0,
                const double                       y0       = 0.0)
      {
        splineEval::HostSplineTable t;
        unpackAlglibSpline(spline, t);
        if (scale != 1.0)
          for (dftfe::uInt i = 0; i < t.knotY.size(); i++)
            {
              t.knotY[i] *= scale;
              t.coefB[i] *= scale;
              t.coefC[i] *= scale;
              t.coefD[i] *= scale;
            }
        const dftfe::uInt fnId = nKnots.size();
        offsets.push_back(knotXAll.size());
        nKnots.push_back(t.knotX.size());
        knotXAll.insert(knotXAll.end(), t.knotX.begin(), t.knotX.end());
        // pad interval arrays to nKnots entries to share the offsets
        auto appendPadded = [&](std::vector<double>       &dstAll,
                                const std::vector<double> &src) {
          dstAll.insert(dstAll.end(), src.begin(), src.end());
          dstAll.push_back(0.0);
        };
        appendPadded(knotYAll, t.knotY);
        appendPadded(coefBAll, t.coefB);
        appendPadded(coefCAll, t.coefC);
        appendPadded(coefDAll, t.coefD);
        gpParams.push_back(t.isGPGrid ? 1.0 : 0.0);
        gpParams.push_back(t.gpA);
        gpParams.push_back(t.gpQ);
        gpParams.push_back(t.invLogGpQ);
        cutoffs.push_back(cutoff);
        rMins.push_back(rMin);
        evalParams.push_back(static_cast<double>(evalMode));
        evalParams.push_back(y0 * scale);
        return fnId;
      }
    };
  } // namespace splineHostTableUtils
} // namespace dftfe

#endif // DFTFE_SPLINEHOSTTABLEUTILS_H
