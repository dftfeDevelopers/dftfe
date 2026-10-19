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
// Host-side construction of the packed radial spline tables evaluated by
// splineDeviceEval.h: alglib splines are unpacked into per-interval
// coefficients, concatenated into a SplineTableGroup, and staged into the
// memory space the kernels run in. The alglib types cannot be parsed by the
// device compiler, so device-compiled translation units must not include
// this header.
//

#ifndef atomTypeSplineTables_H_
#define atomTypeSplineTables_H_
#include <dftfe/splineDeviceEval.h>
#include <dftfe/AtomCenteredSphericalFunctionSpline.h>
#include <dftfe/MemoryStorage.h>
#include <dftfe/MemorySpaceType.h>
#include <dftfe/TypeConfig.h>
#include <interpolation.h>
#include <cmath>
#include <map>
#include <set>
#include <vector>

namespace dftfe
{
  namespace splineHostTableUtils
  {
    /**
     * @brief Flattened coefficient tables of one alglib spline, in the
     * layout of splineEval::SplineTableView.
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

    /**
     * @brief Unpacks an alglib cubic spline into per-interval coefficient
     * tables, in the alglib convention
     * @code
     *   y = c0 + c1 * h + c2 * h^2 + c3 * h^3,  h = x - x_i,  x_i <= x <=
     * x_{i+1}
     * @endcode
     * and detects the geometric-progression closed-form lookup parameters.
     */
    inline void
    unpackAlglibSpline(const alglib::spline1dinterpolant &spline,
                       HostSplineTable                   &table)
    {
      alglib::ae_int_t      n;
      alglib::real_2d_array tbl;
      alglib::spline1dunpack(spline, n, tbl);
      AssertThrow(n >= 2,
                  dealii::ExcMessage(
                    "DFT-FE Error: a radial spline needs at least two knots."));
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
      detectGeometricGrid(table);
    }

    /**
     * @brief Several splines concatenated into contiguous host arrays, in
     * the layout of splineEval::SplineTableGroupView.
     * AtomTypeSplineTables::stage copies them into the memory space the
     * kernels run in.
     *
     * knotY, coefB, coefC and coefD are per-interval arrays padded to
     * nKnots entries, so that one offsets array serves all five.
     */
    struct SplineTableGroup
    {
      std::vector<double>      knotXAll, knotYAll, coefBAll, coefCAll, coefDAll;
      std::vector<dftfe::uInt> offsets; ///< first knot of each function
      std::vector<dftfe::uInt> nKnots;  ///< knot count of each function
      std::vector<double> gpParams; ///< 4 per function: isGP, a, q, 1/log(q)
      std::vector<double> cutoffs;  ///< radial cutoff of each function
      std::vector<double> rMins;    ///< lower clamp radius of each function

      /**
       * @brief Appends one spline to the group.
       *
       * @param[in] spline the alglib spline to unpack
       * @param[in] cutoff radius at and beyond which the function is zero
       * @param[in] rMin radius below which the function is evaluated at
       * rMin, as AtomCenteredSphericalFunctionSpline does
       * @return the function id to index the group with
       */
      dftfe::uInt
      addSpline(const alglib::spline1dinterpolant &spline,
                const double                       cutoff,
                const double                       rMin)
      {
        HostSplineTable t;
        unpackAlglibSpline(spline, t);
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
        return fnId;
      }
    };
  } // namespace splineHostTableUtils

  /**
   * @brief Appends the radial function of every atom type that holds data to
   * a SplineTableGroup.
   *
   * An atom type whose function holds no data (isDataPresent() false, e.g.
   * the core density of a species without nonlinear core correction) gets no
   * entry: it contributes nothing, and callers skip its centers.
   *
   * @param[in] atomTypes atomic numbers to add, in iteration order
   * @param[in] getRadialFunction maps an atomic number to its
   * AtomCenteredSphericalFunctionBase, which must be spline backed
   * @param[in,out] tables group the splines are appended to
   * @param[out] znumToSplineId function id of every atom type with data
   */
  template <typename FnAccessor>
  void
  buildAtomTypeSplineTables(const std::set<dftfe::uInt> &atomTypes,
                            const FnAccessor            &getRadialFunction,
                            splineHostTableUtils::SplineTableGroup &tables,
                            std::map<dftfe::uInt, dftfe::uInt> &znumToSplineId)
  {
    znumToSplineId.clear();
    for (std::set<dftfe::uInt>::const_iterator it = atomTypes.begin();
         it != atomTypes.end();
         ++it)
      {
        const AtomCenteredSphericalFunctionSpline *fnSpline =
          dynamic_cast<const AtomCenteredSphericalFunctionSpline *>(
            getRadialFunction(*it));
        AssertThrow(fnSpline != nullptr,
                    dealii::ExcMessage(
                      "DFT-FE Error: radial function is not spline backed."));
        if (!fnSpline->isDataPresent())
          continue;
        znumToSplineId[*it] =
          tables.addSpline(fnSpline->getRadialSplineObject(),
                           fnSpline->getRadialCutOff(),
                           fnSpline->getrMinVal());
      }
  }

  /**
   * @brief A SplineTableGroup in the memory space the kernels run in.
   */
  template <dftfe::utils::MemorySpace memorySpace>
  struct AtomTypeSplineTables
  {
    dftfe::utils::MemoryStorage<double, memorySpace> knotX, knotY, coefB, coefC,
      coefD, gpParams, cutoffs, rMins;
    dftfe::utils::MemoryStorage<dftfe::uInt, memorySpace> offsets, nKnots;

    /**
     * @brief Copies every array of src into this memory space.
     */
    void
    stage(const splineHostTableUtils::SplineTableGroup &src)
    {
      knotX =
        dftfe::utils::memoryStorageFromSTL<double, memorySpace>(src.knotXAll);
      knotY =
        dftfe::utils::memoryStorageFromSTL<double, memorySpace>(src.knotYAll);
      coefB =
        dftfe::utils::memoryStorageFromSTL<double, memorySpace>(src.coefBAll);
      coefC =
        dftfe::utils::memoryStorageFromSTL<double, memorySpace>(src.coefCAll);
      coefD =
        dftfe::utils::memoryStorageFromSTL<double, memorySpace>(src.coefDAll);
      gpParams =
        dftfe::utils::memoryStorageFromSTL<double, memorySpace>(src.gpParams);
      cutoffs =
        dftfe::utils::memoryStorageFromSTL<double, memorySpace>(src.cutoffs);
      rMins =
        dftfe::utils::memoryStorageFromSTL<double, memorySpace>(src.rMins);
      offsets = dftfe::utils::memoryStorageFromSTL<dftfe::uInt, memorySpace>(
        src.offsets);
      nKnots = dftfe::utils::memoryStorageFromSTL<dftfe::uInt, memorySpace>(
        src.nKnots);
    }

    /**
     * @brief Pointer view of the staged arrays, to pass to the kernels.
     */
    splineEval::SplineTableGroupView
    view() const
    {
      return {knotX.data(),
              knotY.data(),
              coefB.data(),
              coefC.data(),
              coefD.data(),
              offsets.data(),
              nKnots.data(),
              gpParams.data(),
              cutoffs.data(),
              rMins.data()};
    }
  };
} // namespace dftfe
#endif
