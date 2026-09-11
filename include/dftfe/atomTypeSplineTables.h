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
// Kept apart from atomicInitKernels.h because the alglib spline types used
// here cannot be parsed by the device compiler.
//

#ifndef atomTypeSplineTables_H_
#define atomTypeSplineTables_H_
#include <dftfe/splineHostTableUtils.h>
#include <dftfe/AtomCenteredSphericalFunctionSpline.h>
#include <dftfe/MemoryStorage.h>
#include <dftfe/MemorySpaceType.h>
#include <dftfe/TypeConfig.h>
#include <map>
#include <set>

namespace dftfe
{
  /**
   * @brief Packs one radial function per atom type into device spline tables.
   *
   * getRadialFunction maps an atomic number to its
   * AtomCenteredSphericalFunctionBase. Returns false, with nothing uploaded,
   * if any atom type is not spline backed, in which case the caller must use
   * its host path.
   */
  template <typename FnAccessor>
  bool
  buildAtomTypeSplineTables(const std::set<dftfe::uInt> &atomTypes,
                            const FnAccessor            &getRadialFunction,
                            splineHostTableUtils::SplineTableGroup &tables,
                            std::map<dftfe::uInt, dftfe::uInt> &znumToSplineId)
  {
    for (std::set<dftfe::uInt>::const_iterator it = atomTypes.begin();
         it != atomTypes.end();
         ++it)
      {
        const AtomCenteredSphericalFunctionSpline *fnSpline =
          dynamic_cast<const AtomCenteredSphericalFunctionSpline *>(
            getRadialFunction(*it));
        if (fnSpline == nullptr)
          return false;
        znumToSplineId[*it] =
          tables.addSpline(fnSpline->getRadialSplineObject(),
                           fnSpline->getRadialCutOff(),
                           fnSpline->getrMinVal(),
                           1.0);
      }
    return true;
  }

  /**
   * @brief Sizes a MemoryStorage to match a host vector and copies it in.
   */
  template <typename T, dftfe::utils::MemorySpace memorySpace>
  void
  stageHostVector(const std::vector<T>                        &hostData,
                  dftfe::utils::MemoryStorage<T, memorySpace> &stagedData)
  {
    stagedData.resize(hostData.size());
    if (!hostData.empty())
      stagedData.copyFrom(hostData);
  }

  /**
   * @brief The packed spline tables in the memory space the kernels run in.
   */
  template <dftfe::utils::MemorySpace memorySpace>
  struct AtomTypeSplineTables
  {
    dftfe::utils::MemoryStorage<double, memorySpace> knotX, knotY, coefB, coefC,
      coefD, gpParams, cutoffs, rMins, evalParams;
    dftfe::utils::MemoryStorage<dftfe::uInt, memorySpace> offsets, nKnots;

    void
    stage(const splineHostTableUtils::SplineTableGroup &src)
    {
      stageHostVector(src.knotXAll, knotX);
      stageHostVector(src.knotYAll, knotY);
      stageHostVector(src.coefBAll, coefB);
      stageHostVector(src.coefCAll, coefC);
      stageHostVector(src.coefDAll, coefD);
      stageHostVector(src.gpParams, gpParams);
      stageHostVector(src.cutoffs, cutoffs);
      stageHostVector(src.rMins, rMins);
      stageHostVector(src.evalParams, evalParams);
      stageHostVector(src.offsets, offsets);
      stageHostVector(src.nKnots, nKnots);
    }
  };
} // namespace dftfe
#endif
