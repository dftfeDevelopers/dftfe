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
// Launchers for the device kernels in
// src/atom/AtomCenteredSphericalFunctionDeviceKernels.cc: batched
// evaluation of atom-centered radial projector splines times real
// spherical harmonics at finite-element quadrature points, their
// reduction over periodic images with k-point phases.
//
// Common conventions: work is organized as (center, cell) pairs, where a
// center is an atom or one of its periodic images; quadCoords and
// jxwCompact are indexed by a compact support-cell slot
// [cellSlot*numQuadPoints(*3)]; radial splines are passed as a
// splineEval::SplineTableGroupView of the tables unpacked from the host
// alglib objects.
//

#ifndef DFTFE_ATOMCENTEREDSPHERICALFUNCTIONDEVICEKERNELS_H
#define DFTFE_ATOMCENTEREDSPHERICALFUNCTIONDEVICEKERNELS_H

#include <dftfe/deviceKernelsGeneric.h>
#include <dftfe/DeviceAPICalls.h>
#include <dftfe/DeviceDataTypeOverloads.h>
#include <dftfe/DeviceTypeConfig.h>
#include <dftfe/DeviceKernelLauncherHelpers.h>
#include <dftfe/TypeConfig.h>
#include <dftfe/splineDeviceEval.h>

namespace dftfe
{
  namespace atomCenteredSphericalFunctionDeviceKernels
  {
    /**
     * @brief Reduces per-image projector values (from projectorValues)
     * into the k-point-resolved [kpt][cell][lm][q] basis buffer of the
     * nonlocal C-matrix construction, applying for complex builds
     * (interleaved re/im output) the Bloch phase
     * @code
     *   (cos(angle), -sin(angle)),  angle = k . (quadPoint - R_image + R_atom)
     * @endcode
     * while real builds sum with phase 1.
     * When outBufferWithDistance is not null it also receives the same sum
     * weighted by the displacement (quadPoint - R_image) per component, in
     * the [kpt][cell][lm][dim][q] layout used for the cell stress. The
     * outputs hold the numCells cells starting at slot cellSlotOffset and
     * are accumulated into, so the caller zeroes them first.
     */
    void
    reduceImageContributionsWithPhases(const dftfe::uInt  numPairs,
                                       const dftfe::uInt  numQuadPoints,
                                       const dftfe::uInt  totalLM,
                                       const dftfe::uInt  numCells,
                                       const dftfe::uInt  cellSlotOffset,
                                       const dftfe::uInt  numKPoints,
                                       const bool         isComplex,
                                       const double      *quadCoords,
                                       const double      *pairCenters,
                                       const dftfe::uInt *pairCellSlot,
                                       const double      *atomCoordinates,
                                       const double      *kPointCoordinates,
                                       const double      *valuesIn,
                                       double            *outBuffer,
                                       double *outBufferWithDistance);

    /**
     * @brief Projector values times jxw at every quadrature point of every
     * (center, cell) pair of one species, for the nonlocal C-matrix
     * construction.
     *
     * @param[in] numPairs number of (center, cell) pairs
     * @param[in] numQuadPoints quadrature points per cell
     * @param[in] numRadialFunctions radial functions of the species, table
     * entries 0..numRadialFunctions-1
     * @param[in] totalLM sum of 2l+1 over the radial functions
     * @param[in] quadCoords quadrature points, [cellSlot][q][3]
     * @param[in] jxwCompact quadrature weights, [cellSlot][q]
     * @param[in] pairCenters center of each pair, [iPair][3]
     * @param[in] pairCellSlot cell slot of each pair
     * @param[in] radialFnLQuantum angular momentum of each radial function
     * @param[in] tables the radial splines
     * @param[out] valuesOut [iPair][lm][q], radial function alpha filling its
     * channels m = -l..l in order
     */
    void
    projectorValues(const dftfe::uInt                       numPairs,
                    const dftfe::uInt                       numQuadPoints,
                    const dftfe::uInt                       numRadialFunctions,
                    const dftfe::uInt                       totalLM,
                    const double                           *quadCoords,
                    const double                           *jxwCompact,
                    const double                           *pairCenters,
                    const dftfe::uInt                      *pairCellSlot,
                    const dftfe::uInt                      *radialFnLQuantum,
                    const splineEval::SplineTableGroupView &tables,
                    double                                 *valuesOut);
  } // namespace atomCenteredSphericalFunctionDeviceKernels
} // namespace dftfe

#endif // DFTFE_ATOMCENTEREDSPHERICALFUNCTIONDEVICEKERNELS_H
