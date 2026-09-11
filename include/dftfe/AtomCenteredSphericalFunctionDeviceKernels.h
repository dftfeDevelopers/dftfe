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
// reduction over periodic images with k-point phases, and portable
// (NVIDIA/AMD/Intel, decomposition-independent) random wavefunction
// initialization.
//
// Common conventions: work is organized as (center, cell) pairs, where a
// center is an atom or one of its periodic images; quadCoords and
// jxwCompact are indexed by a compact support-cell slot
// [cellSlot*numQuadPoints(*3)]; radial splines are passed as concatenated
// coefficient tables unpacked from the host alglib objects (see
// splineDeviceEval.h) with per-function metadata arrays.
//

#ifndef DFTFE_ATOMCENTEREDSPHERICALFUNCTIONDEVICEKERNELS_H
#define DFTFE_ATOMCENTEREDSPHERICALFUNCTIONDEVICEKERNELS_H

#include <dftfe/deviceKernelsGeneric.h>
#include <dftfe/DeviceAPICalls.h>
#include <dftfe/DeviceDataTypeOverloads.h>
#include <dftfe/DeviceTypeConfig.h>
#include <dftfe/DeviceKernelLauncherHelpers.h>
#include <dftfe/TypeConfig.h>

namespace dftfe
{
  namespace atomCenteredSphericalFunctionDeviceKernels
  {
    /**
     * @brief Fills a flattened (iNode-major, wavefunction-minor) block of
     * real coefficients with uniform random numbers in [-0.5, 0.5), keyed
     * on (seed, global dof id, wavefunction id, blockKey). Complex arrays
     * are filled as interleaved re/im with numRealsPerCoeff = 2. blockKey
     * should encode the (k-point, spin) block so different blocks get
     * independent streams.
     */
    void
    fillRandomWavefunctions(const dftfe::uInt        numLocalDofs,
                            const dftfe::uInt        numWaveFunctions,
                            const dftfe::uInt        numRealsPerCoeff,
                            const dftfe::uInt        numSpinorComponents,
                            const unsigned long long globalDofStartIndex,
                            const unsigned long long blockKey,
                            const unsigned long long seed,
                            const dftfe::uInt       *constrainedMaskOrNull,
                            double                  *wfcData);

    /**
     * @brief Reduces per-image projector values (from projectorValues)
     * into the k-point-resolved [kpt][cell][lm][q] basis buffer of the
     * nonlocal C-matrix construction, applying the Bloch phase
     * (cos, -sin) of angle = k.(quadPoint - R_image + R_atom) for complex
     * builds (interleaved re/im output); real builds sum with phase 1.
     * When outBufferWithDistance is not null it also receives the same sum
     * weighted by the displacement (quadPoint - R_image) per component, in
     * the [kpt][cell][lm][dim][q] layout used for the cell stress.
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
     * @brief Radial-spline x Y_lm family values (optionally times jxw) per
     * (center, cell) pair for the nonlocal C-matrix construction; output
     * layout [iPair * maxTotalLM * numQuadPoints + lm * numQuadPoints + q]
     * with zero padding for species with fewer than maxTotalLM channels.
     */
    void
    projectorValues(const dftfe::uInt  numPairs,
                    const dftfe::uInt  numQuadPoints,
                    const dftfe::uInt  maxTotalLM,
                    const bool         multiplyJxW,
                    const double      *quadCoords,
                    const double      *jxwCompact,
                    const double      *pairCenters,
                    const dftfe::uInt *pairCellSlot,
                    const dftfe::uInt *pairSpecies,
                    const dftfe::uInt *speciesRadialFnStart,
                    const dftfe::uInt *speciesNumRadialFns,
                    const dftfe::uInt *speciesTotalLM,
                    const dftfe::uInt *radialFnLQuantum,
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
                    const double      *splineEvalParams,
                    double            *valuesOut);
  } // namespace atomCenteredSphericalFunctionDeviceKernels
} // namespace dftfe

#endif // DFTFE_ATOMCENTEREDSPHERICALFUNCTIONDEVICEKERNELS_H
