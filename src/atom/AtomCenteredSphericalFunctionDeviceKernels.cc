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
// Device kernels for batched evaluation of atom-centered radial projector
// splines times real spherical harmonics at finite-element quadrature
// points, their reduction over periodic images with k-point phases, and
// the portable random initialization of wavefunctions. Used by the
// nonlocal C-matrix construction and readPSI.
//
#include <dftfe/AtomCenteredSphericalFunctionDeviceKernels.h>
#include <dftfe/BLASWrapper.h>
#include <dftfe/splineDeviceEval.h>
#include <dftfe/AtomCenteredSphericalFunctionDeviceUtils.h>

namespace dftfe
{

  // -------------------------------------------------------------------
  // Portable random initial wavefunctions: fills a flattened block
  // (iNode-major, wavefunction-minor) of real coefficients (a complex
  // array is filled as interleaved re/im via numRealsPerCoeff = 2).
  // Values are keyed on (seed, global dof id, wavefunction id, block key)
  // so the guess is identical on NVIDIA/AMD/Intel devices and independent
  // of the MPI/GPU decomposition.
  // -------------------------------------------------------------------
  DFTFE_CREATE_KERNEL(
    void,
    fillRandomWavefunctionsKernel,
    {
      const dftfe::uInt numberEntries =
        numLocalDofs * numWaveFunctions * numRealsPerCoeff;
      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += nThreadsPerBlock * nThreadBlock)
        {
          const dftfe::uInt coeffIndex = index / numRealsPerCoeff;
          const dftfe::uInt comp       = index % numRealsPerCoeff;
          const dftfe::uInt iNode      = coeffIndex / numWaveFunctions;
          const dftfe::uInt iWave      = coeffIndex % numWaveFunctions;
          // replicate the host initial-guess semantics: only the real part
          // is randomized (imaginary components stay zero) and constrained
          // dofs stay zero
          double value = 0.0;
          if (comp == 0 &&
              (constrainedMaskOrNull == nullptr ||
               constrainedMaskOrNull[iNode / numSpinorComponents] == 0))
            {
              const double u1 =
                atomCenteredSphericalFunctionDeviceUtils::uniformRandomCentered(
                  seed,
                  static_cast<unsigned long long>(globalDofStartIndex + iNode),
                  static_cast<unsigned long long>(iWave),
                  2ULL * blockKey);
              const double u2 =
                atomCenteredSphericalFunctionDeviceUtils::uniformRandomCentered(
                  seed,
                  static_cast<unsigned long long>(globalDofStartIndex + iNode),
                  static_cast<unsigned long long>(iWave),
                  2ULL * blockKey + 1ULL);
              // z uniform in [-1.5, 1.5); value = standard normal pdf(z)
              // with a random sign, matching the host generator
              const double z = 3.0 * u1;
              value          = std::exp(-0.5 * z * z) * 0.3989422804014327 *
                      (u2 < 0.0 ? -1.0 : 1.0);
            }
          wfcData[index] = value;
        }
    },
    const dftfe::uInt        numLocalDofs,
    const dftfe::uInt        numWaveFunctions,
    const dftfe::uInt        numRealsPerCoeff,
    const dftfe::uInt        numSpinorComponents,
    const unsigned long long globalDofStartIndex,
    const unsigned long long blockKey,
    const unsigned long long seed,
    const dftfe::uInt       *constrainedMaskOrNull,
    double                  *wfcData);


  // -------------------------------------------------------------------
  // Projector (or any radial-spline x Y_lm family) values times jxw at the
  // quadrature points of each (center, cell) pair, for the nonlocal
  // C-matrix construction:
  //   out[iPair][alpha_lm][q] = p_alpha(r) Y_lm(theta,phi) * (multJxW ?
  //   jxw[q] : 1)
  // The lm channels of one species are described by contiguous
  // (splineId, l) entries; m runs over -l..l inside.
  // -------------------------------------------------------------------
  DFTFE_CREATE_KERNEL(
    void,
    projectorValuesKernel,
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
          const double x0 = qp[0] - pairCenters[iPair * 3 + 0];
          const double x1 = qp[1] - pairCenters[iPair * 3 + 1];
          const double x2 = qp[2] - pairCenters[iPair * 3 + 2];
          double       r;
          double       theta;
          double       phi;
          atomCenteredSphericalFunctionDeviceUtils::cartesianToSpherical(
            x0, x1, x2, r, theta, phi);
          const dftfe::uInt spec         = pairSpecies[iPair];
          const dftfe::uInt radFnStart   = speciesRadialFnStart[spec];
          const dftfe::uInt numRadialFns = speciesNumRadialFns[spec];
          const dftfe::uInt totalLM      = speciesTotalLM[spec];
          const double      jxwVal =
            multiplyJxW ? jxwCompact[cellSlot * numQuadPoints + iQuad] : 1.0;
          double *out = valuesOut + iPair * maxTotalLM * numQuadPoints + iQuad;
          dftfe::uInt lmIndex = 0;
          for (dftfe::uInt alpha = 0; alpha < numRadialFns; alpha++)
            {
              const dftfe::uInt fnId = radFnStart + alpha;
              const int         l    = static_cast<int>(radialFnLQuantum[fnId]);
              double            radialVal = 0.0;
              if (r < splineCutoff[fnId])
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
                                           fnId);
                  if (splineEvalParams[2 * fnId] > 0.5)
                    {
                      // AtomCenteredSphericalFunctionPAWProjectorSpline2
                      // semantics: spline fit of r*f(r); y0 within 1e-8
                      // of rMin; spline(r)/r otherwise (no rMin clamp,
                      // segment-0 extrapolation below the first knot
                      // matches alglib::spline1dcalc)
                      double diff = r - splineRMin[fnId];
                      if (diff < 0.0)
                        diff = -diff;
                      if (diff <= 1e-8)
                        radialVal = splineEvalParams[2 * fnId + 1];
                      else
                        radialVal = splineEval::splineEvalValue(t, r) / r;
                    }
                  else
                    {
                      const double rEval =
                        r <= splineRMin[fnId] ? splineRMin[fnId] : r;
                      radialVal = splineEval::splineEvalValue(t, rEval);
                    }
                }
              for (int m = -l; m <= l; m++)
                {
                  const double Y = atomCenteredSphericalFunctionDeviceUtils::
                    realSphericalHarmonic(l, m, theta, phi);
                  out[lmIndex * numQuadPoints] = radialVal * Y * jxwVal;
                  lmIndex++;
                }
            }
          for (; lmIndex < maxTotalLM; lmIndex++)
            out[lmIndex * numQuadPoints] = 0.0;
        }
    },
    const dftfe::uInt  numPairs,
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
    const double      *splineEvalParams, // [fn*2] {mode, y0}
    double            *valuesOut);


  // -------------------------------------------------------------------
  // Reduces per-image projector values into the k-point-resolved
  // (cell, lm, q) basis buffer used by the nonlocal C-matrix
  // construction:
  //   out[kpt][cell][lm][q] += phase(kpt, img, q) * val[pair][lm][q]
  // with phase = (cos(angle), -sin(angle)), angle = k . (x + R_atom),
  // x = quadPoint - R_image (matching the USE_COMPLEX host convention).
  // For real builds (isComplex = false) the phase is 1 and the output
  // stride per entry is one double instead of two.
  // -------------------------------------------------------------------
  DFTFE_CREATE_KERNEL(
    void,
    reduceImageContributionsWithPhasesKernel,
    {
      const dftfe::uInt numberEntries = numPairs * numQuadPoints;
      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += nThreadsPerBlock * nThreadBlock)
        {
          const dftfe::uInt iPair    = index / numQuadPoints;
          const dftfe::uInt iQuad    = index % numQuadPoints;
          const dftfe::uInt cellSlot = pairCellSlot[iPair];
          // quadCoords and jxwCompact are indexed over the atom's whole
          // support; the output buffer holds only the current cell block
          const dftfe::uInt outCellSlot = cellSlot - cellSlotOffset;
          const double     *qp =
            quadCoords + (cellSlot * numQuadPoints + iQuad) * 3;
          // x = quadPoint - R_image; the phase uses x + R_atom
          DFTFE_KERNEL_ARGUMENT(
            const double xd[3] = {qp[0] - pairCenters[iPair * 3 + 0],
                                  qp[1] - pairCenters[iPair * 3 + 1],
                                  qp[2] - pairCenters[iPair * 3 + 2]};)
          const double  px = xd[0] + atomCoordinates[0];
          const double  py = xd[1] + atomCoordinates[1];
          const double  pz = xd[2] + atomCoordinates[2];
          const double *val =
            valuesIn + iPair * totalLM * numQuadPoints + iQuad;
          if (isComplex)
            {
              for (dftfe::uInt kPoint = 0; kPoint < numKPoints; kPoint++)
                {
                  const double angle = kPointCoordinates[3 * kPoint + 0] * px +
                                       kPointCoordinates[3 * kPoint + 1] * py +
                                       kPointCoordinates[3 * kPoint + 2] * pz;
                  const double cosA = std::cos(angle);
                  const double sinA = -std::sin(angle);
                  double      *outBase =
                    outBuffer +
                    2 * (kPoint * numCells * totalLM * numQuadPoints +
                         outCellSlot * totalLM * numQuadPoints + iQuad);
                  for (dftfe::uInt lm = 0; lm < totalLM; lm++)
                    {
                      const double v = val[lm * numQuadPoints];
                      dftfe::utils::atomicAddWrapper(
                        &outBase[2 * lm * numQuadPoints], cosA * v);
                      dftfe::utils::atomicAddWrapper(
                        &outBase[2 * lm * numQuadPoints + 1], sinA * v);
                    }
                  if (outBufferWithDistance != nullptr)
                    {
                      double *outDist =
                        outBufferWithDistance +
                        2 * (kPoint * numCells * totalLM * numQuadPoints * 3 +
                             outCellSlot * totalLM * numQuadPoints * 3 + iQuad);
                      for (dftfe::uInt lm = 0; lm < totalLM; lm++)
                        {
                          const double v = val[lm * numQuadPoints];
                          for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                            {
                              const dftfe::uInt o =
                                2 *
                                (lm * numQuadPoints * 3 + iDim * numQuadPoints);
                              dftfe::utils::atomicAddWrapper(&outDist[o],
                                                             cosA * v *
                                                               xd[iDim]);
                              dftfe::utils::atomicAddWrapper(&outDist[o + 1],
                                                             sinA * v *
                                                               xd[iDim]);
                            }
                        }
                    }
                }
            }
          else
            {
              double *outBase =
                outBuffer + outCellSlot * totalLM * numQuadPoints + iQuad;
              for (dftfe::uInt lm = 0; lm < totalLM; lm++)
                dftfe::utils::atomicAddWrapper(&outBase[lm * numQuadPoints],
                                               val[lm * numQuadPoints]);
              if (outBufferWithDistance != nullptr)
                {
                  double *outDist = outBufferWithDistance +
                                    outCellSlot * totalLM * numQuadPoints * 3 +
                                    iQuad;
                  for (dftfe::uInt lm = 0; lm < totalLM; lm++)
                    {
                      const double v = val[lm * numQuadPoints];
                      for (dftfe::uInt iDim = 0; iDim < 3; iDim++)
                        dftfe::utils::atomicAddWrapper(
                          &outDist[lm * numQuadPoints * 3 +
                                   iDim * numQuadPoints],
                          v * xd[iDim]);
                    }
                }
            }
        }
    },
    const dftfe::uInt  numPairs,
    const dftfe::uInt  numQuadPoints,
    const dftfe::uInt  totalLM,
    const dftfe::uInt  numCells,
    const dftfe::uInt  cellSlotOffset,
    const dftfe::uInt  numKPoints,
    const bool         isComplex,
    const double      *quadCoords,
    const double      *pairCenters,
    const dftfe::uInt *pairCellSlot,
    const double      *atomCoordinates,   // [3]
    const double      *kPointCoordinates, // [numKPoints*3]
    const double      *valuesIn,
    double            *outBuffer,
    double            *outBufferWithDistance);


  namespace atomCenteredSphericalFunctionDeviceKernels
  {
    void
    fillRandomWavefunctions(const dftfe::uInt        numLocalDofs,
                            const dftfe::uInt        numWaveFunctions,
                            const dftfe::uInt        numRealsPerCoeff,
                            const dftfe::uInt        numSpinorComponents,
                            const unsigned long long globalDofStartIndex,
                            const unsigned long long blockKey,
                            const unsigned long long seed,
                            const dftfe::uInt       *constrainedMaskOrNull,
                            double                  *wfcData)
    {
      const dftfe::uInt numberEntries =
        numLocalDofs * numWaveFunctions * numRealsPerCoeff;
      DFTFE_LAUNCH_KERNEL(fillRandomWavefunctionsKernel,
                          (dftfe::utils::DEVICE_BLOCK_SIZE + numberEntries) /
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::linearAlgebra::BLASWrapper<
                            dftfe::utils::MemorySpace::DEVICE>::d_streamId,
                          numLocalDofs,
                          numWaveFunctions,
                          numRealsPerCoeff,
                          numSpinorComponents,
                          globalDofStartIndex,
                          blockKey,
                          seed,
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            constrainedMaskOrNull),
                          dftfe::utils::makeDataTypeDeviceCompatible(wfcData));
    }

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
                                       double            *outBufferWithDistance)
    {
      const dftfe::uInt numberEntries = numPairs * numQuadPoints;
      if (numberEntries == 0)
        return;
      DFTFE_LAUNCH_KERNEL(
        reduceImageContributionsWithPhasesKernel,
        (dftfe::utils::DEVICE_BLOCK_SIZE + numberEntries) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::linearAlgebra::BLASWrapper<
          dftfe::utils::MemorySpace::DEVICE>::d_streamId,
        numPairs,
        numQuadPoints,
        totalLM,
        numCells,
        cellSlotOffset,
        numKPoints,
        isComplex,
        dftfe::utils::makeDataTypeDeviceCompatible(quadCoords),
        dftfe::utils::makeDataTypeDeviceCompatible(pairCenters),
        dftfe::utils::makeDataTypeDeviceCompatible(pairCellSlot),
        dftfe::utils::makeDataTypeDeviceCompatible(atomCoordinates),
        dftfe::utils::makeDataTypeDeviceCompatible(kPointCoordinates),
        dftfe::utils::makeDataTypeDeviceCompatible(valuesIn),
        dftfe::utils::makeDataTypeDeviceCompatible(outBuffer),
        dftfe::utils::makeDataTypeDeviceCompatible(outBufferWithDistance));
    }

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
                    double            *valuesOut)
    {
      const dftfe::uInt numberEntries = numPairs * numQuadPoints;
      DFTFE_LAUNCH_KERNEL(
        projectorValuesKernel,
        (dftfe::utils::DEVICE_BLOCK_SIZE + numberEntries) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::linearAlgebra::BLASWrapper<
          dftfe::utils::MemorySpace::DEVICE>::d_streamId,
        numPairs,
        numQuadPoints,
        maxTotalLM,
        multiplyJxW,
        dftfe::utils::makeDataTypeDeviceCompatible(quadCoords),
        dftfe::utils::makeDataTypeDeviceCompatible(jxwCompact),
        dftfe::utils::makeDataTypeDeviceCompatible(pairCenters),
        dftfe::utils::makeDataTypeDeviceCompatible(pairCellSlot),
        dftfe::utils::makeDataTypeDeviceCompatible(pairSpecies),
        dftfe::utils::makeDataTypeDeviceCompatible(speciesRadialFnStart),
        dftfe::utils::makeDataTypeDeviceCompatible(speciesNumRadialFns),
        dftfe::utils::makeDataTypeDeviceCompatible(speciesTotalLM),
        dftfe::utils::makeDataTypeDeviceCompatible(radialFnLQuantum),
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
        dftfe::utils::makeDataTypeDeviceCompatible(splineEvalParams),
        dftfe::utils::makeDataTypeDeviceCompatible(valuesOut));
    }
  } // namespace atomCenteredSphericalFunctionDeviceKernels
} // namespace dftfe
