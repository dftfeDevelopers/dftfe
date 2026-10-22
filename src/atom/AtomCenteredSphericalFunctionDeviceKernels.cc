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
// points and their reduction over periodic images with k-point phases.
// Used by the nonlocal C-matrix construction.
//
#include <dftfe/AtomCenteredSphericalFunctionDeviceKernels.h>
#include <dftfe/BLASWrapper.h>
#include <dftfe/splineDeviceEval.h>
#include <dftfe/AtomCenteredSphericalFunctionDeviceUtils.h>

namespace dftfe
{

  // -------------------------------------------------------------------
  // Projector values times jxw at the quadrature points of each
  // (center, cell) pair of one species, for the nonlocal C-matrix
  // construction:
  //   out[iPair][alpha_lm][q] = p_alpha(r) Y_lm(theta, phi) jxw[q]
  // Radial function alpha has angular momentum radialFnLQuantum[alpha]
  // and contributes its 2l+1 channels m = -l..l in order.
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
          const double jxwVal = jxwCompact[cellSlot * numQuadPoints + iQuad];
          double     *out = valuesOut + iPair * totalLM * numQuadPoints + iQuad;
          dftfe::uInt lmIndex = 0;
          for (dftfe::uInt alpha = 0; alpha < numRadialFunctions; alpha++)
            {
              const int    l = static_cast<int>(radialFnLQuantum[alpha]);
              const double radialVal =
                splineEval::radialSplineValue(tables, alpha, r);
              for (int m = -l; m <= l; m++)
                {
                  const double Y = atomCenteredSphericalFunctionDeviceUtils::
                    realSphericalHarmonic(l, m, theta, phi);
                  out[lmIndex * numQuadPoints] = radialVal * Y * jxwVal;
                  lmIndex++;
                }
            }
        }
    },
    const dftfe::uInt                      numPairs,
    const dftfe::uInt                      numQuadPoints,
    const dftfe::uInt                      numRadialFunctions,
    const dftfe::uInt                      totalLM,
    const double                          *quadCoords,
    const double                          *jxwCompact,
    const double                          *pairCenters,
    const dftfe::uInt                     *pairCellSlot,
    const dftfe::uInt                     *radialFnLQuantum,
    const splineEval::SplineTableGroupView tables,
    double                                *valuesOut);


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
                    double                                 *valuesOut)
    {
      const dftfe::uInt numberEntries = numPairs * numQuadPoints;
      if (numberEntries == 0)
        return;
      DFTFE_LAUNCH_KERNEL(projectorValuesKernel,
                          (dftfe::utils::DEVICE_BLOCK_SIZE + numberEntries) /
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::linearAlgebra::BLASWrapper<
                            dftfe::utils::MemorySpace::DEVICE>::d_streamId,
                          numPairs,
                          numQuadPoints,
                          numRadialFunctions,
                          totalLM,
                          quadCoords,
                          jxwCompact,
                          pairCenters,
                          pairCellSlot,
                          radialFnLQuantum,
                          tables,
                          valuesOut);
    }
  } // namespace atomCenteredSphericalFunctionDeviceKernels
} // namespace dftfe
