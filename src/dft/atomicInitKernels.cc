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

namespace dftfe
{
  namespace atomicInitKernels
  {
    template <>
    void
    atomicWavefunctionSuperposition<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt                       numPoints,
      const dftfe::uInt                       numOrbitals,
      const bool                              isNoncolin,
      const double                            truncationRadius,
      const double                           *pointCoords,
      const dftfe::uInt                      *orbitalCenterStart,
      const double                           *centerCoords,
      const dftfe::Int                       *orbitalL,
      const dftfe::Int                       *orbitalM,
      const dftfe::uInt                      *orbitalSplineId,
      const splineEval::SplineTableGroupView &tables,
      double                                 *valueOut,
      const dftfe::uInt                       numHostThreads)
    {
      const dftfe::uInt numberEntries = numPoints * numOrbitals;
#pragma omp        parallel for num_threads(numHostThreads)
      for (dftfe::uInt index = 0; index < numberEntries; ++index)
        valueOut[index] = atomicWavefunctionEntry(index,
                                                  numOrbitals,
                                                  isNoncolin,
                                                  truncationRadius,
                                                  pointCoords,
                                                  orbitalCenterStart,
                                                  centerCoords,
                                                  orbitalL,
                                                  orbitalM,
                                                  orbitalSplineId,
                                                  tables);
    }

    template <>
    void
    localPseudoPotentialValues<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt                       numPairs,
      const dftfe::uInt                       numQuadPoints,
      const double                            cutOffForPsp,
      const double                           *quadCoords,
      const double                           *pairCenters,
      const double                           *pairCharges,
      const dftfe::uInt                      *pairCellSlot,
      const dftfe::uInt                      *pairSplineId,
      const splineEval::SplineTableGroupView &tables,
      double                                 *valueOut,
      dftfe::uInt                            *pairHasSupport,
      const dftfe::uInt                       numHostThreads)
    {
      const dftfe::uInt numberEntries = numPairs * numQuadPoints;
#pragma omp parallel for num_threads(numHostThreads)
      for (dftfe::uInt index = 0; index < numberEntries; ++index)
        {
          double r;
          valueOut[index] = localPseudoPotentialPairEntry(index,
                                                          numQuadPoints,
                                                          quadCoords,
                                                          pairCenters,
                                                          pairCharges,
                                                          pairCellSlot,
                                                          pairSplineId,
                                                          tables,
                                                          r);
          if (r <= cutOffForPsp)
            {
              // quadrature points of one pair share this flag
#pragma omp atomic write
              pairHasSupport[index / numQuadPoints] = 1;
            }
        }
    }

    template <>
    void
    localPseudoPotentialTotal<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt                       numCells,
      const dftfe::uInt                       numQuadPoints,
      const dftfe::uInt                       numCenters,
      const bool                              analyticSmearedCharge,
      const double                            maxTail,
      const double                           *quadCoords,
      const double                           *centerCoords,
      const double                           *centerCharges,
      const double                           *centerTailCutoff,
      const double                           *centerSmearedWidth,
      const dftfe::uInt                      *centerSplineId,
      const splineEval::SplineTableGroupView &tables,
      double                                 *valueOut,
      const dftfe::uInt                       numHostThreads)
    {
      const dftfe::uInt numberEntries = numCells * numQuadPoints;
#pragma omp        parallel for num_threads(numHostThreads)
      for (dftfe::uInt index = 0; index < numberEntries; ++index)
        valueOut[index] = localPseudoPotentialTotalEntry(index,
                                                         numCenters,
                                                         analyticSmearedCharge,
                                                         maxTail,
                                                         quadCoords,
                                                         centerCoords,
                                                         centerCharges,
                                                         centerTailCutoff,
                                                         centerSmearedWidth,
                                                         centerSplineId,
                                                         tables);
    }

    template <>
    void
    atomicRadialDensityQuadTotal<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt                       numCells,
      const dftfe::uInt                       numQuadPoints,
      const dftfe::uInt                       numCenters,
      const bool                              floatingNuclearCharges,
      const double                           *quadCoords,
      const double                           *centerCoords,
      const dftfe::uInt                      *centerSplineId,
      const splineEval::SplineTableGroupView &tables,
      double                                 *valueOut,
      double                                 *gradOut,
      const dftfe::uInt                       numHostThreads)
    {
      const dftfe::uInt numberEntries = numCells * numQuadPoints;
#pragma omp parallel for num_threads(numHostThreads)
      for (dftfe::uInt index = 0; index < numberEntries; ++index)
        radialDensityQuadTotalEntry(index,
                                    numCenters,
                                    floatingNuclearCharges,
                                    quadCoords,
                                    centerCoords,
                                    centerSplineId,
                                    tables,
                                    valueOut,
                                    gradOut);
    }

    template <>
    void
    atomicRadialDensityPairDerivatives<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt                       numPairs,
      const dftfe::uInt                       numQuadPoints,
      const bool                              floatingNuclearCharges,
      const double                           *quadCoords,
      const double                           *pairCenters,
      const dftfe::uInt                      *pairCellSlot,
      const dftfe::uInt                      *pairSplineId,
      const splineEval::SplineTableGroupView &tables,
      double                                 *gradOut,
      double                                 *hessOut,
      dftfe::uInt                            *pairHasSupport,
      const dftfe::uInt                       numHostThreads)
    {
      const dftfe::uInt numberEntries = numPairs * numQuadPoints;
#pragma omp parallel for num_threads(numHostThreads)
      for (dftfe::uInt index = 0; index < numberEntries; ++index)
        if (radialDensityPairEntry(index,
                                   numQuadPoints,
                                   floatingNuclearCharges,
                                   quadCoords,
                                   pairCenters,
                                   pairCellSlot,
                                   pairSplineId,
                                   tables,
                                   gradOut,
                                   hessOut))
          {
            // quadrature points of one pair share this flag
#pragma omp atomic write
            pairHasSupport[index / numQuadPoints] = 1;
          }
    }

    template <>
    void
    atomicDensityQuadSuperposition<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt                       numPoints,
      const dftfe::uInt                       numCenters,
      const dftfe::uInt                       numMagComponents,
      const bool                              floatingNuclearCharges,
      const bool                              applyConstraintMag,
      const double                            totMagnetization,
      const double                           *pointCoords,
      const double                           *centerCoords,
      const dftfe::uInt                      *centerSplineId,
      const double                           *centerRhoFactor,
      const double                           *centerMagZFactor,
      const double                           *centerMagYFactor,
      const double                           *centerMagXFactor,
      const splineEval::SplineTableGroupView &tables,
      double                                 *rhoOut,
      double                                 *magZOut,
      double                                 *magYOut,
      double                                 *magXOut,
      double                                 *gradRhoOut,
      double                                 *gradMagZOut,
      double                                 *gradMagYOut,
      double                                 *gradMagXOut,
      const dftfe::uInt                       numHostThreads)
    {
#pragma omp parallel for num_threads(numHostThreads)
      for (dftfe::uInt index = 0; index < numPoints; ++index)
        densitySuperpositionQuadPoint(index,
                                      pointCoords,
                                      numCenters,
                                      numMagComponents,
                                      floatingNuclearCharges,
                                      applyConstraintMag,
                                      totMagnetization,
                                      centerCoords,
                                      centerSplineId,
                                      centerRhoFactor,
                                      centerMagZFactor,
                                      centerMagYFactor,
                                      centerMagXFactor,
                                      tables,
                                      rhoOut,
                                      magZOut,
                                      magYOut,
                                      magXOut,
                                      gradRhoOut,
                                      gradMagZOut,
                                      gradMagYOut,
                                      gradMagXOut);
    }

    template <>
    void
    fillRandomWavefunctions<dftfe::utils::MemorySpace::HOST>(
      const dftfe::uInt        numLocalNodes,
      const dftfe::uInt        numWaveFunctions,
      const dftfe::uInt        firstWave,
      const dftfe::uInt        numRealsPerCoeff,
      const dftfe::uInt        numSpinorComponents,
      const unsigned long long globalNodeStartIndex,
      const unsigned long long seed,
      const dftfe::uInt       *constrainedMask,
      double                  *wfcData,
      const dftfe::uInt        numHostThreads)
    {
      if (firstWave >= numWaveFunctions)
        return;
      const dftfe::uInt numberEntries =
        numLocalNodes * (numWaveFunctions - firstWave) * numRealsPerCoeff;
#pragma omp parallel for num_threads(numHostThreads)
      for (dftfe::uInt index = 0; index < numberEntries; ++index)
        randomWavefunctionEntry(index,
                                numWaveFunctions,
                                firstWave,
                                numRealsPerCoeff,
                                numSpinorComponents,
                                globalNodeStartIndex,
                                seed,
                                constrainedMask,
                                wfcData);
    }
  } // namespace atomicInitKernels
} // namespace dftfe
