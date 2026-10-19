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
#include <dftfe/BLASWrapper.h>
#include <dftfe/DeviceAPICalls.h>
#include <dftfe/DeviceDataTypeOverloads.h>
#include <dftfe/DeviceTypeConfig.h>
#include <dftfe/DeviceKernelLauncherHelpers.h>

namespace dftfe
{
  namespace atomicInitKernels
  {
    DFTFE_CREATE_KERNEL(
      void,
      atomicWavefunctionSuperpositionKernel,
      {
        const dftfe::uInt numberEntries = numPoints * numOrbitals;
        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
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
      },
      const dftfe::uInt                      numPoints,
      const dftfe::uInt                      numOrbitals,
      const bool                             isNoncolin,
      const double                           truncationRadius,
      const double                          *pointCoords,
      const dftfe::uInt                     *orbitalCenterStart,
      const double                          *centerCoords,
      const dftfe::Int                      *orbitalL,
      const dftfe::Int                      *orbitalM,
      const dftfe::uInt                     *orbitalSplineId,
      const splineEval::SplineTableGroupView tables,
      double                                *valueOut);

    template <>
    void
    atomicWavefunctionSuperposition<dftfe::utils::MemorySpace::DEVICE>(
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
      const dftfe::uInt)
    {
      const dftfe::uInt numberEntries = numPoints * numOrbitals;
      if (numberEntries == 0)
        return;
      DFTFE_LAUNCH_KERNEL(atomicWavefunctionSuperpositionKernel,
                          (dftfe::utils::DEVICE_BLOCK_SIZE + numberEntries) /
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::linearAlgebra::BLASWrapper<
                            dftfe::utils::MemorySpace::DEVICE>::d_streamId,
                          numPoints,
                          numOrbitals,
                          isNoncolin,
                          truncationRadius,
                          pointCoords,
                          orbitalCenterStart,
                          centerCoords,
                          orbitalL,
                          orbitalM,
                          orbitalSplineId,
                          tables,
                          valueOut);
    }

    DFTFE_CREATE_KERNEL(
      void,
      localPseudoPotentialValuesKernel,
      {
        const dftfe::uInt numberEntries = numPairs * numQuadPoints;
        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
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
              pairHasSupport[index / numQuadPoints] = 1;
          }
      },
      const dftfe::uInt                      numPairs,
      const dftfe::uInt                      numQuadPoints,
      const double                           cutOffForPsp,
      const double                          *quadCoords,
      const double                          *pairCenters,
      const double                          *pairCharges,
      const dftfe::uInt                     *pairCellSlot,
      const dftfe::uInt                     *pairSplineId,
      const splineEval::SplineTableGroupView tables,
      double                                *valueOut,
      dftfe::uInt                           *pairHasSupport);

    template <>
    void
    localPseudoPotentialValues<dftfe::utils::MemorySpace::DEVICE>(
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
      const dftfe::uInt)
    {
      const dftfe::uInt numberEntries = numPairs * numQuadPoints;
      if (numberEntries == 0)
        return;
      DFTFE_LAUNCH_KERNEL(localPseudoPotentialValuesKernel,
                          (dftfe::utils::DEVICE_BLOCK_SIZE + numberEntries) /
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::linearAlgebra::BLASWrapper<
                            dftfe::utils::MemorySpace::DEVICE>::d_streamId,
                          numPairs,
                          numQuadPoints,
                          cutOffForPsp,
                          quadCoords,
                          pairCenters,
                          pairCharges,
                          pairCellSlot,
                          pairSplineId,
                          tables,
                          valueOut,
                          pairHasSupport);
    }

    DFTFE_CREATE_KERNEL(
      void,
      localPseudoPotentialTotalKernel,
      {
        const dftfe::uInt numberEntries = numCells * numQuadPoints;
        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          valueOut[index] =
            localPseudoPotentialTotalEntry(index,
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
      },
      const dftfe::uInt                      numCells,
      const dftfe::uInt                      numQuadPoints,
      const dftfe::uInt                      numCenters,
      const bool                             analyticSmearedCharge,
      const double                           maxTail,
      const double                          *quadCoords,
      const double                          *centerCoords,
      const double                          *centerCharges,
      const double                          *centerTailCutoff,
      const double                          *centerSmearedWidth,
      const dftfe::uInt                     *centerSplineId,
      const splineEval::SplineTableGroupView tables,
      double                                *valueOut);

    template <>
    void
    localPseudoPotentialTotal<dftfe::utils::MemorySpace::DEVICE>(
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
      const dftfe::uInt)
    {
      const dftfe::uInt numberEntries = numCells * numQuadPoints;
      if (numberEntries == 0)
        return;
      DFTFE_LAUNCH_KERNEL(localPseudoPotentialTotalKernel,
                          (dftfe::utils::DEVICE_BLOCK_SIZE + numberEntries) /
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::linearAlgebra::BLASWrapper<
                            dftfe::utils::MemorySpace::DEVICE>::d_streamId,
                          numCells,
                          numQuadPoints,
                          numCenters,
                          analyticSmearedCharge,
                          maxTail,
                          quadCoords,
                          centerCoords,
                          centerCharges,
                          centerTailCutoff,
                          centerSmearedWidth,
                          centerSplineId,
                          tables,
                          valueOut);
    }

    DFTFE_CREATE_KERNEL(
      void,
      atomicRadialDensityQuadTotalKernel,
      {
        const dftfe::uInt numberEntries = numCells * numQuadPoints;
        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          radialDensityQuadTotalEntry(index,
                                      numCenters,
                                      floatingNuclearCharges,
                                      quadCoords,
                                      centerCoords,
                                      centerSplineId,
                                      tables,
                                      valueOut,
                                      gradOut);
      },
      const dftfe::uInt                      numCells,
      const dftfe::uInt                      numQuadPoints,
      const dftfe::uInt                      numCenters,
      const bool                             floatingNuclearCharges,
      const double                          *quadCoords,
      const double                          *centerCoords,
      const dftfe::uInt                     *centerSplineId,
      const splineEval::SplineTableGroupView tables,
      double                                *valueOut,
      double                                *gradOut);

    template <>
    void
    atomicRadialDensityQuadTotal<dftfe::utils::MemorySpace::DEVICE>(
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
      const dftfe::uInt)
    {
      const dftfe::uInt numberEntries = numCells * numQuadPoints;
      if (numberEntries == 0)
        return;
      DFTFE_LAUNCH_KERNEL(atomicRadialDensityQuadTotalKernel,
                          (dftfe::utils::DEVICE_BLOCK_SIZE + numberEntries) /
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::linearAlgebra::BLASWrapper<
                            dftfe::utils::MemorySpace::DEVICE>::d_streamId,
                          numCells,
                          numQuadPoints,
                          numCenters,
                          floatingNuclearCharges,
                          quadCoords,
                          centerCoords,
                          centerSplineId,
                          tables,
                          valueOut,
                          gradOut);
    }

    DFTFE_CREATE_KERNEL(
      void,
      atomicRadialDensityPairDerivativesKernel,
      {
        const dftfe::uInt numberEntries = numPairs * numQuadPoints;
        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
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
            pairHasSupport[index / numQuadPoints] = 1;
      },
      const dftfe::uInt                      numPairs,
      const dftfe::uInt                      numQuadPoints,
      const bool                             floatingNuclearCharges,
      const double                          *quadCoords,
      const double                          *pairCenters,
      const dftfe::uInt                     *pairCellSlot,
      const dftfe::uInt                     *pairSplineId,
      const splineEval::SplineTableGroupView tables,
      double                                *gradOut,
      double                                *hessOut,
      dftfe::uInt                           *pairHasSupport);

    template <>
    void
    atomicRadialDensityPairDerivatives<dftfe::utils::MemorySpace::DEVICE>(
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
      const dftfe::uInt)
    {
      const dftfe::uInt numberEntries = numPairs * numQuadPoints;
      if (numberEntries == 0)
        return;
      DFTFE_LAUNCH_KERNEL(atomicRadialDensityPairDerivativesKernel,
                          (dftfe::utils::DEVICE_BLOCK_SIZE + numberEntries) /
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::linearAlgebra::BLASWrapper<
                            dftfe::utils::MemorySpace::DEVICE>::d_streamId,
                          numPairs,
                          numQuadPoints,
                          floatingNuclearCharges,
                          quadCoords,
                          pairCenters,
                          pairCellSlot,
                          pairSplineId,
                          tables,
                          gradOut,
                          hessOut,
                          pairHasSupport);
    }

    DFTFE_CREATE_KERNEL(
      void,
      atomicDensityQuadSuperpositionKernel,
      {
        const dftfe::uInt numberEntries = numPoints;
        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
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
      },
      const dftfe::uInt                      numPoints,
      const dftfe::uInt                      numCenters,
      const dftfe::uInt                      numMagComponents,
      const bool                             floatingNuclearCharges,
      const bool                             applyConstraintMag,
      const double                           totMagnetization,
      const double                          *pointCoords,
      const double                          *centerCoords,
      const dftfe::uInt                     *centerSplineId,
      const double                          *centerRhoFactor,
      const double                          *centerMagZFactor,
      const double                          *centerMagYFactor,
      const double                          *centerMagXFactor,
      const splineEval::SplineTableGroupView tables,
      double                                *rhoOut,
      double                                *magZOut,
      double                                *magYOut,
      double                                *magXOut,
      double                                *gradRhoOut,
      double                                *gradMagZOut,
      double                                *gradMagYOut,
      double                                *gradMagXOut);

    template <>
    void
    atomicDensityQuadSuperposition<dftfe::utils::MemorySpace::DEVICE>(
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
      const dftfe::uInt)
    {
      const dftfe::uInt numberEntries = numPoints;
      if (numberEntries == 0)
        return;
      DFTFE_LAUNCH_KERNEL(atomicDensityQuadSuperpositionKernel,
                          (dftfe::utils::DEVICE_BLOCK_SIZE + numberEntries) /
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::linearAlgebra::BLASWrapper<
                            dftfe::utils::MemorySpace::DEVICE>::d_streamId,
                          numPoints,
                          numCenters,
                          numMagComponents,
                          floatingNuclearCharges,
                          applyConstraintMag,
                          totMagnetization,
                          pointCoords,
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

    DFTFE_CREATE_KERNEL(
      void,
      fillRandomWavefunctionsKernel,
      {
        const dftfe::uInt numberEntries =
          numLocalNodes * (numWaveFunctions - firstWave) * numRealsPerCoeff;
        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          randomWavefunctionEntry(index,
                                  numWaveFunctions,
                                  firstWave,
                                  numRealsPerCoeff,
                                  numSpinorComponents,
                                  globalNodeStartIndex,
                                  seed,
                                  constrainedMask,
                                  wfcData);
      },
      const dftfe::uInt        numLocalNodes,
      const dftfe::uInt        numWaveFunctions,
      const dftfe::uInt        firstWave,
      const dftfe::uInt        numRealsPerCoeff,
      const dftfe::uInt        numSpinorComponents,
      const unsigned long long globalNodeStartIndex,
      const unsigned long long seed,
      const dftfe::uInt       *constrainedMask,
      double                  *wfcData);

    template <>
    void
    fillRandomWavefunctions<dftfe::utils::MemorySpace::DEVICE>(
      const dftfe::uInt        numLocalNodes,
      const dftfe::uInt        numWaveFunctions,
      const dftfe::uInt        firstWave,
      const dftfe::uInt        numRealsPerCoeff,
      const dftfe::uInt        numSpinorComponents,
      const unsigned long long globalNodeStartIndex,
      const unsigned long long seed,
      const dftfe::uInt       *constrainedMask,
      double                  *wfcData,
      const dftfe::uInt)
    {
      const dftfe::uInt numberEntries =
        firstWave >= numWaveFunctions ?
          0 :
          numLocalNodes * (numWaveFunctions - firstWave) * numRealsPerCoeff;
      if (numberEntries == 0)
        return;
      DFTFE_LAUNCH_KERNEL(fillRandomWavefunctionsKernel,
                          (dftfe::utils::DEVICE_BLOCK_SIZE + numberEntries) /
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::linearAlgebra::BLASWrapper<
                            dftfe::utils::MemorySpace::DEVICE>::d_streamId,
                          numLocalNodes,
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
