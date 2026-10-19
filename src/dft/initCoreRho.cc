// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
// @author Phani Motamarri
//

//
// Initialize rho by reading in single-atom electron-density and fit a spline
//
#include <dftfe/dftParameters.h>
#include <dftfe/dft.h>
#include <dftfe/fileReaders.h>
#include <dftfe/atomicInitKernels.h>
#include <dftfe/atomTypeSplineTables.h>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::initCoreRho()
  {
    // clear existing data
    d_rhoCore.clear();
    d_gradRhoCore.clear();
    d_gradRhoCoreAtoms.clear();
    d_hessianRhoCoreAtoms.clear();

    pcout
      << std::endl
      << "Reading data for core electron-density to be used in nonlinear core-correction....."
      << std::endl;

    double maxCoreRhoTail = 0.0;
    for (std::set<dftfe::uInt>::iterator it = atomTypes.begin();
         it != atomTypes.end();
         it++)
      {
        const double outerMostPointCoreDen =
          d_oncvClassPtr->getRmaxCoreDensity(*it);
        if (outerMostPointCoreDen > maxCoreRhoTail)
          maxCoreRhoTail = outerMostPointCoreDen;
        if (d_dftParamsPtr->verbosity >= 4)
          pcout << " Atomic number: " << *it
                << " Outermost Point Core Den: " << outerMostPointCoreDen
                << std::endl;
      }

    const bool isGradDensityDataDependent =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);

    // Core density of every atom type that has one; a type without
    // nonlinear core correction gets no table entry and contributes nothing.
    splineHostTableUtils::SplineTableGroup coreTables;
    std::map<dftfe::uInt, dftfe::uInt>     znumToCoreSplineId;
    buildAtomTypeSplineTables(
      atomTypes,
      [this](const dftfe::uInt Znum) {
        return d_oncvClassPtr->getRadialCoreDensityFunction(Znum, 0);
      },
      coreTables,
      znumToCoreSplineId);
    AtomTypeSplineTables<memorySpace> coreTablesMem;
    coreTablesMem.stage(coreTables);

    // Totals at every quadrature point; a device run dispatches to the
    // device specialization and a CPU run to the host one.
    computeRadialDensityQuadTotals(coreTablesMem,
                                   znumToCoreSplineId,
                                   d_rhoCore,
                                   d_gradRhoCore);

    // The per-atom maps, read only by the force and stress computation, at
    // every (center-cell pair, quadrature point).
    if (!(d_dftParamsPtr->isIonForce || d_dftParamsPtr->isCellStress))
      return;

    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureId);
    const dftfe::Int  numberGlobalCharges = atomLocations.size();
    const dftfe::Int  numberImageCharges  = d_imageIdsTrunc.size();
    const dftfe::uInt nCells              = d_basisOperationsPtrHost->nCells();
    const dftfe::uInt nQuadsPerCell = d_basisOperationsPtrHost->nQuadsPerCell();
    const double     *quadPointsHost =
      d_basisOperationsPtrHost->quadPoints().data();

    // bounding sphere of each cell, so a center can be rejected without
    // touching its quadrature points
    std::vector<dealii::Point<3>> cellCentroids;
    std::vector<double>           cellRadii;
    dftUtils::computeCellBoundingSpheres(
      quadPointsHost, nCells, nQuadsPerCell, cellCentroids, cellRadii);

    // a pair returns three gradient doubles per point, and nine hessian
    // doubles more for GGA
    const dftfe::uInt pairsBlockSize =
      std::max<dftfe::uInt>(1,
                            atomicInitKernels::maxBlockBytes /
                              (std::size_t)(
                                nQuadsPerCell *
                                (3 + (isGradDensityDataDependent ? 9 : 0)) *
                                sizeof(double)));

    std::vector<double>      blockCoords, blockCenters;
    std::vector<dftfe::uInt> blockCellSlot, blockSplineId, blockCenterId,
      blockCellId;
    std::map<dftfe::uInt, dftfe::uInt> cellToSlot;

    dftfe::utils::MemoryStorage<double, memorySpace> dCoords, dCenters, dGrad,
      dHess;
    dftfe::utils::MemoryStorage<dftfe::uInt, memorySpace> dCellSlot, dSplineId,
      dHasSupport;
    std::vector<double>      hGrad, hHess;
    std::vector<dftfe::uInt> hHasSupport;

    auto flushBlock = [&]() {
      const dftfe::uInt nPairs = blockCenterId.size();
      if (nPairs == 0)
        return;
      const std::size_t nPointsInBlock = (std::size_t)nPairs * nQuadsPerCell;
      dCoords =
        dftfe::utils::memoryStorageFromSTL<double, memorySpace>(blockCoords);
      dCenters =
        dftfe::utils::memoryStorageFromSTL<double, memorySpace>(blockCenters);
      dCellSlot = dftfe::utils::memoryStorageFromSTL<dftfe::uInt, memorySpace>(
        blockCellSlot);
      dSplineId = dftfe::utils::memoryStorageFromSTL<dftfe::uInt, memorySpace>(
        blockSplineId);
      dGrad.resize(nPointsInBlock * 3);
      if (isGradDensityDataDependent)
        dHess.resize(nPointsInBlock * 9);
      dHasSupport.resize(nPairs);
      dHasSupport.setValue(0);

      atomicInitKernels::atomicRadialDensityPairDerivatives<memorySpace>(
        nPairs,
        nQuadsPerCell,
        d_dftParamsPtr->floatingNuclearCharges,
        dCoords.data(),
        dCenters.data(),
        dCellSlot.data(),
        dSplineId.data(),
        coreTablesMem.view(),
        dGrad.data(),
        isGradDensityDataDependent ? dHess.data() : nullptr,
        dHasSupport.data(),
        d_nOMPThreads);

      hGrad.resize(nPointsInBlock * 3);
      dGrad.copyTo(hGrad);
      if (isGradDensityDataDependent)
        {
          hHess.resize(nPointsInBlock * 9);
          dHess.copyTo(hHess);
        }
      hHasSupport.resize(nPairs);
      dHasSupport.copyTo(hHasSupport);

      for (dftfe::uInt iPair = 0; iPair < nPairs; ++iPair)
        {
          if (hHasSupport[iPair] != 1)
            continue;
          const dealii::CellId cid =
            d_basisOperationsPtrHost->cellID(blockCellId[iPair]);
          const std::size_t offset = (std::size_t)iPair * nQuadsPerCell;
          d_gradRhoCoreAtoms[blockCenterId[iPair]][cid].assign(
            hGrad.begin() + offset * 3,
            hGrad.begin() + (offset + nQuadsPerCell) * 3);
          std::vector<double> &hessianRhoCoreAtomCell =
            d_hessianRhoCoreAtoms[blockCenterId[iPair]][cid];
          if (isGradDensityDataDependent)
            hessianRhoCoreAtomCell.assign(hHess.begin() + offset * 9,
                                          hHess.begin() +
                                            (offset + nQuadsPerCell) * 9);
        }

      blockCoords.clear();
      blockCenters.clear();
      blockCellSlot.clear();
      blockSplineId.clear();
      blockCenterId.clear();
      blockCellId.clear();
      cellToSlot.clear();
    };

    // atoms first, then their truncated periodic images, the center index
    // keying d_gradRhoCoreAtoms and d_hessianRhoCoreAtoms
    const dftfe::uInt nCenters = numberGlobalCharges + numberImageCharges;
    for (dftfe::uInt iCell = 0; iCell < nCells; ++iCell)
      for (dftfe::uInt iCenter = 0; iCenter < nCenters; ++iCenter)
        {
          const bool isImage = iCenter >= (dftfe::uInt)numberGlobalCharges;
          const dftfe::uInt iAtom =
            isImage ? d_imageIdsTrunc[iCenter - numberGlobalCharges] : iCenter;
          // atoms of a type without core density contribute nothing
          const auto splineIt =
            znumToCoreSplineId.find(std::round(atomLocations[iAtom][0]));
          if (splineIt == znumToCoreSplineId.end())
            continue;
          dealii::Point<3> center;
          for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
            center[iDim] =
              isImage ?
                d_imagePositionsTrunc[iCenter - numberGlobalCharges][iDim] :
                atomLocations[iAtom][2 + iDim];

          // a center whose tail cannot reach the nearest point of the
          // cell contributes nothing and is never stored
          if (cellCentroids[iCell].distance(center) - cellRadii[iCell] >
              maxCoreRhoTail + dftUtils::cellBoundingSphereTolerance)
            continue;

          if (blockCenterId.size() >= pairsBlockSize)
            flushBlock();

          auto slotIt = cellToSlot.find(iCell);
          if (slotIt == cellToSlot.end())
            {
              const dftfe::uInt slot = cellToSlot.size();
              cellToSlot[iCell]      = slot;
              const double *qp =
                quadPointsHost + (std::size_t)iCell * nQuadsPerCell * 3;
              blockCoords.insert(blockCoords.end(), qp, qp + nQuadsPerCell * 3);
              blockCellSlot.push_back(slot);
            }
          else
            blockCellSlot.push_back(slotIt->second);

          blockCenters.push_back(center[0]);
          blockCenters.push_back(center[1]);
          blockCenters.push_back(center[2]);
          blockSplineId.push_back(splineIt->second);
          blockCenterId.push_back(iCenter);
          blockCellId.push_back(iCell);
        }
    flushBlock();
  }
#include "dft.inst.cc"
} // namespace dftfe
