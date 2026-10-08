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
    // the per-atom maps are only read by the force and stress computation
    const bool storePerAtomCoreData =
      d_dftParamsPtr->isIonForce || d_dftParamsPtr->isCellStress;

    d_gradRhoCoreAtoms.clear();
    d_hessianRhoCoreAtoms.clear();

    // Reading single atom rho initial guess
    pcout
      << std::endl
      << "Reading data for core electron-density to be used in nonlinear core-correction....."
      << std::endl;
    std::map<dftfe::uInt, alglib::spline1dinterpolant> coreDenSpline;
    std::map<dftfe::uInt, std::vector<std::vector<double>>>
                                  singleAtomCoreElectronDensity;
    std::map<dftfe::uInt, double> outerMostPointCoreDen;
    const double                  truncationTol = 1e-12;
    dftfe::uInt                   fileReadFlag  = 0;

    double maxCoreRhoTail = 0.0;
    // loop over atom types
    for (std::set<dftfe::uInt>::iterator it = atomTypes.begin();
         it != atomTypes.end();
         it++)
      {
        outerMostPointCoreDen[*it] = d_oncvClassPtr->getRmaxCoreDensity(*it);
        if (outerMostPointCoreDen[*it] > maxCoreRhoTail)
          maxCoreRhoTail = outerMostPointCoreDen[*it];
        if (d_dftParamsPtr->verbosity >= 4)
          pcout << " Atomic number: " << *it
                << " Outermost Point Core Den: " << outerMostPointCoreDen[*it]
                << std::endl;
      }

    const double cellCenterCutOff = maxCoreRhoTail + 5.0;
    //
    // Initialize rho
    //
    const dealii::Quadrature<3> &quadrature_formula =
      matrix_free_data.get_quadrature(d_densityQuadratureId);
    dealii::FEValues<3> fe_values(FE,
                                  quadrature_formula,
                                  dealii::update_quadrature_points);
    const dftfe::uInt   n_q_points = quadrature_formula.size();

    //
    // get number of global charges
    //
    const dftfe::Int numberGlobalCharges = atomLocations.size();

    //
    // get number of image charges used only for periodic
    //
    const dftfe::Int numberImageCharges = d_imageIdsTrunc.size();

    //
    // loop over elements
    //
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    dealii::Tensor<1, 3, double> zeroTensor1;
    for (dftfe::uInt i = 0; i < 3; i++)
      zeroTensor1[i] = 0.0;

    dealii::Tensor<2, 3, double> zeroTensor2;

    for (dftfe::uInt i = 0; i < 3; i++)
      for (dftfe::uInt j = 0; j < 3; j++)
        zeroTensor2[i][j] = 0.0;

    bool isGradDensityDataDependent =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);

    // Totals at every quadrature point, and the per-atom maps the force and
    // stress read at every (center-cell pair, quadrature point); a device run
    // dispatches to the device specialization and a CPU run to the host one.
    // The loop further down covers any species that is not spline backed.
    const std::function<const AtomCenteredSphericalFunctionBase *(
      const dftfe::uInt)>
      getCoreDensityFunction = [this](const dftfe::uInt Znum) {
        return d_oncvClassPtr->getRadialCoreDensityFunction(Znum, 0);
      };
    const bool coreRhoComputed =
      computeRadialDensityQuadTotals(getCoreDensityFunction,
                                     d_rhoCore,
                                     d_gradRhoCore);

    if (coreRhoComputed && storePerAtomCoreData)
      {
        splineHostTableUtils::SplineTableGroup coreTables;
        std::map<dftfe::uInt, dftfe::uInt>     znumToCoreSplineId;
        const bool coreTablesBuilt = buildAtomTypeSplineTables(
          atomTypes, getCoreDensityFunction, coreTables, znumToCoreSplineId);
        AssertThrow(coreTablesBuilt,
                    dealii::ExcMessage(
                      "DFT-FE Error: core density tables failed to build"));
        AtomTypeSplineTables<memorySpace> coreTablesMem;
        coreTablesMem.stage(coreTables);

        const dftfe::uInt nCells = d_basisOperationsPtrHost->nCells();
        const dftfe::uInt nQuadsPerCell =
          d_basisOperationsPtrHost->nQuadsPerCell();
        const double *quadPointsHost =
          d_basisOperationsPtrHost->quadPoints().data();

        // bounding sphere of each cell, so a center can be rejected without
        // touching its quadrature points
        std::vector<dealii::Point<3>> cellCentroids(nCells);
        std::vector<double>           cellRadii(nCells, 0.0);
        for (dftfe::uInt iCell = 0; iCell < nCells; ++iCell)
          {
            const double *qp =
              quadPointsHost + (std::size_t)iCell * nQuadsPerCell * 3;
            dealii::Point<3> centroid;
            for (dftfe::uInt q = 0; q < nQuadsPerCell; ++q)
              for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                centroid[iDim] += qp[q * 3 + iDim];
            centroid /= static_cast<double>(nQuadsPerCell);
            double r2max = 0.0;
            for (dftfe::uInt q = 0; q < nQuadsPerCell; ++q)
              {
                const dealii::Point<3> p(qp[q * 3 + 0],
                                         qp[q * 3 + 1],
                                         qp[q * 3 + 2]);
                r2max = std::max(r2max, p.distance_square(centroid));
              }
            cellCentroids[iCell] = centroid;
            cellRadii[iCell]     = std::sqrt(r2max);
          }

        // a pair returns three gradient doubles per point, and nine hessian
        // doubles more for GGA
        constexpr std::size_t maxBlockBytes = 256000000;
        const dftfe::uInt     pairsBlockSize =
          std::max<dftfe::uInt>(1,
                                maxBlockBytes /
                                  (std::size_t)(
                                    nQuadsPerCell *
                                    (3 + (isGradDensityDataDependent ? 9 : 0)) *
                                    sizeof(double)));

        std::vector<double>      blockCoords, blockCenters;
        std::vector<dftfe::uInt> blockCellSlot, blockSplineId, blockCenterId,
          blockCellId;
        std::map<dftfe::uInt, dftfe::uInt> cellToSlot;

        dftfe::utils::MemoryStorage<double, memorySpace> dCoords, dCenters,
          dGrad, dHess;
        dftfe::utils::MemoryStorage<dftfe::uInt, memorySpace> dCellSlot,
          dSplineId, dHasSupport;
        std::vector<double>      hGrad, hHess;
        std::vector<dftfe::uInt> hHasSupport;

        auto flushBlock = [&]() {
          const dftfe::uInt nPairs = blockCenterId.size();
          if (nPairs == 0)
            return;
          const std::size_t nPointsInBlock =
            (std::size_t)nPairs * nQuadsPerCell;
          stageHostVector(blockCoords, dCoords);
          stageHostVector(blockCenters, dCenters);
          stageHostVector(blockCellSlot, dCellSlot);
          stageHostVector(blockSplineId, dSplineId);
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
            coreTablesMem.knotX.data(),
            coreTablesMem.knotY.data(),
            coreTablesMem.coefB.data(),
            coreTablesMem.coefC.data(),
            coreTablesMem.coefD.data(),
            coreTablesMem.offsets.data(),
            coreTablesMem.nKnots.data(),
            coreTablesMem.gpParams.data(),
            coreTablesMem.cutoffs.data(),
            coreTablesMem.rMins.data(),
            dGrad.data(),
            isGradDensityDataDependent ? dHess.data() : nullptr,
            dHasSupport.data());

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

        // atoms first, then their periodic images, matching the per-atom
        // indexing of the loop below
        const dftfe::uInt nCenters = numberGlobalCharges + numberImageCharges;
        for (dftfe::uInt iCell = 0; iCell < nCells; ++iCell)
          for (dftfe::uInt iCenter = 0; iCenter < nCenters; ++iCenter)
            {
              const bool isImage = iCenter >= (dftfe::uInt)numberGlobalCharges;
              const dftfe::uInt iAtom =
                isImage ? d_imageIdsTrunc[iCenter - numberGlobalCharges] :
                          iCenter;
              dealii::Point<3> center;
              for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                center[iDim] =
                  isImage ?
                    d_imagePositionsTrunc[iCenter - numberGlobalCharges][iDim] :
                    atomLocations[iAtom][2 + iDim];

              // a center whose tail cannot reach the nearest point of the
              // cell contributes nothing and is never stored
              if (cellCentroids[iCell].distance(center) - cellRadii[iCell] >
                  maxCoreRhoTail + 1e-9)
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
                  blockCoords.insert(blockCoords.end(),
                                     qp,
                                     qp + nQuadsPerCell * 3);
                  blockCellSlot.push_back(slot);
                }
              else
                blockCellSlot.push_back(slotIt->second);

              blockCenters.push_back(center[0]);
              blockCenters.push_back(center[1]);
              blockCenters.push_back(center[2]);
              blockSplineId.push_back(
                znumToCoreSplineId[std::round(atomLocations[iAtom][0])]);
              blockCenterId.push_back(iCenter);
              blockCellId.push_back(iCell);
            }
        flushBlock();
      }

    // loop over elements
    //
    cell = dofHandler.begin_active();
    for (; cell != endc && !coreRhoComputed; ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);

            std::vector<double> &rhoCoreQuadValues = d_rhoCore[cell->id()];
            rhoCoreQuadValues.resize(n_q_points, 0.0);

            std::vector<double> &gradRhoCoreQuadValues =
              d_gradRhoCore[cell->id()];
            gradRhoCoreQuadValues.resize(n_q_points * 3, 0.0);

            std::vector<dealii::Tensor<1, 3, double>> gradRhoCoreAtom(
              n_q_points, zeroTensor1);
            std::vector<dealii::Tensor<2, 3, double>> hessianRhoCoreAtom(
              n_q_points, zeroTensor2);


            // loop over atoms
            for (dftfe::uInt iAtom = 0; iAtom < atomLocations.size(); ++iAtom)
              {
                dealii::Point<3> atom(atomLocations[iAtom][2],
                                      atomLocations[iAtom][3],
                                      atomLocations[iAtom][4]);
                bool             isCoreRhoDataInCell = false;

                if (!d_oncvClassPtr->coreNuclearDensityPresent(
                      atomLocations[iAtom][0]))
                  continue;

                if (atom.distance(cell->center()) > cellCenterCutOff)
                  continue;

                // loop over quad points
                for (dftfe::uInt q = 0; q < n_q_points; ++q)
                  {
                    dealii::Point<3> quadPoint = fe_values.quadrature_point(q);
                    dealii::Tensor<1, 3, double> diff = quadPoint - atom;
                    double distanceToAtom = quadPoint.distance(atom);

                    if (d_dftParamsPtr->floatingNuclearCharges &&
                        distanceToAtom < 1.0e-4)
                      {
                        if (d_dftParamsPtr->verbosity >= 4)
                          std::cout
                            << "Atom close to quad point, iatom: " << iAtom
                            << std::endl;

                        distanceToAtom = 1.0e-4;
                        diff[0]        = (1.0e-4) / std::sqrt(3.0);
                        diff[1]        = (1.0e-4) / std::sqrt(3.0);
                        diff[2]        = (1.0e-4) / std::sqrt(3.0);
                      }

                    double value, radialDensityFirstDerivative,
                      radialDensitySecondDerivative;
                    if (distanceToAtom <=
                        outerMostPointCoreDen[atomLocations[iAtom][0]])
                      {
                        std::vector<double> Vec;
                        d_oncvClassPtr->getRadialCoreDensity(
                          atomLocations[iAtom][0], distanceToAtom, Vec);

                        value                         = Vec[0];
                        radialDensityFirstDerivative  = Vec[1];
                        radialDensitySecondDerivative = Vec[2];
                        // pcout<<distanceToAtom<<" "<<value<<std::endl;
                        isCoreRhoDataInCell = true;
                      }
                    else
                      {
                        value                         = 0.0;
                        radialDensityFirstDerivative  = 0.0;
                        radialDensitySecondDerivative = 0.0;
                      }

                    rhoCoreQuadValues[q] += value;
                    gradRhoCoreAtom[q] =
                      radialDensityFirstDerivative * diff / distanceToAtom;
                    gradRhoCoreQuadValues[3 * q + 0] += gradRhoCoreAtom[q][0];
                    gradRhoCoreQuadValues[3 * q + 1] += gradRhoCoreAtom[q][1];
                    gradRhoCoreQuadValues[3 * q + 2] += gradRhoCoreAtom[q][2];

                    if (isGradDensityDataDependent)
                      {
                        for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                          {
                            for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                              {
                                double temp = (radialDensitySecondDerivative -
                                               radialDensityFirstDerivative /
                                                 distanceToAtom) *
                                              (diff[iDim] / distanceToAtom) *
                                              (diff[jDim] / distanceToAtom);
                                if (iDim == jDim)
                                  temp += radialDensityFirstDerivative /
                                          distanceToAtom;

                                hessianRhoCoreAtom[q][iDim][jDim] = temp;
                              }
                          }
                      }


                  } // end loop over quad points
                if (isCoreRhoDataInCell && storePerAtomCoreData)
                  {
                    std::vector<double> &gradRhoCoreAtomCell =
                      d_gradRhoCoreAtoms[iAtom][cell->id()];
                    gradRhoCoreAtomCell.resize(n_q_points * 3, 0.0);

                    std::vector<double> &hessianRhoCoreAtomCell =
                      d_hessianRhoCoreAtoms[iAtom][cell->id()];
                    if (isGradDensityDataDependent)
                      hessianRhoCoreAtomCell.resize(n_q_points * 9, 0.0);

                    for (dftfe::uInt q = 0; q < n_q_points; ++q)
                      {
                        gradRhoCoreAtomCell[3 * q + 0] = gradRhoCoreAtom[q][0];
                        gradRhoCoreAtomCell[3 * q + 1] = gradRhoCoreAtom[q][1];
                        gradRhoCoreAtomCell[3 * q + 2] = gradRhoCoreAtom[q][2];

                        if (isGradDensityDataDependent)
                          {
                            for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                              {
                                for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                                  {
                                    hessianRhoCoreAtomCell[9 * q + 3 * iDim +
                                                           jDim] =
                                      hessianRhoCoreAtom[q][iDim][jDim];
                                  }
                              }
                          }
                      } // q_point loop
                  }     // if loop
              }         // loop over atoms

            // loop over image charges
            for (dftfe::uInt iImageCharge = 0;
                 iImageCharge < numberImageCharges;
                 ++iImageCharge)
              {
                const dftfe::Int masterAtomId = d_imageIdsTrunc[iImageCharge];
                if (!d_oncvClassPtr->coreNuclearDensityPresent(
                      atomLocations[masterAtomId][0]))
                  continue;

                dealii::Point<3> imageAtom(
                  d_imagePositionsTrunc[iImageCharge][0],
                  d_imagePositionsTrunc[iImageCharge][1],
                  d_imagePositionsTrunc[iImageCharge][2]);

                if (imageAtom.distance(cell->center()) > cellCenterCutOff)
                  continue;

                bool isCoreRhoDataInCell = false;

                // loop over quad points
                for (dftfe::uInt q = 0; q < n_q_points; ++q)
                  {
                    dealii::Point<3> quadPoint = fe_values.quadrature_point(q);
                    dealii::Tensor<1, 3, double> diff = quadPoint - imageAtom;
                    double distanceToAtom = quadPoint.distance(imageAtom);

                    if (d_dftParamsPtr->floatingNuclearCharges &&
                        distanceToAtom < 1.0e-4)
                      {
                        distanceToAtom = 1.0e-4;
                        diff[0]        = (1.0e-4) / std::sqrt(3.0);
                        diff[1]        = (1.0e-4) / std::sqrt(3.0);
                        diff[2]        = (1.0e-4) / std::sqrt(3.0);
                      }

                    double value, radialDensityFirstDerivative,
                      radialDensitySecondDerivative;
                    if (distanceToAtom <=
                        outerMostPointCoreDen[atomLocations[masterAtomId][0]])
                      {
                        std::vector<double> Vec;
                        d_oncvClassPtr->getRadialCoreDensity(
                          atomLocations[masterAtomId][0], distanceToAtom, Vec);
                        value                         = Vec[0];
                        radialDensityFirstDerivative  = Vec[1];
                        radialDensitySecondDerivative = Vec[2];
                        // pcout<<distanceToAtom<<" "<<value<<std::endl;
                        isCoreRhoDataInCell = true;
                      }
                    else
                      {
                        value                         = 0.0;
                        radialDensityFirstDerivative  = 0.0;
                        radialDensitySecondDerivative = 0.0;
                      }

                    rhoCoreQuadValues[q] += value;
                    gradRhoCoreAtom[q] =
                      radialDensityFirstDerivative * diff / distanceToAtom;
                    gradRhoCoreQuadValues[3 * q + 0] += gradRhoCoreAtom[q][0];
                    gradRhoCoreQuadValues[3 * q + 1] += gradRhoCoreAtom[q][1];
                    gradRhoCoreQuadValues[3 * q + 2] += gradRhoCoreAtom[q][2];

                    if (isGradDensityDataDependent)
                      {
                        for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                          {
                            for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                              {
                                double temp = (radialDensitySecondDerivative -
                                               radialDensityFirstDerivative /
                                                 distanceToAtom) *
                                              (diff[iDim] / distanceToAtom) *
                                              (diff[jDim] / distanceToAtom);
                                if (iDim == jDim)
                                  temp += radialDensityFirstDerivative /
                                          distanceToAtom;
                                hessianRhoCoreAtom[q][iDim][jDim] = temp;
                              }
                          }
                      }

                  } // quad point loop

                if (isCoreRhoDataInCell && storePerAtomCoreData)
                  {
                    std::vector<double> &gradRhoCoreAtomCell =
                      d_gradRhoCoreAtoms[numberGlobalCharges + iImageCharge]
                                        [cell->id()];
                    gradRhoCoreAtomCell.resize(n_q_points * 3);

                    std::vector<double> &hessianRhoCoreAtomCell =
                      d_hessianRhoCoreAtoms[numberGlobalCharges + iImageCharge]
                                           [cell->id()];
                    if (isGradDensityDataDependent)
                      hessianRhoCoreAtomCell.resize(n_q_points * 9);

                    for (dftfe::uInt q = 0; q < n_q_points; ++q)
                      {
                        gradRhoCoreAtomCell[3 * q + 0] = gradRhoCoreAtom[q][0];
                        gradRhoCoreAtomCell[3 * q + 1] = gradRhoCoreAtom[q][1];
                        gradRhoCoreAtomCell[3 * q + 2] = gradRhoCoreAtom[q][2];

                        if (isGradDensityDataDependent)
                          {
                            for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
                              {
                                for (dftfe::uInt jDim = 0; jDim < 3; ++jDim)
                                  {
                                    hessianRhoCoreAtomCell[9 * q + 3 * iDim +
                                                           jDim] =
                                      hessianRhoCoreAtom[q][iDim][jDim];
                                  }
                              }
                          }
                      } // q_point loop
                  }     // if loop

              } // end of image charges

          } // cell locally owned check

      } // cell loop
  }
#include "dft.inst.cc"
} // namespace dftfe
