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
// @author Sambit Das
//
#include <dftfe/dft.h>
#include <dftfe/fileReaders.h>
#include <dftfe/atomicInitKernels.h>
#include <dftfe/atomTypeSplineTables.h>

namespace dftfe
{
  //
  // Initialize rho by reading in single-atom electron-density and fit a spline
  //

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::initAtomicRho()
  {
    // clear existing data
    d_rhoAtomsValues.clear();
    d_gradRhoAtomsValues.clear();

    // Reading single atom rho initial guess
    pcout << std::endl
          << "Reading initial guess for electron-density....." << std::endl;

    // A pseudopotential run evaluates the valence densities through
    // atomicInitKernels; a device run dispatches to the device
    // specialization and a CPU run to the host one.
    if (d_dftParamsPtr->isPseudopotential)
      {
        splineHostTableUtils::SplineTableGroup tables;
        std::map<dftfe::uInt, dftfe::uInt>     znumToSplineId;
        buildAtomTypeSplineTables(
          atomTypes,
          [this](const dftfe::uInt Znum) {
            return d_oncvClassPtr->getRadialValenceDensityFunction(Znum, 0);
          },
          tables,
          znumToSplineId);
        AtomTypeSplineTables<memorySpace> tablesMem;
        tablesMem.stage(tables);
        computeRadialDensityQuadTotals(tablesMem,
                                       znumToSplineId,
                                       d_rhoAtomsValues,
                                       d_gradRhoAtomsValues);
        normalizeAtomicRhoQuadValues();
        return;
      }

    // all-electron run: single-atom densities from the data files
    std::map<dftfe::uInt, alglib::spline1dinterpolant> denSpline;
    std::map<dftfe::uInt, std::vector<std::vector<double>>>
                                  singleAtomElectronDensity;
    std::map<dftfe::uInt, double> outerMostPointDen;
    const double                  truncationTol = 1e-10;
    double                        maxRhoTail    = 0.0;

    // loop over atom types
    for (std::set<dftfe::uInt>::iterator it = atomTypes.begin();
         it != atomTypes.end();
         it++)
      {
        char densityFile[256];
        sprintf(
          densityFile,
          "%s/data/electronicStructure/allElectron/z%u/singleAtomData/density.inp",
          DFTFE_PATH,
          *it);

        dftUtils::readFile(2, singleAtomElectronDensity[*it], densityFile);
        dftfe::uInt         numRows = singleAtomElectronDensity[*it].size() - 1;
        std::vector<double> xData(numRows), yData(numRows);

        dftfe::uInt maxRowId = 0;
        for (dftfe::uInt irow = 0; irow < numRows; ++irow)
          {
            xData[irow] = singleAtomElectronDensity[*it][irow][0];
            yData[irow] = singleAtomElectronDensity[*it][irow][1];

            if (yData[irow] > truncationTol)
              maxRowId = irow;
          }
        yData[0] = yData[1];

        // interpolate rho
        alglib::real_1d_array x;
        x.setcontent(numRows, &xData[0]);
        alglib::real_1d_array y;
        y.setcontent(numRows, &yData[0]);
        alglib::ae_int_t natural_bound_type_L = 1;
        alglib::ae_int_t natural_bound_type_R = 1;
        spline1dbuildcubic(x,
                           y,
                           numRows,
                           natural_bound_type_L,
                           0.0,
                           natural_bound_type_R,
                           0.0,
                           denSpline[*it]);
        outerMostPointDen[*it] = xData[maxRowId];
        if (outerMostPointDen[*it] > maxRhoTail)
          maxRhoTail = outerMostPointDen[*it];
      }

    const double cellCenterCutOff = maxRhoTail + 5.0;

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

    dealii::Tensor<1, 3, double> zeroTensor1;
    for (dftfe::uInt i = 0; i < 3; i++)
      zeroTensor1[i] = 0.0;

    // loop over elements
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);

            std::vector<double> &rhoAtomsQuadValues =
              d_rhoAtomsValues[cell->id()];
            rhoAtomsQuadValues.resize(n_q_points, 0.0);

            std::vector<double> &gradRhoAtomsQuadValues =
              d_gradRhoAtomsValues[cell->id()];
            gradRhoAtomsQuadValues.resize(n_q_points * 3, 0.0);

            std::vector<dealii::Tensor<1, 3, double>> gradRhoAtom(n_q_points,
                                                                  zeroTensor1);


            // loop over atoms
            for (dftfe::uInt iAtom = 0; iAtom < atomLocations.size(); ++iAtom)
              {
                dealii::Point<3> atom(atomLocations[iAtom][2],
                                      atomLocations[iAtom][3],
                                      atomLocations[iAtom][4]);

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
                        outerMostPointDen[atomLocations[iAtom][0]])
                      alglib::spline1ddiff(denSpline[atomLocations[iAtom][0]],
                                           distanceToAtom,
                                           value,
                                           radialDensityFirstDerivative,
                                           radialDensitySecondDerivative);
                    else
                      {
                        value                         = 0.0;
                        radialDensityFirstDerivative  = 0.0;
                        radialDensitySecondDerivative = 0.0;
                      }

                    rhoAtomsQuadValues[q] += value;
                    gradRhoAtom[q] =
                      radialDensityFirstDerivative * diff / distanceToAtom;
                    gradRhoAtomsQuadValues[3 * q + 0] += gradRhoAtom[q][0];
                    gradRhoAtomsQuadValues[3 * q + 1] += gradRhoAtom[q][1];
                    gradRhoAtomsQuadValues[3 * q + 2] += gradRhoAtom[q][2];
                  } // end loop over quad points
              }     // loop over atoms

            // loop over image charges
            for (dftfe::uInt iImageCharge = 0;
                 iImageCharge < numberImageCharges;
                 ++iImageCharge)
              {
                const dftfe::Int masterAtomId = d_imageIdsTrunc[iImageCharge];

                dealii::Point<3> imageAtom(
                  d_imagePositionsTrunc[iImageCharge][0],
                  d_imagePositionsTrunc[iImageCharge][1],
                  d_imagePositionsTrunc[iImageCharge][2]);

                if (imageAtom.distance(cell->center()) > cellCenterCutOff)
                  continue;


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
                        outerMostPointDen[atomLocations[masterAtomId][0]])
                      alglib::spline1ddiff(
                        denSpline[atomLocations[masterAtomId][0]],
                        distanceToAtom,
                        value,
                        radialDensityFirstDerivative,
                        radialDensitySecondDerivative);
                    else
                      {
                        value                         = 0.0;
                        radialDensityFirstDerivative  = 0.0;
                        radialDensitySecondDerivative = 0.0;
                      }

                    rhoAtomsQuadValues[q] += value;
                    gradRhoAtom[q] =
                      radialDensityFirstDerivative * diff / distanceToAtom;
                    gradRhoAtomsQuadValues[3 * q + 0] += gradRhoAtom[q][0];
                    gradRhoAtomsQuadValues[3 * q + 1] += gradRhoAtom[q][1];
                    gradRhoAtomsQuadValues[3 * q + 2] += gradRhoAtom[q][2];
                  } // quad point loop

              } // end of image charges

          } // cell locally owned check

      } // cell loop

    normalizeAtomicRhoQuadValues();
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::computeRadialDensityQuadTotals(
    const AtomTypeSplineTables<memorySpace>       &tables,
    const std::map<dftfe::uInt, dftfe::uInt>      &znumToSplineId,
    std::map<dealii::CellId, std::vector<double>> &valueMap,
    std::map<dealii::CellId, std::vector<double>> &gradMap)
  {
    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureId);
    const dftfe::uInt nCells        = d_basisOperationsPtrHost->nCells();
    const dftfe::uInt nQuadsPerCell = d_basisOperationsPtrHost->nQuadsPerCell();

    // atoms first, then their periodic images; atoms of a type without an
    // entry in znumToSplineId contribute nothing
    const dftfe::uInt        nAtoms = atomLocations.size();
    std::vector<double>      centerCoordsHost;
    std::vector<dftfe::uInt> centerSplineIdHost;
    for (dftfe::uInt iCenter = 0; iCenter < nAtoms + d_imageIdsTrunc.size();
         ++iCenter)
      {
        const bool        isImage = iCenter >= nAtoms;
        const dftfe::uInt iAtom =
          isImage ? d_imageIdsTrunc[iCenter - nAtoms] : iCenter;
        const auto splineIt =
          znumToSplineId.find(std::round(atomLocations[iAtom][0]));
        if (splineIt == znumToSplineId.end())
          continue;
        for (dftfe::uInt iDim = 0; iDim < 3; ++iDim)
          centerCoordsHost.push_back(
            isImage ? d_imagePositionsTrunc[iCenter - nAtoms][iDim] :
                      atomLocations[iAtom][2 + iDim]);
        centerSplineIdHost.push_back(splineIt->second);
      }
    const dftfe::uInt nCenters = centerSplineIdHost.size();

    const dftfe::utils::MemoryStorage<double, memorySpace> dCenterCoords =
      dftfe::utils::memoryStorageFromSTL<double, memorySpace>(centerCoordsHost);
    const dftfe::utils::MemoryStorage<dftfe::uInt, memorySpace>
      dCenterSplineId =
        dftfe::utils::memoryStorageFromSTL<dftfe::uInt, memorySpace>(
          centerSplineIdHost);
    dftfe::utils::MemoryStorage<double, memorySpace> dQuad, dValue, dGrad;

    // block over cells so the device transient stays bounded. Each cell
    // stages three coordinates per quadrature point and returns the density
    // and its gradient, so seven doubles per point.
    const dftfe::uInt cellsBlockSize =
      std::max<dftfe::uInt>(1,
                            atomicInitKernels::maxBlockBytes /
                              (std::size_t)(7 * nQuadsPerCell *
                                            sizeof(double)));
    std::vector<double> hValue, hGrad;
    for (dftfe::uInt cellBlockStart = 0; cellBlockStart < nCells;
         cellBlockStart += cellsBlockSize)
      {
        const dftfe::uInt nCellsInBlock =
          std::min(cellsBlockSize, nCells - cellBlockStart);
        const std::size_t nPointsInBlock =
          (std::size_t)nCellsInBlock * nQuadsPerCell;
        dQuad.resize(nPointsInBlock * 3);
        dQuad.template copyFrom<dftfe::utils::MemorySpace::HOST>(
          d_basisOperationsPtrHost->quadPoints().data() +
          (std::size_t)cellBlockStart * nQuadsPerCell * 3);
        dValue.resize(nPointsInBlock);
        dGrad.resize(nPointsInBlock * 3);

        atomicInitKernels::atomicRadialDensityQuadTotal<memorySpace>(
          nCellsInBlock,
          nQuadsPerCell,
          nCenters,
          d_dftParamsPtr->floatingNuclearCharges,
          dQuad.data(),
          dCenterCoords.data(),
          dCenterSplineId.data(),
          tables.view(),
          dValue.data(),
          dGrad.data(),
          d_nOMPThreads);

        hValue.resize(nPointsInBlock);
        dValue.copyTo(hValue);
        hGrad.resize(nPointsInBlock * 3);
        dGrad.copyTo(hGrad);
        for (dftfe::uInt iCell = 0; iCell < nCellsInBlock; ++iCell)
          {
            const dealii::CellId cid =
              d_basisOperationsPtrHost->cellID(cellBlockStart + iCell);
            valueMap[cid].assign(hValue.begin() +
                                   (std::size_t)iCell * nQuadsPerCell,
                                 hValue.begin() +
                                   (std::size_t)(iCell + 1) * nQuadsPerCell);
            gradMap[cid].assign(hGrad.begin() +
                                  (std::size_t)iCell * nQuadsPerCell * 3,
                                hGrad.begin() +
                                  (std::size_t)(iCell + 1) * nQuadsPerCell * 3);
          }
      }
  }


  //
  // Normalize rho
  //
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::normalizeAtomicRhoQuadValues()
  {
    const double charge  = totalCharge(dofHandler, &d_rhoAtomsValues);
    const double scaling = ((double)numElectrons) / charge;

    if (d_dftParamsPtr->verbosity >= 2)
      pcout
        << "Total charge rho single atomic before normalizing to number of electrons: "
        << charge << std::endl;

    for (auto it1 = d_rhoAtomsValues.begin(); it1 != d_rhoAtomsValues.end();
         ++it1)
      for (dftfe::uInt i = 0; i < (it1->second).size(); ++i)
        (it1->second)[i] *= scaling;

    for (auto it1 = d_gradRhoAtomsValues.begin();
         it1 != d_gradRhoAtomsValues.end();
         ++it1)
      for (dftfe::uInt i = 0; i < (it1->second).size(); ++i)
        (it1->second)[i] *= scaling;

    double chargeAfterScaling = totalCharge(dofHandler, &d_rhoAtomsValues);

    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "Total charge rho single atomic after normalizing: "
            << chargeAfterScaling << std::endl;
  }

  //
  //
  //
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::addAtomicRhoQuadValuesGradients(
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &quadratureValueData,
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
              &quadratureGradValueData,
    const bool isConsiderGradData)
  {
    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureId);
    const dftfe::uInt nQuadsPerCell = d_basisOperationsPtrHost->nQuadsPerCell();
    const dftfe::uInt nCells        = d_basisOperationsPtrHost->nCells();

    for (dftfe::uInt iCell = 0; iCell < nCells; ++iCell)
      {
        const std::vector<double> &rhoAtomicValues =
          d_rhoAtomsValues.find(d_basisOperationsPtrHost->cellID(iCell))
            ->second;
        for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
          quadratureValueData[iCell * nQuadsPerCell + iQuad] +=
            rhoAtomicValues[iQuad];

        if (isConsiderGradData)
          {
            const std::vector<double> &gradRhoAtomicValues =
              d_gradRhoAtomsValues
                .find(d_basisOperationsPtrHost->cellID(iCell))
                ->second;
            for (dftfe::uInt iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
              {
                quadratureGradValueData[iCell * nQuadsPerCell * 3 + 3 * iQuad +
                                        0] +=
                  gradRhoAtomicValues[3 * iQuad + 0];
                quadratureGradValueData[iCell * nQuadsPerCell * 3 + 3 * iQuad +
                                        1] +=
                  gradRhoAtomicValues[3 * iQuad + 1];
                quadratureGradValueData[iCell * nQuadsPerCell * 3 + 3 * iQuad +
                                        2] +=
                  gradRhoAtomicValues[3 * iQuad + 2];
              }
          }
      }
  }

  //
  // compute l2 projection of quad data to nodal data
  //
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::l2ProjectionQuadDensityMinusAtomicDensity(
    const std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
                                            &basisOperationsPtr,
    const dealii::AffineConstraints<double> &constraintMatrix,
    const dftfe::uInt                        dofHandlerId,
    const dftfe::uInt                        quadratureId,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                              &quadratureValueData,
    distributedCPUVec<double> &nodalField)
  {
    basisOperationsPtr->reinit(0, 0, quadratureId);
    const dftfe::uInt nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
    std::function<
      double(const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
             const dftfe::uInt                                           q)>
      funcRho =
        [&](const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
            const dftfe::uInt                                           q) {
          return (
            quadratureValueData[basisOperationsPtr->cellIndex(cell->id()) *
                                  nQuadsPerCell +
                                q] -
            d_rhoAtomsValues.find(cell->id())->second[q]);
        };
    dealii::VectorTools::project<3, distributedCPUVec<double>>(
      dealii::MappingQ1<3, 3>(),
      basisOperationsPtr->matrixFreeData().get_dof_handler(dofHandlerId),
      constraintMatrix,
      basisOperationsPtr->matrixFreeData().get_quadrature(quadratureId),
      funcRho,
      nodalField);
    constraintMatrix.set_zero(nodalField);
    nodalField.update_ghost_values();
  }
#include "dft.inst.cc"

} // namespace dftfe
