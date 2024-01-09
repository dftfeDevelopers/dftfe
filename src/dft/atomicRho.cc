// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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
#include <dft.h>
#include <fileReaders.h>

namespace dftfe
{
  //
  // Initialize rho by reading in single-atom electron-density and fit a spline
  //

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::initAtomicRho()
  {
    // clear existing data
    d_rhoAtomsValues.clear();
    d_gradRhoAtomsValues.clear();
    d_hessianRhoAtomsValues.clear();
    d_rhoAtomsValues.clear();
    d_gradRhoAtomsValuesSeparate.clear();
    d_hessianRhoAtomsValuesSeparate.clear();

    // Reading single atom rho initial guess
    pcout << std::endl
          << "Reading initial guess for electron-density....." << std::endl;
    std::map<unsigned int, alglib::spline1dinterpolant> denSpline;
    std::map<unsigned int, std::vector<std::vector<double>>>
                                   singleAtomElectronDensity;
    std::map<unsigned int, double> outerMostPointDen;
    const double                   truncationTol = 1e-10;
    double                         maxRhoTail    = 0.0;

    // loop over atom types
    for (std::set<unsigned int>::iterator it = atomTypes.begin();
         it != atomTypes.end();
         it++)
      {
        char densityFile[256];
        if (d_dftParamsPtr->isPseudopotential)
          {
            strcpy(densityFile,
                   (d_dftfeScratchFolderName + "/z" + std::to_string(*it) +
                    "/density.inp")
                     .c_str());
          }
        else
          {
            sprintf(
              densityFile,
              "%s/data/electronicStructure/allElectron/z%u/singleAtomData/density.inp",
              DFTFE_PATH,
              *it);
          }

        dftUtils::readFile(2, singleAtomElectronDensity[*it], densityFile);
        unsigned int        numRows = singleAtomElectronDensity[*it].size() - 1;
        std::vector<double> xData(numRows), yData(numRows);

        unsigned int maxRowId = 0;
        for (unsigned int irow = 0; irow < numRows; ++irow)
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
    const unsigned int  n_q_points = quadrature_formula.size();

    //
    // get number of global charges
    //
    const int numberGlobalCharges = atomLocations.size();

    //
    // get number of image charges used only for periodic
    //
    const int numberImageCharges = d_imageIdsTrunc.size();

    dealii::Tensor<1, 3, double> zeroTensor1;
    for (unsigned int i = 0; i < 3; i++)
      zeroTensor1[i] = 0.0;

    dealii::Tensor<2, 3, double> zeroTensor2;

    for (unsigned int i = 0; i < 3; i++)
      for (unsigned int j = 0; j < 3; j++)
        zeroTensor2[i][j] = 0.0;

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

            std::vector<double> &hessianRhoAtomsQuadValues =
              d_hessianRhoAtomsValues[cell->id()];
            if (d_excManagerPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA)
              hessianRhoAtomsQuadValues.resize(n_q_points * 9, 0.0);

            std::vector<double>                       rhoAtom(n_q_points, 0.0);
            std::vector<dealii::Tensor<1, 3, double>> gradRhoAtom(n_q_points,
                                                                  zeroTensor1);
            std::vector<dealii::Tensor<2, 3, double>> hessianRhoAtom(
              n_q_points, zeroTensor2);


            // loop over atoms
            for (unsigned int iAtom = 0; iAtom < atomLocations.size(); ++iAtom)
              {
                dealii::Point<3> atom(atomLocations[iAtom][2],
                                      atomLocations[iAtom][3],
                                      atomLocations[iAtom][4]);
                bool             isRhoDataInCell = false;

                if (atom.distance(cell->center()) > cellCenterCutOff)
                  continue;

                // loop over quad points
                for (unsigned int q = 0; q < n_q_points; ++q)
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
                      {
                        alglib::spline1ddiff(denSpline[atomLocations[iAtom][0]],
                                             distanceToAtom,
                                             value,
                                             radialDensityFirstDerivative,
                                             radialDensitySecondDerivative);

                        isRhoDataInCell = true;
                      }
                    else
                      {
                        value                         = 0.0;
                        radialDensityFirstDerivative  = 0.0;
                        radialDensitySecondDerivative = 0.0;
                      }

                    rhoAtom[q] = value;
                    rhoAtomsQuadValues[q] += value;
                    gradRhoAtom[q] =
                      radialDensityFirstDerivative * diff / distanceToAtom;
                    gradRhoAtomsQuadValues[3 * q + 0] += gradRhoAtom[q][0];
                    gradRhoAtomsQuadValues[3 * q + 1] += gradRhoAtom[q][1];
                    gradRhoAtomsQuadValues[3 * q + 2] += gradRhoAtom[q][2];

                    if (d_excManagerPtr->getDensityBasedFamilyType() ==
                        densityFamilyType::GGA)
                      {
                        for (unsigned int iDim = 0; iDim < 3; ++iDim)
                          {
                            for (unsigned int jDim = 0; jDim < 3; ++jDim)
                              {
                                double temp = (radialDensitySecondDerivative -
                                               radialDensityFirstDerivative /
                                                 distanceToAtom) *
                                              diff[iDim] * diff[jDim] /
                                              (distanceToAtom * distanceToAtom);
                                if (iDim == jDim)
                                  temp += radialDensityFirstDerivative /
                                          distanceToAtom;

                                hessianRhoAtom[q][iDim][jDim] = temp;
                                hessianRhoAtomsQuadValues[9 * q + 3 * iDim +
                                                          jDim] += temp;
                              }
                          }
                      }


                  } // end loop over quad points

                if (isRhoDataInCell)
                  {
                    d_rhoAtomsValuesSeparate[iAtom][cell->id()] = rhoAtom;

                    std::vector<double> &gradRhoAtomCell =
                      d_gradRhoAtomsValuesSeparate[iAtom][cell->id()];
                    gradRhoAtomCell.resize(n_q_points * 3, 0.0);

                    std::vector<double> &hessianRhoAtomCell =
                      d_hessianRhoAtomsValuesSeparate[iAtom][cell->id()];
                    if (d_excManagerPtr->getDensityBasedFamilyType() ==
                        densityFamilyType::GGA)
                      hessianRhoAtomCell.resize(n_q_points * 9, 0.0);

                    for (unsigned int q = 0; q < n_q_points; ++q)
                      {
                        gradRhoAtomCell[3 * q + 0] = gradRhoAtom[q][0];
                        gradRhoAtomCell[3 * q + 1] = gradRhoAtom[q][1];
                        gradRhoAtomCell[3 * q + 2] = gradRhoAtom[q][2];

                        if (d_excManagerPtr->getDensityBasedFamilyType() ==
                            densityFamilyType::GGA)
                          {
                            for (unsigned int iDim = 0; iDim < 3; ++iDim)
                              {
                                for (unsigned int jDim = 0; jDim < 3; ++jDim)
                                  {
                                    hessianRhoAtomCell[9 * q + 3 * iDim +
                                                       jDim] =
                                      hessianRhoAtom[q][iDim][jDim];
                                  }
                              }
                          }
                      } // q_point loop
                  }     // if loop
              }         // loop over atoms

            // loop over image charges
            for (unsigned int iImageCharge = 0;
                 iImageCharge < numberImageCharges;
                 ++iImageCharge)
              {
                const int masterAtomId = d_imageIdsTrunc[iImageCharge];

                dealii::Point<3> imageAtom(
                  d_imagePositionsTrunc[iImageCharge][0],
                  d_imagePositionsTrunc[iImageCharge][1],
                  d_imagePositionsTrunc[iImageCharge][2]);

                if (imageAtom.distance(cell->center()) > cellCenterCutOff)
                  continue;

                bool isRhoDataInCell = false;

                // loop over quad points
                for (unsigned int q = 0; q < n_q_points; ++q)
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
                      {
                        alglib::spline1ddiff(
                          denSpline[atomLocations[masterAtomId][0]],
                          distanceToAtom,
                          value,
                          radialDensityFirstDerivative,
                          radialDensitySecondDerivative);

                        isRhoDataInCell = true;
                      }
                    else
                      {
                        value                         = 0.0;
                        radialDensityFirstDerivative  = 0.0;
                        radialDensitySecondDerivative = 0.0;
                      }

                    rhoAtom[q] = value;
                    rhoAtomsQuadValues[q] += value;
                    gradRhoAtom[q] =
                      radialDensityFirstDerivative * diff / distanceToAtom;
                    gradRhoAtomsQuadValues[3 * q + 0] += gradRhoAtom[q][0];
                    gradRhoAtomsQuadValues[3 * q + 1] += gradRhoAtom[q][1];
                    gradRhoAtomsQuadValues[3 * q + 2] += gradRhoAtom[q][2];

                    if (d_excManagerPtr->getDensityBasedFamilyType() ==
                        densityFamilyType::GGA)
                      {
                        for (unsigned int iDim = 0; iDim < 3; ++iDim)
                          {
                            for (unsigned int jDim = 0; jDim < 3; ++jDim)
                              {
                                double temp = (radialDensitySecondDerivative -
                                               radialDensityFirstDerivative /
                                                 distanceToAtom) *
                                              diff[iDim] * diff[jDim] /
                                              (distanceToAtom * distanceToAtom);
                                if (iDim == jDim)
                                  temp += radialDensityFirstDerivative /
                                          distanceToAtom;
                                hessianRhoAtom[q][iDim][jDim] = temp;
                                hessianRhoAtomsQuadValues[9 * q + 3 * iDim +
                                                          jDim] += temp;
                              }
                          }
                      }

                  } // quad point loop

                if (isRhoDataInCell)
                  {
                    d_rhoAtomsValuesSeparate[numberGlobalCharges + iImageCharge]
                                            [cell->id()] = rhoAtom;

                    std::vector<double> &gradRhoAtomCell =
                      d_gradRhoAtomsValuesSeparate[numberGlobalCharges +
                                                   iImageCharge][cell->id()];
                    gradRhoAtomCell.resize(n_q_points * 3);

                    std::vector<double> &hessianRhoAtomCell =
                      d_hessianRhoAtomsValuesSeparate[numberGlobalCharges +
                                                      iImageCharge][cell->id()];
                    if (d_excManagerPtr->getDensityBasedFamilyType() ==
                        densityFamilyType::GGA)
                      hessianRhoAtomCell.resize(n_q_points * 9);

                    for (unsigned int q = 0; q < n_q_points; ++q)
                      {
                        gradRhoAtomCell[3 * q + 0] = gradRhoAtom[q][0];
                        gradRhoAtomCell[3 * q + 1] = gradRhoAtom[q][1];
                        gradRhoAtomCell[3 * q + 2] = gradRhoAtom[q][2];

                        if (d_excManagerPtr->getDensityBasedFamilyType() ==
                            densityFamilyType::GGA)
                          {
                            for (unsigned int iDim = 0; iDim < 3; ++iDim)
                              {
                                for (unsigned int jDim = 0; jDim < 3; ++jDim)
                                  {
                                    hessianRhoAtomCell[9 * q + 3 * iDim +
                                                       jDim] =
                                      hessianRhoAtom[q][iDim][jDim];
                                  }
                              }
                          }
                      } // q_point loop
                  }     // if loop

              } // end of image charges

          } // cell locally owned check

      } // cell loop

    normalizeAtomicRhoQuadValues();
  }


  //
  // Normalize rho
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::normalizeAtomicRhoQuadValues()
  {
    const double charge  = totalCharge(dofHandler, &d_rhoAtomsValues);
    const double scaling = ((double)numElectrons) / charge;

    if (d_dftParamsPtr->verbosity >= 2)
      pcout
        << "Total charge rho single atomic before normalizing to number of electrons: "
        << charge << std::endl;


    for (auto it1 = d_rhoAtomsValues.begin(); it1 != d_rhoAtomsValues.end();
         ++it1)
      for (unsigned int i = 0; i < (it1->second).size(); ++i)
        (it1->second)[i] *= scaling;

    for (auto it1 = d_gradRhoAtomsValues.begin();
         it1 != d_gradRhoAtomsValues.end();
         ++it1)
      for (unsigned int i = 0; i < (it1->second).size(); ++i)
        (it1->second)[i] *= scaling;

    for (auto it1 = d_rhoAtomsValuesSeparate.begin();
         it1 != d_rhoAtomsValuesSeparate.end();
         ++it1)
      for (auto it2 = it1->second.begin(); it2 != it1->second.end(); ++it2)
        for (unsigned int i = 0; i < (it2->second).size(); ++i)
          (it2->second)[i] *= scaling;

    for (auto it1 = d_gradRhoAtomsValuesSeparate.begin();
         it1 != d_gradRhoAtomsValuesSeparate.end();
         ++it1)
      for (auto it2 = it1->second.begin(); it2 != it1->second.end(); ++it2)
        for (unsigned int i = 0; i < (it2->second).size(); ++i)
          (it2->second)[i] *= scaling;

    if (d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
      {
        for (auto it1 = d_hessianRhoAtomsValues.begin();
             it1 != d_hessianRhoAtomsValues.end();
             ++it1)
          for (unsigned int i = 0; i < (it1->second).size(); ++i)
            (it1->second)[i] *= scaling;

        for (auto it1 = d_hessianRhoAtomsValuesSeparate.begin();
             it1 != d_hessianRhoAtomsValuesSeparate.end();
             ++it1)
          for (auto it2 = it1->second.begin(); it2 != it1->second.end(); ++it2)
            for (unsigned int i = 0; i < (it2->second).size(); ++i)
              (it2->second)[i] *= scaling;
      }

    double chargeAfterScaling = totalCharge(dofHandler, &d_rhoAtomsValues);

    if (d_dftParamsPtr->verbosity >= 2)
      pcout << "Total charge rho single atomic after normalizing: "
            << chargeAfterScaling << std::endl;
  }

  //
  //
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::addAtomicRhoQuadValuesGradients(
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &quadratureValueData,
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &        quadratureGradValueData,
    const bool isConsiderGradData)
  {
    d_basisOperationsPtrHost->reinit(0, 0, d_densityQuadratureId, false);
    const unsigned int nQuadsPerCell =
      d_basisOperationsPtrHost->nQuadsPerCell();
    const unsigned int nCells = d_basisOperationsPtrHost->nCells();

    for (unsigned int iCell = 0; iCell < nCells; ++iCell)
      {
        const std::vector<double> &rhoAtomicValues =
          d_rhoAtomsValues.find(d_basisOperationsPtrHost->cellID(iCell))
            ->second;
        for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
          quadratureValueData[iCell * nQuadsPerCell + iQuad] +=
            rhoAtomicValues[iQuad];

        if (isConsiderGradData)
          {
            const std::vector<double> &gradRhoAtomicValues =
              d_gradRhoAtomsValues
                .find(d_basisOperationsPtrHost->cellID(iCell))
                ->second;
            for (unsigned int iQuad = 0; iQuad < nQuadsPerCell; ++iQuad)
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
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::subtractAtomicRhoQuadValuesGradients(
    std::map<dealii::CellId, std::vector<double>> &quadratureValueData,
    std::map<dealii::CellId, std::vector<double>> &quadratureGradValueData,
    const bool                                     isConsiderGradData)
  {
    const dealii::Quadrature<3> &quadrature_formula =
      matrix_free_data.get_quadrature(d_densityQuadratureId);
    const unsigned int n_q_points = quadrature_formula.size();

    dealii::DoFHandler<3>::active_cell_iterator cell =
                                                  dofHandler.begin_active(),
                                                endc = dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          std::vector<double> &rhoValues =
            quadratureValueData.find(cell->id())->second;
          const std::vector<double> &rhoAtomicValues =
            d_rhoAtomsValues.find(cell->id())->second;
          for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
            rhoValues[q_point] -= rhoAtomicValues[q_point];

          if (isConsiderGradData)
            {
              std::vector<double> &gradRhoValues =
                quadratureGradValueData.find(cell->id())->second;
              const std::vector<double> &gradRhoAtomicValues =
                d_gradRhoAtomsValues.find(cell->id())->second;
              for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
                {
                  gradRhoValues[3 * q_point + 0] -=
                    gradRhoAtomicValues[3 * q_point + 0];
                  gradRhoValues[3 * q_point + 1] -=
                    gradRhoAtomicValues[3 * q_point + 1];
                  gradRhoValues[3 * q_point + 2] -=
                    gradRhoAtomicValues[3 * q_point + 2];
                }
            }
        }
  }

  //
  // compute l2 projection of quad data to nodal data
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::l2ProjectionQuadDensityMinusAtomicDensity(
    const std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      &                                      basisOperationsPtr,
    const dealii::AffineConstraints<double> &constraintMatrix,
    const unsigned int                       dofHandlerId,
    const unsigned int                       quadratureId,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                        quadratureValueData,
    distributedCPUVec<double> &nodalField)
  {
    basisOperationsPtr->reinit(0, 0, quadratureId, false);
    const unsigned int nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
    std::function<
      double(const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
             const unsigned int                                          q)>
      funcRho =
        [&](const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
            const unsigned int                                          q) {
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
  }
#include "dft.inst.cc"

} // namespace dftfe
