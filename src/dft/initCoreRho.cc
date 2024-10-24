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
// @author Phani Motamarri
//

//
// Initialize rho by reading in single-atom electron-density and fit a spline
//
#include <dftParameters.h>
#include <dft.h>
#include <fileReaders.h>

namespace dftfe
{
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::initCoreRho()
  {
    // clear existing data
    d_rhoCore.clear();
    d_gradRhoCore.clear();
    d_gradRhoCoreAtoms.clear();
    d_hessianRhoCore.clear();
    d_hessianRhoCoreAtoms.clear();

    // Reading single atom rho initial guess
    pcout
      << std::endl
      << "Reading data for core electron-density to be used in nonlinear core-correction....."
      << std::endl;
    std::map<unsigned int, alglib::spline1dinterpolant> coreDenSpline;
    std::map<unsigned int, std::vector<std::vector<double>>>
                                   singleAtomCoreElectronDensity;
    std::map<unsigned int, double> outerMostPointCoreDen;
    const double                   truncationTol = 1e-12;
    unsigned int                   fileReadFlag  = 0;

    double maxCoreRhoTail = 0.0;
    // loop over atom types
    for (std::set<unsigned int>::iterator it = atomTypes.begin();
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
    const unsigned int  n_q_points = quadrature_formula.size();

    //
    // get number of global charges
    //
    const int numberGlobalCharges = atomLocations.size();

    //
    // get number of image charges used only for periodic
    //
    const int numberImageCharges = d_imageIdsTrunc.size();

    //
    // loop over elements
    //
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    dealii::Tensor<1, 3, double> zeroTensor1;
    for (unsigned int i = 0; i < 3; i++)
      zeroTensor1[i] = 0.0;

    dealii::Tensor<2, 3, double> zeroTensor2;

    for (unsigned int i = 0; i < 3; i++)
      for (unsigned int j = 0; j < 3; j++)
        zeroTensor2[i][j] = 0.0;

    bool isGradDensityDataDependent =
      (d_excManagerPtr->getExcSSDFunctionalObj()->getDensityBasedFamilyType() ==
       densityFamilyType::GGA);

    // loop over elements
    //
    cell = dofHandler.begin_active();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);

            std::vector<double> &rhoCoreQuadValues = d_rhoCore[cell->id()];
            rhoCoreQuadValues.resize(n_q_points, 0.0);

            std::vector<double> &gradRhoCoreQuadValues =
              d_gradRhoCore[cell->id()];
            gradRhoCoreQuadValues.resize(n_q_points * 3, 0.0);

            std::vector<double> &hessianRhoCoreQuadValues =
              d_hessianRhoCore[cell->id()];
            if (isGradDensityDataDependent)
              hessianRhoCoreQuadValues.resize(n_q_points * 9, 0.0);

            std::vector<dealii::Tensor<1, 3, double>> gradRhoCoreAtom(
              n_q_points, zeroTensor1);
            std::vector<dealii::Tensor<2, 3, double>> hessianRhoCoreAtom(
              n_q_points, zeroTensor2);


            // loop over atoms
            for (unsigned int iAtom = 0; iAtom < atomLocations.size(); ++iAtom)
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
                        for (unsigned int iDim = 0; iDim < 3; ++iDim)
                          {
                            for (unsigned int jDim = 0; jDim < 3; ++jDim)
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
                                hessianRhoCoreQuadValues[9 * q + 3 * iDim +
                                                         jDim] += temp;
                              }
                          }
                      }


                  } // end loop over quad points
                if (isCoreRhoDataInCell)
                  {
                    std::vector<double> &gradRhoCoreAtomCell =
                      d_gradRhoCoreAtoms[iAtom][cell->id()];
                    gradRhoCoreAtomCell.resize(n_q_points * 3, 0.0);

                    std::vector<double> &hessianRhoCoreAtomCell =
                      d_hessianRhoCoreAtoms[iAtom][cell->id()];
                    if (isGradDensityDataDependent)
                      hessianRhoCoreAtomCell.resize(n_q_points * 9, 0.0);

                    for (unsigned int q = 0; q < n_q_points; ++q)
                      {
                        gradRhoCoreAtomCell[3 * q + 0] = gradRhoCoreAtom[q][0];
                        gradRhoCoreAtomCell[3 * q + 1] = gradRhoCoreAtom[q][1];
                        gradRhoCoreAtomCell[3 * q + 2] = gradRhoCoreAtom[q][2];

                        if (isGradDensityDataDependent)
                          {
                            for (unsigned int iDim = 0; iDim < 3; ++iDim)
                              {
                                for (unsigned int jDim = 0; jDim < 3; ++jDim)
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
            for (unsigned int iImageCharge = 0;
                 iImageCharge < numberImageCharges;
                 ++iImageCharge)
              {
                const int masterAtomId = d_imageIdsTrunc[iImageCharge];
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
                        for (unsigned int iDim = 0; iDim < 3; ++iDim)
                          {
                            for (unsigned int jDim = 0; jDim < 3; ++jDim)
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
                                hessianRhoCoreQuadValues[9 * q + 3 * iDim +
                                                         jDim] += temp;
                              }
                          }
                      }

                  } // quad point loop

                if (isCoreRhoDataInCell)
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

                    for (unsigned int q = 0; q < n_q_points; ++q)
                      {
                        gradRhoCoreAtomCell[3 * q + 0] = gradRhoCoreAtom[q][0];
                        gradRhoCoreAtomCell[3 * q + 1] = gradRhoCoreAtom[q][1];
                        gradRhoCoreAtomCell[3 * q + 2] = gradRhoCoreAtom[q][2];

                        if (isGradDensityDataDependent)
                          {
                            for (unsigned int iDim = 0; iDim < 3; ++iDim)
                              {
                                for (unsigned int jDim = 0; jDim < 3; ++jDim)
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
