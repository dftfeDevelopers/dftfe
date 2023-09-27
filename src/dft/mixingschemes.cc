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
// @author Shiva Rudraraju, Phani Motamarri, Krishnendu Ghosh
//

// source file for all the mixing schemes
#include <dft.h>
#include <linearAlgebraOperations.h>

namespace dftfe
{
  // implement simple mixing scheme
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::mixing_simple()
  {
    double                       normValue = 0.0;
    const dealii::Quadrature<3> &quadrature =
      matrix_free_data.get_quadrature(d_densityQuadratureId);
    dealii::FEValues<3> fe_values(FE, quadrature, dealii::update_JxW_values);
    const unsigned int  num_quad_points = quadrature.size();


    // create new rhoValue tables
    std::map<dealii::CellId, std::vector<double>> rhoInValuesOld = *rhoInValues;

    rhoInValues =  std::make_shared<std::map<dealii::CellId, std::vector<double>>>();

    // create new gradRhoValue tables
    std::map<dealii::CellId, std::vector<double>> gradRhoInValuesOld;

    if (d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
      {
        gradRhoInValuesOld = *gradRhoInValues;
        gradRhoInValues =  std::make_shared<std::map<dealii::CellId, std::vector<double>>>();

      }

    // parallel loop over all elements
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            (*rhoInValues)[cell->id()] = std::vector<double>(num_quad_points);


            if (d_excManagerPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA)
              (*gradRhoInValues)[cell->id()] =
                std::vector<double>(3 * num_quad_points);


            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
              {
                // Compute (rhoIn-rhoOut)^2
                normValue += std::pow(((rhoInValuesOld)[cell->id()][q_point]) -
                                        ((*rhoOutValues)[cell->id()][q_point]),
                                      2.0) *
                             fe_values.JxW(q_point);

                // Simple mixing scheme
                ((*rhoInValues)[cell->id()][q_point]) =
                  ((1 - d_dftParamsPtr->mixingParameter) *
                     (rhoInValuesOld)[cell->id()][q_point] +
                   d_dftParamsPtr->mixingParameter *
                     (*rhoOutValues)[cell->id()][q_point]);


                if (d_excManagerPtr->getDensityBasedFamilyType() ==
                    densityFamilyType::GGA)
                  {
                    ((*gradRhoInValues)[cell->id()][3 * q_point + 0]) =
                      ((1 - d_dftParamsPtr->mixingParameter) *
                         (gradRhoInValuesOld)[cell->id()][3 * q_point + 0] +
                       d_dftParamsPtr->mixingParameter *
                         (*gradRhoOutValues)[cell->id()][3 * q_point + 0]);
                    ((*gradRhoInValues)[cell->id()][3 * q_point + 1]) =
                      ((1 - d_dftParamsPtr->mixingParameter) *
                         (gradRhoInValuesOld)[cell->id()][3 * q_point + 1] +
                       d_dftParamsPtr->mixingParameter *
                         (*gradRhoOutValues)[cell->id()][3 * q_point + 1]);
                    ((*gradRhoInValues)[cell->id()][3 * q_point + 2]) =
                      ((1 - d_dftParamsPtr->mixingParameter) *
                         (gradRhoInValuesOld)[cell->id()][3 * q_point + 2] +
                       d_dftParamsPtr->mixingParameter *
                         (*gradRhoOutValues)[cell->id()][3 * q_point + 2]);
                  }
              }
          }
      }

    return std::sqrt(dealii::Utilities::MPI::sum(normValue, mpi_communicator));
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::mixing_simple_spinPolarized()
  {
    double                       normValue = 0.0;
    const dealii::Quadrature<3> &quadrature =
      matrix_free_data.get_quadrature(d_densityQuadratureId);
    dealii::FEValues<3> fe_values(FE, quadrature, dealii::update_JxW_values);
    const unsigned int  num_quad_points = quadrature.size();

    // create new rhoValue tables
    std::map<dealii::CellId, std::vector<double>> rhoInValuesOld = *rhoInValues;
    rhoInValues =  std::make_shared<std::map<dealii::CellId, std::vector<double>>>();

    std::map<dealii::CellId, std::vector<double>> rhoInValuesOldSpinPolarized =
      *rhoInValuesSpinPolarized;

    rhoInValuesSpinPolarized =  std::make_shared<std::map<dealii::CellId, std::vector<double>>>();
    //

    // create new gradRhoValue tables
    std::map<dealii::CellId, std::vector<double>> gradRhoInValuesOld;
    std::map<dealii::CellId, std::vector<double>>
      gradRhoInValuesOldSpinPolarized;

    if (d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
      {
        gradRhoInValuesOld = *gradRhoInValues;

        gradRhoInValues =  std::make_shared<std::map<dealii::CellId, std::vector<double>>>();
        //
        gradRhoInValuesOldSpinPolarized = *gradRhoInValuesSpinPolarized;

        gradRhoInValuesSpinPolarized =  std::make_shared<std::map<dealii::CellId, std::vector<double>>>();
      }

    // parallel loop over all elements
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            // if (s==0) {
            (*rhoInValuesSpinPolarized)[cell->id()] =
              std::vector<double>(2 * num_quad_points);
            (*rhoInValues)[cell->id()] = std::vector<double>(num_quad_points);
            // }

            if (d_excManagerPtr->getDensityBasedFamilyType() ==
                densityFamilyType::GGA)
              {
                (*gradRhoInValues)[cell->id()] =
                  std::vector<double>(3 * num_quad_points);
                (*gradRhoInValuesSpinPolarized)[cell->id()] =
                  std::vector<double>(6 * num_quad_points);
              }


            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
              {
                // Compute (rhoIn-rhoOut)^2
                // normValue+=std::pow(((*rhoInValuesOld)[cell->id()][2*q_point+s])-
                // ((*rhoOutValues)[cell->id()][2*q_point+s]),2.0)*fe_values.JxW(q_point);

                // Simple mixing scheme
                (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point] =
                  ((1 - d_dftParamsPtr->mixingParameter) *
                     (rhoInValuesOldSpinPolarized)[cell->id()][2 * q_point] +
                   d_dftParamsPtr->mixingParameter *
                     (*rhoOutValuesSpinPolarized)[cell->id()][2 * q_point]);
                (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point + 1] =
                  ((1 - d_dftParamsPtr->mixingParameter) *
                     (rhoInValuesOldSpinPolarized)[cell->id()]
                                                  [2 * q_point + 1] +
                   d_dftParamsPtr->mixingParameter *
                     (*rhoOutValuesSpinPolarized)[cell->id()][2 * q_point + 1]);

                (*rhoInValues)[cell->id()][q_point] =
                  (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point] +
                  (*rhoInValuesSpinPolarized)[cell->id()][2 * q_point + 1];
                //
                normValue += std::pow((rhoInValuesOld)[cell->id()][q_point] -
                                        (*rhoOutValues)[cell->id()][q_point],
                                      2.0) *
                             fe_values.JxW(q_point);

                if (d_excManagerPtr->getDensityBasedFamilyType() ==
                    densityFamilyType::GGA)
                  {
                    for (unsigned int i = 0; i < 6; ++i)
                      {
                        ((*gradRhoInValuesSpinPolarized)[cell->id()]
                                                        [6 * q_point + i]) =
                          ((1 - d_dftParamsPtr->mixingParameter) *
                             (gradRhoInValuesOldSpinPolarized)[cell->id()]
                                                              [6 * q_point +
                                                               i] +
                           d_dftParamsPtr->mixingParameter *
                             (*gradRhoOutValuesSpinPolarized)[cell->id()]
                                                             [6 * q_point + i]);
                      }

                    //
                    ((*gradRhoInValues)[cell->id()][3 * q_point + 0]) =
                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
                                                      [6 * q_point + 0]) +
                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
                                                      [6 * q_point + 3]);
                    ((*gradRhoInValues)[cell->id()][3 * q_point + 1]) =
                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
                                                      [6 * q_point + 1]) +
                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
                                                      [6 * q_point + 4]);
                    ((*gradRhoInValues)[cell->id()][3 * q_point + 2]) =
                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
                                                      [6 * q_point + 2]) +
                      ((*gradRhoInValuesSpinPolarized)[cell->id()]
                                                      [6 * q_point + 5]);
                  }
              }
          }
      }

    return std::sqrt(dealii::Utilities::MPI::sum(normValue, mpi_communicator));
  }

#include "dft.inst.cc"
} // namespace dftfe
