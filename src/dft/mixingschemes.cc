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
    const unsigned int  numCells        = matrix_free_data.n_physical_cells();


    // create new rhoValue tables
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      rhoInValuesOld = d_densityInQuadValues;

    d_densityInQuadValues.resize(d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
    for (unsigned int iComp = 0; iComp < d_densityInQuadValues.size(); ++iComp)
      d_densityInQuadValues[iComp].resize(numCells * num_quad_points);
    std::vector<
      dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>>
      gradRhoInValuesOld;

    if (d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA)
      {
        gradRhoInValuesOld = d_gradDensityInQuadValues;
        d_gradDensityInQuadValues.resize(
          d_dftParamsPtr->spinPolarized == 1 ? 2 : 1);
        for (unsigned int iComp = 0; iComp < d_gradDensityInQuadValues.size();
             ++iComp)
          d_gradDensityInQuadValues[iComp].resize(3 * numCells *
                                                  num_quad_points);
      }

    // parallel loop over all elements
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell             = dofHandler.begin_active(),
      endc             = dofHandler.end();
    unsigned int iCell = 0;
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
              {
                // Compute (rhoIn-rhoOut)^2
                normValue +=
                  std::pow(
                    rhoInValuesOld[0][iCell * num_quad_points + q_point] -
                      d_densityOutQuadValues[0]
                                            [iCell * num_quad_points + q_point],
                    2.0) *
                  fe_values.JxW(q_point);

                // Simple mixing scheme
                for (unsigned int iComp = 0;
                     iComp < d_densityInQuadValues.size();
                     ++iComp)
                  d_densityInQuadValues[iComp][iCell * num_quad_points +
                                               q_point] =
                    ((1 - d_dftParamsPtr->mixingParameter) *
                       rhoInValuesOld[iComp]
                                     [iCell * num_quad_points + q_point] +
                     d_dftParamsPtr->mixingParameter *
                       d_densityOutQuadValues[iComp][iCell * num_quad_points +
                                                     q_point]);


                if (d_excManagerPtr->getDensityBasedFamilyType() ==
                    densityFamilyType::GGA)
                  {
                    for (unsigned int iComp = 0;
                         iComp < d_densityInQuadValues.size();
                         ++iComp)
                      for (unsigned int iDim = 0;
                           iDim < d_densityInQuadValues.size();
                           ++iDim)
                        d_gradDensityInQuadValues[iComp][iCell * 3 *
                                                           num_quad_points +
                                                         3 * q_point + iDim] =
                          ((1 - d_dftParamsPtr->mixingParameter) *
                             (gradRhoInValuesOld[iComp]
                                                [iCell * 3 * num_quad_points +
                                                 3 * q_point + iDim]) +
                           d_dftParamsPtr->mixingParameter *
                             (d_gradDensityOutQuadValues[iComp]
                                                        [iCell * 3 *
                                                           num_quad_points +
                                                         3 * q_point + iDim]));
                  }
              }
            ++iCell;
          }
      }

    return std::sqrt(dealii::Utilities::MPI::sum(normValue, mpi_communicator));
  }

#include "dft.inst.cc"
} // namespace dftfe
