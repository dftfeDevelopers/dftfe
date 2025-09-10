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


//
//
#include <dft.h>

namespace dftfe
{
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::getQuadGridGSElectronDensity(
    std::vector<double> &quadPointCoordinates,
    std::vector<double> &quadPointWeights,
    std::vector<double> &totalDensityVals,
    std::vector<double> &magDensityVals) const
  {
    const unsigned int poolId =
      dealii::Utilities::MPI::this_mpi_process(interpoolcomm);
    const unsigned int bandGroupId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);

    quadPointCoordinates.clear();
    quadPointWeights.clear();
    totalDensityVals.clear();
    magDensityVals.clear();

    if (poolId == 0 && bandGroupId == 0)
      {
        const dealii::Quadrature<3> &quadrature_formula =
          matrix_free_data.get_quadrature(d_densityQuadratureId);
        dealii::FEValues<3> fe_values(FE,
                                      quadrature_formula,
                                      dealii::update_quadrature_points |
                                        dealii::update_JxW_values);
        const dftfe::uInt   n_q_points = quadrature_formula.size();

        // loop over elements
        typename dealii::DoFHandler<3>::active_cell_iterator
          cell = dofHandler.begin_active(),
          endc = dofHandler.end();

        for (; cell != endc; ++cell)
          if (cell->is_locally_owned())
            {
              fe_values.reinit(cell);
              const dftfe::uInt cellIndex =
                d_basisOperationsPtrHost->cellIndex(cell->id());


              const double *rhoValues =
                d_densityOutQuadValues[0].data() + cellIndex * n_q_points;
              const double *magValues =
                d_dftParamsPtr->spinPolarized == 1 ?
                  d_densityOutQuadValues[1].data() + cellIndex * n_q_points :
                  NULL;


              for (dftfe::uInt q_point = 0; q_point < n_q_points; ++q_point)
                {
                  const dealii::Point<3> &quadPoint =
                    fe_values.quadrature_point(q_point);
                  const double jxw = fe_values.JxW(q_point);


                  quadPointCoordinates.push_back(quadPoint[0]);
                  quadPointCoordinates.push_back(quadPoint[1]);
                  quadPointCoordinates.push_back(quadPoint[2]);
                  quadPointWeights.push_back(jxw);
                  totalDensityVals.push_back(rhoValues[q_point]);
                  if (d_dftParamsPtr->spinPolarized == 1)
                    magDensityVals.push_back(magValues[q_point]);
                }
            }
      }
  }


  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::setAdditionalExternalPotentialQuadGrid(
    std::vector<double> &additionalExternalPotential) const
  {
    const unsigned int poolId =
      dealii::Utilities::MPI::this_mpi_process(interpoolcomm);
    const unsigned int bandGroupId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);

    if (poolId == 0 && bandGroupId == 0)
      {
        const dealii::Quadrature<3> &quadrature_formula =
          matrix_free_data.get_quadrature(d_densityQuadratureId);
        dealii::FEValues<3> fe_values(FE,
                                      quadrature_formula,
                                      dealii::update_quadrature_points |
                                        dealii::update_JxW_values);
        const dftfe::uInt   n_q_points = quadrature_formula.size();

        AssertThrow(
              n_q_points==additionalExternalPotential.size(),
              dealii::ExcMessage(
                std::string(
                  "Local size quad data supplied to setAdditionalExternalPotentialQuadGrid not
                  consistent with current quad grid")));


        d_additionalExternalPotential.clear();
        d_additionalExternalPotential.resize(additionalExternalPotential.size(),0);
        d_additionalExternalPotential.copyFrom(additionalExternalPotential);
      }

    dftfe::uInt sizeArray = d_additionalExternalPotential.size();
    MPI_Bcast(
      &sizeArray, 1, dataTypes::mpi_type_id(&sizeArray), 0, interpoolcomm);
    MPI_Bcast(
      &sizeArray, 1, dataTypes::mpi_type_id(&sizeArray), 0, interBandGroupComm);
    if (poolId != 0 || bandGroupId != 0)
      {
        d_additionalExternalPotential.clear()' d_additionalExternalPotential
          .resize(sizeArray, 0);
      }

    int size;
    MPI_Comm_size(interpoolcomm, &size);
    if (size > 1)
      MPI_Allreduce(MPI_IN_PLACE,
                    d_additionalExternalPotential.data(),
                    sizeArray,
                    dataTypes::mpi_type_id(
                      d_additionalExternalPotential.data()),
                    MPI_SUM,
                    interpoolcomm);

    int size;
    MPI_Comm_size(interBandGroupComm, &size);
    if (size > 1)
      MPI_Allreduce(MPI_IN_PLACE,
                    d_additionalExternalPotential.data(),
                    sizeArray,
                    dataTypes::mpi_type_id(
                      d_additionalExternalPotential.data()),
                    MPI_SUM,
                    interBandGroupComm);
  }
#include "dft.inst.cc"
} // namespace dftfe
