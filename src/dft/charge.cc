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
// @author Shiva Rudraraju, Phani Motamarri
//

// source file for all charge calculations

//
// compute total charge using quad point values
//
#include <dft.h>

namespace dftfe
{
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::totalCharge(
    const dealii::DoFHandler<3> &                        dofHandlerOfField,
    const std::map<dealii::CellId, std::vector<double>> *rhoQuadValues)
  {
    double                       normValue = 0.0;
    const dealii::Quadrature<3> &quadrature_formula =
      matrix_free_data.get_quadrature(d_densityQuadratureId);
    dealii::FEValues<3> fe_values(dofHandlerOfField.get_fe(),
                                  quadrature_formula,
                                  dealii::update_JxW_values);
    const unsigned int dofs_per_cell = dofHandlerOfField.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    dealii::DoFHandler<3>::active_cell_iterator cell = dofHandlerOfField
                                                         .begin_active(),
                                                endc = dofHandlerOfField.end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            const std::vector<double> &rhoValues =
              (*rhoQuadValues).find(cell->id())->second;
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              {
                normValue += rhoValues[q_point] * fe_values.JxW(q_point);
              }
          }
      }
    return dealii::Utilities::MPI::sum(normValue, mpi_communicator);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::totalCharge(
    const dealii::DoFHandler<3> &dofHandlerOfField,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &rhoQuadValues)
  {
    double                       normValue = 0.0;
    const dealii::Quadrature<3> &quadrature_formula =
      matrix_free_data.get_quadrature(d_densityQuadratureId);
    dealii::FEValues<3> fe_values(dofHandlerOfField.get_fe(),
                                  quadrature_formula,
                                  dealii::update_JxW_values);
    const unsigned int dofs_per_cell = dofHandlerOfField.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();

    dealii::DoFHandler<3>::active_cell_iterator cell = dofHandlerOfField
                                                         .begin_active(),
                                                endc = dofHandlerOfField.end();
    unsigned int iCell                               = 0;
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            const double *rhoValues = rhoQuadValues.data() + iCell * n_q_points;
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              {
                normValue += rhoValues[q_point] * fe_values.JxW(q_point);
              }
            ++iCell;
          }
      }
    return dealii::Utilities::MPI::sum(normValue, mpi_communicator);
  }


  //
  // compute total charge using nodal point values
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::totalCharge(
    const dealii::DoFHandler<3> &    dofHandlerOfField,
    const distributedCPUVec<double> &rhoNodalField)
  {
    double                       normValue = 0.0;
    const dealii::Quadrature<3> &quadrature_formula =
      matrix_free_data.get_quadrature(d_densityQuadratureId);
    dealii::FEValues<3> fe_values(dofHandlerOfField.get_fe(),
                                  quadrature_formula,
                                  dealii::update_values |
                                    dealii::update_JxW_values);
    const unsigned int dofs_per_cell = dofHandlerOfField.get_fe().dofs_per_cell;
    const unsigned int n_q_points    = quadrature_formula.size();
    rhoNodalField.update_ghost_values();
    dealii::DoFHandler<3>::active_cell_iterator cell = dofHandlerOfField
                                                         .begin_active(),
                                                endc = dofHandlerOfField.end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            std::vector<double> tempRho(n_q_points);
            fe_values.get_function_values(rhoNodalField, tempRho);
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              {
                normValue += tempRho[q_point] * fe_values.JxW(q_point);
              }
          }
      }
    return dealii::Utilities::MPI::sum(normValue, mpi_communicator);
  }

  //
  // compute total charge using nodal point values by using FEEvaluation object
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::totalCharge(
    const dealii::MatrixFree<3, double> &matrixFreeDataObject,
    const distributedCPUVec<double> &    nodalField)
  {
    dealii::FEEvaluation<
      3,
      C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      1,
      double>
                                    fe_evalField(matrixFreeDataObject,
                   d_densityDofHandlerIndexElectro,
                   d_densityQuadratureIdElectro);
    dealii::VectorizedArray<double> normValueVectorized =
      dealii::make_vectorized_array(0.0);
    const unsigned int numQuadPoints = fe_evalField.n_q_points;
    nodalField.update_ghost_values();
    // AssertThrow(nodalField.partitioners_are_globally_compatible(*matrixFreeDataObject.get_vector_partitioner(d_densityDofHandlerIndexElectro)),
    //        dealii::ExcMessage("DFT-FE Error: mismatch in
    //        partitioner/dofHandler."));

    AssertThrow(
      matrixFreeDataObject.get_quadrature(d_densityQuadratureIdElectro)
          .size() == numQuadPoints,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

    double normValue = 0.0;
    for (unsigned int cell = 0; cell < matrixFreeDataObject.n_cell_batches();
         ++cell)
      {
        fe_evalField.reinit(cell);
        fe_evalField.read_dof_values(nodalField);
        fe_evalField.evaluate(true, false);
        for (unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
          {
            dealii::VectorizedArray<double> temp =
              fe_evalField.get_value(q_point);
            fe_evalField.submit_value(temp, q_point);
          }

        normValueVectorized = fe_evalField.integrate_value();

        for (unsigned int iSubCell = 0;
             iSubCell <
             matrixFreeDataObject.n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          {
            normValue += normValueVectorized[iSubCell];
          }
      }

    return dealii::Utilities::MPI::sum(normValue, mpi_communicator);
  }

  //
  // compute total charge
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::totalMagnetization(
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &magQuadValues)
  {
    double                       normValue = 0.0;
    const dealii::Quadrature<3> &quadrature_formula =
      matrix_free_data.get_quadrature(d_densityQuadratureId);
    dealii::FEValues<3> fe_values(FE,
                                  quadrature_formula,
                                  dealii::update_JxW_values);
    const unsigned int  dofs_per_cell = FE.dofs_per_cell;
    const unsigned int  n_q_points    = quadrature_formula.size();

    dealii::DoFHandler<3>::active_cell_iterator cell =
                                                  dofHandler.begin_active(),
                                                endc = dofHandler.end();
    unsigned int iCell                               = 0;
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
              {
                normValue += (magQuadValues[iCell * n_q_points + q_point]) *
                             fe_values.JxW(q_point);
              }
            ++iCell;
          }
      }
    return dealii::Utilities::MPI::sum(normValue, mpi_communicator);
  }

  //
  // compute field l2 norm
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::rhofieldl2Norm(
    const dealii::MatrixFree<3, double> &matrixFreeDataObject,
    const distributedCPUVec<double> &    nodalField,
    const unsigned int                   dofHandlerId,
    const unsigned int                   quadratureId)

  {
    dealii::FEEvaluation<
      3,
      C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      1,
      double>
                                    fe_evalField(matrixFreeDataObject, dofHandlerId, quadratureId);
    dealii::VectorizedArray<double> normValueVectorized =
      dealii::make_vectorized_array(0.0);
    const unsigned int numQuadPoints = fe_evalField.n_q_points;
    nodalField.update_ghost_values();
    AssertThrow(
      matrixFreeDataObject.get_quadrature(quadratureId).size() == numQuadPoints,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

    double normValue = 0.0;
    for (unsigned int cell = 0; cell < matrixFreeDataObject.n_cell_batches();
         ++cell)
      {
        fe_evalField.reinit(cell);
        fe_evalField.read_dof_values(nodalField);
        fe_evalField.evaluate(true, false);
        for (unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
          {
            dealii::VectorizedArray<double> temp =
              fe_evalField.get_value(q_point) * fe_evalField.get_value(q_point);
            fe_evalField.submit_value(temp, q_point);
          }

        normValueVectorized = fe_evalField.integrate_value();

        for (unsigned int iSubCell = 0;
             iSubCell <
             matrixFreeDataObject.n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          {
            normValue += normValueVectorized[iSubCell];
          }
      }

    return std::sqrt(dealii::Utilities::MPI::sum(normValue, mpi_communicator));
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::rhofieldInnerProduct(
    const dealii::MatrixFree<3, double> &matrixFreeDataObject,
    const distributedCPUVec<double> &    nodalField1,
    const distributedCPUVec<double> &    nodalField2,
    const unsigned int                   dofHandlerId,
    const unsigned int                   quadratureId)

  {
    dealii::FEEvaluation<
      3,
      C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      1,
      double>
                                    fe_evalField(matrixFreeDataObject, dofHandlerId, quadratureId);
    dealii::VectorizedArray<double> valueVectorized =
      dealii::make_vectorized_array(0.0);
    const unsigned int numQuadPoints = fe_evalField.n_q_points;
    nodalField1.update_ghost_values();
    nodalField2.update_ghost_values();
    AssertThrow(
      matrixFreeDataObject.get_quadrature(quadratureId).size() == numQuadPoints,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

    double value = 0.0;
    for (unsigned int cell = 0; cell < matrixFreeDataObject.n_cell_batches();
         ++cell)
      {
        fe_evalField.reinit(cell);
        fe_evalField.read_dof_values(nodalField1);
        fe_evalField.evaluate(true, false);
        dealii::AlignedVector<dealii::VectorizedArray<double>> temp1(
          numQuadPoints, dealii::make_vectorized_array(0.0));
        for (unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
          {
            temp1[q_point] = fe_evalField.get_value(q_point);
          }

        fe_evalField.read_dof_values(nodalField2);
        fe_evalField.evaluate(true, false);
        dealii::AlignedVector<dealii::VectorizedArray<double>> temp2(
          numQuadPoints, dealii::make_vectorized_array(0.0));
        for (unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
          {
            temp2[q_point] = fe_evalField.get_value(q_point);
          }

        for (unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
          {
            fe_evalField.submit_value(temp1[q_point] * temp2[q_point], q_point);
          }


        valueVectorized = fe_evalField.integrate_value();

        for (unsigned int iSubCell = 0;
             iSubCell <
             matrixFreeDataObject.n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          {
            value += valueVectorized[iSubCell];
          }
      }

    return dealii::Utilities::MPI::sum(value, mpi_communicator);
  }
#include "dft.inst.cc"

} // namespace dftfe
