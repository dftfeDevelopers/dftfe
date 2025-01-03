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
// @author Shiva Rudraraju, Phani Motamarri
//

// source file for all charge calculations

//
// compute total charge using quad point values
//
#include <dft.h>

namespace dftfe
{
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::totalCharge(
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

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::totalCharge(
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
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::totalCharge(
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
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::totalCharge(
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
        fe_evalField.evaluate(dealii::EvaluationFlags::values);
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
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::totalMagnetization(
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &magQuadValues)
  {
    double                       normValue    = 0.0;
    double                       absNormValue = 0.0;
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
                absNormValue +=
                  std::abs(magQuadValues[iCell * n_q_points + q_point]) *
                  fe_values.JxW(q_point);
              }
            ++iCell;
          }
      }
    absNormValue = dealii::Utilities::MPI::sum(absNormValue, mpi_communicator);
    normValue    = dealii::Utilities::MPI::sum(normValue, mpi_communicator);
    const auto default_precision{std::cout.precision()};
    if (d_dftParamsPtr->reproducible_output)
      pcout << std::setprecision(3);
    pcout << "Absolute magnetization: " << absNormValue << std::endl;
    pcout << "Net magnetization     : " << normValue << std::endl;
    if (d_dftParamsPtr->reproducible_output)
      pcout << std::setprecision(default_precision);
  }

  //
  // compute field l2 norm
  //
  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::rhofieldl2Norm(
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
        fe_evalField.evaluate(dealii::EvaluationFlags::values);
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


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<FEOrder, FEOrderElectro, memorySpace>::rhofieldInnerProduct(
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
        fe_evalField.evaluate(dealii::EvaluationFlags::values);
        dealii::AlignedVector<dealii::VectorizedArray<double>> temp1(
          numQuadPoints, dealii::make_vectorized_array(0.0));
        for (unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
          {
            temp1[q_point] = fe_evalField.get_value(q_point);
          }

        fe_evalField.read_dof_values(nodalField2);
        fe_evalField.evaluate(dealii::EvaluationFlags::values);
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


  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<FEOrder, FEOrderElectro, memorySpace>::computeMultipoleMoments(
    const std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      &                basisOperationsPtr,
    const unsigned int densityQuadratureId,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &                                                  rhoQuadValues,
    const std::map<dealii::CellId, std::vector<double>> *bQuadValues)
  {
    basisOperationsPtr->reinit(0, 0, densityQuadratureId, false);
    const unsigned int nQuadsPerCellDensity =
      basisOperationsPtr->nQuadsPerCell();
    auto matrixFreeDataObject = basisOperationsPtr->matrixFreeData();

    std::vector<std::function<dealii::VectorizedArray<double>(
      dealii::VectorizedArray<double> &,
      dealii::Point<3, dealii::VectorizedArray<double>>)>>
      momentsAtQuadPoints;
    momentsAtQuadPoints.push_back(
      [](dealii::VectorizedArray<double> &                 i,
         dealii::Point<3, dealii::VectorizedArray<double>> q) { return i; });

    momentsAtQuadPoints.push_back(
      [](dealii::VectorizedArray<double> &                 i,
         dealii::Point<3, dealii::VectorizedArray<double>> q) {
        return i * q[0];
      });

    momentsAtQuadPoints.push_back(
      [](dealii::VectorizedArray<double> &                 i,
         dealii::Point<3, dealii::VectorizedArray<double>> q) {
        return i * q[1];
      });

    momentsAtQuadPoints.push_back(
      [](dealii::VectorizedArray<double> &                 i,
         dealii::Point<3, dealii::VectorizedArray<double>> q) {
        return i * q[2];
      });

    momentsAtQuadPoints.push_back(
      [](dealii::VectorizedArray<double> &                 i,
         dealii::Point<3, dealii::VectorizedArray<double>> q) {
        return i * (3.0 * q[0] * q[0] - q.norm_square());
      });

    momentsAtQuadPoints.push_back(
      [](dealii::VectorizedArray<double> &                 i,
         dealii::Point<3, dealii::VectorizedArray<double>> q) {
        return 3.0 * i * q[0] * q[1];
      });

    momentsAtQuadPoints.push_back(
      [](dealii::VectorizedArray<double> &                 i,
         dealii::Point<3, dealii::VectorizedArray<double>> q) {
        return 3.0 * i * q[0] * q[2];
      });

    momentsAtQuadPoints.push_back(
      [](dealii::VectorizedArray<double> &                 i,
         dealii::Point<3, dealii::VectorizedArray<double>> q) {
        return 3.0 * i * q[1] * q[0];
      });

    momentsAtQuadPoints.push_back(
      [](dealii::VectorizedArray<double> &                 i,
         dealii::Point<3, dealii::VectorizedArray<double>> q) {
        return i * (3.0 * q[1] * q[1] - q.norm_square());
      });

    momentsAtQuadPoints.push_back(
      [](dealii::VectorizedArray<double> &                 i,
         dealii::Point<3, dealii::VectorizedArray<double>> q) {
        return 3.0 * i * q[1] * q[2];
      });

    momentsAtQuadPoints.push_back(
      [](dealii::VectorizedArray<double> &                 i,
         dealii::Point<3, dealii::VectorizedArray<double>> q) {
        return 3.0 * i * q[2] * q[0];
      });

    momentsAtQuadPoints.push_back(
      [](dealii::VectorizedArray<double> &                 i,
         dealii::Point<3, dealii::VectorizedArray<double>> q) {
        return 3.0 * i * q[2] * q[1];
      });

    momentsAtQuadPoints.push_back(
      [](dealii::VectorizedArray<double> &                 i,
         dealii::Point<3, dealii::VectorizedArray<double>> q) {
        return i * (3.0 * q[2] * q[2] - q.norm_square());
      });
    if (!d_smearedChargeMomentsComputed)
      {
        dealii::FEEvaluation<3,
                             C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
                             C_num1DQuadSmearedCharge() *
                               C_numCopies1DQuadSmearedCharge(),
                             1,
                             double>
          FEEvalb(matrixFreeDataObject,
                  d_densityDofHandlerIndexElectro,
                  d_smearedChargeQuadratureIdElectro);
        d_smearedChargeMoments.clear();
        d_smearedChargeMoments.resize(13, 0.0);
        for (unsigned int iMacroCell = 0;
             iMacroCell < matrixFreeDataObject.n_cell_batches();
             ++iMacroCell)
          {
            FEEvalb.reinit(iMacroCell);
            dealii::AlignedVector<dealii::VectorizedArray<double>> bVec(
              FEEvalb.n_q_points, 0.0);
            for (unsigned int iSubCell = 0;
                 iSubCell <
                 matrixFreeDataObject.n_active_entries_per_cell_batch(
                   iMacroCell);
                 ++iSubCell)
              {
                dealii::CellId subCellId =
                  matrixFreeDataObject
                    .get_cell_iterator(iMacroCell,
                                       iSubCell,
                                       d_densityDofHandlerIndexElectro)
                    ->id();
                const std::vector<double> &tempbVec =
                  bQuadValues->find(subCellId)->second;
                if (tempbVec.size() != 0)
                  for (unsigned int iQuad = 0; iQuad < FEEvalb.n_q_points;
                       ++iQuad)
                    {
                      bVec[iQuad][iSubCell] = tempbVec[iQuad];
                    }
              }
            for (unsigned int iMomentComponent = 0; iMomentComponent < 13;
                 ++iMomentComponent)
              {
                for (unsigned int iQuad = 0; iQuad < FEEvalb.n_q_points;
                     ++iQuad)
                  {
                    FEEvalb.submit_value(momentsAtQuadPoints[iMomentComponent](
                                           bVec[iQuad],
                                           FEEvalb.quadrature_point(iQuad)),
                                         iQuad);
                  }
                auto bMacroCellIntegral = FEEvalb.integrate_value();
                for (unsigned int iSubCell = 0;
                     iSubCell <
                     matrixFreeDataObject.n_active_entries_per_cell_batch(
                       iMacroCell);
                     ++iSubCell)
                  {
                    d_smearedChargeMoments[iMomentComponent] +=
                      bMacroCellIntegral[iSubCell];
                  }
              }
          }
        dealii::Utilities::MPI::sum(d_smearedChargeMoments,
                                    mpi_communicator,
                                    d_smearedChargeMoments);
        d_smearedChargeMomentsComputed = true;
      }
    std::vector<double> moments(13, 0.0);
    dealii::FEEvaluation<
      3,
      C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      1,
      double>
      FEEvalRho(matrixFreeDataObject,
                d_densityDofHandlerIndexElectro,
                d_densityQuadratureIdElectro);
    for (unsigned int iMacroCell = 0;
         iMacroCell < matrixFreeDataObject.n_cell_batches();
         ++iMacroCell)
      {
        FEEvalRho.reinit(iMacroCell);
        dealii::AlignedVector<dealii::VectorizedArray<double>> rhoVec(
          FEEvalRho.n_q_points, 0.0);
        for (unsigned int iSubCell = 0;
             iSubCell <
             matrixFreeDataObject.n_active_entries_per_cell_batch(iMacroCell);
             ++iSubCell)
          {
            dealii::CellId subCellId =
              matrixFreeDataObject
                .get_cell_iterator(iMacroCell,
                                   iSubCell,
                                   d_densityDofHandlerIndexElectro)
                ->id();
            const unsigned int cellIndex =
              basisOperationsPtr->cellIndex(subCellId);
            const double *tempVec =
              rhoQuadValues.data() + cellIndex * FEEvalRho.n_q_points;
            for (unsigned int iQuad = 0; iQuad < FEEvalRho.n_q_points; ++iQuad)
              {
                rhoVec[iQuad][iSubCell] = tempVec[iQuad];
              }
          }
        for (unsigned int iMomentComponent = 0; iMomentComponent < 13;
             ++iMomentComponent)
          {
            for (unsigned int iQuad = 0; iQuad < FEEvalRho.n_q_points; ++iQuad)
              {
                FEEvalRho.submit_value((momentsAtQuadPoints[iMomentComponent])(
                                         rhoVec[iQuad],
                                         FEEvalRho.quadrature_point(iQuad)),
                                       iQuad);
              }
            auto rhoMacroCellIntegral = FEEvalRho.integrate_value();
            for (unsigned int iSubCell = 0;
                 iSubCell <
                 matrixFreeDataObject.n_active_entries_per_cell_batch(
                   iMacroCell);
                 ++iSubCell)
              {
                moments[iMomentComponent] += rhoMacroCellIntegral[iSubCell];
              }
          }
      }
    dealii::Utilities::MPI::sum(moments, mpi_communicator, moments);
    for (unsigned int iMomentComponent = 0; iMomentComponent < 13;
         ++iMomentComponent)
      {
        moments[iMomentComponent] += d_smearedChargeMoments[iMomentComponent];
      }
    if (d_dftParamsPtr->verbosity >= 2)
      {
        pcout << "Monopole Moment        : " << moments[0] << std::endl;
        pcout << "Dipole Moment          : " << moments[1] << " " << moments[2]
              << " " << moments[3] << std::endl;
        pcout << "Quadrupole Moment      : " << std::endl
              << moments[4] << " " << moments[5] << " " << moments[6]
              << std::endl
              << moments[7] << " " << moments[8] << " " << moments[9]
              << std::endl
              << moments[10] << " " << moments[11] << " " << moments[12]
              << std::endl;
      }
    d_monopole = moments[0];
    d_dipole.clear();
    d_dipole.resize(3);
    d_quadrupole.clear();
    d_quadrupole.resize(9);
    std::copy(moments.begin() + 1, moments.begin() + 4, d_dipole.begin());
    std::copy(moments.begin() + 4, moments.end(), d_quadrupole.begin());
  }

#include "dft.inst.cc"

} // namespace dftfe
