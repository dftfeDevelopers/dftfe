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
// @author Phani Motamarri, Sambit Das
//
#include <dft.h>

namespace dftfe
{
  //
  // compute field l2 norm
  //
  template <dftfe::utils::MemorySpace memorySpace>
  double
  dftClass<memorySpace>::fieldGradl2Norm(
    const dealii::MatrixFree<3, double> &matrixFreeDataObject,
    const distributedCPUVec<double>     &nodalField)

  {
    FEEvaluationWrapperClass<1> fe_evalField(matrixFreeDataObject, 0, 0);
    const dftfe::uInt           numQuadPoints = fe_evalField.n_q_points;
    nodalField.update_ghost_values();

    // AssertThrow(nodalField.partitioners_are_globally_compatible(*matrixFreeDataObject.get_vector_partitioner(0)),
    //        dealii::ExcMessage("DFT-FE Error: mismatch in
    //        partitioner/dofHandler."));

    AssertThrow(
      matrixFreeDataObject.get_quadrature(0).size() == numQuadPoints,
      dealii::ExcMessage(
        "DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

    dealii::VectorizedArray<double> valueVectorized =
      dealii::make_vectorized_array(0.0);
    double value = 0.0;
    for (dftfe::uInt cell = 0; cell < matrixFreeDataObject.n_cell_batches();
         ++cell)
      {
        fe_evalField.reinit(cell);
        fe_evalField.read_dof_values(nodalField);
        fe_evalField.evaluate(dealii::EvaluationFlags::gradients);
        for (dftfe::uInt q_point = 0; q_point < numQuadPoints; ++q_point)
          {
            dealii::VectorizedArray<double> temp =
              scalar_product(fe_evalField.get_gradient(q_point),
                             fe_evalField.get_gradient(q_point));
            fe_evalField.submit_value(temp, q_point);
          }

        valueVectorized += fe_evalField.integrate_value();
        for (dftfe::uInt iSubCell = 0;
             iSubCell <
             matrixFreeDataObject.n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          value += valueVectorized[iSubCell];
      }

    return dealii::Utilities::MPI::sum(value, mpi_communicator);
  }

  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::l2ProjectionQuadToNodal(
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
    basisOperationsPtr->reinit(0, 0, quadratureId, false);
    const dftfe::uInt nQuadsPerCell = basisOperationsPtr->nQuadsPerCell();
    std::function<dealii::VectorizedArray<double>(const dftfe::uInt cell,
                                                  const dftfe::uInt q)>
      funcRho = [&](const dftfe::uInt cell, const dftfe::uInt q) {
        dealii::VectorizedArray<double> valuesAtQuadPoint =
          dealii::make_vectorized_array(0.0);
        for (dftfe::uInt iSubCell = 0;
             iSubCell < basisOperationsPtr->matrixFreeData()
                          .n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          {
            dealii::DoFHandler<3>::active_cell_iterator subCellPtr =
              basisOperationsPtr->matrixFreeData().get_cell_iterator(
                cell, iSubCell, dofHandlerId);
            dealii::CellId subCellId = subCellPtr->id();
            valuesAtQuadPoint[iSubCell] =
              quadratureValueData[basisOperationsPtr->cellIndex(subCellId) *
                                    nQuadsPerCell +
                                  q];
          }
        return valuesAtQuadPoint;
      };
    auto matrixFreePtr = std::shared_ptr<const dealii::MatrixFree<3, double>>(
      &(basisOperationsPtr->matrixFreeData()),
      [](const dealii::MatrixFree<3, double> *) noexcept {});
    dealii::VectorTools::project<3, distributedCPUVec<double>>(matrixFreePtr,
                                                               constraintMatrix,
                                                               std::cbrt(
                                                                 nQuadsPerCell),
                                                               funcRho,
                                                               nodalField);
    constraintMatrix.set_zero(nodalField);
    constraintMatrix.distribute(nodalField);
    nodalField.update_ghost_values();
  }
  //
  // compute mass Vector
  //
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::computeRhoNodalMassVector(
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &massVec)
  {
    const dftfe::uInt nLocalDoFs =
      d_matrixFreeDataPRefined
        .get_vector_partitioner(d_densityDofHandlerIndexElectro)
        ->locally_owned_size();
    massVec.clear();
    massVec.resize(nLocalDoFs, 0.0);

    distributedCPUVec<double> distributedMassVec;
    d_matrixFreeDataPRefined.initialize_dof_vector(
      distributedMassVec, d_densityDofHandlerIndexElectro);

    dealii::QGaussLobatto<3> quadrature(
      d_dftParamsPtr->finiteElementPolynomialOrderRhoNodal + 1);
    dealii::FEValues<3> fe_values(d_dofHandlerRhoNodal.get_fe(),
                                  quadrature,
                                  dealii::update_values |
                                    dealii::update_JxW_values);
    const dftfe::uInt   dofs_per_cell =
      (d_dofHandlerRhoNodal.get_fe()).dofs_per_cell;
    const dftfe::uInt      num_quad_points = quadrature.size();
    dealii::Vector<double> massVectorLocal(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);


    //
    // parallel loop over all elements
    //
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = d_dofHandlerRhoNodal.begin_active(),
      endc = d_dofHandlerRhoNodal.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          // compute values for the current element
          fe_values.reinit(cell);
          massVectorLocal = 0.0;
          for (dftfe::uInt i = 0; i < dofs_per_cell; ++i)
            for (dftfe::uInt q_point = 0; q_point < num_quad_points; ++q_point)
              massVectorLocal(i) += fe_values.shape_value(i, q_point) *
                                    fe_values.shape_value(i, q_point) *
                                    fe_values.JxW(q_point);

          cell->get_dof_indices(local_dof_indices);
          d_constraintsRhoNodal.distribute_local_to_global(massVectorLocal,
                                                           local_dof_indices,
                                                           distributedMassVec);
        }

    distributedMassVec.compress(dealii::VectorOperation::add);
    d_constraintsRhoNodal.set_zero(distributedMassVec);
    for (dftfe::uInt iDoF = 0; iDoF < nLocalDoFs; ++iDoF)
      massVec[iDoF] = distributedMassVec.local_element(iDoF);
  }


#include "dft.inst.cc"

} // namespace dftfe
