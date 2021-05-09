// ---------------------------------------------------------------------
//
// Copyright (c) 2019-2020 The Regents of the University of Michigan and DFT-FE
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

#include <constants.h>
#include <dftParameters.h>
#include <kerkerSolverProblem.h>

namespace dftfe
{
  //
  // constructor
  //
  template <unsigned int FEOrderElectro>
  kerkerSolverProblem<FEOrderElectro>::kerkerSolverProblem(
    const MPI_Comm &mpi_comm)
    : mpi_communicator(mpi_comm)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm))
    , this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_comm))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
  {}


  template <unsigned int FEOrderElectro>
  void kerkerSolverProblem<FEOrderElectro>::init(
    dealii::MatrixFree<3, double> &    matrixFreeDataPRefined,
    dealii::AffineConstraints<double> &constraintMatrixPRefined,
    distributedCPUVec<double> &        x,
    double                             kerkerMixingParameter,
    const unsigned int                 matrixFreeVectorComponent,
    const unsigned int                 matrixFreeQuadratureComponent)
  {
    d_matrixFreeDataPRefinedPtr     = &matrixFreeDataPRefined;
    d_constraintMatrixPRefinedPtr   = &constraintMatrixPRefined;
    d_gamma                         = kerkerMixingParameter;
    d_matrixFreeVectorComponent     = matrixFreeVectorComponent;
    d_matrixFreeQuadratureComponent = matrixFreeQuadratureComponent;

    matrixFreeDataPRefined.initialize_dof_vector(x,
                                                 d_matrixFreeVectorComponent);
    computeDiagonalA();
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::reinit(
    distributedCPUVec<double> &                          x,
    const std::map<dealii::CellId, std::vector<double>> &quadPointValues)
  {
    d_xPtr                      = &x;
    d_quadGradResidualValuesPtr = &quadPointValues;
  }

  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::distributeX()
  {
    d_constraintMatrixPRefinedPtr->distribute(*d_xPtr);
  }

  template <unsigned int FEOrderElectro>
  distributedCPUVec<double> &
  kerkerSolverProblem<FEOrderElectro>::getX()
  {
    return *d_xPtr;
  }

  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::computeRhs(
    distributedCPUVec<double> &rhs)
  {
    rhs.reinit(*d_xPtr);

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    dealii::FEEvaluation<3, FEOrderElectro, C_num1DQuad<FEOrderElectro>()>
      fe_eval(*d_matrixFreeDataPRefinedPtr,
              d_matrixFreeVectorComponent,
              d_matrixFreeQuadratureComponent);

    Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor;
    for (unsigned int idim = 0; idim < 3; idim++)
      zeroTensor[idim] = make_vectorized_array(0.0);


    dealii::AlignedVector<dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
      residualGradQuads(fe_eval.n_q_points, zeroTensor);
    for (unsigned int macrocell = 0;
         macrocell < d_matrixFreeDataPRefinedPtr->n_macro_cells();
         ++macrocell)
      {
        std::fill(residualGradQuads.begin(),
                  residualGradQuads.end(),
                  zeroTensor);
        const unsigned int numSubCells =
          d_matrixFreeDataPRefinedPtr->n_components_filled(macrocell);
        for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = d_matrixFreeDataPRefinedPtr->get_cell_iterator(
              macrocell, iSubCell, d_matrixFreeVectorComponent);
            dealii::CellId             subCellId = subCellPtr->id();
            const std::vector<double> &tempVec =
              d_quadGradResidualValuesPtr->find(subCellId)->second;

            for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
              {
                residualGradQuads[q][0][iSubCell] = tempVec[3 * q + 0];
                residualGradQuads[q][1][iSubCell] = tempVec[3 * q + 1];
                residualGradQuads[q][2][iSubCell] = tempVec[3 * q + 2];
              }
          }

        fe_eval.reinit(macrocell);
        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
          fe_eval.submit_gradient(residualGradQuads[q], q);

        fe_eval.integrate(false, true);

        fe_eval.distribute_local_to_global(rhs);
      }

    // MPI operation to sync data
    rhs.compress(dealii::VectorOperation::add);

    // FIXME: check if this is really required
    d_constraintMatrixPRefinedPtr->set_zero(rhs);
  }

  // Matrix-Free Jacobi preconditioner application
  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::precondition_Jacobi(
    distributedCPUVec<double> &      dst,
    const distributedCPUVec<double> &src,
    const double                     omega) const
  {
    dst = src;
    dst.scale(d_diagonalA);
  }

  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::computeDiagonalA()
  {
    const dealii::DoFHandler<3> &dofHandler =
      d_matrixFreeDataPRefinedPtr->get_dof_handler(d_matrixFreeVectorComponent);

    d_matrixFreeDataPRefinedPtr->initialize_dof_vector(
      d_diagonalA, d_matrixFreeVectorComponent);
    d_diagonalA = 0.0;

    dealii::QGauss<3>      quadrature(C_num1DQuad<FEOrderElectro>());
    dealii::FEValues<3>    fe_values(dofHandler.get_fe(),
                                  quadrature,
                                  dealii::update_values |
                                    dealii::update_gradients |
                                    dealii::update_JxW_values);
    const unsigned int     dofs_per_cell   = dofHandler.get_fe().dofs_per_cell;
    const unsigned int     num_quad_points = quadrature.size();
    dealii::Vector<double> elementalDiagonalA(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);


    // parallel loop over all elements
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);

          cell->get_dof_indices(local_dof_indices);

          elementalDiagonalA = 0.0;
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
              elementalDiagonalA(i) +=
                (fe_values.shape_grad(i, q_point) *
                   fe_values.shape_grad(i, q_point) +
                 4 * M_PI * d_gamma * fe_values.shape_value(i, q_point) *
                   fe_values.shape_value(i, q_point)) *
                fe_values.JxW(q_point);

          d_constraintMatrixPRefinedPtr->distribute_local_to_global(
            elementalDiagonalA, local_dof_indices, d_diagonalA);
        }

    // MPI operation to sync data
    d_diagonalA.compress(dealii::VectorOperation::add);

    for (dealii::types::global_dof_index i = 0; i < d_diagonalA.size(); ++i)
      if (d_diagonalA.in_local_range(i))
        if (!d_constraintMatrixPRefinedPtr->is_constrained(i))
          d_diagonalA(i) = 1.0 / d_diagonalA(i);

    d_diagonalA.compress(dealii::VectorOperation::insert);
  }

  // Ax
  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::AX(
    const dealii::MatrixFree<3, double> &        matrixFreeData,
    distributedCPUVec<double> &                  dst,
    const distributedCPUVec<double> &            src,
    const std::pair<unsigned int, unsigned int> &cell_range) const
  {
    dealii::FEEvaluation<3, FEOrderElectro, C_num1DQuad<FEOrderElectro>()>
      fe_eval(matrixFreeData,
              d_matrixFreeVectorComponent,
              d_matrixFreeQuadratureComponent);
    // double gamma = dftParameters::kerkerParameter;

    dealii::VectorizedArray<double> kerkerConst =
      dealii::make_vectorized_array(4 * M_PI * d_gamma);


    for (unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        fe_eval.reinit(cell);
        // fe_eval.gather_evaluate(src,dealii::EvaluationFlags::values|dealii::EvaluationFlags::gradients);
        fe_eval.read_dof_values(src);
        fe_eval.evaluate(true, true, false);
        for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
          {
            fe_eval.submit_gradient(fe_eval.get_gradient(q), q);
            fe_eval.submit_value(fe_eval.get_value(q) * kerkerConst, q);
          }
        // fe_eval.integrate_scatter(dealii::EvaluationFlags::values|dealii::EvaluationFlags::gradients,dst);
        fe_eval.integrate(true, true);
        fe_eval.distribute_local_to_global(dst);
      }
  }


  template <unsigned int FEOrderElectro>
  void
  kerkerSolverProblem<FEOrderElectro>::vmult(distributedCPUVec<double> &Ax,
                                             distributedCPUVec<double> &x)
  {
    Ax = 0.0;
    x.update_ghost_values();
    AX(*d_matrixFreeDataPRefinedPtr,
       Ax,
       x,
       std::make_pair(0, d_matrixFreeDataPRefinedPtr->n_macro_cells()));
    Ax.compress(dealii::VectorOperation::add);
    // d_matrixFreeDataPRefinedPtr->cell_loop(
    //  &kerkerSolverProblem<FEOrderElectro>::AX, this, Ax, x);
  }


  template class kerkerSolverProblem<1>;
  template class kerkerSolverProblem<2>;
  template class kerkerSolverProblem<3>;
  template class kerkerSolverProblem<4>;
  template class kerkerSolverProblem<5>;
  template class kerkerSolverProblem<6>;
  template class kerkerSolverProblem<7>;
  template class kerkerSolverProblem<8>;
  template class kerkerSolverProblem<9>;
  template class kerkerSolverProblem<10>;
  template class kerkerSolverProblem<11>;
  template class kerkerSolverProblem<12>;
  template class kerkerSolverProblem<13>;
  template class kerkerSolverProblem<14>;
  template class kerkerSolverProblem<15>;
  template class kerkerSolverProblem<16>;
} // namespace dftfe
