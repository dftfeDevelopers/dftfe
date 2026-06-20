#include <dftfe/constants.h>
#include <dftfe/ldosSolverProblem.h>
#include <dftfe/feevaluationWrapper.h>

namespace dftfe
{
  template <dftfe::uInt FEOrderElectro>
  ldosSolverProblem<FEOrderElectro>::ldosSolverProblem(
    const MPI_Comm &mpi_comm_parent,
    const MPI_Comm &mpi_comm_domain)
    : d_mpiCommParent(mpi_comm_parent)
    , mpi_communicator(mpi_comm_domain)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {}

  template <dftfe::uInt FEOrderElectro>
  void
  ldosSolverProblem<FEOrderElectro>::init(
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
                                      &basisOperationsPtr,
    dealii::AffineConstraints<double> &constraintMatrixPRefined,
    distributedCPUVec<double>         &x,
    const dftfe::uInt                  matrixFreeVectorComponent,
    const dftfe::uInt                  matrixFreeQuadratureComponent,
    const dftfe::uInt                  matrixFreeAxQuadratureComponent)
  {
    d_basisOperationsPtr              = basisOperationsPtr;
    d_matrixFreeDataPRefinedPtr       = &(basisOperationsPtr->matrixFreeData());
    d_constraintMatrixPRefinedPtr     = &constraintMatrixPRefined;
    d_matrixFreeVectorComponent       = matrixFreeVectorComponent;
    d_matrixFreeQuadratureComponent   = matrixFreeQuadratureComponent;
    d_matrixFreeAxQuadratureComponent = matrixFreeAxQuadratureComponent;
    d_matrixFreeDataPRefinedPtr->initialize_dof_vector(
      x, d_matrixFreeVectorComponent);
  }

  template <dftfe::uInt FEOrderElectro>
  void
  ldosSolverProblem<FEOrderElectro>::computeProjectedQuadToNodalField(
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
                              &quadValues,
    distributedCPUVec<double> &nodalField)
  {
    d_matrixFreeDataPRefinedPtr->initialize_dof_vector(
      nodalField, d_matrixFreeVectorComponent);
    nodalField = 0.0;

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
    FEEvaluationWrapperClass<1>     fe_eval(*d_matrixFreeDataPRefinedPtr,
                                        d_matrixFreeVectorComponent,
                                        d_matrixFreeQuadratureComponent);
    dealii::VectorizedArray<double> zeroVec = 0.0;
    dealii::AlignedVector<dealii::VectorizedArray<double>> quadVals(
      fe_eval.n_q_points, zeroVec);

    for (dftfe::uInt macrocell = 0;
         macrocell < d_matrixFreeDataPRefinedPtr->n_cell_batches();
         ++macrocell)
      {
        std::fill(quadVals.begin(), quadVals.end(), zeroVec);
        const dftfe::uInt numSubCells =
          d_matrixFreeDataPRefinedPtr->n_active_entries_per_cell_batch(
            macrocell);
        for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = d_matrixFreeDataPRefinedPtr->get_cell_iterator(
              macrocell, iSubCell, d_matrixFreeVectorComponent);
            const dftfe::uInt cellIndex =
              d_basisOperationsPtr->cellIndex(subCellPtr->id());
            const double *tempVec =
              quadValues.data() + fe_eval.n_q_points * cellIndex;
            for (dftfe::uInt q = 0; q < fe_eval.n_q_points; ++q)
              quadVals[q][iSubCell] = tempVec[q];
          }

        fe_eval.reinit(macrocell);
        for (dftfe::uInt q = 0; q < fe_eval.n_q_points; ++q)
          fe_eval.submit_value(quadVals[q], q);
        fe_eval.integrate(dealii::EvaluationFlags::values);
        fe_eval.distribute_local_to_global(nodalField);
      }

    nodalField.compress(dealii::VectorOperation::add);
    d_constraintMatrixPRefinedPtr->set_zero(nodalField);
  }

  template <dftfe::uInt FEOrderElectro>
  void
  ldosSolverProblem<FEOrderElectro>::reinit(
    distributedCPUVec<double> &x,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &residualQuadValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &ldosQuadValues,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          &ldosAxQuadValues,
    double totalDOS)
  {
    d_xPtr                  = &x;
    d_residualQuadValuesPtr = &residualQuadValues;
    d_ldosQuadValuesPtr     = &ldosQuadValues;
    d_ldosAxQuadValuesPtr   = &ldosAxQuadValues;
    d_totalDOS              = totalDOS;

    computeProjectedQuadToNodalField(ldosQuadValues, d_dlocMassVector);
    computeDiagonalA();
  }

  template <dftfe::uInt FEOrderElectro>
  void
  ldosSolverProblem<FEOrderElectro>::distributeX()
  {
    d_constraintMatrixPRefinedPtr->distribute(*d_xPtr);
  }

  template <dftfe::uInt FEOrderElectro>
  distributedCPUVec<double> &
  ldosSolverProblem<FEOrderElectro>::getX()
  {
    return *d_xPtr;
  }

  template <dftfe::uInt FEOrderElectro>
  void
  ldosSolverProblem<FEOrderElectro>::computeRhs(distributedCPUVec<double> &rhs)
  {
    rhs.reinit(*d_xPtr);

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
    FEEvaluationWrapperClass<1>     fe_eval(*d_matrixFreeDataPRefinedPtr,
                                        d_matrixFreeVectorComponent,
                                        d_matrixFreeQuadratureComponent);
    dealii::VectorizedArray<double> zeroVec = 0.0;
    dealii::AlignedVector<dealii::VectorizedArray<double>> residualQuads(
      fe_eval.n_q_points, zeroVec);

    for (dftfe::uInt macrocell = 0;
         macrocell < d_matrixFreeDataPRefinedPtr->n_cell_batches();
         ++macrocell)
      {
        std::fill(residualQuads.begin(), residualQuads.end(), zeroVec);
        const dftfe::uInt numSubCells =
          d_matrixFreeDataPRefinedPtr->n_active_entries_per_cell_batch(
            macrocell);
        for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            subCellPtr = d_matrixFreeDataPRefinedPtr->get_cell_iterator(
              macrocell, iSubCell, d_matrixFreeVectorComponent);
            const dftfe::uInt cellIndex =
              d_basisOperationsPtr->cellIndex(subCellPtr->id());
            const double *tempVec =
              d_residualQuadValuesPtr->data() + fe_eval.n_q_points * cellIndex;
            for (dftfe::uInt q = 0; q < fe_eval.n_q_points; ++q)
              residualQuads[q][iSubCell] = -tempVec[q];
          }

        fe_eval.reinit(macrocell);
        for (dftfe::uInt q = 0; q < fe_eval.n_q_points; ++q)
          fe_eval.submit_value(residualQuads[q], q);
        fe_eval.integrate(dealii::EvaluationFlags::values);
        fe_eval.distribute_local_to_global(rhs);
      }

    rhs.compress(dealii::VectorOperation::add);
    d_constraintMatrixPRefinedPtr->set_zero(rhs);
  }

  template <dftfe::uInt FEOrderElectro>
  void
  ldosSolverProblem<FEOrderElectro>::precondition_Jacobi(
    distributedCPUVec<double>       &dst,
    const distributedCPUVec<double> &src,
    const double                     omega) const
  {
    dst = src;
    dst.scale(d_diagonalA);
  }

  template <dftfe::uInt FEOrderElectro>
  void
  ldosSolverProblem<FEOrderElectro>::computeDiagonalA()
  {
    const dealii::DoFHandler<3> &dofHandler =
      d_matrixFreeDataPRefinedPtr->get_dof_handler(d_matrixFreeVectorComponent);

    d_matrixFreeDataPRefinedPtr->initialize_dof_vector(
      d_diagonalA, d_matrixFreeVectorComponent);
    d_diagonalA = 0.0;

    const auto &quad_formula = d_matrixFreeDataPRefinedPtr->get_quadrature(
      d_matrixFreeAxQuadratureComponent);
    dealii::FEValues<3> fe_values(dofHandler.get_fe(),
                                  quad_formula,
                                  dealii::update_values |
                                    dealii::update_gradients |
                                    dealii::update_JxW_values);

    const dftfe::uInt      dofs_per_cell   = dofHandler.get_fe().dofs_per_cell;
    const dftfe::uInt      num_quad_points = quad_formula.size();
    dealii::Vector<double> elementalDiagonalA(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);

    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          fe_values.reinit(cell);
          cell->get_dof_indices(local_dof_indices);
          const dftfe::uInt cellIndex =
            d_basisOperationsPtr->cellIndex(cell->id());
          const double *ldosCellQuads =
            d_ldosAxQuadValuesPtr->data() + num_quad_points * cellIndex;

          elementalDiagonalA = 0.0;
          for (dftfe::uInt i = 0; i < dofs_per_cell; ++i)
            for (dftfe::uInt q_point = 0; q_point < num_quad_points; ++q_point)
              elementalDiagonalA(i) += (fe_values.shape_grad(i, q_point) *
                                          fe_values.shape_grad(i, q_point) +
                                        4.0 * M_PI * ldosCellQuads[q_point] *
                                          fe_values.shape_value(i, q_point) *
                                          fe_values.shape_value(i, q_point)) *
                                       fe_values.JxW(q_point);

          d_constraintMatrixPRefinedPtr->distribute_local_to_global(
            elementalDiagonalA, local_dof_indices, d_diagonalA);
        }

    d_diagonalA.compress(dealii::VectorOperation::add);

    for (dealii::types::global_dof_index i = 0; i < d_diagonalA.size(); ++i)
      if (d_diagonalA.in_local_range(i))
        if (!d_constraintMatrixPRefinedPtr->is_constrained(i))
          {
            d_diagonalA(i) -= (4.0 * M_PI / d_totalDOS) * d_dlocMassVector(i) *
                              d_dlocMassVector(i);
            d_diagonalA(i) = 1.0 / d_diagonalA(i);
          }

    d_diagonalA.compress(dealii::VectorOperation::insert);
  }

  template <dftfe::uInt FEOrderElectro>
  void
  ldosSolverProblem<FEOrderElectro>::AX(
    const dealii::MatrixFree<3, double>       &matrixFreeData,
    distributedCPUVec<double>                 &dst,
    const distributedCPUVec<double>           &src,
    const std::pair<dftfe::uInt, dftfe::uInt> &cell_range) const
  {
    FEEvaluationWrapperClass<1> fe_eval(matrixFreeData,
                                        d_matrixFreeVectorComponent,
                                        d_matrixFreeAxQuadratureComponent);

    dealii::VectorizedArray<double>                        zeroVec = 0.0;
    dealii::AlignedVector<dealii::VectorizedArray<double>> ldosQuads(
      fe_eval.n_q_points, zeroVec);

    for (dftfe::uInt cell = cell_range.first; cell < cell_range.second; ++cell)
      {
        std::fill(ldosQuads.begin(), ldosQuads.end(), zeroVec);
        const dftfe::uInt numSubCells =
          matrixFreeData.n_active_entries_per_cell_batch(cell);
        for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
          {
            const auto subCellPtr =
              matrixFreeData.get_cell_iterator(cell,
                                               iSubCell,
                                               d_matrixFreeVectorComponent);
            const dftfe::uInt cellIndex =
              d_basisOperationsPtr->cellIndex(subCellPtr->id());
            const double *tempVec =
              d_ldosAxQuadValuesPtr->data() + fe_eval.n_q_points * cellIndex;
            for (dftfe::uInt q = 0; q < fe_eval.n_q_points; ++q)
              ldosQuads[q][iSubCell] = tempVec[q];
          }

        fe_eval.reinit(cell);
        fe_eval.read_dof_values(src);
        fe_eval.evaluate(dealii::EvaluationFlags::values |
                         dealii::EvaluationFlags::gradients);

        for (dftfe::uInt q = 0; q < fe_eval.n_q_points; ++q)
          {
            fe_eval.submit_gradient(fe_eval.get_gradient(q), q);
            fe_eval.submit_value(dealii::make_vectorized_array(4.0 * M_PI) *
                                   ldosQuads[q] * fe_eval.get_value(q),
                                 q);
          }

        fe_eval.integrate(dealii::EvaluationFlags::values |
                          dealii::EvaluationFlags::gradients);
        fe_eval.distribute_local_to_global(dst);
      }
  }

  template <dftfe::uInt FEOrderElectro>
  void
  ldosSolverProblem<FEOrderElectro>::vmult(distributedCPUVec<double> &Ax,
                                           distributedCPUVec<double> &x)
  {
    Ax = 0.0;
    x.update_ghost_values();

    AX(*d_matrixFreeDataPRefinedPtr,
       Ax,
       x,
       std::make_pair(0, d_matrixFreeDataPRefinedPtr->n_cell_batches()));

    Ax.compress(dealii::VectorOperation::add);

    double globalInner = d_dlocMassVector * x;
    // This is the comment that needs to be undone after sanity check
    Ax.add(-(4.0 * M_PI / d_totalDOS) * globalInner, d_dlocMassVector);
  }

  template <dftfe::uInt FEOrderElectro>
  const distributedCPUVec<double> &
  ldosSolverProblem<FEOrderElectro>::getDlocMassVector() const
  {
    return d_dlocMassVector;
  }

  template <dftfe::uInt FEOrderElectro>
  double
  ldosSolverProblem<FEOrderElectro>::computeDlocIntegral(
    const distributedCPUVec<double> &phi) const
  {
    return d_dlocMassVector * phi;
  }

  template class ldosSolverProblem<1>;
  template class ldosSolverProblem<2>;
  template class ldosSolverProblem<3>;
  template class ldosSolverProblem<4>;
  template class ldosSolverProblem<5>;
  template class ldosSolverProblem<6>;
  template class ldosSolverProblem<7>;
  template class ldosSolverProblem<8>;
  template class ldosSolverProblem<9>;
  template class ldosSolverProblem<10>;
  template class ldosSolverProblem<11>;
  template class ldosSolverProblem<12>;
  template class ldosSolverProblem<13>;
  template class ldosSolverProblem<14>;
  template class ldosSolverProblem<15>;
  template class ldosSolverProblem<16>;

} // namespace dftfe
