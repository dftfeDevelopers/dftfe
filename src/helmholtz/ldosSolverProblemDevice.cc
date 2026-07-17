#include <dftfe/constants.h>
#include <dftfe/ldosSolverProblemDevice.h>
#include <dftfe/MemoryTransfer.h>
#include <dftfe/feevaluationWrapper.h>

namespace dftfe
{
  template <dftfe::uInt FEOrderElectro>
  ldosSolverProblemDevice<FEOrderElectro>::ldosSolverProblemDevice(
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
  ldosSolverProblemDevice<FEOrderElectro>::init(
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
    d_nLocalCells = d_matrixFreeDataPRefinedPtr->n_cell_batches();

    d_matrixFreeDataPRefinedPtr->initialize_dof_vector(
      x, d_matrixFreeVectorComponent);
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      x.get_partitioner(), 1, d_xDevice);

    d_xPtr      = &x;
    d_xLocalDof = d_xDevice.locallyOwnedSize() * d_xDevice.numVectors();
    d_xLen      = d_xDevice.localSize() * d_xDevice.numVectors();

    // pcout << "Entering the Device problem" << std::endl;
    // computeDiagonalA();
    setupConstraints();

    // Construct cellIndexToMacroCellSubCellIndexMap
    {
      const dftfe::uInt nCells =
        d_matrixFreeDataPRefinedPtr->n_physical_cells();
      const dftfe::uInt d_nMacroCells =
        d_matrixFreeDataPRefinedPtr->n_cell_batches();
      auto cellPtr = d_matrixFreeDataPRefinedPtr
                       ->get_dof_handler(d_matrixFreeVectorComponent)
                       .begin_active();
      auto endcPtr = d_matrixFreeDataPRefinedPtr
                       ->get_dof_handler(d_matrixFreeVectorComponent)
                       .end();

      std::map<dealii::CellId, dftfe::uInt> cellIdToCellIndexMap;
      d_cellIndexToMacroCellSubCellIndexMap.resize(nCells);

      dftfe::uInt iCell = 0;
      for (; cellPtr != endcPtr; cellPtr++)
        if (cellPtr->is_locally_owned())
          {
            cellIdToCellIndexMap[cellPtr->id()] = iCell;
            iCell++;
          }

      iCell = 0;
      for (dftfe::uInt iMacroCell = 0; iMacroCell < d_nMacroCells; iMacroCell++)
        {
          const dftfe::uInt numberSubCells =
            d_matrixFreeDataPRefinedPtr->n_active_entries_per_cell_batch(
              iMacroCell);

          for (dftfe::uInt iSubCell = 0; iSubCell < numberSubCells; iSubCell++)
            {
              cellPtr = d_matrixFreeDataPRefinedPtr->get_cell_iterator(
                iMacroCell, iSubCell, d_matrixFreeVectorComponent);

              dftfe::uInt cellIndex = cellIdToCellIndexMap[cellPtr->id()];
              d_cellIndexToMacroCellSubCellIndexMap[cellIndex] = iCell;

              iCell++;
            }
        }
    }

    // The device-side LDOS MatrixFree operator. We use the Helmholtz
    // operator with the per-(cell, quadrature-point) coefficient overload,
    // so the spatially varying LDOS coefficient 4π·n_loc(r_q) is applied
    // per quadrature point inside HelmholtzKernel.
    std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
      BLASWrapperPtrNull;

    d_matrixFreeWrapperDevice = std::make_unique<
      dftfe::MatrixFreeWrapperClass<double,
                                    dftfe::operatorList::Helmholtz,
                                    dftfe::utils::MemorySpace::DEVICE,
                                    false>>(FEOrderElectro + 1, // nDofsPerDim
                                            mpi_communicator,
                                            d_matrixFreeDataPRefinedPtr,
                                            constraintMatrixPRefined,
                                            BLASWrapperPtrNull,
                                            d_matrixFreeVectorComponent,
                                            d_matrixFreeAxQuadratureComponent,
                                            1 /*nVectors*/);

    d_matrixFreeWrapperDevice->init();

    // The coefficient 4π·n_loc(r_q) depends on the iteration's density,
    // so it is uploaded in reinit() via the pointer overload of
    // initOperatorCoeffs.
  }


  template <dftfe::uInt FEOrderElectro>
  void
  ldosSolverProblemDevice<FEOrderElectro>::reinit(
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

    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::DEVICE,
      dftfe::utils::MemorySpace::HOST>::copy(d_xLocalDof,
                                             d_xDevice.begin(),
                                             d_xPtr->begin());

    computeProjectedQuadToNodalField(ldosQuadValues, d_dlocMassVector);

    // Mirror d_dlocMassVector to device
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_dlocMassVector.get_partitioner(), 1, d_dlocMassVectorDevice);
    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::DEVICE,
      dftfe::utils::MemorySpace::HOST>::copy(d_xLocalDof,
                                             d_dlocMassVectorDevice.begin(),
                                             d_dlocMassVector.begin());

    // Iterate in MatrixFree macrocell/subcell order so that iCell (0-based)
    // matches blockIdx.x inside HelmholtzKernel.
    {
      // Use a temporary FEEval just to get n_q_points for this quadrature rule.
      FEEvaluationWrapperClass<1> fe_eval_tm(*d_matrixFreeDataPRefinedPtr,
                                             d_matrixFreeVectorComponent,
                                             d_matrixFreeAxQuadratureComponent);
      const dftfe::uInt           nQPerCell = fe_eval_tm.n_q_points;
      const dftfe::uInt           nCells =
        d_matrixFreeDataPRefinedPtr->n_physical_cells();

      const auto &quad_formula = d_matrixFreeDataPRefinedPtr->get_quadrature(
        d_matrixFreeAxQuadratureComponent);

      std::vector<double> ldosCoeffHost(nCells * nQPerCell, 0.0);

      dftfe::uInt                                 iCell = 0;
      dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
      for (dftfe::uInt macrocell = 0;
           macrocell < d_matrixFreeDataPRefinedPtr->n_cell_batches();
           ++macrocell)
        {
          const dftfe::uInt numSubCells =
            d_matrixFreeDataPRefinedPtr->n_active_entries_per_cell_batch(
              macrocell);
          for (dftfe::uInt iSubCell = 0; iSubCell < numSubCells;
               ++iSubCell, ++iCell)
            {
              subCellPtr = d_matrixFreeDataPRefinedPtr->get_cell_iterator(
                macrocell, iSubCell, d_matrixFreeVectorComponent);
              const dftfe::uInt basisIdx =
                d_basisOperationsPtr->cellIndex(subCellPtr->id());
              const double *src =
                ldosAxQuadValues.data() + nQPerCell * basisIdx;
              for (dftfe::uInt q = 0; q < nQPerCell; ++q)
                ldosCoeffHost[iCell * nQPerCell + q] = 4.0 * M_PI * src[q];
            }
        }

      // Reorder ldosCoeffHost to cell-matrix order
      std::vector<double> reorderedCoeff(nCells * nQPerCell, 0.0);
      for (dftfe::uInt cellIdx = 0; cellIdx < nCells; ++cellIdx)
        {
          const dftfe::uInt macroSubCellIdx =
            d_cellIndexToMacroCellSubCellIndexMap[cellIdx];
          for (dftfe::uInt q = 0; q < nQPerCell; ++q)
            reorderedCoeff[cellIdx * nQPerCell + q] =
              ldosCoeffHost[macroSubCellIdx * nQPerCell + q];
        }

      // Upload host -> device and notify the MatrixFree operator
      d_ldosCoeffQuadDevice.resize(nCells * nQPerCell);
      dftfe::utils::MemoryTransfer<
        dftfe::utils::MemorySpace::DEVICE,
        dftfe::utils::MemorySpace::HOST>::copy(nCells * nQPerCell,
                                               d_ldosCoeffQuadDevice.data(),
                                               reorderedCoeff.data());
      d_matrixFreeWrapperDevice->initOperatorCoeffs(
        d_ldosCoeffQuadDevice.data(), nCells * nQPerCell);
    }

    // Rebuild diagonal preconditioner
    computeDiagonalA();
  }


  template <dftfe::uInt FEOrderElectro>
  void
  ldosSolverProblemDevice<FEOrderElectro>::computeProjectedQuadToNodalField(
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
  ldosSolverProblemDevice<FEOrderElectro>::setupConstraints()
  {
    d_constraintsTotalPotentialInfo.initialize(
      d_matrixFreeDataPRefinedPtr->get_vector_partitioner(
        d_matrixFreeVectorComponent),
      *d_constraintMatrixPRefinedPtr);
  }


  template <dftfe::uInt FEOrderElectro>
  void
  ldosSolverProblemDevice<FEOrderElectro>::distributeX()
  {
    d_constraintsTotalPotentialInfo.distribute(d_xDevice);
  }


  template <dftfe::uInt FEOrderElectro>
  distributedDeviceVec<double> &
  ldosSolverProblemDevice<FEOrderElectro>::getX()
  {
    return d_xDevice;
  }


  template <dftfe::uInt FEOrderElectro>
  distributedDeviceVec<double> &
  ldosSolverProblemDevice<FEOrderElectro>::getPreconditioner()
  {
    return d_diagonalAdevice;
  }


  template <dftfe::uInt FEOrderElectro>
  void
  ldosSolverProblemDevice<FEOrderElectro>::copyXfromDeviceToHost()
  {
    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::HOST,
      dftfe::utils::MemorySpace::DEVICE>::copy(d_xLen,
                                               d_xPtr->begin(),
                                               d_xDevice.begin());
  }


  template <dftfe::uInt FEOrderElectro>
  void
  ldosSolverProblemDevice<FEOrderElectro>::setX()
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }


  template <dftfe::uInt FEOrderElectro>
  void
  ldosSolverProblemDevice<FEOrderElectro>::computeRhs(
    distributedCPUVec<double> &rhs)
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
  ldosSolverProblemDevice<FEOrderElectro>::computeDiagonalA()
  {
    const dealii::DoFHandler<3> &dofHandler =
      d_matrixFreeDataPRefinedPtr->get_dof_handler(d_matrixFreeVectorComponent);

    d_matrixFreeDataPRefinedPtr->initialize_dof_vector(
      d_diagonalA, d_matrixFreeVectorComponent);
    d_diagonalA = 0.0;

    const auto &quad_formula = d_matrixFreeDataPRefinedPtr->get_quadrature(
      d_matrixFreeAxQuadratureComponent);
    // dealii::FEValues<3> fe_values(dofHandler.get_fe(),
    //                               quad_formula,
    //                               dealii::update_gradients |
    //                                 dealii::update_JxW_values);
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
          // comment when needed
          const dftfe::uInt cellIndex =
            d_basisOperationsPtr->cellIndex(cell->id());
          const double *ldosCellQuads =
            d_ldosAxQuadValuesPtr->data() + num_quad_points * cellIndex;

          elementalDiagonalA = 0.0;
          for (dftfe::uInt i = 0; i < dofs_per_cell; ++i)
            for (dftfe::uInt q_point = 0; q_point < num_quad_points; ++q_point)
              // elementalDiagonalA(i) += (fe_values.shape_grad(i, q_point) *
              //                           fe_values.shape_grad(i, q_point)) *
              //                          fe_values.JxW(q_point);
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

    // Subtract low-rank correction and invert
    for (dealii::types::global_dof_index i = 0; i < d_diagonalA.size(); ++i)
      if (d_diagonalA.in_local_range(i))
        if (!d_constraintMatrixPRefinedPtr->is_constrained(i))
          {
            // d_diagonalA(i) -= (4.0 * M_PI / d_totalDOS) * d_dlocMassVector(i)
            // *
            //                   d_dlocMassVector(i);
            d_diagonalA(i) = 1.0 / d_diagonalA(i);
          }

    d_diagonalA.compress(dealii::VectorOperation::insert);

    // Mirror to device
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_diagonalA.get_partitioner(), 1, d_diagonalAdevice);
    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::DEVICE,
      dftfe::utils::MemorySpace::HOST>::copy(d_xLocalDof,
                                             d_diagonalAdevice.begin(),
                                             d_diagonalA.begin());
  }

  template <dftfe::uInt FEOrderElectro>
  void
  ldosSolverProblemDevice<FEOrderElectro>::computeAX(
    distributedDeviceVec<double> &Ax,
    distributedDeviceVec<double> &x)
  {
    dftfe::utils::deviceMemset(Ax.begin(), 0, d_xLen * sizeof(double));

    // Compute globalInner = d_dlocMassVectorDevice · x BEFORE x is modified by
    // constraints!
    double globalInner = 0.0;
    d_BLASWrapperPtr->xdot(d_xLocalDof,
                           d_dlocMassVectorDevice.data(),
                           1,
                           x.data(),
                           1,
                           mpi_communicator,
                           &globalInner);

    x.updateGhostValues();

    d_matrixFreeWrapperDevice->constraintsDistribute(x.data(), false);

    d_matrixFreeWrapperDevice->computeAX(Ax.data(), x.data());

    d_matrixFreeWrapperDevice->constraintsDistributeTranspose(Ax.data(),
                                                              x.data());

    Ax.accumulateAddLocallyOwned();

    //   Ax -= (4π / N_tot) * globalInner * d_dlocMassVectorDevice
    const double alpha = -(4.0 * M_PI / d_totalDOS) * globalInner;
    d_BLASWrapperPtr->xaxpy(
      d_xLocalDof, &alpha, d_dlocMassVectorDevice.data(), 1, Ax.data(), 1);
  }


#include "ldosSolverProblemDevice.inst.cc"

} // namespace dftfe
