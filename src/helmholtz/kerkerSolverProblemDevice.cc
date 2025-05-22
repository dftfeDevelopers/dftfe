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
// @author Gourab Panigrahi
//

#include <constants.h>
#include <kerkerSolverProblemDevice.h>
#include <MemoryTransfer.h>
#include "matrixFreeDeviceKernels.h"
#include <feevaluationWrapper.h>
namespace dftfe
{
  //
  // constructor
  //
  template <dftfe::uInt FEOrderElectro>
  kerkerSolverProblemDevice<FEOrderElectro>::kerkerSolverProblemDevice(
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
  kerkerSolverProblemDevice<FEOrderElectro>::init(
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::DEVICE>>
                                      &basisOperationsPtr,
    dealii::AffineConstraints<double> &constraintMatrixPRefined,
    distributedCPUVec<double>         &x,
    double                             kerkerMixingParameter,
    const dftfe::uInt                  matrixFreeVectorComponent,
    const dftfe::uInt                  matrixFreeQuadratureComponent)
  {
    d_basisOperationsPtr            = basisOperationsPtr;
    d_matrixFreeDataPRefinedPtr     = &(basisOperationsPtr->matrixFreeData());
    d_constraintMatrixPRefinedPtr   = &constraintMatrixPRefined;
    d_gamma                         = kerkerMixingParameter;
    d_matrixFreeVectorComponent     = matrixFreeVectorComponent;
    d_matrixFreeQuadratureComponent = matrixFreeQuadratureComponent;
    d_nLocalCells = d_matrixFreeDataPRefinedPtr->n_cell_batches();

    d_matrixFreeDataPRefinedPtr->initialize_dof_vector(
      x, d_matrixFreeVectorComponent);
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      x.get_partitioner(), 1, d_xDevice);


    d_xPtr      = &x;
    d_xLocalDof = d_xDevice.locallyOwnedSize() * d_xDevice.numVectors();
    d_xLen      = d_xDevice.localSize() * d_xDevice.numVectors();

    computeDiagonalA();

    // Setup MatrixFree Mesh
    setupMatrixFree();

    // Setup MatrixFree Constraints
    setupConstraints();
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::reinit(
    distributedCPUVec<double> &x,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &quadPointValues)
  {
    d_xPtr                  = &x;
    d_residualQuadValuesPtr = &quadPointValues;

    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::DEVICE,
      dftfe::utils::MemorySpace::HOST>::copy(d_xLocalDof,
                                             d_xDevice.begin(),
                                             d_xPtr->begin());
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::setupConstraints()
  {
    d_constraintsTotalPotentialInfo.initialize(
      d_matrixFreeDataPRefinedPtr->get_vector_partitioner(
        d_matrixFreeVectorComponent),
      *d_constraintMatrixPRefinedPtr);
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::distributeX()
  {
    d_constraintsTotalPotentialInfo.distribute(d_xDevice);
  }


  template <dftfe::uInt FEOrderElectro>
  distributedDeviceVec<double> &
  kerkerSolverProblemDevice<FEOrderElectro>::getX()
  {
    return d_xDevice;
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::copyXfromDeviceToHost()
  {
    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::HOST,
      dftfe::utils::MemorySpace::DEVICE>::copy(d_xLen,
                                               d_xPtr->begin(),
                                               d_xDevice.begin());
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::setX()
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::computeRhs(
    distributedCPUVec<double> &rhs)
  {
    rhs.reinit(*d_xPtr);

    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    FEEvaluationWrapperClass<1> fe_eval(FEOrderElectro,
                                        C_num1DQuad(FEOrderElectro),
                                        *d_matrixFreeDataPRefinedPtr,
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
            dealii::CellId    subCellId = subCellPtr->id();
            const dftfe::uInt cellIndex =
              d_basisOperationsPtr->cellIndex(subCellId);
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

    // MPI operation to sync data
    rhs.compress(dealii::VectorOperation::add);

    // FIXME: check if this is really required
    d_constraintMatrixPRefinedPtr->set_zero(rhs);
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::computeDiagonalA()
  {
    const dealii::DoFHandler<3> &dofHandler =
      d_matrixFreeDataPRefinedPtr->get_dof_handler(d_matrixFreeVectorComponent);

    d_matrixFreeDataPRefinedPtr->initialize_dof_vector(
      d_diagonalA, d_matrixFreeVectorComponent);
    d_diagonalA = 0.0;

    dealii::QGauss<3>      quadrature(C_num1DQuad(FEOrderElectro));
    dealii::FEValues<3>    fe_values(dofHandler.get_fe(),
                                  quadrature,
                                  dealii::update_values |
                                    dealii::update_gradients |
                                    dealii::update_JxW_values);
    const dftfe::uInt      dofs_per_cell   = dofHandler.get_fe().dofs_per_cell;
    const dftfe::uInt      num_quad_points = quadrature.size();
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
          for (dftfe::uInt i = 0; i < dofs_per_cell; ++i)
            for (dftfe::uInt q_point = 0; q_point < num_quad_points; ++q_point)
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
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_diagonalA.get_partitioner(), 1, d_diagonalAdevice);


    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::DEVICE,
      dftfe::utils::MemorySpace::HOST>::copy(d_xLocalDof,
                                             d_diagonalAdevice.begin(),
                                             d_diagonalA.begin());
  }


  template <dftfe::uInt FEOrderElectro>
  distributedDeviceVec<double> &
  kerkerSolverProblemDevice<FEOrderElectro>::getPreconditioner()
  {
    return d_diagonalAdevice;
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::setupMatrixFree()
  {
    constexpr dftfe::Int p            = FEOrderElectro + 1;
    constexpr dftfe::Int q            = p;
    constexpr dftfe::Int nDofsPerCell = p * p * p;
    constexpr dftfe::Int dim          = 3;

    auto dofInfo =
      d_matrixFreeDataPRefinedPtr->get_dof_info(d_matrixFreeVectorComponent);
    auto shapeInfo = d_matrixFreeDataPRefinedPtr->get_shape_info(
      d_matrixFreeVectorComponent, d_matrixFreeQuadratureComponent);
    auto mappingData = d_matrixFreeDataPRefinedPtr->get_mapping_info()
                         .cell_data[d_matrixFreeQuadratureComponent];
    auto shapeData = shapeInfo.get_shape_data();

    // Shape Function Values, Gradients and their Transposes
    // P(q*p), D(q*q), PT(p*q), DT(q*q)
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      shapeFunction(2 * q * (p + q));

    for (dftfe::Int i = 0; i < p; i++)
      for (dftfe::Int j = 0; j < q; j++)
        {
#if (DEAL_II_VERSION_MAJOR >= 9 && DEAL_II_VERSION_MINOR >= 6)
          double value = shapeData.shape_values[j + i * q] *
                         std::sqrt(shapeData.quadrature.weight(j));
#else
          double value = shapeData.shape_values[j + i * q][0] *
                         std::sqrt(shapeData.quadrature.weight(j));
#endif
          shapeFunction[j + i * q]               = value;
          shapeFunction[i + j * p + q * (p + q)] = value;
        }

    for (dftfe::Int i = 0; i < q; i++)
      for (dftfe::Int j = 0; j < q; j++)
        {
#if (DEAL_II_VERSION_MAJOR >= 9 && DEAL_II_VERSION_MINOR >= 6)
          double grad = shapeData.shape_gradients_collocation[j + i * q] *
                        std::sqrt(shapeData.quadrature.weight(j)) /
                        std::sqrt(shapeData.quadrature.weight(i));
#else
          double grad = shapeData.shape_gradients_collocation[j + i * q][0] *
                        std::sqrt(shapeData.quadrature.weight(j)) /
                        std::sqrt(shapeData.quadrature.weight(i));
#endif
          shapeFunction[j + i * q + q * p]           = grad;
          shapeFunction[i + j * q + (2 * p + q) * q] = grad;
        }

    // Jacobian
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      jacobianFactor(dim * dim * d_nLocalCells);

    auto cellOffsets = mappingData.data_index_offsets;

    for (dftfe::Int cellIdx = 0; cellIdx < d_nLocalCells; cellIdx++)
      for (dftfe::Int k = 0; k < dim; k++)
        for (dftfe::Int i = 0; i < dim; i++)
          for (dftfe::Int j = 0; j < dim; j++)
            jacobianFactor[j + i * dim + cellIdx * dim * dim] +=
              mappingData
                .JxW_values[cellOffsets[cellIdx / dofInfo.vectorization_length]]
                           [0] *
              mappingData
                .jacobians[0]
                          [cellOffsets[cellIdx / dofInfo.vectorization_length]]
                          [k][j][0] *
              mappingData
                .jacobians[0]
                          [cellOffsets[cellIdx / dofInfo.vectorization_length]]
                          [k][i][0];

    // Map making
    dftfe::utils::MemoryStorage<dftfe::Int, dftfe::utils::MemorySpace::HOST>
      map(nDofsPerCell * d_nLocalCells);

    for (auto cellIdx = 0; cellIdx < d_nLocalCells; ++cellIdx)
      std::transform((dofInfo.row_starts[cellIdx].second ==
                        dofInfo.row_starts[cellIdx + 1].second &&
                      dofInfo.row_starts_plain_indices[cellIdx] ==
                        dealii::numbers::invalid_unsigned_int) ?
                       dofInfo.dof_indices.data() +
                         dofInfo.row_starts[cellIdx].first :
                       dofInfo.plain_dof_indices.data() +
                         dofInfo.row_starts_plain_indices[cellIdx],
                     (dofInfo.row_starts[cellIdx].second ==
                        dofInfo.row_starts[cellIdx + 1].second &&
                      dofInfo.row_starts_plain_indices[cellIdx] ==
                        dealii::numbers::invalid_unsigned_int) ?
                       dofInfo.dof_indices.data() +
                         dofInfo.row_starts[cellIdx].first + nDofsPerCell :
                       dofInfo.plain_dof_indices.data() +
                         dofInfo.row_starts_plain_indices[cellIdx] +
                         nDofsPerCell,
                     map.data() + cellIdx * nDofsPerCell,
                     [](unsigned int &v) { return v; });

    // Construct the device vectors
    d_shapeFunction.resize(shapeFunction.size());
    d_shapeFunction.copyFrom(shapeFunction);

    d_jacobianFactor.resize(jacobianFactor.size());
    d_jacobianFactor.copyFrom(jacobianFactor);

    d_map.resize(map.size());
    d_map.copyFrom(map);

    d_shapeFunctionPtr  = d_shapeFunction.data();
    d_jacobianFactorPtr = d_jacobianFactor.data();
    d_mapPtr            = d_map.data();
    constexpr std::size_t smem =
      (4 * q * q * q + 2 * p * q + 2 * q * q + dim * dim) * sizeof(double);
    matrixFreeDeviceKernels<double, p * p, q, p, dim>::
      computeAXDeviceHelmholtzSetAttributes(smem);
  }


  template <dftfe::uInt FEOrderElectro>
  void
  kerkerSolverProblemDevice<FEOrderElectro>::computeAX(
    distributedDeviceVec<double> &Ax,
    distributedDeviceVec<double> &x)
  {
    constexpr dftfe::Int dim     = 3;
    constexpr dftfe::Int p       = FEOrderElectro + 1;
    constexpr dftfe::Int q       = p;
    constexpr dftfe::Int threads = 64;
    // constexpr dftfe::Int threads =
    //  (FEOrderElectro < 7 ? 96 : FEOrderElectro == 7 ? 64 : 256);
    const dftfe::Int      blocks         = d_nLocalCells;
    const double          coeffHelmholtz = 4 * M_PI * d_gamma;
    constexpr std::size_t smem =
      (4 * q * q * q + 2 * p * q + 2 * q * q + dim * dim) * sizeof(double);

    dftfe::utils::deviceMemset(Ax.begin(), 0, d_xLen * sizeof(double));

    x.updateGhostValues();

    d_constraintsTotalPotentialInfo.distribute(x);

    matrixFreeDeviceKernels<double, p * p, q, p, dim>::computeAXDeviceHelmholtz(
      blocks,
      threads,
      smem,
      Ax.begin(),
      x.begin(),
      d_shapeFunctionPtr,
      d_jacobianFactorPtr,
      d_mapPtr,
      coeffHelmholtz);


    d_constraintsTotalPotentialInfo.set_zero(x);

    d_constraintsTotalPotentialInfo.distribute_slave_to_master(Ax);

    Ax.accumulateAddLocallyOwned();
  }

#include "kerkerSolverProblemDevice.inst.cc"
} // namespace dftfe
