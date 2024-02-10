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
// @author Gourab Panigrahi
//

#include <poissonSolverProblemDevice.h>
#include <DeviceAPICalls.h>
#include <DeviceKernelLauncherConstants.h>
#include <MemoryTransfer.h>

namespace dftfe
{
  //
  // constructor
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  poissonSolverProblemDevice<FEOrder, FEOrderElectro>::
    poissonSolverProblemDevice(const MPI_Comm &mpi_comm)
    : mpi_communicator(mpi_comm)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm))
    , this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_comm))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0))
  {
    d_isMeanValueConstraintComputed = false;
    d_isGradSmearedChargeRhs        = false;
    d_isStoreSmearedChargeRhs       = false;
    d_isReuseSmearedChargeRhs       = false;
    d_isFastConstraintsInitialized  = false;
    d_rhoValuesPtr                  = NULL;
    d_atomsPtr                      = NULL;
    d_smearedChargeValuesPtr        = NULL;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrder, FEOrderElectro>::clear()
  {
    d_diagonalA.reinit(0);
    d_rhsSmearedCharge.reinit(0);
    d_meanValueConstraintVec.reinit(0);
    d_cellShapeFunctionGradientIntegralFlattened.clear();
    d_isMeanValueConstraintComputed = false;
    d_isGradSmearedChargeRhs        = false;
    d_isStoreSmearedChargeRhs       = false;
    d_isReuseSmearedChargeRhs       = false;
    d_isFastConstraintsInitialized  = false;
    d_rhoValuesPtr                  = NULL;
    d_atomsPtr                      = NULL;
    d_smearedChargeValuesPtr        = NULL;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrder, FEOrderElectro>::reinit(
    const std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>
      &                                      basisOperationsPtr,
    distributedCPUVec<double> &              x,
    const dealii::AffineConstraints<double> &constraintMatrix,
    const unsigned int                       matrixFreeVectorComponent,
    const unsigned int matrixFreeQuadratureComponentRhsDensity,
    const unsigned int matrixFreeQuadratureComponentAX,
    const std::map<dealii::types::global_dof_index, double> &atoms,
    const std::map<dealii::CellId, std::vector<double>> &smearedChargeValues,
    const unsigned int smearedChargeQuadratureId,
    const dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      &rhoValues,
    const std::shared_ptr<
      dftfe::linearAlgebra::BLASWrapper<dftfe::utils::MemorySpace::DEVICE>>
                       BLASWrapperPtr,
    const bool         isComputeDiagonalA,
    const bool         isComputeMeanValueConstraint,
    const bool         smearedNuclearCharges,
    const bool         isRhoValues,
    const bool         isGradSmearedChargeRhs,
    const unsigned int smearedChargeGradientComponentId,
    const bool         storeSmearedChargeRhs,
    const bool         reuseSmearedChargeRhs,
    const bool         reinitializeFastConstraints)
  {
    int this_process;
    MPI_Comm_rank(mpi_communicator, &this_process);
    MPI_Barrier(mpi_communicator);
    double time = MPI_Wtime();

    d_basisOperationsPtr = basisOperationsPtr;
    d_matrixFreeDataPtr  = &(basisOperationsPtr->matrixFreeData());
    d_xPtr               = &x;
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_xPtr->get_partitioner(), 1, d_xDevice);

    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::DEVICE,
      dftfe::utils::MemorySpace::HOST>::copy(d_xDevice.locallyOwnedSize() *
                                               d_xDevice.numVectors(),
                                             d_xDevice.begin(),
                                             d_xPtr->begin());


    d_constraintMatrixPtr       = &constraintMatrix;
    d_matrixFreeVectorComponent = matrixFreeVectorComponent;
    d_matrixFreeQuadratureComponentRhsDensity =
      matrixFreeQuadratureComponentRhsDensity;
    d_matrixFreeQuadratureComponentAX = matrixFreeQuadratureComponentAX;
    d_rhoValuesPtr                    = isRhoValues ? &rhoValues : NULL;
    d_atomsPtr                        = smearedNuclearCharges ? NULL : &atoms;
    d_smearedChargeValuesPtr =
      smearedNuclearCharges ? &smearedChargeValues : NULL;
    d_smearedChargeQuadratureId        = smearedChargeQuadratureId;
    d_isGradSmearedChargeRhs           = isGradSmearedChargeRhs;
    d_smearedChargeGradientComponentId = smearedChargeGradientComponentId;
    d_isStoreSmearedChargeRhs          = storeSmearedChargeRhs;
    d_isReuseSmearedChargeRhs          = reuseSmearedChargeRhs;
    d_BLASWrapperPtr                   = BLASWrapperPtr;
    d_nLocalCells                      = d_matrixFreeDataPtr->n_cell_batches();
    d_xLocalDof = d_xDevice.locallyOwnedSize() * d_xDevice.numVectors();
    d_xLen      = d_xDevice.localSize() * d_xDevice.numVectors();

    AssertThrow(
      storeSmearedChargeRhs == false || reuseSmearedChargeRhs == false,
      dealii::ExcMessage(
        "DFT-FE Error: both store and reuse smeared charge rhs cannot be true at the same time."));

    if (isComputeMeanValueConstraint)
      {
        computeMeanValueConstraint();
        d_isMeanValueConstraintComputed = true;
      }

    if (isComputeDiagonalA)
      computeDiagonalA();

    if (!d_isFastConstraintsInitialized || reinitializeFastConstraints)
      {
        d_constraintsInfo.initialize(
          d_matrixFreeDataPtr->get_vector_partitioner(
            matrixFreeVectorComponent),
          constraintMatrix);

        // Setup MatrixFree Mesh
        setupMatrixFree();

        // Setup MatrixFree Constraints
        setupconstraints();

        d_isFastConstraintsInitialized = true;
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrder, FEOrderElectro>::copyXfromDeviceToHost()
  {
    dftfe::utils::MemoryTransfer<
      dftfe::utils::MemorySpace::HOST,
      dftfe::utils::MemorySpace::DEVICE>::copy(d_xLen,
                                               d_xPtr->begin(),
                                               d_xDevice.begin());
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrder, FEOrderElectro>::distributeX()
  {
    d_constraintsTotalPotentialInfo.distribute(d_xDevice, 1);

    if (d_isMeanValueConstraintComputed)
      meanValueConstraintDistribute(d_xDevice);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  distributedDeviceVec<double> &
  poissonSolverProblemDevice<FEOrder, FEOrderElectro>::getX()
  {
    return d_xDevice;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrder, FEOrderElectro>::computeRhs(
    distributedCPUVec<double> &rhs)
  {
    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;
    rhs.reinit(*d_xPtr);
    rhs = 0;

    if (d_isStoreSmearedChargeRhs)
      {
        d_rhsSmearedCharge.reinit(*d_xPtr);
        d_rhsSmearedCharge = 0;
      }

    const dealii::DoFHandler<3> &dofHandler =
      d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);

    const unsigned int     dofs_per_cell = dofHandler.get_fe().dofs_per_cell;
    dealii::Vector<double> elementalRhs(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);
    typename dealii::DoFHandler<3>::active_cell_iterator
      cell = dofHandler.begin_active(),
      endc = dofHandler.end();


    distributedCPUVec<double> tempvec;
    tempvec.reinit(rhs);
    tempvec = 0.0;
    d_constraintsInfo.distribute(tempvec);

    dealii::FEEvaluation<3, FEOrderElectro, FEOrderElectro + 1> fe_eval(
      *d_matrixFreeDataPtr,
      d_matrixFreeVectorComponent,
      d_matrixFreeQuadratureComponentAX);

    int isPerformStaticCondensation = (tempvec.linfty_norm() > 1e-10) ? 1 : 0;

    MPI_Bcast(&isPerformStaticCondensation, 1, MPI_INT, 0, mpi_communicator);

    if (isPerformStaticCondensation == 1)
      {
        dealii::VectorizedArray<double> quarter =
          dealii::make_vectorized_array(1.0 / (4.0 * M_PI));
        for (unsigned int macrocell = 0;
             macrocell < d_matrixFreeDataPtr->n_cell_batches();
             ++macrocell)
          {
            fe_eval.reinit(macrocell);
            fe_eval.read_dof_values_plain(tempvec);
            fe_eval.evaluate(false, true);
            for (unsigned int q = 0; q < fe_eval.n_q_points; ++q)
              {
                fe_eval.submit_gradient(-quarter * fe_eval.get_gradient(q), q);
              }
            fe_eval.integrate(false, true);
            fe_eval.distribute_local_to_global(rhs);
          }
      }

    // rhs contribution from electronic charge
    if (d_rhoValuesPtr)
      {
        dealii::FEEvaluation<
          3,
          FEOrderElectro,
          C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>()>
          fe_eval_density(*d_matrixFreeDataPtr,
                          d_matrixFreeVectorComponent,
                          d_matrixFreeQuadratureComponentRhsDensity);

        dealii::AlignedVector<dealii::VectorizedArray<double>> rhoQuads(
          fe_eval_density.n_q_points, dealii::make_vectorized_array(0.0));
        for (unsigned int macrocell = 0;
             macrocell < d_matrixFreeDataPtr->n_cell_batches();
             ++macrocell)
          {
            fe_eval_density.reinit(macrocell);

            std::fill(rhoQuads.begin(),
                      rhoQuads.end(),
                      dealii::make_vectorized_array(0.0));
            const unsigned int numSubCells =
              d_matrixFreeDataPtr->n_active_entries_per_cell_batch(macrocell);
            for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
              {
                subCellPtr = d_matrixFreeDataPtr->get_cell_iterator(
                  macrocell, iSubCell, d_matrixFreeVectorComponent);
                dealii::CellId subCellId = subCellPtr->id();
                unsigned int   cellIndex =
                  d_basisOperationsPtr->cellIndex(subCellId);
                const double *tempVec = d_rhoValuesPtr->data() +
                                        cellIndex * fe_eval_density.n_q_points;

                for (unsigned int q = 0; q < fe_eval_density.n_q_points; ++q)
                  rhoQuads[q][iSubCell] = tempVec[q];
              }


            for (unsigned int q = 0; q < fe_eval_density.n_q_points; ++q)
              {
                fe_eval_density.submit_value(rhoQuads[q], q);
              }
            fe_eval_density.integrate(true, false);
            fe_eval_density.distribute_local_to_global(rhs);
          }
      }

    // rhs contribution from atomic charge at fem nodes
    if (d_atomsPtr != NULL)
      for (std::map<dealii::types::global_dof_index, double>::const_iterator
             it = (*d_atomsPtr).begin();
           it != (*d_atomsPtr).end();
           ++it)
        {
          std::vector<dealii::AffineConstraints<double>::size_type>
                                 local_dof_indices_origin(1, it->first); // atomic node
          dealii::Vector<double> cell_rhs_origin(1);
          cell_rhs_origin(0) = -(it->second); // atomic charge

          d_constraintMatrixPtr->distribute_local_to_global(
            cell_rhs_origin, local_dof_indices_origin, rhs);
        }
    else if (d_smearedChargeValuesPtr != NULL && !d_isGradSmearedChargeRhs &&
             !d_isReuseSmearedChargeRhs)
      {
        // const unsigned int   num_quad_points_sc =
        // d_matrixFreeDataPtr->get_quadrature(d_smearedChargeQuadratureId).size();

        dealii::FEEvaluation<3, -1> fe_eval_sc(*d_matrixFreeDataPtr,
                                               d_matrixFreeVectorComponent,
                                               d_smearedChargeQuadratureId);

        const unsigned int numQuadPointsSmearedb = fe_eval_sc.n_q_points;

        dealii::AlignedVector<dealii::VectorizedArray<double>> smearedbQuads(
          numQuadPointsSmearedb, dealii::make_vectorized_array(0.0));
        for (unsigned int macrocell = 0;
             macrocell < d_matrixFreeDataPtr->n_cell_batches();
             ++macrocell)
          {
            std::fill(smearedbQuads.begin(),
                      smearedbQuads.end(),
                      dealii::make_vectorized_array(0.0));
            bool               isMacroCellTrivial = true;
            const unsigned int numSubCells =
              d_matrixFreeDataPtr->n_active_entries_per_cell_batch(macrocell);
            for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
              {
                subCellPtr = d_matrixFreeDataPtr->get_cell_iterator(
                  macrocell, iSubCell, d_matrixFreeVectorComponent);
                dealii::CellId             subCellId = subCellPtr->id();
                const std::vector<double> &tempVec =
                  d_smearedChargeValuesPtr->find(subCellId)->second;
                if (tempVec.size() == 0)
                  continue;

                for (unsigned int q = 0; q < numQuadPointsSmearedb; ++q)
                  smearedbQuads[q][iSubCell] = tempVec[q];

                isMacroCellTrivial = false;
              }

            if (!isMacroCellTrivial)
              {
                fe_eval_sc.reinit(macrocell);
                for (unsigned int q = 0; q < fe_eval_sc.n_q_points; ++q)
                  {
                    fe_eval_sc.submit_value(smearedbQuads[q], q);
                  }
                fe_eval_sc.integrate(true, false);

                fe_eval_sc.distribute_local_to_global(rhs);

                if (d_isStoreSmearedChargeRhs)
                  fe_eval_sc.distribute_local_to_global(d_rhsSmearedCharge);
              }
          }
      }
    else if (d_smearedChargeValuesPtr != NULL && d_isGradSmearedChargeRhs)
      {
        dealii::FEEvaluation<3, -1> fe_eval_sc2(*d_matrixFreeDataPtr,
                                                d_matrixFreeVectorComponent,
                                                d_smearedChargeQuadratureId);

        const unsigned int numQuadPointsSmearedb = fe_eval_sc2.n_q_points;

        dealii::Tensor<1, 3, dealii::VectorizedArray<double>> zeroTensor;
        for (unsigned int i = 0; i < 3; i++)
          zeroTensor[i] = dealii::make_vectorized_array(0.0);

        dealii::AlignedVector<
          dealii::Tensor<1, 3, dealii::VectorizedArray<double>>>
          smearedbQuads(numQuadPointsSmearedb, zeroTensor);
        for (unsigned int macrocell = 0;
             macrocell < d_matrixFreeDataPtr->n_cell_batches();
             ++macrocell)
          {
            std::fill(smearedbQuads.begin(),
                      smearedbQuads.end(),
                      dealii::make_vectorized_array(0.0));
            bool               isMacroCellTrivial = true;
            const unsigned int numSubCells =
              d_matrixFreeDataPtr->n_active_entries_per_cell_batch(macrocell);
            for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
              {
                subCellPtr = d_matrixFreeDataPtr->get_cell_iterator(
                  macrocell, iSubCell, d_matrixFreeVectorComponent);
                dealii::CellId             subCellId = subCellPtr->id();
                const std::vector<double> &tempVec =
                  d_smearedChargeValuesPtr->find(subCellId)->second;
                if (tempVec.size() == 0)
                  continue;

                for (unsigned int q = 0; q < numQuadPointsSmearedb; ++q)
                  smearedbQuads[q][d_smearedChargeGradientComponentId]
                               [iSubCell] = tempVec[q];

                isMacroCellTrivial = false;
              }

            if (!isMacroCellTrivial)
              {
                fe_eval_sc2.reinit(macrocell);
                for (unsigned int q = 0; q < fe_eval_sc2.n_q_points; ++q)
                  {
                    fe_eval_sc2.submit_gradient(smearedbQuads[q], q);
                  }
                fe_eval_sc2.integrate(false, true);
                fe_eval_sc2.distribute_local_to_global(rhs);
              }
          }
      }

    // MPI operation to sync data
    rhs.compress(dealii::VectorOperation::add);

    if (d_isReuseSmearedChargeRhs)
      rhs += d_rhsSmearedCharge;

    if (d_isStoreSmearedChargeRhs)
      d_rhsSmearedCharge.compress(dealii::VectorOperation::add);

    if (d_isMeanValueConstraintComputed)
      meanValueConstraintDistributeSlaveToMaster(rhs);

    // FIXME: check if this is really required
    d_constraintMatrixPtr->set_zero(rhs);
  }


  // Compute and fill value at mean value constrained dof
  // u_o= -\sum_{i \neq o} a_i * u_i where i runs over all dofs
  // except the mean value constrained dof (o^{th})
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrder, FEOrderElectro>::
    meanValueConstraintDistribute(distributedDeviceVec<double> &vec) const
  {
    // -\sum_{i \neq o} a_i * u_i computation which involves summation across
    // MPI tasks
    const unsigned int one                  = 1;
    double             constrainedNodeValue = 0.0;
    // dftfe::utils::deviceKernelsGeneric::dot(
    // d_meanValueConstraintDeviceVec.begin(),
    // vec.begin(),
    // d_xLocalDof,
    // mpi_communicator,
    // *d_deviceBlasHandlePtr); //FIX ME
    d_BLASWrapperPtr->xdot(d_xLocalDof,
                           d_meanValueConstraintDeviceVec.begin(),
                           one,
                           vec.begin(),
                           one,
                           mpi_communicator,
                           &constrainedNodeValue);

    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
        d_meanValueConstraintProcId)
      dftfe::utils::deviceSetValue(vec.begin() +
                                     d_meanValueConstraintNodeIdLocal,
                                   constrainedNodeValue,
                                   1);
  }

  // Distribute value at mean value constrained dof (u_o) to all other dofs
  // u_i+= -a_i * u_o, and subsequently set u_o to 0
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrder, FEOrderElectro>::
    meanValueConstraintDistributeSlaveToMaster(
      distributedDeviceVec<double> &vec) const
  {
    double constrainedNodeValue = 0;
    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
        d_meanValueConstraintProcId)


      dftfe::utils::MemoryTransfer<dftfe::utils::MemorySpace::HOST,
                                   dftfe::utils::MemorySpace::DEVICE>::
        copy(1,
             &constrainedNodeValue,
             vec.begin() + d_meanValueConstraintNodeIdLocal);


    // broadcast value at mean value constraint to all other tasks ids
    MPI_Bcast(&constrainedNodeValue,
              1,
              MPI_DOUBLE,
              d_meanValueConstraintProcId,
              mpi_communicator);

    // dftfe::utils::deviceKernelsGeneric::add(
    //   vec.begin(),
    //   d_meanValueConstraintDeviceVec.begin(),
    //   constrainedNodeValue,
    //   d_xLocalDof,
    //   *d_deviceBlasHandlePtr); //FIX ME

    d_BLASWrapperPtr->add(vec.begin(),
                          d_meanValueConstraintDeviceVec.begin(),
                          constrainedNodeValue,
                          d_xLocalDof);

    // meanValueConstraintSetZero
    if (d_isMeanValueConstraintComputed)
      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
          d_meanValueConstraintProcId)
        dftfe::utils::deviceMemset(
          vec.begin() + d_meanValueConstraintNodeIdLocal, 0, sizeof(double));
  }

  // Distribute value at mean value constrained dof (u_o) to all other dofs
  // u_i+= -a_i * u_o, and subsequently set u_o to 0
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrder, FEOrderElectro>::
    meanValueConstraintDistributeSlaveToMaster(
      distributedCPUVec<double> &vec) const
  {
    double constrainedNodeValue = 0;
    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
        d_meanValueConstraintProcId)
      constrainedNodeValue = vec[d_meanValueConstraintNodeIdLocal];

    // broadcast value at mean value constraint to all other tasks ids
    MPI_Bcast(&constrainedNodeValue,
              1,
              MPI_DOUBLE,
              d_meanValueConstraintProcId,
              mpi_communicator);

    vec.add(constrainedNodeValue, d_meanValueConstraintVec);

    meanValueConstraintSetZero(vec);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrder, FEOrderElectro>::
    meanValueConstraintSetZero(distributedCPUVec<double> &vec) const
  {
    if (d_isMeanValueConstraintComputed)
      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
          d_meanValueConstraintProcId)
        vec[d_meanValueConstraintNodeIdLocal] = 0;
  }

  //
  // Compute mean value constraint which is required in case of fully periodic
  // boundary conditions
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrder,
                             FEOrderElectro>::computeMeanValueConstraint()
  {
    // allocate parallel distibuted vector to store mean value constraint
    d_meanValueConstraintVec.reinit(*d_xPtr);
    d_meanValueConstraintVec = 0;

    // allocate parallel distibuted device vector to store mean value constraint
    dftfe::linearAlgebra::createMultiVectorFromDealiiPartitioner(
      d_meanValueConstraintVec.get_partitioner(),
      1,
      d_meanValueConstraintDeviceVec);

    const dealii::DoFHandler<3> &dofHandler =
      d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);

    const dealii::Quadrature<3> &quadrature =
      d_matrixFreeDataPtr->get_quadrature(d_matrixFreeQuadratureComponentAX);
    dealii::FEValues<3>    fe_values(dofHandler.get_fe(),
                                  quadrature,
                                  dealii::update_values |
                                    dealii::update_JxW_values);
    const unsigned int     dofs_per_cell   = dofHandler.get_fe().dofs_per_cell;
    const unsigned int     num_quad_points = quadrature.size();
    dealii::Vector<double> elementalValues(dofs_per_cell);
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

          elementalValues = 0.0;
          for (unsigned int i = 0; i < dofs_per_cell; i++)
            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
              elementalValues(i) +=
                fe_values.shape_value(i, q_point) * fe_values.JxW(q_point);

          d_constraintMatrixPtr->distribute_local_to_global(
            elementalValues, local_dof_indices, d_meanValueConstraintVec);
        }

    // MPI operation to sync data
    d_meanValueConstraintVec.compress(dealii::VectorOperation::add);

    dealii::IndexSet locallyOwnedElements =
      d_meanValueConstraintVec.locally_owned_elements();

    dealii::IndexSet locallyRelevantElements;
    dealii::DoFTools::extract_locally_relevant_dofs(dofHandler,
                                                    locallyRelevantElements);

    // pick mean value constrained node such that it is not part
    // of periodic and hanging node constraint equations (both slave and master
    // node). This is done for simplicity of implementation.
    dealii::IndexSet allIndicesTouchedByConstraints(
      d_meanValueConstraintVec.size());
    std::vector<dealii::types::global_dof_index> tempSet;
    for (dealii::IndexSet::ElementIterator it = locallyRelevantElements.begin();
         it < locallyRelevantElements.end();
         it++)
      if (d_constraintMatrixPtr->is_constrained(*it))
        {
          const dealii::types::global_dof_index lineDof = *it;
          const std::vector<std::pair<dealii::types::global_dof_index, double>>
            *rowData = d_constraintMatrixPtr->get_constraint_entries(lineDof);
          tempSet.push_back(lineDof);
          for (unsigned int j = 0; j < rowData->size(); ++j)
            tempSet.push_back((*rowData)[j].first);
        }

    if (d_atomsPtr)
      for (std::map<dealii::types::global_dof_index, double>::const_iterator
             it = (*d_atomsPtr).begin();
           it != (*d_atomsPtr).end();
           ++it)
        tempSet.push_back(it->first);

    allIndicesTouchedByConstraints.add_indices(tempSet.begin(), tempSet.end());
    locallyOwnedElements.subtract_set(allIndicesTouchedByConstraints);


    const unsigned int localSizeOfPotentialChoices =
      locallyOwnedElements.n_elements();
    const unsigned int totalProcs =
      dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
    const unsigned int this_mpi_process =
      dealii::Utilities::MPI::this_mpi_process(mpi_communicator);
    std::vector<unsigned int> localSizesOfPotentialChoices(totalProcs, 0);
    MPI_Allgather(&localSizeOfPotentialChoices,
                  1,
                  MPI_UNSIGNED,
                  &localSizesOfPotentialChoices[0],
                  1,
                  MPI_UNSIGNED,
                  mpi_communicator);

    d_meanValueConstraintProcId = 0;
    for (unsigned int iproc = 0; iproc < totalProcs; iproc++)
      {
        if (localSizesOfPotentialChoices[iproc] > 0)
          {
            d_meanValueConstraintProcId = iproc;
            break;
          }
      }

    double valueAtConstraintNode = 0;
    if (this_mpi_process == d_meanValueConstraintProcId)
      {
        AssertThrow(locallyOwnedElements.size() != 0,
                    dealii::ExcMessage(
                      "DFT-FE Error: please contact developers."));
        d_meanValueConstraintNodeId = *locallyOwnedElements.begin();
        AssertThrow(!d_constraintMatrixPtr->is_constrained(
                      d_meanValueConstraintNodeId),
                    dealii::ExcMessage(
                      "DFT-FE Error: Mean value constraint creation bug."));
        valueAtConstraintNode =
          d_meanValueConstraintVec[d_meanValueConstraintNodeId];
      }

    MPI_Bcast(&valueAtConstraintNode,
              1,
              MPI_DOUBLE,
              d_meanValueConstraintProcId,
              mpi_communicator);

    d_meanValueConstraintVec /= -valueAtConstraintNode;

    if (this_mpi_process == d_meanValueConstraintProcId)
      d_meanValueConstraintVec[d_meanValueConstraintNodeId] = 0;

    d_meanValueConstraintNodeIdLocal =
      d_meanValueConstraintVec.get_partitioner()->global_to_local(
        d_meanValueConstraintNodeId);

    dftfe::utils::MemoryTransfer<dftfe::utils::MemorySpace::DEVICE,
                                 dftfe::utils::MemorySpace::HOST>::
      copy(d_xLocalDof,
           d_meanValueConstraintDeviceVec.begin(),
           d_meanValueConstraintVec.begin());
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrder, FEOrderElectro>::computeDiagonalA()
  {
    d_diagonalA.reinit(*d_xPtr);
    d_diagonalA = 0;

    const dealii::DoFHandler<3> &dofHandler =
      d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);

    const dealii::Quadrature<3> &quadrature =
      d_matrixFreeDataPtr->get_quadrature(d_matrixFreeQuadratureComponentAX);
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
          for (unsigned int i = 0; i < dofs_per_cell; i++)
            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
              elementalDiagonalA(i) += (1.0 / (4.0 * M_PI)) *
                                       (fe_values.shape_grad(i, q_point) *
                                        fe_values.shape_grad(i, q_point)) *
                                       fe_values.JxW(q_point);

          d_constraintMatrixPtr->distribute_local_to_global(elementalDiagonalA,
                                                            local_dof_indices,
                                                            d_diagonalA);
        }

    // MPI operation to sync data
    d_diagonalA.compress(dealii::VectorOperation::add);

    for (dealii::types::global_dof_index i = 0; i < d_diagonalA.size(); i++)
      if (d_diagonalA.in_local_range(i))
        if (!d_constraintMatrixPtr->is_constrained(i))
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


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrder, FEOrderElectro>::setX()
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  distributedDeviceVec<double> &
  poissonSolverProblemDevice<FEOrder, FEOrderElectro>::getPreconditioner()
  {
    return d_diagonalAdevice;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrder, FEOrderElectro>::setupconstraints()
  {
    d_constraintsTotalPotentialInfo.initialize(
      d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent),
      *d_constraintMatrixPtr);
  }


  template <typename Type, int M, int N, int K, int dim>
  __global__ void
  computeAXKernel(Type *      V,
                  const Type *U,
                  const Type *P,
                  const Type *J,
                  const int * map)
  {
    // V = AU
    // gridDim.x = cells;
    // First index is fastest convention used
    // sharedT is used to temporarily store UP^T/UP
    // P(q*p), D(q*q), PT(p*q), DT(q*q)

    extern __shared__ Type SMem[];

    Type *sharedX  = SMem;
    Type *sharedY  = &sharedX[N * N * N];
    Type *sharedZ  = &sharedY[N * N * N];
    Type *sharedT  = &sharedZ[N * N * N];
    Type *sharedP  = &sharedT[N * N * N];
    Type *sharedD  = &sharedP[N * K];
    Type *sharedPT = &sharedD[N * N];
    Type *sharedDT = &sharedPT[K * N];
    Type *sharedJ  = &sharedDT[N * N];

    const int mapShift = blockIdx.x * M * K;

    // Copy Shape Function Values and Gradients to shared memory
#pragma unroll
    for (int i = threadIdx.x; i < 2 * N * (K + N); i += blockDim.x)
      sharedP[i] = P[i];

    __syncthreads();

    //////////////////////////////////////////////////////////////
    // Interpolation combined with Extraction
    // V -> UPPP
    // Z -> VDz
    // Y -> VDy
    // X -> VDx

    // 1st GEMM of P
    // Z Direction
    for (int i = threadIdx.x; i < M; i += blockDim.x)
      {
        Type x[N], u[K];

#pragma unroll
        for (int j = 0; j < N; j++)
          x[j] = 0.0;

        for (int k = 0; k < K; k++)
          {
            u[k] = U[map[i + k * M + mapShift]];

#pragma unroll
            for (int j = 0; j < N; j++)
              x[j] += sharedP[j + k * N] * u[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[i + j * M] = x[j];
      }

    __syncthreads();

    // 2nd GEMM of P
    // Y Direction
    for (int i = threadIdx.x; i < K * N; i += blockDim.x)
      {
        Type y[N], x[K];

        int a = i % K;
        int b = i / K;

#pragma unroll
        for (int j = 0; j < N; j++)
          y[j] = 0.0;

        for (int k = 0; k < K; k++)
          {
            x[k] = sharedX[a + k * K + b * M];

#pragma unroll
            for (int j = 0; j < N; j++)
              y[j] += sharedP[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedY[a + (j + b * N) * K] = y[j];
      }

    __syncthreads();

    // 3rd GEMM of P
    // X Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type x[N], y[K];

#pragma unroll
        for (int j = 0; j < N; j++)
          x[j] = 0.0;

        for (int k = 0; k < K; k++)
          {
            y[k] = sharedY[k + i * K];

#pragma unroll
            for (int j = 0; j < N; j++)
              x[j] += sharedP[j + k * N] * y[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[j + i * N] = x[j];
      }

    __syncthreads();

    // 1st GEMM of D
    // Z Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type y[N], x[N];

#pragma unroll
        for (int j = 0; j < N; j++)
          y[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[i + k * N * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              y[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedY[i + j * N * N] = y[j];
      }

    // 2nd GEMM of D
    // Y Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type z[N], x[N];

        int a = i % N;
        int b = i / N;

#pragma unroll
        for (int j = 0; j < N; j++)
          z[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[a + (k + b * N) * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              z[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedZ[a + (j + b * N) * N] = z[j];
      }

    // 3rd GEMM of D
    // X Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type t[N], x[N];

#pragma unroll
        for (int j = 0; j < N; j++)
          t[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[k + i * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              t[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedT[j + i * N] = t[j];
      }

    //////////////////////////////////////////////////////////////////
    // sharedT, sharedZ, sharedY have the respective gemms of X, Y, Z
    // directions

    const int JShift = blockIdx.x * dim * dim;

    // Copy Jacobian Factor to shared memory
#pragma unroll
    for (int i = threadIdx.x; i < dim * dim; i += blockDim.x)
      sharedJ[i] = J[i + JShift];

    __syncthreads();

    // Gemm with Jacobian Factor
#pragma unroll
    for (int i = threadIdx.x; i < N * N * N; i += blockDim.x)
      {
        Type v[3];

        v[2] = sharedY[i];
        v[1] = sharedZ[i];
        v[0] = sharedT[i];

        sharedY[i] = sharedJ[6] * v[0] + sharedJ[7] * v[1] + sharedJ[8] * v[2];
        sharedZ[i] = sharedJ[3] * v[0] + sharedJ[4] * v[1] + sharedJ[5] * v[2];
        sharedT[i] = sharedJ[0] * v[0] + sharedJ[1] * v[1] + sharedJ[2] * v[2];
      }

    __syncthreads();

    // Integration
    // Z -> Z(DT)z
    // Y -> Y(DT)y
    // X -> X(DT)x
    // V -> (Z + Y + X)(PT)(PT)(PT)

    // 1st GEMM of DT
    // Z Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type x[N], y[N];

#pragma unroll
        for (int j = 0; j < N; j++)
          x[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            y[k] = sharedY[i + k * N * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              x[j] += sharedDT[j + k * N] * y[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[i + j * N * N] = x[j];
      }

    __syncthreads();

    // 2nd GEMM of DT
    // Y Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type y[N], z[N];

        int a = i % N;
        int b = i / N;

#pragma unroll
        for (int j = 0; j < N; j++)
          y[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            z[k] = sharedZ[a + (k + b * N) * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              y[j] += sharedDT[j + k * N] * z[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[a + (j + b * N) * N] += y[j];
      }

    __syncthreads();

    // 3rd GEMM of DT
    // X Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type z[N], t[N];

#pragma unroll
        for (int j = 0; j < N; j++)
          z[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            t[k] = sharedT[k + i * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              z[j] += sharedDT[j + k * N] * t[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[j + i * N] += z[j];
      }

    __syncthreads();

    // 1st GEMM of PT
    // Z Direction
    for (int i = threadIdx.x; i < N * N; i += blockDim.x)
      {
        Type y[K], x[N];

#pragma unroll
        for (int j = 0; j < K; j++)
          y[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[i + k * N * N];

#pragma unroll
            for (int j = 0; j < K; j++)
              y[j] += sharedPT[j + k * K] * x[k];
          }

#pragma unroll
        for (int j = 0; j < K; j++)
          sharedY[i + j * N * N] = y[j];
      }

    __syncthreads();

    // 2nd GEMM of PT
    // Y Direction
    for (int i = threadIdx.x; i < N * K; i += blockDim.x)
      {
        Type x[K], y[N];

        int a = i % N;
        int b = i / N;

#pragma unroll
        for (int j = 0; j < K; j++)
          x[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            y[k] = sharedY[a + (k + b * N) * N];

#pragma unroll
            for (int j = 0; j < K; j++)
              x[j] += sharedPT[j + k * K] * y[k];
          }

#pragma unroll
        for (int j = 0; j < K; j++)
          sharedX[a + (j + b * K) * N] = x[j];
      }

    __syncthreads();

    // 3rd GEMM of PT
    // X Direction
    for (int i = threadIdx.x; i < M; i += blockDim.x)
      {
        Type y[K], x[N];

#pragma unroll
        for (int j = 0; j < K; j++)
          y[j] = 0.0;

        for (int k = 0; k < N; k++)
          {
            x[k] = sharedX[k + i * N];

#pragma unroll
            for (int j = 0; j < K; j++)
              y[j] += sharedPT[j + k * K] * x[k];
          }

#pragma unroll
        for (int j = 0; j < K; j++)
          atomicAdd(&V[map[j + i * K + mapShift]], y[j]);
      }
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrder, FEOrderElectro>::setupMatrixFree()
  {
    constexpr int    p              = FEOrderElectro + 1;
    constexpr int    q              = p;
    constexpr int    nDofsPerCell   = p * p * p;
    constexpr int    dim            = 3;
    constexpr double coeffLaplacian = 1.0 / (4.0 * M_PI);

    auto dofInfo =
      d_matrixFreeDataPtr->get_dof_info(d_matrixFreeVectorComponent);
    auto shapeInfo =
      d_matrixFreeDataPtr->get_shape_info(d_matrixFreeVectorComponent,
                                          d_matrixFreeQuadratureComponentAX);
    auto mappingData = d_matrixFreeDataPtr->get_mapping_info()
                         .cell_data[d_matrixFreeQuadratureComponentAX];
    auto shapeData = shapeInfo.get_shape_data();

    // Shape Function Values, Gradients and their Transposes
    // P(q*p), D(q*q), PT(p*q), DT(q*q)
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      shapeFunction(2 * q * (p + q));

    for (int i = 0; i < p; i++)
      for (int j = 0; j < q; j++)
        {
          double value = shapeData.shape_values[j + i * q][0] *
                         std::sqrt(shapeData.quadrature.weight(j));
          shapeFunction[j + i * q]               = value;
          shapeFunction[i + j * p + q * (p + q)] = value;
        }

    for (int i = 0; i < q; i++)
      for (int j = 0; j < q; j++)
        {
          double grad = shapeData.shape_gradients_collocation[j + i * q][0] *
                        std::sqrt(shapeData.quadrature.weight(j)) /
                        std::sqrt(shapeData.quadrature.weight(i));
          shapeFunction[j + i * q + q * p]           = grad;
          shapeFunction[i + j * q + (2 * p + q) * q] = grad;
        }

    // Jacobian
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      jacobianFactor(dim * dim * d_nLocalCells);

    auto cellOffsets = mappingData.data_index_offsets;

    for (int cellIdx = 0; cellIdx < d_nLocalCells; cellIdx++)
      for (int k = 0; k < dim; k++)
        for (int i = 0; i < dim; i++)
          for (int j = 0; j < dim; j++)
            jacobianFactor[j + i * dim + cellIdx * dim * dim] +=
              coeffLaplacian *
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
    dftfe::utils::MemoryStorage<int, dftfe::utils::MemorySpace::HOST> map(
      nDofsPerCell * d_nLocalCells);

    for (auto cellIdx = 0; cellIdx < d_nLocalCells; ++cellIdx)
      std::memcpy(map.data() + cellIdx * nDofsPerCell,
                  ((dofInfo.row_starts[cellIdx].second ==
                    dofInfo.row_starts[cellIdx + 1].second) &&
                   (dofInfo.row_starts_plain_indices[cellIdx] ==
                    dealii::numbers::invalid_unsigned_int)) ?
                    dofInfo.dof_indices.data() +
                      dofInfo.row_starts[cellIdx].first :
                    dofInfo.plain_dof_indices.data() +
                      dofInfo.row_starts_plain_indices[cellIdx],
                  nDofsPerCell * sizeof(unsigned int));

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

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    cudaFuncSetAttribute(computeAXKernel<double, p * p, q, p, dim>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);
#endif
  }


  // computeAX
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemDevice<FEOrder, FEOrderElectro>::computeAX(
    distributedDeviceVec<double> &Ax,
    distributedDeviceVec<double> &x)
  {
    constexpr int dim     = 3;
    constexpr int p       = FEOrderElectro + 1;
    constexpr int q       = p;
    constexpr int threads = 64;
    // constexpr int threads =
    //  (FEOrderElectro < 7 ? 96 : FEOrderElectro == 7 ? 64 : 256);
    const int             blocks = d_nLocalCells;
    constexpr std::size_t smem =
      (4 * q * q * q + 2 * p * q + 2 * q * q + dim * dim) * sizeof(double);

    dftfe::utils::deviceMemset(Ax.begin(), 0, d_xLen * sizeof(double));

    if (d_isMeanValueConstraintComputed)
      meanValueConstraintDistribute(x);

    x.updateGhostValues();

    d_constraintsTotalPotentialInfo.distribute(x, 1);

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    computeAXKernel<double, p * p, q, p, dim><<<blocks, threads, smem>>>(
      Ax.begin(), x.begin(), d_shapeFunctionPtr, d_jacobianFactorPtr, d_mapPtr);

#elif DFTFE_WITH_DEVICE_LANG_HIP
    hipLaunchKernelGGL(HIP_KERNEL_NAME(
                         computeAXKernel<double, p * p, q, p, dim>),
                       blocks,
                       threads,
                       smem,
                       0,
                       Ax.begin(),
                       x.begin(),
                       d_shapeFunctionPtr,
                       d_jacobianFactorPtr,
                       d_mapPtr);
#endif

    d_constraintsTotalPotentialInfo.set_zero(x, 1);

    d_constraintsTotalPotentialInfo.distribute_slave_to_master(Ax, 1);

    Ax.accumulateAddLocallyOwned();

    if (d_isMeanValueConstraintComputed)
      meanValueConstraintDistributeSlaveToMaster(Ax);
  }

#include "poissonSolverProblemDevice.inst.cc"
} // namespace dftfe
