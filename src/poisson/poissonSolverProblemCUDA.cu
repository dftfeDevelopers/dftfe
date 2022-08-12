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

#include <poissonSolverProblemCUDA.h>


namespace dftfe
{
  //
  // constructor
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::poissonSolverProblemCUDA(
    const MPI_Comm &mpi_comm)
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
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::clear()
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
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::reinit(
    const dealii::MatrixFree<3, double> &    matrixFreeData,
    distributedCPUVec<double> &              x,
    const dealii::AffineConstraints<double> &constraintMatrix,
    const unsigned int                       matrixFreeVectorComponent,
    const unsigned int matrixFreeQuadratureComponentRhsDensity,
    const unsigned int matrixFreeQuadratureComponentAX,
    const std::map<dealii::types::global_dof_index, double> &atoms,
    const std::map<dealii::CellId, std::vector<double>> &smearedChargeValues,
    const unsigned int smearedChargeQuadratureId,
    const std::map<dealii::CellId, std::vector<double>> &rhoValues,
    cublasHandle_t &                                     cublasHandle,
    const bool                                           isComputeDiagonalA,
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

    d_matrixFreeDataPtr = &matrixFreeData;
    d_xPtr              = &x;
    d_xDevice.reinit(d_xPtr->get_partitioner(), 1);
    cudaUtils::copyHostVecToCUDAVec<double>(d_xPtr->begin(),
                                            d_xDevice.begin(),
                                            d_xDevice.locallyOwnedDofsSize());

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
    d_cublasHandlePtr                  = &cublasHandle;
    d_nLocalCells                      = d_matrixFreeDataPtr->n_macro_cells();
    d_xLenLocalDof                     = d_xDevice.locallyOwnedDofsSize();
    d_xLenGhost                        = d_xDevice.ghostFlattenedSize();
    d_xLen = d_xDevice.locallyOwnedDofsSize() + d_xDevice.ghostFlattenedSize();
    d_fPI  = 1.0 / (4.0 * M_PI);

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
        d_constraintsInfo.initialize(matrixFreeData.get_vector_partitioner(
                                       matrixFreeVectorComponent),
                                     constraintMatrix);

        d_isFastConstraintsInitialized = true;
      }

    constraintsTotalPotentialInfo.initialize(
      d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent),
      *d_constraintMatrixPtr);
    constraintsTotalPotentialInfo.precomputeMaps(
      d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent),
      d_xPtr->get_partitioner(),
      1);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::copyCUDAToHost()
  {
    cudaUtils::copyCUDAVecToHostVec<double>(d_xDevice.begin(),
                                            d_xPtr->begin(),
                                            d_xLen);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::distributeX()
  {
    constraintsTotalPotentialInfo.distribute(d_xDevice, 1);

    // if (d_isMeanValueConstraintComputed)
      // meanValueConstraintDistribute(d_xDevice);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  distributedGPUVec<double> &
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::getX()
  {
    return d_xDevice;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::computeRhs(
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
             macrocell < d_matrixFreeDataPtr->n_macro_cells();
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
             macrocell < d_matrixFreeDataPtr->n_macro_cells();
             ++macrocell)
          {
            fe_eval_density.reinit(macrocell);

            std::fill(rhoQuads.begin(),
                      rhoQuads.end(),
                      dealii::make_vectorized_array(0.0));
            const unsigned int numSubCells =
              d_matrixFreeDataPtr->n_components_filled(macrocell);
            for (unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
              {
                subCellPtr = d_matrixFreeDataPtr->get_cell_iterator(
                  macrocell, iSubCell, d_matrixFreeVectorComponent);
                dealii::CellId             subCellId = subCellPtr->id();
                const std::vector<double> &tempVec =
                  d_rhoValuesPtr->find(subCellId)->second;

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
             macrocell < d_matrixFreeDataPtr->n_macro_cells();
             ++macrocell)
          {
            std::fill(smearedbQuads.begin(),
                      smearedbQuads.end(),
                      dealii::make_vectorized_array(0.0));
            bool               isMacroCellTrivial = true;
            const unsigned int numSubCells =
              d_matrixFreeDataPtr->n_components_filled(macrocell);
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
             macrocell < d_matrixFreeDataPtr->n_macro_cells();
             ++macrocell)
          {
            std::fill(smearedbQuads.begin(),
                      smearedbQuads.end(),
                      dealii::make_vectorized_array(0.0));
            bool               isMacroCellTrivial = true;
            const unsigned int numSubCells =
              d_matrixFreeDataPtr->n_components_filled(macrocell);
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

  // Matrix-Free Jacobi preconditioner application
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::precondition_Jacobi(
    distributedGPUVec<double> &      dst,
    const distributedGPUVec<double> &src) const
  {
    cudaUtils::scale<double>(dst.begin(),
                             d_diagonalAdevice.begin(),
                             src.begin(),
                             d_xLenLocalDof);
  }

  // Compute and fill value at mean value constrained dof
  // u_o= -\sum_{i \neq o} a_i * u_i where i runs over all dofs
  // except the mean value constrained dof (o^{th})
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::
    meanValueConstraintDistribute(distributedGPUVec<double> &vec) const
  {
    // -\sum_{i \neq o} a_i * u_i computation which involves summation across
    // MPI tasks
    const double constrainedNodeValue =
      cudaUtils::dot<double>(d_meanValueConstraintGPUVec.begin(),
                             vec.begin(),
                             d_xLenLocalDof,
                             mpi_communicator,
                             *d_cublasHandlePtr);

    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
        d_meanValueConstraintProcId)
      cudaUtils::set<double>(vec.begin() + d_meanValueConstraintNodeIdLocal,
                             constrainedNodeValue,
                             1);

    // cudaUtils::copyHostVecToCUDAVec<double>(&constrainedNodeValue,
    //                                       vec.begin() +
    //                                       d_meanValueConstraintNodeIdLocal,
    //                                       1);
  }

  // Distribute value at mean value constrained dof (u_o) to all other dofs
  // u_i+= -a_i * u_o, and subsequently set u_o to 0
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::
    meanValueConstraintDistributeSlaveToMaster(
      distributedGPUVec<double> &vec) const
  {
    double constrainedNodeValue = 0;
    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
        d_meanValueConstraintProcId)

      cudaUtils::copyCUDAVecToHostVec<double>(
        vec.begin() + d_meanValueConstraintNodeIdLocal,
        &constrainedNodeValue,
        1);

    // broadcast value at mean value constraint to all other tasks ids
    MPI_Bcast(&constrainedNodeValue,
              1,
              MPI_DOUBLE,
              d_meanValueConstraintProcId,
              mpi_communicator);

    cudaUtils::add<double>(vec.begin(),
                           d_meanValueConstraintGPUVec.begin(),
                           constrainedNodeValue,
                           d_xLenLocalDof,
                           *d_cublasHandlePtr);

    if (d_isMeanValueConstraintComputed)
      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
          d_meanValueConstraintProcId)
        cudaUtils::set<double>(vec.begin() + d_meanValueConstraintNodeIdLocal,
                               0,
                               1);

    // meanValueConstraintSetZero(vec);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::meanValueConstraintSetZero(
    distributedGPUVec<double> &vec) const
  {
    const double zero = 0.0;

    if (d_isMeanValueConstraintComputed)
      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
          d_meanValueConstraintProcId)
        cudaUtils::copyHostVecToCUDAVec<double>(
          &zero, vec.begin() + d_meanValueConstraintNodeIdLocal, 1);
  }

  // Compute and fill value at mean value constrained dof
  // u_o= -\sum_{i \neq o} a_i * u_i where i runs over all dofs
  // except the mean value constrained dof (o^{th})
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::
    meanValueConstraintDistribute(distributedCPUVec<double> &vec) const
  {
    // -\sum_{i \neq o} a_i * u_i computation which involves summation across
    // MPI tasks
    const double constrainedNodeValue = d_meanValueConstraintVec * vec;

    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
        d_meanValueConstraintProcId)
      vec[d_meanValueConstraintNodeIdLocal] = constrainedNodeValue;
  }

  // Distribute value at mean value constrained dof (u_o) to all other dofs
  // u_i+= -a_i * u_o, and subsequently set u_o to 0
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::
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
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::meanValueConstraintSetZero(
    distributedCPUVec<double> &vec) const
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
  poissonSolverProblemCUDA<FEOrder,
                           FEOrderElectro>::computeMeanValueConstraint()
  {
    // allocate parallel distibuted vector to store mean value constraint
    d_meanValueConstraintVec.reinit(*d_xPtr);
    d_meanValueConstraintVec = 0;

    // allocate parallel distibuted gpu vector to store mean value constraint
    d_meanValueConstraintGPUVec.reinit(
      d_meanValueConstraintVec.get_partitioner(), 1);

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
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
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
      // d_xPtr->get_partitioner()->global_to_local(
      d_meanValueConstraintVec.get_partitioner()->global_to_local(
        d_meanValueConstraintNodeId);
    cudaUtils::copyHostVecToCUDAVec<double>(d_meanValueConstraintVec.begin(),
                                            d_meanValueConstraintGPUVec.begin(),
                                            d_xLenLocalDof);
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::computeDiagonalA()
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
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
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

    for (dealii::types::global_dof_index i = 0; i < d_diagonalA.size(); ++i)
      if (d_diagonalA.in_local_range(i))
        if (!d_constraintMatrixPtr->is_constrained(i))
          d_diagonalA(i) = 1.0 / d_diagonalA(i);

    d_diagonalA.compress(dealii::VectorOperation::insert);
    d_diagonalAdevice.reinit(d_diagonalA.get_partitioner(), 1);
    cudaUtils::copyHostVecToCUDAVec<double>(d_diagonalA.begin(),
                                            d_diagonalAdevice.begin(),
                                            d_xLenLocalDof);
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::setX()
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }


  template <typename Type, int blockSize>
  __global__ void
  cgKernel(Type *hvec, Type *gvec, Type *diagA, Type *dev_sum, int N)
  {
    __shared__ Type smem[blockSize];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockSize * 2) + threadIdx.x;
    cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

    Type localSum;

    if (idx < N)
      {
        Type d_diagA = diagA[idx];
        Type d_gvec  = gvec[idx];

        localSum  = d_diagA * d_gvec * d_gvec;
        hvec[idx] = d_diagA * d_gvec;
      }

    else
      {
        localSum = 0;
      }

    if (idx + blockSize < N)
      {
        Type d_diagA = diagA[idx + blockSize];
        Type d_gvec  = gvec[idx + blockSize];
        localSum += d_diagA * d_gvec * d_gvec;
        hvec[idx + blockSize] = d_diagA * d_gvec;
      }

    smem[tid] = localSum;
    cooperative_groups::sync(block);

    if ((blockSize >= 512) && (tid < 256))
      smem[tid] = localSum = localSum + smem[tid + 256];

    cooperative_groups::sync(block);

    if ((blockSize >= 256) && (tid < 128))
      smem[tid] = localSum = localSum + smem[tid + 128];

    cooperative_groups::sync(block);

    if ((blockSize >= 128) && (tid < 64))
      smem[tid] = localSum = localSum + smem[tid + 64];

    cooperative_groups::sync(block);

    cooperative_groups::thread_block_tile<32> tile32 =
      cooperative_groups::tiled_partition<32>(block);

    if (block.thread_rank() < 32)
      {
        if (blockSize >= 64)
          localSum += smem[tid + 32];

        for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
          localSum += tile32.shfl_down(localSum, offset);
      }

    if (block.thread_rank() == 0)
      atomicAdd(&dev_sum[0], localSum);
  }


  template <typename Type, int blockSize>
  __global__ void
  cgKernel2(Type *hvec,
            Type *gvec,
            Type *dvec,
            Type *diagA,
            Type *dev_sum,
            int   N)
  {
    __shared__ Type smem[blockSize];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockSize * 2) + threadIdx.x;
    cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

    Type localSum;

    if (idx < N)
      {
        Type d_diagA = diagA[idx];
        Type d_gvec  = gvec[idx];

        localSum  = d_diagA * d_gvec * d_gvec;
        dvec[idx] = -1 * d_diagA * d_gvec;
      }

    else
      {
        localSum = 0;
      }

    if (idx + blockSize < N)
      {
        Type d_diagA = diagA[idx + blockSize];
        Type d_gvec  = gvec[idx + blockSize];
        localSum += d_diagA * d_gvec * d_gvec;
        dvec[idx + blockSize] = -1 * d_diagA * d_gvec;
      }

    smem[tid] = localSum;
    cooperative_groups::sync(block);

    if ((blockSize >= 512) && (tid < 256))
      smem[tid] = localSum = localSum + smem[tid + 256];

    cooperative_groups::sync(block);

    if ((blockSize >= 256) && (tid < 128))
      smem[tid] = localSum = localSum + smem[tid + 128];

    cooperative_groups::sync(block);

    if ((blockSize >= 128) && (tid < 64))
      smem[tid] = localSum = localSum + smem[tid + 64];

    cooperative_groups::sync(block);

    cooperative_groups::thread_block_tile<32> tile32 =
      cooperative_groups::tiled_partition<32>(block);

    if (block.thread_rank() < 32)
      {
        if (blockSize >= 64)
          localSum += smem[tid + 32];

        for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
          localSum += tile32.shfl_down(localSum, offset);
      }

    if (block.thread_rank() == 0)
      atomicAdd(&dev_sum[0], localSum);
  }

  template <typename Type, int blockSize>
  __global__ void
  cgKernel3(Type *hvec,
            Type *gvec,
            Type *dvec,
            Type *x,
            Type *dev_sum,
            Type  alpha,
            int   N)
  {
    __shared__ Type smem[blockSize];

    int tid = threadIdx.x;
    int idx = blockIdx.x * (blockSize * 2) + threadIdx.x;
    cooperative_groups::thread_block block =
      cooperative_groups::this_thread_block();

    Type localSum;

    if (idx < N)
      {
        Type d_gvec = gvec[idx];
        localSum    = d_gvec * d_gvec;
        x[idx] += alpha * dvec[idx];
        gvec[idx] += alpha * hvec[idx];
      }

    else
      {
        localSum = 0;
      }

    if (idx + blockSize < N)
      {
        Type d_gvec = gvec[idx + blockSize];
        localSum += d_gvec * d_gvec;
        x[idx + blockSize] += alpha * dvec[idx + blockSize];
        gvec[idx + blockSize] += alpha * hvec[idx + blockSize];
      }

    smem[tid] = localSum;
    cooperative_groups::sync(block);

    if ((blockSize >= 512) && (tid < 256))
      smem[tid] = localSum = localSum + smem[tid + 256];

    cooperative_groups::sync(block);

    if ((blockSize >= 256) && (tid < 128))
      smem[tid] = localSum = localSum + smem[tid + 128];

    cooperative_groups::sync(block);

    if ((blockSize >= 128) && (tid < 64))
      smem[tid] = localSum = localSum + smem[tid + 64];

    cooperative_groups::sync(block);

    cooperative_groups::thread_block_tile<32> tile32 =
      cooperative_groups::tiled_partition<32>(block);

    if (block.thread_rank() < warpSize)
      {
        if (blockSize >= 64)
          localSum += smem[tid + 32];

        for (int offset = tile32.size() / 2; offset > 0; offset /= 2)
          localSum += tile32.shfl_down(localSum, offset);
      }

    if (block.thread_rank() == 0)
      atomicAdd(&dev_sum[0], localSum);
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::cg(double *hvec,
                                                        double *gvec)
  {
    double    local_sum = 0.0, sum = 0.0;
    const int blocks = (d_xLenLocalDof + (cudaConstants::blockSize * 2 - 1)) /
                       (cudaConstants::blockSize * 2);

    dev_sum[0] = 0.0;

    cgKernel<double, cudaConstants::blockSize>
      <<<blocks, cudaConstants::blockSize>>>(
        hvec, gvec, d_diagonalAdevice.begin(), sum_ptr, d_xLenLocalDof);

    local_sum = dev_sum[0];

    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    return sum;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::cg2(double *hvec,
                                                         double *gvec,
                                                         double *dvec)
  {
    double    local_sum = 0.0, sum = 0.0;
    const int blocks = (d_xLenLocalDof + (cudaConstants::blockSize * 2 - 1)) /
                       (cudaConstants::blockSize * 2);

    dev_sum[0] = 0.0;

    cgKernel2<double, cudaConstants::blockSize>
      <<<blocks, cudaConstants::blockSize>>>(
        hvec, gvec, dvec, d_diagonalAdevice.begin(), sum_ptr, d_xLenLocalDof);

    local_sum = dev_sum[0];

    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    return sum;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::cg3(double *hvec,
                                                         double *gvec,
                                                         double *dvec,
                                                         double &alpha)
  {
    double    local_sum = 0.0, sum = 0.0;
    const int blocks = (d_xLenLocalDof + (cudaConstants::blockSize * 2 - 1)) /
                       (cudaConstants::blockSize * 2);

    dev_sum[0] = 0.0;

    cgKernel3<double, cudaConstants::blockSize>
      <<<blocks, cudaConstants::blockSize>>>(
        hvec, gvec, dvec, d_xDevice.begin(), sum_ptr, alpha, d_xLenLocalDof);

    local_sum = dev_sum[0];

    MPI_Allreduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);

    return std::sqrt(sum);
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::setupMatrixFree()
  {
    constexpr int p   = FEOrderElectro + 1;
    constexpr int dim = 3;

    // shape info helps to obtain reference cell basis function and lex
    // numbering
    const dealii::DoFHandler<3> &dofHandler =
      d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);
    const int dofs_per_cell = dofHandler.get_fe().dofs_per_cell;

    dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shapeInfo;
    const dealii::Quadrature<3> &                            quadrature =
      d_matrixFreeDataPtr->get_quadrature(d_matrixFreeQuadratureComponentAX);

    int               num_quad_points = std::cbrt(quadrature.size());
    dealii::QGauss<1> quad(num_quad_points);
    shapeInfo.reinit(quad, dofHandler.get_fe());
    std::vector<unsigned int> lexMap3D = shapeInfo.lexicographic_numbering;

    const auto shapeGrad  = shapeInfo.data.front().shape_gradients;
    const auto shapeValue = shapeInfo.data.front().shape_values;

    dealii::FE_Q<1> feCell1D(FEOrderElectro);
    shapeInfo.reinit(quad, feCell1D);
    std::vector<unsigned int> lexMap1D = shapeInfo.lexicographic_numbering;

    thrust::host_vector<double> quad_weights(p);
    for (int i = 0; i < p; i++)
      {
        quad_weights[i] = quad.weight(lexMap1D[i]);
      }

    thrust::host_vector<double> spV(p * p), spG(p * p);
    for (int i = 0; i < p; i++)
      {
        for (int j = 0; j < p; j++)
          {
            spV[i + j * p] = shapeValue[i * p + lexMap1D[j]];
            spG[i + j * p] = shapeGrad[i * p + lexMap1D[j]];
          }
      }

    dealii::Triangulation<1> reference_cell;
    dealii::GridGenerator::hyper_cube(reference_cell, 0, 1);
    dealii::FEValues<1> fe_values_reference(feCell1D,
                                            quad,
                                            dealii::update_values |
                                              dealii::update_gradients |
                                              dealii::update_JxW_values);
    fe_values_reference.reinit(reference_cell.begin());

    // Map making
    thrust::host_vector<int> map(dofs_per_cell * d_nLocalCells);
    std::vector<dealii::types::global_dof_index> local_dof_globalIndices(
      dofs_per_cell);

    // Lexicographic Map making
    int cellIdx = 0;
    for (const auto &cell : dofHandler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            cell->get_dof_indices(local_dof_globalIndices);

            for (int dofIdx = 0; dofIdx < dofs_per_cell; dofIdx++)
              {
                dealii::types::global_dof_index globalIdx =
                  local_dof_globalIndices[lexMap3D[dofIdx]];
                int localIdx =
                  d_xPtr->get_partitioner()->global_to_local(globalIdx);
                map[dofIdx + cellIdx * dofs_per_cell] = localIdx;
              }
            cellIdx++;
          }
      }

    // Jacobian
    dealii::QGauss<dim> quadrature_formula(dofHandler.get_fe().degree + 1);
    const int           n_q_points = quadrature_formula.size();

    std::vector<dealii::DerivativeForm<1, dim, dim>> inv_jacobians_tensor;
    thrust::host_vector<double> invJac(d_nLocalCells * dim * dim);
    dealii::FEValues<dim>       fe_values(dofHandler.get_fe(),
                                    quadrature_formula,
                                    dealii::update_inverse_jacobians |
                                      dealii::update_JxW_values |
                                      dealii::update_quadrature_points);

    cellIdx = 0;
    for (const auto &cell : dofHandler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            inv_jacobians_tensor = fe_values.get_inverse_jacobians();

            for (int d = 0; d < dim; d++)
              for (int e = 0; e < dim; e++)
                invJac[cellIdx * dim * dim + d * dim + e] =
                  inv_jacobians_tensor[0][d][e];

            cellIdx++;
          }
      }

    dev_sum.resize(1);

    // Construct the device vectors
    d_shapeFunctionValue    = spV;
    d_shapeFunctionGradient = spG;
    d_weights               = quad_weights;
    d_inverseJacobian       = invJac;
    d_map                   = map;

    shapeFunctionValue_ptr =
      thrust::raw_pointer_cast(d_shapeFunctionValue.data());
    shapeFunctionGradient_ptr =
      thrust::raw_pointer_cast(d_shapeFunctionGradient.data());
    weights_ptr         = thrust::raw_pointer_cast(d_weights.data());
    inverseJacobian_ptr = thrust::raw_pointer_cast(d_inverseJacobian.data());
    map_ptr             = thrust::raw_pointer_cast(d_map.data());
    sum_ptr             = thrust::raw_pointer_cast(dev_sum.data());
  }


  template <typename Type, int p, int vec_shared, int dim>
  __global__ void
  computeAXKernel(Type *      V,
                  const Type *U,
                  const Type *N,
                  const Type *D,
                  const Type *W,
                  const Type *J_inv,
                  const int * map,
                  const int   lenU,
                  const Type  fPI)
  {
    // V = AU
    // gridDim.y = batch;
    // gridDim.x = cells;
    // n_vec = vec_shared * gridDim.y;
    // vec_shared - No of vectors in shared memory

    if (p < 11)
      {
        __shared__ Type shared_X[vec_shared * p * p * p],
          shared_Y[vec_shared * p * p * p], shared_Z[vec_shared * p * p * p],
          shared_T[vec_shared * p * p * p];
        __shared__ Type shared_N[p * p], shared_D[p * p], shared_N_T[p * p],
          shared_D_T[p * p], shared_W[p], shared_J_inv[dim * dim];
        __shared__ int shared_map[p * p * p];

#pragma unroll
        for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < p * p * p;
             i += blockDim.x * blockDim.y)
          {
            shared_map[i] = map[i + blockIdx.x * p * p * p];
          }

          // Can optimize by separate i and transpose, use shared_X as
          // intermidiary or shared_Y

#pragma unroll
        for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < p * p;
             i += blockDim.x * blockDim.y)
          {
            shared_N[i]                       = N[i];
            shared_N_T[(i / p) + (i % p) * p] = N[i];
            shared_D[i]                       = D[i];
            shared_D_T[(i / p) + (i % p) * p] = D[i];
          }

        // #pragma unroll
        // for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < p; i +=
        // blockDim.x * blockDim.y) { 	shared_W[i] = W[i];
        // }

        // #pragma unroll
        // for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < dim*dim; i
        // += blockDim.x * blockDim.y) { 	shared_J_inv[i] = J_inv[i + blockIdx.x
        // * dim*dim];
        // }

        __syncthreads();

        //////////////////////////////////////////////////////////////

        // First index is the fastest
        // Shared_U = [vec_shared][p^3]
        // For x - DNNU
        // For y - NDNU
        // For z - NNDU

        // Interpolation combined with Extraction

        // 1st GEMM
        // X, Y and Z Directions
        for (int k = threadIdx.y; k < p * p; k += blockDim.y)
          {
            Type temp1[p], temp2[p], u[p];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp1[i] = 0.0;
                temp2[i] = 0.0;
              }

            for (int m = 0; m < p; ++m)
              {
                u[m] = U[threadIdx.x + shared_map[k + m * p * p] * vec_shared +
                         blockIdx.y * vec_shared * lenU];
                // u[m] = U[threadIdx.x + map[k + m*p*p + blockIdx.x * p*p*p] *
                // vec_shared + blockIdx.y * vec_shared*lenU];

#pragma unroll
                for (int i = 0; i < p; ++i)
                  {
                    temp1[i] += shared_D_T[i + m * p] * u[m];
                    temp2[i] += shared_N_T[i + m * p] * u[m];
                  }
              }

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                shared_Y[threadIdx.x + i * vec_shared + k * p * vec_shared] =
                  temp1[i];
                shared_X[threadIdx.x + i * vec_shared + k * p * vec_shared] =
                  temp2[i];
              }
          }

        __syncthreads();

        // 2nd GEMM
        // Z Direction
        for (int k = threadIdx.y; k < p * p; k += blockDim.y)
          {
            Type temp[p], u[p];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] = 0.0;
              }

            for (int m = 0; m < p; ++m)
              {
                u[m] = shared_Y[threadIdx.x + k * vec_shared +
                                m * p * p * vec_shared];

#pragma unroll
                for (int i = 0; i < p; ++i)
                  {
                    temp[i] += shared_N_T[i + m * p] * u[m];
                  }
              }

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                shared_T[threadIdx.x + i * vec_shared + k * p * vec_shared] =
                  temp[i];
              }
          }

        __syncthreads();

        // X and Y Directions
        for (int k = threadIdx.y; k < p * p; k += blockDim.y)
          {
            Type temp1[p], temp2[p], u[p];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp1[i] = 0.0;
                temp2[i] = 0.0;
              }

            for (int m = 0; m < p; ++m)
              {
                u[m] = shared_X[threadIdx.x + k * vec_shared +
                                m * p * p * vec_shared];

#pragma unroll
                for (int i = 0; i < p; ++i)
                  {
                    temp1[i] += shared_N_T[i + m * p] * u[m];
                    temp2[i] += shared_D_T[i + m * p] * u[m];
                  }
              }

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                shared_Y[threadIdx.x + i * vec_shared + k * p * vec_shared] =
                  temp1[i];
                shared_Z[threadIdx.x + i * vec_shared + k * p * vec_shared] =
                  temp2[i];
              }
          }

        __syncthreads();

        // 3rd GEMM
        // X Direction
        for (int k = threadIdx.y; k < p * p; k += blockDim.y)
          {
            Type temp[p], u[p];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] = 0.0;
              }

            for (int m = 0; m < p; ++m)
              {
                u[m] = shared_Y[threadIdx.x + k * vec_shared +
                                m * p * p * vec_shared];

#pragma unroll
                for (int i = 0; i < p; ++i)
                  {
                    temp[i] += shared_D_T[i + m * p] * u[m];
                  }
              }

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                shared_X[threadIdx.x + i * vec_shared + k * p * vec_shared] =
                  temp[i];
              }
          }

        __syncthreads();

        // Y Direction
        for (int k = threadIdx.y; k < p * p; k += blockDim.y)
          {
            Type temp[p], u[p];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] = 0.0;
              }

            for (int m = 0; m < p; ++m)
              {
                u[m] = shared_Z[threadIdx.x + k * vec_shared +
                                m * p * p * vec_shared];

#pragma unroll
                for (int i = 0; i < p; ++i)
                  {
                    temp[i] += shared_N_T[i + m * p] * u[m];
                  }
              }

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                shared_Y[threadIdx.x + i * vec_shared + k * p * vec_shared] =
                  temp[i];
              }
          }

        __syncthreads();

        // Z Direction
        for (int k = threadIdx.y; k < p * p; k += blockDim.y)
          {
            Type temp[p], u[p];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] = 0.0;
              }

            for (int m = 0; m < p; ++m)
              {
                u[m] = shared_T[threadIdx.x + k * vec_shared +
                                m * p * p * vec_shared];

#pragma unroll
                for (int i = 0; i < p; ++i)
                  {
                    temp[i] += shared_N_T[i + m * p] * u[m];
                  }
              }

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                shared_Z[threadIdx.x + i * vec_shared + k * p * vec_shared] =
                  temp[i];
              }
          }

        __syncthreads();

        // shared_X, shared_Y, shared_Z have the respective gemms
        // V used to store pieces of X, Y and Z
        // Copy W and J_inv to shared memory

#pragma unroll
        for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < p;
             i += blockDim.x * blockDim.y)
          {
            shared_W[i] = W[i];
          }

#pragma unroll
        for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < dim * dim;
             i += blockDim.x * blockDim.y)
          {
            shared_J_inv[i] = J_inv[i + blockIdx.x * dim * dim];
          }

        __syncthreads();

        //////////////// Switch to J_inv_T ////////////////////
        // Can separate the 3 lines to 3 loops
        // Change the order of the x, y, z (z first)

        // Gemm with J_inv

#pragma unroll
        for (int k = threadIdx.y; k < p * p * p; k += blockDim.y)
          {
            Type temp[3];
            temp[0] = shared_X[threadIdx.x + k * vec_shared];
            temp[1] = shared_Y[threadIdx.x + k * vec_shared];
            temp[2] = shared_Z[threadIdx.x + k * vec_shared];

            shared_X[threadIdx.x + k * vec_shared] = shared_J_inv[0] * temp[0] +
                                                     shared_J_inv[1] * temp[1] +
                                                     shared_J_inv[2] * temp[2];
            shared_Y[threadIdx.x + k * vec_shared] = shared_J_inv[3] * temp[0] +
                                                     shared_J_inv[4] * temp[1] +
                                                     shared_J_inv[5] * temp[2];
            shared_Z[threadIdx.x + k * vec_shared] = shared_J_inv[6] * temp[0] +
                                                     shared_J_inv[7] * temp[1] +
                                                     shared_J_inv[8] * temp[2];
          }

        __syncthreads();

        // Point-wise multiplication

#pragma unroll
        for (int k = threadIdx.y; k < p * p * p; k += blockDim.y)
          {
            int idx[3];
            idx[0] = k / (p * p);
            idx[1] = (k % (p * p)) / p;
            idx[2] = (k % (p * p)) % p;

            Type R =
              (fPI * shared_W[idx[0]] * shared_W[idx[1]] * shared_W[idx[2]]) /
              (shared_J_inv[0] * (shared_J_inv[4] * shared_J_inv[8] -
                                  shared_J_inv[5] * shared_J_inv[7]) -
               shared_J_inv[3] * (shared_J_inv[1] * shared_J_inv[8] -
                                  shared_J_inv[2] * shared_J_inv[7]) +
               shared_J_inv[6] * (shared_J_inv[1] * shared_J_inv[5] -
                                  shared_J_inv[2] * shared_J_inv[4]));

            shared_X[threadIdx.x + k * vec_shared] *= R;
            shared_Y[threadIdx.x + k * vec_shared] *= R;
            shared_Z[threadIdx.x + k * vec_shared] *= R;
          }

        __syncthreads();

        // Gemm with J_inv transpose

#pragma unroll
        for (int k = threadIdx.y; k < p * p * p; k += blockDim.y)
          {
            Type temp[3];
            temp[0] = shared_X[threadIdx.x + k * vec_shared];
            temp[1] = shared_Y[threadIdx.x + k * vec_shared];
            temp[2] = shared_Z[threadIdx.x + k * vec_shared];

            shared_X[threadIdx.x + k * vec_shared] = shared_J_inv[0] * temp[0] +
                                                     shared_J_inv[3] * temp[1] +
                                                     shared_J_inv[6] * temp[2];
            shared_Y[threadIdx.x + k * vec_shared] = shared_J_inv[1] * temp[0] +
                                                     shared_J_inv[4] * temp[1] +
                                                     shared_J_inv[7] * temp[2];
            shared_Z[threadIdx.x + k * vec_shared] = shared_J_inv[2] * temp[0] +
                                                     shared_J_inv[5] * temp[1] +
                                                     shared_J_inv[8] * temp[2];
          }

        __syncthreads();

        // Integration

        // 1st GEMM
        // Z Direction
        for (int k = threadIdx.y; k < p * p; k += blockDim.y)
          {
            Type temp[p], u[p];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] = 0.0;
              }

            for (int m = 0; m < p; ++m)
              {
                u[m] = shared_Z[threadIdx.x + k * vec_shared +
                                m * p * p * vec_shared];

#pragma unroll
                for (int i = 0; i < p; ++i)
                  {
                    temp[i] += shared_D[i + m * p] * u[m];
                  }
              }

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                shared_T[threadIdx.x + i * vec_shared + k * p * vec_shared] =
                  temp[i];
              }
          }

        __syncthreads();

        // Y Direction
        for (int k = threadIdx.y; k < p * p; k += blockDim.y)
          {
            Type temp[p], u[p];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] = 0.0;
              }

            for (int m = 0; m < p; ++m)
              {
                u[m] = shared_Y[threadIdx.x + k * vec_shared +
                                m * p * p * vec_shared];

#pragma unroll
                for (int i = 0; i < p; ++i)
                  {
                    temp[i] += shared_N[i + m * p] * u[m];
                  }
              }

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                shared_Z[threadIdx.x + i * vec_shared + k * p * vec_shared] =
                  temp[i];
              }
          }

        __syncthreads();

        // X Direction
        for (int k = threadIdx.y; k < p * p; k += blockDim.y)
          {
            Type temp[p], u[p];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] = 0.0;
              }

            for (int m = 0; m < p; ++m)
              {
                u[m] = shared_X[threadIdx.x + k * vec_shared +
                                m * p * p * vec_shared];

#pragma unroll
                for (int i = 0; i < p; ++i)
                  {
                    temp[i] += shared_N[i + m * p] * u[m];
                  }
              }

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                shared_Y[threadIdx.x + i * vec_shared + k * p * vec_shared] =
                  temp[i];
              }
          }

        __syncthreads();

        // 2nd GEMM
        // X Direction
        for (int k = threadIdx.y; k < p * p; k += blockDim.y)
          {
            Type temp[p], u[p];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] = 0.0;
              }

            for (int m = 0; m < p; ++m)
              {
                u[m] = shared_Y[threadIdx.x + k * vec_shared +
                                m * p * p * vec_shared];

#pragma unroll
                for (int i = 0; i < p; ++i)
                  {
                    temp[i] += shared_N[i + m * p] * u[m];
                  }
              }

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                shared_X[threadIdx.x + i * vec_shared + k * p * vec_shared] =
                  temp[i];
              }
          }

        __syncthreads();

        // Y Direction
        for (int k = threadIdx.y; k < p * p; k += blockDim.y)
          {
            Type temp[p], u[p];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] = 0.0;
              }

            for (int m = 0; m < p; ++m)
              {
                u[m] = shared_Z[threadIdx.x + k * vec_shared +
                                m * p * p * vec_shared];

#pragma unroll
                for (int i = 0; i < p; ++i)
                  {
                    temp[i] += shared_D[i + m * p] * u[m];
                  }
              }

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                shared_Y[threadIdx.x + i * vec_shared + k * p * vec_shared] =
                  temp[i];
              }
          }

        __syncthreads();

        // Z Direction
        for (int k = threadIdx.y; k < p * p; k += blockDim.y)
          {
            Type temp[p], u[p];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] = 0.0;
              }

            for (int m = 0; m < p; ++m)
              {
                u[m] = shared_T[threadIdx.x + k * vec_shared +
                                m * p * p * vec_shared];

#pragma unroll
                for (int i = 0; i < p; ++i)
                  {
                    temp[i] += shared_N[i + m * p] * u[m];
                  }
              }

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                shared_Z[threadIdx.x + i * vec_shared + k * p * vec_shared] =
                  temp[i];
              }
          }

        __syncthreads();

        // Combine Y and X Direction to V after Z Direction

        // 3rd GEMM
        // Z Direction
        for (int k = threadIdx.y; k < p * p; k += blockDim.y)
          {
            Type temp[p], u[p];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] = 0.0;
              }

            for (int m = 0; m < p; ++m)
              {
                u[m] = shared_Z[threadIdx.x + k * vec_shared +
                                m * p * p * vec_shared];

#pragma unroll
                for (int i = 0; i < p; ++i)
                  {
                    temp[i] += shared_N[i + m * p] * u[m];
                  }
              }

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                shared_T[threadIdx.x + i * vec_shared + k * p * vec_shared] =
                  temp[i];
              }
          }

        __syncthreads();

        // Y Direction
        for (int k = threadIdx.y; k < p * p; k += blockDim.y)
          {
            Type temp[p], u[p];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] = 0.0;
              }

            for (int m = 0; m < p; ++m)
              {
                u[m] = shared_Y[threadIdx.x + k * vec_shared +
                                m * p * p * vec_shared];

#pragma unroll
                for (int i = 0; i < p; ++i)
                  {
                    temp[i] += shared_N[i + m * p] * u[m];
                  }
              }

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                shared_Z[threadIdx.x + i * vec_shared + k * p * vec_shared] =
                  temp[i];
              }
          }

        __syncthreads();

        // X Direction
        for (int k = threadIdx.y; k < p * p; k += blockDim.y)
          {
            Type temp[p], u[p];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] = 0.0;
              }

            for (int m = 0; m < p; ++m)
              {
                u[m] = shared_X[threadIdx.x + k * vec_shared +
                                m * p * p * vec_shared];

#pragma unroll
                for (int i = 0; i < p; ++i)
                  {
                    temp[i] += shared_D[i + m * p] * u[m];
                  }
              }

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                shared_Y[threadIdx.x + i * vec_shared + k * p * vec_shared] =
                  temp[i];
              }
          }

        __syncthreads();

#pragma unroll
        for (int k = threadIdx.y; k < p * p * p; k += blockDim.y)
          {
            atomicAdd(&V[threadIdx.x + shared_map[k] * vec_shared +
                         blockIdx.y * vec_shared * lenU],
                      shared_Y[threadIdx.x + k * vec_shared] +
                        shared_Z[threadIdx.x + k * vec_shared] +
                        shared_T[threadIdx.x + k * vec_shared]);
            // atomicAdd(&V[threadIdx.x + map[k + blockIdx.x * p*p*p] *
            // vec_shared + blockIdx.y * vec_shared*lenU], shared_Y[threadIdx.x
            // + k * vec_shared] + shared_Z[threadIdx.x + k * vec_shared] +
            // shared_T[threadIdx.x + k * vec_shared]);
          }
      }
  }


  template <typename Type, int p, int vec_shared, int dim>
  __global__ void
  massMatrixKernel(Type *    V,
                   Type *    U,
                   Type *    N,
                   Type *    W,
                   Type *    J,
                   int *     map,
                   const int lenU)
  {
    // gridDim.y = batch;
    // gridDim.x = cells;
    // n_vec = vec_shared * gridDim.y;

    // extern __shared__ Type SMem[];
    // Type *shared_U	 = SMem;
    // Type *shared_V	 = &shared_U[vec_shared*p*p*p];
    // Type *shared_N	 = &shared_V[vec_shared*p*p*p];
    // Type *shared_N_T	 = &shared_N[p*p];
    // Type *shared_W	 = &shared_N_T[p*p];
    // Type *shared_J_inv = &shared_W[p];
    // int *shared_map = (int*) &shared_J_inv[dim*dim];

    __shared__ Type shared_U[vec_shared * p * p * p],
      shared_V[vec_shared * p * p * p];
    __shared__ Type shared_N[p * p], shared_N_T[p * p], shared_W[p],
      shared_J_inv[dim * dim];
    __shared__ int shared_map[p * p * p];

#pragma unroll
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < p * p * p;
         i += blockDim.x * blockDim.y)
      {
        shared_map[i] = map[i + blockIdx.x * p * p * p];
      }

#pragma unroll
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < p * p;
         i += blockDim.x * blockDim.y)
      {
        shared_N[i]                       = N[i];
        shared_N_T[(i / p) + (i % p) * p] = N[i];
      }

    __syncthreads();

    //////////////////////////////////////////////////////////////

    // vec_shared, p^3 fastest index Shared_U = [vec_shared][p^3]
    // Interpolation combined with Extraction

    // 1st GEMM
    for (int k = threadIdx.y; k < p * p; k += blockDim.y)
      {
        Type temp[p], u[p];

#pragma unroll
        for (int i = 0; i < p; ++i)
          {
            temp[i] = 0.0;
          }

        for (int m = 0; m < p; ++m)
          {
            u[m] = U[threadIdx.x + shared_map[k + m * p * p] * vec_shared +
                     blockIdx.y * vec_shared * lenU];
            // u[m] = U[threadIdx.x + map[k + m*p*p + blockIdx.x * p*p*p] *
            // vec_shared + blockIdx.y * vec_shared*lenU];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] += shared_N_T[i + m * p] * u[m];
              }
          }

#pragma unroll
        for (int i = 0; i < p; ++i)
          {
            shared_U[threadIdx.x + i * vec_shared + k * p * vec_shared] =
              temp[i];
          }
      }

    __syncthreads();

    // 2nd GEMM
    for (int k = threadIdx.y; k < p * p; k += blockDim.y)
      {
        Type temp[p], u[p];

#pragma unroll
        for (int i = 0; i < p; ++i)
          {
            temp[i] = 0.0;
          }

        for (int m = 0; m < p; ++m)
          {
            u[m] =
              shared_U[threadIdx.x + k * vec_shared + m * p * p * vec_shared];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] += shared_N_T[i + m * p] * u[m];
              }
          }

#pragma unroll
        for (int i = 0; i < p; ++i)
          {
            shared_V[threadIdx.x + i * vec_shared + k * p * vec_shared] =
              temp[i];
          }
      }

    __syncthreads();

    // 3rd GEMM
    for (int k = threadIdx.y; k < p * p; k += blockDim.y)
      {
        Type temp[p], u[p];

#pragma unroll
        for (int i = 0; i < p; ++i)
          {
            temp[i] = 0.0;
          }

        for (int m = 0; m < p; ++m)
          {
            u[m] =
              shared_V[threadIdx.x + k * vec_shared + m * p * p * vec_shared];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] += shared_N_T[i + m * p] * u[m];
              }
          }

#pragma unroll
        for (int i = 0; i < p; ++i)
          {
            shared_U[threadIdx.x + i * vec_shared + k * p * vec_shared] =
              temp[i];
          }
      }

    __syncthreads();

#pragma unroll
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < p;
         i += blockDim.x * blockDim.y)
      {
        shared_W[i] = W[i];
      }

#pragma unroll
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < dim * dim;
         i += blockDim.x * blockDim.y)
      {
        shared_J_inv[i] = J[i + blockIdx.x * dim * dim];
      }

    __syncthreads();

#pragma unroll
    for (int k = threadIdx.y; k < p * p * p; k += blockDim.y)
      {
        int iz = k / (p * p);
        int iy = (k % (p * p)) / p;
        int ix = (k % (p * p)) % p;

        Type R = (shared_W[ix] * shared_W[iy] * shared_W[iz]) /
                 (shared_J_inv[0] * (shared_J_inv[4] * shared_J_inv[8] -
                                     shared_J_inv[5] * shared_J_inv[7]) -
                  shared_J_inv[3] * (shared_J_inv[1] * shared_J_inv[8] -
                                     shared_J_inv[2] * shared_J_inv[7]) +
                  shared_J_inv[6] * (shared_J_inv[1] * shared_J_inv[5] -
                                     shared_J_inv[2] * shared_J_inv[4]));

        shared_U[threadIdx.x + k * vec_shared] *= R;
        // shared_U[threadIdx.x + k * vec_shared] *= J[blockIdx.x] *
        // shared_W[ix] * shared_W[iy] * shared_W[iz];
      }

    __syncthreads();

    // 1st GEMM
    for (int k = threadIdx.y; k < p * p; k += blockDim.y)
      {
        Type temp[p], u[p];

#pragma unroll
        for (int i = 0; i < p; ++i)
          {
            temp[i] = 0.0;
          }

        for (int m = 0; m < p; ++m)
          {
            u[m] =
              shared_U[threadIdx.x + k * vec_shared + m * p * p * vec_shared];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] += shared_N[i + m * p] * u[m];
              }
          }

#pragma unroll
        for (int i = 0; i < p; ++i)
          {
            shared_V[threadIdx.x + i * vec_shared + k * p * vec_shared] =
              temp[i];
          }
      }

    __syncthreads();

    // 2nd GEMM
    for (int k = threadIdx.y; k < p * p; k += blockDim.y)
      {
        Type temp[p], u[p];

#pragma unroll
        for (int i = 0; i < p; ++i)
          {
            temp[i] = 0.0;
          }

        for (int m = 0; m < p; ++m)
          {
            u[m] =
              shared_V[threadIdx.x + k * vec_shared + m * p * p * vec_shared];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] += shared_N[i + m * p] * u[m];
              }
          }

#pragma unroll
        for (int i = 0; i < p; ++i)
          {
            shared_U[threadIdx.x + i * vec_shared + k * p * vec_shared] =
              temp[i];
          }
      }

    __syncthreads();

    // 3rd GEMM
    for (int k = threadIdx.y; k < p * p; k += blockDim.y)
      {
        Type temp[p], u[p];

#pragma unroll
        for (int i = 0; i < p; ++i)
          {
            temp[i] = 0.0;
          }

        for (int m = 0; m < p; ++m)
          {
            u[m] =
              shared_U[threadIdx.x + k * vec_shared + m * p * p * vec_shared];

#pragma unroll
            for (int i = 0; i < p; ++i)
              {
                temp[i] += shared_N[i + m * p] * u[m];
              }
          }

#pragma unroll
        for (int i = 0; i < p; ++i)
          {
            shared_V[threadIdx.x + i * vec_shared + k * p * vec_shared] =
              temp[i];
          }
      }

    __syncthreads();

#pragma unroll
    for (int k = threadIdx.y; k < p * p * p; k += blockDim.y)
      {
        atomicAdd(&V[threadIdx.x + shared_map[k] * vec_shared +
                     blockIdx.y * vec_shared * lenU],
                  shared_V[threadIdx.x + k * vec_shared]);
        // atomicAdd(&V[threadIdx.x + map[k + blockIdx.x * p*p*p] * vec_shared +
        // blockIdx.y * vec_shared*lenU], shared_V[threadIdx.x + k *
        // vec_shared]); if (blockIdx.x==0 && blockIdx.y ==0 && blockIdx.z==0)
        // printf("%5d\n", j + dof_index * vec_shared + blockIdx.y *
        // vec_shared*lenU);
      }
  }

  // computeAX
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::computeAX(
    distributedGPUVec<double> &Ax,
    distributedGPUVec<double> &x)
  {
    constexpr int d_nVec = 1;
    constexpr int d_dim  = 3;
    constexpr int d_p    = FEOrderElectro + 1;

    constexpr int d_vecShared =
      (d_nVec < 4 ? 1 : FEOrderElectro < 7 ? 4 : FEOrderElectro == 7 ? 5 : 1);
    constexpr int d_yThreads =
      (d_nVec < 4 ? (FEOrderElectro == 8 ? 192 : 128) :
                    FEOrderElectro < 7 ? 96 : FEOrderElectro == 7 ? 128 : 160);
    constexpr int batch = d_nVec / d_vecShared;

    dim3 blocks(d_nLocalCells, batch, 1);
    dim3 threads(d_vecShared, d_yThreads, 1);

    // cudaFuncSetSharedMemConfig(computeAXKernel,
    // cudaSharedMemBankSizeEightByte);

    // const double zero = 0.0;
    // cudaUtils::set<double> (Ax.begin(), zero, d_xLen);

    if (d_isMeanValueConstraintComputed)
      meanValueConstraintDistribute(x);

    x.updateGhostValues();

    constraintsTotalPotentialInfo.distribute(x, 1);

    // massMatrixKernel <double, d_p, d_vecShared, d_dim> <<< blocks, threads
    // >>> (Ax.begin(), x.begin(), shapeFunctionValue_ptr, weights_ptr,
    // inverseJacobian_ptr, map_ptr, d_xLen);

    computeAXKernel<double, d_p, d_vecShared, d_dim>
      <<<blocks, threads>>>(Ax.begin(),
                            x.begin(),
                            shapeFunctionValue_ptr,
                            shapeFunctionGradient_ptr,
                            weights_ptr,
                            inverseJacobian_ptr,
                            map_ptr,
                            d_xLen,
                            d_fPI);

    constraintsTotalPotentialInfo.distribute_slave_to_master(Ax, 1);

    Ax.compressAdd();

    if (d_isMeanValueConstraintComputed)
      meanValueConstraintDistributeSlaveToMaster(Ax);
  }

#include "poissonSolverProblemCUDA.inst.cu"
} // namespace dftfe
