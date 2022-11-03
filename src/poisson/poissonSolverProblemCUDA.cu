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
    d_xLocalDof                        = d_xDevice.locallyOwnedDofsSize();
    d_xLen = d_xDevice.locallyOwnedDofsSize() + d_xDevice.ghostFlattenedSize();

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
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::copyXfromDeviceToHost()
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

    if (d_isMeanValueConstraintComputed)
      meanValueConstraintDistribute(d_xDevice);
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
      cudaUtils::dot(d_meanValueConstraintGPUVec.begin(),
                     vec.begin(),
                     d_xLocalDof,
                     mpi_communicator,
                     *d_cublasHandlePtr);

    if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
        d_meanValueConstraintProcId)
      cudaUtils::set(vec.begin() + d_meanValueConstraintNodeIdLocal,
                     constrainedNodeValue,
                     1);
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

    cudaUtils::add(vec.begin(),
                   d_meanValueConstraintGPUVec.begin(),
                   constrainedNodeValue,
                   d_xLocalDof,
                   *d_cublasHandlePtr);

    // meanValueConstraintSetZero
    if (d_isMeanValueConstraintComputed)
      if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==
          d_meanValueConstraintProcId)
        cudaUtils::set<double>(vec.begin() + d_meanValueConstraintNodeIdLocal,
                               0,
                               1);
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
    cudaUtils::copyHostVecToCUDAVec<double>(d_meanValueConstraintVec.begin(),
                                            d_meanValueConstraintGPUVec.begin(),
                                            d_xLocalDof);
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
    d_diagonalAdevice.reinit(d_diagonalA.get_partitioner(), 1);
    cudaUtils::copyHostVecToCUDAVec<double>(d_diagonalA.begin(),
                                            d_diagonalAdevice.begin(),
                                            d_xLocalDof);
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::setX()
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  distributedGPUVec<double> &
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::getPreconditioner()
  {
    return d_diagonalAdevice;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::setupconstraints()
  {
    constraintsTotalPotentialInfo.initialize(
      d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent),
      *d_constraintMatrixPtr);
    constraintsTotalPotentialInfo.precomputeMaps(
      d_matrixFreeDataPtr->get_vector_partitioner(d_matrixFreeVectorComponent),
      d_xPtr->get_partitioner(),
      1);
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
    // PT(q*p), D(q*q), P(p*q), DT(q*q)

    extern __shared__ Type SMem[];

    Type *sharedX   = SMem;
    Type *sharedY   = &sharedX[M * K];
    Type *sharedZ   = &sharedY[M * K];
    Type *sharedT   = &sharedZ[M * K];
    Type *sharedPT  = &sharedT[M * K];
    Type *sharedD   = &sharedPT[N * K];
    Type *sharedP   = &sharedD[N * N];
    Type *sharedDT  = &sharedP[K * N];
    Type *sharedJ   = &sharedDT[N * N];
    int * sharedMap = (int *)&sharedJ[dim * dim];

    const int mapShift = blockIdx.x * M * K;

    // Copy Map to shared memory
#pragma unroll
    for (int i = threadIdx.x; i < M * K; i += blockDim.x)
      sharedMap[i] = map[i + mapShift];

      // Copy Shape Function Values and Gradients to shared memory
#pragma unroll
    for (int i = threadIdx.x; i < 2 * N * (K + N); i += blockDim.x)
      sharedPT[i] = P[i];

    __syncthreads();

    //////////////////////////////////////////////////////////////
    // First index is the fastest
    // Interpolation combined with Extraction
    // T -> PPPU
    // X -> TD1
    // Y -> TD2
    // Z -> TD3

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
            u[k] = U[sharedMap[i + k * M]];

#pragma unroll
            for (int j = 0; j < N; j++)
              x[j] += sharedPT[j + k * N] * u[k];
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
              y[j] += sharedPT[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedY[a + j * K + b * K * N] = y[j];
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
              x[j] += sharedPT[j + k * N] * y[k];
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
            x[k] = sharedX[a + k * N + b * N * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              z[j] += sharedD[j + k * N] * x[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedZ[a + j * N + b * N * N] = z[j];
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

      // Copy Jacobian Action to shared memory
#pragma unroll
    for (int i = threadIdx.x; i < dim * dim; i += blockDim.x)
      sharedJ[i] = J[i + blockIdx.x * dim * dim];

    __syncthreads();

    // Gemm with Jacobian Action
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
    // X -> TDT1
    // Y -> TDT2
    // Z -> TDT3
    // T -> PPPU

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
            z[k] = sharedZ[a + k * N + b * N * N];

#pragma unroll
            for (int j = 0; j < N; j++)
              y[j] += sharedDT[j + k * N] * z[k];
          }

#pragma unroll
        for (int j = 0; j < N; j++)
          sharedX[a + j * N + b * N * N] += y[j];
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
              y[j] += sharedP[j + k * K] * x[k];
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
            y[k] = sharedY[a + k * N + b * N * N];

#pragma unroll
            for (int j = 0; j < K; j++)
              x[j] += sharedP[j + k * K] * y[k];
          }

#pragma unroll
        for (int j = 0; j < K; j++)
          sharedX[a + j * N + b * N * K] = x[j];
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
              y[j] += sharedP[j + k * K] * x[k];
          }

#pragma unroll
        for (int j = 0; j < K; j++)
          atomicAdd(&V[sharedMap[j + i * K]], y[j]);
      }
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::setupMatrixFree()
  {
    constexpr int    p              = FEOrderElectro + 1;
    constexpr int    dim            = 3;
    constexpr int    q              = p;
    constexpr double coeffLaplacian = 1.0 / (4.0 * M_PI);

    // shape info helps to obtain reference cell basis function and lex
    // numbering
    const dealii::DoFHandler<3> &dofHandler =
      d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);
    const int dofs_per_cell = dofHandler.get_fe().dofs_per_cell;

    dealii::internal::MatrixFreeFunctions::ShapeInfo<double> shapeInfo;

    const dealii::Quadrature<3> &quadrature =
      d_matrixFreeDataPtr->get_quadrature(d_matrixFreeQuadratureComponentAX);

    int numQuadPoints = std::cbrt(quadrature.size());

    dealii::QGauss<1> quad(numQuadPoints);
    shapeInfo.reinit(quad, dofHandler.get_fe());
    std::vector<unsigned int> lexMap3D = shapeInfo.lexicographic_numbering;

    const auto shapeValue = shapeInfo.data.front().shape_values;
    const auto shapeGradquad =
      shapeInfo.data.front().shape_gradients_collocation;

    dealii::FE_Q<1> feCell1D(FEOrderElectro);
    shapeInfo.reinit(quad, feCell1D);
    std::vector<unsigned int> lexMap1D = shapeInfo.lexicographic_numbering;
    std::vector<double>       quadWeights(q);

    for (int j = 0; j < q; j++)
      quadWeights[j] = quad.weight(lexMap1D[j]);

    thrust::host_vector<double> spVG(2 * p * q + 2 * q * q);

    for (int i = 0; i < p; i++)
      for (int j = 0; j < q; j++)
        {
          // PT(q*p), DT(q*q), P(p*q), D(q*q)
          double value =
            shapeValue[lexMap1D[j] + i * p] * std::sqrt(quadWeights[j]);
          double grad = shapeGradquad[lexMap1D[j] + lexMap1D[i] * p] *
                        std::sqrt(quadWeights[j]) / std::sqrt(quadWeights[i]);

          spVG[j + i * q]                     = value;
          spVG[j + i * q + q * p]             = grad;
          spVG[i + j * p + q * p + q * q]     = value;
          spVG[i + j * q + 2 * q * p + q * q] = grad;
        }

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
    const int           qPoints = quadrature_formula.size();

    dealii::FEValues<dim> fe_values(dofHandler.get_fe(),
                                    quadrature_formula,
                                    dealii::update_inverse_jacobians |
                                      dealii::update_values |
                                      dealii::update_gradients |
                                      dealii::update_JxW_values |
                                      dealii::update_quadrature_points);

    std::vector<dealii::DerivativeForm<1, dim, dim>> inv_jacobians_tensor;
    std::vector<double> detJacobian(d_nLocalCells * qPoints),
      invJac(d_nLocalCells * dim * dim);
    thrust::host_vector<double> jacobianAction(d_nLocalCells * dim * dim);

    cellIdx = 0;
    for (const auto &cell : dofHandler.active_cell_iterators())
      {
        if (cell->is_locally_owned())
          {
            fe_values.reinit(cell);
            inv_jacobians_tensor = fe_values.get_inverse_jacobians();

            for (int d = 0; d < dim; d++)
              for (int e = 0; e < dim; e++)
                invJac[e + d * dim + cellIdx * dim * dim] =
                  inv_jacobians_tensor[0][d][e];

            for (int i = 0; i < qPoints; i++)
              detJacobian[i + cellIdx * qPoints] =
                fe_values.JxW(lexMap3D[i]) /
                quadrature_formula.weight(lexMap3D[i]) * coeffLaplacian;

            cellIdx++;
          }
      }

    for (int cellIdx = 0; cellIdx < d_nLocalCells; cellIdx++)
      for (int d = 0; d < dim; d++)
        for (int e = 0; e < dim; e++)
          for (int f = 0; f < dim; f++)
            jacobianAction[e + d * dim + cellIdx * dim * dim] +=
              invJac[f + d * dim + cellIdx * dim * dim] *
              invJac[e + f * dim + cellIdx * dim * dim] *
              detJacobian[cellIdx * qPoints];

    // Construct the device vectors
    d_shapeFunctionAll = spVG;
    d_jacobianAction   = jacobianAction;
    d_map              = map;

    shapeFunctionAllPtr = thrust::raw_pointer_cast(d_shapeFunctionAll.data());
    jacobianActionPtr   = thrust::raw_pointer_cast(d_jacobianAction.data());
    mapPtr              = thrust::raw_pointer_cast(d_map.data());

    constexpr size_t smem =
      (4 * q * q * q + 2 * p * q + 2 * q * q + dim * dim) * sizeof(double) +
      p * p * p * sizeof(int);

    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    cudaFuncSetAttribute(computeAXKernel<double, p * p, q, p, dim>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         smem);
  }


  // computeAX
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  poissonSolverProblemCUDA<FEOrder, FEOrderElectro>::computeAX(
    distributedGPUVec<double> &Ax,
    distributedGPUVec<double> &x)
  {
    constexpr int    dim     = 3;
    constexpr int    p       = FEOrderElectro + 1;
    constexpr int    q       = p;
    constexpr int    threads = (FEOrderElectro == 8 ? 192 : 128);
    const int        blocks  = d_nLocalCells;
    constexpr size_t smem =
      (4 * q * q * q + 2 * p * q + 2 * q * q + dim * dim) * sizeof(double) +
      p * p * p * sizeof(int);

    cudaUtils::set<double>(Ax.begin(), 0, d_xLen);

    if (d_isMeanValueConstraintComputed)
      meanValueConstraintDistribute(x);

    x.updateGhostValues();

    constraintsTotalPotentialInfo.distribute(x, 1);

    computeAXKernel<double, p * p, q, p, dim><<<blocks, threads, smem>>>(
      Ax.begin(), x.begin(), shapeFunctionAllPtr, jacobianActionPtr, mapPtr);

    constraintsTotalPotentialInfo.distribute_slave_to_master(Ax, 1);

    Ax.compressAdd();

    if (d_isMeanValueConstraintComputed)
      meanValueConstraintDistributeSlaveToMaster(Ax);
  }

#include "poissonSolverProblemCUDA.inst.cu"
} // namespace dftfe
