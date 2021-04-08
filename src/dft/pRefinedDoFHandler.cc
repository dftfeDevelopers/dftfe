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

// source file for all charge calculations

//
// compute total charge using quad point values
//
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void dftClass<FEOrder, FEOrderElectro>::createpRefinedDofHandler(
  dealii::parallel::distributed::Triangulation<3> &triaObject)
{
  //
  // initialize electrostatics dofHandler and constraint matrices
  //

  d_dofHandlerPRefined.initialize(
    triaObject, dealii::FE_Q<3>(dealii::QGaussLobatto<1>(FEOrderElectro + 1)));
  d_dofHandlerPRefined.distribute_dofs(d_dofHandlerPRefined.get_fe());

  d_locallyRelevantDofsPRefined.clear();
  dealii::DoFTools::extract_locally_relevant_dofs(
    d_dofHandlerPRefined, d_locallyRelevantDofsPRefined);

  d_constraintsPRefinedOnlyHanging.clear();
  d_constraintsPRefinedOnlyHanging.reinit(d_locallyRelevantDofsPRefined);
  dealii::DoFTools::make_hanging_node_constraints(
    d_dofHandlerPRefined, d_constraintsPRefinedOnlyHanging);
  d_constraintsPRefinedOnlyHanging.close();

  d_constraintsPRefined.clear();
  d_constraintsPRefined.reinit(d_locallyRelevantDofsPRefined);
  dealii::DoFTools::make_hanging_node_constraints(d_dofHandlerPRefined,
                                                  d_constraintsPRefined);

  std::vector<std::vector<double>> unitVectorsXYZ;
  unitVectorsXYZ.resize(3);

  for (unsigned int i = 0; i < 3; ++i)
    {
      unitVectorsXYZ[i].resize(3, 0.0);
      unitVectorsXYZ[i][i] = 0.0;
    }

  std::vector<Tensor<1, 3>> offsetVectors;
  // resize offset vectors
  offsetVectors.resize(3);

  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      offsetVectors[i][j] =
        unitVectorsXYZ[i][j] - d_domainBoundingVectors[i][j];

  std::vector<dealii::GridTools::PeriodicFacePair<
    typename dealii::DoFHandler<3>::cell_iterator>>
                                    periodicity_vector2;
  const std::array<unsigned int, 3> periodic = {dftParameters::periodicX,
                                                dftParameters::periodicY,
                                                dftParameters::periodicZ};

  std::vector<int> periodicDirectionVector;
  for (unsigned int d = 0; d < 3; ++d)
    {
      if (periodic[d] == 1)
        {
          periodicDirectionVector.push_back(d);
        }
    }

  for (unsigned int i = 0;
       i < std::accumulate(periodic.begin(), periodic.end(), 0);
       ++i)
    GridTools::collect_periodic_faces(
      d_dofHandlerPRefined,
      /*b_id1*/ 2 * i + 1,
      /*b_id2*/ 2 * i + 2,
      /*direction*/ periodicDirectionVector[i],
      periodicity_vector2,
      offsetVectors[periodicDirectionVector[i]]);

  dealii::DoFTools::make_periodicity_constraints<dealii::DoFHandler<3>>(
    periodicity_vector2, d_constraintsPRefined);

  d_constraintsPRefined.close();

  //
  // initialize rho nodal dofHandler and constraint matrices
  //

  d_dofHandlerRhoNodal.initialize(
    triaObject,
    dealii::FE_Q<3>(dealii::QGaussLobatto<1>(
      C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>() + 1)));
  d_dofHandlerRhoNodal.distribute_dofs(d_dofHandlerRhoNodal.get_fe());

  d_locallyRelevantDofsRhoNodal.clear();
  dealii::DoFTools::extract_locally_relevant_dofs(
    d_dofHandlerRhoNodal, d_locallyRelevantDofsRhoNodal);

  d_constraintsRhoNodalOnlyHanging.clear();
  d_constraintsRhoNodalOnlyHanging.reinit(d_locallyRelevantDofsRhoNodal);
  dealii::DoFTools::make_hanging_node_constraints(
    d_dofHandlerRhoNodal, d_constraintsRhoNodalOnlyHanging);
  d_constraintsRhoNodalOnlyHanging.close();

  d_constraintsRhoNodal.clear();
  d_constraintsRhoNodal.reinit(d_locallyRelevantDofsRhoNodal);
  dealii::DoFTools::make_hanging_node_constraints(d_dofHandlerRhoNodal,
                                                  d_constraintsRhoNodal);

  std::vector<dealii::GridTools::PeriodicFacePair<
    typename dealii::DoFHandler<3>::cell_iterator>>
    periodicity_vector_rhonodal;
  for (unsigned int i = 0;
       i < std::accumulate(periodic.begin(), periodic.end(), 0);
       ++i)
    GridTools::collect_periodic_faces(
      d_dofHandlerRhoNodal,
      /*b_id1*/ 2 * i + 1,
      /*b_id2*/ 2 * i + 2,
      /*direction*/ periodicDirectionVector[i],
      periodicity_vector_rhonodal,
      offsetVectors[periodicDirectionVector[i]]);

  dealii::DoFTools::make_periodicity_constraints<dealii::DoFHandler<3>>(
    periodicity_vector_rhonodal, d_constraintsRhoNodal);

  d_constraintsRhoNodal.close();

  if (dftParameters::createConstraintsFromSerialDofhandler)
    {
      vectorTools::createParallelConstraintMatrixFromSerial(
        d_mesh.getSerialMeshUnmoved(),
        d_dofHandlerPRefined,
        mpi_communicator,
        d_domainBoundingVectors,
        d_constraintsPRefined,
        d_constraintsPRefinedOnlyHanging);

      vectorTools::createParallelConstraintMatrixFromSerial(
        d_mesh.getSerialMeshUnmoved(),
        d_dofHandlerRhoNodal,
        mpi_communicator,
        d_domainBoundingVectors,
        d_constraintsRhoNodal,
        d_constraintsRhoNodalOnlyHanging);
    }
}


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::initpRefinedObjects(
  const bool meshOnlyDeformed,
  const bool vselfPerturbationUpdateForStress)
{
  d_dofHandlerPRefined.distribute_dofs(d_dofHandlerPRefined.get_fe());
  d_dofHandlerRhoNodal.distribute_dofs(d_dofHandlerRhoNodal.get_fe());

  d_supportPointsPRefined.clear();
  DoFTools::map_dofs_to_support_points(MappingQ1<3, 3>(),
                                       d_dofHandlerPRefined,
                                       d_supportPointsPRefined);

  // matrix free data structure
  typename dealii::MatrixFree<3>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    dealii::MatrixFree<3>::AdditionalData::partition_partition;
  if (dftParameters::isCellStress)
    additional_data.mapping_update_flags = update_values | update_gradients |
                                           update_JxW_values |
                                           update_quadrature_points;
  else
    additional_data.mapping_update_flags =
      update_values | update_gradients | update_JxW_values;

  // clear existing constraints matrix vector
  d_constraintsVectorElectro.clear();
  d_constraintsVectorElectro.push_back(&d_constraintsRhoNodal);
  d_densityDofHandlerIndexElectro = d_constraintsVectorElectro.size() - 1;

  d_constraintsVectorElectro.push_back(&d_constraintsRhoNodalOnlyHanging);
  d_nonPeriodicDensityDofHandlerIndexElectro =
    d_constraintsVectorElectro.size() - 1;

  // Zero Dirichlet BC constraints on the boundary of the domain
  // used for Helmholtz solve
  //
  d_constraintsForHelmholtzRhoNodal.clear();
  d_constraintsForHelmholtzRhoNodal.reinit(d_locallyRelevantDofsRhoNodal);

  applyHomogeneousDirichletBC(d_dofHandlerRhoNodal,
                              d_constraintsRhoNodalOnlyHanging,
                              d_constraintsForHelmholtzRhoNodal);
  d_constraintsForHelmholtzRhoNodal.close();
  d_constraintsForHelmholtzRhoNodal.merge(
    d_constraintsRhoNodal,
    dealii::AffineConstraints<
      double>::MergeConflictBehavior::right_object_wins);
  d_constraintsForHelmholtzRhoNodal.close();
  d_constraintsVectorElectro.push_back(&d_constraintsForHelmholtzRhoNodal);
  d_helmholtzDofHandlerIndexElectro = d_constraintsVectorElectro.size() - 1;

  d_constraintsVectorElectro.push_back(&d_constraintsPRefined);
  d_baseDofHandlerIndexElectro = d_constraintsVectorElectro.size() - 1;

  // Zero Dirichlet BC constraints on the boundary of the domain
  // used for computing total electrostatic potential using Poisson problem
  // with (rho+b) as the rhs
  //
  d_constraintsForTotalPotentialElectro.clear();
  d_constraintsForTotalPotentialElectro.reinit(d_locallyRelevantDofsPRefined);

  if (dftParameters::pinnedNodeForPBC)
    locatePeriodicPinnedNodes(d_dofHandlerPRefined,
                              d_constraintsPRefined,
                              d_constraintsForTotalPotentialElectro);
  applyHomogeneousDirichletBC(d_dofHandlerPRefined,
                              d_constraintsPRefinedOnlyHanging,
                              d_constraintsForTotalPotentialElectro);
  d_constraintsForTotalPotentialElectro.close();
  d_constraintsForTotalPotentialElectro.merge(
    d_constraintsPRefined,
    dealii::AffineConstraints<
      double>::MergeConflictBehavior::right_object_wins);
  d_constraintsForTotalPotentialElectro.close();

  d_constraintsVectorElectro.push_back(&d_constraintsForTotalPotentialElectro);
  d_phiTotDofHandlerIndexElectro = d_constraintsVectorElectro.size() - 1;

  d_binsStartDofHandlerIndexElectro = d_constraintsVectorElectro.size();

  double init_bins;
  MPI_Barrier(MPI_COMM_WORLD);
  init_bins = MPI_Wtime();
  //
  // Dirichlet BC constraints on the boundary of fictitious ball
  // used for computing self-potential (Vself) using Poisson problem
  // with atoms belonging to a given bin
  //
  if (meshOnlyDeformed)
    {
      computing_timer.enter_section("Update atom bins bc");
      d_vselfBinsManager.updateBinsBc(d_constraintsVectorElectro,
                                      d_constraintsPRefinedOnlyHanging,
                                      d_dofHandlerPRefined,
                                      d_constraintsPRefined,
                                      atomLocations,
                                      d_imagePositionsTrunc,
                                      d_imageIdsTrunc,
                                      d_imageChargesTrunc,
                                      vselfPerturbationUpdateForStress);
      computing_timer.exit_section("Update atom bins bc");
    }
  else
    {
      computing_timer.enter_section("Create atom bins");
      d_vselfBinsManager.createAtomBins(d_constraintsVectorElectro,
                                        d_constraintsPRefinedOnlyHanging,
                                        d_dofHandlerPRefined,
                                        d_constraintsPRefined,
                                        atomLocations,
                                        d_imagePositionsTrunc,
                                        d_imageIdsTrunc,
                                        d_imageChargesTrunc,
                                        dftParameters::radiusAtomBall);

      d_netFloatingDispSinceLastBinsUpdate.clear();
      d_netFloatingDispSinceLastBinsUpdate.resize(atomLocations.size() * 3,
                                                  0.0);
      computing_timer.exit_section("Create atom bins");
    }

  MPI_Barrier(MPI_COMM_WORLD);
  init_bins = MPI_Wtime() - init_bins;
  if (dftParameters::verbosity >= 4)
    pcout
      << "updateAtomPositionsAndMoveMesh: initBoundaryConditions: Time taken for bins update: "
      << init_bins << std::endl;

  d_constraintsVectorElectro.push_back(&d_constraintsPRefinedOnlyHanging);
  d_phiExtDofHandlerIndexElectro = d_constraintsVectorElectro.size() - 1;

  if (dftParameters::constraintsParallelCheck)
    {
      IndexSet locally_active_dofs_debug;
      DoFTools::extract_locally_active_dofs(d_dofHandlerPRefined,
                                            locally_active_dofs_debug);

      const std::vector<IndexSet> &locally_owned_dofs_debug =
        d_dofHandlerPRefined.locally_owned_dofs_per_processor();

      AssertThrow(
        d_constraintsForTotalPotentialElectro.is_consistent_in_parallel(
          locally_owned_dofs_debug,
          locally_active_dofs_debug,
          mpi_communicator),
        ExcMessage(
          "DFT-FE Error: Constraints are not consistent in parallel."));
    }

  // Fill dofHandler vector
  std::vector<const dealii::DoFHandler<3> *> matrixFreeDofHandlerVectorInput;
  matrixFreeDofHandlerVectorInput.push_back(&d_dofHandlerRhoNodal);
  matrixFreeDofHandlerVectorInput.push_back(&d_dofHandlerRhoNodal);
  matrixFreeDofHandlerVectorInput.push_back(&d_dofHandlerRhoNodal);

  for (unsigned int i = 3; i < d_constraintsVectorElectro.size(); ++i)
    matrixFreeDofHandlerVectorInput.push_back(&d_dofHandlerPRefined);

  forcePtr->initMoved(matrixFreeDofHandlerVectorInput,
                      d_constraintsVectorElectro,
                      true);
  d_forceDofHandlerIndexElectro = d_constraintsVectorElectro.size() - 1;


  std::vector<Quadrature<1>> quadratureVector;
  quadratureVector.push_back(
    QGauss<1>(C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>()));
  quadratureVector.push_back(QIterated<1>(QGauss<1>(C_num1DQuadLPSP<FEOrder>()),
                                          C_numCopies1DQuadLPSP()));
  if (dftParameters::isCellStress)
    quadratureVector.push_back(
      QIterated<1>(QGauss<1>(C_num1DQuadSmearedChargeStress()),
                   C_numCopies1DQuadSmearedChargeStress()));
  else if (dftParameters::meshSizeOuterBall > 2.2)
    quadratureVector.push_back(
      QIterated<1>(QGauss<1>(C_num1DQuadSmearedChargeHigh()),
                   C_numCopies1DQuadSmearedChargeHigh()));
  else
    quadratureVector.push_back(
      QIterated<1>(QGauss<1>(C_num1DQuadSmearedCharge()),
                   C_numCopies1DQuadSmearedCharge()));
  quadratureVector.push_back(QGauss<1>(FEOrderElectro + 1));


  d_densityQuadratureIdElectro       = 0;
  d_lpspQuadratureIdElectro          = 1;
  d_smearedChargeQuadratureIdElectro = 2;
  d_phiTotAXQuadratureIdElectro      = 3;

  d_matrixFreeDataPRefined.reinit(matrixFreeDofHandlerVectorInput,
                                  d_constraintsVectorElectro,
                                  quadratureVector,
                                  additional_data);

  //
  // locate atom core nodes
  //
  if (!dftParameters::floatingNuclearCharges)
    locateAtomCoreNodes(d_dofHandlerPRefined, d_atomNodeIdToChargeMap);

  //
  // create duplicate constraints object with flattened maps for faster access
  //
  d_constraintsRhoNodalInfo.initialize(
    d_matrixFreeDataPRefined.get_vector_partitioner(
      d_densityDofHandlerIndexElectro),
    d_constraintsRhoNodal);
}
