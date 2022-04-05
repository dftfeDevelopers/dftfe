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
// @author Phani Motamarri
//


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::computeElectrostaticEnergyHRefined(
#ifdef DFTFE_WITH_GPU
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>
    &kohnShamDFTEigenOperator
#endif
)
{
  computing_timer.enter_subsection("h refinement electrostatics");
  computingTimerStandard.enter_subsection("h refinement electrostatics");
  if (dftParameters::verbosity >= 1)
    pcout
      << std::endl
      << "-----------------Re computing electrostatics on h globally refined mesh--------------"
      << std::endl;



  //
  // access quadrature object
  //
  dealii::QGauss<3> quadrature(
    C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>());
  const unsigned int n_q_points = quadrature.size();

  //
  // project and create a nodal field of the same mesh from the quadrature data
  // (L2 projection from quad points to nodes)
  //
  distributedCPUVec<double> rhoNodalFieldCoarse;
  d_matrixFreeDataPRefined.initialize_dof_vector(rhoNodalFieldCoarse,
                                                 d_baseDofHandlerIndexElectro);
  rhoNodalFieldCoarse = 0.0;

  //
  // create a lambda function for L2 projection of quadrature electron-density
  // to nodal electron density
  //
  std::function<
    double(const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
           const unsigned int                                          q)>
    funcRho =
      [&](const typename dealii::DoFHandler<3>::active_cell_iterator &cell,
          const unsigned int                                          q) {
        return (*rhoOutValues).find(cell->id())->second[q];
      };

  dealii::VectorTools::project<3, distributedCPUVec<double>>(
    dealii::MappingQ1<3, 3>(),
    d_matrixFreeDataPRefined.get_dof_handler(d_baseDofHandlerIndexElectro),
    d_constraintsPRefined,
    quadrature,
    funcRho,
    rhoNodalFieldCoarse);

  rhoNodalFieldCoarse.update_ghost_values();
  d_constraintsPRefined.distribute(rhoNodalFieldCoarse);
  rhoNodalFieldCoarse.update_ghost_values();

  //
  // compute the total charge using rho nodal field for debugging purposes
  //
  if (dftParameters::verbosity >= 4)
    {
      const double integralRhoValue =
        totalCharge(d_matrixFreeDataPRefined.get_dof_handler(
                      d_baseDofHandlerIndexElectro),
                    rhoNodalFieldCoarse);

      pcout
        << "Value of total charge on coarse mesh using L2 projected nodal field: "
        << integralRhoValue << std::endl;
    }


  //
  // subdivide the existing mesh and project electron-density onto the new mesh
  //

  //
  // initialize the new dofHandler to refine and do a solution transfer
  //
  dealii::parallel::distributed::Triangulation<3> &electrostaticsTriaRho =
    d_mesh.getElectrostaticsMeshRho();

  dealii::DoFHandler<3> dofHandlerHRefined;
  dofHandlerHRefined.initialize(electrostaticsTriaRho,
                                dealii::FE_Q<3>(dealii::QGaussLobatto<1>(
                                  FEOrderElectro + 1)));
  dofHandlerHRefined.distribute_dofs(dofHandlerHRefined.get_fe());

  //
  // create a solution transfer object and prepare for refinement and solution
  // transfer
  //
  parallel::distributed::SolutionTransfer<3, distributedCPUVec<double>>
    solTrans(dofHandlerHRefined);
  electrostaticsTriaRho.set_all_refine_flags();
  electrostaticsTriaRho.prepare_coarsening_and_refinement();

  std::vector<const distributedCPUVec<double> *> vecAllIn(1);
  vecAllIn[0] = &rhoNodalFieldCoarse;


  solTrans.prepare_for_coarsening_and_refinement(vecAllIn);


  electrostaticsTriaRho.execute_coarsening_and_refinement();

  dofHandlerHRefined.distribute_dofs(dofHandlerHRefined.get_fe());

  //
  // print refined mesh details
  //
  if (dftParameters::verbosity >= 2)
    {
      pcout << std::endl
            << "Finite element mesh information after subdividing the mesh"
            << std::endl;
      pcout << "-------------------------------------------------" << std::endl;
      pcout << "number of elements: "
            << dofHandlerHRefined.get_triangulation().n_global_active_cells()
            << std::endl
            << "number of degrees of freedom: " << dofHandlerHRefined.n_dofs()
            << std::endl;
    }

  dealii::IndexSet locallyRelevantDofs;
  dealii::DoFTools::extract_locally_relevant_dofs(dofHandlerHRefined,
                                                  locallyRelevantDofs);

  IndexSet ghost_indices = locallyRelevantDofs;
  ghost_indices.subtract_set(dofHandlerHRefined.locally_owned_dofs());


  dealii::AffineConstraints<double> onlyHangingNodeConstraints;
  onlyHangingNodeConstraints.reinit(locallyRelevantDofs);
  dealii::DoFTools::make_hanging_node_constraints(dofHandlerHRefined,
                                                  onlyHangingNodeConstraints);
  onlyHangingNodeConstraints.close();

  dealii::AffineConstraints<double> constraintsHRefined;
  constraintsHRefined.reinit(locallyRelevantDofs);
  dealii::DoFTools::make_hanging_node_constraints(dofHandlerHRefined,
                                                  constraintsHRefined);
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
      dofHandlerHRefined,
      /*b_id1*/ 2 * i + 1,
      /*b_id2*/ 2 * i + 2,
      /*direction*/ periodicDirectionVector[i],
      periodicity_vector2,
      offsetVectors[periodicDirectionVector[i]]);

  dealii::DoFTools::make_periodicity_constraints<dealii::DoFHandler<3>>(
    periodicity_vector2, constraintsHRefined);
  constraintsHRefined.close();


  //
  // create rho nodal field on the refined mesh and conduct solution transfer
  //
  d_rhoNodalFieldRefined =
    distributedCPUVec<double>(dofHandlerHRefined.locally_owned_dofs(),
                              ghost_indices,
                              mpi_communicator);
  d_rhoNodalFieldRefined.zero_out_ghosts();

  std::vector<distributedCPUVec<double> *> vecAllOut(1);
  vecAllOut[0] = &d_rhoNodalFieldRefined;

  solTrans.interpolate(vecAllOut);


  d_rhoNodalFieldRefined.update_ghost_values();
  constraintsHRefined.distribute(d_rhoNodalFieldRefined);
  d_rhoNodalFieldRefined.update_ghost_values();

  dealii::parallel::distributed::Triangulation<3> &electrostaticsTriaDisp =
    d_mesh.getElectrostaticsMeshDisp();
  if (!dftParameters::floatingNuclearCharges)
    {
      //
      // move the refined mesh so that it forms exact subdivison of coarse moved
      // mesh
      //


      //
      // create guassian Move object
      //
      if (d_autoMesh == 1)
        moveMeshToAtoms(electrostaticsTriaDisp,
                        d_mesh.getSerialMeshElectrostatics(),
                        true,
                        true);
      else
        {
          //
          // move electrostatics mesh
          //

          d_gaussianMovePar.init(electrostaticsTriaDisp,
                                 d_mesh.getSerialMeshElectrostatics(),
                                 d_domainBoundingVectors);

          d_gaussianMovePar.moveMeshTwoLevelElectro();
        }
    }

  if (!dftParameters::floatingNuclearCharges)
    d_mesh.resetMesh(electrostaticsTriaDisp, electrostaticsTriaRho);

  dofHandlerHRefined.distribute_dofs(dofHandlerHRefined.get_fe());

  //
  // fill in quadrature values of the field on the refined mesh and compute
  // total charge
  //
  std::map<dealii::CellId, std::vector<double>> rhoOutHRefinedQuadValues;
  const double integralRhoValue = totalCharge(dofHandlerHRefined,
                                              d_rhoNodalFieldRefined,
                                              rhoOutHRefinedQuadValues);
  //
  // fill in grad rho at quadrature values of the field on the refined mesh
  //
  std::map<dealii::CellId, std::vector<double>> gradRhoOutHRefinedQuadValues;
  std::map<dealii::CellId, std::vector<double>> rhoOutValuesLpspQuadHRefined;
  std::map<dealii::CellId, std::vector<double>>
    gradRhoOutValuesLpspQuadHRefined;

  FEValues<3>                       fe_values(dofHandlerHRefined.get_fe(),
                        quadrature,
                        update_values | update_gradients);
  std::vector<Tensor<1, 3, double>> tempGradRho(n_q_points);

  DoFHandler<3>::active_cell_iterator cell = dofHandlerHRefined.begin_active(),
                                      endc = dofHandlerHRefined.end();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      {
        fe_values.reinit(cell);
        fe_values.get_function_gradients(d_rhoNodalFieldRefined, tempGradRho);

        gradRhoOutHRefinedQuadValues[cell->id()].resize(3 * n_q_points);
        std::vector<double> &temp = gradRhoOutHRefinedQuadValues[cell->id()];
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point)
          {
            temp[3 * q_point]     = tempGradRho[q_point][0];
            temp[3 * q_point + 1] = tempGradRho[q_point][1];
            temp[3 * q_point + 2] = tempGradRho[q_point][2];
          }
      }

  dealii::QIterated<3> quadraturelpsp(QGauss<1>(C_num1DQuadLPSP<FEOrder>()),
                                      C_numCopies1DQuadLPSP());
  const unsigned int   n_q_points_lpsp = quadraturelpsp.size();
  FEValues<3>          fe_values_lpspquad(dofHandlerHRefined.get_fe(),
                                 quadraturelpsp,
                                 update_values | update_gradients);
  std::vector<double>  rholpsp(n_q_points_lpsp);
  std::vector<Tensor<1, 3, double>> tempGradRhoPsp(n_q_points_lpsp);

  cell = dofHandlerHRefined.begin_active();
  for (; cell != endc; ++cell)
    if (cell->is_locally_owned())
      {
        fe_values_lpspquad.reinit(cell);
        fe_values_lpspquad.get_function_values(d_rhoNodalFieldRefined, rholpsp);

        fe_values_lpspquad.get_function_gradients(d_rhoNodalFieldRefined,
                                                  tempGradRhoPsp);

        std::vector<double> &temp = rhoOutValuesLpspQuadHRefined[cell->id()];
        temp.resize(n_q_points_lpsp);

        std::vector<double> &tempGrad =
          gradRhoOutValuesLpspQuadHRefined[cell->id()];
        tempGrad.resize(3 * n_q_points_lpsp);
        for (unsigned int q_point = 0; q_point < n_q_points_lpsp; ++q_point)
          {
            temp[q_point]             = rholpsp[q_point];
            tempGrad[3 * q_point]     = tempGradRhoPsp[q_point][0];
            tempGrad[3 * q_point + 1] = tempGradRhoPsp[q_point][1];
            tempGrad[3 * q_point + 2] = tempGradRhoPsp[q_point][2];
          }
      }

  //
  // compute total charge using rhoNodalRefined field
  //
  if (dftParameters::verbosity >= 4)
    {
      pcout
        << "Value of total charge computed on moved subdivided mesh after solution transfer: "
        << integralRhoValue << std::endl;
    }


  // matrix free data structure
  typename dealii::MatrixFree<3>::AdditionalData additional_data;
  additional_data.tasks_parallel_scheme =
    dealii::MatrixFree<3>::AdditionalData::partition_partition;

  // Zero Dirichlet BC constraints on the boundary of the domain
  // used for computing total electrostatic potential using Poisson problem
  // with (rho+b) as the rhs
  dealii::AffineConstraints<double> constraintsForTotalPotential;
  constraintsForTotalPotential.reinit(locallyRelevantDofs);

  if (dftParameters::pinnedNodeForPBC)
    locatePeriodicPinnedNodes(dofHandlerHRefined,
                              constraintsHRefined,
                              constraintsForTotalPotential);
  applyHomogeneousDirichletBC(dofHandlerHRefined,
                              onlyHangingNodeConstraints,
                              constraintsForTotalPotential);
  constraintsForTotalPotential.close();

  constraintsForTotalPotential.merge(
    constraintsHRefined,
    AffineConstraints<double>::MergeConflictBehavior::right_object_wins);
  constraintsForTotalPotential.close();

  // clear existing constraints matrix vector
  std::vector<const dealii::AffineConstraints<double> *>
    matrixFreeConstraintsInputVector;

  matrixFreeConstraintsInputVector.push_back(&constraintsHRefined);

  matrixFreeConstraintsInputVector.push_back(&constraintsForTotalPotential);

  // Dirichlet BC constraints on the boundary of fictitious ball
  // used for computing self-potential (Vself) using Poisson problem
  // with atoms belonging to a given bin

  vselfBinsManager<FEOrder, FEOrderElectro> vselfBinsManagerHRefined(
    d_mpiCommParent, mpi_communicator);
  vselfBinsManagerHRefined.createAtomBins(
    matrixFreeConstraintsInputVector,
    onlyHangingNodeConstraints,
    dofHandlerHRefined,
    constraintsHRefined,
    atomLocations,
    d_imagePositionsTrunc,
    d_imageIdsTrunc,
    d_imageChargesTrunc,
    d_vselfBinsManager.getStoredAdaptiveBallRadius());

  if (dftParameters::constraintsParallelCheck)
    {
      IndexSet locally_active_dofs_debug;
      DoFTools::extract_locally_active_dofs(dofHandlerHRefined,
                                            locally_active_dofs_debug);

      const std::vector<IndexSet> &locally_owned_dofs_debug =
        Utilities::MPI::all_gather(mpi_communicator,
                                   dofHandlerHRefined.locally_owned_dofs());

      AssertThrow(
        constraintsHRefined.is_consistent_in_parallel(locally_owned_dofs_debug,
                                                      locally_active_dofs_debug,
                                                      mpi_communicator),
        ExcMessage(
          "DFT-FE Error: Constraints are not consistent in parallel. This is because of a known issue in the deal.ii library, which will be fixed soon. Currently, please set H REFINED ELECTROSTATICS to false."));

      AssertThrow(
        constraintsForTotalPotential.is_consistent_in_parallel(
          locally_owned_dofs_debug,
          locally_active_dofs_debug,
          mpi_communicator),
        ExcMessage(
          "DFT-FE Error: Constraints are not consistent in parallel. This is because of a known issue in the deal.ii library, which will be fixed soon. Currently, please set H REFINED ELECTROSTATICS to false."));

      for (unsigned int i = 2; i < matrixFreeConstraintsInputVector.size(); i++)
        AssertThrow(
          matrixFreeConstraintsInputVector[i]->is_consistent_in_parallel(
            locally_owned_dofs_debug,
            locally_active_dofs_debug,
            mpi_communicator),
          ExcMessage(
            "DFT-FE Error: Constraints are not consistent in parallel. This is because of a known issue in the deal.ii library, which will be fixed soon. Currently, please set H REFINED ELECTROSTATICS to false."));
    }

  std::vector<const dealii::DoFHandler<3> *> matrixFreeDofHandlerVectorInput;

  for (unsigned int i = 0; i < matrixFreeConstraintsInputVector.size(); ++i)
    matrixFreeDofHandlerVectorInput.push_back(&dofHandlerHRefined);

  const unsigned int phiTotDofHandlerIndexHRefined = 1;

  matrixFreeDofHandlerVectorInput.push_back(&dofHandlerHRefined);
  const unsigned phiExtDofHandlerIndexHRefined =
    matrixFreeDofHandlerVectorInput.size() - 1;
  matrixFreeConstraintsInputVector.push_back(&onlyHangingNodeConstraints);


  std::vector<Quadrature<1>> quadratureVector;
  quadratureVector.push_back(
    QGauss<1>(C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>()));
  quadratureVector.push_back(QIterated<1>(QGauss<1>(C_num1DQuadSmearedCharge()),
                                          C_numCopies1DQuadSmearedCharge()));
  quadratureVector.push_back(QIterated<1>(QGauss<1>(C_num1DQuadLPSP<FEOrder>()),
                                          C_numCopies1DQuadLPSP()));
  quadratureVector.push_back(QGauss<1>(FEOrderElectro + 1));

  dealii::MatrixFree<3, double> matrixFreeDataHRefined;

  matrixFreeDataHRefined.reinit(matrixFreeDofHandlerVectorInput,
                                matrixFreeConstraintsInputVector,
                                quadratureVector,
                                additional_data);



  std::map<dealii::types::global_dof_index, double>
    atomHRefinedNodeIdToChargeMap;
  if (!dftParameters::floatingNuclearCharges)
    locateAtomCoreNodes(dofHandlerHRefined, atomHRefinedNodeIdToChargeMap);


  // solve vself in bins on h refined mesh
  std::vector<std::vector<double>> localVselfsHRefined;
  distributedCPUVec<double>        phiExtHRefined;
  matrixFreeDataHRefined.initialize_dof_vector(phiExtHRefined,
                                               phiExtDofHandlerIndexHRefined);
  if (dftParameters::verbosity == 2)
    pcout
      << std::endl
      << "Solving for nuclear charge self potential in bins on h refined mesh: ";
  vselfBinsManagerHRefined.solveVselfInBins(
    matrixFreeDataHRefined,
    2,
    3,
    constraintsHRefined,
    d_imagePositionsTrunc,
    d_imageIdsTrunc,
    d_imageChargesTrunc,
    localVselfsHRefined,
    d_bQuadValuesAllAtoms,
    d_bQuadAtomIdsAllAtoms,
    d_bQuadAtomIdsAllAtomsImages,
    d_bCellNonTrivialAtomIds,
    d_bCellNonTrivialAtomIdsBins,
    d_bCellNonTrivialAtomImageIds,
    d_bCellNonTrivialAtomImageIdsBins,
    d_smearedChargeWidths,
    d_smearedChargeScaling,
    1,
    dftParameters::smearedNuclearCharges);

  //
  // solve the Poisson problem for total rho
  //
  distributedCPUVec<double> phiTotRhoOutHRefined;
  matrixFreeDataHRefined.initialize_dof_vector(phiTotRhoOutHRefined,
                                               phiTotDofHandlerIndexHRefined);

  dealiiLinearSolver                            dealiiCGSolver(d_mpiCommParent,
                                    mpi_communicator,
                                    dealiiLinearSolver::CG);
  poissonSolverProblem<FEOrder, FEOrderElectro> phiTotalSolverProblem(
    mpi_communicator);

  phiTotalSolverProblem.reinit(
    matrixFreeDataHRefined,
    phiTotRhoOutHRefined,
    *matrixFreeConstraintsInputVector[phiTotDofHandlerIndexHRefined],
    phiTotDofHandlerIndexHRefined,
    0,
    3,
    atomHRefinedNodeIdToChargeMap,
    d_bQuadValuesAllAtoms,
    1,
    rhoOutHRefinedQuadValues,
    true,
    dftParameters::periodicX && dftParameters::periodicY &&
      dftParameters::periodicZ && !dftParameters::pinnedNodeForPBC,
    dftParameters::smearedNuclearCharges);

  if (dftParameters::verbosity == 2)
    pcout
      << std::endl
      << "Solving for total electrostatic potential (rhoIn+b) on h refined mesh: ";
  dealiiCGSolver.solve(phiTotalSolverProblem,
                       dftParameters::absLinearSolverTolerance,
                       dftParameters::maxLinearSolverIterations,
                       dftParameters::verbosity);

  std::map<dealii::CellId, std::vector<double>> pseudoVLocHRefined;
  std::map<unsigned int, std::map<dealii::CellId, std::vector<double>>>
    pseudoVLocAtomsHRefined;

  std::map<dealii::types::global_dof_index, Point<3>> supportPointsHRef;
  DoFTools::map_dofs_to_support_points(MappingQ1<3, 3>(),
                                       dofHandlerHRefined,
                                       supportPointsHRef);
  if (dftParameters::isPseudopotential)
    initLocalPseudoPotential(dofHandlerHRefined,
                             2,
                             matrixFreeDataHRefined,
                             phiExtDofHandlerIndexHRefined,
                             onlyHangingNodeConstraints,
                             supportPointsHRef,
                             vselfBinsManagerHRefined,
                             phiExtHRefined,
                             pseudoVLocHRefined,
                             pseudoVLocAtomsHRefined);

  energyCalculator energyCalcHRefined(d_mpiCommParent,
                                      mpi_communicator,
                                      interpoolcomm,
                                      interBandGroupComm);

  dispersionCorrection dispersionCorrHRefined(d_mpiCommParent,
                                              mpi_communicator,
                                              interpoolcomm,
                                              interBandGroupComm);


  const double totalEnergy =
    dftParameters::spinPolarized == 0 ?
      energyCalcHRefined.computeEnergy(dofHandlerHRefined,
                                       dofHandler,
                                       quadrature,
                                       quadrature,
                                       matrixFreeDataHRefined.get_quadrature(1),
                                       matrixFreeDataHRefined.get_quadrature(2),
                                       eigenValues,
                                       d_kPointWeights,
                                       fermiEnergy,
                                       funcX,
                                       funcC,
                                       dispersionCorrHRefined,
                                       d_phiInValues,
                                       phiTotRhoOutHRefined,
                                       *rhoInValues,
                                       *rhoOutValues,
                                       d_rhoOutValuesLpspQuad,
                                       rhoOutHRefinedQuadValues,
                                       rhoOutValuesLpspQuadHRefined,
                                       *gradRhoInValues,
                                       *gradRhoOutValues,
                                       d_rhoCore,
                                       d_gradRhoCore,
                                       d_bQuadValuesAllAtoms,
                                       d_bCellNonTrivialAtomIds,
                                       localVselfsHRefined,
                                       d_pseudoVLoc,
                                       pseudoVLocHRefined,
                                       atomHRefinedNodeIdToChargeMap,
                                       atomLocations.size(),
                                       lowerBoundKindex,
                                       1,
                                       true,
                                       dftParameters::smearedNuclearCharges) :
      energyCalcHRefined.computeEnergySpinPolarized(
        dofHandlerHRefined,
        dofHandler,
        quadrature,
        quadrature,
        matrixFreeDataHRefined.get_quadrature(1),
        matrixFreeDataHRefined.get_quadrature(2),
        eigenValues,
        d_kPointWeights,
        fermiEnergy,
        fermiEnergyUp,
        fermiEnergyDown,
        funcX,
        funcC,
        dispersionCorrHRefined,
        d_phiInValues,
        phiTotRhoOutHRefined,
        *rhoInValues,
        *rhoOutValues,
        d_rhoOutValuesLpspQuad,
        rhoOutHRefinedQuadValues,
        rhoOutValuesLpspQuadHRefined,
        *gradRhoInValues,
        *gradRhoOutValues,
        *rhoInValuesSpinPolarized,
        *rhoOutValuesSpinPolarized,
        *gradRhoInValuesSpinPolarized,
        *gradRhoOutValuesSpinPolarized,
        d_rhoCore,
        d_gradRhoCore,
        d_bQuadValuesAllAtoms,
        d_bCellNonTrivialAtomIds,
        localVselfsHRefined,
        d_pseudoVLoc,
        pseudoVLocHRefined,
        atomHRefinedNodeIdToChargeMap,
        atomLocations.size(),
        lowerBoundKindex,
        1,
        true,
        dftParameters::smearedNuclearCharges);

  d_groundStateEnergy = totalEnergy;
  if (dftParameters::verbosity >= 1)
    pcout << "Entropic energy: " << d_entropicEnergy << std::endl;


  computing_timer.leave_subsection("h refinement electrostatics");
  computingTimerStandard.leave_subsection("h refinement electrostatics");
}
