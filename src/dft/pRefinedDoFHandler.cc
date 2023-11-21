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
#include <dft.h>
#include <vectorUtilities.h>

namespace dftfe
{
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

    d_dofHandlerPRefined.reinit(triaObject);
    d_dofHandlerPRefined.distribute_dofs(
      dealii::FE_Q<3>(dealii::QGaussLobatto<1>(FEOrderElectro + 1)));

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

    std::vector<dealii::Tensor<1, 3>> offsetVectors;
    // resize offset vectors
    offsetVectors.resize(3);

    for (unsigned int i = 0; i < 3; ++i)
      for (unsigned int j = 0; j < 3; ++j)
        offsetVectors[i][j] =
          unitVectorsXYZ[i][j] - d_domainBoundingVectors[i][j];

    std::vector<dealii::GridTools::PeriodicFacePair<
      typename dealii::DoFHandler<3>::cell_iterator>>
                                      periodicity_vector2;
    const std::array<unsigned int, 3> periodic = {d_dftParamsPtr->periodicX,
                                                  d_dftParamsPtr->periodicY,
                                                  d_dftParamsPtr->periodicZ};

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
      dealii::GridTools::collect_periodic_faces(
        d_dofHandlerPRefined,
        /*b_id1*/ 2 * i + 1,
        /*b_id2*/ 2 * i + 2,
        /*direction*/ periodicDirectionVector[i],
        periodicity_vector2,
        offsetVectors[periodicDirectionVector[i]]);

    dealii::DoFTools::make_periodicity_constraints<3, 3>(periodicity_vector2,
                                                         d_constraintsPRefined);

    d_constraintsPRefined.close();

    //
    // initialize rho nodal dofHandler and constraint matrices
    //

    d_dofHandlerRhoNodal.reinit(triaObject);
    d_dofHandlerRhoNodal.distribute_dofs(
      dealii::FE_Q<3>(dealii::QGaussLobatto<1>(
        C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>() + 1)));

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
      dealii::GridTools::collect_periodic_faces(
        d_dofHandlerRhoNodal,
        /*b_id1*/ 2 * i + 1,
        /*b_id2*/ 2 * i + 2,
        /*direction*/ periodicDirectionVector[i],
        periodicity_vector_rhonodal,
        offsetVectors[periodicDirectionVector[i]]);

    dealii::DoFTools::make_periodicity_constraints<3, 3>(
      periodicity_vector_rhonodal, d_constraintsRhoNodal);

    d_constraintsRhoNodal.close();

    if (d_dftParamsPtr->createConstraintsFromSerialDofhandler)
      {
        vectorTools::createParallelConstraintMatrixFromSerial(
          d_mesh.getSerialMeshUnmoved(),
          d_dofHandlerPRefined,
          d_mpiCommParent,
          mpi_communicator,
          d_domainBoundingVectors,
          d_constraintsPRefined,
          d_constraintsPRefinedOnlyHanging,
          d_dftParamsPtr->verbosity,
          d_dftParamsPtr->periodicX,
          d_dftParamsPtr->periodicY,
          d_dftParamsPtr->periodicZ);

        vectorTools::createParallelConstraintMatrixFromSerial(
          d_mesh.getSerialMeshUnmoved(),
          d_dofHandlerRhoNodal,
          d_mpiCommParent,
          mpi_communicator,
          d_domainBoundingVectors,
          d_constraintsRhoNodal,
          d_constraintsRhoNodalOnlyHanging,
          d_dftParamsPtr->verbosity,
          d_dftParamsPtr->periodicX,
          d_dftParamsPtr->periodicY,
          d_dftParamsPtr->periodicZ);
      }
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  dftClass<FEOrder, FEOrderElectro>::initpRefinedObjects(
    const bool recomputeBasisData,
    const bool meshOnlyDeformed,
    const bool vselfPerturbationUpdateForStress)
  {
    d_dofHandlerPRefined.distribute_dofs(d_dofHandlerPRefined.get_fe());
    d_dofHandlerRhoNodal.distribute_dofs(d_dofHandlerRhoNodal.get_fe());

    d_supportPointsPRefined.clear();
    dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<3, 3>(),
                                                 d_dofHandlerPRefined,
                                                 d_supportPointsPRefined);

    // matrix free data structure
    typename dealii::MatrixFree<3>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
      dealii::MatrixFree<3>::AdditionalData::partition_partition;
    if (d_dftParamsPtr->isCellStress)
      additional_data.mapping_update_flags =
        dealii::update_values | dealii::update_gradients |
        dealii::update_JxW_values | dealii::update_quadrature_points;
    else
      additional_data.mapping_update_flags = dealii::update_values |
                                             dealii::update_gradients |
                                             dealii::update_JxW_values;

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

    if (d_dftParamsPtr->pinnedNodeForPBC)
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

    d_constraintsVectorElectro.push_back(
      &d_constraintsForTotalPotentialElectro);
    d_phiTotDofHandlerIndexElectro = d_constraintsVectorElectro.size() - 1;

    d_binsStartDofHandlerIndexElectro = d_constraintsVectorElectro.size();

    double init_bins;
    MPI_Barrier(d_mpiCommParent);
    init_bins = MPI_Wtime();
    //
    // Dirichlet BC constraints on the boundary of fictitious ball
    // used for computing self-potential (Vself) using Poisson problem
    // with atoms belonging to a given bin
    //
    if (meshOnlyDeformed)
      {
        computing_timer.enter_subsection("Update atom bins bc");
        d_vselfBinsManager.updateBinsBc(d_constraintsVectorElectro,
                                        d_constraintsPRefinedOnlyHanging,
                                        d_dofHandlerPRefined,
                                        d_constraintsPRefined,
                                        atomLocations,
                                        d_imagePositionsTrunc,
                                        d_imageIdsTrunc,
                                        d_imageChargesTrunc,
                                        vselfPerturbationUpdateForStress);
        computing_timer.leave_subsection("Update atom bins bc");
      }
    else
      {
        computing_timer.enter_subsection("Create atom bins");
        d_vselfBinsManager.createAtomBins(d_constraintsVectorElectro,
                                          d_constraintsPRefinedOnlyHanging,
                                          d_dofHandlerPRefined,
                                          d_constraintsPRefined,
                                          atomLocations,
                                          d_imagePositionsTrunc,
                                          d_imageIdsTrunc,
                                          d_imageChargesTrunc,
                                          d_dftParamsPtr->radiusAtomBall);

        d_netFloatingDispSinceLastBinsUpdate.clear();
        d_netFloatingDispSinceLastBinsUpdate.resize(atomLocations.size() * 3,
                                                    0.0);
        computing_timer.leave_subsection("Create atom bins");
      }

    MPI_Barrier(d_mpiCommParent);
    init_bins = MPI_Wtime() - init_bins;
    if (d_dftParamsPtr->verbosity >= 4)
      pcout
        << "updateAtomPositionsAndMoveMesh: initBoundaryConditions: Time taken for bins update: "
        << init_bins << std::endl;

    d_constraintsVectorElectro.push_back(&d_constraintsPRefinedOnlyHanging);
    d_phiExtDofHandlerIndexElectro = d_constraintsVectorElectro.size() - 1;

    if (d_dftParamsPtr->constraintsParallelCheck)
      {
        dealii::IndexSet locally_active_dofs_debug;
        dealii::DoFTools::extract_locally_active_dofs(
          d_dofHandlerPRefined, locally_active_dofs_debug);

        const std::vector<dealii::IndexSet> &locally_owned_dofs_debug =
          dealii::Utilities::MPI::all_gather(
            mpi_communicator, d_dofHandlerPRefined.locally_owned_dofs());

        AssertThrow(
          d_constraintsForTotalPotentialElectro.is_consistent_in_parallel(
            locally_owned_dofs_debug,
            locally_active_dofs_debug,
            mpi_communicator),
          dealii::ExcMessage(
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


    std::vector<dealii::Quadrature<1>> quadratureVector;
    quadratureVector.push_back(dealii::QGauss<1>(
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>()));
    quadratureVector.push_back(
      dealii::QIterated<1>(dealii::QGauss<1>(C_num1DQuadLPSP<FEOrder>()),
                           C_numCopies1DQuadLPSP()));
    if (d_dftParamsPtr->isCellStress)
      quadratureVector.push_back(dealii::QIterated<1>(
        dealii::QGauss<1>(C_num1DQuadSmearedChargeStress()),
        C_numCopies1DQuadSmearedChargeStress()));
    else if (d_dftParamsPtr->meshSizeOuterBall > 2.2)
      quadratureVector.push_back(
        dealii::QIterated<1>(dealii::QGauss<1>(C_num1DQuadSmearedChargeHigh()),
                             C_numCopies1DQuadSmearedChargeHigh()));
    else
      quadratureVector.push_back(
        dealii::QIterated<1>(dealii::QGauss<1>(C_num1DQuadSmearedCharge()),
                             C_numCopies1DQuadSmearedCharge()));
    quadratureVector.push_back(dealii::QGauss<1>(FEOrderElectro + 1));


    d_densityQuadratureIdElectro       = 0;
    d_lpspQuadratureIdElectro          = 1;
    d_smearedChargeQuadratureIdElectro = 2;
    d_phiTotAXQuadratureIdElectro      = 3;

    d_matrixFreeDataPRefined.reinit(dealii::MappingQ1<3, 3>(),
                                    matrixFreeDofHandlerVectorInput,
                                    d_constraintsVectorElectro,
                                    quadratureVector,
                                    additional_data);
    if (recomputeBasisData)
      {
        basisOperationsPtrElectroHost = std::make_shared<
          dftfe::basis::
            FEBasisOperations<double, double, dftfe::utils::MemorySpace::HOST>>(
          d_matrixFreeDataPRefined, d_constraintsVectorElectro, d_BLASWrapperPtrHost);
        dftfe::basis::UpdateFlags updateFlags = dftfe::basis::update_values |
                                                dftfe::basis::update_gradients |
                                                dftfe::basis::update_transpose;
        std::vector<unsigned int> quadratureIndices{0, 1, 2, 3};
        basisOperationsPtrElectroHost->init(d_densityDofHandlerIndexElectro,
                                            quadratureIndices,
                                            updateFlags);
      }
#if defined(DFTFE_WITH_DEVICE)
    if (d_dftParamsPtr->useDevice && recomputeBasisData)
      {
        basisOperationsPtrElectroDevice = std::make_shared<
          dftfe::basis::FEBasisOperations<double,
                                          double,
                                          dftfe::utils::MemorySpace::DEVICE>>(
          d_matrixFreeDataPRefined, d_constraintsVectorElectro, d_BLASWrapperPtr);
        basisOperationsPtrElectroDevice->init(*basisOperationsPtrElectroHost);
      }
#endif

    //
    // locate atom core nodes
    //
    if (!d_dftParamsPtr->floatingNuclearCharges)
      locateAtomCoreNodes(d_dofHandlerPRefined, d_atomNodeIdToChargeMap);

    //
    // create duplicate constraints object with flattened maps for faster access
    //
    d_constraintsRhoNodalInfo.initialize(
      d_matrixFreeDataPRefined.get_vector_partitioner(
        d_densityDofHandlerIndexElectro),
      d_constraintsRhoNodal);
  }
#include "dft.inst.cc"
} // namespace dftfe
