// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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
// @author Phani Motamarri, Shiva Rudraraju, Sambit Das
//
#include "applyHomogeneousDirichletBC.cc"
#include "locatenodes.cc"



template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
dftClass<FEOrder, FEOrderElectro>::initBoundaryConditions(
  const bool meshOnlyDeformed,
  const bool vselfPerturbationUpdateForStress)
{
  TimerOutput::Scope scope(computing_timer, "moved setup");

  double init_dofhandlerobjs;
  MPI_Barrier(MPI_COMM_WORLD);
  init_dofhandlerobjs = MPI_Wtime();
  //
  // initialize FE objects again
  //
  dofHandler.distribute_dofs(FE);
  dofHandlerEigen.distribute_dofs(FEEigen);

  if (!vselfPerturbationUpdateForStress)
    {
      pcout << std::endl << "Finite element mesh information" << std::endl;
      pcout << "-------------------------------------------------" << std::endl;
      pcout
        << "FE interpolating polynomial order for Kohn-Sham eigenvalue problem: "
        << FEOrder << "\n"
        << "FE interpolating polynomial order for electrostatics solve: "
        << FEOrderElectro << "\n"
        << "FE interpolating polynomial order for nodal electron density computation: "
        << C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>() << "\n"
        << "number of elements: "
        << dofHandler.get_triangulation().n_global_active_cells() << "\n"
        << "number of degrees of freedom for the Kohn-Sham eigenvalue problem : "
        << dofHandler.n_dofs() << std::endl;
    }

  double minElemLength = 1e+6;
  for (const auto &cell :
       dofHandler.get_triangulation().active_cell_iterators())
    if (cell->is_locally_owned())
      if (cell->minimum_vertex_distance() < minElemLength)
        minElemLength = cell->minimum_vertex_distance();

  minElemLength = Utilities::MPI::min(minElemLength, mpi_communicator);

  if (dftParameters::verbosity >= 1 && !vselfPerturbationUpdateForStress)
    pcout << "Minimum mesh size: " << minElemLength << std::endl;

  if (!vselfPerturbationUpdateForStress)
    pcout << "-------------------------------------------------" << std::endl;

  if (dftParameters::verbosity >= 1 && !meshOnlyDeformed)
    {
      pcout
        << std::endl
        << "-----------------------------------------------------------------------------"
        << std::endl;
#ifdef USE_COMPLEX
      const double totalMem =
        2.0 * dofHandler.n_dofs() * (dftParameters::spinPolarized + 1) *
          d_kPointWeights.size() * d_numEigenValues *
          (2.0 + 3.0 * std::min(dftParameters::wfcBlockSize, d_numEigenValues) /
                   d_numEigenValues) *
          8 / 1e+9 +
        0.5 * Utilities::MPI::n_mpi_processes(mpi_communicator);
#else
      const double totalMem =
        (dftParameters::useMixedPrecCGS_O == true ||
         dftParameters::useMixedPrecCGS_SR == true ||
         dftParameters::useMixedPrecXTHXSpectrumSplit == true) ?
          dofHandler.n_dofs() * (dftParameters::spinPolarized + 1) *
              d_numEigenValues *
              (1.5 + 3.0 *
                       std::min(dftParameters::wfcBlockSize, d_numEigenValues) /
                       d_numEigenValues) *
              8 / 1e+9 +
            0.5 * Utilities::MPI::n_mpi_processes(mpi_communicator) :
          dofHandler.n_dofs() * (dftParameters::spinPolarized + 1) *
              d_numEigenValues *
              (1.0 + 3.0 *
                       std::min(dftParameters::wfcBlockSize, d_numEigenValues) /
                       d_numEigenValues) *
              8 / 1e+9 +
            0.5 * Utilities::MPI::n_mpi_processes(mpi_communicator);
#endif
      const double perProcMem =
        totalMem / Utilities::MPI::n_mpi_processes(mpi_communicator);
      pcout << "Rough estimate of peak memory requirement (RAM) total: "
            << totalMem << " GB." << std::endl;
      pcout << "Rough estimate of peak memory requirement (RAM) per MPI task: "
            << perProcMem << " GB." << std::endl;
      pcout << "DFT-FE Message: many of the memory optimizations implemented\n"
            << "in DFT-FE are useful only for larger system sizes."
            << std::endl;
      pcout
        << "-----------------------------------------------------------------------------"
        << std::endl;
    }

  if (dofHandler.n_dofs() > 15000 && !meshOnlyDeformed)
    {
      if (dofHandler.n_dofs() / n_mpi_processes < 4000 &&
          dftParameters::verbosity >= 1)
        {
          pcout
            << "DFT-FE Warning: The number of degrees of freedom per domain decomposition processor are less than 4000, where the parallel scaling efficiency is not good. We recommend to use 4000 or more degrees of freedom per domain decomposition processor. For further parallelization use input parameters NPBAND and/or NPKPT(in case of multiple k points)."
            << std::endl;
        }
    }

  if (dftParameters::verbosity >= 4)
    dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                      "Dofs distributed again");
  d_supportPoints.clear();
  DoFTools::map_dofs_to_support_points(MappingQ1<3, 3>(),
                                       dofHandler,
                                       d_supportPoints);

  d_supportPointsEigen.clear();
  DoFTools::map_dofs_to_support_points(MappingQ1<3, 3>(),
                                       dofHandlerEigen,
                                       d_supportPointsEigen);

  MPI_Barrier(MPI_COMM_WORLD);
  init_dofhandlerobjs = MPI_Wtime() - init_dofhandlerobjs;
  if (dftParameters::verbosity >= 4)
    pcout
      << "initBoundaryConditions: Time taken for creating dofhandler related objects: "
      << init_dofhandlerobjs << std::endl;

  if (dftParameters::verbosity >= 4)
    dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                      "Created support points");
  //
  // matrix free data structure
  //
  typename MatrixFree<3>::AdditionalData additional_data;
  // comment this if using deal ii version 9
  // additional_data.mpi_communicator = MPI_COMM_WORLD;
  additional_data.tasks_parallel_scheme =
    MatrixFree<3>::AdditionalData::partition_partition;
  if (dftParameters::isCellStress)
    additional_data.mapping_update_flags = update_values | update_gradients |
                                           update_JxW_values |
                                           update_quadrature_points;

  // clear existing constraints matrix vector
  d_constraintsVector.clear();

  // push back into Constraint Matrices
  d_constraintsVector.push_back(&constraintsNone);

  if (dftParameters::constraintsParallelCheck)
    {
      IndexSet locally_active_dofs_debug;
      DoFTools::extract_locally_active_dofs(dofHandler,
                                            locally_active_dofs_debug);

      const std::vector<IndexSet> &locally_owned_dofs_debug =
        Utilities::MPI::all_gather(mpi_communicator,
                                   dofHandler.locally_owned_dofs());

      AssertThrow(
        constraintsNone.is_consistent_in_parallel(locally_owned_dofs_debug,
                                                  locally_active_dofs_debug,
                                                  mpi_communicator),
        ExcMessage(
          "DFT-FE Error: Constraints are not consistent in parallel."));
    }

  //
  // create matrix free structure
  //
  std::vector<const DoFHandler<3> *> dofHandlerVector;

  dofHandlerVector.push_back(&dofHandler);

  d_densityDofHandlerIndex = 0;
  // phiTotDofHandlerIndex = 1;

  dofHandlerVector.push_back(&dofHandlerEigen);         // DofHandler For Eigen
  d_eigenDofHandlerIndex = dofHandlerVector.size() - 1; // For Eigen
  d_constraintsVector.push_back(&constraintsNoneEigen); // For Eigen;

  std::vector<Quadrature<1>> quadratureVector;
  quadratureVector.push_back(
    QGauss<1>(C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>()));
  quadratureVector.push_back(
    QIterated<1>(QGauss<1>(C_num1DQuadNLPSP<FEOrder>()),
                 C_numCopies1DQuadNLPSP()));
  quadratureVector.push_back(
    QGaussLobatto<1>(C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>() + 1));
  quadratureVector.push_back(QIterated<1>(QGauss<1>(C_num1DQuadLPSP<FEOrder>()),
                                          C_numCopies1DQuadLPSP()));

  d_densityQuadratureId = 0;
  d_nlpspQuadratureId   = 1;
  d_gllQuadratureId     = 2;
  d_lpspQuadratureId    = 3;

  double init_force;
  MPI_Barrier(MPI_COMM_WORLD);
  init_force = MPI_Wtime();
  //
  //
  //
  forcePtr->initMoved(dofHandlerVector, d_constraintsVector, false);
  d_forceDofHandlerIndex = d_constraintsVector.size() - 1;
  /*
  forcePtr->initMoved(dofHandlerVector,
      d_constraintsVector,
      true);
  */

  if (dftParameters::verbosity >= 4)
    dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                      "Called force init moved");

  MPI_Barrier(MPI_COMM_WORLD);
  init_force = MPI_Wtime() - init_force;
  if (dftParameters::verbosity >= 4)
    pcout << "initBoundaryConditions: Time taken for force init moved: "
          << init_force << std::endl;


  double init_mf;
  MPI_Barrier(MPI_COMM_WORLD);
  init_mf = MPI_Wtime();

  matrix_free_data.reinit(dofHandlerVector,
                          d_constraintsVector,
                          quadratureVector,
                          additional_data);

  MPI_Barrier(MPI_COMM_WORLD);
  init_mf = MPI_Wtime() - init_mf;
  if (dftParameters::verbosity >= 4)
    pcout << "initBoundaryConditions: Time taken for matrix free reinit: "
          << init_mf << std::endl;

  if (dftParameters::verbosity >= 4)
    dftUtils::printCurrentMemoryUsage(mpi_communicator,
                                      "Called matrix free reinit");

  // compute volume of the domain
  d_domainVolume = computeVolume(dofHandler);

  double init_pref;
  MPI_Barrier(MPI_COMM_WORLD);
  init_pref = MPI_Wtime();
  //
  // init 2p matrix-free objects using appropriate constraint matrix and
  // quadrature rule
  //
  initpRefinedObjects(meshOnlyDeformed, vselfPerturbationUpdateForStress);


  MPI_Barrier(MPI_COMM_WORLD);
  init_pref = MPI_Wtime() - init_pref;
  if (dftParameters::verbosity >= 4)
    pcout << "initBoundaryConditions: Time taken for initpRefinedObjects: "
          << init_pref << std::endl;

  if (!meshOnlyDeformed)
  {  
    createMasterChargeIdToImageIdMaps(d_pspCutOff,
                                      d_imageIds,
                                      d_imagePositions,
                                      d_globalChargeIdToImageIdMap);

    createMasterChargeIdToImageIdMaps(d_nlPSPCutOff,
                                    d_imageIdsTrunc,
                                    d_imagePositionsTrunc,
                                    d_globalChargeIdToImageIdMapTrunc);
  }
}
