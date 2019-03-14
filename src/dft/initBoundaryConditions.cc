// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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




template<unsigned int FEOrder>
void dftClass<FEOrder>::initBoundaryConditions(){
  TimerOutput::Scope scope (computing_timer,"moved setup");
  //
  //initialize FE objects again
  //
  dofHandler.distribute_dofs (FE);
  dofHandlerEigen.distribute_dofs (FEEigen);

  pcout << std::endl<<"Finite element mesh information"<<std::endl;
  pcout<<"-------------------------------------------------"<<std::endl;
  pcout << "number of elements: "
	<< dofHandler.get_triangulation().n_global_active_cells()
	<< std::endl
	<< "number of degrees of freedom: "
	<< dofHandler.n_dofs()
	<< std::endl;

  double minElemLength=1e+6;
  for (const auto &cell :  dofHandler.get_triangulation().active_cell_iterators() )
    if (cell->is_locally_owned())
      if (cell->minimum_vertex_distance()<minElemLength)
	  minElemLength = cell->minimum_vertex_distance();

  minElemLength=Utilities::MPI::min(minElemLength, mpi_communicator);

  if (dftParameters::verbosity>=1)
    pcout<< "Minimum mesh size: "<<minElemLength<<std::endl;
  pcout<<"-------------------------------------------------"<<std::endl;

  if(dofHandler.n_dofs()>15000)
    {
      if(dofHandler.n_dofs()/n_mpi_processes<4000 && dftParameters::verbosity>=1)
	{
	  pcout<<"DFT-FE Warning: The number of degrees of freedom per domain decomposition processor are less than 4000, where the parallel scaling efficiency is not good. We recommend to use 4000 or more degrees of freedom per domain decomposition processor. For further parallelization use input parameters NPBAND and/or NPKPT(in case of multiple k points)."<<std::endl;
	}
    }

  if (dftParameters::verbosity>=4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
	                      "Dofs distributed again");
  d_supportPoints.clear();
  DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), dofHandler, d_supportPoints);

  d_supportPointsEigen.clear();
  DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), dofHandlerEigen, d_supportPointsEigen);

  if (dftParameters::verbosity>=4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
	                      "Created support points");
  //
  //matrix free data structure
  //
  typename MatrixFree<3>::AdditionalData additional_data;
  //comment this if using deal ii version 9
  //additional_data.mpi_communicator = MPI_COMM_WORLD;
  additional_data.tasks_parallel_scheme =MatrixFree<3>::AdditionalData::partition_partition;
  if (dftParameters::nonSelfConsistentForce)
     additional_data.mapping_update_flags = update_values|update_gradients|update_JxW_values|update_hessians;

  //
  //Zero Dirichlet BC constraints on the boundary of the domain
  //used for computing total electrostatic potential using Poisson problem
  //with (rho+b) as the rhs
  //
  d_constraintsForTotalPotential.clear();
  d_constraintsForTotalPotential.reinit(locally_relevant_dofs);

  locatePeriodicPinnedNodes(dofHandler,constraintsNone,d_constraintsForTotalPotential);
  applyHomogeneousDirichletBC(dofHandler,d_constraintsForTotalPotential);
  d_constraintsForTotalPotential.close ();

  //
  //merge with constraintsNone so that d_constraintsForTotalPotential will also have periodic
  //constraints as well for periodic problems
  d_constraintsForTotalPotential.merge(constraintsNone,ConstraintMatrix::MergeConflictBehavior::right_object_wins);
  d_constraintsForTotalPotential.close();

  //clear existing constraints matrix vector
  d_constraintsVector.clear();

  //push back into Constraint Matrices
  d_constraintsVector.push_back(&constraintsNone);

  d_constraintsVector.push_back(&d_constraintsForTotalPotential);

  if (dftParameters::verbosity>=4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
	                      "Created total potential constraint matrices");
  //
  //Dirichlet BC constraints on the boundary of fictitious ball
  //used for computing self-potential (Vself) using Poisson problem
  //with atoms belonging to a given bin
  //
  computing_timer.enter_section("Create atom bins");
  d_vselfBinsManager.createAtomBins(d_constraintsVector,
	                            d_noConstraints,
	                            dofHandler,
				    constraintsNone,
				    atomLocations,
				    d_imagePositionsTrunc,
				    d_imageIdsTrunc,
				    d_imageChargesTrunc,
				    dftParameters::radiusAtomBall);
  computing_timer.exit_section("Create atom bins");

  if (dftParameters::constraintsParallelCheck)
  {
     IndexSet locally_active_dofs_debug;
     DoFTools::extract_locally_active_dofs(dofHandler, locally_active_dofs_debug);

     const std::vector<IndexSet>& locally_owned_dofs_debug= dofHandler.locally_owned_dofs_per_processor();

     AssertThrow(d_constraintsForTotalPotential.is_consistent_in_parallel(locally_owned_dofs_debug,
                                               locally_active_dofs_debug,
                                               mpi_communicator),ExcMessage("DFT-FE Error: Constraints are not consistent in parallel."));

     AssertThrow(d_noConstraints.is_consistent_in_parallel(locally_owned_dofs_debug,
                                               locally_active_dofs_debug,
                                               mpi_communicator),ExcMessage("DFT-FE Error: Constraints are not consistent in parallel."));

     for (unsigned int i=0; i<d_constraintsVector.size();i++)
	 AssertThrow(d_constraintsVector[i]->is_consistent_in_parallel(locally_owned_dofs_debug,
						   locally_active_dofs_debug,
						   mpi_communicator),ExcMessage("DFT-FE Error: Constraints are not consistent in parallel."));
  }

  if (dftParameters::verbosity>=4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
	                      "Created vself bins and constraint matrices");

  //
  //create matrix free structure
  //
  std::vector<const DoFHandler<3> *> dofHandlerVector;

  for(int i = 0; i < d_constraintsVector.size(); ++i)
    dofHandlerVector.push_back(&dofHandler);

  densityDofHandlerIndex=0;
  phiTotDofHandlerIndex = 1;

  dofHandlerVector.push_back(&dofHandlerEigen); //DofHandler For Eigen
  eigenDofHandlerIndex = dofHandlerVector.size() - 1; //For Eigen
  d_constraintsVector.push_back(&constraintsNoneEigen); //For Eigen;

  //
  //push d_noConstraints into constraintsVector
  //
  dofHandlerVector.push_back(&dofHandler);
  phiExtDofHandlerIndex = dofHandlerVector.size()-1;
  d_constraintsVector.push_back(&d_noConstraints);

  std::vector<Quadrature<1> > quadratureVector;
  quadratureVector.push_back(QGauss<1>(C_num1DQuad<FEOrder>()));
  quadratureVector.push_back(QGaussLobatto<1>(FEOrder+1));
  quadratureVector.push_back(QGauss<1>(C_num1DQuadPSP<FEOrder>()));

  //
  //
  //
  forcePtr->initMoved(dofHandlerVector,
	              d_constraintsVector,
	              false);

  forcePtr->initMoved(dofHandlerVector,
	              d_constraintsVector,
	              true);

  if (dftParameters::verbosity>=4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
	                      "Called force init moved");

  matrix_free_data.reinit(dofHandlerVector, d_constraintsVector, quadratureVector, additional_data);

  if (dftParameters::verbosity>=4)
      dftUtils::printCurrentMemoryUsage(mpi_communicator,
	                      "Called matrix free reinit");
  //
  //locate atom core nodes
  //
  locateAtomCoreNodes(dofHandler,d_atomNodeIdToChargeMap);


  //compute volume of the domain
  d_domainVolume=computeVolume(dofHandler);

  //update gaussianMeshMovementClass object
  //d_gaussianMovePar.initMoved(d_domainBoundingVectors);
}
