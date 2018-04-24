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

#ifdef ENABLE_PERIODIC_BC
#include "applyPeriodicBCHigherOrderNodes.cc"
#endif


//init
template<unsigned int FEOrder>
void dftClass<FEOrder>::initBoundaryConditions(){
  TimerOutput::Scope scope (computing_timer,"moved setup");
  //
  //initialize FE objects again
  //
  dofHandler.distribute_dofs (FE);
  dofHandlerEigen.distribute_dofs (FEEigen);

  //write mesh
  if (!dftParameters::reproducible_output)
  {
    DataOut<3> dataOut;
    dataOut.attach_dof_handler(dofHandler);
    dataOut.build_patches(C_num1DQuad<FEOrder>());
    dftUtils::writeDataVTUParallelLowestPoolId(dataOut,
	                                     mpi_communicator,
					     interpoolcomm,
					     std::string("currentMesh"));
  }

  d_supportPoints.clear();
  DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), dofHandler, d_supportPoints);

  d_supportPointsEigen.clear();
  DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), dofHandlerEigen, d_supportPointsEigen);

  //
  //matrix free data structure
  //
  typename MatrixFree<3>::AdditionalData additional_data;
  //comment this if using deal ii version 9
  //additional_data.mpi_communicator = MPI_COMM_WORLD;
  additional_data.tasks_parallel_scheme = MatrixFree<3>::AdditionalData::partition_partition;

  //
  //Zero Dirichlet BC constraints on the boundary of the domain
  //used for computing total electrostatic potential using Poisson problem
  //with (rho+b) as the rhs
  //
  d_constraintsForTotalPotential.clear();
  d_constraintsForTotalPotential.reinit(locally_relevant_dofs);

#ifdef ENABLE_PERIODIC_BC
  locatePeriodicPinnedNodes(dofHandler,constraintsNone,d_constraintsForTotalPotential);
#endif
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

  //
  //Dirichlet BC constraints on the boundary of fictitious ball
  //used for computing self-potential (Vself) using Poisson problem
  //with atoms belonging to a given bin
  //
  d_vselfBinsManager.createAtomBins(d_constraintsVector,
	                            dofHandler,
				    constraintsNone,
				    atomLocations,
				    d_imagePositions,
				    d_imageIds,
				    d_imageCharges,
				    dftParameters::radiusAtomBall);

  //
  //create matrix free structure
  //
  std::vector<const DoFHandler<3> *> dofHandlerVector;

  for(int i = 0; i < d_constraintsVector.size(); ++i)
    dofHandlerVector.push_back(&dofHandler);

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
  //
  //
  forcePtr->initMoved();
  //push dofHandler and constraints for force
  dofHandlerVector.push_back(&(forcePtr->d_dofHandlerForce));
  forcePtr->d_forceDofHandlerIndex = dofHandlerVector.size()-1;
  d_constraintsVector.push_back(&(forcePtr->d_constraintsNoneForce));

  matrix_free_data.reinit(dofHandlerVector, d_constraintsVector, quadratureVector, additional_data);

  //
  //locate atom core nodes
  //
  locateAtomCoreNodes(dofHandler,d_atomNodeIdToChargeMap);

  //compute volume of the domain
  d_domainVolume=computeVolume(dofHandler);

  //initialize eigen solve related object
  eigenPtr->init();
}
