// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Phani Motamarri (2018), Shiva Rudraraju (2016), Sambit Das (2017)
//
#include "applyTotalPotentialDirichletBC.cc"
#include "locatenodes.cc"
#include "createBins.cc"
#include "createBinsExtraSanityCheck.cc"

#ifdef ENABLE_PERIODIC_BC
#include "applyPeriodicBCHigherOrderNodes.cc"
#endif


//init
template<unsigned int FEOrder>
void dftClass<FEOrder>::initBoundaryConditions(){
  computing_timer.enter_section("moved setup");

  //
  //initialize FE objects again
  //
  dofHandler.distribute_dofs (FE);
  dofHandlerEigen.distribute_dofs (FEEigen);

  d_supportPoints.clear();
  DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), dofHandler, d_supportPoints);

  d_supportPointsEigen.clear();
  DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), dofHandlerEigen, d_supportPointsEigen);

  //
  //write mesh to vtk file
  //

  DataOut<3> data_out;
  data_out.attach_dof_handler (dofHandler);
  data_out.build_patches ();
  if (n_mpi_processes==1)
  {
     std::ofstream output ("meshInit.vtu");
     data_out.write_vtu (output);
  }
  else
  {
     //Doesn't work with mvapich2_ib mpi libraries
     //data_out.write_vtu_in_parallel(std::string("meshInit.vtu").c_str(),mpi_communicator); 
  }

  //
  //matrix free data structure
  //
  typename MatrixFree<3>::AdditionalData additional_data;
  additional_data.mpi_communicator = MPI_COMM_WORLD;
  additional_data.tasks_parallel_scheme = MatrixFree<3>::AdditionalData::partition_partition;

  //
  //Zero Dirichlet BC constraints on the boundary of the domain
  //used for computing total electrostatic potential using Poisson problem
  //with (rho+b) as the rhs
  //
  d_constraintsForTotalPotential.clear();  
  d_constraintsForTotalPotential.reinit(locally_relevant_dofs);

#ifdef ENABLE_PERIODIC_BC
  locatePeriodicPinnedNodes();
#else
  //VectorTools::interpolate_boundary_values(dofHandler, 0, ZeroFunction<3>(), d_constraintsForTotalPotential);
  applyTotalPotentialDirichletBC();
#endif
  d_constraintsForTotalPotential.close ();

  //
  //merge with constraintsNone so that d_constraintsForTotalPotential will also have periodic
  //constraints as well for periodic problems
  d_constraintsForTotalPotential.merge(constraintsNone,ConstraintMatrix::MergeConflictBehavior::right_object_wins);
  d_constraintsForTotalPotential.close();

  //clear existing constraints matrix vector
  unsigned int count=0;
  for (std::vector<ConstraintMatrix *>::iterator it = d_constraintsVector.begin() ; it != d_constraintsVector.end(); ++it)
  { 
    if (count > 1 && count < d_bins.size()+2)
    {
      (**it).clear();
      delete (*it);
    }
    count++;
  } 

  d_constraintsVector.clear(); 
  //
  //push back into Constraint Matrices
  //
#ifdef ENABLE_PERIODIC_BC
  d_constraintsVector.push_back(&constraintsNone); 
#else
  d_constraintsVector.push_back(&constraintsNone); 
#endif

  d_constraintsVector.push_back(&d_constraintsForTotalPotential);

  //
  //Dirichlet BC constraints on the boundary of fictitious ball
  //used for computing self-potential (Vself) using Poisson problem
  //with atoms belonging to a given bin
  //
  createAtomBins(d_constraintsVector);
  createAtomBinsExtraSanityCheck();
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
  quadratureVector.push_back(QGaussLobatto<1>(C_num1DQuad<FEOrder>()));  
  //
  //
  forcePtr->initMoved();
  //push dofHandler and constraints for force
  dofHandlerVector.push_back(&(forcePtr->d_dofHandlerForce));
  forcePtr->d_forceDofHandlerIndex = dofHandlerVector.size()-1;
  d_constraintsVector.push_back(&(forcePtr->d_constraintsNoneForce));  

  std::vector<const ConstraintMatrix * > constraintsVectorTemp(d_constraintsVector.size());
  for (unsigned int iconstraint=0; iconstraint< d_constraintsVector.size(); iconstraint++)
  {
     constraintsVectorTemp[iconstraint]=d_constraintsVector[iconstraint];
  }
  matrix_free_data.reinit(dofHandlerVector, constraintsVectorTemp, quadratureVector, additional_data);


  //
  //locate atom core nodes and also locate atom nodes in each bin 
  //
  locateAtomCoreNodes();
  //
  //
  //initialize poisson and eigen problem related objects
  //
  poissonPtr->init();
  eigenPtr->init();
  
 
  computing_timer.exit_section("moved setup");   
}
