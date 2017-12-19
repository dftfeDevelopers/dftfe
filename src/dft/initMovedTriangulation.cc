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
// @author Shiva Rudraraju (2016), Phani Motamarri (2016), Sambit Das (2017)
//


//init
template<unsigned int FEOrder>
void dftClass<FEOrder>::initMovedTriangulation(){
  computing_timer.enter_section("setup");

  //
  //initialize FE objects
  //
  dofHandler.distribute_dofs (FE);
  dofHandlerEigen.distribute_dofs (FEEigen);

  d_supportPoints.clear();
  DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), dofHandler, d_supportPoints);

  //selectedDofsHanging.resize(dofHandler.n_dofs(),false);
  //DoFTools::extract_hanging_node_dofs(dofHandler, selectedDofsHanging);

  d_supportPointsEigen.clear();
  DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), dofHandlerEigen, d_supportPointsEigen);

  //
  //write mesh to vtk file
  //
  DataOut<3> data_out;
  data_out.attach_dof_handler (dofHandler);
  data_out.build_patches ();
  data_out.write_vtu_in_parallel(std::string("mesh.vtu").c_str(),mpi_communicator); 

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
  d_constraintsForTotalPotential.clear ();  

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

 
  //
  //push back into Constraint Matrices
  //
  d_constraintsVector.clear();
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
  quadratureVector.push_back(QGauss<1>(FEOrder+1)); 
  quadratureVector.push_back(QGaussLobatto<1>(FEOrder+1));  


  matrix_free_data.reinit(dofHandlerVector, d_constraintsVector, quadratureVector, additional_data);


  //
  //initialize eigen vectors
  //
  matrix_free_data.initialize_dof_vector(vChebyshev,eigenDofHandlerIndex);
  v0Chebyshev.reinit(vChebyshev);
  fChebyshev.reinit(vChebyshev);
  aj[0].reinit(vChebyshev); aj[1].reinit(vChebyshev); aj[2].reinit(vChebyshev);
  aj[3].reinit(vChebyshev); aj[4].reinit(vChebyshev);
  for (unsigned int i=0; i<eigenVectors[0].size(); ++i)
    {  
      PSI[i]->reinit(vChebyshev);
      tempPSI[i]->reinit(vChebyshev);
      tempPSI2[i]->reinit(vChebyshev);
      tempPSI3[i]->reinit(vChebyshev);
      tempPSI4[i]->reinit(vChebyshev);
    } 
  
  for(unsigned int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
    {
      for(unsigned int i = 0; i < eigenVectors[kPoint].size(); ++i)
	{
	  eigenVectors[kPoint][i]->reinit(vChebyshev);
	  eigenVectorsOrig[kPoint][i]->reinit(vChebyshev);
	}
    }

  //
  //locate atom core nodes and also locate atom nodes in each bin 
  //
  locateAtomCoreNodes();

  
  //
  //initialize density 
  //
  initRho();

  //
  //Initialize libxc (exchange-correlation)
  //
  int exceptParamX, exceptParamC;


  if(xc_id == 1)
    {
      exceptParamX = xc_func_init(&funcX,XC_LDA_X,XC_UNPOLARIZED);
      exceptParamC = xc_func_init(&funcC,XC_LDA_C_PZ,XC_UNPOLARIZED);
    }
  else if(xc_id == 2)
    {
      exceptParamX = xc_func_init(&funcX,XC_LDA_X,XC_UNPOLARIZED);
      exceptParamC = xc_func_init(&funcC,XC_LDA_C_PW,XC_UNPOLARIZED);
    }
  else if(xc_id == 3)
    {
      exceptParamX = xc_func_init(&funcX,XC_LDA_X,XC_UNPOLARIZED);
      exceptParamC = xc_func_init(&funcC,XC_LDA_C_VWN,XC_UNPOLARIZED);
    }
  else if(xc_id == 4)
    {
      exceptParamX = xc_func_init(&funcX,XC_GGA_X_PBE,XC_UNPOLARIZED);
      exceptParamC = xc_func_init(&funcC,XC_GGA_C_PBE,XC_UNPOLARIZED);
    }
  else if(xc_id > 4)
    {
      pcout<<"-------------------------------------"<<std::endl;
      pcout<<"Exchange or Correlation Functional not found"<<std::endl;
      pcout<<"-------------------------------------"<<std::endl;
      exit(-1);
    }

  if(exceptParamX != 0 || exceptParamC != 0)
    {
      pcout<<"-------------------------------------"<<std::endl;
      pcout<<"Exchange or Correlation Functional not found"<<std::endl;
      pcout<<"-------------------------------------"<<std::endl;
      exit(-1);
    }


  //
  //initialize local pseudopotential
  //
  if(isPseudopotential)
    {
      initLocalPseudoPotential();
      initNonLocalPseudoPotential();
      computeSparseStructureNonLocalProjectors();
      computeElementalProjectorKets();
    }
 
  //
  //
  //
  computing_timer.exit_section("setup"); 

  //
  //initialize poisson and eigen problem related objects
  //
  poissonPtr->init();
  eigenPtr->init();
  
  //
  //initialize PSI
  //
  pcout << "reading initial guess for PSI\n";
  readPSI();
}
