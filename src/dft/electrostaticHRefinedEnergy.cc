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
// @author Phani Motamarri
//


template<unsigned int FEOrder>
void dftClass<FEOrder>::computeElectrostaticEnergyHRefined(const bool computeForces)
{
   computing_timer.enter_section("h refinement electrostatics");
   computingTimerStandard.enter_section("h refinement electrostatics");
   if (dftParameters::verbosity>=1)
        pcout<< std::endl<<"-----------------Re computing electrostatics on h globally refined mesh--------------"<<std::endl;




   //
   //access quadrature object
   //
   dealii::QGauss<3> quadrature(C_num1DQuad<FEOrder>());
   const unsigned int n_q_points = quadrature.size();

   std::map<dealii::CellId, std::vector<double> > _gradRhoOutValues;
   std::map<dealii::CellId, std::vector<double> > _gradRhoOutValuesSpinPolarized;
   if (dftParameters::isCellStress || dftParameters::isIonForce)
	if (!(dftParameters::xc_id == 4))
	{

	       gradRhoOutValues=&_gradRhoOutValues;
	       if (dftParameters::spinPolarized==1)
		   gradRhoOutValuesSpinPolarized=&_gradRhoOutValuesSpinPolarized;


	       typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
	       for (; cell!=endc; ++cell)
		  if (cell->is_locally_owned())
		    {
			const dealii::CellId cellId=cell->id();
			(*rhoOutValues)[cellId] = std::vector<double>(n_q_points,0.0);
			(*gradRhoOutValues)[cellId] = std::vector<double>(3*n_q_points,0.0);

			if (dftParameters::spinPolarized==1)
			{
			   (*rhoOutValuesSpinPolarized)[cellId]
				 = std::vector<double>(2*n_q_points,0.0);
			   (*gradRhoOutValuesSpinPolarized)[cellId]
				 = std::vector<double>(6*n_q_points,0.0);
			}
		    }

	       computeRhoFromPSI(rhoOutValues,
			    gradRhoOutValues,
			    rhoOutValuesSpinPolarized,
			    gradRhoOutValuesSpinPolarized,
			    true,
			    false);
	}

   //
   //project and create a nodal field of the same mesh from the quadrature data (L2 projection from quad points to nodes)
   //
   vectorType rhoNodalFieldCoarse;
   matrix_free_data.initialize_dof_vector(rhoNodalFieldCoarse);
   rhoNodalFieldCoarse = 0.0;

   vectorType delxRhoNodalFieldCoarse;
   vectorType delyRhoNodalFieldCoarse;
   vectorType delzRhoNodalFieldCoarse;
   if (dftParameters::isCellStress || dftParameters::isIonForce)
   {

       matrix_free_data.initialize_dof_vector(delxRhoNodalFieldCoarse);
       delxRhoNodalFieldCoarse = 0.0;

       matrix_free_data.initialize_dof_vector(delyRhoNodalFieldCoarse);
       delyRhoNodalFieldCoarse = 0.0;

       matrix_free_data.initialize_dof_vector(delzRhoNodalFieldCoarse);
       delzRhoNodalFieldCoarse = 0.0;
   }

   //
   //create a lambda function for L2 projection of quadrature electron-density to nodal electron density
   //
   std::function<double(const typename dealii::DoFHandler<3>::active_cell_iterator & cell,const unsigned int q)> funcRho = [&](const typename dealii::DoFHandler<3>::active_cell_iterator & cell , const unsigned int q)
     {return (*rhoOutValues).find(cell->id())->second[q];};

   dealii::VectorTools::project<3,dealii::parallel::distributed::Vector<double> >(dealii::MappingQ1<3,3>(),
										  matrix_free_data.get_dof_handler(),
										  constraintsNone,
										  quadrature,
										  funcRho,
										  rhoNodalFieldCoarse);
   if (dftParameters::isCellStress || dftParameters::isIonForce)
   {
       std::function<double(const typename dealii::DoFHandler<3>::active_cell_iterator & cell,const unsigned int q)> funcDelxRho = [&](const typename dealii::DoFHandler<3>::active_cell_iterator & cell , const unsigned int q)
	 {return (*gradRhoOutValues).find(cell->id())->second[3*q];};

       dealii::VectorTools::project<3,dealii::parallel::distributed::Vector<double> >(dealii::MappingQ1<3,3>(),
										      matrix_free_data.get_dof_handler(),
										      constraintsNone,
										      quadrature,
										      funcDelxRho,
										      delxRhoNodalFieldCoarse);

       std::function<double(const typename dealii::DoFHandler<3>::active_cell_iterator & cell,const unsigned int q)> funcDelyRho = [&](const typename dealii::DoFHandler<3>::active_cell_iterator & cell , const unsigned int q)
	 {return (*gradRhoOutValues).find(cell->id())->second[3*q+1];};

       dealii::VectorTools::project<3,dealii::parallel::distributed::Vector<double> >(dealii::MappingQ1<3,3>(),
										      matrix_free_data.get_dof_handler(),
										      constraintsNone,
										      quadrature,
										      funcDelyRho,
										      delyRhoNodalFieldCoarse);

       std::function<double(const typename dealii::DoFHandler<3>::active_cell_iterator & cell,const unsigned int q)> funcDelzRho = [&](const typename dealii::DoFHandler<3>::active_cell_iterator & cell , const unsigned int q)
	 {return (*gradRhoOutValues).find(cell->id())->second[3*q+2];};

       dealii::VectorTools::project<3,dealii::parallel::distributed::Vector<double> >(dealii::MappingQ1<3,3>(),
										      matrix_free_data.get_dof_handler(),
										      constraintsNone,
										      quadrature,
										      funcDelzRho,
										      delzRhoNodalFieldCoarse);
   }

   rhoNodalFieldCoarse.update_ghost_values();
   constraintsNone.distribute(rhoNodalFieldCoarse);
   rhoNodalFieldCoarse.update_ghost_values();


   if (dftParameters::isCellStress || dftParameters::isIonForce)
   {
       delxRhoNodalFieldCoarse.update_ghost_values();
       constraintsNone.distribute(delxRhoNodalFieldCoarse);
       delxRhoNodalFieldCoarse.update_ghost_values();

       delyRhoNodalFieldCoarse.update_ghost_values();
       constraintsNone.distribute(delyRhoNodalFieldCoarse);
       delyRhoNodalFieldCoarse.update_ghost_values();

       delzRhoNodalFieldCoarse.update_ghost_values();
       constraintsNone.distribute(delzRhoNodalFieldCoarse);
       delzRhoNodalFieldCoarse.update_ghost_values();
   }

   //
   //compute the total charge using rho nodal field for debugging purposes
   //
   if(dftParameters::verbosity >= 4)
     {
       const double integralRhoValue = totalCharge(matrix_free_data.get_dof_handler(),
						   rhoNodalFieldCoarse);

       pcout<<"Value of total charge on coarse mesh using L2 projected nodal field: "<< integralRhoValue<<std::endl;
     }


   //
   //subdivide the existing mesh and project electron-density onto the new mesh
   //

   //
   //initialize the new dofHandler to refine and do a solution transfer
   //
   dealii::parallel::distributed::Triangulation<3> & electrostaticsTriaRho = d_mesh.getElectrostaticsMeshRho();

   dealii::DoFHandler<3> dofHandlerHRefined;
   dofHandlerHRefined.initialize(electrostaticsTriaRho,dealii::FE_Q<3>(dealii::QGaussLobatto<1>(FEOrder+1)));
   dofHandlerHRefined.distribute_dofs(dofHandlerHRefined.get_fe());

   //
   //create a solution transfer object and prepare for refinement and solution transfer
   //
   parallel::distributed::SolutionTransfer<3,vectorType> solTrans(dofHandlerHRefined);
   electrostaticsTriaRho.set_all_refine_flags();
   electrostaticsTriaRho.prepare_coarsening_and_refinement();

   if (dftParameters::isCellStress || dftParameters::isIonForce)
   {
       std::vector<const vectorType *> vecAllIn(4);
       vecAllIn[0]=&rhoNodalFieldCoarse;
       vecAllIn[1]=&delxRhoNodalFieldCoarse;
       vecAllIn[2]=&delyRhoNodalFieldCoarse;
       vecAllIn[3]=&delzRhoNodalFieldCoarse;


       solTrans.prepare_for_coarsening_and_refinement(vecAllIn);
   }
   else
   {
       solTrans.prepare_for_coarsening_and_refinement(rhoNodalFieldCoarse);
   }

   electrostaticsTriaRho.execute_coarsening_and_refinement();

   dofHandlerHRefined.distribute_dofs(dofHandlerHRefined.get_fe());

   //
   //print refined mesh details
   //
   if (dftParameters::verbosity>=2)
     {
       pcout << std::endl<<"Finite element mesh information after subdividing the mesh"<<std::endl;
       pcout<<"-------------------------------------------------"<<std::endl;
       pcout << "number of elements: "
	     << dofHandlerHRefined.get_triangulation().n_global_active_cells()
	     << std::endl
	     << "number of degrees of freedom: "
	     << dofHandlerHRefined.n_dofs()
	     << std::endl;
     }

   dealii::IndexSet locallyRelevantDofs;
   dealii::DoFTools::extract_locally_relevant_dofs(dofHandlerHRefined, locallyRelevantDofs);

   IndexSet ghost_indices = locallyRelevantDofs;
   ghost_indices.subtract_set(dofHandlerHRefined.locally_owned_dofs());


   dealii::ConstraintMatrix onlyHangingNodeConstraints;
   onlyHangingNodeConstraints.reinit(locallyRelevantDofs);
   dealii::DoFTools::make_hanging_node_constraints(dofHandlerHRefined, onlyHangingNodeConstraints);
   onlyHangingNodeConstraints.close();

   dealii::ConstraintMatrix constraintsHRefined;
   constraintsHRefined.reinit(locallyRelevantDofs);
   dealii::DoFTools::make_hanging_node_constraints(dofHandlerHRefined, constraintsHRefined);
   std::vector<std::vector<double> > unitVectorsXYZ;
   unitVectorsXYZ.resize(3);

   for(unsigned int i = 0; i < 3; ++i)
    {
      unitVectorsXYZ[i].resize(3,0.0);
      unitVectorsXYZ[i][i] = 0.0;
    }

   std::vector<Tensor<1,3> > offsetVectors;
   //resize offset vectors
   offsetVectors.resize(3);

   for(unsigned int i = 0; i < 3; ++i)
     for(unsigned int j = 0; j < 3; ++j)
	  offsetVectors[i][j] = unitVectorsXYZ[i][j] - d_domainBoundingVectors[i][j];

   std::vector<dealii::GridTools::PeriodicFacePair<typename dealii::DoFHandler<3>::cell_iterator> > periodicity_vector2;
   const std::array<unsigned int,3> periodic = {dftParameters::periodicX, dftParameters::periodicY, dftParameters::periodicZ};

   std::vector<int> periodicDirectionVector;
   for (unsigned int  d= 0; d < 3; ++d)
     {
       if (periodic[d]==1)
	 {
	   periodicDirectionVector.push_back(d);
	 }
     }

   for (unsigned int i = 0; i < std::accumulate(periodic.begin(),periodic.end(),0); ++i)
         GridTools::collect_periodic_faces(dofHandlerHRefined, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ periodicDirectionVector[i], periodicity_vector2,offsetVectors[periodicDirectionVector[i]]);

   dealii::DoFTools::make_periodicity_constraints<dealii::DoFHandler<3> >(periodicity_vector2, constraintsHRefined);
   constraintsHRefined.close();


   //
   //create rho nodal field on the refined mesh and conduct solution transfer
   //
   vectorType rhoNodalFieldRefined = dealii::parallel::distributed::Vector<double>(dofHandlerHRefined.locally_owned_dofs(),
										   ghost_indices,
										   mpi_communicator);
   rhoNodalFieldRefined.zero_out_ghosts();

   vectorType delxRhoNodalFieldRefined;
   vectorType delyRhoNodalFieldRefined;
   vectorType delzRhoNodalFieldRefined;

   if (dftParameters::isCellStress || dftParameters::isIonForce)
   {

       delxRhoNodalFieldRefined.reinit(rhoNodalFieldRefined);
       delxRhoNodalFieldRefined.zero_out_ghosts();

       delyRhoNodalFieldRefined.reinit(rhoNodalFieldRefined);
       delyRhoNodalFieldRefined.zero_out_ghosts();

       delzRhoNodalFieldRefined.reinit(rhoNodalFieldRefined);
       delzRhoNodalFieldRefined.zero_out_ghosts();
   }

   if (dftParameters::isCellStress || dftParameters::isIonForce)
   {
       std::vector<vectorType *> vecAllOut(4);
       vecAllOut[0]=&rhoNodalFieldRefined;
       vecAllOut[1]=&delxRhoNodalFieldRefined;
       vecAllOut[2]=&delyRhoNodalFieldRefined;
       vecAllOut[3]=&delzRhoNodalFieldRefined;

       solTrans.interpolate(vecAllOut);
   }
   else
       solTrans.interpolate(rhoNodalFieldRefined);

   rhoNodalFieldRefined.update_ghost_values();
   constraintsHRefined.distribute(rhoNodalFieldRefined);
   rhoNodalFieldRefined.update_ghost_values();

   if (dftParameters::isCellStress || dftParameters::isIonForce)
   {
       delxRhoNodalFieldRefined.update_ghost_values();
       constraintsHRefined.distribute(delxRhoNodalFieldRefined);
       delxRhoNodalFieldRefined.update_ghost_values();

       delyRhoNodalFieldRefined.update_ghost_values();
       constraintsHRefined.distribute(delyRhoNodalFieldRefined);
       delyRhoNodalFieldRefined.update_ghost_values();

       delzRhoNodalFieldRefined.update_ghost_values();
       constraintsHRefined.distribute(delzRhoNodalFieldRefined);
       delzRhoNodalFieldRefined.update_ghost_values();
   }


   //
   //move the refined mesh so that it forms exact subdivison of coarse moved mesh
   //
   dealii::parallel::distributed::Triangulation<3> & electrostaticsTriaDisp = d_mesh.getElectrostaticsMeshDisp();

   //
   //create guassian Move object
   //
   if(d_autoMesh == 1)
     moveMeshToAtoms(electrostaticsTriaDisp,
		     d_mesh.getSerialMeshElectrostatics(),
		     true,
		     true);
   else
     {

       //
       //move electrostatics mesh
       //

       d_gaussianMovePar.init(electrostaticsTriaDisp,
			      d_mesh.getSerialMeshElectrostatics(),
			      d_domainBoundingVectors);

       d_gaussianMovePar.moveMeshTwoLevelElectro();

     }




   //
   //call init for the force computation subsequently
   //
   forcePtr->initUnmoved(electrostaticsTriaRho,
		         d_mesh.getSerialMeshElectrostatics(),
			 d_domainBoundingVectors,
			 true,
			 d_gaussianConstantForce);

   d_mesh.resetMesh(electrostaticsTriaDisp,
		    electrostaticsTriaRho);

   dofHandlerHRefined.distribute_dofs(dofHandlerHRefined.get_fe());

   //
   //fill in quadrature values of the field on the refined mesh and compute total charge
   //
   std::map<dealii::CellId, std::vector<double> > rhoOutHRefinedQuadValues;
   const double integralRhoValue = totalCharge(dofHandlerHRefined,
					       rhoNodalFieldRefined,
					       rhoOutHRefinedQuadValues);
   //
   //fill in grad rho at quadrature values of the field on the refined mesh
   //
   std::map<dealii::CellId, std::vector<double> > gradRhoOutHRefinedQuadValues;

   if (dftParameters::isCellStress || dftParameters::isIonForce)
   {
       FEValues<3> fe_values (dofHandlerHRefined.get_fe(), quadrature, update_values);
       std::vector<double> tempDelxRho(n_q_points);
       std::vector<double> tempDelyRho(n_q_points);
       std::vector<double> tempDelzRho(n_q_points);

       DoFHandler<3>::active_cell_iterator
       cell = dofHandlerHRefined.begin_active(),
       endc = dofHandlerHRefined.end();
       for(; cell!=endc; ++cell)
	  if(cell->is_locally_owned())
	  {
	      fe_values.reinit (cell);
	      fe_values.get_function_values(delxRhoNodalFieldRefined,tempDelxRho);
	      fe_values.get_function_values(delyRhoNodalFieldRefined,tempDelyRho);
	      fe_values.get_function_values(delzRhoNodalFieldRefined,tempDelzRho);

	      gradRhoOutHRefinedQuadValues[cell->id()].resize(3*n_q_points);
	      for (unsigned int q_point=0; q_point<n_q_points; ++q_point)
		{
		  gradRhoOutHRefinedQuadValues[cell->id()][3*q_point] = tempDelxRho[q_point];
		  gradRhoOutHRefinedQuadValues[cell->id()][3*q_point+1] = tempDelyRho[q_point];
		  gradRhoOutHRefinedQuadValues[cell->id()][3*q_point+2] = tempDelzRho[q_point];
		}
	  }
   }

   //
   //compute total charge using rhoNodalRefined field
   //
   if(dftParameters::verbosity >= 4)
     {
       pcout<<"Value of total charge computed on moved subdivided mesh after solution transfer: "<< integralRhoValue<<std::endl;
     }


   //matrix free data structure
   typename dealii::MatrixFree<3>::AdditionalData additional_data;
   additional_data.tasks_parallel_scheme = dealii::MatrixFree<3>::AdditionalData::partition_partition;

   //Zero Dirichlet BC constraints on the boundary of the domain
   //used for computing total electrostatic potential using Poisson problem
   //with (rho+b) as the rhs
   dealii::ConstraintMatrix constraintsForTotalPotential;
   constraintsForTotalPotential.reinit(locallyRelevantDofs);

   if (dftParameters::pinnedNodeForPBC)
      locatePeriodicPinnedNodes(dofHandlerHRefined,
	                        constraintsHRefined,
	                        constraintsForTotalPotential);
   applyHomogeneousDirichletBC(dofHandlerHRefined,constraintsForTotalPotential);
   constraintsForTotalPotential.close();

   constraintsForTotalPotential.merge(constraintsHRefined,ConstraintMatrix::MergeConflictBehavior::right_object_wins);
   constraintsForTotalPotential.close();

   //clear existing constraints matrix vector
   std::vector<const dealii::ConstraintMatrix*> matrixFreeConstraintsInputVector;

   matrixFreeConstraintsInputVector.push_back(&constraintsHRefined);

   matrixFreeConstraintsInputVector.push_back(&constraintsForTotalPotential);

   //Dirichlet BC constraints on the boundary of fictitious ball
   //used for computing self-potential (Vself) using Poisson problem
   //with atoms belonging to a given bin

   vselfBinsManager<FEOrder> vselfBinsManagerHRefined(mpi_communicator);
   vselfBinsManagerHRefined.createAtomBins(matrixFreeConstraintsInputVector,
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
     DoFTools::extract_locally_active_dofs(dofHandlerHRefined, locally_active_dofs_debug);

     const std::vector<IndexSet>& locally_owned_dofs_debug= dofHandlerHRefined.locally_owned_dofs_per_processor();

     AssertThrow(constraintsHRefined.is_consistent_in_parallel(locally_owned_dofs_debug,
                                               locally_active_dofs_debug,
                                               mpi_communicator),ExcMessage("DFT-FE Error: Constraints are not consistent in parallel. This is because of a known issue in the deal.ii library, which will be fixed soon. Currently, please set H REFINED ELECTROSTATICS to false."));

     AssertThrow(constraintsForTotalPotential.is_consistent_in_parallel(locally_owned_dofs_debug,
                                               locally_active_dofs_debug,
                                               mpi_communicator),ExcMessage("DFT-FE Error: Constraints are not consistent in parallel. This is because of a known issue in the deal.ii library, which will be fixed soon. Currently, please set H REFINED ELECTROSTATICS to false."));

     for (unsigned int i=2; i<matrixFreeConstraintsInputVector.size();i++)
	 AssertThrow(matrixFreeConstraintsInputVector[i]->is_consistent_in_parallel(locally_owned_dofs_debug,
						   locally_active_dofs_debug,
						   mpi_communicator),ExcMessage("DFT-FE Error: Constraints are not consistent in parallel. This is because of a known issue in the deal.ii library, which will be fixed soon. Currently, please set H REFINED ELECTROSTATICS to false."));
   }

   std::vector<const dealii::DoFHandler<3> *> matrixFreeDofHandlerVectorInput;

   for(unsigned int i = 0; i < matrixFreeConstraintsInputVector.size(); ++i)
     matrixFreeDofHandlerVectorInput.push_back(&dofHandlerHRefined);

   const unsigned int phiTotDofHandlerIndexHRefined = 1;

   matrixFreeDofHandlerVectorInput.push_back(&dofHandlerHRefined);
   const unsigned phiExtDofHandlerIndexHRefined = matrixFreeDofHandlerVectorInput.size()-1;
   matrixFreeConstraintsInputVector.push_back(&onlyHangingNodeConstraints);



   forcePtr->initMoved(matrixFreeDofHandlerVectorInput,
	               matrixFreeConstraintsInputVector,
	               true,
		       true);


   std::vector<Quadrature<1> > quadratureVector;
   quadratureVector.push_back(QGauss<1>(C_num1DQuad<FEOrder>()));

   dealii::MatrixFree<3,double> matrixFreeDataHRefined;

   matrixFreeDataHRefined.reinit(matrixFreeDofHandlerVectorInput,
	                         matrixFreeConstraintsInputVector,
			         quadratureVector,
			         additional_data);



   std::map<dealii::types::global_dof_index, double> atomHRefinedNodeIdToChargeMap;
   locateAtomCoreNodes(dofHandlerHRefined,atomHRefinedNodeIdToChargeMap);


   //solve vself in bins on h refined mesh
   std::vector<std::vector<double> > localVselfsHRefined;
   vectorType phiExtHRefined;
   matrixFreeDataHRefined.initialize_dof_vector(phiExtHRefined,phiExtDofHandlerIndexHRefined);
   if (dftParameters::verbosity==2)
        pcout<< std::endl<<"Solving for nuclear charge self potential in bins on h refined mesh: ";
   vselfBinsManagerHRefined.solveVselfInBins(matrixFreeDataHRefined,
		                             2,
	                                     phiExtHRefined,
				             onlyHangingNodeConstraints,
					     d_imagePositions,
					     d_imageIds,
					     d_imageCharges,
	                                     localVselfsHRefined);

   //
   //solve the Poisson problem for total rho
   //
   vectorType phiTotRhoOutHRefined;
   matrixFreeDataHRefined.initialize_dof_vector(phiTotRhoOutHRefined,phiTotDofHandlerIndexHRefined);

   dealiiLinearSolver dealiiCGSolver(mpi_communicator, dealiiLinearSolver::CG);
   poissonSolverProblem<FEOrder> phiTotalSolverProblem(mpi_communicator);

   phiTotalSolverProblem.reinit(matrixFreeDataHRefined,
	                        phiTotRhoOutHRefined,
				*matrixFreeConstraintsInputVector[phiTotDofHandlerIndexHRefined],
                                phiTotDofHandlerIndexHRefined,
	                        atomHRefinedNodeIdToChargeMap,
				rhoOutHRefinedQuadValues,
                                true,
                                dftParameters::periodicX && dftParameters::periodicY && dftParameters::periodicZ && !dftParameters::pinnedNodeForPBC);

   if (dftParameters::verbosity==2)
        pcout<< std::endl<<"Solving for total electrostatic potential (rhoIn+b) on h refined mesh: ";
   dealiiCGSolver.solve(phiTotalSolverProblem,
			dftParameters::absLinearSolverTolerance,
			dftParameters::maxLinearSolverIterations,
			dftParameters::verbosity);

   std::map<dealii::CellId, std::vector<double> > pseudoVLocHRefined;
   std::map<dealii::CellId, std::vector<double> > gradPseudoVLocHRefined;
   std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > gradPseudoVLocAtomsHRefined;

   if(dftParameters::isPseudopotential)
       initLocalPseudoPotential(dofHandlerHRefined,
				quadrature,
				pseudoVLocHRefined,
				gradPseudoVLocHRefined,
				gradPseudoVLocAtomsHRefined);

   energyCalculator energyCalcHRefined(mpi_communicator, interpoolcomm, interBandGroupComm);


  const double totalEnergy = dftParameters::spinPolarized==0 ?
    energyCalcHRefined.computeEnergy(dofHandlerHRefined,
				     dofHandler,
				     quadrature,
				     quadrature,
				     eigenValues,
				     d_kPointWeights,
				     fermiEnergy,
				     funcX,
				     funcC,
				     d_phiTotRhoIn,
				     phiTotRhoOutHRefined,
				     d_phiExt,
				     phiExtHRefined,
				     *rhoInValues,
				     *rhoOutValues,
				     rhoOutHRefinedQuadValues,
				     *gradRhoInValues,
				     *gradRhoOutValues,
				     localVselfsHRefined,
				     d_pseudoVLoc,
				     pseudoVLocHRefined,
				     atomHRefinedNodeIdToChargeMap,
				     atomLocations.size(),
				     lowerBoundKindex,
				     1,
				     true) :
    energyCalcHRefined.computeEnergySpinPolarized(dofHandlerHRefined,
						  dofHandler,
						  quadrature,
						  quadrature,
						  eigenValues,
						  d_kPointWeights,
						  fermiEnergy,
						  fermiEnergyUp,
						  fermiEnergyDown,
						  funcX,
						  funcC,
						  d_phiTotRhoIn,
						  phiTotRhoOutHRefined,
						  d_phiExt,
						  phiExtHRefined,
						  *rhoInValues,
						  *rhoOutValues,
						  rhoOutHRefinedQuadValues,
						  *gradRhoInValues,
						  *gradRhoOutValues,
						  *rhoInValuesSpinPolarized,
						  *rhoOutValuesSpinPolarized,
						  *gradRhoInValuesSpinPolarized,
						  *gradRhoOutValuesSpinPolarized,
						  localVselfsHRefined,
						  d_pseudoVLoc,
						  pseudoVLocHRefined,
						  atomHRefinedNodeIdToChargeMap,
						  atomLocations.size(),
						  lowerBoundKindex,
						  1,
						  true);

  d_groundStateEnergy = totalEnergy;



    if(dftParameters::isCellStress)
      {
	//
	//Create the full dealii partitioned array
	//
	d_eigenVectorsFlattened.resize((1+dftParameters::spinPolarized)*d_kPointWeights.size());

	for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
	  {
	    vectorTools::createDealiiVector<dataTypes::number>(matrix_free_data.get_vector_partitioner(),
							       d_numEigenValues,
							       d_eigenVectorsFlattened[kPoint]);


	    d_eigenVectorsFlattened[kPoint] = dataTypes::number(0.0);

	  }


	Assert(d_eigenVectorsFlattened[0].local_size()==d_eigenVectorsFlattenedSTL[0].size(),
		  dealii::ExcMessage("Incorrect local sizes of STL and dealii arrays"));

	constraintsNoneDataInfo.precomputeMaps(matrix_free_data.get_vector_partitioner(),
					       d_eigenVectorsFlattened[0].get_partitioner(),
					       d_numEigenValues);

	const unsigned int localVectorSize = d_eigenVectorsFlattenedSTL[0].size()/d_numEigenValues;

	for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
	  {
	    for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	      {
		for(unsigned int iWave = 0; iWave < d_numEigenValues; ++iWave)
		  {
		    d_eigenVectorsFlattened[kPoint].local_element(iNode*d_numEigenValues+iWave)
		      = d_eigenVectorsFlattenedSTL[kPoint][iNode*d_numEigenValues+iWave];
		  }
	      }

	    constraintsNoneDataInfo.distribute(d_eigenVectorsFlattened[kPoint],
					       d_numEigenValues);

	  }
      }

    computing_timer.exit_section("h refinement electrostatics");
    computingTimerStandard.exit_section("h refinement electrostatics");

    if (dftParameters::isIonForce)
      {

 	computing_timer.enter_section("Ion force computation");
	computingTimerStandard.enter_section("Ion force computation");
	if (computeForces)
	{
	    forcePtr->computeAtomsForces(matrix_free_data,
					 eigenDofHandlerIndex,
					 phiExtDofHandlerIndex,
					 phiTotDofHandlerIndex,
					 d_phiTotRhoIn,
					 d_phiTotRhoOut,
					 d_phiExt,
					 d_pseudoVLoc,
					 d_gradPseudoVLoc,
					 d_gradPseudoVLocAtoms,
					 d_noConstraints,
					 d_vselfBinsManager,
					 matrixFreeDataHRefined,
					 phiTotDofHandlerIndexHRefined,
					 phiExtDofHandlerIndexHRefined,
					 phiTotRhoOutHRefined,
					 phiExtHRefined,
					 rhoOutHRefinedQuadValues,
					 gradRhoOutHRefinedQuadValues,
					 pseudoVLocHRefined,
					 gradPseudoVLocHRefined,
					 gradPseudoVLocAtomsHRefined,
					 onlyHangingNodeConstraints,
					 vselfBinsManagerHRefined);
	    forcePtr->printAtomsForces();
	}
	computingTimerStandard.exit_section("Ion force computation");
	computing_timer.exit_section("Ion force computation");
      }
#ifdef USE_COMPLEX
    if (dftParameters::isCellStress)
      {

	computing_timer.enter_section("Cell stress computation");
	computingTimerStandard.enter_section("Cell stress computation");
	if (computeForces)
	{
	    forcePtr->computeStress(matrix_free_data,
				    eigenDofHandlerIndex,
				    phiExtDofHandlerIndex,
				    phiTotDofHandlerIndex,
				    d_phiTotRhoIn,
				    d_phiTotRhoOut,
				    d_phiExt,
				    d_pseudoVLoc,
				    d_gradPseudoVLoc,
				    d_gradPseudoVLocAtoms,
				    d_noConstraints,
				    d_vselfBinsManager,
				    matrixFreeDataHRefined,
				    phiTotDofHandlerIndexHRefined,
				    phiExtDofHandlerIndexHRefined,
				    phiTotRhoOutHRefined,
				    phiExtHRefined,
				    rhoOutHRefinedQuadValues,
				    gradRhoOutHRefinedQuadValues,
				    pseudoVLocHRefined,
				    gradPseudoVLocHRefined,
				    gradPseudoVLocAtomsHRefined,
				    onlyHangingNodeConstraints,
				    vselfBinsManagerHRefined);
	    forcePtr->printStress();
	}
	computingTimerStandard.exit_section("Cell stress computation");
	computing_timer.exit_section("Cell stress computation");
      }
#endif
}
