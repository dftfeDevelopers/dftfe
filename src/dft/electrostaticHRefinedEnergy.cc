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
void dftClass<FEOrder>::computeElectrostaticEnergyHRefined()
{

   if (dftParameters::verbosity>=2)
        pcout<< std::endl<<"-----------------Re computing electrostatics on h globally refined mesh--------------"<<std::endl;




   //
   //access quadrature object
   //
   dealii::QGauss<3> quadrature(C_num1DQuad<FEOrder>());
   const unsigned int n_q_points = quadrature.size();




   //
   //project and create a nodal field of the same mesh from the quadrature data (L2 projection from quad points to nodes)
   //
   vectorType rhoNodalFieldCoarse;
   matrix_free_data.initialize_dof_vector(rhoNodalFieldCoarse);
   rhoNodalFieldCoarse = 0.0;

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

   rhoNodalFieldCoarse.update_ghost_values();
   constraintsNone.distribute(rhoNodalFieldCoarse);
   rhoNodalFieldCoarse.update_ghost_values();

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
   computing_timer.enter_section("h refinement electrostatics");
     /*d_mesh.generateSubdividedMeshWithQuadData(matrix_free_data,
					     constraintsNone,
					     quadrature,
					     FEOrder,
					     *rhoOutValues,
					     rhoOutHRefinedQuadValues);*/

   //
   //initialize the new dofHandler to refine and do a solution transfer
   //
   dealii::parallel::distributed::Triangulation<3> & electrostaticsTria = d_mesh.getElectrostaticsMesh();

   dealii::DoFHandler<3> dofHandlerHRefined;
   dofHandlerHRefined.initialize(electrostaticsTria,dealii::FE_Q<3>(dealii::QGaussLobatto<1>(FEOrder+1)));
   dofHandlerHRefined.distribute_dofs(dofHandlerHRefined.get_fe());

   //
   //create a solution transfer object and prepare for refinement and solution transfer
   //
   parallel::distributed::SolutionTransfer<3,vectorType> solTrans(dofHandlerHRefined);
   electrostaticsTria.set_all_refine_flags();
   electrostaticsTria.prepare_coarsening_and_refinement();
   solTrans.prepare_for_coarsening_and_refinement(rhoNodalFieldCoarse);
   electrostaticsTria.execute_coarsening_and_refinement();

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

   /*moveMeshToAtoms(electrostaticsTria,
   		   true);

		   dofHandlerHRefined.distribute_dofs(dofHandlerHRefined.get_fe());

   //
   //compute the electron density on the moved mesh
   //
   if (dftParameters::verbosity>=4)
     {
       FEValues<3> fe_values(dofHandlerHRefined.get_fe(),quadrature,update_values | update_JxW_values | update_quadrature_points);

       typename dealii::DoFHandler<3>::active_cell_iterator cellRefinedNew = dofHandlerHRefined.begin_active(), endcRefinedNew = dofHandlerHRefined.end();
       double totalCharge=0.0;
       for(; cellRefinedNew!=endcRefinedNew; ++cellRefinedNew)
	 {
	   if(cellRefinedNew->is_locally_owned())
	     {
	       fe_values.reinit(cellRefinedNew);
	       for(unsigned int q_point = 0; q_point < quadrature.size(); ++q_point)
		 totalCharge += rhoOutHRefinedQuadValues.find(cellRefinedNew->id())->second[q_point]*fe_values.JxW(q_point);
	     }
	 }
       pcout<<"Value of total charge on refined mesh after solution transfer: "<< Utilities::MPI::sum(totalCharge, mpi_communicator)<<std::endl;
       }*/


   //matrix free data structure
   typename dealii::MatrixFree<3>::AdditionalData additional_data;
   additional_data.tasks_parallel_scheme = dealii::MatrixFree<3>::AdditionalData::partition_partition;

   //Zero Dirichlet BC constraints on the boundary of the domain
   //used for computing total electrostatic potential using Poisson problem
   //with (rho+b) as the rhs
   dealii::ConstraintMatrix constraintsForTotalPotential;
   constraintsForTotalPotential.reinit(locallyRelevantDofs);

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
					   dofHandlerHRefined,
					   constraintsHRefined,
					   atomLocations,
					   d_imagePositionsTrunc,
					   d_imageIdsTrunc,
					   d_imageChargesTrunc,
					   dftParameters::radiusAtomBall);

   std::vector<const dealii::DoFHandler<3> *> matrixFreeDofHandlerVectorInput;

   for(unsigned int i = 0; i < matrixFreeConstraintsInputVector.size(); ++i)
     matrixFreeDofHandlerVectorInput.push_back(&dofHandlerHRefined);

   const unsigned int phiTotDofHandlerIndexHRefined = 1;

   matrixFreeDofHandlerVectorInput.push_back(&dofHandlerHRefined);
   const unsigned phiExtDofHandlerIndexHRefined = matrixFreeDofHandlerVectorInput.size()-1;
   matrixFreeConstraintsInputVector.push_back(&onlyHangingNodeConstraints);

   forcePtr->initUnmoved(electrostaticsTria,d_domainBoundingVectors,true);

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

   //
   //create rho nodal field on the refined mesh and conduct solution transfer
   //
   vectorType rhoNodalFieldRefined;
   matrixFreeDataHRefined.initialize_dof_vector(rhoNodalFieldRefined);
   rhoNodalFieldRefined.zero_out_ghosts();
   solTrans.interpolate(rhoNodalFieldRefined);
   rhoNodalFieldRefined.update_ghost_values();
   constraintsHRefined.distribute(rhoNodalFieldRefined);
   rhoNodalFieldRefined.update_ghost_values();

   //
   //fill in quadrature values of the field on the refined mesh and compute total charge
   //
   std::map<dealii::CellId, std::vector<double> > rhoOutHRefinedQuadValues;
   const double integralRhoValue = totalCharge(dofHandlerHRefined,
					       rhoNodalFieldRefined,
					       rhoOutHRefinedQuadValues);
   //
   //compute total charge using rhoNodalRefined field
   //
   if(dftParameters::verbosity >= 4)
     {
       pcout<<"Value of total charge on refined mesh after rho solution transfer: "<< integralRhoValue<<std::endl;
     }

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
				rhoOutHRefinedQuadValues);

   if (dftParameters::verbosity==2)
        pcout<< std::endl<<"Solving for total electrostatic potential (rhoIn+b) on h refined mesh: ";
   dealiiCGSolver.solve(phiTotalSolverProblem,
			dftParameters::relLinearSolverTolerance,
			dftParameters::maxLinearSolverIterations,
			dftParameters::verbosity);

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
				     *rhoInValues,
				     *rhoOutValues,
				     rhoOutHRefinedQuadValues,
				     *gradRhoInValues,
				     *gradRhoOutValues,
				     localVselfsHRefined,
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
						  atomHRefinedNodeIdToChargeMap,
						  atomLocations.size(),
						  lowerBoundKindex,
						  1,
						  true);
  computing_timer.exit_section("h refinement electrostatics");


    if(dftParameters::isIonForce || dftParameters::isCellStress)
      {
	//
	//Create the full dealii partitioned array
	//
	d_eigenVectorsFlattened.resize((1+dftParameters::spinPolarized)*d_kPointWeights.size());

	for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
	  {
	    vectorTools::createDealiiVector<dataTypes::number>(matrix_free_data.get_vector_partitioner(),
							       numEigenValues,
							       d_eigenVectorsFlattened[kPoint]);


	    d_eigenVectorsFlattened[kPoint] = dataTypes::number(0.0);

	  }


	Assert(d_eigenVectorsFlattened[0].local_size()==d_eigenVectorsFlattenedSTL[0].size(),
		  dealii::ExcMessage("Incorrect local sizes of STL and dealii arrays"));

	constraintsNoneDataInfo.precomputeMaps(matrix_free_data.get_vector_partitioner(),
					       d_eigenVectorsFlattened[0].get_partitioner(),
					       numEigenValues);

	const unsigned int localVectorSize = d_eigenVectorsFlattenedSTL[0].size()/numEigenValues;

	for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
	  {
	    for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	      {
		for(unsigned int iWave = 0; iWave < numEigenValues; ++iWave)
		  {
		    d_eigenVectorsFlattened[kPoint].local_element(iNode*numEigenValues+iWave)
		      = d_eigenVectorsFlattenedSTL[kPoint][iNode*numEigenValues+iWave];
		  }
	      }

	    constraintsNoneDataInfo.distribute(d_eigenVectorsFlattened[kPoint],
					       numEigenValues);

	  }


	for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
	  {
	    d_eigenVectorsFlattenedSTL[kPoint].clear();
	    std::vector<dataTypes::number>().swap(d_eigenVectorsFlattenedSTL[kPoint]);
	  }

      }

    if (dftParameters::isIonForce)
      {

 	computing_timer.enter_section("Ion force computation");
	computingTimerStandard.enter_section("Ion force computation");
	forcePtr->computeAtomsForces(matrix_free_data,
		                     eigenDofHandlerIndex,
				     phiExtDofHandlerIndex,
				     phiTotDofHandlerIndex,
                                     d_phiTotRhoIn,
				     d_phiTotRhoOut,
				     d_phiExt,
				     d_noConstraints,
				     d_vselfBinsManager,
				     matrixFreeDataHRefined,
				     phiTotDofHandlerIndexHRefined,
				     phiExtDofHandlerIndexHRefined,
				     phiTotRhoOutHRefined,
				     phiExtHRefined,
				     rhoOutHRefinedQuadValues,
				     onlyHangingNodeConstraints,
				     vselfBinsManagerHRefined);
	forcePtr->printAtomsForces();
	computingTimerStandard.exit_section("Ion force computation");
	computing_timer.exit_section("Ion force computation");
      }
#ifdef USE_COMPLEX
    if (dftParameters::isCellStress)
      {

	computing_timer.enter_section("Cell stress computation");
	computingTimerStandard.enter_section("Cell stress computation");
	forcePtr->computeStress(matrix_free_data,
		                eigenDofHandlerIndex,
				phiExtDofHandlerIndex,
				phiTotDofHandlerIndex,
                                d_phiTotRhoIn,
				d_phiTotRhoOut,
				d_phiExt,
				d_noConstraints,
				d_vselfBinsManager,
				matrixFreeDataHRefined,
				phiTotDofHandlerIndexHRefined,
				phiExtDofHandlerIndexHRefined,
				phiTotRhoOutHRefined,
				phiExtHRefined,
				rhoOutHRefinedQuadValues,
				onlyHangingNodeConstraints,
			        vselfBinsManagerHRefined);
	forcePtr->printStress();
	computingTimerStandard.exit_section("Cell stress computation");
	computing_timer.exit_section("Cell stress computation");
      }
#endif

    if(dftParameters::isIonForce || dftParameters::isCellStress)
      {
	//
	//Create the full STL array from dealii flattened array
	//
	for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
	  d_eigenVectorsFlattenedSTL[kPoint].resize(numEigenValues*matrix_free_data.get_vector_partitioner()->local_size(),dataTypes::number(0.0));

	Assert(d_eigenVectorsFlattened[0].local_size()==d_eigenVectorsFlattenedSTL[0].size(),
	       dealii::ExcMessage("Incorrect local sizes of STL and dealii arrays"));

	const unsigned int localVectorSize = d_eigenVectorsFlattenedSTL[0].size()/numEigenValues;

	//
	//copy the data into STL array
	//
	for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
	  {
	    for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	      {
		for(unsigned int iWave = 0; iWave < numEigenValues; ++iWave)
		  {
		    d_eigenVectorsFlattenedSTL[kPoint][iNode*numEigenValues+iWave] = d_eigenVectorsFlattened[kPoint].local_element(iNode*numEigenValues+iWave);
		  }
	      }
	  }

	for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
	  {
	    d_eigenVectorsFlattened[kPoint].reinit(0);
	  }

      }


}
