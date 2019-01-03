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
// @author Sambit Das
//


template<unsigned int FEOrder>
void dftClass<FEOrder>::computeElectrostaticEnergyPRefined()
{
#define FEOrder_PRefined FEOrder+4
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


    for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
      {
	d_eigenVectorsFlattenedSTL[kPoint].clear();
	std::vector<dataTypes::number>().swap(d_eigenVectorsFlattenedSTL[kPoint]);
      }


  std::vector<std::vector<vectorType> > eigenVectors((1+dftParameters::spinPolarized)*d_kPointWeights.size());
#ifdef USE_COMPLEX
  for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
    {
      eigenVectors[kPoint].resize(d_numEigenValues);

      for(unsigned int i= 0; i < d_numEigenValues; ++i)
	eigenVectors[kPoint][i].reinit(d_tempEigenVec);

      vectorTools::copyFlattenedDealiiVecToSingleCompVec(d_eigenVectorsFlattened[kPoint],
							 d_numEigenValues,
							 std::make_pair(0,d_numEigenValues),
							 localProc_dof_indicesReal,
							 localProc_dof_indicesImag,
							 eigenVectors[kPoint]);
    }
#else
  for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
    {
      eigenVectors[kPoint].resize(d_numEigenValues);

      for(unsigned int i= 0; i < d_numEigenValues; ++i)
	eigenVectors[kPoint][i].reinit(d_tempEigenVec);


      vectorTools::copyFlattenedDealiiVecToSingleCompVec(d_eigenVectorsFlattened[kPoint],
							 d_numEigenValues,
							 std::make_pair(0,d_numEigenValues),
							 eigenVectors[kPoint]);
    }
#endif

  computing_timer.enter_section("p refinement electrostatics");

   if (dftParameters::verbosity>=2)
        pcout<< std::endl<<"-----------------Re computing electrostatics on p refined mesh with polynomial order: "<<FEOrder_PRefined <<"---------------"<<std::endl;

   dealii::parallel::distributed::Triangulation<3> & triaMoved = d_mesh.getParallelMeshMoved();
   dealii::parallel::distributed::Triangulation<3> & triaUnMoved = d_mesh.getParallelMeshUnmoved();

   d_mesh.resetMesh(triaUnMoved,
		    triaMoved);

   dealii::DoFHandler<3> dofHandlerPRefined;
   dofHandlerPRefined.initialize(triaMoved,dealii::FE_Q<3>(dealii::QGaussLobatto<1>(FEOrder_PRefined+1)));
   dofHandlerPRefined.distribute_dofs(dofHandlerPRefined.get_fe());

   dealii::IndexSet locallyRelevantDofs;
   dealii::DoFTools::extract_locally_relevant_dofs(dofHandlerPRefined, locallyRelevantDofs);

   dealii::ConstraintMatrix onlyHangingNodeConstraints;
   onlyHangingNodeConstraints.reinit(locallyRelevantDofs);
   dealii::DoFTools::make_hanging_node_constraints(dofHandlerPRefined, onlyHangingNodeConstraints);
   onlyHangingNodeConstraints.close();

   dealii::ConstraintMatrix constraintsPRefined;
   constraintsPRefined.reinit(locallyRelevantDofs);
   dealii::DoFTools::make_hanging_node_constraints(dofHandlerPRefined, constraintsPRefined);
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
         GridTools::collect_periodic_faces(dofHandlerPRefined, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ periodicDirectionVector[i], periodicity_vector2,offsetVectors[periodicDirectionVector[i]]);

   dealii::DoFTools::make_periodicity_constraints<dealii::DoFHandler<3> >(periodicity_vector2, constraintsPRefined);
   constraintsPRefined.close();

   moveMeshToAtoms(triaMoved);

   dofHandlerPRefined.distribute_dofs (dofHandlerPRefined.get_fe());


   //matrix free data structure
   typename dealii::MatrixFree<3>::AdditionalData additional_data;
   additional_data.tasks_parallel_scheme = dealii::MatrixFree<3>::AdditionalData::partition_partition;

   //Zero Dirichlet BC constraints on the boundary of the domain
   //used for computing total electrostatic potential using Poisson problem
   //with (rho+b) as the rhs
   dealii::ConstraintMatrix constraintsForTotalPotential;
   constraintsForTotalPotential.reinit(locallyRelevantDofs);

   locatePeriodicPinnedNodes(dofHandlerPRefined,
	                     constraintsPRefined,
	                     constraintsForTotalPotential);
   applyHomogeneousDirichletBC(dofHandlerPRefined,constraintsForTotalPotential);
   constraintsForTotalPotential.close();

   constraintsForTotalPotential.merge(constraintsPRefined,ConstraintMatrix::MergeConflictBehavior::right_object_wins);
   constraintsForTotalPotential.close();

   //clear existing constraints matrix vector
   std::vector<const dealii::ConstraintMatrix*> matrixFreeConstraintsInputVector;

   matrixFreeConstraintsInputVector.push_back(&constraintsPRefined);

   matrixFreeConstraintsInputVector.push_back(&constraintsForTotalPotential);

   //Dirichlet BC constraints on the boundary of fictitious ball
   //used for computing self-potential (Vself) using Poisson problem
   //with atoms belonging to a given bin

   vselfBinsManager<FEOrder_PRefined> vselfBinsManagerPRefined(mpi_communicator);
   vselfBinsManagerPRefined.createAtomBins(matrixFreeConstraintsInputVector,
					   dofHandlerPRefined,
					   constraintsPRefined,
					   atomLocations,
					   d_imagePositions,
					   d_imageIds,
					   d_imageCharges,
					   dftParameters::radiusAtomBall);

   std::vector<const dealii::DoFHandler<3> *> matrixFreeDofHandlerVectorInput;

   for(unsigned int i = 0; i < matrixFreeConstraintsInputVector.size(); ++i)
     matrixFreeDofHandlerVectorInput.push_back(&dofHandlerPRefined);

   const unsigned int phiTotDofHandlerIndexPRefined = 1;

   matrixFreeDofHandlerVectorInput.push_back(&dofHandlerPRefined);
   const unsigned phiExtDofHandlerIndexPRefined = matrixFreeDofHandlerVectorInput.size()-1;
   matrixFreeConstraintsInputVector.push_back(&onlyHangingNodeConstraints);

   std::vector<Quadrature<1> > quadratureVector;
   quadratureVector.push_back(QGauss<1>(C_num1DQuad<FEOrder_PRefined>()));

   dealii::MatrixFree<3,double> matrixFreeDataPRefined;

   matrixFreeDataPRefined.reinit(matrixFreeDofHandlerVectorInput,
	                         matrixFreeConstraintsInputVector,
			         quadratureVector,
			         additional_data);


   std::map<dealii::types::global_dof_index, double> atomPRefinedNodeIdToChargeMap;
   locateAtomCoreNodes(dofHandlerPRefined,atomPRefinedNodeIdToChargeMap);

   //compute p refined quad density values from original eigenvectors (not p refined)
   dealii::QGauss<3>  quadraturePRefined(C_num1DQuad<FEOrder_PRefined>());
   dealii::FEValues<3> fe_values (dofHandlerEigen.get_fe(), quadraturePRefined, dealii::update_values | dealii::update_gradients);
   const unsigned int num_quad_points = quadraturePRefined.size();

   std::map<dealii::CellId, std::vector<double> > rhoOutPRefinedQuadValues;
   typename dealii::DoFHandler<3>::active_cell_iterator cellOld = dofHandlerEigen.begin_active(), endcOld = dofHandlerEigen.end();
   for(; cellOld!=endcOld; ++cellOld)
      if(cellOld->is_locally_owned())
      {
	  rhoOutPRefinedQuadValues[cellOld->id()] = std::vector<double>(num_quad_points);

	  fe_values.reinit (cellOld);
#ifdef USE_COMPLEX
	  std::vector<dealii::Vector<double> > tempPsi(num_quad_points), tempPsi2(num_quad_points);
	  for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
	    {
	      tempPsi[q_point].reinit(2);
	      tempPsi2[q_point].reinit(2);
	    }
#else
	  std::vector<double> tempPsi(num_quad_points), tempPsi2(num_quad_points);
#endif
	  std::vector<double> rhoTemp(num_quad_points);

	  for(unsigned int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
	      for(unsigned int i=0; i<d_numEigenValues; ++i)
	      {

		      fe_values.get_function_values(eigenVectors[(1+dftParameters::spinPolarized)*kPoint][i],
						    tempPsi);
		      if(dftParameters::spinPolarized==1)
			  fe_values.get_function_values(eigenVectors[(1+dftParameters::spinPolarized)*kPoint+1][i],
							tempPsi2);

		      const double partialOccupancy=dftUtils::getPartialOccupancy
						    (eigenValues[kPoint][i],
						     fermiEnergy,
						     C_kb,
						     dftParameters::TVal);

		      double partialOccupancy2;
		      if(dftParameters::spinPolarized==1)
		         partialOccupancy2=dftUtils::getPartialOccupancy
						    (eigenValues[kPoint][i+d_numEigenValues],
						     fermiEnergy,
						     C_kb,
						     dftParameters::TVal);

		      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		      {
#ifdef USE_COMPLEX
			   if(dftParameters::spinPolarized==1)
			   {
			       const double temp1 = partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempPsi[q_point](0) + tempPsi[q_point](1)*tempPsi[q_point](1));
			       const double temp2 =  partialOccupancy2*d_kPointWeights[kPoint]*(tempPsi2[q_point](0)*tempPsi2[q_point](0) + tempPsi2[q_point](1)*tempPsi2[q_point](1));
			       rhoTemp[q_point] +=temp1+temp2;
			   }
			   else
			   {
			       const double temp= 2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempPsi[q_point](0) + tempPsi[q_point](1)*tempPsi[q_point](1));
			       rhoTemp[q_point] +=temp;
			   }
#else
			   if(dftParameters::spinPolarized==1)
			   {
			       const double temp1 = partialOccupancy*tempPsi[q_point]*tempPsi[q_point];
			       const double temp2 = partialOccupancy2*tempPsi2[q_point]*tempPsi2[q_point];
			       rhoTemp[q_point] +=temp1+temp2;
			   }
			   else
			   {
			       const double temp= 2.0*partialOccupancy*tempPsi[q_point]*tempPsi[q_point];
			       rhoTemp[q_point] +=temp;
			   }
#endif
		       }//quad point loop
	      }//eigen vectors loop

	      // gather density from all pools
	      int numPoint = num_quad_points ;
              MPI_Allreduce(&rhoTemp[0], &rhoOutPRefinedQuadValues[cellOld->id()][0], numPoint, MPI_DOUBLE, MPI_SUM, interpoolcomm) ;
      }//cell locally owned loop

   //solve vself in bins on p refined mesh
   std::vector<std::vector<double> > localVselfsPRefined;
   vectorType phiExtPRefined;
   matrixFreeDataPRefined.initialize_dof_vector(phiExtPRefined,phiExtDofHandlerIndexPRefined);
   if (dftParameters::verbosity==2)
        pcout<< std::endl<<"Solving for nuclear charge self potential in bins on p refined mesh: ";
   vselfBinsManagerPRefined.solveVselfInBins(matrixFreeDataPRefined,
		                             2,
	                                     phiExtPRefined,
				             onlyHangingNodeConstraints,
					     d_imagePositions,
					     d_imageIds,
					     d_imageCharges,
	                                     localVselfsPRefined);

   //solve the Poisson problem for total rho
   vectorType phiTotRhoOutPRefined;
   matrixFreeDataPRefined.initialize_dof_vector(phiTotRhoOutPRefined,phiTotDofHandlerIndexPRefined);

   dealiiLinearSolver dealiiCGSolver(mpi_communicator, dealiiLinearSolver::CG);
   poissonSolverProblem<FEOrder_PRefined> phiTotalSolverProblem(mpi_communicator);

   phiTotalSolverProblem.reinit(matrixFreeDataPRefined,
	                        phiTotRhoOutPRefined,
				*matrixFreeConstraintsInputVector[phiTotDofHandlerIndexPRefined],
                                phiTotDofHandlerIndexPRefined,
	                        atomPRefinedNodeIdToChargeMap,
				rhoOutPRefinedQuadValues);

   if (dftParameters::verbosity==2)
        pcout<< std::endl<<"Solving for total electrostatic potential (rhoIn+b) on p refined mesh: ";
   dealiiCGSolver.solve(phiTotalSolverProblem,
			dftParameters::relLinearSolverTolerance,
			dftParameters::maxLinearSolverIterations,
			dftParameters::verbosity);

   std::map<dealii::CellId, std::vector<double> > pseudoVLocPRefined;
   std::map<dealii::CellId, std::vector<double> > gradPseudoVLocPRefined;
   std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > gradPseudoVLocAtomsPRefined;

   if(dftParameters::isPseudopotential)
       initLocalPseudoPotential(dofHandlerPRefined,
				quadraturePRefined,
				pseudoVLocPRefined,
				gradPseudoVLocPRefined,
				gradPseudoVLocAtomsPRefined);

   energyCalculator energyCalcPRefined(mpi_communicator, interpoolcomm, interBandGroupComm);

  QGauss<3>  quadratureElectronic(C_num1DQuad<FEOrder>());

  const double totalEnergy = dftParameters::spinPolarized==0 ?
    energyCalcPRefined.computeEnergy(dofHandlerPRefined,
				     dofHandler,
				     quadraturePRefined,
				     quadratureElectronic,
				     eigenValues,
				     d_kPointWeights,
				     fermiEnergy,
				     funcX,
				     funcC,
				     d_phiTotRhoIn,
				     phiTotRhoOutPRefined,
				     d_phiExt,
				     phiExtPRefined,
				     *rhoInValues,
				     *rhoOutValues,
				     rhoOutPRefinedQuadValues,
				     *gradRhoInValues,
				     *gradRhoOutValues,
				     localVselfsPRefined,
				     d_pseudoVLoc,
				     pseudoVLocPRefined,
				     atomPRefinedNodeIdToChargeMap,
				     atomLocations.size(),
				     lowerBoundKindex,
				     1,
				     true) :
    energyCalcPRefined.computeEnergySpinPolarized(dofHandlerPRefined,
						  dofHandler,
						  quadraturePRefined,
						  quadratureElectronic,
						  eigenValues,
						  d_kPointWeights,
						  fermiEnergy,
						  fermiEnergyUp,
						  fermiEnergyDown,
						  funcX,
						  funcC,
						  d_phiTotRhoIn,
						  phiTotRhoOutPRefined,
						  d_phiExt,
						  phiExtPRefined,
						  *rhoInValues,
						  *rhoOutValues,
						  rhoOutPRefinedQuadValues,
						  *gradRhoInValues,
						  *gradRhoOutValues,
						  *rhoInValuesSpinPolarized,
						  *rhoOutValuesSpinPolarized,
						  *gradRhoInValuesSpinPolarized,
						  *gradRhoOutValuesSpinPolarized,
						  localVselfsPRefined,
						  d_pseudoVLoc,
						  pseudoVLocPRefined,
						  atomPRefinedNodeIdToChargeMap,
						  atomLocations.size(),
						  lowerBoundKindex,
						  1,
						  true);
computing_timer.exit_section("p refinement electrostatics");

}
