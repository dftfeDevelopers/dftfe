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

   
   std::map<dealii::CellId, std::vector<double> > rhoOutHRefinedQuadValues;
   
   //
   //access quadrature object
   //
   dealii::QGauss<3> quadrature(C_num1DQuad<FEOrder>());

   //
   //subdivide the existing mesh and project electron-density onto the new mesh
   //
   d_mesh.generateSubdividedMeshWithQuadData(matrix_free_data,
					     constraintsNone,
					     quadrature,
					     FEOrder,
					     *rhoOutValues,
					     rhoOutHRefinedQuadValues);

					     
   
   dealii::parallel::distributed::Triangulation<3> & electrostaticsTria = d_mesh.getElectrostaticsMesh();
   

   dealii::DoFHandler<3> dofHandlerHRefined;
   dofHandlerHRefined.initialize(electrostaticsTria,dealii::FE_Q<3>(dealii::QGaussLobatto<1>(FEOrder+1)));
   dofHandlerHRefined.distribute_dofs(dofHandlerHRefined.get_fe());

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

   moveMeshToAtoms(electrostaticsTria,
		   true);

   dofHandlerHRefined.distribute_dofs(dofHandlerHRefined.get_fe());


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
					   d_imagePositions,
					   d_imageIds,
					   d_imageCharges,
					   dftParameters::radiusAtomBall);

   std::vector<const dealii::DoFHandler<3> *> matrixFreeDofHandlerVectorInput;

   for(unsigned int i = 0; i < matrixFreeConstraintsInputVector.size(); ++i)
     matrixFreeDofHandlerVectorInput.push_back(&dofHandlerHRefined);

   const unsigned int phiTotDofHandlerIndexHRefined = 1;

   matrixFreeDofHandlerVectorInput.push_back(&dofHandlerHRefined);
   const unsigned phiExtDofHandlerIndexHRefined = matrixFreeDofHandlerVectorInput.size()-1;
   matrixFreeConstraintsInputVector.push_back(&onlyHangingNodeConstraints);

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

   //solve the Poisson problem for total rho
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


}
