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

namespace internal
{
   double computeLocalAllElectronElectroStaticEnergy(const dealii::DoFHandler<3> & dofHandler,
	                             const dealii::QGauss<3> & quadrature,
	                             const std::map<dealii::CellId, std::vector<double> > &rhoValues,
				     const dealii::parallel::distributed::Vector<double> & phiTotRhoOut,
				     const dealii::parallel::distributed::Vector<double> & phiExt,
				     const std::vector<std::vector<double> > & localVselfs,
				     const std::map<dealii::types::global_dof_index, double> & atoms)
   {
      dealii::FEValues<3> fe_values (dofHandler.get_fe(), quadrature, dealii::update_values | dealii::update_JxW_values);
      const unsigned int num_quad_points = quadrature.size();

      std::vector<double> cellPhiTotRho(num_quad_points);
      std::vector<double> cellPhiExt(num_quad_points);
      double electrostaticEnergyTotPot = 0.0;


      //parallel loop over all elements
      typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();

      for(; cell!=endc; ++cell)
	  if(cell->is_locally_owned())
	    {
	       // Compute values for current cell.
	      fe_values.reinit(cell);
	      fe_values.get_function_values(phiTotRhoOut,cellPhiTotRho);
	      fe_values.get_function_values(phiExt,cellPhiExt);
	      for(unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
		{
		  // Vtot, Vext computet with rhoOut
		  double Vtot=cellPhiTotRho[q_point];
		  double Vext=cellPhiExt[q_point];
		  electrostaticEnergyTotPot+=0.5*(Vtot)*(rhoValues.find(cell->id())->second[q_point])*fe_values.JxW(q_point);
		}

	    }

      //
      //get nuclear electrostatic energy 0.5*sum_I*(Z_I*phi_tot(RI) - VselfI(RI))
      //

      //First evaluate sum_I*(Z_I*phi_tot(RI)) on atoms belonging to current processor
      double phiContribution = 0.0,vSelfContribution=0.0;
      for (std::map<dealii::types::global_dof_index, double>::const_iterator it=atoms.begin(); it!=atoms.end(); ++it)
	  phiContribution += (-it->second)*phiTotRhoOut(it->first);//-charge*potential

      //Then evaluate sum_I*(Z_I*Vself_I(R_I)) on atoms belonging to current processor
      for(unsigned int i = 0; i < localVselfs.size(); ++i)
	  vSelfContribution += (-localVselfs[i][0])*(localVselfs[i][1]);//-charge*potential

      const double nuclearElectrostaticEnergy = 0.5*(phiContribution - vSelfContribution);

      return electrostaticEnergyTotPot + nuclearElectrostaticEnergy;
   }

}

template<unsigned int FEOrder>
double dftClass<FEOrder>::computeElectrostaticEnergyPRefined()
{
#define FEOrder_PRefined FEOrder+3
   if (dftParameters::verbosity>=2)
        pcout<< std::endl<<"-----------------Re computing electrostatics on p refined mesh with polynomial order: "<<FEOrder_PRefined <<"---------------"<<std::endl;
   d_mesh.resetParallelMeshMovedToUnmoved();
   dealii::parallel::distributed::Triangulation<3> & tria = d_mesh.getParallelMeshMoved();

   dealii::DoFHandler<3> dofHandlerPRefined;
   dofHandlerPRefined.initialize(tria,dealii::FE_Q<3>(dealii::QGaussLobatto<1>(FEOrder_PRefined+1)));
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
#ifdef ENABLE_PERIODIC_BC
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

   for (unsigned int i = 0; i < std::accumulate(periodic.begin(),periodic.end(),0); ++i)
         GridTools::collect_periodic_faces(dofHandlerPRefined, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vector2,offsetVectors[i]);

   dealii::DoFTools::make_periodicity_constraints<dealii::DoFHandler<3> >(periodicity_vector2, constraintsPRefined);
#endif
   constraintsPRefined.close();

   moveMeshToAtoms(tria);

   dofHandlerPRefined.distribute_dofs (dofHandlerPRefined.get_fe());


   //matrix free data structure
   typename dealii::MatrixFree<3>::AdditionalData additional_data;
   additional_data.tasks_parallel_scheme = dealii::MatrixFree<3>::AdditionalData::partition_partition;

   //Zero Dirichlet BC constraints on the boundary of the domain
   //used for computing total electrostatic potential using Poisson problem
   //with (rho+b) as the rhs
   dealii::ConstraintMatrix constraintsForTotalPotential;
   constraintsForTotalPotential.reinit(locallyRelevantDofs);

#ifdef ENABLE_PERIODIC_BC
   locatePeriodicPinnedNodes(dofHandlerPRefined,
	                     constraintsPRefined,
	                     constraintsForTotalPotential);
#endif
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
   quadratureVector.push_back(QGauss<1>(FEOrder_PRefined+1));

   dealii::MatrixFree<3,double> matrixFreeDataPRefined;

   matrixFreeDataPRefined.reinit(matrixFreeDofHandlerVectorInput,
	                         matrixFreeConstraintsInputVector,
			         quadratureVector,
			         additional_data);


   std::map<dealii::types::global_dof_index, double> atomPRefinedNodeIdToChargeMap;
   locateAtomCoreNodes(dofHandlerPRefined,atomPRefinedNodeIdToChargeMap);

   //compute p refined quad density values from original eigenvectors (not p refined)
   dealii::QGauss<3>  quadraturePRefined(FEOrder_PRefined+1);
   dealii::FEValues<3> fe_values (dofHandlerEigen.get_fe(), quadraturePRefined, dealii::update_values | dealii::update_gradients);
   const unsigned int num_quad_points = quadraturePRefined.size();

   std::map<dealii::CellId, std::vector<double> > rhoQuadRefinedMeshValues;
   typename dealii::DoFHandler<3>::active_cell_iterator cellOld = dofHandlerEigen.begin_active(), endcOld = dofHandlerEigen.end();
   for(; cellOld!=endcOld; ++cellOld)
      if(cellOld->is_locally_owned())
      {
	  rhoQuadRefinedMeshValues[cellOld->id()] = std::vector<double>(num_quad_points);

	  fe_values.reinit (cellOld);
#ifdef ENABLE_PERIODIC_BC
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

	  for(unsigned int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
	      for(unsigned int i=0; i<numEigenValues; ++i)
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
						    (eigenValues[kPoint][i+numEigenValues],
						     fermiEnergy,
						     C_kb,
						     dftParameters::TVal);

		      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		      {
#ifdef ENABLE_PERIODIC_BC
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
              MPI_Allreduce(&rhoTemp[0], &rhoQuadRefinedMeshValues[cellOld->id()][0], numPoint, MPI_DOUBLE, MPI_SUM, interpoolcomm) ;
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
				rhoQuadRefinedMeshValues);

   if (dftParameters::verbosity==2)
        pcout<< std::endl<<"Solving for total electrostatic potential (rhoIn+b) on p refined mesh: ";
   dealiiCGSolver.solve(phiTotalSolverProblem,
			dftParameters::relLinearSolverTolerance,
			dftParameters::maxLinearSolverIterations,
			dftParameters::verbosity);

   //recompute all elctron electrostatic energy
   return Utilities::MPI::sum(internal::computeLocalAllElectronElectroStaticEnergy
                                           (dofHandlerPRefined,
	                                    quadraturePRefined,
	                                    rhoQuadRefinedMeshValues,
				            phiTotRhoOutPRefined,
				            phiExtPRefined,
				            localVselfsPRefined,
				            atomPRefinedNodeIdToChargeMap), mpi_communicator);
}
