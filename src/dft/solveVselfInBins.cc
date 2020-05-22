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
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das
//

#include <dealiiLinearSolver.h>
#include <poissonSolverProblem.h>
#include <constants.h>
#include <constraintMatrixInfo.h>
#ifdef DFTFE_WITH_GPU
#include <solveVselfInBinsCUDA.h>
#endif

namespace dftfe
{
	template<unsigned int FEOrder>
		void vselfBinsManager<FEOrder>::solveVselfInBins
		(const dealii::MatrixFree<3,double> & matrix_free_data,
		 const unsigned int offset,
		 const dealii::AffineConstraints<double> & hangingPeriodicConstraintMatrix,
		 const std::vector<std::vector<double> > & imagePositions,
		 const std::vector<int> & imageIds,
		 const std::vector<double> &imageCharges,
		 std::vector<std::vector<double> > & localVselfs)
		{
			localVselfs.clear();
			d_vselfFieldBins.clear();
			const unsigned int numberBins = d_boundaryFlagOnlyChargeId.size();
			const unsigned int numberGlobalCharges = d_atomLocations.size();

			//set up poisson solver
			dealiiLinearSolver dealiiCGSolver(mpi_communicator,dealiiLinearSolver::CG);
			poissonSolverProblem<FEOrder> vselfSolverProblem(mpi_communicator);

			std::map<dealii::types::global_dof_index, dealii::Point<3> > supportPoints;
			dealii::DoFTools::map_dofs_to_support_points(dealii::MappingQ1<3,3>(), matrix_free_data.get_dof_handler(offset), supportPoints);

			std::map<dealii::types::global_dof_index, int>::iterator iterMap;
			std::map<dealii::types::global_dof_index, double>::iterator iterMapVal;
			d_vselfFieldBins.resize(numberBins);
			for(unsigned int iBin = 0; iBin < numberBins; ++iBin)
			{
				double init_time;
				MPI_Barrier(MPI_COMM_WORLD);
				init_time = MPI_Wtime();

				const unsigned int constraintMatrixId = iBin + offset;
				distributedCPUVec<double> vselfBinScratch;
				matrix_free_data.initialize_dof_vector(vselfBinScratch,constraintMatrixId);
				vselfBinScratch = 0;

				std::map<dealii::types::global_dof_index,dealii::Point<3> >::iterator iterNodalCoorMap;
				std::map<dealii::types::global_dof_index, double> & vSelfBinNodeMap = d_vselfBinField[iBin];

				//
				//set initial guess to vSelfBinScratch
				//
				for(iterNodalCoorMap = supportPoints.begin(); iterNodalCoorMap != supportPoints.end(); ++iterNodalCoorMap)
					if(vselfBinScratch.in_local_range(iterNodalCoorMap->first)
							&& !d_vselfBinConstraintMatrices[iBin].is_constrained(iterNodalCoorMap->first))
					{
						iterMapVal = vSelfBinNodeMap.find(iterNodalCoorMap->first);
						if(iterMapVal != vSelfBinNodeMap.end())
							vselfBinScratch(iterNodalCoorMap->first) = iterMapVal->second;
					}


				vselfBinScratch.compress(dealii::VectorOperation::insert);
				d_vselfBinConstraintMatrices[iBin].distribute(vselfBinScratch);

				MPI_Barrier(MPI_COMM_WORLD);
				init_time = MPI_Wtime() - init_time;
				if (dftParameters::verbosity>=1)
					pcout<<" Time taken for vself field initialization for current bin: "<<init_time<<std::endl;

				double vselfinit_time;
				MPI_Barrier(MPI_COMM_WORLD);
				vselfinit_time = MPI_Wtime();
				//
				//call the poisson solver to compute vSelf in current bin
				//
				vselfSolverProblem.reinit(matrix_free_data,
						vselfBinScratch,
						d_vselfBinConstraintMatrices[iBin],
						constraintMatrixId,
						d_atomsInBin[iBin],
						true,
						iBin==0?true:false);

				MPI_Barrier(MPI_COMM_WORLD);
				vselfinit_time = MPI_Wtime() - vselfinit_time;
				if (dftParameters::verbosity>=1)
					pcout<<" Time taken for vself solver problem init for current bin: "<<vselfinit_time<<std::endl;

				dealiiCGSolver.solve(vselfSolverProblem,
						dftParameters::absLinearSolverTolerance,
						dftParameters::maxLinearSolverIterations,
						dftParameters::verbosity);


				//
				//store Vselfs for atoms in bin
				//
				for(std::map<dealii::types::global_dof_index, double>::iterator it = d_atomsInBin[iBin].begin(); it != d_atomsInBin[iBin].end(); ++it)
				{
					std::vector<double> temp(2,0.0);
					temp[0] = it->second;//charge;
					temp[1] = vselfBinScratch(it->first);//vself
					if (dftParameters::verbosity>=4)
						std::cout<< "(only for debugging: peak value of Vself: "<< temp[1] << ")" <<std::endl;

					localVselfs.push_back(temp);
				}
				//
				//store solved vselfBinScratch field
				//
				d_vselfFieldBins[iBin]=vselfBinScratch;
			}//bin loop
		}

#ifdef DFTFE_WITH_GPU
	template<unsigned int FEOrder>
		void vselfBinsManager<FEOrder>::solveVselfInBinsGPU
		(const dealii::MatrixFree<3,double> & matrix_free_data,
		 const unsigned int offset,
		 operatorDFTCUDAClass & operatorMatrix,
		 const dealii::AffineConstraints<double> & hangingPeriodicConstraintMatrix,
		 const std::vector<std::vector<double> > & imagePositions,
		 const std::vector<int> & imageIds,
		 const std::vector<double> &imageCharges,
		 std::vector<std::vector<double> > & localVselfs)
		{
			localVselfs.clear();
			d_vselfFieldBins.clear();
			//
			const unsigned int numberBins = d_boundaryFlagOnlyChargeId.size();
			const unsigned int numberGlobalCharges = d_atomLocations.size();


			d_vselfFieldBins.resize(numberBins);
			for(unsigned int iBin = 0; iBin < numberBins; ++iBin)
				matrix_free_data.initialize_dof_vector(d_vselfFieldBins[iBin],iBin + offset);

			const unsigned int localSize=d_vselfFieldBins[0].local_size();
			std::vector<double> vselfBinsFieldsFlattened(localSize*numberBins,0.0);
			std::vector<double> rhsFlattened(localSize*numberBins,0.0);

			const dealii::DoFHandler<3> & dofHandler=matrix_free_data.get_dof_handler();
			dealii::QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
			const unsigned int   dofs_per_cell = dofHandler.get_fe().dofs_per_cell;
			const unsigned int   num_quad_points = quadrature.size();

			MPI_Barrier(MPI_COMM_WORLD);
			double time = MPI_Wtime(); 

			//
			// compute rhs for each bin and store in rhsFlattened
			//
			for(unsigned int iBin = 0; iBin < numberBins; ++iBin)
			{ 
				//rhs contribution from static condensation of dirichlet boundary conditions
				const unsigned int constraintMatrixId = iBin + offset;

				distributedCPUVec<double> tempvec;
				matrix_free_data.initialize_dof_vector(tempvec,constraintMatrixId);
				tempvec=0.0;

				distributedCPUVec<double> rhs;
				rhs.reinit(tempvec);
				rhs=0;

				d_vselfBinConstraintMatrices[iBin].distribute(tempvec);
				tempvec.update_ghost_values();

				dealii::FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>()> fe_eval(matrix_free_data,
						constraintMatrixId,
						0);
				dealii::VectorizedArray<double>  quarter = dealii::make_vectorized_array (1.0/(4.0*M_PI));
				for (unsigned int macrocell = 0;macrocell < matrix_free_data.n_macro_cells();
						++macrocell)
				{
					fe_eval.reinit(macrocell);
					fe_eval.read_dof_values_plain(tempvec);
					fe_eval.evaluate(false,true);
					for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
					{
						fe_eval.submit_gradient(-quarter*fe_eval.get_gradient(q), q);
					}
					fe_eval.integrate(false, true);
					fe_eval.distribute_local_to_global(rhs);
				}


				//rhs contribution from atomic charge at fem nodes
				for (std::map<dealii::types::global_dof_index, double>::const_iterator it=d_atomsInBin[iBin].begin(); it!=d_atomsInBin[iBin].end(); ++it)
				{
					std::vector<dealii::ConstraintMatrix::size_type> local_dof_indices_origin(1, it->first); //atomic node
					dealii::Vector<double> cell_rhs_origin (1);
					cell_rhs_origin(0)=-(it->second); //atomic charge

					d_vselfBinConstraintMatrices[iBin].distribute_local_to_global(cell_rhs_origin, local_dof_indices_origin, rhs);
				}

				//MPI operation to sync data
				rhs.compress(dealii::VectorOperation::add);

				//FIXME: check if this is really required
				d_vselfBinConstraintMatrices[iBin].set_zero(rhs);

				for(unsigned int i = 0; i < localSize; ++i)
					rhsFlattened[i*numberBins+iBin]=rhs.local_element(i);
			}

			//
			// compute diagonal
			//
			distributedCPUVec<double> diagonalA;
			matrix_free_data.initialize_dof_vector(diagonalA,0);
			diagonalA=0;


			dealii::FEValues<3> fe_values (dofHandler.get_fe(), quadrature, dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
			dealii::Vector<double>  elementalDiagonalA(dofs_per_cell);
			std::vector<dealii::types::global_dof_index> local_dof_indices (dofs_per_cell);

			typename dealii::DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
			for(; cell!=endc; ++cell)
				if (cell->is_locally_owned())
				{
					fe_values.reinit (cell);

					cell->get_dof_indices (local_dof_indices);

					elementalDiagonalA=0.0;
					for (unsigned int i = 0; i < dofs_per_cell; ++i)
						for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
							elementalDiagonalA(i) += (1.0/(4.0*M_PI))*(fe_values.shape_grad(i, q_point)*fe_values.shape_grad (i, q_point))*fe_values.JxW(q_point);

					hangingPeriodicConstraintMatrix.distribute_local_to_global(elementalDiagonalA,
							local_dof_indices,
							diagonalA);
				}

			diagonalA.compress(dealii::VectorOperation::add);

			for(dealii::types::global_dof_index i = 0; i < diagonalA.size(); ++i)
				if(diagonalA.in_local_range(i))
					if(!hangingPeriodicConstraintMatrix.is_constrained(i))
						diagonalA(i) = 1.0/diagonalA(i);

			diagonalA.compress(dealii::VectorOperation::insert);

			const unsigned int ghostSize   = matrix_free_data.get_vector_partitioner()->n_ghost_indices();

			std::vector<double> inhomoIdsColoredVecFlattened((localSize+ghostSize)*numberBins,1.0);
			for (unsigned int i = 0; i <(localSize+ghostSize); ++i)
			{
				const dealii::types::global_dof_index globalNodeId=matrix_free_data.get_vector_partitioner()->local_to_global(i);
				for(unsigned int iBin = 0; iBin < numberBins; ++iBin)
				{
					if( d_vselfBinConstraintMatrices[iBin].is_inhomogeneously_constrained(globalNodeId)
							&& d_vselfBinConstraintMatrices[iBin].get_constraint_entries(globalNodeId)->size()==0)
						inhomoIdsColoredVecFlattened[i*numberBins+iBin]=0.0;
					//if( d_vselfBinConstraintMatrices[iBin].is_inhomogeneously_constrained(globalNodeId))
					//    inhomoIdsColoredVecFlattened[i*numberBins+iBin]=0.0;
				}
			}

			MPI_Barrier(MPI_COMM_WORLD);
			time = MPI_Wtime() - time;
			if (dftParameters::verbosity >= 2 && this_mpi_process==0)
				std::cout<<"Solve vself in bins: time for compute rhs and diagonal: "<<time<<std::endl;

			MPI_Barrier(MPI_COMM_WORLD);
			time = MPI_Wtime(); 
			//
			// GPU poisson solve 
			//
			poissonCUDA::solveVselfInBins
				(operatorMatrix,
				 matrix_free_data,
				 hangingPeriodicConstraintMatrix,
				 &rhsFlattened[0],
				 diagonalA.begin(),
				 &inhomoIdsColoredVecFlattened[0],
				 localSize,
				 ghostSize,
				 numberBins,
				 mpi_communicator,
				 &vselfBinsFieldsFlattened[0]);

			MPI_Barrier(MPI_COMM_WORLD);
			time = MPI_Wtime() - time;
			if (dftParameters::verbosity >= 2 && this_mpi_process==0)
				std::cout<<"Solve vself in bins: time for poissonCUDA::solveVselfInBins : "<<time<<std::endl; 

			MPI_Barrier(MPI_COMM_WORLD);
			time = MPI_Wtime(); 

			for(unsigned int iBin = 0; iBin < numberBins; ++iBin)
			{
				//
				//store solved vselfBinScratch field
				//
				for(unsigned int i = 0; i < localSize; ++i)
					d_vselfFieldBins[iBin].local_element(i)=vselfBinsFieldsFlattened[numberBins*i+iBin];

				const unsigned int constraintMatrixId = iBin + offset;

				dftUtils::constraintMatrixInfo constraintsMatrixDataInfo;
				constraintsMatrixDataInfo.initialize(matrix_free_data.get_vector_partitioner(constraintMatrixId),
						d_vselfBinConstraintMatrices[iBin]);


				constraintsMatrixDataInfo.precomputeMaps(matrix_free_data.get_vector_partitioner(constraintMatrixId),
						matrix_free_data.get_vector_partitioner(constraintMatrixId),
						1);


				//d_vselfBinConstraintMatrices[iBin].distribute(d_vselfFieldBins[iBin]);
				d_vselfFieldBins[iBin].update_ghost_values();
				constraintsMatrixDataInfo.distribute(d_vselfFieldBins[iBin],1);

				//
				//store Vselfs for atoms in bin
				//
				for(std::map<dealii::types::global_dof_index, double>::iterator it = d_atomsInBin[iBin].begin(); it != d_atomsInBin[iBin].end(); ++it)
				{
					std::vector<double> temp(2,0.0);
					temp[0] = it->second;//charge;
					temp[1] = d_vselfFieldBins[iBin](it->first);//vself
					if (dftParameters::verbosity>=4)
						std::cout<< "(only for debugging: peak value of Vself: "<< temp[1] <<")" <<std::endl;

					localVselfs.push_back(temp);
				}

			}//bin loop

			MPI_Barrier(MPI_COMM_WORLD);
			time = MPI_Wtime() - time;
			if (dftParameters::verbosity >= 2 && this_mpi_process==0)
				std::cout<<"Solve vself in bins: time for updating d_vselfFieldBins : "<<time<<std::endl; 
		}
#endif
}
