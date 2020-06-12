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

#include <poissonSolverProblem.h>
#include <constants.h>

namespace dftfe {
	//
	//constructor
	//
	template<unsigned int FEOrder>
		poissonSolverProblem<FEOrder>::poissonSolverProblem(const  MPI_Comm &mpi_comm):
			mpi_communicator (mpi_comm),
			n_mpi_processes (dealii::Utilities::MPI::n_mpi_processes(mpi_comm)),
			this_mpi_process (dealii::Utilities::MPI::this_mpi_process(mpi_comm)),
			pcout (std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
	{
		d_isShapeGradIntegralPrecomputed=false;
		d_isMeanValueConstraintComputed=false;
	}

	template<unsigned int FEOrder>
		void poissonSolverProblem<FEOrder>::reinit
		(const dealii::MatrixFree<3,double> & matrixFreeData,
		 distributedCPUVec<double> & x,
		 const dealii::ConstraintMatrix & constraintMatrix,
		 const unsigned int matrixFreeVectorComponent,
		 const std::map<dealii::types::global_dof_index, double> & atoms,
		 const std::map<dealii::CellId,std::vector<double> > & smearedChargeValues,
		 const std::map<dealii::CellId,std::vector<double> > & rhoValues,
		 const bool isComputeDiagonalA,
		 const bool isComputeMeanValueConstraint,
		 const bool smearedNuclearCharges,
		 const bool isPrecomputeShapeGradIntegral,
		 const bool isRhoValues)
		{
			int this_process;
			MPI_Comm_rank(mpi_communicator, &this_process);
			MPI_Barrier(mpi_communicator);
			double time=MPI_Wtime();

			d_matrixFreeDataPtr=&matrixFreeData;
			d_xPtr=&x;
			d_constraintMatrixPtr=&constraintMatrix;
			d_matrixFreeVectorComponent=matrixFreeVectorComponent;
			d_rhoValuesPtr=isRhoValues?&rhoValues:NULL;
			d_atomsPtr=smearedNuclearCharges?NULL:&atoms;
			d_smearedChargeValuesPtr=smearedNuclearCharges?&smearedChargeValues:NULL;

			if (isComputeMeanValueConstraint)
			{
				computeMeanValueConstraint();
				d_isMeanValueConstraintComputed=true;
			}

			if (isComputeDiagonalA)
				computeDiagonalA();

			if (isPrecomputeShapeGradIntegral)
				precomputeShapeFunctionGradientIntegral();        
		}


	template<unsigned int FEOrder>
		void poissonSolverProblem<FEOrder>::reinit
		(const dealii::MatrixFree<3,double> & matrixFreeData,
		 distributedCPUVec<double> & x,
		 const dealii::ConstraintMatrix & constraintMatrix,
		 const unsigned int matrixFreeVectorComponent,
		 const std::map<dealii::types::global_dof_index, double> & atoms,
		 const bool isComputeDiagonalA,
		 const bool isPrecomputeShapeGradIntegral)
		{
			d_matrixFreeDataPtr=&matrixFreeData;
			d_xPtr=&x;
			d_constraintMatrixPtr=&constraintMatrix;
			d_matrixFreeVectorComponent=matrixFreeVectorComponent;
			d_rhoValuesPtr=NULL;
			d_atomsPtr=&atoms;

			if (isComputeDiagonalA)
				computeDiagonalA();

			if (isPrecomputeShapeGradIntegral)
				precomputeShapeFunctionGradientIntegral();

		}


	template<unsigned int FEOrder>
		void poissonSolverProblem<FEOrder>::distributeX()
		{
			d_constraintMatrixPtr->distribute(*d_xPtr);

			if (d_isMeanValueConstraintComputed)
				meanValueConstraintDistribute(*d_xPtr);
		}

	template<unsigned int FEOrder>
		distributedCPUVec<double> & poissonSolverProblem<FEOrder>::getX()
		{
			return *d_xPtr;
		}

	template<unsigned int FEOrder>
		void poissonSolverProblem<FEOrder>::precomputeShapeFunctionGradientIntegral()
		{

			const dealii::DoFHandler<3> & dofHandler=
				d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);

			dealii::QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
			dealii::FEValues<3> fe_values (dofHandler.get_fe(), quadrature, dealii::update_gradients | dealii::update_JxW_values);
			const unsigned int   dofs_per_cell = dofHandler.get_fe().dofs_per_cell;
			const unsigned int   num_quad_points = quadrature.size();

			d_cellShapeFunctionGradientIntegralFlattened.clear();
			d_cellShapeFunctionGradientIntegralFlattened.resize(d_matrixFreeDataPtr->n_physical_cells()*dofs_per_cell*dofs_per_cell);

			typename dealii::DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
			unsigned int iElem=0;
			for(; cell!=endc; ++cell)
				if(cell->is_locally_owned())
				{
					fe_values.reinit(cell);

					for(unsigned int j = 0; j < dofs_per_cell; ++j)
					{
						for (unsigned int i = 0; i < dofs_per_cell; ++i)
						{
							double shapeFunctionGradientIntegralValue = 0.0;
							for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
								shapeFunctionGradientIntegralValue +=(fe_values.shape_grad(i,q_point)*fe_values.shape_grad(j,q_point))*fe_values.JxW(q_point);

							d_cellShapeFunctionGradientIntegralFlattened[iElem*dofs_per_cell*dofs_per_cell
								+j*dofs_per_cell+i]=shapeFunctionGradientIntegralValue;
						}

					}

					iElem++;
				}
			d_isShapeGradIntegralPrecomputed=true;
		}

	template<unsigned int FEOrder>
		void poissonSolverProblem<FEOrder>::computeRhs(distributedCPUVec<double>  & rhs)
		{

			rhs.reinit(*d_xPtr);
			rhs=0;

			const dealii::DoFHandler<3> & dofHandler=
				d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);

			dealii::QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
			//dealii::FEValues<3> fe_values (dofHandler.get_fe(), quadrature,
			//                               d_isShapeGradIntegralPrecomputed?dealii::update_values | dealii::update_JxW_values
			//                                                               :dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);

			dealii::FEValues<3> fe_values (dofHandler.get_fe(), quadrature,dealii::update_values | dealii::update_JxW_values);
			const unsigned int   dofs_per_cell = dofHandler.get_fe().dofs_per_cell;
			const unsigned int   num_quad_points = quadrature.size();
			dealii::Vector<double>  elementalRhs(dofs_per_cell);
			std::vector<dealii::types::global_dof_index> local_dof_indices (dofs_per_cell);
			typename dealii::DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();


			distributedCPUVec<double> tempvec;
			tempvec.reinit(rhs);
			tempvec=0.0;
			d_constraintMatrixPtr->distribute(tempvec);
			tempvec.update_ghost_values();

			dealii::FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>()> fe_eval(*d_matrixFreeDataPtr,
					d_matrixFreeVectorComponent,
					0);
			dealii::VectorizedArray<double>  quarter = dealii::make_vectorized_array (1.0/(4.0*M_PI));
			if (d_constraintMatrixPtr->has_inhomogeneities())
				for (unsigned int macrocell = 0;macrocell < d_matrixFreeDataPtr->n_macro_cells();
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

			//rhs contribution from electronic charge
			if (d_rhoValuesPtr)
			{
				cell = dofHandler.begin_active();
				for(; cell!=endc; ++cell)
					if (cell->is_locally_owned())
					{
						fe_values.reinit (cell);
						elementalRhs=0.0;

						const std::vector<double>& tempVec=d_rhoValuesPtr->find(cell->id())->second;
						for (unsigned int i=0; i<dofs_per_cell; ++i)
							for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
								elementalRhs(i) += fe_values.shape_value(i, q_point)*tempVec[q_point]*fe_values.JxW (q_point);

						//assemble to global data structures
						cell->get_dof_indices (local_dof_indices);
						d_constraintMatrixPtr->distribute_local_to_global(elementalRhs, local_dof_indices, rhs);
					}
			}

			//rhs contribution from atomic charge at fem nodes
			if (d_atomsPtr!=NULL)
				for (std::map<dealii::types::global_dof_index, double>::const_iterator it=(*d_atomsPtr).begin(); it!=(*d_atomsPtr).end(); ++it)
				{
					std::vector<dealii::ConstraintMatrix::size_type> local_dof_indices_origin(1, it->first); //atomic node
					dealii::Vector<double> cell_rhs_origin (1);
					cell_rhs_origin(0)=-(it->second); //atomic charge

					d_constraintMatrixPtr->distribute_local_to_global(cell_rhs_origin, local_dof_indices_origin, rhs);
				}
			else if (d_smearedChargeValuesPtr!=NULL)
			{
				const unsigned int   num_quad_points_sc = d_matrixFreeDataPtr->get_quadrature(4).size();
				dealii::FEValues<3> fe_valuesSC (dofHandler.get_fe(), d_matrixFreeDataPtr->get_quadrature(4),dealii::update_values | dealii::update_JxW_values);        
				cell = dofHandler.begin_active();
				for(; cell!=endc; ++cell)
					if (cell->is_locally_owned())
					{
						fe_valuesSC.reinit (cell);
						elementalRhs=0.0;

						const std::vector<double>& tempVec=d_smearedChargeValuesPtr->find(cell->id())->second;
						for (unsigned int i=0; i<dofs_per_cell; ++i)
							for (unsigned int q_point=0; q_point<num_quad_points_sc; ++q_point)
								elementalRhs(i) += fe_valuesSC.shape_value(i, q_point)*tempVec[q_point]*fe_valuesSC.JxW (q_point);

						//assemble to global data structures
						cell->get_dof_indices (local_dof_indices);
						d_constraintMatrixPtr->distribute_local_to_global(elementalRhs, local_dof_indices, rhs);
					}        
			}

			//MPI operation to sync data
			rhs.compress(dealii::VectorOperation::add);

			if (d_isMeanValueConstraintComputed)
				meanValueConstraintDistributeSlaveToMaster(rhs);

			//FIXME: check if this is really required
			d_constraintMatrixPtr->set_zero(rhs);

		}

	//Matrix-Free Jacobi preconditioner application
	template<unsigned int FEOrder>
		void  poissonSolverProblem<FEOrder>::precondition_Jacobi(distributedCPUVec<double>& dst,
				const distributedCPUVec<double>& src,
				const double omega) const
		{
			dst = src;
			dst.scale(d_diagonalA);
		}

	// Compute and fill value at mean value constrained dof
	// u_o= -\sum_{i \neq o} a_i * u_i where i runs over all dofs
	// except the mean value constrained dof (o^{th}) 
	template<unsigned int FEOrder>
		void poissonSolverProblem<FEOrder>::meanValueConstraintDistribute(distributedCPUVec<double>& vec) const
		{
			// -\sum_{i \neq o} a_i * u_i computation which involves summation across MPI tasks
			const double constrainedNodeValue=d_meanValueConstraintVec*vec;

			// mean value constrained node is in the root task id 
			if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==0)
				vec[d_meanValueConstraintNodeId]=constrainedNodeValue;
		}

	// Distribute value at mean value constrained dof (u_o) to all other dofs
	// u_i+= -a_i * u_o, and subsequently set u_o to 0 
	template<unsigned int FEOrder>
		void poissonSolverProblem<FEOrder>::meanValueConstraintDistributeSlaveToMaster(distributedCPUVec<double>& vec) const
		{
			double constrainedNodeValue=0;
			if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==0)
				constrainedNodeValue=vec[d_meanValueConstraintNodeId];

			// broadcast value at mean value constraint dof in root task id to all other tasks ids
			MPI_Bcast(&constrainedNodeValue,
					1,
					MPI_DOUBLE,
					0,
					mpi_communicator);

			vec.add(constrainedNodeValue,d_meanValueConstraintVec);

			meanValueConstraintSetZero(vec);
		}

	template<unsigned int FEOrder>
		void poissonSolverProblem<FEOrder>::meanValueConstraintSetZero(distributedCPUVec<double>& vec) const
		{
			if (d_isMeanValueConstraintComputed)
				if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==0)
					vec[d_meanValueConstraintNodeId]=0;
		}

	//
	// Compute mean value constraint which is required in case of fully periodic
	// boundary conditions
	template<unsigned int FEOrder>
		void poissonSolverProblem<FEOrder>::computeMeanValueConstraint()
		{
			// allocate parallel distibuted vector to store mean value constraint
			d_meanValueConstraintVec.reinit(*d_xPtr);
			d_meanValueConstraintVec=0;

			const dealii::DoFHandler<3> & dofHandler=
				d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);

			dealii::QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
			dealii::FEValues<3> fe_values (dofHandler.get_fe(), quadrature, dealii::update_values| dealii::update_JxW_values);
			const unsigned int   dofs_per_cell = dofHandler.get_fe().dofs_per_cell;
			const unsigned int   num_quad_points = quadrature.size();
			dealii::Vector<double>  elementalValues(dofs_per_cell);
			std::vector<dealii::types::global_dof_index> local_dof_indices (dofs_per_cell);

			//parallel loop over all elements
			typename dealii::DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
			for(; cell!=endc; ++cell)
				if (cell->is_locally_owned())
				{
					fe_values.reinit (cell);

					cell->get_dof_indices (local_dof_indices);

					elementalValues=0.0;
					for (unsigned int i = 0; i < dofs_per_cell; ++i)
						for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
							elementalValues(i) += fe_values.shape_value (i, q_point)*fe_values.JxW(q_point);

					d_constraintMatrixPtr->distribute_local_to_global(elementalValues,
							local_dof_indices,
							d_meanValueConstraintVec);
				}

			//MPI operation to sync data
			d_meanValueConstraintVec.compress(dealii::VectorOperation::add);

			dealii::IndexSet locallyOwnedElements=d_meanValueConstraintVec.locally_owned_elements();

			dealii::IndexSet locallyRelevantElements;
			dealii::DoFTools::extract_locally_relevant_dofs(dofHandler, locallyRelevantElements);

			//pick mean value constrained node in the zeroth processor such that it is not part
			//of periodic and hanging node constraint equations (both slave and master node).
			//This is done for simplicity of implementation.
			dealii::IndexSet allIndicesTouchedByConstraints(d_meanValueConstraintVec.size());
			for (dealii::IndexSet::ElementIterator it=locallyRelevantElements.begin();
					it<locallyRelevantElements.end();it++)
				if (d_constraintMatrixPtr->is_constrained(*it))
				{
					const dealii::types::global_dof_index lineDof = *it;
					const std::vector<std::pair<dealii::types::global_dof_index, double > > * rowData
						=d_constraintMatrixPtr->get_constraint_entries(lineDof);
					allIndicesTouchedByConstraints.add_index(lineDof);
					for(unsigned int j = 0; j < rowData->size();++j)
						allIndicesTouchedByConstraints.add_index((*rowData)[j].first);
				}

			if (d_atomsPtr)
				for (std::map<dealii::types::global_dof_index, double>::const_iterator it=(*d_atomsPtr).begin(); it!=(*d_atomsPtr).end(); ++it)
					allIndicesTouchedByConstraints.add_index(it->first);

			locallyOwnedElements.subtract_set(allIndicesTouchedByConstraints);
			d_meanValueConstraintNodeId=*locallyOwnedElements.begin();
			AssertThrow(!d_constraintMatrixPtr->is_constrained(d_meanValueConstraintNodeId),dealii::ExcMessage("DFT-FE Error: Mean value constraint creation bug."));


			double valueAtConstraintNode=d_meanValueConstraintVec[d_meanValueConstraintNodeId];
			MPI_Bcast(&valueAtConstraintNode,
					1,
					MPI_DOUBLE,
					0,
					mpi_communicator);

			d_meanValueConstraintVec/=-valueAtConstraintNode;
			if (dealii::Utilities::MPI::this_mpi_process(mpi_communicator) ==0)
				d_meanValueConstraintVec[d_meanValueConstraintNodeId]=0;
		}


	template<unsigned int FEOrder>
		void poissonSolverProblem<FEOrder>::computeDiagonalA()
		{
			d_diagonalA.reinit(*d_xPtr);
			d_diagonalA=0;

			const dealii::DoFHandler<3> & dofHandler=
				d_matrixFreeDataPtr->get_dof_handler(d_matrixFreeVectorComponent);

			dealii::QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
			dealii::FEValues<3> fe_values (dofHandler.get_fe(), quadrature, dealii::update_values | dealii::update_gradients | dealii::update_JxW_values);
			const unsigned int   dofs_per_cell = dofHandler.get_fe().dofs_per_cell;
			const unsigned int   num_quad_points = quadrature.size();
			dealii::Vector<double>  elementalDiagonalA(dofs_per_cell);
			std::vector<dealii::types::global_dof_index> local_dof_indices (dofs_per_cell);

			//parallel loop over all elements
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

					d_constraintMatrixPtr->distribute_local_to_global(elementalDiagonalA,
							local_dof_indices,
							d_diagonalA);
				}

			//MPI operation to sync data
			d_diagonalA.compress(dealii::VectorOperation::add);

			for(dealii::types::global_dof_index i = 0; i < d_diagonalA.size(); ++i)
				if(d_diagonalA.in_local_range(i))
					if(! d_constraintMatrixPtr->is_constrained(i))
						d_diagonalA(i) = 1.0/d_diagonalA(i);

			d_diagonalA.compress(dealii::VectorOperation::insert);
		}

	//Ax
	template<unsigned int FEOrder>
		void poissonSolverProblem<FEOrder>::AX (const dealii::MatrixFree<3,double>  &matrixFreeData,
				distributedCPUVec<double> &dst,
				const distributedCPUVec<double> &src,
				const std::pair<unsigned int,unsigned int> &cell_range) const
		{
			dealii::VectorizedArray<double>  quarter = dealii::make_vectorized_array (1.0/(4.0*M_PI));

			dealii::FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>()> fe_eval(matrixFreeData,
					d_matrixFreeVectorComponent,
					0);

			for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
			{
				fe_eval.reinit(cell);
				fe_eval.read_dof_values(src);
				fe_eval.evaluate(false,true,false);
				for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
				{
					fe_eval.submit_gradient(fe_eval.get_gradient(q)*quarter, q);
				}
				fe_eval.integrate(false, true);
				fe_eval.distribute_local_to_global(dst);
			}
		}


	template<unsigned int FEOrder>
		void poissonSolverProblem<FEOrder>::vmult(distributedCPUVec<double> &Ax,const distributedCPUVec<double> &x) const
		{
			Ax=0.0;

			if (d_isMeanValueConstraintComputed)
			{
				distributedCPUVec<double> tempVec=x;
				meanValueConstraintDistribute(tempVec);

				d_matrixFreeDataPtr->cell_loop (&poissonSolverProblem<FEOrder>::AX, this, Ax, tempVec);

				meanValueConstraintDistributeSlaveToMaster(Ax);
			}
			else
				d_matrixFreeDataPtr->cell_loop (&poissonSolverProblem<FEOrder>::AX, this, Ax, x);
		}


	template class poissonSolverProblem<1>;
	template class poissonSolverProblem<2>;
	template class poissonSolverProblem<3>;
	template class poissonSolverProblem<4>;
	template class poissonSolverProblem<5>;
	template class poissonSolverProblem<6>;
	template class poissonSolverProblem<7>;
	template class poissonSolverProblem<8>;
	template class poissonSolverProblem<9>;
	template class poissonSolverProblem<10>;
	template class poissonSolverProblem<11>;
	template class poissonSolverProblem<12>;
	template class poissonSolverProblem<13>;
	template class poissonSolverProblem<14>;
	template class poissonSolverProblem<15>;
	template class poissonSolverProblem<16>;
}
