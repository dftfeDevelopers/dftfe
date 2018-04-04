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

#include "../../include/poisson.h"
#include "../../include/dft.h"
#include "../../include/constants.h"
#include "../../include/dftParameters.h"
//
//constructor
//
template<unsigned int FEOrder>
poissonClass<FEOrder>::poissonClass(dftClass<FEOrder>* _dftPtr,const MPI_Comm &mpi_comm_replica):
  dftPtr(_dftPtr),
  FE (QGaussLobatto<1>(FEOrder+1)),
  mpi_communicator (mpi_comm_replica),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::never, TimerOutput::wall_times)
{

}

//
//initialize poissonClass 
//
template<unsigned int FEOrder>
void poissonClass<FEOrder>::init()
{
  computing_timer.enter_section("poissonClass setup"); 

  //
  //initialize vectors
  //
  dftPtr->matrix_free_data.initialize_dof_vector(rhs);

  rhs2.reinit (rhs);
  jacobianDiagonal.reinit (rhs);
  phiTotRhoIn.reinit (rhs);
  phiTotRhoOut.reinit (rhs);
  //phiExt.reinit (rhs);
  //vselfBinScratch.reinit (rhs);
  
  computing_timer.exit_section("poissonClass setup"); 
}

//
//compute local jacobians
//
template<unsigned int FEOrder>
void poissonClass<FEOrder>::computeRHS2()
{
  computing_timer.enter_section("PoissonClass rhs2 assembly"); 
  rhs2=0.0;
  //jacobianDiagonal=0.0;
  //
  //local data structures
  //
  
  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values(FE, quadrature, update_values | update_gradients | update_JxW_values);
  const unsigned int dofs_per_cell = FE.dofs_per_cell;
  const unsigned int num_quad_points = quadrature.size();
  Vector<double>  elementalrhs (dofs_per_cell);//, elementalJacobianDiagonal(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
  const ConstraintMatrix * constraintMatrix = dftPtr->d_constraintsVector[d_constraintMatrixId];

  //
  //parallel loop over all elements
  //
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  for(; cell!=endc; ++cell) 
    {
      if(cell->is_locally_owned())
	{
	  //compute values for the current element
	  fe_values.reinit(cell);
	  cell->get_dof_indices(local_dof_indices);

	  //rhs2
	  elementalrhs=0.0;
	  bool assembleFlag=false;

	  //local poissonClass operator
	  for(unsigned int j = 0; j < dofs_per_cell; ++j)
	    {
	      unsigned int columnID = local_dof_indices[j];
	      if(constraintMatrix->is_inhomogeneously_constrained(columnID))
		{
		  for (unsigned int i = 0; i < dofs_per_cell; ++i)
		    {
		      //compute contribution to rhs2
		      double localStiffnessMatIJ = 0.0;
		      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
			{
			  localStiffnessMatIJ += (1.0/(4.0*M_PI))*(fe_values.shape_grad(i,q_point)*fe_values.shape_grad(j,q_point))*fe_values.JxW(q_point);
			}
		      elementalrhs(i)+=constraintMatrix->get_inhomogeneity(columnID)*localStiffnessMatIJ;
		      if (!assembleFlag) {assembleFlag=true;}
		    }

		}
	    }
	  if(assembleFlag)
	    {
	      constraintMatrix->distribute_local_to_global(elementalrhs,local_dof_indices,rhs2);
	    }
	  
	}
    }
  rhs2.compress(VectorOperation::add);
  //pcout << "rhs2: " << rhs2.l2_norm() << std::endl;
  computing_timer.exit_section("PoissonClass rhs2 assembly");
}

//
//compute RHS
//
template<unsigned int FEOrder>
void poissonClass<FEOrder>::computeRHS(std::map<dealii::CellId,std::vector<double> >* rhoValues){
  if(!rhoValues)
    computeRHS2();

  computing_timer.enter_section("PoissonClass rhs assembly");
  rhs = 0.0;
  jacobianDiagonal=0.0;

  //
  //local data structures
  //
  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (FE, quadrature, update_values | update_gradients | update_JxW_values);
  const unsigned int   dofs_per_cell = FE.dofs_per_cell;
  const unsigned int   num_quad_points = quadrature.size();
  Vector<double>       elementalResidual (dofs_per_cell), elementalJacobianDiagonal(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  const ConstraintMatrix * constraintMatrix = dftPtr->d_constraintsVector[d_constraintMatrixId];

 
  //
  //parallel loop over all elements
  //
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  for(; cell!=endc; ++cell) 
    {
      if (cell->is_locally_owned())
	{
	  //compute values for the current element
	  fe_values.reinit (cell);
	  elementalResidual=0.0;
	  //local rhs
	  if (rhoValues) 
	    {
	      double* rhoValuesPtr=&((*rhoValues)[cell->id()][0]);
	      for (unsigned int i=0; i<dofs_per_cell; ++i)
		{
		  for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		    { 
		      elementalResidual(i) += fe_values.shape_value(i, q_point)*rhoValuesPtr[q_point]*fe_values.JxW (q_point);
		    }
		}
	    }

	  //assemble to global data structures
	  cell->get_dof_indices (local_dof_indices);
	  constraintMatrix->distribute_local_to_global(elementalResidual, local_dof_indices, rhs);

	  //jacobianDiagonal
	  elementalJacobianDiagonal=0.0;
	  for (unsigned int i = 0; i < dofs_per_cell; ++i)
	    {
	      for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
		{
		  elementalJacobianDiagonal(i) += (1.0/(4.0*M_PI))*(fe_values.shape_grad(i, q_point)*fe_values.shape_grad (i, q_point))*fe_values.JxW(q_point);
		}
	    }

	  constraintMatrix->distribute_local_to_global(elementalJacobianDiagonal, local_dof_indices, jacobianDiagonal);

	}
    }

  //
  //Add nodal force to the node containing the atom
  //
  if(rhoValues)
    {
      for (std::map<unsigned int, double>::iterator it=dftPtr->atoms.begin(); it!=dftPtr->atoms.end(); ++it)
	{
	  std::vector<ConstraintMatrix::size_type> local_dof_indices_origin(1, it->first); //atomic node
	  Vector<double> cell_rhs_origin (1); 
	  cell_rhs_origin(0)=-(it->second); //atomic charge

	  constraintMatrix->distribute_local_to_global(cell_rhs_origin, local_dof_indices_origin, rhs);
	}
    }
  else
    {
      int binId = d_constraintMatrixId - 2;
      for (std::map<unsigned int, double>::iterator it=dftPtr->d_atomsInBin[binId].begin(); it!=dftPtr->d_atomsInBin[binId].end(); ++it)
	{
	  std::vector<ConstraintMatrix::size_type> local_dof_indices_origin(1, it->first); //atomic node
	  Vector<double> cell_rhs_origin (1); 
	  cell_rhs_origin(0)=-(it->second); //atomic charge
	  constraintMatrix->distribute_local_to_global(cell_rhs_origin, local_dof_indices_origin, rhs);
	}
    }

  //
  //MPI operation to sync data 
  //
  rhs.compress(VectorOperation::add);
  jacobianDiagonal.compress(VectorOperation::add);

  
  if (!rhoValues)
    {
      rhs.add(-1.0,rhs2);
    }

  //
  //check if this is really required
  //
  constraintMatrix->set_zero(rhs);


  for(types::global_dof_index i = 0; i < jacobianDiagonal.size(); ++i)
    {
      if(jacobianDiagonal.in_local_range(i))
	{
	  if(!constraintMatrix->is_constrained(i))
	    {   
	      jacobianDiagonal(i) = 1.0/jacobianDiagonal(i);
	    }
	}
    }

  jacobianDiagonal.compress(VectorOperation::insert);

  
  //pcout<< "rhs:" <<rhs.l2_norm()<<std::endl;
  computing_timer.exit_section("PoissonClass rhs assembly");

}

//Ax
template<unsigned int FEOrder>
void poissonClass<FEOrder>::AX (const dealii::MatrixFree<3,double>  &data,
				 vectorType &dst, 
				 const vectorType &src,
				 const std::pair<unsigned int,unsigned int> &cell_range) const
{
  VectorizedArray<double>  quarter = make_vectorized_array (1.0/(4.0*M_PI));
  int constraintId = d_constraintMatrixId;

  FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>()> fe_eval(data, constraintId, 0); 

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

//vmult
template<unsigned int FEOrder>
void poissonClass<FEOrder>::vmult(vectorType &dst, vectorType &src) const
{
 
  dst=0.0;

  /*
  dftPtr->d_constraintsVector[d_constraintMatrixId]->distribute(src);
  
  for(types::global_dof_index i = 0; i < src.size(); ++i)
    {
      if(src.in_local_range(i))
	{	 	  
	  if(dftPtr->d_constraintsVector[d_constraintMatrixId]->is_inhomogeneously_constrained(i))
	    {
	      src(i) -= dftPtr->d_constraintsVector[d_constraintMatrixId]->get_inhomogeneity(i);
	    }
	}
    }
  */ 
  dftPtr->matrix_free_data.cell_loop (&poissonClass<FEOrder>::AX, this, dst, src);

  //This is necessary specifically for periodic boundary conditions 
  //for solving total electrostatic potential with pinned nodes
  //Only master node is pinned and remaining slave nodes are still
  //constrained to master node and setting other slave nodes
  //to zero is necessary after every "Ax" and hence call this function.
  // (Only necessary for dealiiOpt)
  //dftPtr->d_constraintsVector[d_constraintMatrixId]->set_zero(dst);  

}

//
//Matrix-Free Jacobi preconditioner application
//
template<unsigned int FEOrder>
void poissonClass<FEOrder>::precondition_Jacobi(vectorType& dst, const vectorType& src, const double omega) const
{
  dst = src;
  dst.scale(jacobianDiagonal);
}

//
//solve using CG
//
template<unsigned int FEOrder>
void poissonClass<FEOrder>::solve(vectorType& phi, int constraintMatrixId, std::map<dealii::CellId,std::vector<double> >* rhoValues)
{
  //
  //initialize the data member
  //
  d_constraintMatrixId = constraintMatrixId;

  //
  //compute RHS
  //
  computeRHS(rhoValues);

  //solve
  computing_timer.enter_section("poissonClass solve"); 
  SolverControl solver_control(dftParameters::maxLinearSolverIterations,dftParameters::relLinearSolverTolerance*rhs.l2_norm());
  SolverCG<vectorType> solver(solver_control);

  
  PreconditionJacobi<poissonClass<FEOrder> > preconditioner;
  preconditioner.initialize (*this, 0.3);
  try{

    //phi=0.0;
    //solver.solve(*this, phi, rhs, IdentityMatrix(rhs.size()));

    //assumed phi distributed prior
    /*
    for(types::global_dof_index i = 0; i < phi.size(); ++i)
    {
      if(phi.in_local_range(i))
	{	    
	  if(dftPtr->d_constraintsVector[d_constraintMatrixId]->is_inhomogeneously_constrained(i))
	    phi(i)=0;
	}
    }   
    */
    phi.update_ghost_values();
    solver.solve(*this, phi, rhs, preconditioner);
    //phi.update_ghost_values();
    dftPtr->d_constraintsVector[d_constraintMatrixId]->distribute(phi);
    phi.update_ghost_values();
  }
  catch (...) {
    pcout << "\nWarning: solver did not converge as per set tolerances. consider increasing maxLinearSolverIterations or decreasing relLinearSolverTolerance.\n";
  }

  //std::cout<<"L2 norm of phi : "<<phi.l2_norm()<<std::endl;
 // std::cout<<"Max of Phi : "<<phi.linfty_norm()<<std::endl;


  if (dftParameters::verbosity==2)
  {
    pcout<<std::endl;	  
    char buffer[200];
    sprintf(buffer, "initial abs. residual: %12.6e, current abs. residual: %12.6e, nsteps: %u, abs. tolerance criterion: %12.6e\n\n", \
	  solver_control.initial_value(),				\
	  solver_control.last_value(),					\
	  solver_control.last_step(), solver_control.tolerance()); 
    pcout<<buffer;
  }
  computing_timer.exit_section("poissonClass solve"); 
}

template class poissonClass<1>;
template class poissonClass<2>;
template class poissonClass<3>;
template class poissonClass<4>;
template class poissonClass<5>;
template class poissonClass<6>;
template class poissonClass<7>;
template class poissonClass<8>;
template class poissonClass<9>;
template class poissonClass<10>;
template class poissonClass<11>;
template class poissonClass<12>;
