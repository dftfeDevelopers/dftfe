#include "../include/poisson.h"

//constructor
template <int dim>
poisson<dim>::poisson(DoFHandler<dim>* _dofHandler):
  FE (QGaussLobatto<1>(FEOrder+1)),
  dofHandler(_dofHandler), 
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times)
{
  //initialize poisson object
  init();
}

//initialize poisson object
template <int dim>
void poisson<dim>::init(){
  computing_timer.enter_section("poisson setup"); 
  locally_owned_dofs = dofHandler->locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (*dofHandler, locally_relevant_dofs);

  //constraints
  constraints.clear ();
  constraints.reinit (locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints (*dofHandler, constraints);
  constraints.close ();

  //initialize vectors and jacobian matrix using the sparsity pattern.
  solution.reinit (locally_owned_dofs, mpi_communicator); solution=0;
  residual.reinit (locally_owned_dofs, mpi_communicator); residual=0;
  CompressedSimpleSparsityPattern csp (locally_relevant_dofs);
  DoFTools::make_sparsity_pattern (*dofHandler, csp, constraints, false);
  SparsityTools::distribute_sparsity_pattern (csp,
					      dofHandler->n_locally_owned_dofs_per_processor(),
					      mpi_communicator,
					      locally_relevant_dofs);
  jacobian.reinit (locally_owned_dofs, locally_owned_dofs, csp, mpi_communicator);
  computing_timer.exit_section("poisson setup"); 
}

//assemble poisson jacobian and residual
template <int dim>
void poisson<dim>::assemble(){
  computing_timer.enter_section("poisson assembly"); 
  //initialize global data structures
  jacobian=0.0;
  residual=0.0;
  solution=0.0;
  
  //local data structures
  QGauss<dim>  quadrature(quadratureRule);
  FEValues<dim> fe_values (FE, quadrature, update_values | update_gradients | update_JxW_values);
  const unsigned int   dofs_per_cell = FE.dofs_per_cell;
  const unsigned int   num_quad_points = quadrature.size();
  FullMatrix<double>   elementalJacobian (dofs_per_cell, dofs_per_cell);
  Vector<double>       elementalResidual (dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  
  //parallel loop over all elements
  typename DoFHandler<dim>::active_cell_iterator cell = dofHandler->begin_active(), endc = dofHandler->end();
  unsigned int cellID=0;
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      elementalJacobian = 0;
      elementalResidual = 0;
      cell->set_user_index(cellID++);

      //compute values for the current element
      fe_values.reinit (cell);
      
      //local poisson operator
      for (unsigned int i=0; i<dofs_per_cell; ++i){
	for (unsigned int j=0; j<dofs_per_cell; ++j){
	  for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	    elementalJacobian(i,j) += (1.0/(4.0*M_PI))*(fe_values.shape_grad (i, q_point) *
							fe_values.shape_grad (j, q_point) *
							fe_values.JxW (q_point));
	  }
	}
      }
      
      //local rhs
      for (unsigned int i=0; i<dofs_per_cell; ++i){
	for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	  elementalResidual(i) += 0.0; //fe_values.shape_value(i, q_point)*(*rhoInValues)(cellID, q_point)*fe_values.JxW (q_point);
	}
      }
      
      //assemble to global data structures
      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global(elementalJacobian, 
					     elementalResidual,
					     local_dof_indices,
					     jacobian, 
					     residual);
      
    }
  }
  //MPI operation to sync data 
  residual.compress(VectorOperation::add);
  jacobian.compress(VectorOperation::add);
  
  computing_timer.exit_section("poisson assembly"); 
}

//solve linear system of equations AX=b using iterative solver
template <int dim>
void poisson<dim>::solve()
{
  //assemble
  assemble();
  
  //solve
  computing_timer.enter_section("poisson solve"); 
  PETScWrappers::MPI::Vector  completely_distributed_solutionInc (locally_owned_dofs, mpi_communicator);
  SolverControl solver_control(maxLinearSolverIterations, relLinearSolverTolerance*residual.l2_norm());
  PETScWrappers::SolverCG solver(solver_control, mpi_communicator);
  PETScWrappers::PreconditionJacobi preconditioner(jacobian);
  try{
    solver.solve (jacobian, completely_distributed_solutionInc, residual, preconditioner);
    char buffer[200];
    sprintf(buffer, 
	    "linear system solved in %3u iterations\n",
	    solver_control.last_step());
    pcout << buffer;
  }
  catch (...) {
    pcout << "\nWarning: solver did not converge in "
	  << solver_control.last_step()
	  << " iterations as per set tolerances. consider increasing maxSolverIterations or decreasing relSolverTolerance.\n";     
  }
  constraints.distribute (completely_distributed_solutionInc);
  solution=completely_distributed_solutionInc; 
  computing_timer.exit_section("poisson solve"); 
}
