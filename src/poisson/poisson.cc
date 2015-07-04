#include "../../include/poisson.h"

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
}

//initialize poisson object
template <int dim>
void poisson<dim>::init(){
  //intialize the size of Table storing element level jacobians
  localJacobians.reinit(n_locally_owned_active_cells, FE.dofs_per_cell, FE.dofs_per_cell);
}

//compute local jacobians
template <int dim>
void poisson<dim>::computeLocalJacobians(){
  computing_timer.enter_section("poisson local assembly"); 

  //local data structures
  QGauss<dim>  quadrature(quadratureRule);
  FEValues<dim> fe_values(FE, quadrature, update_values | update_gradients | update_JxW_values);
  const unsigned int dofs_per_cell = FE.dofs_per_cell;
  const unsigned int num_quad_points = quadrature.size();
  
  //parallel loop over all elements
  typename DoFHandler<dim>::active_cell_iterator cell = dofHandler->begin_active(), endc = dofHandler->end();
  unsigned int cellID=0;
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      //compute values for the current element
      fe_values.reinit (cell);
      
      //local poisson operator
      for (unsigned int i=0; i<dofs_per_cell; ++i){
	for (unsigned int j=0; j<dofs_per_cell; ++j){
	  localJacobians(cellID,i,j)=0.0;
	  for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	    localJacobians(cellID,i,j) += (1.0/(4.0*M_PI))*(fe_values.shape_grad (i, q_point) *
							fe_values.shape_grad (j, q_point) *
							fe_values.JxW (q_point));
	  }
	}
      }
      //increment cellID
      cellID++;
    }
  }
  computing_timer.exit_section("poisson local assembly");
}

//compute RHS
template <int dim>
void poisson<dim>::computeRHS(Table<2,double>* rhoValues){
  residual=0.0;
  //local data structures
  QGauss<dim>  quadrature(quadratureRule);
  FEValues<dim> fe_values (FE, quadrature, update_values | update_gradients | update_JxW_values);
  const unsigned int   dofs_per_cell = FE.dofs_per_cell;
  const unsigned int   num_quad_points = quadrature.size();
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  
  //parallel loop over all elements
  typename DoFHandler<dim>::active_cell_iterator cell = dofHandler->begin_active(), endc = dofHandler->end();
  unsigned int cellID=0;
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      //compute values for the current element
      fe_values.reinit (cell);
      
      //local rhs
      if (rhoValues) {
	for (unsigned int i=0; i<dofs_per_cell; ++i){
	  for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){ 
	    elementalResidual(i) += fe_values.shape_value(i, q_point)*(*rhoValues)(cellID, q_point)*fe_values.JxW (q_point);
	  }
	}
      }
      //assemble to global data structures
      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global(elementalResidual, local_dof_indices, residual);
      //increment cellID
      cellID++;
    }
  }
  //Add nodal force to the node at the origin
  for (std::map<unsigned int, double>::iterator it=atoms.begin(); it!=atoms.end(); ++it){
    std::vector<unsigned int> local_dof_indices_origin(1, it->first); //atomic node
    Vector<double> cell_rhs_origin (1); 
    cell_rhs_origin(0)=-(it->second); //atomic charge
    constraints.distribute_local_to_global(cell_rhs_origin, local_dof_indices_origin, residual);
  }
  //MPI operation to sync data 
  residual.compress(VectorOperation::add);
}

//compute RHS
template <int dim>
void poisson<dim>::vmult(Vector<double>       &dst,
			 const Vector<double> &src){
  dst=0.0;  
  data.cell_loop (&LaplaceOperator::local_apply, this, dst, src);

  const std::vector<unsigned int> &
    constrained_dofs = data.get_constrained_dofs();
  for (unsigned int i=0; i<constrained_dofs.size(); ++i)
    dst(constrained_dofs[i]) += src(constrained_dofs[i]);
}

//solve linear system of equations AX=b using iterative solver
template <int dim>
void poisson<dim>::solve(PETScWrappers::MPI::Vector& solution, 
			 PETScWrappers::MPI::Vector& residual,
			 PETScWrappers::MPI::SparseMatrix& jacobian,
			 ConstraintMatrix& constraints,
			 std::map<unsigned int, double>& atoms,
			 Table<2,double>* rhoValues
			 ){
  //assemble
  assemble(solution, residual, jacobian, constraints, atoms, rhoValues);
  
  //solve
  computing_timer.enter_section("poisson solve"); 
  SolverControl solver_control(maxLinearSolverIterations, relLinearSolverTolerance*residual.l2_norm());
  PETScWrappers::SolverCG solver(solver_control, mpi_communicator);
  PETScWrappers::PreconditionJacobi preconditioner(jacobian);
  PETScWrappers::MPI::Vector distributed_solution (solution);
  try{
    solver.solve (jacobian, distributed_solution, residual, preconditioner);
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
  constraints.distribute (distributed_solution);
  solution=distributed_solution; 
  computing_timer.exit_section("poisson solve"); 
}
