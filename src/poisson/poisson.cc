#include "../../include/poisson.h"

//constructor
template <int dim>
poisson<dim>::poisson(DoFHandler<dim>* _dofHandler):
  dofHandler(_dofHandler), 
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times)
{
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

//residual and matrix-vector product computation
template <int dim>
void poisson<dim>::computeRHS (const MatrixFree<dim,double> &data,
			       vectorType &dst,
			       const vectorType &src,
			       const std::pair<unsigned int,unsigned int> &cell_range) const
{
  //initialize FE evaluation structure with finite elment order, quadrature rule and related data
  FEEvaluation<dim,FEOrder,FEOrder+1,dim,double> vals(data);

  //loop over all elements
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell){
    //read U values for this cell
    vals.reinit (cell);

    //residual and  jacobian
    if (updateRHSValue){
      //residual
      vals.read_dof_values_plain(src);
      vals.evaluate (false, false, false);
      //loop over quadrature points
      for (unsigned int q=0; q<vals.n_q_points; ++q){
	vals.submit_value((*rhoValues)(cell, q));
      }
      vals.integrate(true, false);
    }
    else{
      //jacobian
      Tensor<1, dim, gradType> phix = vals.get_gradient(q);
      vals.read_dof_values(src);
      vals.evaluate (false,true,false);
      //loop over quadrature points
      for (unsigned int q=0; q<vals.n_q_points; ++q){
	vals.submit_gradient((1.0/(4.0*M_PI))*phix,q);
      }
      vals.integrate(false, true);
    }
    vals.distribute_local_to_global(dst);
  }
}

//update residual
template <int dim>
void poisson<dim>::updateRHS (vectorType& solution, 
			      vectorType& residual){
  updateRHSValue=true;
  //initialize residuals to zero
  residual=0.0;
  //loop over all cells to compute residuals
  data.cell_loop (&poisson<dim>::computeRHS, this, residual, solution);
  updateRHSValue=false;

  //Add nodal force to the node at the origin
  for (std::map<unsigned int, double>::iterator it=atoms.begin(); it!=atoms.end(); ++it){
    std::vector<unsigned int> local_dof_indices_origin(1, it->first); //atomic node
    Vector<double> cell_rhs_origin (1); 
    cell_rhs_origin(0)=-(it->second); //atomic chrage
    constraints.distribute_local_to_global(cell_rhs_origin, local_dof_indices_origin, residual);
    //pcout << " node: " << it->first << " charge: " << it->second << std::endl;
  }

}

//matrix free data structure vmult operations.
template <int dim>
void poisson<dim>::vmult (vectorType &dst, const vectorType &src) const{
  dst=0.0;
  data.cell_loop (&poisson<dim>::computeRHS, this, dst, src);
 
 //Account for dirichlet BC's
  const std::vector<unsigned int>& constrained_dofs = data.get_constrained_dofs();
  for (unsigned int i=0; i<constrained_dofs.size(); ++i){
    unsigned int index=data.get_vector_partitioner()->local_to_global(constrained_dofs[i]);
    dst(index) += src(index);
  }
}

//solve linear system of equations AX=b using iterative solver
template <int dim>
void poisson<dim>::solve (vectorType& solution, 
			  vectorType& residual,
			  ConstraintMatrix& constraints,
			  std::map<unsigned int, double>& atoms,
			  Table<2,double>* rhoValues=0
			  )
{
  //solve
  computing_timer.enter_section("poisson solve"); 
  
  updateRHS(solution, residual);
  //cgSolve(U,residualU);
  SolverControl solver_control(maxSolverIterations, relSolverTolerance*residual.l2_norm());
  solverType<vectorType> cg(solver_control);
  try{
    cg.solve(*this, solution, residual, IdentityMatrix(solution.size()));
  }
  catch (...) {
    pcout << "\nWarning: solver did not converge as per set tolerances. consider increasing maxSolverIterations or decreasing relSolverTolerance.\n";
  }
  char buffer[200];
  sprintf(buffer, "initial residual:%12.6e, current residual:%12.6e, nsteps:%u, tolerance criterion:%12.6e\n", solver_control.initial_value(), solver_control.last_value(), solver_control.last_step(), solver_control.tolerance());
  pcout<<buffer;
 
  //
  computing_timer.exit_section("poisson solve"); 
}
