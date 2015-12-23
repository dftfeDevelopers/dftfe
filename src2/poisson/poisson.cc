#include "../../include/poisson.h"
#include "boundary.cc"

//constructor
poissonClass::poissonClass(dftClass* _dftPtr):
  dftPtr(_dftPtr),
  FE (QGaussLobatto<1>(FEOrder+1)),
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times)
{
}

//initialize poissonClass 
void poissonClass::init(){
  computing_timer.enter_section("poissonClass setup"); 

  //OnebyR constraints (temporarily created to fill values1byR map)
  ConstraintMatrix constraints1byR;
  constraints1byR.clear ();  
  VectorTools::interpolate_boundary_values (dftPtr->dofHandler, 0, OnebyRBoundaryFunction<3>(dftPtr->atomLocations),constraints1byR);
  constraints1byR.close ();

  //initialize vectors
  dftPtr->matrix_free_data.initialize_dof_vector (rhs);
  rhs2.reinit (rhs);
  jacobianDiagonal.reinit (rhs);
  phiTotRhoIn.reinit (rhs);
  phiTotRhoOut.reinit (rhs);
  phiExt.reinit (rhs);
  //store constrained DOF's
  for (types::global_dof_index i=0; i<rhs.size(); i++){
    if (rhs.locally_owned_elements().is_element(i)){
      if (constraints1byR.is_constrained(i)){
	values1byR[i] = constraints1byR.get_inhomogeneity(i);
      }
    }
  }
  computing_timer.exit_section("poissonClass setup"); 
  //compute RHS2
  computeRHS2();
}

//compute local jacobians
void poissonClass::computeRHS2(){
  computing_timer.enter_section("poissonClass rhs2 assembly"); 
  rhs2=0.0;

  //local data structures
  QGauss<3>  quadrature(FEOrder+1);
  FEValues<3> fe_values(FE, quadrature, update_values | update_gradients | update_JxW_values);
  const unsigned int dofs_per_cell = FE.dofs_per_cell;
  const unsigned int num_quad_points = quadrature.size();
  Vector<double>  elementalrhs (dofs_per_cell), elementalJacobianDiagonal(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      //compute values for the current element
      fe_values.reinit (cell);
      cell->get_dof_indices (local_dof_indices);
      //rhs2
      elementalrhs=0.0;
      bool assembleFlag=false;
      //local poissonClass operator
      for (unsigned int j=0; j<dofs_per_cell; ++j){
	unsigned int columnID=local_dof_indices[j];
	if (values1byR.find(columnID)!=values1byR.end()){
	  for (unsigned int i=0; i<dofs_per_cell; ++i){
	    //compute contribution to rhs2
	    double localJacobianIJ=0.0;
	    for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	      localJacobianIJ += (1.0/(4.0*M_PI))*(fe_values.shape_grad(i, q_point)*fe_values.shape_grad (j, q_point))*fe_values.JxW(q_point);
	    }
	    elementalrhs(i)+=values1byR.find(columnID)->second*localJacobianIJ;
	    if (!assembleFlag) {assembleFlag=true;}
	  }
	}
      }
      if (assembleFlag) {
	dftPtr->constraintsNone.distribute_local_to_global(elementalrhs, local_dof_indices, rhs2);
      }
      //jacobianDiagonal
      elementalJacobianDiagonal=0.0;
      for (unsigned int i=0; i<dofs_per_cell; ++i){
	for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	  elementalJacobianDiagonal(i) += (1.0/(4.0*M_PI))*(fe_values.shape_grad(i, q_point)*fe_values.shape_grad (i, q_point))*fe_values.JxW(q_point);
	}
      }
      dftPtr->constraintsNone.distribute_local_to_global(elementalJacobianDiagonal, local_dof_indices, jacobianDiagonal);
    }
  }
  rhs2.compress(VectorOperation::add);
  jacobianDiagonal.compress(VectorOperation::add);
  //remove zero entries of the jacobianDiagonal which occur at the hanging nodes
  for (unsigned int i=0; i<jacobianDiagonal.local_size(); i++){
    if (std::abs(jacobianDiagonal.local_element(i))<1.0e-15){
      jacobianDiagonal.local_element(i)=1.0;
    }
  }
  computing_timer.exit_section("poissonClass rhs2 assembly");
}

//compute RHS
void poissonClass::computeRHS(std::map<dealii::CellId,std::vector<double> >* rhoValues){
  computing_timer.enter_section("poissonClass rhs assembly");
  rhs=0.0;
  //local data structures
  QGauss<3>  quadrature(FEOrder+1);
  FEValues<3> fe_values (FE, quadrature, update_values | update_JxW_values);
  const unsigned int   dofs_per_cell = FE.dofs_per_cell;
  const unsigned int   num_quad_points = quadrature.size();
  Vector<double>       elementalResidual (dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      //compute values for the current element
      fe_values.reinit (cell);
      elementalResidual=0.0;
      //local rhs
      if (rhoValues) {
	double* rhoValuesPtr=&((*rhoValues)[cell->id()][0]);
	for (unsigned int i=0; i<dofs_per_cell; ++i){
	  for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){ 
	    elementalResidual(i) += fe_values.shape_value(i, q_point)*rhoValuesPtr[q_point]*fe_values.JxW (q_point);
	  }
	}
      }
      //assemble to global data structures
      cell->get_dof_indices (local_dof_indices);
      dftPtr->constraintsNone.distribute_local_to_global(elementalResidual, local_dof_indices, rhs);
    }
  }
  //Add nodal force to the node at the origin
  for (std::map<unsigned int, double>::iterator it=dftPtr->atoms.begin(); it!=dftPtr->atoms.end(); ++it){
    std::vector<unsigned int> local_dof_indices_origin(1, it->first); //atomic node
    Vector<double> cell_rhs_origin (1); 
    cell_rhs_origin(0)=-(it->second); //atomic charge
    dftPtr->constraintsNone.distribute_local_to_global(cell_rhs_origin, local_dof_indices_origin, rhs);
  }
  //MPI operation to sync data 
  rhs.compress(VectorOperation::add);
  //Set RHS values corresponding to Dirichlet BC's
  for (std::map<dealii::types::global_dof_index, double>::const_iterator it=values1byR.begin(); it!=values1byR.end(); ++it){
    if (rhoValues) rhs(it->first)=0.0;
    else rhs(it->first)=it->second*jacobianDiagonal(it->first);
  }
  rhs.update_ghost_values();
  if (!rhoValues){
    rhs.add(-1.0,rhs2);
  }
  computing_timer.exit_section("poissonClass rhs assembly");
}

//Ax
void poissonClass::AX (const dealii::MatrixFree<3,double>  &data,
		       vectorType &dst, 
		       const vectorType &src,
		       const std::pair<unsigned int,unsigned int> &cell_range) const{
  VectorizedArray<double>  quarter = make_vectorized_array (1.0/(4.0*M_PI));
  FEEvaluation<3,FEOrder>  fe_eval(data, 1, 0); //select constraintsZero constraints
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell){
    fe_eval.reinit (cell); 
    fe_eval.read_dof_values(src);
    fe_eval.evaluate (false,true,false);
    for (unsigned int q=0; q<fe_eval.n_q_points; ++q){
      fe_eval.submit_gradient (fe_eval.get_gradient(q)*quarter, q);
    }
    fe_eval.integrate (false, true);
    fe_eval.distribute_local_to_global (dst);
  }
}

//vmult
void poissonClass::vmult(vectorType &dst, const vectorType &src) const{
  dst=0.0;
  dftPtr->matrix_free_data.cell_loop (&poissonClass::AX, this, dst, src);
  dst.compress(VectorOperation::add);

  //apply Dirichlet BC's
  for (std::map<types::global_dof_index, double>::const_iterator it=values1byR.begin(); it!=values1byR.end(); ++it){
    dst(it->first) = src(it->first)*jacobianDiagonal(it->first);
  }
}

//Matrix-Free Jacobi preconditioner application
void poissonClass::precondition_Jacobi(vectorType& dst, const vectorType& src, const double omega) const{
  dst.ratio(src, jacobianDiagonal);
}

//solve using CG
void poissonClass::solve(vectorType& phi, std::map<dealii::CellId,std::vector<double> >* rhoValues){
  //compute RHS
  computeRHS(rhoValues);
  //solve
  computing_timer.enter_section("poissonClass solve"); 
  SolverControl solver_control(maxLinearSolverIterations,relLinearSolverTolerance*rhs.l2_norm());
  SolverCG<vectorType> solver(solver_control);
  PreconditionJacobi<poissonClass> preconditioner;
  preconditioner.initialize (*this, 0.6);
  try{
    phi=0.0;
    //solver.solve(*this, phi, rhs, IdentityMatrix(rhs.size()));
    solver.solve(*this, phi, rhs, preconditioner);
    dftPtr->constraintsNone.distribute(phi);
    phi.update_ghost_values();
  }
  catch (...) {
    pcout << "\nWarning: solver did not converge as per set tolerances. consider increasing maxLinearSolverIterations or decreasing relLinearSolverTolerance.\n";
  }
  char buffer[200];
  sprintf(buffer, "poisson solve: initial residual:%12.6e, current residual:%12.6e, nsteps:%u, tolerance criterion:%12.6e\n", \
	  solver_control.initial_value(),				\
	  solver_control.last_value(),					\
	  solver_control.last_step(), solver_control.tolerance()); 
  pcout<<buffer; 
  computing_timer.exit_section("poissonClass solve"); 
}
