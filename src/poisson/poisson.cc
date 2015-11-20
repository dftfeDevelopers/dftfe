#include "../../include/poisson.h"
#include "boundary.cc"

//constructor
poissonClass::poissonClass(dftClass* _dftPtr):
  dftPtr(_dftPtr),
  FE (QGaussLobatto<1>(FEOrder+1)),
  relaxation(1.0),
  jacobianDiagonalValue(1.0),
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
  unsigned int numCells=dftPtr->triangulation.n_locally_owned_active_cells();
  dealii::IndexSet numDofs=dftPtr->locally_relevant_dofs;
  //intialize the size of Table storing element level jacobians
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      localJacobians[cell->id()]=std::vector<double>(FE.dofs_per_cell*FE.dofs_per_cell);
    }
  }

  //constraints
  //no constraints
  constraintsNone.clear ();
  DoFTools::make_hanging_node_constraints (dftPtr->dofHandler, constraintsNone);
  constraintsNone.close();

  //zero constraints
  constraintsZero.clear ();
  VectorTools::interpolate_boundary_values (dftPtr->dofHandler, 0, ZeroFunction<3>(), constraintsZero);
  constraintsZero.close ();

  //OnebyR constraints
  constraints1byR.clear ();
  VectorTools::interpolate_boundary_values (dftPtr->dofHandler, 0, OnebyRBoundaryFunction<3>(dftPtr->atomLocations),constraints1byR);
  constraints1byR.close ();

  //initialize vectors
  dftPtr->matrix_free_data.initialize_dof_vector (rhs);
  rhs2.reinit (rhs);
  Ax.reinit (rhs);
  phiTotRhoIn.reinit (rhs);
  phiTotRhoOut.reinit (rhs);
  phiExt.reinit (rhs);
  //store constrianed DOF's
  for (types::global_dof_index i=0; i<rhs.size(); i++){
    if (rhs.locally_owned_elements().is_element(i)){
      if (constraints1byR.is_constrained(i)){
	values1byR[i] = constraints1byR.get_inhomogeneity(i);
	valuesZero[i] = 0.0;
      }
    }
  }
  //compute elemental jacobians
  computeLocalJacobians();
  localJacobiansPtr=&localJacobians;
  computing_timer.exit_section("poissonClass setup"); 
}

//compute local jacobians
void poissonClass::computeLocalJacobians(){
  computing_timer.enter_section("poissonClass jacobian assembly"); 
  rhs2=0.0;

  //local data structures
  QGauss<3>  quadrature(FEOrder+1);
  FEValues<3> fe_values(FE, quadrature, update_values | update_gradients | update_JxW_values);
  const unsigned int dofs_per_cell = FE.dofs_per_cell;
  const unsigned int num_quad_points = quadrature.size();
  Vector<double>       elementalrhs (dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      double* jacobianPtr=&localJacobians[cell->id()][0];
      //compute values for the current element
      fe_values.reinit (cell);
      elementalrhs=0.0;
      //local poissonClass operator
      for (unsigned int i=0; i<dofs_per_cell; ++i){
	for (unsigned int j=0; j<dofs_per_cell; ++j){
	   jacobianPtr[i*dofs_per_cell+j]=0.0;
	   for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	     jacobianPtr[i*dofs_per_cell+j] += (1.0/(4.0*M_PI))*(fe_values.shape_grad(i, q_point)*fe_values.shape_grad (j, q_point))*fe_values.JxW(q_point);
	   }
	}
      }
    
      cell->get_dof_indices (local_dof_indices);
      //zero out columns and rows corresponding to Dirchlet BC DOF
      for (unsigned int i=0; i<dofs_per_cell; ++i){
	unsigned int rowID=local_dof_indices[i];
	for (unsigned int j=0; j<dofs_per_cell; ++j){
	  unsigned int columnID=local_dof_indices[j];
	  if (values1byR.find(columnID)!=values1byR.end()){
	    elementalrhs(i)+=values1byR.find(columnID)->second*jacobianPtr[i*dofs_per_cell+j];
	  }
	  if ((valuesZero.find(rowID)!= valuesZero.end()) || (valuesZero.find(columnID)!=valuesZero.end())){
	    jacobianPtr[i*dofs_per_cell+j]=0.0;
	  }
	}
      }
      constraintsNone.distribute_local_to_global(elementalrhs, local_dof_indices, rhs2);
    }
  }
  rhs2.compress(VectorOperation::add);
  computing_timer.exit_section("poissonClass jacobian assembly");
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
      constraintsNone.distribute_local_to_global(elementalResidual, local_dof_indices, rhs);
    }
  }
  //Add nodal force to the node at the origin
  for (std::map<unsigned int, double>::iterator it=dftPtr->atoms.begin(); it!=dftPtr->atoms.end(); ++it){
    std::vector<unsigned int> local_dof_indices_origin(1, it->first); //atomic node
    Vector<double> cell_rhs_origin (1); 
    cell_rhs_origin(0)=-(it->second); //atomic charge
    constraintsNone.distribute_local_to_global(cell_rhs_origin, local_dof_indices_origin, rhs);
  }
  //MPI operation to sync data 
  rhs.compress(VectorOperation::add);
  //Set RHS values corresponding to Dirichlet BC's
  std::map<dealii::types::global_dof_index, double>* valuesBC;
  if (rhoValues) valuesBC=&valuesZero;
  else valuesBC=&values1byR;
  for (std::map<dealii::types::global_dof_index, double>::const_iterator it=valuesBC->begin(); it!=valuesBC->end(); ++it){
    rhs(it->first)=it->second*jacobianDiagonalValue;
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
  const unsigned int   dofs_per_cell = FE.dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  typename DoFHandler<3>::active_cell_iterator cell;

  std::vector<double> x(dofs_per_cell), y(dofs_per_cell);
  //loop over all "cells"  (cell blocks)
  for (unsigned int cell_index=cell_range.first; cell_index<cell_range.second; ++cell_index){
    //loop over cells
    for (unsigned int v=0; v<data.n_components_filled(cell_index); ++v){
      cell=data.get_cell_iterator(cell_index, v);
      cell->get_dof_indices (local_dof_indices);
      src.extract_subvector_to(local_dof_indices, x);
      //elemental Ax
      char trans= 'N';
      int m= dofs_per_cell, n= dofs_per_cell, lda= dofs_per_cell, incx= 1, incy= 1;
      double alpha= 1.0, beta= 0.0;
      dgemv_(&trans,&m,&n,&alpha,&((*localJacobiansPtr)[cell->id()][0]),&lda,&x[0],&incx,&beta,&y[0],&incy);
      constraintsNone.distribute_local_to_global(y, local_dof_indices, dst);
    }
  }
}

//vmult
void poissonClass::vmult(vectorType &dst, const vectorType &src) const{
  dst=0.0;
  vectorType x2;
  x2.reinit (rhs);
  x2=src;
  constraintsNone.distribute(x2);
  dftPtr->matrix_free_data.cell_loop (&poissonClass::AX, this, dst, x2);
  dst.compress(VectorOperation::add);

  //apply Dirichlet BC's
  for (std::map<types::global_dof_index, double>::const_iterator it=valuesZero.begin(); it!=valuesZero.end(); ++it){
    dst(it->first) = src(it->first)*jacobianDiagonalValue;
  }
}

//jacobi preconditioning
void poissonClass::precondition_Jacobi(vectorType& dst, const vectorType& src, const double omega) const{
}

//solve using CG
void poissonClass::solve(vectorType& phi, std::map<dealii::CellId,std::vector<double> >* rhoValues){
  //compute RHS
  computeRHS(rhoValues);
  //solve
  computing_timer.enter_section("poissonClass solve"); 
  SolverControl solver_control(maxLinearSolverIterations,relLinearSolverTolerance*rhs.l2_norm());
  SolverCG<vectorType> solver(solver_control);
  //PreconditionJacobi<poissonClass> preconditioner;
  //relaxation=0.6;
  try{
    solver.solve(*this, phi, rhs, IdentityMatrix(rhs.size()));
    constraintsNone.distribute(phi);
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
