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
  constraintsNone.reinit (numDofs);
  DoFTools::make_hanging_node_constraints (dftPtr->dofHandler, constraintsNone);
  constraintsNone.close();

  //zero constraints
  constraintsZero.clear ();
  constraintsZero.reinit (numDofs);
  DoFTools::make_hanging_node_constraints (dftPtr->dofHandler, constraintsZero);
  VectorTools::interpolate_boundary_values (dftPtr->dofHandler, 0, ZeroFunction<3>(), constraintsZero);
  constraintsZero.close ();

  //OnebyR constraints
  constraints1byR.clear ();
  constraints1byR.reinit (numDofs);
  DoFTools::make_hanging_node_constraints (dftPtr->dofHandler, constraints1byR);
  VectorTools::interpolate_boundary_values (dftPtr->dofHandler, 0, OnebyRBoundaryFunction<3>(dftPtr->atomLocations),constraints1byR);
  constraints1byR.close ();

  //initialize vectors
  dftPtr->matrix_free_data.initialize_dof_vector (rhs);
  Ax.reinit (rhs);
  jacobianDiagonal.reinit (rhs);
  phiTotRhoIn.reinit (rhs);
  phiTotRhoOut.reinit (rhs);
  phiExt.reinit (rhs);
  //compute elemental jacobians
  computeLocalJacobians();
  localJacobiansPtr=&localJacobians;

  //store constrianed DOF's
  for (types::global_dof_index i=0; i<rhs.size(); i++){
    if (rhs.locally_owned_elements().is_element(i)){
      if (constraints1byR.is_constrained(i)){
	values1byR[i] = constraints1byR.get_inhomogeneity(i);
	valuesZero[i] = 0.0;
      }
    }
  }
}

//compute local jacobians
void poissonClass::computeLocalJacobians(){
  computing_timer.enter_section("poissonClass jacobian assembly"); 

  //local data structures
  QGauss<3>  quadrature(quadratureRule);
  FEValues<3> fe_values(FE, quadrature, update_values | update_gradients | update_JxW_values);
  const unsigned int dofs_per_cell = FE.dofs_per_cell;
  const unsigned int num_quad_points = quadrature.size();
  
  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      //compute values for the current element
      fe_values.reinit (cell);
      
      //local poissonClass operator
      for (unsigned int i=0; i<dofs_per_cell; ++i){
	for (unsigned int j=0; j<dofs_per_cell; ++j){
	  localJacobians[cell->id()][i*dofs_per_cell+j]=0.0;
	  for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	    localJacobians[cell->id()][i*dofs_per_cell+j] += (1.0/(4.0*M_PI))*(fe_values.shape_grad (i, q_point) *fe_values.shape_grad (j, q_point)*fe_values.JxW (q_point));
	  }
	}
      }
    }
  }
  computing_timer.exit_section("poissonClass jacobian assembly");
}

//compute RHS
void poissonClass::computeRHS(std::map<dealii::CellId,std::vector<double> >* rhoValues){
  computing_timer.enter_section("poissonClass rhs assembly");
  rhs=0.0;
  //local data structures
  QGauss<3>  quadrature(quadratureRule);
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
      
      //local rhs
      if (rhoValues) {
	for (unsigned int i=0; i<dofs_per_cell; ++i){
	  for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){ 
	    elementalResidual(i) += fe_values.shape_value(i, q_point)*((*rhoValues)[cell->id()][q_point])*fe_values.JxW (q_point);
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
      constraintsZero.distribute_local_to_global(y, local_dof_indices, dst);
    }
  }
}

//vmult
void poissonClass::vmult(vectorType &dst, const vectorType &src) const{
  dst=0.0;  
  dftPtr->matrix_free_data.cell_loop (&poissonClass::AX, this, dst, src);
  dst.compress(VectorOperation::add);

  //apply Dirichlet BC's
  for (std::map<types::global_dof_index, double>::const_iterator it=valuesZero.begin(); it!=valuesZero.end(); ++it){
    dst(it->first) += src(it->first);
  }
}

//solve using CG
void poissonClass::solve(std::map<dealii::CellId,std::vector<double> >* rhoValues){
  //compute RHS
  computeRHS(rhoValues);

  SolverControl solver_control(maxLinearSolverIterations,relLinearSolverTolerance*rhs.l2_norm());
  SolverCG<vectorType> solver(solver_control);
  try{
    solver.solve(*this, phiTotRhoOut, rhs, IdentityMatrix(rhs.size()));
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
  
}
