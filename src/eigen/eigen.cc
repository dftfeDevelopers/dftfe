
#include "../../include/eigen.h"

//constructor
template <int dim>
eigen<dim>::eigen(DoFHandler<dim>* _dofHandler):
  FE (QGaussLobatto<1>(FEOrder+1)),
  dofHandler(_dofHandler), 
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times)
{
}

//initialize eigen object
template <int dim>
void eigen<dim>::init(){}

//assemble poisson jacobian and residual
template <int dim>
void eigen<dim>::assemble(PETScWrappers::MPI::Vector& solution, 
			  PETScWrappers::MPI::SparseMatrix& massMatrix,
			  PETScWrappers::MPI::SparseMatrix& hamiltonianMatrix,
			  PETScWrappers::MPI::Vector& massVector, 
			  ConstraintMatrix& constraints,
			  Table<2,double>* rhoValues
			  ){
  computing_timer.enter_section("eigen assembly"); 
  //initialize global data structures
  massMatrix=0.0; hamiltonianMatrix=0.0; massVector=0.0;

  //local data structures
  QGaussLobatto<dim> quadratureM(quadratureRule);
  QGauss<dim>  quadratureH(quadratureRule);
  FEValues<dim> fe_valuesM (FE, quadratureM, update_values | update_gradients | update_JxW_values);
  FEValues<dim> fe_valuesH (FE, quadratureH, update_values | update_gradients | update_JxW_values);
  const unsigned int   dofs_per_cell = FE.dofs_per_cell;
  const unsigned int   num_quad_points = quadratureH.size();
  FullMatrix<double>   elementalMassMatrix (dofs_per_cell, dofs_per_cell);
  FullMatrix<double>   elementalHamiltonianMatrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       elementalMassVector (dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  std::vector<double> cellPhiTotal(num_quad_points);  
  Vector<double>  localsolution(solution);

  //parallel loop over all elements
  typename DoFHandler<dim>::active_cell_iterator cell = dofHandler->begin_active(), endc = dofHandler->end();
  unsigned int cellID=0;
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      elementalMassVector=0;
      elementalHamiltonianMatrix=0;
      elementalMassVector=0;

      //compute values for the current element
      fe_valuesM.reinit (cell);
      fe_valuesH.reinit (cell);
      fe_valuesH.get_function_values(localsolution, cellPhiTotal);

      //Get Exc
      std::vector<double> densityValue(num_quad_points);
      std::vector<double> exchangePotentialVal(num_quad_points);
      std::vector<double> corrPotentialVal(num_quad_points);
      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
	densityValue[q_point] = (*rhoValues)(cellID, q_point);
      xc_lda_vxc(&funcX,num_quad_points,&densityValue[0],&exchangePotentialVal[0]);
      xc_lda_vxc(&funcC,num_quad_points,&densityValue[0],&corrPotentialVal[0]);

      //local operator
      for (unsigned int i=0; i<dofs_per_cell; ++i){
	for (unsigned int j=0; j<dofs_per_cell; ++j){
	  for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	    elementalHamiltonianMatrix(i,j) += (0.5*fe_valuesH.shape_grad (i, q_point)*fe_valuesH.shape_grad (j, q_point)+
					 fe_valuesH.shape_value(i, q_point)*
					 fe_valuesH.shape_value(j, q_point)*
					 (cellPhiTotal[q_point]+exchangePotentialVal[q_point]+corrPotentialVal[q_point]))*fe_valuesH.JxW (q_point);
	    elementalMassMatrix(i,j)+= (fe_valuesM.shape_value(i, q_point)*
					fe_valuesM.shape_value(j, q_point))*fe_valuesM.JxW (q_point);
	  }
	}
      }
      //local mass vector
      for (unsigned int i=0; i<dofs_per_cell; ++i){
	for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	  elementalMassVector(i)+=(fe_valuesM.shape_value(i, q_point)*fe_valuesM.shape_value(i, q_point))*fe_valuesM.JxW (q_point);
	}
      }
      cell->get_dof_indices (local_dof_indices);
      //assemble to global matrices
      constraints.distribute_local_to_global(elementalHamiltonianMatrix,local_dof_indices,hamiltonianMatrix);
      constraints.distribute_local_to_global(elementalMassMatrix,local_dof_indices,massMatrix);
      constraints.distribute_local_to_global(elementalMassVector,local_dof_indices,massVector);

      cellID++;
    }
  }
  //MPI operation to sync data 
  hamiltonianMatrix.compress(VectorOperation::add);
  massMatrix.compress(VectorOperation::add);
  massVector.compress(VectorOperation::add);
  //
  VecSqrtAbs(massVector);
  VecReciprocal(massVector);
  MatDiagonalScale(hamiltonianMatrix,massVector,massVector);
  //
  computing_timer.exit_section("eigen assembly"); 
}

//solve eigen value problem 
template <int dim>
void eigen<dim>::solve(PETScWrappers::MPI::Vector& solution,
		       PETScWrappers::MPI::SparseMatrix& massMatrix,
		       PETScWrappers::MPI::SparseMatrix& hamiltonianMatrix,
		       PETScWrappers::MPI::Vector& massVector, 
		       ConstraintMatrix& constraints,
		       Table<2,double>* rhoValues,
		       std::vector<double>& eigenValues,
		       std::vector<PETScWrappers::MPI::Vector>& eigenVectors
		       ){
  //assemble
  assemble(solution, massMatrix, hamiltonianMatrix, massVector, constraints, rhoValues);
  
  //solve
  computing_timer.enter_section("eigen solve"); 
  SolverControl solver_control (dofHandler->n_dofs(), 1.0e-5); 
  SLEPcWrappers::SolverJacobiDavidson eigensolver (solver_control,mpi_communicator);
  //SLEPcWrappers::SolverKrylovSchur eigensolver (solver_control,mpi_communicator);
  //SLEPcWrappers::SolverArnoldi  eigensolver (solver_control,mpi_communicator);
  eigensolver.set_which_eigenpairs (EPS_SMALLEST_REAL);
  eigensolver.solve(hamiltonianMatrix, eigenValues, eigenVectors, numEigenValues);
  //print rigen values to screen
  for (unsigned int i=0; i<numEigenValues; i++)
    if (this_mpi_process == 0) std::printf("Eigen value %u : %30.20e \n", i, eigenValues[i]);

  computing_timer.exit_section("eigen solve"); 
}

