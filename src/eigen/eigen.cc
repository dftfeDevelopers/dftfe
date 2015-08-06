#include "../../include/eigen.h"

//constructor
eigenClass::eigenClass(dftClass* _dftPtr):
  dftPtr(_dftPtr),
  FE (QGaussLobatto<1>(FEOrder+1)),
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times)
{
}

//initialize eigenClass object
void eigenClass::init(){
  unsigned int numCells=dftPtr->triangulation.n_locally_owned_active_cells();
  dealii::IndexSet numDofs=dftPtr->locally_relevant_dofs;
  //intialize the size of Table storing element level jacobians
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      localHamiltonians[cell->id()]=std::vector<double>(FE.dofs_per_cell*FE.dofs_per_cell);
    }
  }
  //compute mass vector
  massVector=0.0;
  computeMassVector();
  massVector.compress(dealii::VectorOperation::add);
}

void eigenClass::computeMassVector(){
  computing_timer.enter_section("poissonClass mass assembly"); 

  //local data structures
  QGaussLobatto<3> quadratureM(quadratureRule);
  FEValues<3> fe_valuesM (FE, quadratureM, update_values | update_JxW_values);
  const unsigned int dofs_per_cell = FE.dofs_per_cell;
  const unsigned int num_quad_points = quadrature.size();
  Vector<double>       elementalMassVector (dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      elementalMassVector=0;
      //compute values for the current element
      fe_values.reinit (cell);
      //local mass vector
      for (unsigned int i=0; i<dofs_per_cell; ++i){
	for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	  elementalMassVector(i)+=(fe_valuesM.shape_value(i, q_point)*fe_valuesM.shape_value(i, q_point))*fe_valuesM.JxW (q_point);
	}
      }
      cell->get_dof_indices (local_dof_indices);
      massVector.add(local_dof_indices, elementalMassVector);
    }
  }
  computing_timer.exit_section("poissonClass mass assembly");
}

void eigenClass::computeLocalHamiltonians(){
  computing_timer.enter_section("poissonClass Hamiltonian assembly"); 

  //local data structures
  QGauss<3>  quadratureH(quadratureRule);
  FEValues<3> fe_valuesH (FE, quadratureH, update_values | update_gradients | update_JxW_values);
  const unsigned int dofs_per_cell = FE.dofs_per_cell;
  const unsigned int num_quad_points = quadrature.size();
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
  std::vector<double> cellPhiTotal(num_quad_points);  
  Vector<double>  localsolution(solution)

  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      //compute values for the current element
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

      //local poissonClass operator
      for (unsigned int i=0; i<dofs_per_cell; ++i){
	for (unsigned int j=0; j<dofs_per_cell; ++j){
	  localHamiltonians[cell->id()][i*dofs_per_cell+j]=0.0;
	  for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	    localHamiltonians[cell->id()][i*dofs_per_cell+j] += (0.5*fe_valuesH.shape_grad (i, q_point)*fe_valuesH.shape_grad (j, q_point)+
					 fe_valuesH.shape_value(i, q_point)*
					 fe_valuesH.shape_value(j, q_point)*
					 (cellPhiTotal[q_point]+exchangePotentialVal[q_point]+corrPotentialVal[q_point]))*fe_valuesH.JxW (q_point);
	  }
	}
      }
    }
  }
  computing_timer.exit_section("poissonClass Hamiltonian assembly");
}




