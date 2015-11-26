#include "../../include/eigen.h"

//constructor
eigenClass::eigenClass(dftClass* _dftPtr):
  dftPtr(_dftPtr),
  FE (QGaussLobatto<1>(FEOrder+1)),
  XHXValuePtr(&XHXValue),
  mpi_communicator (MPI_COMM_WORLD),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::summary, TimerOutput::wall_times)
{
  for (unsigned int i=0; i<numEigenValues; ++i){
    vectorType* temp=new vectorType;
    HXvalue.push_back(temp);
  } 
}

//initialize eigenClass object
void eigenClass::init(){
  computing_timer.enter_section("eigenClass setup"); 
  unsigned int numCells=dftPtr->triangulation.n_locally_owned_active_cells();
  dealii::IndexSet numDofs=dftPtr->locally_relevant_dofs;
  //intialize the size of Table storing element level jacobians
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      localHamiltonians[cell->id()]=std::vector<double>(FE.dofs_per_cell*FE.dofs_per_cell);
    }
  }
  localHamiltoniansPtr=&localHamiltonians;

  //constraints
  constraintsNone.clear ();
  DoFTools::make_hanging_node_constraints (dftPtr->dofHandler, constraintsNone);
  constraintsNone.close();
  constraintsNone2.clear ();
  constraintsNone2.close();
  //compute mass vector
  computeMassVector();

  //XHX size
  XHXValue.resize(dftPtr->eigenVectors.size()*dftPtr->eigenVectors.size(),0.0);
  //HX
  for (unsigned int i=0; i<numEigenValues; ++i){
    dftPtr->matrix_free_data.initialize_dof_vector(*HXvalue[i]);
  } 
  computing_timer.exit_section("eigenClass setup"); 
} 

void eigenClass::computeMassVector(){
  computing_timer.enter_section("eigenClass Mass assembly"); 
  dftPtr->matrix_free_data.initialize_dof_vector (massVector);
  massVector=0.0;

  //local data structures
  QGaussLobatto<3> quadratureM(FEOrder+1);
  FEValues<3> fe_valuesM (FE, quadratureM, update_values | update_JxW_values);
  const unsigned int dofs_per_cell = FE.dofs_per_cell;
  const unsigned int num_quad_points = quadratureM.size();
  Vector<double>       elementalMassVector (dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      elementalMassVector=0;
      //compute values for the current element
      fe_valuesM.reinit (cell);
      //local mass vector
      for (unsigned int i=0; i<dofs_per_cell; ++i){
	for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	  elementalMassVector(i)+=(fe_valuesM.shape_value(i, q_point)*fe_valuesM.shape_value(i, q_point))*fe_valuesM.JxW (q_point);
	}
      }
      cell->get_dof_indices (local_dof_indices);
      constraintsNone.distribute_local_to_global(elementalMassVector, local_dof_indices, massVector);
    }
  }
  massVector.compress(VectorOperation::add);
  //compute inverse
  for (unsigned int i=0; i<massVector.local_size(); i++){
    if (std::abs(massVector.local_element(i))>1.0e-15){
      massVector.local_element(i)=1.0/std::sqrt(massVector.local_element(i));
    }
  }
  //constraintsNone.distribute(massVector);

  computing_timer.exit_section("eigenClass Mass assembly");
}

void eigenClass::computeLocalHamiltonians(std::map<dealii::CellId,std::vector<double> >* rhoValues, const vectorType& phi){
  computing_timer.enter_section("eigenClass Hamiltonian assembly"); 
  
  //local data structures
  QGauss<3>  quadratureH(FEOrder+1);
  FEValues<3> fe_valuesH (FE, quadratureH, update_values | update_gradients | update_JxW_values);
  const unsigned int dofs_per_cell = FE.dofs_per_cell;
  const unsigned int num_quad_points = quadratureH.size();
  std::vector<dealii::types::global_dof_index> local_dof_indices (dofs_per_cell);
  std::vector<double> cellPhiTotal(num_quad_points);  

  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      //compute values for the current element
      fe_valuesH.reinit (cell);
      fe_valuesH.get_function_values(phi, cellPhiTotal);
      
      //Get Exc
      std::vector<double> densityValue(num_quad_points);
      std::vector<double> exchangePotentialVal(num_quad_points);
      std::vector<double> corrPotentialVal(num_quad_points);
      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
	densityValue[q_point] = ((*rhoValues)[cell->id()][q_point]);
      xc_lda_vxc(&funcX,num_quad_points,&densityValue[0],&exchangePotentialVal[0]);
      xc_lda_vxc(&funcC,num_quad_points,&densityValue[0],&corrPotentialVal[0]);

      //local eigenClass operator
      double* tt=&localHamiltonians[cell->id()][0];
      cell->get_dof_indices (local_dof_indices);
      for (unsigned int i=0; i<dofs_per_cell; ++i){
	for (unsigned int j=0; j<dofs_per_cell; ++j){
	  tt[i*dofs_per_cell+j]=0.0;
	  for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	    //storing local Hamiltonian in Fortran column major format to help Lapack functions
	    tt[i*dofs_per_cell+j] += (0.5*fe_valuesH.shape_grad (j, q_point)*fe_valuesH.shape_grad (i, q_point)+
								 fe_valuesH.shape_value(j, q_point)*
								 fe_valuesH.shape_value(i, q_point)*
								 (cellPhiTotal[q_point]+exchangePotentialVal[q_point]+corrPotentialVal[q_point]))*fe_valuesH.JxW (q_point);
	  }
	  //H'=M^(-0.5)*H*M^(-0.5)
	  //tt[i*dofs_per_cell+j]*=massVector(local_dof_indices[j])*massVector(local_dof_indices[i]);
	}
      }
    }
  }
  computing_timer.exit_section("eigenClass Hamiltonian assembly");
}

//HX
void eigenClass::implementHX (const dealii::MatrixFree<3,double>  &data,
			      std::vector<vectorType*>  &dst, 
			      const std::vector<vectorType*>  &src,
			      const std::pair<unsigned int,unsigned int> &cell_range) const{
  const unsigned int   dofs_per_cell = FE.dofs_per_cell;
  std::vector<dealii::types::global_dof_index> local_dof_indices (dofs_per_cell);
  typename dealii::DoFHandler<3>::active_cell_iterator cell;

  std::vector<double> x(dofs_per_cell*dst.size()), y(dofs_per_cell*dst.size());
  //loop over all "cells"  (cell blocks)
  for (unsigned int cell_index=cell_range.first; cell_index<cell_range.second; ++cell_index){
    //loop over cells
    for (unsigned int v=0; v<data.n_components_filled(cell_index); ++v){
      cell=data.get_cell_iterator(cell_index, v);
      cell->get_dof_indices (local_dof_indices);
      unsigned int index=0;
      for (unsigned int i=0; i<dst.size(); i++){
	src[i]->extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), x.begin()+dofs_per_cell*index);
	index++;
      }
      //elemental HX
      char transA  = 'T', transB  = 'N';
      double alpha = 1.0, beta  = 0.0;
      //check lda, ldb, ldc values
      int m= dofs_per_cell, k=dofs_per_cell, n= dst.size(), lda= dofs_per_cell, ldb=dofs_per_cell, ldc=dofs_per_cell;
      dgemm_(&transA, &transB, &m, &n, &k, &alpha, &((*localHamiltoniansPtr)[cell->id()][0]), &lda, &x[0], &ldb, &beta, &y[0], &ldc);
      //assemble back
      std::vector<double>::iterator iter=y.begin();
      for (std::vector<vectorType*>::iterator it=dst.begin(); it!=dst.end(); it++){
	constraintsNone.distribute_local_to_global(iter, iter+dofs_per_cell,local_dof_indices.begin(), **it);
	iter+=dofs_per_cell;
      }
    }
  }
}

//XHX
void eigenClass::implementXHX (const dealii::MatrixFree<3,double>  &data,
			       std::vector<vectorType*>  &dst, 
			       const std::vector<vectorType*>  &src,
			       const std::pair<unsigned int,unsigned int> &cell_range) const{
  const unsigned int   dofs_per_cell = FE.dofs_per_cell;
  std::vector<dealii::types::global_dof_index> local_dof_indices (dofs_per_cell);
  typename dealii::DoFHandler<3>::active_cell_iterator cell;

  std::vector<double> xhx(src.size()*src.size()), hx(dofs_per_cell*src.size()), x(dofs_per_cell*src.size());
  //loop over all "cells"  (cell blocks)
  for (unsigned int cell_index=cell_range.first; cell_index<cell_range.second; ++cell_index){
    //loop over cells
    for (unsigned int v=0; v<data.n_components_filled(cell_index); ++v){
      cell=data.get_cell_iterator(cell_index, v);
      cell->get_dof_indices (local_dof_indices);
      unsigned int index=0;
      for (std::vector<vectorType*>::const_iterator it=src.begin(); it!=src.end(); it++){
	(*it)->extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), x.begin()+dofs_per_cell*index);
	index++;
      }
      //elemental HX
      char transA  = 'N', transB  = 'N';
      double alpha = 1.0, beta  = 0.0;
      //check lda, ldb, ldc values
      int k=dofs_per_cell, n= src.size();
      int lda= k, ldb=k, ldc=k;
      //HX
      dgemm_(&transA, &transB, &k, &n, &k, &alpha, &((*localHamiltoniansPtr)[cell->id()][0]), &lda, &x[0], &ldb, &beta, &hx[0], &ldc);
      transA  = 'T'; transB  = 'N';
      lda= k; ldb=k; ldc=n;
      //XHX
      dgemm_(&transA, &transB, &n, &n, &k, &alpha, &x[0], &lda, &hx[0], &ldb, &beta, &xhx[0], &ldc);
      //assemble back
      assembler_lock.acquire();
      for (unsigned int i=0; i<xhx.size(); i++){
	(*XHXValuePtr)[i]+=xhx[i]; 
      }
      assembler_lock.release();
    }
  }
}

//HX
void eigenClass::HX(const std::vector<vectorType*> &src, std::vector<vectorType*> &dst) {
  computing_timer.enter_section("eigenClass HX");
  for (unsigned int i=0; i<src.size(); i++){
    *(dftPtr->tempPSI2[i])=*src[i];
    dftPtr->tempPSI2[i]->scale(massVector); //MX
    constraintsNone.distribute(*(dftPtr->tempPSI2[i]));
    *dst[i]=0.0;
  }
  dftPtr->matrix_free_data.cell_loop (&eigenClass::implementHX, this, dst, dftPtr->tempPSI2); //HMX
  for (std::vector<vectorType*>::iterator it=dst.begin(); it!=dst.end(); it++){
    (*it)->scale(massVector); //MHMX  
    (*it)->compress(VectorOperation::add);  
  }
  computing_timer.exit_section("eigenClass HX");
}

//XHX
void eigenClass::XHX(const std::vector<vectorType*> &src){
  computing_timer.enter_section("eigenClass XHX");
  for (unsigned int i=0; i<src.size(); i++){
    *(dftPtr->tempPSI2[i])=*src[i];
    dftPtr->tempPSI2[i]->scale(massVector); //MX
    constraintsNone.distribute(*(dftPtr->tempPSI2[i]));
  }
  for (std::vector<double>::iterator it=XHXValue.begin(); it!=XHXValue.end(); it++){
    (*it)=0.0;  
  }
  dftPtr->matrix_free_data.cell_loop (&eigenClass::implementXHX, this, HXvalue, dftPtr->tempPSI2);
  //all reduce XHXValue
  Utilities::MPI::sum(XHXValue, mpi_communicator, XHXValue); 
  computing_timer.exit_section("eigenClass XHX");
}





