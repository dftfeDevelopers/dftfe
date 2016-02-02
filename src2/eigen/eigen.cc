#include "../../include2/eigen.h"

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
}

//initialize eigenClass object
void eigenClass::init(){
  computing_timer.enter_section("eigenClass setup");
  //init vectors
  for (unsigned int i=0; i<dftPtr->numEigenValues; ++i){
    vectorType* temp=new vectorType;
    HXvalue.push_back(temp);
  } 
  dftPtr->matrix_free_data.initialize_dof_vector (massVector);
  //compute mass vector
  computeMassVector();
  //XHX size
  XHXValue.resize(dftPtr->eigenVectors.size()*dftPtr->eigenVectors.size(),0.0);
  //HX
  for (unsigned int i=0; i<dftPtr->numEigenValues; ++i){
    HXvalue[i]->reinit(massVector);
  }
  computing_timer.exit_section("eigenClass setup"); 
} 

void eigenClass::computeMassVector(){
  computing_timer.enter_section("eigenClass Mass assembly"); 
  
  VectorizedArray<double>  one = make_vectorized_array (1.0);
  FEEvaluation<3,FEOrder>  fe_eval(dftPtr->matrix_free_data, 0, 1); //Selecting GL quadrature points
  const unsigned int       n_q_points = fe_eval.n_q_points;
  for (unsigned int cell=0; cell<dftPtr->matrix_free_data.n_macro_cells(); ++cell){
    fe_eval.reinit(cell);
    for (unsigned int q=0; q<n_q_points; ++q) fe_eval.submit_value(one,q);
    fe_eval.integrate (true,false);
    fe_eval.distribute_local_to_global (massVector);
  }
  massVector.compress(VectorOperation::add);
  //compute inverse
  for (unsigned int i=0; i<massVector.local_size(); i++){
    if (std::abs(massVector.local_element(i))>1.0e-15){
      massVector.local_element(i)=1.0/std::sqrt(massVector.local_element(i));
    }
    else{
      massVector.local_element(i)=0.0;
    }
  }
  //pcout << "massVector norm: " << massVector.l2_norm() << "\n";
  computing_timer.exit_section("eigenClass Mass assembly");
}

void eigenClass::computeVEff(std::map<dealii::CellId,std::vector<double> >* rhoValues, const vectorType& phi){
  const unsigned int n_cells = dftPtr->matrix_free_data.n_macro_cells();
  const unsigned int n_array_elements = VectorizedArray<double>::n_array_elements;
  FEEvaluation<3,FEOrder> fe_eval (dftPtr->matrix_free_data, 0 ,0);
  vEff.reinit (n_cells, fe_eval.n_q_points);
  typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;
  //loop over cell block
  for (unsigned int cell=0; cell<n_cells; ++cell){
    fe_eval.reinit (cell);
    fe_eval.read_dof_values(phi);
    fe_eval.evaluate (true, false, false);
    for (unsigned int q=0; q<fe_eval.n_q_points; ++q){
      //loop over each cell
      unsigned int n_sub_cells=dftPtr->matrix_free_data.n_components_filled(cell);
      std::vector<double> densityValue(n_sub_cells), exchangePotentialVal(n_sub_cells), corrPotentialVal(n_sub_cells);
      for (unsigned int v=0; v < n_sub_cells; ++v){
	cellPtr=dftPtr->matrix_free_data.get_cell_iterator(cell, v);
	densityValue[v] = ((*rhoValues)[cellPtr->id()][q]);
      }
      xc_lda_vxc(&funcX,n_sub_cells,&densityValue[0],&exchangePotentialVal[0]);
      xc_lda_vxc(&funcC,n_sub_cells,&densityValue[0],&corrPotentialVal[0]);
      VectorizedArray<double>  exchangePotential, corrPotential;
      for (unsigned int v=0; v < n_sub_cells; ++v){
	exchangePotential[v]=exchangePotentialVal[v];
	corrPotential[v]=corrPotentialVal[v];
      }
      //sum all to vEffective
      vEff(cell,q)=fe_eval.get_value(q)+exchangePotential+corrPotential;
    }
  }
}

//HX
void eigenClass::implementHX (const dealii::MatrixFree<3,double>  &data,
			      std::vector<vectorType*>  &dst, 
			      const std::vector<vectorType*>  &src,
			      const std::pair<unsigned int,unsigned int> &cell_range) const{
  VectorizedArray<double>  half = make_vectorized_array (0.5);
  FEEvaluation<3,FEOrder>  fe_eval(data, 0, 0);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell){
    fe_eval.reinit (cell); 
    for (unsigned int i=0; i<dst.size(); i++){
      fe_eval.read_dof_values(*src[i]);
      fe_eval.evaluate (true,true,false);
      for (unsigned int q=0; q<fe_eval.n_q_points; ++q){
	fe_eval.submit_gradient (fe_eval.get_gradient(q)*half, q);
	fe_eval.submit_value    (fe_eval.get_value(q)*vEff(cell,q), q);
      }
      fe_eval.integrate (true, true);
      fe_eval.distribute_local_to_global (*dst[i]);
    }
  }
}

//HX
void eigenClass::HX(const std::vector<vectorType*> &src, std::vector<vectorType*> &dst) {
  computing_timer.enter_section("eigenClass HX");
  for (unsigned int i=0; i<src.size(); i++){
    *(dftPtr->tempPSI2[i])=*src[i];
    dftPtr->tempPSI2[i]->scale(massVector); //MX
    dftPtr->constraintsNone.distribute(*(dftPtr->tempPSI2[i]));
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
    *(dftPtr->tempPSI3[i])=0.0;
  }
  for (std::vector<double>::iterator it=XHXValue.begin(); it!=XHXValue.end(); it++){
    (*it)=0.0;  
  }
  //HX
  HX(src, dftPtr->tempPSI3);
  //XHX
  unsigned int dofs_per_proc=src[0]->local_size();
  std::vector<double> xhx(src.size()*src.size()), hx(dofs_per_proc*src.size()), x(dofs_per_proc*src.size());
  char transA  = 'T', transB  = 'N';
  double alpha = 1.0, beta  = 0.0;
  int k=dofs_per_proc, n= src.size();
  int lda= k, ldb=k, ldc=n;
  //extract vectors at the processor level
  std::vector<unsigned int> local_dof_indices(dofs_per_proc);
  src[0]->locally_owned_elements().fill_index_vector(local_dof_indices);
  unsigned int index=0;
  for (std::vector<vectorType*>::const_iterator it=src.begin(); it!=src.end(); it++){
    (*it)->extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), x.begin()+dofs_per_proc*index);
    dftPtr->tempPSI3[index]->extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), hx.begin()+dofs_per_proc*index);
    index++;
  }
  dgemm_(&transA, &transB, &n, &n, &k, &alpha, &x[0], &lda, &hx[0], &ldb, &beta, &xhx[0], &ldc);
  for (unsigned int i=0; i<xhx.size(); i++){
    (*XHXValuePtr)[i]+=xhx[i]; 
  }
  //all reduce XHXValue
  Utilities::MPI::sum(XHXValue, mpi_communicator, XHXValue); 
  computing_timer.exit_section("eigenClass XHX");
}





