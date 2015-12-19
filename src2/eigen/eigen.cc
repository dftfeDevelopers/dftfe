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
  for (unsigned int i=0; i<numEigenValues; ++i){
    vectorType* temp=new vectorType;
    HXvalue.push_back(temp);
  } 
}

//initialize eigenClass object
void eigenClass::init(){
  computing_timer.enter_section("eigenClass setup"); 
  //constraints
  constraintsNone.clear ();
  DoFTools::make_hanging_node_constraints (dftPtr->dofHandler, constraintsNone);
  constraintsNone.close();
  //init vectors
  dftPtr->matrix_free_data.initialize_dof_vector (massVector);
  rhsVeff.reinit(massVector);
  vEffective.reinit(massVector);
  //compute mass vector
  computeMassVector();
  //XHX size
  XHXValue.resize(dftPtr->eigenVectors.size()*dftPtr->eigenVectors.size(),0.0);
  //HX
  for (unsigned int i=0; i<numEigenValues; ++i){
    HXvalue[i]->reinit(massVector);
  }
  computing_timer.exit_section("eigenClass setup"); 
} 

void eigenClass::computeMassVector(){
  computing_timer.enter_section("eigenClass Mass assembly"); 
  
  VectorizedArray<double>  one = make_vectorized_array (1.0);
  FEEvaluation<3,FEOrder>  fe_eval(dftPtr->matrix_free_data);
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

void eigenClass::computeVeff(const vectorType& phi){
  const unsigned int n_cells = dftPtr->matrix_free_dataGauss.n_macro_cells();
  const unsigned int n_array_elements = VectorizedArray<double>::n_array_elements;
  FEEvaluation<3,FEOrder> fe_eval (dftPtr->matrix_free_dataGauss);
  vEffective.reinit (n_cells, fe_eval.n_q_points);
  for (unsigned int cell=0; cell<n_cells; ++cell){
    fe_eval.reinit (cell);
    fe_eval.read_dof_values(phi);
    fe_eval.evaluate (true, false, false);
    for (unsigned int q=0; q<dftPtr->matrix_free_dataGauss.n_q_points; ++q){
      //one quad point per cell in the cell block
      std::vector<double> densityValue(num_quad_points);
      std::vector<double> exchangePotentialVal(num_quad_points);
      std::vector<double> corrPotentialVal(num_quad_points);
      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	densityValue[q_point] = ((*rhoValues)[cell->id()][q_point]);
      }
      
    xc_lda_vxc(&funcX,num_quad_points,&densityValue[0],&exchangePotentialVal[0]);
    xc_lda_vxc(&funcC,num_quad_points,&densityValue[0],&corrPotentialVal[0]);

      
      vEffective(cell,q)=fe_eval.get_value(q)+;


    for (unsigned int v=0; v < dftPtr->matrix_free_dataGauss.n_components_filled(cell); ++v){
      cellPtr=dftPtr->matrix_free_dataGauss.get_cell_iterator(cell, v);
      
  
   
   
      
   

    //Get Exc
    std::vector<double> densityValue(num_quad_points);
    std::vector<double> exchangePotentialVal(num_quad_points);
    std::vector<double> corrPotentialVal(num_quad_points);
    for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
      densityValue[q_point] = ((*rhoValues)[cell->id()][q_point]);
    }
    xc_lda_vxc(&funcX,num_quad_points,&densityValue[0],&exchangePotentialVal[0]);
    xc_lda_vxc(&funcC,num_quad_points,&densityValue[0],&corrPotentialVal[0]);
    
  }
}

void eigenClass::computeVEffectiveRHS(std::map<dealii::CellId,std::vector<double> >* rhoValues, const vectorType& phi){
  rhsVeff=0.0; 
  //local data structures
  QGauss<3>  quadrature(FEOrder+1);
  FEValues<3> fe_values (FE, quadrature, update_values | update_JxW_values);
  const unsigned int dofs_per_cell = FE.dofs_per_cell;
  const unsigned int num_quad_points = quadrature.size();
  std::vector<dealii::types::global_dof_index> local_dof_indices (dofs_per_cell);
  std::vector<double> cellPhiTotal(num_quad_points);  
  Vector<double>      elementalResidual (dofs_per_cell);

  //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      elementalResidual=0.0;
      //compute values for the current element
      fe_values.reinit (cell);
      fe_values.get_function_values(phi, cellPhiTotal);
      
      //Get Exc
      std::vector<double> densityValue(num_quad_points);
      std::vector<double> exchangePotentialVal(num_quad_points);
      std::vector<double> corrPotentialVal(num_quad_points);
      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	densityValue[q_point] = ((*rhoValues)[cell->id()][q_point]);
      }
      xc_lda_vxc(&funcX,num_quad_points,&densityValue[0],&exchangePotentialVal[0]);
      xc_lda_vxc(&funcC,num_quad_points,&densityValue[0],&corrPotentialVal[0]);

      //local rhs
      for (unsigned int i=0; i<dofs_per_cell; ++i){
	for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){ 
	  elementalResidual(i) += fe_values.shape_value(i, q_point)*(cellPhiTotal[q_point]+exchangePotentialVal[q_point]+corrPotentialVal[q_point])*fe_values.JxW(q_point);
	}
      }

      //assemble to global data structures
      cell->get_dof_indices (local_dof_indices);
      constraintsNone.distribute_local_to_global(elementalResidual, local_dof_indices, rhsVeff);
    }
  }
  //MPI operation to sync data 
  rhsVeff.compress(VectorOperation::add);
}

void eigenClass::MX (const dealii::MatrixFree<3,double>           &data,
		     vectorType                      &dst,
		     const vectorType                &src,
		     const std::pair<unsigned int,unsigned int> &cell_range) const{
  FEEvaluation<3,FEOrder>  fe_eval(data);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell){
    fe_eval.reinit (cell);
    fe_eval.read_dof_values(src);
    fe_eval.evaluate (true,false,false);
    for (unsigned int q=0; q<fe_eval.n_q_points; ++q){
      fe_eval.submit_value (fe_eval.get_value(q), q);
    }
    fe_eval.integrate (true, false);
    fe_eval.distribute_local_to_global (dst);
  }
}

void eigenClass::vmult (vectorType       &dst,
			const vectorType &src) const{
  dst=0.0;
  dftPtr->matrix_free_dataGauss.cell_loop (&eigenClass::MX, this, dst, src);
  dst.compress(VectorOperation::add);
}

void eigenClass::computeVEffective(std::map<dealii::CellId,std::vector<double> >* rhoValues, const vectorType& phi){
  //RHS
  computeVEffectiveRHS(rhoValues, phi);
  //solve
  computing_timer.enter_section("eigenClass VEff solve"); 
  SolverControl solver_control(maxLinearSolverIterations,relLinearSolverTolerance*rhsVeff.l2_norm());
  SolverCG<vectorType> solver(solver_control);
  try{
    vEffective=0.0;
    solver.solve(*this, vEffective, rhsVeff, IdentityMatrix(rhsVeff.size()));
    vEffective.update_ghost_values();
  }
  catch (...) {
    pcout << "\nWarning: vEffective solver did not converge as per set tolerances. consider increasing maxLinearSolverIterations or decreasing relLinearSolverTolerance.\n";
  }
  char buffer[200];
  sprintf(buffer, "veff solve: initial residual:%12.6e, current residual:%12.6e, nsteps:%u, tolerance criterion:%12.6e\n", \
	  solver_control.initial_value(),				\
	  solver_control.last_value(),					\
	  solver_control.last_step(), solver_control.tolerance()); 
  pcout<<buffer; 
  computing_timer.exit_section("eigenClass VEff solve"); 
}

//HX
void eigenClass::implementHX (const dealii::MatrixFree<3,double>  &data,
			      std::vector<vectorType*>  &dst, 
			      const std::vector<vectorType*>  &src,
			      const std::pair<unsigned int,unsigned int> &cell_range) const{

  FEEvaluation<3,FEOrder>  fe_evalVeff(data);
  FEEvaluation<3,FEOrder>  fe_eval(data);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell){
    fe_evalVeff.reinit (cell); fe_evalVeff.read_dof_values(vEffective); 
    fe_eval.reinit (cell); 
    for (unsigned int i=0; i<dst.size(); i++){
      fe_eval.read_dof_values(*src[i]);
      fe_eval.evaluate (true,true,false);
      for (unsigned int q=0; q<fe_eval.n_q_points; ++q){
	fe_eval.submit_gradient (fe_eval.get_gradient(q), q);
	fe_eval.submit_value    (fe_eval.get_value(q)*fe_evalVeff.get_value(q), q);
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
    constraintsNone.distribute(*(dftPtr->tempPSI2[i]));
    *dst[i]=0.0;
  }
  dftPtr->matrix_free_dataGauss.cell_loop (&eigenClass::implementHX, this, dst, dftPtr->tempPSI2); //HMX
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





