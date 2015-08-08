//source file for electron density related computations

//calculate electron density
void dftClass::compute_rhoOut(){
  QGauss<3>  quadrature(quadratureRule);
  FEValues<3> fe_values (FE, quadrature, update_values | update_JxW_values | update_quadrature_points);
  const unsigned int   dofs_per_cell = FE.dofs_per_cell;
  const unsigned int   num_quad_points = quadrature.size();
   
  std::vector<Vector<double> > localEigenVectors(numEigenValues);
  //project eigen vectors to regular FEM space by multiplying with M^(-0.5)
  for (unsigned int i=0; i<numEigenValues; ++i){
    VecPointwiseMult(eigenVectorsProjected[i],eigenVectors[i], massVector);    
    localEigenVectors[i]=eigenVectorsProjected[i]; //Check if this correct way to obtain local vector when running in parallel
  }
  
  //create new rhoValue tables
  rhoOutValues=new Table<2,double>(triangulation.n_locally_owned_active_cells(),std::pow(quadratureRule,3));
  rhoOutVals.push_back(rhoOutValues);
  
  //loop over elements
  std::vector<double> rhoOut(num_quad_points);
 //parallel loop over all elements
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  unsigned int cellID=0;
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      fe_values.reinit (cell); 
      std::vector<double> tempPsi(num_quad_points);
      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	rhoOut[q_point]=0.0;
      }
      for (unsigned int i=0; i<numEigenValues; ++i){
	fe_values.get_function_values(localEigenVectors[i], tempPsi);
	for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	  double temp=(eigenValues[i]-fermiEnergy)/(kb*TVal);
	  double partialOccupancy=1.0/(1.0+exp(temp)); 
	  rhoOut[q_point]+=2.0*partialOccupancy*std::pow(tempPsi[q_point],2.0); 
	}
      }
      for (unsigned int q_point=0; q_point<num_quad_points; ++q_point){
	(*rhoOutValues)(cellID,q_point)=rhoOut[q_point];
      }
    cellID++;
    }
  }
}
