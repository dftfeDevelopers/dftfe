//source file for all dft level initializations

//initialize rho
void dft::initRho(){
  //Initialize electron density table storage
  rhoInValues=new Table<2,double>(triangulation.n_locally_owned_active_cells(),std::pow(quadratureRule,3));
  rhoInVals.push_back(rhoInValues);
  
  //Readin single atom rho initial guess
  pcout << "reading initial guess for rho\n";
  std::vector<std::vector<std::vector<double> > > singleAtomElectronDensity(numAtomTypes);
  readFile(singleAtomElectronDensity[0]);
  unsigned int numRows = singleAtomElectronDensity[0].size()-1;
  double xData[numRows], yData[numRows];
  for(unsigned int irow = 0; irow < numRows; ++irow){
    xData[irow] = singleAtomElectronDensity[0][irow][0];
    yData[irow] = singleAtomElectronDensity[0][irow][1];
  }
  
  //interpolate rho
  alglib::real_1d_array x;
  x.setcontent(numRows,xData);
  alglib::real_1d_array y;
  y.setcontent(numRows,yData);
  alglib::ae_int_t natural_bound_type = 1;
  spline1dbuildcubic(x, y, numRows, natural_bound_type, 0.0, natural_bound_type, 0.0, denSpline[0]);

  std::vector<double> outerMostPointDen(numAtomTypes);
  outerMostPointDen[0]= xData[numRows-1];
  //Initialize rho
  QGauss<3>  quadrature_formula(quadratureRule);
  FEValues<3> fe_values (FE, quadrature_formula, update_values);
  const unsigned int n_q_points    = quadrature_formula.size();
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  unsigned int cellID=0;
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      for (unsigned int q=0; q<n_q_points; ++q){
	MappingQ<3> test(1); 
	Point<3> quadPoint(test.transform_unit_to_real_cell(cell, fe_values.get_quadrature().point(q)));
	double distance=std::sqrt(quadPoint.square());
	if(distance <= outerMostPointDen[0]){
	  (*rhoInValues)(cellID,q)=std::abs(alglib::spline1dcalc(denSpline[0], distance));
	}
	else{
	  (*rhoInValues)(cellID,q)=0.0;
	} 
      }
      cellID++;
    }
  }
  //Normalize rho
  double charge=totalCharge();
  pcout << "initial charge: " << charge << std::endl;
  cellID=0;
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      for (unsigned int q=0; q<n_q_points; ++q){
	(*rhoInValues)(cellID,q)*=1.0/charge;
      }
      cellID++;
    }
  }
  
  //Initialize libxc
  int exceptParamX, exceptParamC;
  exceptParamX = xc_func_init(&funcX,XC_LDA_X,XC_UNPOLARIZED);
  exceptParamC = xc_func_init(&funcC,XC_LDA_C_VWN,XC_UNPOLARIZED);
  if(exceptParamX != 0 || exceptParamC != 0){
    pcout<<"-------------------------------------"<<std::endl;
    pcout<<"Exchange or Correlation Functional not found"<<std::endl;
    pcout<<"-------------------------------------"<<std::endl;
  }
  else{
    pcout<<"-------------------------------------"<<std::endl;
    pcout<<"The exchange functional "<<funcX.info->name<<" is defined in the references(s)"
	 << std::endl<< funcX.info->refs << std::endl;
    pcout<<"The exchange functional "<<funcX.info->name<<" is defined in the references(s)"
	 << std::endl<< funcX.info->refs << std::endl;
    pcout<<"-------------------------------------"<<std::endl;	  
  }
  
}

void dft::init(){
  //initialize FE objects
  dofHandler.distribute_dofs (FE);
  locally_owned_dofs = dofHandler.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dofHandler, locally_relevant_dofs);
  pcout << "number of elements: "
	<< triangulation.n_global_active_cells()
	<< std::endl
	<< "number of degrees of freedom: " 
	<< dofHandler.n_dofs() 
	<< std::endl;
  
  //initialize poisson problem related objects
  poissonObject.init();

  //constraints
  //zero constraints
  constraintsZero.clear ();
  constraintsZero.reinit (locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints (dofHandler, constraintsZero);
  VectorTools::interpolate_boundary_values (dofHandler, 0, ZeroFunction<3>(),constraintsZero);
  constraintsZero.close ();
  //OnebyR constraints
  constraints1byR.clear ();
  constraints1byR.reinit (locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints (dofHandler, constraints1byR);
  VectorTools::interpolate_boundary_values (dofHandler, 0, OnebyRBoundaryFunction<3>(atomLocations),constraints1byR);
  constraints1byR.close ();
  
  //initialize vectors and jacobian matrix using the sparsity pattern.
  phiTotRhoIn.reinit (locally_owned_dofs, mpi_communicator);
  phiTotRhoOut.reinit (locally_owned_dofs, mpi_communicator); 
  phiExt.reinit (locally_owned_dofs, mpi_communicator);
  residual.reinit (locally_owned_dofs, mpi_communicator);
  //
  CompressedSimpleSparsityPattern csp (locally_relevant_dofs);
  DoFTools::make_sparsity_pattern (dofHandler, csp, constraintsZero, false);
  SparsityTools::distribute_sparsity_pattern (csp,
					      dofHandler.n_locally_owned_dofs_per_processor(),
					      mpi_communicator,
					      locally_relevant_dofs);
  jacobian.reinit (locally_owned_dofs, locally_owned_dofs, csp, mpi_communicator);
  
  //initialize eigen problem related objects
  eigenObject.init();
  locally_owned_dofs = dofHandler.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dofHandler, locally_relevant_dofs);
  
  //constraints
  constraintsNone.clear ();
  constraintsNone.reinit (locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints (dofHandler, constraintsNone);
  DoFTools::make_zero_boundary_constraints(dofHandler, constraintsNone);
  constraintsNone.close ();
  
  //intialize vectors
  massVector.reinit(locally_owned_dofs, mpi_communicator);
  eigenValues.resize(numEigenValues);
  eigenVectors.resize(numEigenValues);
  eigenVectorsProjected.resize(numEigenValues);
  for (unsigned int i=0; i<numEigenValues; ++i){
    eigenVectors[i].reinit(locally_owned_dofs, mpi_communicator);
    eigenVectorsProjected[i].reinit(locally_owned_dofs, mpi_communicator);
  }
  
  //initialize M, K matrices using the sparsity pattern.
  CompressedSimpleSparsityPattern csp2 (locally_relevant_dofs);
  DoFTools::make_sparsity_pattern (dofHandler, csp2, constraintsNone, false);
  SparsityTools::distribute_sparsity_pattern (csp2,
					      dofHandler.n_locally_owned_dofs_per_processor(),
					      mpi_communicator,
					      locally_relevant_dofs);
  massMatrix.reinit (locally_owned_dofs, locally_owned_dofs, csp2, mpi_communicator);
  hamiltonianMatrix.reinit (locally_owned_dofs, locally_owned_dofs, csp2, mpi_communicator);
}
