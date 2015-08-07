//source file for dft class initializations

//initialize rho
void dftClass::initRho(){
  //Initialize electron density table storage
  rhoInValues=new std::map<dealii::CellId,std::vector<double> >;
  rhoInVals.push_back(rhoInValues);

  //Readin single atom rho initial guess
  pcout << "reading initial guess for rho\n";
  unsigned int numAtomTypes=initialGuessFiles.size();
  std::vector<alglib::spline1dinterpolant> denSpline(numAtomTypes);
  std::vector<std::vector<std::vector<double> > > singleAtomElectronDensity(numAtomTypes);
  std::vector<double> outerMostPointDen(numAtomTypes);
  unsigned int atomType=0;
  std::vector<unsigned int> atomTypeKeys; 
  //loop over atom types
  for (std::map<unsigned int, std::string >::iterator it=initialGuessFiles.begin(); it!=initialGuessFiles.end(); it++){
    atomTypeKeys.push_back(it->first);
    readFile(singleAtomElectronDensity[atomType],it->second);
    unsigned int numRows = singleAtomElectronDensity[atomType].size()-1;
    double xData[numRows], yData[numRows];
    for(unsigned int irow = 0; irow < numRows; ++irow){
      xData[irow] = singleAtomElectronDensity[atomType][irow][0];
      yData[irow] = singleAtomElectronDensity[atomType][irow][1];
    }
  
    //interpolate rho
    alglib::real_1d_array x;
    x.setcontent(numRows,xData);
    alglib::real_1d_array y;
    y.setcontent(numRows,yData);
    alglib::ae_int_t natural_bound_type = 1;
    spline1dbuildcubic(x, y, numRows, natural_bound_type, 0.0, natural_bound_type, 0.0, denSpline[atomType]);
    outerMostPointDen[atomType]= xData[numRows-1];
    atomType++;
  }
  
  //create atom type map (atom number to atom type id)
  unsigned int numAtoms=atomLocations.size()[0];
  std::map<unsigned int, unsigned int> atomTypeMap;
  for (unsigned int n=0; n<numAtoms; n++){
    for (unsigned int z=0; z<atomTypeKeys.size(); z++){
      if (atomTypeKeys[z] == (unsigned int) atomLocations[n][0]){
	atomTypeMap[n]=z; 
	break;
      }
    }
  }

  //Initialize rho
  QGauss<3>  quadrature_formula(quadratureRule);
  FEValues<3> fe_values (FE, quadrature_formula, update_values);
  const unsigned int n_q_points    = quadrature_formula.size();
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      (*rhoInValues)[cell->id()]=std::vector<double>(n_q_points*n_q_points);
      for (unsigned int q=0; q<n_q_points; ++q){
	MappingQ<3> test(1); 
	Point<3> quadPoint(test.transform_unit_to_real_cell(cell, fe_values.get_quadrature().point(q)));
	double rhoValueAtQuadPt=0.0;
	//loop over atoms
	for (unsigned int n=0; n<numAtoms; n++){
	  Point<3> atom(atomLocations[n][1],atomLocations[n][2],atomLocations[n][3]);
	  double distanceToAtom=quadPoint.distance(atom);
	  if(distanceToAtom <= outerMostPointDen[atomTypeMap[n]]){
	      rhoValueAtQuadPt+=std::abs(alglib::spline1dcalc(denSpline[atomTypeMap[n]], distanceToAtom));
	  }
	}
	(*rhoInValues)[cell->id()][q]=rhoValueAtQuadPt;
      }
    }
  }
  //Normalize rho
  double charge=totalCharge();
  pcout << "initial charge: " << charge << std::endl;
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      for (unsigned int q=0; q<n_q_points; ++q){
	(*rhoInValues)[cell->id()][q]*=1.0/charge;
      }
    }
  }
  
  //Initialize libxc
  int exceptParamX, exceptParamC;
  exceptParamX = xc_func_init(&funcX,XC_LDA_X,XC_UNPOLARIZED);
  exceptParamC = xc_func_init(&funcC,XC_LDA_C_PZ,XC_UNPOLARIZED);
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

void dftClass::init(){
  computing_timer.enter_section("dftClass setup"); 
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
  //matrix fee data structure
  QGaussLobatto<1> quadrature (FEOrder+1);
  typename MatrixFree<3>::AdditionalData additional_data;
  additional_data.mpi_communicator = MPI_COMM_WORLD;
  additional_data.tasks_parallel_scheme = MatrixFree<3>::AdditionalData::partition_partition;
  constraintsNone.clear ();
  constraintsNone.reinit (locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints (dofHandler, constraintsNone);
  constraintsNone.close();
  matrix_free_data.reinit (dofHandler, constraintsNone, quadrature, additional_data);
  //initialize eigen vectors
  for (std::vector<parallel::distributed::Vector<double>*>::iterator it=eigenVectors.begin(); it!=eigenVectors.end(); ++it){
    matrix_free_data.initialize_dof_vector(**it);
  } 
  //initialize density and locate atome core nodes
  initRho();
  locateAtomCoreNodes();
  computing_timer.exit_section("dftClass setup"); 

  //initialize poisson and eigen problem related objects
  poisson.init();
  eigen.init();
}
