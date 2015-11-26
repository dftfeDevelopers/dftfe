#include <fstream>
#include <boost/math/special_functions/spherical_harmonic.hpp>

//
void dftClass::readPSIRadialValues(std::vector<std::vector<std::vector<double> > >& singleAtomPSI){
  //Build Splines for Carbon
  alglib::spline1dinterpolant nEqualOnelEqualsZeroSplineCarbon,
    nEqualTwolEqualsZeroSplineCarbon,
    nEqualTwolEqualsOneSplineCarbon;
  double outerMostPointCarbon;
  int numRows = singleAtomPSI[0].size()-1;
  double xData[numRows];
  double yData[numRows];
  //1s
  for(int irow = 0; irow < numRows; ++irow){
    xData[irow] = singleAtomPSI[0][irow][0];
    yData[irow] = singleAtomPSI[0][irow][1];
  }
  outerMostPointCarbon = xData[numRows-1];
 
  alglib::real_1d_array x,yn1l0;
  x.setcontent(numRows,xData);
  yn1l0.setcontent(numRows,yData);
  alglib::ae_int_t natural_bound_typeCarbon = 0;
  alglib::spline1dbuildcubic(x, yn1l0, numRows,
			     natural_bound_typeCarbon,
			     0.0,
			     natural_bound_typeCarbon,
			     0.0,
			     nEqualOnelEqualsZeroSplineCarbon);
  //2s
  for(int irow = 0; irow < numRows; ++irow){
    yData[irow] = singleAtomPSI[0][irow][2];
  }
  alglib::real_1d_array yn2l0;
  yn2l0.setcontent(numRows,yData);
  alglib::spline1dbuildcubic(x, yn2l0, numRows,
			     natural_bound_typeCarbon,
			     0.0,
			     natural_bound_typeCarbon,
			     0.0,
			     nEqualTwolEqualsZeroSplineCarbon);
  //2p
  for(int irow = 0; irow < numRows; ++irow){
    yData[irow] = singleAtomPSI[0][irow][3];
  }  
  alglib::real_1d_array yn2l1;
  yn2l1.setcontent(numRows,yData);
  alglib::spline1dbuildcubic(x, yn2l1, numRows,
			     natural_bound_typeCarbon,
			     0.0,
			     natural_bound_typeCarbon,
			     0.0,
			     nEqualTwolEqualsOneSplineCarbon);
  
  //loop over nodes to set PSI initial guess
  unsigned int dofs_per_cell = matrix_free_data.get_dofs_per_cell();
  //get support points
  std::map<types::global_dof_index, Point<3> > support_points;
  MappingQ<3> mapQ(1);
  DoFTools::map_dofs_to_support_points(mapQ, dofHandler, support_points); 
  IndexSet locallyOwnedSet;
  DoFTools::extract_locally_owned_dofs(dofHandler, locallyOwnedSet);
  std::vector<unsigned int> locallyOwnedDOFs;
  locallyOwnedSet.fill_index_vector(locallyOwnedDOFs);
  std::vector<std::vector<double> > local_dof_values(5, std::vector<double>(locallyOwnedDOFs.size()));
  //loop over nodes
  for(unsigned int dof=0; dof<locallyOwnedDOFs.size(); dof++){
    unsigned int dofID= locallyOwnedDOFs[dof];
     Point<3> node = support_points[dofID];
    for(unsigned int atom=0; atom<atomLocations.size()[0]; atom++){
      Point<3> atomCoord(atomLocations(atom,1),atomLocations(atom,2),atomLocations(atom,3));
      //
      double x =node[0]-atomCoord[0];
      double y =node[1]-atomCoord[1];
      double z =node[2]-atomCoord[2];
      //
      double r = sqrt(x*x + y*y + z*z);
      double theta = acos(z/r);
      double phi = atan2(y,x);
      //pcout << "r:" << r << " t:" << theta << " p:" << phi << std::endl;
      if (r==0){theta=0; phi=0;}
      double R;
      if (atom==0){
	R=0.0;
	//Carbon
	//1s
	if (r<=outerMostPointCarbon) R = alglib::spline1dcalc(nEqualOnelEqualsZeroSplineCarbon,r);
	local_dof_values[0][dof] =  R*boost::math::spherical_harmonic_r(0,0,theta,phi);
	//2s
	if (r<=outerMostPointCarbon) R = alglib::spline1dcalc(nEqualTwolEqualsZeroSplineCarbon,r);
	local_dof_values[1][dof] =  R*boost::math::spherical_harmonic_r(0,0,theta,phi);
	//2px
	if (r<=outerMostPointCarbon) R = alglib::spline1dcalc(nEqualTwolEqualsOneSplineCarbon,r);
	local_dof_values[2][dof] = R*sqrt(2.0)*boost::math::spherical_harmonic_i(1,1,theta,phi);
	//2py
	local_dof_values[3][dof] = R*boost::math::spherical_harmonic_r(1,0,theta,phi);
	//2pz
	local_dof_values[4][dof] = R*sqrt(2.0)*boost::math::spherical_harmonic_r(1,1,theta,phi);
      }
    }
  }
  //
  for (unsigned int i=0; i<eigenVectors.size(); ++i){
    constraintsNone.distribute_local_to_global(local_dof_values[i], locallyOwnedDOFs, *eigenVectors[i]);
  }
  //multiply by M^0.5
  for (unsigned int i=0; i<eigenVectors.size(); ++i){
    for (unsigned int j=0; j<eigenVectors[i]->local_size(); j++){
      if (std::abs(eigen.massVector.local_element(j))>1.0e-15){
	eigenVectors[i]->local_element(j)/=eigen.massVector.local_element(j);
      }
    }  
    eigenVectors[i]->update_ghost_values();
  }
}

//
void dftClass::readPSI(){
  computing_timer.enter_section("dftClass init PSI"); 
  const unsigned int numAtomTypes=1; //1 for C atom
  std::vector<std::vector<std::vector<double> > > singleAtomPSI (numAtomTypes); 
  for (unsigned int atom=0; atom<numAtomTypes; atom++){
    char buffer[100];
    sprintf(buffer, "RadialWaveFunction_AT%u", atom);
    unsigned int numColumns;
    if (atom==0) numColumns=4;
    else numColumns=2;
    //read from file
    std::vector<double> rowData(numColumns,0.0);
    std::ifstream tempFile;
    tempFile.open(buffer);
    unsigned int numRows=0;
    if (tempFile.is_open()) {
      while (!tempFile.eof()) {
	for(int i = 0; i < numColumns; ++i) tempFile>>rowData[i];
	singleAtomPSI[atom].push_back(rowData);
	numRows++;
      }
      tempFile.close();
    }
    else{
      pcout << "file not open. psiInitialGuess.cc \n"; 
      exit(-1);
    }
  }
  //
  readPSIRadialValues(singleAtomPSI);
  //
  computing_timer.exit_section("dftClass init PSI"); 
}
