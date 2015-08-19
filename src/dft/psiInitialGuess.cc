#include <fstream>
#include <boost/math/special_functions/spherical_harmonic.hpp>

//
void dftClass::readPSIRadialValues(std::vector<std::vector<std::vector<double> > >& singleAtomPSI){
  //Build Splines for Carbon
  alglib::spline1dinterpolant nEqualOnelEqualsZeroSplineCarbon;
  alglib::spline1dinterpolant nEqualTwolEqualsZeroSplineCarbon;
  alglib::spline1dinterpolant nEqualTwolEqualsOneSplineCarbon;
  double outerMostPointCarbon;
  int numRows = singleAtomPSI[0].size() - 1;
  double xData[numRows];
  double yData[numRows];
  for(int irow = 0; irow < numRows; ++irow){
    xData[irow] = singleAtomPSI[0][irow][0];
    yData[irow] = singleAtomPSI[0][irow][1];
  }
  outerMostPointCarbon = xData[numRows - 1];
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
  
  //
  //Build Splines for Hydrogen
  //
  alglib::spline1dinterpolant lEqualsZeroSplineHydrogen;
  double outerMostPointHydrogen;
  alglib::ae_int_t natural_bound_typeHydrogen = 0;
  int numRowsHyd = singleAtomPSI[1].size() - 1;
  double xDataHyd[numRowsHyd];
  double yDataHyd[numRowsHyd];
  for(int irow = 0; irow < numRowsHyd; ++irow){
    xDataHyd[irow] = singleAtomPSI[1][irow][0];
    yDataHyd[irow] = singleAtomPSI[1][irow][1];
  }
  outerMostPointHydrogen = xDataHyd[numRows - 1];
  alglib::real_1d_array xH,y0H;
  xH.setcontent(numRowsHyd,xDataHyd);
  y0H.setcontent(numRowsHyd,yDataHyd);
  alglib::spline1dbuildcubic(xH, y0H, numRowsHyd,
		     natural_bound_typeHydrogen,
		     0.0,
		     natural_bound_typeHydrogen,
		     0.0,
		     lEqualsZeroSplineHydrogen);

  //loop over nodes to set PSI initial guess
  QGauss<3>  quadrature_formula(quadratureRule);
  FEValues<3> fe_values (FE, quadrature_formula, update_values);
  //
  unsigned int vertices_per_cell=GeometryInfo<3>::vertices_per_cell;
  DoFHandler<3>::active_cell_iterator
    cell = dofHandler.begin_active(),
    endc = dofHandler.end();
  //element loop
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      fe_values.reinit (cell);
      //node loop
      for (unsigned int i=0; i<vertices_per_cell; ++i){
	unsigned int nodeID=cell->vertex_dof_index(i,0);
	Point<3> node = cell->vertex(i);
	for(unsigned int atom=0; atom<atomLocations.size(0); atom++){
	  Point<3> atomCoord(atomLocations(atom,1),atomLocations(atom,2),atomLocations(atom,3));
	  //
	  double x =node[0]-atomCoord[0];
	  double y =node[1]-atomCoord[1];
	  double z =node[2]-atomCoord[2];
	  //
	  double r = sqrt(x*x + y*y + z*z);
	  double theta = acos(z/r);
	  double phi = atan2(y,x);
	  if (r==0){theta=0; phi=0;}
	  double R=0;
	  if (atom==0){
	    //Carbon
	    double *n1s,*n2s,*n2px,*n2py,*n2pz;
	    if (r<=outerMostPointCarbon) R = alglib::spline1dcalc(nEqualOnelEqualsZeroSplineCarbon,r);
	    //1s
	    eigenVectors[0][nodeID] =  R*boost::math::spherical_harmonic_r(0,0,theta,phi);
	    //2s
	    if (r<=outerMostPointCarbon) R = alglib::spline1dcalc(nEqualTwolEqualsZeroSplineCarbon,r);
	    eigenVectors[1][nodeID] =  R*boost::math::spherical_harmonic_r(0,0,theta,phi);
	    //2px
	    if (r<=outerMostPointCarbon) R = alglib::spline1dcalc(nEqualTwolEqualsOneSplineCarbon,r);
	    eigenVectors[2][nodeID] = R*sqrt(2.0)*boost::math::spherical_harmonic_i(1,1,theta,phi);
	    //2py
	    eigenVectors[3][nodeID] = R*boost::math::spherical_harmonic_r(1,0,theta,phi);
	    //2pz
	    eigenVectors[4][nodeID] = R*sqrt(2.0)*boost::math::spherical_harmonic_r(1,1,theta,phi);
	  }
	  else{
	    //Hydrogen
	    double *n1sH;
	    R=0;
	    if (r<=outerMostPointHydrogen) R = alglib::spline1dcalc(lEqualsZeroSplineHydrogen,r);
	    //1s
	    eigenVectors[4+atom][nodeID] = R*boost::math::spherical_harmonic_r(0,0,theta,phi);
	  }
	}
      }
    }
  }
  //update ghosts for eigenVectors

}

//
void dftClass::readPSI(){
  const unsigned int numAtomTypes=2; //2 for CH4
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
    if (tempFile.is_open()) {
      while (!tempFile.eof()) {
	for(int i = 0; i < numColumns; ++i) tempFile>>rowData[i];
	singleAtomPSI[atom].push_back(rowData);
      }
    }
    tempFile.close();
  }
  //
  readPSIRadialValues(singleAtomPSI);
}
