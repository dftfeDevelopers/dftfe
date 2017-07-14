//
//Initlialize rho by reading in single-atom electron-density and fit a spline
//
void dftClass::initRho()
{ 
  computing_timer.enter_section("dftClass init density"); 

  //Reading single atom rho initial guess
  pcout << "reading initial guess for rho\n";
  std::map<unsigned int, alglib::spline1dinterpolant> denSpline;
  std::map<unsigned int, std::vector<std::vector<double> > > singleAtomElectronDensity;
  std::map<unsigned int, double> outerMostPointDen;
    
  //loop over atom types
  for (std::set<unsigned int>::iterator it=atomTypes.begin(); it!=atomTypes.end(); it++)
    {
      char densityFile[256];
      if(isPseudopotential)
	{
	  sprintf(densityFile, "../../../../data/electronicStructure/pseudoPotential/z%u/singleAtomData/density.inp", *it);
	}
      else
	{
	  sprintf(densityFile, "../../../../data/electronicStructure/allElectron/z%u/singleAtomData/density.inp", *it);
	}
   
      readFile(2, singleAtomElectronDensity[*it], densityFile);
      unsigned int numRows = singleAtomElectronDensity[*it].size()-1;
      std::vector<double> xData(numRows), yData(numRows);
      for(unsigned int irow = 0; irow < numRows; ++irow)
	{
	  xData[irow] = singleAtomElectronDensity[*it][irow][0];
	  yData[irow] = singleAtomElectronDensity[*it][irow][1];
	}
  
      //interpolate rho
      alglib::real_1d_array x;
      x.setcontent(numRows,&xData[0]);
      alglib::real_1d_array y;
      y.setcontent(numRows,&yData[0]);
      alglib::ae_int_t natural_bound_type = 1;
      spline1dbuildcubic(x, y, numRows, natural_bound_type, 0.0, natural_bound_type, 0.0, denSpline[*it]);
      outerMostPointDen[*it]= xData[numRows-1];
    }
  

  //Initialize rho
  QGauss<3>  quadrature_formula(FEOrder+1);
  FEValues<3> fe_values (FE, quadrature_formula, update_values);
  const unsigned int n_q_points    = quadrature_formula.size();

  //Initialize electron density table storage
  rhoInValues=new std::map<dealii::CellId, std::vector<double> >;
  rhoInVals.push_back(rhoInValues);

  //loop over elements
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  for (; cell!=endc; ++cell) 
    {
      if (cell->is_locally_owned())
	{
	  (*rhoInValues)[cell->id()]=std::vector<double>(n_q_points);
	  double *rhoInValuesPtr = &((*rhoInValues)[cell->id()][0]);
	  for (unsigned int q = 0; q < n_q_points; ++q)
	    {
	      MappingQ<3> test(1); 
	      Point<3> quadPoint(test.transform_unit_to_real_cell(cell, fe_values.get_quadrature().point(q)));
	      double rhoValueAtQuadPt=0.0;
	      //loop over atoms
	      for (unsigned int n=0; n<atomLocations.size(); n++)
		{
		  Point<3> atom(atomLocations[n][2],atomLocations[n][3],atomLocations[n][4]);
		  double distanceToAtom=quadPoint.distance(atom);
		  if(distanceToAtom <= outerMostPointDen[atomLocations[n][0]])
		    {
		      rhoValueAtQuadPt+=alglib::spline1dcalc(denSpline[atomLocations[n][0]], distanceToAtom);
		    }
		}
	      rhoInValuesPtr[q]= 1.0;//std::abs(rhoValueAtQuadPt);//1.0
	    }
	}
    } 

#ifdef xc_id
  //loop over elements
#if xc_id == 4
  gradRhoInValues = new std::map<dealii::CellId, std::vector<double> >;

  gradRhoInVals.push_back(gradRhoInValues);


  cell = dofHandler.begin_active();
  for(; cell!=endc; ++cell) 
    {
      if(cell->is_locally_owned())
	{
	  (*gradRhoInValues)[cell->id()]=std::vector<double>(3*n_q_points);

	  double *gradRhoInValuesPtr = &((*gradRhoInValues)[cell->id()][0]);

	  for (unsigned int q = 0; q < n_q_points; ++q)
	    {
	      MappingQ<3> test(1); 
	      Point<3> quadPoint(test.transform_unit_to_real_cell(cell, fe_values.get_quadrature().point(q)));
	      double gradRhoXValueAtQuadPt = 0.0;
	      double gradRhoYValueAtQuadPt = 0.0;
	      double gradRhoZValueAtQuadPt = 0.0;
	      //loop over atoms
	      for (unsigned int n = 0; n < atomLocations.size(); n++)
		{
		  Point<3> atom(atomLocations[n][2],atomLocations[n][3],atomLocations[n][4]);
		  double distanceToAtom = quadPoint.distance(atom);
		  if(distanceToAtom <= outerMostPointDen[atomLocations[n][0]])
		    {
		      //rhoValueAtQuadPt+=alglib::spline1dcalc(denSpline[atomLocations[n][0]], distanceToAtom);
		      double value,radialDensityFirstDerivative,radialDensitySecondDerivative;
		      alglib::spline1ddiff(denSpline[atomLocations[n][0]],
					   distanceToAtom,
					   value,
					   radialDensityFirstDerivative,
					   radialDensitySecondDerivative);
										      
		      gradRhoXValueAtQuadPt += radialDensityFirstDerivative*((quadPoint[0] - atomLocations[n][2])/distanceToAtom);
		      gradRhoYValueAtQuadPt += radialDensityFirstDerivative*((quadPoint[1] - atomLocations[n][3])/distanceToAtom);
		      gradRhoZValueAtQuadPt += radialDensityFirstDerivative*((quadPoint[2] - atomLocations[n][4])/distanceToAtom);
		    }
		}
	      int signRho = (*rhoInValues)[cell->id()][q]/std::abs((*rhoInValues)[cell->id()][q]);
	      gradRhoInValuesPtr[3*q+0] = signRho*gradRhoXValueAtQuadPt;
	      gradRhoInValuesPtr[3*q+1] = signRho*gradRhoYValueAtQuadPt;
	      gradRhoInValuesPtr[3*q+2] = signRho*gradRhoZValueAtQuadPt;
	    }
	}
    } 
#endif
#endif

  //
  //Normalize rho
  //
  double charge = totalCharge(rhoInValues);
  char buffer[100];
  sprintf(buffer, "Initial total charge: %18.10e \n", charge);
  pcout << buffer;
  //scaling rho
  cell = dofHandler.begin_active();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      for (unsigned int q=0; q<n_q_points; ++q){
	(*rhoInValues)[cell->id()][q]*=((double)numElectrons)/charge;
      }
    }
  }
  double chargeAfterScaling = totalCharge(rhoInValues);
  sprintf(buffer, "Initial total charge after scaling: %18.10e \n", chargeAfterScaling);
  pcout << buffer;
  
  
  //
  computing_timer.exit_section("dftClass init density"); 
}
