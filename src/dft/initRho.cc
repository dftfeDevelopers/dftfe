// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
// @author Shiva Rudraraju, Phani Motamarri, Sambit Das
//

//
//Initialize rho by reading in single-atom electron-density and fit a spline
//
#include <dftParameters.h>

template<unsigned int FEOrder>
void dftClass<FEOrder>::clearRhoData()
{
  rhoInVals.clear();
  rhoOutVals.clear();
  gradRhoInVals.clear();
  gradRhoOutVals.clear();
  rhoInValsSpinPolarized.clear();
  rhoOutValsSpinPolarized.clear();
  gradRhoInValsSpinPolarized.clear();
  gradRhoOutValsSpinPolarized.clear();
  dFBroyden.clear();
  graddFBroyden.clear() ;
  uBroyden.clear();
  gradUBroyden.clear() ;
  d_rhoInNodalVals.clear();
  d_rhoOutNodalVals.clear();
}

template<unsigned int FEOrder>
void dftClass<FEOrder>::initRho()
{
  computing_timer.enter_section("initialize density");

  //clear existing data
  clearRhoData();

  //Reading single atom rho initial guess
  pcout <<std::endl<< "Reading initial guess for electron-density....."<<std::endl;
  std::map<unsigned int, alglib::spline1dinterpolant> denSpline;
  std::map<unsigned int, std::vector<std::vector<double> > > singleAtomElectronDensity;
  std::map<unsigned int, double> outerMostPointDen;
  const double truncationTol=1e-8;

  //loop over atom types
  for (std::set<unsigned int>::iterator it=atomTypes.begin(); it!=atomTypes.end(); it++)
    {
      char densityFile[256];
      if(dftParameters::isPseudopotential)
	{
	  sprintf(densityFile,"temp/z%u/density.inp",*it);
	}
      else
	{
	  sprintf(densityFile, "%s/data/electronicStructure/allElectron/z%u/singleAtomData/density.inp", DFT_PATH, *it);
	}

      dftUtils::readFile(2, singleAtomElectronDensity[*it], densityFile);
      unsigned int numRows = singleAtomElectronDensity[*it].size()-1;
      std::vector<double> xData(numRows), yData(numRows);

      unsigned int maxRowId=0;
      for(unsigned int irow = 0; irow < numRows; ++irow)
	{
	  xData[irow] = singleAtomElectronDensity[*it][irow][0];
	  yData[irow] = singleAtomElectronDensity[*it][irow][1];

	  if (yData[irow]>truncationTol)
	    maxRowId=irow;
	}

      //interpolate rho
      alglib::real_1d_array x;
      x.setcontent(numRows,&xData[0]);
      alglib::real_1d_array y;
      y.setcontent(numRows,&yData[0]);
      alglib::ae_int_t natural_bound_type_L = 1;
      alglib::ae_int_t natural_bound_type_R = 1;
      spline1dbuildcubic(x, y, numRows, natural_bound_type_L, 0.0, natural_bound_type_R, 0.0, denSpline[*it]);
      outerMostPointDen[*it]= xData[maxRowId];
    }


  //Initialize rho
  QGauss<3>  quadrature_formula(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (FE, quadrature_formula, update_quadrature_points);
  const unsigned int n_q_points    = quadrature_formula.size();

  //Initialize electron density table storage for rhoIn

  rhoInVals.push_back(std::map<dealii::CellId, std::vector<double> >());
  rhoInValues=&(rhoInVals.back());
  if(dftParameters::spinPolarized==1)
    {
      rhoInValsSpinPolarized.push_back(std::map<dealii::CellId, std::vector<double> >());
      rhoInValuesSpinPolarized=&(rhoInValsSpinPolarized.back());
    }

  if(dftParameters::xc_id == 4)
    {
      gradRhoInVals.push_back(std::map<dealii::CellId, std::vector<double> >());
      gradRhoInValues= &(gradRhoInVals.back());
      //
      if(dftParameters::spinPolarized==1)
        {
          gradRhoInValsSpinPolarized.push_back(std::map<dealii::CellId, std::vector<double> >());
          gradRhoInValuesSpinPolarized=&(gradRhoInValsSpinPolarized.back());
        }
    }

  //Initialize electron density table storage for rhoOut only for Anderson with Kerker
  //for other mixing schemes it is done in density.cc as we need to do this initialization
  //every SCF
  if(dftParameters::mixingMethod == "ANDERSON_WITH_KERKER")
    {
      rhoOutVals.push_back(std::map<dealii::CellId,std::vector<double> > ());
      rhoOutValues = &(rhoOutVals.back());

        if(dftParameters::xc_id == 4)
	  {
	    gradRhoOutVals.push_back(std::map<dealii::CellId, std::vector<double> >());
	    gradRhoOutValues= &(gradRhoOutVals.back());
	  }
    }
  
  

  //
  //get number of image charges used only for periodic
  //
  const int numberImageCharges = d_imageIdsTrunc.size();

  if(dftParameters::mixingMethod == "ANDERSON_WITH_KERKER")
    {
      IndexSet locallyOwnedSet;
      DoFTools::extract_locally_owned_dofs(d_dofHandlerPRefined,locallyOwnedSet);
      std::vector<IndexSet::size_type> locallyOwnedDOFs;
      locallyOwnedSet.fill_index_vector(locallyOwnedDOFs);
      unsigned int numberDofs = locallyOwnedDOFs.size();
      std::map<types::global_dof_index, Point<3> > supportPointsPRefined;
      DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), d_dofHandlerPRefined, supportPointsPRefined);

      //d_matrixFreeDataPRefined.initialize_dof_vector(d_rhoInNodalValues);

      for(unsigned int dof = 0; dof < numberDofs; ++dof)
	{
	  const dealii::types::global_dof_index dofID = locallyOwnedDOFs[dof];
	  Point<3> nodalCoor = supportPointsPRefined[dofID];
	  if(!d_constraintsPRefined.is_constrained(dofID))
	    {
	      //loop over atoms and superimpose electron-density at a given dof from all atoms
	      double rhoNodalValue = 0.0;
	      for(unsigned int iAtom = 0; iAtom < atomLocations.size(); ++iAtom)
		{
		  Point<3> atom(atomLocations[iAtom][2],atomLocations[iAtom][3],atomLocations[iAtom][4]);
		  double distanceToAtom = nodalCoor.distance(atom);
		  if(distanceToAtom <= outerMostPointDen[atomLocations[iAtom][0]])
		    rhoNodalValue += alglib::spline1dcalc(denSpline[atomLocations[iAtom][0]], distanceToAtom);
		  else
		    rhoNodalValue += 0.0;
		}

	      //loop over image charges and do as above
	      for(int iImageCharge = 0; iImageCharge < numberImageCharges; ++iImageCharge)
		{
		  Point<3> imageAtom(d_imagePositionsTrunc[iImageCharge][0],
			             d_imagePositionsTrunc[iImageCharge][1],
				     d_imagePositionsTrunc[iImageCharge][2]);
		  double distanceToAtom = nodalCoor.distance(imageAtom);
		  int masterAtomId = d_imageIdsTrunc[iImageCharge];
		  if(distanceToAtom <= outerMostPointDen[atomLocations[masterAtomId][0]])
		    rhoNodalValue += alglib::spline1dcalc(denSpline[atomLocations[masterAtomId][0]], distanceToAtom);
		  else
		    rhoNodalValue += 0.0;
		}
	      d_rhoInNodalValues.local_element(dof) = std::abs(rhoNodalValue);
	    }
	}

      d_rhoInNodalValues.update_ghost_values();

      //normalize rho
      const double charge = totalCharge(d_matrixFreeDataPRefined,
					d_rhoInNodalValues);

       
      const double scalingFactor = ((double)numElectrons)/charge;

      //scale nodal vector with scalingFactor
      d_rhoInNodalValues *= scalingFactor;

      //push the rhoIn to deque storing the history of nodal values
      d_rhoInNodalVals.push_back(d_rhoInNodalValues);

      if (dftParameters::verbosity>=3)
	{
          pcout<<"Total Charge before Normalizing nodal Rho:  "<<charge<<std::endl;
	  pcout<<"Total Charge after Normalizing nodal Rho: "<< totalCharge(d_matrixFreeDataPRefined,d_rhoInNodalValues)<<std::endl;
	}

      if(dftParameters::xc_id == 4)
	{
	  gradRhoInVals.push_back(std::map<dealii::CellId,std::vector<double> >());
	  gradRhoInValues=&(gradRhoInVals.back());
	}

      interpolateNodalDataToQuadratureData(d_matrixFreeDataPRefined,
					   d_rhoInNodalValues,
					   *rhoInValues,
					   *gradRhoInValues,
                                           *gradRhoInValues,
					   dftParameters::xc_id == 4);
      normalizeRho();

      /*FEEvaluation<C_DIM,C_num1DKerkerPoly<FEOrder>(),C_num1DQuad<FEOrder>(),1,double> rhoEval(d_matrixFreeDataPRefined,0,1);
      const unsigned int numQuadPoints = rhoEval.n_q_points; 
      DoFHandler<C_DIM>::active_cell_iterator subCellPtr;
      for(unsigned int cell = 0; cell < d_matrixFreeDataPRefined.n_macro_cells(); ++cell)
	{
	  rhoEval.reinit(cell);
	  rhoEval.read_dof_values(d_rhoInNodalValues);
	  rhoEval.evaluate(true,true);
	  for(unsigned int iSubCell = 0; iSubCell < d_matrixFreeDataPRefined.n_components_filled(cell); ++iSubCell)
	    {
	      subCellPtr= d_matrixFreeDataPRefined.get_cell_iterator(cell,iSubCell);
	      dealii::CellId subCellId=subCellPtr->id();
	      (*rhoInValues)[subCellId] = std::vector<double>(numQuadPoints);
	      std::vector<double> & tempVec = rhoInValues->find(subCellId)->second;
	      for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
		{
		  tempVec[q_point] = rhoEval.get_value(q_point)[iSubCell];
		}
	    }
      
	  if(dftParameters::xc_id == 4)
	    {
	      for(unsigned int iSubCell = 0; iSubCell < d_matrixFreeDataPRefined.n_components_filled(cell); ++iSubCell)
		{
		  subCellPtr= d_matrixFreeDataPRefined.get_cell_iterator(cell,iSubCell);
		  dealii::CellId subCellId=subCellPtr->id();
		  (*gradRhoInValues)[subCellId]=std::vector<double>(3*numQuadPoints);
		  std::vector<double> & tempVec = gradRhoInValues->find(subCellId)->second;
		  for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
		    {
		      tempVec[3*q_point + 0] = rhoEval.get_gradient(q_point)[0][iSubCell];
		      tempVec[3*q_point + 1] = rhoEval.get_gradient(q_point)[1][iSubCell];
		      tempVec[3*q_point + 2] = rhoEval.get_gradient(q_point)[2][iSubCell];
		    }
		}
	    }

	}*/

    }
  else
    {
      //loop over elements
      typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
      for (; cell!=endc; ++cell)
	{
	  if (cell->is_locally_owned())
	    {
	      fe_values.reinit(cell);
	      (*rhoInValues)[cell->id()]=std::vector<double>(n_q_points);
	      double *rhoInValuesPtr = &((*rhoInValues)[cell->id()][0]);

	      double *rhoInValuesSpinPolarizedPtr;
	      if(dftParameters::spinPolarized==1)
		{
		  (*rhoInValuesSpinPolarized)[cell->id()]=std::vector<double>(2*n_q_points);
		  rhoInValuesSpinPolarizedPtr = &((*rhoInValuesSpinPolarized)[cell->id()][0]);
		}
	      for (unsigned int q = 0; q < n_q_points; ++q)
		{
		  const Point<3> & quadPoint=fe_values.quadrature_point(q);
		  double rhoValueAtQuadPt = 0.0;

		  //loop over atoms
		  for (unsigned int n = 0; n < atomLocations.size(); n++)
		    {
		      Point<3> atom(atomLocations[n][2],atomLocations[n][3],atomLocations[n][4]);
		      double distanceToAtom = quadPoint.distance(atom);
		      if(distanceToAtom <= outerMostPointDen[atomLocations[n][0]])
			{
			  rhoValueAtQuadPt += alglib::spline1dcalc(denSpline[atomLocations[n][0]], distanceToAtom);
			}
		      else
			{
			  rhoValueAtQuadPt += 0.0;
			}
		    }

		  //loop over image charges
		  for(int iImageCharge = 0; iImageCharge < numberImageCharges; ++iImageCharge)
		    {
		      Point<3> imageAtom(d_imagePositionsTrunc[iImageCharge][0],
					 d_imagePositionsTrunc[iImageCharge][1],
					 d_imagePositionsTrunc[iImageCharge][2]);
		      double distanceToAtom = quadPoint.distance(imageAtom);
		      int masterAtomId = d_imageIdsTrunc[iImageCharge];
		      if(distanceToAtom <= outerMostPointDen[atomLocations[masterAtomId][0]])//outerMostPointPseudo[atomLocations[masterAtomId][0]])
			{
			  rhoValueAtQuadPt += alglib::spline1dcalc(denSpline[atomLocations[masterAtomId][0]], distanceToAtom);
			}
		    }

		  rhoInValuesPtr[q] = std::abs(rhoValueAtQuadPt);
		  if(dftParameters::spinPolarized==1)
		    {
		      rhoInValuesSpinPolarizedPtr[2*q+1] =( 0.5 + dftParameters::start_magnetization)*(std::abs(rhoValueAtQuadPt));
		      rhoInValuesSpinPolarizedPtr[2*q] = ( 0.5 - dftParameters::start_magnetization)*(std::abs(rhoValueAtQuadPt));
		    }
		}
	    }
	}


      //loop over elements
      if(dftParameters::xc_id == 4)
	{
	  //
	  cell = dofHandler.begin_active();
	  for(; cell!=endc; ++cell)
	    {
	      if(cell->is_locally_owned())
		{
		  fe_values.reinit(cell);

		  (*gradRhoInValues)[cell->id()]=std::vector<double>(3*n_q_points);
		  double *gradRhoInValuesPtr = &((*gradRhoInValues)[cell->id()][0]);

		  double *gradRhoInValuesSpinPolarizedPtr;
		  if(dftParameters::spinPolarized==1)
		    {
		      (*gradRhoInValuesSpinPolarized)[cell->id()]=std::vector<double>(6*n_q_points);
		      gradRhoInValuesSpinPolarizedPtr = &((*gradRhoInValuesSpinPolarized)[cell->id()][0]);
		    }
		  for (unsigned int q = 0; q < n_q_points; ++q)
		    {
		      const Point<3> & quadPoint=fe_values.quadrature_point(q);
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

		      for(int iImageCharge = 0; iImageCharge < numberImageCharges; ++iImageCharge)
			{
			  Point<3> imageAtom(d_imagePositionsTrunc[iImageCharge][0],
					     d_imagePositionsTrunc[iImageCharge][1],
					     d_imagePositionsTrunc[iImageCharge][2]);
			  double distanceToAtom = quadPoint.distance(imageAtom);
			  int masterAtomId = d_imageIdsTrunc[iImageCharge];
			  if(distanceToAtom <= outerMostPointDen[atomLocations[masterAtomId][0]])//outerMostPointPseudo[atomLocations[masterAtomId][0]])
			    {
			      double value,radialDensityFirstDerivative,radialDensitySecondDerivative;
			      alglib::spline1ddiff(denSpline[atomLocations[masterAtomId][0]],
						   distanceToAtom,
						   value,
						   radialDensityFirstDerivative,
						   radialDensitySecondDerivative);

			      gradRhoXValueAtQuadPt += radialDensityFirstDerivative*((quadPoint[0] - d_imagePositionsTrunc[iImageCharge][0])/distanceToAtom);
			      gradRhoYValueAtQuadPt += radialDensityFirstDerivative*((quadPoint[1] - d_imagePositionsTrunc[iImageCharge][1])/distanceToAtom);
			      gradRhoZValueAtQuadPt += radialDensityFirstDerivative*((quadPoint[2] - d_imagePositionsTrunc[iImageCharge][2])/distanceToAtom);

			    }
			}

		      int signRho = 0 ;
                      /*
                      if (std::abs((*rhoInValues)[cell->id()][q] ) > 1.0E-7)
                          signRho = (*rhoInValues)[cell->id()][q]>0.0?1:-1;
                      */
                      if (std::abs((*rhoInValues)[cell->id()][q] ) > 1.0E-7)
                        int signRho = (*rhoInValues)[cell->id()][q]/std::abs((*rhoInValues)[cell->id()][q]);

		      // KG: the fact that we are forcing gradRho to zero whenever rho is zero is valid. Because rho is always positive, so whenever it is zero, it must have a local minima.
		      //
		      gradRhoInValuesPtr[3*q+0] = signRho*gradRhoXValueAtQuadPt;
		      gradRhoInValuesPtr[3*q+1] = signRho*gradRhoYValueAtQuadPt;
		      gradRhoInValuesPtr[3*q+2] = signRho*gradRhoZValueAtQuadPt;
		      if(dftParameters::spinPolarized==1)
			{
			  gradRhoInValuesSpinPolarizedPtr[6*q+0] =( 0.5 + dftParameters::start_magnetization)*signRho*gradRhoXValueAtQuadPt;
			  gradRhoInValuesSpinPolarizedPtr[6*q+1] = ( 0.5 + dftParameters::start_magnetization)*signRho*gradRhoYValueAtQuadPt;
			  gradRhoInValuesSpinPolarizedPtr[6*q+2] =  ( 0.5 + dftParameters::start_magnetization)*signRho*gradRhoZValueAtQuadPt;
			  gradRhoInValuesSpinPolarizedPtr[6*q+3] =( 0.5 - dftParameters::start_magnetization)*signRho*gradRhoXValueAtQuadPt;
			  gradRhoInValuesSpinPolarizedPtr[6*q+4] = ( 0.5 - dftParameters::start_magnetization)*signRho*gradRhoYValueAtQuadPt;
			  gradRhoInValuesSpinPolarizedPtr[6*q+5] =  ( 0.5 - dftParameters::start_magnetization)*signRho*gradRhoZValueAtQuadPt;
			}
		    }
		}
	    }
	}

      normalizeRho();
    }
  //
  computing_timer.exit_section("initialize density");
}

//
//
//
template <unsigned int FEOrder>
void dftClass<FEOrder>::computeRhoInitialGuessFromPSI(std::vector<std::vector<distributedCPUVec<double>>> eigenVectors)

{
  computing_timer.enter_section("initialize density");

  //clear existing data
  clearRhoData();

  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (FEEigen, quadrature, update_values | update_gradients);
  const unsigned int num_quad_points = quadrature.size();

  //Initialize electron density table storage

  rhoInVals.push_back(std::map<dealii::CellId, std::vector<double> >());
  rhoInValues=&(rhoInVals.back());
  if (dftParameters::spinPolarized==1)
    {
      rhoInValsSpinPolarized.push_back(std::map<dealii::CellId, std::vector<double> >());
      rhoInValuesSpinPolarized=&(rhoInValsSpinPolarized.back());
    }

  if(dftParameters::xc_id == 4)
    {
      gradRhoInVals.push_back(std::map<dealii::CellId, std::vector<double> >());
      gradRhoInValues= &(gradRhoInVals.back());
      //
      if(dftParameters::spinPolarized==1)
        {
          gradRhoInValsSpinPolarized.push_back(std::map<dealii::CellId, std::vector<double> >());
          gradRhoInValuesSpinPolarized=&(gradRhoInValsSpinPolarized.back());
        }
    }

  //temp arrays
  std::vector<double> rhoTemp(num_quad_points), rhoTempSpinPolarized(2*num_quad_points), rhoIn(num_quad_points), rhoInSpinPolarized(2*num_quad_points);
  std::vector<double> gradRhoTemp(3*num_quad_points), gradRhoTempSpinPolarized(6*num_quad_points),gradRhoIn(3*num_quad_points), gradRhoInSpinPolarized(6*num_quad_points);

  //loop over locally owned elements
  typename DoFHandler<3>::active_cell_iterator cell = dofHandlerEigen.begin_active(), endc = dofHandlerEigen.end();
  for (; cell!=endc; ++cell)
    if(cell->is_locally_owned())
      {
	fe_values.reinit (cell);

	(*rhoInValues)[cell->id()] = std::vector<double>(num_quad_points);
	std::fill(rhoTemp.begin(),rhoTemp.end(),0.0); std::fill(rhoIn.begin(),rhoIn.end(),0.0);
	if (dftParameters::spinPolarized==1)
	  {
	    (*rhoInValuesSpinPolarized)[cell->id()] = std::vector<double>(2*num_quad_points);
	    std::fill(rhoTempSpinPolarized.begin(),rhoTempSpinPolarized.end(),0.0);
	  }

#ifdef USE_COMPLEX
	std::vector<Vector<double> > tempPsi(num_quad_points), tempPsi2(num_quad_points);
	for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
	  {
	    tempPsi[q_point].reinit(2);
	    tempPsi2[q_point].reinit(2);
	  }
#else
	std::vector<double> tempPsi(num_quad_points), tempPsi2(num_quad_points);
#endif



	if(dftParameters::xc_id == 4)//GGA
	  {
	    (*gradRhoInValues)[cell->id()] = std::vector<double>(3*num_quad_points);
	    std::fill(gradRhoTemp.begin(),gradRhoTemp.end(),0.0);
	    if (dftParameters::spinPolarized==1)
	      {
		(*gradRhoInValuesSpinPolarized)[cell->id()] = std::vector<double>(6*num_quad_points);
		std::fill(gradRhoTempSpinPolarized.begin(),gradRhoTempSpinPolarized.end(),0.0);
	      }
#ifdef USE_COMPLEX
	    std::vector<std::vector<Tensor<1,3,double> > > tempGradPsi(num_quad_points), tempGradPsi2(num_quad_points);
	    for(unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
	      {
		tempGradPsi[q_point].resize(2);
		tempGradPsi2[q_point].resize(2);
	      }
#else
	    std::vector<Tensor<1,3,double> > tempGradPsi(num_quad_points), tempGradPsi2(num_quad_points);
#endif


	    for(int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
	      {
		for(unsigned int i=0; i<d_numEigenValues; ++i)
		  {
		    fe_values.get_function_values(eigenVectors[(1+dftParameters::spinPolarized)*kPoint][i], tempPsi);
		    if(dftParameters::spinPolarized==1)
		      fe_values.get_function_values(eigenVectors[(1+dftParameters::spinPolarized)*kPoint+1][i], tempPsi2);
		    //
		    fe_values.get_function_gradients(eigenVectors[(1+dftParameters::spinPolarized)*kPoint][i],tempGradPsi);
		    if(dftParameters::spinPolarized==1)
		      fe_values.get_function_gradients(eigenVectors[(1+dftParameters::spinPolarized)*kPoint+1][i], tempGradPsi2);

		    for(unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		      {
			double factor = (eigenValues[kPoint][i]-fermiEnergy)/(C_kb*dftParameters::TVal);
			double partialOccupancy = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)):1.0/(1.0 + std::exp(factor));
			//
			factor=(eigenValues[kPoint][i+dftParameters::spinPolarized*d_numEigenValues]-fermiEnergy)/(C_kb*dftParameters::TVal);
			double partialOccupancy2 = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)):1.0/(1.0 + std::exp(factor));
#ifdef USE_COMPLEX
			if(dftParameters::spinPolarized==1)
			  {
			    rhoTempSpinPolarized[2*q_point] += partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempPsi[q_point](0) + tempPsi[q_point](1)*tempPsi[q_point](1));
			    rhoTempSpinPolarized[2*q_point+1] += partialOccupancy2*d_kPointWeights[kPoint]*(tempPsi2[q_point](0)*tempPsi2[q_point](0) + tempPsi2[q_point](1)*tempPsi2[q_point](1));
			    //
			    gradRhoTempSpinPolarized[6*q_point + 0] +=
			      2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][0] + tempPsi[q_point](1)*tempGradPsi[q_point][1][0]);
			    gradRhoTempSpinPolarized[6*q_point + 1] +=
			      2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][1] + tempPsi[q_point](1)*tempGradPsi[q_point][1][1]);
			    gradRhoTempSpinPolarized[6*q_point + 2] +=
			      2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][2] + tempPsi[q_point](1)*tempGradPsi[q_point][1][2]);
			    gradRhoTempSpinPolarized[6*q_point + 3] +=
			      2.0*partialOccupancy2*d_kPointWeights[kPoint]*(tempPsi2[q_point](0)*tempGradPsi2[q_point][0][0] + tempPsi2[q_point](1)*tempGradPsi2[q_point][1][0]);
			    gradRhoTempSpinPolarized[6*q_point + 4] +=
			      2.0*partialOccupancy2*d_kPointWeights[kPoint]*(tempPsi2[q_point](0)*tempGradPsi2[q_point][0][1] + tempPsi2[q_point](1)*tempGradPsi2[q_point][1][1]);
			    gradRhoTempSpinPolarized[6*q_point + 5] +=
			      2.0*partialOccupancy2*d_kPointWeights[kPoint]*(tempPsi2[q_point](0)*tempGradPsi2[q_point][0][2] + tempPsi2[q_point](1)*tempGradPsi2[q_point][1][2]);
			  }
			else
			  {
			    rhoTemp[q_point] += 2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempPsi[q_point](0) + tempPsi[q_point](1)*tempPsi[q_point](1));
			    gradRhoTemp[3*q_point + 0] += 2.0*2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][0] + tempPsi[q_point](1)*tempGradPsi[q_point][1][0]);
			    gradRhoTemp[3*q_point + 1] += 2.0*2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][1] + tempPsi[q_point](1)*tempGradPsi[q_point][1][1]);
			    gradRhoTemp[3*q_point + 2] += 2.0*2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempGradPsi[q_point][0][2] + tempPsi[q_point](1)*tempGradPsi[q_point][1][2]);
			  }
#else
			if(dftParameters::spinPolarized==1)
			  {
			    rhoTempSpinPolarized[2*q_point] += partialOccupancy*tempPsi[q_point]*tempPsi[q_point];
			    rhoTempSpinPolarized[2*q_point+1] += partialOccupancy2*tempPsi2[q_point]*tempPsi2[q_point];
			    gradRhoTempSpinPolarized[6*q_point + 0] += 2.0*partialOccupancy*(tempPsi[q_point]*tempGradPsi[q_point][0]) ;
			    gradRhoTempSpinPolarized[6*q_point + 1] +=  2.0*partialOccupancy*(tempPsi[q_point]*tempGradPsi[q_point][1]) ;
			    gradRhoTempSpinPolarized[6*q_point + 2] += 2.0*partialOccupancy*(tempPsi[q_point]*tempGradPsi[q_point][2]) ;
			    gradRhoTempSpinPolarized[6*q_point + 3] +=  2.0*partialOccupancy2*(tempPsi2[q_point]*tempGradPsi2[q_point][0]);
			    gradRhoTempSpinPolarized[6*q_point + 4] += 2.0*partialOccupancy2*(tempPsi2[q_point]*tempGradPsi2[q_point][1]) ;
			    gradRhoTempSpinPolarized[6*q_point + 5] += 2.0*partialOccupancy2*(tempPsi2[q_point]*tempGradPsi2[q_point][2]) ;
			  }
			else
			  {
			    rhoTemp[q_point] += 2.0*partialOccupancy*tempPsi[q_point]*tempPsi[q_point];//std::pow(tempPsi[q_point],2.0);
			    gradRhoTemp[3*q_point + 0] += 2.0*2.0*partialOccupancy*tempPsi[q_point]*tempGradPsi[q_point][0];
			    gradRhoTemp[3*q_point + 1] += 2.0*2.0*partialOccupancy*tempPsi[q_point]*tempGradPsi[q_point][1];
			    gradRhoTemp[3*q_point + 2] += 2.0*2.0*partialOccupancy*tempPsi[q_point]*tempGradPsi[q_point][2];
			  }

#endif
		      }
		  }
	      }

	    //  gather density from all pools
	    int numPoint = num_quad_points ;
	    MPI_Allreduce(&rhoTemp[0], &rhoIn[0], numPoint, MPI_DOUBLE, MPI_SUM, interpoolcomm) ;
	    MPI_Allreduce(&gradRhoTemp[0], &gradRhoIn[0], 3*numPoint, MPI_DOUBLE, MPI_SUM, interpoolcomm) ;
	    if (dftParameters::spinPolarized==1) {
	      MPI_Allreduce(&rhoTempSpinPolarized[0], &rhoInSpinPolarized[0], 2*numPoint, MPI_DOUBLE, MPI_SUM, interpoolcomm) ;
	      MPI_Allreduce(&gradRhoTempSpinPolarized[0], &gradRhoInSpinPolarized[0], 6*numPoint, MPI_DOUBLE, MPI_SUM, interpoolcomm) ;
	    }

	    //


	    for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
	      {
		if(dftParameters::spinPolarized==1)
		  {
		    (*rhoInValuesSpinPolarized)[cell->id()][2*q_point]=rhoInSpinPolarized[2*q_point] ;
		    (*rhoInValuesSpinPolarized)[cell->id()][2*q_point+1]=rhoInSpinPolarized[2*q_point+1] ;
		    (*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 0] = gradRhoInSpinPolarized[6*q_point + 0];
		    (*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 1] = gradRhoInSpinPolarized[6*q_point + 1];
		    (*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 2] = gradRhoInSpinPolarized[6*q_point + 2];
		    (*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 3] = gradRhoInSpinPolarized[6*q_point + 3];
		    (*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 4] = gradRhoInSpinPolarized[6*q_point + 4];
		    (*gradRhoInValuesSpinPolarized)[cell->id()][6*q_point + 5] = gradRhoInSpinPolarized[6*q_point + 5];
		    //
		    (*rhoInValues)[cell->id()][q_point]= rhoInSpinPolarized[2*q_point] + rhoInSpinPolarized[2*q_point+1];
		    (*gradRhoInValues)[cell->id()][3*q_point + 0] = gradRhoInSpinPolarized[6*q_point + 0] + gradRhoInSpinPolarized[6*q_point + 3];
		    (*gradRhoInValues)[cell->id()][3*q_point + 1] = gradRhoInSpinPolarized[6*q_point + 1] + gradRhoInSpinPolarized[6*q_point + 4];
		    (*gradRhoInValues)[cell->id()][3*q_point + 2] = gradRhoInSpinPolarized[6*q_point + 2] + gradRhoInSpinPolarized[6*q_point + 5];
		  }
		else
		  {
		    (*rhoInValues)[cell->id()][q_point]  = rhoIn[q_point];
		    (*gradRhoInValues)[cell->id()][3*q_point + 0] = gradRhoIn[3*q_point + 0];
		    (*gradRhoInValues)[cell->id()][3*q_point + 1] = gradRhoIn[3*q_point + 1];
		    (*gradRhoInValues)[cell->id()][3*q_point + 2] = gradRhoIn[3*q_point + 2];
		  }
	      }

	  }
	else
	  {
	    for(int kPoint = 0; kPoint < d_kPointWeights.size(); ++kPoint)
	      {
		for(unsigned int i=0; i<d_numEigenValues; ++i)
		  {
		    fe_values.get_function_values(eigenVectors[(1+dftParameters::spinPolarized)*kPoint][i], tempPsi);
		    if(dftParameters::spinPolarized==1)
		      fe_values.get_function_values(eigenVectors[(1+dftParameters::spinPolarized)*kPoint+1][i], tempPsi2);

		    for(unsigned int q_point=0; q_point<num_quad_points; ++q_point)
		      {
			double factor=(eigenValues[kPoint][i]-fermiEnergy)/(C_kb*dftParameters::TVal);
			double partialOccupancy = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)):1.0/(1.0 + std::exp(factor));
			//
			factor=(eigenValues[kPoint][i+dftParameters::spinPolarized*d_numEigenValues]-fermiEnergy)/(C_kb*dftParameters::TVal);
			double partialOccupancy2 = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)):1.0/(1.0 + std::exp(factor));
#ifdef USE_COMPLEX
			if(dftParameters::spinPolarized==1)
			  {
			    rhoTempSpinPolarized[2*q_point] += partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempPsi[q_point](0) + tempPsi[q_point](1)*tempPsi[q_point](1));
			    rhoTempSpinPolarized[2*q_point+1] += partialOccupancy2*d_kPointWeights[kPoint]*(tempPsi2[q_point](0)*tempPsi2[q_point](0) + tempPsi2[q_point](1)*tempPsi2[q_point](1));
			  }
			else
			  rhoTemp[q_point] += 2.0*partialOccupancy*d_kPointWeights[kPoint]*(tempPsi[q_point](0)*tempPsi[q_point](0) + tempPsi[q_point](1)*tempPsi[q_point](1));
#else
			if(dftParameters::spinPolarized==1)
			  {
			    rhoTempSpinPolarized[2*q_point] += partialOccupancy*tempPsi[q_point]*tempPsi[q_point];
			    rhoTempSpinPolarized[2*q_point+1] += partialOccupancy2*tempPsi2[q_point]*tempPsi2[q_point];
			  }
			else
			  rhoTemp[q_point] += 2.0*partialOccupancy*tempPsi[q_point]*tempPsi[q_point];//std::pow(tempPsi[q_point],2.0);
			//
#endif
		      }
		  }
	      }
	    //  gather density from all pools
	    int numPoint = num_quad_points ;
	    MPI_Allreduce(&rhoTemp[0], &rhoIn[0], numPoint, MPI_DOUBLE, MPI_SUM, interpoolcomm) ;
	    if (dftParameters::spinPolarized==1)
	      MPI_Allreduce(&rhoTempSpinPolarized[0], &rhoInSpinPolarized[0], 2*numPoint, MPI_DOUBLE, MPI_SUM, interpoolcomm) ;
	    //
	    for (unsigned int q_point=0; q_point<num_quad_points; ++q_point)
	      {
		if(dftParameters::spinPolarized==1)
		  {
		    (*rhoInValuesSpinPolarized)[cell->id()][2*q_point]=rhoInSpinPolarized[2*q_point] ;
		    (*rhoInValuesSpinPolarized)[cell->id()][2*q_point+1]=rhoInSpinPolarized[2*q_point+1] ;
		    (*rhoInValues)[cell->id()][q_point]= rhoInSpinPolarized[2*q_point] + rhoInSpinPolarized[2*q_point+1];
		  }
		else
		  (*rhoInValues)[cell->id()][q_point]  = rhoIn[q_point];
	      }

	  }

      }

  normalizeRho();
  //
  computing_timer.exit_section("initialize density");
}

template <unsigned int FEOrder>
void dftClass<FEOrder>::computeNodalRhoFromQuadData()
{
  //
  //compute nodal electron-density from cell quadrature data
  //
  matrix_free_data.initialize_dof_vector(d_rhoNodalField,densityDofHandlerIndex);
  d_rhoNodalField=0;

  std::function<double(const typename dealii::DoFHandler<3>::active_cell_iterator & cell ,
                       const unsigned int q)> funcRho =
    [&](const typename dealii::DoFHandler<3>::active_cell_iterator & cell ,
	const unsigned int q)
    {return (*rhoOutValues).find(cell->id())->second[q];};


  dealii::VectorTools::project<3,distributedCPUVec<double>>
    (dealii::MappingQ1<3,3>(),
     dofHandler,
     constraintsNone,
     QGauss<3>(C_num1DQuad<FEOrder>()),
     funcRho,
     d_rhoNodalField);


  d_rhoNodalField.update_ghost_values();


  if (dftParameters::spinPolarized==1)
    {
      matrix_free_data.initialize_dof_vector(d_rhoNodalFieldSpin0,densityDofHandlerIndex);
      d_rhoNodalFieldSpin0=0;

      std::function<double(const typename dealii::DoFHandler<3>::active_cell_iterator & cell ,
                           const unsigned int q)> funcRhoSpin0 =
	[&](const typename dealii::DoFHandler<3>::active_cell_iterator & cell ,
	    const unsigned int q)
	{return (*rhoOutValuesSpinPolarized).find(cell->id())->second[2*q];};


      dealii::VectorTools::project<3,distributedCPUVec<double>>
	(dealii::MappingQ1<3,3>(),
	 dofHandler,
	 constraintsNone,
	 QGauss<3>(C_num1DQuad<FEOrder>()),
	 funcRhoSpin0,
	 d_rhoNodalFieldSpin0);


      d_rhoNodalFieldSpin0.update_ghost_values();

      matrix_free_data.initialize_dof_vector(d_rhoNodalFieldSpin1,densityDofHandlerIndex);
      d_rhoNodalFieldSpin1=0;

      std::function<double(const typename dealii::DoFHandler<3>::active_cell_iterator & cell ,
                           const unsigned int q)> funcRhoSpin1 =
	[&](const typename dealii::DoFHandler<3>::active_cell_iterator & cell ,
	    const unsigned int q)
	{return (*rhoOutValuesSpinPolarized).find(cell->id())->second[2*q+1];};

      dealii::VectorTools::project<3,distributedCPUVec<double>>
	  (dealii::MappingQ1<3,3>(),
	   dofHandler,
	   constraintsNone,
	   QGauss<3>(C_num1DQuad<FEOrder>()),
	   funcRhoSpin1,
	   d_rhoNodalFieldSpin1);

      d_rhoNodalFieldSpin1.update_ghost_values();
    }
}

template <unsigned int FEOrder>
void dftClass<FEOrder>::initRhoFromPreviousGroundStateRho()

{
  computing_timer.enter_section("init density from prev gs density");

  const unsigned int numQuadPoints =matrix_free_data.get_n_q_points(0);

  if (dftParameters::verbosity>=3)
    pcout<<"L2 Norm Value of previous rho nodal field: "<<d_rhoNodalField.l2_norm()<<std::endl;
  if (dftParameters::verbosity>=3)
    pcout <<std::endl<< "Interpolating previous groundstate density into the new finite element mesh...."<<std::endl;

  std::vector<distributedCPUVec<double>* > rhoFieldsPrevious;
  rhoFieldsPrevious.push_back(&d_rhoNodalField);
  if (dftParameters::spinPolarized==1)
    {
      rhoFieldsPrevious.push_back(&d_rhoNodalFieldSpin0);
      rhoFieldsPrevious.push_back(&d_rhoNodalFieldSpin1);
    }

  distributedCPUVec<double> rhoNodalFieldCurrent;
  distributedCPUVec<double> rhoNodalFieldSpin0Current;
  distributedCPUVec<double> rhoNodalFieldSpin1Current;
  matrix_free_data.initialize_dof_vector(rhoNodalFieldCurrent,densityDofHandlerIndex);
  if (dftParameters::spinPolarized==1)
    {
      matrix_free_data.initialize_dof_vector(rhoNodalFieldSpin0Current,densityDofHandlerIndex);
      matrix_free_data.initialize_dof_vector(rhoNodalFieldSpin1Current,densityDofHandlerIndex);
    }
  std::vector<distributedCPUVec<double>* > rhoFieldsCurrent;
  rhoFieldsCurrent.push_back(&rhoNodalFieldCurrent);
  if (dftParameters::spinPolarized==1)
    {
      rhoFieldsCurrent.push_back(&rhoNodalFieldSpin0Current);
      rhoFieldsCurrent.push_back(&rhoNodalFieldSpin1Current);
    }

  vectorTools::interpolateFieldsFromPreviousMesh interpolateRhoVecsPrev(mpi_communicator);
  interpolateRhoVecsPrev.interpolate(d_mesh.getSerialMeshUnmovedPrevious(),
				     d_mesh.getParallelMeshUnmovedPrevious(),
				     d_mesh.getParallelMeshUnmoved(),
				     FE,
				     FE,
				     rhoFieldsPrevious,
				     rhoFieldsCurrent,
				     &constraintsNone);

  for (unsigned int i=0; i<rhoFieldsCurrent.size();++i)
    rhoFieldsCurrent[i]->update_ghost_values();

  if (dftParameters::verbosity>=3)
    pcout<<"L2 Norm Value of interpolated rho nodal field: "<<rhoNodalFieldCurrent.l2_norm()<<std::endl;
  //clear existing data
  clearRhoData();

  //Initialize and allocate electron density table storage
  resizeAndAllocateRhoTableStorage
    (rhoInVals,
     gradRhoInVals,
     rhoInValsSpinPolarized,
     gradRhoInValsSpinPolarized);

  rhoInValues = &(rhoInVals.back());
  if (dftParameters::spinPolarized==1)
    rhoInValuesSpinPolarized = &(rhoInValsSpinPolarized.back());

  if(dftParameters::xc_id == 4)
    {
      gradRhoInValues = &(gradRhoInVals.back());
      if (dftParameters::spinPolarized==1)
	gradRhoInValuesSpinPolarized = &(gradRhoInValsSpinPolarized.back());
    }

  FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>(),1> rhoEval(matrix_free_data,densityDofHandlerIndex , 0);

  Tensor<1,3,VectorizedArray<double> > zeroTensor;
  for (unsigned int idim=0; idim<3; idim++)
    zeroTensor[idim]=make_vectorized_array(0.0);

  std::vector< VectorizedArray<double> > rhoQuads(numQuadPoints,make_vectorized_array(0.0));
  std::vector< VectorizedArray<double> > rhoQuadsSpin0(numQuadPoints,make_vectorized_array(0.0));
  std::vector< VectorizedArray<double> > rhoQuadsSpin1(numQuadPoints,make_vectorized_array(0.0));
  std::vector<Tensor<1,3,VectorizedArray<double> > > gradRhoQuads(numQuadPoints,zeroTensor);
  std::vector<Tensor<1,3,VectorizedArray<double> > > gradRhoQuadsSpin0(numQuadPoints,zeroTensor);
  std::vector<Tensor<1,3,VectorizedArray<double> > > gradRhoQuadsSpin1(numQuadPoints,zeroTensor);
  for (unsigned int cell=0; cell<matrix_free_data.n_macro_cells(); ++cell)
    {
      rhoEval.reinit(cell);

      rhoEval.read_dof_values_plain(rhoNodalFieldCurrent);
      if(dftParameters::xc_id == 4)
	rhoEval.evaluate(true,true);
      else
	rhoEval.evaluate(true,false);

      for (unsigned int q=0; q<numQuadPoints; ++q)
	{
	  rhoQuads[q]=rhoEval.get_value(q);
	  if(dftParameters::xc_id == 4)
	    gradRhoQuads[q]=rhoEval.get_gradient(q);
	}

      if (dftParameters::spinPolarized==1)
	{
	  rhoEval.read_dof_values_plain(rhoNodalFieldSpin0Current);
	  if(dftParameters::xc_id == 4)
	    rhoEval.evaluate(true,true);
	  else
	    rhoEval.evaluate(true,false);

	  for (unsigned int q=0; q<numQuadPoints; ++q)
	    {
	      rhoQuadsSpin0[q]=rhoEval.get_value(q);
	      if(dftParameters::xc_id == 4)
		gradRhoQuadsSpin0[q]=rhoEval.get_gradient(q);
	    }

	  rhoEval.read_dof_values_plain(rhoNodalFieldSpin1Current);
	  if(dftParameters::xc_id == 4)
	    rhoEval.evaluate(true,true);
	  else
	    rhoEval.evaluate(true,false);

	  for (unsigned int q=0; q<numQuadPoints; ++q)
	    {
	      rhoQuadsSpin1[q]=rhoEval.get_value(q);
	      if(dftParameters::xc_id == 4)
		gradRhoQuadsSpin1[q]=rhoEval.get_gradient(q);
	    }
	}

      const unsigned int numSubCells=matrix_free_data.n_components_filled(cell);

      for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
	{
	  const dealii::CellId subCellId=matrix_free_data.get_cell_iterator(cell,iSubCell)->id();

	  for (unsigned int q=0; q<numQuadPoints; ++q)
	    {
	      if(dftParameters::spinPolarized==1)
		{
		  (*rhoInValuesSpinPolarized)[subCellId][2*q]=rhoQuadsSpin0[q][iSubCell];
		  (*rhoInValuesSpinPolarized)[subCellId][2*q+1]=rhoQuadsSpin1[q][iSubCell];

		  if(dftParameters::xc_id == 4)
		    for(unsigned int idim=0; idim<3; ++idim)
		      {
			(*gradRhoInValuesSpinPolarized)[subCellId][6*q+idim]
			  =gradRhoQuadsSpin0[q][idim][iSubCell];
			(*gradRhoInValuesSpinPolarized)[subCellId][6*q+3+idim]
			  =gradRhoQuadsSpin1[q][idim][iSubCell];
		      }
		}

	      (*rhoInValues)[subCellId][q]= rhoQuads[q][iSubCell];

	      if(dftParameters::xc_id == 4)
		for(unsigned int idim=0; idim<3; ++idim)
		  (*gradRhoInValues)[subCellId][3*q + idim]
		    = gradRhoQuads[q][idim][iSubCell];

	    }//quad point loop
	}//subcell loop
    }//macro cell loop

  //normalize density
  normalizeRho();
  computing_timer.exit_section("init density from prev gs density");
}

//
//Normalize rho
//
template<unsigned int FEOrder>
void dftClass<FEOrder>::normalizeRho()
{
  QGauss<3>  quadrature_formula(C_num1DQuad<FEOrder>());
  const unsigned int n_q_points    = quadrature_formula.size();

  const double charge = totalCharge(dofHandler,
				    rhoInValues);
  const double scaling=((double)numElectrons)/charge;

  if (dftParameters::verbosity>=2)
    pcout<< "initial total charge before normalizing to number of electrons: "<< charge<<std::endl;

  //scaling rho
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      for (unsigned int q=0; q<n_q_points; ++q){
	(*rhoInValues)[cell->id()][q]*=scaling;

	if(dftParameters::xc_id == 4)
	  for (unsigned int idim=0; idim<3; ++idim)
	    (*gradRhoInValues)[cell->id()][3*q+idim]*=scaling;
	if (dftParameters::spinPolarized==1)
	  {
	    (*rhoInValuesSpinPolarized)[cell->id()][2*q+1]*=scaling;
	    (*rhoInValuesSpinPolarized)[cell->id()][2*q]*=scaling;
	    if(dftParameters::xc_id == 4)
	      for (unsigned int idim=0; idim<3; ++idim)
		{
		  (*gradRhoInValuesSpinPolarized)[cell->id()][6*q+idim]*=scaling;
		  (*gradRhoInValuesSpinPolarized)[cell->id()][6*q+3+idim]*=scaling;
		}
	  }
      }
    }
  }
  double chargeAfterScaling = totalCharge(dofHandler,
					  rhoInValues);

  if (dftParameters::verbosity>=1)
    pcout<<"Initial total charge: "<< chargeAfterScaling<<std::endl;
}

//
//Normalize rho
//
template<unsigned int FEOrder>
void dftClass<FEOrder>::normalizeAtomicRhoQuadValues()
{

  const double charge = totalCharge(dofHandler,
				    &d_rhoAtomsValues);
  const double scaling=((double)numElectrons)/charge;

  if (dftParameters::verbosity>=2)
    pcout<< "Total charge rho single atomic before normalizing to number of electrons: "<< charge<<std::endl;

 
  for (auto it1=d_rhoAtomsValues.begin(); it1!=d_rhoAtomsValues.end(); ++it1)
     for (unsigned int i=0; i<(it1->second).size(); ++i)
        (it1->second)[i]*=scaling;

  for (auto it1=d_gradRhoAtomsValues.begin(); it1!=d_gradRhoAtomsValues.end(); ++it1)
     for (unsigned int i=0; i<(it1->second).size(); ++i)
        (it1->second)[i]*=scaling;

  for (auto it1=d_gradRhoAtomsValuesSeparate.begin(); it1!=d_gradRhoAtomsValuesSeparate.end(); ++it1)
     for (auto it2=it1->second.begin(); it2!=it1->second.end(); ++it2)
	  for (unsigned int i=0; i<(it2->second).size(); ++i)
	      (it2->second)[i]*=scaling;
 
  if (dftParameters::xc_id==4)
  {
	  for (auto it1=d_hessianRhoAtomsValues.begin(); it1!=d_hessianRhoAtomsValues.end(); ++it1)
	     for (unsigned int i=0; i<(it1->second).size(); ++i)
		(it1->second)[i]*=scaling;

	  for (auto it1=d_hessianRhoAtomsValuesSeparate.begin(); it1!=d_hessianRhoAtomsValuesSeparate.end(); ++it1)
	     for (auto it2=it1->second.begin(); it2!=it1->second.end(); ++it2)
		  for (unsigned int i=0; i<(it2->second).size(); ++i)
		      (it2->second)[i]*=scaling;
  }

  double chargeAfterScaling = totalCharge(dofHandler,
					  &d_rhoAtomsValues);

  if (dftParameters::verbosity>=2)
    pcout<<"Total charge rho single atomic after normalizing: "<< chargeAfterScaling<<std::endl;
}

template<unsigned int FEOrder>
void dftClass<FEOrder>::initAtomicRho(distributedCPUVec<double> & atomicRho)
{
  computing_timer.enter_section("initialize atomic density for xl bomd");

  //Reading single atom rho initial guess
  if (dftParameters::verbosity>=1)
     pcout <<std::endl<< "Reading initial guess for electron-density....."<<std::endl;

  std::map<unsigned int, alglib::spline1dinterpolant> denSpline;
  std::map<unsigned int, std::vector<std::vector<double> > > singleAtomElectronDensity;
  std::map<unsigned int, double> outerMostPointDen;
  const double truncationTol=1e-8;

  //loop over atom types
  for (std::set<unsigned int>::iterator it=atomTypes.begin(); it!=atomTypes.end(); it++)
    {
      char densityFile[256];
      if(dftParameters::isPseudopotential)
	{
	  sprintf(densityFile,"temp/z%u/density.inp",*it);
	}
      else
	{
	  sprintf(densityFile, "%s/data/electronicStructure/allElectron/z%u/singleAtomData/density.inp", DFT_PATH, *it);
	}

      dftUtils::readFile(2, singleAtomElectronDensity[*it], densityFile);
      unsigned int numRows = singleAtomElectronDensity[*it].size()-1;
      std::vector<double> xData(numRows), yData(numRows);

      unsigned int maxRowId=0;
      for(unsigned int irow = 0; irow < numRows; ++irow)
	{
	  xData[irow] = singleAtomElectronDensity[*it][irow][0];
	  yData[irow] = singleAtomElectronDensity[*it][irow][1];

	  if (yData[irow]>truncationTol)
	    maxRowId=irow;
	}

      //interpolate rho
      alglib::real_1d_array x;
      x.setcontent(numRows,&xData[0]);
      alglib::real_1d_array y;
      y.setcontent(numRows,&yData[0]);
      alglib::ae_int_t natural_bound_type_L = 1;
      alglib::ae_int_t natural_bound_type_R = 1;
      spline1dbuildcubic(x, y, numRows, natural_bound_type_L, 0.0, natural_bound_type_R, 0.0, denSpline[*it]);
      outerMostPointDen[*it]= xData[maxRowId];
    }


  //Initialize rho
  QGauss<3>  quadrature_formula(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (FE, quadrature_formula, update_quadrature_points);
  const unsigned int n_q_points    = quadrature_formula.size();

  //
  //get number of image charges used only for periodic
  //
  const int numberImageCharges = d_imageIdsTrunc.size();

  //IndexSet locallyOwnedSet;
  //DoFTools::extract_locally_owned_dofs(d_dofHandlerPRefined,locallyOwnedSet);
  //std::vector<IndexSet::size_type> locallyOwnedDOFs;
  //locallyOwnedSet.fill_index_vector(locallyOwnedDOFs);
  //unsigned int numberDofs = locallyOwnedDOFs.size();
  std::map<types::global_dof_index, Point<3> > supportPointsPRefined;
  DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), d_dofHandlerPRefined, supportPointsPRefined);

  //d_matrixFreeDataPRefined.initialize_dof_vector(d_rhoInNodalValues);
  std::vector<distributedCPUVec<double>> singleAtomsRho(atomLocations.size()+numberImageCharges);
  for(unsigned int iAtom = 0; iAtom < atomLocations.size()+numberImageCharges; ++iAtom)
  {
    if (iAtom==0)
      d_matrixFreeDataPRefined.initialize_dof_vector(singleAtomsRho[iAtom],1);
    else
      singleAtomsRho[iAtom].reinit(singleAtomsRho[0]);
    
    singleAtomsRho[iAtom]=0.0;
  }

  const std::shared_ptr< const dealii::Utilities::MPI::Partitioner > & partitioner
			     =singleAtomsRho[0].get_partitioner();
  const unsigned int localSize =  partitioner->local_size();
  const unsigned int n_ghosts   = partitioner->n_ghost_indices();
  const unsigned int relevantDofs = localSize + n_ghosts;

  std::vector<bool> isAtomLocal(atomLocations.size()+numberImageCharges,false);
  for(unsigned int dof = 0; dof < relevantDofs; ++dof)
  { 
	  //const dealii::types::global_dof_index dofID = locallyOwnedDOFs[dof];
	  const dealii::types::global_dof_index dofID = partitioner->local_to_global(dof);
	  Point<3> nodalCoor = supportPointsPRefined[dofID];
	  if(!d_constraintsPRefinedOnlyHanging.is_constrained(dofID))
	    {
	      //loop over atoms and superimpose electron-density at a given dof from all atoms
	      double rhoNodalValue = 0.0;
	      for(unsigned int iAtom = 0; iAtom < atomLocations.size(); ++iAtom)
		{
		  Point<3> atom(atomLocations[iAtom][2],atomLocations[iAtom][3],atomLocations[iAtom][4]);
		  double distanceToAtom = nodalCoor.distance(atom);
		  if(distanceToAtom <= outerMostPointDen[atomLocations[iAtom][0]])
                  {
                    const double rhoVal=alglib::spline1dcalc(denSpline[atomLocations[iAtom][0]], distanceToAtom);
                    singleAtomsRho[iAtom].local_element(dof)=std::abs(rhoVal);
		    rhoNodalValue += rhoVal;
                    isAtomLocal[iAtom]=true;
                  }
		}

	      //loop over image charges and do as above
	      for(int iImageCharge = 0; iImageCharge < numberImageCharges; ++iImageCharge)
		{
		  Point<3> imageAtom(d_imagePositionsTrunc[iImageCharge][0],
			             d_imagePositionsTrunc[iImageCharge][1],
				     d_imagePositionsTrunc[iImageCharge][2]);
		  double distanceToAtom = nodalCoor.distance(imageAtom);
		  int masterAtomId = d_imageIdsTrunc[iImageCharge];
		  if(distanceToAtom <= outerMostPointDen[atomLocations[masterAtomId][0]])
                  {
		     const double rhoVal = alglib::spline1dcalc(denSpline[atomLocations[masterAtomId][0]], distanceToAtom);
                     singleAtomsRho[iImageCharge+atomLocations.size()].local_element(dof)=std::abs(rhoVal);
		     rhoNodalValue += rhoVal;
                     isAtomLocal[iImageCharge+atomLocations.size()]=true;
                  }
		}

              if (dof<localSize)
	         atomicRho.local_element(dof) = std::abs(rhoNodalValue);
	    }
   }

   atomicRho.update_ghost_values();

   //for(unsigned int iAtom = 0; iAtom < atomLocations.size()+numberImageCharges; ++iAtom)
   //   singleAtomsRho[iAtom].update_ghost_values(); 

  //normalize rho
  const double charge = totalCharge(d_matrixFreeDataPRefined,
  			             atomicRho);

  VectorizedArray<double> normValueVectorized = make_vectorized_array(0.0);
  FEEvaluation<C_DIM,C_num1DKerkerPoly<FEOrder>(),C_num1DQuad<FEOrder>(),1,double> feEvalObj(d_matrixFreeDataPRefined,1,1);
  const unsigned int numQuadPoints = feEvalObj.n_q_points;
  
       
   const double scalingFactor = ((double)numElectrons)/charge;

   //scale nodal vector with scalingFactor
   atomicRho *= scalingFactor;

   //for(unsigned int iAtom = 0; iAtom < atomLocations.size()+numberImageCharges; ++iAtom)
   //   singleAtomsRho[iAtom]*=scalingFactor; 

   if (dftParameters::verbosity>=3)
   {
          pcout<<"Total Charge before Normalizing nodal Rho:  "<<charge<<std::endl;
	  pcout<<"Total Charge after Normalizing nodal Rho: "<< totalCharge(d_matrixFreeDataPRefined,atomicRho)<<std::endl;
          //pcout<<"Total Charge after Normalizing rho on first atom: "<< totalCharge(d_matrixFreeDataPRefined,singleAtomsRho[0])<<std::endl;
   }

   interpolateNodalDataToQuadratureData(d_matrixFreeDataPRefined,
					atomicRho,
					d_rhoAtomsValues,
		                        d_gradRhoAtomsValues,
                                        d_hessianRhoAtomsValues,
					true,
                                        dftParameters::xc_id==4);
			     
   d_gradRhoAtomsValuesSeparate.clear();

   if (dftParameters::xc_id==4)
      d_hessianRhoAtomsValuesSeparate.clear();

   DoFHandler<C_DIM>::active_cell_iterator subCellPtr;
   for(unsigned int cell = 0; cell < d_matrixFreeDataPRefined.n_macro_cells(); ++cell)
   {
      feEvalObj.reinit(cell);
      for (unsigned int iatom=0; iatom<(atomLocations.size()+numberImageCharges); iatom++)
      {
              if (!isAtomLocal[iatom])
                continue;

	      feEvalObj.read_dof_values(singleAtomsRho[iatom]);
              if (dftParameters::xc_id==4)
	         feEvalObj.evaluate(true,true,true);
              else
	         feEvalObj.evaluate(true,true);

              bool isRhoNonZero=true;
              
              for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
              {
                 VectorizedArray<double> val=feEvalObj.get_value(q_point);
	         for(unsigned int iSubCell = 0; iSubCell < d_matrixFreeDataPRefined.n_components_filled(cell); ++iSubCell)
	         {
                        if (val[iSubCell]>truncationTol)
                           isRhoNonZero=false; 
                 }
              }


              if (!isRhoNonZero)
	              for(unsigned int iSubCell = 0; iSubCell < d_matrixFreeDataPRefined.n_components_filled(cell); ++iSubCell)
		      {
			      subCellPtr= d_matrixFreeDataPRefined.get_cell_iterator(cell,iSubCell);
			      dealii::CellId subCellId=subCellPtr->id();
			      d_gradRhoAtomsValuesSeparate[iatom][subCellId]=std::vector<double>(3*numQuadPoints);
			      std::vector<double> & tempVec1 = d_gradRhoAtomsValuesSeparate[iatom].find(subCellId)->second;
			      for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
			      {
				    tempVec1[3*q_point + 0] = feEvalObj.get_gradient(q_point)[0][iSubCell]*scalingFactor;
				    tempVec1[3*q_point + 1] = feEvalObj.get_gradient(q_point)[1][iSubCell]*scalingFactor;
				    tempVec1[3*q_point + 2] = feEvalObj.get_gradient(q_point)[2][iSubCell]*scalingFactor;
			      }

			      if (dftParameters::xc_id==4)
			      {
			  
				      d_hessianRhoAtomsValuesSeparate[iatom][subCellId]=std::vector<double>(9*numQuadPoints);
				      std::vector<double> & tempVec2 = d_hessianRhoAtomsValuesSeparate[iatom][subCellId];
				      for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
					{
					  const Tensor< 2, 3, VectorizedArray< double> >   & hessianVals=feEvalObj.get_hessian(q_point);
					  for (unsigned int i=0; i<3;i++)
					    for (unsigned int j=0; j<3;j++)
					      tempVec2[9*q_point + 3*i+j] = hessianVals[i][j][iSubCell]*scalingFactor;
					}
			      }
		      }
      }

   }

   normalizeAtomicRhoQuadValues();
   computing_timer.exit_section("initialize atomic density for xl bomd");  
}
