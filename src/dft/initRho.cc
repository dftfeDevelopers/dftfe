// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Shiva Rudraraju (2016), Phani Motamarri (2016)
//

//
//Initlialize rho by reading in single-atom electron-density and fit a spline
//
#include "../../include/dftParameters.h"


template<unsigned int FEOrder>
void dftClass<FEOrder>::initRho()
{ 
  computing_timer.enter_section("initialize density"); 

  //Reading single atom rho initial guess
  pcout <<std::endl<< "Reading initial guess for rho....."<<std::endl;
  std::map<unsigned int, alglib::spline1dinterpolant> denSpline;
  std::map<unsigned int, std::vector<std::vector<double> > > singleAtomElectronDensity;
  std::map<unsigned int, double> outerMostPointDen;
    
  //loop over atom types
  for (std::set<unsigned int>::iterator it=atomTypes.begin(); it!=atomTypes.end(); it++)
    {
      char densityFile[256];
      if(dftParameters::isPseudopotential)
	{
	  if(dftParameters::pseudoProjector==1)
	     sprintf(densityFile, "%s/data/electronicStructure/pseudoPotential/z%u/singleAtomData/density.inp", DFT_PATH, *it);
	  else
	     sprintf(densityFile, "%s/data/electronicStructure/pseudoPotential/z%u/oncv/singleAtomData/density.inp", DFT_PATH, *it);

	}
      else
	{
	  sprintf(densityFile, "%s/data/electronicStructure/allElectron/z%u/singleAtomData/density.inp", DFT_PATH, *it);
	}
   
      dftUtils::readFile(2, singleAtomElectronDensity[*it], densityFile);
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
  QGauss<3>  quadrature_formula(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (FE, quadrature_formula, update_values | update_quadrature_points);
  const unsigned int n_q_points    = quadrature_formula.size();

  //cleanup of existing data
  rhoInVals.clear();
  rhoOutVals.clear();
  gradRhoInVals.clear();
  gradRhoOutVals.clear(); 
  rhoInValsSpinPolarized.clear();
  rhoOutValsSpinPolarized.clear();
  gradRhoInValsSpinPolarized.clear();
  gradRhoOutValsSpinPolarized.clear(); 

  //Initialize electron density table storage

  rhoInVals.push_back(std::map<dealii::CellId, std::vector<double> >());
  rhoInValues=&(rhoInVals.back());
  if (dftParameters::spinPolarized==1)
    {
      rhoInValsSpinPolarized.push_back(std::map<dealii::CellId, std::vector<double> >());
      rhoInValuesSpinPolarized=&(rhoInValsSpinPolarized.back());
    }
  //
  //get number of image charges used only for periodic
  //
  const int numberImageCharges = d_imageIds.size();

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
		  Point<3> imageAtom(d_imagePositions[iImageCharge][0],d_imagePositions[iImageCharge][1],d_imagePositions[iImageCharge][2]);
		  double distanceToAtom = quadPoint.distance(imageAtom);
		  int masterAtomId = d_imageIds[iImageCharge];
		  if(distanceToAtom <= outerMostPointDen[atomLocations[masterAtomId][0]])//outerMostPointPseudo[atomLocations[masterAtomId][0]])
		    {
		      rhoValueAtQuadPt += alglib::spline1dcalc(denSpline[atomLocations[masterAtomId][0]], distanceToAtom);
		    }
		  else
		    {
		      rhoValueAtQuadPt += 0.0;
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
      gradRhoInVals.push_back(std::map<dealii::CellId, std::vector<double> >());
      gradRhoInValues= &(gradRhoInVals.back());
      //
	if(dftParameters::spinPolarized==1)
        {
          gradRhoInValsSpinPolarized.push_back(std::map<dealii::CellId, std::vector<double> >());
          gradRhoInValuesSpinPolarized=&(gradRhoInValsSpinPolarized.back());
        } 
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
		      else
			{
			  gradRhoXValueAtQuadPt += 0.0;
			  gradRhoYValueAtQuadPt += 0.0;
			  gradRhoZValueAtQuadPt += 0.0;
			}
		    }

		  for(int iImageCharge = 0; iImageCharge < numberImageCharges; ++iImageCharge)
		    {
		      Point<3> imageAtom(d_imagePositions[iImageCharge][0],d_imagePositions[iImageCharge][1],d_imagePositions[iImageCharge][2]);
		      double distanceToAtom = quadPoint.distance(imageAtom);
		      int masterAtomId = d_imageIds[iImageCharge];
		      if(distanceToAtom <= outerMostPointDen[atomLocations[masterAtomId][0]])//outerMostPointPseudo[atomLocations[masterAtomId][0]])
			{
			  double value,radialDensityFirstDerivative,radialDensitySecondDerivative;
			  alglib::spline1ddiff(denSpline[atomLocations[masterAtomId][0]],
					       distanceToAtom,
					       value,
					       radialDensityFirstDerivative,
					       radialDensitySecondDerivative);
										      
			  gradRhoXValueAtQuadPt += radialDensityFirstDerivative*((quadPoint[0] - d_imagePositions[iImageCharge][0])/distanceToAtom);
			  gradRhoYValueAtQuadPt += radialDensityFirstDerivative*((quadPoint[1] - d_imagePositions[iImageCharge][1])/distanceToAtom);
			  gradRhoZValueAtQuadPt += radialDensityFirstDerivative*((quadPoint[2] - d_imagePositions[iImageCharge][2])/distanceToAtom);

			}
		      else
			{
			  gradRhoXValueAtQuadPt += 0.0;
			  gradRhoYValueAtQuadPt += 0.0;
			  gradRhoZValueAtQuadPt += 0.0;
			}
		  
		    }

		  int signRho = (*rhoInValues)[cell->id()][q]/std::abs((*rhoInValues)[cell->id()][q]);
		  gradRhoInValuesPtr[3*q+0] = signRho*gradRhoXValueAtQuadPt;
		  gradRhoInValuesPtr[3*q+1] = signRho*gradRhoYValueAtQuadPt;
		  gradRhoInValuesPtr[3*q+2] = signRho*gradRhoZValueAtQuadPt;
		  if(dftParameters::spinPolarized==1)
	           { 
	             gradRhoInValuesSpinPolarizedPtr[6*q+0] =( 0.5 + dftParameters::start_magnetization)*signRho*gradRhoXValueAtQuadPt;
	             gradRhoInValuesSpinPolarizedPtr[6*q+1] = ( 0.5 + dftParameters::start_magnetization)*signRho*gradRhoYValueAtQuadPt ;
		     gradRhoInValuesSpinPolarizedPtr[6*q+2] =  ( 0.5 + dftParameters::start_magnetization)*signRho*gradRhoZValueAtQuadPt ;
		     gradRhoInValuesSpinPolarizedPtr[6*q+3] =( 0.5 - dftParameters::start_magnetization)*signRho*gradRhoXValueAtQuadPt;
	             gradRhoInValuesSpinPolarizedPtr[6*q+4] = ( 0.5 - dftParameters::start_magnetization)*signRho*gradRhoYValueAtQuadPt ;
		     gradRhoInValuesSpinPolarizedPtr[6*q+5] =  ( 0.5 - dftParameters::start_magnetization)*signRho*gradRhoZValueAtQuadPt ;
		   }
		}
	    }
	} 
    }

  normalizeRho();
  //
  computing_timer.exit_section("initialize density"); 
}

//
//Normalize rho
//
template<unsigned int FEOrder>
void dftClass<FEOrder>::normalizeRho()
{
  QGauss<3>  quadrature_formula(C_num1DQuad<FEOrder>());
  const unsigned int n_q_points    = quadrature_formula.size();    

  double charge = totalCharge(rhoInValues);
  char buffer[100];
  sprintf(buffer, "initial total charge: %18.10e \n", charge);
  pcout << buffer;
  //scaling rho
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      for (unsigned int q=0; q<n_q_points; ++q){
	(*rhoInValues)[cell->id()][q]*=((double)numElectrons)/charge;
	if (dftParameters::spinPolarized==1)
	   {
           	(*rhoInValuesSpinPolarized)[cell->id()][2*q+1]*=((double)numElectrons)/charge;
           	(*rhoInValuesSpinPolarized)[cell->id()][2*q]*=((double)numElectrons)/charge;
	   }
      }
    }
  }
  double chargeAfterScaling = totalCharge(rhoInValues);
  sprintf(buffer, "initial total charge after scaling: %18.10e \n", chargeAfterScaling);
  pcout << buffer;
}
