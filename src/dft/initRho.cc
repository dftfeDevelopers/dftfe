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
//Initlialize rho by reading in single-atom electron-density and fit a spline
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

  //loop over atom types
  for (std::set<unsigned int>::iterator it=atomTypes.begin(); it!=atomTypes.end(); it++)
    {
      char densityFile[256];
      if(dftParameters::isPseudopotential)
	{
	  //if(dftParameters::pseudoProjector==1)
	  // sprintf(densityFile, "%s/data/electronicStructure/pseudoPotential/z%u/singleAtomData/density.inp", DFT_PATH, *it);
	  //else
	     sprintf(densityFile,"temp/z%u/density.inp",*it);
	    //sprintf(densityFile, "%s/data/electronicStructure/pseudoPotential/z%u/oncv/singleAtomData/density.inp", DFT_PATH, *it);

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
      alglib::ae_int_t natural_bound_type_L = 1;
      alglib::ae_int_t natural_bound_type_R = 1;
      spline1dbuildcubic(x, y, numRows, natural_bound_type_L, 0.0, natural_bound_type_R, 0.0, denSpline[*it]);
      outerMostPointDen[*it]= xData[numRows-1];
    }


  //Initialize rho
  QGauss<3>  quadrature_formula(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (FE, quadrature_formula, update_quadrature_points);
  const unsigned int n_q_points    = quadrature_formula.size();

  //Initialize electron density table storage

  rhoInVals.push_back(std::map<dealii::CellId, std::vector<double> >());
  rhoInValues=&(rhoInVals.back());
  if(dftParameters::spinPolarized==1)
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

		  int signRho = 0 ;
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
  //
  computing_timer.exit_section("initialize density");
}

//
//
//
template <unsigned int FEOrder>
void dftClass<FEOrder>::computeRhoInitialGuessFromPSI(std::vector<std::vector<vectorType>> eigenVectors)

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
		  for(unsigned int i=0; i<numEigenValues; ++i)
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
			  factor=(eigenValues[kPoint][i+dftParameters::spinPolarized*numEigenValues]-fermiEnergy)/(C_kb*dftParameters::TVal);
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
		  for(unsigned int i=0; i<numEigenValues; ++i)
		    {
		      fe_values.get_function_values(eigenVectors[(1+dftParameters::spinPolarized)*kPoint][i], tempPsi);
		      if(dftParameters::spinPolarized==1)
		         fe_values.get_function_values(eigenVectors[(1+dftParameters::spinPolarized)*kPoint+1][i], tempPsi2);

		      for(unsigned int q_point=0; q_point<num_quad_points; ++q_point)
			{
			  double factor=(eigenValues[kPoint][i]-fermiEnergy)/(C_kb*dftParameters::TVal);
			  double partialOccupancy = (factor >= 0)?std::exp(-factor)/(1.0 + std::exp(-factor)):1.0/(1.0 + std::exp(factor));
			  //
			  factor=(eigenValues[kPoint][i+dftParameters::spinPolarized*numEigenValues]-fermiEnergy)/(C_kb*dftParameters::TVal);
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
  dealii::VectorTools::project<3,dealii::parallel::distributed::Vector<double>>
      (dealii::MappingQ1<3,3>(),
       dofHandler,
       constraintsNone,
       QGauss<3>(C_num1DQuad<FEOrder>()),
       [&](const typename dealii::DoFHandler<3>::active_cell_iterator & cell , const unsigned int q) -> double {return (*rhoOutValues).find(cell->id())->second[q];},
       d_rhoNodalField);
  d_rhoNodalField.update_ghost_values();


  if (dftParameters::spinPolarized==1)
  {
      matrix_free_data.initialize_dof_vector(d_rhoNodalFieldSpin0,densityDofHandlerIndex);
      d_rhoNodalFieldSpin0=0;
      dealii::VectorTools::project<3,dealii::parallel::distributed::Vector<double>>
	  (dealii::MappingQ1<3,3>(),
	   dofHandler,
	   constraintsNone,
	   QGauss<3>(C_num1DQuad<FEOrder>()),
	   [&](const typename dealii::DoFHandler<3>::active_cell_iterator & cell , const unsigned int q) -> double {return (*rhoOutValuesSpinPolarized).find(cell->id())->second[2*q];},
	   d_rhoNodalFieldSpin0);
      d_rhoNodalFieldSpin0.update_ghost_values();

      matrix_free_data.initialize_dof_vector(d_rhoNodalFieldSpin1,densityDofHandlerIndex);
      d_rhoNodalFieldSpin1=0;
      dealii::VectorTools::project<3,dealii::parallel::distributed::Vector<double>>
	  (dealii::MappingQ1<3,3>(),
	   dofHandler,
	   constraintsNone,
	   QGauss<3>(C_num1DQuad<FEOrder>()),
	   [&](const typename dealii::DoFHandler<3>::active_cell_iterator & cell , const unsigned int q) -> double {return (*rhoOutValuesSpinPolarized).find(cell->id())->second[2*q+1];},
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

  std::vector<vectorType* > rhoFieldsPrevious;
  rhoFieldsPrevious.push_back(&d_rhoNodalField);
  if (dftParameters::spinPolarized==1)
  {
    rhoFieldsPrevious.push_back(&d_rhoNodalFieldSpin0);
    rhoFieldsPrevious.push_back(&d_rhoNodalFieldSpin1);
  }

  vectorType rhoNodalFieldCurrent;
  vectorType rhoNodalFieldSpin0Current;
  vectorType rhoNodalFieldSpin1Current;
  matrix_free_data.initialize_dof_vector(rhoNodalFieldCurrent,densityDofHandlerIndex);
  if (dftParameters::spinPolarized==1)
  {
    matrix_free_data.initialize_dof_vector(rhoNodalFieldSpin0Current,densityDofHandlerIndex);
    matrix_free_data.initialize_dof_vector(rhoNodalFieldSpin1Current,densityDofHandlerIndex);
  }
  std::vector<vectorType* > rhoFieldsCurrent;
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

  //gather density from all pools
  sumRhoData(rhoInValues,
	     gradRhoInValues,
	     rhoInValuesSpinPolarized,
	     gradRhoInValuesSpinPolarized,
	     interpoolcomm);

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

  const double charge = totalCharge(rhoInValues);
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
  double chargeAfterScaling = totalCharge(rhoInValues);

  if (dftParameters::verbosity>=1)
     pcout<<"Initial total charge: "<< chargeAfterScaling<<std::endl;
}
