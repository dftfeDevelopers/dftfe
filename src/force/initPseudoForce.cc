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
// @author Sambit Das (2017)
//

#include <boost/math/special_functions/spherical_harmonic.hpp>


namespace initPseudoForceUtils
{
double tolerance = 1e-12;
//some inline functions
inline 
void getRadialFunctionVal(const double radialCoordinate,
			  double &splineVal,
			  const alglib::spline1dinterpolant * spline) 
{
  
  splineVal = alglib::spline1dcalc(*spline,
				   radialCoordinate);
  return;
}

inline
void
getSphericalHarmonicVal(const double theta, const double phi, const int l, const int m, double & sphericalHarmonicVal)
{
      
  if(m < 0)
    sphericalHarmonicVal = sqrt(2.0)*boost::math::spherical_harmonic_i(l,-m,theta,phi);
      
  else if (m == 0)
    sphericalHarmonicVal = boost::math::spherical_harmonic_r(l,m,theta,phi);

  else if (m > 0)
    sphericalHarmonicVal = sqrt(2.0)*boost::math::spherical_harmonic_r(l,m,theta,phi);

  return;

}

void
convertCartesianToSpherical(double *x, double & r, double & theta, double & phi)
{

  r = sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
 
  if(r == 0)
    {
	
      theta = 0.0;
      phi = 0.0;

    }
	
  else
    {

      theta = acos(x[2]/r);
      //
      // check if theta = 0 or PI (i.e, whether the point is on the Z-axis)
      // If yes, assign phi = 0.0.
      // NOTE: In case theta = 0 or PI, phi is undetermined. The actual value 
      // of phi doesn't matter in computing the enriched function value or 
      // its gradient. We assign phi = 0.0 here just as a dummy value
      //
      if(fabs(theta - 0.0) >= tolerance && fabs(theta - M_PI) >= tolerance)
	phi = atan2(x[1],x[0]);

      else
	phi = 0.0;

    }

}

}

//
//Initialize rho by reading in single-atom electron-density and fit a spline
//
template<unsigned int FEOrder>
void forceClass<FEOrder>::initLocalPseudoPotentialForce()
{
  d_gradPseudoVLoc.clear();
  d_gradPseudoVLocAtoms.clear();

  //
  //Reading single atom rho initial guess
  //
  std::map<unsigned int, alglib::spline1dinterpolant> pseudoSpline;
  std::map<unsigned int, std::vector<std::vector<double> > > pseudoPotentialData;
  std::map<unsigned int, double> outerMostPointPseudo;

    
  //
  //loop over atom types
  //
  for(std::set<unsigned int>::iterator it=dftPtr->atomTypes.begin(); it!=dftPtr->atomTypes.end(); it++)
    {
      char pseudoFile[256];
      sprintf(pseudoFile, "%s/data/electronicStructure/pseudoPotential/z%u/pseudoAtomData/locPot.dat", dftPtr->d_currentPath.c_str(),*it);
      pcout<<"Reading Local Pseudo-potential data from: " <<pseudoFile<<std::endl;
      dftUtils::readFile(2, pseudoPotentialData[*it], pseudoFile);
      unsigned int numRows = pseudoPotentialData[*it].size()-1;
      std::vector<double> xData(numRows), yData(numRows);
      for(unsigned int irow = 0; irow < numRows; ++irow)
	{
	  xData[irow] = pseudoPotentialData[*it][irow][0];
	  yData[irow] = pseudoPotentialData[*it][irow][1];
	}
  
      //interpolate pseudopotentials
      alglib::real_1d_array x;
      x.setcontent(numRows,&xData[0]);
      alglib::real_1d_array y;
      y.setcontent(numRows,&yData[0]);
      alglib::ae_int_t natural_bound_type = 1;
      spline1dbuildcubic(x, y, numRows, natural_bound_type, 0.0, natural_bound_type, 0.0, pseudoSpline[*it]);
      outerMostPointPseudo[*it]= xData[numRows-1];
    }
  
  //
  //Initialize pseudopotential
  //
  QGauss<3>  quadrature_formula(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values (dftPtr->FE, quadrature_formula, update_values);
  const unsigned int n_q_points = quadrature_formula.size();


  const int numberGlobalCharges=dftPtr->atomLocations.size();
  //
  //get number of image charges used only for periodic
  //
  const int numberImageCharges = dftPtr->d_imageIds.size();
  
  //
  //loop over elements
  //
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  for(; cell!=endc; ++cell) 
    {
      if(cell->is_locally_owned())
	{
	  d_gradPseudoVLoc[cell->id()]=std::vector<double>(n_q_points*3);
	  std::vector<Tensor<1,3,double> > gradPseudoValContribution(n_q_points);	  
	  //loop over atoms
	  for (unsigned int n=0; n<dftPtr->atomLocations.size(); n++)
	  {
              Point<3> atom(dftPtr->atomLocations[n][2],dftPtr->atomLocations[n][3],dftPtr->atomLocations[n][4]);
	      bool isPseudoDataInCell=false;
	      //loop over quad points
	      for (unsigned int q = 0; q < n_q_points; ++q)
	      {	
	          MappingQ1<3,3> test; 
	          Point<3> quadPoint(test.transform_unit_to_real_cell(cell, fe_values.get_quadrature().point(q)));
		  double distanceToAtom = quadPoint.distance(atom);
		  double value,firstDer,secondDer;
		  if(distanceToAtom <= dftPtr->d_pspTail)//outerMostPointPseudo[atomLocations[n][0]])
		    {
		      alglib::spline1ddiff(pseudoSpline[dftPtr->atomLocations[n][0]], distanceToAtom,value,firstDer,secondDer);	
		      isPseudoDataInCell=true;
		    }
		  else
		    {
		      firstDer= (dftPtr->atomLocations[n][1])/distanceToAtom/distanceToAtom;
		    }
		    gradPseudoValContribution[q]=firstDer*(quadPoint-atom)/distanceToAtom;
		    d_gradPseudoVLoc[cell->id()][q*3+0]+=gradPseudoValContribution[q][0];
		    d_gradPseudoVLoc[cell->id()][q*3+1]+=gradPseudoValContribution[q][1];
		    d_gradPseudoVLoc[cell->id()][q*3+2]+=gradPseudoValContribution[q][2];
	      }//loop over quad points
	      if (isPseudoDataInCell){
	          d_gradPseudoVLocAtoms[n][cell->id()]=std::vector<double>(n_q_points*3);
	          for (unsigned int q = 0; q < n_q_points; ++q)
	          {	
		    d_gradPseudoVLocAtoms[n][cell->id()][q*3+0]=gradPseudoValContribution[q][0];
		    d_gradPseudoVLocAtoms[n][cell->id()][q*3+1]=gradPseudoValContribution[q][1];
		    d_gradPseudoVLocAtoms[n][cell->id()][q*3+2]=gradPseudoValContribution[q][2];
	          }
	      }	      
	  }//loop pver atoms

	  //loop over image charges
	  for(int iImageCharge = 0; iImageCharge < numberImageCharges; ++iImageCharge)
	  {
	      Point<3> imageAtom(dftPtr->d_imagePositions[iImageCharge][0],dftPtr->d_imagePositions[iImageCharge][1],dftPtr->d_imagePositions[iImageCharge][2]);
	      bool isPseudoDataInCell=false;	      
	      //loop over quad points
	      for (unsigned int q = 0; q < n_q_points; ++q)
	      {			 
	          MappingQ1<3,3> test; 
	          Point<3> quadPoint(test.transform_unit_to_real_cell(cell, fe_values.get_quadrature().point(q)));		      
		  double distanceToAtom = quadPoint.distance(imageAtom);
		  int masterAtomId = dftPtr->d_imageIds[iImageCharge];
		  double value,firstDer,secondDer;  
		  if(distanceToAtom <= dftPtr->d_pspTail)//outerMostPointPseudo[atomLocations[masterAtomId][0]])
		    {
		      alglib::spline1ddiff(pseudoSpline[dftPtr->atomLocations[masterAtomId][0]], distanceToAtom,value,firstDer,secondDer);
		      isPseudoDataInCell=true;
		    }
		  else
		    {
		      firstDer= (dftPtr->atomLocations[masterAtomId][1])/distanceToAtom/distanceToAtom;		      
		    }
		    gradPseudoValContribution[q]=firstDer*(quadPoint-imageAtom)/distanceToAtom;
		    d_gradPseudoVLoc[cell->id()][q*3+0]+=gradPseudoValContribution[q][0];
		    d_gradPseudoVLoc[cell->id()][q*3+1]+=gradPseudoValContribution[q][1];
		    d_gradPseudoVLoc[cell->id()][q*3+2]+=gradPseudoValContribution[q][2];
	      }//loop over quad points
	      if (isPseudoDataInCell){
	          d_gradPseudoVLocAtoms[numberGlobalCharges+iImageCharge][cell->id()]=std::vector<double>(n_q_points*3);
	          for (unsigned int q = 0; q < n_q_points; ++q)
	          {	
		    d_gradPseudoVLocAtoms[numberGlobalCharges+iImageCharge][cell->id()][q*3+0]=gradPseudoValContribution[q][0];
		    d_gradPseudoVLocAtoms[numberGlobalCharges+iImageCharge][cell->id()][q*3+1]=gradPseudoValContribution[q][1];
		    d_gradPseudoVLocAtoms[numberGlobalCharges+iImageCharge][cell->id()][q*3+2]=gradPseudoValContribution[q][2];
	          }
	      }	     	      
	   }//loop over image charges
	}//cell locally owned check
    }//cell loop
  
}

template<unsigned int FEOrder>
void forceClass<FEOrder>::initNonLocalPseudoPotentialForce()
{
}

template<unsigned int FEOrder>
void forceClass<FEOrder>::computeSparseStructureNonLocalProjectorsForce()
{
}

template<unsigned int FEOrder>
void forceClass<FEOrder>::computeElementalProjectorKetsForce()
{
}
