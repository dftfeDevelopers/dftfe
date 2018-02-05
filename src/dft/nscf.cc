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
// @author Krishnendu Ghosh(2018)
//

#include "../../include/dftParameters.h"

using namespace dftParameters ;


template<unsigned int FEOrder>
void dftClass<FEOrder>::nscf()
{
  // clear previous scf allocations
  eigenValues.clear();
  eigenValuesTemp.clear();
  a0.clear();
  bLow.clear();
  eigenVectors.clear();

  //set size of eigenvalues and eigenvectors data structures
  eigenValues.resize(d_maxkPoints);
  eigenValuesTemp.resize(d_maxkPoints);
  a0.resize((spinPolarized+1)*d_maxkPoints,lowerEndWantedSpectrum);
  bLow.resize((spinPolarized+1)*d_maxkPoints,0.0);
  eigenVectors.resize((1+spinPolarized)*d_maxkPoints);
  //eigenVectorsOrig.resize((1+spinPolarized)*d_maxkPoints);
  //
  //
  for(unsigned int kPoint = 0; kPoint < (1+spinPolarized)*d_maxkPoints; ++kPoint)
    {
      //for (unsigned int j=0; j<(spinPolarized+1); ++j) // for spin
       //{
        for (unsigned int i=0; i<numEigenValues; ++i)
	  {
	    eigenVectors[kPoint].push_back(new vectorType);
	    //eigenVectorsOrig[kPoint].push_back(new vectorType);
	  }
       //}
    }
   for(unsigned int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
    {
      eigenValues[kPoint].resize((spinPolarized+1)*numEigenValues);  
      eigenValuesTemp[kPoint].resize(numEigenValues); 
    }

  if(isPseudopotential)
      computeElementalProjectorKets();


  for(unsigned int kPoint = 0; kPoint < (1+spinPolarized)*d_maxkPoints; ++kPoint)
    {
        for (unsigned int i=0; i<eigenVectors[kPoint].size(); ++i)
	  {
	    eigenVectors[kPoint][i]->reinit(vChebyshev);
	    *eigenVectors[kPoint][i] = *eigenVectorsOrig[0][i];
	  }
    }

   double norm ;
   char buffer[100] ;
   if (spinPolarized==1)
       norm = sqrt(mixing_anderson_spinPolarized());
   else
       norm = sqrt(mixing_anderson());	
   	
   sprintf(buffer, "Anderson Mixing: L2 norm of electron-density difference: %12.6e\n\n", norm); pcout << buffer;
   poissonPtr->phiTotRhoIn = poissonPtr->phiTotRhoOut;

      //phiTot with rhoIn

      //parallel loop over all elements

      int constraintMatrixId = 1;
      sprintf(buffer, "Poisson solve for total electrostatic potential (rhoIn+b):\n"); pcout << buffer; 
      poissonPtr->solve(poissonPtr->phiTotRhoIn,constraintMatrixId, rhoInValues);
  //
     
     //numPass *= 4 ;
     
      //eigen solve
      if (spinPolarized==1)
	{
	  for(unsigned int s=0; s<2; ++s)
	      {
	       if(xc_id < 4) 
	        {
		  if(isPseudopotential)
		    eigenPtr->computeVEffSpinPolarized(rhoInValuesSpinPolarized, poissonPtr->phiTotRhoIn, poissonPtr->phiExt, s, pseudoValues);
		  else
		    eigenPtr->computeVEffSpinPolarized(rhoInValuesSpinPolarized, poissonPtr->phiTotRhoIn, poissonPtr->phiExt, s);
                }
	       else if (xc_id == 4)
	        {
	          if(isPseudopotential)
		    eigenPtr->computeVEffSpinPolarized(rhoInValuesSpinPolarized, gradRhoInValuesSpinPolarized, poissonPtr->phiTotRhoIn, poissonPtr->phiExt, s, pseudoValues);
	          else
		    eigenPtr->computeVEffSpinPolarized(rhoInValuesSpinPolarized, gradRhoInValuesSpinPolarized, poissonPtr->phiTotRhoIn, poissonPtr->phiExt, s);
	        }
	      for (int kPoint = 0; kPoint < d_maxkPoints; ++kPoint) 
	        {
	          d_kPointIndex = kPoint;
	          char buffer[100];
	          for(int j = 0; j < 3*numPass; ++j)
	            { 
		       sprintf(buffer, "%s:%3u%s:%3u\n", "Beginning Chebyshev filter pass ", j+1, " for spin ", s+1);
		       pcout << buffer;
		       chebyshevSolver(s);
	            }
	        }
	    }
        }
      else
        {
	  if(xc_id < 4)
	      {
	      if(isPseudopotential)
		eigenPtr->computeVEff(rhoInValues, poissonPtr->phiTotRhoIn, poissonPtr->phiExt, pseudoValues);
	      else
		eigenPtr->computeVEff(rhoInValues, poissonPtr->phiTotRhoIn, poissonPtr->phiExt); 
	      }
	  else if (xc_id == 4)
	     {
	      if(isPseudopotential)
		eigenPtr->computeVEff(rhoInValues, gradRhoInValues, poissonPtr->phiTotRhoIn, poissonPtr->phiExt, pseudoValues);
	      else
		eigenPtr->computeVEff(rhoInValues, gradRhoInValues, poissonPtr->phiTotRhoIn, poissonPtr->phiExt);
	     } 
	    for (int kPoint = 0; kPoint < d_maxkPoints; ++kPoint) 
	      {
	        d_kPointIndex = kPoint;
	        for(int j = 0; j < 3*numPass; ++j)
	        { 
		    sprintf(buffer, "%s:%3u\n", "Beginning Chebyshev filter pass ", j+1);
		    pcout << buffer;
		    chebyshevSolver(0);
	        }
	      }

	}

   pcout << " Printing bandstructure data ... " << std::endl ;
   for (int kPoint = 0; kPoint < d_maxkPoints; ++kPoint) 
	for (unsigned int i = 0; i < numEigenValues; ++i)  {
		pcout << kPoint << " " << i << " " << eigenValues[kPoint][i] << std::endl ;
   }
   pcout << " ****************************************************************************************** " << std::endl ;

}
