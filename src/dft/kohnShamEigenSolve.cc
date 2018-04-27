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
// @author  Phani Motamarri (2018)
//
#include <complex>
#include <vector>


void pointWiseScaleWithDiagonal(const vectorType & diagonal,
				std::vector<vectorType> & fieldArray,
				dftUtils::constraintMatrixInfo constraintsNoneEigenDataInfo)
{
   for(unsigned int i = 0; i < fieldArray.size();++i)
    {
      auto & vec = fieldArray[i];
      vec.scale(diagonal);
      constraintsNoneEigenDataInfo.distribute(vec);     
      vec.update_ghost_values();
    }

}


//chebyshev solver
template<unsigned int FEOrder>
void dftClass<FEOrder>::kohnShamEigenSpaceCompute(const unsigned int spinType,
						  const unsigned int kPointIndex,
						  chebyshevOrthogonalizedSubspaceIterationSolver & subspaceIterationSolver) const
{
  computing_timer.enter_section("Chebyshev solve"); 
  
  if (dftParameters::verbosity==2)
    {
      pcout << "kPoint: "<< kPointIndex<<std::endl;
      pcout << "spin: "<< spinType+1 <<std::endl;
    }


  //
  //scale the eigenVectors (initial guess of single atom wavefunctions or previous guess) to convert into Lowden Orthonormalized FE basis
  //multiply by M^{1/2}
  pointWiseScaleWithDiagonal(eigenPtr->sqrtMassVector,
			     eigenVectors[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
			     constraintsNoneEigenDataInfo);


  

  std::vector<double> eigenValuesTemp(numEigenValues,0.0);

  subspaceIterationSolver.reinitSpectrumBounds(a0[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
					       bLow[(1+dftParameters::spinPolarized)*kPointIndex+spinType]);

  subspaceIterationSolver.solve(eigenPtr,
  				eigenVectors[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
  				eigenValuesTemp);
 

  //
  //Compute and print L2 norm
  //
  std::vector<double> residualNorm(numEigenValues,0.0);
  computeResidualNorm(eigenValuesTemp,
		      eigenVectors[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
		      residualNorm);
  
  
  //
  //scale the eigenVectors with M^{-1/2} to represent the wavefunctions in the usual FE basis
  //
  pointWiseScaleWithDiagonal(eigenPtr->invSqrtMassVector,
			     eigenVectors[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
			     constraintsNoneEigenDataInfo);

  
  //
  //copy the eigenValues and corresponding residual norms back to data members
  //
  for(unsigned int i = 0; i < (unsigned int)numEigenValues; i++)
    {
      if(dftParameters::verbosity==2)
          pcout<<"eigen value "<< std::setw(3) <<i <<": "<<eigenValuesTemp[i] <<std::endl;

      eigenValues[kPointIndex][spinType*numEigenValues + i] =  eigenValuesTemp[i];
      d_tempResidualNormWaveFunctions[kPointIndex][i] = residualNorm[i];
    }

  if (dftParameters::verbosity==2)  
     pcout <<std::endl;


  //set a0 and bLow
  a0[(1+dftParameters::spinPolarized)*kPointIndex+spinType]=eigenValuesTemp[0]; 
  bLow[(1+dftParameters::spinPolarized)*kPointIndex+spinType]=eigenValuesTemp.back(); 
  //

 
  computing_timer.exit_section("Chebyshev solve"); 
}



template<unsigned int FEOrder>
void dftClass<FEOrder>::computeResidualNorm(const std::vector<double> & eigenValuesTemp,
					    std::vector<vectorType> & X,
					    std::vector<double> & residualNorm) const
{
  computing_timer.enter_section("computeResidualNorm"); 

  std::vector<vectorType> PSI(X.size());
  
  for(unsigned int i = 0; i < X.size(); ++i)
      PSI[i].reinit(X[0]);


  eigenPtr->HX(X, PSI);

  if (dftParameters::verbosity==2)
     pcout<<"L-2 Norm of residue   :"<<std::endl;

  for(unsigned int i = 0; i < eigenValuesTemp.size(); i++)
    {
      (PSI[i]).add(-eigenValuesTemp[i],X[i]) ;
      const double resNorm= (PSI[i]).l2_norm();
      if (dftParameters::spinPolarized!=1)
	residualNorm[i]=resNorm;
      
      if (dftParameters::verbosity==2)
	pcout<<"eigen vector "<< i<<": "<<resNorm<<std::endl;
    }
  if (dftParameters::verbosity==2)  
    pcout <<std::endl;

  computing_timer.exit_section("computeResidualNorm"); 
}

//compute the maximum of the residual norm of the highest occupied state among all k points 
template<unsigned int FEOrder>
double dftClass<FEOrder>::computeMaximumHighestOccupiedStateResidualNorm()
{
  double maxHighestOccupiedStateResNorm=-1e+6;
  for (int kPoint = 0; kPoint < d_maxkPoints; ++kPoint) 
   {    

      unsigned int highestOccupiedState=0;
      unsigned int n = eigenValues[kPoint].size() ;
      for(unsigned int i = 0; i < n; i++)
      {
         double factor=(eigenValues[kPoint][i]-fermiEnergy)/(C_kb*dftParameters::TVal);
	 if (factor<0)
	 {
	     highestOccupiedState=i;
	 }	 
      }
      if (d_tempResidualNormWaveFunctions[kPoint][highestOccupiedState]>maxHighestOccupiedStateResNorm)
      {
         maxHighestOccupiedStateResNorm=d_tempResidualNormWaveFunctions[kPoint][highestOccupiedState];
      }
   }
  maxHighestOccupiedStateResNorm= Utilities::MPI::max(maxHighestOccupiedStateResNorm, interpoolcomm);
  return maxHighestOccupiedStateResNorm;
} 
