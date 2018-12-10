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
// @author  Phani Motamarri
//
#include <complex>
#include <vector>

namespace internal
{
    void pointWiseScaleWithDiagonal(const vectorType & diagonal,
				    std::vector<vectorType> & fieldArray,
				    dftUtils::constraintMatrixInfo & constraintsNoneEigenDataInfo)
    {
       for(unsigned int i = 0; i < fieldArray.size();++i)
	{
	  auto & vec = fieldArray[i];
	  vec.scale(diagonal);
	  constraintsNoneEigenDataInfo.distribute(vec);
	  vec.update_ghost_values();
	}

    }


    void pointWiseScaleWithDiagonal(const vectorType & diagonal,
				    const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> & singleComponentPartitioner,
				    const unsigned int numberFields,
				    const std::vector<dealii::types::global_dof_index> & localProc_dof_indicesReal,
				    std::vector<dataTypes::number> & fieldsArrayFlattened)
    {

        const unsigned int numberDofs = fieldsArrayFlattened.size()/numberFields;
        const unsigned int inc = 1;

        for(unsigned int i = 0; i < numberDofs; ++i)
        {
#ifdef USE_COMPLEX
	    double scalingCoeff =
		diagonal.local_element(localProc_dof_indicesReal[i]);
	    zdscal_(&numberFields,
	           &scalingCoeff,
	           &fieldsArrayFlattened[i*numberFields],
	           &inc);
#else
	    double scalingCoeff = diagonal.local_element(i);
	    dscal_(&numberFields,
		   &scalingCoeff,
		   &fieldsArrayFlattened[i*numberFields],
		   &inc);
#endif
	}

    }
}

//
template<unsigned int FEOrder>
dataTypes::number dftClass<FEOrder>::computeTraceXtHX(unsigned int numberWaveFunctionsEstimate)
{
  //
  //set up poisson solver
  //
  dealiiLinearSolver dealiiCGSolver(mpi_communicator, dealiiLinearSolver::CG);
  poissonSolverProblem<FEOrder> phiTotalSolverProblem(mpi_communicator);

  //
  //solve for vself and compute Tr(XtHX)
  //
  d_vselfBinsManager.solveVselfInBins(matrix_free_data,
				      2,
				      d_phiExt,
				      d_noConstraints,
				      d_imagePositions,
				      d_imageIds,
				      d_imageCharges,
				      d_localVselfs);

  //
  //solve for potential corresponding to initial electron-density
  //
  phiTotalSolverProblem.reinit(matrix_free_data,
			       d_phiTotRhoIn,
			       *d_constraintsVector[phiTotDofHandlerIndex],
			       phiTotDofHandlerIndex,
			       d_atomNodeIdToChargeMap,
			       *rhoInValues);


  dealiiCGSolver.solve(phiTotalSolverProblem,
		       dftParameters::relLinearSolverTolerance,
		       dftParameters::maxLinearSolverIterations,
		       dftParameters::verbosity);

  //
  //create kohnShamDFTOperatorClass object
  //
  kohnShamDFTOperatorClass<FEOrder> kohnShamDFTEigenOperator(this,mpi_communicator);
  kohnShamDFTEigenOperator.init();

  //
  //precompute shapeFunctions and shapeFunctionGradients and shapeFunctionGradientIntegrals
  //
  kohnShamDFTEigenOperator.preComputeShapeFunctionGradientIntegrals();

  //
  //compute Veff
  //
  if(dftParameters::xc_id < 4)
    {
      kohnShamDFTEigenOperator.computeVEff(rhoInValues, d_phiTotRhoIn, d_phiExt, d_pseudoVLoc);
    }
  else if (dftParameters::xc_id == 4)
    {
      kohnShamDFTEigenOperator.computeVEff(rhoInValues, gradRhoInValues, d_phiTotRhoIn, d_phiExt, d_pseudoVLoc);
    }

  //
  //compute Hamiltonian matrix
  //
  kohnShamDFTEigenOperator.computeHamiltonianMatrix(0);

  //
  //scale the eigenVectors (initial guess of single atom wavefunctions or previous guess) to convert into Lowden Orthonormalized FE basis
  //multiply by M^{1/2}
  internal::pointWiseScaleWithDiagonal(kohnShamDFTEigenOperator.d_sqrtMassVector,
				       matrix_free_data.get_vector_partitioner(),
				       numEigenValues,
				       localProc_dof_indicesReal,
				       d_eigenVectorsFlattenedSTL[0],
				       constraintsNoneDataInfo);

  //
  //compute projected Hamiltonian
  //
  std::vector<dataTypes::number> ProjHam;
  
  kohnShamDFTEigenOperator.XtHX(d_eigenVectorsFlattenedSTL[0],
				numEigenValues,
				ProjHam);

  //
  //scale the eigenVectors with M^{-1/2} to represent the wavefunctions in the usual FE basis
  //
  internal::pointWiseScaleWithDiagonal(kohnShamDFTEigenOperator.d_invSqrtMassVector,
				       matrix_free_data.get_vector_partitioner(),
				       numEigenValues,
				       localProc_dof_indicesReal,
				       d_eigenVectorsFlattenedSTL[0],
				       constraintsNoneDataInfo);

  dataTypes::number trXtHX = 0.0;
  for(unsigned int i = 0; i < numberWaveFunctionsEstimate; ++i)
    {
      trXtHX += ProjHam[numEigenValues*i+i];
    }

  return trXtHX;

}



//chebyshev solver
template<unsigned int FEOrder>
void dftClass<FEOrder>::kohnShamEigenSpaceCompute(const unsigned int spinType,
						  const unsigned int kPointIndex,
						  kohnShamDFTOperatorClass<FEOrder> & kohnShamDFTEigenOperator,
						  chebyshevOrthogonalizedSubspaceIterationSolver & subspaceIterationSolver,
						  std::vector<double>                            & residualNormWaveFunctions,
						  const bool isSpectrumSplit,
						  const bool useMixedPrec)
{
  computing_timer.enter_section("Chebyshev solve");

  if (dftParameters::verbosity>=2)
    {
      pcout << "kPoint: "<< kPointIndex<<std::endl;
      if (dftParameters::spinPolarized==1)
        pcout << "spin: "<< spinType+1 <<std::endl;
    }


  //
  //scale the eigenVectors (initial guess of single atom wavefunctions or previous guess) to convert into Lowden Orthonormalized FE basis
  //multiply by M^{1/2}
  internal::pointWiseScaleWithDiagonal(kohnShamDFTEigenOperator.d_sqrtMassVector,
				       matrix_free_data.get_vector_partitioner(),
				       d_numEigenValues,
				       localProc_dof_indicesReal,
				       d_eigenVectorsFlattenedSTL[(1+dftParameters::spinPolarized)*kPointIndex+spinType]);

  std::vector<double> eigenValuesTemp(isSpectrumSplit?d_numEigenValuesRR
	                              :d_numEigenValues,0.0);

  subspaceIterationSolver.reinitSpectrumBounds(a0[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
					       bLow[(1+dftParameters::spinPolarized)*kPointIndex+spinType]);

  subspaceIterationSolver.solve(kohnShamDFTEigenOperator,
  				d_eigenVectorsFlattenedSTL[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
				d_eigenVectorsRotFracDensityFlattenedSTL[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
				d_tempEigenVec,
				d_numEigenValues,
  				eigenValuesTemp,
				residualNormWaveFunctions,
				interBandGroupComm,
				useMixedPrec);

  //
  //scale the eigenVectors with M^{-1/2} to represent the wavefunctions in the usual FE basis
  //
  internal::pointWiseScaleWithDiagonal(kohnShamDFTEigenOperator.d_invSqrtMassVector,
				       matrix_free_data.get_vector_partitioner(),
				       d_numEigenValues,
				       localProc_dof_indicesReal,
				       d_eigenVectorsFlattenedSTL[(1+dftParameters::spinPolarized)*kPointIndex+spinType]);

  if (isSpectrumSplit && d_numEigenValuesRR!=d_numEigenValues)
  {
       internal::pointWiseScaleWithDiagonal(kohnShamDFTEigenOperator.d_invSqrtMassVector,
				            matrix_free_data.get_vector_partitioner(),
				            d_numEigenValuesRR,
				            localProc_dof_indicesReal,
				            d_eigenVectorsRotFracDensityFlattenedSTL[(1+dftParameters::spinPolarized)*kPointIndex+spinType]);
  }

  //
  //copy the eigenValues and corresponding residual norms back to data members
  //
  if (isSpectrumSplit)
    {
      for(unsigned int i = 0; i < d_numEigenValuesRR; i++)
	{
	  if(dftParameters::verbosity>=4 && d_numEigenValues==d_numEigenValuesRR)
	      pcout<<"eigen value "<< std::setw(3) <<i <<": "<<eigenValuesTemp[i] <<std::endl;
	  else if(dftParameters::verbosity>=4 && d_numEigenValues!=d_numEigenValuesRR)
              pcout<<"valence eigen value "<< std::setw(3) <<i <<": "<<eigenValuesTemp[i] <<std::endl;

	  eigenValuesRRSplit[kPointIndex][spinType*d_numEigenValuesRR + i] =  eigenValuesTemp[i];
	}

      for(unsigned int i = 0; i < d_numEigenValues; i++)
	{
	  if (i>=(d_numEigenValues-d_numEigenValuesRR))
	     eigenValues[kPointIndex][spinType*d_numEigenValues + i]
		 = eigenValuesTemp[i-(d_numEigenValues-d_numEigenValuesRR)];
	  else
             eigenValues[kPointIndex][spinType*d_numEigenValues + i]=-100.0;
	}
    }
  else
    {
      for(unsigned int i = 0; i < d_numEigenValues; i++)
	{
	  if(dftParameters::verbosity>=4)
	      pcout<<"eigen value "<< std::setw(3) <<i <<": "<<eigenValuesTemp[i] <<std::endl;

	  eigenValues[kPointIndex][spinType*d_numEigenValues + i] =  eigenValuesTemp[i];
	}
    }

  if (dftParameters::verbosity>=4)
     pcout <<std::endl;


  //set a0 and bLow
  /* a0[(1+dftParameters::spinPolarized)*kPointIndex+spinType]=isSpectrumSplit?
                                                            dftParameters::lowerEndWantedSpectrum
                                                            :eigenValuesTemp[0];*/

 
  bLow[(1+dftParameters::spinPolarized)*kPointIndex+spinType]=eigenValuesTemp.back();

  if(!isSpectrumSplit)
    {
      a0[(1+dftParameters::spinPolarized)*kPointIndex+spinType] = eigenValuesTemp[0];
    }

  computing_timer.exit_section("Chebyshev solve");
}



template<unsigned int FEOrder>
void dftClass<FEOrder>::computeResidualNorm(const std::vector<double> & eigenValuesTemp,
					    kohnShamDFTOperatorClass<FEOrder> & kohnShamDFTEigenOperator,
					    std::vector<vectorType> & X,
					    std::vector<double> & residualNorm) const
{


  std::vector<vectorType> PSI(X.size());

  for(unsigned int i = 0; i < X.size(); ++i)
      PSI[i].reinit(X[0]);


  kohnShamDFTEigenOperator.HX(X, PSI);

  if (dftParameters::verbosity>=4)
     pcout<<"L-2 Norm of residue   :"<<std::endl;

  for(unsigned int i = 0; i < eigenValuesTemp.size(); i++)
    {
      (PSI[i]).add(-eigenValuesTemp[i],X[i]) ;
      const double resNorm= (PSI[i]).l2_norm();
      residualNorm[i]=resNorm;

      if (dftParameters::verbosity>=4)
	pcout<<"eigen vector "<< i<<": "<<resNorm<<std::endl;
    }
  if (dftParameters::verbosity>=3)
    pcout <<std::endl;


}

//compute the maximum of the residual norm of the highest occupied state among all k points
template<unsigned int FEOrder>
double dftClass<FEOrder>::computeMaximumHighestOccupiedStateResidualNorm(const std::vector<std::vector<double> > & residualNormWaveFunctionsAllkPoints,
									 const std::vector<std::vector<double> > & eigenValuesAllkPoints,
									 const double fermiEnergy)
{
  double maxHighestOccupiedStateResNorm=-1e+6;
  for (int kPoint = 0; kPoint < eigenValuesAllkPoints.size(); ++kPoint)
   {
     unsigned int highestOccupiedState = 0;

     for(unsigned int i = 0; i < eigenValuesAllkPoints[kPoint].size(); i++)
       {
         const double factor=(eigenValuesAllkPoints[kPoint][i]-fermiEnergy)/(C_kb*dftParameters::TVal);
	 if (factor<0)
	   highestOccupiedState=i;
       }

     if(residualNormWaveFunctionsAllkPoints[kPoint][highestOccupiedState]>maxHighestOccupiedStateResNorm)
       {
         maxHighestOccupiedStateResNorm=residualNormWaveFunctionsAllkPoints[kPoint][highestOccupiedState];
       }

   }
  maxHighestOccupiedStateResNorm= Utilities::MPI::max(maxHighestOccupiedStateResNorm, interpoolcomm);
  return maxHighestOccupiedStateResNorm;
}
