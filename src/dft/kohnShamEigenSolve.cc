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
				    dealii::parallel::distributed::Vector<dataTypes::number> & fieldsArrayFlattened,
				    dftUtils::constraintMatrixInfo & constraintsNoneDataInfo)
    {

        constraintsNoneDataInfo.precomputeMaps(singleComponentPartitioner,
					       fieldsArrayFlattened.get_partitioner(),
					       numberFields);
        const unsigned int numberDofs = fieldsArrayFlattened.local_size()/numberFields;
        const unsigned int inc = 1;

        for(unsigned int i = 0; i < numberDofs; ++i)
        {
#ifdef USE_COMPLEX
	    std::complex<double> scalingCoeff =
		diagonal.local_element(localProc_dof_indicesReal[i]);
	    zscal_(&numberFields,
	           &scalingCoeff,
	           fieldsArrayFlattened.begin()+i*numberFields,
	           &inc);
#else
	    double scalingCoeff = diagonal.local_element(i);
	    dscal_(&numberFields,
		   &scalingCoeff,
		   fieldsArrayFlattened.begin()+i*numberFields,
		   &inc);
#endif
	}

	constraintsNoneDataInfo.distribute(fieldsArrayFlattened,
					   numberFields);
	fieldsArrayFlattened.update_ghost_values();
    }
}


//chebyshev solver
template<unsigned int FEOrder>
void dftClass<FEOrder>::kohnShamEigenSpaceCompute(const unsigned int spinType,
						  const unsigned int kPointIndex,
						  eigenClass<FEOrder> & kohnShamDFTEigenOperator,
						  chebyshevOrthogonalizedSubspaceIterationSolver & subspaceIterationSolver,
						  std::vector<double>                            & residualNormWaveFunctions)
{
  computing_timer.enter_section("Chebyshev solve");

  if (dftParameters::verbosity>=2)
    {
      pcout << "kPoint: "<< kPointIndex<<std::endl;
      pcout << "spin: "<< spinType+1 <<std::endl;
    }


  //
  //scale the eigenVectors (initial guess of single atom wavefunctions or previous guess) to convert into Lowden Orthonormalized FE basis
  //multiply by M^{1/2}
  internal::pointWiseScaleWithDiagonal(kohnShamDFTEigenOperator.d_sqrtMassVector,
				       matrix_free_data.get_vector_partitioner(),
				       numEigenValues,
				       localProc_dof_indicesReal,
				       d_eigenVectorsFlattened[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
				       constraintsNoneDataInfo);

  std::vector<double> eigenValuesTemp(numEigenValues,0.0);

  subspaceIterationSolver.reinitSpectrumBounds(a0[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
					       bLow[(1+dftParameters::spinPolarized)*kPointIndex+spinType]);

  subspaceIterationSolver.solve(kohnShamDFTEigenOperator,
  				d_eigenVectorsFlattened[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
				d_tempEigenVec,
				numEigenValues,
  				eigenValuesTemp,
				residualNormWaveFunctions,
				interBandGroupComm);

  //
  //scale the eigenVectors with M^{-1/2} to represent the wavefunctions in the usual FE basis
  //
  internal::pointWiseScaleWithDiagonal(kohnShamDFTEigenOperator.d_invSqrtMassVector,
				       matrix_free_data.get_vector_partitioner(),
				       numEigenValues,
				       localProc_dof_indicesReal,
				       d_eigenVectorsFlattened[(1+dftParameters::spinPolarized)*kPointIndex+spinType],
				       constraintsNoneDataInfo);

  //
  //copy the eigenValues and corresponding residual norms back to data members
  //
  for(unsigned int i = 0; i < (unsigned int)numEigenValues; i++)
    {
      //if(dftParameters::verbosity>=2)
      //    pcout<<"eigen value "<< std::setw(3) <<i <<": "<<eigenValuesTemp[i] <<std::endl;

      eigenValues[kPointIndex][spinType*numEigenValues + i] =  eigenValuesTemp[i];
    }

  if (dftParameters::verbosity>=2)
     pcout <<std::endl;


  //set a0 and bLow
  a0[(1+dftParameters::spinPolarized)*kPointIndex+spinType]=eigenValuesTemp[0];
  bLow[(1+dftParameters::spinPolarized)*kPointIndex+spinType]=eigenValuesTemp.back();
  //


  computing_timer.exit_section("Chebyshev solve");
}



template<unsigned int FEOrder>
void dftClass<FEOrder>::computeResidualNorm(const std::vector<double> & eigenValuesTemp,
					    eigenClass<FEOrder> & kohnShamDFTEigenOperator,
					    std::vector<vectorType> & X,
					    std::vector<double> & residualNorm) const
{


  std::vector<vectorType> PSI(X.size());

  for(unsigned int i = 0; i < X.size(); ++i)
      PSI[i].reinit(X[0]);


  kohnShamDFTEigenOperator.HX(X, PSI);

  if (dftParameters::verbosity>=3)
     pcout<<"L-2 Norm of residue   :"<<std::endl;

  for(unsigned int i = 0; i < eigenValuesTemp.size(); i++)
    {
      (PSI[i]).add(-eigenValuesTemp[i],X[i]) ;
      const double resNorm= (PSI[i]).l2_norm();
      residualNorm[i]=resNorm;

      if (dftParameters::verbosity>=3)
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
