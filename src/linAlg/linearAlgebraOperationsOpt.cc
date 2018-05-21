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
// @author Phani Motamarri (2018)
//


/** @file linearAlgebraOperationsOpt.cc
 *  @brief Contains linear algebra operations
 *
 */

#include <linearAlgebraOperations.h>
#include <dftParameters.h>


namespace dftfe{

  namespace linearAlgebraOperations
  {

    void callevd(const unsigned int dimensionMatrix,
		 double *matrix,
		 double *eigenValues)
    {

      int info;
      const unsigned int lwork = 1 + 6*dimensionMatrix + 2*dimensionMatrix*dimensionMatrix, liwork = 3 + 5*dimensionMatrix;
      std::vector<int> iwork(liwork,0);
      const char jobz='V', uplo='U';
      std::vector<double> work(lwork);

      dsyevd_(&jobz,
	      &uplo,
	      &dimensionMatrix,
	      matrix,
	      &dimensionMatrix,
	      eigenValues,
	      &work[0],
	      &lwork,
	      &iwork[0],
	      &liwork,
	      &info);
    }

    
   


    void callevd(const unsigned int dimensionMatrix,
		 std::complex<double> *matrix,
		 double *eigenValues)
    {
      int info;
      const unsigned int lwork = 1 + 6*dimensionMatrix + 2*dimensionMatrix*dimensionMatrix, liwork = 3 + 5*dimensionMatrix;
      std::vector<int> iwork(liwork,0);
      const char jobz='V', uplo='U';
      const unsigned int lrwork = 1 + 5*dimensionMatrix + 2*dimensionMatrix*dimensionMatrix;
      std::vector<double> rwork(lrwork,0.0);
      std::vector<std::complex<double> > work(lwork);


      zheevd_(&jobz,
	      &uplo,
	      &dimensionMatrix,
	      matrix,
	      &dimensionMatrix,
	      eigenValues,
	      &work[0],
	      &lwork,
	      &rwork[0],
	      &lrwork,
	      &iwork[0],
	      &liwork,
	      &info);
    }


    void callgemm(const unsigned int numberEigenValues,
		  const unsigned int localVectorSize,
		  const std::vector<double> & eigenVectorSubspaceMatrix,
		  const dealii::parallel::distributed::Vector<double> & X,
		  dealii::parallel::distributed::Vector<double> & Y)

    {

      const char transA  = 'T', transB  = 'N';
      const double alpha = 1.0, beta = 0.0;
      dgemm_(&transA,
	     &transB,
	     &numberEigenValues,
	     &localVectorSize,
	     &numberEigenValues,
	     &alpha,
	     &eigenVectorSubspaceMatrix[0],
	     &numberEigenValues,
	     X.begin(),
	     &numberEigenValues,
	     &beta,
	     Y.begin(),
	     &numberEigenValues);

    }


    void callgemm(const unsigned int numberEigenValues,
		  const unsigned int localVectorSize,
		  const std::vector<std::complex<double> > & eigenVectorSubspaceMatrix,
		  const dealii::parallel::distributed::Vector<std::complex<double> > & X,
		  dealii::parallel::distributed::Vector<std::complex<double> > & Y)

    {

      const char transA  = 'T', transB  = 'N';
      const std::complex<double> alpha = 1.0, beta = 0.0;
      zgemm_(&transA,
	     &transB,
	     &numberEigenValues,
	     &localVectorSize,
	     &numberEigenValues,
	     &alpha,
	     &eigenVectorSubspaceMatrix[0],
	     &numberEigenValues,
	     X.begin(),
	     &numberEigenValues,
	     &beta,
	     Y.begin(),
	     &numberEigenValues);

    }



    //
    //chebyshev filtering of given subspace XArray
    //
    template<typename T>
    void chebyshevFilter(operatorDFTClass & operatorMatrix,
			 dealii::parallel::distributed::Vector<T> & XArray,
			 const unsigned int numberWaveFunctions,
			 const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayMacroCellLocalProcIndexIdMap,
			 const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
			 const unsigned int m,
			 const double a,
			 const double b,
			 const double a0)
    {

      double e, c, sigma, sigma1, sigma2, gamma;
      e = (b-a)/2.0; c = (b+a)/2.0;
      sigma = e/(a0-c); sigma1 = sigma; gamma = 2.0/sigma1;

      dealii::parallel::distributed::Vector<T> YArray;//,YNewArray;

      //
      //create YArray
      //
      YArray.reinit(XArray);


      //
      //initialize to zeros.
      //x
      const T zeroValue = 0.0;
      YArray = zeroValue;


      //
      //call HX
      //
      bool scaleFlag = false;
      T scalar = 1.0;
      operatorMatrix.HX(XArray,
			numberWaveFunctions,
			flattenedArrayMacroCellLocalProcIndexIdMap,
			flattenedArrayCellLocalProcIndexIdMap,
			scaleFlag,
			scalar,
			YArray);


      T alpha1 = sigma1/e, alpha2 = -c;

      //
      //YArray = YArray + alpha2*XArray and YArray = alpha1*YArray
      //
      YArray.add(alpha2,XArray);
      YArray *= alpha1;

      //
      //polynomial loop
      //
      for(unsigned int degree = 2; degree < m+1; ++degree)
	{
	  sigma2 = 1.0/(gamma - sigma);
	  alpha1 = 2.0*sigma2/e, alpha2 = -(sigma*sigma2);

	  //
	  //multiply XArray with alpha2
	  //
	  XArray *= alpha2;
	  XArray.add(-c*alpha1,YArray);


	  //
	  //call HX
	  //
	  bool scaleFlag = true;
	  operatorMatrix.HX(YArray,
			    numberWaveFunctions,
			    flattenedArrayMacroCellLocalProcIndexIdMap,
			    flattenedArrayCellLocalProcIndexIdMap,
			    scaleFlag,
			    alpha1,
			    XArray);

	  //
	  //XArray = YArray
	  //
	  XArray.swap(YArray);

	  //
	  //YArray = YNewArray
	  //
	  sigma = sigma2;

	}

      //copy back YArray to XArray
      XArray = YArray;
    }

    template<typename T>
    void gramSchmidtOrthogonalization(dealii::parallel::distributed::Vector<T> & X,
				      const unsigned int numberVectors)
    {

      const unsigned int localVectorSize = X.local_size()/numberVectors;

      //
      //Create template PETSc vector to create BV object later
      //
      Vec templateVec;
      VecCreateMPI(X.get_mpi_communicator(),
		   localVectorSize,
		   PETSC_DETERMINE,
		   &templateVec);
      VecSetFromOptions(templateVec);


      //
      //Set BV options after creating BV object
      //
      BV columnSpaceOfVectors;
      BVCreate(X.get_mpi_communicator(),&columnSpaceOfVectors);
      BVSetSizesFromVec(columnSpaceOfVectors,
			templateVec,
			numberVectors);
      BVSetFromOptions(columnSpaceOfVectors);


      //
      //create list of indices
      //
      std::vector<PetscInt> indices(localVectorSize);
      std::vector<PetscScalar> data(localVectorSize);

      PetscInt low,high;

      VecGetOwnershipRange(templateVec,
			   &low,
			   &high);


      for(PetscInt index = 0;index < localVectorSize; ++index)
	indices[index] = low+index;

      VecDestroy(&templateVec);

      //
      //Fill in data into BV object
      //
      Vec v;
      for(unsigned int iColumn = 0; iColumn < numberVectors; ++iColumn)
	{
	  BVGetColumn(columnSpaceOfVectors,
		      iColumn,
		      &v);
	  VecSet(v,0.0);
	  for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	    data[iNode] = X.local_element(numberVectors*iNode + iColumn);

	  VecSetValues(v,
		       localVectorSize,
		       &indices[0],
		       &data[0],
		       INSERT_VALUES);

	  VecAssemblyBegin(v);
	  VecAssemblyEnd(v);

	  BVRestoreColumn(columnSpaceOfVectors,
			  iColumn,
			  &v);
	}

      //
      //orthogonalize
      //
      BVOrthogonalize(columnSpaceOfVectors,NULL);

      //
      //Copy data back into X
      //
      Vec v1;
      PetscScalar * pointerv1;
      for(unsigned int iColumn = 0; iColumn < numberVectors; ++iColumn)
	{
	  BVGetColumn(columnSpaceOfVectors,
		      iColumn,
		      &v1);

	  VecGetArray(v1,
		      &pointerv1);

	  for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	    X.local_element(numberVectors*iNode + iColumn) = pointerv1[iNode];

	  VecRestoreArray(v1,
			  &pointerv1);

	  BVRestoreColumn(columnSpaceOfVectors,
			  iColumn,
			  &v1);
	}

      BVDestroy(&columnSpaceOfVectors);

    }

    template<typename T>
    void rayleighRitz(operatorDFTClass & operatorMatrix,
		      dealii::parallel::distributed::Vector<T> & X,
		      const unsigned int numberWaveFunctions,
		      const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayMacroCellLocalProcIndexIdMap,
		      const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
		      std::vector<double> & eigenValues)
    {

      //
      //compute projected Hamiltonian
      //
      std::vector<T> ProjHam;
      const unsigned int numberEigenValues = eigenValues.size();
      operatorMatrix.XtHX(X,
			  numberEigenValues,
			  flattenedArrayMacroCellLocalProcIndexIdMap,
			  flattenedArrayCellLocalProcIndexIdMap,
			  ProjHam);

      //
      //compute eigendecomposition of ProjHam
      //
      callevd(numberEigenValues,
	      &ProjHam[0],
	      &eigenValues[0]);


      //
      //rotate the basis in the subspace X = X*Q
      //
      const unsigned int localVectorSize = X.local_size()/numberEigenValues;
      dealii::parallel::distributed::Vector<T> rotatedBasis;
      rotatedBasis.reinit(X);

      callgemm(numberEigenValues,
	       localVectorSize,
	       ProjHam,
	       X,
	       rotatedBasis);


      X = rotatedBasis;

    }

    template<typename T>
    void computeEigenResidualNorm(operatorDFTClass & operatorMatrix,
				  dealii::parallel::distributed::Vector<T> & X,
				  const std::vector<double> & eigenValues,
				  const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayMacroCellLocalProcIndexIdMap,
				  const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
				
				  std::vector<double> & residualNorm)

    {
      
      //
      //get the number of eigenVectors
      //
      const unsigned int numberVectors = eigenValues.size();

      //
      //reinit blockSize require for HX later
      //
      operatorMatrix.reinit(numberVectors);

      //
      //create temp Array
      //
      dealii::parallel::distributed::Vector<T> Y;
      Y.reinit(X);

      //
      //initialize to zero
      //
      const T zeroValue = 0.0;
      Y = zeroValue;

      //
      //compute operator times X
      //
      bool scaleFlag = false;
      T scalar = 1.0;
      operatorMatrix.HX(X,
			numberVectors,
			flattenedArrayMacroCellLocalProcIndexIdMap,
			flattenedArrayCellLocalProcIndexIdMap,
			scaleFlag,
			scalar,
			Y);

      if(dftParameters::verbosity==2)
	{
	  if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
	    std::cout<<"L-2 Norm of residue   :"<<std::endl;
	}
      
      
      const unsigned int localVectorSize = X.local_size()/numberVectors;

      //
      //compute residual norms
      //
      std::vector<T> residualNormSquare(numberVectors,0.0);
      for(unsigned int iDof = 0; iDof < localVectorSize; ++iDof)
	{
	  for(unsigned int iWave = 0; iWave < numberVectors; iWave++)
	    {
	      T value = Y.local_element(numberVectors*iDof + iWave) - eigenValues[iWave]*X.local_element(numberVectors*iDof + iWave);
	      residualNormSquare[iWave] += std::abs(value)*std::abs(value);
	    }
	}


      dealii::Utilities::MPI::sum(residualNormSquare,X.get_mpi_communicator(),residualNormSquare);

      
      for(unsigned int iWave = 0; iWave < numberVectors; ++iWave)
	{
#ifdef USE_COMPLEX
	  double value = residualNormSquare[iWave].real();
#else
	  double value = residualNormSquare[iWave];
#endif
	  residualNorm[iWave] = sqrt(value);

	  if(dftParameters::verbosity==2)
	    {
	      if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
		std::cout<<"eigen vector "<< iWave<<": "<<residualNorm[iWave]<<std::endl;
	    }
	}

      if(dftParameters::verbosity==2)  
      {
	if(dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
	  std::cout <<std::endl;
      }
	       
    }
			     




#ifdef USE_COMPLEX
    void lowdenOrthogonalization(dealii::parallel::distributed::Vector<std::complex<double> > & X,
				 const unsigned int numberVectors)
    {

      const unsigned int localVectorSize = X.local_size()/numberVectors;
      std::vector<std::complex<double> > overlapMatrix(numberVectors*numberVectors,0.0);

      //
      //blas level 3 dgemm flags
      //
      const std::complex<double> alpha = 1.0, beta = 0.0;
      const unsigned int numberEigenValues = numberVectors;

      //
      //compute overlap matrix S = {(Zc)^T}*Z on local proc
      //where Z is a matrix with size number of degrees of freedom times number of column vectors
      //and (Zc)^T is conjugate transpose of Z
      //Since input "X" is stored as number of column vectors times number of degrees of freedom matrix
      //corresponding to column-major format required for blas, we compute
      //the transpose of overlap matrix i.e S^{T} = X*{(Xc)^T} here
      //
      const char uplo = 'U';
      const char trans = 'N';

      zsyrk_(&uplo,
	     &trans,
	     &numberVectors,
	     &localVectorSize,
	     &alpha,
	     X.begin(),
	     &numberVectors,
	     &beta,
	     &overlapMatrix[0],
	     &numberVectors);


      dealii::Utilities::MPI::sum(overlapMatrix, X.get_mpi_communicator(), overlapMatrix); 

      //
      //evaluate the conjugate of {S^T} to get actual overlap matrix
      //
      for(unsigned int i = 0; i < overlapMatrix.size(); ++i)
	overlapMatrix[i] = std::conj(overlapMatrix[i]);


      //
      //set lapack eigen decomposition flags and compute eigendecomposition of S = Q*D*Q^{H}
      //
      int info;
      const unsigned int lwork = 1 + 6*numberVectors + 2*numberVectors*numberVectors, liwork = 3 + 5*numberVectors;
      std::vector<int> iwork(liwork,0);
      const char jobz='V';
      const unsigned int lrwork = 1 + 5*numberVectors + 2*numberVectors*numberVectors;
      std::vector<double> rwork(lrwork,0.0);
      std::vector<std::complex<double> > work(lwork);
      std::vector<double> eigenValuesOverlap(numberVectors,0.0);

      zheevd_(&jobz,
	      &uplo,
	      &numberVectors,
	      &overlapMatrix[0],
	      &numberVectors,
	      &eigenValuesOverlap[0],
	      &work[0],
	      &lwork,
	      &rwork[0],
	      &lrwork,
	      &iwork[0],
	      &liwork,
	      &info);

       //
       //free up memory associated with work
       //
       work.clear();
       iwork.clear();
       rwork.clear();
       std::vector<std::complex<double> >().swap(work);
       std::vector<double>().swap(rwork);
       std::vector<int>().swap(iwork);

       //
       //compute D^{-1/4} where S = Q*D*Q^{H}
       //
       std::vector<double> invFourthRootEigenValuesMatrix(numberEigenValues,0.0);

       for(unsigned i = 0; i < numberEigenValues; ++i)
	 invFourthRootEigenValuesMatrix[i] = 1.0/pow(eigenValuesOverlap[i],1.0/4);

       //
       //Q*D^{-1/4} and note that "Q" is stored in overlapMatrix after calling "zheevd"
       //
       const unsigned int inc = 1;
       for(unsigned int i = 0; i < numberEigenValues; ++i)
	 {
	   std::complex<double> scalingCoeff = invFourthRootEigenValuesMatrix[i];
	   zscal_(&numberEigenValues,
		  &scalingCoeff,
		  &overlapMatrix[0]+i*numberEigenValues,
                  &inc);
	 }

       //
       //Evaluate S^{-1/2} = Q*D^{-1/2}*Q^{H} = (Q*D^{-1/4})*(Q*D^{-1/4))^{H}
       //
       std::vector<std::complex<double> > invSqrtOverlapMatrix(numberEigenValues*numberEigenValues,0.0);
       const char transA1 = 'N';
       const char transB1 = 'C';
       zgemm_(&transA1,
	      &transB1,
	      &numberEigenValues,
	      &numberEigenValues,
	      &numberEigenValues,
	      &alpha,
	      &overlapMatrix[0],
	      &numberEigenValues,
	      &overlapMatrix[0],
	      &numberEigenValues,
	      &beta,
	      &invSqrtOverlapMatrix[0],
	      &numberEigenValues);

       //
       //free up memory associated with overlapMatrix
       //
       overlapMatrix.clear();
       std::vector<std::complex<double> >().swap(overlapMatrix);

       //
       //Rotate the given vectors using S^{-1/2} i.e Y = X*S^{-1/2} but implemented as Y^T = {S^{-1/2}}^T*{X^T}
       //using the column major format of blas
       //
       const char transA2  = 'T', transB2  = 'N';
       dealii::parallel::distributed::Vector<std::complex<double> > orthoNormalizedBasis;
       orthoNormalizedBasis.reinit(X);
       zgemm_(&transA2,
	     &transB2,
	     &numberEigenValues,
             &localVectorSize,
	     &numberEigenValues,
	     &alpha,
	     &invSqrtOverlapMatrix[0],
	     &numberEigenValues,
	     X.begin(),
	     &numberEigenValues,
	     &beta,
	     orthoNormalizedBasis.begin(),
	     &numberEigenValues);

       
       X = orthoNormalizedBasis;
    }
#else
    void lowdenOrthogonalization(dealii::parallel::distributed::Vector<double> & X,
				 const unsigned int numberVectors)
    {

      const unsigned int localVectorSize = X.local_size()/numberVectors;
      std::vector<double> overlapMatrix(numberVectors*numberVectors,0.0);

      //
      //blas level 3 dgemm flags
      //
      const double alpha = 1.0, beta = 0.0;
      const unsigned int numberEigenValues = numberVectors;
      const char uplo = 'U';
      const char trans = 'N';

      //
      //compute overlap matrix S = {(Z)^T}*Z on local proc
      //where Z is a matrix with size number of degrees of freedom times number of column vectors
      //and (Z)^T is transpose of Z
      //Since input "X" is stored as number of column vectors times number of degrees of freedom matrix
      //corresponding to column-major format required for blas, we compute
      //the overlap matrix as S = S^{T} = X*{X^T} here
      //
      dsyrk_(&uplo,
	     &trans,
	     &numberVectors,
	     &localVectorSize,
	     &alpha,
	     X.begin(),
	     &numberVectors,
	     &beta,
	     &overlapMatrix[0],
	     &numberVectors);


      dealii::Utilities::MPI::sum(overlapMatrix, X.get_mpi_communicator(), overlapMatrix); 
      //std::vector<double> overlapMatrix(numberVectors*numberVectors,0.0);
      //const unsigned int sizeEntries = numberEigenValues*numberEigenValues;
      /*MPI_Allreduce(&overlapMatrixLocal[0],
		    &overlapMatrix[0],
		    sizeEntries,
		    MPI_DOUBLE,
		    MPI_SUM,
		    X.get_mpi_communicator());*/

      //
      //Free up memory
      //
      overlapMatrix.clear();
      std::vector<double>().swap(overlapMatrix);

      //
      //set lapack eigen decomposition flags and compute eigendecomposition of S = Q*D*Q^{H}
      //
      int info;
      const unsigned int lwork = 1 + 6*numberVectors + 2*numberVectors*numberVectors, liwork = 3 + 5*numberVectors;
      std::vector<int> iwork(liwork,0);
      const char jobz='V';
      std::vector<double> work(lwork);
      std::vector<double> eigenValuesOverlap(numberVectors,0.0);
       dsyevd_(&jobz,
	       &uplo,
	       &numberVectors,
	       &overlapMatrix[0],
	       &numberVectors,
	       &eigenValuesOverlap[0],
	       &work[0],
	       &lwork,
	       &iwork[0],
	       &liwork,
	       &info);

       //
       //free up memory associated with work
       //
       work.clear();
       iwork.clear();
       std::vector<double>().swap(work);
       std::vector<int>().swap(iwork);

       //
       //compute D^{-1/4} where S = Q*D*Q^{T}
       //
       std::vector<double> invFourthRootEigenValuesMatrix(numberEigenValues,0.0);

       for(unsigned i = 0; i < numberEigenValues; ++i)
	 {
	   invFourthRootEigenValuesMatrix[i] = 1.0/pow(eigenValuesOverlap[i],1.0/4);
	   AssertThrow(!std::isnan(invFourthRootEigenValuesMatrix[i]),dealii::ExcMessage("Eigen values of overlap matrix during Lowden Orthonormalization are very small and close to zero"));
	 }

       //
       //Q*D^{-1/4} and note that "Q" is stored in overlapMatrix after calling "dsyevd"
       //
       const unsigned int inc = 1;
       for(unsigned int i = 0; i < numberEigenValues; ++i)
	 {
	   double scalingCoeff = invFourthRootEigenValuesMatrix[i];
	   dscal_(&numberEigenValues,
		  &scalingCoeff,
		  &overlapMatrix[0]+i*numberEigenValues,
                  &inc);
	 }

       //
       //Evaluate S^{-1/2} = Q*D^{-1/2}*Q^{T} = (Q*D^{-1/4})*(Q*D^{-1/4))^{T}
       //
       std::vector<double> invSqrtOverlapMatrix(numberEigenValues*numberEigenValues,0.0);
       const char transA1 = 'N';
       const char transB1 = 'T';
       dgemm_(&transA1,
	      &transB1,
	      &numberEigenValues,
	      &numberEigenValues,
	      &numberEigenValues,
	      &alpha,
	      &overlapMatrix[0],
	      &numberEigenValues,
	      &overlapMatrix[0],
	      &numberEigenValues,
	      &beta,
	      &invSqrtOverlapMatrix[0],
	      &numberEigenValues);

       //
       //free up memory associated with overlapMatrix
       //
       overlapMatrix.clear();
       std::vector<double>().swap(overlapMatrix);

       //
       //Rotate the given vectors using S^{-1/2} i.e Y = X*S^{-1/2} but implemented as Yt = S^{-1/2}*Xt
       //using the column major format of blas
       //
       const char transA2  = 'N', transB2  = 'N';
       dealii::parallel::distributed::Vector<double> orthoNormalizedBasis;
       orthoNormalizedBasis.reinit(X);
       dgemm_(&transA2,
	     &transB2,
	     &numberEigenValues,
             &localVectorSize,
	     &numberEigenValues,
	     &alpha,
	     &invSqrtOverlapMatrix[0],
	     &numberEigenValues,
	     X.begin(),
	     &numberEigenValues,
	     &beta,
	     orthoNormalizedBasis.begin(),
	     &numberEigenValues);

       
       X = orthoNormalizedBasis;
    }
#endif



    template void chebyshevFilter(operatorDFTClass & operatorMatrix,
				  dealii::parallel::distributed::Vector<dataTypes::number> & ,
				  const unsigned int ,
				  const std::vector<std::vector<dealii::types::global_dof_index> > & ,
				  const std::vector<std::vector<dealii::types::global_dof_index> > & ,
				  const unsigned int,
				  const double ,
				  const double ,
				  const double );


    template void gramSchmidtOrthogonalization(dealii::parallel::distributed::Vector<dataTypes::number> &,
					       const unsigned int );

    template void rayleighRitz(operatorDFTClass  & operatorMatrix,
			       dealii::parallel::distributed::Vector<dataTypes::number> &,
			       const unsigned int numberWaveFunctions,
			       const std::vector<std::vector<dealii::types::global_dof_index> > &,
			       const std::vector<std::vector<dealii::types::global_dof_index> > &,
			       std::vector<double>     & eigenValues);

    template void computeEigenResidualNorm(operatorDFTClass        & operatorMatrix,
					   dealii::parallel::distributed::Vector<dataTypes::number> & X,
					   const std::vector<double> & eigenValues,
					   const std::vector<std::vector<dealii::types::global_dof_index> > & macroCellMap,
					   const std::vector<std::vector<dealii::types::global_dof_index> > & cellMap,
					   std::vector<double>     & residualNorm);




  }//end of namespace

}
