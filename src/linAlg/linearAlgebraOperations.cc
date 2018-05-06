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
// @author Phani Motamarri 
//


/** @file linearAlgebraOperations.cc
 *  @brief Contains linear algebra operations
 *
 */

#include <linearAlgebraOperations.h>
#include <dftParameters.h>



namespace dftfe{


namespace linearAlgebraOperations
{
#ifdef ENABLE_PERIODIC_BC
  std::complex<double> innerProduct(operatorDFTClass & operatorMatrix,
				    const vectorType & X,
				    const vectorType & Y)
    {
      unsigned int dofs_per_proc = X.local_size()/2; 
      std::vector<double> xReal(dofs_per_proc), xImag(dofs_per_proc);
      std::vector<double> yReal(dofs_per_proc), yImag(dofs_per_proc);
      std::vector<std::complex<double> > xlocal(dofs_per_proc);
      std::vector<std::complex<double> > ylocal(dofs_per_proc);

   
      X.extract_subvector_to(operatorMatrix.getLocalDofIndicesReal()->begin(), 
			     operatorMatrix.getLocalDofIndicesReal()->end(), 
			     xReal.begin()); 

      X.extract_subvector_to(operatorMatrix.getLocalDofIndicesImag()->begin(), 
			     operatorMatrix.getLocalDofIndicesImag()->end(), 
			     xImag.begin());
  
      Y.extract_subvector_to(operatorMatrix.getLocalDofIndicesReal()->begin(), 
			     operatorMatrix.getLocalDofIndicesReal()->end(), 
			     yReal.begin()); 

      Y.extract_subvector_to(operatorMatrix.getLocalDofIndicesImag()->begin(), 
			     operatorMatrix.getLocalDofIndicesImag()->end(), 
			     yImag.begin());


      for(int i = 0; i < dofs_per_proc; ++i)
	{
	  xlocal[i].real(xReal[i]);
	  xlocal[i].imag(xImag[i]);
	  ylocal[i].real(yReal[i]);
	  ylocal[i].imag(yImag[i]);
	}

      const int inc = 1;
      const int n = dofs_per_proc;

      std::complex<double>  localInnerProduct;

      zdotc_(&localInnerProduct,
	     &n,
	     &xlocal[0],
	     &inc,
	     &ylocal[0],
	     &inc);

      std::complex<double> returnValue(0.0,0.0);

      MPI_Allreduce(&localInnerProduct,
		    &returnValue,
		    1,
		    MPI_C_DOUBLE_COMPLEX,
		    MPI_SUM,
		    operatorMatrix.getMPICommunicator());

      return returnValue;
    }

  void  alphaTimesXPlusY(operatorDFTClass & operatorMatrix,
			 std::complex<double> & alpha,
			 vectorType & x,
			 vectorType & y)
  {
    const unsigned int dofs_per_proc = x.local_size()/2; 
    std::vector<double> xReal(dofs_per_proc), xImag(dofs_per_proc);
    std::vector<double> yReal(dofs_per_proc), yImag(dofs_per_proc);
    std::vector<std::complex<double> > xlocal(dofs_per_proc);
    std::vector<std::complex<double> > ylocal(dofs_per_proc);

    x.extract_subvector_to(operatorMatrix.getLocalDofIndicesReal()->begin(), 
			   operatorMatrix.getLocalDofIndicesReal()->end(), 
			   xReal.begin()); 

    x.extract_subvector_to(operatorMatrix.getLocalDofIndicesImag()->begin(), 
			   operatorMatrix.getLocalDofIndicesImag()->end(), 
			   xImag.begin());
  
    y.extract_subvector_to(operatorMatrix.getLocalDofIndicesReal()->begin(), 
			   operatorMatrix.getLocalDofIndicesReal()->end(), 
			   yReal.begin()); 

    y.extract_subvector_to(operatorMatrix.getLocalDofIndicesImag()->begin(), 
			   operatorMatrix.getLocalDofIndicesImag()->end(), 
			   yImag.begin());

    for(unsigned int i = 0; i < dofs_per_proc; ++i)
      {
	xlocal[i].real(xReal[i]);
	xlocal[i].imag(xImag[i]);
	ylocal[i].real(yReal[i]);
	ylocal[i].imag(yImag[i]);
      }

    const int n = dofs_per_proc;const int inc = 1;

      //call blas function
      zaxpy_(&n,
	     &alpha,
	     &xlocal[0],
	     &inc,
	     &ylocal[0],
	     &inc);

      //
      //initialize y to zero before copying ylocal values to y
      //
      y = 0.0;
      for(unsigned int i = 0; i < dofs_per_proc; ++i)
	{
	  y.local_element((*operatorMatrix.getLocalProcDofIndicesReal())[i]) = ylocal[i].real();
	  y.local_element((*operatorMatrix.getLocalProcDofIndicesImag())[i]) = ylocal[i].imag();

	}

      y.update_ghost_values();
    }
#endif

  //
  // evaluate upper bound of the spectrum using k-step Lanczos iteration
  //
  double lanczosUpperBoundEigenSpectrum(operatorDFTClass & operatorMatrix,
					const vectorType & vect)
  {
      
      const unsigned int this_mpi_process = dealii::Utilities::MPI::this_mpi_process(operatorMatrix.getMPICommunicator());



    const unsigned int lanczosIterations=10;
    double beta;


#ifdef ENABLE_PERIODIC_BC
      std::complex<double> alpha,alphaNeg;
#else
      double alpha;
#endif

      //
      //generate random vector v
      //
      vectorType vVector, fVector, v0Vector;
      vVector.reinit(vect);
      fVector.reinit(vect);

      vVector = 0.0,fVector = 0.0;
      std::srand(this_mpi_process);
      const unsigned int local_size = vVector.local_size();
      std::vector<dealii::IndexSet::size_type> local_dof_indices(local_size);
      vVector.locally_owned_elements().fill_index_vector(local_dof_indices);
      std::vector<double> local_values(local_size, 0.0);

      for (unsigned int i = 0; i < local_size; i++) 
	{
	  local_values[i] = ((double)std::rand())/((double)RAND_MAX);
	}
 
      operatorMatrix.getConstraintMatrixEigen()->distribute_local_to_global(local_values, 
									     local_dof_indices, 
									     vVector);
      vVector.compress(dealii::VectorOperation::add);
    
      //
      //evaluate l2 norm
      //
      vVector/=vVector.l2_norm();
      vVector.update_ghost_values();

      //
      //call matrix times X
      //
      std::vector<vectorType> v(1),f(1); 
      v[0] = vVector;
      f[0] = fVector;
      operatorMatrix.HX(v,f);
      fVector = f[0];

#ifdef ENABLE_PERIODIC_BC
      //evaluate fVector^{H}*vVector
      alpha=innerProduct(operatorMatrix,fVector,vVector);
      alphaNeg = -alpha;
      alphaTimesXPlusY(operatorMatrix,alphaNeg,vVector,fVector);
      std::vector<std::complex<double> > T(lanczosIterations*lanczosIterations,0.0);
#else
      alpha=fVector*vVector;
      fVector.add(-1.0*alpha,vVector);
      std::vector<double> T(lanczosIterations*lanczosIterations,0.0); 
#endif

      T[0]=alpha;
      unsigned index=0;

      //filling only lower triangular part
      for (unsigned int j=1; j<lanczosIterations; j++)
	{
	  beta=fVector.l2_norm();
	  v0Vector = vVector; vVector.equ(1.0/beta,fVector);
	  v[0] = vVector,f[0] = fVector;
	  operatorMatrix.HX(v,f); 
	  fVector = f[0];
	  fVector.add(-1.0*beta,v0Vector);//beta is real
#ifdef ENABLE_PERIODIC_BC
	  alpha = innerProduct(operatorMatrix,fVector,vVector);
	  alphaNeg = -alpha;
	  alphaTimesXPlusY(operatorMatrix,alphaNeg,vVector,fVector);
#else      
	  alpha = fVector*vVector;  
	  fVector.add(-1.0*alpha,vVector);
#endif
	  index+=1;
	  T[index]=beta; 
	  index+=lanczosIterations;
	  T[index]=alpha;
	}

      //eigen decomposition to find max eigen value of T matrix
      std::vector<double> eigenValuesT(lanczosIterations);
      char jobz='N', uplo='L';
      int n = lanczosIterations, lda = lanczosIterations, info;
      int lwork = 1 + 6*n + 2*n*n, liwork = 3 + 5*n;
      std::vector<int> iwork(liwork, 0);
 
#ifdef ENABLE_PERIODIC_BC
      int lrwork = 1 + 5*n + 2*n*n;
      std::vector<double> rwork(lrwork,0.0); 
      std::vector<std::complex<double> > work(lwork);
      zheevd_(&jobz, &uplo, &n, &T[0], &lda, &eigenValuesT[0], &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info);
#else
      std::vector<double> work(lwork, 0.0);
      dsyevd_(&jobz, &uplo, &n, &T[0], &lda, &eigenValuesT[0], &work[0], &lwork, &iwork[0], &liwork, &info);
#endif

  
      for (unsigned int i=0; i<eigenValuesT.size(); i++){eigenValuesT[i]=std::abs(eigenValuesT[i]);}
      std::sort(eigenValuesT.begin(),eigenValuesT.end()); 
      //
      if (dftParameters::verbosity==2)
	{
	  char buffer[100];
	  sprintf(buffer, "bUp1: %18.10e,  bUp2: %18.10e\n", eigenValuesT[lanczosIterations-1], fVector.l2_norm());
	  //pcout << buffer;
	}

      return (eigenValuesT[lanczosIterations-1]+fVector.l2_norm());
    }


  //
  //chebyshev filtering of given subspace X
  //
  void chebyshevFilter(operatorDFTClass & operatorMatrix,
		       std::vector<vectorType> & X,
		       const unsigned int m,
		       const double a,
		       const double b,
		       const double a0)
  {


      double e, c, sigma, sigma1, sigma2, gamma;
      e=(b-a)/2.0; c=(b+a)/2.0;
      sigma=e/(a0-c); sigma1=sigma; gamma=2.0/sigma1;

      std::vector<vectorType> PSI(X.size());
      std::vector<vectorType> tempPSI(X.size());
  
      for(unsigned int i = 0; i < X.size(); ++i)
	{
	  PSI[i].reinit(X[0]);
	  tempPSI[i].reinit(X[0]);
	}
    
      //Y=alpha1*(HX+alpha2*X)
      double alpha1=sigma1/e, alpha2=-c;
      operatorMatrix.HX(X, PSI);

      for (std::vector<vectorType>::iterator y=PSI.begin(), x=X.begin(); y<PSI.end(); ++y, ++x)
	{  
	  (*y).add(alpha2,*x);
	  (*y)*=alpha1;
	} 

      //loop over polynomial order
      for(unsigned int i=2; i<m+1; i++)
	{
	  sigma2=1.0/(gamma-sigma);
	  //Ynew=alpha1*(HY-cY)+alpha2*X
	  alpha1=2.0*sigma2/e, alpha2=-(sigma*sigma2);
	  operatorMatrix.HX(PSI, tempPSI);
	  for(std::vector<vectorType>::iterator ynew=tempPSI.begin(), y=PSI.begin(), x=X.begin(); ynew<tempPSI.end(); ++ynew, ++y, ++x)
	    {  
	      (*ynew).add(-c,*y);
	      (*ynew)*=alpha1;
	      (*ynew).add(alpha2,*x);
	      *x=*y;
	      *y=*ynew;
	    }
	  sigma=sigma2;
	}
  
    //copy back PSI to eigenVectors
    for (std::vector<vectorType>::iterator y=PSI.begin(), x=X.begin(); y<PSI.end(); ++y, ++x)
      {  
	*x=*y;
      }   
  }

  //
  //Gram-Schmidt orthogonalization of given subspace X
  //
  void gramSchmidtOrthogonalization(operatorDFTClass & operatorMatrix,
				    std::vector<vectorType> & X)
  {

    
#ifdef ENABLE_PERIODIC_BC
      unsigned int localSize = X[0].local_size()/2;
#else
      unsigned int localSize = X[0].local_size();
#endif

  
      //copy to petsc vectors
      unsigned int numVectors = X.size();
      Vec vec;
      VecCreateMPI(operatorMatrix.getMPICommunicator(), localSize, PETSC_DETERMINE, &vec);
      VecSetFromOptions(vec);
      //
      Vec *petscColumnSpace;
      VecDuplicateVecs(vec, numVectors, &petscColumnSpace);
      VecDestroy(&vec);

      //copy into petsc vectors
#ifdef ENABLE_PERIODIC_BC
      PetscScalar ** columnSpacePointer;
      VecGetArrays(petscColumnSpace, numVectors, &columnSpacePointer);
      for(int i = 0; i < numVectors; ++i)
	{
	  std::vector<std::complex<double> > localData(localSize);
	  std::vector<double> tempReal(localSize),tempImag(localSize);
	  X[i].extract_subvector_to(operatorMatrix.getLocalDofIndicesReal()->begin(),
				    operatorMatrix.getLocalDofIndicesReal()->end(),
				    tempReal.begin());

	  X[i].extract_subvector_to(operatorMatrix.getLocalDofIndicesImag()->begin(),
				    operatorMatrix.getLocalDofIndicesImag()->end(),
				    tempImag.begin());

	  for(int j = 0; j < localSize; ++j)
	    {
	      localData[j].real(tempReal[j]);
	      localData[j].imag(tempImag[j]);
	    }
	  std::copy(localData.begin(),localData.end(), &(columnSpacePointer[i][0])); 
	}
      VecRestoreArrays(petscColumnSpace, numVectors, &columnSpacePointer);
#else
      PetscScalar ** columnSpacePointer;
      VecGetArrays(petscColumnSpace, numVectors, &columnSpacePointer);
      for(int i = 0; i < numVectors; ++i)
	{
	  std::vector<double> localData(localSize);
	  std::copy(X[i].begin(),X[i].end(),localData.begin());
	  std::copy(localData.begin(),localData.end(), &(columnSpacePointer[i][0])); 
	}
      VecRestoreArrays(petscColumnSpace, numVectors, &columnSpacePointer);
#endif

      //
      BV slepcColumnSpace;
      BVCreate(operatorMatrix.getMPICommunicator(),&slepcColumnSpace);
      BVSetFromOptions(slepcColumnSpace);
      BVSetSizesFromVec(slepcColumnSpace,petscColumnSpace[0],numVectors);
      BVSetType(slepcColumnSpace,"vecs");
      PetscInt numVectors2=numVectors;
      BVInsertVecs(slepcColumnSpace,0, &numVectors2,petscColumnSpace,PETSC_FALSE);
      BVOrthogonalize(slepcColumnSpace,NULL);
      //
      for(int i = 0; i < numVectors; ++i){
	BVCopyVec(slepcColumnSpace,i,petscColumnSpace[i]);
      }
      BVDestroy(&slepcColumnSpace);
      //

#ifdef ENABLE_PERIODIC_BC
      VecGetArrays(petscColumnSpace, numVectors, &columnSpacePointer);
      for (int i = 0; i < numVectors; ++i)
	{
	  std::vector<std::complex<double> > localData(localSize);
	  std::copy(&(columnSpacePointer[i][0]),&(columnSpacePointer[i][localSize]), localData.begin()); 
	  for(int j = 0; j < localSize; ++j)
	    {
	      X[i].local_element((*operatorMatrix.getLocalProcDofIndicesReal())[j]) = localData[j].real();
	      X[i].local_element((*operatorMatrix.getLocalProcDofIndicesImag())[j]) = localData[j].imag();
	    }
	  X[i].update_ghost_values();
	}
      VecRestoreArrays(petscColumnSpace, numVectors, &columnSpacePointer);
#else
      VecGetArrays(petscColumnSpace, numVectors, &columnSpacePointer);
      for (int i = 0; i < numVectors; ++i)
	{
	  std::vector<double> localData(localSize);
	  std::copy(&(columnSpacePointer[i][0]),&(columnSpacePointer[i][localSize]), localData.begin()); 
	  std::copy(localData.begin(), localData.end(), X[i].begin());
	  X[i].update_ghost_values();
	}
      VecRestoreArrays(petscColumnSpace, numVectors, &columnSpacePointer);
#endif
      //
      VecDestroyVecs(numVectors, &petscColumnSpace);
    }//end of Gram-Schmidt


  //
  //Rayleigh-Ritz projection of given subspace X
  //
  void rayleighRitz(operatorDFTClass        & operatorMatrix,
		    std::vector<vectorType> & X,
		    std::vector<double>     & eigenValues)
  {

#ifdef ENABLE_PERIODIC_BC
      std::vector<std::complex<double> > ProjHam;
#else
      std::vector<double> ProjHam;
#endif

      //compute projected Hamiltonian
      operatorMatrix.XtHX(X,ProjHam);

      //compute the eigen decomposition of ProjHam
      int n = X.size(), lda = X.size(), info;
      int lwork = 1 + 6*n + 2*n*n, liwork = 3 + 5*n;
      std::vector<int> iwork(liwork,0);
      char jobz='V', uplo='U';

#ifdef ENABLE_PERIODIC_BC
      int lrwork = 1 + 5*n + 2*n*n;
      std::vector<double> rwork(lrwork,0.0); 
      std::vector<std::complex<double> > work(lwork);
      zheevd_(&jobz, &uplo, &n, &ProjHam[0],&lda,&eigenValues[0], &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info);
#else
      std::vector<double> work(lwork);
      dsyevd_(&jobz, &uplo, &n, &ProjHam[0], &lda, &eigenValues[0], &work[0], &lwork, &iwork[0], &liwork, &info);
#endif

      //rotate the basis PSI = PSI*Q
      int m = X.size();
#ifdef ENABLE_PERIODIC_BC
      int n1 = X[0].local_size()/2;
      std::vector<std::complex<double> > Xbar(n1*m), Xlocal(n1*m); //Xbar=Xlocal*Q
      std::vector<std::complex<double> >::iterator val = Xlocal.begin();
      for(std::vector<vectorType>::iterator x=X.begin(); x<X.end(); ++x)
	{
	  for (unsigned int i=0; i<(unsigned int)n1; i++)
	    {
	      (*val).real((*x).local_element((*operatorMatrix.getLocalProcDofIndicesReal())[i]));
	      (*val).imag((*x).local_element((*operatorMatrix.getLocalProcDofIndicesImag())[i]));
	      val++;
	    }
	}
#else
      int n1 = X[0].local_size();
      std::vector<double> Xbar(n1*m), Xlocal(n1*m); //Xbar=Xlocal*Q
      std::vector<double>::iterator val = Xlocal.begin();
      for (std::vector<vectorType>::iterator x = X.begin(); x < X.end(); ++x)
	{
	  for (unsigned int i=0; i<(unsigned int)n1; i++)
	    {
	      *val=(*x).local_element(i); 
	      val++;
	    }
	}
#endif

      const char transA  = 'N', transB  = 'N';
      lda=n1; int ldb=m, ldc=n1;

#ifdef ENABLE_PERIODIC_BC
      const std::complex<double> alpha = 1.0, beta  = 0.0;
      zgemm_(&transA, &transB, &n1, &m, &m, &alpha, &Xlocal[0], &lda, &ProjHam[0], &ldb, &beta, &Xbar[0], &ldc);
#else
      const double alpha = 1.0, beta  = 0.0;
      dgemm_(&transA, &transB, &n1, &m, &m, &alpha, &Xlocal[0], &lda, &ProjHam[0], &ldb, &beta, &Xbar[0], &ldc);
#endif


#ifdef ENABLE_PERIODIC_BC
      //copy back Xbar to X
      val = Xbar.begin();
      for(std::vector<vectorType>::iterator x = X.begin(); x < X.end(); ++x)
	{
	  *x=0.0;
	  for(unsigned int i=0; i<(unsigned int)n1; i++)
	    {
	      (*x).local_element((*operatorMatrix.getLocalProcDofIndicesReal())[i]) = (*val).real(); 
	      (*x).local_element((*operatorMatrix.getLocalProcDofIndicesImag())[i]) = (*val).imag(); 
	      val++;
	    }
	  (*x).update_ghost_values();
	}
#else
      //copy back Xbar to X
      val=Xbar.begin();
      for(std::vector<vectorType>::iterator x=X.begin(); x<X.end(); ++x)
	{
	  *x=0.0;
	  for(unsigned int i=0; i<(unsigned int)n1; i++)
	    {
	      (*x).local_element(i)=*val; val++;
	    }
	  (*x).update_ghost_values();
	}
#endif

    }

  }//end of namespace

}
