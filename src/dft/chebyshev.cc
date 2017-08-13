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

#include <complex>
#include<vector>

#ifdef ENABLE_PERIODIC_BC
template<unsigned int FEOrder>
std::complex<double> dftClass<FEOrder>::innerProduct(vectorType &  X,
						     vectorType &  Y)
{

  //std::cout<<"Entering Inner Product: "<<std::endl;
  unsigned int dofs_per_proc = X.local_size()/2; 
  std::vector<double> xReal(dofs_per_proc), xImag(dofs_per_proc);
  std::vector<double> yReal(dofs_per_proc), yImag(dofs_per_proc);
  std::vector<std::complex<double> > xlocal(dofs_per_proc);
  std::vector<std::complex<double> > ylocal(dofs_per_proc);

   
  X.extract_subvector_to(local_dof_indicesReal.begin(), 
  			 local_dof_indicesReal.end(), 
  			 xReal.begin()); 

  X.extract_subvector_to(local_dof_indicesImag.begin(), 
			 local_dof_indicesImag.end(), 
			 xImag.begin());
  
  Y.extract_subvector_to(local_dof_indicesReal.begin(), 
			 local_dof_indicesReal.end(), 
			 yReal.begin()); 

  Y.extract_subvector_to(local_dof_indicesImag.begin(), 
			 local_dof_indicesImag.end(), 
			 yImag.begin());

  //std::cout<<"Finished extracting sub vectors: "<<std::endl;

  for(int i = 0; i < dofs_per_proc; ++i)
    {
      xlocal[i].real(xReal[i]);
      xlocal[i].imag(xImag[i]);
      ylocal[i].real(yReal[i]);
      ylocal[i].imag(yImag[i]);
    }

  int inc = 1;
  int n = dofs_per_proc;

  std::complex<double>  localInnerProduct;

  zdotc_(&localInnerProduct,
	 &n,
	 &xlocal[0],
	 &inc,
	 &ylocal[0],
	 &inc);

  //std::cout<<"Finished blas operation: "<<std::endl;

  std::complex<double> returnValue(0.0,0.0);

  MPI_Allreduce(&localInnerProduct,
		&returnValue,
		1,
		MPI_C_DOUBLE_COMPLEX,
		MPI_SUM,
		mpi_communicator);

  //std::cout<<"Finished MPI Allreduce: "<<std::endl;


  return returnValue; 
}

template<unsigned int FEOrder>
void dftClass<FEOrder>::alphaTimesXPlusY(std::complex<double>   alpha,
				vectorType           & x,
				vectorType           & y)
{

  //
  //compute y = alpha*x + y
  //
  //std::cout<<"Entering alpha times X plus Y: "<<std::endl;
  //
  //extract real and imaginary parts of x and y
  //
  unsigned int dofs_per_proc = x.local_size()/2; 
  std::vector<double> xReal(dofs_per_proc), xImag(dofs_per_proc);
  std::vector<double> yReal(dofs_per_proc), yImag(dofs_per_proc);
  std::vector<std::complex<double> > xlocal(dofs_per_proc);
  std::vector<std::complex<double> > ylocal(dofs_per_proc);

   
  x.extract_subvector_to(local_dof_indicesReal.begin(), 
  			 local_dof_indicesReal.end(), 
  			 xReal.begin()); 

  x.extract_subvector_to(local_dof_indicesImag.begin(), 
			 local_dof_indicesImag.end(), 
			 xImag.begin());
  
  y.extract_subvector_to(local_dof_indicesReal.begin(), 
			 local_dof_indicesReal.end(), 
			 yReal.begin()); 

  y.extract_subvector_to(local_dof_indicesImag.begin(), 
			 local_dof_indicesImag.end(), 
			 yImag.begin());

  for(int i = 0; i < dofs_per_proc; ++i)
    {
      xlocal[i].real(xReal[i]);
      xlocal[i].imag(xImag[i]);
      ylocal[i].real(yReal[i]);
      ylocal[i].imag(yImag[i]);
    }

  int n = dofs_per_proc;int inc = 1;

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
      y.local_element(localProc_dof_indicesReal[i]) = ylocal[i].real();
      y.local_element(localProc_dof_indicesImag[i]) = ylocal[i].imag();

    }

  y.update_ghost_values();
}
#endif

//chebyshev solver
template<unsigned int FEOrder>
void dftClass<FEOrder>::chebyshevSolver()
{
  computing_timer.enter_section("Chebyshev solve"); 

  //
  //compute upper bound of spectrum
  //
  bUp = upperBound(); 

  //
  //set Chebyshev order
  //
  if(chebyshevOrder == 0)
    {
      /*if(bUp >= 1e5)
	chebyshevOrder = 1500;
      else if(bUp >= 2e4 && bUp < 1e5)
	chebyshevOrder = 800;
      else if(bUp >= 7e3 && bUp < 2e4)
	chebyshevOrder = 500;
      else if(bUp >= 2e3  && bUp  < 7e3)
	chebyshevOrder = 300;
      else if(bUp >= 6e2 && bUp < 2e3)
	chebyshevOrder = 100;
      else if(bUp < 6e2)
      chebyshevOrder = 50;*/

      if(bUp <= 500)
	chebyshevOrder = 40;
      else if(bUp > 500  && bUp <= 1000)
	chebyshevOrder = 50;
      else if(bUp > 1000 && bUp <= 2000)
	chebyshevOrder = 80;
      else if(bUp > 2000 && bUp <= 5000)
	chebyshevOrder = 150;
      else if(bUp > 5000 && bUp <= 9000)
	chebyshevOrder = 200;
      else if(bUp > 9000 && bUp <= 14000)
	chebyshevOrder = 250;
      else if(bUp > 14000 && bUp <= 20000)
	chebyshevOrder = 300;
      else if(bUp > 20000 && bUp <= 30000)
	chebyshevOrder = 350;
      else if(bUp > 30000 && bUp <= 50000)
	chebyshevOrder = 450;
      else if(bUp > 50000 && bUp <= 80000)
	chebyshevOrder = 550;
      else if(bUp > 80000 && bUp <= 1e5)
	chebyshevOrder = 800;
      else if(bUp > 1e5 && bUp <= 2e5)
	chebyshevOrder = 1000;
      else if(bUp > 2e5 && bUp <= 5e5)
	chebyshevOrder = 1250;
      else if(bUp > 5e5)
	chebyshevOrder = 1500;

    }

  //
  //output statements
  //
  char buffer[100];

  sprintf(buffer, "%s:%18.10e\n", "upper bound of unwanted spectrum", bUp);
  pcout << buffer;
  sprintf(buffer, "%s:%18.10e\n", "lower bound of unwanted spectrum", bLow[d_kPointIndex]);
  pcout << buffer;
  sprintf(buffer, "%s: %u\n\n", "Chebyshev polynomial degree", chebyshevOrder);
  pcout << buffer;


  //
  //Filter
  //
  chebyshevFilter(eigenVectors[d_kPointIndex], chebyshevOrder, bLow[d_kPointIndex], bUp, a0[d_kPointIndex]);
 

  //
  //Gram Schmidt orthonormalization
  //
  gramSchmidt(eigenVectors[d_kPointIndex]);

  //
  //Rayleigh Ritz step
  //
  rayleighRitz(eigenVectors[d_kPointIndex]);
  computing_timer.exit_section("Chebyshev solve"); 
}

template<unsigned int FEOrder>
double dftClass<FEOrder>::upperBound()
{
  computing_timer.enter_section("Chebyshev upper bound"); 
  unsigned int lanczosIterations=10;
  double beta;

#ifdef ENABLE_PERIODIC_BC
  std::complex<double> alpha;
#else
  double alpha;
#endif

  //
  //generate random vector v
  //
  vChebyshev=0.0;
  std::srand(this_mpi_process);
  const unsigned int local_size = vChebyshev.local_size();
  std::vector<unsigned int> local_dof_indices(local_size);
  vChebyshev.locally_owned_elements().fill_index_vector(local_dof_indices);
  std::vector<double> local_values(local_size, 0.0);

  for (unsigned int i=0; i<local_size; i++) 
    {
      local_values[i]= ((double)std::rand())/((double)RAND_MAX);
    }
 
  constraintsNoneEigen.distribute_local_to_global(local_values, local_dof_indices, vChebyshev);
  vChebyshev.compress(VectorOperation::add);

  //
  vChebyshev/=vChebyshev.l2_norm();
  vChebyshev.update_ghost_values();
  //
  std::vector<vectorType*> v,f; 
  v.push_back(&vChebyshev);
  f.push_back(&fChebyshev);
  eigen.HX(v,f);

  //
#ifdef ENABLE_PERIODIC_BC
  //evaluate fChebyshev^{H}*vChebyshev
  alpha=innerProduct(fChebyshev,vChebyshev);
  alphaTimesXPlusY(-alpha,vChebyshev,fChebyshev);
  std::vector<std::complex<double> > T(lanczosIterations*lanczosIterations,0.0);
#else
  alpha=fChebyshev*vChebyshev;
  fChebyshev.add(-1.0*alpha,vChebyshev);
  std::vector<double> T(lanczosIterations*lanczosIterations,0.0); 
#endif
  
 
  
  T[0]=alpha;
  unsigned index=0;

  //filling only lower trangular part
  for (unsigned int j=1; j<lanczosIterations; j++)
    {
      beta=fChebyshev.l2_norm();
      v0Chebyshev=vChebyshev; vChebyshev.equ(1.0/beta,fChebyshev);
      eigen.HX(v,f); fChebyshev.add(-1.0*beta,v0Chebyshev);//beta is real
#ifdef ENABLE_PERIODIC_BC
      alpha = innerProduct(fChebyshev,vChebyshev);
      alphaTimesXPlusY(-alpha,vChebyshev,fChebyshev);
#else      
      alpha = fChebyshev*vChebyshev;  
      fChebyshev.add(-1.0*alpha,vChebyshev);
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

  //
  computing_timer.exit_section("Chebyshev upper bound");
  for (unsigned int i=0; i<eigenValuesT.size(); i++){eigenValuesT[i]=std::abs(eigenValuesT[i]);}
  std::sort(eigenValuesT.begin(),eigenValuesT.end()); 
  //
  char buffer[100];
  sprintf(buffer, "bUp1: %18.10e,  bUp2: %18.10e\n", eigenValuesT[lanczosIterations-1], fChebyshev.l2_norm());
  //pcout << buffer;
  
  return (eigenValuesT[lanczosIterations-1]+fChebyshev.l2_norm());
}

//Gram-Schmidt orthonormalization
template<unsigned int FEOrder>
void dftClass<FEOrder>::gramSchmidt(std::vector<vectorType*>& X)
{
  computing_timer.enter_section("Chebyshev GS orthonormalization"); 
 
  //Memory optimization required here as well  

#ifdef ENABLE_PERIODIC_BC
  unsigned int localSize = vChebyshev.local_size()/2;
#else
  unsigned int localSize = vChebyshev.local_size();
#endif

  //copy to petsc vectors
  unsigned int numVectors = X.size();
  Vec vec;
  VecCreateMPI(PETSC_COMM_WORLD, localSize, PETSC_DETERMINE, &vec);
  VecSetFromOptions(vec);
  //
  Vec *petscColumnSpace;
  VecDuplicateVecs(vec, numVectors, &petscColumnSpace);
  VecDestroy(&vec);

  //
#ifdef ENABLE_PERIODIC_BC
  PetscScalar ** columnSpacePointer;
  VecGetArrays(petscColumnSpace, numVectors, &columnSpacePointer);
  for (int i = 0; i < numVectors; ++i)
    {
      std::vector<std::complex<double> > localData(localSize);
      //std::vector<double> localData(localSize);
      std::vector<double> tempReal(localSize),tempImag(localSize);
      X[i]->extract_subvector_to(local_dof_indicesReal.begin(),
				 local_dof_indicesReal.end(),
				 tempReal.begin());

      X[i]->extract_subvector_to(local_dof_indicesImag.begin(),
				 local_dof_indicesImag.end(),
				 tempImag.begin());

      for(int j = 0; j < localSize; ++j)
	{
	  localData[j].real(tempReal[j]);
	  localData[j].imag(tempImag[j]);
	  //localData[j] = tempReal[j];
	}
      std::copy(localData.begin(),localData.end(), &(columnSpacePointer[i][0])); 
    }
  VecRestoreArrays(petscColumnSpace, numVectors, &columnSpacePointer);
#else
  PetscScalar ** columnSpacePointer;
  VecGetArrays(petscColumnSpace, numVectors, &columnSpacePointer);
  for (int i = 0; i < numVectors; ++i){
    std::vector<double> localData(localSize);
    std::copy (X[i]->begin(),X[i]->end(),localData.begin());
    std::copy (localData.begin(),localData.end(), &(columnSpacePointer[i][0])); 
  }
  VecRestoreArrays(petscColumnSpace, numVectors, &columnSpacePointer);
#endif


  //
  BV slepcColumnSpace;
  BVCreate(PETSC_COMM_WORLD,&slepcColumnSpace);
  BVSetFromOptions(slepcColumnSpace);
  BVSetSizesFromVec(slepcColumnSpace,petscColumnSpace[0],numVectors);
  BVSetType(slepcColumnSpace,"vecs");
  int numVectors2=numVectors;
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
      //std::vector<double> localData(localSize);
      std::copy(&(columnSpacePointer[i][0]),&(columnSpacePointer[i][localSize]), localData.begin()); 
      for(int j = 0; j < localSize; ++j)
	{
	  X[i]->local_element(localProc_dof_indicesReal[j]) = localData[j].real();
	  X[i]->local_element(localProc_dof_indicesImag[j]) = localData[j].imag();
	}
      X[i]->update_ghost_values();
    }
  VecRestoreArrays(petscColumnSpace, numVectors, &columnSpacePointer);
#else
  VecGetArrays(petscColumnSpace, numVectors, &columnSpacePointer);
  for (int i = 0; i < numVectors; ++i)
    {
      std::vector<double> localData(localSize);
      std::copy(&(columnSpacePointer[i][0]),&(columnSpacePointer[i][localSize]), localData.begin()); 
      std::copy(localData.begin(), localData.end(), X[i]->begin());
      X[i]->update_ghost_values();
    }
  VecRestoreArrays(petscColumnSpace, numVectors, &columnSpacePointer);
#endif
  //
  VecDestroyVecs(numVectors, &petscColumnSpace);
  //
  computing_timer.exit_section("Chebyshev GS orthonormalization"); 
}

template<unsigned int FEOrder>
void dftClass<FEOrder>::rayleighRitz(std::vector<vectorType*> &X){
  computing_timer.enter_section("Chebyshev Rayleigh Ritz"); 
  //Hbar=Psi^T*H*Psi
  eigen.XHX(X);  //Hbar is now available as a 1D array XHXValue 

  //compute the eigen decomposition of Hbar
  int n = X.size(), lda = X.size(), info;
  int lwork = 1 + 6*n + 2*n*n, liwork = 3 + 5*n;
   std::vector<int> iwork(liwork,0);
  char jobz='V', uplo='U';

#ifdef ENABLE_PERIODIC_BC
  int lrwork = 1 + 5*n + 2*n*n;
  std::vector<double> rwork(lrwork,0.0); 
  std::vector<std::complex<double> > work(lwork);
  zheevd_(&jobz, &uplo, &n, &eigen.XHXValue[0],&lda,&eigenValues[d_kPointIndex][0], &work[0], &lwork, &rwork[0], &lrwork, &iwork[0], &liwork, &info);
#else
  std::vector<double> work(lwork);
  dsyevd_(&jobz, &uplo, &n, &eigen.XHXValue[0], &lda, &eigenValues[d_kPointIndex][0], &work[0], &lwork, &iwork[0], &liwork, &info);
#endif

  //
  //print eigen values
  //
  char buffer[100];
  pcout << "kPoint: "<< d_kPointIndex<<std::endl;
  for (unsigned int i=0; i< (unsigned int)n; i++)
    {
      sprintf(buffer, "eigen value %3u: %22.16e\n", i, eigenValues[d_kPointIndex][i]);
      pcout << buffer;
    }
  pcout <<std::endl;

  //rotate the basis PSI=PSI*Q
  int m = X.size(); 
#ifdef ENABLE_PERIODIC_BC
  int n1 = X[0]->local_size()/2;
  std::vector<std::complex<double> > Xbar(n1*m), Xlocal(n1*m); //Xbar=Xlocal*Q
  std::vector<std::complex<double> >::iterator val = Xlocal.begin();
  for(std::vector<vectorType*>::iterator x=X.begin(); x<X.end(); ++x)
    {
      for (unsigned int i=0; i<(unsigned int)n1; i++)
	{
	  (*val).real((**x).local_element(localProc_dof_indicesReal[i]));
	  (*val).imag((**x).local_element(localProc_dof_indicesImag[i]));
	   val++;
	}
    }
#else
  int n1 = X[0]->local_size();
  std::vector<double> Xbar(n1*m), Xlocal(n1*m); //Xbar=Xlocal*Q
  std::vector<double>::iterator val = Xlocal.begin();
  for (std::vector<vectorType*>::iterator x = X.begin(); x < X.end(); ++x)
    {
      for (unsigned int i=0; i<(unsigned int)n1; i++)
	{
	  *val=(**x).local_element(i); 
	  val++;
	}
    }
#endif
  
char transA  = 'N', transB  = 'N';
lda=n1; int ldb=m, ldc=n1;

#ifdef ENABLE_PERIODIC_BC
 std::complex<double> alpha = 1.0, beta  = 0.0;
 zgemm_(&transA, &transB, &n1, &m, &m, &alpha, &Xlocal[0], &lda, &eigen.XHXValue[0], &ldb, &beta, &Xbar[0], &ldc);
#else
 double alpha = 1.0, beta  = 0.0;
 dgemm_(&transA, &transB, &n1, &m, &m, &alpha, &Xlocal[0], &lda, &eigen.XHXValue[0], &ldb, &beta, &Xbar[0], &ldc);
#endif

 
#ifdef ENABLE_PERIODIC_BC
 //copy back Xbar to X
  val=Xbar.begin();
  for (std::vector<vectorType*>::iterator x=X.begin(); x<X.end(); ++x)
    {
      **x=0.0;
      for (unsigned int i=0; i<(unsigned int)n1; i++){
	(**x).local_element(localProc_dof_indicesReal[i])=(*val).real(); 
	(**x).local_element(localProc_dof_indicesImag[i])=(*val).imag(); 
	val++;
      }
      (**x).update_ghost_values();
    }
#else
  //copy back Xbar to X
  val=Xbar.begin();
  for (std::vector<vectorType*>::iterator x=X.begin(); x<X.end(); ++x)
    {
      **x=0.0;
      for (unsigned int i=0; i<(unsigned int)n1; i++){
	(**x).local_element(i)=*val; val++;
      }
      (**x).update_ghost_values();
    }
#endif

  //set a0 and bLow
  a0[d_kPointIndex]=eigenValues[d_kPointIndex][0]; 
  bLow[d_kPointIndex]=eigenValues[d_kPointIndex].back(); 
  //
  computing_timer.exit_section("Chebyshev Rayleigh Ritz"); 
}

//chebyshev solver
//inputs: X - input wave functions, m-polynomial degree, a-lower bound of unwanted spectrum
//b-upper bound of the full spectrum, a0-lower bound of the wanted spectrum
template<unsigned int FEOrder>
void dftClass<FEOrder>::chebyshevFilter(std::vector<vectorType*> & X, 
					unsigned int m, 
					double a, 
					double b, 
					double a0)
{
  computing_timer.enter_section("Chebyshev filtering"); 
  double e, c, sigma, sigma1, sigma2, gamma;
  e=(b-a)/2.0; c=(b+a)/2.0;
  sigma=e/(a0-c); sigma1=sigma; gamma=2.0/sigma1;
  
  //Y=alpha1*(HX+alpha2*X)
  double alpha1=sigma1/e, alpha2=-c;
  eigen.HX(X, PSI);
  for (std::vector<vectorType*>::iterator y=PSI.begin(), x=X.begin(); y<PSI.end(); ++y, ++x){  
    (**y).add(alpha2,**x);
    (**y)*=alpha1;
  } 
  //loop over polynomial order
  for (unsigned int i=2; i<m+1; i++){
    sigma2=1.0/(gamma-sigma);
    //Ynew=alpha1*(HY-cY)+alpha2*X
    alpha1=2.0*sigma2/e, alpha2=-(sigma*sigma2);
    eigen.HX(PSI, tempPSI);
    for (std::vector<vectorType*>::iterator ynew=tempPSI.begin(), y=PSI.begin(), x=X.begin(); ynew<tempPSI.end(); ++ynew, ++y, ++x){  
      (**ynew).add(-c,**y);
      (**ynew)*=alpha1;
      (**ynew).add(alpha2,**x);
      **x=**y;
      **y=**ynew;
    }
    sigma=sigma2;
  }
  
  //copy back PSI to eigenVectors
  for (std::vector<vectorType*>::iterator y=PSI.begin(), x=X.begin(); y<PSI.end(); ++y, ++x){  
    **x=**y;
  }   
  computing_timer.exit_section("Chebyshev filtering"); 
}
 

