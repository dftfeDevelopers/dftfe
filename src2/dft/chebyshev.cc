//chebyshev solver
void dftClass::chebyshevSolver(){
  computing_timer.enter_section("Chebyshev solve"); 
  //compute upper bound of spectrum
  bUp=upperBound();
  char buffer[100];
  sprintf(buffer, "bUp: %18.10e\n", bUp);
  pcout << buffer;
  pcout << "bLow: " << bLow << std::endl;
  pcout << "a0: " << a0 << std::endl;
  //filter
  for (unsigned int i=0; i<eigenVectors.size(); i++){
    sprintf(buffer, "%2u l2: %18.10e     linf: %18.10e \n", i, eigenVectors[i]->l2_norm(), eigenVectors[i]->linfty_norm());
    //pcout << buffer; 
  }
  double t=MPI_Wtime();
  chebyshevFilter(eigenVectors, chebyshevOrder, bLow, bUp, a0);
  pcout << "Total time for only chebyshev filter: " << (MPI_Wtime()-t)/60.0 << "mins\n";
  for (unsigned int i=0; i<eigenVectors.size(); i++){
    sprintf(buffer, "%2u l2: %18.10e     linf: %18.10e \n", i, eigenVectors[i]->l2_norm(), eigenVectors[i]->linfty_norm());
    //pcout << buffer; 
  }
  //Gram Schmidt orthonormalization
  gramSchmidt(eigenVectors);
  //Rayleigh Ritz step
  rayleighRitz(eigenVectors);
  pcout << "Total time for chebyshev filter: " << (MPI_Wtime()-t)/60.0 << "mins\n";
  computing_timer.exit_section("Chebyshev solve"); 
}

double dftClass::upperBound(){
  computing_timer.enter_section("Chebyshev upper bound"); 
  unsigned int lanczosIterations=10;
  double alpha, beta;
  //generate random vector v
  vChebyshev=0.0;
  std::srand(this_mpi_process);
  const unsigned int local_size=vChebyshev.local_size();
  std::vector<unsigned int> local_dof_indices(local_size);
  vChebyshev.locally_owned_elements().fill_index_vector(local_dof_indices);
  std::vector<double> local_values(local_size, 0.0);
  for (unsigned int i=0; i<local_size; i++) local_values[i]= 1.0;//((double)std::rand())/((double)RAND_MAX);
  constraintsNone.distribute_local_to_global(local_values, local_dof_indices, vChebyshev);
  //
  vChebyshev/=vChebyshev.l2_norm();
  vChebyshev.update_ghost_values();
  //
  std::vector<vectorType*> v,f; 
  v.push_back(&vChebyshev);
  f.push_back(&fChebyshev);
  eigen.HX(v,f);
  char buffer2[100];
  sprintf(buffer2, "v: %18.10e,  f: %18.10e\n", vChebyshev.l1_norm(), fChebyshev.l2_norm());
  //pcout << buffer2;
  //
  alpha=fChebyshev*vChebyshev;
  fChebyshev.add(-1.0*alpha,vChebyshev);
  std::vector<double> T(lanczosIterations*lanczosIterations,0.0); 
  T[0]=alpha;
  unsigned index=0;
  //filling only lower trangular part
  for (unsigned int j=1; j<lanczosIterations; j++){
    beta=fChebyshev.l2_norm();
    char buffer1[100];
    sprintf(buffer1, "alpha: %18.10e,  beta: %18.10e\n", alpha, beta);
    //pcout << buffer1;
    v0Chebyshev=vChebyshev; vChebyshev.equ(1.0/beta,fChebyshev);
    eigen.HX(v,f); fChebyshev.add(-1.0*beta,v0Chebyshev);
    alpha=fChebyshev*vChebyshev; fChebyshev.add(-1.0*alpha,vChebyshev);
    index+=1;
    T[index]=beta; 
    index+=lanczosIterations;
    T[index]=alpha;
    sprintf(buffer1, "alpha: %18.10e,  beta: %18.10e\n", alpha, beta);
    //pcout << buffer1;
  }
  //eigen decomposition to find max eigen value of T matrix
  std::vector<double> eigenValuesT(lanczosIterations);
  char jobz='N', uplo='L';
  int n=lanczosIterations, lda=lanczosIterations, info;
  int lwork = 1 + 6*n + 2*n*n, liwork = 3 + 5*n;
  std::vector<int> iwork(liwork, 0);
  std::vector<double> work(lwork, 0.0);
  dsyevd_(&jobz, &uplo, &n, &T[0], &lda, &eigenValuesT[0], &work[0], &lwork, &iwork[0], &liwork, &info);

  //
  computing_timer.exit_section("Chebyshev upper bound");
  for (unsigned int i=0; i<eigenValuesT.size(); i++){eigenValuesT[i]=std::abs(eigenValuesT[i]);}
  std::sort(eigenValuesT.begin(),eigenValuesT.end()); 
  //
  char buffer[100];
  sprintf(buffer, "bUp1: %18.10e,  bUp2: %18.10e\n", eigenValuesT[lanczosIterations-1], fChebyshev.l2_norm());
  //pcout << buffer;
  //
  return (eigenValuesT[lanczosIterations-1]+fChebyshev.l2_norm());
}

//Gram-Schmidt orthonormalization
void dftClass::gramSchmidt(std::vector<vectorType*>& X){
  computing_timer.enter_section("Chebyshev GS orthonormalization"); 
  /*
  //Classical GS with reorthonormalization
  //Ref: The Loss of Orthogonality in the Gram-Schmidt Orthogonalization Process, Computers and Mathematics with Applications 50 (2005)
  //j loop
  for (std::vector<vectorType*>::iterator x=X.begin(); x<X.end(); ++x){
    aj[0]=**x;
    //i loop
    for (unsigned int i=1; i<5; i++){
      aj[i]=aj[i-1];
      //k loop
      for (std::vector<vectorType*>::iterator q=X.begin(); q<x; ++q){
	double r=(**q)*aj[i]; //*aj[i-1];
	aj[i].add(-r,**q);	
      }
    }
    (**x)=aj[4];
    (**x)/=(**x).l2_norm();
  }
  */
  //copy to petsc vectors
  unsigned int localSize=vChebyshev.local_size(), numVectors=X.size();
  Vec vec;
  VecCreateMPI(PETSC_COMM_WORLD, localSize, PETSC_DETERMINE, &vec);
  VecSetFromOptions(vec);
  //
  Vec *petscColumnSpace;
  VecDuplicateVecs(vec, numVectors, &petscColumnSpace);
  VecDestroy(&vec);
  //
  PetscScalar ** columnSpacePointer;
  VecGetArrays(petscColumnSpace, numVectors, &columnSpacePointer);
  for (int i = 0; i < numVectors; ++i){
    std::vector<double> localData(localSize);
    std::copy (X[i]->begin(),X[i]->end(),localData.begin());
    std::copy (localData.begin(),localData.end(), &(columnSpacePointer[i][0])); 
  }
  VecRestoreArrays(petscColumnSpace, numVectors, &columnSpacePointer);
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
  VecGetArrays(petscColumnSpace, numVectors, &columnSpacePointer);
  for (int i = 0; i < numVectors; ++i){
    std::vector<double> localData(localSize);
    std::copy (&(columnSpacePointer[i][0]),&(columnSpacePointer[i][localSize]), localData.begin()); 
    std::copy (localData.begin(), localData.end(), X[i]->begin());
    X[i]->update_ghost_values();
  }
  VecRestoreArrays(petscColumnSpace, numVectors, &columnSpacePointer);
  //
  VecDestroyVecs(numVectors, &petscColumnSpace);
  //
  computing_timer.exit_section("Chebyshev GS orthonormalization"); 
}

void dftClass::rayleighRitz(std::vector<vectorType*>& X){
  computing_timer.enter_section("Chebyshev Rayleigh Ritz"); 
  //Hbar=Psi^T*H*Psi
  eigen.XHX(X);  //Hbar is now available as a 1D array XHXValue 

  //compute the eigen decomposition of Hbar
  int n=X.size(), lda=X.size(), info;
  int lwork = 1 + 6*n + 2*n*n, liwork = 3 + 5*n;
  std::vector<double> work(lwork);
  std::vector<int> iwork(liwork,0);
  char jobz='V', uplo='U';
  dsyevd_(&jobz, &uplo, &n, &eigen.XHXValue[0], &lda, &eigenValues[0], &work[0], &lwork, &iwork[0], &liwork, &info);

  //print eigen values
  char buffer[100];
  for (unsigned int i=0; i< (unsigned int)n; i++){
    sprintf(buffer, "eigen value %2u: %18.10e\n", i, eigenValues[i]);
    pcout << buffer;
  }

  //rotate the basis PSI=PSI*Q
  n=X[0]->local_size(); int m=X.size(); 
  std::vector<double> Xbar(n*m), Xlocal(n*m); //Xbar=Xlocal*Q
  std::vector<double>::iterator val=Xlocal.begin();
  for (std::vector<vectorType*>::iterator x=X.begin(); x<X.end(); ++x){
    for (unsigned int i=0; i<(unsigned int)n; i++){
      *val=(**x).local_element(i); val++;
    }
  }
  char transA  = 'N', transB  = 'N';
  double alpha = 1.0, beta  = 0.0;
  lda=n; int ldb=m, ldc=n;
  dgemm_(&transA, &transB, &n, &m, &m, &alpha, &Xlocal[0], &lda, &eigen.XHXValue[0], &ldb, &beta, &Xbar[0], &ldc);
 
  //copy back Xbar to X
  val=Xbar.begin();
  for (std::vector<vectorType*>::iterator x=X.begin(); x<X.end(); ++x){
    **x=0.0;
    for (unsigned int i=0; i<(unsigned int)n; i++){
      (**x).local_element(i)=*val; val++;
    }
    (**x).update_ghost_values();
  }

  //set a0 and bLow
  a0=eigenValues[0]; 
  bLow=eigenValues.back(); 
  //
  computing_timer.exit_section("Chebyshev Rayleigh Ritz"); 
}

//chebyshev solver
//inputs: X - input wave functions, m-polynomial degree, a-lower bound of unwanted spectrum
//b-upper bound of the full spectrum, a0-lower bound of the wanted spectrum
void dftClass::chebyshevFilter(std::vector<vectorType*>& X, unsigned int m, double a, double b, double a0){
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
 

