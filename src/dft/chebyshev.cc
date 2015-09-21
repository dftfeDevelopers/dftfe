//chebyshev solver
void dftClass::chebyshevSolver(){
  //compute upper bound of spectrum
  bUp=upperBound();
  char buffer[100];
  sprintf(buffer, "bUp: %18.10e\n", bUp);
  pcout << buffer << std::endl;
  pcout << "bLow: " << bLow << std::endl;
  pcout << "a0: " << a0 << std::endl;
  //filter
  for (unsigned int i=0; i<eigenVectors.size(); i++){
    pcout << i << " norm: " << eigenVectors[i]->l2_norm() << "  linf norm: " << eigenVectors[i]->linfty_norm()<< std::endl;
  }
  chebyshevFilter(eigenVectors, chebyshevOrder, bLow, bUp, a0);
  for (unsigned int i=0; i<eigenVectors.size(); i++){
    pcout << i << " norm: " << eigenVectors[i]->l2_norm() << "  linf norm: " << eigenVectors[i]->linfty_norm()<< std::endl;
  }
  //Gram Schmidt orthonormalization
  gramSchmidt(eigenVectors);
  pcout << "afterGS  eigen1 norm: " << eigenVectors[1]->l2_norm() << " linf norm: " << eigenVectors[1] ->linfty_norm()<< std::endl;
  //Rayleigh Ritz step
  rayleighRitz(eigenVectors);
  pcout << "afterRR  eigen1 norm: " << eigenVectors[1]->l2_norm() << " linf norm: " << eigenVectors[1] ->linfty_norm()<< std::endl;
}

double dftClass::upperBound(){
  unsigned int lanczosIterations=10;
  double alpha, beta;
  //generate random vector v
  vChebyshev=0.0;
  std::srand(this_mpi_process);
  for (unsigned int i=0; i<vChebyshev.local_size(); i++){
    vChebyshev.local_element(i)=1.0; //(double(std::rand())/RAND_MAX);
  }
  vChebyshev.update_ghost_values();
  vChebyshev/=vChebyshev.l2_norm();
  //
  std::vector<vectorType*> v,f; 
  v.push_back(&vChebyshev);
  f.push_back(&fChebyshev);
  eigen.HX(v,f);
  //
  alpha=fChebyshev*vChebyshev;
  fChebyshev.add(-1.0*alpha,vChebyshev);
  std::vector<double> T(lanczosIterations*lanczosIterations,0.0); 
  T[0]=alpha;
  unsigned index=0;
  //filling only lower trangular part
  for (unsigned int j=1; j<lanczosIterations; j++){
    beta=fChebyshev.l2_norm();
    v0Chebyshev=vChebyshev; vChebyshev.equ(1.0/beta,fChebyshev);
    eigen.HX(v,f); fChebyshev.add(-1.0*beta,v0Chebyshev);
    alpha=fChebyshev*vChebyshev; fChebyshev.add(-1.0*alpha,vChebyshev);
    index+=1;
    T[index]=beta; 
    index+=lanczosIterations;
    T[index]=alpha;
  }
  //eigen decomposition to find max eigen value of T matrix
  std::vector<double> eigenValuesT(lanczosIterations), work(2*lanczosIterations+1);
  std::vector<int> iwork(10);
  char jobz='N', uplo='L';
  int n=lanczosIterations, lda=lanczosIterations, lwork=2*lanczosIterations+1, liwork=10, info;
  dsyevd_(&jobz, &uplo, &n, &T[0], &lda, &eigenValuesT[0], &work[0], &lwork, &iwork[0], &liwork, &info);
  return (eigenValuesT[lanczosIterations-1]+fChebyshev.l2_norm());
}

//Gram-Schmidt orthonormalization
void dftClass::gramSchmidt(std::vector<vectorType*>& X){
  computing_timer.enter_section("Chebyshev GS orthonormalization"); 
  for (std::vector<vectorType*>::iterator x=X.begin(); x<X.end(); ++x){
    std::vector<double> r(x-X.begin(),0.0);
    unsigned int i=0;
    for (std::vector<vectorType*>::iterator q=X.begin(); q<x; ++q, ++i){
      r[i]=(**q)*(**x);
    }
    i=0;
    for (std::vector<vectorType*>::iterator q=X.begin(); q<x; ++q, ++i){
      (**x).add(-r[i],**q);
    }
    (**x)/=(**x).l2_norm();
  }
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
  for (unsigned int i=0; i<n; i++){
    pcout << "eigen value " << i << ":" << eigenValues[i] << std::endl; 
  }

  //rotate the basis PSI=PSI*Q
  n=X[0]->local_size(); int m=X.size(); 
  std::vector<double> Xbar(n*m), Xlocal(n*m); //Xbar=Xlocal*Q
  std::vector<double>::iterator val=Xlocal.begin();
  for (std::vector<vectorType*>::iterator x=X.begin(); x<X.end(); ++x){
    for (unsigned int i=0; i<n; i++){
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
    for (unsigned int i=0; i<n; i++){
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
 

