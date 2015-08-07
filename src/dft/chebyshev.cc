
//chebyshev solver
void dftClass::chebyshevSolver(){
  computing_timer.enter_section("Chebyshev filtering"); 
  //compute upper bound of spectrum
  bUp=upperBound();
  /*
  //filter
  chebyshevFilter(PSI, X, m, bLow, bUp, a0);
  //Gram Schmidt orthonormalization
  gramSchmidt(PSI);
  //Rayleigh-Ritz
  eigen.XHX(PSI, H1);
  //convert H1 to lapack vector H2
  //compute the eigen decomposition
  int n=X.size(), lda=X.size();
  std::vector<double> eigenValues(n);
  char jobz = 'V', uplo = 'U';
  int lwork = 1 + 6*(n) + 2*(n)*(n);
  std::vector<double> work(lwork);
  int liwork = 3 + 5*(n);
  std::vector<int> iwork(liwork,0);
  int info;
  dsyevd_(&jobz, &uplo, &n, &H2[0], &lda, &eigenValues[0], &work[0], &lwork, &iwork[0], &liwork, &info);
  //convert PSI to lapack vector PSI2
  //rotate the basis PSI=PSI*Q
  char transA  = 'T', transB  = 'N';
  double alpha = 1.0, beta  = 0.0;
  dgemm_(&transA, &transB, m, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
  */
  computing_timer.exit_section("Chebyshev filtering"); 
}

double dftClass::upperBound(){
  unsigned int lanczosIterations=10;
  double alpha, beta;
  //generate random vector v
  vChebyshev=0.0;
  std::srand(this_mpi_process);
  for (unsigned int i=0; i<vChebyshev.local_size(); i++){
    vChebyshev.local_element(i)=(double(std::rand())/RAND_MAX);
  }
  vChebyshev.compress(); 
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
  std::vector<double> T((lanczosIterations*(lanczosIterations+1))/2,0.0);
  T[0]=alpha;
  unsigned index=1;
  for (unsigned int j=1; j<lanczosIterations; j++){
    beta=fChebyshev.l2_norm();
    v0Chebyshev=vChebyshev; vChebyshev.equ(1.0/beta,fChebyshev);
    eigen.HX(v,f); fChebyshev.add(-1.0*beta,v0Chebyshev);
    alpha=fChebyshev*vChebyshev; fChebyshev.add(-1.0*alpha,vChebyshev);
    T[index]=beta; T[index+lanczosIterations]=alpha;
    index+=lanczosIterations+1;
  }
  //eigen decomposition to find max eigen value of T matrix
  std::vector<double> eigenValuesT(lanczosIterations), work(2*lanczosIterations+1);
  std::vector<int> iwork(1);
  char jobz='N', uplo='L';
  int n=lanczosIterations, lda=lanczosIterations, lwork=2*lanczosIterations+1, liwork=1, info;
  dsyevd_(&jobz, &uplo, &n, &T[0], &lda, &eigenValuesT[0], &work[0], &lwork, &iwork[0], &liwork, &info);
  return (eigenValuesT[lanczosIterations-1]+fChebyshev.l2_norm());
}

/*
void dftClass::gramSchmidt(std::vector<vectorType*>& X){
  std::vector<vectorType>::iterator qend=X.end();
  for (std::vector<vectorType>::iterator x=X.begin(); x<X.end(); ++x){
    for (std::vector<vectorType>::iterator q=X.begin(); q<x; ++q){
      double rii=q*x;
      x-=rii*q;
    }
    x/=x.norm();
  }
}

//chebyshev solver
//inputs: X - input wave functions, m-polynomial degree, a-lower bound of unwanted spectrum
//b-upper bound of the full spectrum, a0-lower bound of the wanted spectrum
void dftClass::chebyshevFilter(std::vector<vectorType>& Y, const std::vector<vectorType>& X, double m, double a, double b, double a0){
  double e, c, sigma, sigma1, sigma2, gamma;
  unsigned int inc=1, n;

  //chebyshev filter implementation
  e=(b-a)/2.0; c=(b+a)/2.0;
  sigma=e/(a0-c); sigma1=sigma; gamma=2.0/sigma1;
  
  //Y=alpha1*HX+alpha2*X
  double alpha1=sigma1/e, alpha2=-(c*sigma1)/e;
  eigen.HX(X, Y, alpha1); //Y=alpha*H*X
  for (std::vector<vectorType>::iterator y=Y.begin(), x=X.begin(); y<Y.end() && x<X.end(); ++y, ++x){
    y+=alpha2*x;
  } 

  //loop over polynomial order
  for (unsigned int i=2; i<m; i++){
    sigma2=1.0/(gamma-sigma);
    //Ynew=alpha1*(HY-cY)+alpha2*X
    alpha1=2.0*sigma2/e, alpha2=-(sigma*sigma2);
    eigen.HX(Y, Ynew, alpha1); //Ynew=alpha*H*Y
    for (std::vector<vectorType>::iterator ynew=Ynew.begin(), y=Y.begin(), x=X.begin(); ynew<Ynew.end() && y<Y.end() && x<X.end(); ++ynew, ++y, ++x){
      ynew+=-alpha1*c*y+alpha2*x;
      x=y; 
      y=ynew;
    }
    sigma=sigma2;
  }
}


//dgemm_(char* transA, char* transB, int *m, int *n, int *k, double *alpha, double *A, int *lda, double *B, int *ldb, double *beta, double *C, int *ldc);
//dscal_(n, alpha1, y, inc);
//daxpy_(n, alpha2, x2, inc, y, inc);
*/
