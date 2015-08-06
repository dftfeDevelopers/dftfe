//Include header files
#include "../../include/headers.h"
#include "../../include/dft.h"
#include "../../utils/fileReaders.cc"
#include "../poisson/poisson.cc"
//#include "../eigen/eigen.cc"
 
#define lanczosIterations 10

//dft constructor
dftClass::init_chebyshev(){};

//PSI, X 
//XHX, HX, 

//chebyshev solver
void dftClass::chebyshevSolver(){
  //compute upper bound of spectrum
  bUp=upperBound();
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
}

double dftClass::upperBound(){
  double alpha, beta;
  double T[lanczosIterations][lanczosIterations];
  //generate random vector v (complete this)
  //
  vectorType f, v0;
  eigen.HX(v,f);
  alpha=f*v;
  f-=alpha*v;
  T[0][0]=alpha;
  for (unsigned int j=1; j<std::min(lanczosIterations,10); j++){
    beta=f.norm();
    v0=v; v/=beta;
    eigen.HX(v,f); f-=beta*v0;
    alpha=f*v; f-=alpha*v;
    T[j][j-1]=beta; T[j-1][j]=beta; T[j][j]=alpha;  
  }
  //eigen decomposition to find max eigen value (complete this)
  dsyevd_(&jobz, &uplo, &n, &H2[0], &lda, &eigenValues[0], &work[0], &lwork, &iwork[0], &liwork, &info);
  return (eigenValues[lanczosIterations-1]+f.norm());
}

void dftClass::gramSchmidt(std::vector<vectorType>& X){
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
