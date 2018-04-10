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

/** @file linearAlgebraOperations.h
 *  @brief Contains linear algebra functions used in the implementation of an eigen solver
 *
 *  @author Phani Motamarri (2018)
 */

#include <headers.h>
#include <operator.h>

using namespace dealii;
typedef dealii::parallel::distributed::Vector<double> vectorType;

namespace linearAlgebraOperations
{
  //
  //extern declarations for blas-lapack routines
  //
  extern "C"{
    void dgemv_(char* TRANS, const int* M, const int* N, double* alpha, double* A, const int* LDA, double* X, const int* INCX, double* beta, double* C, const int* INCY);
    void dgesv_( int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, int* info );
    void dscal_(int *n, double *alpha, double *x, int *incx);
    void daxpy_(int *n, double *alpha, double *x, int *incx, double *y, int *incy);
    void dgemm_(const char* transA, const char* transB, const int *m, const int *n, const int *k, const double *alpha, const double *A, const int *lda, const double *B, const int *ldb, const double *beta, double *C, const int *ldc);
    void dsyevd_(char* jobz, char* uplo, int* n, double* A, int *lda, double* w, double* work, int* lwork, int* iwork, int* liwork, int* info);
    void zgemm_(const char* transA, const char* transB, const int *m, const int *n, const int *k, const std::complex<double> *alpha, const std::complex<double> *A, const int *lda, const std::complex<double> *B, const int *ldb, const std::complex<double> *beta, std::complex<double> *C, const int *ldc);
    void zheevd_(char *jobz, char *uplo,int *n,std::complex<double> *A,int *lda,double *w,std::complex<double> *work,int *lwork,double *rwork,int *lrwork,int *iwork,int *liwork,int *info);
    void zdotc_(std::complex<double> *C,int *N,const std::complex<double> *X,int *INCX,const std::complex<double> *Y,int *INCY);
    void zaxpy_(int *n,std::complex<double> *alpha,std::complex<double> *x,int *incx,std::complex<double> *y,int *incy);
}



  
  /** @brief Calculates an estimate of upper bound of a matrix using
   *  k-step Lanczos method.
   *  
   *  @param  operatorMatrix An object which has access to the given matrix
   *  @param  vect A dummy vector  
   *  @return double An estimate of the upper bound of the given matrix
   */
  double lanczosUpperBoundEigenSpectrum(operatorClass * operatorMatrix,
					const vectorType    & vect);


   /** @brief Apply Chebyshev filter to a given subspace
   *  
   *  @param operatorMatrix An object which has access to the given matrix
   *  @param  X Given subspace
   *  @param  m Chebyshev polynomial degree
   *  @param  a lower bound of unwanted spectrum
   *  @param  b upper bound of unwanted spectrum
   *  @param  a0 lower bound of wanted spectrum
   *  @return X In-place update of the given subspace 
   */
  void chebyshevFilter(operatorClass * operatorMatrix,
		       std::vector<vectorType> & X,
		       const unsigned int m,
		       const double a,
		       const double b,
		       const double a0);


  /** @brief Orthogonalize given subspace using GramSchmidt orthogonalization
   *  
   *  @param operatorMatrix An object which has access to the given matrix
   *  @param  X Given subspace
   *
   *  @return X In-place update of the given subspace 
   */
  void gramSchmidtOrthogonalization(operatorClass * operatorMatrix,
				    std::vector<vectorType> & X);



  /** @brief Compute Rayleigh-Ritz projection
   *  
   *  @param operatorMatrix An object which has access to the given matrix
   *  @param  X Given subspace
   *
   *  @return X In-place rotated subspace
   *  @return eigenValues of the Projected Hamiltonian
   */
  void rayleighRitz(operatorClass           * operatorMatrix,
		    std::vector<vectorType> & X,
		    std::vector<double>     & eigenValues);

  

}
