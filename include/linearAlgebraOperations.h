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


#ifndef linearAlgebraOperations_h
#define linearAlgebraOperations_h

#include <headers.h>
#include <operator.h>

namespace dftfe
{
    //
    //extern declarations for blas-lapack routines
    //
    extern "C"{
      void dgemv_(char* TRANS, const int* M, const int* N, double* alpha, double* A, const int* LDA, double* X, const int* INCX, double* beta, double* C, const int* INCY);
      void dgesv_( int* n, int* nrhs, double* a, int* lda, int* ipiv, double* b, int* ldb, int* info );
      void dscal_(const unsigned int *n, double *alpha, double *x, const unsigned int *inc);
      void zscal_(const unsigned int *n, std::complex<double> *alpha, std::complex<double> *x, const unsigned int *inc);
      void daxpy_(const unsigned int *n, const double *alpha, double *x, const unsigned int *incx, double *y, const unsigned int *incy);
      void dgemm_(const char* transA, const char* transB, const unsigned int *m, const unsigned int *n, const unsigned int *k, const double *alpha, const double *A, const unsigned int *lda, const double *B, const unsigned int *ldb, const double *beta, double *C, const unsigned int *ldc);
      void dgemm_batch_(const char* transa_array,const char* transb_array,const unsigned int* m_array,const unsigned int* n_array,const unsigned int* k_array,const double* alpha_array,double** a_array,const unsigned int * lda_array,const double ** b_array,const unsigned int * ldb_array,const double * beta_array,double** c_array,const unsigned int * ldc_array,const unsigned int* group_count,const unsigned int* group_size);
      void dsyevd_(const char* jobz, const char* uplo, const unsigned int* n, double* A, const unsigned int *lda, double* w, double* work, const unsigned int* lwork, int* iwork, const unsigned int* liwork, int* info);
      void dsyevr_(const char *jobz, const char *range, const char *uplo,const unsigned int *n, double *A,const unsigned int *lda,const double *vl, const double *vu, const unsigned int *il, const unsigned int *iu, const double *abstol, const unsigned int *m, double *w, double *Z, const unsigned int * ldz, unsigned int * isuppz, double *work, const int *lwork, int * iwork, const int *liwork, int *info);
      void dsyrk_(const char *uplo, const char *trans, const unsigned int *n, const unsigned int *k, const double *alpha, const double *A, const unsigned int *lda, const double *beta, double *C, const unsigned int * ldc);
      void dcopy_(const unsigned int *n,const double *x,const unsigned int *incx,double *y,const unsigned int *incy);
      void zgemm_(const char* transA, const char* transB, const unsigned int *m, const unsigned int *n, const unsigned int *k, const std::complex<double> *alpha, const std::complex<double> *A, const unsigned int *lda, const std::complex<double> *B, const unsigned int *ldb, const std::complex<double> *beta, std::complex<double> *C, const unsigned int *ldc);
      void zgemm_batch_(const char* transa_array,const char* transb_array,const unsigned int* m_array,const unsigned int* n_array,const unsigned int* k_array,const std::complex<double>* alpha_array,std::complex<double>** a_array,const unsigned int * lda_array,const std::complex<double> ** b_array,const unsigned int * ldb_array,const std::complex<double> * beta_array,std::complex<double>** c_array,const unsigned int * ldc_array,const unsigned int* group_count,const unsigned int* group_size);
      void zheevd_(const char *jobz, const char *uplo, const unsigned int *n,std::complex<double> *A,const unsigned int *lda,double *w,std::complex<double> *work, const unsigned int *lwork,double *rwork, const unsigned int *lrwork, int *iwork,const unsigned int *liwork, int *info);
      void zheevr_(const char *jobz, const char *range, const char *uplo,const unsigned int *n,std::complex<double> *A,const unsigned int *lda,const double *vl, const double *vu, const unsigned int *il, const unsigned int *iu, const double *abstol, const unsigned int *m, double *w, std::complex<double> *Z, const unsigned int * ldz, unsigned int * isuppz, std::complex<double> *work, const int *lwork, double *rwork, const int *lrwork, int * iwork, const int *liwork, int *info);
      void zherk_(const char *uplo, const char *trans, const unsigned int *n, const unsigned int *k, const double *alpha, const std::complex<double> *A, const unsigned int *lda, const double *beta, std::complex<double> *C, const unsigned int * ldc);
      void zcopy_(const unsigned int *n, const std::complex<double> *x, const unsigned int *incx, std::complex<double> *y, const unsigned int *incy);
      void zdotc_(std::complex<double> *C,const int *N,const std::complex<double> *X,const int *INCX,const std::complex<double> *Y,const int *INCY);
      void zaxpy_(const unsigned int *n,const std::complex<double> *alpha,std::complex<double> *x,const unsigned int *incx,std::complex<double> *y,const unsigned int *incy);
    }

/** @file linearAlgebraOperations.h
 *  @brief Contains linear algebra functions used in the implementation of an eigen solver
 *
 *  @author Phani Motamarri, Sambit Das
 */
  namespace linearAlgebraOperations
  {

    /** @brief Calculates an estimate of upper bound of a matrix using
     *  k-step Lanczos method.
     *
     *  @param  operatorMatrix An object which has access to the given matrix
     *  @param  vect A dummy vector
     *  @return double An estimate of the upper bound of the given matrix
     */
    double lanczosUpperBoundEigenSpectrum(operatorDFTClass & operatorMatrix,
					  const vectorType & vect);


    /** @brief Apply Chebyshev filter to a given subspace
     *
     *  @param operatorMatrix An object which has access to the given matrix
     *  @param  X Given subspace as STL vector of dealii vectors
     *  @param  m Chebyshev polynomial degree
     *  @param  a lower bound of unwanted spectrum
     *  @param  b upper bound of unwanted spectrum
     *  @param  a0 lower bound of wanted spectrum
     *  @return X In-place update of the given subspace
     */
    void chebyshevFilter(operatorDFTClass & operatorMatrix,
			 std::vector<vectorType> & X,
			 const unsigned int m,
			 const double a,
			 const double b,
			 const double a0);


    /** @brief Apply Chebyshev filter to a given subspace
     *
     *  @param operatorMatrix An object which has access to the given matrix
     *  @param  X Given subspace as a dealii array representing multiple fields as a flattened array
     *  @param  numberComponents Number of multiple-fields
     *  @param  m Chebyshev polynomial degree
     *  @param  a lower bound of unwanted spectrum
     *  @param  b upper bound of unwanted spectrum
     *  @param  a0 lower bound of wanted spectrum
     *  @return X In-place update of the given subspace
     */
    template<typename T>
    void chebyshevFilter(operatorDFTClass & operatorMatrix,
			 dealii::parallel::distributed::Vector<T> & X,
			 const unsigned int numberComponents,
			 const unsigned int m,
			 const double a,
			 const double b,
			 const double a0);



    /** @brief Orthogonalize given subspace using GramSchmidt orthogonalization
     *
     *  @param operatorMatrix An object which has access to the given matrix
     *  @param  X Given subspace as vector of dealii column vectors
     *  @param startingIndex dealii column vector index to start the orthogonalization procedure
     *
     *  @return X In-place update of the given subspace
     */
    void gramSchmidtOrthogonalization(operatorDFTClass & operatorMatrix,
				      std::vector<vectorType> & X,
				      unsigned int startingIndex = 0);



     /** @brief Orthogonalize given subspace using GramSchmidt orthogonalization
     *
     *  @param  X Given subspace as flattened array of multi-vectors
     *  @param numberComponents Number of multiple-fields
     *  @return X In-place update of the given subspace
     */
    template<typename T>
      void gramSchmidtOrthogonalization(dealii::parallel::distributed::Vector<T> & X,
					const unsigned int numberComponents);


    /** @brief Orthogonalize given subspace using Lowden orthogonalization for double data-type
     *  (serial version using LAPACK)
     *
     *  @param  X Given subspace as flattened array of multi-vectors
     *  @param numberComponents Number of multiple-fields
     *  @return X In-place update of the given subspace
     */
    void lowdenOrthogonalization(dealii::parallel::distributed::Vector<dataTypes::number> & X,
				 const unsigned int numberComponents);


     /** @brief Orthogonalize given subspace using Pseudo-Gram-Schmidt orthogonalization
      *
      *  This is a wrapper to the serial and parallel versions. The parallel version is
      *  called if dealii is linked to ScaLAPACK, otherwise the serial version is called.
      *
      *  @param  X Given subspace as flattened array of multi-vectors
      *  @param numberComponents Number of multiple-fields
      *  @return X In-place update of the given subspace
      */
    template<typename T>
      void pseudoGramSchmidtOrthogonalization(dealii::parallel::distributed::Vector<T> & X,
					      const unsigned int numberComponents);

     /** @brief Orthogonalize given subspace using Pseudo-Gram-Schmidt orthogonalization
      *  (serial version using LAPACK)
      *
      *  @param  X Given subspace as flattened array of multi-vectors
      *  @param numberComponents Number of multiple-fields
      *  @return X In-place update of the given subspace
      */
    template<typename T>
      void pseudoGramSchmidtOrthogonalizationSerial(dealii::parallel::distributed::Vector<T> & X,
					            const unsigned int numberComponents);


     /** @brief Orthogonalize given subspace using Pseudo-Gram-Schmidt orthogonalization
      *  (parallel version using ScaLAPACK)
      *
      *  @param  X Given subspace as flattened array of multi-vectors
      *  @param numberComponents Number of multiple-fields
      *  @return X In-place update of the given subspace
      */
    template<typename T>
      void pseudoGramSchmidtOrthogonalizationParallel(dealii::parallel::distributed::Vector<T> & X,
					              const unsigned int numberComponents);


    /** @brief Compute Rayleigh-Ritz projection
     *
     *  @param operatorMatrix An object which has access to the given matrix
     *  @param  X Given subspace as STL vector dealii vectors
     *
     *  @return X In-place rotated subspace
     *  @return eigenValues of the Projected Hamiltonian
     */
    void rayleighRitz(operatorDFTClass        & operatorMatrix,
		      std::vector<vectorType> & X,
		      std::vector<double>     & eigenValues);



    /** @brief Compute Rayleigh-Ritz projection
     *
     *  @param operatorMatrix An object which has access to the given matrix
     *  @param  X Given subspace as flattened array of multi-vectors
     *  @param  numberComponents Number of multiple-fields
     *  @param  macroCellMap precomputed cell-localindex id map of the multi-wavefuncton field in the order of macrocells
     *  @param  cellMap precomputed cell-localindex id map of the multi-wavefuncton field in the order of local active cells
     *
     *  @return X In-place rotated subspace
     *  @return eigenValues of the Projected Hamiltonian
     */
    template<typename T>
    void rayleighRitz(operatorDFTClass        & operatorMatrix,
		      dealii::parallel::distributed::Vector<T> & X,
		      const unsigned int numberComponents,
		      std::vector<double>     & eigenValues);


     /** @brief Compute Compute residual norm associated with eigenValue problem of the given operator
     *
     *  @param operatorMatrix An object which has access to the given matrix
     *  @param  X Given subspace as STL vector of dealii vectors
     *  @param  eigenValues eigenValues of the operator
     *
     *  @return residualNorms of the eigen Value problem
     */
    void computeEigenResidualNorm(operatorDFTClass        & operatorMatrix,
				  std::vector<vectorType> & X,
				  const std::vector<double>     & eigenValues,
				  std::vector<double> & residualNorm);


     /** @brief Compute residual norm associated with eigenValue problem of the given operator
     *
     *  @param operatorMatrix An object which has access to the given matrix
     *  @param  X Given eigenvector subspace as flattened array of multi-vectors
     *  @param  eigenValues eigenValues of the operator
     *  @param  macroCellMap precomputed cell-localindex id map of the multi-wavefuncton field in the order of macrocells
     *  @param  cellMap precomputed cell-localindex id map of the multi-wavefuncton field in the order of local active cells
     *
     *  @return residualNorms of the eigen Value problem
     */
    template<typename T>
    void computeEigenResidualNorm(operatorDFTClass        & operatorMatrix,
				  dealii::parallel::distributed::Vector<T> & X,
				  const std::vector<double> & eigenValues,
				  std::vector<double>     & residualNorm);





  }

}
#endif
