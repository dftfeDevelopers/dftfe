// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018  The Regents of the University of Michigan and DFT-FE authors.
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

#if defined(DFTFE_WITH_GPU)
#ifndef linearAlgebraOperationsCUDA_h
#define linearAlgebraOperationsCUDA_h

#include <headers.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include <operatorCUDA.h>

namespace dftfe
{

  extern "C"
  {
    void dsyevd_(const char* jobz, const char* uplo, const unsigned int* n, double* A, const unsigned int *lda, double* w, double* work, const unsigned int* lwork, int* iwork, const unsigned int* liwork, int* info);
  }

  /**
   *  @brief Contains functions for linear algebra operations on GPU
   *
   *  @author Sambit Das
   */
  namespace linearAlgebraOperationsCUDA
  {

    /** @brief Computes Sc=X^{T}*Xc.
     *
     *
     */
    void fillParallelOverlapMatScalapack(const double* X,
					 const unsigned int M,
					 const unsigned int N,
					 cublasHandle_t &handle,
					 const MPI_Comm &mpiComm,
                                         const MPI_Comm &interBandGroupComm,
					 const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
					 dealii::ScaLAPACKMatrix<double> & overlapMatPar);


    /** @brief Computes Sc=X^{T}*Xc.
     *
     *
     */
    void fillParallelOverlapMatMixedPrecScalapack(const double* X,
					 const unsigned int M,
					 const unsigned int N,
					 cublasHandle_t &handle,
					 const MPI_Comm &mpiComm,
                                         const MPI_Comm &interBandGroupComm,
					 const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
					 dealii::ScaLAPACKMatrix<double> & overlapMatPar);



    /** @brief PGS orthogonalization
     */
    void pseudoGramSchmidtOrthogonalization(operatorDFTCUDAClass & operatorMatrix,
                                            double * X,
					    const unsigned int M,
					    const unsigned int N,
					    const MPI_Comm &mpiComm,
                                            const MPI_Comm &interBandGroupComm,
					    cublasHandle_t & handle,
                                            const bool useMixedPrecOverall=false);
                              
    void subspaceRotationScalapack(double* X,
				   const unsigned int M,
				   const unsigned int N,
				   cublasHandle_t &handle,
				   const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				   const MPI_Comm &mpiComm,
                                   const MPI_Comm &interBandGroupComm,
				   const dealii::ScaLAPACKMatrix<double> & rotationMatPar,
				   const bool rotationMatTranspose=false,
				   const bool isRotationMatLowerTria=false);


    void subspaceRotationSpectrumSplitScalapack(const double* X,
                                   double * XFrac,
				   const unsigned int M,
				   const unsigned int N,
                                   const unsigned int Nfr,
				   cublasHandle_t &handle,
				   const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				   const MPI_Comm &mpiComm,
				   const dealii::ScaLAPACKMatrix<double> & rotationMatPar,
				   const bool rotationMatTranspose=false);

    void subspaceRotationPGSMixedPrecScalapack
                                  (double* X,
				   const unsigned int M,
				   const unsigned int N,
				   cublasHandle_t &handle,
				   const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				   const MPI_Comm &mpiComm,
                                   const MPI_Comm &interBandGroupComm,
				   const dealii::ScaLAPACKMatrix<double> & rotationMatPar,
				   const bool rotationMatTranspose=false);


    void subspaceRotationRRMixedPrecScalapack
                                  (double* X,
				   const unsigned int M,
				   const unsigned int N,
				   cublasHandle_t &handle,
				   const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				   const MPI_Comm &mpiComm,
                                   const MPI_Comm &interBandGroupComm,
				   const dealii::ScaLAPACKMatrix<double> & rotationMatPar,
				   const bool rotationMatTranspose=false);

    void rayleighRitzSpectrumSplitDirect(operatorDFTCUDAClass & operatorMatrix,
		      const double* X,
                      double* XFrac,
                      cudaVectorType & Xb,
                      cudaVectorType & HXb,
                      cudaVectorType & projectorKetTimesVector,
		      const unsigned int M,
		      const unsigned int N,
                      const unsigned int Noc,
                      const bool isElpaStep1,
                      const bool isElpaStep2,
		      const MPI_Comm &mpiComm,
		      double* eigenValues,
		      cublasHandle_t & handle,
                      dealii::ScaLAPACKMatrix<double> & projHamPar,
                      const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid> & processGrid,
                      const bool useMixedPrecOverall=false);


    void rayleighRitz(operatorDFTCUDAClass & operatorMatrix,
		      double* X,
                      cudaVectorType & Xb,
                      cudaVectorType & HXb,
                      cudaVectorType & projectorKetTimesVector,
		      const unsigned int M,
		      const unsigned int N,
                      const bool isElpaStep1,
                      const bool isElpaStep2,
		      const MPI_Comm &mpiComm,
                      const MPI_Comm &interBandGroupComm,
		      double* eigenValues,
		      cublasHandle_t & handle,
                      dealii::ScaLAPACKMatrix<double> & projHamPar,
                      const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid> & processGrid,
                      const unsigned int gpuLinalgOption=0,
                      const bool useMixedPrecOverall=false);


    void rayleighRitzGEP(operatorDFTCUDAClass & operatorMatrix,
		      double* X,
                      cudaVectorType & Xb,
                      cudaVectorType & HXb,
                      cudaVectorType & projectorKetTimesVector,
		      const unsigned int M,
		      const unsigned int N,
                      const bool isElpaStep1,
                      const bool isElpaStep2,
		      const MPI_Comm &mpiComm,
                      const MPI_Comm &interBandGroupComm,
		      double* eigenValues,
		      cublasHandle_t & handle,
                      dealii::ScaLAPACKMatrix<double> & projHamPar,
                      dealii::ScaLAPACKMatrix<double> & overlapMatPar,
                      const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid> & processGrid,
                      const unsigned int gpuLinalgOption=0,
                      const bool useMixedPrecOverall=false);


    /** @brief Calculates an estimate of upper bound of a matrix using
     *  k-step Lanczos method.
     *
     *  @param  operatorMatrix An object which has access to the given matrix
     *  @param  vect A dummy vector
     *  @return double An estimate of the upper bound of the given matrix
     */
    double lanczosUpperBoundEigenSpectrum(operatorDFTCUDAClass & operatorMatrix,
					  const vectorType & vect); 


    /** @brief Apply Chebyshev filter to a given subspace
     *
     *  @param[in] operatorMatrix An object which has access to the given matrix
     *  @param[in,out]  X Given subspace as a dealii array representing multiple fields
     *  as a flattened array. In-place update of the given subspace.
     *  @param[in]  numberComponents Number of multiple-fields
     *  @param[in]  m Chebyshev polynomial degree
     *  @param[in]  a lower bound of unwanted spectrum
     *  @param[in]  b upper bound of unwanted spectrum
     *  @param[in]  a0 lower bound of wanted spectrum
     */
    void chebyshevFilter(operatorDFTCUDAClass & operatorMatrix,
			 cudaVectorType & X,//thrust::device_vector<dataTypes::number> & X,
                         cudaVectorType & Y,
                         cudaVectorTypeFloat & Z,
                         cudaVectorType & projectorKetTimesVector,
			 const unsigned int localVectorSize,
			 const unsigned int numberComponents,
			 const unsigned int m,
			 const double a,
			 const double b,
			 const double a0);

    void computeEigenResidualNorm(operatorDFTCUDAClass        & operatorMatrix,
			          double* X,
			          cudaVectorType & Xb,
			          cudaVectorType & HXb,
			          cudaVectorType & projectorKetTimesVector,
			          const unsigned int M,
			          const unsigned int N,
				  const std::vector<double>     & eigenValues,
				  const MPI_Comm &mpiComm,
                                  const MPI_Comm &interBandGroupComm,
                                  cublasHandle_t & handle,
				  std::vector<double> & residualNorm,
                                  const bool useBandParal=false);
  }
}
#endif
#endif
