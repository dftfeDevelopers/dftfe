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
#include "gpuDirectCCLWrapper.h"

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
				const MPI_Comm &mpiCommDomain,
        GPUCCLWrapper & gpucclMpiCommDomain,        
				const MPI_Comm &interBandGroupComm,
				const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				dealii::ScaLAPACKMatrix<double> & overlapMatPar);



		/** @brief Computes Sc=X^{T}*Xc.
		 *
		 *
		 */
		void fillParallelOverlapMatScalapackAsyncComputeCommun(const double* X,
				const unsigned int M,
				const unsigned int N,
				cublasHandle_t &handle,
				const MPI_Comm &mpiCommDomain,
        GPUCCLWrapper & gpucclMpiCommDomain,
				const MPI_Comm &interBandGroupComm,
				const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				dealii::ScaLAPACKMatrix<double> & overlapMatPar);




		/** @brief Computes Sc=X^{T}*Xc.
		 *
		 *
		 */
		void fillParallelOverlapMatMixedPrecScalapackAsyncComputeCommun(const double* X,
				const unsigned int M,
				const unsigned int N,
				cublasHandle_t &handle,
				const MPI_Comm &mpiCommDomain,
        GPUCCLWrapper & gpucclMpiCommDomain,        
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
				const MPI_Comm &mpiCommDomain,
        GPUCCLWrapper & gpucclMpiCommDomain,        
				const MPI_Comm &interBandGroupComm,
				const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				dealii::ScaLAPACKMatrix<double> & overlapMatPar);



		/** @brief PGS orthogonalization
		 */
		void pseudoGramSchmidtOrthogonalization(operatorDFTCUDAClass & operatorMatrix,
				double * X,
				const unsigned int M,
				const unsigned int N,
				const MPI_Comm &mpiCommDomain,
        GPUCCLWrapper & gpucclMpiCommDomain,
				const MPI_Comm &interBandGroupComm,
				cublasHandle_t & handle,
				const bool useMixedPrecOverall=false);

		void subspaceRotationScalapack(double* X,
				const unsigned int M,
				const unsigned int N,
				cublasHandle_t &handle,
				const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				const MPI_Comm &mpiCommDomain,
        GPUCCLWrapper & gpucclMpiCommDomain,        
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
				const MPI_Comm &mpiCommDomain,
        GPUCCLWrapper & gpucclMpiCommDomain,        
				const dealii::ScaLAPACKMatrix<double> & rotationMatPar,
				const bool rotationMatTranspose=false);

		void subspaceRotationPGSMixedPrecScalapack
			(double* X,
			 const unsigned int M,
			 const unsigned int N,
			 cublasHandle_t &handle,
			 const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
			 const MPI_Comm &mpiCommDomain,
      GPUCCLWrapper & gpucclMpiCommDomain,       
			 const MPI_Comm &interBandGroupComm,
			 const dealii::ScaLAPACKMatrix<double> & rotationMatPar,
			 const bool rotationMatTranspose=false);


		void subspaceRotationRRMixedPrecScalapack
			(double* X,
			 const unsigned int M,
			 const unsigned int N,
			 cublasHandle_t &handle,
			 const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
			 const MPI_Comm &mpiCommDomain,
       GPUCCLWrapper & gpucclMpiCommDomain,       
			 const MPI_Comm &interBandGroupComm,
			 const dealii::ScaLAPACKMatrix<double> & rotationMatPar,
			 const bool rotationMatTranspose=false);

		void rayleighRitzSpectrumSplitDirect(operatorDFTCUDAClass & operatorMatrix,
				const double* X,
				double* XFrac,
				distributedGPUVec<double> & Xb,
				distributedGPUVec<float> & floatXb,
				distributedGPUVec<double> & HXb,
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int M,
				const unsigned int N,
				const unsigned int Noc,
				const bool isElpaStep1,
				const bool isElpaStep2,
				const MPI_Comm &mpiCommDomain,
        GPUCCLWrapper & gpucclMpiCommDomain,        
				double* eigenValues,
				cublasHandle_t & handle,
				dealii::ScaLAPACKMatrix<double> & projHamPar,
				const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid> & processGrid,
				const bool useMixedPrecOverall=false);


		void rayleighRitz(operatorDFTCUDAClass & operatorMatrix,
				double* X,
				distributedGPUVec<double> & Xb,
				distributedGPUVec<float> & floatXb,
				distributedGPUVec<double> & HXb,
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int M,
				const unsigned int N,
				const bool isElpaStep1,
				const bool isElpaStep2,
				const MPI_Comm &mpiCommDomain,
        GPUCCLWrapper & gpucclMpiCommDomain,        
				const MPI_Comm &interBandGroupComm,
				double* eigenValues,
				cublasHandle_t & handle,
				dealii::ScaLAPACKMatrix<double> & projHamPar,
				const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid> & processGrid,
				const bool useMixedPrecOverall=false);


		void rayleighRitzGEP(operatorDFTCUDAClass & operatorMatrix,
				double* X,
				distributedGPUVec<double> & Xb,
				distributedGPUVec<float> & floatXb,
				distributedGPUVec<double> & HXb,
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int M,
				const unsigned int N,
				const bool isElpaStep1,
				const bool isElpaStep2,
				const MPI_Comm &mpiCommDomain,
        GPUCCLWrapper & gpucclMpiCommDomain,
				const MPI_Comm &interBandGroupComm,
				double* eigenValues,
				cublasHandle_t & handle,
				dealii::ScaLAPACKMatrix<double> & projHamPar,
				dealii::ScaLAPACKMatrix<double> & overlapMatPar,
				const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid> & processGrid,
				const bool useMixedPrecOverall=false);

		void rayleighRitzGEPSpectrumSplitDirect(operatorDFTCUDAClass & operatorMatrix,
				double* X,
				double* XFrac,
				distributedGPUVec<double> & Xb,
				distributedGPUVec<float> & floatXb,
				distributedGPUVec<double> & HXb,
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int M,
				const unsigned int N,
				const unsigned int Noc,
				const bool isElpaStep1,
				const bool isElpaStep2,
				const MPI_Comm &mpiCommDomain,
        GPUCCLWrapper & gpucclMpiCommDomain,        
				const MPI_Comm &interBandGroupComm,
				double* eigenValues,
				cublasHandle_t & handle,
				dealii::ScaLAPACKMatrix<double> & projHamPar,
				dealii::ScaLAPACKMatrix<double> & overlapMatPar,
				const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid> & processGrid,
				const bool useMixedPrecOverall=false);


		/** @brief Calculates an estimate of lower and upper bounds of a matrix using
		 *  k-step Lanczos method.
		 *
		 *  @param  operatorMatrix An object which has access to the given matrix
		 *  @param  vect A dummy vector
		 *  @return double An estimate of the upper bound of the given matrix
		 */
		std::pair<double,double> lanczosLowerUpperBoundEigenSpectrum(operatorDFTCUDAClass & operatorMatrix,
				const distributedCPUVec<double> & vect,
        distributedGPUVec<double> & Xb,
        distributedGPUVec<double> & Yb,
				distributedGPUVec<double> & projectorKetTimesVector,        
        const unsigned int blockSize); 


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
				distributedGPUVec<double> & X,//thrust::device_vector<dataTypes::number> & X,
				distributedGPUVec<double> & Y,
				distributedGPUVec<float> & Z,
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int localVectorSize,
				const unsigned int numberComponents,
				const unsigned int m,
				const double a,
				const double b,
				const double a0,
				const bool mixedPrecOverall);


		void chebyshevFilter(operatorDFTCUDAClass & operatorMatrix,
				distributedGPUVec<double> & X1,
				distributedGPUVec<double> & Y1,
				distributedGPUVec<float> & Z,
				distributedGPUVec<double> & projectorKetTimesVector1,
				distributedGPUVec<float> & projectorKetTimesVectorFloat,
				distributedGPUVec<double> & X2,
				distributedGPUVec<double> & Y2,
				distributedGPUVec<double> & projectorKetTimesVector2,
				const unsigned int localVectorSize,
				const unsigned int numberComponents,
				const unsigned int m,
				const double a,
				const double b,
				const double a0,
				const bool mixedPrecOverall);

		void computeEigenResidualNorm(operatorDFTCUDAClass        & operatorMatrix,
				double* X,
				distributedGPUVec<double> & Xb,
				distributedGPUVec<double> & HXb,
				distributedGPUVec<double> & projectorKetTimesVector,
				const unsigned int M,
				const unsigned int N,
				const std::vector<double>     & eigenValues,
				const MPI_Comm &mpiCommDomain,
				const MPI_Comm &interBandGroupComm,
				cublasHandle_t & handle,
				std::vector<double> & residualNorm,
				const bool useBandParal=false);
	}
}
#endif
#endif
