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
// @author Sambit Das


#include<linearAlgebraOperationsInternalCUDA.h>
#include<linearAlgebraOperationsCUDA.h>
#include<dftParameters.h>

namespace dftfe
{
	namespace linearAlgebraOperationsCUDA
	{


		void pseudoGramSchmidtOrthogonalization(operatorDFTCUDAClass & operatorMatrix,
				double * X,
				const unsigned int M,
				const unsigned int N,
				const MPI_Comm &mpiCommDomain,
        GPUCCLWrapper & gpucclMpiCommDomain,
				const MPI_Comm &interBandGroupComm,
				cublasHandle_t &handle,
				const bool useMixedPrecOverall)
		{

			int this_process;
			MPI_Comm_rank(mpiCommDomain, &this_process);

			double gpu_time;

			if (dftParameters::gpuFineGrainedTimings)
			{
				cudaDeviceSynchronize();
				MPI_Barrier(MPI_COMM_WORLD);
				gpu_time = MPI_Wtime();
			}

			const unsigned int rowsBlockSize=operatorMatrix.getScalapackBlockSize();
			std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  processGrid;
			linearAlgebraOperationsCUDA::internal::createProcessGridSquareMatrix(mpiCommDomain,
					N,
					processGrid);

			dealii::ScaLAPACKMatrix<double> overlapMatPar(N,
					processGrid,
					rowsBlockSize);

			if (dftParameters::gpuFineGrainedTimings)
			{
				cudaDeviceSynchronize();
				MPI_Barrier(MPI_COMM_WORLD);
				gpu_time = MPI_Wtime() - gpu_time;
				if (this_process==0)
					std::cout<<"Time for creating processGrid and ScaLAPACK matrix: "<<gpu_time<<std::endl;
			}

			if (dftParameters::gpuFineGrainedTimings)
			{
				cudaDeviceSynchronize();
				MPI_Barrier(MPI_COMM_WORLD);
				gpu_time = MPI_Wtime();
			}

			//S=X*X^{T}. Implemented as S=X^{T}*X with X^{T} stored in the column major format
			if (dftParameters::useMixedPrecPGS_O && useMixedPrecOverall)
			{
				if(dftParameters::overlapComputeCommunOrthoRR)
					linearAlgebraOperationsCUDA::
						fillParallelOverlapMatMixedPrecScalapackAsyncComputeCommun
						(X,
						 M,
						 N,
						 handle,
						 mpiCommDomain,
             gpucclMpiCommDomain,
						 interBandGroupComm,
						 processGrid,
						 overlapMatPar);
				else
					linearAlgebraOperationsCUDA::
						fillParallelOverlapMatMixedPrecScalapack
						(X,
						 M,
						 N,
						 handle,
						 mpiCommDomain,
             gpucclMpiCommDomain,
						 interBandGroupComm,
						 processGrid,
						 overlapMatPar);
			}
			else
			{
				if(dftParameters::overlapComputeCommunOrthoRR)
					linearAlgebraOperationsCUDA::
						fillParallelOverlapMatScalapackAsyncComputeCommun
						(X,
						 M,
						 N,
						 handle,
						 mpiCommDomain,
             gpucclMpiCommDomain,
						 interBandGroupComm,
						 processGrid,
						 overlapMatPar); 
				else
					linearAlgebraOperationsCUDA::
						fillParallelOverlapMatScalapack
						(X,
						 M,
						 N,
						 handle,
						 mpiCommDomain,
             gpucclMpiCommDomain,
						 interBandGroupComm,
						 processGrid,
						 overlapMatPar); 
			}

			if (dftParameters::gpuFineGrainedTimings)
			{
				cudaDeviceSynchronize();
				MPI_Barrier(MPI_COMM_WORLD);
				gpu_time = MPI_Wtime() - gpu_time;
				if (this_process==0)
				{
					if (dftParameters::useMixedPrecPGS_O && useMixedPrecOverall)
						std::cout<<"Time for PGS Fill overlap matrix GPU mixed prec (option 0): "<<gpu_time<<std::endl;
					else
						std::cout<<"Time for PGS Fill overlap matrix (option 0): "<<gpu_time<<std::endl;
				}
			}

			if (dftParameters::gpuFineGrainedTimings)  
			{
				cudaDeviceSynchronize();
				MPI_Barrier(MPI_COMM_WORLD);
				gpu_time = MPI_Wtime(); 
			}

			overlapMatPar.compute_cholesky_factorization();

			dealii::LAPACKSupport::Property overlapMatPropertyPostCholesky=overlapMatPar.get_property();

			AssertThrow(overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::lower_triangular
					||overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::upper_triangular
					,dealii::ExcMessage("DFT-FE Error: overlap matrix property after cholesky factorization incorrect"));

			dealii::ScaLAPACKMatrix<double> LMatPar(N,
					processGrid,
					rowsBlockSize,
					overlapMatPropertyPostCholesky);

			//copy triangular part of projHamPar into LMatPar
			if (processGrid->is_process_active())
				for (unsigned int i = 0; i < overlapMatPar.local_n(); ++i)
				{
					const unsigned int glob_i = overlapMatPar.global_column(i);
					for (unsigned int j = 0; j < overlapMatPar.local_m(); ++j)
					{
						const unsigned int glob_j = overlapMatPar.global_row(j);
						if (overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::lower_triangular)
						{
							if (glob_i <= glob_j)
								LMatPar.local_el(j, i)=overlapMatPar.local_el(j, i);
							else
								LMatPar.local_el(j, i)=0;
						}
						else
						{
							if (glob_j <= glob_i)
								LMatPar.local_el(j, i)=overlapMatPar.local_el(j, i);
							else
								LMatPar.local_el(j, i)=0;
						}
					}
				}


			LMatPar.invert();

			if (dftParameters::gpuFineGrainedTimings)
			{  
				cudaDeviceSynchronize();
				MPI_Barrier(MPI_COMM_WORLD);
				gpu_time = MPI_Wtime() - gpu_time;
				if (this_process==0)
					std::cout<<"Time for PGS Cholesky Triangular Mat inverse ScaLAPACK (option 0): "<<gpu_time<<std::endl;
			}

			//X=X*L^{-1}^{T} implemented as X^{T}=L^{-1}*X^{T} with X^{T} stored in the column major format
			if (dftParameters::gpuFineGrainedTimings)
			{
				cudaDeviceSynchronize();
				MPI_Barrier(MPI_COMM_WORLD);
				gpu_time = MPI_Wtime();
			}

			if (dftParameters::useMixedPrecPGS_SR && useMixedPrecOverall)
				subspaceRotationPGSMixedPrecScalapack(X,
						M,
						N,
						handle,
						processGrid,
						mpiCommDomain,
            gpucclMpiCommDomain,
						interBandGroupComm,
						LMatPar,
						overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::upper_triangular?true:false);
			else
				subspaceRotationScalapack(X,
						M,
						N,
						handle,
						processGrid,
						mpiCommDomain,
            gpucclMpiCommDomain,
						interBandGroupComm,
						LMatPar,
						overlapMatPropertyPostCholesky==dealii::LAPACKSupport::Property::upper_triangular?true:false,
						dftParameters::triMatPGSOpt?true:false);

			if (dftParameters::gpuFineGrainedTimings)
			{
				cudaDeviceSynchronize();
				MPI_Barrier(MPI_COMM_WORLD);
				gpu_time = MPI_Wtime() - gpu_time;
				if (this_process==0)
				{
					if (dftParameters::useMixedPrecPGS_SR && useMixedPrecOverall)
						std::cout<<"Time for PGS subspace rotation GPU mixed prec (option 0): "<<gpu_time<<std::endl;
					else
						std::cout<<"Time for PGS subspace rotation GPU (option 0): "<<gpu_time<<std::endl;
				}
			}
		}
	}
}
