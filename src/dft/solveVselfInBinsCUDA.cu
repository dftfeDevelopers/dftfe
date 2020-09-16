// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2020 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Sambit Das
//

#ifdef DFTFE_WITH_GPU
#include <solveVselfInBinsCUDA.h>
#include <vectorUtilities.h>
#include <dftParameters.h>

namespace dftfe
{

	namespace poissonCUDA
	{
		namespace
		{
			__global__
				void diagScaleKernel(const unsigned int blockSize,
						const unsigned int numContiguousBlocks,
						const double *srcArray,
						const double *scalingVector,
						double * dstArray)
				{ 

					const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;

					for(unsigned int index = globalThreadId; index < numContiguousBlocks*blockSize; index+= blockDim.x*gridDim.x)
					{
						const unsigned int blockIndex=index/blockSize;
						*(dstArray+index) = *(srcArray+index) *(*(scalingVector+blockIndex));
					}


				}

			__global__
				void dotProductContributionBlockedKernel(const unsigned int numEntries,
						const double *vec1,
						const double *vec2,
						double * vecTemp)
				{ 

					const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;

					for(unsigned int index = globalThreadId; index < numEntries; index+= blockDim.x*gridDim.x)
					{
						vecTemp[index]=vec1[index]*vec2[index];
					}


				}

			__global__
				void scaleBlockedKernel(const unsigned int blockSize,
						const unsigned int numContiguousBlocks,
						double *xArray,
						const double *scalingVector)
				{ 

					const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;

					for(unsigned int index = globalThreadId; index < numContiguousBlocks*blockSize; index+= blockDim.x*gridDim.x)
					{
						const unsigned int intraBlockIndex=index%blockSize;
						*(xArray+index) *= (*(scalingVector+intraBlockIndex));
					}


				}

			__global__
				void scaleKernel(const unsigned int numEntries,
						double *xArray,
						const double *scalingVector)
				{ 

					const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;

					for(unsigned int index = globalThreadId; index < numEntries; index+= blockDim.x*gridDim.x)
					{
						xArray[index] *= scalingVector[index];
					}


				}

			//y=alpha*x+y
			__global__
				void daxpyBlockedKernel(const unsigned int blockSize,
						const unsigned int numContiguousBlocks,
						const double *x,
						const double *alpha,
						double * y)
				{ 

					const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;

					for(unsigned int index = globalThreadId; index < numContiguousBlocks*blockSize; index+= blockDim.x*gridDim.x)
					{
						const unsigned int blockIndex=index/blockSize;
						const unsigned int intraBlockIndex=index-blockIndex*blockSize;
						y[index] += alpha[intraBlockIndex]*x[index];
					}


				}


			//y=-alpha*x+y
			__global__
				void dmaxpyBlockedKernel(const unsigned int blockSize,
						const unsigned int numContiguousBlocks,
						const double *x,
						const double *alpha,
						double * y)
				{ 

					const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;

					for(unsigned int index = globalThreadId; index < numContiguousBlocks*blockSize; index+= blockDim.x*gridDim.x)
					{
						const unsigned int blockIndex=index/blockSize;
						const unsigned int intraBlockIndex=index-blockIndex*blockSize;
						y[index] += -alpha[intraBlockIndex]*x[index];
					}


				}

#if __CUDA_ARCH__ < 600
			__device__ double atomicAdd(double* address, double val)
			{
				unsigned long long int* address_as_ull =
					(unsigned long long int*)address;
				unsigned long long int old = *address_as_ull, assumed;

				do {
					assumed = old;
					old = atomicCAS(address_as_ull, assumed,
							__double_as_longlong(val +
								__longlong_as_double(assumed)));

					// Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
				} while (assumed != old);

				return __longlong_as_double(old);
			}
#endif


			__global__
				void daxpyAtomicAddKernel(const unsigned int contiguousBlockSize,
						const unsigned int numContiguousBlocks,
						const double *addFromVec,
						double *addToVec,
						const dealii::types::global_dof_index *addToVecStartingContiguousBlockIds)
				{

					const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
					const unsigned int numberEntries = numContiguousBlocks*contiguousBlockSize;

					for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
					{
						unsigned int blockIndex = index/contiguousBlockSize;
						unsigned int intraBlockIndex=index%contiguousBlockSize;
						atomicAdd(&addToVec[addToVecStartingContiguousBlockIds[blockIndex]+intraBlockIndex], addFromVec[index]);
					}

				}


			__global__
				void copyCUDAKernel(const unsigned int contiguousBlockSize,
						const unsigned int numContiguousBlocks,
						const double *copyFromVec,
						double *copyToVec,
						const dealii::types::global_dof_index *copyFromVecStartingContiguousBlockIds)
				{

					const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
					const unsigned int numberEntries = numContiguousBlocks*contiguousBlockSize;

					for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
					{
						unsigned int blockIndex = index/contiguousBlockSize;
						unsigned int intraBlockIndex=index%contiguousBlockSize;
						copyToVec[index]
							=copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex]+intraBlockIndex];
					}

				}

			void computeAX(cublasHandle_t &handle,
					dftUtils::constraintMatrixInfoCUDA & constraintsMatrixDataInfoCUDA,
					distributedGPUVec<double> & src,
					distributedGPUVec<double> & temp,
					const unsigned int totalLocallyOwnedCells,
					const unsigned int numberNodesPerElement,
					const unsigned int numberVectors,
					const unsigned int localSize,
					const unsigned int ghostSize,
					const thrust::device_vector<double> & poissonCellStiffnessMatricesD,
					const thrust::device_vector<double> & inhomoIdsColoredVecFlattenedD,
					const thrust::device_vector<dealii::types::global_dof_index> & cellLocalProcIndexIdMapD,
					distributedGPUVec<double> & dst,
					thrust::device_vector<double> & cellNodalVectorD,
					thrust::device_vector<double> & cellStiffnessMatrixTimesVectorD)
			{
				//const unsigned int numberVectors = 1;
				//thrust::fill(dst.begin(),dst.end(),0.0);
				dst=0.0;

				//distributedGPUVec<double> temp;
				//temp.reinit(src);
				//temp=src;
				cudaMemcpy(temp.begin(),
						src.begin(),
						localSize*numberVectors*sizeof(double),
						cudaMemcpyDeviceToDevice);                                     

				//src.update_ghost_values();  
				//constraintsMatrixDataInfoCUDA.distribute(src,numberVectors);
				temp.update_ghost_values(); 
				constraintsMatrixDataInfoCUDA.distribute(temp,numberVectors);

				scaleKernel<<<(numberVectors+255)/256*(localSize+ghostSize),256>>>(numberVectors*(localSize+ghostSize),
						temp.begin(),
						thrust::raw_pointer_cast(&inhomoIdsColoredVecFlattenedD[0]));

				//
				//elemental matrix-multiplication
				//
				const double scalarCoeffAlpha = 1.0/(4.0*M_PI), scalarCoeffBeta = 0.0;

				copyCUDAKernel<<<(numberVectors+255)/256*totalLocallyOwnedCells*numberNodesPerElement,256>>>(numberVectors,
						totalLocallyOwnedCells*numberNodesPerElement,
						temp.begin(),//src.begin(),
						thrust::raw_pointer_cast(&cellNodalVectorD[0]),
						thrust::raw_pointer_cast(&cellLocalProcIndexIdMapD[0]));  



				const unsigned int strideA = numberNodesPerElement*numberVectors;
				const unsigned int strideB = numberNodesPerElement*numberNodesPerElement; 
				const unsigned int strideC = numberNodesPerElement*numberVectors;

				//
				//do matrix-matrix multiplication
				//
				cublasDgemmStridedBatched(handle,
						CUBLAS_OP_N,
						CUBLAS_OP_N,
						numberVectors,
						numberNodesPerElement,
						numberNodesPerElement,
						&scalarCoeffAlpha,
						thrust::raw_pointer_cast(&cellNodalVectorD[0]),
						numberVectors,
						strideA,
						thrust::raw_pointer_cast(&poissonCellStiffnessMatricesD[0]),
						numberNodesPerElement,
						strideB,
						&scalarCoeffBeta,
						thrust::raw_pointer_cast(&cellStiffnessMatrixTimesVectorD[0]),
						numberVectors,
						strideC,
						totalLocallyOwnedCells);


				daxpyAtomicAddKernel<<<(numberVectors+255)/256*totalLocallyOwnedCells*numberNodesPerElement,256>>>(numberVectors,
						totalLocallyOwnedCells*numberNodesPerElement,
						thrust::raw_pointer_cast(&cellStiffnessMatrixTimesVectorD[0]),
						dst.begin(),
						thrust::raw_pointer_cast(&cellLocalProcIndexIdMapD[0]));

				// think dirichlet hanging node linked to two master solved nodes 
				scaleKernel<<<(numberVectors+255)/256*(localSize+ghostSize),256>>>(numberVectors*(localSize+ghostSize),
						dst.begin(),
						thrust::raw_pointer_cast(&inhomoIdsColoredVecFlattenedD[0]));


				constraintsMatrixDataInfoCUDA.distribute_slave_to_master(dst,numberVectors);

				dst.compress(dealii::VectorOperation::add);


				scaleKernel<<<(numberVectors+255)/256*localSize,256>>>(numberVectors*localSize,
						dst.begin(),
						thrust::raw_pointer_cast(&inhomoIdsColoredVecFlattenedD[0]));

				//src.zero_out_ghosts();
				//constraintsMatrixDataInfoCUDA.set_zero(src,numberVectors);
			}

			void precondition_Jacobi(const double * src,
					const double * diagonalA,
					const unsigned int numberVectors,
					const unsigned int localSize,
					double * dst)
			{

				diagScaleKernel<<<(numberVectors+255)/256*localSize,256>>>(numberVectors,
						localSize,
						src,
						diagonalA,
						dst);
			}

			void computeResidualSq(cublasHandle_t &handle,
					const double * vec1,
					const double * vec2,
					double * vecTemp,
					const double * onesVec,
					const unsigned int numberVectors,
					const unsigned int localSize,
					double * residualNormSq)
			{
				dotProductContributionBlockedKernel<<<(numberVectors+255)/256*localSize,256>>>(numberVectors*localSize,
						vec1,
						vec2,
						vecTemp);

				const double alpha = 1.0, beta = 0.0;
				cublasDgemm(handle,
						CUBLAS_OP_N,
						CUBLAS_OP_T,
						1,
						numberVectors,
						localSize,
						&alpha,
						onesVec,
						1,
						vecTemp,
						numberVectors,
						&beta,
						residualNormSq,
						1);
			}
		}

		void solveVselfInBins(operatorDFTCUDAClass & operatorMatrix,
				const dealii::MatrixFree<3,double> & matrixFreeData,
        const unsigned int mfDofHandlerIndex,
				const dealii::AffineConstraints<double> & hangingPeriodicConstraintMatrix,
				const double * bH,
				const double * diagonalAH,
				const double * inhomoIdsColoredVecFlattenedH,
				const unsigned int localSize,
				const unsigned int ghostSize,
				const unsigned int numberBins, 
				const MPI_Comm & mpiComm, 
				double * xH)
		{
			int this_process;
			MPI_Comm_rank(MPI_COMM_WORLD, &this_process);

			const unsigned int blockSize = numberBins;
			const unsigned int totalLocallyOwnedCells=matrixFreeData.n_physical_cells();
			const unsigned int numberNodesPerElement=matrixFreeData.get_dofs_per_cell(mfDofHandlerIndex);

			distributedGPUVec<double> xD;

			MPI_Barrier(MPI_COMM_WORLD);
			double time = MPI_Wtime(); 

			vectorTools::createDealiiVector(matrixFreeData.get_vector_partitioner(mfDofHandlerIndex),
					blockSize,
					xD);
			xD=0.0;

			MPI_Barrier(MPI_COMM_WORLD);
			time = MPI_Wtime() - time;
			if (dftParameters::verbosity >= 2 && this_process==0)
				std::cout<<" poissonCUDA::solveVselfInBins: time for creating xD: "<<time<<std::endl;

			std::vector<dealii::types::global_dof_index> cellLocalProcIndexIdMapH;
			vectorTools::computeCellLocalIndexSetMap(xD.get_partitioner(),
					matrixFreeData,
          mfDofHandlerIndex,
					blockSize,
					cellLocalProcIndexIdMapH);

			dftUtils::constraintMatrixInfoCUDA constraintsMatrixDataInfoCUDA;
			constraintsMatrixDataInfoCUDA.initialize(matrixFreeData.get_vector_partitioner(mfDofHandlerIndex),
					hangingPeriodicConstraintMatrix);


			constraintsMatrixDataInfoCUDA.precomputeMaps(matrixFreeData.get_vector_partitioner(mfDofHandlerIndex),
					xD.get_partitioner(),
					blockSize);

			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			time = MPI_Wtime(); 

			thrust::device_vector<double> bD(localSize*numberBins,0.0);
			thrust::device_vector<double> diagonalAD(localSize,0.0);
			thrust::device_vector<double> inhomoIdsColoredVecFlattenedD((localSize+ghostSize)*numberBins,0.0);
			thrust::device_vector<dealii::types::global_dof_index> cellLocalProcIndexIdMapD(totalLocallyOwnedCells*numberNodesPerElement);

			cudaMemcpy(thrust::raw_pointer_cast(&bD[0]),
					bH,
					localSize*numberBins*sizeof(double),
					cudaMemcpyHostToDevice);

			cudaMemcpy(thrust::raw_pointer_cast(&diagonalAD[0]),
					diagonalAH,
					localSize*sizeof(double),
					cudaMemcpyHostToDevice);

			cudaMemcpy(thrust::raw_pointer_cast(&inhomoIdsColoredVecFlattenedD[0]),
					inhomoIdsColoredVecFlattenedH,
					(localSize+ghostSize)*numberBins*sizeof(double),
					cudaMemcpyHostToDevice);


			cudaMemcpy(thrust::raw_pointer_cast(&cellLocalProcIndexIdMapD[0]),
					&cellLocalProcIndexIdMapH[0],
					totalLocallyOwnedCells*numberNodesPerElement*sizeof(dealii::types::global_dof_index),
					cudaMemcpyHostToDevice);

			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			time = MPI_Wtime() - time;
			if (dftParameters::verbosity >= 2 && this_process==0)
				std::cout<<" poissonCUDA::solveVselfInBins: time for mem allocation: "<<time<<std::endl;

			cgSolver(operatorMatrix.getCublasHandle(),
					constraintsMatrixDataInfoCUDA,
					thrust::raw_pointer_cast(&bD[0]),
					thrust::raw_pointer_cast(&diagonalAD[0]),
					operatorMatrix.getShapeFunctionGradientIntegralElectro(),
					inhomoIdsColoredVecFlattenedD,
					cellLocalProcIndexIdMapD,
					localSize,
					ghostSize,
					numberBins,
					totalLocallyOwnedCells,
					numberNodesPerElement,
					dftParameters::verbosity,
					dftParameters::maxLinearSolverIterations,
					dftParameters::absLinearSolverTolerance,  
					mpiComm,
					xD);

			cudaMemcpy(xH,
					xD.begin(),
					localSize*numberBins*sizeof(double),
					cudaMemcpyDeviceToHost);
		}

		void cgSolver(cublasHandle_t &handle,
				dftUtils::constraintMatrixInfoCUDA & constraintsMatrixDataInfoCUDA,
				const double * bD,
				const double * diagonalAD,
				const thrust::device_vector<double> & poissonCellStiffnessMatricesD,
				const thrust::device_vector<double> & inhomoIdsColoredVecFlattenedD,
				const thrust::device_vector<dealii::types::global_dof_index> & cellLocalProcIndexIdMapD,
				const unsigned int localSize,
				const unsigned int ghostSize,
				const unsigned int numberBins,
				const unsigned int totalLocallyOwnedCells,
				const unsigned int numberNodesPerElement,
				const unsigned int debugLevel,
				const unsigned int maxIter,
				const double absTol,  
				const MPI_Comm & mpiComm,
				distributedGPUVec<double> & x)
		{
			int this_process;
			MPI_Comm_rank(MPI_COMM_WORLD, &this_process);

			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			double start_time = MPI_Wtime(); 

			//initialize certain variables
			const double negOne = -1.0;
			//const double posOne = 1.0;
			const unsigned int inc = 1;

			thrust::device_vector<double> delta_newD(numberBins,0.0);
			thrust::device_vector<double> delta_oldD(numberBins,0.0);
			thrust::device_vector<double> delta_0D(numberBins,0.0);
			thrust::device_vector<double> alphaD(numberBins,0.0);
			thrust::device_vector<double> betaD(numberBins,0.0);
			thrust::device_vector<double> scalarD(numberBins,0.0);
			thrust::device_vector<double> residualNormSqD(numberBins,0.0);
			thrust::device_vector<double> negOneD(numberBins,-1.0);
			thrust::device_vector<double> posOneD(numberBins,1.0);
			thrust::device_vector<double> vecTempD(localSize*numberBins,1.0);
			thrust::device_vector<double> onesVecD(localSize,1.0);
			thrust::device_vector<double>  cellNodalVectorD(totalLocallyOwnedCells*numberNodesPerElement*numberBins);
			thrust::device_vector<double>  cellStiffnessMatrixTimesVectorD(totalLocallyOwnedCells*numberNodesPerElement*numberBins);

			std::vector<double> delta_newH(numberBins,0.0);
			std::vector<double> delta_oldH(numberBins,0.0);
			std::vector<double> alphaH(numberBins,0.0);
			std::vector<double> betaH(numberBins,0.0);
			std::vector<double> scalarH(numberBins,0.0);
			std::vector<double> residualNormSqH(numberBins,0.0);

			//compute RHS b
			//thrust::device_vector<double> b;

			//double start_timeRhs = MPI_Wtime(); 
			//problem.computeRhs(b);
			//double end_timeRhs = MPI_Wtime() - start_timeRhs;

			//if(debugLevel >= 2)
			// std::cout<<" Time for Poisson problem compute rhs: "<<end_timeRhs<<std::endl;

			//get size of vectors
			//unsigned int localSize = b.size();


			//get access to initial guess for solving Ax=b
			//thrust::device_vector<double> & x = problem.getX();
			//x.update_ghost_values();


			//compute Ax
			//thrust::device_vector<double> Ax;
			//Ax.resize(localSize,0.0);
			distributedGPUVec<double> Ax;
			Ax.reinit(x);
			//computeAX(x,Ax);

			distributedGPUVec<double> r;
			r.reinit(x);

			distributedGPUVec<double> q,s;
			q.reinit(x);
			s.reinit(x);

			distributedGPUVec<double> d, temp;
			d.reinit(x);
			temp.reinit(x);

			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			double gpu_time = MPI_Wtime() - start_time;
			if (debugLevel >= 2 && this_process==0)
				std::cout<<" poissonCUDA::solveVselfInBins: time for GPU CG solver memory allocation: "<<gpu_time<<std::endl;

			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			start_time = MPI_Wtime(); 

			computeAX(handle,
					constraintsMatrixDataInfoCUDA,
					x,
					temp,
					totalLocallyOwnedCells,
					numberNodesPerElement,
					numberBins,
					localSize,
					ghostSize,
					poissonCellStiffnessMatricesD,
					inhomoIdsColoredVecFlattenedD,
					cellLocalProcIndexIdMapD,
					Ax,
					cellNodalVectorD,
					cellStiffnessMatrixTimesVectorD);


			//compute residue r = b - Ax
			//thrust::device_vector<double> r;
			//r.resize(localSize,0.0);

			//r = b
			cublasDcopy(handle,
					localSize*numberBins,
					bD,
					inc,
					r.begin(),
					inc);


			//r = b - Ax i.e r - Ax
			cublasDaxpy(handle,
					localSize*numberBins,
					&negOne,
					Ax.begin(),
					inc,
					r.begin(),
					inc);


			//precondition r
			//thrust::device_vector<double> d;
			//d.resize(localSize,0.0);

			//precondition_Jacobi(r,d);
			precondition_Jacobi(r.begin(),
					diagonalAD,
					numberBins,
					localSize,
					d.begin());

			//compute delta_new delta_new = r*d;
			/*
			   cublasDdot(handle,
			   localSize*numberBins,
			   thrust::raw_pointer_cast(&r[0]),
			   inc,
			   thrust::raw_pointer_cast(&d[0]),
			   inc,
			   &delta_new);
			 */

			computeResidualSq(handle,
					r.begin(),
					d.begin(),
					thrust::raw_pointer_cast(&vecTempD[0]),
					thrust::raw_pointer_cast(&onesVecD[0]),
					numberBins,
					localSize,
					thrust::raw_pointer_cast(&delta_newD[0]));

			cudaMemcpy(&delta_newH[0],
					thrust::raw_pointer_cast(&delta_newD[0]),
					numberBins*sizeof(double),
					cudaMemcpyDeviceToHost);


			MPI_Allreduce(MPI_IN_PLACE,
					&delta_newH[0],
					numberBins,
					MPI_DOUBLE,
					MPI_SUM,
					mpiComm);

			cudaMemcpy(thrust::raw_pointer_cast(&delta_newD[0]),
					&delta_newH[0],
					numberBins*sizeof(double),
					cudaMemcpyHostToDevice);

			//assign delta0 to delta_new
			delta_0D = delta_newD;

			//allocate memory for q
			//thrust::device_vector<double> q,s;
			//q.resize(localSize,0.0);
			//s.resize(localSize,0.0);

			unsigned int iterationNumber = 0;

			/*
			   cublasDdot(d_cublasHandle,
			   localSize,
			   thrust::raw_pointer_cast(&r[0]),
			   inc,
			   thrust::raw_pointer_cast(&r[0]),
			   inc,
			   &residualNorm);
			 */
			computeResidualSq(handle,
					r.begin(),
					r.begin(),
					thrust::raw_pointer_cast(&vecTempD[0]),
					thrust::raw_pointer_cast(&onesVecD[0]),
					numberBins,
					localSize,
					thrust::raw_pointer_cast(&residualNormSqD[0]));

			cudaMemcpy(&residualNormSqH[0],
					thrust::raw_pointer_cast(&residualNormSqD[0]),
					numberBins*sizeof(double),
					cudaMemcpyDeviceToHost);


			MPI_Allreduce(MPI_IN_PLACE,
					&residualNormSqH[0],
					numberBins,
					MPI_DOUBLE,
					MPI_SUM,
					mpiComm);

			if(debugLevel >= 2 && this_process==0)
			{
				for (unsigned int i=0;i <numberBins; i++)
					std::cout<< "GPU based Linear Conjugate Gradient solver for bin: "<<i<< " started with residual norm squared: "<<residualNormSqH[i]<<std::endl;

			}

			for(unsigned int iter = 0; iter < maxIter; ++iter)
			{
				//q = Ad
				//computeAX(d,q);


				computeAX(handle,
						constraintsMatrixDataInfoCUDA,
						d,
						temp,
						totalLocallyOwnedCells,
						numberNodesPerElement,
						numberBins,
						localSize,
						ghostSize,
						poissonCellStiffnessMatricesD,
						inhomoIdsColoredVecFlattenedD,
						cellLocalProcIndexIdMapD,
						q,
						cellNodalVectorD,
						cellStiffnessMatrixTimesVectorD);

				//compute alpha
				//double scalar;
				/*
				   cublasDdot(d_cublasHandle,
				   localSize,
				   thrust::raw_pointer_cast(&d[0]),
				   inc,
				   thrust::raw_pointer_cast(&q[0]),
				   inc,
				   &scalar);
				 */
				computeResidualSq(handle,
						d.begin(),
						q.begin(),
						thrust::raw_pointer_cast(&vecTempD[0]),
						thrust::raw_pointer_cast(&onesVecD[0]),
						numberBins,
						localSize,
						thrust::raw_pointer_cast(&scalarD[0]));

				cudaMemcpy(&scalarH[0],
						thrust::raw_pointer_cast(&scalarD[0]),
						numberBins*sizeof(double),
						cudaMemcpyDeviceToHost);


				MPI_Allreduce(MPI_IN_PLACE,
						&scalarH[0],
						numberBins,
						MPI_DOUBLE,
						MPI_SUM,
						mpiComm);

				//for (unsigned int i=0;i <numberBins; i++)
				//   std::cout<< "scalar "<<scalarH[i]<<std::endl;

				for (unsigned int i=0;i <numberBins; i++)
					alphaH[i] = delta_newH[i]/scalarH[i];

				//for (unsigned int i=0;i <numberBins; i++)
				//   std::cout<< "alpha "<<alphaH[i]<<std::endl;

				cudaMemcpy(thrust::raw_pointer_cast(&alphaD[0]),
						&alphaH[0],
						numberBins*sizeof(double),
						cudaMemcpyHostToDevice);

				//update x; x = x + alpha*d
				/*
				   cublasDaxpy(d_cublasHandle,
				   localSize,
				   &alpha,
				   thrust::raw_pointer_cast(&d[0]),
				   inc,
				   thrust::raw_pointer_cast(&x[0]),
				   inc);
				 */
				daxpyBlockedKernel<<<(numberBins+255)/256*localSize,256>>>(numberBins,
						localSize,
						d.begin(),
						thrust::raw_pointer_cast(&alphaD[0]),
						x.begin());

				if(iter%50 == 0)
				{
					//r = b
					cublasDcopy(handle,
							localSize*numberBins,
							bD,
							inc,
							r.begin(),
							inc);

					//computeAX(x,Ax);

					computeAX(handle,
							constraintsMatrixDataInfoCUDA,
							x,
							temp,
							totalLocallyOwnedCells,
							numberNodesPerElement,
							numberBins,
							localSize,
							ghostSize,
							poissonCellStiffnessMatricesD,
							inhomoIdsColoredVecFlattenedD,
							cellLocalProcIndexIdMapD,
							Ax,
							cellNodalVectorD,
							cellStiffnessMatrixTimesVectorD);
					/*
					   cublasDaxpy(d_cublasHandle,
					   localSize,
					   &negOne,
					   thrust::raw_pointer_cast(&Ax[0]),
					   inc,
					   thrust::raw_pointer_cast(&r[0]),
					   inc);
					 */
					daxpyBlockedKernel<<<(numberBins+255)/256*localSize,256>>>(numberBins,
							localSize,
							Ax.begin(),
							thrust::raw_pointer_cast(&negOneD[0]),
							r.begin());
				}
				else
				{
					//negAlphaD = -alpha;
					/*
					   cublasDaxpy(d_cublasHandle,
					   localSize,
					   &negAlpha,
					   thrust::raw_pointer_cast(&q[0]),
					   inc,
					   thrust::raw_pointer_cast(&r[0]),
					   inc);
					 */
					dmaxpyBlockedKernel<<<(numberBins+255)/256*localSize,256>>>(numberBins,
							localSize,
							q.begin(),
							thrust::raw_pointer_cast(&alphaD[0]),
							r.begin());
				}

				//precondition_Jacobi(r,s);
				precondition_Jacobi(r.begin(),
						diagonalAD,
						numberBins,
						localSize,
						s.begin());

				delta_oldD = delta_newD;

				cudaMemcpy(&delta_oldH[0],
						thrust::raw_pointer_cast(&delta_oldD[0]),
						numberBins*sizeof(double),
						cudaMemcpyDeviceToHost);


				//delta_new = r*s;
				/*
				   cublasDdot(d_cublasHandle,
				   localSize,
				   thrust::raw_pointer_cast(&r[0]),
				   inc,
				   thrust::raw_pointer_cast(&s[0]),
				   inc,
				   &delta_new);
				 */

				computeResidualSq(handle,
						r.begin(),
						s.begin(),
						thrust::raw_pointer_cast(&vecTempD[0]),
						thrust::raw_pointer_cast(&onesVecD[0]),
						numberBins,
						localSize,
						thrust::raw_pointer_cast(&delta_newD[0]));

				//beta = delta_new/delta_old;


				cudaMemcpy(&delta_newH[0],
						thrust::raw_pointer_cast(&delta_newD[0]),
						numberBins*sizeof(double),
						cudaMemcpyDeviceToHost);


				MPI_Allreduce(MPI_IN_PLACE,
						&delta_newH[0],
						numberBins,
						MPI_DOUBLE,
						MPI_SUM,
						mpiComm);


				//for (unsigned int i=0;i <numberBins; i++)
				//   std::cout<< "delta_new "<<delta_newH[i]<<std::endl;

				for (unsigned int i=0;i <numberBins; i++)
					betaH[i] = delta_newH[i]/delta_oldH[i];

				cudaMemcpy(thrust::raw_pointer_cast(&betaD[0]),
						&betaH[0],
						numberBins*sizeof(double),
						cudaMemcpyHostToDevice);

				cudaMemcpy(thrust::raw_pointer_cast(&delta_newD[0]),
						&delta_newH[0],
						numberBins*sizeof(double),
						cudaMemcpyHostToDevice);

				//d *= beta;
				/*
				   cublasDscal(handle,
				   localSize,
				   &beta,
				   thrust::raw_pointer_cast(&d[0]),
				   inc);
				 */
				scaleBlockedKernel<<<(numberBins+255)/256*localSize,256>>>(numberBins,
						localSize,
						d.begin(),
						thrust::raw_pointer_cast(&betaD[0]));

				//d.add(1.0,s);
				/*
				   cublasDaxpy(handle,
				   localSize*numberBins,
				   &posOne,
				   s.begin(),
				   inc,
				   d.begin(),
				   inc);
				 */
				daxpyBlockedKernel<<<(numberBins+255)/256*localSize,256>>>(numberBins,
						localSize,
						s.begin(),
						thrust::raw_pointer_cast(&posOneD[0]),
						d.begin());
				unsigned int isBreak = 1;
				//if(delta_new < relTolerance*relTolerance*delta_0)
				//  isBreak = 1;

				for (unsigned int i=0;i <numberBins; i++)             
					if(delta_newH[i] > absTol*absTol)
						isBreak = 0;

				if(isBreak == 1)
					break;

				iterationNumber += 1;

			}



			//compute residual norm at end
			/*
			   cublasDdot(handle,
			   localSize,
			   thrust::raw_pointer_cast(&r[0]),
			   inc,
			   thrust::raw_pointer_cast(&r[0]),
			   inc,
			   &residualNorm);
			 */

			computeResidualSq(handle,
					r.begin(),
					r.begin(),
					thrust::raw_pointer_cast(&vecTempD[0]),
					thrust::raw_pointer_cast(&onesVecD[0]),
					numberBins,
					localSize,
					thrust::raw_pointer_cast(&residualNormSqD[0]));

			cudaMemcpy(&residualNormSqH[0],
					thrust::raw_pointer_cast(&residualNormSqD[0]),
					numberBins*sizeof(double),
					cudaMemcpyDeviceToHost);

			MPI_Allreduce(MPI_IN_PLACE,
					&residualNormSqH[0],
					numberBins,
					MPI_DOUBLE,
					MPI_SUM,
					mpiComm);

			//residualNorm = std::sqrt(residualNorm);

			//
			// set error condition
			//
			unsigned int solveStatus = 1;

			if(iterationNumber == maxIter)
				solveStatus = 0;


			if(debugLevel >= 2 && this_process==0)
			{
				if(solveStatus == 1)
				{
					for (unsigned int i=0;i <numberBins; i++)
						std::cout<< "Linear Conjugate Gradient solver for bin: "<<i<< " converged after "
							<<iterationNumber+1 <<" iterations. with residual norm squared "<<residualNormSqH[i]<<std::endl;
				}
				else
				{
					for (unsigned int i=0;i <numberBins; i++)
						std::cout<< "Linear Conjugate Gradient solver for bin: "<<i<< " failed to converge after "
							<<iterationNumber <<" iterations. with residual norm squared "<<residualNormSqH[i]<<std::endl;
				}
			}


			//problem.setX();
			x.update_ghost_values();
			constraintsMatrixDataInfoCUDA.distribute(x,
					numberBins);
			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime() - start_time;
			if (debugLevel >= 2 && this_process==0)
				std::cout<<" poissonCUDA::solveVselfInBins: time for Poisson problem iterations: "<<gpu_time<<std::endl;

		}

	} 
}
#endif
