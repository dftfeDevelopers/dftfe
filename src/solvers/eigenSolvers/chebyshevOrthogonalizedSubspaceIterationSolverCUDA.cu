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
//
// @author Phani Motamarri, Sambit Das

#include <chebyshevOrthogonalizedSubspaceIterationSolverCUDA.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsCUDA.h>
#include <linearAlgebraOperationsInternalCUDA.h>
#include <vectorUtilities.h>
#include <dftUtils.h>
#include <dftParameters.h>

static const unsigned int order_lookup[][2] = {
	{500, 24}, // <= 500 ~> chebyshevOrder = 24
	{750, 30},
	{1000, 39},
	{1500, 50},
	{2000, 53},
	{3000, 57},
	{4000, 62},
	{5000, 69},
	{9000, 77},
	{14000, 104},
	{20000, 119},
	{30000, 162},
	{50000, 300},
	{80000, 450},
	{100000, 550},
	{200000, 700},
	{500000, 1000}
};

namespace dftfe
{


	namespace 
	{
		__global__
			void stridedCopyToBlockKernel(const unsigned int BVec, 
					const unsigned int M, 
					const double *xVec,
					const unsigned int N,
					double *yVec,
					const unsigned int startingXVecId)
			{
				const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
				const unsigned int numGangsPerBVec
					=(BVec+blockDim.x-1)/blockDim.x;
				const unsigned int gangBlockId=blockIdx.x/numGangsPerBVec;
				const unsigned int localThreadId=globalThreadId-gangBlockId*numGangsPerBVec*blockDim.x;

				if (globalThreadId<M*numGangsPerBVec*blockDim.x && localThreadId<BVec)
				{
					*(yVec+gangBlockId*BVec+localThreadId)=*(xVec+gangBlockId*N+startingXVecId+localThreadId); 
				}
			}

		__global__
			void stridedCopyFromBlockKernel(const unsigned int BVec, 
					const unsigned int M, 
					const double *xVec,
					const unsigned int N,
					double *yVec,
					const unsigned int startingXVecId)
			{
				const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
				const unsigned int numGangsPerBVec
					=(BVec+blockDim.x-1)/blockDim.x;
				const unsigned int gangBlockId=blockIdx.x/numGangsPerBVec;
				const unsigned int localThreadId=globalThreadId-gangBlockId*numGangsPerBVec*blockDim.x;

				if (globalThreadId<M*numGangsPerBVec*blockDim.x && localThreadId<BVec)
				{
					*(yVec+gangBlockId*N+startingXVecId+localThreadId) = *(xVec+gangBlockId*BVec+localThreadId);
				}
			}


		__global__
			void scaleCUDAKernel(const unsigned int contiguousBlockSize,
					const unsigned int numContiguousBlocks,
					const double scalar,
					double *srcArray,
					const double *scalingVector)
			{

				const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
				const unsigned int numGangsPerContiguousBlock = (contiguousBlockSize + (blockDim.x-1))/blockDim.x;
				const unsigned int gangBlockId = blockIdx.x/numGangsPerContiguousBlock;
				const unsigned int localThreadId = globalThreadId-gangBlockId*numGangsPerContiguousBlock*blockDim.x;
				if(globalThreadId < numContiguousBlocks*numGangsPerContiguousBlock*blockDim.x && localThreadId < contiguousBlockSize)
				{
					*(srcArray+(localThreadId+gangBlockId*contiguousBlockSize)) = *(srcArray+(localThreadId+gangBlockId*contiguousBlockSize)) * (*(scalingVector+gangBlockId)*scalar); 

				}

			}

		__global__
			void setZeroKernel(const unsigned int BVec, 
					const unsigned int M, 
					const unsigned int N,
					double *yVec,
					const unsigned int startingXVecId)
			{
				const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
				const unsigned int numGangsPerBVec
					=(BVec+blockDim.x-1)/blockDim.x;
				const unsigned int gangBlockId=blockIdx.x/numGangsPerBVec;
				const unsigned int localThreadId=globalThreadId-gangBlockId*numGangsPerBVec*blockDim.x;

				if (globalThreadId<M*numGangsPerBVec*blockDim.x && localThreadId<BVec)
				{
					*(yVec+gangBlockId*N+startingXVecId+localThreadId)=0.0;  
				}
			}

		__global__
			void convDoubleArrToFloatArr(const unsigned int size,
					const double *doubleArr,
					float *floatArr)
			{

				const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;

				for(unsigned int index = globalThreadId; index < size; index+= blockDim.x*gridDim.x)
				{
					floatArr[index]
						=doubleArr[index];//__double2float_rd(doubleArr[index]);
				}

			}

		__global__
			void computeDiagQTimesXKernel(const double *diagValues,
					double *X,
					const unsigned int N,
					const unsigned int M)
			{
				const unsigned int numEntries=N*M;        
				for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
						i < numEntries; 
						i += blockDim.x * gridDim.x) 
				{
					const unsigned int idof = i/N;
					const unsigned int ivec=  i%N; 

					*(X+N*idof+ivec)
						= *(X+N*idof+ivec)*diagValues[ivec];
				}

			}

		__global__
			void addSubspaceRotatedBlockToXKernel(const unsigned int BDof,
					const unsigned int BVec,
					const float *rotatedXBlockSP,
					double *X,
					const unsigned int startingDofId,
					const unsigned int startingVecId,
					const unsigned int N)
			{

				const unsigned int numEntries=BVec*BDof;        
				for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
						i < numEntries; 
						i += blockDim.x * gridDim.x) 
				{
					const unsigned int ibdof = i/BVec;
					const unsigned int ivec=  i%BVec; 

					*(X+N*(startingDofId+ibdof)+startingVecId+ivec)
						+=rotatedXBlockSP[ibdof*BVec+ivec];
				}

			}

		namespace internal
		{
			unsigned int setChebyshevOrder(const unsigned int d_upperBoundUnWantedSpectrum)
			{
				for(int i=0; i<sizeof(order_lookup)/sizeof(order_lookup[0]); i++) {
					if(d_upperBoundUnWantedSpectrum <= order_lookup[i][0])
						return order_lookup[i][1];
				}
				return 1250;
			}
		}
	}

	//
	// Constructor.
	//
	chebyshevOrthogonalizedSubspaceIterationSolverCUDA::chebyshevOrthogonalizedSubspaceIterationSolverCUDA
		(const MPI_Comm &mpi_comm_domain,
		 double lowerBoundWantedSpectrum,
		 double lowerBoundUnWantedSpectrum,
     double upperBoundUnWantedSpectrum):
			d_lowerBoundWantedSpectrum(lowerBoundWantedSpectrum),
			d_lowerBoundUnWantedSpectrum(lowerBoundUnWantedSpectrum),
      d_upperBoundUnWantedSpectrum(upperBoundUnWantedSpectrum),
			pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
			computing_timer(mpi_comm_domain,
					pcout,
					dftParameters::reproducible_output ||
					dftParameters::verbosity<4? dealii::TimerOutput::never : dealii::TimerOutput::summary,
					dealii::TimerOutput::wall_times)
			{

			}

	//
	// Destructor.
	//
	chebyshevOrthogonalizedSubspaceIterationSolverCUDA::~chebyshevOrthogonalizedSubspaceIterationSolverCUDA()
	{

		//
		//
		//
		return;

	}

	//
	//reinitialize spectrum bounds
	//
	void
		chebyshevOrthogonalizedSubspaceIterationSolverCUDA::reinitSpectrumBounds(double lowerBoundWantedSpectrum,
				double lowerBoundUnWantedSpectrum)
		{
			d_lowerBoundWantedSpectrum = lowerBoundWantedSpectrum;
			d_lowerBoundUnWantedSpectrum = lowerBoundUnWantedSpectrum;
		}

	//
	//
	//
	void
		chebyshevOrthogonalizedSubspaceIterationSolverCUDA::onlyRR(operatorDFTCUDAClass  & operatorMatrix,
				double* eigenVectorsFlattenedCUDA,
				double* eigenVectorsRotFracDensityFlattenedCUDA,
				const unsigned int flattenedSize,
				distributedCPUVec<double>  & tempEigenVec,
				const unsigned int totalNumberWaveFunctions,
				std::vector<double>        & eigenValues,
        GPUCCLWrapper & gpucclMpiCommDomain,
				const MPI_Comm &interBandGroupComm,
				dealii::ScaLAPACKMatrix<double> & projHamPar,
				dealii::ScaLAPACKMatrix<double> & overlapMatPar,
				const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid> & processGrid,
				const bool useMixedPrecOverall,
				const bool isElpaStep1,
				const bool isElpaStep2)
		{
#ifdef USE_COMPLEX
			AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
			double gpu_time, start_time, sub_gpu_time;
			int this_process;

			MPI_Comm_rank(MPI_COMM_WORLD, &this_process);


			cublasHandle_t & cublasHandle =
				operatorMatrix.getCublasHandle();

			//
			//allocate memory for full flattened array on device and fill it up
			//
			const unsigned int localVectorSize = flattenedSize/totalNumberWaveFunctions;

			cudaDeviceSynchronize(); 
			MPI_Barrier(MPI_COMM_WORLD);
			start_time = MPI_Wtime();

			//band group parallelization data structures
			const unsigned int numberBandGroups=
				dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);


			const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
			std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
			dftUtils::createBandParallelizationIndices(interBandGroupComm,
					totalNumberWaveFunctions,
					bandGroupLowHighPlusOneIndices);


			const unsigned int vectorsBlockSize=std::min(dftParameters::chebyWfcBlockSize,
					totalNumberWaveFunctions);

			distributedGPUVec<double> cudaFlattenedArrayBlock;
			vectorTools::createDealiiVector(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
					vectorsBlockSize,
					cudaFlattenedArrayBlock);


			distributedGPUVec<double> YArray;
			YArray.reinit(cudaFlattenedArrayBlock);


			distributedGPUVec<float> cudaFlattenedFloatArrayBlock;
			vectorTools::createDealiiVector(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
					vectorsBlockSize,
					cudaFlattenedFloatArrayBlock);


			distributedGPUVec<double> projectorKetTimesVector;
			vectorTools::createDealiiVector(operatorMatrix.getProjectorKetTimesVectorSingle().get_partitioner(),
					vectorsBlockSize,
					projectorKetTimesVector);


			if(!isElpaStep2)
			{
				//
				//scale the eigenVectors (initial guess of single atom wavefunctions or previous guess) to convert into Lowden Orthonormalized FE basis
				//multiply by M^{1/2}
				scaleCUDAKernel<<<(totalNumberWaveFunctions+255)/256*localVectorSize,256>>>(totalNumberWaveFunctions,
						localVectorSize,
						1.0,
						eigenVectorsFlattenedCUDA,
						operatorMatrix.getSqrtMassVec());


				//gpu_time = MPI_Wtime();
				for (unsigned int i=0;i<eigenValues.size();i++)
					eigenValues[i]=0.0;

			}

			if (eigenValues.size()!=totalNumberWaveFunctions)
			{
				linearAlgebraOperationsCUDA::rayleighRitzSpectrumSplitDirect(operatorMatrix,
						eigenVectorsFlattenedCUDA,
						eigenVectorsRotFracDensityFlattenedCUDA,
						cudaFlattenedArrayBlock,
						cudaFlattenedFloatArrayBlock,
						YArray,
						projectorKetTimesVector,
						localVectorSize,
						totalNumberWaveFunctions,
						totalNumberWaveFunctions-eigenValues.size(),
						isElpaStep1,
						isElpaStep2,
						operatorMatrix.getMPICommunicator(),
            gpucclMpiCommDomain,
						&eigenValues[0],
						cublasHandle,
						projHamPar,
						processGrid,
						useMixedPrecOverall);


				if (isElpaStep1)
				{
					cudaDeviceSynchronize();
					MPI_Barrier(MPI_COMM_WORLD);
					gpu_time = MPI_Wtime() - start_time;
					if (this_process==0 && dftParameters::verbosity>=2)
						std::cout<<"Time for all steps of subspace iteration on GPU till ELPA step 1: "<<gpu_time<<std::endl; 
					return;
				}

			}
			else
			{
				linearAlgebraOperationsCUDA::rayleighRitz(operatorMatrix,
						eigenVectorsFlattenedCUDA,
						cudaFlattenedArrayBlock,
						cudaFlattenedFloatArrayBlock,
						YArray,
						projectorKetTimesVector,
						localVectorSize,
						totalNumberWaveFunctions,
						isElpaStep1,
						isElpaStep2,
						operatorMatrix.getMPICommunicator(),
            gpucclMpiCommDomain,
						interBandGroupComm,
						&eigenValues[0],
						cublasHandle,
						projHamPar,
						processGrid,
						useMixedPrecOverall);

				if (isElpaStep1)
				{
					cudaDeviceSynchronize();
					MPI_Barrier(MPI_COMM_WORLD);
					gpu_time = MPI_Wtime() - start_time;
					if (this_process==0 && dftParameters::verbosity>=2)
						std::cout<<"Time for all steps of subspace iteration on GPU till ELPA step 1: "<<gpu_time<<std::endl; 
					return;
				}

			}
			//gpu_time = MPI_Wtime() - gpu_time;
			//if (this_process==0)
			//    std::cout<<"Time for Rayleigh Ritz on GPU: "<<gpu_time<<std::endl;



			if(dftParameters::verbosity >= 4)
			{
				pcout<<"Rayleigh-Ritz Done: "<<std::endl;
				pcout<<std::endl;
			}


			//
			//scale the eigenVectors with M^{-1/2} to represent the wavefunctions in the usual FE basis
			//
			scaleCUDAKernel<<<(totalNumberWaveFunctions+255)/256*localVectorSize,256>>>(totalNumberWaveFunctions,
					localVectorSize,
					1.0,
					eigenVectorsFlattenedCUDA,
					operatorMatrix.getInvSqrtMassVec());

			if (eigenValues.size()!=totalNumberWaveFunctions)
				scaleCUDAKernel<<<(eigenValues.size()+255)/256*localVectorSize,256>>>(eigenValues.size(),
						localVectorSize,
						1.0,
						eigenVectorsRotFracDensityFlattenedCUDA,
						operatorMatrix.getInvSqrtMassVec());

			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime() - start_time;

			if (isElpaStep2)
				if (this_process==0 && dftParameters::verbosity>=2)
					std::cout<<"Time for ELPA step 2 on GPU: "<<gpu_time<<std::endl;
				else
					if (this_process==0 && dftParameters::verbosity>=2)
						std::cout<<"Time for all steps of subspace iteration on GPU: "<<gpu_time<<std::endl;
			return;
#endif
		}


	//
	// solve
	//
	void
		chebyshevOrthogonalizedSubspaceIterationSolverCUDA::solve(operatorDFTCUDAClass  & operatorMatrix,
				double* eigenVectorsFlattenedCUDA,
				double* eigenVectorsRotFracDensityFlattenedCUDA,
				const unsigned int flattenedSize,
				distributedCPUVec<double>  & tempEigenVec,
				const unsigned int totalNumberWaveFunctions,
				std::vector<double>        & eigenValues,
				std::vector<double>        & residualNorms,
        GPUCCLWrapper & gpucclMpiCommDomain,
				const MPI_Comm &interBandGroupComm,
				dealii::ScaLAPACKMatrix<double> & projHamPar,
				dealii::ScaLAPACKMatrix<double> & overlapMatPar,
				const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid> & processGrid,
        const bool isFirstFilteringCall,
				const bool isXlBOMDLinearizedSolve,
				const bool useMixedPrecOverall,
				const bool isFirstScf,
				const bool isElpaStep1,
				const bool isElpaStep2)
		{
#ifdef USE_COMPLEX
			AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
			double gpu_time, start_time, sub_gpu_time;
			int this_process;

			MPI_Comm_rank(MPI_COMM_WORLD, &this_process);


			cublasHandle_t & cublasHandle =
				operatorMatrix.getCublasHandle();

			//
			//allocate memory for full flattened array on device and fill it up
			//
			const unsigned int localVectorSize = flattenedSize/totalNumberWaveFunctions;

			cudaDeviceSynchronize(); 
			MPI_Barrier(MPI_COMM_WORLD);
			start_time = MPI_Wtime();

			//band group parallelization data structures
			const unsigned int numberBandGroups=
				dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);


			const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
			std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
			dftUtils::createBandParallelizationIndices(interBandGroupComm,
					totalNumberWaveFunctions,
					bandGroupLowHighPlusOneIndices);


			const unsigned int vectorsBlockSize=std::min(dftParameters::chebyWfcBlockSize,
					totalNumberWaveFunctions);

			distributedGPUVec<double> cudaFlattenedArrayBlock;
			vectorTools::createDealiiVector(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
					vectorsBlockSize,
					cudaFlattenedArrayBlock);


			distributedGPUVec<double> YArray;
			YArray.reinit(cudaFlattenedArrayBlock);

			distributedGPUVec<float> cudaFlattenedFloatArrayBlock;
			vectorTools::createDealiiVector(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
					vectorsBlockSize,
					cudaFlattenedFloatArrayBlock);


			distributedGPUVec<double> projectorKetTimesVector;
			vectorTools::createDealiiVector(operatorMatrix.getProjectorKetTimesVectorSingle().get_partitioner(),
					vectorsBlockSize,
					projectorKetTimesVector);


			distributedGPUVec<float>  projectorKetTimesVectorFloat;
			if (dftParameters::useMixedPrecChebyNonLocal)
				vectorTools::createDealiiVector(operatorMatrix.getProjectorKetTimesVectorSingle().get_partitioner(),
						vectorsBlockSize,
						projectorKetTimesVectorFloat);


			distributedGPUVec<double> cudaFlattenedArrayBlock2;
			if (dftParameters::overlapComputeCommunCheby)
				cudaFlattenedArrayBlock2.reinit(cudaFlattenedArrayBlock);


			distributedGPUVec<double> YArray2;
			if (dftParameters::overlapComputeCommunCheby)
				YArray2.reinit(cudaFlattenedArrayBlock2);


			distributedGPUVec<double> projectorKetTimesVector2;
			if (dftParameters::overlapComputeCommunCheby)
				projectorKetTimesVector2.reinit(projectorKetTimesVector);

			if(!isElpaStep2)
			{
				cudaDeviceSynchronize();
				MPI_Barrier(MPI_COMM_WORLD);
				double lanczos_time = MPI_Wtime();

        const std::pair<double,double> bounds =linearAlgebraOperationsCUDA::lanczosLowerUpperBoundEigenSpectrum(operatorMatrix,
            tempEigenVec,
            cudaFlattenedArrayBlock,
            YArray,
            projectorKetTimesVector,
            vectorsBlockSize);

        if (isFirstFilteringCall)
        {
          d_lowerBoundWantedSpectrum=bounds.first;
          d_upperBoundUnWantedSpectrum=bounds.second;
          d_lowerBoundUnWantedSpectrum=d_lowerBoundWantedSpectrum+(d_upperBoundUnWantedSpectrum-d_lowerBoundWantedSpectrum)*totalNumberWaveFunctions/tempEigenVec.size()*10.0;
        }
        else
        {
          d_upperBoundUnWantedSpectrum=bounds.second;
        }

				cudaDeviceSynchronize();
				MPI_Barrier(MPI_COMM_WORLD);
				double gpu_time = MPI_Wtime();
				if (this_process==0 && dftParameters::verbosity>=2)
					std::cout<<"Time for Lanczos Upper Bound: "<<gpu_time-lanczos_time<<std::endl;

				unsigned int chebyshevOrder = dftParameters::chebyshevOrder;

				//
				//set Chebyshev order
				//
				if(chebyshevOrder == 0)
					chebyshevOrder=internal::setChebyshevOrder(d_upperBoundUnWantedSpectrum);

				chebyshevOrder = (isFirstScf && dftParameters::isPseudopotential)?chebyshevOrder*dftParameters::chebyshevFilterPolyDegreeFirstScfScalingFactor:chebyshevOrder;


				//
				//output statements
				//
				if (dftParameters::verbosity>=2)
				{
					char buffer[100];

					sprintf(buffer, "%s:%18.10e\n", "upper bound of unwanted spectrum", d_upperBoundUnWantedSpectrum);
					pcout << buffer;
					sprintf(buffer, "%s:%18.10e\n", "lower bound of unwanted spectrum", d_lowerBoundUnWantedSpectrum);
					pcout << buffer;
					sprintf(buffer, "%s: %u\n\n", "Chebyshev polynomial degree", chebyshevOrder);
					pcout << buffer;
				}


				//
				//scale the eigenVectors (initial guess of single atom wavefunctions or previous guess) to convert into Lowden Orthonormalized FE basis
				//multiply by M^{1/2}
				scaleCUDAKernel<<<(totalNumberWaveFunctions+255)/256*localVectorSize,256>>>(totalNumberWaveFunctions,
						localVectorSize,
						1.0,
						eigenVectorsFlattenedCUDA,
						operatorMatrix.getSqrtMassVec());


				//two blocks of wavefunctions are filtered simultaneously when overlap compute communication in chebyshev
				//filtering is toggled on
				const unsigned int numSimultaneousBlocks=dftParameters::overlapComputeCommunCheby?2:1;
				unsigned int numSimultaneousBlocksCurrent=numSimultaneousBlocks;
				const unsigned int numWfcsInBandGroup=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1]-bandGroupLowHighPlusOneIndices[2*bandGroupTaskId];
				int startIndexBandParal=totalNumberWaveFunctions;
				int numVectorsBandParal=0;
				for (unsigned int jvec = 0; jvec < totalNumberWaveFunctions; jvec += numSimultaneousBlocksCurrent*vectorsBlockSize)
				{

					// Correct block dimensions if block "goes off edge of" the matrix
					const unsigned int BVec = vectorsBlockSize;//std::min(vectorsBlockSize, totalNumberWaveFunctions-jvec);

					//handle edge case when total number of blocks in a given band group is not even in case of 
					//overlapping computation and communciation in chebyshev filtering 
					const unsigned int leftIndexBandGroupMargin=(jvec/numWfcsInBandGroup)*numWfcsInBandGroup;
					numSimultaneousBlocksCurrent
						=((jvec+numSimultaneousBlocks*BVec-leftIndexBandGroupMargin)<=numWfcsInBandGroup && numSimultaneousBlocks==2)?2:1;

					if ((jvec+numSimultaneousBlocksCurrent*BVec)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
							(jvec+numSimultaneousBlocksCurrent*BVec)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
					{

						if (jvec<startIndexBandParal)
							startIndexBandParal=jvec;
						numVectorsBandParal= jvec+numSimultaneousBlocksCurrent*BVec-startIndexBandParal;

						//copy from vector containg all wavefunction vectors to current wavefunction vectors block
						stridedCopyToBlockKernel<<<(BVec+255)/256*localVectorSize, 256>>>(BVec,
								localVectorSize,
								eigenVectorsFlattenedCUDA,
								totalNumberWaveFunctions,
								cudaFlattenedArrayBlock.begin(),
								jvec);

						if (dftParameters::overlapComputeCommunCheby && numSimultaneousBlocksCurrent==2)
							stridedCopyToBlockKernel<<<(BVec+255)/256*localVectorSize, 256>>>(BVec,
									localVectorSize,
									eigenVectorsFlattenedCUDA,
									totalNumberWaveFunctions,
									cudaFlattenedArrayBlock2.begin(),
									jvec+BVec);

						//
						//call Chebyshev filtering function only for the current block or two simulataneous blocks
						//(in case of overlap computation and communication) to be filtered and does in-place filtering
						if (dftParameters::overlapComputeCommunCheby && numSimultaneousBlocksCurrent==2)
						{
              linearAlgebraOperationsCUDA::chebyshevFilter(operatorMatrix,
                  cudaFlattenedArrayBlock,
                  YArray,
                  cudaFlattenedFloatArrayBlock,
                  projectorKetTimesVector,
                  projectorKetTimesVectorFloat,
                  cudaFlattenedArrayBlock2,
                  YArray2,
                  projectorKetTimesVector2,
                  localVectorSize,
                  BVec,
                  chebyshevOrder,
                  d_lowerBoundUnWantedSpectrum,
                  d_upperBoundUnWantedSpectrum,
                  d_lowerBoundWantedSpectrum,
                  useMixedPrecOverall);	
						}
						else
						{
              linearAlgebraOperationsCUDA::chebyshevFilter(operatorMatrix,
                  cudaFlattenedArrayBlock,
                  YArray,
                  cudaFlattenedFloatArrayBlock,
                  projectorKetTimesVector,
                  localVectorSize,
                  BVec,
                  chebyshevOrder,
                  d_lowerBoundUnWantedSpectrum,
                  d_upperBoundUnWantedSpectrum,
                  d_lowerBoundWantedSpectrum,
                  useMixedPrecOverall);	
						}

						//copy current wavefunction vectors block to vector containing all wavefunction vectors
						stridedCopyFromBlockKernel<<<(BVec+255)/256*localVectorSize, 256>>>(BVec,
								localVectorSize,
								cudaFlattenedArrayBlock.begin(),
								totalNumberWaveFunctions,
								eigenVectorsFlattenedCUDA,
								jvec);

						if (dftParameters::overlapComputeCommunCheby && numSimultaneousBlocksCurrent==2)
							stridedCopyFromBlockKernel<<<(BVec+255)/256*localVectorSize, 256>>>(BVec,
									localVectorSize,
									cudaFlattenedArrayBlock2.begin(),
									totalNumberWaveFunctions,
									eigenVectorsFlattenedCUDA,
									jvec+BVec);
					}
					else
					{
						//set to zero wavefunctions which wont go through chebyshev filtering inside a given band group
						setZeroKernel<<<(numSimultaneousBlocksCurrent*BVec+255)/256*localVectorSize, 256>>>(numSimultaneousBlocksCurrent*BVec,
								localVectorSize,
								totalNumberWaveFunctions,
								eigenVectorsFlattenedCUDA,
								jvec);
					}

				}//block loop

				cudaDeviceSynchronize();
				MPI_Barrier(MPI_COMM_WORLD);
				gpu_time = MPI_Wtime() - gpu_time;
				if (this_process==0 && dftParameters::verbosity>=2)
					std::cout<<"Time for chebyshev filtering on GPU: "<<gpu_time<<std::endl;


				if(dftParameters::verbosity >= 4)
					pcout<<"ChebyShev Filtering Done: "<<std::endl;


				if (numberBandGroups>1)
				{
					cudaDeviceSynchronize();
					MPI_Barrier(MPI_COMM_WORLD);
					double band_paral_time=MPI_Wtime();

					std::vector<double> eigenVectorsFlattened(totalNumberWaveFunctions*localVectorSize,0);

					//cudaDeviceSynchronize();
					//double copytime=MPI_Wtime();
					cudaMemcpy(&eigenVectorsFlattened[0],
							eigenVectorsFlattenedCUDA,
							totalNumberWaveFunctions*localVectorSize*sizeof(double),
							cudaMemcpyDeviceToHost);
					//cudaDeviceSynchronize();
					//copytime = MPI_Wtime() - copytime;
					//if (this_process==0)
					//   std::cout<<"copy time on GPU: "<<copytime<<std::endl;

					MPI_Barrier(interBandGroupComm);

					if (true)
					{
						MPI_Allreduce(MPI_IN_PLACE,
								&eigenVectorsFlattened[0],
								totalNumberWaveFunctions*localVectorSize,
								MPI_DOUBLE,
								MPI_SUM,
								interBandGroupComm);
					}
					else
					{
						std::vector<double> eigenVectorsBandGroup(numVectorsBandParal*localVectorSize,0);
						std::vector<double> eigenVectorsBandGroupTransposed(numVectorsBandParal*localVectorSize,0);
						std::vector<double> eigenVectorsTransposed(totalNumberWaveFunctions*localVectorSize,0);

						for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
							for(unsigned int iWave = 0; iWave < numVectorsBandParal; ++iWave)
								eigenVectorsBandGroup[iNode*numVectorsBandParal+iWave]
									= eigenVectorsFlattened[iNode*totalNumberWaveFunctions+startIndexBandParal+iWave];


						for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
							for(unsigned int iWave = 0; iWave < numVectorsBandParal; ++iWave)
								eigenVectorsBandGroupTransposed[iWave*localVectorSize+iNode]
									= eigenVectorsBandGroup[iNode*numVectorsBandParal+iWave];

						std::vector<int> recvcounts(numberBandGroups,0);
						std::vector<int> displs(numberBandGroups,0);

						int recvcount=numVectorsBandParal*localVectorSize;
						MPI_Allgather(&recvcount,
								1,
								MPI_INT,
								&recvcounts[0],
								1,
								MPI_INT,
								interBandGroupComm);

						int displ=startIndexBandParal*localVectorSize;
						MPI_Allgather(&displ,
								1,
								MPI_INT,
								&displs[0],
								1,
								MPI_INT,
								interBandGroupComm);

						MPI_Allgatherv(&eigenVectorsBandGroupTransposed[0],
								numVectorsBandParal*localVectorSize,
								MPI_DOUBLE,
								&eigenVectorsTransposed[0],
								&recvcounts[0],
								&displs[0],
								dataTypes::mpi_type_id(&eigenVectorsTransposed[0]),
								interBandGroupComm);


						for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
							for(unsigned int iWave = 0; iWave < totalNumberWaveFunctions; ++iWave)
								eigenVectorsFlattened[iNode*totalNumberWaveFunctions+iWave]
									= eigenVectorsTransposed[iWave*localVectorSize+iNode];
					}
					MPI_Barrier(interBandGroupComm);

					//cudaDeviceSynchronize();
					//copytime=MPI_Wtime();
					cudaMemcpy(eigenVectorsFlattenedCUDA,
							&eigenVectorsFlattened[0],
							totalNumberWaveFunctions*localVectorSize*sizeof(double),
							cudaMemcpyHostToDevice);
					//cudaDeviceSynchronize();
					//copytime = MPI_Wtime() - copytime;
					//if (this_process==0)
					//   std::cout<<"copy time on GPU: "<<copytime<<std::endl;
					cudaDeviceSynchronize();
					MPI_Barrier(MPI_COMM_WORLD);
					band_paral_time = MPI_Wtime() - band_paral_time;

					if (this_process==0 && dftParameters::verbosity>=2)
						std::cout<<"Time for band parallelization communication: "<<band_paral_time<<std::endl;

				}

				//if (dftParameters::measureOnlyChebyTime)
				//  exit(0);

				/*
				   int inc=1;
				   double result=0.0;
				   cublasDnrm2(cublasHandle,
				   flattenedSize,
				   eigenVectorsFlattenedCUDA,
				   inc, 
				   &result);
				   result=result*result;
				   result=dealii::Utilities::MPI::sum(result,operatorMatrix.getMPICommunicator());
				   std::cout<<"l2 norm Chebyshev filtered x: "<<std::sqrt(result)<<std::endl;
				 */

				if (dftParameters::rrGEP==false)
				{
					if(dftParameters::orthogType.compare("LW") == 0)
					{

						AssertThrow(false,dealii::ExcMessage("Lowden Gram-Schmidt Orthonormalization Not implemented in CUDA:"));

					}
					else if (dftParameters::orthogType.compare("PGS") == 0)
					{
						//gpu_time = MPI_Wtime();
						linearAlgebraOperationsCUDA::pseudoGramSchmidtOrthogonalization
							(operatorMatrix,
							 eigenVectorsFlattenedCUDA,
							 localVectorSize,
							 totalNumberWaveFunctions,
							 operatorMatrix.getMPICommunicator(),
               gpucclMpiCommDomain,
							 interBandGroupComm,
							 cublasHandle,
							 useMixedPrecOverall);

						//gpu_time = MPI_Wtime() - gpu_time;
						//if (this_process==0)
						//    std::cout<<"Time for PGS on GPU: "<<gpu_time<<std::endl;

					}
					else if (dftParameters::orthogType.compare("GS") == 0)
					{

						AssertThrow(false,dealii::ExcMessage("Classical Gram-Schmidt Orthonormalization not implemented in CUDA:"));

					}

					if(dftParameters::verbosity >= 4)
						pcout<<"Orthogonalization Done: "<<std::endl;

				}

				//gpu_time = MPI_Wtime();
				for (unsigned int i=0;i<eigenValues.size();i++)
					eigenValues[i]=0.0;

			}

			if (eigenValues.size()!=totalNumberWaveFunctions)
			{
				if (dftParameters::rrGEP==false)             
					linearAlgebraOperationsCUDA::rayleighRitzSpectrumSplitDirect(operatorMatrix,
							eigenVectorsFlattenedCUDA,
							eigenVectorsRotFracDensityFlattenedCUDA,
							cudaFlattenedArrayBlock,
							cudaFlattenedFloatArrayBlock,
							YArray,
							projectorKetTimesVector,
							localVectorSize,
							totalNumberWaveFunctions,
							totalNumberWaveFunctions-eigenValues.size(),
							isElpaStep1,
							isElpaStep2,
							operatorMatrix.getMPICommunicator(),
              gpucclMpiCommDomain,
							&eigenValues[0],
							cublasHandle,
							projHamPar,
							processGrid,
							useMixedPrecOverall);
				else           
					linearAlgebraOperationsCUDA::rayleighRitzGEPSpectrumSplitDirect(operatorMatrix,
							eigenVectorsFlattenedCUDA,
							eigenVectorsRotFracDensityFlattenedCUDA,
							cudaFlattenedArrayBlock,
							cudaFlattenedFloatArrayBlock,
							YArray,
							projectorKetTimesVector,
							localVectorSize,
							totalNumberWaveFunctions,
							totalNumberWaveFunctions-eigenValues.size(),
							isElpaStep1,
							isElpaStep2,
							operatorMatrix.getMPICommunicator(),
              gpucclMpiCommDomain,
							interBandGroupComm,
							&eigenValues[0],
							cublasHandle,
							projHamPar,
							overlapMatPar,
							processGrid,
							useMixedPrecOverall);

				if (isElpaStep1)
				{
					cudaDeviceSynchronize();
					MPI_Barrier(MPI_COMM_WORLD);
					gpu_time = MPI_Wtime() - start_time;
					if (this_process==0 && dftParameters::verbosity>=2)
						std::cout<<"Time for all steps of subspace iteration on GPU till ELPA step 1: "<<gpu_time<<std::endl; 
					return;
				}

			}
			else
			{
				if (dftParameters::rrGEP==false)
					linearAlgebraOperationsCUDA::rayleighRitz(operatorMatrix,
							eigenVectorsFlattenedCUDA,
							cudaFlattenedArrayBlock,
							cudaFlattenedFloatArrayBlock,
							YArray,
							projectorKetTimesVector,
							localVectorSize,
							totalNumberWaveFunctions,
							isElpaStep1,
							isElpaStep2,
							operatorMatrix.getMPICommunicator(),
              gpucclMpiCommDomain,
							interBandGroupComm,
							&eigenValues[0],
							cublasHandle,
							projHamPar,
							processGrid,
							useMixedPrecOverall);
				else
					linearAlgebraOperationsCUDA::rayleighRitzGEP(operatorMatrix,
							eigenVectorsFlattenedCUDA,
							cudaFlattenedArrayBlock,
							cudaFlattenedFloatArrayBlock,
							YArray,
							projectorKetTimesVector,
							localVectorSize,
							totalNumberWaveFunctions,
							isElpaStep1,
							isElpaStep2,
							operatorMatrix.getMPICommunicator(),
              gpucclMpiCommDomain,
							interBandGroupComm,
							&eigenValues[0],
							cublasHandle,
							projHamPar,
							overlapMatPar,
							processGrid,
							useMixedPrecOverall);


				if (isElpaStep1)
				{
					cudaDeviceSynchronize();
					MPI_Barrier(MPI_COMM_WORLD);
					gpu_time = MPI_Wtime() - start_time;
					if (this_process==0 && dftParameters::verbosity>=2)
						std::cout<<"Time for all steps of subspace iteration on GPU till ELPA step 1: "<<gpu_time<<std::endl; 
					return;
				}

			}
			//gpu_time = MPI_Wtime() - gpu_time;
			//if (this_process==0)
			//    std::cout<<"Time for Rayleigh Ritz on GPU: "<<gpu_time<<std::endl;



			if(dftParameters::verbosity >= 4)
			{
				pcout<<"Rayleigh-Ritz Done: "<<std::endl;
				pcout<<std::endl;
			}

			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime();
			if (eigenValues.size()!=totalNumberWaveFunctions)
				linearAlgebraOperationsCUDA::computeEigenResidualNorm(operatorMatrix,
						eigenVectorsRotFracDensityFlattenedCUDA,
						cudaFlattenedArrayBlock,
						YArray,
						projectorKetTimesVector,
						localVectorSize,
						eigenValues.size(),
						eigenValues,
						operatorMatrix.getMPICommunicator(),
						interBandGroupComm,
						cublasHandle,
						residualNorms);
			else
				linearAlgebraOperationsCUDA::computeEigenResidualNorm(operatorMatrix,
						eigenVectorsFlattenedCUDA,
						cudaFlattenedArrayBlock,
						YArray,
						projectorKetTimesVector,
						localVectorSize,
						totalNumberWaveFunctions,
						eigenValues,
						operatorMatrix.getMPICommunicator(),
						interBandGroupComm,
						cublasHandle,
						residualNorms,
						true);

			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime() - gpu_time;
			if (this_process==0 && dftParameters::verbosity>=2)
				std::cout<<"Time to compute residual norm: "<<gpu_time<<std::endl;

			//
			//scale the eigenVectors with M^{-1/2} to represent the wavefunctions in the usual FE basis
			//
			scaleCUDAKernel<<<(totalNumberWaveFunctions+255)/256*localVectorSize,256>>>(totalNumberWaveFunctions,
					localVectorSize,
					1.0,
					eigenVectorsFlattenedCUDA,
					operatorMatrix.getInvSqrtMassVec());

			if (eigenValues.size()!=totalNumberWaveFunctions)
				scaleCUDAKernel<<<(eigenValues.size()+255)/256*localVectorSize,256>>>(eigenValues.size(),
						localVectorSize,
						1.0,
						eigenVectorsRotFracDensityFlattenedCUDA,
						operatorMatrix.getInvSqrtMassVec());

			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime() - start_time;

			if (isElpaStep2)
				if (this_process==0 && dftParameters::verbosity>=2)
					std::cout<<"Time for ELPA step 2 on GPU: "<<gpu_time<<std::endl;
				else
					if (this_process==0 && dftParameters::verbosity>=2)
						std::cout<<"Time for all steps of subspace iteration on GPU: "<<gpu_time<<std::endl;
			return;
#endif
		}

	//
	// solve
	//
	void
		chebyshevOrthogonalizedSubspaceIterationSolverCUDA::solveNoRR(operatorDFTCUDAClass  & operatorMatrix,
				double* eigenVectorsFlattenedCUDA,
				const unsigned int flattenedSize,
				distributedCPUVec<double>  & tempEigenVec,
				const unsigned int totalNumberWaveFunctions,
				std::vector<double>        & eigenValues,
        GPUCCLWrapper & gpucclMpiCommDomain,
				const MPI_Comm &interBandGroupComm,
				dealii::ScaLAPACKMatrix<double> & projHamPar,
				dealii::ScaLAPACKMatrix<double> & overlapMatPar,
				const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid> & processGrid,
				const bool isXlBOMDLinearizedSolve,
				const unsigned int numberPasses,
				const bool useMixedPrecOverall)
		{
#ifdef USE_COMPLEX
			AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
			double gpu_time, start_time, sub_gpu_time;
			int this_process;

			MPI_Comm_rank(MPI_COMM_WORLD, &this_process);


			cublasHandle_t & cublasHandle =
				operatorMatrix.getCublasHandle();

			//
			//allocate memory for full flattened array on device and fill it up
			//
			const unsigned int localVectorSize = flattenedSize/totalNumberWaveFunctions;

			cudaDeviceSynchronize(); 
			MPI_Barrier(MPI_COMM_WORLD);
			start_time = MPI_Wtime();

			//band group parallelization data structures
			const unsigned int numberBandGroups=
				dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);


			const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
			std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
			dftUtils::createBandParallelizationIndices(interBandGroupComm,
					totalNumberWaveFunctions,
					bandGroupLowHighPlusOneIndices);

			const unsigned int wfcBlockSize=std::min(dftParameters::wfcBlockSize,
					totalNumberWaveFunctions);


			const unsigned int chebyBlockSize=std::min(dftParameters::chebyWfcBlockSize,
					totalNumberWaveFunctions);

			distributedGPUVec<double> cudaFlattenedArrayBlock;
			vectorTools::createDealiiVector(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
					chebyBlockSize,
					cudaFlattenedArrayBlock);


			distributedGPUVec<double> YArray;
			YArray.reinit(cudaFlattenedArrayBlock);

			distributedGPUVec<float> cudaFlattenedFloatArrayBlock;
			vectorTools::createDealiiVector(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
					chebyBlockSize,
					cudaFlattenedFloatArrayBlock);


			distributedGPUVec<double> projectorKetTimesVector;
			vectorTools::createDealiiVector(operatorMatrix.getProjectorKetTimesVectorSingle().get_partitioner(),
					chebyBlockSize,
					projectorKetTimesVector);



			distributedGPUVec<float>  projectorKetTimesVectorFloat;
			if (dftParameters::useMixedPrecChebyNonLocal)
				vectorTools::createDealiiVector(operatorMatrix.getProjectorKetTimesVectorSingle().get_partitioner(),
						chebyBlockSize,
						projectorKetTimesVectorFloat);


			distributedGPUVec<double> cudaFlattenedArrayBlock2;
			if (dftParameters::overlapComputeCommunCheby)
				cudaFlattenedArrayBlock2.reinit(cudaFlattenedArrayBlock);


			distributedGPUVec<double> YArray2;
			if (dftParameters::overlapComputeCommunCheby)
				YArray2.reinit(cudaFlattenedArrayBlock2);


			distributedGPUVec<double> projectorKetTimesVector2;
			if (dftParameters::overlapComputeCommunCheby)
				projectorKetTimesVector2.reinit(projectorKetTimesVector);

			computing_timer.enter_section("Lanczos k-step Upper Bound");
		  const double d_upperBoundUnWantedSpectrum =linearAlgebraOperationsCUDA::lanczosLowerUpperBoundEigenSpectrum(operatorMatrix,
					tempEigenVec,
          cudaFlattenedArrayBlock,
          YArray,
          projectorKetTimesVector,          
          chebyBlockSize).second;
			computing_timer.exit_section("Lanczos k-step Upper Bound");
			unsigned int chebyshevOrder = dftParameters::chebyshevOrder;

			//
			//set Chebyshev order
			//
			if(chebyshevOrder == 0)
				chebyshevOrder=internal::setChebyshevOrder(d_upperBoundUnWantedSpectrum);

			chebyshevOrder = (dftParameters::isPseudopotential)?chebyshevOrder*dftParameters::chebyshevFilterPolyDegreeFirstScfScalingFactor:chebyshevOrder;


			//
			//output statements
			//
			if (dftParameters::verbosity>=2)
			{
				char buffer[100];

				sprintf(buffer, "%s:%18.10e\n", "upper bound of unwanted spectrum", d_upperBoundUnWantedSpectrum);
				pcout << buffer;
				sprintf(buffer, "%s:%18.10e\n", "lower bound of unwanted spectrum", d_lowerBoundUnWantedSpectrum);
				pcout << buffer;
				sprintf(buffer, "%s: %u\n\n", "Chebyshev polynomial degree", chebyshevOrder);
				pcout << buffer;
			}


			//
			//scale the eigenVectors (initial guess of single atom wavefunctions or previous guess) to convert into Lowden Orthonormalized FE basis
			//multiply by M^{1/2}
			scaleCUDAKernel<<<(totalNumberWaveFunctions+255)/256*localVectorSize,256>>>(totalNumberWaveFunctions,
					localVectorSize,
					1.0,
					eigenVectorsFlattenedCUDA,
					operatorMatrix.getSqrtMassVec());

			for (unsigned int ipass = 0; ipass < numberPasses; ipass++)
			{
				if (this_process==0 && dftParameters::verbosity>=1)
					std::cout<<"Beginning no RR Chebyshev filter subpspace iteration pass: "<<ipass+1<<std::endl;

				for (unsigned int ivec = 0; ivec < totalNumberWaveFunctions; ivec += wfcBlockSize)
				{
					//two blocks of wavefunctions are filtered simultaneously when overlap compute communication in chebyshev
					//filtering is toggled on
					const unsigned int numSimultaneousBlocks=dftParameters::overlapComputeCommunCheby?2:1;
					unsigned int numSimultaneousBlocksCurrent=numSimultaneousBlocks;
					const unsigned int numWfcsInBandGroup=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1]-bandGroupLowHighPlusOneIndices[2*bandGroupTaskId];
					for (unsigned int jvec = ivec; jvec < (ivec+wfcBlockSize); jvec += numSimultaneousBlocksCurrent*chebyBlockSize)
					{

						// Correct block dimensions if block "goes off edge of" the matrix
						const unsigned int BVec = chebyBlockSize;//std::min(vectorsBlockSize, totalNumberWaveFunctions-jvec);

						//handle edge case when total number of blocks in a given band group is not even in case of 
						//overlapping computation and communciation in chebyshev filtering 
						const unsigned int leftIndexBandGroupMargin=(jvec/numWfcsInBandGroup)*numWfcsInBandGroup;
						numSimultaneousBlocksCurrent
							=((jvec+numSimultaneousBlocks*BVec-leftIndexBandGroupMargin)<=numWfcsInBandGroup && numSimultaneousBlocks==2)?2:1;

						if ((jvec+numSimultaneousBlocksCurrent*BVec)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
								(jvec+numSimultaneousBlocksCurrent*BVec)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
						{

							//copy from vector containg all wavefunction vectors to current wavefunction vectors block
							stridedCopyToBlockKernel<<<(BVec+255)/256*localVectorSize, 256>>>(BVec,
									localVectorSize,
									eigenVectorsFlattenedCUDA,
									totalNumberWaveFunctions,
									cudaFlattenedArrayBlock.begin(),
									jvec);

							if (dftParameters::overlapComputeCommunCheby && numSimultaneousBlocksCurrent==2)
								stridedCopyToBlockKernel<<<(BVec+255)/256*localVectorSize, 256>>>(BVec,
										localVectorSize,
										eigenVectorsFlattenedCUDA,
										totalNumberWaveFunctions,
										cudaFlattenedArrayBlock2.begin(),
										jvec+BVec);

							//
							//call Chebyshev filtering function only for the current block or two simulataneous blocks
							//(in case of overlap computation and communication) to be filtered and does in-place filtering
							if (dftParameters::overlapComputeCommunCheby && numSimultaneousBlocksCurrent==2)
							{
                linearAlgebraOperationsCUDA::chebyshevFilter(operatorMatrix,
                    cudaFlattenedArrayBlock,
                    YArray,
                    cudaFlattenedFloatArrayBlock,
                    projectorKetTimesVector,
                    projectorKetTimesVectorFloat,
                    cudaFlattenedArrayBlock2,
                    YArray2,
                    projectorKetTimesVector2,
                    localVectorSize,
                    BVec,
                    chebyshevOrder,
                    d_lowerBoundUnWantedSpectrum,
                    d_upperBoundUnWantedSpectrum,
                    d_lowerBoundWantedSpectrum,
                    useMixedPrecOverall);	
							}
							else
							{
                linearAlgebraOperationsCUDA::chebyshevFilter(operatorMatrix,
                    cudaFlattenedArrayBlock,
                    YArray,
                    cudaFlattenedFloatArrayBlock,
                    projectorKetTimesVector,
                    localVectorSize,
                    BVec,
                    chebyshevOrder,
                    d_lowerBoundUnWantedSpectrum,
                    d_upperBoundUnWantedSpectrum,
                    d_lowerBoundWantedSpectrum,
                    useMixedPrecOverall);	
							}	

							//copy current wavefunction vectors block to vector containing all wavefunction vectors
							stridedCopyFromBlockKernel<<<(BVec+255)/256*localVectorSize, 256>>>(BVec,
									localVectorSize,
									cudaFlattenedArrayBlock.begin(),
									totalNumberWaveFunctions,
									eigenVectorsFlattenedCUDA,
									jvec);

							if (dftParameters::overlapComputeCommunCheby && numSimultaneousBlocksCurrent==2)
								stridedCopyFromBlockKernel<<<(BVec+255)/256*localVectorSize, 256>>>(BVec,
										localVectorSize,
										cudaFlattenedArrayBlock2.begin(),
										totalNumberWaveFunctions,
										eigenVectorsFlattenedCUDA,
										jvec+BVec);
						}
						else
						{
							//set to zero wavefunctions which wont go through chebyshev filtering inside a given band group
							setZeroKernel<<<(numSimultaneousBlocksCurrent*BVec+255)/256*localVectorSize, 256>>>(numSimultaneousBlocksCurrent*BVec,
									localVectorSize,
									totalNumberWaveFunctions,
									eigenVectorsFlattenedCUDA,
									jvec);
						}

					}//cheby block loop
				}//wfc block loop

				if(dftParameters::verbosity >= 4)
					pcout<<"ChebyShev Filtering Done: "<<std::endl;


				linearAlgebraOperationsCUDA::pseudoGramSchmidtOrthogonalization
					(operatorMatrix,
					 eigenVectorsFlattenedCUDA,
					 localVectorSize,
					 totalNumberWaveFunctions,
					 operatorMatrix.getMPICommunicator(),
           gpucclMpiCommDomain,
					 interBandGroupComm,
					 cublasHandle,
					 useMixedPrecOverall);

			}

			//
			//scale the eigenVectors with M^{-1/2} to represent the wavefunctions in the usual FE basis
			//
			scaleCUDAKernel<<<(totalNumberWaveFunctions+255)/256*localVectorSize,256>>>(totalNumberWaveFunctions,
					localVectorSize,
					1.0,
					eigenVectorsFlattenedCUDA,
					operatorMatrix.getInvSqrtMassVec());

			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime() - start_time;

			if (this_process==0 && dftParameters::verbosity>=2)
				std::cout<<"Time for all no RR Chebyshev filtering passes on GPU: "<<gpu_time<<std::endl;
#endif
		}
}
