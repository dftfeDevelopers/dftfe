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
	      unsigned int setChebyshevOrder(const unsigned int upperBoundUnwantedSpectrum)
	      {
		for(int i=0; i<sizeof(order_lookup)/sizeof(order_lookup[0]); i++) {
		  if(upperBoundUnwantedSpectrum <= order_lookup[i][0])
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
  (const MPI_Comm &mpi_comm,
   double lowerBoundWantedSpectrum,
   double lowerBoundUnWantedSpectrum):
    d_lowerBoundWantedSpectrum(lowerBoundWantedSpectrum),
    d_lowerBoundUnWantedSpectrum(lowerBoundUnWantedSpectrum),
    pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
    computing_timer(mpi_comm,
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
							    const MPI_Comm &interBandGroupComm,
                                                            dealii::ScaLAPACKMatrix<double> & projHamPar,
                                                            dealii::ScaLAPACKMatrix<double> & overlapMatPar,
                                                            const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid> & processGrid,
                                                            const bool isXlBOMDLinearizedSolve,
                                                            const bool useMixedPrecOverall,
                                                            const bool isFirstScf,
                                                            const bool useFullMassMatrixGEP,
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

    distributedGPUVec<double> YArrayCA; 
    if (dftParameters::chebyCommunAvoidanceAlgo)
         YArrayCA.reinit(cudaFlattenedArrayBlock);


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
    if (dftParameters::overlapComputeCommunCheby || dftParameters::chebyCommunAvoidanceAlgo)
           cudaFlattenedArrayBlock2.reinit(cudaFlattenedArrayBlock);


    distributedGPUVec<double> YArray2;
    if (dftParameters::overlapComputeCommunCheby)
           YArray2.reinit(cudaFlattenedArrayBlock2);


    distributedGPUVec<double> projectorKetTimesVector2;
    if (dftParameters::overlapComputeCommunCheby)
           projectorKetTimesVector2.reinit(projectorKetTimesVector);

    if(!isElpaStep2)
    {
	    computing_timer.enter_section("Lanczos k-step Upper Bound");
	    operatorMatrix.reinit(1);
	    const double upperBoundUnwantedSpectrum =linearAlgebraOperationsCUDA::lanczosUpperBoundEigenSpectrum(operatorMatrix,
													      tempEigenVec);
	    computing_timer.exit_section("Lanczos k-step Upper Bound");
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);
	    gpu_time = MPI_Wtime();
	    unsigned int chebyshevOrder = dftParameters::chebyshevOrder;

	    //
	    //set Chebyshev order
	    //
	    if(chebyshevOrder == 0)
	      chebyshevOrder=internal::setChebyshevOrder(upperBoundUnwantedSpectrum);

            chebyshevOrder = (isFirstScf && dftParameters::isPseudopotential)?chebyshevOrder*dftParameters::chebyshevFilterPolyDegreeFirstScfScalingFactor:chebyshevOrder;


	    if(dftParameters::lowerBoundUnwantedFracUpper > 1e-6)
	      d_lowerBoundUnWantedSpectrum=dftParameters::lowerBoundUnwantedFracUpper*upperBoundUnwantedSpectrum;

	    //
	    //output statements
	    //
	    if (dftParameters::verbosity>=2)
	      {
		char buffer[100];

		sprintf(buffer, "%s:%18.10e\n", "upper bound of unwanted spectrum", upperBoundUnwantedSpectrum);
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

            double computeAvoidanceTolerance=1e-14;

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
                              if (dftParameters::chebyCommunAvoidanceAlgo && false)
				 linearAlgebraOperationsCUDA::chebyshevFilterComputeAvoidance(operatorMatrix,
									      cudaFlattenedArrayBlock,
									      YArray,
                                                                              YArrayCA,
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
									      upperBoundUnwantedSpectrum,
									      d_lowerBoundWantedSpectrum,
                                                                              isXlBOMDLinearizedSolve,
                                                                              false,
                                                                              useMixedPrecOverall,
                                                                              computeAvoidanceTolerance);
                             else
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
									      upperBoundUnwantedSpectrum,
									      d_lowerBoundWantedSpectrum,
                                                                              useMixedPrecOverall);	
                         }
                         else
                         {
                             if (dftParameters::chebyCommunAvoidanceAlgo && false)
				 linearAlgebraOperationsCUDA::chebyshevFilterComputeAvoidance(operatorMatrix,
										   cudaFlattenedArrayBlock,
										   YArray,
                                                                                   cudaFlattenedArrayBlock2,
										   cudaFlattenedFloatArrayBlock,
										   projectorKetTimesVector,
										   localVectorSize,
										   BVec,
										   chebyshevOrder,
										   d_lowerBoundUnWantedSpectrum,
										   upperBoundUnwantedSpectrum,
										   d_lowerBoundWantedSpectrum,
                                                                                   isXlBOMDLinearizedSolve,
                                                                                   false,
                                                                                   useMixedPrecOverall);	
                             else 
				 linearAlgebraOperationsCUDA::chebyshevFilter(operatorMatrix,
									   cudaFlattenedArrayBlock,
									   YArray,
									   cudaFlattenedFloatArrayBlock,
									   projectorKetTimesVector,
									   localVectorSize,
									   BVec,
									   chebyshevOrder,
									   d_lowerBoundUnWantedSpectrum,
									   upperBoundUnwantedSpectrum,
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

    distributedGPUVec<double> YArrayCA; 
    if (dftParameters::chebyCommunAvoidanceAlgo)
         YArrayCA.reinit(cudaFlattenedArrayBlock);

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
    if (dftParameters::overlapComputeCommunCheby || dftParameters::chebyCommunAvoidanceAlgo)
           cudaFlattenedArrayBlock2.reinit(cudaFlattenedArrayBlock);


    distributedGPUVec<double> YArray2;
    if (dftParameters::overlapComputeCommunCheby)
           YArray2.reinit(cudaFlattenedArrayBlock2);


    distributedGPUVec<double> projectorKetTimesVector2;
    if (dftParameters::overlapComputeCommunCheby)
           projectorKetTimesVector2.reinit(projectorKetTimesVector);

    computing_timer.enter_section("Lanczos k-step Upper Bound");
    operatorMatrix.reinit(1);
    const double upperBoundUnwantedSpectrum =linearAlgebraOperationsCUDA::lanczosUpperBoundEigenSpectrum(operatorMatrix,
												      tempEigenVec);
    computing_timer.exit_section("Lanczos k-step Upper Bound");
    unsigned int chebyshevOrder = dftParameters::chebyshevOrder;

    //
    //set Chebyshev order
    //
    if(chebyshevOrder == 0)
      chebyshevOrder=internal::setChebyshevOrder(upperBoundUnwantedSpectrum);

    chebyshevOrder = (dftParameters::isPseudopotential)?chebyshevOrder*dftParameters::chebyshevFilterPolyDegreeFirstScfScalingFactor:chebyshevOrder;


    if(dftParameters::lowerBoundUnwantedFracUpper > 1e-6)
      d_lowerBoundUnWantedSpectrum=dftParameters::lowerBoundUnwantedFracUpper*upperBoundUnwantedSpectrum;

    //
    //output statements
    //
    if (dftParameters::verbosity>=2)
      {
	char buffer[100];

	sprintf(buffer, "%s:%18.10e\n", "upper bound of unwanted spectrum", upperBoundUnwantedSpectrum);
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

            double computeAvoidanceTolerance=1e-8;
            if (ipass>=2 && ipass<4)
               computeAvoidanceTolerance=1e-10;
            else if (ipass>=4)
               computeAvoidanceTolerance=1e-12;

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
				      if (dftParameters::chebyCommunAvoidanceAlgo)
					 linearAlgebraOperationsCUDA::chebyshevFilterComputeAvoidance(operatorMatrix,
										      cudaFlattenedArrayBlock,
										      YArray,
										      YArrayCA,
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
										      upperBoundUnwantedSpectrum,
										      d_lowerBoundWantedSpectrum,
										      isXlBOMDLinearizedSolve,
										      false,
                                                                                      useMixedPrecOverall,
                                                                                      computeAvoidanceTolerance);
				     else
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
										      upperBoundUnwantedSpectrum,
										      d_lowerBoundWantedSpectrum,
                                                                                      useMixedPrecOverall);	
				 }
				 else
				 {
				     if (dftParameters::chebyCommunAvoidanceAlgo)
					 linearAlgebraOperationsCUDA::chebyshevFilterComputeAvoidance(operatorMatrix,
											   cudaFlattenedArrayBlock,
											   YArray,
											   cudaFlattenedArrayBlock2,
											   cudaFlattenedFloatArrayBlock,
											   projectorKetTimesVector,
											   localVectorSize,
											   BVec,
											   chebyshevOrder,
											   d_lowerBoundUnWantedSpectrum,
											   upperBoundUnwantedSpectrum,
											   d_lowerBoundWantedSpectrum,
											   isXlBOMDLinearizedSolve,
											   false,
                                                                                           useMixedPrecOverall);	
				     else 
					 linearAlgebraOperationsCUDA::chebyshevFilter(operatorMatrix,
											   cudaFlattenedArrayBlock,
											   YArray,
											   cudaFlattenedFloatArrayBlock,
											   projectorKetTimesVector,
											   localVectorSize,
											   BVec,
											   chebyshevOrder,
											   d_lowerBoundUnWantedSpectrum,
											   upperBoundUnwantedSpectrum,
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

 //
  // solve
  //
  void
  chebyshevOrthogonalizedSubspaceIterationSolverCUDA::solveNoRRMixedPrec(operatorDFTCUDAClass  & operatorMatrix,
							    double* eigenVectorsFlattenedCUDA,
                                                            const unsigned int flattenedSize,
							    distributedCPUVec<double>  & tempEigenVec,
							    const unsigned int totalNumberWaveFunctions,
							    std::vector<double>        & eigenValues,
							    const MPI_Comm &interBandGroupComm,
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


     distributedGPUVec<double> YArrayCA;
    if (dftParameters::chebyCommunAvoidanceAlgo) 
         YArrayCA.reinit(cudaFlattenedArrayBlock);

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
    if (dftParameters::overlapComputeCommunCheby || dftParameters::chebyCommunAvoidanceAlgo)
           cudaFlattenedArrayBlock2.reinit(cudaFlattenedArrayBlock);


    distributedGPUVec<double> YArray2;
    if (dftParameters::overlapComputeCommunCheby)
           YArray2.reinit(cudaFlattenedArrayBlock2);


    distributedGPUVec<double> projectorKetTimesVector2;
    if (dftParameters::overlapComputeCommunCheby)
           projectorKetTimesVector2.reinit(projectorKetTimesVector);

    computing_timer.enter_section("Lanczos k-step Upper Bound");
    operatorMatrix.reinit(1);
    const double upperBoundUnwantedSpectrum =linearAlgebraOperationsCUDA::lanczosUpperBoundEigenSpectrum(operatorMatrix,
												      tempEigenVec);
    computing_timer.exit_section("Lanczos k-step Upper Bound");
    unsigned int chebyshevOrder = dftParameters::chebyshevOrder;

    //
    //set Chebyshev order
    //
    if(chebyshevOrder == 0)
      chebyshevOrder=internal::setChebyshevOrder(upperBoundUnwantedSpectrum);

    chebyshevOrder = (dftParameters::isPseudopotential)?chebyshevOrder*dftParameters::chebyshevFilterPolyDegreeFirstScfScalingFactor:chebyshevOrder;


    if(dftParameters::lowerBoundUnwantedFracUpper > 1e-6)
      d_lowerBoundUnWantedSpectrum=dftParameters::lowerBoundUnwantedFracUpper*upperBoundUnwantedSpectrum;

    //
    //output statements
    //
    if (dftParameters::verbosity>=2)
      {
	char buffer[100];

	sprintf(buffer, "%s:%18.10e\n", "upper bound of unwanted spectrum", upperBoundUnwantedSpectrum);
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

      const unsigned int M = localVectorSize;
      const unsigned int N = totalNumberWaveFunctions;

      unsigned int vectorsBlockSize=std::min(dftParameters::wfcBlockSize,
                                                   N);

      const unsigned int maxNumLocalDofs=dealii::Utilities::MPI::max(M,operatorMatrix.getMPICommunicator());

      const unsigned int dofsBlockSize=std::min(maxNumLocalDofs,
                                                dftParameters::subspaceRotDofsBlockSize);

      const float scalarCoeffAlphaSP = 1.0,scalarCoeffBetaSP = 0.0;

     thrust::device_vector<float> rotationMatBlockSP(vectorsBlockSize*N,0.0);
     thrust::device_vector<float> rotatedVectorsMatBlockSP(vectorsBlockSize*dofsBlockSize,0.0);

     float * rotationMatBlockHostSP;
     cudaMallocHost((void **)&rotationMatBlockHostSP,vectorsBlockSize*N*sizeof(float));
     std::memset(rotationMatBlockHostSP,0,vectorsBlockSize*N*sizeof(float));

     double * diagValuesHost;
     cudaMallocHost((void **)&diagValuesHost,N*sizeof(double));


     const unsigned int MPadded=std::ceil(M*1.0/8.0)*8.0+0.5;
     thrust::device_vector<float> XSP(MPadded*N,0.0);


     std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid> processGrid;

     const unsigned int rowsBlockSize=operatorMatrix.getScalapackBlockSize();
     linearAlgebraOperationsCUDA::internal::createProcessGridSquareMatrix(operatorMatrix.getMPICommunicator(),
                                                                          N,
                                                                          processGrid);

     dealii::ScaLAPACKMatrix<double> LMatPar(N,
                                             processGrid,
                                             rowsBlockSize);


     dealii::ScaLAPACKMatrix<double> overlapMatPar(N,
                                                   processGrid,
                                                   rowsBlockSize);



    for (unsigned int ipass = 0; ipass < numberPasses; ipass++)
    {
            if (this_process==0 && dftParameters::verbosity>=1)
               std::cout<<"Beginning no RR Chebyshev filter subpspace iteration pass: "<<ipass+1<<std::endl;


            double computeAvoidanceTolerance=1e-8;
            if (ipass>=2 && ipass<4)
               computeAvoidanceTolerance=1e-10;
            else if (ipass>=4)
               computeAvoidanceTolerance=1e-12;



	    convDoubleArrToFloatArr<<<(N+255)/256*M,256>>>(N*M,
							   eigenVectorsFlattenedCUDA,
							   thrust::raw_pointer_cast(&XSP[0]));


             if (dftParameters::gpuFineGrainedTimings)
             {
                cudaDeviceSynchronize();
                MPI_Barrier(MPI_COMM_WORLD);
                gpu_time = MPI_Wtime();
             }


             if(dftParameters::gpuFineGrainedTimings)
             {
               cudaDeviceSynchronize();
               MPI_Barrier(MPI_COMM_WORLD);
	       gpu_time = MPI_Wtime() - gpu_time;
	       if (this_process==0)
	          std::cout<<"Time for creating processGrid and ScaLAPACK matrix for pass: "<<ipass<<" is "<<gpu_time<<std::endl;
             }



	    if(ipass > 0)
	      {

                 if(dftParameters::gpuFineGrainedTimings)
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
			(eigenVectorsFlattenedCUDA,
			 M,
			 N,
			 cublasHandle,
			 operatorMatrix.getMPICommunicator(),
			 interBandGroupComm,
			 processGrid,
			 overlapMatPar);
		    else
		      linearAlgebraOperationsCUDA::
			fillParallelOverlapMatMixedPrecScalapack
			(eigenVectorsFlattenedCUDA,
			 M,
			 N,
			 cublasHandle,
			 operatorMatrix.getMPICommunicator(),
			 interBandGroupComm,
			 processGrid,
			 overlapMatPar);
		  }
		else
		  {
		    if(dftParameters::overlapComputeCommunOrthoRR)
		      linearAlgebraOperationsCUDA::
			fillParallelOverlapMatScalapackAsyncComputeCommun
			(eigenVectorsFlattenedCUDA,
			 M,
			 N,
			 cublasHandle,
			 operatorMatrix.getMPICommunicator(),
			 interBandGroupComm,
			 processGrid,
			 overlapMatPar); 
		    else
		      linearAlgebraOperationsCUDA::
			fillParallelOverlapMatScalapack
			(eigenVectorsFlattenedCUDA,
			 M,
			 N,
			 cublasHandle,
			 operatorMatrix.getMPICommunicator(),
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
 
                if(dftParameters::gpuFineGrainedTimings)  
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


                LMatPar.set_property(overlapMatPropertyPostCholesky);


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
	     
               if(dftParameters::gpuFineGrainedTimings)
               {
                 cudaDeviceSynchronize();
                 MPI_Barrier(MPI_COMM_WORLD);
                 gpu_time = MPI_Wtime() - gpu_time;
                 if(this_process==0)
                  std::cout<<"Time for PGS Cholesky Triangular Mat inverse ScaLAPACK (option 0): "<<gpu_time<<std::endl;
               }

            }//end of iPass

	   //start of subspace rotation procedure
           std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
	   std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
	   linearAlgebraOperationsCUDA::internal::createGlobalToLocalIdMapsScaLAPACKMat(processGrid,
											LMatPar,
											globalToLocalRowIdMap,
											globalToLocalColumnIdMap);

          thrust::device_vector<double> diagValues(N,0.0);
          std::memset(diagValuesHost,0,N*sizeof(double));


           if(ipass > 0)
            {
		//Extract DiagQ from parallel ScaLAPACK matrix Q
		if(LMatPar.get_property()==dealii::LAPACKSupport::Property::upper_triangular)
		  {
		    if (processGrid->is_process_active())
		      for (unsigned int i = 0; i <N; ++i)
			if (globalToLocalRowIdMap.find(i)
			    !=globalToLocalRowIdMap.end())
			  {
			    const unsigned int localRowId=globalToLocalRowIdMap[i];
			    std::map<unsigned int, unsigned int>::iterator it=
			      globalToLocalColumnIdMap.find(i);
			    if (it!=globalToLocalColumnIdMap.end())
			      {
				diagValuesHost[i]=LMatPar.local_el(localRowId,
									  it->second);
			      }
			  }
		  }
		else
		  {
		    if (processGrid->is_process_active())
		      for (unsigned int i = 0; i <N; ++i)
			if(globalToLocalColumnIdMap.find(i)
			   !=globalToLocalColumnIdMap.end())
			  {
			    const unsigned int localColumnId=globalToLocalColumnIdMap[i];
			    std::map<unsigned int, unsigned int>::iterator it=
			      globalToLocalRowIdMap.find(i);
			    if (globalToLocalRowIdMap.find(i)!=globalToLocalRowIdMap.end())
			      {
				diagValuesHost[i]
				  =LMatPar.local_el(it->second,
							   localColumnId);
			      }
			  }
		  }

		MPI_Allreduce(MPI_IN_PLACE,
			      diagValuesHost,
			      N,
			      MPI_DOUBLE,
			      MPI_SUM,
			      operatorMatrix.getMPICommunicator());

		cudaMemcpy(thrust::raw_pointer_cast(&diagValues[0]),
			   diagValuesHost,
			   N*sizeof(double),
			   cudaMemcpyHostToDevice);

		computeDiagQTimesXKernel<<<(M*N+255)/256,256>>>(thrust::raw_pointer_cast(&diagValues[0]),
								eigenVectorsFlattenedCUDA,
								N,
								M);
	    
	    

	      }//end of iPass condition

	    
	    //think you have to modify ivec limits
	    unsigned int iVecLimit = vectorsBlockSize;
	    	   
	      
            for (unsigned int ivec = 0; ivec < totalNumberWaveFunctions+iVecLimit; ivec += vectorsBlockSize)
            {

	      // Correct block dimensions if block "goes off edge of" the matrix
	      const unsigned int BVecSR = std::min(vectorsBlockSize, N-ivec);
	      const unsigned int D=ivec+BVecSR;
              MPI_Request requestForExtractingLMatInv;

	      if(ipass > 0 && ivec < totalNumberWaveFunctions)
		{
                 if(dftParameters::verbosity >= 4)
                     pcout<<" Begin extraction of inverse Cholesky factor for the WFC BLOCK : "<<ivec<<std::endl;

		  std::memset(rotationMatBlockHostSP,0,BVecSR*N*sizeof(float));
		  
		  //Extract QBVec from parallel ScaLAPACK matrix Q
		  if(LMatPar.get_property()==dealii::LAPACKSupport::Property::upper_triangular)
		    {
		      if (processGrid->is_process_active())
			for (unsigned int i = 0; i <D; ++i)
			  if (globalToLocalRowIdMap.find(i)
			      !=globalToLocalRowIdMap.end())
			    {
			      const unsigned int localRowId=globalToLocalRowIdMap[i];
			      for (unsigned int j = 0; j <BVecSR; ++j)
				{
				  std::map<unsigned int, unsigned int>::iterator it=
				    globalToLocalColumnIdMap.find(j+ivec);
				  if(it!=globalToLocalColumnIdMap.end())
				    {
				      rotationMatBlockHostSP[i*BVecSR+j]=
					LMatPar.local_el(localRowId,
								it->second);
				    }
				}

			      if (i>=ivec && i<(ivec+BVecSR))
				{
				  std::map<unsigned int, unsigned int>::iterator it=
				    globalToLocalColumnIdMap.find(i);
				  if (it!=globalToLocalColumnIdMap.end())
				    {
				      rotationMatBlockHostSP[i*BVecSR+i-ivec]=0.0;
				    }
				}
			    }
		    }
		  else
		    {
		      if (processGrid->is_process_active())
			for (unsigned int i = 0; i <D; ++i)
			  if(globalToLocalColumnIdMap.find(i)
			     !=globalToLocalColumnIdMap.end())
			    {
			      const unsigned int localColumnId=globalToLocalColumnIdMap[i];
			      for (unsigned int j = 0; j <BVecSR; ++j)
				{
				  std::map<unsigned int, unsigned int>::iterator it=
				    globalToLocalRowIdMap.find(j+ivec);
				  if (it!=globalToLocalRowIdMap.end())
				    {
				      rotationMatBlockHostSP[i*BVecSR+j]=
					LMatPar.local_el(it->second,
								localColumnId);
				    }
				}

			      if (i>=ivec && i<(ivec+BVecSR))
				{
				  std::map<unsigned int, unsigned int>::iterator it=
				    globalToLocalRowIdMap.find(i);
				  if (globalToLocalRowIdMap.find(i)!=globalToLocalRowIdMap.end())
				    {
				      rotationMatBlockHostSP[i*BVecSR+i-ivec]=0.0;
				    }
				}
			    }
		    }

		  //call MPI_IAllReduce later
		  if(ivec == 0)
		    {
		      MPI_Allreduce(MPI_IN_PLACE,
				    rotationMatBlockHostSP,
				    BVecSR*D,
				    MPI_FLOAT,
				    MPI_SUM,
				    operatorMatrix.getMPICommunicator());
		    }
                  else
                   {
                     MPI_Iallreduce(MPI_IN_PLACE,
                                    rotationMatBlockHostSP,
                                    BVecSR*D,
                                    MPI_FLOAT,
                                    MPI_SUM,
                                    operatorMatrix.getMPICommunicator(),
                                    &requestForExtractingLMatInv);
                   }


                  if(dftParameters::verbosity >= 4)
                     pcout<<" End extraction of inverse Cholesky factor for the WFC BLOCK : "<<ivec<<std::endl;

		}//end of iPass>0
               
		    //two blocks of wavefunctions are filtered simultaneously when overlap compute communication in chebyshev
		    //filtering is toggled on
		    const unsigned int numSimultaneousBlocks=dftParameters::overlapComputeCommunCheby?2:1;
		    unsigned int numSimultaneousBlocksCurrent=numSimultaneousBlocks;
		    const unsigned int numWfcsInBandGroup= totalNumberWaveFunctions;//bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1]-bandGroupLowHighPlusOneIndices[2*bandGroupTaskId];
		    unsigned int jvecInit = ivec;
		    if(ivec > 0)
		      jvecInit = ivec - vectorsBlockSize;
		    
		    if((ivec > 0 && ipass < numberPasses - 1))
		      {
                        if(dftParameters::verbosity >= 4)
                           pcout<<" Begin ChebyShev Filtering of WFC BLOCK : "<<jvecInit<<std::endl;

			for (unsigned int jvec = jvecInit; jvec < (jvecInit+wfcBlockSize); jvec += numSimultaneousBlocksCurrent*chebyBlockSize)
			  {
                              if(dftParameters::verbosity >= 4)
                               pcout<<" Begin ChebyShev Filtering of CHEB BLOCK : "<<jvec<<std::endl;
                            
			    // Correct block dimensions if block "goes off edge of" the matrix
			    const unsigned int BVec = chebyBlockSize;//std::min(vectorsBlockSize, totalNumberWaveFunctions-jvec);
		      
			    //handle edge case when total number of blocks in a given band group is not even in case of 
			    //overlapping computation and communciation in chebyshev filtering 
			    const unsigned int leftIndexBandGroupMargin=(jvec/numWfcsInBandGroup)*numWfcsInBandGroup;
			    numSimultaneousBlocksCurrent
			      =((jvec+numSimultaneousBlocks*BVec-leftIndexBandGroupMargin)<=numWfcsInBandGroup && numSimultaneousBlocks==2)?2:1;

			    //if ((jvec+numSimultaneousBlocksCurrent*BVec)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
			    //(jvec+numSimultaneousBlocksCurrent*BVec)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
			    if((jvec+numSimultaneousBlocksCurrent*BVec)<=totalNumberWaveFunctions &&
			       (jvec+numSimultaneousBlocksCurrent*BVec)> 0)
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
                                      if (dftParameters::chebyCommunAvoidanceAlgo)
                                         linearAlgebraOperationsCUDA::chebyshevFilterComputeAvoidance(operatorMatrix,
                                                                                      cudaFlattenedArrayBlock,
                                                                                      YArray,
                                                                                      YArrayCA,
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
                                                                                      upperBoundUnwantedSpectrum,
                                                                                      d_lowerBoundWantedSpectrum,
                                                                                      isXlBOMDLinearizedSolve,
                                                                                      false,
                                                                                      useMixedPrecOverall,
                                                                                      computeAvoidanceTolerance);
                                        else
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
                                                                                      upperBoundUnwantedSpectrum,
                                                                                      d_lowerBoundWantedSpectrum,
                                                                                      useMixedPrecOverall);
                                 }
                                 else
                                 {
                                   if (dftParameters::chebyCommunAvoidanceAlgo)
                                         linearAlgebraOperationsCUDA::chebyshevFilterComputeAvoidance(operatorMatrix,
                                                                                           cudaFlattenedArrayBlock,
                                                                                           YArray,
                                                                                           cudaFlattenedArrayBlock2,
                                                                                           cudaFlattenedFloatArrayBlock,
                                                                                           projectorKetTimesVector,
                                                                                           localVectorSize,
                                                                                           BVec,
                                                                                           chebyshevOrder,
                                                                                           d_lowerBoundUnWantedSpectrum,
                                                                                           upperBoundUnwantedSpectrum,
                                                                                           d_lowerBoundWantedSpectrum,
                                                                                           isXlBOMDLinearizedSolve,
                                                                                           false,
                                                                                           useMixedPrecOverall);
                                     else
                                        linearAlgebraOperationsCUDA::chebyshevFilter(operatorMatrix,
                                                                                           cudaFlattenedArrayBlock,
                                                                                           YArray,
                                                                                           cudaFlattenedFloatArrayBlock,
                                                                                           projectorKetTimesVector,
                                                                                           localVectorSize,
                                                                                           BVec,
                                                                                           chebyshevOrder,
                                                                                           d_lowerBoundUnWantedSpectrum,
                                                                                           upperBoundUnwantedSpectrum,
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

                       if(dftParameters::verbosity >= 4)
                           pcout<<" End ChebyShev Filtering of WFC BLOCK : "<<jvecInit<<std::endl;

		      }//if(ivec > 0 || iPass == 0) condition


		    if(ipass > 0 && ivec < totalNumberWaveFunctions)
		      {
                        if(dftParameters::verbosity >= 4)
                           pcout<<" Begin compute operation of subspace rotation in WFC BLOCK : "<<ivec<<std::endl;
                        
                        if(ivec > 0)
                          MPI_Wait(&requestForExtractingLMatInv,MPI_STATUS_IGNORE);

			cudaMemcpy(thrust::raw_pointer_cast(&rotationMatBlockSP[0]),
				   rotationMatBlockHostSP,
				   BVecSR*D*sizeof(float),
				   cudaMemcpyHostToDevice);

			for (unsigned int idof = 0; idof < maxNumLocalDofs; idof += dofsBlockSize)
			  {

			    // Correct block dimensions if block "goes off edge of" the matrix
			    unsigned int BDof=0;
			    if (M>=idof)
			      BDof = std::min(dofsBlockSize, M-idof);

			    if (BDof!=0)
			      {
				cublasSgemm(cublasHandle,
					    CUBLAS_OP_N,
					    CUBLAS_OP_N,
					    BVecSR,
					    BDof,
					    D,
					    &scalarCoeffAlphaSP,
					    thrust::raw_pointer_cast(&rotationMatBlockSP[0]),
					    BVecSR,
					    thrust::raw_pointer_cast(&XSP[0])+idof*N,
					    N,
					    &scalarCoeffBetaSP,
					    thrust::raw_pointer_cast(&rotatedVectorsMatBlockSP[0]),
					    BVecSR);

				
				addSubspaceRotatedBlockToXKernel<<<(BVecSR*BDof+255)/256,256>>>(BDof,
											      BVecSR,
											      thrust::raw_pointer_cast(&rotatedVectorsMatBlockSP[0]),
											      eigenVectorsFlattenedCUDA,
											      idof,
											      ivec,
											      N);
			      }
			  }//block loop over dofs
                          
                           if(dftParameters::verbosity >= 4)
                             pcout<<" End compute operation of subspace rotation in WFC BLOCK : "<<ivec<<std::endl;

		      }
	    }//wfc block loop

	    if(dftParameters::verbosity >= 4)
	      pcout<<" ChebyShev Filtering with Orthogonalization overlap Done: "<<std::endl;

	    //copy eigenVectorsFlattenedCUDANext to eigenVectorsFlattenedCUDA after cheb filtering of all the wavefunctions


	    /*linearAlgebraOperationsCUDA::pseudoGramSchmidtOrthogonalization
		     (operatorMatrix,
		      eigenVectorsFlattenedCUDA,
		      localVectorSize,
		      totalNumberWaveFunctions,
		      operatorMatrix.getMPICommunicator(),
		      interBandGroupComm,
		      cublasHandle,
		      useMixedPrecOverall);*/

     //cudaFreeHost(diagValuesHost);
     //cudaFreeHost(rotationMatBlockHostSP);
    }
    cudaFreeHost(diagValuesHost);
    cudaFreeHost(rotationMatBlockHostSP);
    

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
