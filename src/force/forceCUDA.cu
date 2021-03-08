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

//source file for force related computations

#include <forceCUDA.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <constants.h>
#include <vectorUtilities.h>

namespace dftfe
{
	namespace forceCUDA
	{

		namespace
		{

			__global__
				void stridedCopyToBlockKernel(const unsigned int BVec,
						const double *xVec,
						const unsigned int M,
						const unsigned int N,
						double * yVec,
						const unsigned int startingXVecId)
				{

					const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
					const unsigned int numberEntries = M*BVec;

					for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
					{
						unsigned int blockIndex = index/BVec;
						unsigned int intraBlockIndex=index-blockIndex*BVec;
						yVec[index]
							=xVec[blockIndex*N+startingXVecId+intraBlockIndex];
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
						unsigned int intraBlockIndex=index-blockIndex*contiguousBlockSize;
						copyToVec[index]
							=copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex]+intraBlockIndex];
					}

				}


			__global__
				void computeELocWfcEshelbyTensorContributions(const unsigned int contiguousBlockSize,
						const unsigned int numContiguousBlocks,
						const unsigned int numQuads,
						const double * psiQuadValues,
						const double * gradPsiQuadValuesX,
						const double * gradPsiQuadValuesY,
						const double * gradPsiQuadValuesZ,
						const double * eigenValues,
						const double * partialOccupancies,
						double *eshelbyTensor)
				{

					const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
					const unsigned int numberEntries = numContiguousBlocks*contiguousBlockSize;

					for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
					{
						const unsigned int blockIndex = index/contiguousBlockSize;
						const unsigned int intraBlockIndex=index-blockIndex*contiguousBlockSize;
						const unsigned int blockIndex2=blockIndex/6;
						const unsigned int eshelbyIndex=blockIndex-6*blockIndex2;
						const unsigned int cellIndex=blockIndex2/numQuads;
						const unsigned int quadId=blockIndex2-cellIndex*numQuads;
						const unsigned int tempIndex=(cellIndex)*numQuads*contiguousBlockSize+quadId*contiguousBlockSize+intraBlockIndex;
						const double psi=psiQuadValues[tempIndex];
						const double gradPsiX=gradPsiQuadValuesX[tempIndex];
						const double gradPsiY=gradPsiQuadValuesY[tempIndex];
						const double gradPsiZ=gradPsiQuadValuesZ[tempIndex];
						const double eigenValue=eigenValues[intraBlockIndex];
						const double partOcc=partialOccupancies[intraBlockIndex];

						const double identityFactor=partOcc*(gradPsiX*gradPsiX+gradPsiY*gradPsiY+gradPsiZ*gradPsiZ)-2.0*partOcc*eigenValue*psi*psi;

						if (eshelbyIndex==0)
							eshelbyTensor[index]=-2.0*partOcc*gradPsiX*gradPsiX+identityFactor;
						else if (eshelbyIndex==1)
							eshelbyTensor[index]=-2.0*partOcc*gradPsiY*gradPsiX;
						else if (eshelbyIndex==2)
							eshelbyTensor[index]=-2.0*partOcc*gradPsiY*gradPsiY+identityFactor;
						else if (eshelbyIndex==3)
							eshelbyTensor[index]=-2.0*partOcc*gradPsiZ*gradPsiX;
						else if (eshelbyIndex==4)
							eshelbyTensor[index]=-2.0*partOcc*gradPsiZ*gradPsiY;
						else if (eshelbyIndex==5)
							eshelbyTensor[index]=-2.0*partOcc*gradPsiZ*gradPsiZ+identityFactor;
					}

				}


			__global__
				void nlpPsiContractionCUDAKernel(const unsigned int numPsi,
						const unsigned int numQuadsNLP, 
						const unsigned int totalNonTrivialPseudoWfcs,
						const unsigned int startingId,
						const double * projectorKetTimesVectorPar,
						const double * psiQuadValuesNLP,
						const double * partialOccupancies,
						const unsigned int * nonTrivialIdToElemIdMap,
						const unsigned int * projecterKetTimesFlattenedVectorLocalIds,
						double *nlpContractionContribution)
				{

					const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
					const unsigned int numberEntries = totalNonTrivialPseudoWfcs*numQuadsNLP*numPsi;

					for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
					{
						const unsigned int blockIndex = index/numPsi;
						const unsigned int wfcId=index-blockIndex*numPsi;
						unsigned int pseudoWfcId=blockIndex/numQuadsNLP;
						const unsigned int quadId=blockIndex-pseudoWfcId*numQuadsNLP;
						pseudoWfcId+=startingId;
						nlpContractionContribution[index]=partialOccupancies[wfcId]*psiQuadValuesNLP[nonTrivialIdToElemIdMap[pseudoWfcId]*numQuadsNLP*numPsi+quadId*numPsi+wfcId]*projectorKetTimesVectorPar[projecterKetTimesFlattenedVectorLocalIds[pseudoWfcId]*numPsi+wfcId];
					}

				}

		}


		void computeNonLocalProjectorKetTimesPsiTimesVH(operatorDFTCUDAClass & operatorMatrix,
				const double * X,
				const unsigned int startingVecId,
				const unsigned int BVec,
				const unsigned int N,
				double * projectorKetTimesPsiTimesVH)
		{

			distributedGPUVec<double> cudaFlattenedArrayBlock;
			vectorTools::createDealiiVector(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
					BVec,
					cudaFlattenedArrayBlock);


			distributedGPUVec<double> projectorKetTimesVector;
			vectorTools::createDealiiVector(operatorMatrix.getProjectorKetTimesVectorSingle().get_partitioner(),
					BVec,
					projectorKetTimesVector);


			const unsigned int M=operatorMatrix.getMatrixFreeData()->get_vector_partitioner()->local_size();
			stridedCopyToBlockKernel<<<(BVec+255)/256*M, 256>>>(BVec,
					X,
					M,
					N,
					cudaFlattenedArrayBlock.begin(),
					startingVecId);
			cudaFlattenedArrayBlock.update_ghost_values();

			(operatorMatrix.getOverloadedConstraintMatrix())->distribute(cudaFlattenedArrayBlock,
					BVec);

			operatorMatrix.computeNonLocalProjectorKetTimesXTimesV(cudaFlattenedArrayBlock.begin(),
					projectorKetTimesVector,
					BVec);


			const unsigned int totalSize=projectorKetTimesVector.get_partitioner()->n_ghost_indices()+projectorKetTimesVector.local_size();

			cudaMemcpy(projectorKetTimesPsiTimesVH,
					projectorKetTimesVector.begin(),
					totalSize*sizeof(double),
					cudaMemcpyDeviceToHost);  
		}


		void interpolatePsiComputeELocWfcEshelbyTensorNonPeriodicD(operatorDFTCUDAClass & operatorMatrix,
				distributedGPUVec<double> & Xb,
				const unsigned int BVec,
				const unsigned int numCells,
				const unsigned int numQuads,
				const unsigned int numQuadsNLP,
				const unsigned int numNodesPerElement,
				const thrust::device_vector<double> & eigenValuesD,
				const thrust::device_vector<double> & partialOccupanciesD,
				const thrust::device_vector<double> & onesVecD,
				const unsigned int cellsBlockSize,
				thrust::device_vector<double> & psiQuadsFlatD,
				thrust::device_vector<double> & gradPsiQuadsXFlatD,
				thrust::device_vector<double> & gradPsiQuadsYFlatD,
				thrust::device_vector<double> & gradPsiQuadsZFlatD,
				thrust::device_vector<double> & gradPsiQuadsNLPFlatD,
				thrust::device_vector<double> & eshelbyTensorContributionsD,
				thrust::device_vector<double> & eshelbyTensorQuadValuesD)
		{
			thrust::device_vector<double> & cellWaveFunctionMatrix = operatorMatrix.getCellWaveFunctionMatrix();

			copyCUDAKernel<<<(BVec+255)/256*numCells*numNodesPerElement,256>>>
				(BVec,
				 numCells*numNodesPerElement,
				 Xb.begin(),
				 thrust::raw_pointer_cast(&cellWaveFunctionMatrix[0]),
				 thrust::raw_pointer_cast(&(operatorMatrix.getFlattenedArrayCellLocalProcIndexIdMap())[0]));

			const int blockSize=cellsBlockSize;
			const int numberBlocks=numCells/blockSize;
			const int remBlockSize=numCells-numberBlocks*blockSize;

			thrust::device_vector<double> shapeFunctionGradientValuesNLPReferenceD(blockSize*numQuadsNLP*3*numNodesPerElement,0.0);
			thrust::device_vector<double> shapeFunctionGradientValuesNLPD(blockSize*numQuadsNLP*3*numNodesPerElement,0.0);

      for (unsigned int i=0; i<blockSize;i++)
        thrust::copy(operatorMatrix.getShapeFunctionGradientValuesNLPInverted().begin(), operatorMatrix.getShapeFunctionGradientValuesNLPInverted().end(), shapeFunctionGradientValuesNLPReferenceD.begin()+i*numQuadsNLP*3*numNodesPerElement);
			
      for (int iblock=0; iblock<(numberBlocks+1); iblock++)
			{
				const int currentBlockSize= (iblock==numberBlocks)?remBlockSize:blockSize;
				const int startingId=iblock*blockSize;

				if (currentBlockSize>0)
				{
          double scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;
          int strideA = BVec*numNodesPerElement;
          int strideB = 0;
          int strideC = BVec*numQuads;

          cublasDgemmStridedBatched(operatorMatrix.getCublasHandle(),
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              BVec,
              numQuads,
              numNodesPerElement,
              &scalarCoeffAlpha,
              thrust::raw_pointer_cast(&cellWaveFunctionMatrix[startingId*numNodesPerElement*BVec]),
              BVec,
              strideA,
              thrust::raw_pointer_cast(&(operatorMatrix.getShapeFunctionValuesInverted())[0]),
              numNodesPerElement,
              strideB,
              &scalarCoeffBeta,
              thrust::raw_pointer_cast(&psiQuadsFlatD[0]),
              BVec,
              strideC,
              currentBlockSize);

          strideB=numNodesPerElement*numQuads;

          cublasDgemmStridedBatched(operatorMatrix.getCublasHandle(),
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              BVec,
              numQuads,
              numNodesPerElement,
              &scalarCoeffAlpha,
              thrust::raw_pointer_cast(&cellWaveFunctionMatrix[startingId*numNodesPerElement*BVec]),
              BVec,
              strideA,
              thrust::raw_pointer_cast(&(operatorMatrix.getShapeFunctionGradientValuesXInverted())[startingId*numQuads*numNodesPerElement]),
              numNodesPerElement,
              strideB,
              &scalarCoeffBeta,
              thrust::raw_pointer_cast(&gradPsiQuadsXFlatD[0]),
              BVec,
              strideC,
              currentBlockSize);


          cublasDgemmStridedBatched(operatorMatrix.getCublasHandle(),
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              BVec,
              numQuads,
              numNodesPerElement,
              &scalarCoeffAlpha,
              thrust::raw_pointer_cast(&cellWaveFunctionMatrix[startingId*numNodesPerElement*BVec]),
              BVec,
              strideA,
              thrust::raw_pointer_cast(&(operatorMatrix.getShapeFunctionGradientValuesYInverted())[startingId*numQuads*numNodesPerElement]),
              numNodesPerElement,
              strideB,
              &scalarCoeffBeta,
              thrust::raw_pointer_cast(&gradPsiQuadsYFlatD[0]),
              BVec,
              strideC,
              currentBlockSize);

          cublasDgemmStridedBatched(operatorMatrix.getCublasHandle(),
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              BVec,
              numQuads,
              numNodesPerElement,
              &scalarCoeffAlpha,
              thrust::raw_pointer_cast(&cellWaveFunctionMatrix[startingId*numNodesPerElement*BVec]),
              BVec,
              strideA,
              thrust::raw_pointer_cast(&(operatorMatrix.getShapeFunctionGradientValuesZInverted())[startingId*numQuads*numNodesPerElement]),
              numNodesPerElement,
              strideB,
              &scalarCoeffBeta,
              thrust::raw_pointer_cast(&gradPsiQuadsZFlatD[0]),
              BVec,
              strideC,
              currentBlockSize);

          //shapeGradRef^T*invJacobian^T
          cublasDgemmStridedBatched(operatorMatrix.getCublasHandle(),
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              numNodesPerElement,
              3,
              3,
              &scalarCoeffAlpha,
              thrust::raw_pointer_cast(&shapeFunctionGradientValuesNLPReferenceD[0]),
              numNodesPerElement,
              numNodesPerElement*3,
              thrust::raw_pointer_cast(&(operatorMatrix.getInverseJacobiansNLP())[startingId*numQuadsNLP*3*3]),
              3,
              3*3,
              &scalarCoeffBeta,
              thrust::raw_pointer_cast(&shapeFunctionGradientValuesNLPD[0]),
              numNodesPerElement,
              numNodesPerElement*3,
              currentBlockSize*numQuadsNLP);  
          

          const int strideCNLPGrad = BVec*3*numQuadsNLP;
          const int strideBNLPGrad=numNodesPerElement*3*numQuadsNLP;
          
          cublasDgemmStridedBatched(operatorMatrix.getCublasHandle(),
              CUBLAS_OP_N,
              CUBLAS_OP_N,
              BVec,
              3*numQuadsNLP,
              numNodesPerElement,
              &scalarCoeffAlpha,
              thrust::raw_pointer_cast(&cellWaveFunctionMatrix[startingId*numNodesPerElement*BVec]),
              BVec,
              strideA,
              thrust::raw_pointer_cast(&shapeFunctionGradientValuesNLPD[0]),
              numNodesPerElement,
              strideBNLPGrad,
              &scalarCoeffBeta,
              thrust::raw_pointer_cast(&gradPsiQuadsNLPFlatD[startingId*numQuadsNLP*3*BVec]),
              BVec,
              strideCNLPGrad,
              currentBlockSize);
          
					computeELocWfcEshelbyTensorContributions<<<(BVec+255)/256*currentBlockSize*numQuads*6,256>>>
						(BVec,
						 currentBlockSize*numQuads*6,
						 numQuads,
						 thrust::raw_pointer_cast(&psiQuadsFlatD[0]),
						 thrust::raw_pointer_cast(&gradPsiQuadsXFlatD[0]),
						 thrust::raw_pointer_cast(&gradPsiQuadsYFlatD[0]),
						 thrust::raw_pointer_cast(&gradPsiQuadsZFlatD[0]),
						 thrust::raw_pointer_cast(&eigenValuesD[0]),
						 thrust::raw_pointer_cast(&partialOccupanciesD[0]),
						 thrust::raw_pointer_cast(&eshelbyTensorContributionsD[0]));

					scalarCoeffAlpha = 1.0;
					scalarCoeffBeta = 1.0;



					cublasDgemm(operatorMatrix.getCublasHandle(),
							CUBLAS_OP_N,
							CUBLAS_OP_N,
							1,
							currentBlockSize*numQuads*6,
							BVec,
							&scalarCoeffAlpha,
							thrust::raw_pointer_cast(&onesVecD[0]),
							1,
							thrust::raw_pointer_cast(&eshelbyTensorContributionsD[0]),
							BVec,
							&scalarCoeffBeta,
							thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD[startingId*numQuads*6]),
							1);

				}
			}
		}

		void nlpPsiContractionD(operatorDFTCUDAClass & operatorMatrix,
				const thrust::device_vector<double> & gradPsiQuadsNLPD,
				const thrust::device_vector<double> & partialOccupanciesD,
				const thrust::device_vector<double> & onesVecD,
				const double * projectorKetTimesVectorParFlattenedD,
				const thrust::device_vector<unsigned int> & nonTrivialIdToElemIdMapD,
				const thrust::device_vector<unsigned int> & projecterKetTimesFlattenedVectorLocalIdsD,
				const unsigned int numCells, 
				const unsigned int numQuadsNLP,
				const unsigned int numPsi,
				const unsigned int totalNonTrivialPseudoWfcs,
				const unsigned int innerBlockSizeEnlp,
				thrust::device_vector<double> & nlpContractionContributionD,
				thrust::device_vector<double> & projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedD)
		{
			const int blockSizeNlp=innerBlockSizeEnlp;
			const int numberBlocksNlp=totalNonTrivialPseudoWfcs/blockSizeNlp;
			const int remBlockSizeNlp=totalNonTrivialPseudoWfcs-numberBlocksNlp*blockSizeNlp;

			double scalarCoeffAlphaNlp = 1.0,scalarCoeffBetaNlp = 1.0;      

			for (int iblocknlp=0; iblocknlp<(numberBlocksNlp+1); iblocknlp++)
			{
				const int currentBlockSizeNlp= (iblocknlp==numberBlocksNlp)?remBlockSizeNlp:blockSizeNlp;
				const int startingIdNlp=iblocknlp*blockSizeNlp;
				if (currentBlockSizeNlp>0)
				{
					nlpPsiContractionCUDAKernel<<<(numPsi+255)/256*numQuadsNLP*3*currentBlockSizeNlp,256>>>
						(numPsi,
						 numQuadsNLP*3,
						 currentBlockSizeNlp,
						 startingIdNlp,
						 projectorKetTimesVectorParFlattenedD,
						 thrust::raw_pointer_cast(&gradPsiQuadsNLPD[0]),
						 thrust::raw_pointer_cast(&partialOccupanciesD[0]),
						 thrust::raw_pointer_cast(&nonTrivialIdToElemIdMapD[0]),
						 thrust::raw_pointer_cast(&projecterKetTimesFlattenedVectorLocalIdsD[0]),
						 thrust::raw_pointer_cast(&nlpContractionContributionD[0]));

					cublasDgemm(operatorMatrix.getCublasHandle(),
							CUBLAS_OP_N,
							CUBLAS_OP_N,
							1,
							currentBlockSizeNlp*numQuadsNLP*3,
							numPsi,
							&scalarCoeffAlphaNlp,
							thrust::raw_pointer_cast(&onesVecD[0]),
							1,
							thrust::raw_pointer_cast(&nlpContractionContributionD[0]),
							numPsi,
							&scalarCoeffBetaNlp,
							thrust::raw_pointer_cast(&projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedD[startingIdNlp*numQuadsNLP*3]),
							1);
				}
			}
		}


		void gpuPortedForceKernelsAllD(operatorDFTCUDAClass & operatorMatrix,
				distributedGPUVec<double> & cudaFlattenedArrayBlock,
				distributedGPUVec<double> & projectorKetTimesVectorD,
				const double * X,
				const thrust::device_vector<double> & eigenValuesD,
				const thrust::device_vector<double> & partialOccupanciesD,
				const thrust::device_vector<double> & onesVecD,
				const thrust::device_vector<unsigned int> & nonTrivialIdToElemIdMapD,
				const thrust::device_vector<unsigned int> & projecterKetTimesFlattenedVectorLocalIdsD, 
				const unsigned int startingVecId,
				const unsigned int N,
				const unsigned int numPsi,
				const unsigned int numCells,
				const unsigned int numQuads,
				const unsigned int numQuadsNLP,
				const unsigned int numNodesPerElement,
				const unsigned int totalNonTrivialPseudoWfcs,
				thrust::device_vector<double> & psiQuadsFlatD,
				thrust::device_vector<double> & gradPsiQuadsXFlatD,
				thrust::device_vector<double> & gradPsiQuadsYFlatD,
				thrust::device_vector<double> & gradPsiQuadsZFlatD,
				thrust::device_vector<double> & gradPsiQuadsNLPFlatD,
				thrust::device_vector<double> & eshelbyTensorContributionsD,
				thrust::device_vector<double> & eshelbyTensorQuadValuesD,
				thrust::device_vector<double> & nlpContractionContributionD,
				thrust::device_vector<double> & projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedD,
				const unsigned int cellsBlockSize,
				const unsigned int innerBlockSizeEnlp,
				const bool isPsp)
				{

					int this_process;
					MPI_Comm_rank(MPI_COMM_WORLD, &this_process);

					const unsigned int M=operatorMatrix.getMatrixFreeData()->get_vector_partitioner()->local_size();
					stridedCopyToBlockKernel<<<(numPsi+255)/256*M, 256>>>(numPsi,
							X,
							M,
							N,
							cudaFlattenedArrayBlock.begin(),
							startingVecId);
					cudaFlattenedArrayBlock.update_ghost_values();

					(operatorMatrix.getOverloadedConstraintMatrix())->distribute(cudaFlattenedArrayBlock,
							numPsi);


					//cudaDeviceSynchronize();
					//MPI_Barrier(MPI_COMM_WORLD);
					//double kernel1_time = MPI_Wtime();

					interpolatePsiComputeELocWfcEshelbyTensorNonPeriodicD(operatorMatrix,
							cudaFlattenedArrayBlock,
							numPsi,
							numCells,
							numQuads,
							numQuadsNLP,
							numNodesPerElement,
							eigenValuesD,
							partialOccupanciesD,
							onesVecD,
							cellsBlockSize,
							psiQuadsFlatD,
							gradPsiQuadsXFlatD,
							gradPsiQuadsYFlatD,
							gradPsiQuadsZFlatD,
				      gradPsiQuadsNLPFlatD,
							eshelbyTensorContributionsD,
							eshelbyTensorQuadValuesD);

					//cudaDeviceSynchronize();
					//MPI_Barrier(MPI_COMM_WORLD);
					//kernel1_time = MPI_Wtime() - kernel1_time;

					//if (this_process==0 && dftParameters::verbosity>=5)
					//	 std::cout<<"Time for interpolatePsiComputeELocWfcEshelbyTensorNonPeriodicD inside blocked loop: "<<kernel1_time<<std::endl;

					if (isPsp)
					{
						//cudaDeviceSynchronize();
						//MPI_Barrier(MPI_COMM_WORLD);
						//double kernel2_time = MPI_Wtime();

						operatorMatrix.computeNonLocalProjectorKetTimesXTimesV(cudaFlattenedArrayBlock.begin(),
								projectorKetTimesVectorD,
								numPsi);

						//cudaDeviceSynchronize();
						//MPI_Barrier(MPI_COMM_WORLD);
						//kernel2_time = MPI_Wtime() - kernel2_time;

						//if (this_process==0 && dftParameters::verbosity>=5)
						//  std::cout<<"Time for computeNonLocalProjectorKetTimesXTimesV inside blocked loop: "<<kernel2_time<<std::endl;

						//cudaDeviceSynchronize();
						//MPI_Barrier(MPI_COMM_WORLD);
						//double kernel3_time = MPI_Wtime();

						if (totalNonTrivialPseudoWfcs>0)
						{
							nlpPsiContractionD(operatorMatrix,
									gradPsiQuadsNLPFlatD,
									partialOccupanciesD,
									onesVecD,
									projectorKetTimesVectorD.begin(),
									nonTrivialIdToElemIdMapD,
									projecterKetTimesFlattenedVectorLocalIdsD,
									numCells, 
									numQuadsNLP,
									numPsi,
									totalNonTrivialPseudoWfcs,
									innerBlockSizeEnlp,
									nlpContractionContributionD,
									projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedD);
						}

						//cudaDeviceSynchronize();
						//MPI_Barrier(MPI_COMM_WORLD);
						//kernel3_time = MPI_Wtime() - kernel3_time;

						//if (this_process==0 && dftParameters::verbosity>=5)
						//	 std::cout<<"Time for nlpPsiContractionD inside blocked loop: "<<kernel3_time<<std::endl;
					}
				}

		void gpuPortedForceKernelsAllH(operatorDFTCUDAClass & operatorMatrix,
				const double * X,
				const double * eigenValuesH,
				const double  fermiEnergy,
				const unsigned int * nonTrivialIdToElemIdMapH,
				const unsigned int * projecterKetTimesFlattenedVectorLocalIdsH, 
				const unsigned int N,
				const unsigned int numCells,
				const unsigned int numQuads,
				const unsigned int numQuadsNLP,
				const unsigned int numNodesPerElement,
				const unsigned int totalNonTrivialPseudoWfcs,
				double * eshelbyTensorQuadValuesH,
				double * projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
				const MPI_Comm & interBandGroupComm,
				const bool isPsp)
		{
			//band group parallelization data structures
			const unsigned int numberBandGroups=
				dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
			const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
			std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
			dftUtils::createBandParallelizationIndices(interBandGroupComm,
					N,
					bandGroupLowHighPlusOneIndices);

			const unsigned int blockSize=std::min(dftParameters::chebyWfcBlockSize,
					bandGroupLowHighPlusOneIndices[1]);

			int this_process;
			MPI_Comm_rank(MPI_COMM_WORLD, &this_process);
			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			double gpu_time=MPI_Wtime();

			distributedGPUVec<double> cudaFlattenedArrayBlock;
			distributedGPUVec<double> projectorKetTimesVectorD;
			vectorTools::createDealiiVector(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
					blockSize,
					cudaFlattenedArrayBlock);

      if (isPsp)
        vectorTools::createDealiiVector(operatorMatrix.getProjectorKetTimesVectorSingle().get_partitioner(),
            blockSize,
            projectorKetTimesVectorD);

			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime() - gpu_time;

			if (this_process==0 && dftParameters::verbosity>=2)
				std::cout<<"Time for creating cuda parallel vectors for force computation: "<<gpu_time<<std::endl;

			gpu_time = MPI_Wtime();

			thrust::device_vector<double> eigenValuesD(blockSize,0.0);
			thrust::device_vector<double> partialOccupanciesD(blockSize,0.0);
			thrust::device_vector<double> elocWfcEshelbyTensorQuadValuesD(numCells*numQuads*6,0.0);
				
			thrust::device_vector<double> onesVecD(blockSize,1.0);

			const unsigned int cellsBlockSize=std::min((unsigned int)10,numCells);

			thrust::device_vector<double> psiQuadsFlatD(cellsBlockSize*numQuads*blockSize,0.0);
			thrust::device_vector<double> gradPsiQuadsXFlatD(cellsBlockSize*numQuads*blockSize,0.0);
			thrust::device_vector<double> gradPsiQuadsYFlatD(cellsBlockSize*numQuads*blockSize,0.0);
			thrust::device_vector<double> gradPsiQuadsZFlatD(cellsBlockSize*numQuads*blockSize,0.0);
			thrust::device_vector<double> gradPsiQuadsNLPFlatD(numCells*numQuadsNLP*3*blockSize,0.0);

			thrust::device_vector<double> eshelbyTensorContributionsD(cellsBlockSize*numQuads*blockSize*6,0.0);

		  const unsigned int innerBlockSizeEnlp=std::min((unsigned int)10,totalNonTrivialPseudoWfcs);
			thrust::device_vector<double> nlpContractionContributionD(innerBlockSizeEnlp*numQuadsNLP*3*blockSize,0.0);
			thrust::device_vector<double> projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedD;
			thrust::device_vector<unsigned int> projecterKetTimesFlattenedVectorLocalIdsD;
			thrust::device_vector<unsigned int> nonTrivialIdToElemIdMapD;
			if (totalNonTrivialPseudoWfcs>0)
			{
				projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedD.resize(totalNonTrivialPseudoWfcs*numQuadsNLP*3,0.0);
				projecterKetTimesFlattenedVectorLocalIdsD.resize(totalNonTrivialPseudoWfcs,0.0);
				nonTrivialIdToElemIdMapD.resize(totalNonTrivialPseudoWfcs,0.0);

				cudaMemcpy(thrust::raw_pointer_cast(&nonTrivialIdToElemIdMapD[0]),
						nonTrivialIdToElemIdMapH,
						totalNonTrivialPseudoWfcs*sizeof(unsigned int),
						cudaMemcpyHostToDevice);


				cudaMemcpy(thrust::raw_pointer_cast(&projecterKetTimesFlattenedVectorLocalIdsD[0]),
						projecterKetTimesFlattenedVectorLocalIdsH,
						totalNonTrivialPseudoWfcs*sizeof(unsigned int),
						cudaMemcpyHostToDevice);
			}


			for(unsigned int ivec = 0; ivec < N; ivec+=blockSize)
			{
				if((ivec+blockSize)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
						(ivec+blockSize)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
				{
					std::vector<double> blockedEigenValues(blockSize,0.0);
					std::vector<double> blockedPartialOccupancies(blockSize,0.0);
					for (unsigned int iWave=0; iWave<blockSize;++iWave)
					{
						blockedEigenValues[iWave]=eigenValuesH[ivec+iWave];
						blockedPartialOccupancies[iWave]
							=dftUtils::getPartialOccupancy(blockedEigenValues[iWave],
									fermiEnergy,
									C_kb,
									dftParameters::TVal);

					}



					cudaMemcpy(thrust::raw_pointer_cast(&eigenValuesD[0]),
							&blockedEigenValues[0],
							blockSize*sizeof(double),
							cudaMemcpyHostToDevice);

					cudaMemcpy(thrust::raw_pointer_cast(&partialOccupanciesD[0]),
							&blockedPartialOccupancies[0],
							blockSize*sizeof(double),
							cudaMemcpyHostToDevice);

					//cudaDeviceSynchronize();
					//MPI_Barrier(MPI_COMM_WORLD);
					//double kernel_time = MPI_Wtime();

					gpuPortedForceKernelsAllD(operatorMatrix,
							cudaFlattenedArrayBlock,
							projectorKetTimesVectorD,
							X,
							eigenValuesD,
							partialOccupanciesD,
							onesVecD,
							nonTrivialIdToElemIdMapD,
							projecterKetTimesFlattenedVectorLocalIdsD,
							ivec,
							N,
							blockSize,
							numCells,
							numQuads,
							numQuadsNLP,
							numNodesPerElement,
							totalNonTrivialPseudoWfcs,
							psiQuadsFlatD,
							gradPsiQuadsXFlatD,
							gradPsiQuadsYFlatD,
							gradPsiQuadsZFlatD,
							gradPsiQuadsNLPFlatD,
							eshelbyTensorContributionsD,
							elocWfcEshelbyTensorQuadValuesD,
							nlpContractionContributionD,
							projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedD,
							cellsBlockSize,
							innerBlockSizeEnlp,
							isPsp);

					//cudaDeviceSynchronize();
					//MPI_Barrier(MPI_COMM_WORLD);
					//kernel_time = MPI_Wtime() - kernel_time;

					//if (this_process==0 && dftParameters::verbosity>=5)
					//   std::cout<<"Time for force kernels all insided block loop: "<<kernel_time<<std::endl;
				}//band parallelization
			}//ivec loop

			cudaMemcpy(eshelbyTensorQuadValuesH,
					thrust::raw_pointer_cast(&elocWfcEshelbyTensorQuadValuesD[0]),
					numCells*numQuads*6*sizeof(double),
					cudaMemcpyDeviceToHost);  


			if (totalNonTrivialPseudoWfcs>0)
      {
				cudaMemcpy(projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedH,
						thrust::raw_pointer_cast(&projectorKetTimesPsiTimesVTimesPartOccContractionGradPsiQuadsFlattenedD[0]),
						totalNonTrivialPseudoWfcs*numQuadsNLP*3*sizeof(double),
						cudaMemcpyDeviceToHost);
      }
			cudaDeviceSynchronize();
			MPI_Barrier(MPI_COMM_WORLD);
			gpu_time = MPI_Wtime() - gpu_time;

			if (this_process==0 && dftParameters::verbosity>=1)
				std::cout<<"Time taken for all gpu kernels force computation: "<<gpu_time<<std::endl;
		}

	}//forceCUDA namespace
}//dftfe namespace
