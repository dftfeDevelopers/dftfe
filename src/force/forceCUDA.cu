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
                                                        const unsigned int startingCellId,
                                                        const unsigned int numQuads,
					                const double * psiQuadValues,
					                const double * gradPsiQuadValuesX,
					                const double * gradPsiQuadValuesY,
					                const double * gradPsiQuadValuesZ,
					                const double * eigenValues,
					                const double * partialOccupancies,
                                                        double *eshelbyTensor00,
                                                        double *eshelbyTensor10,
                                                        double *eshelbyTensor11,
                                                        double *eshelbyTensor20,
                                                        double *eshelbyTensor21,
                                                        double *eshelbyTensor22)
          {

		  const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
		  const unsigned int numberEntries = numContiguousBlocks*contiguousBlockSize;

		  for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
		   {
		      const unsigned int blockIndex = index/contiguousBlockSize;
		      const unsigned int intraBlockIndex=index-blockIndex*contiguousBlockSize;
                      const unsigned int cellIndex=blockIndex/numQuads;
                      const unsigned int quadId=blockIndex-cellIndex*numQuads;
                      const unsigned int tempIndex=(startingCellId+cellIndex)*numQuads*contiguousBlockSize+quadId*contiguousBlockSize+intraBlockIndex;
                      const double psi=psiQuadValues[tempIndex];
                      const double gradPsiX=gradPsiQuadValuesX[tempIndex];
                      const double gradPsiY=gradPsiQuadValuesY[tempIndex];
                      const double gradPsiZ=gradPsiQuadValuesZ[tempIndex];
                      const double eigenValue=eigenValues[intraBlockIndex];
                      const double partOcc=partialOccupancies[intraBlockIndex];

                      const double identityFactor=partOcc*(gradPsiX*gradPsiX+gradPsiY*gradPsiY+gradPsiZ*gradPsiZ)-2.0*partOcc*eigenValue*psi*psi;
		      eshelbyTensor00[index]=-2.0*partOcc*gradPsiX*gradPsiX+identityFactor;
                      eshelbyTensor10[index]=-2.0*partOcc*gradPsiY*gradPsiX;
                      eshelbyTensor11[index]=-2.0*partOcc*gradPsiY*gradPsiY+identityFactor;
                      eshelbyTensor20[index]=-2.0*partOcc*gradPsiZ*gradPsiX;
                      eshelbyTensor21[index]=-2.0*partOcc*gradPsiZ*gradPsiY;
                      eshelbyTensor22[index]=-2.0*partOcc*gradPsiZ*gradPsiZ+identityFactor;
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


          __global__
          void computeRhoGradRhoFromInterpolatedValues(const unsigned int numberEntries,
			    double *rhoCellsWfcContributions,
                            double *gradRhoCellsWfcContributionsX,
                            double *gradRhoCellsWfcContributionsY,
                            double *gradRhoCellsWfcContributionsZ,
                            const bool isEvaluateGradRho)
          {

		  const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;

		  for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
		   {
                      const double psi=rhoCellsWfcContributions[index];
                      rhoCellsWfcContributions[index]=psi*psi;

                      if (isEvaluateGradRho)
                      {
                        const double gradPsiX=gradRhoCellsWfcContributionsX[index];
                        gradRhoCellsWfcContributionsX[index]=2.0*psi*gradPsiX;
                         
                        const double gradPsiY=gradRhoCellsWfcContributionsY[index];
                        gradRhoCellsWfcContributionsY[index]=2.0*psi*gradPsiY;

                        const double gradPsiZ=gradRhoCellsWfcContributionsZ[index];
                        gradRhoCellsWfcContributionsZ[index]=2.0*psi*gradPsiZ;
                         
                      }
		   }

          }


      }

      void computeELocWfcEshelbyTensorNonPeriodicH(operatorDFTCUDAClass & operatorMatrix,
                                                 const double * psiQuadValuesH,
                                                 const double * gradPsiQuadValuesXH,
                                                 const double * gradPsiQuadValuesYH,
                                                 const double * gradPsiQuadValuesZH,
                                                 const double * eigenValuesH,
                                                 const double * partialOccupanciesH,
                                                 const unsigned int numCells,
                                                 const unsigned int numQuads,
                                                 const unsigned int numPsi,
                                                 double * eshelbyTensorQuadValuesH00,
                                                 double * eshelbyTensorQuadValuesH10,
                                                 double * eshelbyTensorQuadValuesH11,
                                                 double * eshelbyTensorQuadValuesH20,
                                                 double * eshelbyTensorQuadValuesH21,
                                                 double * eshelbyTensorQuadValuesH22)
      { 
           cudaDeviceSynchronize(); 
           thrust::device_vector<double> psiQuadValuesD(numCells*numQuads*numPsi,0.0);
           thrust::device_vector<double> gradPsiQuadValuesXD(numCells*numQuads*numPsi,0.0);
           thrust::device_vector<double> gradPsiQuadValuesYD(numCells*numQuads*numPsi,0.0);
           thrust::device_vector<double> gradPsiQuadValuesZD(numCells*numQuads*numPsi,0.0);
           thrust::device_vector<double> eigenValuesD(numPsi,0.0);
           thrust::device_vector<double> partialOccupanciesD(numPsi,0.0);
           thrust::device_vector<double> eshelbyTensorQuadValuesD00(numCells*numQuads,0.0);
           thrust::device_vector<double> eshelbyTensorQuadValuesD10(numCells*numQuads,0.0);
           thrust::device_vector<double> eshelbyTensorQuadValuesD11(numCells*numQuads,0.0);
           thrust::device_vector<double> eshelbyTensorQuadValuesD20(numCells*numQuads,0.0);
           thrust::device_vector<double> eshelbyTensorQuadValuesD21(numCells*numQuads,0.0);
           thrust::device_vector<double> eshelbyTensorQuadValuesD22(numCells*numQuads,0.0);


           cudaMemcpy(thrust::raw_pointer_cast(&psiQuadValuesD[0]),
		      psiQuadValuesH,
		      numCells*numQuads*numPsi*sizeof(double),
		      cudaMemcpyHostToDevice);

           cudaMemcpy(thrust::raw_pointer_cast(&gradPsiQuadValuesXD[0]),
		      gradPsiQuadValuesXH,
		      numCells*numQuads*numPsi*sizeof(double),
		      cudaMemcpyHostToDevice);

           cudaMemcpy(thrust::raw_pointer_cast(&gradPsiQuadValuesYD[0]),
		      gradPsiQuadValuesYH,
		      numCells*numQuads*numPsi*sizeof(double),
		      cudaMemcpyHostToDevice);

           cudaMemcpy(thrust::raw_pointer_cast(&gradPsiQuadValuesZD[0]),
		      gradPsiQuadValuesZH,
		      numCells*numQuads*numPsi*sizeof(double),
		      cudaMemcpyHostToDevice);

           cudaMemcpy(thrust::raw_pointer_cast(&eigenValuesD[0]),
		      eigenValuesH,
		      numPsi*sizeof(double),
		      cudaMemcpyHostToDevice);

           cudaMemcpy(thrust::raw_pointer_cast(&partialOccupanciesD[0]),
		      partialOccupanciesH,
		      numPsi*sizeof(double),
		      cudaMemcpyHostToDevice);

           cudaMemcpy(thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD00[0]),
		      eshelbyTensorQuadValuesH00,
		      numCells*numQuads*sizeof(double),
		      cudaMemcpyHostToDevice); 

           cudaMemcpy(thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD10[0]),
		      eshelbyTensorQuadValuesH10,
		      numCells*numQuads*sizeof(double),
		      cudaMemcpyHostToDevice); 

           cudaMemcpy(thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD11[0]),
		      eshelbyTensorQuadValuesH11,
		      numCells*numQuads*sizeof(double),
		      cudaMemcpyHostToDevice); 

           cudaMemcpy(thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD20[0]),
		      eshelbyTensorQuadValuesH20,
		      numCells*numQuads*sizeof(double),
		      cudaMemcpyHostToDevice); 

           cudaMemcpy(thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD21[0]),
		      eshelbyTensorQuadValuesH21,
		      numCells*numQuads*sizeof(double),
		      cudaMemcpyHostToDevice); 

           cudaMemcpy(thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD22[0]),
		      eshelbyTensorQuadValuesH22,
		      numCells*numQuads*sizeof(double),
		      cudaMemcpyHostToDevice); 
           
           computeELocWfcEshelbyTensorNonPeriodicD(operatorMatrix,
                                                  psiQuadValuesD,
                                                  gradPsiQuadValuesXD,
                                                  gradPsiQuadValuesYD,
                                                  gradPsiQuadValuesZD,
                                                  eigenValuesD,
                                                  partialOccupanciesD,
                                                  numCells,
                                                  numQuads,
                                                  numPsi,
                                                  eshelbyTensorQuadValuesD00,
                                                  eshelbyTensorQuadValuesD10,
                                                  eshelbyTensorQuadValuesD11,
                                                  eshelbyTensorQuadValuesD20,
                                                  eshelbyTensorQuadValuesD21,
                                                  eshelbyTensorQuadValuesD22); 
           
           cudaMemcpy(eshelbyTensorQuadValuesH00,
		      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD00[0]),
		      numCells*numQuads*sizeof(double),
		      cudaMemcpyDeviceToHost);  

           cudaMemcpy(eshelbyTensorQuadValuesH10,
		      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD10[0]),
		      numCells*numQuads*sizeof(double),
		      cudaMemcpyDeviceToHost);   
           cudaMemcpy(eshelbyTensorQuadValuesH11,
		      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD11[0]),
		      numCells*numQuads*sizeof(double),
		      cudaMemcpyDeviceToHost);   
           cudaMemcpy(eshelbyTensorQuadValuesH20,
		      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD20[0]),
		      numCells*numQuads*sizeof(double),
		      cudaMemcpyDeviceToHost);   
           cudaMemcpy(eshelbyTensorQuadValuesH21,
		      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD21[0]),
		      numCells*numQuads*sizeof(double),
		      cudaMemcpyDeviceToHost);   
           cudaMemcpy(eshelbyTensorQuadValuesH22,
		      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD22[0]),
		      numCells*numQuads*sizeof(double),
		      cudaMemcpyDeviceToHost);    
           cudaDeviceSynchronize();
      }

      void computeELocWfcEshelbyTensorNonPeriodicD(operatorDFTCUDAClass & operatorMatrix,
                                                 const thrust::device_vector<double> & psiQuadValuesD,
                                                 const thrust::device_vector<double> & gradPsiQuadValuesXD,
                                                 const thrust::device_vector<double> & gradPsiQuadValuesYD,
                                                 const thrust::device_vector<double> & gradPsiQuadValuesZD,
                                                 const thrust::device_vector<double> & eigenValuesD,
                                                 const thrust::device_vector<double> & partialOccupanciesD,
                                                 const unsigned int numCells, 
                                                 const unsigned int numQuads,
                                                 const unsigned int numPsi,
                                                 thrust::device_vector<double> & eshelbyTensorQuadValuesD00,
                                                 thrust::device_vector<double> & eshelbyTensorQuadValuesD10,
                                                 thrust::device_vector<double> & eshelbyTensorQuadValuesD11,
                                                 thrust::device_vector<double> & eshelbyTensorQuadValuesD20,
                                                 thrust::device_vector<double> & eshelbyTensorQuadValuesD21,
                                                 thrust::device_vector<double> & eshelbyTensorQuadValuesD22)
      { 
   
           thrust::device_vector<double> eshelbyTensorContributionsD00(numCells*numQuads*numPsi,0.0);
           thrust::device_vector<double> eshelbyTensorContributionsD10(numCells*numQuads*numPsi,0.0); 
           thrust::device_vector<double> eshelbyTensorContributionsD11(numCells*numQuads*numPsi,0.0); 
           thrust::device_vector<double> eshelbyTensorContributionsD20(numCells*numQuads*numPsi,0.0); 
           thrust::device_vector<double> eshelbyTensorContributionsD21(numCells*numQuads*numPsi,0.0); 
           thrust::device_vector<double> eshelbyTensorContributionsD22(numCells*numQuads*numPsi,0.0); 
           

           thrust::device_vector<double> onesVectorD(numPsi,1.0);
           
	   computeELocWfcEshelbyTensorContributions<<<(numPsi+255)/256*numCells*numQuads,256>>>
							  (numPsi,
							   numCells*numQuads,
                                                           0,
                                                           numQuads,
                                                           thrust::raw_pointer_cast(&psiQuadValuesD[0]),
                                                           thrust::raw_pointer_cast(&gradPsiQuadValuesXD[0]),
                                                           thrust::raw_pointer_cast(&gradPsiQuadValuesYD[0]),
                                                           thrust::raw_pointer_cast(&gradPsiQuadValuesZD[0]),
							   thrust::raw_pointer_cast(&eigenValuesD[0]),
                                                           thrust::raw_pointer_cast(&partialOccupanciesD[0]),
							   thrust::raw_pointer_cast(&eshelbyTensorContributionsD00[0]),
                                                           thrust::raw_pointer_cast(&eshelbyTensorContributionsD10[0]),
                                                           thrust::raw_pointer_cast(&eshelbyTensorContributionsD11[0]),
                                                           thrust::raw_pointer_cast(&eshelbyTensorContributionsD20[0]),
                                                           thrust::raw_pointer_cast(&eshelbyTensorContributionsD21[0]),
                                                           thrust::raw_pointer_cast(&eshelbyTensorContributionsD22[0]));

          


	   double scalarCoeffAlpha = 1.0,scalarCoeffBeta = 1.0;


	  
	   cublasDgemm(operatorMatrix.getCublasHandle(),
		      CUBLAS_OP_N,
		      CUBLAS_OP_N,
		      1,
		      numCells*numQuads,
		      numPsi,
		      &scalarCoeffAlpha,
		      thrust::raw_pointer_cast(&onesVectorD[0]),
		      1,
		      thrust::raw_pointer_cast(&eshelbyTensorContributionsD00[0]),
		      numPsi,
		      &scalarCoeffBeta,
		      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD00[0]),
		      1);

	   cublasDgemm(operatorMatrix.getCublasHandle(),
		      CUBLAS_OP_N,
		      CUBLAS_OP_N,
		      1,
		      numCells*numQuads,
		      numPsi,
		      &scalarCoeffAlpha,
		      thrust::raw_pointer_cast(&onesVectorD[0]),
		      1,
		      thrust::raw_pointer_cast(&eshelbyTensorContributionsD10[0]),
		      numPsi,
		      &scalarCoeffBeta,
		      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD10[0]),
		      1); 

	   cublasDgemm(operatorMatrix.getCublasHandle(),
		      CUBLAS_OP_N,
		      CUBLAS_OP_N,
		      1,
		      numCells*numQuads,
		      numPsi,
		      &scalarCoeffAlpha,
		      thrust::raw_pointer_cast(&onesVectorD[0]),
		      1,
		      thrust::raw_pointer_cast(&eshelbyTensorContributionsD11[0]),
		      numPsi,
		      &scalarCoeffBeta,
		      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD11[0]),
		      1); 

	   cublasDgemm(operatorMatrix.getCublasHandle(),
		      CUBLAS_OP_N,
		      CUBLAS_OP_N,
		      1,
		      numCells*numQuads,
		      numPsi,
		      &scalarCoeffAlpha,
		      thrust::raw_pointer_cast(&onesVectorD[0]),
		      1,
		      thrust::raw_pointer_cast(&eshelbyTensorContributionsD20[0]),
		      numPsi,
		      &scalarCoeffBeta,
		      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD20[0]),
		      1); 

	   cublasDgemm(operatorMatrix.getCublasHandle(),
		      CUBLAS_OP_N,
		      CUBLAS_OP_N,
		      1,
		      numCells*numQuads,
		      numPsi,
		      &scalarCoeffAlpha,
		      thrust::raw_pointer_cast(&onesVectorD[0]),
		      1,
		      thrust::raw_pointer_cast(&eshelbyTensorContributionsD21[0]),
		      numPsi,
		      &scalarCoeffBeta,
		      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD21[0]),
		      1); 

	   cublasDgemm(operatorMatrix.getCublasHandle(),
		      CUBLAS_OP_N,
		      CUBLAS_OP_N,
		      1,
		      numCells*numQuads,
		      numPsi,
		      &scalarCoeffAlpha,
		      thrust::raw_pointer_cast(&onesVectorD[0]),
		      1,
		      thrust::raw_pointer_cast(&eshelbyTensorContributionsD22[0]),
		      numPsi,
		      &scalarCoeffBeta,
		      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD22[0]),
		      1);       
      }

      void computeNonLocalProjectorKetTimesPsiTimesVH(operatorDFTCUDAClass & operatorMatrix,
                                                      const double * X,
                                                      const unsigned int startingVecId,
                                                      const unsigned int BVec,
                                                      const unsigned int N,
                                                      double * projectorKetTimesPsiTimesVH)
      {

	    cudaVectorType cudaFlattenedArrayBlock;
	    vectorTools::createDealiiVector(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
					    BVec,
					    cudaFlattenedArrayBlock);


	    cudaVectorType projectorKetTimesVector;
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


     void interpolatePsiH(operatorDFTCUDAClass & operatorMatrix,
                          const double * X,
                          const unsigned int startingVecId,
                          const unsigned int BVec,
                          const unsigned int N,
                          const unsigned int numCells,
                          const unsigned int numQuads,
                          const unsigned int numQuadsNLP,
                          const unsigned int numNodesPerElement,
                          double * psiQuadsFlatH,
                          double * psiQuadsNLPFlatH,
                          double * gradPsiQuadsXFlatH,
                          double * gradPsiQuadsYFlatH,
                          double * gradPsiQuadsZFlatH,
                          const bool interpolateForNLPQuad)
     {
            thrust::device_vector<double> psiQuadsFlatD(numCells*numQuads*BVec,0.0);
            thrust::device_vector<double> psiQuadsNLPFlatD(numCells*numQuadsNLP*BVec,0.0);
            thrust::device_vector<double> gradPsiQuadsXFlatD(numCells*numQuads*BVec,0.0);
            thrust::device_vector<double> gradPsiQuadsYFlatD(numCells*numQuads*BVec,0.0);
            thrust::device_vector<double> gradPsiQuadsZFlatD(numCells*numQuads*BVec,0.0);


	    cudaVectorType cudaFlattenedArrayBlock;
	    vectorTools::createDealiiVector(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
					    BVec,
					    cudaFlattenedArrayBlock);


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

	    interpolatePsiD(operatorMatrix,
			    cudaFlattenedArrayBlock,
			    BVec,
			    N,
			    numCells,
			    numQuads,
			    numQuadsNLP,
			    numNodesPerElement,
			    psiQuadsFlatD,
			    psiQuadsNLPFlatD,
			    gradPsiQuadsXFlatD,
			    gradPsiQuadsYFlatD,
			    gradPsiQuadsZFlatD,
			    interpolateForNLPQuad);

            cudaMemcpy(psiQuadsFlatH,
		      thrust::raw_pointer_cast(&psiQuadsFlatD[0]),
		      numCells*numQuads*BVec*sizeof(double),
		      cudaMemcpyDeviceToHost);

            if (interpolateForNLPQuad)
		    cudaMemcpy(psiQuadsNLPFlatH,
			      thrust::raw_pointer_cast(&psiQuadsNLPFlatD[0]),
			      numCells*numQuadsNLP*BVec*sizeof(double),
			      cudaMemcpyDeviceToHost); 

            cudaMemcpy(gradPsiQuadsXFlatH,
		      thrust::raw_pointer_cast(&gradPsiQuadsXFlatD[0]),
		      numCells*numQuads*BVec*sizeof(double),
		      cudaMemcpyDeviceToHost);

            cudaMemcpy(gradPsiQuadsYFlatH,
		      thrust::raw_pointer_cast(&gradPsiQuadsYFlatD[0]),
		      numCells*numQuads*BVec*sizeof(double),
		      cudaMemcpyDeviceToHost);

            cudaMemcpy(gradPsiQuadsZFlatH,
		      thrust::raw_pointer_cast(&gradPsiQuadsZFlatD[0]),
		      numCells*numQuads*BVec*sizeof(double),
		      cudaMemcpyDeviceToHost); 

     }

     void interpolatePsiD(operatorDFTCUDAClass & operatorMatrix,
                          cudaVectorType & Xb,
                          const unsigned int BVec,
                          const unsigned int N,
                          const unsigned int numCells,
                          const unsigned int numQuads,
                          const unsigned int numQuadsNLP,
                          const unsigned int numNodesPerElement,
                          thrust::device_vector<double> & psiQuadsFlatD,
                          thrust::device_vector<double> & psiQuadsNLPFlatD,
                          thrust::device_vector<double> & gradPsiQuadsXFlatD,
                          thrust::device_vector<double> & gradPsiQuadsYFlatD,
                          thrust::device_vector<double> & gradPsiQuadsZFlatD,
                          const bool interpolateForNLPQuad)
     {
            thrust::device_vector<double> & cellWaveFunctionMatrix = operatorMatrix.getCellWaveFunctionMatrix();

	    copyCUDAKernel<<<(BVec+255)/256*numCells*numNodesPerElement,256>>>
							  (BVec,
							   numCells*numNodesPerElement,
							   Xb.begin(),
							   thrust::raw_pointer_cast(&cellWaveFunctionMatrix[0]),
							   thrust::raw_pointer_cast(&(operatorMatrix.getFlattenedArrayCellLocalProcIndexIdMap())[0]));

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
				    thrust::raw_pointer_cast(&cellWaveFunctionMatrix[0]),
				    BVec,
				    strideA,
				    thrust::raw_pointer_cast(&(operatorMatrix.getShapeFunctionValuesInverted())[0]),
				    numNodesPerElement,
				    strideB,
				    &scalarCoeffBeta,
				    thrust::raw_pointer_cast(&psiQuadsFlatD[0]),
				    BVec,
				    strideC,
				    numCells);

            if (interpolateForNLPQuad)
            {
		    int strideCNLP = BVec*numQuadsNLP;
		    cublasDgemmStridedBatched(operatorMatrix.getCublasHandle(),
					    CUBLAS_OP_N,
					    CUBLAS_OP_N,
					    BVec,
					    numQuadsNLP,
					    numNodesPerElement,
					    &scalarCoeffAlpha,
					    thrust::raw_pointer_cast(&cellWaveFunctionMatrix[0]),
					    BVec,
					    strideA,
					    thrust::raw_pointer_cast(&(operatorMatrix.getShapeFunctionValuesNLPInverted())[0]),
					    numNodesPerElement,
					    strideB,
					    &scalarCoeffBeta,
					    thrust::raw_pointer_cast(&psiQuadsNLPFlatD[0]),
					    BVec,
					    strideCNLP,
					    numCells);
            }

	    strideB=numNodesPerElement*numQuads;

	    cublasDgemmStridedBatched(operatorMatrix.getCublasHandle(),
				    CUBLAS_OP_N,
				    CUBLAS_OP_N,
				    BVec,
				    numQuads,
				    numNodesPerElement,
				    &scalarCoeffAlpha,
				    thrust::raw_pointer_cast(&cellWaveFunctionMatrix[0]),
				    BVec,
				    strideA,
				    thrust::raw_pointer_cast(&(operatorMatrix.getShapeFunctionGradientValuesXInverted())[0]),
				    numNodesPerElement,
				    strideB,
				    &scalarCoeffBeta,
				    thrust::raw_pointer_cast(&gradPsiQuadsXFlatD[0]),
				    BVec,
				    strideC,
				    numCells);


	    cublasDgemmStridedBatched(operatorMatrix.getCublasHandle(),
				    CUBLAS_OP_N,
				    CUBLAS_OP_N,
				    BVec,
				    numQuads,
				    numNodesPerElement,
				    &scalarCoeffAlpha,
				    thrust::raw_pointer_cast(&cellWaveFunctionMatrix[0]),
				    BVec,
				    strideA,
				    thrust::raw_pointer_cast(&(operatorMatrix.getShapeFunctionGradientValuesYInverted())[0]),
				    numNodesPerElement,
				    strideB,
				    &scalarCoeffBeta,
				    thrust::raw_pointer_cast(&gradPsiQuadsYFlatD[0]),
				    BVec,
				    strideC,
				    numCells);

	    cublasDgemmStridedBatched(operatorMatrix.getCublasHandle(),
				    CUBLAS_OP_N,
				    CUBLAS_OP_N,
				    BVec,
				    numQuads,
				    numNodesPerElement,
				    &scalarCoeffAlpha,
				    thrust::raw_pointer_cast(&cellWaveFunctionMatrix[0]),
				    BVec,
				    strideA,
				    thrust::raw_pointer_cast(&(operatorMatrix.getShapeFunctionGradientValuesZInverted())[0]),
				    numNodesPerElement,
				    strideB,
				    &scalarCoeffBeta,
				    thrust::raw_pointer_cast(&gradPsiQuadsZFlatD[0]),
				    BVec,
				    strideC,
				    numCells);
     }

     void interpolatePsiComputeELocWfcEshelbyTensorNonPeriodicD(operatorDFTCUDAClass & operatorMatrix,
						  cudaVectorType & Xb,
						  const unsigned int BVec,
						  const unsigned int numCells,
						  const unsigned int numQuads,
						  const unsigned int numNodesPerElement,
                                                  const thrust::device_vector<double> & eigenValuesD,
                                                  const thrust::device_vector<double> & partialOccupanciesD,
                                                  const unsigned int innerBlockSizeEloc,
                                                  thrust::device_vector<double> & psiQuadsFlatD,
                                                  thrust::device_vector<double> & gradPsiQuadsXFlatD,
                                                  thrust::device_vector<double> & gradPsiQuadsYFlatD,
                                                  thrust::device_vector<double> & gradPsiQuadsZFlatD,
				                  thrust::device_vector<double> & eshelbyTensorContributionsD00,
			                     	  thrust::device_vector<double> & eshelbyTensorContributionsD10,
					          thrust::device_vector<double> & eshelbyTensorContributionsD11,
					          thrust::device_vector<double> & eshelbyTensorContributionsD20,
					          thrust::device_vector<double> & eshelbyTensorContributionsD21,
					          thrust::device_vector<double> & eshelbyTensorContributionsD22,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD00,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD10,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD11,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD20,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD21,
                                                  thrust::device_vector<double> & eshelbyTensorQuadValuesD22)
     {
            //thrust::device_vector<double> gradPsiQuadsXFlatD(numCells*numQuads*BVec,0.0);
            //thrust::device_vector<double> gradPsiQuadsYFlatD(numCells*numQuads*BVec,0.0);
            //thrust::device_vector<double> gradPsiQuadsZFlatD(numCells*numQuads*BVec,0.0);

            thrust::device_vector<double> & cellWaveFunctionMatrix = operatorMatrix.getCellWaveFunctionMatrix();

	    copyCUDAKernel<<<(BVec+255)/256*numCells*numNodesPerElement,256>>>
							  (BVec,
							   numCells*numNodesPerElement,
							   Xb.begin(),
							   thrust::raw_pointer_cast(&cellWaveFunctionMatrix[0]),
							   thrust::raw_pointer_cast(&(operatorMatrix.getFlattenedArrayCellLocalProcIndexIdMap())[0]));
            
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
				    thrust::raw_pointer_cast(&cellWaveFunctionMatrix[0]),
				    BVec,
				    strideA,
				    thrust::raw_pointer_cast(&(operatorMatrix.getShapeFunctionValuesInverted())[0]),
				    numNodesPerElement,
				    strideB,
				    &scalarCoeffBeta,
				    thrust::raw_pointer_cast(&psiQuadsFlatD[0]),
				    BVec,
				    strideC,
				    numCells);

	    strideB=numNodesPerElement*numQuads;

	    cublasDgemmStridedBatched(operatorMatrix.getCublasHandle(),
				    CUBLAS_OP_N,
				    CUBLAS_OP_N,
				    BVec,
				    numQuads,
				    numNodesPerElement,
				    &scalarCoeffAlpha,
				    thrust::raw_pointer_cast(&cellWaveFunctionMatrix[0]),
				    BVec,
				    strideA,
				    thrust::raw_pointer_cast(&(operatorMatrix.getShapeFunctionGradientValuesXInverted())[0]),
				    numNodesPerElement,
				    strideB,
				    &scalarCoeffBeta,
				    thrust::raw_pointer_cast(&gradPsiQuadsXFlatD[0]),
				    BVec,
				    strideC,
				    numCells);


	    cublasDgemmStridedBatched(operatorMatrix.getCublasHandle(),
				    CUBLAS_OP_N,
				    CUBLAS_OP_N,
				    BVec,
				    numQuads,
				    numNodesPerElement,
				    &scalarCoeffAlpha,
				    thrust::raw_pointer_cast(&cellWaveFunctionMatrix[0]),
				    BVec,
				    strideA,
				    thrust::raw_pointer_cast(&(operatorMatrix.getShapeFunctionGradientValuesYInverted())[0]),
				    numNodesPerElement,
				    strideB,
				    &scalarCoeffBeta,
				    thrust::raw_pointer_cast(&gradPsiQuadsYFlatD[0]),
				    BVec,
				    strideC,
				    numCells);

	    cublasDgemmStridedBatched(operatorMatrix.getCublasHandle(),
				    CUBLAS_OP_N,
				    CUBLAS_OP_N,
				    BVec,
				    numQuads,
				    numNodesPerElement,
				    &scalarCoeffAlpha,
				    thrust::raw_pointer_cast(&cellWaveFunctionMatrix[0]),
				    BVec,
				    strideA,
				    thrust::raw_pointer_cast(&(operatorMatrix.getShapeFunctionGradientValuesZInverted())[0]),
				    numNodesPerElement,
				    strideB,
				    &scalarCoeffBeta,
				    thrust::raw_pointer_cast(&gradPsiQuadsZFlatD[0]),
				    BVec,
				    strideC,
				    numCells);
           
           const int blockSize=innerBlockSizeEloc;
           const int numberBlocks=numCells/blockSize;
           const int remBlockSize=numCells-numberBlocks*blockSize;
           //thrust::device_vector<double> eshelbyTensorContributionsD00(blockSize*numQuads*BVec,0.0);
           //thrust::device_vector<double> eshelbyTensorContributionsD10(blockSize*numQuads*BVec,0.0); 
           //thrust::device_vector<double> eshelbyTensorContributionsD11(blockSize*numQuads*BVec,0.0); 
           //thrust::device_vector<double> eshelbyTensorContributionsD20(blockSize*numQuads*BVec,0.0); 
           //thrust::device_vector<double> eshelbyTensorContributionsD21(blockSize*numQuads*BVec,0.0); 
           //thrust::device_vector<double> eshelbyTensorContributionsD22(blockSize*numQuads*BVec,0.0); 
           thrust::device_vector<double> onesVectorD(BVec,1.0);
           
           for (int iblock=0; iblock<(numberBlocks+1); iblock++)
	   {
                   const int currentBlockSize= (iblock==numberBlocks)?remBlockSize:blockSize;
                   const int startingId=iblock*blockSize;
                  
                   if (currentBlockSize>0)
	           {
                           
			   computeELocWfcEshelbyTensorContributions<<<(BVec+255)/256*currentBlockSize*numQuads,256>>>
									  (BVec,
									   currentBlockSize*numQuads,
									   startingId,
									   numQuads,
									   thrust::raw_pointer_cast(&psiQuadsFlatD[0]),
									   thrust::raw_pointer_cast(&gradPsiQuadsXFlatD[0]),
									   thrust::raw_pointer_cast(&gradPsiQuadsYFlatD[0]),
									   thrust::raw_pointer_cast(&gradPsiQuadsZFlatD[0]),
									   thrust::raw_pointer_cast(&eigenValuesD[0]),
									   thrust::raw_pointer_cast(&partialOccupanciesD[0]),
									   thrust::raw_pointer_cast(&eshelbyTensorContributionsD00[0]),
									   thrust::raw_pointer_cast(&eshelbyTensorContributionsD10[0]),
									   thrust::raw_pointer_cast(&eshelbyTensorContributionsD11[0]),
									   thrust::raw_pointer_cast(&eshelbyTensorContributionsD20[0]),
									   thrust::raw_pointer_cast(&eshelbyTensorContributionsD21[0]),
									   thrust::raw_pointer_cast(&eshelbyTensorContributionsD22[0]));
			  
			   scalarCoeffAlpha = 1.0;
			   scalarCoeffBeta = 1.0;

			  
			   cublasDgemm(operatorMatrix.getCublasHandle(),
				      CUBLAS_OP_N,
				      CUBLAS_OP_N,
				      1,
				      currentBlockSize*numQuads,
				      BVec,
				      &scalarCoeffAlpha,
				      thrust::raw_pointer_cast(&onesVectorD[0]),
				      1,
				      thrust::raw_pointer_cast(&eshelbyTensorContributionsD00[0]),
				      BVec,
				      &scalarCoeffBeta,
				      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD00[startingId*numQuads]),
				      1);

			   cublasDgemm(operatorMatrix.getCublasHandle(),
				      CUBLAS_OP_N,
				      CUBLAS_OP_N,
				      1,
				      currentBlockSize*numQuads,
				      BVec,
				      &scalarCoeffAlpha,
				      thrust::raw_pointer_cast(&onesVectorD[0]),
				      1,
				      thrust::raw_pointer_cast(&eshelbyTensorContributionsD10[0]),
				      BVec,
				      &scalarCoeffBeta,
				      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD10[startingId*numQuads]),
				      1); 

			   cublasDgemm(operatorMatrix.getCublasHandle(),
				      CUBLAS_OP_N,
				      CUBLAS_OP_N,
				      1,
				      currentBlockSize*numQuads,
				      BVec,
				      &scalarCoeffAlpha,
				      thrust::raw_pointer_cast(&onesVectorD[0]),
				      1,
				      thrust::raw_pointer_cast(&eshelbyTensorContributionsD11[0]),
				      BVec,
				      &scalarCoeffBeta,
				      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD11[startingId*numQuads]),
				      1); 

			   cublasDgemm(operatorMatrix.getCublasHandle(),
				      CUBLAS_OP_N,
				      CUBLAS_OP_N,
				      1,
				      currentBlockSize*numQuads,
				      BVec,
				      &scalarCoeffAlpha,
				      thrust::raw_pointer_cast(&onesVectorD[0]),
				      1,
				      thrust::raw_pointer_cast(&eshelbyTensorContributionsD20[0]),
				      BVec,
				      &scalarCoeffBeta,
				      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD20[startingId*numQuads]),
				      1); 

			   cublasDgemm(operatorMatrix.getCublasHandle(),
				      CUBLAS_OP_N,
				      CUBLAS_OP_N,
				      1,
				      currentBlockSize*numQuads,
				      BVec,
				      &scalarCoeffAlpha,
				      thrust::raw_pointer_cast(&onesVectorD[0]),
				      1,
				      thrust::raw_pointer_cast(&eshelbyTensorContributionsD21[0]),
				      BVec,
				      &scalarCoeffBeta,
				      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD21[startingId*numQuads]),
				      1); 

			   cublasDgemm(operatorMatrix.getCublasHandle(),
				      CUBLAS_OP_N,
				      CUBLAS_OP_N,
				      1,
				      currentBlockSize*numQuads,
				      BVec,
				      &scalarCoeffAlpha,
				      thrust::raw_pointer_cast(&onesVectorD[0]),
				      1,
				      thrust::raw_pointer_cast(&eshelbyTensorContributionsD22[0]),
				      BVec,
				      &scalarCoeffBeta,
				      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD22[startingId*numQuads]),
				      1); 
		   }
	   }
     }

     void interpolatePsiNLPD(operatorDFTCUDAClass & operatorMatrix,
                          cudaVectorType & Xb,
                          const unsigned int BVec,
                          const unsigned int N,
                          const unsigned int numCells,
                          const unsigned int numQuadsNLP,
                          const unsigned int numNodesPerElement,
                          thrust::device_vector<double> & psiQuadsNLPFlatD)
     {
            thrust::device_vector<double> & cellWaveFunctionMatrix = operatorMatrix.getCellWaveFunctionMatrix();

	    copyCUDAKernel<<<(BVec+255)/256*numCells*numNodesPerElement,256>>>
							  (BVec,
							   numCells*numNodesPerElement,
							   Xb.begin(),
							   thrust::raw_pointer_cast(&cellWaveFunctionMatrix[0]),
							   thrust::raw_pointer_cast(&(operatorMatrix.getFlattenedArrayCellLocalProcIndexIdMap())[0]));

	    double scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;
	    int strideA = BVec*numNodesPerElement;
	    int strideB = 0;

	    int strideCNLP = BVec*numQuadsNLP;
	    cublasDgemmStridedBatched(operatorMatrix.getCublasHandle(),
				    CUBLAS_OP_N,
				    CUBLAS_OP_N,
				    BVec,
				    numQuadsNLP,
				    numNodesPerElement,
				    &scalarCoeffAlpha,
				    thrust::raw_pointer_cast(&cellWaveFunctionMatrix[0]),
				    BVec,
				    strideA,
				    thrust::raw_pointer_cast(&(operatorMatrix.getShapeFunctionValuesNLPInverted())[0]),
				    numNodesPerElement,
				    strideB,
				    &scalarCoeffBeta,
				    thrust::raw_pointer_cast(&psiQuadsNLPFlatD[0]),
				    BVec,
				    strideCNLP,
				    numCells);
     }


     void nlpPsiContractionD(operatorDFTCUDAClass & operatorMatrix,
			    const thrust::device_vector<double> & psiQuadValuesNLPD,
                            const thrust::device_vector<double> & partialOccupanciesD,
                            const double * projectorKetTimesVectorParFlattenedD,
                            const thrust::device_vector<unsigned int> & nonTrivialIdToElemIdMapD,
                            const thrust::device_vector<unsigned int> & projecterKetTimesFlattenedVectorLocalIdsD,
                            const unsigned int numCells, 
			    const unsigned int numQuadsNLP,
			    const unsigned int numPsi,
                            const unsigned int totalNonTrivialPseudoWfcs,
                            const unsigned int innerBlockSizeEnlp,
                            thrust::device_vector<double> & nlpContractionContributionD,
                            thrust::device_vector<double> & projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedD)
     {
            const int blockSize=innerBlockSizeEnlp;
            const int numberBlocks=totalNonTrivialPseudoWfcs/blockSize;
            const int remBlockSize=totalNonTrivialPseudoWfcs-numberBlocks*blockSize;
            //thrust::device_vector<double> nlpContractionContributionD(blockSize*numQuadsNLP*numPsi,0.0);
	    thrust::device_vector<double> onesMatD(numPsi,1.0);

            for (int iblock=0; iblock<(numberBlocks+1); iblock++)
            {
                    const int currentBlockSize= (iblock==numberBlocks)?remBlockSize:blockSize;
                    const int startingId=iblock*blockSize;
                    if (currentBlockSize>0)
                    {
			    nlpPsiContractionCUDAKernel<<<(numPsi+255)/256*numQuadsNLP*currentBlockSize,256>>>
									  (numPsi,
									   numQuadsNLP,
									   currentBlockSize,
                                                                           startingId,
									   projectorKetTimesVectorParFlattenedD,
									   thrust::raw_pointer_cast(&psiQuadValuesNLPD[0]),
									   thrust::raw_pointer_cast(&partialOccupanciesD[0]),
									   thrust::raw_pointer_cast(&nonTrivialIdToElemIdMapD[0]),
									   thrust::raw_pointer_cast(&projecterKetTimesFlattenedVectorLocalIdsD[0]),
									   thrust::raw_pointer_cast(&nlpContractionContributionD[0]));
			    double scalarCoeffAlpha = 1.0,scalarCoeffBeta = 1.0;

			  
			    cublasDgemm(operatorMatrix.getCublasHandle(),
				      CUBLAS_OP_N,
				      CUBLAS_OP_N,
				      1,
				      currentBlockSize*numQuadsNLP,
				      numPsi,
				      &scalarCoeffAlpha,
				      thrust::raw_pointer_cast(&onesMatD[0]),
				      1,
				      thrust::raw_pointer_cast(&nlpContractionContributionD[0]),
				      numPsi,
				      &scalarCoeffBeta,
				      thrust::raw_pointer_cast(&projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedD[startingId*numQuadsNLP]),
				      1);
                    }
            }
     }


     void gpuPortedForceKernelsAllD(operatorDFTCUDAClass & operatorMatrix,
                             cudaVectorType & cudaFlattenedArrayBlock,
                             cudaVectorType & projectorKetTimesVectorD,
                             const double * X,
		             const thrust::device_vector<double> & eigenValuesD,
			     const thrust::device_vector<double> & partialOccupanciesD,
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
	                     thrust::device_vector<double> & eshelbyTensorContributionsD00,
			     thrust::device_vector<double> & eshelbyTensorContributionsD10,
			     thrust::device_vector<double> & eshelbyTensorContributionsD11,
			     thrust::device_vector<double> & eshelbyTensorContributionsD20,
			     thrust::device_vector<double> & eshelbyTensorContributionsD21,
			     thrust::device_vector<double> & eshelbyTensorContributionsD22,
			     thrust::device_vector<double> & eshelbyTensorQuadValuesD00,
			     thrust::device_vector<double> & eshelbyTensorQuadValuesD10,
			     thrust::device_vector<double> & eshelbyTensorQuadValuesD11,
			     thrust::device_vector<double> & eshelbyTensorQuadValuesD20,
			     thrust::device_vector<double> & eshelbyTensorQuadValuesD21,
			     thrust::device_vector<double> & eshelbyTensorQuadValuesD22,
                             thrust::device_vector<double> & nlpContractionContributionD,
			     thrust::device_vector<double> & projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedD,
                             const unsigned int innerBlockSizeEloc,
                             const unsigned int innerBlockSizeEnlp,
                             const bool isPsp,
			     const bool interpolateForNLPQuad)
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


            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);
            double kernel1_time = MPI_Wtime();

           interpolatePsiComputeELocWfcEshelbyTensorNonPeriodicD(operatorMatrix,
						   cudaFlattenedArrayBlock,
						   numPsi,
						   numCells,
						   numQuads,
						   numNodesPerElement,
                                                   eigenValuesD,
                                                   partialOccupanciesD,
                                                   innerBlockSizeEloc,
                                                   psiQuadsFlatD,
                                                   gradPsiQuadsXFlatD,
                                                   gradPsiQuadsYFlatD,
                                                   gradPsiQuadsZFlatD,
	                                           eshelbyTensorContributionsD00,
			                           eshelbyTensorContributionsD10,
			                           eshelbyTensorContributionsD11,
			                           eshelbyTensorContributionsD20,
			                           eshelbyTensorContributionsD21,
			                           eshelbyTensorContributionsD22,
                                                   eshelbyTensorQuadValuesD00,
                                                   eshelbyTensorQuadValuesD10,
                                                   eshelbyTensorQuadValuesD11,
                                                   eshelbyTensorQuadValuesD20,
                                                   eshelbyTensorQuadValuesD21,
                                                   eshelbyTensorQuadValuesD22);

	   cudaDeviceSynchronize();
	   MPI_Barrier(MPI_COMM_WORLD);
	   kernel1_time = MPI_Wtime() - kernel1_time;
	    
	   if (this_process==0 && dftParameters::verbosity>=2)
		 std::cout<<"Time for interpolatePsiComputeELocWfcEshelbyTensorNonPeriodicD inside blocked loop: "<<kernel1_time<<std::endl;

           if (isPsp)
           {
                   thrust::device_vector<double> psiQuadsNLPFlatD;
                   if (interpolateForNLPQuad)
                   {
                           psiQuadsFlatD.clear();
			   psiQuadsNLPFlatD.resize(numCells*numQuadsNLP*numPsi,0.0);
			   interpolatePsiNLPD(operatorMatrix,
				    cudaFlattenedArrayBlock,
				    numPsi,
				    N,
				    numCells,
				    numQuadsNLP,
				    numNodesPerElement,
				    psiQuadsNLPFlatD);
                   }

		   cudaDeviceSynchronize();
		   MPI_Barrier(MPI_COMM_WORLD);
		   double kernel2_time = MPI_Wtime();

		   operatorMatrix.computeNonLocalProjectorKetTimesXTimesV(cudaFlattenedArrayBlock.begin(),
									   projectorKetTimesVectorD,
									   numPsi);

		   cudaDeviceSynchronize();
		   MPI_Barrier(MPI_COMM_WORLD);
		   kernel2_time = MPI_Wtime() - kernel2_time;
		    
		   if (this_process==0 && dftParameters::verbosity>=2)
			 std::cout<<"Time for computeNonLocalProjectorKetTimesXTimesV inside blocked loop: "<<kernel2_time<<std::endl;

		   cudaDeviceSynchronize();
		   MPI_Barrier(MPI_COMM_WORLD);
		   double kernel3_time = MPI_Wtime();

		   if (totalNonTrivialPseudoWfcs>0)
		   {
			   nlpPsiContractionD(operatorMatrix,
					      interpolateForNLPQuad?psiQuadsNLPFlatD:psiQuadsFlatD,
					      partialOccupanciesD,
					      projectorKetTimesVectorD.begin(),
					      nonTrivialIdToElemIdMapD,
					      projecterKetTimesFlattenedVectorLocalIdsD,
					      numCells, 
					      numQuadsNLP,
					      numPsi,
					      totalNonTrivialPseudoWfcs,
                                              innerBlockSizeEnlp,
                                              nlpContractionContributionD,
					      projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedD);
		   }

		   cudaDeviceSynchronize();
		   MPI_Barrier(MPI_COMM_WORLD);
		   kernel3_time = MPI_Wtime() - kernel3_time;
		    
		   if (this_process==0 && dftParameters::verbosity>=2)
			 std::cout<<"Time for nlpPsiContractionD inside blocked loop: "<<kernel3_time<<std::endl;
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
			     double * eshelbyTensorQuadValuesH00,
			     double * eshelbyTensorQuadValuesH10,
			     double * eshelbyTensorQuadValuesH11,
			     double * eshelbyTensorQuadValuesH20,
			     double * eshelbyTensorQuadValuesH21,
			     double * eshelbyTensorQuadValuesH22,
			     double * projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
                             const MPI_Comm & interBandGroupComm,
                             const bool isPsp,
			     const bool interpolateForNLPQuad)
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

            cudaVectorType cudaFlattenedArrayBlock;
            cudaVectorType projectorKetTimesVectorD;
	    vectorTools::createDealiiVector(operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
					   blockSize,
					   cudaFlattenedArrayBlock);
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
            thrust::device_vector<double> elocWfcEshelbyTensorQuadValuesD00(numCells*numQuads,0.0);
            thrust::device_vector<double> elocWfcEshelbyTensorQuadValuesD10(numCells*numQuads,0.0);
            thrust::device_vector<double> elocWfcEshelbyTensorQuadValuesD11(numCells*numQuads,0.0);
            thrust::device_vector<double> elocWfcEshelbyTensorQuadValuesD20(numCells*numQuads,0.0);
            thrust::device_vector<double> elocWfcEshelbyTensorQuadValuesD21(numCells*numQuads,0.0);
            thrust::device_vector<double> elocWfcEshelbyTensorQuadValuesD22(numCells*numQuads,0.0);


            thrust::device_vector<double> psiQuadsFlatD(numCells*numQuads*blockSize,0.0);
            thrust::device_vector<double> gradPsiQuadsXFlatD(numCells*numQuads*blockSize,0.0);
            thrust::device_vector<double> gradPsiQuadsYFlatD(numCells*numQuads*blockSize,0.0);
            thrust::device_vector<double> gradPsiQuadsZFlatD(numCells*numQuads*blockSize,0.0);

            const unsigned int innerBlockSizeEloc=50;
            thrust::device_vector<double> eshelbyTensorContributionsD00(innerBlockSizeEloc*numQuads*blockSize,0.0);
            thrust::device_vector<double> eshelbyTensorContributionsD10(innerBlockSizeEloc*numQuads*blockSize,0.0); 
            thrust::device_vector<double> eshelbyTensorContributionsD11(innerBlockSizeEloc*numQuads*blockSize,0.0); 
            thrust::device_vector<double> eshelbyTensorContributionsD20(innerBlockSizeEloc*numQuads*blockSize,0.0); 
            thrust::device_vector<double> eshelbyTensorContributionsD21(innerBlockSizeEloc*numQuads*blockSize,0.0); 
            thrust::device_vector<double> eshelbyTensorContributionsD22(innerBlockSizeEloc*numQuads*blockSize,0.0); 

            const unsigned int innerBlockSizeEnlp=200;
            thrust::device_vector<double> nlpContractionContributionD(innerBlockSizeEnlp*numQuadsNLP*blockSize,0.0);
            thrust::device_vector<double> projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedD;
	    thrust::device_vector<unsigned int> projecterKetTimesFlattenedVectorLocalIdsD;
	    thrust::device_vector<unsigned int> nonTrivialIdToElemIdMapD;
            if (totalNonTrivialPseudoWfcs>0)
            {
		    projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedD.resize(totalNonTrivialPseudoWfcs*numQuadsNLP,0.0);
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
                      
                      cudaDeviceSynchronize();
                      MPI_Barrier(MPI_COMM_WORLD);
                      double kernel_time = MPI_Wtime();

		      gpuPortedForceKernelsAllD(operatorMatrix,
                                               cudaFlattenedArrayBlock,
                                               projectorKetTimesVectorD,
				               X,
					       eigenValuesD,
					       partialOccupanciesD,
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
	                                       eshelbyTensorContributionsD00,
			                       eshelbyTensorContributionsD10,
			                       eshelbyTensorContributionsD11,
			                       eshelbyTensorContributionsD20,
			                       eshelbyTensorContributionsD21,
			                       eshelbyTensorContributionsD22,
					       elocWfcEshelbyTensorQuadValuesD00,
					       elocWfcEshelbyTensorQuadValuesD10,
					       elocWfcEshelbyTensorQuadValuesD11,
					       elocWfcEshelbyTensorQuadValuesD20,
					       elocWfcEshelbyTensorQuadValuesD21,
					       elocWfcEshelbyTensorQuadValuesD22,
                                               nlpContractionContributionD,
					       projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedD,
                                               innerBlockSizeEloc,
                                               innerBlockSizeEnlp,
                                               isPsp,
			                       interpolateForNLPQuad);

		      cudaDeviceSynchronize();
		      MPI_Barrier(MPI_COMM_WORLD);
		      kernel_time = MPI_Wtime() - kernel_time;
		    
		      if (this_process==0 && dftParameters::verbosity>=2)
		         std::cout<<"Time for force kernels all insided block loop: "<<kernel_time<<std::endl;
                 }//band parallelization
            }//ivec loop

            cudaMemcpy(eshelbyTensorQuadValuesH00,
		      thrust::raw_pointer_cast(&elocWfcEshelbyTensorQuadValuesD00[0]),
		      numCells*numQuads*sizeof(double),
		      cudaMemcpyDeviceToHost);  

            cudaMemcpy(eshelbyTensorQuadValuesH10,
		      thrust::raw_pointer_cast(&elocWfcEshelbyTensorQuadValuesD10[0]),
		      numCells*numQuads*sizeof(double),
		      cudaMemcpyDeviceToHost);   
            cudaMemcpy(eshelbyTensorQuadValuesH11,
		      thrust::raw_pointer_cast(&elocWfcEshelbyTensorQuadValuesD11[0]),
		      numCells*numQuads*sizeof(double),
		      cudaMemcpyDeviceToHost);   
            cudaMemcpy(eshelbyTensorQuadValuesH20,
		      thrust::raw_pointer_cast(&elocWfcEshelbyTensorQuadValuesD20[0]),
		      numCells*numQuads*sizeof(double),
		      cudaMemcpyDeviceToHost);   
            cudaMemcpy(eshelbyTensorQuadValuesH21,
		      thrust::raw_pointer_cast(&elocWfcEshelbyTensorQuadValuesD21[0]),
		      numCells*numQuads*sizeof(double),
		      cudaMemcpyDeviceToHost);   
            cudaMemcpy(eshelbyTensorQuadValuesH22,
		      thrust::raw_pointer_cast(&elocWfcEshelbyTensorQuadValuesD22[0]),
		      numCells*numQuads*sizeof(double),
		      cudaMemcpyDeviceToHost);  

            if (totalNonTrivialPseudoWfcs>0)
			   cudaMemcpy(projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
				      thrust::raw_pointer_cast(&projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedD[0]),
				      totalNonTrivialPseudoWfcs*numQuadsNLP*sizeof(double),
				      cudaMemcpyDeviceToHost); 
            cudaDeviceSynchronize();
            MPI_Barrier(MPI_COMM_WORLD);
            gpu_time = MPI_Wtime() - gpu_time;
            
            if (this_process==0 && dftParameters::verbosity>=1)
              std::cout<<"Time taken for all gpu kernels force computation: "<<gpu_time<<std::endl;
     }

   }//forceCUDA namespace
}//dftfe namespace
