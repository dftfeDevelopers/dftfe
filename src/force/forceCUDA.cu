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
		      unsigned int blockIndex = index/contiguousBlockSize;
		      unsigned int intraBlockIndex=index-blockIndex*contiguousBlockSize;

                      const double psi=psiQuadValues[index];
                      const double gradPsiX=gradPsiQuadValuesX[index];
                      const double gradPsiY=gradPsiQuadValuesY[index];
                      const double gradPsiZ=gradPsiQuadValuesZ[index];
                      const double eigenValue=eigenValues[intraBlockIndex];
                      const double partOcc=partialOccupancies[intraBlockIndex];
	 // identityTensorFactor+=make_vectorized_array(partialOccupancies_[eigenIndex])*scalar_product(gradPsi,gradPsi)-make_vectorized_array(2*partialOccupancies_[eigenIndex]*eigenValues_[eigenIndex])*psi*psi;
	  //eshelbyTensor-=make_vectorized_array(2.0*partialOccupancies_[eigenIndex])*outer_product(gradPsi,gradPsi);

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
                      const unsigned int pseudoWfcId=blockIndex/numQuadsNLP;
                      const unsigned int quadId=blockIndex-pseudoWfcId*numQuadsNLP;
                      //double temp=projectorKetTimesVectorPar[projecterKetTimesFlattenedVectorLocalIds[pseudoWfcId]*numPsi+wfcId];
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


     void nlpPsiContractionH(operatorDFTCUDAClass & operatorMatrix,
			    const double * psiQuadValuesNLPH,
                            const double * partialOccupanciesH,
                            const double * projectorKetTimesVectorParFlattenedH,
                            const unsigned int * nonTrivialIdToElemIdMapH,
                            const unsigned int * projecterKetTimesFlattenedVectorLocalIdsH,
                            const unsigned int numCells,
			    const unsigned int numQuadsNLP,
			    const unsigned int numPsi,
                            const unsigned int totalNonTrivialPseudoWfcs,
                            double * projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH)
     {
           thrust::device_vector<double> psiQuadValuesNLPD(numCells*numQuadsNLP*numPsi,0.0);
           thrust::device_vector<double> partialOccupanciesD(numPsi,0.0);
           thrust::device_vector<double> projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedD(totalNonTrivialPseudoWfcs*numQuadsNLP,0.0);

           cudaVectorType projectorKetTimesVectorD;
           vectorTools::createDealiiVector(operatorMatrix.getProjectorKetTimesVectorSingle().get_partitioner(),
					   numPsi,
					   projectorKetTimesVectorD);
           projectorKetTimesVectorD=0.0;

           if (totalNonTrivialPseudoWfcs>0)
           {
		   thrust::device_vector<unsigned int> projecterKetTimesFlattenedVectorLocalIdsD(totalNonTrivialPseudoWfcs,0.0);
		   thrust::device_vector<unsigned int> nonTrivialIdToElemIdMapD(totalNonTrivialPseudoWfcs,0.0);

		   cudaMemcpy(thrust::raw_pointer_cast(&psiQuadValuesNLPD[0]),
			      psiQuadValuesNLPH,
			      numCells*numQuadsNLP*numPsi*sizeof(double),
			      cudaMemcpyHostToDevice);

		   cudaMemcpy(thrust::raw_pointer_cast(&partialOccupanciesD[0]),
			      partialOccupanciesH,
			      numPsi*sizeof(double),
			      cudaMemcpyHostToDevice);

		   cudaMemcpy(projectorKetTimesVectorD.begin(),
			      projectorKetTimesVectorParFlattenedH,
			      (projectorKetTimesVectorD.local_size()+projectorKetTimesVectorD.get_partitioner()->n_ghost_indices())*sizeof(double),
			      cudaMemcpyHostToDevice);

		   cudaMemcpy(thrust::raw_pointer_cast(&nonTrivialIdToElemIdMapD[0]),
			      nonTrivialIdToElemIdMapH,
			      totalNonTrivialPseudoWfcs*sizeof(unsigned int),
			      cudaMemcpyHostToDevice);


		   cudaMemcpy(thrust::raw_pointer_cast(&projecterKetTimesFlattenedVectorLocalIdsD[0]),
			      projecterKetTimesFlattenedVectorLocalIdsH,
			      totalNonTrivialPseudoWfcs*sizeof(unsigned int),
			      cudaMemcpyHostToDevice);


		   cudaMemcpy(thrust::raw_pointer_cast(&projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedD[0]),
			      projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
			      totalNonTrivialPseudoWfcs*numQuadsNLP*sizeof(double),
			      cudaMemcpyHostToDevice);

		   nlpPsiContractionD(operatorMatrix,
				      psiQuadValuesNLPD,
				      partialOccupanciesD,
				      projectorKetTimesVectorD.begin(),
				      nonTrivialIdToElemIdMapD,
				      projecterKetTimesFlattenedVectorLocalIdsD,
				      numCells, 
				      numQuadsNLP,
				      numPsi,
				      totalNonTrivialPseudoWfcs,
				      projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedD);

		   cudaMemcpy(projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedH,
			      thrust::raw_pointer_cast(&projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedD[0]),
			      totalNonTrivialPseudoWfcs*numQuadsNLP*sizeof(double),
			      cudaMemcpyDeviceToHost);
	 }
     }

     void nlpPsiContractionD(operatorDFTCUDAClass & operatorMatrix,
			    const thrust::device_vector<double> & psiQuadValuesNLPD,
                            const thrust::device_vector<double> & partialOccupanciesD,
                            const double * projectorKetTimesVectorParFlattenedD,
                            thrust::device_vector<unsigned int> & nonTrivialIdToElemIdMapD,
                            thrust::device_vector<unsigned int> & projecterKetTimesFlattenedVectorLocalIdsD,
                            const unsigned int numCells, 
			    const unsigned int numQuadsNLP,
			    const unsigned int numPsi,
                            const unsigned int totalNonTrivialPseudoWfcs,
                            thrust::device_vector<double> & projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedD)
     {
            thrust::device_vector<double> nlpContractionContributionD(totalNonTrivialPseudoWfcs*numQuadsNLP*numPsi,0.0);
            thrust::device_vector<double> onesMatD(numPsi,1.0);
           

	    nlpPsiContractionCUDAKernel<<<(numPsi+255)/256*numQuadsNLP*totalNonTrivialPseudoWfcs,256>>>
							  (numPsi,
							   numQuadsNLP,
							   totalNonTrivialPseudoWfcs,
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
		      totalNonTrivialPseudoWfcs*numQuadsNLP,
		      numPsi,
		      &scalarCoeffAlpha,
		      thrust::raw_pointer_cast(&onesMatD[0]),
		      1,
		      thrust::raw_pointer_cast(&nlpContractionContributionD[0]),
		      numPsi,
		      &scalarCoeffBeta,
		      thrust::raw_pointer_cast(&projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattenedD[0]),
		      1);
     }

   }//forceCUDA namespace
}//dftfe namespace
