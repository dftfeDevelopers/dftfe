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

      void computeELocWfcEshelbyTensorNonPeriodicH(const double * psiQuadValuesH,
                                                 const double * gradPsiQuadValuesXH,
                                                 const double * gradPsiQuadValuesYH,
                                                 const double * gradPsiQuadValuesZH,
                                                 const double * eigenValuesH,
                                                 const double * partialOccupanciesH,
                                                 const unsigned int numCells,
                                                 const unsigned int numPsi,
                                                 const unsigned int numQuads,
                                                 double * eshelbyTensorQuadValuesH)
      { 
    
           thrust::device_vector<double> psiQuadValuesD(numPsi*numQuads,0.0);
           thrust::device_vector<double> gradPsiQuadValuesXD(numPsi*numQuads,0.0);
           thrust::device_vector<double> gradPsiQuadValuesYD(numPsi*numQuads,0.0);
           thrust::device_vector<double> gradPsiQuadValuesZD(numPsi*numQuads,0.0);
           thrust::device_vector<double> eigenValuesD(numPsi,0.0);
           thrust::device_vector<double> partialOccupanciesD(numPsi,0.0);
           thrust::device_vector<double> eshelbyTensorQuadValuesD(numCells*6,0.0);


           cudaMemcpy(thrust::raw_pointer_cast(&psiQuadValuesD[0]),
		      &psiQuadValuesH,
		      numCells*numPsi*numQuads*sizeof(double),
		      cudaMemcpyHostToDevice);

           cudaMemcpy(thrust::raw_pointer_cast(&gradPsiQuadValuesXD[0]),
		      &gradPsiQuadValuesXH,
		      numCells*numPsi*numQuads*sizeof(double),
		      cudaMemcpyHostToDevice);

           cudaMemcpy(thrust::raw_pointer_cast(&gradPsiQuadValuesYD[0]),
		      &gradPsiQuadValuesYD,
		      numCells*numPsi*numQuads*sizeof(double),
		      cudaMemcpyHostToDevice);

           cudaMemcpy(thrust::raw_pointer_cast(&gradPsiQuadValuesZD[0]),
		      &gradPsiQuadValuesZD,
		      numCells*numPsi*numQuads*sizeof(double),
		      cudaMemcpyHostToDevice);

           cudaMemcpy(thrust::raw_pointer_cast(&eigenValuesD[0]),
		      &eigenValuesH,
		      numCells*numPsi*sizeof(double),
		      cudaMemcpyHostToDevice);

           cudaMemcpy(thrust::raw_pointer_cast(&partialOccupanciesD[0]),
		      &partialOccupanciesH,
		      numCells*numPsi*sizeof(double),
		      cudaMemcpyHostToDevice);

           computeELocWfcEshelbyTensorNonPeriodic(thrust::raw_pointer_cast(&psiQuadValuesD[0]),
                                                  thrust::raw_pointer_cast(&gradPsiQuadValuesXD[0]),
                                                  thrust::raw_pointer_cast(&gradPsiQuadValuesYD[0]),
                                                  thrust::raw_pointer_cast(&gradPsiQuadValuesZD[0]),
                                                  thrust::raw_pointer_cast(&eigenValuesD[0]),
                                                  thrust::raw_pointer_cast(&partialOccupanciesD[0]),
                                                  numCells,
                                                  numPsi,
                                                  numQuads,
                                                  thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD[0])); 

           cudaMemcpy(eshelbyTensorQuadValuesH,
		      thrust::raw_pointer_cast(&eshelbyTensorQuadValuesD[0]),
		      numCells*6*sizeof(double),
		      cudaMemcpyDeviceToHost);   
          
      }

      void computeELocWfcEshelbyTensorNonPeriodic(const double * psiQuadValuesD,
                                                 const double * gradPsiQuadValuesXD,
                                                 const double * gradPsiQuadValuesYD,
                                                 const double * gradPsiQuadValuesZD,
                                                 const double * eigenValuesD,
                                                 const double * partialOccupanciesD,
                                                 const unsigned int numCells, 
                                                 const unsigned int numPsi,
                                                 const unsigned int numQuads,
                                                 double * eshelbyTensorQuadValuesD)
      { 
    
          
      }

   }
}
