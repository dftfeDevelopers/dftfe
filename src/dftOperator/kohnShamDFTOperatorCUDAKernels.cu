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
//

#include <kohnShamDFTOperatorCUDAKernels.h>

namespace dftfe
{

   namespace
   {
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

	__global__
	void memCpyKernel(const unsigned int size,
			    const double *copyFromVec,
			    double *copyToVec)
	{

	  const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;

	  for(unsigned int index = globalThreadId; index < size; index+= blockDim.x*gridDim.x)
	   {
	      copyToVec[index]
		      =copyFromVec[index];
	   }

	}


	__global__
	void addKernel(const unsigned int size,
		       const double *addVec,
		       double *addToVec)
	{

	  const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;

	  for(unsigned int index = globalThreadId; index < size; index+= blockDim.x*gridDim.x)
	   {
	      addToVec[index]
		      +=addVec[index];
	   }

	}


	__global__
	void daxpyCUDAKernel(const unsigned int contiguousBlockSize,
			     const unsigned int numContiguousBlocks,
			     const double *xVec,
			     double *yVec,
			     const dealii::types::global_dof_index *yVecStartingContiguousBlockIds,
			     const double a)
	{
	  const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
	  const unsigned int numberEntries = numContiguousBlocks*contiguousBlockSize;

	  for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
	   {
	      unsigned int blockIndex = index/contiguousBlockSize;
	      unsigned int intraBlockIndex=index%contiguousBlockSize;
	      yVec[yVecStartingContiguousBlockIds[blockIndex]+intraBlockIndex]
		      +=a*xVec[index];
	   }

	}

	__global__
	void daxpyBinCUDAKernel(const unsigned int innerContiguousBlockSize,
				const unsigned int outerContiguousBlockSize,
				const unsigned int numContiguousBlocks,
				const unsigned int *binContents,
				const double *xVec,
				double *yVec,
				const dealii::types::global_dof_index *yVecStartingContiguousBlockIds,
				const double a)
	{
	  const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
	  const unsigned int numberEntries = numContiguousBlocks*innerContiguousBlockSize;

	  for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
	   {
	     unsigned int blockIndex = index/innerContiguousBlockSize;//degree of freedom index
	     unsigned int outerBlockIndex = blockIndex/outerContiguousBlockSize; //cell id
	     unsigned int actualOuterBlockIndex = binContents[outerBlockIndex];
	     unsigned int actualIndex = actualOuterBlockIndex*outerContiguousBlockSize*innerContiguousBlockSize;
	     unsigned int intraOuterBlockIndex = blockIndex%outerContiguousBlockSize;
	     unsigned int intraInnerBlockIndex=index%innerContiguousBlockSize;
	      yVec[yVecStartingContiguousBlockIds[blockIndex]+intraInnerBlockIndex]
		      +=a*xVec[actualIndex+intraOuterBlockIndex*innerContiguousBlockSize+intraInnerBlockIndex];
	   }

	}


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
	void daxpyAtomicAddKernelNonBoundary(const unsigned int contiguousBlockSize,
		    const unsigned int numContiguousBlocks,
		    const double *addFromVec,
                    const unsigned int * boundaryIdVec,
		    double *addToVec,
		    const dealii::types::global_dof_index *addToVecStartingContiguousBlockIds)
	{

	     const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
	     const unsigned int numberEntries = numContiguousBlocks*contiguousBlockSize;

	     for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
	     {
		  const unsigned int blockIndex = index/contiguousBlockSize;
		  const unsigned int intraBlockIndex=index%contiguousBlockSize;
                  const unsigned int flattenedId=addToVecStartingContiguousBlockIds[blockIndex];
                  if (boundaryIdVec[flattenedId/contiguousBlockSize]==0)
		      atomicAdd(&addToVec[flattenedId+intraBlockIndex], addFromVec[index]);
	     }

	}

        __global__
        void copyToParallelNonLocalVecFromReducedVec(const unsigned int numWfcs,
                               const unsigned int totalPseudoWfcs,
                               const double * reducedProjectorKetTimesWfcVec,
                               double *projectorKetTimesWfcParallelVec,
                               const unsigned int * indexMapFromParallelVecToReducedVec)
        {
          const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
          const unsigned int numberEntries = totalPseudoWfcs*numWfcs;

          for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
           {
              const unsigned int blockIndex = index/numWfcs;
              const unsigned int intraBlockIndex=index%numWfcs;
              //projectorKetTimesWfcParallelVec[index]
              //        =reducedProjectorKetTimesWfcVec[indexMapFromParallelVecToReducedVec[blockIndex]*numWfcs+intraBlockIndex];
              projectorKetTimesWfcParallelVec[indexMapFromParallelVecToReducedVec[blockIndex]*numWfcs+intraBlockIndex]
                      =reducedProjectorKetTimesWfcVec[index];

           }

        }

        __global__
        void copyFromParallelNonLocalVecToAllCellsVec(const unsigned int numWfcs,
                               const unsigned int numNonLocalCells,
                               const unsigned int maxSingleAtomPseudoWfc,
                               const double *projectorKetTimesWfcParallelVec,
                               double * projectorKetTimesWfcAllCellsVec,
                               const int * indexMapPaddedToParallelVec)
        {
          const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
          const unsigned int numberEntries = numNonLocalCells*maxSingleAtomPseudoWfc*numWfcs;

          for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
           {
              const unsigned int blockIndex = index/numWfcs;
              const unsigned int intraBlockIndex=index%numWfcs;
              const int mappedIndex=indexMapPaddedToParallelVec[blockIndex];
              if (mappedIndex!=-1)
                projectorKetTimesWfcAllCellsVec[index]
                      =projectorKetTimesWfcParallelVec[mappedIndex*numWfcs+intraBlockIndex];
           }

        }


        __global__
        void copyToDealiiParallelNonLocalVec(const unsigned int numWfcs,
                               const unsigned int totalPseudoWfcs,
                               const double *projectorKetTimesWfcParallelVec,
                               double * projectorKetTimesWfcDealiiParallelVec,
                               const unsigned int * indexMapDealiiParallelNumbering)
        {
          const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
          const unsigned int numberEntries = totalPseudoWfcs*numWfcs;

          for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
           {
              const unsigned int blockIndex = index/numWfcs;
              const unsigned int intraBlockIndex=index%numWfcs;
              const unsigned int mappedIndex=indexMapDealiiParallelNumbering[blockIndex];
                
              projectorKetTimesWfcDealiiParallelVec[mappedIndex*numWfcs+intraBlockIndex]
                      =projectorKetTimesWfcParallelVec[index];
           }
        }


        __global__
        void copyFromDealiiParallelNonLocalVec(const unsigned int numWfcs,
                               const unsigned int totalPseudoWfcs,
                               double *projectorKetTimesWfcParallelVec,
                               const double * projectorKetTimesWfcDealiiParallelVec,
                               const unsigned int * indexMapDealiiParallelNumbering)
        {
          const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
          const unsigned int numberEntries = totalPseudoWfcs*numWfcs;

          for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
           {
              const unsigned int blockIndex = index/numWfcs;
              const unsigned int intraBlockIndex=index%numWfcs;
              const unsigned int mappedIndex=indexMapDealiiParallelNumbering[blockIndex];
                
              projectorKetTimesWfcParallelVec[index]=projectorKetTimesWfcDealiiParallelVec[mappedIndex*numWfcs+intraBlockIndex];
           }
        }

        __global__
        void addNonLocalContributionCUDAKernel(const dealii::types::global_dof_index contiguousBlockSize,
                             const dealii::types::global_dof_index numContiguousBlocks,
                             const double *xVec,
                             double *yVec,
                             const unsigned int *xVecToyVecBlockIdMap)
        {
          const dealii::types::global_dof_index globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
          const dealii::types::global_dof_index numberEntries = numContiguousBlocks*contiguousBlockSize;

          for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
           {
              dealii::types::global_dof_index blockIndex = index/contiguousBlockSize;
              dealii::types::global_dof_index intraBlockIndex=index%contiguousBlockSize;
              yVec[xVecToyVecBlockIdMap[blockIndex]*contiguousBlockSize+intraBlockIndex]
                      +=xVec[index];
           }
        }

}
