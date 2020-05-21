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
// @author Sambit Das, Phani Motamarri


#if defined(DFTFE_WITH_GPU)
#include<linearAlgebraOperationsInternalCUDA.h>
#include<linearAlgebraOperationsCUDA.h>
#include<dftParameters.h>
#include<vectorUtilities.h>
#include <dftUtils.h>
#include <nvToolsExt.h>

namespace dftfe
{
  namespace linearAlgebraOperationsCUDA
  {
    namespace
    {
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

      //y=a*x+b*y, with inc=1

      __global__
      void daxpbyCUDAKernel(const int n,
			    const double *x,
			    double *y,
			    const double a,
			    const double b)
      {
                
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
	     i < n; 
	     i += blockDim.x * gridDim.x) 
	  {
	    y[i] = a * x[i] + b*y[i];
	  }
                
      }

      __global__
      void combinedCUDAKernel(const unsigned int contiguousBlockSize,
			      const unsigned int numContiguousBlocks,
			      double *x,
			      double *y,
			      const double a,
			      const double b,
			      const double scalar,
			      const double scalarOld,
			      const double *invSqrtMassVec,
			      const double *sqrtMassVec)
      {
	const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int numberEntries = numContiguousBlocks*contiguousBlockSize;

	for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
	  {
	    unsigned int blockIndex = index/contiguousBlockSize;
	    *(y+index) *= (*(sqrtMassVec+blockIndex)*1.0/scalarOld);
	    *(x+index) *= (*(invSqrtMassVec+blockIndex));
	    y[index] = a * x[index] + b*y[index];
	    *(x+index) *= (*(invSqrtMassVec+blockIndex)*scalar);
	    *(y+index) *= (*(sqrtMassVec+blockIndex));
	  }

      }


      __global__
      void scaleXArrayRayleighQuotientsCUDAKernel(const unsigned int numVectors,
			      const unsigned int numBoundaryPlusGhostNodes,
                              const unsigned int * boundaryGhostIdToLocalIdMap, 
                              const double * rayleighQuotients,  
			      const double *y,
                              const double *sqrtMassVec,
                              double * x)
      {
	const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int numberEntries = numVectors*numBoundaryPlusGhostNodes;

	for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
	  {
            const unsigned int blockIndex=index/numVectors;
	    const unsigned int intraBlockIndex = index%numVectors;
            const unsigned int localId=boundaryGhostIdToLocalIdMap[blockIndex];
            const unsigned int flattenedWfcId=localId*numVectors+intraBlockIndex;
            x[flattenedWfcId]=y[flattenedWfcId]*rayleighQuotients[intraBlockIndex]*sqrtMassVec[localId]*sqrtMassVec[localId];

	  }

      }

      __global__
      void addScaleXArrayRayleighQuotientsCUDAKernel(const unsigned int numVectors,
			      const unsigned int numBoundaryPlusGhostNodes,
                              const unsigned int * boundaryGhostIdToLocalIdMap, 
                              const double * rayleighQuotients,  
			      const double *y,
                              const double *sqrtMassVec,
                              double * x)
      {
	const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int numberEntries = numVectors*numBoundaryPlusGhostNodes;

	for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
	  {
            const unsigned int blockIndex=index/numVectors;
	    const unsigned int intraBlockIndex = index%numVectors;
            const unsigned int localId=boundaryGhostIdToLocalIdMap[blockIndex];
            const unsigned int flattenedWfcId=localId*numVectors+intraBlockIndex;
            x[flattenedWfcId]+=y[flattenedWfcId]*rayleighQuotients[intraBlockIndex]*sqrtMassVec[localId]*sqrtMassVec[localId];

	  }

      }

      __global__
      void copySubspaceRotatedBlockToXKernel(const unsigned int BDof,
					     const float *rotatedXBlockSP,
					     const double *diagValues,
					     double *X,
					     const unsigned int startingDofId,
					     const unsigned int N)
      {
          
        const unsigned int numEntries=N*BDof;        
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
	     i < numEntries; 
	     i += blockDim.x * gridDim.x) 
	  {
            const unsigned int ibdof = i/N;
            const unsigned int ivec=  i%N; 

	    *(X+N*(startingDofId+ibdof)+ivec)
	      = *(X+N*(startingDofId+ibdof)+ivec)*diagValues[ivec]
	      +rotatedXBlockSP[ibdof*N+ivec];
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

      //R=Y-X*Gamma
      __global__
      void computeResidualCUDAKernel(const unsigned int numVectors,
				     const unsigned int numDofs,
				     const unsigned int N,
				     const unsigned int startingVecId,
				     const double *eigenValues,
				     const double *x,
				     const double *y,
				     double * r)
      {
                
	for (int i = blockIdx.x * blockDim.x + threadIdx.x; 
	     i < numVectors*numDofs; 
	     i += blockDim.x * gridDim.x) 
	  {
            const unsigned int dofIndex = i/numVectors;
            const unsigned int waveIndex = i%numVectors; 
	    r[i]=y[i]-x[dofIndex*N+startingVecId+waveIndex]*eigenValues[startingVecId+waveIndex];
            r[i]=r[i]*r[i];
	  }
                
      }

      __global__
      void convFloatArrToDoubleArr(const unsigned int size,
				   const float *floatArr,
				   double *doubleArr)
      {

	const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;

	for(unsigned int index = globalThreadId; index < size; index+= blockDim.x*gridDim.x)
	  doubleArr[index]=floatArr[index];

      }

      __global__
      void copyFloatArrToDoubleArrLocallyOwned(const unsigned int contiguousBlockSize,
					       const unsigned int numContiguousBlocks,
					       const float *floatArr,
					       const unsigned int *locallyOwnedFlagArr,
					       double *doubleArr)
      {

	const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int numberEntries = numContiguousBlocks*contiguousBlockSize;

	for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
	  {
	    unsigned int blockIndex = index/contiguousBlockSize;
	    if(locallyOwnedFlagArr[blockIndex] == 1)
	      doubleArr[index]=floatArr[index];
	  }

      }

      __global__
      void dotProductContributionBlockedKernelMassVector(const unsigned int contiguousBlockSize,
					       const unsigned int numContiguousBlocks,
			   const double *vec1,
			   const double *vec2,
                           const double * sqrtMassVector,
                           double * vecTemp)
      { 
        const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
	const unsigned int numberEntries = numContiguousBlocks*contiguousBlockSize;

	for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
	 {
	   const unsigned int blockIndex = index/contiguousBlockSize;
           const double temp= sqrtMassVector[blockIndex];
           vecTemp[2*index]=vec1[index]*vec2[index];           
	   vecTemp[2*index+1]=vec1[index]*vec1[index]*temp*temp;
	 }


      }

      void computeRayleighQuotients(cublasHandle_t &handle,
			 const double * xarray,
			 const double * yarray,
                         const double * sqrtMassVector,
			 const double * onesVec,
			 const unsigned int numberVectors,
			 const unsigned int localSize,
                         const MPI_Comm & mpiComm,
                         MPI_Request & request,
                         double * temparray,
                         double * dotarrayD,
                         double * dotarrayH)
      {
             
	     dotProductContributionBlockedKernelMassVector<<<(numberVectors+255)/256*localSize,256>>>(numberVectors,
                                                localSize,
						xarray,
						yarray,
                                                sqrtMassVector,
                                                temparray);

	     const double alpha = 1.0, beta = 0.0;
	     cublasDgemm(handle,
			  CUBLAS_OP_N,
			  CUBLAS_OP_T,
			  1,
			  2*numberVectors,
			  localSize,
			  &alpha,
			  onesVec,
			  1,
			  temparray,
			  2*numberVectors,
			  &beta,
			  dotarrayD,
			  1);

            
	     cudaMemcpy(dotarrayH,
		        dotarrayD,
		   2*numberVectors*sizeof(double),
		   cudaMemcpyDeviceToHost);
             


	     MPI_Iallreduce(MPI_IN_PLACE,
			  dotarrayH,
			  2*numberVectors,
			  MPI_DOUBLE,
			  MPI_SUM,
			  mpiComm,
                          &request);
              
      }

      void checkRayleighQuotients(const unsigned int numberVectors,
                         const double tolCommun,
                         const double tolCompute,
                         double * dotarrayH,
                         double * rayleighQuotientsH,
                         double * rayleighQuotientsDiffH,
                         bool & isConvergedToTol1,
                         bool & isConvergedToTol2)
      {

             isConvergedToTol1=true;
             isConvergedToTol2=true;

             for (unsigned int i=0; i<numberVectors;++i)
             {
                   const double temp=rayleighQuotientsH[i];
                   //std::cout<<"ytdotxarrayH: "<<ytdotxarrayH[i]<<std::endl;
                   //std::cout<<"xtdotxarrayH: "<<xtdotxarrayH[i]<<std::endl;
                   rayleighQuotientsH[i]= dotarrayH[2*i]/dotarrayH[2*i+1];
                   const double diff=rayleighQuotientsH[i]-temp;
                   if (std::fabs(diff)>tolCommun)
                       isConvergedToTol1=false;
                   if (std::fabs(diff)>tolCompute)
                       isConvergedToTol2=false;

                   //rayleighQuotientsDiffH[i]=diff;
             }
      }

      void checkRayleighQuotients(const unsigned int numberVectors,
                         const double tolCompute,
                         double * dotarrayH,
                         double * rayleighQuotientsH,
                         double * rayleighQuotientsDiffH,
                         bool & isConvergedToTol)
      {

             isConvergedToTol=true;

             for (unsigned int i=0; i<numberVectors;++i)
             {
                   const double temp=rayleighQuotientsH[i];
                   //std::cout<<"ytdotxarrayH: "<<ytdotxarrayH[i]<<std::endl;
                   //std::cout<<"xtdotxarrayH: "<<xtdotxarrayH[i]<<std::endl;
                   rayleighQuotientsH[i]= dotarrayH[2*i]/dotarrayH[2*i+1];
                   const double diff=rayleighQuotientsH[i]-temp;
                   if (std::fabs(diff)>tolCompute)
                       isConvergedToTol=false;

                   //rayleighQuotientsDiffH[i]=diff;
             }
      }



    }

 
    //
    // evaluate upper bound of the spectrum using k-step Lanczos iteration
    //
    double lanczosUpperBoundEigenSpectrum(operatorDFTCUDAClass & operatorMatrix,
					  const distributedCPUVec<double> & vect)
    {
#ifdef USE_COMPLEX
      AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
      const unsigned int this_mpi_process = dealii::Utilities::MPI::this_mpi_process(operatorMatrix.getMPICommunicator());



      const unsigned int lanczosIterations=dftParameters::reproducible_output?40:20;
      double beta;


      dataTypes::number alpha,alphaNeg;

      //
      //generate random vector v
      //
      distributedCPUVec<double> vVector, fVector, v0Vector;
      vVector.reinit(vect);
      fVector.reinit(vect);

      vVector = 0.0,fVector = 0.0;
      //std::srand(this_mpi_process);
      const unsigned int local_size = vVector.local_size();

      for (unsigned int i = 0; i < local_size; i++)
	vVector.local_element(i) = ((double)std::rand())/((double)RAND_MAX);

      operatorMatrix.getConstraintMatrixEigen()->set_zero(vVector);
      vVector.update_ghost_values();

      //
      //evaluate l2 norm
      //
      vVector/=vVector.l2_norm();
      vVector.update_ghost_values();

      //
      //call matrix times X
      //
      std::vector<distributedCPUVec<double>> v(1),f(1);
      v[0] = vVector;
      f[0] = fVector;
      operatorMatrix.HX(v,f);
      operatorMatrix.getConstraintMatrixEigen()->set_zero(v[0]);
      fVector = f[0];

      alpha=fVector*vVector;
      fVector.add(-1.0*alpha,vVector);
      std::vector<double> T(lanczosIterations*lanczosIterations,0.0);

      T[0]=alpha;
      unsigned index=0;

      //filling only lower triangular part
      for (unsigned int j=1; j<lanczosIterations; j++)
	{
	  beta=fVector.l2_norm();
	  v0Vector = vVector; vVector.equ(1.0/beta,fVector);
	  v[0] = vVector,f[0] = fVector;
	  operatorMatrix.HX(v,f);
          operatorMatrix.getConstraintMatrixEigen()->set_zero(v[0]);
	  fVector = f[0];
	  fVector.add(-1.0*beta,v0Vector);//beta is real
	  alpha = fVector*vVector;
	  fVector.add(-1.0*alpha,vVector);
	  index+=1;
	  T[index]=beta;
	  index+=lanczosIterations;
	  T[index]=alpha;
	}

      //eigen decomposition to find max eigen value of T matrix
      std::vector<double> eigenValuesT(lanczosIterations);
      char jobz='N', uplo='L';
      const unsigned int n = lanczosIterations, lda = lanczosIterations;
      int info;
      const unsigned int lwork = 1 + 6*n + 2*n*n, liwork = 3 + 5*n;
      std::vector<int> iwork(liwork, 0);

      std::vector<double> work(lwork, 0.0);
      dsyevd_(&jobz, &uplo, &n, &T[0], &lda, &eigenValuesT[0], &work[0], &lwork, &iwork[0], &liwork, &info);


      for (unsigned int i=0; i<eigenValuesT.size(); i++){eigenValuesT[i]=std::abs(eigenValuesT[i]);}
      std::sort(eigenValuesT.begin(),eigenValuesT.end());
      //
      if (dftParameters::verbosity==2)
	{
	  char buffer[100];
	  sprintf(buffer, "bUp1: %18.10e,  bUp2: %18.10e\n", eigenValuesT[lanczosIterations-1], fVector.l2_norm());
	  //pcout << buffer;
	}
      double upperBound=eigenValuesT[lanczosIterations-1]+fVector.l2_norm();
      return (std::ceil(upperBound));
#endif
    }



    void chebyshevFilter(operatorDFTCUDAClass & operatorMatrix,
			 distributedGPUVec<double> & XArray,
                         distributedGPUVec<double> & YArray,
			 distributedGPUVec<float> & tempFloatArray,
                         distributedGPUVec<double> & projectorKetTimesVector,
			 const unsigned int localVectorSize,
			 const unsigned int numberVectors,
			 const unsigned int m,
			 const double a,
			 const double b,
			 const double a0,
                         const bool mixedPrecOverall)
    {
#ifdef USE_COMPLEX
      AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
      double e, c, sigma, sigma1, sigma2, gamma, gpu_time;
      e = (b-a)/2.0; c = (b+a)/2.0;
      sigma = e/(a0-c); sigma1 = sigma; gamma = 2.0/sigma1;
      const unsigned int totalVectorSize = localVectorSize*numberVectors;
      int inc = 1;

      YArray=0.0;
      //
      //call HX
      //
      bool scaleFlag = false;
      double scalar = 1.0;

      operatorMatrix.HX(XArray,
                        projectorKetTimesVector,
			localVectorSize,
			numberVectors,
			scaleFlag,
			scalar,
			YArray);

      double  alpha1 = sigma1/e, alpha2 = -c;
      double alpha1Old=alpha1;
  
      //
      //YArray = YArray + alpha2*XArray and YArray = alpha1*YArray
      //
      cublasDaxpy(operatorMatrix.getCublasHandle(),
		  totalVectorSize,
		  &alpha2,
		  XArray.begin(),
		  inc,
		  YArray.begin(),
		  inc);

      cublasDscal(operatorMatrix.getCublasHandle(),
		  totalVectorSize,
		  &alpha1,
		  YArray.begin(),
		  inc);
	       
	       
      //
      //polynomial loop
      //
      for(unsigned int degree = 2; degree < m+1; ++degree)
	{
	  sigma2 = 1.0/(gamma - sigma);
	  alpha1 = 2.0*sigma2/e, alpha2 = -(sigma*sigma2);

	  double coeff = -c*alpha1;
	  
	  if (degree==2)
	    {
	      daxpbyCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(totalVectorSize,
									     YArray.begin(),
									     XArray.begin(),
          		                                                     coeff,
		                                                             alpha2);


	      //scale src vector with M^{-1/2}
	      //
	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       alpha1,
									       YArray.begin(),
									       operatorMatrix.getInvSqrtMassVec());

	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       1.0,
									       XArray.begin(),
									       operatorMatrix.getSqrtMassVec());

	      //
	      //call HX
	      //
	      operatorMatrix.HXCheby(YArray,
				     tempFloatArray,
				     projectorKetTimesVector,
				     localVectorSize,
				     numberVectors,
				     XArray,
				     mixedPrecOverall && dftParameters::useMixedPrecCheby);
	    }
	  else if (degree==m)
	    {

	      //unscale src vector with M^{1/2}
	      //
	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       1.0/alpha1Old,
									       XArray.begin(),
									       operatorMatrix.getSqrtMassVec());

	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       1.0,
									       YArray.begin(),
									       operatorMatrix.getInvSqrtMassVec());

	      daxpbyCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(totalVectorSize,
									     YArray.begin(),
									     XArray.begin(),
									     coeff,
									     alpha2);
	      scaleFlag=true;
	      //
	      //call HX
	      //
	      operatorMatrix.HX(YArray,
				projectorKetTimesVector,
				localVectorSize,
				numberVectors,
				scaleFlag,
				alpha1,
				XArray);

	    }
	  else
	    {
	      combinedCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(numberVectors,
									       localVectorSize,
									       YArray.begin(),
									       XArray.begin(),
									       coeff,
									       alpha2,
									       alpha1,
									       alpha1Old,
									       operatorMatrix.getInvSqrtMassVec(),
									       operatorMatrix.getSqrtMassVec());
	      //
	      //call HX
	      //
	      operatorMatrix.HXCheby(YArray,
				     tempFloatArray,
				     projectorKetTimesVector,
				     localVectorSize,
				     numberVectors,
				     XArray,
				     mixedPrecOverall && dftParameters::useMixedPrecCheby);
          
	    }

	  XArray.swap(YArray);

      
	  sigma = sigma2;
	  alpha1Old=alpha1;

	}

      //copy back YArray to XArray
      cudaMemcpy(XArray.begin(),
		 YArray.begin(),
		 totalVectorSize*sizeof(double),
		 cudaMemcpyDeviceToDevice);
#endif
    }
  
    void chebyshevFilterComputeAvoidance(operatorDFTCUDAClass & operatorMatrix,
			 distributedGPUVec<double> & XArray,
                         distributedGPUVec<double> & YArray,
                         distributedGPUVec<double> & XArray2,
			 distributedGPUVec<float> & tempFloatArray,
                         distributedGPUVec<double> & projectorKetTimesVector,
			 const unsigned int localVectorSize,
			 const unsigned int numberVectors,
			 const unsigned int m,
			 const double a,
			 const double b,
			 const double a0,
                         const bool isXlBOMDLinearizedSolve,
                         const bool communAvoidance,
                         const bool mixedPrecOverall)
    {
#ifdef USE_COMPLEX
      AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
      double e, c, sigma, sigma1, sigma2, gamma, gpu_time;
      e = (b-a)/2.0; c = (b+a)/2.0;
      sigma = e/(a0-c); sigma1 = sigma; gamma = 2.0/sigma1;
      const unsigned int totalVectorSize = localVectorSize*numberVectors;
      const unsigned int ghostVectorSize=operatorMatrix.getMatrixFreeData()->get_vector_partitioner()->n_ghost_indices();
      int inc = 1;

      const unsigned int this_mpi_process=dealii::Utilities::MPI::this_mpi_process(operatorMatrix.getMPICommunicator());

      YArray=0;
      XArray2=0;

      thrust::device_vector<double> onesVecD(localVectorSize,1.0);
      thrust::device_vector<double> tempArrayD(2*localVectorSize*numberVectors,0.0);
      thrust::device_vector<double> dotarrayD(2*numberVectors,0.0);
      std::vector<double> dotarrayH(2*numberVectors,0.0);
      thrust::device_vector<double> rayleighQuotientsD(numberVectors,0.0);
      std::vector<double> rayleighQuotientsH(numberVectors,0.0);
      std::vector<double> rayleighQuotientsDiffH(numberVectors,0.0);

      //
      //call HX
      //
      bool scaleFlag = false;
      double scalar = 1.0;

      operatorMatrix.HX(XArray,
                        projectorKetTimesVector,
			localVectorSize,
			numberVectors,
			scaleFlag,
			scalar,
			YArray);

      double  alpha1 = sigma1/e, alpha2 = -c;
      double alpha1Old=alpha1;
  
      //
      //YArray = YArray + alpha2*XArray and YArray = alpha1*YArray
      //
      cublasDaxpy(operatorMatrix.getCublasHandle(),
		  totalVectorSize,
		  &alpha2,
		  XArray.begin(),
		  inc,
		  YArray.begin(),
		  inc);

      cublasDscal(operatorMatrix.getCublasHandle(),
		  totalVectorSize,
		  &alpha1,
		  YArray.begin(),
		  inc);
	     
      const bool useCommunAvoidanceOpt=isXlBOMDLinearizedSolve?false:false; 
      const double communAvoidanceTolerance=isXlBOMDLinearizedSolve?1e-12:1e-12;
      const double computeAvoidanceTolerance=isXlBOMDLinearizedSolve?1e-10:1e-16;  
      bool isCommunAvoidanceToleranceReached=false;
      bool isComputeAvoidanceToleranceReached=false;
      bool isFirstCallToCommunAvoidance=false;
      const unsigned int boundaryVectorSize=operatorMatrix.getBoundaryIdToLocalIdMap().size(); 

      MPI_Request request;
      int count=0;
      //
      //polynomial loop
      //
      for(unsigned int degree = 2; degree < m+1; ++degree)
	{
	  sigma2 = 1.0/(gamma - sigma);
	  alpha1 = 2.0*sigma2/e, alpha2 = -(sigma*sigma2);

	  double coeff = -c*alpha1;
	  
	  if (degree==2)
	    {
	      daxpbyCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(totalVectorSize,
									     YArray.begin(),
									     XArray.begin(),
          		                                                     coeff,
		                                                             alpha2);


	      //scale src vector with M^{-1/2}
	      //
	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       alpha1,
									       YArray.begin(),
									       operatorMatrix.getInvSqrtMassVec());

	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       1.0,
									       XArray.begin(),
									       operatorMatrix.getSqrtMassVec());

	      //
	      //call HX
	      //
	      operatorMatrix.HXCheby(YArray,
				     tempFloatArray,
				     projectorKetTimesVector,
				     localVectorSize,
				     numberVectors,
				     XArray,
				     mixedPrecOverall && dftParameters::useMixedPrecCheby);
	    }
	  else if (degree==m)
	    {

	      //unscale src vector with M^{1/2}
	      //
	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       1.0/alpha1Old,
									       XArray.begin(),
									       operatorMatrix.getSqrtMassVec());

	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       1.0,
									       YArray.begin(),
									       operatorMatrix.getInvSqrtMassVec());

	      daxpbyCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(totalVectorSize,
									     YArray.begin(),
									     XArray.begin(),
									     coeff,
									     alpha2);
	      scaleFlag=true;
	      //
	      //call HX
	      //
	      operatorMatrix.HX(YArray,
				projectorKetTimesVector,
				localVectorSize,
				numberVectors,
				scaleFlag,
				alpha1,
				XArray);

	    }
	  else
	    {
              if (isComputeAvoidanceToleranceReached || (isCommunAvoidanceToleranceReached && useCommunAvoidanceOpt))
              {
                      if (isComputeAvoidanceToleranceReached)
                      {
                        if (this_mpi_process==0)
                           std::cout<<"  Skipping Chebyshev polynomial filtering after iter: "<<(degree-1)<<std::endl;

		  
			XArray.zero_out_ghosts();
			YArray.zero_out_ghosts();

			//unscale src vector with M^{1/2}
			//
			scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
										       localVectorSize,
										       1.0/alpha1Old,
										       XArray.begin(),
										       operatorMatrix.getSqrtMassVec());

			scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
										       localVectorSize,
										       1.0,
										       YArray.begin(),
										       operatorMatrix.getInvSqrtMassVec());

			daxpbyCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(totalVectorSize,
										     YArray.begin(),
										     XArray.begin(),
										     coeff,
										     alpha2);
			scaleFlag=true;
			//
			//call HX
			//
			operatorMatrix.HX(YArray,
					projectorKetTimesVector,
					localVectorSize,
					numberVectors,
					scaleFlag,
					alpha1,
					XArray);

           	        XArray.swap(YArray);

                        break;
                      }

		      if (this_mpi_process==0 && isFirstCallToCommunAvoidance)
			std::cout<<"   Communication avoidance HX from Chebyshev polynomial iter: "<<degree<<", localVectorSize: "<<localVectorSize<<", ghostVectorSize: "<<ghostVectorSize<<", boundaryVectorSize: "<<boundaryVectorSize<<std::endl;
		      
		      combinedCUDAKernel<<<min((numberVectors*(localVectorSize+ghostVectorSize)+255)/256,30000),256>>>(numberVectors,
										       localVectorSize+ghostVectorSize,
										       YArray.begin(),
										       XArray.begin(),
										       coeff,
										       alpha2,
										       alpha1,
										       alpha1Old,
										       operatorMatrix.getInvSqrtMassVec(),
										       operatorMatrix.getSqrtMassVec());


		       if (isFirstCallToCommunAvoidance)
		       {
			  YArray.update_ghost_values();
			  XArray.update_ghost_values();
			  isFirstCallToCommunAvoidance=false;
		       }


                      bool dummy;
                      if ((degree-3)%5==0 && count>0)
                      {

			       XArray2=0;
			       operatorMatrix.HXChebyNoCommun(YArray,
						     projectorKetTimesVector,
						     localVectorSize,
						     numberVectors,
						     XArray2);

			       //XArray2.compress(dealii::VectorOperation::add);
			       //XArray2.update_ghost_values();
			       

			       // update ghost values of XArray by scaling YArray ghost values with Rayleigh quotients
			       // XArray2= M * YArray' * Lamda = M^(1/2)*M^(-1/2)*H*M^(-1/2)*M^(1/2)*YArray'
			       // YArray'=M^(-1/2)*YArray

			       if (boundaryVectorSize>0)
				  scaleXArrayRayleighQuotientsCUDAKernel<<<(numberVectors+255)/256*boundaryVectorSize,256>>>(numberVectors,
													    boundaryVectorSize,
													    thrust::raw_pointer_cast(&operatorMatrix.getBoundaryIdToLocalIdMap()[0]),
													    thrust::raw_pointer_cast(&rayleighQuotientsD[0]), 
													    YArray.begin(),
													    operatorMatrix.getSqrtMassVec(),
													    XArray2.begin());


			      daxpbyCUDAKernel<<<min(((ghostVectorSize+localVectorSize)*numberVectors+255)/256,30000),256>>>((ghostVectorSize+localVectorSize)*numberVectors,
											     XArray2.begin(),
											     XArray.begin(),
											     1.0,
											     1.0);

                              MPI_Wait(&request, MPI_STATUS_IGNORE);
			      checkRayleighQuotients(numberVectors,
					             communAvoidanceTolerance,
						     computeAvoidanceTolerance,
                                                     &dotarrayH[0],
					             &rayleighQuotientsH[0],
						     &rayleighQuotientsDiffH[0],
                                                     dummy,
                                                     isComputeAvoidanceToleranceReached);

		              cudaMemcpy(thrust::raw_pointer_cast(&rayleighQuotientsD[0]),
					 &rayleighQuotientsH[0],
					 numberVectors*sizeof(double),
					 cudaMemcpyHostToDevice);
                      }
                      else
                      {
                               /*
			       operatorMatrix.HXChebyNoCommun(YArray,
						     projectorKetTimesVector,
						     localVectorSize,
						     numberVectors,
						     XArray);

			       //XArray2.compress(dealii::VectorOperation::add);
			       //XArray2.update_ghost_values();
			       

			       // update ghost values of XArray by scaling YArray ghost values with Rayleigh quotients
			       // XArray2= M * YArray' * Lamda = M^(1/2)*M^(-1/2)*H*M^(-1/2)*M^(1/2)*YArray'
			       // YArray'=M^(-1/2)*YArray

			       if (boundaryVectorSize>0)
				  addScaleXArrayRayleighQuotientsCUDAKernel<<<(numberVectors+255)/256*boundaryVectorSize,256>>>(numberVectors,
													    boundaryVectorSize,
													    thrust::raw_pointer_cast(&operatorMatrix.getBoundaryIdToLocalIdMap()[0]),
													    thrust::raw_pointer_cast(&rayleighQuotientsD[0]), 
													    YArray.begin(),
													    operatorMatrix.getSqrtMassVec(),
													    XArray.begin());
                               
                               
                               */
			       XArray2=0;
			       operatorMatrix.HXChebyNoCommun(YArray,
						     projectorKetTimesVector,
						     localVectorSize,
						     numberVectors,
						     XArray2);

			       //XArray2.compress(dealii::VectorOperation::add);
			       //XArray2.update_ghost_values();
			       

			       // update ghost values of XArray by scaling YArray ghost values with Rayleigh quotients
			       // XArray2= M * YArray' * Lamda = M^(1/2)*M^(-1/2)*H*M^(-1/2)*M^(1/2)*YArray'
			       // YArray'=M^(-1/2)*YArray

			       if (boundaryVectorSize>0)
				  scaleXArrayRayleighQuotientsCUDAKernel<<<(numberVectors+255)/256*boundaryVectorSize,256>>>(numberVectors,
													    boundaryVectorSize,
													    thrust::raw_pointer_cast(&operatorMatrix.getBoundaryIdToLocalIdMap()[0]),
													    thrust::raw_pointer_cast(&rayleighQuotientsD[0]), 
													    YArray.begin(),
													    operatorMatrix.getSqrtMassVec(),
													    XArray2.begin());


			      daxpbyCUDAKernel<<<min(((ghostVectorSize+localVectorSize)*numberVectors+255)/256,30000),256>>>((ghostVectorSize+localVectorSize)*numberVectors,
											     XArray2.begin(),
											     XArray.begin(),
											     1.0,
											     1.0);
                              
                      }
                    

		      //(YArray^T*M^(-1/2)*H*M^(-1/2)*YArray)/(YArray^T*YArray)
		      // =(YArray'_i^T*XArray2)/(YArray'^T_i* M*YArray')
                      if ((degree-3)%5==0)
                      {
			     computeRayleighQuotients(operatorMatrix.getCublasHandle(),
							       YArray.begin(),
							       XArray2.begin(),
							       operatorMatrix.getSqrtMassVec(),
							       thrust::raw_pointer_cast(&onesVecD[0]),
							       numberVectors,
							       localVectorSize,
							       operatorMatrix.getMPICommunicator(),
                                                               request,
							       thrust::raw_pointer_cast(&tempArrayD[0]),
							       thrust::raw_pointer_cast(&dotarrayD[0]),
							       &dotarrayH[0]);
                             count+=1;
                      }
                      

                      if (degree==(m-1))
                      {
                         XArray.zero_out_ghosts();
                         YArray.zero_out_ghosts();
                      }
              }
              else
              {

		      combinedCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(numberVectors,
										       localVectorSize,
										       YArray.begin(),
										       XArray.begin(),
										       coeff,
										       alpha2,
										       alpha1,
										       alpha1Old,
										       operatorMatrix.getInvSqrtMassVec(),
										       operatorMatrix.getSqrtMassVec());
		      //
		      //call HX
		      // H  * YArray'= H * M^(-1/2) * YArray
		      // YArray=M^(1/2)*YArray'
		      // XArray2= H *YArray'
		      // 
                      if ((degree-3)%5==0)
                      {
			      XArray2=0.0;
			      operatorMatrix.HXCheby(YArray,
						     tempFloatArray,
						     projectorKetTimesVector,
						     localVectorSize,
						     numberVectors,
						     XArray2,
						     mixedPrecOverall && dftParameters::useMixedPrecCheby);

			      daxpbyCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(totalVectorSize,
											     XArray2.begin(),
											     XArray.begin(),
											     1.0,
											     1.0);     
                      }       
                      else
                      {
			      operatorMatrix.HXCheby(YArray,
						     tempFloatArray,
						     projectorKetTimesVector,
						     localVectorSize,
						     numberVectors,
						     XArray,
						     mixedPrecOverall && dftParameters::useMixedPrecCheby);
                      }          
 
                      if ((degree-3)%5==0 && count>0)
                      {
                              MPI_Wait(&request, MPI_STATUS_IGNORE);
			      checkRayleighQuotients(numberVectors,
					             communAvoidanceTolerance,
						     computeAvoidanceTolerance,
                                                     &dotarrayH[0],
					             &rayleighQuotientsH[0],
						     &rayleighQuotientsDiffH[0],
                                                     isCommunAvoidanceToleranceReached,
                                                     isComputeAvoidanceToleranceReached);

			      if (this_mpi_process==0)
			      {
				//std::cout<<"Chebyshev polynomial iter: "<<degree-5<<std::endl;
				//for (unsigned int i=0; i<numberVectors;i++)
				//   std::cout<<" Difference of rayleigh quotient from previous iteration for vector: "<<i <<", "<<rayleighQuotientsDiffH[i]<<std::endl;
			      }
                      }

		      //(YArray^T*M^(-1/2)*H*M^(-1/2)*YArray)/(YArray^T*YArray)
		      // =(YArray'_i^T*XArray2)/(YArray'^T_i* M*YArray')
                      if ((degree-3)%5==0)
                      {
			     computeRayleighQuotients(operatorMatrix.getCublasHandle(),
							       YArray.begin(),
							       XArray2.begin(),
							       operatorMatrix.getSqrtMassVec(),
							       thrust::raw_pointer_cast(&onesVecD[0]),
							       numberVectors,
							       localVectorSize,
							       operatorMatrix.getMPICommunicator(),
                                                               request,
							       thrust::raw_pointer_cast(&tempArrayD[0]),
							       thrust::raw_pointer_cast(&dotarrayD[0]),
							       &dotarrayH[0]);
                             count+=1;
                      }

                      if (isCommunAvoidanceToleranceReached) 
                      {
			  cudaMemcpy(thrust::raw_pointer_cast(&rayleighQuotientsD[0]),
			             &rayleighQuotientsH[0],
				     numberVectors*sizeof(double),
				     cudaMemcpyHostToDevice);

                           isFirstCallToCommunAvoidance=true;    
                      }

              }       


	    }

	  XArray.swap(YArray);

      
	  sigma = sigma2;
	  alpha1Old=alpha1;

	}

      //copy back YArray to XArray
      cudaMemcpy(XArray.begin(),
		 YArray.begin(),
		 totalVectorSize*sizeof(double),
		 cudaMemcpyDeviceToDevice);

      MPI_Wait(&request, MPI_STATUS_IGNORE);
#endif
    }
 
    //
    // Compute and comunication of two blocks (1) and (2) are overlapped during chebyshev filtering.
    //
    void chebyshevFilter(operatorDFTCUDAClass & operatorMatrix,
			 distributedGPUVec<double> & XArray1,
                         distributedGPUVec<double> & YArray1,
			 distributedGPUVec<float> & tempFloatArray,
                         distributedGPUVec<double> & projectorKetTimesVector1,
                         distributedGPUVec<float> & projectorKetTimesVectorFloat,
			 distributedGPUVec<double> & XArray2,
                         distributedGPUVec<double> & YArray2,
                         distributedGPUVec<double> & projectorKetTimesVector2,
			 const unsigned int localVectorSize,
			 const unsigned int numberVectors,
			 const unsigned int m,
			 const double a,
			 const double b,
			 const double a0,
                         const bool mixedPrecOverall)
    {
#ifdef USE_COMPLEX
      AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
      double e, c, sigma, sigma1, sigma2, gamma, gpu_time;
      e = (b-a)/2.0; c = (b+a)/2.0;
      sigma = e/(a0-c); sigma1 = sigma; gamma = 2.0/sigma1;
      const unsigned int totalVectorSize = localVectorSize*numberVectors;
      int inc = 1;

      YArray1=0.0;
      YArray2=0.0;
 
      const unsigned int n_ghosts   = YArray1.get_partitioner()->n_ghost_indices()/numberVectors;
      const unsigned int totalSize  = localVectorSize + n_ghosts;

      const unsigned int localSizeNLP   = projectorKetTimesVector1.local_size()/numberVectors;
      const unsigned int n_ghosts_nlp   = projectorKetTimesVector1.get_partitioner()->n_ghost_indices()/numberVectors;
      const unsigned int totalSizeNLP   = localSizeNLP + n_ghosts_nlp;

      //
      //call HX
      //
      bool scaleFlag = false;
      double scalar = 1.0;

      operatorMatrix.HX(XArray1,
                        projectorKetTimesVector1,
			localVectorSize,
			numberVectors,
			scaleFlag,
			scalar,
			YArray1);

      operatorMatrix.HX(XArray2,
                        projectorKetTimesVector2,
			localVectorSize,
			numberVectors,
			scaleFlag,
			scalar,
			YArray2);

      double  alpha1 = sigma1/e, alpha2 = -c;
      double alpha1Old=alpha1;
  
      //
      //YArray = YArray + alpha2*XArray and YArray = alpha1*YArray
      //
      cublasDaxpy(operatorMatrix.getCublasHandle(),
		  totalVectorSize,
		  &alpha2,
		  XArray1.begin(),
		  inc,
		  YArray1.begin(),
		  inc);

      cublasDscal(operatorMatrix.getCublasHandle(),
		  totalVectorSize,
		  &alpha1,
		  YArray1.begin(),
		  inc);


      cublasDaxpy(operatorMatrix.getCublasHandle(),
		  totalVectorSize,
		  &alpha2,
		  XArray2.begin(),
		  inc,
		  YArray2.begin(),
		  inc);

      cublasDscal(operatorMatrix.getCublasHandle(),
		  totalVectorSize,
		  &alpha1,
		  YArray2.begin(),
		  inc);
	      
      bool overlap=false; 
      //
      //polynomial loop
      //
      for(unsigned int degree = 2; degree < m+1; ++degree)
	{
	  sigma2 = 1.0/(gamma - sigma);
	  alpha1 = 2.0*sigma2/e, alpha2 = -(sigma*sigma2);


	  double coeff = -c*alpha1;

	  
	  if (degree==2)
	    {
	      daxpbyCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(totalVectorSize,
									     YArray1.begin(),
									     XArray1.begin(),
          		                                                     coeff,
		                                                             alpha2);


	      //scale src vector with M^{-1/2}
	      //
	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       alpha1,
									       YArray1.begin(),
									       operatorMatrix.getInvSqrtMassVec());

	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       1.0,
									       XArray1.begin(),
									       operatorMatrix.getSqrtMassVec());

	      //
	      //call HX
	      //
	      operatorMatrix.HXCheby(YArray1,
				     tempFloatArray,
				     projectorKetTimesVector1,
				     localVectorSize,
				     numberVectors,
				     XArray1,
				     mixedPrecOverall && dftParameters::useMixedPrecCheby);

	      daxpbyCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(totalVectorSize,
									     YArray2.begin(),
									     XArray2.begin(),
									     coeff,
									     alpha2);


	      //scale src vector with M^{-1/2}
	      //
	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       alpha1,
									       YArray2.begin(),
									       operatorMatrix.getInvSqrtMassVec());

	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       1.0,
									       XArray2.begin(),
									       operatorMatrix.getSqrtMassVec());

	      //
	      //call HX
	      //
	      operatorMatrix.HXCheby(YArray2,
				     tempFloatArray,
				     projectorKetTimesVector2,
				     localVectorSize,
				     numberVectors,
				     XArray2,
				     mixedPrecOverall && dftParameters::useMixedPrecCheby);
	      overlap=false;
	    }
	  else if (degree==m)
	    {

	      //unscale src vector with M^{1/2}
	      //
	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       1.0/alpha1Old,
									       XArray1.begin(),
									       operatorMatrix.getSqrtMassVec());

	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       1.0,
									       YArray1.begin(),
									       operatorMatrix.getInvSqrtMassVec());

	      daxpbyCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(totalVectorSize,
									     YArray1.begin(),
									     XArray1.begin(),
									     coeff,
									     alpha2);
	      scaleFlag=true;
	      //
	      //call HX
	      //
	      operatorMatrix.HX(YArray1,
				projectorKetTimesVector1,
				localVectorSize,
				numberVectors,
				scaleFlag,
				alpha1,
				XArray1);


	      //unscale src vector with M^{1/2}
	      //
	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       1.0/alpha1Old,
									       XArray2.begin(),
									       operatorMatrix.getSqrtMassVec());

	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       1.0,
									       YArray2.begin(),
									       operatorMatrix.getInvSqrtMassVec());

	      daxpbyCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(totalVectorSize,
									     YArray2.begin(),
									     XArray2.begin(),
									     coeff,
									     alpha2);
	      //
	      //call HX
	      //
	      operatorMatrix.HX(YArray2,
				projectorKetTimesVector2,
				localVectorSize,
				numberVectors,
				scaleFlag,
				alpha1,
				XArray2);
	      overlap=false;
	    }
	  else
	    {
	      /////////////PSEUDO CODE for the implementation below for Overlapping compute and communication in HX/////////////////
	      //
	      // In the algorithm below the communication and computation of two blocks of wavefunctions: block 1 and
	      // block 2 are overlapped. CM-NB and CM-B denotes non-blocking and blocking communications respectively.
	      // CP denotes compute. The HX computation is divided into compute part 1 and compute part 2 which are separately
	      // overlapped. Note that the first and the last iterations of the algorithm are edge cases and are handled 
	      // a bit differently (Look for step skipped remarks below).
	      //
	      // 1) [CM-NB] Initiate compress of nonlocal projectorKetTimesVector of block 2 (skipped in first overlap iteration)  
	      // 2) [CP] Call combinedCUDAKernel of block 1
	      // 3) [CM-B] Finish compress of nonlocal projectorKetTimesVector of block 2. (skipped in first overlap iteration) 
	      // 4) [CM-NB] Call update_ghost_values on nonlocal projectorKetTimesVector of block 2. (skipped in first overlap iteration)
	      // 5) [CM-NB] Initiate update_ghost_values on wavefunctions of block 1.
	      // 6) [CP] Call HX compute part 2 on block 2. (skipped in first overlap iteration)
	      // 7) [CM-B] Finish update_ghost_values on wavefunctions of block 1.
	      // 8) [CM-NB] Initiate compress on wavefunctions of block 2.
	      // 9) [CP] Call HX compute part 1 on block 1.
	      // 10)[CM-B] Finish compress on wavefunctions of block 2.
	      // 11)[CM-NB] Initiate compress of nonlocal projectorKetTimesVector of block 1.
	      // 12)[CP] Call combinedCUDAKernel of block 2
	      // 13)[CM-B] Finish compress of nonlocal projectorKetTimesVector of block 1.
	      // 14)[CM-NB] Initiate update_ghost_values on wavefunctions of block 2.
	      // 15)[CP] Call HX compute part 2 on block 1. 
	      // 16)[CM-B] Finish update_ghost_values on wavefunctions of block 2.
	      // 17)[CM-NB] Initiate compress on wavefunctions of block 1.
	      // 18)[CP] Call HX compute part 1 on block 2.
	      // 19)[CM-B] Finish compress on wavefunctions of block 1. 
	      // 20) Perform chebyshev recursion related swap and scalar operations and go back to step 1)
	      //
	      // Extra steps for second to last chebyshev filter iteration or the last overlapped iteration:
	      // 21) Call compress and update_ghost_values on projectorKetTimesVector of block 2
	      // 22) Call HX compute part 2 on block 2.
	      // 23) Call compress on wavefunctions of block 2.
	      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	      //overlap flag is used to handle the edge case for the very first overlap is performed,
	      //where the previous chebyshev filtering iteration did not use the overlap algorithm
	      if (overlap)
		{
		  if (mixedPrecOverall && dftParameters::useMixedPrecChebyNonLocal)
		    {
                      if (totalSizeNLP>0)
			      convDoubleArrToFloatArr<<<(numberVectors+255)/256*totalSizeNLP,256>>>(numberVectors*totalSizeNLP,
												    projectorKetTimesVector2.begin(),
												    projectorKetTimesVectorFloat.begin());
		      projectorKetTimesVectorFloat.compress_start(dealii::VectorOperation::add);
		    }
		  else
		    projectorKetTimesVector2.compress_start(dealii::VectorOperation::add);
		}

	      combinedCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(numberVectors,
									       localVectorSize,
									       YArray1.begin(),
									       XArray1.begin(),
									       coeff,
									       alpha2,
									       alpha1,
									       alpha1Old,
									       operatorMatrix.getInvSqrtMassVec(),
									       operatorMatrix.getSqrtMassVec());


	      if (overlap)
		{

		  if (mixedPrecOverall && dftParameters::useMixedPrecChebyNonLocal)
		    {
		      projectorKetTimesVectorFloat.compress_finish(dealii::VectorOperation::add);

                      if (localSizeNLP>0)
			     copyFloatArrToDoubleArrLocallyOwned<<<(numberVectors+255)/256*localSizeNLP,256>>>(numberVectors,
														   localSizeNLP,
														   projectorKetTimesVectorFloat.begin(),
														   thrust::raw_pointer_cast(&operatorMatrix.getLocallyOwnedProcProjectorKetBoundaryNodesVectorDevice()[0]),
														   projectorKetTimesVector2.begin());

		      projectorKetTimesVector2.zero_out_ghosts();
		    }
		  else
                      projectorKetTimesVector2.compress_finish(dealii::VectorOperation::add);

		  if(mixedPrecOverall && dftParameters::useMixedPrecChebyNonLocal)
		    {
                        if (localSizeNLP>0)
		           convDoubleArrToFloatArr<<<(numberVectors+255)/256*localSizeNLP,256>>>(numberVectors*localSizeNLP,
												   projectorKetTimesVector2.begin(),
												   projectorKetTimesVectorFloat.begin());
			projectorKetTimesVectorFloat.update_ghost_values();

			if(n_ghosts_nlp>0)
			   convFloatArrToDoubleArr<<<(numberVectors+255)/256*n_ghosts,256>>>(numberVectors*n_ghosts_nlp,
								 			    projectorKetTimesVectorFloat.begin()+localSizeNLP*numberVectors,
											    projectorKetTimesVector2.begin()+localSizeNLP*numberVectors);


		      }
		  else
		     projectorKetTimesVector2.update_ghost_values();
		}

	      //unsigned int id2=nvtxRangeStartA("ghost1");
	      if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
		{
		  convDoubleArrToFloatArr<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors*localVectorSize,
											   YArray1.begin(),
											   tempFloatArray.begin());
		  tempFloatArray.update_ghost_values_start();
		}
	      else
		YArray1.update_ghost_values_start();

	      // call compute part 2 of block 2
	      if (overlap)
		operatorMatrix.HXCheby(YArray2,
				       tempFloatArray,
				       projectorKetTimesVector2,
				       localVectorSize,
				       numberVectors,
				       XArray2,
				       mixedPrecOverall && dftParameters::useMixedPrecCheby,
				       false,
				       true);

	      if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
		{
		  tempFloatArray.update_ghost_values_finish();
		  if(n_ghosts!=0)
		    convFloatArrToDoubleArr<<<(numberVectors+255)/256*n_ghosts,256>>>(numberVectors*n_ghosts,
										      tempFloatArray.begin()+localVectorSize*numberVectors,
										      YArray1.begin()+localVectorSize*numberVectors);
		}
	      else
		YArray1.update_ghost_values_finish();

	      if (overlap)
		YArray2.zero_out_ghosts();
	      //nvtxRangeEnd(id2);

	      projectorKetTimesVector1=0.0;
	      //unsigned int id1=nvtxRangeStartA("compress2");
	      if (overlap)
		{
		  if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
		    {
		      convDoubleArrToFloatArr<<<(numberVectors+255)/256*totalSize,256>>>(numberVectors*totalSize,
											 XArray2.begin(),
											 tempFloatArray.begin());
		      tempFloatArray.compress_start(dealii::VectorOperation::add);
		    }
		  else
		    XArray2.compress_start(dealii::VectorOperation::add);
		}

	      // call compute part 1 of block 1
	      operatorMatrix.HXCheby(YArray1,
				     tempFloatArray,
				     projectorKetTimesVector1,
				     localVectorSize,
				     numberVectors,
				     XArray1,
				     mixedPrecOverall && dftParameters::useMixedPrecCheby,
				     true,
				     false);

	      if (overlap)
		{
		  if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
		    {
		      tempFloatArray.compress_finish(dealii::VectorOperation::add);

		      copyFloatArrToDoubleArrLocallyOwned<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
													   localVectorSize,
													   tempFloatArray.begin(),
													   thrust::raw_pointer_cast(&operatorMatrix.getLocallyOwnedProcBoundaryNodesVectorDevice()[0]),
													   XArray2.begin());

		      XArray2.zero_out_ghosts();
		    }
		  else
		    XArray2.compress_finish(dealii::VectorOperation::add);
		  XArray2.swap(YArray2);
		}
	      //nvtxRangeEnd(id1);

	      if (mixedPrecOverall && dftParameters::useMixedPrecChebyNonLocal)
	      {
                    if (totalSizeNLP>0)
			    convDoubleArrToFloatArr<<<(numberVectors+255)/256*totalSizeNLP,256>>>(numberVectors*totalSizeNLP,
												    projectorKetTimesVector1.begin(),
												    projectorKetTimesVectorFloat.begin());
	            projectorKetTimesVectorFloat.compress_start(dealii::VectorOperation::add);
	      }
	      else
		    projectorKetTimesVector1.compress_start(dealii::VectorOperation::add);

	      combinedCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(numberVectors,
									       localVectorSize,
									       YArray2.begin(),
									       XArray2.begin(),
									       coeff,
									       alpha2,
									       alpha1,
									       alpha1Old,
									       operatorMatrix.getInvSqrtMassVec(),
									       operatorMatrix.getSqrtMassVec());

	       if (mixedPrecOverall && dftParameters::useMixedPrecChebyNonLocal)
	       {
		      projectorKetTimesVectorFloat.compress_finish(dealii::VectorOperation::add);

                      if (localSizeNLP>0)
			      copyFloatArrToDoubleArrLocallyOwned<<<(numberVectors+255)/256*localSizeNLP,256>>>(numberVectors,
														localSizeNLP,
														 projectorKetTimesVectorFloat.begin(),
														 thrust::raw_pointer_cast(&operatorMatrix.getLocallyOwnedProcProjectorKetBoundaryNodesVectorDevice()[0]),
														 projectorKetTimesVector1.begin());

		      projectorKetTimesVector1.zero_out_ghosts();
	       }
	       else
                      projectorKetTimesVector1.compress_finish(dealii::VectorOperation::add);

	       if(mixedPrecOverall && dftParameters::useMixedPrecChebyNonLocal)
	       {
                      if (localSizeNLP>0)
			  convDoubleArrToFloatArr<<<(numberVectors+255)/256*localSizeNLP,256>>>(numberVectors*localSizeNLP,
												    projectorKetTimesVector1.begin(),
												    projectorKetTimesVectorFloat.begin());
		      projectorKetTimesVectorFloat.update_ghost_values();

		      if(n_ghosts_nlp>0)
			  convFloatArrToDoubleArr<<<(numberVectors+255)/256*n_ghosts,256>>>(numberVectors*n_ghosts_nlp,
								 			    projectorKetTimesVectorFloat.begin()+localSizeNLP*numberVectors,
											    projectorKetTimesVector1.begin()+localSizeNLP*numberVectors);
	       }
	       else
		     projectorKetTimesVector1.update_ghost_values();

	      //unsigned int id3=nvtxRangeStartA("ghost2");
	      if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
		{
		  convDoubleArrToFloatArr<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors*localVectorSize,
											   YArray2.begin(),
											   tempFloatArray.begin());
		  tempFloatArray.update_ghost_values_start();
		}
	      else
		YArray2.update_ghost_values_start(); 

	      // call compute part 2 of block 1
	      operatorMatrix.HXCheby(YArray1,
				     tempFloatArray,
				     projectorKetTimesVector1,
				     localVectorSize,
				     numberVectors,
				     XArray1,
				     mixedPrecOverall && dftParameters::useMixedPrecCheby,
				     false,
				     true);

	      if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
		{
		  tempFloatArray.update_ghost_values_finish();
		  if(n_ghosts!=0)
		    convFloatArrToDoubleArr<<<(numberVectors+255)/256*n_ghosts,256>>>(numberVectors*n_ghosts,
										      tempFloatArray.begin()+localVectorSize*numberVectors,
										      YArray2.begin()+localVectorSize*numberVectors);
		}
	      else
		YArray2.update_ghost_values_finish();
	      YArray1.zero_out_ghosts();
	      //nvtxRangeEnd(id3);


	      projectorKetTimesVector2=0.0;

	      //unsigned int id4=nvtxRangeStartA("compress1");
	      if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
		{
		  convDoubleArrToFloatArr<<<(numberVectors+255)/256*totalSize,256>>>(numberVectors*totalSize,
										     XArray1.begin(),
										     tempFloatArray.begin());
		  tempFloatArray.compress_start(dealii::VectorOperation::add);
		}
	      else
		XArray1.compress_start(dealii::VectorOperation::add);

	      // call compute part 1 of block 2
	      operatorMatrix.HXCheby(YArray2,
				     tempFloatArray,
				     projectorKetTimesVector2,
				     localVectorSize,
				     numberVectors,
				     XArray2,
				     mixedPrecOverall && dftParameters::useMixedPrecCheby,
				     true,
				     false);

	      if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
		{
		  tempFloatArray.compress_finish(dealii::VectorOperation::add);

		  copyFloatArrToDoubleArrLocallyOwned<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
												       localVectorSize,
												       tempFloatArray.begin(),
												       thrust::raw_pointer_cast(&operatorMatrix.getLocallyOwnedProcBoundaryNodesVectorDevice()[0]),
												       XArray1.begin());

		  XArray1.zero_out_ghosts();
		}
	      else
		XArray1.compress_finish(dealii::VectorOperation::add);
	      //nvtxRangeEnd(id4);

	      //Handle edge case for the second to last Chebyshev filter iteration as there is no overlap
	      //algorithm for the next filter iteration.
	      if (degree==(m-1))
		{

		  projectorKetTimesVector2.compress(dealii::VectorOperation::add);
		  projectorKetTimesVector2.update_ghost_values();

		  operatorMatrix.HXCheby(YArray2,
					 tempFloatArray,
					 projectorKetTimesVector2,
					 localVectorSize,
					 numberVectors,
					 XArray2,
					 mixedPrecOverall && dftParameters::useMixedPrecCheby,
					 false,
					 true);
		  YArray2.zero_out_ghosts();
		  if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
		    {
		      convDoubleArrToFloatArr<<<(numberVectors+255)/256*totalSize,256>>>(numberVectors*totalSize,
											 XArray2.begin(),
											 tempFloatArray.begin());
		      tempFloatArray.compress(dealii::VectorOperation::add);

		      copyFloatArrToDoubleArrLocallyOwned<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
													   localVectorSize,
													   tempFloatArray.begin(),
													   thrust::raw_pointer_cast(&operatorMatrix.getLocallyOwnedProcBoundaryNodesVectorDevice()[0]),
													   XArray2.begin());

		      XArray2.zero_out_ghosts();
		    }
		  else
		    XArray2.compress(dealii::VectorOperation::add);
		  overlap=false;
		}
	      else
		overlap=true;
	    }

	  XArray1.swap(YArray1);
	  //Handle edge cases for the first and last iteration involving overlap of communication and computation 
	  if (!overlap)
	    {
	      XArray2.swap(YArray2);
	    }

      
	  sigma = sigma2;
	  alpha1Old=alpha1;

	}

      //copy back YArray to XArray
      cudaMemcpy(XArray1.begin(),
		 YArray1.begin(),
		 totalVectorSize*sizeof(double),
		 cudaMemcpyDeviceToDevice);

      cudaMemcpy(XArray2.begin(),
		 YArray2.begin(),
		 totalVectorSize*sizeof(double),
		 cudaMemcpyDeviceToDevice);
#endif
    }


    //
    // Compute and comunication of two blocks (1) and (2) are overlapped during chebyshev filtering.
    //
    void chebyshevFilterComputeAvoidance(operatorDFTCUDAClass & operatorMatrix,
			 distributedGPUVec<double> & XArray1,
                         distributedGPUVec<double> & XArrayCA,
                         distributedGPUVec<double> & YArray1,
			 distributedGPUVec<float> & tempFloatArray,
                         distributedGPUVec<double> & projectorKetTimesVector1,
                         distributedGPUVec<float> & projectorKetTimesVectorFloat,
			 distributedGPUVec<double> & XArray2,
                         distributedGPUVec<double> & YArray2,
                         distributedGPUVec<double> & projectorKetTimesVector2,
			 const unsigned int localVectorSize,
			 const unsigned int numberVectors,
			 const unsigned int m,
			 const double a,
			 const double b,
			 const double a0,
                         const bool isXlBOMDLinearizedSolve,
                         const bool communAvoidance,
                         const bool mixedPrecOverall,
                         const double computeAvoidanceTolerance)
    {
#ifdef USE_COMPLEX
      AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
      double e, c, sigma, sigma1, sigma2, gamma, gpu_time;
      e = (b-a)/2.0; c = (b+a)/2.0;
      sigma = e/(a0-c); sigma1 = sigma; gamma = 2.0/sigma1;
      const unsigned int totalVectorSize = localVectorSize*numberVectors;
      int inc = 1;

      YArray1=0.0;
      YArray2=0.0;
 
      const unsigned int n_ghosts   = YArray1.get_partitioner()->n_ghost_indices()/numberVectors;
      const unsigned int totalSize  = localVectorSize + n_ghosts;
      const unsigned int totalVectorSizeWithGhosts = totalSize*numberVectors;

      const unsigned int localSizeNLP   = projectorKetTimesVector1.local_size()/numberVectors;
      const unsigned int n_ghosts_nlp   = projectorKetTimesVector1.get_partitioner()->n_ghost_indices()/numberVectors;
      const unsigned int totalSizeNLP   = localSizeNLP + n_ghosts_nlp;

      const unsigned int this_mpi_process=dealii::Utilities::MPI::this_mpi_process(operatorMatrix.getMPICommunicator());
      XArrayCA=0;
      thrust::device_vector<double> onesVecD(localVectorSize,1.0);
      thrust::device_vector<double> tempArrayD(2*localVectorSize*numberVectors,0.0);
      thrust::device_vector<double> dotarrayD(2*numberVectors,0.0);
      std::vector<double> dotarrayH(2*numberVectors,0.0);
      thrust::device_vector<double> rayleighQuotientsD(numberVectors,0.0);
      std::vector<double> rayleighQuotientsH(numberVectors,0.0);
      std::vector<double> rayleighQuotientsDiffH(numberVectors,0.0);

      //
      //call HX
      //
      bool scaleFlag = false;
      double scalar = 1.0;

      operatorMatrix.HX(XArray1,
                        projectorKetTimesVector1,
			localVectorSize,
			numberVectors,
			scaleFlag,
			scalar,
			YArray1);

      operatorMatrix.HX(XArray2,
                        projectorKetTimesVector2,
			localVectorSize,
			numberVectors,
			scaleFlag,
			scalar,
			YArray2);

      double  alpha1 = sigma1/e, alpha2 = -c;
      double alpha1Old=alpha1;
  
      //
      //YArray = YArray + alpha2*XArray and YArray = alpha1*YArray
      //
      cublasDaxpy(operatorMatrix.getCublasHandle(),
		  totalVectorSize,
		  &alpha2,
		  XArray1.begin(),
		  inc,
		  YArray1.begin(),
		  inc);

      cublasDscal(operatorMatrix.getCublasHandle(),
		  totalVectorSize,
		  &alpha1,
		  YArray1.begin(),
		  inc);


      cublasDaxpy(operatorMatrix.getCublasHandle(),
		  totalVectorSize,
		  &alpha2,
		  XArray2.begin(),
		  inc,
		  YArray2.begin(),
		  inc);

      cublasDscal(operatorMatrix.getCublasHandle(),
		  totalVectorSize,
		  &alpha1,
		  YArray2.begin(),
		  inc);

      //const double computeAvoidanceTolerance=isXlBOMDLinearizedSolve?1e-9:1e-14;  
      bool isComputeAvoidanceToleranceReached=false;

      MPI_Request request;
      int count=0;
   
      bool overlap=false; 
      //
      //polynomial loop
      //
      for(unsigned int degree = 2; degree < m+1; ++degree)
	{
	  sigma2 = 1.0/(gamma - sigma);
	  alpha1 = 2.0*sigma2/e, alpha2 = -(sigma*sigma2);


	  double coeff = -c*alpha1;

	  
	  if (degree==2)
	    {
	      daxpbyCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(totalVectorSize,
									     YArray1.begin(),
									     XArray1.begin(),
          		                                                     coeff,
		                                                             alpha2);


	      //scale src vector with M^{-1/2}
	      //
	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       alpha1,
									       YArray1.begin(),
									       operatorMatrix.getInvSqrtMassVec());

	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       1.0,
									       XArray1.begin(),
									       operatorMatrix.getSqrtMassVec());

	      //
	      //call HX
	      //
	      operatorMatrix.HXCheby(YArray1,
				     tempFloatArray,
				     projectorKetTimesVector1,
				     localVectorSize,
				     numberVectors,
				     XArray1,
				     mixedPrecOverall && dftParameters::useMixedPrecCheby);

	      daxpbyCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(totalVectorSize,
									     YArray2.begin(),
									     XArray2.begin(),
									     coeff,
									     alpha2);


	      //scale src vector with M^{-1/2}
	      //
	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       alpha1,
									       YArray2.begin(),
									       operatorMatrix.getInvSqrtMassVec());

	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       1.0,
									       XArray2.begin(),
									       operatorMatrix.getSqrtMassVec());

	      //
	      //call HX
	      //
	      operatorMatrix.HXCheby(YArray2,
				     tempFloatArray,
				     projectorKetTimesVector2,
				     localVectorSize,
				     numberVectors,
				     XArray2,
				     mixedPrecOverall && dftParameters::useMixedPrecCheby);
	      overlap=false;
	    }
	  else if (degree==m)
	    {

	      //unscale src vector with M^{1/2}
	      //
	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       1.0/alpha1Old,
									       XArray1.begin(),
									       operatorMatrix.getSqrtMassVec());

	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       1.0,
									       YArray1.begin(),
									       operatorMatrix.getInvSqrtMassVec());

	      daxpbyCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(totalVectorSize,
									     YArray1.begin(),
									     XArray1.begin(),
									     coeff,
									     alpha2);
	      scaleFlag=true;
	      //
	      //call HX
	      //
	      operatorMatrix.HX(YArray1,
				projectorKetTimesVector1,
				localVectorSize,
				numberVectors,
				scaleFlag,
				alpha1,
				XArray1);


	      //unscale src vector with M^{1/2}
	      //
	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       1.0/alpha1Old,
									       XArray2.begin(),
									       operatorMatrix.getSqrtMassVec());

	      scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
									       localVectorSize,
									       1.0,
									       YArray2.begin(),
									       operatorMatrix.getInvSqrtMassVec());

	      daxpbyCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(totalVectorSize,
									     YArray2.begin(),
									     XArray2.begin(),
									     coeff,
									     alpha2);
	      //
	      //call HX
	      //
	      operatorMatrix.HX(YArray2,
				projectorKetTimesVector2,
				localVectorSize,
				numberVectors,
				scaleFlag,
				alpha1,
				XArray2);
	      overlap=false;
	    }
	  else
	    {
	      /////////////PSEUDO CODE for the implementation below for Overlapping compute and communication in HX/////////////////
	      //
	      // In the algorithm below the communication and computation of two blocks of wavefunctions: block 1 and
	      // block 2 are overlapped. CM-NB and CM-B denotes non-blocking and blocking communications respectively.
	      // CP denotes compute. The HX computation is divided into compute part 1 and compute part 2 which are separately
	      // overlapped. Note that the first and the last iterations of the algorithm are edge cases and are handled 
	      // a bit differently (Look for step skipped remarks below).
	      //
	      // 1) [CM-NB] Initiate compress of nonlocal projectorKetTimesVector of block 2 (skipped in first overlap iteration)  
	      // 2) [CP] Call combinedCUDAKernel of block 1
	      // 3) [CM-B] Finish compress of nonlocal projectorKetTimesVector of block 2. (skipped in first overlap iteration) 
	      // 4) [CM-NB] Call update_ghost_values on nonlocal projectorKetTimesVector of block 2. (skipped in first overlap iteration)
	      // 5) [CM-NB] Initiate update_ghost_values on wavefunctions of block 1.
	      // 6) [CP] Call HX compute part 2 on block 2. (skipped in first overlap iteration)
	      // 7) [CM-B] Finish update_ghost_values on wavefunctions of block 1.
	      // 8) [CM-NB] Initiate compress on wavefunctions of block 2.
	      // 9) [CP] Call HX compute part 1 on block 1.
	      // 10)[CM-B] Finish compress on wavefunctions of block 2.
	      // 11)[CM-NB] Initiate compress of nonlocal projectorKetTimesVector of block 1.
	      // 12)[CP] Call combinedCUDAKernel of block 2
	      // 13)[CM-B] Finish compress of nonlocal projectorKetTimesVector of block 1.
	      // 14)[CM-NB] Initiate update_ghost_values on wavefunctions of block 2.
	      // 15)[CP] Call HX compute part 2 on block 1. 
	      // 16)[CM-B] Finish update_ghost_values on wavefunctions of block 2.
	      // 17)[CM-NB] Initiate compress on wavefunctions of block 1.
	      // 18)[CP] Call HX compute part 1 on block 2.
	      // 19)[CM-B] Finish compress on wavefunctions of block 1. 
	      // 20) Perform chebyshev recursion related swap and scalar operations and go back to step 1)
	      //
	      // Extra steps for second to last chebyshev filter iteration or the last overlapped iteration:
	      // 21) Call compress and update_ghost_values on projectorKetTimesVector of block 2
	      // 22) Call HX compute part 2 on block 2.
	      // 23) Call compress on wavefunctions of block 2.
	      /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

              if (isComputeAvoidanceToleranceReached)
              {
		  //unscale src vector with M^{1/2}
		  //
		  scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
										       localVectorSize,
										       1.0/alpha1Old,
										       XArray1.begin(),
										       operatorMatrix.getSqrtMassVec());

		  scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
										       localVectorSize,
										       1.0,
										       YArray1.begin(),
										       operatorMatrix.getInvSqrtMassVec());

		  daxpbyCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(totalVectorSize,
										     YArray1.begin(),
										     XArray1.begin(),
										     coeff,
										     alpha2);
		  scaleFlag=true;
		  //
		  //call HX
		  //
		  operatorMatrix.HX(YArray1,
				    projectorKetTimesVector1,
				    localVectorSize,
				    numberVectors,
				    scaleFlag,
				    alpha1,
				    XArray1);


		  //unscale src vector with M^{1/2}
		  //
		  scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
										       localVectorSize,
										       1.0/alpha1Old,
										       XArray2.begin(),
										       operatorMatrix.getSqrtMassVec());

		  scaleCUDAKernel<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
										       localVectorSize,
										       1.0,
										       YArray2.begin(),
										       operatorMatrix.getInvSqrtMassVec());

		  daxpbyCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(totalVectorSize,
										     YArray2.begin(),
										     XArray2.begin(),
										     coeff,
										     alpha2);
		  //
		  //call HX
		  //
		  operatorMatrix.HX(YArray2,
		 		    projectorKetTimesVector2,
				    localVectorSize,
				    numberVectors,
				    scaleFlag,
				    alpha1,
				    XArray2);

		  XArray1.swap(YArray1);
		  XArray2.swap(YArray2);

                  if (this_mpi_process==0)
                      std::cout<<"  Skipping Chebyshev polynomial filtering (overlap compute-commun algo) after iter: "<<(degree-1)<<std::endl;
                  break;
              }

	      //overlap flag is used to handle the edge case for the very first overlap is performed,
	      //where the previous chebyshev filtering iteration did not use the overlap algorithm
	      if (overlap)
		{
		  if (mixedPrecOverall && dftParameters::useMixedPrecChebyNonLocal)
		    {
                      if (totalSizeNLP>0)
			      convDoubleArrToFloatArr<<<(numberVectors+255)/256*totalSizeNLP,256>>>(numberVectors*totalSizeNLP,
												    projectorKetTimesVector2.begin(),
												    projectorKetTimesVectorFloat.begin());
		      projectorKetTimesVectorFloat.compress_start(dealii::VectorOperation::add);
		    }
		  else
		    projectorKetTimesVector2.compress_start(dealii::VectorOperation::add);
		}

	      combinedCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(numberVectors,
									       localVectorSize,
									       YArray1.begin(),
									       XArray1.begin(),
									       coeff,
									       alpha2,
									       alpha1,
									       alpha1Old,
									       operatorMatrix.getInvSqrtMassVec(),
									       operatorMatrix.getSqrtMassVec());


	      if (overlap)
		{

		  if (mixedPrecOverall && dftParameters::useMixedPrecChebyNonLocal)
		    {
		      projectorKetTimesVectorFloat.compress_finish(dealii::VectorOperation::add);

                      if (localSizeNLP>0)
			     copyFloatArrToDoubleArrLocallyOwned<<<(numberVectors+255)/256*localSizeNLP,256>>>(numberVectors,
														   localSizeNLP,
														   projectorKetTimesVectorFloat.begin(),
														   thrust::raw_pointer_cast(&operatorMatrix.getLocallyOwnedProcProjectorKetBoundaryNodesVectorDevice()[0]),
														   projectorKetTimesVector2.begin());

		      projectorKetTimesVector2.zero_out_ghosts();
		    }
		  else
                      projectorKetTimesVector2.compress_finish(dealii::VectorOperation::add);

		  if(mixedPrecOverall && dftParameters::useMixedPrecChebyNonLocal)
		    {
                        if (localSizeNLP>0)
		           convDoubleArrToFloatArr<<<(numberVectors+255)/256*localSizeNLP,256>>>(numberVectors*localSizeNLP,
												   projectorKetTimesVector2.begin(),
												   projectorKetTimesVectorFloat.begin());
			projectorKetTimesVectorFloat.update_ghost_values();

			if(n_ghosts_nlp>0)
			   convFloatArrToDoubleArr<<<(numberVectors+255)/256*n_ghosts,256>>>(numberVectors*n_ghosts_nlp,
								 			    projectorKetTimesVectorFloat.begin()+localSizeNLP*numberVectors,
											    projectorKetTimesVector2.begin()+localSizeNLP*numberVectors);


		      }
		  else
		     projectorKetTimesVector2.update_ghost_values();
		}

	      //unsigned int id2=nvtxRangeStartA("ghost1");
	      if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
		{
		  convDoubleArrToFloatArr<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors*localVectorSize,
											   YArray1.begin(),
											   tempFloatArray.begin());
		  tempFloatArray.update_ghost_values_start();
		}
	      else
		YArray1.update_ghost_values_start();

	      // call compute part 2 of block 2
	      if (overlap)
		operatorMatrix.HXCheby(YArray2,
				       tempFloatArray,
				       projectorKetTimesVector2,
				       localVectorSize,
				       numberVectors,
				       XArray2,
				       mixedPrecOverall && dftParameters::useMixedPrecCheby,
				       false,
				       true);

	      if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
		{
		  tempFloatArray.update_ghost_values_finish();
		  if(n_ghosts!=0)
		    convFloatArrToDoubleArr<<<(numberVectors+255)/256*n_ghosts,256>>>(numberVectors*n_ghosts,
										      tempFloatArray.begin()+localVectorSize*numberVectors,
										      YArray1.begin()+localVectorSize*numberVectors);
		}
	      else
		YArray1.update_ghost_values_finish();

	      if (overlap)
		YArray2.zero_out_ghosts();
	      //nvtxRangeEnd(id2);

	      projectorKetTimesVector1=0.0;
	      //unsigned int id1=nvtxRangeStartA("compress2");
	      if (overlap)
		{
		  if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
		    {
		      convDoubleArrToFloatArr<<<(numberVectors+255)/256*totalSize,256>>>(numberVectors*totalSize,
											 XArray2.begin(),
											 tempFloatArray.begin());
		      tempFloatArray.compress_start(dealii::VectorOperation::add);
		    }
		  else
		    XArray2.compress_start(dealii::VectorOperation::add);
		}

	      // call compute part 1 of block 1
              if ((degree-3)%5==0)
              {
		      XArrayCA=0.0;
		      operatorMatrix.HXCheby(YArray1,
					     tempFloatArray,
					     projectorKetTimesVector1,
					     localVectorSize,
					     numberVectors,
					     XArrayCA,
					     mixedPrecOverall && dftParameters::useMixedPrecCheby,
                                             true,
                                             false);
              }  
              else
		      operatorMatrix.HXCheby(YArray1,
					     tempFloatArray,
					     projectorKetTimesVector1,
					     localVectorSize,
					     numberVectors,
					     XArray1,
					     mixedPrecOverall && dftParameters::useMixedPrecCheby,
					     true,
					     false);

	      if (overlap)
		{
		  if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
		    {
		      tempFloatArray.compress_finish(dealii::VectorOperation::add);

		      copyFloatArrToDoubleArrLocallyOwned<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
													   localVectorSize,
													   tempFloatArray.begin(),
													   thrust::raw_pointer_cast(&operatorMatrix.getLocallyOwnedProcBoundaryNodesVectorDevice()[0]),
													   XArray2.begin());

		      XArray2.zero_out_ghosts();
		    }
		  else
		    XArray2.compress_finish(dealii::VectorOperation::add);
		  XArray2.swap(YArray2);
		}
	      //nvtxRangeEnd(id1);

	      if (mixedPrecOverall && dftParameters::useMixedPrecChebyNonLocal)
	      {
                    if (totalSizeNLP>0)
			    convDoubleArrToFloatArr<<<(numberVectors+255)/256*totalSizeNLP,256>>>(numberVectors*totalSizeNLP,
												    projectorKetTimesVector1.begin(),
												    projectorKetTimesVectorFloat.begin());
	            projectorKetTimesVectorFloat.compress_start(dealii::VectorOperation::add);
	      }
	      else
		    projectorKetTimesVector1.compress_start(dealii::VectorOperation::add);

	      combinedCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(numberVectors,
									       localVectorSize,
									       YArray2.begin(),
									       XArray2.begin(),
									       coeff,
									       alpha2,
									       alpha1,
									       alpha1Old,
									       operatorMatrix.getInvSqrtMassVec(),
									       operatorMatrix.getSqrtMassVec());

	       if (mixedPrecOverall && dftParameters::useMixedPrecChebyNonLocal)
	       {
		      projectorKetTimesVectorFloat.compress_finish(dealii::VectorOperation::add);

                      if (localSizeNLP>0)
			      copyFloatArrToDoubleArrLocallyOwned<<<(numberVectors+255)/256*localSizeNLP,256>>>(numberVectors,
														localSizeNLP,
														 projectorKetTimesVectorFloat.begin(),
														 thrust::raw_pointer_cast(&operatorMatrix.getLocallyOwnedProcProjectorKetBoundaryNodesVectorDevice()[0]),
														 projectorKetTimesVector1.begin());

		      projectorKetTimesVector1.zero_out_ghosts();
	       }
	       else
                      projectorKetTimesVector1.compress_finish(dealii::VectorOperation::add);

	       if(mixedPrecOverall && dftParameters::useMixedPrecChebyNonLocal)
	       {
                      if (localSizeNLP>0)
			  convDoubleArrToFloatArr<<<(numberVectors+255)/256*localSizeNLP,256>>>(numberVectors*localSizeNLP,
												    projectorKetTimesVector1.begin(),
												    projectorKetTimesVectorFloat.begin());
		      projectorKetTimesVectorFloat.update_ghost_values();

		      if(n_ghosts_nlp>0)
			  convFloatArrToDoubleArr<<<(numberVectors+255)/256*n_ghosts,256>>>(numberVectors*n_ghosts_nlp,
								 			    projectorKetTimesVectorFloat.begin()+localSizeNLP*numberVectors,
											    projectorKetTimesVector1.begin()+localSizeNLP*numberVectors);
	       }
	       else
		     projectorKetTimesVector1.update_ghost_values();

	      //unsigned int id3=nvtxRangeStartA("ghost2");
	      if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
		{
		  convDoubleArrToFloatArr<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors*localVectorSize,
											   YArray2.begin(),
											   tempFloatArray.begin());
		  tempFloatArray.update_ghost_values_start();
		}
	      else
		YArray2.update_ghost_values_start(); 

	      // call compute part 2 of block 1
              if ((degree-3)%5==0)
              {
		      operatorMatrix.HXCheby(YArray1,
					     tempFloatArray,
					     projectorKetTimesVector1,
					     localVectorSize,
					     numberVectors,
					     XArrayCA,
					     mixedPrecOverall && dftParameters::useMixedPrecCheby,
					     false,
					     true);
              }  
              else
		      operatorMatrix.HXCheby(YArray1,
					     tempFloatArray,
					     projectorKetTimesVector1,
					     localVectorSize,
					     numberVectors,
					     XArray1,
					     mixedPrecOverall && dftParameters::useMixedPrecCheby,
					     false,
					     true);

	      if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
		{
		  tempFloatArray.update_ghost_values_finish();
		  if(n_ghosts!=0)
		    convFloatArrToDoubleArr<<<(numberVectors+255)/256*n_ghosts,256>>>(numberVectors*n_ghosts,
										      tempFloatArray.begin()+localVectorSize*numberVectors,
										      YArray2.begin()+localVectorSize*numberVectors);
		}
	      else
		YArray2.update_ghost_values_finish();
	      YArray1.zero_out_ghosts();
	      //nvtxRangeEnd(id3);


	      projectorKetTimesVector2=0.0;

	      //unsigned int id4=nvtxRangeStartA("compress1");
	      if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
		{
		  convDoubleArrToFloatArr<<<(numberVectors+255)/256*totalSize,256>>>(numberVectors*totalSize,
										     (degree-3)%5==0?XArrayCA.begin():XArray1.begin(),
										     tempFloatArray.begin());
		  tempFloatArray.compress_start(dealii::VectorOperation::add);
		}
	      else
                {
                  if ((degree-3)%5==0)
                     XArrayCA.compress_start(dealii::VectorOperation::add); 
                  else
		     XArray1.compress_start(dealii::VectorOperation::add);
                }


	      // call compute part 1 of block 2
	      operatorMatrix.HXCheby(YArray2,
				     tempFloatArray,
				     projectorKetTimesVector2,
				     localVectorSize,
				     numberVectors,
				     XArray2,
				     mixedPrecOverall && dftParameters::useMixedPrecCheby,
				     true,
				     false);

	      if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
		{
		  tempFloatArray.compress_finish(dealii::VectorOperation::add);

		  copyFloatArrToDoubleArrLocallyOwned<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
												       localVectorSize,
												       tempFloatArray.begin(),
												       thrust::raw_pointer_cast(&operatorMatrix.getLocallyOwnedProcBoundaryNodesVectorDevice()[0]),
												       (degree-3)%5==0
?XArrayCA.begin():XArray1.begin());

                  if ((degree-3)%5==0)
                     XArrayCA.zero_out_ghosts();

		  XArray1.zero_out_ghosts();
		}
	      else
                {
                  if ((degree-3)%5==0)
 		     XArrayCA.compress_finish(dealii::VectorOperation::add);
                  else
 		     XArray1.compress_finish(dealii::VectorOperation::add);
                }
	      //nvtxRangeEnd(id4);


              if ((degree-3)%5==0)
		      daxpbyCUDAKernel<<<min((totalVectorSize+255)/256,30000),256>>>(totalVectorSize,
										     XArrayCA.begin(),
										     XArray1.begin(),
										     1.0,
										     1.0);  

	      if ((degree-3)%5==0 && count>0)
	      {
		      MPI_Wait(&request, MPI_STATUS_IGNORE);
		      checkRayleighQuotients(numberVectors,
					     computeAvoidanceTolerance,
					     &dotarrayH[0],
					     &rayleighQuotientsH[0],
					     &rayleighQuotientsDiffH[0],
					     isComputeAvoidanceToleranceReached);

                      /*
		      if (this_mpi_process==0)
		      {
			 std::cout<<"Chebyshev polynomial iter: "<<degree-5<<std::endl;
			 for (unsigned int i=0; i<numberVectors;i++)
			   std::cout<<" Difference of rayleigh quotient from previous iteration for vector: "<<i <<", "<<rayleighQuotientsDiffH[i]<<std::endl;
		      } 
                      */
	      }


	      //(YArray^T*M^(-1/2)*H*M^(-1/2)*YArray)/(YArray^T*YArray)
	      // =(YArray'_i^T*XArray2)/(YArray'^T_i* M*YArray')
	      if ((degree-3)%5==0)
	      {
		     computeRayleighQuotients(operatorMatrix.getCublasHandle(),
					       YArray1.begin(),
					       XArrayCA.begin(),
					       operatorMatrix.getSqrtMassVec(),
					       thrust::raw_pointer_cast(&onesVecD[0]),
					       numberVectors,
					       localVectorSize,
					       operatorMatrix.getMPICommunicator(),
					       request,
					       thrust::raw_pointer_cast(&tempArrayD[0]),
					       thrust::raw_pointer_cast(&dotarrayD[0]),
					       &dotarrayH[0]);
		     count+=1;
	      }

	      //Handle edge case for the second to last Chebyshev filter iteration as there is no overlap
	      //algorithm for the next filter iteration.
	      if (degree==(m-1) || isComputeAvoidanceToleranceReached)
		{

		  projectorKetTimesVector2.compress(dealii::VectorOperation::add);
		  projectorKetTimesVector2.update_ghost_values();

		  operatorMatrix.HXCheby(YArray2,
					 tempFloatArray,
					 projectorKetTimesVector2,
					 localVectorSize,
					 numberVectors,
					 XArray2,
					 mixedPrecOverall && dftParameters::useMixedPrecCheby,
					 false,
					 true);
		  YArray2.zero_out_ghosts();
		  if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
		    {
		      convDoubleArrToFloatArr<<<(numberVectors+255)/256*totalSize,256>>>(numberVectors*totalSize,
											 XArray2.begin(),
											 tempFloatArray.begin());
		      tempFloatArray.compress(dealii::VectorOperation::add);

		      copyFloatArrToDoubleArrLocallyOwned<<<(numberVectors+255)/256*localVectorSize,256>>>(numberVectors,
													   localVectorSize,
													   tempFloatArray.begin(),
													   thrust::raw_pointer_cast(&operatorMatrix.getLocallyOwnedProcBoundaryNodesVectorDevice()[0]),
													   XArray2.begin());

		      XArray2.zero_out_ghosts();
		    }
		  else
		    XArray2.compress(dealii::VectorOperation::add);
		  overlap=false;
		}
	      else
		overlap=true;
	    }

	  XArray1.swap(YArray1);
	  //Handle edge cases for the first and last iteration involving overlap of communication and computation 
	  if (!overlap)
	    {
	      XArray2.swap(YArray2);
	    }

      
	  sigma = sigma2;
	  alpha1Old=alpha1;

	}

      //copy back YArray to XArray
      cudaMemcpy(XArray1.begin(),
		 YArray1.begin(),
		 totalVectorSize*sizeof(double),
		 cudaMemcpyDeviceToDevice);

      cudaMemcpy(XArray2.begin(),
		 YArray2.begin(),
		 totalVectorSize*sizeof(double),
		 cudaMemcpyDeviceToDevice);
#endif
    }

    void subspaceRotationSpectrumSplitScalapack(const double* X,
						double * XFrac,
						const unsigned int M,
						const unsigned int N,
						const unsigned int Nfr,
						cublasHandle_t &handle,
						const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
						const MPI_Comm &mpiComm,
						const dealii::ScaLAPACKMatrix<double> & rotationMatPar,
						const bool rotationMatTranspose)
    {
#ifdef USE_COMPLEX
      AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
      const unsigned int maxNumLocalDofs=dealii::Utilities::MPI::max(M,mpiComm);

      std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperationsCUDA::internal::createGlobalToLocalIdMapsScaLAPACKMat(processGrid,
										   rotationMatPar,
										   globalToLocalRowIdMap,
										   globalToLocalColumnIdMap);

      const unsigned int vectorsBlockSize=std::min(dftParameters::wfcBlockSize,
						   Nfr);
      const unsigned int dofsBlockSize=std::min(maxNumLocalDofs,
						dftParameters::subspaceRotDofsBlockSize);

      double * rotationMatBlockHost;

      if (dftParameters::allowFullCPUMemSubspaceRot)
	{
	  cudaMallocHost((void **)&rotationMatBlockHost,N*Nfr*sizeof(double));
	  std::memset(rotationMatBlockHost,0,N*Nfr*sizeof(double));
	}
      else
	{
	  cudaMallocHost((void **)&rotationMatBlockHost,vectorsBlockSize*N*sizeof(double));
	  std::memset(rotationMatBlockHost,0,vectorsBlockSize*N*sizeof(double));
	}
   
 
    
      thrust::device_vector<double> rotationMatBlock(vectorsBlockSize*N,0.0);
      thrust::device_vector<double> rotatedVectorsMatBlock(Nfr*dofsBlockSize,0.0);

      for (unsigned int idof = 0; idof < maxNumLocalDofs; idof += dofsBlockSize)
	{
	  // Correct block dimensions if block "goes off edge of" the matrix
	  unsigned int BDof=0;
	  if (M>=idof)
	    BDof = std::min(dofsBlockSize, M-idof);

	  //thrust::fill(rotatedVectorsMatBlock.begin(),rotatedVectorsMatBlock.end(),0.);
	  for (unsigned int jvec = 0; jvec < Nfr; jvec += vectorsBlockSize)
	    {
	      // Correct block dimensions if block "goes off edge of" the matrix
	      const unsigned int BVec = std::min(vectorsBlockSize, Nfr-jvec);

	      const double scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;

	      if (dftParameters::allowFullCPUMemSubspaceRot)
		{
		  if (idof==0)
		    {
		      //Extract QBVec from parallel ScaLAPACK matrix Q
		      if (rotationMatTranspose)
			{
			  if (processGrid->is_process_active())
			    for (unsigned int i = 0; i <N; ++i)
			      if (globalToLocalRowIdMap.find(i)
				  !=globalToLocalRowIdMap.end())
				{
				  const unsigned int localRowId=globalToLocalRowIdMap[i];
				  for (unsigned int j = 0; j <BVec; ++j)
				    {
				      std::map<unsigned int, unsigned int>::iterator it=
					globalToLocalColumnIdMap.find(j+jvec);
				      if(it!=globalToLocalColumnIdMap.end())
					rotationMatBlockHost[jvec*N+i*BVec+j]=
					  rotationMatPar.local_el(localRowId,
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
				  for (unsigned int j = 0; j <BVec; ++j)
				    {
				      std::map<unsigned int, unsigned int>::iterator it=
					globalToLocalRowIdMap.find(j+jvec);
				      if (it!=globalToLocalRowIdMap.end())
					rotationMatBlockHost[jvec*N+i*BVec+j]=
					  rotationMatPar.local_el(it->second,
								  localColumnId);
				    }
				}
			}
		    }
		}
	      else
		{
		  std::memset(rotationMatBlockHost,0,BVec*N*sizeof(double));

		  //Extract QBVec from parallel ScaLAPACK matrix Q
		  if (rotationMatTranspose)
		    {
		      if (processGrid->is_process_active())
			for (unsigned int i = 0; i <N; ++i)
			  if (globalToLocalRowIdMap.find(i)
			      !=globalToLocalRowIdMap.end())
			    {
			      const unsigned int localRowId=globalToLocalRowIdMap[i];
			      for (unsigned int j = 0; j <BVec; ++j)
				{
				  std::map<unsigned int, unsigned int>::iterator it=
				    globalToLocalColumnIdMap.find(j+jvec);
				  if(it!=globalToLocalColumnIdMap.end())
				    rotationMatBlockHost[i*BVec+j]=
				      rotationMatPar.local_el(localRowId,
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
			      for (unsigned int j = 0; j <BVec; ++j)
				{
				  std::map<unsigned int, unsigned int>::iterator it=
				    globalToLocalRowIdMap.find(j+jvec);
				  if (it!=globalToLocalRowIdMap.end())
				    rotationMatBlockHost[i*BVec+j]=
				      rotationMatPar.local_el(it->second,
							      localColumnId);
				}
			    }
		    }
		}


	      if (dftParameters::allowFullCPUMemSubspaceRot)
		{
		  if (idof==0)
		    MPI_Allreduce(MPI_IN_PLACE,
				  rotationMatBlockHost+jvec*N,
				  BVec*N,
				  MPI_DOUBLE,
				  MPI_SUM,
				  mpiComm);

		  cudaMemcpy(thrust::raw_pointer_cast(&rotationMatBlock[0]),
			     rotationMatBlockHost+jvec*N,
			     BVec*N*sizeof(double),
			     cudaMemcpyHostToDevice);
		}
	      else
		{
		  MPI_Allreduce(MPI_IN_PLACE,
				rotationMatBlockHost,
				BVec*N,
				MPI_DOUBLE,
				MPI_SUM,
				mpiComm);

		  cudaMemcpy(thrust::raw_pointer_cast(&rotationMatBlock[0]),
			     rotationMatBlockHost,
			     BVec*N*sizeof(double),
			     cudaMemcpyHostToDevice);
		}

	      if (BDof!=0)
		{
                       
		  cublasDgemm(handle,
			      CUBLAS_OP_N,
			      CUBLAS_OP_N,
			      BVec,
			      BDof,
			      N,
			      &scalarCoeffAlpha,
			      thrust::raw_pointer_cast(&rotationMatBlock[0]),
			      BVec,
			      X+idof*N,
			      N,
			      &scalarCoeffBeta,
			      thrust::raw_pointer_cast(&rotatedVectorsMatBlock[0])+jvec,
			      Nfr);
                       
		}

	    }//block loop over vectors

              
	  if (BDof!=0)
	    {
	      cudaMemcpy(XFrac+idof*Nfr,
			 thrust::raw_pointer_cast(&rotatedVectorsMatBlock[0]),
			 Nfr*BDof*sizeof(double),
			 cudaMemcpyDeviceToDevice);
	    }
              
	}//block loop over dofs

      cudaFreeHost(rotationMatBlockHost);
#endif
    }



    void subspaceRotationScalapack(double* X,
				   const unsigned int M,
				   const unsigned int N,
				   cublasHandle_t &handle,
				   const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				   const MPI_Comm &mpiComm,
				   const MPI_Comm &interBandGroupComm,
				   const dealii::ScaLAPACKMatrix<double> & rotationMatPar,
				   const bool rotationMatTranspose,
				   const bool isRotationMatLowerTria)
    {
#ifdef USE_COMPLEX
      AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
      const unsigned int maxNumLocalDofs=dealii::Utilities::MPI::max(M,mpiComm);

      std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperationsCUDA::internal::createGlobalToLocalIdMapsScaLAPACKMat(processGrid,
										   rotationMatPar,
										   globalToLocalRowIdMap,
										   globalToLocalColumnIdMap);

      //band group parallelization data structures
      const unsigned int numberBandGroups=
	dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(interBandGroupComm,
						 N,
						 bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize=std::min(dftParameters::wfcBlockSize,
						   N);
      const unsigned int dofsBlockSize=std::min(maxNumLocalDofs,
						dftParameters::subspaceRotDofsBlockSize);

      double * rotationMatBlockHost;

      if (dftParameters::allowFullCPUMemSubspaceRot)
	{
	  cudaMallocHost((void **)&rotationMatBlockHost,N*N*sizeof(double));
	  std::memset(rotationMatBlockHost,0,N*N*sizeof(double));
	}
      else
	{
	  cudaMallocHost((void **)&rotationMatBlockHost,vectorsBlockSize*N*sizeof(double));
	  std::memset(rotationMatBlockHost,0,vectorsBlockSize*N*sizeof(double));
	}
   
 
    
      thrust::device_vector<double> rotationMatBlock(vectorsBlockSize*N,0.0);
      thrust::device_vector<double> rotatedVectorsMatBlock(N*dofsBlockSize,0.0);

      for (unsigned int idof = 0; idof < maxNumLocalDofs; idof += dofsBlockSize)
	{
	  // Correct block dimensions if block "goes off edge of" the matrix
	  unsigned int BDof=0;
	  if (M>=idof)
	    BDof = std::min(dofsBlockSize, M-idof);

	  //thrust::fill(rotatedVectorsMatBlock.begin(),rotatedVectorsMatBlock.end(),0.);
	  for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
	    {
	      // Correct block dimensions if block "goes off edge of" the matrix
	      const unsigned int BVec = std::min(vectorsBlockSize, N-jvec);

	      const unsigned int D=isRotationMatLowerTria?
		(jvec+BVec)
		:N;

	      if ((jvec+BVec)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
		  (jvec+BVec)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
		{

		  const double scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;

		  if (dftParameters::allowFullCPUMemSubspaceRot)
		    {
		      if (idof==0)
			{
			  //Extract QBVec from parallel ScaLAPACK matrix Q
			  if (rotationMatTranspose)
			    {
			      if (processGrid->is_process_active())
				for (unsigned int i = 0; i <D; ++i)
				  if (globalToLocalRowIdMap.find(i)
				      !=globalToLocalRowIdMap.end())
				    {
				      const unsigned int localRowId=globalToLocalRowIdMap[i];
				      for (unsigned int j = 0; j <BVec; ++j)
					{
					  std::map<unsigned int, unsigned int>::iterator it=
					    globalToLocalColumnIdMap.find(j+jvec);
					  if(it!=globalToLocalColumnIdMap.end())
					    rotationMatBlockHost[jvec*N+i*BVec+j]=
					      rotationMatPar.local_el(localRowId,
								      it->second);
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
				      for (unsigned int j = 0; j <BVec; ++j)
					{
					  std::map<unsigned int, unsigned int>::iterator it=
					    globalToLocalRowIdMap.find(j+jvec);
					  if (it!=globalToLocalRowIdMap.end())
					    rotationMatBlockHost[jvec*N+i*BVec+j]=
					      rotationMatPar.local_el(it->second,
								      localColumnId);
					}
				    }
			    }
			}
		    }
		  else
		    {
		      std::memset(rotationMatBlockHost,0,BVec*N*sizeof(double));

		      //Extract QBVec from parallel ScaLAPACK matrix Q
		      if (rotationMatTranspose)
			{
			  if (processGrid->is_process_active())
			    for (unsigned int i = 0; i <D; ++i)
			      if (globalToLocalRowIdMap.find(i)
				  !=globalToLocalRowIdMap.end())
				{
				  const unsigned int localRowId=globalToLocalRowIdMap[i];
				  for (unsigned int j = 0; j <BVec; ++j)
				    {
				      std::map<unsigned int, unsigned int>::iterator it=
					globalToLocalColumnIdMap.find(j+jvec);
				      if(it!=globalToLocalColumnIdMap.end())
					rotationMatBlockHost[i*BVec+j]=
					  rotationMatPar.local_el(localRowId,
								  it->second);
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
				  for (unsigned int j = 0; j <BVec; ++j)
				    {
				      std::map<unsigned int, unsigned int>::iterator it=
					globalToLocalRowIdMap.find(j+jvec);
				      if (it!=globalToLocalRowIdMap.end())
					rotationMatBlockHost[i*BVec+j]=
					  rotationMatPar.local_el(it->second,
								  localColumnId);
				    }
				}
			}
		    }

		  if (dftParameters::allowFullCPUMemSubspaceRot)
		    {
		      if (idof==0)
			MPI_Allreduce(MPI_IN_PLACE,
				      rotationMatBlockHost+jvec*N,
				      BVec*D,
				      MPI_DOUBLE,
				      MPI_SUM,
				      mpiComm);

		      cudaMemcpy(thrust::raw_pointer_cast(&rotationMatBlock[0]),
				 rotationMatBlockHost+jvec*N,
				 BVec*D*sizeof(double),
				 cudaMemcpyHostToDevice);
		    }
		  else
		    {
		      MPI_Allreduce(MPI_IN_PLACE,
				    rotationMatBlockHost,
				    BVec*D,
				    MPI_DOUBLE,
				    MPI_SUM,
				    mpiComm);

		      cudaMemcpy(thrust::raw_pointer_cast(&rotationMatBlock[0]),
				 rotationMatBlockHost,
				 BVec*D*sizeof(double),
				 cudaMemcpyHostToDevice);
		    }

		  if (BDof!=0)
		    {
			       
		      cublasDgemm(handle,
				  CUBLAS_OP_N,
				  CUBLAS_OP_N,
				  BVec,
				  BDof,
				  D,
				  &scalarCoeffAlpha,
				  thrust::raw_pointer_cast(&rotationMatBlock[0]),
				  BVec,
				  X+idof*N,
				  N,
				  &scalarCoeffBeta,
				  thrust::raw_pointer_cast(&rotatedVectorsMatBlock[0])+jvec,
				  N);
			       
		    }
		}//band parallelization
	    }//block loop over vectors

              
	  if (BDof!=0)
	    {
	      cudaMemcpy(X+idof*N,
			 thrust::raw_pointer_cast(&rotatedVectorsMatBlock[0]),
			 N*BDof*sizeof(double),
			 cudaMemcpyDeviceToDevice);
	    }
              
	}//block loop over dofs

      cudaFreeHost(rotationMatBlockHost);
#endif
    }

    void subspaceRotationPGSMixedPrecScalapack(double* X,
					       const unsigned int M,
					       const unsigned int N,
					       cublasHandle_t &handle,
					       const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
					       const MPI_Comm &mpiComm,
					       const MPI_Comm &interBandGroupComm,
					       const dealii::ScaLAPACKMatrix<double> & rotationMatPar,
					       const bool rotationMatTranspose)
    {
#ifdef USE_COMPLEX
      AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
      const unsigned int maxNumLocalDofs=dealii::Utilities::MPI::max(M,mpiComm);

      std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperationsCUDA::internal::createGlobalToLocalIdMapsScaLAPACKMat(processGrid,
										   rotationMatPar,
										   globalToLocalRowIdMap,
										   globalToLocalColumnIdMap);

      //band group parallelization data structures
      const unsigned int numberBandGroups=
	dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(interBandGroupComm,
						 N,
						 bandGroupLowHighPlusOneIndices);

      const unsigned int MPadded=std::ceil(M*1.0/8.0)*8.0+0.5;
      thrust::device_vector<float> XSP(MPadded*N,0.0);

      convDoubleArrToFloatArr<<<(N+255)/256*M,256>>>(N*M,
						     X,
						     thrust::raw_pointer_cast(&XSP[0]));


      const unsigned int vectorsBlockSize=std::min(dftParameters::wfcBlockSize,
						   N);
      const unsigned int dofsBlockSize=std::min(maxNumLocalDofs,
						dftParameters::subspaceRotDofsBlockSize);

      float * rotationMatBlockHostSP;
      cudaMallocHost((void **)&rotationMatBlockHostSP,vectorsBlockSize*N*sizeof(float));
      std::memset(rotationMatBlockHostSP,0,vectorsBlockSize*N*sizeof(float));

      std::vector<float> rotationMatDiagBandHostSP;
    
      double * diagValuesHost;
      cudaMallocHost((void **)&diagValuesHost,N*sizeof(double));
      std::memset(diagValuesHost,0,N*sizeof(double));
    

      thrust::device_vector<float> rotationMatBlockSP(vectorsBlockSize*N,0.0);
      thrust::device_vector<double> diagValues(N,0.0);
      thrust::device_vector<float> rotatedVectorsMatBlockSP(vectorsBlockSize*dofsBlockSize,0.0);
 
      const float scalarCoeffAlphaSP = 1.0,scalarCoeffBetaSP = 0.0;


      //Extract DiagQ from parallel ScaLAPACK matrix Q
      if (rotationMatTranspose)
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
		      diagValuesHost[i]=rotationMatPar.local_el(localRowId,
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
			=rotationMatPar.local_el(it->second,
						 localColumnId);
		    }
		}
	}

      MPI_Allreduce(MPI_IN_PLACE,
		    diagValuesHost,
		    N,
		    MPI_DOUBLE,
		    MPI_SUM,
		    mpiComm);

      cudaMemcpy(thrust::raw_pointer_cast(&diagValues[0]),
		 diagValuesHost,
		 N*sizeof(double),
		 cudaMemcpyHostToDevice);

      computeDiagQTimesXKernel<<<(M*N+255)/256,256>>>(thrust::raw_pointer_cast(&diagValues[0]),
						      X,
						      N,
						      M);

      for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
	{
	  // Correct block dimensions if block "goes off edge of" the matrix
	  const unsigned int BVec = std::min(vectorsBlockSize, N-jvec);

	  const unsigned int D=jvec+BVec;

	  std::memset(rotationMatBlockHostSP,0,BVec*N*sizeof(float));

	  if ((jvec+BVec)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
	      (jvec+BVec)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
	    {
	      //Extract QBVec from parallel ScaLAPACK matrix Q
	      if (rotationMatTranspose)
		{
		  if (processGrid->is_process_active())
		    for (unsigned int i = 0; i <D; ++i)
		      if (globalToLocalRowIdMap.find(i)
			  !=globalToLocalRowIdMap.end())
			{
			  const unsigned int localRowId=globalToLocalRowIdMap[i];
			  for (unsigned int j = 0; j <BVec; ++j)
			    {
			      std::map<unsigned int, unsigned int>::iterator it=
				globalToLocalColumnIdMap.find(j+jvec);
			      if(it!=globalToLocalColumnIdMap.end())
				{
				  rotationMatBlockHostSP[i*BVec+j]=
				    rotationMatPar.local_el(localRowId,
							    it->second);

				}
			    }

			  if (i>=jvec && i<(jvec+BVec))
			    {
			      std::map<unsigned int, unsigned int>::iterator it=
				globalToLocalColumnIdMap.find(i);
			      if (it!=globalToLocalColumnIdMap.end())
				{
				  rotationMatBlockHostSP[i*BVec+i-jvec]=0.0;
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
			  for (unsigned int j = 0; j <BVec; ++j)
			    {
			      std::map<unsigned int, unsigned int>::iterator it=
				globalToLocalRowIdMap.find(j+jvec);
			      if (it!=globalToLocalRowIdMap.end())
				{
				  rotationMatBlockHostSP[i*BVec+j]=
				    rotationMatPar.local_el(it->second,
							    localColumnId);
				}
			    }

			  if (i>=jvec && i<(jvec+BVec))
			    {
			      std::map<unsigned int, unsigned int>::iterator it=
				globalToLocalRowIdMap.find(i);
			      if (globalToLocalRowIdMap.find(i)!=globalToLocalRowIdMap.end())
				{
				  rotationMatBlockHostSP[i*BVec+i-jvec]=0.0;
				}
			    }
			}
		}


	      MPI_Allreduce(MPI_IN_PLACE,
			    rotationMatBlockHostSP,
			    BVec*D,
			    MPI_FLOAT,
			    MPI_SUM,
			    mpiComm);

	      cudaMemcpy(thrust::raw_pointer_cast(&rotationMatBlockSP[0]),
			 rotationMatBlockHostSP,
			 BVec*D*sizeof(float),
			 cudaMemcpyHostToDevice);
	 
	      for (unsigned int idof = 0; idof < maxNumLocalDofs; idof += dofsBlockSize)
		{

		  // Correct block dimensions if block "goes off edge of" the matrix
		  unsigned int BDof=0;
		  if (M>=idof)
		    BDof = std::min(dofsBlockSize, M-idof);

		  if (BDof!=0)
		    {
		      cublasSgemm(handle,
				  CUBLAS_OP_N,
				  CUBLAS_OP_N,
				  BVec,
				  BDof,
				  D,
				  &scalarCoeffAlphaSP,
				  thrust::raw_pointer_cast(&rotationMatBlockSP[0]),
				  BVec,
				  thrust::raw_pointer_cast(&XSP[0])+idof*N,
				  N,
				  &scalarCoeffBetaSP,
				  thrust::raw_pointer_cast(&rotatedVectorsMatBlockSP[0]),
				  BVec);
				
		      addSubspaceRotatedBlockToXKernel<<<(BVec*BDof+255)/256,256>>>(BDof,
										    BVec,
										    thrust::raw_pointer_cast(&rotatedVectorsMatBlockSP[0]),
										    X,
										    idof,
										    jvec,
										    N);
		    }
		}//block loop over dofs
            }//band parallalelization loop
	}//block loop over vectors

      cudaFreeHost(rotationMatBlockHostSP);
      cudaFreeHost(diagValuesHost);
#endif
    }

    void subspaceRotationRRMixedPrecScalapack(double* X,
					      const unsigned int M,
					      const unsigned int N,
					      cublasHandle_t &handle,
					      const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
					      const MPI_Comm &mpiComm,
					      const MPI_Comm &interBandGroupComm,
					      const dealii::ScaLAPACKMatrix<double> & rotationMatPar,
					      const bool rotationMatTranspose)
    {
#ifdef USE_COMPLEX
      AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
      const unsigned int maxNumLocalDofs=dealii::Utilities::MPI::max(M,mpiComm);

      std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperationsCUDA::internal::createGlobalToLocalIdMapsScaLAPACKMat(processGrid,
										   rotationMatPar,
										   globalToLocalRowIdMap,
										   globalToLocalColumnIdMap);

      const unsigned int MPadded=std::ceil(M*1.0/8.0)*8.0+0.5;
      thrust::device_vector<float> XSP(MPadded*N,0.0);

      convDoubleArrToFloatArr<<<(N+255)/256*M,256>>>(N*M,
						     X,
						     thrust::raw_pointer_cast(&XSP[0]));


      //band group parallelization data structures
      const unsigned int numberBandGroups=
	dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(interBandGroupComm,
						 N,
						 bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize=std::min(dftParameters::wfcBlockSize,
						   N);
      const unsigned int dofsBlockSize=std::min(maxNumLocalDofs,
						dftParameters::subspaceRotDofsBlockSize);

      float * rotationMatBlockHostSP;
      cudaMallocHost((void **)&rotationMatBlockHostSP,vectorsBlockSize*N*sizeof(float));
      std::memset(rotationMatBlockHostSP,0,vectorsBlockSize*N*sizeof(float));

      std::vector<float> rotationMatDiagBandHostSP;
    
      double * diagValuesHost;
      cudaMallocHost((void **)&diagValuesHost,N*sizeof(double));
      std::memset(diagValuesHost,0,N*sizeof(double));
    

      thrust::device_vector<float> rotationMatBlockSP(vectorsBlockSize*N,0.0);
      thrust::device_vector<double> diagValues(N,0.0);
      thrust::device_vector<float> rotatedVectorsMatBlockSP(vectorsBlockSize*dofsBlockSize,0.0);
 
      const float scalarCoeffAlphaSP = 1.0,scalarCoeffBetaSP = 0.0;


      //Extract DiagQ from parallel ScaLAPACK matrix Q
      if (rotationMatTranspose)
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
		      diagValuesHost[i]=rotationMatPar.local_el(localRowId,
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
			=rotationMatPar.local_el(it->second,
						 localColumnId);
		    }
		}
	}

      MPI_Allreduce(MPI_IN_PLACE,
		    diagValuesHost,
		    N,
		    MPI_DOUBLE,
		    MPI_SUM,
		    mpiComm);

      cudaMemcpy(thrust::raw_pointer_cast(&diagValues[0]),
		 diagValuesHost,
		 N*sizeof(double),
		 cudaMemcpyHostToDevice);

      computeDiagQTimesXKernel<<<(M*N+255)/256,256>>>(thrust::raw_pointer_cast(&diagValues[0]),
						      X,
						      N,
						      M);

      for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
	{
	  // Correct block dimensions if block "goes off edge of" the matrix
	  const unsigned int BVec = std::min(vectorsBlockSize, N-jvec);

	  const unsigned int D=N;

	  if ((jvec+BVec)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
	      (jvec+BVec)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
	    {

	      std::memset(rotationMatBlockHostSP,0,BVec*N*sizeof(float));

	      //Extract QBVec from parallel ScaLAPACK matrix Q
	      if (rotationMatTranspose)
		{
		  if (processGrid->is_process_active())
		    for (unsigned int i = 0; i <D; ++i)
		      if (globalToLocalRowIdMap.find(i)
			  !=globalToLocalRowIdMap.end())
			{
			  const unsigned int localRowId=globalToLocalRowIdMap[i];
			  for (unsigned int j = 0; j <BVec; ++j)
			    {
			      std::map<unsigned int, unsigned int>::iterator it=
				globalToLocalColumnIdMap.find(j+jvec);
			      if(it!=globalToLocalColumnIdMap.end())
				{
				  rotationMatBlockHostSP[i*BVec+j]=
				    rotationMatPar.local_el(localRowId,
							    it->second);

				}
			    }

			  if (i>=jvec && i<(jvec+BVec))
			    {
			      std::map<unsigned int, unsigned int>::iterator it=
				globalToLocalColumnIdMap.find(i);
			      if (it!=globalToLocalColumnIdMap.end())
				{
				  rotationMatBlockHostSP[i*BVec+i-jvec]=0.0;
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
			  for (unsigned int j = 0; j <BVec; ++j)
			    {
			      std::map<unsigned int, unsigned int>::iterator it=
				globalToLocalRowIdMap.find(j+jvec);
			      if (it!=globalToLocalRowIdMap.end())
				{
				  rotationMatBlockHostSP[i*BVec+j]=
				    rotationMatPar.local_el(it->second,
							    localColumnId);
				}
			    }

			  if (i>=jvec && i<(jvec+BVec))
			    {
			      std::map<unsigned int, unsigned int>::iterator it=
				globalToLocalRowIdMap.find(i);
			      if (globalToLocalRowIdMap.find(i)!=globalToLocalRowIdMap.end())
				{
				  rotationMatBlockHostSP[i*BVec+i-jvec]=0.0;
				}
			    }
			}
		}

	    
	      MPI_Allreduce(MPI_IN_PLACE,
			    rotationMatBlockHostSP,
			    BVec*D,
			    MPI_FLOAT,
			    MPI_SUM,
			    mpiComm);

	      cudaMemcpy(thrust::raw_pointer_cast(&rotationMatBlockSP[0]),
			 rotationMatBlockHostSP,
			 BVec*D*sizeof(float),
			 cudaMemcpyHostToDevice);

	 
	      for (unsigned int idof = 0; idof < maxNumLocalDofs; idof += dofsBlockSize)
		{

		  // Correct block dimensions if block "goes off edge of" the matrix
		  unsigned int BDof=0;
		  if (M>=idof)
		    BDof = std::min(dofsBlockSize, M-idof);

		  if (BDof!=0)
		    {
		      cublasSgemm(handle,
				  CUBLAS_OP_N,
				  CUBLAS_OP_N,
				  BVec,
				  BDof,
				  D,
				  &scalarCoeffAlphaSP,
				  thrust::raw_pointer_cast(&rotationMatBlockSP[0]),
				  BVec,
				  thrust::raw_pointer_cast(&XSP[0])+idof*N,
				  N,
				  &scalarCoeffBetaSP,
				  thrust::raw_pointer_cast(&rotatedVectorsMatBlockSP[0]),
				  BVec);

				
		      addSubspaceRotatedBlockToXKernel<<<(BVec*BDof+255)/256,256>>>(BDof,
										    BVec,
										    thrust::raw_pointer_cast(&rotatedVectorsMatBlockSP[0]),
										    X,
										    idof,
										    jvec,
										    N);
		    }
		}//block loop over dofs
	    }//band parallelization
	}//block loop over vectors

      cudaFreeHost(rotationMatBlockHostSP);
      cudaFreeHost(diagValuesHost);
#endif
    }

    void fillParallelOverlapMatScalapack(const double* X,
					 const unsigned int M,
					 const unsigned int N,
					 cublasHandle_t &handle,
					 const MPI_Comm &mpiComm,
					 const MPI_Comm &interBandGroupComm,
					 const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
					 dealii::ScaLAPACKMatrix<double> & overlapMatPar)
    {
#ifdef USE_COMPLEX
      AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
      //get global to local index maps for Scalapack matrix
      std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperationsCUDA::internal::createGlobalToLocalIdMapsScaLAPACKMat(processGrid,
										   overlapMatPar,
										   globalToLocalRowIdMap,
										   globalToLocalColumnIdMap);

      //band group parallelization data structures
      const unsigned int numberBandGroups=
	dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(interBandGroupComm,
						 N,
						 bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize=std::min(dftParameters::wfcBlockSize,N);

      thrust::device_vector<double> overlapMatrixBlock(N*vectorsBlockSize,0.0);

      double * overlapMatrixBlockHost;
      cudaMallocHost((void **) &overlapMatrixBlockHost, N*vectorsBlockSize*sizeof(double));
      std::memset(overlapMatrixBlockHost,0,vectorsBlockSize*N*sizeof(double));

      const double scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;

      for (unsigned int ivec = 0; ivec < N; ivec += vectorsBlockSize)
	{
	  // Correct block dimensions if block "goes off edge of" the matrix
	  const unsigned int B = std::min(vectorsBlockSize, N-ivec);

	  //thrust::fill(overlapMatrixBlock.begin(),overlapMatrixBlock.end(),0.);

	  const unsigned int D=N-ivec;

	  if ((ivec+B)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
	      (ivec+B)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
	    {

	      // Comptute local XTrunc^{T}*XcBlock.
	      cublasDgemm(handle,
			  CUBLAS_OP_N,
			  CUBLAS_OP_T,
			  D,
			  B,
			  M,
			  &scalarCoeffAlpha,
			  X+ivec,
			  N,
			  X+ivec,
			  N,
			  &scalarCoeffBeta,
			  thrust::raw_pointer_cast(&overlapMatrixBlock[0]),
			  D);

	      cudaMemcpy(overlapMatrixBlockHost,
			 thrust::raw_pointer_cast(&overlapMatrixBlock[0]),
			 D*B*sizeof(double),
			 cudaMemcpyDeviceToHost);

	      // Sum local XTrunc^{T}*XcBlock across domain decomposition processors
	      MPI_Allreduce(MPI_IN_PLACE,
			    overlapMatrixBlockHost,
			    D*B,
			    MPI_DOUBLE,
			    MPI_SUM,
			    mpiComm);


	      //Copying only the lower triangular part to the ScaLAPACK overlap matrix
	      if (processGrid->is_process_active())
		for(unsigned int i = 0; i <B; ++i)
		  if (globalToLocalColumnIdMap.find(i+ivec)
		      !=globalToLocalColumnIdMap.end())
		    {
		      const unsigned int localColumnId
			=globalToLocalColumnIdMap[i+ivec];
		      for (unsigned int j = ivec+i; j <N; ++j)
			{
			  std::map<unsigned int, unsigned int>::iterator it=
			    globalToLocalRowIdMap.find(j);
			  if(it!=globalToLocalRowIdMap.end())
			    overlapMatPar.local_el
			      (it->second,
			       localColumnId)
			      =overlapMatrixBlockHost[i*D+j-ivec];
			}
		    }

	    }//band parallelization
	}//end block loop

      cudaFreeHost(overlapMatrixBlockHost);

      if (numberBandGroups>1)
	linearAlgebraOperationsCUDA::internal::sumAcrossInterCommScaLAPACKMat
	  (processGrid,
	   overlapMatPar,
	   interBandGroupComm);
#endif
    }

    /////////////PSEUDO CODE for the implementation below for Overlapping compute and communication in the computation of overlap matrix/////////////////
    //
    // In the algorithm below the communication and computation of two consecutive blocks of wavefunctions: block i and
    // block i+1 are overlapped.
    // ----------------------------------------------------------  
    // CMP denotes computuation of X^{T} times XBlock
    // COP denotes GPU->CPU copy of X^{T} times XBlock
    // COM denotes blocking MPI_Allreduce on X^{T}XBlock and copy to scalapack matrix
    // ----------------------------------------------------------
    // Two CUDA streams are created: compute and copy
    // CMP is performed in compute CUDA stream and COP is performed in copy CUDA stream.
    // COP for a block can only start after the CMP for that block in the compute stream
    // is completed. COM is performed for a block only after COP even for that block is completed.
    //
    // In a blocked loop do:
    // 1) [CMP] Call compute on first block (edge case only for first iteration)
    // 2) Wait for CMP event for current block to be completed. 
    // 3) Swap current and next block memory (all iterations except edge case)
    // 4) [COP] Call copy on current block
    // 5) [CMP] Call compute on next block
    // 6) Wait for COP event for current block to be completed
    // 7) [COM] Perform blocking MPI_Allreduce on curent block and copy to scalapack matrix
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void fillParallelOverlapMatScalapackAsyncComputeCommun(const double* X,
							   const unsigned int M,
							   const unsigned int N,
							   cublasHandle_t &handle,
							   const MPI_Comm &mpiComm,
							   const MPI_Comm &interBandGroupComm,
							   const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
							   dealii::ScaLAPACKMatrix<double> & overlapMatPar)
    {
#ifdef USE_COMPLEX
      AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
      //get global to local index maps for Scalapack matrix
      std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperationsCUDA::internal::createGlobalToLocalIdMapsScaLAPACKMat(processGrid,
										   overlapMatPar,
										   globalToLocalRowIdMap,
										   globalToLocalColumnIdMap);

      //band group parallelization data structures
      const unsigned int numberBandGroups=
	dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(interBandGroupComm,
						 N,
						 bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize=std::min(dftParameters::wfcBlockSize,N);
      const unsigned int numberBlocks=N/vectorsBlockSize;

      //create separate CUDA streams for GPU->CPU copy and computation
      cudaStream_t streamCompute, streamCopy;
      cudaStreamCreate(&streamCompute);
      cudaStreamCreate(&streamCopy);

      // attach cublas handle to compute stream
      cublasSetStream(handle,streamCompute);  

      // create array of compute and copy events on GPUs
      // for all the blocks. These are required for synchronization
      // between compute, copy and communication as discussed above in the 
      // pseudo code
      cudaEvent_t computeEvents[numberBlocks];
      cudaEvent_t copyEvents[numberBlocks];

      for(int i = 0; i < numberBlocks; ++i)
	{
	  cudaEventCreate(&computeEvents[i]);
	  cudaEventCreate(&copyEvents[i]);
	}

      //create pinned memory used later to copy from GPU->CPU
      double * overlapMatrixBlockHost;
      cudaMallocHost((void **) &overlapMatrixBlockHost, N*vectorsBlockSize*sizeof(double));
      std::memset(overlapMatrixBlockHost,0,vectorsBlockSize*N*sizeof(double));

      //allocate device vectors to be used later
      thrust::device_vector<double> overlapMatrixBlock(N*vectorsBlockSize,0.0);
      thrust::device_vector<double> overlapMatrixBlockNext(N*vectorsBlockSize,0.0);

      const double scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;

      unsigned int blockCount = 0;
      for (unsigned int ivec = 0; ivec < N; ivec += vectorsBlockSize)
	{
	  // Correct block dimensions if block "goes off edge of" the matrix
	  const unsigned int B = std::min(vectorsBlockSize, N-ivec);
	  const unsigned int D=N-ivec;

	  if((ivec+B)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
	     (ivec+B)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
	    {

	      // Compute local XTrunc^{T}*XcBlock.
	      if(ivec == bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
		{

		  //thrust::fill(overlapMatrixBlock.begin(),overlapMatrixBlock.end(),0.0);

		  cublasDgemm(handle,
			      CUBLAS_OP_N,
			      CUBLAS_OP_T,
			      D,
			      B,
			      M,
			      &scalarCoeffAlpha,
			      X+ivec,
			      N,
			      X+ivec,
			      N,
			      &scalarCoeffBeta,
			      thrust::raw_pointer_cast(&overlapMatrixBlock[0]),
			      D);

		  // record completion of compute for first block
		  cudaEventRecord(computeEvents[blockCount],streamCompute);
		}

	      // check for completion of compute on current block before proceeding
	      // to swapping memories and GPU->CPU copy on current block
	      cudaStreamWaitEvent(streamCopy,computeEvents[blockCount],0);

	      if(ivec > bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
		overlapMatrixBlock.swap(overlapMatrixBlockNext);

	      cudaMemcpyAsync(overlapMatrixBlockHost,
			      thrust::raw_pointer_cast(&overlapMatrixBlock[0]),
			      D*B*sizeof(double),
			      cudaMemcpyDeviceToHost,
			      streamCopy);

	      // record completion of GPU->CPU copy for current block
	      cudaEventRecord(copyEvents[blockCount],streamCopy);

	      const unsigned int ivecNew = ivec + vectorsBlockSize;
	      const unsigned int DNew = N - ivecNew;
	      const unsigned int BNew = min(vectorsBlockSize, N - ivecNew);


	      //start computations on the next block
	      if(ivecNew < bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1])
		{
		  //thrust::fill(overlapMatrixBlockNext.begin(),overlapMatrixBlockNext.end(),0.);
		     
		  //evaluate X^{T} times XBlock
		  cublasDgemm(handle,
			      CUBLAS_OP_N,
			      CUBLAS_OP_T,
			      DNew,
			      BNew,
			      M,
			      &scalarCoeffAlpha,
			      X+ivecNew,
			      N,
			      X+ivecNew,
			      N,
			      &scalarCoeffBeta,
			      thrust::raw_pointer_cast(&overlapMatrixBlockNext[0]),
			      DNew);

		  //record completion of compute for next block
		  cudaEventRecord(computeEvents[blockCount+1],streamCompute);

		}

	      // Check that GPU->CPU on the current block has been completed. If completed,
	      // perform blocking MPI commmunication on the current block and copy to ScaLAPACK matri
	      if(cudaEventSynchronize(copyEvents[blockCount]) == cudaSuccess)
		{
		  // Sum local XTrunc^{T}*XcBlock across domain decomposition processors
		  MPI_Allreduce(MPI_IN_PLACE,
				overlapMatrixBlockHost,
				D*B,
				MPI_DOUBLE,
				MPI_SUM,
				mpiComm);


		  //Copying only the lower triangular part to the ScaLAPACK overlap matrix
		  if (processGrid->is_process_active())
		    for(unsigned int i = 0; i <B; ++i)
		      if (globalToLocalColumnIdMap.find(i+ivec)
			  !=globalToLocalColumnIdMap.end())
			{
			  const unsigned int localColumnId
			    =globalToLocalColumnIdMap[i+ivec];
			  for (unsigned int j = ivec+i; j <N; ++j)
			    {
			      std::map<unsigned int, unsigned int>::iterator it=
				globalToLocalRowIdMap.find(j);
			      if(it!=globalToLocalRowIdMap.end())
				overlapMatPar.local_el
				  (it->second,
				   localColumnId)
				  =overlapMatrixBlockHost[i*D+j-ivec];
			    }
			}

		}
	    }//band parallelization
		 
	  blockCount+=1;

	}//end block loop

      cudaFreeHost(overlapMatrixBlockHost);
      // return cublas handle to default stream
      cublasSetStream(handle,NULL);

      if (numberBandGroups>1)
	{

	  MPI_Barrier(interBandGroupComm);

	  linearAlgebraOperationsCUDA::internal::sumAcrossInterCommScaLAPACKMat
	    (processGrid,
	     overlapMatPar,
	     interBandGroupComm);
	}
#endif
    }


    void fillParallelOverlapMatMixedPrecScalapack(const double* X,
						  const unsigned int M,
						  const unsigned int N,
						  cublasHandle_t &handle,
						  const MPI_Comm &mpiComm,
						  const MPI_Comm &interBandGroupComm,
						  const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
						  dealii::ScaLAPACKMatrix<double> & overlapMatPar)
    {
#ifdef USE_COMPLEX
      AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
      //get global to local index maps for Scalapack matrix
      std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperationsCUDA::internal::createGlobalToLocalIdMapsScaLAPACKMat(processGrid,
										   overlapMatPar,
										   globalToLocalRowIdMap,
										   globalToLocalColumnIdMap);

      //band group parallelization data structures
      const unsigned int numberBandGroups=
	dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(interBandGroupComm,
						 N,
						 bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize=std::min(dftParameters::wfcBlockSize,N);


      thrust::device_vector<float> overlapMatrixBlockSP(N*vectorsBlockSize,0.0);
      thrust::device_vector<double> overlapMatrixBlockDP(vectorsBlockSize*vectorsBlockSize,0.0);

      const unsigned int MPadded=std::ceil(M*1.0/8.0)*8.0+0.5;
      thrust::device_vector<float> XSP(MPadded*N,0.0);

      convDoubleArrToFloatArr<<<(N+255)/256*M,256>>>(N*M,
						     X,
						     thrust::raw_pointer_cast(&XSP[0]));
      double * overlapMatrixBlockHostDP;
      cudaMallocHost((void **) &overlapMatrixBlockHostDP, vectorsBlockSize*vectorsBlockSize*sizeof(double));
      std::memset(overlapMatrixBlockHostDP,0,vectorsBlockSize*vectorsBlockSize*sizeof(double));

      float * overlapMatrixBlockHostSP;
      cudaMallocHost((void **) &overlapMatrixBlockHostSP, N*vectorsBlockSize*sizeof(float));
      std::memset(overlapMatrixBlockHostSP,0,N*vectorsBlockSize*sizeof(float));

      const double scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;
      const float  scalarCoeffAlphaSP=1.0,scalarCoeffBetaSP=0.0;

      for (unsigned int ivec = 0; ivec < N; ivec += vectorsBlockSize)
	{
	  // Correct block dimensions if block "goes off edge of" the matrix
	  const unsigned int B = std::min(vectorsBlockSize, N-ivec);


	  const unsigned int D=N-ivec;

	  if ((ivec+B)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
	      (ivec+B)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
	    {

	      cublasDgemm(handle,
			  CUBLAS_OP_N,
			  CUBLAS_OP_T,
			  B,
			  B,
			  M,
			  &scalarCoeffAlpha,
			  X+ivec,
			  N,
			  X+ivec,
			  N,
			  &scalarCoeffBeta,
			  thrust::raw_pointer_cast(&overlapMatrixBlockDP[0]),
			  B);

	      const unsigned int DRem=D-B;

	      if (DRem!=0)
		{
		  cublasSgemm(handle,
			      CUBLAS_OP_N,
			      CUBLAS_OP_T,
			      DRem,
			      B,
			      M,
			      &scalarCoeffAlphaSP,
			      thrust::raw_pointer_cast(&XSP[0])+ivec+B,
			      N,
			      thrust::raw_pointer_cast(&XSP[0])+ivec,
			      N,
			      &scalarCoeffBetaSP,
			      thrust::raw_pointer_cast(&overlapMatrixBlockSP[0]),
			      DRem);
		}


	      cudaMemcpy(overlapMatrixBlockHostDP,
			 thrust::raw_pointer_cast(&overlapMatrixBlockDP[0]),
			 B*B*sizeof(double),
			 cudaMemcpyDeviceToHost);

	      cudaMemcpy(overlapMatrixBlockHostSP,
			 thrust::raw_pointer_cast(&overlapMatrixBlockSP[0]),
			 DRem*B*sizeof(float),
			 cudaMemcpyDeviceToHost);

	      // Sum local XTrunc^{T}*XcBlock for double precision across domain decomposition processors
	      MPI_Allreduce(MPI_IN_PLACE,
			    overlapMatrixBlockHostDP,
			    B*B,
			    MPI_DOUBLE,
			    MPI_SUM,
			    mpiComm);

	      // Sum local XTrunc^{T}*XcBlock for single precision across domain decomposition processors
	      MPI_Allreduce(MPI_IN_PLACE,
			    overlapMatrixBlockHostSP,
			    DRem*B,
			    MPI_FLOAT,
			    MPI_SUM,
			    mpiComm);
	       
	      //Copying only the lower triangular part to the ScaLAPACK overlap matrix
	      if (processGrid->is_process_active())
		for(unsigned int i = 0; i <B; ++i)
		  if (globalToLocalColumnIdMap.find(i+ivec)
		      !=globalToLocalColumnIdMap.end())
		    {
		      const unsigned int localColumnId
			=globalToLocalColumnIdMap[i+ivec];
		      for (unsigned int j = ivec+i; j <ivec+B; ++j)
			{
			  std::map<unsigned int, unsigned int>::iterator it=
			    globalToLocalRowIdMap.find(j);
			  if(it!=globalToLocalRowIdMap.end())
			    overlapMatPar.local_el
			      (it->second,
			       localColumnId)
			      =overlapMatrixBlockHostDP[i*B+j-ivec];
			}

		      for (unsigned int j = ivec+B; j <N; ++j)
			{
			  std::map<unsigned int, unsigned int>::iterator it=
			    globalToLocalRowIdMap.find(j);
			  if(it!=globalToLocalRowIdMap.end())
			    overlapMatPar.local_el
			      (it->second,
			       localColumnId)
			      =overlapMatrixBlockHostSP[i*DRem+j-ivec-B];
			}
		    }
	    }//band parallelization
	}//end block loop

      cudaFreeHost(overlapMatrixBlockHostDP);
      cudaFreeHost(overlapMatrixBlockHostSP);

      if (numberBandGroups>1)
	linearAlgebraOperationsCUDA::internal::sumAcrossInterCommScaLAPACKMat
	  (processGrid,
	   overlapMatPar,
	   interBandGroupComm);
#endif
    }


    /////////////PSEUDO CODE for the implementation below for Overlapping compute and communication in the computation of overlap matrix using mixed precision arithmetic/////////////////
    //
    // In the algorithm below the communication and computation of two consecutive blocks of wavefunctions: block i and
    // block i+1 are overlapped.
    // ----------------------------------------------------------  
    // CMP denotes computuation of X^{T} times XBlock
    // COP denotes GPU->CPU copy of X^{T} times XBlock
    // COM denotes blocking MPI_Allreduce on X^{T}XBlock and copy to scalapack matrix
    // ----------------------------------------------------------
    // Two CUDA streams are created: compute and copy
    // CMP is performed in compute CUDA stream and COP is performed in copy CUDA stream.
    // COP for a block can only start after the CMP for that block in the compute stream
    // is completed. COM is performed for a block only after COP even for that block is completed.
    //
    // In a blocked loop do:
    // 1) [CMP] Call compute on first block (edge case only for first iteration)
    // 2) Wait for CMP event for current block to be completed. 
    // 3) Swap current and next block memory (all iterations except edge case)
    // 4) [COP] Call copy on current block
    // 5) [CMP] Call compute on next block
    // 6) Wait for COP event for current block to be completed
    // 7) [COM] Perform blocking MPI_Allreduce on curent block and copy to scalapack matrix
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
 
    void fillParallelOverlapMatMixedPrecScalapackAsyncComputeCommun(const double* X,
								    const unsigned int M,
								    const unsigned int N,
								    cublasHandle_t &handle,
								    const MPI_Comm &mpiComm,
								    const MPI_Comm &interBandGroupComm,
								    const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
								    dealii::ScaLAPACKMatrix<double> & overlapMatPar)
    {
#ifdef USE_COMPLEX
      AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
      //get global to local index maps for Scalapack matrix
      std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperationsCUDA::internal::createGlobalToLocalIdMapsScaLAPACKMat(processGrid,
										   overlapMatPar,
										   globalToLocalRowIdMap,
										   globalToLocalColumnIdMap);

      //band group parallelization data structures
      const unsigned int numberBandGroups=
	dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(interBandGroupComm,
						 N,
						 bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize=std::min(dftParameters::wfcBlockSize,N);
      const unsigned int numberBlocks=N/vectorsBlockSize;

      //create separate CUDA streams for GPU->CPU copy and computation
      cudaStream_t streamCompute, streamCopy;
      cudaStreamCreate(&streamCompute);
      cudaStreamCreate(&streamCopy);

      // attach cublas handle to compute stream
      cublasSetStream(handle,streamCompute);  

      // create array of compute and copy events on GPUs
      // for all the blocks. These are required for synchronization
      // between compute, copy and communication as discussed above in the 
      // pseudo code
      cudaEvent_t computeEvents[numberBlocks];
      cudaEvent_t copyEvents[numberBlocks];

      for(int i = 0; i < numberBlocks; ++i)
	{
	  cudaEventCreate(&computeEvents[i]);
	  cudaEventCreate(&copyEvents[i]);
	}

      thrust::device_vector<float> overlapMatrixBlockSP(N*vectorsBlockSize,0.0);
      thrust::device_vector<double> overlapMatrixBlockDP(vectorsBlockSize*vectorsBlockSize,0.0);
      thrust::device_vector<float> overlapMatrixBlockSPNext(N*vectorsBlockSize,0.0);
      thrust::device_vector<double> overlapMatrixBlockDPNext(vectorsBlockSize*vectorsBlockSize,0.0);

      const unsigned int MPadded=std::ceil(M*1.0/8.0)*8.0+0.5;
      thrust::device_vector<float> XSP(MPadded*N,0.0);

      convDoubleArrToFloatArr<<<(N+255)/256*M,256>>>(N*M,
						     X,
						     thrust::raw_pointer_cast(&XSP[0]));
      double * overlapMatrixBlockHostDP;
      cudaMallocHost((void **) &overlapMatrixBlockHostDP, vectorsBlockSize*vectorsBlockSize*sizeof(double));
      std::memset(overlapMatrixBlockHostDP,0,vectorsBlockSize*vectorsBlockSize*sizeof(double));

      float * overlapMatrixBlockHostSP;
      cudaMallocHost((void **) &overlapMatrixBlockHostSP, N*vectorsBlockSize*sizeof(float));
      std::memset(overlapMatrixBlockHostSP,0,N*vectorsBlockSize*sizeof(float));

      const double scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;
      const float  scalarCoeffAlphaSP=1.0,scalarCoeffBetaSP=0.0;

      unsigned int blockCount = 0;
      for (unsigned int ivec = 0; ivec < N; ivec += vectorsBlockSize)
	{
	  // Correct block dimensions if block "goes off edge of" the matrix
	  const unsigned int B = std::min(vectorsBlockSize, N-ivec);
	  const unsigned int D=N-ivec;

	  if ((ivec+B)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
	      (ivec+B)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
	    {
	       // Compute local XTrunc^{T}*XcBlock
	      if(ivec == bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
		{
		  //thrust::fill(overlapMatrixBlockDP.begin(),overlapMatrixBlockDP.end(),0.0);

		  cublasDgemm(handle,
			      CUBLAS_OP_N,
			      CUBLAS_OP_T,
			      B,
			      B,
			      M,
			      &scalarCoeffAlpha,
			      X+ivec,
			      N,
			      X+ivec,
			      N,
			      &scalarCoeffBeta,
			      thrust::raw_pointer_cast(&overlapMatrixBlockDP[0]),
			      B);

		  const unsigned int DRem=D-B;

		  if (DRem!=0)
		    {
			
		      //thrust::fill(overlapMatrixBlockSP.begin(),overlapMatrixBlockSP.end(),0.0);

		      cublasSgemm(handle,
				  CUBLAS_OP_N,
				  CUBLAS_OP_T,
				  DRem,
				  B,
				  M,
				  &scalarCoeffAlphaSP,
				  thrust::raw_pointer_cast(&XSP[0])+ivec+B,
				  N,
				  thrust::raw_pointer_cast(&XSP[0])+ivec,
				  N,
				  &scalarCoeffBetaSP,
				  thrust::raw_pointer_cast(&overlapMatrixBlockSP[0]),
				  DRem);
		    }

		  // record completion of compute for first block
		  cudaEventRecord(computeEvents[blockCount],streamCompute);
		}

	      // check for completion of compute on current block before proceeding
	      // to swapping memories and GPU->CPU copy on current block
	      cudaStreamWaitEvent(streamCopy,computeEvents[blockCount],0);

	      if(ivec > bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
		{
		  overlapMatrixBlockDP.swap(overlapMatrixBlockDPNext);
		  overlapMatrixBlockSP.swap(overlapMatrixBlockSPNext);
		}

              const unsigned int DRem=D-B;

	      cudaMemcpyAsync(overlapMatrixBlockHostDP,
			      thrust::raw_pointer_cast(&overlapMatrixBlockDP[0]),
			      B*B*sizeof(double),
			      cudaMemcpyDeviceToHost,
			      streamCopy);

	      cudaMemcpyAsync(overlapMatrixBlockHostSP,
			      thrust::raw_pointer_cast(&overlapMatrixBlockSP[0]),
			      DRem*B*sizeof(float),
			      cudaMemcpyDeviceToHost,
			      streamCopy);

	      // record completion of GPU->CPU copy for current block
	      cudaEventRecord(copyEvents[blockCount],streamCopy);

	      const unsigned int ivecNew = ivec + vectorsBlockSize;
	      const unsigned int DNew = N - ivecNew;
	      const unsigned int BNew = min(vectorsBlockSize, N - ivecNew);

	      if(ivecNew < bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1])
		{
		  //thrust::fill(overlapMatrixBlockDPNext.begin(),overlapMatrixBlockDPNext.end(),0.0);

		  //evaluate X^{T} times XBlock
		  cublasDgemm(handle,
			      CUBLAS_OP_N,
			      CUBLAS_OP_T,
			      BNew,
			      BNew,
			      M,
			      &scalarCoeffAlpha,
			      X+ivecNew,
			      N,
			      X+ivecNew,
			      N,
			      &scalarCoeffBeta,
			      thrust::raw_pointer_cast(&overlapMatrixBlockDPNext[0]),
			      BNew);

		  const unsigned int DRemNew=DNew-BNew;

		  if (DRemNew!=0)
		    {
			
		      //thrust::fill(overlapMatrixBlockSPNext.begin(),overlapMatrixBlockSPNext.end(),0.0);

		      cublasSgemm(handle,
				  CUBLAS_OP_N,
				  CUBLAS_OP_T,
				  DRemNew,
				  BNew,
				  M,
				  &scalarCoeffAlphaSP,
				  thrust::raw_pointer_cast(&XSP[0])+ivecNew+BNew,
				  N,
				  thrust::raw_pointer_cast(&XSP[0])+ivecNew,
				  N,
				  &scalarCoeffBetaSP,
				  thrust::raw_pointer_cast(&overlapMatrixBlockSPNext[0]),
				  DRemNew);
		    }

		  //record completion of compute for next block
		  cudaEventRecord(computeEvents[blockCount+1],streamCompute);
		}

	      // Check that GPU->CPU on the current block has been completed. If completed,
	      // perform blocking MPI commmunication on the current block and copy to ScaLAPACK matri
	      if(cudaEventSynchronize(copyEvents[blockCount]) == cudaSuccess)
		{
		
                  const unsigned int DRem=D-B;

		  // Sum local XTrunc^{T}*XcBlock for double precision across domain decomposition processors
		  MPI_Allreduce(MPI_IN_PLACE,
				overlapMatrixBlockHostDP,
				B*B,
				MPI_DOUBLE,
				MPI_SUM,
				mpiComm);

		  // Sum local XTrunc^{T}*XcBlock for single precision across domain decomposition processors
		  MPI_Allreduce(MPI_IN_PLACE,
				overlapMatrixBlockHostSP,
				DRem*B,
				MPI_FLOAT,
				MPI_SUM,
				mpiComm);
	       
		  //Copying only the lower triangular part to the ScaLAPACK overlap matrix
		  if (processGrid->is_process_active())
		    for(unsigned int i = 0; i <B; ++i)
		      if (globalToLocalColumnIdMap.find(i+ivec)
			  !=globalToLocalColumnIdMap.end())
			{
			  const unsigned int localColumnId
			    =globalToLocalColumnIdMap[i+ivec];
			  for (unsigned int j = ivec+i; j <ivec+B; ++j)
			    {
			      std::map<unsigned int, unsigned int>::iterator it=
				globalToLocalRowIdMap.find(j);
			      if(it!=globalToLocalRowIdMap.end())
				overlapMatPar.local_el
				  (it->second,
				   localColumnId)
				  =overlapMatrixBlockHostDP[i*B+j-ivec];
			    }

			  for (unsigned int j = ivec+B; j <N; ++j)
			    {
			      std::map<unsigned int, unsigned int>::iterator it=
				globalToLocalRowIdMap.find(j);
			      if(it!=globalToLocalRowIdMap.end())
				overlapMatPar.local_el
				  (it->second,
				   localColumnId)
				  =overlapMatrixBlockHostSP[i*DRem+j-ivec-B];
			    }
			}
		}
	    }//band parallelization

	  blockCount+=1;

	}//end block loop

      cudaFreeHost(overlapMatrixBlockHostDP);
      cudaFreeHost(overlapMatrixBlockHostSP);
      //return cublas handle to default stream
      cublasSetStream(handle,NULL);

      if (numberBandGroups>1)
	{
	  MPI_Barrier(interBandGroupComm);

	  linearAlgebraOperationsCUDA::internal::sumAcrossInterCommScaLAPACKMat
	    (processGrid,
	     overlapMatPar,
	     interBandGroupComm);
	}
#endif
    }



    void computeEigenResidualNorm(operatorDFTCUDAClass        & operatorMatrix,
				  double* X,
				  distributedGPUVec<double> & XBlock,
				  distributedGPUVec<double> & HXBlock,
				  distributedGPUVec<double> & projectorKetTimesVector,
				  const unsigned int M,
				  const unsigned int N,
				  const std::vector<double>     & eigenValues,
				  const MPI_Comm &mpiComm,
				  const MPI_Comm &interBandGroupComm,
				  cublasHandle_t & handle,
				  std::vector<double> & residualNorm,
				  const bool useBandParal)
    {
#ifdef USE_COMPLEX
      AssertThrow(false,dftUtils::ExcNotImplementedYet());
#else
      //band group parallelization data structures
      const unsigned int numberBandGroups=
	dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(interBandGroupComm,
						 N,
						 bandGroupLowHighPlusOneIndices);


      const unsigned int vectorsBlockSize=dftParameters::wfcBlockSize;
      thrust::device_vector<double> residualNormSquareDevice(N,0.0);
      thrust::device_vector<double> HXBlockFull(vectorsBlockSize*M,0.0);
      thrust::device_vector<double> residualSqDevice(vectorsBlockSize*M,0.0);
      thrust::device_vector<double> onesVecDevice(M,1.0);


      thrust::device_vector<double> eigenValuesDevice(N,0.0);
      cudaMemcpy(thrust::raw_pointer_cast(&eigenValuesDevice[0]),
		 &eigenValues[0],
		 N*sizeof(double),
		 cudaMemcpyHostToDevice);

      const bool scaleFlag = false;
      const double scalar = 1.0;
      const double alpha = 1.0, beta = 0.0;

      for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
	{
	  // Correct block dimensions if block "goes off edge of" the matrix
	  const unsigned int B = std::min(vectorsBlockSize, N-jvec);


	  if (((jvec+B)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
	       (jvec+B)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId]) || !useBandParal)
	    {

	      const unsigned int chebyBlockSize=std::min(dftParameters::chebyWfcBlockSize,N);

	      for (unsigned int k = jvec; k < jvec+B; k +=chebyBlockSize)
		{
		  stridedCopyToBlockKernel<<<(chebyBlockSize+255)/256*M, 256>>>(chebyBlockSize,
										M,
										X,
										N,
										XBlock.begin(),
										k);

		  //evaluate H times XBlock^{T} and store in HXBlock^{T}
		  HXBlock=0.0;
		  operatorMatrix.HX(XBlock,
				    projectorKetTimesVector,
				    M,
				    chebyBlockSize,
				    scaleFlag,
				    scalar,
				    HXBlock);

		  stridedCopyFromBlockKernel<<<(chebyBlockSize+255)/256*M, 256>>>(chebyBlockSize,
										  M,
										  HXBlock.begin(),
										  B,
										  thrust::raw_pointer_cast(&HXBlockFull[0]),
										  k-jvec);
		}

	      computeResidualCUDAKernel<<<(B+255)/256*M, 256>>>(B,
								M,
								N,
								jvec,
								thrust::raw_pointer_cast(&eigenValuesDevice[0]),
								X,
								thrust::raw_pointer_cast(&HXBlockFull[0]),
								thrust::raw_pointer_cast(&residualSqDevice[0]));

	      cublasDgemm(handle,
			  CUBLAS_OP_N,
			  CUBLAS_OP_T,
			  1,
			  B,
			  M,
			  &alpha,
			  thrust::raw_pointer_cast(&onesVecDevice[0]),
			  1,
			  thrust::raw_pointer_cast(&residualSqDevice[0]),
			  B,
			  &beta,
			  thrust::raw_pointer_cast(&residualNormSquareDevice[0]+jvec),
			  1);
	    }
	}


      cudaMemcpy(&residualNorm[0],
		 thrust::raw_pointer_cast(&residualNormSquareDevice[0]),
		 N*sizeof(double),
		 cudaMemcpyDeviceToHost);

      MPI_Allreduce(MPI_IN_PLACE,
		    &residualNorm[0],
		    N,
		    MPI_DOUBLE,
		    MPI_SUM,
		    mpiComm);

      if (numberBandGroups>1 || !useBandParal)
	MPI_Allreduce(MPI_IN_PLACE,
		      &residualNorm[0],
		      N,
		      MPI_DOUBLE,
		      MPI_SUM,
		      interBandGroupComm);
             

      for(unsigned int iWave = 0; iWave < N; ++iWave)
	residualNorm[iWave] = std::sqrt(residualNorm[iWave]);
#endif
    }
  }
}
#endif
