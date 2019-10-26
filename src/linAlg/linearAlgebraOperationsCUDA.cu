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

    }

 
    //
    // evaluate upper bound of the spectrum using k-step Lanczos iteration
    //
    double lanczosUpperBoundEigenSpectrum(operatorDFTCUDAClass & operatorMatrix,
					  const vectorType & vect)
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
      vectorType vVector, fVector, v0Vector;
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
      std::vector<vectorType> v(1),f(1);
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
			 cudaVectorType & XArray,
                         cudaVectorType & YArray,
			 cudaVectorTypeFloat & tempFloatArray,
                         cudaVectorType & projectorKetTimesVector,
			 const unsigned int localVectorSize,
			 const unsigned int numberVectors,
			 const unsigned int m,
			 const double a,
			 const double b,
			 const double a0)
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

	  //
	  //multiply XArray with alpha2
	  //
	  /*
	    cublasDscal(operatorMatrix.getCublasHandle(),
	    totalVectorSize,
	    &alpha2,
	    thrust::raw_pointer_cast(&XArray[0]),
	    inc);
	  */
	  double coeff = -c*alpha1;

	  /*
	    cublasDaxpy(operatorMatrix.getCublasHandle(),
	    totalVectorSize,
	    &coeff,
	    thrust::raw_pointer_cast(&YArray[0]),
	    inc,
	    thrust::raw_pointer_cast(&XArray[0]),
	    inc);
	  */
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
				 dftParameters::useMixedPrecCheby);
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
				 dftParameters::useMixedPrecCheby);
          
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
    
  void chebyshevFilter(operatorDFTCUDAClass & operatorMatrix,
			 cudaVectorType & XArray1,
                         cudaVectorType & YArray1,
			 cudaVectorTypeFloat & tempFloatArray,
                         cudaVectorType & projectorKetTimesVector1,
			 cudaVectorType & XArray2,
                         cudaVectorType & YArray2,
                         cudaVectorType & projectorKetTimesVector2,
			 const unsigned int localVectorSize,
			 const unsigned int numberVectors,
			 const unsigned int m,
			 const double a,
			 const double b,
			 const double a0)
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
				 dftParameters::useMixedPrecCheby);

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
				 dftParameters::useMixedPrecCheby);
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
          if (overlap)
          {
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
            projectorKetTimesVector2.compress_finish(dealii::VectorOperation::add);
            projectorKetTimesVector2.update_ghost_values();
          }

          //unsigned int id2=nvtxRangeStartA("ghost1");
          YArray1.update_ghost_values_start();

          if (overlap)
             operatorMatrix.HXCheby(YArray2,
				 tempFloatArray,
                                 projectorKetTimesVector2,
				 localVectorSize,
				 numberVectors,
				 XArray2,
				 dftParameters::useMixedPrecCheby,
                                 false,
                                 true);

          YArray1.update_ghost_values_finish();
          //nvtxRangeEnd(id2);

          projectorKetTimesVector1=0.0;
          //unsigned int id1=nvtxRangeStartA("compress2");
          if (overlap)
          {
             XArray2.compress_start(dealii::VectorOperation::add);
          }
          operatorMatrix.HXCheby(YArray1,
				 tempFloatArray,
                                 projectorKetTimesVector1,
				 localVectorSize,
				 numberVectors,
				 XArray1,
				 dftParameters::useMixedPrecCheby,
                                 true,
                                 false);

          if (overlap)
          {
            XArray2.compress_finish(dealii::VectorOperation::add);
            XArray2.swap(YArray2);
          }
          //nvtxRangeEnd(id1);

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

          projectorKetTimesVector1.compress_finish(dealii::VectorOperation::add);
          projectorKetTimesVector1.update_ghost_values();

          //unsigned int id3=nvtxRangeStartA("ghost2");
          YArray2.update_ghost_values_start(); 
          operatorMatrix.HXCheby(YArray1,
				 tempFloatArray,
                                 projectorKetTimesVector1,
				 localVectorSize,
				 numberVectors,
				 XArray1,
				 dftParameters::useMixedPrecCheby,
                                 false,
                                 true);

          YArray2.update_ghost_values_finish();
          //nvtxRangeEnd(id3);


          projectorKetTimesVector2=0.0;

          //unsigned int id4=nvtxRangeStartA("compress1");
          XArray1.compress_start(dealii::VectorOperation::add);

          operatorMatrix.HXCheby(YArray2,
				 tempFloatArray,
                                 projectorKetTimesVector2,
				 localVectorSize,
				 numberVectors,
				 XArray2,
				 dftParameters::useMixedPrecCheby,
                                 true,
                                 false);

          XArray1.compress_finish(dealii::VectorOperation::add);
          //nvtxRangeEnd(id4);

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
				   dftParameters::useMixedPrecCheby,
				   false,
				   true);
            XArray2.compress(dealii::VectorOperation::add);
            overlap=false;
          }
          else
            overlap=true;
	}

      XArray1.swap(YArray1); 
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

  void computeEigenResidualNorm(operatorDFTCUDAClass        & operatorMatrix,
			          double* X,
			          cudaVectorType & XBlock,
			          cudaVectorType & HXBlock,
			          cudaVectorType & projectorKetTimesVector,
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
