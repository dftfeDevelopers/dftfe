// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
// authors.
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

#ifndef USE_COMPLEX
#  include <cudaHelpers.h>
#  include <kohnShamDFTOperatorCUDA.h>
#  include <linearAlgebraOperations.h>
#  include <linearAlgebraOperationsInternal.h>
#  include <linearAlgebraOperationsCUDA.h>
#  include <vectorUtilities.h>
#  include <dft.h>
#  include <dftParameters.h>
#  include <dftUtils.h>


namespace dftfe
{
  namespace
  {
#  if __CUDA_ARCH__ < 600
    __device__ double
    atomicAdd(double *address, double val)
    {
      unsigned long long int *address_as_ull =
        (unsigned long long int *)address;
      unsigned long long int old = *address_as_ull, assumed;

      do
        {
          assumed = old;
          old     = atomicCAS(address_as_ull,
                          assumed,
                          __double_as_longlong(val +
                                               __longlong_as_double(assumed)));

          // Note: uses integer comparison to avoid hang in case of NaN (since
          // NaN != NaN)
        }
      while (assumed != old);

      return __longlong_as_double(old);
    }
#  endif


    template <typename numberType>
    __global__ void
    scaleCUDAKernel(const unsigned int contiguousBlockSize,
                    const unsigned int numContiguousBlocks,
                    const double       scalar,
                    numberType *       srcArray,
                    const double *     scalingVector)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numGangsPerContiguousBlock =
        (contiguousBlockSize + (blockDim.x - 1)) / blockDim.x;
      const unsigned int gangBlockId = blockIdx.x / numGangsPerContiguousBlock;
      const unsigned int localThreadId =
        globalThreadId - gangBlockId * numGangsPerContiguousBlock * blockDim.x;
      if (globalThreadId <
            numContiguousBlocks * numGangsPerContiguousBlock * blockDim.x &&
          localThreadId < contiguousBlockSize)
        {
          *(srcArray + (localThreadId + gangBlockId * contiguousBlockSize)) =
            *(srcArray + (localThreadId + gangBlockId * contiguousBlockSize)) *
            (*(scalingVector + gangBlockId) * numberType(scalar));
        }
    }

    template <typename numberType>
    __global__ void
    stridedCopyToBlockKernel(const unsigned int BVec,
                             const unsigned int M,
                             const numberType * xVec,
                             const unsigned int N,
                             numberType *       yVec,
                             const unsigned int startingXVecId)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numGangsPerBVec = (BVec + blockDim.x - 1) / blockDim.x;
      const unsigned int gangBlockId     = blockIdx.x / numGangsPerBVec;
      const unsigned int localThreadId =
        globalThreadId - gangBlockId * numGangsPerBVec * blockDim.x;

      if (globalThreadId < M * numGangsPerBVec * blockDim.x &&
          localThreadId < BVec)
        {
          *(yVec + gangBlockId * BVec + localThreadId) =
            *(xVec + gangBlockId * N + startingXVecId + localThreadId);
        }
    }


    template <typename numberType>
    __global__ void
    stridedCopyFromBlockKernel(const unsigned int BVec,
                               const unsigned int M,
                               const numberType * xVec,
                               const unsigned int N,
                               numberType *       yVec,
                               const unsigned int startingXVecId)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numGangsPerBVec = (BVec + blockDim.x - 1) / blockDim.x;
      const unsigned int gangBlockId     = blockIdx.x / numGangsPerBVec;
      const unsigned int localThreadId =
        globalThreadId - gangBlockId * numGangsPerBVec * blockDim.x;

      if (globalThreadId < M * numGangsPerBVec * blockDim.x &&
          localThreadId < BVec)
        {
          *(yVec + gangBlockId * N + startingXVecId + localThreadId) =
            *(xVec + gangBlockId * BVec + localThreadId);
        }
    }

    template <typename numberType, typename numberTypeFP32>
    __global__ void
    stridedCopyFromBlockKernelSP(const unsigned int BVec,
                                 const unsigned int M,
                                 const numberType * xVec,
                                 const unsigned int N,
                                 numberTypeFP32 *   yVec,
                                 const unsigned int startingXVecId)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numGangsPerBVec = (BVec + blockDim.x - 1) / blockDim.x;
      const unsigned int gangBlockId     = blockIdx.x / numGangsPerBVec;
      const unsigned int localThreadId =
        globalThreadId - gangBlockId * numGangsPerBVec * blockDim.x;

      if (globalThreadId < M * numGangsPerBVec * blockDim.x &&
          localThreadId < BVec)
        {
          *(yVec + gangBlockId * N + startingXVecId + localThreadId) =
            *(xVec + gangBlockId * BVec + localThreadId);
        }
    }

    template <typename numberType, typename numberTypeFP32>
    __global__ void
    convDoubleArrToFloatArr(const unsigned int size,
                            const numberType * doubleArr,
                            numberTypeFP32 *   floatArr)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

      for (unsigned int index = globalThreadId; index < size;
           index += blockDim.x * gridDim.x)
        floatArr[index] = doubleArr[index];
    }


    template <typename numberType, typename numberTypeFP32>
    __global__ void
    convFloatArrToDoubleArr(const unsigned int    size,
                            const numberTypeFP32 *floatArr,
                            numberType *          doubleArr)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

      for (unsigned int index = globalThreadId; index < size;
           index += blockDim.x * gridDim.x)
        doubleArr[index] = floatArr[index];
    }


    template <typename numberType, typename numberTypeFP32>
    __global__ void
    copyFloatArrToDoubleArrLocallyOwned(const unsigned int contiguousBlockSize,
                                        const unsigned int numContiguousBlocks,
                                        const numberTypeFP32 *floatArr,
                                        const unsigned int *locallyOwnedFlagArr,
                                        numberType *        doubleArr)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          if (locallyOwnedFlagArr[blockIndex] == 1)
            doubleArr[index] = floatArr[index];
        }
    }

    template <typename numberType>
    __global__ void
    copyCUDAKernel(const unsigned int contiguousBlockSize,
                   const unsigned int numContiguousBlocks,
                   const numberType * copyFromVec,
                   numberType *       copyToVec,
                   const dealii::types::global_dof_index
                     *copyFromVecStartingContiguousBlockIds)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;
          copyToVec[index] =
            copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex];
        }
    }

    template <typename numberType>
    __global__ void
    daxpyAtomicAddKernel(
      const unsigned int                     contiguousBlockSize,
      const unsigned int                     numContiguousBlocks,
      const numberType *                     addFromVec,
      numberType *                           addToVec,
      const dealii::types::global_dof_index *addToVecStartingContiguousBlockIds)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;
          atomicAdd(&addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                              intraBlockIndex],
                    addFromVec[index]);
        }
    }


    template <typename numberType>
    __global__ void
    copyToParallelNonLocalVecFromReducedVec(
      const unsigned int  numWfcs,
      const unsigned int  totalPseudoWfcs,
      const numberType *  reducedProjectorKetTimesWfcVec,
      numberType *        projectorKetTimesWfcParallelVec,
      const unsigned int *indexMapFromParallelVecToReducedVec)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries  = totalPseudoWfcs * numWfcs;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const unsigned int blockIndex      = index / numWfcs;
          const unsigned int intraBlockIndex = index % numWfcs;
          // projectorKetTimesWfcParallelVec[index]
          //        =reducedProjectorKetTimesWfcVec[indexMapFromParallelVecToReducedVec[blockIndex]*numWfcs+intraBlockIndex];
          projectorKetTimesWfcParallelVec
            [indexMapFromParallelVecToReducedVec[blockIndex] * numWfcs +
             intraBlockIndex] = reducedProjectorKetTimesWfcVec[index];
        }
    }

    template <typename numberType>
    __global__ void
    copyFromParallelNonLocalVecToAllCellsVec(
      const unsigned int numWfcs,
      const unsigned int numNonLocalCells,
      const unsigned int maxSingleAtomPseudoWfc,
      const numberType * projectorKetTimesWfcParallelVec,
      numberType *       projectorKetTimesWfcAllCellsVec,
      const int *        indexMapPaddedToParallelVec)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries =
        numNonLocalCells * maxSingleAtomPseudoWfc * numWfcs;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const unsigned int blockIndex      = index / numWfcs;
          const unsigned int intraBlockIndex = index % numWfcs;
          const int mappedIndex = indexMapPaddedToParallelVec[blockIndex];
          if (mappedIndex != -1)
            projectorKetTimesWfcAllCellsVec[index] =
              projectorKetTimesWfcParallelVec[mappedIndex * numWfcs +
                                              intraBlockIndex];
        }
    }


    template <typename numberType>
    __global__ void
    copyToDealiiParallelNonLocalVec(
      const unsigned int  numWfcs,
      const unsigned int  totalPseudoWfcs,
      const numberType *  projectorKetTimesWfcParallelVec,
      numberType *        projectorKetTimesWfcDealiiParallelVec,
      const unsigned int *indexMapDealiiParallelNumbering)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries  = totalPseudoWfcs * numWfcs;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const unsigned int blockIndex      = index / numWfcs;
          const unsigned int intraBlockIndex = index % numWfcs;
          const unsigned int mappedIndex =
            indexMapDealiiParallelNumbering[blockIndex];

          projectorKetTimesWfcDealiiParallelVec[mappedIndex * numWfcs +
                                                intraBlockIndex] =
            projectorKetTimesWfcParallelVec[index];
        }
    }

    template <typename numberType>
    __global__ void
    copyFromDealiiParallelNonLocalVec(
      const unsigned int  numWfcs,
      const unsigned int  totalPseudoWfcs,
      numberType *        projectorKetTimesWfcParallelVec,
      const numberType *  projectorKetTimesWfcDealiiParallelVec,
      const unsigned int *indexMapDealiiParallelNumbering)
    {
      const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const unsigned int numberEntries  = totalPseudoWfcs * numWfcs;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const unsigned int blockIndex      = index / numWfcs;
          const unsigned int intraBlockIndex = index % numWfcs;
          const unsigned int mappedIndex =
            indexMapDealiiParallelNumbering[blockIndex];

          projectorKetTimesWfcParallelVec[index] =
            projectorKetTimesWfcDealiiParallelVec[mappedIndex * numWfcs +
                                                  intraBlockIndex];
        }
    }

    template <typename numberType>
    __global__ void
    addNonLocalContributionCUDAKernel(
      const dealii::types::global_dof_index contiguousBlockSize,
      const dealii::types::global_dof_index numContiguousBlocks,
      const numberType *                    xVec,
      numberType *                          yVec,
      const unsigned int *                  xVecToyVecBlockIdMap)
    {
      const dealii::types::global_dof_index globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dealii::types::global_dof_index numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dealii::types::global_dof_index blockIndex =
            index / contiguousBlockSize;
          dealii::types::global_dof_index intraBlockIndex =
            index % contiguousBlockSize;
          yVec[xVecToyVecBlockIdMap[blockIndex] * contiguousBlockSize +
               intraBlockIndex] += xVec[index];
        }
    }


  } // namespace

  //
  // constructor
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    kohnShamDFTOperatorCUDAClass(dftClass<FEOrder, FEOrderElectro> *_dftPtr,
                                 const MPI_Comm &mpi_comm_replica)
    : dftPtr(_dftPtr)
    , d_kPointIndex(0)
    , d_numberNodesPerElement(_dftPtr->matrix_free_data.get_dofs_per_cell())
    , d_numberMacroCells(_dftPtr->matrix_free_data.n_macro_cells())
    , d_numLocallyOwnedCells(dftPtr->matrix_free_data.n_physical_cells())
    , d_numQuadPoints(
        dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId)
          .size())
    , d_isStiffnessMatrixExternalPotCorrComputed(false)
    , d_isMallocCalled(false)
    , mpi_communicator(mpi_comm_replica)
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_comm_replica))
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_comm_replica))
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
    , computing_timer(mpi_comm_replica,
                      pcout,
                      TimerOutput::never,
                      TimerOutput::wall_times)
    , operatorDFTCUDAClass(mpi_comm_replica,
                           _dftPtr->getMatrixFreeData(),
                           _dftPtr->constraintsNoneDataInfo,
                           _dftPtr->d_constraintsNoneDataInfoCUDA)
  {}

  //
  // destructor
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  kohnShamDFTOperatorCUDAClass<FEOrder,
                               FEOrderElectro>::~kohnShamDFTOperatorCUDAClass()
  {
    if (d_isMallocCalled)
      {
        free(h_d_A);
        free(h_d_B);
        free(h_d_C);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
      }
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::createCublasHandle()
  {
    int n_devices = 0;
    cudaGetDeviceCount(&n_devices);
    int device_id =
      dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) % n_devices;
    cublasCreate(&d_cublasHandle);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::destroyCublasHandle()
  {
    cublasDestroy(d_cublasHandle);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  cublasHandle_t &
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::getCublasHandle()
  {
    return d_cublasHandle;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  const double *
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::getSqrtMassVec()
  {
    return thrust::raw_pointer_cast(&d_sqrtMassVectorDevice[0]);
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  const double *
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::getInvSqrtMassVec()
  {
    return thrust::raw_pointer_cast(&d_invSqrtMassVectorDevice[0]);
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  distributedCPUVec<dataTypes::number> &
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    getProjectorKetTimesVectorSingle()
  {
    return dftPtr->d_projectorKetTimesVectorPar[0];
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  thrust::device_vector<double> &
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    getShapeFunctionGradientIntegral()
  {
    return d_cellShapeFunctionGradientIntegralFlattenedDevice;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  thrust::device_vector<double> &
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    getShapeFunctionGradientIntegralElectro()
  {
    return d_cellShapeFunctionGradientIntegralFlattenedDeviceElectro;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  thrust::device_vector<double> &
  kohnShamDFTOperatorCUDAClass<FEOrder,
                               FEOrderElectro>::getShapeFunctionValues()
  {
    return d_shapeFunctionValueDevice;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  thrust::device_vector<double> &
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    getShapeFunctionValuesInverted(const bool use2pPlusOneGLQuad)
  {
    return use2pPlusOneGLQuad ? d_glShapeFunctionValueInvertedDevice :
                                d_shapeFunctionValueInvertedDevice;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  thrust::device_vector<double> &
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    getShapeFunctionValuesNLPInverted()
  {
    return d_shapeFunctionValueNLPInvertedDevice;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  thrust::device_vector<double> &
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    getShapeFunctionGradientValuesX()
  {
    return d_shapeFunctionGradientValueXDevice;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  thrust::device_vector<double> &
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    getShapeFunctionGradientValuesY()
  {
    return d_shapeFunctionGradientValueYDevice;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  thrust::device_vector<double> &
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    getShapeFunctionGradientValuesZ()
  {
    return d_shapeFunctionGradientValueZDevice;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  thrust::device_vector<double> &
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    getShapeFunctionGradientValuesXInverted()
  {
    return d_shapeFunctionGradientValueXInvertedDevice;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  thrust::device_vector<double> &
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    getShapeFunctionGradientValuesYInverted()
  {
    return d_shapeFunctionGradientValueYInvertedDevice;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  thrust::device_vector<double> &
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    getShapeFunctionGradientValuesZInverted()
  {
    return d_shapeFunctionGradientValueZInvertedDevice;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  thrust::device_vector<double> &
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    getShapeFunctionGradientValuesNLPInverted()
  {
    return d_shapeFunctionGradientValueNLPInvertedDevice;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  thrust::device_vector<double> &
  kohnShamDFTOperatorCUDAClass<FEOrder,
                               FEOrderElectro>::getInverseJacobiansNLP()
  {
    return d_inverseJacobiansNLPDevice;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  thrust::device_vector<dealii::types::global_dof_index> &
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    getFlattenedArrayCellLocalProcIndexIdMap()
  {
    return d_flattenedArrayCellLocalProcIndexIdMapDevice;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  thrust::device_vector<dataTypes::numberThrustGPU> &
  kohnShamDFTOperatorCUDAClass<FEOrder,
                               FEOrderElectro>::getCellWaveFunctionMatrix()
  {
    return d_cellWaveFunctionMatrix;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  distributedGPUVec<dataTypes::numberGPU> &
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    getParallelChebyBlockVectorDevice()
  {
    return d_parallelChebyBlockVectorDevice;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  distributedGPUVec<dataTypes::numberGPU> &
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    getParallelProjectorKetTimesBlockVectorDevice()
  {
    return d_parallelProjectorKetTimesBlockVectorDevice;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  thrust::device_vector<unsigned int> &
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    getLocallyOwnedProcBoundaryNodesVectorDevice()
  {
    return d_locallyOwnedProcBoundaryNodesVectorDevice;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  thrust::device_vector<unsigned int> &
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    getLocallyOwnedProcProjectorKetBoundaryNodesVectorDevice()
  {
    return d_locallyOwnedProcProjectorKetBoundaryNodesVectorDevice;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  thrust::device_vector<unsigned int> &
  kohnShamDFTOperatorCUDAClass<FEOrder,
                               FEOrderElectro>::getBoundaryIdToLocalIdMap()
  {
    return d_boundaryIdToLocalIdMapDevice;
  }

  //
  // initialize kohnShamDFTOperatorCUDAClass object
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::init()
  {
    computing_timer.enter_subsection("kohnShamDFTOperatorCUDAClass setup");


    dftPtr->matrix_free_data.initialize_dof_vector(
      d_invSqrtMassVector, dftPtr->d_densityDofHandlerIndex);
    d_sqrtMassVector.reinit(d_invSqrtMassVector);



    //
    // compute mass vector
    //
    computeMassVector(dftPtr->dofHandler,
                      dftPtr->constraintsNone,
                      d_sqrtMassVector,
                      d_invSqrtMassVector);

    computing_timer.leave_subsection("kohnShamDFTOperatorCUDAClass setup");
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::reinit(
    const unsigned int numberWaveFunctions,
    bool               flag)
  {
    distributedCPUVec<dataTypes::number> flattenedArray;
    if (flag)
      vectorTools::createDealiiVector<dataTypes::number>(
        dftPtr->matrix_free_data.get_vector_partitioner(
          dftPtr->d_densityDofHandlerIndex),
        numberWaveFunctions,
        flattenedArray);

    size_t free_t, total_t;

    cudaMemGetInfo(&free_t, &total_t);
    if (dftParameters::verbosity >= 2)
      pcout << "starting free mem: " << free_t << ", total mem: " << total_t
            << std::endl;

    const unsigned int BVec =
      std::min(dftParameters::chebyWfcBlockSize, numberWaveFunctions);
    d_parallelChebyBlockVectorDevice.reinit(
      dftPtr->matrix_free_data.get_vector_partitioner(
        dftPtr->d_densityDofHandlerIndex),
      BVec);

    const unsigned int n_ghosts =
      dftPtr->matrix_free_data
        .get_vector_partitioner(dftPtr->d_densityDofHandlerIndex)
        ->n_ghost_indices();
    const unsigned int localSize =
      dftPtr->matrix_free_data
        .get_vector_partitioner(dftPtr->d_densityDofHandlerIndex)
        ->local_size();

    thrust::host_vector<unsigned int> locallyOwnedProcBoundaryNodesVector(
      localSize, 0);

    const std::vector<std::pair<unsigned int, unsigned int>>
      &locallyOwnedProcBoundaryNodes =
        dftPtr->matrix_free_data
          .get_vector_partitioner(dftPtr->d_densityDofHandlerIndex)
          ->import_indices();

    for (unsigned int iset = 0; iset < locallyOwnedProcBoundaryNodes.size();
         ++iset)
      {
        const std::pair<unsigned int, unsigned int> &localIndices =
          locallyOwnedProcBoundaryNodes[iset];
        for (unsigned int inode = localIndices.first;
             inode < localIndices.second;
             ++inode)
          {
            locallyOwnedProcBoundaryNodesVector[inode] = 1;
          }
      }

    d_locallyOwnedProcBoundaryNodesVectorDevice.resize(localSize);

    /*cudaMemcpy(thrust::raw_pointer_cast(&d_locallyOwnedProcBoundaryNodesVectorDevice[0]),
      locallyOwnedProcBoundaryNodesVector.begin(),
      localSize*sizeof(unsigned int),
      cudaMemcpyHostToDevice);*/

    d_locallyOwnedProcBoundaryNodesVectorDevice =
      locallyOwnedProcBoundaryNodesVector;


    vectorTools::computeCellLocalIndexSetMap(
      flattenedArray.get_partitioner(),
      dftPtr->matrix_free_data,
      dftPtr->d_densityDofHandlerIndex,
      numberWaveFunctions,
      d_flattenedArrayMacroCellLocalProcIndexIdMapFlattened,
      d_normalCellIdToMacroCellIdMap,
      d_macroCellIdToNormalCellIdMap,
      d_flattenedArrayCellLocalProcIndexIdMap);

    d_flattenedArrayCellLocalProcIndexIdMapDevice =
      d_flattenedArrayCellLocalProcIndexIdMap;



    getOverloadedConstraintMatrix()->precomputeMaps(
      dftPtr->matrix_free_data.get_vector_partitioner(
        dftPtr->d_densityDofHandlerIndex),
      flattenedArray.get_partitioner(),
      numberWaveFunctions);

    getOverloadedConstraintMatrixHost()->precomputeMaps(
      dftPtr->matrix_free_data.get_vector_partitioner(),
      dftPtr->matrix_free_data.get_vector_partitioner(),
      1);


    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();

    d_cellHamiltonianMatrixFlattenedDevice.resize(
      d_numLocallyOwnedCells * d_numberNodesPerElement *
        d_numberNodesPerElement * dftPtr->d_kPointWeights.size() *
        (1 + dftParameters::spinPolarized),
      0.0);

    if (dftParameters::isPseudopotential)
      d_cellHamiltonianMatrixExternalPotCorrFlattenedDevice.resize(
        d_numLocallyOwnedCells * d_numberNodesPerElement *
          d_numberNodesPerElement,
        0.0);
    else
      d_cellHamiltonianMatrixExternalPotCorrFlattenedDevice.resize(10, 0.0);

    d_cellWaveFunctionMatrix.resize(totalLocallyOwnedCells *
                                      d_numberNodesPerElement *
                                      numberWaveFunctions,
                                    0.0);

    d_cellHamMatrixTimesWaveMatrix.resize(totalLocallyOwnedCells *
                                            d_numberNodesPerElement *
                                            numberWaveFunctions,
                                          0.0);

    if (dftParameters::isPseudopotential)
      {
        d_parallelProjectorKetTimesBlockVectorDevice.reinit(
          dftPtr->d_projectorKetTimesVectorPar[0].get_partitioner(), BVec);


        thrust::host_vector<unsigned int>
          locallyOwnedProcProjectorKetBoundaryNodesVector(
            dftPtr->d_projectorKetTimesVectorPar[0]
              .get_partitioner()
              ->local_size(),
            0);

        const std::vector<std::pair<unsigned int, unsigned int>>
          &locallyOwnedProcProjectorKetBoundaryNodes =
            dftPtr->d_projectorKetTimesVectorPar[0]
              .get_partitioner()
              ->import_indices();

        for (unsigned int iset = 0;
             iset < locallyOwnedProcProjectorKetBoundaryNodes.size();
             ++iset)
          {
            const std::pair<unsigned int, unsigned int> &localIndices =
              locallyOwnedProcProjectorKetBoundaryNodes[iset];
            for (unsigned int inode = localIndices.first;
                 inode < localIndices.second;
                 ++inode)
              {
                locallyOwnedProcProjectorKetBoundaryNodesVector[inode] = 1;
              }
          }

        d_locallyOwnedProcProjectorKetBoundaryNodesVectorDevice.resize(
          dftPtr->d_projectorKetTimesVectorPar[0]
            .get_partitioner()
            ->local_size());
        d_locallyOwnedProcProjectorKetBoundaryNodesVectorDevice =
          locallyOwnedProcProjectorKetBoundaryNodesVector;


        d_totalPseudoWfcNonLocal = 0;
        d_totalNonlocalElems     = 0;
        d_totalNonlocalAtomsCurrentProc =
          dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
        unsigned int maxPseudoWfc = 0;
        d_numberCellsAccumNonLocalAtoms.resize(d_totalNonlocalAtomsCurrentProc);
        std::vector<unsigned int> numPseduoWfcsAccum(
          d_totalNonlocalAtomsCurrentProc);
        for (unsigned int iAtom = 0;
             iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
             ++iAtom)
          {
            const unsigned int atomId =
              dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
            const unsigned int numberSingleAtomPseudoWaveFunctions =
              dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
            if (numberSingleAtomPseudoWaveFunctions > maxPseudoWfc)
              maxPseudoWfc = numberSingleAtomPseudoWaveFunctions;

            numPseduoWfcsAccum[iAtom] = d_totalPseudoWfcNonLocal;
            d_totalPseudoWfcNonLocal += numberSingleAtomPseudoWaveFunctions;
            const unsigned int numberElementsInCompactSupport =
              dftPtr->d_elementIteratorsInAtomCompactSupport[atomId].size();
            d_numberCellsAccumNonLocalAtoms[iAtom] = d_totalNonlocalElems;
            d_totalNonlocalElems += numberElementsInCompactSupport;
          }

        d_maxSingleAtomPseudoWfc = maxPseudoWfc;
        d_cellHamMatrixTimesWaveMatrixNonLocalDevice.resize(
          d_totalNonlocalElems * numberWaveFunctions * d_numberNodesPerElement,
          0.0);
        d_cellHamiltonianMatrixNonLocalFlattened.resize(
          d_totalNonlocalElems * d_numberNodesPerElement *
            d_maxSingleAtomPseudoWfc,
          0.0);
        d_cellHamiltonianMatrixNonLocalFlattenedTranspose.resize(
          d_totalNonlocalElems * d_numberNodesPerElement *
            d_maxSingleAtomPseudoWfc,
          0.0);
        d_nonLocalPseudoPotentialConstants.resize(d_totalPseudoWfcNonLocal,
                                                  0.0);
        d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal.resize(
          d_totalNonlocalElems * d_numberNodesPerElement, 0);
        d_projectorKetTimesVectorAllCellsDevice.resize(
          d_totalNonlocalElems * numberWaveFunctions * d_maxSingleAtomPseudoWfc,
          0.0);

        d_projectorIdsParallelNumberingMap.resize(d_totalPseudoWfcNonLocal, 0);
        d_projectorKetTimesVectorParFlattenedDevice.resize(
          numberWaveFunctions * d_totalPseudoWfcNonLocal, 0.0);

        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.clear();
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec.resize(
          d_totalNonlocalElems * d_maxSingleAtomPseudoWfc, -1);

        d_nonlocalElemIdToLocalElemIdMap.clear();
        d_nonlocalElemIdToLocalElemIdMap.resize(d_totalNonlocalElems, 0);

        d_projectorKetTimesVectorAllCellsReduction.clear();
        d_projectorKetTimesVectorAllCellsReduction.resize(
          d_totalNonlocalElems * d_maxSingleAtomPseudoWfc *
            d_totalPseudoWfcNonLocal,
          0.0);

        d_cellNodeIdMapNonLocalToLocal.clear();
        d_cellNodeIdMapNonLocalToLocal.resize(d_totalNonlocalElems *
                                              d_numberNodesPerElement);

        unsigned int countElemNode   = 0;
        unsigned int countElem       = 0;
        unsigned int countPseudoWfc1 = 0;
        d_numberCellsNonLocalAtoms.resize(d_totalNonlocalAtomsCurrentProc);
        for (unsigned int iAtom = 0; iAtom < d_totalNonlocalAtomsCurrentProc;
             ++iAtom)
          {
            const unsigned int atomId =
              dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
            const unsigned int numberPseudoWaveFunctions =
              dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];

            d_numberCellsNonLocalAtoms[iAtom] =
              dftPtr->d_elementIteratorsInAtomCompactSupport[atomId].size();

            for (unsigned int ipseudowfc = 0;
                 ipseudowfc < numberPseudoWaveFunctions;
                 ++ipseudowfc)
              {
                const unsigned int id =
                  dftPtr->d_projectorKetTimesVectorPar[0]
                    .get_partitioner()
                    ->global_to_local(
                      dftPtr->d_projectorIdsNumberingMapCurrentProcess
                        [std::make_pair(atomId, ipseudowfc)]);

                d_projectorIdsParallelNumberingMap[countPseudoWfc1] = id;
                // std::cout<<"iAtom: "<< iAtom<<", ipseudo: "<< ipseudowfc <<",
                // netpseudo: "<<countPseudoWfc1<<", parallel id:
                // "<<id<<std::endl;
                // d_nonLocalPseudoPotentialConstants[countPseudoWfc1]
                //   =dftPtr->d_nonLocalPseudoPotentialConstants[atomId][ipseudowfc];
                d_nonLocalPseudoPotentialConstants[id] =
                  dftPtr
                    ->d_nonLocalPseudoPotentialConstants[atomId][ipseudowfc];
                for (unsigned int iElemComp = 0;
                     iElemComp <
                     dftPtr->d_elementIteratorsInAtomCompactSupport[atomId]
                       .size();
                     ++iElemComp)
                  d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec
                    [d_numberCellsAccumNonLocalAtoms[iAtom] *
                       d_maxSingleAtomPseudoWfc +
                     iElemComp * d_maxSingleAtomPseudoWfc + ipseudowfc] =
                      id; // countPseudoWfc1;//id;

                countPseudoWfc1++;
              }

            for (unsigned int iElemComp = 0;
                 iElemComp <
                 dftPtr->d_elementIteratorsInAtomCompactSupport[atomId].size();
                 ++iElemComp)
              {
                const unsigned int elementId =
                  dftPtr->d_elementIdsInAtomCompactSupport[atomId][iElemComp];
                for (unsigned int iNode = 0; iNode < d_numberNodesPerElement;
                     ++iNode)
                  {
                    dealii::types::global_dof_index localNodeId =
                      d_flattenedArrayCellLocalProcIndexIdMap
                        [elementId * d_numberNodesPerElement + iNode];
                    d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal
                      [countElemNode] = localNodeId;
                    d_cellNodeIdMapNonLocalToLocal[countElemNode] =
                      elementId * d_numberNodesPerElement + iNode;
                    countElemNode++;
                  }
              }

            for (unsigned int iElemComp = 0;
                 iElemComp <
                 dftPtr->d_elementIteratorsInAtomCompactSupport[atomId].size();
                 ++iElemComp)
              {
                const unsigned int elementId =
                  dftPtr->d_elementIdsInAtomCompactSupport[atomId][iElemComp];
                d_nonlocalElemIdToLocalElemIdMap[countElem] = elementId;
                for (unsigned int iNode = 0; iNode < d_numberNodesPerElement;
                     ++iNode)
                  {
                    for (unsigned int iPseudoWave = 0;
                         iPseudoWave < numberPseudoWaveFunctions;
                         ++iPseudoWave)
                      {
                        d_cellHamiltonianMatrixNonLocalFlattened
                          [countElem * d_maxSingleAtomPseudoWfc *
                             d_numberNodesPerElement +
                           d_numberNodesPerElement * iPseudoWave + iNode] =
                            dftPtr->d_nonLocalProjectorElementMatrices
                              [atomId][iElemComp]
                              [d_numberNodesPerElement * iPseudoWave + iNode];
                        d_cellHamiltonianMatrixNonLocalFlattenedTranspose
                          [countElem * d_numberNodesPerElement *
                             d_maxSingleAtomPseudoWfc +
                           d_maxSingleAtomPseudoWfc * iNode + iPseudoWave] =
                            dftPtr->d_nonLocalProjectorElementMatricesTranspose
                              [atomId][iElemComp]
                              [numberPseudoWaveFunctions * iNode + iPseudoWave];
                      }
                  }


                for (unsigned int iPseudoWave = 0;
                     iPseudoWave < numberPseudoWaveFunctions;
                     ++iPseudoWave)
                  {
                    const unsigned int columnStartId =
                      (numPseduoWfcsAccum[iAtom] + iPseudoWave) *
                      d_totalNonlocalElems * d_maxSingleAtomPseudoWfc;
                    const unsigned int columnRowId =
                      countElem * d_maxSingleAtomPseudoWfc + iPseudoWave;
                    d_projectorKetTimesVectorAllCellsReduction[columnStartId +
                                                               columnRowId] =
                      1.0;
                  }

                countElem++;
              }
          }

        d_cellHamiltonianMatrixNonLocalFlattenedDevice =
          d_cellHamiltonianMatrixNonLocalFlattened;
        d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice =
          d_cellHamiltonianMatrixNonLocalFlattenedTranspose;
        d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocalDevice =
          d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocal;
        d_projectorIdsParallelNumberingMapDevice =
          d_projectorIdsParallelNumberingMap;
        // d_indexMapFromParallelNonLocalVecToReducedVecDevice=d_indexMapFromParallelNonLocalVecToReducedVec;
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice =
          d_indexMapFromPaddedNonLocalVecToParallelNonLocalVec;
        d_projectorKetTimesVectorAllCellsReductionDevice =
          d_projectorKetTimesVectorAllCellsReduction;
        d_nonLocalPseudoPotentialConstantsDevice =
          d_nonLocalPseudoPotentialConstants;
        d_cellNodeIdMapNonLocalToLocalDevice = d_cellNodeIdMapNonLocalToLocal;


        h_d_A = (dataTypes::number **)malloc(d_totalNonlocalElems *
                                             sizeof(dataTypes::numberGPU *));
        h_d_B = (dataTypes::number **)malloc(d_totalNonlocalElems *
                                             sizeof(dataTypes::numberGPU *));
        h_d_C = (dataTypes::number **)malloc(d_totalNonlocalElems *
                                             sizeof(dataTypes::numberGPU *));

        for (unsigned int i = 0; i < d_totalNonlocalElems; i++)
          {
            h_d_A[i] = thrust::raw_pointer_cast(
              &d_cellWaveFunctionMatrix[d_nonlocalElemIdToLocalElemIdMap[i] *
                                        numberWaveFunctions *
                                        d_numberNodesPerElement]);
            h_d_B[i] = thrust::raw_pointer_cast(
              &d_cellHamiltonianMatrixNonLocalFlattenedDevice
                [i * d_numberNodesPerElement * d_maxSingleAtomPseudoWfc]);
            h_d_C[i] = thrust::raw_pointer_cast(
              &d_projectorKetTimesVectorAllCellsDevice
                [i * numberWaveFunctions * d_maxSingleAtomPseudoWfc]);
          }

        cudaMalloc((void **)&d_A,
                   d_totalNonlocalElems * sizeof(dataTypes::numberGPU *));
        cudaMalloc((void **)&d_B,
                   d_totalNonlocalElems * sizeof(dataTypes::numberGPU *));
        cudaMalloc((void **)&d_C,
                   d_totalNonlocalElems * sizeof(dataTypes::numberGPU *));

        cudaMemcpy(d_A,
                   h_d_A,
                   d_totalNonlocalElems * sizeof(dataTypes::number *),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_B,
                   h_d_B,
                   d_totalNonlocalElems * sizeof(dataTypes::number *),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_C,
                   h_d_C,
                   d_totalNonlocalElems * sizeof(dataTypes::number *),
                   cudaMemcpyHostToDevice);

        d_isMallocCalled = true;
      }

    cudaMemGetInfo(&free_t, &total_t);
    if (dftParameters::verbosity >= 2)
      pcout << "free mem after reinit allocations: " << free_t
            << ", total mem: " << total_t << std::endl;
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::reinitNoRemesh(
    const unsigned int numberWaveFunctions)
  {
    if (dftParameters::isPseudopotential)
      {
        d_cellHamiltonianMatrixNonLocalFlattened.resize(
          d_totalNonlocalElems * d_numberNodesPerElement *
            d_maxSingleAtomPseudoWfc,
          0.0);
        d_cellHamiltonianMatrixNonLocalFlattenedTranspose.resize(
          d_totalNonlocalElems * d_numberNodesPerElement *
            d_maxSingleAtomPseudoWfc,
          0.0);
        d_nonLocalPseudoPotentialConstants.resize(d_totalPseudoWfcNonLocal,
                                                  0.0);

        unsigned int countElem = 0;
        d_numberCellsNonLocalAtoms.resize(d_totalNonlocalAtomsCurrentProc);
        for (unsigned int iAtom = 0; iAtom < d_totalNonlocalAtomsCurrentProc;
             ++iAtom)
          {
            const unsigned int atomId =
              dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
            const unsigned int numberPseudoWaveFunctions =
              dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];


            for (unsigned int ipseudowfc = 0;
                 ipseudowfc < numberPseudoWaveFunctions;
                 ++ipseudowfc)
              {
                const unsigned int id =
                  dftPtr->d_projectorKetTimesVectorPar[0]
                    .get_partitioner()
                    ->global_to_local(
                      dftPtr->d_projectorIdsNumberingMapCurrentProcess
                        [std::make_pair(atomId, ipseudowfc)]);

                d_nonLocalPseudoPotentialConstants[id] =
                  dftPtr
                    ->d_nonLocalPseudoPotentialConstants[atomId][ipseudowfc];
              }

            for (unsigned int iElemComp = 0;
                 iElemComp <
                 dftPtr->d_elementIteratorsInAtomCompactSupport[atomId].size();
                 ++iElemComp)
              {
                for (unsigned int iNode = 0; iNode < d_numberNodesPerElement;
                     ++iNode)
                  {
                    for (unsigned int iPseudoWave = 0;
                         iPseudoWave < numberPseudoWaveFunctions;
                         ++iPseudoWave)
                      {
                        d_cellHamiltonianMatrixNonLocalFlattened
                          [countElem * d_maxSingleAtomPseudoWfc *
                             d_numberNodesPerElement +
                           d_numberNodesPerElement * iPseudoWave + iNode] =
                            dftPtr->d_nonLocalProjectorElementMatrices
                              [atomId][iElemComp]
                              [d_numberNodesPerElement * iPseudoWave + iNode];
                        d_cellHamiltonianMatrixNonLocalFlattenedTranspose
                          [countElem * d_numberNodesPerElement *
                             d_maxSingleAtomPseudoWfc +
                           d_maxSingleAtomPseudoWfc * iNode + iPseudoWave] =
                            dftPtr->d_nonLocalProjectorElementMatricesTranspose
                              [atomId][iElemComp]
                              [numberPseudoWaveFunctions * iNode + iPseudoWave];
                      }
                  }

                countElem++;
              }
          }

        d_cellHamiltonianMatrixNonLocalFlattenedDevice =
          d_cellHamiltonianMatrixNonLocalFlattened;
        d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice =
          d_cellHamiltonianMatrixNonLocalFlattenedTranspose;
        d_nonLocalPseudoPotentialConstantsDevice =
          d_nonLocalPseudoPotentialConstants;
      }
  }


  //
  // compute mass Vector
  //
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::computeMassVector(
    const dealii::DoFHandler<3> &            dofHandler,
    const dealii::AffineConstraints<double> &constraintMatrix,
    distributedCPUVec<double> &              sqrtMassVec,
    distributedCPUVec<double> &              invSqrtMassVec)
  {
    computing_timer.enter_subsection(
      "kohnShamDFTOperatorCUDAClass Mass assembly");
    invSqrtMassVec = 0.0;
    sqrtMassVec    = 0.0;

    QGaussLobatto<3>   quadrature(FEOrder + 1);
    FEValues<3>        fe_values(dofHandler.get_fe(),
                          quadrature,
                          update_values | update_JxW_values);
    const unsigned int dofs_per_cell   = (dofHandler.get_fe()).dofs_per_cell;
    const unsigned int num_quad_points = quadrature.size();
    Vector<double>     massVectorLocal(dofs_per_cell);
    std::vector<dealii::types::global_dof_index> local_dof_indices(
      dofs_per_cell);


    //
    // parallel loop over all elements
    //
    typename DoFHandler<3>::active_cell_iterator cell =
                                                   dofHandler.begin_active(),
                                                 endc = dofHandler.end();
    for (; cell != endc; ++cell)
      if (cell->is_locally_owned())
        {
          // compute values for the current element
          fe_values.reinit(cell);
          massVectorLocal = 0.0;
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
              massVectorLocal(i) += fe_values.shape_value(i, q_point) *
                                    fe_values.shape_value(i, q_point) *
                                    fe_values.JxW(q_point);

          cell->get_dof_indices(local_dof_indices);
          constraintMatrix.distribute_local_to_global(massVectorLocal,
                                                      local_dof_indices,
                                                      invSqrtMassVec);
        }

    invSqrtMassVec.compress(VectorOperation::add);


    for (dealii::types::global_dof_index i = 0; i < invSqrtMassVec.size(); ++i)
      if (invSqrtMassVec.in_local_range(i) &&
          !constraintMatrix.is_constrained(i))
        {
          if (std::abs(invSqrtMassVec(i)) > 1.0e-15)
            {
              sqrtMassVec(i)    = std::sqrt(invSqrtMassVec(i));
              invSqrtMassVec(i) = 1.0 / std::sqrt(invSqrtMassVec(i));
            }
          AssertThrow(
            !std::isnan(invSqrtMassVec(i)),
            ExcMessage(
              "Value of inverse square root of mass matrix on the unconstrained node is undefined"));
        }

    invSqrtMassVec.compress(VectorOperation::insert);
    sqrtMassVec.compress(VectorOperation::insert);

    invSqrtMassVec.update_ghost_values();
    sqrtMassVec.update_ghost_values();

    const unsigned int numberLocalDofs = invSqrtMassVec.local_size();
    const unsigned int numberGhostDofs =
      invSqrtMassVec.get_partitioner()->n_ghost_indices();
    d_invSqrtMassVectorDevice.clear();
    d_sqrtMassVectorDevice.clear();
    d_invSqrtMassVectorDevice.resize(numberLocalDofs + numberGhostDofs);
    d_sqrtMassVectorDevice.resize(numberLocalDofs + numberGhostDofs);

    cudaMemcpy(thrust::raw_pointer_cast(&d_invSqrtMassVectorDevice[0]),
               invSqrtMassVec.begin(),
               (numberLocalDofs + numberGhostDofs) * sizeof(double),
               cudaMemcpyHostToDevice);

    cudaMemcpy(thrust::raw_pointer_cast(&d_sqrtMassVectorDevice[0]),
               sqrtMassVec.begin(),
               (numberLocalDofs + numberGhostDofs) * sizeof(double),
               cudaMemcpyHostToDevice);

    computing_timer.leave_subsection(
      "kohnShamDFTOperatorCUDAClass Mass assembly");
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::reinitkPointSpinIndex(
    const unsigned int kPointIndex,
    const unsigned int spinIndex)
  {
    d_kPointIndex = kPointIndex;
    d_spinIndex   = spinIndex;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::computeVEff(
    const std::map<dealii::CellId, std::vector<double>> *rhoValues,
    const std::map<dealii::CellId, std::vector<double>> &phiValues,
    const std::map<dealii::CellId, std::vector<double>> &externalPotCorrValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const unsigned int externalPotCorrQuadratureId)
  {
    const unsigned int n_cells = dftPtr->matrix_free_data.n_macro_cells();
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();

    const Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    FEValues<3> fe_values(dftPtr->FE, quadrature_formula, update_JxW_values);
    const unsigned int numberQuadraturePoints = quadrature_formula.size();

    d_vEff.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr =
      dftPtr->matrix_free_data.get_dof_handler().begin_active();
    typename dealii::DoFHandler<3>::active_cell_iterator endcPtr =
      dftPtr->matrix_free_data.get_dof_handler().end();
    unsigned int iElemCount = 0;

    std::vector<double> exchangePotentialVal(numberQuadraturePoints);
    std::vector<double> corrPotentialVal(numberQuadraturePoints);
    for (; cellPtr != endcPtr; ++cellPtr)
      if (cellPtr->is_locally_owned())
        {
          fe_values.reinit(cellPtr);

          std::vector<double> densityValue =
            (*rhoValues).find(cellPtr->id())->second;

          if (dftParameters::nonLinearCoreCorrection)
            {
              const std::vector<double> &temp2 =
                rhoCoreValues.find(cellPtr->id())->second;
              for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                densityValue[q] += temp2[q];
            }

          const std::vector<double> &tempPhi =
            phiValues.find(cellPtr->id())->second;

          xc_lda_vxc(&(dftPtr->funcX),
                     numberQuadraturePoints,
                     &densityValue[0],
                     &exchangePotentialVal[0]);
          xc_lda_vxc(&(dftPtr->funcC),
                     numberQuadraturePoints,
                     &densityValue[0],
                     &corrPotentialVal[0]);

          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              d_vEff[iElemCount * numberQuadraturePoints + q] =
                tempPhi[q] + exchangePotentialVal[q] + corrPotentialVal[q];

              d_vEffJxW[iElemCount * numberQuadraturePoints + q] =
                d_vEff[iElemCount * numberQuadraturePoints + q] *
                fe_values.JxW(q);
            }

          iElemCount++;
        }

    d_vEffJxWDevice = d_vEffJxW;
    if ((dftParameters::isPseudopotential ||
         dftParameters::smearedNuclearCharges) &&
        !d_isStiffnessMatrixExternalPotCorrComputed)
      computeVEffExternalPotCorr(externalPotCorrValues,
                                 externalPotCorrQuadratureId);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::computeVEff(
    const std::map<dealii::CellId, std::vector<double>> *rhoValues,
    const std::map<dealii::CellId, std::vector<double>> *gradRhoValues,
    const std::map<dealii::CellId, std::vector<double>> &phiValues,
    const std::map<dealii::CellId, std::vector<double>> &externalPotCorrValues,
    const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
    const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
    const unsigned int externalPotCorrQuadratureId)
  {
    const unsigned int n_cells = dftPtr->matrix_free_data.n_macro_cells();
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();

    const Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    FEValues<3> fe_values(dftPtr->FE, quadrature_formula, update_JxW_values);
    const unsigned int numberQuadraturePoints = quadrature_formula.size();


    d_vEff.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_derExcWithSigmaTimesGradRhoJxW.resize(totalLocallyOwnedCells *
                                              numberQuadraturePoints * 3,
                                            0.0);

    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr =
      dftPtr->matrix_free_data.get_dof_handler().begin_active();
    typename dealii::DoFHandler<3>::active_cell_iterator endcPtr =
      dftPtr->matrix_free_data.get_dof_handler().end();
    unsigned int iElemCount = 0;

    std::vector<double> sigmaValue(numberQuadraturePoints);
    std::vector<double> derExchEnergyWithSigmaVal(numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithSigmaVal(numberQuadraturePoints);
    std::vector<double> derExchEnergyWithDensityVal(numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithDensityVal(numberQuadraturePoints);

    for (; cellPtr != endcPtr; ++cellPtr)
      if (cellPtr->is_locally_owned())
        {
          fe_values.reinit(cellPtr);

          std::vector<double> densityValue =
            (*rhoValues).find(cellPtr->id())->second;
          std::vector<double> gradDensityValue =
            (*gradRhoValues).find(cellPtr->id())->second;

          if (dftParameters::nonLinearCoreCorrection)
            {
              const std::vector<double> &temp2 =
                rhoCoreValues.find(cellPtr->id())->second;
              const std::vector<double> &temp3 =
                gradRhoCoreValues.find(cellPtr->id())->second;
              for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                {
                  densityValue[q] += temp2[q];
                  gradDensityValue[3 * q + 0] += temp3[3 * q + 0];
                  gradDensityValue[3 * q + 1] += temp3[3 * q + 1];
                  gradDensityValue[3 * q + 2] += temp3[3 * q + 2];
                }
            }

          const std::vector<double> &tempPhi =
            phiValues.find(cellPtr->id())->second;

          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              const double gradRhoX = gradDensityValue[3 * q + 0];
              const double gradRhoY = gradDensityValue[3 * q + 1];
              const double gradRhoZ = gradDensityValue[3 * q + 2];
              sigmaValue[q] =
                gradRhoX * gradRhoX + gradRhoY * gradRhoY + gradRhoZ * gradRhoZ;
            }

          xc_gga_vxc(&(dftPtr->funcX),
                     numberQuadraturePoints,
                     &densityValue[0],
                     &sigmaValue[0],
                     &derExchEnergyWithDensityVal[0],
                     &derExchEnergyWithSigmaVal[0]);
          xc_gga_vxc(&(dftPtr->funcC),
                     numberQuadraturePoints,
                     &densityValue[0],
                     &sigmaValue[0],
                     &derCorrEnergyWithDensityVal[0],
                     &derCorrEnergyWithSigmaVal[0]);

          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              const double jxw      = fe_values.JxW(q);
              const double gradRhoX = gradDensityValue[3 * q + 0];
              const double gradRhoY = gradDensityValue[3 * q + 1];
              const double gradRhoZ = gradDensityValue[3 * q + 2];
              const double term =
                derExchEnergyWithSigmaVal[q] + derCorrEnergyWithSigmaVal[q];
              d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                 numberQuadraturePoints * 3 +
                                               3 * q] = term * gradRhoX * jxw;
              d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                 numberQuadraturePoints * 3 +
                                               3 * q + 1] =
                term * gradRhoY * jxw;
              d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                 numberQuadraturePoints * 3 +
                                               3 * q + 2] =
                term * gradRhoZ * jxw;
            }

          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              d_vEff[iElemCount * numberQuadraturePoints + q] =
                tempPhi[q] + derExchEnergyWithDensityVal[q] +
                derCorrEnergyWithDensityVal[q];

              d_vEffJxW[iElemCount * numberQuadraturePoints + q] =
                d_vEff[iElemCount * numberQuadraturePoints + q] *
                fe_values.JxW(q);
            }

          iElemCount++;
        }

    d_vEffJxWDevice                        = d_vEffJxW;
    d_derExcWithSigmaTimesGradRhoJxWDevice = d_derExcWithSigmaTimesGradRhoJxW;

    if ((dftParameters::isPseudopotential ||
         dftParameters::smearedNuclearCharges) &&
        !d_isStiffnessMatrixExternalPotCorrComputed)
      computeVEffExternalPotCorr(externalPotCorrValues,
                                 externalPotCorrQuadratureId);
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    computeVEffSpinPolarized(
      const std::map<dealii::CellId, std::vector<double>> *rhoValues,
      const std::map<dealii::CellId, std::vector<double>> &phiValues,
      const unsigned int                                   spinIndex,
      const std::map<dealii::CellId, std::vector<double>>
        &externalPotCorrValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const unsigned int externalPotCorrQuadratureId)
  {
    const unsigned int n_cells = dftPtr->matrix_free_data.n_macro_cells();
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();

    const Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    FEValues<3> fe_values(dftPtr->FE, quadrature_formula, update_JxW_values);
    const unsigned int numberQuadraturePoints = quadrature_formula.size();

    d_vEff.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr =
      dftPtr->matrix_free_data.get_dof_handler().begin_active();
    typename dealii::DoFHandler<3>::active_cell_iterator endcPtr =
      dftPtr->matrix_free_data.get_dof_handler().end();
    unsigned int iElemCount = 0;

    std::vector<double> exchangePotentialVal(2 * numberQuadraturePoints);
    std::vector<double> corrPotentialVal(2 * numberQuadraturePoints);
    for (; cellPtr != endcPtr; ++cellPtr)
      if (cellPtr->is_locally_owned())
        {
          fe_values.reinit(cellPtr);

          std::vector<double> densityValue =
            (*rhoValues).find(cellPtr->id())->second;
          const std::vector<double> &tempPhi =
            phiValues.find(cellPtr->id())->second;

          if (dftParameters::nonLinearCoreCorrection)
            {
              const std::vector<double> &temp2 =
                rhoCoreValues.find(cellPtr->id())->second;
              for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                {
                  densityValue[2 * q] += temp2[q] / 2.0;
                  densityValue[2 * q + 1] += temp2[q] / 2.0;
                }
            }

          xc_lda_vxc(&(dftPtr->funcX),
                     numberQuadraturePoints,
                     &densityValue[0],
                     &exchangePotentialVal[0]);
          xc_lda_vxc(&(dftPtr->funcC),
                     numberQuadraturePoints,
                     &densityValue[0],
                     &corrPotentialVal[0]);

          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              d_vEff[iElemCount * numberQuadraturePoints + q] =
                tempPhi[q] + exchangePotentialVal[2 * q + spinIndex] +
                corrPotentialVal[2 * q + spinIndex];

              d_vEffJxW[iElemCount * numberQuadraturePoints + q] =
                d_vEff[iElemCount * numberQuadraturePoints + q] *
                fe_values.JxW(q);
            }

          iElemCount++;
        }

    d_vEffJxWDevice = d_vEffJxW;

    if ((dftParameters::isPseudopotential ||
         dftParameters::smearedNuclearCharges) &&
        !d_isStiffnessMatrixExternalPotCorrComputed)
      computeVEffExternalPotCorr(externalPotCorrValues,
                                 externalPotCorrQuadratureId);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    computeVEffSpinPolarized(
      const std::map<dealii::CellId, std::vector<double>> *rhoValues,
      const std::map<dealii::CellId, std::vector<double>> *gradRhoValues,
      const std::map<dealii::CellId, std::vector<double>> &phiValues,
      const unsigned int                                   spinIndex,
      const std::map<dealii::CellId, std::vector<double>>
        &externalPotCorrValues,
      const std::map<dealii::CellId, std::vector<double>> &rhoCoreValues,
      const std::map<dealii::CellId, std::vector<double>> &gradRhoCoreValues,
      const unsigned int externalPotCorrQuadratureId)
  {
    const unsigned int n_cells = dftPtr->matrix_free_data.n_macro_cells();
    const unsigned int totalLocallyOwnedCells =
      dftPtr->matrix_free_data.n_physical_cells();

    const Quadrature<3> &quadrature_formula =
      dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
    FEValues<3> fe_values(dftPtr->FE, quadrature_formula, update_JxW_values);
    const unsigned int numberQuadraturePoints = quadrature_formula.size();

    d_vEff.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_vEffJxW.resize(totalLocallyOwnedCells * numberQuadraturePoints, 0.0);
    d_derExcWithSigmaTimesGradRhoJxW.resize(totalLocallyOwnedCells *
                                              numberQuadraturePoints * 3,
                                            0.0);

    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr =
      dftPtr->matrix_free_data.get_dof_handler().begin_active();
    typename dealii::DoFHandler<3>::active_cell_iterator endcPtr =
      dftPtr->matrix_free_data.get_dof_handler().end();
    unsigned int iElemCount = 0;

    std::vector<double> sigmaValue(3 * numberQuadraturePoints);
    std::vector<double> derExchEnergyWithSigmaVal(3 * numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithSigmaVal(3 * numberQuadraturePoints);
    std::vector<double> derExchEnergyWithDensityVal(2 * numberQuadraturePoints);
    std::vector<double> derCorrEnergyWithDensityVal(2 * numberQuadraturePoints);

    for (; cellPtr != endcPtr; ++cellPtr)
      if (cellPtr->is_locally_owned())
        {
          fe_values.reinit(cellPtr);

          std::vector<double> densityValue =
            (*rhoValues).find(cellPtr->id())->second;
          std::vector<double> gradDensityValue =
            (*gradRhoValues).find(cellPtr->id())->second;
          const std::vector<double> &tempPhi =
            phiValues.find(cellPtr->id())->second;


          if (dftParameters::nonLinearCoreCorrection)
            {
              const std::vector<double> &temp2 =
                rhoCoreValues.find(cellPtr->id())->second;
              const std::vector<double> &temp3 =
                gradRhoCoreValues.find(cellPtr->id())->second;
              for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                {
                  densityValue[2 * q] += temp2[q] / 2.0;
                  densityValue[2 * q + 1] += temp2[q] / 2.0;
                  gradDensityValue[6 * q + 0] += temp3[3 * q + 0] / 2.0;
                  gradDensityValue[6 * q + 1] += temp3[3 * q + 1] / 2.0;
                  gradDensityValue[6 * q + 2] += temp3[3 * q + 2] / 2.0;
                  gradDensityValue[6 * q + 3] += temp3[3 * q + 0] / 2.0;
                  gradDensityValue[6 * q + 4] += temp3[3 * q + 1] / 2.0;
                  gradDensityValue[6 * q + 5] += temp3[3 * q + 2] / 2.0;
                }
            }

          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              double gradRhoX1 = gradDensityValue[6 * q + 0];
              double gradRhoY1 = gradDensityValue[6 * q + 1];
              double gradRhoZ1 = gradDensityValue[6 * q + 2];
              double gradRhoX2 = gradDensityValue[6 * q + 3];
              double gradRhoY2 = gradDensityValue[6 * q + 4];
              double gradRhoZ2 = gradDensityValue[6 * q + 5];
              //
              sigmaValue[3 * q + 0] = gradRhoX1 * gradRhoX1 +
                                      gradRhoY1 * gradRhoY1 +
                                      gradRhoZ1 * gradRhoZ1;
              sigmaValue[3 * q + 1] = gradRhoX1 * gradRhoX2 +
                                      gradRhoY1 * gradRhoY2 +
                                      gradRhoZ1 * gradRhoZ2;
              sigmaValue[3 * q + 2] = gradRhoX2 * gradRhoX2 +
                                      gradRhoY2 * gradRhoY2 +
                                      gradRhoZ2 * gradRhoZ2;
            }

          xc_gga_vxc(&(dftPtr->funcX),
                     numberQuadraturePoints,
                     &densityValue[0],
                     &sigmaValue[0],
                     &derExchEnergyWithDensityVal[0],
                     &derExchEnergyWithSigmaVal[0]);
          xc_gga_vxc(&(dftPtr->funcC),
                     numberQuadraturePoints,
                     &densityValue[0],
                     &sigmaValue[0],
                     &derCorrEnergyWithDensityVal[0],
                     &derCorrEnergyWithSigmaVal[0]);

          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              const double jxw = fe_values.JxW(q);
              const double gradRhoX =
                gradDensityValue[6 * q + 0 + 3 * spinIndex];
              const double gradRhoY =
                gradDensityValue[6 * q + 1 + 3 * spinIndex];
              const double gradRhoZ =
                gradDensityValue[6 * q + 2 + 3 * spinIndex];
              const double gradRhoOtherX =
                gradDensityValue[6 * q + 0 + 3 * (1 - spinIndex)];
              const double gradRhoOtherY =
                gradDensityValue[6 * q + 1 + 3 * (1 - spinIndex)];
              const double gradRhoOtherZ =
                gradDensityValue[6 * q + 2 + 3 * (1 - spinIndex)];
              const double term =
                derExchEnergyWithSigmaVal[3 * q + 2 * spinIndex] +
                derCorrEnergyWithSigmaVal[3 * q + 2 * spinIndex];
              const double termOff = derExchEnergyWithSigmaVal[3 * q + 1] +
                                     derCorrEnergyWithSigmaVal[3 * q + 1];
              d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                 numberQuadraturePoints * 3 +
                                               3 * q] =
                (term * gradRhoX + 0.5 * termOff * gradRhoOtherX) * jxw;
              d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                 numberQuadraturePoints * 3 +
                                               3 * q + 1] =
                (term * gradRhoY + 0.5 * termOff * gradRhoOtherY) * jxw;
              d_derExcWithSigmaTimesGradRhoJxW[iElemCount *
                                                 numberQuadraturePoints * 3 +
                                               3 * q + 2] =
                (term * gradRhoZ + 0.5 * termOff * gradRhoOtherZ) * jxw;
            }

          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            {
              d_vEff[iElemCount * numberQuadraturePoints + q] =
                tempPhi[q] + derExchEnergyWithDensityVal[2 * q + spinIndex] +
                derCorrEnergyWithDensityVal[2 * q + spinIndex];

              d_vEffJxW[iElemCount * numberQuadraturePoints + q] =
                d_vEff[iElemCount * numberQuadraturePoints + q] *
                fe_values.JxW(q);
            }

          iElemCount++;
        }

    d_vEffJxWDevice                        = d_vEffJxW;
    d_derExcWithSigmaTimesGradRhoJxWDevice = d_derExcWithSigmaTimesGradRhoJxW;

    if ((dftParameters::isPseudopotential ||
         dftParameters::smearedNuclearCharges) &&
        !d_isStiffnessMatrixExternalPotCorrComputed)
      computeVEffExternalPotCorr(externalPotCorrValues,
                                 externalPotCorrQuadratureId);
  }

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    computeVEffExternalPotCorr(
      const std::map<dealii::CellId, std::vector<double>>
        &                externalPotCorrValues,
      const unsigned int externalPotCorrQuadratureId)
  {
    d_externalPotCorrQuadratureId = externalPotCorrQuadratureId;
    const unsigned int numberPhysicalCells =
      dftPtr->matrix_free_data.n_physical_cells();
    const int numberQuadraturePoints =
      dftPtr->matrix_free_data.get_quadrature(externalPotCorrQuadratureId)
        .size();
    FEValues<3> feValues(dftPtr->matrix_free_data.get_dof_handler().get_fe(),
                         dftPtr->matrix_free_data.get_quadrature(
                           externalPotCorrQuadratureId),
                         update_JxW_values);
    d_vEffExternalPotCorrJxW.resize(numberPhysicalCells *
                                    numberQuadraturePoints);


    typename dealii::DoFHandler<3>::active_cell_iterator cellPtr =
      dftPtr->matrix_free_data.get_dof_handler().begin_active();
    typename dealii::DoFHandler<3>::active_cell_iterator endcPtr =
      dftPtr->matrix_free_data.get_dof_handler().end();

    unsigned int iElem = 0;
    for (; cellPtr != endcPtr; ++cellPtr)
      if (cellPtr->is_locally_owned())
        {
          feValues.reinit(cellPtr);
          const std::vector<double> &temp =
            externalPotCorrValues.find(cellPtr->id())->second;
          for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
            d_vEffExternalPotCorrJxW[iElem * numberQuadraturePoints + q] =
              temp[q] * feValues.JxW(q);

          iElem++;
        }

    d_vEffExternalPotCorrJxWDevice = d_vEffExternalPotCorrJxW;
  }


  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::HX(
    distributedGPUVec<dataTypes::numberGPU> &    src,
    distributedGPUVec<dataTypes::numberFP32GPU> &tempFloatArray,
    distributedGPUVec<dataTypes::numberGPU> &    projectorKetTimesVector,
    const unsigned int                           localVectorSize,
    const unsigned int                           numberWaveFunctions,
    const bool                                   scaleFlag,
    const double                                 scalar,
    distributedGPUVec<dataTypes::numberGPU> &    dst,
    const bool                                   doUnscalingSrc,
    const bool                                   singlePrecCommun)
  {
    const unsigned int n_ghosts =
      dftPtr->matrix_free_data
        .get_vector_partitioner(dftPtr->d_densityDofHandlerIndex)
        ->n_ghost_indices();
    const unsigned int localSize =
      dftPtr->matrix_free_data
        .get_vector_partitioner(dftPtr->d_densityDofHandlerIndex)
        ->local_size();
    const unsigned int totalSize = localSize + n_ghosts;
    //
    // scale src vector with M^{-1/2}
    //
    scaleCUDAKernel<dataTypes::numberGPU>
      <<<(numberWaveFunctions + 255) / 256 * localVectorSize, 256>>>(
        numberWaveFunctions,
        localVectorSize,
        scalar,
        src.begin(),
        thrust::raw_pointer_cast(&d_invSqrtMassVectorDevice[0]));

    if (scaleFlag)
      {
        scaleCUDAKernel<dataTypes::numberGPU>
          <<<(numberWaveFunctions + 255) / 256 * localVectorSize, 256>>>(
            numberWaveFunctions,
            localVectorSize,
            1.0,
            dst.begin(),
            thrust::raw_pointer_cast(&d_sqrtMassVectorDevice[0]));
      }


    if (singlePrecCommun)
      {
        convDoubleArrToFloatArr<dataTypes::numberGPU, dataTypes::numberFP32GPU>
          <<<(numberWaveFunctions + 255) / 256 * localSize, 256>>>(
            numberWaveFunctions * localSize,
            src.begin(),
            tempFloatArray.begin());
        tempFloatArray.updateGhostValues();

        if (n_ghosts != 0)
          convFloatArrToDoubleArr<dataTypes::numberGPU,
                                  dataTypes::numberFP32GPU>
            <<<(numberWaveFunctions + 255) / 256 * n_ghosts, 256>>>(
              numberWaveFunctions * n_ghosts,
              tempFloatArray.begin() + localSize * numberWaveFunctions,
              src.begin() + localSize * numberWaveFunctions);
      }
    else
      {
        src.updateGhostValues();
      }
    getOverloadedConstraintMatrix()->distribute(src, numberWaveFunctions);

    computeLocalHamiltonianTimesX(src.begin(),
                                  numberWaveFunctions,
                                  dst.begin());

    // H^{nloc}*M^{-1/2}*X
    if (dftParameters::isPseudopotential &&
        dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0)
      {
        computeNonLocalHamiltonianTimesX(src.begin(),
                                         projectorKetTimesVector,
                                         numberWaveFunctions,
                                         dst.begin());
      }

    getOverloadedConstraintMatrix()->distribute_slave_to_master(
      dst, numberWaveFunctions);


    src.zeroOutGhosts();
    if (singlePrecCommun)
      {
        convDoubleArrToFloatArr<dataTypes::numberGPU, dataTypes::numberFP32GPU>
          <<<(numberWaveFunctions + 255) / 256 * totalSize, 256>>>(
            numberWaveFunctions * totalSize,
            dst.begin(),
            tempFloatArray.begin());
        tempFloatArray.compressAdd();

        // copy locally owned processor boundary nodes only to dst vector
        copyFloatArrToDoubleArrLocallyOwned<dataTypes::numberGPU,
                                            dataTypes::numberFP32GPU>
          <<<(numberWaveFunctions + 255) / 256 * localSize, 256>>>(
            numberWaveFunctions,
            localSize,
            tempFloatArray.begin(),
            thrust::raw_pointer_cast(
              &d_locallyOwnedProcBoundaryNodesVectorDevice[0]),
            dst.begin());

        dst.zeroOutGhosts();
      }
    else
      {
        dst.compressAdd();
      }

    //
    // M^{-1/2}*H*M^{-1/2}*X
    //
    scaleCUDAKernel<dataTypes::numberGPU>
      <<<(numberWaveFunctions + 255) / 256 * localVectorSize, 256>>>(
        numberWaveFunctions,
        localVectorSize,
        1.0,
        dst.begin(),
        thrust::raw_pointer_cast(&d_invSqrtMassVectorDevice[0]));


    //
    // unscale src M^{1/2}*X
    //
    if (doUnscalingSrc)
      scaleCUDAKernel<dataTypes::numberGPU>
        <<<(numberWaveFunctions + 255) / 256 * localVectorSize, 256>>>(
          numberWaveFunctions,
          localVectorSize,
          1.0 / scalar,
          src.begin(),
          thrust::raw_pointer_cast(&d_sqrtMassVectorDevice[0]));
  }



  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::HX(
    distributedGPUVec<dataTypes::numberGPU> &src,
    distributedGPUVec<dataTypes::numberGPU> &projectorKetTimesVector,
    const unsigned int                       localVectorSize,
    const unsigned int                       numberWaveFunctions,
    const bool                               scaleFlag,
    const double                             scalar,
    distributedGPUVec<dataTypes::numberGPU> &dst,
    const bool                               doUnscalingSrc)
  {
    const unsigned int n_ghosts =
      dftPtr->matrix_free_data
        .get_vector_partitioner(dftPtr->d_densityDofHandlerIndex)
        ->n_ghost_indices();
    const unsigned int localSize =
      dftPtr->matrix_free_data
        .get_vector_partitioner(dftPtr->d_densityDofHandlerIndex)
        ->local_size();
    const unsigned int totalSize = localSize + n_ghosts;
    //
    // scale src vector with M^{-1/2}
    //
    scaleCUDAKernel<dataTypes::numberGPU>
      <<<(numberWaveFunctions + 255) / 256 * localVectorSize, 256>>>(
        numberWaveFunctions,
        localVectorSize,
        scalar,
        src.begin(),
        thrust::raw_pointer_cast(&d_invSqrtMassVectorDevice[0]));

    if (scaleFlag)
      {
        scaleCUDAKernel<dataTypes::numberGPU>
          <<<(numberWaveFunctions + 255) / 256 * localVectorSize, 256>>>(
            numberWaveFunctions,
            localVectorSize,
            1.0,
            dst.begin(),
            thrust::raw_pointer_cast(&d_sqrtMassVectorDevice[0]));
      }


    src.updateGhostValues();
    getOverloadedConstraintMatrix()->distribute(src, numberWaveFunctions);

    computeLocalHamiltonianTimesX(src.begin(),
                                  numberWaveFunctions,
                                  dst.begin());

    // H^{nloc}*M^{-1/2}*X
    if (dftParameters::isPseudopotential &&
        dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0)
      {
        computeNonLocalHamiltonianTimesX(src.begin(),
                                         projectorKetTimesVector,
                                         numberWaveFunctions,
                                         dst.begin());
      }

    getOverloadedConstraintMatrix()->distribute_slave_to_master(
      dst, numberWaveFunctions);


    src.zeroOutGhosts();
    dst.compressAdd();

    //
    // M^{-1/2}*H*M^{-1/2}*X
    //
    scaleCUDAKernel<dataTypes::numberGPU>
      <<<(numberWaveFunctions + 255) / 256 * localVectorSize, 256>>>(
        numberWaveFunctions,
        localVectorSize,
        1.0,
        dst.begin(),
        thrust::raw_pointer_cast(&d_invSqrtMassVectorDevice[0]));


    //
    // unscale src M^{1/2}*X
    //
    if (doUnscalingSrc)
      scaleCUDAKernel<dataTypes::numberGPU>
        <<<(numberWaveFunctions + 255) / 256 * localVectorSize, 256>>>(
          numberWaveFunctions,
          localVectorSize,
          1.0 / scalar,
          src.begin(),
          thrust::raw_pointer_cast(&d_sqrtMassVectorDevice[0]));
  }


  // computePart1 and computePart2 are flags used by chebyshevFilter function to
  // perform overlap of computation and communication. When either computePart1
  // or computePart1 flags are set to true all communication calls are skipped
  // as they are directly called in chebyshevFilter. Only either of computePart1
  // or computePart2 can be set to true at one time. When computePart1 is set to
  // true distrubute, computeLocalHamiltonianTimesX, and first compute part of
  // nonlocalHX are performed before the control returns back to
  // chebyshevFilter. When computePart2 is set to true, the computations in
  // computePart1 are skipped and only computations performed are: second
  // compute part of nonlocalHX, assembly (only local processor), and
  // distribute_slave_to_master.
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::HXCheby(
    distributedGPUVec<dataTypes::numberGPU> &    src,
    distributedGPUVec<dataTypes::numberFP32GPU> &tempFloatArray,
    distributedGPUVec<dataTypes::numberGPU> &    projectorKetTimesVector,
    const unsigned int                           localVectorSize,
    const unsigned int                           numberWaveFunctions,
    distributedGPUVec<dataTypes::numberGPU> &    dst,
    bool                                         chebMixedPrec,
    bool                                         computePart1,
    bool                                         computePart2)
  {
    const unsigned int n_ghosts =
      dftPtr->matrix_free_data
        .get_vector_partitioner(dftPtr->d_densityDofHandlerIndex)
        ->n_ghost_indices();
    const unsigned int localSize =
      dftPtr->matrix_free_data
        .get_vector_partitioner(dftPtr->d_densityDofHandlerIndex)
        ->local_size();
    const unsigned int totalSize = localSize + n_ghosts;

    if (!(computePart1 || computePart2))
      {
        if (chebMixedPrec)
          {
            convDoubleArrToFloatArr<dataTypes::numberGPU,
                                    dataTypes::numberFP32GPU>
              <<<(numberWaveFunctions + 255) / 256 * localSize, 256>>>(
                numberWaveFunctions * localSize,
                src.begin(),
                tempFloatArray.begin());
            tempFloatArray.updateGhostValues();

            if (n_ghosts != 0)
              convFloatArrToDoubleArr<dataTypes::numberGPU,
                                      dataTypes::numberFP32GPU>
                <<<(numberWaveFunctions + 255) / 256 * n_ghosts, 256>>>(
                  numberWaveFunctions * n_ghosts,
                  tempFloatArray.begin() + localSize * numberWaveFunctions,
                  src.begin() + localSize * numberWaveFunctions);
          }
        else
          {
            src.updateGhostValues();
          }
      }

    if (!computePart2)
      getOverloadedConstraintMatrix()->distribute(src, numberWaveFunctions);


    if (!computePart2)
      computeLocalHamiltonianTimesX(src.begin(),
                                    numberWaveFunctions,
                                    dst.begin());


    // H^{nloc}*M^{-1/2}*X
    if (dftParameters::isPseudopotential &&
        dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0)
      {
        computeNonLocalHamiltonianTimesX(src.begin(),
                                         projectorKetTimesVector,
                                         numberWaveFunctions,
                                         dst.begin(),
                                         computePart2,
                                         computePart1);
      }

    if (computePart1)
      return;


    getOverloadedConstraintMatrix()->distribute_slave_to_master(
      dst, numberWaveFunctions);

    if (computePart2)
      return;

    src.zeroOutGhosts();

    if (chebMixedPrec)
      {
        convDoubleArrToFloatArr<dataTypes::numberGPU, dataTypes::numberFP32GPU>
          <<<(numberWaveFunctions + 255) / 256 * totalSize, 256>>>(
            numberWaveFunctions * totalSize,
            dst.begin(),
            tempFloatArray.begin());
        tempFloatArray.compressAdd();

        // copy locally owned processor boundary nodes only to dst vector
        copyFloatArrToDoubleArrLocallyOwned<dataTypes::numberGPU,
                                            dataTypes::numberFP32GPU>
          <<<(numberWaveFunctions + 255) / 256 * localSize, 256>>>(
            numberWaveFunctions,
            localSize,
            tempFloatArray.begin(),
            thrust::raw_pointer_cast(
              &d_locallyOwnedProcBoundaryNodesVectorDevice[0]),
            dst.begin());

        dst.zeroOutGhosts();
      }
    else
      {
        dst.compressAdd();
      }
  }


  // X^{T}HXc (Xc denotes complex conjugate)
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::XtHX(
    const dataTypes::numberGPU *                     X,
    distributedGPUVec<dataTypes::numberGPU> &        XBlock,
    distributedGPUVec<dataTypes::numberGPU> &        HXBlock,
    distributedGPUVec<dataTypes::numberGPU> &        projectorKetTimesVector,
    const unsigned int                               M,
    const unsigned int                               N,
    cublasHandle_t &                                 handle,
    const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
    dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
    GPUCCLWrapper &                                  gpucclMpiCommDomain)
  {
    std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
    std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
    linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
      processGrid, projHamPar, globalToLocalRowIdMap, globalToLocalColumnIdMap);

    // band group parallelization data structures
    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(dftPtr->interBandGroupComm);
    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(dftPtr->interBandGroupComm,
                                               N,
                                               bandGroupLowHighPlusOneIndices);



    const unsigned int vectorsBlockSize =
      std::min(dftParameters::wfcBlockSize, N);

    dataTypes::number *projHamBlockHost;
    cudaMallocHost((void **)&projHamBlockHost,
                   vectorsBlockSize * N * sizeof(dataTypes::number));
    std::memset(projHamBlockHost,
                0,
                vectorsBlockSize * N * sizeof(dataTypes::number));

    thrust::device_vector<dataTypes::numberThrustGPU> HXBlockFull(
      vectorsBlockSize * M, dataTypes::numberGPU(0.0));
    thrust::device_vector<dataTypes::numberThrustGPU> projHamBlock(
      vectorsBlockSize * N, dataTypes::numberGPU(0.0));

    for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
      {
        // Correct block dimensions if block "goes off edge of" the matrix
        const unsigned int B = std::min(vectorsBlockSize, N - jvec);

        if ((jvec + B) <=
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
            (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
          {
            const unsigned int chebyBlockSize =
              std::min(dftParameters::chebyWfcBlockSize, N);

            for (unsigned int k = jvec; k < jvec + B; k += chebyBlockSize)
              {
                stridedCopyToBlockKernel<dataTypes::numberGPU>
                  <<<(chebyBlockSize + 255) / 256 * M, 256>>>(
                    chebyBlockSize, M, X, N, XBlock.begin(), k);

                // evaluate H times XBlock^{T} and store in HXBlock^{T}
                HXBlock.setZero();
                // thrust::fill(HXBlock.begin(),HXBlock.end(),0.0);
                const bool   scaleFlag = false;
                const double scalar    = 1.0;
                HX(XBlock,
                   projectorKetTimesVector,
                   M,
                   chebyBlockSize,
                   scaleFlag,
                   scalar,
                   HXBlock,
                   false);

                stridedCopyFromBlockKernel<dataTypes::numberGPU>
                  <<<(chebyBlockSize + 255) / 256 * M, 256>>>(
                    chebyBlockSize,
                    M,
                    HXBlock.begin(),
                    B,
                    reinterpret_cast<dataTypes::numberGPU *>(
                      thrust::raw_pointer_cast(&HXBlockFull[0])),
                    k - jvec);
              }

            // Comptute local XTrunc^{T}*HXcBlock.
            const dataTypes::number alpha = dataTypes::number(1.0),
                                    beta  = dataTypes::number(0.0);
            const unsigned int D          = N - jvec;
            cublasXgemm(
              handle,
              CUBLAS_OP_N,
              std::is_same<dataTypes::number, std::complex<double>>::value ?
                CUBLAS_OP_C :
                CUBLAS_OP_T,
              D,
              B,
              M,
              reinterpret_cast<const dataTypes::numberGPU *>(&alpha),
              X + jvec,
              N,
              reinterpret_cast<const dataTypes::numberGPU *>(
                thrust::raw_pointer_cast(&HXBlockFull[0])),
              B,
              reinterpret_cast<const dataTypes::numberGPU *>(&beta),
              reinterpret_cast<dataTypes::numberGPU *>(
                thrust::raw_pointer_cast(&projHamBlock[0])),
              D);

            cudaMemcpy(projHamBlockHost,
                       reinterpret_cast<dataTypes::numberGPU *>(
                         thrust::raw_pointer_cast(&projHamBlock[0])),
                       D * B * sizeof(dataTypes::numberGPU),
                       cudaMemcpyDeviceToHost);


            // Sum local projHamBlock across domain decomposition processors
            MPI_Allreduce(MPI_IN_PLACE,
                          projHamBlockHost,
                          D * B,
                          dataTypes::mpi_type_id(projHamBlockHost),
                          MPI_SUM,
                          mpi_communicator);

            // Copying only the lower triangular part to the ScaLAPACK projected
            // Hamiltonian matrix
            if (processGrid->is_process_active())
              for (unsigned int j = 0; j < B; ++j)
                if (globalToLocalColumnIdMap.find(j + jvec) !=
                    globalToLocalColumnIdMap.end())
                  {
                    const unsigned int localColumnId =
                      globalToLocalColumnIdMap[j + jvec];
                    for (unsigned int i = j + jvec; i < N; ++i)
                      {
                        std::map<unsigned int, unsigned int>::iterator it =
                          globalToLocalRowIdMap.find(i);
                        if (it != globalToLocalRowIdMap.end())
                          projHamPar.local_el(it->second, localColumnId) =
                            projHamBlockHost[j * D + i - jvec];
                      }
                  }

          } // band parallelization
      }

    cudaFreeHost(projHamBlockHost);

    if (numberBandGroups > 1)
      {
        MPI_Barrier(dftPtr->interBandGroupComm);
        linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
          processGrid, projHamPar, dftPtr->interBandGroupComm);
      }
  }

  // X^{T}HXc (Xc denotes complex conjugate) with overlap of computation and
  // communication
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    XtHXOverlapComputeCommun(
      const dataTypes::numberGPU *                     X,
      distributedGPUVec<dataTypes::numberGPU> &        XBlock,
      distributedGPUVec<dataTypes::numberGPU> &        HXBlock,
      distributedGPUVec<dataTypes::numberGPU> &        projectorKetTimesVector,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      GPUCCLWrapper &                                  gpucclMpiCommDomain)
  {
    /////////////PSEUDO CODE for the implementation below for Overlapping
    /// compute and communication/////////////////
    //
    // In the algorithm below the communication and computation of two
    // consecutive blocks of wavefunctions: block i and block i+1 are
    // overlapped.
    // ----------------------------------------------------------
    // CMP denotes computuation of X^{T} times HXBlock
    // COP denotes GPU->CPU copy of X^{T} times HXBlock
    // COM denotes blocking MPI_Allreduce on X^{T}HXBlock and copy to scalapack
    // matrix
    // ----------------------------------------------------------
    // Two CUDA streams are created: compute and copy
    // CMP is performed in compute CUDA stream and COP is performed in copy CUDA
    // stream. COP for a block can only start after the CMP for that block in
    // the compute stream is completed. COM is performed for a block only after
    // COP even for that block is completed.
    //
    // In a blocked loop do:
    // 1) [CMP] Call compute on first block (edge case only for first iteration)
    // 2) Wait for CMP event for current block to be completed.
    // 3) Swap current and next block memory (all iterations except edge case)
    // 4) [COP] Call copy on current block
    // 5) [CMP] Call compute on next block
    // 6) Wait for COP event for current block to be completed
    // 7) [COM] Perform blocking MPI_Allreduce on curent block and copy to
    // scalapack matrix
    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
    std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
    linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
      processGrid, projHamPar, globalToLocalRowIdMap, globalToLocalColumnIdMap);

    // band group parallelization data structures
    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(dftPtr->interBandGroupComm);
    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(dftPtr->interBandGroupComm,
                                               N,
                                               bandGroupLowHighPlusOneIndices);



    const unsigned int vectorsBlockSize =
      std::min(dftParameters::wfcBlockSize, N);
    const unsigned int numberBlocks = N / vectorsBlockSize;

    // create separate CUDA streams for GPU->CPU copy and computation
    cudaStream_t streamCompute, streamDataMove;
    CUDACHECK(cudaStreamCreate(&streamCompute));
    CUDACHECK(cudaStreamCreate(&streamDataMove));

    // attach cublas handle to compute stream
    cublasSetStream(handle, streamCompute);

    // create array of compute and copy events on GPUs
    // for all the blocks. These are required for synchronization
    // between compute, copy and communication as discussed above in the
    // pseudo code
    cudaEvent_t computeEvents[numberBlocks];
    cudaEvent_t copyEvents[numberBlocks];

    for (int i = 0; i < numberBlocks; ++i)
      {
        CUDACHECK(cudaEventCreate(&computeEvents[i]));
        CUDACHECK(cudaEventCreate(&copyEvents[i]));
      }

    dataTypes::number *projHamBlockHost;
    CUDACHECK(cudaMallocHost((void **)&projHamBlockHost,
                             vectorsBlockSize * N * sizeof(dataTypes::number)));
    std::memset(projHamBlockHost,
                0,
                vectorsBlockSize * N * sizeof(dataTypes::number));

    thrust::device_vector<dataTypes::numberThrustGPU> HXBlockFull(
      vectorsBlockSize * M, dataTypes::numberThrustGPU(0.0));
    thrust::device_vector<dataTypes::numberThrustGPU> projHamBlock(
      vectorsBlockSize * N, dataTypes::numberThrustGPU(0.0));
    thrust::device_vector<dataTypes::numberThrustGPU> projHamBlockNext(
      vectorsBlockSize * N, dataTypes::numberThrustGPU(0.0));

    unsigned int blockCount = 0;
    for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
      {
        // Correct block dimensions if block "goes off edge of" the matrix
        const unsigned int B = std::min(vectorsBlockSize, N - jvec);

        if ((jvec + B) <=
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
            (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
          {
            const unsigned int chebyBlockSize =
              std::min(dftParameters::chebyWfcBlockSize, N);

            const dataTypes::number alpha = dataTypes::number(1.0),
                                    beta  = dataTypes::number(0.0);
            const unsigned int D          = N - jvec;

            // handle edge case for the first block or the first block in the
            // band group in case of band parallelization
            if (jvec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
              {
                // compute HXBlockFull in an inner loop over blocks of B
                // wavefunction vectors
                for (unsigned int k = jvec; k < jvec + B; k += chebyBlockSize)
                  {
                    stridedCopyToBlockKernel<dataTypes::numberGPU>
                      <<<(chebyBlockSize + 255) / 256 * M, 256>>>(
                        chebyBlockSize, M, X, N, XBlock.begin(), k);

                    // evaluate H times XBlock^{T} and store in HXBlock^{T}
                    HXBlock.setZero();
                    const bool   scaleFlag = false;
                    const double scalar    = 1.0;
                    HX(XBlock,
                       projectorKetTimesVector,
                       M,
                       chebyBlockSize,
                       scaleFlag,
                       scalar,
                       HXBlock,
                       false);

                    stridedCopyFromBlockKernel<dataTypes::numberGPU>
                      <<<(chebyBlockSize + 255) / 256 * M, 256>>>(
                        chebyBlockSize,
                        M,
                        HXBlock.begin(),
                        B,
                        reinterpret_cast<dataTypes::numberGPU *>(
                          thrust::raw_pointer_cast(&HXBlockFull[0])),
                        k - jvec);
                  }

                // evalute X^{T} times HXBlock
                cublasXgemm(
                  handle,
                  CUBLAS_OP_N,
                  std::is_same<dataTypes::number, std::complex<double>>::value ?
                    CUBLAS_OP_C :
                    CUBLAS_OP_T,
                  D,
                  B,
                  M,
                  reinterpret_cast<const dataTypes::numberGPU *>(&alpha),
                  X + jvec,
                  N,
                  reinterpret_cast<const dataTypes::numberGPU *>(
                    thrust::raw_pointer_cast(&HXBlockFull[0])),
                  B,
                  reinterpret_cast<const dataTypes::numberGPU *>(&beta),
                  reinterpret_cast<dataTypes::numberGPU *>(
                    thrust::raw_pointer_cast(&projHamBlock[0])),
                  D);

                // record completion of compute for first block
                CUDACHECK(
                  cudaEventRecord(computeEvents[blockCount], streamCompute));
              }


            // Before swap host thread needs to wait till compute on
            // currentblock is over. Since swap occurs on the null stream, any
            // future calls in the streamDataMove will only occur after both the
            // compute on currentblock and swap is over. Note that at this point
            // there is nothing queued in the streamDataMove as all previous
            // operations in that stream are over.
            if ((cudaEventSynchronize(computeEvents[blockCount]) ==
                 cudaSuccess) &&
                (jvec > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]))
              projHamBlock.swap(projHamBlockNext);

            const unsigned int jvecNew = jvec + vectorsBlockSize;
            const unsigned int DNew    = N - jvecNew;

            // start computations on the next block
            if (jvecNew <
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
              {
                for (unsigned int k = jvecNew; k < jvecNew + B;
                     k += chebyBlockSize)
                  {
                    stridedCopyToBlockKernel<dataTypes::numberGPU>
                      <<<(chebyBlockSize + 255) / 256 * M, 256>>>(
                        chebyBlockSize, M, X, N, XBlock.begin(), k);

                    // evaluate H times XBlock^{T} and store in HXBlock^{T}
                    HXBlock.setZero();
                    const bool   scaleFlag = false;
                    const double scalar    = 1.0;
                    HX(XBlock,
                       projectorKetTimesVector,
                       M,
                       chebyBlockSize,
                       scaleFlag,
                       scalar,
                       HXBlock,
                       false);

                    stridedCopyFromBlockKernel<dataTypes::numberGPU>
                      <<<(chebyBlockSize + 255) / 256 * M, 256>>>(
                        chebyBlockSize,
                        M,
                        HXBlock.begin(),
                        B,
                        reinterpret_cast<dataTypes::numberGPU *>(
                          thrust::raw_pointer_cast(&HXBlockFull[0])),
                        k - jvecNew);
                  }

                // evalute X^{T} times HXBlock
                cublasXgemm(
                  handle,
                  CUBLAS_OP_N,
                  std::is_same<dataTypes::number, std::complex<double>>::value ?
                    CUBLAS_OP_C :
                    CUBLAS_OP_T,
                  DNew,
                  B,
                  M,
                  reinterpret_cast<const dataTypes::numberGPU *>(&alpha),
                  X + jvecNew,
                  N,
                  reinterpret_cast<const dataTypes::numberGPU *>(
                    thrust::raw_pointer_cast(&HXBlockFull[0])),
                  B,
                  reinterpret_cast<const dataTypes::numberGPU *>(&beta),
                  reinterpret_cast<dataTypes::numberGPU *>(
                    thrust::raw_pointer_cast(&projHamBlockNext[0])),
                  DNew);

                // record completion of compute for next block
                CUDACHECK(cudaEventRecord(computeEvents[blockCount + 1],
                                          streamCompute));
              }

            if (dftParameters::useGPUDirectAllReduce)
              {
                // Sum local projHamBlock across domain decomposition processors
                gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                  reinterpret_cast<dataTypes::numberGPU *>(
                    thrust::raw_pointer_cast(&projHamBlock[0])),
                  reinterpret_cast<dataTypes::numberGPU *>(
                    thrust::raw_pointer_cast(&projHamBlock[0])),
                  D * B,
                  streamDataMove);
              }

            cudaMemcpyAsync(projHamBlockHost,
                            reinterpret_cast<dataTypes::numberGPU *>(
                              thrust::raw_pointer_cast(&projHamBlock[0])),
                            D * B * sizeof(dataTypes::numberGPU),
                            cudaMemcpyDeviceToHost,
                            streamDataMove);

            // record completion of GPU->CPU copy for current block
            CUDACHECK(cudaEventRecord(copyEvents[blockCount], streamDataMove));

            // Check that GPU->CPU on the current block has been completed. If
            // completed, perform blocking MPI commmunication on the current
            // block and copy to ScaLAPACK matrix
            if (cudaEventSynchronize(copyEvents[blockCount]) == cudaSuccess)
              {
                // Sum local projHamBlock across domain decomposition processors
                if (!dftParameters::useGPUDirectAllReduce)
                  MPI_Allreduce(MPI_IN_PLACE,
                                projHamBlockHost,
                                D * B,
                                dataTypes::mpi_type_id(projHamBlockHost),
                                MPI_SUM,
                                mpi_communicator);

                // Copying only the lower triangular part to the ScaLAPACK
                // projected Hamiltonian matrix
                if (processGrid->is_process_active())
                  for (unsigned int j = 0; j < B; ++j)
                    if (globalToLocalColumnIdMap.find(j + jvec) !=
                        globalToLocalColumnIdMap.end())
                      {
                        const unsigned int localColumnId =
                          globalToLocalColumnIdMap[j + jvec];
                        for (unsigned int i = j + jvec; i < N; ++i)
                          {
                            std::map<unsigned int, unsigned int>::iterator it =
                              globalToLocalRowIdMap.find(i);
                            if (it != globalToLocalRowIdMap.end())
                              projHamPar.local_el(it->second, localColumnId) =
                                projHamBlockHost[j * D + i - jvec];
                          }
                      }
              }

          } // band parallelization
        blockCount += 1;
      }

    CUDACHECK(cudaFreeHost(projHamBlockHost));
    // return cublas handle to default stream
    cublasSetStream(handle, NULL);

    for (int i = 0; i < numberBlocks; ++i)
      {
        CUDACHECK(cudaEventDestroy(computeEvents[i]));
        CUDACHECK(cudaEventDestroy(copyEvents[i]));
      }

    CUDACHECK(cudaStreamDestroy(streamCompute));
    CUDACHECK(cudaStreamDestroy(streamDataMove));

    if (numberBandGroups > 1)
      {
        MPI_Barrier(dftPtr->interBandGroupComm);
        linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
          processGrid, projHamPar, dftPtr->interBandGroupComm);
      }
  }


  // X^{T}HXc (Xc denotes complex conjugate)
  /////////////PSEUDO CODE for the implementation below for Overlapping compute
  /// and communication/////////////////
  //
  // In the algorithm below the communication and computation of two consecutive
  // blocks of wavefunctions: block i and block i+1 are overlapped.
  // ----------------------------------------------------------
  // CMP denotes computuation of X^{T} times HXBlock
  // COP denotes GPU->CPU copy of X^{T} times HXBlock
  // COM denotes blocking MPI_Allreduce on X^{T}HXBlock and copy to scalapack
  // matrix
  // ----------------------------------------------------------
  // Two CUDA streams are created: compute and copy
  // CMP is performed in compute CUDA stream and COP is performed in copy CUDA
  // stream. COP for a block can only start after the CMP for that block in the
  // compute stream is completed. COM is performed for a block only after COP
  // even for that block is completed.
  //
  // In a blocked loop do:
  // 1) [CMP] Call compute on first block (edge case only for first iteration)
  // 2) Wait for CMP event for current block to be completed.
  // 3) Swap current and next block memory (all iterations except edge case)
  // 4) [COP] Call copy on current block
  // 5) [CMP] Call compute on next block
  // 6) Wait for COP event for current block to be completed
  // 7) [COM] Perform blocking MPI_Allreduce on curent block and copy to
  // scalapack matrix
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  void
  kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
    XtHXMixedPrecOverlapComputeCommun(
      const dataTypes::numberGPU *                     X,
      distributedGPUVec<dataTypes::numberGPU> &        XBlock,
      distributedGPUVec<dataTypes::numberFP32GPU> &    tempFloatBlock,
      distributedGPUVec<dataTypes::numberGPU> &        HXBlock,
      distributedGPUVec<dataTypes::numberGPU> &        projectorKetTimesVector,
      const unsigned int                               M,
      const unsigned int                               N,
      const unsigned int                               Noc,
      cublasHandle_t &                                 handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      projHamPar,
      GPUCCLWrapper &                                  gpucclMpiCommDomain)
  {
    std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
    std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
    linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
      processGrid, projHamPar, globalToLocalRowIdMap, globalToLocalColumnIdMap);

    // band group parallelization data structures
    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(dftPtr->interBandGroupComm);
    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(dftPtr->interBandGroupComm,
                                               N,
                                               bandGroupLowHighPlusOneIndices);


    const unsigned int vectorsBlockSize =
      std::min(dftParameters::wfcBlockSize, N);

    const unsigned int numberBlocks = N / vectorsBlockSize;

    // create cuda compute and copy streams
    cudaStream_t streamCompute, streamDataMove;
    CUDACHECK(cudaStreamCreate(&streamCompute));
    CUDACHECK(cudaStreamCreate(&streamDataMove));

    // attach cublas handle to compute stream
    cublasSetStream(handle, streamCompute);

    // create array of compute and copy events on GPUs
    // for all the blocks. These are required for synchronization
    // between compute, copy and communication as discussed above in the
    // pseudo code
    cudaEvent_t computeEvents[numberBlocks];
    cudaEvent_t copyEvents[numberBlocks];

    for (int i = 0; i < numberBlocks; ++i)
      {
        CUDACHECK(cudaEventCreate(&computeEvents[i]));
        CUDACHECK(cudaEventCreate(&copyEvents[i]));
      }

    thrust::device_vector<dataTypes::numberFP32ThrustGPU> XSP(
      M * N, dataTypes::numberFP32ThrustGPU(0.0));
    convDoubleArrToFloatArr<dataTypes::numberGPU, dataTypes::numberFP32GPU>
      <<<(N + 255) / 256 * M, 256>>>(
        N * M,
        X,
        reinterpret_cast<dataTypes::numberFP32GPU *>(
          thrust::raw_pointer_cast(&XSP[0])));

    dataTypes::number *projHamBlockHost;
    CUDACHECK(cudaMallocHost((void **)&projHamBlockHost,
                             vectorsBlockSize * N * sizeof(dataTypes::number)));
    std::memset(projHamBlockHost,
                0,
                vectorsBlockSize * N * sizeof(dataTypes::number));

    dataTypes::numberFP32 *projHamBlockHostSP;
    CUDACHECK(
      cudaMallocHost((void **)&projHamBlockHostSP,
                     vectorsBlockSize * N * sizeof(dataTypes::numberFP32)));
    std::memset(projHamBlockHostSP,
                0,
                vectorsBlockSize * N * sizeof(dataTypes::numberFP32));

    thrust::device_vector<dataTypes::numberThrustGPU> HXBlockFull(
      vectorsBlockSize * M, dataTypes::numberThrustGPU(0.0));
    thrust::device_vector<dataTypes::numberFP32ThrustGPU> HXBlockFullSP(
      vectorsBlockSize * M, dataTypes::numberFP32ThrustGPU(0.0));
    thrust::device_vector<dataTypes::numberThrustGPU> projHamBlock(
      vectorsBlockSize * N, dataTypes::numberThrustGPU(0.0));
    thrust::device_vector<dataTypes::numberFP32ThrustGPU> projHamBlockSP(
      vectorsBlockSize * N, dataTypes::numberFP32ThrustGPU(0.0));
    thrust::device_vector<dataTypes::numberThrustGPU> projHamBlockNext(
      vectorsBlockSize * N, dataTypes::numberThrustGPU(0.0));
    thrust::device_vector<dataTypes::numberFP32ThrustGPU> projHamBlockSPNext(
      vectorsBlockSize * N, dataTypes::numberFP32ThrustGPU(0.0));

    unsigned int blockCount = 0;
    for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
      {
        // Correct block dimensions if block "goes off edge of" the matrix
        const unsigned int B = std::min(vectorsBlockSize, N - jvec);

        if ((jvec + B) <=
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
            (jvec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
          {
            const unsigned int chebyBlockSize =
              std::min(dftParameters::chebyWfcBlockSize, N);

            const dataTypes::number alpha       = dataTypes::number(1.0),
                                    beta        = dataTypes::number(0.0);
            const dataTypes::numberFP32 alphaSP = dataTypes::numberFP32(1.0),
                                        betaSP  = dataTypes::numberFP32(0.0);
            const unsigned int D                = N - jvec;

            // handle edge case for the first block or the first block in the
            // band group in case of band parallelization
            if (jvec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
              {
                // compute HXBlockFull or HXBlockFullSP in an inner loop over
                // blocks of B wavefunction vectors
                for (unsigned int k = jvec; k < jvec + B; k += chebyBlockSize)
                  {
                    stridedCopyToBlockKernel<dataTypes::numberGPU>
                      <<<(chebyBlockSize + 255) / 256 * M, 256>>>(
                        chebyBlockSize, M, X, N, XBlock.begin(), k);

                    // evaluate H times XBlock^{T} and store in HXBlock^{T}
                    HXBlock.setZero();
                    const bool   scaleFlag = false;
                    const double scalar    = 1.0;
                    if (jvec + B > Noc)
                      HX(XBlock,
                         projectorKetTimesVector,
                         M,
                         chebyBlockSize,
                         scaleFlag,
                         scalar,
                         HXBlock,
                         false);
                    else
                      HX(XBlock,
                         tempFloatBlock,
                         projectorKetTimesVector,
                         M,
                         chebyBlockSize,
                         scaleFlag,
                         scalar,
                         HXBlock,
                         false,
                         true);

                    if (jvec + B > Noc)
                      stridedCopyFromBlockKernel<dataTypes::numberGPU>
                        <<<(chebyBlockSize + 255) / 256 * M, 256>>>(
                          chebyBlockSize,
                          M,
                          HXBlock.begin(),
                          B,
                          reinterpret_cast<dataTypes::numberGPU *>(
                            thrust::raw_pointer_cast(&HXBlockFull[0])),
                          k - jvec);
                    else
                      stridedCopyFromBlockKernelSP<dataTypes::numberGPU,
                                                   dataTypes::numberFP32GPU>
                        <<<(chebyBlockSize + 255) / 256 * M, 256>>>(
                          chebyBlockSize,
                          M,
                          HXBlock.begin(),
                          B,
                          reinterpret_cast<dataTypes::numberFP32GPU *>(
                            thrust::raw_pointer_cast(&HXBlockFullSP[0])),
                          k - jvec);
                  }

                // evaluate X^{T} times HXBlockFullConj or XSP^{T} times
                // HXBlockFullSPConj
                if (jvec + B > Noc)
                  cublasXgemm(
                    handle,
                    CUBLAS_OP_N,
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      CUBLAS_OP_C :
                      CUBLAS_OP_T,
                    D,
                    B,
                    M,
                    reinterpret_cast<const dataTypes::numberGPU *>(&alpha),
                    X + jvec,
                    N,
                    reinterpret_cast<const dataTypes::numberGPU *>(
                      thrust::raw_pointer_cast(&HXBlockFull[0])),
                    B,
                    reinterpret_cast<const dataTypes::numberGPU *>(&beta),
                    reinterpret_cast<dataTypes::numberGPU *>(
                      thrust::raw_pointer_cast(&projHamBlock[0])),
                    D);
                else
                  cublasXgemm(
                    handle,
                    CUBLAS_OP_N,
                    std::is_same<dataTypes::numberFP32,
                                 std::complex<float>>::value ?
                      CUBLAS_OP_C :
                      CUBLAS_OP_T,
                    D,
                    B,
                    M,
                    reinterpret_cast<const dataTypes::numberFP32GPU *>(
                      &alphaSP),
                    reinterpret_cast<const dataTypes::numberFP32GPU *>(
                      thrust::raw_pointer_cast(&XSP[0])) +
                      jvec,
                    N,
                    reinterpret_cast<const dataTypes::numberFP32GPU *>(
                      thrust::raw_pointer_cast(&HXBlockFullSP[0])),
                    B,
                    reinterpret_cast<const dataTypes::numberFP32GPU *>(&betaSP),
                    reinterpret_cast<dataTypes::numberFP32GPU *>(
                      thrust::raw_pointer_cast(&projHamBlockSP[0])),
                    D);

                // record completion of compute for next block
                CUDACHECK(
                  cudaEventRecord(computeEvents[blockCount], streamCompute));
              }

            // Before swap host thread needs to wait till compute on
            // currentblock is over. Since swap occurs on the null stream, any
            // future calls in the streamDataMove will only occur after both the
            // compute on currentblock and swap is over. Note that at this point
            // there is nothing queued in the streamDataMove as all previous
            // operations in that stream are over.
            if ((cudaEventSynchronize(computeEvents[blockCount]) ==
                 cudaSuccess) &&
                (jvec > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]))
              {
                if (jvec + B > Noc)
                  projHamBlock.swap(projHamBlockNext);
                else
                  projHamBlockSP.swap(projHamBlockSPNext);
              }

            const unsigned int jvecNew = jvec + vectorsBlockSize;
            const unsigned int DNew    = N - jvecNew;

            if (jvecNew <
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
              {
                // compute HXBlockFull or HXBlockFullSP in an inner loop over
                // blocks of B wavefunction vectors
                for (unsigned int k = jvecNew; k < jvecNew + B;
                     k += chebyBlockSize)
                  {
                    stridedCopyToBlockKernel<dataTypes::numberGPU>
                      <<<(chebyBlockSize + 255) / 256 * M, 256>>>(
                        chebyBlockSize, M, X, N, XBlock.begin(), k);

                    // evaluate H times XBlock^{T} and store in HXBlock^{T}
                    HXBlock.setZero();
                    const bool   scaleFlag = false;
                    const double scalar    = 1.0;
                    if (jvecNew + B > Noc)
                      HX(XBlock,
                         projectorKetTimesVector,
                         M,
                         chebyBlockSize,
                         scaleFlag,
                         scalar,
                         HXBlock,
                         false);
                    else
                      HX(XBlock,
                         tempFloatBlock,
                         projectorKetTimesVector,
                         M,
                         chebyBlockSize,
                         scaleFlag,
                         scalar,
                         HXBlock,
                         false,
                         true);

                    if (jvecNew + B > Noc)
                      stridedCopyFromBlockKernel<dataTypes::numberGPU>
                        <<<(chebyBlockSize + 255) / 256 * M, 256>>>(
                          chebyBlockSize,
                          M,
                          HXBlock.begin(),
                          B,
                          reinterpret_cast<dataTypes::numberGPU *>(
                            thrust::raw_pointer_cast(&HXBlockFull[0])),
                          k - jvecNew);
                    else
                      stridedCopyFromBlockKernelSP<dataTypes::numberGPU,
                                                   dataTypes::numberFP32GPU>
                        <<<(chebyBlockSize + 255) / 256 * M, 256>>>(
                          chebyBlockSize,
                          M,
                          HXBlock.begin(),
                          B,
                          reinterpret_cast<dataTypes::numberFP32GPU *>(
                            thrust::raw_pointer_cast(&HXBlockFullSP[0])),
                          k - jvecNew);
                  }

                // evaluate X^{T} times HXBlockFullConj or XSP^{T} times
                // HXBlockFullSPConj
                if (jvecNew + B > Noc)
                  cublasXgemm(
                    handle,
                    CUBLAS_OP_N,
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      CUBLAS_OP_C :
                      CUBLAS_OP_T,
                    DNew,
                    B,
                    M,
                    reinterpret_cast<const dataTypes::numberGPU *>(&alpha),
                    X + jvecNew,
                    N,
                    reinterpret_cast<const dataTypes::numberGPU *>(
                      thrust::raw_pointer_cast(&HXBlockFull[0])),
                    B,
                    reinterpret_cast<const dataTypes::numberGPU *>(&beta),
                    reinterpret_cast<dataTypes::numberGPU *>(
                      thrust::raw_pointer_cast(&projHamBlockNext[0])),
                    DNew);
                else
                  cublasXgemm(
                    handle,
                    CUBLAS_OP_N,
                    std::is_same<dataTypes::numberFP32,
                                 std::complex<float>>::value ?
                      CUBLAS_OP_C :
                      CUBLAS_OP_T,
                    DNew,
                    B,
                    M,
                    reinterpret_cast<const dataTypes::numberFP32GPU *>(
                      &alphaSP),
                    reinterpret_cast<const dataTypes::numberFP32GPU *>(
                      thrust::raw_pointer_cast(&XSP[0])) +
                      jvecNew,
                    N,
                    reinterpret_cast<const dataTypes::numberFP32GPU *>(
                      thrust::raw_pointer_cast(&HXBlockFullSP[0])),
                    B,
                    reinterpret_cast<const dataTypes::numberFP32GPU *>(&betaSP),
                    reinterpret_cast<dataTypes::numberFP32GPU *>(
                      thrust::raw_pointer_cast(&projHamBlockSPNext[0])),
                    DNew);

                // record completion of compute for next block
                CUDACHECK(cudaEventRecord(computeEvents[blockCount + 1],
                                          streamCompute));
              }

            if (dftParameters::useGPUDirectAllReduce)
              {
                if (jvec + B > Noc)
                  {
                    gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&projHamBlock[0])),
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&projHamBlock[0])),
                      D * B,
                      streamDataMove);
                  }
                else
                  {
                    gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                      reinterpret_cast<dataTypes::numberFP32GPU *>(
                        thrust::raw_pointer_cast(&projHamBlockSP[0])),
                      reinterpret_cast<dataTypes::numberFP32GPU *>(
                        thrust::raw_pointer_cast(&projHamBlockSP[0])),
                      D * B,
                      streamDataMove);
                  }
              }

            if (jvec + B > Noc)
              cudaMemcpyAsync(projHamBlockHost,
                              reinterpret_cast<dataTypes::numberGPU *>(
                                thrust::raw_pointer_cast(&projHamBlock[0])),
                              D * B * sizeof(dataTypes::number),
                              cudaMemcpyDeviceToHost,
                              streamDataMove);
            else
              cudaMemcpyAsync(projHamBlockHostSP,
                              reinterpret_cast<dataTypes::numberFP32GPU *>(
                                thrust::raw_pointer_cast(&projHamBlockSP[0])),
                              D * B * sizeof(dataTypes::numberFP32),
                              cudaMemcpyDeviceToHost,
                              streamDataMove);

            // record completion of GPU->CPU copy for current block
            CUDACHECK(cudaEventRecord(copyEvents[blockCount], streamDataMove));

            // Check that GPU->CPU on the current block has been completed. If
            // completed, perform blocking MPI commmunication on the current
            // block and copy to ScaLAPACK matrix
            if (cudaEventSynchronize(copyEvents[blockCount]) == cudaSuccess)
              {
                if (jvec + B > Noc)
                  {
                    // Sum local projHamBlock across domain decomposition
                    // processors
                    if (!dftParameters::useGPUDirectAllReduce)
                      MPI_Allreduce(MPI_IN_PLACE,
                                    projHamBlockHost,
                                    D * B,
                                    dataTypes::mpi_type_id(projHamBlockHost),
                                    MPI_SUM,
                                    mpi_communicator);

                    // Copying only the lower triangular part to the ScaLAPACK
                    // projected Hamiltonian matrix
                    if (processGrid->is_process_active())
                      for (unsigned int j = 0; j < B; ++j)
                        if (globalToLocalColumnIdMap.find(j + jvec) !=
                            globalToLocalColumnIdMap.end())
                          {
                            const unsigned int localColumnId =
                              globalToLocalColumnIdMap[j + jvec];
                            for (unsigned int i = j + jvec; i < N; ++i)
                              {
                                std::map<unsigned int, unsigned int>::iterator
                                  it = globalToLocalRowIdMap.find(i);
                                if (it != globalToLocalRowIdMap.end())
                                  projHamPar.local_el(it->second,
                                                      localColumnId) =
                                    projHamBlockHost[j * D + i - jvec];
                              }
                          }
                  }
                else
                  {
                    // Sum local projHamBlock across domain decomposition
                    // processors
                    if (!dftParameters::useGPUDirectAllReduce)
                      MPI_Allreduce(MPI_IN_PLACE,
                                    projHamBlockHostSP,
                                    D * B,
                                    dataTypes::mpi_type_id(projHamBlockHostSP),
                                    MPI_SUM,
                                    mpi_communicator);

                    // Copying only the lower triangular part to the ScaLAPACK
                    // projected Hamiltonian matrix
                    if (processGrid->is_process_active())
                      for (unsigned int j = 0; j < B; ++j)
                        if (globalToLocalColumnIdMap.find(j + jvec) !=
                            globalToLocalColumnIdMap.end())
                          {
                            const unsigned int localColumnId =
                              globalToLocalColumnIdMap[j + jvec];
                            for (unsigned int i = j + jvec; i < N; ++i)
                              {
                                std::map<unsigned int, unsigned int>::iterator
                                  it = globalToLocalRowIdMap.find(i);
                                if (it != globalToLocalRowIdMap.end())
                                  projHamPar.local_el(it->second,
                                                      localColumnId) =
                                    projHamBlockHostSP[j * D + i - jvec];
                              }
                          }
                  }
              }
          } // band parallelization
        blockCount += 1;
      }

    CUDACHECK(cudaFreeHost(projHamBlockHost));
    CUDACHECK(cudaFreeHost(projHamBlockHostSP));

    // return cublas handle to default stream
    cublasSetStream(handle, NULL);

    for (int i = 0; i < numberBlocks; ++i)
      {
        CUDACHECK(cudaEventDestroy(computeEvents[i]));
        CUDACHECK(cudaEventDestroy(copyEvents[i]));
      }

    CUDACHECK(cudaStreamDestroy(streamCompute));
    CUDACHECK(cudaStreamDestroy(streamDataMove));

    if (numberBandGroups > 1)
      {
        MPI_Barrier(dftPtr->interBandGroupComm);
        linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
          processGrid, projHamPar, dftPtr->interBandGroupComm);
      }
  }

#  include "computeNonLocalHamiltonianTimesXMemoryOptBatchGEMMCUDA.cu"
#  include "hamiltonianMatrixCalculatorFlattenedCUDA.cu"
#  include "inst.cu"
#  include "matrixVectorProductImplementationsCUDA.cu"
#  include "shapeFunctionDataCalculatorCUDA.cu"
} // namespace dftfe
#endif
