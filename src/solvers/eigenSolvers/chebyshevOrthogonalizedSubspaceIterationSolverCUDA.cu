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

#include <chebyshevOrthogonalizedSubspaceIterationSolverCUDA.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <linearAlgebraOperations.h>
#include <linearAlgebraOperationsCUDA.h>
#include <vectorUtilities.h>

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
  {500000, 1000}};

namespace dftfe
{
  namespace
  {
    template <typename NumberType>
    __global__ void
    stridedCopyToBlockKernel(const unsigned int BVec,
                             const unsigned int M,
                             const NumberType * xVec,
                             const unsigned int N,
                             NumberType *       yVec,
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

    template <typename NumberType>
    __global__ void
    stridedCopyFromBlockKernel(const unsigned int BVec,
                               const unsigned int M,
                               const NumberType * xVec,
                               const unsigned int N,
                               NumberType *       yVec,
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


    __global__ void
    scaleCUDAKernel(const unsigned int contiguousBlockSize,
                    const unsigned int numContiguousBlocks,
                    double *           srcArray,
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
            (*(scalingVector + gangBlockId));
        }
    }


    __global__ void
    scaleCUDAKernel(const unsigned int contiguousBlockSize,
                    const unsigned int numContiguousBlocks,
                    cuDoubleComplex *  srcArray,
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
            make_cuDoubleComplex(
              (srcArray + (localThreadId + gangBlockId * contiguousBlockSize))
                  ->x *
                (*(scalingVector + gangBlockId)),
              (srcArray + (localThreadId + gangBlockId * contiguousBlockSize))
                  ->y *
                (*(scalingVector + gangBlockId)));
        }
    }


    __global__ void
    setZeroKernel(const unsigned int BVec,
                  const unsigned int M,
                  const unsigned int N,
                  double *           yVec,
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
          *(yVec + gangBlockId * N + startingXVecId + localThreadId) = 0.0;
        }
    }


    __global__ void
    setZeroKernel(const unsigned int BVec,
                  const unsigned int M,
                  const unsigned int N,
                  cuDoubleComplex *  yVec,
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
            make_cuDoubleComplex(0.0, 0.0);
        }
    }

    namespace internal
    {
      unsigned int
      setChebyshevOrder(const unsigned int d_upperBoundUnWantedSpectrum)
      {
        for (int i = 0; i < sizeof(order_lookup) / sizeof(order_lookup[0]); i++)
          {
            if (d_upperBoundUnWantedSpectrum <= order_lookup[i][0])
              return order_lookup[i][1];
          }
        return 1250;
      }
    } // namespace internal
  }   // namespace

  //
  // Constructor.
  //
  chebyshevOrthogonalizedSubspaceIterationSolverCUDA::
    chebyshevOrthogonalizedSubspaceIterationSolverCUDA(
      const MPI_Comm &mpi_comm_domain,
      double          lowerBoundWantedSpectrum,
      double          lowerBoundUnWantedSpectrum,
      double          upperBoundUnWantedSpectrum)
    : d_lowerBoundWantedSpectrum(lowerBoundWantedSpectrum)
    , d_lowerBoundUnWantedSpectrum(lowerBoundUnWantedSpectrum)
    , d_upperBoundUnWantedSpectrum(upperBoundUnWantedSpectrum)
    , d_isTemporaryParallelVectorsCreated(false)
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
    , computing_timer(mpi_comm_domain,
                      pcout,
                      dftParameters::reproducible_output ||
                          dftParameters::verbosity < 4 ?
                        dealii::TimerOutput::never :
                        dealii::TimerOutput::summary,
                      dealii::TimerOutput::wall_times)
  {}


  //
  // reinitialize spectrum bounds
  //
  void
  chebyshevOrthogonalizedSubspaceIterationSolverCUDA::reinitSpectrumBounds(
    double lowerBoundWantedSpectrum,
    double lowerBoundUnWantedSpectrum,
    double upperBoundUnWantedSpectrum)
  {
    d_lowerBoundWantedSpectrum   = lowerBoundWantedSpectrum;
    d_lowerBoundUnWantedSpectrum = lowerBoundUnWantedSpectrum;
    d_upperBoundUnWantedSpectrum = upperBoundUnWantedSpectrum;
  }


  //
  // solve
  //
  double
  chebyshevOrthogonalizedSubspaceIterationSolverCUDA::solve(
    operatorDFTCUDAClass &operatorMatrix,
    elpaScalaManager &    elpaScala,
    dataTypes::numberGPU *eigenVectorsFlattenedCUDA,
    dataTypes::numberGPU *eigenVectorsRotFracDensityFlattenedCUDA,
    const unsigned int    flattenedSize,
    const unsigned int    totalNumberWaveFunctions,
    std::vector<double> & eigenValues,
    std::vector<double> & residualNorms,
    GPUCCLWrapper &       gpucclMpiCommDomain,
    const MPI_Comm &      interBandGroupComm,
    const bool            isFirstFilteringCall,
    const bool            computeResidual,
    const bool            useMixedPrecOverall,
    const bool            isFirstScf)
  {
    dealii::TimerOutput computingTimerStandard(
      operatorMatrix.getMPICommunicator(),
      pcout,
      dftParameters::reproducible_output || dftParameters::verbosity < 2 ?
        dealii::TimerOutput::never :
        dealii::TimerOutput::every_call,
      dealii::TimerOutput::wall_times);

    cublasHandle_t &cublasHandle = operatorMatrix.getCublasHandle();

    //
    // allocate memory for full flattened array on device and fill it up
    //
    const unsigned int localVectorSize =
      flattenedSize / totalNumberWaveFunctions;

    // band group parallelization data structures
    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);


    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(interBandGroupComm,
                                               totalNumberWaveFunctions,
                                               bandGroupLowHighPlusOneIndices);


    const unsigned int vectorsBlockSize =
      std::min(dftParameters::chebyWfcBlockSize, totalNumberWaveFunctions);

    distributedGPUVec<dataTypes::numberGPU> &cudaFlattenedArrayBlock =
      operatorMatrix.getParallelChebyBlockVectorDevice();

    distributedGPUVec<dataTypes::numberGPU> &projectorKetTimesVector =
      operatorMatrix.getParallelProjectorKetTimesBlockVectorDevice();


    if (isFirstFilteringCall || !d_isTemporaryParallelVectorsCreated)
      {
        d_YArray.reinit(cudaFlattenedArrayBlock);

        d_cudaFlattenedFloatArrayBlock.reinit(
          operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
          vectorsBlockSize);

        if (dftParameters::isPseudopotential)
          {
            if (dftParameters::overlapComputeCommunCheby)
              d_projectorKetTimesVector2.reinit(projectorKetTimesVector);
          }


        if (dftParameters::overlapComputeCommunCheby)
          d_cudaFlattenedArrayBlock2.reinit(cudaFlattenedArrayBlock);


        if (dftParameters::overlapComputeCommunCheby)
          d_YArray2.reinit(d_cudaFlattenedArrayBlock2);


        d_isTemporaryParallelVectorsCreated = true;
      }

    if (isFirstFilteringCall)
      {
        if (dftParameters::gpuFineGrainedTimings)
          {
            cudaDeviceSynchronize();
            computingTimerStandard.enter_subsection("Lanczos upper bound");
          }

        const std::pair<double, double> bounds =
          linearAlgebraOperationsCUDA::lanczosLowerUpperBoundEigenSpectrum(
            operatorMatrix,
            cudaFlattenedArrayBlock,
            d_YArray,
            projectorKetTimesVector,
            vectorsBlockSize);

        if (dftParameters::gpuFineGrainedTimings)
          {
            cudaDeviceSynchronize();
            computingTimerStandard.leave_subsection("Lanczos upper bound");
          }

        d_lowerBoundWantedSpectrum   = bounds.first;
        d_upperBoundUnWantedSpectrum = bounds.second;
        d_lowerBoundUnWantedSpectrum =
          d_lowerBoundWantedSpectrum +
          (d_upperBoundUnWantedSpectrum - d_lowerBoundWantedSpectrum) *
            totalNumberWaveFunctions /
            operatorMatrix.getParallelVecSingleComponent().size() *
            (dftParameters::reproducible_output ? 10.0 : 200.0);
      }
    else if (!dftParameters::reuseLanczosUpperBoundFromFirstCall)
      {
        if (dftParameters::gpuFineGrainedTimings)
          {
            cudaDeviceSynchronize();
            computingTimerStandard.enter_subsection("Lanczos upper bound");
          }

        const std::pair<double, double> bounds =
          linearAlgebraOperationsCUDA::lanczosLowerUpperBoundEigenSpectrum(
            operatorMatrix,
            cudaFlattenedArrayBlock,
            d_YArray,
            projectorKetTimesVector,
            vectorsBlockSize);

        if (dftParameters::gpuFineGrainedTimings)
          {
            cudaDeviceSynchronize();
            computingTimerStandard.leave_subsection("Lanczos upper bound");
          }

        d_upperBoundUnWantedSpectrum = bounds.second;
      }

    if (dftParameters::gpuFineGrainedTimings)
      {
        cudaDeviceSynchronize();
        computingTimerStandard.enter_subsection("Chebyshev filtering on GPU");
      }


    unsigned int chebyshevOrder = dftParameters::chebyshevOrder;

    //
    // set Chebyshev order
    //
    if (chebyshevOrder == 0)
      {
        chebyshevOrder =
          internal::setChebyshevOrder(d_upperBoundUnWantedSpectrum);

        if (dftParameters::orthogType.compare("CGS") == 0 &&
            !dftParameters::isPseudopotential)
          chebyshevOrder *= 0.5;
      }

    chebyshevOrder =
      (isFirstScf && dftParameters::isPseudopotential) ?
        chebyshevOrder *
          dftParameters::chebyshevFilterPolyDegreeFirstScfScalingFactor :
        chebyshevOrder;


    //
    // output statements
    //
    if (dftParameters::verbosity >= 2)
      {
        char buffer[100];

        sprintf(buffer,
                "%s:%18.10e\n",
                "upper bound of unwanted spectrum",
                d_upperBoundUnWantedSpectrum);
        pcout << buffer;
        sprintf(buffer,
                "%s:%18.10e\n",
                "lower bound of unwanted spectrum",
                d_lowerBoundUnWantedSpectrum);
        pcout << buffer;
        sprintf(buffer,
                "%s: %u\n\n",
                "Chebyshev polynomial degree",
                chebyshevOrder);
        pcout << buffer;
      }


    //
    // scale the eigenVectors (initial guess of single atom wavefunctions or
    // previous guess) to convert into Lowden Orthonormalized FE basis
    // multiply by M^{1/2}
    scaleCUDAKernel<<<(totalNumberWaveFunctions + 255) / 256 * localVectorSize,
                      256>>>(totalNumberWaveFunctions,
                             localVectorSize,
                             eigenVectorsFlattenedCUDA,
                             operatorMatrix.getSqrtMassVec());


    // two blocks of wavefunctions are filtered simultaneously when overlap
    // compute communication in chebyshev filtering is toggled on
    const unsigned int numSimultaneousBlocks =
      dftParameters::overlapComputeCommunCheby ? 2 : 1;
    unsigned int       numSimultaneousBlocksCurrent = numSimultaneousBlocks;
    const unsigned int numWfcsInBandGroup =
      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] -
      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId];
    int startIndexBandParal = totalNumberWaveFunctions;
    int numVectorsBandParal = 0;
    for (unsigned int jvec = 0; jvec < totalNumberWaveFunctions;
         jvec += numSimultaneousBlocksCurrent * vectorsBlockSize)
      {
        // Correct block dimensions if block "goes off edge of" the matrix
        const unsigned int BVec =
          vectorsBlockSize; // std::min(vectorsBlockSize,
                            // totalNumberWaveFunctions-jvec);

        // handle edge case when total number of blocks in a given band
        // group is not even in case of overlapping computation and
        // communciation in chebyshev filtering
        const unsigned int leftIndexBandGroupMargin =
          (jvec / numWfcsInBandGroup) * numWfcsInBandGroup;
        numSimultaneousBlocksCurrent =
          ((jvec + numSimultaneousBlocks * BVec - leftIndexBandGroupMargin) <=
             numWfcsInBandGroup &&
           numSimultaneousBlocks == 2) ?
            2 :
            1;

        if ((jvec + numSimultaneousBlocksCurrent * BVec) <=
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
            (jvec + numSimultaneousBlocksCurrent * BVec) >
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
          {
            if (jvec < startIndexBandParal)
              startIndexBandParal = jvec;
            numVectorsBandParal =
              jvec + numSimultaneousBlocksCurrent * BVec - startIndexBandParal;

            // copy from vector containg all wavefunction vectors to current
            // wavefunction vectors block
            stridedCopyToBlockKernel<<<(BVec + 255) / 256 * localVectorSize,
                                       256>>>(BVec,
                                              localVectorSize,
                                              eigenVectorsFlattenedCUDA,
                                              totalNumberWaveFunctions,
                                              cudaFlattenedArrayBlock.begin(),
                                              jvec);

            if (dftParameters::overlapComputeCommunCheby &&
                numSimultaneousBlocksCurrent == 2)
              stridedCopyToBlockKernel<<<(BVec + 255) / 256 * localVectorSize,
                                         256>>>(
                BVec,
                localVectorSize,
                eigenVectorsFlattenedCUDA,
                totalNumberWaveFunctions,
                d_cudaFlattenedArrayBlock2.begin(),
                jvec + BVec);

            //
            // call Chebyshev filtering function only for the current block
            // or two simulataneous blocks (in case of overlap computation
            // and communication) to be filtered and does in-place filtering
            if (dftParameters::overlapComputeCommunCheby &&
                numSimultaneousBlocksCurrent == 2)
              {
                linearAlgebraOperationsCUDA::chebyshevFilter(
                  operatorMatrix,
                  cudaFlattenedArrayBlock,
                  d_YArray,
                  d_cudaFlattenedFloatArrayBlock,
                  projectorKetTimesVector,
                  d_cudaFlattenedArrayBlock2,
                  d_YArray2,
                  d_projectorKetTimesVector2,
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
                linearAlgebraOperationsCUDA::chebyshevFilter(
                  operatorMatrix,
                  cudaFlattenedArrayBlock,
                  d_YArray,
                  d_cudaFlattenedFloatArrayBlock,
                  projectorKetTimesVector,
                  localVectorSize,
                  BVec,
                  chebyshevOrder,
                  d_lowerBoundUnWantedSpectrum,
                  d_upperBoundUnWantedSpectrum,
                  d_lowerBoundWantedSpectrum,
                  useMixedPrecOverall);
              }

            // copy current wavefunction vectors block to vector containing
            // all wavefunction vectors
            stridedCopyFromBlockKernel<<<(BVec + 255) / 256 * localVectorSize,
                                         256>>>(BVec,
                                                localVectorSize,
                                                cudaFlattenedArrayBlock.begin(),
                                                totalNumberWaveFunctions,
                                                eigenVectorsFlattenedCUDA,
                                                jvec);

            if (dftParameters::overlapComputeCommunCheby &&
                numSimultaneousBlocksCurrent == 2)
              stridedCopyFromBlockKernel<<<(BVec + 255) / 256 * localVectorSize,
                                           256>>>(
                BVec,
                localVectorSize,
                d_cudaFlattenedArrayBlock2.begin(),
                totalNumberWaveFunctions,
                eigenVectorsFlattenedCUDA,
                jvec + BVec);
          }
        else
          {
            // set to zero wavefunctions which wont go through chebyshev
            // filtering inside a given band group
            setZeroKernel<<<(numSimultaneousBlocksCurrent * BVec + 255) / 256 *
                              localVectorSize,
                            256>>>(numSimultaneousBlocksCurrent * BVec,
                                   localVectorSize,
                                   totalNumberWaveFunctions,
                                   eigenVectorsFlattenedCUDA,
                                   jvec);
          }

      } // block loop

    if (dftParameters::gpuFineGrainedTimings)
      {
        cudaDeviceSynchronize();
        computingTimerStandard.leave_subsection("Chebyshev filtering on GPU");

        if (dftParameters::verbosity >= 4)
          pcout << "ChebyShev Filtering Done: " << std::endl;
      }


    if (numberBandGroups > 1)
      {
        std::vector<dataTypes::number> eigenVectorsFlattened(
          totalNumberWaveFunctions * localVectorSize, dataTypes::number(0.0));

        cudaMemcpy(reinterpret_cast<dataTypes::numberGPU *>(
                     &eigenVectorsFlattened[0]),
                   eigenVectorsFlattenedCUDA,
                   totalNumberWaveFunctions * localVectorSize *
                     sizeof(dataTypes::numberGPU),
                   cudaMemcpyDeviceToHost);

        MPI_Barrier(interBandGroupComm);


        MPI_Allreduce(MPI_IN_PLACE,
                      &eigenVectorsFlattened[0],
                      totalNumberWaveFunctions * localVectorSize,
                      dataTypes::mpi_type_id(&eigenVectorsFlattened[0]),
                      MPI_SUM,
                      interBandGroupComm);

        MPI_Barrier(interBandGroupComm);

        cudaMemcpy(eigenVectorsFlattenedCUDA,
                   reinterpret_cast<dataTypes::numberGPU *>(
                     &eigenVectorsFlattened[0]),
                   totalNumberWaveFunctions * localVectorSize *
                     sizeof(dataTypes::numberGPU),
                   cudaMemcpyHostToDevice);
      }

    // if (dftParameters::measureOnlyChebyTime)
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
       std::cout<<"l2 norm Chebyshev filtered x:
       "<<std::sqrt(result)<<std::endl;
     */

    if (dftParameters::orthogType.compare("GS") == 0)
      {
        AssertThrow(
          false,
          dealii::ExcMessage(
            "Classical Gram-Schmidt Orthonormalization not implemented in CUDA:"));
      }

    std::fill(eigenValues.begin(), eigenValues.end(), 0.0);

    if (eigenValues.size() != totalNumberWaveFunctions)
      {
        linearAlgebraOperationsCUDA::rayleighRitzGEPSpectrumSplitDirect(
          operatorMatrix,
          elpaScala,
          eigenVectorsFlattenedCUDA,
          eigenVectorsRotFracDensityFlattenedCUDA,
          cudaFlattenedArrayBlock,
          d_cudaFlattenedFloatArrayBlock,
          d_YArray,
          projectorKetTimesVector,
          localVectorSize,
          totalNumberWaveFunctions,
          totalNumberWaveFunctions - eigenValues.size(),
          operatorMatrix.getMPICommunicator(),
          gpucclMpiCommDomain,
          interBandGroupComm,
          eigenValues,
          cublasHandle,
          useMixedPrecOverall);
      }
    else
      {
        linearAlgebraOperationsCUDA::rayleighRitzGEP(
          operatorMatrix,
          elpaScala,
          eigenVectorsFlattenedCUDA,
          cudaFlattenedArrayBlock,
          d_cudaFlattenedFloatArrayBlock,
          d_YArray,
          projectorKetTimesVector,
          localVectorSize,
          totalNumberWaveFunctions,
          operatorMatrix.getMPICommunicator(),
          gpucclMpiCommDomain,
          interBandGroupComm,
          eigenValues,
          cublasHandle,
          useMixedPrecOverall);
      }


    if (computeResidual)
      {
        if (dftParameters::gpuFineGrainedTimings)
          {
            cudaDeviceSynchronize();
            computingTimerStandard.enter_subsection("Residual norm");
          }

        if (eigenValues.size() != totalNumberWaveFunctions)
          linearAlgebraOperationsCUDA::computeEigenResidualNorm(
            operatorMatrix,
            eigenVectorsRotFracDensityFlattenedCUDA,
            cudaFlattenedArrayBlock,
            d_YArray,
            projectorKetTimesVector,
            localVectorSize,
            eigenValues.size(),
            eigenValues,
            operatorMatrix.getMPICommunicator(),
            interBandGroupComm,
            cublasHandle,
            residualNorms);
        else
          linearAlgebraOperationsCUDA::computeEigenResidualNorm(
            operatorMatrix,
            eigenVectorsFlattenedCUDA,
            cudaFlattenedArrayBlock,
            d_YArray,
            projectorKetTimesVector,
            localVectorSize,
            totalNumberWaveFunctions,
            eigenValues,
            operatorMatrix.getMPICommunicator(),
            interBandGroupComm,
            cublasHandle,
            residualNorms,
            true);

        if (dftParameters::gpuFineGrainedTimings)
          {
            cudaDeviceSynchronize();
            computingTimerStandard.leave_subsection("Residual norm");
          }
      }

    //
    // scale the eigenVectors with M^{-1/2} to represent the wavefunctions in
    // the usual FE basis
    //
    scaleCUDAKernel<<<(totalNumberWaveFunctions + 255) / 256 * localVectorSize,
                      256>>>(totalNumberWaveFunctions,
                             localVectorSize,
                             eigenVectorsFlattenedCUDA,
                             operatorMatrix.getInvSqrtMassVec());

    if (eigenValues.size() != totalNumberWaveFunctions)
      scaleCUDAKernel<<<(eigenValues.size() + 255) / 256 * localVectorSize,
                        256>>>(eigenValues.size(),
                               localVectorSize,
                               eigenVectorsRotFracDensityFlattenedCUDA,
                               operatorMatrix.getInvSqrtMassVec());

    return d_upperBoundUnWantedSpectrum;
  }

  //
  // solve
  //
  void
  chebyshevOrthogonalizedSubspaceIterationSolverCUDA::solveNoRR(
    operatorDFTCUDAClass &operatorMatrix,
    elpaScalaManager &    elpaScala,
    dataTypes::numberGPU *eigenVectorsFlattenedCUDA,
    const unsigned int    flattenedSize,
    const unsigned int    totalNumberWaveFunctions,
    std::vector<double> & eigenValues,
    GPUCCLWrapper &       gpucclMpiCommDomain,
    const MPI_Comm &      interBandGroupComm,
    const unsigned int    numberPasses,
    const bool            useMixedPrecOverall)
  {
    cublasHandle_t &cublasHandle = operatorMatrix.getCublasHandle();

    //
    // allocate memory for full flattened array on device and fill it up
    //
    const unsigned int localVectorSize =
      flattenedSize / totalNumberWaveFunctions;


    // band group parallelization data structures
    const unsigned int numberBandGroups =
      dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);


    const unsigned int bandGroupTaskId =
      dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
    std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
    dftUtils::createBandParallelizationIndices(interBandGroupComm,
                                               totalNumberWaveFunctions,
                                               bandGroupLowHighPlusOneIndices);

    const unsigned int wfcBlockSize =
      std::min(dftParameters::wfcBlockSize, totalNumberWaveFunctions);


    const unsigned int chebyBlockSize =
      std::min(dftParameters::chebyWfcBlockSize, totalNumberWaveFunctions);

    distributedGPUVec<dataTypes::numberGPU> &cudaFlattenedArrayBlock =
      operatorMatrix.getParallelChebyBlockVectorDevice();

    distributedGPUVec<dataTypes::numberGPU> &projectorKetTimesVector =
      operatorMatrix.getParallelProjectorKetTimesBlockVectorDevice();


    if (!d_isTemporaryParallelVectorsCreated)
      {
        d_YArray.reinit(cudaFlattenedArrayBlock);

        d_cudaFlattenedFloatArrayBlock.reinit(
          operatorMatrix.getMatrixFreeData()->get_vector_partitioner(),
          chebyBlockSize);


        if (dftParameters::overlapComputeCommunCheby)
          d_cudaFlattenedArrayBlock2.reinit(cudaFlattenedArrayBlock);


        if (dftParameters::overlapComputeCommunCheby)
          d_YArray2.reinit(d_cudaFlattenedArrayBlock2);


        if (dftParameters::overlapComputeCommunCheby)
          d_projectorKetTimesVector2.reinit(projectorKetTimesVector);
      }

    if (!dftParameters::reuseLanczosUpperBoundFromFirstCall)
      {
        const std::pair<double, double> bounds =
          linearAlgebraOperationsCUDA::lanczosLowerUpperBoundEigenSpectrum(
            operatorMatrix,
            cudaFlattenedArrayBlock,
            d_YArray,
            projectorKetTimesVector,
            chebyBlockSize);

        d_upperBoundUnWantedSpectrum = bounds.second;
      }

    unsigned int chebyshevOrder = dftParameters::chebyshevOrder;

    //
    // set Chebyshev order
    //
    if (chebyshevOrder == 0)
      chebyshevOrder =
        internal::setChebyshevOrder(d_upperBoundUnWantedSpectrum);

    chebyshevOrder =
      (dftParameters::isPseudopotential) ?
        chebyshevOrder *
          dftParameters::chebyshevFilterPolyDegreeFirstScfScalingFactor :
        chebyshevOrder;


    //
    // output statements
    //
    if (dftParameters::verbosity >= 2)
      {
        char buffer[100];

        sprintf(buffer,
                "%s:%18.10e\n",
                "upper bound of unwanted spectrum",
                d_upperBoundUnWantedSpectrum);
        pcout << buffer;
        sprintf(buffer,
                "%s:%18.10e\n",
                "lower bound of unwanted spectrum",
                d_lowerBoundUnWantedSpectrum);
        pcout << buffer;
        sprintf(buffer,
                "%s: %u\n\n",
                "Chebyshev polynomial degree",
                chebyshevOrder);
        pcout << buffer;
      }


    //
    // scale the eigenVectors (initial guess of single atom wavefunctions or
    // previous guess) to convert into Lowden Orthonormalized FE basis multiply
    // by M^{1/2}
    scaleCUDAKernel<<<(totalNumberWaveFunctions + 255) / 256 * localVectorSize,
                      256>>>(totalNumberWaveFunctions,
                             localVectorSize,
                             eigenVectorsFlattenedCUDA,
                             operatorMatrix.getSqrtMassVec());

    for (unsigned int ipass = 0; ipass < numberPasses; ipass++)
      {
        pcout << "Beginning no RR Chebyshev filter subpspace iteration pass: "
              << ipass + 1 << std::endl;

        for (unsigned int ivec = 0; ivec < totalNumberWaveFunctions;
             ivec += wfcBlockSize)
          {
            // two blocks of wavefunctions are filtered simultaneously when
            // overlap compute communication in chebyshev filtering is toggled
            // on
            const unsigned int numSimultaneousBlocks =
              dftParameters::overlapComputeCommunCheby ? 2 : 1;
            unsigned int numSimultaneousBlocksCurrent = numSimultaneousBlocks;
            const unsigned int numWfcsInBandGroup =
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] -
              bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId];
            for (unsigned int jvec = ivec; jvec < (ivec + wfcBlockSize);
                 jvec += numSimultaneousBlocksCurrent * chebyBlockSize)
              {
                // Correct block dimensions if block "goes off edge of" the
                // matrix
                const unsigned int BVec =
                  chebyBlockSize; // std::min(vectorsBlockSize,
                                  // totalNumberWaveFunctions-jvec);

                // handle edge case when total number of blocks in a given band
                // group is not even in case of overlapping computation and
                // communciation in chebyshev filtering
                const unsigned int leftIndexBandGroupMargin =
                  (jvec / numWfcsInBandGroup) * numWfcsInBandGroup;
                numSimultaneousBlocksCurrent =
                  ((jvec + numSimultaneousBlocks * BVec -
                    leftIndexBandGroupMargin) <= numWfcsInBandGroup &&
                   numSimultaneousBlocks == 2) ?
                    2 :
                    1;

                if ((jvec + numSimultaneousBlocksCurrent * BVec) <=
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                    (jvec + numSimultaneousBlocksCurrent * BVec) >
                      bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                  {
                    // copy from vector containg all wavefunction vectors to
                    // current wavefunction vectors block
                    stridedCopyToBlockKernel<<<
                      (BVec + 255) / 256 * localVectorSize,
                      256>>>(BVec,
                             localVectorSize,
                             eigenVectorsFlattenedCUDA,
                             totalNumberWaveFunctions,
                             cudaFlattenedArrayBlock.begin(),
                             jvec);

                    if (dftParameters::overlapComputeCommunCheby &&
                        numSimultaneousBlocksCurrent == 2)
                      stridedCopyToBlockKernel<<<
                        (BVec + 255) / 256 * localVectorSize,
                        256>>>(BVec,
                               localVectorSize,
                               eigenVectorsFlattenedCUDA,
                               totalNumberWaveFunctions,
                               d_cudaFlattenedArrayBlock2.begin(),
                               jvec + BVec);

                    //
                    // call Chebyshev filtering function only for the current
                    // block or two simulataneous blocks (in case of overlap
                    // computation and communication) to be filtered and does
                    // in-place filtering
                    if (dftParameters::overlapComputeCommunCheby &&
                        numSimultaneousBlocksCurrent == 2)
                      {
                        linearAlgebraOperationsCUDA::chebyshevFilter(
                          operatorMatrix,
                          cudaFlattenedArrayBlock,
                          d_YArray,
                          d_cudaFlattenedFloatArrayBlock,
                          projectorKetTimesVector,
                          d_cudaFlattenedArrayBlock2,
                          d_YArray2,
                          d_projectorKetTimesVector2,
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
                        linearAlgebraOperationsCUDA::chebyshevFilter(
                          operatorMatrix,
                          cudaFlattenedArrayBlock,
                          d_YArray,
                          d_cudaFlattenedFloatArrayBlock,
                          projectorKetTimesVector,
                          localVectorSize,
                          BVec,
                          chebyshevOrder,
                          d_lowerBoundUnWantedSpectrum,
                          d_upperBoundUnWantedSpectrum,
                          d_lowerBoundWantedSpectrum,
                          useMixedPrecOverall);
                      }

                    // copy current wavefunction vectors block to vector
                    // containing all wavefunction vectors
                    stridedCopyFromBlockKernel<<<
                      (BVec + 255) / 256 * localVectorSize,
                      256>>>(BVec,
                             localVectorSize,
                             cudaFlattenedArrayBlock.begin(),
                             totalNumberWaveFunctions,
                             eigenVectorsFlattenedCUDA,
                             jvec);

                    if (dftParameters::overlapComputeCommunCheby &&
                        numSimultaneousBlocksCurrent == 2)
                      stridedCopyFromBlockKernel<<<
                        (BVec + 255) / 256 * localVectorSize,
                        256>>>(BVec,
                               localVectorSize,
                               d_cudaFlattenedArrayBlock2.begin(),
                               totalNumberWaveFunctions,
                               eigenVectorsFlattenedCUDA,
                               jvec + BVec);
                  }
                else
                  {
                    // set to zero wavefunctions which wont go through chebyshev
                    // filtering inside a given band group
                    setZeroKernel<<<(numSimultaneousBlocksCurrent * BVec +
                                     255) /
                                      256 * localVectorSize,
                                    256>>>(numSimultaneousBlocksCurrent * BVec,
                                           localVectorSize,
                                           totalNumberWaveFunctions,
                                           eigenVectorsFlattenedCUDA,
                                           jvec);
                  }

              } // cheby block loop
          }     // wfc block loop

        if (dftParameters::verbosity >= 4)
          pcout << "ChebyShev Filtering Done: " << std::endl;


        linearAlgebraOperationsCUDA::pseudoGramSchmidtOrthogonalization(
          elpaScala,
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
    // scale the eigenVectors with M^{-1/2} to represent the wavefunctions in
    // the usual FE basis
    //
    scaleCUDAKernel<<<(totalNumberWaveFunctions + 255) / 256 * localVectorSize,
                      256>>>(totalNumberWaveFunctions,
                             localVectorSize,
                             eigenVectorsFlattenedCUDA,
                             operatorMatrix.getInvSqrtMassVec());
  }
} // namespace dftfe
