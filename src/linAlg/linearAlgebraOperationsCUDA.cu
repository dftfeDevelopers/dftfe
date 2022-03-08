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
// @author Sambit Das, Phani Motamarri


#include <cudaHelpers.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <linearAlgebraOperationsCUDA.h>
#include <linearAlgebraOperationsInternal.h>
#include <nvToolsExt.h>
#include <vectorUtilities.h>

namespace dftfe
{
  namespace linearAlgebraOperationsCUDA
  {
    namespace
    {
      __global__ void
      scaleCUDAKernel(const unsigned int contiguousBlockSize,
                      const unsigned int numContiguousBlocks,
                      const double       scalar,
                      double *           srcArray,
                      const double *     scalingVector)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numGangsPerContiguousBlock =
          (contiguousBlockSize + (blockDim.x - 1)) / blockDim.x;
        const unsigned int gangBlockId =
          blockIdx.x / numGangsPerContiguousBlock;
        const unsigned int localThreadId =
          globalThreadId -
          gangBlockId * numGangsPerContiguousBlock * blockDim.x;
        if (globalThreadId <
              numContiguousBlocks * numGangsPerContiguousBlock * blockDim.x &&
            localThreadId < contiguousBlockSize)
          {
            *(srcArray + (localThreadId + gangBlockId * contiguousBlockSize)) =
              *(srcArray +
                (localThreadId + gangBlockId * contiguousBlockSize)) *
              (*(scalingVector + gangBlockId) * scalar);
          }
      }


      __global__ void
      scaleCUDAKernel(const unsigned int contiguousBlockSize,
                      const unsigned int numContiguousBlocks,
                      const double       scalar,
                      cuDoubleComplex *  srcArray,
                      const double *     scalingVector)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numGangsPerContiguousBlock =
          (contiguousBlockSize + (blockDim.x - 1)) / blockDim.x;
        const unsigned int gangBlockId =
          blockIdx.x / numGangsPerContiguousBlock;
        const unsigned int localThreadId =
          globalThreadId -
          gangBlockId * numGangsPerContiguousBlock * blockDim.x;
        if (globalThreadId <
              numContiguousBlocks * numGangsPerContiguousBlock * blockDim.x &&
            localThreadId < contiguousBlockSize)
          {
            *(srcArray + (localThreadId + gangBlockId * contiguousBlockSize)) =
              cuCmul(*(srcArray +
                       (localThreadId + gangBlockId * contiguousBlockSize)),
                     make_cuDoubleComplex(
                       (*(scalingVector + gangBlockId) * scalar), 0.0));
          }
      }


      __global__ void
      convDoubleArrToFloatArr(const unsigned int size,
                              const double *     doubleArr,
                              float *            floatArr)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId; index < size;
             index += blockDim.x * gridDim.x)
          floatArr[index] = doubleArr[index];
      }

      __global__ void
      convDoubleArrToFloatArr(const unsigned int     size,
                              const cuDoubleComplex *doubleArr,
                              cuFloatComplex *       floatArr)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId; index < size;
             index += blockDim.x * gridDim.x)
          floatArr[index] = cuComplexDoubleToFloat(doubleArr[index]);
      }

      // y=a*y, with inc=1
      __global__ void
      dscalCUDAKernel(const int n, double *y, const double a)
      {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x)
          y[i] = a * y[i];
      }

      // y=a*y, with inc=1
      __global__ void
      dscalCUDAKernel(const int n, cuDoubleComplex *Y, const double a)
      {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x)
          Y[i] = make_cuDoubleComplex(a * Y[i].x, a * Y[i].y);
      }

      // y=a*x+b*y, with inc=1
      __global__ void
      daxpyCUDAKernel(const int n, const double *x, double *y, const double a)
      {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x)
          y[i] = a * x[i] + y[i];
      }

      // y=a*x+b*y, with inc=1
      __global__ void
      daxpyCUDAKernel(const int              n,
                      const cuDoubleComplex *X,
                      cuDoubleComplex *      Y,
                      const double           a)
      {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x)
          Y[i] = make_cuDoubleComplex(a * X[i].x + Y[i].x, a * X[i].y + Y[i].y);
      }


      // y=a*x+b*y, with inc=1
      __global__ void
      daxpbyCUDAKernel(const int     n,
                       const double *x,
                       double *      y,
                       const double  a,
                       const double  b)
      {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x)
          y[i] = a * x[i] + b * y[i];
      }

      // y=a*x+b*y, with inc=1
      __global__ void
      daxpbyCUDAKernel(const int              n,
                       const cuDoubleComplex *X,
                       cuDoubleComplex *      Y,
                       const double           a,
                       const double           b)
      {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x)
          Y[i] = make_cuDoubleComplex(a * X[i].x + b * Y[i].x,
                                      a * X[i].y + b * Y[i].y);
      }

      __global__ void
      combinedCUDAKernel(const unsigned int contiguousBlockSize,
                         const unsigned int numContiguousBlocks,
                         double *           x,
                         double *           y,
                         const double       a,
                         const double       b,
                         const double       scalar,
                         const double       scalarOld,
                         const double *     invSqrtMassVec,
                         const double *     sqrtMassVec)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            unsigned int blockIndex = index / contiguousBlockSize;
            *(y + index) *= (*(sqrtMassVec + blockIndex) * 1.0 / scalarOld);
            *(x + index) *= (*(invSqrtMassVec + blockIndex));
            y[index] = a * x[index] + b * y[index];
            *(x + index) *= (*(invSqrtMassVec + blockIndex) * scalar);
            *(y + index) *= (*(sqrtMassVec + blockIndex));
          }
      }


      __global__ void
      combinedCUDAKernel(const unsigned int contiguousBlockSize,
                         const unsigned int numContiguousBlocks,
                         cuDoubleComplex *  X,
                         cuDoubleComplex *  Y,
                         const double       a,
                         const double       b,
                         const double       scalar,
                         const double       scalarOld,
                         const double *     invSqrtMassVec,
                         const double *     sqrtMassVec)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            unsigned int blockIndex = index / contiguousBlockSize;
            *(Y + index)            = make_cuDoubleComplex(
              (Y + index)->x * (*(sqrtMassVec + blockIndex) * 1.0 / scalarOld),
              (Y + index)->y * (*(sqrtMassVec + blockIndex) * 1.0 / scalarOld));
            *(X + index) = make_cuDoubleComplex(
              (X + index)->x * (*(invSqrtMassVec + blockIndex)),
              (X + index)->y * (*(invSqrtMassVec + blockIndex)));
            Y[index]     = make_cuDoubleComplex(a * X[index].x + b * Y[index].x,
                                            a * X[index].y + b * Y[index].y);
            *(X + index) = make_cuDoubleComplex(
              (X + index)->x * (*(invSqrtMassVec + blockIndex) * scalar),
              (X + index)->y * (*(invSqrtMassVec + blockIndex) * scalar));
            *(Y + index) = make_cuDoubleComplex(
              (Y + index)->x * (*(sqrtMassVec + blockIndex)),
              (Y + index)->y * (*(sqrtMassVec + blockIndex)));
          }
      }


      __global__ void
      scaleXArrayRayleighQuotientsCUDAKernel(
        const unsigned int  numVectors,
        const unsigned int  numBoundaryPlusGhostNodes,
        const unsigned int *boundaryGhostIdToLocalIdMap,
        const double *      rayleighQuotients,
        const double *      y,
        const double *      sqrtMassVec,
        double *            x)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries =
          numVectors * numBoundaryPlusGhostNodes;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / numVectors;
            const unsigned int intraBlockIndex = index % numVectors;
            const unsigned int localId =
              boundaryGhostIdToLocalIdMap[blockIndex];
            const unsigned int flattenedWfcId =
              localId * numVectors + intraBlockIndex;
            x[flattenedWfcId] = y[flattenedWfcId] *
                                rayleighQuotients[intraBlockIndex] *
                                sqrtMassVec[localId] * sqrtMassVec[localId];
          }
      }

      __global__ void
      addScaleXArrayRayleighQuotientsCUDAKernel(
        const unsigned int  numVectors,
        const unsigned int  numBoundaryPlusGhostNodes,
        const unsigned int *boundaryGhostIdToLocalIdMap,
        const double *      rayleighQuotients,
        const double *      y,
        const double *      sqrtMassVec,
        double *            x)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries =
          numVectors * numBoundaryPlusGhostNodes;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / numVectors;
            const unsigned int intraBlockIndex = index % numVectors;
            const unsigned int localId =
              boundaryGhostIdToLocalIdMap[blockIndex];
            const unsigned int flattenedWfcId =
              localId * numVectors + intraBlockIndex;
            x[flattenedWfcId] += y[flattenedWfcId] *
                                 rayleighQuotients[intraBlockIndex] *
                                 sqrtMassVec[localId] * sqrtMassVec[localId];
          }
      }

      __global__ void
      addSubspaceRotatedBlockToXKernel(const unsigned int BDof,
                                       const unsigned int BVec,
                                       const float *      rotatedXBlockSP,
                                       double *           X,
                                       const unsigned int startingDofId,
                                       const unsigned int startingVecId,
                                       const unsigned int N)
      {
        const unsigned int numEntries = BVec * BDof;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEntries;
             i += blockDim.x * gridDim.x)
          {
            const unsigned int ibdof = i / BVec;
            const unsigned int ivec  = i % BVec;

            *(X + N * (startingDofId + ibdof) + startingVecId + ivec) +=
              rotatedXBlockSP[ibdof * BVec + ivec];
          }
      }

      __global__ void
      addSubspaceRotatedBlockToXKernel(const unsigned int    BDof,
                                       const unsigned int    BVec,
                                       const cuFloatComplex *rotatedXBlockSP,
                                       cuDoubleComplex *     X,
                                       const unsigned int    startingDofId,
                                       const unsigned int    startingVecId,
                                       const unsigned int    N)
      {
        const unsigned int numEntries = BVec * BDof;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEntries;
             i += blockDim.x * gridDim.x)
          {
            const unsigned int ibdof = i / BVec;
            const unsigned int ivec  = i % BVec;

            *(X + N * (startingDofId + ibdof) + startingVecId + ivec) =
              cuCadd(*(X + N * (startingDofId + ibdof) + startingVecId + ivec),
                     cuComplexFloatToDouble(
                       rotatedXBlockSP[ibdof * BVec + ivec]));
          }
      }


      __global__ void
      computeDiagQTimesXKernel(const double *     diagValues,
                               double *           X,
                               const unsigned int N,
                               const unsigned int M)
      {
        const unsigned int numEntries = N * M;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEntries;
             i += blockDim.x * gridDim.x)
          {
            const unsigned int idof = i / N;
            const unsigned int ivec = i % N;

            *(X + N * idof + ivec) = *(X + N * idof + ivec) * diagValues[ivec];
          }
      }


      __global__ void
      computeDiagQTimesXKernel(const cuDoubleComplex *diagValues,
                               cuDoubleComplex *      X,
                               const unsigned int     N,
                               const unsigned int     M)
      {
        const unsigned int numEntries = N * M;
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEntries;
             i += blockDim.x * gridDim.x)
          {
            const unsigned int idof = i / N;
            const unsigned int ivec = i % N;

            *(X + N * idof + ivec) =
              cuCmul(*(X + N * idof + ivec), diagValues[ivec]);
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
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numGangsPerBVec =
          (BVec + blockDim.x - 1) / blockDim.x;
        const unsigned int gangBlockId = blockIdx.x / numGangsPerBVec;
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
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numGangsPerBVec =
          (BVec + blockDim.x - 1) / blockDim.x;
        const unsigned int gangBlockId = blockIdx.x / numGangsPerBVec;
        const unsigned int localThreadId =
          globalThreadId - gangBlockId * numGangsPerBVec * blockDim.x;

        if (globalThreadId < M * numGangsPerBVec * blockDim.x &&
            localThreadId < BVec)
          {
            *(yVec + gangBlockId * N + startingXVecId + localThreadId) =
              *(xVec + gangBlockId * BVec + localThreadId);
          }
      }

      // R^2=||Y-X*Gamma||^2
      __global__ void
      computeResidualCUDAKernel(const unsigned int numVectors,
                                const unsigned int numDofs,
                                const unsigned int N,
                                const unsigned int startingVecId,
                                const double *     eigenValues,
                                const double *     x,
                                const double *     y,
                                double *           r)
      {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < numVectors * numDofs;
             i += blockDim.x * gridDim.x)
          {
            const unsigned int dofIndex  = i / numVectors;
            const unsigned int waveIndex = i % numVectors;
            r[i] = y[i] - x[dofIndex * N + startingVecId + waveIndex] *
                            eigenValues[startingVecId + waveIndex];
            r[i] = r[i] * r[i];
          }
      }

      // R^2=||Y-X*Gamma||^2
      __global__ void
      computeResidualCUDAKernel(const unsigned int     numVectors,
                                const unsigned int     numDofs,
                                const unsigned int     N,
                                const unsigned int     startingVecId,
                                const double *         eigenValues,
                                const cuDoubleComplex *X,
                                const cuDoubleComplex *Y,
                                double *               r)
      {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x;
             i < numVectors * numDofs;
             i += blockDim.x * gridDim.x)
          {
            const unsigned int    dofIndex  = i / numVectors;
            const unsigned int    waveIndex = i % numVectors;
            const cuDoubleComplex diff      = make_cuDoubleComplex(
              Y[i].x - X[dofIndex * N + startingVecId + waveIndex].x *
                         eigenValues[startingVecId + waveIndex],
              Y[i].y - X[dofIndex * N + startingVecId + waveIndex].y *
                         eigenValues[startingVecId + waveIndex]);
            r[i] = diff.x * diff.x + diff.y * diff.y;
          }
      }

      __global__ void
      convFloatArrToDoubleArr(const unsigned int size,
                              const float *      floatArr,
                              double *           doubleArr)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId; index < size;
             index += blockDim.x * gridDim.x)
          doubleArr[index] = floatArr[index];
      }

      __global__ void
      convFloatArrToDoubleArr(const unsigned int    size,
                              const cuFloatComplex *floatArr,
                              cuDoubleComplex *     doubleArr)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId; index < size;
             index += blockDim.x * gridDim.x)
          doubleArr[index] = cuComplexFloatToDouble(floatArr[index]);
      }

      __global__ void
      copyFloatArrToDoubleArrLocallyOwned(
        const unsigned int  contiguousBlockSize,
        const unsigned int  numContiguousBlocks,
        const float *       floatArr,
        const unsigned int *locallyOwnedFlagArr,
        double *            doubleArr)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
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


      __global__ void
      copyFloatArrToDoubleArrLocallyOwned(
        const unsigned int    contiguousBlockSize,
        const unsigned int    numContiguousBlocks,
        const cuFloatComplex *floatArr,
        const unsigned int *  locallyOwnedFlagArr,
        cuDoubleComplex *     doubleArr)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            unsigned int blockIndex = index / contiguousBlockSize;
            if (locallyOwnedFlagArr[blockIndex] == 1)
              doubleArr[index] = cuComplexFloatToDouble(floatArr[index]);
          }
      }
      __global__ void
      dotProductContributionBlockedKernelMassVector(
        const unsigned int contiguousBlockSize,
        const unsigned int numContiguousBlocks,
        const double *     vec1,
        const double *     vec2,
        const double *     sqrtMassVector,
        double *           vecTemp)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (unsigned int index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex = index / contiguousBlockSize;
            const double       temp       = sqrtMassVector[blockIndex];
            vecTemp[2 * index]            = vec1[index] * vec2[index];
            vecTemp[2 * index + 1] = vec1[index] * vec1[index] * temp * temp;
          }
      }

      void
      computeRayleighQuotients(cublasHandle_t &   handle,
                               const double *     xarray,
                               const double *     yarray,
                               const double *     sqrtMassVector,
                               const double *     onesVec,
                               const unsigned int numberVectors,
                               const unsigned int localSize,
                               const MPI_Comm &   mpiCommDomain,
                               MPI_Request &      request,
                               double *           temparray,
                               double *           dotarrayD,
                               double *           dotarrayH)
      {
        dotProductContributionBlockedKernelMassVector<<<(numberVectors + 255) /
                                                          256 * localSize,
                                                        256>>>(
          numberVectors, localSize, xarray, yarray, sqrtMassVector, temparray);

        const double alpha = 1.0, beta = 0.0;
        cublasDgemm(handle,
                    CUBLAS_OP_N,
                    CUBLAS_OP_T,
                    1,
                    2 * numberVectors,
                    localSize,
                    &alpha,
                    onesVec,
                    1,
                    temparray,
                    2 * numberVectors,
                    &beta,
                    dotarrayD,
                    1);


        cudaMemcpy(dotarrayH,
                   dotarrayD,
                   2 * numberVectors * sizeof(double),
                   cudaMemcpyDeviceToHost);



        MPI_Iallreduce(MPI_IN_PLACE,
                       dotarrayH,
                       2 * numberVectors,
                       MPI_DOUBLE,
                       MPI_SUM,
                       mpiCommDomain,
                       &request);
      }

      void
      checkRayleighQuotients(const unsigned int numberVectors,
                             const double       tolCommun,
                             const double       tolCompute,
                             double *           dotarrayH,
                             double *           rayleighQuotientsH,
                             double *           rayleighQuotientsDiffH,
                             bool &             isConvergedToTol1,
                             bool &             isConvergedToTol2)
      {
        isConvergedToTol1 = true;
        isConvergedToTol2 = true;

        for (unsigned int i = 0; i < numberVectors; ++i)
          {
            const double temp = rayleighQuotientsH[i];
            // std::cout<<"ytdotxarrayH: "<<ytdotxarrayH[i]<<std::endl;
            // std::cout<<"xtdotxarrayH: "<<xtdotxarrayH[i]<<std::endl;
            rayleighQuotientsH[i] = dotarrayH[2 * i] / dotarrayH[2 * i + 1];
            const double diff     = rayleighQuotientsH[i] - temp;
            if (std::fabs(diff) > tolCommun)
              isConvergedToTol1 = false;
            if (std::fabs(diff) > tolCompute)
              isConvergedToTol2 = false;

            // rayleighQuotientsDiffH[i]=diff;
          }
      }

      void
      checkRayleighQuotients(const unsigned int numberVectors,
                             const double       tolCompute,
                             double *           dotarrayH,
                             double *           rayleighQuotientsH,
                             double *           rayleighQuotientsDiffH,
                             bool &             isConvergedToTol)
      {
        isConvergedToTol = true;

        for (unsigned int i = 0; i < numberVectors; ++i)
          {
            const double temp = rayleighQuotientsH[i];
            // std::cout<<"ytdotxarrayH: "<<ytdotxarrayH[i]<<std::endl;
            // std::cout<<"xtdotxarrayH: "<<xtdotxarrayH[i]<<std::endl;
            rayleighQuotientsH[i] = dotarrayH[2 * i] / dotarrayH[2 * i + 1];
            const double diff     = rayleighQuotientsH[i] - temp;
            if (std::fabs(diff) > tolCompute)
              isConvergedToTol = false;

            // rayleighQuotientsDiffH[i]=diff;
          }
      }



    } // namespace


    //
    // evaluate upper bound of the spectrum using k-step Lanczos iteration
    //
    std::pair<double, double>
    lanczosLowerUpperBoundEigenSpectrum(
      operatorDFTCUDAClass &                   operatorMatrix,
      distributedGPUVec<dataTypes::numberGPU> &Xb,
      distributedGPUVec<dataTypes::numberGPU> &Yb,
      distributedGPUVec<dataTypes::numberGPU> &projectorKetTimesVector,
      const unsigned int                       blockSize)
    {
      const unsigned int this_mpi_process =
        dealii::Utilities::MPI::this_mpi_process(
          operatorMatrix.getMPICommunicator());

      const unsigned int lanczosIterations =
        dftParameters::reproducible_output ? 40 : 20;
      double beta;


      dataTypes::number alpha, alphaNeg;

      //
      // generate random vector v
      //
      distributedCPUVec<dataTypes::number> vVector, fVector, v0Vector;
      vVector.reinit(operatorMatrix.getParallelVecSingleComponent());
      fVector.reinit(operatorMatrix.getParallelVecSingleComponent());

      vVector = dataTypes::number(0.0), fVector = dataTypes::number(0.0);
      std::srand(this_mpi_process);
      const unsigned int local_size = vVector.local_size();

      for (unsigned int i = 0; i < local_size; i++)
        vVector.local_element(i) = ((double)std::rand()) / ((double)RAND_MAX);

      operatorMatrix.getOverloadedConstraintMatrixHost()->set_zero(vVector, 1);
      vVector.update_ghost_values();

      //
      // evaluate l2 norm
      //
      vVector /= vVector.l2_norm();
      // vVector.update_ghost_values();

      //
      // call matrix times X
      //
      std::vector<distributedCPUVec<dataTypes::number>> v(1), f(1);
      v[0] = vVector;
      f[0] = fVector;

      distributedCPUVec<dataTypes::number> &vvec = v[0];

      CUDACHECK(cudaMemcpy2D(Xb.begin(),
                             blockSize * sizeof(dataTypes::number),
                             vvec.begin(),
                             1 * sizeof(dataTypes::number),
                             1 * sizeof(dataTypes::number),
                             local_size,
                             cudaMemcpyHostToDevice));

      Yb.setZero();
      operatorMatrix.HX(
        Xb, projectorKetTimesVector, local_size, blockSize, false, 1.0, Yb);

      distributedCPUVec<dataTypes::number> &fvec = f[0];
      CUDACHECK(cudaMemcpy2D(fvec.begin(),
                             1 * sizeof(dataTypes::number),
                             Yb.begin(),
                             blockSize * sizeof(dataTypes::number),
                             1 * sizeof(dataTypes::number),
                             local_size,
                             cudaMemcpyDeviceToHost));

      operatorMatrix.getOverloadedConstraintMatrixHost()->set_zero(v[0], 1);
      fVector = f[0];

      alpha = fVector * vVector;
      fVector.add(-1.0 * alpha, vVector);
      std::vector<dataTypes::number> Tlanczos(lanczosIterations *
                                                lanczosIterations,
                                              0.0);

      Tlanczos[0]    = alpha;
      unsigned index = 0;

      // filling only lower triangular part
      for (unsigned int j = 1; j < lanczosIterations; j++)
        {
          beta     = fVector.l2_norm();
          v0Vector = vVector;
          vVector.equ(1.0 / beta, fVector);
          v[0] = vVector, f[0] = fVector;
          // operatorMatrix.HX(v,f);

          distributedCPUVec<dataTypes::number> &vvec = v[0];
          CUDACHECK(cudaMemcpy2D(Xb.begin(),
                                 blockSize * sizeof(dataTypes::number),
                                 vvec.begin(),
                                 1 * sizeof(dataTypes::number),
                                 1 * sizeof(dataTypes::number),
                                 local_size,
                                 cudaMemcpyHostToDevice));

          Yb.setZero();
          operatorMatrix.HX(
            Xb, projectorKetTimesVector, local_size, blockSize, false, 1.0, Yb);

          distributedCPUVec<dataTypes::number> &fvec = f[0];
          CUDACHECK(cudaMemcpy2D(fvec.begin(),
                                 1 * sizeof(dataTypes::number),
                                 Yb.begin(),
                                 blockSize * sizeof(dataTypes::number),
                                 1 * sizeof(dataTypes::number),
                                 local_size,
                                 cudaMemcpyDeviceToHost));

          operatorMatrix.getOverloadedConstraintMatrixHost()->set_zero(v[0], 1);
          fVector = f[0];
          fVector.add(-1.0 * beta, v0Vector); // beta is real
          alpha = fVector * vVector;
          fVector.add(-1.0 * alpha, vVector);
          index += 1;
          Tlanczos[index] = beta;
          index += lanczosIterations;
          Tlanczos[index] = alpha;
        }

      // eigen decomposition to find max eigen value of T matrix
      std::vector<double> eigenValuesT(lanczosIterations);
      char                jobz = 'N', uplo = 'L';
      const unsigned int  n = lanczosIterations, lda = lanczosIterations;
      int                 info;
      const unsigned int  lwork = 1 + 6 * n + 2 * n * n, liwork = 3 + 5 * n;
      std::vector<int>    iwork(liwork, 0);

#ifdef USE_COMPLEX
      const unsigned int                lrwork = 1 + 5 * n + 2 * n * n;
      std::vector<double>               rwork(lrwork, 0.0);
      std::vector<std::complex<double>> work(lwork);
      zheevd_(&jobz,
              &uplo,
              &n,
              &Tlanczos[0],
              &lda,
              &eigenValuesT[0],
              &work[0],
              &lwork,
              &rwork[0],
              &lrwork,
              &iwork[0],
              &liwork,
              &info);
#else
      std::vector<double> work(lwork, 0.0);
      dsyevd_(&jobz,
              &uplo,
              &n,
              &Tlanczos[0],
              &lda,
              &eigenValuesT[0],
              &work[0],
              &lwork,
              &iwork[0],
              &liwork,
              &info);
#endif


      for (unsigned int i = 0; i < eigenValuesT.size(); i++)
        {
          eigenValuesT[i] = eigenValuesT[i];
        }
      std::sort(eigenValuesT.begin(), eigenValuesT.end());
      //
      const double fvectorNorm = fVector.l2_norm();
      if (dftParameters::verbosity >= 5 && this_mpi_process == 0)
        {
          std::cout << "bUp1: " << eigenValuesT[lanczosIterations - 1]
                    << ", fvector norm: " << fvectorNorm << std::endl;
          std::cout << "aLow: " << eigenValuesT[0] << std::endl;
        }

      double lowerBound = std::floor(eigenValuesT[0]);
      double upperBound =
        std::ceil(eigenValuesT[lanczosIterations - 1] +
                  (dftParameters::reproducible_output ? fvectorNorm :
                                                        fvectorNorm / 10.0));
      return (std::make_pair(lowerBound, upperBound));
    }



    void
    chebyshevFilter(
      operatorDFTCUDAClass &                       operatorMatrix,
      distributedGPUVec<dataTypes::numberGPU> &    XArray,
      distributedGPUVec<dataTypes::numberGPU> &    YArray,
      distributedGPUVec<dataTypes::numberFP32GPU> &tempFloatArray,
      distributedGPUVec<dataTypes::numberGPU> &    projectorKetTimesVector,
      const unsigned int                           localVectorSize,
      const unsigned int                           numberVectors,
      const unsigned int                           m,
      const double                                 a,
      const double                                 b,
      const double                                 a0,
      const bool                                   mixedPrecOverall)
    {
      double e, c, sigma, sigma1, sigma2, gamma, gpu_time;
      e                                  = (b - a) / 2.0;
      c                                  = (b + a) / 2.0;
      sigma                              = e / (a0 - c);
      sigma1                             = sigma;
      gamma                              = 2.0 / sigma1;
      const unsigned int totalVectorSize = localVectorSize * numberVectors;
      int                inc             = 1;

      YArray.setZero();
      //
      // call HX
      //
      bool   scaleFlag = false;
      double scalar    = 1.0;

      operatorMatrix.HX(XArray,
                        projectorKetTimesVector,
                        localVectorSize,
                        numberVectors,
                        scaleFlag,
                        scalar,
                        YArray);

      double alpha1 = sigma1 / e, alpha2 = -c;
      double alpha1Old = alpha1;

      //
      // YArray = YArray + alpha2*XArray and YArray = alpha1*YArray
      //
      /*
      cublasDaxpy(operatorMatrix.getCublasHandle(),
                  totalVectorSize,
                  &alpha2,
                  XArray.begin(),
                  inc,
                  YArray.begin(),
                  inc);
      */

      daxpyCUDAKernel<<<min((totalVectorSize + 255) / 256, 30000), 256>>>(
        totalVectorSize, XArray.begin(), YArray.begin(), alpha2);

      /*
      cublasDscal(operatorMatrix.getCublasHandle(),
                  totalVectorSize,
                  &alpha1,
                  YArray.begin(),
                  inc);
      */

      dscalCUDAKernel<<<min((totalVectorSize + 255) / 256, 30000), 256>>>(
        totalVectorSize, YArray.begin(), alpha1);

      //
      // polynomial loop
      //
      for (unsigned int degree = 2; degree < m + 1; ++degree)
        {
          sigma2 = 1.0 / (gamma - sigma);
          alpha1 = 2.0 * sigma2 / e, alpha2 = -(sigma * sigma2);

          double coeff = -c * alpha1;

          if (degree == 2)
            {
              daxpbyCUDAKernel<<<min((totalVectorSize + 255) / 256, 30000),
                                 256>>>(
                totalVectorSize, YArray.begin(), XArray.begin(), coeff, alpha2);


              // scale src vector with M^{-1/2}
              //
              scaleCUDAKernel<<<(numberVectors + 255) / 256 * localVectorSize,
                                256>>>(numberVectors,
                                       localVectorSize,
                                       alpha1,
                                       YArray.begin(),
                                       operatorMatrix.getInvSqrtMassVec());

              scaleCUDAKernel<<<(numberVectors + 255) / 256 * localVectorSize,
                                256>>>(numberVectors,
                                       localVectorSize,
                                       1.0,
                                       XArray.begin(),
                                       operatorMatrix.getSqrtMassVec());

              //
              // call HX
              //
              operatorMatrix.HXCheby(YArray,
                                     tempFloatArray,
                                     projectorKetTimesVector,
                                     localVectorSize,
                                     numberVectors,
                                     XArray,
                                     mixedPrecOverall &&
                                       dftParameters::useMixedPrecCheby);
            }
          else if (degree == m)
            {
              // unscale src vector with M^{1/2}
              //
              scaleCUDAKernel<<<(numberVectors + 255) / 256 * localVectorSize,
                                256>>>(numberVectors,
                                       localVectorSize,
                                       1.0 / alpha1Old,
                                       XArray.begin(),
                                       operatorMatrix.getSqrtMassVec());

              scaleCUDAKernel<<<(numberVectors + 255) / 256 * localVectorSize,
                                256>>>(numberVectors,
                                       localVectorSize,
                                       1.0,
                                       YArray.begin(),
                                       operatorMatrix.getInvSqrtMassVec());

              daxpbyCUDAKernel<<<min((totalVectorSize + 255) / 256, 30000),
                                 256>>>(
                totalVectorSize, YArray.begin(), XArray.begin(), coeff, alpha2);
              scaleFlag = true;
              //
              // call HX
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
              combinedCUDAKernel<<<min((totalVectorSize + 255) / 256, 30000),
                                   256>>>(numberVectors,
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
              // call HX
              //
              operatorMatrix.HXCheby(YArray,
                                     tempFloatArray,
                                     projectorKetTimesVector,
                                     localVectorSize,
                                     numberVectors,
                                     XArray,
                                     mixedPrecOverall &&
                                       dftParameters::useMixedPrecCheby);
            }

          XArray.swap(YArray);


          sigma     = sigma2;
          alpha1Old = alpha1;
        }

      // copy back YArray to XArray
      cudaMemcpy(XArray.begin(),
                 YArray.begin(),
                 totalVectorSize * sizeof(dataTypes::numberGPU),
                 cudaMemcpyDeviceToDevice);
    }


    //
    // Compute and comunication of two blocks (1) and (2) are overlapped during
    // chebyshev filtering.
    //
    void
    chebyshevFilter(
      operatorDFTCUDAClass &                       operatorMatrix,
      distributedGPUVec<dataTypes::numberGPU> &    XArray1,
      distributedGPUVec<dataTypes::numberGPU> &    YArray1,
      distributedGPUVec<dataTypes::numberFP32GPU> &tempFloatArray,
      distributedGPUVec<dataTypes::numberGPU> &    projectorKetTimesVector1,
      distributedGPUVec<dataTypes::numberGPU> &    XArray2,
      distributedGPUVec<dataTypes::numberGPU> &    YArray2,
      distributedGPUVec<dataTypes::numberGPU> &    projectorKetTimesVector2,
      const unsigned int                           localVectorSize,
      const unsigned int                           numberVectors,
      const unsigned int                           m,
      const double                                 a,
      const double                                 b,
      const double                                 a0,
      const bool                                   mixedPrecOverall)
    {
      double e, c, sigma, sigma1, sigma2, gamma, gpu_time;
      e                                  = (b - a) / 2.0;
      c                                  = (b + a) / 2.0;
      sigma                              = e / (a0 - c);
      sigma1                             = sigma;
      gamma                              = 2.0 / sigma1;
      const unsigned int totalVectorSize = localVectorSize * numberVectors;
      int                inc             = 1;

      YArray1.setZero();
      YArray2.setZero();

      const unsigned int n_ghosts =
        YArray1.ghostFlattenedSize() / numberVectors;
      const unsigned int totalSize = localVectorSize + n_ghosts;

      const unsigned int localSizeNLP =
        projectorKetTimesVector1.locallyOwnedFlattenedSize() / numberVectors;
      const unsigned int n_ghosts_nlp =
        projectorKetTimesVector1.ghostFlattenedSize() / numberVectors;
      const unsigned int totalSizeNLP = localSizeNLP + n_ghosts_nlp;

      //
      // call HX
      //
      bool   scaleFlag = false;
      double scalar    = 1.0;

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

      double alpha1 = sigma1 / e, alpha2 = -c;
      double alpha1Old = alpha1;

      //
      // YArray = YArray + alpha2*XArray and YArray = alpha1*YArray
      //
      /*
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
      */

      daxpyCUDAKernel<<<min((totalVectorSize + 255) / 256, 30000), 256>>>(
        totalVectorSize, XArray1.begin(), YArray1.begin(), alpha2);

      dscalCUDAKernel<<<min((totalVectorSize + 255) / 256, 30000), 256>>>(
        totalVectorSize, YArray1.begin(), alpha1);

      daxpyCUDAKernel<<<min((totalVectorSize + 255) / 256, 30000), 256>>>(
        totalVectorSize, XArray2.begin(), YArray2.begin(), alpha2);

      dscalCUDAKernel<<<min((totalVectorSize + 255) / 256, 30000), 256>>>(
        totalVectorSize, YArray2.begin(), alpha1);

      bool overlap = false;
      //
      // polynomial loop
      //
      for (unsigned int degree = 2; degree < m + 1; ++degree)
        {
          sigma2 = 1.0 / (gamma - sigma);
          alpha1 = 2.0 * sigma2 / e, alpha2 = -(sigma * sigma2);


          double coeff = -c * alpha1;


          if (degree == 2)
            {
              daxpbyCUDAKernel<<<min((totalVectorSize + 255) / 256, 30000),
                                 256>>>(totalVectorSize,
                                        YArray1.begin(),
                                        XArray1.begin(),
                                        coeff,
                                        alpha2);


              // scale src vector with M^{-1/2}
              //
              scaleCUDAKernel<<<(numberVectors + 255) / 256 * localVectorSize,
                                256>>>(numberVectors,
                                       localVectorSize,
                                       alpha1,
                                       YArray1.begin(),
                                       operatorMatrix.getInvSqrtMassVec());

              scaleCUDAKernel<<<(numberVectors + 255) / 256 * localVectorSize,
                                256>>>(numberVectors,
                                       localVectorSize,
                                       1.0,
                                       XArray1.begin(),
                                       operatorMatrix.getSqrtMassVec());

              //
              // call HX
              //
              operatorMatrix.HXCheby(YArray1,
                                     tempFloatArray,
                                     projectorKetTimesVector1,
                                     localVectorSize,
                                     numberVectors,
                                     XArray1,
                                     mixedPrecOverall &&
                                       dftParameters::useMixedPrecCheby);

              daxpbyCUDAKernel<<<min((totalVectorSize + 255) / 256, 30000),
                                 256>>>(totalVectorSize,
                                        YArray2.begin(),
                                        XArray2.begin(),
                                        coeff,
                                        alpha2);


              // scale src vector with M^{-1/2}
              //
              scaleCUDAKernel<<<(numberVectors + 255) / 256 * localVectorSize,
                                256>>>(numberVectors,
                                       localVectorSize,
                                       alpha1,
                                       YArray2.begin(),
                                       operatorMatrix.getInvSqrtMassVec());

              scaleCUDAKernel<<<(numberVectors + 255) / 256 * localVectorSize,
                                256>>>(numberVectors,
                                       localVectorSize,
                                       1.0,
                                       XArray2.begin(),
                                       operatorMatrix.getSqrtMassVec());

              //
              // call HX
              //
              operatorMatrix.HXCheby(YArray2,
                                     tempFloatArray,
                                     projectorKetTimesVector2,
                                     localVectorSize,
                                     numberVectors,
                                     XArray2,
                                     mixedPrecOverall &&
                                       dftParameters::useMixedPrecCheby);
              overlap = false;
            }
          else if (degree == m)
            {
              // unscale src vector with M^{1/2}
              //
              scaleCUDAKernel<<<(numberVectors + 255) / 256 * localVectorSize,
                                256>>>(numberVectors,
                                       localVectorSize,
                                       1.0 / alpha1Old,
                                       XArray1.begin(),
                                       operatorMatrix.getSqrtMassVec());

              scaleCUDAKernel<<<(numberVectors + 255) / 256 * localVectorSize,
                                256>>>(numberVectors,
                                       localVectorSize,
                                       1.0,
                                       YArray1.begin(),
                                       operatorMatrix.getInvSqrtMassVec());

              daxpbyCUDAKernel<<<min((totalVectorSize + 255) / 256, 30000),
                                 256>>>(totalVectorSize,
                                        YArray1.begin(),
                                        XArray1.begin(),
                                        coeff,
                                        alpha2);
              scaleFlag = true;
              //
              // call HX
              //
              operatorMatrix.HX(YArray1,
                                projectorKetTimesVector1,
                                localVectorSize,
                                numberVectors,
                                scaleFlag,
                                alpha1,
                                XArray1);


              // unscale src vector with M^{1/2}
              //
              scaleCUDAKernel<<<(numberVectors + 255) / 256 * localVectorSize,
                                256>>>(numberVectors,
                                       localVectorSize,
                                       1.0 / alpha1Old,
                                       XArray2.begin(),
                                       operatorMatrix.getSqrtMassVec());

              scaleCUDAKernel<<<(numberVectors + 255) / 256 * localVectorSize,
                                256>>>(numberVectors,
                                       localVectorSize,
                                       1.0,
                                       YArray2.begin(),
                                       operatorMatrix.getInvSqrtMassVec());

              daxpbyCUDAKernel<<<min((totalVectorSize + 255) / 256, 30000),
                                 256>>>(totalVectorSize,
                                        YArray2.begin(),
                                        XArray2.begin(),
                                        coeff,
                                        alpha2);
              //
              // call HX
              //
              operatorMatrix.HX(YArray2,
                                projectorKetTimesVector2,
                                localVectorSize,
                                numberVectors,
                                scaleFlag,
                                alpha1,
                                XArray2);
              overlap = false;
            }
          else
            {
              /////////////PSEUDO CODE for the implementation below for
              /// Overlapping compute and communication in HX/////////////////
              //
              // In the algorithm below the communication and computation of two
              // blocks of wavefunctions: block 1 and block 2 are overlapped.
              // CM-NB and CM-B denotes non-blocking and blocking communications
              // respectively. CP denotes compute. The HX computation is divided
              // into compute part 1 and compute part 2 which are separately
              // overlapped. Note that the first and the last iterations of the
              // algorithm are edge cases and are handled a bit differently
              // (Look for step skipped remarks below).
              //
              // 1) [CM-NB] Initiate compress of nonlocal
              // projectorKetTimesVector of block 2 (skipped in first overlap
              // iteration) 2) [CP] Call combinedCUDAKernel of block 1 3) [CM-B]
              // Finish compress of nonlocal projectorKetTimesVector of block 2.
              // (skipped in first overlap iteration) 4) [CM-NB] Call
              // update_ghost_values on nonlocal projectorKetTimesVector of
              // block 2. (skipped in first overlap iteration) 5) [CM-NB]
              // Initiate update_ghost_values on wavefunctions of block 1. 6)
              // [CP] Call HX compute part 2 on block 2. (skipped in first
              // overlap iteration) 7) [CM-B] Finish update_ghost_values on
              // wavefunctions of block 1. 8) [CM-NB] Initiate compress on
              // wavefunctions of block 2. 9) [CP] Call HX compute part 1 on
              // block 1. 10)[CM-B] Finish compress on wavefunctions of block 2.
              // 11)[CM-NB] Initiate compress of nonlocal
              // projectorKetTimesVector of block 1. 12)[CP] Call
              // combinedCUDAKernel of block 2 13)[CM-B] Finish compress of
              // nonlocal projectorKetTimesVector of block 1. 14)[CM-NB]
              // Initiate update_ghost_values on wavefunctions of block 2.
              // 15)[CP] Call HX compute part 2 on block 1.
              // 16)[CM-B] Finish update_ghost_values on wavefunctions of
              // block 2. 17)[CM-NB] Initiate compress on wavefunctions of
              // block 1. 18)[CP] Call HX compute part 1 on block 2. 19)[CM-B]
              // Finish compress on wavefunctions of block 1. 20) Perform
              // chebyshev recursion related swap and scalar operations and go
              // back to step 1)
              //
              // Extra steps for second to last chebyshev filter iteration or
              // the last overlapped iteration: 21) Call compress and
              // update_ghost_values on projectorKetTimesVector of block 2 22)
              // Call HX compute part 2 on block 2. 23) Call compress on
              // wavefunctions of block 2.
              /////////////////////////////////////////////////////////////////////////////////////////////////////////////////

              // overlap flag is used to handle the edge case for the very first
              // overlap is performed, where the previous chebyshev filtering
              // iteration did not use the overlap algorithm
              if (overlap)
                {
                  projectorKetTimesVector2.compressAddStart();
                }

              combinedCUDAKernel<<<min((totalVectorSize + 255) / 256, 30000),
                                   256>>>(numberVectors,
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
                  projectorKetTimesVector2.compressAddFinish();

                  projectorKetTimesVector2.updateGhostValues();
                }

              // unsigned int id2=nvtxRangeStartA("ghost1");
              if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
                {
                  convDoubleArrToFloatArr<<<(numberVectors + 255) / 256 *
                                              localVectorSize,
                                            256>>>(numberVectors *
                                                     localVectorSize,
                                                   YArray1.begin(),
                                                   tempFloatArray.begin());
                  tempFloatArray.updateGhostValuesStart();
                }
              else
                YArray1.updateGhostValuesStart();

              // call compute part 2 of block 2
              if (overlap)
                operatorMatrix.HXCheby(YArray2,
                                       tempFloatArray,
                                       projectorKetTimesVector2,
                                       localVectorSize,
                                       numberVectors,
                                       XArray2,
                                       mixedPrecOverall &&
                                         dftParameters::useMixedPrecCheby,
                                       false,
                                       true);

              if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
                {
                  tempFloatArray.updateGhostValuesFinish();
                  if (n_ghosts != 0)
                    convFloatArrToDoubleArr<<<(numberVectors + 255) / 256 *
                                                n_ghosts,
                                              256>>>(
                      numberVectors * n_ghosts,
                      tempFloatArray.begin() + localVectorSize * numberVectors,
                      YArray1.begin() + localVectorSize * numberVectors);
                }
              else
                YArray1.updateGhostValuesFinish();

              if (overlap)
                YArray2.zeroOutGhosts();
              // nvtxRangeEnd(id2);

              projectorKetTimesVector1.setZero();
              // unsigned int id1=nvtxRangeStartA("compress2");
              if (overlap)
                {
                  if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
                    {
                      convDoubleArrToFloatArr<<<(numberVectors + 255) / 256 *
                                                  totalSize,
                                                256>>>(numberVectors *
                                                         totalSize,
                                                       XArray2.begin(),
                                                       tempFloatArray.begin());
                      tempFloatArray.compressAddStart();
                    }
                  else
                    XArray2.compressAddStart();
                }

              // call compute part 1 of block 1
              operatorMatrix.HXCheby(YArray1,
                                     tempFloatArray,
                                     projectorKetTimesVector1,
                                     localVectorSize,
                                     numberVectors,
                                     XArray1,
                                     mixedPrecOverall &&
                                       dftParameters::useMixedPrecCheby,
                                     true,
                                     false);

              if (overlap)
                {
                  if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
                    {
                      tempFloatArray.compressAddFinish();

                      copyFloatArrToDoubleArrLocallyOwned<<<
                        (numberVectors + 255) / 256 * localVectorSize,
                        256>>>(
                        numberVectors,
                        localVectorSize,
                        tempFloatArray.begin(),
                        thrust::raw_pointer_cast(
                          &operatorMatrix
                             .getLocallyOwnedProcBoundaryNodesVectorDevice()
                               [0]),
                        XArray2.begin());

                      XArray2.zeroOutGhosts();
                    }
                  else
                    XArray2.compressAddFinish();
                  XArray2.swap(YArray2);
                }
              // nvtxRangeEnd(id1);

              projectorKetTimesVector1.compressAddStart();

              combinedCUDAKernel<<<min((totalVectorSize + 255) / 256, 30000),
                                   256>>>(numberVectors,
                                          localVectorSize,
                                          YArray2.begin(),
                                          XArray2.begin(),
                                          coeff,
                                          alpha2,
                                          alpha1,
                                          alpha1Old,
                                          operatorMatrix.getInvSqrtMassVec(),
                                          operatorMatrix.getSqrtMassVec());

              projectorKetTimesVector1.compressAddFinish();

              projectorKetTimesVector1.updateGhostValues();

              // unsigned int id3=nvtxRangeStartA("ghost2");
              if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
                {
                  convDoubleArrToFloatArr<<<(numberVectors + 255) / 256 *
                                              localVectorSize,
                                            256>>>(numberVectors *
                                                     localVectorSize,
                                                   YArray2.begin(),
                                                   tempFloatArray.begin());
                  tempFloatArray.updateGhostValuesStart();
                }
              else
                YArray2.updateGhostValuesStart();

              // call compute part 2 of block 1
              operatorMatrix.HXCheby(YArray1,
                                     tempFloatArray,
                                     projectorKetTimesVector1,
                                     localVectorSize,
                                     numberVectors,
                                     XArray1,
                                     mixedPrecOverall &&
                                       dftParameters::useMixedPrecCheby,
                                     false,
                                     true);

              if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
                {
                  tempFloatArray.updateGhostValuesFinish();
                  if (n_ghosts != 0)
                    convFloatArrToDoubleArr<<<(numberVectors + 255) / 256 *
                                                n_ghosts,
                                              256>>>(
                      numberVectors * n_ghosts,
                      tempFloatArray.begin() + localVectorSize * numberVectors,
                      YArray2.begin() + localVectorSize * numberVectors);
                }
              else
                YArray2.updateGhostValuesFinish();
              YArray1.zeroOutGhosts();
              // nvtxRangeEnd(id3);


              projectorKetTimesVector2.setZero();

              // unsigned int id4=nvtxRangeStartA("compress1");
              if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
                {
                  convDoubleArrToFloatArr<<<(numberVectors + 255) / 256 *
                                              totalSize,
                                            256>>>(numberVectors * totalSize,
                                                   XArray1.begin(),
                                                   tempFloatArray.begin());
                  tempFloatArray.compressAddStart();
                }
              else
                XArray1.compressAddStart();

              // call compute part 1 of block 2
              operatorMatrix.HXCheby(YArray2,
                                     tempFloatArray,
                                     projectorKetTimesVector2,
                                     localVectorSize,
                                     numberVectors,
                                     XArray2,
                                     mixedPrecOverall &&
                                       dftParameters::useMixedPrecCheby,
                                     true,
                                     false);

              if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
                {
                  tempFloatArray.compressAddFinish();

                  copyFloatArrToDoubleArrLocallyOwned<<<(numberVectors + 255) /
                                                          256 * localVectorSize,
                                                        256>>>(
                    numberVectors,
                    localVectorSize,
                    tempFloatArray.begin(),
                    thrust::raw_pointer_cast(
                      &operatorMatrix
                         .getLocallyOwnedProcBoundaryNodesVectorDevice()[0]),
                    XArray1.begin());

                  XArray1.zeroOutGhosts();
                }
              else
                XArray1.compressAddFinish();
              // nvtxRangeEnd(id4);

              // Handle edge case for the second to last Chebyshev filter
              // iteration as there is no overlap algorithm for the next filter
              // iteration.
              if (degree == (m - 1))
                {
                  projectorKetTimesVector2.compressAdd();
                  projectorKetTimesVector2.updateGhostValues();

                  operatorMatrix.HXCheby(YArray2,
                                         tempFloatArray,
                                         projectorKetTimesVector2,
                                         localVectorSize,
                                         numberVectors,
                                         XArray2,
                                         mixedPrecOverall &&
                                           dftParameters::useMixedPrecCheby,
                                         false,
                                         true);
                  YArray2.zeroOutGhosts();
                  if (mixedPrecOverall && dftParameters::useMixedPrecCheby)
                    {
                      convDoubleArrToFloatArr<<<(numberVectors + 255) / 256 *
                                                  totalSize,
                                                256>>>(numberVectors *
                                                         totalSize,
                                                       XArray2.begin(),
                                                       tempFloatArray.begin());
                      tempFloatArray.compressAdd();

                      copyFloatArrToDoubleArrLocallyOwned<<<
                        (numberVectors + 255) / 256 * localVectorSize,
                        256>>>(
                        numberVectors,
                        localVectorSize,
                        tempFloatArray.begin(),
                        thrust::raw_pointer_cast(
                          &operatorMatrix
                             .getLocallyOwnedProcBoundaryNodesVectorDevice()
                               [0]),
                        XArray2.begin());

                      XArray2.zeroOutGhosts();
                    }
                  else
                    XArray2.compressAdd();
                  overlap = false;
                }
              else
                overlap = true;
            }

          XArray1.swap(YArray1);
          // Handle edge cases for the first and last iteration involving
          // overlap of communication and computation
          if (!overlap)
            {
              XArray2.swap(YArray2);
            }


          sigma     = sigma2;
          alpha1Old = alpha1;
        }

      // copy back YArray to XArray
      cudaMemcpy(XArray1.begin(),
                 YArray1.begin(),
                 totalVectorSize * sizeof(dataTypes::numberGPU),
                 cudaMemcpyDeviceToDevice);

      cudaMemcpy(XArray2.begin(),
                 YArray2.begin(),
                 totalVectorSize * sizeof(dataTypes::numberGPU),
                 cudaMemcpyDeviceToDevice);
    }


    void
    subspaceRotationSpectrumSplitScalapack(
      const dataTypes::numberGPU *                     X,
      dataTypes::numberGPU *                           XFrac,
      const unsigned int                               M,
      const unsigned int                               N,
      const unsigned int                               Nfr,
      cublasHandle_t &                                 handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      GPUCCLWrapper &                                  gpucclMpiCommDomain,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const bool                                       rotationMatTranspose)
    {
      const unsigned int maxNumLocalDofs =
        dealii::Utilities::MPI::max(M, mpiCommDomain);

      std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        rotationMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      const unsigned int vectorsBlockSize =
        std::min(dftParameters::wfcBlockSize, Nfr);
      const unsigned int dofsBlockSize =
        std::min(maxNumLocalDofs, dftParameters::subspaceRotDofsBlockSize);

      dataTypes::number *rotationMatBlockHost;

      if (dftParameters::allowFullCPUMemSubspaceRot)
        {
          CUDACHECK(cudaMallocHost((void **)&rotationMatBlockHost,
                                   N * Nfr * sizeof(dataTypes::number)));
          std::memset(rotationMatBlockHost,
                      0,
                      N * Nfr * sizeof(dataTypes::number));
        }
      else
        {
          CUDACHECK(
            cudaMallocHost((void **)&rotationMatBlockHost,
                           vectorsBlockSize * N * sizeof(dataTypes::number)));
          std::memset(rotationMatBlockHost,
                      0,
                      vectorsBlockSize * N * sizeof(dataTypes::number));
        }

      cudaStream_t streamCompute, streamGPUCCL;
      CUDACHECK(cudaStreamCreate(&streamCompute));
      CUDACHECK(cudaStreamCreate(&streamGPUCCL));

      // attach cublas handle to compute stream
      cublasSetStream(handle, streamCompute);

      // create array of compute and gpu direct commun events on GPUs
      // for all the blocks. These are required for synchronization
      const unsigned int numberBlocks =
        (N / vectorsBlockSize) * (maxNumLocalDofs / dofsBlockSize + 1);
      cudaEvent_t computeEvents[numberBlocks];
      cudaEvent_t communEvents[numberBlocks];
      for (int i = 0; i < numberBlocks; ++i)
        {
          CUDACHECK(cudaEventCreate(&computeEvents[i]));
          CUDACHECK(cudaEventCreate(&communEvents[i]));
        }

      thrust::device_vector<dataTypes::numberThrustGPU> rotationMatBlock(
        vectorsBlockSize * N, dataTypes::numberThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberThrustGPU> rotationMatBlockNext(
        vectorsBlockSize * N, dataTypes::numberThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberThrustGPU> rotatedVectorsMatBlock(
        Nfr * dofsBlockSize, dataTypes::numberThrustGPU(0.0));

      dataTypes::numberValueType *tempReal;
      dataTypes::numberValueType *tempImag;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          CUDACHECK(cudaMalloc((void **)&tempReal,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberValueType)));
          CUDACHECK(cudaMalloc((void **)&tempImag,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberValueType)));
        }

      unsigned int blockCount = 0;
      for (unsigned int idof = 0; idof < maxNumLocalDofs; idof += dofsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          unsigned int BDof = 0;
          if (M >= idof)
            BDof = std::min(dofsBlockSize, M - idof);

          // thrust::fill(rotatedVectorsMatBlock.begin(),rotatedVectorsMatBlock.end(),0.);
          for (unsigned int jvec = 0; jvec < Nfr; jvec += vectorsBlockSize)
            {
              // Correct block dimensions if block "goes off edge of" the matrix
              const unsigned int BVec = std::min(vectorsBlockSize, Nfr - jvec);

              const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0);
              const dataTypes::number scalarCoeffBeta  = dataTypes::number(0.0);

              if (dftParameters::allowFullCPUMemSubspaceRot)
                {
                  if (idof == 0)
                    {
                      // Extract QBVec from parallel ScaLAPACK matrix Q
                      if (rotationMatTranspose)
                        {
                          if (processGrid->is_process_active())
                            for (unsigned int i = 0; i < N; ++i)
                              if (globalToLocalRowIdMap.find(i) !=
                                  globalToLocalRowIdMap.end())
                                {
                                  const unsigned int localRowId =
                                    globalToLocalRowIdMap[i];
                                  for (unsigned int j = 0; j < BVec; ++j)
                                    {
                                      std::map<unsigned int,
                                               unsigned int>::iterator it =
                                        globalToLocalColumnIdMap.find(j + jvec);
                                      if (it != globalToLocalColumnIdMap.end())
                                        rotationMatBlockHost[jvec * N +
                                                             i * BVec + j] =
                                          rotationMatPar.local_el(localRowId,
                                                                  it->second);
                                    }
                                }
                        }
                      else
                        {
                          if (processGrid->is_process_active())
                            for (unsigned int i = 0; i < N; ++i)
                              if (globalToLocalColumnIdMap.find(i) !=
                                  globalToLocalColumnIdMap.end())
                                {
                                  const unsigned int localColumnId =
                                    globalToLocalColumnIdMap[i];
                                  for (unsigned int j = 0; j < BVec; ++j)
                                    {
                                      std::map<unsigned int,
                                               unsigned int>::iterator it =
                                        globalToLocalRowIdMap.find(j + jvec);
                                      if (it != globalToLocalRowIdMap.end())
                                        rotationMatBlockHost[jvec * N +
                                                             i * BVec + j] =
                                          rotationMatPar.local_el(
                                            it->second, localColumnId);
                                    }
                                }
                        }
                    }
                }
              else
                {
                  std::memset(rotationMatBlockHost,
                              0,
                              BVec * N * sizeof(dataTypes::numberGPU));

                  // Extract QBVec from parallel ScaLAPACK matrix Q
                  if (rotationMatTranspose)
                    {
                      if (processGrid->is_process_active())
                        for (unsigned int i = 0; i < N; ++i)
                          if (globalToLocalRowIdMap.find(i) !=
                              globalToLocalRowIdMap.end())
                            {
                              const unsigned int localRowId =
                                globalToLocalRowIdMap[i];
                              for (unsigned int j = 0; j < BVec; ++j)
                                {
                                  std::map<unsigned int, unsigned int>::iterator
                                    it =
                                      globalToLocalColumnIdMap.find(j + jvec);
                                  if (it != globalToLocalColumnIdMap.end())
                                    rotationMatBlockHost[i * BVec + j] =
                                      rotationMatPar.local_el(localRowId,
                                                              it->second);
                                }
                            }
                    }
                  else
                    {
                      if (processGrid->is_process_active())
                        for (unsigned int i = 0; i < N; ++i)
                          if (globalToLocalColumnIdMap.find(i) !=
                              globalToLocalColumnIdMap.end())
                            {
                              const unsigned int localColumnId =
                                globalToLocalColumnIdMap[i];
                              for (unsigned int j = 0; j < BVec; ++j)
                                {
                                  std::map<unsigned int, unsigned int>::iterator
                                    it = globalToLocalRowIdMap.find(j + jvec);
                                  if (it != globalToLocalRowIdMap.end())
                                    rotationMatBlockHost[i * BVec + j] =
                                      rotationMatPar.local_el(it->second,
                                                              localColumnId);
                                }
                            }
                    }
                }


              if (dftParameters::allowFullCPUMemSubspaceRot)
                {
                  if (dftParameters::useGPUDirectAllReduce)
                    {
                      CUDACHECK(cudaMemcpyAsync(
                        reinterpret_cast<dataTypes::numberGPU *>(
                          thrust::raw_pointer_cast(&rotationMatBlockNext[0])),
                        rotationMatBlockHost + jvec * N,
                        BVec * N * sizeof(dataTypes::numberGPU),
                        cudaMemcpyHostToDevice,
                        streamGPUCCL));

                      if (idof == 0)
                        {
                          if (std::is_same<dataTypes::number,
                                           std::complex<double>>::value)
                            gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                              reinterpret_cast<dataTypes::numberGPU *>(
                                thrust::raw_pointer_cast(
                                  &rotationMatBlockNext[0])),
                              reinterpret_cast<dataTypes::numberGPU *>(
                                thrust::raw_pointer_cast(
                                  &rotationMatBlockNext[0])),
                              BVec * N,
                              tempReal,
                              tempImag,
                              streamGPUCCL);
                          else
                            gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                              reinterpret_cast<dataTypes::numberGPU *>(
                                thrust::raw_pointer_cast(
                                  &rotationMatBlockNext[0])),
                              reinterpret_cast<dataTypes::numberGPU *>(
                                thrust::raw_pointer_cast(
                                  &rotationMatBlockNext[0])),
                              BVec * N,
                              streamGPUCCL);

                          CUDACHECK(cudaMemcpyAsync(
                            rotationMatBlockHost + jvec * N,
                            reinterpret_cast<const dataTypes::numberGPU *>(
                              thrust::raw_pointer_cast(
                                &rotationMatBlockNext[0])),
                            BVec * N * sizeof(dataTypes::numberGPU),
                            cudaMemcpyDeviceToHost,
                            streamGPUCCL));
                        }
                    }
                  else
                    {
                      if (idof == 0)
                        MPI_Allreduce(MPI_IN_PLACE,
                                      reinterpret_cast<dataTypes::number *>(
                                        rotationMatBlockHost + jvec * N),
                                      BVec * N,
                                      dataTypes::mpi_type_id(
                                        reinterpret_cast<dataTypes::number *>(
                                          rotationMatBlockHost)),
                                      MPI_SUM,
                                      mpiCommDomain);

                      CUDACHECK(cudaMemcpy(
                        reinterpret_cast<dataTypes::numberGPU *>(
                          thrust::raw_pointer_cast(&rotationMatBlock[0])),
                        reinterpret_cast<dataTypes::numberGPU *>(
                          rotationMatBlockHost + jvec * N),
                        BVec * N * sizeof(dataTypes::numberGPU),
                        cudaMemcpyHostToDevice));
                    }
                }
              else
                {
                  if (dftParameters::useGPUDirectAllReduce)
                    {
                      CUDACHECK(cudaMemcpyAsync(
                        reinterpret_cast<dataTypes::numberGPU *>(
                          thrust::raw_pointer_cast(&rotationMatBlockNext[0])),
                        reinterpret_cast<dataTypes::numberGPU *>(
                          rotationMatBlockHost),
                        BVec * N * sizeof(dataTypes::numberGPU),
                        cudaMemcpyHostToDevice,
                        streamGPUCCL));

                      if (std::is_same<dataTypes::number,
                                       std::complex<double>>::value)
                        gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                          reinterpret_cast<dataTypes::numberGPU *>(
                            thrust::raw_pointer_cast(&rotationMatBlockNext[0])),
                          reinterpret_cast<dataTypes::numberGPU *>(
                            thrust::raw_pointer_cast(&rotationMatBlockNext[0])),
                          BVec * N,
                          tempReal,
                          tempImag,
                          streamGPUCCL);
                      else
                        gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                          reinterpret_cast<dataTypes::numberGPU *>(
                            thrust::raw_pointer_cast(&rotationMatBlockNext[0])),
                          reinterpret_cast<dataTypes::numberGPU *>(
                            thrust::raw_pointer_cast(&rotationMatBlockNext[0])),
                          BVec * N,
                          streamGPUCCL);
                    }
                  else
                    {
                      MPI_Allreduce(MPI_IN_PLACE,
                                    rotationMatBlockHost,
                                    BVec * N,
                                    dataTypes::mpi_type_id(
                                      rotationMatBlockHost),
                                    MPI_SUM,
                                    mpiCommDomain);

                      CUDACHECK(cudaMemcpy(
                        reinterpret_cast<dataTypes::numberGPU *>(
                          thrust::raw_pointer_cast(&rotationMatBlock[0])),
                        reinterpret_cast<const dataTypes::numberGPU *>(
                          rotationMatBlockHost),
                        BVec * N * sizeof(dataTypes::numberGPU),
                        cudaMemcpyHostToDevice));
                    }
                }

              if (dftParameters::useGPUDirectAllReduce)
                {
                  // check for completion of compute of previous block in
                  // compute stream before proceeding to rewriting
                  // rotationMatBlock in communication stream
                  CUDACHECK(
                    cudaEventRecord(computeEvents[blockCount], streamCompute));
                  CUDACHECK(cudaStreamWaitEvent(streamGPUCCL,
                                                computeEvents[blockCount],
                                                0));

                  // synchronize host to communication stream before doing swap
                  // this automatically also makes sure the compute stream has
                  // the correct rotationMatBlock for dgemm
                  CUDACHECK(
                    cudaEventRecord(communEvents[blockCount], streamGPUCCL));
                  if (cudaEventSynchronize(communEvents[blockCount]) ==
                      cudaSuccess)
                    rotationMatBlock.swap(rotationMatBlockNext);
                }

              if (BDof != 0)
                {
                  cublasXgemm(handle,
                              CUBLAS_OP_N,
                              CUBLAS_OP_N,
                              BVec,
                              BDof,
                              N,
                              reinterpret_cast<const dataTypes::numberGPU *>(
                                &scalarCoeffAlpha),
                              reinterpret_cast<const dataTypes::numberGPU *>(
                                thrust::raw_pointer_cast(&rotationMatBlock[0])),
                              BVec,
                              X + idof * N,
                              N,
                              reinterpret_cast<const dataTypes::numberGPU *>(
                                &scalarCoeffBeta),
                              reinterpret_cast<dataTypes::numberGPU *>(
                                thrust::raw_pointer_cast(
                                  &rotatedVectorsMatBlock[0]) +
                                jvec),
                              Nfr);
                }

              blockCount++;
            } // block loop over vectors


          if (BDof != 0)
            {
              CUDACHECK(cudaMemcpyAsync(
                XFrac + idof * Nfr,
                reinterpret_cast<const dataTypes::numberGPU *>(
                  thrust::raw_pointer_cast(&rotatedVectorsMatBlock[0])),
                Nfr * BDof * sizeof(dataTypes::numberGPU),
                cudaMemcpyDeviceToDevice,
                streamCompute));
            }

        } // block loop over dofs

      CUDACHECK(cudaFreeHost(rotationMatBlockHost));
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          CUDACHECK(cudaFree(tempReal));
          CUDACHECK(cudaFree(tempImag));
        }

      // return cublas handle to default stream
      cublasSetStream(handle, NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          CUDACHECK(cudaEventDestroy(computeEvents[i]));
          CUDACHECK(cudaEventDestroy(communEvents[i]));
        }

      CUDACHECK(cudaStreamDestroy(streamCompute));
      CUDACHECK(cudaStreamDestroy(streamGPUCCL));
    }



    void
    subspaceRotationScalapack(
      dataTypes::numberGPU *                           X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      GPUCCLWrapper &                                  gpucclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const bool                                       rotationMatTranspose,
      const bool                                       isRotationMatLowerTria)
    {
      const unsigned int maxNumLocalDofs =
        dealii::Utilities::MPI::max(M, mpiCommDomain);

      std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        rotationMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize =
        std::min(dftParameters::wfcBlockSize, N);
      const unsigned int dofsBlockSize =
        std::min(maxNumLocalDofs, dftParameters::subspaceRotDofsBlockSize);

      dataTypes::number *rotationMatBlockHost;

      if (dftParameters::allowFullCPUMemSubspaceRot)
        {
          CUDACHECK(cudaMallocHost((void **)&rotationMatBlockHost,
                                   N * N * sizeof(dataTypes::number)));
          std::memset(rotationMatBlockHost,
                      0,
                      N * N * sizeof(dataTypes::number));
        }
      else
        {
          CUDACHECK(
            cudaMallocHost((void **)&rotationMatBlockHost,
                           vectorsBlockSize * N * sizeof(dataTypes::number)));
          std::memset(rotationMatBlockHost,
                      0,
                      vectorsBlockSize * N * sizeof(dataTypes::number));
        }

      cudaStream_t streamCompute, streamGPUCCL;
      CUDACHECK(cudaStreamCreate(&streamCompute));
      CUDACHECK(cudaStreamCreate(&streamGPUCCL));

      // attach cublas handle to compute stream
      cublasSetStream(handle, streamCompute);

      // create array of compute and gpu direct commun events on GPUs
      // for all the blocks. These are required for synchronization
      const unsigned int numberBlocks =
        (N / vectorsBlockSize) * (maxNumLocalDofs / dofsBlockSize + 1);
      cudaEvent_t computeEvents[numberBlocks];
      cudaEvent_t communEvents[numberBlocks];
      for (int i = 0; i < numberBlocks; ++i)
        {
          CUDACHECK(cudaEventCreate(&computeEvents[i]));
          CUDACHECK(cudaEventCreate(&communEvents[i]));
        }

      thrust::device_vector<dataTypes::numberThrustGPU> rotationMatBlock(
        vectorsBlockSize * N, dataTypes::numberThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberThrustGPU> rotationMatBlockTemp(
        vectorsBlockSize * N, dataTypes::numberThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberThrustGPU> rotatedVectorsMatBlock(
        N * dofsBlockSize, dataTypes::numberThrustGPU(0.0));

      dataTypes::numberValueType *tempReal;
      dataTypes::numberValueType *tempImag;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          CUDACHECK(cudaMalloc((void **)&tempReal,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberValueType)));
          CUDACHECK(cudaMalloc((void **)&tempImag,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberValueType)));
        }

      unsigned int blockCount = 0;
      for (unsigned int idof = 0; idof < maxNumLocalDofs; idof += dofsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          unsigned int BDof = 0;
          if (M >= idof)
            BDof = std::min(dofsBlockSize, M - idof);

          // thrust::fill(rotatedVectorsMatBlock.begin(),rotatedVectorsMatBlock.end(),0.);
          for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
            {
              // Correct block dimensions if block "goes off edge of" the matrix
              const unsigned int BVec = std::min(vectorsBlockSize, N - jvec);

              const unsigned int D = isRotationMatLowerTria ? (jvec + BVec) : N;

              if ((jvec + BVec) <=
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
                  (jvec + BVec) >
                    bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  const dataTypes::number scalarCoeffAlpha =
                    dataTypes::number(1.0);
                  const dataTypes::number scalarCoeffBeta =
                    dataTypes::number(0.0);

                  if (dftParameters::allowFullCPUMemSubspaceRot)
                    {
                      if (idof == 0)
                        {
                          // Extract QBVec from parallel ScaLAPACK matrix Q
                          if (rotationMatTranspose)
                            {
                              if (processGrid->is_process_active())
                                for (unsigned int i = 0; i < D; ++i)
                                  if (globalToLocalRowIdMap.find(i) !=
                                      globalToLocalRowIdMap.end())
                                    {
                                      const unsigned int localRowId =
                                        globalToLocalRowIdMap[i];
                                      for (unsigned int j = 0; j < BVec; ++j)
                                        {
                                          std::map<unsigned int,
                                                   unsigned int>::iterator it =
                                            globalToLocalColumnIdMap.find(j +
                                                                          jvec);
                                          if (it !=
                                              globalToLocalColumnIdMap.end())
                                            rotationMatBlockHost[jvec * N +
                                                                 i * BVec + j] =
                                              rotationMatPar.local_el(
                                                localRowId, it->second);
                                        }
                                    }
                            }
                          else
                            {
                              if (processGrid->is_process_active())
                                for (unsigned int i = 0; i < D; ++i)
                                  if (globalToLocalColumnIdMap.find(i) !=
                                      globalToLocalColumnIdMap.end())
                                    {
                                      const unsigned int localColumnId =
                                        globalToLocalColumnIdMap[i];
                                      for (unsigned int j = 0; j < BVec; ++j)
                                        {
                                          std::map<unsigned int,
                                                   unsigned int>::iterator it =
                                            globalToLocalRowIdMap.find(j +
                                                                       jvec);
                                          if (it != globalToLocalRowIdMap.end())
                                            rotationMatBlockHost[jvec * N +
                                                                 i * BVec + j] =
                                              rotationMatPar.local_el(
                                                it->second, localColumnId);
                                        }
                                    }
                            }
                        }
                    }
                  else
                    {
                      std::memset(rotationMatBlockHost,
                                  0,
                                  BVec * N * sizeof(dataTypes::number));

                      // Extract QBVec from parallel ScaLAPACK matrix Q
                      if (rotationMatTranspose)
                        {
                          if (processGrid->is_process_active())
                            for (unsigned int i = 0; i < D; ++i)
                              if (globalToLocalRowIdMap.find(i) !=
                                  globalToLocalRowIdMap.end())
                                {
                                  const unsigned int localRowId =
                                    globalToLocalRowIdMap[i];
                                  for (unsigned int j = 0; j < BVec; ++j)
                                    {
                                      std::map<unsigned int,
                                               unsigned int>::iterator it =
                                        globalToLocalColumnIdMap.find(j + jvec);
                                      if (it != globalToLocalColumnIdMap.end())
                                        rotationMatBlockHost[i * BVec + j] =
                                          rotationMatPar.local_el(localRowId,
                                                                  it->second);
                                    }
                                }
                        }
                      else
                        {
                          if (processGrid->is_process_active())
                            for (unsigned int i = 0; i < D; ++i)
                              if (globalToLocalColumnIdMap.find(i) !=
                                  globalToLocalColumnIdMap.end())
                                {
                                  const unsigned int localColumnId =
                                    globalToLocalColumnIdMap[i];
                                  for (unsigned int j = 0; j < BVec; ++j)
                                    {
                                      std::map<unsigned int,
                                               unsigned int>::iterator it =
                                        globalToLocalRowIdMap.find(j + jvec);
                                      if (it != globalToLocalRowIdMap.end())
                                        rotationMatBlockHost[i * BVec + j] =
                                          rotationMatPar.local_el(
                                            it->second, localColumnId);
                                    }
                                }
                        }
                    }

                  if (dftParameters::allowFullCPUMemSubspaceRot)
                    {
                      if (dftParameters::useGPUDirectAllReduce)
                        {
                          CUDACHECK(cudaMemcpyAsync(
                            reinterpret_cast<dataTypes::numberGPU *>(
                              thrust::raw_pointer_cast(
                                &rotationMatBlockTemp[0])),
                            reinterpret_cast<const dataTypes::numberGPU *>(
                              rotationMatBlockHost + jvec * N),
                            BVec * D * sizeof(dataTypes::numberGPU),
                            cudaMemcpyHostToDevice,
                            streamGPUCCL));

                          if (idof == 0)
                            {
                              if (std::is_same<dataTypes::number,
                                               std::complex<double>>::value)
                                gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                                  reinterpret_cast<dataTypes::numberGPU *>(
                                    thrust::raw_pointer_cast(
                                      &rotationMatBlockTemp[0])),
                                  reinterpret_cast<dataTypes::numberGPU *>(
                                    thrust::raw_pointer_cast(
                                      &rotationMatBlockTemp[0])),
                                  BVec * D,
                                  tempReal,
                                  tempImag,
                                  streamGPUCCL);
                              else
                                gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                                  reinterpret_cast<dataTypes::numberGPU *>(
                                    thrust::raw_pointer_cast(
                                      &rotationMatBlockTemp[0])),
                                  reinterpret_cast<dataTypes::numberGPU *>(
                                    thrust::raw_pointer_cast(
                                      &rotationMatBlockTemp[0])),
                                  BVec * D,
                                  streamGPUCCL);

                              CUDACHECK(cudaMemcpyAsync(
                                reinterpret_cast<dataTypes::numberGPU *>(
                                  rotationMatBlockHost + jvec * N),
                                reinterpret_cast<const dataTypes::numberGPU *>(
                                  thrust::raw_pointer_cast(
                                    &rotationMatBlockTemp[0])),
                                BVec * D * sizeof(dataTypes::numberGPU),
                                cudaMemcpyDeviceToHost,
                                streamGPUCCL));
                            }
                        }
                      else
                        {
                          if (idof == 0)
                            MPI_Allreduce(MPI_IN_PLACE,
                                          rotationMatBlockHost + jvec * N,
                                          BVec * D,
                                          dataTypes::mpi_type_id(
                                            rotationMatBlockHost),
                                          MPI_SUM,
                                          mpiCommDomain);

                          cudaMemcpy(
                            reinterpret_cast<dataTypes::numberGPU *>(
                              thrust::raw_pointer_cast(&rotationMatBlock[0])),
                            reinterpret_cast<const dataTypes::numberGPU *>(
                              rotationMatBlockHost + jvec * N),
                            BVec * D * sizeof(dataTypes::numberGPU),
                            cudaMemcpyHostToDevice);
                        }
                    }
                  else
                    {
                      if (dftParameters::useGPUDirectAllReduce)
                        {
                          CUDACHECK(cudaMemcpyAsync(
                            reinterpret_cast<dataTypes::numberGPU *>(
                              thrust::raw_pointer_cast(
                                &rotationMatBlockTemp[0])),
                            reinterpret_cast<const dataTypes::numberGPU *>(
                              rotationMatBlockHost),
                            BVec * D * sizeof(dataTypes::numberGPU),
                            cudaMemcpyHostToDevice,
                            streamGPUCCL));

                          if (std::is_same<dataTypes::number,
                                           std::complex<double>>::value)
                            gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                              reinterpret_cast<dataTypes::numberGPU *>(
                                thrust::raw_pointer_cast(
                                  &rotationMatBlockTemp[0])),
                              reinterpret_cast<dataTypes::numberGPU *>(
                                thrust::raw_pointer_cast(
                                  &rotationMatBlockTemp[0])),
                              BVec * D,
                              tempReal,
                              tempImag,
                              streamGPUCCL);
                          else
                            gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                              reinterpret_cast<dataTypes::numberGPU *>(
                                thrust::raw_pointer_cast(
                                  &rotationMatBlockTemp[0])),
                              reinterpret_cast<dataTypes::numberGPU *>(
                                thrust::raw_pointer_cast(
                                  &rotationMatBlockTemp[0])),
                              BVec * D,
                              streamGPUCCL);
                        }
                      else
                        {
                          MPI_Allreduce(MPI_IN_PLACE,
                                        rotationMatBlockHost,
                                        BVec * D,
                                        dataTypes::mpi_type_id(
                                          rotationMatBlockHost),
                                        MPI_SUM,
                                        mpiCommDomain);

                          CUDACHECK(cudaMemcpy(
                            reinterpret_cast<dataTypes::numberGPU *>(
                              thrust::raw_pointer_cast(&rotationMatBlock[0])),
                            reinterpret_cast<const dataTypes::numberGPU *>(
                              rotationMatBlockHost),
                            BVec * D * sizeof(dataTypes::numberGPU),
                            cudaMemcpyHostToDevice));
                        }
                    }

                  if (dftParameters::useGPUDirectAllReduce)
                    {
                      // check for completion of compute of previous block in
                      // compute stream before proceeding to rewriting
                      // rotationMatBlock in communication stream
                      CUDACHECK(cudaEventRecord(computeEvents[blockCount],
                                                streamCompute));
                      CUDACHECK(cudaStreamWaitEvent(streamGPUCCL,
                                                    computeEvents[blockCount],
                                                    0));

                      // synchronize host to communication stream before doing
                      // swap this automatically also makes sure the compute
                      // stream has the correct rotationMatBlock for dgemm
                      CUDACHECK(cudaEventRecord(communEvents[blockCount],
                                                streamGPUCCL));
                      if (cudaEventSynchronize(communEvents[blockCount]) ==
                          cudaSuccess)
                        rotationMatBlock.swap(rotationMatBlockTemp);
                    }

                  if (BDof != 0)
                    {
                      cublasXgemm(
                        handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        BVec,
                        BDof,
                        D,
                        reinterpret_cast<const dataTypes::numberGPU *>(
                          &scalarCoeffAlpha),
                        reinterpret_cast<const dataTypes::numberGPU *>(
                          thrust::raw_pointer_cast(&rotationMatBlock[0])),
                        BVec,
                        X + idof * N,
                        N,
                        reinterpret_cast<const dataTypes::numberGPU *>(
                          &scalarCoeffBeta),
                        reinterpret_cast<dataTypes::numberGPU *>(
                          thrust::raw_pointer_cast(&rotatedVectorsMatBlock[0]) +
                          jvec),
                        N);
                    }
                } // band parallelization
              blockCount++;
            } // block loop over vectors


          if (BDof != 0)
            {
              CUDACHECK(cudaMemcpyAsync(
                X + idof * N,
                reinterpret_cast<const dataTypes::numberGPU *>(
                  thrust::raw_pointer_cast(&rotatedVectorsMatBlock[0])),
                N * BDof * sizeof(dataTypes::numberGPU),
                cudaMemcpyDeviceToDevice,
                streamCompute));
            }

        } // block loop over dofs

      CUDACHECK(cudaFreeHost(rotationMatBlockHost));

      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          CUDACHECK(cudaFree(tempReal));
          CUDACHECK(cudaFree(tempImag));
        }

      // return cublas handle to default stream
      cublasSetStream(handle, NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          CUDACHECK(cudaEventDestroy(computeEvents[i]));
          CUDACHECK(cudaEventDestroy(communEvents[i]));
        }

      CUDACHECK(cudaStreamDestroy(streamCompute));
      CUDACHECK(cudaStreamDestroy(streamGPUCCL));
    }

    void
    subspaceRotationCGSMixedPrecScalapack(
      dataTypes::numberGPU *                           X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      GPUCCLWrapper &                                  gpucclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const bool                                       rotationMatTranspose)
    {
      const unsigned int maxNumLocalDofs =
        dealii::Utilities::MPI::max(M, mpiCommDomain);

      std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        rotationMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const unsigned int MPadded = std::ceil(M * 1.0 / 8.0) * 8.0 + 0.5;
      thrust::device_vector<dataTypes::numberFP32ThrustGPU> XSP(
        MPadded * N, dataTypes::numberFP32ThrustGPU(0.0));

      convDoubleArrToFloatArr<<<(N + 255) / 256 * M, 256>>>(
        N * M,
        X,
        reinterpret_cast<dataTypes::numberFP32GPU *>(
          thrust::raw_pointer_cast(&XSP[0])));


      const unsigned int vectorsBlockSize =
        std::min(dftParameters::wfcBlockSize, N);
      const unsigned int dofsBlockSize =
        std::min(maxNumLocalDofs, dftParameters::subspaceRotDofsBlockSize);

      dataTypes::numberFP32 *rotationMatBlockHostSP;
      CUDACHECK(
        cudaMallocHost((void **)&rotationMatBlockHostSP,
                       vectorsBlockSize * N * sizeof(dataTypes::numberFP32)));
      std::memset(rotationMatBlockHostSP,
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::numberFP32));


      dataTypes::number *diagValuesHost;
      CUDACHECK(cudaMallocHost((void **)&diagValuesHost,
                               N * sizeof(dataTypes::number)));
      std::memset(diagValuesHost, 0, N * sizeof(dataTypes::number));

      cudaStream_t streamCompute, streamGPUCCL;
      CUDACHECK(cudaStreamCreate(&streamCompute));
      CUDACHECK(cudaStreamCreate(&streamGPUCCL));

      // attach cublas handle to compute stream
      cublasSetStream(handle, streamCompute);

      // create array of compute and gpu direct commun events on GPUs
      // for all the blocks. These are required for synchronization
      const unsigned int numberBlocks = (N / vectorsBlockSize);
      cudaEvent_t        computeEvents[numberBlocks];
      cudaEvent_t        communEvents[numberBlocks];
      for (int i = 0; i < numberBlocks; ++i)
        {
          CUDACHECK(cudaEventCreate(&computeEvents[i]));
          CUDACHECK(cudaEventCreate(&communEvents[i]));
        }

      thrust::device_vector<dataTypes::numberFP32ThrustGPU> rotationMatBlockSP(
        vectorsBlockSize * N, dataTypes::numberFP32ThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberFP32ThrustGPU>
                                                        rotationMatBlockSPTemp(vectorsBlockSize * N,
                               dataTypes::numberFP32ThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberThrustGPU> diagValues(
        N, dataTypes::numberThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberFP32ThrustGPU>
        rotatedVectorsMatBlockSP(vectorsBlockSize * dofsBlockSize,
                                 dataTypes::numberFP32ThrustGPU(0.0));

      const dataTypes::numberFP32 scalarCoeffAlphaSP =
        dataTypes::numberFP32(1.0);
      const dataTypes::numberFP32 scalarCoeffBetaSP =
        dataTypes::numberFP32(0.0);


      // Extract DiagQ from parallel ScaLAPACK matrix Q
      if (rotationMatTranspose)
        {
          if (processGrid->is_process_active())
            for (unsigned int i = 0; i < N; ++i)
              if (globalToLocalRowIdMap.find(i) != globalToLocalRowIdMap.end())
                {
                  const unsigned int localRowId = globalToLocalRowIdMap[i];
                  std::map<unsigned int, unsigned int>::iterator it =
                    globalToLocalColumnIdMap.find(i);
                  if (it != globalToLocalColumnIdMap.end())
                    {
                      diagValuesHost[i] =
                        rotationMatPar.local_el(localRowId, it->second);
                    }
                }
        }
      else
        {
          if (processGrid->is_process_active())
            for (unsigned int i = 0; i < N; ++i)
              if (globalToLocalColumnIdMap.find(i) !=
                  globalToLocalColumnIdMap.end())
                {
                  const unsigned int localColumnId =
                    globalToLocalColumnIdMap[i];
                  std::map<unsigned int, unsigned int>::iterator it =
                    globalToLocalRowIdMap.find(i);
                  if (globalToLocalRowIdMap.find(i) !=
                      globalToLocalRowIdMap.end())
                    {
                      diagValuesHost[i] =
                        rotationMatPar.local_el(it->second, localColumnId);
                    }
                }
        }

      MPI_Allreduce(MPI_IN_PLACE,
                    diagValuesHost,
                    N,
                    dataTypes::mpi_type_id(diagValuesHost),
                    MPI_SUM,
                    mpiCommDomain);

      cudaMemcpy(reinterpret_cast<dataTypes::numberGPU *>(
                   thrust::raw_pointer_cast(&diagValues[0])),
                 reinterpret_cast<const dataTypes::numberGPU *>(diagValuesHost),
                 N * sizeof(dataTypes::numberGPU),
                 cudaMemcpyHostToDevice);

      computeDiagQTimesXKernel<<<(M * N + 255) / 256, 256>>>(
        reinterpret_cast<const dataTypes::numberGPU *>(
          thrust::raw_pointer_cast(&diagValues[0])),
        X,
        N,
        M);

      dataTypes::numberFP32ValueType *tempRealFP32;
      dataTypes::numberFP32ValueType *tempImagFP32;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          CUDACHECK(cudaMalloc((void **)&tempRealFP32,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberFP32ValueType)));
          CUDACHECK(cudaMalloc((void **)&tempImagFP32,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberFP32ValueType)));
        }

      unsigned int blockCount = 0;
      for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int BVec = std::min(vectorsBlockSize, N - jvec);

          const unsigned int D = jvec + BVec;

          std::memset(rotationMatBlockHostSP,
                      0,
                      BVec * N * sizeof(dataTypes::numberFP32));

          if ((jvec + BVec) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + BVec) >
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              // Extract QBVec from parallel ScaLAPACK matrix Q
              if (rotationMatTranspose)
                {
                  if (processGrid->is_process_active())
                    for (unsigned int i = 0; i < D; ++i)
                      if (globalToLocalRowIdMap.find(i) !=
                          globalToLocalRowIdMap.end())
                        {
                          const unsigned int localRowId =
                            globalToLocalRowIdMap[i];
                          for (unsigned int j = 0; j < BVec; ++j)
                            {
                              std::map<unsigned int, unsigned int>::iterator
                                it = globalToLocalColumnIdMap.find(j + jvec);
                              if (it != globalToLocalColumnIdMap.end())
                                {
                                  rotationMatBlockHostSP[i * BVec + j] =
                                    rotationMatPar.local_el(localRowId,
                                                            it->second);
                                }
                            }

                          if (i >= jvec && i < (jvec + BVec))
                            {
                              std::map<unsigned int, unsigned int>::iterator
                                it = globalToLocalColumnIdMap.find(i);
                              if (it != globalToLocalColumnIdMap.end())
                                {
                                  rotationMatBlockHostSP[i * BVec + i - jvec] =
                                    dataTypes::numberFP32(0.0);
                                }
                            }
                        }
                }
              else
                {
                  if (processGrid->is_process_active())
                    for (unsigned int i = 0; i < D; ++i)
                      if (globalToLocalColumnIdMap.find(i) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const unsigned int localColumnId =
                            globalToLocalColumnIdMap[i];
                          for (unsigned int j = 0; j < BVec; ++j)
                            {
                              std::map<unsigned int, unsigned int>::iterator
                                it = globalToLocalRowIdMap.find(j + jvec);
                              if (it != globalToLocalRowIdMap.end())
                                {
                                  rotationMatBlockHostSP[i * BVec + j] =
                                    rotationMatPar.local_el(it->second,
                                                            localColumnId);
                                }
                            }

                          if (i >= jvec && i < (jvec + BVec))
                            {
                              std::map<unsigned int, unsigned int>::iterator
                                it = globalToLocalRowIdMap.find(i);
                              if (globalToLocalRowIdMap.find(i) !=
                                  globalToLocalRowIdMap.end())
                                {
                                  rotationMatBlockHostSP[i * BVec + i - jvec] =
                                    dataTypes::numberFP32(0.0);
                                }
                            }
                        }
                }

              if (dftParameters::useGPUDirectAllReduce)
                {
                  cudaMemcpyAsync(
                    reinterpret_cast<dataTypes::numberFP32GPU *>(
                      thrust::raw_pointer_cast(&rotationMatBlockSPTemp[0])),
                    reinterpret_cast<const dataTypes::numberFP32GPU *>(
                      rotationMatBlockHostSP),
                    BVec * D * sizeof(dataTypes::numberFP32GPU),
                    cudaMemcpyHostToDevice,
                    streamGPUCCL);

                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                      reinterpret_cast<dataTypes::numberFP32GPU *>(
                        thrust::raw_pointer_cast(&rotationMatBlockSPTemp[0])),
                      reinterpret_cast<dataTypes::numberFP32GPU *>(
                        thrust::raw_pointer_cast(&rotationMatBlockSPTemp[0])),
                      BVec * D,
                      tempRealFP32,
                      tempImagFP32,
                      streamGPUCCL);
                  else
                    gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                      reinterpret_cast<dataTypes::numberFP32GPU *>(
                        thrust::raw_pointer_cast(&rotationMatBlockSPTemp[0])),
                      reinterpret_cast<dataTypes::numberFP32GPU *>(
                        thrust::raw_pointer_cast(&rotationMatBlockSPTemp[0])),
                      BVec * D,
                      streamGPUCCL);
                }
              else
                {
                  MPI_Allreduce(MPI_IN_PLACE,
                                rotationMatBlockHostSP,
                                BVec * D,
                                dataTypes::mpi_type_id(rotationMatBlockHostSP),
                                MPI_SUM,
                                mpiCommDomain);

                  cudaMemcpy(reinterpret_cast<dataTypes::numberFP32GPU *>(
                               thrust::raw_pointer_cast(
                                 &rotationMatBlockSP[0])),
                             reinterpret_cast<const dataTypes::numberFP32GPU *>(
                               rotationMatBlockHostSP),
                             BVec * D * sizeof(dataTypes::numberFP32GPU),
                             cudaMemcpyHostToDevice);
                }

              if (dftParameters::useGPUDirectAllReduce)
                {
                  // check for completion of compute of previous block in
                  // compute stream before proceeding to rewriting
                  // rotationMatBlock in communication stream
                  CUDACHECK(
                    cudaEventRecord(computeEvents[blockCount], streamCompute));
                  CUDACHECK(cudaStreamWaitEvent(streamGPUCCL,
                                                computeEvents[blockCount],
                                                0));

                  // synchronize host to communication stream before doing swap
                  // this automatically also makes sure the compute stream has
                  // the correct rotationMatBlock for dgemm
                  CUDACHECK(
                    cudaEventRecord(communEvents[blockCount], streamGPUCCL));
                  if (cudaEventSynchronize(communEvents[blockCount]) ==
                      cudaSuccess)
                    rotationMatBlockSP.swap(rotationMatBlockSPTemp);
                }

              for (unsigned int idof = 0; idof < maxNumLocalDofs;
                   idof += dofsBlockSize)
                {
                  // Correct block dimensions if block "goes off edge of" the
                  // matrix
                  unsigned int BDof = 0;
                  if (M >= idof)
                    BDof = std::min(dofsBlockSize, M - idof);

                  if (BDof != 0)
                    {
                      cublasXgemm(
                        handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        BVec,
                        BDof,
                        D,
                        reinterpret_cast<const dataTypes::numberFP32GPU *>(
                          &scalarCoeffAlphaSP),
                        reinterpret_cast<const dataTypes::numberFP32GPU *>(
                          thrust::raw_pointer_cast(&rotationMatBlockSP[0])),
                        BVec,
                        reinterpret_cast<const dataTypes::numberFP32GPU *>(
                          thrust::raw_pointer_cast(&XSP[0]) + idof * N),
                        N,
                        reinterpret_cast<const dataTypes::numberFP32GPU *>(
                          &scalarCoeffBetaSP),
                        reinterpret_cast<dataTypes::numberFP32GPU *>(
                          thrust::raw_pointer_cast(
                            &rotatedVectorsMatBlockSP[0])),
                        BVec);

                      addSubspaceRotatedBlockToXKernel<<<(BVec * BDof + 255) /
                                                           256,
                                                         256,
                                                         0,
                                                         streamCompute>>>(
                        BDof,
                        BVec,
                        reinterpret_cast<const dataTypes::numberFP32GPU *>(
                          thrust::raw_pointer_cast(
                            &rotatedVectorsMatBlockSP[0])),
                        X,
                        idof,
                        jvec,
                        N);
                    }
                } // block loop over dofs
            }     // band parallalelization loop
          blockCount++;
        } // block loop over vectors

      CUDACHECK(cudaFreeHost(rotationMatBlockHostSP));
      CUDACHECK(cudaFreeHost(diagValuesHost));

      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          CUDACHECK(cudaFree(tempRealFP32));
          CUDACHECK(cudaFree(tempImagFP32));
        }

      // return cublas handle to default stream
      cublasSetStream(handle, NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          CUDACHECK(cudaEventDestroy(computeEvents[i]));
          CUDACHECK(cudaEventDestroy(communEvents[i]));
        }

      CUDACHECK(cudaStreamDestroy(streamCompute));
      CUDACHECK(cudaStreamDestroy(streamGPUCCL));
    }

    void
    subspaceRotationRRMixedPrecScalapack(
      dataTypes::numberGPU *                           X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      GPUCCLWrapper &                                  gpucclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const bool                                       rotationMatTranspose)
    {
      const unsigned int maxNumLocalDofs =
        dealii::Utilities::MPI::max(M, mpiCommDomain);

      std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        rotationMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      const unsigned int MPadded = std::ceil(M * 1.0 / 8.0) * 8.0 + 0.5;
      thrust::device_vector<dataTypes::numberFP32ThrustGPU> XSP(
        MPadded * N, dataTypes::numberFP32ThrustGPU(0.0));

      convDoubleArrToFloatArr<<<(N + 255) / 256 * M, 256>>>(
        N * M,
        X,
        reinterpret_cast<dataTypes::numberFP32GPU *>(
          thrust::raw_pointer_cast(&XSP[0])));


      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize =
        std::min(dftParameters::wfcBlockSize, N);
      const unsigned int dofsBlockSize =
        std::min(maxNumLocalDofs, dftParameters::subspaceRotDofsBlockSize);

      dataTypes::numberFP32 *rotationMatBlockHostSP;
      CUDACHECK(
        cudaMallocHost((void **)&rotationMatBlockHostSP,
                       vectorsBlockSize * N * sizeof(dataTypes::numberFP32)));
      std::memset(rotationMatBlockHostSP,
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::numberFP32));


      dataTypes::number *diagValuesHost;
      CUDACHECK(cudaMallocHost((void **)&diagValuesHost,
                               N * sizeof(dataTypes::number)));
      std::memset(diagValuesHost, 0, N * sizeof(dataTypes::number));

      cudaStream_t streamCompute, streamGPUCCL;
      CUDACHECK(cudaStreamCreate(&streamCompute));
      CUDACHECK(cudaStreamCreate(&streamGPUCCL));

      // attach cublas handle to compute stream
      cublasSetStream(handle, streamCompute);

      // create array of compute and gpu direct commun events on GPUs
      // for all the blocks. These are required for synchronization
      const unsigned int numberBlocks = (N / vectorsBlockSize);
      cudaEvent_t        computeEvents[numberBlocks];
      cudaEvent_t        communEvents[numberBlocks];
      for (int i = 0; i < numberBlocks; ++i)
        {
          CUDACHECK(cudaEventCreate(&computeEvents[i]));
          CUDACHECK(cudaEventCreate(&communEvents[i]));
        }

      thrust::device_vector<dataTypes::numberFP32ThrustGPU> rotationMatBlockSP(
        vectorsBlockSize * N, dataTypes::numberFP32ThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberFP32ThrustGPU>
                                                        rotationMatBlockSPTemp(vectorsBlockSize * N,
                               dataTypes::numberFP32ThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberThrustGPU> diagValues(
        N, dataTypes::numberThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberFP32ThrustGPU>
        rotatedVectorsMatBlockSP(vectorsBlockSize * dofsBlockSize,
                                 dataTypes::numberFP32ThrustGPU(0.0));

      const dataTypes::numberFP32 scalarCoeffAlphaSP =
        dataTypes::numberFP32(1.0);
      const dataTypes::numberFP32 scalarCoeffBetaSP =
        dataTypes::numberFP32(0.0);


      // Extract DiagQ from parallel ScaLAPACK matrix Q
      if (rotationMatTranspose)
        {
          if (processGrid->is_process_active())
            for (unsigned int i = 0; i < N; ++i)
              if (globalToLocalRowIdMap.find(i) != globalToLocalRowIdMap.end())
                {
                  const unsigned int localRowId = globalToLocalRowIdMap[i];
                  std::map<unsigned int, unsigned int>::iterator it =
                    globalToLocalColumnIdMap.find(i);
                  if (it != globalToLocalColumnIdMap.end())
                    {
                      diagValuesHost[i] =
                        rotationMatPar.local_el(localRowId, it->second);
                    }
                }
        }
      else
        {
          if (processGrid->is_process_active())
            for (unsigned int i = 0; i < N; ++i)
              if (globalToLocalColumnIdMap.find(i) !=
                  globalToLocalColumnIdMap.end())
                {
                  const unsigned int localColumnId =
                    globalToLocalColumnIdMap[i];
                  std::map<unsigned int, unsigned int>::iterator it =
                    globalToLocalRowIdMap.find(i);
                  if (globalToLocalRowIdMap.find(i) !=
                      globalToLocalRowIdMap.end())
                    {
                      diagValuesHost[i] =
                        rotationMatPar.local_el(it->second, localColumnId);
                    }
                }
        }

      MPI_Allreduce(MPI_IN_PLACE,
                    diagValuesHost,
                    N,
                    dataTypes::mpi_type_id(diagValuesHost),
                    MPI_SUM,
                    mpiCommDomain);

      cudaMemcpy(reinterpret_cast<dataTypes::numberGPU *>(
                   thrust::raw_pointer_cast(&diagValues[0])),
                 reinterpret_cast<const dataTypes::numberGPU *>(diagValuesHost),
                 N * sizeof(dataTypes::numberGPU),
                 cudaMemcpyHostToDevice);

      computeDiagQTimesXKernel<<<(M * N + 255) / 256, 256>>>(
        reinterpret_cast<const dataTypes::numberGPU *>(
          thrust::raw_pointer_cast(&diagValues[0])),
        X,
        N,
        M);

      dataTypes::numberFP32ValueType *tempRealFP32;
      dataTypes::numberFP32ValueType *tempImagFP32;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          CUDACHECK(cudaMalloc((void **)&tempRealFP32,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberFP32ValueType)));
          CUDACHECK(cudaMalloc((void **)&tempImagFP32,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberFP32ValueType)));
        }

      unsigned int blockCount = 0;
      for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int BVec = std::min(vectorsBlockSize, N - jvec);

          const unsigned int D = N;

          if ((jvec + BVec) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (jvec + BVec) >
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              std::memset(rotationMatBlockHostSP,
                          0,
                          BVec * N * sizeof(dataTypes::numberFP32));

              // Extract QBVec from parallel ScaLAPACK matrix Q
              if (rotationMatTranspose)
                {
                  if (processGrid->is_process_active())
                    for (unsigned int i = 0; i < D; ++i)
                      if (globalToLocalRowIdMap.find(i) !=
                          globalToLocalRowIdMap.end())
                        {
                          const unsigned int localRowId =
                            globalToLocalRowIdMap[i];
                          for (unsigned int j = 0; j < BVec; ++j)
                            {
                              std::map<unsigned int, unsigned int>::iterator
                                it = globalToLocalColumnIdMap.find(j + jvec);
                              if (it != globalToLocalColumnIdMap.end())
                                {
                                  rotationMatBlockHostSP[i * BVec + j] =
                                    rotationMatPar.local_el(localRowId,
                                                            it->second);
                                }
                            }

                          if (i >= jvec && i < (jvec + BVec))
                            {
                              std::map<unsigned int, unsigned int>::iterator
                                it = globalToLocalColumnIdMap.find(i);
                              if (it != globalToLocalColumnIdMap.end())
                                {
                                  rotationMatBlockHostSP[i * BVec + i - jvec] =
                                    dataTypes::numberFP32(0.0);
                                }
                            }
                        }
                }
              else
                {
                  if (processGrid->is_process_active())
                    for (unsigned int i = 0; i < D; ++i)
                      if (globalToLocalColumnIdMap.find(i) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const unsigned int localColumnId =
                            globalToLocalColumnIdMap[i];
                          for (unsigned int j = 0; j < BVec; ++j)
                            {
                              std::map<unsigned int, unsigned int>::iterator
                                it = globalToLocalRowIdMap.find(j + jvec);
                              if (it != globalToLocalRowIdMap.end())
                                {
                                  rotationMatBlockHostSP[i * BVec + j] =
                                    rotationMatPar.local_el(it->second,
                                                            localColumnId);
                                }
                            }

                          if (i >= jvec && i < (jvec + BVec))
                            {
                              std::map<unsigned int, unsigned int>::iterator
                                it = globalToLocalRowIdMap.find(i);
                              if (globalToLocalRowIdMap.find(i) !=
                                  globalToLocalRowIdMap.end())
                                {
                                  rotationMatBlockHostSP[i * BVec + i - jvec] =
                                    dataTypes::numberFP32(0.0);
                                }
                            }
                        }
                }


              if (dftParameters::useGPUDirectAllReduce)
                {
                  cudaMemcpyAsync(
                    reinterpret_cast<dataTypes::numberFP32GPU *>(
                      thrust::raw_pointer_cast(&rotationMatBlockSPTemp[0])),
                    reinterpret_cast<const dataTypes::numberFP32GPU *>(
                      rotationMatBlockHostSP),
                    BVec * D * sizeof(dataTypes::numberFP32GPU),
                    cudaMemcpyHostToDevice,
                    streamGPUCCL);

                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                      reinterpret_cast<dataTypes::numberFP32GPU *>(
                        thrust::raw_pointer_cast(&rotationMatBlockSPTemp[0])),
                      reinterpret_cast<dataTypes::numberFP32GPU *>(
                        thrust::raw_pointer_cast(&rotationMatBlockSPTemp[0])),
                      BVec * D,
                      tempRealFP32,
                      tempImagFP32,
                      streamGPUCCL);
                  else
                    gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                      reinterpret_cast<dataTypes::numberFP32GPU *>(
                        thrust::raw_pointer_cast(&rotationMatBlockSPTemp[0])),
                      reinterpret_cast<dataTypes::numberFP32GPU *>(
                        thrust::raw_pointer_cast(&rotationMatBlockSPTemp[0])),
                      BVec * D,
                      streamGPUCCL);
                }
              else
                {
                  MPI_Allreduce(MPI_IN_PLACE,
                                rotationMatBlockHostSP,
                                BVec * D,
                                dataTypes::mpi_type_id(rotationMatBlockHostSP),
                                MPI_SUM,
                                mpiCommDomain);

                  cudaMemcpy(reinterpret_cast<dataTypes::numberFP32GPU *>(
                               thrust::raw_pointer_cast(
                                 &rotationMatBlockSP[0])),
                             reinterpret_cast<const dataTypes::numberFP32GPU *>(
                               rotationMatBlockHostSP),
                             BVec * D * sizeof(dataTypes::numberFP32GPU),
                             cudaMemcpyHostToDevice);
                }

              if (dftParameters::useGPUDirectAllReduce)
                {
                  // check for completion of compute of previous block in
                  // compute stream before proceeding to rewriting
                  // rotationMatBlock in communication stream
                  CUDACHECK(
                    cudaEventRecord(computeEvents[blockCount], streamCompute));
                  CUDACHECK(cudaStreamWaitEvent(streamGPUCCL,
                                                computeEvents[blockCount],
                                                0));

                  // synchronize host to communication stream before doing swap
                  // this automatically also makes sure the compute stream has
                  // the correct rotationMatBlock for dgemm
                  CUDACHECK(
                    cudaEventRecord(communEvents[blockCount], streamGPUCCL));
                  if (cudaEventSynchronize(communEvents[blockCount]) ==
                      cudaSuccess)
                    rotationMatBlockSP.swap(rotationMatBlockSPTemp);
                }

              for (unsigned int idof = 0; idof < maxNumLocalDofs;
                   idof += dofsBlockSize)
                {
                  // Correct block dimensions if block "goes off edge of" the
                  // matrix
                  unsigned int BDof = 0;
                  if (M >= idof)
                    BDof = std::min(dofsBlockSize, M - idof);

                  if (BDof != 0)
                    {
                      cublasXgemm(
                        handle,
                        CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        BVec,
                        BDof,
                        D,
                        reinterpret_cast<const dataTypes::numberFP32GPU *>(
                          &scalarCoeffAlphaSP),
                        reinterpret_cast<const dataTypes::numberFP32GPU *>(
                          thrust::raw_pointer_cast(&rotationMatBlockSP[0])),
                        BVec,
                        reinterpret_cast<const dataTypes::numberFP32GPU *>(
                          thrust::raw_pointer_cast(&XSP[0]) + idof * N),
                        N,
                        reinterpret_cast<const dataTypes::numberFP32GPU *>(
                          &scalarCoeffBetaSP),
                        reinterpret_cast<dataTypes::numberFP32GPU *>(
                          thrust::raw_pointer_cast(
                            &rotatedVectorsMatBlockSP[0])),
                        BVec);


                      addSubspaceRotatedBlockToXKernel<<<(BVec * BDof + 255) /
                                                           256,
                                                         256,
                                                         0,
                                                         streamCompute>>>(
                        BDof,
                        BVec,
                        reinterpret_cast<const dataTypes::numberFP32GPU *>(
                          thrust::raw_pointer_cast(
                            &rotatedVectorsMatBlockSP[0])),
                        X,
                        idof,
                        jvec,
                        N);
                    }
                } // block loop over dofs
            }     // band parallelization
          blockCount++;
        } // block loop over vectors

      CUDACHECK(cudaFreeHost(rotationMatBlockHostSP));
      CUDACHECK(cudaFreeHost(diagValuesHost));

      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          CUDACHECK(cudaFree(tempRealFP32));
          CUDACHECK(cudaFree(tempImagFP32));
        }

      // return cublas handle to default stream
      cublasSetStream(handle, NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          CUDACHECK(cudaEventDestroy(computeEvents[i]));
          CUDACHECK(cudaEventDestroy(communEvents[i]));
        }

      CUDACHECK(cudaStreamDestroy(streamCompute));
      CUDACHECK(cudaStreamDestroy(streamGPUCCL));
    }

    void
    fillParallelOverlapMatScalapack(
      const dataTypes::numberGPU *                     X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const MPI_Comm &                                 mpiCommDomain,
      GPUCCLWrapper &                                  gpucclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar)
    {
      // get global to local index maps for Scalapack matrix
      std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        overlapMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize =
        std::min(dftParameters::wfcBlockSize, N);

      thrust::device_vector<dataTypes::numberThrustGPU> overlapMatrixBlock(
        N * vectorsBlockSize, dataTypes::numberThrustGPU(0.0));

      dataTypes::number *overlapMatrixBlockHost;
      CUDACHECK(
        cudaMallocHost((void **)&overlapMatrixBlockHost,
                       N * vectorsBlockSize * sizeof(dataTypes::number)));
      std::memset(overlapMatrixBlockHost,
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::number));

      cudaStream_t streamGPUCCL;
      cudaStreamCreate(&streamGPUCCL);

      const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0);
      const dataTypes::number scalarCoeffBeta  = dataTypes::number(0.0);

      dataTypes::numberValueType *tempReal;
      dataTypes::numberValueType *tempImag;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          CUDACHECK(cudaMalloc((void **)&tempReal,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberValueType)));
          CUDACHECK(cudaMalloc((void **)&tempImag,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberValueType)));
        }

      for (unsigned int ivec = 0; ivec < N; ivec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int B = std::min(vectorsBlockSize, N - ivec);

          // thrust::fill(overlapMatrixBlock.begin(),overlapMatrixBlock.end(),0.);

          const unsigned int D = N - ivec;

          if ((ivec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (ivec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              // Comptute local XTrunc^{T}*XcBlock.
              cublasXgemm(
                handle,
                CUBLAS_OP_N,
                std::is_same<dataTypes::number, std::complex<double>>::value ?
                  CUBLAS_OP_C :
                  CUBLAS_OP_T,
                D,
                B,
                M,
                reinterpret_cast<const dataTypes::numberGPU *>(
                  &scalarCoeffAlpha),
                X + ivec,
                N,
                X + ivec,
                N,
                reinterpret_cast<const dataTypes::numberGPU *>(
                  &scalarCoeffBeta),
                reinterpret_cast<dataTypes::numberGPU *>(
                  thrust::raw_pointer_cast(&overlapMatrixBlock[0])),
                D);


              if (dftParameters::useGPUDirectAllReduce)
                {
                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlock[0])),
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlock[0])),
                      D * B,
                      tempReal,
                      tempImag,
                      streamGPUCCL);
                  else
                    gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlock[0])),
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlock[0])),
                      D * B,
                      streamGPUCCL);
                }

              cudaMemcpy(reinterpret_cast<dataTypes::numberGPU *>(
                           overlapMatrixBlockHost),
                         reinterpret_cast<dataTypes::numberGPU *>(
                           thrust::raw_pointer_cast(&overlapMatrixBlock[0])),
                         D * B * sizeof(dataTypes::numberGPU),
                         cudaMemcpyDeviceToHost);

              // Sum local XTrunc^{T}*XcBlock across domain decomposition
              // processors
              if (!dftParameters::useGPUDirectAllReduce)
                MPI_Allreduce(MPI_IN_PLACE,
                              overlapMatrixBlockHost,
                              D * B,
                              dataTypes::mpi_type_id(overlapMatrixBlockHost),
                              MPI_SUM,
                              mpiCommDomain);


              // Copying only the lower triangular part to the ScaLAPACK overlap
              // matrix
              if (processGrid->is_process_active())
                for (unsigned int i = 0; i < B; ++i)
                  if (globalToLocalColumnIdMap.find(i + ivec) !=
                      globalToLocalColumnIdMap.end())
                    {
                      const unsigned int localColumnId =
                        globalToLocalColumnIdMap[i + ivec];
                      for (unsigned int j = ivec + i; j < N; ++j)
                        {
                          std::map<unsigned int, unsigned int>::iterator it =
                            globalToLocalRowIdMap.find(j);
                          if (it != globalToLocalRowIdMap.end())
                            overlapMatPar.local_el(it->second, localColumnId) =
                              overlapMatrixBlockHost[i * D + j - ivec];
                        }
                    }

            } // band parallelization
        }     // end block loop

      CUDACHECK(cudaFreeHost(overlapMatrixBlockHost));
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          CUDACHECK(cudaFree(tempReal));
          CUDACHECK(cudaFree(tempImag));
        }

      cudaStreamDestroy(streamGPUCCL);

      if (numberBandGroups > 1)
        linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
          processGrid, overlapMatPar, interBandGroupComm);
    }

    /////////////PSEUDO CODE for the implementation below for Overlapping
    /// compute and communication in the computation of overlap
    /// matrix/////////////////
    //
    // In the algorithm below the communication and computation of two
    // consecutive blocks of wavefunctions: block i and block i+1 are
    // overlapped.
    // ----------------------------------------------------------
    // CMP denotes computuation of X^{T} times XBlock
    // COP denotes GPU->CPU copy of X^{T} times XBlock
    // COM denotes blocking MPI_Allreduce on X^{T}XBlock and copy to scalapack
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

    void
    fillParallelOverlapMatScalapackAsyncComputeCommun(
      const dataTypes::numberGPU *                     X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const MPI_Comm &                                 mpiCommDomain,
      GPUCCLWrapper &                                  gpucclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar)
    {
      // get global to local index maps for Scalapack matrix
      std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        overlapMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize =
        std::min(dftParameters::wfcBlockSize, N);
      const unsigned int numberBlocks = N / vectorsBlockSize;

      // create separate CUDA streams for data movement and computation
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

      // create pinned memory used later to copy from GPU->CPU
      dataTypes::number *overlapMatrixBlockHost;
      CUDACHECK(
        cudaMallocHost((void **)&overlapMatrixBlockHost,
                       N * vectorsBlockSize * sizeof(dataTypes::number)));
      std::memset(overlapMatrixBlockHost,
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::number));

      // allocate device vectors to be used later
      thrust::device_vector<dataTypes::numberThrustGPU> overlapMatrixBlock(
        N * vectorsBlockSize, dataTypes::numberThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberThrustGPU> overlapMatrixBlockNext(
        N * vectorsBlockSize, dataTypes::numberThrustGPU(0.0));

      const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0);
      const dataTypes::number scalarCoeffBeta  = dataTypes::number(0.0);

      dataTypes::numberValueType *tempReal;
      dataTypes::numberValueType *tempImag;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          CUDACHECK(cudaMalloc((void **)&tempReal,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberValueType)));
          CUDACHECK(cudaMalloc((void **)&tempImag,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberValueType)));
        }

      unsigned int blockCount = 0;
      for (unsigned int ivec = 0; ivec < N; ivec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int B = std::min(vectorsBlockSize, N - ivec);
          const unsigned int D = N - ivec;

          if ((ivec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (ivec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              // Compute local XTrunc^{T}*XcBlock.
              if (ivec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  cublasXgemm(handle,
                              CUBLAS_OP_N,
                              std::is_same<dataTypes::number,
                                           std::complex<double>>::value ?
                                CUBLAS_OP_C :
                                CUBLAS_OP_T,
                              D,
                              B,
                              M,
                              reinterpret_cast<const dataTypes::numberGPU *>(
                                &scalarCoeffAlpha),
                              X + ivec,
                              N,
                              X + ivec,
                              N,
                              reinterpret_cast<const dataTypes::numberGPU *>(
                                &scalarCoeffBeta),
                              reinterpret_cast<dataTypes::numberGPU *>(
                                thrust::raw_pointer_cast(
                                  &overlapMatrixBlock[0])),
                              D);

                  // record completion of compute for first block
                  CUDACHECK(
                    cudaEventRecord(computeEvents[blockCount], streamCompute));
                }

              // Before swap host thread needs to wait till compute on
              // currentblock is over. Since swap occurs on the null stream, any
              // future operations in the streamDataMove will only occur after
              // both the compute on currentblock and swap is over. Note that at
              // this point there is nothing queued in the streamDataMove as all
              // previous operations in that stream are over.
              if ((cudaEventSynchronize(computeEvents[blockCount]) ==
                   cudaSuccess) &&
                  (ivec > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]))
                overlapMatrixBlock.swap(overlapMatrixBlockNext);

              const unsigned int ivecNew = ivec + vectorsBlockSize;
              const unsigned int DNew    = N - ivecNew;
              const unsigned int BNew    = min(vectorsBlockSize, N - ivecNew);


              // start computations on the next block
              if (ivecNew <
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
                {
                  // thrust::fill(overlapMatrixBlockNext.begin(),overlapMatrixBlockNext.end(),0.);

                  // evaluate X^{T} times XBlock
                  cublasXgemm(handle,
                              CUBLAS_OP_N,
                              std::is_same<dataTypes::number,
                                           std::complex<double>>::value ?
                                CUBLAS_OP_C :
                                CUBLAS_OP_T,
                              DNew,
                              BNew,
                              M,
                              reinterpret_cast<const dataTypes::numberGPU *>(
                                &scalarCoeffAlpha),
                              X + ivecNew,
                              N,
                              X + ivecNew,
                              N,
                              reinterpret_cast<const dataTypes::numberGPU *>(
                                &scalarCoeffBeta),
                              reinterpret_cast<dataTypes::numberGPU *>(
                                thrust::raw_pointer_cast(
                                  &overlapMatrixBlockNext[0])),
                              DNew);

                  // record completion of compute for next block
                  CUDACHECK(cudaEventRecord(computeEvents[blockCount + 1],
                                            streamCompute));
                }

              if (dftParameters::useGPUDirectAllReduce)
                {
                  // Sum local XTrunc^{T}*XcBlock across domain decomposition
                  // processors
                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlock[0])),
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlock[0])),
                      D * B,
                      tempReal,
                      tempImag,
                      streamDataMove);
                  else
                    gpucclMpiCommDomain.gpuDirectAllReduceWrapper(
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlock[0])),
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlock[0])),
                      D * B,
                      streamDataMove);
                }

              CUDACHECK(cudaMemcpyAsync(
                reinterpret_cast<dataTypes::numberGPU *>(
                  overlapMatrixBlockHost),
                reinterpret_cast<const dataTypes::numberGPU *>(
                  thrust::raw_pointer_cast(&overlapMatrixBlock[0])),
                D * B * sizeof(dataTypes::numberGPU),
                cudaMemcpyDeviceToHost,
                streamDataMove));

              // record completion of GPU->CPU copy for current block
              CUDACHECK(
                cudaEventRecord(copyEvents[blockCount], streamDataMove));

              // Check that GPU->CPU on the current block has been completed. If
              // completed, perform blocking MPI commmunication on the current
              // block and copy to ScaLAPACK matri
              if (cudaEventSynchronize(copyEvents[blockCount]) == cudaSuccess)
                {
                  // Sum local XTrunc^{T}*XcBlock across domain decomposition
                  // processors
                  if (!dftParameters::useGPUDirectAllReduce)
                    MPI_Allreduce(MPI_IN_PLACE,
                                  overlapMatrixBlockHost,
                                  D * B,
                                  dataTypes::mpi_type_id(
                                    overlapMatrixBlockHost),
                                  MPI_SUM,
                                  mpiCommDomain);


                  // Copying only the lower triangular part to the ScaLAPACK
                  // overlap matrix
                  if (processGrid->is_process_active())
                    for (unsigned int i = 0; i < B; ++i)
                      if (globalToLocalColumnIdMap.find(i + ivec) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const unsigned int localColumnId =
                            globalToLocalColumnIdMap[i + ivec];
                          for (unsigned int j = ivec + i; j < N; ++j)
                            {
                              std::map<unsigned int, unsigned int>::iterator
                                it = globalToLocalRowIdMap.find(j);
                              if (it != globalToLocalRowIdMap.end())
                                overlapMatPar.local_el(it->second,
                                                       localColumnId) =
                                  overlapMatrixBlockHost[i * D + j - ivec];
                            }
                        }
                }
            } // band parallelization

          blockCount += 1;
        } // end block loop

      CUDACHECK(cudaFreeHost(overlapMatrixBlockHost));
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          CUDACHECK(cudaFree(tempReal));
          CUDACHECK(cudaFree(tempImag));
        }

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
          MPI_Barrier(interBandGroupComm);

          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, overlapMatPar, interBandGroupComm);
        }
    }


    void
    fillParallelOverlapMatMixedPrecScalapack(
      const dataTypes::numberGPU *                     X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const MPI_Comm &                                 mpiCommDomain,
      GPUCCLWrapper &                                  gpucclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar)
    {
      // get global to local index maps for Scalapack matrix
      std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        overlapMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize =
        std::min(dftParameters::wfcBlockSize, N);


      thrust::device_vector<dataTypes::numberFP32ThrustGPU>
                                                        overlapMatrixBlockSP(N * vectorsBlockSize,
                             dataTypes::numberFP32ThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberThrustGPU> overlapMatrixBlockDP(
        vectorsBlockSize * vectorsBlockSize, dataTypes::numberThrustGPU(0.0));

      const unsigned int MPadded = std::ceil(M * 1.0 / 8.0) * 8.0 + 0.5;
      thrust::device_vector<dataTypes::numberFP32ThrustGPU> XSP(
        MPadded * N, dataTypes::numberFP32ThrustGPU(0.0));

      convDoubleArrToFloatArr<<<(N + 255) / 256 * M, 256>>>(
        N * M,
        X,
        reinterpret_cast<dataTypes::numberFP32GPU *>(
          thrust::raw_pointer_cast(&XSP[0])));
      dataTypes::number *overlapMatrixBlockHostDP;
      CUDACHECK(cudaMallocHost((void **)&overlapMatrixBlockHostDP,
                               vectorsBlockSize * vectorsBlockSize *
                                 sizeof(dataTypes::number)));
      std::memset(overlapMatrixBlockHostDP,
                  0,
                  vectorsBlockSize * vectorsBlockSize *
                    sizeof(dataTypes::number));

      dataTypes::numberFP32 *overlapMatrixBlockHostSP;
      CUDACHECK(
        cudaMallocHost((void **)&overlapMatrixBlockHostSP,
                       N * vectorsBlockSize * sizeof(dataTypes::numberFP32)));
      std::memset(overlapMatrixBlockHostSP,
                  0,
                  N * vectorsBlockSize * sizeof(dataTypes::numberFP32));

      cudaStream_t streamGPUCCL;
      cudaStreamCreate(&streamGPUCCL);

      const dataTypes::number     scalarCoeffAlpha = dataTypes::number(1.0);
      const dataTypes::number     scalarCoeffBeta  = dataTypes::number(0.0);
      const dataTypes::numberFP32 scalarCoeffAlphaSP =
        dataTypes::numberFP32(1.0);
      const dataTypes::numberFP32 scalarCoeffBetaSP =
        dataTypes::numberFP32(0.0);

      dataTypes::numberValueType *    tempReal;
      dataTypes::numberValueType *    tempImag;
      dataTypes::numberFP32ValueType *tempRealFP32;
      dataTypes::numberFP32ValueType *tempImagFP32;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          CUDACHECK(cudaMalloc((void **)&tempReal,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberValueType)));
          CUDACHECK(cudaMalloc((void **)&tempImag,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberValueType)));
          CUDACHECK(cudaMalloc((void **)&tempRealFP32,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberFP32ValueType)));
          CUDACHECK(cudaMalloc((void **)&tempImagFP32,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberFP32ValueType)));
        }

      for (unsigned int ivec = 0; ivec < N; ivec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int B = std::min(vectorsBlockSize, N - ivec);


          const unsigned int D = N - ivec;

          if ((ivec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (ivec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              cublasXgemm(
                handle,
                CUBLAS_OP_N,
                std::is_same<dataTypes::number, std::complex<double>>::value ?
                  CUBLAS_OP_C :
                  CUBLAS_OP_T,
                B,
                B,
                M,
                reinterpret_cast<const dataTypes::numberGPU *>(
                  &scalarCoeffAlpha),
                X + ivec,
                N,
                X + ivec,
                N,
                reinterpret_cast<const dataTypes::numberGPU *>(
                  &scalarCoeffBeta),
                reinterpret_cast<dataTypes::numberGPU *>(
                  thrust::raw_pointer_cast(&overlapMatrixBlockDP[0])),
                B);

              const unsigned int DRem = D - B;

              if (DRem != 0)
                {
                  cublasXgemm(
                    handle,
                    CUBLAS_OP_N,
                    std::is_same<dataTypes::number,
                                 std::complex<double>>::value ?
                      CUBLAS_OP_C :
                      CUBLAS_OP_T,
                    DRem,
                    B,
                    M,
                    reinterpret_cast<const dataTypes::numberFP32GPU *>(
                      &scalarCoeffAlphaSP),
                    reinterpret_cast<const dataTypes::numberFP32GPU *>(
                      thrust::raw_pointer_cast(&XSP[0]) + ivec + B),
                    N,
                    reinterpret_cast<const dataTypes::numberFP32GPU *>(
                      thrust::raw_pointer_cast(&XSP[0]) + ivec),
                    N,
                    reinterpret_cast<const dataTypes::numberFP32GPU *>(
                      &scalarCoeffBetaSP),
                    reinterpret_cast<dataTypes::numberFP32GPU *>(
                      thrust::raw_pointer_cast(&overlapMatrixBlockSP[0])),
                    DRem);
                }

              if (dftParameters::useGPUDirectAllReduce)
                {
                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    gpucclMpiCommDomain.gpuDirectAllReduceMixedPrecGroupWrapper(
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlockDP[0])),
                      reinterpret_cast<dataTypes::numberFP32GPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlockSP[0])),
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlockDP[0])),
                      reinterpret_cast<dataTypes::numberFP32GPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlockSP[0])),
                      B * B,
                      DRem * B,
                      tempReal,
                      tempRealFP32,
                      tempImag,
                      tempImagFP32,
                      streamGPUCCL);
                  else
                    gpucclMpiCommDomain.gpuDirectAllReduceMixedPrecGroupWrapper(
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlockDP[0])),
                      reinterpret_cast<dataTypes::numberFP32GPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlockSP[0])),
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlockDP[0])),
                      reinterpret_cast<dataTypes::numberFP32GPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlockSP[0])),
                      B * B,
                      DRem * B,
                      streamGPUCCL);
                }

              cudaMemcpy(reinterpret_cast<dataTypes::numberGPU *>(
                           overlapMatrixBlockHostDP),
                         reinterpret_cast<const dataTypes::numberGPU *>(
                           thrust::raw_pointer_cast(&overlapMatrixBlockDP[0])),
                         B * B * sizeof(dataTypes::numberGPU),
                         cudaMemcpyDeviceToHost);

              cudaMemcpy(reinterpret_cast<dataTypes::numberFP32GPU *>(
                           overlapMatrixBlockHostSP),
                         reinterpret_cast<const dataTypes::numberFP32GPU *>(
                           thrust::raw_pointer_cast(&overlapMatrixBlockSP[0])),
                         DRem * B * sizeof(dataTypes::numberFP32GPU),
                         cudaMemcpyDeviceToHost);

              if (!dftParameters::useGPUDirectAllReduce)
                {
                  // Sum local XTrunc^{T}*XcBlock for double precision across
                  // domain decomposition processors
                  MPI_Allreduce(MPI_IN_PLACE,
                                overlapMatrixBlockHostDP,
                                B * B,
                                dataTypes::mpi_type_id(
                                  overlapMatrixBlockHostDP),
                                MPI_SUM,
                                mpiCommDomain);

                  // Sum local XTrunc^{T}*XcBlock for single precision across
                  // domain decomposition processors
                  MPI_Allreduce(MPI_IN_PLACE,
                                overlapMatrixBlockHostSP,
                                DRem * B,
                                dataTypes::mpi_type_id(
                                  overlapMatrixBlockHostSP),
                                MPI_SUM,
                                mpiCommDomain);
                }

              // Copying only the lower triangular part to the ScaLAPACK overlap
              // matrix
              if (processGrid->is_process_active())
                for (unsigned int i = 0; i < B; ++i)
                  if (globalToLocalColumnIdMap.find(i + ivec) !=
                      globalToLocalColumnIdMap.end())
                    {
                      const unsigned int localColumnId =
                        globalToLocalColumnIdMap[i + ivec];
                      for (unsigned int j = ivec + i; j < ivec + B; ++j)
                        {
                          std::map<unsigned int, unsigned int>::iterator it =
                            globalToLocalRowIdMap.find(j);
                          if (it != globalToLocalRowIdMap.end())
                            overlapMatPar.local_el(it->second, localColumnId) =
                              overlapMatrixBlockHostDP[i * B + j - ivec];
                        }

                      for (unsigned int j = ivec + B; j < N; ++j)
                        {
                          std::map<unsigned int, unsigned int>::iterator it =
                            globalToLocalRowIdMap.find(j);
                          if (it != globalToLocalRowIdMap.end())
                            overlapMatPar.local_el(it->second, localColumnId) =
                              overlapMatrixBlockHostSP[i * DRem + j - ivec - B];
                        }
                    }
            } // band parallelization
        }     // end block loop

      CUDACHECK(cudaFreeHost(overlapMatrixBlockHostDP));
      CUDACHECK(cudaFreeHost(overlapMatrixBlockHostSP));

      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          CUDACHECK(cudaFree(tempReal));
          CUDACHECK(cudaFree(tempImag));
          CUDACHECK(cudaFree(tempRealFP32));
          CUDACHECK(cudaFree(tempImagFP32));
        }

      cudaStreamDestroy(streamGPUCCL);

      if (numberBandGroups > 1)
        linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
          processGrid, overlapMatPar, interBandGroupComm);
    }


    /////////////PSEUDO CODE for the implementation below for Overlapping
    /// compute and communication in the computation of overlap matrix using
    /// mixed precision arithmetic/////////////////
    //
    // In the algorithm below the communication and computation of two
    // consecutive blocks of wavefunctions: block i and block i+1 are
    // overlapped.
    // ----------------------------------------------------------
    // CMP denotes computuation of X^{T} times XBlock
    // COP denotes GPU->CPU copy of X^{T} times XBlock
    // COM denotes blocking MPI_Allreduce on X^{T}XBlock and copy to scalapack
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

    void
    fillParallelOverlapMatMixedPrecScalapackAsyncComputeCommun(
      const dataTypes::numberGPU *                     X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const MPI_Comm &                                 mpiCommDomain,
      GPUCCLWrapper &                                  gpucclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar)
    {
      // get global to local index maps for Scalapack matrix
      std::map<unsigned int, unsigned int> globalToLocalColumnIdMap;
      std::map<unsigned int, unsigned int> globalToLocalRowIdMap;
      linearAlgebraOperations::internal::createGlobalToLocalIdMapsScaLAPACKMat(
        processGrid,
        overlapMatPar,
        globalToLocalRowIdMap,
        globalToLocalColumnIdMap);

      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize =
        std::min(dftParameters::wfcBlockSize, N);
      const unsigned int numberBlocks = N / vectorsBlockSize;

      // create separate CUDA streams for GPU->CPU copy and computation
      cudaStream_t streamCompute, streamDataMove;
      cudaStreamCreate(&streamCompute);
      cudaStreamCreate(&streamDataMove);

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
          cudaEventCreate(&computeEvents[i]);
          cudaEventCreate(&copyEvents[i]);
        }

      thrust::device_vector<dataTypes::numberFP32ThrustGPU>
                                                        overlapMatrixBlockSP(N * vectorsBlockSize,
                             dataTypes::numberFP32ThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberThrustGPU> overlapMatrixBlockDP(
        vectorsBlockSize * vectorsBlockSize, dataTypes::numberThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberFP32ThrustGPU>
        overlapMatrixBlockSPNext(N * vectorsBlockSize,
                                 dataTypes::numberFP32ThrustGPU(0.0));
      thrust::device_vector<dataTypes::numberThrustGPU>
        overlapMatrixBlockDPNext(vectorsBlockSize * vectorsBlockSize,
                                 dataTypes::numberThrustGPU(0.0));

      const unsigned int MPadded = std::ceil(M * 1.0 / 8.0) * 8.0 + 0.5;
      thrust::device_vector<dataTypes::numberFP32ThrustGPU> XSP(
        MPadded * N, dataTypes::numberFP32ThrustGPU(0.0));

      convDoubleArrToFloatArr<<<(N + 255) / 256 * M, 256>>>(
        N * M,
        X,
        reinterpret_cast<dataTypes::numberFP32GPU *>(
          thrust::raw_pointer_cast(&XSP[0])));
      dataTypes::number *overlapMatrixBlockHostDP;
      CUDACHECK(cudaMallocHost((void **)&overlapMatrixBlockHostDP,
                               vectorsBlockSize * vectorsBlockSize *
                                 sizeof(dataTypes::number)));
      std::memset(overlapMatrixBlockHostDP,
                  0,
                  vectorsBlockSize * vectorsBlockSize *
                    sizeof(dataTypes::number));

      dataTypes::numberFP32 *overlapMatrixBlockHostSP;
      CUDACHECK(
        cudaMallocHost((void **)&overlapMatrixBlockHostSP,
                       N * vectorsBlockSize * sizeof(dataTypes::numberFP32)));
      std::memset(overlapMatrixBlockHostSP,
                  0,
                  N * vectorsBlockSize * sizeof(dataTypes::numberFP32));

      const dataTypes::number     scalarCoeffAlpha = dataTypes::number(1.0);
      const dataTypes::number     scalarCoeffBeta  = dataTypes::number(0.0);
      const dataTypes::numberFP32 scalarCoeffAlphaSP =
        dataTypes::numberFP32(1.0);
      const dataTypes::numberFP32 scalarCoeffBetaSP =
        dataTypes::numberFP32(0.0);

      dataTypes::numberValueType *    tempReal;
      dataTypes::numberValueType *    tempImag;
      dataTypes::numberFP32ValueType *tempRealFP32;
      dataTypes::numberFP32ValueType *tempImagFP32;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          CUDACHECK(cudaMalloc((void **)&tempReal,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberValueType)));
          CUDACHECK(cudaMalloc((void **)&tempImag,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberValueType)));
          CUDACHECK(cudaMalloc((void **)&tempRealFP32,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberFP32ValueType)));
          CUDACHECK(cudaMalloc((void **)&tempImagFP32,
                               vectorsBlockSize * N *
                                 sizeof(dataTypes::numberFP32ValueType)));
        }

      unsigned int blockCount = 0;
      for (unsigned int ivec = 0; ivec < N; ivec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int B = std::min(vectorsBlockSize, N - ivec);
          const unsigned int D = N - ivec;

          if ((ivec + B) <=
                bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
              (ivec + B) > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
            {
              // Compute local XTrunc^{T}*XcBlock
              if (ivec == bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId])
                {
                  // thrust::fill(overlapMatrixBlockDP.begin(),overlapMatrixBlockDP.end(),0.0);

                  cublasXgemm(handle,
                              CUBLAS_OP_N,
                              std::is_same<dataTypes::number,
                                           std::complex<double>>::value ?
                                CUBLAS_OP_C :
                                CUBLAS_OP_T,
                              B,
                              B,
                              M,
                              reinterpret_cast<const dataTypes::numberGPU *>(
                                &scalarCoeffAlpha),
                              X + ivec,
                              N,
                              X + ivec,
                              N,
                              reinterpret_cast<const dataTypes::numberGPU *>(
                                &scalarCoeffBeta),
                              reinterpret_cast<dataTypes::numberGPU *>(
                                thrust::raw_pointer_cast(
                                  &overlapMatrixBlockDP[0])),
                              B);

                  const unsigned int DRem = D - B;

                  if (DRem != 0)
                    {
                      // thrust::fill(overlapMatrixBlockSP.begin(),overlapMatrixBlockSP.end(),0.0);

                      cublasXgemm(
                        handle,
                        CUBLAS_OP_N,
                        std::is_same<dataTypes::number,
                                     std::complex<double>>::value ?
                          CUBLAS_OP_C :
                          CUBLAS_OP_T,
                        DRem,
                        B,
                        M,
                        reinterpret_cast<const dataTypes::numberFP32GPU *>(
                          &scalarCoeffAlphaSP),
                        reinterpret_cast<const dataTypes::numberFP32GPU *>(
                          thrust::raw_pointer_cast(&XSP[0]) + ivec + B),
                        N,
                        reinterpret_cast<const dataTypes::numberFP32GPU *>(
                          thrust::raw_pointer_cast(&XSP[0]) + ivec),
                        N,
                        reinterpret_cast<const dataTypes::numberFP32GPU *>(
                          &scalarCoeffBetaSP),
                        reinterpret_cast<dataTypes::numberFP32GPU *>(
                          thrust::raw_pointer_cast(&overlapMatrixBlockSP[0])),
                        DRem);
                    }

                  // record completion of compute for first block
                  CUDACHECK(
                    cudaEventRecord(computeEvents[blockCount], streamCompute));
                }

              // Before swap host thread needs to wait till compute on
              // currentblock is over. Since swap occurs on the null stream, any
              // future operations in the streamDataMove will only occur after
              // both the compute on currentblock and swap is over. Note that at
              // this point there is nothing queued in the streamDataMove as all
              // previous operations in that stream are over.
              if ((cudaEventSynchronize(computeEvents[blockCount]) ==
                   cudaSuccess) &&
                  (ivec > bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]))
                {
                  overlapMatrixBlockDP.swap(overlapMatrixBlockDPNext);
                  overlapMatrixBlockSP.swap(overlapMatrixBlockSPNext);
                }

              const unsigned int DRem = D - B;

              const unsigned int ivecNew = ivec + vectorsBlockSize;
              const unsigned int DNew    = N - ivecNew;
              const unsigned int BNew    = min(vectorsBlockSize, N - ivecNew);

              if (ivecNew <
                  bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1])
                {
                  // thrust::fill(overlapMatrixBlockDPNext.begin(),overlapMatrixBlockDPNext.end(),0.0);

                  // evaluate X^{T} times XBlock
                  cublasXgemm(handle,
                              CUBLAS_OP_N,
                              std::is_same<dataTypes::number,
                                           std::complex<double>>::value ?
                                CUBLAS_OP_C :
                                CUBLAS_OP_T,
                              BNew,
                              BNew,
                              M,
                              reinterpret_cast<const dataTypes::numberGPU *>(
                                &scalarCoeffAlpha),
                              X + ivecNew,
                              N,
                              X + ivecNew,
                              N,
                              reinterpret_cast<const dataTypes::numberGPU *>(
                                &scalarCoeffBeta),
                              reinterpret_cast<dataTypes::numberGPU *>(
                                thrust::raw_pointer_cast(
                                  &overlapMatrixBlockDPNext[0])),
                              BNew);

                  const unsigned int DRemNew = DNew - BNew;

                  if (DRemNew != 0)
                    {
                      // thrust::fill(overlapMatrixBlockSPNext.begin(),overlapMatrixBlockSPNext.end(),0.0);

                      cublasXgemm(
                        handle,
                        CUBLAS_OP_N,
                        std::is_same<dataTypes::number,
                                     std::complex<double>>::value ?
                          CUBLAS_OP_C :
                          CUBLAS_OP_T,
                        DRemNew,
                        BNew,
                        M,
                        reinterpret_cast<const dataTypes::numberFP32GPU *>(
                          &scalarCoeffAlphaSP),
                        reinterpret_cast<const dataTypes::numberFP32GPU *>(
                          thrust::raw_pointer_cast(&XSP[0]) + ivecNew + BNew),
                        N,
                        reinterpret_cast<const dataTypes::numberFP32GPU *>(
                          thrust::raw_pointer_cast(&XSP[0]) + ivecNew),
                        N,
                        reinterpret_cast<const dataTypes::numberFP32GPU *>(
                          &scalarCoeffBetaSP),
                        reinterpret_cast<dataTypes::numberFP32GPU *>(
                          thrust::raw_pointer_cast(
                            &overlapMatrixBlockSPNext[0])),
                        DRemNew);
                    }

                  // record completion of compute for next block
                  CUDACHECK(cudaEventRecord(computeEvents[blockCount + 1],
                                            streamCompute));
                }

              if (dftParameters::useGPUDirectAllReduce)
                {
                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    gpucclMpiCommDomain.gpuDirectAllReduceMixedPrecGroupWrapper(
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlockDP[0])),
                      reinterpret_cast<dataTypes::numberFP32GPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlockSP[0])),
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlockDP[0])),
                      reinterpret_cast<dataTypes::numberFP32GPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlockSP[0])),
                      B * B,
                      DRem * B,
                      tempReal,
                      tempRealFP32,
                      tempImag,
                      tempImagFP32,
                      streamDataMove);
                  else
                    gpucclMpiCommDomain.gpuDirectAllReduceMixedPrecGroupWrapper(
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlockDP[0])),
                      reinterpret_cast<dataTypes::numberFP32GPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlockSP[0])),
                      reinterpret_cast<dataTypes::numberGPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlockDP[0])),
                      reinterpret_cast<dataTypes::numberFP32GPU *>(
                        thrust::raw_pointer_cast(&overlapMatrixBlockSP[0])),
                      B * B,
                      DRem * B,
                      streamDataMove);
                }

              cudaMemcpyAsync(reinterpret_cast<dataTypes::numberGPU *>(
                                overlapMatrixBlockHostDP),
                              reinterpret_cast<const dataTypes::numberGPU *>(
                                thrust::raw_pointer_cast(
                                  &overlapMatrixBlockDP[0])),
                              B * B * sizeof(dataTypes::numberGPU),
                              cudaMemcpyDeviceToHost,
                              streamDataMove);

              cudaMemcpyAsync(
                reinterpret_cast<dataTypes::numberFP32GPU *>(
                  overlapMatrixBlockHostSP),
                reinterpret_cast<const dataTypes::numberFP32GPU *>(
                  thrust::raw_pointer_cast(&overlapMatrixBlockSP[0])),
                DRem * B * sizeof(dataTypes::numberFP32GPU),
                cudaMemcpyDeviceToHost,
                streamDataMove);

              // record completion of GPU->CPU copy for current block
              cudaEventRecord(copyEvents[blockCount], streamDataMove);

              // Check that GPU->CPU on the current block has been completed. If
              // completed, perform blocking MPI commmunication on the current
              // block and copy to ScaLAPACK matri
              if (cudaEventSynchronize(copyEvents[blockCount]) == cudaSuccess)
                {
                  const unsigned int DRem = D - B;

                  if (!dftParameters::useGPUDirectAllReduce)
                    {
                      // Sum local XTrunc^{T}*XcBlock for double precision
                      // across domain decomposition processors
                      MPI_Allreduce(MPI_IN_PLACE,
                                    overlapMatrixBlockHostDP,
                                    B * B,
                                    dataTypes::mpi_type_id(
                                      overlapMatrixBlockHostDP),
                                    MPI_SUM,
                                    mpiCommDomain);

                      // Sum local XTrunc^{T}*XcBlock for single precision
                      // across domain decomposition processors
                      MPI_Allreduce(MPI_IN_PLACE,
                                    overlapMatrixBlockHostSP,
                                    DRem * B,
                                    dataTypes::mpi_type_id(
                                      overlapMatrixBlockHostSP),
                                    MPI_SUM,
                                    mpiCommDomain);
                    }

                  // Copying only the lower triangular part to the ScaLAPACK
                  // overlap matrix
                  if (processGrid->is_process_active())
                    for (unsigned int i = 0; i < B; ++i)
                      if (globalToLocalColumnIdMap.find(i + ivec) !=
                          globalToLocalColumnIdMap.end())
                        {
                          const unsigned int localColumnId =
                            globalToLocalColumnIdMap[i + ivec];
                          for (unsigned int j = ivec + i; j < ivec + B; ++j)
                            {
                              std::map<unsigned int, unsigned int>::iterator
                                it = globalToLocalRowIdMap.find(j);
                              if (it != globalToLocalRowIdMap.end())
                                overlapMatPar.local_el(it->second,
                                                       localColumnId) =
                                  overlapMatrixBlockHostDP[i * B + j - ivec];
                            }

                          for (unsigned int j = ivec + B; j < N; ++j)
                            {
                              std::map<unsigned int, unsigned int>::iterator
                                it = globalToLocalRowIdMap.find(j);
                              if (it != globalToLocalRowIdMap.end())
                                overlapMatPar.local_el(it->second,
                                                       localColumnId) =
                                  overlapMatrixBlockHostSP[i * DRem + j - ivec -
                                                           B];
                            }
                        }
                }
            } // band parallelization

          blockCount += 1;

        } // end block loop

      CUDACHECK(cudaFreeHost(overlapMatrixBlockHostDP));
      CUDACHECK(cudaFreeHost(overlapMatrixBlockHostSP));

      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          CUDACHECK(cudaFree(tempReal));
          CUDACHECK(cudaFree(tempImag));
          CUDACHECK(cudaFree(tempRealFP32));
          CUDACHECK(cudaFree(tempImagFP32));
        }

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
          MPI_Barrier(interBandGroupComm);

          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, overlapMatPar, interBandGroupComm);
        }
    }



    void
    computeEigenResidualNorm(
      operatorDFTCUDAClass &                   operatorMatrix,
      dataTypes::numberGPU *                   X,
      distributedGPUVec<dataTypes::numberGPU> &XBlock,
      distributedGPUVec<dataTypes::numberGPU> &HXBlock,
      distributedGPUVec<dataTypes::numberGPU> &projectorKetTimesVector,
      const unsigned int                       M,
      const unsigned int                       N,
      const std::vector<double> &              eigenValues,
      const MPI_Comm &                         mpiCommDomain,
      const MPI_Comm &                         interBandGroupComm,
      cublasHandle_t &                         handle,
      std::vector<double> &                    residualNorm,
      const bool                               useBandParal)
    {
      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);


      const unsigned int vectorsBlockSize = dftParameters::wfcBlockSize;
      thrust::device_vector<double> residualNormSquareDevice(N, 0.0);
      thrust::device_vector<dataTypes::numberThrustGPU> HXBlockFull(
        vectorsBlockSize * M, dataTypes::numberThrustGPU(0.0));
      thrust::device_vector<double> residualSqDevice(vectorsBlockSize * M, 0.0);
      thrust::device_vector<double> onesVecDevice(M, 1.0);


      thrust::device_vector<double> eigenValuesDevice(N, 0.0);
      cudaMemcpy(thrust::raw_pointer_cast(&eigenValuesDevice[0]),
                 &eigenValues[0],
                 N * sizeof(double),
                 cudaMemcpyHostToDevice);

      const bool   scaleFlag = false;
      const double scalar    = 1.0;
      const double alpha = 1.0, beta = 0.0;

      for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int B = std::min(vectorsBlockSize, N - jvec);


          if (((jvec + B) <=
                 bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId + 1] &&
               (jvec + B) >
                 bandGroupLowHighPlusOneIndices[2 * bandGroupTaskId]) ||
              !useBandParal)
            {
              const unsigned int chebyBlockSize =
                std::min(dftParameters::chebyWfcBlockSize, N);

              for (unsigned int k = jvec; k < jvec + B; k += chebyBlockSize)
                {
                  stridedCopyToBlockKernel<<<(chebyBlockSize + 255) / 256 * M,
                                             256>>>(
                    chebyBlockSize, M, X, N, XBlock.begin(), k);

                  // evaluate H times XBlock^{T} and store in HXBlock^{T}
                  HXBlock.setZero();
                  operatorMatrix.HX(XBlock,
                                    projectorKetTimesVector,
                                    M,
                                    chebyBlockSize,
                                    scaleFlag,
                                    scalar,
                                    HXBlock);

                  stridedCopyFromBlockKernel<<<(chebyBlockSize + 255) / 256 * M,
                                               256>>>(
                    chebyBlockSize,
                    M,
                    HXBlock.begin(),
                    B,
                    reinterpret_cast<dataTypes::numberGPU *>(
                      thrust::raw_pointer_cast(&HXBlockFull[0])),
                    k - jvec);
                }

              computeResidualCUDAKernel<<<(B + 255) / 256 * M, 256>>>(
                B,
                M,
                N,
                jvec,
                thrust::raw_pointer_cast(&eigenValuesDevice[0]),
                X,
                reinterpret_cast<const dataTypes::numberGPU *>(
                  thrust::raw_pointer_cast(&HXBlockFull[0])),
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
                          thrust::raw_pointer_cast(
                            &residualNormSquareDevice[0] + jvec),
                          1);
            }
        }


      cudaMemcpy(&residualNorm[0],
                 thrust::raw_pointer_cast(&residualNormSquareDevice[0]),
                 N * sizeof(double),
                 cudaMemcpyDeviceToHost);

      MPI_Allreduce(
        MPI_IN_PLACE, &residualNorm[0], N, MPI_DOUBLE, MPI_SUM, mpiCommDomain);

      if (numberBandGroups > 1 || !useBandParal)
        MPI_Allreduce(MPI_IN_PLACE,
                      &residualNorm[0],
                      N,
                      MPI_DOUBLE,
                      MPI_SUM,
                      interBandGroupComm);


      for (unsigned int iWave = 0; iWave < N; ++iWave)
        residualNorm[iWave] = std::sqrt(residualNorm[iWave]);
    }
  } // namespace linearAlgebraOperationsCUDA
} // namespace dftfe
