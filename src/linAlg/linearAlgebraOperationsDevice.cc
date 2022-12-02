// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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


#include <deviceHelpers.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <MemoryStorage.h>
#include <dftUtils.h>
#include <linearAlgebraOperationsDevice.h>
#include <linearAlgebraOperationsInternal.h>
#include <nvToolsExt.h>
#include <vectorUtilities.h>

namespace dftfe
{
  namespace linearAlgebraOperationsDevice
  {
    namespace
    {
      __global__ void
      scaleDeviceKernel(const unsigned int contiguousBlockSize,
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
      scaleDeviceKernel(const unsigned int contiguousBlockSize,
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
                       (*(scalingVector + gangBlockId) * scalar), 0));
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
      dscalDeviceKernel(const int n, double *y, const double a)
      {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x)
          y[i] = a * y[i];
      }

      // y=a*y, with inc=1
      __global__ void
      dscalDeviceKernel(const int n, cuDoubleComplex *Y, const double a)
      {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x)
          Y[i] = make_cuDoubleComplex(a * Y[i].x, a * Y[i].y);
      }

      // y=a*x+b*y, with inc=1
      __global__ void
      daxpyDeviceKernel(const int n, const double *x, double *y, const double a)
      {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
             i += blockDim.x * gridDim.x)
          y[i] = a * x[i] + y[i];
      }

      // y=a*x+b*y, with inc=1
      __global__ void
      daxpyDeviceKernel(const int              n,
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
      daxpbyDeviceKernel(const int     n,
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
      daxpbyDeviceKernel(const int              n,
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
      combinedDeviceKernel(const unsigned int contiguousBlockSize,
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
      combinedDeviceKernel(const unsigned int contiguousBlockSize,
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
      scaleXArrayRayleighQuotientsDeviceKernel(
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
      addScaleXArrayRayleighQuotientsDeviceKernel(
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
      computeResidualDeviceKernel(const unsigned int numVectors,
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
      computeResidualDeviceKernel(const unsigned int     numVectors,
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
        dotProductContributionBlockedKernelMassVector<<<
          (numberVectors + (deviceConstants::blockSize - 1)) /
            deviceConstants::blockSize * localSize,
          deviceConstants::blockSize>>>(
          numberVectors, localSize, xarray, yarray, sqrtMassVector, temparray);

        const double alpha = 1.0, beta = 0;
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


        dftfe::utils::deviceMemcpyD2H(dotarrayH,
                   dotarrayD,
                   2 * numberVectors * sizeof(double));



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
      operatorDFTDeviceClass &                       operatorMatrix,
      distributedDeviceVec<dataTypes::numberDevice> &Xb,
      distributedDeviceVec<dataTypes::numberDevice> &Yb,
      distributedDeviceVec<dataTypes::numberDevice> &projectorKetTimesVector,
      const unsigned int                             blockSize,
      const dftParameters &                          dftParams)
    {
      const unsigned int this_mpi_process =
        dealii::Utilities::MPI::this_mpi_process(
          operatorMatrix.getMPICommunicator());

      const unsigned int lanczosIterations =
        dftParams.reproducible_output ? 40 : 20;
      double beta;


      dataTypes::number alpha, alphaNeg;

      //
      // generate random vector v
      //
      distributedCPUVec<dataTypes::number> vVector, fVector, v0Vector;
      vVector.reinit(operatorMatrix.getParallelVecSingleComponent());
      fVector.reinit(operatorMatrix.getParallelVecSingleComponent());

      vVector = dataTypes::number(0), fVector = dataTypes::number(0);
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

      dftfe::utils::deviceMemcpyH2D_2D(Xb.begin(),
                               blockSize * sizeof(dataTypes::number),
                               vvec.begin(),
                               1 * sizeof(dataTypes::number),
                               1 * sizeof(dataTypes::number),
                               local_size);

      Yb.setZero();
      operatorMatrix.HX(
        Xb, projectorKetTimesVector, local_size, blockSize, false, 1.0, Yb);

      distributedCPUVec<dataTypes::number> &fvec = f[0];
      dftfe::utils::deviceMemcpyD2H_2D(fvec.begin(),
                               1 * sizeof(dataTypes::number),
                               Yb.begin(),
                               blockSize * sizeof(dataTypes::number),
                               1 * sizeof(dataTypes::number),
                               local_size);

      operatorMatrix.getOverloadedConstraintMatrixHost()->set_zero(v[0], 1);
      fVector = f[0];

      alpha = fVector * vVector;
      fVector.add(-1.0 * alpha, vVector);
      std::vector<dataTypes::number> Tlanczos(lanczosIterations *
                                                lanczosIterations,
                                              0);

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
          dftfe::utils::deviceMemcpyH2D_2D(Xb.begin(),
                                   blockSize * sizeof(dataTypes::number),
                                   vvec.begin(),
                                   1 * sizeof(dataTypes::number),
                                   1 * sizeof(dataTypes::number),
                                   local_size);

          Yb.setZero();
          operatorMatrix.HX(
            Xb, projectorKetTimesVector, local_size, blockSize, false, 1.0, Yb);

          distributedCPUVec<dataTypes::number> &fvec = f[0];
          dftfe::utils::deviceMemcpyD2H_2D(fvec.begin(),
                                   1 * sizeof(dataTypes::number),
                                   Yb.begin(),
                                   blockSize * sizeof(dataTypes::number),
                                   1 * sizeof(dataTypes::number),
                                   local_size);

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
      std::vector<double>               rwork(lrwork, 0);
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
      std::vector<double> work(lwork, 0);
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
      if (dftParams.verbosity >= 5 && this_mpi_process == 0)
        {
          std::cout << "bUp1: " << eigenValuesT[lanczosIterations - 1]
                    << ", fvector norm: " << fvectorNorm << std::endl;
          std::cout << "aLow: " << eigenValuesT[0] << std::endl;
        }

      double lowerBound = std::floor(eigenValuesT[0]);
      double upperBound = std::ceil(
        eigenValuesT[lanczosIterations - 1] +
        (dftParams.reproducible_output ? fvectorNorm : fvectorNorm / 10));
      return (std::make_pair(lowerBound, upperBound));
    }



    void
    chebyshevFilter(
      operatorDFTDeviceClass &                           operatorMatrix,
      distributedDeviceVec<dataTypes::numberDevice> &    XArray,
      distributedDeviceVec<dataTypes::numberDevice> &    YArray,
      distributedDeviceVec<dataTypes::numberFP32Device> &tempFloatArray,
      distributedDeviceVec<dataTypes::numberDevice> &projectorKetTimesVector,
      const unsigned int                             localVectorSize,
      const unsigned int                             numberVectors,
      const unsigned int                             m,
      const double                                   a,
      const double                                   b,
      const double                                   a0,
      const bool                                     mixedPrecOverall,
      const dftParameters &                          dftParams)
    {
      double e, c, sigma, sigma1, sigma2, gamma, device_time;
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

      daxpyDeviceKernel<<<min((totalVectorSize +
                               (deviceConstants::blockSize - 1)) /
                                deviceConstants::blockSize,
                              30000),
                          deviceConstants::blockSize>>>(totalVectorSize,
                                                        XArray.begin(),
                                                        YArray.begin(),
                                                        alpha2);

      /*
      cublasDscal(operatorMatrix.getCublasHandle(),
                  totalVectorSize,
                  &alpha1,
                  YArray.begin(),
                  inc);
      */

      dscalDeviceKernel<<<
        min((totalVectorSize + (deviceConstants::blockSize - 1)) /
              deviceConstants::blockSize,
            30000),
        deviceConstants::blockSize>>>(totalVectorSize, YArray.begin(), alpha1);

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
              daxpbyDeviceKernel<<<min((totalVectorSize +
                                        (deviceConstants::blockSize - 1)) /
                                         deviceConstants::blockSize,
                                       30000),
                                   deviceConstants::blockSize>>>(
                totalVectorSize, YArray.begin(), XArray.begin(), coeff, alpha2);


              // scale src vector with M^{-1/2}
              //
              scaleDeviceKernel<<<
                (numberVectors + (deviceConstants::blockSize - 1)) /
                  deviceConstants::blockSize * localVectorSize,
                deviceConstants::blockSize>>>(
                numberVectors,
                localVectorSize,
                alpha1,
                YArray.begin(),
                operatorMatrix.getInvSqrtMassVec());

              scaleDeviceKernel<<<
                (numberVectors + (deviceConstants::blockSize - 1)) /
                  deviceConstants::blockSize * localVectorSize,
                deviceConstants::blockSize>>>(numberVectors,
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
                                       dftParams.useMixedPrecCheby);
            }
          else if (degree == m)
            {
              // unscale src vector with M^{1/2}
              //
              scaleDeviceKernel<<<
                (numberVectors + (deviceConstants::blockSize - 1)) /
                  deviceConstants::blockSize * localVectorSize,
                deviceConstants::blockSize>>>(numberVectors,
                                              localVectorSize,
                                              1.0 / alpha1Old,
                                              XArray.begin(),
                                              operatorMatrix.getSqrtMassVec());

              scaleDeviceKernel<<<
                (numberVectors + (deviceConstants::blockSize - 1)) /
                  deviceConstants::blockSize * localVectorSize,
                deviceConstants::blockSize>>>(
                numberVectors,
                localVectorSize,
                1.0,
                YArray.begin(),
                operatorMatrix.getInvSqrtMassVec());

              daxpbyDeviceKernel<<<min((totalVectorSize +
                                        (deviceConstants::blockSize - 1)) /
                                         deviceConstants::blockSize,
                                       30000),
                                   deviceConstants::blockSize>>>(
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
              combinedDeviceKernel<<<min((totalVectorSize +
                                          (deviceConstants::blockSize - 1)) /
                                           deviceConstants::blockSize,
                                         30000),
                                     deviceConstants::blockSize>>>(
                numberVectors,
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
                                       dftParams.useMixedPrecCheby);
            }

          XArray.swap(YArray);


          sigma     = sigma2;
          alpha1Old = alpha1;
        }

      // copy back YArray to XArray
      dftfe::utils::deviceMemcpyD2D(XArray.begin(),
                 YArray.begin(),
                 totalVectorSize * sizeof(dataTypes::number));
    }


    //
    // Compute and comunication of two blocks (1) and (2) are overlapped during
    // chebyshev filtering.
    //
    void
    chebyshevFilter(
      operatorDFTDeviceClass &                           operatorMatrix,
      distributedDeviceVec<dataTypes::numberDevice> &    XArray1,
      distributedDeviceVec<dataTypes::numberDevice> &    YArray1,
      distributedDeviceVec<dataTypes::numberFP32Device> &tempFloatArray,
      distributedDeviceVec<dataTypes::numberDevice> &projectorKetTimesVector1,
      distributedDeviceVec<dataTypes::numberDevice> &XArray2,
      distributedDeviceVec<dataTypes::numberDevice> &YArray2,
      distributedDeviceVec<dataTypes::numberDevice> &projectorKetTimesVector2,
      const unsigned int                             localVectorSize,
      const unsigned int                             numberVectors,
      const unsigned int                             m,
      const double                                   a,
      const double                                   b,
      const double                                   a0,
      const bool                                     mixedPrecOverall,
      const dftParameters &                          dftParams)
    {
      double e, c, sigma, sigma1, sigma2, gamma, device_time;
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

      daxpyDeviceKernel<<<min((totalVectorSize +
                               (deviceConstants::blockSize - 1)) /
                                deviceConstants::blockSize,
                              30000),
                          deviceConstants::blockSize>>>(totalVectorSize,
                                                        XArray1.begin(),
                                                        YArray1.begin(),
                                                        alpha2);

      dscalDeviceKernel<<<
        min((totalVectorSize + (deviceConstants::blockSize - 1)) /
              deviceConstants::blockSize,
            30000),
        deviceConstants::blockSize>>>(totalVectorSize, YArray1.begin(), alpha1);

      daxpyDeviceKernel<<<min((totalVectorSize +
                               (deviceConstants::blockSize - 1)) /
                                deviceConstants::blockSize,
                              30000),
                          deviceConstants::blockSize>>>(totalVectorSize,
                                                        XArray2.begin(),
                                                        YArray2.begin(),
                                                        alpha2);

      dscalDeviceKernel<<<
        min((totalVectorSize + (deviceConstants::blockSize - 1)) /
              deviceConstants::blockSize,
            30000),
        deviceConstants::blockSize>>>(totalVectorSize, YArray2.begin(), alpha1);

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
              daxpbyDeviceKernel<<<
                min((totalVectorSize + (deviceConstants::blockSize - 1)) /
                      deviceConstants::blockSize,
                    30000),
                deviceConstants::blockSize>>>(totalVectorSize,
                                              YArray1.begin(),
                                              XArray1.begin(),
                                              coeff,
                                              alpha2);


              // scale src vector with M^{-1/2}
              //
              scaleDeviceKernel<<<
                (numberVectors + (deviceConstants::blockSize - 1)) /
                  deviceConstants::blockSize * localVectorSize,
                deviceConstants::blockSize>>>(
                numberVectors,
                localVectorSize,
                alpha1,
                YArray1.begin(),
                operatorMatrix.getInvSqrtMassVec());

              scaleDeviceKernel<<<
                (numberVectors + (deviceConstants::blockSize - 1)) /
                  deviceConstants::blockSize * localVectorSize,
                deviceConstants::blockSize>>>(numberVectors,
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
                                       dftParams.useMixedPrecCheby);

              daxpbyDeviceKernel<<<
                min((totalVectorSize + (deviceConstants::blockSize - 1)) /
                      deviceConstants::blockSize,
                    30000),
                deviceConstants::blockSize>>>(totalVectorSize,
                                              YArray2.begin(),
                                              XArray2.begin(),
                                              coeff,
                                              alpha2);


              // scale src vector with M^{-1/2}
              //
              scaleDeviceKernel<<<
                (numberVectors + (deviceConstants::blockSize - 1)) /
                  deviceConstants::blockSize * localVectorSize,
                deviceConstants::blockSize>>>(
                numberVectors,
                localVectorSize,
                alpha1,
                YArray2.begin(),
                operatorMatrix.getInvSqrtMassVec());

              scaleDeviceKernel<<<
                (numberVectors + (deviceConstants::blockSize - 1)) /
                  deviceConstants::blockSize * localVectorSize,
                deviceConstants::blockSize>>>(numberVectors,
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
                                       dftParams.useMixedPrecCheby);
              overlap = false;
            }
          else if (degree == m)
            {
              // unscale src vector with M^{1/2}
              //
              scaleDeviceKernel<<<
                (numberVectors + (deviceConstants::blockSize - 1)) /
                  deviceConstants::blockSize * localVectorSize,
                deviceConstants::blockSize>>>(numberVectors,
                                              localVectorSize,
                                              1.0 / alpha1Old,
                                              XArray1.begin(),
                                              operatorMatrix.getSqrtMassVec());

              scaleDeviceKernel<<<
                (numberVectors + (deviceConstants::blockSize - 1)) /
                  deviceConstants::blockSize * localVectorSize,
                deviceConstants::blockSize>>>(
                numberVectors,
                localVectorSize,
                1.0,
                YArray1.begin(),
                operatorMatrix.getInvSqrtMassVec());

              daxpbyDeviceKernel<<<
                min((totalVectorSize + (deviceConstants::blockSize - 1)) /
                      deviceConstants::blockSize,
                    30000),
                deviceConstants::blockSize>>>(totalVectorSize,
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
              scaleDeviceKernel<<<
                (numberVectors + (deviceConstants::blockSize - 1)) /
                  deviceConstants::blockSize * localVectorSize,
                deviceConstants::blockSize>>>(numberVectors,
                                              localVectorSize,
                                              1.0 / alpha1Old,
                                              XArray2.begin(),
                                              operatorMatrix.getSqrtMassVec());

              scaleDeviceKernel<<<
                (numberVectors + (deviceConstants::blockSize - 1)) /
                  deviceConstants::blockSize * localVectorSize,
                deviceConstants::blockSize>>>(
                numberVectors,
                localVectorSize,
                1.0,
                YArray2.begin(),
                operatorMatrix.getInvSqrtMassVec());

              daxpbyDeviceKernel<<<
                min((totalVectorSize + (deviceConstants::blockSize - 1)) /
                      deviceConstants::blockSize,
                    30000),
                deviceConstants::blockSize>>>(totalVectorSize,
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
              // iteration) 2) [CP] Call combinedDeviceKernel of block 1 3)
              // [CM-B] Finish compress of nonlocal projectorKetTimesVector of
              // block 2. (skipped in first overlap iteration) 4) [CM-NB] Call
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
              // combinedDeviceKernel of block 2 13)[CM-B] Finish compress of
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

              combinedDeviceKernel<<<min((totalVectorSize +
                                          (deviceConstants::blockSize - 1)) /
                                           deviceConstants::blockSize,
                                         30000),
                                     deviceConstants::blockSize>>>(
                numberVectors,
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
              if (mixedPrecOverall && dftParams.useMixedPrecCheby)
                {
                  convDoubleArrToFloatArr<<<
                    (numberVectors + (deviceConstants::blockSize - 1)) /
                      deviceConstants::blockSize * localVectorSize,
                    deviceConstants::blockSize>>>(numberVectors *
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
                                         dftParams.useMixedPrecCheby,
                                       false,
                                       true);

              if (mixedPrecOverall && dftParams.useMixedPrecCheby)
                {
                  tempFloatArray.updateGhostValuesFinish();
                  if (n_ghosts != 0)
                    convFloatArrToDoubleArr<<<
                      (numberVectors + (deviceConstants::blockSize - 1)) /
                        deviceConstants::blockSize * n_ghosts,
                      deviceConstants::blockSize>>>(
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
                  if (mixedPrecOverall && dftParams.useMixedPrecCheby)
                    {
                      convDoubleArrToFloatArr<<<
                        (numberVectors + (deviceConstants::blockSize - 1)) /
                          deviceConstants::blockSize * totalSize,
                        deviceConstants::blockSize>>>(numberVectors * totalSize,
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
                                       dftParams.useMixedPrecCheby,
                                     true,
                                     false);

              if (overlap)
                {
                  if (mixedPrecOverall && dftParams.useMixedPrecCheby)
                    {
                      tempFloatArray.compressAddFinish();

                      copyFloatArrToDoubleArrLocallyOwned<<<
                        (numberVectors + (deviceConstants::blockSize - 1)) /
                          deviceConstants::blockSize * localVectorSize,
                        deviceConstants::blockSize>>>(
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

              combinedDeviceKernel<<<min((totalVectorSize +
                                          (deviceConstants::blockSize - 1)) /
                                           deviceConstants::blockSize,
                                         30000),
                                     deviceConstants::blockSize>>>(
                numberVectors,
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
              if (mixedPrecOverall && dftParams.useMixedPrecCheby)
                {
                  convDoubleArrToFloatArr<<<
                    (numberVectors + (deviceConstants::blockSize - 1)) /
                      deviceConstants::blockSize * localVectorSize,
                    deviceConstants::blockSize>>>(numberVectors *
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
                                       dftParams.useMixedPrecCheby,
                                     false,
                                     true);

              if (mixedPrecOverall && dftParams.useMixedPrecCheby)
                {
                  tempFloatArray.updateGhostValuesFinish();
                  if (n_ghosts != 0)
                    convFloatArrToDoubleArr<<<
                      (numberVectors + (deviceConstants::blockSize - 1)) /
                        deviceConstants::blockSize * n_ghosts,
                      deviceConstants::blockSize>>>(
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
              if (mixedPrecOverall && dftParams.useMixedPrecCheby)
                {
                  convDoubleArrToFloatArr<<<
                    (numberVectors + (deviceConstants::blockSize - 1)) /
                      deviceConstants::blockSize * totalSize,
                    deviceConstants::blockSize>>>(numberVectors * totalSize,
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
                                       dftParams.useMixedPrecCheby,
                                     true,
                                     false);

              if (mixedPrecOverall && dftParams.useMixedPrecCheby)
                {
                  tempFloatArray.compressAddFinish();

                  copyFloatArrToDoubleArrLocallyOwned<<<
                    (numberVectors + (deviceConstants::blockSize - 1)) /
                      deviceConstants::blockSize * localVectorSize,
                    deviceConstants::blockSize>>>(
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
                                           dftParams.useMixedPrecCheby,
                                         false,
                                         true);
                  YArray2.zeroOutGhosts();
                  if (mixedPrecOverall && dftParams.useMixedPrecCheby)
                    {
                      convDoubleArrToFloatArr<<<
                        (numberVectors + (deviceConstants::blockSize - 1)) /
                          deviceConstants::blockSize * totalSize,
                        deviceConstants::blockSize>>>(numberVectors * totalSize,
                                                      XArray2.begin(),
                                                      tempFloatArray.begin());
                      tempFloatArray.compressAdd();

                      copyFloatArrToDoubleArrLocallyOwned<<<
                        (numberVectors + (deviceConstants::blockSize - 1)) /
                          deviceConstants::blockSize * localVectorSize,
                        deviceConstants::blockSize>>>(
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
      dftfe::utils::deviceMemcpyD2D(XArray1.begin(),
                 YArray1.begin(),
                 totalVectorSize * sizeof(dataTypes::number));

      dftfe::utils::deviceMemcpyD2D(XArray2.begin(),
                 YArray2.begin(),
                 totalVectorSize * sizeof(dataTypes::number));
    }


    void
    subspaceRotationSpectrumSplitScalapack(
      const dataTypes::numberDevice *                  X,
      dataTypes::numberDevice *                        XFrac,
      const unsigned int                               M,
      const unsigned int                               N,
      const unsigned int                               Nfr,
      cublasHandle_t &                                 handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      DeviceCCLWrapper &                               devicecclMpiCommDomain,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters &                            dftParams,
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
        std::min(dftParams.wfcBlockSize, Nfr);
      const unsigned int dofsBlockSize =
        std::min(maxNumLocalDofs, dftParams.subspaceRotDofsBlockSize);

      dftfe::utils::MemoryStorage<dataTypes::number,dftfe::utils::MemorySpace::HOST_PINNED> rotationMatBlockHost;

      if (dftParams.allowFullCPUMemSubspaceRot)
        {
          rotationMatBlockHost.resize(N * Nfr,dataTypes::number(0));
          rotationMatBlockHost.setValue(0);
        }
      else
        {
          rotationMatBlockHost.resize(vectorsBlockSize * N,dataTypes::number(0));
          rotationMatBlockHost.setValue(0);          
        }

      cudaStream_t streamCompute, streamDeviceCCL;
      DeviceCHECK(cudaStreamCreate(&streamCompute));
      DeviceCHECK(cudaStreamCreate(&streamDeviceCCL));

      // attach cublas handle to compute stream
      cublasSetStream(handle, streamCompute);

      // create array of compute and device direct commun events on Devices
      // for all the blocks. These are required for synchronization
      const unsigned int numberBlocks =
        (N / vectorsBlockSize) * (maxNumLocalDofs / dofsBlockSize + 1);
      cudaEvent_t computeEvents[numberBlocks];
      cudaEvent_t communEvents[numberBlocks];
      for (int i = 0; i < numberBlocks; ++i)
        {
          DeviceCHECK(cudaEventCreate(&computeEvents[i]));
          DeviceCHECK(cudaEventCreate(&communEvents[i]));
        }

      dftfe::utils::MemoryStorage<dataTypes::number,dftfe::utils::MemorySpace::DEVICE> rotationMatBlock(
        vectorsBlockSize * N, dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::number,dftfe::utils::MemorySpace::DEVICE> rotationMatBlockNext(
        vectorsBlockSize * N, dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::number,dftfe::utils::MemorySpace::DEVICE>
        rotatedVectorsMatBlock(Nfr * dofsBlockSize,
                               dataTypes::number(0));

      dataTypes::numberValueType *tempReal;
      dataTypes::numberValueType *tempImag;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          DeviceCHECK(cudaMalloc((void **)&tempReal,
                                 vectorsBlockSize * N *
                                   sizeof(dataTypes::numberValueType)));
          DeviceCHECK(cudaMalloc((void **)&tempImag,
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
              const dataTypes::number scalarCoeffBeta  = dataTypes::number(0);

              if (dftParams.allowFullCPUMemSubspaceRot)
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
                                        *(rotationMatBlockHost.begin()+jvec * N +
                                                             i * BVec + j) =
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
                                        *(rotationMatBlockHost.begin()+jvec * N +
                                                             i * BVec + j) =
                                          rotationMatPar.local_el(
                                            it->second, localColumnId);
                                    }
                                }
                        }
                    }
                }
              else
                {
                  std::memset(rotationMatBlockHost.begin(),
                              0,
                              BVec * N * sizeof(dataTypes::number));

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
                                    *(rotationMatBlockHost.begin()+i * BVec + j) =
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
                                    *(rotationMatBlockHost.begin()+i * BVec + j) =
                                      rotationMatPar.local_el(it->second,
                                                              localColumnId);
                                }
                            }
                    }
                }


              if (dftParams.allowFullCPUMemSubspaceRot)
                {
                  if (dftParams.useDeviceDirectAllReduce)
                    {
                      DeviceCHECK(cudaMemcpyAsync(
                        dftfe::utils::makeDataTypeDeviceCompatible(
                        rotationMatBlockNext.begin()),
                        rotationMatBlockHost.begin() + jvec * N,
                        BVec * N * sizeof(dataTypes::number),
                        cudaMemcpyHostToDevice,
                        streamDeviceCCL));

                      if (idof == 0)
                        {
                          if (std::is_same<dataTypes::number,
                                           std::complex<double>>::value)
                            devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                  rotationMatBlockNext.begin()),
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                  rotationMatBlockNext.begin()),
                              BVec * N,
                              tempReal,
                              tempImag,
                              streamDeviceCCL);
                          else
                            devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                  rotationMatBlockNext.begin()),
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                  rotationMatBlockNext.begin()),
                              BVec * N,
                              streamDeviceCCL);

                          DeviceCHECK(cudaMemcpyAsync(
                            rotationMatBlockHost.begin() + jvec * N,
                            dftfe::utils::makeDataTypeDeviceCompatible(
                                rotationMatBlockNext.begin()),
                            BVec * N * sizeof(dataTypes::number),
                            cudaMemcpyDeviceToHost,
                            streamDeviceCCL));
                        }
                    }
                  else
                    {
                      if (idof == 0)
                        MPI_Allreduce(MPI_IN_PLACE,
                                      dftfe::utils::makeDataTypeDeviceCompatible(
                                        rotationMatBlockHost.begin() + jvec * N),
                                      BVec * N,
                                      dataTypes::mpi_type_id(
                                          rotationMatBlockHost.begin()),
                                      MPI_SUM,
                                      mpiCommDomain);

                      dftfe::utils::deviceMemcpyH2D(
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotationMatBlock.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotationMatBlockHost.begin() + jvec * N),
                        BVec * N * sizeof(dataTypes::number));
                    }
                }
              else
                {
                  if (dftParams.useDeviceDirectAllReduce)
                    {
                      DeviceCHECK(cudaMemcpyAsync(
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotationMatBlockNext.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotationMatBlockHost.begin()),
                        BVec * N * sizeof(dataTypes::number),
                        cudaMemcpyHostToDevice,
                        streamDeviceCCL));

                      if (std::is_same<dataTypes::number,
                                       std::complex<double>>::value)
                        devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            rotationMatBlockNext.begin()),
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            rotationMatBlockNext.begin()),
                          BVec * N,
                          tempReal,
                          tempImag,
                          streamDeviceCCL);
                      else
                        devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            rotationMatBlockNext.begin()),
                          dftfe::utils::makeDataTypeDeviceCompatible(
                           rotationMatBlockNext.begin()),
                          BVec * N,
                          streamDeviceCCL);
                    }
                  else
                    {
                      MPI_Allreduce(MPI_IN_PLACE,
                                    rotationMatBlockHost.begin(),
                                    BVec * N,
                                    dataTypes::mpi_type_id(
                                      rotationMatBlockHost.begin()),
                                    MPI_SUM,
                                    mpiCommDomain);

                      dftfe::utils::deviceMemcpyH2D(
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotationMatBlock.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotationMatBlockHost.begin()),
                        BVec * N * sizeof(dataTypes::number));
                    }
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  // check for completion of compute of previous block in
                  // compute stream before proceeding to rewriting
                  // rotationMatBlock in communication stream
                  DeviceCHECK(
                    cudaEventRecord(computeEvents[blockCount], streamCompute));
                  DeviceCHECK(cudaStreamWaitEvent(streamDeviceCCL,
                                                  computeEvents[blockCount],
                                                  0));

                  // synchronize host to communication stream before doing swap
                  // this automatically also makes sure the compute stream has
                  // the correct rotationMatBlock for dgemm
                  DeviceCHECK(
                    cudaEventRecord(communEvents[blockCount], streamDeviceCCL));
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
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                &scalarCoeffAlpha),
                              dftfe::utils::makeDataTypeDeviceCompatible(
                              rotationMatBlock.begin()),
                              BVec,
                              X + idof * N,
                              N,
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                &scalarCoeffBeta),
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                rotatedVectorsMatBlock.begin() +
                                jvec),
                              Nfr);
                }

              blockCount++;
            } // block loop over vectors


          if (BDof != 0)
            {
              DeviceCHECK(cudaMemcpyAsync(
                XFrac + idof * Nfr,
                dftfe::utils::makeDataTypeDeviceCompatible(
                  rotatedVectorsMatBlock.begin()),
                Nfr * BDof * sizeof(dataTypes::number),
                cudaMemcpyDeviceToDevice,
                streamCompute));
            }

        } // block loop over dofs

      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          DeviceCHECK(cudaFree(tempReal));
          DeviceCHECK(cudaFree(tempImag));
        }

      // return cublas handle to default stream
      cublasSetStream(handle, NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          DeviceCHECK(cudaEventDestroy(computeEvents[i]));
          DeviceCHECK(cudaEventDestroy(communEvents[i]));
        }

      DeviceCHECK(cudaStreamDestroy(streamCompute));
      DeviceCHECK(cudaStreamDestroy(streamDeviceCCL));
    }



    void
    subspaceRotationScalapack(
      dataTypes::numberDevice *                        X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      DeviceCCLWrapper &                               devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters &                            dftParams,
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

      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const unsigned int dofsBlockSize =
        std::min(maxNumLocalDofs, dftParams.subspaceRotDofsBlockSize);

      dftfe::utils::MemoryStorage<dataTypes::number,dftfe::utils::MemorySpace::HOST_PINNED> rotationMatBlockHost;

      if (dftParams.allowFullCPUMemSubspaceRot)
        {
          rotationMatBlockHost.resize(N * N,dataTypes::number(0));
          rotationMatBlockHost.setValue(0);
        }
      else
        {
          rotationMatBlockHost.resize(vectorsBlockSize * N,dataTypes::number(0));
          rotationMatBlockHost.setValue(0);          
        }


      cudaStream_t streamCompute, streamDeviceCCL;
      DeviceCHECK(cudaStreamCreate(&streamCompute));
      DeviceCHECK(cudaStreamCreate(&streamDeviceCCL));

      // attach cublas handle to compute stream
      cublasSetStream(handle, streamCompute);

      // create array of compute and device direct commun events on Devices
      // for all the blocks. These are required for synchronization
      const unsigned int numberBlocks =
        (N / vectorsBlockSize) * (maxNumLocalDofs / dofsBlockSize + 1);
      cudaEvent_t computeEvents[numberBlocks];
      cudaEvent_t communEvents[numberBlocks];
      for (int i = 0; i < numberBlocks; ++i)
        {
          DeviceCHECK(cudaEventCreate(&computeEvents[i]));
          DeviceCHECK(cudaEventCreate(&communEvents[i]));
        }

      dftfe::utils::MemoryStorage<dataTypes::number,dftfe::utils::MemorySpace::DEVICE> rotationMatBlock(
        vectorsBlockSize * N, dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::number,dftfe::utils::MemorySpace::DEVICE> rotationMatBlockTemp(
        vectorsBlockSize * N, dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::number,dftfe::utils::MemorySpace::DEVICE>
        rotatedVectorsMatBlock(N * dofsBlockSize,
                               dataTypes::number(0));

      dataTypes::numberValueType *tempReal;
      dataTypes::numberValueType *tempImag;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          DeviceCHECK(cudaMalloc((void **)&tempReal,
                                 vectorsBlockSize * N *
                                   sizeof(dataTypes::numberValueType)));
          DeviceCHECK(cudaMalloc((void **)&tempImag,
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
                    dataTypes::number(0);

                  if (dftParams.allowFullCPUMemSubspaceRot)
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
                                            *(rotationMatBlockHost.begin()+jvec * N +
                                                                 i * BVec + j) =
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
                                            *(rotationMatBlockHost.begin()+jvec * N +
                                                                 i * BVec + j) =
                                              rotationMatPar.local_el(
                                                it->second, localColumnId);
                                        }
                                    }
                            }
                        }
                    }
                  else
                    {
                      std::memset(rotationMatBlockHost.begin(),
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
                                        *(rotationMatBlockHost.begin()+i * BVec + j) =
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
                                        *(rotationMatBlockHost.begin()+i * BVec + j) =
                                          rotationMatPar.local_el(
                                            it->second, localColumnId);
                                    }
                                }
                        }
                    }

                  if (dftParams.allowFullCPUMemSubspaceRot)
                    {
                      if (dftParams.useDeviceDirectAllReduce)
                        {
                          DeviceCHECK(cudaMemcpyAsync(
                            dftfe::utils::makeDataTypeDeviceCompatible(
                                rotationMatBlockTemp.begin()),
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              rotationMatBlockHost.begin() + jvec * N),
                            BVec * D * sizeof(dataTypes::number),
                            cudaMemcpyHostToDevice,
                            streamDeviceCCL));

                          if (idof == 0)
                            {
                              if (std::is_same<dataTypes::number,
                                               std::complex<double>>::value)
                                devicecclMpiCommDomain
                                  .deviceDirectAllReduceWrapper(
                                    dftfe::utils::makeDataTypeDeviceCompatible(
                                        rotationMatBlockTemp.begin()),
                                    dftfe::utils::makeDataTypeDeviceCompatible(
                                        rotationMatBlockTemp.begin()),
                                    BVec * D,
                                    tempReal,
                                    tempImag,
                                    streamDeviceCCL);
                              else
                                devicecclMpiCommDomain
                                  .deviceDirectAllReduceWrapper(
                                    dftfe::utils::makeDataTypeDeviceCompatible(
                                        rotationMatBlockTemp.begin()),
                                    dftfe::utils::makeDataTypeDeviceCompatible(
                                        rotationMatBlockTemp.begin()),
                                    BVec * D,
                                    streamDeviceCCL);

                              DeviceCHECK(cudaMemcpyAsync(
                                dftfe::utils::makeDataTypeDeviceCompatible(
                                  rotationMatBlockHost.begin() + jvec * N),
                                dftfe::utils::makeDataTypeDeviceCompatible(rotationMatBlockTemp.begin()),
                                BVec * D * sizeof(dataTypes::number),
                                cudaMemcpyDeviceToHost,
                                streamDeviceCCL));
                            }
                        }
                      else
                        {
                          if (idof == 0)
                            MPI_Allreduce(MPI_IN_PLACE,
                                          rotationMatBlockHost.begin() + jvec * N,
                                          BVec * D,
                                          dataTypes::mpi_type_id(
                                            rotationMatBlockHost.begin()),
                                          MPI_SUM,
                                          mpiCommDomain);

                          dftfe::utils::deviceMemcpyH2D(
                            dftfe::utils::makeDataTypeDeviceCompatible(rotationMatBlock.begin()),
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              rotationMatBlockHost.begin() + jvec * N),
                            BVec * D * sizeof(dataTypes::number));
                        }
                    }
                  else
                    {
                      if (dftParams.useDeviceDirectAllReduce)
                        {
                          DeviceCHECK(cudaMemcpyAsync(
                            dftfe::utils::makeDataTypeDeviceCompatible(
                                rotationMatBlockTemp.begin()),
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              rotationMatBlockHost.begin()),
                            BVec * D * sizeof(dataTypes::number),
                            cudaMemcpyHostToDevice,
                            streamDeviceCCL));

                          if (std::is_same<dataTypes::number,
                                           std::complex<double>>::value)
                            devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                  rotationMatBlockTemp.begin()),
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                  rotationMatBlockTemp.begin()),
                              BVec * D,
                              tempReal,
                              tempImag,
                              streamDeviceCCL);
                          else
                            devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                  rotationMatBlockTemp.begin()),
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                  rotationMatBlockTemp.begin()),
                              BVec * D,
                              streamDeviceCCL);
                        }
                      else
                        {
                          MPI_Allreduce(MPI_IN_PLACE,
                                        rotationMatBlockHost.begin(),
                                        BVec * D,
                                        dataTypes::mpi_type_id(
                                          rotationMatBlockHost.begin()),
                                        MPI_SUM,
                                        mpiCommDomain);

                          dftfe::utils::deviceMemcpyH2D(
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              rotationMatBlock.begin()),
                            dftfe::utils::makeDataTypeDeviceCompatible(
                              rotationMatBlockHost.begin()),
                            BVec * D * sizeof(dataTypes::number));
                        }
                    }

                  if (dftParams.useDeviceDirectAllReduce)
                    {
                      // check for completion of compute of previous block in
                      // compute stream before proceeding to rewriting
                      // rotationMatBlock in communication stream
                      DeviceCHECK(cudaEventRecord(computeEvents[blockCount],
                                                  streamCompute));
                      DeviceCHECK(cudaStreamWaitEvent(streamDeviceCCL,
                                                      computeEvents[blockCount],
                                                      0));

                      // synchronize host to communication stream before doing
                      // swap this automatically also makes sure the compute
                      // stream has the correct rotationMatBlock for dgemm
                      DeviceCHECK(cudaEventRecord(communEvents[blockCount],
                                                  streamDeviceCCL));
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
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          &scalarCoeffAlpha),
                        dftfe::utils::makeDataTypeDeviceCompatible(rotationMatBlock.begin()),
                        BVec,
                        X + idof * N,
                        N,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          &scalarCoeffBeta),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotatedVectorsMatBlock.begin() +
                          jvec),
                        N);
                    }
                } // band parallelization
              blockCount++;
            } // block loop over vectors


          if (BDof != 0)
            {
              DeviceCHECK(cudaMemcpyAsync(
                X + idof * N,
                dftfe::utils::makeDataTypeDeviceCompatible(
                  rotatedVectorsMatBlock.begin()),
                N * BDof * sizeof(dataTypes::number),
                cudaMemcpyDeviceToDevice,
                streamCompute));
            }

        } // block loop over dofs


      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          DeviceCHECK(cudaFree(tempReal));
          DeviceCHECK(cudaFree(tempImag));
        }

      // return cublas handle to default stream
      cublasSetStream(handle, NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          DeviceCHECK(cudaEventDestroy(computeEvents[i]));
          DeviceCHECK(cudaEventDestroy(communEvents[i]));
        }

      DeviceCHECK(cudaStreamDestroy(streamCompute));
      DeviceCHECK(cudaStreamDestroy(streamDeviceCCL));
    }

    void
    subspaceRotationCGSMixedPrecScalapack(
      dataTypes::numberDevice *                        X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      DeviceCCLWrapper &                               devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters &                            dftParams,
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
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,dftfe::utils::MemorySpace::DEVICE> XSP(
        MPadded * N, dataTypes::numberFP32(0));

      convDoubleArrToFloatArr<<<(N + (deviceConstants::blockSize - 1)) /
                                  deviceConstants::blockSize * M,
                                deviceConstants::blockSize>>>(
        N * M,
        X,
        dftfe::utils::makeDataTypeDeviceCompatible(XSP.begin()));


      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const unsigned int dofsBlockSize =
        std::min(maxNumLocalDofs, dftParams.subspaceRotDofsBlockSize);


      dftfe::utils::MemoryStorage<dataTypes::numberFP32,dftfe::utils::MemorySpace::HOST_PINNED> rotationMatBlockHostSP(vectorsBlockSize * N);

      std::memset(rotationMatBlockHostSP.begin(),
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::numberFP32));

      dataTypes::number *diagValuesHost;
      DeviceCHECK(cudaMallocHost((void **)&diagValuesHost,
                                 N * sizeof(dataTypes::number)));
      std::memset(diagValuesHost, 0, N * sizeof(dataTypes::number));

      cudaStream_t streamCompute, streamDeviceCCL;
      DeviceCHECK(cudaStreamCreate(&streamCompute));
      DeviceCHECK(cudaStreamCreate(&streamDeviceCCL));

      // attach cublas handle to compute stream
      cublasSetStream(handle, streamCompute);

      // create array of compute and device direct commun events on Devices
      // for all the blocks. These are required for synchronization
      const unsigned int numberBlocks = (N / vectorsBlockSize);
      cudaEvent_t        computeEvents[numberBlocks];
      cudaEvent_t        communEvents[numberBlocks];
      for (int i = 0; i < numberBlocks; ++i)
        {
          DeviceCHECK(cudaEventCreate(&computeEvents[i]));
          DeviceCHECK(cudaEventCreate(&communEvents[i]));
        }

      dftfe::utils::MemoryStorage<dataTypes::numberFP32,dftfe::utils::MemorySpace::DEVICE>
        rotationMatBlockSP(vectorsBlockSize * N,
                           dataTypes::numberFP32(0));
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,dftfe::utils::MemorySpace::DEVICE>
                                                           rotationMatBlockSPTemp(vectorsBlockSize * N,
                               dataTypes::numberFP32(0));
      dftfe::utils::MemoryStorage<dataTypes::number,dftfe::utils::MemorySpace::DEVICE> diagValues(
        N, dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,dftfe::utils::MemorySpace::DEVICE>
        rotatedVectorsMatBlockSP(vectorsBlockSize * dofsBlockSize,
                                 dataTypes::numberFP32(0));

      const dataTypes::numberFP32 scalarCoeffAlphaSP =
        dataTypes::numberFP32(1.0);
      const dataTypes::numberFP32 scalarCoeffBetaSP =
        dataTypes::numberFP32(0);


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

      dftfe::utils::deviceMemcpyH2D(dftfe::utils::makeDataTypeDeviceCompatible(
                   diagValues.begin()),
                 dftfe::utils::makeDataTypeDeviceCompatible(
                   diagValuesHost),
                 N * sizeof(dataTypes::number));

      computeDiagQTimesXKernel<<<(M * N + (deviceConstants::blockSize - 1)) /
                                   deviceConstants::blockSize,
                                 deviceConstants::blockSize>>>(
        dftfe::utils::makeDataTypeDeviceCompatible(
          diagValues.begin()),
        X,
        N,
        M);

      dataTypes::numberFP32ValueType *tempRealFP32;
      dataTypes::numberFP32ValueType *tempImagFP32;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          DeviceCHECK(cudaMalloc((void **)&tempRealFP32,
                                 vectorsBlockSize * N *
                                   sizeof(dataTypes::numberFP32ValueType)));
          DeviceCHECK(cudaMalloc((void **)&tempImagFP32,
                                 vectorsBlockSize * N *
                                   sizeof(dataTypes::numberFP32ValueType)));
        }

      unsigned int blockCount = 0;
      for (unsigned int jvec = 0; jvec < N; jvec += vectorsBlockSize)
        {
          // Correct block dimensions if block "goes off edge of" the matrix
          const unsigned int BVec = std::min(vectorsBlockSize, N - jvec);

          const unsigned int D = jvec + BVec;

          std::memset(rotationMatBlockHostSP.begin(),
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
                                  *(rotationMatBlockHostSP.begin()+i * BVec + j) =
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
                                  *(rotationMatBlockHostSP.begin()+i * BVec + i - jvec) =
                                    dataTypes::numberFP32(0);
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
                                  *(rotationMatBlockHostSP.begin()+i * BVec + j) =
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
                                  *(rotationMatBlockHostSP.begin()+i * BVec + i - jvec) =
                                    dataTypes::numberFP32(0);
                                }
                            }
                        }
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  cudaMemcpyAsync(
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      rotationMatBlockSPTemp.begin()),
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      rotationMatBlockHostSP.begin()),
                    BVec * D * sizeof(dataTypes::numberFP32),
                    cudaMemcpyHostToDevice,
                    streamDeviceCCL);

                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        rotationMatBlockSPTemp.begin()),
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        rotationMatBlockSPTemp.begin()),
                      BVec * D,
                      tempRealFP32,
                      tempImagFP32,
                      streamDeviceCCL);
                  else
                    devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        rotationMatBlockSPTemp.begin()),
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        rotationMatBlockSPTemp.begin()),
                      BVec * D,
                      streamDeviceCCL);
                }
              else
                {
                  MPI_Allreduce(MPI_IN_PLACE,
                                rotationMatBlockHostSP.begin(),
                                BVec * D,
                                dataTypes::mpi_type_id(rotationMatBlockHostSP.begin()),
                                MPI_SUM,
                                mpiCommDomain);

                  dftfe::utils::deviceMemcpyH2D(
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      rotationMatBlockSP.begin()),
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      rotationMatBlockHostSP.begin()),
                    BVec * D * sizeof(dataTypes::numberFP32));
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  // check for completion of compute of previous block in
                  // compute stream before proceeding to rewriting
                  // rotationMatBlock in communication stream
                  DeviceCHECK(
                    cudaEventRecord(computeEvents[blockCount], streamCompute));
                  DeviceCHECK(cudaStreamWaitEvent(streamDeviceCCL,
                                                  computeEvents[blockCount],
                                                  0));

                  // synchronize host to communication stream before doing swap
                  // this automatically also makes sure the compute stream has
                  // the correct rotationMatBlock for dgemm
                  DeviceCHECK(
                    cudaEventRecord(communEvents[blockCount], streamDeviceCCL));
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
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          &scalarCoeffAlphaSP),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotationMatBlockSP.begin()),
                        BVec,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          XSP.begin() + idof * N),
                        N,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          &scalarCoeffBetaSP),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotatedVectorsMatBlockSP.begin()),
                        BVec);

                      addSubspaceRotatedBlockToXKernel<<<
                        (BVec * BDof + (deviceConstants::blockSize - 1)) /
                          deviceConstants::blockSize,
                        deviceConstants::blockSize,
                        0,
                        streamCompute>>>(
                        BDof,
                        BVec,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotatedVectorsMatBlockSP.begin()),
                        X,
                        idof,
                        jvec,
                        N);
                    }
                } // block loop over dofs
            }     // band parallalelization loop
          blockCount++;
        } // block loop over vectors

      DeviceCHECK(cudaFreeHost(diagValuesHost));

      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          DeviceCHECK(cudaFree(tempRealFP32));
          DeviceCHECK(cudaFree(tempImagFP32));
        }

      // return cublas handle to default stream
      cublasSetStream(handle, NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          DeviceCHECK(cudaEventDestroy(computeEvents[i]));
          DeviceCHECK(cudaEventDestroy(communEvents[i]));
        }

      DeviceCHECK(cudaStreamDestroy(streamCompute));
      DeviceCHECK(cudaStreamDestroy(streamDeviceCCL));
    }

    void
    subspaceRotationRRMixedPrecScalapack(
      dataTypes::numberDevice *                        X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      const MPI_Comm &                                 mpiCommDomain,
      DeviceCCLWrapper &                               devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const dftfe::ScaLAPACKMatrix<dataTypes::number> &rotationMatPar,
      const dftParameters &                            dftParams,
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
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,dftfe::utils::MemorySpace::DEVICE>  XSP(
        MPadded * N, dataTypes::numberFP32(0));

      convDoubleArrToFloatArr<<<(N + (deviceConstants::blockSize - 1)) /
                                  deviceConstants::blockSize * M,
                                deviceConstants::blockSize>>>(
        N * M,
        X,
        dftfe::utils::makeDataTypeDeviceCompatible(
          XSP.begin()));


      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);

      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const unsigned int dofsBlockSize =
        std::min(maxNumLocalDofs, dftParams.subspaceRotDofsBlockSize);

      dftfe::utils::MemoryStorage<dataTypes::numberFP32,dftfe::utils::MemorySpace::HOST_PINNED> rotationMatBlockHostSP(vectorsBlockSize * N);

      std::memset(rotationMatBlockHostSP.begin(),
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::numberFP32));

      dataTypes::number *diagValuesHost;
      DeviceCHECK(cudaMallocHost((void **)&diagValuesHost,
                                 N * sizeof(dataTypes::number)));
      std::memset(diagValuesHost, 0, N * sizeof(dataTypes::number));

      cudaStream_t streamCompute, streamDeviceCCL;
      DeviceCHECK(cudaStreamCreate(&streamCompute));
      DeviceCHECK(cudaStreamCreate(&streamDeviceCCL));

      // attach cublas handle to compute stream
      cublasSetStream(handle, streamCompute);

      // create array of compute and device direct commun events on Devices
      // for all the blocks. These are required for synchronization
      const unsigned int numberBlocks = (N / vectorsBlockSize);
      cudaEvent_t        computeEvents[numberBlocks];
      cudaEvent_t        communEvents[numberBlocks];
      for (int i = 0; i < numberBlocks; ++i)
        {
          DeviceCHECK(cudaEventCreate(&computeEvents[i]));
          DeviceCHECK(cudaEventCreate(&communEvents[i]));
        }

      dftfe::utils::MemoryStorage<dataTypes::numberFP32,dftfe::utils::MemorySpace::DEVICE> 
        rotationMatBlockSP(vectorsBlockSize * N,
                           dataTypes::numberFP32(0));
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,dftfe::utils::MemorySpace::DEVICE> 
                                                           rotationMatBlockSPTemp(vectorsBlockSize * N,
                               dataTypes::numberFP32(0));
      dftfe::utils::MemoryStorage<dataTypes::number,dftfe::utils::MemorySpace::DEVICE> diagValues(
        N, dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,dftfe::utils::MemorySpace::DEVICE> 
        rotatedVectorsMatBlockSP(vectorsBlockSize * dofsBlockSize,
                                 dataTypes::numberFP32(0));

      const dataTypes::numberFP32 scalarCoeffAlphaSP =
        dataTypes::numberFP32(1.0);
      const dataTypes::numberFP32 scalarCoeffBetaSP =
        dataTypes::numberFP32(0);


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

      dftfe::utils::deviceMemcpyH2D(dftfe::utils::makeDataTypeDeviceCompatible(
                   diagValues.begin()),
                 dftfe::utils::makeDataTypeDeviceCompatible(
                   diagValuesHost),
                 N * sizeof(dataTypes::number));

      computeDiagQTimesXKernel<<<(M * N + (deviceConstants::blockSize - 1)) /
                                   deviceConstants::blockSize,
                                 deviceConstants::blockSize>>>(
        dftfe::utils::makeDataTypeDeviceCompatible(
          diagValues.begin()),
        X,
        N,
        M);

      dataTypes::numberFP32ValueType *tempRealFP32;
      dataTypes::numberFP32ValueType *tempImagFP32;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          DeviceCHECK(cudaMalloc((void **)&tempRealFP32,
                                 vectorsBlockSize * N *
                                   sizeof(dataTypes::numberFP32ValueType)));
          DeviceCHECK(cudaMalloc((void **)&tempImagFP32,
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
              std::memset(rotationMatBlockHostSP.begin(),
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
                                  *(rotationMatBlockHostSP.begin()+i * BVec + j) =
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
                                  *(rotationMatBlockHostSP.begin()+i * BVec + i - jvec) =
                                    dataTypes::numberFP32(0);
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
                                  *(rotationMatBlockHostSP.begin()+i * BVec + j) =
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
                                  *(rotationMatBlockHostSP.begin()+i * BVec + i - jvec) =
                                    dataTypes::numberFP32(0);
                                }
                            }
                        }
                }


              if (dftParams.useDeviceDirectAllReduce)
                {
                  cudaMemcpyAsync(
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      rotationMatBlockSPTemp.begin()),
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      rotationMatBlockHostSP.begin()),
                    BVec * D * sizeof(dataTypes::numberFP32),
                    cudaMemcpyHostToDevice,
                    streamDeviceCCL);

                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        rotationMatBlockSPTemp.begin()),
                      dftfe::utils::makeDataTypeDeviceCompatible(rotationMatBlockSPTemp.begin()),
                      BVec * D,
                      tempRealFP32,
                      tempImagFP32,
                      streamDeviceCCL);
                  else
                    devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                      dftfe::utils::makeDataTypeDeviceCompatible(rotationMatBlockSPTemp.begin()),
                      dftfe::utils::makeDataTypeDeviceCompatible(rotationMatBlockSPTemp.begin()),
                      BVec * D,
                      streamDeviceCCL);
                }
              else
                {
                  MPI_Allreduce(MPI_IN_PLACE,
                                rotationMatBlockHostSP.begin(),
                                BVec * D,
                                dataTypes::mpi_type_id(rotationMatBlockHostSP.begin()),
                                MPI_SUM,
                                mpiCommDomain);

                  dftfe::utils::deviceMemcpyH2D(
                    dftfe::utils::makeDataTypeDeviceCompatible(
                     rotationMatBlockSP.begin()),
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      rotationMatBlockHostSP.begin()),
                    BVec * D * sizeof(dataTypes::numberFP32));
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  // check for completion of compute of previous block in
                  // compute stream before proceeding to rewriting
                  // rotationMatBlock in communication stream
                  DeviceCHECK(
                    cudaEventRecord(computeEvents[blockCount], streamCompute));
                  DeviceCHECK(cudaStreamWaitEvent(streamDeviceCCL,
                                                  computeEvents[blockCount],
                                                  0));

                  // synchronize host to communication stream before doing swap
                  // this automatically also makes sure the compute stream has
                  // the correct rotationMatBlock for dgemm
                  DeviceCHECK(
                    cudaEventRecord(communEvents[blockCount], streamDeviceCCL));
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
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          &scalarCoeffAlphaSP),
                        dftfe::utils::makeDataTypeDeviceCompatible(rotationMatBlockSP.begin()),
                        BVec,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          XSP.begin() + idof * N),
                        N,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          &scalarCoeffBetaSP),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          rotatedVectorsMatBlockSP.begin()),
                        BVec);


                      addSubspaceRotatedBlockToXKernel<<<
                        (BVec * BDof + (deviceConstants::blockSize - 1)) /
                          deviceConstants::blockSize,
                        deviceConstants::blockSize,
                        0,
                        streamCompute>>>(
                        BDof,
                        BVec,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                            rotatedVectorsMatBlockSP.begin()),
                        X,
                        idof,
                        jvec,
                        N);
                    }
                } // block loop over dofs
            }     // band parallelization
          blockCount++;
        } // block loop over vectors

      DeviceCHECK(cudaFreeHost(diagValuesHost));

      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          DeviceCHECK(cudaFree(tempRealFP32));
          DeviceCHECK(cudaFree(tempImagFP32));
        }

      // return cublas handle to default stream
      cublasSetStream(handle, NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          DeviceCHECK(cudaEventDestroy(computeEvents[i]));
          DeviceCHECK(cudaEventDestroy(communEvents[i]));
        }

      DeviceCHECK(cudaStreamDestroy(streamCompute));
      DeviceCHECK(cudaStreamDestroy(streamDeviceCCL));
    }

    void
    fillParallelOverlapMatScalapack(
      const dataTypes::numberDevice *                  X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const MPI_Comm &                                 mpiCommDomain,
      DeviceCCLWrapper &                               devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar,
      const dftParameters &                            dftParams)
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

      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);

      dftfe::utils::MemoryStorage<dataTypes::number,dftfe::utils::MemorySpace::DEVICE> overlapMatrixBlock(
        N * vectorsBlockSize, dataTypes::number(0));

      dataTypes::number *overlapMatrixBlockHost;
      DeviceCHECK(
        cudaMallocHost((void **)&overlapMatrixBlockHost,
                       N * vectorsBlockSize * sizeof(dataTypes::number)));
      std::memset(overlapMatrixBlockHost,
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::number));

      cudaStream_t streamDeviceCCL;
      cudaStreamCreate(&streamDeviceCCL);

      const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0);
      const dataTypes::number scalarCoeffBeta  = dataTypes::number(0);

      dataTypes::numberValueType *tempReal;
      dataTypes::numberValueType *tempImag;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          DeviceCHECK(cudaMalloc((void **)&tempReal,
                                 vectorsBlockSize * N *
                                   sizeof(dataTypes::numberValueType)));
          DeviceCHECK(cudaMalloc((void **)&tempImag,
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
                dftfe::utils::makeDataTypeDeviceCompatible(
                  &scalarCoeffAlpha),
                X + ivec,
                N,
                X + ivec,
                N,
                dftfe::utils::makeDataTypeDeviceCompatible(
                  &scalarCoeffBeta),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlock.begin()),
                D);


              if (dftParams.useDeviceDirectAllReduce)
                {
                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                      dftfe::utils::makeDataTypeDeviceCompatible(
                         overlapMatrixBlock.begin()),
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        overlapMatrixBlock.begin()),
                      D * B,
                      tempReal,
                      tempImag,
                      streamDeviceCCL);
                  else
                    devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        overlapMatrixBlock.begin()),
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        overlapMatrixBlock.begin()),
                      D * B,
                      streamDeviceCCL);
                }

              dftfe::utils::deviceMemcpyD2H(dftfe::utils::makeDataTypeDeviceCompatible(
                           overlapMatrixBlockHost),
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           overlapMatrixBlock.begin()),
                         D * B * sizeof(dataTypes::number));

              // Sum local XTrunc^{T}*XcBlock across domain decomposition
              // processors
              if (!dftParams.useDeviceDirectAllReduce)
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

      DeviceCHECK(cudaFreeHost(overlapMatrixBlockHost));
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          DeviceCHECK(cudaFree(tempReal));
          DeviceCHECK(cudaFree(tempImag));
        }

      cudaStreamDestroy(streamDeviceCCL);

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
    // COP denotes Device->CPU copy of X^{T} times XBlock
    // COM denotes blocking MPI_Allreduce on X^{T}XBlock and copy to scalapack
    // matrix
    // ----------------------------------------------------------
    // Two Device streams are created: compute and copy
    // CMP is performed in compute Device stream and COP is performed in copy
    // Device stream. COP for a block can only start after the CMP for that
    // block in the compute stream is completed. COM is performed for a block
    // only after COP even for that block is completed.
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
      const dataTypes::numberDevice *                  X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const MPI_Comm &                                 mpiCommDomain,
      DeviceCCLWrapper &                               devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar,
      const dftParameters &                            dftParams)
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

      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const unsigned int numberBlocks     = N / vectorsBlockSize;

      // create separate Device streams for data movement and computation
      cudaStream_t streamCompute, streamDataMove;
      DeviceCHECK(cudaStreamCreate(&streamCompute));
      DeviceCHECK(cudaStreamCreate(&streamDataMove));

      // attach cublas handle to compute stream
      cublasSetStream(handle, streamCompute);

      // create array of compute and copy events on Devices
      // for all the blocks. These are required for synchronization
      // between compute, copy and communication as discussed above in the
      // pseudo code
      cudaEvent_t computeEvents[numberBlocks];
      cudaEvent_t copyEvents[numberBlocks];

      for (int i = 0; i < numberBlocks; ++i)
        {
          DeviceCHECK(cudaEventCreate(&computeEvents[i]));
          DeviceCHECK(cudaEventCreate(&copyEvents[i]));
        }

      // create pinned memory used later to copy from Device->CPU
      dataTypes::number *overlapMatrixBlockHost;
      DeviceCHECK(
        cudaMallocHost((void **)&overlapMatrixBlockHost,
                       N * vectorsBlockSize * sizeof(dataTypes::number)));
      std::memset(overlapMatrixBlockHost,
                  0,
                  vectorsBlockSize * N * sizeof(dataTypes::number));

      // allocate device vectors to be used later
      dftfe::utils::MemoryStorage<dataTypes::number,dftfe::utils::MemorySpace::DEVICE> overlapMatrixBlock(
        N * vectorsBlockSize, dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::number,dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlockNext(N * vectorsBlockSize,
                               dataTypes::number(0));

      const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0);
      const dataTypes::number scalarCoeffBeta  = dataTypes::number(0);

      dataTypes::numberValueType *tempReal;
      dataTypes::numberValueType *tempImag;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          DeviceCHECK(cudaMalloc((void **)&tempReal,
                                 vectorsBlockSize * N *
                                   sizeof(dataTypes::numberValueType)));
          DeviceCHECK(cudaMalloc((void **)&tempImag,
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
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                &scalarCoeffAlpha),
                              X + ivec,
                              N,
                              X + ivec,
                              N,
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                &scalarCoeffBeta),
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                  overlapMatrixBlock.begin()),
                              D);

                  // record completion of compute for first block
                  DeviceCHECK(
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
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                &scalarCoeffAlpha),
                              X + ivecNew,
                              N,
                              X + ivecNew,
                              N,
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                &scalarCoeffBeta),
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                  overlapMatrixBlockNext.begin()),
                              DNew);

                  // record completion of compute for next block
                  DeviceCHECK(cudaEventRecord(computeEvents[blockCount + 1],
                                              streamCompute));
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  // Sum local XTrunc^{T}*XcBlock across domain decomposition
                  // processors
                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        overlapMatrixBlock.begin()),
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        overlapMatrixBlock.begin()),
                      D * B,
                      tempReal,
                      tempImag,
                      streamDataMove);
                  else
                    devicecclMpiCommDomain.deviceDirectAllReduceWrapper(
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        overlapMatrixBlock.begin()),
                      dftfe::utils::makeDataTypeDeviceCompatible(
                        overlapMatrixBlock.begin()),
                      D * B,
                      streamDataMove);
                }

              DeviceCHECK(cudaMemcpyAsync(
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlockHost),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlock.begin()),
                D * B * sizeof(dataTypes::number),
                cudaMemcpyDeviceToHost,
                streamDataMove));

              // record completion of Device->CPU copy for current block
              DeviceCHECK(
                cudaEventRecord(copyEvents[blockCount], streamDataMove));

              // Check that Device->CPU on the current block has been completed.
              // If completed, perform blocking MPI commmunication on the
              // current block and copy to ScaLAPACK matri
              if (cudaEventSynchronize(copyEvents[blockCount]) == cudaSuccess)
                {
                  // Sum local XTrunc^{T}*XcBlock across domain decomposition
                  // processors
                  if (!dftParams.useDeviceDirectAllReduce)
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

      DeviceCHECK(cudaFreeHost(overlapMatrixBlockHost));
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          DeviceCHECK(cudaFree(tempReal));
          DeviceCHECK(cudaFree(tempImag));
        }

      // return cublas handle to default stream
      cublasSetStream(handle, NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          DeviceCHECK(cudaEventDestroy(computeEvents[i]));
          DeviceCHECK(cudaEventDestroy(copyEvents[i]));
        }

      DeviceCHECK(cudaStreamDestroy(streamCompute));
      DeviceCHECK(cudaStreamDestroy(streamDataMove));

      if (numberBandGroups > 1)
        {
          MPI_Barrier(interBandGroupComm);

          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, overlapMatPar, interBandGroupComm);
        }
    }


    void
    fillParallelOverlapMatMixedPrecScalapack(
      const dataTypes::numberDevice *                  X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const MPI_Comm &                                 mpiCommDomain,
      DeviceCCLWrapper &                               devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar,
      const dftParameters &                            dftParams)
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

      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);


      dftfe::utils::MemoryStorage<dataTypes::numberFP32,dftfe::utils::MemorySpace::DEVICE> 
                                                           overlapMatrixBlockSP(N * vectorsBlockSize,
                             dataTypes::numberFP32(0));
      dftfe::utils::MemoryStorage<dataTypes::number,dftfe::utils::MemorySpace::DEVICE> overlapMatrixBlockDP(
        vectorsBlockSize * vectorsBlockSize,
        dataTypes::number(0));

      const unsigned int MPadded = std::ceil(M * 1.0 / 8.0) * 8.0 + 0.5;
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,dftfe::utils::MemorySpace::DEVICE>  XSP(
        MPadded * N, dataTypes::numberFP32(0));

      convDoubleArrToFloatArr<<<(N + (deviceConstants::blockSize - 1)) /
                                  deviceConstants::blockSize * M,
                                deviceConstants::blockSize>>>(
        N * M,
        X,
        dftfe::utils::makeDataTypeDeviceCompatible(XSP.begin()));
      dataTypes::number *overlapMatrixBlockHostDP;
      DeviceCHECK(cudaMallocHost((void **)&overlapMatrixBlockHostDP,
                                 vectorsBlockSize * vectorsBlockSize *
                                   sizeof(dataTypes::number)));
      std::memset(overlapMatrixBlockHostDP,
                  0,
                  vectorsBlockSize * vectorsBlockSize *
                    sizeof(dataTypes::number));

      dataTypes::numberFP32 *overlapMatrixBlockHostSP;
      DeviceCHECK(
        cudaMallocHost((void **)&overlapMatrixBlockHostSP,
                       N * vectorsBlockSize * sizeof(dataTypes::numberFP32)));
      std::memset(overlapMatrixBlockHostSP,
                  0,
                  N * vectorsBlockSize * sizeof(dataTypes::numberFP32));

      cudaStream_t streamDeviceCCL;
      cudaStreamCreate(&streamDeviceCCL);

      const dataTypes::number     scalarCoeffAlpha = dataTypes::number(1.0);
      const dataTypes::number     scalarCoeffBeta  = dataTypes::number(0);
      const dataTypes::numberFP32 scalarCoeffAlphaSP =
        dataTypes::numberFP32(1.0);
      const dataTypes::numberFP32 scalarCoeffBetaSP =
        dataTypes::numberFP32(0);

      dataTypes::numberValueType *    tempReal;
      dataTypes::numberValueType *    tempImag;
      dataTypes::numberFP32ValueType *tempRealFP32;
      dataTypes::numberFP32ValueType *tempImagFP32;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          DeviceCHECK(cudaMalloc((void **)&tempReal,
                                 vectorsBlockSize * N *
                                   sizeof(dataTypes::numberValueType)));
          DeviceCHECK(cudaMalloc((void **)&tempImag,
                                 vectorsBlockSize * N *
                                   sizeof(dataTypes::numberValueType)));
          DeviceCHECK(cudaMalloc((void **)&tempRealFP32,
                                 vectorsBlockSize * N *
                                   sizeof(dataTypes::numberFP32ValueType)));
          DeviceCHECK(cudaMalloc((void **)&tempImagFP32,
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
                dftfe::utils::makeDataTypeDeviceCompatible(
                  &scalarCoeffAlpha),
                X + ivec,
                N,
                X + ivec,
                N,
                dftfe::utils::makeDataTypeDeviceCompatible(
                  &scalarCoeffBeta),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlockDP.begin()),
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
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      &scalarCoeffAlphaSP),
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      XSP.begin() + ivec + B),
                    N,
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      XSP.begin() + ivec),
                    N,
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      &scalarCoeffBetaSP),
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      overlapMatrixBlockSP.begin()),
                    DRem);
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    devicecclMpiCommDomain
                      .deviceDirectAllReduceMixedPrecGroupWrapper(
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          overlapMatrixBlockDP.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          overlapMatrixBlockSP.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          overlapMatrixBlockDP.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          overlapMatrixBlockSP.begin()),
                        B * B,
                        DRem * B,
                        tempReal,
                        tempRealFP32,
                        tempImag,
                        tempImagFP32,
                        streamDeviceCCL);
                  else
                    devicecclMpiCommDomain
                      .deviceDirectAllReduceMixedPrecGroupWrapper(
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          overlapMatrixBlockDP.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          overlapMatrixBlockSP.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          overlapMatrixBlockDP.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          overlapMatrixBlockSP.begin()),
                        B * B,
                        DRem * B,
                        streamDeviceCCL);
                }

              dftfe::utils::deviceMemcpyD2H(dftfe::utils::makeDataTypeDeviceCompatible(
                           overlapMatrixBlockHostDP),
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           overlapMatrixBlockDP.begin()),
                         B * B * sizeof(dataTypes::number));

              dftfe::utils::deviceMemcpyD2H(dftfe::utils::makeDataTypeDeviceCompatible(
                           overlapMatrixBlockHostSP),
                         dftfe::utils::makeDataTypeDeviceCompatible(
                           overlapMatrixBlockSP.begin()),
                         DRem * B * sizeof(dataTypes::numberFP32));

              if (!dftParams.useDeviceDirectAllReduce)
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

      DeviceCHECK(cudaFreeHost(overlapMatrixBlockHostDP));
      DeviceCHECK(cudaFreeHost(overlapMatrixBlockHostSP));

      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          DeviceCHECK(cudaFree(tempReal));
          DeviceCHECK(cudaFree(tempImag));
          DeviceCHECK(cudaFree(tempRealFP32));
          DeviceCHECK(cudaFree(tempImagFP32));
        }

      cudaStreamDestroy(streamDeviceCCL);

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
    // COP denotes Device->CPU copy of X^{T} times XBlock
    // COM denotes blocking MPI_Allreduce on X^{T}XBlock and copy to scalapack
    // matrix
    // ----------------------------------------------------------
    // Two Device streams are created: compute and copy
    // CMP is performed in compute Device stream and COP is performed in copy
    // Device stream. COP for a block can only start after the CMP for that
    // block in the compute stream is completed. COM is performed for a block
    // only after COP even for that block is completed.
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
      const dataTypes::numberDevice *                  X,
      const unsigned int                               M,
      const unsigned int                               N,
      cublasHandle_t &                                 handle,
      const MPI_Comm &                                 mpiCommDomain,
      DeviceCCLWrapper &                               devicecclMpiCommDomain,
      const MPI_Comm &                                 interBandGroupComm,
      const std::shared_ptr<const dftfe::ProcessGrid> &processGrid,
      dftfe::ScaLAPACKMatrix<dataTypes::number> &      overlapMatPar,
      const dftParameters &                            dftParams)
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

      const unsigned int vectorsBlockSize = std::min(dftParams.wfcBlockSize, N);
      const unsigned int numberBlocks     = N / vectorsBlockSize;

      // create separate Device streams for Device->CPU copy and computation
      cudaStream_t streamCompute, streamDataMove;
      cudaStreamCreate(&streamCompute);
      cudaStreamCreate(&streamDataMove);

      // attach cublas handle to compute stream
      cublasSetStream(handle, streamCompute);

      // create array of compute and copy events on Devices
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

      dftfe::utils::MemoryStorage<dataTypes::numberFP32,dftfe::utils::MemorySpace::DEVICE> 
                                                           overlapMatrixBlockSP(N * vectorsBlockSize,
                             dataTypes::numberFP32(0));
      dftfe::utils::MemoryStorage<dataTypes::number,dftfe::utils::MemorySpace::DEVICE> overlapMatrixBlockDP(
        vectorsBlockSize * vectorsBlockSize,
        dataTypes::number(0));
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,dftfe::utils::MemorySpace::DEVICE> 
        overlapMatrixBlockSPNext(N * vectorsBlockSize,
                                 dataTypes::numberFP32(0));
      dftfe::utils::MemoryStorage<dataTypes::number,dftfe::utils::MemorySpace::DEVICE>
        overlapMatrixBlockDPNext(vectorsBlockSize * vectorsBlockSize,
                                 dataTypes::number(0));

      const unsigned int MPadded = std::ceil(M * 1.0 / 8.0) * 8.0 + 0.5;
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,dftfe::utils::MemorySpace::DEVICE>  XSP(
        MPadded * N, dataTypes::numberFP32(0));

      convDoubleArrToFloatArr<<<(N + (deviceConstants::blockSize - 1)) /
                                  deviceConstants::blockSize * M,
                                deviceConstants::blockSize>>>(
        N * M,
        X,
        dftfe::utils::makeDataTypeDeviceCompatible(
          XSP.begin()));
      dataTypes::number *overlapMatrixBlockHostDP;
      DeviceCHECK(cudaMallocHost((void **)&overlapMatrixBlockHostDP,
                                 vectorsBlockSize * vectorsBlockSize *
                                   sizeof(dataTypes::number)));
      std::memset(overlapMatrixBlockHostDP,
                  0,
                  vectorsBlockSize * vectorsBlockSize *
                    sizeof(dataTypes::number));

      dataTypes::numberFP32 *overlapMatrixBlockHostSP;
      DeviceCHECK(
        cudaMallocHost((void **)&overlapMatrixBlockHostSP,
                       N * vectorsBlockSize * sizeof(dataTypes::numberFP32)));
      std::memset(overlapMatrixBlockHostSP,
                  0,
                  N * vectorsBlockSize * sizeof(dataTypes::numberFP32));

      const dataTypes::number     scalarCoeffAlpha = dataTypes::number(1.0);
      const dataTypes::number     scalarCoeffBeta  = dataTypes::number(0);
      const dataTypes::numberFP32 scalarCoeffAlphaSP =
        dataTypes::numberFP32(1.0);
      const dataTypes::numberFP32 scalarCoeffBetaSP =
        dataTypes::numberFP32(0);

      dataTypes::numberValueType *    tempReal;
      dataTypes::numberValueType *    tempImag;
      dataTypes::numberFP32ValueType *tempRealFP32;
      dataTypes::numberFP32ValueType *tempImagFP32;
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          DeviceCHECK(cudaMalloc((void **)&tempReal,
                                 vectorsBlockSize * N *
                                   sizeof(dataTypes::numberValueType)));
          DeviceCHECK(cudaMalloc((void **)&tempImag,
                                 vectorsBlockSize * N *
                                   sizeof(dataTypes::numberValueType)));
          DeviceCHECK(cudaMalloc((void **)&tempRealFP32,
                                 vectorsBlockSize * N *
                                   sizeof(dataTypes::numberFP32ValueType)));
          DeviceCHECK(cudaMalloc((void **)&tempImagFP32,
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
                  // thrust::fill(overlapMatrixBlockDP.begin(),overlapMatrixBlockDP.end(),0);

                  cublasXgemm(handle,
                              CUBLAS_OP_N,
                              std::is_same<dataTypes::number,
                                           std::complex<double>>::value ?
                                CUBLAS_OP_C :
                                CUBLAS_OP_T,
                              B,
                              B,
                              M,
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                &scalarCoeffAlpha),
                              X + ivec,
                              N,
                              X + ivec,
                              N,
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                &scalarCoeffBeta),
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                  overlapMatrixBlockDP.begin()),
                              B);

                  const unsigned int DRem = D - B;

                  if (DRem != 0)
                    {
                      // thrust::fill(overlapMatrixBlockSP.begin(),overlapMatrixBlockSP.end(),0);

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
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          &scalarCoeffAlphaSP),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          XSP.begin() + ivec + B),
                        N,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          XSP.begin() + ivec),
                        N,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          &scalarCoeffBetaSP),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          overlapMatrixBlockSP.begin()),
                        DRem);
                    }

                  // record completion of compute for first block
                  DeviceCHECK(
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
                  // thrust::fill(overlapMatrixBlockDPNext.begin(),overlapMatrixBlockDPNext.end(),0);

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
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                &scalarCoeffAlpha),
                              X + ivecNew,
                              N,
                              X + ivecNew,
                              N,
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                &scalarCoeffBeta),
                              dftfe::utils::makeDataTypeDeviceCompatible(
                                  overlapMatrixBlockDPNext.begin()),
                              BNew);

                  const unsigned int DRemNew = DNew - BNew;

                  if (DRemNew != 0)
                    {
                      // thrust::fill(overlapMatrixBlockSPNext.begin(),overlapMatrixBlockSPNext.end(),0);

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
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          &scalarCoeffAlphaSP),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          XSP.begin() + ivecNew + BNew),
                        N,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          XSP.begin() + ivecNew),
                        N,
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          &scalarCoeffBetaSP),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                            overlapMatrixBlockSPNext.begin()),
                        DRemNew);
                    }

                  // record completion of compute for next block
                  DeviceCHECK(cudaEventRecord(computeEvents[blockCount + 1],
                                              streamCompute));
                }

              if (dftParams.useDeviceDirectAllReduce)
                {
                  if (std::is_same<dataTypes::number,
                                   std::complex<double>>::value)
                    devicecclMpiCommDomain
                      .deviceDirectAllReduceMixedPrecGroupWrapper(
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          overlapMatrixBlockDP.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          overlapMatrixBlockSP.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          overlapMatrixBlockDP.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          overlapMatrixBlockSP.begin()),
                        B * B,
                        DRem * B,
                        tempReal,
                        tempRealFP32,
                        tempImag,
                        tempImagFP32,
                        streamDataMove);
                  else
                    devicecclMpiCommDomain
                      .deviceDirectAllReduceMixedPrecGroupWrapper(
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          overlapMatrixBlockDP.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          overlapMatrixBlockSP.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          overlapMatrixBlockDP.begin()),
                        dftfe::utils::makeDataTypeDeviceCompatible(
                          overlapMatrixBlockSP.begin()),
                        B * B,
                        DRem * B,
                        streamDataMove);
                }

              cudaMemcpyAsync(dftfe::utils::makeDataTypeDeviceCompatible(
                                overlapMatrixBlockHostDP),
                              dftfe::utils::makeDataTypeDeviceCompatible(
                              overlapMatrixBlockDP.begin()),
                              B * B * sizeof(dataTypes::number),
                              cudaMemcpyDeviceToHost,
                              streamDataMove);

              cudaMemcpyAsync(
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlockHostSP),
                dftfe::utils::makeDataTypeDeviceCompatible(
                  overlapMatrixBlockSP.begin()),
                DRem * B * sizeof(dataTypes::numberFP32),
                cudaMemcpyDeviceToHost,
                streamDataMove);

              // record completion of Device->CPU copy for current block
              cudaEventRecord(copyEvents[blockCount], streamDataMove);

              // Check that Device->CPU on the current block has been completed.
              // If completed, perform blocking MPI commmunication on the
              // current block and copy to ScaLAPACK matri
              if (cudaEventSynchronize(copyEvents[blockCount]) == cudaSuccess)
                {
                  const unsigned int DRem = D - B;

                  if (!dftParams.useDeviceDirectAllReduce)
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

      DeviceCHECK(cudaFreeHost(overlapMatrixBlockHostDP));
      DeviceCHECK(cudaFreeHost(overlapMatrixBlockHostSP));

      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          DeviceCHECK(cudaFree(tempReal));
          DeviceCHECK(cudaFree(tempImag));
          DeviceCHECK(cudaFree(tempRealFP32));
          DeviceCHECK(cudaFree(tempImagFP32));
        }

      // return cublas handle to default stream
      cublasSetStream(handle, NULL);

      for (int i = 0; i < numberBlocks; ++i)
        {
          DeviceCHECK(cudaEventDestroy(computeEvents[i]));
          DeviceCHECK(cudaEventDestroy(copyEvents[i]));
        }

      DeviceCHECK(cudaStreamDestroy(streamCompute));
      DeviceCHECK(cudaStreamDestroy(streamDataMove));

      if (numberBandGroups > 1)
        {
          MPI_Barrier(interBandGroupComm);

          linearAlgebraOperations::internal::sumAcrossInterCommScaLAPACKMat(
            processGrid, overlapMatPar, interBandGroupComm);
        }
    }



    void
    computeEigenResidualNorm(
      operatorDFTDeviceClass &                       operatorMatrix,
      dataTypes::numberDevice *                      X,
      distributedDeviceVec<dataTypes::numberDevice> &XBlock,
      distributedDeviceVec<dataTypes::numberDevice> &HXBlock,
      distributedDeviceVec<dataTypes::numberDevice> &projectorKetTimesVector,
      const unsigned int                             M,
      const unsigned int                             N,
      const std::vector<double> &                    eigenValues,
      const MPI_Comm &                               mpiCommDomain,
      const MPI_Comm &                               interBandGroupComm,
      cublasHandle_t &                               handle,
      std::vector<double> &                          residualNorm,
      const dftParameters &                          dftParams,
      const bool                                     useBandParal)
    {
      // band group parallelization data structures
      const unsigned int numberBandGroups =
        dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
      const unsigned int bandGroupTaskId =
        dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
      std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
      dftUtils::createBandParallelizationIndices(
        interBandGroupComm, N, bandGroupLowHighPlusOneIndices);


      const unsigned int            vectorsBlockSize = dftParams.wfcBlockSize;
      dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE> residualNormSquareDevice(N, 0);
      dftfe::utils::MemoryStorage<dataTypes::number,dftfe::utils::MemorySpace::DEVICE> HXBlockFull(
        vectorsBlockSize * M, dataTypes::number(0));
      dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE> residualSqDevice(vectorsBlockSize * M, 0);
      dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE> onesVecDevice(M, 1.0);


      dftfe::utils::MemoryStorage<double,dftfe::utils::MemorySpace::DEVICE> eigenValuesDevice(N, 0);
      dftfe::utils::deviceMemcpyH2D(eigenValuesDevice.begin(),
                 &eigenValues[0],
                 N * sizeof(double));

      const bool   scaleFlag = false;
      const double scalar    = 1.0;
      const double alpha = 1.0, beta = 0;

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
                std::min(dftParams.chebyWfcBlockSize, N);

              for (unsigned int k = jvec; k < jvec + B; k += chebyBlockSize)
                {
                  stridedCopyToBlockKernel<<<
                    (chebyBlockSize + (deviceConstants::blockSize - 1)) /
                      deviceConstants::blockSize * M,
                    deviceConstants::blockSize>>>(
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

                  stridedCopyFromBlockKernel<<<
                    (chebyBlockSize + (deviceConstants::blockSize - 1)) /
                      deviceConstants::blockSize * M,
                    deviceConstants::blockSize>>>(
                    chebyBlockSize,
                    M,
                    HXBlock.begin(),
                    B,
                    dftfe::utils::makeDataTypeDeviceCompatible(
                      HXBlockFull.begin()),
                    k - jvec);
                }

              computeResidualDeviceKernel<<<(B +
                                             (deviceConstants::blockSize - 1)) /
                                              deviceConstants::blockSize * M,
                                            deviceConstants::blockSize>>>(
                B,
                M,
                N,
                jvec,
                eigenValuesDevice.begin(),
                X,
                dftfe::utils::makeDataTypeDeviceCompatible(
                HXBlockFull.begin()),
                residualSqDevice.begin());

              cublasDgemm(handle,
                          CUBLAS_OP_N,
                          CUBLAS_OP_T,
                          1,
                          B,
                          M,
                          &alpha,
                          onesVecDevice.begin(),
                          1,
                          residualSqDevice.begin(),
                          B,
                          &beta,
                          residualNormSquareDevice.begin() + jvec,
                          1);
            }
        }


      dftfe::utils::deviceMemcpyD2H(&residualNorm[0],
                 residualNormSquareDevice.begin(),
                 N * sizeof(double));

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
  } // namespace linearAlgebraOperationsDevice
} // namespace dftfe
