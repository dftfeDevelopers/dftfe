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
//

/**
 * @author Gourab Panigrahi
 *
 */

#include <MatrixFreeDevice.h>
#include <iostream>
#include "DeviceKernelLauncherHelpers.h"
#include "DeviceDataTypeOverloads.cu.h"

namespace dftfe
{
  constexpr std::uint32_t maxDofsPerDim = 17;
  __constant__ double
    constMemDevice[maxDofsPerDim * maxDofsPerDim * 5 + maxDofsPerDim];

  __device__ inline std::uint32_t
  getMultiVectorIndexDevice(const std::uint32_t node,
                            const std::uint32_t batch,
                            const std::uint32_t nLocalDofs,
                            const std::uint32_t nGhostDofs,
                            const std::uint32_t *__restrict__ ghostMap)
  {
    return (node < nLocalDofs ?
              (node + batch * nLocalDofs) :
              (ghostMap[node - nLocalDofs + batch * nGhostDofs]));
  }


  template <typename T, std::uint32_t nDofsPerDim, std::uint32_t batchSize>
  __global__ void
  constraintsDistributeKernel(
    T *__restrict__ x,
    const std::uint32_t *__restrict__ constrainingNodeBuckets,
    const std::uint32_t *__restrict__ constrainingNodeOffset,
    const std::uint32_t *__restrict__ constrainedNodeBuckets,
    const std::uint32_t *__restrict__ constrainedNodeOffset,
    const T *__restrict__ weightMatrixList,
    const std::uint32_t *__restrict__ weightMatrixOffset,
    const T *__restrict__ inhomogenityList,
    const std::uint32_t *__restrict__ ghostMap,
    const std::uint32_t nLocalDofs,
    const std::uint32_t nGhostDofs)
  {
    __shared__ T sharedConstrainingData[batchSize * nDofsPerDim * nDofsPerDim];

    std::uint32_t constrainingBucketStart = constrainingNodeOffset[blockIdx.x];
    std::uint32_t constrainingBucketSize =
      constrainingNodeOffset[blockIdx.x + 1] -
      constrainingNodeOffset[blockIdx.x];

    for (std::uint32_t k = threadIdx.y; k < constrainingBucketSize;
         k += blockDim.y)
      {
        std::uint32_t idx = getMultiVectorIndexDevice(
          constrainingNodeBuckets[k + constrainingBucketStart],
          blockIdx.y,
          nLocalDofs,
          nGhostDofs,
          ghostMap);

        sharedConstrainingData[threadIdx.x + k * batchSize] =
          x[threadIdx.x + idx * batchSize];
      }

    __syncthreads();

    std::uint32_t constrainedBucketStart = constrainedNodeOffset[blockIdx.x];
    std::uint32_t constrainedBucketSize =
      constrainedNodeOffset[blockIdx.x + 1] - constrainedNodeOffset[blockIdx.x];
    std::uint32_t weightMatrixStart = weightMatrixOffset[blockIdx.x];

    T inhomogenity = inhomogenityList[blockIdx.x];

    for (std::uint32_t j = threadIdx.y; j < constrainedBucketSize;
         j += blockDim.y)
      {
        T tmp = inhomogenity;

        for (std::uint32_t k = 0; k < constrainingBucketSize; k++)
          tmp += weightMatrixList[k + j * constrainingBucketSize +
                                  weightMatrixStart] *
                 sharedConstrainingData[threadIdx.x + k * batchSize];

        std::uint32_t idx = getMultiVectorIndexDevice(
          constrainedNodeBuckets[j + constrainedBucketStart],
          blockIdx.y,
          nLocalDofs,
          nGhostDofs,
          ghostMap);

        x[threadIdx.x + idx * batchSize] = tmp;
      }
  }


  template <typename T, std::uint32_t nDofsPerDim, std::uint32_t batchSize>
  __global__ void
  constraintsDistributeTransposeKernel(
    T *__restrict__ Ax,
    T *__restrict__ x,
    const std::uint32_t *__restrict__ constrainingNodeBuckets,
    const std::uint32_t *__restrict__ constrainingNodeOffset,
    const std::uint32_t *__restrict__ constrainedNodeBuckets,
    const std::uint32_t *__restrict__ constrainedNodeOffset,
    const T *__restrict__ weightMatrixList,
    const std::uint32_t *__restrict__ weightMatrixOffset,
    const std::uint32_t *__restrict__ ghostMap,
    const std::uint32_t nLocalDofs,
    const std::uint32_t nGhostDofs)
  {
    __shared__ T
      sharedConstrainedData[batchSize * nDofsPerDim * nDofsPerDim * 4];

    std::uint32_t constrainingBucketStart = constrainingNodeOffset[blockIdx.x];
    std::uint32_t constrainingBucketSize =
      constrainingNodeOffset[blockIdx.x + 1] -
      constrainingNodeOffset[blockIdx.x];

    std::uint32_t constrainedBucketStart = constrainedNodeOffset[blockIdx.x];
    std::uint32_t constrainedBucketSize =
      constrainedNodeOffset[blockIdx.x + 1] - constrainedNodeOffset[blockIdx.x];

    if (constrainingBucketSize > 0)
      {
        for (std::uint32_t k = threadIdx.y; k < constrainedBucketSize;
             k += blockDim.y)
          {
            std::uint32_t idx = getMultiVectorIndexDevice(
              constrainedNodeBuckets[k + constrainedBucketStart],
              blockIdx.y,
              nLocalDofs,
              nGhostDofs,
              ghostMap);

            sharedConstrainedData[threadIdx.x + k * batchSize] =
              Ax[threadIdx.x + idx * batchSize];

            Ax[threadIdx.x + idx * batchSize] = 0.;
            x[threadIdx.x + idx * batchSize]  = 0.;
          }

        __syncthreads();

        std::uint32_t weightMatrixStart = weightMatrixOffset[blockIdx.x];

        for (std::uint32_t j = threadIdx.y; j < constrainingBucketSize;
             j += blockDim.y)
          {
            T tmp = 0.;

            for (std::uint32_t k = 0; k < constrainedBucketSize; k++)
              tmp += weightMatrixList[j + k * constrainingBucketSize +
                                      weightMatrixStart] *
                     sharedConstrainedData[threadIdx.x + k * batchSize];

            std::uint32_t idx = getMultiVectorIndexDevice(
              constrainingNodeBuckets[j + constrainingBucketStart],
              blockIdx.y,
              nLocalDofs,
              nGhostDofs,
              ghostMap);

            atomicAdd(&Ax[threadIdx.x + idx * batchSize], tmp);
          }
      }
    else
      {
        for (std::uint32_t k = threadIdx.y; k < constrainedBucketSize;
             k += blockDim.y)
          {
            std::uint32_t idx = getMultiVectorIndexDevice(
              constrainedNodeBuckets[k + constrainedBucketStart],
              blockIdx.y,
              nLocalDofs,
              nGhostDofs,
              ghostMap);

            Ax[threadIdx.x + idx * batchSize] = 0.;
            x[threadIdx.x + idx * batchSize]  = 0.;
          }
      }
  }


  template <typename T,
            std::uint32_t nDofsPerDim,
            std::uint32_t nQuadPointsPerDim,
            std::uint32_t batchSize,
            std::uint32_t dim>
  __global__ void
  laplaceKernel(T *__restrict__ dst,
                const T *__restrict__ src,
                const T *__restrict__ J,
                const std::uint32_t *__restrict__ map)
  {
    // dst = A.src
    // gridDim.x = cells;
    // gridDim.y = batch;
    // nVec = batchSize * batch;
    // batchSize -> No of vectors in shared memory
    // First index is the fastest (Order -> x, y, z)
    // N(nQuadPointsPerDim*nDofsPerDim),
    // D(nQuadPointsPerDim*nQuadPointsPerDim),
    // NT(nDofsPerDim*nQuadPointsPerDim),
    // DT(nQuadPointsPerDim*nQuadPointsPerDim)

    extern __shared__ __align__(sizeof(T)) unsigned char sharedMemory[];

    constexpr std::uint32_t padding = 0;
    constexpr std::uint32_t pOdd    = nDofsPerDim / 2;
    constexpr std::uint32_t pEven   = nDofsPerDim % 2 == 1 ? pOdd + 1 : pOdd;
    constexpr std::uint32_t qOdd    = nQuadPointsPerDim / 2;
    constexpr std::uint32_t qEven =
      nQuadPointsPerDim % 2 == 1 ? qOdd + 1 : qOdd;

    T *__restrict__ sharedU = reinterpret_cast<T *>(sharedMemory);
    T *__restrict__ sharedV = &sharedU[batchSize * nQuadPointsPerDim *
                                         nQuadPointsPerDim * nQuadPointsPerDim +
                                       padding];

    T *__restrict__ constN      = reinterpret_cast<T *>(constMemDevice);
    T *__restrict__ constD      = &constN[qEven * pEven + qOdd * pOdd];
    T *__restrict__ constNT     = &constD[2 * qEven * qOdd];
    T *__restrict__ constDT     = &constNT[pEven * qEven + pOdd * qOdd];
    T *__restrict__ constNprime = &constDT[2 * qEven * qOdd];
    T *__restrict__ constW      = &constNprime[nQuadPointsPerDim * nDofsPerDim];

    T regP[qEven + qOdd], regQ[qEven + qOdd], regR[qEven + qOdd],
      regT[qEven + qOdd];

    const std::uint32_t mapOffset = (blockIdx.x + blockIdx.y * gridDim.x) *
                                    nDofsPerDim * nDofsPerDim * nDofsPerDim;

    //////////////////////////////////////////////////////////////////
    // Interpolation combined with Extraction
    // sharedU -> Nx.Ny.Nz.src(xyz)
    // Nx.Ny.Nz.src(xyz) -> src.NT.NT.NT

    // 1st GEMM of N
    // Z Direction
    for (std::uint32_t i = threadIdx.y; i < nDofsPerDim * nDofsPerDim;
         i += blockDim.y)
      {
        // Unroll to exclude k = 0, eliminate memset
        memset(regT, 0, nQuadPointsPerDim * sizeof(T));

        for (std::uint32_t k = 0; k < nDofsPerDim; k++)
          {
            std::uint32_t dof =
              __ldg(&map[i + k * nDofsPerDim * nDofsPerDim + mapOffset]);
            regP[k] = src[threadIdx.x + dof];

#pragma unroll
            for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
              regT[j] += constNprime[j + k * nQuadPointsPerDim] * regP[k];
          }

#pragma unroll
        for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
          sharedU[threadIdx.x + i * batchSize +
                  j * batchSize * nDofsPerDim * nDofsPerDim] = regT[j];
      }

    __syncthreads();

    // 2nd GEMM of N
    // Y Direction
    for (std::uint32_t i = threadIdx.y; i < nDofsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        std::uint32_t a = i % nDofsPerDim;
        std::uint32_t b = i / nDofsPerDim;

        T tempE, tempO, temp1, temp2;

        memset(regT, 0, nQuadPointsPerDim * sizeof(T));

        for (std::uint32_t k = 0; k < pOdd; k++)
          {
            temp1 = sharedU[threadIdx.x + a * batchSize +
                            k * batchSize * nDofsPerDim +
                            b * batchSize * nDofsPerDim * nDofsPerDim];

            temp2 = sharedU[threadIdx.x + a * batchSize +
                            (nDofsPerDim - 1 - k) * batchSize * nDofsPerDim +
                            b * batchSize * nDofsPerDim * nDofsPerDim];

            tempE = temp1 + temp2;
            tempO = temp1 - temp2;

#pragma unroll
            for (std::uint32_t j = 0; j < qEven; j++)
              regT[j] += constN[j + k * qEven] * tempE;

#pragma unroll
            for (std::uint32_t j = 0; j < qOdd; j++)
              regT[j + qEven] += constN[j + k * qOdd + qEven * pEven] * tempO;
          }

        if constexpr (nDofsPerDim % 2 == 1)
          {
            tempE = sharedU[threadIdx.x + a * batchSize +
                            pOdd * batchSize * nDofsPerDim +
                            b * batchSize * nDofsPerDim * nDofsPerDim];

#pragma unroll
            for (std::uint32_t j = 0; j < qEven; j++)
              regT[j] += constN[j + pOdd * qEven] * tempE;
          }

#pragma unroll
        for (std::uint32_t j = 0; j < qOdd; j++)
          {
            sharedV[threadIdx.x + a * batchSize + j * batchSize * nDofsPerDim +
                    b * batchSize * nDofsPerDim * nQuadPointsPerDim] =
              regT[j] + regT[j + qEven];

            sharedV[threadIdx.x + a * batchSize +
                    (nQuadPointsPerDim - 1 - j) * batchSize * nDofsPerDim +
                    b * batchSize * nDofsPerDim * nQuadPointsPerDim] =
              regT[j] - regT[j + qEven];
          }

        if constexpr (nQuadPointsPerDim % 2 == 1)
          {
            sharedV[threadIdx.x + a * batchSize +
                    (qOdd)*batchSize * nDofsPerDim +
                    b * batchSize * nDofsPerDim * nQuadPointsPerDim] =
              regT[qOdd];
          }
      }

    __syncthreads();

    for (std::uint32_t i = threadIdx.y; i < nDofsPerDim * nDofsPerDim;
         i += blockDim.y)
      for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
        {
          std::uint32_t dof =
            __ldg(&map[i + j * nDofsPerDim * nDofsPerDim + mapOffset]);
          atomicAdd(&dst[threadIdx.x + dof],
                    sharedV[threadIdx.x + i * batchSize +
                            j * batchSize * nDofsPerDim * nDofsPerDim]);
        }

    // 3rd GEMM of N
    // X Direction
    /*for (std::uint32_t i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        T tempE, tempO, temp1, temp2;

        memset(regT, 0, nQuadPointsPerDim * sizeof(T));

        for (std::uint32_t k = 0; k < pOdd; k++)
          {
            temp1 = sharedV[threadIdx.x + k * batchSize +
                            i * batchSize * nDofsPerDim];

            temp2 = sharedV[threadIdx.x + (nDofsPerDim - 1 - k) * batchSize +
                            i * batchSize * nDofsPerDim];

            tempE = temp1 + temp2;
            tempO = temp1 - temp2;

#pragma unroll
            for (std::uint32_t j = 0; j < qOdd; j++)
              {
                regT[j] += constN[j + k * qOdd] * tempE;
                regT[j + qOdd] += constN[j + k * qOdd + qOdd * pOdd] * tempO;
              }
          }

#pragma unroll
        for (std::uint32_t j = 0; j < qOdd; j++)
          {
            sharedU[threadIdx.x + j * batchSize +
                    i * batchSize * nQuadPointsPerDim] =
              regT[j] + regT[j + qOdd];

            sharedU[threadIdx.x + (nQuadPointsPerDim - 1 - j) * batchSize +
                    i * batchSize * nQuadPointsPerDim] =
              regT[j] - regT[j + qOdd];
          }
      }

    __syncthreads();

    //////////////////////////////////////////////////////////////////
    // Grad operation in each direction
    // regR    -> Nx.Ny.Nz.Uxyz
    // regQ    -> Dz.Nx.Ny.Nz.Uxyz
    // sharedV -> Dy.Nx.Ny.Nz.Uxyz
    // sharedU -> Dx.Nx.Ny.Nz.Uxyz

    // 1st GEMM of D
    // Z Direction
    for (std::uint32_t i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        T tempE, tempO, temp1, temp2;

        memset(regT, 0, nQuadPointsPerDim * sizeof(T));

        for (std::uint32_t k = 0; k < qOdd; k++)
          {
            temp1 =
              sharedU[threadIdx.x + i * batchSize +
                      k * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

            temp2 = sharedU[threadIdx.x + i * batchSize +
                            (nQuadPointsPerDim - 1 - k) * batchSize *
                              nQuadPointsPerDim * nQuadPointsPerDim];

            tempE = temp1 + temp2;
            tempO = temp1 - temp2;

#pragma unroll
            for (std::uint32_t j = 0; j < qOdd; j++)
              {
                regT[j] += constD[j + k * qOdd] * tempE;
                regT[j + qOdd] += constD[j + k * qOdd + qOdd * qEven] * tempO;
              }
          }

#pragma unroll
        for (std::uint32_t j = 0; j < qOdd; j++)
          {
            regR[j]                         = regT[j + qOdd] + regT[j];
            regR[nQuadPointsPerDim - 1 - j] = regT[j + qOdd] - regT[j];
          }
      }

    // 2nd GEMM of D
    // Y Direction
    for (std::uint32_t i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        std::uint32_t a = i % nQuadPointsPerDim;
        std::uint32_t b = i / nQuadPointsPerDim;

        T tempE, tempO, temp1, temp2;

        memset(regT, 0, nQuadPointsPerDim * sizeof(T));

        for (std::uint32_t k = 0; k < qOdd; k++)
          {
            temp1 =
              sharedU[threadIdx.x + a * batchSize +
                      k * batchSize * nQuadPointsPerDim +
                      b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

            temp2 =
              sharedU[threadIdx.x + a * batchSize +
                      (nQuadPointsPerDim - 1 - k) * batchSize *
                        nQuadPointsPerDim +
                      b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

            tempE = temp1 + temp2;
            tempO = temp1 - temp2;

#pragma unroll
            for (std::uint32_t j = 0; j < qOdd; j++)
              {
                regT[j] += constD[j + k * qOdd] * tempE;
                regT[j + qOdd] += constD[j + k * qOdd + qOdd * qEven] * tempO;
              }
          }

#pragma unroll
        for (std::uint32_t j = 0; j < qOdd; j++)
          {
            sharedV[threadIdx.x + a * batchSize +
                    j * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
              regT[j + qOdd] + regT[j];

            sharedV[threadIdx.x + a * batchSize +
                    (nQuadPointsPerDim - 1 - j) * batchSize *
                      nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
              regT[j + qOdd] - regT[j];
          }
      }

    // 3rd GEMM of D
    // X Direction
    for (std::uint32_t i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        T tempE, tempO, temp1, temp2;

        memset(regT, 0, nQuadPointsPerDim * sizeof(T));

        for (std::uint32_t k = 0; k < qOdd; k++)
          {
            temp1 = sharedU[threadIdx.x + k * batchSize +
                            i * batchSize * nQuadPointsPerDim];

            temp2 =
              sharedU[threadIdx.x + (nQuadPointsPerDim - 1 - k) * batchSize +
                      i * batchSize * nQuadPointsPerDim];

            tempE = temp1 + temp2;
            tempO = temp1 - temp2;

#pragma unroll
            for (std::uint32_t j = 0; j < qOdd; j++)
              {
                regT[j] += constD[j + k * qOdd] * tempE;
                regT[j + qOdd] += constD[j + k * qOdd + qOdd * qEven] * tempO;
              }
          }
      }

    __syncthreads();

    for (std::uint32_t i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
#pragma unroll
        for (std::uint32_t j = 0; j < qOdd; j++)
          {
            sharedU[threadIdx.x + j * batchSize +
                    i * batchSize * nQuadPointsPerDim] =
              regT[j + qOdd] + regT[j];

            sharedU[threadIdx.x + (nQuadPointsPerDim - 1 - j) * batchSize +
                    i * batchSize * nQuadPointsPerDim] =
              regT[j + qOdd] - regT[j];
          }
      }

    __syncthreads();

    //////////////////////////////////////////////////////////////////
    // Jacobian Action
    // coeff.J.[sharedU sharedV regQ]

    for (std::uint32_t i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        T t[dim];

        std::uint32_t jOffset = blockIdx.x * dim * dim;

        // #pragma unroll
        for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
          {
            t[0] = sharedU[threadIdx.x +
                           (i + j * nQuadPointsPerDim * nQuadPointsPerDim) *
                             batchSize];
            t[1] = sharedV[threadIdx.x +
                           (i + j * nQuadPointsPerDim * nQuadPointsPerDim) *
                             batchSize];
            t[2] = regR[j];

            sharedU[threadIdx.x +
                    (i + j * nQuadPointsPerDim * nQuadPointsPerDim) *
                      batchSize] = J[0 + jOffset] * t[0] +
                                   J[1 + jOffset] * t[1] +
                                   J[2 + jOffset] * t[2];
            sharedV[threadIdx.x +
                    (i + j * nQuadPointsPerDim * nQuadPointsPerDim) *
                      batchSize] = J[3 + jOffset] * t[0] +
                                   J[4 + jOffset] * t[1] +
                                   J[5 + jOffset] * t[2];
            regR[j] = J[6 + jOffset] * t[0] + J[7 + jOffset] * t[1] +
                      J[8 + jOffset] * t[2];
          }
      }

    __syncthreads();

    //////////////////////////////////////////////////////////////////////////////////////////
    // regP -> [DT.(coeff.J.W.D + detJ.vGGA.W) + detJ.Veff.W +
    // detJ.W.(vGGA.D)].Nx.Ny.Nz.Uxyz

    // 1st GEMM of DT
    // Z Direction
    for (std::uint32_t i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        T tempE, tempO;

        memset(regT, 0, nQuadPointsPerDim * sizeof(T));

        for (std::uint32_t k = 0; k < qOdd; k++)
          {
            tempE = regR[k] + regR[nQuadPointsPerDim - 1 - k];
            tempO = regR[k] - regR[nQuadPointsPerDim - 1 - k];

#pragma unroll
            for (std::uint32_t j = 0; j < qOdd; j++)
              {
                regT[j] += constDT[j + k * qOdd + qOdd * qEven] * tempE;
                regT[j + qOdd] += constDT[j + k * qOdd] * tempO;
              }
          }

#pragma unroll
        for (std::uint32_t j = 0; j < qOdd; j++)
          {
            regR[j]                         = regT[j + qOdd] + regT[j];
            regR[nQuadPointsPerDim - 1 - j] = regT[j + qOdd] - regT[j];
          }
      }

    // 2nd GEMM of DT
    // Y Direction
    for (std::uint32_t i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        std::uint32_t a = i % nQuadPointsPerDim;
        std::uint32_t b = i / nQuadPointsPerDim;

        T tempE, tempO, temp1, temp2;

        memset(regT, 0, nQuadPointsPerDim * sizeof(T));

        for (std::uint32_t k = 0; k < qOdd; k++)
          {
            temp1 =
              sharedV[threadIdx.x + a * batchSize +
                      k * batchSize * nQuadPointsPerDim +
                      b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

            temp2 =
              sharedV[threadIdx.x + a * batchSize +
                      (nQuadPointsPerDim - 1 - k) * batchSize *
                        nQuadPointsPerDim +
                      b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

            tempE = temp1 + temp2;
            tempO = temp1 - temp2;

#pragma unroll
            for (std::uint32_t j = 0; j < qOdd; j++)
              {
                regT[j] += constDT[j + k * qOdd + qOdd * qEven] * tempE;
                regT[j + qOdd] += constDT[j + k * qOdd] * tempO;
              }
          }

#pragma unroll
        for (std::uint32_t j = 0; j < qOdd; j++)
          {
            regQ[j]                         = regT[j + qOdd] + regT[j];
            regQ[nQuadPointsPerDim - 1 - j] = regT[j + qOdd] - regT[j];
          }
      }

    // 3rd GEMM of DT
    // X Direction
    for (std::uint32_t i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        T tempE, tempO, temp1, temp2;

        memset(regT, 0, nQuadPointsPerDim * sizeof(T));

        for (std::uint32_t k = 0; k < qOdd; k++)
          {
            temp1 = sharedU[threadIdx.x + k * batchSize +
                            i * batchSize * nQuadPointsPerDim];

            temp2 =
              sharedU[threadIdx.x + (nQuadPointsPerDim - 1 - k) * batchSize +
                      i * batchSize * nQuadPointsPerDim];

            tempE = temp1 + temp2;
            tempO = temp1 - temp2;

#pragma unroll
            for (std::uint32_t j = 0; j < qOdd; j++)
              {
                regT[j] += constDT[j + k * qOdd + qOdd * qEven] * tempE;
                regT[j + qOdd] += constDT[j + k * qOdd] * tempO;
              }
          }

#pragma unroll
        for (std::uint32_t j = 0; j < qOdd; j++)
          {
            regP[j]                         = regT[j + qOdd] + regT[j];
            regP[nQuadPointsPerDim - 1 - j] = regT[j + qOdd] - regT[j];
          }
      }

    __syncthreads();

    for (std::uint32_t i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        std::uint32_t a = i % nQuadPointsPerDim;
        std::uint32_t b = i / nQuadPointsPerDim;

#pragma unroll
        for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
          sharedV[threadIdx.x + a * batchSize +
                  j * batchSize * nQuadPointsPerDim +
                  b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
            regQ[j];

#pragma unroll
        for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
          sharedU[threadIdx.x + j * batchSize +
                  i * batchSize * nQuadPointsPerDim] = regP[j];
      }

    __syncthreads();

    for (std::uint32_t i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
#pragma unroll
        for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
          {
            regR[j] =
              regR[j] +
              sharedU[threadIdx.x + i * batchSize +
                      j * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] +
              sharedV[threadIdx.x + i * batchSize +
                      j * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];
          }
      }

    __syncthreads();

    //////////////////////////////////////////////////////////////////////////////////////////////////////
    // Integration combined with Assembly
    // regQ -> NTx.NTy.NTz.[DT.(1/2.J.W.D + detJ.vGGA.W) + detJ.Veff.W +
    // detJ.W.(vGGA.D)].Nx.Ny.Nz.Uxyz

    // 1st GEMM of NT
    // Z Direction
    for (std::uint32_t i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        T tempE, tempO;

        memset(regT, 0, nQuadPointsPerDim * sizeof(T));

        for (std::uint32_t k = 0; k < qOdd; k++)
          {
            tempE = regR[k] + regR[nQuadPointsPerDim - 1 - k];
            tempO = regR[k] - regR[nQuadPointsPerDim - 1 - k];

#pragma unroll
            for (std::uint32_t j = 0; j < pOdd; j++)
              {
                regT[j] += constNT[j + k * pOdd] * tempE;
                regT[j + pOdd] += constNT[j + k * pOdd + qOdd * pOdd] * tempO;
              }
          }

#pragma unroll
        for (std::uint32_t j = 0; j < pOdd; j++)
          {
            sharedV[threadIdx.x + i * batchSize +
                    j * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
              regT[j] + regT[j + pOdd];

            sharedV[threadIdx.x + i * batchSize +
                    (nDofsPerDim - 1 - j) * batchSize * nQuadPointsPerDim *
                      nQuadPointsPerDim] = regT[j] - regT[j + pOdd];
          }
      }

    __syncthreads();

    // 2nd GEMM of NT
    // Y Direction
    for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nDofsPerDim;
         i += blockDim.y)
      {
        std::uint32_t a = i % nQuadPointsPerDim;
        std::uint32_t b = i / nQuadPointsPerDim;

        T tempE, tempO, temp1, temp2;

        memset(regT, 0, nQuadPointsPerDim * sizeof(T));

        for (std::uint32_t k = 0; k < qOdd; k++)
          {
            temp1 =
              sharedV[threadIdx.x + a * batchSize +
                      k * batchSize * nQuadPointsPerDim +
                      b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

            temp2 =
              sharedV[threadIdx.x + a * batchSize +
                      (nQuadPointsPerDim - 1 - k) * batchSize *
                        nQuadPointsPerDim +
                      b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

            tempE = temp1 + temp2;
            tempO = temp1 - temp2;

#pragma unroll
            for (std::uint32_t j = 0; j < pOdd; j++)
              {
                regT[j] += constNT[j + k * pOdd] * tempE;
                regT[j + pOdd] += constNT[j + k * pOdd + qOdd * pOdd] * tempO;
              }
          }

#pragma unroll
        for (std::uint32_t j = 0; j < pOdd; j++)
          {
            sharedU[threadIdx.x + a * batchSize +
                    j * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nDofsPerDim] =
              regT[j] + regT[j + pOdd];

            sharedU[threadIdx.x + a * batchSize +
                    (nDofsPerDim - 1 - j) * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nDofsPerDim] =
              regT[j] - regT[j + pOdd];
          }
      }

    __syncthreads();

    // 3rd GEMM of NT
    // X Direction
    for (std::uint32_t i = threadIdx.y; i < nDofsPerDim * nDofsPerDim;
         i += blockDim.y)
      {
        T tempE, tempO, temp1, temp2;

        memset(regT, 0, nQuadPointsPerDim * sizeof(T));

        for (std::uint32_t k = 0; k < qOdd; k++)
          {
            temp1 = sharedU[threadIdx.x + k * batchSize +
                            i * batchSize * nQuadPointsPerDim];

            temp2 =
              sharedU[threadIdx.x + (nQuadPointsPerDim - 1 - k) * batchSize +
                      i * batchSize * nQuadPointsPerDim];

            tempE = temp1 + temp2;
            tempO = temp1 - temp2;

#pragma unroll
            for (std::uint32_t j = 0; j < pOdd; j++)
              {
                regT[j] += constNT[j + k * pOdd] * tempE;
                regT[j + pOdd] += constNT[j + k * pOdd + +qOdd * pOdd] * tempO;
              }
          }

#pragma unroll
        for (std::uint32_t j = 0; j < pOdd; j++)
          {
            std::uint32_t dof1 = __ldg(&map[j + i * nDofsPerDim + mapOffset]);
            atomicAdd(&dst[threadIdx.x + dof1], regT[j] + regT[j + pOdd]);

            std::uint32_t dof2 =
              __ldg(&map[(nDofsPerDim - 1 - j) + i * nDofsPerDim + mapOffset]);
            atomicAdd(&dst[threadIdx.x + dof2], regT[j] - regT[j + pOdd]);
          }
      } //*/
  }


  template <typename T,
            std::uint32_t nDofsPerDim,
            std::uint32_t nQuadPointsPerDim,
            std::uint32_t batchSize>
  inline void
  MatrixFreeDevice<T, nDofsPerDim, nQuadPointsPerDim, batchSize>::init(
    T          *constMemHost,
    std::size_t constMemSize)
  {
    constexpr std::uint32_t dim           = 3;
    constexpr size_t        sharedMemSize = 2 * batchSize * nQuadPointsPerDim *
                                     nQuadPointsPerDim * nQuadPointsPerDim *
                                     sizeof(T);

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    cudaFuncSetAttribute(
      laplaceKernel<T, nDofsPerDim, nQuadPointsPerDim, batchSize, dim>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      sharedMemSize);

    cudaFuncSetSharedMemConfig(
      laplaceKernel<T, nDofsPerDim, nQuadPointsPerDim, batchSize, dim>,
      std::is_same_v<T, double> ? cudaSharedMemBankSizeEightByte :
                                  cudaSharedMemBankSizeFourByte);

    cudaMemcpyToSymbol(constMemDevice,
                       constMemHost,
                       constMemSize * sizeof(T),
                       0,
                       cudaMemcpyHostToDevice);
#endif
  }


  template <typename T,
            std::uint32_t nDofsPerDim,
            std::uint32_t nQuadPointsPerDim,
            std::uint32_t batchSize>
  inline void
  MatrixFreeDevice<T, nDofsPerDim, nQuadPointsPerDim, batchSize>::
    constraintsDistribute(T *src)
  {
    // if (d_constrainedNodeBucketsDevice.size() == 0)
    //   return;

    // constexpr int yThreads = 64;
    // const int     batch    = numberWaveFunctions / d_batchsize;

    // dim3 blocks(d_inhomogenityListDevice.size(), batch, 1);
    // dim3 threads(d_batchsize, yThreads, 1);

    // constraintsDistributeKernel<double, d_batchsize, d_ndofsPerDim>
    //   <<<blocks, threads>>>(XBlock,
    //                         constrainingNodeBuckets.data(),
    //                         d_constrainingNodeOffsetDevice.data(),
    //                         d_constrainedNodeBucketsDevice.data(),
    //                         d_constrainedNodeOffsetDevice.data(),
    //                         d_weightMatrixListDevice.data(),
    //                         d_weightMatrixOffsetDevice.data(),
    //                         d_inhomogenityListDevice.data(),
    //                         ghostMapDevice.data(),
    //                         d_nLocalDofs,
    //                         d_nGhostDofs);
  }


  template <typename T,
            std::uint32_t nDofsPerDim,
            std::uint32_t nQuadPointsPerDim,
            std::uint32_t batchSize>
  inline void
  MatrixFreeDevice<T, nDofsPerDim, nQuadPointsPerDim, batchSize>::
    constraintsDistributeTranspose(T *dst, T *src)
  {}


  template <typename T,
            std::uint32_t nDofsPerDim,
            std::uint32_t nQuadPointsPerDim,
            std::uint32_t batchSize>
  inline void
  MatrixFreeDevice<T, nDofsPerDim, nQuadPointsPerDim, batchSize>::
    constraintsSetZero(T *src)
  {}


  template <typename T,
            std::uint32_t nDofsPerDim,
            std::uint32_t nQuadPointsPerDim,
            std::uint32_t batchSize>
  inline void
  MatrixFreeDevice<T, nDofsPerDim, nQuadPointsPerDim, batchSize>::
    computeLaplaceX(T             *dst,
                    T             *src,
                    T             *jacobianFactor,
                    std::uint32_t *map,
                    std::uint32_t  nCells,
                    std::uint32_t  nBatch)
  {
    constexpr std::uint32_t dim = 3;
    constexpr std::uint32_t yThreads =
      dftfe::utils::DEVICE_WARP_SIZE * ((nQuadPointsPerDim * nQuadPointsPerDim +
                                         dftfe::utils::DEVICE_WARP_SIZE - 1) /
                                        dftfe::utils::DEVICE_WARP_SIZE);
    constexpr size_t sharedMemSize = 2 * batchSize * nQuadPointsPerDim *
                                     nQuadPointsPerDim * nQuadPointsPerDim *
                                     sizeof(T);

    const dim3 blocks(nCells, nBatch, 1);
    const dim3 threads(batchSize, yThreads, 1);

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    laplaceKernel<T, nDofsPerDim, nQuadPointsPerDim, batchSize, dim>
      <<<blocks, threads, sharedMemSize>>>(dst, src, jacobianFactor, map);
#endif
  }

#include "MatrixFreeDevice.inst.cc"
} // namespace dftfe
