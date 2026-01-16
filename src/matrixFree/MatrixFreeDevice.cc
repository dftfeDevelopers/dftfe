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
#include "DeviceKernelLauncherHelpers.h"
#include "DeviceDataTypeOverloads.cu.h"

namespace dftfe
{
  constexpr std::uint32_t maxDofsPerDim = 17;
  __constant__ double
    constMem[maxDofsPerDim * maxDofsPerDim * 5 + maxDofsPerDim];

  __device__ inline std::uint32_t
  getMultiVectorIndex(const std::uint32_t node,
                      const std::uint32_t batch,
                      const std::uint32_t nOwnedDofs,
                      const std::uint32_t nGhostDofs,
                      const std::uint32_t *__restrict__ ghostMap)
  {
    return (node < nOwnedDofs ?
              (node + batch * nOwnedDofs) :
              (ghostMap[node - nOwnedDofs + batch * nGhostDofs]));
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
    const std::uint32_t nOwnedDofs,
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
        std::uint32_t idx;

        if constexpr (batchSize == 1)
          idx = constrainingNodeBuckets[k + constrainingBucketStart];
        else
          idx = getMultiVectorIndex(
            constrainingNodeBuckets[k + constrainingBucketStart],
            blockIdx.y,
            nOwnedDofs,
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

        std::uint32_t idx;

        if constexpr (batchSize == 1)
          idx = constrainedNodeBuckets[j + constrainedBucketStart];
        else
          idx = getMultiVectorIndex(
            constrainedNodeBuckets[j + constrainedBucketStart],
            blockIdx.y,
            nOwnedDofs,
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
    const std::uint32_t nOwnedDofs,
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
            std::uint32_t idx;

            if constexpr (batchSize == 1)
              idx = constrainedNodeBuckets[k + constrainedBucketStart];
            else
              idx = getMultiVectorIndex(
                constrainedNodeBuckets[k + constrainedBucketStart],
                blockIdx.y,
                nOwnedDofs,
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

            std::uint32_t idx;

            if constexpr (batchSize == 1)
              idx = constrainingNodeBuckets[j + constrainingBucketStart];
            else
              idx = getMultiVectorIndex(
                constrainingNodeBuckets[j + constrainingBucketStart],
                blockIdx.y,
                nOwnedDofs,
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
            std::uint32_t idx;

            if constexpr (batchSize == 1)
              idx = constrainedNodeBuckets[k + constrainedBucketStart];
            else
              idx = getMultiVectorIndex(
                constrainedNodeBuckets[k + constrainedBucketStart],
                blockIdx.y,
                nOwnedDofs,
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

    extern __shared__ __align__(sizeof(T)) unsigned char sharedMem[];

    constexpr std::uint32_t padding = 0;
    constexpr std::uint32_t pOdd    = nDofsPerDim / 2;
    constexpr std::uint32_t pEven   = nDofsPerDim % 2 == 1 ? pOdd + 1 : pOdd;
    constexpr std::uint32_t qOdd    = nQuadPointsPerDim / 2;
    constexpr std::uint32_t qEven =
      nQuadPointsPerDim % 2 == 1 ? qOdd + 1 : qOdd;

    T *__restrict__ sharedU = reinterpret_cast<T *>(sharedMem);
    T *__restrict__ sharedV = &sharedU[batchSize * nQuadPointsPerDim *
                                         nQuadPointsPerDim * nQuadPointsPerDim +
                                       padding];

    T *__restrict__ constN      = reinterpret_cast<T *>(constMem);
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
          sharedV[threadIdx.x + a * batchSize + qOdd * batchSize * nDofsPerDim +
                  b * batchSize * nDofsPerDim * nQuadPointsPerDim] = regT[qOdd];
      }

    __syncthreads();

    // 3rd GEMM of N
    // X Direction
    for (std::uint32_t i = threadIdx.y;
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
            for (std::uint32_t j = 0; j < qEven; j++)
              regT[j] += constN[j + k * qEven] * tempE;

#pragma unroll
            for (std::uint32_t j = 0; j < qOdd; j++)
              regT[j + qEven] += constN[j + k * qOdd + qEven * pEven] * tempO;
          }

        if constexpr (nDofsPerDim % 2 == 1)
          {
            tempE = sharedV[threadIdx.x + pOdd * batchSize +
                            i * batchSize * nDofsPerDim];

#pragma unroll
            for (std::uint32_t j = 0; j < qEven; j++)
              regT[j] += constN[j + pOdd * qEven] * tempE;
          }

#pragma unroll
        for (std::uint32_t j = 0; j < qOdd; j++)
          {
            sharedU[threadIdx.x + j * batchSize +
                    i * batchSize * nQuadPointsPerDim] =
              regT[j] + regT[j + qEven];

            sharedU[threadIdx.x + (nQuadPointsPerDim - 1 - j) * batchSize +
                    i * batchSize * nQuadPointsPerDim] =
              regT[j] - regT[j + qEven];
          }

        if constexpr (nQuadPointsPerDim % 2 == 1)
          sharedU[threadIdx.x + qOdd * batchSize +
                  i * batchSize * nQuadPointsPerDim] = regT[qOdd];
      }

    __syncthreads();

    //////////////////////////////////////////////////////////////////
    // Grad operation in each direction
    // sharedU -> Nx.Ny.Nz.Uxyz
    // regR    -> Dz.Nx.Ny.Nz.Uxyz
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
              regT[j] += constD[j + k * qOdd] * tempE;

#pragma unroll
            for (std::uint32_t j = 0; j < qEven; j++)
              regT[j + qOdd] += constD[j + k * qEven + qOdd * qEven] * tempO;
          }

        if constexpr (nQuadPointsPerDim % 2 == 1)
          {
            tempE =
              sharedU[threadIdx.x + i * batchSize +
                      qOdd * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

#pragma unroll
            for (std::uint32_t j = 0; j < qOdd; j++)
              regT[j] += constD[j + qOdd * qOdd] * tempE;
          }

#pragma unroll
        for (std::uint32_t j = 0; j < qOdd; j++)
          {
            regR[j]                         = regT[j + qOdd] + regT[j];
            regR[nQuadPointsPerDim - 1 - j] = regT[j + qOdd] - regT[j];
          }

        if constexpr (nQuadPointsPerDim % 2 == 1)
          regR[qOdd] = regT[2 * qOdd];
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
              regT[j] += constD[j + k * qOdd] * tempE;

#pragma unroll
            for (std::uint32_t j = 0; j < qEven; j++)
              regT[j + qOdd] += constD[j + k * qEven + qOdd * qEven] * tempO;
          }

        if constexpr (nQuadPointsPerDim % 2 == 1)
          {
            tempE =
              sharedU[threadIdx.x + a * batchSize +
                      qOdd * batchSize * nQuadPointsPerDim +
                      b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

#pragma unroll
            for (std::uint32_t j = 0; j < qOdd; j++)
              regT[j] += constD[j + qOdd * qOdd] * tempE;
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

        if constexpr (nQuadPointsPerDim % 2 == 1)
          sharedV[threadIdx.x + a * batchSize +
                  qOdd * batchSize * nQuadPointsPerDim +
                  b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
            regT[2 * qOdd];
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
              regT[j] += constD[j + k * qOdd] * tempE;

#pragma unroll
            for (std::uint32_t j = 0; j < qEven; j++)
              regT[j + qOdd] += constD[j + k * qEven + qOdd * qEven] * tempO;
          }

        if constexpr (nQuadPointsPerDim % 2 == 1)
          {
            tempE = sharedU[threadIdx.x + qOdd * batchSize +
                            i * batchSize * nQuadPointsPerDim];

#pragma unroll
            for (std::uint32_t j = 0; j < qOdd; j++)
              regT[j] += constD[j + qOdd * qOdd] * tempE;
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

        if constexpr (nQuadPointsPerDim % 2 == 1)
          sharedU[threadIdx.x + qOdd * batchSize +
                  i * batchSize * nQuadPointsPerDim] = regT[2 * qOdd];
      }

    __syncthreads();

    //////////////////////////////////////////////////////////////////
    // Jacobian Action
    // coeff.J^-T.J^-1.[sharedU sharedV regR]

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
    // Grad operation in each direction
    // regR -> Dz.Nx.Ny.Nz.Uxyz
    // regQ -> Dy.Nx.Ny.Nz.Uxyz
    // regP -> Dx.Nx.Ny.Nz.Uxyz
    // regR -> [DT.coeff.JF.D].Nx.Ny.Nz.Uxyz

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
              regT[j] += constDT[j + k * qOdd + qEven * qOdd] * tempE;

#pragma unroll
            for (std::uint32_t j = 0; j < qEven; j++)
              regT[j + qOdd] += constDT[j + k * qEven] * tempO;
          }

        if constexpr (nQuadPointsPerDim % 2 == 1)
          {
#pragma unroll
            for (std::uint32_t j = 0; j < qOdd; j++)
              regT[j] += constDT[j + qOdd * qOdd + qEven * qOdd] * regR[qOdd];
          }

#pragma unroll
        for (std::uint32_t j = 0; j < qOdd; j++)
          {
            regR[j]                         = regT[j + qOdd] + regT[j];
            regR[nQuadPointsPerDim - 1 - j] = regT[j + qOdd] - regT[j];
          }

        if constexpr (nQuadPointsPerDim % 2 == 1)
          regR[qOdd] = regT[2 * qOdd];
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
              regT[j] += constDT[j + k * qOdd + qEven * qOdd] * tempE;

#pragma unroll
            for (std::uint32_t j = 0; j < qEven; j++)
              regT[j + qOdd] += constDT[j + k * qEven] * tempO;
          }

        if constexpr (nQuadPointsPerDim % 2 == 1)
          {
            tempE =
              sharedV[threadIdx.x + a * batchSize +
                      qOdd * batchSize * nQuadPointsPerDim +
                      b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

#pragma unroll
            for (std::uint32_t j = 0; j < qOdd; j++)
              regT[j] += constDT[j + qOdd * qOdd + qEven * qOdd] * tempE;
          }

#pragma unroll
        for (std::uint32_t j = 0; j < qOdd; j++)
          {
            regQ[j]                         = regT[j + qOdd] + regT[j];
            regQ[nQuadPointsPerDim - 1 - j] = regT[j + qOdd] - regT[j];
          }

        if constexpr (nQuadPointsPerDim % 2 == 1)
          regQ[qOdd] = regT[2 * qOdd];
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
              regT[j] += constDT[j + k * qOdd + qEven * qOdd] * tempE;

#pragma unroll
            for (std::uint32_t j = 0; j < qEven; j++)
              regT[j + qOdd] += constDT[j + k * qEven] * tempO;
          }

        if constexpr (nQuadPointsPerDim % 2 == 1)
          {
            tempE = sharedU[threadIdx.x + qOdd * batchSize +
                            i * batchSize * nQuadPointsPerDim];

#pragma unroll
            for (std::uint32_t j = 0; j < qOdd; j++)
              regT[j] += constDT[j + qOdd * qOdd + qEven * qOdd] * tempE;
          }

#pragma unroll
        for (std::uint32_t j = 0; j < qOdd; j++)
          {
            regP[j]                         = regT[j + qOdd] + regT[j];
            regP[nQuadPointsPerDim - 1 - j] = regT[j + qOdd] - regT[j];
          }

        if constexpr (nQuadPointsPerDim % 2 == 1)
          regP[qOdd] = regT[2 * qOdd];
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
    // V -> NTx.NTy.NTz.[DT.coeff.JF.D].Nx.Ny.Nz.Uxyz

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
            for (std::uint32_t j = 0; j < pEven; j++)
              regT[j] += constNT[j + k * pEven] * tempE;

#pragma unroll
            for (std::uint32_t j = 0; j < pOdd; j++)
              regT[j + pEven] += constNT[j + k * pOdd + pEven * qEven] * tempO;
          }

        if constexpr (nQuadPointsPerDim % 2 == 1)
          {
#pragma unroll
            for (std::uint32_t j = 0; j < pEven; j++)
              regT[j] += constNT[j + qOdd * pEven] * regR[qOdd];
          }

#pragma unroll
        for (std::uint32_t j = 0; j < pOdd; j++)
          {
            sharedV[threadIdx.x + i * batchSize +
                    j * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
              regT[j] + regT[j + pEven];

            sharedV[threadIdx.x + i * batchSize +
                    (nDofsPerDim - 1 - j) * batchSize * nQuadPointsPerDim *
                      nQuadPointsPerDim] = regT[j] - regT[j + pEven];
          }

        if constexpr (nDofsPerDim % 2 == 1)
          sharedV[threadIdx.x + i * batchSize +
                  pOdd * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
            regT[pOdd];
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
            for (std::uint32_t j = 0; j < pEven; j++)
              regT[j] += constNT[j + k * pEven] * tempE;

#pragma unroll
            for (std::uint32_t j = 0; j < pOdd; j++)
              regT[j + pEven] += constNT[j + k * pOdd + pEven * qEven] * tempO;
          }

        if constexpr (nQuadPointsPerDim % 2 == 1)
          {
            tempE =
              sharedV[threadIdx.x + a * batchSize +
                      qOdd * batchSize * nQuadPointsPerDim +
                      b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

#pragma unroll
            for (std::uint32_t j = 0; j < pEven; j++)
              regT[j] += constNT[j + qOdd * pEven] * tempE;
          }

#pragma unroll
        for (std::uint32_t j = 0; j < pOdd; j++)
          {
            sharedU[threadIdx.x + a * batchSize +
                    j * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nDofsPerDim] =
              regT[j] + regT[j + pEven];

            sharedU[threadIdx.x + a * batchSize +
                    (nDofsPerDim - 1 - j) * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nDofsPerDim] =
              regT[j] - regT[j + pEven];
          }

        if constexpr (nDofsPerDim % 2 == 1)
          sharedU[threadIdx.x + a * batchSize +
                  pOdd * batchSize * nQuadPointsPerDim +
                  b * batchSize * nQuadPointsPerDim * nDofsPerDim] = regT[pOdd];
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
            for (std::uint32_t j = 0; j < pEven; j++)
              regT[j] += constNT[j + k * pEven] * tempE;

#pragma unroll
            for (std::uint32_t j = 0; j < pOdd; j++)
              regT[j + pEven] += constNT[j + k * pOdd + pEven * qEven] * tempO;
          }

        if constexpr (nQuadPointsPerDim % 2 == 1)
          {
            tempE = sharedU[threadIdx.x + qOdd * batchSize +
                            i * batchSize * nQuadPointsPerDim];

#pragma unroll
            for (std::uint32_t j = 0; j < pEven; j++)
              regT[j] += constNT[j + qOdd * pEven] * tempE;
          }

#pragma unroll
        for (std::uint32_t j = 0; j < pOdd; j++)
          {
            std::uint32_t dof1 = __ldg(&map[j + i * nDofsPerDim + mapOffset]);
            atomicAdd(&dst[threadIdx.x + dof1], regT[j] + regT[j + pEven]);

            std::uint32_t dof2 =
              __ldg(&map[(nDofsPerDim - 1 - j) + i * nDofsPerDim + mapOffset]);
            atomicAdd(&dst[threadIdx.x + dof2], regT[j] - regT[j + pEven]);
          }

        if constexpr (nDofsPerDim % 2 == 1)
          {
            std::uint32_t dof = __ldg(&map[pOdd + i * nDofsPerDim + mapOffset]);
            atomicAdd(&dst[threadIdx.x + dof], regT[pOdd]);
          }
      }
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
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    constexpr std::uint32_t dim           = 3;
    constexpr std::size_t   sharedMemSize = 2 * batchSize * nQuadPointsPerDim *
                                          nQuadPointsPerDim *
                                          nQuadPointsPerDim * sizeof(T);

    // Copy shape functions and gradients to constant memory on device
    DEVICE_API_CHECK(cudaMemcpyToSymbol(constMem,
                                        constMemHost,
                                        constMemSize * sizeof(T),
                                        0,
                                        cudaMemcpyHostToDevice));

    int deviceId = 0;
    DEVICE_API_CHECK(cudaGetDevice(&deviceId));

    int maxDynSharedDefault = 0;

#  ifdef cudaDevAttrMaxDynamicSharedMemoryPerBlock
    DEVICE_API_CHECK(
      cudaDeviceGetAttribute(&maxDynSharedDefault,
                             cudaDevAttrMaxDynamicSharedMemoryPerBlock,
                             deviceId));
#  else
    // Fallback for older CUDA versions without the dynamic shared attribute
    DEVICE_API_CHECK(cudaDeviceGetAttribute(&maxDynSharedDefault,
                                            cudaDevAttrMaxSharedMemoryPerBlock,
                                            deviceId));
#  endif

    int maxDynSharedOptIn = 0;
    DEVICE_API_CHECK(cudaDeviceGetAttribute(
      &maxDynSharedOptIn, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceId));

    if (sharedMemSize > static_cast<std::size_t>(maxDynSharedDefault))
      {
        if (sharedMemSize > static_cast<std::size_t>(maxDynSharedOptIn))
          throw std::runtime_error(
            "Requested dynamic shared memory exceeds opt-in limit");

        DEVICE_API_CHECK(cudaFuncSetAttribute(
          laplaceKernel<T, nDofsPerDim, nQuadPointsPerDim, batchSize, dim>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          sharedMemSize));
      }

#endif
  }


  template <typename T,
            std::uint32_t nDofsPerDim,
            std::uint32_t nQuadPointsPerDim,
            std::uint32_t batchSize>
  inline void
  MatrixFreeDevice<T, nDofsPerDim, nQuadPointsPerDim, batchSize>::
    constraintsDistribute(T                   *src,
                          const std::uint32_t *constrainingNodeBuckets,
                          const std::uint32_t *constrainingNodeOffset,
                          const std::uint32_t *constrainedNodeBuckets,
                          const std::uint32_t *constrainedNodeOffset,
                          const T             *weightMatrixList,
                          const std::uint32_t *weightMatrixOffset,
                          const T             *inhomogenityList,
                          const std::uint32_t *ghostMap,
                          const std::uint32_t  inhomogenityListSize,
                          const std::uint32_t  nBatch,
                          const std::uint32_t  nOwnedDofs,
                          const std::uint32_t  nGhostDofs)
  {
    constexpr int yThreads = 64;

    dim3 blocks(inhomogenityListSize, nBatch, 1);
    dim3 threads(batchSize, yThreads, 1);

    constraintsDistributeKernel<double, nDofsPerDim, batchSize>
      <<<blocks, threads>>>(src,
                            constrainingNodeBuckets,
                            constrainingNodeOffset,
                            constrainedNodeBuckets,
                            constrainedNodeOffset,
                            weightMatrixList,
                            weightMatrixOffset,
                            inhomogenityList,
                            ghostMap,
                            nOwnedDofs,
                            nGhostDofs);
  }


  template <typename T,
            std::uint32_t nDofsPerDim,
            std::uint32_t nQuadPointsPerDim,
            std::uint32_t batchSize>
  inline void
  MatrixFreeDevice<T, nDofsPerDim, nQuadPointsPerDim, batchSize>::
    constraintsDistributeTranspose(T                   *dst,
                                   T                   *src,
                                   const std::uint32_t *constrainingNodeBuckets,
                                   const std::uint32_t *constrainingNodeOffset,
                                   const std::uint32_t *constrainedNodeBuckets,
                                   const std::uint32_t *constrainedNodeOffset,
                                   const T             *weightMatrixList,
                                   const std::uint32_t *weightMatrixOffset,
                                   const std::uint32_t *ghostMap,
                                   const std::uint32_t  inhomogenityListSize,
                                   const std::uint32_t  nBatch,
                                   const std::uint32_t  nOwnedDofs,
                                   const std::uint32_t  nGhostDofs)
  {
    constexpr int yThreads = 64;

    dim3 blocks(inhomogenityListSize, nBatch, 1);
    dim3 threads(batchSize, yThreads, 1);

    constraintsDistributeTransposeKernel<double, nDofsPerDim, batchSize>
      <<<blocks, threads>>>(dst,
                            src,
                            constrainingNodeBuckets,
                            constrainingNodeOffset,
                            constrainedNodeBuckets,
                            constrainedNodeOffset,
                            weightMatrixList,
                            weightMatrixOffset,
                            ghostMap,
                            nOwnedDofs,
                            nGhostDofs);
  }


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
    constexpr std::size_t sharedMemSize = 2 * batchSize * nQuadPointsPerDim *
                                          nQuadPointsPerDim *
                                          nQuadPointsPerDim * sizeof(T);

    const dim3 blocks(nCells, nBatch, 1);
    const dim3 threads(batchSize, yThreads, 1);

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    laplaceKernel<T, nDofsPerDim, nQuadPointsPerDim, batchSize, dim>
      <<<blocks, threads, sharedMemSize>>>(dst, src, jacobianFactor, map);
#endif
  }

#include "MatrixFreeDevice.inst.cc"
} // namespace dftfe
