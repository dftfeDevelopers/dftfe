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
// @author Gourab Panigrahi
//

#include <MatrixFreeDevice.h>

namespace dftfe
{
  constexpr int maxDofsPerDim = 17;
  __constant__ double
    constMemData[maxDofsPerDim * maxDofsPerDim * 5 + maxDofsPerDim];


  __device__ inline unsigned int
  getMultiVectorIndexDevice(const unsigned int node,
                            const unsigned int batch,
                            const unsigned int nLocalDofs,
                            const unsigned int nGhostDofs,
                            const unsigned int *__restrict__ ghostMap)
  {
    return (node < nLocalDofs ?
              (node + batch * nLocalDofs) :
              (ghostMap[node - nLocalDofs + batch * nGhostDofs]));
  }


  template <typename T, unsigned int nDofsPerDim, unsigned int batchSize>
  __global__ void
  constraintsDistributeKernel(
    T *__restrict__ x,
    const unsigned int *__restrict__ constrainingNodeBuckets,
    const unsigned int *__restrict__ constrainingNodeOffset,
    const unsigned int *__restrict__ constrainedNodeBuckets,
    const unsigned int *__restrict__ constrainedNodeOffset,
    const T *__restrict__ weightMatrixList,
    const unsigned int *__restrict__ weightMatrixOffset,
    const T *__restrict__ inhomogenityList,
    const unsigned int *__restrict__ ghostMap,
    const unsigned int nLocalDofs,
    const unsigned int nGhostDofs)
  {
    __shared__ T sharedConstrainingData[batchSize * nDofsPerDim * nDofsPerDim];

    unsigned int constrainingBucketStart = constrainingNodeOffset[blockIdx.x];
    unsigned int constrainingBucketSize =
      constrainingNodeOffset[blockIdx.x + 1] -
      constrainingNodeOffset[blockIdx.x];

    for (unsigned int k = threadIdx.y; k < constrainingBucketSize;
         k += blockDim.y)
      {
        unsigned int idx = getMultiVectorIndexDevice(
          constrainingNodeBuckets[k + constrainingBucketStart],
          blockIdx.y,
          nLocalDofs,
          nGhostDofs,
          ghostMap);

        sharedConstrainingData[threadIdx.x + k * batchSize] =
          x[threadIdx.x + idx * batchSize];
      }

    __syncthreads();

    unsigned int constrainedBucketStart = constrainedNodeOffset[blockIdx.x];
    unsigned int constrainedBucketSize =
      constrainedNodeOffset[blockIdx.x + 1] - constrainedNodeOffset[blockIdx.x];
    unsigned int weightMatrixStart = weightMatrixOffset[blockIdx.x];

    T inhomogenity = inhomogenityList[blockIdx.x];

    for (unsigned int j = threadIdx.y; j < constrainedBucketSize;
         j += blockDim.y)
      {
        T tmp = inhomogenity;

        for (unsigned int k = 0; k < constrainingBucketSize; k++)
          tmp += weightMatrixList[k + j * constrainingBucketSize +
                                  weightMatrixStart] *
                 sharedConstrainingData[threadIdx.x + k * batchSize];

        unsigned int idx = getMultiVectorIndexDevice(
          constrainedNodeBuckets[j + constrainedBucketStart],
          blockIdx.y,
          nLocalDofs,
          nGhostDofs,
          ghostMap);

        x[threadIdx.x + idx * batchSize] = tmp;
      }
  }


  template <typename T, unsigned int nDofsPerDim, unsigned int batchSize>
  __global__ void
  constraintsDistributeTransposeKernel(
    T *__restrict__ Ax,
    T *__restrict__ x,
    const unsigned int *__restrict__ constrainingNodeBuckets,
    const unsigned int *__restrict__ constrainingNodeOffset,
    const unsigned int *__restrict__ constrainedNodeBuckets,
    const unsigned int *__restrict__ constrainedNodeOffset,
    const T *__restrict__ weightMatrixList,
    const unsigned int *__restrict__ weightMatrixOffset,
    const unsigned int *__restrict__ ghostMap,
    const unsigned int nLocalDofs,
    const unsigned int nGhostDofs)
  {
    __shared__ T
      sharedConstrainedData[batchSize * nDofsPerDim * nDofsPerDim * 4];

    unsigned int constrainingBucketStart = constrainingNodeOffset[blockIdx.x];
    unsigned int constrainingBucketSize =
      constrainingNodeOffset[blockIdx.x + 1] -
      constrainingNodeOffset[blockIdx.x];

    unsigned int constrainedBucketStart = constrainedNodeOffset[blockIdx.x];
    unsigned int constrainedBucketSize =
      constrainedNodeOffset[blockIdx.x + 1] - constrainedNodeOffset[blockIdx.x];

    if (constrainingBucketSize > 0)
      {
        for (unsigned int k = threadIdx.y; k < constrainedBucketSize;
             k += blockDim.y)
          {
            unsigned int idx = getMultiVectorIndexDevice(
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

        unsigned int weightMatrixStart = weightMatrixOffset[blockIdx.x];

        for (unsigned int j = threadIdx.y; j < constrainingBucketSize;
             j += blockDim.y)
          {
            T tmp = 0.;

            for (unsigned int k = 0; k < constrainedBucketSize; k++)
              tmp += weightMatrixList[j + k * constrainingBucketSize +
                                      weightMatrixStart] *
                     sharedConstrainedData[threadIdx.x + k * batchSize];

            unsigned int idx = getMultiVectorIndexDevice(
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
        for (unsigned int k = threadIdx.y; k < constrainedBucketSize;
             k += blockDim.y)
          {
            unsigned int idx = getMultiVectorIndexDevice(
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
            unsigned int nDofsPerDim,
            unsigned int nQuadPointsPerDim,
            unsigned int batchSize,
            unsigned int dim>
  __global__ void
  laplaceKernel(T *__restrict__ V,
                const T *__restrict__ U,
                const T *__restrict__ J,
                const unsigned int *__restrict__ map)
  {
    // V = AU
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

    constexpr unsigned int padding = 0;
    constexpr unsigned int pOdd    = nDofsPerDim / 2;
    constexpr unsigned int pEven   = nDofsPerDim % 2 == 1 ? pOdd + 1 : pOdd;
    constexpr unsigned int qOdd    = nQuadPointsPerDim / 2;
    constexpr unsigned int qEven = nQuadPointsPerDim % 2 == 1 ? qOdd + 1 : qOdd;

    T *__restrict__ sharedP = reinterpret_cast<T *>(sharedMemory);
    T *__restrict__ sharedQ = &sharedP[batchSize * nQuadPointsPerDim *
                                         nQuadPointsPerDim * nQuadPointsPerDim +
                                       padding];

    T *__restrict__ constN      = reinterpret_cast<T *>(constMemData);
    T *__restrict__ constD      = &constN[qEven * pEven + qOdd * pOdd];
    T *__restrict__ constNT     = &constD[2 * qEven * qOdd];
    T *__restrict__ constDT     = &constNT[pEven * qEven + pOdd * qOdd];
    T *__restrict__ constNprime = &constDT[2 * qEven * qOdd];
    T *__restrict__ constW      = &constNprime[nQuadPointsPerDim * nDofsPerDim];

    T regP[qEven + qOdd], regQ[qEven + qOdd], regR[qEven + qOdd];

    const unsigned int mapOffset = (blockIdx.x + blockIdx.y * gridDim.x) *
                                   nDofsPerDim * nDofsPerDim * nDofsPerDim;

    //////////////////////////////////////////////////////////////////
    // Interpolation combined with Extraction
    // sharedP -> Nx.Ny.Nz.Uxyz
    // Nx.Ny.Nz.Uxyz -> U.NT.NT.NT

    // 1st GEMM of N
    // Z Direction
    for (unsigned int i = threadIdx.y; i < nDofsPerDim * nDofsPerDim;
         i += blockDim.y)
      {
        // Unroll to exclude k = 0, eliminate memset
        memset(regP, 0, nQuadPointsPerDim * sizeof(T));

        for (unsigned int k = 0; k < nDofsPerDim; k++)
          {
            unsigned int dof =
              __ldg(&map[i + k * nDofsPerDim * nDofsPerDim + mapOffset]);
            regQ[k] = U[threadIdx.x + dof];

#pragma unroll
            for (unsigned int j = 0; j < nQuadPointsPerDim; j++)
              regP[j] += constNprime[j + k * nQuadPointsPerDim] * regQ[k];
          }

#pragma unroll
        for (unsigned int j = 0; j < nQuadPointsPerDim; j++)
          sharedP[threadIdx.x + i * batchSize +
                  j * batchSize * nDofsPerDim * nDofsPerDim] = regP[j];
      }

    __syncthreads();

    // 2nd GEMM of N
    // Y Direction
    for (unsigned int i = threadIdx.y; i < nDofsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        unsigned int a = i % nDofsPerDim;
        unsigned int b = i / nDofsPerDim;

        memset(regQ, 0, nQuadPointsPerDim * sizeof(T));

        for (unsigned int k = 0; k < pOdd; k++)
          {
            regP[0] = sharedP[threadIdx.x + a * batchSize +
                              k * batchSize * nDofsPerDim +
                              b * batchSize * nDofsPerDim * nDofsPerDim];

            regP[1] = sharedP[threadIdx.x + a * batchSize +
                              (nDofsPerDim - 1 - k) * batchSize * nDofsPerDim +
                              b * batchSize * nDofsPerDim * nDofsPerDim];

            regP[2] = regP[0] + regP[1];
            regP[3] = regP[0] - regP[1];

#pragma unroll
            for (unsigned int j = 0; j < qOdd; j++)
              {
                regQ[j] += constN[j + k * qOdd] * regP[2];
                regQ[j + qOdd] += constN[j + k * qOdd + qOdd * pOdd] * regP[3];
              }
          }

#pragma unroll
        for (unsigned int j = 0; j < qOdd; j++)
          {
            sharedQ[threadIdx.x + a * batchSize + j * batchSize * nDofsPerDim +
                    b * batchSize * nDofsPerDim * nQuadPointsPerDim] =
              regQ[j] + regQ[j + qOdd];

            sharedQ[threadIdx.x + a * batchSize +
                    (nQuadPointsPerDim - 1 - j) * batchSize * nDofsPerDim +
                    b * batchSize * nDofsPerDim * nQuadPointsPerDim] =
              regQ[j] - regQ[j + qOdd];
          }
      }

    __syncthreads();

    // 3rd GEMM of N
    // X Direction
    for (unsigned int i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        memset(regP, 0, nQuadPointsPerDim * sizeof(T));

        for (unsigned int k = 0; k < pOdd; k++)
          {
            regQ[0] = sharedQ[threadIdx.x + k * batchSize +
                              i * batchSize * nDofsPerDim];

            regQ[1] = sharedQ[threadIdx.x + (nDofsPerDim - 1 - k) * batchSize +
                              i * batchSize * nDofsPerDim];

            regQ[2] = regQ[0] + regQ[1];
            regQ[3] = regQ[0] - regQ[1];

#pragma unroll
            for (unsigned int j = 0; j < qOdd; j++)
              {
                regP[j] += constN[j + k * qOdd] * regQ[2];
                regP[j + qOdd] += constN[j + k * qOdd + qOdd * pOdd] * regQ[3];
              }
          }

#pragma unroll
        for (unsigned int j = 0; j < qOdd; j++)
          {
            sharedP[threadIdx.x + j * batchSize +
                    i * batchSize * nQuadPointsPerDim] =
              regP[j] + regP[j + qOdd];

            sharedP[threadIdx.x + (nQuadPointsPerDim - 1 - j) * batchSize +
                    i * batchSize * nQuadPointsPerDim] =
              regP[j] - regP[j + qOdd];
          }
      }

    __syncthreads();

    //////////////////////////////////////////////////////////////////
    // Grad operation in each direction
    // regR    -> Nx.Ny.Nz.Uxyz
    // regQ    -> Dz.Nx.Ny.Nz.Uxyz
    // sharedQ -> Dy.Nx.Ny.Nz.Uxyz
    // sharedP -> Dx.Nx.Ny.Nz.Uxyz

    // 1st GEMM of D
    // Z Direction
    for (unsigned int i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        T RE, RO, tempR[nQuadPointsPerDim];

        memset(tempR, 0, nQuadPointsPerDim * sizeof(T));
        memset(regQ, 0, nQuadPointsPerDim * sizeof(T));

        for (unsigned int k = 0; k < qOdd; k++)
          {
            regR[k] =
              sharedP[threadIdx.x + i * batchSize +
                      k * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

            regR[nQuadPointsPerDim - 1 - k] =
              sharedP[threadIdx.x + i * batchSize +
                      (nQuadPointsPerDim - 1 - k) * batchSize *
                        nQuadPointsPerDim * nQuadPointsPerDim];

            RE = regR[k] + regR[nQuadPointsPerDim - 1 - k];
            RO = regR[k] - regR[nQuadPointsPerDim - 1 - k];

#pragma unroll
            for (unsigned int j = 0; j < qOdd; j++)
              {
                tempR[j] += constD[j + k * qOdd] * RE;
                tempR[j + qOdd] += constD[j + k * qOdd + qOdd * qEven] * RO;
              }
          }

#pragma unroll
        for (unsigned int j = 0; j < qOdd; j++)
          {
            regQ[j]                         = tempR[j + qOdd] + tempR[j];
            regQ[nQuadPointsPerDim - 1 - j] = tempR[j + qOdd] - tempR[j];
          }
      }

    // 2nd GEMM of D
    // Y Direction
    for (unsigned int i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        unsigned int a = i % nQuadPointsPerDim;
        unsigned int b = i / nQuadPointsPerDim;

        T PE, PO, temp1, temp2;

        memset(regP, 0, nQuadPointsPerDim * sizeof(T));

        for (unsigned int k = 0; k < qOdd; k++)
          {
            temp1 =
              sharedP[threadIdx.x + a * batchSize +
                      k * batchSize * nQuadPointsPerDim +
                      b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

            temp2 =
              sharedP[threadIdx.x + a * batchSize +
                      (nQuadPointsPerDim - 1 - k) * batchSize *
                        nQuadPointsPerDim +
                      b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

            PE = temp1 + temp2;
            PO = temp1 - temp2;

#pragma unroll
            for (unsigned int j = 0; j < qOdd; j++)
              {
                regP[j] += constD[j + k * qOdd] * PE;
                regP[j + qOdd] += constD[j + k * qOdd + qOdd * qEven] * PO;
              }
          }

#pragma unroll
        for (unsigned int j = 0; j < qOdd; j++)
          {
            sharedQ[threadIdx.x + a * batchSize +
                    j * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
              regP[j + qOdd] + regP[j];

            sharedQ[threadIdx.x + a * batchSize +
                    (nQuadPointsPerDim - 1 - j) * batchSize *
                      nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
              regP[j + qOdd] - regP[j];
          }
      }

    __syncthreads();

    // 3rd GEMM of D
    // X Direction
    for (unsigned int i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        T PE, PO, temp1, temp2;

        memset(regP, 0, nQuadPointsPerDim * sizeof(T));

        for (unsigned int k = 0; k < qOdd; k++)
          {
            temp1 = sharedP[threadIdx.x + k * batchSize +
                            i * batchSize * nQuadPointsPerDim];

            temp2 =
              sharedP[threadIdx.x + (nQuadPointsPerDim - 1 - k) * batchSize +
                      i * batchSize * nQuadPointsPerDim];

            PE = temp1 + temp2;
            PO = temp1 - temp2;

#pragma unroll
            for (unsigned int j = 0; j < qOdd; j++)
              {
                regP[j] += constD[j + k * qOdd] * PE;
                regP[j + qOdd] += constD[j + k * qOdd + qOdd * qEven] * PO;
              }
          }
      }

    __syncthreads();

    for (unsigned int i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
#pragma unroll
        for (unsigned int j = 0; j < qOdd; j++)
          {
            sharedP[threadIdx.x + j * batchSize +
                    i * batchSize * nQuadPointsPerDim] =
              regP[j + qOdd] + regP[j];

            sharedP[threadIdx.x + (nQuadPointsPerDim - 1 - j) * batchSize +
                    i * batchSize * nQuadPointsPerDim] =
              regP[j + qOdd] - regP[j];
          }
      }

    __syncthreads();

    //////////////////////////////////////////////////////////////////
    // Jacobian Action
    // 1/2.J.[sharedP sharedQ regQ]

    for (unsigned int i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        T v[3];

        unsigned int jOffset = blockIdx.x * dim * dim;

        // #pragma unroll
        for (unsigned int j = 0; j < nQuadPointsPerDim; j++)
          {
            v[0] = sharedP[threadIdx.x +
                           (i + j * nQuadPointsPerDim * nQuadPointsPerDim) *
                             batchSize];
            v[1] = sharedQ[threadIdx.x +
                           (i + j * nQuadPointsPerDim * nQuadPointsPerDim) *
                             batchSize];
            v[2] = regQ[j];

            sharedP[threadIdx.x +
                    (i + j * nQuadPointsPerDim * nQuadPointsPerDim) *
                      batchSize] = J[0 + jOffset] * v[0] +
                                   J[1 + jOffset] * v[1] +
                                   J[2 + jOffset] * v[2];
            sharedQ[threadIdx.x +
                    (i + j * nQuadPointsPerDim * nQuadPointsPerDim) *
                      batchSize] = J[3 + jOffset] * v[0] +
                                   J[4 + jOffset] * v[1] +
                                   J[5 + jOffset] * v[2];
            regQ[j] = J[6 + jOffset] * v[0] + J[7 + jOffset] * v[1] +
                      J[8 + jOffset] * v[2];
          }
      }

    __syncthreads();

    //////////////////////////////////////////////////////////////////////////////////////////
    // regP -> [DT.(1/2.J.W.D + detJ.vGGA.W) + detJ.Veff.W +
    // detJ.W.(vGGA.D)].Nx.Ny.Nz.Uxyz

    // 1st GEMM of DT
    // Z Direction
    for (unsigned int i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        T QE, QO, temp1, temp2;

        memset(regR, 0, nQuadPointsPerDim * sizeof(T));

        for (unsigned int k = 0; k < qOdd; k++)
          {
            QE = regQ[k] + regQ[nQuadPointsPerDim - 1 - k];
            QO = regQ[k] - regQ[nQuadPointsPerDim - 1 - k];

#pragma unroll
            for (unsigned int j = 0; j < qOdd; j++)
              {
                regR[j] += constDT[j + k * qOdd + qOdd * qEven] * QE;
                regR[j + qOdd] += constDT[j + k * qOdd] * QO;
              }
          }

#pragma unroll
        for (unsigned int j = 0; j < qOdd; j++)
          {
            regP[j] += regR[j + qOdd] + regR[j];
            regP[nQuadPointsPerDim - 1 - j] += regR[j + qOdd] - regR[j];
          }
      }

    __syncthreads();

    // 2nd GEMM of DT
    // Y Direction
    for (unsigned int i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        unsigned int a = i % nQuadPointsPerDim;
        unsigned int b = i / nQuadPointsPerDim;

        T QE, QO, temp1, temp2, tempQ[nQuadPointsPerDim];

        memset(tempQ, 0, nQuadPointsPerDim * sizeof(T));
        memset(regQ, 0, nQuadPointsPerDim * sizeof(T));

        for (unsigned int k = 0; k < qOdd; k++)
          {
            temp1 =
              sharedQ[threadIdx.x + a * batchSize +
                      k * batchSize * nQuadPointsPerDim +
                      b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

            temp2 =
              sharedQ[threadIdx.x + a * batchSize +
                      (nQuadPointsPerDim - 1 - k) * batchSize *
                        nQuadPointsPerDim +
                      b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

            QE = temp1 + temp2;
            QO = temp1 - temp2;

#pragma unroll
            for (unsigned int j = 0; j < qOdd; j++)
              {
                tempQ[j] += constDT[j + k * qOdd + qOdd * qEven] * QE;
                tempQ[j + qOdd] += constDT[j + k * qOdd] * QO;
              }
          }

#pragma unroll
        for (unsigned int j = 0; j < qOdd; j++)
          {
            regQ[j] += tempQ[j + qOdd] + tempQ[j];
            regQ[nQuadPointsPerDim - 1 - j] += tempQ[j + qOdd] - tempQ[j];
          }
      }

    // 3rd GEMM of DT
    // X Direction
    for (unsigned int i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        T PE, PO, temp1, temp2, tempR[nQuadPointsPerDim];

        memset(tempR, 0, nQuadPointsPerDim * sizeof(T));
        memset(regR, 0, nQuadPointsPerDim * sizeof(T));

        for (unsigned int k = 0; k < qOdd; k++)
          {
            temp1 = sharedP[threadIdx.x + k * batchSize +
                            i * batchSize * nQuadPointsPerDim];

            temp2 =
              sharedP[threadIdx.x + (nQuadPointsPerDim - 1 - k) * batchSize +
                      i * batchSize * nQuadPointsPerDim];

            PE = temp1 + temp2;
            PO = temp1 - temp2;

#pragma unroll
            for (unsigned int j = 0; j < qOdd; j++)
              {
                tempR[j] += constDT[j + k * qOdd + qOdd * qEven] * PE;
                tempR[j + qOdd] += constDT[j + k * qOdd] * PO;
              }
          }

#pragma unroll
        for (unsigned int j = 0; j < qOdd; j++)
          {
            regR[j] += tempR[j + qOdd] + tempR[j];
            regR[nQuadPointsPerDim - 1 - j] += tempR[j + qOdd] - tempR[j];
          }
      }

    __syncthreads();

    for (unsigned int i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        unsigned int a = i % nQuadPointsPerDim;
        unsigned int b = i / nQuadPointsPerDim;

#pragma unroll
        for (unsigned int j = 0; j < nQuadPointsPerDim; j++)
          sharedQ[threadIdx.x + a * batchSize +
                  j * batchSize * nQuadPointsPerDim +
                  b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
            regQ[j];

#pragma unroll
        for (unsigned int j = 0; j < nQuadPointsPerDim; j++)
          sharedP[threadIdx.x + j * batchSize +
                  i * batchSize * nQuadPointsPerDim] = regR[j];
      }

    __syncthreads();

    for (unsigned int i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
#pragma unroll
        for (unsigned int j = 0; j < nQuadPointsPerDim; j++)
          {
            regP[j] =
              regP[j] +
              sharedP[threadIdx.x + i * batchSize +
                      j * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] +
              sharedQ[threadIdx.x + i * batchSize +
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
    for (unsigned int i = threadIdx.y;
         i < nQuadPointsPerDim * nQuadPointsPerDim;
         i += blockDim.y)
      {
        memset(regQ, 0, nDofsPerDim * sizeof(T));

        for (unsigned int k = 0; k < qOdd; k++)
          {
            regR[0] = regP[k] + regP[nQuadPointsPerDim - 1 - k];
            regR[1] = regP[k] - regP[nQuadPointsPerDim - 1 - k];

#pragma unroll
            for (unsigned int j = 0; j < pOdd; j++)
              {
                regQ[j] += constNT[j + k * pOdd] * regR[0];
                regQ[j + pOdd] += constNT[j + k * pOdd + qOdd * pOdd] * regR[1];
              }
          }

#pragma unroll
        for (unsigned int j = 0; j < pOdd; j++)
          {
            sharedQ[threadIdx.x + i * batchSize +
                    j * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
              regQ[j] + regQ[j + pOdd];

            sharedQ[threadIdx.x + i * batchSize +
                    (nDofsPerDim - 1 - j) * batchSize * nQuadPointsPerDim *
                      nQuadPointsPerDim] = regQ[j] - regQ[j + pOdd];
          }
      }

    __syncthreads();

    // 2nd GEMM of NT
    // Y Direction
    for (unsigned int i = threadIdx.y; i < nQuadPointsPerDim * nDofsPerDim;
         i += blockDim.y)
      {
        unsigned int a = i % nQuadPointsPerDim;
        unsigned int b = i / nQuadPointsPerDim;

        memset(regP, 0, nDofsPerDim * sizeof(T));

        for (unsigned int k = 0; k < qOdd; k++)
          {
            regQ[0] =
              sharedQ[threadIdx.x + a * batchSize +
                      k * batchSize * nQuadPointsPerDim +
                      b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

            regQ[1] =
              sharedQ[threadIdx.x + a * batchSize +
                      (nQuadPointsPerDim - 1 - k) * batchSize *
                        nQuadPointsPerDim +
                      b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

            regR[0] = regQ[0] + regQ[1];
            regR[1] = regQ[0] - regQ[1];

#pragma unroll
            for (unsigned int j = 0; j < pOdd; j++)
              {
                regP[j] += constNT[j + k * pOdd] * regR[0];
                regP[j + pOdd] += constNT[j + k * pOdd + qOdd * pOdd] * regR[1];
              }
          }

#pragma unroll
        for (unsigned int j = 0; j < pOdd; j++)
          {
            sharedP[threadIdx.x + a * batchSize +
                    j * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nDofsPerDim] =
              regP[j] + regP[j + pOdd];

            sharedP[threadIdx.x + a * batchSize +
                    (nDofsPerDim - 1 - j) * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nDofsPerDim] =
              regP[j] - regP[j + pOdd];
          }
      }

    __syncthreads();

    // 3rd GEMM of NT
    // X Direction
    for (unsigned int i = threadIdx.y; i < nDofsPerDim * nDofsPerDim;
         i += blockDim.y)
      {
        memset(regQ, 0, nDofsPerDim * sizeof(T));

        for (unsigned int k = 0; k < qOdd; k++)
          {
            regP[0] = sharedP[threadIdx.x + k * batchSize +
                              i * batchSize * nQuadPointsPerDim];

            regP[1] =
              sharedP[threadIdx.x + (nQuadPointsPerDim - 1 - k) * batchSize +
                      i * batchSize * nQuadPointsPerDim];

            regR[0] = regP[0] + regP[1];
            regR[1] = regP[0] - regP[1];

#pragma unroll
            for (unsigned int j = 0; j < pOdd; j++)
              {
                regQ[j] += constNT[j + k * pOdd] * regR[0];
                regQ[j + pOdd] +=
                  constNT[j + k * pOdd + +qOdd * pOdd] * regR[1];
              }
          }

#pragma unroll
        for (unsigned int j = 0; j < pOdd; j++)
          {
            unsigned int dof1 = __ldg(&map[j + i * nDofsPerDim + mapOffset]);
            atomicAdd(&V[threadIdx.x + dof1], regQ[j] + regQ[j + pOdd]);

            unsigned int dof2 =
              __ldg(&map[(nDofsPerDim - 1 - j) + i * nDofsPerDim + mapOffset]);
            atomicAdd(&V[threadIdx.x + dof2], regQ[j] - regQ[j + pOdd]);
          }
      }
  }


  template <typename T,
            unsigned int nDofsPerDim,
            unsigned int nQuadPointsPerDim,
            unsigned int batchSize>
  MatrixFreeDevice<T, nDofsPerDim, nQuadPointsPerDim, batchSize>::
    MatrixFreeDevice(const unsigned int nVectors,
                     const unsigned int nCells,
                     const unsigned int nOwnedDofs,
                     const unsigned int nGhostDofs)
    : d_nVectors(nVectors)
    , d_nBatch(nVectors / batchSize)
    , d_nCells(nCells)
    , d_nOwnedDofs(nOwnedDofs)
    , d_nGhostDofs(nGhostDofs)
  {}


  template <typename T,
            unsigned int nDofsPerDim,
            unsigned int nQuadPointsPerDim,
            unsigned int batchSize>
  void
  MatrixFreeDevice<T, nDofsPerDim, nQuadPointsPerDim, batchSize>::init(
    T           *constMemDataHost,
    unsigned int constMemDataSize,
    dftfe::utils::MemoryStorage<T, dftfe::utils::MemorySpace::HOST>
      &jacobianFactor,
    dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::HOST>
                                           &map,
    std::vector<std::vector<unsigned int>> &constrainingNodeBuckets,
    std::vector<std::vector<unsigned int>> &constrainedNodeBuckets,
    std::vector<std::vector<T>>            &weightMatrixList)
  {
    dftfe::utils::MemoryStorage<unsigned int, dftfe::utils::MemorySpace::HOST>
      constrainingNodeOffset(constrainingNodeBuckets.size() + 1),
      constrainedNodeOffset(constrainedNodeBuckets.size() + 1),
      weightMatrixOffset(weightMatrixList.size() + 1);

    unsigned int k = 0;

    for (unsigned int i = 0; i < constrainingNodeBuckets.size(); i++)
      {
        constrainingNodeOffset[i] = k;
        k += constrainingNodeBuckets[i].size();
      }

    constrainingNodeOffset[constrainingNodeBuckets.size()] = k;
    d_constrainingNodeBuckets.resize(k);

    for (unsigned int i = 0; i < constrainingNodeBuckets.size(); i++)
      dftfe::utils::MemoryTransfer<dftfe::utils::MemorySpace::DEVICE,
                                   dftfe::utils::MemorySpace::HOST>::
        copy(constrainingNodeBuckets[i].size(),
             d_constrainingNodeBuckets.data() + constrainingNodeOffset[i],
             constrainingNodeBuckets[i].data());

    k = 0;

    for (unsigned int i = 0; i < constrainedNodeBuckets.size(); i++)
      {
        constrainedNodeOffset[i] = k;
        k += constrainedNodeBuckets[i].size();
      }

    constrainedNodeOffset[constrainedNodeBuckets.size()] = k;
    d_constrainedNodeBuckets.resize(k);

    for (unsigned int i = 0; i < constrainedNodeBuckets.size(); i++)
      dftfe::utils::MemoryTransfer<dftfe::utils::MemorySpace::DEVICE,
                                   dftfe::utils::MemorySpace::HOST>::
        copy(constrainedNodeBuckets[i].size(),
             d_constrainedNodeBuckets.data() + constrainedNodeOffset[i],
             constrainedNodeBuckets[i].data());

    k = 0;

    for (unsigned int i = 0; i < weightMatrixList.size(); i++)
      {
        weightMatrixOffset[i] = k;
        k += weightMatrixList[i].size();
      }

    weightMatrixOffset[weightMatrixList.size()] = k;
    d_weightMatrixList.resize(k);

    for (unsigned int i = 0; i < weightMatrixList.size(); i++)
      dftfe::utils::MemoryTransfer<
        dftfe::utils::MemorySpace::DEVICE,
        dftfe::utils::MemorySpace::HOST>::copy(weightMatrixList[i].size(),
                                               d_weightMatrixList.data() +
                                                 weightMatrixOffset[i],
                                               weightMatrixList[i].data());

    d_jacobianFactor.resize(jacobianFactor.size());
    d_jacobianFactor.copyFrom(jacobianFactor);
    d_map.resize(map.size());
    d_map.copyFrom(map);

    d_constrainingNodeOffset.resize(constrainingNodeOffset.size());
    d_constrainingNodeOffset.copyFrom(constrainingNodeOffset);

    d_constrainedNodeOffset.resize(constrainedNodeOffset.size());
    d_constrainedNodeOffset.copyFrom(constrainedNodeOffset);

    d_weightMatrixOffset.resize(weightMatrixOffset.size());
    d_weightMatrixOffset.copyFrom(weightMatrixOffset);

    d_inhomogenityList.resize(d_inhomogenityList.size());
    d_inhomogenityList.copyFrom(d_inhomogenityList);

    constexpr unsigned int dim           = 3;
    constexpr size_t       sharedMemSize = 2 * batchSize * nQuadPointsPerDim *
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

    // cudaFuncSetCacheConfig(laplaceKernel<T,
    //                                                 nDofsPerDim,
    //                                                 nQuadPointsPerDim,
    //                                                 batchSize,
    //                                                 dim>,
    //                        cudaFuncCachePreferShared);

    cudaMemcpyToSymbol((T *)constMemData,
                       constMemDataHost,
                       constMemDataSize * sizeof(T),
                       0,
                       cudaMemcpyHostToDevice);
#endif
  }


  template <typename T,
            unsigned int nDofsPerDim,
            unsigned int nQuadPointsPerDim,
            unsigned int batchSize>
  inline void
  MatrixFreeDevice<T, nDofsPerDim, nQuadPointsPerDim, batchSize>::
    constraintsDistribute(T *src)
  {
    // if (d_constrainedNodeBuckets.size() == 0)
    //   return;

    // constexpr int yThreads = 64;
    // const int     batch    = numberWaveFunctions / d_batchsize;

    // dim3 blocks(inhomogenityListDevice.size(), batch, 1);
    // dim3 threads(d_batchsize, yThreads, 1);

    // constraintsDistributeKernel<double, d_batchsize, d_ndofsPerDim>
    //   <<<blocks, threads>>>(XBlock,
    //                         constrainingNodeBuckets.data(),
    //                         constrainingNodeOffsetDevice.data(),
    //                         d_constrainedNodeBuckets.data(),
    //                         constrainedNodeOffsetDevice.data(),
    //                         weightMatrixListDevice.data(),
    //                         weightMatrixOffsetDevice.data(),
    //                         inhomogenityListDevice.data(),
    //                         ghostMapDevice.data(),
    //                         d_nLocalDofs,
    //                         d_nGhostDofs);
  }


  template <typename T,
            unsigned int nDofsPerDim,
            unsigned int nQuadPointsPerDim,
            unsigned int batchSize>
  inline void
  MatrixFreeDevice<T, nDofsPerDim, nQuadPointsPerDim, batchSize>::
    constraintsDistributeTranspose(T *dst, T *src)
  {}


  template <typename T,
            unsigned int nDofsPerDim,
            unsigned int nQuadPointsPerDim,
            unsigned int batchSize>
  inline void
  MatrixFreeDevice<T, nDofsPerDim, nQuadPointsPerDim, batchSize>::
    constraintsSetZero(T *src)
  {}


  template <typename T,
            unsigned int nDofsPerDim,
            unsigned int nQuadPointsPerDim,
            unsigned int batchSize>
  inline void
  MatrixFreeDevice<T, nDofsPerDim, nQuadPointsPerDim, batchSize>::
    computeLaplaceX(T *dst, T *src)
  {
    constexpr int dim = 3;
    constexpr int yThreads =
      (nQuadPointsPerDim != nDofsPerDim ? 128 : (nDofsPerDim < 9 ? 64 : 128));
    constexpr size_t sharedMemSize = 2 * batchSize * nQuadPointsPerDim *
                                     nQuadPointsPerDim * nQuadPointsPerDim *
                                     sizeof(T);

    dim3 blocks(d_nCells, d_nBatch, 1);
    dim3 threads(batchSize, yThreads, 1);

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    laplaceKernel<T, nDofsPerDim, nQuadPointsPerDim, batchSize, dim>
      <<<blocks, threads, sharedMemSize>>>(dst,
                                           src,
                                           d_jacobianFactor.data(),
                                           d_map.data());
#endif
  }

#include "MatrixFreeDevice.inst.cc"

} // namespace dftfe
