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

constexpr std::uint32_t maxDofsPerDim = 17;

__constant__ double
  constMem[(maxDofsPerDim * maxDofsPerDim * 5 + maxDofsPerDim) *
           static_cast<std::uint32_t>(operatorList::Count)];

__device__ inline dftfe::uInt
getMultiVectorIndex(const dftfe::uInt node,
                    const dftfe::uInt batch,
                    const dftfe::uInt nOwnedDofs,
                    const dftfe::uInt nGhostDofs,
                    const dftfe::uInt *__restrict__ ghostMap)
{
  return (node < nOwnedDofs ?
            (node + batch * nOwnedDofs) :
            (ghostMap[node - nOwnedDofs + batch * nGhostDofs]));
}


template <typename T, std::uint32_t nDofsPerDim, std::uint32_t batchSize>
__global__ void
constraintsDistributeKernel(
  T *__restrict__ x,
  const dftfe::uInt *__restrict__ constrainingNodeBuckets,
  const dftfe::uInt *__restrict__ constrainingNodeOffset,
  const dftfe::uInt *__restrict__ constrainedNodeBuckets,
  const dftfe::uInt *__restrict__ constrainedNodeOffset,
  const T *__restrict__ weightMatrixList,
  const dftfe::uInt *__restrict__ weightMatrixOffset,
  const T *__restrict__ inhomogenityList,
  const dftfe::uInt *__restrict__ ghostMap,
  const dftfe::uInt nOwnedDofs,
  const dftfe::uInt nGhostDofs)
{
  __shared__ T sharedConstrainingData[batchSize * nDofsPerDim * nDofsPerDim];

  constexpr int yThreads                = 64;
  dftfe::uInt   constrainingBucketStart = constrainingNodeOffset[blockIdx.x];
  dftfe::uInt   constrainingBucketSize =
    constrainingNodeOffset[blockIdx.x + 1] - constrainingNodeOffset[blockIdx.x];

  for (dftfe::uInt k = threadIdx.y; k < constrainingBucketSize; k += yThreads)
    {
      dftfe::uInt idx;

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

  dftfe::uInt constrainedBucketStart = constrainedNodeOffset[blockIdx.x];
  dftfe::uInt constrainedBucketSize =
    constrainedNodeOffset[blockIdx.x + 1] - constrainedNodeOffset[blockIdx.x];
  dftfe::uInt weightMatrixStart = weightMatrixOffset[blockIdx.x];

  T inhomogenity = inhomogenityList[blockIdx.x];

  for (dftfe::uInt j = threadIdx.y; j < constrainedBucketSize; j += yThreads)
    {
      T tmp = inhomogenity;

      for (dftfe::uInt k = 0; k < constrainingBucketSize; k++)
        tmp +=
          weightMatrixList[k + j * constrainingBucketSize + weightMatrixStart] *
          sharedConstrainingData[threadIdx.x + k * batchSize];

      dftfe::uInt idx;

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
  const dftfe::uInt *__restrict__ constrainingNodeBuckets,
  const dftfe::uInt *__restrict__ constrainingNodeOffset,
  const dftfe::uInt *__restrict__ constrainedNodeBuckets,
  const dftfe::uInt *__restrict__ constrainedNodeOffset,
  const T *__restrict__ weightMatrixList,
  const dftfe::uInt *__restrict__ weightMatrixOffset,
  const dftfe::uInt *__restrict__ ghostMap,
  const dftfe::uInt nOwnedDofs,
  const dftfe::uInt nGhostDofs)
{
  __shared__ T sharedConstrainedData[batchSize * nDofsPerDim * nDofsPerDim * 4];

  constexpr int yThreads                = 64;
  dftfe::uInt   constrainingBucketStart = constrainingNodeOffset[blockIdx.x];
  dftfe::uInt   constrainingBucketSize =
    constrainingNodeOffset[blockIdx.x + 1] - constrainingNodeOffset[blockIdx.x];

  dftfe::uInt constrainedBucketStart = constrainedNodeOffset[blockIdx.x];
  dftfe::uInt constrainedBucketSize =
    constrainedNodeOffset[blockIdx.x + 1] - constrainedNodeOffset[blockIdx.x];

  if (constrainingBucketSize > 0)
    {
      for (dftfe::uInt k = threadIdx.y; k < constrainedBucketSize;
           k += yThreads)
        {
          dftfe::uInt idx;

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

          Ax[threadIdx.x + idx * batchSize] = T(0);
          x[threadIdx.x + idx * batchSize]  = T(0);
        }

      __syncthreads();

      dftfe::uInt weightMatrixStart = weightMatrixOffset[blockIdx.x];

      for (dftfe::uInt j = threadIdx.y; j < constrainingBucketSize;
           j += yThreads)
        {
          T tmp = T(0);

          for (dftfe::uInt k = 0; k < constrainedBucketSize; k++)
            tmp += weightMatrixList[j + k * constrainingBucketSize +
                                    weightMatrixStart] *
                   sharedConstrainedData[threadIdx.x + k * batchSize];

          dftfe::uInt idx;

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
      for (dftfe::uInt k = threadIdx.y; k < constrainedBucketSize;
           k += yThreads)
        {
          dftfe::uInt idx;

          if constexpr (batchSize == 1)
            idx = constrainedNodeBuckets[k + constrainedBucketStart];
          else
            idx = getMultiVectorIndex(
              constrainedNodeBuckets[k + constrainedBucketStart],
              blockIdx.y,
              nOwnedDofs,
              nGhostDofs,
              ghostMap);

          Ax[threadIdx.x + idx * batchSize] = T(0);
          x[threadIdx.x + idx * batchSize]  = T(0);
        }
    }
}


template <typename T,
          std::uint32_t nDofsPerDim,
          std::uint32_t nQuadPointsPerDim,
          std::uint32_t batchSize,
          std::uint32_t dim>
__global__ void
LaplaceKernel(T *__restrict__ dst,
              const T *__restrict__ src,
              const T *__restrict__ J,
              const dftfe::uInt *__restrict__ map)
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
  constexpr std::uint32_t qEven = nQuadPointsPerDim % 2 == 1 ? qOdd + 1 : qOdd;
  constexpr std::uint32_t yThreads =
    dftfe::utils::DEVICE_WARP_SIZE * ((nQuadPointsPerDim * nQuadPointsPerDim +
                                       dftfe::utils::DEVICE_WARP_SIZE - 1) /
                                      dftfe::utils::DEVICE_WARP_SIZE);

  T *__restrict__ sharedU = reinterpret_cast<T *>(sharedMem);
  T *__restrict__ sharedV = &sharedU[batchSize * nQuadPointsPerDim *
                                       nQuadPointsPerDim * nQuadPointsPerDim +
                                     padding];

  T *__restrict__ constN = reinterpret_cast<T *>(
    constMem + 0 * (maxDofsPerDim * maxDofsPerDim * 5 + maxDofsPerDim));
  T *__restrict__ constD      = &constN[qEven * pEven + qOdd * pOdd];
  T *__restrict__ constNT     = &constD[2 * qEven * qOdd];
  T *__restrict__ constDT     = &constNT[pEven * qEven + pOdd * qOdd];
  T *__restrict__ constNprime = &constDT[2 * qEven * qOdd];
  T *__restrict__ constW      = &constNprime[nQuadPointsPerDim * nDofsPerDim];

  T regP[qEven + qOdd], regQ[qEven + qOdd], regR[qEven + qOdd],
    regT[qEven + qOdd];

  const dftfe::uInt mapOffset = (blockIdx.x + blockIdx.y * gridDim.x) *
                                nDofsPerDim * nDofsPerDim * nDofsPerDim;

  //////////////////////////////////////////////////////////////////
  // Interpolation combined with Extraction
  // sharedU -> Nx.Ny.Nz.src(xyz)
  // Nx.Ny.Nz.src(xyz) -> src.NT.NT.NT

  // 1st GEMM of N
  // Z Direction
  for (std::uint32_t i = threadIdx.y; i < nDofsPerDim * nDofsPerDim;
       i += yThreads)
    {
      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < nDofsPerDim; k++)
        {
          dftfe::uInt dof =
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
       i += yThreads)
    {
      std::uint32_t a = i % nDofsPerDim;
      std::uint32_t b = i / nDofsPerDim;

      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < pOdd; k++)
        {
          temp1 =
            sharedU[threadIdx.x + a * batchSize + k * batchSize * nDofsPerDim +
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
  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < pOdd; k++)
        {
          temp1 =
            sharedV[threadIdx.x + k * batchSize + i * batchSize * nDofsPerDim];

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
  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
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
  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
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
                  (nQuadPointsPerDim - 1 - j) * batchSize * nQuadPointsPerDim +
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
  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
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

  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
#pragma unroll
      for (std::uint32_t j = 0; j < qOdd; j++)
        {
          sharedU[threadIdx.x + j * batchSize +
                  i * batchSize * nQuadPointsPerDim] = regT[j + qOdd] + regT[j];

          sharedU[threadIdx.x + (nQuadPointsPerDim - 1 - j) * batchSize +
                  i * batchSize * nQuadPointsPerDim] = regT[j + qOdd] - regT[j];
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        sharedU[threadIdx.x + qOdd * batchSize +
                i * batchSize * nQuadPointsPerDim] = regT[2 * qOdd];
    }

  __syncthreads();

  //////////////////////////////////////////////////////////////////
  // Jacobian Action
  // coeff.J^-T.J^-1.[sharedU sharedV regR]

  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T t[dim];

      dftfe::uInt jOffset = blockIdx.x * dim * dim;

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
                  (i + j * nQuadPointsPerDim * nQuadPointsPerDim) * batchSize] =
            J[0 + jOffset] * t[0] + J[1 + jOffset] * t[1] +
            J[2 + jOffset] * t[2];
          sharedV[threadIdx.x +
                  (i + j * nQuadPointsPerDim * nQuadPointsPerDim) * batchSize] =
            J[3 + jOffset] * t[0] + J[4 + jOffset] * t[1] +
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
  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
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
  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
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
  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
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

  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
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

  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
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
  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
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
       i += yThreads)
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
       i += yThreads)
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
          dftfe::uInt dof1 = __ldg(&map[j + i * nDofsPerDim + mapOffset]);
          atomicAdd(&dst[threadIdx.x + dof1], regT[j] + regT[j + pEven]);

          dftfe::uInt dof2 =
            __ldg(&map[(nDofsPerDim - 1 - j) + i * nDofsPerDim + mapOffset]);
          atomicAdd(&dst[threadIdx.x + dof2], regT[j] - regT[j + pEven]);
        }

      if constexpr (nDofsPerDim % 2 == 1)
        {
          dftfe::uInt dof = __ldg(&map[pOdd + i * nDofsPerDim + mapOffset]);
          atomicAdd(&dst[threadIdx.x + dof], regT[pOdd]);
        }
    }
}


template <typename T,
          std::uint32_t nDofsPerDim,
          std::uint32_t nQuadPointsPerDim,
          std::uint32_t batchSize,
          std::uint32_t dim>
__global__ void
HelmholtzKernel(T *__restrict__ dst,
                const T *__restrict__ src,
                const T *__restrict__ J,
                const dftfe::uInt *__restrict__ map,
                const T coeffHelmholtz)
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
  constexpr std::uint32_t qEven = nQuadPointsPerDim % 2 == 1 ? qOdd + 1 : qOdd;
  constexpr std::uint32_t yThreads =
    dftfe::utils::DEVICE_WARP_SIZE * ((nQuadPointsPerDim * nQuadPointsPerDim +
                                       dftfe::utils::DEVICE_WARP_SIZE - 1) /
                                      dftfe::utils::DEVICE_WARP_SIZE);

  T *__restrict__ sharedU = reinterpret_cast<T *>(sharedMem);
  T *__restrict__ sharedV = &sharedU[batchSize * nQuadPointsPerDim *
                                       nQuadPointsPerDim * nQuadPointsPerDim +
                                     padding];

  T *__restrict__ constN = reinterpret_cast<T *>(
    constMem + 1 * (maxDofsPerDim * maxDofsPerDim * 5 + maxDofsPerDim));
  T *__restrict__ constD      = &constN[qEven * pEven + qOdd * pOdd];
  T *__restrict__ constNT     = &constD[2 * qEven * qOdd];
  T *__restrict__ constDT     = &constNT[pEven * qEven + pOdd * qOdd];
  T *__restrict__ constNprime = &constDT[2 * qEven * qOdd];
  T *__restrict__ constW      = &constNprime[nQuadPointsPerDim * nDofsPerDim];

  T regP[qEven + qOdd], regQ[qEven + qOdd], regR[qEven + qOdd],
    regT[qEven + qOdd];

  const dftfe::uInt mapOffset = (blockIdx.x + blockIdx.y * gridDim.x) *
                                nDofsPerDim * nDofsPerDim * nDofsPerDim;

  //////////////////////////////////////////////////////////////////
  // Interpolation combined with Extraction
  // sharedU -> Nx.Ny.Nz.src(xyz)
  // Nx.Ny.Nz.src(xyz) -> src.NT.NT.NT

  // 1st GEMM of N
  // Z Direction
  for (std::uint32_t i = threadIdx.y; i < nDofsPerDim * nDofsPerDim;
       i += yThreads)
    {
      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < nDofsPerDim; k++)
        {
          dftfe::uInt dof =
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
       i += yThreads)
    {
      std::uint32_t a = i % nDofsPerDim;
      std::uint32_t b = i / nDofsPerDim;

      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < pOdd; k++)
        {
          temp1 =
            sharedU[threadIdx.x + a * batchSize + k * batchSize * nDofsPerDim +
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
  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < pOdd; k++)
        {
          temp1 =
            sharedV[threadIdx.x + k * batchSize + i * batchSize * nDofsPerDim];

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
  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T tempE, tempO;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          regP[k] =
            sharedU[threadIdx.x + i * batchSize +
                    k * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          regP[nQuadPointsPerDim - 1 - k] =
            sharedU[threadIdx.x + i * batchSize +
                    (nQuadPointsPerDim - 1 - k) * batchSize *
                      nQuadPointsPerDim * nQuadPointsPerDim];

          tempE = regP[k] + regP[nQuadPointsPerDim - 1 - k];
          tempO = regP[k] - regP[nQuadPointsPerDim - 1 - k];

#pragma unroll
          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constD[j + k * qOdd] * tempE;

#pragma unroll
          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j + qOdd] += constD[j + k * qEven + qOdd * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          regP[qOdd] =
            sharedU[threadIdx.x + i * batchSize +
                    qOdd * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

#pragma unroll
          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constD[j + qOdd * qOdd] * regP[qOdd];
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
  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
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
                  (nQuadPointsPerDim - 1 - j) * batchSize * nQuadPointsPerDim +
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
  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
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

  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
#pragma unroll
      for (std::uint32_t j = 0; j < qOdd; j++)
        {
          sharedU[threadIdx.x + j * batchSize +
                  i * batchSize * nQuadPointsPerDim] = regT[j + qOdd] + regT[j];

          sharedU[threadIdx.x + (nQuadPointsPerDim - 1 - j) * batchSize +
                  i * batchSize * nQuadPointsPerDim] = regT[j + qOdd] - regT[j];
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        sharedU[threadIdx.x + qOdd * batchSize +
                i * batchSize * nQuadPointsPerDim] = regT[2 * qOdd];
    }

  __syncthreads();

  //////////////////////////////////////////////////////////////////
  // Jacobian Action
  // coeff.J^-T.J^-1.[sharedU sharedV regR]

  T detJ;

  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T t[dim];

      dftfe::uInt jOffset = blockIdx.x * dim * dim;

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
                  (i + j * nQuadPointsPerDim * nQuadPointsPerDim) * batchSize] =
            J[0 + jOffset] * t[0] + J[1 + jOffset] * t[1] +
            J[2 + jOffset] * t[2];
          sharedV[threadIdx.x +
                  (i + j * nQuadPointsPerDim * nQuadPointsPerDim) * batchSize] =
            J[3 + jOffset] * t[0] + J[4 + jOffset] * t[1] +
            J[5 + jOffset] * t[2];
          regR[j] = J[6 + jOffset] * t[0] + J[7 + jOffset] * t[1] +
                    J[8 + jOffset] * t[2];

          detJ = J[0 + jOffset] * (J[4 + jOffset] * J[8 + jOffset] -
                                   J[5 + jOffset] * J[7 + jOffset]) -
                 J[1 + jOffset] * (J[3 + jOffset] * J[8 + jOffset] -
                                   J[5 + jOffset] * J[6 + jOffset]) +
                 J[2 + jOffset] * (J[3 + jOffset] * J[7 + jOffset] -
                                   J[4 + jOffset] * J[6 + jOffset]);
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
  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
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
          regR[j] = regT[j + qOdd] + regT[j] + coeffHelmholtz * detJ * regP[j];
          regR[nQuadPointsPerDim - 1 - j] =
            regT[j + qOdd] - regT[j] +
            coeffHelmholtz * detJ * regP[nQuadPointsPerDim - 1 - j];
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        regR[qOdd] = regT[2 * qOdd] + coeffHelmholtz * detJ * regP[qOdd];
    }

  // 2nd GEMM of DT
  // Y Direction
  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
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
  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
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

  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
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

  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
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
  for (std::uint32_t i = threadIdx.y; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
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
       i += yThreads)
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
       i += yThreads)
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
          dftfe::uInt dof1 = __ldg(&map[j + i * nDofsPerDim + mapOffset]);
          atomicAdd(&dst[threadIdx.x + dof1], regT[j] + regT[j + pEven]);

          dftfe::uInt dof2 =
            __ldg(&map[(nDofsPerDim - 1 - j) + i * nDofsPerDim + mapOffset]);
          atomicAdd(&dst[threadIdx.x + dof2], regT[j] - regT[j + pEven]);
        }

      if constexpr (nDofsPerDim % 2 == 1)
        {
          dftfe::uInt dof = __ldg(&map[pOdd + i * nDofsPerDim + mapOffset]);
          atomicAdd(&dst[threadIdx.x + dof], regT[pOdd]);
        }
    }
}
