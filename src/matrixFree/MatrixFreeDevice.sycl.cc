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

#include <sycl/sycl.hpp>

inline dftfe::uInt
getMultiVectorIndex(const dftfe::uInt  node,
                    const dftfe::uInt  batch,
                    const dftfe::uInt  nOwnedDofs,
                    const dftfe::uInt  nGhostDofs,
                    const dftfe::uInt *ghostMap)
{
  return (node < nOwnedDofs ?
            (node + batch * nOwnedDofs) :
            (ghostMap[node - nOwnedDofs + batch * nGhostDofs]));
}


template <typename T, std::uint32_t nDofsPerDim, std::uint32_t batchSize>
void
constraintsDistributeKernel(sycl::nd_item<3>           item,
                            T                         *x,
                            const dftfe::uInt         *constrainingNodeBuckets,
                            const dftfe::uInt         *constrainingNodeOffset,
                            const dftfe::uInt         *constrainedNodeBuckets,
                            const dftfe::uInt         *constrainedNodeOffset,
                            const T                   *weightMatrixList,
                            const dftfe::uInt         *weightMatrixOffset,
                            const T                   *inhomogenityList,
                            const dftfe::uInt         *ghostMap,
                            const dftfe::uInt          nOwnedDofs,
                            const dftfe::uInt          nGhostDofs,
                            sycl::local_accessor<T, 1> sharedConstrainingData)
{
  constexpr int yThreads = 64;

  const dftfe::uInt blockIdxX  = item.get_group(2);
  const dftfe::uInt blockIdxY  = item.get_group(1);
  const dftfe::uInt threadIdxX = item.get_local_id(2);
  const dftfe::uInt threadIdxY = item.get_local_id(1);

  dftfe::uInt constrainingBucketStart = constrainingNodeOffset[blockIdxX];
  dftfe::uInt constrainingBucketSize =
    constrainingNodeOffset[blockIdxX + 1] - constrainingNodeOffset[blockIdxX];

  for (dftfe::uInt k = threadIdxY; k < constrainingBucketSize; k += yThreads)
    {
      dftfe::uInt idx;

      if constexpr (batchSize == 1)
        idx = constrainingNodeBuckets[k + constrainingBucketStart];
      else
        idx = getMultiVectorIndex(
          constrainingNodeBuckets[k + constrainingBucketStart],
          blockIdxY,
          nOwnedDofs,
          nGhostDofs,
          ghostMap);

      sharedConstrainingData[threadIdxX + k * batchSize] =
        x[threadIdxX + idx * batchSize];
    }

  item.barrier(sycl::access::fence_space::local_space);

  dftfe::uInt constrainedBucketStart = constrainedNodeOffset[blockIdxX];
  dftfe::uInt constrainedBucketSize =
    constrainedNodeOffset[blockIdxX + 1] - constrainedNodeOffset[blockIdxX];
  dftfe::uInt weightMatrixStart = weightMatrixOffset[blockIdxX];

  T inhomogenity = inhomogenityList[blockIdxX];

  for (dftfe::uInt j = threadIdxY; j < constrainedBucketSize; j += yThreads)
    {
      T tmp = inhomogenity;

      for (dftfe::uInt k = 0; k < constrainingBucketSize; k++)
        tmp +=
          weightMatrixList[k + j * constrainingBucketSize + weightMatrixStart] *
          sharedConstrainingData[threadIdxX + k * batchSize];

      dftfe::uInt idx;

      if constexpr (batchSize == 1)
        idx = constrainedNodeBuckets[j + constrainedBucketStart];
      else
        idx = getMultiVectorIndex(
          constrainedNodeBuckets[j + constrainedBucketStart],
          blockIdxY,
          nOwnedDofs,
          nGhostDofs,
          ghostMap);

      x[threadIdxX + idx * batchSize] = tmp;
    }
}


template <typename T, std::uint32_t nDofsPerDim, std::uint32_t batchSize>
void
constraintsDistributeTransposeKernel(
  sycl::nd_item<3>           item,
  T                         *Ax,
  T                         *x,
  const dftfe::uInt         *constrainingNodeBuckets,
  const dftfe::uInt         *constrainingNodeOffset,
  const dftfe::uInt         *constrainedNodeBuckets,
  const dftfe::uInt         *constrainedNodeOffset,
  const T                   *weightMatrixList,
  const dftfe::uInt         *weightMatrixOffset,
  const dftfe::uInt         *ghostMap,
  const dftfe::uInt          nOwnedDofs,
  const dftfe::uInt          nGhostDofs,
  sycl::local_accessor<T, 1> sharedConstrainedData)
{
  constexpr int yThreads = 64;

  const dftfe::uInt blockIdxX  = item.get_group(2);
  const dftfe::uInt blockIdxY  = item.get_group(1);
  const dftfe::uInt threadIdxX = item.get_local_id(2);
  const dftfe::uInt threadIdxY = item.get_local_id(1);

  dftfe::uInt constrainingBucketStart = constrainingNodeOffset[blockIdxX];
  dftfe::uInt constrainingBucketSize =
    constrainingNodeOffset[blockIdxX + 1] - constrainingNodeOffset[blockIdxX];

  dftfe::uInt constrainedBucketStart = constrainedNodeOffset[blockIdxX];
  dftfe::uInt constrainedBucketSize =
    constrainedNodeOffset[blockIdxX + 1] - constrainedNodeOffset[blockIdxX];

  if (constrainingBucketSize > 0)
    {
      for (dftfe::uInt k = threadIdxY; k < constrainedBucketSize; k += yThreads)
        {
          dftfe::uInt idx;

          if constexpr (batchSize == 1)
            idx = constrainedNodeBuckets[k + constrainedBucketStart];
          else
            idx = getMultiVectorIndex(
              constrainedNodeBuckets[k + constrainedBucketStart],
              blockIdxY,
              nOwnedDofs,
              nGhostDofs,
              ghostMap);

          sharedConstrainedData[threadIdxX + k * batchSize] =
            Ax[threadIdxX + idx * batchSize];

          Ax[threadIdxX + idx * batchSize] = T(0);
          x[threadIdxX + idx * batchSize]  = T(0);
        }

      item.barrier(sycl::access::fence_space::local_space);

      dftfe::uInt weightMatrixStart = weightMatrixOffset[blockIdxX];

      for (dftfe::uInt j = threadIdxY; j < constrainingBucketSize;
           j += yThreads)
        {
          T tmp = T(0);

          for (dftfe::uInt k = 0; k < constrainedBucketSize; k++)
            tmp += weightMatrixList[j + k * constrainingBucketSize +
                                    weightMatrixStart] *
                   sharedConstrainedData[threadIdxX + k * batchSize];

          dftfe::uInt idx;

          if constexpr (batchSize == 1)
            idx = constrainingNodeBuckets[j + constrainingBucketStart];
          else
            idx = getMultiVectorIndex(
              constrainingNodeBuckets[j + constrainingBucketStart],
              blockIdxY,
              nOwnedDofs,
              nGhostDofs,
              ghostMap);

          sycl::atomic_ref<T,
                           sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
            atomicRef(Ax[threadIdxX + idx * batchSize]);
          atomicRef += tmp;
        }
    }
  else
    {
      for (dftfe::uInt k = threadIdxY; k < constrainedBucketSize; k += yThreads)
        {
          dftfe::uInt idx;

          if constexpr (batchSize == 1)
            idx = constrainedNodeBuckets[k + constrainedBucketStart];
          else
            idx = getMultiVectorIndex(
              constrainedNodeBuckets[k + constrainedBucketStart],
              blockIdxY,
              nOwnedDofs,
              nGhostDofs,
              ghostMap);

          Ax[threadIdxX + idx * batchSize] = T(0);
          x[threadIdxX + idx * batchSize]  = T(0);
        }
    }
}


template <typename T,
          std::uint32_t nDofsPerDim,
          std::uint32_t nQuadPointsPerDim,
          std::uint32_t batchSize,
          std::uint32_t dim>
void
LaplaceKernel(sycl::nd_item<3>           item,
              T                         *dst,
              const T                   *src,
              const T                   *J,
              const dftfe::uInt         *map,
              const T                   *shapeBufferDevice,
              sycl::local_accessor<T, 1> sharedMem)
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

  constexpr std::uint32_t padding = 0;
  constexpr std::uint32_t pOdd    = nDofsPerDim / 2;
  constexpr std::uint32_t pEven   = nDofsPerDim % 2 == 1 ? pOdd + 1 : pOdd;
  constexpr std::uint32_t qOdd    = nQuadPointsPerDim / 2;
  constexpr std::uint32_t qEven = nQuadPointsPerDim % 2 == 1 ? qOdd + 1 : qOdd;
  constexpr std::uint32_t yThreads =
    dftfe::utils::DEVICE_WARP_SIZE * ((nQuadPointsPerDim * nQuadPointsPerDim +
                                       dftfe::utils::DEVICE_WARP_SIZE - 1) /
                                      dftfe::utils::DEVICE_WARP_SIZE);

  const dftfe::uInt blockIdxX  = item.get_group(2);
  const dftfe::uInt blockIdxY  = item.get_group(1);
  const dftfe::uInt gridDimX   = item.get_group_range(2);
  const dftfe::uInt threadIdxX = item.get_local_id(2);
  const dftfe::uInt threadIdxY = item.get_local_id(1);

  T *sharedU = sharedMem.get_pointer();
  T *sharedV = &sharedU[batchSize * nQuadPointsPerDim * nQuadPointsPerDim *
                          nQuadPointsPerDim +
                        padding];

  // sharedShape is stored at the tail of sharedMem (local memory)
  constexpr std::uint32_t shapeBufferElements =
    2 * (qEven * pEven + qOdd * pOdd) + 4 * qEven * qOdd +
    nQuadPointsPerDim * nDofsPerDim + nQuadPointsPerDim;
  T *sharedShape = &sharedV[batchSize * nQuadPointsPerDim * nQuadPointsPerDim *
                              nQuadPointsPerDim +
                            padding];

  // Cooperatively load shapeBuffer from global to local memory
  for (std::uint32_t idx = threadIdxY * batchSize + threadIdxX;
       idx < shapeBufferElements;
       idx += yThreads * batchSize)
    sharedShape[idx] = shapeBufferDevice[idx];

  item.barrier(sycl::access::fence_space::local_space);

  const T *constN      = sharedShape;
  const T *constD      = &constN[qEven * pEven + qOdd * pOdd];
  const T *constNT     = &constD[2 * qEven * qOdd];
  const T *constDT     = &constNT[pEven * qEven + pOdd * qOdd];
  const T *constNprime = &constDT[2 * qEven * qOdd];
  const T *constW      = &constNprime[nQuadPointsPerDim * nDofsPerDim];

  T regP[qEven + qOdd], regQ[qEven + qOdd], regR[qEven + qOdd],
    regT[qEven + qOdd];

  const dftfe::uInt mapOffset = (blockIdxX + blockIdxY * gridDimX) *
                                nDofsPerDim * nDofsPerDim * nDofsPerDim;

  //////////////////////////////////////////////////////////////////
  // Interpolation combined with Extraction
  // sharedU -> Nx.Ny.Nz.src(xyz)
  // Nx.Ny.Nz.src(xyz) -> src.NT.NT.NT

  // 1st GEMM of N
  // Z Direction
  for (std::uint32_t i = threadIdxY; i < nDofsPerDim * nDofsPerDim;
       i += yThreads)
    {
      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < nDofsPerDim; k++)
        {
          dftfe::uInt dof = map[i + k * nDofsPerDim * nDofsPerDim + mapOffset];
          regP[k]         = src[threadIdxX + dof];

          for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
            regT[j] += constNprime[j + k * nQuadPointsPerDim] * regP[k];
        }

      for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
        sharedU[threadIdxX + i * batchSize +
                j * batchSize * nDofsPerDim * nDofsPerDim] = regT[j];
    }

  item.barrier(sycl::access::fence_space::local_space);

  // 2nd GEMM of N
  // Y Direction
  for (std::uint32_t i = threadIdxY; i < nDofsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      std::uint32_t a = i % nDofsPerDim;
      std::uint32_t b = i / nDofsPerDim;

      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < pOdd; k++)
        {
          temp1 =
            sharedU[threadIdxX + a * batchSize + k * batchSize * nDofsPerDim +
                    b * batchSize * nDofsPerDim * nDofsPerDim];

          temp2 = sharedU[threadIdxX + a * batchSize +
                          (nDofsPerDim - 1 - k) * batchSize * nDofsPerDim +
                          b * batchSize * nDofsPerDim * nDofsPerDim];

          tempE = temp1 + temp2;
          tempO = temp1 - temp2;

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j] += constN[j + k * qEven] * tempE;

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j + qEven] += constN[j + k * qOdd + qEven * pEven] * tempO;
        }

      if constexpr (nDofsPerDim % 2 == 1)
        {
          tempE = sharedU[threadIdxX + a * batchSize +
                          pOdd * batchSize * nDofsPerDim +
                          b * batchSize * nDofsPerDim * nDofsPerDim];

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j] += constN[j + pOdd * qEven] * tempE;
        }

      for (std::uint32_t j = 0; j < qOdd; j++)
        {
          sharedV[threadIdxX + a * batchSize + j * batchSize * nDofsPerDim +
                  b * batchSize * nDofsPerDim * nQuadPointsPerDim] =
            regT[j] + regT[j + qEven];

          sharedV[threadIdxX + a * batchSize +
                  (nQuadPointsPerDim - 1 - j) * batchSize * nDofsPerDim +
                  b * batchSize * nDofsPerDim * nQuadPointsPerDim] =
            regT[j] - regT[j + qEven];
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        sharedV[threadIdxX + a * batchSize + qOdd * batchSize * nDofsPerDim +
                b * batchSize * nDofsPerDim * nQuadPointsPerDim] = regT[qOdd];
    }

  item.barrier(sycl::access::fence_space::local_space);

  // 3rd GEMM of N
  // X Direction
  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < pOdd; k++)
        {
          temp1 =
            sharedV[threadIdxX + k * batchSize + i * batchSize * nDofsPerDim];

          temp2 = sharedV[threadIdxX + (nDofsPerDim - 1 - k) * batchSize +
                          i * batchSize * nDofsPerDim];

          tempE = temp1 + temp2;
          tempO = temp1 - temp2;

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j] += constN[j + k * qEven] * tempE;

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j + qEven] += constN[j + k * qOdd + qEven * pEven] * tempO;
        }

      if constexpr (nDofsPerDim % 2 == 1)
        {
          tempE = sharedV[threadIdxX + pOdd * batchSize +
                          i * batchSize * nDofsPerDim];

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j] += constN[j + pOdd * qEven] * tempE;
        }

      for (std::uint32_t j = 0; j < qOdd; j++)
        {
          sharedU[threadIdxX + j * batchSize +
                  i * batchSize * nQuadPointsPerDim] =
            regT[j] + regT[j + qEven];

          sharedU[threadIdxX + (nQuadPointsPerDim - 1 - j) * batchSize +
                  i * batchSize * nQuadPointsPerDim] =
            regT[j] - regT[j + qEven];
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        sharedU[threadIdxX + qOdd * batchSize +
                i * batchSize * nQuadPointsPerDim] = regT[qOdd];
    }

  item.barrier(sycl::access::fence_space::local_space);

  //////////////////////////////////////////////////////////////////
  // Grad operation in each direction
  // sharedU -> Nx.Ny.Nz.Uxyz
  // regR    -> Dz.Nx.Ny.Nz.Uxyz
  // sharedV -> Dy.Nx.Ny.Nz.Uxyz
  // sharedU -> Dx.Nx.Ny.Nz.Uxyz

  // 1st GEMM of D
  // Z Direction
  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          temp1 =
            sharedU[threadIdxX + i * batchSize +
                    k * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          temp2 = sharedU[threadIdxX + i * batchSize +
                          (nQuadPointsPerDim - 1 - k) * batchSize *
                            nQuadPointsPerDim * nQuadPointsPerDim];

          tempE = temp1 + temp2;
          tempO = temp1 - temp2;

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constD[j + k * qOdd] * tempE;

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j + qOdd] += constD[j + k * qEven + qOdd * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          tempE =
            sharedU[threadIdxX + i * batchSize +
                    qOdd * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constD[j + qOdd * qOdd] * tempE;
        }

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
  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      std::uint32_t a = i % nQuadPointsPerDim;
      std::uint32_t b = i / nQuadPointsPerDim;

      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          temp1 =
            sharedU[threadIdxX + a * batchSize +
                    k * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          temp2 =
            sharedU[threadIdxX + a * batchSize +
                    (nQuadPointsPerDim - 1 - k) * batchSize *
                      nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          tempE = temp1 + temp2;
          tempO = temp1 - temp2;

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constD[j + k * qOdd] * tempE;

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j + qOdd] += constD[j + k * qEven + qOdd * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          tempE =
            sharedU[threadIdxX + a * batchSize +
                    qOdd * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constD[j + qOdd * qOdd] * tempE;
        }

      for (std::uint32_t j = 0; j < qOdd; j++)
        {
          sharedV[threadIdxX + a * batchSize +
                  j * batchSize * nQuadPointsPerDim +
                  b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
            regT[j + qOdd] + regT[j];

          sharedV[threadIdxX + a * batchSize +
                  (nQuadPointsPerDim - 1 - j) * batchSize * nQuadPointsPerDim +
                  b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
            regT[j + qOdd] - regT[j];
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        sharedV[threadIdxX + a * batchSize +
                qOdd * batchSize * nQuadPointsPerDim +
                b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
          regT[2 * qOdd];
    }

  // 3rd GEMM of D
  // X Direction
  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          temp1 = sharedU[threadIdxX + k * batchSize +
                          i * batchSize * nQuadPointsPerDim];

          temp2 = sharedU[threadIdxX + (nQuadPointsPerDim - 1 - k) * batchSize +
                          i * batchSize * nQuadPointsPerDim];

          tempE = temp1 + temp2;
          tempO = temp1 - temp2;

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constD[j + k * qOdd] * tempE;

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j + qOdd] += constD[j + k * qEven + qOdd * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          tempE = sharedU[threadIdxX + qOdd * batchSize +
                          i * batchSize * nQuadPointsPerDim];

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constD[j + qOdd * qOdd] * tempE;
        }
    }

  item.barrier(sycl::access::fence_space::local_space);

  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      for (std::uint32_t j = 0; j < qOdd; j++)
        {
          sharedU[threadIdxX + j * batchSize +
                  i * batchSize * nQuadPointsPerDim] = regT[j + qOdd] + regT[j];

          sharedU[threadIdxX + (nQuadPointsPerDim - 1 - j) * batchSize +
                  i * batchSize * nQuadPointsPerDim] = regT[j + qOdd] - regT[j];
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        sharedU[threadIdxX + qOdd * batchSize +
                i * batchSize * nQuadPointsPerDim] = regT[2 * qOdd];
    }

  item.barrier(sycl::access::fence_space::local_space);

  //////////////////////////////////////////////////////////////////
  // Jacobian Action
  // coeff.J^-T.J^-1.[sharedU sharedV regR]

  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T t[dim];

      dftfe::uInt jOffset = blockIdxX * dim * dim;

      for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
        {
          t[0] = sharedU[threadIdxX +
                         (i + j * nQuadPointsPerDim * nQuadPointsPerDim) *
                           batchSize];
          t[1] = sharedV[threadIdxX +
                         (i + j * nQuadPointsPerDim * nQuadPointsPerDim) *
                           batchSize];
          t[2] = regR[j];

          sharedU[threadIdxX + (i + j * nQuadPointsPerDim * nQuadPointsPerDim) *
                                 batchSize] = J[0 + jOffset] * t[0] +
                                              J[1 + jOffset] * t[1] +
                                              J[2 + jOffset] * t[2];
          sharedV[threadIdxX + (i + j * nQuadPointsPerDim * nQuadPointsPerDim) *
                                 batchSize] = J[3 + jOffset] * t[0] +
                                              J[4 + jOffset] * t[1] +
                                              J[5 + jOffset] * t[2];
          regR[j] = J[6 + jOffset] * t[0] + J[7 + jOffset] * t[1] +
                    J[8 + jOffset] * t[2];
        }
    }

  item.barrier(sycl::access::fence_space::local_space);

  //////////////////////////////////////////////////////////////////////////////////////////
  // Grad operation in each direction
  // regR -> Dz.Nx.Ny.Nz.Uxyz
  // regQ -> Dy.Nx.Ny.Nz.Uxyz
  // regP -> Dx.Nx.Ny.Nz.Uxyz
  // regR -> [DT.coeff.JF.D].Nx.Ny.Nz.Uxyz

  // 1st GEMM of DT
  // Z Direction
  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T tempE, tempO;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          tempE = regR[k] + regR[nQuadPointsPerDim - 1 - k];
          tempO = regR[k] - regR[nQuadPointsPerDim - 1 - k];

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constDT[j + k * qOdd + qEven * qOdd] * tempE;

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j + qOdd] += constDT[j + k * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constDT[j + qOdd * qOdd + qEven * qOdd] * regR[qOdd];
        }

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
  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      std::uint32_t a = i % nQuadPointsPerDim;
      std::uint32_t b = i / nQuadPointsPerDim;

      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          temp1 =
            sharedV[threadIdxX + a * batchSize +
                    k * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          temp2 =
            sharedV[threadIdxX + a * batchSize +
                    (nQuadPointsPerDim - 1 - k) * batchSize *
                      nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          tempE = temp1 + temp2;
          tempO = temp1 - temp2;

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constDT[j + k * qOdd + qEven * qOdd] * tempE;

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j + qOdd] += constDT[j + k * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          tempE =
            sharedV[threadIdxX + a * batchSize +
                    qOdd * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constDT[j + qOdd * qOdd + qEven * qOdd] * tempE;
        }

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
  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          temp1 = sharedU[threadIdxX + k * batchSize +
                          i * batchSize * nQuadPointsPerDim];

          temp2 = sharedU[threadIdxX + (nQuadPointsPerDim - 1 - k) * batchSize +
                          i * batchSize * nQuadPointsPerDim];

          tempE = temp1 + temp2;
          tempO = temp1 - temp2;

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constDT[j + k * qOdd + qEven * qOdd] * tempE;

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j + qOdd] += constDT[j + k * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          tempE = sharedU[threadIdxX + qOdd * batchSize +
                          i * batchSize * nQuadPointsPerDim];

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constDT[j + qOdd * qOdd + qEven * qOdd] * tempE;
        }

      for (std::uint32_t j = 0; j < qOdd; j++)
        {
          regP[j]                         = regT[j + qOdd] + regT[j];
          regP[nQuadPointsPerDim - 1 - j] = regT[j + qOdd] - regT[j];
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        regP[qOdd] = regT[2 * qOdd];
    }

  item.barrier(sycl::access::fence_space::local_space);

  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      std::uint32_t a = i % nQuadPointsPerDim;
      std::uint32_t b = i / nQuadPointsPerDim;

      for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
        sharedV[threadIdxX + a * batchSize + j * batchSize * nQuadPointsPerDim +
                b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
          regQ[j];

      for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
        sharedU[threadIdxX + j * batchSize +
                i * batchSize * nQuadPointsPerDim] = regP[j];
    }

  item.barrier(sycl::access::fence_space::local_space);

  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
        {
          regR[j] =
            regR[j] +
            sharedU[threadIdxX + i * batchSize +
                    j * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] +
            sharedV[threadIdxX + i * batchSize +
                    j * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];
        }
    }

  item.barrier(sycl::access::fence_space::local_space);

  //////////////////////////////////////////////////////////////////////////////////////////////////////
  // Integration combined with Assembly
  // V -> NTx.NTy.NTz.[DT.coeff.JF.D].Nx.Ny.Nz.Uxyz

  // 1st GEMM of NT
  // Z Direction
  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T tempE, tempO;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          tempE = regR[k] + regR[nQuadPointsPerDim - 1 - k];
          tempO = regR[k] - regR[nQuadPointsPerDim - 1 - k];

          for (std::uint32_t j = 0; j < pEven; j++)
            regT[j] += constNT[j + k * pEven] * tempE;

          for (std::uint32_t j = 0; j < pOdd; j++)
            regT[j + pEven] += constNT[j + k * pOdd + pEven * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          for (std::uint32_t j = 0; j < pEven; j++)
            regT[j] += constNT[j + qOdd * pEven] * regR[qOdd];
        }

      for (std::uint32_t j = 0; j < pOdd; j++)
        {
          sharedV[threadIdxX + i * batchSize +
                  j * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
            regT[j] + regT[j + pEven];

          sharedV[threadIdxX + i * batchSize +
                  (nDofsPerDim - 1 - j) * batchSize * nQuadPointsPerDim *
                    nQuadPointsPerDim] = regT[j] - regT[j + pEven];
        }

      if constexpr (nDofsPerDim % 2 == 1)
        sharedV[threadIdxX + i * batchSize +
                pOdd * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
          regT[pOdd];
    }

  item.barrier(sycl::access::fence_space::local_space);

  // 2nd GEMM of NT
  // Y Direction
  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nDofsPerDim;
       i += yThreads)
    {
      std::uint32_t a = i % nQuadPointsPerDim;
      std::uint32_t b = i / nQuadPointsPerDim;

      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          temp1 =
            sharedV[threadIdxX + a * batchSize +
                    k * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          temp2 =
            sharedV[threadIdxX + a * batchSize +
                    (nQuadPointsPerDim - 1 - k) * batchSize *
                      nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          tempE = temp1 + temp2;
          tempO = temp1 - temp2;

          for (std::uint32_t j = 0; j < pEven; j++)
            regT[j] += constNT[j + k * pEven] * tempE;

          for (std::uint32_t j = 0; j < pOdd; j++)
            regT[j + pEven] += constNT[j + k * pOdd + pEven * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          tempE =
            sharedV[threadIdxX + a * batchSize +
                    qOdd * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          for (std::uint32_t j = 0; j < pEven; j++)
            regT[j] += constNT[j + qOdd * pEven] * tempE;
        }

      for (std::uint32_t j = 0; j < pOdd; j++)
        {
          sharedU[threadIdxX + a * batchSize +
                  j * batchSize * nQuadPointsPerDim +
                  b * batchSize * nQuadPointsPerDim * nDofsPerDim] =
            regT[j] + regT[j + pEven];

          sharedU[threadIdxX + a * batchSize +
                  (nDofsPerDim - 1 - j) * batchSize * nQuadPointsPerDim +
                  b * batchSize * nQuadPointsPerDim * nDofsPerDim] =
            regT[j] - regT[j + pEven];
        }

      if constexpr (nDofsPerDim % 2 == 1)
        sharedU[threadIdxX + a * batchSize +
                pOdd * batchSize * nQuadPointsPerDim +
                b * batchSize * nQuadPointsPerDim * nDofsPerDim] = regT[pOdd];
    }

  item.barrier(sycl::access::fence_space::local_space);

  // 3rd GEMM of NT
  // X Direction
  for (std::uint32_t i = threadIdxY; i < nDofsPerDim * nDofsPerDim;
       i += yThreads)
    {
      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          temp1 = sharedU[threadIdxX + k * batchSize +
                          i * batchSize * nQuadPointsPerDim];

          temp2 = sharedU[threadIdxX + (nQuadPointsPerDim - 1 - k) * batchSize +
                          i * batchSize * nQuadPointsPerDim];

          tempE = temp1 + temp2;
          tempO = temp1 - temp2;

          for (std::uint32_t j = 0; j < pEven; j++)
            regT[j] += constNT[j + k * pEven] * tempE;

          for (std::uint32_t j = 0; j < pOdd; j++)
            regT[j + pEven] += constNT[j + k * pOdd + pEven * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          tempE = sharedU[threadIdxX + qOdd * batchSize +
                          i * batchSize * nQuadPointsPerDim];

          for (std::uint32_t j = 0; j < pEven; j++)
            regT[j] += constNT[j + qOdd * pEven] * tempE;
        }

      for (std::uint32_t j = 0; j < pOdd; j++)
        {
          dftfe::uInt dof1 = map[j + i * nDofsPerDim + mapOffset];

          sycl::atomic_ref<T,
                           sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
            atomicRef1(dst[threadIdxX + dof1]);
          atomicRef1 += regT[j] + regT[j + pEven];

          dftfe::uInt dof2 =
            map[(nDofsPerDim - 1 - j) + i * nDofsPerDim + mapOffset];

          sycl::atomic_ref<T,
                           sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
            atomicRef2(dst[threadIdxX + dof2]);
          atomicRef2 += regT[j] - regT[j + pEven];
        }

      if constexpr (nDofsPerDim % 2 == 1)
        {
          dftfe::uInt dof = map[pOdd + i * nDofsPerDim + mapOffset];

          sycl::atomic_ref<T,
                           sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
            atomicRef(dst[threadIdxX + dof]);
          atomicRef += regT[pOdd];
        }
    }
}


template <typename T,
          std::uint32_t nDofsPerDim,
          std::uint32_t nQuadPointsPerDim,
          std::uint32_t batchSize,
          std::uint32_t dim>
void
HelmholtzKernel(sycl::nd_item<3>           item,
                T                         *dst,
                const T                   *src,
                const T                   *J,
                const dftfe::uInt         *map,
                const T                    coeffHelmholtz,
                const T                   *shapeBufferDevice,
                sycl::local_accessor<T, 1> sharedMem)
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

  constexpr std::uint32_t padding = 0;
  constexpr std::uint32_t pOdd    = nDofsPerDim / 2;
  constexpr std::uint32_t pEven   = nDofsPerDim % 2 == 1 ? pOdd + 1 : pOdd;
  constexpr std::uint32_t qOdd    = nQuadPointsPerDim / 2;
  constexpr std::uint32_t qEven = nQuadPointsPerDim % 2 == 1 ? qOdd + 1 : qOdd;
  constexpr std::uint32_t yThreads =
    dftfe::utils::DEVICE_WARP_SIZE * ((nQuadPointsPerDim * nQuadPointsPerDim +
                                       dftfe::utils::DEVICE_WARP_SIZE - 1) /
                                      dftfe::utils::DEVICE_WARP_SIZE);

  const dftfe::uInt blockIdxX  = item.get_group(2);
  const dftfe::uInt blockIdxY  = item.get_group(1);
  const dftfe::uInt gridDimX   = item.get_group_range(2);
  const dftfe::uInt threadIdxX = item.get_local_id(2);
  const dftfe::uInt threadIdxY = item.get_local_id(1);

  T *sharedU = sharedMem.get_pointer();
  T *sharedV = &sharedU[batchSize * nQuadPointsPerDim * nQuadPointsPerDim *
                          nQuadPointsPerDim +
                        padding];

  // sharedShape is stored at the tail of sharedMem (local memory)
  constexpr std::uint32_t shapeBufferElements =
    2 * (qEven * pEven + qOdd * pOdd) + 4 * qEven * qOdd +
    nQuadPointsPerDim * nDofsPerDim + nQuadPointsPerDim;
  T *sharedShape = &sharedV[batchSize * nQuadPointsPerDim * nQuadPointsPerDim *
                              nQuadPointsPerDim +
                            padding];

  // Cooperatively load shapeBuffer from global to local memory
  for (std::uint32_t idx = threadIdxY * batchSize + threadIdxX;
       idx < shapeBufferElements;
       idx += yThreads * batchSize)
    sharedShape[idx] = shapeBufferDevice[idx];

  item.barrier(sycl::access::fence_space::local_space);

  const T *constN      = sharedShape;
  const T *constD      = &constN[qEven * pEven + qOdd * pOdd];
  const T *constNT     = &constD[2 * qEven * qOdd];
  const T *constDT     = &constNT[pEven * qEven + pOdd * qOdd];
  const T *constNprime = &constDT[2 * qEven * qOdd];
  const T *constW      = &constNprime[nQuadPointsPerDim * nDofsPerDim];

  T regP[qEven + qOdd], regQ[qEven + qOdd], regR[qEven + qOdd],
    regT[qEven + qOdd];

  const dftfe::uInt mapOffset = (blockIdxX + blockIdxY * gridDimX) *
                                nDofsPerDim * nDofsPerDim * nDofsPerDim;

  //////////////////////////////////////////////////////////////////
  // Interpolation combined with Extraction
  // sharedU -> Nx.Ny.Nz.src(xyz)
  // Nx.Ny.Nz.src(xyz) -> src.NT.NT.NT

  // 1st GEMM of N
  // Z Direction
  for (std::uint32_t i = threadIdxY; i < nDofsPerDim * nDofsPerDim;
       i += yThreads)
    {
      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < nDofsPerDim; k++)
        {
          dftfe::uInt dof = map[i + k * nDofsPerDim * nDofsPerDim + mapOffset];
          regP[k]         = src[threadIdxX + dof];

          for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
            regT[j] += constNprime[j + k * nQuadPointsPerDim] * regP[k];
        }

      for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
        sharedU[threadIdxX + i * batchSize +
                j * batchSize * nDofsPerDim * nDofsPerDim] = regT[j];
    }

  item.barrier(sycl::access::fence_space::local_space);

  // 2nd GEMM of N
  // Y Direction
  for (std::uint32_t i = threadIdxY; i < nDofsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      std::uint32_t a = i % nDofsPerDim;
      std::uint32_t b = i / nDofsPerDim;

      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < pOdd; k++)
        {
          temp1 =
            sharedU[threadIdxX + a * batchSize + k * batchSize * nDofsPerDim +
                    b * batchSize * nDofsPerDim * nDofsPerDim];

          temp2 = sharedU[threadIdxX + a * batchSize +
                          (nDofsPerDim - 1 - k) * batchSize * nDofsPerDim +
                          b * batchSize * nDofsPerDim * nDofsPerDim];

          tempE = temp1 + temp2;
          tempO = temp1 - temp2;

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j] += constN[j + k * qEven] * tempE;

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j + qEven] += constN[j + k * qOdd + qEven * pEven] * tempO;
        }

      if constexpr (nDofsPerDim % 2 == 1)
        {
          tempE = sharedU[threadIdxX + a * batchSize +
                          pOdd * batchSize * nDofsPerDim +
                          b * batchSize * nDofsPerDim * nDofsPerDim];

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j] += constN[j + pOdd * qEven] * tempE;
        }

      for (std::uint32_t j = 0; j < qOdd; j++)
        {
          sharedV[threadIdxX + a * batchSize + j * batchSize * nDofsPerDim +
                  b * batchSize * nDofsPerDim * nQuadPointsPerDim] =
            regT[j] + regT[j + qEven];

          sharedV[threadIdxX + a * batchSize +
                  (nQuadPointsPerDim - 1 - j) * batchSize * nDofsPerDim +
                  b * batchSize * nDofsPerDim * nQuadPointsPerDim] =
            regT[j] - regT[j + qEven];
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        sharedV[threadIdxX + a * batchSize + qOdd * batchSize * nDofsPerDim +
                b * batchSize * nDofsPerDim * nQuadPointsPerDim] = regT[qOdd];
    }

  item.barrier(sycl::access::fence_space::local_space);

  // 3rd GEMM of N
  // X Direction
  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < pOdd; k++)
        {
          temp1 =
            sharedV[threadIdxX + k * batchSize + i * batchSize * nDofsPerDim];

          temp2 = sharedV[threadIdxX + (nDofsPerDim - 1 - k) * batchSize +
                          i * batchSize * nDofsPerDim];

          tempE = temp1 + temp2;
          tempO = temp1 - temp2;

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j] += constN[j + k * qEven] * tempE;

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j + qEven] += constN[j + k * qOdd + qEven * pEven] * tempO;
        }

      if constexpr (nDofsPerDim % 2 == 1)
        {
          tempE = sharedV[threadIdxX + pOdd * batchSize +
                          i * batchSize * nDofsPerDim];

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j] += constN[j + pOdd * qEven] * tempE;
        }

      for (std::uint32_t j = 0; j < qOdd; j++)
        {
          sharedU[threadIdxX + j * batchSize +
                  i * batchSize * nQuadPointsPerDim] =
            regT[j] + regT[j + qEven];

          sharedU[threadIdxX + (nQuadPointsPerDim - 1 - j) * batchSize +
                  i * batchSize * nQuadPointsPerDim] =
            regT[j] - regT[j + qEven];
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        sharedU[threadIdxX + qOdd * batchSize +
                i * batchSize * nQuadPointsPerDim] = regT[qOdd];
    }

  item.barrier(sycl::access::fence_space::local_space);

  //////////////////////////////////////////////////////////////////
  // Grad operation in each direction
  // sharedU -> Nx.Ny.Nz.Uxyz
  // regR    -> Dz.Nx.Ny.Nz.Uxyz
  // sharedV -> Dy.Nx.Ny.Nz.Uxyz
  // sharedU -> Dx.Nx.Ny.Nz.Uxyz

  // 1st GEMM of D
  // Z Direction
  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          regP[k] =
            sharedU[threadIdxX + i * batchSize +
                    k * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          regP[nQuadPointsPerDim - 1 - k] =
            sharedU[threadIdxX + i * batchSize +
                    (nQuadPointsPerDim - 1 - k) * batchSize *
                      nQuadPointsPerDim * nQuadPointsPerDim];

          tempE = regP[k] + regP[nQuadPointsPerDim - 1 - k];
          tempO = regP[k] - regP[nQuadPointsPerDim - 1 - k];

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constD[j + k * qOdd] * tempE;

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j + qOdd] += constD[j + k * qEven + qOdd * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          regP[qOdd] =
            sharedU[threadIdxX + i * batchSize +
                    qOdd * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constD[j + qOdd * qOdd] * regP[qOdd];
        }

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
  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      std::uint32_t a = i % nQuadPointsPerDim;
      std::uint32_t b = i / nQuadPointsPerDim;

      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          temp1 =
            sharedU[threadIdxX + a * batchSize +
                    k * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          temp2 =
            sharedU[threadIdxX + a * batchSize +
                    (nQuadPointsPerDim - 1 - k) * batchSize *
                      nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          tempE = temp1 + temp2;
          tempO = temp1 - temp2;

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constD[j + k * qOdd] * tempE;

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j + qOdd] += constD[j + k * qEven + qOdd * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          tempE =
            sharedU[threadIdxX + a * batchSize +
                    qOdd * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constD[j + qOdd * qOdd] * tempE;
        }

      for (std::uint32_t j = 0; j < qOdd; j++)
        {
          sharedV[threadIdxX + a * batchSize +
                  j * batchSize * nQuadPointsPerDim +
                  b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
            regT[j + qOdd] + regT[j];

          sharedV[threadIdxX + a * batchSize +
                  (nQuadPointsPerDim - 1 - j) * batchSize * nQuadPointsPerDim +
                  b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
            regT[j + qOdd] - regT[j];
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        sharedV[threadIdxX + a * batchSize +
                qOdd * batchSize * nQuadPointsPerDim +
                b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
          regT[2 * qOdd];
    }

  // 3rd GEMM of D
  // X Direction
  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          temp1 = sharedU[threadIdxX + k * batchSize +
                          i * batchSize * nQuadPointsPerDim];

          temp2 = sharedU[threadIdxX + (nQuadPointsPerDim - 1 - k) * batchSize +
                          i * batchSize * nQuadPointsPerDim];

          tempE = temp1 + temp2;
          tempO = temp1 - temp2;

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constD[j + k * qOdd] * tempE;

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j + qOdd] += constD[j + k * qEven + qOdd * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          tempE = sharedU[threadIdxX + qOdd * batchSize +
                          i * batchSize * nQuadPointsPerDim];

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constD[j + qOdd * qOdd] * tempE;
        }
    }

  item.barrier(sycl::access::fence_space::local_space);

  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      for (std::uint32_t j = 0; j < qOdd; j++)
        {
          sharedU[threadIdxX + j * batchSize +
                  i * batchSize * nQuadPointsPerDim] = regT[j + qOdd] + regT[j];

          sharedU[threadIdxX + (nQuadPointsPerDim - 1 - j) * batchSize +
                  i * batchSize * nQuadPointsPerDim] = regT[j + qOdd] - regT[j];
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        sharedU[threadIdxX + qOdd * batchSize +
                i * batchSize * nQuadPointsPerDim] = regT[2 * qOdd];
    }

  item.barrier(sycl::access::fence_space::local_space);

  //////////////////////////////////////////////////////////////////
  // Jacobian Action
  // coeff.J^-T.J^-1.[sharedU sharedV regR]

  T detJ;

  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T t[dim];

      dftfe::uInt jOffset = blockIdxX * dim * dim;

      for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
        {
          t[0] = sharedU[threadIdxX +
                         (i + j * nQuadPointsPerDim * nQuadPointsPerDim) *
                           batchSize];
          t[1] = sharedV[threadIdxX +
                         (i + j * nQuadPointsPerDim * nQuadPointsPerDim) *
                           batchSize];
          t[2] = regR[j];

          sharedU[threadIdxX + (i + j * nQuadPointsPerDim * nQuadPointsPerDim) *
                                 batchSize] = J[0 + jOffset] * t[0] +
                                              J[1 + jOffset] * t[1] +
                                              J[2 + jOffset] * t[2];
          sharedV[threadIdxX + (i + j * nQuadPointsPerDim * nQuadPointsPerDim) *
                                 batchSize] = J[3 + jOffset] * t[0] +
                                              J[4 + jOffset] * t[1] +
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

  item.barrier(sycl::access::fence_space::local_space);

  //////////////////////////////////////////////////////////////////////////////////////////
  // Grad operation in each direction
  // regR -> Dz.Nx.Ny.Nz.Uxyz
  // regQ -> Dy.Nx.Ny.Nz.Uxyz
  // regP -> Dx.Nx.Ny.Nz.Uxyz
  // regR -> [DT.coeff.JF.D].Nx.Ny.Nz.Uxyz

  // 1st GEMM of DT
  // Z Direction
  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T tempE, tempO;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          tempE = regR[k] + regR[nQuadPointsPerDim - 1 - k];
          tempO = regR[k] - regR[nQuadPointsPerDim - 1 - k];

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constDT[j + k * qOdd + qEven * qOdd] * tempE;

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j + qOdd] += constDT[j + k * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constDT[j + qOdd * qOdd + qEven * qOdd] * regR[qOdd];
        }

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
  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      std::uint32_t a = i % nQuadPointsPerDim;
      std::uint32_t b = i / nQuadPointsPerDim;

      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          temp1 =
            sharedV[threadIdxX + a * batchSize +
                    k * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          temp2 =
            sharedV[threadIdxX + a * batchSize +
                    (nQuadPointsPerDim - 1 - k) * batchSize *
                      nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          tempE = temp1 + temp2;
          tempO = temp1 - temp2;

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constDT[j + k * qOdd + qEven * qOdd] * tempE;

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j + qOdd] += constDT[j + k * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          tempE =
            sharedV[threadIdxX + a * batchSize +
                    qOdd * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constDT[j + qOdd * qOdd + qEven * qOdd] * tempE;
        }

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
  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          temp1 = sharedU[threadIdxX + k * batchSize +
                          i * batchSize * nQuadPointsPerDim];

          temp2 = sharedU[threadIdxX + (nQuadPointsPerDim - 1 - k) * batchSize +
                          i * batchSize * nQuadPointsPerDim];

          tempE = temp1 + temp2;
          tempO = temp1 - temp2;

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constDT[j + k * qOdd + qEven * qOdd] * tempE;

          for (std::uint32_t j = 0; j < qEven; j++)
            regT[j + qOdd] += constDT[j + k * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          tempE = sharedU[threadIdxX + qOdd * batchSize +
                          i * batchSize * nQuadPointsPerDim];

          for (std::uint32_t j = 0; j < qOdd; j++)
            regT[j] += constDT[j + qOdd * qOdd + qEven * qOdd] * tempE;
        }

      for (std::uint32_t j = 0; j < qOdd; j++)
        {
          regP[j]                         = regT[j + qOdd] + regT[j];
          regP[nQuadPointsPerDim - 1 - j] = regT[j + qOdd] - regT[j];
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        regP[qOdd] = regT[2 * qOdd];
    }

  item.barrier(sycl::access::fence_space::local_space);

  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      std::uint32_t a = i % nQuadPointsPerDim;
      std::uint32_t b = i / nQuadPointsPerDim;

      for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
        sharedV[threadIdxX + a * batchSize + j * batchSize * nQuadPointsPerDim +
                b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
          regQ[j];

      for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
        sharedU[threadIdxX + j * batchSize +
                i * batchSize * nQuadPointsPerDim] = regP[j];
    }

  item.barrier(sycl::access::fence_space::local_space);

  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      for (std::uint32_t j = 0; j < nQuadPointsPerDim; j++)
        {
          regR[j] =
            regR[j] +
            sharedU[threadIdxX + i * batchSize +
                    j * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] +
            sharedV[threadIdxX + i * batchSize +
                    j * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];
        }
    }

  item.barrier(sycl::access::fence_space::local_space);

  //////////////////////////////////////////////////////////////////////////////////////////////////////
  // Integration combined with Assembly
  // V -> NTx.NTy.NTz.[DT.coeff.JF.D].Nx.Ny.Nz.Uxyz

  // 1st GEMM of NT
  // Z Direction
  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nQuadPointsPerDim;
       i += yThreads)
    {
      T tempE, tempO;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          tempE = regR[k] + regR[nQuadPointsPerDim - 1 - k];
          tempO = regR[k] - regR[nQuadPointsPerDim - 1 - k];

          for (std::uint32_t j = 0; j < pEven; j++)
            regT[j] += constNT[j + k * pEven] * tempE;

          for (std::uint32_t j = 0; j < pOdd; j++)
            regT[j + pEven] += constNT[j + k * pOdd + pEven * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          for (std::uint32_t j = 0; j < pEven; j++)
            regT[j] += constNT[j + qOdd * pEven] * regR[qOdd];
        }

      for (std::uint32_t j = 0; j < pOdd; j++)
        {
          sharedV[threadIdxX + i * batchSize +
                  j * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
            regT[j] + regT[j + pEven];

          sharedV[threadIdxX + i * batchSize +
                  (nDofsPerDim - 1 - j) * batchSize * nQuadPointsPerDim *
                    nQuadPointsPerDim] = regT[j] - regT[j + pEven];
        }

      if constexpr (nDofsPerDim % 2 == 1)
        sharedV[threadIdxX + i * batchSize +
                pOdd * batchSize * nQuadPointsPerDim * nQuadPointsPerDim] =
          regT[pOdd];
    }

  item.barrier(sycl::access::fence_space::local_space);

  // 2nd GEMM of NT
  // Y Direction
  for (std::uint32_t i = threadIdxY; i < nQuadPointsPerDim * nDofsPerDim;
       i += yThreads)
    {
      std::uint32_t a = i % nQuadPointsPerDim;
      std::uint32_t b = i / nQuadPointsPerDim;

      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          temp1 =
            sharedV[threadIdxX + a * batchSize +
                    k * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          temp2 =
            sharedV[threadIdxX + a * batchSize +
                    (nQuadPointsPerDim - 1 - k) * batchSize *
                      nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          tempE = temp1 + temp2;
          tempO = temp1 - temp2;

          for (std::uint32_t j = 0; j < pEven; j++)
            regT[j] += constNT[j + k * pEven] * tempE;

          for (std::uint32_t j = 0; j < pOdd; j++)
            regT[j + pEven] += constNT[j + k * pOdd + pEven * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          tempE =
            sharedV[threadIdxX + a * batchSize +
                    qOdd * batchSize * nQuadPointsPerDim +
                    b * batchSize * nQuadPointsPerDim * nQuadPointsPerDim];

          for (std::uint32_t j = 0; j < pEven; j++)
            regT[j] += constNT[j + qOdd * pEven] * tempE;
        }

      for (std::uint32_t j = 0; j < pOdd; j++)
        {
          sharedU[threadIdxX + a * batchSize +
                  j * batchSize * nQuadPointsPerDim +
                  b * batchSize * nQuadPointsPerDim * nDofsPerDim] =
            regT[j] + regT[j + pEven];

          sharedU[threadIdxX + a * batchSize +
                  (nDofsPerDim - 1 - j) * batchSize * nQuadPointsPerDim +
                  b * batchSize * nQuadPointsPerDim * nDofsPerDim] =
            regT[j] - regT[j + pEven];
        }

      if constexpr (nDofsPerDim % 2 == 1)
        sharedU[threadIdxX + a * batchSize +
                pOdd * batchSize * nQuadPointsPerDim +
                b * batchSize * nQuadPointsPerDim * nDofsPerDim] = regT[pOdd];
    }

  item.barrier(sycl::access::fence_space::local_space);

  // 3rd GEMM of NT
  // X Direction
  for (std::uint32_t i = threadIdxY; i < nDofsPerDim * nDofsPerDim;
       i += yThreads)
    {
      T tempE, tempO, temp1, temp2;

      memset(regT, 0, nQuadPointsPerDim * sizeof(T));

      for (std::uint32_t k = 0; k < qOdd; k++)
        {
          temp1 = sharedU[threadIdxX + k * batchSize +
                          i * batchSize * nQuadPointsPerDim];

          temp2 = sharedU[threadIdxX + (nQuadPointsPerDim - 1 - k) * batchSize +
                          i * batchSize * nQuadPointsPerDim];

          tempE = temp1 + temp2;
          tempO = temp1 - temp2;

          for (std::uint32_t j = 0; j < pEven; j++)
            regT[j] += constNT[j + k * pEven] * tempE;

          for (std::uint32_t j = 0; j < pOdd; j++)
            regT[j + pEven] += constNT[j + k * pOdd + pEven * qEven] * tempO;
        }

      if constexpr (nQuadPointsPerDim % 2 == 1)
        {
          tempE = sharedU[threadIdxX + qOdd * batchSize +
                          i * batchSize * nQuadPointsPerDim];

          for (std::uint32_t j = 0; j < pEven; j++)
            regT[j] += constNT[j + qOdd * pEven] * tempE;
        }

      for (std::uint32_t j = 0; j < pOdd; j++)
        {
          dftfe::uInt dof1 = map[j + i * nDofsPerDim + mapOffset];

          sycl::atomic_ref<T,
                           sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
            atomicRef1(dst[threadIdxX + dof1]);
          atomicRef1 += regT[j] + regT[j + pEven];

          dftfe::uInt dof2 =
            map[(nDofsPerDim - 1 - j) + i * nDofsPerDim + mapOffset];

          sycl::atomic_ref<T,
                           sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
            atomicRef2(dst[threadIdxX + dof2]);
          atomicRef2 += regT[j] - regT[j + pEven];
        }

      if constexpr (nDofsPerDim % 2 == 1)
        {
          dftfe::uInt dof = map[pOdd + i * nDofsPerDim + mapOffset];

          sycl::atomic_ref<T,
                           sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
            atomicRef(dst[threadIdxX + dof]);
          atomicRef += regT[pOdd];
        }
    }
}
