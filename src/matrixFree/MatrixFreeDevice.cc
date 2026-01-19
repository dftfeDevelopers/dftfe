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

namespace dftfe
{

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
#  include "MatrixFreeDevice.cu.cc"
#elif DFTFE_WITH_DEVICE_LANG_HIP
#  include "MatrixFreeDevice.hip.cc"
#elif DFTFE_WITH_DEVICE_LANG_SYCL
#  include "MatrixFreeDevice.sycl.cc"
#endif

  template <typename T,
            dftfe::operatorList operatorID,
            std::uint32_t       nDofsPerDim,
            std::uint32_t       nQuadPointsPerDim,
            std::uint32_t       batchSize>
  inline void
  MatrixFreeDevice<T, operatorID, nDofsPerDim, nQuadPointsPerDim, batchSize>::
    init(T *constMemHost, std::size_t constMemSize)
  {
    constexpr std::uint32_t dim           = 3;
    constexpr std::size_t   sharedMemSize = 2 * batchSize * nQuadPointsPerDim *
                                          nQuadPointsPerDim *
                                          nQuadPointsPerDim * sizeof(T);

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
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

        if constexpr (operatorID == dftfe::operatorList::Laplace)
          DEVICE_API_CHECK(cudaFuncSetAttribute(
            LaplaceKernel<T, nDofsPerDim, nQuadPointsPerDim, batchSize, dim>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            sharedMemSize));

        if constexpr (operatorID == dftfe::operatorList::Helmholtz)
          DEVICE_API_CHECK(cudaFuncSetAttribute(
            HelmholtzKernel<T, nDofsPerDim, nQuadPointsPerDim, batchSize, dim>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            sharedMemSize));
      }

#elif DFTFE_WITH_DEVICE_LANG_HIP
    // Copy shape functions and gradients to constant memory on device
    DEVICE_API_CHECK(hipMemcpyToSymbol(constMem,
                                       constMemHost,
                                       constMemSize * sizeof(T),
                                       0,
                                       hipMemcpyHostToDevice));
#endif
  }


  template <typename T,
            dftfe::operatorList operatorID,
            std::uint32_t       nDofsPerDim,
            std::uint32_t       nQuadPointsPerDim,
            std::uint32_t       batchSize>
  inline void
  MatrixFreeDevice<T, operatorID, nDofsPerDim, nQuadPointsPerDim, batchSize>::
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

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA

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

#elif DFTFE_WITH_DEVICE_LANG_HIP

    hipLaunchKernelGGL(
      HIP_KERNEL_NAME(
        constraintsDistributeKernel<double, nDofsPerDim, batchSize>),
      blocks,
      threads,
      0,
      0,
      src,
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

#elif DFTFE_WITH_DEVICE_LANG_SYCL
#endif
  }


  template <typename T,
            dftfe::operatorList operatorID,
            std::uint32_t       nDofsPerDim,
            std::uint32_t       nQuadPointsPerDim,
            std::uint32_t       batchSize>
  inline void
  MatrixFreeDevice<T, operatorID, nDofsPerDim, nQuadPointsPerDim, batchSize>::
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

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA

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

#elif DFTFE_WITH_DEVICE_LANG_HIP

    hipLaunchKernelGGL(
      HIP_KERNEL_NAME(
        constraintsDistributeTransposeKernel<double, nDofsPerDim, batchSize>),
      blocks,
      threads,
      0,
      0,
      dst,
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

#elif DFTFE_WITH_DEVICE_LANG_SYCL
#endif
  }


  template <typename T,
            dftfe::operatorList operatorID,
            std::uint32_t       nDofsPerDim,
            std::uint32_t       nQuadPointsPerDim,
            std::uint32_t       batchSize>
  inline void
  MatrixFreeDevice<T, operatorID, nDofsPerDim, nQuadPointsPerDim, batchSize>::
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

    LaplaceKernel<T, nDofsPerDim, nQuadPointsPerDim, batchSize, dim>
      <<<blocks, threads, sharedMemSize>>>(dst, src, jacobianFactor, map);

#elif DFTFE_WITH_DEVICE_LANG_HIP

    hipLaunchKernelGGL(
      HIP_KERNEL_NAME(
        LaplaceKernel<T, nDofsPerDim, nQuadPointsPerDim, batchSize, dim>),
      blocks,
      threads,
      sharedMemSize,
      0,
      dst,
      src,
      jacobianFactor,
      map);

#elif DFTFE_WITH_DEVICE_LANG_SYCL
#endif
  }

  template <typename T,
            dftfe::operatorList operatorID,
            std::uint32_t       nDofsPerDim,
            std::uint32_t       nQuadPointsPerDim,
            std::uint32_t       batchSize>
  inline void
  MatrixFreeDevice<T, operatorID, nDofsPerDim, nQuadPointsPerDim, batchSize>::
    computeHelmholtzX(T             *dst,
                      T             *src,
                      T             *jacobianFactor,
                      std::uint32_t *map,
                      T              coeffHelmholtz,
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

    HelmholtzKernel<T, nDofsPerDim, nQuadPointsPerDim, batchSize, dim>
      <<<blocks, threads, sharedMemSize>>>(
        dst, src, jacobianFactor, map, coeffHelmholtz);

#elif DFTFE_WITH_DEVICE_LANG_HIP

    hipLaunchKernelGGL(
      HIP_KERNEL_NAME(
        HelmholtzKernel<T, nDofsPerDim, nQuadPointsPerDim, batchSize, dim>),
      blocks,
      threads,
      sharedMemSize,
      0,
      dst,
      src,
      jacobianFactor,
      map,
      coeffHelmholtz);

#elif DFTFE_WITH_DEVICE_LANG_SYCL
#endif
  }

#include "MatrixFreeDevice.inst.cc"
} // namespace dftfe
