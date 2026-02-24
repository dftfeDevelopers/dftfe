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
    constexpr std::uint32_t maxDofsPerDim = 17;

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    // Copy shape functions and gradients to constant memory on device
    DEVICE_API_CHECK(cudaMemcpyToSymbol(
      constMem,
      constMemHost,
      constMemSize * sizeof(T),
      (operatorID * (maxDofsPerDim * maxDofsPerDim * 5 + maxDofsPerDim)) *
        sizeof(T),
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
    DEVICE_API_CHECK(hipMemcpyToSymbol(
      constMem,
      constMemHost,
      constMemSize * sizeof(T),
      (operatorID * (maxDofsPerDim * maxDofsPerDim * 5 + maxDofsPerDim)) *
        sizeof(T),
      hipMemcpyHostToDevice));

    int deviceId = 0;
    DEVICE_API_CHECK(hipGetDevice(&deviceId));

    int maxDynSharedDefault = 0;
    DEVICE_API_CHECK(
      hipDeviceGetAttribute(&maxDynSharedDefault,
                            hipDeviceAttributeMaxSharedMemoryPerBlock,
                            deviceId));

    if (sharedMemSize > static_cast<std::size_t>(maxDynSharedDefault))
      throw std::runtime_error(
        "Requested dynamic shared memory exceeds max limit");
#endif
  }


  template <typename T,
            dftfe::operatorList operatorID,
            std::uint32_t       nDofsPerDim,
            std::uint32_t       nQuadPointsPerDim,
            std::uint32_t       batchSize>
  inline void
  MatrixFreeDevice<T, operatorID, nDofsPerDim, nQuadPointsPerDim, batchSize>::
    constraintsDistribute(T                 *src,
                          const dftfe::uInt *constrainingNodeBuckets,
                          const dftfe::uInt *constrainingNodeOffset,
                          const dftfe::uInt *constrainedNodeBuckets,
                          const dftfe::uInt *constrainedNodeOffset,
                          const T           *weightMatrixList,
                          const dftfe::uInt *weightMatrixOffset,
                          const T           *inhomogenityList,
                          const dftfe::uInt *ghostMap,
                          const dftfe::uInt  inhomogenityListSize,
                          const dftfe::uInt  nBatch,
                          const dftfe::uInt  nOwnedDofs,
                          const dftfe::uInt  nGhostDofs)
  {
    constexpr int yThreads = 64;

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA

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

#elif DFTFE_WITH_DEVICE_LANG_HIP

    dim3 blocks(inhomogenityListSize, nBatch, 1);
    dim3 threads(batchSize, yThreads, 1);

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
    sycl::queue &queue =
      dftfe::utils::queueRegistry.find(dftfe::utils::defaultStream)->second;

    constexpr std::size_t sharedMemSizeConstraints =
      batchSize * nDofsPerDim * nDofsPerDim;

    queue.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<T, 1> sharedConstrainingData(
        sharedMemSizeConstraints, cgh);
      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, nBatch, inhomogenityListSize) *
                            sycl::range<3>(1, yThreads, batchSize),
                          sycl::range<3>(1, yThreads, batchSize)),
        [=](sycl::nd_item<3> item) {
          constraintsDistributeKernel<T, nDofsPerDim, batchSize>(
            item,
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
            nGhostDofs,
            sharedConstrainingData);
        });
    });
#endif
  }


  template <typename T,
            dftfe::operatorList operatorID,
            std::uint32_t       nDofsPerDim,
            std::uint32_t       nQuadPointsPerDim,
            std::uint32_t       batchSize>
  inline void
  MatrixFreeDevice<T, operatorID, nDofsPerDim, nQuadPointsPerDim, batchSize>::
    constraintsDistributeTranspose(T                 *dst,
                                   T                 *src,
                                   const dftfe::uInt *constrainingNodeBuckets,
                                   const dftfe::uInt *constrainingNodeOffset,
                                   const dftfe::uInt *constrainedNodeBuckets,
                                   const dftfe::uInt *constrainedNodeOffset,
                                   const T           *weightMatrixList,
                                   const dftfe::uInt *weightMatrixOffset,
                                   const dftfe::uInt *ghostMap,
                                   const dftfe::uInt  inhomogenityListSize,
                                   const dftfe::uInt  nBatch,
                                   const dftfe::uInt  nOwnedDofs,
                                   const dftfe::uInt  nGhostDofs)
  {
    constexpr int yThreads = 64;

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA

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

#elif DFTFE_WITH_DEVICE_LANG_HIP

    dim3 blocks(inhomogenityListSize, nBatch, 1);
    dim3 threads(batchSize, yThreads, 1);

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
    sycl::queue &queue =
      dftfe::utils::queueRegistry.find(dftfe::utils::defaultStream)->second;

    constexpr std::size_t sharedMemSizeConstraints =
      batchSize * nDofsPerDim * nDofsPerDim * 4;

    queue.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<T, 1> sharedConstrainedData(sharedMemSizeConstraints,
                                                       cgh);
      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, nBatch, inhomogenityListSize) *
                            sycl::range<3>(1, yThreads, batchSize),
                          sycl::range<3>(1, yThreads, batchSize)),
        [=](sycl::nd_item<3> item) {
          constraintsDistributeTransposeKernel<T, nDofsPerDim, batchSize>(
            item,
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
            nGhostDofs,
            sharedConstrainedData);
        });
    });
#endif
  }


  template <typename T,
            dftfe::operatorList operatorID,
            std::uint32_t       nDofsPerDim,
            std::uint32_t       nQuadPointsPerDim,
            std::uint32_t       batchSize>
  inline void
  MatrixFreeDevice<T, operatorID, nDofsPerDim, nQuadPointsPerDim, batchSize>::
    computeLaplaceX(T           *dst,
                    T           *src,
                    T           *jacobianFactor,
                    dftfe::uInt *map,
                    T           *shapeBuffer,
                    dftfe::uInt  nCells,
                    dftfe::uInt  nBatch)
  {
    constexpr std::uint32_t dim = 3;
    constexpr std::uint32_t yThreads =
      dftfe::utils::DEVICE_WARP_SIZE * ((nQuadPointsPerDim * nQuadPointsPerDim +
                                         dftfe::utils::DEVICE_WARP_SIZE - 1) /
                                        dftfe::utils::DEVICE_WARP_SIZE);
    constexpr std::size_t sharedMemSize = 2 * batchSize * nQuadPointsPerDim *
                                          nQuadPointsPerDim *
                                          nQuadPointsPerDim * sizeof(T);

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA

    const dim3 blocks(nCells, nBatch, 1);
    const dim3 threads(batchSize, yThreads, 1);

    LaplaceKernel<T, nDofsPerDim, nQuadPointsPerDim, batchSize, dim>
      <<<blocks, threads, sharedMemSize>>>(dst, src, jacobianFactor, map);

#elif DFTFE_WITH_DEVICE_LANG_HIP

    const dim3 blocks(nCells, nBatch, 1);
    const dim3 threads(batchSize, yThreads, 1);

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
    sycl::queue &queue =
      dftfe::utils::queueRegistry.find(dftfe::utils::defaultStream)->second;

    constexpr std::uint32_t pOddL  = nDofsPerDim / 2;
    constexpr std::uint32_t pEvenL = nDofsPerDim % 2 == 1 ? pOddL + 1 : pOddL;
    constexpr std::uint32_t qOddL  = nQuadPointsPerDim / 2;
    constexpr std::uint32_t qEvenL =
      nQuadPointsPerDim % 2 == 1 ? qOddL + 1 : qOddL;
    constexpr std::size_t shapeBufferElements =
      2 * (qEvenL * pEvenL + qOddL * pOddL) + 4 * qEvenL * qOddL +
      nQuadPointsPerDim * nDofsPerDim + nQuadPointsPerDim;
    constexpr std::size_t sharedMemElements =
      2 * batchSize * nQuadPointsPerDim * nQuadPointsPerDim *
        nQuadPointsPerDim +
      shapeBufferElements;

    queue.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<T, 1> sharedMem(sharedMemElements, cgh);
      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, nBatch, nCells) *
                            sycl::range<3>(1, yThreads, batchSize),
                          sycl::range<3>(1, yThreads, batchSize)),
        [=](sycl::nd_item<3> item) {
          LaplaceKernel<T, nDofsPerDim, nQuadPointsPerDim, batchSize, dim>(
            item, dst, src, jacobianFactor, map, shapeBuffer, sharedMem);
        });
    });
#endif
  }

  template <typename T,
            dftfe::operatorList operatorID,
            std::uint32_t       nDofsPerDim,
            std::uint32_t       nQuadPointsPerDim,
            std::uint32_t       batchSize>
  inline void
  MatrixFreeDevice<T, operatorID, nDofsPerDim, nQuadPointsPerDim, batchSize>::
    computeHelmholtzX(T           *dst,
                      T           *src,
                      T           *jacobianFactor,
                      dftfe::uInt *map,
                      T           *shapeBuffer,
                      T            coeffHelmholtz,
                      dftfe::uInt  nCells,
                      dftfe::uInt  nBatch)
  {
    constexpr std::uint32_t dim = 3;
    constexpr std::uint32_t yThreads =
      dftfe::utils::DEVICE_WARP_SIZE * ((nQuadPointsPerDim * nQuadPointsPerDim +
                                         dftfe::utils::DEVICE_WARP_SIZE - 1) /
                                        dftfe::utils::DEVICE_WARP_SIZE);
    constexpr std::size_t sharedMemSize = 2 * batchSize * nQuadPointsPerDim *
                                          nQuadPointsPerDim *
                                          nQuadPointsPerDim * sizeof(T);

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA

    const dim3 blocks(nCells, nBatch, 1);
    const dim3 threads(batchSize, yThreads, 1);

    HelmholtzKernel<T, nDofsPerDim, nQuadPointsPerDim, batchSize, dim>
      <<<blocks, threads, sharedMemSize>>>(
        dst, src, jacobianFactor, map, coeffHelmholtz);

#elif DFTFE_WITH_DEVICE_LANG_HIP

    const dim3 blocks(nCells, nBatch, 1);
    const dim3 threads(batchSize, yThreads, 1);

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
    sycl::queue &queue =
      dftfe::utils::queueRegistry.find(dftfe::utils::defaultStream)->second;

    constexpr std::uint32_t pOddH  = nDofsPerDim / 2;
    constexpr std::uint32_t pEvenH = nDofsPerDim % 2 == 1 ? pOddH + 1 : pOddH;
    constexpr std::uint32_t qOddH  = nQuadPointsPerDim / 2;
    constexpr std::uint32_t qEvenH =
      nQuadPointsPerDim % 2 == 1 ? qOddH + 1 : qOddH;
    constexpr std::size_t shapeBufferElements =
      2 * (qEvenH * pEvenH + qOddH * pOddH) + 4 * qEvenH * qOddH +
      nQuadPointsPerDim * nDofsPerDim + nQuadPointsPerDim;
    constexpr std::size_t sharedMemElements =
      2 * batchSize * nQuadPointsPerDim * nQuadPointsPerDim *
        nQuadPointsPerDim +
      shapeBufferElements;

    queue.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<T, 1> sharedMem(sharedMemElements, cgh);
      cgh.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, nBatch, nCells) *
                            sycl::range<3>(1, yThreads, batchSize),
                          sycl::range<3>(1, yThreads, batchSize)),
        [=](sycl::nd_item<3> item) {
          HelmholtzKernel<T, nDofsPerDim, nQuadPointsPerDim, batchSize, dim>(
            item,
            dst,
            src,
            jacobianFactor,
            map,
            coeffHelmholtz,
            shapeBuffer,
            sharedMem);
        });
    });
#endif
  }

#include "MatrixFreeDevice.inst.cc"
} // namespace dftfe
