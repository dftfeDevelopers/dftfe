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


#ifndef kohnShamDFTOperatorCUDAKernels_H_
#define kohnShamDFTOperatorCUDAKernels_H_

#include <headers.h>

namespace dftfe
{
  __global__ void
  memCpyKernel(const unsigned int size,
               const double *     copyFromVec,
               double *           copyToVec);

  __global__ void
  addKernel(const unsigned int size, const double *addVec, double *addToVec);

  __global__ void
  copyCUDAKernel(const unsigned int contiguousBlockSize,
                 const unsigned int numContiguousBlocks,
                 const double *     copyFromVec,
                 double *           copyToVec,
                 const dealii::types::global_dof_index
                   *copyFromVecStartingContiguousBlockIds);


  __global__ void
  daxpyCUDAKernel(
    const unsigned int                     contiguousBlockSize,
    const unsigned int                     numContiguousBlocks,
    const double *                         xVec,
    double *                               yVec,
    const dealii::types::global_dof_index *yVecStartingContiguousBlockIds,
    const double                           a);


  __global__ void
  daxpyAtomicAddKernel(
    const unsigned int                     contiguousBlockSize,
    const unsigned int                     numContiguousBlocks,
    const double *                         addFromVec,
    double *                               addToVec,
    const dealii::types::global_dof_index *addToVecStartingContiguousBlockIds);

  __global__ void
  daxpyAtomicAddKernelNonBoundary(
    const unsigned int                     contiguousBlockSize,
    const unsigned int                     numContiguousBlocks,
    const double *                         addFromVec,
    const unsigned int *                   boundaryIdVec,
    double *                               addToVec,
    const dealii::types::global_dof_index *addToVecStartingContiguousBlockIds);


  __global__ void
  copyToParallelNonLocalVecFromReducedVec(
    const unsigned int  numWfcs,
    const unsigned int  totalPseudoWfcs,
    const double *      reducedProjectorKetTimesWfcVec,
    double *            projectorKetTimesWfcParallelVec,
    const unsigned int *indexMapFromParallelVecToReducedVec);


  __global__ void
  copyFromParallelNonLocalVecToAllCellsVec(
    const unsigned int numWfcs,
    const unsigned int numNonLocalCells,
    const unsigned int maxSingleAtomPseudoWfc,
    const double *     projectorKetTimesWfcParallelVec,
    double *           projectorKetTimesWfcAllCellsVec,
    const int *        indexMapPaddedToParallelVec);


  __global__ void
  copyToDealiiParallelNonLocalVec(
    const unsigned int  numWfcs,
    const unsigned int  totalPseudoWfcs,
    const double *      projectorKetTimesWfcParallelVec,
    double *            projectorKetTimesWfcDealiiParallelVec,
    const unsigned int *indexMapDealiiParallelNumbering);


  __global__ void
  copyFromDealiiParallelNonLocalVec(
    const unsigned int  numWfcs,
    const unsigned int  totalPseudoWfcs,
    double *            projectorKetTimesWfcParallelVec,
    const double *      projectorKetTimesWfcDealiiParallelVec,
    const unsigned int *indexMapDealiiParallelNumbering);

  __global__ void
  addNonLocalContributionCUDAKernel(
    const dealii::types::global_dof_index contiguousBlockSize,
    const dealii::types::global_dof_index numContiguousBlocks,
    const double *                        xVec,
    double *                              yVec,
    const unsigned int *                  xVecToyVecBlockIdMap);

} // namespace dftfe
#endif
