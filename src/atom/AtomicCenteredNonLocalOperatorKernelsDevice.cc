// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025 The Regents of the University of Michigan and DFT-FE
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
// @author Kartick Ramakrishnan
//

#include <AtomicCenteredNonLocalOperatorKernelsDevice.h>
#include <deviceKernelsGeneric.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherConstants.h>
namespace dftfe
{
  namespace
  {
    template <typename ValueType>
    __global__ void
    sqrtAlphaScalingWaveFunctionEntriesKernel(
      const dftfe::uInt numWfcs,
      const dftfe::uInt totalAtomsInCurrentProcessor,
      const dftfe::uInt maxSingleAtomPseudoWfc,
      const double     *scalingVector,
      ValueType        *sphericalFnTimesWfcPadded)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        totalAtomsInCurrentProcessor * maxSingleAtomPseudoWfc * numWfcs;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::uInt iAtom = index / (maxSingleAtomPseudoWfc * numWfcs);
          const dftfe::uInt iOrb =
            (index - iAtom * maxSingleAtomPseudoWfc * numWfcs) / numWfcs;
          const dftfe::uInt wfcIndex =
            (index - iAtom * maxSingleAtomPseudoWfc * numWfcs) % numWfcs;
          const double alpha = scalingVector[wfcIndex];
          dftfe::utils::copyValue(
            sphericalFnTimesWfcPadded + index,
            dftfe::utils::mult(alpha, sphericalFnTimesWfcPadded[index]));
        }
    }

    template <typename ValueType>
    __global__ void
    copyFromParallelNonLocalVecToAllCellsVecKernel(
      const dftfe::uInt numWfcs,
      const dftfe::uInt numNonLocalCells,
      const dftfe::uInt maxSingleAtomPseudoWfc,
      const ValueType  *sphericalFnTimesWfcParallelVec,
      ValueType        *sphericalFnTimesWfcAllCellsVec,
      const dftfe::Int *indexMapPaddedToParallelVec)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numNonLocalCells * maxSingleAtomPseudoWfc * numWfcs;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::uInt blockIndex      = index / numWfcs;
          const dftfe::uInt intraBlockIndex = index % numWfcs;
          const dftfe::Int  mappedIndex =
            indexMapPaddedToParallelVec[blockIndex];
          if (mappedIndex != -1)
            sphericalFnTimesWfcAllCellsVec[index] =
              sphericalFnTimesWfcParallelVec[mappedIndex * numWfcs +
                                             intraBlockIndex];
        }
    }


    template <typename ValueType>
    __global__ void
    copyToDealiiParallelNonLocalVecKernel(
      const dftfe::uInt  numWfcs,
      const dftfe::uInt  totalPseudoWfcs,
      const ValueType   *sphericalFnTimesWfcParallelVec,
      ValueType         *sphericalFnTimesWfcDealiiParallelVec,
      const dftfe::uInt *indexMapDealiiParallelNumbering)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries  = totalPseudoWfcs * numWfcs;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::uInt blockIndex      = index / numWfcs;
          const dftfe::uInt intraBlockIndex = index % numWfcs;
          const dftfe::uInt mappedIndex =
            indexMapDealiiParallelNumbering[blockIndex];

          sphericalFnTimesWfcDealiiParallelVec[mappedIndex * numWfcs +
                                               intraBlockIndex] =
            sphericalFnTimesWfcParallelVec[index];
        }
    }
    template <typename ValueType>
    __global__ void
    copyToDealiiParallelNonLocalVecFromPaddedVecKernel(
      const dftfe::uInt numWfcs,
      const dftfe::uInt totalPaddedPseudoWfcs,
      const ValueType  *sphericalFnTimesWfcPaddedVec,
      ValueType        *sphericalFnTimesWfcDealiiParallelVec,
      const dftfe::Int *indexMapDealiiParallelNumbering)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries  = totalPaddedPseudoWfcs * numWfcs;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::uInt blockIndex      = index / numWfcs;
          const dftfe::uInt intraBlockIndex = index % numWfcs;
          const dftfe::Int  mappedIndex =
            indexMapDealiiParallelNumbering[blockIndex];

          if (mappedIndex != -1)
            sphericalFnTimesWfcDealiiParallelVec[mappedIndex * numWfcs +
                                                 intraBlockIndex] =
              sphericalFnTimesWfcPaddedVec[index];
        }
    }
    template <typename ValueType>
    __global__ void
    copyFromDealiiParallelNonLocalVecToPaddedVecKernel(
      const dftfe::uInt numWfcs,
      const dftfe::uInt totalPaddedPseudoWfcs,
      const ValueType  *sphericalFnTimesWfcDealiiParallelVec,
      ValueType        *sphericalFnTimesWfcPaddedVec,
      const dftfe::Int *indexMapDealiiParallelNumbering)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries  = totalPaddedPseudoWfcs * numWfcs;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::uInt blockIndex      = index / numWfcs;
          const dftfe::uInt intraBlockIndex = index % numWfcs;
          const dftfe::Int  mappedIndex =
            indexMapDealiiParallelNumbering[blockIndex];

          if (mappedIndex != -1)
            sphericalFnTimesWfcPaddedVec[index] =
              sphericalFnTimesWfcDealiiParallelVec[mappedIndex * numWfcs +
                                                   intraBlockIndex];
        }
    }

    __global__ void
    addNonLocalContributionDeviceKernel(
      const dftfe::uInt  totalNonLocalElements,
      const dftfe::uInt  numberWfc,
      const dftfe::uInt  numberNodesPerElement,
      const dftfe::uInt *iElemNonLocalToElemIndexMap,
      const double      *xVec,
      double            *yVec)
    {
      const dealii::types::global_dof_index globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dealii::types::global_dof_index totalEntries =
        totalNonLocalElements * numberWfc * numberNodesPerElement;
      for (dftfe::uInt index = globalThreadId; index < totalEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::uInt iElem = index / (numberWfc * numberNodesPerElement);
          const dftfe::uInt elemIndex = iElemNonLocalToElemIndexMap[iElem];
          const dftfe::uInt index2 =
            index % (numberWfc * numberNodesPerElement);
          const dftfe::uInt iDof     = index2 / numberWfc;
          const dftfe::uInt wfcIndex = index2 % numberWfc;
          atomicAdd(&yVec[elemIndex * numberNodesPerElement * numberWfc +
                          iDof * numberWfc + wfcIndex],
                    xVec[index]);
        }
    }

    __global__ void
    addNonLocalContributionDeviceKernel(
      const dftfe::uInt  totalNonLocalElements,
      const dftfe::uInt  numberWfc,
      const dftfe::uInt  numberNodesPerElement,
      const dftfe::uInt *iElemNonLocalToElemIndexMap,
      const float       *xVec,
      float             *yVec)
    {
      const dealii::types::global_dof_index globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dealii::types::global_dof_index totalEntries =
        totalNonLocalElements * numberWfc * numberNodesPerElement;
      for (dftfe::uInt index = globalThreadId; index < totalEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::uInt iElem = index / (numberWfc * numberNodesPerElement);
          const dftfe::uInt elemIndex = iElemNonLocalToElemIndexMap[iElem];
          const dftfe::uInt index2 =
            index % (numberWfc * numberNodesPerElement);
          const dftfe::uInt iDof     = index2 / numberWfc;
          const dftfe::uInt wfcIndex = index2 % numberWfc;
          atomicAdd(&yVec[elemIndex * numberNodesPerElement * numberWfc +
                          iDof * numberWfc + wfcIndex],
                    xVec[index]);
        }
    }


    __global__ void
    addNonLocalContributionDeviceKernel(
      const dftfe::uInt                        totalNonLocalElements,
      const dftfe::uInt                        numberWfc,
      const dftfe::uInt                        numberNodesPerElement,
      const dftfe::uInt                       *iElemNonLocalToElemIndexMap,
      const dftfe::utils::deviceDoubleComplex *xVec,
      dftfe::utils::deviceDoubleComplex       *yVec)
    {
      const dealii::types::global_dof_index globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dealii::types::global_dof_index totalEntries =
        totalNonLocalElements * numberWfc * numberNodesPerElement;
      for (dftfe::uInt index = globalThreadId; index < totalEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::uInt iElem = index / (numberWfc * numberNodesPerElement);
          const dftfe::uInt elemIndex = iElemNonLocalToElemIndexMap[iElem];
          const dftfe::uInt index2 =
            index % (numberWfc * numberNodesPerElement);
          const dftfe::uInt iDof     = index2 / numberWfc;
          const dftfe::uInt wfcIndex = index2 % numberWfc;
          atomicAdd(&yVec[elemIndex * numberNodesPerElement * numberWfc +
                          iDof * numberWfc + wfcIndex]
                       .x,
                    xVec[index].x);
          atomicAdd(&yVec[elemIndex * numberNodesPerElement * numberWfc +
                          iDof * numberWfc + wfcIndex]
                       .y,
                    xVec[index].y);
        }
    }


    __global__ void
    addNonLocalContributionDeviceKernel(
      const dftfe::uInt                       totalNonLocalElements,
      const dftfe::uInt                       numberWfc,
      const dftfe::uInt                       numberNodesPerElement,
      const dftfe::uInt                      *iElemNonLocalToElemIndexMap,
      const dftfe::utils::deviceFloatComplex *xVec,
      dftfe::utils::deviceFloatComplex       *yVec)
    {
      const dealii::types::global_dof_index globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dealii::types::global_dof_index totalEntries =
        totalNonLocalElements * numberWfc * numberNodesPerElement;
      for (dftfe::uInt index = globalThreadId; index < totalEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::uInt iElem = index / (numberWfc * numberNodesPerElement);
          const dftfe::uInt elemIndex = iElemNonLocalToElemIndexMap[iElem];
          const dftfe::uInt index2 =
            index % (numberWfc * numberNodesPerElement);
          const dftfe::uInt iDof     = index2 / numberWfc;
          const dftfe::uInt wfcIndex = index2 % numberWfc;
          atomicAdd(&yVec[elemIndex * numberNodesPerElement * numberWfc +
                          iDof * numberWfc + wfcIndex]
                       .x,
                    xVec[index].x);
          atomicAdd(&yVec[elemIndex * numberNodesPerElement * numberWfc +
                          iDof * numberWfc + wfcIndex]
                       .y,
                    xVec[index].y);
        }
    }

    template <typename ValueType>
    __global__ void
    addNonLocalContributionDeviceKernel(const dftfe::uInt  contiguousBlockSize,
                                        const dftfe::uInt  numContiguousBlocks,
                                        const ValueType   *xVec,
                                        ValueType         *yVec,
                                        const dftfe::uInt *xVecToyVecBlockIdMap)
    {
      const dealii::types::global_dof_index globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dealii::types::global_dof_index numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dealii::types::global_dof_index blockIndex =
            index / contiguousBlockSize;
          dealii::types::global_dof_index intraBlockIndex =
            index % contiguousBlockSize;
          yVec[xVecToyVecBlockIdMap[blockIndex] * contiguousBlockSize +
               intraBlockIndex] =
            dftfe::utils::add(
              yVec[xVecToyVecBlockIdMap[blockIndex] * contiguousBlockSize +
                   intraBlockIndex],
              xVec[index]);
        }
    }
    __global__ void
    assembleAtomLevelContributionsFromCellLevelKernel(
      const dftfe::uInt  numWfc,
      const dftfe::uInt  totalNonLocalElements,
      const dftfe::uInt  maxSingleAtomSphericalFn,
      const dftfe::uInt  totalNonLocalEntries,
      const double      *sphericalFnTimesXCellLevel,
      const dftfe::uInt *mappingCellLevelToAtomLevel,
      double            *sphericalFnTimesX)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        totalNonLocalElements * maxSingleAtomSphericalFn * numWfc;
      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::uInt blockIndex      = index / (numWfc);
          const dftfe::uInt intraBlockIndex = index % numWfc;

          const dftfe::uInt toIndex = mappingCellLevelToAtomLevel[blockIndex];

          if (toIndex < totalNonLocalEntries)
            atomicAdd(&sphericalFnTimesX[toIndex * numWfc + intraBlockIndex],
                      sphericalFnTimesXCellLevel[index]);
        }
    }


    __global__ void
    assembleAtomLevelContributionsFromCellLevelKernel(
      const dftfe::uInt  numWfc,
      const dftfe::uInt  totalNonLocalElements,
      const dftfe::uInt  maxSingleAtomSphericalFn,
      const dftfe::uInt  totalNonLocalEntries,
      const float       *sphericalFnTimesXCellLevel,
      const dftfe::uInt *mappingCellLevelToAtomLevel,
      float             *sphericalFnTimesX)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        totalNonLocalElements * maxSingleAtomSphericalFn * numWfc;
      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::uInt blockIndex      = index / (numWfc);
          const dftfe::uInt intraBlockIndex = index % numWfc;

          const dftfe::uInt toIndex = mappingCellLevelToAtomLevel[blockIndex];
          if (toIndex < totalNonLocalEntries)
            atomicAdd(&sphericalFnTimesX[toIndex * numWfc + intraBlockIndex],
                      sphericalFnTimesXCellLevel[index]);
        }
    }

    __global__ void
    assembleAtomLevelContributionsFromCellLevelKernel(
      const dftfe::uInt                        numWfc,
      const dftfe::uInt                        totalNonLocalElements,
      const dftfe::uInt                        maxSingleAtomSphericalFn,
      const dftfe::uInt                        totalNonLocalEntries,
      const dftfe::utils::deviceDoubleComplex *sphericalFnTimesXCellLevel,
      const dftfe::uInt                       *mappingCellLevelToAtomLevel,
      dftfe::utils::deviceDoubleComplex       *sphericalFnTimesX)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        totalNonLocalElements * maxSingleAtomSphericalFn * numWfc;
      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::uInt blockIndex      = index / (numWfc);
          const dftfe::uInt intraBlockIndex = index % numWfc;

          const dftfe::uInt toIndex = mappingCellLevelToAtomLevel[blockIndex];
          if (toIndex < totalNonLocalEntries)
            {
              atomicAdd(
                &sphericalFnTimesX[toIndex * numWfc + intraBlockIndex].x,
                sphericalFnTimesXCellLevel[index].x);
              atomicAdd(
                &sphericalFnTimesX[toIndex * numWfc + intraBlockIndex].y,
                sphericalFnTimesXCellLevel[index].y);
            }
        }
    }

    __global__ void
    assembleAtomLevelContributionsFromCellLevelKernel(
      const dftfe::uInt                       numWfc,
      const dftfe::uInt                       totalNonLocalElements,
      const dftfe::uInt                       maxSingleAtomSphericalFn,
      const dftfe::uInt                       totalNonLocalEntries,
      const dftfe::utils::deviceFloatComplex *sphericalFnTimesXCellLevel,
      const dftfe::uInt                      *mappingCellLevelToAtomLevel,
      dftfe::utils::deviceFloatComplex       *sphericalFnTimesX)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        totalNonLocalElements * maxSingleAtomSphericalFn * numWfc;
      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::uInt blockIndex      = index / (numWfc);
          const dftfe::uInt intraBlockIndex = index % numWfc;

          const dftfe::uInt toIndex = mappingCellLevelToAtomLevel[blockIndex];
          if (toIndex < totalNonLocalEntries)
            {
              atomicAdd(
                &sphericalFnTimesX[toIndex * numWfc + intraBlockIndex].x,
                sphericalFnTimesXCellLevel[index].x);
              atomicAdd(
                &sphericalFnTimesX[toIndex * numWfc + intraBlockIndex].y,
                sphericalFnTimesXCellLevel[index].y);
            }
        }
    }

  } // namespace

  namespace AtomicCenteredNonLocalOperatorKernelsDevice
  {
    template <typename ValueType>
    void
    sqrtAlphaScalingWaveFunctionEntries(
      const dftfe::uInt maxSingleAtomContribution,
      const dftfe::uInt numWfcs,
      const dftfe::uInt totalAtomsInCurrentProcessor,
      const double     *scalingVector,
      ValueType        *sphericalFnTimesWfcPadded)

    {
      DFTFE_LAUNCH_KERNEL(
        sqrtAlphaScalingWaveFunctionEntriesKernel,
        (numWfcs + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE * totalAtomsInCurrentProcessor *
          maxSingleAtomContribution,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        numWfcs,
        totalAtomsInCurrentProcessor,
        maxSingleAtomContribution,
        dftfe::utils::makeDataTypeDeviceCompatible(scalingVector),
        dftfe::utils::makeDataTypeDeviceCompatible(sphericalFnTimesWfcPadded));
    }



    template <typename ValueType>
    void
    copyFromParallelNonLocalVecToAllCellsVec(
      const dftfe::uInt numWfcs,
      const dftfe::uInt numNonLocalCells,
      const dftfe::uInt maxSingleAtomContribution,
      const ValueType  *sphericalFnTimesWfcParallelVec,
      ValueType        *sphericalFnTimesWfcAllCellsVec,
      const dftfe::Int *indexMapPaddedToParallelVec)
    {
      DFTFE_LAUNCH_KERNEL(copyFromParallelNonLocalVecToAllCellsVecKernel,
                          (numWfcs + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                            dftfe::utils::DEVICE_BLOCK_SIZE * numNonLocalCells *
                            maxSingleAtomContribution,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          0,
                          0,
                          numWfcs,
                          numNonLocalCells,
                          maxSingleAtomContribution,
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            sphericalFnTimesWfcParallelVec),
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            sphericalFnTimesWfcAllCellsVec),
                          indexMapPaddedToParallelVec);
    }
    template <typename ValueType>
    void
    copyToDealiiParallelNonLocalVec(
      const dftfe::uInt  numWfcs,
      const dftfe::uInt  totalEntries,
      const ValueType   *sphericalFnTimesWfcParallelVec,
      ValueType         *sphericalFnTimesWfcDealiiParallelVec,
      const dftfe::uInt *indexMapDealiiParallelNumbering)
    {
      DFTFE_LAUNCH_KERNEL(copyToDealiiParallelNonLocalVecKernel,
                          (numWfcs + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                            dftfe::utils::DEVICE_BLOCK_SIZE * totalEntries,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          0,
                          0,
                          numWfcs,
                          totalEntries,
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            sphericalFnTimesWfcParallelVec),
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            sphericalFnTimesWfcDealiiParallelVec),
                          indexMapDealiiParallelNumbering);
    }

    template <typename ValueType>
    void
    copyToDealiiParallelNonLocalVecFromPaddedVector(
      const dftfe::uInt numWfcs,
      const dftfe::uInt totalEntriesPadded,
      const ValueType  *sphericalFnTimesWfcPaddedVec,
      ValueType        *sphericalFnTimesWfcDealiiParallelVec,
      const dftfe::Int *indexMapDealiiParallelNumbering)
    {
      DFTFE_LAUNCH_KERNEL(copyToDealiiParallelNonLocalVecFromPaddedVecKernel,
                          (numWfcs + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                            dftfe::utils::DEVICE_BLOCK_SIZE *
                            totalEntriesPadded,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          0,
                          0,
                          numWfcs,
                          totalEntriesPadded,
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            sphericalFnTimesWfcPaddedVec),
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            sphericalFnTimesWfcDealiiParallelVec),
                          indexMapDealiiParallelNumbering);
    }
    template <typename ValueType>
    void
    copyFromDealiiParallelNonLocalVecToPaddedVector(
      const dftfe::uInt numWfcs,
      const dftfe::uInt totalEntriesPadded,
      const ValueType  *sphericalFnTimesWfcDealiiParallelVec,
      ValueType        *sphericalFnTimesWfcPaddedVec,
      const dftfe::Int *indexMapDealiiParallelNumbering)
    {
      DFTFE_LAUNCH_KERNEL(copyFromDealiiParallelNonLocalVecToPaddedVecKernel,
                          (numWfcs + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                            dftfe::utils::DEVICE_BLOCK_SIZE *
                            totalEntriesPadded,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          0,
                          0,
                          numWfcs,
                          totalEntriesPadded,
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            sphericalFnTimesWfcDealiiParallelVec),
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            sphericalFnTimesWfcPaddedVec),
                          indexMapDealiiParallelNumbering);
    }

    template <typename ValueType>
    void
    addNonLocalContribution(
      const dftfe::uInt totalNonLocalElements,
      const dftfe::uInt numberWfc,
      const dftfe::uInt numberNodesPerElement,
      const dftfe::utils::MemoryStorage<dftfe::uInt,
                                        dftfe::utils::MemorySpace::DEVICE>
        &iElemNonLocalToElemIndexMap,
      const dftfe::utils::MemoryStorage<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE>
                &nonLocalContribution,
      ValueType *TotalContribution)
    {
      const dftfe::uInt totalEntries =
        totalNonLocalElements * numberWfc * numberNodesPerElement;
      DFTFE_LAUNCH_KERNEL(addNonLocalContributionDeviceKernel,
                          (dftfe::utils::DEVICE_BLOCK_SIZE + totalEntries) /
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          0,
                          0,
                          totalNonLocalElements,
                          numberWfc,
                          numberNodesPerElement,
                          iElemNonLocalToElemIndexMap.begin(),
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            nonLocalContribution.begin()),
                          dftfe::utils::makeDataTypeDeviceCompatible(
                            TotalContribution));
    }



    template <typename ValueType>
    void
    addNonLocalContribution(
      const dftfe::uInt numberCellsForAtom,
      const dftfe::uInt numberNodesPerElement,
      const dftfe::uInt numberWfc,
      const dftfe::uInt numberCellsTraversed,
      const dftfe::utils::MemoryStorage<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE>
                &nonLocalContribution,
      ValueType *TotalContribution,
      const dftfe::utils::MemoryStorage<dftfe::uInt,
                                        dftfe::utils::MemorySpace::DEVICE>
        &cellNodeIdMapNonLocalToLocal)
    {
      DFTFE_LAUNCH_KERNEL(
        addNonLocalContributionDeviceKernel,
        (numberWfc + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE * numberCellsForAtom *
          numberNodesPerElement,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        numberWfc,
        numberCellsForAtom * numberNodesPerElement,
        dftfe::utils::makeDataTypeDeviceCompatible(
          nonLocalContribution.begin() +
          numberCellsTraversed * numberNodesPerElement * numberWfc),
        dftfe::utils::makeDataTypeDeviceCompatible(TotalContribution),
        cellNodeIdMapNonLocalToLocal.begin() +
          numberCellsTraversed * numberNodesPerElement);
    }

    template <typename ValueType>
    void
    assembleAtomLevelContributionsFromCellLevel(
      const dftfe::uInt numberWaveFunctions,
      const dftfe::uInt totalNonlocalElems,
      const dftfe::uInt maxSingleAtomContribution,
      const dftfe::uInt totalNonlocalEntries,
      const dftfe::utils::MemoryStorage<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE>
        &sphericalFnTimesVectorAllCellsDevice,
      const dftfe::utils::MemoryStorage<dftfe::uInt,
                                        dftfe::utils::MemorySpace::DEVICE>
        &mapSphericalFnTimesVectorAllCellsReductionDevice,
      dftfe::utils::MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
        &sphericalFnTimesWavefunctionMatrix)
    {
      const dftfe::uInt totalEntries =
        totalNonlocalElems * numberWaveFunctions * maxSingleAtomContribution;
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      assembleAtomLevelContributionsFromCellLevelKernel<<<
        (dftfe::utils::DEVICE_BLOCK_SIZE + totalEntries) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        numberWaveFunctions,
        totalNonlocalElems,
        maxSingleAtomContribution,
        totalNonlocalEntries,
        dftfe::utils::makeDataTypeDeviceCompatible(
          sphericalFnTimesVectorAllCellsDevice.begin()),
        mapSphericalFnTimesVectorAllCellsReductionDevice.begin(),
        dftfe::utils::makeDataTypeDeviceCompatible(
          sphericalFnTimesWavefunctionMatrix.begin()));
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        assembleAtomLevelContributionsFromCellLevelKernel,
        (dftfe::utils::DEVICE_BLOCK_SIZE + totalEntries) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        numberWaveFunctions,
        totalNonlocalElems,
        maxSingleAtomContribution,
        totalNonlocalEntries,
        dftfe::utils::makeDataTypeDeviceCompatible(
          sphericalFnTimesVectorAllCellsDevice.begin()),
        mapSphericalFnTimesVectorAllCellsReductionDevice.begin(),
        dftfe::utils::makeDataTypeDeviceCompatible(
          sphericalFnTimesWavefunctionMatrix.begin()));
#endif
    }

    template void
    copyToDealiiParallelNonLocalVec(
      const dftfe::uInt        numWfcs,
      const dftfe::uInt        totalEntries,
      const dataTypes::number *sphericalFnTimesWfcParallelVec,
      dataTypes::number       *sphericalFnTimesWfcDealiiParallelVec,
      const dftfe::uInt       *indexMapDealiiParallelNumbering);

    template void
    copyToDealiiParallelNonLocalVec(
      const dftfe::uInt            numWfcs,
      const dftfe::uInt            totalEntries,
      const dataTypes::numberFP32 *sphericalFnTimesWfcParallelVec,
      dataTypes::numberFP32       *sphericalFnTimesWfcDealiiParallelVec,
      const dftfe::uInt           *indexMapDealiiParallelNumbering);

    template void
    copyFromDealiiParallelNonLocalVecToPaddedVector(
      const dftfe::uInt        numWfcs,
      const dftfe::uInt        totalEntriesPadded,
      const dataTypes::number *sphericalFnTimesWfcDealiiParallelVec,
      dataTypes::number       *sphericalFnTimesWfcPaddedVec,
      const dftfe::Int        *indexMapDealiiParallelNumbering);

    template void
    copyToDealiiParallelNonLocalVecFromPaddedVector(
      const dftfe::uInt        numWfcs,
      const dftfe::uInt        totalEntriesPadded,
      const dataTypes::number *sphericalFnTimesWfcPaddedVec,
      dataTypes::number       *sphericalFnTimesWfcDealiiParallelVec,
      const dftfe::Int        *indexMapDealiiParallelNumbering);

    template void
    copyFromDealiiParallelNonLocalVecToPaddedVector(
      const dftfe::uInt            numWfcs,
      const dftfe::uInt            totalEntriesPadded,
      const dataTypes::numberFP32 *sphericalFnTimesWfcDealiiParallelVec,
      dataTypes::numberFP32       *sphericalFnTimesWfcPaddedVec,
      const dftfe::Int            *indexMapDealiiParallelNumbering);

    template void
    copyToDealiiParallelNonLocalVecFromPaddedVector(
      const dftfe::uInt            numWfcs,
      const dftfe::uInt            totalEntriesPadded,
      const dataTypes::numberFP32 *sphericalFnTimesWfcPaddedVec,
      dataTypes::numberFP32       *sphericalFnTimesWfcDealiiParallelVec,
      const dftfe::Int            *indexMapDealiiParallelNumbering);

    template void
    copyFromParallelNonLocalVecToAllCellsVec(
      const dftfe::uInt        numWfcs,
      const dftfe::uInt        numNonLocalCells,
      const dftfe::uInt        maxSingleAtomContribution,
      const dataTypes::number *sphericalFnTimesWfcParallelVec,
      dataTypes::number       *sphericalFnTimesWfcAllCellsVec,
      const dftfe::Int        *indexMapPaddedToParallelVec);


    template void
    copyFromParallelNonLocalVecToAllCellsVec(
      const dftfe::uInt            numWfcs,
      const dftfe::uInt            numNonLocalCells,
      const dftfe::uInt            maxSingleAtomContribution,
      const dataTypes::numberFP32 *sphericalFnTimesWfcParallelVec,
      dataTypes::numberFP32       *sphericalFnTimesWfcAllCellsVec,
      const dftfe::Int            *indexMapPaddedToParallelVec);

    template void
    sqrtAlphaScalingWaveFunctionEntries(
      const dftfe::uInt  maxSingleAtomContribution,
      const dftfe::uInt  numWfcs,
      const dftfe::uInt  totalAtomsInCurrentProcessor,
      const double      *scalingVector,
      dataTypes::number *sphericalFnTimesWfcPadded);

    template void
    sqrtAlphaScalingWaveFunctionEntries(
      const dftfe::uInt      maxSingleAtomContribution,
      const dftfe::uInt      numWfcs,
      const dftfe::uInt      totalAtomsInCurrentProcessor,
      const double          *scalingVector,
      dataTypes::numberFP32 *sphericalFnTimesWfcPadded);


    template void
    addNonLocalContribution(
      const dftfe::uInt numberCellsForAtom,
      const dftfe::uInt numberNodesPerElement,
      const dftfe::uInt numberWfc,
      const dftfe::uInt numberCellsTraversed,
      const dftfe::utils::MemoryStorage<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE>
                        &nonLocalContribution,
      dataTypes::number *TotalContribution,
      const dftfe::utils::MemoryStorage<dftfe::uInt,
                                        dftfe::utils::MemorySpace::DEVICE>
        &cellNodeIdMapNonLocalToLocal);

    template void
    addNonLocalContribution(
      const dftfe::uInt numberCellsForAtom,
      const dftfe::uInt numberNodesPerElement,
      const dftfe::uInt numberWfc,
      const dftfe::uInt numberCellsTraversed,
      const dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                        dftfe::utils::MemorySpace::DEVICE>
                            &nonLocalContribution,
      dataTypes::numberFP32 *TotalContribution,
      const dftfe::utils::MemoryStorage<dftfe::uInt,
                                        dftfe::utils::MemorySpace::DEVICE>
        &cellNodeIdMapNonLocalToLocal);

    template void
    addNonLocalContribution(
      const dftfe::uInt totalNonLocalElements,
      const dftfe::uInt numberWfc,
      const dftfe::uInt numberNodesPerElement,
      const dftfe::utils::MemoryStorage<dftfe::uInt,
                                        dftfe::utils::MemorySpace::DEVICE>
        &iElemNonLocalToElemIndexMap,
      const dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                        dftfe::utils::MemorySpace::DEVICE>
                            &nonLocalContribution,
      dataTypes::numberFP32 *TotalContribution);

    template void
    addNonLocalContribution(
      const dftfe::uInt totalNonLocalElements,
      const dftfe::uInt numberWfc,
      const dftfe::uInt numberNodesPerElement,
      const dftfe::utils::MemoryStorage<dftfe::uInt,
                                        dftfe::utils::MemorySpace::DEVICE>
        &iElemNonLocalToElemIndexMap,
      const dftfe::utils::MemoryStorage<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE>
                        &nonLocalContribution,
      dataTypes::number *TotalContribution);
    template void
    assembleAtomLevelContributionsFromCellLevel(
      const dftfe::uInt numberWaveFunctions,
      const dftfe::uInt totalNonlocalElems,
      const dftfe::uInt maxSingleAtomContribution,
      const dftfe::uInt totalNonlocalEntries,
      const dftfe::utils::MemoryStorage<dataTypes::number,
                                        dftfe::utils::MemorySpace::DEVICE>
        &sphericalFnTimesVectorAllCellsDevice,
      const dftfe::utils::MemoryStorage<dftfe::uInt,
                                        dftfe::utils::MemorySpace::DEVICE>
        &mapSphericalFnTimesVectorAllCellsReductionDevice,
      dftfe::utils::MemoryStorage<dataTypes::number,
                                  dftfe::utils::MemorySpace::DEVICE>
        &sphericalFnTimesWavefunctionMatrix);

    template void
    assembleAtomLevelContributionsFromCellLevel(
      const dftfe::uInt numberWaveFunctions,
      const dftfe::uInt totalNonlocalElems,
      const dftfe::uInt maxSingleAtomContribution,
      const dftfe::uInt totalNonlocalEntries,
      const dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                        dftfe::utils::MemorySpace::DEVICE>
        &sphericalFnTimesVectorAllCellsDevice,
      const dftfe::utils::MemoryStorage<dftfe::uInt,
                                        dftfe::utils::MemorySpace::DEVICE>
        &mapSphericalFnTimesVectorAllCellsReductionDevice,
      dftfe::utils::MemoryStorage<dataTypes::numberFP32,
                                  dftfe::utils::MemorySpace::DEVICE>
        &sphericalFnTimesWavefunctionMatrix);
  } // namespace AtomicCenteredNonLocalOperatorKernelsDevice

} // namespace dftfe
