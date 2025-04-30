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
#ifndef DFTFE_ATOMICCENTEREDNONLOCALOPERATORDEVICEKERNELS_H
#define DFTFE_ATOMICCENTEREDNONLOCALOPERATORDEVICEKERNELS_H

#include <deviceKernelsGeneric.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherConstants.h>
#include <MemoryStorage.h>
namespace dftfe
{
  namespace AtomicCenteredNonLocalOperatorKernelsDevice
  {
    template <typename ValueType>
    void
    copyFromParallelNonLocalVecToAllCellsVec(
      const dftfe::uInt numWfcs,
      const dftfe::uInt numNonLocalCells,
      const dftfe::uInt maxSingleAtomPseudoWfc,
      const ValueType  *sphericalFnTimesWfcParallelVec,
      ValueType        *sphericalFnTimesWfcAllCellsVec,
      const dftfe::Int *indexMapPaddedToParallelVec);


    template <typename ValueType>
    void
    copyToDealiiParallelNonLocalVec(
      const dftfe::uInt  numWfcs,
      const dftfe::uInt  totalEntries,
      const ValueType   *sphericalFnTimesWfcParallelVec,
      ValueType         *sphericalFnTimesWfcDealiiParallelVec,
      const dftfe::uInt *indexMapDealiiParallelNumbering);

    template <typename ValueType>
    void
    copyFromDealiiParallelNonLocalVecToPaddedVector(
      const dftfe::uInt numWfcs,
      const dftfe::uInt totalEntriesPadded,
      const ValueType  *sphericalFnTimesWfcDealiiParallelVec,
      ValueType        *sphericalFnTimesWfcPaddedVec,
      const dftfe::Int *indexMapDealiiParallelNumbering);

    template <typename ValueType>
    void
    copyToDealiiParallelNonLocalVecFromPaddedVector(
      const dftfe::uInt numWfcs,
      const dftfe::uInt totalEntriesPadded,
      const ValueType  *sphericalFnTimesWfcPaddedVec,
      ValueType        *sphericalFnTimesWfcDealiiParallelVec,
      const dftfe::Int *indexMapDealiiParallelNumbering);

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
        &cellNodeIdMapNonLocalToLocal);
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
      ValueType *TotalContribution);

    template <typename ValueType>
    void
    sqrtAlphaScalingWaveFunctionEntries(
      const dftfe::uInt maxSingleAtomContribution,
      const dftfe::uInt numWfcs,
      const dftfe::uInt totalAtomsInCurrentProcessor,
      const double     *scalingVector,
      ValueType        *sphericalFnTimesWfcPadded);

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
        &sphericalFnTimesWavefunctionMatrix);


  } // namespace AtomicCenteredNonLocalOperatorKernelsDevice



} // namespace dftfe

#endif // DFTFE_ATOMICCENTEREDNONLOCALOPERATORDEVICEKERNELS_H
