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
// @author Kartick Ramakrishnan
//
#ifndef DFTFE_ATOMICCENTEREDNONLOCALOPERATORDEVICEKERNELS_H
#define DFTFE_ATOMICCENTEREDNONLOCALOPERATORDEVICEKERNELS_H

#include <deviceKernelsGeneric.h>
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceTypeConfig.h>
#include <DeviceKernelLauncherConstants.h>

namespace dftfe
{
  namespace AtomicCenteredNonLocalOperatorKernelsDevice
  {
    template <typename ValueType>
    void
    copyFromParallelNonLocalVecToAllCellsVec(
      const unsigned int numWfcs,
      const unsigned int numNonLocalCells,
      const unsigned int maxSingleAtomPseudoWfc,
      const ValueType *  sphericalFnTimesWfcParallelVec,
      ValueType *        sphericalFnTimesWfcAllCellsVec,
      const int *        indexMapPaddedToParallelVec);


    template <typename ValueType>
    void
    copyToDealiiParallelNonLocalVec(
      const unsigned int  numWfcs,
      const unsigned int  totalEntries,
      const ValueType *   sphericalFnTimesWfcParallelVec,
      ValueType *         sphericalFnTimesWfcDealiiParallelVec,
      const unsigned int *indexMapDealiiParallelNumbering);

    template <typename ValueType>
    void
    copyFromDealiiParallelNonLocalVecToPaddedVector(
      const unsigned int numWfcs,
      const unsigned int totalEntriesPadded,
      const ValueType *  sphericalFnTimesWfcDealiiParallelVec,
      ValueType *        sphericalFnTimesWfcPaddedVec,
      const int *        indexMapDealiiParallelNumbering);

    template <typename ValueType>
    void
    copyToDealiiParallelNonLocalVecFromPaddedVector(
      const unsigned int numWfcs,
      const unsigned int totalEntriesPadded,
      const ValueType *  sphericalFnTimesWfcPaddedVec,
      ValueType *        sphericalFnTimesWfcDealiiParallelVec,
      const int *        indexMapDealiiParallelNumbering);

    template <typename ValueType>
    void
    addNonLocalContribution(
      const unsigned int numberCellsForAtom,
      const unsigned int numberNodesPerElement,
      const unsigned int numberWfc,
      const unsigned int numberCellsTraversed,
      const dftfe::utils::MemoryStorage<ValueType,
                                        dftfe::utils::MemorySpace::DEVICE>
        &        nonLocalContribution,
      ValueType *TotalContribution,
      const dftfe::utils::MemoryStorage<unsigned int,
                                        dftfe::utils::MemorySpace::DEVICE>
        &cellNodeIdMapNonLocalToLocal);

    template <typename ValueType>
    void
    sqrtAlphaScalingWaveFunctionEntries(
      const unsigned int maxSingleAtomContribution,
      const unsigned int numWfcs,
      const unsigned int totalAtomsInCurrentProcessor,
      const double *     scalingVector,
      ValueType *        sphericalFnTimesWfcPadded);


  } // namespace AtomicCenteredNonLocalOperatorKernelsDevice



} // namespace dftfe

#endif // DFTFE_ATOMICCENTEREDNONLOCALOPERATORDEVICEKERNELS_H
