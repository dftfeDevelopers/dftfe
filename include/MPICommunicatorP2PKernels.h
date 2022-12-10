// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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

/*
 * @author Sambit Das
 */

#ifndef dftfeMPICommunicatorP2PKernels_h
#define dftfeMPICommunicatorP2PKernels_h

#include <MemorySpaceType.h>
#include <MemoryStorage.h>
#include <TypeConfig.h>
#include <DataTypeOverloads.h>

namespace dftfe
{
  namespace utils
  {
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    class MPICommunicatorP2PKernels
    {
    public:
      using SizeTypeVector = utils::MemoryStorage<size_type, memorySpace>;

      /**
       * @brief Function template for architecture adaptable gather kernel to send buffer
       * @tparam ValueType the type of the number
       * @tparam memorySpace
       * @param[in] dataArray data array with locally owned entries
       * @param[in] ownedLocalIndicesForTargetProcs
       * @param[in] blockSize
       * @param[out] sendBuffer
       */
      static void
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<ValueType, memorySpace> &dataArray,
        const SizeTypeVector &                 ownedLocalIndicesForTargetProcs,
        const size_type                        blockSize,
        MemoryStorage<ValueType, memorySpace> &sendBuffer);

      /**
       * @brief Function template for architecture adaptable accumlate kernel from recv buffer
       * @tparam ValueType the type of the number
       * @tparam memorySpace
       * @param[in] recvBuffer
       * @param[in] ownedLocalIndicesForTargetProcs
       * @param[in] blockSize
       * @param[out] dataArray
       */
      static void
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<ValueType, memorySpace> &recvBuffer,
        const SizeTypeVector &                 ownedLocalIndicesForTargetProcs,
        const size_type                        blockSize,
        const size_type                        locallyOwnedSize,
               const size_type                        ghostSize,         
        MemoryStorage<double, memorySpace> &tempDoubleRealDataArray,
        MemoryStorage<double, memorySpace> &tempDoubleImagDataArray,
        MemoryStorage<float, memorySpace> &tempFloatRealDataArray,
        MemoryStorage<float, memorySpace> &tempFloatImagDataArray,
        MemoryStorage<ValueType, memorySpace> &dataArray);
    };

#ifdef DFTFE_WITH_DEVICE
    template <typename ValueType>
    class MPICommunicatorP2PKernels<ValueType,
                                    dftfe::utils::MemorySpace::DEVICE>
    {
    public:
      static void
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
          &dataArray,
        const MemoryStorage<size_type, dftfe::utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
          &sendBuffer);

      static void
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
          &recvBuffer,
        const MemoryStorage<size_type, dftfe::utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        const size_type                        locallyOwnedSize,
               const size_type                        ghostSize, 
        MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> &tempDoubleRealDataArray,
        MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> &tempDoubleImagDataArray,     MemoryStorage<float, dftfe::utils::MemorySpace::DEVICE> &tempFloatRealDataArray,
        MemoryStorage<float, dftfe::utils::MemorySpace::DEVICE> &tempFloatImagDataArray,
        MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
          &dataArray);
    };
#endif
  } // namespace utils
} // namespace dftfe


#endif
