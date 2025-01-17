// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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
#ifdef DFTFE_WITH_DEVICE
#  include <DeviceTypeConfig.h>
#endif
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
      template <typename ValueTypeComm>
      static void
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<ValueType, memorySpace> &dataArray,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        MemoryStorage<ValueTypeComm, memorySpace> &sendBuffer);

      /**
       * @brief Function template for architecture adaptable accumulate add kernel from recv buffer
       * @tparam ValueType the type of the number
       * @tparam memorySpace
       * @param[in] recvBuffer
       * @param[in] ownedLocalIndicesForTargetProcs
       * @param[in] blockSize
       * @param[out] dataArray
       */
      template <typename ValueTypeComm>
      static void
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<ValueTypeComm, memorySpace> &recvBuffer,
        const SizeTypeVector &                 ownedLocalIndicesForTargetProcs,
        const size_type                        blockSize,
        const size_type                        locallyOwnedSize,
        const size_type                        ghostSize,
        MemoryStorage<ValueType, memorySpace> &dataArray);

      /**
       * @brief Function template for architecture adaptable accumulate insert kernel from recv buffer
       * @tparam ValueType the type of the number
       * @tparam memorySpace
       * @param[in] recvBuffer
       * @param[in] ownedLocalIndicesForTargetProcs
       * @param[in] blockSize
       * @param[out] dataArray
       */
      template <typename ValueTypeComm>
      static void
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<ValueTypeComm, memorySpace> &recvBuffer,
        const SizeTypeVector &                 ownedLocalIndicesForTargetProcs,
        const size_type                        blockSize,
        const size_type                        locallyOwnedSize,
        const size_type                        ghostSize,
        MemoryStorage<ValueType, memorySpace> &dataArray);

      /**
       * @brief Function template for copying type1 to type2
       * @param[in] blockSize
       * @param[in] type1Array
       * @param[out] type2Array
       */
      template <typename ValueType1, typename ValueType2>
      static void
      copyValueType1ArrToValueType2Arr(const size_type   blockSize,
                                       const ValueType1 *type1Array,
                                       ValueType2 *      type2Array);
    };

#ifdef DFTFE_WITH_DEVICE
    template <typename ValueType>
    class MPICommunicatorP2PKernels<ValueType,
                                    dftfe::utils::MemorySpace::DEVICE>
    {
    public:
      template <typename ValueTypeComm>
      static void
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE>
          &dataArray,
        const MemoryStorage<size_type, dftfe::utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        MemoryStorage<ValueTypeComm, dftfe::utils::MemorySpace::DEVICE>
          &                          sendBuffer,
        dftfe::utils::deviceStream_t deviceCommStream);

      template <typename ValueTypeComm>
      static void
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<ValueTypeComm, dftfe::utils::MemorySpace::DEVICE>
          &recvBuffer,
        const MemoryStorage<size_type, dftfe::utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        const size_type locallyOwnedSize,
        const size_type ghostSize,
        MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE> &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

      /**
       * @brief Function template for architecture adaptable accumulate insert kernel from recv buffer
       * @tparam ValueType the type of the number
       * @tparam memorySpace
       * @param[in] recvBuffer
       * @param[in] ownedLocalIndicesForTargetProcs
       * @param[in] blockSize
       * @param[out] dataArray
       */
      template <typename ValueTypeComm>
      static void
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<ValueTypeComm, dftfe::utils::MemorySpace::DEVICE>
          &recvBuffer,
        const MemoryStorage<size_type, dftfe::utils::MemorySpace::DEVICE>
          &             ownedLocalIndicesForTargetProcs,
        const size_type blockSize,
        const size_type locallyOwnedSize,
        const size_type ghostSize,
        MemoryStorage<ValueType, dftfe::utils::MemorySpace::DEVICE> &dataArray,
        dftfe::utils::deviceStream_t deviceCommStream);

      /**
       * @brief Function template for copying type1 to type2
       * @param[in] blockSize
       * @param[in] type1Array
       * @param[out] type2Array
       */
      template <typename ValueType1, typename ValueType2>
      static void
      copyValueType1ArrToValueType2Arr(
        const size_type              blockSize,
        const ValueType1 *           type1Array,
        ValueType2 *                 type2Array,
        dftfe::utils::deviceStream_t deviceCommStream);
    };
#endif
  } // namespace utils
} // namespace dftfe


#endif
