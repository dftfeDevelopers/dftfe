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
 * @author Sambit Das.
 */

#include <dftfe/config.h>
#include <dftfe/MPICommunicatorP2P.h>
#include <dftfe/MPICommunicatorP2PKernels.h>
#include <dftfe/MPITags.h>
#include <dftfe/Exceptions.h>
#include <dftfe/DeviceAPICalls.h>
#include <dftfe/deviceDirectCCLWrapper.h>
namespace dftfe
{
  namespace utils
  {
    namespace mpi
    {
      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      MPICommunicatorP2P<ValueType, memorySpace>::MPICommunicatorP2P(
        std::shared_ptr<const MPIPatternP2P<memorySpace>> mpiPatternP2P,
        const dftfe::uInt                                 blockSize)
        : d_mpiPatternP2P(mpiPatternP2P)
        , d_blockSize(blockSize)
        , d_locallyOwnedSize(mpiPatternP2P->localOwnedSize())
        , d_ghostSize(mpiPatternP2P->localGhostSize())
        , d_commPrecision(communicationPrecision::standard)
        , d_updateGhostValuesInFlight(false)
        , d_accumulateAddLocallyOwnedInFlight(false)
        , d_accumulateInsertLocallyOwnedInFlight(false)
      {
        d_commProtocol = communicationProtocol::mpiHost;
#if defined(DFTFE_WITH_DEVICE) && defined(DFTFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          d_commProtocol = communicationProtocol::mpiDevice;
#endif
#if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
        if (memorySpace == MemorySpace::DEVICE &&
            dftfe::utils::DeviceCCLWrapper::dcclCommInit)
          d_commProtocol = communicationProtocol::dccl;
#endif

        d_mpiCommunicator = d_mpiPatternP2P->mpiCommunicator();
        d_sendRecvBuffer.resize(
          d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size() *
            blockSize,
          0.0);

        d_requestsUpdateGhostValues.resize(
          d_mpiPatternP2P->getGhostProcIds().size() +
          d_mpiPatternP2P->getTargetProcIds().size(),
		      MPI_REQUEST_NULL);
        d_requestsAccumulateAddLocallyOwned.resize(
          d_mpiPatternP2P->getGhostProcIds().size() +
          d_mpiPatternP2P->getTargetProcIds().size(),
          MPI_REQUEST_NULL);

        d_requestsAccumulateInsertLocallyOwned.resize(
          d_mpiPatternP2P->getGhostProcIds().size() +
          d_mpiPatternP2P->getTargetProcIds().size(),
          MPI_REQUEST_NULL);

#ifdef DFTFE_WITH_DEVICE

        if constexpr (memorySpace == MemorySpace::DEVICE)
          if (d_commProtocol == communicationProtocol::mpiHost)
            {
              d_ghostDataCopyHostPinnedPtr = std::make_shared<
                MemoryStorage<ValueType, MemorySpace::HOST_PINNED>>(
                d_mpiPatternP2P->localGhostSize() * blockSize, 0.0);

              d_sendRecvBufferHostPinnedPtr = std::make_shared<
                MemoryStorage<ValueType, MemorySpace::HOST_PINNED>>(
                d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size() *
                  blockSize,
                0.0);

              d_ghostDataCopySinglePrecHostPinnedPtr =
                std::make_shared<MemoryStorage<
                  typename dftfe::dataTypes::singlePrecType<ValueType>::type,
                  MemorySpace::HOST_PINNED>>(d_mpiPatternP2P->localGhostSize() *
                                               d_blockSize,
                                             0.0);

              d_sendRecvBufferSinglePrecHostPinnedPtr =
                std::make_shared<MemoryStorage<
                  typename dftfe::dataTypes::singlePrecType<ValueType>::type,
                  MemorySpace::HOST_PINNED>>(
                  d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size() *
                    d_blockSize,
                  0.0);

              d_ghostDataCopyHalfPrecHostPinnedPtr =
                std::make_shared<MemoryStorage<
                  typename dftfe::dataTypes::halfPrecType<ValueType>::type,
                  MemorySpace::HOST_PINNED>>(d_mpiPatternP2P->localGhostSize() *
                                               d_blockSize,
                                             0.0);

              d_sendRecvBufferHalfPrecHostPinnedPtr =
                std::make_shared<MemoryStorage<
                  typename dftfe::dataTypes::halfPrecType<ValueType>::type,
                  MemorySpace::HOST_PINNED>>(
                  d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size() *
                    d_blockSize,
                  0.0);

              // Allocate pinned buffers at max BPV so setCommunicationPrecision
              // never reallocates.
              d_compressBitsPerValue = 16;
              // Exact bytes: blockSize is a multiple of 4 and all supported
              // BPVs are even.
              d_compressedTargetBytes =
                (d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size() *
                 d_blockSize * d_compressBitsPerValue) /
                8;

              d_compressedGhostBytes = (d_mpiPatternP2P->localGhostSize() *
                                        d_blockSize * d_compressBitsPerValue) /
                                       8;

              d_ghostDataCopyCompressHostPinnedPtr =
                std::make_shared<MemoryStorage<
                  typename dftfe::dataTypes::compressType<ValueType>::type,
                  MemorySpace::HOST_PINNED>>(d_compressedGhostBytes, 0);

              d_sendRecvBufferCompressHostPinnedPtr =
                std::make_shared<MemoryStorage<
                  typename dftfe::dataTypes::compressType<ValueType>::type,
                  MemorySpace::HOST_PINNED>>(d_compressedTargetBytes, 0);
            }
#endif
      }


      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::reclaimPendingRequests(
        bool                     &inFlight,
        std::vector<MPI_Request> &requests,
        const std::string        &opName)
      {
        if (!inFlight)
          return;

        inFlight = false;

        std::string errMsg =
          "An MPICommunicatorP2P was destroyed with an outstanding " + opName +
          " operation: " + opName + "Begin() was called without a matching " +
          opName +
          "End(). An unmatched Begin() leaks its MPI_Request objects, which "
          "are drawn from a finite pool inside the MPI implementation.";

        // Nothing can be done through MPI once it has been torn down.
        int mpiFinalized = 0;
        MPI_Finalized(&mpiFinalized);
        if (mpiFinalized == 0)
          {
            int rank = -1;
            MPI_Comm_rank(d_mpiCommunicator, &rank);
            errMsg = "Rank " + std::to_string(rank) + ": " + errMsg;

            // complete the requests before reporting, so that they are handed
            // back to the MPI implementation rather than leaked
            if (requests.size() > 0)
              MPI_Waitall(requests.size(),
                          requests.data(),
                          MPI_STATUSES_IGNORE);
          }
        else
          errMsg += " MPI was already finalized, so the pending requests could "
                    "not be completed.";

        // Reported and not thrown: a destructor is implicitly noexcept, and it
        // is routinely run while another exception is unwinding the stack,
        // where throwing would call std::terminate and discard the original
        // error. The unmatched Begin()/End() is caught by the checks in those
        // functions; this is only the backstop for an object that is destroyed
        // between the two.
        std::cerr << "[dftfe] " << errMsg << std::endl;
      }

      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      MPICommunicatorP2P<ValueType, memorySpace>::~MPICommunicatorP2P()
      {
        // Note: completing the requests here can block if the peer never posts
        // the matching operation. That is preferable to the alternatives:
        // MPI_Request_free on an active receive would leave MPI writing into
        // buffers that are about to be destroyed, and abandoning the requests
        // exhausts the MPI implementation's request pool over time.
        reclaimPendingRequests(d_updateGhostValuesInFlight,
                               d_requestsUpdateGhostValues,
                               "updateGhostValues");
        reclaimPendingRequests(d_accumulateAddLocallyOwnedInFlight,
                               d_requestsAccumulateAddLocallyOwned,
                               "accumulateAddLocallyOwned");
        reclaimPendingRequests(d_accumulateInsertLocallyOwnedInFlight,
                               d_requestsAccumulateInsertLocallyOwned,
                               "accumulateInsertLocallyOwned");
       }


      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::setCompressBitsPerValue(
        dftfe::uInt bpv)
      {
        d_compressBitsPerValue = bpv;
      }

      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::setCommunicationPrecision(
        communicationPrecision precision)
      {
        // Prevents explicit reduction of precision to FP32 or BF16 when running
        // on CPUs
        if constexpr (memorySpace == MemorySpace::HOST)
          return;
        if (d_commPrecision == precision)
          return;
        d_commPrecision = precision;
        if (precision == communicationPrecision::standard)
          {
            if (d_sendRecvBuffer.size() !=
                d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size() *
                  d_blockSize)
              d_sendRecvBuffer.resize(
                d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size() *
                  d_blockSize,
                0.0);

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::mpiHost)
                {
                  if (d_ghostDataCopyHostPinnedPtr->size() !=
                      d_mpiPatternP2P->localGhostSize() * d_blockSize)
                    d_ghostDataCopyHostPinnedPtr = std::make_shared<
                      MemoryStorage<ValueType, MemorySpace::HOST_PINNED>>(
                      d_mpiPatternP2P->localGhostSize() * d_blockSize, 0.0);

                  if (d_sendRecvBufferHostPinnedPtr->size() !=
                      d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs()
                          .size() *
                        d_blockSize)
                    d_sendRecvBufferHostPinnedPtr = std::make_shared<
                      MemoryStorage<ValueType, MemorySpace::HOST_PINNED>>(
                      d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs()
                          .size() *
                        d_blockSize,
                      0.0);
                }
#endif
          }
        else if (precision == communicationPrecision::single)
          {
            if (d_sendRecvBufferSinglePrec.size() !=
                d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size() *
                  d_blockSize)
              d_sendRecvBufferSinglePrec.resize(
                d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size() *
                  d_blockSize,
                0.0);
            if (d_ghostDataCopySinglePrec.size() !=
                d_mpiPatternP2P->localGhostSize() * d_blockSize)
              d_ghostDataCopySinglePrec.resize(
                d_mpiPatternP2P->localGhostSize() * d_blockSize, 0.0);
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::mpiHost)
                {
                  if (!d_ghostDataCopySinglePrecHostPinnedPtr)
                    d_ghostDataCopySinglePrecHostPinnedPtr = std::make_shared<
                      MemoryStorage<typename dftfe::dataTypes::singlePrecType<
                                      ValueType>::type,
                                    MemorySpace::HOST_PINNED>>(
                      d_mpiPatternP2P->localGhostSize() * d_blockSize, 0.0);

                  if (!d_sendRecvBufferSinglePrecHostPinnedPtr)
                    d_sendRecvBufferSinglePrecHostPinnedPtr = std::make_shared<
                      MemoryStorage<typename dftfe::dataTypes::singlePrecType<
                                      ValueType>::type,
                                    MemorySpace::HOST_PINNED>>(
                      d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs()
                          .size() *
                        d_blockSize,
                      0.0);

                  if (d_ghostDataCopySinglePrecHostPinnedPtr->size() !=
                      d_mpiPatternP2P->localGhostSize() * d_blockSize)
                    d_ghostDataCopySinglePrecHostPinnedPtr = std::make_shared<
                      MemoryStorage<typename dftfe::dataTypes::singlePrecType<
                                      ValueType>::type,
                                    MemorySpace::HOST_PINNED>>(
                      d_mpiPatternP2P->localGhostSize() * d_blockSize, 0.0);

                  if (d_sendRecvBufferSinglePrecHostPinnedPtr->size() !=
                      d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs()
                          .size() *
                        d_blockSize)
                    d_sendRecvBufferSinglePrecHostPinnedPtr = std::make_shared<
                      MemoryStorage<typename dftfe::dataTypes::singlePrecType<
                                      ValueType>::type,
                                    MemorySpace::HOST_PINNED>>(
                      d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs()
                          .size() *
                        d_blockSize,
                      0.0);
                }
#endif
          }

        else if (precision == communicationPrecision::half)
          {
            if (d_sendRecvBufferHalfPrec.size() !=
                d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size() *
                  d_blockSize)
              d_sendRecvBufferHalfPrec.resize(
                d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size() *
                  d_blockSize,
                0.0);
            if (d_ghostDataCopyHalfPrec.size() !=
                d_mpiPatternP2P->localGhostSize() * d_blockSize)
              d_ghostDataCopyHalfPrec.resize(d_mpiPatternP2P->localGhostSize() *
                                               d_blockSize,
                                             0.0);
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::mpiHost)
                {
                  if (!d_ghostDataCopyHalfPrecHostPinnedPtr)
                    d_ghostDataCopyHalfPrecHostPinnedPtr = std::make_shared<
                      MemoryStorage<typename dftfe::dataTypes::halfPrecType<
                                      ValueType>::type,
                                    MemorySpace::HOST_PINNED>>(
                      d_mpiPatternP2P->localGhostSize() * d_blockSize, 0.0);

                  if (!d_sendRecvBufferHalfPrecHostPinnedPtr)
                    d_sendRecvBufferHalfPrecHostPinnedPtr = std::make_shared<
                      MemoryStorage<typename dftfe::dataTypes::halfPrecType<
                                      ValueType>::type,
                                    MemorySpace::HOST_PINNED>>(
                      d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs()
                          .size() *
                        d_blockSize,
                      0.0);

                  if (d_ghostDataCopyHalfPrecHostPinnedPtr->size() !=
                      d_mpiPatternP2P->localGhostSize() * d_blockSize)
                    d_ghostDataCopyHalfPrecHostPinnedPtr = std::make_shared<
                      MemoryStorage<typename dftfe::dataTypes::halfPrecType<
                                      ValueType>::type,
                                    MemorySpace::HOST_PINNED>>(
                      d_mpiPatternP2P->localGhostSize() * d_blockSize, 0.0);

                  if (d_sendRecvBufferHalfPrecHostPinnedPtr->size() !=
                      d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs()
                          .size() *
                        d_blockSize)
                    d_sendRecvBufferHalfPrecHostPinnedPtr = std::make_shared<
                      MemoryStorage<typename dftfe::dataTypes::halfPrecType<
                                      ValueType>::type,
                                    MemorySpace::HOST_PINNED>>(
                      d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs()
                          .size() *
                        d_blockSize,
                      0.0);
                }
#endif
          }

        else if (precision == communicationPrecision::compress)
          {
            d_compressedTargetBytes =
              (d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size() *
               d_blockSize * d_compressBitsPerValue) /
              8;

            d_compressedGhostBytes = (d_mpiPatternP2P->localGhostSize() *
                                      d_blockSize * d_compressBitsPerValue) /
                                     8;

            if (d_sendRecvBufferCompress.size() != d_compressedTargetBytes)
              d_sendRecvBufferCompress.resize(d_compressedTargetBytes, 0);

            if (d_ghostDataCopyCompress.size() != d_compressedGhostBytes)
              d_ghostDataCopyCompress.resize(d_compressedGhostBytes, 0);

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::mpiHost)
                {
                  if (!d_ghostDataCopyCompressHostPinnedPtr)
                    d_ghostDataCopyCompressHostPinnedPtr = std::make_shared<
                      MemoryStorage<typename dftfe::dataTypes::compressType<
                                      ValueType>::type,
                                    MemorySpace::HOST_PINNED>>(
                      d_compressedGhostBytes, 0);

                  if (!d_sendRecvBufferCompressHostPinnedPtr)
                    d_sendRecvBufferCompressHostPinnedPtr = std::make_shared<
                      MemoryStorage<typename dftfe::dataTypes::compressType<
                                      ValueType>::type,
                                    MemorySpace::HOST_PINNED>>(
                      d_compressedTargetBytes, 0);

                  // Pre-allocated at max 16 bpv; reuse if active bytes fit.
                  if (d_ghostDataCopyCompressHostPinnedPtr->size() <
                      d_compressedGhostBytes)
                    d_ghostDataCopyCompressHostPinnedPtr = std::make_shared<
                      MemoryStorage<typename dftfe::dataTypes::compressType<
                                      ValueType>::type,
                                    MemorySpace::HOST_PINNED>>(
                      d_compressedGhostBytes, 0);

                  if (d_sendRecvBufferCompressHostPinnedPtr->size() <
                      d_compressedTargetBytes)
                    d_sendRecvBufferCompressHostPinnedPtr = std::make_shared<
                      MemoryStorage<typename dftfe::dataTypes::compressType<
                                      ValueType>::type,
                                    MemorySpace::HOST_PINNED>>(
                      d_compressedTargetBytes, 0);
                }
#endif
          }
      }

      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::updateGhostValues(
        MemoryStorage<ValueType, memorySpace> &dataArray,
        const dftfe::uInt                      communicationChannel)
      {
        updateGhostValuesBegin(dataArray, communicationChannel);
        updateGhostValuesEnd(dataArray);
      }

      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::updateGhostValuesBegin(
        MemoryStorage<ValueType, memorySpace> &dataArray,
        const dftfe::uInt                      communicationChannel)
      {
        throwException<LogicError>(
          !d_updateGhostValuesInFlight,
          "updateGhostValuesBegin() was called on an "
          "MPICommunicatorP2P that already has an outstanding "
          "updateGhostValues operation. The two calls share the "
          "same set of MPI_Request handles, so the requests of the earlier "
          "call would be overwritten and leaked. Call "
          "updateGhostValuesEnd() before starting the next one.");
        
		    d_updateGhostValuesInFlight = true;
        
				// initiate non-blocking receives from ghost processors
        if (d_commPrecision == communicationPrecision::standard)
          {
            ValueType *recvArrayStartPtr =
              dataArray.data() +
              d_mpiPatternP2P->localOwnedSize() * d_blockSize;

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_commProtocol == communicationProtocol::mpiHost)
                  recvArrayStartPtr = d_ghostDataCopyHostPinnedPtr->begin();
                dftfe::utils::deviceSynchronize();
              }
#endif
            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getGhostProcIds()).size();
                   ++i)
                {
                  const dftfe::Int err =
                    MPI_Irecv(recvArrayStartPtr,
                              (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                                 .data()[2 * i + 1] -
                               d_mpiPatternP2P->getGhostLocalIndicesRanges()
                                 .data()[2 * i]) *
                                d_blockSize * sizeof(ValueType),
                              MPI_BYTE,
                              d_mpiPatternP2P->getGhostProcIds().data()[i],
                              static_cast<dftfe::uInt>(
                                MPITags::MPI_P2P_COMMUNICATOR_SCATTER_TAG) +
                                communicationChannel,
                              d_mpiCommunicator,
                              &d_requestsUpdateGhostValues[i]);

                  std::string errMsg = "Error occured while using MPI_Irecv. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);

                  recvArrayStartPtr +=
                    (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i + 1] -
                     d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i]) *
                    d_blockSize;
                }

            // gather locally owned entries into a contiguous send buffer
            if ((d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size()) >
                0)
#ifdef DFTFE_WITH_DEVICE
              if constexpr (memorySpace == MemorySpace::DEVICE)
                MPICommunicatorP2PKernels<ValueType, memorySpace>::
                  gatherLocallyOwnedEntriesSendBufferToTargetProcs(
                    dataArray,
                    d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
                    d_blockSize,
                    d_sendRecvBuffer,
                    dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
              else
#endif
                MPICommunicatorP2PKernels<ValueType, memorySpace>::
                  gatherLocallyOwnedEntriesSendBufferToTargetProcs(
                    dataArray,
                    d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
                    d_blockSize,
                    d_sendRecvBuffer);

            // initiate non-blocking sends to target processors
            ValueType *sendArrayStartPtr = d_sendRecvBuffer.data();

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_commProtocol != communicationProtocol::dccl)
                  dftfe::utils::deviceStreamSynchronize(
                    dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
                if (d_commProtocol == communicationProtocol::mpiHost)
                  {
                    MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
                      memoryTransfer;

                    if (d_sendRecvBufferHostPinnedPtr->size() > 0)
                      memoryTransfer.copy(
                        d_sendRecvBufferHostPinnedPtr->size(),
                        d_sendRecvBufferHostPinnedPtr->begin(),
                        d_sendRecvBuffer.begin());

                    sendArrayStartPtr = d_sendRecvBufferHostPinnedPtr->begin();
                  }
              }
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::dccl)
                {
                  NCCLCHECK(ncclGroupStart());
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getTargetProcIds()).size();
                       ++i)
                    {
                      if (d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                            .data()[i] > 0)
                        NCCLCHECK(ncclSend(
                          reinterpret_cast<float *>(sendArrayStartPtr),
                          d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                              .data()[i] *
                            d_blockSize * (sizeof(ValueType) / 4),
                          ncclFloat,
                          d_mpiPatternP2P->getTargetProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      sendArrayStartPtr +=
                        d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                          .data()[i] *
                        d_blockSize;
                    }
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getGhostProcIds()).size();
                       ++i)
                    {
                      if ((d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) > 0)
                        NCCLCHECK(ncclRecv(
                          reinterpret_cast<float *>(recvArrayStartPtr),
                          (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) *
                            d_blockSize * (sizeof(ValueType) / 4),
                          ncclFloat,
                          d_mpiPatternP2P->getGhostProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      recvArrayStartPtr +=
                        (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i + 1] -
                         d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i]) *
                        d_blockSize;
                    }
                  NCCLCHECK(ncclGroupEnd());
                }
#  endif
#endif
            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getTargetProcIds()).size();
                   ++i)
                {
                  const dftfe::Int err = MPI_Isend(
                    sendArrayStartPtr,
                    d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                        .data()[i] *
                      d_blockSize * sizeof(ValueType),
                    MPI_BYTE,
                    d_mpiPatternP2P->getTargetProcIds().data()[i],
                    static_cast<dftfe::uInt>(
                      MPITags::MPI_P2P_COMMUNICATOR_SCATTER_TAG) +
                      communicationChannel,

                    d_mpiCommunicator,
                    &d_requestsUpdateGhostValues
                      [d_mpiPatternP2P->getGhostProcIds().size() + i]);

                  std::string errMsg = "Error occured while using MPI_Isend. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);

                  sendArrayStartPtr +=
                    d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                      .data()[i] *
                    d_blockSize;
                }
          }
        else if (d_commPrecision == communicationPrecision::single)
          {
            typename dftfe::dataTypes::singlePrecType<ValueType>::type
              *recvArrayStartPtr = d_ghostDataCopySinglePrec.data();

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_commProtocol == communicationProtocol::mpiHost)
                  recvArrayStartPtr =
                    d_ghostDataCopySinglePrecHostPinnedPtr->begin();
                dftfe::utils::deviceSynchronize();
              }
#endif
            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getGhostProcIds()).size();
                   ++i)
                {
                  const dftfe::Int err =
                    MPI_Irecv(recvArrayStartPtr,
                              (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                                 .data()[2 * i + 1] -
                               d_mpiPatternP2P->getGhostLocalIndicesRanges()
                                 .data()[2 * i]) *
                                d_blockSize *
                                sizeof(
                                  typename dftfe::dataTypes::singlePrecType<
                                    ValueType>::type),
                              MPI_BYTE,
                              d_mpiPatternP2P->getGhostProcIds().data()[i],
                              static_cast<dftfe::uInt>(
                                MPITags::MPI_P2P_COMMUNICATOR_SCATTER_TAG) +
                                communicationChannel,
                              d_mpiCommunicator,
                              &d_requestsUpdateGhostValues[i]);

                  std::string errMsg = "Error occured while using MPI_Irecv. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);

                  recvArrayStartPtr +=
                    (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i + 1] -
                     d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i]) *
                    d_blockSize;
                }

            // gather locally owned entries into a contiguous send buffer
            if ((d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size()) >
                0)
#ifdef DFTFE_WITH_DEVICE
              if constexpr (memorySpace == MemorySpace::DEVICE)
                MPICommunicatorP2PKernels<ValueType, memorySpace>::
                  gatherLocallyOwnedEntriesSendBufferToTargetProcs(
                    dataArray,
                    d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
                    d_blockSize,
                    d_sendRecvBufferSinglePrec,
                    dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
              else
#endif
                MPICommunicatorP2PKernels<ValueType, memorySpace>::
                  gatherLocallyOwnedEntriesSendBufferToTargetProcs(
                    dataArray,
                    d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
                    d_blockSize,
                    d_sendRecvBufferSinglePrec);

            // initiate non-blocking sends to target processors
            typename dftfe::dataTypes::singlePrecType<ValueType>::type
              *sendArrayStartPtr = d_sendRecvBufferSinglePrec.data();

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_commProtocol != communicationProtocol::dccl)
                  dftfe::utils::deviceStreamSynchronize(
                    dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
                if (d_commProtocol == communicationProtocol::mpiHost)
                  {
                    MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
                      memoryTransfer;

                    if (d_sendRecvBufferSinglePrecHostPinnedPtr->size() > 0)
                      memoryTransfer.copy(
                        d_sendRecvBufferSinglePrecHostPinnedPtr->size(),
                        d_sendRecvBufferSinglePrecHostPinnedPtr->begin(),
                        d_sendRecvBufferSinglePrec.begin());

                    sendArrayStartPtr =
                      d_sendRecvBufferSinglePrecHostPinnedPtr->begin();
                  }
              }
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::dccl)
                {
                  NCCLCHECK(ncclGroupStart());
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getTargetProcIds()).size();
                       ++i)
                    {
                      if (d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                            .data()[i] > 0)
                        NCCLCHECK(ncclSend(
                          reinterpret_cast<float *>(sendArrayStartPtr),
                          d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                              .data()[i] *
                            d_blockSize *
                            (sizeof(typename dftfe::dataTypes::singlePrecType<
                                    ValueType>::type) /
                             4),
                          ncclFloat,
                          d_mpiPatternP2P->getTargetProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      sendArrayStartPtr +=
                        d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                          .data()[i] *
                        d_blockSize;
                    }
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getGhostProcIds()).size();
                       ++i)
                    {
                      if ((d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) > 0)
                        NCCLCHECK(ncclRecv(
                          reinterpret_cast<float *>(recvArrayStartPtr),
                          (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) *
                            d_blockSize *
                            (sizeof(typename dftfe::dataTypes::singlePrecType<
                                    ValueType>::type) /
                             4),
                          ncclFloat,
                          d_mpiPatternP2P->getGhostProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      recvArrayStartPtr +=
                        (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i + 1] -
                         d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i]) *
                        d_blockSize;
                    }
                  NCCLCHECK(ncclGroupEnd());
                }
#  endif
#endif
            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getTargetProcIds()).size();
                   ++i)
                {
                  const dftfe::Int err = MPI_Isend(
                    sendArrayStartPtr,
                    d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                        .data()[i] *
                      d_blockSize *
                      sizeof(typename dftfe::dataTypes::singlePrecType<
                             ValueType>::type),
                    MPI_BYTE,
                    d_mpiPatternP2P->getTargetProcIds().data()[i],
                    static_cast<dftfe::uInt>(
                      MPITags::MPI_P2P_COMMUNICATOR_SCATTER_TAG) +
                      communicationChannel,

                    d_mpiCommunicator,
                    &d_requestsUpdateGhostValues
                      [d_mpiPatternP2P->getGhostProcIds().size() + i]);

                  std::string errMsg = "Error occured while using MPI_Isend. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);

                  sendArrayStartPtr +=
                    d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                      .data()[i] *
                    d_blockSize;
                }
          }
        else if (d_commPrecision == communicationPrecision::half)
          {
            typename dftfe::dataTypes::halfPrecType<ValueType>::type
              *recvArrayStartPtr = d_ghostDataCopyHalfPrec.data();

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_commProtocol == communicationProtocol::mpiHost)
                  recvArrayStartPtr =
                    d_ghostDataCopyHalfPrecHostPinnedPtr->begin();
                dftfe::utils::deviceSynchronize();
              }
#endif
            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getGhostProcIds()).size();
                   ++i)
                {
                  const dftfe::Int err =
                    MPI_Irecv(recvArrayStartPtr,
                              (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                                 .data()[2 * i + 1] -
                               d_mpiPatternP2P->getGhostLocalIndicesRanges()
                                 .data()[2 * i]) *
                                d_blockSize *
                                sizeof(typename dftfe::dataTypes::halfPrecType<
                                       ValueType>::type),
                              MPI_BYTE,
                              d_mpiPatternP2P->getGhostProcIds().data()[i],
                              static_cast<dftfe::uInt>(
                                MPITags::MPI_P2P_COMMUNICATOR_SCATTER_TAG) +
                                communicationChannel,
                              d_mpiCommunicator,
                              &d_requestsUpdateGhostValues[i]);

                  std::string errMsg = "Error occured while using MPI_Irecv. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);

                  recvArrayStartPtr +=
                    (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i + 1] -
                     d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i]) *
                    d_blockSize;
                }
            // gather locally owned entries into a contiguous send buffer
            if ((d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size()) >
                0)
#ifdef DFTFE_WITH_DEVICE
              if constexpr (memorySpace == MemorySpace::DEVICE)
                MPICommunicatorP2PKernels<ValueType, memorySpace>::
                  gatherLocallyOwnedEntriesSendBufferToTargetProcs(
                    dataArray,
                    d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
                    d_blockSize,
                    d_sendRecvBufferHalfPrec,
                    dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
              else
#endif
                {
                  std::string errMsg = "Not Implemented";
                  throwException(false, errMsg);
                }

            // initiate non-blocking sends to target processors
            typename dftfe::dataTypes::halfPrecType<ValueType>::type
              *sendArrayStartPtr = d_sendRecvBufferHalfPrec.data();

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_commProtocol != communicationProtocol::dccl)
                  dftfe::utils::deviceStreamSynchronize(
                    dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
                if (d_commProtocol == communicationProtocol::mpiHost)
                  {
                    MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
                      memoryTransfer;

                    if (d_sendRecvBufferHalfPrecHostPinnedPtr->size() > 0)
                      memoryTransfer.copy(
                        d_sendRecvBufferHalfPrecHostPinnedPtr->size(),
                        d_sendRecvBufferHalfPrecHostPinnedPtr->begin(),
                        d_sendRecvBufferHalfPrec.begin());

                    sendArrayStartPtr =
                      d_sendRecvBufferHalfPrecHostPinnedPtr->begin();
                  }
              }

#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::dccl)
                {
                  NCCLCHECK(ncclGroupStart());
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getTargetProcIds()).size();
                       ++i)
                    {
                      if (d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                            .data()[i] > 0)
                        NCCLCHECK(ncclSend(
                          reinterpret_cast<char *>(sendArrayStartPtr),
                          d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                              .data()[i] *
                            d_blockSize *
                            sizeof(typename dftfe::dataTypes::halfPrecType<
                                   ValueType>::type),
                          ncclChar,
                          d_mpiPatternP2P->getTargetProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      sendArrayStartPtr +=
                        d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                          .data()[i] *
                        d_blockSize;
                    }
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getGhostProcIds()).size();
                       ++i)
                    {
                      if ((d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) > 0)
                        NCCLCHECK(ncclRecv(
                          reinterpret_cast<char *>(recvArrayStartPtr),
                          (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) *
                            d_blockSize *
                            sizeof(typename dftfe::dataTypes::halfPrecType<
                                   ValueType>::type),
                          ncclChar,
                          d_mpiPatternP2P->getGhostProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      recvArrayStartPtr +=
                        (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i + 1] -
                         d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i]) *
                        d_blockSize;
                    }
                  NCCLCHECK(ncclGroupEnd());
                }
#  endif
#endif
            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getTargetProcIds()).size();
                   ++i)
                {
                  const int err = MPI_Isend(
                    sendArrayStartPtr,
                    d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                        .data()[i] *
                      d_blockSize *
                      sizeof(typename dftfe::dataTypes::halfPrecType<
                             ValueType>::type),
                    MPI_BYTE,
                    d_mpiPatternP2P->getTargetProcIds().data()[i],
                    static_cast<dftfe::uInt>(
                      MPITags::MPI_P2P_COMMUNICATOR_SCATTER_TAG) +
                      communicationChannel,

                    d_mpiCommunicator,
                    &d_requestsUpdateGhostValues
                      [d_mpiPatternP2P->getGhostProcIds().size() + i]);

                  std::string errMsg = "Error occured while using MPI_Isend. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);

                  sendArrayStartPtr +=
                    d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                      .data()[i] *
                    d_blockSize;
                }
          }

        else if (d_commPrecision == communicationPrecision::compress)
          {
            typename dftfe::dataTypes::compressType<ValueType>::type
              *recvArrayStartPtr = d_ghostDataCopyCompress.data();

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_commProtocol == communicationProtocol::mpiHost)
                  recvArrayStartPtr =
                    d_ghostDataCopyCompressHostPinnedPtr->begin();
                dftfe::utils::deviceSynchronize();
              }
#endif
            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getGhostProcIds()).size();
                   ++i)
                {
                  const dftfe::Int err =
                    MPI_Irecv(recvArrayStartPtr,
                              (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                                 .data()[2 * i + 1] -
                               d_mpiPatternP2P->getGhostLocalIndicesRanges()
                                 .data()[2 * i]) *
                                d_blockSize * d_compressBitsPerValue *
                                sizeof(typename dftfe::dataTypes::compressType<
                                       ValueType>::type) /
                                8,
                              MPI_BYTE,
                              d_mpiPatternP2P->getGhostProcIds().data()[i],
                              static_cast<dftfe::uInt>(
                                MPITags::MPI_P2P_COMMUNICATOR_SCATTER_TAG) +
                                communicationChannel,
                              d_mpiCommunicator,
                              &d_requestsUpdateGhostValues[i]);

                  std::string errMsg = "Error occured while using MPI_Irecv. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);

                  recvArrayStartPtr +=
                    (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i + 1] -
                     d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i]) *
                    d_blockSize * d_compressBitsPerValue / 8;
                }
            // compressGather: fused gather+compress
            if ((d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size()) >
                0)
#ifdef DFTFE_WITH_DEVICE
              if constexpr (memorySpace == MemorySpace::DEVICE)
                {
                  dftfe::compressionWrapper::compressGather(
                    dataArray.data(),
                    d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs()
                      .data(),
                    d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs()
                      .size(),
                    d_blockSize,
                    d_sendRecvBufferCompress.data(),
                    d_compressBitsPerValue,
                    dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
                }
              else
#endif
                {
                  std::string errMsg = "Not Implemented";
                  throwException(false, errMsg);
                }

            typename dftfe::dataTypes::compressType<ValueType>::type
              *sendArrayStartPtr = d_sendRecvBufferCompress.data();

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_commProtocol != communicationProtocol::dccl)
                  dftfe::utils::deviceStreamSynchronize(
                    dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
                if (d_commProtocol == communicationProtocol::mpiHost)
                  {
                    MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
                      memoryTransfer;
                    // copies only active bytes
                    if (d_compressedTargetBytes > 0)
                      memoryTransfer.copy(
                        d_compressedTargetBytes,
                        d_sendRecvBufferCompressHostPinnedPtr->begin(),
                        d_sendRecvBufferCompress.begin());

                    sendArrayStartPtr =
                      d_sendRecvBufferCompressHostPinnedPtr->begin();
                  }
              }

#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::dccl)
                {
                  NCCLCHECK(ncclGroupStart());
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getTargetProcIds()).size();
                       ++i)
                    {
                      if (d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                            .data()[i] > 0)
                        NCCLCHECK(ncclSend(
                          reinterpret_cast<char *>(sendArrayStartPtr),
                          d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                              .data()[i] *
                            d_blockSize * d_compressBitsPerValue *
                            sizeof(typename dftfe::dataTypes::compressType<
                                   ValueType>::type) /
                            8,
                          ncclChar,
                          d_mpiPatternP2P->getTargetProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      sendArrayStartPtr +=
                        d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                          .data()[i] *
                        d_blockSize * d_compressBitsPerValue / 8;
                    }
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getGhostProcIds()).size();
                       ++i)
                    {
                      if ((d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) > 0)
                        NCCLCHECK(ncclRecv(
                          reinterpret_cast<char *>(recvArrayStartPtr),
                          (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) *
                            d_blockSize * d_compressBitsPerValue *
                            sizeof(typename dftfe::dataTypes::compressType<
                                   ValueType>::type) /
                            8,
                          ncclChar,
                          d_mpiPatternP2P->getGhostProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      recvArrayStartPtr +=
                        (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i + 1] -
                         d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i]) *
                        d_blockSize * d_compressBitsPerValue / 8;
                    }
                  NCCLCHECK(ncclGroupEnd());
                }
#  endif
#endif
            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getTargetProcIds()).size();
                   ++i)
                {
                  const int err = MPI_Isend(
                    sendArrayStartPtr,
                    d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                        .data()[i] *
                      d_blockSize * d_compressBitsPerValue *
                      sizeof(typename dftfe::dataTypes::compressType<
                             ValueType>::type) /
                      8,
                    MPI_BYTE,
                    d_mpiPatternP2P->getTargetProcIds().data()[i],
                    static_cast<dftfe::uInt>(
                      MPITags::MPI_P2P_COMMUNICATOR_SCATTER_TAG) +
                      communicationChannel,

                    d_mpiCommunicator,
                    &d_requestsUpdateGhostValues
                      [d_mpiPatternP2P->getGhostProcIds().size() + i]);

                  std::string errMsg = "Error occured while using MPI_Isend. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);

                  sendArrayStartPtr +=
                    d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                      .data()[i] *
                    d_blockSize * d_compressBitsPerValue / 8;
                }
          }
      }

      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::updateGhostValuesEnd(
        MemoryStorage<ValueType, memorySpace> &dataArray)
      {
        throwException<LogicError>(
          d_updateGhostValuesInFlight,
          "updateGhostValuesEnd() was called without a matching "
          "updateGhostValuesBegin().");
        // wait for all send and recv requests to be completed
#if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
        if constexpr (memorySpace == MemorySpace::DEVICE)
          if (d_commProtocol == communicationProtocol::dccl)
            dftfe::utils::deviceStreamSynchronize(
              dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
#endif
        if (d_requestsUpdateGhostValues.size() > 0)
          {
            if (d_commProtocol != communicationProtocol::dccl)
              {
                const dftfe::Int err =
                  MPI_Waitall(d_requestsUpdateGhostValues.size(),
                              d_requestsUpdateGhostValues.data(),
                              MPI_STATUSES_IGNORE);
                std::string errMsg = "Error occured while using MPI_Waitall. "
                                     "Error code: " +
                                     std::to_string(err);
                throwException(err == MPI_SUCCESS, errMsg);
              }
          }
        if (d_commPrecision == communicationPrecision::standard)
          {
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::mpiHost)
                {
                  MemoryTransfer<memorySpace, MemorySpace::HOST_PINNED>
                    memoryTransfer;
                  if (d_ghostDataCopyHostPinnedPtr->size() > 0)
                    memoryTransfer.copy(d_ghostDataCopyHostPinnedPtr->size(),
                                        dataArray.begin() +
                                          d_mpiPatternP2P->localOwnedSize() *
                                            d_blockSize,
                                        d_ghostDataCopyHostPinnedPtr->data());
                }
#endif
          }
        else if (d_commPrecision == communicationPrecision::single)
          {
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::mpiHost)
                {
                  MemoryTransfer<memorySpace, MemorySpace::HOST_PINNED>
                    memoryTransfer;
                  if (d_ghostDataCopySinglePrecHostPinnedPtr->size() > 0)
                    memoryTransfer.copy(
                      d_ghostDataCopySinglePrecHostPinnedPtr->size(),
                      d_ghostDataCopySinglePrec.data(),
                      d_ghostDataCopySinglePrecHostPinnedPtr->data());
                }
            if constexpr (memorySpace == MemorySpace::DEVICE)
              MPICommunicatorP2PKernels<ValueType, memorySpace>::
                copyValueType1ArrToValueType2Arr(
                  d_ghostDataCopySinglePrec.size(),
                  d_ghostDataCopySinglePrec.data(),
                  dataArray.begin() +
                    d_mpiPatternP2P->localOwnedSize() * d_blockSize,
                  dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
            else
#endif
              MPICommunicatorP2PKernels<ValueType, memorySpace>::
                copyValueType1ArrToValueType2Arr(
                  d_ghostDataCopySinglePrec.size(),
                  d_ghostDataCopySinglePrec.data(),
                  dataArray.begin() +
                    d_mpiPatternP2P->localOwnedSize() * d_blockSize);
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              dftfe::utils::deviceStreamSynchronize(
                dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
#endif
          }

        else if (d_commPrecision == communicationPrecision::half)
          {
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::mpiHost)
                {
                  MemoryTransfer<memorySpace, MemorySpace::HOST_PINNED>
                    memoryTransfer;
                  if (d_ghostDataCopyHalfPrecHostPinnedPtr->size() > 0)
                    memoryTransfer.copy(
                      d_ghostDataCopyHalfPrecHostPinnedPtr->size(),
                      d_ghostDataCopyHalfPrec.data(),
                      d_ghostDataCopyHalfPrecHostPinnedPtr->data());
                }
            if constexpr (memorySpace == MemorySpace::DEVICE)
              MPICommunicatorP2PKernels<ValueType, memorySpace>::
                copyValueType1ArrToValueType2Arr(
                  d_ghostDataCopyHalfPrec.size(),
                  d_ghostDataCopyHalfPrec.data(),
                  dataArray.begin() +
                    d_mpiPatternP2P->localOwnedSize() * d_blockSize,
                  dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
            else
#endif
              {
                std::string errMsg = "Not Implemented";
                throwException(false, errMsg);
              }
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              dftfe::utils::deviceStreamSynchronize(
                dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
#endif
          }

        else if (d_commPrecision == communicationPrecision::compress)
          {
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::mpiHost)
                {
                  MemoryTransfer<memorySpace, MemorySpace::HOST_PINNED>
                    memoryTransfer;
                  if (d_compressedGhostBytes > 0)
                    memoryTransfer.copy(
                      d_compressedGhostBytes,
                      d_ghostDataCopyCompress.data(),
                      d_ghostDataCopyCompressHostPinnedPtr->data());
                }
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                dftfe::compressionWrapper::decompress(
                  d_ghostDataCopyCompress.data(),
                  dataArray.data() +
                    d_mpiPatternP2P->localOwnedSize() * d_blockSize,
                  d_mpiPatternP2P->localGhostSize() * d_blockSize,
                  d_compressBitsPerValue,
                  dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
              }
            else
#endif
              {
                std::string errMsg = "Not Implemented";
                throwException(false, errMsg);
              }
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              dftfe::utils::deviceStreamSynchronize(
                dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
#endif
          }
          
					d_updateGhostValuesInFlight = false;
      }

      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::accumulateAddLocallyOwned(
        MemoryStorage<ValueType, memorySpace> &dataArray,
        const dftfe::uInt                      communicationChannel)
      {
        accumulateAddLocallyOwnedBegin(dataArray, communicationChannel);
        accumulateAddLocallyOwnedEnd(dataArray);
      }

      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::
        accumulateAddLocallyOwnedBegin(
          MemoryStorage<ValueType, memorySpace> &dataArray,
          const dftfe::uInt                      communicationChannel)
      {

        throwException<LogicError>(
          !d_accumulateAddLocallyOwnedInFlight,
          "accumulateAddLocallyOwnedBegin() was called on an "
          "MPICommunicatorP2P that already has an outstanding "
          "accumulateAddLocallyOwned operation. The two calls share the same "
          "set of MPI_Request handles, so the requests of the earlier call "
          "would be overwritten and leaked. Call "
          "accumulateAddLocallyOwnedEnd() before starting the next one.");
        
			  d_accumulateAddLocallyOwnedInFlight = true;

        if (d_commPrecision == communicationPrecision::standard)
          {
            // initiate non-blocking receives from target processors
            ValueType *recvArrayStartPtr = d_sendRecvBuffer.data();
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_commProtocol == communicationProtocol::mpiHost)
                  recvArrayStartPtr = d_sendRecvBufferHostPinnedPtr->begin();

                dftfe::utils::deviceSynchronize();
              }
#endif
            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getTargetProcIds()).size();
                   ++i)
                {
                  const dftfe::Int err =
                    MPI_Irecv(recvArrayStartPtr,
                              d_mpiPatternP2P
                                  ->getNumOwnedIndicesForTargetProcs()
                                  .data()[i] *
                                d_blockSize * sizeof(ValueType),
                              MPI_BYTE,
                              d_mpiPatternP2P->getTargetProcIds().data()[i],
                              static_cast<dftfe::uInt>(
                                MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
                                communicationChannel,
                              d_mpiCommunicator,
                              &d_requestsAccumulateAddLocallyOwned[i]);

                  std::string errMsg = "Error occured while using MPI_Irecv. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);


                  recvArrayStartPtr +=
                    d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                      .data()[i] *
                    d_blockSize;
                }

            // initiate non-blocking sends to ghost processors
            ValueType *sendArrayStartPtr =
              dataArray.data() +
              d_mpiPatternP2P->localOwnedSize() * d_blockSize;

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::mpiHost)
                {
                  MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
                    memoryTransfer;
                  if (d_ghostDataCopyHostPinnedPtr->size() > 0)
                    memoryTransfer.copy(d_ghostDataCopyHostPinnedPtr->size(),
                                        d_ghostDataCopyHostPinnedPtr->begin(),
                                        dataArray.begin() +
                                          d_mpiPatternP2P->localOwnedSize() *
                                            d_blockSize);

                  sendArrayStartPtr = d_ghostDataCopyHostPinnedPtr->begin();
                }
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::dccl)
                {
                  NCCLCHECK(ncclGroupStart());
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getGhostProcIds()).size();
                       ++i)
                    {
                      if ((d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) > 0)
                        NCCLCHECK(ncclSend(
                          reinterpret_cast<float *>(sendArrayStartPtr),
                          (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) *
                            d_blockSize * (sizeof(ValueType) / 4),
                          ncclFloat,
                          d_mpiPatternP2P->getGhostProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      sendArrayStartPtr +=
                        (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i + 1] -
                         d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i]) *
                        d_blockSize;
                    }
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getTargetProcIds()).size();
                       ++i)
                    {
                      if (d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                            .data()[i] > 0)
                        NCCLCHECK(ncclRecv(
                          reinterpret_cast<float *>(recvArrayStartPtr),
                          d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                              .data()[i] *
                            d_blockSize * (sizeof(ValueType) / 4),
                          ncclFloat,
                          d_mpiPatternP2P->getTargetProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      recvArrayStartPtr +=
                        d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                          .data()[i] *
                        d_blockSize;
                    }
                  NCCLCHECK(ncclGroupEnd());
                }
#  endif
#endif
            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getGhostProcIds()).size();
                   ++i)
                {
                  const dftfe::Int err = MPI_Isend(
                    sendArrayStartPtr,
                    (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i + 1] -
                     d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i]) *
                      d_blockSize * sizeof(ValueType),
                    MPI_BYTE,
                    d_mpiPatternP2P->getGhostProcIds().data()[i],
                    static_cast<dftfe::uInt>(
                      MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
                      communicationChannel,
                    d_mpiCommunicator,
                    &d_requestsAccumulateAddLocallyOwned
                      [(d_mpiPatternP2P->getTargetProcIds()).size() + i]);


                  std::string errMsg = "Error occured while using MPI_Isend. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);

                  sendArrayStartPtr +=
                    (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i + 1] -
                     d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i]) *
                    d_blockSize;
                }
          }
        else if (d_commPrecision == communicationPrecision::single)
          {
            // initiate non-blocking receives from target processors
            typename dftfe::dataTypes::singlePrecType<ValueType>::type
              *recvArrayStartPtr = d_sendRecvBufferSinglePrec.data();
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_commProtocol == communicationProtocol::mpiHost)
                  recvArrayStartPtr =
                    d_sendRecvBufferSinglePrecHostPinnedPtr->begin();

                dftfe::utils::deviceSynchronize();
              }
#endif
            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getTargetProcIds()).size();
                   ++i)
                {
                  const dftfe::Int err = MPI_Irecv(
                    recvArrayStartPtr,
                    d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                        .data()[i] *
                      d_blockSize *
                      sizeof(typename dftfe::dataTypes::singlePrecType<
                             ValueType>::type),
                    MPI_BYTE,
                    d_mpiPatternP2P->getTargetProcIds().data()[i],
                    static_cast<dftfe::uInt>(
                      MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
                      communicationChannel,
                    d_mpiCommunicator,
                    &d_requestsAccumulateAddLocallyOwned[i]);

                  std::string errMsg = "Error occured while using MPI_Irecv. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);


                  recvArrayStartPtr +=
                    d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                      .data()[i] *
                    d_blockSize;
                }

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              MPICommunicatorP2PKernels<ValueType, memorySpace>::
                copyValueType1ArrToValueType2Arr(
                  d_ghostDataCopySinglePrec.size(),
                  dataArray.data() +
                    d_mpiPatternP2P->localOwnedSize() * d_blockSize,
                  d_ghostDataCopySinglePrec.data(),
                  dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
            else
#endif
              MPICommunicatorP2PKernels<ValueType, memorySpace>::
                copyValueType1ArrToValueType2Arr(
                  d_ghostDataCopySinglePrec.size(),
                  dataArray.data() +
                    d_mpiPatternP2P->localOwnedSize() * d_blockSize,
                  d_ghostDataCopySinglePrec.data());

            // initiate non-blocking sends to ghost processors
            typename dftfe::dataTypes::singlePrecType<ValueType>::type
              *sendArrayStartPtr = d_ghostDataCopySinglePrec.data();

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_commProtocol != communicationProtocol::dccl)
                  dftfe::utils::deviceStreamSynchronize(
                    dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
                if (d_commProtocol == communicationProtocol::mpiHost)
                  {
                    MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
                      memoryTransfer;
                    if (d_ghostDataCopySinglePrecHostPinnedPtr->size() > 0)
                      memoryTransfer.copy(
                        d_ghostDataCopySinglePrecHostPinnedPtr->size(),
                        d_ghostDataCopySinglePrecHostPinnedPtr->begin(),
                        d_ghostDataCopySinglePrec.data());

                    sendArrayStartPtr =
                      d_ghostDataCopySinglePrecHostPinnedPtr->begin();
                  }
              }
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::dccl)
                {
                  NCCLCHECK(ncclGroupStart());
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getGhostProcIds()).size();
                       ++i)
                    {
                      if ((d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) > 0)
                        NCCLCHECK(ncclSend(
                          reinterpret_cast<float *>(sendArrayStartPtr),
                          (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) *
                            d_blockSize *
                            (sizeof(typename dftfe::dataTypes::singlePrecType<
                                    ValueType>::type) /
                             4),
                          ncclFloat,
                          d_mpiPatternP2P->getGhostProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      sendArrayStartPtr +=
                        (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i + 1] -
                         d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i]) *
                        d_blockSize;
                    }
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getTargetProcIds()).size();
                       ++i)
                    {
                      if (d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                            .data()[i] > 0)
                        NCCLCHECK(ncclRecv(
                          reinterpret_cast<float *>(recvArrayStartPtr),
                          d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                              .data()[i] *
                            d_blockSize *
                            (sizeof(typename dftfe::dataTypes::singlePrecType<
                                    ValueType>::type) /
                             4),
                          ncclFloat,
                          d_mpiPatternP2P->getTargetProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      recvArrayStartPtr +=
                        d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                          .data()[i] *
                        d_blockSize;
                    }
                  NCCLCHECK(ncclGroupEnd());
                }
#  endif
#endif
            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getGhostProcIds()).size();
                   ++i)
                {
                  const dftfe::Int err = MPI_Isend(
                    sendArrayStartPtr,
                    (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i + 1] -
                     d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i]) *
                      d_blockSize *
                      sizeof(typename dftfe::dataTypes::singlePrecType<
                             ValueType>::type),
                    MPI_BYTE,
                    d_mpiPatternP2P->getGhostProcIds().data()[i],
                    static_cast<dftfe::uInt>(
                      MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
                      communicationChannel,
                    d_mpiCommunicator,
                    &d_requestsAccumulateAddLocallyOwned
                      [(d_mpiPatternP2P->getTargetProcIds()).size() + i]);


                  std::string errMsg = "Error occured while using MPI_Isend. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);

                  sendArrayStartPtr +=
                    (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i + 1] -
                     d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i]) *
                    d_blockSize;
                }
          }
        else if (d_commPrecision == communicationPrecision::half)
          {
            // initiate non-blocking receives from target processors
            typename dftfe::dataTypes::halfPrecType<ValueType>::type
              *recvArrayStartPtr = d_sendRecvBufferHalfPrec.data();
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_commProtocol == communicationProtocol::mpiHost)
                  recvArrayStartPtr =
                    d_sendRecvBufferHalfPrecHostPinnedPtr->begin();

                dftfe::utils::deviceSynchronize();
              }
#endif

            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getTargetProcIds()).size();
                   ++i)
                {
                  const int err =
                    MPI_Irecv(recvArrayStartPtr,
                              d_mpiPatternP2P
                                  ->getNumOwnedIndicesForTargetProcs()
                                  .data()[i] *
                                d_blockSize *
                                sizeof(typename dftfe::dataTypes::halfPrecType<
                                       ValueType>::type),
                              MPI_BYTE,
                              d_mpiPatternP2P->getTargetProcIds().data()[i],
                              static_cast<dftfe::uInt>(
                                MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
                                communicationChannel,
                              d_mpiCommunicator,
                              &d_requestsAccumulateAddLocallyOwned[i]);

                  std::string errMsg = "Error occured while using MPI_Irecv. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);


                  recvArrayStartPtr +=
                    d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                      .data()[i] *
                    d_blockSize;
                }

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              MPICommunicatorP2PKernels<ValueType, memorySpace>::
                copyValueType1ArrToValueType2Arr(
                  d_ghostDataCopyHalfPrec.size(),
                  dataArray.data() +
                    d_mpiPatternP2P->localOwnedSize() * d_blockSize,
                  d_ghostDataCopyHalfPrec.data(),
                  dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
            else
#endif
              {
                std::string errMsg = "Not Implemented";
                throwException(false, errMsg);
              }

            // initiate non-blocking sends to ghost processors
            typename dftfe::dataTypes::halfPrecType<ValueType>::type
              *sendArrayStartPtr = d_ghostDataCopyHalfPrec.data();

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_commProtocol != communicationProtocol::dccl)
                  dftfe::utils::deviceStreamSynchronize(
                    dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
                if (d_commProtocol == communicationProtocol::mpiHost)
                  {
                    MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
                      memoryTransfer;
                    if (d_ghostDataCopyHalfPrecHostPinnedPtr->size() > 0)
                      memoryTransfer.copy(
                        d_ghostDataCopyHalfPrecHostPinnedPtr->size(),
                        d_ghostDataCopyHalfPrecHostPinnedPtr->begin(),
                        d_ghostDataCopyHalfPrec.data());

                    sendArrayStartPtr =
                      d_ghostDataCopyHalfPrecHostPinnedPtr->begin();
                  }
              }
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::dccl)
                {
                  NCCLCHECK(ncclGroupStart());
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getGhostProcIds()).size();
                       ++i)
                    {
                      if ((d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) > 0)
                        NCCLCHECK(ncclSend(
                          reinterpret_cast<char *>(sendArrayStartPtr),
                          (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) *
                            d_blockSize *
                            sizeof(typename dftfe::dataTypes::halfPrecType<
                                   ValueType>::type),
                          ncclChar,
                          d_mpiPatternP2P->getGhostProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      sendArrayStartPtr +=
                        (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i + 1] -
                         d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i]) *
                        d_blockSize;
                    }
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getTargetProcIds()).size();
                       ++i)
                    {
                      if (d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                            .data()[i] > 0)
                        NCCLCHECK(ncclRecv(
                          reinterpret_cast<char *>(recvArrayStartPtr),
                          d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                              .data()[i] *
                            d_blockSize *
                            sizeof(typename dftfe::dataTypes::halfPrecType<
                                   ValueType>::type),
                          ncclChar,
                          d_mpiPatternP2P->getTargetProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      recvArrayStartPtr +=
                        d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                          .data()[i] *
                        d_blockSize;
                    }
                  NCCLCHECK(ncclGroupEnd());
                }
#  endif
#endif
            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getGhostProcIds()).size();
                   ++i)
                {
                  const int err = MPI_Isend(
                    sendArrayStartPtr,
                    (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i + 1] -
                     d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i]) *
                      d_blockSize *
                      sizeof(typename dftfe::dataTypes::halfPrecType<
                             ValueType>::type),
                    MPI_BYTE,
                    d_mpiPatternP2P->getGhostProcIds().data()[i],
                    static_cast<dftfe::uInt>(
                      MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
                      communicationChannel,
                    d_mpiCommunicator,
                    &d_requestsAccumulateAddLocallyOwned
                      [(d_mpiPatternP2P->getTargetProcIds()).size() + i]);


                  std::string errMsg = "Error occured while using MPI_Isend. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);

                  sendArrayStartPtr +=
                    (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i + 1] -
                     d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i]) *
                    d_blockSize;
                }
          }

        else if (d_commPrecision == communicationPrecision::compress)
          {
            typename dftfe::dataTypes::compressType<ValueType>::type
              *recvArrayStartPtr = d_sendRecvBufferCompress.data();
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_commProtocol == communicationProtocol::mpiHost)
                  recvArrayStartPtr =
                    d_sendRecvBufferCompressHostPinnedPtr->begin();

                dftfe::utils::deviceSynchronize();
              }
#endif

            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getTargetProcIds()).size();
                   ++i)
                {
                  const int err =
                    MPI_Irecv(recvArrayStartPtr,
                              d_mpiPatternP2P
                                  ->getNumOwnedIndicesForTargetProcs()
                                  .data()[i] *
                                d_blockSize * d_compressBitsPerValue *
                                sizeof(typename dftfe::dataTypes::compressType<
                                       ValueType>::type) /
                                8,
                              MPI_BYTE,
                              d_mpiPatternP2P->getTargetProcIds().data()[i],
                              static_cast<dftfe::uInt>(
                                MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
                                communicationChannel,
                              d_mpiCommunicator,
                              &d_requestsAccumulateAddLocallyOwned[i]);

                  std::string errMsg = "Error occured while using MPI_Irecv. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);


                  recvArrayStartPtr +=
                    d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                      .data()[i] *
                    d_blockSize * d_compressBitsPerValue / 8;
                }

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_mpiPatternP2P->localGhostSize() > 0)
                  {
                    dftfe::compressionWrapper::compress(
                      dataArray.data() +
                        d_mpiPatternP2P->localOwnedSize() * d_blockSize,
                      d_ghostDataCopyCompress.data(),
                      d_mpiPatternP2P->localGhostSize() * d_blockSize,
                      d_compressBitsPerValue,
                      dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
                  }
              }
            else
#endif
              {
                std::string errMsg = "Not Implemented";
                throwException(false, errMsg);
              }

            typename dftfe::dataTypes::compressType<ValueType>::type
              *sendArrayStartPtr = d_ghostDataCopyCompress.data();

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_commProtocol != communicationProtocol::dccl)
                  dftfe::utils::deviceStreamSynchronize(
                    dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
                if (d_commProtocol == communicationProtocol::mpiHost)
                  {
                    MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
                      memoryTransfer;
                    if (d_compressedGhostBytes > 0)
                      memoryTransfer.copy(
                        d_compressedGhostBytes,
                        d_ghostDataCopyCompressHostPinnedPtr->begin(),
                        d_ghostDataCopyCompress.data());

                    sendArrayStartPtr =
                      d_ghostDataCopyCompressHostPinnedPtr->begin();
                  }
              }
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::dccl)
                {
                  NCCLCHECK(ncclGroupStart());
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getGhostProcIds()).size();
                       ++i)
                    {
                      if ((d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) > 0)
                        NCCLCHECK(ncclSend(
                          reinterpret_cast<char *>(sendArrayStartPtr),
                          (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) *
                            d_blockSize * d_compressBitsPerValue *
                            sizeof(typename dftfe::dataTypes::compressType<
                                   ValueType>::type) /
                            8,
                          ncclChar,
                          d_mpiPatternP2P->getGhostProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      sendArrayStartPtr +=
                        (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i + 1] -
                         d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i]) *
                        d_blockSize * d_compressBitsPerValue / 8;
                    }
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getTargetProcIds()).size();
                       ++i)
                    {
                      if (d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                            .data()[i] > 0)
                        NCCLCHECK(ncclRecv(
                          reinterpret_cast<char *>(recvArrayStartPtr),
                          d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                              .data()[i] *
                            d_blockSize * d_compressBitsPerValue *
                            sizeof(typename dftfe::dataTypes::compressType<
                                   ValueType>::type) /
                            8,
                          ncclChar,
                          d_mpiPatternP2P->getTargetProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      recvArrayStartPtr +=
                        d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                          .data()[i] *
                        d_blockSize * d_compressBitsPerValue / 8;
                    }
                  NCCLCHECK(ncclGroupEnd());
                }
#  endif
#endif
            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getGhostProcIds()).size();
                   ++i)
                {
                  const int err = MPI_Isend(
                    sendArrayStartPtr,
                    (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i + 1] -
                     d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i]) *
                      d_blockSize * d_compressBitsPerValue *
                      sizeof(typename dftfe::dataTypes::compressType<
                             ValueType>::type) /
                      8,
                    MPI_BYTE,
                    d_mpiPatternP2P->getGhostProcIds().data()[i],
                    static_cast<dftfe::uInt>(
                      MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
                      communicationChannel,
                    d_mpiCommunicator,
                    &d_requestsAccumulateAddLocallyOwned
                      [(d_mpiPatternP2P->getTargetProcIds()).size() + i]);

                  std::string errMsg = "Error occured while using MPI_Isend. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);

                  sendArrayStartPtr +=
                    (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i + 1] -
                     d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i]) *
                    d_blockSize * d_compressBitsPerValue / 8;
                }
          }
      }

      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::accumulateAddLocallyOwnedEnd(
        MemoryStorage<ValueType, memorySpace> &dataArray)
      {
        throwException<LogicError>(
          d_accumulateAddLocallyOwnedInFlight,
          "accumulateAddLocallyOwnedEnd() was called without a matching "
          "accumulateAddLocallyOwnedBegin().");
		
        // wait for all send and recv requests to be completed
#if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
        if constexpr (memorySpace == MemorySpace::DEVICE)
          if (d_commProtocol == communicationProtocol::dccl)
            dftfe::utils::deviceStreamSynchronize(
              dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
#endif
        if (d_requestsAccumulateAddLocallyOwned.size() > 0)
          {
            if (d_commProtocol != communicationProtocol::dccl)
              {
                const dftfe::Int err =
                  MPI_Waitall(d_requestsAccumulateAddLocallyOwned.size(),
                              d_requestsAccumulateAddLocallyOwned.data(),
                              MPI_STATUSES_IGNORE);

                std::string errMsg = "Error occured while using MPI_Waitall. "
                                     "Error code: " +
                                     std::to_string(err);
                throwException(err == MPI_SUCCESS, errMsg);
              }
          }
        
				if (d_commPrecision == communicationPrecision::standard)
          {
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::mpiHost)
                {
                  MemoryTransfer<memorySpace, MemorySpace::HOST_PINNED>
                    memoryTransfer;
                  if (d_sendRecvBufferHostPinnedPtr->size() > 0)
                    memoryTransfer.copy(d_sendRecvBufferHostPinnedPtr->size(),
                                        d_sendRecvBuffer.data(),
                                        d_sendRecvBufferHostPinnedPtr->data());
                }
#endif
            // accumulate add into locally owned entries from recv buffer
            if ((d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size()) >
                0)
#ifdef DFTFE_WITH_DEVICE
              if constexpr (memorySpace == MemorySpace::DEVICE)
                MPICommunicatorP2PKernels<ValueType, memorySpace>::
                  accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
                    d_sendRecvBuffer,
                    d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
                    d_blockSize,
                    d_locallyOwnedSize,
                    d_ghostSize,
                    dataArray,
                    dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
              else
#endif
                MPICommunicatorP2PKernels<ValueType, memorySpace>::
                  accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
                    d_sendRecvBuffer,
                    d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
                    d_blockSize,
                    d_locallyOwnedSize,
                    d_ghostSize,
                    dataArray);
          }
        else if (d_commPrecision == communicationPrecision::single)
          {
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::mpiHost)
                {
                  MemoryTransfer<memorySpace, MemorySpace::HOST_PINNED>
                    memoryTransfer;
                  if (d_sendRecvBufferSinglePrecHostPinnedPtr->size() > 0)
                    memoryTransfer.copy(
                      d_sendRecvBufferSinglePrecHostPinnedPtr->size(),
                      d_sendRecvBufferSinglePrec.data(),
                      d_sendRecvBufferSinglePrecHostPinnedPtr->data());
                }
#endif
            // accumulate add into locally owned entries from recv buffer
            if ((d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size()) >
                0)
#ifdef DFTFE_WITH_DEVICE
              if constexpr (memorySpace == MemorySpace::DEVICE)
                MPICommunicatorP2PKernels<ValueType, memorySpace>::
                  accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
                    d_sendRecvBufferSinglePrec,
                    d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
                    d_blockSize,
                    d_locallyOwnedSize,
                    d_ghostSize,
                    dataArray,
                    dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
              else
#endif
                MPICommunicatorP2PKernels<ValueType, memorySpace>::
                  accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
                    d_sendRecvBufferSinglePrec,
                    d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
                    d_blockSize,
                    d_locallyOwnedSize,
                    d_ghostSize,
                    dataArray);
          }

        else if (d_commPrecision == communicationPrecision::half)
          {
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::mpiHost)
                {
                  MemoryTransfer<memorySpace, MemorySpace::HOST_PINNED>
                    memoryTransfer;
                  if (d_sendRecvBufferHalfPrecHostPinnedPtr->size() > 0)
                    memoryTransfer.copy(
                      d_sendRecvBufferHalfPrecHostPinnedPtr->size(),
                      d_sendRecvBufferHalfPrec.data(),
                      d_sendRecvBufferHalfPrecHostPinnedPtr->data());
                }
#endif
            // accumulate add into locally owned entries from recv buffer
            if ((d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size()) >
                0)
#ifdef DFTFE_WITH_DEVICE
              if constexpr (memorySpace == MemorySpace::DEVICE)
                MPICommunicatorP2PKernels<ValueType, memorySpace>::
                  accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
                    d_sendRecvBufferHalfPrec,
                    d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
                    d_blockSize,
                    d_locallyOwnedSize,
                    d_ghostSize,
                    dataArray,
                    dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
              else
#endif
                {
                  std::string errMsg = "Not implemented.";
                  throwException(false, errMsg);
                }
          }

        else if (d_commPrecision == communicationPrecision::compress)
          {
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::mpiHost)
                {
                  MemoryTransfer<memorySpace, MemorySpace::HOST_PINNED>
                    memoryTransfer;
                  if (d_compressedTargetBytes > 0)
                    memoryTransfer.copy(
                      d_compressedTargetBytes,
                      d_sendRecvBufferCompress.data(),
                      d_sendRecvBufferCompressHostPinnedPtr->data());
                }
            // decompressScatterAdd: fused decompress+scatter+atomicadd
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                dftfe::compressionWrapper::decompressScatterAdd(
                  d_sendRecvBufferCompress.data(),
                  d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().data(),
                  d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size(),
                  d_blockSize,
                  dataArray.data(),
                  d_compressBitsPerValue,
                  dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
              }
            else
#endif
              {
                std::string errMsg = "Not Implemented";
                throwException(false, errMsg);
              }
          }

#ifdef DFTFE_WITH_DEVICE
        if constexpr (memorySpace == MemorySpace::DEVICE)
          dftfe::utils::deviceStreamSynchronize(
            dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
#endif
				d_accumulateAddLocallyOwnedInFlight = false;
      }

      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::accumulateInsertLocallyOwned(
        MemoryStorage<ValueType, memorySpace> &dataArray,
        const dftfe::uInt                      communicationChannel)
      {
        accumulateInsertLocallyOwnedBegin(dataArray, communicationChannel);
        accumulateInsertLocallyOwnedEnd(dataArray);
      }

      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::
        accumulateInsertLocallyOwnedBegin(
          MemoryStorage<ValueType, memorySpace> &dataArray,
          const dftfe::uInt                      communicationChannel)
      {
        
        throwException<LogicError>(
          !d_accumulateInsertLocallyOwnedInFlight,
          "accumulateInsertLocallyOwnedBegin() was called on an "
          "MPICommunicatorP2P that already has an outstanding "
          "accumulateInsertLocallyOwned operation. The two calls share the "
          "same set of MPI_Request handles, so the requests of the earlier "
          "call would be overwritten and leaked. Call "
          "accumulateInsertLocallyOwnedEnd() before starting the next one.");
        
		    d_accumulateInsertLocallyOwnedInFlight = true;

				if (d_commPrecision == communicationPrecision::standard)
          {
            // initiate non-blocking receives from target processors
            ValueType *recvArrayStartPtr = d_sendRecvBuffer.data();

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_commProtocol == communicationProtocol::mpiHost)
                  recvArrayStartPtr = d_sendRecvBufferHostPinnedPtr->begin();

                dftfe::utils::deviceSynchronize();
              }
#endif

            if (d_commProtocol != communicationProtocol::dccl)
              {
                for (dftfe::uInt i = 0;
                     i < (d_mpiPatternP2P->getTargetProcIds()).size();
                     ++i)
                  {
                    const dftfe::Int err =
                      MPI_Irecv(recvArrayStartPtr,
                                d_mpiPatternP2P
                                    ->getNumOwnedIndicesForTargetProcs()
                                    .data()[i] *
                                  d_blockSize * sizeof(ValueType),
                                MPI_BYTE,
                                d_mpiPatternP2P->getTargetProcIds().data()[i],
                                static_cast<dftfe::uInt>(
                                  MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
                                  communicationChannel,
                                d_mpiCommunicator,
                                &d_requestsAccumulateInsertLocallyOwned[i]);

                    std::string errMsg = "Error occured while using MPI_Irecv. "
                                         "Error code: " +
                                         std::to_string(err);
                    throwException(err == MPI_SUCCESS, errMsg);


                    recvArrayStartPtr +=
                      d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                        .data()[i] *
                      d_blockSize;
                  }
              }

            // initiate non-blocking sends to ghost processors
            ValueType *sendArrayStartPtr =
              dataArray.data() +
              d_mpiPatternP2P->localOwnedSize() * d_blockSize;

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::mpiHost)
                {
                  MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
                    memoryTransfer;
                  if (d_ghostDataCopyHostPinnedPtr->size() > 0)
                    memoryTransfer.copy(d_ghostDataCopyHostPinnedPtr->size(),
                                        d_ghostDataCopyHostPinnedPtr->begin(),
                                        dataArray.begin() +
                                          d_mpiPatternP2P->localOwnedSize() *
                                            d_blockSize);

                  sendArrayStartPtr = d_ghostDataCopyHostPinnedPtr->begin();
                }
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::dccl)
                {
                  NCCLCHECK(ncclGroupStart());
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getGhostProcIds()).size();
                       ++i)
                    {
                      if ((d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) > 0)
                        NCCLCHECK(ncclSend(
                          reinterpret_cast<float *>(sendArrayStartPtr),
                          (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) *
                            d_blockSize * (sizeof(ValueType) / 4),
                          ncclFloat,
                          d_mpiPatternP2P->getGhostProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      sendArrayStartPtr +=
                        (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i + 1] -
                         d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i]) *
                        d_blockSize;
                    }
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getTargetProcIds()).size();
                       ++i)
                    {
                      if (d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                            .data()[i] > 0)
                        NCCLCHECK(ncclRecv(
                          reinterpret_cast<float *>(recvArrayStartPtr),
                          d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                              .data()[i] *
                            d_blockSize * (sizeof(ValueType) / 4),
                          ncclFloat,
                          d_mpiPatternP2P->getTargetProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      recvArrayStartPtr +=
                        d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                          .data()[i] *
                        d_blockSize;
                    }
                  NCCLCHECK(ncclGroupEnd());
                }
#  endif
#endif
            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getGhostProcIds()).size();
                   ++i)
                {
                  const dftfe::Int err = MPI_Isend(
                    sendArrayStartPtr,
                    (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i + 1] -
                     d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i]) *
                      d_blockSize * sizeof(ValueType),
                    MPI_BYTE,
                    d_mpiPatternP2P->getGhostProcIds().data()[i],
                    static_cast<dftfe::uInt>(
                      MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
                      communicationChannel,
                    d_mpiCommunicator,
                    &d_requestsAccumulateInsertLocallyOwned
                      [(d_mpiPatternP2P->getTargetProcIds()).size() + i]);


                  std::string errMsg = "Error occured while using MPI_Isend. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);

                  sendArrayStartPtr +=
                    (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i + 1] -
                     d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i]) *
                    d_blockSize;
                }
          }
        else
          {
            // initiate non-blocking receives from target processors
            typename dftfe::dataTypes::singlePrecType<ValueType>::type
              *recvArrayStartPtr = d_sendRecvBufferSinglePrec.data();

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_commProtocol == communicationProtocol::mpiHost)
                  recvArrayStartPtr =
                    d_sendRecvBufferSinglePrecHostPinnedPtr->begin();

                dftfe::utils::deviceSynchronize();
              }
#endif
            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getTargetProcIds()).size();
                   ++i)
                {
                  const dftfe::Int err = MPI_Irecv(
                    recvArrayStartPtr,
                    d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                        .data()[i] *
                      d_blockSize *
                      sizeof(typename dftfe::dataTypes::singlePrecType<
                             ValueType>::type),
                    MPI_BYTE,
                    d_mpiPatternP2P->getTargetProcIds().data()[i],
                    static_cast<dftfe::uInt>(
                      MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
                      communicationChannel,
                    d_mpiCommunicator,
                    &d_requestsAccumulateInsertLocallyOwned[i]);

                  std::string errMsg = "Error occured while using MPI_Irecv. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);


                  recvArrayStartPtr +=
                    d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                      .data()[i] *
                    d_blockSize;
                }

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              MPICommunicatorP2PKernels<ValueType, memorySpace>::
                copyValueType1ArrToValueType2Arr(
                  d_ghostDataCopySinglePrec.size(),
                  dataArray.data() +
                    d_mpiPatternP2P->localOwnedSize() * d_blockSize,
                  d_ghostDataCopySinglePrec.data(),
                  dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
            else
#endif
              MPICommunicatorP2PKernels<ValueType, memorySpace>::
                copyValueType1ArrToValueType2Arr(
                  d_ghostDataCopySinglePrec.size(),
                  dataArray.data() +
                    d_mpiPatternP2P->localOwnedSize() * d_blockSize,
                  d_ghostDataCopySinglePrec.data());

            // initiate non-blocking sends to ghost processors
            typename dftfe::dataTypes::singlePrecType<ValueType>::type
              *sendArrayStartPtr = d_ghostDataCopySinglePrec.data();

#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              {
                if (d_commProtocol != communicationProtocol::dccl)
                  dftfe::utils::deviceStreamSynchronize(
                    dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
                if (d_commProtocol == communicationProtocol::mpiHost)
                  {
                    MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
                      memoryTransfer;
                    if (d_ghostDataCopySinglePrecHostPinnedPtr->size() > 0)
                      memoryTransfer.copy(
                        d_ghostDataCopySinglePrecHostPinnedPtr->size(),
                        d_ghostDataCopySinglePrecHostPinnedPtr->begin(),
                        d_ghostDataCopySinglePrec.data());

                    sendArrayStartPtr =
                      d_ghostDataCopySinglePrecHostPinnedPtr->begin();
                  }
              }
#  if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::dccl)
                {
                  NCCLCHECK(ncclGroupStart());
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getGhostProcIds()).size();
                       ++i)
                    {
                      if ((d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) > 0)
                        NCCLCHECK(ncclSend(
                          reinterpret_cast<float *>(sendArrayStartPtr),
                          (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i + 1] -
                           d_mpiPatternP2P->getGhostLocalIndicesRanges()
                             .data()[2 * i]) *
                            d_blockSize *
                            (sizeof(typename dftfe::dataTypes::singlePrecType<
                                    ValueType>::type) /
                             4),
                          ncclFloat,
                          d_mpiPatternP2P->getGhostProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      sendArrayStartPtr +=
                        (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i + 1] -
                         d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i]) *
                        d_blockSize;
                    }
                  for (dftfe::uInt i = 0;
                       i < (d_mpiPatternP2P->getTargetProcIds()).size();
                       ++i)
                    {
                      if (d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                            .data()[i] > 0)
                        NCCLCHECK(ncclRecv(
                          reinterpret_cast<float *>(recvArrayStartPtr),
                          d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                              .data()[i] *
                            d_blockSize *
                            (sizeof(typename dftfe::dataTypes::singlePrecType<
                                    ValueType>::type) /
                             4),
                          ncclFloat,
                          d_mpiPatternP2P->getTargetProcIds().data()[i],
                          *dftfe::utils::DeviceCCLWrapper::dcclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      recvArrayStartPtr +=
                        d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                          .data()[i] *
                        d_blockSize;
                    }
                  NCCLCHECK(ncclGroupEnd());
                }
#  endif
#endif
            if (d_commProtocol != communicationProtocol::dccl)
              for (dftfe::uInt i = 0;
                   i < (d_mpiPatternP2P->getGhostProcIds()).size();
                   ++i)
                {
                  const dftfe::Int err = MPI_Isend(
                    sendArrayStartPtr,
                    (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i + 1] -
                     d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i]) *
                      d_blockSize *
                      sizeof(typename dftfe::dataTypes::singlePrecType<
                             ValueType>::type),
                    MPI_BYTE,
                    d_mpiPatternP2P->getGhostProcIds().data()[i],
                    static_cast<dftfe::uInt>(
                      MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
                      communicationChannel,
                    d_mpiCommunicator,
                    &d_requestsAccumulateInsertLocallyOwned
                      [(d_mpiPatternP2P->getTargetProcIds()).size() + i]);


                  std::string errMsg = "Error occured while using MPI_Isend. "
                                       "Error code: " +
                                       std::to_string(err);
                  throwException(err == MPI_SUCCESS, errMsg);

                  sendArrayStartPtr +=
                    (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i + 1] -
                     d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i]) *
                    d_blockSize;
                }
          }
      }


      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::
        accumulateInsertLocallyOwnedEnd(
          MemoryStorage<ValueType, memorySpace> &dataArray)
      {
        throwException<LogicError>(
          d_accumulateInsertLocallyOwnedInFlight,
          "accumulateInsertLocallyOwnedEnd() was called without a matching "
          "accumulateInsertLocallyOwnedBegin().");
        // wait for all send and recv requests to be completed
#if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
        if constexpr (memorySpace == MemorySpace::DEVICE)
          if (d_commProtocol == communicationProtocol::dccl)
            dftfe::utils::deviceStreamSynchronize(
              dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
#endif

        // wait for all send and recv requests to be completed
        if (d_requestsAccumulateInsertLocallyOwned.size() > 0)
          {
            if (d_commProtocol != communicationProtocol::dccl)
              {
                const dftfe::Int err =
                  MPI_Waitall(d_requestsAccumulateInsertLocallyOwned.size(),
                              d_requestsAccumulateInsertLocallyOwned.data(),
                              MPI_STATUSES_IGNORE);

                std::string errMsg = "Error occured while using MPI_Waitall. "
                                     "Error code: " +
                                     std::to_string(err);
                throwException(err == MPI_SUCCESS, errMsg);
              }
          }
        if (d_commPrecision == communicationPrecision::standard)
          {
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::mpiHost)
                {
                  MemoryTransfer<memorySpace, MemorySpace::HOST_PINNED>
                    memoryTransfer;
                  if (d_sendRecvBufferHostPinnedPtr->size() > 0)
                    memoryTransfer.copy(d_sendRecvBufferHostPinnedPtr->size(),
                                        d_sendRecvBuffer.data(),
                                        d_sendRecvBufferHostPinnedPtr->data());
                }
#endif
            // accumulate insert into locally owned entries from recv buffer
            if ((d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size()) >
                0)
#ifdef DFTFE_WITH_DEVICE
              if constexpr (memorySpace == MemorySpace::DEVICE)
                MPICommunicatorP2PKernels<ValueType, memorySpace>::
                  accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
                    d_sendRecvBuffer,
                    d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
                    d_blockSize,
                    d_locallyOwnedSize,
                    d_ghostSize,
                    dataArray,
                    dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
              else
#endif
                MPICommunicatorP2PKernels<ValueType, memorySpace>::
                  accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
                    d_sendRecvBuffer,
                    d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
                    d_blockSize,
                    d_locallyOwnedSize,
                    d_ghostSize,
                    dataArray);
          }
        else
          {
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::mpiHost)
                {
                  MemoryTransfer<memorySpace, MemorySpace::HOST_PINNED>
                    memoryTransfer;
                  if (d_sendRecvBufferSinglePrecHostPinnedPtr->size() > 0)
                    memoryTransfer.copy(
                      d_sendRecvBufferSinglePrecHostPinnedPtr->size(),
                      d_sendRecvBufferSinglePrec.data(),
                      d_sendRecvBufferSinglePrecHostPinnedPtr->data());
                }
#endif
            // accumulate insert into locally owned entries from recv buffer
            if ((d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size()) >
                0)
#ifdef DFTFE_WITH_DEVICE
              if constexpr (memorySpace == MemorySpace::DEVICE)
                MPICommunicatorP2PKernels<ValueType, memorySpace>::
                  accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
                    d_sendRecvBufferSinglePrec,
                    d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
                    d_blockSize,
                    d_locallyOwnedSize,
                    d_ghostSize,
                    dataArray,
                    dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
              else
#endif
                MPICommunicatorP2PKernels<ValueType, memorySpace>::
                  accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
                    d_sendRecvBufferSinglePrec,
                    d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
                    d_blockSize,
                    d_locallyOwnedSize,
                    d_ghostSize,
                    dataArray);
          }

#ifdef DFTFE_WITH_DEVICE
        if constexpr (memorySpace == MemorySpace::DEVICE)
          dftfe::utils::deviceStreamSynchronize(
            dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
#endif
		    d_accumulateInsertLocallyOwnedInFlight = false;
      }

      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      std::shared_ptr<const MPIPatternP2P<memorySpace>>
      MPICommunicatorP2P<ValueType, memorySpace>::getMPIPatternP2P() const
      {
        return d_mpiPatternP2P;
      }

      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      dftfe::Int
      MPICommunicatorP2P<ValueType, memorySpace>::getBlockSize() const
      {
        return d_blockSize;
      }

#ifdef DFTFE_WITH_DEVICE
      template class MPICommunicatorP2P<double,
                                        dftfe::utils::MemorySpace::DEVICE>;
      template class MPICommunicatorP2P<float,
                                        dftfe::utils::MemorySpace::DEVICE>;
      template class MPICommunicatorP2P<std::complex<double>,
                                        dftfe::utils::MemorySpace::DEVICE>;
      template class MPICommunicatorP2P<std::complex<float>,
                                        dftfe::utils::MemorySpace::DEVICE>;

      template class MPICommunicatorP2P<double,
                                        dftfe::utils::MemorySpace::HOST_PINNED>;
      template class MPICommunicatorP2P<float,
                                        dftfe::utils::MemorySpace::HOST_PINNED>;
      template class MPICommunicatorP2P<std::complex<double>,
                                        dftfe::utils::MemorySpace::HOST_PINNED>;
      template class MPICommunicatorP2P<std::complex<float>,
                                        dftfe::utils::MemorySpace::HOST_PINNED>;

#endif // DFTFE_WITH_DEVICE

      template class MPICommunicatorP2P<double,
                                        dftfe::utils::MemorySpace::HOST>;
      template class MPICommunicatorP2P<float, dftfe::utils::MemorySpace::HOST>;
      template class MPICommunicatorP2P<std::complex<double>,
                                        dftfe::utils::MemorySpace::HOST>;
      template class MPICommunicatorP2P<std::complex<float>,
                                        dftfe::utils::MemorySpace::HOST>;


    } // namespace mpi
  }   // namespace utils
} // namespace dftfe
