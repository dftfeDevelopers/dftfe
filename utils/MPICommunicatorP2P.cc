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
 * @author Sambit Das.
 */

#include <MPICommunicatorP2P.h>
#include <MPICommunicatorP2PKernels.h>
#include <MPITags.h>
#include <Exceptions.h>
#include <DeviceAPICalls.h>
#include <deviceDirectCCLWrapper.h>
namespace dftfe
{
  namespace utils
  {
    namespace mpi
    {
      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      MPICommunicatorP2P<ValueType, memorySpace>::MPICommunicatorP2P(
        std::shared_ptr<const MPIPatternP2P<memorySpace>> mpiPatternP2P,
        const size_type                                   blockSize)
        : d_mpiPatternP2P(mpiPatternP2P)
        , d_blockSize(blockSize)
        , d_locallyOwnedSize(mpiPatternP2P->localOwnedSize())
        , d_ghostSize(mpiPatternP2P->localGhostSize())
        , d_commPrecision(communicationPrecision::full)
      {
        d_commProtocol = communicationProtocol::mpiHost;
#if defined(DFTFE_WITH_DEVICE) && defined(DFTFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          d_commProtocol = communicationProtocol::mpiDevice;
#endif
#if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
        if (memorySpace == MemorySpace::DEVICE &&
            dftfe::utils::DeviceCCLWrapper::ncclCommInit)
          d_commProtocol = communicationProtocol::nccl;
#endif

        d_mpiCommunicator = d_mpiPatternP2P->mpiCommunicator();
        d_sendRecvBuffer.resize(
          d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size() *
            blockSize,
          0.0);

        d_requestsUpdateGhostValues.resize(
          d_mpiPatternP2P->getGhostProcIds().size() +
          d_mpiPatternP2P->getTargetProcIds().size());
        d_requestsAccumulateAddLocallyOwned.resize(
          d_mpiPatternP2P->getGhostProcIds().size() +
          d_mpiPatternP2P->getTargetProcIds().size());


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
            }
#endif
      }

      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::setCommunicationPrecision(
        communicationPrecision precision)
      {
        if (d_commPrecision == precision)
          return;
        d_commPrecision = precision;
        if (precision == communicationPrecision::full)
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
                  d_ghostDataCopyHostPinnedPtr = std::make_shared<
                    MemoryStorage<ValueType, MemorySpace::HOST_PINNED>>(
                    d_mpiPatternP2P->localGhostSize() * d_blockSize, 0.0);
                  d_sendRecvBufferHostPinnedPtr = std::make_shared<
                    MemoryStorage<ValueType, MemorySpace::HOST_PINNED>>(
                    d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs()
                        .size() *
                      d_blockSize,
                    0.0);
                }
#endif
          }
        if (precision == communicationPrecision::single)
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
                d_mpiPatternP2P->localGhostSize() * d_blockSize);
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::mpiHost)
                {
                  d_ghostDataCopySinglePrecHostPinnedPtr = std::make_shared<
                    MemoryStorage<typename dftfe::dataTypes::singlePrecType<
                                    ValueType>::type,
                                  MemorySpace::HOST_PINNED>>(
                    d_mpiPatternP2P->localGhostSize() * d_blockSize, 0.0);
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
      }

      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::updateGhostValues(
        MemoryStorage<ValueType, memorySpace> &dataArray,
        const size_type                        communicationChannel)
      {
        updateGhostValuesBegin(dataArray, communicationChannel);
        updateGhostValuesEnd(dataArray);
      }

      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::updateGhostValuesBegin(
        MemoryStorage<ValueType, memorySpace> &dataArray,
        const size_type                        communicationChannel)
      {
        // initiate non-blocking receives from ghost processors
        if (d_commPrecision == communicationPrecision::full)
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
            if (d_commProtocol != communicationProtocol::nccl)
              for (size_type i = 0;
                   i < (d_mpiPatternP2P->getGhostProcIds()).size();
                   ++i)
                {
                  const int err =
                    MPI_Irecv(recvArrayStartPtr,
                              (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                                 .data()[2 * i + 1] -
                               d_mpiPatternP2P->getGhostLocalIndicesRanges()
                                 .data()[2 * i]) *
                                d_blockSize * sizeof(ValueType),
                              MPI_BYTE,
                              d_mpiPatternP2P->getGhostProcIds().data()[i],
                              static_cast<size_type>(
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
                if (d_commProtocol != communicationProtocol::nccl)
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
              if (d_commProtocol == communicationProtocol::nccl)
                {
                  NCCLCHECK(ncclGroupStart());
                  for (size_type i = 0;
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
                          *dftfe::utils::DeviceCCLWrapper::ncclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      sendArrayStartPtr +=
                        d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                          .data()[i] *
                        d_blockSize;
                    }
                  for (size_type i = 0;
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
                          *dftfe::utils::DeviceCCLWrapper::ncclCommPtr,
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
            if (d_commProtocol != communicationProtocol::nccl)
              for (size_type i = 0;
                   i < (d_mpiPatternP2P->getTargetProcIds()).size();
                   ++i)
                {
                  const int err = MPI_Isend(
                    sendArrayStartPtr,
                    d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                        .data()[i] *
                      d_blockSize * sizeof(ValueType),
                    MPI_BYTE,
                    d_mpiPatternP2P->getTargetProcIds().data()[i],
                    static_cast<size_type>(
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
        else
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
            if (d_commProtocol != communicationProtocol::nccl)
              for (size_type i = 0;
                   i < (d_mpiPatternP2P->getGhostProcIds()).size();
                   ++i)
                {
                  const int err =
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
                              static_cast<size_type>(
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
                if (d_commProtocol != communicationProtocol::nccl)
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
              if (d_commProtocol == communicationProtocol::nccl)
                {
                  NCCLCHECK(ncclGroupStart());
                  for (size_type i = 0;
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
                          *dftfe::utils::DeviceCCLWrapper::ncclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      sendArrayStartPtr +=
                        d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                          .data()[i] *
                        d_blockSize;
                    }
                  for (size_type i = 0;
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
                          *dftfe::utils::DeviceCCLWrapper::ncclCommPtr,
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
            if (d_commProtocol != communicationProtocol::nccl)
              for (size_type i = 0;
                   i < (d_mpiPatternP2P->getTargetProcIds()).size();
                   ++i)
                {
                  const int err = MPI_Isend(
                    sendArrayStartPtr,
                    d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                        .data()[i] *
                      d_blockSize *
                      sizeof(typename dftfe::dataTypes::singlePrecType<
                             ValueType>::type),
                    MPI_BYTE,
                    d_mpiPatternP2P->getTargetProcIds().data()[i],
                    static_cast<size_type>(
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
      }


      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::updateGhostValuesEnd(
        MemoryStorage<ValueType, memorySpace> &dataArray)
      {
        // wait for all send and recv requests to be completed
#if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
        if constexpr (memorySpace == MemorySpace::DEVICE)
          if (d_commProtocol == communicationProtocol::nccl)
            dftfe::utils::deviceStreamSynchronize(
              dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
#endif
        if (d_requestsUpdateGhostValues.size() > 0)
          {
            if (d_commProtocol != communicationProtocol::nccl)
              {
                const int err = MPI_Waitall(d_requestsUpdateGhostValues.size(),
                                            d_requestsUpdateGhostValues.data(),
                                            MPI_STATUSES_IGNORE);
                std::string errMsg = "Error occured while using MPI_Waitall. "
                                     "Error code: " +
                                     std::to_string(err);
                throwException(err == MPI_SUCCESS, errMsg);
              }
          }
        if (d_commPrecision == communicationPrecision::full)
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
        else
          {
#ifdef DFTFE_WITH_DEVICE
            if constexpr (memorySpace == MemorySpace::DEVICE)
              if (d_commProtocol == communicationProtocol::mpiHost)
                {
                  MemoryTransfer<memorySpace, MemorySpace::HOST_PINNED>
                    memoryTransfer;
                  if (d_ghostDataCopyHostPinnedPtr->size() > 0)
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
      }


      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::accumulateAddLocallyOwned(
        MemoryStorage<ValueType, memorySpace> &dataArray,
        const size_type                        communicationChannel)
      {
        accumulateAddLocallyOwnedBegin(dataArray, communicationChannel);
        accumulateAddLocallyOwnedEnd(dataArray);
      }

      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::
        accumulateAddLocallyOwnedBegin(
          MemoryStorage<ValueType, memorySpace> &dataArray,
          const size_type                        communicationChannel)
      {
        if (d_commPrecision == communicationPrecision::full)
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
            if (d_commProtocol != communicationProtocol::nccl)
              for (size_type i = 0;
                   i < (d_mpiPatternP2P->getTargetProcIds()).size();
                   ++i)
                {
                  const int err =
                    MPI_Irecv(recvArrayStartPtr,
                              d_mpiPatternP2P
                                  ->getNumOwnedIndicesForTargetProcs()
                                  .data()[i] *
                                d_blockSize * sizeof(ValueType),
                              MPI_BYTE,
                              d_mpiPatternP2P->getTargetProcIds().data()[i],
                              static_cast<size_type>(
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
              if (d_commProtocol == communicationProtocol::nccl)
                {
                  NCCLCHECK(ncclGroupStart());
                  for (size_type i = 0;
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
                          *dftfe::utils::DeviceCCLWrapper::ncclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      sendArrayStartPtr +=
                        (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i + 1] -
                         d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i]) *
                        d_blockSize;
                    }
                  for (size_type i = 0;
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
                          *dftfe::utils::DeviceCCLWrapper::ncclCommPtr,
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
            if (d_commProtocol != communicationProtocol::nccl)
              for (size_type i = 0;
                   i < (d_mpiPatternP2P->getGhostProcIds()).size();
                   ++i)
                {
                  const int err = MPI_Isend(
                    sendArrayStartPtr,
                    (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i + 1] -
                     d_mpiPatternP2P->getGhostLocalIndicesRanges()
                       .data()[2 * i]) *
                      d_blockSize * sizeof(ValueType),
                    MPI_BYTE,
                    d_mpiPatternP2P->getGhostProcIds().data()[i],
                    static_cast<size_type>(
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
            if (d_commProtocol != communicationProtocol::nccl)
              for (size_type i = 0;
                   i < (d_mpiPatternP2P->getTargetProcIds()).size();
                   ++i)
                {
                  const int err = MPI_Irecv(
                    recvArrayStartPtr,
                    d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs()
                        .data()[i] *
                      d_blockSize *
                      sizeof(typename dftfe::dataTypes::singlePrecType<
                             ValueType>::type),
                    MPI_BYTE,
                    d_mpiPatternP2P->getTargetProcIds().data()[i],
                    static_cast<size_type>(
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
                if (d_commProtocol != communicationProtocol::nccl)
                  dftfe::utils::deviceStreamSynchronize(
                    dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
                if (d_commProtocol == communicationProtocol::mpiHost)
                  {
                    MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
                      memoryTransfer;
                    if (d_ghostDataCopyHostPinnedPtr->size() > 0)
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
              if (d_commProtocol == communicationProtocol::nccl)
                {
                  NCCLCHECK(ncclGroupStart());
                  for (size_type i = 0;
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
                          *dftfe::utils::DeviceCCLWrapper::ncclCommPtr,
                          dftfe::utils::DeviceCCLWrapper::d_deviceCommStream));

                      sendArrayStartPtr +=
                        (d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i + 1] -
                         d_mpiPatternP2P->getGhostLocalIndicesRanges()
                           .data()[2 * i]) *
                        d_blockSize;
                    }
                  for (size_type i = 0;
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
                          *dftfe::utils::DeviceCCLWrapper::ncclCommPtr,
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
            if (d_commProtocol != communicationProtocol::nccl)
              for (size_type i = 0;
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
                      sizeof(typename dftfe::dataTypes::singlePrecType<
                             ValueType>::type),
                    MPI_BYTE,
                    d_mpiPatternP2P->getGhostProcIds().data()[i],
                    static_cast<size_type>(
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
      }


      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::accumulateAddLocallyOwnedEnd(
        MemoryStorage<ValueType, memorySpace> &dataArray)
      {
        // wait for all send and recv requests to be completed
#if defined(DFTFE_WITH_CUDA_NCCL) || defined(DFTFE_WITH_HIP_RCCL)
        if constexpr (memorySpace == MemorySpace::DEVICE)
          if (d_commProtocol == communicationProtocol::nccl)
            dftfe::utils::deviceStreamSynchronize(
              dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
#endif
        if (d_requestsAccumulateAddLocallyOwned.size() > 0)
          {
            if (d_commProtocol != communicationProtocol::nccl)
              {
                const int err =
                  MPI_Waitall(d_requestsAccumulateAddLocallyOwned.size(),
                              d_requestsAccumulateAddLocallyOwned.data(),
                              MPI_STATUSES_IGNORE);

                std::string errMsg = "Error occured while using MPI_Waitall. "
                                     "Error code: " +
                                     std::to_string(err);
                throwException(err == MPI_SUCCESS, errMsg);
              }
          }
        if (d_commPrecision == communicationPrecision::full)
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
#ifdef DFTFE_WITH_DEVICE
        if constexpr (memorySpace == MemorySpace::DEVICE)
          dftfe::utils::deviceStreamSynchronize(
            dftfe::utils::DeviceCCLWrapper::d_deviceCommStream);
#endif
      }

      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      std::shared_ptr<const MPIPatternP2P<memorySpace>>
      MPICommunicatorP2P<ValueType, memorySpace>::getMPIPatternP2P() const
      {
        return d_mpiPatternP2P;
      }

      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      int
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
