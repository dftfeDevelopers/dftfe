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
      {
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


#if defined(DFTFE_WITH_DEVICE) && !defined(DFTFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          {
            d_ghostDataCopyHostPinnedPtr= std::make_shared<MemoryStorage<ValueType, MemorySpace::HOST_PINNED>>(d_mpiPatternP2P->localGhostSize() *
                                               blockSize,
                                             0.0);
            d_sendRecvBufferHostPinnedPtr=std::make_shared<MemoryStorage<ValueType, MemorySpace::HOST_PINNED>>(
              d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size() *
                blockSize,
              0.0);
          }
#endif // defined(DFTFE_WITH_DEVICE) && !defined(DFTFE_WITH_DEVICE_AWARE_MPI)

#if defined(DFTFE_WITH_DEVICE)
        if (memorySpace == MemorySpace::DEVICE)
          {
            if (std::is_same<ValueType, std::complex<double>>::value)
              {
                d_tempDoubleRealArrayForAtomics.resize(
                  (d_locallyOwnedSize + d_ghostSize) * d_blockSize, 0);
                d_tempDoubleImagArrayForAtomics.resize(
                  (d_locallyOwnedSize + d_ghostSize) * d_blockSize, 0);
              }
            else if (std::is_same<ValueType, std::complex<float>>::value)
              {
                d_tempFloatRealArrayForAtomics.resize(
                  (d_locallyOwnedSize + d_ghostSize) * d_blockSize, 0);
                d_tempFloatImagArrayForAtomics.resize(
                  (d_locallyOwnedSize + d_ghostSize) * d_blockSize, 0);
              }
          }
#endif
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
        // d_requestsUpdateGhostValues.resize(
        //  d_mpiPatternP2P->getGhostProcIds().size() +
        //  d_mpiPatternP2P->getTargetProcIds().size());

        // initiate non-blocking receives from ghost processors
        ValueType *recvArrayStartPtr =
          dataArray.data() + d_mpiPatternP2P->localOwnedSize() * d_blockSize;

#if defined(DFTFE_WITH_DEVICE) && !defined(DFTFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          recvArrayStartPtr = d_ghostDataCopyHostPinnedPtr->begin();
#endif // defined(DFTFE_WITH_DEVICE) &&
       // !defined(DFTFE_WITH_DEVICE_AWARE_MPI)

#if defined(DFTFE_WITH_DEVICE) && defined(DFTFE_WITH_DEVICE_AWARE_MPI)
        dftfe::utils::deviceSynchronize();
#endif
        for (size_type i = 0; i < (d_mpiPatternP2P->getGhostProcIds()).size();
             ++i)
          {
            const int err = MPI_Irecv(
              recvArrayStartPtr,
              (d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i + 1] -
               d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i]) *
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
              (d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i + 1] -
               d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i]) *
              d_blockSize;
          }

        // gather locally owned entries into a contiguous send buffer
        if ((d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size()) > 0)
          MPICommunicatorP2PKernels<ValueType, memorySpace>::
            gatherLocallyOwnedEntriesSendBufferToTargetProcs(
              dataArray,
              d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
              d_blockSize,
              d_sendRecvBuffer);

        // initiate non-blocking sends to target processors
        ValueType *sendArrayStartPtr = d_sendRecvBuffer.data();

#if defined(DFTFE_WITH_DEVICE) && !defined(DFTFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          {
            MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
              memoryTransfer;

            if (d_sendRecvBufferHostPinnedPtr->size() > 0)
              memoryTransfer.copy(d_sendRecvBufferHostPinnedPtr->size(),
                                  d_sendRecvBufferHostPinnedPtr->begin(),
                                  d_sendRecvBuffer.begin());

            sendArrayStartPtr = d_sendRecvBufferHostPinnedPtr->begin();
          }
#endif // defined(DFTFE_WITH_DEVICE) &&
       // !defined(DFTFE_WITH_DEVICE_AWARE_MPI)

#if defined(DFTFE_WITH_DEVICE) && defined(DFTFE_WITH_DEVICE_AWARE_MPI)
        dftfe::utils::deviceSynchronize();
#endif

        for (size_type i = 0; i < (d_mpiPatternP2P->getTargetProcIds()).size();
             ++i)
          {
            const int err = MPI_Isend(
              sendArrayStartPtr,
              d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs().data()[i] *
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
              d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs().data()[i] *
              d_blockSize;
          }
      }


      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::updateGhostValuesEnd(
        MemoryStorage<ValueType, memorySpace> &dataArray)
      {
        // wait for all send and recv requests to be completed
        if (d_requestsUpdateGhostValues.size() > 0)
          {
            const int   err    = MPI_Waitall(d_requestsUpdateGhostValues.size(),
                                        d_requestsUpdateGhostValues.data(),
                                        MPI_STATUSES_IGNORE);
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);

#if defined(DFTFE_WITH_DEVICE) && !defined(DFTFE_WITH_DEVICE_AWARE_MPI)
            if (memorySpace == MemorySpace::DEVICE)
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
#endif // defined(DFTFE_WITH_DEVICE) &&
       // !defined(DFTFE_WITH_DEVICE_AWARE_MPI)

#if defined(DFTFE_WITH_DEVICE) && defined(DFTFE_WITH_DEVICE_AWARE_MPI)
            dftfe::utils::deviceSynchronize();
#endif
          }
        // d_requestsUpdateGhostValues.resize(0);
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
        // d_requestsAccumulateAddLocallyOwned.resize(
        //  d_mpiPatternP2P->getGhostProcIds().size() +
        //  d_mpiPatternP2P->getTargetProcIds().size());

        // initiate non-blocking receives from target processors
        ValueType *recvArrayStartPtr = d_sendRecvBuffer.data();
#if defined(DFTFE_WITH_DEVICE) && !defined(DFTFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          recvArrayStartPtr = d_sendRecvBufferHostPinnedPtr->begin();
#endif // defined(DFTFE_WITH_DEVICE) &&
       // !defined(DFTFE_WITH_DEVICE_AWARE_MPI)

#if defined(DFTFE_WITH_DEVICE) && defined(DFTFE_WITH_DEVICE_AWARE_MPI)
        dftfe::utils::deviceSynchronize();
#endif
        for (size_type i = 0; i < (d_mpiPatternP2P->getTargetProcIds()).size();
             ++i)
          {
            const int err = MPI_Irecv(
              recvArrayStartPtr,
              d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs().data()[i] *
                d_blockSize * sizeof(ValueType),
              MPI_BYTE,
              d_mpiPatternP2P->getTargetProcIds().data()[i],
              static_cast<size_type>(MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
                communicationChannel,
              d_mpiCommunicator,
              &d_requestsAccumulateAddLocallyOwned[i]);

            std::string errMsg = "Error occured while using MPI_Irecv. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);


            recvArrayStartPtr +=
              d_mpiPatternP2P->getNumOwnedIndicesForTargetProcs().data()[i] *
              d_blockSize;
          }



        // initiate non-blocking sends to ghost processors
        ValueType *sendArrayStartPtr =
          dataArray.data() + d_mpiPatternP2P->localOwnedSize() * d_blockSize;

#if defined(DFTFE_WITH_DEVICE) && !defined(DFTFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
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
#endif // defined(DFTFE_WITH_DEVICE) &&
       // !defined(DFTFE_WITH_DEVICE_AWARE_MPI)

#if defined(DFTFE_WITH_DEVICE) && defined(DFTFE_WITH_DEVICE_AWARE_MPI)
        dftfe::utils::deviceSynchronize();
#endif

        for (size_type i = 0; i < (d_mpiPatternP2P->getGhostProcIds()).size();
             ++i)
          {
            const int err = MPI_Isend(
              sendArrayStartPtr,
              (d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i + 1] -
               d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i]) *
                d_blockSize * sizeof(ValueType),
              MPI_BYTE,
              d_mpiPatternP2P->getGhostProcIds().data()[i],
              static_cast<size_type>(MPITags::MPI_P2P_COMMUNICATOR_GATHER_TAG) +
                communicationChannel,
              d_mpiCommunicator,
              &d_requestsAccumulateAddLocallyOwned
                [(d_mpiPatternP2P->getTargetProcIds()).size() + i]);


            std::string errMsg = "Error occured while using MPI_Isend. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);

            sendArrayStartPtr +=
              (d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i + 1] -
               d_mpiPatternP2P->getGhostLocalIndicesRanges().data()[2 * i]) *
              d_blockSize;
          }
      }


      template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
      void
      MPICommunicatorP2P<ValueType, memorySpace>::accumulateAddLocallyOwnedEnd(
        MemoryStorage<ValueType, memorySpace> &dataArray)
      {
        // wait for all send and recv requests to be completed
        if (d_requestsAccumulateAddLocallyOwned.size() > 0)
          {
            const int err =
              MPI_Waitall(d_requestsAccumulateAddLocallyOwned.size(),
                          d_requestsAccumulateAddLocallyOwned.data(),
                          MPI_STATUSES_IGNORE);

            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);

#if defined(DFTFE_WITH_DEVICE) && !defined(DFTFE_WITH_DEVICE_AWARE_MPI)
            if (memorySpace == MemorySpace::DEVICE)
              {
                MemoryTransfer<memorySpace, MemorySpace::HOST_PINNED>
                  memoryTransfer;
                if (d_sendRecvBufferHostPinnedPtr->size() > 0)
                  memoryTransfer.copy(d_sendRecvBufferHostPinnedPtr->size(),
                                      d_sendRecvBuffer.data(),
                                      d_sendRecvBufferHostPinnedPtr->data());
              }
#endif // defined(DFTFE_WITH_DEVICE) &&
       // !defined(DFTFE_WITH_DEVICE_AWARE_MPI)

#if defined(DFTFE_WITH_DEVICE) && defined(DFTFE_WITH_DEVICE_AWARE_MPI)
            dftfe::utils::deviceSynchronize();
#endif
            // accumulate add into locally owned entries from recv buffer
            if ((d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size()) >
                0)
              MPICommunicatorP2PKernels<ValueType, memorySpace>::
                accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
                  d_sendRecvBuffer,
                  d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
                  d_blockSize,
                  d_locallyOwnedSize,
                  d_ghostSize,
                  d_tempDoubleRealArrayForAtomics,
                  d_tempDoubleImagArrayForAtomics,
                  d_tempFloatRealArrayForAtomics,
                  d_tempFloatImagArrayForAtomics,
                  dataArray);
          }
        // d_requestsAccumulateAddLocallyOwned.resize(0);
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
