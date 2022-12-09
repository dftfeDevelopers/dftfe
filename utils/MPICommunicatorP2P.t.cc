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
            d_ghostDataCopyHostPinned.resize(d_mpiPatternP2P->localGhostSize() *
                                               blockSize,
                                             0.0);
            d_sendRecvBufferHostPinned.resize(
              d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size() *
                blockSize,
              0.0);

            if (std::is_same<ValueType, std::complex<double>>::value || std::is_same<ValueType, std::complex<float>>::value)
            {
              d_tempRealArrayForAtomics.resize((d_locallyOwnedSize+d_ghostSize)*d_blockSize,0);
              d_tempImagArrayForAtomics.resize((d_locallyOwnedSize+d_ghostSize)*d_blockSize,0);
            }
          }
#endif // defined(DFTFE_WITH_DEVICE) && !defined(DFTFE_WITH_DEVICE_AWARE_MPI)
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
        //d_requestsUpdateGhostValues.resize(
        //  d_mpiPatternP2P->getGhostProcIds().size() +
        //  d_mpiPatternP2P->getTargetProcIds().size());

        // initiate non-blocking receives from ghost processors
        ValueType *recvArrayStartPtr =
          dataArray.begin() + d_mpiPatternP2P->localOwnedSize() * d_blockSize;

#  if defined(DFTFE_WITH_DEVICE) && !defined(DFTFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          recvArrayStartPtr = d_ghostDataCopyHostPinned.begin();
#  endif // defined(DFTFE_WITH_DEVICE) &&
         // !defined(DFTFE_WITH_DEVICE_AWARE_MPI)

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
        if ((d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size())>0)
        MPICommunicatorP2PKernels<ValueType, memorySpace>::
          gatherLocallyOwnedEntriesSendBufferToTargetProcs(
            dataArray,
            d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
            d_blockSize,
            d_sendRecvBuffer);

        // initiate non-blocking sends to target processors
        ValueType *sendArrayStartPtr = d_sendRecvBuffer.begin();

#  if defined(DFTFE_WITH_DEVICE) && !defined(DFTFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          {
            MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
              memoryTransfer;

            if (d_sendRecvBufferHostPinned.size()>0)
            memoryTransfer.copy(d_sendRecvBufferHostPinned.size(),
                                d_sendRecvBufferHostPinned.begin(),
                                d_sendRecvBuffer.begin());

            sendArrayStartPtr = d_sendRecvBufferHostPinned.begin();
          }
#  endif // defined(DFTFE_WITH_DEVICE) &&
         // !defined(DFTFE_WITH_DEVICE_AWARE_MPI)

        for (size_type i = 0; i < (d_mpiPatternP2P->getTargetProcIds()).size();
             ++i)
          {
#  if defined(DFTFE_WITH_DEVICE) && defined(DFTFE_WITH_DEVICE_AWARE_MPI)
            //FIXME: Is this required?
            dftfe::utils::deviceSynchronize();
#endif            
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
            const int err = MPI_Waitall(d_requestsUpdateGhostValues.size(),
                                       d_requestsUpdateGhostValues.data(),
                                       MPI_STATUSES_IGNORE);
            std::string errMsg = "Error occured while using MPI_Waitall. "
                             "Error code: " +
                             std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);

#  if defined(DFTFE_WITH_DEVICE) && !defined(DFTFE_WITH_DEVICE_AWARE_MPI)
            if (memorySpace == MemorySpace::DEVICE)
              {
                MemoryTransfer<memorySpace, MemorySpace::HOST_PINNED>
                  memoryTransfer;
                if ( d_ghostDataCopyHostPinned.size()>0)
                memoryTransfer.copy(d_ghostDataCopyHostPinned.size(),
                                    dataArray.begin() +
                                      d_mpiPatternP2P->localOwnedSize() *
                                        d_blockSize,
                                    d_ghostDataCopyHostPinned.data());
              }
#  endif // defined(DFTFE_WITH_DEVICE) &&
         // !defined(DFTFE_WITH_DEVICE_AWARE_MPI)
          }
        //d_requestsUpdateGhostValues.resize(0);
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
        //d_requestsAccumulateAddLocallyOwned.resize(
        //  d_mpiPatternP2P->getGhostProcIds().size() +
        //  d_mpiPatternP2P->getTargetProcIds().size()); 

        // initiate non-blocking receives from target processors
        ValueType *recvArrayStartPtr = d_sendRecvBuffer.begin();
#  if defined(DFTFE_WITH_DEVICE) && !defined(DFTFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          recvArrayStartPtr = d_sendRecvBufferHostPinned.begin();
#  endif // defined(DFTFE_WITH_DEVICE) &&
         // !defined(DFTFE_WITH_DEVICE_AWARE_MPI)

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
          dataArray.begin() + d_mpiPatternP2P->localOwnedSize() * d_blockSize;

#  if defined(DFTFE_WITH_DEVICE) && !defined(DFTFE_WITH_DEVICE_AWARE_MPI)
        if (memorySpace == MemorySpace::DEVICE)
          {
            MemoryTransfer<MemorySpace::HOST_PINNED, memorySpace>
              memoryTransfer;
            if (d_ghostDataCopyHostPinned.size()>0)
            memoryTransfer.copy(d_ghostDataCopyHostPinned.size(),
                                d_ghostDataCopyHostPinned.begin(),
                                dataArray.begin() +
                                  d_mpiPatternP2P->localOwnedSize() *
                                    d_blockSize);

            sendArrayStartPtr = d_ghostDataCopyHostPinned.begin();
          }
#  endif // defined(DFTFE_WITH_DEVICE) &&
         // !defined(DFTFE_WITH_DEVICE_AWARE_MPI)

        for (size_type i = 0; i < (d_mpiPatternP2P->getGhostProcIds()).size();
             ++i)
          {
#  if defined(DFTFE_WITH_DEVICE) && defined(DFTFE_WITH_DEVICE_AWARE_MPI)
            //FIXME: Is this required?
            dftfe::utils::deviceSynchronize();
#endif            
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

#  if defined(DFTFE_WITH_DEVICE) && !defined(DFTFE_WITH_DEVICE_AWARE_MPI)
            if (memorySpace == MemorySpace::DEVICE)
              {
                MemoryTransfer<memorySpace,MemorySpace::HOST_PINNED>
                  memoryTransfer;
                if (d_sendRecvBufferHostPinned.size()>0)
                memoryTransfer.copy(d_sendRecvBufferHostPinned.size(),
                                    d_sendRecvBuffer.data(),
                                    d_sendRecvBufferHostPinned.data());
              }
#  endif // defined(DFTFE_WITH_DEVICE) &&
         // !defined(DFTFE_WITH_DEVICE_AWARE_MPI)

        // accumulate add into locally owned entries from recv buffer
        if ((d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs().size())>0)
        MPICommunicatorP2PKernels<ValueType, memorySpace>::
          accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
            d_sendRecvBuffer,
            d_mpiPatternP2P->getOwnedLocalIndicesForTargetProcs(),
            d_blockSize,
            d_locallyOwnedSize,
            d_ghostSize,
            d_tempRealArrayForAtomics,
            d_tempImagArrayForAtomics,
            dataArray);

          }
         //d_requestsAccumulateAddLocallyOwned.resize(0); 
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

    } // namespace mpi
  }   // namespace utils
} // namespace dftfe
