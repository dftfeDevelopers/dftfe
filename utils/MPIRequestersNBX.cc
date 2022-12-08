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
 * @author Bikash Kanungo
 */

#include <MPIRequestersNBX.h>
#include <MPITags.h>
#include <Exceptions.h>
#include <string>
namespace dftfe
{
  namespace utils
  {
    namespace mpi
    {
      MPIRequestersNBX::MPIRequestersNBX(
        const std::vector<size_type> &targetIDs,
        const MPI_Comm &               comm)
        : d_targetIDs(targetIDs)
        , d_comm(comm)
        , d_recvBuffers(0)
        , d_recvRequests(0)
      {
        d_myRank           = 0;
        d_numProcessors    = 1;
        int         err    = MPI_Comm_size(d_comm, &d_numProcessors);
        std::string errMsg = "Error occured while using MPI_Comm_size. "
                             "Error code: " +
                             std::to_string(err);
        throwException(err == MPI_SUCCESS, errMsg);

        err    = MPI_Comm_rank(d_comm, &d_myRank);
        errMsg = "Error occured while using MPI_Comm_rank. "
                 "Error code: " +
                 std::to_string(err);
        throwException(err == MPI_SUCCESS, errMsg);
      }

      std::vector<size_type>
      MPIRequestersNBX::getRequestingRankIds()
      {
        startLocalSend();

        while (haveAllLocalSendReceived() == false)
          probeAndReceiveIncomingMsg();

        signalLocalSendCompletion();

        while (haveAllIncomingMsgsReceived() == false)
          probeAndReceiveIncomingMsg();

        finish();

        return std::vector<size_type>(d_requestingProcesses.begin(),
                                      d_requestingProcesses.end());
      }

      void
      MPIRequestersNBX::startLocalSend()
      {
        const size_type numTargets = d_targetIDs.size();
        const int       tag = static_cast<int>(MPITags::MPI_REQUESTERS_NBX_TAG);

        d_sendRequests.resize(numTargets);
        d_sendBuffers.resize(numTargets);
        for (unsigned int i = 0; i < numTargets; ++i)
          {
            const unsigned int rank = d_targetIDs[i];
            throwException<DomainError>(
              rank < d_numProcessors,
              "Target rank " + std::to_string(rank) +
                " is outside the range of number of processors(i.e., " +
                std::to_string(d_numProcessors) + ")");

            int &sendBuffer = d_sendBuffers[i];
            auto err        = MPI_Issend(
              &sendBuffer, 1, MPI_INT, rank, tag, d_comm, &d_sendRequests[i]);

            std::string errMsg = "Error occured while using MPI_Issend. "
                                 "Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }
      }

      void
      MPIRequestersNBX::probeAndReceiveIncomingMsg()
      {
        const int tag = static_cast<int>(MPITags::MPI_REQUESTERS_NBX_TAG);

        // Check if there is an incoming message to be received.
        // If yes, extract the source rank and then receive the
        // message i
        MPI_Status status;
        int       foundIncomingMsg;
        int       err =
          MPI_Iprobe(MPI_ANY_SOURCE, tag, d_comm, &foundIncomingMsg, &status);
        std::string errMsg = "Error occured while using MPI_Iprobe. "
                             "Error code: " +
                             std::to_string(err);
        throwException(err == MPI_SUCCESS, errMsg);

        if (foundIncomingMsg != 0)
          {
            // Get the rank of the source process
            // and add it to the set of requesting processes
            const auto sourceRank = status.MPI_SOURCE;

            //
            // Check if the source process has already sent message.
            // It is supposed to send message only once
            //
            bool hasRankAlreadySent = (d_requestingProcesses.find(sourceRank) !=
                                       d_requestingProcesses.end());
            errMsg =
              "Process " + std::to_string(sourceRank) +
              " is sending message to " + std::to_string(d_myRank) +
              " second time.\n"
              "The NBX algorithm is designed to receive at most one incoming"
              " message from any source process.";
            throwException(hasRankAlreadySent == false, errMsg);
            d_requestingProcesses.insert(sourceRank);

            //
            // get the current size of receive buffers
            //
            size_type N = d_recvBuffers.size();

            //
            // increase the size of receive buffers and
            // receive requests by 1 to allocate memory
            // for this found incoming message
            //
            int dummyVal = 0;
            d_recvBuffers.push_back(dummyVal);
            MPI_Request request;
            d_recvRequests.push_back(request);

            err    = MPI_Irecv(&d_recvBuffers[N],
                                              1,
                                              MPI_INT,
                                              sourceRank,
                                              tag,
                                              d_comm,
                                              &d_recvRequests[N]);
            errMsg = "Error occured while using MPI_Irecv. "
                     "Error code: " +
                     std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }
      }

      bool
      MPIRequestersNBX::haveAllLocalSendReceived()
      {
        if (d_sendRequests.size() > 0)
          {
            int         allLocalSendCompletedFlag;
            const auto  err    = MPI_Testall(d_sendRequests.size(),
                                        d_sendRequests.data(),
                                        &allLocalSendCompletedFlag,
                                        MPI_STATUSES_IGNORE);
            std::string errMsg = "Error occured while using MPI_Testall. "
                                 " Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);

            return allLocalSendCompletedFlag != 0;
          }
        else
          return true;
      }

      void
      MPIRequestersNBX::signalLocalSendCompletion()
      {
        const auto  err    = MPI_Ibarrier(d_comm, &d_barrierRequest);
        std::string errMsg = "Error occured while using MPI_Ibarrier. "
                             " Error code: " +
                             std::to_string(err);
        throwException(err == MPI_SUCCESS, errMsg);
      }

      bool
      MPIRequestersNBX::haveAllIncomingMsgsReceived()
      {
        int         allProcessorsInvokedIBarrier;
        const auto  err    = MPI_Test(&d_barrierRequest,
                                 &allProcessorsInvokedIBarrier,
                                 MPI_STATUSES_IGNORE);
        std::string errMsg = "Error occured while using MPI_Test. "
                             " Error code: " +
                             std::to_string(err);
        throwException(err == MPI_SUCCESS, errMsg);
        return allProcessorsInvokedIBarrier != 0;
      }

      void
      MPIRequestersNBX::finish()
      {
        if (d_sendRequests.size() > 0)
          {
            const int   err    = MPI_Waitall(d_sendRequests.size(),
                                       d_sendRequests.data(),
                                       MPI_STATUSES_IGNORE);
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 " Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        if (d_recvRequests.size() > 0)
          {
            const int   err    = MPI_Waitall(d_recvRequests.size(),
                                       d_recvRequests.data(),
                                       MPI_STATUSES_IGNORE);
            std::string errMsg = "Error occured while using MPI_Waitall. "
                                 " Error code: " +
                                 std::to_string(err);
            throwException(err == MPI_SUCCESS, errMsg);
          }

        int         err    = MPI_Wait(&d_barrierRequest, MPI_STATUS_IGNORE);
        std::string errMsg = "Error occured while using MPI_Wait. "
                             " Error code: " +
                             std::to_string(err);
        throwException(err == MPI_SUCCESS, errMsg);

#  ifndef NDEBUG
        // note: MPI_Ibarrier seems to make problem during testing, this
        // additional Barrier seems to help
        err    = MPI_Barrier(d_comm);
        errMsg = "Error occured while using MPI_Barrier. "
                 " Error code: " +
                 std::to_string(err);
        throwException(err == MPI_SUCCESS, errMsg);
#  endif
      }
    } // namespace mpi
  }   // namespace utils
} // namespace dftfe
