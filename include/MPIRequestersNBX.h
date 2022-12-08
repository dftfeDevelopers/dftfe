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

#ifndef dftfeMPIRequestersNBX_h
#define dftfeMPIRequestersNBX_h

#  include <mpi.h>
#include <TypeConfig.h>
#include <MPIRequestersBase.h>
#include <vector>
#include <set>
namespace dftfe
{
  namespace utils
  {
    namespace mpi
    {
      class MPIRequestersNBX : public MPIRequestersBase
      {
        /*
         * @brief Implements the Non-blocking Consensus (NBX) algorithm as
         * described in the following paper to determine the list of requesting
         * processors for the current processors
         * @article{hoefler2010scalable,
         *   title={Scalable communication protocols for dynamic sparse data
         * exchange}, author={Hoefler, Torsten and Siebert, Christian and
         * Lumsdaine, Andrew}, journal={ACM Sigplan Notices}, volume={45},
         *   number={5},
         *   pages={159--168},
         *   year={2010},
         *   publisher={ACM New York, NY, USA}
         * }
         */

        /*
         * The following is a brief description of the typical use case
         * situation. Each processor has a list of target processors to which it
         * wants to send a message (think of it as a message to another
         * processor to request some data that is owned by the other processor).
         * Similarly, other processors might be requesting the current processor
         * for some of the data owned by the current processor. However, the
         * current processor has no apriori knowledge of which processors will
         * be requesting data from it. The challenge is to utilize the current
         * processor's list of target processors to determine the current
         * processor's requesting processors. In other words, we have to use a
         * one way communication information to figure out the other way (its
         * dual).
         *
         * Perhaps a more concrete example might help. Let's say, we have a
         * vector/array which is distributed across a set of processors.
         * Each processors own part of the vector. The ownership is exclusive,
         * i.e., a processor is the sole owner of its part of the vector.
         * In practice, it means that the processor owns a set of indices of the
         * vector. Additionally, the different sets of owning indices across all
         * the processors are disjoint. Moreover, the union of the sets across
         * all the processors gives the set of indices of the distributed
         * vector. However, each processor also needs information on a set of
         * non-owned indices (hereafter termed ghost indices) based on the needs
         * of the application. Based on the ghost indices, the current processor
         * can easily determine the processor where it is owned. These
         * processors are termed as target processors to which the current
         * processor has to send a request to access the ghost data. Similarly,
         * the ghost indices in some other processor might be owned by this
         * processor. In that case, the other processor will be sending a
         * request to the current processor to access some of its data (data
         * which is ghost to the other processor but owned by the current
         * processor). But the current processor has no apriori knowledge of
         * which processors will be requesting data from it. A knowledge of it
         * will help the current processor to prepare for the request of data.
         *
         * In cases of sparse communication, that is, where each processor only
         * needs to communicate with a small subset of the total number of
         * processors, the NBX algorithm offers an algorithm of complexity
         * O(log(P)) (where P is the number of processors) to determing the
         * list of requesting processors. The algorithm works as follows:
         *
         * 1. The current processor sends  nonblocking synchronous message
         * (i.e., MPI_ISsend) to all its target processors. Remember that
         * the current processor already has information about its target
         * processors. Also, note that the status of the  nonblocking
         * synchronous send turns to "completed" only when a when
         * the message has been received by a receiving processor. Let's
         * call this operation as the "local-send", as we are sending
         * requests to target processors that are locally known by the current
         * processor.
         *
         * 2. The current processor keeps on doing nonblocking probe for
         * incoming message (i.e., MPI_IProbe). The MPI_IProbe checks if there
         * is an incoming message matching a given source and tag or not. The
         * source is the index of the source processor sending the message and
         * tag is an MPI_tag associated with exchange. It does not initiate any
         * receive operation , it only verfies whether there is something to be
         * received or not. For our purpose, we will use a wildcards
         * MPI_ANY_SOURCE and MPI_ANY_TAG, as we just want to know if there is
         * an incoming message or not. In the event that there is an incoming
         * message (i.e., the MPI_IProbe's flag is true), we can extract the
         * source processor from the status handle of the MPI_IProbe and append
         * it to a list that stores the requesting processor IDs. Addtionally,
         * in the event that there is an incoming messag, we call a non-blocking
         * receive (i.e., MPI_IRecv) to initiate the actual
         * reception of the incoming. The MPI_Recv, in turn, will complete
         * the status of source processor's MPI_ISsend through which the
         * incoming message was sent to the current processor. Thus, we
         * achieve two things over here: we detected a requesting processor
         * and we also signaled the requesting processor that we have received
         * their message. But this is only job half-done. How do we tell the
         * current processor to stop probing for incoming message? And how do
         * inform all the processors involved that all the incoming messages
         * across all the processors have been received? This kind of problem
         * is what is called a Consensus Problem
         * (https://en.wikipedia.org/wiki/Consensus_(computer_science)).
         * The way to reach the consensus in NBX is a two-step process:
         * (a) the current processor checks if all the "local-send"
         *     (see #1 above) has been received or not.
         *     That is, if the status handle of all its MPI_ISsend have turned
         *     to completed or not. If all the local"local-send" have been
         *     completed, we initiate a non-blocking barrier (i.e.,
         * MPI_IBarrier) on the current processor. This informs the network that
         * the current processor has witnessed its part of an event (in this
         * case the event is the completion of all its "local-send"). (b) the
         * above only informs the network that the all "local-send" of the
         *     current processor have been received. But the current processor
         * can still have incoming messages to be receieved. Hence, the current
         *     processor keeps on probing and receiving incoming messages, until
         *     the non-blocking barrier (MPI_IBarrier) (as mentioned
         *     above in (a)) has been invoked by all the processors. This can be
         *     checked from the status handle of the MPI_IBarrier, which
         *     completes only when all processors call it.
         *     At a stage when the status of MPI_IBarrier turns to completed,
         *     we know for sure that all the "local-send" of all
         *     the processors have been received and that there are no more
         *     incoming messages in any processor to be received. Thus, we
         *     can now safely terminate the nonblocking probe on all processors.
         *
         *
         *
         * @note: Since we are only interested in knowing the requesting
         * processors for the current processor, we only need token
         * MPI sends and receives (e.g., just an integer across) instead
         * of large chunks of data. To that end, we harcode all the send
         * and receive buffers to be of integer type
         */

      public:
        MPIRequestersNBX(const std::vector<size_type> &targetIDs,
                         const MPI_Comm &               comm);
        //
        // default Constructor for serial (without MPI) compilation
        //
        MPIRequestersNBX() = default;

        std::vector<size_type>
        getRequestingRankIds() override;

      private:
        /**
         * List of processes this processor wants to send requests to.
         */
        std::vector<size_type> d_targetIDs;

        /**
         * Buffers for sending requests.
         */
        std::vector<int> d_sendBuffers;

        /**
         * Requests for sending requests.
         */
        std::vector<MPI_Request> d_sendRequests;

        /**
         * Buffers for receiving requests.
         *
         */
        std::vector<int> d_recvBuffers;

        /**
         * Requests for receiving requests.
         */
        std::vector<MPI_Request> d_recvRequests;

        //
        // request for barrier
        //
        MPI_Request d_barrierRequest;

        //
        // MPI communicator
        //
        const MPI_Comm &d_comm;

        /**
         * List of processes who have made a request to this process.
         */
        std::set<size_type> d_requestingProcesses;

        int d_numProcessors;
        int d_myRank;

        /**
         * Check whether all of message sent from the current processor
         * to other processors have been received or not
         */
        bool
        haveAllLocalSendReceived();

        /**
         * Signal to all other processors that for this processor
         * all its message sent to other processors have been received.
         * This is done nonblocking barrier (i.e., MPI_IBarrier).
         */
        void
        signalLocalSendCompletion();

        /**
         * Check whether all of the incoming messages from other processors to
         * the current processor have been received.
         */
        bool
        haveAllIncomingMsgsReceived();

        /**
         * Probe for an incoming message and if there is one receive it
         */
        void
        probeAndReceiveIncomingMsg();

        /**
         * Start to sending message to all the target processors
         */
        void
        startLocalSend();

        /**
         * After all processors have received all the incoming messages,
         * the MPI data structures can be freed and the received messages
         * can be processed.
         */
        void
        finish();
      };

    } // end of namespace mpi
  }   // end of namespace utils
} // end of namespace dftfe
#endif // dftfeMPIRequestersNBX_h
