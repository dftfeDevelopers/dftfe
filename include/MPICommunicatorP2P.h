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

#ifndef dftfeMPICommunicatorP2P_h
#define dftfeMPICommunicatorP2P_h

#include <MemorySpaceType.h>
#include <MPIPatternP2P.h>
#include <TypeConfig.h>
#include <MemoryStorage.h>
#include <DataTypeOverloads.h>
#include <dftfeDataTypes.h>


namespace dftfe
{
  namespace utils
  {
    namespace mpi
    {
      template <typename ValueType, MemorySpace memorySpace>
      class MPICommunicatorP2P
      {
      public:
        MPICommunicatorP2P(
          std::shared_ptr<const MPIPatternP2P<memorySpace>> mpiPatternP2P,
          const size_type                                   blockSize);

        void
        updateGhostValues(MemoryStorage<ValueType, memorySpace> &dataArray,
                          const size_type communicationChannel = 0);

        void
        accumulateAddLocallyOwned(
          MemoryStorage<ValueType, memorySpace> &dataArray,
          const size_type                        communicationChannel = 0);


        void
        updateGhostValuesBegin(MemoryStorage<ValueType, memorySpace> &dataArray,
                               const size_type communicationChannel = 0);

        void
        updateGhostValuesEnd(MemoryStorage<ValueType, memorySpace> &dataArray);

        void
        accumulateAddLocallyOwnedBegin(
          MemoryStorage<ValueType, memorySpace> &dataArray,
          const size_type                        communicationChannel = 0);

        void
        accumulateAddLocallyOwnedEnd(
          MemoryStorage<ValueType, memorySpace> &dataArray);

        std::shared_ptr<const MPIPatternP2P<memorySpace>>
        getMPIPatternP2P() const;

        int
        getBlockSize() const;

      private:
        std::shared_ptr<const MPIPatternP2P<memorySpace>> d_mpiPatternP2P;

        size_type d_blockSize;

        size_type d_locallyOwnedSize;

        size_type d_ghostSize;

        MemoryStorage<ValueType, memorySpace> d_sendRecvBuffer;

        MemoryStorage<double, memorySpace> d_tempDoubleRealArrayForAtomics;

        MemoryStorage<double, memorySpace> d_tempDoubleImagArrayForAtomics;

        MemoryStorage<float, memorySpace> d_tempFloatRealArrayForAtomics;

        MemoryStorage<float, memorySpace> d_tempFloatImagArrayForAtomics;


#ifdef DFTFE_WITH_DEVICE
        std::shared_ptr<MemoryStorage<ValueType, MemorySpace::HOST_PINNED>>
          d_ghostDataCopyHostPinnedPtr;

        std::shared_ptr<MemoryStorage<ValueType, MemorySpace::HOST_PINNED>>
          d_sendRecvBufferHostPinnedPtr;
#endif // DFTFE_WITH_DEVICE

        std::vector<MPI_Request> d_requestsUpdateGhostValues;
        std::vector<MPI_Request> d_requestsAccumulateAddLocallyOwned;
        MPI_Comm                 d_mpiCommunicator;
      };

    } // namespace mpi
  }   // namespace utils
} // namespace dftfe
#endif // dftfeMPICommunicatorP2P_h
