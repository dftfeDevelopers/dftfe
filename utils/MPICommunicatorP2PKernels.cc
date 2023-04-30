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

#include <MPICommunicatorP2PKernels.h>
#include <Exceptions.h>
#include <complex>
#include <algorithm>


namespace dftfe
{
  namespace utils
  {
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2PKernels<ValueType, memorySpace>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<ValueType, memorySpace> &dataArray,
        const SizeTypeVector &                 ownedLocalIndicesForTargetProcs,
        const size_type                        blockSize,
        MemoryStorage<ValueType, memorySpace> &sendBuffer)
    {
      for (size_type i = 0; i < ownedLocalIndicesForTargetProcs.size(); ++i)
        std::copy(dataArray.data() +
                    ownedLocalIndicesForTargetProcs.data()[i] * blockSize,
                  dataArray.data() +
                    ownedLocalIndicesForTargetProcs.data()[i] * blockSize +
                    blockSize,
                  sendBuffer.data() + i * blockSize);
    }


    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MPICommunicatorP2PKernels<ValueType, memorySpace>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<ValueType, memorySpace> &recvBuffer,
        const SizeTypeVector &                 ownedLocalIndicesForTargetProcs,
        const size_type                        blockSize,
        const size_type                        locallyOwnedSize,
        const size_type                        ghostSize,
        MemoryStorage<double, memorySpace> &   tempDoubleRealDataArray,
        MemoryStorage<double, memorySpace> &   tempDoubleImagDataArray,
        MemoryStorage<float, memorySpace> &    tempFloatRealDataArray,
        MemoryStorage<float, memorySpace> &    tempFloatImagDataArray,
        MemoryStorage<ValueType, memorySpace> &dataArray)
    {
      for (size_type i = 0; i < ownedLocalIndicesForTargetProcs.size(); ++i)
        std::transform(recvBuffer.data() + i * blockSize,
                       recvBuffer.data() + (i + 1) * blockSize,
                       dataArray.data() +
                         ownedLocalIndicesForTargetProcs.data()[i] * blockSize,
                       dataArray.data() +
                         ownedLocalIndicesForTargetProcs.data()[i] * blockSize,
                       std::plus<>{});
    }


    template class MPICommunicatorP2PKernels<double,
                                             dftfe::utils::MemorySpace::HOST>;
    template class MPICommunicatorP2PKernels<float,
                                             dftfe::utils::MemorySpace::HOST>;
    template class MPICommunicatorP2PKernels<std::complex<double>,
                                             dftfe::utils::MemorySpace::HOST>;
    template class MPICommunicatorP2PKernels<std::complex<float>,
                                             dftfe::utils::MemorySpace::HOST>;

#ifdef DFTFE_WITH_DEVICE
    template class MPICommunicatorP2PKernels<
      double,
      dftfe::utils::MemorySpace::HOST_PINNED>;
    template class MPICommunicatorP2PKernels<
      float,
      dftfe::utils::MemorySpace::HOST_PINNED>;
    template class MPICommunicatorP2PKernels<
      std::complex<double>,
      dftfe::utils::MemorySpace::HOST_PINNED>;
    template class MPICommunicatorP2PKernels<
      std::complex<float>,
      dftfe::utils::MemorySpace::HOST_PINNED>;
#endif


  } // namespace utils
} // namespace dftfe
