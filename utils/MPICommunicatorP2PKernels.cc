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

#include <MPICommunicatorP2PKernels.h>
#include <Exceptions.h>
#include <complex>
#include <algorithm>


namespace dftfe
{
  namespace utils
  {
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    template <typename ValueTypeComm>
    void
    MPICommunicatorP2PKernels<ValueType, memorySpace>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<ValueType, memorySpace> &dataArray,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        MemoryStorage<ValueTypeComm, memorySpace> &sendBuffer)
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
    template <typename ValueTypeComm>
    void
    MPICommunicatorP2PKernels<ValueType, memorySpace>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<ValueTypeComm, memorySpace> &recvBuffer,
        const SizeTypeVector &                 ownedLocalIndicesForTargetProcs,
        const size_type                        blockSize,
        const size_type                        locallyOwnedSize,
        const size_type                        ghostSize,
        MemoryStorage<ValueType, memorySpace> &dataArray)
    {
      if constexpr (std::is_same<ValueType, std::complex<double>>::value ||
                    std::is_same<ValueType, std::complex<float>>::value)
        for (size_type i = 0; i < ownedLocalIndicesForTargetProcs.size(); ++i)
          std::transform(
            recvBuffer.data() + i * blockSize,
            recvBuffer.data() + (i + 1) * blockSize,
            dataArray.data() +
              ownedLocalIndicesForTargetProcs.data()[i] * blockSize,
            dataArray.data() +
              ownedLocalIndicesForTargetProcs.data()[i] * blockSize,
            [](auto &a, auto &b) {
              return ValueType(a.real() + b.real(), a.imag() + b.imag());
            });
      else
        for (size_type i = 0; i < ownedLocalIndicesForTargetProcs.size(); ++i)
          std::transform(recvBuffer.data() + i * blockSize,
                         recvBuffer.data() + (i + 1) * blockSize,
                         dataArray.data() +
                           ownedLocalIndicesForTargetProcs.data()[i] *
                             blockSize,
                         dataArray.data() +
                           ownedLocalIndicesForTargetProcs.data()[i] *
                             blockSize,
                         std::plus<>{});
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    template <typename ValueTypeComm>
    void
    MPICommunicatorP2PKernels<ValueType, memorySpace>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<ValueTypeComm, memorySpace> &recvBuffer,
        const SizeTypeVector &                 ownedLocalIndicesForTargetProcs,
        const size_type                        blockSize,
        const size_type                        locallyOwnedSize,
        const size_type                        ghostSize,
        MemoryStorage<ValueType, memorySpace> &dataArray)
    {
      for (size_type i = 0; i < ownedLocalIndicesForTargetProcs.size(); ++i)
        std::copy(recvBuffer.data() + i * blockSize,
                  recvBuffer.data() + (i + 1) * blockSize,
                  dataArray.data() +
                    ownedLocalIndicesForTargetProcs.data()[i] * blockSize);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    template <typename ValueType1, typename ValueType2>
    void
    MPICommunicatorP2PKernels<ValueType, memorySpace>::
      copyValueType1ArrToValueType2Arr(const size_type   blockSize,
                                       const ValueType1 *type1Array,
                                       ValueType2 *      type2Array)
    {
      if constexpr (std::is_same<ValueType, std::complex<double>>::value ||
                    std::is_same<ValueType, std::complex<float>>::value)
        std::transform(type1Array,
                       type1Array + blockSize,
                       type2Array,
                       [](auto &a) { return ValueType2(a.real(), a.imag()); });
      else
        std::transform(type1Array,
                       type1Array + blockSize,
                       type2Array,
                       [](auto &a) { return ValueType2(a); });
    }


    template class MPICommunicatorP2PKernels<double,
                                             dftfe::utils::MemorySpace::HOST>;
    template class MPICommunicatorP2PKernels<float,
                                             dftfe::utils::MemorySpace::HOST>;
    template class MPICommunicatorP2PKernels<std::complex<double>,
                                             dftfe::utils::MemorySpace::HOST>;
    template class MPICommunicatorP2PKernels<std::complex<float>,
                                             dftfe::utils::MemorySpace::HOST>;
    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::HOST>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &dataArray,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &sendBuffer);
    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::HOST>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &dataArray,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        MemoryStorage<float, dftfe::utils::MemorySpace::HOST> &sendBuffer);
    template void
    MPICommunicatorP2PKernels<float, dftfe::utils::MemorySpace::HOST>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<float, dftfe::utils::MemorySpace::HOST> &dataArray,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        MemoryStorage<float, dftfe::utils::MemorySpace::HOST> &sendBuffer);
    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::HOST>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<std::complex<double>,
                            dftfe::utils::MemorySpace::HOST> &dataArray,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        MemoryStorage<std::complex<double>, dftfe::utils::MemorySpace::HOST>
          &sendBuffer);
    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::HOST>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<std::complex<double>,
                            dftfe::utils::MemorySpace::HOST> &dataArray,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        MemoryStorage<std::complex<float>, dftfe::utils::MemorySpace::HOST>
          &sendBuffer);
    template void
    MPICommunicatorP2PKernels<std::complex<float>,
                              dftfe::utils::MemorySpace::HOST>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<std::complex<float>,
                            dftfe::utils::MemorySpace::HOST> &dataArray,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        MemoryStorage<std::complex<float>, dftfe::utils::MemorySpace::HOST>
          &sendBuffer);

    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::HOST>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          &                   recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &dataArray);
    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::HOST>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<float, dftfe::utils::MemorySpace::HOST> &recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &dataArray);
    template void
    MPICommunicatorP2PKernels<float, dftfe::utils::MemorySpace::HOST>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<float, dftfe::utils::MemorySpace::HOST> &recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<float, dftfe::utils::MemorySpace::HOST> &dataArray);
    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::HOST>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<double>,
                            dftfe::utils::MemorySpace::HOST> &recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<std::complex<double>, dftfe::utils::MemorySpace::HOST>
          &dataArray);
    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::HOST>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<float>,
                            dftfe::utils::MemorySpace::HOST> &recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<std::complex<double>, dftfe::utils::MemorySpace::HOST>
          &dataArray);
    template void
    MPICommunicatorP2PKernels<std::complex<float>,
                              dftfe::utils::MemorySpace::HOST>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<float>,
                            dftfe::utils::MemorySpace::HOST> &recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<std::complex<float>, dftfe::utils::MemorySpace::HOST>
          &dataArray);


    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::HOST>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
          &                   recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &dataArray);
    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::HOST>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<float, dftfe::utils::MemorySpace::HOST> &recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::HOST> &dataArray);
    template void
    MPICommunicatorP2PKernels<float, dftfe::utils::MemorySpace::HOST>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<float, dftfe::utils::MemorySpace::HOST> &recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<float, dftfe::utils::MemorySpace::HOST> &dataArray);
    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::HOST>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<double>,
                            dftfe::utils::MemorySpace::HOST> &recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<std::complex<double>, dftfe::utils::MemorySpace::HOST>
          &dataArray);
    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::HOST>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<float>,
                            dftfe::utils::MemorySpace::HOST> &recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<std::complex<double>, dftfe::utils::MemorySpace::HOST>
          &dataArray);
    template void
    MPICommunicatorP2PKernels<std::complex<float>,
                              dftfe::utils::MemorySpace::HOST>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<float>,
                            dftfe::utils::MemorySpace::HOST> &recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<std::complex<float>, dftfe::utils::MemorySpace::HOST>
          &dataArray);

    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::HOST>::
      copyValueType1ArrToValueType2Arr(const size_type blockSize,
                                       const double *  type1Array,
                                       float *         type2Array);

    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::HOST>::
      copyValueType1ArrToValueType2Arr(const size_type blockSize,
                                       const float *   type1Array,
                                       double *        type2Array);

    template void
    MPICommunicatorP2PKernels<float, dftfe::utils::MemorySpace::HOST>::
      copyValueType1ArrToValueType2Arr(const size_type blockSize,
                                       const float *   type1Array,
                                       float *         type2Array);

    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::HOST>::
      copyValueType1ArrToValueType2Arr(const size_type             blockSize,
                                       const std::complex<double> *type1Array,
                                       std::complex<float> *       type2Array);

    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::HOST>::
      copyValueType1ArrToValueType2Arr(const size_type            blockSize,
                                       const std::complex<float> *type1Array,
                                       std::complex<double> *     type2Array);

    template void
    MPICommunicatorP2PKernels<std::complex<float>,
                              dftfe::utils::MemorySpace::HOST>::
      copyValueType1ArrToValueType2Arr(const size_type            blockSize,
                                       const std::complex<float> *type1Array,
                                       std::complex<float> *      type2Array);
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
    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::HOST_PINNED>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<double, dftfe::utils::MemorySpace::HOST_PINNED>
          &                   dataArray,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::HOST_PINNED>
          &sendBuffer);
    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::HOST_PINNED>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<double, dftfe::utils::MemorySpace::HOST_PINNED>
          &                   dataArray,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        MemoryStorage<float, dftfe::utils::MemorySpace::HOST_PINNED>
          &sendBuffer);
    template void
    MPICommunicatorP2PKernels<float, dftfe::utils::MemorySpace::HOST_PINNED>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<float, dftfe::utils::MemorySpace::HOST_PINNED>
          &                   dataArray,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        MemoryStorage<float, dftfe::utils::MemorySpace::HOST_PINNED>
          &sendBuffer);
    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::HOST_PINNED>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<std::complex<double>,
                            dftfe::utils::MemorySpace::HOST_PINNED> &dataArray,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        MemoryStorage<std::complex<double>,
                      dftfe::utils::MemorySpace::HOST_PINNED> &sendBuffer);
    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::HOST_PINNED>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<std::complex<double>,
                            dftfe::utils::MemorySpace::HOST_PINNED> &dataArray,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        MemoryStorage<std::complex<float>,
                      dftfe::utils::MemorySpace::HOST_PINNED> &sendBuffer);
    template void
    MPICommunicatorP2PKernels<std::complex<float>,
                              dftfe::utils::MemorySpace::HOST_PINNED>::
      gatherLocallyOwnedEntriesSendBufferToTargetProcs(
        const MemoryStorage<std::complex<float>,
                            dftfe::utils::MemorySpace::HOST_PINNED> &dataArray,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        MemoryStorage<std::complex<float>,
                      dftfe::utils::MemorySpace::HOST_PINNED> &sendBuffer);

    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::HOST_PINNED>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<double, dftfe::utils::MemorySpace::HOST_PINNED>
          &                   recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::HOST_PINNED>
          &dataArray);
    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::HOST_PINNED>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<float, dftfe::utils::MemorySpace::HOST_PINNED>
          &                   recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::HOST_PINNED>
          &dataArray);
    template void
    MPICommunicatorP2PKernels<float, dftfe::utils::MemorySpace::HOST_PINNED>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<float, dftfe::utils::MemorySpace::HOST_PINNED>
          &                   recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<float, dftfe::utils::MemorySpace::HOST_PINNED>
          &dataArray);
    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::HOST_PINNED>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<double>,
                            dftfe::utils::MemorySpace::HOST_PINNED> &recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<std::complex<double>,
                      dftfe::utils::MemorySpace::HOST_PINNED> &dataArray);
    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::HOST_PINNED>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<float>,
                            dftfe::utils::MemorySpace::HOST_PINNED> &recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<std::complex<double>,
                      dftfe::utils::MemorySpace::HOST_PINNED> &dataArray);
    template void
    MPICommunicatorP2PKernels<std::complex<float>,
                              dftfe::utils::MemorySpace::HOST_PINNED>::
      accumAddLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<float>,
                            dftfe::utils::MemorySpace::HOST_PINNED> &recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<std::complex<float>,
                      dftfe::utils::MemorySpace::HOST_PINNED> &dataArray);

    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::HOST_PINNED>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<double, dftfe::utils::MemorySpace::HOST_PINNED>
          &                   recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::HOST_PINNED>
          &dataArray);
    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::HOST_PINNED>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<float, dftfe::utils::MemorySpace::HOST_PINNED>
          &                   recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<double, dftfe::utils::MemorySpace::HOST_PINNED>
          &dataArray);
    template void
    MPICommunicatorP2PKernels<float, dftfe::utils::MemorySpace::HOST_PINNED>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<float, dftfe::utils::MemorySpace::HOST_PINNED>
          &                   recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<float, dftfe::utils::MemorySpace::HOST_PINNED>
          &dataArray);
    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::HOST_PINNED>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<double>,
                            dftfe::utils::MemorySpace::HOST_PINNED> &recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<std::complex<double>,
                      dftfe::utils::MemorySpace::HOST_PINNED> &dataArray);
    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::HOST_PINNED>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<float>,
                            dftfe::utils::MemorySpace::HOST_PINNED> &recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<std::complex<double>,
                      dftfe::utils::MemorySpace::HOST_PINNED> &dataArray);
    template void
    MPICommunicatorP2PKernels<std::complex<float>,
                              dftfe::utils::MemorySpace::HOST_PINNED>::
      accumInsertLocallyOwnedContrRecvBufferFromTargetProcs(
        const MemoryStorage<std::complex<float>,
                            dftfe::utils::MemorySpace::HOST_PINNED> &recvBuffer,
        const SizeTypeVector &ownedLocalIndicesForTargetProcs,
        const size_type       blockSize,
        const size_type       locallyOwnedSize,
        const size_type       ghostSize,
        MemoryStorage<std::complex<float>,
                      dftfe::utils::MemorySpace::HOST_PINNED> &dataArray);

    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::HOST_PINNED>::
      copyValueType1ArrToValueType2Arr(const size_type blockSize,
                                       const double *  type1Array,
                                       float *         type2Array);

    template void
    MPICommunicatorP2PKernels<double, dftfe::utils::MemorySpace::HOST_PINNED>::
      copyValueType1ArrToValueType2Arr(const size_type blockSize,
                                       const float *   type1Array,
                                       double *        type2Array);

    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::HOST_PINNED>::
      copyValueType1ArrToValueType2Arr(const size_type             blockSize,
                                       const std::complex<double> *type1Array,
                                       std::complex<float> *       type2Array);

    template void
    MPICommunicatorP2PKernels<std::complex<double>,
                              dftfe::utils::MemorySpace::HOST_PINNED>::
      copyValueType1ArrToValueType2Arr(const size_type            blockSize,
                                       const std::complex<float> *type1Array,
                                       std::complex<double> *     type2Array);
    template void
    MPICommunicatorP2PKernels<float, dftfe::utils::MemorySpace::HOST_PINNED>::
      copyValueType1ArrToValueType2Arr(const size_type blockSize,
                                       const float *   type1Array,
                                       float *         type2Array);

    template void
    MPICommunicatorP2PKernels<std::complex<float>,
                              dftfe::utils::MemorySpace::HOST_PINNED>::
      copyValueType1ArrToValueType2Arr(const size_type            blockSize,
                                       const std::complex<float> *type1Array,
                                       std::complex<float> *      type2Array);
#endif


  } // namespace utils
} // namespace dftfe
