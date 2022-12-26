// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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

/*
 * @author Ian C. Lin, Sambit Das.
 */

#include <algorithm>
#include <MemoryTransfer.h>
#include <MemoryTransferKernelsDevice.h>


namespace dftfe
{
  namespace utils
  {
    template <typename ValueType>
    void
    MemoryTransfer<MemorySpace::HOST, MemorySpace::HOST>::copy(
      size_type        size,
      ValueType *      dst,
      const ValueType *src)
    {
      std::copy(src, src + size, dst);
    }

#ifdef DFTFE_WITH_DEVICE
    template <typename ValueType>
    void
    MemoryTransfer<MemorySpace::HOST, MemorySpace::HOST_PINNED>::copy(
      size_type        size,
      ValueType *      dst,
      const ValueType *src)
    {
      std::copy(src, src + size, dst);
    }

    template <typename ValueType>
    void
    MemoryTransfer<MemorySpace::HOST, MemorySpace::DEVICE>::copy(
      size_type        size,
      ValueType *      dst,
      const ValueType *src)
    {
      memoryTransferKernelsDevice::deviceMemcpyD2H(dst,
                                                   src,
                                                   size * sizeof(ValueType));
    }

    template <typename ValueType>
    void
    MemoryTransfer<MemorySpace::HOST_PINNED, MemorySpace::HOST>::copy(
      size_type        size,
      ValueType *      dst,
      const ValueType *src)
    {
      std::copy(src, src + size, dst);
    }

    template <typename ValueType>
    void
    MemoryTransfer<MemorySpace::HOST_PINNED, MemorySpace::HOST_PINNED>::copy(
      size_type        size,
      ValueType *      dst,
      const ValueType *src)
    {
      std::copy(src, src + size, dst);
    }

    template <typename ValueType>
    void
    MemoryTransfer<MemorySpace::HOST_PINNED, MemorySpace::DEVICE>::copy(
      size_type        size,
      ValueType *      dst,
      const ValueType *src)
    {
      memoryTransferKernelsDevice::deviceMemcpyD2H(dst,
                                                   src,
                                                   size * sizeof(ValueType));
    }

    template <typename ValueType>
    void
    MemoryTransfer<MemorySpace::DEVICE, MemorySpace::HOST>::copy(
      size_type        size,
      ValueType *      dst,
      const ValueType *src)
    {
      memoryTransferKernelsDevice::deviceMemcpyH2D(dst,
                                                   src,
                                                   size * sizeof(ValueType));
    }

    template <typename ValueType>
    void
    MemoryTransfer<MemorySpace::DEVICE, MemorySpace::HOST_PINNED>::copy(
      size_type        size,
      ValueType *      dst,
      const ValueType *src)
    {
      memoryTransferKernelsDevice::deviceMemcpyH2D(dst,
                                                   src,
                                                   size * sizeof(ValueType));
    }

    template <typename ValueType>
    void
    MemoryTransfer<MemorySpace::DEVICE, MemorySpace::DEVICE>::copy(
      size_type        size,
      ValueType *      dst,
      const ValueType *src)
    {
      memoryTransferKernelsDevice::deviceMemcpyD2D(dst,
                                                   src,
                                                   size * sizeof(ValueType));
    }
#endif // DFTFE_WITH_DEVICE
  }    // namespace utils
} // namespace dftfe
