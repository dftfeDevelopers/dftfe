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
 * @author Ian C. Lin, Sambit Das
 */

#ifndef dftfeMemoryTransfer_h
#define dftfeMemoryTransfer_h

#include <MemorySpaceType.h>
#include <TypeConfig.h>

namespace dftfe
{
  namespace utils
  {
    template <MemorySpace memorySpaceDst, MemorySpace memorySpaceSrc>
    class MemoryTransfer
    {
    public:
      /**
       * @brief Copy array from the memory space of source to the memory space of destination
       * @param size the length of the array
       * @param dst pointer to the destination
       * @param src pointer to the source
       */
      template <typename ValueType>
      static void
      copy(size_t size, ValueType *dst, const ValueType *src);
    };

    template <>
    class MemoryTransfer<MemorySpace::HOST, MemorySpace::HOST>
    {
    public:
      template <typename ValueType>
      static void
      copy(size_t size, ValueType *dst, const ValueType *src);
    };

#ifdef DFTFE_WITH_DEVICE
    template <>
    class MemoryTransfer<MemorySpace::HOST, MemorySpace::HOST_PINNED>
    {
    public:
      template <typename ValueType>
      static void
      copy(size_t size, ValueType *dst, const ValueType *src);
    };

    template <>
    class MemoryTransfer<MemorySpace::HOST, MemorySpace::DEVICE>
    {
    public:
      template <typename ValueType>
      static void
      copy(size_t size, ValueType *dst, const ValueType *src);
    };

    template <>
    class MemoryTransfer<MemorySpace::HOST_PINNED, MemorySpace::HOST>
    {
    public:
      template <typename ValueType>
      static void
      copy(size_t size, ValueType *dst, const ValueType *src);
    };

    template <>
    class MemoryTransfer<MemorySpace::HOST_PINNED, MemorySpace::HOST_PINNED>
    {
    public:
      template <typename ValueType>
      static void
      copy(size_t size, ValueType *dst, const ValueType *src);
    };

    template <>
    class MemoryTransfer<MemorySpace::HOST_PINNED, MemorySpace::DEVICE>
    {
    public:
      template <typename ValueType>
      static void
      copy(size_t size, ValueType *dst, const ValueType *src);
    };

    template <>
    class MemoryTransfer<MemorySpace::DEVICE, MemorySpace::HOST>
    {
    public:
      template <typename ValueType>
      static void
      copy(size_t size, ValueType *dst, const ValueType *src);
    };

    template <>
    class MemoryTransfer<MemorySpace::DEVICE, MemorySpace::HOST_PINNED>
    {
    public:
      template <typename ValueType>
      static void
      copy(size_t size, ValueType *dst, const ValueType *src);
    };

    template <>
    class MemoryTransfer<MemorySpace::DEVICE, MemorySpace::DEVICE>
    {
    public:
      template <typename ValueType>
      static void
      copy(size_t size, ValueType *dst, const ValueType *src);
    };
#endif // DFTFE_WITH_DEVICE
  }    // namespace utils
} // namespace dftfe

#include "../utils/MemoryTransfer.t.cc"

#endif // dftfeMemoryTransfer_h
