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

#include <DeviceAPICalls.h>
#include <algorithm>
#include <MemoryManager.h>
#include <complex>

namespace dftfe
{
  namespace utils
  {
    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST>::allocate(std::size_t size,
                                                          ValueType **ptr)
    {
      *ptr = new ValueType[size];
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST>::deallocate(ValueType *ptr)
    {
      if (ptr != nullptr)
        delete[] ptr;
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST>::set(std::size_t size,
                                                     ValueType * ptr,
                                                     ValueType   val)
    {
      if (size != 0)
        std::fill(ptr, ptr + size, val);
    }

#ifdef DFTFE_WITH_DEVICE
    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST_PINNED>::allocate(
      std::size_t size,
      ValueType **ptr)
    {
      deviceHostMalloc((void **)ptr, size * sizeof(ValueType));
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST_PINNED>::deallocate(
      ValueType *ptr)
    {
      if (ptr != nullptr)
        deviceHostFree(ptr);
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::HOST_PINNED>::set(std::size_t size,
                                                            ValueType * ptr,
                                                            ValueType   val)
    {
      std::fill(ptr, ptr + size, val);
    }


    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::DEVICE>::allocate(std::size_t size,
                                                            ValueType **ptr)
    {
      deviceMalloc((void **)ptr, size * sizeof(ValueType));
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::DEVICE>::deallocate(ValueType *ptr)
    {
      if (ptr != nullptr)
        deviceFree(ptr);
    }

    template <typename ValueType>
    void
    MemoryManager<ValueType, MemorySpace::DEVICE>::set(std::size_t size,
                                                       ValueType * ptr,
                                                       ValueType   val)
    {
      deviceSetValue(ptr, val, size);
    }

    template class MemoryManager<int, dftfe::utils::MemorySpace::DEVICE>;

    template class MemoryManager<unsigned int,
                                 dftfe::utils::MemorySpace::DEVICE>;

    template class MemoryManager<long int, dftfe::utils::MemorySpace::DEVICE>;

    template class MemoryManager<unsigned long int,
                                 dftfe::utils::MemorySpace::DEVICE>;

    template class MemoryManager<double, dftfe::utils::MemorySpace::DEVICE>;
    template class MemoryManager<float, dftfe::utils::MemorySpace::DEVICE>;
    template class MemoryManager<std::complex<double>,
                                 dftfe::utils::MemorySpace::DEVICE>;
    template class MemoryManager<std::complex<float>,
                                 dftfe::utils::MemorySpace::DEVICE>;

    template class MemoryManager<int, dftfe::utils::MemorySpace::HOST_PINNED>;

    template class MemoryManager<unsigned int,
                                 dftfe::utils::MemorySpace::HOST_PINNED>;

    template class MemoryManager<long int,
                                 dftfe::utils::MemorySpace::HOST_PINNED>;

    template class MemoryManager<unsigned long int,
                                 dftfe::utils::MemorySpace::HOST_PINNED>;


    template class MemoryManager<double,
                                 dftfe::utils::MemorySpace::HOST_PINNED>;
    template class MemoryManager<float, dftfe::utils::MemorySpace::HOST_PINNED>;
    template class MemoryManager<std::complex<double>,
                                 dftfe::utils::MemorySpace::HOST_PINNED>;
    template class MemoryManager<std::complex<float>,
                                 dftfe::utils::MemorySpace::HOST_PINNED>;

#endif // DFTFE_WITH_DEVICE

    template class MemoryManager<int, dftfe::utils::MemorySpace::HOST>;

    template class MemoryManager<unsigned int, dftfe::utils::MemorySpace::HOST>;

    template class MemoryManager<long int, dftfe::utils::MemorySpace::HOST>;

    template class MemoryManager<unsigned long int,
                                 dftfe::utils::MemorySpace::HOST>;


    template class MemoryManager<double, dftfe::utils::MemorySpace::HOST>;
    template class MemoryManager<float, dftfe::utils::MemorySpace::HOST>;
    template class MemoryManager<std::complex<double>,
                                 dftfe::utils::MemorySpace::HOST>;
    template class MemoryManager<std::complex<float>,
                                 dftfe::utils::MemorySpace::HOST>;

  } // namespace utils

} // namespace dftfe
