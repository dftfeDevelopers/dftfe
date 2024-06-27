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

#include <MemoryManager.h>
#include <Exceptions.h>
#include <MemoryTransfer.h>

namespace dftfe
{
  namespace utils
  {
    //
    // Constructor
    //
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MemoryStorage<ValueType, memorySpace>::MemoryStorage(
      const std::size_t size,
      const ValueType   initVal)
      : d_size(size)
    {
      dftfe::utils::MemoryManager<ValueType, memorySpace>::allocate(size,
                                                                    &d_data);
      dftfe::utils::MemoryManager<ValueType, memorySpace>::set(size,
                                                               d_data,
                                                               initVal);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MemoryStorage<ValueType, memorySpace>::resize(const std::size_t size,
                                                  const ValueType   initVal)
    {
      dftfe::utils::MemoryManager<ValueType, memorySpace>::deallocate(d_data);
      d_size = size;
      if (size > 0)
        {
          dftfe::utils::MemoryManager<ValueType, memorySpace>::allocate(
            size, &d_data);
          dftfe::utils::MemoryManager<ValueType, memorySpace>::set(size,
                                                                   d_data,
                                                                   initVal);
        }
      else
        d_data = nullptr;
    }

    //
    // Destructor
    //
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MemoryStorage<ValueType, memorySpace>::~MemoryStorage()
    {
      dftfe::utils::MemoryManager<ValueType, memorySpace>::deallocate(d_data);
      d_data = nullptr;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MemoryStorage<ValueType, memorySpace>::clear()
    {
      dftfe::utils::MemoryManager<ValueType, memorySpace>::deallocate(d_data);
      d_size = 0;
      d_data = nullptr;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MemoryStorage<ValueType, memorySpace>::MemoryStorage(
      const MemoryStorage<ValueType, memorySpace> &u)
      : d_size(u.d_size)
    {
      dftfe::utils::MemoryManager<ValueType, memorySpace>::allocate(d_size,
                                                                    &d_data);
      utils::MemoryTransfer<memorySpace, memorySpace>::copy(d_size,
                                                            d_data,
                                                            u.d_data);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MemoryStorage<ValueType, memorySpace>::setValue(const ValueType val)
    {
      dftfe::utils::MemoryManager<ValueType, memorySpace>::set(d_size,
                                                               d_data,
                                                               val);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MemoryStorage<ValueType, memorySpace>::MemoryStorage(
      MemoryStorage<ValueType, memorySpace> &&u) noexcept
      : d_size(u.d_size)
      , d_data(nullptr)
    {
      *this = std::move(u);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    std::size_t
    MemoryStorage<ValueType, memorySpace>::size() const
    {
      return d_size;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename MemoryStorage<ValueType, memorySpace>::iterator
    MemoryStorage<ValueType, memorySpace>::begin()
    {
      return d_data;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename MemoryStorage<ValueType, memorySpace>::const_iterator
    MemoryStorage<ValueType, memorySpace>::begin() const
    {
      return d_data;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename MemoryStorage<ValueType, memorySpace>::iterator
    MemoryStorage<ValueType, memorySpace>::end()
    {
      return (d_data + d_size);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename MemoryStorage<ValueType, memorySpace>::const_iterator
    MemoryStorage<ValueType, memorySpace>::end() const
    {
      return (d_data + d_size);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MemoryStorage<ValueType, memorySpace> &
    MemoryStorage<ValueType, memorySpace>::
    operator=(const MemoryStorage<ValueType, memorySpace> &rhs)
    {
      if (&rhs != this)
        {
          if (rhs.d_size != d_size)
            {
              this->resize(rhs.d_size);
            }
          utils::MemoryTransfer<memorySpace, memorySpace>::copy(rhs.d_size,
                                                                this->d_data,
                                                                rhs.d_data);
        }
      return (*this);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MemoryStorage<ValueType, memorySpace> &
    MemoryStorage<ValueType, memorySpace>::
    operator=(MemoryStorage<ValueType, memorySpace> &&rhs) noexcept
    {
      if (&rhs != this)
        {
          delete[] d_data;
          d_data     = rhs.d_data;
          d_size     = rhs.d_size;
          rhs.d_size = 0;
          rhs.d_data = nullptr;
        }
      return (*this);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename MemoryStorage<ValueType, memorySpace>::reference
      MemoryStorage<ValueType, memorySpace>::operator[](const std::size_t i)
    {
      // throwException<InvalidArgument>(
      //   memorySpace != dftfe::utils::MemorySpace::DEVICE,
      //   "[] operator return reference to element not implemented for
      //   DEVICE");
      return d_data[i];
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename MemoryStorage<ValueType, memorySpace>::const_reference
      MemoryStorage<ValueType, memorySpace>::
      operator[](const std::size_t i) const
    {
      // throwException<InvalidArgument>(
      //   memorySpace != dftfe::utils::MemorySpace::DEVICE,
      //   "[] operator return const reference to element not implemented for
      //   DEVICE");
      return d_data[i];
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MemoryStorage<ValueType, memorySpace>::swap(
      MemoryStorage<ValueType, memorySpace> &rhs)
    {
      ValueType *       tempData = d_data;
      const std::size_t tempSize = d_size;
      d_data                     = rhs.d_data;
      d_size                     = rhs.d_size;
      rhs.d_data                 = tempData;
      rhs.d_size                 = tempSize;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    ValueType *
    MemoryStorage<ValueType, memorySpace>::data() noexcept
    {
      return d_data;
    }
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    const ValueType *
    MemoryStorage<ValueType, memorySpace>::data() const noexcept
    {
      return d_data;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    template <dftfe::utils::MemorySpace memorySpaceDst>
    void
    MemoryStorage<ValueType, memorySpace>::copyTo(
      MemoryStorage<ValueType, memorySpaceDst> &dstMemoryStorage) const
    {
      throwException<LengthError>(
        d_size <= dstMemoryStorage.size(),
        "The allocated size of destination MemoryStorage is insufficient to "
        "copy from the the MemoryStorage.");
      MemoryTransfer<memorySpaceDst, memorySpace>::copy(
        d_size, dstMemoryStorage.begin(), this->begin());
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    template <dftfe::utils::MemorySpace memorySpaceDst>
    void
    MemoryStorage<ValueType, memorySpace>::copyTo(
      MemoryStorage<ValueType, memorySpaceDst> &dstMemoryStorage,
      const std::size_t                         N,
      const std::size_t                         srcOffset,
      const std::size_t                         dstOffset) const
    {
      throwException<LengthError>(
        srcOffset + N <= d_size,
        "The offset and copy size specified for the source MemoryStorage"
        " is out of range for it.");

      throwException<LengthError>(
        dstOffset + N <= dstMemoryStorage.size(),
        "The offset and size specified for the destination MemoryStorage"
        " is out of range for it.");

      MemoryTransfer<memorySpaceDst, memorySpace>::copy(
        N, dstMemoryStorage.begin() + dstOffset, this->begin() + srcOffset);
    }


    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    template <dftfe::utils::MemorySpace memorySpaceSrc>
    void
    MemoryStorage<ValueType, memorySpace>::copyFrom(
      const MemoryStorage<ValueType, memorySpaceSrc> &srcMemoryStorage)
    {
      throwException<LengthError>(
        srcMemoryStorage.size() <= d_size,
        "The allocated size of the MemoryStorage is insufficient to "
        " copy from the source MemoryStorage.");
      MemoryTransfer<memorySpace, memorySpaceSrc>::copy(
        srcMemoryStorage.size(), this->begin(), srcMemoryStorage.begin());
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    template <dftfe::utils::MemorySpace memorySpaceSrc>
    void
    MemoryStorage<ValueType, memorySpace>::copyFrom(
      const MemoryStorage<ValueType, memorySpaceSrc> &srcMemoryStorage,
      const std::size_t                               N,
      const std::size_t                               srcOffset,
      const std::size_t                               dstOffset)
    {
      throwException<LengthError>(
        srcOffset + N <= srcMemoryStorage.size(),
        "The offset and copy size specified for the source MemoryStorage"
        " is out of range for it.");

      throwException<LengthError>(
        dstOffset + N <= d_size,
        "The offset and size specified for the destination MemoryStorage"
        " is out of range for it.");

      MemoryTransfer<memorySpace, memorySpaceSrc>::copy(
        N, this->begin() + dstOffset, srcMemoryStorage.begin() + srcOffset);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    template <dftfe::utils::MemorySpace memorySpaceDst>
    void
    MemoryStorage<ValueType, memorySpace>::copyTo(ValueType *dst) const
    {
      MemoryTransfer<memorySpaceDst, memorySpace>::copy(d_size,
                                                        dst,
                                                        this->begin());
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    template <dftfe::utils::MemorySpace memorySpaceDst>
    void
    MemoryStorage<ValueType, memorySpace>::copyTo(
      ValueType *       dst,
      const std::size_t N,
      const std::size_t srcOffset,
      const std::size_t dstOffset) const
    {
      throwException<LengthError>(
        srcOffset + N <= d_size,
        "The offset and copy size specified for the source MemoryStorage"
        " is out of range for it.");
      MemoryTransfer<memorySpaceDst, memorySpace>::copy(
        N, dst + dstOffset, this->begin() + srcOffset);
    }


    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    template <dftfe::utils::MemorySpace memorySpaceSrc>
    void
    MemoryStorage<ValueType, memorySpace>::copyFrom(const ValueType *src)
    {
      MemoryTransfer<memorySpace, memorySpaceSrc>::copy(d_size,
                                                        this->begin(),
                                                        src);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    template <dftfe::utils::MemorySpace memorySpaceSrc>
    void
    MemoryStorage<ValueType, memorySpace>::copyFrom(const ValueType * src,
                                                    const std::size_t N,
                                                    const std::size_t srcOffset,
                                                    const std::size_t dstOffset)
    {
      throwException<LengthError>(
        dstOffset + N <= d_size,
        "The offset and size specified for the destination MemoryStorage"
        " is out of range for it.");

      MemoryTransfer<memorySpace, memorySpaceSrc>::copy(
        N, this->begin() + dstOffset, src + srcOffset);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MemoryStorage<ValueType, memorySpace>::copyTo(
      std::vector<ValueType> &dst) const
    {
      if (dst.size() < d_size)
        dst.resize(d_size);

      MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
        d_size, dst.data(), this->begin());
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MemoryStorage<ValueType, memorySpace>::copyTo(
      std::vector<ValueType> &dst,
      const std::size_t       N,
      const std::size_t       srcOffset,
      const std::size_t       dstOffset) const
    {
      throwException<LengthError>(
        srcOffset + N <= d_size,
        "The offset and copy size specified for the source MemoryStorage"
        " is out of range for it.");
      if (dst.size() < N + dstOffset)
        dst.resize(N + dstOffset);

      MemoryTransfer<utils::MemorySpace::HOST, memorySpace>::copy(
        N, dst.data() + dstOffset, this->begin() + srcOffset);
    }


    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MemoryStorage<ValueType, memorySpace>::copyFrom(
      const std::vector<ValueType> &src)
    {
      throwException<LengthError>(
        src.size() <= d_size,
        "The allocated size of the MemoryStorage is insufficient to copy from "
        "the source STL vector");
      MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(src.size(),
                                                                  this->begin(),
                                                                  src.data());
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MemoryStorage<ValueType, memorySpace>::copyFrom(
      const std::vector<ValueType> &src,
      const std::size_t             N,
      const std::size_t             srcOffset,
      const std::size_t             dstOffset)
    {
      throwException<LengthError>(
        dstOffset + N <= d_size,
        "The offset and size specified for the destination MemoryStorage"
        " is out of range for it.");

      throwException<LengthError>(
        srcOffset + N <= src.size(),
        "The offset and size specified for the source STL vector "
        " is out of range for it.");

      MemoryTransfer<memorySpace, utils::MemorySpace::HOST>::copy(
        N, this->begin() + dstOffset, src.data() + srcOffset);
    }

    template <typename ValueType, utils::MemorySpace memorySpaceDst>
    MemoryStorage<ValueType, memorySpaceDst>
    memoryStorageFromSTL(const std::vector<ValueType> &src)
    {
      MemoryStorage<ValueType, memorySpaceDst> returnValue(src.size());
      MemoryTransfer<memorySpaceDst, utils::MemorySpace::HOST>::copy(
        src.size(), returnValue.begin(), src.data());
      return returnValue;
    }

  } // namespace utils
} // namespace dftfe
