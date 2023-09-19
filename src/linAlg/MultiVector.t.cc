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
 * @author Sambit Das, Bikash Kanungo
 */
#include <Exceptions.h>
#include <cmath>

namespace dftfe
{
  namespace linearAlgebra
  {
    /**
     * @brief Constructor for a \b serial MultiVector using size, numVectors and
     * init value
     **/
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(
      const size_type size,
      const size_type numVectors,
      const ValueType initVal /* = utils::Types<ValueType>::zero*/)
    {
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          size * numVectors, initVal);
      d_globalSize       = size;
      d_locallyOwnedSize = size;
      d_ghostSize        = 0;
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_numVectors       = numVectors;
      d_mpiPatternP2P =
        std::make_shared<const utils::mpi::MPIPatternP2P<memorySpace>>(size);
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(d_mpiPatternP2P,
                                                                numVectors);
    }

    /**
     * @brief Constructor for a \serial MultiVector with a predefined
     * MultiVector::Storage (i.e., utils::MemoryStorage).
     * This constructor transfers the ownership of the input Storage to the
     * MultiVector. This is useful when one does not want to allocate new
     * memory and instead use memory allocated in the MultiVector::Storage
     * (i.e., utils::MemoryStorage).
     */
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(
      std::unique_ptr<typename MultiVector<ValueType, memorySpace>::Storage>
                      storage,
      const size_type numVectors)
    {
      d_storage          = std::move(storage);
      d_globalSize       = d_storage.size();
      d_locallyOwnedSize = d_storage.size();
      d_ghostSize        = 0;
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_numVectors       = numVectors;
      d_mpiPatternP2P =
        std::make_shared<const utils::mpi::MPIPatternP2P<memorySpace>>(
          d_localSize);
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(d_mpiPatternP2P,
                                                                numVectors);
    }

    //
    // Constructor for \distributed MultiVector using an existing
    // MPIPatternP2P object
    //
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(
      std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                      mpiPatternP2P,
      const size_type numVectors,
      const ValueType initVal /* = utils::Types<ValueType>::zero*/)
      : d_mpiPatternP2P(mpiPatternP2P)
    {
      d_globalSize       = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize        = d_mpiPatternP2P->localGhostSize();
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_numVectors       = numVectors;
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          d_localSize * d_numVectors, initVal);
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(mpiPatternP2P,
                                                                numVectors);
    }

    /**
     * @brief Constructor for a \b distributed MultiVector with a predefined
     * MultiVector::Storage (i.e., utils::MemoryStorage) and MPIPatternP2P.
     * This constructor transfers the ownership of the input Storage to the
     * MultiVector. This is useful when one does not want to allocate new
     * memory and instead use memory allocated in the input MultiVector::Storage
     * (i.e., utils::MemoryStorage).
     */
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(
      std::unique_ptr<typename MultiVector<ValueType, memorySpace>::Storage>
        &storage,
      std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                      mpiPatternP2P,
      const size_type numVectors)
      : d_mpiPatternP2P(mpiPatternP2P)
    {
      d_storage            = std::move(storage);
      d_globalSize         = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize   = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize          = d_mpiPatternP2P->localGhostSize();
      d_localSize          = d_locallyOwnedSize + d_ghostSize;
      d_numVectors         = numVectors;
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(mpiPatternP2P,
                                                                numVectors);
    }

    /**
     * @brief Constructor for a \distributed MultiVector based on locally
     * owned and ghost indices.
     * @note This way of construction is expensive. One should use the other
     * constructor based on an input MPIPatternP2P as far as possible.
     */
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(
      const std::pair<global_size_type, global_size_type> locallyOwnedRange,
      const std::vector<global_size_type> &               ghostIndices,
      const MPI_Comm &                                    mpiComm,
      const size_type                                     numVectors,
      const ValueType initVal /* = utils::Types<ValueType>::zero*/)
    {
      //
      // TODO Move the warning message to a Logger class
      //
      int         mpiRank;
      int         err = MPI_Comm_rank(mpiComm, &mpiRank);
      std::string msg;
      if (mpiRank == 0)
        {
          msg =
            "WARNING: Constructing a distributed MultiVector using locally owned "
            "range and ghost indices is expensive. As far as possible, one should "
            " use the constructor based on an input MPIPatternP2P.";
          std::cout << msg << std::endl;
        }
      ////////////

      d_mpiPatternP2P =
        std::make_shared<const utils::mpi::MPIPatternP2P<memorySpace>>(
          locallyOwnedRange, ghostIndices, mpiComm);

      d_mpiCommunicatorP2P = std::make_unique<
        const utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(
        d_mpiPatternP2P, numVectors);

      d_globalSize       = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize        = d_mpiPatternP2P->localGhostSize();
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_numVectors       = numVectors;
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          d_localSize * numVectors, initVal);
    }

    /**
     * @brief Constructor for a special case of \b distributed MultiVector where none
     * none of the processors have any ghost indices.
     * @note This way of construction is expensive. One should use the other
     * constructor based on an input MPIPatternP2P as far as possible.
     */
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(
      const std::pair<global_size_type, global_size_type> locallyOwnedRange,
      const MPI_Comm &                                    mpiComm,
      const size_type                                     numVectors,
      const ValueType initVal /* = utils::Types<ValueType>::zero*/)
    {
      //
      // TODO Move the warning message to a Logger class
      //
      int         mpiRank;
      int         err = MPI_Comm_rank(mpiComm, &mpiRank);
      std::string msg;
      if (mpiRank == 0)
        {
          msg =
            "WARNING: Constructing a distributed MultiVector using only locally owned "
            "range is expensive. As far as possible, one should "
            " use the constructor based on an input MPIPatternP2P.";
          std::cout << msg << std::endl;
        }
      ////////////
      std::vector<dftfe::global_size_type> ghostIndices;
      ghostIndices.resize(0);
      d_mpiPatternP2P =
        std::make_shared<const utils::mpi::MPIPatternP2P<memorySpace>>(
          locallyOwnedRange, ghostIndices, mpiComm);

      d_mpiCommunicatorP2P = std::make_unique<
        const utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(
        d_mpiPatternP2P, numVectors);

      d_globalSize       = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize        = d_mpiPatternP2P->localGhostSize();
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_numVectors       = numVectors;
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          d_localSize * numVectors, initVal);
    }


    /**
     * @brief Constructor for a \b distributed MultiVector based on total number of global indices.
     * The resulting MultiVector will not contain any ghost indices on any of
     * the processors. Internally, the vector is divided to ensure as much
     * equitable distribution across all the processors much as possible.
     * @note This way of construction is expensive. One should use the other
     * constructor based on an input MPIPatternP2P as far as possible.
     * Further, the decomposition is not compatible with other ways of
     * distributed MultiVector construction.
     */
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(
      const global_size_type globalSize,
      const MPI_Comm &       mpiComm,
      const size_type        numVectors,
      const ValueType        initVal /* = utils::Types<ValueType>::zero*/)
    {
      std::vector<dftfe::global_size_type> ghostIndices;
      ghostIndices.resize(0);

      std::pair<global_size_type, global_size_type> locallyOwnedRange;

      //
      // TODO Move the warning message to a Logger class
      //
      int         mpiRank;
      int         err = MPI_Comm_rank(mpiComm, &mpiRank);
      std::string msg;
      if (mpiRank == 0)
        {
          msg =
            "WARNING: Constructing a MultiVector using total number of indices across all processors "
            "is expensive. As far as possible, one should "
            " use the constructor based on an input MPIPatternP2P.";
          std::cout << msg << std::endl;
        }

      int mpiProcess;
      int errProc = MPI_Comm_size(mpiComm, &mpiProcess);

      dftfe::global_size_type locallyOwnedSize = globalSize / mpiProcess;
      if (mpiRank < globalSize % mpiProcess)
        locallyOwnedSize++;

      dftfe::global_size_type startIndex = mpiRank * (globalSize / mpiProcess);
      if (mpiRank < globalSize % mpiProcess)
        startIndex += mpiRank;
      else
        startIndex += globalSize % mpiProcess;

      locallyOwnedRange.first  = startIndex;
      locallyOwnedRange.second = startIndex + locallyOwnedSize;


      ////////////

      d_mpiPatternP2P =
        std::make_shared<const utils::mpi::MPIPatternP2P<memorySpace>>(
          locallyOwnedRange, ghostIndices, mpiComm);

      d_mpiCommunicatorP2P = std::make_unique<
        const utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(
        d_mpiPatternP2P, numVectors);

      d_globalSize       = d_mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize = d_mpiPatternP2P->localOwnedSize();
      d_ghostSize        = d_mpiPatternP2P->localGhostSize();
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_numVectors       = numVectors;
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          d_localSize * numVectors, initVal);
    }



    //
    // Copy Constructor
    //
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(const MultiVector &u)
    {
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          (u.d_storage)->size());
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(
        u.d_mpiPatternP2P, u.d_numVectors);
      *d_storage         = *(u.d_storage);
      d_localSize        = u.d_localSize;
      d_locallyOwnedSize = u.d_locallyOwnedSize;
      d_ghostSize        = u.d_ghostSize;
      d_globalSize       = u.d_globalSize;
      d_numVectors       = u.d_numVectors;
      d_mpiPatternP2P    = u.d_mpiPatternP2P;
    }

    //
    // Copy Constructor with reinitialization
    //
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(
      const MultiVector &u,
      const ValueType    initVal /* = utils::Types<ValueType>::zero*/)
    {
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          (u.d_storage)->size(), initVal);
      d_localSize          = u.d_localSize;
      d_locallyOwnedSize   = u.d_locallyOwnedSize;
      d_ghostSize          = u.d_ghostSize;
      d_globalSize         = u.d_globalSize;
      d_numVectors         = u.d_numVectors;
      d_mpiPatternP2P      = u.d_mpiPatternP2P;
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(d_mpiPatternP2P,
                                                                d_numVectors);
    }

    //
    // Move Constructor
    //
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace>::MultiVector(MultiVector &&u) noexcept
    {
      d_storage            = std::move(u.d_storage);
      d_localSize          = std::move(u.d_localSize);
      d_locallyOwnedSize   = std::move(u.d_locallyOwnedSize);
      d_ghostSize          = std::move(u.d_ghostSize);
      d_globalSize         = std::move(u.d_globalSize);
      d_numVectors         = std::move(u.d_numVectors);
      d_mpiCommunicatorP2P = std::move(u.d_mpiCommunicatorP2P);
      d_mpiPatternP2P      = std::move(u.d_mpiPatternP2P);
    }

    //
    // Copy Assignment
    //
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace> &
    MultiVector<ValueType, memorySpace>::operator=(const MultiVector &u)
    {
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          (u.d_storage)->size());
      *d_storage           = *(u.d_storage);
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(
        u.d_mpiPatternP2P, u.d_numVectors);
      d_localSize        = u.d_localSize;
      d_locallyOwnedSize = u.d_locallyOwnedSize;
      d_ghostSize        = u.d_ghostSize;
      d_globalSize       = u.d_globalSize;
      d_numVectors       = u.d_numVectors;
      d_mpiPatternP2P    = u.d_mpiPatternP2P;
      return *this;
    }

    //
    // Move Assignment
    //
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    MultiVector<ValueType, memorySpace> &
    MultiVector<ValueType, memorySpace>::operator=(MultiVector &&u)
    {
      d_storage            = std::move(u.d_storage);
      d_localSize          = std::move(u.d_localSize);
      d_locallyOwnedSize   = std::move(u.d_locallyOwnedSize);
      d_ghostSize          = std::move(u.d_ghostSize);
      d_globalSize         = std::move(u.d_globalSize);
      d_numVectors         = std::move(u.d_numVectors);
      d_mpiCommunicatorP2P = std::move(u.d_mpiCommunicatorP2P);
      d_mpiPatternP2P      = std::move(u.d_mpiPatternP2P);
      return *this;
    }


    //
    // swap
    //
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::swap(MultiVector &u)
    {
      d_storage.swap(u.d_storage);
      d_mpiCommunicatorP2P.swap(u.d_mpiCommunicatorP2P);
      d_mpiPatternP2P.swap(u.d_mpiPatternP2P);

      const size_type        tempLocalSizeLeft        = d_localSize;
      const size_type        tempLocallyOwnedSizeLeft = d_locallyOwnedSize;
      const size_type        tempGhostSizeLeft        = d_ghostSize;
      const global_size_type tempGlobalSizeLeft       = d_globalSize;
      const size_type        tempNumVectorsLeft       = d_numVectors;

      d_localSize        = u.d_localSize;
      d_locallyOwnedSize = u.d_locallyOwnedSize;
      d_ghostSize        = u.d_ghostSize;
      d_globalSize       = u.d_globalSize;
      d_numVectors       = u.d_numVectors;

      u.d_localSize        = tempLocalSizeLeft;
      u.d_locallyOwnedSize = tempLocallyOwnedSizeLeft;
      u.d_ghostSize        = tempGhostSizeLeft;
      u.d_globalSize       = tempGlobalSizeLeft;
      u.d_numVectors       = tempNumVectorsLeft;
    }

    //
    // reinit for \distributed MultiVector using an existing
    // MPIPatternP2P object
    //
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::reinit(
      std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
                      mpiPatternP2P,
      const size_type numVectors,
      const ValueType initVal)
    {
      d_globalSize       = mpiPatternP2P->nGlobalIndices();
      d_locallyOwnedSize = mpiPatternP2P->localOwnedSize();
      d_ghostSize        = mpiPatternP2P->localGhostSize();
      d_localSize        = d_locallyOwnedSize + d_ghostSize;
      d_numVectors       = numVectors;
      d_storage =
        std::make_unique<typename MultiVector<ValueType, memorySpace>::Storage>(
          d_localSize * d_numVectors, initVal);
      d_mpiPatternP2P      = mpiPatternP2P;
      d_mpiCommunicatorP2P = std::make_unique<
        utils::mpi::MPICommunicatorP2P<ValueType, memorySpace>>(mpiPatternP2P,
                                                                numVectors);
    }

    //
    // reinit for \distributed MultiVector using an existing
    // MultiVector object
    //
    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::reinit(const MultiVector &u)
    {
      this->reinit(u.d_mpiPatternP2P, u.d_numVectors);
    }


    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename MultiVector<ValueType, memorySpace>::iterator
    MultiVector<ValueType, memorySpace>::begin()
    {
      return d_storage->begin();
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename MultiVector<ValueType, memorySpace>::const_iterator
    MultiVector<ValueType, memorySpace>::begin() const
    {
      return d_storage->begin();
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename MultiVector<ValueType, memorySpace>::iterator
    MultiVector<ValueType, memorySpace>::end()
    {
      return d_storage->end();
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    typename MultiVector<ValueType, memorySpace>::const_iterator
    MultiVector<ValueType, memorySpace>::end() const
    {
      return d_storage->end();
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    ValueType *
    MultiVector<ValueType, memorySpace>::data()
    {
      return d_storage->data();
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    const ValueType *
    MultiVector<ValueType, memorySpace>::data() const
    {
      return d_storage->data();
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::setValue(const ValueType val)
    {
      d_storage->setValue(val);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::zeroOutGhosts()
    {
      dftfe::utils::MemoryManager<ValueType, memorySpace>::set(
        d_ghostSize * d_numVectors,
        this->data() + d_locallyOwnedSize * d_numVectors,
        0);
    }


    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::updateGhostValues(
      const size_type communicationChannel /*= 0*/)
    {
      d_mpiCommunicatorP2P->updateGhostValues(*d_storage, communicationChannel);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::accumulateAddLocallyOwned(
      const size_type communicationChannel /*= 0*/)
    {
      d_mpiCommunicatorP2P->accumulateAddLocallyOwned(*d_storage,
                                                      communicationChannel);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::updateGhostValuesBegin(
      const size_type communicationChannel /*= 0*/)
    {
      d_mpiCommunicatorP2P->updateGhostValuesBegin(*d_storage,
                                                   communicationChannel);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::updateGhostValuesEnd()
    {
      d_mpiCommunicatorP2P->updateGhostValuesEnd(*d_storage);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::accumulateAddLocallyOwnedBegin(
      const size_type communicationChannel /*= 0*/)
    {
      d_mpiCommunicatorP2P->accumulateAddLocallyOwnedBegin(
        *d_storage, communicationChannel);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::accumulateAddLocallyOwnedEnd()
    {
      d_mpiCommunicatorP2P->accumulateAddLocallyOwnedEnd(*d_storage);
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    bool
    MultiVector<ValueType, memorySpace>::isCompatible(
      const MultiVector<ValueType, memorySpace> &rhs) const
    {
      if (d_numVectors != rhs.d_numVectors)
        return false;
      else if (d_globalSize != rhs.d_globalSize)
        return false;
      else if (d_localSize != rhs.d_localSize)
        return false;
      else if (d_locallyOwnedSize != rhs.d_locallyOwnedSize)
        return false;
      else if (d_ghostSize != rhs.d_ghostSize)
        return false;
      else
        return (d_mpiPatternP2P->isCompatible(*(rhs.d_mpiPatternP2P)));
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    std::shared_ptr<const utils::mpi::MPIPatternP2P<memorySpace>>
    MultiVector<ValueType, memorySpace>::getMPIPatternP2P() const
    {
      return d_mpiPatternP2P;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    global_size_type
    MultiVector<ValueType, memorySpace>::globalSize() const
    {
      return d_globalSize;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    size_type
    MultiVector<ValueType, memorySpace>::localSize() const
    {
      return d_localSize;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    size_type
    MultiVector<ValueType, memorySpace>::locallyOwnedSize() const
    {
      return d_locallyOwnedSize;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    size_type
    MultiVector<ValueType, memorySpace>::ghostSize() const
    {
      return d_ghostSize;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    size_type
    MultiVector<ValueType, memorySpace>::numVectors() const
    {
      return d_numVectors;
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    template <typename ValueBaseType>
    void
    MultiVector<ValueType, memorySpace>::l2Norm(ValueBaseType *normVec) const
    {
      dftfe::utils::throwException<dftfe::utils::InvalidArgument>(
        memorySpace != dftfe::utils::MemorySpace::DEVICE,
        "[] L2-Norm evaluation not implemented for DEVICE");
      if (d_locallyOwnedSize > 0)
        std::transform(begin(), begin() + d_numVectors, normVec, [](auto &a) {
          return dftfe::utils::realPart(dftfe::utils::complexConj(a) * (a));
        });
      for (auto k = 1; k < d_locallyOwnedSize; ++k)
        {
          std::transform(begin() + k * d_numVectors,
                         begin() + (k + 1) * d_numVectors,
                         normVec,
                         normVec,
                         [](auto &a, auto &b) {
                           return b + dftfe::utils::realPart(
                                        dftfe::utils::complexConj(a) * (a));
                         });
        }
      MPI_Allreduce(MPI_IN_PLACE,
                    normVec,
                    d_numVectors,
                    dataTypes::mpi_type_id(normVec),
                    MPI_SUM,
                    d_mpiPatternP2P->mpiCommunicator());
      std::transform(normVec, normVec + d_numVectors, normVec, [](auto &a) {
        return std::sqrt(a);
      });
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    template <typename ValueBaseType>
    void
    MultiVector<ValueType, memorySpace>::add(const ValueBaseType *valVec,
                                             const MultiVector &  u)
    {
      dftfe::utils::throwException<dftfe::utils::InvalidArgument>(
        memorySpace != dftfe::utils::MemorySpace::DEVICE,
        "[] Add not implemented for DEVICE");
      for (auto k = 0; k < d_locallyOwnedSize; ++k)
        for (auto ib = 0; ib < d_numVectors; ++ib)
          (*d_storage)[k * d_numVectors + ib] +=
            valVec[ib] * u.data()[k * d_numVectors + ib];
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    template <typename ValueBaseType>
    void
    MultiVector<ValueType, memorySpace>::add(const ValueBaseType val,
                                             const MultiVector & u)
    {
      dftfe::utils::throwException<dftfe::utils::InvalidArgument>(
        memorySpace != dftfe::utils::MemorySpace::DEVICE,
        "[] Add not implemented for DEVICE");
      std::transform(begin(),
                     begin() + d_locallyOwnedSize * d_numVectors,
                     u.begin(),
                     begin(),
                     [&val](auto &a, auto &b) { return (a + val * b); });
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    template <typename ValueBaseType1, typename ValueBaseType2>
    void
    MultiVector<ValueType, memorySpace>::addAndScale(
      const ValueBaseType1 valScale,
      const ValueBaseType2 valAdd,
      const MultiVector &  u)
    {
      dftfe::utils::throwException<dftfe::utils::InvalidArgument>(
        memorySpace != dftfe::utils::MemorySpace::DEVICE,
        "[] Add not implemented for DEVICE");
      std::transform(begin(),
                     begin() + d_locallyOwnedSize * d_numVectors,
                     u.begin(),
                     begin(),
                     [&valScale, &valAdd](auto &a, auto &b) {
                       return valScale * (a + valAdd * b);
                     });
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    template <typename ValueBaseType1, typename ValueBaseType2>
    void
    MultiVector<ValueType, memorySpace>::scaleAndAdd(
      const ValueBaseType1 valScale,
      const ValueBaseType2 valAdd,
      const MultiVector &  u)
    {
      dftfe::utils::throwException<dftfe::utils::InvalidArgument>(
        memorySpace != dftfe::utils::MemorySpace::DEVICE,
        "[] Add not implemented for DEVICE");
      std::transform(begin(),
                     begin() + d_locallyOwnedSize * d_numVectors,
                     u.begin(),
                     begin(),
                     [&valScale, &valAdd](auto &a, auto &b) {
                       return valScale * a + valAdd * b;
                     });
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    template <typename ValueBaseType>
    void
    MultiVector<ValueType, memorySpace>::scale(const ValueBaseType val)
    {
      dftfe::utils::throwException<dftfe::utils::InvalidArgument>(
        memorySpace != dftfe::utils::MemorySpace::DEVICE,
        "[] Add not implemented for DEVICE");
      std::transform(begin(),
                     begin() + d_locallyOwnedSize * d_numVectors,
                     begin(),
                     [&val](auto &a) { return val * a; });
    }

    template <typename ValueType, dftfe::utils::MemorySpace memorySpace>
    void
    MultiVector<ValueType, memorySpace>::dot(const MultiVector &u,
                                             ValueType *        dotVec)
    {
      dftfe::utils::throwException<dftfe::utils::InvalidArgument>(
        memorySpace != dftfe::utils::MemorySpace::DEVICE,
        "[] dot product evaluation not implemented for DEVICE");
      if (d_locallyOwnedSize > 0)
        for (auto ib = 0; ib < d_numVectors; ++ib)
          {
            dotVec[ib] = (*d_storage)[ib] *
                         dftfe::utils::complexConj((*(u.d_storage))[ib]);
          }
      else
        for (auto ib = 0; ib < d_numVectors; ++ib)
          {
            dotVec[ib] = 0.0;
          }
      for (auto k = 1; k < d_locallyOwnedSize; ++k)
        {
          for (auto ib = 0; ib < d_numVectors; ++ib)
            {
              dotVec[ib] += (*d_storage)[k * d_numVectors + ib] *
                            dftfe::utils::complexConj(
                              (*(u.d_storage))[k * d_numVectors + ib]);
            }
        }
      MPI_Allreduce(MPI_IN_PLACE,
                    dotVec,
                    d_numVectors,
                    dataTypes::mpi_type_id(dotVec),
                    MPI_SUM,
                    d_mpiPatternP2P->mpiCommunicator());
    }

    template <typename ValueType, utils::MemorySpace memorySpace>
    void
    createMultiVectorFromDealiiPartitioner(
      const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
        &                                  partitioner,
      const size_type                      numVectors,
      MultiVector<ValueType, memorySpace> &multiVector)
    {
      const std::pair<global_size_type, global_size_type> &locallyOwnedRange =
        partitioner->local_range();
      // std::cout<<locallyOwnedRange.first<<"
      // "<<locallyOwnedRange.second<<std::endl;
      std::vector<global_size_type> ghostIndices;
      (partitioner->ghost_indices()).fill_index_vector(ghostIndices);

      // for (unsigned int i=0;i<ghostIndices.size();++i)
      // if (ghostIndices.size()>0)
      // std::cout<<ghostIndices.back()<<std::endl;

      // std::sort(ghostIndices.begin(),ghostIndices.end());
      std::shared_ptr<dftfe::utils::mpi::MPIPatternP2P<memorySpace>>
        mpiPatternP2PPtr =
          std::make_shared<dftfe::utils::mpi::MPIPatternP2P<memorySpace>>(
            locallyOwnedRange,
            ghostIndices,
            partitioner->get_mpi_communicator());

      multiVector.reinit(mpiPatternP2PPtr, numVectors);
    }

  } // end of namespace linearAlgebra
} // namespace dftfe
