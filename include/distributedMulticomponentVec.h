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

#ifndef DistributedMulticomponentVec_h
#define DistributedMulticomponentVec_h

#include "dftfeDataTypes.h"
#include "memorySpace.h"
#include <deal.II/base/partitioner.h>

namespace dftfe
{
  /**
   *  @brief Contains distributed multicomponent vector class acting as a wrapper over the
   *  dealii vector class
   *
   *  @author Sambit Das
   */
  template <typename NumberType,
            typename MemorySpace = dftfe::MemorySpace::Host>
  class DistributedMulticomponentVec
  {
  public:
    DistributedMulticomponentVec();

    ~DistributedMulticomponentVec();

    void
    reinit(const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
             &                              partitionerSingleVec,
           const dataTypes::local_size_type numberComponents);

    void
    reinit(const DistributedMulticomponentVec<NumberType, MemorySpace> &vec);

    void
    setZero();


    NumberType *
    begin();

    const NumberType *
    begin() const;

    dataTypes::global_size_type
    globalSize() const;

    dataTypes::local_size_type
    locallyOwnedFlattenedSize() const;

    dataTypes::local_size_type
    ghostFlattenedSize() const;

    dataTypes::local_size_type
    locallyOwnedDofsSize() const;

    dataTypes::local_size_type
    numberComponents() const;

    void
    updateGhostValues();

    void
    updateGhostValuesStart();


    void
    updateGhostValuesFinish();

    void
    compressAdd();

    void
    compressAddStart();

    void
    compressAddFinish();


    void
    zeroOutGhosts();

    void
    swap(DistributedMulticomponentVec<NumberType, MemorySpace> &vec);

    void
    clear();

    const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
    getDealiiPartitioner() const;


  private:
    const void *
    getDealiiVec() const;

    NumberType *d_vecData;

    void *d_dealiiVecData;

    void *d_dealiiVecTempDataReal;

    void *d_dealiiVecTempDataImag;

    dataTypes::global_size_type d_globalSize;

    dataTypes::local_size_type d_locallyOwnedSize;

    dataTypes::local_size_type d_ghostSize;

    dataTypes::local_size_type d_locallyOwnedDofsSize;

    dataTypes::local_size_type d_numberComponents;
  };
} // namespace dftfe
#endif
