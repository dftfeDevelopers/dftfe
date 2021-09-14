// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2021  The Regents of the University of Michigan and DFT-FE
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

#ifndef distributedDealiiBlockVec_h
#define distributedDealiiBlockVec_h

#include <headers.h>
#include "memorySpace.h"

namespace dftfe
{
  /**
   *  @brief Contains distributed block vector class acting as a wrapper over the
   *  dealii vector class
   *
   *  @author Sambit Das
   */
  template <typename NumberType,
            typename MemorySpace = dftfe::MemorySpace::Host>
  class DistributedDealiiBlockVec
  {
  public:
    DistributedDealiiBlockVec();

    ~DistributedDealiiBlockVec();

    void
    reinit(const std::shared_ptr<const dealii::Utilities::MPI::Partitioner>
             &                partitionerSingleVec,
           const unsigned int blockSize);

    NumberType *
    begin();

    const NumberType *
    begin() const;

    unsigned int
    size() const;

    unsigned int
    dofsSize() const;

    unsigned int
    blockSize() const;

    void
    updateGhostValues();

    void
    compressAdd();

  private:
    NumberType *d_vecData;

    void *d_dealiiVecData;

    void *d_dealiiVecDataReal;

    void *d_dealiiVecDataImag;

    unsigned int d_size;

    unsigned int d_dofsSize;

    unsigned int d_blockSize;
  };
} // namespace dftfe
#endif
