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
 * @author Bikash Kanungo, Sambit Das
 */

#ifndef dftfeMPITags_h
#define dftfeMPITags_h

#include <TypeConfig.h>
#include <vector>
#include <cstdint>

#  include <mpi.h>

namespace dftfe
{
  namespace utils
  {
    namespace mpi
    {
      enum class MPITags : std::uint16_t
      {
        DUMMY_MPI_TAG = 100,
        MPI_REQUESTERS_NBX_TAG,
        MPI_P2P_PATTERN_TAG,

        MPI_P2P_COMMUNICATOR_SCATTER_TAG,

        MPI_P2P_COMMUNICATOR_GATHER_TAG = MPI_P2P_COMMUNICATOR_SCATTER_TAG + 200
      };
    } // end of namespace mpi
  }   // end of namespace utils
} // end of namespace dftfe
#endif // dftfeMPITags_h
