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
 * @author Bikash Kanungo
 */

#ifndef dftfeMPIRequestersBase_h
#define dftfeMPIRequestersBase_h
#include <TypeConfig.h>
#include <vector>
namespace dftfe
{
  namespace utils
  {
    namespace mpi
    {
      class MPIRequestersBase
      {
        /*
         *
         * @brief A pure virtual class to evaluate the list of rank Ids that the
         * current processor needs to send data.
         *
         * In a typical case of distributed data (a vector or array), a
         * processor needs to communicate part of its part of the data to a
         * requesting processor. It is useful for the current processor to know
         * apriori which processors it has to send its part of the distributed
         * data. This base class provides an interface to indentify the Ids of
         * the processors (also known as ranks) that it has to send data to. In
         * MPI parlance, the other processors to which this processor needs to
         * send data are termed as requesting processors/ranks.
         *
         * The actual process of identifying the list of requesting processors
         * is  implemented in the derived classes. There are various different
         * algorithms with varying computational/communication complexity. Some
         * use cases are trivial, for example, (a) a serial case where
         * there are no requesting processors, (b) an all-to-all communication
         * case where all the other processors are requesting from the
         * current proccesor.
         *
         */

      public:
        virtual ~MPIRequestersBase() = default;
        virtual std::vector<size_type>
        getRequestingRankIds() = 0;
      };

    } // end of namespace mpi
  }   // end of namespace utils
} // end of namespace dftfe
#endif // dftfeMPIRequestersBase_h
