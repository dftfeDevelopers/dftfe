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
#ifndef dftfeOptimizedIndexSet_h
#define dftfeOptimizedIndexSet_h

#include <TypeConfig.h>
#include <set>
#include <vector>
namespace dftfe
{
  namespace utils
  {
    /*
     * @brief Class to create an optimized index set which
     * creates contiguous sub-ranges within an index set for faster
     * search operation. This is useful when the number of contiguous sub-ranges
     * are fewer compared to the size of the index set. If the number of
     * contiguous sub-ranges competes with the size of the index set (i.e., the
     * index set is very random) then it default to the behavior of an std::set.
     *
     * @tparam ValueType The data type of the indices (e.g., unsigned int, unsigned long int)
     */

    template <typename T>
    class OptimizedIndexSet
    {
    public:
      /**
       * @brief Constructor
       *
       * @param[in] inputSet A set of unsigned int or unsigned long int
       * for which an OptimizedIndexSet is to be created
       */
      OptimizedIndexSet(const std::set<T> &inputSet = std::set<T>());
      ~OptimizedIndexSet() = default;

      void
      getPosition(const T &index, size_type &pos, bool &found) const;

      bool
      getPosition(const OptimizedIndexSet<T> &rhs) const;


    private:
      /// Store the number of contiguous ranges in the input set of indices
      size_type d_numContiguousRanges;

      /*
       * Vector of size 2*(d_numContiguousRanges in d_set).
       * The entries are arranged as:
       * <contiguous range1 startId> <continguous range1 endId> <contiguous
       * range2 startId> <continguous range2 endId> ... NOTE: The endId is one
       * past the lastId in the continguous range
       */
      std::vector<T> d_contiguousRanges;

      /// Vector of size d_numContiguousRanges which stores the accumulated
      /// number of elements in d_set prior to the i-th contiguous range
      std::vector<size_type> d_numEntriesBefore;

      bool
      operator==(const OptimizedIndexSet<T> &rhs) const;
    };

  } // end of namespace utils

} // end of namespace dftfe
#include "../utils/OptimizedIndexSet.t.cc"
#endif // dftfeOptimizedSet_h
