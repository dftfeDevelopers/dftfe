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

#include <iterator>
#include <algorithm>
#include <type_traits>
#include <Exceptions.h>
namespace dftfe
{
  namespace utils
  {
    //
    // Constructor
    //
    template <typename T>
    OptimizedIndexSet<T>::OptimizedIndexSet(
      const std::set<T> &inputSet /*=std::set<T>()*/)
      : d_numContiguousRanges(0)
      , d_contiguousRanges(0)
      , d_numEntriesBefore(0)
    {
      bool isValid = std::is_same<size_type, T>::value ||
                     std::is_same<global_size_type, T>::value;
      utils::throwException<utils::InvalidArgument>(
        isValid,
        "OptimizedIndexSet expects the template parameter to be of type unsigned int or unsigned long int.");
      if (!inputSet.empty())
        {
          typename std::set<T>::const_iterator itLastRange = inputSet.begin();
          typename std::set<T>::const_iterator itPrev      = inputSet.begin();
          typename std::set<T>::const_iterator it          = itPrev;
          it++;
          for (; it != inputSet.end(); ++it)
            {
              bool isContiguous = ((*it - 1) == *(itPrev));
              if (!isContiguous)
                {
                  d_contiguousRanges.push_back(*itLastRange);
                  d_contiguousRanges.push_back(*(itPrev) + 1);
                  itLastRange = it;
                }
              itPrev = it;
            }

          d_contiguousRanges.push_back(*itLastRange);
          d_contiguousRanges.push_back(*(itPrev) + 1);

          d_numContiguousRanges = d_contiguousRanges.size() / 2;
          d_numEntriesBefore.resize(d_numContiguousRanges, 0);
          size_type cumulativeEntries = 0;
          for (unsigned int i = 0; i < d_numContiguousRanges; ++i)
            {
              d_numEntriesBefore[i] = cumulativeEntries;
              cumulativeEntries +=
                d_contiguousRanges[2 * i + 1] - d_contiguousRanges[2 * i];
            }
        }
    }

    template <typename T>
    void
    OptimizedIndexSet<T>::getPosition(const T &  index,
                                      size_type &pos,
                                      bool &     found) const
    {
      found = false;
      /*
       * The logic used for finding an index is as follows:
       * 1. Find the position of the element in d_contiguousRanges
       *    which is greater than (strictly greater) the input index.
       *    Let's call this position as upPos and the value at upPos as
       *    upVal. The complexity of finding it is
       *    O(log(size of d_contiguousRanges))
       * 2. Since d_contiguousRanges stores pairs of startId and endId
       *    (endId not inclusive) of contiguous ranges in inputSet,
       *    any index for which upPos is even (i.e., it corresponds to a
       * startId) cannot belong to inputSet. Why? Consider two consequtive
       * ranges [k1,k2) and [k3,k4) where k1 < k2 < k3 < k4. If upVal for index
       * corresponds to k3 (i.e., startId of a range), then (a) index does not
       * lie in the [k3,k4) as index < upVal (=k3). (b) index cannot lie in
       * [k1,k2), because if index lies in [k1,k2), then upVal should be k2 (not
       * k3)
       *  3. If upPos is odd (i.e, it corresponds to an endId), we find the
       * relative position of index in that range. Subsequently, we determine
       * the global position of index in inputSet by adding the relative
       * position to the number of entries in inputSet prior to the range where
       * index lies
       */

      auto up = std::upper_bound(d_contiguousRanges.begin(),
                                 d_contiguousRanges.end(),
                                 index);
      if (up != d_contiguousRanges.end())
        {
          size_type upPos = std::distance(d_contiguousRanges.begin(), up);
          if (upPos % 2 == 1)
            {
              found             = true;
              size_type rangeId = upPos / 2;
              pos               = d_numEntriesBefore[rangeId] + index -
                    d_contiguousRanges[upPos - 1];
            }
        }
    }

    template <typename T>
    bool
    OptimizedIndexSet<T>::getPosition(const OptimizedIndexSet<T> &rhs) const
    {
      if (d_numContiguousRanges != rhs.d_numContiguousRanges)
        return false;
      else
        return (d_contiguousRanges == rhs.d_contiguousRanges);
    }
  } // end of namespace utils
} // end of namespace dftfe
