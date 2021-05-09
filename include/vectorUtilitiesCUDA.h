// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018  The Regents of the University of Michigan and DFT-FE
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

#if defined(DFTFE_WITH_GPU)
#  ifndef vectorUtilitiesCUDA_h
#    define vectorUtilitiesCUDA_h

#    include <headers.h>
#    include <thrust/device_vector.h>


namespace dftfe
{
  /**
   *  @brief Contains generic utils functions
   *
   *  @author Sambit Das
   */
  namespace vectorToolsCUDA
  {
    class cudaThrustVector
    {
    public:
      cudaThrustVector();

      void
      resize(const unsigned int size);

      double *
      begin();

      const double *
      begin() const;

      unsigned int
      size() const;

      double
      l2_norm() const;

      double
      l1_norm() const;

    private:
      thrust::device_vector<double> d_data;
    };

    void
    copyHostVecToCUDAVec(const double *     hostVec,
                         double *           cudaVector,
                         const unsigned int size);

    void
    copyCUDAVecToHostVec(const double *     cudaVector,
                         double *           hostVec,
                         const unsigned int size);
  } // namespace vectorToolsCUDA
} // namespace dftfe
#  endif
#endif
