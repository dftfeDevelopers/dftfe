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


#ifndef dftfeDataTypeOverloads_h
#define dftfeDataTypeOverloads_h

#include <complex>
namespace dftfe
{
  namespace utils
   {
    inline double
    realPart(const double x)
    {
      return x;
    }

    inline float
    realPart(const float x)
    {
      return x;
    }

    inline double
    realPart(const std::complex<double> x)
    {
      return x.real();
    }

    inline float
    realPart(const std::complex<float> x)
    {
      return x.real();
    }

    inline double
    imagPart(const double x)
    { 
        return 0;
    }


    inline float
    imagPart(const float x)
    { 
        return 0;
    }

    inline double
    imagPart(const std::complex<double> x)
    { 
        return x.imag();
    }

    inline float
    imagPart(const std::complex<float> x)
    { 
        return x.imag();
    }

    inline double
    complexConj(const double x)
    {
      return x;
    }

    inline float
    complexConj(const float x)
    {
      return x;
    }

    inline std::complex<double>
    complexConj(const std::complex<double> x)
    {
      return std::conj(x);
    }

    inline std::complex<float>
    complexConj(const std::complex<float> x)
    {
      return std::conj(x);
    }
  }
} // namespace dftfe

#endif
