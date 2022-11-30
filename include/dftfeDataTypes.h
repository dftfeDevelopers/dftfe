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
//
// @author Sambit Das
//

#ifndef dftfeDataTypes_H_
#define dftfeDataTypes_H_

#include <deal.II/base/config.h>
#include <deal.II/base/types.h>

#if defined(DFTFE_WITH_DEVICE)
#  include <cuComplex.h>
#  include <thrust/device_vector.h>
#  include <thrust/complex.h>
#endif

// Include generic C++ headers
#include <fstream>
#include <iostream>
#include <fenv.h>

// commonly used  typedefs used in dftfe go here
namespace dftfe
{
  namespace dataTypes
  {
    typedef dealii::types::global_dof_index global_size_type;
    typedef unsigned int                    local_size_type;
#ifdef USE_COMPLEX
    typedef std::complex<double> number;
    typedef std::complex<float>  numberFP32;
    typedef double               numberValueType;
    typedef float                numberFP32ValueType;
#  if defined(DFTFE_WITH_DEVICE)
    typedef cuDoubleComplex         numberGPU;
    typedef cuFloatComplex          numberFP32GPU;
    typedef thrust::complex<double> numberThrustGPU;
    typedef thrust::complex<float>  numberFP32ThrustGPU;
#  endif
#else
    typedef double number;
    typedef float  numberFP32;
    typedef double numberValueType;
    typedef float  numberFP32ValueType;
#  if defined(DFTFE_WITH_DEVICE)
    typedef double numberGPU;
    typedef float  numberFP32GPU;
    typedef double numberThrustGPU;
    typedef float  numberFP32ThrustGPU;
#  endif
#endif

    inline MPI_Datatype
    mpi_type_id(const int *)
    {
      return MPI_INT;
    }

    inline MPI_Datatype
    mpi_type_id(const long int *)
    {
      return MPI_LONG;
    }

    inline MPI_Datatype
    mpi_type_id(const unsigned int *)
    {
      return MPI_UNSIGNED;
    }

    inline MPI_Datatype
    mpi_type_id(const unsigned long int *)
    {
      return MPI_UNSIGNED_LONG;
    }

    inline MPI_Datatype
    mpi_type_id(const unsigned long long int *)
    {
      return MPI_UNSIGNED_LONG_LONG;
    }


    inline MPI_Datatype
    mpi_type_id(const float *)
    {
      return MPI_FLOAT;
    }


    inline MPI_Datatype
    mpi_type_id(const double *)
    {
      return MPI_DOUBLE;
    }

    inline MPI_Datatype
    mpi_type_id(const long double *)
    {
      return MPI_LONG_DOUBLE;
    }

    inline MPI_Datatype
    mpi_type_id(const std::complex<float> *)
    {
      return MPI_COMPLEX;
    }

    inline MPI_Datatype
    mpi_type_id(const std::complex<double> *)
    {
      return MPI_DOUBLE_COMPLEX;
    }
  } // namespace dataTypes
} // namespace dftfe

#endif
