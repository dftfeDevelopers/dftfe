
//
// -------------------------------------------------------------------------------------
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
// --------------------------------------------------------------------------------------
//
// @author Phani Motamarri, Sambit Das
//
#include <operatorDevice.h>
#include <linearAlgebraOperationsInternal.h>

//
// Constructor.
//
namespace dftfe
{
  operatorDFTDeviceClass::operatorDFTDeviceClass(
    const MPI_Comm &                      mpi_comm_replica,
    const dealii::MatrixFree<3, double> & matrix_free_data,
    dftUtils::constraintMatrixInfo &      constraintMatrixNone,
    dftUtils::constraintMatrixInfoDevice &constraintMatrixNoneDevice)
    : d_mpi_communicator(mpi_comm_replica)
    , d_matrix_free_data(&matrix_free_data)
    , d_constraintMatrixData(&constraintMatrixNone)
    , d_constraintMatrixDataDevice(&constraintMatrixNoneDevice)
  {}

  //
  // Destructor.
  //
  operatorDFTDeviceClass::~operatorDFTDeviceClass()
  {
    //
    //
    //
    return;
  }

  //
  // Get overloaded constraint matrix object constructed using 1-component FE
  // object
  //
  dftUtils::constraintMatrixInfo *
  operatorDFTDeviceClass::getOverloadedConstraintMatrixHost() const
  {
    return d_constraintMatrixData;
  }

  //
  // Get overloaded constraint matrix object constructed using 1-component FE
  // object
  //
  dftUtils::constraintMatrixInfoDevice *
  operatorDFTDeviceClass::getOverloadedConstraintMatrix() const
  {
    return d_constraintMatrixDataDevice;
  }

  //
  // Get matrix free data
  //
  const dealii::MatrixFree<3, double> *
  operatorDFTDeviceClass::getMatrixFreeData() const
  {
    return d_matrix_free_data;
  }

  //
  // Get relevant mpi communicator
  //
  const MPI_Comm &
  operatorDFTDeviceClass::getMPICommunicator() const
  {
    return d_mpi_communicator;
  }
} // namespace dftfe
