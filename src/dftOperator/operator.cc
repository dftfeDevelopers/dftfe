
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
// @author Phani Motamarri
//
#include <dftParameters.h>
#include <operator.h>

//
// Constructor.
//
namespace dftfe
{
  operatorDFTClass::operatorDFTClass(
    const MPI_Comm &                     mpi_comm_replica,
    const dealii::MatrixFree<3, double> &matrix_free_data,
    dftUtils::constraintMatrixInfo &     constraintMatrixNone)
    : d_mpi_communicator(mpi_comm_replica)
    , d_matrix_free_data(&matrix_free_data)
    , d_constraintMatrixData(&constraintMatrixNone)
  {}

  //
  // Destructor.
  //
  operatorDFTClass::~operatorDFTClass()
  {
    //
    //
    //
    return;
  }

  // set the data member of operator class
  void
  operatorDFTClass::setInvSqrtMassVector(
    distributedCPUVec<double> &invSqrtMassVector)
  {
    d_invSqrtMassVector = invSqrtMassVector;
  }

  // get access to the data member of operator class
  distributedCPUVec<double> &
  operatorDFTClass::getInvSqrtMassVector()
  {
    return d_invSqrtMassVector;
  }

  //
  // Get overloaded constraint matrix object constructed using 1-component FE
  // object
  //
  dftUtils::constraintMatrixInfo *
  operatorDFTClass::getOverloadedConstraintMatrix() const
  {
    return d_constraintMatrixData;
  }

  //
  // Get matrix free data
  //
  const dealii::MatrixFree<3, double> *
  operatorDFTClass::getMatrixFreeData() const
  {
    return d_matrix_free_data;
  }

  //
  // Get relevant mpi communicator
  //
  const MPI_Comm &
  operatorDFTClass::getMPICommunicator() const
  {
    return d_mpi_communicator;
  }
} // namespace dftfe
