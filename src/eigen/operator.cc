//
// File:      operator.cc
// Package:   dft
//
// Density Functional Theory
//
// -------------------------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Phani Motamarri (2018)
//
#include "../../include/operator.h"

//
// Constructor.
//
namespace dftfe {
  operatorDFTClass::operatorDFTClass(const MPI_Comm                  & mpi_comm_replica,
				     const std::vector<dealii::types::global_dof_index> & localDofIndicesReal,
				     const std::vector<dealii::types::global_dof_index> & localDofIndicesImag,
				     const std::vector<dealii::types::global_dof_index> & localProcDofIndicesReal,
				     const std::vector<dealii::types::global_dof_index> & localProcDofIndicesImag,
				     const dealii::ConstraintMatrix  & constraintMatrixEigen):
    d_mpi_communicator   (mpi_comm_replica),
    d_localDofIndicesReal(&localDofIndicesReal),
    d_localDofIndicesImag(&localDofIndicesImag),
    d_localProcDofIndicesReal(&localProcDofIndicesReal),
    d_localProcDofIndicesImag(&localProcDofIndicesImag),
    d_constraintMatrixEigen(&constraintMatrixEigen)
  {

   

  }

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

  //
  //Get local dof indices real
  //
  const std::vector<dealii::types::global_dof_index> * operatorDFTClass::getLocalDofIndicesReal() const
  {
    return d_localDofIndicesReal;
  }

  //
  //Get local dof indices imag
  //
  const std::vector<dealii::types::global_dof_index> * operatorDFTClass::getLocalDofIndicesImag() const
  {
    return d_localDofIndicesImag;
  }

  //
  //Get local proc dof indices real
  //
  const std::vector<dealii::types::global_dof_index> * operatorDFTClass::getLocalProcDofIndicesReal() const
  {
    return d_localProcDofIndicesReal;
  }


  //
  //Get local proc dof indices imag
  //
  const std::vector<dealii::types::global_dof_index> * operatorDFTClass::getLocalProcDofIndicesImag() const
  {
    return d_localProcDofIndicesImag;
  }

  //
  //Get constraint matrix
  //
  const dealii::ConstraintMatrix * operatorDFTClass::getConstraintMatrixEigen() const
  {
    return d_constraintMatrixEigen;
  }

  //
  //Get relevant mpi communicator
  //
  const MPI_Comm & operatorDFTClass::getMPICommunicator() const
  {
    return d_mpi_communicator;
  }

}
