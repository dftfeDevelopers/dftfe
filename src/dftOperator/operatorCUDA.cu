
//
// -------------------------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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
#include <operatorCUDA.h>
#include <linearAlgebraOperationsInternalCUDA.h>
#include <dftParameters.h>

//
// Constructor.
//
namespace dftfe {

  operatorDFTCUDAClass::operatorDFTCUDAClass(const MPI_Comm                                        & mpi_comm_replica,
		        	             const dealii::MatrixFree<3,double>                    & matrix_free_data,
				             const std::vector<dealii::types::global_dof_index>    & localDofIndicesReal,
				             const std::vector<dealii::types::global_dof_index>    & localDofIndicesImag,
				             const std::vector<dealii::types::global_dof_index>    & localProcDofIndicesReal,
				             const std::vector<dealii::types::global_dof_index>    & localProcDofIndicesImag,
				             const dealii::AffineConstraints<double>                        & constraintMatrixEigen,
				             dftUtils::constraintMatrixInfoCUDA                    & constraintMatrixNone):
    d_mpi_communicator(mpi_comm_replica),
    d_matrix_free_data(&matrix_free_data),
    d_localDofIndicesReal(&localDofIndicesReal),
    d_localDofIndicesImag(&localDofIndicesImag),
    d_localProcDofIndicesReal(&localProcDofIndicesReal),
    d_localProcDofIndicesImag(&localProcDofIndicesImag),
    d_constraintMatrixEigen(&constraintMatrixEigen),
    d_constraintMatrixData(&constraintMatrixNone)
  {


  }

  //
  // Destructor.
  //
  operatorDFTCUDAClass::~operatorDFTCUDAClass()
  {

    //
    //
    //
    return;

  }

  //
  //Get local dof indices real
  //
  const std::vector<dealii::types::global_dof_index> * operatorDFTCUDAClass::getLocalDofIndicesReal() const
  {
    return d_localDofIndicesReal;
  }

  //
  //Get local dof indices imag
  //
  const std::vector<dealii::types::global_dof_index> * operatorDFTCUDAClass::getLocalDofIndicesImag() const
  {
    return d_localDofIndicesImag;
  }

  //
  //Get local proc dof indices real
  //
  const std::vector<dealii::types::global_dof_index> * operatorDFTCUDAClass::getLocalProcDofIndicesReal() const
  {
    return d_localProcDofIndicesReal;
  }


  //
  //Get local proc dof indices imag
  //
  const std::vector<dealii::types::global_dof_index> * operatorDFTCUDAClass::getLocalProcDofIndicesImag() const
  {
    return d_localProcDofIndicesImag;
  }

  //
  //Get dealii constraint matrix used for the eigen problem (2-component FE Object for Periodic, 1-component FE object for non-periodic)
  //
  const dealii::AffineConstraints<double> * operatorDFTCUDAClass::getConstraintMatrixEigen() const
  {
    return d_constraintMatrixEigen;
  }

  //
  //Get overloaded constraint matrix object constructed using 1-component FE object 
  //
  dftUtils::constraintMatrixInfoCUDA * operatorDFTCUDAClass::getOverloadedConstraintMatrix() const
  {
    return d_constraintMatrixData;
  }

  //
  //Get matrix free data
  //
  const dealii::MatrixFree<3,double> * operatorDFTCUDAClass::getMatrixFreeData() const
  {
    return d_matrix_free_data;
  }

  //
  //Get relevant mpi communicator
  //
  const MPI_Comm & operatorDFTCUDAClass::getMPICommunicator() const
  {
    return d_mpi_communicator;
  }


  void operatorDFTCUDAClass::processGridSetup(const unsigned int na,
                                                      const unsigned int nev)
  {


       std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  processGrid;
       linearAlgebraOperationsCUDA::internal::createProcessGridSquareMatrix(getMPICommunicator(),
                                               na,
                                               processGrid);


       d_scalapackBlockSize=std::min(dftParameters::scalapackBlockSize,
                             (na+processGrid->get_process_grid_rows()-1)
                             /processGrid->get_process_grid_rows());

  }

}
