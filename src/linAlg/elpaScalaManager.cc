
//
// -------------------------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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
// @author Sambit Das
//
#include <dftParameters.h>
#include <elpaScalaManager.h>
#include <linearAlgebraOperationsInternal.h>

//
// Constructor.
//
namespace dftfe
{
  elpaScalaManager::elpaScalaManager(const MPI_Comm &mpi_comm_replica)
    : d_mpi_communicator(mpi_comm_replica)
    , d_processGridCommunicatorActive(MPI_COMM_NULL)
    , d_processGridCommunicatorActivePartial(MPI_COMM_NULL)
  {}


  //
  // Destructor.
  //
  elpaScalaManager::~elpaScalaManager()
  {
    if (d_processGridCommunicatorActive != MPI_COMM_NULL)
      MPI_Comm_free(&d_processGridCommunicatorActive);

    if (d_processGridCommunicatorActivePartial != MPI_COMM_NULL)
      MPI_Comm_free(&d_processGridCommunicatorActivePartial);
    //
    //
    //
    return;
  }
  //
  // Get relevant mpi communicator
  //
  const MPI_Comm &
  elpaScalaManager::getMPICommunicator() const
  {
    return d_mpi_communicator;
  }


  void
  elpaScalaManager::processGridELPASetup(const unsigned int na,
                                                 const unsigned int nev,
						 const MPI_Comm &mpi_comm_interband,
						 const MPI_Comm &mpi_comm_interpool)
  {
    linearAlgebraOperations::internal::createProcessGridSquareMatrix(
      getMPICommunicator(), na, d_processGridDftfeWrapper);


    d_scalapackBlockSize =
      std::min(dftParameters::scalapackBlockSize,
               (na + d_processGridDftfeWrapper->get_process_grid_rows() - 1) /
                 d_processGridDftfeWrapper->get_process_grid_rows());
    if (dftParameters::useELPA)
      linearAlgebraOperations::internal::setupELPAHandle(
        getMPICommunicator(),
	mpi_comm_interband,
	mpi_comm_interpool,
        d_processGridCommunicatorActive,
        d_processGridDftfeWrapper,
        na,
        na,
        d_scalapackBlockSize,
        d_elpaHandle);

    if (nev != na)
      {
        if (dftParameters::useELPA)
          linearAlgebraOperations::internal::setupELPAHandle(
            getMPICommunicator(),
            mpi_comm_interband,
            mpi_comm_interpool,	    
            d_processGridCommunicatorActivePartial,
            d_processGridDftfeWrapper,
            na,
            nev,
            d_scalapackBlockSize,
            d_elpaHandlePartialEigenVec);
      }

    // std::cout<<"nblk: "<<d_scalapackBlockSize<<std::endl;
  }

  void
  elpaScalaManager::elpaDeallocateHandles(const unsigned int na,
                                          const unsigned int nev)
  {
    // elpa_autotune_deallocate(d_elpaAutoTuneHandle);

    int error;
    elpa_deallocate(d_elpaHandle, &error);
    AssertThrow(error == ELPA_OK,
                dealii::ExcMessage("DFT-FE Error: elpa error."));

    if (na != nev)
      {
        elpa_deallocate(d_elpaHandlePartialEigenVec, &error);
        AssertThrow(error == ELPA_OK,
                    dealii::ExcMessage("DFT-FE Error: elpa error."));
      }
  }
} // namespace dftfe
