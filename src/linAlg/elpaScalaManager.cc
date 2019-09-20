
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
// @author Sambit Das
//
#include <elpaScalaManager.h>
#include <linearAlgebraOperationsInternal.h>
#include <dftParameters.h>

//
// Constructor.
//
namespace dftfe {

  elpaScalaManager::elpaScalaManager(const MPI_Comm                                        & mpi_comm_replica):
    d_mpi_communicator(mpi_comm_replica)
#ifdef DFTFE_WITH_ELPA	
    ,d_processGridCommunicatorActive(MPI_COMM_NULL),
    d_processGridCommunicatorActivePartial(MPI_COMM_NULL),
    d_processGridCommunicatorActiveValence(MPI_COMM_NULL)
#endif	
  {


  }


  //
  // Destructor.
  //
  elpaScalaManager::~elpaScalaManager()
  {
#ifdef DFTFE_WITH_ELPA	  
      if (d_processGridCommunicatorActive != MPI_COMM_NULL)
	  MPI_Comm_free(&d_processGridCommunicatorActive);

      if (d_processGridCommunicatorActivePartial != MPI_COMM_NULL)
	  MPI_Comm_free(&d_processGridCommunicatorActivePartial);

      if (d_processGridCommunicatorActiveValence != MPI_COMM_NULL)
	 MPI_Comm_free(&d_processGridCommunicatorActiveValence);
#endif      
    //
    //
    //
    return;

  }
  //
  //Get relevant mpi communicator
  //
  const MPI_Comm & elpaScalaManager::getMPICommunicator() const
  {
    return d_mpi_communicator;
  }


  void elpaScalaManager::processGridOptionalELPASetup(const unsigned int na,
    		                                      const unsigned int nev)
  {


       std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  processGrid;
       linearAlgebraOperations::internal::createProcessGridSquareMatrix(getMPICommunicator(),
                                               na,
                                               processGrid);


       d_scalapackBlockSize=std::min(dftParameters::scalapackBlockSize,
	                     (na+processGrid->get_process_grid_rows()-1)
                             /processGrid->get_process_grid_rows());
#ifdef DFTFE_WITH_ELPA
       if (dftParameters::useELPA)
           linearAlgebraOperations::internal::setupELPAHandle(getMPICommunicator(),
                                                              d_processGridCommunicatorActive,
                                                              processGrid,
							      na,
							      na,
							      d_scalapackBlockSize,
							      d_elpaHandle);
#endif

       if (nev!=na)
       {
#ifdef DFTFE_WITH_ELPA
	   if (dftParameters::useELPA)
	       linearAlgebraOperations::internal::setupELPAHandle(getMPICommunicator(),
                                                                  d_processGridCommunicatorActivePartial,
								  processGrid,
								  na,
								  nev,
								  d_scalapackBlockSize,
								  d_elpaHandlePartialEigenVec);
#endif

	   std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  processGridValence;
	   linearAlgebraOperations::internal::createProcessGridSquareMatrix(getMPICommunicator(),
						   nev,
						   processGridValence,
						   true);


	   d_scalapackBlockSizeValence=std::min(dftParameters::scalapackBlockSize,
				 (nev+processGridValence->get_process_grid_rows()-1)
				 /processGridValence->get_process_grid_rows());

#ifdef DFTFE_WITH_ELPA
	   if (dftParameters::useELPA)
	       linearAlgebraOperations::internal::setupELPAHandle(getMPICommunicator(),
                                                                  d_processGridCommunicatorActiveValence,
								  processGridValence,
								  nev,
								  nev,
								  d_scalapackBlockSizeValence,
								  d_elpaHandleValence);
#endif
	   //std::cout<<"nblkvalence: "<<d_scalapackBlockSizeValence<<std::endl;
       }

       //std::cout<<"nblk: "<<d_scalapackBlockSize<<std::endl;

  }

#ifdef DFTFE_WITH_ELPA
  void elpaScalaManager::elpaDeallocateHandles(const unsigned int na,
		                    const unsigned int nev)
  {
       //elpa_autotune_deallocate(d_elpaAutoTuneHandle);
       
       int error;
       elpa_deallocate(d_elpaHandle,&error);
       AssertThrow(error == ELPA_OK,
                dealii::ExcMessage("DFT-FE Error: elpa error."));

       if (na!=nev)
       {
          
	  elpa_deallocate(d_elpaHandlePartialEigenVec,&error);
          AssertThrow(error == ELPA_OK,
                dealii::ExcMessage("DFT-FE Error: elpa error."));

	  elpa_deallocate(d_elpaHandleValence,&error);
          AssertThrow(error == ELPA_OK,
                dealii::ExcMessage("DFT-FE Error: elpa error."));
       }

  }
#endif

}
