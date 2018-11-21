
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
// @author Phani Motamarri
//
#include <operator.h>
#include <linearAlgebraOperationsInternal.h>
#include <dftParameters.h>

//
// Constructor.
//
namespace dftfe {

  operatorDFTClass::operatorDFTClass(const MPI_Comm                                        & mpi_comm_replica,
				     const dealii::MatrixFree<3,double>                    & matrix_free_data,
				     const std::vector<dealii::types::global_dof_index>    & localDofIndicesReal,
				     const std::vector<dealii::types::global_dof_index>    & localDofIndicesImag,
				     const std::vector<dealii::types::global_dof_index>    & localProcDofIndicesReal,
				     const std::vector<dealii::types::global_dof_index>    & localProcDofIndicesImag,
				     const dealii::ConstraintMatrix                        & constraintMatrixEigen,
				     dftUtils::constraintMatrixInfo                        & constraintMatrixNone):
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
  //Get dealii constraint matrix used for the eigen problem (2-component FE Object for Periodic, 1-component FE object for non-periodic)
  //
  const dealii::ConstraintMatrix * operatorDFTClass::getConstraintMatrixEigen() const
  {
    return d_constraintMatrixEigen;
  }

  //
  //Get overloaded constraint matrix object constructed using 1-component FE object
  //
  dftUtils::constraintMatrixInfo * operatorDFTClass::getOverloadedConstraintMatrix() const
  {
    return d_constraintMatrixData;
  }

  //
  //Get matrix free data
  //
  const dealii::MatrixFree<3,double> * operatorDFTClass::getMatrixFreeData() const
  {
    return d_matrix_free_data;
  }

  //
  //Get relevant mpi communicator
  //
  const MPI_Comm & operatorDFTClass::getMPICommunicator() const
  {
    return d_mpi_communicator;
  }

#ifdef DEAL_II_WITH_SCALAPACK

  void operatorDFTClass::processGridOptionalELPASetup(const unsigned int na,
    		                                      const unsigned int nev)
  {


       std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  processGrid;
       linearAlgebraOperations::internal::createProcessGridSquareMatrix(getMPICommunicator(),
                                               na,
                                               processGrid);


       d_scalapackBlockSize=std::min(dftParameters::scalapackBlockSize,
	                     (na+processGrid->get_process_grid_rows()-1)
                             /processGrid->get_process_grid_rows());

       //std::cout<<"nblk: "<<d_scalapackBlockSize<<std::endl;

#ifdef DFTFE_WITH_ELPA
       if (dftParameters::useELPA)
       {
	   int error;

	   if (elpa_init(20180525) != ELPA_OK) {
	     fprintf(stderr, "Error: ELPA API version not supported");
	     exit(1);
	   }

	   d_elpaHandle = elpa_allocate(&error);
	   AssertThrow(error== ELPA_OK,
		    dealii::ExcMessage("DFT-FE Error: ELPA Error."));

	   // Get the group of processes in mpi_communicator
	   int       ierr = 0;
	   MPI_Group all_group;
	   ierr = MPI_Comm_group(getMPICommunicator(), &all_group);
	   AssertThrowMPI(ierr);

	   // Construct the group containing all ranks we need:
	   const unsigned int n_active_mpi_processes =
	   processGrid->get_process_grid_rows() * processGrid->get_process_grid_columns();
	   std::vector<int> active_ranks;
	   for (unsigned int i = 0; i < n_active_mpi_processes; ++i)
	      active_ranks.push_back(i);

	   MPI_Group active_group;
	   const int n = active_ranks.size();
	   ierr        = MPI_Group_incl(all_group,
				n,
				active_ranks.data(),
				&active_group);
	   AssertThrowMPI(ierr);

	   // Create the communicator based on active_group.
	   // Note that on all the inactive processs the resulting MPI_Comm d_processGridCommunicatorActive
	   // will be MPI_COMM_NULL.
	   ierr = dealii::Utilities::MPI::create_group(getMPICommunicator(),
					      active_group,
					      50,
					      &d_processGridCommunicatorActive);
	   AssertThrowMPI(ierr);

	   ierr = MPI_Group_free(&all_group);
	   AssertThrowMPI(ierr);
	   ierr = MPI_Group_free(&active_group);
	   AssertThrowMPI(ierr);


	   dealii::ScaLAPACKMatrix<double> tempMat(na,
						   processGrid,
						   d_scalapackBlockSize);
	   if (processGrid->is_process_active())
	   {

	       /* Set parameters the matrix and it's MPI distribution */
	       elpa_set_integer(d_elpaHandle, "na", na, &error);
	       AssertThrow(error== ELPA_OK,
			dealii::ExcMessage("DFT-FE Error: ELPA Error."));


	       elpa_set_integer(d_elpaHandle, "nev", nev, &error);
	       AssertThrow(error== ELPA_OK,
			dealii::ExcMessage("DFT-FE Error: ELPA Error."));

	       elpa_set_integer(d_elpaHandle, "nblk", d_scalapackBlockSize, &error);
	       AssertThrow(error== ELPA_OK,
			dealii::ExcMessage("DFT-FE Error: ELPA Error."));

	       elpa_set_integer(d_elpaHandle, "mpi_comm_parent", MPI_Comm_c2f(d_processGridCommunicatorActive), &error);
	       AssertThrow(error== ELPA_OK,
			dealii::ExcMessage("DFT-FE Error: ELPA Error."));


	       //std::cout<<"local_nrows: "<<tempMat.local_m() <<std::endl;
	       //std::cout<<"local_ncols: "<<tempMat.local_n() <<std::endl;
	       //std::cout<<"process_row: "<<processGrid->get_this_process_row()<<std::endl;
	       //std::cout<<"process_col: "<<processGrid->get_this_process_column()<<std::endl;

	       elpa_set_integer(d_elpaHandle, "local_nrows", tempMat.local_m(), &error);
	       AssertThrow(error== ELPA_OK,
			dealii::ExcMessage("DFT-FE Error: ELPA Error."));

	       elpa_set_integer(d_elpaHandle, "local_ncols", tempMat.local_n(), &error);
	       AssertThrow(error== ELPA_OK,
			dealii::ExcMessage("DFT-FE Error: ELPA Error."));

	       elpa_set_integer(d_elpaHandle, "process_row", processGrid->get_this_process_row(), &error);
	       AssertThrow(error== ELPA_OK,
			dealii::ExcMessage("DFT-FE Error: ELPA Error."));

	       elpa_set_integer(d_elpaHandle, "process_col", processGrid->get_this_process_column(), &error);
	       AssertThrow(error== ELPA_OK,
			dealii::ExcMessage("DFT-FE Error: ELPA Error."));


	       /* Setup */
	       AssertThrow(elpa_setup(d_elpaHandle)==ELPA_OK,
			dealii::ExcMessage("DFT-FE Error: ELPA Error."));

	       elpa_set_integer(d_elpaHandle, "solver", ELPA_SOLVER_2STAGE, &error);
	       AssertThrow(error==ELPA_OK,
			dealii::ExcMessage("DFT-FE Error: ELPA Error."));

#ifdef DEBUG
	       elpa_set_integer(d_elpaHandle, "debug", 1, &error);
	       AssertThrow(error==ELPA_OK,
			dealii::ExcMessage("DFT-FE Error: ELPA Error."));
#endif
	  }

	   //d_elpaAutoTuneHandle = elpa_autotune_setup(d_elpaHandle, ELPA_AUTOTUNE_FAST, ELPA_AUTOTUNE_DOMAIN_REAL, &error);   // create autotune object
      }
#endif
  }

#ifdef DFTFE_WITH_ELPA
  void operatorDFTClass::elpaUninit()
  {
       //elpa_autotune_deallocate(d_elpaAutoTuneHandle);
       elpa_deallocate(d_elpaHandle);
       elpa_uninit();
  }
#endif
#endif

}
