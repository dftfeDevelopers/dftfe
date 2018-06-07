// ---------------------------------------------------------------------
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
// ---------------------------------------------------------------------
//
// @author Sambit Das
//

#include <linearAlgebraOperationsUtils.h>

/** @file linearAlgebraOperationsUtils.cc
 *  @brief Contains small utils functions used in linearAlgebraOperations
 *
 */
namespace dftfe
{

  namespace linearAlgebraOperations
  {

    namespace utils
    {
#ifdef WITH_SCALAPACK
	void createProcessGridSquareMatrix(const MPI_Comm & mpi_communicator,
		                      const unsigned size,
		                      std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				      const unsigned int rowsBlockSize)
	{
	      const unsigned int numberProcs = dealii::Utilities::MPI::n_mpi_processes(mpi_communicator);
	      const unsigned int blocksPerProc=4;
	      const unsigned int rowProcs=std::min(std::floor(std::sqrt(numberProcs)),
				std::ceil((double)size/(double)(blocksPerProc*rowsBlockSize)));
	      if(dftParameters::verbosity>=2)
	      {
		 dealii::ConditionalOStream   pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));
		 pcout<<"Scalapack Matrix created, "<<"rowsBlockSize: "<<rowsBlockSize<<", blocksPerProc: "<<blocksPerProc<<", row procs: "<< rowProcs<<std::endl;
	      }

	      processGrid=std::make_shared<const dealii::Utilities::MPI::ProcessGrid>(mpi_communicator,
										      rowProcs,
										      rowProcs);
	}
#endif
    }
  }
}
