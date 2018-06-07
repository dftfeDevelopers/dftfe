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

#ifndef linearAlgebraOperationsUtils_h
#define linearAlgebraOperationsUtils_h

#include <headers.h>
#include <dftParameters.h>

namespace dftfe
{

  namespace linearAlgebraOperations
  {
    /** @file linearAlgebraOperationsUtils.h
     *  @brief Contains small utils functions used in linearAlgebraOperations
     *
     *  @author Sambit Das
     */
    namespace utils
    {
#ifdef WITH_SCALAPACK
	void createProcessGridSquareMatrix(const MPI_Comm & mpi_communicator,
		                           const unsigned size,
		                           std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				           const unsigned int rowsBlockSize);
#endif
    }
  }
}
#endif
