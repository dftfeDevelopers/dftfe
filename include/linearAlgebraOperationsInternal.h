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

#ifndef linearAlgebraOperationsInternal_h
#define linearAlgebraOperationsInternal_h

#include <headers.h>

namespace dftfe
{

  namespace linearAlgebraOperations
  {
    /** @file linearAlgebraOperationsInternal.h
     *  @brief Contains small internal functions used in linearAlgebraOperations
     *
     *  @author Sambit Das
     */
    namespace internal
    {
#ifdef WITH_SCALAPACK
	void createProcessGridSquareMatrix(const MPI_Comm & mpi_communicator,
		                           const unsigned size,
		                           std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				           const unsigned int rowsBlockSize);


	template<typename T>
	void fillParallelOverlapMatrix(const dealii::parallel::distributed::Vector<T> & X,
		                       const unsigned int numberVectors,
		                       const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				       dealii::ScaLAPACKMatrix<T> & overlapMatPar);

	template<typename T>
	void subspaceRotation(dealii::parallel::distributed::Vector<T> & subspaceVectorsArray,
		              const unsigned int numberSubspaceVectors,
		              const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
			      const dealii::ScaLAPACKMatrix<T> & rotationMatPar);

#endif
    }
  }
}
#endif
