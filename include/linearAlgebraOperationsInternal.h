// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018  The Regents of the University of Michigan and DFT-FE authors.
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
    /**
     *  @brief Contains internal functions used in linearAlgebraOperations
     *
     *  @author Sambit Das
     */
    namespace internal
    {
#ifdef DEAL_II_WITH_SCALAPACK
	/** @brief Wrapper function to create a two dimensional processor grid for a square matrix in
	 * dealii::ScaLAPACKMatrix storage format.
	 *
	 */
	void createProcessGridSquareMatrix(const MPI_Comm & mpi_communicator,
		                           const unsigned size,
					   std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid);


	/** @brief Creates global row/column id to local row/column ids for dealii::ScaLAPACKMatrix
	 *
	 */
        template<typename T>
	void createGlobalToLocalIdMapsScaLAPACKMat(const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
		                                   const dealii::ScaLAPACKMatrix<T> & mat,
				                   std::map<unsigned int, unsigned int> & globalToLocalRowIdMap,
					           std::map<unsigned int, unsigned int> & globalToLocalColumnIdMap);


	/** @brief Mpi all reduce of ScaLAPACKMat across a given inter communicator.
	 * Used for band parallelization.
	 *
	 */
        template<typename T>
	void sumAcrossInterCommScaLAPACKMat(const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
		                            dealii::ScaLAPACKMatrix<T> & mat,
				            const MPI_Comm &interComm);

	/** @brief MPI_Bcast of ScaLAPACKMat across a given inter communicator from a given broadcast root.
	 * Used for band parallelization.
	 *
	 */
        template<typename T>
	void broadcastAcrossInterCommScaLAPACKMat
	                                   (const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
		                            dealii::ScaLAPACKMatrix<T> & mat,
				            const MPI_Comm &interComm,
					    const unsigned int broadcastRoot);

	/** @brief Computes Sc=X^{T}*Xc and stores in a parallel ScaLAPACK matrix.
	 * X^{T} is the subspaceVectorsArray stored in the column major format (N x M).
	 * Sc is the overlapMatPar.
	 *
	 * The overlap matrix computation and filling is done in a blocked approach
	 * which avoids creation of full serial overlap matrix memory, and also avoids creation
	 * of another full X memory.
	 *
	 */
	template<typename T>
	void fillParallelOverlapMatrix(const T* X,
		                       const unsigned int XLocalSize,
		                       const unsigned int numberVectors,
		                       const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
				       const MPI_Comm &interBandGroupComm,
				       const MPI_Comm &mpiComm,
				       dealii::ScaLAPACKMatrix<T> & overlapMatPar);

	/** @brief Computes X^{T}=Q*X^{T} inplace. X^{T} is the subspaceVectorsArray
	 * stored in the column major format (N x M). Q is rotationMatPar.
	 *
	 * The subspace rotation inside this function is done in a blocked approach
	 * which avoids creation of full serial rotation matrix memory, and also avoids creation
	 * of another full subspaceVectorsArray memory.
	 *
	 */
	template<typename T>
	void subspaceRotation(T* subspaceVectorsArray,
		              const unsigned int subspaceVectorsArrayLocalSize,
		              const unsigned int numberSubspaceVectors,
			      const unsigned int numberCoreVectors,
			      T* nonCoreVectorsArray,
		              const std::shared_ptr< const dealii::Utilities::MPI::ProcessGrid>  & processGrid,
			      const MPI_Comm &interBandGroupComm,
			      const MPI_Comm &mpiComm,
			      const dealii::ScaLAPACKMatrix<T> & rotationMatPar,
			      const bool rotationMatTranspose=false,
			      const bool isRotationMatLowerTria=false);
#endif
    }
  }
}
#endif
