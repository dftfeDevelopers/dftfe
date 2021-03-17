// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018  The Regents of the University of Michigan and DFT-FE
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
// ---------------------------------------------------------------------

#ifndef linearAlgebraOperationsInternalCUDA_h
#define linearAlgebraOperationsInternalCUDA_h

#include <headers.h>
namespace dftfe
{
  namespace linearAlgebraOperationsCUDA
  {
    /**
     *  @brief Contains internal functions used in linearAlgebraOperations
     *
     *  @author Sambit Das
     */
    namespace internal
    {
      /** @brief Wrapper function to create a two dimensional processor grid for a square matrix in
       * dealii::ScaLAPACKMatrix storage format.
       *
       */
      void
      createProcessGridSquareMatrix(
        const MPI_Comm &mpi_communicator,
        const unsigned  size,
        std::shared_ptr<const dealii::Utilities::MPI::ProcessGrid> &processGrid,
        const bool useOnlyThumbRule = false);

      /** @brief Wrapper function to create a two dimensional processor grid for a rectangular matrix in
       * dealii::ScaLAPACKMatrix storage format.
       *
       */
      void
      createProcessGridRectangularMatrix(
        const MPI_Comm &mpi_communicator,
        const unsigned  sizeRows,
        const unsigned  sizeColumns,
        std::shared_ptr<const dealii::Utilities::MPI::ProcessGrid>
          &processGrid);


      /** @brief Creates global row/column id to local row/column ids for dealii::ScaLAPACKMatrix
       *
       */
      template <typename T>
      void
      createGlobalToLocalIdMapsScaLAPACKMat(
        const std::shared_ptr<const dealii::Utilities::MPI::ProcessGrid>
          &                                   processGrid,
        const dealii::ScaLAPACKMatrix<T> &    mat,
        std::map<unsigned int, unsigned int> &globalToLocalRowIdMap,
        std::map<unsigned int, unsigned int> &globalToLocalColumnIdMap);


      /** @brief Mpi all reduce of ScaLAPACKMat across a given inter communicator.
       * Used for band parallelization.
       *
       */
      template <typename T>
      void
      sumAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dealii::Utilities::MPI::ProcessGrid>
          &                         processGrid,
        dealii::ScaLAPACKMatrix<T> &mat,
        const MPI_Comm &            interComm);



      /** @brief MPI_Bcast of ScaLAPACKMat across a given inter communicator from a given broadcast root.
       * Used for band parallelization.
       *
       */
      template <typename T>
      void
      broadcastAcrossInterCommScaLAPACKMat(
        const std::shared_ptr<const dealii::Utilities::MPI::ProcessGrid>
          &                         processGrid,
        dealii::ScaLAPACKMatrix<T> &mat,
        const MPI_Comm &            interComm,
        const unsigned int          broadcastRoot);
    } // namespace internal
  }   // namespace linearAlgebraOperationsCUDA
} // namespace dftfe
#endif
