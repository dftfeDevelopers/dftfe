// ---------------------------------------------------------------------
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
// ---------------------------------------------------------------------

/** @file linearAlgebraOperations.h
 *  @brief Contains linear algebra functions used in the implementation of an eigen solver
 *
 *  @author Phani Motamarri (2018)
 */

#include <headers.h>
#include <operator.h>


typedef dealii::parallel::distributed::Vector<double> vectorType;

namespace dftfe{
  namespace vectorTools
  {
  
    /** @brief Creates a custom partitioned flattened dealii vector.
     *  stores multiple components asociated with a node sequentially.
     *
     *  @param partitioner associated with single component vector
     *  @param mpi_communicator communicator to be used for the new parallel vector  
     *  @param globalNumberDegreesOfFreedom total number of nodes in mesh
     *  @param blockSize number of components associated with each node
     *  @return flattenedArray custom partitioned dealii vector
     */
    template<typename T>
      void createDealiiVector(const std::shared_ptr< const dealii::Utilities::MPI::Partitioner > & partitioner,
			      const MPI_Comm                                             & mpi_communicator,
			      const dealii::types::global_dof_index                      & globalNumberDegreesOfFreedom,
			      const unsigned int                                           blockSize,
			      dealii::parallel::distributed::Vector<T>                   & flattenedArray);



    /** @brief Creates a cell local index set map for flattened array
     *
     *  @param partitioner associated with the flattened array
     *  @param matrix_free_data object pointer associated with the matrix free data structure
     *  @param blockSize number of components associated with each node
     *  @return flattenedArrayMacroCellLocalProcIndexId macrocell's subcell local proc index map
     *  @return flattenedArrayCellLocalProcIndexId cell local proc index map
     */
    void computeCellLocalIndexSetMap(const std::shared_ptr< const dealii::Utilities::MPI::Partitioner > & partitioner,
				     const dealii::MatrixFree<3,double>                                 * matrix_free_data,
				     const unsigned int                                                   blockSize,
				     std::vector<std::vector<dealii::types::global_dof_index> >         & flattenedArrayMacroCellLocalProcIndexId,
				     std::vector<std::vector<dealii::types::global_dof_index> >         & flattenedArrayCellLocalProcIndexId);



  }
}
