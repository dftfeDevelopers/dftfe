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


#ifndef vectorUtilities_h
#define vectorUtilities_h

#include <headers.h>
#include <operator.h>



namespace dftfe{

/** @file vectorUtilities.h
 *  @brief Contains generic utils functions related to custom partitioned flattened dealii vector
 *
 *  @author Phani Motamarri
 */
  namespace vectorTools
  {

    /** @brief Creates a custom partitioned flattened dealii vector.
     *  stores multiple components asociated with a node sequentially.
     *
     *  @param partitioner associated with single component vector
     *  @param blockSize number of components associated with each node
     *
     *  @return flattenedArray custom partitioned dealii vector
     */
    template<typename T>
      void createDealiiVector(const std::shared_ptr< const dealii::Utilities::MPI::Partitioner > & partitioner,
			      const unsigned int                                           blockSize,
			      dealii::parallel::distributed::Vector<T>                   & flattenedArray);



    /** @brief Creates a cell local index set map for flattened array
     *
     *  @param partitioner associated with the flattened array
     *  @param matrix_free_data object pointer associated with the matrix free data structure
     *  @param blockSize number of components associated with each node
     *
     *  @return flattenedArrayMacroCellLocalProcIndexId macrocell's subcell local proc index map
     *  @return flattenedArrayCellLocalProcIndexId cell local proc index map
     */
    void computeCellLocalIndexSetMap(const std::shared_ptr< const dealii::Utilities::MPI::Partitioner > & partitioner,
				     const dealii::MatrixFree<3,double>                                 & matrix_free_data,
				     const unsigned int                                                   blockSize,
				     std::vector<std::vector<dealii::types::global_dof_index> >         & flattenedArrayMacroCellLocalProcIndexId,
				     std::vector<std::vector<dealii::types::global_dof_index> >         & flattenedArrayCellLocalProcIndexId);



  }
}
#endif
