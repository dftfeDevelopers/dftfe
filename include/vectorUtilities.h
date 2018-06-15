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
 *  @author Phani Motamarri, Sambit Das
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


#ifdef USE_COMPLEX
    /** @brief Copies a single field component from a flattenedArray parallel distributed
     * vector containing multiple component fields to a 2-component field (real and complex)
     * parallel distributed vector.
     *
     *  @param[in] flattenedArray flattened parallel distributed vector with multiple component fields
     *  @param[in] totalNumberComponents total number of component fiels in flattenedArray
     *  @param[in] componentIndex desired single field component
     *  @param[in] localProcDofIndicesReal local dof indices in the current processor
     *  which correspond to component-1 of 2-component parallel distributed array
     *  @param[in] localProcDofIndicesImag local dof indices in the current processor
     *  which correspond to component-2 of 2-component parallel distributed array
     *  @param[out] componentVector two component field parallel distributed vector with
     *  the values corresponding to componentIndex field of flattenedArray.
     *  componentVector is assumed to be already initialized with the 2-component version of the
     *  same single component partitioner used in the creation of the flattenedArray partitioner.
     */
     void copyFlattenedDealiiVecToSingleCompVec
                             (const dealii::parallel::distributed::Vector<std::complex<double>>  & flattenedArray,
			      const unsigned int                        totalNumberComponents,
			      const unsigned int                        componentIndex,
			      const std::vector<dealii::types::global_dof_index> & localProcDofIndicesReal,
                              const std::vector<dealii::types::global_dof_index> & localProcDofIndicesImag,
			      dealii::parallel::distributed::Vector<double>  & componentVector);

#else
    /** @brief Copies a single field component from a flattenedArray parallel distributed
     * vector containing multiple component fields to a single field parallel distributed vector.
     *
     *  @param[in] flattenedArray flattened parallel distributed vector with multiple component fields
     *  @param[in] totalNumberComponents total number of component fiels in flattenedArray
     *  @param[in] componentIndex desired single field component
     *  @param[out] componentVector parallel distributed vector with field corresponding to
     *  componentIndex. componentVector is assumed to be already initialized with the
     *  same single component partitioner used in the creation of the flattenedArray partitioner.
     */
     void copyFlattenedDealiiVecToSingleCompVec
                             (const dealii::parallel::distributed::Vector<double>  & flattenedArray,
			      const unsigned int                        totalNumberComponents,
			      const unsigned int                        componentIndex,
			      dealii::parallel::distributed::Vector<double>  & componentVector);

#endif

  }
}
#endif
