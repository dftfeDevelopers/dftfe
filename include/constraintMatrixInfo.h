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
//
// @author Phani Motamarri (2018)
//

#ifndef constraintMatrixInfo_H_
#define constraintMatrixInfo_H_

#include <vector>

#include "headers.h"

namespace dftfe {
//
//Declare dftUtils functions
//
namespace dftUtils
{
  //
  //method which overloads dealii's constraints.distribute. Stores the constraintMatrix data
  //into STL vectors and then sets all constrained degrees of freedom to values so that constraints
  //are satisfied using these STL vectors
  //

  class constraintMatrixInfo
  {

  public:

    /**
     * class constructor
     */
    constraintMatrixInfo();

    /**
     * class destructor
     */
    ~constraintMatrixInfo();

    /**
     * convert a given constraintMatrix to simple arrays (STL) for fast access
     */
    void initialize(const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> & partitioner,
		    const dealii::ConstraintMatrix & constraintMatrixData);

    /**
     * precompute certain maps from unflattened array to flattened array
     */
    void precomputeMaps(const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> & partitioner1,
			const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> & partitioner2,
			const unsigned int blockSize);

    /**
     * overload dealii internal function distribute
     */
    void distribute(dealii::parallel::distributed::Vector<double> &fieldVector) const;

    /**
     * overload dealii internal function distribute for flattened array
     */
    template<typename T>
    void distribute(dealii::parallel::distributed::Vector<T> &fieldVector,
		    const unsigned int blockSize) const;


    /**
     * distribute the contributions from slave to master nodes
     */
    template<typename T>
    void distribute_slave_to_master(dealii::parallel::distributed::Vector<T> &fieldVector,
				    const unsigned int blockSize) const;


    
    

    /**
     * clear data members
     */
    void clear();


  private:

    std::vector<dealii::types::global_dof_index> d_rowIdsGlobal;
    std::vector<dealii::types::global_dof_index> d_rowIdsLocal;
    std::vector<dealii::types::global_dof_index> d_columnIdsLocal;
    std::vector<dealii::types::global_dof_index> d_columnIdsGlobal;
    std::vector<double> d_columnValues;
    std::vector<double> d_inhomogenities;
    std::vector<dealii::types::global_dof_index> d_rowSizes;
    std::vector<dealii::types::global_dof_index> d_localIndexMapUnflattenedToFlattened;


  };

};

}
#endif
