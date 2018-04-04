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
     * overload dealii internal function distribute
     */
    void distribute(dealii::parallel::distributed::Vector<double> &fieldVector) const;

    /**
     * clear data members
     */
    void clear();


  private:

    std::vector<unsigned int> d_rowIdsGlobal;
    std::vector<unsigned int> d_rowIdsLocal;
    std::vector<unsigned int> d_columnIdsLocal;
    std::vector<double> d_columnValues;
    std::vector<double> d_inhomogenities;
    std::vector<unsigned int> d_rowSizes;


  };

};

}
#endif
