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

#ifndef dealiiOverloadedFunc_H_
#define dealiiOverloadedFunc_H_

#include <vector>

#include "headers.h"

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


  struct constraintMatrixInfo
  {
    std::vector<unsigned int> rowIdsGlobal;
    std::vector<unsigned int> rowIdsLocal;
    std::vector<unsigned int> columnIdsLocal;
    std::vector<double> columnValues;
    std::vector<double> inhomogenities;
    std::vector<unsigned int> rowSizes;
  };

  /**
   *  convert a given constraintMatrix to simple arrays (STL) for fast access 
   */
  void convertConstraintMatrixToSTLVector(const dealii::parallel::distributed::Vector<double> &fieldVector,
					  const dealii::ConstraintMatrix & constraintMatrixData,
					  const dealii::IndexSet         & locally_owned_dofs,
					  constraintMatrixInfo     & constraintMatrixDataInVector);


  /**
   *  overload dealii internal function distribute  
   */
  void distribute(const constraintMatrixInfo &constraintMatrixDataInVector, 
		  dealii::parallel::distributed::Vector<double> &fieldVector);

  /**
   * 
   */
  void clearData(constraintMatrixInfo & constraintMatrixDataInVector);
 

};

#endif
