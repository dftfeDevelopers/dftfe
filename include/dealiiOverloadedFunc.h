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
  void convertConstraintMatrixToSTLVector(dealii::parallel::distributed::Vector<double> &fieldVector,
					  dealii::ConstraintMatrix & constraintMatrixData,
					  dealii::IndexSet         & locally_owned_dofs,
					  constraintMatrixInfo     & constraintMatrixDataInVector);


  /**
   *  overload dealii internal function distribute  
   */
  void distribute(constraintMatrixInfo &constraintMatrixDataInVector, 
		  dealii::parallel::distributed::Vector<double> &fieldVector);
 

};

#endif
