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
// @author  Phani Motamarri (2018)
//
#include "../include/dealiiOverloadedFunc.h" 




namespace dftUtils{

  void convertConstraintMatrixToSTLVector(dealii::parallel::distributed::Vector<double> &fieldVector,
					  dealii::ConstraintMatrix & constraintMatrixData,
					  dealii::IndexSet         & locally_owned_dofs,
					  constraintMatrixInfo     & constraintMatrixDataInVector)
  {
    const std::shared_ptr< const dealii::Utilities::MPI::Partitioner > & partitioner = fieldVector.get_partitioner();
    constraintMatrixDataInVector.rowIdsGlobal.clear();
    constraintMatrixDataInVector.rowIdsLocal.clear();
    constraintMatrixDataInVector.columnIdsLocal.clear();
    constraintMatrixDataInVector.columnValues.clear();
    constraintMatrixDataInVector.inhomogenities.clear();
    constraintMatrixDataInVector.rowSizes.clear();

    for(dealii::IndexSet::ElementIterator it = locally_owned_dofs.begin(); it != locally_owned_dofs.end();++it)
      {
	if(constraintMatrixData.is_constrained(*it))
	  {
	    (constraintMatrixDataInVector.rowIdsGlobal).push_back(*it);
	    (constraintMatrixDataInVector.rowIdsLocal).push_back(partitioner->global_to_local(*it));
	  }
      }

    for(unsigned int i = 0; i < (constraintMatrixDataInVector.rowIdsGlobal).size(); ++i)
      {
	const int lineDof = constraintMatrixDataInVector.rowIdsGlobal[i];
	(constraintMatrixDataInVector.inhomogenities).push_back(constraintMatrixData.get_inhomogeneity(lineDof));
	const std::vector<std::pair<dealii::types::global_dof_index, double > > * rowData=constraintMatrixData.get_constraint_entries(lineDof);
	(constraintMatrixDataInVector.rowSizes).push_back(rowData->size());
	for(unsigned int j = 0; j < rowData->size();++j)
	  {
	    (constraintMatrixDataInVector.columnIdsLocal).push_back(partitioner->global_to_local((*rowData)[j].first));
	    (constraintMatrixDataInVector.columnValues).push_back((*rowData)[j].second);
	  }
      }

    return;

  }



  void distribute(constraintMatrixInfo &constraintMatrixDataInVector, 
		  dealii::parallel::distributed::Vector<double> &fieldVector)
  {
    unsigned int count = 0;
    //std::cout<<"Size Local: "<< (constraintMatrixDataInVector.rowIdsLocal).size()<<std::endl;
    //std::cout<<"Size Global: "<< (constraintMatrixDataInVector.rowIdsGlobal).size()<<std::endl;

    for(unsigned int i = 0; i < (constraintMatrixDataInVector.rowIdsLocal).size();++i)
      {
	double new_value = constraintMatrixDataInVector.inhomogenities[i];

	for(unsigned int j = 0; j < constraintMatrixDataInVector.rowSizes[i]; ++j)
	  {

	    new_value += fieldVector.local_element(constraintMatrixDataInVector.columnIdsLocal[count])*constraintMatrixDataInVector.columnValues[count];
	    count++;
	  }

	fieldVector.local_element(constraintMatrixDataInVector.rowIdsLocal[i]) = new_value;
	  
      }

    return;
  }

}



