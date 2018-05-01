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
#include "../include/constraintMatrixInfo.h"

namespace dftfe {
//
//Declare dftUtils functions
//
namespace dftUtils
{

  //
  //constructor
  //
  constraintMatrixInfo::constraintMatrixInfo()
  {


  }

  //
  //destructor
  //
  constraintMatrixInfo::~constraintMatrixInfo()
  {


  }


  //
  //store constraintMatrix row data in STL vector
  //
  void constraintMatrixInfo::initialize(const std::shared_ptr< const dealii::Utilities::MPI::Partitioner > & partitioner,
					const dealii::ConstraintMatrix & constraintMatrixData)

  {

    clear();

    const dealii::IndexSet & locally_owned_dofs = partitioner->locally_owned_range();
    const dealii::IndexSet & ghost_dofs = partitioner->ghost_indices();

    for(dealii::IndexSet::ElementIterator it = locally_owned_dofs.begin(); it != locally_owned_dofs.end();++it)
      {
	if(constraintMatrixData.is_constrained(*it))
	  {
	    const dealii::types::global_dof_index lineDof = *it;
	    d_rowIdsLocal.push_back(partitioner->global_to_local(lineDof));
	    d_rowIdsGlobal.push_back(lineDof);
	    d_inhomogenities.push_back(constraintMatrixData.get_inhomogeneity(lineDof));
	    const std::vector<std::pair<dealii::types::global_dof_index, double > > * rowData=constraintMatrixData.get_constraint_entries(lineDof);
	    d_rowSizes.push_back(rowData->size());
	    for(unsigned int j = 0; j < rowData->size();++j)
	      {
		Assert((*rowData)[j].first<partitioner->size(),
	    	   dealii::ExcMessage("Index out of bounds")); 
		d_columnIdsGlobal.push_back((*rowData)[j].first);
		d_columnIdsLocal.push_back(partitioner->global_to_local((*rowData)[j].first));
		d_columnValues.push_back((*rowData)[j].second);
	      }
	  }
      }

    
    for(dealii::IndexSet::ElementIterator it = ghost_dofs.begin(); it != ghost_dofs.end();++it)
      {
	if(constraintMatrixData.is_constrained(*it))
	  {
	    const dealii::types::global_dof_index lineDof = *it;
	    d_rowIdsLocal.push_back(partitioner->global_to_local(lineDof));
	    d_rowIdsGlobal.push_back(lineDof);
	    d_inhomogenities.push_back(constraintMatrixData.get_inhomogeneity(lineDof));
	    const std::vector<std::pair<dealii::types::global_dof_index, double > > * rowData=constraintMatrixData.get_constraint_entries(lineDof);
	    d_rowSizes.push_back(rowData->size());
	    for(unsigned int j = 0; j < rowData->size();++j)
	      {
		Assert((*rowData)[j].first<partitioner->size(),
		       dealii::ExcMessage("Index out of bounds")); 
		d_columnIdsGlobal.push_back((*rowData)[j].first);
		d_columnIdsLocal.push_back(partitioner->global_to_local((*rowData)[j].first));
		d_columnValues.push_back((*rowData)[j].second);
	      }
	  }
      }
    


  }


  /* void constraintMatrixInfo::precomputeMaps(const std::shared_ptr< const dealii::Utilities::MPI::Partitioner> & unflattenedPartitioner,
					    const std::shared_ptr< const dealii::Utilities::MPI::Partitioner> & flattenedPartitioner)
  {

      //
      //Get required sizes
      //
      unsigned int n_ghosts   = unflattenedPartitioner->n_ghost_indices();
      unsigned int localSize  = flattenedPartitioner->local_size();

      //
      //fill the data array
      //
      


      } */
  



  //
  //set the constrained degrees of freedom to values so that constraints
  //are satisfied
  //
  void constraintMatrixInfo::distribute(dealii::parallel::distributed::Vector<double> &fieldVector) const
  {
    fieldVector.update_ghost_values();
    unsigned int count = 0;
    for(unsigned int i = 0; i < d_rowIdsLocal.size();++i)
      {
	double new_value = d_inhomogenities[i];
	for(unsigned int j = 0; j < d_rowSizes[i]; ++j)
	  {
	    new_value += fieldVector.local_element(d_columnIdsLocal[count])*d_columnValues[count];
	    count++;
	  }
	fieldVector.local_element(d_rowIdsLocal[i]) = new_value;
      }
  }


  template<typename T>
  void constraintMatrixInfo::distribute(dealii::parallel::distributed::Vector<T> &fieldVector,
					const unsigned int blockSize) const
  {
    fieldVector.update_ghost_values();


    unsigned int count = 0;
    for(unsigned int i = 0; i < d_rowIdsLocal.size(); ++i)
      {
	std::vector<T> newValuesBlock(blockSize,d_inhomogenities[i]);
	for(unsigned int j = 0; j < d_rowSizes[i]; ++j)
	  {

	    Assert(count<d_columnIdsGlobal.size(),
	    	   dealii::ExcMessage("Overloaded distribute for flattened array has indices out of bounds")); 

	    const dealii::types::global_dof_index startingLocalDofIndexColumn=fieldVector.get_partitioner()->global_to_local(d_columnIdsGlobal[count]*blockSize);

	    for(unsigned int k = 0; k < blockSize; ++k)
	      {
		newValuesBlock[k] += fieldVector.local_element(startingLocalDofIndexColumn + k)*d_columnValues[count];
	      }

	    count++;
	  }

	const dealii::types::global_dof_index startingLocalDofIndexRow=fieldVector.get_partitioner()->global_to_local(d_rowIdsGlobal[i]*blockSize);
	for(unsigned int k = 0; k < blockSize; ++k)
	  {
	    fieldVector.local_element(startingLocalDofIndexRow + k) = newValuesBlock[k];
	  }
      }
  }


  


  //
  //set the constrained degrees of freedom to values so that constraints
  //are satisfied for flattened array
  //
  template<typename T>
  void constraintMatrixInfo::distribute_slave_to_master(dealii::parallel::distributed::Vector<T> & fieldVector,
							const unsigned int blockSize) const
  {
    unsigned int count = 0;
    for(unsigned int i = 0; i < d_rowIdsLocal.size(); ++i)
      {
	const dealii::types::global_dof_index startingLocalDofIndexRow=fieldVector.get_partitioner()->global_to_local(d_rowIdsGlobal[i]*blockSize);
	for(unsigned int j = 0; j < d_rowSizes[i]; ++j)
	  {
	    const dealii::types::global_dof_index startingLocalDofIndexColumn=fieldVector.get_partitioner()->global_to_local(d_columnIdsGlobal[count]*blockSize);
	    for(unsigned int k = 0; k < blockSize; ++k)
	      {

		fieldVector.local_element(startingLocalDofIndexColumn + k) += d_columnValues[count]*fieldVector.local_element(startingLocalDofIndexRow + k);

	      }
	    count++;
	  }
      }
  }
							


  //
  //
  //clear the data variables
  //
  void constraintMatrixInfo::clear()
  {
    d_rowIdsGlobal.clear();
    d_rowIdsLocal.clear();
    d_columnIdsLocal.clear();
    d_columnIdsGlobal.clear();
    d_columnValues.clear();
    d_inhomogenities.clear();
    d_rowSizes.clear();
  }


#ifdef ENABLE_PERIODIC_BC
  template void constraintMatrixInfo::distribute(dealii::parallel::distributed::Vector<std::complex<double> > & fieldVector,
						 const unsigned int blockSize) const;

  template void constraintMatrixInfo::distribute_slave_to_master(dealii::parallel::distributed::Vector<std::complex<double> > & fieldVector,
						 const unsigned int blockSize) const;

#else
  template void constraintMatrixInfo::distribute(dealii::parallel::distributed::Vector<double> & fieldVector,
						 const unsigned int blockSize) const;

  template void constraintMatrixInfo::distribute_slave_to_master(dealii::parallel::distributed::Vector<double> & fieldVector,
						 const unsigned int blockSize) const;
#endif


}

}

