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
//
// @author Phani Motamarri, Sambit Das
//

#include <vectorUtilities.h>
#include <exception>

namespace dftfe
{

  namespace vectorTools
  {

    template<typename T>
    void createDealiiVector(const std::shared_ptr< const dealii::Utilities::MPI::Partitioner > & partitioner,
			    const unsigned int                                                   blockSize,
			    dealii::parallel::distributed::Vector<T>                           & flattenedArray)
    {

      const MPI_Comm & mpi_communicator=partitioner->get_communicator();
      //
      //Get required sizes
      //
      const unsigned int n_ghosts   = partitioner->n_ghost_indices();
      const unsigned int localSize  = partitioner->local_size();
      const unsigned int totalSize  = localSize + n_ghosts;
      const  dealii::types::global_dof_index globalNumberDegreesOfFreedom=partitioner->size();

      //
      //create data for new parallel layout
      //
      dealii::IndexSet locallyOwnedFlattenedNodesSet, ghostFlattenedNodesSet;
      locallyOwnedFlattenedNodesSet.clear();ghostFlattenedNodesSet.clear();

      //
      //Set the maximal size of the indices upon which this object operates.
      //
      locallyOwnedFlattenedNodesSet.set_size(globalNumberDegreesOfFreedom*blockSize);
      ghostFlattenedNodesSet.set_size(globalNumberDegreesOfFreedom*blockSize);


      for(unsigned int ilocaldof = 0; ilocaldof < totalSize; ++ilocaldof)
	{
          std::vector<dealii::types::global_dof_index> newLocallyOwnedGlobalNodeIds;
          std::vector<dealii::types::global_dof_index> newGhostGlobalNodeIds;
	  const dealii::types::global_dof_index globalIndex = partitioner->local_to_global(ilocaldof);
	  const bool isGhost = partitioner->is_ghost_entry(globalIndex);
	  if(isGhost)
	    {
	      for(unsigned int iwave = 0; iwave < blockSize; ++iwave)
		{
		  newGhostGlobalNodeIds.push_back(blockSize*globalIndex+iwave);
		}
	    }
	  else
	    {
	      for(unsigned int iwave = 0; iwave < blockSize; ++iwave)
		{
		  newLocallyOwnedGlobalNodeIds.push_back(blockSize*globalIndex+iwave);
		}
	    }

            //insert into dealii index sets
            locallyOwnedFlattenedNodesSet.add_indices(newLocallyOwnedGlobalNodeIds.begin(),newLocallyOwnedGlobalNodeIds.end());
            ghostFlattenedNodesSet.add_indices(newGhostGlobalNodeIds.begin(),newGhostGlobalNodeIds.end());

	}

      //compress index set ranges
      locallyOwnedFlattenedNodesSet.compress();
      ghostFlattenedNodesSet.compress();

      bool print = false;
      if(print)
	{
	  std::cout<<"Number of Wave Functions per Block: "<<blockSize<<std::endl;
	  std::stringstream ss1;locallyOwnedFlattenedNodesSet.print(ss1);
	  std::stringstream ss2;ghostFlattenedNodesSet.print(ss2);
	  std::string s1(ss1.str());s1.pop_back(); std::string s2(ss2.str());s2.pop_back();
	  std::cout<<"procId: "<< dealii::Utilities::MPI::this_mpi_process(mpi_communicator)<< " new owned: "<< s1<< " new ghost: "<< s2<<std::endl;
	}

      //
      //sanity check
      //
      AssertThrow(locallyOwnedFlattenedNodesSet.is_ascending_and_one_to_one(mpi_communicator),
		  dealii::ExcMessage("Incorrect renumbering and/or partitioning of flattened wave function matrix"));

      //
      //create flattened wave function matrix
      //
      flattenedArray.reinit(locallyOwnedFlattenedNodesSet,
			    ghostFlattenedNodesSet,
			    mpi_communicator);
    }



    void computeCellLocalIndexSetMap(const std::shared_ptr< const dealii::Utilities::MPI::Partitioner > & partitioner,
				     const dealii::MatrixFree<3,double>                                 & matrix_free_data,
				     const unsigned int                                                   blockSize,
				     std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayMacroCellLocalProcIndexIdMap,
				     std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap)

    {

      //
      //get FE cell data
      //
      const unsigned int numberMacroCells = matrix_free_data.n_macro_cells();
      const unsigned int numberNodesPerElement = matrix_free_data.get_dofs_per_cell();


      std::vector<dealii::types::global_dof_index> cell_dof_indicesGlobal(numberNodesPerElement);

      //
      //get total locally owned cells
      //
      int totalLocallyOwnedCells = 0;
      for(unsigned int iMacroCell = 0; iMacroCell < numberMacroCells; ++iMacroCell)
	{
	  const  unsigned int n_sub_cells = matrix_free_data.n_components_filled(iMacroCell);
	  for(unsigned int iSubCell = 0; iSubCell < n_sub_cells; ++iSubCell)
	    {
	      totalLocallyOwnedCells++;
	    }
	}

      flattenedArrayMacroCellLocalProcIndexIdMap.clear();
      flattenedArrayMacroCellLocalProcIndexIdMap.resize(totalLocallyOwnedCells);

      //
      //create map for all locally owned cells in the order of macrocell, subcell order
      //
      unsigned int iElem = 0;
      typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;
      for(unsigned int iMacroCell = 0; iMacroCell < numberMacroCells; ++iMacroCell)
	{
	  const unsigned int n_sub_cells = matrix_free_data.n_components_filled(iMacroCell);
	  for(unsigned int iCell = 0; iCell < n_sub_cells; ++iCell)
	    {
	      cellPtr = matrix_free_data.get_cell_iterator(iMacroCell,iCell);
	      cellPtr->get_dof_indices(cell_dof_indicesGlobal);
	      for(unsigned int iNode = 0; iNode < numberNodesPerElement; ++iNode)
		{
		  dealii::types::global_dof_index globalIndex = cell_dof_indicesGlobal[iNode];

		  //Think about variable blockSize
		  dealii::types::global_dof_index globalIndexFlattenedArray = blockSize*globalIndex;
		  dealii::types::global_dof_index localIndexFlattenedArray = partitioner->global_to_local(globalIndexFlattenedArray);
		  flattenedArrayMacroCellLocalProcIndexIdMap[iElem].push_back(localIndexFlattenedArray);
		}//idof loop
	      ++iElem;
	    }//subcell loop
	}//macrocell loop


      //
      //create map for all locally owned cells in the same order
      //
      typename dealii::DoFHandler<3>::active_cell_iterator cell = matrix_free_data.get_dof_handler().begin_active(), endc = matrix_free_data.get_dof_handler().end();
      std::vector<dealii::types::global_dof_index> cell_dof_indices(numberNodesPerElement);

      flattenedArrayCellLocalProcIndexIdMap.clear();
      flattenedArrayCellLocalProcIndexIdMap.resize(totalLocallyOwnedCells);

      unsigned int iElemCount = 0;
      for(; cell!=endc; ++cell)
	{
	  if(cell->is_locally_owned())
	    {
	      cell->get_dof_indices(cell_dof_indices);
	      for(unsigned int iNode = 0; iNode < numberNodesPerElement; ++iNode)
		{
		  dealii::types::global_dof_index globalIndex = cell_dof_indices[iNode];

		  //Think about variable blockSize
		  dealii::types::global_dof_index globalIndexFlattenedArray = blockSize*globalIndex;
		  dealii::types::global_dof_index localIndexFlattenedArray = partitioner->global_to_local(globalIndexFlattenedArray);
		  flattenedArrayCellLocalProcIndexIdMap[iElemCount].push_back(localIndexFlattenedArray);
		}
	      ++iElemCount;
	    }
	}

    }


#ifdef USE_COMPLEX
    void copyFlattenedDealiiVecToSingleCompVec
                             (const dealii::parallel::distributed::Vector<std::complex<double>>  & flattenedArray,
			      const unsigned int                        totalNumberComponents,
			      const std::pair<unsigned int,unsigned int> componentIndexRange,
			      const std::vector<dealii::types::global_dof_index> & localProcDofIndicesReal,
                              const std::vector<dealii::types::global_dof_index> & localProcDofIndicesImag,
			      std::vector<dealii::parallel::distributed::Vector<double>>  & componentVectors)
    {
        Assert(componentVectors.size()==(componentIndexRange.second-componentIndexRange.first),
		  dealii::ExcMessage("Incorrect dimensions of componentVectors"));
        Assert(componentIndexRange.first <totalNumberComponents
		&& componentIndexRange.second <=totalNumberComponents,
		  dealii::ExcMessage("componentIndexRange doesn't lie within totalNumberComponents"));

	const unsigned int localVectorSize = flattenedArray.local_size()/totalNumberComponents;
	for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	  {
	      for(unsigned int icomp = componentIndexRange.first; icomp<componentIndexRange.second; ++icomp)
	      {
		  const unsigned int flattenedArrayLocalIndex =
		      totalNumberComponents*iNode + icomp;

		  componentVectors[icomp-componentIndexRange.first].local_element(localProcDofIndicesReal[iNode])
			= flattenedArray.local_element(flattenedArrayLocalIndex).real();
		  componentVectors[icomp-componentIndexRange.first].local_element(localProcDofIndicesImag[iNode])
			= flattenedArray.local_element(flattenedArrayLocalIndex).imag();
	      }
	  }
	for(unsigned int i=0; i<componentVectors.size(); ++i)
	      componentVectors[i].update_ghost_values();
    }
#else
    void copyFlattenedDealiiVecToSingleCompVec
                             (const dealii::parallel::distributed::Vector<double>  & flattenedArray,
			      const unsigned int                        totalNumberComponents,
			      const std::pair<unsigned int,unsigned int>  componentIndexRange,
			      std::vector<dealii::parallel::distributed::Vector<double>>  & componentVectors)
    {
        Assert(componentVectors.size()==(componentIndexRange.second-componentIndexRange.first),
		  dealii::ExcMessage("Incorrect dimensions of componentVectors"));
        Assert(componentIndexRange.first <totalNumberComponents
		&& componentIndexRange.second <=totalNumberComponents,
		  dealii::ExcMessage("componentIndexRange doesn't lie within totalNumberComponents"));
	const unsigned int localVectorSize = flattenedArray.local_size()/totalNumberComponents;
	for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
	  {
	      for(unsigned int icomp = componentIndexRange.first; icomp<componentIndexRange.second; ++icomp)
	      {
		  const unsigned int flattenedArrayLocalIndex =
		      totalNumberComponents*iNode + icomp;
		  componentVectors[icomp-componentIndexRange.first].local_element(iNode)
			= flattenedArray.local_element(flattenedArrayLocalIndex);
	      }
	  }

	for(unsigned int i=0; i<componentVectors.size(); ++i)
	      componentVectors[i].update_ghost_values();
    }
#endif

    template void createDealiiVector(const std::shared_ptr<const dealii::Utilities::MPI::Partitioner> &,
				     const unsigned int                                                ,
				     dealii::parallel::distributed::Vector<dataTypes::number>     &);


  }//end of namespace

}
