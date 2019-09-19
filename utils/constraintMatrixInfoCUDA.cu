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
// @author  Sambit Das, Phani Motamarri
//
#include <constraintMatrixInfoCUDA.h>

namespace dftfe {
        //Declare dftUtils functions
        namespace dftUtils
	{
          namespace
          {


#if __CUDA_ARCH__ < 600
__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                              (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                               __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

		__global__
		void distributeKernel(const unsigned int contiguousBlockSize,
					double *xVec,
					const unsigned int *constraintLocalRowIdsUnflattened,
					const unsigned int numConstraints,
					const unsigned int  *constraintRowSizes,
					const unsigned int *constraintRowSizesAccumulated,
					const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
					const double *constraintColumnValuesAllRowsUnflattened,
					const double *inhomogenities,
					const dealii::types::global_dof_index *localIndexMapUnflattenedToFlattened)
		{
		  const dealii::types::global_dof_index globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
		  const dealii::types::global_dof_index numberEntries = numConstraints*contiguousBlockSize;

		  for(dealii::types::global_dof_index index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
		   {
		      const unsigned int blockIndex = index/contiguousBlockSize;
		      const unsigned int intraBlockIndex=index%contiguousBlockSize;
		      const unsigned int constrainedRowId=constraintLocalRowIdsUnflattened[blockIndex];
		      const unsigned int numberColumns=constraintRowSizes[blockIndex];
		      const unsigned int startingColumnNumber=constraintRowSizesAccumulated[blockIndex];
		      const dealii::types::global_dof_index xVecStartingIdRow=localIndexMapUnflattenedToFlattened[constrainedRowId];
		      xVec[xVecStartingIdRow+intraBlockIndex]= inhomogenities[blockIndex];
		      for (unsigned int i=0;i<numberColumns;++i)
		      {
			  const unsigned int constrainedColumnId=constraintLocalColumnIdsAllRowsUnflattened[startingColumnNumber+i];
			  const dealii::types::global_dof_index xVecStartingIdColumn=localIndexMapUnflattenedToFlattened[constrainedColumnId];
			  xVec[xVecStartingIdRow+intraBlockIndex]+=constraintColumnValuesAllRowsUnflattened[startingColumnNumber+i]
								   *xVec[xVecStartingIdColumn+intraBlockIndex];
		      }

		   }

		}

				__global__
		void distributeSlaveToMasterKernel(const dealii::types::global_dof_index contiguousBlockSize,
				   double *xVec,
				   const dealii::types::global_dof_index xVecStartingIdRow,
				   const unsigned int numberColumns,
				   const unsigned int startingColumnNumber,
				   const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
				   const double *constraintColumnValuesAllRowsUnflattened,
				   const dealii::types::global_dof_index *localIndexMapUnflattenedToFlattened)
		{
		  const dealii::types::global_dof_index globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
		  const dealii::types::global_dof_index numberEntries = contiguousBlockSize*numberColumns;

		  for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
		   {
		      const unsigned int blockIndex = index/contiguousBlockSize;
		      const unsigned int intraBlockIndex=index%contiguousBlockSize;
	              const unsigned int constrainedColumnId=constraintLocalColumnIdsAllRowsUnflattened[startingColumnNumber+blockIndex];
	              const dealii::types::global_dof_index xVecStartingIdColumn=localIndexMapUnflattenedToFlattened[constrainedColumnId];
		      xVec[xVecStartingIdColumn+intraBlockIndex]+=xVec[xVecStartingIdRow+intraBlockIndex]*constraintColumnValuesAllRowsUnflattened[startingColumnNumber+blockIndex];

		   }

		}


            
                __global__
                void distributeSlaveToMasterKernelAtomicAdd(const unsigned int contiguousBlockSize,
                                        double *xVec,
                                        const unsigned int *constraintLocalRowIdsUnflattened,
                                        const unsigned int numConstraints,
                                        const unsigned int  *constraintRowSizes,
                                        const unsigned int *constraintRowSizesAccumulated,
                                        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
                                        const double *constraintColumnValuesAllRowsUnflattened,
                                        const dealii::types::global_dof_index *localIndexMapUnflattenedToFlattened)
                {
                  const dealii::types::global_dof_index globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
                  const dealii::types::global_dof_index numberEntries = numConstraints*contiguousBlockSize;

                  for(dealii::types::global_dof_index index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
                   {
                      const unsigned int blockIndex = index/contiguousBlockSize;
                      const unsigned int intraBlockIndex=index%contiguousBlockSize;
                      const unsigned int constrainedRowId=constraintLocalRowIdsUnflattened[blockIndex];
                      const unsigned int numberColumns=constraintRowSizes[blockIndex];
                      const unsigned int startingColumnNumber=constraintRowSizesAccumulated[blockIndex];
                      const dealii::types::global_dof_index xVecStartingIdRow=localIndexMapUnflattenedToFlattened[constrainedRowId];
                      for (unsigned int i=0;i<numberColumns;++i)
                      {
                          const unsigned int constrainedColumnId=constraintLocalColumnIdsAllRowsUnflattened[startingColumnNumber+i];
                          const dealii::types::global_dof_index xVecStartingIdColumn=localIndexMapUnflattenedToFlattened[constrainedColumnId];
                          atomicAdd(&(xVec[xVecStartingIdColumn+intraBlockIndex]),
                                    constraintColumnValuesAllRowsUnflattened[startingColumnNumber+i]
                                                                   *xVec[xVecStartingIdRow+intraBlockIndex]);
                      }
                      xVec[xVecStartingIdRow+intraBlockIndex]= 0.0;

                   }

                }



	/*	template<typename T>
		__global__
		void distributeSlaveToMasterNonInteractingBinKernel(const unsigned int contiguousBlockSize,
					T *xVec,
					const unsigned int numColumnConstraintsBin,
					const unsigned int *constraintLocalColumnIdsAllRowsUnflattenedBin,
                                        const unsigned int *columnIdToRowIdMapBinsBin,
					const double *constraintColumnValuesAllRowsUnflattenedBin,
					const dealii::types::global_dof_index *localIndexMapUnflattenedToFlattened)
		{
		  const dealii::types::global_dof_index globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
		  const dealii::types::global_dof_index numberEntries = numColumnConstraintsBin*contiguousBlockSize;

		  for(dealii::types::global_dof_index index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
		   {
		      const unsigned int blockIndex = index/contiguousBlockSize;
		      const unsigned int intraBlockIndex=index%contiguousBlockSize;
		      const unsigned int constrainedRowId=columnIdToRowIdMapBinsBin[blockIndex];
		      const dealii::types::global_dof_index xVecStartingIdRow=localIndexMapUnflattenedToFlattened[constrainedRowId];
                    
	              const unsigned int constrainedColumnId=constraintLocalColumnIdsAllRowsUnflattenedBin[blockIndex];
		      const dealii::types::global_dof_index xVecStartingIdColumn=localIndexMapUnflattenedToFlattened[constrainedColumnId];
			  
                      xVec[xVecStartingIdColumn+intraBlockIndex]
                               +=xVec[xVecStartingIdRow+intraBlockIndex]*constraintColumnValuesAllRowsUnflattenedBin[blockIndex];

		   }

		}*/

		template<typename T>
		__global__
		void setzeroKernel(const unsigned int contiguousBlockSize,
				    T *xVec,
				   const unsigned int *constraintLocalRowIdsUnflattened,
				   const unsigned int numConstraints,
				   const dealii::types::global_dof_index *localIndexMapUnflattenedToFlattened)
		{

		  const dealii::types::global_dof_index globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
		  const dealii::types::global_dof_index numberEntries = numConstraints*contiguousBlockSize;

		  for(dealii::types::global_dof_index index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
		   {
		      const unsigned int blockIndex = index/contiguousBlockSize;
		      const unsigned int intraBlockIndex=index%contiguousBlockSize;
		      xVec[localIndexMapUnflattenedToFlattened[constraintLocalRowIdsUnflattened[blockIndex]]+intraBlockIndex]= 0;
		   }

		}

               
                void createConstraintBins(const std::vector<std::set<unsigned int> > & slaveToMasterSet,
                                          const std::vector<unsigned int> & rowIdsLocal,
                                          const std::vector<unsigned int> & rowSizes,
                                          const std::vector<unsigned int> & rowSizesAccumulated,
                                          const std::vector<unsigned int> & columnIdsLocal,
                                          const std::vector<double> & columnValues,
                                          std::vector<unsigned int> & rowIdsLocalBins,
                                          std::vector<unsigned int> & columnIdsLocalBins,
                                          std::vector<unsigned int> & columnIdToRowIdMapBins,
                                          std::vector<double> & columnValuesBins,
                                          std::vector<unsigned int> & binColumnSizes,
                                          std::vector<unsigned int> & binColumnSizesAccumulated)
                {
                    const unsigned int numRows=rowIdsLocal.size();
                    
                    if (numRows==0)
                       return;

                    std::map<unsigned int,std::set<unsigned int> > interactionMap;

	            for(int irow = 0; irow < numRows; ++irow)
		    {
		      for(int jrow = irow - 1; jrow > -1; jrow--)
		      {
			  std::vector<unsigned int> columnIdsIntersection;

			  std::set_intersection(slaveToMasterSet[irow].begin(),
						slaveToMasterSet[irow].end(),
						slaveToMasterSet[jrow].begin(),
						slaveToMasterSet[jrow].end(),
						std::back_inserter(columnIdsIntersection));

			  if(columnIdsIntersection.size() > 0)
			  {
		             interactionMap[irow].insert(jrow);
			     interactionMap[jrow].insert(irow);
                          }
		       }
                    }//create interaction map

                    std::map<unsigned int,std::set<unsigned int> > bins; 
                    std::map<unsigned int,std::set<unsigned int> >::iterator iter;
		    //start by adding row 0 to bin 0
		    (bins[0]).insert(0);
		    unsigned int binCount = 0;
		    // iterate from row 1 onwards
		    for(int i = 1; i < numRows; ++i)
                    {

			const std::set<unsigned int> & interactingIds = interactionMap[i];
			
                        if(interactingIds.size() == 0)
                        {
			  (bins[binCount]).insert(i);
			  continue;
			}

			bool isBinFound;
			// iterate over each existing bin and see if row i fits into the bin
			for(iter = bins.begin();iter!= bins.end();++iter)
                        {

			  // pick out ids in this bin
			  std::set<unsigned int>& idsInThisBin = iter->second;
			  const unsigned int index = std::distance(bins.begin(),iter);

			  isBinFound = true;

			  // to belong to this bin, this id must not overlap with any other
			  // id already present in this bin
			  for(std::set<unsigned int>::iterator iter2 = interactingIds.begin(); iter2!= interactingIds.end();++iter2)
                          {
			    const unsigned int id = *iter2;

			    if(idsInThisBin.find(id) != idsInThisBin.end())
                            {
			      isBinFound = false;
			      break;
			    }
			  }

			  if(isBinFound == true)
                          {
			    (bins[index]).insert(i);
			    break;
			  }
			}
			// if all current bins have been iterated over w/o a match then
			// create a new bin for this id
			if(isBinFound == false)
                        {
			  binCount++;
			  (bins[binCount]).insert(i);
			}
		    }
                    const unsigned numBins=bins.size();
                    std::cout<<"number constraint bins: "<< bins.size()<<std::endl;

                    rowIdsLocalBins.resize(numRows);
                    columnIdsLocalBins.resize(columnIdsLocal.size());
                    columnValuesBins.resize(columnValues.size());
                    columnIdToRowIdMapBins.resize(columnValues.size());
                    binColumnSizes.resize(numBins);
                    binColumnSizesAccumulated.resize(numBins);

                    unsigned int rowCount=0;
                    unsigned int columnCount=0;
		    for(unsigned int ibin = 0; ibin < numBins; ++ibin)
                    {
                          binColumnSizesAccumulated[ibin]=columnCount;
                          std::set<unsigned int>& idsInThisBin = bins[ibin];
			  for(std::set<unsigned int>::iterator iter2 =idsInThisBin.begin(); iter2!= idsInThisBin.end();++iter2)
                          {
			    const unsigned int id = *iter2;
                            const unsigned int rowIdLocal=rowIdsLocal[id];
                            rowIdsLocalBins[rowCount]=rowIdLocal;
                            const unsigned int rowSize=rowSizes[id];
                            binColumnSizes[ibin]+=rowSize;
                            rowCount++;
                            //std::cout<<"rowCount: "<<rowCount<<std::endl; 
                            const unsigned int startColumnCount= rowSizesAccumulated[id];
                            for (unsigned int j=0; j< rowSize;++j)
                            {
                               //std::cout<<"columnCount: "<<columnCount<<std::endl;
                               //std::cout<<"startColumnCount: "<<startColumnCount<<std::endl;
                               //std::cout<<"rowSize: "<<rowSize<<std::endl;
                               columnIdsLocalBins[columnCount]=columnIdsLocal[startColumnCount+j];
                               columnValuesBins[columnCount]=columnValues[startColumnCount+j];
                               columnIdToRowIdMapBins[columnCount]=rowIdLocal;
                               columnCount++;
                            } 

			  }
                    }
	        }

          }

	  //constructor
	  //
	  constraintMatrixInfoCUDA::constraintMatrixInfoCUDA()
	  {


	  }

	  //
	  //destructor
	  //
	  constraintMatrixInfoCUDA::~constraintMatrixInfoCUDA()
	  {


	  }


	  //
	  //store constraintMatrix row data in STL vector
	  //
	  void constraintMatrixInfoCUDA::initialize(const std::shared_ptr< const dealii::Utilities::MPI::Partitioner > & partitioner,
						const dealii::ConstraintMatrix & constraintMatrixData)

	  {

	    clear();
	    const dealii::IndexSet & locally_owned_dofs = partitioner->locally_owned_range();
	    const dealii::IndexSet & ghost_dofs = partitioner->ghost_indices();

            dealii::types::global_dof_index count=0;
            std::vector<std::set<unsigned int> > slaveToMasterSet;
	    for(dealii::IndexSet::ElementIterator it = locally_owned_dofs.begin(); it != locally_owned_dofs.end();++it)
	      {
		if(constraintMatrixData.is_constrained(*it))
		  {
		    const dealii::types::global_dof_index lineDof = *it;
		    d_rowIdsLocal.push_back(partitioner->global_to_local(lineDof));
		    d_inhomogenities.push_back(constraintMatrixData.get_inhomogeneity(lineDof));
		    const std::vector<std::pair<dealii::types::global_dof_index, double > > * rowData=constraintMatrixData.get_constraint_entries(lineDof);
		    d_rowSizes.push_back(rowData->size());
                    d_rowSizesAccumulated.push_back(count);
                    count+=rowData->size();
                    std::set<unsigned int> columnIds;
		    for(unsigned int j = 0; j < rowData->size();++j)
		      {
			Assert((*rowData)[j].first<partitioner->size(),
			   dealii::ExcMessage("Index out of bounds"));
                        const unsigned int columnId=partitioner->global_to_local((*rowData)[j].first);
			d_columnIdsLocal.push_back(columnId);
			d_columnValues.push_back((*rowData)[j].second);
                        columnIds.insert(columnId);
		      }
                    slaveToMasterSet.push_back(columnIds);
		  }
	      }


	    for(dealii::IndexSet::ElementIterator it = ghost_dofs.begin(); it != ghost_dofs.end();++it)
	      {
		if(constraintMatrixData.is_constrained(*it))
		  {
		    const dealii::types::global_dof_index lineDof = *it;
		    d_rowIdsLocal.push_back(partitioner->global_to_local(lineDof));
		    d_inhomogenities.push_back(constraintMatrixData.get_inhomogeneity(lineDof));
		    const std::vector<std::pair<dealii::types::global_dof_index, double > > * rowData=constraintMatrixData.get_constraint_entries(lineDof);
		    d_rowSizes.push_back(rowData->size());
                    d_rowSizesAccumulated.push_back(count);
                    count+=rowData->size();
                    std::set<unsigned int> columnIds;
		    for(unsigned int j = 0; j < rowData->size();++j)
		      {
			Assert((*rowData)[j].first<partitioner->size(),
			       dealii::ExcMessage("Index out of bounds"));
                        const unsigned int columnId=partitioner->global_to_local((*rowData)[j].first);
			d_columnIdsLocal.push_back(columnId);
			d_columnValues.push_back((*rowData)[j].second);
                        columnIds.insert(columnId);
		      }
                    slaveToMasterSet.push_back(columnIds);
		  }
	      }

	     d_rowIdsLocalDevice=d_rowIdsLocal;
             d_columnIdsLocalDevice=d_columnIdsLocal;
	     d_columnValuesDevice=d_columnValues;
	     d_inhomogenitiesDevice=d_inhomogenities;
	     d_rowSizesDevice=d_rowSizes;
             d_rowSizesAccumulatedDevice=d_rowSizesAccumulated;
             d_numConstrainedDofs=d_rowIdsLocal.size();

             /*
             createConstraintBins(slaveToMasterSet,
                                  d_rowIdsLocal,
                                  d_rowSizes,
                                  d_rowSizesAccumulated,
                                  d_columnIdsLocal,
                                  d_columnValues,
                                  d_rowIdsLocalBins,
                                  d_columnIdsLocalBins,
                                  d_columnIdToRowIdMapBins,
                                  d_columnValuesBins,
                                  d_binColumnSizes,
                                  d_binColumnSizesAccumulated);


	     d_rowIdsLocalBinsDevice=d_rowIdsLocalBins;
	     d_columnIdsLocalBinsDevice=d_columnIdsLocalBins;
             d_columnIdToRowIdMapBinsDevice=d_columnIdToRowIdMapBins;
	     d_columnValuesBinsDevice=d_columnValuesBins;
             */
	  }


	  void constraintMatrixInfoCUDA::precomputeMaps(const std::shared_ptr< const dealii::Utilities::MPI::Partitioner> & unFlattenedPartitioner,
						    const std::shared_ptr< const dealii::Utilities::MPI::Partitioner> & flattenedPartitioner,
						    const unsigned int blockSize)
	  {

	    //
	    //Get required sizes
	    //
	    const unsigned int n_ghosts   = unFlattenedPartitioner->n_ghost_indices();
	    const unsigned int localSize  = unFlattenedPartitioner->local_size();
	    const unsigned int totalSize = n_ghosts + localSize;

	    d_localIndexMapUnflattenedToFlattened.clear();
	    d_localIndexMapUnflattenedToFlattened.resize(totalSize);

	    //
	    //fill the data array
	    //
	    for(unsigned int ilocalDof = 0; ilocalDof < totalSize; ++ilocalDof)
	      {
		const dealii::types::global_dof_index globalIndex = unFlattenedPartitioner->local_to_global(ilocalDof);
		d_localIndexMapUnflattenedToFlattened[ilocalDof] = flattenedPartitioner->global_to_local(globalIndex*blockSize);
	      }
            d_localIndexMapUnflattenedToFlattenedDevice=d_localIndexMapUnflattenedToFlattened; 

	  }



	  
	  void constraintMatrixInfoCUDA::distribute(cudaVectorType & fieldVector,
						const unsigned int blockSize) const
	  {
            if (d_numConstrainedDofs==0)
               return;
	    //fieldVector.update_ghost_values();
            
	    distributeKernel<<<min((blockSize+255)/256*d_numConstrainedDofs,30000), 256>>>(blockSize,
					   fieldVector.begin(),
					   thrust::raw_pointer_cast(&d_rowIdsLocalDevice[0]),
					   d_numConstrainedDofs,
					   thrust::raw_pointer_cast(&d_rowSizesDevice[0]),
					   thrust::raw_pointer_cast(&d_rowSizesAccumulatedDevice[0]),
					   thrust::raw_pointer_cast(&d_columnIdsLocalDevice[0]),
					   thrust::raw_pointer_cast(&d_columnValuesDevice[0]),
					   thrust::raw_pointer_cast(&d_inhomogenitiesDevice[0]),
					   thrust::raw_pointer_cast(&d_localIndexMapUnflattenedToFlattenedDevice[0]));

            
	  }





	  //
	  //set the constrained degrees of freedom to values so that constraints
	  //are satisfied for flattened array
	  //
	  
	  void constraintMatrixInfoCUDA::distribute_slave_to_master(cudaVectorType & fieldVector,
								const unsigned int blockSize) const
	  {
            if (d_numConstrainedDofs==0)
               return;
            distributeSlaveToMasterKernelAtomicAdd<<<min((blockSize+255)/256*d_numConstrainedDofs,30000), 256>>>(blockSize,
                                           fieldVector.begin(),
                                           thrust::raw_pointer_cast(&d_rowIdsLocalDevice[0]),
                                           d_numConstrainedDofs,
                                           thrust::raw_pointer_cast(&d_rowSizesDevice[0]),
                                           thrust::raw_pointer_cast(&d_rowSizesAccumulatedDevice[0]),
                                           thrust::raw_pointer_cast(&d_columnIdsLocalDevice[0]),
                                           thrust::raw_pointer_cast(&d_columnValuesDevice[0]),
                                           thrust::raw_pointer_cast(&d_localIndexMapUnflattenedToFlattenedDevice[0]));
            /*
            for (unsigned int i=0; i<d_binColumnSizesAccumulated.size();i++)
            {
                    const unsigned int binColumnStartId=d_binColumnSizesAccumulated[i];
                    const unsigned int binColumnsSize=d_binColumnSizes[i];

		    distributeSlaveToMasterNonInteractingBinKernel<T><<<min((blockSize+256)/256*binColumnsSize,30000), 256>>>(blockSize,
						   thrust::raw_pointer_cast(&fieldVector[0]),
						   binColumnsSize,
						   thrust::raw_pointer_cast(&d_columnIdsLocalBinsDevice[binColumnStartId]),
                                                   thrust::raw_pointer_cast(&d_columnIdToRowIdMapBinsDevice[binColumnStartId]),
						   thrust::raw_pointer_cast(&d_columnValuesBinsDevice[binColumnStartId]),
						   thrust::raw_pointer_cast(&d_localIndexMapUnflattenedToFlattenedDevice[0]));          
            }
            
            set_zero(fieldVector,
                     blockSize);
            */
	  }

	  template<typename T>
	  void constraintMatrixInfoCUDA::set_zero(thrust::device_vector<T>& fieldVector,
					      const unsigned int blockSize) const
	  {
            if (d_numConstrainedDofs==0)
               return;

             const unsigned int numConstrainedDofs=d_rowIdsLocal.size();
	     setzeroKernel<T><<<min((blockSize+255)/256*numConstrainedDofs,30000), 256>>>(blockSize,
					  thrust::raw_pointer_cast(&fieldVector[0]),
					  thrust::raw_pointer_cast(&d_rowIdsLocalDevice[0]),
					  numConstrainedDofs,
					  thrust::raw_pointer_cast(&d_localIndexMapUnflattenedToFlattenedDevice[0]));
	  }

	  //
	  //
	  //clear the data variables
	  //
	  void constraintMatrixInfoCUDA::clear()
	  {
	    d_rowIdsLocal.clear();
	    d_columnIdsLocal.clear();
	    d_columnValues.clear();
	    d_inhomogenities.clear();
	    d_rowSizes.clear();
            d_rowSizesAccumulated.clear();
	    d_rowIdsLocalBins.clear();
	    d_columnIdsLocalBins.clear();
	    d_columnValuesBins.clear();
            d_binColumnSizesAccumulated.clear();
            d_binColumnSizes.clear();

	    d_rowIdsLocalDevice.clear();
	    d_columnIdsLocalDevice.clear();
	    d_columnValuesDevice.clear();
	    d_inhomogenitiesDevice.clear();
	    d_rowSizesDevice.clear();
            d_rowSizesAccumulatedDevice.clear();
	    d_rowIdsLocalBinsDevice.clear();
	    d_columnIdsLocalBinsDevice.clear();
	    d_columnValuesBinsDevice.clear();
	  }


	 
	  template void constraintMatrixInfoCUDA::set_zero(thrust::device_vector<double>& fieldVector,
							 const unsigned int blockSize) const;

	}

}

