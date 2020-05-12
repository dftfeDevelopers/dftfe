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
// @author Sambit Das
//

template<unsigned int FEOrder>
void forceClass<FEOrder>::computeNonLocalProjectorKetTimesPsiTimesVFlattened
                           (const distributedCPUVec<dataTypes::number> &src,
			    const unsigned int numberWaveFunctions)
{
/*
  vectorTools::createDealiiVector<dataTypes::number>(dftPtr->d_projectorKetTimesVectorPar[0].get_partitioner(),
                                                     numberWaveFunctions,
                                                     dftPtr->d_projectorKetTimesVectorParFlattened);

  std::vector<std::vector<dealii::types::global_dof_index> > flattenedArrayMacroCellLocalProcIndexIdMap;
  std::vector<std::vector<dealii::types::global_dof_index> > flattenedArrayCellLocalProcIndexIdMap;
  vectorTools::computeCellLocalIndexSetMap(src.get_partitioner(),
					   dftPtr->matrix_free_data,
					   numberWaveFunctions,
					   flattenedArrayMacroCellLocalProcIndexIdMap,
					   flattenedArrayCellLocalProcIndexIdMap);

#ifdef USE_COMPLEX
  const unsigned int numberNodesPerElement = dftPtr->FEEigen.dofs_per_cell/2;
#else
  const unsigned int numberNodesPerElement = dftPtr->FEEigen.dofs_per_cell;
#endif

  std::vector<std::vector<dataTypes::number> >  projectorKetTimesVector;
  projectorKetTimesVector.clear();
  projectorKetTimesVector.resize(dftPtr->d_nonLocalAtomIdsInCurrentProcess.size());
  //
  //allocate memory for matrix-vector product
  //
  std::map<unsigned int, unsigned int> globalToLocalMap;
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      globalToLocalMap[atomId]=iAtom;
      const int numberSingleAtomPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      projectorKetTimesVector[iAtom].resize(numberWaveFunctions*numberSingleAtomPseudoWaveFunctions,0.0);
    }

  std::vector<dataTypes::number> cellWaveFunctionMatrix(numberNodesPerElement*numberWaveFunctions,0.0);

  //blas required settings
  const unsigned int inc = 1;


  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  int iElem = -1;
  for(; cell!=endc; ++cell)
    {
      if(cell->is_locally_owned())
	{
	  iElem++;
	  if (dftPtr->d_nonLocalAtomIdsInElement[iElem].size()>0)
	  {
	    for(unsigned int iNode = 0; iNode <numberNodesPerElement; ++iNode)
	    {
	      dealii::types::global_dof_index localNodeId = flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];
#ifdef USE_COMPLEX
	      zcopy_(&numberWaveFunctions,
		     src.begin()+localNodeId,
		     &inc,
		     &cellWaveFunctionMatrix[numberWaveFunctions*iNode],
		     &inc);
#else
	      dcopy_(&numberWaveFunctions,
		     src.begin()+localNodeId,
		     &inc,
		     &cellWaveFunctionMatrix[numberWaveFunctions*iNode],
		     &inc);
#endif
	    }
	  }

	  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInElement[iElem].size();++iAtom)
	    {
	      const unsigned int atomId = dftPtr->d_nonLocalAtomIdsInElement[iElem][iAtom];
	      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
	      const int nonZeroElementMatrixId = dftPtr->d_sparsityPattern[atomId][iElem];

	      const dataTypes::number alpha = dataTypes::number(1.0);
              const dataTypes::number beta = dataTypes::number(1.0);
#ifdef USE_COMPLEX
              const char transA = 'C';
              const char transB = 'T';
	      zgemm_(&transA,
		     &transB,
		     &numberPseudoWaveFunctions,
		     &numberWaveFunctions,
		     &numberNodesPerElement,
		     &alpha,
		     &dftPtr->d_nonLocalProjectorElementMatrices[atomId][nonZeroElementMatrixId][kPointIndex][0],
		     &numberNodesPerElement,
		     &cellWaveFunctionMatrix[0],
		     &numberWaveFunctions,
		     &beta,
		     &projectorKetTimesVector[globalToLocalMap[atomId]][0],
		     &numberPseudoWaveFunctions);
#else
              const char transA = 'T';
              const char transB = 'T';
	      dgemm_(&transA,
		     &transB,
		     &numberPseudoWaveFunctions,
		     &numberWaveFunctions,
		     &numberNodesPerElement,
		     &alpha,
		     &dftPtr->d_nonLocalProjectorElementMatrices[atomId][nonZeroElementMatrixId][0],
		     &numberNodesPerElement,
		     &cellWaveFunctionMatrix[0],
		     &numberWaveFunctions,
		     &beta,
		     &projectorKetTimesVector[globalToLocalMap[atomId]][0],
		     &numberPseudoWaveFunctions);
#endif
	    }


	}

    }//cell loop

  dftPtr->d_projectorKetTimesVectorParFlattened=dataTypes::number(0.0);


  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	{
	  for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	    {
	       const unsigned int id=dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)];
	       dftPtr->d_projectorKetTimesVectorParFlattened[id*numberWaveFunctions+iWave]
		  =projectorKetTimesVector[iAtom][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave];
	    }
	}
    }


  dftPtr->d_projectorKetTimesVectorParFlattened.compress(VectorOperation::add);
  dftPtr->d_projectorKetTimesVectorParFlattened.update_ghost_values();

  //
  //compute V*C^{T}*X
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions =  dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
      {
          const unsigned int id=dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)];
          for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	    dftPtr->d_projectorKetTimesVectorParFlattened[id*numberWaveFunctions+iWave]
                *= dftPtr->d_nonLocalPseudoPotentialConstants[atomId][iPseudoAtomicWave];
      }
    }
  */
}

template<unsigned int FEOrder>
void forceClass<FEOrder>::computeNonLocalProjectorKetTimesPsiTimesVFlattened
                           (const distributedCPUVec<dataTypes::number> &src,
			    const unsigned int numberWaveFunctions,
			    std::vector<std::vector<dataTypes::number> > & projectorKetTimesPsiTimesV,
			    const unsigned int kPointIndex,
			    const std::vector<double> & partialOccupancies,
                            const bool oldRoute)
{

  vectorTools::createDealiiVector<dataTypes::number>(dftPtr->d_projectorKetTimesVectorPar[0].get_partitioner(),
                                                     numberWaveFunctions,
                                                     dftPtr->d_projectorKetTimesVectorParFlattened);

  std::vector<std::vector<dealii::types::global_dof_index> > flattenedArrayMacroCellLocalProcIndexIdMap;
  std::vector<std::vector<dealii::types::global_dof_index> > flattenedArrayCellLocalProcIndexIdMap;
  vectorTools::computeCellLocalIndexSetMap(src.get_partitioner(),
					   dftPtr->matrix_free_data,
					   numberWaveFunctions,
					   flattenedArrayMacroCellLocalProcIndexIdMap,
					   flattenedArrayCellLocalProcIndexIdMap);

#ifdef USE_COMPLEX
  const unsigned int numberNodesPerElement = dftPtr->FEEigen.dofs_per_cell/2;
#else
  const unsigned int numberNodesPerElement = dftPtr->FEEigen.dofs_per_cell;
#endif

  std::vector<std::vector<dataTypes::number> > & projectorKetTimesVector=projectorKetTimesPsiTimesV;
  projectorKetTimesVector.clear();
  projectorKetTimesVector.resize(dftPtr->d_nonLocalAtomIdsInCurrentProcess.size());
  //
  //allocate memory for matrix-vector product
  //
  std::map<unsigned int, unsigned int> globalToLocalMap;
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      globalToLocalMap[atomId]=iAtom;
      const int numberSingleAtomPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      projectorKetTimesVector[iAtom].resize(numberWaveFunctions*numberSingleAtomPseudoWaveFunctions,0.0);
    }

  std::vector<dataTypes::number> cellWaveFunctionMatrix(numberNodesPerElement*numberWaveFunctions,0.0);

  //blas required settings
  const unsigned int inc = 1;


  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  int iElem = -1;
  for(; cell!=endc; ++cell)
    {
      if(cell->is_locally_owned())
	{
	  iElem++;
	  if (dftPtr->d_nonLocalAtomIdsInElement[iElem].size()>0)
	  {
	    for(unsigned int iNode = 0; iNode <numberNodesPerElement; ++iNode)
	    {
	      dealii::types::global_dof_index localNodeId = flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];
#ifdef USE_COMPLEX
	      zcopy_(&numberWaveFunctions,
		     src.begin()+localNodeId,
		     &inc,
		     &cellWaveFunctionMatrix[numberWaveFunctions*iNode],
		     &inc);
#else
	      dcopy_(&numberWaveFunctions,
		     src.begin()+localNodeId,
		     &inc,
		     &cellWaveFunctionMatrix[numberWaveFunctions*iNode],
		     &inc);
#endif
	    }
	  }

	  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInElement[iElem].size();++iAtom)
	    {
	      const unsigned int atomId = dftPtr->d_nonLocalAtomIdsInElement[iElem][iAtom];
	      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
	      const int nonZeroElementMatrixId = dftPtr->d_sparsityPattern[atomId][iElem];

	      const dataTypes::number alpha = dataTypes::number(1.0);
              const dataTypes::number beta = dataTypes::number(1.0);
#ifdef USE_COMPLEX
              const char transA = 'C';
              const char transB = 'T';
	      zgemm_(&transA,
		     &transB,
		     &numberPseudoWaveFunctions,
		     &numberWaveFunctions,
		     &numberNodesPerElement,
		     &alpha,
		     &dftPtr->d_nonLocalProjectorElementMatrices[atomId][nonZeroElementMatrixId][kPointIndex][0],
		     &numberNodesPerElement,
		     &cellWaveFunctionMatrix[0],
		     &numberWaveFunctions,
		     &beta,
		     &projectorKetTimesVector[globalToLocalMap[atomId]][0],
		     &numberPseudoWaveFunctions);
#else
              const char transA = 'T';
              const char transB = 'T';
	      dgemm_(&transA,
		     &transB,
		     &numberPseudoWaveFunctions,
		     &numberWaveFunctions,
		     &numberNodesPerElement,
		     &alpha,
		     &dftPtr->d_nonLocalProjectorElementMatrices[atomId][nonZeroElementMatrixId][0],
		     &numberNodesPerElement,
		     &cellWaveFunctionMatrix[0],
		     &numberWaveFunctions,
		     &beta,
		     &projectorKetTimesVector[globalToLocalMap[atomId]][0],
		     &numberPseudoWaveFunctions);
#endif
	    }


	}

    }//cell loop

  dftPtr->d_projectorKetTimesVectorParFlattened=dataTypes::number(0.0);


  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	{
	  for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	    {
	       const unsigned int id=dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)];
	       dftPtr->d_projectorKetTimesVectorParFlattened[id*numberWaveFunctions+iWave]
		  =projectorKetTimesVector[iAtom][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave];
	    }
	}
    }


  dftPtr->d_projectorKetTimesVectorParFlattened.compress(VectorOperation::add);
  dftPtr->d_projectorKetTimesVectorParFlattened.update_ghost_values();

  if (oldRoute)
  {
	  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
	    {
	      const int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
	      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
	      for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
		{
		  for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
		    {
		      const unsigned int id=dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)];
		      projectorKetTimesVector[iAtom][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave]
			   =dftPtr->d_projectorKetTimesVectorParFlattened[id*numberWaveFunctions+iWave];

		    }
		}
	    }


	  //
	  //compute V*C^{T}*X
	  //
	  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
	    {
	      const int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
	      const unsigned int numberPseudoWaveFunctions =  dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
	      for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
		{
		  for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
		    projectorKetTimesVector[iAtom][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave] *= dftPtr->d_nonLocalPseudoPotentialConstants[atomId][iPseudoAtomicWave]*partialOccupancies[iWave];
		}
	    }
  }
  else
  {
	  //
	  //compute V*C^{T}*X
	  //
	  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
	    {
	      const int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
	      const unsigned int numberPseudoWaveFunctions =  dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
	      for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	      {
		  const unsigned int id=dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)];
		  for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
		    projectorKetTimesVector[iAtom][iPseudoAtomicWave*numberWaveFunctions+iWave] 
		      = dftPtr->d_projectorKetTimesVectorParFlattened[id*numberWaveFunctions+iWave]
			* dftPtr->d_nonLocalPseudoPotentialConstants[atomId][iPseudoAtomicWave]*partialOccupancies[iWave];
	      }
	    }
  }
}
