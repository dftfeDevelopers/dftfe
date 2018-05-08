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
// @author Phani Motamarri, Sambit Das 
//


template<unsigned int FEOrder>
void eigenClass<FEOrder>::computeNonLocalHamiltonianTimesX(const std::vector<vectorType> &src,
							   std::vector<vectorType>       &dst)const
{

  //
  //get access to triangulation objects from meshGenerator class
  //
  const unsigned int kPointIndex = d_kPointIndex;
  const unsigned int dofs_per_cell = dftPtr->FEEigen.dofs_per_cell;

#ifdef ENABLE_PERIODIC_BC
  const unsigned int numberNodesPerElement = dftPtr->FEEigen.dofs_per_cell/2;//GeometryInfo<3>::vertices_per_cell;
#else
  const unsigned int numberNodesPerElement = dftPtr->FEEigen.dofs_per_cell;
#endif

  //
  //compute nonlocal projector ket times x i.e C^{T}*X
  //
  std::vector<dealii::types::global_dof_index> local_dof_indices(dofs_per_cell);
#ifdef ENABLE_PERIODIC_BC
  std::map<unsigned int, std::vector<std::complex<double> > > projectorKetTimesVector;
#else
  std::map<unsigned int, std::vector<double> > projectorKetTimesVector;
#endif

  const unsigned int numberWaveFunctions = src.size();
  projectorKetTimesVector.clear();

  //
  //allocate memory for matrix-vector product
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberSingleAtomPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      projectorKetTimesVector[atomId].resize(numberWaveFunctions*numberSingleAtomPseudoWaveFunctions,0.0);
    }

  //
  //some useful vectors
  //
#ifdef ENABLE_PERIODIC_BC
  std::vector<std::complex<double> > inputVectors(numberNodesPerElement*numberWaveFunctions,0.0);
#else
  std::vector<double> inputVectors(numberNodesPerElement*numberWaveFunctions,0.0);
#endif


  //
  //parallel loop over all elements to compute nonlocal projector ket times x i.e C^{T}*X
  //
  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandlerEigen.begin_active(), endc = dftPtr->dofHandlerEigen.end();
  int iElem = -1;
  for(; cell!=endc; ++cell)
    {
      if(cell->is_locally_owned())
	{
	  iElem ++;
	  cell->get_dof_indices(local_dof_indices);

	  unsigned int index=0;

	  std::vector<double> temp(dofs_per_cell,0.0);
	  for (std::vector<vectorType>::const_iterator it=src.begin(); it!=src.end(); it++)
	    {
#ifdef ENABLE_PERIODIC_BC
	      (*it).extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), temp.begin());
	      for(unsigned int idof = 0; idof < dofs_per_cell; ++idof)
		{
		  //
		  //This is the component index 0(real) or 1(imag).
		  //
		  const unsigned int ck = dftPtr->FEEigen.system_to_component_index(idof).first;
		  const unsigned int iNode = dftPtr->FEEigen.system_to_component_index(idof).second;
		  if(ck == 0)
		    inputVectors[numberNodesPerElement*index + iNode].real(temp[idof]);
		  else
		    inputVectors[numberNodesPerElement*index + iNode].imag(temp[idof]);
		}
#else
	      (*it).extract_subvector_to(local_dof_indices.begin(), local_dof_indices.end(), inputVectors.begin()+numberNodesPerElement*index);
#endif
	      index++;
	    }


	  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInElement[iElem].size();++iAtom)
	    {
	      const int atomId = dftPtr->d_nonLocalAtomIdsInElement[iElem][iAtom];
	      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
	      const int nonZeroElementMatrixId = dftPtr->d_sparsityPattern[atomId][iElem];
#ifdef ENABLE_PERIODIC_BC
	      const char transA = 'C';
	      const char transB = 'N';
	      const std::complex<double> alpha = 1.0;
	      const std::complex<double> beta = 1.0;
	      zgemm_(&transA,
		     &transB,
		     &numberPseudoWaveFunctions,
		     &numberWaveFunctions,
		     &numberNodesPerElement,
		     &alpha,
		     &dftPtr->d_nonLocalProjectorElementMatrices[atomId][nonZeroElementMatrixId][kPointIndex][0],
		     &numberNodesPerElement,
		     &inputVectors[0],
		     &numberNodesPerElement,
		     &beta,
		     &projectorKetTimesVector[atomId][0],
		     &numberPseudoWaveFunctions);
#else
	      const char transA = 'T';
	      const char transB = 'N';
	      const double alpha = 1.0;
	      const double beta = 1.0;
	      dgemm_(&transA,
		     &transB,
		     &numberPseudoWaveFunctions,
		     &numberWaveFunctions,
		     &numberNodesPerElement,
		     &alpha,
		     &dftPtr->d_nonLocalProjectorElementMatrices[atomId][nonZeroElementMatrixId][kPointIndex][0],
		     &numberNodesPerElement,
		     &inputVectors[0],
		     &numberNodesPerElement,
		     &beta,
		     &projectorKetTimesVector[atomId][0],
		     &numberPseudoWaveFunctions);
#endif
	    }

	}

    }//element loop

  for (unsigned int i=0; i<numberWaveFunctions;++i)
    {
#ifdef ENABLE_PERIODIC_BC
      dftPtr->d_projectorKetTimesVectorPar[i]=std::complex<double>(0.0,0.0);
#else
      dftPtr->d_projectorKetTimesVectorPar[i]=0;
#endif
    }



  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	{
	  for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	    {
	      dftPtr->d_projectorKetTimesVectorPar[iWave][dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)]]
		=projectorKetTimesVector[atomId][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave];
	    }
	}
    }

  for (unsigned int i=0; i<numberWaveFunctions;++i)
    {
      dftPtr->d_projectorKetTimesVectorPar[i].compress(VectorOperation::add);
      dftPtr->d_projectorKetTimesVectorPar[i].update_ghost_values();
    }

  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	{
	  for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	    {
	      projectorKetTimesVector[atomId][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave]
		=dftPtr->d_projectorKetTimesVectorPar[iWave][dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)]];

	    }
	}
    }


  //
  //compute V*C^{T}*X
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions =  dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	{
	  for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	    projectorKetTimesVector[atomId][numberPseudoWaveFunctions*iWave + iPseudoAtomicWave] *= dftPtr->d_nonLocalPseudoPotentialConstants[atomId][iPseudoAtomicWave];
	}
    }

  //std::cout<<"Scaling V*C^{T} "<<std::endl;

  const char transA1 = 'N';
  const char transB1 = 'N';

  //
  //access elementIdsInAtomCompactSupport
  //

#ifdef ENABLE_PERIODIC_BC
  std::vector<std::complex<double> > outputVectors(numberNodesPerElement*numberWaveFunctions,0.0);
#else
  std::vector<double> outputVectors(numberNodesPerElement*numberWaveFunctions,0.0);
#endif

  //
  //compute C*V*C^{T}*x
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions =  dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(unsigned int iElemComp = 0; iElemComp < dftPtr->d_elementIteratorsInAtomCompactSupport[atomId].size(); ++iElemComp)
	{

	  DoFHandler<3>::active_cell_iterator cell = dftPtr->d_elementIteratorsInAtomCompactSupport[atomId][iElemComp];

#ifdef ENABLE_PERIODIC_BC
	  const std::complex<double> alpha1 = 1.0;
	  const std::complex<double> beta1 = 0.0;

	  zgemm_(&transA1,
		 &transB1,
		 &numberNodesPerElement,
		 &numberWaveFunctions,
		 &numberPseudoWaveFunctions,
		 &alpha1,
		 &dftPtr->d_nonLocalProjectorElementMatrices[atomId][iElemComp][kPointIndex][0],
		 &numberNodesPerElement,
		 &projectorKetTimesVector[atomId][0],
		 &numberPseudoWaveFunctions,
		 &beta1,
		 &outputVectors[0],
		 &numberNodesPerElement);
#else
	  const double alpha1 = 1.0;
	  const double beta1 = 0.0;

	  dgemm_(&transA1,
		 &transB1,
		 &numberNodesPerElement,
		 &numberWaveFunctions,
		 &numberPseudoWaveFunctions,
		 &alpha1,
		 &dftPtr->d_nonLocalProjectorElementMatrices[atomId][iElemComp][kPointIndex][0],
		 &numberNodesPerElement,
		 &projectorKetTimesVector[atomId][0],
		 &numberPseudoWaveFunctions,
		 &beta1,
		 &outputVectors[0],
		 &numberNodesPerElement);
#endif

	  cell->get_dof_indices(local_dof_indices);

#ifdef ENABLE_PERIODIC_BC
	  unsigned int index = 0;
	  std::vector<double> temp(dofs_per_cell,0.0);
	  for(std::vector<vectorType>::iterator it = dst.begin(); it != dst.end(); ++it)
	    {
	      for(unsigned int idof = 0; idof < dofs_per_cell; ++idof)
		{
		  const unsigned int ck = dftPtr->FEEigen.system_to_component_index(idof).first;
		  const unsigned int iNode = dftPtr->FEEigen.system_to_component_index(idof).second;

		  if(ck == 0)
		    temp[idof] = outputVectors[numberNodesPerElement*index + iNode].real();
		  else
		    temp[idof] = outputVectors[numberNodesPerElement*index + iNode].imag();

		}
	      dftPtr->constraintsNoneEigen.distribute_local_to_global(temp.begin(), temp.end(),local_dof_indices.begin(), *it);
	      index++;
	    }
#else
	  std::vector<double>::iterator iter = outputVectors.begin();
	  for (std::vector<vectorType>::iterator it=dst.begin(); it!=dst.end(); ++it)
	    {
	      dftPtr->constraintsNoneEigen.distribute_local_to_global(iter, iter+numberNodesPerElement,local_dof_indices.begin(), *it);
	      iter+=numberNodesPerElement;
	    }
#endif

	}

    }
}


#ifdef ENABLE_PERIODIC_BC
template<unsigned int FEOrder>
void eigenClass<FEOrder>::computeNonLocalHamiltonianTimesX(const dealii::parallel::distributed::Vector<std::complex<double> > & src,
							   const unsigned int numberWaveFunctions,
							   const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
							   dealii::parallel::distributed::Vector<std::complex<double> >       & dst) const
{
  
  std::map<unsigned int, std::vector<std::complex<double> > > projectorKetTimesVector;
  projectorKetTimesVector.clear();

  //
  //allocate memory for matrix-vector product
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const int numberSingleAtomPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      projectorKetTimesVector[atomId].resize(numberWaveFunctions*numberSingleAtomPseudoWaveFunctions,0.0);
    }

  
  std::vector<std::complex<double> > cellWaveFunctionMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);

  //
  //blas required settings
  //
  const char transA = 'N';
  const char transB = 'N';
  const std::complex<double> alpha = 1.0;
  const std::complex<double> beta = 1.0;
  const unsigned int inc = 1;


  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  int iElem = -1;
  for(; cell!=endc; ++cell)
    {
      if(cell->is_locally_owned())
	{
	  iElem++;
	  for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
	    {
	      int localNodeId = flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];
	      zcopy_(&numberWaveFunctions,
		     src.begin()+localNodeId,
		     &inc,
		     &cellWaveFunctionMatrix[numberWaveFunctions*iNode],
		     &inc);
	    }

	  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInElement[iElem].size();++iAtom)
	    {
	      const unsigned int atomId = dftPtr->d_nonLocalAtomIdsInElement[iElem][iAtom];
	      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
	      const int nonZeroElementMatrixId = dftPtr->d_sparsityPattern[atomId][iElem];

	      zgemm_(&transA,
		     &transB,
		     &numberWaveFunctions,
		     &numberPseudoWaveFunctions,
		     &d_numberNodesPerElement,
		     &alpha,
		     &cellWaveFunctionMatrix[0],
		     &numberWaveFunctions,
		     &dftPtr->d_nonLocalProjectorElementMatricesConjugate[atomId][nonZeroElementMatrixId][d_kPointIndex][0],
		     &d_numberNodesPerElement,
		     &beta,
		     &projectorKetTimesVector[atomId][0],
		     &numberWaveFunctions);
	    }


	}

    }//cell loop
  

  for(unsigned int i=0; i<numberWaveFunctions;++i)
    dftPtr->d_projectorKetTimesVectorPar[i] = std::complex<double>(0.0,0.0);
  


  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      
      for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	{
	  for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	    {
	      
	      dftPtr->d_projectorKetTimesVectorPar[iWave][dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)]]=projectorKetTimesVector[atomId][numberWaveFunctions*iPseudoAtomicWave + iWave];

	    }
	}
    }

  for (unsigned int i=0; i<numberWaveFunctions;++i)
    {
      dftPtr->d_projectorKetTimesVectorPar[i].compress(VectorOperation::add);
      dftPtr->d_projectorKetTimesVectorPar[i].update_ghost_values();
    }


  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	{
	  for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	    {
	      projectorKetTimesVector[atomId][numberWaveFunctions*iPseudoAtomicWave + iWave] = dftPtr->d_projectorKetTimesVectorPar[iWave][dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)]];
	    }

	}

    }
  
  //
  //compute V*C^{T}*X
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId = dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions =  dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	{
	  for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	    projectorKetTimesVector[atomId][numberWaveFunctions*iPseudoAtomicWave + iWave] *= dftPtr->d_nonLocalPseudoPotentialConstants[atomId][iPseudoAtomicWave];
	}
    }


  std::vector<std::complex<double> > cellNonLocalHamTimesWaveMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);

  //blas required settings
  const char transA1 = 'N';
  const char transB1 = 'T';
  const std::complex<double> alpha1 = 1.0;
  const std::complex<double> beta1 = 0.0;
  const unsigned int inc1 = 1;

  //
  //compute C*V*C^{T}*x
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId = dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
       for(unsigned int iElemComp = 0; iElemComp < dftPtr->d_elementIteratorsInAtomCompactSupport[atomId].size(); ++iElemComp)
	{
	 
	  zgemm_(&transA1,
	         &transB1,
                 &numberWaveFunctions,
                 &d_numberNodesPerElement,
		 &numberPseudoWaveFunctions,
		 &alpha1,
		 &projectorKetTimesVector[atomId][0],
		 &numberWaveFunctions,
		 &dftPtr->d_nonLocalProjectorElementMatrices[atomId][iElemComp][d_kPointIndex][0],
		 &d_numberNodesPerElement,
		 &beta1,
		 &cellNonLocalHamTimesWaveMatrix[0],
		 &numberWaveFunctions);

        unsigned int elementId =  dftPtr->d_elementIdsInAtomCompactSupport[atomId][iElemComp]; 		 

	for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
	    {
	      int localNodeId = flattenedArrayCellLocalProcIndexIdMap[elementId][iNode];
	      zaxpy_(&numberWaveFunctions,
		     &alpha1,
		     &cellNonLocalHamTimesWaveMatrix[numberWaveFunctions*iNode],
		     &inc1,
		     dst.begin()+localNodeId,
		     &inc1);
	    }


	}

    }

}
#else
template<unsigned int FEOrder>
void eigenClass<FEOrder>::computeNonLocalHamiltonianTimesX(const dealii::parallel::distributed::Vector<double> & src,
							   const unsigned int numberWaveFunctions,
							   const std::vector<std::vector<dealii::types::global_dof_index> > & flattenedArrayCellLocalProcIndexIdMap,
							   dealii::parallel::distributed::Vector<double>       & dst) const
{

  
  std::map<unsigned int, std::vector<double> > projectorKetTimesVector;
  projectorKetTimesVector.clear();

  //
  //allocate memory for matrix-vector product
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const int numberSingleAtomPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      projectorKetTimesVector[atomId].resize(numberWaveFunctions*numberSingleAtomPseudoWaveFunctions,0.0);
    }

  
  std::vector<double> cellWaveFunctionMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);

  //
  //blas required settings
  //
  const char transA = 'N';
  const char transB = 'N';
  const double alpha = 1.0;
  const double beta = 1.0;
  const unsigned int inc = 1;


  typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
  int iElem = -1;
  for(; cell!=endc; ++cell)
    {
      if(cell->is_locally_owned())
	{
	  iElem++;
	  for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
	    {
	      int localNodeId = flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];
	      dcopy_(&numberWaveFunctions,
		     src.begin()+localNodeId,
		     &inc,
		     &cellWaveFunctionMatrix[numberWaveFunctions*iNode],
		     &inc);
	    }

	  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInElement[iElem].size();++iAtom)
	    {
	      const unsigned int atomId = dftPtr->d_nonLocalAtomIdsInElement[iElem][iAtom];
	      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
	      const int nonZeroElementMatrixId = dftPtr->d_sparsityPattern[atomId][iElem];

	      dgemm_(&transA,
		     &transB,
		     &numberWaveFunctions,
		     &numberPseudoWaveFunctions,
		     &d_numberNodesPerElement,
		     &alpha,
		     &cellWaveFunctionMatrix[0],
		     &numberWaveFunctions,
		     &dftPtr->d_nonLocalProjectorElementMatrices[atomId][nonZeroElementMatrixId][d_kPointIndex][0],
		     &d_numberNodesPerElement,
		     &beta,
		     &projectorKetTimesVector[atomId][0],
		     &numberWaveFunctions);
	    }


	}

    }//cell loop
  

  for(unsigned int i=0; i<numberWaveFunctions;++i)
    dftPtr->d_projectorKetTimesVectorPar[i] = 0.0;
  


  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      
      for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	{
	  for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	    {
	      
	      dftPtr->d_projectorKetTimesVectorPar[iWave][dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)]]=projectorKetTimesVector[atomId][numberWaveFunctions*iPseudoAtomicWave + iWave];

	    }
	}
    }

  for (unsigned int i=0; i<numberWaveFunctions;++i)
    {
      dftPtr->d_projectorKetTimesVectorPar[i].compress(VectorOperation::add);
      dftPtr->d_projectorKetTimesVectorPar[i].update_ghost_values();
    }


  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	{
	  for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	    {
	      projectorKetTimesVector[atomId][numberWaveFunctions*iPseudoAtomicWave + iWave] = dftPtr->d_projectorKetTimesVectorPar[iWave][dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)]];
	    }

	}

    }
  
  
  //
  //compute V*C^{T}*X
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId = dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions =  dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
	{
	  for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
	    projectorKetTimesVector[atomId][numberWaveFunctions*iPseudoAtomicWave + iWave] *= dftPtr->d_nonLocalPseudoPotentialConstants[atomId][iPseudoAtomicWave];
	}
    }


  

  std::vector<double> cellNonLocalHamTimesWaveMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);

  //blas required settings
  const char transA1 = 'N';
  const char transB1 = 'T';
  const double alpha1 = 1.0;
  const double beta1 = 0.0;
  const unsigned int inc1 = 1;

  //
  //compute C*V*C^{T}*x
  //
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId = dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
       for(unsigned int iElemComp = 0; iElemComp < dftPtr->d_elementIteratorsInAtomCompactSupport[atomId].size(); ++iElemComp)
	{
	 
	  dgemm_(&transA1,
	         &transB1,
                 &numberWaveFunctions,
                 &d_numberNodesPerElement,
		 &numberPseudoWaveFunctions,
		 &alpha1,
		 &projectorKetTimesVector[atomId][0],
		 &numberWaveFunctions,
		 &dftPtr->d_nonLocalProjectorElementMatrices[atomId][iElemComp][d_kPointIndex][0],
		 &d_numberNodesPerElement,
		 &beta1,
		 &cellNonLocalHamTimesWaveMatrix[0],
		 &numberWaveFunctions);

        unsigned int elementId =  dftPtr->d_elementIdsInAtomCompactSupport[atomId][iElemComp]; 		 

	for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
	    {
	      int localNodeId = flattenedArrayCellLocalProcIndexIdMap[elementId][iNode];
	      daxpy_(&numberWaveFunctions,
		     &alpha1,
		     &cellNonLocalHamTimesWaveMatrix[numberWaveFunctions*iNode],
		     &inc1,
		     dst.begin()+localNodeId,
		     &inc1);
	    }


	}

    }

}
#endif




