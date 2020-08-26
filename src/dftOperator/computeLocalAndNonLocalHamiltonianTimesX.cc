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
// @author Phani Motamarri, Department of Computational and Data Sciences, IISc Bangalore
//


/** @file matrixVectorProductImplementations.cc
 *  @brief Contains linear algebra operations
 *
 */

#ifdef USE_COMPLEX
void kohnShamDFTOperatorClass<FEOrder>::computeHamiltonianTimesX(const distributedCPUVec<std::complex<double> > & src,
								      std::vector<std::complex<double> >  & cellSrcWaveFunctionMatrix,
								      const unsigned int numberWaveFunctions,
								      distributedCPUVec<std::complex<double> > & dst,
								      std::vector<std::complex<double> >  & cellDstWaveFunctionMatrix,
								      const double scalar)
{

  AssertThrow(false,dftUtils::ExcNotImplementedYet());

}
#else
template<unsigned int FEOrder>
void kohnShamDFTOperatorClass<FEOrder>::computeHamiltonianTimesX(const distributedCPUVec<double> & src,
										 std::vector<double>  & cellSrcWaveFunctionMatrix,
										 const unsigned int numberWaveFunctions,
										 distributedCPUVec<double> & dst,
										 std::vector<double>  & cellDstWaveFunctionMatrix,
										 const double scalar)
								       
{
	const unsigned int kpointSpinIndex=(1+dftParameters::spinPolarized)*d_kPointIndex+d_spinIndex;
	//
	//element level matrix-vector multiplications
	//
	const char transA = 'N',transB = 'N';
	const double scalarCoeffAlpha1 = scalar,scalarCoeffBeta = 1.0,scalarCoeffAlpha = 1.0;
	const unsigned int inc = 1;

	//std::vector<double> cellWaveFunctionMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);
        //cellWaveFunctionMatrix = d_cellWaveFunctionMatrix;
	//std::vector<double> cellHamMatrixTimesWaveMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);

	unsigned int iElem = 0;
        unsigned int indexTemp1 = d_numberNodesPerElement*numberWaveFunctions;
	for(unsigned int iMacroCell = 0; iMacroCell < d_numberMacroCells; ++iMacroCell)
	  {
	    for(unsigned int iCell = 0; iCell < d_macroCellSubCellMap[iMacroCell]; ++iCell)
	      {
                unsigned int indexTemp2 = indexTemp1*iElem;
		for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
		  {
		    if(d_nodesPerCellClassificationMap[iNode] == 1)
		      {
                        unsigned int indexVal = indexTemp2+numberWaveFunctions*iNode;
			dealii::types::global_dof_index localNodeId = d_flattenedArrayMacroCellLocalProcIndexIdMap[iElem][iNode];

			dcopy_(&numberWaveFunctions,
			       src.begin()+localNodeId,
			       &inc,
			       &cellSrcWaveFunctionMatrix[indexVal],//&cellSrcWaveFunctionMatrix[iElem][numberWaveFunctions*iNode],
			       &inc);


			for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
			  {
			    cellDstWaveFunctionMatrix[indexVal+iWave] = 0.0;
			  }
		      }

		  }
		

		dgemm_(&transA,
		       &transB,
		       &numberWaveFunctions,
		       &d_numberNodesPerElement,
		       &d_numberNodesPerElement,
		       &scalarCoeffAlpha1,
		       &cellSrcWaveFunctionMatrix[indexTemp2],
		       &numberWaveFunctions,
		       &d_cellHamiltonianMatrix[kpointSpinIndex][iElem][0],
		       &d_numberNodesPerElement,
		       &scalarCoeffBeta,
		       &cellDstWaveFunctionMatrix[indexTemp2],
		       &numberWaveFunctions);

		++iElem;
	      }//subcell loop
	  }//macrocell loop

	//
	//start nonlocal HX
	//
	std::map<unsigned int, std::vector<double> > projectorKetTimesVector;

	//
	//allocate memory for matrix-vector product
	//
	if(dftParameters::isPseudopotential && dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0)
	  {
	    for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
	      {
		const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
		const int numberSingleAtomPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
		projectorKetTimesVector[atomId].resize(numberWaveFunctions*numberSingleAtomPseudoWaveFunctions,0.0);
	      }


	
	    //
	    //blas required settings
	    //
	    const double alpha = 1.0;
	    const double beta = 1.0;


	    typename DoFHandler<3>::active_cell_iterator cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
	    int iElem = -1;
	    for(; cell!=endc; ++cell)
	      {
		if(cell->is_locally_owned())
		  {
		    iElem++;
		
		    const unsigned int macroCellId = d_normalCellIdToMacroCellIdMap[iElem];
		    const unsigned int indexVal = d_numberNodesPerElement*numberWaveFunctions*macroCellId;
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
			       &cellSrcWaveFunctionMatrix[indexVal],
			       &numberWaveFunctions,
			       &dftPtr->d_nonLocalProjectorElementMatrices[atomId][nonZeroElementMatrixId][0],
			       &d_numberNodesPerElement,
			       &beta,
			       &projectorKetTimesVector[atomId][0],
			       &numberWaveFunctions);
		      }


		  }

	      }//cell loop


	    dftPtr->d_projectorKetTimesVectorParFlattened=0.0;

	    for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
	      {
		const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
		const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];

		for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
		  {
		    const unsigned int id=dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)];

		    dcopy_(&numberWaveFunctions,
			   &projectorKetTimesVector[atomId][numberWaveFunctions*iPseudoAtomicWave],
			   &inc,
			   &dftPtr->d_projectorKetTimesVectorParFlattened[id*numberWaveFunctions],
			   &inc);

		  }
	      }

	    dftPtr->d_projectorKetTimesVectorParFlattened.compress(VectorOperation::add);
	    dftPtr->d_projectorKetTimesVectorParFlattened.update_ghost_values();

	    //
	    //compute V*C^{T}*X
	    //
	    for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
	      {
		const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
		const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
		for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
		  {
		    double nonlocalConstantV=dftPtr->d_nonLocalPseudoPotentialConstants[atomId][iPseudoAtomicWave];

		    const unsigned int id=dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)];

		    dscal_(&numberWaveFunctions,
			   &nonlocalConstantV,
			   &dftPtr->d_projectorKetTimesVectorParFlattened[id*numberWaveFunctions],
			   &inc);

		    dcopy_(&numberWaveFunctions,
			   &dftPtr->d_projectorKetTimesVectorParFlattened[id*numberWaveFunctions],
			   &inc,
			   &projectorKetTimesVector[atomId][numberWaveFunctions*iPseudoAtomicWave],
			   &inc);

		  }

	      }
	  }

	//start cell loop for assembling localHX and nonlocalHX simultaneously

	cell = dftPtr->dofHandler.begin_active(), endc = dftPtr->dofHandler.end();
	int iElem = -1;
	//blas required settings
	const char transA1 = 'N';
	const char transB1 = 'N';
	const double alpha1 = 1.0;
	const double beta1 = 1.0;
	const unsigned int inc1 = 1;
	const double alpha2 = scalar;
	for(; cell!=endc; ++cell)
	  {
	    if(cell->is_locally_owned())
	      {
		iElem++;
		unsigned int macroCellId = d_normalCellIdToMacroCellIdMap[iElem];
		unsigned int indexValTemp = d_numberNodesPerElement*numberWaveFunctions*macroCellId;

		if(dftParameters::isPseudopotential && dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0)
		  {
		    for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInElement[iElem].size();++iAtom)
		      {
			const unsigned int atomId = dftPtr->d_nonLocalAtomIdsInElement[iElem][iAtom];
			const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
			const int nonZeroElementMatrixId = dftPtr->d_sparsityPattern[atomId][iElem];
		    

			dgemm_(&transA1,
			       &transB1,
			       &numberWaveFunctions,
			       &d_numberNodesPerElement,
			       &numberPseudoWaveFunctions,
			       &alpha2,
			       &projectorKetTimesVector[atomId][0],
			       &numberWaveFunctions,
			       &dftPtr->d_nonLocalProjectorElementMatricesTranspose[atomId][nonZeroElementMatrixId][0],
			       &numberPseudoWaveFunctions,
			       &beta1,
			       &cellDstWaveFunctionMatrix[indexValTemp],
			       &numberWaveFunctions);

		      }
		  }

		for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
		  {

		    if(d_nodesPerCellClassificationMap[iNode] == 1)
		      {
			 dealii::types::global_dof_index localNodeId = d_flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];
			 const unsigned int indexVal  = indexValTemp + numberWaveFunctions*iNode;
			 daxpy_(&numberWaveFunctions,
				&alpha1,
				&cellDstWaveFunctionMatrix[indexVal],
				&inc1,
				dst.begin()+localNodeId,
				&inc1);
		      }
		  }
	      }
	  }//cell loop

}
#endif
