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


#ifdef USE_COMPLEX
template<unsigned int FEOrder,unsigned int FEOrderElectro>
void kohnShamDFTOperatorClass<FEOrder,FEOrderElectro>::computeNonLocalHamiltonianTimesX(const distributedCPUVec<std::complex<double> > & src,
											const unsigned int numberWaveFunctions,
											distributedCPUVec<std::complex<double> >       & dst,
											const double scalar) const
{

	std::map<unsigned int, std::vector<std::complex<double> > > projectorKetTimesVector;

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
			if (dftPtr->d_nonLocalAtomIdsInElement[iElem].size()>0)
			{
				for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
				{
					dealii::types::global_dof_index localNodeId = d_flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];
					zcopy_(&numberWaveFunctions,
							src.begin()+localNodeId,
							&inc,
							&cellWaveFunctionMatrix[numberWaveFunctions*iNode],
							&inc);
				}
			}

			for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInElement[iElem].size();++iAtom)
			{
				const unsigned int atomId = dftPtr->d_nonLocalAtomIdsInElement[iElem][iAtom];
				const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
				const int nonZeroElementMatrixId = dftPtr->d_sparsityPattern[atomId][iElem];

				if(dftParameters::cellLevelMassMatrixScaling)
				  {
				    zgemm_(&transA,
					   &transB,
					   &numberWaveFunctions,
					   &numberPseudoWaveFunctions,
					   &d_numberNodesPerElement,
					   &alpha,
					   &cellWaveFunctionMatrix[0],
					   &numberWaveFunctions,
					   &dftPtr->d_nonLocalProjectorElementMatricesConjugateCellMassMatrixScaled[atomId][nonZeroElementMatrixId][d_kPointIndex][0],
					   &d_numberNodesPerElement,
					   &beta,
					   &projectorKetTimesVector[atomId][0],
					   &numberWaveFunctions);
				  }
				else
				  {

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


		}

	}//cell loop

	dftPtr->d_projectorKetTimesVectorParFlattened=std::complex<double>(0.0,0.0);


	for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
	{
		const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
		const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];

		for(unsigned int iPseudoAtomicWave = 0; iPseudoAtomicWave < numberPseudoWaveFunctions; ++iPseudoAtomicWave)
		{
			const unsigned int id=dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)];
			zcopy_(&numberWaveFunctions,
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
			std::complex<double> nonlocalConstantV;
			nonlocalConstantV.real(dftPtr->d_nonLocalPseudoPotentialConstants[atomId][iPseudoAtomicWave]);
			nonlocalConstantV.imag(0);

			const unsigned int id=dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(atomId,iPseudoAtomicWave)];

			zscal_(&numberWaveFunctions,
					&nonlocalConstantV,
					&dftPtr->d_projectorKetTimesVectorParFlattened[id*numberWaveFunctions],
					&inc);

			zcopy_(&numberWaveFunctions,
					&dftPtr->d_projectorKetTimesVectorParFlattened[id*numberWaveFunctions],
					&inc,
					&projectorKetTimesVector[atomId][numberWaveFunctions*iPseudoAtomicWave],
					&inc);
		}

	}


	std::vector<std::complex<double> > cellNonLocalHamTimesWaveMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);

	//blas required settings
	const char transA1 = 'N';
	const char transB1 = 'N';
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

		  if(dftParameters::cellLevelMassMatrixScaling)
		    {
		      zgemm_(&transA1,
			     &transB1,
			     &numberWaveFunctions,
			     &d_numberNodesPerElement,
			     &numberPseudoWaveFunctions,
			     &alpha1,
			     &projectorKetTimesVector[atomId][0],
			     &numberWaveFunctions,
			     &dftPtr->d_nonLocalProjectorElementMatricesTransposeCellMassMatrixScaled[atomId][iElemComp][d_kPointIndex][0],
			     &numberPseudoWaveFunctions,
			     &beta1,
			     &cellNonLocalHamTimesWaveMatrix[0],
			     &numberWaveFunctions);
		    }
		  else
		    {
		      zgemm_(&transA1,
			     &transB1,
			     &numberWaveFunctions,
			     &d_numberNodesPerElement,
			     &numberPseudoWaveFunctions,
			     &alpha1,
			     &projectorKetTimesVector[atomId][0],
			     &numberWaveFunctions,
			     &dftPtr->d_nonLocalProjectorElementMatricesTranspose[atomId][iElemComp][d_kPointIndex][0],
			     &numberPseudoWaveFunctions,
			     &beta1,
			     &cellNonLocalHamTimesWaveMatrix[0],
			     &numberWaveFunctions);
		    }

			unsigned int elementId =  dftPtr->d_elementIdsInAtomCompactSupport[atomId][iElemComp];

			for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
			{
				dealii::types::global_dof_index localNodeId = d_flattenedArrayCellLocalProcIndexIdMap[elementId][iNode];
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
template<unsigned int FEOrder,unsigned int FEOrderElectro>
void kohnShamDFTOperatorClass<FEOrder,FEOrderElectro>::computeNonLocalHamiltonianTimesX(const distributedCPUVec<double> & src,
		                                                                        const unsigned int numberWaveFunctions,
		                                                                        distributedCPUVec<double>       & dst,
                                                                                        const double scalar) const
{
	std::map<unsigned int, std::vector<double> > projectorKetTimesVector;

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
			if (dftPtr->d_nonLocalAtomIdsInElement[iElem].size()>0)
			{
			  for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
				{
					dealii::types::global_dof_index localNodeId = d_flattenedArrayCellLocalProcIndexIdMap[iElem][iNode];
					dcopy_(&numberWaveFunctions,
					       src.begin()+localNodeId,
				               &inc,
					       &cellWaveFunctionMatrix[numberWaveFunctions*iNode],
					       &inc);
	                         }
			}

			for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInElement[iElem].size();++iAtom)
			{
				const unsigned int atomId = dftPtr->d_nonLocalAtomIdsInElement[iElem][iAtom];
				const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
				const int nonZeroElementMatrixId = dftPtr->d_sparsityPattern[atomId][iElem];

				if(dftParameters::cellLevelMassMatrixScaling)
				  {
				    dgemm_(&transA,
					   &transB,
					   &numberWaveFunctions,
					   &numberPseudoWaveFunctions,
					   &d_numberNodesPerElement,
					   &alpha,
					   &cellWaveFunctionMatrix[0],
					   &numberWaveFunctions,
					   &dftPtr->d_nonLocalProjectorElementMatricesCellMassMatrixScaled[atomId][nonZeroElementMatrixId][0],
					   &d_numberNodesPerElement,
					   &beta,
					   &projectorKetTimesVector[atomId][0],
					   &numberWaveFunctions);
				  }
				else
				  {
				    dgemm_(&transA,
					   &transB,
					   &numberWaveFunctions,
					   &numberPseudoWaveFunctions,
					   &d_numberNodesPerElement,
					   &alpha,
					   &cellWaveFunctionMatrix[0],
					   &numberWaveFunctions,
					   &dftPtr->d_nonLocalProjectorElementMatrices[atomId][nonZeroElementMatrixId][0],
					   &d_numberNodesPerElement,
					   &beta,
					   &projectorKetTimesVector[atomId][0],
					   &numberWaveFunctions);
				  }
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


	std::vector<double> cellNonLocalHamTimesWaveMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);

	//blas required settings
	const char transA1 = 'N';
	const char transB1 = 'N';
	const double alpha1 = 1.0;
	const double beta1 = 0.0;
	const unsigned int inc1 = 1;
	const double alpha2 = scalar;

	//
	//compute C*V*C^{T}*x
	//
	for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
	{
		const unsigned int atomId = dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
		const unsigned int numberPseudoWaveFunctions = dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
		for(unsigned int iElemComp = 0; iElemComp < dftPtr->d_elementIteratorsInAtomCompactSupport[atomId].size(); ++iElemComp)
		{

		  if(dftParameters::cellLevelMassMatrixScaling)
		    {
		      dgemm_(&transA1,
			     &transB1,
			     &numberWaveFunctions,
			     &d_numberNodesPerElement,
			     &numberPseudoWaveFunctions,
			     &alpha2,
			     &projectorKetTimesVector[atomId][0],
			     &numberWaveFunctions,
			     &dftPtr->d_nonLocalProjectorElementMatricesTransposeCellMassMatrixScaled[atomId][iElemComp][0],
			     &numberPseudoWaveFunctions,
			     &beta1,
			     &cellNonLocalHamTimesWaveMatrix[0],
			     &numberWaveFunctions);
		    }
		  else
		    {
		       dgemm_(&transA1,
			     &transB1,
			     &numberWaveFunctions,
			     &d_numberNodesPerElement,
			     &numberPseudoWaveFunctions,
			     &alpha2,
			     &projectorKetTimesVector[atomId][0],
			     &numberWaveFunctions,
			     &dftPtr->d_nonLocalProjectorElementMatricesTranspose[atomId][iElemComp][0],
			     &numberPseudoWaveFunctions,
			     &beta1,
			     &cellNonLocalHamTimesWaveMatrix[0],
			     &numberWaveFunctions);
		    }

		  unsigned int elementId =  dftPtr->d_elementIdsInAtomCompactSupport[atomId][iElemComp];

		  for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
		    {
		      dealii::types::global_dof_index localNodeId = d_flattenedArrayCellLocalProcIndexIdMap[elementId][iNode];
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
