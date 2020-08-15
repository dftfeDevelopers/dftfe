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


/** @file matrixVectorProductImplementations.cc
 *  @brief Contains linear algebra operations
 *
 */

#ifdef USE_COMPLEX
template<unsigned int FEOrder>
void kohnShamDFTOperatorClass<FEOrder>::computeLocalHamiltonianTimesX(const distributedCPUVec<std::complex<double> > & src,
		const unsigned int numberWaveFunctions,
		distributedCPUVec<std::complex<double> > & dst) 
{
	const unsigned int kpointSpinIndex=(1+dftParameters::spinPolarized)*d_kPointIndex+d_spinIndex;
	//
	//element level matrix-vector multiplications
	//
	const char transA = 'N',transB = 'T';
	const std::complex<double> scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;
	const unsigned int inc = 1;

	std::vector<std::complex<double> > cellWaveFunctionMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);
	std::vector<std::complex<double> > cellHamMatrixTimesWaveMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);

	unsigned int iElem = 0;
	for(unsigned int iMacroCell = 0; iMacroCell < d_numberMacroCells; ++iMacroCell)
	{
		for(unsigned int iCell = 0; iCell < d_macroCellSubCellMap[iMacroCell]; ++iCell)
		{
			for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
			{
				dealii::types::global_dof_index localNodeId = d_flattenedArrayMacroCellLocalProcIndexIdMap[iElem][iNode];
				zcopy_(&numberWaveFunctions,
						src.begin()+localNodeId,
						&inc,
						&cellWaveFunctionMatrix[numberWaveFunctions*iNode],
						&inc);
			}

			zgemm_(&transA,
					&transB,
					&numberWaveFunctions,
					&d_numberNodesPerElement,
					&d_numberNodesPerElement,
					&scalarCoeffAlpha,
					&cellWaveFunctionMatrix[0],
					&numberWaveFunctions,
					&d_cellHamiltonianMatrix[kpointSpinIndex][iElem][0],
					&d_numberNodesPerElement,
					&scalarCoeffBeta,
					&cellHamMatrixTimesWaveMatrix[0],
					&numberWaveFunctions);

			for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
			{
				dealii::types::global_dof_index localNodeId = d_flattenedArrayMacroCellLocalProcIndexIdMap[iElem][iNode];
				zaxpy_(&numberWaveFunctions,
						&scalarCoeffAlpha,
						&cellHamMatrixTimesWaveMatrix[numberWaveFunctions*iNode],
						&inc,
						dst.begin()+localNodeId,
						&inc);
			}

			++iElem;
		}//subcell loop
	}//macrocell loop

}

#ifdef WITH_MKL
template<unsigned int FEOrder>
void kohnShamDFTOperatorClass<FEOrder>::computeLocalHamiltonianTimesXBatchGEMM (const distributedCPUVec<std::complex<double> > & src,
		const unsigned int numberWaveFunctions,
		distributedCPUVec<std::complex<double> > & dst) const

{
	const unsigned int kpointSpinIndex=(1+dftParameters::spinPolarized)*d_kPointIndex+d_spinIndex;
	//
	//element level matrix-vector multiplications
	//
	const char transA = 'N',transB = 'T';
	const std::complex<double> scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;
	const unsigned int inc = 1;

	const unsigned int groupCount=1;
	const unsigned int groupSize=VectorizedArray<double>::n_array_elements;

	std::complex<double> ** cellWaveFunctionMatrixBatch = new std::complex<double>*[groupSize];
	std::complex<double> ** cellHamMatrixTimesWaveMatrixBatch = new std::complex<double>*[groupSize];
	const std::complex<double> ** cellHamMatrixBatch = new std::complex<double>*[groupSize];
	for(unsigned int i = 0; i < groupSize; i++)
	{
		cellWaveFunctionMatrixBatch[i] = new std::complex<double>[d_numberNodesPerElement*numberWaveFunctions];
		cellHamMatrixTimesWaveMatrixBatch[i] = new std::complex<double>[d_numberNodesPerElement*numberWaveFunctions];
	}

	unsigned int iElem= 0;
	for(unsigned int iMacroCell = 0; iMacroCell < d_numberMacroCells; ++iMacroCell)
	{

		for(unsigned int isubcell = 0; isubcell < d_macroCellSubCellMap[iMacroCell]; isubcell++)
		{
			for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
			{
				dealii::types::global_dof_index localNodeId = d_flattenedArrayMacroCellLocalProcIndexIdMap[iElem+isubcell][iNode];
				zcopy_(&numberWaveFunctions,
						src.begin()+localNodeId,
						&inc,
						&cellWaveFunctionMatrixBatch[isubcell][numberWaveFunctions*iNode],
						&inc);
			}

			cellHamMatrixBatch[isubcell] =&d_cellHamiltonianMatrix[kpointSpinIndex][iElem+isubcell][0];
		}

		zgemm_batch_(&transA,
				&transB,
				&numberWaveFunctions,
				&d_numberNodesPerElement,
				&d_numberNodesPerElement,
				&scalarCoeffAlpha,
				cellWaveFunctionMatrixBatch,
				&numberWaveFunctions,
				cellHamMatrixBatch,
				&d_numberNodesPerElement,
				&scalarCoeffBeta,
				cellHamMatrixTimesWaveMatrixBatch,
				&numberWaveFunctions,
				&groupCount,
				&d_macroCellSubCellMap[iMacroCell]);

		for(unsigned int isubcell = 0; isubcell < d_macroCellSubCellMap[iMacroCell]; isubcell++)
			for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
			{
				dealii::types::global_dof_index localNodeId = d_flattenedArrayMacroCellLocalProcIndexIdMap[iElem+isubcell][iNode];
				zaxpy_(&numberWaveFunctions,
						&scalarCoeffAlpha,
						&cellHamMatrixTimesWaveMatrixBatch[isubcell][numberWaveFunctions*iNode],
						&inc,
						dst.begin()+localNodeId,
						&inc);
			}


		iElem+=d_macroCellSubCellMap[iMacroCell];
	}//macrocell loop

	for(unsigned int i = 0; i < groupSize; i++)
	{
		delete [] cellWaveFunctionMatrixBatch[i];
		delete [] cellHamMatrixTimesWaveMatrixBatch[i];
	}
	delete [] cellWaveFunctionMatrixBatch;
	delete []  cellHamMatrixTimesWaveMatrixBatch;
	delete []  cellHamMatrixBatch;
}

#endif
#else
template<unsigned int FEOrder>
void kohnShamDFTOperatorClass<FEOrder>::computeLocalHamiltonianTimesX(const distributedCPUVec<double> & src,
								      const unsigned int numberWaveFunctions,
								      distributedCPUVec<double> & dst,
								      const double scalar) 
{

	const unsigned int kpointSpinIndex=(1+dftParameters::spinPolarized)*d_kPointIndex+d_spinIndex;
	//
	//element level matrix-vector multiplications
	//
	const char transA = 'N',transB = 'N';
	const double scalarCoeffAlpha1 = scalar,scalarCoeffBeta = 0.0,scalarCoeffAlpha = 1.0;
	const unsigned int inc = 1;

	std::vector<double> cellWaveFunctionMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);
  	std::vector<double> cellHamMatrixTimesWaveMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);

	unsigned int iElem = 0;
	for(unsigned int iMacroCell = 0; iMacroCell < d_numberMacroCells; ++iMacroCell)
	{
		for(unsigned int iCell = 0; iCell < d_macroCellSubCellMap[iMacroCell]; ++iCell)
		{
		  
		  for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
		    {
		      dealii::types::global_dof_index localNodeId = d_flattenedArrayMacroCellLocalProcIndexIdMap[iElem][iNode];
		      dcopy_(&numberWaveFunctions,
			     src.begin()+localNodeId,
			     &inc,
			     &cellWaveFunctionMatrix[numberWaveFunctions*iNode],
			     &inc);
		    }

		 
		

		  dgemm_(&transA,
			 &transB,
			 &numberWaveFunctions,
			 &d_numberNodesPerElement,
			 &d_numberNodesPerElement,
			 &scalarCoeffAlpha1,
			 &cellWaveFunctionMatrix[0],
			 &numberWaveFunctions,
			 &d_cellHamiltonianMatrix[kpointSpinIndex][iElem][0],
			 &d_numberNodesPerElement,
			 &scalarCoeffBeta,
			 &cellHamMatrixTimesWaveMatrix[0],
			 &numberWaveFunctions);

		  for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
		    {
			  
		      dealii::types::global_dof_index localNodeId = d_flattenedArrayMacroCellLocalProcIndexIdMap[iElem][iNode];
		      daxpy_(&numberWaveFunctions,
			     &scalarCoeffAlpha,
			     &cellHamMatrixTimesWaveMatrix[numberWaveFunctions*iNode],
			     &inc,
			     dst.begin()+localNodeId,
			     &inc);
			    
		    }

			++iElem;
		}//subcell loop
	}//macrocell loop

}

template<unsigned int FEOrder>
void kohnShamDFTOperatorClass<FEOrder>::computeLocalHamiltonianTimesX(const distributedCPUVec<double> & src,
								      std::vector<std::vector<double> > & cellSrcWaveFunctionMatrix,
								      const unsigned int numberWaveFunctions,
								      distributedCPUVec<double> & dst,
								      std::vector<std::vector<double> > & cellDstWaveFunctionMatrix,
								      const double scalar)
								       
{

	const unsigned int kpointSpinIndex=(1+dftParameters::spinPolarized)*d_kPointIndex+d_spinIndex;
	//
	//element level matrix-vector multiplications
	//
	const char transA = 'N',transB = 'N';
	const double scalarCoeffAlpha1 = scalar,scalarCoeffBeta = 0.0,scalarCoeffAlpha = 1.0;
	const unsigned int inc = 1;

	//std::vector<double> cellWaveFunctionMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);
        //cellWaveFunctionMatrix = d_cellWaveFunctionMatrix;
	std::vector<double> cellHamMatrixTimesWaveMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);

	unsigned int iElem = 0;
	for(unsigned int iMacroCell = 0; iMacroCell < d_numberMacroCells; ++iMacroCell)
	  {
	    for(unsigned int iCell = 0; iCell < d_macroCellSubCellMap[iMacroCell]; ++iCell)
	      {

		for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
		  {
		    if(d_nodesPerCellClassificationMap[iNode] == 1)
		      {
			dealii::types::global_dof_index localNodeId = d_flattenedArrayMacroCellLocalProcIndexIdMap[iElem][iNode];

			dcopy_(&numberWaveFunctions,
			       src.begin()+localNodeId,
			       &inc,
			       &cellSrcWaveFunctionMatrix[iElem][numberWaveFunctions*iNode],
			       &inc);
		      }

		  }
		

		dgemm_(&transA,
		       &transB,
		       &numberWaveFunctions,
		       &d_numberNodesPerElement,
		       &d_numberNodesPerElement,
		       &scalarCoeffAlpha1,
		       &cellSrcWaveFunctionMatrix[iElem][0],
		       &numberWaveFunctions,
		       &d_cellHamiltonianMatrix[kpointSpinIndex][iElem][0],
		       &d_numberNodesPerElement,
		       &scalarCoeffBeta,
		       &cellHamMatrixTimesWaveMatrix[0],
		       &numberWaveFunctions);

		for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
		  {
		    if(d_nodesPerCellClassificationMap[iNode] == 1)
		      {
			dealii::types::global_dof_index localNodeId = d_flattenedArrayMacroCellLocalProcIndexIdMap[iElem][iNode];
			daxpy_(&numberWaveFunctions,
			       &scalarCoeffAlpha,
			       &cellHamMatrixTimesWaveMatrix[numberWaveFunctions*iNode],
			       &inc,
			       dst.begin()+localNodeId,
			       &inc);
		      }
		    else
		      {
			for(unsigned int iWave = 0; iWave < numberWaveFunctions; ++iWave)
			  {
			    cellDstWaveFunctionMatrix[iElem][numberWaveFunctions*iNode + iWave] += cellHamMatrixTimesWaveMatrix[numberWaveFunctions*iNode + iWave];
			  }
		      }
		  }

		++iElem;
	      }//subcell loop
	  }//macrocell loop

}



template<unsigned int FEOrder>
void kohnShamDFTOperatorClass<FEOrder>::computeMassMatrixTimesX(const distributedCPUVec<double> & src,
		const unsigned int numberWaveFunctions,
		distributedCPUVec<double> & dst) const
{
	const unsigned int kpointSpinIndex=(1+dftParameters::spinPolarized)*d_kPointIndex+d_spinIndex;
	//
	//element level matrix-vector multiplications
	//
	const char transA = 'N',transB = 'N';
	const double scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;
	const unsigned int inc = 1;

	std::vector<double> cellWaveFunctionMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);
	std::vector<double> cellMassMatrixTimesWaveMatrix(d_numberNodesPerElement*numberWaveFunctions,0.0);

	unsigned int iElem = 0;
	for(unsigned int iMacroCell = 0; iMacroCell < d_numberMacroCells; ++iMacroCell)
	{
		for(unsigned int iCell = 0; iCell < d_macroCellSubCellMap[iMacroCell]; ++iCell)
		{
			for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
			{
				dealii::types::global_dof_index localNodeId = d_flattenedArrayMacroCellLocalProcIndexIdMap[iElem][iNode];
				dcopy_(&numberWaveFunctions,
						src.begin()+localNodeId,
						&inc,
						&cellWaveFunctionMatrix[numberWaveFunctions*iNode],
						&inc);
			}

			dgemm_(&transA,
			       &transB,
			       &numberWaveFunctions,
			       &d_numberNodesPerElement,
			       &d_numberNodesPerElement,
			       &scalarCoeffAlpha,
			       &cellWaveFunctionMatrix[0],
			       &numberWaveFunctions,
			       &d_cellMassMatrix[iElem][0],
			       &d_numberNodesPerElement,
			       &scalarCoeffBeta,
			       &cellMassMatrixTimesWaveMatrix[0],
			       &numberWaveFunctions);

			for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
			{
				dealii::types::global_dof_index localNodeId = d_flattenedArrayMacroCellLocalProcIndexIdMap[iElem][iNode];
				daxpy_(&numberWaveFunctions,
						&scalarCoeffAlpha,
						&cellMassMatrixTimesWaveMatrix[numberWaveFunctions*iNode],
						&inc,
						dst.begin()+localNodeId,
						&inc);
			}

			++iElem;
		}//subcell loop
	}//macrocell loop

}

#ifdef WITH_MKL
template<unsigned int FEOrder>
void kohnShamDFTOperatorClass<FEOrder>::computeLocalHamiltonianTimesXBatchGEMM (const distributedCPUVec<double> & src,
										const unsigned int numberWaveFunctions,
										distributedCPUVec<double> & dst,
										const double scalar) const
{
	const unsigned int kpointSpinIndex=(1+dftParameters::spinPolarized)*d_kPointIndex+d_spinIndex;
	//
	//element level matrix-vector multiplications
	//
	const char transA = 'N',transB = 'N';
	const double scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0,scalarCoeffAlpha1 = scalar;
	const unsigned int inc = 1;

	const unsigned int groupCount=1;
	const unsigned int groupSize=VectorizedArray<double>::n_array_elements;

	double ** cellWaveFunctionMatrixBatch = new double*[groupSize];
	double ** cellHamMatrixTimesWaveMatrixBatch = new double*[groupSize];
	const double ** cellHamMatrixBatch = new double*[groupSize];
	for(unsigned int i = 0; i < groupSize; i++)
	{
		cellWaveFunctionMatrixBatch[i] = new double[d_numberNodesPerElement*numberWaveFunctions];
		cellHamMatrixTimesWaveMatrixBatch[i] = new double[d_numberNodesPerElement*numberWaveFunctions];
	}

	unsigned int iElem= 0;
	for(unsigned int iMacroCell = 0; iMacroCell < d_numberMacroCells; ++iMacroCell)
	{

		for(unsigned int isubcell = 0; isubcell < d_macroCellSubCellMap[iMacroCell]; isubcell++)
		{
			for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
			{
				dealii::types::global_dof_index localNodeId = d_flattenedArrayMacroCellLocalProcIndexIdMap[iElem+isubcell][iNode];
				dcopy_(&numberWaveFunctions,
						src.begin()+localNodeId,
						&inc,
						&cellWaveFunctionMatrixBatch[isubcell][numberWaveFunctions*iNode],
						&inc);
			}

			cellHamMatrixBatch[isubcell] =&d_cellHamiltonianMatrix[kpointSpinIndex][iElem+isubcell][0];
		}

		dgemm_batch_(&transA,
				&transB,
				&numberWaveFunctions,
				&d_numberNodesPerElement,
				&d_numberNodesPerElement,
				&scalarCoeffAlpha1,
				cellWaveFunctionMatrixBatch,
				&numberWaveFunctions,
				cellHamMatrixBatch,
				&d_numberNodesPerElement,
				&scalarCoeffBeta,
				cellHamMatrixTimesWaveMatrixBatch,
				&numberWaveFunctions,
				&groupCount,
				&d_macroCellSubCellMap[iMacroCell]);

		for(unsigned int isubcell = 0; isubcell < d_macroCellSubCellMap[iMacroCell]; isubcell++)
			for(unsigned int iNode = 0; iNode < d_numberNodesPerElement; ++iNode)
			{
				dealii::types::global_dof_index localNodeId = d_flattenedArrayMacroCellLocalProcIndexIdMap[iElem+isubcell][iNode];
				daxpy_(&numberWaveFunctions,
						&scalarCoeffAlpha,
						&cellHamMatrixTimesWaveMatrixBatch[isubcell][numberWaveFunctions*iNode],
						&inc,
						dst.begin()+localNodeId,
						&inc);
			}


		iElem+=d_macroCellSubCellMap[iMacroCell];
	}//macrocell loop

	for(unsigned int i = 0; i < groupSize; i++)
	{
		delete [] cellWaveFunctionMatrixBatch[i];
		delete [] cellHamMatrixTimesWaveMatrixBatch[i];
	}
	delete [] cellWaveFunctionMatrixBatch;
	delete []  cellHamMatrixTimesWaveMatrixBatch;
	delete []  cellHamMatrixBatch;
}
#endif
#endif
