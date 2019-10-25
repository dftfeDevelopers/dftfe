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
void kohnShamDFTOperatorCUDAClass<FEOrder>::computeNonLocalHamiltonianTimesX(const double* src,
									     cudaVectorType &  projectorKetTimesVector,
									     const unsigned int numberWaveFunctions,
									     double* dst,
const bool skip1,
const bool skip2)

{
	 
  const double scalarCoeffAlpha = 1.0,scalarCoeffBeta = 0.0;
  
  //
  //compute C^{T}*X
  //
  unsigned int strideA = numberWaveFunctions*d_numberNodesPerElement;
  unsigned int strideB = d_numberNodesPerElement*d_maxSingleAtomPseudoWfc; 
  unsigned int strideC = numberWaveFunctions*d_maxSingleAtomPseudoWfc;

  if (d_totalNonlocalElems>0 && !skip1)
  { 
	  copyCUDAKernel<<<(numberWaveFunctions+255)/256*d_totalNonlocalElems*d_numberNodesPerElement,256>>>
									     (numberWaveFunctions, 
									      d_totalNonlocalElems*d_numberNodesPerElement,
									      src,
									      thrust::raw_pointer_cast(&d_cellWaveFunctionMatrixNonLocalDevice[0]),
									      thrust::raw_pointer_cast(&d_flattenedArrayCellLocalProcIndexIdFlattenedMapNonLocalDevice[0]));
	 
	 
	  
	  cublasDgemmStridedBatched(d_cublasHandle,
				    CUBLAS_OP_N,
				    CUBLAS_OP_N,
				    numberWaveFunctions,
				    d_maxSingleAtomPseudoWfc,
				    d_numberNodesPerElement,
				    &scalarCoeffAlpha,
				    thrust::raw_pointer_cast(&d_cellWaveFunctionMatrixNonLocalDevice[0]),
				    numberWaveFunctions,
				    strideA,
				    thrust::raw_pointer_cast(&d_cellHamiltonianMatrixNonLocalFlattenedDevice[0]),
				    d_numberNodesPerElement,
				    strideB,
				    &scalarCoeffBeta,
				    thrust::raw_pointer_cast(&d_projectorKetTimesVectorAllCellsDevice[0]),
				    numberWaveFunctions,
				    strideC,
				    d_totalNonlocalElems);

	  cublasDgemm(d_cublasHandle,
		      CUBLAS_OP_N,
		      CUBLAS_OP_N,
		      numberWaveFunctions,
		      d_totalPseudoWfcNonLocal,
		      d_totalNonlocalElems*d_maxSingleAtomPseudoWfc,
		      &scalarCoeffAlpha,
		      thrust::raw_pointer_cast(&d_projectorKetTimesVectorAllCellsDevice[0]),
		      numberWaveFunctions,
		      thrust::raw_pointer_cast(&d_projectorKetTimesVectorAllCellsReductionDevice[0]),
		      d_totalNonlocalElems*d_maxSingleAtomPseudoWfc,
		      &scalarCoeffBeta,
		      thrust::raw_pointer_cast(&d_projectorKetTimesVectorParFlattenedDevice[0]),
		      numberWaveFunctions);

  }

  if (skip2)
   return;

  projectorKetTimesVector=0.0;

  //std::cout<<"nonlocal l2 norm: "<<d_projectorKetTimesVectorDealiiParFlattenedDevice.l2_norm()<<std::endl;
  
  if (d_totalNonlocalElems>0)
    copyToDealiiParallelNonLocalVec<<<(numberWaveFunctions+255)/256*d_totalPseudoWfcNonLocal,256>>>
						     (numberWaveFunctions, 
						      d_totalPseudoWfcNonLocal,
                                                      thrust::raw_pointer_cast(&d_projectorKetTimesVectorParFlattenedDevice[0]),
                                                      projectorKetTimesVector.begin(),
						      thrust::raw_pointer_cast(&d_projectorIdsParallelNumberingMapDevice[0]));

  
    projectorKetTimesVector.compress(VectorOperation::add);
    projectorKetTimesVector.update_ghost_values();
  
  //std::cout<<"nonlocal l2 norm: "<<projectorKetTimesVector.l2_norm()<<std::endl;

  if (d_totalNonlocalElems>0) 
  {
	  //
	  //compute V*C^{T}*X
	  //
	  scaleCUDAKernel<<<(numberWaveFunctions+255)/256*d_totalPseudoWfcNonLocal,256>>>(numberWaveFunctions,
										 d_totalPseudoWfcNonLocal,
										 1.0,
										 projectorKetTimesVector.begin(),
										 thrust::raw_pointer_cast(&d_nonLocalPseudoPotentialConstantsDevice[0]));

	  copyFromParallelNonLocalVecToAllCellsVec<<<(numberWaveFunctions+255)/256*d_totalNonlocalElems*d_maxSingleAtomPseudoWfc,256>>>
							     (numberWaveFunctions, 
							      d_totalNonlocalElems,
							      d_maxSingleAtomPseudoWfc,
							      projectorKetTimesVector.begin(),
							      thrust::raw_pointer_cast(&d_projectorKetTimesVectorAllCellsDevice[0]),
							      thrust::raw_pointer_cast(&d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice[0]));
	   
	  //
	  //compute C*V*C^{T}*x
	  //

	  strideA = numberWaveFunctions*d_maxSingleAtomPseudoWfc;
	  strideB = d_maxSingleAtomPseudoWfc*d_numberNodesPerElement; 
	  strideC = numberWaveFunctions*d_numberNodesPerElement;
	  cublasDgemmStridedBatched(d_cublasHandle,
				    CUBLAS_OP_N,
				    CUBLAS_OP_N,
				    numberWaveFunctions,
				    d_numberNodesPerElement,
				    d_maxSingleAtomPseudoWfc,
				    &scalarCoeffAlpha,
				    thrust::raw_pointer_cast(&d_projectorKetTimesVectorAllCellsDevice[0]),
				    numberWaveFunctions,
				    strideA,
				    thrust::raw_pointer_cast(&d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice[0]),
				    d_maxSingleAtomPseudoWfc,
				    strideB,
				    &scalarCoeffBeta,
				    thrust::raw_pointer_cast(&d_cellHamMatrixTimesWaveMatrixNonLocalDevice[0]),
				    numberWaveFunctions,
				    strideC,
				    d_totalNonlocalElems);
	 
	   
	  for(unsigned int iAtom = 0; iAtom < d_totalNonlocalAtomsCurrentProc; ++iAtom)
	  {
		  const unsigned int accum= d_numberCellsAccumNonLocalAtoms[iAtom];
		  addNonLocalContributionCUDAKernel<<<(numberWaveFunctions+255)/256*d_numberCellsNonLocalAtoms[iAtom]*d_numberNodesPerElement,256>>>
										     (numberWaveFunctions, 
										      d_numberCellsNonLocalAtoms[iAtom]*d_numberNodesPerElement,
										      thrust::raw_pointer_cast(&d_cellHamMatrixTimesWaveMatrixNonLocalDevice[0])
										+accum*d_numberNodesPerElement*numberWaveFunctions,
										      thrust::raw_pointer_cast(&d_cellHamMatrixTimesWaveMatrix[0]),
										      thrust::raw_pointer_cast(&d_cellNodeIdMapNonLocalToLocalDevice[0])
										  +accum*d_numberNodesPerElement);
		  
	  }
	  
  }   

  daxpyAtomicAddKernel<<<(numberWaveFunctions+255)/256*d_numLocallyOwnedCells*d_numberNodesPerElement,256>>>
                                                                     (numberWaveFunctions,
                                                                      d_numLocallyOwnedCells*d_numberNodesPerElement,
                                                                      thrust::raw_pointer_cast(&d_cellHamMatrixTimesWaveMatrix[0]),
                                                                      dst,
                                                                      thrust::raw_pointer_cast(&d_flattenedArrayCellLocalProcIndexIdMapDevice[0]));

   
   //std::cout<<"dst norm: "<<dst.l2_norm()<<std::endl;
}
