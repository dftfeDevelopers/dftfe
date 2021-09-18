// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
// authors.
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

// skip1 and skip2 are flags used by chebyshevFilter function to perform overlap
// of computation and communication. When either skip1 or skip2 flags are set to
// true all communication calls are skipped as they are directly called in
// chebyshevFilter Only one of the skip flags is set to true in a call. When
// skip1 is set to true extraction and C^{T}*X computation are skipped and
// computations directly start from V*C^{T}*X. When skip2 is set to true only
// extraction and C^{T}*X computations are performed.
template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
  computeNonLocalHamiltonianTimesX(
    const dataTypes::numberGPU *src,
    distributedGPUVec<double> & projectorKetTimesVector,
    const unsigned int          numberWaveFunctions,
    dataTypes::numberGPU *      dst,
    const bool                  skip1,
    const bool                  skip2)
{
  const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0),
                          scalarCoeffBeta  = dataTypes::number(0.0);

  //
  // compute C^{\dagger}*X
  //
  unsigned int strideA = numberWaveFunctions * d_numberNodesPerElement;
  unsigned int strideB = d_numberNodesPerElement * d_maxSingleAtomPseudoWfc;
  unsigned int strideC = numberWaveFunctions * d_maxSingleAtomPseudoWfc;

  if (d_totalNonlocalElems > 0 && !skip1)
    {
      cublasXgemmBatched(
        d_cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        numberWaveFunctions,
        d_maxSingleAtomPseudoWfc,
        d_numberNodesPerElement,
        reinterpret_cast<const dataTypes::numberGPU *>(&scalarCoeffAlpha),
        (const dataTypes::numberGPU **)d_A,
        numberWaveFunctions,
        (const dataTypes::numberGPU **)d_B,
        d_numberNodesPerElement,
        reinterpret_cast<const dataTypes::numberGPU *>(&scalarCoeffBeta),
        d_C,
        numberWaveFunctions,
        d_totalNonlocalElems);

      cublasXgemm(
        d_cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        numberWaveFunctions,
        d_totalPseudoWfcNonLocal,
        d_totalNonlocalElems * d_maxSingleAtomPseudoWfc,
        reinterpret_cast<const dataTypes::numberGPU *>(&scalarCoeffAlpha),
        reinterpret_cast<const dataTypes::numberGPU *>(thrust::raw_pointer_cast(
          &d_projectorKetTimesVectorAllCellsDevice[0])),
        numberWaveFunctions,
        reinterpret_cast<const dataTypes::numberGPU *>(thrust::raw_pointer_cast(
          &d_projectorKetTimesVectorAllCellsReductionDevice[0])),
        d_totalNonlocalElems * d_maxSingleAtomPseudoWfc,
        reinterpret_cast<const dataTypes::numberGPU *>(&scalarCoeffBeta),
        reinterpret_cast<dataTypes::numberGPU *>(thrust::raw_pointer_cast(
          &d_projectorKetTimesVectorParFlattenedDevice[0])),
        numberWaveFunctions);
    }

  // this routine was interfering with overlapping communication and compute. So
  // called separately inside chebyshevFilter. So skip this if either skip1 or
  // skip2 are set to true
  if (!skip1 && !skip2)
    projectorKetTimesVector.setZero();


  if (d_totalNonlocalElems > 0 && !skip1)
    copyToDealiiParallelNonLocalVec<dataTypes::numberGPU>
      <<<(numberWaveFunctions + 255) / 256 * d_totalPseudoWfcNonLocal, 256>>>(
        numberWaveFunctions,
        d_totalPseudoWfcNonLocal,
        reinterpret_cast<dataTypes::numberGPU *>(thrust::raw_pointer_cast(
          &d_projectorKetTimesVectorParFlattenedDevice[0])),
        projectorKetTimesVector.begin(),
        thrust::raw_pointer_cast(&d_projectorIdsParallelNumberingMapDevice[0]));

  // Operations related to skip2 (extraction and C^{T}*X) are over. So return
  // control back to chebyshevFilter
  if (skip2)
    return;

  if (!skip1)
    {
      projectorKetTimesVector.compressAdd();
      projectorKetTimesVector.updateGhostValues();
    }

  //
  // Start operations related to skip1 (V*C^{\dagger}*X, C*V*C^{\dagger}*X and
  // assembly)
  //
  if (d_totalNonlocalElems > 0)
    {
      //
      // compute V*C^{\dagger}*X
      //
      scaleCUDAKernel<dataTypes::numberGPU>
        <<<(numberWaveFunctions + 255) / 256 * d_totalPseudoWfcNonLocal, 256>>>(
          numberWaveFunctions,
          d_totalPseudoWfcNonLocal,
          1.0,
          projectorKetTimesVector.begin(),
          thrust::raw_pointer_cast(
            &d_nonLocalPseudoPotentialConstantsDevice[0]));

      copyFromParallelNonLocalVecToAllCellsVec<dataTypes::numberGPU>
        <<<(numberWaveFunctions + 255) / 256 * d_totalNonlocalElems *
             d_maxSingleAtomPseudoWfc,
           256>>>(
          numberWaveFunctions,
          d_totalNonlocalElems,
          d_maxSingleAtomPseudoWfc,
          projectorKetTimesVector.begin(),
          reinterpret_cast<dataTypes::numberGPU *>(thrust::raw_pointer_cast(
            &d_projectorKetTimesVectorAllCellsDevice[0])),
          thrust::raw_pointer_cast(
            &d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice[0]));

      //
      // compute C*V*C^{\dagger}*x
      //

      strideA = numberWaveFunctions * d_maxSingleAtomPseudoWfc;
      strideB = d_maxSingleAtomPseudoWfc * d_numberNodesPerElement;
      strideC = numberWaveFunctions * d_numberNodesPerElement;
      cublasXgemmStridedBatched(
        d_cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        numberWaveFunctions,
        d_numberNodesPerElement,
        d_maxSingleAtomPseudoWfc,
        reinterpret_cast<const dataTypes::numberGPU *>(&scalarCoeffAlpha),
        reinterpret_cast<const dataTypes::numberGPU *>(thrust::raw_pointer_cast(
          &d_projectorKetTimesVectorAllCellsDevice[0])),
        numberWaveFunctions,
        strideA,
        reinterpret_cast<dataTypes::numberGPU *>(thrust::raw_pointer_cast(
          &d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice[0])),
        d_maxSingleAtomPseudoWfc,
        strideB,
        reinterpret_cast<const dataTypes::numberGPU *>(&scalarCoeffBeta),
        reinterpret_cast<dataTypes::numberGPU *>(thrust::raw_pointer_cast(
          &d_cellHamMatrixTimesWaveMatrixNonLocalDevice[0])),
        numberWaveFunctions,
        strideC,
        d_totalNonlocalElems);


      for (unsigned int iAtom = 0; iAtom < d_totalNonlocalAtomsCurrentProc;
           ++iAtom)
        {
          const unsigned int accum = d_numberCellsAccumNonLocalAtoms[iAtom];
          addNonLocalContributionCUDAKernel<dataTypes::numberGPU>
            <<<(numberWaveFunctions + 255) / 256 *
                 d_numberCellsNonLocalAtoms[iAtom] * d_numberNodesPerElement,
               256>>>(
              numberWaveFunctions,
              d_numberCellsNonLocalAtoms[iAtom] * d_numberNodesPerElement,
              reinterpret_cast<const dataTypes::numberGPU *>(
                thrust::raw_pointer_cast(
                  &d_cellHamMatrixTimesWaveMatrixNonLocalDevice[0])) +
                accum * d_numberNodesPerElement * numberWaveFunctions,
              reinterpret_cast<dataTypes::numberGPU *>(
                thrust::raw_pointer_cast(&d_cellHamMatrixTimesWaveMatrix[0])),
              thrust::raw_pointer_cast(
                &d_cellNodeIdMapNonLocalToLocalDevice[0]) +
                accum * d_numberNodesPerElement);
        }
    }

  daxpyAtomicAddKernel<dataTypes::numberGPU>
    <<<(numberWaveFunctions + 255) / 256 * d_numLocallyOwnedCells *
         d_numberNodesPerElement,
       256>>>(numberWaveFunctions,
              d_numLocallyOwnedCells * d_numberNodesPerElement,
              reinterpret_cast<dataTypes::numberGPU *>(
                thrust::raw_pointer_cast(&d_cellHamMatrixTimesWaveMatrix[0])),
              dst,
              thrust::raw_pointer_cast(
                &d_flattenedArrayCellLocalProcIndexIdMapDevice[0]));
}


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::
  computeNonLocalProjectorKetTimesXTimesV(
    const dataTypes::numberGPU *             src,
    distributedGPUVec<dataTypes::numberGPU> &projectorKetTimesVector,
    const unsigned int                       numberWaveFunctions)
{
  const unsigned int totalLocallyOwnedCells =
    dftPtr->matrix_free_data.n_physical_cells();
  const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0),
                          scalarCoeffBeta  = dataTypes::number(0.0);

  //
  // compute C^{\dagger}*X
  //

  if (d_totalNonlocalElems > 0)
    {
      copyCUDAKernel<dataTypes::numberGPU>
        <<<(numberWaveFunctions + 255) / 256 * totalLocallyOwnedCells *
             d_numberNodesPerElement,
           256>>>(numberWaveFunctions,
                  totalLocallyOwnedCells * d_numberNodesPerElement,
                  src,
                  reinterpret_cast<dataTypes::numberGPU *>(
                    thrust::raw_pointer_cast(&d_cellWaveFunctionMatrix[0])),
                  thrust::raw_pointer_cast(
                    &d_flattenedArrayCellLocalProcIndexIdMapDevice[0]));



      cublasXgemmBatched(
        d_cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        numberWaveFunctions,
        d_maxSingleAtomPseudoWfc,
        d_numberNodesPerElement,
        reinterpret_cast<const dataTypes::numberGPU *>(&scalarCoeffAlpha),
        (const double **)d_A,
        numberWaveFunctions,
        (const double **)d_B,
        d_numberNodesPerElement,
        reinterpret_cast<const dataTypes::numberGPU *>(&scalarCoeffBeta),
        d_C,
        numberWaveFunctions,
        d_totalNonlocalElems);

      cublasXgemm(
        d_cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        numberWaveFunctions,
        d_totalPseudoWfcNonLocal,
        d_totalNonlocalElems * d_maxSingleAtomPseudoWfc,
        reinterpret_cast<const dataTypes::numberGPU *>(&scalarCoeffAlpha),
        reinterpret_cast<const dataTypes::numberGPU *>(thrust::raw_pointer_cast(
          &d_projectorKetTimesVectorAllCellsDevice[0])),
        numberWaveFunctions,
        reinterpret_cast<const dataTypes::numberGPU *>(thrust::raw_pointer_cast(
          &d_projectorKetTimesVectorAllCellsReductionDevice[0])),
        d_totalNonlocalElems * d_maxSingleAtomPseudoWfc,
        reinterpret_cast<const dataTypes::numberGPU *>(&scalarCoeffBeta),
        reinterpret_cast<dataTypes::numberGPU *>(thrust::raw_pointer_cast(
          &d_projectorKetTimesVectorParFlattenedDevice[0])),
        numberWaveFunctions);
    }

  projectorKetTimesVector.setZero();


  if (d_totalNonlocalElems > 0)
    copyToDealiiParallelNonLocalVec<dataTypes::numberGPU>
      <<<(numberWaveFunctions + 255) / 256 * d_totalPseudoWfcNonLocal, 256>>>(
        numberWaveFunctions,
        d_totalPseudoWfcNonLocal,
        reinterpret_cast<dataTypes::numberGPU *>(thrust::raw_pointer_cast(
          &d_projectorKetTimesVectorParFlattenedDevice[0])),
        projectorKetTimesVector.begin(),
        thrust::raw_pointer_cast(&d_projectorIdsParallelNumberingMapDevice[0]));

  projectorKetTimesVector.compressAdd();
  projectorKetTimesVector.updateGhostValues();

  //
  // compute V*C^{\dagger}*X
  //
  if (d_totalNonlocalElems > 0)
    scaleCUDAKernel<dataTypes::numberGPU>
      <<<(numberWaveFunctions + 255) / 256 * d_totalPseudoWfcNonLocal, 256>>>(
        numberWaveFunctions,
        d_totalPseudoWfcNonLocal,
        1.0,
        projectorKetTimesVector.begin(),
        thrust::raw_pointer_cast(&d_nonLocalPseudoPotentialConstantsDevice[0]));
}
