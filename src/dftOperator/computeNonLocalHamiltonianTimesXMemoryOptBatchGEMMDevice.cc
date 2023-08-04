// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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
kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>::
  computeNonLocalHamiltonianTimesX(
    const dataTypes::number *                src,
    distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
    const unsigned int                       numberWaveFunctions,
    dataTypes::number *                      dst,
    const bool                               skip1,
    const bool                               skip2)
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
      dftfe::utils::deviceBlasWrapper::gemmBatched(
        d_deviceBlasHandle,
        dftfe::utils::DEVICEBLAS_OP_N,
        dftfe::utils::DEVICEBLAS_OP_N,
        numberWaveFunctions,
        d_maxSingleAtomPseudoWfc,
        d_numberNodesPerElement,
        &scalarCoeffAlpha,
        (const dataTypes::number **)d_A,
        numberWaveFunctions,
        (const dataTypes::number **)d_B,
        d_numberNodesPerElement,
        &scalarCoeffBeta,
        d_C,
        numberWaveFunctions,
        d_totalNonlocalElems);

      dftfe::utils::deviceBlasWrapper::gemm(
        d_deviceBlasHandle,
        dftfe::utils::DEVICEBLAS_OP_N,
        dftfe::utils::DEVICEBLAS_OP_N,
        numberWaveFunctions,
        d_totalPseudoWfcNonLocal,
        d_totalNonlocalElems * d_maxSingleAtomPseudoWfc,
        &scalarCoeffAlpha,
        d_projectorKetTimesVectorAllCellsDevice.begin(),
        numberWaveFunctions,
        d_projectorKetTimesVectorAllCellsReductionDevice.begin(),
        d_totalNonlocalElems * d_maxSingleAtomPseudoWfc,
        &scalarCoeffBeta,
        d_projectorKetTimesVectorParFlattenedDevice.begin(),
        numberWaveFunctions);
    }

  // this routine was interfering with overlapping communication and compute. So
  // called separately inside chebyshevFilter. So skip this if either skip1 or
  // skip2 are set to true
  if (!skip1 && !skip2)
    projectorKetTimesVector.setValue(0);


  if (d_totalNonlocalElems > 0 && !skip1)
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    copyToDealiiParallelNonLocalVec<<<
      (numberWaveFunctions + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
        dftfe::utils::DEVICE_BLOCK_SIZE * d_totalPseudoWfcNonLocal,
      dftfe::utils::DEVICE_BLOCK_SIZE>>>(
      numberWaveFunctions,
      d_totalPseudoWfcNonLocal,
      dftfe::utils::makeDataTypeDeviceCompatible(
        d_projectorKetTimesVectorParFlattenedDevice.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(
        projectorKetTimesVector.begin()),
      d_projectorIdsParallelNumberingMapDevice.begin());
#elif DFTFE_WITH_DEVICE_LANG_HIP
    hipLaunchKernelGGL(copyToDealiiParallelNonLocalVec,
                       (numberWaveFunctions +
                        (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                         dftfe::utils::DEVICE_BLOCK_SIZE *
                         d_totalPseudoWfcNonLocal,
                       dftfe::utils::DEVICE_BLOCK_SIZE,
                       0,
                       0,
                       numberWaveFunctions,
                       d_totalPseudoWfcNonLocal,
                       dftfe::utils::makeDataTypeDeviceCompatible(
                         d_projectorKetTimesVectorParFlattenedDevice.begin()),
                       dftfe::utils::makeDataTypeDeviceCompatible(
                         projectorKetTimesVector.begin()),
                       d_projectorIdsParallelNumberingMapDevice.begin());
#endif

  // Operations related to skip2 (extraction and C^{T}*X) are over. So return
  // control back to chebyshevFilter
  if (skip2)
    return;

  if (!skip1)
    {
      projectorKetTimesVector.accumulateAddLocallyOwned(1);
      projectorKetTimesVector.updateGhostValues(1);
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
      dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
        numberWaveFunctions,
        d_totalPseudoWfcNonLocal,
        1.0,
        d_nonLocalPseudoPotentialConstantsDevice.begin(),
        projectorKetTimesVector.begin());

#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      copyFromParallelNonLocalVecToAllCellsVec<<<
        (numberWaveFunctions + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE * d_totalNonlocalElems *
          d_maxSingleAtomPseudoWfc,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        numberWaveFunctions,
        d_totalNonlocalElems,
        d_maxSingleAtomPseudoWfc,
        dftfe::utils::makeDataTypeDeviceCompatible(
          projectorKetTimesVector.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          d_projectorKetTimesVectorAllCellsDevice.begin()),
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice.begin());
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        copyFromParallelNonLocalVecToAllCellsVec,
        (numberWaveFunctions + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE * d_totalNonlocalElems *
          d_maxSingleAtomPseudoWfc,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        numberWaveFunctions,
        d_totalNonlocalElems,
        d_maxSingleAtomPseudoWfc,
        dftfe::utils::makeDataTypeDeviceCompatible(
          projectorKetTimesVector.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(
          d_projectorKetTimesVectorAllCellsDevice.begin()),
        d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice.begin());
#endif

      //
      // compute C*V*C^{\dagger}*x
      //

      strideA = numberWaveFunctions * d_maxSingleAtomPseudoWfc;
      strideB = d_maxSingleAtomPseudoWfc * d_numberNodesPerElement;
      strideC = numberWaveFunctions * d_numberNodesPerElement;
      dftfe::utils::deviceBlasWrapper::gemmStridedBatched(
        d_deviceBlasHandle,
        dftfe::utils::DEVICEBLAS_OP_N,
        dftfe::utils::DEVICEBLAS_OP_N,
        numberWaveFunctions,
        d_numberNodesPerElement,
        d_maxSingleAtomPseudoWfc,
        &scalarCoeffAlpha,
        d_projectorKetTimesVectorAllCellsDevice.begin(),
        numberWaveFunctions,
        strideA,
        d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice.begin() +
          d_kPointIndex * d_totalNonlocalElems * d_maxSingleAtomPseudoWfc *
            d_numberNodesPerElement,
        d_maxSingleAtomPseudoWfc,
        strideB,
        &scalarCoeffBeta,
        d_cellHamMatrixTimesWaveMatrixNonLocalDevice.begin(),
        numberWaveFunctions,
        strideC,
        d_totalNonlocalElems);


      for (unsigned int iAtom = 0; iAtom < d_totalNonlocalAtomsCurrentProc;
           ++iAtom)
        {
          const unsigned int accum = d_numberCellsAccumNonLocalAtoms[iAtom];
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
          addNonLocalContributionDeviceKernel<<<
            (numberWaveFunctions + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE *
              d_numberCellsNonLocalAtoms[iAtom] * d_numberNodesPerElement,
            dftfe::utils::DEVICE_BLOCK_SIZE>>>(
            numberWaveFunctions,
            d_numberCellsNonLocalAtoms[iAtom] * d_numberNodesPerElement,
            dftfe::utils::makeDataTypeDeviceCompatible(
              d_cellHamMatrixTimesWaveMatrixNonLocalDevice.begin() +
              accum * d_numberNodesPerElement * numberWaveFunctions),
            dftfe::utils::makeDataTypeDeviceCompatible(
              d_cellHamMatrixTimesWaveMatrix.begin()),
            d_cellNodeIdMapNonLocalToLocalDevice.begin() +
              accum * d_numberNodesPerElement);
#elif DFTFE_WITH_DEVICE_LANG_HIP
          hipLaunchKernelGGL(
            addNonLocalContributionDeviceKernel,
            (numberWaveFunctions + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE *
              d_numberCellsNonLocalAtoms[iAtom] * d_numberNodesPerElement,
            dftfe::utils::DEVICE_BLOCK_SIZE,
            0,
            0,
            numberWaveFunctions,
            d_numberCellsNonLocalAtoms[iAtom] * d_numberNodesPerElement,
            dftfe::utils::makeDataTypeDeviceCompatible(
              d_cellHamMatrixTimesWaveMatrixNonLocalDevice.begin() +
              accum * d_numberNodesPerElement * numberWaveFunctions),
            dftfe::utils::makeDataTypeDeviceCompatible(
              d_cellHamMatrixTimesWaveMatrix.begin()),
            d_cellNodeIdMapNonLocalToLocalDevice.begin() +
              accum * d_numberNodesPerElement);
#endif
        }
    }

  if (std::is_same<dataTypes::number, std::complex<double>>::value)
    {
      utils::deviceKernelsGeneric::copyComplexArrToRealArrsDevice(
        (d_parallelChebyBlockVectorDevice.localSize() *
         d_parallelChebyBlockVectorDevice.numVectors()),
        dst,
        d_tempRealVec.begin(),
        d_tempImagVec.begin());

      dftfe::utils::deviceKernelsGeneric::axpyStridedBlockAtomicAdd(
        numberWaveFunctions,
        d_numLocallyOwnedCells * d_numberNodesPerElement,
        d_cellHamMatrixTimesWaveMatrix.begin(),
        d_tempRealVec.begin(),
        d_tempImagVec.begin(),
        d_flattenedArrayCellLocalProcIndexIdMapDevice.begin());


      utils::deviceKernelsGeneric::copyRealArrsToComplexArrDevice(
        (d_parallelChebyBlockVectorDevice.localSize() *
         d_parallelChebyBlockVectorDevice.numVectors()),
        d_tempRealVec.begin(),
        d_tempImagVec.begin(),
        dst);
    }
  else
    dftfe::utils::deviceKernelsGeneric::axpyStridedBlockAtomicAdd(
      numberWaveFunctions,
      d_numLocallyOwnedCells * d_numberNodesPerElement,
      d_cellHamMatrixTimesWaveMatrix.begin(),
      dst,
      d_flattenedArrayCellLocalProcIndexIdMapDevice.begin());
}


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>::
  computeNonLocalProjectorKetTimesXTimesV(
    const dataTypes::number *                src,
    distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
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
      dftfe::utils::deviceKernelsGeneric::stridedCopyToBlock(
        numberWaveFunctions,
        totalLocallyOwnedCells * d_numberNodesPerElement,
        src,
        d_cellWaveFunctionMatrix.begin(),
        d_flattenedArrayCellLocalProcIndexIdMapDevice.begin());

      dftfe::utils::deviceBlasWrapper::gemmBatched(
        d_deviceBlasHandle,
        dftfe::utils::DEVICEBLAS_OP_N,
        dftfe::utils::DEVICEBLAS_OP_N,
        numberWaveFunctions,
        d_maxSingleAtomPseudoWfc,
        d_numberNodesPerElement,
        &scalarCoeffAlpha,
        (const dataTypes::number **)d_A,
        numberWaveFunctions,
        (const dataTypes::number **)d_B,
        d_numberNodesPerElement,
        &scalarCoeffBeta,
        d_C,
        numberWaveFunctions,
        d_totalNonlocalElems);

      dftfe::utils::deviceBlasWrapper::gemm(
        d_deviceBlasHandle,
        dftfe::utils::DEVICEBLAS_OP_N,
        dftfe::utils::DEVICEBLAS_OP_N,
        numberWaveFunctions,
        d_totalPseudoWfcNonLocal,
        d_totalNonlocalElems * d_maxSingleAtomPseudoWfc,
        &scalarCoeffAlpha,
        d_projectorKetTimesVectorAllCellsDevice.begin(),
        numberWaveFunctions,
        d_projectorKetTimesVectorAllCellsReductionDevice.begin(),
        d_totalNonlocalElems * d_maxSingleAtomPseudoWfc,
        &scalarCoeffBeta,
        d_projectorKetTimesVectorParFlattenedDevice.begin(),
        numberWaveFunctions);
    }

  projectorKetTimesVector.setValue(0);


  if (d_totalNonlocalElems > 0)
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    copyToDealiiParallelNonLocalVec<<<
      (numberWaveFunctions + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
        dftfe::utils::DEVICE_BLOCK_SIZE * d_totalPseudoWfcNonLocal,
      dftfe::utils::DEVICE_BLOCK_SIZE>>>(
      numberWaveFunctions,
      d_totalPseudoWfcNonLocal,
      dftfe::utils::makeDataTypeDeviceCompatible(
        d_projectorKetTimesVectorParFlattenedDevice.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(
        projectorKetTimesVector.begin()),
      d_projectorIdsParallelNumberingMapDevice.begin());
#elif DFTFE_WITH_DEVICE_LANG_HIP
    hipLaunchKernelGGL(copyToDealiiParallelNonLocalVec,
                       (numberWaveFunctions +
                        (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                         dftfe::utils::DEVICE_BLOCK_SIZE *
                         d_totalPseudoWfcNonLocal,
                       dftfe::utils::DEVICE_BLOCK_SIZE,
                       0,
                       0,
                       numberWaveFunctions,
                       d_totalPseudoWfcNonLocal,
                       dftfe::utils::makeDataTypeDeviceCompatible(
                         d_projectorKetTimesVectorParFlattenedDevice.begin()),
                       dftfe::utils::makeDataTypeDeviceCompatible(
                         projectorKetTimesVector.begin()),
                       d_projectorIdsParallelNumberingMapDevice.begin());
#endif

  projectorKetTimesVector.accumulateAddLocallyOwned(1);
  projectorKetTimesVector.updateGhostValues(1);

  //
  // compute V*C^{\dagger}*X
  //
  if (d_totalNonlocalElems > 0)
    dftfe::utils::deviceKernelsGeneric::stridedBlockScale(
      numberWaveFunctions,
      d_totalPseudoWfcNonLocal,
      1.0,
      d_nonLocalPseudoPotentialConstantsDevice.begin(),
      projectorKetTimesVector.begin());
}
