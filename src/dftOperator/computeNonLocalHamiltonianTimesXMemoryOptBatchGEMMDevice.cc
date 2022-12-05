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
    const unsigned int                             numberWaveFunctions,
    dataTypes::number *                      dst,
    const bool                                     skip1,
    const bool                                     skip2)
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
        dftfe::utils::makeDataTypeDeviceCompatible(&scalarCoeffAlpha),
        (const dataTypes::numberDevice **)d_A,
        numberWaveFunctions,
        (const dataTypes::numberDevice **)d_B,
        d_numberNodesPerElement,
        dftfe::utils::makeDataTypeDeviceCompatible(&scalarCoeffBeta),
        (dataTypes::numberDevice **)d_C,
        numberWaveFunctions,
        d_totalNonlocalElems);

      cublasXgemm(
        d_cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        numberWaveFunctions,
        d_totalPseudoWfcNonLocal,
        d_totalNonlocalElems * d_maxSingleAtomPseudoWfc,
        dftfe::utils::makeDataTypeDeviceCompatible(&scalarCoeffAlpha),
        dftfe::utils::makeDataTypeDeviceCompatible(
        d_projectorKetTimesVectorAllCellsDevice.begin()),
        numberWaveFunctions,
        dftfe::utils::makeDataTypeDeviceCompatible(
            d_projectorKetTimesVectorAllCellsReductionDevice.begin()),
        d_totalNonlocalElems * d_maxSingleAtomPseudoWfc,
        dftfe::utils::makeDataTypeDeviceCompatible(&scalarCoeffBeta),
        dftfe::utils::makeDataTypeDeviceCompatible(d_projectorKetTimesVectorParFlattenedDevice.begin()),
        numberWaveFunctions);
    }

  // this routine was interfering with overlapping communication and compute. So
  // called separately inside chebyshevFilter. So skip this if either skip1 or
  // skip2 are set to true
  if (!skip1 && !skip2)
    projectorKetTimesVector.setZero();


  if (d_totalNonlocalElems > 0 && !skip1)
    copyToDealiiParallelNonLocalVec<<<
      (numberWaveFunctions + (deviceConstants::blockSize - 1)) /
        deviceConstants::blockSize * d_totalPseudoWfcNonLocal,
      deviceConstants::blockSize>>>(
      numberWaveFunctions,
      d_totalPseudoWfcNonLocal,
      dftfe::utils::makeDataTypeDeviceCompatible(d_projectorKetTimesVectorParFlattenedDevice.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(projectorKetTimesVector.begin()),
      d_projectorIdsParallelNumberingMapDevice.begin());

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
      scaleDeviceKernel<<<
        (numberWaveFunctions + (deviceConstants::blockSize - 1)) /
          deviceConstants::blockSize * d_totalPseudoWfcNonLocal,
        deviceConstants::blockSize>>>(
        numberWaveFunctions,
        d_totalPseudoWfcNonLocal,
        1.0,
        dftfe::utils::makeDataTypeDeviceCompatible(projectorKetTimesVector.begin()),
        d_nonLocalPseudoPotentialConstantsDevice.begin());

      copyFromParallelNonLocalVecToAllCellsVec<<<
        (numberWaveFunctions + (deviceConstants::blockSize - 1)) /
          deviceConstants::blockSize * d_totalNonlocalElems *
          d_maxSingleAtomPseudoWfc,
        deviceConstants::blockSize>>>(
        numberWaveFunctions,
        d_totalNonlocalElems,
        d_maxSingleAtomPseudoWfc,
        dftfe::utils::makeDataTypeDeviceCompatible(projectorKetTimesVector.begin()),
        dftfe::utils::makeDataTypeDeviceCompatible(d_projectorKetTimesVectorAllCellsDevice.begin()),
          d_indexMapFromPaddedNonLocalVecToParallelNonLocalVecDevice.begin());

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
        dftfe::utils::makeDataTypeDeviceCompatible(&scalarCoeffAlpha),
        dftfe::utils::makeDataTypeDeviceCompatible(
        d_projectorKetTimesVectorAllCellsDevice.begin()),
        numberWaveFunctions,
        strideA,
        dftfe::utils::makeDataTypeDeviceCompatible(d_cellHamiltonianMatrixNonLocalFlattenedTransposeDevice.begin()+d_kPointIndex * d_totalNonlocalElems * d_maxSingleAtomPseudoWfc *
             d_numberNodesPerElement),
        d_maxSingleAtomPseudoWfc,
        strideB,
        dftfe::utils::makeDataTypeDeviceCompatible(&scalarCoeffBeta),
        dftfe::utils::makeDataTypeDeviceCompatible(d_cellHamMatrixTimesWaveMatrixNonLocalDevice.begin()),
        numberWaveFunctions,
        strideC,
        d_totalNonlocalElems);


      for (unsigned int iAtom = 0; iAtom < d_totalNonlocalAtomsCurrentProc;
           ++iAtom)
        {
          const unsigned int accum = d_numberCellsAccumNonLocalAtoms[iAtom];
          addNonLocalContributionDeviceKernel<<<
            (numberWaveFunctions + (deviceConstants::blockSize - 1)) /
              deviceConstants::blockSize * d_numberCellsNonLocalAtoms[iAtom] *
              d_numberNodesPerElement,
            deviceConstants::blockSize>>>(
            numberWaveFunctions,
            d_numberCellsNonLocalAtoms[iAtom] * d_numberNodesPerElement,
            dftfe::utils::makeDataTypeDeviceCompatible(
                d_cellHamMatrixTimesWaveMatrixNonLocalDevice.begin() +
              accum * d_numberNodesPerElement * numberWaveFunctions),
            dftfe::utils::makeDataTypeDeviceCompatible(
              d_cellHamMatrixTimesWaveMatrix.begin()),
            d_cellNodeIdMapNonLocalToLocalDevice.begin() +
              accum * d_numberNodesPerElement);
        }
    }

  if (std::is_same<dataTypes::number, std::complex<double>>::value)
    {
      deviceUtils::copyComplexArrToRealArrsDevice(
        (d_parallelChebyBlockVectorDevice.locallyOwnedFlattenedSize() +
         d_parallelChebyBlockVectorDevice.ghostFlattenedSize()),
        dst,
        d_tempRealVec.begin(),
        d_tempImagVec.begin());


      daxpyAtomicAddKernel<<<(numberWaveFunctions +
                              (deviceConstants::blockSize - 1)) /
                               deviceConstants::blockSize *
                               d_numLocallyOwnedCells * d_numberNodesPerElement,
                             deviceConstants::blockSize>>>(
        numberWaveFunctions,
        d_numLocallyOwnedCells * d_numberNodesPerElement,
        dftfe::utils::makeDataTypeDeviceCompatible(
        d_cellHamMatrixTimesWaveMatrix.begin()),
        d_tempRealVec.begin(),
        d_tempImagVec.begin(),
        d_flattenedArrayCellLocalProcIndexIdMapDevice.begin());

      deviceUtils::copyRealArrsToComplexArrDevice(
        (d_parallelChebyBlockVectorDevice.locallyOwnedFlattenedSize() +
         d_parallelChebyBlockVectorDevice.ghostFlattenedSize()),
        d_tempRealVec.begin(),
        d_tempImagVec.begin(),
        dst);
    }
  else
    daxpyAtomicAddKernel<<<(numberWaveFunctions +
                            (deviceConstants::blockSize - 1)) /
                             deviceConstants::blockSize *
                             d_numLocallyOwnedCells * d_numberNodesPerElement,
                           deviceConstants::blockSize>>>(
      numberWaveFunctions,
      d_numLocallyOwnedCells * d_numberNodesPerElement,
      dftfe::utils::makeDataTypeDeviceCompatible(
        d_cellHamMatrixTimesWaveMatrix.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(dst),
        d_flattenedArrayCellLocalProcIndexIdMapDevice.begin());
}


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>::
  computeNonLocalProjectorKetTimesXTimesV(
    const dataTypes::number *                src,
    distributedDeviceVec<dataTypes::number> &projectorKetTimesVector,
    const unsigned int                             numberWaveFunctions)
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
      copyDeviceKernel<<<(numberWaveFunctions +
                          (deviceConstants::blockSize - 1)) /
                           deviceConstants::blockSize * totalLocallyOwnedCells *
                           d_numberNodesPerElement,
                         deviceConstants::blockSize>>>(
        numberWaveFunctions,
        totalLocallyOwnedCells * d_numberNodesPerElement,
        dftfe::utils::makeDataTypeDeviceCompatible(src),
        dftfe::utils::makeDataTypeDeviceCompatible(
          d_cellWaveFunctionMatrix.begin()),
          d_flattenedArrayCellLocalProcIndexIdMapDevice.begin());



      cublasXgemmBatched(
        d_cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        numberWaveFunctions,
        d_maxSingleAtomPseudoWfc,
        d_numberNodesPerElement,
        dftfe::utils::makeDataTypeDeviceCompatible(&scalarCoeffAlpha),
        (const dataTypes::numberDevice **)d_A,
        numberWaveFunctions,
        (const dataTypes::numberDevice **)d_B,
        d_numberNodesPerElement,
        dftfe::utils::makeDataTypeDeviceCompatible(&scalarCoeffBeta),
        (dataTypes::numberDevice **)d_C,
        numberWaveFunctions,
        d_totalNonlocalElems);

      cublasXgemm(
        d_cublasHandle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        numberWaveFunctions,
        d_totalPseudoWfcNonLocal,
        d_totalNonlocalElems * d_maxSingleAtomPseudoWfc,
        dftfe::utils::makeDataTypeDeviceCompatible(&scalarCoeffAlpha),
        dftfe::utils::makeDataTypeDeviceCompatible(
        d_projectorKetTimesVectorAllCellsDevice.begin()),
        numberWaveFunctions,
        dftfe::utils::makeDataTypeDeviceCompatible(
        d_projectorKetTimesVectorAllCellsReductionDevice.begin()),
        d_totalNonlocalElems * d_maxSingleAtomPseudoWfc,
        dftfe::utils::makeDataTypeDeviceCompatible(&scalarCoeffBeta),
        dftfe::utils::makeDataTypeDeviceCompatible(d_projectorKetTimesVectorParFlattenedDevice.begin()),
        numberWaveFunctions);
    }

  projectorKetTimesVector.setZero();


  if (d_totalNonlocalElems > 0)
    copyToDealiiParallelNonLocalVec<<<
      (numberWaveFunctions + (deviceConstants::blockSize - 1)) /
        deviceConstants::blockSize * d_totalPseudoWfcNonLocal,
      deviceConstants::blockSize>>>(
      numberWaveFunctions,
      d_totalPseudoWfcNonLocal,
      dftfe::utils::makeDataTypeDeviceCompatible(d_projectorKetTimesVectorParFlattenedDevice.begin()),
      dftfe::utils::makeDataTypeDeviceCompatible(projectorKetTimesVector.begin()),
      d_projectorIdsParallelNumberingMapDevice.begin());

  projectorKetTimesVector.compressAdd();
  projectorKetTimesVector.updateGhostValues();

  //
  // compute V*C^{\dagger}*X
  //
  if (d_totalNonlocalElems > 0)
    scaleDeviceKernel<<<(numberWaveFunctions +
                         (deviceConstants::blockSize - 1)) /
                          deviceConstants::blockSize * d_totalPseudoWfcNonLocal,
                        deviceConstants::blockSize>>>(
      numberWaveFunctions,
      d_totalPseudoWfcNonLocal,
      1.0,
      dftfe::utils::makeDataTypeDeviceCompatible(projectorKetTimesVector.begin()),
      d_nonLocalPseudoPotentialConstantsDevice.begin());
}
