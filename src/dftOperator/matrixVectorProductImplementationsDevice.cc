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
// @author Phani Motamarri, Sambit Das
//


/** @file matrixVectorProductImplementations.cc
 *  @brief Contains linear algebra operations
 *
 */


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>::
  computeLocalHamiltonianTimesX(
    const dataTypes::number *src,
    const unsigned int       numberWaveFunctions,
    dataTypes::number *      dst,
    const bool               onlyHPrimePartForFirstOrderDensityMatResponse)
{
  const unsigned int kpointSpinIndex =
    (1 + dftPtr->d_dftParamsPtr->spinPolarized) * d_kPointIndex + d_spinIndex;
  const unsigned int totalLocallyOwnedCells =
    dftPtr->matrix_free_data.n_physical_cells();

  copyDeviceKernel<<<(numberWaveFunctions +
                      (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                       dftfe::utils::DEVICE_BLOCK_SIZE *
                       totalLocallyOwnedCells * d_numberNodesPerElement,
                     dftfe::utils::DEVICE_BLOCK_SIZE>>>(
    numberWaveFunctions,
    totalLocallyOwnedCells * d_numberNodesPerElement,
    dftfe::utils::makeDataTypeDeviceCompatible(src),
    dftfe::utils::makeDataTypeDeviceCompatible(
      d_cellWaveFunctionMatrix.begin()),
    d_flattenedArrayCellLocalProcIndexIdMapDevice.begin());


  const dataTypes::number scalarCoeffAlpha = dataTypes::number(1.0),
                          scalarCoeffBeta  = dataTypes::number(0.0);
  const unsigned int strideA = d_numberNodesPerElement * numberWaveFunctions;
  const unsigned int strideB =
    d_numberNodesPerElement * d_numberNodesPerElement;
  const unsigned int strideC = d_numberNodesPerElement * numberWaveFunctions;


  dftfe::utils::deviceBlasWrapper::gemmStridedBatched(
    d_deviceBlasHandle,
    dftfe::utils::DEVICEBLAS_OP_N,
    std::is_same<dataTypes::number, std::complex<double>>::value ?
      dftfe::utils::DEVICEBLAS_OP_T :
      dftfe::utils::DEVICEBLAS_OP_N,
    numberWaveFunctions,
    d_numberNodesPerElement,
    d_numberNodesPerElement,
    &scalarCoeffAlpha,
    d_cellWaveFunctionMatrix.begin(),
    numberWaveFunctions,
    strideA,
    d_cellHamiltonianMatrixFlattenedDevice.begin() +
      d_numLocallyOwnedCells * d_numberNodesPerElement *
        d_numberNodesPerElement * kpointSpinIndex,
    d_numberNodesPerElement,
    strideB,
    &scalarCoeffBeta,
    d_cellHamMatrixTimesWaveMatrix.begin(),
    numberWaveFunctions,
    strideC,
    totalLocallyOwnedCells);


  if (!(dftPtr->d_dftParamsPtr->isPseudopotential &&
        dftPtr->d_nonLocalAtomGlobalChargeIds.size() > 0) ||
      onlyHPrimePartForFirstOrderDensityMatResponse)
    {
      if (std::is_same<dataTypes::number, std::complex<double>>::value)
        {
          deviceUtils::copyComplexArrToRealArrsDevice(
            (d_parallelChebyBlockVectorDevice.localSize() *
             d_parallelChebyBlockVectorDevice.numVectors()),
            dst,
            d_tempRealVec.begin(),
            d_tempImagVec.begin());


          daxpyAtomicAddKernel<<<
            (numberWaveFunctions + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE * d_numLocallyOwnedCells *
              d_numberNodesPerElement,
            dftfe::utils::DEVICE_BLOCK_SIZE>>>(
            numberWaveFunctions,
            d_numLocallyOwnedCells * d_numberNodesPerElement,
            dftfe::utils::makeDataTypeDeviceCompatible(
              d_cellHamMatrixTimesWaveMatrix.begin()),
            d_tempRealVec.begin(),
            d_tempImagVec.begin(),
            d_flattenedArrayCellLocalProcIndexIdMapDevice.begin());

          deviceUtils::copyRealArrsToComplexArrDevice(
            (d_parallelChebyBlockVectorDevice.localSize() *
             d_parallelChebyBlockVectorDevice.numVectors()),
            d_tempRealVec.begin(),
            d_tempImagVec.begin(),
            dst);
        }
      else
        daxpyAtomicAddKernel<<<
          (numberWaveFunctions + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
            dftfe::utils::DEVICE_BLOCK_SIZE * d_numLocallyOwnedCells *
            d_numberNodesPerElement,
          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          numberWaveFunctions,
          d_numLocallyOwnedCells * d_numberNodesPerElement,
          dftfe::utils::makeDataTypeDeviceCompatible(
            d_cellHamMatrixTimesWaveMatrix.begin()),
          dftfe::utils::makeDataTypeDeviceCompatible(dst),
          d_flattenedArrayCellLocalProcIndexIdMapDevice.begin());
    }
}
