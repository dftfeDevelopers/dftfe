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
// @author  Sambit Das
//


namespace
{
  __global__ void
  hamMatrixExtPotCorr(const unsigned int numCells,
                      const unsigned int numDofsPerCell,
                      const unsigned int numQuadPoints,
                      const double *     shapeFunctionValues,
                      const double *     shapeFunctionValuesInverted,
                      const double *     vExternalPotCorrJxW,
                      double *cellHamiltonianMatrixExternalPotCorrFlattened)
  {
    const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int index = globalThreadId;
         index < numCells * numDofsPerCell * numDofsPerCell;
         index += blockDim.x * gridDim.x)
      {
        const unsigned int cellIndex =
          index / (numDofsPerCell * numDofsPerCell);
        const unsigned int flattenedCellDofIndex =
          index % (numDofsPerCell * numDofsPerCell);
        const unsigned int cellDofIndexI =
          flattenedCellDofIndex / numDofsPerCell;
        const unsigned int cellDofIndexJ =
          flattenedCellDofIndex % numDofsPerCell;

        double val = 0;
        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            val +=
              vExternalPotCorrJxW[cellIndex * numQuadPoints + q] *
              shapeFunctionValues[cellDofIndexI * numQuadPoints + q] *
              shapeFunctionValuesInverted[q * numDofsPerCell + cellDofIndexJ];
          }

        cellHamiltonianMatrixExternalPotCorrFlattened[index] = val;
      }
  }

  __global__ void
  hamMatrixKernelLDA(
    const unsigned int numCells,
    const unsigned int numDofsPerCell,
    const unsigned int numQuadPoints,
    const double *     shapeFunctionValues,
    const double *     shapeFunctionValuesInverted,
    const double *     shapeFunctionGradientValuesXInverted,
    const double *     shapeFunctionGradientValuesYInverted,
    const double *     shapeFunctionGradientValuesZInverted,
    const double *     cellShapeFunctionGradientIntegral,
    const double *     vEffJxW,
    const double *     JxW,
    const double *     cellHamiltonianMatrixExternalPotCorrFlattened,
    double *           cellHamiltonianMatrixFlattened,
    const double       kPointCoordX,
    const double       kPointCoordY,
    const double       kPointCoordZ,
    const double       kSquareTimesHalf,
    const bool         externalPotCorr)
  {
    const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int index = globalThreadId;
         index < numCells * numDofsPerCell * numDofsPerCell;
         index += blockDim.x * gridDim.x)
      {
        const unsigned int cellIndex =
          index / (numDofsPerCell * numDofsPerCell);
        const unsigned int flattenedCellDofIndex =
          index % (numDofsPerCell * numDofsPerCell);
        const unsigned int cellDofIndexI =
          flattenedCellDofIndex / numDofsPerCell;
        const unsigned int cellDofIndexJ =
          flattenedCellDofIndex % numDofsPerCell;

        double val = 0;
        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            val +=
              vEffJxW[cellIndex * numQuadPoints + q] *
              shapeFunctionValues[cellDofIndexI * numQuadPoints + q] *
              shapeFunctionValuesInverted[q * numDofsPerCell + cellDofIndexJ];
          }

        cellHamiltonianMatrixFlattened[index] =
          0.5 * cellShapeFunctionGradientIntegral[index] + val;
        if (externalPotCorr)
          cellHamiltonianMatrixFlattened[index] +=
            cellHamiltonianMatrixExternalPotCorrFlattened[index];
      }
  }


  __global__ void
  hamMatrixKernelLDA(
    const unsigned int numCells,
    const unsigned int numDofsPerCell,
    const unsigned int numQuadPoints,
    const double *     shapeFunctionValues,
    const double *     shapeFunctionValuesInverted,
    const double *     shapeFunctionGradientValuesXInverted,
    const double *     shapeFunctionGradientValuesYInverted,
    const double *     shapeFunctionGradientValuesZInverted,
    const double *     cellShapeFunctionGradientIntegral,
    const double *     vEffJxW,
    const double *     JxW,
    const double *     cellHamiltonianMatrixExternalPotCorrFlattened,
    cuDoubleComplex *  cellHamiltonianMatrixFlattened,
    const double       kPointCoordX,
    const double       kPointCoordY,
    const double       kPointCoordZ,
    const double       kSquareTimesHalf,
    const bool         externalPotCorr)
  {
    const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int index = globalThreadId;
         index < numCells * numDofsPerCell * numDofsPerCell;
         index += blockDim.x * gridDim.x)
      {
        const unsigned int cellIndex =
          index / (numDofsPerCell * numDofsPerCell);
        const unsigned int flattenedCellDofIndex =
          index % (numDofsPerCell * numDofsPerCell);
        const unsigned int cellDofIndexI =
          flattenedCellDofIndex / numDofsPerCell;
        const unsigned int cellDofIndexJ =
          flattenedCellDofIndex % numDofsPerCell;

        double val     = 0.0;
        double valImag = 0.0;
        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            const double shapeI =
              shapeFunctionValues[cellDofIndexI * numQuadPoints + q];
            const double shapeJ =
              shapeFunctionValuesInverted[q * numDofsPerCell + cellDofIndexJ];

            const double gradShapeXI =
              shapeFunctionGradientValuesXInverted[cellIndex * numQuadPoints *
                                                     numDofsPerCell +
                                                   numDofsPerCell * q +
                                                   cellDofIndexI];
            const double gradShapeYI =
              shapeFunctionGradientValuesYInverted[cellIndex * numQuadPoints *
                                                     numDofsPerCell +
                                                   numDofsPerCell * q +
                                                   cellDofIndexI];
            const double gradShapeZI =
              shapeFunctionGradientValuesZInverted[cellIndex * numQuadPoints *
                                                     numDofsPerCell +
                                                   numDofsPerCell * q +
                                                   cellDofIndexI];

            val += (vEffJxW[cellIndex * numQuadPoints + q] +
                    +kSquareTimesHalf * JxW[cellIndex * numQuadPoints + q]) *
                   shapeI * shapeJ;
            valImag -=
              (kPointCoordX * gradShapeXI + kPointCoordY * gradShapeYI +
               kPointCoordZ * gradShapeZI) *
              shapeJ * JxW[cellIndex * numQuadPoints + q];
          }

        cellHamiltonianMatrixFlattened[index] = make_cuDoubleComplex(
          0.5 * cellShapeFunctionGradientIntegral[index] + val, valImag);
        if (externalPotCorr)
          cellHamiltonianMatrixFlattened[index] = make_cuDoubleComplex(
            cellHamiltonianMatrixFlattened[index].x +
              cellHamiltonianMatrixExternalPotCorrFlattened[index],
            cellHamiltonianMatrixFlattened[index].y);
      }
  }


  __global__ void
  hamMatrixKernelGGAMemOpt(
    const unsigned int numCells,
    const unsigned int numDofsPerCell,
    const unsigned int numQuadPoints,
    const double *     shapeFunctionValues,
    const double *     shapeFunctionValuesInverted,
    const double *     shapeFunctionGradientValuesXInverted,
    const double *     shapeFunctionGradientValuesYInverted,
    const double *     shapeFunctionGradientValuesZInverted,
    const double *     cellShapeFunctionGradientIntegral,
    const double *     vEffJxW,
    const double *     JxW,
    const double *     derExcWithSigmaTimesGradRhoJxW,
    const double *     cellHamiltonianMatrixExternalPotCorrFlattened,
    double *           cellHamiltonianMatrixFlattened,
    const double       kPointCoordX,
    const double       kPointCoordY,
    const double       kPointCoordZ,
    const double       kSquareTimesHalf,
    const bool         externalPotCorr)
  {
    const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int index = globalThreadId;
         index < numCells * numDofsPerCell * numDofsPerCell;
         index += blockDim.x * gridDim.x)
      {
        const unsigned int cellIndex =
          index / (numDofsPerCell * numDofsPerCell);
        const unsigned int flattenedCellDofIndex =
          index % (numDofsPerCell * numDofsPerCell);
        const unsigned int cellDofIndexI =
          flattenedCellDofIndex / numDofsPerCell;
        const unsigned int cellDofIndexJ =
          flattenedCellDofIndex % numDofsPerCell;

        double val = 0;
        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            const double shapeI =
              shapeFunctionValues[cellDofIndexI * numQuadPoints + q];
            const double shapeJ =
              shapeFunctionValuesInverted[q * numDofsPerCell + cellDofIndexJ];

            const double gradShapeXI =
              shapeFunctionGradientValuesXInverted[cellIndex * numQuadPoints *
                                                     numDofsPerCell +
                                                   numDofsPerCell * q +
                                                   cellDofIndexI];
            const double gradShapeYI =
              shapeFunctionGradientValuesYInverted[cellIndex * numQuadPoints *
                                                     numDofsPerCell +
                                                   numDofsPerCell * q +
                                                   cellDofIndexI];
            const double gradShapeZI =
              shapeFunctionGradientValuesZInverted[cellIndex * numQuadPoints *
                                                     numDofsPerCell +
                                                   numDofsPerCell * q +
                                                   cellDofIndexI];

            const double gradShapeXJ =
              shapeFunctionGradientValuesXInverted[cellIndex * numQuadPoints *
                                                     numDofsPerCell +
                                                   numDofsPerCell * q +
                                                   cellDofIndexJ];
            const double gradShapeYJ =
              shapeFunctionGradientValuesYInverted[cellIndex * numQuadPoints *
                                                     numDofsPerCell +
                                                   numDofsPerCell * q +
                                                   cellDofIndexJ];
            const double gradShapeZJ =
              shapeFunctionGradientValuesZInverted[cellIndex * numQuadPoints *
                                                     numDofsPerCell +
                                                   numDofsPerCell * q +
                                                   cellDofIndexJ];


            val +=
              vEffJxW[cellIndex * numQuadPoints + q] * shapeI * shapeJ +
              2.0 *
                (derExcWithSigmaTimesGradRhoJxW[cellIndex * numQuadPoints * 3 +
                                                3 * q] *
                   (gradShapeXI * shapeJ + gradShapeXJ * shapeI) +
                 derExcWithSigmaTimesGradRhoJxW[cellIndex * numQuadPoints * 3 +
                                                3 * q + 1] *
                   (gradShapeYI * shapeJ + gradShapeYJ * shapeI) +
                 derExcWithSigmaTimesGradRhoJxW[cellIndex * numQuadPoints * 3 +
                                                3 * q + 2] *
                   (gradShapeZI * shapeJ + gradShapeZJ * shapeI));
          }

        cellHamiltonianMatrixFlattened[index] =
          0.5 * cellShapeFunctionGradientIntegral[index] + val;
        if (externalPotCorr)
          cellHamiltonianMatrixFlattened[index] +=
            cellHamiltonianMatrixExternalPotCorrFlattened[index];
      }
  }

  __global__ void
  hamMatrixKernelGGAMemOpt(
    const unsigned int numCells,
    const unsigned int numDofsPerCell,
    const unsigned int numQuadPoints,
    const double *     shapeFunctionValues,
    const double *     shapeFunctionValuesInverted,
    const double *     shapeFunctionGradientValuesXInverted,
    const double *     shapeFunctionGradientValuesYInverted,
    const double *     shapeFunctionGradientValuesZInverted,
    const double *     cellShapeFunctionGradientIntegral,
    const double *     vEffJxW,
    const double *     JxW,
    const double *     derExcWithSigmaTimesGradRhoJxW,
    const double *     cellHamiltonianMatrixExternalPotCorrFlattened,
    cuDoubleComplex *  cellHamiltonianMatrixFlattened,
    const double       kPointCoordX,
    const double       kPointCoordY,
    const double       kPointCoordZ,
    const double       kSquareTimesHalf,
    const bool         externalPotCorr)
  {
    const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    for (unsigned int index = globalThreadId;
         index < numCells * numDofsPerCell * numDofsPerCell;
         index += blockDim.x * gridDim.x)
      {
        const unsigned int cellIndex =
          index / (numDofsPerCell * numDofsPerCell);
        const unsigned int flattenedCellDofIndex =
          index % (numDofsPerCell * numDofsPerCell);
        const unsigned int cellDofIndexI =
          flattenedCellDofIndex / numDofsPerCell;
        const unsigned int cellDofIndexJ =
          flattenedCellDofIndex % numDofsPerCell;

        double val     = 0.0;
        double valImag = 0.0;
        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            const double shapeI =
              shapeFunctionValues[cellDofIndexI * numQuadPoints + q];
            const double shapeJ =
              shapeFunctionValuesInverted[q * numDofsPerCell + cellDofIndexJ];

            const double gradShapeXI =
              shapeFunctionGradientValuesXInverted[cellIndex * numQuadPoints *
                                                     numDofsPerCell +
                                                   numDofsPerCell * q +
                                                   cellDofIndexI];
            const double gradShapeYI =
              shapeFunctionGradientValuesYInverted[cellIndex * numQuadPoints *
                                                     numDofsPerCell +
                                                   numDofsPerCell * q +
                                                   cellDofIndexI];
            const double gradShapeZI =
              shapeFunctionGradientValuesZInverted[cellIndex * numQuadPoints *
                                                     numDofsPerCell +
                                                   numDofsPerCell * q +
                                                   cellDofIndexI];

            const double gradShapeXJ =
              shapeFunctionGradientValuesXInverted[cellIndex * numQuadPoints *
                                                     numDofsPerCell +
                                                   numDofsPerCell * q +
                                                   cellDofIndexJ];
            const double gradShapeYJ =
              shapeFunctionGradientValuesYInverted[cellIndex * numQuadPoints *
                                                     numDofsPerCell +
                                                   numDofsPerCell * q +
                                                   cellDofIndexJ];
            const double gradShapeZJ =
              shapeFunctionGradientValuesZInverted[cellIndex * numQuadPoints *
                                                     numDofsPerCell +
                                                   numDofsPerCell * q +
                                                   cellDofIndexJ];


            val +=
              (vEffJxW[cellIndex * numQuadPoints + q] +
               kSquareTimesHalf * JxW[cellIndex * numQuadPoints + q]) *
                shapeI * shapeJ +
              2.0 *
                (derExcWithSigmaTimesGradRhoJxW[cellIndex * numQuadPoints * 3 +
                                                3 * q] *
                   (gradShapeXI * shapeJ + gradShapeXJ * shapeI) +
                 derExcWithSigmaTimesGradRhoJxW[cellIndex * numQuadPoints * 3 +
                                                3 * q + 1] *
                   (gradShapeYI * shapeJ + gradShapeYJ * shapeI) +
                 derExcWithSigmaTimesGradRhoJxW[cellIndex * numQuadPoints * 3 +
                                                3 * q + 2] *
                   (gradShapeZI * shapeJ + gradShapeZJ * shapeI));

            valImag -=
              (kPointCoordX * gradShapeXI + kPointCoordY * gradShapeYI +
               kPointCoordZ * gradShapeZI) *
              shapeJ * JxW[cellIndex * numQuadPoints + q];
          }

        cellHamiltonianMatrixFlattened[index] = make_cuDoubleComplex(
          0.5 * cellShapeFunctionGradientIntegral[index] + val, valImag);
        if (externalPotCorr)
          cellHamiltonianMatrixFlattened[index] = make_cuDoubleComplex(
            cellHamiltonianMatrixFlattened[index].x +
              cellHamiltonianMatrixExternalPotCorrFlattened[index],
            cellHamiltonianMatrixFlattened[index].y);
      }
  }

} // namespace


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorCUDAClass<FEOrder, FEOrderElectro>::computeHamiltonianMatrix(
  const unsigned int kPointIndex,
  const unsigned int spinIndex)
{
  dealii::TimerOutput computingTimerStandard(
      this->getMPICommunicator(),
      pcout,
      dftParameters::reproducible_output || dftParameters::verbosity < 2 ?
        dealii::TimerOutput::never :
        dealii::TimerOutput::every_call,
      dealii::TimerOutput::wall_times);

  if (dftParameters::gpuFineGrainedTimings)
  {
        cudaDeviceSynchronize();
        computingTimerStandard.enter_subsection("Hamiltonian construction on GPU");
  }  

  const unsigned int kpointSpinIndex =
    (1 + dftParameters::spinPolarized) * kPointIndex + spinIndex;

  const double kPointCoordX = dftPtr->d_kPointCoordinates[3 * kPointIndex + 0];
  const double kPointCoordY = dftPtr->d_kPointCoordinates[3 * kPointIndex + 1];
  const double kPointCoordZ = dftPtr->d_kPointCoordinates[3 * kPointIndex + 2];

  const double kSquareTimesHalf =
    0.5 * (kPointCoordX * kPointCoordX + kPointCoordY * kPointCoordY +
           kPointCoordZ * kPointCoordZ);

  if ((dftParameters::isPseudopotential ||
       dftParameters::smearedNuclearCharges) &&
      !d_isStiffnessMatrixExternalPotCorrComputed)
    {
      hamMatrixExtPotCorr<<<(d_numLocallyOwnedCells * d_numberNodesPerElement *
                               d_numberNodesPerElement +
                             255) /
                              256,
                            256>>>(
        d_numLocallyOwnedCells,
        d_numberNodesPerElement,
        d_numQuadPointsLpsp,
        thrust::raw_pointer_cast(&d_shapeFunctionValueLpspDevice[0]),
        thrust::raw_pointer_cast(&d_shapeFunctionValueInvertedLpspDevice[0]),
        thrust::raw_pointer_cast(&d_vEffExternalPotCorrJxWDevice[0]),
        thrust::raw_pointer_cast(
          &d_cellHamiltonianMatrixExternalPotCorrFlattenedDevice[0]));

      d_isStiffnessMatrixExternalPotCorrComputed = true;
    }

  if (dftParameters::xcFamilyType == "GGA")
    {
      hamMatrixKernelGGAMemOpt<<<(d_numLocallyOwnedCells *
                                    d_numberNodesPerElement *
                                    d_numberNodesPerElement +
                                  255) /
                                   256,
                                 256>>>(
        d_numLocallyOwnedCells,
        d_numberNodesPerElement,
        d_numQuadPoints,
        thrust::raw_pointer_cast(&d_shapeFunctionValueDevice[0]),
        thrust::raw_pointer_cast(&d_shapeFunctionValueInvertedDevice[0]),
        thrust::raw_pointer_cast(
          &d_shapeFunctionGradientValueXInvertedDevice[0]),
        thrust::raw_pointer_cast(
          &d_shapeFunctionGradientValueYInvertedDevice[0]),
        thrust::raw_pointer_cast(
          &d_shapeFunctionGradientValueZInvertedDevice[0]),
        thrust::raw_pointer_cast(
          &d_cellShapeFunctionGradientIntegralFlattenedDevice[0]),
        thrust::raw_pointer_cast(&d_vEffJxWDevice[0]),
        thrust::raw_pointer_cast(&d_cellJxWValuesDevice[0]),
        thrust::raw_pointer_cast(&d_derExcWithSigmaTimesGradRhoJxWDevice[0]),
        thrust::raw_pointer_cast(
          &d_cellHamiltonianMatrixExternalPotCorrFlattenedDevice[0]),
        reinterpret_cast<dataTypes::numberGPU *>(thrust::raw_pointer_cast(
          &d_cellHamiltonianMatrixFlattenedDevice[kpointSpinIndex *
                                                  d_numLocallyOwnedCells *
                                                  d_numberNodesPerElement *
                                                  d_numberNodesPerElement])),
        kPointCoordX,
        kPointCoordY,
        kPointCoordZ,
        kSquareTimesHalf,
        dftParameters::isPseudopotential ||
          dftParameters::smearedNuclearCharges);
    }
  else
    hamMatrixKernelLDA<<<(d_numLocallyOwnedCells * d_numberNodesPerElement *
                            d_numberNodesPerElement +
                          255) /
                           256,
                         256>>>(
      d_numLocallyOwnedCells,
      d_numberNodesPerElement,
      d_numQuadPoints,
      thrust::raw_pointer_cast(&d_shapeFunctionValueDevice[0]),
      thrust::raw_pointer_cast(&d_shapeFunctionValueInvertedDevice[0]),
      thrust::raw_pointer_cast(&d_shapeFunctionGradientValueXInvertedDevice[0]),
      thrust::raw_pointer_cast(&d_shapeFunctionGradientValueYInvertedDevice[0]),
      thrust::raw_pointer_cast(&d_shapeFunctionGradientValueZInvertedDevice[0]),
      thrust::raw_pointer_cast(
        &d_cellShapeFunctionGradientIntegralFlattenedDevice[0]),
      thrust::raw_pointer_cast(&d_vEffJxWDevice[0]),
      thrust::raw_pointer_cast(&d_cellJxWValuesDevice[0]),
      thrust::raw_pointer_cast(
        &d_cellHamiltonianMatrixExternalPotCorrFlattenedDevice[0]),
      reinterpret_cast<dataTypes::numberGPU *>(thrust::raw_pointer_cast(
        &d_cellHamiltonianMatrixFlattenedDevice[kpointSpinIndex *
                                                d_numLocallyOwnedCells *
                                                d_numberNodesPerElement *
                                                d_numberNodesPerElement])),
      kPointCoordX,
      kPointCoordY,
      kPointCoordZ,
      kSquareTimesHalf,
      dftParameters::isPseudopotential || dftParameters::smearedNuclearCharges);

      if (dftParameters::gpuFineGrainedTimings)
      {
        cudaDeviceSynchronize();
        computingTimerStandard.leave_subsection("Hamiltonian construction on GPU");
      }
}
