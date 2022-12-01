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
// @author  Sambit Das
//


namespace
{
  __global__ void
  hamMatrixExtPotCorr(const unsigned int numCells,
                      const unsigned int numDofsPerCell,
                      const unsigned int numQuadPoints,
                      const double *     shapeFunctionValues,
                      const double *     shapeFunctionValuesTransposed,
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
              shapeFunctionValuesTransposed[q * numDofsPerCell + cellDofIndexJ];
          }

        cellHamiltonianMatrixExternalPotCorrFlattened[index] = val;
      }
  }

  __global__ void
  hamMatrixKernelLDA(
    const unsigned int numCells,
    const unsigned int numDofsPerCell,
    const unsigned int numQuadPoints,
    const unsigned int spinIndex,
    const unsigned int nspin,
    const unsigned int numkPoints,
    const double *     shapeFunctionValues,
    const double *     shapeFunctionValuesTransposed,
    const double *     shapeFunctionGradientValuesXTransposed,
    const double *     shapeFunctionGradientValuesYTransposed,
    const double *     shapeFunctionGradientValuesZTransposed,
    const double *     cellShapeFunctionGradientIntegral,
    const double *     vEffJxW,
    const double *     JxW,
    const double *     cellHamiltonianMatrixExternalPotCorrFlattened,
    double *           cellHamiltonianMatrixFlattened,
    const double *     kPointCoordsVec,
    const double *     kSquareTimesHalfVec,
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
              shapeFunctionValuesTransposed[q * numDofsPerCell + cellDofIndexJ];
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
    const unsigned int spinIndex,
    const unsigned int nspin,
    const unsigned int numkPoints,
    const double *     shapeFunctionValues,
    const double *     shapeFunctionValuesTransposed,
    const double *     shapeFunctionGradientValuesXTransposed,
    const double *     shapeFunctionGradientValuesYTransposed,
    const double *     shapeFunctionGradientValuesZTransposed,
    const double *     cellShapeFunctionGradientIntegral,
    const double *     vEffJxW,
    const double *     JxW,
    const double *     cellHamiltonianMatrixExternalPotCorrFlattened,
    cuDoubleComplex *  cellHamiltonianMatrixFlattened,
    const double *     kPointCoordsVec,
    const double *     kSquareTimesHalfVec,
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

        double val         = 0.0;
        double valRealKpt  = 0.0;
        double valImagKptX = 0;
        double valImagKptY = 0;
        double valImagKptZ = 0;

        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            const double shapeI =
              shapeFunctionValues[cellDofIndexI * numQuadPoints + q];
            const double shapeJ =
              shapeFunctionValuesTransposed[q * numDofsPerCell + cellDofIndexJ];

            const double gradShapeXI =
              shapeFunctionGradientValuesXTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexI];
            const double gradShapeYI =
              shapeFunctionGradientValuesYTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexI];
            const double gradShapeZI =
              shapeFunctionGradientValuesZTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexI];

            val += vEffJxW[cellIndex * numQuadPoints + q] * shapeI * shapeJ;

            valRealKpt += JxW[cellIndex * numQuadPoints + q] * shapeI * shapeJ;

            valImagKptX -=
              gradShapeXI * shapeJ * JxW[cellIndex * numQuadPoints + q];
            valImagKptY -=
              gradShapeYI * shapeJ * JxW[cellIndex * numQuadPoints + q];
            valImagKptZ -=
              gradShapeZI * shapeJ * JxW[cellIndex * numQuadPoints + q];
          }

        for (unsigned int ikpt = 0; ikpt < numkPoints; ++ikpt)
          {
            const unsigned int startIndex = (nspin * ikpt + spinIndex) *
                                            numCells * numDofsPerCell *
                                            numDofsPerCell;
            cellHamiltonianMatrixFlattened[startIndex + index] =
              make_cuDoubleComplex(
                0.5 * cellShapeFunctionGradientIntegral[index] + val +
                  kSquareTimesHalfVec[ikpt] * valRealKpt,
                kPointCoordsVec[3 * ikpt + 0] * valImagKptX +
                  kPointCoordsVec[3 * ikpt + 1] * valImagKptY +
                  kPointCoordsVec[3 * ikpt + 2] * valImagKptZ);
            if (externalPotCorr)
              cellHamiltonianMatrixFlattened[startIndex + index] =
                make_cuDoubleComplex(
                  cellHamiltonianMatrixFlattened[startIndex + index].x +
                    cellHamiltonianMatrixExternalPotCorrFlattened[index],
                  cellHamiltonianMatrixFlattened[startIndex + index].y);
          }
      }
  }


  __global__ void
  hamMatrixKernelGGAMemOpt(
    const unsigned int numCells,
    const unsigned int numDofsPerCell,
    const unsigned int numQuadPoints,
    const unsigned int spinIndex,
    const unsigned int nspin,
    const unsigned int numkPoints,
    const double *     shapeFunctionValues,
    const double *     shapeFunctionValuesTransposed,
    const double *     shapeFunctionGradientValuesXTransposed,
    const double *     shapeFunctionGradientValuesYTransposed,
    const double *     shapeFunctionGradientValuesZTransposed,
    const double *     cellShapeFunctionGradientIntegral,
    const double *     vEffJxW,
    const double *     JxW,
    const double *     derExcWithSigmaTimesGradRhoJxW,
    const double *     cellHamiltonianMatrixExternalPotCorrFlattened,
    double *           cellHamiltonianMatrixFlattened,
    const double *     kPointCoordsVec,
    const double *     kSquareTimesHalfVec,
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
              shapeFunctionValuesTransposed[q * numDofsPerCell + cellDofIndexJ];

            const double gradShapeXI =
              shapeFunctionGradientValuesXTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexI];
            const double gradShapeYI =
              shapeFunctionGradientValuesYTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexI];
            const double gradShapeZI =
              shapeFunctionGradientValuesZTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexI];

            const double gradShapeXJ =
              shapeFunctionGradientValuesXTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexJ];
            const double gradShapeYJ =
              shapeFunctionGradientValuesYTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexJ];
            const double gradShapeZJ =
              shapeFunctionGradientValuesZTransposed[cellIndex * numQuadPoints *
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

        cellHamiltonianMatrixFlattened[spinIndex * numCells * numDofsPerCell *
                                         numDofsPerCell +
                                       index] =
          0.5 * cellShapeFunctionGradientIntegral[index] + val;
        if (externalPotCorr)
          cellHamiltonianMatrixFlattened[spinIndex * numCells * numDofsPerCell *
                                           numDofsPerCell +
                                         index] +=
            cellHamiltonianMatrixExternalPotCorrFlattened[index];
      }
  }


  __global__ void
  hamMatrixKernelGGAMemOpt(
    const unsigned int numCells,
    const unsigned int numDofsPerCell,
    const unsigned int numQuadPoints,
    const unsigned int spinIndex,
    const unsigned int nspin,
    const unsigned int numkPoints,
    const double *     shapeFunctionValues,
    const double *     shapeFunctionValuesTransposed,
    const double *     shapeFunctionGradientValuesXTransposed,
    const double *     shapeFunctionGradientValuesYTransposed,
    const double *     shapeFunctionGradientValuesZTransposed,
    const double *     cellShapeFunctionGradientIntegral,
    const double *     vEffJxW,
    const double *     JxW,
    const double *     derExcWithSigmaTimesGradRhoJxW,
    const double *     cellHamiltonianMatrixExternalPotCorrFlattened,
    cuDoubleComplex *  cellHamiltonianMatrixFlattened,
    const double *     kPointCoordsVec,
    const double *     kSquareTimesHalfVec,
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

        double val         = 0.0;
        double valRealKpt  = 0;
        double valImagKptX = 0;
        double valImagKptY = 0;
        double valImagKptZ = 0;

        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            const double shapeI =
              shapeFunctionValues[cellDofIndexI * numQuadPoints + q];
            const double shapeJ =
              shapeFunctionValuesTransposed[q * numDofsPerCell + cellDofIndexJ];

            const double gradShapeXI =
              shapeFunctionGradientValuesXTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexI];
            const double gradShapeYI =
              shapeFunctionGradientValuesYTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexI];
            const double gradShapeZI =
              shapeFunctionGradientValuesZTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexI];

            const double gradShapeXJ =
              shapeFunctionGradientValuesXTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexJ];
            const double gradShapeYJ =
              shapeFunctionGradientValuesYTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexJ];
            const double gradShapeZJ =
              shapeFunctionGradientValuesZTransposed[cellIndex * numQuadPoints *
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

            valRealKpt += JxW[cellIndex * numQuadPoints + q] * shapeI * shapeJ;

            valImagKptX -=
              gradShapeXI * shapeJ * JxW[cellIndex * numQuadPoints + q];
            valImagKptY -=
              gradShapeYI * shapeJ * JxW[cellIndex * numQuadPoints + q];
            valImagKptZ -=
              gradShapeZI * shapeJ * JxW[cellIndex * numQuadPoints + q];
          }

        for (unsigned int ikpt = 0; ikpt < numkPoints; ++ikpt)
          {
            const unsigned int startIndex = (nspin * ikpt + spinIndex) *
                                            numCells * numDofsPerCell *
                                            numDofsPerCell;
            cellHamiltonianMatrixFlattened[startIndex + index] =
              make_cuDoubleComplex(
                0.5 * cellShapeFunctionGradientIntegral[index] + val +
                  kSquareTimesHalfVec[ikpt] * valRealKpt,
                kPointCoordsVec[3 * ikpt + 0] * valImagKptX +
                  kPointCoordsVec[3 * ikpt + 1] * valImagKptY +
                  kPointCoordsVec[3 * ikpt + 2] * valImagKptZ);
            if (externalPotCorr)
              cellHamiltonianMatrixFlattened[startIndex + index] =
                make_cuDoubleComplex(
                  cellHamiltonianMatrixFlattened[startIndex + index].x +
                    cellHamiltonianMatrixExternalPotCorrFlattened[index],
                  cellHamiltonianMatrixFlattened[startIndex + index].y);
          }
      }
  }


  __global__ void
  hamPrimeMatrixKernelLDA(const unsigned int numCells,
                          const unsigned int numDofsPerCell,
                          const unsigned int numQuadPoints,
                          const double *     shapeFunctionValues,
                          const double *     shapeFunctionValuesTransposed,
                          const double *shapeFunctionGradientValuesXTransposed,
                          const double *shapeFunctionGradientValuesYTransposed,
                          const double *shapeFunctionGradientValuesZTransposed,
                          const double *vEffPrimeJxW,
                          const double *JxW,
                          double *      cellHamiltonianPrimeMatrixFlattened)
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
              vEffPrimeJxW[cellIndex * numQuadPoints + q] *
              shapeFunctionValues[cellDofIndexI * numQuadPoints + q] *
              shapeFunctionValuesTransposed[q * numDofsPerCell + cellDofIndexJ];
          }

        cellHamiltonianPrimeMatrixFlattened[index] = val;
      }
  }


  __global__ void
  hamPrimeMatrixKernelLDA(const unsigned int numCells,
                          const unsigned int numDofsPerCell,
                          const unsigned int numQuadPoints,
                          const double *     shapeFunctionValues,
                          const double *     shapeFunctionValuesTransposed,
                          const double *shapeFunctionGradientValuesXTransposed,
                          const double *shapeFunctionGradientValuesYTransposed,
                          const double *shapeFunctionGradientValuesZTransposed,
                          const double *vEffPrimeJxW,
                          const double *JxW,
                          cuDoubleComplex *cellHamiltonianPrimeMatrixFlattened)
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

        double val = 0.0;
        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            const double shapeI =
              shapeFunctionValues[cellDofIndexI * numQuadPoints + q];
            const double shapeJ =
              shapeFunctionValuesTransposed[q * numDofsPerCell + cellDofIndexJ];

            val +=
              (vEffPrimeJxW[cellIndex * numQuadPoints + q]) * shapeI * shapeJ;
          }

        cellHamiltonianPrimeMatrixFlattened[index] =
          make_cuDoubleComplex(val, 0.0);
      }
  }

  __global__ void
  hamPrimeMatrixKernelGGAMemOpt(
    const unsigned int numCells,
    const unsigned int numDofsPerCell,
    const unsigned int numQuadPoints,
    const double *     shapeFunctionValues,
    const double *     shapeFunctionValuesTransposed,
    const double *     shapeFunctionGradientValuesXTransposed,
    const double *     shapeFunctionGradientValuesYTransposed,
    const double *     shapeFunctionGradientValuesZTransposed,
    const double *     vEffPrimeJxW,
    const double *     JxW,
    const double *     derExcPrimeWithSigmaTimesGradRhoJxW,
    cuDoubleComplex *  cellHamiltonianPrimeMatrixFlattened)
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

        double val = 0.0;
        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            const double shapeI =
              shapeFunctionValues[cellDofIndexI * numQuadPoints + q];
            const double shapeJ =
              shapeFunctionValuesTransposed[q * numDofsPerCell + cellDofIndexJ];

            const double gradShapeXI =
              shapeFunctionGradientValuesXTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexI];
            const double gradShapeYI =
              shapeFunctionGradientValuesYTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexI];
            const double gradShapeZI =
              shapeFunctionGradientValuesZTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexI];

            const double gradShapeXJ =
              shapeFunctionGradientValuesXTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexJ];
            const double gradShapeYJ =
              shapeFunctionGradientValuesYTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexJ];
            const double gradShapeZJ =
              shapeFunctionGradientValuesZTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexJ];


            val +=
              (vEffPrimeJxW[cellIndex * numQuadPoints + q]) * shapeI * shapeJ +
              2.0 * (derExcPrimeWithSigmaTimesGradRhoJxW[cellIndex *
                                                           numQuadPoints * 3 +
                                                         3 * q] *
                       (gradShapeXI * shapeJ + gradShapeXJ * shapeI) +
                     derExcPrimeWithSigmaTimesGradRhoJxW[cellIndex *
                                                           numQuadPoints * 3 +
                                                         3 * q + 1] *
                       (gradShapeYI * shapeJ + gradShapeYJ * shapeI) +
                     derExcPrimeWithSigmaTimesGradRhoJxW[cellIndex *
                                                           numQuadPoints * 3 +
                                                         3 * q + 2] *
                       (gradShapeZI * shapeJ + gradShapeZJ * shapeI));
          }

        cellHamiltonianPrimeMatrixFlattened[index] =
          make_cuDoubleComplex(val, 0.0);
      }
  }


  __global__ void
  hamPrimeMatrixKernelGGAMemOpt(
    const unsigned int numCells,
    const unsigned int numDofsPerCell,
    const unsigned int numQuadPoints,
    const double *     shapeFunctionValues,
    const double *     shapeFunctionValuesTransposed,
    const double *     shapeFunctionGradientValuesXTransposed,
    const double *     shapeFunctionGradientValuesYTransposed,
    const double *     shapeFunctionGradientValuesZTransposed,
    const double *     vEffPrimeJxW,
    const double *     JxW,
    const double *     derExcPrimeWithSigmaTimesGradRhoJxW,
    double *           cellHamiltonianPrimeMatrixFlattened)
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

        double val = 0.0;
        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            const double shapeI =
              shapeFunctionValues[cellDofIndexI * numQuadPoints + q];
            const double shapeJ =
              shapeFunctionValuesTransposed[q * numDofsPerCell + cellDofIndexJ];

            const double gradShapeXI =
              shapeFunctionGradientValuesXTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexI];
            const double gradShapeYI =
              shapeFunctionGradientValuesYTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexI];
            const double gradShapeZI =
              shapeFunctionGradientValuesZTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexI];

            const double gradShapeXJ =
              shapeFunctionGradientValuesXTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexJ];
            const double gradShapeYJ =
              shapeFunctionGradientValuesYTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexJ];
            const double gradShapeZJ =
              shapeFunctionGradientValuesZTransposed[cellIndex * numQuadPoints *
                                                       numDofsPerCell +
                                                     numDofsPerCell * q +
                                                     cellDofIndexJ];


            val +=
              (vEffPrimeJxW[cellIndex * numQuadPoints + q]) * shapeI * shapeJ +
              2.0 * (derExcPrimeWithSigmaTimesGradRhoJxW[cellIndex *
                                                           numQuadPoints * 3 +
                                                         3 * q] *
                       (gradShapeXI * shapeJ + gradShapeXJ * shapeI) +
                     derExcPrimeWithSigmaTimesGradRhoJxW[cellIndex *
                                                           numQuadPoints * 3 +
                                                         3 * q + 1] *
                       (gradShapeYI * shapeJ + gradShapeYJ * shapeI) +
                     derExcPrimeWithSigmaTimesGradRhoJxW[cellIndex *
                                                           numQuadPoints * 3 +
                                                         3 * q + 2] *
                       (gradShapeZI * shapeJ + gradShapeZJ * shapeI));
          }

        cellHamiltonianPrimeMatrixFlattened[index] = val;
      }
  }

} // namespace


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>::
  computeHamiltonianMatricesAllkpt(
    const unsigned int spinIndex,
    const bool         onlyHPrimePartForFirstOrderDensityMatResponse)
{
  dealii::TimerOutput computingTimerStandard(
    this->getMPICommunicator(),
    pcout,
    dftPtr->d_dftParamsPtr->reproducible_output ||
        dftPtr->d_dftParamsPtr->verbosity < 2 ?
      dealii::TimerOutput::never :
      dealii::TimerOutput::every_call,
    dealii::TimerOutput::wall_times);

  if (dftPtr->d_dftParamsPtr->deviceFineGrainedTimings)
    {
      cudaDeviceSynchronize();
      computingTimerStandard.enter_subsection(
        "Hamiltonian construction on Device");
    }


  if ((dftPtr->d_dftParamsPtr->isPseudopotential ||
       dftPtr->d_dftParamsPtr->smearedNuclearCharges) &&
      !d_isStiffnessMatrixExternalPotCorrComputed &&
      !onlyHPrimePartForFirstOrderDensityMatResponse)
    {
      hamMatrixExtPotCorr<<<(d_numLocallyOwnedCells * d_numberNodesPerElement *
                               d_numberNodesPerElement +
                             (deviceConstants::blockSize - 1)) /
                              deviceConstants::blockSize,
                            deviceConstants::blockSize>>>(
        d_numLocallyOwnedCells,
        d_numberNodesPerElement,
        d_numQuadPointsLpsp,
        thrust::raw_pointer_cast(&d_shapeFunctionValueLpspDevice[0]),
        thrust::raw_pointer_cast(&d_shapeFunctionValueTransposedLpspDevice[0]),
        thrust::raw_pointer_cast(&d_vEffExternalPotCorrJxWDevice[0]),
        thrust::raw_pointer_cast(
          &d_cellHamiltonianMatrixExternalPotCorrFlattenedDevice[0]));

      d_isStiffnessMatrixExternalPotCorrComputed = true;
    }

  if (onlyHPrimePartForFirstOrderDensityMatResponse)
    {
      if (dftPtr->excFunctionalPtr->getDensityBasedFamilyType() ==
          densityFamilyType::GGA)
        hamPrimeMatrixKernelGGAMemOpt<<<(d_numLocallyOwnedCells *
                                           d_numberNodesPerElement *
                                           d_numberNodesPerElement +
                                         (deviceConstants::blockSize - 1)) /
                                          deviceConstants::blockSize,
                                        deviceConstants::blockSize>>>(
          d_numLocallyOwnedCells,
          d_numberNodesPerElement,
          d_numQuadPoints,
          thrust::raw_pointer_cast(&d_shapeFunctionValueDevice[0]),
          thrust::raw_pointer_cast(&d_shapeFunctionValueTransposedDevice[0]),
          thrust::raw_pointer_cast(
            &d_shapeFunctionGradientValueXTransposedDevice[0]),
          thrust::raw_pointer_cast(
            &d_shapeFunctionGradientValueYTransposedDevice[0]),
          thrust::raw_pointer_cast(
            &d_shapeFunctionGradientValueZTransposedDevice[0]),
          thrust::raw_pointer_cast(&d_vEffJxWDevice[0]),
          thrust::raw_pointer_cast(&d_cellJxWValuesDevice[0]),
          thrust::raw_pointer_cast(&d_derExcWithSigmaTimesGradRhoJxWDevice[0]),
          reinterpret_cast<dataTypes::numberDevice *>(thrust::raw_pointer_cast(
            &d_cellHamiltonianMatrixFlattenedDevice[spinIndex *
                                                    d_numLocallyOwnedCells *
                                                    d_numberNodesPerElement *
                                                    d_numberNodesPerElement])));
      else if (dftPtr->excFunctionalPtr->getDensityBasedFamilyType() ==
               densityFamilyType::LDA)
        hamPrimeMatrixKernelLDA<<<(d_numLocallyOwnedCells *
                                     d_numberNodesPerElement *
                                     d_numberNodesPerElement +
                                   (deviceConstants::blockSize - 1)) /
                                    deviceConstants::blockSize,
                                  deviceConstants::blockSize>>>(
          d_numLocallyOwnedCells,
          d_numberNodesPerElement,
          d_numQuadPoints,
          thrust::raw_pointer_cast(&d_shapeFunctionValueDevice[0]),
          thrust::raw_pointer_cast(&d_shapeFunctionValueTransposedDevice[0]),
          thrust::raw_pointer_cast(
            &d_shapeFunctionGradientValueXTransposedDevice[0]),
          thrust::raw_pointer_cast(
            &d_shapeFunctionGradientValueYTransposedDevice[0]),
          thrust::raw_pointer_cast(
            &d_shapeFunctionGradientValueZTransposedDevice[0]),
          thrust::raw_pointer_cast(&d_vEffJxWDevice[0]),
          thrust::raw_pointer_cast(&d_cellJxWValuesDevice[0]),
          reinterpret_cast<dataTypes::numberDevice *>(thrust::raw_pointer_cast(
            &d_cellHamiltonianMatrixFlattenedDevice[spinIndex *
                                                    d_numLocallyOwnedCells *
                                                    d_numberNodesPerElement *
                                                    d_numberNodesPerElement])));
    }
  else
    {
      if (dftPtr->excFunctionalPtr->getDensityBasedFamilyType() ==
          densityFamilyType::GGA)
        hamMatrixKernelGGAMemOpt<<<(d_numLocallyOwnedCells *
                                      d_numberNodesPerElement *
                                      d_numberNodesPerElement +
                                    (deviceConstants::blockSize - 1)) /
                                     deviceConstants::blockSize,
                                   deviceConstants::blockSize>>>(
          d_numLocallyOwnedCells,
          d_numberNodesPerElement,
          d_numQuadPoints,
          spinIndex,
          (1 + dftPtr->d_dftParamsPtr->spinPolarized),
          dftPtr->d_kPointWeights.size(),
          thrust::raw_pointer_cast(&d_shapeFunctionValueDevice[0]),
          thrust::raw_pointer_cast(&d_shapeFunctionValueTransposedDevice[0]),
          thrust::raw_pointer_cast(
            &d_shapeFunctionGradientValueXTransposedDevice[0]),
          thrust::raw_pointer_cast(
            &d_shapeFunctionGradientValueYTransposedDevice[0]),
          thrust::raw_pointer_cast(
            &d_shapeFunctionGradientValueZTransposedDevice[0]),
          thrust::raw_pointer_cast(
            &d_cellShapeFunctionGradientIntegralFlattenedDevice[0]),
          thrust::raw_pointer_cast(&d_vEffJxWDevice[0]),
          thrust::raw_pointer_cast(&d_cellJxWValuesDevice[0]),
          thrust::raw_pointer_cast(&d_derExcWithSigmaTimesGradRhoJxWDevice[0]),
          thrust::raw_pointer_cast(
            &d_cellHamiltonianMatrixExternalPotCorrFlattenedDevice[0]),
          reinterpret_cast<dataTypes::numberDevice *>(thrust::raw_pointer_cast(
            &d_cellHamiltonianMatrixFlattenedDevice[0])),
          thrust::raw_pointer_cast(&d_kpointCoordsVecDevice[0]),
          thrust::raw_pointer_cast(&d_kSquareTimesHalfVecDevice[0]),
          dftPtr->d_dftParamsPtr->isPseudopotential ||
            dftPtr->d_dftParamsPtr->smearedNuclearCharges);
      else if (dftPtr->excFunctionalPtr->getDensityBasedFamilyType() ==
               densityFamilyType::LDA)
        hamMatrixKernelLDA<<<(d_numLocallyOwnedCells * d_numberNodesPerElement *
                                d_numberNodesPerElement +
                              (deviceConstants::blockSize - 1)) /
                               deviceConstants::blockSize,
                             deviceConstants::blockSize>>>(
          d_numLocallyOwnedCells,
          d_numberNodesPerElement,
          d_numQuadPoints,
          spinIndex,
          (1 + dftPtr->d_dftParamsPtr->spinPolarized),
          dftPtr->d_kPointWeights.size(),
          thrust::raw_pointer_cast(&d_shapeFunctionValueDevice[0]),
          thrust::raw_pointer_cast(&d_shapeFunctionValueTransposedDevice[0]),
          thrust::raw_pointer_cast(
            &d_shapeFunctionGradientValueXTransposedDevice[0]),
          thrust::raw_pointer_cast(
            &d_shapeFunctionGradientValueYTransposedDevice[0]),
          thrust::raw_pointer_cast(
            &d_shapeFunctionGradientValueZTransposedDevice[0]),
          thrust::raw_pointer_cast(
            &d_cellShapeFunctionGradientIntegralFlattenedDevice[0]),
          thrust::raw_pointer_cast(&d_vEffJxWDevice[0]),
          thrust::raw_pointer_cast(&d_cellJxWValuesDevice[0]),
          thrust::raw_pointer_cast(
            &d_cellHamiltonianMatrixExternalPotCorrFlattenedDevice[0]),
          reinterpret_cast<dataTypes::numberDevice *>(thrust::raw_pointer_cast(
            &d_cellHamiltonianMatrixFlattenedDevice[0])),
          thrust::raw_pointer_cast(&d_kpointCoordsVecDevice[0]),
          thrust::raw_pointer_cast(&d_kSquareTimesHalfVecDevice[0]),
          dftPtr->d_dftParamsPtr->isPseudopotential ||
            dftPtr->d_dftParamsPtr->smearedNuclearCharges);
    }


  if (dftPtr->d_dftParamsPtr->deviceFineGrainedTimings)
    {
      cudaDeviceSynchronize();
      computingTimerStandard.leave_subsection(
        "Hamiltonian construction on Device");
    }
}
