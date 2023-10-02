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
#pragma unroll
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
    const double *     shapeFunctionGradientValues,
    const double *     inverseJacobianValues,
    const int          areAllCellsAffineOrCartesianFlag,
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
#pragma unroll
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
    const double *     shapeFunctionGradientValues,
    const double *     inverseJacobianValues,
    const int          areAllCellsAffineOrCartesianFlag,
    const double *     cellShapeFunctionGradientIntegral,
    const double *     vEffJxW,
    const double *     JxW,
    const double *     cellHamiltonianMatrixExternalPotCorrFlattened,
    dftfe::utils::deviceDoubleComplex *cellHamiltonianMatrixFlattened,
    const double *                     kPointCoordsVec,
    const double *                     kSquareTimesHalfVec,
    const bool                         externalPotCorr)
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

#pragma unroll
        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            const double shapeI =
              shapeFunctionValues[cellDofIndexI * numQuadPoints + q];
            const double shapeJ =
              shapeFunctionValuesTransposed[q * numDofsPerCell + cellDofIndexJ];

            double gradShapeXI, gradShapeXJ, gradShapeYI, gradShapeYJ,
              gradShapeZI, gradShapeZJ;
            const double gradShapeXIRef =
              shapeFunctionGradientValues[numDofsPerCell * q + cellDofIndexI];
            const double gradShapeYIRef =
              shapeFunctionGradientValues[numDofsPerCell * numQuadPoints +
                                          numDofsPerCell * q + cellDofIndexI];
            const double gradShapeZIRef =
              shapeFunctionGradientValues[numDofsPerCell * numQuadPoints * 2 +
                                          numDofsPerCell * q + cellDofIndexI];
            if (areAllCellsAffineOrCartesianFlag == 0)
              {
                const double Jxx =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        0];
                const double Jxy =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        1];
                const double Jxz =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        2];
                const double Jyx =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        3];
                const double Jyy =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        4];
                const double Jyz =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        5];
                const double Jzx =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        6];
                const double Jzy =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        7];
                const double Jzz =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        8];

                gradShapeXI = gradShapeXIRef * Jxx + gradShapeYIRef * Jxy +
                              gradShapeZIRef * Jxz;
                gradShapeYI = gradShapeXIRef * Jyx + gradShapeYIRef * Jyy +
                              gradShapeZIRef * Jyz;
                gradShapeZI = gradShapeXIRef * Jzx + gradShapeYIRef * Jzy +
                              gradShapeZIRef * Jzz;
              }
            else if (areAllCellsAffineOrCartesianFlag == 1)
              {
                const double Jxx = inverseJacobianValues[cellIndex * 9 + 0];
                const double Jxy = inverseJacobianValues[cellIndex * 9 + 1];
                const double Jxz = inverseJacobianValues[cellIndex * 9 + 2];
                const double Jyx = inverseJacobianValues[cellIndex * 9 + 3];
                const double Jyy = inverseJacobianValues[cellIndex * 9 + 4];
                const double Jyz = inverseJacobianValues[cellIndex * 9 + 5];
                const double Jzx = inverseJacobianValues[cellIndex * 9 + 6];
                const double Jzy = inverseJacobianValues[cellIndex * 9 + 7];
                const double Jzz = inverseJacobianValues[cellIndex * 9 + 8];

                gradShapeXI = gradShapeXIRef * Jxx + gradShapeYIRef * Jxy +
                              gradShapeZIRef * Jxz;
                gradShapeYI = gradShapeXIRef * Jyx + gradShapeYIRef * Jyy +
                              gradShapeZIRef * Jyz;
                gradShapeZI = gradShapeXIRef * Jzx + gradShapeYIRef * Jzy +
                              gradShapeZIRef * Jzz;
              }
            else if (areAllCellsAffineOrCartesianFlag == 2)
              {
                const double Jxx = inverseJacobianValues[cellIndex * 3 + 0];
                const double Jyy = inverseJacobianValues[cellIndex * 3 + 1];
                const double Jzz = inverseJacobianValues[cellIndex * 3 + 2];

                gradShapeXI = gradShapeXIRef * Jxx;
                gradShapeYI = gradShapeYIRef * Jyy;
                gradShapeZI = gradShapeZIRef * Jzz;
              }

            val += vEffJxW[cellIndex * numQuadPoints + q] * shapeI * shapeJ;

            valRealKpt += JxW[cellIndex * numQuadPoints + q] * shapeI * shapeJ;

            valImagKptX -=
              gradShapeXI * shapeJ * JxW[cellIndex * numQuadPoints + q];
            valImagKptY -=
              gradShapeYI * shapeJ * JxW[cellIndex * numQuadPoints + q];
            valImagKptZ -=
              gradShapeZI * shapeJ * JxW[cellIndex * numQuadPoints + q];
          }

#pragma unroll
        for (unsigned int ikpt = 0; ikpt < numkPoints; ++ikpt)
          {
            const unsigned int startIndex = (nspin * ikpt + spinIndex) *
                                            numCells * numDofsPerCell *
                                            numDofsPerCell;
            cellHamiltonianMatrixFlattened[startIndex + index] =
              dftfe::utils::makeComplex(
                0.5 * cellShapeFunctionGradientIntegral[index] + val +
                  kSquareTimesHalfVec[ikpt] * valRealKpt,
                kPointCoordsVec[3 * ikpt + 0] * valImagKptX +
                  kPointCoordsVec[3 * ikpt + 1] * valImagKptY +
                  kPointCoordsVec[3 * ikpt + 2] * valImagKptZ);
            if (externalPotCorr)
              cellHamiltonianMatrixFlattened[startIndex + index] =
                dftfe::utils::makeComplex(
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
    const double *     shapeFunctionGradientValues,
    const double *     inverseJacobianValues,
    const int          areAllCellsAffineOrCartesianFlag,
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
#pragma unroll
        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            const double shapeI =
              shapeFunctionValues[cellDofIndexI * numQuadPoints + q];
            const double shapeJ =
              shapeFunctionValuesTransposed[q * numDofsPerCell + cellDofIndexJ];

            double gradShapeXI, gradShapeXJ, gradShapeYI, gradShapeYJ,
              gradShapeZI, gradShapeZJ;
            const double gradShapeXIRef =
              shapeFunctionGradientValues[numDofsPerCell * q + cellDofIndexI];
            const double gradShapeYIRef =
              shapeFunctionGradientValues[numDofsPerCell * numQuadPoints +
                                          numDofsPerCell * q + cellDofIndexI];
            const double gradShapeZIRef =
              shapeFunctionGradientValues[numDofsPerCell * numQuadPoints * 2 +
                                          numDofsPerCell * q + cellDofIndexI];
            const double gradShapeXJRef =
              shapeFunctionGradientValues[numDofsPerCell * q + cellDofIndexJ];
            const double gradShapeYJRef =
              shapeFunctionGradientValues[numDofsPerCell * numQuadPoints +
                                          numDofsPerCell * q + cellDofIndexJ];
            const double gradShapeZJRef =
              shapeFunctionGradientValues[numDofsPerCell * numQuadPoints * 2 +
                                          numDofsPerCell * q + cellDofIndexJ];
            if (areAllCellsAffineOrCartesianFlag == 0)
              {
                const double Jxx =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        0];
                const double Jxy =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        1];
                const double Jxz =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        2];
                const double Jyx =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        3];
                const double Jyy =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        4];
                const double Jyz =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        5];
                const double Jzx =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        6];
                const double Jzy =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        7];
                const double Jzz =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        8];

                gradShapeXI = gradShapeXIRef * Jxx + gradShapeYIRef * Jxy +
                              gradShapeZIRef * Jxz;
                gradShapeYI = gradShapeXIRef * Jyx + gradShapeYIRef * Jyy +
                              gradShapeZIRef * Jyz;
                gradShapeZI = gradShapeXIRef * Jzx + gradShapeYIRef * Jzy +
                              gradShapeZIRef * Jzz;
                gradShapeXJ = gradShapeXJRef * Jxx + gradShapeYJRef * Jxy +
                              gradShapeZJRef * Jxz;
                gradShapeYJ = gradShapeXJRef * Jyx + gradShapeYJRef * Jyy +
                              gradShapeZJRef * Jyz;
                gradShapeZJ = gradShapeXJRef * Jzx + gradShapeYJRef * Jzy +
                              gradShapeZJRef * Jzz;
              }
            else if (areAllCellsAffineOrCartesianFlag == 1)
              {
                const double Jxx = inverseJacobianValues[cellIndex * 9 + 0];
                const double Jxy = inverseJacobianValues[cellIndex * 9 + 1];
                const double Jxz = inverseJacobianValues[cellIndex * 9 + 2];
                const double Jyx = inverseJacobianValues[cellIndex * 9 + 3];
                const double Jyy = inverseJacobianValues[cellIndex * 9 + 4];
                const double Jyz = inverseJacobianValues[cellIndex * 9 + 5];
                const double Jzx = inverseJacobianValues[cellIndex * 9 + 6];
                const double Jzy = inverseJacobianValues[cellIndex * 9 + 7];
                const double Jzz = inverseJacobianValues[cellIndex * 9 + 8];

                gradShapeXI = gradShapeXIRef * Jxx + gradShapeYIRef * Jxy +
                              gradShapeZIRef * Jxz;
                gradShapeYI = gradShapeXIRef * Jyx + gradShapeYIRef * Jyy +
                              gradShapeZIRef * Jyz;
                gradShapeZI = gradShapeXIRef * Jzx + gradShapeYIRef * Jzy +
                              gradShapeZIRef * Jzz;
                gradShapeXJ = gradShapeXJRef * Jxx + gradShapeYJRef * Jxy +
                              gradShapeZJRef * Jxz;
                gradShapeYJ = gradShapeXJRef * Jyx + gradShapeYJRef * Jyy +
                              gradShapeZJRef * Jyz;
                gradShapeZJ = gradShapeXJRef * Jzx + gradShapeYJRef * Jzy +
                              gradShapeZJRef * Jzz;
              }
            else if (areAllCellsAffineOrCartesianFlag == 2)
              {
                const double Jxx = inverseJacobianValues[cellIndex * 3 + 0];
                const double Jyy = inverseJacobianValues[cellIndex * 3 + 1];
                const double Jzz = inverseJacobianValues[cellIndex * 3 + 2];

                gradShapeXI = gradShapeXIRef * Jxx;
                gradShapeYI = gradShapeYIRef * Jyy;
                gradShapeZI = gradShapeZIRef * Jzz;
                gradShapeXJ = gradShapeXJRef * Jxx;
                gradShapeYJ = gradShapeYJRef * Jyy;
                gradShapeZJ = gradShapeZJRef * Jzz;
              }


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
    const double *     shapeFunctionGradientValues,
    const double *     inverseJacobianValues,
    const int          areAllCellsAffineOrCartesianFlag,
    const double *     cellShapeFunctionGradientIntegral,
    const double *     vEffJxW,
    const double *     JxW,
    const double *     derExcWithSigmaTimesGradRhoJxW,
    const double *     cellHamiltonianMatrixExternalPotCorrFlattened,
    dftfe::utils::deviceDoubleComplex *cellHamiltonianMatrixFlattened,
    const double *                     kPointCoordsVec,
    const double *                     kSquareTimesHalfVec,
    const bool                         externalPotCorr)
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

#pragma unroll
        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            const double shapeI =
              shapeFunctionValues[cellDofIndexI * numQuadPoints + q];
            const double shapeJ =
              shapeFunctionValuesTransposed[q * numDofsPerCell + cellDofIndexJ];

            double gradShapeXI, gradShapeXJ, gradShapeYI, gradShapeYJ,
              gradShapeZI, gradShapeZJ;
            const double gradShapeXIRef =
              shapeFunctionGradientValues[numDofsPerCell * q + cellDofIndexI];
            const double gradShapeYIRef =
              shapeFunctionGradientValues[numDofsPerCell * numQuadPoints +
                                          numDofsPerCell * q + cellDofIndexI];
            const double gradShapeZIRef =
              shapeFunctionGradientValues[numDofsPerCell * numQuadPoints * 2 +
                                          numDofsPerCell * q + cellDofIndexI];
            const double gradShapeXJRef =
              shapeFunctionGradientValues[numDofsPerCell * q + cellDofIndexJ];
            const double gradShapeYJRef =
              shapeFunctionGradientValues[numDofsPerCell * numQuadPoints +
                                          numDofsPerCell * q + cellDofIndexJ];
            const double gradShapeZJRef =
              shapeFunctionGradientValues[numDofsPerCell * numQuadPoints * 2 +
                                          numDofsPerCell * q + cellDofIndexJ];
            if (areAllCellsAffineOrCartesianFlag == 0)
              {
                const double Jxx =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        0];
                const double Jxy =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        1];
                const double Jxz =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        2];
                const double Jyx =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        3];
                const double Jyy =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        4];
                const double Jyz =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        5];
                const double Jzx =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        6];
                const double Jzy =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        7];
                const double Jzz =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        8];

                gradShapeXI = gradShapeXIRef * Jxx + gradShapeYIRef * Jxy +
                              gradShapeZIRef * Jxz;
                gradShapeYI = gradShapeXIRef * Jyx + gradShapeYIRef * Jyy +
                              gradShapeZIRef * Jyz;
                gradShapeZI = gradShapeXIRef * Jzx + gradShapeYIRef * Jzy +
                              gradShapeZIRef * Jzz;
                gradShapeXJ = gradShapeXJRef * Jxx + gradShapeYJRef * Jxy +
                              gradShapeZJRef * Jxz;
                gradShapeYJ = gradShapeXJRef * Jyx + gradShapeYJRef * Jyy +
                              gradShapeZJRef * Jyz;
                gradShapeZJ = gradShapeXJRef * Jzx + gradShapeYJRef * Jzy +
                              gradShapeZJRef * Jzz;
              }
            else if (areAllCellsAffineOrCartesianFlag == 1)
              {
                const double Jxx = inverseJacobianValues[cellIndex * 9 + 0];
                const double Jxy = inverseJacobianValues[cellIndex * 9 + 1];
                const double Jxz = inverseJacobianValues[cellIndex * 9 + 2];
                const double Jyx = inverseJacobianValues[cellIndex * 9 + 3];
                const double Jyy = inverseJacobianValues[cellIndex * 9 + 4];
                const double Jyz = inverseJacobianValues[cellIndex * 9 + 5];
                const double Jzx = inverseJacobianValues[cellIndex * 9 + 6];
                const double Jzy = inverseJacobianValues[cellIndex * 9 + 7];
                const double Jzz = inverseJacobianValues[cellIndex * 9 + 8];

                gradShapeXI = gradShapeXIRef * Jxx + gradShapeYIRef * Jxy +
                              gradShapeZIRef * Jxz;
                gradShapeYI = gradShapeXIRef * Jyx + gradShapeYIRef * Jyy +
                              gradShapeZIRef * Jyz;
                gradShapeZI = gradShapeXIRef * Jzx + gradShapeYIRef * Jzy +
                              gradShapeZIRef * Jzz;
                gradShapeXJ = gradShapeXJRef * Jxx + gradShapeYJRef * Jxy +
                              gradShapeZJRef * Jxz;
                gradShapeYJ = gradShapeXJRef * Jyx + gradShapeYJRef * Jyy +
                              gradShapeZJRef * Jyz;
                gradShapeZJ = gradShapeXJRef * Jzx + gradShapeYJRef * Jzy +
                              gradShapeZJRef * Jzz;
              }
            else if (areAllCellsAffineOrCartesianFlag == 2)
              {
                const double Jxx = inverseJacobianValues[cellIndex * 3 + 0];
                const double Jyy = inverseJacobianValues[cellIndex * 3 + 1];
                const double Jzz = inverseJacobianValues[cellIndex * 3 + 2];

                gradShapeXI = gradShapeXIRef * Jxx;
                gradShapeYI = gradShapeYIRef * Jyy;
                gradShapeZI = gradShapeZIRef * Jzz;
                gradShapeXJ = gradShapeXJRef * Jxx;
                gradShapeYJ = gradShapeYJRef * Jyy;
                gradShapeZJ = gradShapeZJRef * Jzz;
              }


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

#pragma unroll
        for (unsigned int ikpt = 0; ikpt < numkPoints; ++ikpt)
          {
            const unsigned int startIndex = (nspin * ikpt + spinIndex) *
                                            numCells * numDofsPerCell *
                                            numDofsPerCell;
            cellHamiltonianMatrixFlattened[startIndex + index] =
              dftfe::utils::makeComplex(
                0.5 * cellShapeFunctionGradientIntegral[index] + val +
                  kSquareTimesHalfVec[ikpt] * valRealKpt,
                kPointCoordsVec[3 * ikpt + 0] * valImagKptX +
                  kPointCoordsVec[3 * ikpt + 1] * valImagKptY +
                  kPointCoordsVec[3 * ikpt + 2] * valImagKptZ);
            if (externalPotCorr)
              cellHamiltonianMatrixFlattened[startIndex + index] =
                dftfe::utils::makeComplex(
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
                          const double *     shapeFunctionGradientValues,
                          const double *     inverseJacobianValues,
                          const int          areAllCellsAffineOrCartesianFlag,
                          const double *     vEffPrimeJxW,
                          const double *     JxW,
                          double *cellHamiltonianPrimeMatrixFlattened)
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
#pragma unroll
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
  hamPrimeMatrixKernelLDA(
    const unsigned int                 numCells,
    const unsigned int                 numDofsPerCell,
    const unsigned int                 numQuadPoints,
    const double *                     shapeFunctionValues,
    const double *                     shapeFunctionValuesTransposed,
    const double *                     shapeFunctionGradientValues,
    const double *                     inverseJacobianValues,
    const int                          areAllCellsAffineOrCartesianFlag,
    const double *                     vEffPrimeJxW,
    const double *                     JxW,
    dftfe::utils::deviceDoubleComplex *cellHamiltonianPrimeMatrixFlattened)
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
#pragma unroll
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
          dftfe::utils::makeComplex(val, 0.0);
      }
  }

  __global__ void
  hamPrimeMatrixKernelGGAMemOpt(
    const unsigned int                 numCells,
    const unsigned int                 numDofsPerCell,
    const unsigned int                 numQuadPoints,
    const double *                     shapeFunctionValues,
    const double *                     shapeFunctionValuesTransposed,
    const double *                     shapeFunctionGradientValues,
    const double *                     inverseJacobianValues,
    const int                          areAllCellsAffineOrCartesianFlag,
    const double *                     vEffPrimeJxW,
    const double *                     JxW,
    const double *                     derExcPrimeWithSigmaTimesGradRhoJxW,
    dftfe::utils::deviceDoubleComplex *cellHamiltonianPrimeMatrixFlattened)
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
#pragma unroll
        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            const double shapeI =
              shapeFunctionValues[cellDofIndexI * numQuadPoints + q];
            const double shapeJ =
              shapeFunctionValuesTransposed[q * numDofsPerCell + cellDofIndexJ];

            double gradShapeXI, gradShapeXJ, gradShapeYI, gradShapeYJ,
              gradShapeZI, gradShapeZJ;
            const double gradShapeXIRef =
              shapeFunctionGradientValues[numDofsPerCell * q + cellDofIndexI];
            const double gradShapeYIRef =
              shapeFunctionGradientValues[numDofsPerCell * numQuadPoints +
                                          numDofsPerCell * q + cellDofIndexI];
            const double gradShapeZIRef =
              shapeFunctionGradientValues[numDofsPerCell * numQuadPoints * 2 +
                                          numDofsPerCell * q + cellDofIndexI];
            const double gradShapeXJRef =
              shapeFunctionGradientValues[numDofsPerCell * q + cellDofIndexJ];
            const double gradShapeYJRef =
              shapeFunctionGradientValues[numDofsPerCell * numQuadPoints +
                                          numDofsPerCell * q + cellDofIndexJ];
            const double gradShapeZJRef =
              shapeFunctionGradientValues[numDofsPerCell * numQuadPoints * 2 +
                                          numDofsPerCell * q + cellDofIndexJ];
            if (areAllCellsAffineOrCartesianFlag == 0)
              {
                const double Jxx =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        0];
                const double Jxy =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        1];
                const double Jxz =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        2];
                const double Jyx =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        3];
                const double Jyy =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        4];
                const double Jyz =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        5];
                const double Jzx =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        6];
                const double Jzy =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        7];
                const double Jzz =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        8];

                gradShapeXI = gradShapeXIRef * Jxx + gradShapeYIRef * Jxy +
                              gradShapeZIRef * Jxz;
                gradShapeYI = gradShapeXIRef * Jyx + gradShapeYIRef * Jyy +
                              gradShapeZIRef * Jyz;
                gradShapeZI = gradShapeXIRef * Jzx + gradShapeYIRef * Jzy +
                              gradShapeZIRef * Jzz;
                gradShapeXJ = gradShapeXJRef * Jxx + gradShapeYJRef * Jxy +
                              gradShapeZJRef * Jxz;
                gradShapeYJ = gradShapeXJRef * Jyx + gradShapeYJRef * Jyy +
                              gradShapeZJRef * Jyz;
                gradShapeZJ = gradShapeXJRef * Jzx + gradShapeYJRef * Jzy +
                              gradShapeZJRef * Jzz;
              }
            else if (areAllCellsAffineOrCartesianFlag == 1)
              {
                const double Jxx = inverseJacobianValues[cellIndex * 9 + 0];
                const double Jxy = inverseJacobianValues[cellIndex * 9 + 1];
                const double Jxz = inverseJacobianValues[cellIndex * 9 + 2];
                const double Jyx = inverseJacobianValues[cellIndex * 9 + 3];
                const double Jyy = inverseJacobianValues[cellIndex * 9 + 4];
                const double Jyz = inverseJacobianValues[cellIndex * 9 + 5];
                const double Jzx = inverseJacobianValues[cellIndex * 9 + 6];
                const double Jzy = inverseJacobianValues[cellIndex * 9 + 7];
                const double Jzz = inverseJacobianValues[cellIndex * 9 + 8];

                gradShapeXI = gradShapeXIRef * Jxx + gradShapeYIRef * Jxy +
                              gradShapeZIRef * Jxz;
                gradShapeYI = gradShapeXIRef * Jyx + gradShapeYIRef * Jyy +
                              gradShapeZIRef * Jyz;
                gradShapeZI = gradShapeXIRef * Jzx + gradShapeYIRef * Jzy +
                              gradShapeZIRef * Jzz;
                gradShapeXJ = gradShapeXJRef * Jxx + gradShapeYJRef * Jxy +
                              gradShapeZJRef * Jxz;
                gradShapeYJ = gradShapeXJRef * Jyx + gradShapeYJRef * Jyy +
                              gradShapeZJRef * Jyz;
                gradShapeZJ = gradShapeXJRef * Jzx + gradShapeYJRef * Jzy +
                              gradShapeZJRef * Jzz;
              }
            else if (areAllCellsAffineOrCartesianFlag == 2)
              {
                const double Jxx = inverseJacobianValues[cellIndex * 3 + 0];
                const double Jyy = inverseJacobianValues[cellIndex * 3 + 1];
                const double Jzz = inverseJacobianValues[cellIndex * 3 + 2];

                gradShapeXI = gradShapeXIRef * Jxx;
                gradShapeYI = gradShapeYIRef * Jyy;
                gradShapeZI = gradShapeZIRef * Jzz;
                gradShapeXJ = gradShapeXJRef * Jxx;
                gradShapeYJ = gradShapeYJRef * Jyy;
                gradShapeZJ = gradShapeZJRef * Jzz;
              }


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
          dftfe::utils::makeComplex(val, 0.0);
      }
  }


  __global__ void
  hamPrimeMatrixKernelGGAMemOpt(
    const unsigned int numCells,
    const unsigned int numDofsPerCell,
    const unsigned int numQuadPoints,
    const double *     shapeFunctionValues,
    const double *     shapeFunctionValuesTransposed,
    const double *     shapeFunctionGradientValues,
    const double *     inverseJacobianValues,
    const int          areAllCellsAffineOrCartesianFlag,
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
#pragma unroll
        for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            const double shapeI =
              shapeFunctionValues[cellDofIndexI * numQuadPoints + q];
            const double shapeJ =
              shapeFunctionValuesTransposed[q * numDofsPerCell + cellDofIndexJ];

            double gradShapeXI, gradShapeXJ, gradShapeYI, gradShapeYJ,
              gradShapeZI, gradShapeZJ;
            const double gradShapeXIRef =
              shapeFunctionGradientValues[numDofsPerCell * q + cellDofIndexI];
            const double gradShapeYIRef =
              shapeFunctionGradientValues[numDofsPerCell * numQuadPoints +
                                          numDofsPerCell * q + cellDofIndexI];
            const double gradShapeZIRef =
              shapeFunctionGradientValues[numDofsPerCell * numQuadPoints * 2 +
                                          numDofsPerCell * q + cellDofIndexI];
            const double gradShapeXJRef =
              shapeFunctionGradientValues[numDofsPerCell * q + cellDofIndexJ];
            const double gradShapeYJRef =
              shapeFunctionGradientValues[numDofsPerCell * numQuadPoints +
                                          numDofsPerCell * q + cellDofIndexJ];
            const double gradShapeZJRef =
              shapeFunctionGradientValues[numDofsPerCell * numQuadPoints * 2 +
                                          numDofsPerCell * q + cellDofIndexJ];
            if (areAllCellsAffineOrCartesianFlag == 0)
              {
                const double Jxx =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        0];
                const double Jxy =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        1];
                const double Jxz =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        2];
                const double Jyx =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        3];
                const double Jyy =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        4];
                const double Jyz =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        5];
                const double Jzx =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        6];
                const double Jzy =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        7];
                const double Jzz =
                  inverseJacobianValues[cellIndex * numQuadPoints * 9 + q * 9 +
                                        8];

                gradShapeXI = gradShapeXIRef * Jxx + gradShapeYIRef * Jxy +
                              gradShapeZIRef * Jxz;
                gradShapeYI = gradShapeXIRef * Jyx + gradShapeYIRef * Jyy +
                              gradShapeZIRef * Jyz;
                gradShapeZI = gradShapeXIRef * Jzx + gradShapeYIRef * Jzy +
                              gradShapeZIRef * Jzz;
                gradShapeXJ = gradShapeXJRef * Jxx + gradShapeYJRef * Jxy +
                              gradShapeZJRef * Jxz;
                gradShapeYJ = gradShapeXJRef * Jyx + gradShapeYJRef * Jyy +
                              gradShapeZJRef * Jyz;
                gradShapeZJ = gradShapeXJRef * Jzx + gradShapeYJRef * Jzy +
                              gradShapeZJRef * Jzz;
              }
            else if (areAllCellsAffineOrCartesianFlag == 1)
              {
                const double Jxx = inverseJacobianValues[cellIndex * 9 + 0];
                const double Jxy = inverseJacobianValues[cellIndex * 9 + 1];
                const double Jxz = inverseJacobianValues[cellIndex * 9 + 2];
                const double Jyx = inverseJacobianValues[cellIndex * 9 + 3];
                const double Jyy = inverseJacobianValues[cellIndex * 9 + 4];
                const double Jyz = inverseJacobianValues[cellIndex * 9 + 5];
                const double Jzx = inverseJacobianValues[cellIndex * 9 + 6];
                const double Jzy = inverseJacobianValues[cellIndex * 9 + 7];
                const double Jzz = inverseJacobianValues[cellIndex * 9 + 8];

                gradShapeXI = gradShapeXIRef * Jxx + gradShapeYIRef * Jxy +
                              gradShapeZIRef * Jxz;
                gradShapeYI = gradShapeXIRef * Jyx + gradShapeYIRef * Jyy +
                              gradShapeZIRef * Jyz;
                gradShapeZI = gradShapeXIRef * Jzx + gradShapeYIRef * Jzy +
                              gradShapeZIRef * Jzz;
                gradShapeXJ = gradShapeXJRef * Jxx + gradShapeYJRef * Jxy +
                              gradShapeZJRef * Jxz;
                gradShapeYJ = gradShapeXJRef * Jyx + gradShapeYJRef * Jyy +
                              gradShapeZJRef * Jyz;
                gradShapeZJ = gradShapeXJRef * Jzx + gradShapeYJRef * Jzy +
                              gradShapeZJRef * Jzz;
              }
            else if (areAllCellsAffineOrCartesianFlag == 2)
              {
                const double Jxx = inverseJacobianValues[cellIndex * 3 + 0];
                const double Jyy = inverseJacobianValues[cellIndex * 3 + 1];
                const double Jzz = inverseJacobianValues[cellIndex * 3 + 2];

                gradShapeXI = gradShapeXIRef * Jxx;
                gradShapeYI = gradShapeYIRef * Jyy;
                gradShapeZI = gradShapeZIRef * Jzz;
                gradShapeXJ = gradShapeXJRef * Jxx;
                gradShapeYJ = gradShapeYJRef * Jyy;
                gradShapeZJ = gradShapeZJRef * Jzz;
              }


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
      dftfe::utils::deviceSynchronize();
      computingTimerStandard.enter_subsection(
        "Hamiltonian construction on Device");
    }


  if ((dftPtr->d_dftParamsPtr->isPseudopotential ||
       dftPtr->d_dftParamsPtr->smearedNuclearCharges) &&
      !d_isStiffnessMatrixExternalPotCorrComputed &&
      !onlyHPrimePartForFirstOrderDensityMatResponse)
    {
      basisOperationsPtrDevice->reinit(0, 0, dftPtr->d_lpspQuadratureId);
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      hamMatrixExtPotCorr<<<(d_numLocallyOwnedCells * d_numberNodesPerElement *
                               d_numberNodesPerElement +
                             (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                              dftfe::utils::DEVICE_BLOCK_SIZE,
                            dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        basisOperationsPtrDevice->nCells(),
        basisOperationsPtrDevice->nDofsPerCell(),
        basisOperationsPtrDevice->nQuadsPerCell(),
        basisOperationsPtrDevice->shapeFunctionData(true),
        basisOperationsPtrDevice->shapeFunctionData(false),
        d_vEffExternalPotCorrJxWDevice.begin(),
        d_cellHamiltonianMatrixExternalPotCorrFlattenedDevice.begin());
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(
        hamMatrixExtPotCorr,
        (d_numLocallyOwnedCells * d_numberNodesPerElement *
           d_numberNodesPerElement +
         (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE,
        0,
        0,
        basisOperationsPtrDevice->nCells(),
        basisOperationsPtrDevice->nDofsPerCell(),
        basisOperationsPtrDevice->nQuadsPerCell(),
        basisOperationsPtrDevice->shapeFunctionData(true),
        basisOperationsPtrDevice->shapeFunctionData(false),
        d_vEffExternalPotCorrJxWDevice.begin(),
        d_cellHamiltonianMatrixExternalPotCorrFlattenedDevice.begin());
#endif

      d_isStiffnessMatrixExternalPotCorrComputed = true;
    }
  basisOperationsPtrDevice->reinit(0, 0, dftPtr->d_densityQuadratureId);
  if (onlyHPrimePartForFirstOrderDensityMatResponse)
    {
      if (dftPtr->d_excManagerPtr->getDensityBasedFamilyType() ==
          densityFamilyType::GGA)
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
        hamPrimeMatrixKernelGGAMemOpt<<<
          (d_numLocallyOwnedCells * d_numberNodesPerElement *
             d_numberNodesPerElement +
           (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
            dftfe::utils::DEVICE_BLOCK_SIZE,
          dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          basisOperationsPtrDevice->nCells(),
          basisOperationsPtrDevice->nDofsPerCell(),
          basisOperationsPtrDevice->nQuadsPerCell(),
          basisOperationsPtrDevice->shapeFunctionData(true),
          basisOperationsPtrDevice->shapeFunctionData(false),
          basisOperationsPtrDevice->shapeFunctionGradientData(),
          basisOperationsPtrDevice->inverseJacobians(),
          basisOperationsPtrDevice->cellsTypeFlag(),
          d_vEffJxWDevice.begin(),
          d_cellJxWValuesDevice.begin(),
          d_derExcWithSigmaTimesGradRhoJxWDevice.begin(),
          dftfe::utils::makeDataTypeDeviceCompatible(
            d_cellHamiltonianMatrixFlattenedDevice.begin() +
            spinIndex * d_numLocallyOwnedCells * d_numberNodesPerElement *
              d_numberNodesPerElement));
#elif DFTFE_WITH_DEVICE_LANG_HIP
        hipLaunchKernelGGL(
          hamPrimeMatrixKernelGGAMemOpt,
          (d_numLocallyOwnedCells * d_numberNodesPerElement *
             d_numberNodesPerElement +
           (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
            dftfe::utils::DEVICE_BLOCK_SIZE,
          dftfe::utils::DEVICE_BLOCK_SIZE,
          0,
          0,
          basisOperationsPtrDevice->nCells(),
          basisOperationsPtrDevice->nDofsPerCell(),
          basisOperationsPtrDevice->nQuadsPerCell(),
          basisOperationsPtrDevice->shapeFunctionData(true),
          basisOperationsPtrDevice->shapeFunctionData(false),
          basisOperationsPtrDevice->shapeFunctionGradientData(),
          basisOperationsPtrDevice->inverseJacobians(),
          basisOperationsPtrDevice->cellsTypeFlag(),
          d_vEffJxWDevice.begin(),
          d_cellJxWValuesDevice.begin(),
          d_derExcWithSigmaTimesGradRhoJxWDevice.begin(),
          dftfe::utils::makeDataTypeDeviceCompatible(
            d_cellHamiltonianMatrixFlattenedDevice.begin() +
            spinIndex * d_numLocallyOwnedCells * d_numberNodesPerElement *
              d_numberNodesPerElement));
#endif
      else if (dftPtr->d_excManagerPtr->getDensityBasedFamilyType() ==
               densityFamilyType::LDA)
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
        hamPrimeMatrixKernelLDA<<<(d_numLocallyOwnedCells *
                                     d_numberNodesPerElement *
                                     d_numberNodesPerElement +
                                   (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                    dftfe::utils::DEVICE_BLOCK_SIZE,
                                  dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          basisOperationsPtrDevice->nCells(),
          basisOperationsPtrDevice->nDofsPerCell(),
          basisOperationsPtrDevice->nQuadsPerCell(),
          basisOperationsPtrDevice->shapeFunctionData(true),
          basisOperationsPtrDevice->shapeFunctionData(false),
          basisOperationsPtrDevice->shapeFunctionGradientData(),
          basisOperationsPtrDevice->inverseJacobians(),
          basisOperationsPtrDevice->cellsTypeFlag(),
          d_vEffJxWDevice.begin(),
          d_cellJxWValuesDevice.begin(),
          dftfe::utils::makeDataTypeDeviceCompatible(
            d_cellHamiltonianMatrixFlattenedDevice.begin() +
            spinIndex * d_numLocallyOwnedCells * d_numberNodesPerElement *
              d_numberNodesPerElement));
#elif DFTFE_WITH_DEVICE_LANG_HIP
        hipLaunchKernelGGL(
          hamPrimeMatrixKernelLDA,
          (d_numLocallyOwnedCells * d_numberNodesPerElement *
             d_numberNodesPerElement +
           (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
            dftfe::utils::DEVICE_BLOCK_SIZE,
          dftfe::utils::DEVICE_BLOCK_SIZE,
          0,
          0,
          basisOperationsPtrDevice->nCells(),
          basisOperationsPtrDevice->nDofsPerCell(),
          basisOperationsPtrDevice->nQuadsPerCell(),
          basisOperationsPtrDevice->shapeFunctionData(true),
          basisOperationsPtrDevice->shapeFunctionData(false),
          basisOperationsPtrDevice->shapeFunctionGradientData(),
          basisOperationsPtrDevice->inverseJacobians(),
          basisOperationsPtrDevice->cellsTypeFlag(),
          d_vEffJxWDevice.begin(),
          d_cellJxWValuesDevice.begin(),
          dftfe::utils::makeDataTypeDeviceCompatible(
            d_cellHamiltonianMatrixFlattenedDevice.begin() +
            spinIndex * d_numLocallyOwnedCells * d_numberNodesPerElement *
              d_numberNodesPerElement));
#endif
    }
  else
    {
      if (dftPtr->d_excManagerPtr->getDensityBasedFamilyType() ==
          densityFamilyType::GGA)
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
        hamMatrixKernelGGAMemOpt<<<(d_numLocallyOwnedCells *
                                      d_numberNodesPerElement *
                                      d_numberNodesPerElement +
                                    (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                     dftfe::utils::DEVICE_BLOCK_SIZE,
                                   dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          basisOperationsPtrDevice->nCells(),
          basisOperationsPtrDevice->nDofsPerCell(),
          basisOperationsPtrDevice->nQuadsPerCell(),
          spinIndex,
          (1 + dftPtr->d_dftParamsPtr->spinPolarized),
          dftPtr->d_kPointWeights.size(),
          basisOperationsPtrDevice->shapeFunctionData(true),
          basisOperationsPtrDevice->shapeFunctionData(false),
          basisOperationsPtrDevice->shapeFunctionGradientData(),
          basisOperationsPtrDevice->inverseJacobians(),
          basisOperationsPtrDevice->cellsTypeFlag(),
          d_cellShapeFunctionGradientIntegralFlattenedDevice.begin(),
          d_vEffJxWDevice.begin(),
          d_cellJxWValuesDevice.begin(),
          d_derExcWithSigmaTimesGradRhoJxWDevice.begin(),
          d_cellHamiltonianMatrixExternalPotCorrFlattenedDevice.begin(),
          dftfe::utils::makeDataTypeDeviceCompatible(
            d_cellHamiltonianMatrixFlattenedDevice.begin()),
          d_kpointCoordsVecDevice.begin(),
          d_kSquareTimesHalfVecDevice.begin(),
          dftPtr->d_dftParamsPtr->isPseudopotential ||
            dftPtr->d_dftParamsPtr->smearedNuclearCharges);
#elif DFTFE_WITH_DEVICE_LANG_HIP
        hipLaunchKernelGGL(
          hamMatrixKernelGGAMemOpt,
          (d_numLocallyOwnedCells * d_numberNodesPerElement *
             d_numberNodesPerElement +
           (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
            dftfe::utils::DEVICE_BLOCK_SIZE,
          dftfe::utils::DEVICE_BLOCK_SIZE,
          0,
          0,
          basisOperationsPtrDevice->nCells(),
          basisOperationsPtrDevice->nDofsPerCell(),
          basisOperationsPtrDevice->nQuadsPerCell(),
          spinIndex,
          (1 + dftPtr->d_dftParamsPtr->spinPolarized),
          dftPtr->d_kPointWeights.size(),
          basisOperationsPtrDevice->shapeFunctionData(true),
          basisOperationsPtrDevice->shapeFunctionData(false),
          basisOperationsPtrDevice->shapeFunctionGradientData(),
          basisOperationsPtrDevice->inverseJacobians(),
          basisOperationsPtrDevice->cellsTypeFlag(),
          d_cellShapeFunctionGradientIntegralFlattenedDevice.begin(),
          d_vEffJxWDevice.begin(),
          d_cellJxWValuesDevice.begin(),
          d_derExcWithSigmaTimesGradRhoJxWDevice.begin(),
          d_cellHamiltonianMatrixExternalPotCorrFlattenedDevice.begin(),
          dftfe::utils::makeDataTypeDeviceCompatible(
            d_cellHamiltonianMatrixFlattenedDevice.begin()),
          d_kpointCoordsVecDevice.begin(),
          d_kSquareTimesHalfVecDevice.begin(),
          dftPtr->d_dftParamsPtr->isPseudopotential ||
            dftPtr->d_dftParamsPtr->smearedNuclearCharges);
#endif
      else if (dftPtr->d_excManagerPtr->getDensityBasedFamilyType() ==
               densityFamilyType::LDA)
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
        hamMatrixKernelLDA<<<(d_numLocallyOwnedCells * d_numberNodesPerElement *
                                d_numberNodesPerElement +
                              (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                               dftfe::utils::DEVICE_BLOCK_SIZE,
                             dftfe::utils::DEVICE_BLOCK_SIZE>>>(
          basisOperationsPtrDevice->nCells(),
          basisOperationsPtrDevice->nDofsPerCell(),
          basisOperationsPtrDevice->nQuadsPerCell(),
          spinIndex,
          (1 + dftPtr->d_dftParamsPtr->spinPolarized),
          dftPtr->d_kPointWeights.size(),
          basisOperationsPtrDevice->shapeFunctionData(true),
          basisOperationsPtrDevice->shapeFunctionData(false),
          basisOperationsPtrDevice->shapeFunctionGradientData(),
          basisOperationsPtrDevice->inverseJacobians(),
          basisOperationsPtrDevice->cellsTypeFlag(),
          d_cellShapeFunctionGradientIntegralFlattenedDevice.begin(),
          d_vEffJxWDevice.begin(),
          d_cellJxWValuesDevice.begin(),
          d_cellHamiltonianMatrixExternalPotCorrFlattenedDevice.begin(),
          dftfe::utils::makeDataTypeDeviceCompatible(
            d_cellHamiltonianMatrixFlattenedDevice.begin()),
          d_kpointCoordsVecDevice.begin(),
          d_kSquareTimesHalfVecDevice.begin(),
          dftPtr->d_dftParamsPtr->isPseudopotential ||
            dftPtr->d_dftParamsPtr->smearedNuclearCharges);
#elif DFTFE_WITH_DEVICE_LANG_HIP
        hipLaunchKernelGGL(
          hamMatrixKernelLDA,
          (d_numLocallyOwnedCells * d_numberNodesPerElement *
             d_numberNodesPerElement +
           (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
            dftfe::utils::DEVICE_BLOCK_SIZE,
          dftfe::utils::DEVICE_BLOCK_SIZE,
          0,
          0,
          basisOperationsPtrDevice->nCells(),
          basisOperationsPtrDevice->nDofsPerCell(),
          basisOperationsPtrDevice->nQuadsPerCell(),
          spinIndex,
          (1 + dftPtr->d_dftParamsPtr->spinPolarized),
          dftPtr->d_kPointWeights.size(),
          basisOperationsPtrDevice->shapeFunctionData(true),
          basisOperationsPtrDevice->shapeFunctionData(false),
          basisOperationsPtrDevice->shapeFunctionGradientData(),
          basisOperationsPtrDevice->inverseJacobians(),
          basisOperationsPtrDevice->cellsTypeFlag(),
          d_cellShapeFunctionGradientIntegralFlattenedDevice.begin(),
          d_vEffJxWDevice.begin(),
          d_cellJxWValuesDevice.begin(),
          d_cellHamiltonianMatrixExternalPotCorrFlattenedDevice.begin(),
          dftfe::utils::makeDataTypeDeviceCompatible(
            d_cellHamiltonianMatrixFlattenedDevice.begin()),
          d_kpointCoordsVecDevice.begin(),
          d_kSquareTimesHalfVecDevice.begin(),
          dftPtr->d_dftParamsPtr->isPseudopotential ||
            dftPtr->d_dftParamsPtr->smearedNuclearCharges);
#endif
    }


  if (dftPtr->d_dftParamsPtr->deviceFineGrainedTimings)
    {
      dftfe::utils::deviceSynchronize();
      computingTimerStandard.leave_subsection(
        "Hamiltonian construction on Device");
    }
}
