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
// @author  Sambit Das
//


namespace
{
  __global__
  void hamMatrixKernelLDA(const unsigned int numCells,
                    const unsigned int numDofsPerCell,
                    const unsigned int numQuadPoints,
                    const double *shapeFunctionValues,
                    const double *shapeFunctionValuesInverted,
                    const double *cellShapeFunctionGradientIntegral,
                    const double *vEffJxW,
                    double *cellHamiltonianMatrixFlattened)
  {

     const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;

     for(unsigned int index = globalThreadId; index < numCells*numDofsPerCell*numDofsPerCell; index+= blockDim.x*gridDim.x)
     {
        const unsigned int cellIndex=index/(numDofsPerCell*numDofsPerCell);
        const unsigned int flattenedCellDofIndex=index%(numDofsPerCell*numDofsPerCell);
        const unsigned int cellDofIndexI=flattenedCellDofIndex/numDofsPerCell;
        const unsigned int cellDofIndexJ=flattenedCellDofIndex%numDofsPerCell;

        double val=0;
        for(unsigned int q = 0; q < numQuadPoints; ++q)
        {
             val+= vEffJxW[cellIndex*numQuadPoints+q]*shapeFunctionValues[cellDofIndexI*numQuadPoints+q]
                                 *shapeFunctionValuesInverted[q*numDofsPerCell+cellDofIndexJ];
        }
              
        cellHamiltonianMatrixFlattened[index] = 0.5*cellShapeFunctionGradientIntegral[index]+ val;
     }

  }


  __global__
  void hamMatrixKernelGGA(const unsigned int numCells,
                    const unsigned int numDofsPerCell,
                    const unsigned int numQuadPoints,
                    const double *shapeFunctionValues,
                    const double *shapeFunctionValuesInverted,
                    const double *shapeFunctionGradientValuesX,
                    const double *shapeFunctionGradientValuesY,
                    const double *shapeFunctionGradientValuesZ,
                    const double *shapeFunctionGradientValuesXInverted,
                    const double *shapeFunctionGradientValuesYInverted,
                    const double *shapeFunctionGradientValuesZInverted,
                    const double *cellShapeFunctionGradientIntegral,
                    const double *vEffJxW,
                    const double * derExcWithSigmaTimesGradRhoJxW,
                    double *cellHamiltonianMatrixFlattened)
  {

     const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;

     for(unsigned int index = globalThreadId; index < numCells*numDofsPerCell*numDofsPerCell; index+= blockDim.x*gridDim.x)
     {
        const unsigned int cellIndex=index/(numDofsPerCell*numDofsPerCell);
        const unsigned int flattenedCellDofIndex=index%(numDofsPerCell*numDofsPerCell);
        const unsigned int cellDofIndexI=flattenedCellDofIndex/numDofsPerCell;
        const unsigned int cellDofIndexJ=flattenedCellDofIndex%numDofsPerCell;

        double val=0;
        for(unsigned int q = 0; q < numQuadPoints; ++q)
        {
             const double shapeI=shapeFunctionValues[cellDofIndexI*numQuadPoints+q];
             const double shapeJ=shapeFunctionValuesInverted[q*numDofsPerCell+cellDofIndexJ];

             const double gradShapeXI=shapeFunctionGradientValuesX[cellIndex*numDofsPerCell*numQuadPoints
                                                                   +cellDofIndexI*numQuadPoints+q];
             const double gradShapeYI=shapeFunctionGradientValuesY[cellIndex*numDofsPerCell*numQuadPoints
                                                                   +cellDofIndexI*numQuadPoints+q];
             const double gradShapeZI=shapeFunctionGradientValuesZ[cellIndex*numDofsPerCell*numQuadPoints
                                                                   +cellDofIndexI*numQuadPoints+q];

             const double gradShapeXJ=shapeFunctionGradientValuesXInverted[cellIndex*numQuadPoints*numDofsPerCell
                                                                   +numDofsPerCell*q+cellDofIndexJ];
             const double gradShapeYJ=shapeFunctionGradientValuesYInverted[cellIndex*numQuadPoints*numDofsPerCell
                                                                   +numDofsPerCell*q+cellDofIndexJ];
             const double gradShapeZJ=shapeFunctionGradientValuesZInverted[cellIndex*numQuadPoints*numDofsPerCell
                                                                   +numDofsPerCell*q+cellDofIndexJ];


             val+= vEffJxW[cellIndex*numQuadPoints+q]*shapeI*shapeJ
                   +2.0*(derExcWithSigmaTimesGradRhoJxW[cellIndex*numQuadPoints*3+3*q]*(gradShapeXI*shapeJ+gradShapeXJ*shapeI)
                   +derExcWithSigmaTimesGradRhoJxW[cellIndex*numQuadPoints*3+3*q+1]*(gradShapeYI*shapeJ+gradShapeYJ*shapeI)
                   +derExcWithSigmaTimesGradRhoJxW[cellIndex*numQuadPoints*3+3*q+2]*(gradShapeZI*shapeJ+gradShapeZJ*shapeI));

/*
                      const Tensor<1,3, VectorizedArray<double> > tempVec =
                          nonCachedShapeGrad[iNode*numberQuadraturePoints+q_point]
                          *make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*jNode+q_point])
                          + nonCachedShapeGrad[jNode*numberQuadraturePoints+q_point]
                            *make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*iNode+q_point]);
*/

        }

        cellHamiltonianMatrixFlattened[index] = 0.5*cellShapeFunctionGradientIntegral[index]+ val;
     }

  }

}


template<unsigned int FEOrder>
void kohnShamDFTOperatorCUDAClass<FEOrder>::computeHamiltonianMatrix(unsigned int kPointIndex)
{
  d_cellHamiltonianMatrixFlattenedDevice.resize(d_numLocallyOwnedCells*d_numberNodesPerElement*d_numberNodesPerElement,0.0);
  double gpu_time=MPI_Wtime();
  if(dftParameters::xc_id == 4)
     hamMatrixKernelGGA<<<(d_numLocallyOwnedCells*d_numberNodesPerElement*d_numberNodesPerElement+255)/256,256>>>
                                               (d_numLocallyOwnedCells,
                                                d_numberNodesPerElement,
                                                d_numQuadPoints,
                                                thrust::raw_pointer_cast(&d_shapeFunctionValueDevice[0]),
                                                thrust::raw_pointer_cast(&d_shapeFunctionValueInvertedDevice[0]),
                                                thrust::raw_pointer_cast(&d_shapeFunctionGradientValueXDevice[0]),
                                                thrust::raw_pointer_cast(&d_shapeFunctionGradientValueYDevice[0]),
                                                thrust::raw_pointer_cast(&d_shapeFunctionGradientValueZDevice[0]),
                                                thrust::raw_pointer_cast(&d_shapeFunctionGradientValueXInvertedDevice[0]),
                                                thrust::raw_pointer_cast(&d_shapeFunctionGradientValueYInvertedDevice[0]),
                                                thrust::raw_pointer_cast(&d_shapeFunctionGradientValueZInvertedDevice[0]),
                                                thrust::raw_pointer_cast(&d_cellShapeFunctionGradientIntegralFlattenedDevice[0]),
                                                thrust::raw_pointer_cast(&d_vEffJxWDevice[0]),
                                                thrust::raw_pointer_cast(&d_derExcWithSigmaTimesGradRhoJxWDevice[0]),
                                                thrust::raw_pointer_cast(&d_cellHamiltonianMatrixFlattenedDevice[0]));
  else
     hamMatrixKernelLDA<<<(d_numLocallyOwnedCells*d_numberNodesPerElement*d_numberNodesPerElement+255)/256,256>>>
                                               (d_numLocallyOwnedCells,
                                                d_numberNodesPerElement,
                                                d_numQuadPoints,
                                                thrust::raw_pointer_cast(&d_shapeFunctionValueDevice[0]),
                                                thrust::raw_pointer_cast(&d_shapeFunctionValueInvertedDevice[0]),
                                                thrust::raw_pointer_cast(&d_cellShapeFunctionGradientIntegralFlattenedDevice[0]),
                                                thrust::raw_pointer_cast(&d_vEffJxWDevice[0]),
                                                thrust::raw_pointer_cast(&d_cellHamiltonianMatrixFlattenedDevice[0]));

  cudaDeviceSynchronize();
  gpu_time = MPI_Wtime() - gpu_time;
  if (dftParameters::verbosity>=2)
    pcout<<"Time for elemental Hamiltonian matrix computation on GPU: "<<gpu_time<<std::endl;
}


