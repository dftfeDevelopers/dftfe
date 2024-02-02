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
// the top level of the DFT-FE udistribution.
//
// ---------------------------------------------------------------------
//
// @author  Sambit Das
//

namespace shapeFuncDevice
{
  __global__ void
  computeShapeGradNINJIntegralContribution(
    const unsigned int numCells,
    const unsigned int numDofsPerCell,
    const unsigned int numQuadPoints,
    const double *     shapeFunctionGradientValues,
    const double *     inverseJacobianValues,
    const double *     JxW,
    const int          areAllCellsAffineOrCartesianFlag,
    double *           shapeGradNINJIntegralContribution)
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
            const double *jacobianPtr =
              inverseJacobianValues +
              (areAllCellsAffineOrCartesianFlag == 0 ?
                 cellIndex * numQuadPoints * 9 + q * 9 :
                 (areAllCellsAffineOrCartesianFlag == 1 ? cellIndex * 9 :
                                                          cellIndex * 3));

            if (areAllCellsAffineOrCartesianFlag == 2)
              {
                double gradShapeXI =
                  shapeFunctionGradientValues[numDofsPerCell * q +
                                              cellDofIndexI] *
                  jacobianPtr[0];
                double gradShapeYI =
                  shapeFunctionGradientValues[numDofsPerCell * numQuadPoints +
                                              numDofsPerCell * q +
                                              cellDofIndexI] *
                  jacobianPtr[1];
                double gradShapeZI =
                  shapeFunctionGradientValues[numDofsPerCell * numQuadPoints *
                                                2 +
                                              numDofsPerCell * q +
                                              cellDofIndexI] *
                  jacobianPtr[2];
                double gradShapeXJ =
                  shapeFunctionGradientValues[numDofsPerCell * q +
                                              cellDofIndexJ] *
                  jacobianPtr[0];
                double gradShapeYJ =
                  shapeFunctionGradientValues[numDofsPerCell * numQuadPoints +
                                              numDofsPerCell * q +
                                              cellDofIndexJ] *
                  jacobianPtr[1];
                double gradShapeZJ =
                  shapeFunctionGradientValues[numDofsPerCell * numQuadPoints *
                                                2 +
                                              numDofsPerCell * q +
                                              cellDofIndexJ] *
                  jacobianPtr[2];
                val +=
                  ((gradShapeXI * gradShapeXJ) + (gradShapeYI * gradShapeYJ) +
                   (gradShapeZI * gradShapeZJ)) *
                  JxW[cellIndex * numQuadPoints + q];
              }
            else
              {
                double gradShapeXI =
                  shapeFunctionGradientValues[numDofsPerCell * q +
                                              cellDofIndexI] *
                    jacobianPtr[0] +
                  shapeFunctionGradientValues[numDofsPerCell * numQuadPoints +
                                              numDofsPerCell * q +
                                              cellDofIndexI] *
                    jacobianPtr[1] +
                  shapeFunctionGradientValues[numDofsPerCell * numQuadPoints *
                                                2 +
                                              numDofsPerCell * q +
                                              cellDofIndexI] *
                    jacobianPtr[2];
                double gradShapeYI =
                  shapeFunctionGradientValues[numDofsPerCell * q +
                                              cellDofIndexI] *
                    jacobianPtr[3] +
                  shapeFunctionGradientValues[numDofsPerCell * numQuadPoints +
                                              numDofsPerCell * q +
                                              cellDofIndexI] *
                    jacobianPtr[4] +
                  shapeFunctionGradientValues[numDofsPerCell * numQuadPoints *
                                                2 +
                                              numDofsPerCell * q +
                                              cellDofIndexI] *
                    jacobianPtr[5];
                double gradShapeZI =
                  shapeFunctionGradientValues[numDofsPerCell * q +
                                              cellDofIndexI] *
                    jacobianPtr[6] +
                  shapeFunctionGradientValues[numDofsPerCell * numQuadPoints +
                                              numDofsPerCell * q +
                                              cellDofIndexI] *
                    jacobianPtr[7] +
                  shapeFunctionGradientValues[numDofsPerCell * numQuadPoints *
                                                2 +
                                              numDofsPerCell * q +
                                              cellDofIndexI] *
                    jacobianPtr[8];
                double gradShapeXJ =
                  shapeFunctionGradientValues[numDofsPerCell * q +
                                              cellDofIndexJ] *
                    jacobianPtr[0] +
                  shapeFunctionGradientValues[numDofsPerCell * numQuadPoints +
                                              numDofsPerCell * q +
                                              cellDofIndexJ] *
                    jacobianPtr[1] +
                  shapeFunctionGradientValues[numDofsPerCell * numQuadPoints *
                                                2 +
                                              numDofsPerCell * q +
                                              cellDofIndexJ] *
                    jacobianPtr[2];
                double gradShapeYJ =
                  shapeFunctionGradientValues[numDofsPerCell * q +
                                              cellDofIndexJ] *
                    jacobianPtr[3] +
                  shapeFunctionGradientValues[numDofsPerCell * numQuadPoints +
                                              numDofsPerCell * q +
                                              cellDofIndexJ] *
                    jacobianPtr[4] +
                  shapeFunctionGradientValues[numDofsPerCell * numQuadPoints *
                                                2 +
                                              numDofsPerCell * q +
                                              cellDofIndexJ] *
                    jacobianPtr[5];
                double gradShapeZJ =
                  shapeFunctionGradientValues[numDofsPerCell * q +
                                              cellDofIndexJ] *
                    jacobianPtr[6] +
                  shapeFunctionGradientValues[numDofsPerCell * numQuadPoints +
                                              numDofsPerCell * q +
                                              cellDofIndexJ] *
                    jacobianPtr[7] +
                  shapeFunctionGradientValues[numDofsPerCell * numQuadPoints *
                                                2 +
                                              numDofsPerCell * q +
                                              cellDofIndexJ] *
                    jacobianPtr[8];
                val +=
                  ((gradShapeXI * gradShapeXJ) + (gradShapeYI * gradShapeYJ) +
                   (gradShapeZI * gradShapeZJ)) *
                  JxW[cellIndex * numQuadPoints + q];
              }
          }

        shapeGradNINJIntegralContribution[index] = val;
      }
  }

  template <typename T>
  void
  computeShapeGradNINJIntegral(
    std::shared_ptr<
      dftfe::basis::
        FEBasisOperations<T, double, dftfe::utils::MemorySpace::DEVICE>>
      &basisOperationsPtrDevice,
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      &shapeGradNINJIntegralD)
  {
    shapeGradNINJIntegralD.clear();
    shapeGradNINJIntegralD.resize(basisOperationsPtrDevice->nCells() *
                                    basisOperationsPtrDevice->nDofsPerCell() *
                                    basisOperationsPtrDevice->nDofsPerCell(),
                                  0.0);
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
    computeShapeGradNINJIntegralContribution<<<
      (basisOperationsPtrDevice->nCells() *
         basisOperationsPtrDevice->nDofsPerCell() *
         basisOperationsPtrDevice->nDofsPerCell() +
       (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
        dftfe::utils::DEVICE_BLOCK_SIZE,
      dftfe::utils::DEVICE_BLOCK_SIZE>>>(
      basisOperationsPtrDevice->nCells(),
      basisOperationsPtrDevice->nDofsPerCell(),
      basisOperationsPtrDevice->nQuadsPerCell(),
      basisOperationsPtrDevice->shapeFunctionGradientBasisData().data(),
      basisOperationsPtrDevice->inverseJacobiansBasisData().data(),
      basisOperationsPtrDevice->JxWBasisData().data(),
      basisOperationsPtrDevice->cellsTypeFlag(),
      shapeGradNINJIntegralD.data());
#elif DFTFE_WITH_DEVICE_LANG_HIP
    hipLaunchKernelGGL(
      computeShapeGradNINJIntegralContribution,
      (basisOperationsPtrDevice->nCells() *
         basisOperationsPtrDevice->nDofsPerCell() *
         basisOperationsPtrDevice->nDofsPerCell() +
       (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
        dftfe::utils::DEVICE_BLOCK_SIZE,
      dftfe::utils::DEVICE_BLOCK_SIZE,
      0,
      0,
      basisOperationsPtrDevice->nCells(),
      basisOperationsPtrDevice->nDofsPerCell(),
      basisOperationsPtrDevice->nQuadsPerCell(),
      basisOperationsPtrDevice->shapeFunctionGradientBasisData().data(),
      basisOperationsPtrDevice->inverseJacobiansBasisData().data(),
      basisOperationsPtrDevice->JxWBasisData().data(),
      basisOperationsPtrDevice->cellsTypeFlag(),
      shapeGradNINJIntegralD.data());
#endif
  }
} // namespace shapeFuncDevice

template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorDeviceClass<FEOrder, FEOrderElectro>::
  preComputeShapeFunctionGradientIntegrals(
    const unsigned int lpspQuadratureId,
    const bool         onlyUpdateGradNiNjIntegral)
{
  const unsigned int numberPhysicalCells =
    dftPtr->matrix_free_data.n_physical_cells();

  dftfe::utils::deviceSynchronize();
  MPI_Barrier(d_mpiCommParent);
  double device_time = MPI_Wtime();


  d_basisOperationsPtrDevice->reinit(0,
                                     0,
                                     dftPtr->d_feOrderPlusOneQuadratureId);
  unsigned int numberQuadraturePointsPlusOne =
    d_basisOperationsPtrDevice->nQuadsPerCell();
  shapeFuncDevice::computeShapeGradNINJIntegral(
    d_basisOperationsPtrDevice,
    d_cellShapeFunctionGradientIntegralFlattenedDevice);

  dftfe::utils::deviceSynchronize();
  MPI_Barrier(d_mpiCommParent);
  device_time = MPI_Wtime() - device_time;

  if (this_mpi_process == 0 && dftPtr->d_dftParamsPtr->verbosity >= 2)
    std::cout
      << "Time for shapeFuncDevice::computeShapeGradNINJIntegral for FEOrder: "
      << device_time << std::endl;

  if (FEOrderElectro != FEOrder)
    {
      dftfe::utils::deviceSynchronize();
      MPI_Barrier(d_mpiCommParent);
      device_time = MPI_Wtime();

      dftPtr->d_basisOperationsPtrElectroDevice->reinit(
        0, 0, dftPtr->d_phiTotAXQuadratureIdElectro);
      shapeFuncDevice::computeShapeGradNINJIntegral(
        dftPtr->d_basisOperationsPtrElectroDevice,
        d_cellShapeFunctionGradientIntegralFlattenedDeviceElectro);

      dftfe::utils::deviceSynchronize();
      MPI_Barrier(d_mpiCommParent);
      device_time = MPI_Wtime() - device_time;

      if (this_mpi_process == 0 && dftPtr->d_dftParamsPtr->verbosity >= 2)
        std::cout
          << "Time for shapeFuncDevice::computeShapeGradNINJIntegral for FEOrderElectro: "
          << device_time << std::endl;
    }
  if (!onlyUpdateGradNiNjIntegral)
    {
      dealii::QIterated<3> quadratureNLP(dealii::QGauss<1>(
                                           C_num1DQuadNLPSP<FEOrder>()),
                                         C_numCopies1DQuadNLPSP());
      dealii::FEValues<3>  fe_valuesNLP(
        dftPtr->matrix_free_data
          .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
          .get_fe(),
        quadratureNLP,
        dealii::update_values | dealii::update_gradients |
          dealii::update_jacobians | dealii::update_inverse_jacobians);
      const unsigned int  numberQuadraturePointsNLP = quadratureNLP.size();
      std::vector<double> inverseJacobiansNLP(
        numberPhysicalCells * numberQuadraturePointsNLP * 3 * 3, 0.0);
      std::vector<double> shapeFunctionGradientValueNLPTransposed(
        numberQuadraturePointsNLP * d_basisOperationsPtrHost->nDofsPerCell() *
          3,
        0.0);

      auto cellPtr = dftPtr->matrix_free_data
                       .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                       .begin_active();
      auto endcPtr = dftPtr->matrix_free_data
                       .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                       .end();

      unsigned int iElem = 0;
      for (; cellPtr != endcPtr; ++cellPtr)
        if (cellPtr->is_locally_owned())
          {
            fe_valuesNLP.reinit(cellPtr);

            const std::vector<dealii::DerivativeForm<1, 3, 3>>
              &inverseJacobians = fe_valuesNLP.get_inverse_jacobians();

            // dealii returns inverse jacobian tensor in transposed format
            // J^{-T}
            for (unsigned int q_point = 0; q_point < numberQuadraturePointsNLP;
                 ++q_point)
              for (unsigned int i = 0; i < 3; ++i)
                for (unsigned int j = 0; j < 3; ++j)
                  inverseJacobiansNLP[iElem * numberQuadraturePointsNLP * 3 *
                                        3 +
                                      q_point * 3 * 3 + j * 3 + i] =
                    inverseJacobians[q_point][i][j];


            if (iElem == 0)
              {
                const std::vector<dealii::DerivativeForm<1, 3, 3>> &jacobians =
                  fe_valuesNLP.get_jacobians();
                for (unsigned int iNode = 0;
                     iNode < d_basisOperationsPtrHost->nDofsPerCell();
                     ++iNode)
                  for (unsigned int q_point = 0;
                       q_point < numberQuadraturePointsNLP;
                       ++q_point)
                    {
                      const dealii::Tensor<1, 3, double> &shape_grad_real =
                        fe_valuesNLP.shape_grad(iNode, q_point);

                      // J^{T}*grad(u_h)
                      const dealii::Tensor<1, 3, double> &shape_grad_reference =
                        apply_transformation(jacobians[q_point].transpose(),
                                             shape_grad_real);

                      shapeFunctionGradientValueNLPTransposed
                        [q_point * d_basisOperationsPtrHost->nDofsPerCell() *
                           3 +
                         iNode] = shape_grad_reference[0];
                      shapeFunctionGradientValueNLPTransposed
                        [q_point * d_basisOperationsPtrHost->nDofsPerCell() *
                           3 +
                         d_basisOperationsPtrHost->nDofsPerCell() + iNode] =
                          shape_grad_reference[1];
                      shapeFunctionGradientValueNLPTransposed
                        [q_point * d_basisOperationsPtrHost->nDofsPerCell() *
                           3 +
                         d_basisOperationsPtrHost->nDofsPerCell() * 2 + iNode] =
                          shape_grad_reference[2];
                    }
              }

            iElem++;
          }
      d_shapeFunctionGradientValueNLPTransposedDevice.resize(
        shapeFunctionGradientValueNLPTransposed.size());
      d_shapeFunctionGradientValueNLPTransposedDevice.copyFrom(
        shapeFunctionGradientValueNLPTransposed);

      d_inverseJacobiansNLPDevice.resize(inverseJacobiansNLP.size());
      d_inverseJacobiansNLPDevice.copyFrom(inverseJacobiansNLP);
    }
}
