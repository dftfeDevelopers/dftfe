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
    const unsigned int numQuadsBlock,
    const unsigned int numQuadsTotal,
    const unsigned int startingQuadId,
    const unsigned int numNodesPerElem,
    const unsigned int numElems,
    const double *     gradNQuadValuesXI,
    const double *     gradNQuadValuesYI,
    const double *     gradNQuadValuesZI,
    const double *     gradNQuadValuesXJ,
    const double *     gradNQuadValuesYJ,
    const double *     gradNQuadValuesZJ,
    const double *     jxwQuadValues,
    double *           shapeGradNINJIntegralContribution)
  {
    const unsigned int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int numberEntries =
      numElems * numNodesPerElem * numNodesPerElem * numQuadsBlock;

    for (unsigned int index = globalThreadId; index < numberEntries;
         index += blockDim.x * gridDim.x)
      {
        const unsigned int blockIndex1 = index / numQuadsBlock;
        const unsigned int quadIndex   = index - blockIndex1 * numQuadsBlock;
        const unsigned int blockIndex2 = blockIndex1 / numNodesPerElem;
        const unsigned int cellId      = blockIndex2 / numNodesPerElem;
        const unsigned int idJ =
          cellId * numNodesPerElem * numQuadsTotal +
          (blockIndex1 - blockIndex2 * numNodesPerElem) * numQuadsTotal +
          quadIndex + startingQuadId;
        const unsigned int idI =
          cellId * numNodesPerElem * numQuadsTotal +
          (blockIndex2 - cellId * numNodesPerElem) * numQuadsTotal + quadIndex +
          startingQuadId;


        shapeGradNINJIntegralContribution[index] =
          (gradNQuadValuesXI[idI] * gradNQuadValuesXJ[idJ] +
           gradNQuadValuesYI[idI] * gradNQuadValuesYJ[idJ] +
           gradNQuadValuesZI[idI] * gradNQuadValuesZJ[idJ]) *
          jxwQuadValues[cellId * numQuadsTotal + quadIndex + startingQuadId];
      }
  }

  void
  computeShapeGradNINJIntegral(
    dftfe::utils::deviceBlasHandle_t &handle,
    dealii::FEValues<3> &             fe_values,
    const dealii::DoFHandler<3> &     dofHandler,
    const unsigned int                numElems,
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      &shapeGradNINJIntegralD)
  {
    const unsigned int numQuads        = fe_values.get_quadrature().size();
    const unsigned int numNodesPerElem = fe_values.get_fe().dofs_per_cell;

    shapeGradNINJIntegralD.clear();
    shapeGradNINJIntegralD.resize(numElems * numNodesPerElem * numNodesPerElem,
                                  0.0);

    const int blockSizeElems    = 1;
    const int blockSizeQuads    = 100;
    const int numberElemBlocks  = numElems / blockSizeElems;
    const int remBlockSizeElems = numElems - numberElemBlocks * blockSizeElems;

    const int numberQuadsBlocks = numQuads / blockSizeQuads;
    const int remBlockSizeQuads = numQuads - numberQuadsBlocks * blockSizeQuads;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      shapeGradNINJIntegralContributionD(blockSizeElems * numNodesPerElem *
                                           numNodesPerElem * blockSizeQuads,
                                         0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      onesVecD(blockSizeQuads, 1.0);

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST_PINNED>
      cellJxWValues(blockSizeElems * numQuads);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST_PINNED>
      shapeFunctionGradientValuesX(blockSizeElems * numQuads * numNodesPerElem,
                                   0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST_PINNED>
      shapeFunctionGradientValuesY(blockSizeElems * numQuads * numNodesPerElem,
                                   0.0);
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST_PINNED>
      shapeFunctionGradientValuesZ(blockSizeElems * numQuads * numNodesPerElem,
                                   0.0);

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      jxwQuadValuesD(cellJxWValues.size());
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      gradNQuadValuesXD(shapeFunctionGradientValuesX.size());
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      gradNQuadValuesYD(shapeFunctionGradientValuesX.size());
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
      gradNQuadValuesZD(shapeFunctionGradientValuesX.size());

    for (int iblock = 0; iblock < (numberElemBlocks + 1); iblock++)
      {
        const int currentElemsBlockSize =
          (iblock == numberElemBlocks) ? remBlockSizeElems : blockSizeElems;
        if (currentElemsBlockSize > 0)
          {
            const int startingElemId = iblock * blockSizeElems;

            typename dealii::DoFHandler<3>::active_cell_iterator cellPtr =
              dofHandler.begin_active();
            typename dealii::DoFHandler<3>::active_cell_iterator endcPtr =
              dofHandler.end();

            unsigned int iElem = 0;
            for (; cellPtr != endcPtr; ++cellPtr)
              if (cellPtr->is_locally_owned())
                {
                  if (iElem >= startingElemId &&
                      iElem < (startingElemId + currentElemsBlockSize))
                    {
                      const unsigned int intraBlockElemId =
                        iElem - startingElemId;
                      fe_values.reinit(cellPtr);

                      for (unsigned int q_point = 0; q_point < numQuads;
                           ++q_point)
                        cellJxWValues[intraBlockElemId * numQuads + q_point] =
                          fe_values.JxW(q_point);

                      for (unsigned int iNode = 0; iNode < numNodesPerElem;
                           ++iNode)
                        for (unsigned int q_point = 0; q_point < numQuads;
                             ++q_point)
                          {
                            const dealii::Tensor<1, 3, double> &shape_grad =
                              fe_values.shape_grad(iNode, q_point);

                            shapeFunctionGradientValuesX
                              [intraBlockElemId * numNodesPerElem * numQuads +
                               iNode * numQuads + q_point] = shape_grad[0];

                            shapeFunctionGradientValuesY
                              [intraBlockElemId * numNodesPerElem * numQuads +
                               iNode * numQuads + q_point] = shape_grad[1];

                            shapeFunctionGradientValuesZ
                              [intraBlockElemId * numNodesPerElem * numQuads +
                               iNode * numQuads + q_point] = shape_grad[2];
                          }
                    }

                  iElem++;
                }

            jxwQuadValuesD.copyFrom(cellJxWValues);
            gradNQuadValuesXD.copyFrom(shapeFunctionGradientValuesX);
            gradNQuadValuesYD.copyFrom(shapeFunctionGradientValuesY);
            gradNQuadValuesZD.copyFrom(shapeFunctionGradientValuesZ);

            for (int jblock = 0; jblock < (numberQuadsBlocks + 1); jblock++)
              {
                const int currentQuadsBlockSize =
                  (jblock == numberQuadsBlocks) ? remBlockSizeQuads :
                                                  blockSizeQuads;
                const int startingQuadId = jblock * blockSizeQuads;
                if (currentQuadsBlockSize > 0)
                  {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
                    computeShapeGradNINJIntegralContribution<<<
                      (currentQuadsBlockSize +
                       (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                        dftfe::utils::DEVICE_BLOCK_SIZE * numNodesPerElem *
                        numNodesPerElem * currentElemsBlockSize,
                      dftfe::utils::DEVICE_BLOCK_SIZE>>>(
                      currentQuadsBlockSize,
                      numQuads,
                      startingQuadId,
                      numNodesPerElem,
                      currentElemsBlockSize,
                      gradNQuadValuesXD.begin(),
                      gradNQuadValuesYD.begin(),
                      gradNQuadValuesZD.begin(),
                      gradNQuadValuesXD.begin(),
                      gradNQuadValuesYD.begin(),
                      gradNQuadValuesZD.begin(),
                      jxwQuadValuesD.begin(),
                      shapeGradNINJIntegralContributionD.begin());
#elif DFTFE_WITH_DEVICE_LANG_HIP
                    hipLaunchKernelGGL(
                      computeShapeGradNINJIntegralContribution,
                      (currentQuadsBlockSize +
                       (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                        dftfe::utils::DEVICE_BLOCK_SIZE * numNodesPerElem *
                        numNodesPerElem * currentElemsBlockSize,
                      dftfe::utils::DEVICE_BLOCK_SIZE,
                      0,
                      0,
                      currentQuadsBlockSize,
                      numQuads,
                      startingQuadId,
                      numNodesPerElem,
                      currentElemsBlockSize,
                      gradNQuadValuesXD.begin(),
                      gradNQuadValuesYD.begin(),
                      gradNQuadValuesZD.begin(),
                      gradNQuadValuesXD.begin(),
                      gradNQuadValuesYD.begin(),
                      gradNQuadValuesZD.begin(),
                      jxwQuadValuesD.begin(),
                      shapeGradNINJIntegralContributionD.begin());
#endif

                    const double scalarCoeffAlpha = 1.0;
                    const double scalarCoeffBeta  = 1.0;



                    dftfe::utils::deviceBlasWrapper::gemm(
                      handle,
                      dftfe::utils::DEVICEBLAS_OP_N,
                      dftfe::utils::DEVICEBLAS_OP_N,
                      1,
                      currentElemsBlockSize * numNodesPerElem * numNodesPerElem,
                      currentQuadsBlockSize,
                      &scalarCoeffAlpha,
                      onesVecD.begin(),
                      1,
                      shapeGradNINJIntegralContributionD.begin(),
                      currentQuadsBlockSize,
                      &scalarCoeffBeta,
                      shapeGradNINJIntegralD.begin() +
                        startingElemId * numNodesPerElem * numNodesPerElem,
                      1);
                  }
              } // block loop over nodes per elem
          }
      } // block loop over elems
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

  dealii::QGauss<3>   quadraturePlusOne(FEOrder + 1);
  unsigned int        numberQuadraturePointsPlusOne = quadraturePlusOne.size();
  dealii::FEValues<3> fe_values_plusone(
    dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
      .get_fe(),
    quadraturePlusOne,
    dealii::update_gradients | dealii::update_JxW_values);


  shapeFuncDevice::computeShapeGradNINJIntegral(
    d_deviceBlasHandle,
    fe_values_plusone,
    dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex),
    numberPhysicalCells,
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

      dealii::QGauss<3> quadratureElectroPlusOne(FEOrderElectro + 1);
      numberQuadraturePointsPlusOne = quadratureElectroPlusOne.size();
      dealii::FEValues<3> fe_values_electro_plusone(
        dftPtr->d_matrixFreeDataPRefined
          .get_dof_handler(dftPtr->d_baseDofHandlerIndexElectro)
          .get_fe(),
        quadratureElectroPlusOne,
        dealii::update_gradients | dealii::update_JxW_values);

      shapeFuncDevice::computeShapeGradNINJIntegral(
        d_deviceBlasHandle,
        fe_values_electro_plusone,
        dftPtr->d_matrixFreeDataPRefined.get_dof_handler(
          dftPtr->d_baseDofHandlerIndexElectro),
        numberPhysicalCells,
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
      //
      // get FE data
      //
      const dealii::Quadrature<3> &quadrature =
        dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
      dealii::FEValues<3> fe_values(dftPtr->matrix_free_data
                                      .get_dof_handler(
                                        dftPtr->d_densityDofHandlerIndex)
                                      .get_fe(),
                                    quadrature,
                                    dealii::update_values |
                                      dealii::update_gradients |
                                      dealii::update_JxW_values);
      const unsigned int  numberDofsPerElement =
        dftPtr->matrix_free_data
          .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
          .get_fe()
          .dofs_per_cell;
      const unsigned int numberDofsPerElementElectro =
        dftPtr->d_matrixFreeDataPRefined
          .get_dof_handler(dftPtr->d_baseDofHandlerIndexElectro)
          .get_fe()
          .dofs_per_cell;
      const unsigned int numberQuadraturePoints = quadrature.size();

      dealii::FEValues<3> fe_values_lpsp(
        dftPtr->matrix_free_data
          .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
          .get_fe(),
        dftPtr->matrix_free_data.get_quadrature(lpspQuadratureId),
        dealii::update_values);
      const unsigned int numberQuadraturePointsLpsp =
        dftPtr->matrix_free_data.get_quadrature(lpspQuadratureId).size();
      d_numQuadPointsLpsp = numberQuadraturePointsLpsp;

      //
      // resize data members
      //
      // d_cellShapeFunctionGradientIntegralFlattened.clear();
      // d_cellShapeFunctionGradientIntegralFlattened.resize(numberPhysicalCells*numberDofsPerElement*numberDofsPerElement);

      d_cellJxWValues.clear();
      d_cellJxWValues.resize(numberPhysicalCells * numberQuadraturePoints);

      d_shapeFunctionValue.resize(numberQuadraturePoints * numberDofsPerElement,
                                  0.0);
      d_shapeFunctionValueTransposed.resize(numberQuadraturePoints *
                                              numberDofsPerElement,
                                            0.0);

      // d_shapeFunctionGradientValueX.resize(numberPhysicalCells *
      //                                        numberQuadraturePoints *
      //                                        numberDofsPerElement,
      //                                      0.0);
      // d_shapeFunctionGradientValueXTransposed.resize(numberPhysicalCells *
      //                                                  numberQuadraturePoints
      //                                                  *
      //                                                  numberDofsPerElement,
      //                                                0.0);

      // d_shapeFunctionGradientValueY.resize(numberPhysicalCells *
      //                                        numberQuadraturePoints *
      //                                        numberDofsPerElement,
      //                                      0.0);
      // d_shapeFunctionGradientValueYTransposed.resize(numberPhysicalCells *
      //                                                  numberQuadraturePoints
      //                                                  *
      //                                                  numberDofsPerElement,
      //                                                0.0);

      // d_shapeFunctionGradientValueZ.resize(numberPhysicalCells *
      //                                        numberQuadraturePoints *
      //                                        numberDofsPerElement,
      //                                      0.0);
      // d_shapeFunctionGradientValueZTransposed.resize(numberPhysicalCells *
      //                                                  numberQuadraturePoints
      //                                                  *
      //                                                  numberDofsPerElement,
      //                                                0.0);

      std::vector<double> shapeFunctionValueLpsp(numberQuadraturePointsLpsp *
                                                   numberDofsPerElement,
                                                 0.0);
      std::vector<double> shapeFunctionValueTransposedLpsp(
        numberQuadraturePointsLpsp * numberDofsPerElement, 0.0);



      typename dealii::DoFHandler<3>::active_cell_iterator cellPtr =
        dftPtr->matrix_free_data
          .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
          .begin_active();
      typename dealii::DoFHandler<3>::active_cell_iterator endcPtr =
        dftPtr->matrix_free_data
          .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
          .end();

      unsigned int iElem = 0;
      for (; cellPtr != endcPtr; ++cellPtr)
        if (cellPtr->is_locally_owned())
          {
            fe_values.reinit(cellPtr);

            for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
                 ++q_point)
              d_cellJxWValues[iElem * numberQuadraturePoints + q_point] =
                fe_values.JxW(q_point);

            // for (unsigned int iNode = 0; iNode < numberDofsPerElement;
            // ++iNode)
            //   for (unsigned int q_point = 0; q_point <
            //   numberQuadraturePoints;
            //        ++q_point)
            //     {
            //       const dealii::Tensor<1, 3, double> &shape_grad =
            //         fe_values.shape_grad(iNode, q_point);

            //       d_shapeFunctionGradientValueX[iElem * numberDofsPerElement
            //       *
            //                                       numberQuadraturePoints +
            //                                     iNode *
            //                                     numberQuadraturePoints +
            //                                     q_point] = shape_grad[0];
            //       d_shapeFunctionGradientValueXTransposed
            //         [iElem * numberQuadraturePoints * numberDofsPerElement +
            //          q_point * numberDofsPerElement + iNode] = shape_grad[0];

            //       d_shapeFunctionGradientValueY[iElem * numberDofsPerElement
            //       *
            //                                       numberQuadraturePoints +
            //                                     iNode *
            //                                     numberQuadraturePoints +
            //                                     q_point] = shape_grad[1];
            //       d_shapeFunctionGradientValueYTransposed
            //         [iElem * numberQuadraturePoints * numberDofsPerElement +
            //          q_point * numberDofsPerElement + iNode] = shape_grad[1];

            //       d_shapeFunctionGradientValueZ[iElem * numberDofsPerElement
            //       *
            //                                       numberQuadraturePoints +
            //                                     iNode *
            //                                     numberQuadraturePoints +
            //                                     q_point] = shape_grad[2];
            //       d_shapeFunctionGradientValueZTransposed
            //         [iElem * numberQuadraturePoints * numberDofsPerElement +
            //          q_point * numberDofsPerElement + iNode] = shape_grad[2];
            //     }

            if (iElem == 0)
              {
                fe_values_lpsp.reinit(cellPtr);

                for (unsigned int iNode = 0; iNode < numberDofsPerElement;
                     ++iNode)
                  for (unsigned int q_point = 0;
                       q_point < numberQuadraturePoints;
                       ++q_point)
                    {
                      const double val = fe_values.shape_value(iNode, q_point);
                      d_shapeFunctionValue[numberQuadraturePoints * iNode +
                                           q_point]         = val;
                      d_shapeFunctionValueTransposed[q_point *
                                                       numberDofsPerElement +
                                                     iNode] = val;
                    }

                for (unsigned int iNode = 0; iNode < numberDofsPerElement;
                     ++iNode)
                  for (unsigned int q_point = 0;
                       q_point < numberQuadraturePointsLpsp;
                       ++q_point)
                    {
                      const double val =
                        fe_values_lpsp.shape_value(iNode, q_point);
                      shapeFunctionValueLpsp[numberQuadraturePointsLpsp *
                                               iNode +
                                             q_point]         = val;
                      shapeFunctionValueTransposedLpsp[q_point *
                                                         numberDofsPerElement +
                                                       iNode] = val;
                    }
              }

            iElem++;
          }

      d_shapeFunctionValueDevice.resize(d_shapeFunctionValue.size());
      d_shapeFunctionValueDevice.copyFrom(d_shapeFunctionValue);
      d_shapeFunctionValueTransposedDevice.resize(
        d_shapeFunctionValueTransposed.size());
      d_shapeFunctionValueTransposedDevice.copyFrom(
        d_shapeFunctionValueTransposed);

      // d_shapeFunctionGradientValueXTransposedDevice.resize(
      //   d_shapeFunctionGradientValueXTransposed.size());
      // d_shapeFunctionGradientValueXTransposedDevice.copyFrom(
      //   d_shapeFunctionGradientValueXTransposed);

      // d_shapeFunctionGradientValueYTransposedDevice.resize(
      //   d_shapeFunctionGradientValueYTransposed.size());
      // d_shapeFunctionGradientValueYTransposedDevice.copyFrom(
      //   d_shapeFunctionGradientValueYTransposed);

      // d_shapeFunctionGradientValueZTransposedDevice.resize(
      //   d_shapeFunctionGradientValueZTransposed.size());
      // d_shapeFunctionGradientValueZTransposedDevice.copyFrom(
      //   d_shapeFunctionGradientValueZTransposed);

      d_shapeFunctionValueLpspDevice.resize(shapeFunctionValueLpsp.size());
      d_shapeFunctionValueLpspDevice.copyFrom(shapeFunctionValueLpsp);

      d_shapeFunctionValueTransposedLpspDevice.resize(
        shapeFunctionValueTransposedLpsp.size());
      d_shapeFunctionValueTransposedLpspDevice.copyFrom(
        shapeFunctionValueTransposedLpsp);

      d_cellJxWValuesDevice.resize(d_cellJxWValues.size());
      d_cellJxWValuesDevice.copyFrom(d_cellJxWValues);

      dealii::QGaussLobatto<3> quadratureGl(
        C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>() + 1);
      dealii::FEValues<3> fe_valuesGl(dftPtr->matrix_free_data
                                        .get_dof_handler(
                                          dftPtr->d_densityDofHandlerIndex)
                                        .get_fe(),
                                      quadratureGl,
                                      dealii::update_values);
      const unsigned int  numberQuadraturePointsGl = quadratureGl.size();

      //
      // resize data members
      //
      std::vector<double> glShapeFunctionValueTransposed(
        numberQuadraturePointsGl * numberDofsPerElement, 0.0);

      cellPtr = dftPtr->matrix_free_data
                  .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                  .begin_active();
      fe_valuesGl.reinit(cellPtr);

      for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
        for (unsigned int q_point = 0; q_point < numberQuadraturePointsGl;
             ++q_point)
          {
            const double val = fe_valuesGl.shape_value(iNode, q_point);
            glShapeFunctionValueTransposed[q_point * numberDofsPerElement +
                                           iNode] = val;
          }

      d_glShapeFunctionValueTransposedDevice.resize(
        glShapeFunctionValueTransposed.size());
      d_glShapeFunctionValueTransposedDevice.copyFrom(
        glShapeFunctionValueTransposed);

      // dealii::QGauss<3>  quadratureNLP(C_num1DQuadNLPSP<FEOrder>());
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
      const unsigned int numberQuadraturePointsNLP = quadratureNLP.size();

      //
      // resize data members
      //
      std::vector<double> nlpShapeFunctionValueTransposed(
        numberQuadraturePointsNLP * numberDofsPerElement, 0.0);
      std::vector<double> inverseJacobiansNLP(
        numberPhysicalCells * numberQuadraturePointsNLP * 3 * 3, 0.0);
      std::vector<double> shapeFunctionGradientValueNLPTransposed(
        numberQuadraturePointsNLP * numberDofsPerElement * 3, 0.0);

      cellPtr = dftPtr->matrix_free_data
                  .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                  .begin_active();
      endcPtr = dftPtr->matrix_free_data
                  .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                  .end();


      iElem = 0;
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
                for (unsigned int iNode = 0; iNode < numberDofsPerElement;
                     ++iNode)
                  for (unsigned int q_point = 0;
                       q_point < numberQuadraturePointsNLP;
                       ++q_point)
                    {
                      const double val =
                        fe_valuesNLP.shape_value(iNode, q_point);
                      nlpShapeFunctionValueTransposed[q_point *
                                                        numberDofsPerElement +
                                                      iNode] = val;

                      const dealii::Tensor<1, 3, double> &shape_grad_real =
                        fe_valuesNLP.shape_grad(iNode, q_point);

                      // J^{T}*grad(u_h)
                      const dealii::Tensor<1, 3, double> &shape_grad_reference =
                        apply_transformation(jacobians[q_point].transpose(),
                                             shape_grad_real);

                      shapeFunctionGradientValueNLPTransposed
                        [q_point * numberDofsPerElement * 3 + iNode] =
                          shape_grad_reference[0];
                      shapeFunctionGradientValueNLPTransposed
                        [q_point * numberDofsPerElement * 3 +
                         numberDofsPerElement + iNode] =
                          shape_grad_reference[1];
                      shapeFunctionGradientValueNLPTransposed
                        [q_point * numberDofsPerElement * 3 +
                         numberDofsPerElement * 2 + iNode] =
                          shape_grad_reference[2];
                    }
              }

            iElem++;
          }

      d_shapeFunctionValueNLPTransposedDevice.resize(
        nlpShapeFunctionValueTransposed.size());
      d_shapeFunctionValueNLPTransposedDevice.copyFrom(
        nlpShapeFunctionValueTransposed);

      d_shapeFunctionGradientValueNLPTransposedDevice.resize(
        shapeFunctionGradientValueNLPTransposed.size());
      d_shapeFunctionGradientValueNLPTransposedDevice.copyFrom(
        shapeFunctionGradientValueNLPTransposed);

      d_inverseJacobiansNLPDevice.resize(inverseJacobiansNLP.size());
      d_inverseJacobiansNLPDevice.copyFrom(inverseJacobiansNLP);
    }
}
