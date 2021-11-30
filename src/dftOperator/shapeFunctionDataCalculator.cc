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
// the top level of the DFT-FE udistribution.
//
// ---------------------------------------------------------------------
//
// @author  Phani Motamarri, Sambit Das
//


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::
  preComputeShapeFunctionGradientIntegrals(const unsigned int lpspQuadratureId)
{
  //
  // get FE data
  //
  const unsigned int numberMacroCells =
    dftPtr->matrix_free_data.n_macro_cells();
  const unsigned int numberPhysicalCells =
    dftPtr->matrix_free_data.n_physical_cells();
  const Quadrature<3> &quadrature =
    dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
  FEValues<3> fe_values(dftPtr->matrix_free_data.get_dof_handler().get_fe(),
                        quadrature,
                        update_values | update_gradients | update_jacobians |
                          update_JxW_values | update_inverse_jacobians);

  const unsigned int numberDofsPerElement =
    dftPtr->matrix_free_data.get_dof_handler().get_fe().dofs_per_cell;
  const unsigned int numberQuadraturePoints = quadrature.size();

  QGauss<3>   quadraturePlusOne(FEOrder + 1);
  FEValues<3> fe_values_quadplusone(
    dftPtr->matrix_free_data.get_dof_handler().get_fe(),
    quadraturePlusOne,
    update_gradients | update_JxW_values);
  const unsigned int numberQuadraturePointsPlusOne = quadraturePlusOne.size();

  FEValues<3> fe_values_lpsp(
    dftPtr->matrix_free_data.get_dof_handler().get_fe(),
    dftPtr->matrix_free_data.get_quadrature(lpspQuadratureId),
    update_values);
  const unsigned int numberQuadraturePointsLpsp =
    dftPtr->matrix_free_data.get_quadrature(lpspQuadratureId).size();

  //
  // resize data members
  //
  unsigned int sizeNiNj =
    (numberDofsPerElement * (numberDofsPerElement + 1)) / 2;
  d_shapeFunctionData.resize(numberDofsPerElement * numberQuadraturePoints,
                             0.0);
  d_shapeFunctionLpspQuadData.resize(numberDofsPerElement *
                                       numberQuadraturePointsLpsp,
                                     0.0);
  d_cellShapeFunctionGradientIntegral.resize(numberPhysicalCells * sizeNiNj);

#ifdef USE_COMPLEX
  d_NiNjIntegral.resize(numberPhysicalCells * sizeNiNj);
  d_invJacKPointTimesJxW.resize(dftPtr->d_kPointWeights.size());
  for (unsigned int kPointIndex = 0;
       kPointIndex < dftPtr->d_kPointWeights.size();
       ++kPointIndex)
    d_invJacKPointTimesJxW[kPointIndex].resize(numberPhysicalCells * 3 *
                                               numberQuadraturePoints);
  std::vector<double> kPointCoors(3, 0.0);
#endif

  //
  // some more data members, local variables
  //
  std::vector<double> shapeFunctionGradientValueRef;


  if (dftParameters::xcFamilyType == "GGA")
    {
      d_shapeFunctionGradientValueRefX.resize(numberQuadraturePoints *
                                                numberDofsPerElement,
                                              0.0);
      d_shapeFunctionGradientValueRefY.resize(numberQuadraturePoints *
                                                numberDofsPerElement,
                                              0.0);
      d_shapeFunctionGradientValueRefZ.resize(numberQuadraturePoints *
                                                numberDofsPerElement,
                                              0.0);
    }

#ifdef USE_COMPLEX
  if (dftParameters::xcFamilyType != "GGA")
    {
      d_shapeFunctionGradientValueRefX.resize(numberQuadraturePoints *
                                                numberDofsPerElement,
                                              0.0);
      d_shapeFunctionGradientValueRefY.resize(numberQuadraturePoints *
                                                numberDofsPerElement,
                                              0.0);
      d_shapeFunctionGradientValueRefZ.resize(numberQuadraturePoints *
                                                numberDofsPerElement,
                                              0.0);
    }
#endif



  typename dealii::DoFHandler<3>::active_cell_iterator cellPtr = dftPtr->matrix_free_data.get_dof_handler(d_densityDofHandlerIndex).begin_active(),
    endcellPtr = dftPtr->matrix_free_data.get_dof_handler(d_densityDofHandlerIndex).end();
  
  //
  // compute cell-level shapefunctiongradientintegral generator by going over
  // dealii macrocells which allows efficient integration of cell-level matrix
  // integrals using dealii vectorized arrays
  unsigned int iElemCount = 0;
  //for (int iMacroCell = 0; iMacroCell < numberMacroCells; ++iMacroCell)
  for (; cellPtr != endcellPtr; ++cellPtr)
    {
      //unsigned int n_sub_cells =
      //dftPtr->matrix_free_data.n_components_filled(iMacroCell);

      //for (unsigned int iCell = 0; iCell < n_sub_cells; ++iCell)
      if(cellPtr->is_locally_owned())
        {
          //cellPtr = dftPtr->matrix_free_data.get_cell_iterator(
	  //iMacroCell, iCell, dftPtr->d_densityDofHandlerIndex);
          fe_values_quadplusone.reinit(cellPtr);

#ifdef USE_COMPLEX
          fe_values.reinit(cellPtr);
          const std::vector<DerivativeForm<1, 3, 3>> &inverseJacobians =
            fe_values.get_inverse_jacobians();
#endif

          unsigned int count = 0;
          for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            {
              for (unsigned int jNode = iNode; jNode < numberDofsPerElement;
                   ++jNode)
                {
                  double shapeFunctionGradientValue = 0.0;

                  for (unsigned int q_point = 0;
                       q_point < numberQuadraturePointsPlusOne;
                       ++q_point)
                    {
                      shapeFunctionGradientValue +=
                        (fe_values_quadplusone.shape_grad(iNode, q_point) *
                         fe_values_quadplusone.shape_grad(jNode, q_point)) *
                        fe_values_quadplusone.JxW(q_point);
                    }

                  d_cellShapeFunctionGradientIntegral[sizeNiNj * iElemCount +
                                                      count] =
                    shapeFunctionGradientValue;
#ifdef USE_COMPLEX
                  double shapeFunctionValue = 0.0;
                  for (unsigned int q_point = 0;
                       q_point < numberQuadraturePoints;
                       ++q_point)
                    {
                      shapeFunctionValue +=
                        (fe_values.shape_value(iNode, q_point) *
                         fe_values.shape_value(jNode, q_point)) *
                        fe_values.JxW(q_point);
                    }

                  d_NiNjIntegral[sizeNiNj * iElemCount + count] =
                    shapeFunctionValue;
#endif

                  count += 1;
                } // j node loop

            } // i node loop


#ifdef USE_COMPLEX
          // Precompute and store "J^{-1}_pq k_p" for every cell and quadpoint.
          // J^{-1} computed from dealii returns J^{-T} and hence the following
          // logic
          for (unsigned int kPointIndex = 0;
               kPointIndex < dftPtr->d_kPointWeights.size();
               ++kPointIndex)
            {
              kPointCoors[0] = dftPtr->d_kPointCoordinates[3 * kPointIndex + 0];
              kPointCoors[1] = dftPtr->d_kPointCoordinates[3 * kPointIndex + 1];
              kPointCoors[2] = dftPtr->d_kPointCoordinates[3 * kPointIndex + 2];
              for (unsigned int q = 0; q < numberQuadraturePoints; ++q)
                {
                  d_invJacKPointTimesJxW[kPointIndex][numberPhysicalCells * 3 *
                                                        q +
                                                      iElemCount] =
                    -(inverseJacobians[q][0][0] * kPointCoors[0] +
                      inverseJacobians[q][0][1] * kPointCoors[1] +
                      inverseJacobians[q][0][2] * kPointCoors[2]) *
                    fe_values.JxW(q);
                  d_invJacKPointTimesJxW[kPointIndex][numberPhysicalCells *
                                                        (3 * q + 1) +
                                                      iElemCount] =
                    -(inverseJacobians[q][1][0] * kPointCoors[0] +
                      inverseJacobians[q][1][1] * kPointCoors[1] +
                      inverseJacobians[q][1][2] * kPointCoors[2]) *
                    fe_values.JxW(q);
                  d_invJacKPointTimesJxW[kPointIndex][numberPhysicalCells *
                                                        (3 * q + 2) +
                                                      iElemCount] =
                    -(inverseJacobians[q][2][0] * kPointCoors[0] +
                      inverseJacobians[q][2][1] * kPointCoors[1] +
                      inverseJacobians[q][2][2] * kPointCoors[2]) *
                    fe_values.JxW(q);
                }
            }

#endif



          if (iElemCount == 0)
            {
              fe_values.reinit(cellPtr);
              fe_values_lpsp.reinit(cellPtr);

              if (dftParameters::xcFamilyType == "GGA")
                {
                  const std::vector<dealii::DerivativeForm<1, 3, 3>>
                    &jacobians = fe_values.get_jacobians();


                  for (unsigned int q_point = 0;
                       q_point < numberQuadraturePoints;
                       ++q_point)
                    {
                      for (unsigned int iNode = 0; iNode < numberDofsPerElement;
                           ++iNode)
                        {
                          const dealii::Tensor<1, 3, double> &shape_grad_real =
                            fe_values.shape_grad(iNode, q_point);
                          const dealii::Tensor<1, 3, double>
                            &shape_grad_reference = apply_transformation(
                              jacobians[q_point].transpose(), shape_grad_real);

                          d_shapeFunctionGradientValueRefX
                            [numberDofsPerElement * q_point + iNode] =
                              shape_grad_reference[0];
                          d_shapeFunctionGradientValueRefY
                            [numberDofsPerElement * q_point + iNode] =
                              shape_grad_reference[1];
                          d_shapeFunctionGradientValueRefZ
                            [numberDofsPerElement * q_point + iNode] =
                              shape_grad_reference[2];
                        }
                    }
                }

#ifdef USE_COMPLEX
              if (dftParameters::xcFamilyType != "GGA")
                {
                  const std::vector<dealii::DerivativeForm<1, 3, 3>>
                    &jacobians = fe_values.get_jacobians();
                  for (unsigned int q_point = 0;
                       q_point < numberQuadraturePoints;
                       ++q_point)
                    {
                      for (unsigned int iNode = 0; iNode < numberDofsPerElement;
                           ++iNode)
                        {
                          const dealii::Tensor<1, 3, double> &shape_grad_real =
                            fe_values.shape_grad(iNode, q_point);
                          const dealii::Tensor<1, 3, double>
                            &shape_grad_reference = apply_transformation(
                              jacobians[q_point].transpose(), shape_grad_real);

                          d_shapeFunctionGradientValueRefX
                            [numberDofsPerElement * q_point + iNode] =
                              shape_grad_reference[0];
                          d_shapeFunctionGradientValueRefY
                            [numberDofsPerElement * q_point + iNode] =
                              shape_grad_reference[1];
                          d_shapeFunctionGradientValueRefZ
                            [numberDofsPerElement * q_point + iNode] =
                              shape_grad_reference[2];
                        }
                    }
                }
#endif

              for (unsigned int q_point = 0;
                   q_point < numberQuadraturePointsLpsp;
                   ++q_point)
                {
                  for (unsigned int iNode = 0; iNode < numberDofsPerElement;
                       ++iNode)
                    {
                      d_shapeFunctionLpspQuadData[numberDofsPerElement *
                                                    q_point +
                                                  iNode] =
                        fe_values_lpsp.shape_value(iNode, q_point);
                    }
                }



              for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
                   ++q_point)
                {
                  for (unsigned int iNode = 0; iNode < numberDofsPerElement;
                       ++iNode)
                    {
                      d_shapeFunctionData[numberDofsPerElement * q_point +
                                          iNode] =
                        fe_values.shape_value(iNode, q_point);
                    }
                }

              unsigned int numBlocks              = (FEOrder + 1);
              unsigned int numberEntriesEachBlock = sizeNiNj / numBlocks;
              unsigned int count                  = 0;
              unsigned int blockCount             = 0;
              unsigned int indexCount             = 0;
              d_blockiNodeIndex.resize(sizeNiNj);
              d_blockjNodeIndex.resize(sizeNiNj);

              for (unsigned int iNode = 0; iNode < numberDofsPerElement;
                   ++iNode)
                {
                  for (unsigned int jNode = iNode; jNode < numberDofsPerElement;
                       ++jNode)
                    {
                      d_blockiNodeIndex[numberEntriesEachBlock * blockCount +
                                        indexCount] = iNode;
                      d_blockjNodeIndex[numberEntriesEachBlock * blockCount +
                                        indexCount] = jNode;
                      count += 1;
                      indexCount += 1;
                      if (count % numberEntriesEachBlock == 0)
                        {
                          blockCount += 1;
                          indexCount = 0;
                        }
                    }
                }
#ifdef USE_COMPLEX
              unsigned int sizeNiNjNoSym =
                numberDofsPerElement * numberDofsPerElement;
              unsigned int numberEntriesEachBlockNoSym =
                sizeNiNjNoSym / numBlocks;
              count      = 0;
              blockCount = 0;
              indexCount = 0;
              d_blockiNodeIndexNoSym.resize(sizeNiNjNoSym);
              d_blockjNodeIndexNoSym.resize(sizeNiNjNoSym);

              for (unsigned int iNode = 0; iNode < numberDofsPerElement;
                   ++iNode)
                {
                  for (unsigned int jNode = 0; jNode < numberDofsPerElement;
                       ++jNode)
                    {
                      d_blockiNodeIndexNoSym[numberEntriesEachBlockNoSym *
                                               blockCount +
                                             indexCount] = iNode;
                      d_blockjNodeIndexNoSym[numberEntriesEachBlockNoSym *
                                               blockCount +
                                             indexCount] = jNode;
                      count += 1;
                      indexCount += 1;
                      if (count % numberEntriesEachBlockNoSym == 0)
                        {
                          blockCount += 1;
                          indexCount = 0;
                        }
                    }
                }
#endif
            }

          iElemCount++;

        } // if cell locally owned condition 

    } // cell iter loop



  //
  // Fill FE datastructures required for density computation from wavefunctions
  //
  d_densityGaussQuadShapeFunctionValues.clear();
  d_densityGaussQuadShapeFunctionGradientValues.clear();

  d_densityGaussQuadShapeFunctionValues.resize(numberQuadraturePoints *
                                                 numberDofsPerElement,
                                               0.0);
  d_densityGaussQuadShapeFunctionGradientValues.resize(
    numberPhysicalCells * numberQuadraturePoints * 3 * numberDofsPerElement,
    0.0);

  cellPtr =
    dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
      .begin_active();
  typename dealii::DoFHandler<3>::active_cell_iterator endcPtr =
    dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
      .end();

  unsigned int iElem = 0;
  for (; cellPtr != endcPtr; ++cellPtr)
    if (cellPtr->is_locally_owned())
      {
        fe_values.reinit(cellPtr);

        for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
             ++q_point)
          for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            {
              const dealii::Tensor<1, 3, double> &shape_grad =
                fe_values.shape_grad(iNode, q_point);

              d_densityGaussQuadShapeFunctionGradientValues
                [iElem * numberQuadraturePoints * 3 * numberDofsPerElement +
                 q_point * 3 * numberDofsPerElement + iNode] = shape_grad[0];
            }

        for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
             ++q_point)
          for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            {
              const dealii::Tensor<1, 3, double> &shape_grad =
                fe_values.shape_grad(iNode, q_point);

              d_densityGaussQuadShapeFunctionGradientValues
                [iElem * numberQuadraturePoints * 3 * numberDofsPerElement +
                 q_point * 3 * numberDofsPerElement + numberDofsPerElement +
                 iNode] = shape_grad[1];
            }

        for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
             ++q_point)
          for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            {
              const dealii::Tensor<1, 3, double> &shape_grad =
                fe_values.shape_grad(iNode, q_point);

              d_densityGaussQuadShapeFunctionGradientValues
                [iElem * numberQuadraturePoints * 3 * numberDofsPerElement +
                 q_point * 3 * numberDofsPerElement + 2 * numberDofsPerElement +
                 iNode] = shape_grad[2];
            }


        if (iElem == 0)
          {
            for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
              for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
                   ++q_point)
                {
                  const double val = fe_values.shape_value(iNode, q_point);
                  d_densityGaussQuadShapeFunctionValues[q_point *
                                                          numberDofsPerElement +
                                                        iNode] = val;
                }
          }

        iElem++;
      }


  QGaussLobatto<3> quadratureGl(C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>() +
                                1);
  FEValues<3>      fe_valuesGl(dftPtr->matrix_free_data
                            .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                            .get_fe(),
                          quadratureGl,
                          update_values);
  const unsigned int numberQuadraturePointsGl = quadratureGl.size();

  //
  // resize data members
  //
  d_densityGlQuadShapeFunctionValues.clear();
  d_densityGlQuadShapeFunctionValues.resize(numberQuadraturePointsGl *
                                              numberDofsPerElement,
                                            0.0);

  cellPtr =
    dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
      .begin_active();
  fe_valuesGl.reinit(cellPtr);

  for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
    for (unsigned int q_point = 0; q_point < numberQuadraturePointsGl;
         ++q_point)
      {
        const double val = fe_valuesGl.shape_value(iNode, q_point);
        d_densityGlQuadShapeFunctionValues[q_point * numberDofsPerElement +
                                           iNode] = val;
      }
}
