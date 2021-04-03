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
// @author  Phani Motamarri
//



template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::computeHamiltonianMatrix(
  const unsigned int kPointIndex,
  const unsigned int spinIndex)
{
  dealii::TimerOutput computingTimerStandard(
    mpi_communicator,
    pcout,
    dftParameters::reproducible_output || dftParameters::verbosity < 1 ?
      dealii::TimerOutput::never :
      dealii::TimerOutput::every_call,
    dealii::TimerOutput::wall_times);

  computingTimerStandard.enter_section(
    "Elemental Hamiltonian matrix computation on CPU");

  //
  // Get the number of locally owned cells
  //
  const unsigned int numberMacroCells =
    dftPtr->matrix_free_data.n_macro_cells();
  const unsigned int totalLocallyOwnedCells =
    dftPtr->matrix_free_data.n_physical_cells();
  const unsigned int kpointSpinIndex =
    (1 + dftParameters::spinPolarized) * kPointIndex + spinIndex;

  //inputs to blas
  const char transA = 'N',transB = 'N';
  const double alpha = 1.0;
  const double beta = 1.0;
  const unsigned int inc = 1;
  const unsigned int numberNodesPerElementSquare = d_numberNodesPerElement*d_numberNodesPerElement;
  const unsigned int sizeNiNj = d_numberNodesPerElement*(d_numberNodesPerElement + 1)/2;

  if ((dftParameters::isPseudopotential ||
       dftParameters::smearedNuclearCharges) &&
      !d_isStiffnessMatrixExternalPotCorrComputed)
    {
      const unsigned int numberDofsPerElement =
        dftPtr->matrix_free_data
          .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
          .get_fe()
          .dofs_per_cell;
      d_cellHamiltonianMatrixExternalPotCorr.clear();
      d_cellHamiltonianMatrixExternalPotCorr.resize(sizeNiNj*totalLocallyOwnedCells);

      FEEvaluation<3,
                   FEOrder,
                   C_num1DQuadLPSP<FEOrder>() * C_numCopies1DQuadLPSP(),
                   1,
                   double>
	           fe_eval(dftPtr->matrix_free_data, 0, d_externalPotCorrQuadratureId);
      
      const unsigned int numberQuadraturePoints = fe_eval.n_q_points;
      typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

      AssertThrow(
        dftPtr->matrix_free_data.get_quadrature(d_externalPotCorrQuadratureId)
            .size() == numberQuadraturePoints,
        dealii::ExcMessage(
          "DFT-FE Error: mismatch in quadrature rule usage in computeHamiltonianMatrix."));

      /*unsigned int                         iElem = 0;
      VectorizedArray<double>              temp;
      std::vector<VectorizedArray<double>> elementHamiltonianMatrix;
      elementHamiltonianMatrix.resize(numberDofsPerElement *
                                      numberDofsPerElement);

      for (unsigned int iMacroCell = 0; iMacroCell < numberMacroCells;
           ++iMacroCell)
        {
          fe_eval.reinit(iMacroCell);
          const unsigned int n_sub_cells =
            dftPtr->matrix_free_data.n_components_filled(iMacroCell);

          for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            {
              for (unsigned int jNode = iNode; jNode < numberDofsPerElement;
                   ++jNode)
                {
                  for (unsigned int q_point = 0;
                       q_point < numberQuadraturePoints;
                       ++q_point)
                    {
                      temp = d_vEffExternalPotCorr(iMacroCell, q_point) *
                             make_vectorized_array(
                               d_shapeFunctionValueLpspQuad
                                 [numberQuadraturePoints * iNode + q_point] *
                               d_shapeFunctionValueLpspQuad
                                 [numberQuadraturePoints * jNode + q_point]);
                      fe_eval.submit_value(temp, q_point);
                    }

                  elementHamiltonianMatrix[numberDofsPerElement * iNode +
                                           jNode] = fe_eval.integrate_value();

                } // jNode loop
            }     // iNode loop

          for (unsigned int iSubCell = 0; iSubCell < n_sub_cells; ++iSubCell)
            {
              for (unsigned int iNode = 0; iNode < numberDofsPerElement;
                   ++iNode)
                for (unsigned int jNode = iNode; jNode < numberDofsPerElement;
                     ++jNode)
                  d_cellHamiltonianMatrixExternalPotCorr
                    [iElem][numberDofsPerElement * iNode + jNode] =
                      elementHamiltonianMatrix[numberDofsPerElement * iNode +
                                               jNode][iSubCell];

              iElem += 1;
            }

        } // macrocell loop*/

      dgemm_(&transA,
	     &transB,
	     &sizeNiNj,//M
	     &totalLocallyOwnedCells,//N
	     &numberQuadraturePoints,//K
	     &alpha,
	     &d_NiNjLpspQuad[0],
	     &sizeNiNj,
	     &d_vEffExternalPotCorrJxW[0],
	     &numberQuadraturePoints,
	     &beta,
	     &d_cellHamiltonianMatrixExternalPotCorr[0],
	     &sizeNiNj);

      d_isStiffnessMatrixExternalPotCorrComputed = true;
      
    }


 
  
  //
  // Resize the cell-level hamiltonian  matrix
  //
  d_cellHamiltonianMatrix[kpointSpinIndex].clear();
  d_cellHamiltonianMatrix[kpointSpinIndex].resize(totalLocallyOwnedCells);

  //
  // Get some FE related Data
  //
  const Quadrature<3> &quadrature =
    dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
  
  FEEvaluation<3,
               FEOrder,
               C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
               1,
               double>
    fe_eval(dftPtr->matrix_free_data, 0, 0);
  
  FEValues<3> fe_values(dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex).get_fe(),
                        quadrature,
                        update_gradients);
  
  const unsigned int numberDofsPerElement =
    dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex).get_fe().dofs_per_cell;
  
  const unsigned int numberQuadraturePoints = quadrature.size();

  //
  //create temp storage for stiffness matrix across all cells
  //
  std::vector<dataTypes::number> cellHamiltonianMatrix(totalLocallyOwnedCells*sizeNiNj,0.0);
  dgemm_(&transA,
	 &transB,
	 &sizeNiNj,//M
	 &totalLocallyOwnedCells,//N
	 &numberQuadraturePoints,//K
	 &alpha,
	 &d_NiNj[0],
	 &sizeNiNj,
	 &d_vEffJxW[0],
	 &numberQuadraturePoints,
	 &beta,
	 &cellHamiltonianMatrix[0],
	 &sizeNiNj);
  
  typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

  //
  // access the kPoint coordinates
  //
#ifdef USE_COMPLEX
  Tensor<1, 3, VectorizedArray<double>> kPointCoors;
  kPointCoors[0] =
    make_vectorized_array(dftPtr->d_kPointCoordinates[3 * kPointIndex + 0]);
  kPointCoors[1] =
    make_vectorized_array(dftPtr->d_kPointCoordinates[3 * kPointIndex + 1]);
  kPointCoors[2] =
    make_vectorized_array(dftPtr->d_kPointCoordinates[3 * kPointIndex + 2]);
  double kSquareTimesHalf =
    0.5 * (dftPtr->d_kPointCoordinates[3 * kPointIndex + 0] *
             dftPtr->d_kPointCoordinates[3 * kPointIndex + 0] +
           dftPtr->d_kPointCoordinates[3 * kPointIndex + 1] *
             dftPtr->d_kPointCoordinates[3 * kPointIndex + 1] +
           dftPtr->d_kPointCoordinates[3 * kPointIndex + 2] *
             dftPtr->d_kPointCoordinates[3 * kPointIndex + 2]);
  VectorizedArray<double> halfkSquare = make_vectorized_array(kSquareTimesHalf);
#endif

  //
  // compute cell-level stiffness matrix by going over dealii macrocells
  // which allows efficient integration of cell-level stiffness matrix integrals
  // using dealii vectorized arrays
  unsigned int iElem = 0;
  //std::vector<VectorizedArray<double>> elementHamiltonianMatrix;
  //elementHamiltonianMatrix.resize(numberDofsPerElement * numberDofsPerElement);
  for (unsigned int iMacroCell = 0; iMacroCell < numberMacroCells; ++iMacroCell)
    {
      //fe_eval.reinit(iMacroCell);
      const unsigned int n_sub_cells =
        dftPtr->matrix_free_data.n_components_filled(iMacroCell);

      /*std::vector<Tensor<1, 3, VectorizedArray<double>>> nonCachedShapeGrad;

      nonCachedShapeGrad.resize(numberDofsPerElement * numberQuadraturePoints);
      for (unsigned int iCell = 0; iCell < n_sub_cells; ++iCell)
        {
          cellPtr = dftPtr->matrix_free_data.get_cell_iterator(
            iMacroCell, iCell, dftPtr->d_densityDofHandlerIndex);
          fe_values.reinit(cellPtr);

          for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
                 ++q_point)
              {
                const Tensor<1, 3, double> tempGrad =
                  fe_values.shape_grad(iNode, q_point);
                for (unsigned int idim = 0; idim < 3; ++idim)
                  nonCachedShapeGrad[iNode * numberQuadraturePoints + q_point]
                                    [idim][iCell] = tempGrad[idim];
              }
        }

      std::vector<VectorizedArray<double>>
        shapeGradDotDerExcWithSigmaTimesGradRho;

      if (dftParameters::xcFamilyType == "GGA")
        {
          shapeGradDotDerExcWithSigmaTimesGradRho.resize(
            numberDofsPerElement * numberQuadraturePoints);
          for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
                 ++q_point)
              shapeGradDotDerExcWithSigmaTimesGradRho[iNode *
                                                        numberQuadraturePoints +
                                                      q_point] =
                make_vectorized_array(2.0) *
                scalar_product(
                  derExcWithSigmaTimesGradRho(iMacroCell, q_point),
                  nonCachedShapeGrad[iNode * numberQuadraturePoints + q_point]);
        }

      for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
        {
          for (unsigned int jNode = iNode; jNode < numberDofsPerElement;
               ++jNode)
            {
              for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
                   ++q_point)
                {
                  const VectorizedArray<double> shapei = make_vectorized_array(
                    d_shapeFunctionValue[numberQuadraturePoints * iNode +
                                         q_point]);
                  const VectorizedArray<double> shapej = make_vectorized_array(
                    d_shapeFunctionValue[numberQuadraturePoints * jNode +
                                         q_point]);
#ifdef USE_COMPLEX
                  VectorizedArray<double> temp =
                    (vEff(iMacroCell, q_point) + halfkSquare) * shapei * shapej;
#else
                  VectorizedArray<double> temp =
                    vEff(iMacroCell, q_point) * shapei * shapej;
#endif

                  if (dftParameters::xcFamilyType == "GGA")
                    {
                      temp += shapeGradDotDerExcWithSigmaTimesGradRho
                                  [iNode * numberQuadraturePoints + q_point] *
                                shapej +
                              shapeGradDotDerExcWithSigmaTimesGradRho
                                  [jNode * numberQuadraturePoints + q_point] *
                                shapei;
                    }

                  fe_eval.submit_value(temp, q_point);
                }

              elementHamiltonianMatrix[numberDofsPerElement * iNode + jNode] =
                make_vectorized_array(0.5) *
                  d_cellShapeFunctionGradientIntegral
                    [iMacroCell][numberDofsPerElement * iNode + jNode] +
                fe_eval.integrate_value();

            } // jNode loop

        }*/   // iNode loop

#ifdef USE_COMPLEX
      std::vector<VectorizedArray<double>> elementHamiltonianMatrixImag;
      elementHamiltonianMatrixImag.resize(numberDofsPerElement *
                                          numberDofsPerElement);
      //
      for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
        for (unsigned int jNode = iNode; jNode < numberDofsPerElement; ++jNode)
          {
            for (unsigned int q_point = 0; q_point < numberQuadraturePoints;
                 ++q_point)
              {
                const VectorizedArray<double> temp =
                  scalar_product(
                    nonCachedShapeGrad[iNode * numberQuadraturePoints +
                                       q_point],
                    -kPointCoors) *
                  make_vectorized_array(
                    d_shapeFunctionValue[numberQuadraturePoints * jNode +
                                         q_point]);

                fe_eval.submit_value(temp, q_point);
              }

            elementHamiltonianMatrixImag[numberDofsPerElement * iNode + jNode] =
              fe_eval.integrate_value();

          } // jNode loop
#endif

      for (unsigned int iSubCell = 0; iSubCell < n_sub_cells; ++iSubCell)
        {
          // FIXME: Use functions like mkl_malloc for 64 byte memory alignment.
          d_cellHamiltonianMatrix[kpointSpinIndex][iElem].resize(
            numberDofsPerElement * numberDofsPerElement, 0.0);
          unsigned int count = 0;
          for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            {
              for (unsigned int jNode = iNode; jNode < numberDofsPerElement;
                   ++jNode)
                {
#ifdef USE_COMPLEX
                  d_cellHamiltonianMatrix
                    [kpointSpinIndex][iElem]
                    [numberDofsPerElement * iNode + jNode]
                      .real(
                        elementHamiltonianMatrix[numberDofsPerElement * iNode +
                                                 jNode][iSubCell]);
                  d_cellHamiltonianMatrix[kpointSpinIndex][iElem]
                                         [numberDofsPerElement * iNode + jNode]
                                           .imag(
                                             elementHamiltonianMatrixImag
                                               [numberDofsPerElement * iNode +
                                                jNode][iSubCell]);

#else
                  d_cellHamiltonianMatrix
                    [kpointSpinIndex][iElem]
                    [numberDofsPerElement * iNode + jNode] =
                      cellHamiltonianMatrix[sizeNiNj*iElem +
                                             count]+0.5*d_cellShapeFunctionGradientIntegral[sizeNiNj*iElem + count];

#endif
              count+=1;
                }
            }

          if (dftParameters::isPseudopotential ||
              dftParameters::smearedNuclearCharges)
{
            unsigned int count = 0;
            for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
              for (unsigned int jNode = iNode; jNode < numberDofsPerElement;
                   ++jNode)
                {
#ifdef USE_COMPLEX
                  /*d_cellHamiltonianMatrix[kpointSpinIndex][iElem]
                                         [numberDofsPerElement * iNode +
                                          jNode] +=
                    dataTypes::number(
                      d_cellHamiltonianMatrixExternalPotCorr
                        [iElem][numberDofsPerElement * iNode + jNode],
			0.0);*/


		  d_cellHamiltonianMatrix[kpointSpinIndex][iElem][numberDofsPerElement*iNode + jNode]+=dataTypes::number(d_cellHamiltonianMatrixExternalPotCorr[numberNodesPerElementSquare*iElem + d_numberNodesPerElement*iNode + jNode],0.0);
		  
#else
                  /*d_cellHamiltonianMatrix[kpointSpinIndex][iElem]
                                         [numberDofsPerElement * iNode +
                                          jNode] +=
                    d_cellHamiltonianMatrixExternalPotCorr
		    [iElem][numberDofsPerElement * iNode + jNode];*/

                 d_cellHamiltonianMatrix[kpointSpinIndex][iElem][numberDofsPerElement*iNode + jNode] += d_cellHamiltonianMatrixExternalPotCorr[sizeNiNj*iElem + count];
		  
		  
#endif
                 count += 1;
                }
}

#ifdef USE_COMPLEX
          for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            for (unsigned int jNode = 0; jNode < iNode; ++jNode)
              d_cellHamiltonianMatrix
                [kpointSpinIndex][iElem][numberDofsPerElement * iNode + jNode] =
                  std::conj(
                    d_cellHamiltonianMatrix[kpointSpinIndex][iElem]
                                           [numberDofsPerElement * jNode +
                                            iNode]);
#else
          for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            for (unsigned int jNode = 0; jNode < iNode; ++jNode)
              d_cellHamiltonianMatrix
                [kpointSpinIndex][iElem][numberDofsPerElement * iNode + jNode] =
                  d_cellHamiltonianMatrix[kpointSpinIndex][iElem]
                                         [numberDofsPerElement * jNode + iNode];
#endif

          iElem += 1;
        }



    } // macrocell loop

  computingTimerStandard.exit_section(
    "Elemental Hamiltonian matrix computation on CPU");
}


template <unsigned int FEOrder, unsigned int FEOrderElectro>
void
kohnShamDFTOperatorClass<FEOrder, FEOrderElectro>::computeKineticMatrix()
{
  //
  // Get the number of locally owned cells
  //
  const unsigned int numberMacroCells =
    dftPtr->matrix_free_data.n_macro_cells();
  const unsigned int totalLocallyOwnedCells =
    dftPtr->matrix_free_data.n_physical_cells();

  //
  // Resize the cell-level hamiltonian  matrix
  //
  d_cellHamiltonianMatrix[0].clear();
  d_cellHamiltonianMatrix[0].resize(totalLocallyOwnedCells);

  //
  // Get some FE related Data
  //
  const Quadrature<3> &quadrature =
    dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
  FEEvaluation<3,
               FEOrder,
               C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
               1,
               double>
                     fe_eval(dftPtr->matrix_free_data, 0, 0);
  FEValues<3>        fe_values(dftPtr->matrix_free_data
                          .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
                          .get_fe(),
                        quadrature,
                        update_gradients);
  const unsigned int numberDofsPerElement =
    dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex)
      .get_fe()
      .dofs_per_cell;


  //
  // compute cell-level stiffness matrix by going over dealii macrocells
  // which allows efficient integration of cell-level stiffness matrix integrals
  // using dealii vectorized arrays
  unsigned int iElem = 0;
  for (unsigned int iMacroCell = 0; iMacroCell < numberMacroCells; ++iMacroCell)
    {
      std::vector<VectorizedArray<double>> elementHamiltonianMatrix;
      elementHamiltonianMatrix.resize(numberDofsPerElement *
                                      numberDofsPerElement);
      fe_eval.reinit(iMacroCell);
      const unsigned int n_sub_cells =
        dftPtr->matrix_free_data.n_components_filled(iMacroCell);

      for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
        {
          for (unsigned int jNode = 0; jNode < numberDofsPerElement; ++jNode)
            {
              elementHamiltonianMatrix[numberDofsPerElement * iNode + jNode] =
                d_cellShapeFunctionGradientIntegral[numberDofsPerElement*numberDofsPerElement*iElem + numberDofsPerElement*iNode + jNode];

            } // jNode loop

        } // iNode loop


      for (unsigned int iSubCell = 0; iSubCell < n_sub_cells; ++iSubCell)
        {
          // FIXME: Use functions like mkl_malloc for 64 byte memory alignment.
          d_cellHamiltonianMatrix[0][iElem].resize(numberDofsPerElement *
                                                     numberDofsPerElement,
                                                   0.0);

          for (unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            {
              for (unsigned int jNode = 0; jNode < numberDofsPerElement;
                   ++jNode)
                {
                  d_cellHamiltonianMatrix[0][iElem][numberDofsPerElement *
                                                      iNode +
                                                    jNode] =
                    elementHamiltonianMatrix[numberDofsPerElement * iNode +
                                             jNode][iSubCell];
                }
            }

          iElem += 1;
        }

    } // macrocell loop
}
