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
  const char transA1 = 'N',transB1 = 'T';
  const double alpha = 1.0;
  const double beta = 1.0;
  const unsigned int inc = 1;
  const unsigned int numberNodesPerElementSquare = d_numberNodesPerElement*d_numberNodesPerElement;
  const unsigned int sizeNiNj = d_numberNodesPerElement*(d_numberNodesPerElement + 1)/2;
  const unsigned int fullSizeNiNj = d_numberNodesPerElement*d_numberNodesPerElement;
  unsigned int numBlocks = FEOrder + 1;
  unsigned int numberEntriesEachBlock = sizeNiNj/numBlocks;
  unsigned int count = 0;
  unsigned int blockCount = 0;
  unsigned int indexCount = 0;
  unsigned int flag = 0;

  std::vector<double> cellHamiltonianMatrixExternalPotCorr(totalLocallyOwnedCells*sizeNiNj,0.0);
  if ((dftParameters::isPseudopotential ||
       dftParameters::smearedNuclearCharges) &&
      !d_isStiffnessMatrixExternalPotCorrComputed)
    {
      const unsigned int numberDofsPerElement =
        dftPtr->matrix_free_data
          .get_dof_handler(dftPtr->d_densityDofHandlerIndex)
          .get_fe()
          .dofs_per_cell;


       FEEvaluation<3,
                   FEOrder,
                   C_num1DQuadLPSP<FEOrder>() * C_numCopies1DQuadLPSP(),
                   1,
                   double>
	           fe_eval(dftPtr->matrix_free_data, 0, d_externalPotCorrQuadratureId);
      
      const unsigned int numberQuadraturePoints = fe_eval.n_q_points;
      
      //d_cellHamiltonianMatrixExternalPotCorr.clear();
      //d_cellHamiltonianMatrixExternalPotCorr.resize(sizeNiNj*totalLocallyOwnedCells);

    
      std::vector<double> NiNjLpspQuad_currentBlock(numberEntriesEachBlock*numberQuadraturePoints,0.0);
        

      AssertThrow(
        dftPtr->matrix_free_data.get_quadrature(d_externalPotCorrQuadratureId)
            .size() == numberQuadraturePoints,
        dealii::ExcMessage(
          "DFT-FE Error: mismatch in quadrature rule usage in computeHamiltonianMatrix."));

      /*dgemm_(&transA,
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
	     &sizeNiNj);*/

      
      
      while(blockCount < numBlocks)
	{
	  for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
	    {
	      flag = 0;
	      for(unsigned int iNode = d_blockiNodeIndex[numberEntriesEachBlock*blockCount]; iNode < numberDofsPerElement; ++iNode)
		{
                  double shapeI = d_shapeFunctionLpspQuadData[numberDofsPerElement*q_point + iNode];
		  for(unsigned int jNode = d_blockjNodeIndex[numberEntriesEachBlock*blockCount+indexCount]; jNode < numberDofsPerElement; ++jNode)
		    {
		      double shapeJ = d_shapeFunctionLpspQuadData[numberDofsPerElement*q_point + jNode];
		      NiNjLpspQuad_currentBlock[numberEntriesEachBlock*q_point + indexCount]= shapeI*shapeJ;
		      indexCount += 1;
		      if(indexCount%numberEntriesEachBlock == 0)
			{
			  flag = 1;
			  indexCount = 0;
			  break;
			}
		    }//jNode

		  if(flag == 1)
		    {
		      if(q_point == (numberQuadraturePoints - 1))
			{
			  dgemm_(&transA1,
				 &transB1,
				 &totalLocallyOwnedCells,//M
				 &numberEntriesEachBlock,//N
				 &numberQuadraturePoints,//K
				 &alpha,
				 &d_vEffExternalPotCorrJxW[0],
				 &totalLocallyOwnedCells,
				 &NiNjLpspQuad_currentBlock[0],
				 &numberEntriesEachBlock,
				 &beta,
				 &cellHamiltonianMatrixExternalPotCorr[totalLocallyOwnedCells*numberEntriesEachBlock*blockCount],
				 &totalLocallyOwnedCells);
		      
			  blockCount += 1;

			}
		      break;
		    }

		}
	      

	    }
      	}

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
  const unsigned int numberQuadraturePointsTimesThree = 3*numberQuadraturePoints;
  const unsigned int numberQuadraturePointsTimesNine = 9*numberQuadraturePoints;

  //
  //create temp storage for stiffness matrix across all cells
  //
  count = 0;
  blockCount = 0;
  indexCount = 0;
  flag = 0;
  
  std::vector<double> cellHamiltonianMatrix(totalLocallyOwnedCells*sizeNiNj,0.0);
  std::vector<double> NiNj_currentBlock(numberEntriesEachBlock*numberQuadraturePoints,0.0);
 
 while(blockCount < numBlocks)
{ 
  for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
    {
      flag = 0;
      for(unsigned int iNode = d_blockiNodeIndex[numberEntriesEachBlock*blockCount]; iNode < numberDofsPerElement; ++iNode)
	{
	  double shapeI = d_shapeFunctionData[numberDofsPerElement*q_point + iNode];
	  for(unsigned int jNode = d_blockjNodeIndex[numberEntriesEachBlock*blockCount+indexCount]; jNode < numberDofsPerElement; ++jNode)
	    {
	      double shapeJ = d_shapeFunctionData[numberDofsPerElement*q_point + jNode];
	      NiNj_currentBlock[numberEntriesEachBlock*q_point + indexCount] = shapeI*shapeJ;
	      indexCount += 1;
	      if(indexCount%numberEntriesEachBlock == 0)
		{
		  flag = 1;
		  indexCount = 0;
		  break;
		}
	    }//jNode
	  if(flag == 1)
	    {
	      if(q_point == (numberQuadraturePoints - 1))
		{
		  dgemm_(&transA1,
			 &transB1,
			 &totalLocallyOwnedCells,//M
			 &numberEntriesEachBlock,//N
			 &numberQuadraturePoints,//K
			 &alpha,
			 &d_vEffJxW[0],
			 &totalLocallyOwnedCells,
			 &NiNj_currentBlock[0],
			 &numberEntriesEachBlock,
			 &beta,
			 &cellHamiltonianMatrix[totalLocallyOwnedCells*numberEntriesEachBlock*blockCount],
			 &totalLocallyOwnedCells);
		      
		  blockCount += 1;

		}
	      break;
	    }
	}//iNode
    }
 }
     

 


  if(dftParameters::xcFamilyType == "GGA")
    {
      std::vector<double> gradNiNjPlusgradNjNi_currentBlock(numberEntriesEachBlock*3*numberQuadraturePoints,0.0);
      blockCount = 0;
      indexCount = 0;
      while(blockCount < numBlocks)
	{
	   for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
	     {
	       flag = 0;
	       for(unsigned int iNode = d_blockiNodeIndex[numberEntriesEachBlock*blockCount]; iNode < numberDofsPerElement; ++iNode)
		 {
		   double shapeGradXRefINode = d_shapeFunctionGradientValueRefX[numberDofsPerElement*q_point + iNode];
		   double shapeGradYRefINode = d_shapeFunctionGradientValueRefY[numberDofsPerElement*q_point + iNode];
		   double shapeGradZRefINode = d_shapeFunctionGradientValueRefZ[numberDofsPerElement*q_point + iNode];
		   double shapeI = d_shapeFunctionData[numberDofsPerElement*q_point + iNode];
		    for(unsigned int jNode = d_blockjNodeIndex[numberEntriesEachBlock*blockCount+indexCount]; jNode < numberDofsPerElement; ++jNode)
		      {
			double shapeJ = d_shapeFunctionData[numberDofsPerElement*q_point + jNode];
			gradNiNjPlusgradNjNi_currentBlock[3*numberEntriesEachBlock*q_point + indexCount] = shapeGradXRefINode*shapeJ + shapeI*d_shapeFunctionGradientValueRefX[numberDofsPerElement*q_point + jNode];
			gradNiNjPlusgradNjNi_currentBlock[3*numberEntriesEachBlock*q_point + numberEntriesEachBlock + indexCount] = shapeGradYRefINode*shapeJ + shapeI*d_shapeFunctionGradientValueRefY[numberDofsPerElement*q_point + jNode];
			gradNiNjPlusgradNjNi_currentBlock[3*numberEntriesEachBlock*q_point + 2*numberEntriesEachBlock + indexCount] = shapeGradZRefINode*shapeJ + shapeI*d_shapeFunctionGradientValueRefZ[numberDofsPerElement*q_point + jNode];
			indexCount += 1;
			if(indexCount%numberEntriesEachBlock == 0)
			  {
			    flag = 1;
			    indexCount = 0;
			    break;
			  }
		      }//jnode
		    if(flag == 1)
		      {
			if(q_point == (numberQuadraturePoints - 1))
			  {
			    dgemm_(&transA1,
				   &transB1,
				   &totalLocallyOwnedCells,//M
				   &numberEntriesEachBlock,//N
				   &numberQuadraturePointsTimesThree,//K
				   &alpha,
				   &d_invJacderExcWithSigmaTimesGradRhoJxW[0],
				   &totalLocallyOwnedCells,
				   &gradNiNjPlusgradNjNi_currentBlock[0],
				   &numberEntriesEachBlock,
				   &beta,
				   &cellHamiltonianMatrix[totalLocallyOwnedCells*numberEntriesEachBlock*blockCount],
				   &totalLocallyOwnedCells);
			    
			    blockCount += 1;
			  }
			break;
		      }
		 }//iNode
	     }
	}
      
    }
  
  typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

  //
  // access the kPoint coordinates
  //
#ifdef USE_COMPLEX
   std::vector<double> kPointCoors(3,0.0);
  kPointCoors[0] = dftPtr->d_kPointCoordinates[3*kPointIndex + 0];
  kPointCoors[1] = dftPtr->d_kPointCoordinates[3*kPointIndex + 1];
  kPointCoors[2] = dftPtr->d_kPointCoordinates[3*kPointIndex + 2];

  double kSquareTimesHalf = 0.5*(kPointCoors[0]*kPointCoors[0] + kPointCoors[1]*kPointCoors[1] + kPointCoors[2]*kPointCoors[2]);

  std::vector<double> kPointTimesGradNiNj_currentBlock(numberEntriesEachBlock*9*numberQuadraturePoints,0.0);
  std::vector<double> shapeGradRefINode(3,0.0);
  std::vector<double> elementHamiltonianMatrixImag(totalLocallyOwnedCells*sizeNiNj,0.0);
  unsigned int numberEntriesEachBlockComplex = fullSizeNiNj/numBlocks;
 
  
  blockCount = 0;
  indexCount = 0;
  unsigned int dimCount = 0;
  while(blockCount < numBlocks)
    {
      for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
	{
          flag = 0;
	  for(unsigned int iNode = d_blockiNodeIndex[numberEntriesEachBlock*blockCount]; iNode < numberDofsPerElement; ++iNode)
	    {
	      shapeGradRefINode[0] = d_shapeFunctionGradientValueRefX[numberDofsPerElement*q_point + iNode];
	      shapeGradRefINode[1] = d_shapeFunctionGradientValueRefY[numberDofsPerElement*q_point + iNode];
	      shapeGradRefINode[2] = d_shapeFunctionGradientValueRefZ[numberDofsPerElement*q_point + iNode];
	      
	      for(unsigned int jNode = d_blockjNodeIndex[numberEntriesEachBlock*blockCount+indexCount]; jNode < numberDofsPerElement; ++jNode)
		{
		    double shapeJ = d_shapeFunctionData[numberDofsPerElement*q_point + jNode];
		    dimCount = 0;
		    for(unsigned int iDim = 0; iDim < 3; ++iDim)
		      {
			for(unsigned int jDim = 0; jDim < 3; ++jDim)
			  {
			    kPointTimesGradNiNj_currentBlock[9*numberEntriesEachBlock*q_point + dimCount*numberEntriesEachBlock + indexCount] = -kPointCoors[iDim]*shapeGradRefINode[jDim]*shapeJ;
			    dimCount += 1;
			  }
		      }
		    indexCount += 1;
		    if(indexCount%numberEntriesEachBlock == 0)
		      {
			flag = 1;
			indexCount = 0;
			break;
		      }
		}//jNode
	      if(flag == 1)
		{
		  if(q_point == (numberQuadraturePoints - 1))
		    {
		      //dgemm
		      dgemm_(&transA1,
			     &transB1,
			     &totalLocallyOwnedCells,//M
			     &numberEntriesEachBlock,//N
			     &numberQuadraturePointsTimesNine,
			     &alpha,
			     &d_invJacJxW[0],
			     &totalLocallyOwnedCells,
			     &kPointTimesGradNiNj_currentBlock[0],
			     &numberEntriesEachBlock,
			     &beta,
			     &elementHamiltonianMatrixImag[totalLocallyOwnedCells*numberEntriesEachBlock*blockCount],
			     &totalLocallyOwnedCells);
                      blockCount += 1;                    
		    }
		  break;
		}
	    }//iNode
	}
    }
#endif

  //
  // compute cell-level stiffness matrix by going over dealii macrocells
  // which allows efficient integration of cell-level stiffness matrix integrals
  // using dealii vectorized arrays
  unsigned int iElem = 0;
  for (unsigned int iMacroCell = 0; iMacroCell < numberMacroCells; ++iMacroCell)
    {
      //fe_eval.reinit(iMacroCell);
      const unsigned int n_sub_cells =
        dftPtr->matrix_free_data.n_components_filled(iMacroCell);

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
 		  d_cellHamiltonianMatrix[kpointSpinIndex][iElem][numberDofsPerElement*iNode + jNode].real(cellHamiltonianMatrix[totalLocallyOwnedCells*count +
																 iElem]+0.5*d_cellShapeFunctionGradientIntegral[sizeNiNj*iElem + count]+kSquareTimesHalf*d_NiNjIntegral[sizeNiNj*iElem + count]);

		   d_cellHamiltonianMatrix[kpointSpinIndex][iElem]
		    [numberDofsPerElement * iNode + jNode]
		     .imag(elementHamiltonianMatrixImag[totalLocallyOwnedCells*count + iElem]);

#else
                  d_cellHamiltonianMatrix
                    [kpointSpinIndex][iElem]
                    [numberDofsPerElement * iNode + jNode] =cellHamiltonianMatrix[totalLocallyOwnedCells*count +
		    iElem]+0.5*d_cellShapeFunctionGradientIntegral[sizeNiNj*iElem + count];
		
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
		    d_cellHamiltonianMatrix[kpointSpinIndex][iElem][numberDofsPerElement*iNode + jNode] += dataTypes::number(cellHamiltonianMatrixExternalPotCorr[totalLocallyOwnedCells*count + iElem],0.0);
		  
#else
		    d_cellHamiltonianMatrix[kpointSpinIndex][iElem][numberDofsPerElement*iNode + jNode] += cellHamiltonianMatrixExternalPotCorr[totalLocallyOwnedCells*count + iElem];
		  
		  
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
      dealii::AlignedVector<VectorizedArray<double>> elementHamiltonianMatrix;
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
