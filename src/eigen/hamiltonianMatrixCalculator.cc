// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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



template<unsigned int FEOrder>
void eigenClass<FEOrder>::computeHamiltonianMatrix(unsigned int kPointIndex)
{

  //
  //Get the number of locally owned cells
  //
  const unsigned int numberMacroCells = dftPtr->matrix_free_data.n_macro_cells();
  const unsigned int totalLocallyOwnedCells = dftPtr->matrix_free_data.n_physical_cells();

  //
  //Resize the cell-level hamiltonian  matrix
  //
  d_cellHamiltonianMatrix.clear();
  d_cellHamiltonianMatrix.resize(totalLocallyOwnedCells);

  //
  //Get some FE related Data
  //
  QGauss<3> quadrature(C_num1DQuad<FEOrder>());
  FEEvaluation<3, FEOrder, C_num1DQuad<FEOrder>(), 1, double>  fe_eval(dftPtr->matrix_free_data, 0, 0);
  const unsigned int numberDofsPerElement = dftPtr->matrix_free_data.get_dof_handler().get_fe().dofs_per_cell;
  const unsigned int numberQuadraturePoints = quadrature.size();
  typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

  //
  //access the kPoint coordinates
  //
#ifdef USE_COMPLEX
  Tensor<1,3,VectorizedArray<double> > kPointCoors;
  kPointCoors[0] = make_vectorized_array(dftPtr->d_kPointCoordinates[3*kPointIndex+0]);
  kPointCoors[1] = make_vectorized_array(dftPtr->d_kPointCoordinates[3*kPointIndex+1]);
  kPointCoors[2] = make_vectorized_array(dftPtr->d_kPointCoordinates[3*kPointIndex+2]);
  double kSquareTimesHalf =  0.5*(dftPtr->d_kPointCoordinates[3*kPointIndex+0]*dftPtr->d_kPointCoordinates[3*kPointIndex+0] + dftPtr->d_kPointCoordinates[3*kPointIndex+1]*dftPtr->d_kPointCoordinates[3*kPointIndex+1] + dftPtr->d_kPointCoordinates[3*kPointIndex+2]*dftPtr->d_kPointCoordinates[3*kPointIndex+2]);
  VectorizedArray<double> halfkSquare = make_vectorized_array(kSquareTimesHalf);
#endif

  //
  //compute cell-level stiffness matrix by going over dealii macrocells
  //which allows efficient integration of cell-level stiffness matrix integrals
  //using dealii vectorized arrays
  unsigned int iElem = 0;
  for(unsigned int iMacroCell = 0; iMacroCell < numberMacroCells; ++iMacroCell)
    {
      std::vector<VectorizedArray<double> > elementHamiltonianMatrix;
      elementHamiltonianMatrix.resize(numberDofsPerElement*numberDofsPerElement);
      fe_eval.reinit(iMacroCell);

      for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
	{
	  for(unsigned int jNode = 0; jNode < numberDofsPerElement; ++jNode)
	    {
	      for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
		{

#ifdef USE_COMPLEX
		  VectorizedArray<double> temp = (vEff(iMacroCell,q_point)+halfkSquare)*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*iNode+q_point])*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*jNode+q_point]);
#else
		 VectorizedArray<double> temp = vEff(iMacroCell,q_point)*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*iNode+q_point])*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*jNode+q_point]); 
#endif
		  fe_eval.submit_value(temp,q_point);
		}

	      elementHamiltonianMatrix[numberDofsPerElement*iNode + jNode] = make_vectorized_array(0.5)*d_cellShapeFunctionGradientIntegral[iMacroCell][numberDofsPerElement*iNode + jNode] + fe_eval.integrate_value();

	    }//jNode loop

	}//iNode loop


#ifdef USE_COMPLEX     
      std::vector<VectorizedArray<double> > elementHamiltonianMatrixImag;
      elementHamiltonianMatrixImag.resize(numberDofsPerElement*numberDofsPerElement);
      //
      for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
	{
	  for(unsigned int jNode = 0; jNode < numberDofsPerElement; ++jNode)
	    {
	      for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
		{
		  VectorizedArray<double> temp = ((d_cellShapeFunctionGradientValue(iMacroCell,iNode,3*q_point))*(-kPointCoors[0])+(d_cellShapeFunctionGradientValue(iMacroCell,iNode,3*q_point+1))*(-kPointCoors[1])+(d_cellShapeFunctionGradientValue(iMacroCell,iNode,3*q_point+2))*(-kPointCoors[2]))*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*jNode+q_point]) ;

		  fe_eval.submit_value(temp,q_point);
		}
	      
	      elementHamiltonianMatrixImag[numberDofsPerElement*iNode + jNode] =  fe_eval.integrate_value();

	    }//jNode loop

	}//iNode loop
#endif



      if(dftParameters::xc_id == 4)
	{
	  for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
	    {
	      for(unsigned int jNode = 0; jNode < numberDofsPerElement; ++jNode)
		{
		  for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
		    {
		      VectorizedArray<double> tempx = d_cellShapeFunctionGradientValue(iMacroCell,iNode,3*q_point)*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*jNode+q_point])+ d_cellShapeFunctionGradientValue(iMacroCell,jNode,3*q_point)*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*iNode+q_point]);

		      VectorizedArray<double> tempy = d_cellShapeFunctionGradientValue(iMacroCell,iNode,3*q_point+1)*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*jNode+q_point])+ d_cellShapeFunctionGradientValue(iMacroCell,jNode,3*q_point+1)*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*iNode+q_point]);

		      VectorizedArray<double> tempz = d_cellShapeFunctionGradientValue(iMacroCell,iNode,3*q_point+2)*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*jNode+q_point])+ d_cellShapeFunctionGradientValue(iMacroCell,jNode,3*q_point+2)*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*iNode+q_point]);

		      VectorizedArray<double> temp = derExcWithSigmaTimesGradRho(iMacroCell,q_point,0)*tempx + derExcWithSigmaTimesGradRho(iMacroCell,q_point,1)*tempy + derExcWithSigmaTimesGradRho(iMacroCell,q_point,2)*tempz;

		      fe_eval.submit_value(make_vectorized_array(2.0)*temp,q_point);
		    }

		  elementHamiltonianMatrix[numberDofsPerElement*iNode + jNode] +=  fe_eval.integrate_value();

		}//jNode loop

	    }//iNode loop

	}


      const  unsigned int n_sub_cells = dftPtr->matrix_free_data.n_components_filled(iMacroCell);


      for(unsigned int iSubCell = 0; iSubCell < n_sub_cells; ++iSubCell)
	{
	  d_cellHamiltonianMatrix[iElem].resize(numberDofsPerElement*numberDofsPerElement,0.0);
	  for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
	    {
	      for(unsigned int jNode = 0; jNode < numberDofsPerElement; ++jNode)
		{
#ifdef USE_COMPLEX
		  d_cellHamiltonianMatrix[iElem][numberDofsPerElement*iNode + jNode].real(elementHamiltonianMatrix[numberDofsPerElement*iNode + jNode][iSubCell]);
		  d_cellHamiltonianMatrix[iElem][numberDofsPerElement*iNode + jNode].imag(elementHamiltonianMatrixImag[numberDofsPerElement*iNode + jNode][iSubCell]);
#else
		  d_cellHamiltonianMatrix[iElem][numberDofsPerElement*iNode + jNode] = elementHamiltonianMatrix[numberDofsPerElement*iNode + jNode][iSubCell];
#endif

		}
	    }

	  iElem += 1;
	}

    }//macrocell loop

}


