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
  //FEEvaluation<3, FEOrder, C_num1DQuad<FEOrder>(), 1, double>  fe_eval1(dftPtr->matrix_free_data, 0, 0);
  const unsigned int numberDofsPerElement = dftPtr->matrix_free_data.get_dof_handler().get_fe().dofs_per_cell;
  const unsigned int numberQuadraturePoints = quadrature.size();
  typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

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
      //fe_eval1.reinit(iMacroCell);
      for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
	{
	  for(unsigned int jNode = 0; jNode < numberDofsPerElement; ++jNode)
	    {
	      for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
		{
		  VectorizedArray<double> temp = vEff(iMacroCell,q_point)*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*iNode+q_point])*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*jNode+q_point]);

		  fe_eval.submit_value(temp,q_point);
		}
	      
	      elementHamiltonianMatrix[numberDofsPerElement*iNode + jNode] = make_vectorized_array(0.5)*d_cellShapeFunctionGradientIntegral[iMacroCell][numberDofsPerElement*iNode + jNode] + fe_eval.integrate_value();

	    }//jNode loop

	}//iNode loop


      if(dftParameters::xc_id == 4)
	{
	  for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
	    {
	      for(unsigned int jNode = 0; jNode < numberDofsPerElement; ++jNode)
		{
		  for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
		    {
		      VectorizedArray<double> tempx = d_cellShapeFunctionGradientValue(iMacroCell,iNode,q_point,0)*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*jNode+q_point])+ d_cellShapeFunctionGradientValue(iMacroCell,jNode,q_point,0)*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*iNode+q_point]);

		      VectorizedArray<double> tempy = d_cellShapeFunctionGradientValue(iMacroCell,iNode,q_point,1)*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*jNode+q_point])+ d_cellShapeFunctionGradientValue(iMacroCell,jNode,q_point,1)*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*iNode+q_point]);

		      VectorizedArray<double> tempz = d_cellShapeFunctionGradientValue(iMacroCell,iNode,q_point,2)*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*jNode+q_point])+ d_cellShapeFunctionGradientValue(iMacroCell,jNode,q_point,2)*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*iNode+q_point]);

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
#ifdef ENABLE_PERIODIC_BC
		  d_cellHamiltonianMatrix[iElem][numberDofsPerElement*iNode + jNode].real(elementHamiltonianMatrix[numberDofsPerElement*iNode + jNode][iSubCell]);
		  d_cellHamiltonianMatrix[iElem][numberDofsPerElement*iNode + jNode].imag(0.0);
#else
		  d_cellHamiltonianMatrix[iElem][numberDofsPerElement*iNode + jNode] = elementHamiltonianMatrix[numberDofsPerElement*iNode + jNode][iSubCell];
#endif

		}
	    }

	  iElem += 1;
	}

    }//macrocell loop

}


