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
// the top level of the DFT-FE udistribution.
//
// ---------------------------------------------------------------------
//
// @author  Phani Motamarri
//


template<unsigned int FEOrder>
void eigenClass<FEOrder>::preComputeShapeFunctionGradientIntegrals()
{

  const unsigned int numberMacroCells = dftPtr->matrix_free_data.n_macro_cells();
  d_cellShapeFunctionGradientIntegral.resize(numberMacroCells);


  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values(dftPtr->matrix_free_data.get_dof_handler().get_fe(), quadrature, update_values | update_gradients | update_JxW_values);
  
  const unsigned int numberDofsPerElement = dftPtr->matrix_free_data.get_dof_handler().get_fe().dofs_per_cell;
  const unsigned int numberQuadraturePoints = quadrature.size();

  d_shapeFunctionValue.resize(numberQuadraturePoints*numberDofsPerElement,0.0);

  typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

  //
  //compute cell-level shapefunctiongradientintegral generator by going over dealii macrocells 
  //which allows efficient integration of cell-level matrix integrals
  //using dealii vectorized arrays
  for(int iMacroCell = 0; iMacroCell < numberMacroCells; ++iMacroCell)
    {
      std::vector<VectorizedArray<double> > & shapeFunctionGradients = d_cellShapeFunctionGradientIntegral[iMacroCell];
      shapeFunctionGradients.resize(numberDofsPerElement*numberDofsPerElement);
      unsigned int n_sub_cells=dftPtr->matrix_free_data.n_components_filled(iMacroCell);
  
      for(unsigned int iCell = 0; iCell < n_sub_cells; ++iCell)
	{
	  cellPtr = dftPtr->matrix_free_data.get_cell_iterator(iMacroCell,iCell);
	  fe_values.reinit(cellPtr);
	  for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
	    {
	      for(unsigned int jNode = 0; jNode < numberDofsPerElement; ++jNode)
		{
		  double shapeFunctionGradientValue = 0.0;
		  for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
		    {
		      shapeFunctionGradientValue += (fe_values.shape_grad(iNode,q_point)*fe_values.shape_grad(jNode,q_point))*fe_values.JxW(q_point);
		    }

		  shapeFunctionGradients[numberDofsPerElement*iNode + jNode][iCell] = shapeFunctionGradientValue;
		}//j node loop

	    }//i node loop

	  if(iMacroCell == 0 && iCell == 0)
	    {
	      for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
		{
		  for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
		    {
		      d_shapeFunctionValue[numberQuadraturePoints*iNode + q_point] = fe_values.shape_value(iNode,q_point);
		    }
		}
	    }


	}//icell loop

    }//macrocell loop

}


