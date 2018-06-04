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

  //
  //get FE data
  //
  const unsigned int numberMacroCells = dftPtr->matrix_free_data.n_macro_cells();
  const unsigned int numberPhysicalCells = dftPtr->matrix_free_data.n_physical_cells();
  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values(dftPtr->matrix_free_data.get_dof_handler().get_fe(), quadrature, update_values | update_gradients | update_JxW_values);
  const unsigned int numberDofsPerElement = dftPtr->matrix_free_data.get_dof_handler().get_fe().dofs_per_cell;
  const unsigned int numberQuadraturePoints = quadrature.size();

  //
  //resize data members
  //
  d_cellShapeFunctionGradientIntegral.resize(numberMacroCells);
  d_cellShapeFunctionGradientValue.reinit(TableIndices<4>(numberMacroCells,numberDofsPerElement,numberQuadraturePoints,3));
  d_shapeFunctionValue.resize(numberQuadraturePoints*numberDofsPerElement,0.0);
  std::vector<std::vector<std::vector<std::vector<double> > > > tempShapeFuncGradData;
  tempShapeFuncGradData.resize(numberMacroCells);



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
      tempShapeFuncGradData.resize(n_sub_cells);

      for(unsigned int iCell = 0; iCell < n_sub_cells; ++iCell)
	{
	  cellPtr = dftPtr->matrix_free_data.get_cell_iterator(iMacroCell,iCell);
	  fe_values.reinit(cellPtr);
	  tempShapeFuncGradData[iCell].resize(numberDofsPerElement);

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


	  for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
	    {
	      tempShapeFuncGradData[iCell][iNode].resize(numberQuadraturePoints);
	      for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
		{
		  tempShapeFuncGradData[iCell][iNode][q_point].resize(3);
		  tempShapeFuncGradData[iCell][iNode][q_point][0] = fe_values.shape_grad(iNode,q_point)[0];
		  tempShapeFuncGradData[iCell][iNode][q_point][1] = fe_values.shape_grad(iNode,q_point)[1];
		  tempShapeFuncGradData[iCell][iNode][q_point][2] = fe_values.shape_grad(iNode,q_point)[2];
		}

	    }


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


      for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
	{
	  for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
	    {
	      VectorizedArray<double> gradX,gradY,gradZ;
	      for(unsigned int iCell = 0; iCell < n_sub_cells; ++iCell)
		{
		  gradX[iCell] = tempShapeFuncGradData[iCell][iNode][q_point][0];
		  gradY[iCell] = tempShapeFuncGradData[iCell][iNode][q_point][1];
		  gradZ[iCell] = tempShapeFuncGradData[iCell][iNode][q_point][2];
		}

	      d_cellShapeFunctionGradientValue(iMacroCell,iNode,q_point,0) = gradX;
	      d_cellShapeFunctionGradientValue(iMacroCell,iNode,q_point,1) = gradY;
	      d_cellShapeFunctionGradientValue(iMacroCell,iNode,q_point,2) = gradZ;

	    }//q_point loop
	}//iNodeloop

    }//macrocell loop

}


