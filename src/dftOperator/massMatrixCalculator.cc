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
// @author  Phani Motamarri
//



template<unsigned int FEOrder>
void kohnShamDFTOperatorClass<FEOrder>::computeMassMatrix()
{

  //
  //Get the number of locally owned cells
  //
  const unsigned int numberMacroCells = dftPtr->matrix_free_data.n_macro_cells();
  const unsigned int totalLocallyOwnedCells = dftPtr->matrix_free_data.n_physical_cells();

  //
  //Resize the cell-level hamiltonian  matrix
  //
  d_cellMassMatrix.clear();
  d_cellMassMatrix.resize(totalLocallyOwnedCells);

  //
  //Get some FE related Data
  //
  QGauss<3> quadrature(C_num1DQuad<FEOrder>());
  FEEvaluation<3, FEOrder, C_num1DQuad<FEOrder>(), 1, double>  fe_eval(dftPtr->matrix_free_data, 0, 0);
  FEValues<3> fe_values(dftPtr->matrix_free_data.get_dof_handler().get_fe(), quadrature,update_gradients);
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
      std::vector<VectorizedArray<double> > elementMassMatrix;
      elementMassMatrix.resize(numberDofsPerElement*numberDofsPerElement);
      fe_eval.reinit(iMacroCell);
      const  unsigned int n_sub_cells = dftPtr->matrix_free_data.n_components_filled(iMacroCell);

      for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
	{
	  for(unsigned int jNode = 0; jNode < numberDofsPerElement; ++jNode)
	    {
	      for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
		{


		 VectorizedArray<double> temp = make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*iNode+q_point])*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*jNode+q_point]);

		  fe_eval.submit_value(temp,q_point);
		}

	      elementMassMatrix[numberDofsPerElement*iNode + jNode] =  fe_eval.integrate_value();

	    }//jNode loop

	}//iNode loop

      std::vector<Tensor<1,3,VectorizedArray<double> > > nonCachedShapeGrad;


      for(unsigned int iSubCell = 0; iSubCell < n_sub_cells; ++iSubCell)
	{
	  //FIXME: Use functions like mkl_malloc for 64 byte memory alignment.
	  d_cellMassMatrix[iElem].resize(numberDofsPerElement*numberDofsPerElement,0.0);

	  for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
	    {
	      for(unsigned int jNode = 0; jNode < numberDofsPerElement; ++jNode)
		{
		  d_cellMassMatrix[iElem][numberDofsPerElement*iNode + jNode]
		      = elementMassMatrix[numberDofsPerElement*iNode + jNode][iSubCell];
		}
	    }

	  iElem += 1;
	}

    }//macrocell loop

}


