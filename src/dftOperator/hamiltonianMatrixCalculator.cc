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
void kohnShamDFTOperatorClass<FEOrder>::computeHamiltonianMatrix(const unsigned int kPointIndex, const unsigned int spinIndex)
{

	//
	//Get the number of locally owned cells
	//
	const unsigned int numberMacroCells = dftPtr->matrix_free_data.n_macro_cells();
	const unsigned int totalLocallyOwnedCells = dftPtr->matrix_free_data.n_physical_cells();
	const unsigned int kpointSpinIndex=(1+dftParameters::spinPolarized)*kPointIndex+spinIndex;

	//
	//Resize the cell-level hamiltonian  matrix
	//
	d_cellHamiltonianMatrix[kpointSpinIndex].clear();
	d_cellHamiltonianMatrix[kpointSpinIndex].resize(totalLocallyOwnedCells);

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
		const  unsigned int n_sub_cells = dftPtr->matrix_free_data.n_components_filled(iMacroCell);

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

		std::vector<Tensor<1,3,VectorizedArray<double> > > nonCachedShapeGrad;

		nonCachedShapeGrad.resize(numberDofsPerElement*numberQuadraturePoints);
		for(unsigned int iCell = 0; iCell < n_sub_cells; ++iCell)
		{
			cellPtr = dftPtr->matrix_free_data.get_cell_iterator(iMacroCell,iCell);
			fe_values.reinit(cellPtr);

			for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
				for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
				{
					const Tensor<1,3,double> tempGrad=fe_values.shape_grad(iNode,q_point);
					for(unsigned int idim = 0; idim < 3; ++idim)
						nonCachedShapeGrad[iNode*numberQuadraturePoints+q_point][idim][iCell] =tempGrad[idim];
				}
		}

#ifdef USE_COMPLEX
		std::vector<VectorizedArray<double> > elementHamiltonianMatrixImag;
		elementHamiltonianMatrixImag.resize(numberDofsPerElement*numberDofsPerElement);
		//
		for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
			for(unsigned int jNode = 0; jNode < numberDofsPerElement; ++jNode)
			{
				for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
				{
					const VectorizedArray<double> temp =
						scalar_product(nonCachedShapeGrad[iNode*numberQuadraturePoints+q_point],-kPointCoors)
						*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*jNode+q_point]);

					fe_eval.submit_value(temp,q_point);
				}

				elementHamiltonianMatrixImag[numberDofsPerElement*iNode + jNode] =  fe_eval.integrate_value();

			}//jNode loop
#endif



		if(dftParameters::xc_id == 4)
			for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
				for(unsigned int jNode = 0; jNode < numberDofsPerElement; ++jNode)
				{
					for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
					{
						const Tensor<1,3, VectorizedArray<double> > tempVec =
							nonCachedShapeGrad[iNode*numberQuadraturePoints+q_point]
							*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*jNode+q_point])
							+ nonCachedShapeGrad[jNode*numberQuadraturePoints+q_point]
							*make_vectorized_array(d_shapeFunctionValue[numberQuadraturePoints*iNode+q_point]);

						const VectorizedArray<double> temp =
							make_vectorized_array(2.0)*scalar_product(derExcWithSigmaTimesGradRho(iMacroCell,q_point),tempVec);

						fe_eval.submit_value(temp,q_point);
					}

					elementHamiltonianMatrix[numberDofsPerElement*iNode + jNode] +=  fe_eval.integrate_value();

				}//jNode loop

		for(unsigned int iSubCell = 0; iSubCell < n_sub_cells; ++iSubCell)
		{
			//FIXME: Use functions like mkl_malloc for 64 byte memory alignment.
			d_cellHamiltonianMatrix[kpointSpinIndex][iElem].resize(numberDofsPerElement*numberDofsPerElement,0.0);

			for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
			{
				for(unsigned int jNode = 0; jNode < numberDofsPerElement; ++jNode)
				{
#ifdef USE_COMPLEX
					d_cellHamiltonianMatrix[kpointSpinIndex][iElem][numberDofsPerElement*iNode + jNode].real(elementHamiltonianMatrix[numberDofsPerElement*iNode + jNode][iSubCell]);
					d_cellHamiltonianMatrix[kpointSpinIndex][iElem][numberDofsPerElement*iNode + jNode].imag(elementHamiltonianMatrixImag[numberDofsPerElement*iNode + jNode][iSubCell]);

#else
					d_cellHamiltonianMatrix[kpointSpinIndex][iElem][numberDofsPerElement*iNode + jNode]
						= elementHamiltonianMatrix[numberDofsPerElement*iNode + jNode][iSubCell];

#endif

				}
			}

			iElem += 1;
		}

	}//macrocell loop

}


	template<unsigned int FEOrder>
void kohnShamDFTOperatorClass<FEOrder>::computeKineticMatrix()
{

	//
	//Get the number of locally owned cells
	//
	const unsigned int numberMacroCells = dftPtr->matrix_free_data.n_macro_cells();
	const unsigned int totalLocallyOwnedCells = dftPtr->matrix_free_data.n_physical_cells();

	//
	//Resize the cell-level hamiltonian  matrix
	//
	d_cellHamiltonianMatrix[0].clear();
	d_cellHamiltonianMatrix[0].resize(totalLocallyOwnedCells);

	//
	//Get some FE related Data
	//
	QGauss<3> quadrature(C_num1DQuad<FEOrder>());
	FEEvaluation<3, FEOrder, C_num1DQuad<FEOrder>(), 1, double>  fe_eval(dftPtr->matrix_free_data, 0, 0);
	FEValues<3> fe_values(dftPtr->matrix_free_data.get_dof_handler().get_fe(), quadrature,update_gradients);
	const unsigned int numberDofsPerElement = dftPtr->matrix_free_data.get_dof_handler().get_fe().dofs_per_cell;


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
		const  unsigned int n_sub_cells = dftPtr->matrix_free_data.n_components_filled(iMacroCell);

		for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
		{
			for(unsigned int jNode = 0; jNode < numberDofsPerElement; ++jNode)
			{

				elementHamiltonianMatrix[numberDofsPerElement*iNode + jNode] = d_cellShapeFunctionGradientIntegral[iMacroCell][numberDofsPerElement*iNode + jNode];

			}//jNode loop

		}//iNode loop


		for(unsigned int iSubCell = 0; iSubCell < n_sub_cells; ++iSubCell)
		{
			//FIXME: Use functions like mkl_malloc for 64 byte memory alignment.
			d_cellHamiltonianMatrix[0][iElem].resize(numberDofsPerElement*numberDofsPerElement,0.0);

			if(dftParameters::cellLevelMassMatrixScaling)
			  {
			    for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
			      {
                            	dealii::types::global_dof_index localProcINode = d_flattenedArrayMacroCellLocalProcIndexIdMap[iElem][iNode];
				
				for(unsigned int jNode = 0; jNode < numberDofsPerElement; ++jNode)
				  {
				    dealii::types::global_dof_index localProcJNode = d_flattenedArrayMacroCellLocalProcIndexIdMap[iElem][jNode];
				    
				    double stiffMatrixEntry = d_invSqrtMassVector.local_element(localProcINode)*elementHamiltonianMatrix[numberDofsPerElement*iNode + jNode][iSubCell]*d_invSqrtMassVector.local_element(localProcJNode);
				    d_cellHamiltonianMatrix[0][iElem][numberDofsPerElement*iNode + jNode] = stiffMatrixEntry;
				  
				  }
			      }
			  }
			else
			  {
			    for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
			      {
                            				
				for(unsigned int jNode = 0; jNode < numberDofsPerElement; ++jNode)
				  {
				   
				    d_cellHamiltonianMatrix[0][iElem][numberDofsPerElement*iNode + jNode]
				    = elementHamiltonianMatrix[numberDofsPerElement*iNode + jNode][iSubCell];
				   
				  
				  }
			      }
			  }

			iElem += 1;
		}

	}//macrocell loop

}
