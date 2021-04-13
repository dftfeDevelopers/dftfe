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
                        update_values | update_gradients| update_jacobians);
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
  unsigned int sizeNiNj = numberDofsPerElement*(numberDofsPerElement + 1)/2;
  d_NiNjLpspQuad.resize(sizeNiNj*numberQuadraturePointsLpsp,0.0);
  //d_NiNj.resize(sizeNiNj*numberQuadraturePoints,0.0);
  d_shapeFunctionData(numberDofsPerElement*numberQuadraturePoints,0.0);
  d_cellShapeFunctionGradientIntegral.resize(numberPhysicalCells*sizeNiNj);

  //
  //some more data members, local variables
  //
  std::vector<double> shapeFunctionGradientValueRef;
 

   if(dftParameters::xcFamilyType == "GGA")
     {
       d_gradNiNjPlusgradNjNi.resize(sizeNiNj*3*numberQuadraturePoints,0.0);
       shapeFunctionGradientValueRef.resize(numberQuadraturePoints * numberDofsPerElement * 3, 0.0);
     }
      
 

  typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;

  //
  // compute cell-level shapefunctiongradientintegral generator by going over
  // dealii macrocells which allows efficient integration of cell-level matrix
  // integrals using dealii vectorized arrays
  unsigned int iElemCount = 0;
  for (int iMacroCell = 0; iMacroCell < numberMacroCells; ++iMacroCell)
    {

      unsigned int n_sub_cells =
        dftPtr->matrix_free_data.n_components_filled(iMacroCell);
     

      for (unsigned int iCell = 0; iCell < n_sub_cells; ++iCell)
        {
          cellPtr = dftPtr->matrix_free_data.get_cell_iterator(
            iMacroCell, iCell, dftPtr->d_densityDofHandlerIndex);
          fe_values_quadplusone.reinit(cellPtr);

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
                    shapeFunctionGradientValue +=
                      (fe_values_quadplusone.shape_grad(iNode, q_point) *
                       fe_values_quadplusone.shape_grad(jNode, q_point)) *
                      fe_values_quadplusone.JxW(q_point);

                  d_cellShapeFunctionGradientIntegral[sizeNiNj*iElemCount + count]
                                         = shapeFunctionGradientValue;
          
                  count += 1;
                } // j node loop

            } // i node loop



          if (iMacroCell == 0 && iCell == 0)
            {
              fe_values.reinit(cellPtr);
              fe_values_lpsp.reinit(cellPtr);

	      
              const std::vector<dealii::DerivativeForm<1, 3, 3>> &jacobians =
              fe_values.get_jacobians();
	      for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
		{
		  for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
 		    {
                      const dealii::Tensor<1, 3, double> &shape_grad_real = fe_values.shape_grad(iNode, q_point);
		      const dealii::Tensor<1, 3, double> &shape_grad_reference =
                        apply_transformation(jacobians[q_point].transpose(),
                                             shape_grad_real);

		      shapeFunctionGradientValueRef[3*numberDofsPerElement*q_point + iNode] = shape_grad_reference[0];
		      shapeFunctionGradientValueRef[3*numberDofsPerElement*q_point + numberDofsPerElement + iNode] = shape_grad_reference[1];
		      shapeFunctionGradientValueRef[3*numberDofsPerElement*q_point + 2*numberDofsPerElement + iNode] = shape_grad_reference[2];
		    }
		}
            
              if(dftParameters::xcFamilyType == "GGA")
		{
		  for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
		    { 
		      unsigned int count = 0;
		      for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
			{ 
			  for(unsigned int jNode = iNode; jNode < numberDofsPerElement; ++jNode)
			    { 
			      d_gradNiNjPlusgradNjNi[3*sizeNiNj*q_point + count] = shapeFunctionGradientValueRef[3*numberDofsPerElement*q_point + iNode]*fe_values.shape_value(jNode,q_point) + fe_values.shape_value(iNode,q_point)*shapeFunctionGradientValueRef[3*numberDofsPerElement*q_point + jNode];
			      d_gradNiNjPlusgradNjNi[3*sizeNiNj*q_point + sizeNiNj + count] = shapeFunctionGradientValueRef[3*numberDofsPerElement*q_point + numberDofsPerElement + iNode]*fe_values.shape_value(jNode,q_point) + fe_values.shape_value(iNode,q_point)*shapeFunctionGradientValueRef[3*numberDofsPerElement*q_point + numberDofsPerElement + jNode];
			      d_gradNiNjPlusgradNjNi[3*sizeNiNj*q_point + 2*sizeNiNj + count] = shapeFunctionGradientValueRef[3*numberDofsPerElement*q_point + 2*numberDofsPerElement + iNode]*fe_values.shape_value(jNode,q_point) + fe_values.shape_value(iNode,q_point)*shapeFunctionGradientValueRef[3*numberDofsPerElement*q_point + 2*numberDofsPerElement + jNode];
			      count+=1;

			    }
			}
		    }
		}

              for(unsigned int q_point = 0; q_point < numberQuadraturePointsLpsp; ++q_point)
		{
		  unsigned int count = 0;
		  for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
		    {
                      for(unsigned int jNode = iNode; jNode < numberDofsPerElement; ++jNode)
			{
                          d_NiNjLpspQuad[sizeNiNj*q_point + count] = fe_values_lpsp.shape_value(iNode,q_point)*fe_values_lpsp.shape_value(jNode,q_point);
			  count+=1;

			}
		    }
		}

              /*for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
                { 
                  unsigned int count = 0;
                  for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
                    { 
                      for(unsigned int jNode = iNode; jNode < numberDofsPerElement; ++jNode)
                        { 
                          d_NiNj[sizeNiNj*q_point + count] = fe_values.shape_value(iNode,q_point)*fe_values.shape_value(jNode,q_point);
                          count+=1;

                        }
                    }
		    }*/

	      for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
                { 
		  for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
                    { 
		      d_shapeFunctionData[numberDofsPerElement*q_point + iNode] = fe_values.shape_value(iNode,q_point);
                    }
		}

	      unsigned int numBlocks = FEOrder + 1;
	      unsigned int numberEntriesEachBlock = sizeNiNj/numBlocks;
	      unsigned int count = 0;
	      unsigned int blockCount = 0;
	      unsigned int indexCount = 0;
	      d_blockiNodeIndex.resize(sizeNiNj);
	      d_blockjNodeIndex.resize(sizeNiNj);

	      for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
		{
		  for(unsigned int jNode = iNode; jNode < numberDofsPerElement; ++jNode)
		    {
		      d_blockiNodeIndex[numberEntriesEachBlock*blockCount+indexCount] = iNode;
		      d_blockjNodeIndex[numberEntriesEachBlock*blockCount+indexCount] = jNode;
		      count += 1;
		      indexCount += 1;
		      if(count%numberEntriesEachBlock == 0)
			{
			  blockCount += 1;
			  indexCount = 0;
			}
			
		    }
		}
	      
            }

          iElemCount++;

        } // icell loop

    } // macrocell loop
}
