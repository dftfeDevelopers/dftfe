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
// the top level of the DFT-FE udistribution.
//
// ---------------------------------------------------------------------
//
// @author  Phani Motamarri
//

template<unsigned int FEOrder>
void kohnShamDFTOperatorCUDAClass<FEOrder>::preComputeShapeFunctionGradientIntegrals()
{

  //
  //get FE data
  //
  const unsigned int numberPhysicalCells = dftPtr->matrix_free_data.n_physical_cells();
  QGauss<3>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<3> fe_values(dftPtr->matrix_free_data.get_dof_handler().get_fe(), quadrature, update_values | update_gradients | update_JxW_values);
  const unsigned int numberDofsPerElement = dftPtr->matrix_free_data.get_dof_handler().get_fe().dofs_per_cell;
  const unsigned int numberQuadraturePoints = quadrature.size();

  //
  //resize data members
  //
  d_cellShapeFunctionGradientIntegralFlattened.clear();
  d_cellShapeFunctionGradientIntegralFlattened.resize(numberPhysicalCells*numberDofsPerElement*numberDofsPerElement);

  d_cellJxWValues.clear();
  d_cellJxWValues.resize(numberPhysicalCells*numberQuadraturePoints);

  d_shapeFunctionValue.resize(numberQuadraturePoints*numberDofsPerElement,0.0);
  d_shapeFunctionValueInverted.resize(numberQuadraturePoints*numberDofsPerElement,0.0);

  d_shapeFunctionGradientValueX.resize(numberPhysicalCells*numberQuadraturePoints*numberDofsPerElement,0.0);
  d_shapeFunctionGradientValueXInverted.resize(numberPhysicalCells*numberQuadraturePoints*numberDofsPerElement,0.0);

  d_shapeFunctionGradientValueY.resize(numberPhysicalCells*numberQuadraturePoints*numberDofsPerElement,0.0);
  d_shapeFunctionGradientValueYInverted.resize(numberPhysicalCells*numberQuadraturePoints*numberDofsPerElement,0.0);

  d_shapeFunctionGradientValueZ.resize(numberPhysicalCells*numberQuadraturePoints*numberDofsPerElement,0.0);
  d_shapeFunctionGradientValueZInverted.resize(numberPhysicalCells*numberQuadraturePoints*numberDofsPerElement,0.0);


  typename dealii::DoFHandler<3>::active_cell_iterator cellPtr=dftPtr->matrix_free_data.get_dof_handler().begin_active();
  typename dealii::DoFHandler<3>::active_cell_iterator endcPtr = dftPtr->matrix_free_data.get_dof_handler().end();

  unsigned int iElem=0;
  for(; cellPtr!=endcPtr; ++cellPtr)
      if(cellPtr->is_locally_owned())
       {
          fe_values.reinit (cellPtr);
          
          for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
             d_cellJxWValues[iElem*numberQuadraturePoints+q_point]=fe_values.JxW(q_point); 

          for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
          {
              for(unsigned int jNode = 0; jNode < numberDofsPerElement; ++jNode)
              {
                  double shapeFunctionGradientIntegralValue = 0.0;
                  for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
                  {
                      shapeFunctionGradientIntegralValue += (fe_values.shape_grad(iNode,q_point)*fe_values.shape_grad(jNode,q_point))
                                                    *fe_values.JxW(q_point);
                  }
                  d_cellShapeFunctionGradientIntegralFlattened[iElem*numberDofsPerElement*numberDofsPerElement
                                                               +iNode*numberDofsPerElement+jNode]=shapeFunctionGradientIntegralValue;
              }

              for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
              {
                  const dealii::Tensor<1,3,double> & shape_grad=fe_values.shape_grad(iNode,q_point);

                  d_shapeFunctionGradientValueX[iElem*numberDofsPerElement*numberQuadraturePoints
                                                +iNode*numberQuadraturePoints+q_point]=shape_grad[0];
                  d_shapeFunctionGradientValueXInverted[iElem*numberQuadraturePoints*numberDofsPerElement
                                                      +q_point*numberDofsPerElement+iNode]=shape_grad[0];

                  d_shapeFunctionGradientValueY[iElem*numberDofsPerElement*numberQuadraturePoints
                                                +iNode*numberQuadraturePoints+q_point]=shape_grad[1];
                  d_shapeFunctionGradientValueYInverted[iElem*numberQuadraturePoints*numberDofsPerElement
                                                      +q_point*numberDofsPerElement+iNode]=shape_grad[1];

                  d_shapeFunctionGradientValueZ[iElem*numberDofsPerElement*numberQuadraturePoints
                                                +iNode*numberQuadraturePoints+q_point]=shape_grad[2];
                  d_shapeFunctionGradientValueZInverted[iElem*numberQuadraturePoints*numberDofsPerElement
                                                      +q_point*numberDofsPerElement+iNode]=shape_grad[2];

              }
          }
  
              if(iElem == 0)
                 for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
                    for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
                    {
                       const double val=fe_values.shape_value(iNode,q_point);
                       d_shapeFunctionValue[numberQuadraturePoints*iNode + q_point] = val;
                       d_shapeFunctionValueInverted[q_point*numberDofsPerElement+iNode] = val;
                    }


          iElem++;
       }

    d_shapeFunctionValueDevice=d_shapeFunctionValue;
    d_shapeFunctionValueInvertedDevice=d_shapeFunctionValueInverted;

    d_shapeFunctionGradientValueXDevice=d_shapeFunctionGradientValueX;
    d_shapeFunctionGradientValueXInvertedDevice=d_shapeFunctionGradientValueXInverted;

    d_shapeFunctionGradientValueYDevice=d_shapeFunctionGradientValueY;
    d_shapeFunctionGradientValueYInvertedDevice=d_shapeFunctionGradientValueYInverted;

    d_shapeFunctionGradientValueZDevice=d_shapeFunctionGradientValueZ;
    d_shapeFunctionGradientValueZInvertedDevice=d_shapeFunctionGradientValueZInverted;

    d_cellShapeFunctionGradientIntegralFlattenedDevice=d_cellShapeFunctionGradientIntegralFlattened;
    d_cellJxWValuesDevice=d_cellJxWValues;

    if (dftParameters::mixingMethod=="ANDERSON_WITH_KERKER" || (dftParameters::isBOMD))
    {
	  QGaussLobatto<3>  quadratureGl(C_num1DQuad<C_num1DKerkerPoly<FEOrder>()>());
	  FEValues<3> fe_valuesGl(dftPtr->matrix_free_data.get_dof_handler().get_fe(), quadratureGl, update_values | update_gradients);
	  const unsigned int numberQuadraturePointsGl = quadratureGl.size();

	  //
	  //resize data members
	  //
	  std::vector<double> glShapeFunctionValueInverted(numberQuadraturePointsGl*numberDofsPerElement,0.0);

	  std::vector<double> glShapeFunctionGradientValueXInverted(numberPhysicalCells*numberQuadraturePointsGl*numberDofsPerElement,0.0);

	  std::vector<double> glShapeFunctionGradientValueYInverted(numberPhysicalCells*numberQuadraturePointsGl*numberDofsPerElement,0.0);

	  std::vector<double> glShapeFunctionGradientValueZInverted(numberPhysicalCells*numberQuadraturePointsGl*numberDofsPerElement,0.0);


	  typename dealii::DoFHandler<3>::active_cell_iterator cellPtr=dftPtr->matrix_free_data.get_dof_handler().begin_active();
	  typename dealii::DoFHandler<3>::active_cell_iterator endcPtr = dftPtr->matrix_free_data.get_dof_handler().end();

	  unsigned int iElem=0;
	  for(; cellPtr!=endcPtr; ++cellPtr)
	      if(cellPtr->is_locally_owned())
	       {
		  fe_valuesGl.reinit (cellPtr);


		  for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
		      for(unsigned int q_point = 0; q_point < numberQuadraturePointsGl; ++q_point)
		      {
			  const dealii::Tensor<1,3,double> & shape_grad=fe_values.shape_grad(iNode,q_point);

			  glShapeFunctionGradientValueXInverted[iElem*numberQuadraturePointsGl*numberDofsPerElement
							      +q_point*numberDofsPerElement+iNode]=shape_grad[0];

			  glShapeFunctionGradientValueYInverted[iElem*numberQuadraturePointsGl*numberDofsPerElement
							      +q_point*numberDofsPerElement+iNode]=shape_grad[1];

			  glShapeFunctionGradientValueZInverted[iElem*numberQuadraturePointsGl*numberDofsPerElement
							      +q_point*numberDofsPerElement+iNode]=shape_grad[2];

		      }
	  
		      if(iElem == 0)
			 for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
			    for(unsigned int q_point = 0; q_point < numberQuadraturePointsGl; ++q_point)
			    {
			       const double val=fe_valuesGl.shape_value(iNode,q_point);
			       glShapeFunctionValueInverted[q_point*numberDofsPerElement+iNode] = val;
			    }


		  iElem++;
	       }

	    d_glShapeFunctionValueInvertedDevice=glShapeFunctionValueInverted;

	    d_glShapeFunctionGradientValueXInvertedDevice=glShapeFunctionGradientValueXInverted;

	    d_glShapeFunctionGradientValueYInvertedDevice=glShapeFunctionGradientValueYInverted;

	    d_glShapeFunctionGradientValueZInvertedDevice=glShapeFunctionGradientValueZInverted;
    }

    if (dftParameters::useHigherQuadNLP)
    {
	  QGauss<3>  quadratureNLP(C_num1DQuadPSP<FEOrder>());
	  FEValues<3> fe_valuesNLP(dftPtr->matrix_free_data.get_dof_handler().get_fe(), quadratureNLP, update_values);
	  const unsigned int numberQuadraturePointsNLP = quadratureNLP.size();

	  //
	  //resize data members
	  //
	  std::vector<double> nlpShapeFunctionValueInverted(numberQuadraturePointsNLP*numberDofsPerElement,0.0);

	  typename dealii::DoFHandler<3>::active_cell_iterator cellPtr=dftPtr->matrix_free_data.get_dof_handler().begin_active();
          typename dealii::DoFHandler<3>::active_cell_iterator endcPtr = dftPtr->matrix_free_data.get_dof_handler().end();

		 
	  for(; cellPtr!=endcPtr; ++cellPtr)
	      if(cellPtr->is_locally_owned())
	       { 
                     fe_valuesNLP.reinit (cellPtr);
                     break;
               }


	  for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
	    for(unsigned int q_point = 0; q_point < numberQuadraturePointsNLP; ++q_point)
	    {
	       const double val=fe_valuesNLP.shape_value(iNode,q_point);
	       nlpShapeFunctionValueInverted[q_point*numberDofsPerElement+iNode] = val;
	    }

	  d_shapeFunctionValueNLPInvertedDevice=nlpShapeFunctionValueInverted;
    }
}
