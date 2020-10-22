// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2020 The Regents of the University of Michigan and DFT-FE authors.
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
// @author  Sambit Das
//

namespace shapeFuncCUDA
{
	__global__
		void computeShapeGradNINJIntegralContribution(const unsigned int numQuadsBlock,
        const unsigned int numQuadsTotal,
        const unsigned int startingQuadId,
				const unsigned int numNodesPerElem,
				const unsigned int numElems,
				const double * gradNQuadValuesXI,
				const double * gradNQuadValuesYI,
				const double * gradNQuadValuesZI,
				const double * gradNQuadValuesXJ,
				const double * gradNQuadValuesYJ,
				const double * gradNQuadValuesZJ,
				const double * jxwQuadValues,
				double * shapeGradNINJIntegralContribution)
		{

			const unsigned int globalThreadId = blockIdx.x*blockDim.x + threadIdx.x;
			const unsigned int numberEntries = numElems*numNodesPerElem*numNodesPerElem*numQuadsBlock;

			for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
			{
				const unsigned int blockIndex1 = index/numQuadsBlock;
				const unsigned int quadIndex=index-blockIndex1*numQuadsBlock;
				const unsigned int blockIndex2=blockIndex1/numNodesPerElem;
				const unsigned int cellId=blockIndex2/numNodesPerElem;
				const unsigned int idJ=cellId*numNodesPerElem*numQuadsTotal+(blockIndex1-blockIndex2*numNodesPerElem)*numQuadsTotal+quadIndex+startingQuadId;
				const unsigned int idI=cellId*numNodesPerElem*numQuadsTotal+(blockIndex2-cellId*numNodesPerElem)*numQuadsTotal+quadIndex+startingQuadId;


				shapeGradNINJIntegralContribution[index]=(gradNQuadValuesXI[idI]*gradNQuadValuesXJ[idJ]+gradNQuadValuesYI[idI]*gradNQuadValuesYJ[idJ]+gradNQuadValuesZI[idI]*gradNQuadValuesZJ[idJ])*jxwQuadValues[cellId*numQuadsTotal+quadIndex+startingQuadId];
			}

		}

	void computeShapeGradNINJIntegral(cublasHandle_t &handle,
      FEValues<3> & fe_values,
      const dealii::DoFHandler<3> & dofHandler,
      const unsigned int numElems,
			thrust::device_vector<double> & shapeGradNINJIntegralD)
	{
    const unsigned int numQuads=fe_values.get_quadrature().size();
    const unsigned int numNodesPerElem = fe_values.get_fe().dofs_per_cell;

		shapeGradNINJIntegralD.clear();
		shapeGradNINJIntegralD.resize(numElems*numNodesPerElem*numNodesPerElem,0.0);

		const int blockSizeElems=1;
    const int blockSizeQuads=100;
		const int numberElemBlocks=numElems/blockSizeElems;
		const int remBlockSizeElems=numElems-numberElemBlocks*blockSizeElems;

		const int numberQuadsBlocks=numQuads/blockSizeQuads;
		const int remBlockSizeQuads=numQuads-numberQuadsBlocks*blockSizeQuads;    

		thrust::device_vector<double> shapeGradNINJIntegralContributionD(blockSizeElems*numNodesPerElem*numNodesPerElem*blockSizeQuads,0.0);
		thrust::device_vector<double> onesVecD(blockSizeQuads,1.0);

    std::vector<double> cellJxWValues(blockSizeElems*numQuads);
    std::vector<double> shapeFunctionGradientValuesX(blockSizeElems*numQuads*numNodesPerElem,0.0);
    std::vector<double> shapeFunctionGradientValuesY(blockSizeElems*numQuads*numNodesPerElem,0.0);
    std::vector<double> shapeFunctionGradientValuesZ(blockSizeElems*numQuads*numNodesPerElem,0.0);

    thrust::device_vector<double> jxwQuadValuesD;
    thrust::device_vector<double> gradNQuadValuesXD;
    thrust::device_vector<double> gradNQuadValuesYD;
    thrust::device_vector<double> gradNQuadValuesZD;

    for (int iblock=0; iblock<(numberElemBlocks+1); iblock++)
    {
      const int currentElemsBlockSize= (iblock==numberElemBlocks)?remBlockSizeElems:blockSizeElems;
      if (currentElemsBlockSize>0)
      {     
        const int startingElemId=iblock*blockSizeElems;       

	      typename dealii::DoFHandler<3>::active_cell_iterator cellPtr=dofHandler.begin_active();
	      typename dealii::DoFHandler<3>::active_cell_iterator endcPtr = dofHandler.end();

        unsigned int iElem=0;
        for(; cellPtr!=endcPtr; ++cellPtr)
          if(cellPtr->is_locally_owned())
          {
            if (iElem>=startingElemId && iElem<(startingElemId+currentElemsBlockSize))
            {
              const unsigned int intraBlockElemId=iElem-startingElemId;
              fe_values.reinit (cellPtr);

              for(unsigned int q_point = 0; q_point < numQuads; ++q_point)
                cellJxWValues[intraBlockElemId*numQuads+q_point]=fe_values.JxW(q_point); 

              for(unsigned int iNode = 0; iNode < numNodesPerElem; ++iNode)
                for(unsigned int q_point = 0; q_point < numQuads; ++q_point)
                {
                  const dealii::Tensor<1,3,double> & shape_grad=fe_values.shape_grad(iNode,q_point);

                  shapeFunctionGradientValuesX[intraBlockElemId*numNodesPerElem*numQuads
                    +iNode*numQuads+q_point]=shape_grad[0];

                  shapeFunctionGradientValuesY[intraBlockElemId*numNodesPerElem*numQuads
                    +iNode*numQuads+q_point]=shape_grad[1];

                  shapeFunctionGradientValuesZ[intraBlockElemId*numNodesPerElem*numQuads
                    +iNode*numQuads+q_point]=shape_grad[2];
                }
            }

            iElem++;
          }

	      jxwQuadValuesD=cellJxWValues;
	      gradNQuadValuesXD=shapeFunctionGradientValuesX;
	      gradNQuadValuesYD=shapeFunctionGradientValuesY;
	      gradNQuadValuesZD=shapeFunctionGradientValuesZ;

        for (int jblock=0; jblock<(numberQuadsBlocks+1); jblock++)
        {
          const int currentQuadsBlockSize= (jblock==numberQuadsBlocks)?remBlockSizeQuads:blockSizeQuads;          
          const int startingQuadId=jblock*blockSizeQuads;           
          if (currentQuadsBlockSize>0)
          {             
            computeShapeGradNINJIntegralContribution<<<(currentQuadsBlockSize+255)/256*numNodesPerElem*numNodesPerElem*currentElemsBlockSize,256>>>
              (currentQuadsBlockSize,
               numQuads,
               startingQuadId,
               numNodesPerElem,
               currentElemsBlockSize,
               thrust::raw_pointer_cast(&gradNQuadValuesXD[0]),
               thrust::raw_pointer_cast(&gradNQuadValuesYD[0]),
               thrust::raw_pointer_cast(&gradNQuadValuesZD[0]),
               thrust::raw_pointer_cast(&gradNQuadValuesXD[0]),
               thrust::raw_pointer_cast(&gradNQuadValuesYD[0]),
               thrust::raw_pointer_cast(&gradNQuadValuesZD[0]),
               thrust::raw_pointer_cast(&jxwQuadValuesD[0]),
               thrust::raw_pointer_cast(&shapeGradNINJIntegralContributionD[0]));

            const double scalarCoeffAlpha = 1.0;
            const double scalarCoeffBeta = 1.0;



            cublasDgemm(handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                1,
                currentElemsBlockSize*numNodesPerElem*numNodesPerElem,
                currentQuadsBlockSize,
                &scalarCoeffAlpha,
                thrust::raw_pointer_cast(&onesVecD[0]),
                1,
                thrust::raw_pointer_cast(&shapeGradNINJIntegralContributionD[0]),
                currentQuadsBlockSize,
                &scalarCoeffBeta,
                thrust::raw_pointer_cast(&shapeGradNINJIntegralD[startingElemId*numNodesPerElem*numNodesPerElem]),
                1);
          }
        }//block loop over nodes per elem
      }
    }//block loop over elems
	} 
}

	template<unsigned int FEOrder,unsigned int FEOrderElectro>
void kohnShamDFTOperatorCUDAClass<FEOrder,FEOrderElectro>::preComputeShapeFunctionGradientIntegrals(const unsigned int lpspQuadratureId)
{

	//
	//get FE data
	//
	const unsigned int numberPhysicalCells = dftPtr->matrix_free_data.n_physical_cells();
	const Quadrature<3> &  quadrature=dftPtr->matrix_free_data.get_quadrature(dftPtr->d_densityQuadratureId);
	FEValues<3> fe_values(dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex).get_fe(), quadrature, update_values | update_gradients | update_JxW_values);
	const unsigned int numberDofsPerElement = dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex).get_fe().dofs_per_cell;
  const unsigned int numberDofsPerElementElectro = dftPtr->d_matrixFreeDataPRefined.get_dof_handler(dftPtr->d_baseDofHandlerIndexElectro).get_fe().dofs_per_cell;
	const unsigned int numberQuadraturePoints = quadrature.size();

	FEValues<3> fe_values_lpsp(dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex).get_fe(), dftPtr->matrix_free_data.get_quadrature(lpspQuadratureId), update_values);
	const unsigned int numberQuadraturePointsLpsp = dftPtr->matrix_free_data.get_quadrature(lpspQuadratureId).size();
  d_numQuadPointsLpsp=numberQuadraturePointsLpsp;

	//
	//resize data members
	//
	//d_cellShapeFunctionGradientIntegralFlattened.clear();
	//d_cellShapeFunctionGradientIntegralFlattened.resize(numberPhysicalCells*numberDofsPerElement*numberDofsPerElement);

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

  std::vector<double> shapeFunctionValueLpsp(numberQuadraturePointsLpsp*numberDofsPerElement,0.0);
  std::vector<double> shapeFunctionValueInvertedLpsp(numberQuadraturePointsLpsp*numberDofsPerElement,0.0);



	typename dealii::DoFHandler<3>::active_cell_iterator cellPtr=dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex).begin_active();
	typename dealii::DoFHandler<3>::active_cell_iterator endcPtr = dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex).end();

	unsigned int iElem=0;
	for(; cellPtr!=endcPtr; ++cellPtr)
		if(cellPtr->is_locally_owned())
		{
			fe_values.reinit (cellPtr);

			for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
				d_cellJxWValues[iElem*numberQuadraturePoints+q_point]=fe_values.JxW(q_point); 

			for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
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

			if(iElem == 0)
      {
        fe_values_lpsp.reinit(cellPtr);

				for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
					for(unsigned int q_point = 0; q_point < numberQuadraturePoints; ++q_point)
					{
						const double val=fe_values.shape_value(iNode,q_point);
						d_shapeFunctionValue[numberQuadraturePoints*iNode + q_point] = val;
						d_shapeFunctionValueInverted[q_point*numberDofsPerElement+iNode] = val;
					}

				for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
					for(unsigned int q_point = 0; q_point < numberQuadraturePointsLpsp; ++q_point)
          {
						const double val=fe_values_lpsp.shape_value(iNode,q_point);            
						shapeFunctionValueLpsp[numberQuadraturePointsLpsp*iNode + q_point] = val; 
            shapeFunctionValueInvertedLpsp[q_point*numberDofsPerElement+iNode] = val;  
          }
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

	d_shapeFunctionValueLpspDevice=shapeFunctionValueLpsp;
	d_shapeFunctionValueInvertedLpspDevice=shapeFunctionValueInvertedLpsp;  

	//d_cellShapeFunctionGradientIntegralFlattenedDevice=d_cellShapeFunctionGradientIntegralFlattened;
	d_cellJxWValuesDevice=d_cellJxWValues;

	cudaDeviceSynchronize();
	MPI_Barrier(MPI_COMM_WORLD);
	double gpu_time=MPI_Wtime();

	QGauss<3>  quadraturePlusOne(FEOrder+1);
  unsigned int numberQuadraturePointsPlusOne = quadraturePlusOne.size();  
	FEValues<3> fe_values_plusone(dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex).get_fe(), quadraturePlusOne, update_gradients | update_JxW_values);


	shapeFuncCUDA::computeShapeGradNINJIntegral(d_cublasHandle,
      fe_values_plusone,
      dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex),
      numberPhysicalCells,
			d_cellShapeFunctionGradientIntegralFlattenedDevice);

	cudaDeviceSynchronize();
	MPI_Barrier(MPI_COMM_WORLD);
	gpu_time = MPI_Wtime() - gpu_time;

	if (this_mpi_process==0 && dftParameters::verbosity>=2)
		std::cout<<"Time for shapeFuncCUDA::computeShapeGradNINJIntegral for FEOrder: "<<gpu_time<<std::endl;

  if (FEOrderElectro!=FEOrder)
  {
    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    gpu_time=MPI_Wtime();

    QGauss<3>  quadratureElectroPlusOne(FEOrderElectro+1);
    numberQuadraturePointsPlusOne = quadratureElectroPlusOne.size();  
    FEValues<3> fe_values_electro_plusone(dftPtr->d_matrixFreeDataPRefined.get_dof_handler(dftPtr->d_baseDofHandlerIndexElectro).get_fe(), quadratureElectroPlusOne, update_gradients | update_JxW_values);

    shapeFuncCUDA::computeShapeGradNINJIntegral(d_cublasHandle,
        fe_values_electro_plusone,
        dftPtr->d_matrixFreeDataPRefined.get_dof_handler(dftPtr->d_baseDofHandlerIndexElectro),
        numberPhysicalCells, 
        d_cellShapeFunctionGradientIntegralFlattenedDeviceElectro);

    cudaDeviceSynchronize();
    MPI_Barrier(MPI_COMM_WORLD);
    gpu_time = MPI_Wtime() - gpu_time;

    if (this_mpi_process==0 && dftParameters::verbosity>=2)
      std::cout<<"Time for shapeFuncCUDA::computeShapeGradNINJIntegral for FEOrderElectro: "<<gpu_time<<std::endl;
  } 

  QGaussLobatto<3>  quadratureGl(C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()+1);
  FEValues<3> fe_valuesGl(dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex).get_fe(), quadratureGl, update_values | update_gradients);
  const unsigned int numberQuadraturePointsGl = quadratureGl.size();

  //
  //resize data members
  //
  std::vector<double> glShapeFunctionValueInverted(numberQuadraturePointsGl*numberDofsPerElement,0.0);

  std::vector<double> glShapeFunctionGradientValueXInverted(numberPhysicalCells*numberQuadraturePointsGl*numberDofsPerElement,0.0);

  std::vector<double> glShapeFunctionGradientValueYInverted(numberPhysicalCells*numberQuadraturePointsGl*numberDofsPerElement,0.0);

  std::vector<double> glShapeFunctionGradientValueZInverted(numberPhysicalCells*numberQuadraturePointsGl*numberDofsPerElement,0.0);


  cellPtr=dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex).begin_active();
  endcPtr = dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex).end();

  iElem=0;
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

  //QGauss<3>  quadratureNLP(C_num1DQuadNLPSP<FEOrder>());
  QIterated<3> quadratureNLP(QGauss<1>(C_num1DQuadNLPSP<FEOrder>()),C_numCopies1DQuadNLPSP());
  FEValues<3> fe_valuesNLP(dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex).get_fe(), quadratureNLP, update_values|update_gradients|update_jacobians|update_inverse_jacobians);
  const unsigned int numberQuadraturePointsNLP = quadratureNLP.size();

  //
  //resize data members
  //
  std::vector<double> nlpShapeFunctionValueInverted(numberQuadraturePointsNLP*numberDofsPerElement,0.0);
  std::vector<double> inverseJacobiansNLP(numberPhysicalCells*numberQuadraturePointsNLP*3*3,0.0);
  //std::vector<double> shapeFunctionGradientValueNLPInverted(numberPhysicalCells*numberQuadraturePointsNLP*3*numberDofsPerElement,0.0);
  std::vector<double> shapeFunctionGradientValueNLPInverted(numberQuadraturePointsNLP*numberDofsPerElement*3,0.0);

  cellPtr=dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex).begin_active();
  endcPtr = dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex).end();


  iElem=0;
  for(; cellPtr!=endcPtr; ++cellPtr)
    if(cellPtr->is_locally_owned())
    { 
      fe_valuesNLP.reinit (cellPtr);

      const std::vector<DerivativeForm<1,3,3> >& inverseJacobians=fe_valuesNLP.get_inverse_jacobians();

      //dealii returns inverse jacobian tensor in transposed format J^{-T}
      for(unsigned int q_point = 0; q_point < numberQuadraturePointsNLP; ++q_point)
        for(unsigned int i = 0; i < 3; ++i)  
          for(unsigned int j = 0; j < 3; ++j)
            inverseJacobiansNLP[iElem*numberQuadraturePointsNLP*3*3+q_point*3*3+j*3+i]=inverseJacobians[q_point][i][j];
      /*
      for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
        for(unsigned int q_point = 0; q_point < numberQuadraturePointsNLP; ++q_point)
        {
          const dealii::Tensor<1,3,double> & shape_grad=fe_valuesNLP.shape_grad(iNode,q_point);
          
          shapeFunctionGradientValueNLPInverted[iElem*numberQuadraturePointsNLP*numberDofsPerElement*3+q_point*numberDofsPerElement*3+iNode]=shape_grad[0];
          shapeFunctionGradientValueNLPInverted[iElem*numberQuadraturePointsNLP*numberDofsPerElement*3+q_point*numberDofsPerElement*3+numberDofsPerElement+iNode]=shape_grad[1];
          shapeFunctionGradientValueNLPInverted[iElem*numberQuadraturePointsNLP*numberDofsPerElement*3+q_point*numberDofsPerElement*3+2*numberDofsPerElement+iNode]=shape_grad[2];
        } 
        */

      if (iElem==0)
      {
        const std::vector<DerivativeForm<1,3,3> >& jacobians=fe_valuesNLP.get_jacobians();
        for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
          for(unsigned int q_point = 0; q_point < numberQuadraturePointsNLP; ++q_point)
          {
            const double val=fe_valuesNLP.shape_value(iNode,q_point);
            nlpShapeFunctionValueInverted[q_point*numberDofsPerElement+iNode] = val;

            const dealii::Tensor<1,3,double> & shape_grad_real=fe_valuesNLP.shape_grad(iNode,q_point);
            
            const dealii::Tensor<1,3,double> & shape_grad_reference= apply_transformation(jacobians[q_point],shape_grad_real);

            shapeFunctionGradientValueNLPInverted[q_point*numberDofsPerElement*3+iNode]=shape_grad_reference[0];
            shapeFunctionGradientValueNLPInverted[q_point*numberDofsPerElement*3+numberDofsPerElement+iNode]=shape_grad_reference[1];
            shapeFunctionGradientValueNLPInverted[q_point*numberDofsPerElement*3+numberDofsPerElement*2+iNode]=shape_grad_reference[2];
          }     
      }

      iElem++;
    }

  d_shapeFunctionValueNLPInvertedDevice=nlpShapeFunctionValueInverted;
  d_shapeFunctionGradientValueNLPInvertedDevice=shapeFunctionGradientValueNLPInverted;
  d_inverseJacobiansNLPDevice=inverseJacobiansNLP;
}
