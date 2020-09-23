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
		void computeShapeGradNINJIntegralContribution(const unsigned int numQuads,
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
			const unsigned int numberEntries = numElems*numNodesPerElem*numNodesPerElem*numQuads;

			for(unsigned int index = globalThreadId; index < numberEntries; index+= blockDim.x*gridDim.x)
			{
				const unsigned int blockIndex1 = index/numQuads;
				const unsigned int quadIndex=index-blockIndex1*numQuads;
				const unsigned int blockIndex2=blockIndex1/numNodesPerElem;
				const unsigned int cellId=blockIndex2/numNodesPerElem;
				const unsigned int idJ=cellId*numNodesPerElem*numQuads+(blockIndex1-blockIndex2*numNodesPerElem)*numQuads+quadIndex;
				const unsigned int idI=cellId*numNodesPerElem*numQuads+(blockIndex2-cellId*numNodesPerElem)*numQuads+quadIndex;


				shapeGradNINJIntegralContribution[index]=(gradNQuadValuesXI[idI]*gradNQuadValuesXJ[idJ]+gradNQuadValuesYI[idI]*gradNQuadValuesYJ[idJ]+gradNQuadValuesZI[idI]*gradNQuadValuesZJ[idJ])*jxwQuadValues[cellId*numQuads+quadIndex];
			}

		}

	void computeShapeGradNINJIntegral(cublasHandle_t &handle,
			const unsigned int numQuads,
			const unsigned int numNodesPerElem,
			const unsigned int numElems,
			const thrust::device_vector<double> & gradNQuadValuesXD,
			const thrust::device_vector<double> & gradNQuadValuesYD,
			const thrust::device_vector<double> & gradNQuadValuesZD,
			const thrust::device_vector<double> & jxwQuadValuesD,
			thrust::device_vector<double> & shapeGradNINJIntegralD)
	{
		shapeGradNINJIntegralD.clear();
		shapeGradNINJIntegralD.resize(numElems*numNodesPerElem*numNodesPerElem,0.0);
		//thrust::device_vector<double> gradPsiQuadValuesXDJ=gradPsiQuadValuesXD;
		//thrust::device_vector<double> gradPsiQuadValuesYDJ=gradPsiQuadValuesYD;
		//thrust::device_vector<double> gradPsiQuadValuesZDJ=gradPsiQuadValuesZD;

		const int blockSize=1;
		const int numberBlocks=numElems/blockSize;
		const int remBlockSize=numElems-numberBlocks*blockSize;

		thrust::device_vector<double> shapeGradNINJIntegralContributionD(blockSize*numNodesPerElem*numNodesPerElem*numQuads,0.0);
		thrust::device_vector<double> onesVecD(numQuads,1.0); 
		for (int iblock=0; iblock<(numberBlocks+1); iblock++)
		{
			const int currentBlockSize= (iblock==numberBlocks)?remBlockSize:blockSize;
			if (currentBlockSize>0)
			{
				const int startingId=iblock*blockSize;

				computeShapeGradNINJIntegralContribution<<<(numQuads+255)/256*numNodesPerElem*numNodesPerElem*currentBlockSize,256>>>
					(numQuads,
					 numNodesPerElem,
					 currentBlockSize,
					 thrust::raw_pointer_cast(&gradNQuadValuesXD[startingId*numNodesPerElem*numQuads]),
					 thrust::raw_pointer_cast(&gradNQuadValuesYD[startingId*numNodesPerElem*numQuads]),
					 thrust::raw_pointer_cast(&gradNQuadValuesZD[startingId*numNodesPerElem*numQuads]),
					 thrust::raw_pointer_cast(&gradNQuadValuesXD[startingId*numNodesPerElem*numQuads]),
					 thrust::raw_pointer_cast(&gradNQuadValuesYD[startingId*numNodesPerElem*numQuads]),
					 thrust::raw_pointer_cast(&gradNQuadValuesZD[startingId*numNodesPerElem*numQuads]),
					 thrust::raw_pointer_cast(&jxwQuadValuesD[startingId*numQuads]),
					 thrust::raw_pointer_cast(&shapeGradNINJIntegralContributionD[0]));

				const double scalarCoeffAlpha = 1.0;
				const double scalarCoeffBeta = 0.0;



				cublasDgemm(handle,
						CUBLAS_OP_N,
						CUBLAS_OP_N,
						1,
						currentBlockSize*numNodesPerElem*numNodesPerElem,
						numQuads,
						&scalarCoeffAlpha,
						thrust::raw_pointer_cast(&onesVecD[0]),
						1,
						thrust::raw_pointer_cast(&shapeGradNINJIntegralContributionD[0]),
						numQuads,
						&scalarCoeffBeta,
						thrust::raw_pointer_cast(&shapeGradNINJIntegralD[startingId*numNodesPerElem*numNodesPerElem]),
						1);
			}
		}
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


	std::vector<double> cellJxWValuesQuadPlusOne(numberPhysicalCells*numberQuadraturePointsPlusOne);
	std::vector<double> shapeFunctionGradientValueQuadPlusOneX(numberPhysicalCells*numberQuadraturePointsPlusOne*numberDofsPerElement,0.0);
	std::vector<double> shapeFunctionGradientValueQuadPlusOneY(numberPhysicalCells*numberQuadraturePointsPlusOne*numberDofsPerElement,0.0);
	std::vector<double> shapeFunctionGradientValueQuadPlusOneZ(numberPhysicalCells*numberQuadraturePointsPlusOne*numberDofsPerElement,0.0);


	cellPtr=dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex).begin_active();

	iElem=0;
	for(; cellPtr!=endcPtr; ++cellPtr)
		if(cellPtr->is_locally_owned())
		{
			fe_values_plusone.reinit (cellPtr);

			for(unsigned int q_point = 0; q_point < numberQuadraturePointsPlusOne; ++q_point)
				cellJxWValuesQuadPlusOne[iElem*numberQuadraturePointsPlusOne+q_point]=fe_values_plusone.JxW(q_point); 

			for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
				for(unsigned int q_point = 0; q_point < numberQuadraturePointsPlusOne; ++q_point)
				{
					const dealii::Tensor<1,3,double> & shape_grad=fe_values_plusone.shape_grad(iNode,q_point);

					shapeFunctionGradientValueQuadPlusOneX[iElem*numberDofsPerElement*numberQuadraturePointsPlusOne
						+iNode*numberQuadraturePointsPlusOne+q_point]=shape_grad[0];

					shapeFunctionGradientValueQuadPlusOneY[iElem*numberDofsPerElement*numberQuadraturePointsPlusOne
						+iNode*numberQuadraturePointsPlusOne+q_point]=shape_grad[1];

					shapeFunctionGradientValueQuadPlusOneZ[iElem*numberDofsPerElement*numberQuadraturePointsPlusOne
						+iNode*numberQuadraturePointsPlusOne+q_point]=shape_grad[2];
				}

			iElem++;
		}

	thrust::device_vector<double> cellJxWValuesQuadPlusOneDevice=cellJxWValuesQuadPlusOne;
	thrust::device_vector<double> shapeFunctionGradientValueQuadPlusOneXDevice=shapeFunctionGradientValueQuadPlusOneX;
	thrust::device_vector<double> shapeFunctionGradientValueQuadPlusOneYDevice=shapeFunctionGradientValueQuadPlusOneY;
	thrust::device_vector<double> shapeFunctionGradientValueQuadPlusOneZDevice=shapeFunctionGradientValueQuadPlusOneZ;

	shapeFuncCUDA::computeShapeGradNINJIntegral(d_cublasHandle,
			numberQuadraturePointsPlusOne,
			numberDofsPerElement,
			numberPhysicalCells,
			shapeFunctionGradientValueQuadPlusOneXDevice,
			shapeFunctionGradientValueQuadPlusOneYDevice,
			shapeFunctionGradientValueQuadPlusOneZDevice,
			cellJxWValuesQuadPlusOneDevice,
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


    cellJxWValuesQuadPlusOne.resize(numberPhysicalCells*numberQuadraturePointsPlusOne);
    shapeFunctionGradientValueQuadPlusOneX.resize(numberPhysicalCells*numberQuadraturePointsPlusOne*numberDofsPerElementElectro,0.0);
    shapeFunctionGradientValueQuadPlusOneY.resize(numberPhysicalCells*numberQuadraturePointsPlusOne*numberDofsPerElementElectro,0.0);
    shapeFunctionGradientValueQuadPlusOneZ.resize(numberPhysicalCells*numberQuadraturePointsPlusOne*numberDofsPerElementElectro,0.0);


    typename dealii::DoFHandler<3>::active_cell_iterator cellPtrElectro=dftPtr->d_matrixFreeDataPRefined.get_dof_handler(dftPtr->d_baseDofHandlerIndexElectro).begin_active();
    typename dealii::DoFHandler<3>::active_cell_iterator endcPtrElectro = dftPtr->d_matrixFreeDataPRefined.get_dof_handler(dftPtr->d_baseDofHandlerIndexElectro).end();

    iElem=0;
    for(; cellPtrElectro!=endcPtrElectro; ++cellPtrElectro)
      if(cellPtrElectro->is_locally_owned())
      {
        fe_values_electro_plusone.reinit (cellPtrElectro);

        for(unsigned int q_point = 0; q_point < numberQuadraturePointsPlusOne; ++q_point)
          cellJxWValuesQuadPlusOne[iElem*numberQuadraturePointsPlusOne+q_point]=fe_values_electro_plusone.JxW(q_point); 

        for(unsigned int iNode = 0; iNode < numberDofsPerElementElectro; ++iNode)
          for(unsigned int q_point = 0; q_point < numberQuadraturePointsPlusOne; ++q_point)
          {
            const dealii::Tensor<1,3,double> & shape_grad=fe_values_electro_plusone.shape_grad(iNode,q_point);

            shapeFunctionGradientValueQuadPlusOneX[iElem*numberDofsPerElementElectro*numberQuadraturePointsPlusOne
              +iNode*numberQuadraturePointsPlusOne+q_point]=shape_grad[0];

            shapeFunctionGradientValueQuadPlusOneY[iElem*numberDofsPerElementElectro*numberQuadraturePointsPlusOne
              +iNode*numberQuadraturePointsPlusOne+q_point]=shape_grad[1];

            shapeFunctionGradientValueQuadPlusOneZ[iElem*numberDofsPerElementElectro*numberQuadraturePointsPlusOne
              +iNode*numberQuadraturePointsPlusOne+q_point]=shape_grad[2];
          }

        iElem++;
      }

    cellJxWValuesQuadPlusOneDevice=cellJxWValuesQuadPlusOne;
    shapeFunctionGradientValueQuadPlusOneXDevice=shapeFunctionGradientValueQuadPlusOneX;
    shapeFunctionGradientValueQuadPlusOneYDevice=shapeFunctionGradientValueQuadPlusOneY;
    shapeFunctionGradientValueQuadPlusOneZDevice=shapeFunctionGradientValueQuadPlusOneZ;

    shapeFuncCUDA::computeShapeGradNINJIntegral(d_cublasHandle,
        numberQuadraturePointsPlusOne,
        numberDofsPerElementElectro,
        numberPhysicalCells,
        shapeFunctionGradientValueQuadPlusOneXDevice,
        shapeFunctionGradientValueQuadPlusOneYDevice,
        shapeFunctionGradientValueQuadPlusOneZDevice,
        cellJxWValuesQuadPlusOneDevice,
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

	if (dftParameters::useHigherQuadNLP)
	{
		//QGauss<3>  quadratureNLP(C_num1DQuadNLPSP<FEOrder>());
    QIterated<3> quadratureNLP(QGauss<1>(C_num1DQuadNLPSP<FEOrder>()),C_numCopies1DQuadNLPSP());
		FEValues<3> fe_valuesNLP(dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex).get_fe(), quadratureNLP, update_values|update_gradients);
		const unsigned int numberQuadraturePointsNLP = quadratureNLP.size();

		//
		//resize data members
		//
		std::vector<double> nlpShapeFunctionValueInverted(numberQuadraturePointsNLP*numberDofsPerElement,0.0);
		std::vector<double> shapeFunctionGradientValueNLPXInverted(numberPhysicalCells*numberQuadraturePointsNLP*numberDofsPerElement,0.0);
		std::vector<double> shapeFunctionGradientValueNLPYInverted(numberPhysicalCells*numberQuadraturePointsNLP*numberDofsPerElement,0.0);
		std::vector<double> shapeFunctionGradientValueNLPZInverted(numberPhysicalCells*numberQuadraturePointsNLP*numberDofsPerElement,0.0);    

		cellPtr=dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex).begin_active();
		endcPtr = dftPtr->matrix_free_data.get_dof_handler(dftPtr->d_densityDofHandlerIndex).end();


    iElem=0;
		for(; cellPtr!=endcPtr; ++cellPtr)
			if(cellPtr->is_locally_owned())
			{ 
				fe_valuesNLP.reinit (cellPtr);

        for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
          for(unsigned int q_point = 0; q_point < numberQuadraturePointsNLP; ++q_point)
          {
            const dealii::Tensor<1,3,double> & shape_grad=fe_valuesNLP.shape_grad(iNode,q_point);

            shapeFunctionGradientValueNLPXInverted[iElem*numberQuadraturePointsNLP*numberDofsPerElement
              +q_point*numberDofsPerElement+iNode]=shape_grad[0];

            shapeFunctionGradientValueNLPYInverted[iElem*numberQuadraturePointsNLP*numberDofsPerElement
              +q_point*numberDofsPerElement+iNode]=shape_grad[1];

            shapeFunctionGradientValueNLPZInverted[iElem*numberQuadraturePointsNLP*numberDofsPerElement
              +q_point*numberDofsPerElement+iNode]=shape_grad[2];

          }

        if (iElem==0)
          for(unsigned int iNode = 0; iNode < numberDofsPerElement; ++iNode)
            for(unsigned int q_point = 0; q_point < numberQuadraturePointsNLP; ++q_point)
            {
              const double val=fe_valuesNLP.shape_value(iNode,q_point);
              nlpShapeFunctionValueInverted[q_point*numberDofsPerElement+iNode] = val;
            }      

        iElem++;
			}

		d_shapeFunctionValueNLPInvertedDevice=nlpShapeFunctionValueInverted;
	  d_shapeFunctionGradientValueNLPXInvertedDevice=shapeFunctionGradientValueNLPXInverted;
    d_shapeFunctionGradientValueNLPYInvertedDevice=shapeFunctionGradientValueNLPYInverted;
    d_shapeFunctionGradientValueNLPZInvertedDevice=shapeFunctionGradientValueNLPZInverted;    
	}
}
