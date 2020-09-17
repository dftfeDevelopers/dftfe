// ---------------------------------------------------------------------
//
// Copyright (c) 2019-2020 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Phani Motamarri, Sambit Das
//


//
//interpolate nodal data to quadrature values using FEEvaluation
//
	template <unsigned int FEOrder,unsigned int FEOrderElectro>
void dftClass<FEOrder,FEOrderElectro>::interpolateRhoNodalDataToQuadratureDataGeneral(dealii::MatrixFree<3,double> & matrixFreeData,
		const unsigned int dofHandlerId,          
		const unsigned int quadratureId,  
		const distributedCPUVec<double> & nodalField,    
		std::map<dealii::CellId, std::vector<double> > & quadratureValueData,
		std::map<dealii::CellId, std::vector<double> > & quadratureGradValueData,
		std::map<dealii::CellId, std::vector<double> > & quadratureHessianValueData,
		const bool isEvaluateGradData,
		const bool isEvaluateHessianData)
{
	quadratureValueData.clear();
	if(isEvaluateGradData)
		quadratureGradValueData.clear();
	if (isEvaluateHessianData)
		quadratureHessianValueData.clear();

	FEEvaluation<C_DIM,C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>(),C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),1,double> feEvalObj(matrixFreeData,dofHandlerId,quadratureId);
	const unsigned int numQuadPoints = feEvalObj.n_q_points; 

  //AssertThrow(nodalField.partitioners_are_globally_compatible(*matrixFreeData.get_vector_partitioner(dofHandlerId)),
  //        dealii::ExcMessage("DFT-FE Error: mismatch in partitioner/dofHandler."));

  AssertThrow(matrixFreeData.get_quadrature(quadratureId).size() == numQuadPoints,
          dealii::ExcMessage("DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

	DoFHandler<C_DIM>::active_cell_iterator subCellPtr;
	for(unsigned int cell = 0; cell < matrixFreeData.n_macro_cells(); ++cell)
	{
		feEvalObj.reinit(cell);
		feEvalObj.read_dof_values(nodalField);
		feEvalObj.evaluate(true,isEvaluateGradData?true:false,isEvaluateHessianData?true:false);

		for(unsigned int iSubCell = 0; iSubCell < matrixFreeData.n_components_filled(cell); ++iSubCell)
		{
			subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell,dofHandlerId);
			dealii::CellId subCellId=subCellPtr->id();
			quadratureValueData[subCellId] = std::vector<double>(numQuadPoints);

			std::vector<double> & tempVec = quadratureValueData.find(subCellId)->second;

			for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
			{
				tempVec[q_point] = feEvalObj.get_value(q_point)[iSubCell];
			}

			if(isEvaluateGradData)
			{ 
				quadratureGradValueData[subCellId]=std::vector<double>(3*numQuadPoints);

				std::vector<double> & tempVec2 = quadratureGradValueData.find(subCellId)->second;

				for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
				{
					const Tensor< 1, 3, VectorizedArray< double> >  & gradVals=	feEvalObj.get_gradient(q_point);
					tempVec2[3*q_point + 0] = gradVals[0][iSubCell];
					tempVec2[3*q_point + 1] = gradVals[1][iSubCell];
					tempVec2[3*q_point + 2] = gradVals[2][iSubCell];
				}
			}

			if(isEvaluateHessianData)
			{ 
				quadratureHessianValueData[subCellId]=std::vector<double>(9*numQuadPoints);

				std::vector<double> & tempVec3 = quadratureHessianValueData[subCellId];

				for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
				{
					const Tensor< 2, 3, VectorizedArray< double> >   & hessianVals=feEvalObj.get_hessian(q_point);
					for (unsigned int i=0; i<3;i++)
						for (unsigned int j=0; j<3;j++)
							tempVec3[9*q_point + 3*i+j] = hessianVals[i][j][iSubCell];
				}
			}
		}
	}

}


//
//interpolate nodal data to quadrature values using FEEvaluation
//
	template <unsigned int FEOrder,unsigned int FEOrderElectro>
void dftClass<FEOrder,FEOrderElectro>::interpolateElectroNodalDataToQuadratureDataGeneral(dealii::MatrixFree<3,double> & matrixFreeData,
		const unsigned int dofHandlerId,          
		const unsigned int quadratureId,  
		const distributedCPUVec<double> & nodalField,    
		std::map<dealii::CellId, std::vector<double> > & quadratureValueData,
		std::map<dealii::CellId, std::vector<double> > & quadratureGradValueData,
		const bool isEvaluateGradData)
{
	quadratureValueData.clear();
	if(isEvaluateGradData)
		quadratureGradValueData.clear();

	FEEvaluation<C_DIM,FEOrderElectro,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),1,double> feEvalObj(matrixFreeData,dofHandlerId,quadratureId);
	const unsigned int numQuadPoints = feEvalObj.n_q_points; 

  //AssertThrow(nodalField.partitioners_are_globally_compatible(*matrixFreeData.get_vector_partitioner(dofHandlerId)),
  //        dealii::ExcMessage("DFT-FE Error: mismatch in partitioner/dofHandler."));

  AssertThrow(matrixFreeData.get_quadrature(quadratureId).size() == numQuadPoints,
          dealii::ExcMessage("DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

	DoFHandler<C_DIM>::active_cell_iterator subCellPtr;
	for(unsigned int cell = 0; cell < matrixFreeData.n_macro_cells(); ++cell)
	{
		feEvalObj.reinit(cell);
		feEvalObj.read_dof_values(nodalField);
		feEvalObj.evaluate(true,isEvaluateGradData?true:false);

		for(unsigned int iSubCell = 0; iSubCell < matrixFreeData.n_components_filled(cell); ++iSubCell)
		{
			subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell,dofHandlerId);
			dealii::CellId subCellId=subCellPtr->id();
			quadratureValueData[subCellId] = std::vector<double>(numQuadPoints);

			std::vector<double> & tempVec = quadratureValueData.find(subCellId)->second;

			for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
			{
				tempVec[q_point] = feEvalObj.get_value(q_point)[iSubCell];
			}

			if(isEvaluateGradData)
			{ 
				quadratureGradValueData[subCellId]=std::vector<double>(3*numQuadPoints);

				std::vector<double> & tempVec2 = quadratureGradValueData.find(subCellId)->second;

				for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
				{
					const Tensor< 1, 3, VectorizedArray< double> >  & gradVals=	feEvalObj.get_gradient(q_point);
					tempVec2[3*q_point + 0] = gradVals[0][iSubCell];
					tempVec2[3*q_point + 1] = gradVals[1][iSubCell];
					tempVec2[3*q_point + 2] = gradVals[2][iSubCell];
				}
			}
		}
	}

}


	template <unsigned int FEOrder,unsigned int FEOrderElectro>
void dftClass<FEOrder,FEOrderElectro>::interpolateRhoNodalDataToQuadratureDataLpsp(dealii::MatrixFree<3,double> & matrixFreeData,
		const unsigned int dofHandlerId,
		const unsigned int quadratureId,
		const distributedCPUVec<double> & nodalField,
		std::map<dealii::CellId, std::vector<double> > & quadratureValueData,
		std::map<dealii::CellId, std::vector<double> > & quadratureGradValueData,
		const bool isEvaluateGradData)
{

	quadratureValueData.clear();
	quadratureGradValueData.clear();
  FEEvaluation<C_DIM,C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>(),C_num1DQuadLPSP<FEOrder>()*C_numCopies1DQuadLPSP(),1,double> feEvalObj(matrixFreeData,dofHandlerId,quadratureId);
	const unsigned int numQuadPoints = feEvalObj.n_q_points;

  //AssertThrow(nodalField.partitioners_are_globally_compatible(*matrixFreeData.get_vector_partitioner(dofHandlerId)),
  //        dealii::ExcMessage("DFT-FE Error: mismatch in partitioner/dofHandler."));

  AssertThrow(matrixFreeData.get_quadrature(quadratureId).size() == numQuadPoints,
          dealii::ExcMessage("DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

	DoFHandler<C_DIM>::active_cell_iterator subCellPtr;
	for(unsigned int cell = 0; cell < matrixFreeData.n_macro_cells(); ++cell)
	{
		feEvalObj.reinit(cell);
		feEvalObj.read_dof_values(nodalField);
		feEvalObj.evaluate(true,true);
		for(unsigned int iSubCell = 0; iSubCell < matrixFreeData.n_components_filled(cell); ++iSubCell)
		{
			subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell,dofHandlerId);
			dealii::CellId subCellId=subCellPtr->id();
			quadratureValueData[subCellId] = std::vector<double>(numQuadPoints);
			std::vector<double> & tempVec = quadratureValueData.find(subCellId)->second;
			for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
			{
				tempVec[q_point] = feEvalObj.get_value(q_point)[iSubCell];
			}
		}

		if(isEvaluateGradData)
		{
			for(unsigned int iSubCell = 0; iSubCell < matrixFreeData.n_components_filled(cell); ++iSubCell)
			{
				subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell,dofHandlerId);
				dealii::CellId subCellId=subCellPtr->id();
				quadratureGradValueData[subCellId]=std::vector<double>(3*numQuadPoints);
				std::vector<double> & tempVec = quadratureGradValueData.find(subCellId)->second;
				for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
				{
					tempVec[3*q_point + 0] = feEvalObj.get_gradient(q_point)[0][iSubCell];
					tempVec[3*q_point + 1] = feEvalObj.get_gradient(q_point)[1][iSubCell];
					tempVec[3*q_point + 2] = feEvalObj.get_gradient(q_point)[2][iSubCell];
				}
			}
		}

	}

}




	template <unsigned int FEOrder,unsigned int FEOrderElectro>
void dftClass<FEOrder,FEOrderElectro>::interpolateFieldsFromPrevToCurrentMesh(std::vector<distributedCPUVec<double>*> fieldsPrevious,
		std::vector<distributedCPUVec<double>* > fieldsCurrent,
		const dealii::FESystem<3> & FEPrev,
		const dealii::FESystem<3> & FECurrent,
		const dealii::ConstraintMatrix & constraintsCurrent)

{
	vectorTools::interpolateFieldsFromPreviousMesh interpolateFromPrev(mpi_communicator);
	interpolateFromPrev.interpolate(d_mesh.getSerialMeshUnmovedPrevious(),
			d_mesh.getParallelMeshUnmovedPrevious(),
			d_mesh.getParallelMeshUnmoved(),
			FEPrev,
			FECurrent,
			fieldsPrevious,
			fieldsCurrent,
			&constraintsCurrent);

	for (unsigned int i=0; i<fieldsCurrent.size();++i)
		fieldsCurrent[i]->update_ghost_values();
}

//
//compute field l2 norm
//
	template <unsigned int FEOrder,unsigned int FEOrderElectro>
double dftClass<FEOrder,FEOrderElectro>::fieldGradl2Norm(const dealii::MatrixFree<3,double> & matrixFreeDataObject,
		const distributedCPUVec<double> & nodalField)

{
	FEEvaluation<C_DIM,C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>(),C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),1,double> fe_evalField(matrixFreeDataObject,0,0);
	const unsigned int numQuadPoints = fe_evalField.n_q_points;

  //AssertThrow(nodalField.partitioners_are_globally_compatible(*matrixFreeDataObject.get_vector_partitioner(0)),
  //        dealii::ExcMessage("DFT-FE Error: mismatch in partitioner/dofHandler."));

  AssertThrow(matrixFreeDataObject.get_quadrature(0).size() == numQuadPoints,
          dealii::ExcMessage("DFT-FE Error: mismatch in quadrature rule usage in interpolateNodalDataToQuadratureData."));

	VectorizedArray<double> valueVectorized = make_vectorized_array(0.0);

	for(unsigned int cell = 0; cell < matrixFreeDataObject.n_macro_cells(); ++cell)
	{
		fe_evalField.reinit(cell);
		fe_evalField.read_dof_values(nodalField);
		fe_evalField.evaluate(false,true);
		for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
		{
			VectorizedArray<double> temp = scalar_product(fe_evalField.get_gradient(q_point),fe_evalField.get_gradient(q_point));
			fe_evalField.submit_value(temp,q_point);
		}

		valueVectorized += fe_evalField.integrate_value();
	}

	double value = 0.0;
	for(unsigned int iSubCell = 0; iSubCell < VectorizedArray<double>::n_array_elements; ++iSubCell)
	{
		value += valueVectorized[iSubCell];
	}

	return Utilities::MPI::sum(value, mpi_communicator);

}
