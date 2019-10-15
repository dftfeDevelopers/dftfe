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
// @author Phani Motamarri
//


//
//interpolate nodal data to quadrature values using FEEvaluation
//
template <unsigned int FEOrder>
void dftClass<FEOrder>::interpolateNodalDataToQuadratureData(dealii::MatrixFree<3,double> & matrixFreeData,
							     vectorType & nodalField,
							     std::map<dealii::CellId, std::vector<double> > & quadratureValueData,
							     std::map<dealii::CellId, std::vector<double> > & quadratureGradValueData,
							     const bool isEvaluateGradData)
{
  

  FEEvaluation<C_DIM,C_num1DKerkerPoly<FEOrder>(),C_num1DQuad<FEOrder>(),1,double> feEvalObj(matrixFreeData,0,1);
  const unsigned int numQuadPoints = feEvalObj.n_q_points; 

  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;
  for(unsigned int cell = 0; cell < matrixFreeData.n_macro_cells(); ++cell)
    {
      feEvalObj.reinit(cell);
      feEvalObj.read_dof_values(d_rhoInNodalValues);
      feEvalObj.evaluate(true,true);
      for(unsigned int iSubCell = 0; iSubCell < matrixFreeData.n_components_filled(cell); ++iSubCell)
	{
	  subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell);
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
	      subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell);
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


