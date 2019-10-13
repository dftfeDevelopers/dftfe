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
// @author Shiva Rudraraju, Phani Motamarri, Krishnendu Ghosh, Sambit Das
//

//source file for electron density related computations

//calculate electron density
template<unsigned int FEOrder>
void dftClass<FEOrder>::compute_rhoOut(const bool isConsiderSpectrumSplitting)
{
  
  if(dftParameters::mixingMethod=="ANDERSON_WITH_KERKER")
    {
      computeRhoNodalFromPSI(isConsiderSpectrumSplitting);
      d_rhoOutNodalValues.update_ghost_values();
      d_rhoOutNodalVals.push_back(d_rhoOutNodalValues);

      //assign pointer-type to rhoOutValues and gradRhoOutValues
      //	
     
      //fill in rhoOutValues and gradRhoOutValues
      FEEvaluation<C_DIM,2*FEOrder,C_num1DQuad<FEOrder>(),1,double> rhoEval(d_matrixFreeDataPRefined,0,1);
      const unsigned int numQuadPoints = rhoEval.n_q_points;
      DoFHandler<C_DIM>::active_cell_iterator subCellPtr;
      for(unsigned int cell = 0; cell < d_matrixFreeDataPRefined.n_macro_cells(); ++cell)
	{
	  rhoEval.reinit(cell);
	  rhoEval.read_dof_values(d_rhoOutNodalValues);
	  rhoEval.evaluate(true,true);
	  for(unsigned int iSubCell = 0; iSubCell < d_matrixFreeDataPRefined.n_components_filled(cell); ++iSubCell)
	    {
	      subCellPtr= d_matrixFreeDataPRefined.get_cell_iterator(cell,iSubCell);
	      dealii::CellId subCellId=subCellPtr->id();
	      (*rhoOutValues)[subCellId] = std::vector<double>(numQuadPoints);
	      std::vector<double> & tempVec = rhoOutValues->find(subCellId)->second;
	      for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
		{
		  tempVec[q_point] = rhoEval.get_value(q_point)[iSubCell];
		}
	    }

	  if(dftParameters::xc_id == 4)
	    {
	      for(unsigned int iSubCell = 0; iSubCell < d_matrixFreeDataPRefined.n_components_filled(cell); ++iSubCell)
		{
		  subCellPtr= d_matrixFreeDataPRefined.get_cell_iterator(cell,iSubCell);
		  dealii::CellId subCellId=subCellPtr->id();
		  (*gradRhoOutValues)[subCellId] = std::vector<double>(3*numQuadPoints);
		  std::vector<double> & tempVec = gradRhoOutValues->find(subCellId)->second;
		  for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
		    {
		      tempVec[3*q_point + 0] = rhoEval.get_gradient(q_point)[0][iSubCell];
		      tempVec[3*q_point + 1] = rhoEval.get_gradient(q_point)[1][iSubCell];
		      tempVec[3*q_point + 2] = rhoEval.get_gradient(q_point)[2][iSubCell];
		    }
		}
	    }
	}

      if(dftParameters::verbosity>=3)
	{
	  pcout<<"Total Charge using nodal Rho out: "<< totalCharge(d_matrixFreeDataPRefined,d_rhoOutNodalValues)<<std::endl;
	}

    }
  else
    {

      resizeAndAllocateRhoTableStorage
	(rhoOutVals,
	 gradRhoOutVals,
	 rhoOutValsSpinPolarized,
	 gradRhoOutValsSpinPolarized);

      rhoOutValues = &(rhoOutVals.back());
      if (dftParameters::spinPolarized==1)
	rhoOutValuesSpinPolarized = &(rhoOutValsSpinPolarized.back());

      if(dftParameters::xc_id == 4)
	{
	  gradRhoOutValues = &(gradRhoOutVals.back());
	  if (dftParameters::spinPolarized==1)
	    gradRhoOutValuesSpinPolarized = &(gradRhoOutValsSpinPolarized.back());
	}


      computeRhoFromPSI(rhoOutValues,
			gradRhoOutValues,
			rhoOutValuesSpinPolarized,
			gradRhoOutValuesSpinPolarized,
			dftParameters::xc_id == 4,
			isConsiderSpectrumSplitting);
    }


  if(dftParameters::mixingMethod=="ANDERSON_WITH_KERKER")
    {
      if(d_rhoInNodalVals.size() == dftParameters::mixingHistory)
	{
	  d_rhoInNodalVals.pop_front();
	  d_rhoOutNodalVals.pop_front();
	}
    }
  else
    {
      //pop out rhoInVals and rhoOutVals if their size exceeds mixing history size
      if(rhoInVals.size() == dftParameters::mixingHistory)
	{
	  rhoInVals.pop_front();
	  rhoOutVals.pop_front();

	  if(dftParameters::spinPolarized==1)
	    {
	      rhoInValsSpinPolarized.pop_front();
	      rhoOutValsSpinPolarized.pop_front();
	    }

	  if(dftParameters::xc_id == 4)//GGA
	    {
	      gradRhoInVals.pop_front();
	      gradRhoOutVals.pop_front();
	    }

	  if(dftParameters::spinPolarized==1 && dftParameters::xc_id==4)
	    {
	      gradRhoInValsSpinPolarized.pop_front();
	      gradRhoOutValsSpinPolarized.pop_front();
	    }

	  if (dftParameters::mixingMethod=="BROYDEN")
	    {
	      dFBroyden.pop_front();
	      uBroyden.pop_front();
	      if(dftParameters::xc_id == 4)//GGA
		{
		  graddFBroyden.pop_front();
		  gradUBroyden.pop_front();
		}
	    }
     
	}

    }

}



template<unsigned int FEOrder>
void dftClass<FEOrder>::resizeAndAllocateRhoTableStorage
(std::deque<std::map<dealii::CellId,std::vector<double> >> & rhoVals,
 std::deque<std::map<dealii::CellId,std::vector<double> >> & gradRhoVals,
 std::deque<std::map<dealii::CellId,std::vector<double> >> & rhoValsSpinPolarized,
 std::deque<std::map<dealii::CellId,std::vector<double> >> & gradRhoValsSpinPolarized)
{
  const unsigned int numQuadPoints = matrix_free_data.get_n_q_points(0);;

  //create new rhoValue tables
  rhoVals.push_back(std::map<dealii::CellId,std::vector<double> > ());
  if (dftParameters::spinPolarized==1)
    rhoValsSpinPolarized.push_back(std::map<dealii::CellId,std::vector<double> > ());

  if(dftParameters::xc_id == 4)
    {
      gradRhoVals.push_back(std::map<dealii::CellId, std::vector<double> >());
      if (dftParameters::spinPolarized==1)
	gradRhoValsSpinPolarized.push_back(std::map<dealii::CellId, std::vector<double> >());
    }


  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
      {
	const dealii::CellId cellId=cell->id();
	rhoVals.back()[cellId] = std::vector<double>(numQuadPoints,0.0);
	if(dftParameters::xc_id == 4)
	  gradRhoVals.back()[cellId] = std::vector<double>(3*numQuadPoints,0.0);

	if (dftParameters::spinPolarized==1)
	  {
	    rhoValsSpinPolarized.back()[cellId] = std::vector<double>(2*numQuadPoints,0.0);
	    if(dftParameters::xc_id == 4)
	      gradRhoValsSpinPolarized.back()[cellId]
		= std::vector<double>(6*numQuadPoints,0.0);
	  }
      }
}

template<unsigned int FEOrder>
void dftClass<FEOrder>::sumRhoData(std::map<dealii::CellId, std::vector<double> > * _rhoValues,
				   std::map<dealii::CellId, std::vector<double> > * _gradRhoValues,
				   std::map<dealii::CellId, std::vector<double> > * _rhoValuesSpinPolarized,
				   std::map<dealii::CellId, std::vector<double> > * _gradRhoValuesSpinPolarized,
				   const bool isGradRhoDataPresent,
				   const MPI_Comm &interComm)
{
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();

  //gather density from inter communicator
  if (dealii::Utilities::MPI::n_mpi_processes(interComm)>1)
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned())
	{
	  const dealii::CellId cellId=cell->id();

	  dealii::Utilities::MPI::sum((*_rhoValues)[cellId],
				      interComm,
				      (*_rhoValues)[cellId]);
	  if(isGradRhoDataPresent)
	    dealii::Utilities::MPI::sum((*_gradRhoValues)[cellId],
					interComm,
					(*_gradRhoValues)[cellId]);

	  if (dftParameters::spinPolarized==1)
	    {
	      dealii::Utilities::MPI::sum((*_rhoValuesSpinPolarized)[cellId],
					  interComm,
					  (*_rhoValuesSpinPolarized)[cellId]);
	      if(isGradRhoDataPresent)
		dealii::Utilities::MPI::sum((*_gradRhoValuesSpinPolarized)[cellId],
					    interComm,
					    (*_gradRhoValuesSpinPolarized)[cellId]);
	    }
	}
}

//rho data reinitilization without remeshing. The rho out of last ground state solve is made the rho in of the new solve
template<unsigned int FEOrder>
void dftClass<FEOrder>::noRemeshRhoDataInit()
{

 

  //create temporary copies of rho Out data
  std::map<dealii::CellId, std::vector<double> > rhoOutValuesCopy=*(rhoOutValues);

  std::map<dealii::CellId, std::vector<double> > gradRhoOutValuesCopy;
  if (dftParameters::xc_id==4)
    {
      gradRhoOutValuesCopy=*(gradRhoOutValues);
    }

  std::map<dealii::CellId, std::vector<double> > rhoOutValuesSpinPolarizedCopy;
  if(dftParameters::spinPolarized==1)
    {
      rhoOutValuesSpinPolarizedCopy=*(rhoOutValuesSpinPolarized);

    }

  std::map<dealii::CellId, std::vector<double> > gradRhoOutValuesSpinPolarizedCopy;
  if(dftParameters::spinPolarized==1 && dftParameters::xc_id==4)
    {
      gradRhoOutValuesSpinPolarizedCopy=*(gradRhoOutValuesSpinPolarized);

    }
  //cleanup of existing rho Out and rho In data
  clearRhoData();

  ///copy back temporary rho out to rho in data
  rhoInVals.push_back(rhoOutValuesCopy);
  rhoInValues=&(rhoInVals.back());

  if (dftParameters::xc_id==4)
    {
      gradRhoInVals.push_back(gradRhoOutValuesCopy);
      gradRhoInValues=&(gradRhoInVals.back());
    }

  if(dftParameters::spinPolarized==1)
    {
      rhoInValsSpinPolarized.push_back(rhoOutValuesSpinPolarizedCopy);
      rhoInValuesSpinPolarized=&(rhoInValsSpinPolarized.back());
    }

  if (dftParameters::xc_id==4 && dftParameters::spinPolarized==1)
    {
      gradRhoInValsSpinPolarized.push_back(gradRhoOutValuesSpinPolarizedCopy);
      gradRhoInValuesSpinPolarized=&(gradRhoInValsSpinPolarized.back());
    }

  normalizeRho();

}

template <unsigned int FEOrder>
void dftClass<FEOrder>::computeRhoNodalFromPSI(bool isConsiderSpectrumSplitting)
{
  std::map<dealii::CellId, std::vector<double> >  rhoPRefinedNodalData;

  //initialize variables to be used later
  const unsigned int dofs_per_cell = d_dofHandlerPRefined.get_fe().dofs_per_cell;
  typename DoFHandler<3>::active_cell_iterator cell = d_dofHandlerPRefined.begin_active(), endc = d_dofHandlerPRefined.end();
  dealii::IndexSet locallyOwnedDofs = d_dofHandlerPRefined.locally_owned_dofs();
  QGaussLobatto<3>  quadrature_formula(2*FEOrder+1);
  const unsigned int numQuadPoints = quadrature_formula.size();

  AssertThrow(numQuadPoints == matrix_free_data.get_n_q_points(3),ExcMessage("Number of quadrature points from Quadrature object does not match with number of quadrature points obtained from matrix_free object"));

  //get access to quadrature point coordinates and 2p DoFHandler nodal points
  const std::vector<Point<3> > & quadraturePointCoor = quadrature_formula.get_points();
  const std::vector<Point<3> > & supportPointNaturalCoor = d_dofHandlerPRefined.get_fe().get_unit_support_points();
  std::vector<unsigned int> renumberingMap(numQuadPoints);
  
  //create renumbering map between the numbering order of quadrature points and lobatto support points
  for(unsigned int i = 0; i < numQuadPoints; ++i)
    {
      const Point<3> & nodalCoor = supportPointNaturalCoor[i];
      for(unsigned int j = 0; j < numQuadPoints; ++j)
	{
	  const Point<3> & quadCoor = quadraturePointCoor[j];
	  double dist = quadCoor.distance(nodalCoor);
	  if(dist <= 1e-08)
	    {
	      renumberingMap[i] = j;
	      break;
	    }
	}
    }

  //allocate the storage to compute 2p nodal values from wavefunctions
  for(; cell!=endc; ++cell)
    {
      if (cell->is_locally_owned())
	{
	  const dealii::CellId cellId=cell->id();
	  rhoPRefinedNodalData[cellId] = std::vector<double>(numQuadPoints,0.0);
	}
    }

  //compute rho from wavefunctions at nodal locations of 2p DoFHandler nodes in each cell
  computeRhoFromPSI(&rhoPRefinedNodalData,
		    gradRhoOutValues,
		    rhoOutValuesSpinPolarized,
		    gradRhoOutValuesSpinPolarized,
		    false,
		    isConsiderSpectrumSplitting,
		    true);

  //copy Lobatto quadrature data to fill in 2p DoFHandler nodal data  
   DoFHandler<3>::active_cell_iterator
    cellP = d_dofHandlerPRefined.begin_active(),
    endcP = d_dofHandlerPRefined.end();

   for (; cellP!=endcP; ++cellP)
     {
       if (cellP->is_locally_owned())
	 {
	   std::vector<dealii::types::global_dof_index> cell_dof_indices(dofs_per_cell);
	   cellP->get_dof_indices(cell_dof_indices);
	   const std::vector<double> & nodalValues = rhoPRefinedNodalData.find(cellP->id())->second;
	   AssertThrow(nodalValues.size() == dofs_per_cell,ExcMessage("Number of nodes in 2p DoFHandler does not match with data stored in rhoNodal Values variable"));
          
	   for(unsigned int iNode = 0; iNode < dofs_per_cell; ++iNode)
	     {
	       const dealii::types::global_dof_index nodeID = cell_dof_indices[iNode];
	       if(!d_constraintsPRefined.is_constrained(nodeID))
		 {
		   if(locallyOwnedDofs.is_element(nodeID))
		     d_rhoOutNodalValues(nodeID) =  nodalValues[renumberingMap[iNode]];
		 }
	     }
	 }
     }


}


template <unsigned int FEOrder>
void dftClass<FEOrder>::computeRhoFromPSI(std::map<dealii::CellId, std::vector<double> > * _rhoValues,
					  std::map<dealii::CellId, std::vector<double> > * _gradRhoValues,
					  std::map<dealii::CellId, std::vector<double> > * _rhoValuesSpinPolarized,
					  std::map<dealii::CellId, std::vector<double> > * _gradRhoValuesSpinPolarized,
					  const bool isEvaluateGradRho,
					  const bool isConsiderSpectrumSplitting,
					  const bool lobattoNodesFlag)
{

  const unsigned int numEigenVectorsTotal=d_numEigenValues;
  const unsigned int numEigenVectorsFrac=d_numEigenValuesRR;
  const unsigned int numEigenVectorsCore=d_numEigenValues-d_numEigenValuesRR;
  const unsigned int numKPoints=d_kPointWeights.size();

#ifdef USE_COMPLEX
  FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>(),2> psiEval(matrix_free_data,eigenDofHandlerIndex,0);
  FEEvaluation<3,FEOrder,2*FEOrder+1,2> psiEvalRefined(matrix_free_data,eigenDofHandlerIndex,3);
#else
  FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>(),1> psiEval(matrix_free_data,eigenDofHandlerIndex,0);
  FEEvaluation<3,FEOrder,2*FEOrder+1,1> psiEvalRefined(matrix_free_data,eigenDofHandlerIndex,3);
#endif

  const unsigned int numQuadPoints= lobattoNodesFlag?psiEvalRefined.n_q_points:psiEval.n_q_points;

  //initialization to zero
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  if(!lobattoNodesFlag)
  {
  for (; cell!=endc; ++cell)
    if (cell->is_locally_owned())
      {
	const dealii::CellId cellId=cell->id();
	(*_rhoValues)[cellId] = std::vector<double>(numQuadPoints,0.0);
	if(dftParameters::xc_id == 4)
	  (*_gradRhoValues)[cellId] = std::vector<double>(3*numQuadPoints,0.0);

	if (dftParameters::spinPolarized==1)
	  {
	    (*_rhoValuesSpinPolarized)[cellId] = std::vector<double>(2*numQuadPoints,0.0);
	    if(dftParameters::xc_id == 4)
	      (*_gradRhoValuesSpinPolarized)[cellId]
		= std::vector<double>(6*numQuadPoints,0.0);
	  }
      }
   } 

  Tensor<1,2,VectorizedArray<double> > zeroTensor1;
  zeroTensor1[0]=make_vectorized_array(0.0);
  zeroTensor1[1]=make_vectorized_array(0.0);
  Tensor<1,2, Tensor<1,3,VectorizedArray<double> > > zeroTensor2;
  Tensor<1,3,VectorizedArray<double> > zeroTensor3;
  for (unsigned int idim=0; idim<3; idim++)
    {
      zeroTensor2[0][idim]=make_vectorized_array(0.0);
      zeroTensor2[1][idim]=make_vectorized_array(0.0);
      zeroTensor3[idim]=make_vectorized_array(0.0);
    }

  //temp arrays
  std::vector<double> rhoTemp(numQuadPoints), rhoTempSpinPolarized(2*numQuadPoints), rho(numQuadPoints), rhoSpinPolarized(2*numQuadPoints);
  std::vector<double> gradRhoTemp(3*numQuadPoints), gradRhoTempSpinPolarized(6*numQuadPoints),gradRho(3*numQuadPoints), gradRhoSpinPolarized(6*numQuadPoints);


  //band group parallelization data structures
  const unsigned int numberBandGroups=
    dealii::Utilities::MPI::n_mpi_processes(interBandGroupComm);
  const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(interBandGroupComm);
  std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
  dftUtils::createBandParallelizationIndices(interBandGroupComm,
					     numEigenVectorsTotal,
					     bandGroupLowHighPlusOneIndices);

  const unsigned int eigenVectorsBlockSize=std::min(dftParameters::wfcBlockSize,
						    bandGroupLowHighPlusOneIndices[1]);

  const unsigned int localVectorSize = d_eigenVectorsFlattenedSTL[0].size()/numEigenVectorsTotal;

  std::vector<std::vector<vectorType>> eigenVectors((1+dftParameters::spinPolarized)*d_kPointWeights.size());
  std::vector<dealii::parallel::distributed::Vector<dataTypes::number> > eigenVectorsFlattenedBlock((1+dftParameters::spinPolarized)*d_kPointWeights.size());


  std::vector<std::vector<vectorType>> eigenVectorsRotFrac((1+dftParameters::spinPolarized)*d_kPointWeights.size());
  std::vector<dealii::parallel::distributed::Vector<dataTypes::number> > eigenVectorsRotFracFlattenedBlock((1+dftParameters::spinPolarized)*d_kPointWeights.size());

  for(unsigned int ivec = 0; ivec < numEigenVectorsTotal; ivec+=eigenVectorsBlockSize)
    {
      const unsigned int currentBlockSize=std::min(eigenVectorsBlockSize,numEigenVectorsTotal-ivec);

      if (currentBlockSize!=eigenVectorsBlockSize || ivec==0)
	{
	  for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
	    {
	      eigenVectors[kPoint].resize(currentBlockSize);
	      for(unsigned int i= 0; i < currentBlockSize; ++i)
		eigenVectors[kPoint][i].reinit(d_tempEigenVec);


	      vectorTools::createDealiiVector<dataTypes::number>(matrix_free_data.get_vector_partitioner(),
							         currentBlockSize,
							         eigenVectorsFlattenedBlock[kPoint]);
	      eigenVectorsFlattenedBlock[kPoint] = dataTypes::number(0.0);
	    }

	  constraintsNoneDataInfo.precomputeMaps(matrix_free_data.get_vector_partitioner(),
						 eigenVectorsFlattenedBlock[0].get_partitioner(),
						 currentBlockSize);
	}

      const bool isRotFracEigenVectorsInBlock=
	((numEigenVectorsFrac!=numEigenVectorsTotal)
	 && (ivec+currentBlockSize)> numEigenVectorsCore
	 && isConsiderSpectrumSplitting)?true:false;

      unsigned int currentBlockSizeFrac=eigenVectorsBlockSize;
      unsigned int startingIndexFracGlobal=0;
      unsigned int startingIndexFrac=0;
      if (isRotFracEigenVectorsInBlock)
	{
	  if (ivec<numEigenVectorsCore)
	    {
	      currentBlockSizeFrac=ivec+currentBlockSize-numEigenVectorsCore;
	      startingIndexFracGlobal=0;
	      startingIndexFrac=numEigenVectorsCore-ivec;
	    }
	  else
	    {
	      currentBlockSizeFrac=currentBlockSize;
	      startingIndexFracGlobal=ivec-numEigenVectorsCore;
	      startingIndexFrac=0;
	    }

	  if (currentBlockSizeFrac!=eigenVectorsRotFrac[0].size() || eigenVectorsRotFrac[0].size()==0)
	    {
	      for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
		{
		  eigenVectorsRotFrac[kPoint].resize(currentBlockSizeFrac);
		  for(unsigned int i= 0; i < currentBlockSizeFrac; ++i)
		    eigenVectorsRotFrac[kPoint][i].reinit(d_tempEigenVec);


		  vectorTools::createDealiiVector<dataTypes::number>
		    (matrix_free_data.get_vector_partitioner(),
		     currentBlockSizeFrac,
		     eigenVectorsRotFracFlattenedBlock[kPoint]);
		  eigenVectorsRotFracFlattenedBlock[kPoint] = dataTypes::number(0.0);
		}

	      constraintsNoneDataInfo2.precomputeMaps(matrix_free_data.get_vector_partitioner(),
						      eigenVectorsRotFracFlattenedBlock[0].get_partitioner(),
						      currentBlockSizeFrac);
	    }
	}

      if((ivec+currentBlockSize)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
	 (ivec+currentBlockSize)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
	{
	  for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
	    {


	      for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
		for(unsigned int iWave = 0; iWave < currentBlockSize; ++iWave)
		  eigenVectorsFlattenedBlock[kPoint].local_element(iNode*currentBlockSize+iWave)
		    = d_eigenVectorsFlattenedSTL[kPoint][iNode*numEigenVectorsTotal+ivec+iWave];

	      constraintsNoneDataInfo.distribute(eigenVectorsFlattenedBlock[kPoint],
						 currentBlockSize);
	      eigenVectorsFlattenedBlock[kPoint].update_ghost_values();

#ifdef USE_COMPLEX
	      vectorTools::copyFlattenedDealiiVecToSingleCompVec
		(eigenVectorsFlattenedBlock[kPoint],
		 currentBlockSize,
		 std::make_pair(0,currentBlockSize),
		 localProc_dof_indicesReal,
		 localProc_dof_indicesImag,
		 eigenVectors[kPoint],
		 false);

	      //FIXME: The underlying call to update_ghost_values
	      //is required because currently localProc_dof_indicesReal
	      //and localProc_dof_indicesImag are only available for
	      //locally owned nodes. Once they are also made available
	      //for ghost nodes- use true for the last argument in
	      //copyFlattenedDealiiVecToSingleCompVec(..) above and supress
	      //underlying call.
	      for(unsigned int i= 0; i < currentBlockSize; ++i)
		eigenVectors[kPoint][i].update_ghost_values();
#else
	      vectorTools::copyFlattenedDealiiVecToSingleCompVec
		(eigenVectorsFlattenedBlock[kPoint],
		 currentBlockSize,
		 std::make_pair(0,currentBlockSize),
		 eigenVectors[kPoint],
		 true);

#endif

	      if (isRotFracEigenVectorsInBlock)
		{

		  for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
		    for(unsigned int iWave = 0; iWave < currentBlockSizeFrac; ++iWave)
		      eigenVectorsRotFracFlattenedBlock[kPoint].local_element(iNode*currentBlockSizeFrac
									      +iWave)
			= d_eigenVectorsRotFracDensityFlattenedSTL[kPoint][iNode*numEigenVectorsFrac
									   +startingIndexFracGlobal+iWave];

		  constraintsNoneDataInfo2.distribute(eigenVectorsRotFracFlattenedBlock[kPoint],
						      currentBlockSizeFrac);
		  eigenVectorsRotFracFlattenedBlock[kPoint].update_ghost_values();

#ifdef USE_COMPLEX
		  vectorTools::copyFlattenedDealiiVecToSingleCompVec
		    (eigenVectorsRotFracFlattenedBlock[kPoint],
		     currentBlockSizeFrac,
		     std::make_pair(0,currentBlockSizeFrac),
		     localProc_dof_indicesReal,
		     localProc_dof_indicesImag,
		     eigenVectorsRotFrac[kPoint],
		     false);

		  //FIXME: The underlying call to update_ghost_values
		  //is required because currently localProc_dof_indicesReal
		  //and localProc_dof_indicesImag are only available for
		  //locally owned nodes. Once they are also made available
		  //for ghost nodes- use true for the last argument in
		  //copyFlattenedDealiiVecToSingleCompVec(..) above and supress
		  //underlying call.
		  for(unsigned int i= 0; i < currentBlockSizeFrac; ++i)
		    eigenVectorsRotFrac[kPoint][i].update_ghost_values();
#else
		  vectorTools::copyFlattenedDealiiVecToSingleCompVec
		    (eigenVectorsRotFracFlattenedBlock[kPoint],
		     currentBlockSizeFrac,
		     std::make_pair(0,currentBlockSizeFrac),
		     eigenVectorsRotFrac[kPoint],
		     true);

#endif
		}
	    }

#ifdef USE_COMPLEX
	  std::vector<Tensor<1,2,VectorizedArray<double> > > psiQuads(numQuadPoints*currentBlockSize*numKPoints,zeroTensor1);
	  std::vector<Tensor<1,2,VectorizedArray<double> > > psiQuads2(numQuadPoints*currentBlockSize*numKPoints,zeroTensor1);
	  std::vector<Tensor<1,2,Tensor<1,3,VectorizedArray<double> > > > gradPsiQuads(numQuadPoints*currentBlockSize*numKPoints,zeroTensor2);
	  std::vector<Tensor<1,2,Tensor<1,3,VectorizedArray<double> > > > gradPsiQuads2(numQuadPoints*currentBlockSize*numKPoints,zeroTensor2);

	  std::vector<Tensor<1,2,VectorizedArray<double> > > psiRotFracQuads(numQuadPoints*currentBlockSizeFrac*numKPoints,zeroTensor1);
	  std::vector<Tensor<1,2,VectorizedArray<double> > > psiRotFracQuads2(numQuadPoints*currentBlockSizeFrac*numKPoints,zeroTensor1);
	  std::vector<Tensor<1,2,Tensor<1,3,VectorizedArray<double> > > > gradPsiRotFracQuads(numQuadPoints*currentBlockSizeFrac*numKPoints,zeroTensor2);
	  std::vector<Tensor<1,2,Tensor<1,3,VectorizedArray<double> > > > gradPsiRotFracQuads2(numQuadPoints*currentBlockSizeFrac*numKPoints,zeroTensor2);
#else
	  std::vector< VectorizedArray<double> > psiQuads(numQuadPoints*currentBlockSize,make_vectorized_array(0.0));
	  std::vector< VectorizedArray<double> > psiQuads2(numQuadPoints*currentBlockSize,make_vectorized_array(0.0));
	  std::vector<Tensor<1,3,VectorizedArray<double> > > gradPsiQuads(numQuadPoints*currentBlockSize,zeroTensor3);
	  std::vector<Tensor<1,3,VectorizedArray<double> > > gradPsiQuads2(numQuadPoints*currentBlockSize,zeroTensor3);

	  std::vector< VectorizedArray<double> > psiRotFracQuads(numQuadPoints*currentBlockSizeFrac,make_vectorized_array(0.0));
	  std::vector< VectorizedArray<double> > psiRotFracQuads2(numQuadPoints*currentBlockSizeFrac,make_vectorized_array(0.0));
	  std::vector<Tensor<1,3,VectorizedArray<double> > > gradPsiRotFracQuads(numQuadPoints*currentBlockSizeFrac,zeroTensor3);
	  std::vector<Tensor<1,3,VectorizedArray<double> > > gradPsiRotFracQuads2(numQuadPoints*currentBlockSizeFrac,zeroTensor3);
#endif

	  for (unsigned int cell=0; cell<matrix_free_data.n_macro_cells(); ++cell)
	    {

              lobattoNodesFlag?psiEvalRefined.reinit(cell):psiEval.reinit(cell);

	      const unsigned int numSubCells=matrix_free_data.n_components_filled(cell);

	      for(unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
		for(unsigned int iEigenVec=0; iEigenVec<currentBlockSize; ++iEigenVec)
		  {

                    lobattoNodesFlag?psiEvalRefined.read_dof_values_plain(eigenVectors[(1+dftParameters::spinPolarized)*kPoint][iEigenVec]):psiEval.read_dof_values_plain(eigenVectors[(1+dftParameters::spinPolarized)*kPoint][iEigenVec]);

		    if(isEvaluateGradRho)
		      lobattoNodesFlag?psiEvalRefined.evaluate(true,true):psiEval.evaluate(true,true);
		    else
		      lobattoNodesFlag?psiEvalRefined.evaluate(true,false):psiEval.evaluate(true,false);

		    for (unsigned int q=0; q<numQuadPoints; ++q)
		      {
			psiQuads[q*currentBlockSize*numKPoints+currentBlockSize*kPoint+iEigenVec]=lobattoNodesFlag?psiEvalRefined.get_value(q):psiEval.get_value(q);
			if(isEvaluateGradRho)
			  gradPsiQuads[q*currentBlockSize*numKPoints+currentBlockSize*kPoint+iEigenVec]=lobattoNodesFlag?psiEvalRefined.get_gradient(q):psiEval.get_gradient(q);
		      }

		    if(dftParameters::spinPolarized==1)
		      {

                        lobattoNodesFlag?psiEvalRefined.read_dof_values_plain
			  (eigenVectors[(1+dftParameters::spinPolarized)*kPoint+1][iEigenVec]):psiEval.read_dof_values_plain
			  (eigenVectors[(1+dftParameters::spinPolarized)*kPoint+1][iEigenVec]);

			if(isEvaluateGradRho)
                          lobattoNodesFlag?psiEvalRefined.evaluate(true,true):psiEval.evaluate(true,true);
			else
                          lobattoNodesFlag?psiEvalRefined.evaluate(true,false):psiEval.evaluate(true,false);
			 

			for (unsigned int q=0; q<numQuadPoints; ++q)
			  {
			    psiQuads2[q*currentBlockSize*numKPoints+currentBlockSize*kPoint+iEigenVec]=lobattoNodesFlag?psiEvalRefined.get_value(q):psiEval.get_value(q);
			    if(isEvaluateGradRho)
			      gradPsiQuads2[q*currentBlockSize*numKPoints+currentBlockSize*kPoint+iEigenVec]= lobattoNodesFlag?psiEvalRefined.get_gradient(q):psiEval.get_gradient(q);
			  }
		      }

		    if (isRotFracEigenVectorsInBlock && iEigenVec>=startingIndexFrac)
		      {

			const unsigned int vectorIndex=iEigenVec-startingIndexFrac;

			lobattoNodesFlag?psiEvalRefined.read_dof_values_plain(eigenVectorsRotFrac[(1+dftParameters::spinPolarized)*kPoint][vectorIndex]):psiEval.read_dof_values_plain(eigenVectorsRotFrac[(1+dftParameters::spinPolarized)*kPoint][vectorIndex]);


			if(isEvaluateGradRho)
			  lobattoNodesFlag?psiEvalRefined.evaluate(true,true):psiEval.evaluate(true,true);
			else
			  lobattoNodesFlag?psiEvalRefined.evaluate(true,false):psiEval.evaluate(true,false);

			for (unsigned int q=0; q<numQuadPoints; ++q)
			  {
			    psiRotFracQuads[q*currentBlockSizeFrac*numKPoints
					    +currentBlockSizeFrac*kPoint+vectorIndex]=lobattoNodesFlag?psiEvalRefined.get_value(q):psiEval.get_value(q);
			    if(isEvaluateGradRho)
			      gradPsiRotFracQuads[q*currentBlockSizeFrac*numKPoints
						  +currentBlockSizeFrac*kPoint+vectorIndex]=lobattoNodesFlag?psiEvalRefined.get_gradient(q):psiEval.get_gradient(q);
			  }

			if(dftParameters::spinPolarized==1)
			  {

			    lobattoNodesFlag?psiEvalRefined.read_dof_values_plain(eigenVectorsRotFrac[(1+dftParameters::spinPolarized)*kPoint+1][vectorIndex]):psiEval.read_dof_values_plain(eigenVectorsRotFrac[(1+dftParameters::spinPolarized)*kPoint+1][vectorIndex]);


			    if(isEvaluateGradRho)
			      lobattoNodesFlag?psiEvalRefined.evaluate(true,true):psiEval.evaluate(true,true);
			    else
			      lobattoNodesFlag?psiEvalRefined.evaluate(true,false):psiEval.evaluate(true,false);

			    for (unsigned int q=0; q<numQuadPoints; ++q)
			      {
				psiRotFracQuads2[q*currentBlockSizeFrac*numKPoints
						 +currentBlockSizeFrac*kPoint+vectorIndex]=lobattoNodesFlag?psiEvalRefined.get_value(q):psiEval.get_value(q);
				if(isEvaluateGradRho)
				  gradPsiRotFracQuads2[q*currentBlockSizeFrac*numKPoints
						       +currentBlockSizeFrac*kPoint+vectorIndex]=lobattoNodesFlag?psiEvalRefined.get_gradient(q):psiEval.get_gradient(q);
			      }
			  }
		      }

		  }//eigenvector per k point

	      for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
		{
		  const dealii::CellId subCellId=matrix_free_data.get_cell_iterator(cell,iSubCell)->id();

		  std::fill(rhoTemp.begin(),rhoTemp.end(),0.0); std::fill(rho.begin(),rho.end(),0.0);

		  if (dftParameters::spinPolarized==1)
		    std::fill(rhoTempSpinPolarized.begin(),rhoTempSpinPolarized.end(),0.0);

		  if(isEvaluateGradRho)
		    {
		      std::fill(gradRhoTemp.begin(),gradRhoTemp.end(),0.0);
		      if (dftParameters::spinPolarized==1)
			std::fill(gradRhoTempSpinPolarized.begin(),gradRhoTempSpinPolarized.end(),0.0);
		    }

		  for(unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
		    {
		      for(unsigned int iEigenVec=0; iEigenVec<currentBlockSize; ++iEigenVec)
			{

			  double partialOccupancy=dftUtils::getPartialOccupancy
			    (eigenValues[kPoint][ivec+iEigenVec],
			     fermiEnergy,
			     C_kb,
			     dftParameters::TVal);

			  double partialOccupancy2=dftUtils::getPartialOccupancy
			    (eigenValues[kPoint][ivec+iEigenVec
						 +dftParameters::spinPolarized*numEigenVectorsTotal],
			     fermiEnergy,
			     C_kb,
			     dftParameters::TVal);
			  if(dftParameters::constraintMagnetization)
			    {
			      partialOccupancy = 1.0 , partialOccupancy2 = 1.0 ;
			      if (eigenValues[kPoint][ivec+iEigenVec
						      +dftParameters::spinPolarized*numEigenVectorsTotal] > fermiEnergyDown)
				partialOccupancy2 = 0.0 ;
			      if (eigenValues[kPoint][ivec+iEigenVec] > fermiEnergyUp)
				partialOccupancy = 0.0 ;

			    }

			  for(unsigned int q=0; q<numQuadPoints; ++q)
			    {
			      const unsigned int id=q*currentBlockSize*numKPoints+currentBlockSize*kPoint+iEigenVec;
#ifdef USE_COMPLEX
			      Vector<double> psi, psi2;
			      psi.reinit(2); psi2.reinit(2);

			      psi(0)= psiQuads[id][0][iSubCell];
			      psi(1)=psiQuads[id][1][iSubCell];

			      if(dftParameters::spinPolarized==1)
				{
				  psi2(0)=psiQuads2[id][0][iSubCell];
				  psi2(1)=psiQuads2[id][1][iSubCell];
				}

			      std::vector<Tensor<1,3,double> > gradPsi(2),gradPsi2(2);

			      if(isEvaluateGradRho)
				for(unsigned int idim=0; idim<3; ++idim)
				  {
				    gradPsi[0][idim]=gradPsiQuads[id][0][idim][iSubCell];
				    gradPsi[1][idim]=gradPsiQuads[id][1][idim][iSubCell];

				    if(dftParameters::spinPolarized==1)
				      {
					gradPsi2[0][idim]=gradPsiQuads2[id][0][idim][iSubCell];
					gradPsi2[1][idim]=gradPsiQuads2[id][1][idim][iSubCell];
				      }
				  }
#else
			      double psi, psi2;
			      psi=psiQuads[id][iSubCell];
			      if (dftParameters::spinPolarized==1)
				psi2=psiQuads2[id][iSubCell];

			      Tensor<1,3,double> gradPsi,gradPsi2;
			      if(isEvaluateGradRho)
				for(unsigned int idim=0; idim<3; ++idim)
				  {
				    gradPsi[idim]=gradPsiQuads[id][idim][iSubCell];
				    if(dftParameters::spinPolarized==1)
				      gradPsi2[idim]=gradPsiQuads2[id][idim][iSubCell];
				  }

#endif

			      if (isRotFracEigenVectorsInBlock && iEigenVec>=startingIndexFrac)
				{
				  const unsigned int idFrac=q*currentBlockSizeFrac*numKPoints
				    +currentBlockSizeFrac*kPoint
				    +iEigenVec-startingIndexFrac;
#ifdef USE_COMPLEX
				  Vector<double> psiRotFrac, psiRotFrac2;
				  psiRotFrac.reinit(2); psiRotFrac2.reinit(2);

				  psiRotFrac(0)= psiRotFracQuads[idFrac][0][iSubCell];
				  psiRotFrac(1)=psiRotFracQuads[idFrac][1][iSubCell];

				  if(dftParameters::spinPolarized==1)
				    {
				      psiRotFrac2(0)=psiRotFracQuads2[idFrac][0][iSubCell];
				      psiRotFrac2(1)=psiRotFracQuads2[idFrac][1][iSubCell];
				    }

				  std::vector<Tensor<1,3,double> > gradPsiRotFrac(2),gradPsiRotFrac2(2);

				  if(isEvaluateGradRho)
				    for(unsigned int idim=0; idim<3; ++idim)
				      {
					gradPsiRotFrac[0][idim]
					  =gradPsiRotFracQuads[idFrac][0][idim][iSubCell];
					gradPsiRotFrac[1][idim]
					  =gradPsiRotFracQuads[idFrac][1][idim][iSubCell];

					if(dftParameters::spinPolarized==1)
					  {
					    gradPsiRotFrac2[0][idim]
					      =gradPsiRotFracQuads2[idFrac][0][idim][iSubCell];
					    gradPsiRotFrac2[1][idim]
					      =gradPsiRotFracQuads2[idFrac][1][idim][iSubCell];
					  }
				      }
#else
				  double psiRotFrac, psiRotFrac2;
				  psiRotFrac=psiRotFracQuads[idFrac][iSubCell];
				  if (dftParameters::spinPolarized==1)
				    psiRotFrac2=psiRotFracQuads2[idFrac][iSubCell];

				  Tensor<1,3,double> gradPsiRotFrac,gradPsiRotFrac2;
				  if(isEvaluateGradRho)
				    for(unsigned int idim=0; idim<3; ++idim)
				      {
					gradPsiRotFrac[idim]
					  =gradPsiRotFracQuads[idFrac][idim][iSubCell];
					if(dftParameters::spinPolarized==1)
					  gradPsiRotFrac2[idim]
					    =gradPsiRotFracQuads2[idFrac][idim][iSubCell];
				      }

#endif

#ifdef USE_COMPLEX
				  if(dftParameters::spinPolarized==1)
				    {
				      rhoTempSpinPolarized[2*q]
					+=
					d_kPointWeights[kPoint]
					*(partialOccupancy*(psiRotFrac(0)*psiRotFrac(0)
							    + psiRotFrac(1)*psiRotFrac(1))
					  -(psiRotFrac(0)*psiRotFrac(0)
					    + psiRotFrac(1)*psiRotFrac(1))
					  +(psi(0)*psi(0) + psi(1)*psi(1)));

				      rhoTempSpinPolarized[2*q+1]
					+=
					d_kPointWeights[kPoint]
					*(partialOccupancy2*(psiRotFrac2(0)*psiRotFrac2(0)
							     + psiRotFrac2(1)*psiRotFrac2(1))
					  -(psiRotFrac2(0)*psiRotFrac2(0)
					    + psiRotFrac2(1)*psiRotFrac2(1))
					  +(psi2(0)*psi2(0)+ psi2(1)*psi2(1)));
				      //
				      if(isEvaluateGradRho)
					for(unsigned int idim=0; idim<3; ++idim)
					  {
					    gradRhoTempSpinPolarized[6*q + idim]
					      += 2.0*d_kPointWeights[kPoint]
					      *(partialOccupancy*(psiRotFrac(0)*gradPsiRotFrac[0][idim]
								  + psiRotFrac(1)*gradPsiRotFrac[1][idim])
						-(psiRotFrac(0)*gradPsiRotFrac[0][idim]
						  + psiRotFrac(1)*gradPsiRotFrac[1][idim])
						+(psi(0)*gradPsi[0][idim] + psi(1)*gradPsi[1][idim]));
					    gradRhoTempSpinPolarized[6*q + 3+idim]
					      += 2.0*d_kPointWeights[kPoint]
					      *(partialOccupancy2*(psiRotFrac2(0)*gradPsiRotFrac2[0][idim]
								   + psiRotFrac2(1)*gradPsiRotFrac2[1][idim])
						-(psiRotFrac2(0)*gradPsiRotFrac2[0][idim]
						  + psiRotFrac2(1)*gradPsiRotFrac2[1][idim])
						+(psi2(0)*gradPsi2[0][idim] + psi2(1)*gradPsi2[1][idim]));
					  }
				    }
				  else
				    {
				      rhoTemp[q] += 2.0*d_kPointWeights[kPoint]
					*(partialOccupancy*(psiRotFrac(0)*psiRotFrac(0)
							    + psiRotFrac(1)*psiRotFrac(1))
					  -(psiRotFrac(0)*psiRotFrac(0)
					    + psiRotFrac(1)*psiRotFrac(1))
					  +(psi(0)*psi(0) + psi(1)*psi(1)));
				      if(isEvaluateGradRho)
					for(unsigned int idim=0; idim<3; ++idim)
					  gradRhoTemp[3*q + idim]
					    += 2.0*2.0*d_kPointWeights[kPoint]
					    *(partialOccupancy*(psiRotFrac(0)*gradPsiRotFrac[0][idim]
								+ psiRotFrac(1)*gradPsiRotFrac[1][idim])
					      -(psiRotFrac(0)*gradPsiRotFrac[0][idim]
						+ psiRotFrac(1)*gradPsiRotFrac[1][idim])
					      +(psi(0)*gradPsi[0][idim] + psi(1)*gradPsi[1][idim]));
				    }
#else
				  if(dftParameters::spinPolarized==1)
				    {
				      rhoTempSpinPolarized[2*q] += (partialOccupancy*psiRotFrac*psiRotFrac
								    -psiRotFrac*psiRotFrac
								    +psi*psi);
				      rhoTempSpinPolarized[2*q+1] +=(partialOccupancy2*psiRotFrac2*psiRotFrac2
								     -psiRotFrac2*psiRotFrac2
								     +psi2*psi2);

				      if(isEvaluateGradRho)
					for(unsigned int idim=0; idim<3; ++idim)
					  {
					    gradRhoTempSpinPolarized[6*q + idim]
					      += 2.0*(partialOccupancy*psiRotFrac*gradPsiRotFrac[idim]
						      -psiRotFrac*gradPsiRotFrac[idim]
						      +psi*gradPsi[idim]);
					    gradRhoTempSpinPolarized[6*q + 3+idim]
					      +=  2.0*(partialOccupancy2*psiRotFrac2*gradPsiRotFrac2[idim]
						       -psiRotFrac2*gradPsiRotFrac2[idim]
						       +psi2*gradPsi2[idim]);
					  }
				    }
				  else
				    {
				      rhoTemp[q] += 2.0*(partialOccupancy*psiRotFrac*psiRotFrac
							 -psiRotFrac*psiRotFrac
							 +psi*psi);

				      if(isEvaluateGradRho)
					for(unsigned int idim=0; idim<3; ++idim)
					  gradRhoTemp[3*q + idim]
					    += 2.0*2.0*(partialOccupancy*psiRotFrac*gradPsiRotFrac[idim]
							-psiRotFrac*gradPsiRotFrac[idim]
							+psi*gradPsi[idim]);
				    }

#endif

				}
			      else
				{
#ifdef USE_COMPLEX
				  if(dftParameters::spinPolarized==1)
				    {
				      rhoTempSpinPolarized[2*q] += partialOccupancy*d_kPointWeights[kPoint]*(psi(0)*psi(0) + psi(1)*psi(1));
				      rhoTempSpinPolarized[2*q+1] += partialOccupancy2*d_kPointWeights[kPoint]*(psi2(0)*psi2(0) + psi2(1)*psi2(1));
				      //
				      if(isEvaluateGradRho)
					for(unsigned int idim=0; idim<3; ++idim)
					  {
					    gradRhoTempSpinPolarized[6*q + idim] +=
					      2.0*partialOccupancy*d_kPointWeights[kPoint]*(psi(0)*gradPsi[0][idim] + psi(1)*gradPsi[1][idim]);
					    gradRhoTempSpinPolarized[6*q + 3+idim] +=
					      2.0*partialOccupancy2*d_kPointWeights[kPoint]*(psi2(0)*gradPsi2[0][idim] + psi2(1)*gradPsi2[1][idim]);
					  }
				    }
				  else
				    {
				      rhoTemp[q] += 2.0*partialOccupancy*d_kPointWeights[kPoint]*(psi(0)*psi(0) + psi(1)*psi(1));
				      if(isEvaluateGradRho)
					for(unsigned int idim=0; idim<3; ++idim)
					  gradRhoTemp[3*q + idim] += 2.0*2.0*partialOccupancy*d_kPointWeights[kPoint]*(psi(0)*gradPsi[0][idim] + psi(1)*gradPsi[1][idim]);
				    }
#else
				  if(dftParameters::spinPolarized==1)
				    {
				      rhoTempSpinPolarized[2*q] += partialOccupancy*psi*psi;
				      rhoTempSpinPolarized[2*q+1] += partialOccupancy2*psi2*psi2;

				      if(isEvaluateGradRho)
					for(unsigned int idim=0; idim<3; ++idim)
					  {
					    gradRhoTempSpinPolarized[6*q + idim] += 2.0*partialOccupancy*(psi*gradPsi[idim]);
					    gradRhoTempSpinPolarized[6*q + 3+idim] +=  2.0*partialOccupancy2*(psi2*gradPsi2[idim]);
					  }
				    }
				  else
				    {
				      rhoTemp[q] += 2.0*partialOccupancy*psi*psi;

				      if(isEvaluateGradRho)
					for(unsigned int idim=0; idim<3; ++idim)
					  gradRhoTemp[3*q + idim] += 2.0*2.0*partialOccupancy*psi*gradPsi[idim];
				    }

#endif
				}

			    }//quad point loop
			}//block eigenvectors per k point
		    }

		  for (unsigned int q=0; q<numQuadPoints; ++q)
		    {
		      if(dftParameters::spinPolarized==1)
			{
			  (*_rhoValuesSpinPolarized)[subCellId][2*q]+=rhoTempSpinPolarized[2*q];
			  (*_rhoValuesSpinPolarized)[subCellId][2*q+1]+=rhoTempSpinPolarized[2*q+1];

			  if(isEvaluateGradRho)
			    for(unsigned int idim=0; idim<3; ++idim)
			      {
				(*_gradRhoValuesSpinPolarized)[subCellId][6*q+idim]
				  +=gradRhoTempSpinPolarized[6*q + idim];
				(*_gradRhoValuesSpinPolarized)[subCellId][6*q+3+idim]
				  +=gradRhoTempSpinPolarized[6*q + 3+idim];
			      }

			  (*_rhoValues)[subCellId][q]+= rhoTempSpinPolarized[2*q] + rhoTempSpinPolarized[2*q+1];

			  if(isEvaluateGradRho)
			    for(unsigned int idim=0; idim<3; ++idim)
			      (*_gradRhoValues)[subCellId][3*q + idim]
				+= gradRhoTempSpinPolarized[6*q + idim]
				+ gradRhoTempSpinPolarized[6*q + 3+idim];
			}
		      else
			{
			  (*_rhoValues)[subCellId][q] += rhoTemp[q];

			  if(isEvaluateGradRho)
			    for(unsigned int idim=0; idim<3; ++idim)
			      (*_gradRhoValues)[subCellId][3*q+idim]+= gradRhoTemp[3*q+idim];
			}
		    }
		}//subcell loop
	    }//macro cell loop
	}//band parallelization
    }//eigenvectors block loop

  //gather density from all inter communicators
  sumRhoData(_rhoValues,
	     _gradRhoValues,
	     _rhoValuesSpinPolarized,
	     _gradRhoValuesSpinPolarized,
	     isEvaluateGradRho,
	     interBandGroupComm);

  sumRhoData(_rhoValues,
	     _gradRhoValues,
	     _rhoValuesSpinPolarized,
	     _gradRhoValuesSpinPolarized,
	     isEvaluateGradRho,
	     interpoolcomm);
}
