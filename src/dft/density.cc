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
void dftClass<FEOrder>::compute_rhoOut()
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
		    true);


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
void dftClass<FEOrder>::computeRhoFromPSI
                                (std::map<dealii::CellId, std::vector<double> > * _rhoValues,
	                         std::map<dealii::CellId, std::vector<double> > * _gradRhoValues,
	                         std::map<dealii::CellId, std::vector<double> > * _rhoValuesSpinPolarized,
		                 std::map<dealii::CellId, std::vector<double> > * _gradRhoValuesSpinPolarized,
		                 const bool isEvaluateGradRho,
				 const bool isConsiderUnrotatedFractionalEigenVec)
{
   const unsigned int numEigenVectorsTotal=d_numEigenValues;
   const unsigned int numEigenVectorsFrac=d_numEigenValuesRR;
   const unsigned int numEigenVectorsCore=d_numEigenValues-d_numEigenValuesRR;
   const unsigned int numKPoints=d_kPointWeights.size();

#ifdef USE_COMPLEX
   FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>(),2> psiEval(matrix_free_data,eigenDofHandlerIndex , 0);
#else
   FEEvaluation<3,FEOrder,C_num1DQuad<FEOrder>(),1> psiEval(matrix_free_data,eigenDofHandlerIndex , 0);
#endif
   const unsigned int numQuadPoints=psiEval.n_q_points;

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


   std::vector<std::vector<vectorType>> eigenVectorsUnrotFrac((1+dftParameters::spinPolarized)*d_kPointWeights.size());
   std::vector<dealii::parallel::distributed::Vector<dataTypes::number> > eigenVectorsUnrotFracFlattenedBlock((1+dftParameters::spinPolarized)*d_kPointWeights.size());

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

      const bool isUnrotFracEigenVectorsInBlock=
	              ((numEigenVectorsFrac!=numEigenVectorsTotal)
	               && (ivec+currentBlockSize)> numEigenVectorsCore
		       && isConsiderUnrotatedFractionalEigenVec)?true:false;

      unsigned int currentBlockSizeFrac=eigenVectorsBlockSize;
      unsigned int startingIndexFracGlobal=ivec;
      unsigned int startingIndexFrac=0;
      if (isUnrotFracEigenVectorsInBlock)
      {
	  if (ivec<numEigenVectorsCore)
	  {
	    currentBlockSizeFrac=ivec+currentBlockSize-numEigenVectorsCore;
	    startingIndexFracGlobal=numEigenVectorsCore;
	    startingIndexFrac=numEigenVectorsCore-ivec;
	  }
	  else
	  {
	    currentBlockSizeFrac=currentBlockSize;
	    startingIndexFracGlobal=ivec;
	    startingIndexFrac=0;
	  }

	  if (currentBlockSizeFrac!=eigenVectorsUnrotFrac.size() || eigenVectorsUnrotFrac.size()==0)
	  {
	       for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*d_kPointWeights.size(); ++kPoint)
	       {
		  eigenVectorsUnrotFrac[kPoint].resize(currentBlockSizeFrac);
		  for(unsigned int i= 0; i < currentBlockSizeFrac; ++i)
		      eigenVectorsUnrotFrac[kPoint][i].reinit(d_tempEigenVec);


		  vectorTools::createDealiiVector<dataTypes::number>
		                               (matrix_free_data.get_vector_partitioner(),
					        currentBlockSizeFrac,
					        eigenVectorsUnrotFracFlattenedBlock[kPoint]);
		  eigenVectorsUnrotFracFlattenedBlock[kPoint] = dataTypes::number(0.0);
	       }

	       constraintsNoneDataInfo2.precomputeMaps(matrix_free_data.get_vector_partitioner(),
						       eigenVectorsUnrotFracFlattenedBlock[0].get_partitioner(),
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

                 if (isUnrotFracEigenVectorsInBlock)
		 {

		     for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
			for(unsigned int iWave = 0; iWave < currentBlockSizeFrac; ++iWave)
			    eigenVectorsUnrotFracFlattenedBlock[kPoint].local_element(iNode*currentBlockSizeFrac
				                                                     +iWave)
			      = d_eigenVectorsUnrotFracFlattenedSTL[kPoint][iNode*numEigenVectorsFrac
			                                                    +startingIndexFracGlobal+iWave];

		     constraintsNoneDataInfo2.distribute(eigenVectorsUnrotFracFlattenedBlock[kPoint],
							currentBlockSizeFrac);
		     eigenVectorsUnrotFracFlattenedBlock[kPoint].update_ghost_values();

#ifdef USE_COMPLEX
		     vectorTools::copyFlattenedDealiiVecToSingleCompVec
			     (eigenVectorsUnrotFracFlattenedBlock[kPoint],
			      currentBlockSizeFrac,
			      std::make_pair(0,currentBlockSizeFrac),
			      localProc_dof_indicesReal,
			      localProc_dof_indicesImag,
			      eigenVectorsUnrotFrac[kPoint],
			      false);

		     //FIXME: The underlying call to update_ghost_values
		     //is required because currently localProc_dof_indicesReal
		     //and localProc_dof_indicesImag are only available for
		     //locally owned nodes. Once they are also made available
		     //for ghost nodes- use true for the last argument in
		     //copyFlattenedDealiiVecToSingleCompVec(..) above and supress
		     //underlying call.
		     for(unsigned int i= 0; i < currentBlockSizeFrac; ++i)
			 eigenVectors[kPoint][i].update_ghost_values();
#else
		     vectorTools::copyFlattenedDealiiVecToSingleCompVec
			     (eigenVectorsUnrotFracFlattenedBlock[kPoint],
			      currentBlockSizeFrac,
			      std::make_pair(0,currentBlockSizeFrac),
			      eigenVectorsUnrotFrac[kPoint],
			      true);

#endif
		 }
	  }

#ifdef USE_COMPLEX
	  std::vector<Tensor<1,2,VectorizedArray<double> > > psiQuads(numQuadPoints*currentBlockSize*numKPoints,zeroTensor1);
	  std::vector<Tensor<1,2,VectorizedArray<double> > > psiQuads2(numQuadPoints*currentBlockSize*numKPoints,zeroTensor1);
	  std::vector<Tensor<1,2,Tensor<1,3,VectorizedArray<double> > > > gradPsiQuads(numQuadPoints*currentBlockSize*numKPoints,zeroTensor2);
	  std::vector<Tensor<1,2,Tensor<1,3,VectorizedArray<double> > > > gradPsiQuads2(numQuadPoints*currentBlockSize*numKPoints,zeroTensor2);

	  std::vector<Tensor<1,2,VectorizedArray<double> > > psiUnrotFracQuads(numQuadPoints*currentBlockSizeFrac*numKPoints,zeroTensor1);
	  std::vector<Tensor<1,2,VectorizedArray<double> > > psiUnrotFracQuads2(numQuadPoints*currentBlockSizeFrac*numKPoints,zeroTensor1);
	  std::vector<Tensor<1,2,Tensor<1,3,VectorizedArray<double> > > > gradPsiUnrotFracQuads(numQuadPoints*currentBlockSizeFrac*numKPoints,zeroTensor2);
	  std::vector<Tensor<1,2,Tensor<1,3,VectorizedArray<double> > > > gradPsiUnrotFracQuads2(numQuadPoints*currentBlockSizeFrac*numKPoints,zeroTensor2);
#else
	  std::vector< VectorizedArray<double> > psiQuads(numQuadPoints*currentBlockSize,make_vectorized_array(0.0));
	  std::vector< VectorizedArray<double> > psiQuads2(numQuadPoints*currentBlockSize,make_vectorized_array(0.0));
	  std::vector<Tensor<1,3,VectorizedArray<double> > > gradPsiQuads(numQuadPoints*currentBlockSize,zeroTensor3);
	  std::vector<Tensor<1,3,VectorizedArray<double> > > gradPsiQuads2(numQuadPoints*currentBlockSize,zeroTensor3);

	  std::vector< VectorizedArray<double> > psiUnrotFracQuads(numQuadPoints*currentBlockSizeFrac,make_vectorized_array(0.0));
	  std::vector< VectorizedArray<double> > psiUnrotFracQuads2(numQuadPoints*currentBlockSizeFrac,make_vectorized_array(0.0));
	  std::vector<Tensor<1,3,VectorizedArray<double> > > gradPsiUnrotFracQuads(numQuadPoints*currentBlockSizeFrac,zeroTensor3);
	  std::vector<Tensor<1,3,VectorizedArray<double> > > gradPsiUnrotFracQuads2(numQuadPoints*currentBlockSizeFrac,zeroTensor3);
#endif

	  for (unsigned int cell=0; cell<matrix_free_data.n_macro_cells(); ++cell)
	  {
		  psiEval.reinit(cell);

		  const unsigned int numSubCells=matrix_free_data.n_components_filled(cell);

		  for(unsigned int kPoint = 0; kPoint < numKPoints; ++kPoint)
		      for(unsigned int iEigenVec=0; iEigenVec<currentBlockSize; ++iEigenVec)
			{

			   psiEval.read_dof_values_plain
			       (eigenVectors[(1+dftParameters::spinPolarized)*kPoint][iEigenVec]);

			   if(isEvaluateGradRho)
			      psiEval.evaluate(true,true);
			   else
			      psiEval.evaluate(true,false);

			   for (unsigned int q=0; q<numQuadPoints; ++q)
			   {
			     psiQuads[q*currentBlockSize*numKPoints+currentBlockSize*kPoint+iEigenVec]=psiEval.get_value(q);
			     if(isEvaluateGradRho)
				gradPsiQuads[q*currentBlockSize*numKPoints+currentBlockSize*kPoint+iEigenVec]=psiEval.get_gradient(q);
			   }

			   if(dftParameters::spinPolarized==1)
			   {

			       psiEval.read_dof_values_plain
				   (eigenVectors[(1+dftParameters::spinPolarized)*kPoint+1][iEigenVec]);

			       if(isEvaluateGradRho)
				  psiEval.evaluate(true,true);
			       else
				  psiEval.evaluate(true,false);

			       for (unsigned int q=0; q<numQuadPoints; ++q)
			       {
				 psiQuads2[q*currentBlockSize*numKPoints+currentBlockSize*kPoint+iEigenVec]=psiEval.get_value(q);
				 if(isEvaluateGradRho)
				    gradPsiQuads2[q*currentBlockSize*numKPoints+currentBlockSize*kPoint+iEigenVec]=psiEval.get_gradient(q);
			       }
			   }

			   if (isUnrotFracEigenVectorsInBlock && startingIndexFrac>=iEigenVec)
			   {

			       const unsigned int vectorIndex=iEigenVec-startingIndexFrac;
			       psiEval.read_dof_values_plain
				   (eigenVectorsUnrotFrac
				    [(1+dftParameters::spinPolarized)*kPoint][vectorIndex]);

			       if(isEvaluateGradRho)
				  psiEval.evaluate(true,true);
			       else
				  psiEval.evaluate(true,false);

			       for (unsigned int q=0; q<numQuadPoints; ++q)
			       {
				 psiUnrotFracQuads[q*currentBlockSizeFrac*numKPoints
				     +currentBlockSizeFrac*kPoint+vectorIndex]=psiEval.get_value(q);
				 if(isEvaluateGradRho)
				    gradPsiUnrotFracQuads[q*currentBlockSizeFrac*numKPoints
					+currentBlockSizeFrac*kPoint+vectorIndex]=psiEval.get_gradient(q);
			       }

			       if(dftParameters::spinPolarized==1)
			       {

				   psiEval.read_dof_values_plain
				       (eigenVectorsUnrotFrac[(1+dftParameters::spinPolarized)*kPoint
					                       +1][vectorIndex]);

				   if(isEvaluateGradRho)
				      psiEval.evaluate(true,true);
				   else
				      psiEval.evaluate(true,false);

				   for (unsigned int q=0; q<numQuadPoints; ++q)
				   {
				     psiUnrotFracQuads2[q*currentBlockSizeFrac*numKPoints
					 +currentBlockSizeFrac*kPoint+vectorIndex]=psiEval.get_value(q);
				     if(isEvaluateGradRho)
					gradPsiUnrotFracQuads2[q*currentBlockSizeFrac*numKPoints
					    +currentBlockSizeFrac*kPoint+vectorIndex]=psiEval.get_gradient(q);
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


				  if (isUnrotFracEigenVectorsInBlock && startingIndexFrac>=iEigenVec)
				  {
				      const unsigned int idFrac=q*currentBlockSizeFrac*numKPoints
					             +currentBlockSizeFrac*kPoint
						     +iEigenVec-startingIndexFrac;
#ifdef USE_COMPLEX
				      Vector<double> psiUnrotFrac, psiUnrotFrac2;
				      psiUnrotFrac.reinit(2); psiUnrotFrac2.reinit(2);

				      psi(0)= psiUnrotFracQuads[idFrac][0][iSubCell];
				      psi(1)=psiUnrotFracQuads[idFrac][1][iSubCell];

				      if(dftParameters::spinPolarized==1)
				      {
					psi2(0)=psiUnrotFracQuads2[idFrac][0][iSubCell];
					psi2(1)=psiUnrotFracQuads2[idFrac][1][iSubCell];
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
