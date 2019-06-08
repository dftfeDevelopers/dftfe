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
// @author Sambit Das
//

template<unsigned int FEOrder>
void forceClass<FEOrder>::computeConfigurationalForceSpinPolarizedEEshelbyTensorFPSPFnlLinFE
			      (const MatrixFree<3,double> & matrixFreeData,
			      const unsigned int eigenDofHandlerIndex,
			      const unsigned int phiExtDofHandlerIndex,
			      const unsigned int phiTotDofHandlerIndex,
			      const vectorType & phiTotRhoIn,
			      const vectorType & phiTotRhoOut,
			      const vectorType & phiExt,
		              const std::map<dealii::CellId, std::vector<double> > & pseudoVLoc,
		              const std::map<dealii::CellId, std::vector<double> > & gradPseudoVLoc,
		              const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & gradPseudoVLocAtoms,
			      const vselfBinsManager<FEOrder> & vselfBinsManagerEigen,
			      const MatrixFree<3,double> & matrixFreeDataElectro,
		              const unsigned int phiTotDofHandlerIndexElectro,
		              const unsigned int phiExtDofHandlerIndexElectro,
		              const vectorType & phiTotRhoOutElectro,
		              const vectorType & phiExtElectro,
			      const std::map<dealii::CellId, std::vector<double> > & rhoOutValuesElectro,
			      const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValuesElectro,
		              const std::map<dealii::CellId, std::vector<double> > & pseudoVLocElectro,
		              const std::map<dealii::CellId, std::vector<double> > & gradPseudoVLocElectro,
		              const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & gradPseudoVLocAtomsElectro,
			      const vselfBinsManager<FEOrder> & vselfBinsManagerElectro)
{
  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  std::map<unsigned int, std::vector<double> > forceContributionFPSPLocalGammaAtomsPSP;
  std::map<unsigned int, std::vector<double> > forceContributionFnlGammaAtoms;

  const bool isPseudopotential = dftParameters::isPseudopotential;

  const unsigned int numVectorizedArrayElements=VectorizedArray<double>::n_array_elements;
  FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  forceEval(matrixFreeData,
	                                                        d_forceDofHandlerIndex,
								0);
  FEEvaluation<C_DIM,1,C_num1DQuadPSP<FEOrder>(),C_DIM>  forceEvalNLP(matrixFreeData,
	                                                              d_forceDofHandlerIndex,
								      2);
#ifdef USE_COMPLEX
  FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  forceEvalKPoints(matrixFreeData,
	                                                               d_forceDofHandlerIndex,
								       0);
  FEEvaluation<C_DIM,1,C_num1DQuadPSP<FEOrder>(),C_DIM>  forceEvalKPointsNLP(matrixFreeData,
	                                                                     d_forceDofHandlerIndex,
									     2);
#endif

#ifdef USE_COMPLEX
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),2> psiEvalSpin0(matrixFreeData,
	                                                            eigenDofHandlerIndex,
								    0);
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),2> psiEvalSpin1(matrixFreeData,
	                                                            eigenDofHandlerIndex,
								    0);
  FEEvaluation<C_DIM,FEOrder,C_num1DQuadPSP<FEOrder>(),2> psiEvalSpin0NLP(matrixFreeData,
	                                                            eigenDofHandlerIndex,
								    2);
  FEEvaluation<C_DIM,FEOrder,C_num1DQuadPSP<FEOrder>(),2> psiEvalSpin1NLP(matrixFreeData,
	                                                            eigenDofHandlerIndex,
								    2);
#else
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> psiEvalSpin0(matrixFreeData,
	                                                            eigenDofHandlerIndex,
								    0);
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> psiEvalSpin1(matrixFreeData,
	                                                            eigenDofHandlerIndex,
								    0);
  FEEvaluation<C_DIM,FEOrder,C_num1DQuadPSP<FEOrder>(),1> psiEvalSpin0NLP(matrixFreeData,
	                                                               eigenDofHandlerIndex,
								       2);
  FEEvaluation<C_DIM,FEOrder,C_num1DQuadPSP<FEOrder>(),1> psiEvalSpin1NLP(matrixFreeData,
	                                                                  eigenDofHandlerIndex,
								          2);
#endif

  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiTotOutEval(matrixFreeData,
	                                                          phiTotDofHandlerIndex,
								  0);
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiTotInEval(matrixFreeData,
	                                                            phiTotDofHandlerIndex,
								    0);
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiExtEval(matrixFreeData,
	                                                          phiExtDofHandlerIndex,
								  0);

  QGauss<C_DIM>  quadrature(C_num1DQuad<FEOrder>());

  const unsigned int numQuadPoints=forceEval.n_q_points;
  const unsigned int numQuadPointsNLP=dftParameters::useHigherQuadNLP?
                                      forceEvalNLP.n_q_points:numQuadPoints;
  const unsigned int numEigenVectors=dftPtr->d_numEigenValues;
  const unsigned int numKPoints=dftPtr->d_kPointWeights.size();
  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;
  Tensor<1,2,VectorizedArray<double> > zeroTensor1;zeroTensor1[0]=make_vectorized_array(0.0);zeroTensor1[1]=make_vectorized_array(0.0);
  Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > zeroTensor2;
  Tensor<1,C_DIM,VectorizedArray<double> > zeroTensor3;
  Tensor<2,C_DIM,VectorizedArray<double> > zeroTensor4;
  for (unsigned int idim=0; idim<C_DIM; idim++)
  {
    zeroTensor2[0][idim]=make_vectorized_array(0.0);
    zeroTensor2[1][idim]=make_vectorized_array(0.0);
    zeroTensor3[idim]=make_vectorized_array(0.0);
  }
  for (unsigned int idim=0; idim<C_DIM; idim++)
  {
    for (unsigned int jdim=0; jdim<C_DIM; jdim++)
    {
	zeroTensor4[idim][jdim]=make_vectorized_array(0.0);
    }
  }
  VectorizedArray<double> phiExtFactor=isPseudopotential?make_vectorized_array(1.0):make_vectorized_array(0.0);

  //band group parallelization data structures
  const unsigned int numberBandGroups=
	dealii::Utilities::MPI::n_mpi_processes(dftPtr->interBandGroupComm);
  const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm);
  std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
  dftUtils::createBandParallelizationIndices(dftPtr->interBandGroupComm,
					     numEigenVectors,
					     bandGroupLowHighPlusOneIndices);

  const unsigned int blockSize=std::min(dftParameters::wfcBlockSize,
	                                bandGroupLowHighPlusOneIndices[1]);

  const unsigned int localVectorSize = dftPtr->d_eigenVectorsFlattenedSTL[0].size()/numEigenVectors;
  std::vector<std::vector<vectorType>> eigenVectors((1+dftParameters::spinPolarized)*dftPtr->d_kPointWeights.size());
  std::vector<dealii::parallel::distributed::Vector<dataTypes::number> > eigenVectorsFlattenedBlock((1+dftParameters::spinPolarized)*dftPtr->d_kPointWeights.size());

   for(unsigned int ivec = 0; ivec < numEigenVectors; ivec+=blockSize)
   {
      const unsigned int currentBlockSize=std::min(blockSize,numEigenVectors-ivec);

      if (currentBlockSize!=blockSize || ivec==0)
      {
	   for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*dftPtr->d_kPointWeights.size(); ++kPoint)
	   {
	      eigenVectors[kPoint].resize(currentBlockSize);
	      for(unsigned int i= 0; i < currentBlockSize; ++i)
		  eigenVectors[kPoint][i].reinit(dftPtr->d_tempEigenVec);


	      vectorTools::createDealiiVector<dataTypes::number>(dftPtr->matrix_free_data.get_vector_partitioner(),
							         currentBlockSize,
							         eigenVectorsFlattenedBlock[kPoint]);
	      eigenVectorsFlattenedBlock[kPoint] = dataTypes::number(0.0);
	   }

	   dftPtr->constraintsNoneDataInfo.precomputeMaps(dftPtr->matrix_free_data.get_vector_partitioner(),
					          eigenVectorsFlattenedBlock[0].get_partitioner(),
					          currentBlockSize);
      }

      if((ivec+currentBlockSize)<=bandGroupLowHighPlusOneIndices[2*bandGroupTaskId+1] &&
      	  (ivec+currentBlockSize)>bandGroupLowHighPlusOneIndices[2*bandGroupTaskId])
      {
          std::vector<std::vector<double>> blockedEigenValues(dftPtr->d_kPointWeights.size(),std::vector<double>(2*currentBlockSize,0.0));
	  for(unsigned int kPoint = 0; kPoint < dftPtr->d_kPointWeights.size(); ++kPoint)
	     for (unsigned int iWave=0; iWave<currentBlockSize;++iWave)
	     {
		 blockedEigenValues[kPoint][iWave]=dftPtr->eigenValues[kPoint][ivec+iWave];
		 blockedEigenValues[kPoint][currentBlockSize+iWave]
		     =dftPtr->eigenValues[kPoint][numEigenVectors+ivec+iWave];
	     }

	  for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*dftPtr->d_kPointWeights.size(); ++kPoint)
	  {
		 for(unsigned int iNode = 0; iNode < localVectorSize; ++iNode)
		    for(unsigned int iWave = 0; iWave < currentBlockSize; ++iWave)
			eigenVectorsFlattenedBlock[kPoint].local_element(iNode*currentBlockSize+iWave)
			  = dftPtr->d_eigenVectorsFlattenedSTL[kPoint][iNode*numEigenVectors+ivec+iWave];

		 dftPtr->constraintsNoneDataInfo.distribute(eigenVectorsFlattenedBlock[kPoint],
						    currentBlockSize);
		 eigenVectorsFlattenedBlock[kPoint].update_ghost_values();

#ifdef USE_COMPLEX
		 vectorTools::copyFlattenedDealiiVecToSingleCompVec
			 (eigenVectorsFlattenedBlock[kPoint],
			  currentBlockSize,
			  std::make_pair(0,currentBlockSize),
			  dftPtr->localProc_dof_indicesReal,
			  dftPtr->localProc_dof_indicesImag,
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
	  }

	  std::vector<std::vector<std::vector<dataTypes::number > > > projectorKetTimesPsiSpin0TimesV(numKPoints);
	  std::vector<std::vector<std::vector<dataTypes::number > > > projectorKetTimesPsiSpin1TimesV(numKPoints);
	  if (isPseudopotential)
	  {
	    for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
		 computeNonLocalProjectorKetTimesPsiTimesVFlattened(eigenVectorsFlattenedBlock[2*ikPoint],
							   currentBlockSize,
							   projectorKetTimesPsiSpin0TimesV[ikPoint],
							   ikPoint);
	    for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
		 computeNonLocalProjectorKetTimesPsiTimesVFlattened(eigenVectorsFlattenedBlock[2*ikPoint+1],
							   currentBlockSize,
							   projectorKetTimesPsiSpin1TimesV[ikPoint],
							   ikPoint);
	  }

          for (unsigned int cell=0; cell<matrixFreeData.n_macro_cells(); ++cell)
	  {
	    forceEval.reinit(cell);
#ifdef USE_COMPLEX
	    forceEvalKPoints.reinit(cell);
#endif
	    psiEvalSpin0.reinit(cell);
	    psiEvalSpin1.reinit(cell);

	    if (isPseudopotential && dftParameters::useHigherQuadNLP)
	    {
	      forceEvalNLP.reinit(cell);
#ifdef USE_COMPLEX
	      forceEvalKPointsNLP.reinit(cell);
#endif

	      psiEvalSpin0NLP.reinit(cell);
	      psiEvalSpin1NLP.reinit(cell);
	    }

	    if (d_isElectrostaticsMeshSubdivided)
	    {
	      phiTotOutEval.reinit(cell);
	      phiTotOutEval.read_dof_values_plain(phiTotRhoOut);
	      phiTotOutEval.evaluate(true,false);
	    }

	    if (d_isElectrostaticsMeshSubdivided)
	    {
	      phiExtEval.reinit(cell);
	      phiExtEval.read_dof_values_plain(phiExt);
	      phiExtEval.evaluate(true,false);
	    }

#ifdef USE_COMPLEX
	    //vector of quadPoints, nonlocal atom id, pseudo wave, k point
	    //FIXME: flatten nonlocal atomid id and pseudo wave and k point
	    std::vector<std::vector<std::vector<std::vector<Tensor<1,2,VectorizedArray<double> > > > > >ZetaDeltaVQuads;
	    std::vector<std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > >gradZetaDeltaVQuads;
	    std::vector<std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > >pspnlGammaAtomsQuads;
#else
	    //FIXME: flatten nonlocal atom id and pseudo wave
	    //vector of quadPoints, nonlocal atom id, pseudo wave
	    std::vector<std::vector<std::vector<VectorizedArray<double> > > > ZetaDeltaVQuads;
	    std::vector<std::vector<std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > > > gradZetaDeltaVQuads;
#endif
	    if(isPseudopotential)
	    {
		ZetaDeltaVQuads.resize(numQuadPointsNLP);
		gradZetaDeltaVQuads.resize(numQuadPointsNLP);
#ifdef USE_COMPLEX
		pspnlGammaAtomsQuads.resize(numQuadPointsNLP);
#endif

		for (unsigned int q=0; q<numQuadPointsNLP; ++q)
		{
		  ZetaDeltaVQuads[q].resize(d_nonLocalPSP_ZetalmDeltaVl.size());
		  gradZetaDeltaVQuads[q].resize(d_nonLocalPSP_ZetalmDeltaVl.size());
#ifdef USE_COMPLEX
		  pspnlGammaAtomsQuads[q].resize(d_nonLocalPSP_ZetalmDeltaVl.size());
#endif
		  for (unsigned int i=0; i < d_nonLocalPSP_ZetalmDeltaVl.size(); ++i)
		  {
		    const int numberPseudoWaveFunctions = d_nonLocalPSP_ZetalmDeltaVl[i].size();
#ifdef USE_COMPLEX
		    ZetaDeltaVQuads[q][i].resize(numberPseudoWaveFunctions);
		    gradZetaDeltaVQuads[q][i].resize(numberPseudoWaveFunctions);
		    pspnlGammaAtomsQuads[q][i].resize(numberPseudoWaveFunctions);
		    for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
		    {
			ZetaDeltaVQuads[q][i][iPseudoWave].resize(numKPoints,zeroTensor1);
			gradZetaDeltaVQuads[q][i][iPseudoWave].resize(numKPoints,zeroTensor2);
			pspnlGammaAtomsQuads[q][i][iPseudoWave].resize(numKPoints,zeroTensor2);
		    }
#else
		    ZetaDeltaVQuads[q][i].resize(numberPseudoWaveFunctions,make_vectorized_array(0.0));
		    gradZetaDeltaVQuads[q][i].resize(numberPseudoWaveFunctions,zeroTensor3);
#endif
		  }
		}
	    }

	    const unsigned int numSubCells=matrixFreeData.n_components_filled(cell);

#ifdef USE_COMPLEX
	    std::vector<Tensor<1,2,VectorizedArray<double> > > psiSpin0Quads(numQuadPoints*currentBlockSize*numKPoints,zeroTensor1);
	    std::vector<Tensor<1,2,VectorizedArray<double> > > psiSpin1Quads(numQuadPoints*currentBlockSize*numKPoints,zeroTensor1);
	    std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > > gradPsiSpin0Quads(numQuadPoints*currentBlockSize*numKPoints,zeroTensor2);
	    std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > > gradPsiSpin1Quads(numQuadPoints*currentBlockSize*numKPoints,zeroTensor2);
#else
	    std::vector< VectorizedArray<double> > psiSpin0Quads(numQuadPoints*currentBlockSize,make_vectorized_array(0.0));
	    std::vector< VectorizedArray<double> > psiSpin1Quads(numQuadPoints*currentBlockSize,make_vectorized_array(0.0));
	    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPsiSpin0Quads(numQuadPoints*currentBlockSize,zeroTensor3);
	    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPsiSpin1Quads(numQuadPoints*currentBlockSize,zeroTensor3);
#endif

	    for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
		for (unsigned int iEigenVec=0; iEigenVec<currentBlockSize; ++iEigenVec)
		{
		  psiEvalSpin0.read_dof_values_plain(eigenVectors[2*ikPoint][iEigenVec]);
		  psiEvalSpin0.evaluate(true,true);

		  psiEvalSpin1.read_dof_values_plain(eigenVectors[2*ikPoint+1][iEigenVec]);
		  psiEvalSpin1.evaluate(true,true);

		  for (unsigned int q=0; q<numQuadPoints; ++q)
		  {
		     const int id=q*currentBlockSize*numKPoints+currentBlockSize*ikPoint+iEigenVec;
		     psiSpin0Quads[id]=psiEvalSpin0.get_value(q);
		     psiSpin1Quads[id]=psiEvalSpin1.get_value(q);
		     gradPsiSpin0Quads[id]=psiEvalSpin0.get_gradient(q);
		     gradPsiSpin1Quads[id]=psiEvalSpin1.get_gradient(q);
		  }//quad point loop
		} //eigenvector loop

#ifdef USE_COMPLEX
	    std::vector<Tensor<1,2,VectorizedArray<double> > > psiSpin0QuadsNLP;
	    std::vector<Tensor<1,2,VectorizedArray<double> > > psiSpin1QuadsNLP;
#else
	    std::vector< VectorizedArray<double> > psiSpin0QuadsNLP;
	    std::vector< VectorizedArray<double> > psiSpin1QuadsNLP;
#endif

	    if (isPseudopotential && dftParameters::useHigherQuadNLP)
	    {
#ifdef USE_COMPLEX
		psiSpin0QuadsNLP.resize(numQuadPointsNLP*currentBlockSize*numKPoints,zeroTensor1);
		psiSpin1QuadsNLP.resize(numQuadPointsNLP*currentBlockSize*numKPoints,zeroTensor1);
#else
		psiSpin0QuadsNLP.resize(numQuadPointsNLP*currentBlockSize,make_vectorized_array(0.0));
		psiSpin1QuadsNLP.resize(numQuadPointsNLP*currentBlockSize,make_vectorized_array(0.0));
#endif
		for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
		    for (unsigned int iEigenVec=0; iEigenVec<currentBlockSize; ++iEigenVec)
		    {
		      psiEvalSpin0NLP.read_dof_values_plain(eigenVectors[2*ikPoint][iEigenVec]);
		      psiEvalSpin0NLP.evaluate(true,false);

		      psiEvalSpin1NLP.read_dof_values_plain(eigenVectors[2*ikPoint+1][iEigenVec]);
		      psiEvalSpin1NLP.evaluate(true,false);

		      for (unsigned int q=0; q<numQuadPointsNLP; ++q)
		      {
			 const int id=q*currentBlockSize*numKPoints+currentBlockSize*ikPoint+iEigenVec;
			 psiSpin0QuadsNLP[id]=psiEvalSpin0NLP.get_value(q);
			 psiSpin1QuadsNLP[id]=psiEvalSpin1NLP.get_value(q);
		      }//quad point loop
		    } //eigenvector loop
	    }

	    if(isPseudopotential)
	    {
	       for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
	       {
		  subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell);
		  dealii::CellId subCellId=subCellPtr->id();

		  for (unsigned int q=0; q<numQuadPointsNLP; ++q)
		  {
		    for (unsigned int i=0; i < d_nonLocalPSP_ZetalmDeltaVl.size(); ++i)
		    {
		      const int numberPseudoWaveFunctions = d_nonLocalPSP_ZetalmDeltaVl[i].size();
		      for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
		      {
			if (d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave].find(subCellId)!=d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave].end())
			{
#ifdef USE_COMPLEX
			   for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
			   {
			      ZetaDeltaVQuads[q][i][iPseudoWave][ikPoint][0][iSubCell]=d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*2+q*2+0];
			      ZetaDeltaVQuads[q][i][iPseudoWave][ikPoint][1][iSubCell]=d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*2+q*2+1];
			      for (unsigned int idim=0; idim<C_DIM; idim++)
			      {
				 gradZetaDeltaVQuads[q][i][iPseudoWave][ikPoint][0][idim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*C_DIM*2+q*C_DIM*2+idim*2+0];
				 gradZetaDeltaVQuads[q][i][iPseudoWave][ikPoint][1][idim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*C_DIM*2+q*C_DIM*2+idim*2+1];
				 pspnlGammaAtomsQuads[q][i][iPseudoWave][ikPoint][0][idim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVl_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*C_DIM*2+q*C_DIM*2+idim*2+0];
				 pspnlGammaAtomsQuads[q][i][iPseudoWave][ikPoint][1][idim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVl_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*C_DIM*2+q*C_DIM*2+idim*2+1];
			      }
			   }
#else

			      ZetaDeltaVQuads[q][i][iPseudoWave][iSubCell]=
			       d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave][subCellId][q];

			   for (unsigned int idim=0; idim<C_DIM; idim++)
			       gradZetaDeltaVQuads[q][i][iPseudoWave][idim][iSubCell]=
					   d_nonLocalPSP_gradZetalmDeltaVl[i][iPseudoWave][subCellId][q*C_DIM+idim];
#endif
			}//non-trivial cellId check
		      }//iPseudoWave loop
		    }//i loop
		  }//q loop
	       }//subcell loop
	       //compute FPSPLocalGammaAtoms  (contibution due to Gamma(Rj))

#ifdef USE_COMPLEX

	       FnlGammaAtomsElementalContributionPeriodicSpinPolarized(forceContributionFnlGammaAtoms,
								      forceEval,
								      forceEvalNLP,
								      cell,
								      pspnlGammaAtomsQuads,
								      projectorKetTimesPsiSpin0TimesV,
								      projectorKetTimesPsiSpin1TimesV,
								      dftParameters::useHigherQuadNLP?
								      psiSpin0QuadsNLP:
								      psiSpin0Quads,
								      dftParameters::useHigherQuadNLP?
								      psiSpin1QuadsNLP:
								      psiSpin1Quads,
								      blockedEigenValues);

#else
	       FnlGammaAtomsElementalContributionNonPeriodicSpinPolarized
							    (forceContributionFnlGammaAtoms,
							     forceEval,
							     forceEvalNLP,
							     cell,
							     gradZetaDeltaVQuads,
							     projectorKetTimesPsiSpin0TimesV[0],
							     projectorKetTimesPsiSpin1TimesV[0],
							     dftParameters::useHigherQuadNLP?
							     psiSpin0QuadsNLP:
							     psiSpin0Quads,
							     dftParameters::useHigherQuadNLP?
							     psiSpin1QuadsNLP:
							     psiSpin1Quads,
							     blockedEigenValues);
#endif
	    }//is pseudopotential check

	    for (unsigned int q=0; q<numQuadPoints; ++q)
	    {
	       Tensor<2,C_DIM,VectorizedArray<double> > E=zeroTensor4;
#ifdef USE_COMPLEX
	       Tensor<2,C_DIM,VectorizedArray<double> > EKPoints=eshelbyTensorSP::getELocWfcEshelbyTensorPeriodicKPoints
								 (psiSpin0Quads.begin()+q*currentBlockSize*numKPoints,
								  psiSpin1Quads.begin()+q*currentBlockSize*numKPoints,
								  gradPsiSpin0Quads.begin()+q*currentBlockSize*numKPoints,
								  gradPsiSpin1Quads.begin()+q*currentBlockSize*numKPoints,
								  dftPtr->d_kPointCoordinates,
								  dftPtr->d_kPointWeights,
								  blockedEigenValues,
								  dftPtr->fermiEnergy,
								  dftPtr->fermiEnergyUp,
								  dftPtr->fermiEnergyDown,
								  dftParameters::TVal);
#else
	       E+=eshelbyTensorSP::getELocWfcEshelbyTensorNonPeriodic
						     (psiSpin0Quads.begin()+q*currentBlockSize,
						     psiSpin1Quads.begin()+q*currentBlockSize,
						     gradPsiSpin0Quads.begin()+q*currentBlockSize,
						     gradPsiSpin1Quads.begin()+q*currentBlockSize,
						     blockedEigenValues[0],
						     dftPtr->fermiEnergy,
						     dftPtr->fermiEnergyUp,
						     dftPtr->fermiEnergyDown,
						     dftParameters::TVal);
#endif
	       Tensor<1,C_DIM,VectorizedArray<double> > F=zeroTensor3;

	       if(isPseudopotential)
	       {
		   if (!dftParameters::useHigherQuadNLP)
		   {
#ifdef USE_COMPLEX
		       Tensor<1,C_DIM,VectorizedArray<double> > FKPoints
			 =eshelbyTensorSP::getFnlPeriodic
						       (gradZetaDeltaVQuads[q],
							projectorKetTimesPsiSpin0TimesV,
							projectorKetTimesPsiSpin1TimesV,
							psiSpin0Quads.begin()+q*currentBlockSize*numKPoints,
							psiSpin1Quads.begin()+q*currentBlockSize*numKPoints,
							dftPtr->d_kPointWeights,
							blockedEigenValues,
							dftPtr->fermiEnergy,
							dftPtr->fermiEnergyUp,
							dftPtr->fermiEnergyDown,
							dftParameters::TVal);


		       EKPoints+=eshelbyTensorSP::getEnlEshelbyTensorPeriodic
								    (ZetaDeltaVQuads[q],
								     projectorKetTimesPsiSpin0TimesV,
								     projectorKetTimesPsiSpin1TimesV,
								     psiSpin0Quads.begin()+q*currentBlockSize*numKPoints,
								     psiSpin1Quads.begin()+q*currentBlockSize*numKPoints,
								     dftPtr->d_kPointWeights,
								     blockedEigenValues,
								     dftPtr->fermiEnergy,
								     dftPtr->fermiEnergyUp,
								     dftPtr->fermiEnergyDown,
								     dftParameters::TVal);
		       forceEvalKPoints.submit_value(FKPoints,q);
#else
		       F+=eshelbyTensorSP::getFnlNonPeriodic
							  (gradZetaDeltaVQuads[q],
							   projectorKetTimesPsiSpin0TimesV[0],
							   projectorKetTimesPsiSpin1TimesV[0],
							   psiSpin0Quads.begin()+q*currentBlockSize,
							   psiSpin1Quads.begin()+q*currentBlockSize,
							   blockedEigenValues[0],
							   dftPtr->fermiEnergy,
							   dftPtr->fermiEnergyUp,
							   dftPtr->fermiEnergyDown,
							   dftParameters::TVal);

		       E+=eshelbyTensorSP::getEnlEshelbyTensorNonPeriodic(ZetaDeltaVQuads[q],
									projectorKetTimesPsiSpin0TimesV[0],
									projectorKetTimesPsiSpin1TimesV[0],
									psiSpin0Quads.begin()+q*currentBlockSize,
									psiSpin1Quads.begin()+q*currentBlockSize,
									blockedEigenValues[0],
									dftPtr->fermiEnergy,
									dftPtr->fermiEnergyUp,
									dftPtr->fermiEnergyDown,
									dftParameters::TVal);
#endif
		   }


	       }

	       forceEval.submit_value(F,q);
	       forceEval.submit_gradient(E,q);
#ifdef USE_COMPLEX
	       forceEvalKPoints.submit_gradient(EKPoints,q);
#endif
	    }//quad point loop

	    if (isPseudopotential && dftParameters::useHigherQuadNLP)
		for (unsigned int q=0; q<numQuadPointsNLP; ++q)
		{
#ifdef USE_COMPLEX
		       Tensor<1,C_DIM,VectorizedArray<double> > FKPoints
			    =eshelbyTensorSP::getFnlPeriodic
						       (gradZetaDeltaVQuads[q],
							projectorKetTimesPsiSpin0TimesV,
							projectorKetTimesPsiSpin1TimesV,
							psiSpin0QuadsNLP.begin()+q*currentBlockSize*numKPoints,
							psiSpin1QuadsNLP.begin()+q*currentBlockSize*numKPoints,
							dftPtr->d_kPointWeights,
							blockedEigenValues,
							dftPtr->fermiEnergy,
							dftPtr->fermiEnergyUp,
							dftPtr->fermiEnergyDown,
							dftParameters::TVal);

		       Tensor<2,C_DIM,VectorizedArray<double> > EKPoints
			   =eshelbyTensorSP::getEnlEshelbyTensorPeriodic
								    (ZetaDeltaVQuads[q],
								     projectorKetTimesPsiSpin0TimesV,
								     projectorKetTimesPsiSpin1TimesV,
								     psiSpin0QuadsNLP.begin()+q*currentBlockSize*numKPoints,
								     psiSpin1QuadsNLP.begin()+q*currentBlockSize*numKPoints,
								     dftPtr->d_kPointWeights,
								     blockedEigenValues,
								     dftPtr->fermiEnergy,
								     dftPtr->fermiEnergyUp,
								     dftPtr->fermiEnergyDown,
								     dftParameters::TVal);
		       forceEvalKPointsNLP.submit_value(FKPoints,q);
		       forceEvalKPointsNLP.submit_gradient(EKPoints,q);
#else
		       Tensor<1,C_DIM,VectorizedArray<double> > F
			 =eshelbyTensorSP::getFnlNonPeriodic
							  (gradZetaDeltaVQuads[q],
							   projectorKetTimesPsiSpin0TimesV[0],
							   projectorKetTimesPsiSpin1TimesV[0],
							   psiSpin0QuadsNLP.begin()+q*currentBlockSize,
							   psiSpin1QuadsNLP.begin()+q*currentBlockSize,
							   blockedEigenValues[0],
							   dftPtr->fermiEnergy,
							   dftPtr->fermiEnergyUp,
							   dftPtr->fermiEnergyDown,
							   dftParameters::TVal);
		       Tensor<2,C_DIM,VectorizedArray<double> >	E
			 =eshelbyTensorSP::getEnlEshelbyTensorNonPeriodic(ZetaDeltaVQuads[q],
									projectorKetTimesPsiSpin0TimesV[0],
									projectorKetTimesPsiSpin1TimesV[0],
									psiSpin0QuadsNLP.begin()+q*currentBlockSize,
									psiSpin1QuadsNLP.begin()+q*currentBlockSize,
									blockedEigenValues[0],
									dftPtr->fermiEnergy,
									dftPtr->fermiEnergyUp,
									dftPtr->fermiEnergyDown,
									dftParameters::TVal);
		       forceEvalNLP.submit_value(F,q);
		       forceEvalNLP.submit_gradient(E,q);
#endif
		}//nonlocal psp quad points loop

	    if(isPseudopotential)
	    {
	      forceEval.integrate(true,true);
#ifdef USE_COMPLEX
	      if (dftParameters::useHigherQuadNLP)
		  forceEvalKPoints.integrate(false,true);
	      else
		  forceEvalKPoints.integrate(true,true);
#endif
	       if (dftParameters::useHigherQuadNLP)
	       {
#ifdef USE_COMPLEX
		  forceEvalKPointsNLP.integrate(true,true);
#else
		  forceEvalNLP.integrate(true,true);
#endif
	       }
	    }
	    else
	    {
	      forceEval.integrate (false,true);
#ifdef USE_COMPLEX
	      forceEvalKPoints.integrate(false,true);
#endif
	    }

	    forceEval.distribute_local_to_global(d_configForceVectorLinFE);//also takes care of constraints
#ifdef USE_COMPLEX
	    forceEvalKPoints.distribute_local_to_global(d_configForceVectorLinFEKPoints);
#endif
	    if (isPseudopotential && dftParameters::useHigherQuadNLP)
	    {
#ifdef USE_COMPLEX
		forceEvalKPointsNLP.distribute_local_to_global(d_configForceVectorLinFEKPoints);
#else
		forceEvalNLP.distribute_local_to_global(d_configForceVectorLinFE);
#endif
	    }
	  }//macro cell loop
      }//band parallelization loop
  }//wavefunction block loop

  // add global FPSPLocal contribution due to Gamma(Rj) to the configurational force vector
  if(isPseudopotential)
  {
     distributeForceContributionFnlGammaAtoms(forceContributionFnlGammaAtoms);
  }

  /////////// Compute contribution independent of wavefunctions /////////////////
  if (bandGroupTaskId==0)
  {
      std::vector<VectorizedArray<double> > rhoQuads(numQuadPoints,make_vectorized_array(0.0));
      std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoSpin0Quads(numQuadPoints,zeroTensor3);
      std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoSpin1Quads(numQuadPoints,zeroTensor3);
      std::vector<Tensor<2,C_DIM,VectorizedArray<double> > > hessianRhoSpin0Quads(numQuadPoints,zeroTensor4);
      std::vector<Tensor<2,C_DIM,VectorizedArray<double> > > hessianRhoSpin1Quads(numQuadPoints,zeroTensor4);
      std::vector<VectorizedArray<double> > excQuads(numQuadPoints,make_vectorized_array(0.0));
      std::vector<VectorizedArray<double> > pseudoVLocQuads(numQuadPoints,make_vectorized_array(0.0));
      std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPseudoVLocQuads(numQuadPoints,zeroTensor3);
      std::vector<VectorizedArray<double> > vEffRhoInSpin0Quads(numQuadPoints,make_vectorized_array(0.0));
      std::vector<VectorizedArray<double> > vEffRhoInSpin1Quads(numQuadPoints,make_vectorized_array(0.0));
      std::vector<VectorizedArray<double> > vEffRhoOutSpin0Quads(numQuadPoints,make_vectorized_array(0.0));
      std::vector<VectorizedArray<double> > vEffRhoOutSpin1Quads(numQuadPoints,make_vectorized_array(0.0));
      std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > derExchCorrEnergyWithGradRhoInSpin0Quads(numQuadPoints,zeroTensor3);
      std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > derExchCorrEnergyWithGradRhoInSpin1Quads(numQuadPoints,zeroTensor3);
      std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > derExchCorrEnergyWithGradRhoOutSpin0Quads(numQuadPoints,zeroTensor3);
      std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > derExchCorrEnergyWithGradRhoOutSpin1Quads(numQuadPoints,zeroTensor3);

      for (unsigned int cell=0; cell<matrixFreeData.n_macro_cells(); ++cell)
      {
	forceEval.reinit(cell);

	if (d_isElectrostaticsMeshSubdivided)
	{
	  phiTotOutEval.reinit(cell);
	  phiTotOutEval.read_dof_values_plain(phiTotRhoOut);
	  phiTotOutEval.evaluate(true,false);
	}

	if (d_isElectrostaticsMeshSubdivided)
	{
	  phiExtEval.reinit(cell);
	  phiExtEval.read_dof_values_plain(phiExt);
	  phiExtEval.evaluate(true,false);
	}

	std::fill(rhoQuads.begin(),rhoQuads.end(),make_vectorized_array(0.0));
	std::fill(gradRhoSpin0Quads.begin(),gradRhoSpin0Quads.end(),zeroTensor3);
	std::fill(gradRhoSpin1Quads.begin(),gradRhoSpin1Quads.end(),zeroTensor3);
	std::fill(hessianRhoSpin0Quads.begin(),hessianRhoSpin0Quads.end(),zeroTensor4);
	std::fill(hessianRhoSpin1Quads.begin(),hessianRhoSpin1Quads.end(),zeroTensor4);
	std::fill(excQuads.begin(),excQuads.end(),make_vectorized_array(0.0));
	std::fill(pseudoVLocQuads.begin(),pseudoVLocQuads.end(),make_vectorized_array(0.0));
	std::fill(gradPseudoVLocQuads.begin(),gradPseudoVLocQuads.end(),zeroTensor3);
	std::fill(vEffRhoInSpin0Quads.begin(),vEffRhoInSpin0Quads.end(),make_vectorized_array(0.0));
	std::fill(vEffRhoInSpin1Quads.begin(),vEffRhoInSpin1Quads.end(),make_vectorized_array(0.0));
	std::fill(vEffRhoOutSpin0Quads.begin(),vEffRhoOutSpin0Quads.end(),make_vectorized_array(0.0));
	std::fill(vEffRhoOutSpin1Quads.begin(),vEffRhoOutSpin1Quads.end(),make_vectorized_array(0.0));
	std::fill(derExchCorrEnergyWithGradRhoInSpin0Quads.begin(),derExchCorrEnergyWithGradRhoInSpin0Quads.end(),zeroTensor3);
	std::fill(derExchCorrEnergyWithGradRhoInSpin1Quads.begin(),derExchCorrEnergyWithGradRhoInSpin1Quads.end(),zeroTensor3);
	std::fill(derExchCorrEnergyWithGradRhoOutSpin0Quads.begin(),derExchCorrEnergyWithGradRhoOutSpin0Quads.end(),zeroTensor3);
	std::fill(derExchCorrEnergyWithGradRhoOutSpin1Quads.begin(),derExchCorrEnergyWithGradRhoOutSpin1Quads.end(),zeroTensor3);

	const unsigned int numSubCells=matrixFreeData.n_components_filled(cell);
	//For LDA
	std::vector<double> exchValRhoOut(numQuadPoints);
	std::vector<double> corrValRhoOut(numQuadPoints);
	std::vector<double> exchPotValRhoOut(2*numQuadPoints);
	std::vector<double> corrPotValRhoOut(2*numQuadPoints);
	std::vector<double> exchValRhoIn(numQuadPoints);
	std::vector<double> corrValRhoIn(numQuadPoints);
	std::vector<double> exchPotValRhoIn(2*numQuadPoints);
	std::vector<double> corrPotValRhoIn(2*numQuadPoints);
	//
	//For GGA
	std::vector<double> sigmaValRhoOut(3*numQuadPoints);
	std::vector<double> derExchEnergyWithDensityValRhoOut(2*numQuadPoints), derCorrEnergyWithDensityValRhoOut(2*numQuadPoints), derExchEnergyWithSigmaRhoOut(3*numQuadPoints),derCorrEnergyWithSigmaRhoOut(3*numQuadPoints);
	std::vector<double> sigmaValRhoIn(3*numQuadPoints);
	std::vector<double> derExchEnergyWithDensityValRhoIn(2*numQuadPoints), derCorrEnergyWithDensityValRhoIn(2*numQuadPoints), derExchEnergyWithSigmaRhoIn(3*numQuadPoints),derCorrEnergyWithSigmaRhoIn(3*numQuadPoints);
	std::vector<Tensor<1,C_DIM,double > > gradRhoInSpin0(numQuadPoints);
	std::vector<Tensor<1,C_DIM,double > > gradRhoInSpin1(numQuadPoints);
	std::vector<Tensor<1,C_DIM,double > > gradRhoOutSpin0(numQuadPoints);
	std::vector<Tensor<1,C_DIM,double > > gradRhoOutSpin1(numQuadPoints);
	//
	for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
	{
	   subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell);
	   dealii::CellId subCellId=subCellPtr->id();
	   if(dftParameters::xc_id == 4)
	   {
	      for (unsigned int q = 0; q < numQuadPoints; ++q)
	      {
		  for (unsigned int idim=0; idim<C_DIM; idim++)
		  {
		    gradRhoOutSpin0[q][idim] = ((*dftPtr->gradRhoOutValuesSpinPolarized)[subCellId][6*q + idim]);
		    gradRhoOutSpin1[q][idim] = ((*dftPtr->gradRhoOutValuesSpinPolarized)[subCellId][6*q +3+idim]);
		  }
		  sigmaValRhoOut[3*q+0] = scalar_product(gradRhoOutSpin0[q],gradRhoOutSpin0[q]);
		  sigmaValRhoOut[3*q+1] = scalar_product(gradRhoOutSpin0[q],gradRhoOutSpin1[q]);
		  sigmaValRhoOut[3*q+2] = scalar_product(gradRhoOutSpin1[q],gradRhoOutSpin1[q]);

		  for (unsigned int idim=0; idim<C_DIM; idim++)
		  {
		    gradRhoInSpin0[q][idim] = ((*dftPtr->gradRhoInValuesSpinPolarized)[subCellId][6*q + idim]);
		    gradRhoInSpin1[q][idim] = ((*dftPtr->gradRhoInValuesSpinPolarized)[subCellId][6*q +3+idim]);
		  }
		  sigmaValRhoIn[3*q+0] = scalar_product(gradRhoInSpin0[q],gradRhoInSpin0[q]);
		  sigmaValRhoIn[3*q+1] = scalar_product(gradRhoInSpin0[q],gradRhoInSpin1[q]);
		  sigmaValRhoIn[3*q+2] = scalar_product(gradRhoInSpin1[q],gradRhoInSpin1[q]);
	      }
	      xc_gga_exc_vxc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoOutValuesSpinPolarized)[subCellId][0]),&sigmaValRhoOut[0],&exchValRhoOut[0],&derExchEnergyWithDensityValRhoOut[0],&derExchEnergyWithSigmaRhoOut[0]);
	      xc_gga_exc_vxc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoOutValuesSpinPolarized)[subCellId][0]),&sigmaValRhoOut[0],&corrValRhoOut[0],&derCorrEnergyWithDensityValRhoOut[0],&derCorrEnergyWithSigmaRhoOut[0]);
	      xc_gga_exc_vxc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoInValuesSpinPolarized)[subCellId][0]),&sigmaValRhoIn[0],&exchValRhoIn[0],&derExchEnergyWithDensityValRhoIn[0],&derExchEnergyWithSigmaRhoIn[0]);
	      xc_gga_exc_vxc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoInValuesSpinPolarized)[subCellId][0]),&sigmaValRhoIn[0],&corrValRhoIn[0],&derCorrEnergyWithDensityValRhoIn[0],&derCorrEnergyWithSigmaRhoIn[0]);
	      for (unsigned int q=0; q<numQuadPoints; ++q)
	      {
		 excQuads[q][iSubCell]=exchValRhoOut[q]+corrValRhoOut[q];
		 vEffRhoInSpin0Quads[q][iSubCell]+= derExchEnergyWithDensityValRhoIn[2*q]+derCorrEnergyWithDensityValRhoIn[2*q];
		 vEffRhoInSpin1Quads[q][iSubCell]+= derExchEnergyWithDensityValRhoIn[2*q+1]+derCorrEnergyWithDensityValRhoIn[2*q+1];
		 vEffRhoOutSpin0Quads[q][iSubCell]+= derExchEnergyWithDensityValRhoOut[2*q]+derCorrEnergyWithDensityValRhoOut[2*q];
		 vEffRhoOutSpin1Quads[q][iSubCell]+= derExchEnergyWithDensityValRhoOut[2*q+1]+derCorrEnergyWithDensityValRhoOut[2*q+1];
		  for (unsigned int idim=0; idim<C_DIM; idim++)
		  {
		     derExchCorrEnergyWithGradRhoInSpin0Quads[q][idim][iSubCell]=2.0*(derExchEnergyWithSigmaRhoIn[3*q+0]+derCorrEnergyWithSigmaRhoIn[3*q+0])*gradRhoInSpin0[q][idim];
		     derExchCorrEnergyWithGradRhoInSpin0Quads[q][idim][iSubCell]+=(derExchEnergyWithSigmaRhoIn[3*q+1]+derCorrEnergyWithSigmaRhoIn[3*q+1])*gradRhoInSpin1[q][idim];

		     derExchCorrEnergyWithGradRhoInSpin1Quads[q][idim][iSubCell]+=2.0*(derExchEnergyWithSigmaRhoIn[3*q+2]+derCorrEnergyWithSigmaRhoIn[3*q+2])*gradRhoInSpin1[q][idim];
		     derExchCorrEnergyWithGradRhoInSpin1Quads[q][idim][iSubCell]+=(derExchEnergyWithSigmaRhoIn[3*q+1]+derCorrEnergyWithSigmaRhoIn[3*q+1])*gradRhoInSpin0[q][idim];

		     derExchCorrEnergyWithGradRhoOutSpin0Quads[q][idim][iSubCell]=2.0*(derExchEnergyWithSigmaRhoOut[3*q+0]+derCorrEnergyWithSigmaRhoOut[3*q+0])*gradRhoOutSpin0[q][idim];
		     derExchCorrEnergyWithGradRhoOutSpin0Quads[q][idim][iSubCell]+=(derExchEnergyWithSigmaRhoOut[3*q+1]+derCorrEnergyWithSigmaRhoOut[3*q+1])*gradRhoOutSpin1[q][idim];

		     derExchCorrEnergyWithGradRhoOutSpin1Quads[q][idim][iSubCell]+=2.0*(derExchEnergyWithSigmaRhoOut[3*q+2]+derCorrEnergyWithSigmaRhoOut[3*q+2])*gradRhoOutSpin1[q][idim];
		     derExchCorrEnergyWithGradRhoOutSpin1Quads[q][idim][iSubCell]+=(derExchEnergyWithSigmaRhoOut[3*q+1]+derCorrEnergyWithSigmaRhoOut[3*q+1])*gradRhoOutSpin0[q][idim];
		  }
	      }

	   }
	   else
	   {
	      xc_lda_exc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoOutValuesSpinPolarized)[subCellId][0]),&exchValRhoOut[0]);
	      xc_lda_exc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoOutValuesSpinPolarized)[subCellId][0]),&corrValRhoOut[0]);
	      xc_lda_vxc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoOutValuesSpinPolarized)[subCellId][0]),&exchPotValRhoOut[0]);
	      xc_lda_vxc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoOutValuesSpinPolarized)[subCellId][0]),&corrPotValRhoOut[0]);
	      xc_lda_vxc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoInValuesSpinPolarized)[subCellId][0]),&exchPotValRhoIn[0]);
	      xc_lda_vxc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoInValuesSpinPolarized)[subCellId][0]),&corrPotValRhoIn[0]);
	      for (unsigned int q=0; q<numQuadPoints; ++q)
	      {
		 excQuads[q][iSubCell]=exchValRhoOut[q]+corrValRhoOut[q];
		 vEffRhoInSpin0Quads[q][iSubCell]+= exchPotValRhoIn[2*q]+corrPotValRhoIn[2*q];
		 vEffRhoInSpin1Quads[q][iSubCell]+= exchPotValRhoIn[2*q+1]+corrPotValRhoIn[2*q+1];
		 vEffRhoOutSpin0Quads[q][iSubCell]+= exchPotValRhoOut[2*q]+corrPotValRhoOut[2*q];
		 vEffRhoOutSpin1Quads[q][iSubCell]+= exchPotValRhoOut[2*q+1]+corrPotValRhoOut[2*q+1];

	      }
	   }

	   for (unsigned int q=0; q<numQuadPoints; ++q)
	   {
	     rhoQuads[q][iSubCell]=(*dftPtr->rhoOutValues)[subCellId][q];

	     for (unsigned int idim=0; idim<C_DIM; idim++)
	     {
		gradRhoSpin0Quads[q][idim][iSubCell]=(*dftPtr->gradRhoOutValuesSpinPolarized)[subCellId][6*q+idim];
		gradRhoSpin1Quads[q][idim][iSubCell]=(*dftPtr->gradRhoOutValuesSpinPolarized)[subCellId][6*q+3+idim];
	     }
	   }
	}

	if(isPseudopotential)
	   for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
	   {
	      subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell);
	      dealii::CellId subCellId=subCellPtr->id();
	      for (unsigned int q=0; q<numQuadPoints; ++q)
		 pseudoVLocQuads[q][iSubCell]=pseudoVLoc.find(subCellId)->second[q];
	    }

	for (unsigned int q=0; q<numQuadPoints; ++q)
	{
	   const VectorizedArray<double> phiTot_q =d_isElectrostaticsMeshSubdivided?
						    phiTotOutEval.get_value(q)
						    :make_vectorized_array(0.0);
	   const VectorizedArray<double> phiExt_q =d_isElectrostaticsMeshSubdivided?
						    phiExtEval.get_value(q)
						    :make_vectorized_array(0.0);

	   Tensor<2,C_DIM,VectorizedArray<double> > E=eshelbyTensorSP::getELocXcEshelbyTensor
					  (rhoQuads[q],
					   gradRhoSpin0Quads[q],
					   gradRhoSpin1Quads[q],
					   excQuads[q],
					   derExchCorrEnergyWithGradRhoOutSpin0Quads[q],
					   derExchCorrEnergyWithGradRhoOutSpin1Quads[q]);

	   Tensor<1,C_DIM,VectorizedArray<double> > F=zeroTensor3;

	   if(d_isElectrostaticsMeshSubdivided)
	       F-=(gradRhoSpin0Quads[q]+gradRhoSpin1Quads[q])*phiTot_q;

	   if(isPseudopotential)
	       if(d_isElectrostaticsMeshSubdivided)
		  F-=(gradRhoSpin0Quads[q]+gradRhoSpin1Quads[q])*(pseudoVLocQuads[q]-phiExt_q);

	   forceEval.submit_value(F,q);
	   forceEval.submit_gradient(E,q);
	}//quad point loop

	forceEval.integrate(true,true);
	forceEval.distribute_local_to_global(d_configForceVectorLinFE);//also takes care of constraints
      }

      ////Add electrostatic configurational force contribution////////////////
      computeConfigurationalForceEEshelbyEElectroPhiTot
			    (matrixFreeDataElectro,
			     phiTotDofHandlerIndexElectro,
			     phiExtDofHandlerIndexElectro,
			     phiTotRhoOutElectro,
			     phiExtElectro,
			     rhoOutValuesElectro,
			     gradRhoOutValuesElectro,
			     pseudoVLocElectro,
			     gradPseudoVLocElectro,
			     gradPseudoVLocAtomsElectro,
			     vselfBinsManagerElectro);
  }
}
