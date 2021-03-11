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
//compute configurational stress contribution from all terms except the nuclear self energy
template<unsigned int FEOrder,unsigned int FEOrderElectro>
	void forceClass<FEOrder,FEOrderElectro>::computeStressSpinPolarizedEEshelbyEPSPEnlEk
(const MatrixFree<3,double> & matrixFreeData,
 const unsigned int eigenDofHandlerIndex,
 const unsigned int smearedChargeQuadratureId,
  const unsigned int lpspQuadratureIdElectro,          
 const MatrixFree<3,double> & matrixFreeDataElectro,
 const unsigned int phiTotDofHandlerIndexElectro,
 const distributedCPUVec<double> & phiTotRhoOutElectro,
  const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValuesLpsp,         
  const std::map<dealii::CellId, std::vector<double> > & rhoOutValuesElectro,
  const std::map<dealii::CellId, std::vector<double> > & rhoOutValuesElectroLpsp,         
  const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValuesElectro,
  const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValuesElectroLpsp,
  const std::map<dealii::CellId, std::vector<double> > & pseudoVLocElectro,
  const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & pseudoVLocAtomsElectro,
	const std::map<dealii::CellId, std::vector<double> > & rhoCoreValues,
	const std::map<dealii::CellId, std::vector<double> > & gradRhoCoreValues,
	const std::map<dealii::CellId, std::vector<double> > & hessianRhoCoreValues,
	const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & gradRhoCoreAtoms,
	const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & hessianRhoCoreAtoms,    
 const vselfBinsManager<FEOrder,FEOrderElectro> & vselfBinsManagerElectro)
{
	const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
	std::map<unsigned int, std::vector<double> > forceContributionFPSPLocalGammaAtomsPSP;
	std::map<unsigned int, std::vector<double> > forceContributionFnlGammaAtoms;

	const bool isPseudopotential = dftParameters::isPseudopotential;

	FEEvaluation<C_DIM,1,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),C_DIM>  forceEval(matrixFreeData,
			d_forceDofHandlerIndex,
			dftPtr->d_densityQuadratureId);
	FEEvaluation<C_DIM,1,C_num1DQuadNLPSP<FEOrder>()*C_numCopies1DQuadNLPSP(),C_DIM>  forceEvalNLP(matrixFreeData,
			d_forceDofHandlerIndex,
			dftPtr->d_nlpspQuadratureId);
#ifdef USE_COMPLEX
	FEEvaluation<C_DIM,1,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),C_DIM>  forceEvalKPoints(matrixFreeData,
			d_forceDofHandlerIndex,
			dftPtr->d_densityQuadratureId);
	FEEvaluation<C_DIM,1,C_num1DQuadNLPSP<FEOrder>()*C_numCopies1DQuadNLPSP(),C_DIM>  forceEvalKPointsNLP(matrixFreeData,
			d_forceDofHandlerIndex,
			dftPtr->d_nlpspQuadratureId);
#endif


#ifdef USE_COMPLEX
	FEEvaluation<C_DIM,FEOrder,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),2> psiEvalSpin0(matrixFreeData,
			eigenDofHandlerIndex,
			dftPtr->d_densityQuadratureId);
	FEEvaluation<C_DIM,FEOrder,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),2> psiEvalSpin1(matrixFreeData,
			eigenDofHandlerIndex,
			dftPtr->d_densityQuadratureId);
	FEEvaluation<C_DIM,FEOrder,C_num1DQuadNLPSP<FEOrder>()*C_numCopies1DQuadNLPSP(),2> psiEvalSpin0NLP(matrixFreeData,
			eigenDofHandlerIndex,
			dftPtr->d_nlpspQuadratureId);
	FEEvaluation<C_DIM,FEOrder,C_num1DQuadNLPSP<FEOrder>()*C_numCopies1DQuadNLPSP(),2> psiEvalSpin1NLP(matrixFreeData,
			eigenDofHandlerIndex,
			dftPtr->d_nlpspQuadratureId);
#else
	FEEvaluation<C_DIM,FEOrder,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),1> psiEvalSpin0(matrixFreeData,
			eigenDofHandlerIndex,
			dftPtr->d_densityQuadratureId);
	FEEvaluation<C_DIM,FEOrder,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),1> psiEvalSpin1(matrixFreeData,
			eigenDofHandlerIndex,
			dftPtr->d_densityQuadratureId);
	FEEvaluation<C_DIM,FEOrder,C_num1DQuadNLPSP<FEOrder>()*C_numCopies1DQuadNLPSP(),1> psiEvalSpin0NLP(matrixFreeData,
			eigenDofHandlerIndex,
			dftPtr->d_nlpspQuadratureId);
	FEEvaluation<C_DIM,FEOrder,C_num1DQuadNLPSP<FEOrder>()*C_numCopies1DQuadNLPSP(),1> psiEvalSpin1NLP(matrixFreeData,
			eigenDofHandlerIndex,
			dftPtr->d_nlpspQuadratureId);
#endif


	QGauss<C_DIM>  quadrature(C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>());

	const unsigned int numQuadPoints=forceEval.n_q_points;
	const unsigned int numQuadPointsNLP=forceEvalNLP.n_q_points;
	const unsigned int numEigenVectors=dftPtr->d_numEigenValues;
	const unsigned int numKPoints=dftPtr->d_kPointWeights.size();

	DoFHandler<C_DIM>::active_cell_iterator subCellPtr;
	Tensor<1,2,VectorizedArray<double> > zeroTensor1;zeroTensor1[0]=make_vectorized_array(0.0);zeroTensor1[1]=make_vectorized_array(0.0);
	Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > zeroTensor2;
	Tensor<1,C_DIM,VectorizedArray<double> > zeroTensor3;
	Tensor<2,C_DIM,VectorizedArray<double> > zeroTensor4;
	Tensor<1,2, Tensor<2,C_DIM,VectorizedArray<double> > > zeroTensor5;
	for (unsigned int idim=0; idim<C_DIM; idim++){
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
	zeroTensor5[0]=zeroTensor4;
	zeroTensor5[1]=zeroTensor4;

	std::map<unsigned int,std::vector<unsigned int>> macroIdToNonlocalAtomsSetMap;
	for (unsigned int cell=0; cell<matrixFreeData.n_macro_cells(); ++cell)
	{
		const unsigned int numSubCells=matrixFreeData.n_components_filled(cell);
		std::set<unsigned int> mergedSet;
		for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
		{
			subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell);
			dealii::CellId subCellId=subCellPtr->id();

			std::set<unsigned int> s;
			std::set_union(mergedSet.begin(), mergedSet.end(),
					dftPtr->d_cellIdToNonlocalAtomIdsLocalCompactSupportMap[subCellId].begin(), dftPtr->d_cellIdToNonlocalAtomIdsLocalCompactSupportMap[subCellId].end(),
					std::inserter(s, s.begin()));
			mergedSet=s;
		}
		macroIdToNonlocalAtomsSetMap[cell]=std::vector<unsigned int>(mergedSet.begin(),mergedSet.end());
	}

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

  //allocate storage for vector of quadPoints, nonlocal atom id, pseudo wave, k point
  //FIXME: flatten nonlocal atomid id and pseudo wave and k point
#ifdef USE_COMPLEX
  std::vector<std::vector<std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > >>zetalmDeltaVlProductDistImageAtomsQuads;
#else   
  std::vector<std::vector<std::vector<std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > > > > zetalmDeltaVlProductDistImageAtomsQuads;
#endif 

	if(isPseudopotential)
	{
    zetalmDeltaVlProductDistImageAtomsQuads.resize(matrixFreeData.n_macro_cells());
    for (unsigned int cell=0; cell<matrixFreeData.n_macro_cells(); ++cell)
    {
      const unsigned int numSubCells=matrixFreeData.n_components_filled(cell);    
      zetalmDeltaVlProductDistImageAtomsQuads[cell].resize(numQuadPointsNLP);
      for (unsigned int q=0; q<numQuadPointsNLP; ++q)
      {
        zetalmDeltaVlProductDistImageAtomsQuads[cell][q].resize(dftPtr->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint.size());
        for (unsigned int i=0; i < dftPtr->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint.size(); ++i)
        {
          const int numberPseudoWaveFunctions = dftPtr->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint[i].size();
          zetalmDeltaVlProductDistImageAtomsQuads[cell][q][i].resize(numberPseudoWaveFunctions);
          for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
          {
#ifdef USE_COMPLEX            
            zetalmDeltaVlProductDistImageAtomsQuads[cell][q][i][iPseudoWave].resize(numKPoints,zeroTensor2);
#else
            zetalmDeltaVlProductDistImageAtomsQuads[cell][q][i][iPseudoWave]=zeroTensor3;
#endif            
          }
        }
      }
    }

    for (unsigned int cell=0; cell<matrixFreeData.n_macro_cells(); ++cell)
    {
      const unsigned int numSubCells=matrixFreeData.n_components_filled(cell);
			for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
			{
				subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell);
				dealii::CellId subCellId=subCellPtr->id();

				for (unsigned int q=0; q<numQuadPointsNLP; ++q)
				{
					for (unsigned int i=0; i < dftPtr->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint.size(); ++i)
					{
						const int numberPseudoWaveFunctions = dftPtr->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint[i].size();
						for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
						{
							if (dftPtr->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint[i][iPseudoWave].find(subCellId)!=dftPtr->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint[i][iPseudoWave].end())
							{
								for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
								{
									for (unsigned int idim=0; idim<C_DIM; idim++)
									{
#ifdef USE_COMPLEX                      
											zetalmDeltaVlProductDistImageAtomsQuads[cell][q][i][iPseudoWave][ikPoint][0][idim][iSubCell]=dftPtr->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*C_DIM*2+q*C_DIM*2+idim*2+0];
											zetalmDeltaVlProductDistImageAtomsQuads[cell][q][i][iPseudoWave][ikPoint][1][idim][iSubCell]=dftPtr->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*C_DIM*2+q*C_DIM*2+idim*2+1];
#else
											zetalmDeltaVlProductDistImageAtomsQuads[cell][q][i][iPseudoWave][idim][iSubCell]=dftPtr->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*C_DIM+q*C_DIM+idim];
#endif                      
									}
								}
							}//non-trivial cellId check
						}//iPseudoWave loop
					}//i loop
				}//q loop
			}//subcell loop   
    }//macrocell loop
	}

	const unsigned int localVectorSize = dftPtr->d_eigenVectorsFlattenedSTL[0].size()/numEigenVectors;
	std::vector<std::vector<distributedCPUVec<double>>> eigenVectors((1+dftParameters::spinPolarized)*dftPtr->d_kPointWeights.size());
	std::vector<distributedCPUVec<dataTypes::number> > eigenVectorsFlattenedBlock((1+dftParameters::spinPolarized)*dftPtr->d_kPointWeights.size());

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
			std::vector<std::vector<double>> blockedPartialOccupanciesSpin0(dftPtr->d_kPointWeights.size(),
					std::vector<double>(currentBlockSize,0.0));
			std::vector<std::vector<double>> blockedPartialOccupanciesSpin1(dftPtr->d_kPointWeights.size(),
					std::vector<double>(currentBlockSize,0.0));
			for(unsigned int kPoint = 0; kPoint < dftPtr->d_kPointWeights.size(); ++kPoint)
				for (unsigned int iWave=0; iWave<currentBlockSize;++iWave)
				{
					blockedEigenValues[kPoint][iWave]=dftPtr->eigenValues[kPoint][ivec+iWave];
					blockedEigenValues[kPoint][currentBlockSize+iWave]
						=dftPtr->eigenValues[kPoint][numEigenVectors+ivec+iWave];

					blockedPartialOccupanciesSpin0[kPoint][iWave]
						=dftUtils::getPartialOccupancy(blockedEigenValues[kPoint][iWave],
								dftPtr->fermiEnergy,
								C_kb,
								dftParameters::TVal);
					blockedPartialOccupanciesSpin1[kPoint][iWave]
						=dftUtils::getPartialOccupancy(blockedEigenValues[kPoint][iWave+currentBlockSize],
								dftPtr->fermiEnergy,
								C_kb,
								dftParameters::TVal);
					if(dftParameters::constraintMagnetization)
					{
						blockedPartialOccupanciesSpin0[kPoint][iWave] = 1.0;
						blockedPartialOccupanciesSpin1[kPoint][iWave] = 1.0 ;
						if (blockedEigenValues[kPoint][iWave]> dftPtr->fermiEnergyUp)
							blockedPartialOccupanciesSpin0[kPoint][iWave] = 0.0 ;
						if (blockedEigenValues[kPoint][iWave+currentBlockSize] > dftPtr->fermiEnergyDown)
							blockedPartialOccupanciesSpin1[kPoint][iWave] = 0.0 ;
					}
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

			std::vector<std::vector<std::vector<dataTypes::number > > > projectorKetTimesPsiSpin0TimesVTimesPartOcc(numKPoints);
			std::vector<std::vector<std::vector<dataTypes::number > > > projectorKetTimesPsiSpin1TimesVTimesPartOcc(numKPoints);
			if (isPseudopotential)
			{
				for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
					computeNonLocalProjectorKetTimesPsiTimesVFlattened(eigenVectorsFlattenedBlock[2*ikPoint],
							currentBlockSize,
							projectorKetTimesPsiSpin0TimesVTimesPartOcc[ikPoint],
							ikPoint,
							blockedPartialOccupanciesSpin0[ikPoint],
							true);
				for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
					computeNonLocalProjectorKetTimesPsiTimesVFlattened(eigenVectorsFlattenedBlock[2*ikPoint+1],
							currentBlockSize,
							projectorKetTimesPsiSpin1TimesVTimesPartOcc[ikPoint],
							ikPoint,
							blockedPartialOccupanciesSpin1[ikPoint],
							true);
			}

			for (unsigned int cell=0; cell<matrixFreeData.n_macro_cells(); ++cell)
			{
				forceEval.reinit(cell);
#ifdef USE_COMPLEX
				forceEvalKPoints.reinit(cell);
#endif
				psiEvalSpin0.reinit(cell);
				psiEvalSpin1.reinit(cell);

				if (isPseudopotential)
				{
					forceEvalNLP.reinit(cell);
#ifdef USE_COMPLEX
					forceEvalKPointsNLP.reinit(cell);
#endif

					psiEvalSpin0NLP.reinit(cell);
					psiEvalSpin1NLP.reinit(cell);
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
				std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > > gradPsiSpin0QuadsNLP;
				std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > > gradPsiSpin1QuadsNLP;        
#else
				std::vector< VectorizedArray<double> > psiSpin0QuadsNLP;
				std::vector< VectorizedArray<double> > psiSpin1QuadsNLP;
				std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPsiSpin0QuadsNLP;
				std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPsiSpin1QuadsNLP;        
#endif

				if (isPseudopotential)
				{
#ifdef USE_COMPLEX
					psiSpin0QuadsNLP.resize(numQuadPointsNLP*currentBlockSize*numKPoints,zeroTensor1);
					psiSpin1QuadsNLP.resize(numQuadPointsNLP*currentBlockSize*numKPoints,zeroTensor1);
          gradPsiSpin0QuadsNLP.resize(numQuadPointsNLP*currentBlockSize*numKPoints,zeroTensor2);
          gradPsiSpin1QuadsNLP.resize(numQuadPointsNLP*currentBlockSize*numKPoints,zeroTensor2);
#else
					psiSpin0QuadsNLP.resize(numQuadPointsNLP*currentBlockSize,make_vectorized_array(0.0));
					psiSpin1QuadsNLP.resize(numQuadPointsNLP*currentBlockSize,make_vectorized_array(0.0));
          gradPsiSpin0QuadsNLP.resize(numQuadPointsNLP*currentBlockSize,zeroTensor3);
          gradPsiSpin1QuadsNLP.resize(numQuadPointsNLP*currentBlockSize,zeroTensor3);          
#endif
					for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
						for (unsigned int iEigenVec=0; iEigenVec<currentBlockSize; ++iEigenVec)
						{
							psiEvalSpin0NLP.read_dof_values_plain(eigenVectors[2*ikPoint][iEigenVec]);
							psiEvalSpin0NLP.evaluate(true,true);

							psiEvalSpin1NLP.read_dof_values_plain(eigenVectors[2*ikPoint+1][iEigenVec]);
							psiEvalSpin1NLP.evaluate(true,true);

							for (unsigned int q=0; q<numQuadPointsNLP; ++q)
							{
								const int id=q*currentBlockSize*numKPoints+currentBlockSize*ikPoint+iEigenVec;
								psiSpin0QuadsNLP[id]=psiEvalSpin0NLP.get_value(q);
								psiSpin1QuadsNLP[id]=psiEvalSpin1NLP.get_value(q);
								gradPsiSpin0QuadsNLP[id]=psiEvalSpin0NLP.get_gradient(q);
								gradPsiSpin1QuadsNLP[id]=psiEvalSpin1NLP.get_gradient(q);                
							}//quad point loop
						} //eigenvector loop
				}

        Tensor<2,C_DIM,VectorizedArray<double> > EKPointsQuadSum=zeroTensor4;
        for (unsigned int q=0; q<numQuadPoints; ++q)
        {
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

          EKPoints+=eshelbyTensorSP::getEKStress
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
          Tensor<2,C_DIM,VectorizedArray<double> > EKPoints=eshelbyTensorSP::getELocWfcEshelbyTensorNonPeriodic
            (psiSpin0Quads.begin()+q*currentBlockSize*numKPoints,
             psiSpin1Quads.begin()+q*currentBlockSize*numKPoints,
             gradPsiSpin0Quads.begin()+q*currentBlockSize*numKPoints,
             gradPsiSpin1Quads.begin()+q*currentBlockSize*numKPoints,
             blockedEigenValues[0],
             dftPtr->fermiEnergy,
             dftPtr->fermiEnergyUp,
             dftPtr->fermiEnergyDown,
             dftParameters::TVal);
#endif

          EKPointsQuadSum+=EKPoints*forceEval.JxW(q);
        }//quad point loop

        if (isPseudopotential)
        {
          for (unsigned int q=0; q<numQuadPointsNLP; ++q)
          {
#ifdef USE_COMPLEX              
            Tensor<2,C_DIM,VectorizedArray<double> > Enl
              =eshelbyTensorSP::getEnlStress(zetalmDeltaVlProductDistImageAtomsQuads[cell][q],
                projectorKetTimesPsiSpin0TimesVTimesPartOcc,
                projectorKetTimesPsiSpin1TimesVTimesPartOcc,
                psiSpin0QuadsNLP.begin()+q*currentBlockSize*numKPoints,
                psiSpin1QuadsNLP.begin()+q*currentBlockSize*numKPoints,
                gradPsiSpin0QuadsNLP.begin()+q*currentBlockSize*numKPoints,
                gradPsiSpin1QuadsNLP.begin()+q*currentBlockSize*numKPoints,            
                dftPtr->d_kPointWeights,
                dftPtr->d_kPointCoordinates,
                macroIdToNonlocalAtomsSetMap[cell],
                currentBlockSize);
#else
            Tensor<2,C_DIM,VectorizedArray<double> > Enl
              =eshelbyTensorSP::getEnlStress(zetalmDeltaVlProductDistImageAtomsQuads[cell][q],
                projectorKetTimesPsiSpin0TimesVTimesPartOcc,
                projectorKetTimesPsiSpin1TimesVTimesPartOcc,
                psiSpin0QuadsNLP.begin()+q*currentBlockSize*numKPoints,
                psiSpin1QuadsNLP.begin()+q*currentBlockSize*numKPoints,
                gradPsiSpin0QuadsNLP.begin()+q*currentBlockSize*numKPoints,
                gradPsiSpin1QuadsNLP.begin()+q*currentBlockSize*numKPoints,            
                macroIdToNonlocalAtomsSetMap[cell],
                currentBlockSize);
#endif          


            EKPointsQuadSum+=Enl*forceEvalNLP.JxW(q);
          }
        }

        for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
          for (unsigned int idim=0; idim<C_DIM; ++idim)
            for (unsigned int jdim=0; jdim<C_DIM; ++jdim)
            {
              d_stressKPoints[idim][jdim]+=EKPointsQuadSum[idim][jdim][iSubCell];
            }  

			}//macro cell loop
		}//band parallelization loop
	}//wavefunction block loop

	/////////// Compute contribution independent of wavefunctions /////////////////
	if (bandGroupTaskId==0)
	{
    std::vector<VectorizedArray<double> > rhoXCQuadsVect(numQuadPoints,make_vectorized_array(0.0)); 
		std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoSpin0QuadsVect(numQuadPoints,zeroTensor3);
		std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoSpin1QuadsVect(numQuadPoints,zeroTensor3);
		std::vector<VectorizedArray<double> > excQuads(numQuadPoints,make_vectorized_array(0.0));
		std::vector<VectorizedArray<double> > vxcRhoOutSpin0Quads(numQuadPoints,make_vectorized_array(0.0));
		std::vector<VectorizedArray<double> > vxcRhoOutSpin1Quads(numQuadPoints,make_vectorized_array(0.0));
		std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > derExchCorrEnergyWithGradRhoOutSpin0Quads(numQuadPoints,zeroTensor3);
		std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > derExchCorrEnergyWithGradRhoOutSpin1Quads(numQuadPoints,zeroTensor3);
		std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoCoreQuads(numQuadPoints,zeroTensor3);
		std::vector<Tensor<2,C_DIM,VectorizedArray<double> > > hessianRhoCoreQuads(numQuadPoints,zeroTensor4);        

		for (unsigned int cell=0; cell<matrixFreeData.n_macro_cells(); ++cell)
		{
			forceEval.reinit(cell);

			std::fill(rhoXCQuadsVect.begin(),rhoXCQuadsVect.end(),make_vectorized_array(0.0));       
			std::fill(gradRhoSpin0QuadsVect.begin(),gradRhoSpin0QuadsVect.end(),zeroTensor3);
			std::fill(gradRhoSpin1QuadsVect.begin(),gradRhoSpin1QuadsVect.end(),zeroTensor3);
			std::fill(excQuads.begin(),excQuads.end(),make_vectorized_array(0.0));
			std::fill(vxcRhoOutSpin0Quads.begin(),vxcRhoOutSpin0Quads.end(),make_vectorized_array(0.0));
			std::fill(vxcRhoOutSpin1Quads.begin(),vxcRhoOutSpin1Quads.end(),make_vectorized_array(0.0));
			std::fill(derExchCorrEnergyWithGradRhoOutSpin0Quads.begin(),derExchCorrEnergyWithGradRhoOutSpin0Quads.end(),zeroTensor3);
			std::fill(derExchCorrEnergyWithGradRhoOutSpin1Quads.begin(),derExchCorrEnergyWithGradRhoOutSpin1Quads.end(),zeroTensor3);
			std::fill(gradRhoCoreQuads.begin(),gradRhoCoreQuads.end(),zeroTensor3);
			std::fill(hessianRhoCoreQuads.begin(),hessianRhoCoreQuads.end(),zeroTensor4);      

			const unsigned int numSubCells=matrixFreeData.n_components_filled(cell);
			//For LDA
			std::vector<double> exchValRhoOut(numQuadPoints);
			std::vector<double> corrValRhoOut(numQuadPoints);
			std::vector<double> exchPotValRhoOut(2*numQuadPoints);
			std::vector<double> corrPotValRhoOut(2*numQuadPoints);
			std::vector<double> rhoOutQuadsXC(2*numQuadPoints);

			//
			//For GGA
			std::vector<double> sigmaValRhoOut(3*numQuadPoints);
			std::vector<double> derExchEnergyWithDensityValRhoOut(2*numQuadPoints), derCorrEnergyWithDensityValRhoOut(2*numQuadPoints), derExchEnergyWithSigmaRhoOut(3*numQuadPoints),derCorrEnergyWithSigmaRhoOut(3*numQuadPoints);
			std::vector<Tensor<1,C_DIM,double > > gradRhoOutQuadsXCSpin0(numQuadPoints);
			std::vector<Tensor<1,C_DIM,double > > gradRhoOutQuadsXCSpin1(numQuadPoints);

			//
			for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
			{
				subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell);
				dealii::CellId subCellId=subCellPtr->id();

        const std::vector<double> & temp=(*dftPtr->rhoOutValues).find(subCellId)->second;
        const std::vector<double> & temp1=(*dftPtr->rhoOutValuesSpinPolarized).find(subCellId)->second;

        rhoOutQuadsXC=temp1;
				for (unsigned int q=0; q<numQuadPoints; ++q)
        {
          rhoXCQuadsVect[q][iSubCell]=temp[q];
        }

				if(dftParameters::nonLinearCoreCorrection)
        {
          const std::vector<double> & temp2=rhoCoreValues.find(subCellId)->second;
          for (unsigned int q=0; q<numQuadPoints; ++q)
          {
            rhoOutQuadsXC[2*q+0]+=temp2[q]/2.0;
            rhoOutQuadsXC[2*q+1]+=temp2[q]/2.0;
            rhoXCQuadsVect[q][iSubCell]+=temp2[q];
          }
        }

				if(dftParameters::xcFamilyType=="GGA")
				{
          const std::vector<double> & temp3=(*dftPtr->gradRhoOutValuesSpinPolarized).find(subCellId)->second;          
					for (unsigned int q = 0; q < numQuadPoints; ++q)
						for (unsigned int idim=0; idim<C_DIM; idim++)
            {
              gradRhoOutQuadsXCSpin0[q][idim]= temp3[6*q + idim];
              gradRhoOutQuadsXCSpin1[q][idim]= temp3[6*q + 3+idim];
							gradRhoSpin0QuadsVect[q][idim][iSubCell]=temp3[6*q+idim];   
              gradRhoSpin1QuadsVect[q][idim][iSubCell]=temp3[6*q+3+idim]; 
            }

          if(dftParameters::nonLinearCoreCorrection)
          {
            const std::vector<double> & temp4=gradRhoCoreValues.find(subCellId)->second;
            for (unsigned int q = 0; q < numQuadPoints; ++q)
              for (unsigned int idim=0; idim<C_DIM; idim++)
              {
                gradRhoOutQuadsXCSpin0[q][idim]+= temp4[3*q + idim]/2.0;
                gradRhoOutQuadsXCSpin1[q][idim]+= temp4[3*q + idim]/2.0;
              }        
          }          
        }

        if(dftParameters::xcFamilyType=="GGA")
				{
					for (unsigned int q = 0; q < numQuadPoints; ++q)
					{
						sigmaValRhoOut[3*q+0] = scalar_product(gradRhoOutQuadsXCSpin0[q],gradRhoOutQuadsXCSpin0[q]);
						sigmaValRhoOut[3*q+1] = scalar_product(gradRhoOutQuadsXCSpin0[q],gradRhoOutQuadsXCSpin1[q]);
						sigmaValRhoOut[3*q+2] = scalar_product(gradRhoOutQuadsXCSpin1[q],gradRhoOutQuadsXCSpin1[q]);
					}

					xc_gga_exc_vxc(&(dftPtr->funcX),numQuadPoints,&(rhoOutQuadsXC[0]),&sigmaValRhoOut[0],&exchValRhoOut[0],&derExchEnergyWithDensityValRhoOut[0],&derExchEnergyWithSigmaRhoOut[0]);
					xc_gga_exc_vxc(&(dftPtr->funcC),numQuadPoints,&(rhoOutQuadsXC[0]),&sigmaValRhoOut[0],&corrValRhoOut[0],&derCorrEnergyWithDensityValRhoOut[0],&derCorrEnergyWithSigmaRhoOut[0]);

					for (unsigned int q=0; q<numQuadPoints; ++q)
					{
						excQuads[q][iSubCell]=exchValRhoOut[q]+corrValRhoOut[q];
						vxcRhoOutSpin0Quads[q][iSubCell]= derExchEnergyWithDensityValRhoOut[2*q]+derCorrEnergyWithDensityValRhoOut[2*q];
						vxcRhoOutSpin1Quads[q][iSubCell]= derExchEnergyWithDensityValRhoOut[2*q+1]+derCorrEnergyWithDensityValRhoOut[2*q+1];
						for (unsigned int idim=0; idim<C_DIM; idim++)
						{
							derExchCorrEnergyWithGradRhoOutSpin0Quads[q][idim][iSubCell]=2.0*(derExchEnergyWithSigmaRhoOut[3*q+0]+derCorrEnergyWithSigmaRhoOut[3*q+0])*gradRhoOutQuadsXCSpin0[q][idim];
							derExchCorrEnergyWithGradRhoOutSpin0Quads[q][idim][iSubCell]+=(derExchEnergyWithSigmaRhoOut[3*q+1]+derCorrEnergyWithSigmaRhoOut[3*q+1])*gradRhoOutQuadsXCSpin1[q][idim];

							derExchCorrEnergyWithGradRhoOutSpin1Quads[q][idim][iSubCell]+=2.0*(derExchEnergyWithSigmaRhoOut[3*q+2]+derCorrEnergyWithSigmaRhoOut[3*q+2])*gradRhoOutQuadsXCSpin1[q][idim];
							derExchCorrEnergyWithGradRhoOutSpin1Quads[q][idim][iSubCell]+=(derExchEnergyWithSigmaRhoOut[3*q+1]+derCorrEnergyWithSigmaRhoOut[3*q+1])*gradRhoOutQuadsXCSpin0[q][idim];
						}
					}

				}
				else
				{
					xc_lda_exc(&(dftPtr->funcX),numQuadPoints,&(rhoOutQuadsXC[0]),&exchValRhoOut[0]);
					xc_lda_exc(&(dftPtr->funcC),numQuadPoints,&(rhoOutQuadsXC[0]),&corrValRhoOut[0]);
					xc_lda_vxc(&(dftPtr->funcX),numQuadPoints,&(rhoOutQuadsXC[0]),&exchPotValRhoOut[0]);
					xc_lda_vxc(&(dftPtr->funcC),numQuadPoints,&(rhoOutQuadsXC[0]),&corrPotValRhoOut[0]);
					for (unsigned int q=0; q<numQuadPoints; ++q)
					{
						excQuads[q][iSubCell]=exchValRhoOut[q]+corrValRhoOut[q];
						vxcRhoOutSpin0Quads[q][iSubCell]= exchPotValRhoOut[2*q]+corrPotValRhoOut[2*q];
						vxcRhoOutSpin1Quads[q][iSubCell]= exchPotValRhoOut[2*q+1]+corrPotValRhoOut[2*q+1];

					}
				}

        for (unsigned int q=0; q<numQuadPoints; ++q)
        {
          if(dftParameters::nonLinearCoreCorrection == true)
          {
            const std::vector<double> & temp1=gradRhoCoreValues.find(subCellId)->second;
            for (unsigned int q=0; q<numQuadPoints; ++q)
              for (unsigned int idim=0; idim<C_DIM; idim++)
                  gradRhoCoreQuads[q][idim][iSubCell] = temp1[3*q+idim]/2.0;

            if(dftParameters::xcFamilyType=="GGA")
            {
              const std::vector<double> & temp2=hessianRhoCoreValues.find(subCellId)->second;
              for (unsigned int q=0; q<numQuadPoints; ++q)
                for(unsigned int idim = 0; idim < C_DIM; ++idim)
                  for(unsigned int jdim = 0; jdim < C_DIM; ++jdim)
                    hessianRhoCoreQuads[q][idim][jdim][iSubCell] = temp2[9*q + 3*idim + jdim]/2.0;
            }
          }
        }
        
			}//subcell loop

      Tensor<2,C_DIM,VectorizedArray<double> > EQuadSum=zeroTensor4;
      for (unsigned int q=0; q<numQuadPoints; ++q)
      {
        Tensor<2,C_DIM,VectorizedArray<double> > E=eshelbyTensorSP::getELocXcEshelbyTensor
          (rhoXCQuadsVect[q],
           gradRhoSpin0QuadsVect[q],
           gradRhoSpin1QuadsVect[q],
           excQuads[q],
           derExchCorrEnergyWithGradRhoOutSpin0Quads[q],
           derExchCorrEnergyWithGradRhoOutSpin1Quads[q]);

        EQuadSum+=E*forceEval.JxW(q);
      }//quad point loop

      if (isPseudopotential)
      {
        if (dftParameters::nonLinearCoreCorrection)
          addENonlinearCoreCorrectionStressContributionSpinPolarized(forceEval,
              matrixFreeData,
              cell,
              vxcRhoOutSpin0Quads,
              vxcRhoOutSpin1Quads,
              derExchCorrEnergyWithGradRhoOutSpin0Quads,
              derExchCorrEnergyWithGradRhoOutSpin1Quads,
              gradRhoCoreAtoms,
              hessianRhoCoreAtoms,
              dftParameters::xcFamilyType=="GGA");  
      }

      for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
        for (unsigned int idim=0; idim<C_DIM; ++idim)
          for (unsigned int jdim=0; jdim<C_DIM; ++jdim)
          {
            d_stress[idim][jdim]+=EQuadSum[idim][jdim][iSubCell];
          }   
		}//macrocell loop

    ////Add electrostatic configurational stress contribution////////////////
    computeStressEEshelbyEElectroPhiTot
      (matrixFreeDataElectro,
       phiTotDofHandlerIndexElectro,
       smearedChargeQuadratureId,
       lpspQuadratureIdElectro,
       phiTotRhoOutElectro,
       rhoOutValuesElectro,
       rhoOutValuesElectroLpsp,         
       gradRhoOutValuesElectro,
       gradRhoOutValuesElectroLpsp,
       pseudoVLocElectro,
       pseudoVLocAtomsElectro,
       vselfBinsManagerElectro);
	}
}
