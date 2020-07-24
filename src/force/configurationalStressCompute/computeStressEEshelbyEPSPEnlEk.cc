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
#ifdef USE_COMPLEX
//compute configurational stress contribution from all terms except the nuclear self energy
	template<unsigned int FEOrder>
void forceClass<FEOrder>::computeStressEEshelbyEPSPEnlEk(const MatrixFree<3,double> & matrixFreeData,
		const unsigned int eigenDofHandlerIndex,
		const unsigned int phiTotDofHandlerIndex,
    const unsigned int smearedChargeQuadratureId,
             const unsigned int lpspQuadratureId,
      const unsigned int lpspQuadratureIdElectro,          
		const distributedCPUVec<double> & phiTotRhoOut,
		const std::map<dealii::CellId, std::vector<double> > & pseudoVLoc,
		const vselfBinsManager<FEOrder> & vselfBinsManagerEigen,
		const MatrixFree<3,double> & matrixFreeDataElectro,
		const unsigned int phiTotDofHandlerIndexElectro,
		const distributedCPUVec<double> & phiTotRhoOutElectro,
    const std::map<dealii::CellId, std::vector<double> > & rhoOutValues,
    const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValues,
    const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValuesLpsp,         
    const std::map<dealii::CellId, std::vector<double> > & rhoOutValuesElectro,
    const std::map<dealii::CellId, std::vector<double> > & rhoOutValuesElectroLpsp,         
    const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValuesElectro,
    const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValuesElectroLpsp,
		const std::map<dealii::CellId, std::vector<double> > & pseudoVLocElectro,
		const std::map<dealii::CellId, std::vector<double> > & gradPseudoVLocElectro,
		const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & gradPseudoVLocAtomsElectro,
		const vselfBinsManager<FEOrder> & vselfBinsManagerElectro)
{
	std::vector<std::vector<distributedCPUVec<double>>> eigenVectors((1+dftParameters::spinPolarized)*dftPtr->d_kPointWeights.size());
	for(unsigned int kPoint = 0; kPoint < (1+dftParameters::spinPolarized)*dftPtr->d_kPointWeights.size(); ++kPoint)
	{
		eigenVectors[kPoint].resize(dftPtr->d_numEigenValues);
		for(unsigned int i = 0; i < dftPtr->d_numEigenValues; ++i)
			eigenVectors[kPoint][i].reinit(dftPtr->d_tempEigenVec);

#ifdef USE_COMPLEX
		vectorTools::copyFlattenedDealiiVecToSingleCompVec
			(dftPtr->d_eigenVectorsFlattened[kPoint],
			 dftPtr->d_numEigenValues,
			 std::make_pair(0,dftPtr->d_numEigenValues),
			 dftPtr->localProc_dof_indicesReal,
			 dftPtr->localProc_dof_indicesImag,
			 eigenVectors[kPoint]);
#else
		vectorTools::copyFlattenedDealiiVecToSingleCompVec
			(dftPtr->d_eigenVectorsFlattened[kPoint],
			 dftPtr->d_numEigenValues,
			 std::make_pair(0,dftPtr->d_numEigenValues),
			 eigenVectors[kPoint]);
#endif
	}

	const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
	const unsigned int numberImageCharges = dftPtr->d_imageIds.size();
	const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
	const bool isPseudopotential = dftParameters::isPseudopotential;
	const unsigned int numVectorizedArrayElements=VectorizedArray<double>::n_array_elements;

	FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  forceEval(matrixFreeData,
			d_forceDofHandlerIndex,
			0);
	FEEvaluation<C_DIM,1,C_num1DQuadNLPSP<FEOrder>()*C_numCopies1DQuadNLPSP(),C_DIM>  forceEvalNLP(matrixFreeData,
			d_forceDofHandlerIndex,
			2);

	FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),2> psiEval(matrixFreeData,
			eigenDofHandlerIndex,
			0);
	FEEvaluation<C_DIM,FEOrder,C_num1DQuadNLPSP<FEOrder>()*C_numCopies1DQuadNLPSP(),2> psiEvalNLP(matrixFreeData,
			eigenDofHandlerIndex,
			2);

	FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiTotOutEval(matrixFreeData,
			phiTotDofHandlerIndex,
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

	std::vector<std::vector<double>> partialOccupancies(dftPtr->d_kPointWeights.size(),std::vector<double>(numEigenVectors,0.0));
	for(unsigned int kPoint = 0; kPoint < dftPtr->d_kPointWeights.size(); ++kPoint)
		for (unsigned int iWave=0; iWave<numEigenVectors;++iWave)
		{
			partialOccupancies[kPoint][iWave]
				=dftUtils::getPartialOccupancy(dftPtr->eigenValues[kPoint][iWave],
						dftPtr->fermiEnergy,
						C_kb,
						dftParameters::TVal);

		}

	VectorizedArray<double> phiExtFactor=make_vectorized_array(0.0);
	std::vector<std::vector<std::vector<dataTypes::number > > > projectorKetTimesPsiTimesVTimesPartOcc(numKPoints);
	if (isPseudopotential){
		phiExtFactor=make_vectorized_array(1.0);
		for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
			computeNonLocalProjectorKetTimesPsiTimesVFlattened
				(dftPtr->d_eigenVectorsFlattened[ikPoint],
				 numEigenVectors,
				 projectorKetTimesPsiTimesVTimesPartOcc[ikPoint],
				 ikPoint,
				 partialOccupancies[ikPoint],
				 true);
	}

	std::vector<VectorizedArray<double> > rhoQuads(numQuadPoints,make_vectorized_array(0.0));
	std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoQuads(numQuadPoints,zeroTensor3);
	std::vector<VectorizedArray<double> > excQuads(numQuadPoints,make_vectorized_array(0.0));
	std::vector<VectorizedArray<double> > pseudoVLocQuads(numQuadPoints,make_vectorized_array(0.0));
	std::vector<VectorizedArray<double> > vEffRhoOutQuads(numQuadPoints,make_vectorized_array(0.0));
	std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > derExchCorrEnergyWithGradRhoOutQuads(numQuadPoints,zeroTensor3);

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
					d_cellIdToNonlocalAtomIdsLocalCompactSupportMap[subCellId].begin(), d_cellIdToNonlocalAtomIdsLocalCompactSupportMap[subCellId].end(),
					std::inserter(s, s.begin()));
			mergedSet=s;
		}
		macroIdToNonlocalAtomsSetMap[cell]=std::vector<unsigned int>(mergedSet.begin(),mergedSet.end());
	}


	for (unsigned int cell=0; cell<matrixFreeData.n_macro_cells(); ++cell)
	{
		forceEval.reinit(cell);
		psiEval.reinit(cell);

		if (isPseudopotential && dftParameters::useHigherQuadNLP)
		{
			forceEvalNLP.reinit(cell);
			psiEvalNLP.reinit(cell);
		}

		if (d_isElectrostaticsMeshSubdivided || dftParameters::nonSelfConsistentForce)
		{
			phiTotOutEval.reinit(cell);
			phiTotOutEval.read_dof_values_plain(phiTotRhoOut);
			phiTotOutEval.evaluate(true,false);
		}

		std::fill(rhoQuads.begin(),rhoQuads.end(),make_vectorized_array(0.0));
		std::fill(gradRhoQuads.begin(),gradRhoQuads.end(),zeroTensor3);
		std::fill(excQuads.begin(),excQuads.end(),make_vectorized_array(0.0));
		std::fill(pseudoVLocQuads.begin(),pseudoVLocQuads.end(),make_vectorized_array(0.0));
		std::fill(vEffRhoOutQuads.begin(),vEffRhoOutQuads.end(),make_vectorized_array(0.0));
		std::fill(derExchCorrEnergyWithGradRhoOutQuads.begin(),derExchCorrEnergyWithGradRhoOutQuads.end(),zeroTensor3);

		if (dftParameters::nonSelfConsistentForce)
			for (unsigned int q=0; q<numQuadPoints; ++q)
			{
				vEffRhoOutQuads[q]=phiTotOutEval.get_value(q);
			}
		//allocate storage for vector of quadPoints, nonlocal atom id, pseudo wave, k point
		//FIXME: flatten nonlocal atomid id and pseudo wave and k point
		std::vector<std::vector<std::vector<std::vector<Tensor<1,2,VectorizedArray<double> > > > > >ZetaDeltaVQuads;
		std::vector<std::vector<std::vector<std::vector<Tensor<1,2, Tensor<2,C_DIM,VectorizedArray<double> > > > > > >gradZetalmDeltaVlDyadicDistImageAtomsQuads;
		if(isPseudopotential)
		{
			ZetaDeltaVQuads.resize(numQuadPointsNLP);
			gradZetalmDeltaVlDyadicDistImageAtomsQuads.resize(numQuadPointsNLP);
			for (unsigned int q=0; q<numQuadPointsNLP; ++q)
			{
				ZetaDeltaVQuads[q].resize(d_nonLocalPSP_ZetalmDeltaVl.size());
				gradZetalmDeltaVlDyadicDistImageAtomsQuads[q].resize(d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint.size());
				for (unsigned int i=0; i < d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint.size(); ++i)
				{
					const int numberPseudoWaveFunctions = d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[i].size();
					ZetaDeltaVQuads[q][i].resize(numberPseudoWaveFunctions);
					gradZetalmDeltaVlDyadicDistImageAtomsQuads[q][i].resize(numberPseudoWaveFunctions);
					for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
					{
						ZetaDeltaVQuads[q][i][iPseudoWave].resize(numKPoints,zeroTensor1);
						gradZetalmDeltaVlDyadicDistImageAtomsQuads[q][i][iPseudoWave].resize(numKPoints,zeroTensor5);
					}
				}
			}
		}

		const unsigned int numSubCells=matrixFreeData.n_components_filled(cell);
		//For LDA
		std::vector<double> exchValRhoOut(numQuadPoints);
		std::vector<double> corrValRhoOut(numQuadPoints);
		std::vector<double> exchPotValRhoOut(numQuadPoints);
		std::vector<double> corrPotValRhoOut(numQuadPoints);
		//
		//For GGA
		std::vector<double> sigmaValRhoOut(numQuadPoints);
		std::vector<double> derExchEnergyWithDensityValRhoOut(numQuadPoints), derCorrEnergyWithDensityValRhoOut(numQuadPoints), derExchEnergyWithSigmaRhoOut(numQuadPoints),derCorrEnergyWithSigmaRhoOut(numQuadPoints);
		std::vector<Tensor<1,C_DIM,double > > gradRhoOut(numQuadPoints);
		//
		for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
		{
			subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell);
			dealii::CellId subCellId=subCellPtr->id();
			if(dftParameters::xc_id == 4)
			{
				for (unsigned int q = 0; q < numQuadPoints; ++q)
				{
					gradRhoOut[q][0] = ((*dftPtr->gradRhoOutValues)[subCellId][3*q + 0]);
					gradRhoOut[q][1] = ((*dftPtr->gradRhoOutValues)[subCellId][3*q + 1]);
					gradRhoOut[q][2] = ((*dftPtr->gradRhoOutValues)[subCellId][3*q + 2]);
					sigmaValRhoOut[q] = gradRhoOut[q].norm_square();

				}
				xc_gga_exc_vxc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoOutValues)[subCellId][0]),&sigmaValRhoOut[0],&exchValRhoOut[0],&derExchEnergyWithDensityValRhoOut[0],&derExchEnergyWithSigmaRhoOut[0]);
				xc_gga_exc_vxc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoOutValues)[subCellId][0]),&sigmaValRhoOut[0],&corrValRhoOut[0],&derCorrEnergyWithDensityValRhoOut[0],&derCorrEnergyWithSigmaRhoOut[0]);
				for (unsigned int q=0; q<numQuadPoints; ++q)
				{
					excQuads[q][iSubCell]=exchValRhoOut[q]+corrValRhoOut[q];
					const double temp = derExchEnergyWithSigmaRhoOut[q]+derCorrEnergyWithSigmaRhoOut[q];
					vEffRhoOutQuads[q][iSubCell]+= derExchEnergyWithDensityValRhoOut[q]+derCorrEnergyWithDensityValRhoOut[q];
					for (unsigned int idim=0; idim<C_DIM; idim++)
					{
						derExchCorrEnergyWithGradRhoOutQuads[q][idim][iSubCell]=2.0*(derExchEnergyWithSigmaRhoOut[q]+derCorrEnergyWithSigmaRhoOut[q])*gradRhoOut[q][idim];
					}
				}

			}
			else
			{
				xc_lda_exc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoOutValues)[subCellId][0]),&exchValRhoOut[0]);
				xc_lda_exc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoOutValues)[subCellId][0]),&corrValRhoOut[0]);
				xc_lda_vxc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoOutValues)[subCellId][0]),&exchPotValRhoOut[0]);
				xc_lda_vxc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoOutValues)[subCellId][0]),&corrPotValRhoOut[0]);
				for (unsigned int q=0; q<numQuadPoints; ++q)
				{
					excQuads[q][iSubCell]=exchValRhoOut[q]+corrValRhoOut[q];
					vEffRhoOutQuads[q][iSubCell]+= exchPotValRhoOut[q]+corrPotValRhoOut[q];
				}
			}

			for (unsigned int q=0; q<numQuadPoints; ++q)
			{
				rhoQuads[q][iSubCell]=(*dftPtr->rhoOutValues)[subCellId][q];
			}
		}

		std::vector<Tensor<1,2,VectorizedArray<double> > > psiQuads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor1);
		std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > > gradPsiQuads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor2);

		for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
			for (unsigned int iEigenVec=0; iEigenVec<numEigenVectors; ++iEigenVec)
			{
				psiEval.read_dof_values_plain(eigenVectors[ikPoint][iEigenVec]);
				psiEval.evaluate(true,true);

				for (unsigned int q=0; q<numQuadPoints; ++q)
				{
					const unsigned int id=q*numEigenVectors*numKPoints+numEigenVectors*ikPoint+iEigenVec;
					psiQuads[id]=psiEval.get_value(q);
					gradPsiQuads[id]=psiEval.get_gradient(q);

					const double partOcc =dftUtils::getPartialOccupancy(dftPtr->eigenValues[ikPoint][iEigenVec],
							dftPtr->fermiEnergy,
							C_kb,
							dftParameters::TVal);
					const VectorizedArray<double> factor=make_vectorized_array(2.0*dftPtr->d_kPointWeights[ikPoint]*partOcc);
					gradRhoQuads[q]+=factor*internalforce::computeGradRhoContribution(psiQuads[id],gradPsiQuads[id]);

				}//quad point loop
			} //eigenvector loop

		//accumulate gradRho quad point contribution from all pools
		for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
			for (unsigned int q=0; q<numQuadPoints; ++q)
				for (unsigned int idim=0; idim<C_DIM; idim++)
					gradRhoQuads[q][idim][iSubCell]=
						Utilities::MPI::sum(gradRhoQuads[q][idim][iSubCell],dftPtr->interpoolcomm);

		std::vector<Tensor<1,2,VectorizedArray<double> > > psiQuadsNLP;
		if (isPseudopotential && dftParameters::useHigherQuadNLP)
		{
			psiQuadsNLP.resize(numQuadPointsNLP*numEigenVectors*numKPoints,zeroTensor1);
			for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
				for (unsigned int iEigenVec=0; iEigenVec<numEigenVectors; ++iEigenVec)
				{
					psiEvalNLP.read_dof_values_plain(eigenVectors[ikPoint][iEigenVec]);
					psiEvalNLP.evaluate(true,false);

					for (unsigned int q=0; q<numQuadPointsNLP; ++q)
					{
						const unsigned int id=q*numEigenVectors*numKPoints+numEigenVectors*ikPoint+iEigenVec;
						psiQuadsNLP[id]=psiEvalNLP.get_value(q);
					}//quad point loop
				} //eigenvector loop
		}

		if(isPseudopotential)
		{
			for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
			{
				subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell);
				dealii::CellId subCellId=subCellPtr->id();
				for (unsigned int q=0; q<numQuadPoints; ++q)
				{
					pseudoVLocQuads[q][iSubCell]=pseudoVLoc.find(subCellId)->second[q];
				}

				for (unsigned int q=0; q<numQuadPointsNLP; ++q)
				{
					for (unsigned int i=0; i < d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint.size(); ++i)
					{
						const int numberPseudoWaveFunctions = d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[i].size();
						for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
						{
							if (d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[i][iPseudoWave].find(subCellId)!=d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[i][iPseudoWave].end())
							{
								for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
								{
									ZetaDeltaVQuads[q][i][iPseudoWave][ikPoint][0][iSubCell]=d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*2+q*2+0];
									ZetaDeltaVQuads[q][i][iPseudoWave][ikPoint][1][iSubCell]=d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*2+q*2+1];
									for (unsigned int idim=0; idim<C_DIM; idim++)
									{
										for (unsigned int jdim=0; jdim<C_DIM; jdim++)
										{
											gradZetalmDeltaVlDyadicDistImageAtomsQuads[q][i][iPseudoWave][ikPoint][0][idim][jdim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*C_DIM*C_DIM*2+q*C_DIM*C_DIM*2+idim*C_DIM*2+jdim*2+0];
											gradZetalmDeltaVlDyadicDistImageAtomsQuads[q][i][iPseudoWave][ikPoint][1][idim][jdim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*C_DIM*C_DIM*2+q*C_DIM*C_DIM*2+idim*C_DIM*2+jdim*2+1];
										}
									}
								}
							}//non-trivial cellId check
						}//iPseudoWave loop
					}//i loop
				}//q loop
			}//subcell loop

		}//is pseudopotential check

		Tensor<2,C_DIM,VectorizedArray<double> > EQuadSum=zeroTensor4;
		Tensor<2,C_DIM,VectorizedArray<double> > EKPointsQuadSum=zeroTensor4;
		for (unsigned int q=0; q<numQuadPoints; ++q)
		{

			const VectorizedArray<double> phiTot_q =d_isElectrostaticsMeshSubdivided?
				phiTotOutEval.get_value(q)
				:make_vectorized_array(0.0);
			const VectorizedArray<double> phiExt_q =make_vectorized_array(0.0);
			Point< 3, VectorizedArray<double> > quadPoint_q;
			if (d_isElectrostaticsMeshSubdivided && false)
				quadPoint_q=phiTotOutEval.quadrature_point(q);

			Tensor<2,C_DIM,VectorizedArray<double> > E=eshelbyTensor::getELocXcEshelbyTensor
				(rhoQuads[q],
				 gradRhoQuads[q],
				 excQuads[q],
				 derExchCorrEnergyWithGradRhoOutQuads[q]);

			if(d_isElectrostaticsMeshSubdivided && false)
			{
				VectorizedArray<double> val=scalar_product(gradRhoQuads[q]*phiTot_q,quadPoint_q);
				E[0][0]-=val;
				E[1][1]-=val;
				E[2][2]-=val;

				if (isPseudopotential)
				{
					val=scalar_product(gradRhoQuads[q]*(pseudoVLocQuads[q]),quadPoint_q);
					E[0][0]-=val;
					E[1][1]-=val;
					E[2][2]-=val;
				}
			}

			Tensor<2,C_DIM,VectorizedArray<double> > EKPoints=eshelbyTensor::getELocWfcEshelbyTensorPeriodicKPoints
				(psiQuads.begin()+q*numEigenVectors*numKPoints,
				 gradPsiQuads.begin()+q*numEigenVectors*numKPoints,
				 dftPtr->d_kPointCoordinates,
				 dftPtr->d_kPointWeights,
				 dftPtr->eigenValues,
				 dftPtr->fermiEnergy,
				 dftParameters::TVal);

			EKPoints+=eshelbyTensor::getEKStress
				(psiQuads.begin()+q*numEigenVectors*numKPoints,
				 gradPsiQuads.begin()+q*numEigenVectors*numKPoints,
				 dftPtr->d_kPointCoordinates,
				 dftPtr->d_kPointWeights,
				 dftPtr->eigenValues,
				 dftPtr->fermiEnergy,
				 dftParameters::TVal);

			if(isPseudopotential && !dftParameters::useHigherQuadNLP)
			{
				EKPoints+=eshelbyTensor::getEnlEshelbyTensorPeriodic(ZetaDeltaVQuads[q],
						projectorKetTimesPsiTimesVTimesPartOcc,
						psiQuads.begin()+q*numEigenVectors*numKPoints,
						dftPtr->d_kPointWeights,
						macroIdToNonlocalAtomsSetMap[cell],
						numEigenVectors);

				EKPoints+=eshelbyTensor::getEnlStress(gradZetalmDeltaVlDyadicDistImageAtomsQuads[q],
						projectorKetTimesPsiTimesVTimesPartOcc,
						psiQuads.begin()+q*numEigenVectors*numKPoints,
						dftPtr->d_kPointWeights,
						macroIdToNonlocalAtomsSetMap[cell],
						numEigenVectors);

			}//is pseudopotential check

			EQuadSum+=E*forceEval.JxW(q);
			EKPointsQuadSum+=EKPoints*forceEval.JxW(q);
		}//quad point loop

		if (isPseudopotential && dftParameters::useHigherQuadNLP)
			for (unsigned int q=0; q<numQuadPointsNLP; ++q)
			{
				Tensor<2,C_DIM,VectorizedArray<double> > EKPoints
					=eshelbyTensor::getEnlEshelbyTensorPeriodic(ZetaDeltaVQuads[q],
							projectorKetTimesPsiTimesVTimesPartOcc,
							psiQuadsNLP.begin()+q*numEigenVectors*numKPoints,
							dftPtr->d_kPointWeights,
							macroIdToNonlocalAtomsSetMap[cell],
							numEigenVectors);

				EKPoints+=eshelbyTensor::getEnlStress(gradZetalmDeltaVlDyadicDistImageAtomsQuads[q],
						projectorKetTimesPsiTimesVTimesPartOcc,
						psiQuadsNLP.begin()+q*numEigenVectors*numKPoints,
						dftPtr->d_kPointWeights,
						macroIdToNonlocalAtomsSetMap[cell],
						numEigenVectors);


				EKPointsQuadSum+=EKPoints*forceEvalNLP.JxW(q);

			}

		for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
			for (unsigned int idim=0; idim<C_DIM; ++idim)
				for (unsigned int jdim=0; jdim<C_DIM; ++jdim)
				{
					d_stress[idim][jdim]+=EQuadSum[idim][jdim][iSubCell];
					d_stressKPoints[idim][jdim]+=EKPointsQuadSum[idim][jdim][iSubCell];
				}
	}

	////Add electrostatic configurational force contribution////////////////
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
		 gradPseudoVLocAtomsElectro,
		 vselfBinsManagerElectro);
}

template<unsigned int FEOrder>
	void forceClass<FEOrder>::computeStressEEshelbyEElectroPhiTot
(const MatrixFree<3,double> & matrixFreeDataElectro,
 const unsigned int phiTotDofHandlerIndexElectro,
 const unsigned int smearedChargeQuadratureId,
          const unsigned int lpspQuadratureIdElectro,
 const distributedCPUVec<double> & phiTotRhoOutElectro,
 const std::map<dealii::CellId, std::vector<double> > & rhoOutValuesElectro,
 const std::map<dealii::CellId, std::vector<double> > & rhoOutValuesElectroLpsp,         
 const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValuesElectro,
 const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValuesElectroLpsp,
 const std::map<dealii::CellId, std::vector<double> > & pseudoVLocElectro,
 const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & gradPseudoVLocAtomsElectro,
 const vselfBinsManager<FEOrder> & vselfBinsManagerElectro)
{
	FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  forceEvalElectro(matrixFreeDataElectro,
			d_forceDofHandlerIndexElectro,
			0);

	FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiTotEvalElectro(matrixFreeDataElectro,
			phiTotDofHandlerIndexElectro,
			0);

	FEEvaluation<C_DIM,FEOrder,C_num1DQuadSmearedCharge<FEOrder>()*C_numCopies1DQuadSmearedCharge(),1>  phiTotEvalSmearedCharge(matrixFreeDataElectro,
			phiTotDofHandlerIndexElectro,
			smearedChargeQuadratureId);

	FEEvaluation<C_DIM,1,C_num1DQuadSmearedCharge<FEOrder>()*C_numCopies1DQuadSmearedCharge(),C_DIM>  forceEvalSmearedCharge(matrixFreeDataElectro,
			d_forceDofHandlerIndexElectro,
			smearedChargeQuadratureId);

	FEEvaluation<C_DIM,1,C_num1DQuadLPSP<FEOrder>()*C_numCopies1DQuadLPSP(),C_DIM>  forceEvalElectroLpsp(matrixFreeDataElectro,
			d_forceDofHandlerIndexElectro,
			lpspQuadratureIdElectro);  

	FEValues<C_DIM> feVselfValuesElectro (matrixFreeDataElectro.
			get_dof_handler(phiTotDofHandlerIndexElectro).get_fe(),
			matrixFreeDataElectro.get_quadrature(lpspQuadratureIdElectro),
			update_gradients | update_quadrature_points);

	const unsigned int numQuadPoints=forceEvalElectro.n_q_points;
  const unsigned int numQuadPointsSmearedb=forceEvalSmearedCharge.n_q_points;
  const unsigned int numQuadPointsLpsp=forceEvalElectroLpsp.n_q_points;

  if (gradRhoOutValuesElectroLpsp.size()!=0)
    AssertThrow(gradRhoOutValuesElectroLpsp.begin()->second.size() == 3*numQuadPointsLpsp,
            dealii::ExcMessage("DFT-FE Error: mismatch in quadrature rule usage in force computation.")); 
  
	DoFHandler<C_DIM>::active_cell_iterator subCellPtr;


	Tensor<1,C_DIM,VectorizedArray<double> > zeroTensor;
	for (unsigned int idim=0; idim<C_DIM; idim++)
	{
		zeroTensor[idim]=make_vectorized_array(0.0);
	}

	Tensor<2,C_DIM,VectorizedArray<double> > zeroTensor2;
	for (unsigned int idim=0; idim<C_DIM; idim++)
		for (unsigned int jdim=0; jdim<C_DIM; jdim++)
			zeroTensor2[idim][jdim]=make_vectorized_array(0.0);

	std::vector<VectorizedArray<double> > rhoQuadsElectro(numQuadPoints,make_vectorized_array(0.0));
  std::vector<VectorizedArray<double> > rhoQuadsElectroLpsp(numQuadPointsLpsp,make_vectorized_array(0.0));  
  std::vector<VectorizedArray<double> > smearedbQuads(numQuadPointsSmearedb,make_vectorized_array(0.0));
  std::vector< Tensor<1,C_DIM,VectorizedArray<double> >  > smearedGradbQuads(numQuadPointsSmearedb,zeroTensor);  
  std::vector< VectorizedArray<double> > phiTotSmearedChargeQuads(numQuadPointsSmearedb,make_vectorized_array(0.0));  
	std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoQuadsElectro(numQuadPoints,zeroTensor);
  std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoQuadsElectroLpsp(numQuadPointsLpsp,zeroTensor);
	std::vector<VectorizedArray<double> > pseudoVLocQuadsElectro(numQuadPointsLpsp,make_vectorized_array(0.0));
	for (unsigned int cell=0; cell<matrixFreeDataElectro.n_macro_cells(); ++cell)
	{
		forceEvalElectro.reinit(cell);
    forceEvalElectroLpsp.reinit(cell);

		phiTotEvalElectro.reinit(cell);
		phiTotEvalElectro.read_dof_values_plain(phiTotRhoOutElectro);
		phiTotEvalElectro.evaluate(true,true);

    if (dftParameters::smearedNuclearCharges)
    {
      forceEvalSmearedCharge.reinit(cell);
      phiTotEvalSmearedCharge.reinit(cell);
      phiTotEvalSmearedCharge.read_dof_values_plain(phiTotRhoOutElectro);
      phiTotEvalSmearedCharge.evaluate(true,false);        
    }

		std::fill(rhoQuadsElectro.begin(),rhoQuadsElectro.end(),make_vectorized_array(0.0));
    std::fill(rhoQuadsElectroLpsp.begin(),rhoQuadsElectroLpsp.end(),make_vectorized_array(0.0));    
		std::fill(gradRhoQuadsElectro.begin(),gradRhoQuadsElectro.end(),zeroTensor);
    std::fill(gradRhoQuadsElectroLpsp.begin(),gradRhoQuadsElectroLpsp.end(),zeroTensor);    
		std::fill(pseudoVLocQuadsElectro.begin(),pseudoVLocQuadsElectro.end(),make_vectorized_array(0.0));
    std::fill(smearedbQuads.begin(),smearedbQuads.end(),make_vectorized_array(0.0));
    std::fill(smearedGradbQuads.begin(),smearedGradbQuads.end(),zeroTensor);    
    std::fill(phiTotSmearedChargeQuads.begin(),phiTotSmearedChargeQuads.end(),make_vectorized_array(0.0));    

		const unsigned int numSubCells=matrixFreeDataElectro.n_components_filled(cell);

    double sum=0.0; 
		for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
		{
			subCellPtr= matrixFreeDataElectro.get_cell_iterator(cell,iSubCell);
			dealii::CellId subCellId=subCellPtr->id();
			for (unsigned int q=0; q<numQuadPoints; ++q)
				rhoQuadsElectro[q][iSubCell]=rhoOutValuesElectro.find(subCellId)->second[q];

			if(d_isElectrostaticsMeshSubdivided)
				for (unsigned int q=0; q<numQuadPoints; ++q)
				{
					gradRhoQuadsElectro[q][0][iSubCell]=gradRhoOutValuesElectro.find(subCellId)->second[C_DIM*q+0];
					gradRhoQuadsElectro[q][1][iSubCell]=gradRhoOutValuesElectro.find(subCellId)->second[C_DIM*q+1];
					gradRhoQuadsElectro[q][2][iSubCell]=gradRhoOutValuesElectro.find(subCellId)->second[C_DIM*q+2];
				}

			if(dftParameters::isPseudopotential)
      {
        const std::vector<double> & tempPseudoVal=pseudoVLocElectro.find(subCellId)->second;
        const std::vector<double> & tempLpspRhoVal=rhoOutValuesElectroLpsp.find(subCellId)->second;
        const std::vector<double> & tempLpspGradRhoVal=gradRhoOutValuesElectroLpsp.find(subCellId)->second;
				for (unsigned int q=0; q<numQuadPointsLpsp; ++q)
        {
          pseudoVLocQuadsElectro[q][iSubCell]=tempPseudoVal[q];
					rhoQuadsElectroLpsp[q][iSubCell]=tempLpspRhoVal[q];
				  gradRhoQuadsElectroLpsp[q][0][iSubCell]=tempLpspGradRhoVal[3*q+0];
					gradRhoQuadsElectroLpsp[q][1][iSubCell]=tempLpspGradRhoVal[3*q+1];
					gradRhoQuadsElectroLpsp[q][2][iSubCell]=tempLpspGradRhoVal[3*q+2];            
        }
      }

      if (dftParameters::smearedNuclearCharges)
      {
        const std::vector<double> & gradbQuadValuesCell= dftPtr->d_gradbQuadValuesAllAtoms.find(subCellId)->second;
        const std::vector<double> & bQuadValuesCell= dftPtr->d_bQuadValuesAllAtoms.find(subCellId)->second;
        for (unsigned int q=0; q<numQuadPointsSmearedb; ++q)
        {
          smearedbQuads[q][iSubCell]=bQuadValuesCell[q];
          sum+=bQuadValuesCell[q];
          smearedGradbQuads[q][0][iSubCell]=gradbQuadValuesCell[3*q+0];
          smearedGradbQuads[q][1][iSubCell]=gradbQuadValuesCell[3*q+1];
          smearedGradbQuads[q][2][iSubCell]=gradbQuadValuesCell[3*q+2];
        }            
      }
		}

		if (dftParameters::isPseudopotential)
		{

			addEPSPStressContribution(feVselfValuesElectro,
					forceEvalElectroLpsp,
					matrixFreeDataElectro,
					cell,
					rhoQuadsElectroLpsp,
					gradPseudoVLocAtomsElectro,
					vselfBinsManagerElectro,
					d_cellsVselfBallsClosestAtomIdDofHandlerElectro);

		}

		Tensor<2,C_DIM,VectorizedArray<double> > EQuadSum=zeroTensor2;
		for (unsigned int q=0; q<numQuadPoints; ++q)
		{
			VectorizedArray<double> phiTotElectro_q =phiTotEvalElectro.get_value(q);
			VectorizedArray<double> phiExtElectro_q =make_vectorized_array(0.0);
			Tensor<1,C_DIM,VectorizedArray<double> > gradPhiTotElectro_q =phiTotEvalElectro.get_gradient(q);

			Tensor<2,C_DIM,VectorizedArray<double> > E=eshelbyTensor::getEElectroEshelbyTensor
				(phiTotElectro_q,
				 gradPhiTotElectro_q,
				 rhoQuadsElectro[q]);

			Point< 3, VectorizedArray<double> > quadPoint_q;
			if (d_isElectrostaticsMeshSubdivided && false)
				quadPoint_q=phiTotEvalElectro.quadrature_point(q);


			if(d_isElectrostaticsMeshSubdivided && false)
			{
				VectorizedArray<double> val=scalar_product(gradRhoQuadsElectro[q]*phiTotElectro_q,quadPoint_q);
				E[0][0]+=val;
				E[1][1]+=val;
				E[2][2]+=val;
			}

			EQuadSum+=E*forceEvalElectro.JxW(q);
		}

		if(dftParameters::isPseudopotential)
      for (unsigned int q=0; q<numQuadPointsLpsp; ++q)
      {
			  VectorizedArray<double> phiExtElectro_q =make_vectorized_array(0.0);
        Tensor<2,C_DIM,VectorizedArray<double> > E=zeroTensor2;

				E+=eshelbyTensor::getELocPspEshelbyTensor
					(rhoQuadsElectroLpsp[q],
					 pseudoVLocQuadsElectro[q],
					 phiExtElectro_q);

        Point< 3, VectorizedArray<double> > quadPoint_q;
        if (d_isElectrostaticsMeshSubdivided && false)
          quadPoint_q=forceEvalElectroLpsp.quadrature_point(q);

        if(d_isElectrostaticsMeshSubdivided && false)
        {
					VectorizedArray<double> val=scalar_product(gradRhoQuadsElectroLpsp[q]*(pseudoVLocQuadsElectro[q]),quadPoint_q);
					E[0][0]+=val;
					E[1][1]+=val;
					E[2][2]+=val;
        }

        EQuadSum+=E*forceEvalElectroLpsp.JxW(q);
      }    

		for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
			for (unsigned int idim=0; idim<C_DIM; ++idim)
				for (unsigned int jdim=0; jdim<C_DIM; ++jdim)
					d_stress[idim][jdim]+=EQuadSum[idim][jdim][iSubCell];

    VectorizedArray<double>  identityTensorContributionSmearedCharge=make_vectorized_array(0.0);
    if (dftParameters::smearedNuclearCharges && std::abs(sum)>1e-9)
    {
      for (unsigned int q=0; q<numQuadPointsSmearedb; ++q)
      {
        phiTotSmearedChargeQuads[q]=phiTotEvalSmearedCharge.get_value(q);
        identityTensorContributionSmearedCharge+=phiTotSmearedChargeQuads[q]*smearedbQuads[q]*forceEvalSmearedCharge.JxW(q);
      }

			addEPhiTotSmearedStressContribution(forceEvalSmearedCharge,
					matrixFreeDataElectro,
					cell,
					phiTotSmearedChargeQuads,
          dftPtr->d_bQuadAtomIdsAllAtomsImages,
				  smearedGradbQuads);
      
      for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)  
      {
         d_stress[0][0]+=identityTensorContributionSmearedCharge[iSubCell];
         d_stress[1][1]+=identityTensorContributionSmearedCharge[iSubCell];
         d_stress[2][2]+=identityTensorContributionSmearedCharge[iSubCell];
      }  
    }

	}//cell loop
}
#endif
