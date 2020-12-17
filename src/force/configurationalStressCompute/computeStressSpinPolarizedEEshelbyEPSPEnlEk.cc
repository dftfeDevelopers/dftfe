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

	FEEvaluation<C_DIM,1,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),C_DIM>  forceEval(matrixFreeData,
			d_forceDofHandlerIndex,
			0);
	FEEvaluation<C_DIM,1,C_num1DQuadNLPSP<FEOrder>()*C_numCopies1DQuadNLPSP(),C_DIM>  forceEvalNLP(matrixFreeData,
			d_forceDofHandlerIndex,
			dftPtr->d_nlpspQuadratureId);

#ifdef USE_COMPLEX
	FEEvaluation<C_DIM,FEOrder,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),2> psiEvalSpin0(matrixFreeData,
			eigenDofHandlerIndex,
			0);
	FEEvaluation<C_DIM,FEOrder,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),2> psiEvalSpin1(matrixFreeData,
			eigenDofHandlerIndex,
			0);
	FEEvaluation<C_DIM,FEOrder,C_num1DQuadNLPSP<FEOrder>()*C_numCopies1DQuadNLPSP(),2> psiEvalSpin0NLP(matrixFreeData,
			eigenDofHandlerIndex,
			dftPtr->d_nlpspQuadratureId);
	FEEvaluation<C_DIM,FEOrder,C_num1DQuadNLPSP<FEOrder>()*C_numCopies1DQuadNLPSP(),2> psiEvalSpin1NLP(matrixFreeData,
			eigenDofHandlerIndex,
			dftPtr->d_nlpspQuadratureId);
#else
	FEEvaluation<C_DIM,FEOrder,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),1> psiEvalSpin0(matrixFreeData,
			eigenDofHandlerIndex,
			0);
	FEEvaluation<C_DIM,FEOrder,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),1> psiEvalSpin1(matrixFreeData,
			eigenDofHandlerIndex,
			0);
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

	std::vector<std::vector<double>> partialOccupanciesSpin0(dftPtr->d_kPointWeights.size(),
			std::vector<double>(numEigenVectors,0.0));
	std::vector<std::vector<double>> partialOccupanciesSpin1(dftPtr->d_kPointWeights.size(),
			std::vector<double>(numEigenVectors,0.0));
	for(unsigned int kPoint = 0; kPoint < dftPtr->d_kPointWeights.size(); ++kPoint)
		for (unsigned int iWave=0; iWave<numEigenVectors;++iWave)
		{

			partialOccupanciesSpin0[kPoint][iWave]
				=dftUtils::getPartialOccupancy(dftPtr->eigenValues[kPoint][iWave],
						dftPtr->fermiEnergy,
						C_kb,
						dftParameters::TVal);
			partialOccupanciesSpin1[kPoint][iWave]
				=dftUtils::getPartialOccupancy(dftPtr->eigenValues[kPoint][numEigenVectors+iWave],
						dftPtr->fermiEnergy,
						C_kb,
						dftParameters::TVal);
			if(dftParameters::constraintMagnetization)
			{
				partialOccupanciesSpin0[kPoint][iWave] = 1.0;
				partialOccupanciesSpin1[kPoint][iWave] = 1.0 ;
				if (dftPtr->eigenValues[kPoint][iWave]> dftPtr->fermiEnergyUp)
					partialOccupanciesSpin0[kPoint][iWave] = 0.0 ;
				if (dftPtr->eigenValues[kPoint][numEigenVectors+iWave] > dftPtr->fermiEnergyDown)
					partialOccupanciesSpin1[kPoint][iWave] = 0.0 ;
			}
		}

	std::vector<std::vector<std::vector<dataTypes::number > > > projectorKetTimesPsiSpin0TimesVTimesPartOcc(numKPoints);
	std::vector<std::vector<std::vector<dataTypes::number > > > projectorKetTimesPsiSpin1TimesVTimesPartOcc(numKPoints);
	if (isPseudopotential)
	{
		for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
		{
			computeNonLocalProjectorKetTimesPsiTimesVFlattened(dftPtr->d_eigenVectorsFlattened[2*ikPoint],
					numEigenVectors,
					projectorKetTimesPsiSpin0TimesVTimesPartOcc[ikPoint],
					ikPoint,
					partialOccupanciesSpin0[ikPoint],
					true
          );
		}
		for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
		{
			computeNonLocalProjectorKetTimesPsiTimesVFlattened(dftPtr->d_eigenVectorsFlattened[2*ikPoint+1],
					numEigenVectors,
					projectorKetTimesPsiSpin1TimesVTimesPartOcc[ikPoint],
					ikPoint,
					partialOccupanciesSpin1[ikPoint],
					true
          );
		}
	}

	std::vector<VectorizedArray<double> > rhoQuads(numQuadPoints,make_vectorized_array(0.0));
	std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoSpin0Quads(numQuadPoints,zeroTensor3);
	std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoSpin1Quads(numQuadPoints,zeroTensor3);
	std::vector<VectorizedArray<double> > excQuads(numQuadPoints,make_vectorized_array(0.0));
	std::vector<VectorizedArray<double> > vEffRhoOutSpin0Quads(numQuadPoints,make_vectorized_array(0.0));
	std::vector<VectorizedArray<double> > vEffRhoOutSpin1Quads(numQuadPoints,make_vectorized_array(0.0));
	std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > derExchCorrEnergyWithGradRhoOutSpin0Quads(numQuadPoints,zeroTensor3);
	std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > derExchCorrEnergyWithGradRhoOutSpin1Quads(numQuadPoints,zeroTensor3);

	for (unsigned int cell=0; cell<matrixFreeData.n_macro_cells(); ++cell)
	{
		forceEval.reinit(cell);
		psiEvalSpin0.reinit(cell);
		psiEvalSpin1.reinit(cell);

		if (isPseudopotential)
		{
			forceEvalNLP.reinit(cell);
			psiEvalSpin0NLP.reinit(cell);
			psiEvalSpin1NLP.reinit(cell);
		}


		std::fill(rhoQuads.begin(),rhoQuads.end(),make_vectorized_array(0.0));
		std::fill(gradRhoSpin0Quads.begin(),gradRhoSpin0Quads.end(),zeroTensor3);
		std::fill(gradRhoSpin1Quads.begin(),gradRhoSpin1Quads.end(),zeroTensor3);
		std::fill(excQuads.begin(),excQuads.end(),make_vectorized_array(0.0));
		std::fill(vEffRhoOutSpin0Quads.begin(),vEffRhoOutSpin0Quads.end(),make_vectorized_array(0.0));
		std::fill(vEffRhoOutSpin1Quads.begin(),vEffRhoOutSpin1Quads.end(),make_vectorized_array(0.0));
		std::fill(derExchCorrEnergyWithGradRhoOutSpin0Quads.begin(),derExchCorrEnergyWithGradRhoOutSpin0Quads.end(),zeroTensor3);
		std::fill(derExchCorrEnergyWithGradRhoOutSpin1Quads.begin(),derExchCorrEnergyWithGradRhoOutSpin1Quads.end(),zeroTensor3);


		//allocate storage for vector of quadPoints, nonlocal atom id, pseudo wave, k point
		//FIXME: flatten nonlocal atomid id and pseudo wave and k point
#ifdef USE_COMPLEX
		std::vector<std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > >zetalmDeltaVlProductDistImageAtomsQuads;
#else  
		std::vector<std::vector<std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > > > zetalmDeltaVlProductDistImageAtomsQuads;
#endif    
		if(isPseudopotential)
		{
			zetalmDeltaVlProductDistImageAtomsQuads.resize(numQuadPointsNLP);
			for (unsigned int q=0; q<numQuadPointsNLP; ++q)
			{
				zetalmDeltaVlProductDistImageAtomsQuads[q].resize(dftPtr->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint.size());
				for (unsigned int i=0; i < dftPtr->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint.size(); ++i)
				{
					const int numberPseudoWaveFunctions = dftPtr->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint[i].size();
					zetalmDeltaVlProductDistImageAtomsQuads[q][i].resize(numberPseudoWaveFunctions);
					for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
					{
#ifdef USE_COMPLEX              
						zetalmDeltaVlProductDistImageAtomsQuads[q][i][iPseudoWave].resize(numKPoints,zeroTensor2);
#endif            
					}
				}
			}
		}

		const unsigned int numSubCells=matrixFreeData.n_components_filled(cell);
		//For LDA
		std::vector<double> exchValRhoOut(numQuadPoints);
		std::vector<double> corrValRhoOut(numQuadPoints);
		std::vector<double> exchPotValRhoOut(2*numQuadPoints);
		std::vector<double> corrPotValRhoOut(2*numQuadPoints);
		//
		//For GGA
		std::vector<double> sigmaValRhoOut(3*numQuadPoints);
		std::vector<double> derExchEnergyWithDensityValRhoOut(2*numQuadPoints), derCorrEnergyWithDensityValRhoOut(2*numQuadPoints), derExchEnergyWithSigmaRhoOut(3*numQuadPoints),derCorrEnergyWithSigmaRhoOut(3*numQuadPoints);
		std::vector<Tensor<1,C_DIM,double > > gradRhoOutSpin0(numQuadPoints);
		std::vector<Tensor<1,C_DIM,double > > gradRhoOutSpin1(numQuadPoints);
		//
		for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
		{
			subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell);
			dealii::CellId subCellId=subCellPtr->id();
			if(dftParameters::xcFamilyType=="GGA")
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
				}
				xc_gga_exc_vxc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoOutValuesSpinPolarized)[subCellId][0]),&sigmaValRhoOut[0],&exchValRhoOut[0],&derExchEnergyWithDensityValRhoOut[0],&derExchEnergyWithSigmaRhoOut[0]);
				xc_gga_exc_vxc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoOutValuesSpinPolarized)[subCellId][0]),&sigmaValRhoOut[0],&corrValRhoOut[0],&derCorrEnergyWithDensityValRhoOut[0],&derCorrEnergyWithSigmaRhoOut[0]);
				for (unsigned int q=0; q<numQuadPoints; ++q)
				{
					excQuads[q][iSubCell]=exchValRhoOut[q]+corrValRhoOut[q];
					vEffRhoOutSpin0Quads[q][iSubCell]+= derExchEnergyWithDensityValRhoOut[2*q]+derCorrEnergyWithDensityValRhoOut[2*q];
					vEffRhoOutSpin1Quads[q][iSubCell]+= derExchEnergyWithDensityValRhoOut[2*q+1]+derCorrEnergyWithDensityValRhoOut[2*q+1];
					for (unsigned int idim=0; idim<C_DIM; idim++)
					{
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
				for (unsigned int q=0; q<numQuadPoints; ++q)
				{
					excQuads[q][iSubCell]=exchValRhoOut[q]+corrValRhoOut[q];
					vEffRhoOutSpin0Quads[q][iSubCell]+= exchPotValRhoOut[2*q]+corrPotValRhoOut[2*q];
					vEffRhoOutSpin1Quads[q][iSubCell]+= exchPotValRhoOut[2*q+1]+corrPotValRhoOut[2*q+1];

				}
			}

			for (unsigned int q=0; q<numQuadPoints; ++q)
			{
				rhoQuads[q][iSubCell]=(*dftPtr->rhoOutValues)[subCellId][q];

        if(dftParameters::xcFamilyType=="GGA")
          for (unsigned int idim=0; idim<C_DIM; idim++)
          {
            gradRhoSpin0Quads[q][idim][iSubCell]=(*dftPtr->gradRhoOutValuesSpinPolarized)[subCellId][6*q+idim];
            gradRhoSpin1Quads[q][idim][iSubCell]=(*dftPtr->gradRhoOutValuesSpinPolarized)[subCellId][6*q+3+idim];
          }        
			}
		}

#ifdef USE_COMPLEX  
		std::vector<Tensor<1,2,VectorizedArray<double> > > psiSpin0Quads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor1);
		std::vector<Tensor<1,2,VectorizedArray<double> > > psiSpin1Quads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor1);
		std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > > gradPsiSpin0Quads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor2);
		std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > > gradPsiSpin1Quads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor2);
#else
		std::vector<VectorizedArray<double> > psiSpin0Quads(numQuadPoints*numEigenVectors*numKPoints,make_vectorized_array(0.0));
		std::vector<VectorizedArray<double> > psiSpin1Quads(numQuadPoints*numEigenVectors*numKPoints,make_vectorized_array(0.0));
		std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPsiSpin0Quads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor3);
		std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPsiSpin1Quads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor3);
#endif    

		for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
			for (unsigned int iEigenVec=0; iEigenVec<numEigenVectors; ++iEigenVec)
			{
				psiEvalSpin0.read_dof_values_plain(eigenVectors[2*ikPoint][iEigenVec]);
				psiEvalSpin0.evaluate(true,true);
				psiEvalSpin1.read_dof_values_plain(eigenVectors[2*ikPoint+1][iEigenVec]);
				psiEvalSpin1.evaluate(true,true);

				for (unsigned int q=0; q<numQuadPoints; ++q)
				{
					const int id=q*numEigenVectors*numKPoints+numEigenVectors*ikPoint+iEigenVec;
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
		std::vector<VectorizedArray<double> > psiSpin0QuadsNLP;
		std::vector<VectorizedArray<double> > psiSpin1QuadsNLP;
		std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPsiSpin0QuadsNLP;
		std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPsiSpin1QuadsNLP; 
#endif    
		if (isPseudopotential)
		{
#ifdef USE_COMPLEX        
			psiSpin0QuadsNLP.resize(numQuadPointsNLP*numEigenVectors*numKPoints,zeroTensor1);
			psiSpin1QuadsNLP.resize(numQuadPointsNLP*numEigenVectors*numKPoints,zeroTensor1);
			gradPsiSpin0QuadsNLP.resize(numQuadPointsNLP*numEigenVectors*numKPoints,zeroTensor2);
			gradPsiSpin1QuadsNLP.resize(numQuadPointsNLP*numEigenVectors*numKPoints,zeroTensor2);    
#else
			psiSpin0QuadsNLP.resize(numQuadPointsNLP*numEigenVectors*numKPoints,make_vectorized_array(0.0));
			psiSpin1QuadsNLP.resize(numQuadPointsNLP*numEigenVectors*numKPoints,make_vectorized_array(0.0));
			gradPsiSpin0QuadsNLP.resize(numQuadPointsNLP*numEigenVectors*numKPoints,zeroTensor3);
			gradPsiSpin1QuadsNLP.resize(numQuadPointsNLP*numEigenVectors*numKPoints,zeroTensor3);
#endif
			for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
				for (unsigned int iEigenVec=0; iEigenVec<numEigenVectors; ++iEigenVec)
				{
					psiEvalSpin0NLP.read_dof_values_plain(eigenVectors[2*ikPoint][iEigenVec]);
					psiEvalSpin0NLP.evaluate(true,true);

					psiEvalSpin1NLP.read_dof_values_plain(eigenVectors[2*ikPoint+1][iEigenVec]);
					psiEvalSpin1NLP.evaluate(true,true);

					for (unsigned int q=0; q<numQuadPointsNLP; ++q)
					{
						const int id=q*numEigenVectors*numKPoints+numEigenVectors*ikPoint+iEigenVec;
						psiSpin0QuadsNLP[id]=psiEvalSpin0NLP.get_value(q);
						psiSpin1QuadsNLP[id]=psiEvalSpin1NLP.get_value(q);
						gradPsiSpin0QuadsNLP[id]=psiEvalSpin0NLP.get_gradient(q);
						gradPsiSpin1QuadsNLP[id]=psiEvalSpin1NLP.get_gradient(q);            
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
										  zetalmDeltaVlProductDistImageAtomsQuads[q][i][iPseudoWave][ikPoint][0][idim][iSubCell]=dftPtr->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*C_DIM*2+q*C_DIM*2+idim*2+0];
											zetalmDeltaVlProductDistImageAtomsQuads[q][i][iPseudoWave][ikPoint][1][idim][iSubCell]=dftPtr->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*C_DIM*2+q*C_DIM*2+idim*2+1];
#else  
										  zetalmDeltaVlProductDistImageAtomsQuads[q][i][iPseudoWave][idim][iSubCell]=dftPtr->d_nonLocalPSP_zetalmDeltaVlProductDistImageAtoms_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*C_DIM+q*C_DIM+idim];
#endif                      
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
			const VectorizedArray<double> phiExt_q =make_vectorized_array(0.0);

			Tensor<2,C_DIM,VectorizedArray<double> > E=eshelbyTensorSP::getELocXcEshelbyTensor
				(rhoQuads[q],
				 gradRhoSpin0Quads[q],
				 gradRhoSpin1Quads[q],
				 excQuads[q],
				 derExchCorrEnergyWithGradRhoOutSpin0Quads[q],
				 derExchCorrEnergyWithGradRhoOutSpin1Quads[q]);

#ifdef USE_COMPLEX
			Tensor<2,C_DIM,VectorizedArray<double> > EKPoints=eshelbyTensorSP::getELocWfcEshelbyTensorPeriodicKPoints
				(psiSpin0Quads.begin()+q*numEigenVectors*numKPoints,
				 psiSpin1Quads.begin()+q*numEigenVectors*numKPoints,
				 gradPsiSpin0Quads.begin()+q*numEigenVectors*numKPoints,
				 gradPsiSpin1Quads.begin()+q*numEigenVectors*numKPoints,
				 dftPtr->d_kPointCoordinates,
				 dftPtr->d_kPointWeights,
				 dftPtr->eigenValues,
				 dftPtr->fermiEnergy,
				 dftPtr->fermiEnergyUp,
				 dftPtr->fermiEnergyDown,
				 dftParameters::TVal);

			EKPoints+=eshelbyTensorSP::getEKStress
				(psiSpin0Quads.begin()+q*numEigenVectors*numKPoints,
				 psiSpin1Quads.begin()+q*numEigenVectors*numKPoints,
				 gradPsiSpin0Quads.begin()+q*numEigenVectors*numKPoints,
				 gradPsiSpin1Quads.begin()+q*numEigenVectors*numKPoints,
				 dftPtr->d_kPointCoordinates,
				 dftPtr->d_kPointWeights,
				 dftPtr->eigenValues,
				 dftPtr->fermiEnergy,
				 dftPtr->fermiEnergyUp,
				 dftPtr->fermiEnergyDown,
				 dftParameters::TVal);
#else
			Tensor<2,C_DIM,VectorizedArray<double> > EKPoints=eshelbyTensorSP::getELocWfcEshelbyTensorNonPeriodic
				(psiSpin0Quads.begin()+q*numEigenVectors*numKPoints,
				 psiSpin1Quads.begin()+q*numEigenVectors*numKPoints,
				 gradPsiSpin0Quads.begin()+q*numEigenVectors*numKPoints,
				 gradPsiSpin1Quads.begin()+q*numEigenVectors*numKPoints,
				 dftPtr->eigenValues[0],
				 dftPtr->fermiEnergy,
				 dftPtr->fermiEnergyUp,
				 dftPtr->fermiEnergyDown,
				 dftParameters::TVal);
#endif        

			EQuadSum+=E*forceEval.JxW(q);
			EKPointsQuadSum+=EKPoints*forceEval.JxW(q);

		}//quad point loop

		if (isPseudopotential)
			for (unsigned int q=0; q<numQuadPointsNLP; ++q)
			{
#ifdef USE_COMPLEX              
				Tensor<2,C_DIM,VectorizedArray<double> > Enl
					=eshelbyTensorSP::getEnlStress(zetalmDeltaVlProductDistImageAtomsQuads[q],
						projectorKetTimesPsiSpin0TimesVTimesPartOcc,
						projectorKetTimesPsiSpin1TimesVTimesPartOcc,
						psiSpin0QuadsNLP.begin()+q*numEigenVectors*numKPoints,
						psiSpin1QuadsNLP.begin()+q*numEigenVectors*numKPoints,
						gradPsiSpin0QuadsNLP.begin()+q*numEigenVectors*numKPoints,
						gradPsiSpin1QuadsNLP.begin()+q*numEigenVectors*numKPoints,            
						dftPtr->d_kPointWeights,
            dftPtr->d_kPointCoordinates,
						macroIdToNonlocalAtomsSetMap[cell],
						numEigenVectors);
#else
				Tensor<2,C_DIM,VectorizedArray<double> > Enl
					=eshelbyTensorSP::getEnlStress(zetalmDeltaVlProductDistImageAtomsQuads[q],
						projectorKetTimesPsiSpin0TimesVTimesPartOcc,
						projectorKetTimesPsiSpin1TimesVTimesPartOcc,
						psiSpin0QuadsNLP.begin()+q*numEigenVectors*numKPoints,
						psiSpin1QuadsNLP.begin()+q*numEigenVectors*numKPoints,
						gradPsiSpin0QuadsNLP.begin()+q*numEigenVectors*numKPoints,
						gradPsiSpin1QuadsNLP.begin()+q*numEigenVectors*numKPoints,            
						macroIdToNonlocalAtomsSetMap[cell],
						numEigenVectors);
#endif          


				EKPointsQuadSum+=Enl*forceEvalNLP.JxW(q);
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
		 pseudoVLocAtomsElectro,
		 vselfBinsManagerElectro);
}
