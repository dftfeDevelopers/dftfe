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
void forceClass<FEOrder,FEOrderElectro>::computeStressEEshelbyEPSPEnlEk(const MatrixFree<3,double> & matrixFreeData,
		const unsigned int eigenDofHandlerIndex,
    const unsigned int smearedChargeQuadratureId,
      const unsigned int lpspQuadratureIdElectro,          
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
			 eigenVectors[kPoint],
       false);

    //FIXME: The underlying call to update_ghost_values
    //is required because currently localProc_dof_indicesReal
    //and localProc_dof_indicesImag are only available for
    //locally owned nodes. Once they are also made available
    //for ghost nodes- use true for the last argument in
    //copyFlattenedDealiiVecToSingleCompVec(..) above and supress
    //underlying call.
    for(unsigned int i= 0; i < dftPtr->d_numEigenValues; ++i)
      eigenVectors[kPoint][i].update_ghost_values();      
#else
		vectorTools::copyFlattenedDealiiVecToSingleCompVec
			(dftPtr->d_eigenVectorsFlattened[kPoint],
			 dftPtr->d_numEigenValues,
			 std::make_pair(0,dftPtr->d_numEigenValues),
			 eigenVectors[kPoint],
       true);
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
	FEEvaluation<C_DIM,FEOrder,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),2> psiEval(matrixFreeData,
			eigenDofHandlerIndex,
			0);
	FEEvaluation<C_DIM,FEOrder,C_num1DQuadNLPSP<FEOrder>()*C_numCopies1DQuadNLPSP(),2> psiEvalNLP(matrixFreeData,
			eigenDofHandlerIndex,
		  dftPtr->d_nlpspQuadratureId);
#else
	FEEvaluation<C_DIM,FEOrder,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),1> psiEval(matrixFreeData,
			eigenDofHandlerIndex,
			0);
	FEEvaluation<C_DIM,FEOrder,C_num1DQuadNLPSP<FEOrder>()*C_numCopies1DQuadNLPSP(),1> psiEvalNLP(matrixFreeData,
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

	std::vector<std::vector<std::vector<dataTypes::number > > > projectorKetTimesPsiTimesVTimesPartOcc(numKPoints);
	if (isPseudopotential)
  {
		for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
			computeNonLocalProjectorKetTimesPsiTimesVFlattened
				(dftPtr->d_eigenVectorsFlattened[ikPoint],
				 numEigenVectors,
				 projectorKetTimesPsiTimesVTimesPartOcc[ikPoint],
				 ikPoint,
				 partialOccupancies[ikPoint],
				 true
         );
	}

  std::vector<VectorizedArray<double> > rhoXCQuads(numQuadPoints,make_vectorized_array(0.0));  
	std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoQuads(numQuadPoints,zeroTensor3);
	std::vector<VectorizedArray<double> > excQuads(numQuadPoints,make_vectorized_array(0.0));
  std::vector<VectorizedArray<double> > vxcRhoOutQuads(numQuadPoints,make_vectorized_array(0.0));
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
					dftPtr->d_cellIdToNonlocalAtomIdsLocalCompactSupportMap[subCellId].begin(), dftPtr->d_cellIdToNonlocalAtomIdsLocalCompactSupportMap[subCellId].end(),
					std::inserter(s, s.begin()));
			mergedSet=s;
		}
		macroIdToNonlocalAtomsSetMap[cell]=std::vector<unsigned int>(mergedSet.begin(),mergedSet.end());
	}


	for (unsigned int cell=0; cell<matrixFreeData.n_macro_cells(); ++cell)
	{
		forceEval.reinit(cell);
		psiEval.reinit(cell);

		if (isPseudopotential)
		{
			forceEvalNLP.reinit(cell);
			psiEvalNLP.reinit(cell);
		}

    std::fill(rhoXCQuads.begin(),rhoXCQuads.end(),make_vectorized_array(0.0));
		std::fill(gradRhoQuads.begin(),gradRhoQuads.end(),zeroTensor3);
		std::fill(excQuads.begin(),excQuads.end(),make_vectorized_array(0.0));
	  std::fill(vxcRhoOutQuads.begin(),vxcRhoOutQuads.end(),make_vectorized_array(0.0));    
		std::fill(derExchCorrEnergyWithGradRhoOutQuads.begin(),derExchCorrEnergyWithGradRhoOutQuads.end(),zeroTensor3);

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
#else
            zetalmDeltaVlProductDistImageAtomsQuads[q][i][iPseudoWave]=zeroTensor3;
#endif            
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
		std::vector<double> rhoOutQuadsXC(numQuadPoints);

		//
		//For GGA
		std::vector<double> sigmaValRhoOut(numQuadPoints);
		std::vector<double> derExchEnergyWithDensityValRhoOut(numQuadPoints), derCorrEnergyWithDensityValRhoOut(numQuadPoints), derExchEnergyWithSigmaRhoOut(numQuadPoints),derCorrEnergyWithSigmaRhoOut(numQuadPoints);
    std::vector<Tensor<1,C_DIM,double > > gradRhoOutQuadsXC(numQuadPoints);

		//
		for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
		{
			subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell);
			dealii::CellId subCellId=subCellPtr->id();

      const std::vector<double> & temp1=rhoOutValues.find(subCellId)->second;
      for (unsigned int q=0; q<numQuadPoints; ++q)
      {
        rhoOutQuadsXC[q]=temp1[q];
        rhoXCQuads[q][iSubCell]=temp1[q];
      }

      if(dftParameters::nonLinearCoreCorrection)
      {
        const std::vector<double> & temp2=rhoCoreValues.find(subCellId)->second;
        for (unsigned int q=0; q<numQuadPoints; ++q)
        {
          rhoOutQuadsXC[q]+=temp2[q];
          rhoXCQuads[q][iSubCell]+=temp2[q];
        }
      }

      if(dftParameters::xcFamilyType=="GGA")
      {
        const std::vector<double> & temp3=gradRhoOutValues.find(subCellId)->second;          
        for (unsigned int q = 0; q < numQuadPoints; ++q)
          for (unsigned int idim=0; idim<C_DIM; idim++)
          {
            gradRhoOutQuadsXC[q][idim] = temp3[3*q + idim];
            gradRhoQuads[q][idim][iSubCell]=temp3[3*q+idim];     
          }

        if(dftParameters::nonLinearCoreCorrection)
        {
          const std::vector<double> & temp4=gradRhoCoreValues.find(subCellId)->second;
          for (unsigned int q = 0; q < numQuadPoints; ++q)
          {
            gradRhoOutQuadsXC[q][0] += temp4[3*q + 0];
            gradRhoOutQuadsXC[q][1] += temp4[3*q + 1];
            gradRhoOutQuadsXC[q][2] += temp4[3*q + 2];
          }        
        }          
      }

			if(dftParameters::xcFamilyType=="GGA")
			{
				for (unsigned int q = 0; q < numQuadPoints; ++q)
					sigmaValRhoOut[q] = gradRhoOutQuadsXC[q].norm_square();

				xc_gga_exc_vxc(&(dftPtr->funcX),numQuadPoints,&(rhoOutQuadsXC[0]),&sigmaValRhoOut[0],&exchValRhoOut[0],&derExchEnergyWithDensityValRhoOut[0],&derExchEnergyWithSigmaRhoOut[0]);
				xc_gga_exc_vxc(&(dftPtr->funcC),numQuadPoints,&(rhoOutQuadsXC[0]),&sigmaValRhoOut[0],&corrValRhoOut[0],&derCorrEnergyWithDensityValRhoOut[0],&derCorrEnergyWithSigmaRhoOut[0]);
				for (unsigned int q=0; q<numQuadPoints; ++q)
				{
					excQuads[q][iSubCell]=exchValRhoOut[q]+corrValRhoOut[q];
          vxcRhoOutQuads[q][iSubCell]= derExchEnergyWithDensityValRhoOut[q]+derCorrEnergyWithDensityValRhoOut[q];
					for (unsigned int idim=0; idim<C_DIM; idim++)
					{
						derExchCorrEnergyWithGradRhoOutQuads[q][idim][iSubCell]=2.0*(derExchEnergyWithSigmaRhoOut[q]+derCorrEnergyWithSigmaRhoOut[q])*gradRhoOutQuadsXC[q][idim];
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
					vxcRhoOutQuads[q][iSubCell]= exchPotValRhoOut[q]+corrPotValRhoOut[q];          
				}
			}
		}

#ifdef USE_COMPLEX  
		std::vector<Tensor<1,2,VectorizedArray<double> > > psiQuads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor1);
		std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > > gradPsiQuads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor2);
#else
		std::vector<VectorizedArray<double> > psiQuads(numQuadPoints*numEigenVectors*numKPoints,make_vectorized_array(0.0));
		std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPsiQuads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor3);
#endif    

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
				}//quad point loop
			} //eigenvector loop


#ifdef USE_COMPLEX  
		std::vector<Tensor<1,2,VectorizedArray<double> > > psiQuadsNLP;
		std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > > gradPsiQuadsNLP;    
#else
		std::vector<VectorizedArray<double> > psiQuadsNLP;
		std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPsiQuadsNLP;       
#endif   
		if (isPseudopotential)
		{
#ifdef USE_COMPLEX        
			psiQuadsNLP.resize(numQuadPointsNLP*numEigenVectors*numKPoints,zeroTensor1);
      gradPsiQuadsNLP.resize(numQuadPointsNLP*numEigenVectors*numKPoints,zeroTensor2);  
#else
			psiQuadsNLP.resize(numQuadPointsNLP*numEigenVectors*numKPoints,make_vectorized_array(0.0));
      gradPsiQuadsNLP.resize(numQuadPointsNLP*numEigenVectors*numKPoints,zeroTensor3);  
#endif      
			for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
				for (unsigned int iEigenVec=0; iEigenVec<numEigenVectors; ++iEigenVec)
				{
					psiEvalNLP.read_dof_values_plain(eigenVectors[ikPoint][iEigenVec]);
				  psiEvalNLP.evaluate(true,true);

					for (unsigned int q=0; q<numQuadPointsNLP; ++q)
					{
						const unsigned int id=q*numEigenVectors*numKPoints+numEigenVectors*ikPoint+iEigenVec;
						psiQuadsNLP[id]=psiEvalNLP.get_value(q);
            gradPsiQuadsNLP[id]=psiEvalNLP.get_gradient(q);
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
			Point< 3, VectorizedArray<double> > quadPoint_q;

			Tensor<2,C_DIM,VectorizedArray<double> > E=eshelbyTensor::getELocXcEshelbyTensor
				(rhoXCQuads[q],
				 gradRhoQuads[q],
				 excQuads[q],
				 derExchCorrEnergyWithGradRhoOutQuads[q]);

#ifdef USE_COMPLEX
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
#else        
			Tensor<2,C_DIM,VectorizedArray<double> > EKPoints=eshelbyTensor::getELocWfcEshelbyTensorNonPeriodic
				(psiQuads.begin()+q*numEigenVectors*numKPoints,
				 gradPsiQuads.begin()+q*numEigenVectors*numKPoints,
				 dftPtr->eigenValues[0],
				 partialOccupancies[0]);
#endif

			EQuadSum+=E*forceEval.JxW(q);
			EKPointsQuadSum+=EKPoints*forceEval.JxW(q);
		}//quad point loop

		if (isPseudopotential)
    {
			for (unsigned int q=0; q<numQuadPointsNLP; ++q)
			{
#ifdef USE_COMPLEX        
				Tensor<2,C_DIM,VectorizedArray<double> > Enl
					=eshelbyTensor::getEnlStress(zetalmDeltaVlProductDistImageAtomsQuads[q],
						projectorKetTimesPsiTimesVTimesPartOcc,
						psiQuadsNLP.begin()+q*numEigenVectors*numKPoints,
						gradPsiQuadsNLP.begin()+q*numEigenVectors*numKPoints,            
						dftPtr->d_kPointWeights,
            dftPtr->d_kPointCoordinates,            
						macroIdToNonlocalAtomsSetMap[cell],
						numEigenVectors);
#else
				Tensor<2,C_DIM,VectorizedArray<double> > Enl
					=eshelbyTensor::getEnlStress(zetalmDeltaVlProductDistImageAtomsQuads[q],
						projectorKetTimesPsiTimesVTimesPartOcc,
						psiQuadsNLP.begin()+q*numEigenVectors*numKPoints,
						gradPsiQuadsNLP.begin()+q*numEigenVectors*numKPoints,            
						macroIdToNonlocalAtomsSetMap[cell],
						numEigenVectors);
#endif          


				EKPointsQuadSum+=Enl*forceEvalNLP.JxW(q);
			}

      if (dftParameters::nonLinearCoreCorrection)
        addENonlinearCoreCorrectionStressContribution(forceEval,
            matrixFreeData,
            cell,
            vxcRhoOutQuads,
            derExchCorrEnergyWithGradRhoOutQuads,
            gradRhoCoreAtoms,
            hessianRhoCoreAtoms);
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

template<unsigned int FEOrder,unsigned int FEOrderElectro>
	void forceClass<FEOrder,FEOrderElectro>::computeStressEEshelbyEElectroPhiTot
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
 const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & pseudoVLocAtomsElectro,
 const vselfBinsManager<FEOrder,FEOrderElectro> & vselfBinsManagerElectro)
{
	FEEvaluation<C_DIM,1,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),C_DIM>  forceEvalElectro(matrixFreeDataElectro,
			d_forceDofHandlerIndexElectro,
			0);

	FEEvaluation<C_DIM,FEOrderElectro,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),1> phiTotEvalElectro(matrixFreeDataElectro,
			phiTotDofHandlerIndexElectro,
			0);

	FEEvaluation<C_DIM,FEOrderElectro,C_num1DQuadSmearedCharge()*C_numCopies1DQuadSmearedCharge(),1>  phiTotEvalSmearedCharge(matrixFreeDataElectro,
			phiTotDofHandlerIndexElectro,
			smearedChargeQuadratureId);

	FEEvaluation<C_DIM,1,C_num1DQuadSmearedCharge()*C_numCopies1DQuadSmearedCharge(),C_DIM>  forceEvalSmearedCharge(matrixFreeDataElectro,
			d_forceDofHandlerIndexElectro,
			smearedChargeQuadratureId);

	FEEvaluation<C_DIM,1,C_num1DQuadLPSP<FEOrderElectro>()*C_numCopies1DQuadLPSP(),C_DIM>  forceEvalElectroLpsp(matrixFreeDataElectro,
			d_forceDofHandlerIndexElectro,
			lpspQuadratureIdElectro);  

	FEValues<C_DIM> feVselfValuesElectro (matrixFreeDataElectro.
			get_dof_handler(phiTotDofHandlerIndexElectro).get_fe(),
			matrixFreeDataElectro.get_quadrature(lpspQuadratureIdElectro),
			update_values| update_quadrature_points);

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
  std::vector< Tensor<1,C_DIM,VectorizedArray<double> > > gradPhiTotSmearedChargeQuads(numQuadPointsSmearedb,zeroTensor);    
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
      phiTotEvalSmearedCharge.evaluate(false,true);        
    }

		std::fill(rhoQuadsElectro.begin(),rhoQuadsElectro.end(),make_vectorized_array(0.0));
    std::fill(rhoQuadsElectroLpsp.begin(),rhoQuadsElectroLpsp.end(),make_vectorized_array(0.0));    
		std::fill(gradRhoQuadsElectro.begin(),gradRhoQuadsElectro.end(),zeroTensor);
    std::fill(gradRhoQuadsElectroLpsp.begin(),gradRhoQuadsElectroLpsp.end(),zeroTensor);    
		std::fill(pseudoVLocQuadsElectro.begin(),pseudoVLocQuadsElectro.end(),make_vectorized_array(0.0));
    std::fill(smearedbQuads.begin(),smearedbQuads.end(),make_vectorized_array(0.0));
    std::fill(gradPhiTotSmearedChargeQuads.begin(),gradPhiTotSmearedChargeQuads.end(),zeroTensor);    

		const unsigned int numSubCells=matrixFreeDataElectro.n_components_filled(cell);

    double sum=0.0; 
		for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
		{
			subCellPtr= matrixFreeDataElectro.get_cell_iterator(cell,iSubCell);
			dealii::CellId subCellId=subCellPtr->id();
			for (unsigned int q=0; q<numQuadPoints; ++q)
				rhoQuadsElectro[q][iSubCell]=rhoOutValuesElectro.find(subCellId)->second[q];


			if(dftParameters::isPseudopotential || dftParameters::smearedNuclearCharges)
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
        const std::vector<double> & bQuadValuesCell= dftPtr->d_bQuadValuesAllAtoms.find(subCellId)->second;
        for (unsigned int q=0; q<numQuadPointsSmearedb; ++q)
        {
          smearedbQuads[q][iSubCell]=bQuadValuesCell[q];
          sum+=bQuadValuesCell[q];
        }            
      }
		}

		if (dftParameters::isPseudopotential || dftParameters::smearedNuclearCharges)
		{
			addEPSPStressContribution(feVselfValuesElectro,
					forceEvalElectroLpsp,
					matrixFreeDataElectro,
          phiTotDofHandlerIndexElectro,
					cell,
					gradRhoQuadsElectroLpsp,
					pseudoVLocAtomsElectro,
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

			EQuadSum+=E*forceEvalElectro.JxW(q);
		}

		if(dftParameters::isPseudopotential || dftParameters::smearedNuclearCharges)
      for (unsigned int q=0; q<numQuadPointsLpsp; ++q)
      {
			  VectorizedArray<double> phiExtElectro_q =make_vectorized_array(0.0);
        Tensor<2,C_DIM,VectorizedArray<double> > E=zeroTensor2;

        EQuadSum+=E*forceEvalElectroLpsp.JxW(q);
      }    

		for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
			for (unsigned int idim=0; idim<C_DIM; ++idim)
				for (unsigned int jdim=0; jdim<C_DIM; ++jdim)
					d_stress[idim][jdim]+=EQuadSum[idim][jdim][iSubCell];

    if (dftParameters::smearedNuclearCharges && std::abs(sum)>1e-9)
    {
      for (unsigned int q=0; q<numQuadPointsSmearedb; ++q)
      {
        gradPhiTotSmearedChargeQuads[q]=phiTotEvalSmearedCharge.get_gradient(q);
      }

			addEPhiTotSmearedStressContribution(forceEvalSmearedCharge,
					matrixFreeDataElectro,
					cell,
          gradPhiTotSmearedChargeQuads,
          dftPtr->d_bQuadAtomIdsAllAtomsImages,
				  smearedbQuads);
    }

	}//cell loop
}
