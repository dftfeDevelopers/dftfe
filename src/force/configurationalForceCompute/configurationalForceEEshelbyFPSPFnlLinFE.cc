// ---------------------------------------------------------------------
//
// Copyright (c) 2017-18 The Regents of the University of Michigan and DFT-FE authors.
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
namespace internalforce
{
   //for real valued eigenvectors
   Tensor<1,C_DIM,VectorizedArray<double> > computeGradRhoContribution
                                                     (const VectorizedArray<double> &  psi,
	                                              const Tensor<1,C_DIM,VectorizedArray<double>> & gradPsi)
   {
       return make_vectorized_array(2.0)*(gradPsi*psi);
   }

   //for complex valued eigenvectors
   Tensor<1,C_DIM,VectorizedArray<double> > computeGradRhoContribution
                                     (const Tensor<1,2,VectorizedArray<double> > & psi,
	                              const Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > >  & gradPsi)
   {
       return make_vectorized_array(2.0)*(gradPsi[0]*psi[0]+ gradPsi[1]*psi[1]);
   }

   //for real valued eigenvectors
   Tensor<2,C_DIM,VectorizedArray<double> > computeHessianRhoContribution
                                      (const VectorizedArray<double> psi,
	                               const Tensor<1,C_DIM,VectorizedArray<double> > & gradPsi,
                                       const Tensor<2,C_DIM,VectorizedArray<double> > & hessianPsi)
   {
       return make_vectorized_array(2.0)*(hessianPsi*psi+outer_product(gradPsi,gradPsi));
   }

   //for complex valued eigenvectors
   Tensor<2,C_DIM,VectorizedArray<double> > computeHessianRhoContribution
                                            (const Tensor<1,2,VectorizedArray<double> > & psi,
	                                     const Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > >  & gradPsi,
                                             const Tensor<1,2,Tensor<2,C_DIM,VectorizedArray<double> > >  & hessianPsi)
   {
       return make_vectorized_array(2.0)*(hessianPsi[0]*psi[0]+ hessianPsi[1]*psi[1]+ outer_product(gradPsi[0],gradPsi[0])+outer_product(gradPsi[1],gradPsi[1]));
   }

}

//compute configurational force contribution from all terms except the nuclear self energy
template<unsigned int FEOrder>
void forceClass<FEOrder>::computeConfigurationalForceEEshelbyTensorFPSPFnlLinFE
			      (const MatrixFree<3,double> & matrixFreeData,
#ifdef DFTFE_WITH_GPU
                               kohnShamDFTOperatorCUDAClass<FEOrder> & kohnShamDFTEigenOperator,
#endif
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
                              const std::map<dealii::CellId, std::vector<double> > & rhoOutValues,
                              const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValues,
			      const std::map<dealii::CellId, std::vector<double> > & rhoOutValuesElectro,
			      const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValuesElectro,
		              const std::map<dealii::CellId, std::vector<double> > & pseudoVLocElectro,
		              const std::map<dealii::CellId, std::vector<double> > & gradPseudoVLocElectro,
		              const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & gradPseudoVLocAtomsElectro,
			      const vselfBinsManager<FEOrder> & vselfBinsManagerElectro,
                              const std::map<dealii::CellId, std::vector<double> > & shadowKSRhoMinValues,
                              const std::map<dealii::CellId, std::vector<double> > & shadowKSGradRhoMinValues,
                              const vectorType & phiRhoMinusApproxRho,
                              const bool shadowPotentialForce)
{
  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
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
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),2> psiEval(matrixFreeData,
	                                                       eigenDofHandlerIndex,
							       0);

  FEEvaluation<C_DIM,FEOrder,C_num1DQuadPSP<FEOrder>(),2> psiEvalNLP(matrixFreeData,
	                                                       eigenDofHandlerIndex,
							       2);
#else
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> psiEval(matrixFreeData,
	                                                       eigenDofHandlerIndex,
							       0);

  FEEvaluation<C_DIM,FEOrder,C_num1DQuadPSP<FEOrder>(),1> psiEvalNLP(matrixFreeData,
	                                                       eigenDofHandlerIndex,
							       2);
#endif

  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiTotInEval(matrixFreeData,
	                                                            phiTotDofHandlerIndex,
								    0);

  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiTotOutEval(matrixFreeData,
	                                                            phiTotDofHandlerIndex,
								    0);

  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiTotEval2(matrixFreeData,
	                                                          phiTotDofHandlerIndex,
								  0);

  //FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiExtEval(matrixFreeData,
  //	                                                          phiExtDofHandlerIndex,
  //								  0);

  std::map<unsigned int, std::vector<double> > forceContributionShadowLocalGammaAtoms;


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

  std::vector<unsigned int> nonlocalPseudoWfcsAccum(dftPtr->d_nonLocalAtomIdsInCurrentProcess.size());
  std::vector<unsigned int> numPseudoWfcsAtom(dftPtr->d_nonLocalAtomIdsInCurrentProcess.size());
  unsigned int numPseudo=0;
  for(unsigned int iAtom = 0; iAtom < dftPtr->d_nonLocalAtomIdsInCurrentProcess.size(); ++iAtom)
    {
      const unsigned int atomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iAtom];
      nonlocalPseudoWfcsAccum[iAtom]=numPseudo;
      numPseudo+= dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
      numPseudoWfcsAtom[iAtom]=dftPtr->d_numberPseudoAtomicWaveFunctions[atomId];
    }

  //band group parallelization data structures
  const unsigned int numberBandGroups=
	dealii::Utilities::MPI::n_mpi_processes(dftPtr->interBandGroupComm);
  const unsigned int bandGroupTaskId = dealii::Utilities::MPI::this_mpi_process(dftPtr->interBandGroupComm);
  std::vector<unsigned int> bandGroupLowHighPlusOneIndices;
  dftUtils::createBandParallelizationIndices(dftPtr->interBandGroupComm,
					     numEigenVectors,
					     bandGroupLowHighPlusOneIndices);

  const unsigned int blockSize=std::min(dftParameters::chebyWfcBlockSize,
	                                bandGroupLowHighPlusOneIndices[1]);

  const unsigned int localVectorSize = dftPtr->d_eigenVectorsFlattenedSTL[0].size()/numEigenVectors;
  std::vector<std::vector<vectorType>> eigenVectors((1+dftParameters::spinPolarized)*dftPtr->d_kPointWeights.size());
  std::vector<dealii::LinearAlgebra::distributed::Vector<dataTypes::number> > eigenVectorsFlattenedBlock((1+dftParameters::spinPolarized)*dftPtr->d_kPointWeights.size());

   double wfc_time_total=0.0;
   double fnlgamma_time_total=0.0;
   double enlfnl_time_total=0.0;
   double eloc_time_total=0.0;
   double nlpinit_time_total=0.0;
   double projketpsi_time_total=0.0;
   double nlppsicontract_time_total=0.0;
   double gpuportedforce_time_total=0.0;

   const unsigned int numMacroCells=matrixFreeData.n_macro_cells();
   const unsigned int numPhysicalCells=matrixFreeData.n_physical_cells();

#if defined(DFTFE_WITH_GPU)
   AssertThrow(numMacroCells==numPhysicalCells,ExcMessage("DFT-FE Error: dealii for GPU DFT-FE must be compiled without any vectorization enabled."));

   //create map between macro cell id and normal cell id
   std::vector<unsigned int> normalCellIdToMacroCellIdMap(numPhysicalCells);
   std::vector<unsigned int> macroCellIdToNormalCellIdMap(numPhysicalCells);

   typename dealii::DoFHandler<3>::active_cell_iterator cellPtr;
   unsigned int iElemNormal = 0;
   for(const auto &cell : matrixFreeData.get_dof_handler().active_cell_iterators())
   {
          if(cell->is_locally_owned())
            {
                bool isFound=false;
                unsigned int iElemMacroCell = 0;
                for(unsigned int iMacroCell = 0; iMacroCell < numMacroCells; ++iMacroCell)
                {
                       const unsigned int n_sub_cells = matrixFreeData.n_components_filled(iMacroCell);
                       for(unsigned int iCell = 0; iCell < n_sub_cells; ++iCell)
                       {
                              cellPtr = matrixFreeData.get_cell_iterator(iMacroCell,iCell);
                              if (cell->id()==cellPtr->id())
                              {
                                normalCellIdToMacroCellIdMap[iElemNormal]=iElemMacroCell;
                                macroCellIdToNormalCellIdMap[iElemMacroCell]=iElemNormal;
                                isFound=true;
                                break;
                              }
                              iElemMacroCell++;
                       }
                       
                       if (isFound)
                         break;
                }
                iElemNormal++;
            }
   }

   std::vector<unsigned int> nonTrivialNonLocalIdsAllCells;
   std::vector<unsigned int> nonTrivialIdToElemIdMap;
   std::vector<unsigned int> nonTrivialIdToAllPseudoWfcIdMap;
   std::vector<unsigned int> projecterKetTimesFlattenedVectorLocalIds; 
   if (isPseudopotential)
   {
          double nlpinit_time;
          nlpinit_time = clock();
          for (unsigned int ielem=0; ielem<numPhysicalCells; ++ielem)
          {
            const unsigned int numNonLocalAtomsCurrentProc= dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();

              for (unsigned int iatom=0; iatom<numNonLocalAtomsCurrentProc; ++iatom)
              {
                 bool isNonTrivial=false;
		 for (unsigned int i=0;i<macroIdToNonlocalAtomsSetMap[normalCellIdToMacroCellIdMap[ielem]].size();i++)
		      if (macroIdToNonlocalAtomsSetMap[normalCellIdToMacroCellIdMap[ielem]][i]==iatom)
		      {
                          isNonTrivial=true;
                          break;
		      }
                 if (isNonTrivial)
                 {
                    const int globalAtomId=dftPtr->d_nonLocalAtomIdsInCurrentProcess[iatom];
                    const unsigned int numberSingleAtomPseudoWaveFunctions=numPseudoWfcsAtom[iatom];
                    for (unsigned int ipsp=0; ipsp<numberSingleAtomPseudoWaveFunctions; ++ipsp)
		    {
                         nonTrivialNonLocalIdsAllCells.push_back(iatom);
                         nonTrivialIdToElemIdMap.push_back(ielem);
                         nonTrivialIdToAllPseudoWfcIdMap.push_back(nonlocalPseudoWfcsAccum[iatom]+ipsp);
			 const unsigned int id=dftPtr->d_projectorKetTimesVectorPar[0].get_partitioner()->global_to_local(dftPtr->d_projectorIdsNumberingMapCurrentProcess[std::make_pair(globalAtomId,ipsp)]);
                         projecterKetTimesFlattenedVectorLocalIds.push_back(id);
                    }
                 }
              }
          }
          nlpinit_time = clock() - nlpinit_time;
          nlpinit_time_total+=nlpinit_time;
   }
#endif


#ifdef USE_COMPLEX
   //vector of quadPoints times macrocells, nonlocal atom id, pseudo wave, k point
   //FIXME: flatten nonlocal atomid id and pseudo wave and k point
   std::vector<std::vector<std::vector<std::vector<Tensor<1,2,VectorizedArray<double> > > > > >ZetaDeltaVQuads;
   std::vector<std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > >gradZetaDeltaVQuads; 
   std::vector<std::vector<std::vector<std::vector<Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > > > > >pspnlGammaAtomsQuads;
#else
   //FIXME: flatten nonlocal atom id and pseudo wave
   //vector of quadPoints times macrocells, nonlocal atom id, pseudo wave
   std::vector<std::vector<std::vector<VectorizedArray<double> > > > ZetaDeltaVQuads;
   std::vector<std::vector<std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > > > gradZetaDeltaVQuads;
#endif

#if defined(DFTFE_WITH_GPU) && !defined(USE_COMPLEX)
   std::vector<double>  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened(nonTrivialNonLocalIdsAllCells.size()*numQuadPointsNLP,0.0);
   std::vector<double> elocWfcEshelbyTensorQuadValuesH00(numPhysicalCells*numQuadPoints,0.0);
   std::vector<double> elocWfcEshelbyTensorQuadValuesH10(numPhysicalCells*numQuadPoints,0.0);
   std::vector<double> elocWfcEshelbyTensorQuadValuesH11(numPhysicalCells*numQuadPoints,0.0);
   std::vector<double> elocWfcEshelbyTensorQuadValuesH20(numPhysicalCells*numQuadPoints,0.0);
   std::vector<double> elocWfcEshelbyTensorQuadValuesH21(numPhysicalCells*numQuadPoints,0.0);
   std::vector<double> elocWfcEshelbyTensorQuadValuesH22(numPhysicalCells*numQuadPoints,0.0);
#endif
   std::vector<std::vector<std::vector<dataTypes::number> > > projectorKetTimesPsiTimesVTimesPartOcc(numKPoints);
   std::vector<std::vector<VectorizedArray<double>> >  projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuads(numMacroCells*numQuadPointsNLP,std::vector<VectorizedArray<double>>(numPseudo,make_vectorized_array(0.0)));

   if(isPseudopotential)
   {
       double nlpinit_time;
       nlpinit_time = clock();

       if(isPseudopotential)
       {
		ZetaDeltaVQuads.resize(numMacroCells*numQuadPointsNLP);
		gradZetaDeltaVQuads.resize(numMacroCells*numQuadPointsNLP);
#ifdef USE_COMPLEX
		pspnlGammaAtomsQuads.resize(numMacroCells*numQuadPointsNLP);
#endif

		for (unsigned int q=0; q<numQuadPointsNLP*numMacroCells; ++q)
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

       for (unsigned int cell=0; cell<matrixFreeData.n_macro_cells(); ++cell)
       {
               const unsigned int numSubCells=matrixFreeData.n_components_filled(cell);
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
			      ZetaDeltaVQuads[cell*numQuadPointsNLP+q][i][iPseudoWave][ikPoint][0][iSubCell]=d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*2+q*2+0];
			      ZetaDeltaVQuads[cell*numQuadPointsNLP+q][i][iPseudoWave][ikPoint][1][iSubCell]=d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*2+q*2+1];
			      for (unsigned int idim=0; idim<C_DIM; idim++)
			      {
				 gradZetaDeltaVQuads[cell*numQuadPointsNLP+q][i][iPseudoWave][ikPoint][0][idim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*C_DIM*2+q*C_DIM*2+idim*2+0];
				 gradZetaDeltaVQuads[cell*numQuadPointsNLP+q][i][iPseudoWave][ikPoint][1][idim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*C_DIM*2+q*C_DIM*2+idim*2+1];
				 pspnlGammaAtomsQuads[cell*numQuadPointsNLP+q][i][iPseudoWave][ikPoint][0][idim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVl_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*C_DIM*2+q*C_DIM*2+idim*2+0];
				 pspnlGammaAtomsQuads[cell*numQuadPointsNLP+q][i][iPseudoWave][ikPoint][1][idim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVl_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPointsNLP*C_DIM*2+q*C_DIM*2+idim*2+1];
			      }
			   }
#else
			   ZetaDeltaVQuads[cell*numQuadPointsNLP+q][i][iPseudoWave][iSubCell]=d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave][subCellId][q];

			   for (unsigned int idim=0; idim<C_DIM; idim++)
			       gradZetaDeltaVQuads[cell*numQuadPointsNLP+q][i][iPseudoWave][idim][iSubCell]=
				   d_nonLocalPSP_gradZetalmDeltaVl[i][iPseudoWave][subCellId][q*C_DIM+idim];
#endif
			}//non-trivial cellId check
		      }//iPseudoWave loop
		    }//i loop
		  }//q loop
	       }//subcell loop
       }
	      
       nlpinit_time = clock() - nlpinit_time;
       nlpinit_time_total+=nlpinit_time;
   }
  
   double parallelflattened_time;
   parallelflattened_time = clock();
#if defined(DFTFE_WITH_GPU) && !defined(USE_COMPLEX)
   cudaVectorType cudaFlattenedArrayBlock;
   cudaVectorType projectorKetTimesVectorD;

   if (dftParameters::useGPU)
   {
	   vectorTools::createDealiiVector(kohnShamDFTEigenOperator.getMatrixFreeData()->get_vector_partitioner(),
					   blockSize,
					   cudaFlattenedArrayBlock);
	   vectorTools::createDealiiVector(kohnShamDFTEigenOperator.getProjectorKetTimesVectorSingle().get_partitioner(),
					    blockSize,
					    projectorKetTimesVectorD);
   }
#endif
   parallelflattened_time = clock() - parallelflattened_time;
 
   for(unsigned int ivec = 0; ivec < numEigenVectors; ivec+=blockSize)
   {
      const unsigned int currentBlockSize=std::min(blockSize,numEigenVectors-ivec);

      if ((currentBlockSize!=blockSize || ivec==0) && !dftParameters::useGPU)
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
          std::vector<std::vector<double>> blockedEigenValues(dftPtr->d_kPointWeights.size(),std::vector<double>(currentBlockSize,0.0));
	  std::vector<std::vector<double>> blockedPartialOccupancies(dftPtr->d_kPointWeights.size(),std::vector<double>(currentBlockSize,0.0));
	  for(unsigned int kPoint = 0; kPoint < dftPtr->d_kPointWeights.size(); ++kPoint)
	     for (unsigned int iWave=0; iWave<currentBlockSize;++iWave)
	     {
		 blockedEigenValues[kPoint][iWave]=dftPtr->eigenValues[kPoint][ivec+iWave];
		 blockedPartialOccupancies[kPoint][iWave]
		     =dftUtils::getPartialOccupancy(blockedEigenValues[kPoint][iWave],
				                                         dftPtr->fermiEnergy,
									 C_kb,
									 dftParameters::TVal);

	     }

          if (dftParameters::useGPU)
          {
#if defined(DFTFE_WITH_GPU) && !defined(USE_COMPLEX)
                  
		  double gpuportedforce_time;
		  gpuportedforce_time = clock();
	    
		  forceCUDA::gpuPortedForceKernelsAll(kohnShamDFTEigenOperator,
                                                      cudaFlattenedArrayBlock,
                                                      projectorKetTimesVectorD,
				                      dftPtr->d_eigenVectorsFlattenedCUDA.begin(),
						      &blockedEigenValues[0][0],
						      &blockedPartialOccupancies[0][0],
						      &nonTrivialIdToElemIdMap[0],
						      &projecterKetTimesFlattenedVectorLocalIds[0],
						      ivec,
						      numEigenVectors,
						      currentBlockSize,
						      numPhysicalCells,
						      numQuadPoints,
						      numQuadPointsNLP,
					              dftPtr->matrix_free_data.get_dofs_per_cell(),
						      nonTrivialNonLocalIdsAllCells.size(),
						      &elocWfcEshelbyTensorQuadValuesH00[0],
						      &elocWfcEshelbyTensorQuadValuesH10[0],
						      &elocWfcEshelbyTensorQuadValuesH11[0],
						      &elocWfcEshelbyTensorQuadValuesH20[0],
						      &elocWfcEshelbyTensorQuadValuesH21[0],
						      &elocWfcEshelbyTensorQuadValuesH22[0],
						      &projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened[0],
						      isPseudopotential,
						      isPseudopotential && dftParameters::useHigherQuadNLP);

		  gpuportedforce_time = clock() - gpuportedforce_time;
		  gpuportedforce_time_total+=gpuportedforce_time;
                 
	          /*
                  
		  double projketpsi_time;
		  projketpsi_time = clock();
		  
		  if (isPseudopotential)
		  {
		      vectorTools::createDealiiVector<double>(dftPtr->d_projectorKetTimesVectorPar[0].get_partitioner(),
							      currentBlockSize,
							      dftPtr->d_projectorKetTimesVectorParFlattened);
	 
		      forceCUDA::computeNonLocalProjectorKetTimesPsiTimesVH(kohnShamDFTEigenOperator,
									    dftPtr->d_eigenVectorsFlattenedCUDA.begin(),
									    ivec,
									    currentBlockSize,
									    numEigenVectors,
									    dftPtr->d_projectorKetTimesVectorParFlattened.begin());
		  }
		  
		  projketpsi_time = clock() - projketpsi_time;
		  projketpsi_time_total+=projketpsi_time;
	 
		  std::vector<double> psiQuadsFlat(numPhysicalCells*numQuadPoints*currentBlockSize);
		  std::vector<double> psiQuadsNLPFlat(numPhysicalCells*numQuadPointsNLP*currentBlockSize);
		  std::vector<double> gradPsiQuadsXFlat(numPhysicalCells*numQuadPoints*currentBlockSize);
		  std::vector<double> gradPsiQuadsYFlat(numPhysicalCells*numQuadPoints*currentBlockSize);
		  std::vector<double> gradPsiQuadsZFlat(numPhysicalCells*numQuadPoints*currentBlockSize);

		  double wfc_time;
		  wfc_time = clock();

		  forceCUDA::interpolatePsiH(kohnShamDFTEigenOperator,
				  dftPtr->d_eigenVectorsFlattenedCUDA.begin(),
				  ivec,
				  currentBlockSize,
				  numEigenVectors,
				  numPhysicalCells,
				  numQuadPoints,
				  numQuadPointsNLP,
				  dftPtr->matrix_free_data.get_dofs_per_cell(),
				  &psiQuadsFlat[0],
				  &psiQuadsNLPFlat[0],
				  &gradPsiQuadsXFlat[0],
				  &gradPsiQuadsYFlat[0],
				  &gradPsiQuadsZFlat[0],
				  isPseudopotential && dftParameters::useHigherQuadNLP);

		  
		  wfc_time = clock() - wfc_time;
		  wfc_time_total+=wfc_time;          

		  double nlppsicontract_time;
		  nlppsicontract_time = clock();


		  if (isPseudopotential)
			forceCUDA::nlpPsiContractionH(kohnShamDFTEigenOperator,
						      dftParameters::useHigherQuadNLP?&psiQuadsNLPFlat[0]:&psiQuadsFlat[0],
						      &blockedPartialOccupancies[0][0],
						      dftPtr->d_projectorKetTimesVectorParFlattened.begin(),
						      &nonTrivialIdToElemIdMap[0],
						      &projecterKetTimesFlattenedVectorLocalIds[0],
						      numPhysicalCells,
						      numQuadPointsNLP,
						      currentBlockSize,
						      nonTrivialNonLocalIdsAllCells.size(),
						      &projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened[0]);
		  nlppsicontract_time = clock() - nlppsicontract_time;
		  nlppsicontract_time_total+=nlppsicontract_time;


		  double eloc_time;
		  eloc_time = clock();
		  forceCUDA::computeELocWfcEshelbyTensorNonPeriodicH(kohnShamDFTEigenOperator,
								       &psiQuadsFlat[0],
								       &gradPsiQuadsXFlat[0],
								       &gradPsiQuadsYFlat[0],
								       &gradPsiQuadsZFlat[0],
								       &blockedEigenValues[0][0],
								       &blockedPartialOccupancies[0][0],
								       numPhysicalCells,
								       numQuadPoints,
								       currentBlockSize,
								       &elocWfcEshelbyTensorQuadValuesH00[0],
								       &elocWfcEshelbyTensorQuadValuesH10[0],
								       &elocWfcEshelbyTensorQuadValuesH11[0],
								       &elocWfcEshelbyTensorQuadValuesH20[0],
								       &elocWfcEshelbyTensorQuadValuesH21[0],
								       &elocWfcEshelbyTensorQuadValuesH22[0]);
		  eloc_time = clock() - eloc_time;
		  eloc_time_total+=eloc_time;
                  */
                  
#endif
	  }
          else
	  {
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

		  double projketpsi_time;
		  projketpsi_time = clock();
		  if (isPseudopotential)
		    for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
		    {
			 computeNonLocalProjectorKetTimesPsiTimesVFlattened(eigenVectorsFlattenedBlock[ikPoint],
									    currentBlockSize,
									    projectorKetTimesPsiTimesVTimesPartOcc[ikPoint],
									    ikPoint,
									    blockedPartialOccupancies[ikPoint]);
		    }
		  projketpsi_time = clock() - projketpsi_time;
		  projketpsi_time_total+=projketpsi_time;


		  for (unsigned int cell=0; cell<matrixFreeData.n_macro_cells(); ++cell)
		  {
		    forceEval.reinit(cell);
#ifdef USE_COMPLEX
		    forceEvalKPoints.reinit(cell);
#endif

		    psiEval.reinit(cell);

		    if (isPseudopotential && dftParameters::useHigherQuadNLP)
		    {
		      forceEvalNLP.reinit(cell);
#ifdef USE_COMPLEX
		      forceEvalKPointsNLP.reinit(cell);
#endif

		      psiEvalNLP.reinit(cell);
		    }

		    const unsigned int numSubCells=matrixFreeData.n_components_filled(cell);
#ifdef USE_COMPLEX
		    std::vector<Tensor<1,2,VectorizedArray<double> > > psiQuads(numQuadPoints*currentBlockSize*numKPoints,zeroTensor1);
		    std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > > gradPsiQuads(numQuadPoints*currentBlockSize*numKPoints,zeroTensor2);
#else
		    std::vector< VectorizedArray<double> > psiQuads(numQuadPoints*currentBlockSize,make_vectorized_array(0.0));
		    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPsiQuads(numQuadPoints*currentBlockSize,zeroTensor3);
#endif
		    double wfc_time;
		    wfc_time = clock();

		    for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
			for (unsigned int iEigenVec=0; iEigenVec<currentBlockSize; ++iEigenVec)
			{
			  psiEval.read_dof_values_plain(eigenVectors[ikPoint][iEigenVec]);
			  psiEval.evaluate(true,true);

			  for (unsigned int q=0; q<numQuadPoints; ++q)
			  {
			     const unsigned int id=q*currentBlockSize*numKPoints+currentBlockSize*ikPoint+iEigenVec;
			     psiQuads[id]=psiEval.get_value(q);
			     gradPsiQuads[id]=psiEval.get_gradient(q);
			  }//quad point loop
			} //eigenvector loop

#ifdef USE_COMPLEX
		    std::vector<Tensor<1,2,VectorizedArray<double> > > psiQuadsNLP;
#else
		    std::vector< VectorizedArray<double> > psiQuadsNLP;
#endif

		    if (isPseudopotential && dftParameters::useHigherQuadNLP)
		    {
#ifdef USE_COMPLEX
			psiQuadsNLP.resize(numQuadPointsNLP*currentBlockSize*numKPoints,zeroTensor1);
#else
			psiQuadsNLP.resize(numQuadPointsNLP*currentBlockSize,make_vectorized_array(0.0));
#endif

			for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
			    for (unsigned int iEigenVec=0; iEigenVec<currentBlockSize; ++iEigenVec)
			    {
			      psiEvalNLP.read_dof_values_plain(eigenVectors[ikPoint][iEigenVec]);
			      psiEvalNLP.evaluate(true,false);

			      for (unsigned int q=0; q<numQuadPointsNLP; ++q)
			      {
				 const unsigned int id=q*currentBlockSize*numKPoints+currentBlockSize*ikPoint+iEigenVec;
				 psiQuadsNLP[id]=psiEvalNLP.get_value(q);
			      }//quad point loop
			    } //eigenvector loop

		    }

		    wfc_time = clock() - wfc_time;
		    wfc_time_total+=wfc_time;

#ifndef USE_COMPLEX
		    double nlppsicontract_time;
		    nlppsicontract_time = clock();
		    const unsigned int numNonLocalAtomsCurrentProc=projectorKetTimesPsiTimesVTimesPartOcc[0].size();
		    std::vector<bool> isAtomInCell(numNonLocalAtomsCurrentProc,false);
		    if (isPseudopotential)
		    {
		      std::vector<unsigned int> nonTrivialNonLocalIds;
		      for (unsigned int iatom=0; iatom<numNonLocalAtomsCurrentProc; ++iatom)
		      {
			 for (unsigned int i=0;i<macroIdToNonlocalAtomsSetMap[cell].size();i++)
			      if (macroIdToNonlocalAtomsSetMap[cell][i]==iatom)
			      {
				  isAtomInCell[iatom]=true;
				  nonTrivialNonLocalIds.push_back(iatom);
				  break;
			      }
		      }

		      
		      if (dftParameters::useHigherQuadNLP)
		      {
			       for (unsigned int q=0; q<numQuadPointsNLP; ++q)
			       {
				  std::vector<VectorizedArray<double> > & temp1= projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuads[cell*numQuadPointsNLP+q];
				  //std::fill(temp1.begin(),temp1.end(),make_vectorized_array(0.0));
				  for (unsigned int i=0; i<nonTrivialNonLocalIds.size(); ++i)
				  {
				       const unsigned int iatom=nonTrivialNonLocalIds[i];
				       const unsigned int numberSingleAtomPseudoWaveFunctions=numPseudoWfcsAtom[iatom];
				       const unsigned int startingId=nonlocalPseudoWfcsAccum[iatom];
				       const std::vector<double> & temp2=projectorKetTimesPsiTimesVTimesPartOcc[0][iatom];
				       for (unsigned int ipsp=0; ipsp<numberSingleAtomPseudoWaveFunctions; ++ipsp) 
					 for (unsigned int iEigenVec=0; iEigenVec<currentBlockSize; ++iEigenVec)
					 {
					     temp1[startingId+ipsp]
					      += psiQuadsNLP[q*currentBlockSize+iEigenVec]
						 *make_vectorized_array(temp2[ipsp*currentBlockSize+iEigenVec]);
					 }
				  }
			      }
		      }
		      else
		      {
			       for (unsigned int q=0; q<numQuadPoints; ++q)
			       {
				  std::vector<VectorizedArray<double> > & temp1= projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuads[cell*numQuadPoints+q];
				  //std::fill(temp1.begin(),temp1.end(),make_vectorized_array(0.0));
				  for (unsigned int i=0; i<nonTrivialNonLocalIds.size(); ++i)
				  {
				       const unsigned int iatom=nonTrivialNonLocalIds[i];
				       const unsigned int numberSingleAtomPseudoWaveFunctions=numPseudoWfcsAtom[iatom];
				       const unsigned int startingId=nonlocalPseudoWfcsAccum[iatom];
				       const std::vector<double> & temp2=projectorKetTimesPsiTimesVTimesPartOcc[0][iatom];
				       for (unsigned int ipsp=0; ipsp<numberSingleAtomPseudoWaveFunctions; ++ipsp) 
					 for (unsigned int iEigenVec=0; iEigenVec<currentBlockSize; ++iEigenVec)
					 {
					     temp1[startingId+ipsp]
					      += psiQuads[q*currentBlockSize+iEigenVec]
						 *make_vectorized_array(temp2[ipsp*currentBlockSize+iEigenVec]);
					 }
				  }
			      }
		      }
		    }
		   
		    nlppsicontract_time = clock() - nlppsicontract_time;
		    nlppsicontract_time_total+=nlppsicontract_time;
#endif

		    if(isPseudopotential)
		    {
		      //compute FnlGammaAtoms  (contibution due to Gamma(Rj)) 
		      double fnlgamma_time;
		      fnlgamma_time = clock();
		      
#ifdef USE_COMPLEX

		       FnlGammaAtomsElementalContributionPeriodic
						     (forceContributionFnlGammaAtoms,
						      forceEval,
						      forceEvalNLP,
						      cell,
						      pspnlGammaAtomsQuads,
						      projectorKetTimesPsiTimesVTimesPartOcc,
						      dftParameters::useHigherQuadNLP?psiQuadsNLP:psiQuads,
						      blockedEigenValues,
						      macroIdToNonlocalAtomsSetMap[cell]);


#else
		       /*
		       FnlGammaAtomsElementalContributionNonPeriodic
						     (forceContributionFnlGammaAtoms,
						      forceEval,
						      forceEvalNLP,
						      cell,
						      gradZetaDeltaVQuads,
						      projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuads,
						      isAtomInCell,
						      nonlocalPseudoWfcsAccum);
		       */
#endif
		       
		       fnlgamma_time = clock() - fnlgamma_time;
		       fnlgamma_time_total+=fnlgamma_time;
		    }//is pseudopotential check

		    double eloc_time;
		    eloc_time = clock();
		    for (unsigned int q=0; q<numQuadPoints; ++q)
		    {
		       Tensor<2,C_DIM,VectorizedArray<double> > E=zeroTensor4;

#ifdef USE_COMPLEX
		       Tensor<2,C_DIM,VectorizedArray<double> > EKPoints=eshelbyTensor::getELocWfcEshelbyTensorPeriodicKPoints
									     (psiQuads.begin()+q*currentBlockSize*numKPoints,
									      gradPsiQuads.begin()+q*currentBlockSize*numKPoints,
									      dftPtr->d_kPointCoordinates,
									      dftPtr->d_kPointWeights,
									      blockedEigenValues,
									      dftPtr->fermiEnergy,
									      dftParameters::TVal);
#else
		       E+=eshelbyTensor::getELocWfcEshelbyTensorNonPeriodic
							 (psiQuads.begin()+q*currentBlockSize,
							 gradPsiQuads.begin()+q*currentBlockSize,
							 blockedEigenValues[0],
							 blockedPartialOccupancies[0]);
#endif
		       Tensor<1,C_DIM,VectorizedArray<double> > F=zeroTensor3;

		       if(isPseudopotential)
		       {
			   if (!dftParameters::useHigherQuadNLP)
			   {
#ifdef USE_COMPLEX
			       Tensor<1,C_DIM,VectorizedArray<double> > FKPoints;
			       Tensor<2,C_DIM,VectorizedArray<double> > EnlKPoints;

			       eshelbyTensor::getFnlEnlMergedPeriodic(gradZetaDeltaVQuads[cell*numQuadPoints+q],
									 ZetaDeltaVQuads[cell*numQuadPoints+q],
									 projectorKetTimesPsiTimesVTimesPartOcc,
									 psiQuads.begin()+q*currentBlockSize*numKPoints,
									 dftPtr->d_kPointWeights,
									 currentBlockSize,
									 macroIdToNonlocalAtomsSetMap[cell],
									 FKPoints,
									 EnlKPoints);
			       EKPoints+=EnlKPoints;
			       forceEvalKPoints.submit_value(FKPoints,q);
#else
			       /*
			       Tensor<1,C_DIM,VectorizedArray<double> > Fnl;
			       Tensor<2,C_DIM,VectorizedArray<double> >	Enl;

			       eshelbyTensor::getFnlEnlMergedNonPeriodic(gradZetaDeltaVQuads[cell*numQuadPoints+q],
									 ZetaDeltaVQuads[cell*numQuadPoints+q],
									 projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuads[q],
									 isAtomInCell,
									 nonlocalPseudoWfcsAccum,
									 Fnl,
									 Enl);
			       F+=Fnl;
			       E+=Enl;
			       */
#endif
			   }

		       }

		       forceEval.submit_value(F,q);
		       forceEval.submit_gradient(E,q);
#ifdef USE_COMPLEX
		       forceEvalKPoints.submit_gradient(EKPoints,q);
#endif
		    }//quad point loop
		    eloc_time = clock() - eloc_time;
		    eloc_time_total+=eloc_time;

		    double enlfnl_time;
		    enlfnl_time = clock();

		    if (isPseudopotential && dftParameters::useHigherQuadNLP)
			for (unsigned int q=0; q<numQuadPointsNLP; ++q)
			{
#ifdef USE_COMPLEX
			       Tensor<1,C_DIM,VectorizedArray<double> > FKPoints;
			       Tensor<2,C_DIM,VectorizedArray<double> > EKPoints;

			       eshelbyTensor::getFnlEnlMergedPeriodic(gradZetaDeltaVQuads[cell*numQuadPointsNLP+q],
								      ZetaDeltaVQuads[cell*numQuadPointsNLP+q],
								      projectorKetTimesPsiTimesVTimesPartOcc,
								      psiQuadsNLP.begin()+q*currentBlockSize*numKPoints,
								      dftPtr->d_kPointWeights,
								      currentBlockSize,
								      macroIdToNonlocalAtomsSetMap[cell],
								      FKPoints,
								      EKPoints);
			       forceEvalKPointsNLP.submit_value(FKPoints,q);
			       forceEvalKPointsNLP.submit_gradient(EKPoints,q);
#else
			       /*
			       Tensor<1,C_DIM,VectorizedArray<double> > F;
			       Tensor<2,C_DIM,VectorizedArray<double> >	E;

			       eshelbyTensor::getFnlEnlMergedNonPeriodic(gradZetaDeltaVQuads[cell*numQuadPointsNLP+q],
									 ZetaDeltaVQuads[cell*numQuadPointsNLP+q],
									 projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuads[q],
									 isAtomInCell,
									 nonlocalPseudoWfcsAccum,
									 F,
									 E);

			       forceEvalNLP.submit_value(F,q);
			       forceEvalNLP.submit_gradient(E,q);
			       */
#endif
			}//nonlocal psp quad points loop

		    enlfnl_time = clock() - enlfnl_time;
		    enlfnl_time_total+=enlfnl_time;

		    forceEval.integrate(true,true);

		    if(isPseudopotential)
		    {
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
			  //forceEvalNLP.integrate(true,true);
#endif
		       }
		    }
		    else
		    {
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
			//forceEvalNLP.distribute_local_to_global(d_configForceVectorLinFE);
#endif
		    }
		  }//macro cell loop
	}
      }//band parallelization loop
  }//wavefunction block loop

  double enowfc_time;
  enowfc_time = clock();
#if defined(DFTFE_WITH_GPU) && !defined(USE_COMPLEX)
  if (dftParameters::useGPU)
  {
	  for (unsigned int cell=0; cell<matrixFreeData.n_macro_cells(); ++cell)
	  {
	     forceEval.reinit(cell);

	     for (unsigned int q=0; q<numQuadPoints; ++q)
	     {
	       Tensor<2,C_DIM,VectorizedArray<double> > E;
	       const unsigned int physicalCellId=macroCellIdToNormalCellIdMap[cell];
	       const unsigned int id=physicalCellId*numQuadPoints+q;
	       E[0][0]=make_vectorized_array(elocWfcEshelbyTensorQuadValuesH00[id]);
	       E[1][0]=make_vectorized_array(elocWfcEshelbyTensorQuadValuesH10[id]);
	       E[1][1]=make_vectorized_array(elocWfcEshelbyTensorQuadValuesH11[id]);
	       E[2][0]=make_vectorized_array(elocWfcEshelbyTensorQuadValuesH20[id]);
	       E[2][1]=make_vectorized_array(elocWfcEshelbyTensorQuadValuesH21[id]);
	       E[2][2]=make_vectorized_array(elocWfcEshelbyTensorQuadValuesH22[id]);
	       E[0][1]=E[1][0];
	       E[0][2]=E[2][0];
	       E[1][2]=E[2][1];
	       forceEval.submit_gradient(E,q);

	     }//quad point loop
	     forceEval.integrate(false,true);
	     forceEval.distribute_local_to_global(d_configForceVectorLinFE);
	  }

	  if (isPseudopotential)
		  for (unsigned int i=0; i<nonTrivialNonLocalIdsAllCells.size(); ++i)
		  {
			 const unsigned int cell=normalCellIdToMacroCellIdMap[nonTrivialIdToElemIdMap[i]];
			 const unsigned int id=nonTrivialIdToAllPseudoWfcIdMap[i];
			 for (unsigned int q=0; q<numQuadPointsNLP; ++q)
			     projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuads[cell*numQuadPointsNLP+q][id]=make_vectorized_array(projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuadsFlattened[i*numQuadPointsNLP+q]);

		  }
  }
#endif

#ifndef USE_COMPLEX
  if (isPseudopotential)
          for (unsigned int cell=0; cell<matrixFreeData.n_macro_cells(); ++cell)
	  {
	    if (dftParameters::useHigherQuadNLP)
	      forceEvalNLP.reinit(cell);
	    else
	      forceEval.reinit(cell);

	    const unsigned int numNonLocalAtomsCurrentProc=dftPtr->d_nonLocalAtomIdsInCurrentProcess.size();
	    std::vector<bool> isAtomInCell(numNonLocalAtomsCurrentProc,false);
	      
	    std::vector<unsigned int> nonTrivialNonLocalIds;
	    for (unsigned int iatom=0; iatom<numNonLocalAtomsCurrentProc; ++iatom)
	    {
		 for (unsigned int i=0;i<macroIdToNonlocalAtomsSetMap[cell].size();i++)
		      if (macroIdToNonlocalAtomsSetMap[cell][i]==iatom)
		      {
			  isAtomInCell[iatom]=true;
			  nonTrivialNonLocalIds.push_back(iatom);
			  break;
		      }
	    }

	    //compute FnlGammaAtoms  (contibution due to Gamma(Rj)) 
	    double fnlgamma_time;
	    fnlgamma_time = clock();
	      
	    FnlGammaAtomsElementalContributionNonPeriodic
					     (forceContributionFnlGammaAtoms,
					      forceEval,
					      forceEvalNLP,
					      cell,
					      gradZetaDeltaVQuads,
					      projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuads,
					      isAtomInCell,
					      nonlocalPseudoWfcsAccum);
	       
	    fnlgamma_time = clock() - fnlgamma_time;
	    fnlgamma_time_total+=fnlgamma_time;

	    double enlfnl_time;
	    enlfnl_time = clock();

            Tensor<1,C_DIM,VectorizedArray<double> > F;
            Tensor<2,C_DIM,VectorizedArray<double> > E;

	    for (unsigned int q=0; q<numQuadPointsNLP; ++q)
	    {

	       Tensor<1,C_DIM,VectorizedArray<double> > F;
	       Tensor<2,C_DIM,VectorizedArray<double> >	E;

	       eshelbyTensor::getFnlEnlMergedNonPeriodic(gradZetaDeltaVQuads[cell*numQuadPointsNLP+q],
							 ZetaDeltaVQuads[cell*numQuadPointsNLP+q],
							 projectorKetTimesPsiTimesVTimesPartOccContractionPsiQuads[cell*numQuadPointsNLP+q],
							 isAtomInCell,
							 nonlocalPseudoWfcsAccum,
							 F,
							 E);
               if (dftParameters::useHigherQuadNLP)
               {
	          forceEvalNLP.submit_value(F,q);
	          forceEvalNLP.submit_gradient(E,q);
               }
               else
               {
		   forceEval.submit_value(F,q);
		   forceEval.submit_gradient(E,q);
               }
	    }//nonlocal psp quad points loop

	    enlfnl_time = clock() - enlfnl_time;
	    enlfnl_time_total+=enlfnl_time;


	    if (dftParameters::useHigherQuadNLP)
		  forceEvalNLP.integrate(true,true);
	    else
		  forceEval.integrate(true,true); 


	    if (dftParameters::useHigherQuadNLP)
		forceEvalNLP.distribute_local_to_global(d_configForceVectorLinFE);
	    else
		forceEval.distribute_local_to_global(d_configForceVectorLinFE);
	  }
#endif

  // add global Fnl contribution due to Gamma(Rj) to the configurational force vector
  if(isPseudopotential)
  {
     distributeForceContributionFnlGammaAtoms(forceContributionFnlGammaAtoms);
  }


  /////////// Compute contribution independent of wavefunctions /////////////////
  if (bandGroupTaskId==0)
  {
      std::vector<VectorizedArray<double> > rhoQuads(numQuadPoints,make_vectorized_array(0.0));
      std::vector<VectorizedArray<double> > derVxcWithRhoOutTimesRhoDiffQuads(numQuadPoints,make_vectorized_array(0.0));
      std::vector<VectorizedArray<double> > phiRhoMinMinusApproxRhoQuads(numQuadPoints,make_vectorized_array(0.0));
      std::vector<VectorizedArray<double> > shadowKSRhoMinMinusRhoQuads(numQuadPoints,make_vectorized_array(0.0));
      std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > shadowKSGradRhoMinMinusGradRhoQuads(numQuadPoints,zeroTensor3);
      std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoQuads(numQuadPoints,zeroTensor3);
      std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoAtomsQuads(numQuadPoints,zeroTensor3);
      std::vector<Tensor<2,C_DIM,VectorizedArray<double> > > hessianRhoQuads(numQuadPoints,zeroTensor4);
      std::vector<Tensor<2,C_DIM,VectorizedArray<double> > > hessianRhoAtomsQuads(numQuadPoints,zeroTensor4);
      std::vector<VectorizedArray<double> > excQuads(numQuadPoints,make_vectorized_array(0.0));
      std::vector<VectorizedArray<double> > pseudoVLocQuads(numQuadPoints,make_vectorized_array(0.0));
      std::vector<VectorizedArray<double> > vxcRhoInQuads(numQuadPoints,make_vectorized_array(0.0));
      std::vector<VectorizedArray<double> > vxcRhoOutQuads(numQuadPoints,make_vectorized_array(0.0));
      std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > derExchCorrEnergyWithGradRhoInQuads(numQuadPoints,zeroTensor3);
      std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > derExchCorrEnergyWithGradRhoOutQuads(numQuadPoints,zeroTensor3);
      std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > derVxcWithGradRhoOutQuads(numQuadPoints,zeroTensor3);
      std::vector<VectorizedArray<double> > derVxcWithRhoOutQuads(numQuadPoints,make_vectorized_array(0.0));
      std::vector<Tensor<2,C_DIM,VectorizedArray<double> > > der2ExcWithGradRhoOutQuads(numQuadPoints,zeroTensor4);
      for (unsigned int cell=0; cell<matrixFreeData.n_macro_cells(); ++cell)
      {
	forceEval.reinit(cell);

	if (d_isElectrostaticsMeshSubdivided)
	{
	  phiTotOutEval.reinit(cell);
	  phiTotOutEval.read_dof_values_plain(phiTotRhoOut);
	  phiTotOutEval.evaluate(true,false);

	  //phiExtEval.reinit(cell);
	  //phiExtEval.read_dof_values_plain(phiExt);
	  //phiExtEval.evaluate(true,false);
	}

        if  (shadowPotentialForce)
        {
           phiTotEval2.reinit(cell);
           phiTotEval2.read_dof_values_plain(phiRhoMinusApproxRho);
           phiTotEval2.evaluate(true,false);
        }

	std::fill(rhoQuads.begin(),rhoQuads.end(),make_vectorized_array(0.0));
        std::fill(derVxcWithRhoOutTimesRhoDiffQuads.begin(),derVxcWithRhoOutTimesRhoDiffQuads.end(),make_vectorized_array(0.0));
        std::fill(phiRhoMinMinusApproxRhoQuads.begin(),phiRhoMinMinusApproxRhoQuads.end(),make_vectorized_array(0.0));
        std::fill(shadowKSRhoMinMinusRhoQuads.begin(),shadowKSRhoMinMinusRhoQuads.end(),make_vectorized_array(0.0));
        std::fill(shadowKSGradRhoMinMinusGradRhoQuads.begin(),shadowKSGradRhoMinMinusGradRhoQuads.end(),zeroTensor3);  
	std::fill(gradRhoQuads.begin(),gradRhoQuads.end(),zeroTensor3);
        std::fill(gradRhoAtomsQuads.begin(),gradRhoAtomsQuads.end(),zeroTensor3);
	std::fill(hessianRhoQuads.begin(),hessianRhoQuads.end(),zeroTensor4);
        std::fill(hessianRhoAtomsQuads.begin(),hessianRhoAtomsQuads.end(),zeroTensor4);
	std::fill(excQuads.begin(),excQuads.end(),make_vectorized_array(0.0));
	std::fill(pseudoVLocQuads.begin(),pseudoVLocQuads.end(),make_vectorized_array(0.0));
	std::fill(vxcRhoInQuads.begin(),vxcRhoInQuads.end(),make_vectorized_array(0.0));
	std::fill(vxcRhoOutQuads.begin(),vxcRhoOutQuads.end(),make_vectorized_array(0.0));
	std::fill(derExchCorrEnergyWithGradRhoInQuads.begin(),derExchCorrEnergyWithGradRhoInQuads.end(),zeroTensor3);
	std::fill(derExchCorrEnergyWithGradRhoOutQuads.begin(),derExchCorrEnergyWithGradRhoOutQuads.end(),zeroTensor3);
        std::fill(derVxcWithGradRhoOutQuads.begin(),derVxcWithGradRhoOutQuads.end(),zeroTensor3);
        std::fill(derVxcWithRhoOutQuads.begin(),derVxcWithRhoOutQuads.end(),make_vectorized_array(0.0));
        std::fill(der2ExcWithGradRhoOutQuads.begin(),der2ExcWithGradRhoOutQuads.end(),zeroTensor4);

	const unsigned int numSubCells=matrixFreeData.n_components_filled(cell);
	//For LDA
	std::vector<double> exchValRhoOut(numQuadPoints);
	std::vector<double> corrValRhoOut(numQuadPoints);
	std::vector<double> exchPotValRhoOut(numQuadPoints);
	std::vector<double> corrPotValRhoOut(numQuadPoints);
	std::vector<double> exchValRhoIn(numQuadPoints);
	std::vector<double> corrValRhoIn(numQuadPoints);
	std::vector<double> exchPotValRhoIn(numQuadPoints);
	std::vector<double> corrPotValRhoIn(numQuadPoints);
	//
	//For GGA
	std::vector<double> sigmaValRhoOut(numQuadPoints);
	std::vector<double> derExchEnergyWithDensityValRhoOut(numQuadPoints), derCorrEnergyWithDensityValRhoOut(numQuadPoints), derExchEnergyWithSigmaRhoOut(numQuadPoints),derCorrEnergyWithSigmaRhoOut(numQuadPoints);
	std::vector<double> sigmaValRhoIn(numQuadPoints);
	std::vector<double> derExchEnergyWithDensityValRhoIn(numQuadPoints), derCorrEnergyWithDensityValRhoIn(numQuadPoints), derExchEnergyWithSigmaRhoIn(numQuadPoints),derCorrEnergyWithSigmaRhoIn(numQuadPoints);
	std::vector<Tensor<1,C_DIM,double > > gradRhoIn(numQuadPoints);
	std::vector<Tensor<1,C_DIM,double > > gradRhoOut(numQuadPoints);
        std::vector<double> derVxWithSigmaRhoOut(numQuadPoints);
        std::vector<double> derVcWithSigmaRhoOut(numQuadPoints);
        std::vector<double> der2ExWithSigmaRhoOut(numQuadPoints);
        std::vector<double> der2EcWithSigmaRhoOut(numQuadPoints);
        std::vector<double> derVxWithRhoOut(numQuadPoints);
        std::vector<double> derVcWithRhoOut(numQuadPoints);

	//
	for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
	{
	   subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell);
	   dealii::CellId subCellId=subCellPtr->id();
	   if(dftParameters::xc_id == 4)
	   {
	      for (unsigned int q = 0; q < numQuadPoints; ++q)
	      {
		  gradRhoOut[q][0] = gradRhoOutValues.find(subCellId)->second[3*q + 0];
		  gradRhoOut[q][1] = gradRhoOutValues.find(subCellId)->second[3*q + 1];
		  gradRhoOut[q][2] = gradRhoOutValues.find(subCellId)->second[3*q + 2];
		  sigmaValRhoOut[q] = gradRhoOut[q].norm_square();

		  gradRhoIn[q][0] = ((*dftPtr->gradRhoInValues)[subCellId][3*q + 0]);
		  gradRhoIn[q][1] = ((*dftPtr->gradRhoInValues)[subCellId][3*q + 1]);
		  gradRhoIn[q][2] = ((*dftPtr->gradRhoInValues)[subCellId][3*q + 2]);
		  sigmaValRhoIn[q] = gradRhoIn[q].norm_square();
	      }
	      xc_gga_exc_vxc(&(dftPtr->funcX),numQuadPoints,&(rhoOutValues.find(subCellId)->second[0]),&sigmaValRhoOut[0],&exchValRhoOut[0],&derExchEnergyWithDensityValRhoOut[0],&derExchEnergyWithSigmaRhoOut[0]);
	      xc_gga_exc_vxc(&(dftPtr->funcC),numQuadPoints,&(rhoOutValues.find(subCellId)->second[0]),&sigmaValRhoOut[0],&corrValRhoOut[0],&derCorrEnergyWithDensityValRhoOut[0],&derCorrEnergyWithSigmaRhoOut[0]);
	      xc_gga_exc_vxc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoInValues)[subCellId][0]),&sigmaValRhoIn[0],&exchValRhoIn[0],&derExchEnergyWithDensityValRhoIn[0],&derExchEnergyWithSigmaRhoIn[0]);
	      xc_gga_exc_vxc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoInValues)[subCellId][0]),&sigmaValRhoIn[0],&corrValRhoIn[0],&derCorrEnergyWithDensityValRhoIn[0],&derCorrEnergyWithSigmaRhoIn[0]);
              if  (shadowPotentialForce)
              {
                 xc_gga_fxc(&(dftPtr->funcX),numQuadPoints,&(rhoOutValues.find(subCellId)->second[0]),&sigmaValRhoOut[0], &derVxWithRhoOut[0], &derVxWithSigmaRhoOut[0],  &der2ExWithSigmaRhoOut[0]);
                 xc_gga_fxc(&(dftPtr->funcC),numQuadPoints,&(rhoOutValues.find(subCellId)->second[0]),&sigmaValRhoOut[0], &derVcWithRhoOut[0], &derVcWithSigmaRhoOut[0],  &der2EcWithSigmaRhoOut[0]);              
              }

	      for (unsigned int q=0; q<numQuadPoints; ++q)
	      {
		 excQuads[q][iSubCell]=exchValRhoOut[q]+corrValRhoOut[q];
		 const double temp = derExchEnergyWithSigmaRhoOut[q]+derCorrEnergyWithSigmaRhoOut[q];
		 vxcRhoInQuads[q][iSubCell]= derExchEnergyWithDensityValRhoIn[q]+derCorrEnergyWithDensityValRhoIn[q];
		 vxcRhoOutQuads[q][iSubCell]= derExchEnergyWithDensityValRhoOut[q]+derCorrEnergyWithDensityValRhoOut[q];

                 if  (shadowPotentialForce)
                   derVxcWithRhoOutQuads[q][iSubCell]=derVxWithRhoOut[q]+derVcWithRhoOut[q];

		  for (unsigned int idim=0; idim<C_DIM; idim++)
		  {
		     derExchCorrEnergyWithGradRhoInQuads[q][idim][iSubCell]=2.0*(derExchEnergyWithSigmaRhoIn[q]+derCorrEnergyWithSigmaRhoIn[q])*gradRhoIn[q][idim];
		     derExchCorrEnergyWithGradRhoOutQuads[q][idim][iSubCell]=2.0*(derExchEnergyWithSigmaRhoOut[q]+derCorrEnergyWithSigmaRhoOut[q])*gradRhoOut[q][idim];
 
                     if  (shadowPotentialForce)
                     {
		        derVxcWithGradRhoOutQuads[q][idim][iSubCell]=2.0*(derVxWithSigmaRhoOut[q]+derVcWithSigmaRhoOut[q])*gradRhoOut[q][idim];

                        for (unsigned int jdim=0; jdim<C_DIM; jdim++)
                        {
                           if (idim==jdim)
                            der2ExcWithGradRhoOutQuads[q][idim][idim][iSubCell]=2.0*(derExchEnergyWithSigmaRhoOut[q]+derCorrEnergyWithSigmaRhoOut[q])
                                                                      +4.0*(der2ExWithSigmaRhoOut[q]+der2EcWithSigmaRhoOut[q])*gradRhoOut[q][idim]*gradRhoOut[q][idim];
                           else
                            der2ExcWithGradRhoOutQuads[q][idim][jdim][iSubCell]=4.0*(der2ExWithSigmaRhoOut[q]+der2EcWithSigmaRhoOut[q])*gradRhoOut[q][idim]*gradRhoOut[q][jdim];
                        }
                     }
		  }
	      }

	   }
	   else
	   {
	      xc_lda_exc(&(dftPtr->funcX),numQuadPoints,&(rhoOutValues.find(subCellId)->second[0]),&exchValRhoOut[0]);
	      xc_lda_exc(&(dftPtr->funcC),numQuadPoints,&(rhoOutValues.find(subCellId)->second[0]),&corrValRhoOut[0]);
	      xc_lda_vxc(&(dftPtr->funcX),numQuadPoints,&(rhoOutValues.find(subCellId)->second[0]),&exchPotValRhoOut[0]);
	      xc_lda_vxc(&(dftPtr->funcC),numQuadPoints,&(rhoOutValues.find(subCellId)->second[0]),&corrPotValRhoOut[0]);
	      xc_lda_vxc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoInValues)[subCellId][0]),&exchPotValRhoIn[0]);
	      xc_lda_vxc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoInValues)[subCellId][0]),&corrPotValRhoIn[0]);
              xc_lda_fxc(&(dftPtr->funcX),numQuadPoints,&(rhoOutValues.find(subCellId)->second[0]),&derVxWithRhoOut[0]);
              xc_lda_fxc(&(dftPtr->funcC),numQuadPoints,&(rhoOutValues.find(subCellId)->second[0]),&derVcWithRhoOut[0]);
	      for (unsigned int q=0; q<numQuadPoints; ++q)
	      {
		 excQuads[q][iSubCell]=exchValRhoOut[q]+corrValRhoOut[q];
		 vxcRhoInQuads[q][iSubCell]= exchPotValRhoIn[q]+corrPotValRhoIn[q];
		 vxcRhoOutQuads[q][iSubCell]= exchPotValRhoOut[q]+corrPotValRhoOut[q];

                 if  (shadowPotentialForce)
                   derVxcWithRhoOutQuads[q][iSubCell]=derVxWithRhoOut[q]+derVcWithRhoOut[q];
	      }
	   }

	   for (unsigned int q=0; q<numQuadPoints; ++q)
	   {
	     rhoQuads[q][iSubCell]=rhoOutValues.find(subCellId)->second[q];

             if (shadowPotentialForce)
             {
                shadowKSRhoMinMinusRhoQuads[q][iSubCell]=shadowKSRhoMinValues.find(subCellId)->second[q]-rhoQuads[q][iSubCell];
  		    
                for (unsigned int idim=0; idim<C_DIM; idim++)
                    gradRhoAtomsQuads[q][idim][iSubCell]=dftPtr->d_gradRhoAtomsValues.find(subCellId)->second[3*q+idim];

                if(dftParameters::xc_id == 4)
                {
                    for (unsigned int idim=0; idim<C_DIM; idim++)
                       for (unsigned int jdim=0; jdim<C_DIM; jdim++)
                          hessianRhoAtomsQuads[q][idim][jdim][iSubCell]=dftPtr->d_hessianRhoAtomsValues.find(subCellId)->second[9*q+3*idim+jdim];

  		    for (unsigned int idim=0; idim<C_DIM; idim++)
                        shadowKSGradRhoMinMinusGradRhoQuads[q][idim][iSubCell]=shadowKSGradRhoMinValues.find(subCellId)->second[3*q+idim]
                                                                          -gradRhoOutValues.find(subCellId)->second[3*q+idim];
                }
             }

             if(dftParameters::xc_id == 4)
		 for (unsigned int idim=0; idim<C_DIM; idim++)
		    gradRhoQuads[q][idim][iSubCell]=gradRhoOutValues.find(subCellId)->second[3*q+idim];
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
	   //const VectorizedArray<double> phiExt_q =d_isElectrostaticsMeshSubdivided?
	   //					    phiExtEval.get_value(q)
	   //					    :make_vectorized_array(0.0);

           if (shadowPotentialForce)
           {
              derVxcWithRhoOutTimesRhoDiffQuads[q]=derVxcWithRhoOutQuads[q]*shadowKSRhoMinMinusRhoQuads[q];
              phiRhoMinMinusApproxRhoQuads[q]= phiTotEval2.get_value(q);
           }


	   Tensor<2,C_DIM,VectorizedArray<double> > E=eshelbyTensor::getELocXcEshelbyTensor
						      (rhoQuads[q],
						      gradRhoQuads[q],
						      excQuads[q],
						      derExchCorrEnergyWithGradRhoOutQuads[q]);
           if (shadowPotentialForce)
               E+=eshelbyTensor::getShadowPotentialForceRhoDiffXcEshelbyTensor
			         (shadowKSRhoMinMinusRhoQuads[q],
                                  shadowKSGradRhoMinMinusGradRhoQuads[q],
				  gradRhoQuads[q],
				  vxcRhoOutQuads[q],
				  derVxcWithGradRhoOutQuads[q],
                                  derExchCorrEnergyWithGradRhoOutQuads[q],
                                  der2ExcWithGradRhoOutQuads[q]);

	   Tensor<1,C_DIM,VectorizedArray<double> > F=zeroTensor3;

           if (shadowPotentialForce && dftParameters::useAtomicRhoXLBOMD)
           {
              F+=gradRhoAtomsQuads[q]*(derVxcWithRhoOutTimesRhoDiffQuads[q]+phiRhoMinMinusApproxRhoQuads[q]);

              if(dftParameters::xc_id == 4)
              {
                  F+=shadowKSGradRhoMinMinusGradRhoQuads[q]*der2ExcWithGradRhoOutQuads[q]*hessianRhoAtomsQuads[q];
                  F+=shadowKSGradRhoMinMinusGradRhoQuads[q]*outer_product(derVxcWithGradRhoOutQuads[q],gradRhoAtomsQuads[q]);
                  F+=shadowKSRhoMinMinusRhoQuads[q]*derVxcWithGradRhoOutQuads[q]*hessianRhoAtomsQuads[q];
              }
           }

	   if(d_isElectrostaticsMeshSubdivided)
	       F-=gradRhoQuads[q]*phiTot_q;

	   if(isPseudopotential && d_isElectrostaticsMeshSubdivided)
		  F-=gradRhoQuads[q]*(pseudoVLocQuads[q]);//-phiExt_q);

	   forceEval.submit_value(F,q);
	   forceEval.submit_gradient(E,q);
	}//quad point loop


        
        if(shadowPotentialForce && dftParameters::useAtomicRhoXLBOMD)
            FShadowLocalGammaAtomsElementalContribution(forceContributionShadowLocalGammaAtoms,
						forceEval,
						matrixFreeData,
						cell,
						dftPtr->d_gradRhoAtomsValuesSeparate,
                                                derVxcWithRhoOutTimesRhoDiffQuads,
                                                phiRhoMinMinusApproxRhoQuads,
                                                dftPtr->d_hessianRhoAtomsValuesSeparate,
                                                der2ExcWithGradRhoOutQuads,
                                                derVxcWithGradRhoOutQuads,
                                                shadowKSGradRhoMinMinusGradRhoQuads,
                                                shadowKSRhoMinMinusRhoQuads);
        


	forceEval.integrate(true,true);
	forceEval.distribute_local_to_global(d_configForceVectorLinFE);//also takes care of constraints
      }//cell loop

      
      if (shadowPotentialForce && dftParameters::useAtomicRhoXLBOMD)
         distributeForceContributionFPSPLocalGammaAtoms(forceContributionShadowLocalGammaAtoms,
	                                                d_atomsForceDofs,
						        d_constraintsNoneForce,
						        d_configForceVectorLinFE);
      
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
			     vselfBinsManagerElectro,
                             shadowKSRhoMinValues,
                             phiRhoMinusApproxRho,
                             shadowPotentialForce);
    }
    
    enowfc_time = clock() - enowfc_time;
 
    if (dftParameters::verbosity>=1)
    {
        pcout<<" Time taken for wfc interpolation in force: "<<dealii::Utilities::MPI::max(wfc_time_total/CLOCKS_PER_SEC,mpi_communicator)<<std::endl;
        pcout<<" Time taken for fnl gamma in force: "<<dealii::Utilities::MPI::max(fnlgamma_time_total/CLOCKS_PER_SEC,mpi_communicator)<<std::endl;
        pcout<<" Time taken for enl fnl in force: "<<dealii::Utilities::MPI::max(enlfnl_time_total/CLOCKS_PER_SEC,mpi_communicator)<<std::endl;
        pcout<<" Time taken for eloc in force: "<<dealii::Utilities::MPI::max(eloc_time_total/CLOCKS_PER_SEC,mpi_communicator)<<std::endl;
        pcout<<" Time taken for non wfc in force: "<<dealii::Utilities::MPI::max(enowfc_time/CLOCKS_PER_SEC,mpi_communicator)<<std::endl;
        pcout<<" Time taken for nlp init in force: "<<dealii::Utilities::MPI::max(nlpinit_time_total/CLOCKS_PER_SEC,mpi_communicator)<<std::endl;
        pcout<<" Time taken for projector ket times psi in force: "<<dealii::Utilities::MPI::max(projketpsi_time_total/CLOCKS_PER_SEC,mpi_communicator)<<std::endl;
        pcout<<" Time taken for nlp psi contraction in force: "<<dealii::Utilities::MPI::max(nlppsicontract_time_total/CLOCKS_PER_SEC,mpi_communicator)<<std::endl;
        pcout<<" Time taken for gpu ported force computation: "<<dealii::Utilities::MPI::max(gpuportedforce_time_total/CLOCKS_PER_SEC,mpi_communicator)<<std::endl;
        pcout<<" Time taken for parallel flattened cuda vectors creation: "<<dealii::Utilities::MPI::max(parallelflattened_time/CLOCKS_PER_SEC,mpi_communicator)<<std::endl;
    }
}

template<unsigned int FEOrder>
void forceClass<FEOrder>::computeConfigurationalForceEEshelbyEElectroPhiTot
			(const MatrixFree<3,double> & matrixFreeDataElectro,
		         const unsigned int phiTotDofHandlerIndexElectro,
		         const unsigned int phiExtDofHandlerIndexElectro,
		         const vectorType & phiTotRhoOutElectro,
		         const vectorType & phiExtElectro,
			 const std::map<dealii::CellId, std::vector<double> > & rhoOutValuesElectro,
			 const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValuesElectro,
		         const std::map<dealii::CellId, std::vector<double> > & pseudoVLocElectro,
		         const std::map<dealii::CellId, std::vector<double> > & gradPseudoVLocElectro,
			 const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & gradPseudoVLocAtomsElectro,
			 const vselfBinsManager<FEOrder> & vselfBinsManagerElectro,
                         const std::map<dealii::CellId, std::vector<double> > & shadowKSRhoMinValues,
                         const vectorType & phiRhoMinusApproxRho,
                         const bool shadowPotentialForce)
{
  FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  forceEvalElectro(matrixFreeDataElectro,
	                                                        d_forceDofHandlerIndexElectro,
								0);

  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiTotEvalElectro(matrixFreeDataElectro,
	                                                          phiTotDofHandlerIndexElectro,
								  0);

  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiTotEvalElectro2(matrixFreeDataElectro,
	                                                          phiTotDofHandlerIndexElectro,
								  0);

  //FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiExtEvalElectro(matrixFreeDataElectro,
  //	                                                          phiExtDofHandlerIndexElectro,
  //								  0);

  std::map<unsigned int, std::vector<double> > forceContributionFPSPLocalGammaAtoms;

  const unsigned int numQuadPoints=forceEvalElectro.n_q_points;
  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;

  QGauss<C_DIM>  quadrature(C_num1DQuad<FEOrder>());
  FEValues<C_DIM> feVselfValuesElectro (matrixFreeDataElectro.
	                                get_dof_handler(phiExtDofHandlerIndexElectro).get_fe(),
	                                quadrature,
				        update_gradients | update_quadrature_points);

  Tensor<1,C_DIM,VectorizedArray<double> > zeroTensor;
  for (unsigned int idim=0; idim<C_DIM; idim++)
  {
    zeroTensor[idim]=make_vectorized_array(0.0);
  }

  std::vector<VectorizedArray<double> > rhoQuadsElectro(numQuadPoints,make_vectorized_array(0.0));
  std::vector<VectorizedArray<double> > shadowKSRhoMinQuadsElectro(numQuadPoints,make_vectorized_array(0.0));
  std::vector<VectorizedArray<double> > shadowKSRhoMinMinusRhoQuadsElectro(numQuadPoints,make_vectorized_array(0.0));
  std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoQuadsElectro(numQuadPoints,zeroTensor);
  std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoAtomsQuadsElectro(numQuadPoints,zeroTensor);
  std::vector<VectorizedArray<double> > pseudoVLocQuadsElectro(numQuadPoints,make_vectorized_array(0.0));
  std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPseudoVLocQuadsElectro(numQuadPoints,zeroTensor);

  for (unsigned int cell=0; cell<matrixFreeDataElectro.n_macro_cells(); ++cell)
  {
    forceEvalElectro.reinit(cell);

    phiTotEvalElectro.reinit(cell);
    phiTotEvalElectro.read_dof_values_plain(phiTotRhoOutElectro);
    phiTotEvalElectro.evaluate(true,true);

    if (shadowPotentialForce)
    {
       phiTotEvalElectro2.reinit(cell);
       phiTotEvalElectro2.read_dof_values_plain(phiRhoMinusApproxRho);
       phiTotEvalElectro2.evaluate(true,true);
    }

    //phiExtEvalElectro.reinit(cell);
    //phiExtEvalElectro.read_dof_values_plain(phiExtElectro);
    //if (dftParameters::isPseudopotential)
    //  phiExtEvalElectro.evaluate(true,true);
    //else
    //  phiExtEvalElectro.evaluate(true,false);

    std::fill(rhoQuadsElectro.begin(),rhoQuadsElectro.end(),make_vectorized_array(0.0));
    std::fill(shadowKSRhoMinQuadsElectro.begin(),shadowKSRhoMinQuadsElectro.end(),make_vectorized_array(0.0));
    std::fill(shadowKSRhoMinMinusRhoQuadsElectro.begin(),shadowKSRhoMinMinusRhoQuadsElectro.end(),make_vectorized_array(0.0));
    std::fill(gradRhoQuadsElectro.begin(),gradRhoQuadsElectro.end(),zeroTensor);
    std::fill(gradRhoAtomsQuadsElectro.begin(),gradRhoAtomsQuadsElectro.end(),zeroTensor);
    std::fill(pseudoVLocQuadsElectro.begin(),pseudoVLocQuadsElectro.end(),make_vectorized_array(0.0));
    std::fill(gradPseudoVLocQuadsElectro.begin(),gradPseudoVLocQuadsElectro.end(),zeroTensor);

    const unsigned int numSubCells=matrixFreeDataElectro.n_components_filled(cell);

    for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
    {
       subCellPtr= matrixFreeDataElectro.get_cell_iterator(cell,iSubCell);
       dealii::CellId subCellId=subCellPtr->id();
       for (unsigned int q=0; q<numQuadPoints; ++q)
       {
         rhoQuadsElectro[q][iSubCell]=rhoOutValuesElectro.find(subCellId)->second[q];
         if (shadowPotentialForce)
         {
            shadowKSRhoMinQuadsElectro[q][iSubCell]=shadowKSRhoMinValues.find(subCellId)->second[q];
            shadowKSRhoMinMinusRhoQuadsElectro[q][iSubCell]=shadowKSRhoMinQuadsElectro[q][iSubCell]-rhoQuadsElectro[q][iSubCell];

	    gradRhoAtomsQuadsElectro[q][0][iSubCell]=dftPtr->d_gradRhoAtomsValues.find(subCellId)->second[C_DIM*q+0];
	    gradRhoAtomsQuadsElectro[q][1][iSubCell]=dftPtr->d_gradRhoAtomsValues.find(subCellId)->second[C_DIM*q+1];
	    gradRhoAtomsQuadsElectro[q][2][iSubCell]=dftPtr->d_gradRhoAtomsValues.find(subCellId)->second[C_DIM*q+2];
         }
       }

       if(d_isElectrostaticsMeshSubdivided)
	  for (unsigned int q=0; q<numQuadPoints; ++q)
	  {
	     gradRhoQuadsElectro[q][0][iSubCell]=gradRhoOutValuesElectro.find(subCellId)->second[C_DIM*q+0];
	     gradRhoQuadsElectro[q][1][iSubCell]=gradRhoOutValuesElectro.find(subCellId)->second[C_DIM*q+1];
	     gradRhoQuadsElectro[q][2][iSubCell]=gradRhoOutValuesElectro.find(subCellId)->second[C_DIM*q+2];
	  }

       if(dftParameters::isPseudopotential)
	  for (unsigned int q=0; q<numQuadPoints; ++q)
	  {
	     pseudoVLocQuadsElectro[q][iSubCell]=pseudoVLocElectro.find(subCellId)->second[q];
	     gradPseudoVLocQuadsElectro[q][0][iSubCell]=gradPseudoVLocElectro.find(subCellId)->second[C_DIM*q+0];
	     gradPseudoVLocQuadsElectro[q][1][iSubCell]=gradPseudoVLocElectro.find(subCellId)->second[C_DIM*q+1];
	     gradPseudoVLocQuadsElectro[q][2][iSubCell]=gradPseudoVLocElectro.find(subCellId)->second[C_DIM*q+2];
	  }
    }

    if(dftParameters::isPseudopotential)
    {

       FPSPLocalGammaAtomsElementalContribution(forceContributionFPSPLocalGammaAtoms,
						feVselfValuesElectro,
						forceEvalElectro,
						matrixFreeDataElectro,
						cell,
						shadowPotentialForce?shadowKSRhoMinQuadsElectro:rhoQuadsElectro,
						gradPseudoVLocAtomsElectro,
						vselfBinsManagerElectro,
						d_cellsVselfBallsClosestAtomIdDofHandlerElectro);

    }

    for (unsigned int q=0; q<numQuadPoints; ++q)
    {
       VectorizedArray<double> phiTotElectro_q =phiTotEvalElectro.get_value(q);
       VectorizedArray<double> phiExtElectro_q =make_vectorized_array(0.0);
       Tensor<1,C_DIM,VectorizedArray<double> > gradPhiTotElectro_q =phiTotEvalElectro.get_gradient(q);

       Tensor<2,C_DIM,VectorizedArray<double> > E=eshelbyTensor::getEElectroEshelbyTensor
	                                                     (phiTotElectro_q,
			                                      gradPhiTotElectro_q,
						              rhoQuadsElectro[q]);
       if (shadowPotentialForce)
       {
          VectorizedArray<double> identityTensorFactor=shadowKSRhoMinMinusRhoQuadsElectro[q]*phiTotElectro_q;

          E[0][0]+=identityTensorFactor;
          E[1][1]+=identityTensorFactor;
          E[2][2]+=identityTensorFactor;
       }

       if (dftParameters::isPseudopotential)
       {
          E+=eshelbyTensor::getELocPspEshelbyTensor
				(rhoQuadsElectro[q],
				 pseudoVLocQuadsElectro[q],
				 phiExtElectro_q);

          if (shadowPotentialForce)
		  E+=eshelbyTensor::getELocPspEshelbyTensor
					(shadowKSRhoMinMinusRhoQuadsElectro[q],
					 pseudoVLocQuadsElectro[q],
					 phiExtElectro_q);
       }

       Tensor<1,C_DIM,VectorizedArray<double> > F=zeroTensor;

       if(d_isElectrostaticsMeshSubdivided)
	   F+=gradRhoQuadsElectro[q]*phiTotElectro_q;

       if(dftParameters::isPseudopotential)
       {
	   Tensor<1,C_DIM,VectorizedArray<double> > gradPhiExt_q =zeroTensor;//phiExtEvalElectro.get_gradient(q);
	   F+=eshelbyTensor::getFPSPLocal
	                        (rhoQuadsElectro[q],
	          	         gradPseudoVLocQuadsElectro[q],
				 gradPhiExt_q);

           if (shadowPotentialForce)
		   F+=eshelbyTensor::getFPSPLocal
					(shadowKSRhoMinMinusRhoQuadsElectro[q],
					 gradPseudoVLocQuadsElectro[q],
					 gradPhiExt_q);
           

	   if(d_isElectrostaticsMeshSubdivided)
	       F+=gradRhoQuadsElectro[q]*(pseudoVLocQuadsElectro[q]);//-phiExtElectro_q);
       }

       if (shadowPotentialForce)
       {
          VectorizedArray<double> phiRhoMinusApproxRho_q =phiTotEvalElectro2.get_value(q);
          Tensor<1,C_DIM,VectorizedArray<double> > gradPhiRhoMinusApproxRho_q =phiTotEvalElectro2.get_gradient(q);
          VectorizedArray<double> identityTensorFactor=make_vectorized_array(-1.0/(4.0*M_PI))*scalar_product(gradPhiRhoMinusApproxRho_q,gradPhiTotElectro_q)+phiRhoMinusApproxRho_q*rhoQuadsElectro[q];
          E+= (outer_product(gradPhiRhoMinusApproxRho_q,gradPhiTotElectro_q)+outer_product(gradPhiTotElectro_q,gradPhiRhoMinusApproxRho_q))*make_vectorized_array(1.0/(4.0*M_PI));
          E[0][0]+=identityTensorFactor;
          E[1][1]+=identityTensorFactor;
          E[2][2]+=identityTensorFactor;
       }
       
       forceEvalElectro.submit_value(F,q);
       forceEvalElectro.submit_gradient(E,q);
    }

    forceEvalElectro.integrate (true,true);
    forceEvalElectro.distribute_local_to_global(d_configForceVectorLinFEElectro);

  }

  // add global FPSPLocal contribution due to Gamma(Rj) to the configurational force vector
  if(dftParameters::isPseudopotential)
  {
     distributeForceContributionFPSPLocalGammaAtoms(forceContributionFPSPLocalGammaAtoms,
	                                            d_atomsForceDofsElectro,
						    d_constraintsNoneForceElectro,
						    d_configForceVectorLinFEElectro);
  }
}
