// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
// @author Sambit Das (2017)
//

//compute configurational stress
template<unsigned int FEOrder>
void forceClass<FEOrder>::computeStressEEshelbyEPSPEnlEk()
{
  //std::cout<< "BUG "<<d_nonLocalPSP_ZetalmDeltaVl.size()  <<std::endl;
  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numberImageCharges = dftPtr->d_imageIds.size();
  const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges; 

  bool isPseudopotential = dftParameters::isPseudopotential;

  const unsigned int numVectorizedArrayElements=VectorizedArray<double>::n_array_elements;
  const MatrixFree<3,double> & matrix_free_data=dftPtr->matrix_free_data;
  //std::cout<< "n array elements" << numVectorizedArrayElements <<std::endl;

  FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  forceEval(matrix_free_data,d_forceDofHandlerIndex, 0);
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiTotEval(matrix_free_data,dftPtr->phiTotDofHandlerIndex, 0);
#ifdef ENABLE_PERIODIC_BC
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),2> psiEval(matrix_free_data,dftPtr->eigenDofHandlerIndex , 0);
#else  
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> psiEval(matrix_free_data,dftPtr->eigenDofHandlerIndex , 0);
#endif  
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiExtEval(matrix_free_data, dftPtr->phiExtDofHandlerIndex, 0);
  QGauss<C_DIM>  quadrature(C_num1DQuad<FEOrder>());   
  FEValues<C_DIM> feVselfValues (dftPtr->FE, quadrature, update_gradients | update_quadrature_points);

  const unsigned int numQuadPoints=forceEval.n_q_points;
  const unsigned int numEigenVectors=dftPtr->eigenVectorsOrig[0].size();  
  const unsigned int numKPoints=dftPtr->d_kPointWeights.size();
  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;
  Tensor<1,2,VectorizedArray<double> > zeroTensor1;zeroTensor1[0]=make_vectorized_array(0.0);zeroTensor1[1]=make_vectorized_array(0.0);
  Tensor<1,2, Tensor<1,C_DIM,VectorizedArray<double> > > zeroTensor2;
  Tensor<1,C_DIM,VectorizedArray<double> > zeroTensor3;
  for (unsigned int idim=0; idim<C_DIM; idim++){
    zeroTensor2[0][idim]=make_vectorized_array(0.0);
    zeroTensor2[1][idim]=make_vectorized_array(0.0);
    zeroTensor3[idim]=make_vectorized_array(0.0);
  }

  VectorizedArray<double> phiExtFactor=make_vectorized_array(0.0);
  std::vector<std::vector<double> > projectorKetTimesPsiTimesVReal;
  std::vector<std::vector<std::vector<std::complex<double> > > > projectorKetTimesPsiTimesVComplexKPoints(numKPoints);
  if (isPseudopotential){
    phiExtFactor=make_vectorized_array(1.0);
    for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint){
         computeNonLocalProjectorKetTimesPsiTimesV(dftPtr->eigenVectorsOrig[ikPoint],
			                           projectorKetTimesPsiTimesVReal,
                                                   projectorKetTimesPsiTimesVComplexKPoints[ikPoint],
						   ikPoint);	
    }    
  }

  for (unsigned int cell=0; cell<matrix_free_data.n_macro_cells(); ++cell){
    forceEval.reinit(cell);
    phiTotEval.reinit(cell);
    psiEval.reinit(cell);    
    phiTotEval.read_dof_values_plain(dftPtr->poissonPtr->phiTotRhoOut);//read without taking constraints into account
    phiTotEval.evaluate(true,true);

    phiExtEval.reinit(cell);
    phiExtEval.read_dof_values_plain(dftPtr->poissonPtr->phiExt);
    phiExtEval.evaluate(true,true);

    std::vector<VectorizedArray<double> > rhoQuads(numQuadPoints,make_vectorized_array(0.0));
    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoQuads(numQuadPoints,zeroTensor3);
    std::vector<VectorizedArray<double> > excQuads(numQuadPoints,make_vectorized_array(0.0));
    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > derExcGradRhoQuads(numQuadPoints,zeroTensor3);
    std::vector<VectorizedArray<double> > pseudoVLocQuads(numQuadPoints,make_vectorized_array(0.0));
    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPseudoVLocQuads(numQuadPoints,zeroTensor3);
#ifdef ENABLE_PERIODIC_BC   
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
	ZetaDeltaVQuads.resize(numQuadPoints);
	gradZetaDeltaVQuads.resize(numQuadPoints);
#ifdef ENABLE_PERIODIC_BC	
	pspnlGammaAtomsQuads.resize(numQuadPoints);
#endif

	for (unsigned int q=0; q<numQuadPoints; ++q)
	{
	  ZetaDeltaVQuads[q].resize(d_nonLocalPSP_ZetalmDeltaVl.size());
	  gradZetaDeltaVQuads[q].resize(d_nonLocalPSP_ZetalmDeltaVl.size());
#ifdef ENABLE_PERIODIC_BC		  
	  pspnlGammaAtomsQuads[q].resize(d_nonLocalPSP_ZetalmDeltaVl.size());
#endif
	  for (unsigned int i=0; i < d_nonLocalPSP_ZetalmDeltaVl.size(); ++i)
	  {
	    const int numberPseudoWaveFunctions = d_nonLocalPSP_ZetalmDeltaVl[i].size();
#ifdef ENABLE_PERIODIC_BC 
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
    const unsigned int numSubCells=matrix_free_data.n_components_filled(cell);
    std::vector<double> exchValQuads(numQuadPoints);
    std::vector<double> corrValQuads(numQuadPoints); 
    std::vector<double> sigmaValQuads(numQuadPoints);
    std::vector<double> derExchEnergyWithDensityVal(numQuadPoints), derCorrEnergyWithDensityVal(numQuadPoints), derExchEnergyWithSigma(numQuadPoints),derCorrEnergyWithSigma(numQuadPoints);    
    for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
    {
       subCellPtr= matrix_free_data.get_cell_iterator(cell,iSubCell);
       dealii::CellId subCellId=subCellPtr->id();
       if(dftParameters::xc_id == 4)
       {
	  for (unsigned int q = 0; q < numQuadPoints; ++q)
	  {
	      double gradRhoX = ((*dftPtr->gradRhoOutValues)[subCellId][3*q + 0]);
	      double gradRhoY = ((*dftPtr->gradRhoOutValues)[subCellId][3*q + 1]);
	      double gradRhoZ = ((*dftPtr->gradRhoOutValues)[subCellId][3*q + 2]);
	      sigmaValQuads[q] = gradRhoX*gradRhoX + gradRhoY*gradRhoY + gradRhoZ*gradRhoZ;
	  }	   
	  xc_gga_exc_vxc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoOutValues)[subCellId][0]),&sigmaValQuads[0],&exchValQuads[0],&derExchEnergyWithDensityVal[0],&derExchEnergyWithSigma[0]);
	  xc_gga_exc_vxc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoOutValues)[subCellId][0]),&sigmaValQuads[0],&corrValQuads[0],&derCorrEnergyWithDensityVal[0],&derCorrEnergyWithSigma[0]);
          for (unsigned int q=0; q<numQuadPoints; ++q)
	  {
	     excQuads[q][iSubCell]=exchValQuads[q]+corrValQuads[q];
	     double temp = derExchEnergyWithSigma[q]+derCorrEnergyWithSigma[q];
	     derExcGradRhoQuads[q][0][iSubCell]=2*(*dftPtr->gradRhoOutValues)[subCellId][3*q]*temp;
	     derExcGradRhoQuads[q][1][iSubCell]=2*(*dftPtr->gradRhoOutValues)[subCellId][3*q+1]*temp;
             derExcGradRhoQuads[q][2][iSubCell]=2*(*dftPtr->gradRhoOutValues)[subCellId][3*q+2]*temp; 	     
          }	  
	  
       }
       else
       {
          xc_lda_exc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoOutValues)[subCellId][0]),&exchValQuads[0]);
          xc_lda_exc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoOutValues)[subCellId][0]),&corrValQuads[0]);     
          for (unsigned int q=0; q<numQuadPoints; ++q)
	  {
	     excQuads[q][iSubCell]=exchValQuads[q]+corrValQuads[q];
          }
       }

       for (unsigned int q=0; q<numQuadPoints; ++q)
       {
         rhoQuads[q][iSubCell]=(*dftPtr->rhoOutValues)[subCellId][q];
         if(dftParameters::xc_id == 4)
	 {
	   gradRhoQuads[q][0][iSubCell]=(*dftPtr->gradRhoOutValues)[subCellId][3*q];
	   gradRhoQuads[q][1][iSubCell]=(*dftPtr->gradRhoOutValues)[subCellId][3*q+1];
           gradRhoQuads[q][2][iSubCell]=(*dftPtr->gradRhoOutValues)[subCellId][3*q+2]; 
	 }	 
       }
    }   
   
#ifdef ENABLE_PERIODIC_BC
    std::vector<Tensor<1,2,VectorizedArray<double> > > psiQuads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor1);
    std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > > gradPsiQuads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor2);
#else     
    std::vector< VectorizedArray<double> > psiQuads(numQuadPoints*numEigenVectors,make_vectorized_array(0.0));
    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPsiQuads(numQuadPoints*numEigenVectors,zeroTensor3);
#endif    
    for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint){ 
     for (unsigned int iEigenVec=0; iEigenVec<numEigenVectors; ++iEigenVec){
       //psiEval.reinit(cell);	    
       psiEval.read_dof_values_plain(*((dftPtr->eigenVectorsOrig)[ikPoint][iEigenVec]));//read without taking constraints into account
       psiEval.evaluate(true,true);
       for (unsigned int q=0; q<numQuadPoints; ++q){
         psiQuads[q*numEigenVectors*numKPoints+numEigenVectors*ikPoint+iEigenVec]=psiEval.get_value(q);   
         gradPsiQuads[q*numEigenVectors*numKPoints+numEigenVectors*ikPoint+iEigenVec]=psiEval.get_gradient(q);	 
       }     
     } 
    }

    if(isPseudopotential)
    {
       for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
       {
          subCellPtr= matrix_free_data.get_cell_iterator(cell,iSubCell);
          dealii::CellId subCellId=subCellPtr->id();	       
	  for (unsigned int q=0; q<numQuadPoints; ++q)
	  {
	     pseudoVLocQuads[q][iSubCell]=((*dftPtr->pseudoValues)[subCellId][q]);
	     gradPseudoVLocQuads[q][0][iSubCell]=d_gradPseudoVLoc[subCellId][C_DIM*q+0];
             gradPseudoVLocQuads[q][1][iSubCell]=d_gradPseudoVLoc[subCellId][C_DIM*q+1];
	     gradPseudoVLocQuads[q][2][iSubCell]=d_gradPseudoVLoc[subCellId][C_DIM*q+2];
	  }
	  
	  for (unsigned int q=0; q<numQuadPoints; ++q)
	  {
            for (unsigned int i=0; i < d_nonLocalPSP_ZetalmDeltaVl.size(); ++i)
	    {
	      const int numberPseudoWaveFunctions = d_nonLocalPSP_ZetalmDeltaVl[i].size();
	      for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
	      {	
		if (d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave].find(subCellId)!=d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave].end())
		{
#ifdef ENABLE_PERIODIC_BC 
                   for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
		   { 		    
                      ZetaDeltaVQuads[q][i][iPseudoWave][ikPoint][0][iSubCell]=d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave][subCellId][ikPoint*numQuadPoints*2+q*2+0];
                      ZetaDeltaVQuads[q][i][iPseudoWave][ikPoint][1][iSubCell]=d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave][subCellId][ikPoint*numQuadPoints*2+q*2+1];	
		      for (unsigned int idim=0; idim<C_DIM; idim++)
		      {
                         gradZetaDeltaVQuads[q][i][iPseudoWave][ikPoint][0][idim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPoints*C_DIM*2+q*C_DIM*2+idim*2+0];
                         gradZetaDeltaVQuads[q][i][iPseudoWave][ikPoint][1][idim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVl_minusZetalmDeltaVl_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPoints*C_DIM*2+q*C_DIM*2+idim*2+1];
                         pspnlGammaAtomsQuads[q][i][iPseudoWave][ikPoint][0][idim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVl_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPoints*C_DIM*2+q*C_DIM*2+idim*2+0];
                         pspnlGammaAtomsQuads[q][i][iPseudoWave][ikPoint][1][idim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVl_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPoints*C_DIM*2+q*C_DIM*2+idim*2+1];				 
		      }
		   }
#else
		   ZetaDeltaVQuads[q][i][iPseudoWave][iSubCell]=d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave][subCellId][q];
		   for (unsigned int idim=0; idim<C_DIM; idim++)
		      gradZetaDeltaVQuads[q][i][iPseudoWave][idim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVl[i][iPseudoWave][subCellId][q*C_DIM+idim];
#endif
		}//non-trivial cellId check
	      }//iPseudoWave loop
	    }//i loop
	  }//q loop
       }//subcell loop
 
    }//is pseudopotential check    

    for (unsigned int q=0; q<numQuadPoints; ++q)
    {
       VectorizedArray<double> phiTot_q =phiTotEval.get_value(q);   
       Tensor<1,C_DIM,VectorizedArray<double> > gradPhiTot_q =phiTotEval.get_gradient(q);
       VectorizedArray<double> phiExt_q =phiExtEval.get_value(q)*phiExtFactor;
#ifdef ENABLE_PERIODIC_BC      
       Tensor<2,C_DIM,VectorizedArray<double> > E=eshelbyTensor::getELocEshelbyTensorPeriodic(phiTot_q,
			                                      gradPhiTot_q,
						              rhoQuads[q],
						              gradRhoQuads[q],
						              excQuads[q],
						              derExcGradRhoQuads[q],
							      pseudoVLocQuads[q],
							      phiExt_q,				      
						              psiQuads.begin()+q*numEigenVectors*numKPoints,
						              gradPsiQuads.begin()+q*numEigenVectors*numKPoints,
							      dftPtr->d_kPointCoordinates,
							      dftPtr->d_kPointWeights,
							      dftPtr->eigenValues,
							      dftPtr->fermiEnergy,
							      dftParameters::TVal);
#else         
       Tensor<2,C_DIM,VectorizedArray<double> > E=eshelbyTensor::getELocEshelbyTensorNonPeriodic(phiTot_q,
			                                         gradPhiTot_q,
						                 rhoQuads[q],
						                 gradRhoQuads[q],
						                 excQuads[q],
						                 derExcGradRhoQuads[q],
								 pseudoVLocQuads[q],
								 phiExt_q,
						                 psiQuads.begin()+q*numEigenVectors,
						                 gradPsiQuads.begin()+q*numEigenVectors,
								 (dftPtr->eigenValues)[0],
								 dftPtr->fermiEnergy,
								 dftParameters::TVal);
#endif
       if(isPseudopotential)
       {
           Tensor<1,C_DIM,VectorizedArray<double> > gradPhiExt_q =phiExtEval.get_gradient(q);
	   Tensor<1,C_DIM,VectorizedArray<double> > E+=eshelbyTensor::getFPSPLocal(rhoQuads[q],
		                                                                  gradPseudoVLocQuads[q],
			                                                          gradPhiExt_q);

#ifdef ENABLE_PERIODIC_BC    
           F+=eshelbyTensor::getFnlPeriodic(gradZetaDeltaVQuads[q],
					    projectorKetTimesPsiTimesVComplexKPoints,
					    psiQuads.begin()+q*numEigenVectors*numKPoints,
					    dftPtr->d_kPointWeights,
					    dftPtr->eigenValues,
					    dftPtr->fermiEnergy,
					    dftParameters::TVal);
 

           E+=eshelbyTensor::getEnlEshelbyTensorPeriodic(ZetaDeltaVQuads[q],
		                                         projectorKetTimesPsiTimesVComplexKPoints,
						         psiQuads.begin()+q*numEigenVectors*numKPoints,
							 dftPtr->d_kPointWeights,
						         dftPtr->eigenValues,
						         dftPtr->fermiEnergy,
						         dftParameters::TVal);
#else     
           F+=eshelbyTensor::getFnlNonPeriodic(gradZetaDeltaVQuads[q],
					       projectorKetTimesPsiTimesVReal,
					       psiQuads.begin()+q*numEigenVectors,
					       (dftPtr->eigenValues)[0],
					       dftPtr->fermiEnergy,
					       dftParameters::TVal);  
	   
           E+=eshelbyTensor::getEnlEshelbyTensorNonPeriodic(ZetaDeltaVQuads[q],
		                                            projectorKetTimesPsiTimesVReal,
						            psiQuads.begin()+q*numEigenVectors,
						            (dftPtr->eigenValues)[0],
						            dftPtr->fermiEnergy,
						            dftParameters::TVal);
							    
            							    
#endif	  
	   //forceEval.submit_value(F,q);    
       }//is pseudopotential check
       E=E*forceEval.JxW(q);
       for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
       {
	    for (unsigned int idim=0; idim<C_dim; ++idim)
		for (unsigned int jdim=0; jdim<C_dim; ++jdim)
		    d_stress+=E[idim][jdim][iSubCell];
       }
       //forceEval.submit_gradient(E,q);       
    }//quad point loop
    if(isPseudopotential)
    {
      //forceEval.integrate(true,true);
    }
    else
    {
      //forceEval.integrate (false,true);
    }    
  }

}
