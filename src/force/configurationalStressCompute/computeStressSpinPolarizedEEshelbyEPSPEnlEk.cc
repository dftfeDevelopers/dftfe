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
#ifdef ENABLE_PERIODIC_BC 
//compute configurational stress contribution from all terms except the nuclear self energy
template<unsigned int FEOrder>
void forceClass<FEOrder>::computeStressSpinPolarizedEEshelbyEPSPEnlEk()
{
  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numberImageCharges = dftPtr->d_imageIds.size();
  const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges; 
  const bool isPseudopotential = dftParameters::isPseudopotential;
  const unsigned int numVectorizedArrayElements=VectorizedArray<double>::n_array_elements;
  const MatrixFree<3,double> & matrix_free_data=dftPtr->matrix_free_data;

  FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  forceEval(matrix_free_data,d_forceDofHandlerIndex, 0);
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiTotEval(matrix_free_data,dftPtr->phiTotDofHandlerIndex, 0);
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiTotInEval(matrix_free_data,dftPtr->phiTotDofHandlerIndex, 0);  
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiExtEval(matrix_free_data, dftPtr->phiExtDofHandlerIndex, 0);
  QGauss<C_DIM>  quadrature(C_num1DQuad<FEOrder>());   
  FEValues<C_DIM> feVselfValues (dftPtr->FE, quadrature, update_gradients | update_quadrature_points);
  FEValues<C_DIM> psiValues(dftPtr->FEEigen, quadrature, update_values | update_gradients| update_hessians);

  const unsigned int numQuadPoints=forceEval.n_q_points;
  const unsigned int numEigenVectors=dftPtr->eigenVectors[0].size();  
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

  VectorizedArray<double> phiExtFactor=make_vectorized_array(0.0);
  std::vector<std::vector<double> > projectorKetTimesPsiSpin0TimesVReal;
  std::vector<std::vector<std::vector<std::complex<double> > > > projectorKetTimesPsiSpin0TimesVComplexKPoints(numKPoints);
  std::vector<std::vector<double> > projectorKetTimesPsiSpin1TimesVReal;
  std::vector<std::vector<std::vector<std::complex<double> > > > projectorKetTimesPsiSpin1TimesVComplexKPoints(numKPoints);  
  if (isPseudopotential)
  {
    phiExtFactor=make_vectorized_array(1.0);
    for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
    {
         computeNonLocalProjectorKetTimesPsiTimesV(dftPtr->eigenVectors[2*ikPoint],
			                           projectorKetTimesPsiSpin0TimesVReal,
                                                   projectorKetTimesPsiSpin0TimesVComplexKPoints[ikPoint],
						   ikPoint);
    } 
    for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
    {
         computeNonLocalProjectorKetTimesPsiTimesV(dftPtr->eigenVectors[2*ikPoint+1],
			                           projectorKetTimesPsiSpin1TimesVReal,
                                                   projectorKetTimesPsiSpin1TimesVComplexKPoints[ikPoint],
						   ikPoint);
    }     
  }

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

  for (unsigned int cell=0; cell<matrix_free_data.n_macro_cells(); ++cell)
  {
    forceEval.reinit(cell);

    phiTotEval.reinit(cell);
    phiTotEval.read_dof_values_plain(dftPtr->poissonPtr->phiTotRhoOut);//read without taking constraints into account
    phiTotEval.evaluate(true,true);

    phiTotInEval.reinit(cell);
    phiTotInEval.read_dof_values_plain(dftPtr->poissonPtr->phiTotRhoIn);//read without taking constraints into account
    phiTotInEval.evaluate(true,true);    

    phiExtEval.reinit(cell);
    phiExtEval.read_dof_values_plain(dftPtr->poissonPtr->phiExt);
    phiExtEval.evaluate(true,true);

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
    for (unsigned int q=0; q<numQuadPoints; ++q)
    {
	 vEffRhoInSpin0Quads[q]=phiTotInEval.get_value(q);
	 vEffRhoInSpin1Quads[q]=phiTotInEval.get_value(q); 	 
	 vEffRhoOutSpin0Quads[q]=phiTotEval.get_value(q); 
	 vEffRhoOutSpin1Quads[q]=phiTotEval.get_value(q); 
    }      
    //allocate storage for vector of quadPoints, nonlocal atom id, pseudo wave, k point
    //FIXME: flatten nonlocal atomid id and pseudo wave and k point
    std::vector<std::vector<std::vector<std::vector<Tensor<1,2,VectorizedArray<double> > > > > >ZetaDeltaVQuads;
    std::vector<std::vector<std::vector<std::vector<Tensor<1,2, Tensor<2,C_DIM,VectorizedArray<double> > > > > > >gradZetalmDeltaVlDyadicDistImageAtomsQuads; 
    if(isPseudopotential)
    {
	ZetaDeltaVQuads.resize(numQuadPoints);	
	gradZetalmDeltaVlDyadicDistImageAtomsQuads.resize(numQuadPoints);
	for (unsigned int q=0; q<numQuadPoints; ++q)
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

    const unsigned int numSubCells=matrix_free_data.n_components_filled(cell);
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
       subCellPtr= matrix_free_data.get_cell_iterator(cell,iSubCell);
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
       }
    }   
   
    std::vector<Tensor<1,2,VectorizedArray<double> > > psiSpin0Quads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor1);
    std::vector<Tensor<1,2,VectorizedArray<double> > > psiSpin1Quads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor1);    
    std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > > gradPsiSpin0Quads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor2);
    std::vector<Tensor<1,2,Tensor<1,C_DIM,VectorizedArray<double> > > > gradPsiSpin1Quads(numQuadPoints*numEigenVectors*numKPoints,zeroTensor2);    
    std::vector<Vector<double> > tempPsiSpin0(numQuadPoints);
    std::vector<Vector<double> > tempPsiSpin1(numQuadPoints);    
    std::vector<std::vector<Tensor<1,C_DIM,double > > >  tempGradPsiSpin0(numQuadPoints); 
    std::vector<std::vector<Tensor<1,C_DIM,double > > >  tempGradPsiSpin1(numQuadPoints);      
    std::vector<std::vector<Tensor<2,C_DIM,double > > >  tempHessianPsiSpin0(numQuadPoints);
    std::vector<std::vector<Tensor<2,C_DIM,double > > >  tempHessianPsiSpin1(numQuadPoints);    
    for (unsigned int q=0; q<numQuadPoints; ++q)
    {
	  tempPsiSpin0[q].reinit(2);
	  tempPsiSpin1[q].reinit(2);	  
	  tempGradPsiSpin0[q].resize(2);
	  tempGradPsiSpin1[q].resize(2);
	  tempHessianPsiSpin0[q].resize(2);
	  tempHessianPsiSpin1[q].resize(2);
	  
    }    

    for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
    {   
      subCellPtr= dftPtr->matrix_free_data.get_cell_iterator(cell,iSubCell,dftPtr->eigenDofHandlerIndex);
      psiValues.reinit(subCellPtr);
      for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
      { 
        for (unsigned int iEigenVec=0; iEigenVec<numEigenVectors; ++iEigenVec)
        {
	  psiValues.get_function_values((dftPtr->eigenVectors[2*ikPoint][iEigenVec]), tempPsiSpin0);
	  psiValues.get_function_values((dftPtr->eigenVectors[2*ikPoint+1][iEigenVec]), tempPsiSpin1);	  
          psiValues.get_function_gradients((dftPtr->eigenVectors[2*ikPoint][iEigenVec]), tempGradPsiSpin0);
          psiValues.get_function_gradients((dftPtr->eigenVectors[2*ikPoint+1][iEigenVec]), tempGradPsiSpin1);  
          psiValues.get_function_hessians((dftPtr->eigenVectors[2*ikPoint][iEigenVec]), tempHessianPsiSpin0);
          psiValues.get_function_hessians((dftPtr->eigenVectors[2*ikPoint+1][iEigenVec]), tempHessianPsiSpin1);
          for (unsigned int q=0; q<numQuadPoints; ++q)
          {
	     const int id=q*numEigenVectors*numKPoints+numEigenVectors*ikPoint+iEigenVec;
	     for (unsigned int icomp=0;icomp<2;++icomp)
	     {
		 psiSpin0Quads[id][icomp][iSubCell]=tempPsiSpin0[q][icomp];
		 psiSpin1Quads[id][icomp][iSubCell]=tempPsiSpin1[q][icomp];		 
		 for (unsigned int idim=0; idim<C_DIM; idim++)
		 {
		     gradPsiSpin0Quads[id][icomp][idim][iSubCell]=tempGradPsiSpin0[q][icomp][idim];
		     gradPsiSpin1Quads[id][icomp][idim][iSubCell]=tempGradPsiSpin1[q][icomp][idim];		     
		 }
	     }
             const double partOccSpin0 =dftUtils::getPartialOccupancy
		                                                     (dftPtr->eigenValues[ikPoint][iEigenVec],
		                                                      dftPtr->fermiEnergy,
								      C_kb,
								      dftParameters::TVal); 
             const double partOccSpin1 =dftUtils::getPartialOccupancy
		                                                     (dftPtr->eigenValues[ikPoint][iEigenVec+numEigenVectors],
		                                                      dftPtr->fermiEnergy,
								      C_kb,
								      dftParameters::TVal); 	     
	     const Tensor<1,C_DIM,double > tempGradRhoSpin0Contribution=dftPtr->d_kPointWeights[ikPoint]*partOccSpin0*internalforce::computeGradRhoContribution(tempPsiSpin0[q], tempGradPsiSpin0[q]);
	     const Tensor<1,C_DIM,double > tempGradRhoSpin1Contribution=dftPtr->d_kPointWeights[ikPoint]*partOccSpin1*internalforce::computeGradRhoContribution(tempPsiSpin1[q], tempGradPsiSpin1[q]);	     
	     const Tensor<2,C_DIM,double > tempHessianRhoSpin0Contribution=dftPtr->d_kPointWeights[ikPoint]*partOccSpin0*internalforce::computeHessianRhoContribution(tempPsiSpin0[q], tempGradPsiSpin0[q], tempHessianPsiSpin0[q]);
	     const Tensor<2,C_DIM,double > tempHessianRhoSpin1Contribution=dftPtr->d_kPointWeights[ikPoint]*partOccSpin1*internalforce::computeHessianRhoContribution(tempPsiSpin1[q], tempGradPsiSpin1[q], tempHessianPsiSpin1[q]);

	     for (unsigned int idim=0; idim<C_DIM; idim++)
	     {
	       gradRhoSpin0Quads[q][idim][iSubCell]+=tempGradRhoSpin0Contribution[idim];
	       gradRhoSpin1Quads[q][idim][iSubCell]+=tempGradRhoSpin1Contribution[idim];
	       for (unsigned int jdim=0; jdim<C_DIM; jdim++)
	       {		 
	         hessianRhoSpin0Quads[q][idim][jdim][iSubCell]+=tempHessianRhoSpin0Contribution[idim][jdim];
	         hessianRhoSpin1Quads[q][idim][jdim][iSubCell]+=tempHessianRhoSpin1Contribution[idim][jdim];	 
	       }
	     }
          }//quad point loop
        } //eigenvector loop
      }//k point loop
      //accumulate hessian rho quad point contribution from all pools
      for (unsigned int q=0; q<numQuadPoints; ++q)
      {      
	for (unsigned int idim=0; idim<C_DIM; idim++)
	{
	  gradRhoSpin0Quads[q][idim][iSubCell]=Utilities::MPI::sum(gradRhoSpin0Quads[q][idim][iSubCell],dftPtr->interpoolcomm);
	  gradRhoSpin1Quads[q][idim][iSubCell]=Utilities::MPI::sum(gradRhoSpin1Quads[q][idim][iSubCell],dftPtr->interpoolcomm);	  
	  for (unsigned int jdim=0; jdim<C_DIM; jdim++)
	  {		 
	    hessianRhoSpin0Quads[q][idim][jdim][iSubCell]=Utilities::MPI::sum(hessianRhoSpin0Quads[q][idim][jdim][iSubCell],dftPtr->interpoolcomm);
;
	    hessianRhoSpin1Quads[q][idim][jdim][iSubCell]=Utilities::MPI::sum(hessianRhoSpin1Quads[q][idim][jdim][iSubCell],dftPtr->interpoolcomm);
	  }
	}    
      }
    }//subcell loop

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
            for (unsigned int i=0; i < d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint.size(); ++i)
	    {
	      const int numberPseudoWaveFunctions = d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[i].size();
	      for (unsigned int iPseudoWave=0; iPseudoWave < numberPseudoWaveFunctions; ++iPseudoWave)
	      {	
		if (d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[i][iPseudoWave].find(subCellId)!=d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[i][iPseudoWave].end())
		{
                   for (unsigned int ikPoint=0; ikPoint<numKPoints; ++ikPoint)
		   { 
                      ZetaDeltaVQuads[q][i][iPseudoWave][ikPoint][0][iSubCell]=d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave][subCellId][ikPoint*numQuadPoints*2+q*2+0];
                      ZetaDeltaVQuads[q][i][iPseudoWave][ikPoint][1][iSubCell]=d_nonLocalPSP_ZetalmDeltaVl[i][iPseudoWave][subCellId][ikPoint*numQuadPoints*2+q*2+1];		       
		      for (unsigned int idim=0; idim<C_DIM; idim++)
		      {
		        for (unsigned int jdim=0; jdim<C_DIM; jdim++)
		        {			  
                           gradZetalmDeltaVlDyadicDistImageAtomsQuads[q][i][iPseudoWave][ikPoint][0][idim][jdim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPoints*C_DIM*C_DIM*2+q*C_DIM*C_DIM*2+idim*C_DIM*2+jdim*2+0];
                           gradZetalmDeltaVlDyadicDistImageAtomsQuads[q][i][iPseudoWave][ikPoint][1][idim][jdim][iSubCell]=d_nonLocalPSP_gradZetalmDeltaVlDyadicDistImageAtoms_KPoint[i][iPseudoWave][subCellId][ikPoint*numQuadPoints*C_DIM*C_DIM*2+q*C_DIM*C_DIM*2+idim*C_DIM*2+jdim*2+1];
			}
		      }
		   }
		}//non-trivial cellId check
	      }//iPseudoWave loop
	    }//i loop
	  }//q loop
       }//subcell loop   

       addEPSPStressContribution(feVselfValues,
	                         forceEval,
			         cell,
			         rhoQuads);       
    }//is pseudopotential check

    for (unsigned int q=0; q<numQuadPoints; ++q)
    {
       VectorizedArray<double> phiTot_q =phiTotEval.get_value(q);   
       Tensor<1,C_DIM,VectorizedArray<double> > gradPhiTot_q =phiTotEval.get_gradient(q);
       VectorizedArray<double> phiExt_q =phiExtEval.get_value(q)*phiExtFactor;
       Tensor<2,C_DIM,VectorizedArray<double> > E=eshelbyTensorSP::getELocEshelbyTensorPeriodicNoKPoints
	                                                     (phiTot_q,
			                                      gradPhiTot_q,
						              rhoQuads[q],
						              gradRhoSpin0Quads[q],
							      gradRhoSpin1Quads[q],
						              excQuads[q],
						              derExchCorrEnergyWithGradRhoOutSpin0Quads[q],
							      derExchCorrEnergyWithGradRhoOutSpin1Quads[q],
							      pseudoVLocQuads[q],
							      phiExt_q);

       Tensor<2,C_DIM,VectorizedArray<double> > EKPoints=eshelbyTensorSP::getELocEshelbyTensorPeriodicKPoints
							 (psiSpin0Quads.begin()+q*numEigenVectors*numKPoints,
                                                          psiSpin1Quads.begin()+q*numEigenVectors*numKPoints,	  
							  gradPsiSpin0Quads.begin()+q*numEigenVectors*numKPoints,
							  gradPsiSpin1Quads.begin()+q*numEigenVectors*numKPoints, 
							  dftPtr->d_kPointCoordinates,
							  dftPtr->d_kPointWeights,
							  dftPtr->eigenValues,
							  dftPtr->fermiEnergy,
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
					  dftParameters::TVal);     
       
       if(isPseudopotential)
       {
           EKPoints+=eshelbyTensorSP::getEnlEshelbyTensorPeriodic
	                                                (ZetaDeltaVQuads[q],
		                                         projectorKetTimesPsiSpin0TimesVComplexKPoints,
		                                         projectorKetTimesPsiSpin1TimesVComplexKPoints,		 
						         psiSpin0Quads.begin()+q*numEigenVectors*numKPoints,
						         psiSpin1Quads.begin()+q*numEigenVectors*numKPoints,	 
							 dftPtr->d_kPointWeights,
						         dftPtr->eigenValues,
						         dftPtr->fermiEnergy,
						         dftParameters::TVal);

           EKPoints+=eshelbyTensorSP::getEnlStress(gradZetalmDeltaVlDyadicDistImageAtomsQuads[q],
		                                 projectorKetTimesPsiSpin0TimesVComplexKPoints,
						 projectorKetTimesPsiSpin1TimesVComplexKPoints,
					         psiSpin0Quads.begin()+q*numEigenVectors*numKPoints,
					         psiSpin1Quads.begin()+q*numEigenVectors*numKPoints,
					         dftPtr->d_kPointWeights,
						 dftPtr->eigenValues,
						 dftPtr->fermiEnergy,
						 dftParameters::TVal);
  
       }//is pseudopotential check

       E=E*forceEval.JxW(q);
       EKPoints=EKPoints*forceEval.JxW(q);       
       for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
       {
	    for (unsigned int idim=0; idim<C_DIM; ++idim)
	    {
		for (unsigned int jdim=0; jdim<C_DIM; ++jdim)
		{
		    d_stress[idim][jdim]+=E[idim][jdim][iSubCell];
		    d_stressKPoints[idim][jdim]+=EKPoints[idim][jdim][iSubCell];
		}
	    }
       }
    }//quad point loop
  }
}
#endif
