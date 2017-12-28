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
 
//(locally used function) compute FPSPLocal contibution due to Gamma(Rj) for given set of cells  
template<unsigned int FEOrder>
void forceClass<FEOrder>::computeForceContributionFPSPLocalGammaAtoms(std::map<unsigned int, std::vector<double> > & forceContributionFPSPLocalGammaAtoms,
		                                               FEValues<C_DIM> & feVselfValues,
							       FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  & forceEval,
							       const unsigned int cell,
							       const std::vector<VectorizedArray<double> > & rhoQuads){
  Tensor<1,C_DIM,VectorizedArray<double> > zeroTensor1;
  for (unsigned int idim=0; idim<C_DIM; idim++)
    zeroTensor1[idim]=make_vectorized_array(0.0);  
  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numberImageCharges = dftPtr->d_imageIds.size();
  const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;   
  const unsigned int numSubCells= dftPtr->matrix_free_data.n_components_filled(cell);
  const unsigned int numQuadPoints=forceEval.n_q_points;
  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;

  for (unsigned int iAtom=0;iAtom <totalNumberAtoms; iAtom++){
    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPseudoVLocAtomsQuads(numQuadPoints,zeroTensor1);
    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradVselfQuads(numQuadPoints,zeroTensor1);

    bool isLocalDomainOutsideVselfBall=false;
    bool isLocalDomainOutsidePspTail= false;
    if (d_gradPseudoVLocAtoms.find(iAtom)==d_gradPseudoVLocAtoms.end())
       isLocalDomainOutsidePspTail=true;		  
    if (d_AtomIdBinIdLocalDofHandler.find(iAtom)==d_AtomIdBinIdLocalDofHandler.end())
       isLocalDomainOutsideVselfBall=true;
    if (isLocalDomainOutsideVselfBall && isLocalDomainOutsidePspTail)
       continue;

    double atomCharge;
    int atomId=iAtom;
    Point<C_DIM> atomLocation;
    if(iAtom < numberGlobalAtoms)
    {
       atomLocation[0]=dftPtr->atomLocations[iAtom][2];
       atomLocation[1]=dftPtr->atomLocations[iAtom][3];
       atomLocation[2]=dftPtr->atomLocations[iAtom][4];
       if(dftPtr->d_isPseudopotential)
         atomCharge = dftPtr->atomLocations[iAtom][1];
       else
         atomCharge = dftPtr->atomLocations[iAtom][0];
    }
    else{
       const int imageId=iAtom-numberGlobalAtoms;
       atomId=dftPtr->d_imageIds[imageId];	
       atomCharge = dftPtr->d_imageCharges[imageId];
       atomLocation[0]=dftPtr->d_imagePositions[imageId][0];
       atomLocation[1]=dftPtr->d_imagePositions[imageId][1];
       atomLocation[2]=dftPtr->d_imagePositions[imageId][2];
    }

    for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell){
       subCellPtr= dftPtr->matrix_free_data.get_cell_iterator(cell,iSubCell);
       dealii::CellId subCellId=subCellPtr->id();
       feVselfValues.reinit(subCellPtr);	     
       //get grad vself for iAtom
       bool isCellOutsideVselfBall=true;
       if (!isLocalDomainOutsideVselfBall){
	  const unsigned int binIdiAtom=d_AtomIdBinIdLocalDofHandler[iAtom];	       
	  std::map<dealii::CellId, unsigned int >::const_iterator it=d_cellsVselfBallsClosestAtomIdDofHandler[binIdiAtom].find(subCellId);
	  if (it!=d_cellsVselfBallsClosestAtomIdDofHandler[binIdiAtom].end()){
            if(it->second ==iAtom){		  
	     isCellOutsideVselfBall=false;			       
  	     std::vector<Tensor<1,C_DIM,double> > gradVselfQuadsSubCell(numQuadPoints);		    
	     feVselfValues.get_function_gradients(dftPtr->d_vselfFieldBins[binIdiAtom],gradVselfQuadsSubCell);
	     for (unsigned int q=0; q<numQuadPoints; ++q){
	        gradVselfQuads[q][0][iSubCell]=gradVselfQuadsSubCell[q][0];
	        gradVselfQuads[q][1][iSubCell]=gradVselfQuadsSubCell[q][1];
	        gradVselfQuads[q][2][iSubCell]=gradVselfQuadsSubCell[q][2];
  	     }
	    }
	  }
       }
       if (isCellOutsideVselfBall){       
	  for (unsigned int q=0; q<numQuadPoints; ++q)
	  {  
	      Point<C_DIM> quadPoint=feVselfValues.quadrature_point(q);
	      Tensor<1,C_DIM,double> dispAtom=quadPoint-atomLocation;
	      const double dist=dispAtom.norm();
	      Tensor<1,C_DIM,double> temp=-atomCharge*dispAtom/dist/dist/dist;
	      gradVselfQuads[q][0][iSubCell]=temp[0];
	      gradVselfQuads[q][1][iSubCell]=temp[1];
	      gradVselfQuads[q][2][iSubCell]=temp[2];		   
	  }
       }

       //get grad pseudo VLoc for iAtom
       bool isCellOutsidePspTail=true;
       if (!isLocalDomainOutsidePspTail){
	  std::map<dealii::CellId, std::vector<double> >::const_iterator it=d_gradPseudoVLocAtoms[iAtom].find(subCellId);
	  if (it!=d_gradPseudoVLocAtoms[iAtom].end()){
	    isCellOutsidePspTail=false;		       
	    for (unsigned int q=0; q<numQuadPoints; ++q){
	       gradPseudoVLocAtomsQuads[q][0][iSubCell]=(it->second)[q*C_DIM];
	       gradPseudoVLocAtomsQuads[q][1][iSubCell]=(it->second)[q*C_DIM+1];
	       gradPseudoVLocAtomsQuads[q][2][iSubCell]=(it->second)[q*C_DIM+2];
	    }
	  }
       }
    
       if (isCellOutsidePspTail){
	  for (unsigned int q=0; q<numQuadPoints; ++q)
	  {  
	      Point<C_DIM> quadPoint=feVselfValues.quadrature_point(q);
	      Tensor<1,C_DIM,double> dispAtom=quadPoint-atomLocation;
	      const double dist=dispAtom.norm();
	      Tensor<1,C_DIM,double> temp=-atomCharge*dispAtom/dist/dist/dist;
	      gradPseudoVLocAtomsQuads[q][0][iSubCell]=temp[0];
	      gradPseudoVLocAtomsQuads[q][1][iSubCell]=temp[1];
	      gradPseudoVLocAtomsQuads[q][2][iSubCell]=temp[2];		   
	  }
       }
    }//subCell loop
    for (unsigned int q=0; q<numQuadPoints; ++q)
    {  	   
        forceEval.submit_value(-eshelbyTensor::getFPSPLocal(rhoQuads[q],
							    gradPseudoVLocAtomsQuads[q],
							    gradVselfQuads[q]),
							    q);
    }
    Tensor<1,C_DIM,VectorizedArray<double> > forceContributionFPSPLocalGammaiAtomCells
						 =forceEval.integrate_value();

    if (forceContributionFPSPLocalGammaAtoms.find(atomId)==forceContributionFPSPLocalGammaAtoms.end())
       forceContributionFPSPLocalGammaAtoms[atomId]=std::vector<double>(C_DIM,0.0);
    for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell){
       for (unsigned int idim=0; idim<C_DIM; idim++){
         forceContributionFPSPLocalGammaAtoms[atomId][idim]+=
	       forceContributionFPSPLocalGammaiAtomCells[idim][iSubCell];
       }
    }
  }//iAtom loop
}

//(locally used function) accumulate and distribute FPSPLocal contibution due to Gamma(Rj)
template<unsigned int FEOrder>
void forceClass<FEOrder>::distributeForceContributionFPSPLocalGammaAtoms(const std::map<unsigned int,std::vector<double> > & forceContributionFPSPLocalGammaAtoms)
{
    for (unsigned int iAtom=0;iAtom <dftPtr->atomLocations.size(); iAtom++){
      bool doesAtomOrImageIdExistOnLocallyOwnedNode=false;
      if (d_atomsForceDofs.find(std::pair<unsigned int,unsigned int>(iAtom,0))!=d_atomsForceDofs.end()){
        doesAtomOrImageIdExistOnLocallyOwnedNode=true;		  
      }

      std::vector<double> forceContributionFPSPLocalGammaiAtomGlobal(C_DIM);
      std::vector<double> forceContributionFPSPLocalGammaiAtomLocal(C_DIM,0.0);

      if (forceContributionFPSPLocalGammaAtoms.find(iAtom)!=forceContributionFPSPLocalGammaAtoms.end())
	 forceContributionFPSPLocalGammaiAtomLocal=forceContributionFPSPLocalGammaAtoms.find(iAtom)->second;
      // accumulate value
      MPI_Allreduce(&(forceContributionFPSPLocalGammaiAtomLocal[0]),
		  &(forceContributionFPSPLocalGammaiAtomGlobal[0]),
		  3,
		  MPI_DOUBLE,
		  MPI_SUM,
		  mpi_communicator);  

      if (doesAtomOrImageIdExistOnLocallyOwnedNode){
        std::vector<types::global_dof_index> forceLocalDofIndices(C_DIM);
        for (unsigned int idim=0; idim<C_DIM; idim++)
	    forceLocalDofIndices[idim]=d_atomsForceDofs[std::pair<unsigned int,unsigned int>(iAtom,idim)];

        d_constraintsNoneForce.distribute_local_to_global(forceContributionFPSPLocalGammaiAtomGlobal,forceLocalDofIndices,d_configForceVectorLinFE); 
      }
    }	
}

//compute configurational force on the mesh nodes using linear shape function generators
template<unsigned int FEOrder>
void forceClass<FEOrder>::computeConfigurationalForceEEshelbyTensorFPSPNonPeriodicLinFE()
{
  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numberImageCharges = dftPtr->d_imageIds.size();
  const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges; 
  std::map<unsigned int, std::vector<double> > forceContributionFPSPLocalGammaAtoms;  

  const unsigned int numVectorizedArrayElements=VectorizedArray<double>::n_array_elements;
  const MatrixFree<3,double> & matrix_free_data=dftPtr->matrix_free_data;
  //std::cout<< "n array elements" << numVectorizedArrayElements <<std::endl;

  FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  forceEval(matrix_free_data,d_forceDofHandlerIndex, 0);
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiTotEval(matrix_free_data,dftPtr->phiTotDofHandlerIndex, 0);
  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> psiEval(matrix_free_data,dftPtr->eigenDofHandlerIndex , 0);  FEEvaluation<C_DIM,FEOrder,C_num1DQuad<FEOrder>(),1> phiExtEval(matrix_free_data, dftPtr->phiExtDofHandlerIndex, 0);
  QGauss<C_DIM>  quadrature(C_num1DQuad<FEOrder>());   
  FEValues<C_DIM> feVselfValues (dftPtr->FE, quadrature, update_gradients | update_quadrature_points);


  const unsigned int numQuadPoints=forceEval.n_q_points;
  const unsigned int numEigenVectors=dftPtr->eigenVectorsOrig[0].size();  
  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;
  Tensor<1,C_DIM,VectorizedArray<double> > zeroTensor1;

  for (unsigned int idim=0; idim<C_DIM; idim++)
    zeroTensor1[idim]=make_vectorized_array(0.0);

  VectorizedArray<double> phiExtFactor=make_vectorized_array(0.0);

  if (dftPtr->d_isPseudopotential){
    phiExtFactor=make_vectorized_array(1.0);
  }

  for (unsigned int cell=0; cell<matrix_free_data.n_macro_cells(); ++cell){
    forceEval.reinit(cell);
    phiTotEval.reinit(cell);
    phiTotEval.read_dof_values_plain(dftPtr->poissonPtr->phiTotRhoOut);//read without taking constraints into account
    phiTotEval.evaluate(true,true);
    psiEval.reinit(cell);

    phiExtEval.reinit(cell);
    phiExtEval.read_dof_values_plain(dftPtr->poissonPtr->phiExt);
    phiExtEval.evaluate(true,true);

    std::vector<VectorizedArray<double> > rhoQuads(numQuadPoints,make_vectorized_array(0.0));
    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoQuads(numQuadPoints,zeroTensor1);
    std::vector<VectorizedArray<double> > excQuads(numQuadPoints,make_vectorized_array(0.0));
    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoExcQuads(numQuadPoints,zeroTensor1);
    std::vector<VectorizedArray<double> > pseudoVLocQuads(numQuadPoints,make_vectorized_array(0.0));
    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPseudoVLocQuads(numQuadPoints,zeroTensor1);
    const unsigned int numSubCells=matrix_free_data.n_components_filled(cell);
    
    for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell){
       subCellPtr= matrix_free_data.get_cell_iterator(cell,iSubCell);
       dealii::CellId subCellId=subCellPtr->id();
       std::vector<double> exchValQuads(numQuadPoints);
       std::vector<double> corrValQuads(numQuadPoints); 
       if(dftPtr->d_xc_id == 4){
           pcout<< " GGA force computation not implemented yet"<<std::endl;
	   exit(-1);
       }
       else{
         xc_lda_exc(&(dftPtr->funcX),numQuadPoints,&((*dftPtr->rhoOutValues)[subCellId][0]),&exchValQuads[0]);
         xc_lda_exc(&(dftPtr->funcC),numQuadPoints,&((*dftPtr->rhoOutValues)[subCellId][0]),&corrValQuads[0]);     
         for (unsigned int q=0; q<numQuadPoints; ++q){
	   excQuads[q][iSubCell]=exchValQuads[q]+corrValQuads[q];
         }
       }

       for (unsigned int q=0; q<numQuadPoints; ++q){
         rhoQuads[q][iSubCell]=(*dftPtr->rhoOutValues)[subCellId][q];
         if(dftPtr->d_xc_id == 4){
	   gradRhoQuads[q][0][iSubCell]=(*dftPtr->gradRhoOutValues)[subCellId][3*q];
	   gradRhoQuads[q][1][iSubCell]=(*dftPtr->gradRhoOutValues)[subCellId][3*q+1];
           gradRhoQuads[q][2][iSubCell]=(*dftPtr->gradRhoOutValues)[subCellId][3*q+2]; 
	 }
       }
    }  

    if(dftPtr->d_isPseudopotential){
       for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell){
          subCellPtr= matrix_free_data.get_cell_iterator(cell,iSubCell);
          dealii::CellId subCellId=subCellPtr->id();	       
	  for (unsigned int q=0; q<numQuadPoints; ++q){
	     pseudoVLocQuads[q][iSubCell]=((*dftPtr->pseudoValues)[subCellId][q]);
	     gradPseudoVLocQuads[q][0][iSubCell]=d_gradPseudoVLoc[subCellId][C_DIM*q+0];
             gradPseudoVLocQuads[q][1][iSubCell]=d_gradPseudoVLoc[subCellId][C_DIM*q+1];
	     gradPseudoVLocQuads[q][2][iSubCell]=d_gradPseudoVLoc[subCellId][C_DIM*q+2];
	  }
       }
       //compute forceContributionFPSPLocalGammaAtoms  (contibution due to Gamma(Rj)) 
       computeForceContributionFPSPLocalGammaAtoms(forceContributionFPSPLocalGammaAtoms,
		                                   feVselfValues,
			                           forceEval,
					           cell,
					           rhoQuads);      
    }//is pseudopotential check
    
    std::vector< VectorizedArray<double> > psiQuads(numQuadPoints*numEigenVectors,make_vectorized_array(0.0));
    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPsiQuads(numQuadPoints*numEigenVectors,zeroTensor1);
    for (unsigned int iEigenVec=0; iEigenVec<numEigenVectors; ++iEigenVec){
      //psiEval.reinit(cell);	    
      psiEval.read_dof_values_plain(*((dftPtr->eigenVectorsOrig)[0][iEigenVec]));//read without taking constraints into account
      psiEval.evaluate(true,true);
      for (unsigned int q=0; q<numQuadPoints; ++q){
        psiQuads[q*numEigenVectors+iEigenVec]=psiEval.get_value(q);   
        gradPsiQuads[q*numEigenVectors+iEigenVec]=psiEval.get_gradient(q);	      
      }     
    }    
    

    for (unsigned int q=0; q<numQuadPoints; ++q){
       VectorizedArray<double> phiTot_q =phiTotEval.get_value(q);   
       Tensor<1,C_DIM,VectorizedArray<double> > gradPhiTot_q =phiTotEval.get_gradient(q);
       VectorizedArray<double> phiExt_q =phiExtEval.get_value(q)*phiExtFactor;   
       forceEval.submit_gradient(eshelbyTensor::getELocEshelbyTensorNonPeriodic(phiTot_q,
			                                         gradPhiTot_q,
						                 rhoQuads[q],
						                 gradRhoQuads[q],
						                 excQuads[q],
						                 gradRhoExcQuads[q],
								 pseudoVLocQuads[q],
								 phiExt_q,
						                 psiQuads.begin()+q*numEigenVectors,
						                 gradPsiQuads.begin()+q*numEigenVectors,
								 (dftPtr->eigenValues)[0],
								 dftPtr->fermiEnergy,
								 dftPtr->d_TVal),
		                                                 q);
    }
  
    if(dftPtr->d_isPseudopotential){
      for (unsigned int q=0; q<numQuadPoints; ++q){
        Tensor<1,C_DIM,VectorizedArray<double> > gradPhiExt_q =phiExtEval.get_gradient(q);   
        forceEval.submit_value(eshelbyTensor::getFPSPLocal(rhoQuads[q],
		                              gradPseudoVLocQuads[q],
			                      gradPhiExt_q),
			                      q);	      
      }
      forceEval.integrate(true,true);
    }
    else{
      forceEval.integrate (false,true);
    }
    forceEval.distribute_local_to_global(d_configForceVectorLinFE);//also takes care of constraints
  }

  // add global FPSPLocal contribution due to Gamma(Rj) to the configurational force vector
  if(dftPtr->d_isPseudopotential){
     distributeForceContributionFPSPLocalGammaAtoms(forceContributionFPSPLocalGammaAtoms);
  }
} 
