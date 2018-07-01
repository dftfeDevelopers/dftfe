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
// @author Sambit Das (2018)
//
#ifdef USE_COMPLEX
//compute EPSP contribution stress (local pseudopotential)
template<unsigned int FEOrder>
void forceClass<FEOrder>::addEPSPStressContribution
                             (FEValues<C_DIM> & feVselfValues,
			      FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  & forceEval,
			      const unsigned int cell,
			      const std::vector<VectorizedArray<double> > & rhoQuads)
{
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
    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > xMinusAtomLoc(numQuadPoints,zeroTensor1);


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
       if(dftParameters::isPseudopotential)
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

    for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
    {
       subCellPtr= dftPtr->matrix_free_data.get_cell_iterator(cell,iSubCell);
       dealii::CellId subCellId=subCellPtr->id();
       feVselfValues.reinit(subCellPtr);

	for (unsigned int q=0; q<numQuadPoints; ++q)
	{
	   const Point<C_DIM> & quadPoint=feVselfValues.quadrature_point(q);
	   const Tensor<1,C_DIM,double> dispAtom=quadPoint-atomLocation;
           for (unsigned int idim=0; idim<C_DIM; idim++)
           {
	       xMinusAtomLoc[q][idim][iSubCell]=dispAtom[idim];
	   }
	}

       // get computed grad vself for iAtom
       bool isCellOutsideVselfBall=true;
       if (!isLocalDomainOutsideVselfBall){
	  const unsigned int binIdiAtom=d_AtomIdBinIdLocalDofHandler[iAtom];
	  std::map<dealii::CellId, unsigned int >::const_iterator it=d_cellsVselfBallsClosestAtomIdDofHandler[binIdiAtom].find(subCellId);
	  if (it!=d_cellsVselfBallsClosestAtomIdDofHandler[binIdiAtom].end()){
            if(it->second ==iAtom){
	     isCellOutsideVselfBall=false;
  	     std::vector<Tensor<1,C_DIM,double> > gradVselfQuadsSubCell(numQuadPoints);
	     feVselfValues.get_function_gradients(dftPtr->d_vselfBinsManager.getVselfFieldBins()[binIdiAtom],gradVselfQuadsSubCell);
	     for (unsigned int q=0; q<numQuadPoints; ++q){
	        gradVselfQuads[q][0][iSubCell]=gradVselfQuadsSubCell[q][0];
	        gradVselfQuads[q][1][iSubCell]=gradVselfQuadsSubCell[q][1];
	        gradVselfQuads[q][2][iSubCell]=gradVselfQuadsSubCell[q][2];
  	     }
	    }
	  }
       }

       //get computed grad pseudo VLoc for iAtom
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

       // get exact solution (Z/r^2) for grad vself and grad pseudo VLoc for iAtom
       if (isCellOutsideVselfBall || isCellOutsidePspTail){
	  for (unsigned int q=0; q<numQuadPoints; ++q)
	  {
	      const Point<C_DIM> & quadPoint=feVselfValues.quadrature_point(q);
	      const Tensor<1,C_DIM,double> dispAtom=quadPoint-atomLocation;
	      const double dist=dispAtom.norm();
	      const Tensor<1,C_DIM,double> temp=atomCharge*dispAtom/dist/dist/dist;

	      if (isCellOutsideVselfBall)
	      {
	       for (unsigned int idim=0; idim< C_DIM; idim++)
	       {
		 gradVselfQuads[q][idim][iSubCell]=temp[idim];
	       }
	      }

	      if (isCellOutsidePspTail)
	      {
	       for (unsigned int idim=0; idim< C_DIM; idim++)
	       {
		 gradPseudoVLocAtomsQuads[q][idim][iSubCell]=temp[idim];
	       }
	      }
	  }
       }
    }//subCell loop

    for (unsigned int q=0; q<numQuadPoints; ++q)
    {
        const Tensor<2,C_DIM,VectorizedArray<double> > EPSPStressContribution
	           =outer_product(eshelbyTensor::getFPSPLocal(
			                              rhoQuads[q],
		                                      gradPseudoVLocAtomsQuads[q],
		                                      gradVselfQuads[q]),xMinusAtomLoc[q])
		                                      *forceEval.JxW(q);
	for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
	{
	   for (unsigned int idim=0; idim<C_DIM; idim++)
	   {
	      for (unsigned int jdim=0; jdim<C_DIM; jdim++)
	      {
		  d_stress[idim][jdim]+=EPSPStressContribution[idim][jdim][iSubCell];
	      }
	   }
	}
    }
  }//iAtom loop
}
#endif
