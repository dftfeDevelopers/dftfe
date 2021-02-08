// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2020 The Regents of the University of Michigan and DFT-FE authors.
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

//(locally used function) compute FShadowLocal contibution due to Gamma(Rj) for given set of cells
template<unsigned int FEOrder,unsigned int FEOrderElectro>
	void forceClass<FEOrder,FEOrderElectro>::FShadowLocalGammaAtomsElementalContributionElectronic
(std::map<unsigned int, std::vector<double> > & forceContributionLocalGammaAtoms,
 FEEvaluation<C_DIM,1,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),C_DIM>  & forceEval,
 const MatrixFree<3,double> & matrixFreeData,
 const unsigned int cell,
 const std::vector<VectorizedArray<double> > & derVxcWithRhoTimesRhoDiffQuads,
 const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & gradRhoAtomsQuadsSeparate,
 const std::vector<Tensor<2,C_DIM,VectorizedArray<double> > >  & der2ExcWithGradRhoQuads,
 const std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >  & derVxcWithGradRhoQuads,
 const std::vector<VectorizedArray<double> >  & shadowKSRhoMinMinusRhoQuads,         
 const std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >  & shadowKSGradRhoMinMinusGradRhoQuads,         
 const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & hessianRhoAtomsQuadsSeparate,         
 const bool isXCGGA) 
{
  Tensor<1,C_DIM,VectorizedArray<double> > zeroTensor1;
  for (unsigned int idim=0; idim<C_DIM; idim++)
    zeroTensor1[idim]=make_vectorized_array(0.0);

	Tensor<2,C_DIM,VectorizedArray<double> > zeroTensor2;
	for (unsigned int idim=0; idim<C_DIM; idim++)
		for (unsigned int jdim=0; jdim<C_DIM; jdim++)
			zeroTensor2[idim][jdim]=make_vectorized_array(0.0);

  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numberImageCharges = dftPtr->d_imageIdsTrunc.size();
  const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
  const unsigned int numSubCells= matrixFreeData.n_components_filled(cell);
  const unsigned int numQuadPoints=forceEval.n_q_points;
  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;

  for (unsigned int iAtom = 0;iAtom < totalNumberAtoms; iAtom++)
  {
    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoAtomQuads(numQuadPoints,zeroTensor1);
		std::vector<Tensor<2,C_DIM,VectorizedArray<double> > > hessianRhoAtomQuads(numQuadPoints,zeroTensor2);   

    unsigned int atomId = iAtom;
    if(iAtom >= numberGlobalAtoms)
    {
       const int imageId=iAtom-numberGlobalAtoms;
       atomId=dftPtr->d_imageIdsTrunc[imageId];
    }

    bool isLocalDomainOutsideRhoTail= false;
    if(gradRhoAtomsQuadsSeparate.find(iAtom)==gradRhoAtomsQuadsSeparate.end())
       isLocalDomainOutsideRhoTail = true;
    
    if(isLocalDomainOutsideRhoTail)
       continue;

    bool isCellOutsideRhoTail = true;
    for(unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
    {
       subCellPtr = matrixFreeData.get_cell_iterator(cell,iSubCell);
       dealii::CellId subCellId = subCellPtr->id();
  
       //get grad rho for iAtom
       if(!isLocalDomainOutsideRhoTail)
       {
        std::map<dealii::CellId, std::vector<double> >::const_iterator it
            =gradRhoAtomsQuadsSeparate.find(iAtom)->second.find(subCellId);
        if(it!=gradRhoAtomsQuadsSeparate.find(iAtom)->second.end())
        {
          isCellOutsideRhoTail=false;
          const std::vector<double> & temp=it->second;
          for (unsigned int q=0; q<numQuadPoints; ++q)
          {
             gradRhoAtomQuads[q][0][iSubCell]=temp[q*C_DIM];
             gradRhoAtomQuads[q][1][iSubCell]=temp[q*C_DIM+1];
             gradRhoAtomQuads[q][2][iSubCell]=temp[q*C_DIM+2];
          }
        }

        if (isXCGGA)
        {
          std::map<dealii::CellId, std::vector<double> >::const_iterator it2
              =hessianRhoAtomsQuadsSeparate.find(iAtom)->second.find(subCellId);
          if(it2!=hessianRhoAtomsQuadsSeparate.find(iAtom)->second.end())
          {
            const std::vector<double> & temp=it2->second;
            for (unsigned int q=0; q<numQuadPoints; ++q)
            {
              for (unsigned int idim=0; idim<C_DIM; idim++)
                for (unsigned int jdim=0; jdim<C_DIM; jdim++)
                  hessianRhoAtomQuads[q][idim][jdim][iSubCell]=(it2->second)[9*q+idim*C_DIM+jdim];
            }
          }          
        }
       }
    }//subCell loop

    if (isCellOutsideRhoTail)
      continue;

    for(unsigned int q=0; q<numQuadPoints; ++q)
    {
      if (isXCGGA)
      {
				forceEval.submit_value(-derVxcWithRhoTimesRhoDiffQuads[q]*gradRhoAtomQuads[q]
						-shadowKSGradRhoMinMinusGradRhoQuads[q]*(der2ExcWithGradRhoQuads[q]*hessianRhoAtomQuads[q])
						-shadowKSGradRhoMinMinusGradRhoQuads[q]*outer_product(derVxcWithGradRhoQuads[q],gradRhoAtomQuads[q])
						-shadowKSRhoMinMinusRhoQuads[q]*derVxcWithGradRhoQuads[q]*hessianRhoAtomQuads[q],
						q);        
      }
      else
        forceEval.submit_value(-derVxcWithRhoTimesRhoDiffQuads[q]*gradRhoAtomQuads[q],q);
    }
    Tensor<1,C_DIM,VectorizedArray<double> > forceContributionLocalGammaiAtom
						 = forceEval.integrate_value();

    if(forceContributionLocalGammaAtoms.find(atomId)==forceContributionLocalGammaAtoms.end())
       forceContributionLocalGammaAtoms[atomId]=std::vector<double>(C_DIM,0.0);
    for(unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
       for(unsigned int idim=0; idim<C_DIM; idim++)
       {
         forceContributionLocalGammaAtoms[atomId][idim]+=
	       forceContributionLocalGammaiAtom[idim][iSubCell];
       }
  }//iAtom loop
}

template<unsigned int FEOrder,unsigned int FEOrderElectro>
void forceClass<FEOrder,FEOrderElectro>::FShadowLocalGammaAtomsElementalContributionElectrostatic
      (std::map<unsigned int, std::vector<double> > & forceContributionLocalGammaAtoms,
	      FEEvaluation<C_DIM,1,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),C_DIM>  & forceEval,
	      const MatrixFree<3,double> & matrixFreeData,
	      const unsigned int cell,
	      const std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > & gradPhiRhoMinusApproxRhoElectroQuads,
        const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & rhoAtomsQuadsSeparate)
{
  Tensor<1,C_DIM,VectorizedArray<double> > zeroTensor1;
  for (unsigned int idim=0; idim<C_DIM; idim++)
    zeroTensor1[idim]=make_vectorized_array(0.0);
  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numberImageCharges = dftPtr->d_imageIdsTrunc.size();
  const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
  const unsigned int numSubCells= matrixFreeData.n_components_filled(cell);
  const unsigned int numQuadPoints=forceEval.n_q_points;
  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;

  for (unsigned int iAtom = 0;iAtom < totalNumberAtoms; iAtom++)
  {
    std::vector<VectorizedArray<double> > rhoAtomQuads(numQuadPoints,make_vectorized_array(0.0));
    
    unsigned int atomId = iAtom;
    if(iAtom >= numberGlobalAtoms)
    {
       const int imageId=iAtom-numberGlobalAtoms;
       atomId=dftPtr->d_imageIdsTrunc[imageId];
    }

    bool isLocalDomainOutsideRhoTail= false;
    if(rhoAtomsQuadsSeparate.find(iAtom)==rhoAtomsQuadsSeparate.end())
       isLocalDomainOutsideRhoTail = true;
    
    if(isLocalDomainOutsideRhoTail)
       continue;

    bool isCellOutsideRhoTail = true;
    for(unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
    {
       subCellPtr = matrixFreeData.get_cell_iterator(cell,iSubCell);
       dealii::CellId subCellId = subCellPtr->id();
  
       //get grad rho for iAtom
       if(!isLocalDomainOutsideRhoTail)
       {
        std::map<dealii::CellId, std::vector<double> >::const_iterator it
            =rhoAtomsQuadsSeparate.find(iAtom)->second.find(subCellId);
        if(it!=rhoAtomsQuadsSeparate.find(iAtom)->second.end())
        {
          isCellOutsideRhoTail=false;
          const std::vector<double> & temp=it->second;
          for (unsigned int q=0; q<numQuadPoints; ++q)
          {
             rhoAtomQuads[q][iSubCell]=temp[q];
          }
        }
       }
    }//subCell loop

    if (isCellOutsideRhoTail)
      continue;

    for(unsigned int q=0; q<numQuadPoints; ++q)
    {
      forceEval.submit_value(gradPhiRhoMinusApproxRhoElectroQuads[q]*rhoAtomQuads[q],q);
    }
    Tensor<1,C_DIM,VectorizedArray<double> > forceContributionLocalGammaiAtom
						 = forceEval.integrate_value();

    if(forceContributionLocalGammaAtoms.find(atomId)==forceContributionLocalGammaAtoms.end())
       forceContributionLocalGammaAtoms[atomId]=std::vector<double>(C_DIM,0.0);
    for(unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
       for(unsigned int idim=0; idim<C_DIM; idim++)
       {
         forceContributionLocalGammaAtoms[atomId][idim]+=
	       forceContributionLocalGammaiAtom[idim][iSubCell];
       }
  }//iAtom loop
}
