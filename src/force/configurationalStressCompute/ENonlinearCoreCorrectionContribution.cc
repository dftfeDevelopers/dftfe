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

//compute nonlinear core correction contribution to stress 
template<unsigned int FEOrder,unsigned int FEOrderElectro>
	void forceClass<FEOrder,FEOrderElectro>::addENonlinearCoreCorrectionStressContribution
(FEEvaluation<C_DIM,1,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),C_DIM>  & forceEval,
 const MatrixFree<3,double> & matrixFreeData,
 const unsigned int cell,
 const std::vector<VectorizedArray<double> > & vxcQuads,
 const std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > & derExcGradRho,
 const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & gradRhoCoreAtoms,
 const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & hessianRhoCoreAtoms)
{
  Tensor<1,C_DIM,VectorizedArray<double> > zeroTensor1;
  for (unsigned int idim=0; idim<C_DIM; idim++)
    zeroTensor1[idim]=make_vectorized_array(0.0);

  Tensor<2,C_DIM,VectorizedArray<double> > zeroTensor2;
  for(unsigned int idim = 0; idim < C_DIM; idim++)
    for(unsigned int jdim = 0; jdim < C_DIM; jdim++)
	    zeroTensor2[idim][jdim] = make_vectorized_array(0.0);

  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numberImageCharges = dftPtr->d_imageIdsTrunc.size();
  const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
  const unsigned int numSubCells= matrixFreeData.n_components_filled(cell);
  const unsigned int numQuadPoints=forceEval.n_q_points;
  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;

  std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > xMinusAtomLoc(numQuadPoints,zeroTensor1);


  for (unsigned int iAtom = 0;iAtom < totalNumberAtoms; iAtom++)
  {
    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoCoreAtomsQuads(numQuadPoints,zeroTensor1);    
    std::vector<Tensor<2,C_DIM,VectorizedArray<double> > > hessianRhoCoreAtomsQuads(numQuadPoints,zeroTensor2);
    
    double atomCharge;
    unsigned int atomId = iAtom;
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
    else
    {
       const int imageId=iAtom-numberGlobalAtoms;
       atomId=dftPtr->d_imageIdsTrunc[imageId];
       atomCharge = dftPtr->d_imageChargesTrunc[imageId];
       atomLocation[0]=dftPtr->d_imagePositionsTrunc[imageId][0];
       atomLocation[1]=dftPtr->d_imagePositionsTrunc[imageId][1];
       atomLocation[2]=dftPtr->d_imagePositionsTrunc[imageId][2];
    }

    bool isLocalDomainOutsideCoreRhoTail= false;
    if(gradRhoCoreAtoms.find(iAtom)==gradRhoCoreAtoms.end())
       isLocalDomainOutsideCoreRhoTail = true;
    
    if(isLocalDomainOutsideCoreRhoTail)
       continue;

    bool isCellOutsideCoreRhoTail = true;

    for(unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
    {
       subCellPtr = matrixFreeData.get_cell_iterator(cell,iSubCell);
       dealii::CellId subCellId = subCellPtr->id();
 
       std::map<dealii::CellId, std::vector<double> >::const_iterator it
          =gradRhoCoreAtoms.find(iAtom)->second.find(subCellId);
       if(it!=gradRhoCoreAtoms.find(iAtom)->second.end())
       {
         isCellOutsideCoreRhoTail=false;
         const std::vector<double> & temp=it->second;
         for (unsigned int q=0; q<numQuadPoints; ++q)
         {
            gradRhoCoreAtomsQuads[q][0][iSubCell]=temp[q*C_DIM];
            gradRhoCoreAtomsQuads[q][1][iSubCell]=temp[q*C_DIM+1];
            gradRhoCoreAtomsQuads[q][2][iSubCell]=temp[q*C_DIM+2];
         }
       }

       if(dftParameters::xcFamilyType=="GGA" && !isCellOutsideCoreRhoTail)
       {
         std::map<dealii::CellId, std::vector<double> >::const_iterator it2
              =hessianRhoCoreAtoms.find(iAtom)->second.find(subCellId);
          
         if(it2!=hessianRhoCoreAtoms.find(iAtom)->second.end())
         {
            const std::vector<double> & temp=it2->second;            
            for (unsigned int q=0; q<numQuadPoints; ++q)
            {
              for(unsigned int iDim = 0; iDim < 3; ++iDim)
                for(unsigned int jDim = 0; jDim < 3; ++jDim)
                 hessianRhoCoreAtomsQuads[q][iDim][jDim][iSubCell] = temp[q*C_DIM*C_DIM + C_DIM*iDim + jDim];
            }
         }
       }

       if (!isCellOutsideCoreRhoTail)
         for (unsigned int q=0; q<numQuadPoints; ++q)
         {
            const Point<C_DIM,VectorizedArray<double>> & quadPointVectorized=forceEval.quadrature_point(q);
            Point<C_DIM> quadPoint;
            quadPoint[0]=quadPointVectorized[0][iSubCell];
            quadPoint[1]=quadPointVectorized[1][iSubCell];
            quadPoint[2]=quadPointVectorized[2][iSubCell];            
            const Tensor<1,C_DIM,double> dispAtom=quadPoint-atomLocation;
            for (unsigned int idim=0; idim<C_DIM; idim++)
            {
              xMinusAtomLoc[q][idim][iSubCell]=dispAtom[idim];
            }
         }          
    }//subCell loop

    if (isCellOutsideCoreRhoTail)
      continue;


		Tensor<2,C_DIM,VectorizedArray<double> > stressContribution=zeroTensor2;

		for (unsigned int q=0; q<numQuadPoints; ++q)
		{

			stressContribution
				+=outer_product(eshelbyTensor::getFNonlinearCoreCorrection(vxcQuads[q],gradRhoCoreAtomsQuads[q])+eshelbyTensor::getFNonlinearCoreCorrection(derExcGradRho[q],hessianRhoCoreAtomsQuads[q]),xMinusAtomLoc[q])
				*forceEval.JxW(q);
		}

		for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
			for (unsigned int idim=0; idim<C_DIM; idim++)
				for (unsigned int jdim=0; jdim<C_DIM; jdim++)
					d_stress[idim][jdim]+=stressContribution[idim][jdim][iSubCell];
	}//iAtom loop
}

//compute nonlinear core correction contribution to stress 
template<unsigned int FEOrder,unsigned int FEOrderElectro>
	void forceClass<FEOrder,FEOrderElectro>::addENonlinearCoreCorrectionStressContributionSpinPolarized
(FEEvaluation<C_DIM,1,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),C_DIM>  & forceEval,
  const MatrixFree<3,double> & matrixFreeData,
  const unsigned int cell,
  const std::vector<VectorizedArray<double> > & vxcQuadsSpin0,
  const std::vector<VectorizedArray<double> > & vxcQuadsSpin1,        
  const std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > & derExcGradRhoSpin0,
  const std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > & derExcGradRhoSpin1,     
  const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & gradRhoCoreAtoms,
  const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & hessianRhoCoreAtoms,
  const bool isXCGGA) 
{
  Tensor<1,C_DIM,VectorizedArray<double> > zeroTensor1;
  for (unsigned int idim=0; idim<C_DIM; idim++)
    zeroTensor1[idim]=make_vectorized_array(0.0);

  Tensor<2,C_DIM,VectorizedArray<double> > zeroTensor2;
  for(unsigned int idim = 0; idim < C_DIM; idim++)
    for(unsigned int jdim = 0; jdim < C_DIM; jdim++)
	    zeroTensor2[idim][jdim] = make_vectorized_array(0.0);

  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numberImageCharges = dftPtr->d_imageIdsTrunc.size();
  const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
  const unsigned int numSubCells= matrixFreeData.n_components_filled(cell);
  const unsigned int numQuadPoints=forceEval.n_q_points;
  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;

  std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > xMinusAtomLoc(numQuadPoints,zeroTensor1);


  for (unsigned int iAtom = 0;iAtom < totalNumberAtoms; iAtom++)
  {
    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoCoreAtomsQuads(numQuadPoints,zeroTensor1);    
    std::vector<Tensor<2,C_DIM,VectorizedArray<double> > > hessianRhoCoreAtomsQuads(numQuadPoints,zeroTensor2);
    
    double atomCharge;
    unsigned int atomId = iAtom;
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
    else
    {
       const int imageId=iAtom-numberGlobalAtoms;
       atomId=dftPtr->d_imageIdsTrunc[imageId];
       atomCharge = dftPtr->d_imageChargesTrunc[imageId];
       atomLocation[0]=dftPtr->d_imagePositionsTrunc[imageId][0];
       atomLocation[1]=dftPtr->d_imagePositionsTrunc[imageId][1];
       atomLocation[2]=dftPtr->d_imagePositionsTrunc[imageId][2];
    }

    bool isLocalDomainOutsideCoreRhoTail= false;
    if(gradRhoCoreAtoms.find(iAtom)==gradRhoCoreAtoms.end())
       isLocalDomainOutsideCoreRhoTail = true;
    
    if(isLocalDomainOutsideCoreRhoTail)
       continue;

    bool isCellOutsideCoreRhoTail = true;

    for(unsigned int iSubCell = 0; iSubCell < numSubCells; ++iSubCell)
    {
       subCellPtr = matrixFreeData.get_cell_iterator(cell,iSubCell);
       dealii::CellId subCellId = subCellPtr->id();
 
       std::map<dealii::CellId, std::vector<double> >::const_iterator it
          =gradRhoCoreAtoms.find(iAtom)->second.find(subCellId);
       if(it!=gradRhoCoreAtoms.find(iAtom)->second.end())
       {
         isCellOutsideCoreRhoTail=false;
         const std::vector<double> & temp=it->second;
         for (unsigned int q=0; q<numQuadPoints; ++q)
         {
            gradRhoCoreAtomsQuads[q][0][iSubCell]=temp[q*C_DIM]/2.0;
            gradRhoCoreAtomsQuads[q][1][iSubCell]=temp[q*C_DIM+1]/2.0;
            gradRhoCoreAtomsQuads[q][2][iSubCell]=temp[q*C_DIM+2]/2.0;
         }
       }

       if(dftParameters::xcFamilyType=="GGA" && !isCellOutsideCoreRhoTail)
       {
         std::map<dealii::CellId, std::vector<double> >::const_iterator it2
              =hessianRhoCoreAtoms.find(iAtom)->second.find(subCellId);
          
         if(it2!=hessianRhoCoreAtoms.find(iAtom)->second.end())
         {
            const std::vector<double> & temp2=it2->second;            
            for (unsigned int q=0; q<numQuadPoints; ++q)
            {
              for(unsigned int iDim = 0; iDim < 3; ++iDim)
                for(unsigned int jDim = 0; jDim < 3; ++jDim)
                 hessianRhoCoreAtomsQuads[q][iDim][jDim][iSubCell] = temp2[q*C_DIM*C_DIM + C_DIM*iDim + jDim]/2.0;
            }
         }
       }

       if (!isCellOutsideCoreRhoTail)
         for (unsigned int q=0; q<numQuadPoints; ++q)
         {
            const Point<C_DIM,VectorizedArray<double>> & quadPointVectorized=forceEval.quadrature_point(q);
            Point<C_DIM> quadPoint;
            quadPoint[0]=quadPointVectorized[0][iSubCell];
            quadPoint[1]=quadPointVectorized[1][iSubCell];
            quadPoint[2]=quadPointVectorized[2][iSubCell];            
            const Tensor<1,C_DIM,double> dispAtom=quadPoint-atomLocation;
            for (unsigned int idim=0; idim<C_DIM; idim++)
            {
              xMinusAtomLoc[q][idim][iSubCell]=dispAtom[idim];
            }
         }          
    }//subCell loop

    if (isCellOutsideCoreRhoTail)
      continue;


		Tensor<2,C_DIM,VectorizedArray<double> > stressContribution=zeroTensor2;

		for (unsigned int q=0; q<numQuadPoints; ++q)
        stressContribution+=outer_product(eshelbyTensorSP::getFNonlinearCoreCorrection(vxcQuadsSpin0[q],
                                                                                       vxcQuadsSpin1[q],
                                                                                       derExcGradRhoSpin0[q],
                                                                                       derExcGradRhoSpin1[q],
                                                                                       gradRhoCoreAtomsQuads[q],
                                                                                       hessianRhoCoreAtomsQuads[q],
                                                                                       isXCGGA),xMinusAtomLoc[q])*forceEval.JxW(q);

		for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
			for (unsigned int idim=0; idim<C_DIM; idim++)
				for (unsigned int jdim=0; jdim<C_DIM; jdim++)
					d_stress[idim][jdim]+=stressContribution[idim][jdim][iSubCell];
	}//iAtom loop
}
