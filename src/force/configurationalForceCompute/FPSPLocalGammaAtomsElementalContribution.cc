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
// @author Sambit Das (2017)
//

//(locally used function) compute FPSPLocal contibution due to Gamma(Rj) for given set of cells
template<unsigned int FEOrder,unsigned int FEOrderElectro>
	void forceClass<FEOrder,FEOrderElectro>::FPSPLocalGammaAtomsElementalContribution
(std::map<unsigned int, std::vector<double> > & forceContributionFPSPLocalGammaAtoms,
 FEValues<C_DIM> & feValues,
 FEFaceValues<C_DIM> & feFaceValues,
 FEEvaluation<C_DIM,1,C_num1DQuadLPSP<FEOrder>()*C_numCopies1DQuadLPSP(),C_DIM>  & forceEval,
 const MatrixFree<3,double> & matrixFreeData,
 const unsigned int phiTotDofHandlerIndexElectro,
 const unsigned int cell,
 const std::vector<VectorizedArray<double> > & rhoQuads, 
 const std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > & gradRhoQuads,
 const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & pseudoVLocAtoms,
 const vselfBinsManager<FEOrder,FEOrderElectro> & vselfBinsManager,
 const std::vector<std::map<dealii::CellId , unsigned int> > & cellsVselfBallsClosestAtomIdDofHandler)
{
	Tensor<1,C_DIM,VectorizedArray<double> > zeroTensor1;
	for (unsigned int idim=0; idim<C_DIM; idim++)
		zeroTensor1[idim]=make_vectorized_array(0.0);

	Tensor<1,C_DIM,double> zeroTensorNonvect;
	for (unsigned int idim=0; idim<C_DIM; idim++)
		zeroTensorNonvect[idim]=0.0;   

	const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
	const unsigned int numberImageCharges = dftPtr->d_imageIdsTrunc.size();
	const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
	const unsigned int numSubCells= matrixFreeData.n_components_filled(cell);
	const unsigned int numQuadPoints=forceEval.n_q_points;
  const unsigned int dofs_per_cell = matrixFreeData.get_dof_handler(0).get_fe().dofs_per_cell;

	const unsigned int faces_per_cell=GeometryInfo<C_DIM>::faces_per_cell;
  const unsigned int   numFaceQuadPoints = feFaceValues.get_quadrature().size();

  std::vector<Tensor<1,C_DIM,double> > surfaceIntegralSubcells(numSubCells);
  std::vector<double> rhoFaceQuads(numFaceQuadPoints);
	std::vector<VectorizedArray<double> > vselfQuads(numQuadPoints,make_vectorized_array(0.0));    
	std::vector<VectorizedArray<double> > pseudoVLocAtomsQuads(numQuadPoints,make_vectorized_array(0.0));
	std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > vselfDerRQuads(numQuadPoints,zeroTensor1);
  std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > totalContribution(numQuadPoints,zeroTensor1);
  std::vector<std::vector<dealii::Point<C_DIM> > > quadPointsSubCells(numSubCells,std::vector<dealii::Point<C_DIM> >(numQuadPoints));

	DoFHandler<C_DIM>::active_cell_iterator subCellPtr;

  for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
  {
    subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell,phiTotDofHandlerIndexElectro);
		feValues.reinit(subCellPtr);  

    std::vector<dealii::Point<C_DIM> > & temp=quadPointsSubCells[iSubCell];
    for (unsigned int q=0; q<numQuadPoints; ++q)
      temp[q]=feValues.quadrature_point(q);
  }

	for (unsigned int iAtom=0;iAtom <totalNumberAtoms; iAtom++)
	{
		bool isLocalDomainOutsideVselfBall=false;
		bool isLocalDomainOutsidePspTail= false;
		if (pseudoVLocAtoms.find(iAtom)==pseudoVLocAtoms.end())
			isLocalDomainOutsidePspTail=true;

		double atomCharge;
		unsigned int atomId=iAtom;
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

		unsigned int binIdiAtom;
		std::map<unsigned int,unsigned int>::const_iterator it1=
			vselfBinsManager.getAtomIdBinIdMapLocalAllImages().find(atomId);
		if (it1==vselfBinsManager.getAtomIdBinIdMapLocalAllImages().end())
			isLocalDomainOutsideVselfBall=true;
		else
			binIdiAtom=it1->second;

		//Assuming psp tail is larger than vself ball
		if (isLocalDomainOutsidePspTail && isLocalDomainOutsideVselfBall)
			continue;

    std::fill(surfaceIntegralSubcells.begin(),surfaceIntegralSubcells.end(),zeroTensorNonvect);
	  std::fill(vselfQuads.begin(),vselfQuads.end(),make_vectorized_array(0.0));    
	  std::fill(pseudoVLocAtomsQuads.begin(),pseudoVLocAtomsQuads.end(),make_vectorized_array(0.0));   
    std::fill(vselfDerRQuads.begin(),vselfDerRQuads.end(),zeroTensor1);
    std::fill(totalContribution.begin(),totalContribution.end(),zeroTensor1);

    bool isTrivial=true;
		for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
		{
			subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell,phiTotDofHandlerIndexElectro);
			dealii::CellId subCellId=subCellPtr->id();

			//get derivative R vself for iAtom
			bool isCellOutsideVselfBall=true;
			if (!isLocalDomainOutsideVselfBall)
			{
				std::map<dealii::CellId, unsigned int >::const_iterator
					it2=cellsVselfBallsClosestAtomIdDofHandler[binIdiAtom].find(subCellId);
				if (it2!=cellsVselfBallsClosestAtomIdDofHandler[binIdiAtom].end())
				{
					Point<C_DIM> closestAtomLocation;
					const unsigned int closestAtomId=it2->second;
					if(it2->second >= numberGlobalAtoms)
					{
						const unsigned int imageIdTrunc=closestAtomId-numberGlobalAtoms;
						closestAtomLocation[0]=dftPtr->d_imagePositionsTrunc[imageIdTrunc][0];
						closestAtomLocation[1]=dftPtr->d_imagePositionsTrunc[imageIdTrunc][1];
						closestAtomLocation[2]=dftPtr->d_imagePositionsTrunc[imageIdTrunc][2];
					}
					else
					{
						closestAtomLocation[0]=dftPtr->atomLocations[closestAtomId][2];
						closestAtomLocation[1]=dftPtr->atomLocations[closestAtomId][3];
						closestAtomLocation[2]=dftPtr->atomLocations[closestAtomId][4];
					}

					if(atomLocation.distance(closestAtomLocation)<1e-5)
		 			{
			      feValues.reinit(subCellPtr);            
						isCellOutsideVselfBall=false;

            if (dftParameters::floatingNuclearCharges && dftParameters::smearedNuclearCharges)
            {
              std::vector<double> vselfDerRQuadsSubCell(numQuadPoints);
              for (unsigned int idim=0; idim<3; ++idim)
              {
                feValues.get_function_values(vselfBinsManager.getVselfFieldDerRBins()[3*binIdiAtom+idim],
                   vselfDerRQuadsSubCell);
                for (unsigned int q=0; q<numQuadPoints; ++q)
                  vselfDerRQuads[q][idim][iSubCell]=vselfDerRQuadsSubCell[q];
              }
            }
            else if (!dftParameters::floatingNuclearCharges && dftParameters::smearedNuclearCharges)
            {
              for (unsigned int q=0; q<numQuadPoints; ++q)
              {
                Point<C_DIM> quadPoint=feValues.quadrature_point(q);
                Tensor<1,C_DIM,double> dispAtom=quadPoint-atomLocation;
                const double dist=dispAtom.norm();
                Tensor<1,C_DIM,double> temp= atomCharge*dftUtils::smearedPotDr(dist, dftPtr->d_smearedChargeWidths[atomId])*dispAtom/dist*dftPtr->d_smearedChargeScaling[atomId];
                vselfDerRQuads[q][0][iSubCell]=temp[0];
                vselfDerRQuads[q][1][iSubCell]=temp[1];
                vselfDerRQuads[q][2][iSubCell]=temp[2];
              }  
            }
            else
            {
              std::vector<Tensor<1,C_DIM,double> > gradVselfQuadsSubCell(numQuadPoints);
              feValues.get_function_gradients(vselfBinsManager.getVselfFieldBins()[binIdiAtom],
                  gradVselfQuadsSubCell);
              for (unsigned int q=0; q<numQuadPoints; ++q)
              {
                vselfDerRQuads[q][0][iSubCell]=-gradVselfQuadsSubCell[q][0];
                vselfDerRQuads[q][1][iSubCell]=-gradVselfQuadsSubCell[q][1];
                vselfDerRQuads[q][2][iSubCell]=-gradVselfQuadsSubCell[q][2];
              }
            }
            
					}
				}

			}

			//get grad pseudo VLoc for iAtom
			bool isCellOutsidePspTail=true;
			if (!isLocalDomainOutsidePspTail)
			{
				std::map<dealii::CellId, std::vector<double> >::const_iterator it
					=pseudoVLocAtoms.find(iAtom)->second.find(subCellId);
				if (it!=pseudoVLocAtoms.find(iAtom)->second.end())
				{
					isCellOutsidePspTail=false;
					for (unsigned int q=0; q<numQuadPoints; ++q)
						pseudoVLocAtomsQuads[q][iSubCell]=(it->second)[q];
				}
			}
      else if (!isCellOutsideVselfBall)
      {
        std::vector<dealii::Point<C_DIM> > & temp=quadPointsSubCells[iSubCell];        
				for (unsigned int q=0; q<numQuadPoints; ++q)
				{
					Tensor<1,C_DIM,double> dispAtom=temp[q]-atomLocation;
					const double dist=dispAtom.norm();
					pseudoVLocAtomsQuads[q][iSubCell]=-atomCharge/dist;
				}        
      }

			if (isCellOutsideVselfBall && !isCellOutsidePspTail)
      {
        std::vector<dealii::Point<C_DIM> > & temp=quadPointsSubCells[iSubCell];        
				for (unsigned int q=0; q<numQuadPoints; ++q)
				{
					Tensor<1,C_DIM,double> dispAtom=temp[q]-atomLocation;
					const double dist=dispAtom.norm();
					vselfQuads[q][iSubCell]=-atomCharge/dist;
				}
      }

      if (!isCellOutsideVselfBall)
      {
       
        Tensor<1,C_DIM,double> & surfaceIntegral=surfaceIntegralSubcells[iSubCell];

		    const std::map<DoFHandler<C_DIM>::active_cell_iterator,std::vector<unsigned int > >  & cellsVselfBallSurfacesDofHandler=d_cellFacesVselfBallSurfacesDofHandlerElectro[binIdiAtom];

        if (cellsVselfBallSurfacesDofHandler.find(subCellPtr)!=cellsVselfBallSurfacesDofHandler.end())
        {
          const std::vector<unsigned int > & dirichletFaceIds= cellsVselfBallSurfacesDofHandler.find(subCellPtr)->second;
          for (unsigned int index=0; index< dirichletFaceIds.size(); index++)
          {
            const unsigned int faceId=dirichletFaceIds[index];

            feFaceValues.reinit(d_cellIdToActiveCellIteratorMapDofHandlerRhoNodalElectro.find(subCellId)->second,faceId);
            feFaceValues.get_function_values(dftPtr->d_rhoOutNodalValuesDistributed,rhoFaceQuads);
            for (unsigned int qPoint=0; qPoint<numFaceQuadPoints; ++qPoint)
            {
              const Point<C_DIM> quadPoint=feFaceValues.quadrature_point(qPoint);
              const Tensor<1,C_DIM,double> dispClosestAtom=quadPoint-atomLocation;
              const double dist=dispClosestAtom.norm();
              const double vselfFaceQuadExact=-atomCharge/dist;

              surfaceIntegral-=rhoFaceQuads[qPoint]*vselfFaceQuadExact*feFaceValues.normal_vector(qPoint)*feFaceValues.JxW(qPoint);
            }//q point loop
          }//face loop
        }//surface cells
      }// inside or intersecting vself ball

      if (isCellOutsideVselfBall && !isCellOutsidePspTail)
      {
        isTrivial=false;
        for (unsigned int q=0; q<numQuadPoints; ++q)
          	for (unsigned int idim=0; idim<C_DIM; idim++)
              totalContribution[q][idim][iSubCell]=-gradRhoQuads[q][idim][iSubCell]*vselfQuads[q][iSubCell]+gradRhoQuads[q][idim][iSubCell]*pseudoVLocAtomsQuads[q][iSubCell];
      }
      else if (!isCellOutsideVselfBall)
      {
        isTrivial=false;
        for (unsigned int q=0; q<numQuadPoints; ++q)
          for (unsigned int idim=0; idim<C_DIM; idim++)
            totalContribution[q][idim][iSubCell]=-rhoQuads[q][iSubCell]*vselfDerRQuads[q][idim][iSubCell]+gradRhoQuads[q][idim][iSubCell]*pseudoVLocAtomsQuads[q][iSubCell];
      }
		}//subCell loop

    if (isTrivial)
       continue;

    for (unsigned int q=0; q<numQuadPoints; ++q)
      forceEval.submit_value(totalContribution[q],q);

		Tensor<1,C_DIM,VectorizedArray<double> > forceContributionFPSPLocalGammaiAtomCells
			=forceEval.integrate_value();

		if (forceContributionFPSPLocalGammaAtoms.find(atomId)==forceContributionFPSPLocalGammaAtoms.end())
			forceContributionFPSPLocalGammaAtoms[atomId]=std::vector<double>(C_DIM,0.0);
		for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
			for (unsigned int idim=0; idim<C_DIM; idim++)
			{
				forceContributionFPSPLocalGammaAtoms[atomId][idim]+=
					forceContributionFPSPLocalGammaiAtomCells[idim][iSubCell]+surfaceIntegralSubcells[iSubCell][idim];
			}
	}//iAtom loop
}

//(locally used function) accumulate and distribute FPSPLocal contibution due to Gamma(Rj)
template<unsigned int FEOrder,unsigned int FEOrderElectro>
	void forceClass<FEOrder,FEOrderElectro>::distributeForceContributionFPSPLocalGammaAtoms
(const std::map<unsigned int,std::vector<double> > & forceContributionFPSPLocalGammaAtoms,
 const std::map<std::pair<unsigned int,unsigned int>, unsigned int> & atomsForceDofs,
 const dealii::AffineConstraints<double> &  constraintsNoneForce,
 distributedCPUVec<double> & configForceVectorLinFE)
{
	for (unsigned int iAtom=0;iAtom <dftPtr->atomLocations.size(); iAtom++)
	{

		bool doesAtomIdExistOnLocallyOwnedNode=false;
		if (atomsForceDofs.find(std::pair<unsigned int,unsigned int>(iAtom,0))!=atomsForceDofs.end())
		{
			doesAtomIdExistOnLocallyOwnedNode=true;
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

		if (doesAtomIdExistOnLocallyOwnedNode)
		{
			std::vector<types::global_dof_index> forceLocalDofIndices(C_DIM);
			for (unsigned int idim=0; idim<C_DIM; idim++)
				forceLocalDofIndices[idim]=atomsForceDofs.find(std::pair<unsigned int,unsigned int>(iAtom,idim))->second;

			constraintsNoneForce.distribute_local_to_global
				(forceContributionFPSPLocalGammaiAtomGlobal,
				 forceLocalDofIndices,
				 configForceVectorLinFE);
		}
	}
}
