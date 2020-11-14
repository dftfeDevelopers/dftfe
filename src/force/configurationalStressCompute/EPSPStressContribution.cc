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

//compute EPSP contribution stress (local pseudopotential)
template<unsigned int FEOrder,unsigned int FEOrderElectro>
	void forceClass<FEOrder,FEOrderElectro>::addEPSPStressContribution
(FEValues<C_DIM> & feValues,
 FEEvaluation<C_DIM,1,C_num1DQuadLPSP<FEOrder>()*C_numCopies1DQuadLPSP(),C_DIM>  & forceEval,
 const MatrixFree<3,double> & matrixFreeData,
 const unsigned int phiTotDofHandlerIndexElectro,
 const unsigned int cell,
 const std::vector< Tensor<1,3,VectorizedArray<double> >  > & gradRhoQuads,
 const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & pseudoVLocAtoms,
 const vselfBinsManager<FEOrder,FEOrderElectro> & vselfBinsManager,
 const std::vector<std::map<dealii::CellId , unsigned int> > & cellsVselfBallsClosestAtomIdDofHandler)
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

  std::vector<VectorizedArray<double> > pseudoVLocAtomsQuads(numQuadPoints,make_vectorized_array(0.0));
  std::vector<VectorizedArray<double> > vselfQuads(numQuadPoints,make_vectorized_array(0.0));
  std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > xMinusAtomLoc(numQuadPoints,zeroTensor1);

	for (unsigned int iAtom=0;iAtom <totalNumberAtoms; iAtom++)
	{
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
			atomId=dftPtr->d_imageIdsTrunc[imageId];
			atomCharge = dftPtr->d_imageChargesTrunc[imageId];
			atomLocation[0]=dftPtr->d_imagePositionsTrunc[imageId][0];
			atomLocation[1]=dftPtr->d_imagePositionsTrunc[imageId][1];
			atomLocation[2]=dftPtr->d_imagePositionsTrunc[imageId][2];
		}

		bool isLocalDomainOutsideVselfBall=false;
		bool isLocalDomainOutsidePspTail= false;
		if (pseudoVLocAtoms.find(iAtom)==pseudoVLocAtoms.end())
			isLocalDomainOutsidePspTail=true;

		unsigned int binIdiAtom;
		std::map<unsigned int,unsigned int>::const_iterator it1=
			vselfBinsManager.getAtomIdBinIdMapLocalAllImages().find(atomId);
		if (it1==vselfBinsManager.getAtomIdBinIdMapLocalAllImages().end())
			isLocalDomainOutsideVselfBall=true;
		else
			binIdiAtom=it1->second;

		if (isLocalDomainOutsidePspTail && isLocalDomainOutsideVselfBall)
			continue;



		for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
		{
			subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell,phiTotDofHandlerIndexElectro);
			dealii::CellId subCellId=subCellPtr->id();
			feValues.reinit(subCellPtr);

			for (unsigned int q=0; q<numQuadPoints; ++q)
			{
				const Point<C_DIM> & quadPoint=feValues.quadrature_point(q);
				const Tensor<1,C_DIM,double> dispAtom=quadPoint-atomLocation;
				for (unsigned int idim=0; idim<C_DIM; idim++)
				{
					xMinusAtomLoc[q][idim][iSubCell]=dispAtom[idim];
				}
			}

			// get computed grad vself for iAtom
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
						isCellOutsideVselfBall=false;
						std::vector<double > vselfQuadsSubCell(numQuadPoints);
						feValues.get_function_values(vselfBinsManager.getVselfFieldBins()[binIdiAtom],vselfQuadsSubCell);
						for (unsigned int q=0; q<numQuadPoints; ++q)
							vselfQuads[q][iSubCell]=vselfQuadsSubCell[q];
					}
				}
			}

			// get exact solution (-Z/r) for grad vself  for iAtom
			if (isCellOutsideVselfBall)
				for (unsigned int q=0; q<numQuadPoints; ++q)
				{
					const Point<C_DIM> & quadPoint=feValues.quadrature_point(q);
					const Tensor<1,C_DIM,double> dispAtom=quadPoint-atomLocation;
					const double dist=dispAtom.norm();
					vselfQuads[q][iSubCell]=-atomCharge/dist;
				}


			//get computed grad pseudo VLoc for iAtom
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

			// get exact solution (Z/r^2) for grad pseudo VLoc for iAtom
			if (isCellOutsidePspTail)
				for (unsigned int q=0; q<numQuadPoints; ++q)
				{
					const Point<C_DIM> & quadPoint=feValues.quadrature_point(q);
					const Tensor<1,C_DIM,double> dispAtom=quadPoint-atomLocation;
					const double dist=dispAtom.norm();
					pseudoVLocAtomsQuads[q][iSubCell]=-atomCharge/dist;
				}
		}//subCell loop

		Tensor<2,C_DIM,VectorizedArray<double> > EPSPStressContribution;
		for (unsigned int idim=0; idim<C_DIM; idim++)
			for (unsigned int jdim=0; jdim<C_DIM; jdim++)
				EPSPStressContribution[idim][jdim]=make_vectorized_array(0.0);

		for (unsigned int q=0; q<numQuadPoints; ++q)
		{

			EPSPStressContribution
				-=outer_product(gradRhoQuads[q]*(pseudoVLocAtomsQuads[q]-vselfQuads[q]),xMinusAtomLoc[q])
				*forceEval.JxW(q);
		}

		for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
			for (unsigned int idim=0; idim<C_DIM; idim++)
				for (unsigned int jdim=0; jdim<C_DIM; jdim++)
					d_stress[idim][jdim]+=EPSPStressContribution[idim][jdim][iSubCell];
	}//iAtom loop
}
