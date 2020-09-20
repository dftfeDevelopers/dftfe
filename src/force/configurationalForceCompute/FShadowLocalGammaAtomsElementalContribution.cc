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
	void forceClass<FEOrder,FEOrderElectro>::FShadowLocalGammaAtomsElementalContribution
(std::map<unsigned int, std::vector<double> > & forceContributionLocalGammaAtoms,
 FEEvaluation<C_DIM,1,C_num1DQuad<C_rhoNodalPolyOrder<FEOrder,FEOrderElectro>()>(),C_DIM>  & forceEval,
 const MatrixFree<3,double> & matrixFreeData,
 const unsigned int cell,
 const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & gradRhoAtomsQuads,
 const std::vector< VectorizedArray<double> > & derVxcWithRhoOutTimesRhoDiffQuads,
 const std::vector< VectorizedArray<double> > & phiRhoMinusApproxRhoQuads,
 const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & hessianRhoAtomsQuads,
 const std::vector<Tensor<2,C_DIM,VectorizedArray<double> > >  & der2ExcWithGradRhoOutQuads,
 const std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >  & derVxcWithGradRhoOutQuads,
 const std::vector<Tensor<1,C_DIM,VectorizedArray<double> > >  & shadowKSGradRhoMinMinusGradRhoQuads,
 const std::vector<VectorizedArray<double> >  & shadowKSRhoMinMinusRhoQuads)
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

	for (unsigned int iAtom=0;iAtom <totalNumberAtoms; iAtom++)
	{
		std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradRhoQuadsiAtom(numQuadPoints,zeroTensor1);
		std::vector<Tensor<2,C_DIM,VectorizedArray<double> > > hessianRhoQuadsiAtom(numQuadPoints,zeroTensor2);

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

		if (gradRhoAtomsQuads.find(iAtom)==gradRhoAtomsQuads.end())
			continue;

		for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
		{
			subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell);
			dealii::CellId subCellId=subCellPtr->id();

			std::map<dealii::CellId, std::vector<double> >::const_iterator it
				=gradRhoAtomsQuads.find(iAtom)->second.find(subCellId);

			std::map<dealii::CellId, std::vector<double> >::const_iterator it2
				=hessianRhoAtomsQuads.find(iAtom)->second.find(subCellId);

			if (it!=gradRhoAtomsQuads.find(iAtom)->second.end())
				for (unsigned int q=0; q<numQuadPoints; ++q)
				{
					gradRhoQuadsiAtom[q][0][iSubCell]=(it->second)[q*C_DIM];
					gradRhoQuadsiAtom[q][1][iSubCell]=(it->second)[q*C_DIM+1];
					gradRhoQuadsiAtom[q][2][iSubCell]=(it->second)[q*C_DIM+2];

					if(dftParameters::xc_id == 4)
						for (unsigned int idim=0; idim<C_DIM; idim++)
							for (unsigned int jdim=0; jdim<C_DIM; jdim++)
								hessianRhoQuadsiAtom[q][idim][jdim][iSubCell]=(it2->second)[9*q+idim*C_DIM+jdim];
				}

		}//subCell loop

		if(dftParameters::xc_id == 4)
			for (unsigned int q=0; q<numQuadPoints; ++q)
				forceEval.submit_value(-gradRhoQuadsiAtom[q]*(derVxcWithRhoOutTimesRhoDiffQuads[q]+phiRhoMinusApproxRhoQuads[q])
						-shadowKSGradRhoMinMinusGradRhoQuads[q]*der2ExcWithGradRhoOutQuads[q]*hessianRhoQuadsiAtom[q]
						-shadowKSGradRhoMinMinusGradRhoQuads[q]*outer_product(derVxcWithGradRhoOutQuads[q],gradRhoQuadsiAtom[q])
						-shadowKSRhoMinMinusRhoQuads[q]*derVxcWithGradRhoOutQuads[q]*hessianRhoQuadsiAtom[q],
						q);
		else
			for (unsigned int q=0; q<numQuadPoints; ++q)
				forceEval.submit_value(-gradRhoQuadsiAtom[q]*(derVxcWithRhoOutTimesRhoDiffQuads[q]+phiRhoMinusApproxRhoQuads[q]),
						q);

		Tensor<1,C_DIM,VectorizedArray<double> > forceContributionLocalGammaiAtomCells
			=forceEval.integrate_value();

		if (forceContributionLocalGammaAtoms.find(atomId)==forceContributionLocalGammaAtoms.end())
			forceContributionLocalGammaAtoms[atomId]=std::vector<double>(C_DIM,0.0);

		for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
			for (unsigned int idim=0; idim<C_DIM; idim++)
				forceContributionLocalGammaAtoms[atomId][idim]+=
					forceContributionLocalGammaiAtomCells[idim][iSubCell];
	}//iAtom loop
}
