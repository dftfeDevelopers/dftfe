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
template<unsigned int FEOrder>
void forceClass<FEOrder>::FPSPLocalGammaAtomsElementalContribution
             (std::map<unsigned int, std::vector<double> > & forceContributionFPSPLocalGammaAtoms,
	      FEValues<C_DIM> & feValues,
	      FEEvaluation<C_DIM,1,C_num1DQuad<FEOrder>(),C_DIM>  & forceEval,
	      const MatrixFree<3,double> & matrixFreeData,
	      const unsigned int cell,
	      const std::vector<VectorizedArray<double> > & rhoQuads,
              const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & gradPseudoVLocAtoms,
	      const vselfBinsManager<FEOrder> & vselfBinsManager,
	      const std::vector<std::map<dealii::CellId , unsigned int> > & cellsVselfBallsClosestAtomIdDofHandler)
{
  Tensor<1,C_DIM,VectorizedArray<double> > zeroTensor1;
  for (unsigned int idim=0; idim<C_DIM; idim++)
    zeroTensor1[idim]=make_vectorized_array(0.0);
  const unsigned int numberGlobalAtoms = dftPtr->atomLocations.size();
  const unsigned int numberImageCharges = dftPtr->d_imageIds.size();
  const unsigned int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;
  const unsigned int numSubCells= matrixFreeData.n_components_filled(cell);
  const unsigned int numQuadPoints=forceEval.n_q_points;
  DoFHandler<C_DIM>::active_cell_iterator subCellPtr;

  for (unsigned int iAtom=0;iAtom <totalNumberAtoms; iAtom++)
  {
    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradPseudoVLocAtomsQuads(numQuadPoints,zeroTensor1);
    std::vector<Tensor<1,C_DIM,VectorizedArray<double> > > gradVselfQuads(numQuadPoints,zeroTensor1);

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
       atomId=dftPtr->d_imageIds[imageId];
       atomCharge = dftPtr->d_imageCharges[imageId];
       atomLocation[0]=dftPtr->d_imagePositions[imageId][0];
       atomLocation[1]=dftPtr->d_imagePositions[imageId][1];
       atomLocation[2]=dftPtr->d_imagePositions[imageId][2];
    }

    if (gradPseudoVLocAtoms.find(iAtom)==gradPseudoVLocAtoms.end())
       continue;

    for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
    {
       subCellPtr= matrixFreeData.get_cell_iterator(cell,iSubCell);
       dealii::CellId subCellId=subCellPtr->id();

       std::map<dealii::CellId, std::vector<double> >::const_iterator it
	      =gradPseudoVLocAtoms.find(iAtom)->second.find(subCellId);
       if (it!=gradPseudoVLocAtoms.find(iAtom)->second.end())
       {
	    for (unsigned int q=0; q<numQuadPoints; ++q)
	    {
	       gradPseudoVLocAtomsQuads[q][0][iSubCell]=(it->second)[q*C_DIM];
	       gradPseudoVLocAtomsQuads[q][1][iSubCell]=(it->second)[q*C_DIM+1];
	       gradPseudoVLocAtomsQuads[q][2][iSubCell]=(it->second)[q*C_DIM+2];
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
    for (unsigned int iSubCell=0; iSubCell<numSubCells; ++iSubCell)
       for (unsigned int idim=0; idim<C_DIM; idim++)
       {
         forceContributionFPSPLocalGammaAtoms[atomId][idim]+=
	       forceContributionFPSPLocalGammaiAtomCells[idim][iSubCell];
       }
  }//iAtom loop
}

//(locally used function) accumulate and distribute FPSPLocal contibution due to Gamma(Rj)
template<unsigned int FEOrder>
void forceClass<FEOrder>::distributeForceContributionFPSPLocalGammaAtoms
              (const std::map<unsigned int,std::vector<double> > & forceContributionFPSPLocalGammaAtoms,
	       const std::map<std::pair<unsigned int,unsigned int>, unsigned int> & atomsForceDofs,
	       const ConstraintMatrix &  constraintsNoneForce,
	       vectorType & configForceVectorLinFE)
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
