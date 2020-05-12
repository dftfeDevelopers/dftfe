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
// @author Sambit Das(2018)
//

#ifdef USE_COMPLEX
template<unsigned int FEOrder>
void forceClass<FEOrder>::computeStress
		 (const MatrixFree<3,double> & matrixFreeData,
		 const unsigned int eigenDofHandlerIndex,
		 const unsigned int phiExtDofHandlerIndex,
		 const unsigned int phiTotDofHandlerIndex,
		 const distributedCPUVec<double> & phiTotRhoIn,
		 const distributedCPUVec<double> & phiTotRhoOut,
		 const distributedCPUVec<double> & phiExt,
		 const std::map<dealii::CellId, std::vector<double> > & pseudoVLoc,
		 const std::map<dealii::CellId, std::vector<double> > & gradPseudoVLoc,
		 const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & gradPseudoVLocAtoms,
		 const ConstraintMatrix  & noConstraints,
		 const vselfBinsManager<FEOrder> & vselfBinsManagerEigen,
	         const MatrixFree<3,double> & matrixFreeDataElectro,
		 const unsigned int phiTotDofHandlerIndexElectro,
		 const unsigned int phiExtDofHandlerIndexElectro,
		 const distributedCPUVec<double> & phiTotRhoOutElectro,
		 const distributedCPUVec<double> & phiExtElectro,
		 const std::map<dealii::CellId, std::vector<double> > & rhoOutValuesElectro,
		 const std::map<dealii::CellId, std::vector<double> > & gradRhoOutValuesElectro,
		 const std::map<dealii::CellId, std::vector<double> > & pseudoVLocElectro,
		 const std::map<dealii::CellId, std::vector<double> > & gradPseudoVLocElectro,
		 const std::map<unsigned int,std::map<dealii::CellId, std::vector<double> > > & gradPseudoVLocAtomsElectro,
	         const ConstraintMatrix  & noConstraintsElectro,
		 const vselfBinsManager<FEOrder> & vselfBinsManagerElectro)
{
  /*
  createBinObjectsForce(matrixFreeData.get_dof_handler(phiTotDofHandlerIndex),
	                d_dofHandlerForce,
	                noConstraints,
	                vselfBinsManagerEigen,
                        d_cellsVselfBallsDofHandler,
                        d_cellsVselfBallsDofHandlerForce,
                        d_cellsVselfBallsClosestAtomIdDofHandler,
                        d_AtomIdBinIdLocalDofHandler,
                        d_cellFacesVselfBallSurfacesDofHandler,
                        d_cellFacesVselfBallSurfacesDofHandlerForce);
 */

  createBinObjectsForce(matrixFreeDataElectro.get_dof_handler(phiTotDofHandlerIndexElectro),
	                d_dofHandlerForceElectro,
	                noConstraintsElectro,
	                vselfBinsManagerElectro,
                        d_cellsVselfBallsDofHandlerElectro,
                        d_cellsVselfBallsDofHandlerForceElectro,
                        d_cellsVselfBallsClosestAtomIdDofHandlerElectro,
                        d_AtomIdBinIdLocalDofHandlerElectro,
                        d_cellFacesVselfBallSurfacesDofHandlerElectro,
                        d_cellFacesVselfBallSurfacesDofHandlerForceElectro);

  //reset to zero
  for (unsigned int idim=0; idim<C_DIM; idim++)
  {
    for (unsigned int jdim=0; jdim<C_DIM; jdim++)
    {
	d_stress[idim][jdim]=0.0;
	d_stressKPoints[idim][jdim]=0.0;
    }
  }

  //configurational stress contribution from all terms except those from nuclear self energy
  if (dftParameters::spinPolarized)
     computeStressSpinPolarizedEEshelbyEPSPEnlEk(matrixFreeData,
		                        eigenDofHandlerIndex,
		                        phiExtDofHandlerIndex,
		                        phiTotDofHandlerIndex,
		                        phiTotRhoIn,
		                        phiTotRhoOut,
		                        phiExt,
	                                pseudoVLoc,
					gradPseudoVLoc,
					gradPseudoVLocAtoms,
		                        vselfBinsManagerEigen,
	                                matrixFreeDataElectro,
		                        phiTotDofHandlerIndexElectro,
		                        phiExtDofHandlerIndexElectro,
		                        phiTotRhoOutElectro,
		                        phiExtElectro,
		                        rhoOutValuesElectro,
					gradRhoOutValuesElectro,
	                                pseudoVLocElectro,
					gradPseudoVLocElectro,
					gradPseudoVLocAtomsElectro,
					vselfBinsManagerElectro);
  else
     computeStressEEshelbyEPSPEnlEk(matrixFreeData,
		                        eigenDofHandlerIndex,
		                        phiExtDofHandlerIndex,
		                        phiTotDofHandlerIndex,
		                        phiTotRhoIn,
		                        phiTotRhoOut,
		                        phiExt,
	                                pseudoVLoc,
					gradPseudoVLoc,
					gradPseudoVLocAtoms,
		                        vselfBinsManagerEigen,
	                                matrixFreeDataElectro,
		                        phiTotDofHandlerIndexElectro,
		                        phiExtDofHandlerIndexElectro,
		                        phiTotRhoOutElectro,
		                        phiExtElectro,
		                        rhoOutValuesElectro,
					gradRhoOutValuesElectro,
	                                pseudoVLocElectro,
					gradPseudoVLocElectro,
					gradPseudoVLocAtomsElectro,
					vselfBinsManagerElectro);

  //configurational stress contribution from nuclear self energy. This is handled separately as it involves
  // a surface integral over the vself ball surface
  computeStressEself(matrixFreeDataElectro.get_dof_handler(phiTotDofHandlerIndexElectro),
	             vselfBinsManagerElectro);

  //Sum all processor contributions and distribute to all processors
  d_stress=Utilities::MPI::sum(d_stress,mpi_communicator);

  //Sum k point stress contribution over all processors
  //and k point pools and add to total stress
  d_stressKPoints=Utilities::MPI::sum(d_stressKPoints,mpi_communicator);
  d_stressKPoints=Utilities::MPI::sum(d_stressKPoints,dftPtr->interpoolcomm);
  d_stress+=d_stressKPoints;

  //Scale by inverse of domain volume
  d_stress=d_stress*(1.0/dftPtr->d_domainVolume);
}


template<unsigned int FEOrder>
void forceClass<FEOrder>::printStress()
{
    if (!dftParameters::reproducible_output)
    {
	pcout<<std::endl;
	pcout<<"Cell stress (Hartree/Bohr^3)"<<std::endl;
	pcout<< "------------------------------------------------------------------------"<< std::endl;
	for (unsigned int idim=0; idim< 3; idim++)
	    pcout<< d_stress[idim][0]<<"  "<<d_stress[idim][1]<<"  "<<d_stress[idim][2]<< std::endl;
	pcout<< "------------------------------------------------------------------------"<<std::endl;
    }
    else
    {
	pcout<<std::endl;
	pcout<<"Absolute value of cell stress (Hartree/Bohr^3)"<<std::endl;
	pcout<< "------------------------------------------------------------------------"<< std::endl;
	for (unsigned int idim=0; idim< 3; idim++)
	{
	   std::vector<double> truncatedStress(3);
	   for (unsigned int jdim=0; jdim< 3; jdim++)
		truncatedStress[jdim]  = std::fabs(std::floor(10000000 * d_stress[idim][jdim]) / 10000000.0);
	    pcout<<  std::fixed<<std::setprecision(6)<< truncatedStress[0]<<"  "<<truncatedStress[1]<<"  "<<truncatedStress[2]<< std::endl;
	}
	pcout<< "------------------------------------------------------------------------"<<std::endl;
    }

}

#endif
