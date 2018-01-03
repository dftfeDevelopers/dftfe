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
// @author Sambit Das(2017)
//



//
//
//
//
template<unsigned int FEOrder>
void forceClass<FEOrder>::updateAtomPositionsAndMoveMesh(const std::vector<Point<C_DIM> > & globalAtomsDisplacements)
{
  std::vector<std::vector<double> > & atomLocations=dftPtr->atomLocations;
  std::vector<std::vector<double> > & imagePositions=dftPtr->d_imagePositions;
  const std::vector<int > & imageIds=dftPtr->d_imageIds;
  const int numberGlobalAtoms = atomLocations.size();
  const int numberImageCharges = imageIds.size();
  const int totalNumberAtoms = numberGlobalAtoms + numberImageCharges; 
  double dispTol=1e-8;

  std::vector<Point<C_DIM> > controlPointLocations;
  std::vector<Tensor<1,C_DIM,double> > controlPointDisplacements;
  
  for (unsigned int iAtom=0;iAtom <totalNumberAtoms; iAtom++){
     Point<C_DIM> atomCoor;
     int atomId=iAtom;
     if(iAtom < numberGlobalAtoms)
     {
        atomCoor[0] = atomLocations[iAtom][2];
        atomCoor[1] = atomLocations[iAtom][3];
        atomCoor[2] = atomLocations[iAtom][4];
        atomLocations[iAtom][2]+=globalAtomsDisplacements[atomId][0];
        atomLocations[iAtom][3]+=globalAtomsDisplacements[atomId][1];
        atomLocations[iAtom][4]+=globalAtomsDisplacements[atomId][2];	
     }
     else
     {
	atomCoor[0] = imagePositions[iAtom-numberGlobalAtoms][0];
	atomCoor[1] = imagePositions[iAtom-numberGlobalAtoms][1];
	atomCoor[2] = imagePositions[iAtom-numberGlobalAtoms][2];
	atomId=imageIds[iAtom-numberGlobalAtoms];
	imagePositions[iAtom-numberGlobalAtoms][0]+=globalAtomsDisplacements[atomId][0];
	imagePositions[iAtom-numberGlobalAtoms][1]+=globalAtomsDisplacements[atomId][1];
	imagePositions[iAtom-numberGlobalAtoms][2]+=globalAtomsDisplacements[atomId][2];
	
     }
     if (globalAtomsDisplacements[atomId].norm()>dispTol){
	   controlPointLocations.push_back(atomCoor);
	   controlPointDisplacements.push_back(globalAtomsDisplacements[atomId]);
     }
  }
  MPI_Barrier(mpi_communicator); 

  gaussianMove.moveMesh(controlPointLocations,controlPointDisplacements,d_gaussianConstant);
  pcout << "Reinitializing all moved triangulation dependent objects..." << std::endl;  
  
  //reinitialize dirichlet BCs for total potential and vSelf poisson solutions
  dftPtr->initBoundaryConditions();
  //reinitialize guesses for electron-density and wavefunctions (not required for relaxation update)
  //dftPtr->initElectronicFields();
  //reinitialize local pseudopotential
  if(dftParameters::isPseudopotential)
  {
     dftPtr->initLocalPseudoPotential();
     dftPtr->initNonLocalPseudoPotential();
     dftPtr->computeSparseStructureNonLocalProjectors();
     dftPtr->computeElementalProjectorKets();
  }
  
  pcout << "...Reinitialization end" << std::endl;   
}
