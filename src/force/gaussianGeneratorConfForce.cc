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

//Configurational force on atoms corresponding to Gaussian generator. Generator is discretized using linear FE shape functions. Configurational force on nodes due to linear FE shape functions precomputed 
template<unsigned int FEOrder>
void forceClass<FEOrder>::computeAtomsForcesGaussianGenerator(bool allowGaussianOverlapOnAtoms)
{
  const std::vector<std::vector<double> > & atomLocations=dftPtr->atomLocations;
  const std::vector<std::vector<double> > & imagePositions=dftPtr->d_imagePositions;
  const std::vector<int > & imageIds=dftPtr->d_imageIds;
  const int numberGlobalAtoms = atomLocations.size();
  const int numberImageCharges = imageIds.size();
  const int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;  
  std::vector<double> globalAtomsGaussianForcesLocalPart(numberGlobalAtoms*C_DIM,0);
  d_globalAtomsGaussianForces.resize(numberGlobalAtoms*C_DIM);
  //std::fill(globalAtomsGaussianForces.begin(), globalAtomsGaussianForces.end(), 0);
  std::map<types::global_dof_index,Point<C_DIM> >::iterator iterMap;

  for (unsigned int iAtom=0;iAtom <totalNumberAtoms; iAtom++){
     //if (iAtom>0)
     //   continue;
     Point<C_DIM> atomCoor;
     int atomId=iAtom;
     if(iAtom < numberGlobalAtoms)
     {
        atomCoor[0] = atomLocations[iAtom][2];
        atomCoor[1] = atomLocations[iAtom][3];
        atomCoor[2] = atomLocations[iAtom][4];
      }
      else
      {
	atomCoor[0] = imagePositions[iAtom-numberGlobalAtoms][0];
	atomCoor[1] = imagePositions[iAtom-numberGlobalAtoms][1];
	atomCoor[2] = imagePositions[iAtom-numberGlobalAtoms][2];
	atomId=imageIds[iAtom-numberGlobalAtoms];
      }

      for(iterMap = d_locallyOwnedSupportPointsForceX.begin(); iterMap != d_locallyOwnedSupportPointsForceX.end(); ++iterMap)
      {
	  const int globalDofIndex=iterMap->first; 
	  Point<3> nodalCoor = iterMap->second;
          bool isGaussianOverlapOtherAtom=false;
	  for (unsigned int jAtom=0;jAtom <totalNumberAtoms; jAtom++){
	     if (iAtom !=jAtom){
               Point<C_DIM> jAtomCoor;
               if(jAtom < numberGlobalAtoms)
               {
                 jAtomCoor[0] = atomLocations[jAtom][2];
                 jAtomCoor[1] = atomLocations[jAtom][3];
                 jAtomCoor[2] = atomLocations[jAtom][4];
               }
               else
               {
	         jAtomCoor[0] = imagePositions[jAtom-numberGlobalAtoms][0];
	         jAtomCoor[1] = imagePositions[jAtom-numberGlobalAtoms][1];
	         jAtomCoor[2] = imagePositions[jAtom-numberGlobalAtoms][2];
               }
               const double distanceSq=(nodalCoor-jAtomCoor).norm_square();
	       if (distanceSq < 1e-6){
		   isGaussianOverlapOtherAtom=true;
		   break;
	       }
	     }
	  }
	  if (d_constraintsNoneForce.is_constrained(globalDofIndex)|| (isGaussianOverlapOtherAtom && !allowGaussianOverlapOnAtoms))
	     continue;		  
	  const double rsq= (nodalCoor-atomCoor).norm_square();
	  globalAtomsGaussianForcesLocalPart[C_DIM*atomId]+=std::exp(-d_gaussianConstant*rsq)*d_configForceVectorLinFE[globalDofIndex];

      }//x component support points loop

      for(iterMap = d_locallyOwnedSupportPointsForceY.begin(); iterMap != d_locallyOwnedSupportPointsForceY.end(); ++iterMap)
      {
	  const int globalDofIndex=iterMap->first; 
	  Point<3> nodalCoor = iterMap->second;
          bool isGaussianOverlapOtherAtom=false;
	  for (unsigned int jAtom=0;jAtom <totalNumberAtoms; jAtom++){
	     if (iAtom !=jAtom){
               Point<C_DIM> jAtomCoor;
               if(jAtom < numberGlobalAtoms)
               {
                 jAtomCoor[0] = atomLocations[jAtom][2];
                 jAtomCoor[1] = atomLocations[jAtom][3];
                 jAtomCoor[2] = atomLocations[jAtom][4];
               }
               else
               {
	         jAtomCoor[0] = imagePositions[jAtom-numberGlobalAtoms][0];
	         jAtomCoor[1] = imagePositions[jAtom-numberGlobalAtoms][1];
	         jAtomCoor[2] = imagePositions[jAtom-numberGlobalAtoms][2];
               }
               const double distanceSq=(nodalCoor-jAtomCoor).norm_square();
	       if (distanceSq < 1e-6){
		   isGaussianOverlapOtherAtom=true;
		   break;
	       }
	     }
	  }
	  if (d_constraintsNoneForce.is_constrained(globalDofIndex)|| (isGaussianOverlapOtherAtom && !allowGaussianOverlapOnAtoms))
	     continue;		  
	  const double rsq= (nodalCoor-atomCoor).norm_square();
	  globalAtomsGaussianForcesLocalPart[C_DIM*atomId+1]+=std::exp(-d_gaussianConstant*rsq)*d_configForceVectorLinFE[globalDofIndex];
      }//y component support points loop

      for(iterMap = d_locallyOwnedSupportPointsForceZ.begin(); iterMap != d_locallyOwnedSupportPointsForceZ.end(); ++iterMap)
      {
	  const int globalDofIndex=iterMap->first;   
	  Point<3> nodalCoor = iterMap->second;
          bool isGaussianOverlapOtherAtom=false;
	  for (unsigned int jAtom=0;jAtom <totalNumberAtoms; jAtom++){
	     if (iAtom !=jAtom){
               Point<C_DIM> jAtomCoor;
               if(jAtom < numberGlobalAtoms)
               {
                 jAtomCoor[0] = atomLocations[jAtom][2];
                 jAtomCoor[1] = atomLocations[jAtom][3];
                 jAtomCoor[2] = atomLocations[jAtom][4];
               }
               else
               {
	         jAtomCoor[0] = imagePositions[jAtom-numberGlobalAtoms][0];
	         jAtomCoor[1] = imagePositions[jAtom-numberGlobalAtoms][1];
	         jAtomCoor[2] = imagePositions[jAtom-numberGlobalAtoms][2];
               }
               const double distanceSq=(nodalCoor-jAtomCoor).norm_square();
	       if (distanceSq < 1e-6){
		   isGaussianOverlapOtherAtom=true;
		   break;
	       }
	     }
	  }
	  if (d_constraintsNoneForce.is_constrained(globalDofIndex)|| (isGaussianOverlapOtherAtom && !allowGaussianOverlapOnAtoms))
	     continue;	  
	  const double rsq= (nodalCoor-atomCoor).norm_square();
	  globalAtomsGaussianForcesLocalPart[C_DIM*atomId+2]+=std::exp(-d_gaussianConstant*rsq)*d_configForceVectorLinFE[globalDofIndex];
	  
      }//z component support points loop      
  }//total atoms loop

  //Sum all processor contributions and distribute to all processors
  MPI_Allreduce(&(globalAtomsGaussianForcesLocalPart[0]),
		&(d_globalAtomsGaussianForces[0]), 
		numberGlobalAtoms*C_DIM,
		MPI_DOUBLE,
		MPI_SUM,
                mpi_communicator);

}

template<unsigned int FEOrder>
void forceClass<FEOrder>::printAtomsForces()
{
  if (this_mpi_process==0){
    const int numberGlobalAtoms = dftPtr->atomLocations.size();	  
    std::cout<< "------------Configurational force on atoms using Gaussian generator with constant: "<< d_gaussianConstant << "-------------"<<std::endl;
    for (unsigned int i=0; i< numberGlobalAtoms; i++)
	std::cout<< "Global atomId: "<< i << ",Force vec: "<< d_globalAtomsGaussianForces[3*i]<<","<< d_globalAtomsGaussianForces[3*i+1]<<","<<d_globalAtomsGaussianForces[3*i+2]<<std::endl;   
    std::cout<< "------------------------------------------------------------------------"<<std::endl;
  }
}
