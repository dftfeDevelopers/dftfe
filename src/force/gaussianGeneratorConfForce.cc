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
void forceClass<FEOrder>::computeAtomsForcesGaussianGenerator()
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

  /*
  for(iterMap = locallyOwnedSupportPointsForceX.begin(); iterMap != locallyOwnedSupportPointsForceX.end(); ++iterMap)
  {
     const int globalDofIndex=iterMap->first;    
     Point<3> nodalCoor = iterMap->second;
     std::map<types::global_dof_index,Point<C_DIM> >::iterator iterMap2;

     if (nodalCoor[0]>1e-5 || std::fabs(configForceVectorLinFE[globalDofIndex])<1e-10)
     //if (nodalCoor[0]>1e-5)	     
	  continue;

     for(iterMap2 = locallyOwnedSupportPointsForceX.begin(); iterMap2 != locallyOwnedSupportPointsForceX.end(); ++iterMap2)
     {
        const int globalDofIndex2=iterMap2->first;    
        Point<3> nodalCoor2 = iterMap2->second;	    
	if (nodalCoor2[0]<1e-5)
	  continue;
        Point<3> nodalCoor2Flip=nodalCoor2; nodalCoor2Flip[0]=-nodalCoor2Flip[0];
	const double distance= nodalCoor.distance(nodalCoor2Flip);
	if (distance < 1e-5 && std::fabs(configForceVectorLinFE[globalDofIndex]+configForceVectorLinFE[globalDofIndex2])>1e-10){
		std::cout<< nodalCoor << " value: "<<configForceVectorLinFE[globalDofIndex] << " ,  "<< nodalCoor2 << " value: "<< configForceVectorLinFE[globalDofIndex2] <<std::endl;
          break;
	}
     }
  }
  */

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
	  const double rsq= (nodalCoor-atomCoor).norm_square();
	  if(!d_constraintsNoneForce.is_constrained(globalDofIndex))
	    globalAtomsGaussianForcesLocalPart[C_DIM*atomId]+=std::exp(-d_gaussianConstant*rsq)*d_configForceVectorLinFE[globalDofIndex];

	  //if (std::fabs(d_configForceVectorLinFE[globalDofIndex])>1e+2)
  	  //	  std::cout<<"globalIndex: " << globalDofIndex << ",nodalCoor: "<< nodalCoor << ",value: "<<std::fabs(d_configForceVectorLinFE[globalDofIndex])<<std::endl;
      }//x component support points loop

      for(iterMap = d_locallyOwnedSupportPointsForceY.begin(); iterMap != d_locallyOwnedSupportPointsForceY.end(); ++iterMap)
      {
	  const int globalDofIndex=iterMap->first;    
	  Point<3> nodalCoor = iterMap->second;
	  const double rsq= (nodalCoor-atomCoor).norm_square();
	  if(!d_constraintsNoneForce.is_constrained(globalDofIndex))	  
	    globalAtomsGaussianForcesLocalPart[C_DIM*atomId+1]+=std::exp(-d_gaussianConstant*rsq)*d_configForceVectorLinFE[globalDofIndex];
	  //if (std::fabs(d_configForceVectorLinFE[globalDofIndex])>1e+2)
	  //	  std::cout<<"globalIndex: " << globalDofIndex << ",nodalCoor: "<< nodalCoor << ",value: "<<std::fabs(d_configForceVectorLinFE[globalDofIndex])<<std::endl;;	  
      }//y component support points loop

      for(iterMap = d_locallyOwnedSupportPointsForceZ.begin(); iterMap != d_locallyOwnedSupportPointsForceZ.end(); ++iterMap)
      {
	  const int globalDofIndex=iterMap->first;    
	  Point<3> nodalCoor = iterMap->second;
	  const double rsq= (nodalCoor-atomCoor).norm_square();
	  if(!d_constraintsNoneForce.is_constrained(globalDofIndex))	  
	    globalAtomsGaussianForcesLocalPart[C_DIM*atomId+2]+=std::exp(-d_gaussianConstant*rsq)*d_configForceVectorLinFE[globalDofIndex];
	  //if (std::fabs(d_configForceVectorLinFE[globalDofIndex])>1e+2)
	  //	  std::cout<<"globalIndex: " << globalDofIndex << ",nodalCoor: "<< nodalCoor << ",value: "<<std::fabs(d_configForceVectorLinFE[globalDofIndex])<<std::endl;	  
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

/*
template<unsigned int FEOrder>
void forceClass<FEOrder>::moveAtomsMeshGaussianGenerator(const std::vector<double> & globalAtomsDisplacements, bool isForceRelaxation)
{

  std::vector<std::vector<double> > & atomLocations=dftPtr->atomLocations;
  std::vector<std::vector<double> > & imagePositions=dftPtr->d_imagePositions;
  const std::vector<int > & imageIds=dftPtr->d_imageIds;
  const int numberGlobalAtoms = atomLocations.size();
  const int numberImageCharges = imageIds.size();
  const int totalNumberAtoms = numberGlobalAtoms + numberImageCharges; 
  double dispTol=1e-8;

  for (unsigned int iAtom=0;iAtom <totalNumberAtoms; iAtom++){
     Point<C_DIM> atomCoor;
     int atomId=iAtom;
     if(iAtom < numberGlobalAtoms)
     {
        atomCoor[0] = atomLocations[iAtom][2];
        atomCoor[1] = atomLocations[iAtom][3];
        atomCoor[2] = atomLocations[iAtom][4];
        atomLocations[iAtom][2]+=globalAtomsDisplacements[C_DIM*atomId+0];
        atomLocations[iAtom][3]+=globalAtomsDisplacements[C_DIM*atomId+1];
        atomLocations[iAtom][4]+=globalAtomsDisplacements[C_DIM*atomId+2];	
     }
     else
     {
	atomCoor[0] = imagePositions[iAtom-numberGlobalAtoms][0];
	atomCoor[1] = imagePositions[iAtom-numberGlobalAtoms][1];
	atomCoor[2] = imagePositions[iAtom-numberGlobalAtoms][2];
	atomId=imageIds[iAtom-numberGlobalAtoms];
	imagePositions[iAtom-numberGlobalAtoms][0]+=globalAtomsDisplacements[C_DIM*atomId+0];
	imagePositions[iAtom-numberGlobalAtoms][1]+=globalAtomsDisplacements[C_DIM*atomId+1];
	imagePositions[iAtom-numberGlobalAtoms][2]+=globalAtomsDisplacements[C_DIM*atomId+2];
	
     }	

     if ((std::fabs(globalAtomsDisplacements[C_DIM*atomId+0])
         +std::fabs(globalAtomsDisplacements[C_DIM*atomId+1])
	 +std::fabs(globalAtomsDisplacements[C_DIM*atomId+2]))<dispTol)
	 continue;

     Triangulation<C_DIM>::active_vertex_iterator
     vertexIter = dftPtr->triangulation.begin_active_vertex(),
     endVertexIter = dftPtr->triangulation.end_vertex();
     for (; vertexIter!=endVertexIter; ++vertexIter)
     {
          Point<C_DIM> &nodalCoor = vertexIter->vertex(0);
	  const double rsq= (nodalCoor-atomCoor).norm_square();
	  for (unsigned int idim=0; idim<C_DIM; ++idim)
	    nodalCoor[idim]+=std::exp(-d_gaussianConstant*rsq)*globalAtomsDisplacements[C_DIM*atomId+idim];	    
     }
  }
 
   
  dftPtr->dofHandler.clear();
  dftPtr->dofHandlerEigen.clear();
  d_dofHandlerForce.clear();
  dftPtr->dofHandler.initialize(dftPtr->triangulation,dftPtr->FE);
  dftPtr->dofHandlerEigen.initialize(dftPtr->triangulation,dftPtr->FEEigen);
  d_dofHandlerForce.initialize(dftPtr->triangulation,FEForce);  
   

  if (isForceRelaxation)
    dftPtr->reinitialize();  
  
}
*/
//
