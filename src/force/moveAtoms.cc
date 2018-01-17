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
void forceClass<FEOrder>::updateAtomPositionsAndMoveMesh(const std::vector<Point<C_DIM> > & globalAtomsDisplacements, double maximumForceOnAtom)
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
  const double tol=1e-6;
  const  double maxJacobianRatio=30.0;
  const double break1=1e-2;//try 1e-3
  const double break2=1e-4;//try 1e-5
  if (maximumForceOnAtom >(break1-tol))
  {
      pcout << "Auto remeshing and reinitialization of dft problem for new atom coordinates as max force is greater than: "<< break1 << " Hatree/Bohr..." << std::endl;  
      dftPtr->init(); 
      pcout << "...Reinitialization end" << std::endl;       
  }
  else if (maximumForceOnAtom <(break1+tol) && maximumForceOnAtom>break2)
  {

      const double gaussianParameter=2.0;
      pcout << "Trying to Move using a wide Gaussian with Gaussian constant: "<<gaussianParameter<<" as max force is between "<< break2<<" and "<<break1<<" Hatree/Bohr"<<std::endl;
      
      std::pair<bool,double> meshQualityMetrics= gaussianMove.moveMesh(controlPointLocations,controlPointDisplacements,gaussianParameter);
      //AssertThrow(!meshQualityMetrics.first,ExcMessage("Negative jacobian created after moving closest nodes to atoms. Suggestion: use auto remeshing"));
      if (meshQualityMetrics.first || meshQualityMetrics.second>maxJacobianRatio)
      {
	  if (meshQualityMetrics.first)
	     pcout<< " Auto remeshing and reinitialization of dft problem for new atom coordinates due to negative jacobian after Gaussian mesh movement using Gaussian constant: "<< gaussianParameter<<std::endl; 	  
	  else
	     pcout<< " Auto remeshing and reinitialization of dft problem for new atom coordinates due to maximum jacobian ratio: "<< meshQualityMetrics.second<< " exceeding set bound of: "<< maxJacobianRatio<<" after Gaussian mesh movement using Gaussian constant: "<< gaussianParameter<<std::endl;
          dftPtr->init(); 
          pcout << "...Reinitialization end" << std::endl;  
      }
      else
      {
	  pcout<< " Mesh quality check: maximum jacobian ratio after movement: "<< meshQualityMetrics.second<<std::endl;  
	  std::cout<<"Now reinitializing all moved triangulation dependent objects..." << std::endl;        

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
	     initPseudoData();
	  }    
	  pcout << "...Reinitialization end" << std::endl;  
      }
  }
  else
  {
       pcout << "Trying to Move using a narrow Gaussian with same Gaussian constant for computing the forces: "<<d_gaussianConstant<<" as max force is below " << break2 <<" Hatree/Bohr"<<std::endl;       
      std::pair<bool,double> meshQualityMetrics=gaussianMove.moveMesh(controlPointLocations,controlPointDisplacements,d_gaussianConstant);
      //AssertThrow(!meshQualityMetrics.first,ExcMessage("Negative jacobian created after moving closest nodes to atoms. Suggestion: use auto remeshing"));
      if (meshQualityMetrics.first || meshQualityMetrics.second>maxJacobianRatio)
      {
	  if (meshQualityMetrics.first)
	     pcout<< " Auto remeshing and reinitialization of dft problem for new atom coordinates due to negative jacobian after Gaussian mesh movement using Gaussian constant: "<< d_gaussianConstant<<std::endl; 	  
	  else
	     pcout<< " Auto remeshing and reinitialization of dft problem for new atom coordinates due to maximum jacobian ratio: "<< meshQualityMetrics.second<< " exceeding set bound of: "<< maxJacobianRatio<<" after Gaussian mesh movement using Gaussian constant: "<< d_gaussianConstant<<std::endl;
          dftPtr->init(); 
          pcout << "...Reinitialization end" << std::endl;  
      }
      else
      {
	  pcout<< " Mesh quality check: maximum jacobian ratio after movement: "<< meshQualityMetrics.second<<std::endl; 
	  pcout << "Now Reinitializing all moved triangulation dependent objects..." << std::endl;  
	 
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
	     initPseudoData();
	  }    
	  pcout << "...Reinitialization end" << std::endl;  
      }
  }
}
