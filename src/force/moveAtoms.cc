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

template<unsigned int FEOrder>
void forceClass<FEOrder>::gaussianUpdateRhoDataCleanup()
{

  std::map<dealii::CellId, std::vector<double> > *rhoOutValuesCopy=new std::map<dealii::CellId, std::vector<double> >;
  *rhoOutValuesCopy=*(dftPtr->rhoOutValues);
  std::map<dealii::CellId, std::vector<double> > *gradRhoOutValuesCopy;
  if (dftParameters::xc_id==4)
  {
     gradRhoOutValuesCopy = new std::map<dealii::CellId, std::vector<double> >;
    *gradRhoOutValuesCopy=*(dftPtr->gradRhoOutValues);
  }
  //cleanup of existing data
  for (std::deque<std::map<dealii::CellId,std::vector<double> >*>::iterator it = dftPtr->rhoInVals.begin(); it!=dftPtr->rhoInVals.end(); ++it)
  {
     (**it).clear();	  
     delete (*it);
  }
  dftPtr->rhoInVals.clear();
  for (std::deque<std::map<dealii::CellId,std::vector<double> >*>::iterator it = dftPtr->rhoOutVals.begin(); it!=dftPtr->rhoOutVals.end(); ++it)
  {
     (**it).clear();	  
     delete (*it);
  }
  dftPtr->rhoOutVals.clear();
  for (std::deque<std::map<dealii::CellId,std::vector<double> >*>::iterator it = dftPtr->gradRhoInVals.begin(); it!=dftPtr->gradRhoInVals.end(); ++it)
  {
     (**it).clear();	  
     delete (*it);
  }
  dftPtr->gradRhoInVals.clear();
  for (std::deque<std::map<dealii::CellId,std::vector<double> >*>::iterator it = dftPtr->gradRhoOutVals.begin(); it!=dftPtr->gradRhoOutVals.end(); ++it)
  {
     (**it).clear();	  
     delete (*it);
  }
  dftPtr->gradRhoOutVals.clear();

  dftPtr->rhoInValues=new std::map<dealii::CellId, std::vector<double> >;
  *(dftPtr->rhoInValues)=*rhoOutValuesCopy;
  rhoOutValuesCopy->clear();  delete rhoOutValuesCopy;
  dftPtr->rhoInVals.push_back(dftPtr->rhoInValues);
  if (dftParameters::xc_id==4)
  {
    dftPtr->gradRhoInValues = new std::map<dealii::CellId, std::vector<double> >;
    *(dftPtr->gradRhoInValues)=*gradRhoOutValuesCopy;
    gradRhoOutValuesCopy->clear();  delete gradRhoOutValuesCopy;
    dftPtr->gradRhoInVals.push_back(dftPtr->gradRhoInValues);
  }



}

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
 
  double maxDispAtom=-1;
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
	double temp=globalAtomsDisplacements[atomId].norm();
	if (temp>maxDispAtom)
	    maxDispAtom=temp;
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
  const  double maxJacobianRatio=10.0;
  const double break1=1e-2;//try 1e-3
  const double break2=1e-4;//try 1e-5
  //maxDispAtom=0;//HARDCODING TO ONLY USE GAUSSIAN MOVE WITH GAUSSIAN CONSTANT SAME AS FORCE COMPUTATION
  if (maxDispAtom >(break1-tol))
  {
      pcout << "Auto remeshing and reinitialization of dft problem for new atom coordinates as max displacement magnitude: "<<maxDispAtom<< " is greater than: "<< break1 << " Hatree/Bohr..." << std::endl;  
      dftPtr->init(); 
      pcout << "...Reinitialization end" << std::endl;       
  }
  else if (maxDispAtom <(break1+tol) && maxDispAtom>break2)
  {

      const double gaussianParameter=2.0;
      pcout << "Trying to Move using a wide Gaussian with Gaussian constant: "<<gaussianParameter<<" as max displacement magnitude: "<<maxDispAtom<< " is between "<< break2<<" and "<<break1<<" Hatree/Bohr"<<std::endl;
      
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
	  pcout<<"Now reinitializing all moved triangulation dependent objects..." << std::endl;        

	  //reinitialize dirichlet BCs for total potential and vSelf poisson solutions
	  dftPtr->initBoundaryConditions();
	  //reinitialize guesses for electron-density and wavefunctions (not required for relaxation update)
	  //dftPtr->initElectronicFields();
          gaussianUpdateRhoDataCleanup();
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
       pcout << "Trying to Move using a narrow Gaussian with same Gaussian constant for computing the forces: "<<d_gaussianConstant<<" as max displacement magnitude: "<< maxDispAtom<< " is below " << break2 <<" Hatree/Bohr"<<std::endl;       
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
          gaussianUpdateRhoDataCleanup();
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
