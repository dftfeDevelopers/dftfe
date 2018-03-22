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
// @author Shiva Rudraraju (2016), Phani Motamarri (2016)
//

//source file for locating core atom nodes
template<unsigned int FEOrder>
void dftClass<FEOrder>::locateAtomCoreNodes(){
  atoms.clear();
  d_atomsInBin.clear();
  unsigned int vertices_per_cell=GeometryInfo<3>::vertices_per_cell;

  bool isPseudopotential = dftParameters::isPseudopotential;

  DoFHandler<3>::active_cell_iterator
    cell = dofHandler.begin_active(),
    endc = dofHandler.end();
  //
  //IndexSet locally_owned_elements=eigenVectors[0][0]->locally_owned_elements();
  //locating atom nodes
  unsigned int numAtoms=atomLocations.size();
  std::set<unsigned int> atomsTolocate;
  for (unsigned int i = 0; i < numAtoms; i++) atomsTolocate.insert(i);
  //element loop
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      for (unsigned int i=0; i<vertices_per_cell; ++i){
	unsigned int nodeID=cell->vertex_dof_index(i,0);
	Point<3> feNodeGlobalCoord = cell->vertex(i);
	//
	//loop over all atoms to locate the corresponding nodes
	//
	for (std::set<unsigned int>::iterator it=atomsTolocate.begin(); it!=atomsTolocate.end(); ++it){
	  Point<3> atomCoord(atomLocations[*it][2],atomLocations[*it][3],atomLocations[*it][4]);
	   if(feNodeGlobalCoord.distance(atomCoord) < 1.0e-5){ 
	     if(isPseudopotential)
	       {
		 if (dftParameters::verbosity==1)
                 {
		   std::cout << "atom core with valence charge " << atomLocations[*it][1] << " located with node id " << nodeID << " in processor " << this_mpi_process<<" nodal coor "<<feNodeGlobalCoord[0]<<" "<<feNodeGlobalCoord[1]<<" "<<feNodeGlobalCoord[2]<<std::endl;
		 }
	       }
	     else
	       {
		 if (dftParameters::verbosity==1)
                 {		   
		    std::cout << "atom core with charge " << atomLocations[*it][0] << " located with node id " << nodeID << " in processor " << this_mpi_process<<" nodal coor "<<feNodeGlobalCoord[0]<<" "<<feNodeGlobalCoord[1]<<" "<<feNodeGlobalCoord[2]<<std::endl;
		 }
	       }
	     if (locally_owned_dofs.is_element(nodeID)){
	       if(isPseudopotential)
		 atoms.insert(std::pair<unsigned int,double>(nodeID,atomLocations[*it][1]));
	       else
		 atoms.insert(std::pair<unsigned int,double>(nodeID,atomLocations[*it][0]));
		 
	       if (dftParameters::verbosity==1)
	          std::cout << " and added \n";
	     }
	     else{
	       if (dftParameters::verbosity==1)		 
	         std::cout << " but skipped \n"; 
	     }
	     atomsTolocate.erase(*it);
	     break;
	   }//tolerance check if loop
	}//atomsTolocate loop
      }//vertices_per_cell loop
    }//locally owned cell if loop
  }//cell loop
  MPI_Barrier(mpi_communicator);

  const unsigned int totalAtomNodesFound = Utilities::MPI::sum(atoms.size(), mpi_communicator);
  AssertThrow(totalAtomNodesFound==numAtoms,ExcMessage("Atleast one atom doesn't lie on a triangulation vertex"));

  int numberBins = d_boundaryFlag.size();
  d_atomsInBin.resize(numberBins);


  for(int iBin = 0; iBin < numberBins; ++iBin)
    {
      unsigned int vertices_per_cell=GeometryInfo<3>::vertices_per_cell;
      DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(),endc = dofHandler.end();

      std::set<int> & atomsInBinSet = d_bins[iBin];
      std::vector<int> atomsInCurrentBin(atomsInBinSet.begin(),atomsInBinSet.end());
      unsigned int numberGlobalAtomsInBin = atomsInCurrentBin.size();
      std::set<unsigned int> atomsTolocate;
      for (unsigned int i = 0; i < numberGlobalAtomsInBin; i++) atomsTolocate.insert(i);

      for (; cell!=endc; ++cell) {
	if (cell->is_locally_owned()){
	  for (unsigned int i=0; i<vertices_per_cell; ++i){
	    unsigned int nodeID=cell->vertex_dof_index(i,0);
	    Point<3> feNodeGlobalCoord = cell->vertex(i);
	    //
	    //loop over all atoms to locate the corresponding nodes
	    //
	    for (std::set<unsigned int>::iterator it=atomsTolocate.begin(); it!=atomsTolocate.end(); ++it)
	      {
		int chargeId = atomsInCurrentBin[*it];
		Point<3> atomCoord(atomLocations[chargeId][2],atomLocations[chargeId][3],atomLocations[chargeId][4]);
		if(feNodeGlobalCoord.distance(atomCoord) < 1.0e-5){ 
		  if(isPseudopotential)
		  {
		    if (dftParameters::verbosity==1)
		      std::cout << "atom core in bin " << iBin<<" with valence charge "<<atomLocations[chargeId][1] << " located with node id " << nodeID << " in processor " << this_mpi_process;
		  }
		  else
		  {
		    if (dftParameters::verbosity==1)  
		      std::cout << "atom core in bin " << iBin<<" with charge "<<atomLocations[chargeId][0] << " located with node id " << nodeID << " in processor " << this_mpi_process;
		  }
		  if (locally_owned_dofs.is_element(nodeID)){
		    if(isPseudopotential)
		      d_atomsInBin[iBin].insert(std::pair<unsigned int,double>(nodeID,atomLocations[chargeId][1]));
		    else
		      d_atomsInBin[iBin].insert(std::pair<unsigned int,double>(nodeID,atomLocations[chargeId][0]));
		    if (dftParameters::verbosity==1)
		       std::cout << " and added \n";
		  }
		  else
		  {
            	    if (dftParameters::verbosity==1)
		       std::cout << " but skipped \n"; 
		  }
		  atomsTolocate.erase(*it);
		  break;
		}//tolerance check if loop
	      }//atomsTolocate loop
	  }//vertices_per_cell loop
	}//locally owned cell if loop
      }//cell loop
      MPI_Barrier(mpi_communicator);
    }//iBin loop

}

template<unsigned int FEOrder>
void dftClass<FEOrder>::locatePeriodicPinnedNodes()
{ 

  const int numberImageCharges = d_imageIds.size();
  const int numberGlobalAtoms = atomLocations.size();
  const int totalNumberAtoms = numberGlobalAtoms + numberImageCharges;


  //
  //find vertex furthest from all nuclear charges
  //
  double maxDistance = -1.0;
  unsigned int maxNode,minNode;

  std::map<types::global_dof_index,Point<3> >::iterator iterMap;
  for(iterMap = d_supportPoints.begin(); iterMap != d_supportPoints.end(); ++iterMap)
    {
      if(locally_owned_dofs.is_element(iterMap->first))
	{
	  if(!d_noConstraints.is_constrained(iterMap->first))
	    {
	      double minDistance = 1e10;
	      minNode = -1;
	      Point<3> nodalPointCoordinates = iterMap->second;
	      for(unsigned int iAtom = 0; iAtom < totalNumberAtoms; ++iAtom)
		{
		  Point<3> atomCoor;

		  if(iAtom < numberGlobalAtoms)
		    {
		      atomCoor[0] = atomLocations[iAtom][2];
		      atomCoor[1] = atomLocations[iAtom][3];
		      atomCoor[2] = atomLocations[iAtom][4];
		    }
		  else
		    {
		      //
		      //Fill with ImageAtom Coors
		      //
		      atomCoor[0] = d_imagePositions[iAtom-numberGlobalAtoms][0];
		      atomCoor[1] = d_imagePositions[iAtom-numberGlobalAtoms][1];
		      atomCoor[2] = d_imagePositions[iAtom-numberGlobalAtoms][2];
		    }

		  double distance = atomCoor.distance(nodalPointCoordinates);
	      
		  if(distance <= minDistance)
		    {
		      minDistance = distance;
		      minNode = iterMap->first;
		    }

		}

	      if(minDistance > maxDistance)
		{
		  maxDistance = minDistance;
		  maxNode = iterMap->first;
		}
	    }
	}
    }

  double globalMaxDistance;

  MPI_Allreduce(&maxDistance,
		&globalMaxDistance,
		1,
		MPI_DOUBLE,
		MPI_MAX,
		mpi_communicator);

  

  //locating pinned nodes
  std::vector<std::vector<double> > pinnedLocations;
  std::vector<double> temp(3,0.0);
  std::vector<double> tempLocal(3,0.0);
  unsigned int taskId = 0;

  if(std::abs(maxDistance - globalMaxDistance) < 1e-07)
    taskId = Utilities::MPI::this_mpi_process(mpi_communicator);

  unsigned int maxTaskId;

  MPI_Allreduce(&taskId,
		&maxTaskId,
		1,
		MPI_INT,
		MPI_MAX,
		mpi_communicator);

  if(Utilities::MPI::this_mpi_process(mpi_communicator) == maxTaskId)
    {
      if (dftParameters::verbosity==1)
          std::cout<<"Found Node locally on processor Id: "<<Utilities::MPI::this_mpi_process(mpi_communicator)<<std::endl;
      if(locally_owned_dofs.is_element(maxNode))
	{
	  if(constraintsNone.is_identity_constrained(maxNode))
	    {
	      unsigned int masterNode = (*constraintsNone.get_constraint_entries(maxNode))[0].first;
	      Point<3> nodalPointCoordinates = d_supportPoints.find(masterNode)->second;
	      tempLocal[0] = nodalPointCoordinates[0];
	      tempLocal[1] = nodalPointCoordinates[1];
	      tempLocal[2] = nodalPointCoordinates[2];
	    }
	  else
	    {
	      Point<3> nodalPointCoordinates = d_supportPoints.find(maxNode)->second;
	      tempLocal[0] = nodalPointCoordinates[0];
	      tempLocal[1] = nodalPointCoordinates[1];
	      tempLocal[2] = nodalPointCoordinates[2];
	    }
	  //checkFlag = 1;
	}
    }
  

  MPI_Allreduce(&tempLocal[0],
		&temp[0],
		3,
		MPI_DOUBLE,
		MPI_SUM,
		mpi_communicator);
		
  pinnedLocations.push_back(temp);
  

  const unsigned int dofs_per_cell = FE.dofs_per_cell;
  DoFHandler<3>::active_cell_iterator
    cell = dofHandler.begin_active(),
    endc = dofHandler.end();
  
  unsigned int numberNodes = pinnedLocations.size();
  std::set<unsigned int> nodesTolocate;
  for (unsigned int i = 0; i < numberNodes; i++) nodesTolocate.insert(i);

  for (; cell!=endc; ++cell) 
    {
      if (cell->is_locally_owned())
	{
	  std::vector<types::global_dof_index> cell_dof_indices(dofs_per_cell);
	  cell->get_dof_indices(cell_dof_indices);

	  for (unsigned int i = 0; i < dofs_per_cell; ++i)
	    {

	      unsigned int nodeID = cell_dof_indices[i];
	      Point<3> feNodeGlobalCoord = d_supportPoints[cell_dof_indices[i]];

	      //
	      //loop over all atoms to locate the corresponding nodes
	      //
	      for (std::set<unsigned int>::iterator it=nodesTolocate.begin(); it!=nodesTolocate.end(); ++it)
		{

		  Point<3> pinnedNodeCoord(pinnedLocations[*it][0],pinnedLocations[*it][1],pinnedLocations[*it][2]);
		  if(feNodeGlobalCoord.distance(pinnedNodeCoord) < 1.0e-5)
		    {
	              if (dftParameters::verbosity==1)			
		         std::cout << "Pinned core with nodal coordinates (" << pinnedLocations[*it][0] << " " << pinnedLocations[*it][1] << " "<<pinnedLocations[*it][2]<< ") located with node id " << nodeID << " in processor " << this_mpi_process;
		      if (locally_relevant_dofs.is_element(nodeID))
			{
			  d_constraintsForTotalPotential.add_line(nodeID);
			  d_constraintsForTotalPotential.set_inhomogeneity(nodeID,0.0);
			  if (dftParameters::verbosity==1)
			     std::cout << " and added \n";
			}
		      else
			{
			  if (dftParameters::verbosity==1)  
			     std::cout << " but skipped \n"; 
			}
		      nodesTolocate.erase(*it);
		      break;
		    }//tolerance check if loop

		}//atomsTolocate loop

	    }//vertices_per_cell loop

	}//locally owned cell if loop

    }//cell loop

  MPI_Barrier(mpi_communicator);
}

