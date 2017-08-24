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
  unsigned int vertices_per_cell=GeometryInfo<3>::vertices_per_cell;
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
	//loop over all atoms to locate the corresponding nodes
	for (std::set<unsigned int>::iterator it=atomsTolocate.begin(); it!=atomsTolocate.end(); ++it){
	  Point<3> atomCoord(atomLocations[*it][2],atomLocations[*it][3],atomLocations[*it][4]);
	   if(feNodeGlobalCoord.distance(atomCoord) < 1.0e-5){ 
	     if(isPseudopotential)
	       {
		 std::cout << "atom core with valence charge " << atomLocations[*it][1] << " located with node id " << nodeID << " in processor " << this_mpi_process<<" nodal coor "<<feNodeGlobalCoord[0]<<" "<<feNodeGlobalCoord[1]<<" "<<feNodeGlobalCoord[2]<<std::endl;
	       }
	     else
	       {
		 std::cout << "atom core with charge " << atomLocations[*it][0] << " located with node id " << nodeID << " in processor " << this_mpi_process<<" nodal coor "<<feNodeGlobalCoord[0]<<" "<<feNodeGlobalCoord[1]<<" "<<feNodeGlobalCoord[2]<<std::endl;
	       }
	     if (locally_owned_dofs.is_element(nodeID)){
	       if(isPseudopotential)
		 atoms.insert(std::pair<unsigned int,double>(nodeID,atomLocations[*it][1]));
	       else
		 atoms.insert(std::pair<unsigned int,double>(nodeID,atomLocations[*it][0]));
	       std::cout << " and added \n";
	     }
	     else{
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
	    //loop over all atoms to locate the corresponding nodes
	    for (std::set<unsigned int>::iterator it=atomsTolocate.begin(); it!=atomsTolocate.end(); ++it)
	      {
		int chargeId = atomsInCurrentBin[*it];
		Point<3> atomCoord(atomLocations[chargeId][2],atomLocations[chargeId][3],atomLocations[chargeId][4]);
		if(feNodeGlobalCoord.distance(atomCoord) < 1.0e-5){ 
		  if(isPseudopotential)
		    std::cout << "atom core in bin " << iBin<<" with valence charge "<<atomLocations[chargeId][1] << " located with node id " << nodeID << " in processor " << this_mpi_process;
		  else
		    std::cout << "atom core in bin " << iBin<<" with charge "<<atomLocations[chargeId][0] << " located with node id " << nodeID << " in processor " << this_mpi_process;
		  if (locally_owned_dofs.is_element(nodeID)){
		    if(isPseudopotential)
		      d_atomsInBin[iBin].insert(std::pair<unsigned int,double>(nodeID,atomLocations[chargeId][1]));
		    else
		      d_atomsInBin[iBin].insert(std::pair<unsigned int,double>(nodeID,atomLocations[chargeId][0]));
		    std::cout << " and added \n";
		  }
		  else{
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
void dftClass<FEOrder>::locatePeriodicPinnedNodes(){ 
  unsigned int vertices_per_cell=GeometryInfo<3>::vertices_per_cell;
  DoFHandler<3>::active_cell_iterator
    cell = dofHandler.begin_active(),
    endc = dofHandler.end();

  //locating pinned nodes
  std::vector<std::vector<double> > pinnedLocations;
  std::vector<double> temp(3,0.0); 
  std::vector<double> tempLocal(3,0.0);
  //temp.push_back(3.8); temp.push_back(3.8); temp.push_back(3.8);//(center)
  //temp.push_back(0.0); temp.push_back(0.0); temp.push_back(0.0);//(corner)
  //temp.push_back(2.28); temp.push_back(0.0); temp.push_back(3.8);//(bcc)
  //temp.push_back(3.8628);temp.push_back(3.8628);temp.push_back(3.8628);
  
  if(this_mpi_process==0)
    { 
      for(; cell!=endc; ++cell)
	{
	  if (cell->is_locally_owned())
	    {
	      if (! cell->has_boundary_lines())
		{
		  Point<3> pinPoint = cell->vertex(0);
		  tempLocal[0] = pinPoint[0]; 
		  tempLocal[1] = pinPoint[1]; 
		  tempLocal[2] = pinPoint[2];
		  break;
		}
	    }
	}
    }

  MPI_Allreduce(&tempLocal[0],
		&temp[0],
		3,
		MPI_DOUBLE,
		MPI_SUM,
		mpi_communicator);
		


  pinnedLocations.push_back(temp);
  cell = dofHandler.begin_active();
  
  unsigned int numberNodes = pinnedLocations.size();
  std::set<unsigned int> nodesTolocate;
  for (unsigned int i = 0; i < numberNodes; i++) nodesTolocate.insert(i);

  //element loop
  //if(temp.size() > 0)
  //{
      for (; cell!=endc; ++cell) 
	{
	  if (cell->is_locally_owned())
	    {
	      for (unsigned int i=0; i<vertices_per_cell; ++i)
		{
		  unsigned int nodeID=cell->vertex_dof_index(i,0);
		  Point<3> feNodeGlobalCoord = cell->vertex(i);
		  //loop over all atoms to locate the corresponding nodes
		  for (std::set<unsigned int>::iterator it=nodesTolocate.begin(); it!=nodesTolocate.end(); ++it)
		    {
		      Point<3> pinnedNodeCoord(pinnedLocations[*it][0],pinnedLocations[*it][1],pinnedLocations[*it][2]);
		      if(feNodeGlobalCoord.distance(pinnedNodeCoord) < 1.0e-5)
			{ 
			  std::cout << "Pinned core with nodal coordinates (" << pinnedLocations[*it][0] << " " << pinnedLocations[*it][1] << " "<<pinnedLocations[*it][2]<< ") located with node id " << nodeID << " in processor " << this_mpi_process;
			  if (locally_relevant_dofs.is_element(nodeID))
			    {
			      d_constraintsForTotalPotential.add_line(nodeID);
			      d_constraintsForTotalPotential.set_inhomogeneity(nodeID,0.0);
			      std::cout << " and added \n";
			    }
			  else
			    {
			      std::cout << " but skipped \n"; 
			    }
			  nodesTolocate.erase(*it);
			  break;
			}//tolerance check if loop
		    }//atomsTolocate loop
		}//vertices_per_cell loop
	    }//locally owned cell if loop
	}//cell loop
      //}
  MPI_Barrier(mpi_communicator);
}
