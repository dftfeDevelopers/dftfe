//source file for locating core atom nodes

void dftClass::locateAtomCoreNodes(){ 
  unsigned int vertices_per_cell=GeometryInfo<3>::vertices_per_cell;
  DoFHandler<3>::active_cell_iterator
    cell = dofHandler.begin_active(),
    endc = dofHandler.end();
  //
  IndexSet locally_owned_elements=eigenVectors[0]->locally_owned_elements();
  //locating atom nodes
  unsigned int numAtoms=atomLocations.size();
  std::set<unsigned int> atomsTolocate;
  for (unsigned int i=0; i<numAtoms; i++) atomsTolocate.insert(i);
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
	     std::cout << "Atom core (" << atomLocations[*it][0] << ") located with node id " << nodeID << " in processor " << this_mpi_process;
	     if (locally_owned_elements.is_element(nodeID)){
	       if(isPseudopotential)
		 atoms.insert (std::pair<unsigned int,double>(nodeID,atomLocations[*it][1]));
	       else
		 atoms.insert (std::pair<unsigned int,double>(nodeID,atomLocations[*it][0]));
	       std::cout << " and added \n";
	     }
	     else{
	       std::cout << " but skipped \n"; 
	     }
	     atomsTolocate.erase(*it);
	     break;
	   }
	}
      }
    }
  }
  MPI_Barrier(mpi_communicator);
}
