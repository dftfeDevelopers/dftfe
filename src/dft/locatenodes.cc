//source file for locating core atom nodes

void dft::locateAtomCoreNodes(){ 
  QGauss<3>  quadrature_formula(quadratureRule);
  FEValues<3> fe_values (FE, quadrature_formula, update_values);
  //
  unsigned int vertices_per_cell=GeometryInfo<3>::vertices_per_cell;
  DoFHandler<3>::active_cell_iterator
    cell = dofHandler.begin_active(),
    endc = dofHandler.end();
  unsigned int numAtoms=atoms.size()[0];
  std::cout << "numAtoms:" << numAtoms << "\n";
  std::vector<bool> located(numAtoms,false);
  unsigned int cellID=0; 
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      fe_values.reinit (cell);
      for (unsigned int i=0; i<vertices_per_cell; ++i){
	Point<3> feNodeGlobalCoord = cell->vertex(i);
	//loop over all atoms to locate the corresponding nodes
	for (unsigned int z=0; z<numAtoms; ++z){
	  if (!located[z]){
	    Point<3> atomCoord(atoms(z,1),atoms(z,2),atoms(z,3));
	    if(feNodeGlobalCoord.distance(atomCoord)<1.0e-5){ 
	      originIDs[z]=cell->vertex_dof_index(i,0);  
	      std::cout << "Atom core (" << atoms(z,0) << ") located at ("<< cell->vertex(i) << ") with node id " << cell->vertex_dof_index(i,0) << " in processor " << this_mpi_process << std::endl;
	      located[z]=true;
	    }
	  }
	}
      }
    }
  }
  //Sync originIDs with all other processors
  for (unsigned int z=0; z<located.size(); z++){
    if (located[z]){
      for (unsigned int i=0; i<n_mpi_processes; i++){
	if (i!=this_mpi_process) {
	  MPI_Bsend(&originIDs[z], 1, MPI_UNSIGNED, i, z, mpi_communicator);
	}
      }
    }
    else{
      MPI_Status status;
      MPI_Recv(&originIDs[z], 1, MPI_UNSIGNED, MPI_ANY_SOURCE, z, mpi_communicator, &status);
    }
  }
  MPI_Barrier(mpi_communicator);
}
