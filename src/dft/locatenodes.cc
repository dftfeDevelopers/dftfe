//source file for locating core atom nodes

void dft::locateAtomCoreNodes(){ 
  QGauss<3>  quadrature_formula(quadratureRule);
  FEValues<3> fe_values (FE, quadrature_formula, update_values);
  //
  unsigned int vertices_per_cell=GeometryInfo<3>::vertices_per_cell;
  DoFHandler<3>::active_cell_iterator
    cell = dofHandler.begin_active(),
    endc = dofHandler.end();
  bool located=false;
  unsigned int cellID=0; 
  for (; cell!=endc; ++cell) {
    if (cell->is_locally_owned()){
      fe_values.reinit (cell);
      for (unsigned int i=0; i<vertices_per_cell; ++i){
	Point<3> feNodeGlobalCoord = cell->vertex(i);
	if (sqrt(feNodeGlobalCoord.square())<1.0e-12){  
	  originIDs[0]=cell->vertex_dof_index(i,0);
	  std::cout << "Atom core located at ("<< cell->vertex(i) << ") with node id " << cell->vertex_dof_index(i,0) << " in processor " << this_mpi_process << std::endl;
	  located=true;
	}
      }
    }
    if (located) break;
  }
  //Sync originIDs with all other processors
  if (located){
    for (unsigned int i=0; i<n_mpi_processes; i++){
      if (i!=this_mpi_process) MPI_Bsend(&originIDs[0], numAtomTypes, MPI_UNSIGNED,i, 0, mpi_communicator);
    }
  }
  else{
    MPI_Status status;
    MPI_Recv(&originIDs[0], numAtomTypes, MPI_UNSIGNED, MPI_ANY_SOURCE, 0, mpi_communicator, &status);
  }
  MPI_Barrier(mpi_communicator);
}
