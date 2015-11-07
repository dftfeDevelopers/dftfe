//source file for all mesh reading/generation functions

//Generate triangulation.
void dftClass::mesh(){
  computing_timer.enter_section("mesh"); 

  //define mesh parameters
  double L=20;
  double meshBias=0.7;
  double numMeshSegments=10;
  //
  GridGenerator::hyper_cube (triangulation, -L, L);
  triangulation.refine_global(1);
  //compute h by geometric progression
  double sum=1.0;
  for (unsigned int i=1; i<numMeshSegments; i++){
    sum+=std::pow(1.0/meshBias, i);
  }
  double l=L/sum;
  pcout << "Mesh parameters L:" << L << ", numberSegments:" << numMeshSegments << ", l:" << l << "\n";
  std::vector<double> liVector(1,0.0);
  double li=0.0;
  for (unsigned int i=0; i<numMeshSegments; i++){
    li+=l*std::pow(1.0/meshBias, i);
    liVector.push_back(li);
    //pcout<< "(" << l*std::pow(1.0/meshBias, i) << ", " << li << ") ";
  } 
  
  //refine mesh
  bool refineFlag=true;
  unsigned int numLevels=0;
  while(refineFlag){
    refineFlag=false;
    typename Triangulation<3>::active_cell_iterator cell = triangulation.begin_active(),
      end_cell = triangulation.end();
    for ( ; cell != end_cell; ++cell){
      if (cell->is_locally_owned()){
	dealii::Point<3> center(cell->center());  
	for (unsigned int i=0; i<liVector.size()-1; i++){
	  if ((center.norm()>liVector[i]) && (center.norm()<=liVector[i+1])){
	    double h=liVector[i+1]-liVector[i];
	    //pcout << "(" << center.norm() << ", "<< h << ", " << cell->minimum_vertex_distance() << ")" << std::endl;
	    if ((cell->minimum_vertex_distance()/2)>=h){
	      cell->set_refine_flag();
	      refineFlag=true; 
	    }
	    break;
	  }
	}
      }
    }
    if (refineFlag){
      triangulation.execute_coarsening_and_refinement();
      numLevels++;
    }
  }

  double minElemLength=L;
  typename Triangulation<3>::active_cell_iterator cell = triangulation.begin_active(),
    end_cell = triangulation.end();
  for ( ; cell != end_cell; ++cell){
    if (cell->is_locally_owned()){
      if (cell->minimum_vertex_distance()<minElemLength) minElemLength = cell->minimum_vertex_distance();
    }
  }
  //Utilities::MPI::sum(minElemLength, mpi_communicator);
  //
  pcout << "Refinement levels executed: " << numLevels << std::endl;
  char buffer[100];
  sprintf(buffer, "Smallest element size: %5.2e\n", minElemLength);
  pcout << buffer;
  pcout << "   Number of global active cells: "
	<< triangulation.n_global_active_cells()
	<< std::endl;
  
  //dof_handle
  dofHandler.distribute_dofs (FE);
  locally_owned_dofs = dofHandler.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dofHandler, locally_relevant_dofs);
  pcout << "number of elements: "
	<< triangulation.n_global_active_cells()
	<< std::endl
	<< "number of degrees of freedom: " 
	<< dofHandler.n_dofs() 
	<< std::endl;

  DataOut<3> data_out;
  data_out.attach_dof_handler (dofHandler);
  //data_out.add_data_vector (solution, "solution");
  data_out.build_patches ();
  std::ofstream output("mesh.vtu");
  data_out.write_vtu(output); 
  //
  computing_timer.exit_section("mesh"); 
}
