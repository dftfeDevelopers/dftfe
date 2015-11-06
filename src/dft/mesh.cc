//source file for all mesh reading/generation functions

//Generate triangulation.
void dftClass::mesh(){
  computing_timer.enter_section("mesh"); 


  //define mesh parameters
  double L=20;
  double meshBias=0.7;
  double numMeshSegments=4;
  //
  GridGenerator::hyper_cube (triangulation, -L, L);
  //compute h by geometric progression
  double sum=1.0;
  for (unsigned int i=1; i<=numMeshSegments; i++){
    sum+=std::pow(1.0/meshBias, i);
  }
  double l=L/sum;
  pcout << "Mesh parameters L:" << L << ", numberSegments:" << numMeshSegments << ", l:" << l << "\n";
  //triangulation.refine_global (n_refinement_steps);  
  bool refineFlag=true;
  unsigned int numLevels=0;
  while(refineFlag){
    refineFlag=false;
    typename Triangulation<3>::active_cell_iterator cell = triangulation.begin_active(),
      end_cell = triangulation.end();
    for ( ; cell != end_cell; ++cell){
      if (cell->is_locally_owned()){
	double li=l, h=l;
	for (unsigned int i=1; i<=numMeshSegments; i++){
	  if (cell->center().norm()<li) break;
	  h=l*std::pow(1.0/meshBias,i); 
	  li+=h; 
	}
	if (cell->minimum_vertex_distance()>h) {
	  cell->set_refine_flag();
	  refineFlag=true;
	}
      }
    }
    if (refineFlag){
      triangulation.execute_coarsening_and_refinement();
      numLevels++;
    }
  }

  pcout << "Refinement levels executed: " << numLevels << std::endl;
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
