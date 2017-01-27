#include "initRho.cc"
#include "initPseudo.cc"

//
//source file for dft class initializations
//
void dftClass::init(){
  computing_timer.enter_section("dftClass setup");
    
  //
  //initialize FE objects
  //
  dofHandler.distribute_dofs (FE);
  locally_owned_dofs = dofHandler.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dofHandler, locally_relevant_dofs);
  DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), dofHandler, d_supportPoints);
  
  pcout << "number of elements: "
	<< triangulation.n_global_active_cells()
	<< std::endl
	<< "number of degrees of freedom: " 
	<< dofHandler.n_dofs() 
	<< std::endl;

  //
  //write mesh to vtk file
  //
  DataOut<3> data_out;
  data_out.attach_dof_handler (dofHandler);
  data_out.build_patches ();
  std::ofstream output("mesh.vtu");
  data_out.write_vtu(output); 

  //
  //matrix free data structure
  //
  typename MatrixFree<3>::AdditionalData additional_data;
  additional_data.mpi_communicator = MPI_COMM_WORLD;
  additional_data.tasks_parallel_scheme = MatrixFree<3>::AdditionalData::partition_partition;

  //
  //constraints
  //

  //
  //hanging node constraints
  //
  constraintsNone.clear ();
  DoFTools::make_hanging_node_constraints (dofHandler, constraintsNone);

  double halfxyzSpan = 3.8;

#ifdef ENABLE_PERIODIC_BC
  //mark faces
  typename DoFHandler<3>::active_cell_iterator cell = dofHandler.begin_active(), endc = dofHandler.end();
  for (; cell!=endc; ++cell) {
    //if (cell->is_locally_owned()){
      for (unsigned int f=0; f < GeometryInfo<3>::faces_per_cell; ++f){
	const Point<3> face_center = cell->face(f)->center();
	if (cell->face(f)->at_boundary()){
	  if (std::abs(face_center[0]+halfxyzSpan)<1.0e-8)
	    cell->face(f)->set_boundary_id(1);
	  else if (std::abs(face_center[0]-halfxyzSpan)<1.0e-8)
	    cell->face(f)->set_boundary_id(2);
	  else if (std::abs(face_center[1]+halfxyzSpan)<1.0e-8)
	    cell->face(f)->set_boundary_id(3);
	  else if (std::abs(face_center[1]-halfxyzSpan)<1.0e-8)
	    cell->face(f)->set_boundary_id(4);
	  else if (std::abs(face_center[2]+halfxyzSpan)<1.0e-8)
	    cell->face(f)->set_boundary_id(5);
	  else if (std::abs(face_center[2]-halfxyzSpan)<1.0e-8)
	    cell->face(f)->set_boundary_id(6);
	}
      }
      //}
  }

  pcout << "Done with Boundary Flags\n";
  std::vector<GridTools::PeriodicFacePair<typename parallel::distributed::Triangulation<3>::cell_iterator> > periodicity_vector;
  for (int i=0; i<3; ++i){
    GridTools::collect_periodic_faces(triangulation, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vector);
  }
  triangulation.add_periodicity(periodicity_vector);
  pcout << "Periodic Facepairs: " << periodicity_vector.size() << std::endl;

  std::vector<GridTools::PeriodicFacePair<typename DoFHandler<3>::cell_iterator> > periodicity_vector2;
  for (int i=0; i<3; ++i){
    GridTools::collect_periodic_faces(dofHandler, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vector2);
  }
  DoFTools::make_periodicity_constraints<DoFHandler<3> >(periodicity_vector2, constraintsNone);
  pcout << "Detected Periodic Face Pairs: " << constraintsNone.n_constraints() << std::endl;


  pcout<<"Size of ConstraintsNone: "<< constraintsNone.n_constraints()<<std::endl;

  //
  //modify constraintsNone to account for the bug in higher order nodes
  //
  ConstraintMatrix constraintsTemp(constraintsNone); constraintsNone.clear();
  std::set<unsigned int> masterNodes;
  double periodicPrecision=1.0e-8;
  //fill all masters
  for(types::global_dof_index i = 0; i <dofHandler.n_dofs(); ++i){
    if(locally_relevant_dofs.is_element(i)){
      if(constraintsTemp.is_constrained(i)){
	if (constraintsTemp.is_identity_constrained(i)){
	  Point<3> p=d_supportPoints.find(i)->second;
	  unsigned int masterNode=(*constraintsTemp.get_constraint_entries(i))[0].first;
	  masterNodes.insert(masterNode);
	}
      }
    }
  }

  //
  //fix wrong master map
  //
  for(types::global_dof_index i = 0; i <dofHandler.n_dofs(); ++i){
    if(locally_relevant_dofs.is_element(i)){
      if(constraintsTemp.is_constrained(i)){
	if (constraintsTemp.is_identity_constrained(i)){
	  Point<3> p=d_supportPoints.find(i)->second;
	  unsigned int masterNode=(*constraintsTemp.get_constraint_entries(i))[0].first;
	  unsigned int count=0, index=0;
	  for (unsigned int k=0; k<3; k++){
	    if (std::abs(std::abs(p[k])-halfxyzSpan)<periodicPrecision) {
	      count++;
	      index=k;
	    }
	  }
	  if (count==1){
	    Point<3> q=d_supportPoints.find(masterNode)->second;
	    unsigned int l=1, m=2;
	    if (index==1){l=0; m=2;}
	    else if (index==2){l=0; m=1;} 
	    if (!((std::abs(p[l]-q[l])<periodicPrecision) and (std::abs(p[m]-q[m])<periodicPrecision))){
	      bool foundNewMaster=false;
	      for (std::set<unsigned int>::iterator it=masterNodes.begin(); it!=masterNodes.end(); ++it){
		q=d_supportPoints.find(*it)->second;
		if (((std::abs(p[l]-q[l])<periodicPrecision) and (std::abs(p[m]-q[m])<periodicPrecision))){
		  constraintsNone.add_line(i);
		  constraintsNone.add_entry(i, *it, 1.0);
		  foundNewMaster=true;
		  break;
		}
	      }
	      if (!foundNewMaster){
		pcout << "\nError: Didnot find a replacement master node for a wrong master-slave periodic pair\n";
		exit(-1);
	      }
	    }
	    else{
	      constraintsNone.add_line(i);
	      constraintsNone.add_entry(i, masterNode, 1.0);
	    }
	  }
	  else{
	    constraintsNone.add_line(i);
	    constraintsNone.add_entry(i, masterNode, 1.0);
	  }
	}
      }
    }
  }
  constraintsNone.close();
  std::cout<<"Size of ConstraintsNone after fixing periodicity: "<< constraintsNone.n_constraints()<<std::endl;
#endif

  //
  //Zero Dirichlet BC constraints on the boundary of the domain
  //used for computing total electrostatic potential using Poisson problem
  //with (rho+b) as the rhs
  //
  d_constraintsForTotalPotential.clear ();  
  DoFTools::make_hanging_node_constraints (dofHandler, d_constraintsForTotalPotential);
#ifdef ENABLE_PERIODIC_BC
  locatePeriodicPinnedNodes();
#else
  VectorTools::interpolate_boundary_values (dofHandler, 0, ZeroFunction<3>(), d_constraintsForTotalPotential);
#endif
  d_constraintsForTotalPotential.close ();

#ifdef ENABLE_PERIODIC_BC 
  d_constraintsPeriodicWithDirichlet.clear();
  DoFTools::make_hanging_node_constraints (dofHandler, d_constraintsPeriodicWithDirichlet);
  d_constraintsPeriodicWithDirichlet.merge(d_constraintsForTotalPotential);
  d_constraintsPeriodicWithDirichlet.close();
  d_constraintsPeriodicWithDirichlet.merge(constraintsNone);
  d_constraintsPeriodicWithDirichlet.close();  
  std::cout<<"Updated Size of ConstraintsPeriodic with Dirichlet B.Cs: "<< d_constraintsPeriodicWithDirichlet.n_constraints()<<std::endl;
#endif
 
  

  //
  //push back into Constraint Matrices
  //
  d_constraintsVector.clear();
#ifdef ENABLE_PERIODIC_BC
  //d_constraintsVector.push_back(&d_constraintsPeriodicWithDirichlet); 
  d_constraintsVector.push_back(&constraintsNone); 
#else
  d_constraintsVector.push_back(&constraintsNone); 
#endif

  d_constraintsVector.push_back(&d_constraintsForTotalPotential);

  //
  //Dirichlet BC constraints on the boundary of fictitious ball
  //used for computing self-potential (Vself) using Poisson problem
  //with atoms belonging to a given bin
  //
  createAtomBins(d_constraintsVector);
 
 
  //
  //create matrix free structure
  //
  std::vector<const DoFHandler<3> *> dofHandlerVector; 
  
  for(int i = 0; i < d_constraintsVector.size(); ++i)
    dofHandlerVector.push_back(&dofHandler);
 
  std::vector<Quadrature<1> > quadratureVector; 
  quadratureVector.push_back(QGauss<1>(FEOrder+1)); 
  quadratureVector.push_back(QGaussLobatto<1>(FEOrder+1));  


  matrix_free_data.reinit(dofHandlerVector, d_constraintsVector, quadratureVector, additional_data);

  //
  //initialize eigen vectors
  //
  matrix_free_data.initialize_dof_vector(vChebyshev);
  v0Chebyshev.reinit(vChebyshev);
  fChebyshev.reinit(vChebyshev);
  aj[0].reinit(vChebyshev); aj[1].reinit(vChebyshev); aj[2].reinit(vChebyshev);
  aj[3].reinit(vChebyshev); aj[4].reinit(vChebyshev);
  for (unsigned int i=0; i<eigenVectors.size(); ++i){  
    eigenVectors[i]->reinit(vChebyshev);
    PSI[i]->reinit(vChebyshev);
    tempPSI[i]->reinit(vChebyshev);
    tempPSI2[i]->reinit(vChebyshev);
    tempPSI3[i]->reinit(vChebyshev);
    tempPSI4[i]->reinit(vChebyshev);
  } 
  
  //
  //locate atom core nodes and also locate atom nodes in each bin 
  //
  locateAtomCoreNodes();

  
  //
  //initialize density 
  //
  initRho();

  //
  //Initialize libxc (exchange-correlation)
  //
  int exceptParamX, exceptParamC;
  exceptParamX = xc_func_init(&funcX,XC_LDA_X,XC_UNPOLARIZED);
  exceptParamC = xc_func_init(&funcC,XC_LDA_C_PZ,XC_UNPOLARIZED);
  if(exceptParamX != 0 || exceptParamC != 0){
    pcout<<"-------------------------------------"<<std::endl;
    pcout<<"Exchange or Correlation Functional not found"<<std::endl;
    pcout<<"-------------------------------------"<<std::endl;
    exit(-1);
  }

  //
  //initialize local pseudopotential
  //
  if(isPseudopotential)
    {
      initLocalPseudoPotential();
      initNonLocalPseudoPotential();
      computeSparseStructureNonLocalProjectors();
      computeElementalProjectorKets();
    }

 

  //
  //
  //
  computing_timer.exit_section("dftClass setup"); 

  //
  //initialize poisson and eigen problem related objects
  //
  poisson.init();
  eigen.init();
  
  //
  //initialize PSI
  //
  pcout << "reading initial guess for PSI\n";
  readPSI();
}
