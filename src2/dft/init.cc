#include "initRho.cc"
#include "initPseudo.cc"

#ifdef ENABLE_PERIODIC_BC
#include "initkPointData.cc"
#endif

//
//source file for dft class initializations
//
void exchangeMasterNodesList(std::set<unsigned int> & masterNodeIdSet,
			     std::set<unsigned int> & globalMasterNodeIdSet,
			     unsigned int numMeshPartitions,
			     const MPI_Comm & mpi_communicator)
{

  std::vector<int> localMasterNodeIdList;
  std::copy(masterNodeIdSet.begin(),
	    masterNodeIdSet.end(),
	    std::back_inserter(localMasterNodeIdList));

  int numberMasterNodesOnLocalProc = localMasterNodeIdList.size();

  int * masterNodeIdListSizes = new int[numMeshPartitions];

  MPI_Allgather(&numberMasterNodesOnLocalProc,
		1,
		MPI_INT,
		masterNodeIdListSizes,
		1,
		MPI_INT,
		mpi_communicator);

  int newMasterNodeIdListSize = std::accumulate(&(masterNodeIdListSizes[0]),
						&(masterNodeIdListSizes[numMeshPartitions]),
						0);

  std::vector<int> globalMasterNodeIdList(newMasterNodeIdListSize);

  int * mpiOffsets = new int[numMeshPartitions];

  mpiOffsets[0] = 0;

  for(int i = 1; i < numMeshPartitions; ++i)
    mpiOffsets[i] = masterNodeIdListSizes[i-1] + mpiOffsets[i-1];

  MPI_Allgatherv(&(localMasterNodeIdList[0]),
		 numberMasterNodesOnLocalProc,
		 MPI_INT,
		 &(globalMasterNodeIdList[0]),
		 &(masterNodeIdListSizes[0]),
		 &(mpiOffsets[0]),
		 MPI_INT,
		 mpi_communicator);


  for(int i = 0; i < globalMasterNodeIdList.size(); ++i)
    globalMasterNodeIdSet.insert(globalMasterNodeIdList[i]);

  delete [] masterNodeIdListSizes;
  delete [] mpiOffsets;
	    
  return;

}



void dftClass::init(){
  computing_timer.enter_section("dftClass setup");


#ifdef ENABLE_PERIODIC_BC
  //mark faces
  double halfxyzSpan = 3.8;
  typename parallel::distributed::Triangulation<3>::active_cell_iterator cell = triangulation.begin_active(), endc = triangulation.end();
  for (; cell!=endc; ++cell) 
    {
      for (unsigned int f=0; f < GeometryInfo<3>::faces_per_cell; ++f)
	{
	  const Point<3> face_center = cell->face(f)->center();
	  if (cell->face(f)->at_boundary())
	    {
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
    }

  //mark faces
  cell = triangulation.begin_active(), endc = triangulation.end();
  for (; cell!=endc; ++cell) 
    {
      for (unsigned int f=0; f < GeometryInfo<3>::faces_per_cell; ++f)
	{
	  const Point<3> face_center = cell->face(f)->center();
	  if (cell->face(f)->at_boundary())
	    {
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
    }

  pcout << "Done with Boundary Flags\n";
  std::vector<GridTools::PeriodicFacePair<typename parallel::distributed::Triangulation<3>::cell_iterator> > periodicity_vector;
  for (int i = 0; i < 3; ++i)
    {
      GridTools::collect_periodic_faces(triangulation, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vector);
    }
  triangulation.add_periodicity(periodicity_vector);
  std::cout << "Periodic Facepairs: " << periodicity_vector.size() << std::endl;
#endif  
    
  //
  //initialize FE objects
  //
  dofHandler.distribute_dofs (FE);
  dofHandlerEigen.distribute_dofs (FEEigen);

  locally_owned_dofs = dofHandler.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dofHandler, locally_relevant_dofs);
  DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), dofHandler, d_supportPoints);

  locally_owned_dofsEigen = dofHandlerEigen.locally_owned_dofs ();
  DoFTools::extract_locally_relevant_dofs (dofHandlerEigen, locally_relevant_dofsEigen);
  DoFTools::map_dofs_to_support_points(MappingQ1<3,3>(), dofHandlerEigen, d_supportPointsEigen);

  //
  //Extract real and imag DOF indices from the global vector - this will be needed in XHX operation, etc.
  //
#ifdef ENABLE_PERIODIC_BC
  FEValuesExtractors::Scalar real(0); //For Eigen
  ComponentMask componentMaskForRealDOF = FEEigen.component_mask (real);
  std::vector<bool> selectedDofsReal(locally_owned_dofsEigen.n_elements(), false);
  DoFTools::extract_dofs(dofHandlerEigen, componentMaskForRealDOF, selectedDofsReal);
  std::vector<unsigned int> local_dof_indices(locally_owned_dofsEigen.n_elements());
  locally_owned_dofsEigen.fill_index_vector(local_dof_indices);
  for (unsigned int i = 0; i < locally_owned_dofsEigen.n_elements(); i++)
    {
      if (selectedDofsReal[i]) 
	{
	  local_dof_indicesReal.push_back(local_dof_indices[i]);
	  localProc_dof_indicesReal.push_back(i);
	}
      else
	{
	  local_dof_indicesImag.push_back(local_dof_indices[i]);
	  localProc_dof_indicesImag.push_back(i);	  
	}
    }
#endif


  
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
  constraintsNone.clear(); constraintsNoneEigen.clear();
  DoFTools::make_hanging_node_constraints (dofHandler, constraintsNone);
  DoFTools::make_hanging_node_constraints (dofHandlerEigen, constraintsNoneEigen);


#ifdef ENABLE_PERIODIC_BC
  std::vector<GridTools::PeriodicFacePair<typename DoFHandler<3>::cell_iterator> > periodicity_vector2, periodicity_vector2Eigen;
  for (int i = 0; i < 3; ++i)
    {
      GridTools::collect_periodic_faces(dofHandler, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vector2);
      GridTools::collect_periodic_faces(dofHandlerEigen, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vector2Eigen);
    }
  DoFTools::make_periodicity_constraints<DoFHandler<3> >(periodicity_vector2, constraintsNone);
  DoFTools::make_periodicity_constraints<DoFHandler<3> >(periodicity_vector2Eigen, constraintsNoneEigen);

  constraintsNone.close();
  constraintsNoneEigen.close();

  pcout << "Detected Periodic Face Pairs: " << constraintsNone.n_constraints() << std::endl;

  pcout<<"Size of ConstraintsNone: "<< constraintsNone.n_constraints()<<std::endl;
  //constraintsNone.print(std::cout);

  pcout<<"Size of ConstraintsNoneEigen: "<< constraintsNoneEigen.n_constraints()<<std::endl;
  //constraintsNoneEigen.print(std::cout);

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
	  Point<3> p = d_supportPoints.find(i)->second;
	  unsigned int masterNode=(*constraintsTemp.get_constraint_entries(i))[0].first;
	  masterNodes.insert(masterNode);
	}
      }
    }
  }
  
  std::cout<<"Size of Master Nodes: "<<masterNodes.size()<<std::endl;
  

  //
  //fix wrong master map
  //
  for(types::global_dof_index i = 0; i <dofHandler.n_dofs(); ++i){
    if(locally_relevant_dofs.is_element(i)){
      if(constraintsTemp.is_constrained(i)){
	if (constraintsTemp.is_identity_constrained(i)){
	  Point<3> p = d_supportPoints.find(i)->second;
	  unsigned int masterNode=(*constraintsTemp.get_constraint_entries(i))[0].first;
	  unsigned int count=0, index=0;
	  for (unsigned int k=0; k<3; k++){
	    if (std::abs(std::abs(p[k])-halfxyzSpan)<periodicPrecision) {
	      count++;
	      index=k;
	    }
	  }
	  if (count==1){
	    Point<3> q = d_supportPoints.find(masterNode)->second;
	    unsigned int l = 1, m = 2;
	    if (index==1){l = 0; m = 2;}
	    else if (index==2){l = 0; m = 1;} 
	    if (!((std::abs(p[l]-q[l])<periodicPrecision) and (std::abs(p[m]-q[m])<periodicPrecision))){
	      bool foundNewMaster=false;
	      for (std::set<unsigned int>::iterator it=masterNodes.begin(); it!=masterNodes.end(); ++it){
		q=d_supportPoints.find(*it)->second;
		if (((std::abs(p[l]-q[l])<periodicPrecision) and (std::abs(p[m]-q[m])<periodicPrecision))){

		  //store the correct masterNodeId
		  unsigned int correctMasterDof = *it;

		  //One component
		  constraintsNone.add_line(i);
		  constraintsNone.add_entry(i, correctMasterDof, 1.0);
		  
		  foundNewMaster=true;
		  break;
		}
	      }
	      if (!foundNewMaster){
		std::cout<< " Wrong MasterNode for slave node: "<<masterNode<<" "<<i<<std::endl;
		std::cout<< "Error: Did not find a replacement master node for a wrong master-slave periodic pair"<<" "<<masterNode<<" slaveNode: "<<i<<std::endl;
		exit(-1);
	      }
	    }
	    else{

	      //One component
	      constraintsNone.add_line(i);
	      constraintsNone.add_entry(i, masterNode, 1.0);
	      
	    }
	  }
	  else{

	    //One component
	    constraintsNone.add_line(i);
	    constraintsNone.add_entry(i, masterNode, 1.0);

	  }
	}
      }
    }
  }
  constraintsNone.close();

  pcout<<"Size of ConstraintsNone New: "<< constraintsNone.n_constraints()<<std::endl;
  //constraintsNone.print(std::cout);
  
  //
  //fix periodic match for two-component fields
  //
  ConstraintMatrix constraintsTempEigen(constraintsNoneEigen); constraintsNoneEigen.clear();

  //
  //fill temp masterNodes and slaveNodes set
  //
  std::set<unsigned int> masterNodesEigen, slaveNodesEigen;

  //
  //fill all masters
  //
  for(types::global_dof_index i = 0; i <dofHandlerEigen.n_dofs(); ++i)
    {
      if(locally_relevant_dofsEigen.is_element(i))
	{
	  if(constraintsTempEigen.is_constrained(i))
	    {
	      if (constraintsTempEigen.is_identity_constrained(i))
		{
		  unsigned int masterNode=(*constraintsTempEigen.get_constraint_entries(i))[0].first;
		  masterNodesEigen.insert(masterNode);
		  slaveNodesEigen.insert(i);
		}
	    }
	}
    }

  std::set<unsigned int> globalMasterNodesEigen;
  exchangeMasterNodesList(masterNodesEigen,
			  globalMasterNodesEigen,
			  n_mpi_processes,
			  mpi_communicator);

  
  //
  //Now separate this set to real and imag sets
  //
  QGauss<3>  quadrature_formula(FEOrder+1);
  FEValues<3> fe_values (FEEigen, quadrature_formula, update_values);
  const unsigned int n_q_points    = quadrature_formula.size();
  const unsigned int dofs_per_cell = FEEigen.dofs_per_cell;
  std::vector<unsigned int> local_dof_indicesEigen(dofs_per_cell);

  std::set<unsigned int> masterNodesReal, masterNodesImag, slaveNodesReal, slaveNodesImag;
  typename DoFHandler<3>::active_cell_iterator cellEigen = dofHandlerEigen.begin_active(), endcellEigen = dofHandlerEigen.end();
  for (; cellEigen!=endcellEigen; ++cellEigen)
    {
      if (cellEigen->is_locally_owned())
	{
	  fe_values.reinit (cellEigen);
	  cellEigen->get_dof_indices (local_dof_indicesEigen);

	  for (unsigned int i = 0; i < dofs_per_cell; ++i)
	    {
	      const unsigned int ck = fe_values.get_fe().system_to_component_index(i).first; //This is the component index 0(real) or 1 (imag).
	      const unsigned int globalDOF = local_dof_indicesEigen[i];
	      //std::cout<<"Real or Imag: "<<ck<<std::endl;
	      
	      if (globalMasterNodesEigen.count(globalDOF)==1)
		{
		  if (ck == 0)
		    {
		      masterNodesReal.insert(globalDOF);
		    }
		  else 
		    {
		      masterNodesImag.insert(globalDOF);
		    }

		}
	      else if (slaveNodesEigen.count(globalDOF)==1)
		{

		  if (ck == 0)
		    {
		      slaveNodesReal.insert(globalDOF);
		    }
		  else 
		    {
		      slaveNodesImag.insert(globalDOF);

		    }

		}

	    }
	}
    }//end of cellEigen loop

  std::set<unsigned int> globalMasterNodesReal;
  std::set<unsigned int> globalMasterNodesImag;

  exchangeMasterNodesList(masterNodesReal,
			  globalMasterNodesReal,
			  n_mpi_processes,
			  mpi_communicator);

  exchangeMasterNodesList(masterNodesImag,
			  globalMasterNodesImag,
			  n_mpi_processes,
			  mpi_communicator);


  std::cout<<"Master Nodes Size: "<<globalMasterNodesEigen.size()<<std::endl;
  std::cout<<"Slave Nodes Size: "<<slaveNodesEigen.size()<<std::endl;

  std::cout<<"Master Nodes Real Size: "<<globalMasterNodesReal.size()<<std::endl;
  std::cout<<"Master Nodes Imag Size: "<<globalMasterNodesImag.size()<<std::endl;
  
  std::cout<<"Slave Nodes Real Size: "<<slaveNodesReal.size()<<std::endl;
  std::cout<<"Slave Nodes Imag Size: "<<slaveNodesImag.size()<<std::endl;

  

  //
  //fix wrong master map
  //
  for(types::global_dof_index i = 0; i <dofHandlerEigen.n_dofs(); i++)
    {
      if(locally_relevant_dofsEigen.is_element(i))
	{
	  if(constraintsTempEigen.is_constrained(i))
	    {
	      if (constraintsTempEigen.is_identity_constrained(i))
		{
		  Point<3> p=d_supportPointsEigen.find(i)->second;
		  unsigned int masterNode=(*constraintsTempEigen.get_constraint_entries(i))[0].first;
		  unsigned int count=0, index=0;
		  for (unsigned int k=0; k<3; k++)
		    {
		      if (std::abs(std::abs(p[k])-halfxyzSpan)<periodicPrecision) 
			{
			  count++;
			  index=k;
			}
		    }
		  if (count==1)
		    {
		      Point<3> q=d_supportPointsEigen.find(masterNode)->second;
		      unsigned int l=1, m=2;
		      if (index==1){l=0; m=2;}
		      else if (index==2){l=0; m=1;} 
		      if (!((std::abs(p[l]-q[l])<periodicPrecision) and (std::abs(p[m]-q[m])<periodicPrecision)))
			{
			  bool foundNewMaster=false;
			  if(slaveNodesReal.count(i) == 1)
			    {
			      for (std::set<unsigned int>::iterator it=globalMasterNodesReal.begin(); it!=globalMasterNodesReal.end(); ++it)
				{
				  q=d_supportPointsEigen.find(*it)->second;
				  if (((std::abs(p[l]-q[l])<periodicPrecision) and (std::abs(p[m]-q[m])<periodicPrecision)))
				    {
				      constraintsNoneEigen.add_line(i);
				      constraintsNoneEigen.add_entry(i, *it, 1.0);
				      foundNewMaster=true;
				      break;
				    }
				}
			    }
			  else
			    {
			      for (std::set<unsigned int>::iterator it=globalMasterNodesImag.begin(); it!=globalMasterNodesImag.end(); ++it)
				{
				  q=d_supportPointsEigen.find(*it)->second;
				  if (((std::abs(p[l]-q[l])<periodicPrecision) and (std::abs(p[m]-q[m])<periodicPrecision)))
				    {
				      constraintsNoneEigen.add_line(i);
				      constraintsNoneEigen.add_entry(i, *it, 1.0);
				      foundNewMaster=true;
				      break;
				    }
				}
			    }
			  if (!foundNewMaster)
			    {
			      pcout << "\nError: Did not find a replacement master node for a wrong master-slave periodic pair\n";
			      exit(-1);
			    }
			}
		      else
			{
			  constraintsNoneEigen.add_line(i);
			  constraintsNoneEigen.add_entry(i, masterNode, 1.0);
			}
		    }
		  else
		    {
		      constraintsNoneEigen.add_line(i);
		      constraintsNoneEigen.add_entry(i, masterNode, 1.0);
		    }
		}
	    }
	}
    }
  
  //loop over real slaves
  /*for(std::set<unsigned int>::iterator it = slaveNodesReal.begin(); it != slaveNodesReal.end(); ++it)
    {
      //check if slave on this face
      Point<3> S=d_supportPointsEigen.find(*it)->second;
      bool foundMaster=false;
      double D=2*halfxyzSpan;
      for(std::set<unsigned int>::iterator itM = globalMasterNodesReal.begin(); itM != globalMasterNodesReal.end(); ++itM){
	Point<3> M=d_supportPointsEigen.find(*itM)->second;
	double d= S.distance(M);
	if ((std::abs(d-D)<periodicPrecision) || (std::abs(d-std::sqrt(2)*D)<periodicPrecision) || (std::abs(d-std::sqrt(3)*D)<periodicPrecision)){
	  foundMaster=true;
	  constraintsNoneEigen.add_line(*it);
	  constraintsNoneEigen.add_entry(*it, *itM, 1.0);	
	  break;
	}
      }
      if (!foundMaster)
	{
	  std::cout<< "Error Real: Did not find a replacement master node for a wrong master-slave periodic pair "<<std::endl;
	  exit(-1);
	}
    }
  //
  for(std::set<unsigned int>::iterator it = slaveNodesImag.begin(); it != slaveNodesImag.end(); ++it)
    {
      //check if slave on this face
      Point<3> S=d_supportPointsEigen.find(*it)->second;
      bool foundMaster=false;
      double D=2*halfxyzSpan;
      for(std::set<unsigned int>::iterator itM = globalMasterNodesImag.begin(); itM != globalMasterNodesImag.end(); ++itM){
	Point<3> M=d_supportPointsEigen.find(*itM)->second;
	double d= S.distance(M);
	if ((std::abs(d-D)<periodicPrecision) || (std::abs(d-std::sqrt(2)*D)<periodicPrecision) || (std::abs(d-std::sqrt(3)*D)<periodicPrecision)){
	  foundMaster=true;
	  constraintsNoneEigen.add_line(*it);
	  constraintsNoneEigen.add_entry(*it, *itM, 1.0);	
	  break;
	}
      }
      if (!foundMaster)
	{
	  std::cout<< "Error Imag: Did not find a replacement master node for a wrong master-slave periodic pair "<<std::endl;
	  exit(-1);
	}
	}*/
 
  constraintsNoneEigen.close();
#else
  constraintsNone.close();
  constraintsNoneEigen.close();
#endif

  pcout<<"Size of ConstraintsNoneEigen New: "<< constraintsNoneEigen.n_constraints()<<std::endl;

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
 
  dofHandlerVector.push_back(&dofHandlerEigen); //DofHandler For Eigen
  eigenDofHandlerIndex=dofHandlerVector.size()-1; //For Eigen
  d_constraintsVector.push_back(&constraintsNoneEigen); //For Eigen;

  std::vector<Quadrature<1> > quadratureVector; 
  quadratureVector.push_back(QGauss<1>(FEOrder+1)); 
  quadratureVector.push_back(QGaussLobatto<1>(FEOrder+1));  


  matrix_free_data.reinit(dofHandlerVector, d_constraintsVector, quadratureVector, additional_data);


  //
  //initialize eigen vectors
  //
  matrix_free_data.initialize_dof_vector(vChebyshev,eigenDofHandlerIndex);
  v0Chebyshev.reinit(vChebyshev);
  fChebyshev.reinit(vChebyshev);
  aj[0].reinit(vChebyshev); aj[1].reinit(vChebyshev); aj[2].reinit(vChebyshev);
  aj[3].reinit(vChebyshev); aj[4].reinit(vChebyshev);
  for (unsigned int i=0; i<eigenVectors[0].size(); ++i)
    {  
      PSI[i]->reinit(vChebyshev);
      tempPSI[i]->reinit(vChebyshev);
      tempPSI2[i]->reinit(vChebyshev);
      tempPSI3[i]->reinit(vChebyshev);
      tempPSI4[i]->reinit(vChebyshev);
    } 
  
  for(unsigned int kPoint = 0; kPoint < d_maxkPoints; ++kPoint)
    {
      for (unsigned int i=0; i<eigenVectors[kPoint].size(); ++i)
	{
	  eigenVectors[kPoint][i]->reinit(vChebyshev);
	  eigenVectorsOrig[kPoint][i]->reinit(vChebyshev);
	}
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

#ifdef xc_id
#if   xc_id == 1
  exceptParamX = xc_func_init(&funcX,XC_LDA_X,XC_UNPOLARIZED);
  exceptParamC = xc_func_init(&funcC,XC_LDA_C_PZ,XC_UNPOLARIZED);
#elif xc_id == 2
  exceptParamX = xc_func_init(&funcX,XC_LDA_X,XC_UNPOLARIZED);
  exceptParamC = xc_func_init(&funcC,XC_LDA_C_PW,XC_UNPOLARIZED);
#elif xc_id == 3
  exceptParamX = xc_func_init(&funcX,XC_LDA_X,XC_UNPOLARIZED);
  exceptParamC = xc_func_init(&funcC,XC_LDA_C_VWN,XC_UNPOLARIZED);
#elif xc_id == 4
  exceptParamX = xc_func_init(&funcX,XC_GGA_X_PBE,XC_UNPOLARIZED);
  exceptParamC = xc_func_init(&funcC,XC_GGA_C_PBE,XC_UNPOLARIZED);
#elif xc_id > 4
#error exchange correlation id <= 4 required. 
#endif
#else
  exceptParamX = xc_func_init(&funcX,XC_LDA_X,XC_UNPOLARIZED);
  exceptParamC = xc_func_init(&funcC,XC_LDA_C_PZ,XC_UNPOLARIZED);
  if(exceptParamX != 0 || exceptParamC != 0){
    pcout<<"-------------------------------------"<<std::endl;
    pcout<<"Exchange or Correlation Functional not found"<<std::endl;
    pcout<<"-------------------------------------"<<std::endl;
    exit(-1);
  }
#endif

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
