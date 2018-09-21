// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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

namespace dftfe {

  struct quadData
  {
    double density;
  };

  namespace internal
  {
    void checkTriangulationEqualityAcrossProcessorPools
    (const parallel::distributed::Triangulation<3>& parallelTriangulation,
     const unsigned int numLocallyOwnedCells,
     const MPI_Comm & interpool_comm)
    {

      const unsigned int numberGlobalCellsParallelMinPools =
	Utilities::MPI::min(parallelTriangulation.n_global_active_cells(), interpool_comm);
      const unsigned int numberGlobalCellsParallelMaxPools =
	Utilities::MPI::max(parallelTriangulation.n_global_active_cells(), interpool_comm);
      AssertThrow(numberGlobalCellsParallelMinPools==numberGlobalCellsParallelMaxPools,ExcMessage("Number of global cells are different across pools."));

      const unsigned int numberLocalCellsMinPools =
	Utilities::MPI::min(numLocallyOwnedCells, interpool_comm);
      const unsigned int numberLocalCellsMaxPools =
	Utilities::MPI::max(numLocallyOwnedCells, interpool_comm);
      AssertThrow(numberLocalCellsMinPools==numberLocalCellsMaxPools,ExcMessage("Number of local cells are different across pools or in other words the physical partitions don't have the same ordering across pools."));
    }
  }

  void triangulationManager::generateCoarseMesh(parallel::distributed::Triangulation<3>& parallelTriangulation)
  {
    //
    //compute magnitudes of domainBounding Vectors
    //
    const double domainBoundingVectorMag1 = sqrt(d_domainBoundingVectors[0][0]*d_domainBoundingVectors[0][0] + d_domainBoundingVectors[0][1]*d_domainBoundingVectors[0][1] +  d_domainBoundingVectors[0][2]*d_domainBoundingVectors[0][2]);
    const double domainBoundingVectorMag2 = sqrt(d_domainBoundingVectors[1][0]*d_domainBoundingVectors[1][0] + d_domainBoundingVectors[1][1]*d_domainBoundingVectors[1][1] +  d_domainBoundingVectors[1][2]*d_domainBoundingVectors[1][2]);
    const double domainBoundingVectorMag3 = sqrt(d_domainBoundingVectors[2][0]*d_domainBoundingVectors[2][0] + d_domainBoundingVectors[2][1]*d_domainBoundingVectors[2][1] +  d_domainBoundingVectors[2][2]*d_domainBoundingVectors[2][2]);

    unsigned int subdivisions[3];subdivisions[0]=1.0;subdivisions[1]=1.0;subdivisions[2]=1.0;



    std::vector<double> numberIntervalsEachDirection;
    numberIntervalsEachDirection.push_back(domainBoundingVectorMag1/dftParameters::meshSizeOuterDomain);
    numberIntervalsEachDirection.push_back(domainBoundingVectorMag2/dftParameters::meshSizeOuterDomain);
    numberIntervalsEachDirection.push_back(domainBoundingVectorMag3/dftParameters::meshSizeOuterDomain);


    Point<3> vector1(d_domainBoundingVectors[0][0],d_domainBoundingVectors[0][1],d_domainBoundingVectors[0][2]);
    Point<3> vector2(d_domainBoundingVectors[1][0],d_domainBoundingVectors[1][1],d_domainBoundingVectors[1][2]);
    Point<3> vector3(d_domainBoundingVectors[2][0],d_domainBoundingVectors[2][1],d_domainBoundingVectors[2][2]);

    //
    //Generate coarse mesh
    //
    Point<3> basisVectors[3] = {vector1,vector2,vector3};


    for (unsigned int i=0; i<3;i++)
      {
	const double temp = numberIntervalsEachDirection[i]-std::floor(numberIntervalsEachDirection[i]);
	if(temp >= 0.5)
	  subdivisions[i] = std::ceil(numberIntervalsEachDirection[i]);
	else
	  subdivisions[i] = std::floor(numberIntervalsEachDirection[i]);
      }


    GridGenerator::subdivided_parallelepiped<3>(parallelTriangulation,
						subdivisions,
						basisVectors);

    //
    //Translate the main grid so that midpoint is at center
    //
    const Point<3> translation = 0.5*(vector1+vector2+vector3);
    GridTools::shift(-translation,parallelTriangulation);

    //
    //collect periodic faces of the first level mesh to set up periodic boundary conditions later
    //
    meshGenUtils::markPeriodicFacesNonOrthogonal(parallelTriangulation,d_domainBoundingVectors);

    if (dftParameters::verbosity>=4)
      pcout<<std::endl<< "Coarse triangulation number of elements: "<< parallelTriangulation.n_global_active_cells()<<std::endl;
  }

  void triangulationManager::refinementAlgorithmA(parallel::distributed::Triangulation<3>   & parallelTriangulation,
						  parallel::distributed::Triangulation<3>   & electrostaticsTriangulationRho,
						  parallel::distributed::Triangulation<3>   & electrostaticsTriangulationDisp,
						  const bool                                  generateElectrostaticsTria,
						  std::vector<unsigned int>                 & locallyOwnedCellsRefineFlags,
						  std::map<dealii::CellId,unsigned int>     & cellIdToCellRefineFlagMapLocal)
  {
    //
    //compute magnitudes of domainBounding Vectors
    //
    const double domainBoundingVectorMag1 = sqrt(d_domainBoundingVectors[0][0]*d_domainBoundingVectors[0][0] + d_domainBoundingVectors[0][1]*d_domainBoundingVectors[0][1] +  d_domainBoundingVectors[0][2]*d_domainBoundingVectors[0][2]);
    const double domainBoundingVectorMag2 = sqrt(d_domainBoundingVectors[1][0]*d_domainBoundingVectors[1][0] + d_domainBoundingVectors[1][1]*d_domainBoundingVectors[1][1] +  d_domainBoundingVectors[1][2]*d_domainBoundingVectors[1][2]);
    const double domainBoundingVectorMag3 = sqrt(d_domainBoundingVectors[2][0]*d_domainBoundingVectors[2][0] + d_domainBoundingVectors[2][1]*d_domainBoundingVectors[2][1] +  d_domainBoundingVectors[2][2]*d_domainBoundingVectors[2][2]);

    locallyOwnedCellsRefineFlags.clear();
    cellIdToCellRefineFlagMapLocal.clear();
    typename parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc, cellElectroRho, cellElectroDisp;
    cell = parallelTriangulation.begin_active();
    endc = parallelTriangulation.end();

    if(generateElectrostaticsTria)
      {
	cellElectroRho = electrostaticsTriangulationRho.begin_active();
	cellElectroDisp = electrostaticsTriangulationDisp.begin_active();
      }

    //
    //
    //
    for(;cell != endc; ++cell)
      {
      if(cell->is_locally_owned())
	{
	  const dealii::Point<3> center(cell->center());
	  double currentMeshSize = cell->minimum_vertex_distance();

	  //
	  //compute projection of the vector joining the center of domain and centroid of cell onto
	  //each of the domain bounding vectors
	  //
	  double projComponent_1 = (center[0]*d_domainBoundingVectors[0][0]+center[1]*d_domainBoundingVectors[0][1]+center[2]*d_domainBoundingVectors[0][2])/domainBoundingVectorMag1;
	  double projComponent_2 = (center[0]*d_domainBoundingVectors[1][0]+center[1]*d_domainBoundingVectors[1][1]+center[2]*d_domainBoundingVectors[1][2])/domainBoundingVectorMag2;
	  double projComponent_3 = (center[0]*d_domainBoundingVectors[2][0]+center[1]*d_domainBoundingVectors[2][1]+center[2]*d_domainBoundingVectors[2][2])/domainBoundingVectorMag3;


	  bool cellRefineFlag = false;


	  //loop over all atoms
	  double distanceToClosestAtom = 1e8;
	  Point<3> closestAtom;
	  for (unsigned int n=0; n<d_atomPositions.size(); n++)
	    {
	      Point<3> atom(d_atomPositions[n][2],d_atomPositions[n][3],d_atomPositions[n][4]);
	      if(center.distance(atom) < distanceToClosestAtom)
		{
		  distanceToClosestAtom = center.distance(atom);
		  closestAtom = atom;
		}
	    }

	  for(unsigned int iImageCharge=0; iImageCharge < d_imageAtomPositions.size(); ++iImageCharge)
	    {
	      Point<3> imageAtom(d_imageAtomPositions[iImageCharge][0],d_imageAtomPositions[iImageCharge][1],d_imageAtomPositions[iImageCharge][2]);
	      if(center.distance(imageAtom) < distanceToClosestAtom)
		{
		  distanceToClosestAtom = center.distance(imageAtom);
		  closestAtom = imageAtom;
		}
	    }

	  bool inOuterAtomBall = false;

	  if(distanceToClosestAtom <= dftParameters::outerAtomBallRadius)
	    inOuterAtomBall = true;

	  if(inOuterAtomBall && currentMeshSize > dftParameters::meshSizeOuterBall)
	    cellRefineFlag = true;

	  MappingQ1<3,3> mapping;
	  try
	    {
	      Point<3> p_cell = mapping.transform_real_to_unit_cell(cell,closestAtom);
	      double dist = GeometryInfo<3>::distance_to_unit_cell(p_cell);

	      if(dist < 1e-08 && currentMeshSize > dftParameters::meshSizeInnerBall)
		cellRefineFlag = true;

	    }
	  catch(MappingQ1<3>::ExcTransformationFailed)
	    {
	    }
	  //
	  //set refine flags
	  if(cellRefineFlag)
	    {
	      locallyOwnedCellsRefineFlags.push_back(1);
	      cellIdToCellRefineFlagMapLocal[cell->id()]=1;
	      cell->set_refine_flag();
	      if(generateElectrostaticsTria)
		{
		  cellElectroRho->set_refine_flag();
		  cellElectroDisp->set_refine_flag();
		}
	    }
	  else
	    {
	      cellIdToCellRefineFlagMapLocal[cell->id()]=0;
	      locallyOwnedCellsRefineFlags.push_back(0);
	    }
	}
      if(generateElectrostaticsTria)
	{
	  ++cellElectroRho;
	  ++cellElectroDisp;
	}
      }


  }

  //
  //generate adaptive mesh
  //

  void triangulationManager::generateMesh(parallel::distributed::Triangulation<3> & parallelTriangulation,
					  parallel::distributed::Triangulation<3> & electrostaticsTriangulationRho,
					  parallel::distributed::Triangulation<3> & electrostaticsTriangulationDisp,
					  const bool generateElectrostaticsTria)
  {
    if(!dftParameters::meshFileName.empty())
      {
	GridIn<3> gridinParallel;
	gridinParallel.attach_triangulation(parallelTriangulation);

	//
	//Read mesh in UCD format generated from Cubit
	//
	std::ifstream f1(dftParameters::meshFileName.c_str());
	gridinParallel.read_ucd(f1);

	meshGenUtils::markPeriodicFacesNonOrthogonal(parallelTriangulation,d_domainBoundingVectors);
      }
    else
      {

	generateCoarseMesh(parallelTriangulation);
	if(generateElectrostaticsTria)
	  {
	    generateCoarseMesh(electrostaticsTriangulationRho);
	    generateCoarseMesh(electrostaticsTriangulationDisp);
	    AssertThrow(parallelTriangulation.n_global_active_cells()==electrostaticsTriangulationRho.n_global_active_cells(),ExcMessage("Number of coarse mesh cells are different in electrostatics triangulations having rho field."));
	    AssertThrow(parallelTriangulation.n_global_active_cells()==electrostaticsTriangulationDisp.n_global_active_cells(),ExcMessage("Number of coarse mesh cells are different in electrostatics triangulations having disp field."));
	  }

	//
	//Multilayer refinement
	//
	unsigned int numLevels=0;
	bool refineFlag = true;

	while(refineFlag)
	  {
	    refineFlag = false;

	    std::vector<unsigned int> locallyOwnedCellsRefineFlags;
	    std::map<dealii::CellId,unsigned int> cellIdToCellRefineFlagMapLocal;
	    refinementAlgorithmA(parallelTriangulation,
				 electrostaticsTriangulationRho,
				 electrostaticsTriangulationDisp,
				 generateElectrostaticsTria,
				 locallyOwnedCellsRefineFlags,
				 cellIdToCellRefineFlagMapLocal);

	    //This sets the global refinement sweep flag
	    refineFlag = std::accumulate(locallyOwnedCellsRefineFlags.begin(),
					 locallyOwnedCellsRefineFlags.end(), 0)>0?1:0;
	    refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);

	    if (refineFlag)
	      {
		if(numLevels<d_max_refinement_steps)
		  {
		    if (dftParameters::verbosity>=4)
		      pcout<< "refinement in progress, level: "<< numLevels<<std::endl;

		    parallelTriangulation.execute_coarsening_and_refinement();
		    if(generateElectrostaticsTria)
		      {
			electrostaticsTriangulationRho.execute_coarsening_and_refinement();
			electrostaticsTriangulationDisp.execute_coarsening_and_refinement();
		      }
		    numLevels++;
		  }
		else
		  {
		    refineFlag=false;
		  }
	      }
	  }
	//
	//compute some adaptive mesh metrics
	//
	double minElemLength = dftParameters::meshSizeOuterDomain;
	unsigned int numLocallyOwnedCells=0;
	typename parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc, cellDisp;
	cell = parallelTriangulation.begin_active();
	endc = parallelTriangulation.end();
	for( ; cell != endc; ++cell)
	  {
	    if(cell->is_locally_owned())
	      {
		numLocallyOwnedCells++;
		if(cell->minimum_vertex_distance() < minElemLength) minElemLength = cell->minimum_vertex_distance();
	      }
	  }

	minElemLength = Utilities::MPI::min(minElemLength, mpi_communicator);

	//
	//print out adaptive mesh metrics
	//
	if (dftParameters::verbosity>=4)
	  {
            pcout<< "Triangulation generation summary: "<<std::endl<<" num elements: "<<parallelTriangulation.n_global_active_cells()<<", num refinement levels: "<<numLevels<<", min element length: "<<minElemLength<<std::endl;		  
	  }

	internal::checkTriangulationEqualityAcrossProcessorPools(parallelTriangulation,
								 numLocallyOwnedCells,
								 interpoolcomm);
	internal::checkTriangulationEqualityAcrossProcessorPools(parallelTriangulation,
								 numLocallyOwnedCells,
								 interBandGroupComm);


	if(generateElectrostaticsTria)
	  {
	    numLocallyOwnedCells = 0;
	    double minElemLengthRho = dftParameters::meshSizeOuterDomain;
	    double minElemLengthDisp = dftParameters::meshSizeOuterDomain;

	    cell = electrostaticsTriangulationRho.begin_active();
	    endc = electrostaticsTriangulationRho.end();
	    cellDisp = electrostaticsTriangulationDisp.begin_active();
	    for( ; cell != endc; ++cell)
	      {
		if(cell->is_locally_owned())
		  {
		    numLocallyOwnedCells++;
		    if(cell->minimum_vertex_distance() < minElemLengthRho) minElemLengthRho = cell->minimum_vertex_distance();
		    if(cellDisp->minimum_vertex_distance() < minElemLengthDisp) minElemLengthDisp = cellDisp->minimum_vertex_distance();
		  }
		++cellDisp;
	      }

	    minElemLengthRho = Utilities::MPI::min(minElemLengthRho, mpi_communicator);
	    minElemLengthDisp = Utilities::MPI::min(minElemLengthDisp, mpi_communicator);

	    //
	    //print out adaptive electrostatics mesh metrics 
	    //
	    if (dftParameters::verbosity>=4)
	      {
		pcout<< "Electrostatics Triangulation generation summary: "<<std::endl<<" num elements: "<<electrostaticsTriangulationRho.n_global_active_cells()<<", num refinement levels: "<<numLevels<<", min element length: "<<minElemLengthRho<<std::endl;		  

		pcout<< "Electrostatics Triangulation generation summary: "<<std::endl<<" num elements: "<<electrostaticsTriangulationDisp.n_global_active_cells()<<", num refinement levels: "<<numLevels<<", min element length: "<<minElemLengthDisp<<std::endl;

	      }


	    internal::checkTriangulationEqualityAcrossProcessorPools(electrostaticsTriangulationRho,
								     numLocallyOwnedCells,
								     interpoolcomm);

	    internal::checkTriangulationEqualityAcrossProcessorPools(electrostaticsTriangulationRho,
								     numLocallyOwnedCells,
								     interBandGroupComm);


	    internal::checkTriangulationEqualityAcrossProcessorPools(electrostaticsTriangulationDisp,
								     numLocallyOwnedCells,
								     interpoolcomm);

	    internal::checkTriangulationEqualityAcrossProcessorPools(electrostaticsTriangulationDisp,
								     numLocallyOwnedCells,
								     interBandGroupComm);

	  }

      }
  }


  /*void triangulationManager::generateSubdividedMeshWithQuadData(const dealii::MatrixFree<3,double> & matrixFreeData,
								const dealii::ConstraintMatrix & constraints,
								const dealii::Quadrature<3> & quadrature,
								const unsigned int FEOrder,
								const std::map<dealii::CellId,std::vector<double> > & rhoQuadValuesCoarse,
							        std::map<dealii::CellId,std::vector<double> > & rhoQuadValuesRefined)
  {
    
    //
    //create a new nodal field to store electron-density
    //
    vectorType rhoNodalFieldCoarse;
    matrixFreeData.initialize_dof_vector(rhoNodalFieldCoarse);
    rhoNodalFieldCoarse = 0.0;
    rhoQuadValuesRefined.clear();

    //
    //Get number of quadrature points
    //
    const unsigned int n_q_points = quadrature.size();

    //
    //create a lambda function for L2 projection of quadrature electron-density to nodal electron density
    //
     std::function<double(const typename dealii::DoFHandler<3>::active_cell_iterator & cell,const unsigned int q)> funcRho = [&](const typename dealii::DoFHandler<3>::active_cell_iterator & cell , const unsigned int q)
       {return rhoQuadValuesCoarse.find(cell->id())->second[q];};

   
    //
    //project and create a nodal field of the same mesh from the quadrature data (L2 projection from quad points to nodes)
    //
    dealii::VectorTools::project<3,dealii::parallel::distributed::Vector<double> >(dealii::MappingQ1<3,3>(),
										   matrixFreeData.get_dof_handler(),
										   constraints,
										   quadrature,
										   funcRho,
										   rhoNodalFieldCoarse);


    rhoNodalFieldCoarse.update_ghost_values();
    constraints.distribute(rhoNodalFieldCoarse);
    rhoNodalFieldCoarse.update_ghost_values();

    //
    //compute total charge using rho nodal field for debugging purposes
    //

    if(dftParameters::verbosity >= 4)
      {
	double normValue=0.0;
	QGauss<3>  quadrature_formula(FEOrder+1);
	FEValues<3> fe_valuesC(matrixFreeData.get_dof_handler().get_fe(), quadrature_formula, update_values | update_JxW_values);
	const unsigned int   dofs_per_cell = matrixFreeData.get_dof_handler().get_fe().dofs_per_cell;
    

	DoFHandler<3>::active_cell_iterator
	  cell = matrixFreeData.get_dof_handler().begin_active(),
	  endc = matrixFreeData.get_dof_handler().end();
	for (; cell!=endc; ++cell) {
	  if (cell->is_locally_owned()){
	    fe_valuesC.reinit (cell);
	    std::vector<double> tempRho(n_q_points);
	    fe_valuesC.get_function_values(rhoNodalFieldCoarse,tempRho);
	    for (unsigned int q_point=0; q_point<n_q_points; ++q_point){
	      normValue+=tempRho[q_point]*fe_valuesC.JxW(q_point);
	    }
	  }
	}
	pcout<<"Value of total charge on coarse mesh using L2 projected nodal field: "<< Utilities::MPI::sum(normValue, mpi_communicator)<<std::endl;
      }


    //
    //uniformly subdivide the mesh to prepare for solution transfer
    //
    dealii::DoFHandler<3> dofHandlerHRefined;
    dofHandlerHRefined.initialize(d_triangulationElectrostatics,dealii::FE_Q<3>(dealii::QGaussLobatto<1>(FEOrder+1)));
    
    dofHandlerHRefined.distribute_dofs(dofHandlerHRefined.get_fe());
    parallel::distributed::SolutionTransfer<3,vectorType> solTrans(dofHandlerHRefined);
    d_triangulationElectrostatics.set_all_refine_flags();

    d_triangulationElectrostatics.prepare_coarsening_and_refinement();
    solTrans.prepare_for_coarsening_and_refinement(rhoNodalFieldCoarse);
    d_triangulationElectrostatics.execute_coarsening_and_refinement();

    dofHandlerHRefined.distribute_dofs(dofHandlerHRefined.get_fe());

   
    //
    //print refined mesh details
    //
    if (dftParameters::verbosity>=2)
      {
	pcout << std::endl<<"Finite element mesh information after subdividing the mesh"<<std::endl;
	pcout<<"-------------------------------------------------"<<std::endl;
	pcout << "number of elements: "
	      << dofHandlerHRefined.get_triangulation().n_global_active_cells()
	      << std::endl
	      << "number of degrees of freedom: "
	      << dofHandlerHRefined.n_dofs()
	      << std::endl;
      }


    //
    //create a local matrix free object
    //
    dealii::IndexSet locallyRelevantDofs;
    dealii::DoFTools::extract_locally_relevant_dofs(dofHandlerHRefined, locallyRelevantDofs);

    dealii::ConstraintMatrix onlyHangingNodeConstraints;
    onlyHangingNodeConstraints.reinit(locallyRelevantDofs);
    dealii::DoFTools::make_hanging_node_constraints(dofHandlerHRefined, onlyHangingNodeConstraints);
    onlyHangingNodeConstraints.close();


    std::vector<const dealii::ConstraintMatrix*> matrixFreeConstraintsInputVector;
    matrixFreeConstraintsInputVector.push_back(&onlyHangingNodeConstraints);

    std::vector<const dealii::DoFHandler<3> *> matrixFreeDofHandlerVectorInput;
    matrixFreeDofHandlerVectorInput.push_back(&dofHandlerHRefined);

    std::vector<Quadrature<1> > quadratureVector;
    quadratureVector.push_back(QGauss<1>(FEOrder+1));

    //
    //matrix free data structure
    //
    typename dealii::MatrixFree<3>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = dealii::MatrixFree<3>::AdditionalData::partition_partition;
    dealii::MatrixFree<3,double> matrixFreeDataHRefined;
    matrixFreeDataHRefined.reinit(matrixFreeDofHandlerVectorInput,
    				  matrixFreeConstraintsInputVector,
    				  quadratureVector,
				  additional_data);

    //
    //create nodal field on the refined mesh
    //
    vectorType rhoNodalFieldRefined;
    matrixFreeDataHRefined.initialize_dof_vector(rhoNodalFieldRefined);
    rhoNodalFieldRefined.zero_out_ghosts();
    solTrans.interpolate(rhoNodalFieldRefined);
    rhoNodalFieldRefined.update_ghost_values();
    onlyHangingNodeConstraints.distribute(rhoNodalFieldRefined);
    rhoNodalFieldRefined.update_ghost_values();

    
    //
    //fill in quadrature values of the field on the refined mesh
    //
    FEValues<3> fe_values(dofHandlerHRefined.get_fe(),quadrature,update_values | update_JxW_values | update_quadrature_points);
    
    typename dealii::DoFHandler<3>::active_cell_iterator cellRefined = dofHandlerHRefined.begin_active(), endcRefined = dofHandlerHRefined.end();
    std::vector<double> tempRho1(n_q_points);

    double totalCharge1=0.0;
    for(; cellRefined!=endcRefined; ++cellRefined)
      {
	if(cellRefined->is_locally_owned())
	  {
	    fe_values.reinit(cellRefined);
	    //fe_values.get_function_values(rhoNodalFieldRefined,tempRho1);
	    fe_values.get_function_values(rhoNodalFieldCoarse,tempRho1);
	    rhoQuadValuesRefined[cellRefined->id()].resize(n_q_points);
	    for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
	      {
		rhoQuadValuesRefined[cellRefined->id()][q_point] = tempRho1[q_point];
		totalCharge1+=tempRho1[q_point]*fe_values.JxW(q_point);
	      }
	  }
      }

    pcout<<"Value of total charge on refined unmoved mesh after solution transfer: "<< Utilities::MPI::sum(totalCharge1, mpi_communicator)<<std::endl;

    //
    //compute total charge using quadrature values on the refined mesh
    //
    if (dftParameters::verbosity>=4)
      {
	typename dealii::DoFHandler<3>::active_cell_iterator cellRefinedNew = dofHandlerHRefined.begin_active(), endcRefinedNew = dofHandlerHRefined.end();
	double totalCharge=0.0;
	for(; cellRefinedNew!=endcRefinedNew; ++cellRefinedNew)
	  {
	    if(cellRefinedNew->is_locally_owned())
	      {
		fe_values.reinit(cellRefinedNew);
		for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
		  totalCharge += rhoQuadValuesRefined.find(cellRefinedNew->id())->second[q_point]*fe_values.JxW(q_point);
	      }
	  }
	pcout<<"Value of total charge on refined unmoved mesh after solution transfer: "<< Utilities::MPI::sum(totalCharge, mpi_communicator)<<std::endl;
      }
   
  }*/


  void triangulationManager::generateMesh(parallel::distributed::Triangulation<3> & parallelTriangulation,
					  parallel::distributed::Triangulation<3> & serialTriangulation,
					  parallel::distributed::Triangulation<3> & electrostaticsTriangulationRho,
					  parallel::distributed::Triangulation<3> & electrostaticsTriangulationDisp,
					  const bool generateElectrostaticsTria)
  {
    if(!dftParameters::meshFileName.empty())
      {
	GridIn<3> gridinParallel, gridinSerial;
	gridinParallel.attach_triangulation(parallelTriangulation);
	gridinSerial.attach_triangulation(serialTriangulation);

	//
	//Read mesh in UCD format generated from Cubit
	//
	std::ifstream f1(dftParameters::meshFileName.c_str());
	std::ifstream f2(dftParameters::meshFileName.c_str());
	gridinParallel.read_ucd(f1);
	gridinSerial.read_ucd(f2);

	meshGenUtils::markPeriodicFacesNonOrthogonal(parallelTriangulation,d_domainBoundingVectors);
	meshGenUtils::markPeriodicFacesNonOrthogonal(serialTriangulation,d_domainBoundingVectors);
      }
    else
      {

	generateCoarseMesh(parallelTriangulation);
	generateCoarseMesh(serialTriangulation);
	AssertThrow(parallelTriangulation.n_global_active_cells()==serialTriangulation.n_global_active_cells(),ExcMessage("Number of coarse mesh cells are different in serial and parallel triangulations."));

	if(generateElectrostaticsTria)
	  {
	    generateCoarseMesh(electrostaticsTriangulationRho);
	    generateCoarseMesh(electrostaticsTriangulationDisp);
	    AssertThrow(parallelTriangulation.n_global_active_cells()==electrostaticsTriangulationRho.n_global_active_cells(),ExcMessage("Number of coarse mesh cells are different in electrostatics triangulations having rho field."));
	    AssertThrow(parallelTriangulation.n_global_active_cells()==electrostaticsTriangulationDisp.n_global_active_cells(),ExcMessage("Number of coarse mesh cells are different in electrostatics triangulations disp field."));
	  }



	//
	//Multilayer refinement
	//
	unsigned int numLevels=0;
	bool refineFlag = true;
	while(refineFlag)
	  {
	    refineFlag = false;
	    std::vector<unsigned int> locallyOwnedCellsRefineFlags;
	    std::map<dealii::CellId,unsigned int> cellIdToCellRefineFlagMapLocal;
	    refinementAlgorithmA(parallelTriangulation,
				 electrostaticsTriangulationRho,
				 electrostaticsTriangulationDisp,
				 generateElectrostaticsTria,
				 locallyOwnedCellsRefineFlags,
				 cellIdToCellRefineFlagMapLocal);


	    //This sets the global refinement sweep flag
	    refineFlag = std::accumulate(locallyOwnedCellsRefineFlags.begin(),
					 locallyOwnedCellsRefineFlags.end(), 0)>0?1:0;
	    refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);

	    //Refine
	    if (refineFlag)
	      {

		//First refine serial mesh
		refineSerialMesh(cellIdToCellRefineFlagMapLocal,
				 mpi_communicator,
				 serialTriangulation) ;

		if(numLevels<d_max_refinement_steps)
		  {
		    if (dftParameters::verbosity>=4)
		      pcout<< "refinement in progress, level: "<< numLevels<<std::endl;

		    parallelTriangulation.execute_coarsening_and_refinement();
		    if(generateElectrostaticsTria)
		      {
			electrostaticsTriangulationRho.execute_coarsening_and_refinement();
			electrostaticsTriangulationDisp.execute_coarsening_and_refinement();
		      }

		    numLevels++;
		  }
		else
		  {
		    refineFlag=false;
		  }
	      }

	  }
	//
	//compute some adaptive mesh metrics
	//
	double minElemLength = dftParameters::meshSizeOuterDomain;
	typename parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc, cellDisp;
	cell = parallelTriangulation.begin_active();
	endc = parallelTriangulation.end();
	unsigned int numLocallyOwnedCells=0;
	for( ; cell != endc; ++cell)
	  {
	    if(cell->is_locally_owned())
	      {
		numLocallyOwnedCells++;
		if(cell->minimum_vertex_distance() < minElemLength) minElemLength = cell->minimum_vertex_distance();
	      }
	  }

	minElemLength = Utilities::MPI::min(minElemLength, mpi_communicator);

	//
	//print out adaptive mesh metrics and check mesh generation synchronization across pools
	//
	if (dftParameters::verbosity>=4)
	  {
	    pcout<< "Triangulation generation summary: "<<std::endl<<" num elements: "<<parallelTriangulation.n_global_active_cells()<<", num refinement levels: "<<numLevels<<", min element length: "<<minElemLength<<std::endl;
	  }

	internal::checkTriangulationEqualityAcrossProcessorPools(parallelTriangulation,
								 numLocallyOwnedCells,
								 interpoolcomm);
	internal::checkTriangulationEqualityAcrossProcessorPools(parallelTriangulation,
								 numLocallyOwnedCells,
								 interBandGroupComm);

	const unsigned int numberGlobalCellsParallel = parallelTriangulation.n_global_active_cells();
	const unsigned int numberGlobalCellsSerial = serialTriangulation.n_global_active_cells();

	if (dftParameters::verbosity>=4)
	  pcout<<" numParallelCells: "<< numberGlobalCellsParallel<<", numSerialCells: "<< numberGlobalCellsSerial<<std::endl;

	AssertThrow(numberGlobalCellsParallel==numberGlobalCellsSerial,ExcMessage("Number of cells are different for parallel and serial triangulations"));


	if(generateElectrostaticsTria)
	  {
	    numLocallyOwnedCells = 0;
	    double minElemLengthElectroTria = dftParameters::meshSizeOuterDomain;
	    double minElemLengthDisp = dftParameters::meshSizeOuterDomain;

	    cell = electrostaticsTriangulationRho.begin_active();
	    endc = electrostaticsTriangulationRho.end();
	    cellDisp = electrostaticsTriangulationDisp.begin_active();
	    for( ; cell != endc; ++cell)
	      {
		if(cell->is_locally_owned())
		  {
		    numLocallyOwnedCells++;
		    if(cell->minimum_vertex_distance() < minElemLengthElectroTria) minElemLengthElectroTria = cell->minimum_vertex_distance();
		    if(cellDisp->minimum_vertex_distance() < minElemLengthDisp) minElemLengthDisp = cellDisp->minimum_vertex_distance();
		  }
	      }

	    minElemLengthElectroTria = Utilities::MPI::min(minElemLengthElectroTria, mpi_communicator);
	    minElemLengthDisp = Utilities::MPI::min(minElemLengthDisp, mpi_communicator);

	    //
	    //print out adaptive electrostatics mesh metrics 
	    //
	    if (dftParameters::verbosity>=4)
	      {
		pcout<< "Electrostatics Triangulation generation summary: "<<std::endl<<" num elements: "<<electrostaticsTriangulationRho.n_global_active_cells()<<", num refinement levels: "<<numLevels<<", min element length: "<<minElemLengthElectroTria<<std::endl;	

		pcout<< "Electrostatics Triangulation generation summary: "<<std::endl<<" num elements: "<<electrostaticsTriangulationDisp.n_global_active_cells()<<", num refinement levels: "<<numLevels<<", min element length: "<<minElemLengthDisp<<std::endl;
	  
	      }


	    internal::checkTriangulationEqualityAcrossProcessorPools(electrostaticsTriangulationRho,
								     numLocallyOwnedCells,
								     interpoolcomm);

	    internal::checkTriangulationEqualityAcrossProcessorPools(electrostaticsTriangulationRho,
								     numLocallyOwnedCells,
								     interBandGroupComm);

	    internal::checkTriangulationEqualityAcrossProcessorPools(electrostaticsTriangulationDisp,
								     numLocallyOwnedCells,
								     interpoolcomm);

	    internal::checkTriangulationEqualityAcrossProcessorPools(electrostaticsTriangulationDisp,
								     numLocallyOwnedCells,
								     interBandGroupComm);


	  }

      }
  }


  //
  void triangulationManager::refineSerialMesh
  (const std::map<dealii::CellId,unsigned int> & cellIdToCellRefineFlagMapLocal,
   const MPI_Comm &mpi_comm,
   parallel::distributed::Triangulation<3>& serialTriangulation)

  {
    typename parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc;
    cell = serialTriangulation.begin_active();
    endc = serialTriangulation.end();
    for(;cell != endc; ++cell)
      if(cell->is_locally_owned())
	{
	  unsigned int refineFlag =0;

	  std::map<dealii::CellId,unsigned int>::const_iterator
	    iter=cellIdToCellRefineFlagMapLocal.find(cell->id());
	  if (iter!=cellIdToCellRefineFlagMapLocal.end())
	    refineFlag=iter->second;

	  refineFlag = Utilities::MPI::sum(refineFlag,mpi_comm);
	  if (refineFlag==1)
	    cell->set_refine_flag();
	}

    serialTriangulation.execute_coarsening_and_refinement();
  }


}
