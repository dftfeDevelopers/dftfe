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
                                                  parallel::distributed::Triangulation<3>   & electrostaticsTriangulationForce,
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
    typename parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc, cellElectroRho, cellElectroDisp, cellElectroForce;
    cell = parallelTriangulation.begin_active();
    endc = parallelTriangulation.end();

    if(generateElectrostaticsTria)
      {
	cellElectroRho = electrostaticsTriangulationRho.begin_active();
	cellElectroDisp = electrostaticsTriangulationDisp.begin_active();
	cellElectroForce = electrostaticsTriangulationForce.begin_active();
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
		  cellElectroForce->set_refine_flag();
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
	  ++cellElectroForce;
	}
      }


  }

  //
  //generate adaptive mesh
  //

  void triangulationManager::generateMesh(parallel::distributed::Triangulation<3> & parallelTriangulation,
					  parallel::distributed::Triangulation<3> & electrostaticsTriangulationRho,
					  parallel::distributed::Triangulation<3> & electrostaticsTriangulationDisp,
					  parallel::distributed::Triangulation<3> & electrostaticsTriangulationForce,
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
	    generateCoarseMesh(electrostaticsTriangulationForce);
	    AssertThrow(parallelTriangulation.n_global_active_cells()==electrostaticsTriangulationRho.n_global_active_cells(),ExcMessage("Number of coarse mesh cells are different in electrostatics triangulations having rho field."));
	    AssertThrow(parallelTriangulation.n_global_active_cells()==electrostaticsTriangulationDisp.n_global_active_cells(),ExcMessage("Number of coarse mesh cells are different in electrostatics triangulations having disp field."));
	    AssertThrow(parallelTriangulation.n_global_active_cells()==electrostaticsTriangulationForce.n_global_active_cells(),ExcMessage("Number of coarse mesh cells are different in electrostatics triangulation for force computation."));
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
				 electrostaticsTriangulationForce,
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
			electrostaticsTriangulationForce.execute_coarsening_and_refinement();
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
	    double minElemLengthForce = dftParameters::meshSizeOuterDomain;

	    cell = electrostaticsTriangulationRho.begin_active();
	    endc = electrostaticsTriangulationRho.end();
	    cellDisp = electrostaticsTriangulationDisp.begin_active();
	    for( ; cell != endc; ++cell)
	      {
		if(cell->is_locally_owned())
		  {
		    numLocallyOwnedCells++;
		    if(cell->minimum_vertex_distance() < minElemLengthRho)
			minElemLengthRho = cell->minimum_vertex_distance();
		    if(cellDisp->minimum_vertex_distance() < minElemLengthDisp)
			minElemLengthDisp = cellDisp->minimum_vertex_distance();
		    if(cellDisp->minimum_vertex_distance() < minElemLengthForce)
			minElemLengthForce = cellDisp->minimum_vertex_distance();
		  }
		++cellDisp;
	      }

	    minElemLengthRho = Utilities::MPI::min(minElemLengthRho, mpi_communicator);
	    minElemLengthDisp = Utilities::MPI::min(minElemLengthDisp, mpi_communicator);
            minElemLengthForce = Utilities::MPI::min(minElemLengthDisp, mpi_communicator);
	    //
	    //print out adaptive electrostatics mesh metrics
	    //
	    if (dftParameters::verbosity>=4)
	      {
		pcout<< "Electrostatics Triangulation generation summary: "<<std::endl<<" num elements: "<<electrostaticsTriangulationRho.n_global_active_cells()<<", num refinement levels: "<<numLevels<<", min element length: "<<minElemLengthRho<<std::endl;

		pcout<< "Electrostatics Triangulation generation summary: "<<std::endl<<" num elements: "<<electrostaticsTriangulationDisp.n_global_active_cells()<<", num refinement levels: "<<numLevels<<", min element length: "<<minElemLengthDisp<<std::endl;

		pcout<< "Electrostatics Triangulation generation summary: "<<std::endl<<" num elements: "<<electrostaticsTriangulationForce.n_global_active_cells()<<", num refinement levels: "<<numLevels<<", min element length: "<<minElemLengthForce<<std::endl;

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

	    internal::checkTriangulationEqualityAcrossProcessorPools(electrostaticsTriangulationForce,
								     numLocallyOwnedCells,
								     interpoolcomm);

	    internal::checkTriangulationEqualityAcrossProcessorPools(electrostaticsTriangulationForce,
								     numLocallyOwnedCells,
								     interBandGroupComm);

	  }

      }
  }


  void triangulationManager::generateMesh
          (parallel::distributed::Triangulation<3> & parallelTriangulation,
	   parallel::distributed::Triangulation<3> & serialTriangulation,
	   parallel::distributed::Triangulation<3> & electrostaticsTriangulationRho,
	   parallel::distributed::Triangulation<3> & electrostaticsTriangulationDisp,
	   parallel::distributed::Triangulation<3> & electrostaticsTriangulationForce,
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
	    generateCoarseMesh(electrostaticsTriangulationForce);
	    AssertThrow(parallelTriangulation.n_global_active_cells()==electrostaticsTriangulationRho.n_global_active_cells(),ExcMessage("Number of coarse mesh cells are different in electrostatics triangulations having rho field."));
	    AssertThrow(parallelTriangulation.n_global_active_cells()==electrostaticsTriangulationDisp.n_global_active_cells(),ExcMessage("Number of coarse mesh cells are different in electrostatics triangulations disp field."));
	    AssertThrow(parallelTriangulation.n_global_active_cells()==electrostaticsTriangulationForce.n_global_active_cells(),ExcMessage("Number of coarse mesh cells are different in electrostatics triangulations for force computation."));
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
				 electrostaticsTriangulationForce,
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
			electrostaticsTriangulationForce.execute_coarsening_and_refinement();
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
	    double minElemLengthRho = dftParameters::meshSizeOuterDomain;
	    double minElemLengthDisp = dftParameters::meshSizeOuterDomain;
	    double minElemLengthForce = dftParameters::meshSizeOuterDomain;

	    cell = electrostaticsTriangulationRho.begin_active();
	    endc = electrostaticsTriangulationRho.end();
	    cellDisp = electrostaticsTriangulationDisp.begin_active();
	    for( ; cell != endc; ++cell)
	      {
		if(cell->is_locally_owned())
		  {
		    numLocallyOwnedCells++;
		    if(cell->minimum_vertex_distance() < minElemLengthRho)
			minElemLengthRho = cell->minimum_vertex_distance();
		    if(cellDisp->minimum_vertex_distance() < minElemLengthDisp)
			minElemLengthDisp = cellDisp->minimum_vertex_distance();
		    if(cellDisp->minimum_vertex_distance() < minElemLengthForce)
			minElemLengthForce = cellDisp->minimum_vertex_distance();
		  }
		++cellDisp;
	      }

	    minElemLengthRho = Utilities::MPI::min(minElemLengthRho, mpi_communicator);
	    minElemLengthDisp = Utilities::MPI::min(minElemLengthDisp, mpi_communicator);
            minElemLengthForce = Utilities::MPI::min(minElemLengthDisp, mpi_communicator);
	    //
	    //print out adaptive electrostatics mesh metrics
	    //
	    if (dftParameters::verbosity>=4)
	      {
		pcout<< "Electrostatics Triangulation generation summary: "<<std::endl<<" num elements: "<<electrostaticsTriangulationRho.n_global_active_cells()<<", num refinement levels: "<<numLevels<<", min element length: "<<minElemLengthRho<<std::endl;

		pcout<< "Electrostatics Triangulation generation summary: "<<std::endl<<" num elements: "<<electrostaticsTriangulationDisp.n_global_active_cells()<<", num refinement levels: "<<numLevels<<", min element length: "<<minElemLengthDisp<<std::endl;

		pcout<< "Electrostatics Triangulation generation summary: "<<std::endl<<" num elements: "<<electrostaticsTriangulationForce.n_global_active_cells()<<", num refinement levels: "<<numLevels<<", min element length: "<<minElemLengthForce<<std::endl;

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

	    internal::checkTriangulationEqualityAcrossProcessorPools(electrostaticsTriangulationForce,
								     numLocallyOwnedCells,
								     interpoolcomm);

	    internal::checkTriangulationEqualityAcrossProcessorPools(electrostaticsTriangulationForce,
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
