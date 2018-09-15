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

    void triangulationManager::refinementAlgorithmA(parallel::distributed::Triangulation<3>& parallelTriangulation,
	                      std::vector<unsigned int> & locallyOwnedCellsRefineFlags,
		              std::map<dealii::CellId,unsigned int> & cellIdToCellRefineFlagMapLocal)
    {
	  //
	  //compute magnitudes of domainBounding Vectors
	  //
	  const double domainBoundingVectorMag1 = sqrt(d_domainBoundingVectors[0][0]*d_domainBoundingVectors[0][0] + d_domainBoundingVectors[0][1]*d_domainBoundingVectors[0][1] +  d_domainBoundingVectors[0][2]*d_domainBoundingVectors[0][2]);
	  const double domainBoundingVectorMag2 = sqrt(d_domainBoundingVectors[1][0]*d_domainBoundingVectors[1][0] + d_domainBoundingVectors[1][1]*d_domainBoundingVectors[1][1] +  d_domainBoundingVectors[1][2]*d_domainBoundingVectors[1][2]);
	  const double domainBoundingVectorMag3 = sqrt(d_domainBoundingVectors[2][0]*d_domainBoundingVectors[2][0] + d_domainBoundingVectors[2][1]*d_domainBoundingVectors[2][1] +  d_domainBoundingVectors[2][2]*d_domainBoundingVectors[2][2]);

	  locallyOwnedCellsRefineFlags.clear();
	  cellIdToCellRefineFlagMapLocal.clear();
	  typename parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc;
	  cell = parallelTriangulation.begin_active();
	  endc = parallelTriangulation.end();
	  //
	  for(;cell != endc; ++cell)
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
		      }
		    else
		      {
			  cellIdToCellRefineFlagMapLocal[cell->id()]=0;
			  locallyOwnedCellsRefineFlags.push_back(0);
		      }
		}


    }

    //
    //generate adaptive mesh
    //

    void triangulationManager::generateMesh(parallel::distributed::Triangulation<3>& parallelTriangulation)
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
	  typename parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc;
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
	}
    }


  void triangulationManager::generateSubdividedMeshWithQuadData(const dealii::MatrixFree<3,double> & matrixFreeData,
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

    //
    //Get number of quadrature points
    //
    const unsigned int n_q_points = quadrature.size();


    //
    //create electron-density quadrature data using "CellDataStorage" class of dealii
    //
    /*CellDataStorage<typename DoFHandler<3>::active_cell_iterator,quadData> rhoQuadDataStorage;

    rhoQuadDataStorage.initialize(matrixFreeData.get_dof_handler().begin_active(),
				  matrixFreeData.get_dof_handler().end(),
				  n_q_points);


    //
    //Copy rho values into CellDataStorage Container
    //
    typename dealii::DoFHandler<3>::active_cell_iterator cell = matrixFreeData.get_dof_handler().begin_active(), endc = matrixFreeData.get_dof_handler().end();

    for(; cell!=endc; ++cell)
      {
	if(cell->is_locally_owned())
	  {
	    const std::vector<std::shared_ptr<quadData> > rhoQuadPointVector = rhoQuadDataStorage.get_data(cell);
	    for(unsigned int q = 0; q < n_q_points; ++q)
	      {
		rhoQuadPointVector[q]->density = rhoQuadValuesCoarse.find(cell->id())->second[q];
	      }
	  }
	  }*/


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

    //
    //uniformly subdivide the mesh to prepare for solution transfer
    //
    parallel::distributed::SolutionTransfer<3,vectorType> solTrans(matrixFreeData.get_dof_handler());
    d_triangulationElectrostatics.set_all_refine_flags();
    d_triangulationElectrostatics.prepare_coarsening_and_refinement();
    solTrans.prepare_for_coarsening_and_refinement(rhoNodalFieldCoarse);
    d_triangulationElectrostatics.execute_coarsening_and_refinement();


    //
    //create a dofHandler for the refined Mesh
    //
    dealii::DoFHandler<3> dofHandlerHRefined;
    dofHandlerHRefined.initialize(d_triangulationElectrostatics,dealii::FE_Q<3>(dealii::QGaussLobatto<1>(FEOrder+1)));
    dofHandlerHRefined.distribute_dofs(dofHandlerHRefined.get_fe());


    //
    //create nodal field on the refined mesh
    //
    vectorType rhoNodalFieldRefined;
    rhoNodalFieldRefined.reinit(dofHandlerHRefined.n_dofs());
    solTrans.interpolate(rhoNodalFieldRefined);
    rhoNodalFieldRefined.update_ghost_values();

    
    //
    //Remove other updates
    //
    FEValues<3> fe_values(dofHandlerHRefined.get_fe(),quadrature,update_values | update_JxW_values | update_quadrature_points);

    typename dealii::DoFHandler<3>::active_cell_iterator cellRefined = dofHandlerHRefined.begin_active(), endcRefined = dofHandlerHRefined.end();


    std::vector<double> tempRho(n_q_points);

    for(; cellRefined!=endcRefined; ++cellRefined)
      {
	if(cellRefined->is_locally_owned())
	  {
	    fe_values.reinit(cellRefined);
	    fe_values.get_function_values(rhoNodalFieldRefined,tempRho);
	    //rhoQuadValuesRefined.find(cellRefined->id()]->second.clear();
	    for(unsigned int q_point = 0; q_point < n_q_points; ++q_point)
	      rhoQuadValuesRefined[cellRefined->id()].push_back(tempRho[q_point]);
	  }
      }

    
  }


    void triangulationManager::generateMesh(parallel::distributed::Triangulation<3>& parallelTriangulation,
					    parallel::distributed::Triangulation<3>& serialTriangulation)
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
	  typename parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc;
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
