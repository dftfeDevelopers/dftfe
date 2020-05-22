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


		void computeMeshMetrics(const parallel::distributed::Triangulation<3> & parallelTriangulation,
				const std::string & printCommand,
				const dealii::ConditionalOStream &  pcout,
				const MPI_Comm & mpi_comm,
				const MPI_Comm & interpool_comm1,
				const MPI_Comm & interpool_comm2)

		{
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

			minElemLength = Utilities::MPI::min(minElemLength, mpi_comm);

			//
			//print out adaptive mesh metrics
			//
			if (dftParameters::verbosity>=4)
			{
				pcout<< printCommand <<std::endl<<" num elements: "<<parallelTriangulation.n_global_active_cells()<<", min element length: "<<minElemLength<<std::endl;
			}

			checkTriangulationEqualityAcrossProcessorPools(parallelTriangulation,
					numLocallyOwnedCells,
					interpool_comm1);
			checkTriangulationEqualityAcrossProcessorPools(parallelTriangulation,
					numLocallyOwnedCells,
					interpool_comm2);

		}

		void computeLocalFiniteElementError(const dealii::DoFHandler<3> & dofHandler,
				const std::vector<const distributedCPUVec<double>*> & eigenVectorsArray,
				std::vector<double>  & errorInEachCell,
				const unsigned int FEOrder)
		{

			typename dealii::DoFHandler<3>::active_cell_iterator cell, endc;
			cell = dofHandler.begin_active();
			endc = dofHandler.end();

			errorInEachCell.clear();

			//
			//create some FE data structures
			//
			dealii::QGauss<3>  quadrature(FEOrder+1);
			dealii::FEValues<3> fe_values (dofHandler.get_fe(), quadrature, dealii::update_values | dealii::update_JxW_values | dealii::update_3rd_derivatives);
			const unsigned int num_quad_points = quadrature.size();

			std::vector<Tensor<3,3,double> > thirdDerivatives(num_quad_points);

			for(;cell != endc; ++cell)
			{
				if(cell->is_locally_owned())
				{
					fe_values.reinit(cell);

					const dealii::Point<3> center(cell->center());
					double currentMeshSize = cell->minimum_vertex_distance();//cell->diameter();
					//
					//Estimate the error for the current mesh
					//
					double derPsiSquare = 0.0;
					for(unsigned int iwave = 0; iwave < eigenVectorsArray.size(); ++iwave)
					{
						fe_values.get_function_third_derivatives(*eigenVectorsArray[iwave],
								thirdDerivatives);
						for(unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
						{
							double sum = 0.0;
							for(unsigned int i = 0; i < 3; ++i)
							{
								for(unsigned int j = 0; j < 3; ++j)
								{
									for(unsigned int k = 0; k < 3; ++k)
									{
										sum += std::abs(thirdDerivatives[q_point][i][j][k])*std::abs(thirdDerivatives[q_point][i][j][k]);
									}
								}
							}

							derPsiSquare += sum*fe_values.JxW(q_point);
						}//q_point
					}//iwave
					double exponent = 4.0;
					double error = pow(currentMeshSize,exponent)*derPsiSquare;
					errorInEachCell.push_back(error);
				}
				else
				{
					errorInEachCell.push_back(0.0);
				}
			}

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

		double largestMeshSizeAroundAtom=dftParameters::meshSizeOuterBall;

		if (dftParameters::useMeshSizesFromAtomsFile)
		{
			largestMeshSizeAroundAtom=1e-6;
			for (unsigned int n=0; n<d_atomPositions.size(); n++)
			{
				if(d_atomPositions[n][5] >largestMeshSizeAroundAtom)
					largestMeshSizeAroundAtom=d_atomPositions[n][5];
			}
		}

		if (dftParameters::autoUserMeshParams && !dftParameters::reproducible_output)
		{
			double baseMeshSize1, baseMeshSize2, baseMeshSize3;
			if (dftParameters::periodicX ||dftParameters::periodicY ||dftParameters::periodicZ)
			{
				baseMeshSize1=std::pow(2,round(log2(2.0/largestMeshSizeAroundAtom)))*largestMeshSizeAroundAtom;
				baseMeshSize2=std::pow(2,round(log2(2.0/largestMeshSizeAroundAtom)))*largestMeshSizeAroundAtom;
				baseMeshSize3=std::pow(2,round(log2(2.0/largestMeshSizeAroundAtom)))*largestMeshSizeAroundAtom;
			}
			else
			{
				baseMeshSize1=std::pow(2,round(log2(std::min(domainBoundingVectorMag1/8.0,8.0)/largestMeshSizeAroundAtom)))*largestMeshSizeAroundAtom;
				baseMeshSize2=std::pow(2,round(log2(std::min(domainBoundingVectorMag2/8.0,8.0)/largestMeshSizeAroundAtom)))*largestMeshSizeAroundAtom;
				baseMeshSize3=std::pow(2,round(log2(std::min(domainBoundingVectorMag3/8.0,8.0)/largestMeshSizeAroundAtom)))*largestMeshSizeAroundAtom;
			}

			numberIntervalsEachDirection.push_back(domainBoundingVectorMag1/baseMeshSize1);
			numberIntervalsEachDirection.push_back(domainBoundingVectorMag2/baseMeshSize2);
			numberIntervalsEachDirection.push_back(domainBoundingVectorMag3/baseMeshSize3);
		}
		else
		{
			numberIntervalsEachDirection.push_back(domainBoundingVectorMag1/dftParameters::meshSizeOuterDomain);
			numberIntervalsEachDirection.push_back(domainBoundingVectorMag2/dftParameters::meshSizeOuterDomain);
			numberIntervalsEachDirection.push_back(domainBoundingVectorMag3/dftParameters::meshSizeOuterDomain);
		}

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

	bool triangulationManager::refinementAlgorithmA(parallel::distributed::Triangulation<3>   & parallelTriangulation,
			parallel::distributed::Triangulation<3>   & electrostaticsTriangulationRho,
			parallel::distributed::Triangulation<3>   & electrostaticsTriangulationDisp,
			parallel::distributed::Triangulation<3>   & electrostaticsTriangulationForce,
			const bool                                  generateElectrostaticsTria,
			std::vector<unsigned int>                 & locallyOwnedCellsRefineFlags,
			std::map<dealii::CellId,unsigned int>     & cellIdToCellRefineFlagMapLocal,
			const bool smoothenCellsOnPeriodicBoundary,
			const double smootheningFactor)
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

		std::map<dealii::CellId,unsigned int> cellIdToLocallyOwnedId;
		unsigned int locallyOwnedCount=0;

		bool isAnyCellRefined=false;
		double smallestMeshSizeAroundAtom=dftParameters::meshSizeOuterBall;

		if (dftParameters::useMeshSizesFromAtomsFile)
		{
			smallestMeshSizeAroundAtom=1e+6;
			for (unsigned int n=0; n<d_atomPositions.size(); n++)
			{
				if(d_atomPositions[n][5] <smallestMeshSizeAroundAtom)
					smallestMeshSizeAroundAtom=d_atomPositions[n][5];
			}
		}

		//
		//
		//
		for(;cell != endc; ++cell)
		{
			if(cell->is_locally_owned())
			{
				cellIdToLocallyOwnedId[cell->id()]=locallyOwnedCount;
				locallyOwnedCount++;

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
				unsigned int closestAtomId=0;
				for (unsigned int n=0; n<d_atomPositions.size(); n++)
				{
					Point<3> atom(d_atomPositions[n][2],d_atomPositions[n][3],d_atomPositions[n][4]);
					if(center.distance(atom) < distanceToClosestAtom)
					{
						distanceToClosestAtom = center.distance(atom);
						closestAtom = atom;
						closestAtomId=n;
					}
				}

				int closestImageId=-1;
				for(unsigned int iImageCharge=0; iImageCharge < d_imageAtomPositions.size(); ++iImageCharge)
				{
					Point<3> imageAtom(d_imageAtomPositions[iImageCharge][0],d_imageAtomPositions[iImageCharge][1],d_imageAtomPositions[iImageCharge][2]);
					if(center.distance(imageAtom) < distanceToClosestAtom)
					{
						distanceToClosestAtom = center.distance(imageAtom);
						closestAtom = imageAtom;
						closestImageId=iImageCharge;
					}
				}
				if (closestImageId!=-1)
					closestAtomId=d_imageIds[closestImageId];

				if (dftParameters::autoUserMeshParams  && !dftParameters::reproducible_output)
				{
					bool inOuterAtomBall = false;

					if(distanceToClosestAtom <= 
							(dftParameters::useMeshSizesFromAtomsFile?d_atomPositions[closestAtomId][6]:dftParameters::outerAtomBallRadius))
						inOuterAtomBall = true;

					if(inOuterAtomBall && (currentMeshSize > 
								(1.1*(dftParameters::useMeshSizesFromAtomsFile?d_atomPositions[closestAtomId][5]:dftParameters::meshSizeOuterBall))))
						cellRefineFlag = true;

					bool inInnerAtomBall = false;

					if(distanceToClosestAtom <= dftParameters::innerAtomBallRadius)
						inInnerAtomBall = true;

					if(inInnerAtomBall && currentMeshSize > 1.5*dftParameters::meshSizeInnerBall)
						cellRefineFlag = true;
				}
				else
				{
					bool inOuterAtomBall = false;

					if(distanceToClosestAtom <= 
							(dftParameters::useMeshSizesFromAtomsFile?d_atomPositions[closestAtomId][6]:dftParameters::outerAtomBallRadius))
						inOuterAtomBall = true;

					if(inOuterAtomBall && (currentMeshSize > 
								(dftParameters::useMeshSizesFromAtomsFile?d_atomPositions[closestAtomId][5]:dftParameters::meshSizeOuterBall)))
						cellRefineFlag = true;

					bool inInnerAtomBall = false;

					if(distanceToClosestAtom <= dftParameters::innerAtomBallRadius)
						inInnerAtomBall = true;

					if(inInnerAtomBall && currentMeshSize > dftParameters::meshSizeInnerBall)
						cellRefineFlag = true;
				}

				if (dftParameters::autoUserMeshParams  && !dftParameters::reproducible_output)
				{
					bool inBiggerAtomBall = false;

					if(distanceToClosestAtom <= 10.0)
						inBiggerAtomBall = true;

					if(inBiggerAtomBall && currentMeshSize > 6.0)
						cellRefineFlag = true;
				}

				MappingQ1<3,3> mapping;
				try
				{
					Point<3> p_cell = mapping.transform_real_to_unit_cell(cell,closestAtom);
					double dist = GeometryInfo<3>::distance_to_unit_cell(p_cell);

					if(dist < 1e-08 && currentMeshSize > (dftParameters::autoUserMeshParams?1.5:1)*dftParameters::meshSizeInnerBall)
						cellRefineFlag = true;

				}
				catch(MappingQ1<3>::ExcTransformationFailed)
				{
				}

				cellRefineFlag= Utilities::MPI::max((unsigned int) cellRefineFlag, interpoolcomm);
				cellRefineFlag= Utilities::MPI::max((unsigned int) cellRefineFlag, interBandGroupComm);

				//
				//set refine flags
				if(cellRefineFlag)
				{
					locallyOwnedCellsRefineFlags.push_back(1);
					cellIdToCellRefineFlagMapLocal[cell->id()]=1;
					cell->set_refine_flag();
					isAnyCellRefined=true;
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


		//
		// refine cells on periodic boundary if their length is greater than
		// mesh size around atom by a factor (set by smootheningFactor)
		//
		if (smoothenCellsOnPeriodicBoundary)
		{
			locallyOwnedCount=0;
			cell = parallelTriangulation.begin_active();
			endc = parallelTriangulation.end();

			if(generateElectrostaticsTria)
			{
				cellElectroRho = electrostaticsTriangulationRho.begin_active();
				cellElectroDisp = electrostaticsTriangulationDisp.begin_active();
				cellElectroForce = electrostaticsTriangulationForce.begin_active();
			}

			const unsigned int faces_per_cell=dealii::GeometryInfo<3>::faces_per_cell;

			for(;cell != endc; ++cell)
			{
				if (cell->is_locally_owned())
				{
					if(cell->at_boundary()
							&& cell->minimum_vertex_distance()>(dftParameters::autoUserMeshParams?1.5:1)*smootheningFactor*smallestMeshSizeAroundAtom
							&& !cell->refine_flag_set() )
						for(unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
							if (cell->has_periodic_neighbor(iFace))
							{
								cell->set_refine_flag();
								isAnyCellRefined=true;
								locallyOwnedCellsRefineFlags[cellIdToLocallyOwnedId[cell->id()]]=1;
								cellIdToCellRefineFlagMapLocal[cell->id()]=1;
								if(generateElectrostaticsTria)
								{
									cellElectroRho->set_refine_flag();
									cellElectroDisp->set_refine_flag();
									cellElectroForce->set_refine_flag();
								}
								break;
							}
					locallyOwnedCount++;
				}
				if(generateElectrostaticsTria)
				{
					++cellElectroRho;
					++cellElectroDisp;
					++cellElectroForce;
				}
			}
		}

		return isAnyCellRefined;
	}

	//
	// internal function which sets refinement flags to have consistent refinement across periodic boundary
	//
	bool triangulationManager::consistentPeriodicBoundaryRefinement(parallel::distributed::Triangulation<3>   & parallelTriangulation,
			parallel::distributed::Triangulation<3>   & electrostaticsTriangulationRho,
			parallel::distributed::Triangulation<3>   & electrostaticsTriangulationDisp,
			parallel::distributed::Triangulation<3>   & electrostaticsTriangulationForce,
			const bool                                  generateElectrostaticsTria,
			std::vector<unsigned int>                 & locallyOwnedCellsRefineFlags,
			std::map<dealii::CellId,unsigned int>     & cellIdToCellRefineFlagMapLocal)
	{
		locallyOwnedCellsRefineFlags.clear();
		cellIdToCellRefineFlagMapLocal.clear();
		typename parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc, cellElectroRho, cellElectroDisp, cellElectroForce;
		cell = parallelTriangulation.begin_active();
		endc = parallelTriangulation.end();

		//
		// populate maps refinement flag maps to zero values
		//
		std::map<dealii::CellId,unsigned int> cellIdToLocallyOwnedId;
		unsigned int locallyOwnedCount=0;
		for(;cell != endc; ++cell)
			if(cell->is_locally_owned())
			{
				cellIdToLocallyOwnedId[cell->id()]=locallyOwnedCount;
				locallyOwnedCellsRefineFlags.push_back(0);
				cellIdToCellRefineFlagMapLocal[cell->id()]=0;
				locallyOwnedCount++;
			}


		cell = parallelTriangulation.begin_active();
		endc = parallelTriangulation.end();

		if(generateElectrostaticsTria)
		{
			cellElectroRho = electrostaticsTriangulationRho.begin_active();
			cellElectroDisp = electrostaticsTriangulationDisp.begin_active();
			cellElectroForce = electrostaticsTriangulationForce.begin_active();
		}


		//
		// go to each locally owned or ghost cell which has a face on the periodic boundary->
		// query if cell has a periodic neighbour which is coarser -> if yes and the coarse
		// cell is locally owned set refinement flag on that cell
		//
		const unsigned int faces_per_cell=dealii::GeometryInfo<3>::faces_per_cell;
		bool isAnyCellRefined=false;
		for(;cell != endc; ++cell)
		{
			if((cell->is_locally_owned() || cell->is_ghost()) &&
					cell->at_boundary())
				for(unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
					if (cell->has_periodic_neighbor(iFace))
						if (cell->periodic_neighbor_is_coarser(iFace))
						{
							typename parallel::distributed::Triangulation<3>::active_cell_iterator periodicCell=cell->periodic_neighbor(iFace);

							if (periodicCell->is_locally_owned())
							{
								locallyOwnedCellsRefineFlags[cellIdToLocallyOwnedId[periodicCell->id()]]=1;
								cellIdToCellRefineFlagMapLocal[periodicCell->id()]=1;
								periodicCell->set_refine_flag();

								isAnyCellRefined=true;
								if(generateElectrostaticsTria)
								{
									cellElectroRho->periodic_neighbor(iFace)->set_refine_flag();
									cellElectroDisp->periodic_neighbor(iFace)->set_refine_flag();
									cellElectroForce->periodic_neighbor(iFace)->set_refine_flag();
								}
							}
						}
			if(generateElectrostaticsTria)
			{
				++cellElectroRho;
				++cellElectroDisp;
				++cellElectroForce;
			}
		}
		return isAnyCellRefined;
	}

	//
	// check that triangulation has consistent refinement across periodic boundary
	//
	bool triangulationManager::checkPeriodicSurfaceRefinementConsistency(parallel::distributed::Triangulation<3>& parallelTriangulation)
	{
		typename parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc;
		cell = parallelTriangulation.begin_active();
		endc = parallelTriangulation.end();

		const unsigned int faces_per_cell=dealii::GeometryInfo<3>::faces_per_cell;

		unsigned int notConsistent=0;
		for(;cell != endc; ++cell)
			if((cell->is_locally_owned() || cell->is_ghost()) &&
					cell->at_boundary())
				for(unsigned int iFace = 0; iFace < faces_per_cell; ++iFace)
					if (cell->has_periodic_neighbor(iFace))
					{
						typename parallel::distributed::Triangulation<3>::active_cell_iterator periodicCell=cell->periodic_neighbor(iFace);
						if (periodicCell->is_locally_owned() || cell->is_locally_owned())
							if (cell->periodic_neighbor_is_coarser(iFace) || periodicCell->has_children())
								notConsistent=1;

					}
		notConsistent=Utilities::MPI::max(notConsistent, mpi_communicator);
		return notConsistent==1?false:true;
	}


	//
	// check that FEOrder=1 dofHandler using the triangulation has parallel consistent
	// combined hanging node and periodic constraints
	//
	bool triangulationManager::checkConstraintsConsistency(parallel::distributed::Triangulation<3>& parallelTriangulation)
	{
		FESystem<3> FE(FE_Q<3>(QGaussLobatto<1>(d_FEOrder+1)), 1);
		//FESystem<3> FE(FE_Q<3>(QGaussLobatto<1>(1+1)), 1);
		DoFHandler<3> dofHandler;
		dofHandler.initialize(parallelTriangulation,FE);
		dofHandler.distribute_dofs(FE);
		IndexSet   locally_relevant_dofs;
		DoFTools::extract_locally_relevant_dofs(dofHandler, locally_relevant_dofs);

		ConstraintMatrix constraints;
		constraints.clear();
		constraints.reinit(locally_relevant_dofs);
		DoFTools::make_hanging_node_constraints(dofHandler, constraints);
		std::vector<GridTools::PeriodicFacePair<typename DoFHandler<3>::cell_iterator> > periodicity_vector;

		//create unitVectorsXYZ
		std::vector<std::vector<double> > unitVectorsXYZ;
		unitVectorsXYZ.resize(3);

		for(int i = 0; i < 3; ++i)
		{
			unitVectorsXYZ[i].resize(3,0.0);
			unitVectorsXYZ[i][i] = 0.0;
		}

		std::vector<Tensor<1,3> > offsetVectors;
		//resize offset vectors
		offsetVectors.resize(3);

		for(int i = 0; i < 3; ++i)
		{
			for(int j = 0; j < 3; ++j)
			{
				offsetVectors[i][j] = unitVectorsXYZ[i][j] - d_domainBoundingVectors[i][j];
			}
		}

		const std::array<int,3> periodic = {dftParameters::periodicX,
			dftParameters::periodicY,
			dftParameters::periodicZ};

		std::vector<int> periodicDirectionVector;
		for (unsigned int  d= 0; d < 3; ++d)
		{
			if (periodic[d]==1)
			{
				periodicDirectionVector.push_back(d);
			}
		}

		for (int i = 0; i < std::accumulate(periodic.begin(),periodic.end(),0); ++i)
		{
			GridTools::collect_periodic_faces(dofHandler, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ periodicDirectionVector[i], periodicity_vector,offsetVectors[periodicDirectionVector[i]]);
		}

		DoFTools::make_periodicity_constraints<DoFHandler<C_DIM> >(periodicity_vector, constraints);
		constraints.close();

		IndexSet locally_active_dofs_debug;
		DoFTools::extract_locally_active_dofs(dofHandler, locally_active_dofs_debug);

		const std::vector<IndexSet>& locally_owned_dofs_debug= dofHandler.locally_owned_dofs_per_processor();

		return constraints.is_consistent_in_parallel(locally_owned_dofs_debug,
				locally_active_dofs_debug,
				mpi_communicator,
				!dftParameters::reproducible_output);
	}

	//
	//generate mesh based on a-posteriori estimates
	//
	void triangulationManager::generateAutomaticMeshApriori(const DoFHandler<3> & dofHandler,
			parallel::distributed::Triangulation<3> & parallelTriangulation,
			const std::vector<distributedCPUVec<double>> & eigenVectorsArrayIn,
			const unsigned int FEOrder,
			const bool generateElectrostaticsTria)
	{

		double topfrac = dftParameters::topfrac;
		double bottomfrac = 0.0;

		//
		//create an array of pointers holding the eigenVectors on starting mesh
		//
		unsigned int numberWaveFunctionsEstimate = eigenVectorsArrayIn.size();
		std::vector<const distributedCPUVec<double>*> eigenVectorsArrayOfPtrsIn(numberWaveFunctionsEstimate);
		for(int iWave = 0; iWave < numberWaveFunctionsEstimate; ++iWave)
		{
			eigenVectorsArrayOfPtrsIn[iWave] = &eigenVectorsArrayIn[iWave];
		}

		//
		//create storage for storing errors in each cell
		//
		dealii::Vector<double> estimated_error_per_cell(parallelTriangulation.n_active_cells());
		std::vector<double> errorInEachCell;

		//
		//fill in the errors corresponding to each cell
		//

		internal::computeLocalFiniteElementError(dofHandler,
				eigenVectorsArrayOfPtrsIn,
				errorInEachCell,
				FEOrder);

		for(unsigned int i = 0; i < errorInEachCell.size(); ++i)
			estimated_error_per_cell(i) = errorInEachCell[i];

		//
		//print certain error metrics of each cell
		//
		if (dftParameters::verbosity>=4)
		{
			double maxErrorIndicator = *std::max_element(errorInEachCell.begin(),errorInEachCell.end());
			double globalMaxError = Utilities::MPI::max(maxErrorIndicator,mpi_communicator);
			double errorSum = std::accumulate(errorInEachCell.begin(),errorInEachCell.end(),0.0);
			double globalSum = Utilities::MPI::sum(errorSum,mpi_communicator);
			pcout<<" Sum Error of all Cells: "<<globalSum<<" Max Error of all Cells:" <<globalMaxError<<std::endl;
		}

		//
		//reset moved parallel triangulation to unmoved triangulation
		//
		resetMesh(d_parallelTriangulationUnmoved,
				parallelTriangulation);

		//
		//prepare all meshes for refinement using estimated errors in each cell
		//
		parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(parallelTriangulation,
				estimated_error_per_cell,
				topfrac,
				bottomfrac);

		parallelTriangulation.prepare_coarsening_and_refinement();
		parallelTriangulation.execute_coarsening_and_refinement();

		parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(d_parallelTriangulationUnmoved,
				estimated_error_per_cell,
				topfrac,
				bottomfrac);

		d_parallelTriangulationUnmoved.prepare_coarsening_and_refinement();
		d_parallelTriangulationUnmoved.execute_coarsening_and_refinement();

		std::string printCommand = "Automatic Adaptive Mesh Refinement Based Triangulation Summary";
		internal::computeMeshMetrics(parallelTriangulation,
				printCommand,
				pcout,
				mpi_communicator,
				interpoolcomm,
				interBandGroupComm);

		if(generateElectrostaticsTria)
		{
			parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(d_triangulationElectrostaticsRho,
					estimated_error_per_cell,
					topfrac,
					bottomfrac);

			d_triangulationElectrostaticsRho.prepare_coarsening_and_refinement();
			d_triangulationElectrostaticsRho.execute_coarsening_and_refinement();


			parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(d_triangulationElectrostaticsDisp,
					estimated_error_per_cell,
					topfrac,
					bottomfrac);

			d_triangulationElectrostaticsDisp.prepare_coarsening_and_refinement();
			d_triangulationElectrostaticsDisp.execute_coarsening_and_refinement();


			parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(d_triangulationElectrostaticsForce,
					estimated_error_per_cell,
					topfrac,
					bottomfrac);

			d_triangulationElectrostaticsForce.prepare_coarsening_and_refinement();
			d_triangulationElectrostaticsForce.execute_coarsening_and_refinement();

			std::string printCommand1 = "A-posteriori Electrostatics Rho Triangulation Summary";
			internal::computeMeshMetrics(d_triangulationElectrostaticsRho,
					printCommand1,
					pcout,
					mpi_communicator,
					interpoolcomm,
					interBandGroupComm);

			std::string printCommand2 = "A-posteriori Electrostatics Disp Triangulation Summary";
			internal::computeMeshMetrics(d_triangulationElectrostaticsDisp,
					printCommand2,
					pcout,
					mpi_communicator,
					interpoolcomm,
					interBandGroupComm);


			std::string printCommand3 = "A-posteriori Electrostatics Force Triangulation Summary";
			internal::computeMeshMetrics(d_triangulationElectrostaticsForce,
					printCommand3,
					pcout,
					mpi_communicator,
					interpoolcomm,
					interBandGroupComm);
		}

	}


	void triangulationManager::generateMesh
		(parallel::distributed::Triangulation<3> & parallelTriangulation,
		 parallel::distributed::Triangulation<3> & serialTriangulation,
		 parallel::distributed::Triangulation<3> & serialTriangulationElectrostatics,
		 parallel::distributed::Triangulation<3> & electrostaticsTriangulationRho,
		 parallel::distributed::Triangulation<3> & electrostaticsTriangulationDisp,
		 parallel::distributed::Triangulation<3> & electrostaticsTriangulationForce,
		 const bool generateElectrostaticsTria,
		 const bool generateSerialTria)
		{
			if(!dftParameters::meshFileName.empty())
			{
				GridIn<3> gridinParallel, gridinSerial;
				gridinParallel.attach_triangulation(parallelTriangulation);
				if (generateSerialTria)
					gridinSerial.attach_triangulation(serialTriangulation);

				//
				//Read mesh in UCD format generated from Cubit
				//
				std::ifstream f1(dftParameters::meshFileName.c_str());
				std::ifstream f2(dftParameters::meshFileName.c_str());
				gridinParallel.read_ucd(f1);
				if (generateSerialTria)
					gridinSerial.read_ucd(f2);

				meshGenUtils::markPeriodicFacesNonOrthogonal(parallelTriangulation,d_domainBoundingVectors);
				if (generateSerialTria)
					meshGenUtils::markPeriodicFacesNonOrthogonal(serialTriangulation,d_domainBoundingVectors);
			}
			else
			{

				generateCoarseMesh(parallelTriangulation);
				if (generateSerialTria)
				{
					generateCoarseMesh(serialTriangulation);
					AssertThrow(parallelTriangulation.n_global_active_cells()==serialTriangulation.n_global_active_cells(),ExcMessage("Number of coarse mesh cells are different in serial and parallel triangulations."));
				}

				if(generateElectrostaticsTria)
				{
					generateCoarseMesh(electrostaticsTriangulationRho);
					generateCoarseMesh(electrostaticsTriangulationDisp);
					generateCoarseMesh(electrostaticsTriangulationForce);
					AssertThrow(parallelTriangulation.n_global_active_cells()==electrostaticsTriangulationRho.n_global_active_cells(),ExcMessage("Number of coarse mesh cells are different in electrostatics triangulations having rho field."));
					AssertThrow(parallelTriangulation.n_global_active_cells()==electrostaticsTriangulationDisp.n_global_active_cells(),ExcMessage("Number of coarse mesh cells are different in electrostatics triangulations disp field."));
					AssertThrow(parallelTriangulation.n_global_active_cells()==electrostaticsTriangulationForce.n_global_active_cells(),ExcMessage("Number of coarse mesh cells are different in electrostatics triangulations for force computation."));

					if (generateSerialTria)
					{
						generateCoarseMesh(serialTriangulationElectrostatics);
						AssertThrow(parallelTriangulation.n_global_active_cells()==serialTriangulationElectrostatics.n_global_active_cells(),ExcMessage("Number of coarse mesh cells are different in electrostatics serial triangulation computation."));
					}
				}

				d_parallelTriaCurrentRefinement.clear();
				if (generateSerialTria)
					d_serialTriaCurrentRefinement.clear();

				//
				//Multilayer refinement. Refinement algorithm is progressively modified
				//if check of parallel consistency of combined periodic
				//and hanging node constraitns fails (Related to https://github.com/dealii/dealii/issues/7053).
				//

				//
				//STAGE0: Call only refinementAlgorithmA. Multilevel refinement is performed until
				//refinementAlgorithmA does not set refinement flags on any cell.
				//
				unsigned int numLevels=0;
				bool refineFlag = true;
				while(refineFlag)
				{
					refineFlag = false;
					std::vector<unsigned int> locallyOwnedCellsRefineFlags;
					std::map<dealii::CellId,unsigned int> cellIdToCellRefineFlagMapLocal;

					refineFlag=refinementAlgorithmA(parallelTriangulation,
							electrostaticsTriangulationRho,
							electrostaticsTriangulationDisp,
							electrostaticsTriangulationForce,
							generateElectrostaticsTria,
							locallyOwnedCellsRefineFlags,
							cellIdToCellRefineFlagMapLocal);

					//This sets the global refinement sweep flag
					refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);

					//Refine
					if (refineFlag)
					{

						if(numLevels<d_max_refinement_steps)
						{
							if (dftParameters::verbosity>=4)
								pcout<< "refinement in progress, level: "<< numLevels<<std::endl;

							if (generateSerialTria)
							{
								d_serialTriaCurrentRefinement.push_back(std::vector<bool>());

								//First refine serial mesh
								refineSerialMesh(cellIdToCellRefineFlagMapLocal,
										mpi_communicator,
										serialTriangulation,
										parallelTriangulation,
										d_serialTriaCurrentRefinement[numLevels]) ;


								if(generateElectrostaticsTria)
									refineSerialMesh(cellIdToCellRefineFlagMapLocal,
											mpi_communicator,
											serialTriangulationElectrostatics,
											parallelTriangulation,
											d_serialTriaCurrentRefinement[numLevels]);
							}

							d_parallelTriaCurrentRefinement.push_back(std::vector<bool>());
							parallelTriangulation.save_refine_flags(d_parallelTriaCurrentRefinement[numLevels]);

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
				//STAGE1: This stage is only activated if combined periodic and hanging node constraints are
				//not consistent in parallel.
				//Call refinementAlgorithmA and consistentPeriodicBoundaryRefinement alternatively.
				//In the call to refinementAlgorithmA there is no additional reduction of adaptivity performed
				//on the periodic boundary. Multilevel refinement is performed until both refinementAlgorithmA
				//and consistentPeriodicBoundaryRefinement do not set refinement flags on any cell.
				//
				if (!dftParameters::reproducible_output)
				{
					if (!checkConstraintsConsistency(parallelTriangulation))
					{
						refineFlag=true;
						while(refineFlag)
						{
							refineFlag = false;
							std::vector<unsigned int> locallyOwnedCellsRefineFlags;
							std::map<dealii::CellId,unsigned int> cellIdToCellRefineFlagMapLocal;
							if (numLevels%2==0)
							{
								refineFlag=refinementAlgorithmA(parallelTriangulation,
										electrostaticsTriangulationRho,
										electrostaticsTriangulationDisp,
										electrostaticsTriangulationForce,
										generateElectrostaticsTria,
										locallyOwnedCellsRefineFlags,
										cellIdToCellRefineFlagMapLocal);

								//This sets the global refinement sweep flag
								refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);

								//try the other type of refinement to prevent while loop from ending prematurely
								if (!refineFlag)
								{
									//call refinement algorithm  which sets refinement flags such as to
									//create consistent refinement across periodic boundary
									refineFlag=consistentPeriodicBoundaryRefinement(parallelTriangulation,
											electrostaticsTriangulationRho,
											electrostaticsTriangulationDisp,
											electrostaticsTriangulationForce,
											generateElectrostaticsTria,
											locallyOwnedCellsRefineFlags,
											cellIdToCellRefineFlagMapLocal);

									//This sets the global refinement sweep flag
									refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);
								}
							}
							else
							{
								//call refinement algorithm  which sets refinement flags such as to
								//create consistent refinement across periodic boundary
								refineFlag=consistentPeriodicBoundaryRefinement(parallelTriangulation,
										electrostaticsTriangulationRho,
										electrostaticsTriangulationDisp,
										electrostaticsTriangulationForce,
										generateElectrostaticsTria,
										locallyOwnedCellsRefineFlags,
										cellIdToCellRefineFlagMapLocal);

								//This sets the global refinement sweep flag
								refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);

								//try the other type of refinement to prevent while loop from ending prematurely
								if (!refineFlag)
								{
									refineFlag=refinementAlgorithmA(parallelTriangulation,
											electrostaticsTriangulationRho,
											electrostaticsTriangulationDisp,
											electrostaticsTriangulationForce,
											generateElectrostaticsTria,
											locallyOwnedCellsRefineFlags,
											cellIdToCellRefineFlagMapLocal);

									//This sets the global refinement sweep flag
									refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);
								}
							}

							//Refine
							if (refineFlag)
							{

								if(numLevels<d_max_refinement_steps)
								{
									if (dftParameters::verbosity>=4)
										pcout<< "refinement in progress, level: "<< numLevels<<std::endl;

									if (generateSerialTria)
									{
										d_serialTriaCurrentRefinement.push_back(std::vector<bool>());

										//First refine serial mesh
										refineSerialMesh(cellIdToCellRefineFlagMapLocal,
												mpi_communicator,
												serialTriangulation,
												parallelTriangulation,
												d_serialTriaCurrentRefinement[numLevels]) ;


										if(generateElectrostaticsTria)
											refineSerialMesh(cellIdToCellRefineFlagMapLocal,
													mpi_communicator,
													serialTriangulationElectrostatics,
													parallelTriangulation,
													d_serialTriaCurrentRefinement[numLevels]);
									}

									d_parallelTriaCurrentRefinement.push_back(std::vector<bool>());
									parallelTriangulation.save_refine_flags(d_parallelTriaCurrentRefinement[numLevels]);

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
					}
				}

				if (!dftParameters::reproducible_output)
				{
					//
					//STAGE2: This stage is only activated if combined periodic and hanging node constraints are
					//still not consistent in parallel.
					//Call refinementAlgorithmA and consistentPeriodicBoundaryRefinement alternatively.
					//In the call to refinementAlgorithmA there is an additional reduction of adaptivity performed
					//on the periodic boundary such that the maximum cell length on the periodic boundary is less
					//than two times the MESH SIZE AROUND ATOM. Multilevel refinement is performed until both
					//refinementAlgorithmAand consistentPeriodicBoundaryRefinement do not set refinement flags
					//on any cell.
					//
					if (!checkConstraintsConsistency(parallelTriangulation))
					{
						refineFlag=true;
						while (refineFlag)
						{
							refineFlag = false;
							std::vector<unsigned int> locallyOwnedCellsRefineFlags;
							std::map<dealii::CellId,unsigned int> cellIdToCellRefineFlagMapLocal;
							if (numLevels%2==0)
							{
								refineFlag=refinementAlgorithmA(parallelTriangulation,
										electrostaticsTriangulationRho,
										electrostaticsTriangulationDisp,
										electrostaticsTriangulationForce,
										generateElectrostaticsTria,
										locallyOwnedCellsRefineFlags,
										cellIdToCellRefineFlagMapLocal,
										true,
										2.0);

								//This sets the global refinement sweep flag
								refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);

								//try the other type of refinement to prevent while loop from ending prematurely
								if (!refineFlag)
								{
									//call refinement algorithm  which sets refinement flags such as to
									//create consistent refinement across periodic boundary
									refineFlag=consistentPeriodicBoundaryRefinement(parallelTriangulation,
											electrostaticsTriangulationRho,
											electrostaticsTriangulationDisp,
											electrostaticsTriangulationForce,
											generateElectrostaticsTria,
											locallyOwnedCellsRefineFlags,
											cellIdToCellRefineFlagMapLocal);

									//This sets the global refinement sweep flag
									refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);
								}
							}
							else
							{
								//call refinement algorithm  which sets refinement flags such as to
								//create consistent refinement across periodic boundary
								refineFlag=consistentPeriodicBoundaryRefinement(parallelTriangulation,
										electrostaticsTriangulationRho,
										electrostaticsTriangulationDisp,
										electrostaticsTriangulationForce,
										generateElectrostaticsTria,
										locallyOwnedCellsRefineFlags,
										cellIdToCellRefineFlagMapLocal);

								//This sets the global refinement sweep flag
								refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);

								//try the other type of refinement to prevent while loop from ending prematurely
								if (!refineFlag)
								{
									refineFlag=refinementAlgorithmA(parallelTriangulation,
											electrostaticsTriangulationRho,
											electrostaticsTriangulationDisp,
											electrostaticsTriangulationForce,
											generateElectrostaticsTria,
											locallyOwnedCellsRefineFlags,
											cellIdToCellRefineFlagMapLocal,
											true,
											2.0);

									//This sets the global refinement sweep flag
									refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);
								}
							}

							//Refine
							if (refineFlag)
							{

								if(numLevels<d_max_refinement_steps)
								{
									if (dftParameters::verbosity>=4)
										pcout<< "refinement in progress, level: "<< numLevels<<std::endl;

									if (generateSerialTria)
									{
										d_serialTriaCurrentRefinement.push_back(std::vector<bool>());

										//First refine serial mesh
										refineSerialMesh(cellIdToCellRefineFlagMapLocal,
												mpi_communicator,
												serialTriangulation,
												parallelTriangulation,
												d_serialTriaCurrentRefinement[numLevels]) ;


										if(generateElectrostaticsTria)
											refineSerialMesh(cellIdToCellRefineFlagMapLocal,
													mpi_communicator,
													serialTriangulationElectrostatics,
													parallelTriangulation,
													d_serialTriaCurrentRefinement[numLevels]);
									}

									d_parallelTriaCurrentRefinement.push_back(std::vector<bool>());
									parallelTriangulation.save_refine_flags(d_parallelTriaCurrentRefinement[numLevels]);

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
					}

					//
					//STAGE3: This stage is only activated if combined periodic and hanging node constraints are
					//still not consistent in parallel.
					//Call refinementAlgorithmA and consistentPeriodicBoundaryRefinement alternatively.
					//In the call to refinementAlgorithmA there is an additional reduction of adaptivity performed
					//on the periodic boundary such that the maximum cell length on the periodic boundary is less
					//than MESH SIZE AROUND ATOM essentially ensuring uniform refinement on the periodic boundary
					//in the case of MESH SIZE AROUND ATOM being same as MESH SIZE AT ATOM.
					//Multilevel refinement is performed until both refinementAlgorithmA and
					//consistentPeriodicBoundaryRefinement do not set refinement flags on any cell.
					//
					if (!checkConstraintsConsistency(parallelTriangulation))
					{
						refineFlag=true;
						while (refineFlag)
						{
							refineFlag = false;
							std::vector<unsigned int> locallyOwnedCellsRefineFlags;
							std::map<dealii::CellId,unsigned int> cellIdToCellRefineFlagMapLocal;
							if (numLevels%2==0)
							{
								refineFlag=refinementAlgorithmA(parallelTriangulation,
										electrostaticsTriangulationRho,
										electrostaticsTriangulationDisp,
										electrostaticsTriangulationForce,
										generateElectrostaticsTria,
										locallyOwnedCellsRefineFlags,
										cellIdToCellRefineFlagMapLocal,
										true,
										1.0);

								//This sets the global refinement sweep flag
								refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);

								//try the other type of refinement to prevent while loop from ending prematurely
								if (!refineFlag)
								{
									//call refinement algorithm  which sets refinement flags such as to
									//create consistent refinement across periodic boundary
									refineFlag=consistentPeriodicBoundaryRefinement(parallelTriangulation,
											electrostaticsTriangulationRho,
											electrostaticsTriangulationDisp,
											electrostaticsTriangulationForce,
											generateElectrostaticsTria,
											locallyOwnedCellsRefineFlags,
											cellIdToCellRefineFlagMapLocal);

									//This sets the global refinement sweep flag
									refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);
								}
							}
							else
							{

								//call refinement algorithm  which sets refinement flags such as to
								//create consistent refinement across periodic boundary
								refineFlag=consistentPeriodicBoundaryRefinement(parallelTriangulation,
										electrostaticsTriangulationRho,
										electrostaticsTriangulationDisp,
										electrostaticsTriangulationForce,
										generateElectrostaticsTria,
										locallyOwnedCellsRefineFlags,
										cellIdToCellRefineFlagMapLocal);

								//This sets the global refinement sweep flag
								refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);

								//try the other type of refinement to prevent while loop from ending prematurely
								if (!refineFlag)
								{
									refineFlag=refinementAlgorithmA(parallelTriangulation,
											electrostaticsTriangulationRho,
											electrostaticsTriangulationDisp,
											electrostaticsTriangulationForce,
											generateElectrostaticsTria,
											locallyOwnedCellsRefineFlags,
											cellIdToCellRefineFlagMapLocal,
											true,
											1.0);

									//This sets the global refinement sweep flag
									refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);
								}
							}

							//Refine
							if (refineFlag)
							{

								if(numLevels<d_max_refinement_steps)
								{
									if (dftParameters::verbosity>=4)
										pcout<< "refinement in progress, level: "<< numLevels<<std::endl;

									if (generateSerialTria)
									{
										d_serialTriaCurrentRefinement.push_back(std::vector<bool>());

										//First refine serial mesh
										refineSerialMesh(cellIdToCellRefineFlagMapLocal,
												mpi_communicator,
												serialTriangulation,
												parallelTriangulation,
												d_serialTriaCurrentRefinement[numLevels]) ;


										if(generateElectrostaticsTria)
											refineSerialMesh(cellIdToCellRefineFlagMapLocal,
													mpi_communicator,
													serialTriangulationElectrostatics,
													parallelTriangulation,
													d_serialTriaCurrentRefinement[numLevels]);
									}

									d_parallelTriaCurrentRefinement.push_back(std::vector<bool>());
									parallelTriangulation.save_refine_flags(d_parallelTriaCurrentRefinement[numLevels]);

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
					}

					if (checkConstraintsConsistency(parallelTriangulation))
					{
						if (dftParameters::verbosity>=4)
							pcout<< "Hanging node and periodic constraints parallel consistency achieved."<<std::endl;

						dftParameters::createConstraintsFromSerialDofhandler=false;
						dftParameters::constraintsParallelCheck=false;
					}
					else
					{
						if (dftParameters::verbosity>=4)
							pcout<< "Hanging node and periodic constraints parallel consistency not achieved."<<std::endl;

						dftParameters::createConstraintsFromSerialDofhandler=true;
					}
				}

				//
				//compute some adaptive mesh metrics
				//
				double minElemLength = dftParameters::meshSizeOuterDomain;
				double maxElemLength = 0.0;
				typename parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc, cellDisp, cellForce;
				cell = parallelTriangulation.begin_active();
				endc = parallelTriangulation.end();
				unsigned int numLocallyOwnedCells=0;
				for( ; cell != endc; ++cell)
				{
					if(cell->is_locally_owned())
					{
						numLocallyOwnedCells++;
						if(cell->minimum_vertex_distance() < minElemLength)
							minElemLength = cell->minimum_vertex_distance();

						if(cell->minimum_vertex_distance() > maxElemLength)
							maxElemLength = cell->minimum_vertex_distance();
					}
				}

				minElemLength = Utilities::MPI::min(minElemLength, mpi_communicator);
				maxElemLength = Utilities::MPI::max(maxElemLength, mpi_communicator);

				//
				//print out adaptive mesh metrics and check mesh generation synchronization across pools
				//
				if (dftParameters::verbosity>=4)
				{
					pcout<< "Triangulation generation summary: "<<std::endl<<" num elements: "<<parallelTriangulation.n_global_active_cells()<<", num refinement levels: "<<numLevels<<", min element length: "<<minElemLength<<", max element length: "<<maxElemLength<<std::endl;
				}

				internal::checkTriangulationEqualityAcrossProcessorPools(parallelTriangulation,
						numLocallyOwnedCells,
						interpoolcomm);
				internal::checkTriangulationEqualityAcrossProcessorPools(parallelTriangulation,
						numLocallyOwnedCells,
						interBandGroupComm);

				if (generateSerialTria)
				{
					const unsigned int numberGlobalCellsParallel = parallelTriangulation.n_global_active_cells();
					const unsigned int numberGlobalCellsSerial = serialTriangulation.n_global_active_cells();

					if (dftParameters::verbosity>=4)
						pcout<<" numParallelCells: "<< numberGlobalCellsParallel<<", numSerialCells: "<< numberGlobalCellsSerial<<std::endl;

					AssertThrow(numberGlobalCellsParallel==numberGlobalCellsSerial,ExcMessage("Number of cells are different for parallel and serial triangulations"));
				}
				else
				{
					const unsigned int numberGlobalCellsParallel = parallelTriangulation.n_global_active_cells();

					if (dftParameters::verbosity>=4)
						pcout<<" numParallelCells: "<< numberGlobalCellsParallel<<std::endl;
				}


				if(generateElectrostaticsTria)
				{
					numLocallyOwnedCells = 0;
					double minElemLengthRho = dftParameters::meshSizeOuterDomain;
					double minElemLengthDisp = dftParameters::meshSizeOuterDomain;
					double minElemLengthForce = dftParameters::meshSizeOuterDomain;

					cell = electrostaticsTriangulationRho.begin_active();
					endc = electrostaticsTriangulationRho.end();
					cellDisp = electrostaticsTriangulationDisp.begin_active();
					cellForce = electrostaticsTriangulationForce.begin_active();
					for( ; cell != endc; ++cell)
					{
						if(cell->is_locally_owned())
						{
							numLocallyOwnedCells++;
							if(cell->minimum_vertex_distance() < minElemLengthRho)
								minElemLengthRho = cell->minimum_vertex_distance();
							if(cellDisp->minimum_vertex_distance() < minElemLengthDisp)
								minElemLengthDisp = cellDisp->minimum_vertex_distance();
							if(cellForce->minimum_vertex_distance() < minElemLengthForce)
								minElemLengthForce = cellForce->minimum_vertex_distance();
						}
						++cellDisp;
						++cellForce;
					}

					minElemLengthRho = Utilities::MPI::min(minElemLengthRho, mpi_communicator);
					minElemLengthDisp = Utilities::MPI::min(minElemLengthDisp, mpi_communicator);
					minElemLengthForce = Utilities::MPI::min(minElemLengthForce, mpi_communicator);
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
		 parallel::distributed::Triangulation<3>& serialTriangulation,
		 const parallel::distributed::Triangulation<3>& parallelTriangulation,
		 std::vector<bool> & serialTriaCurrentRefinement)

		{
			const unsigned int numberGlobalCellsSerial = serialTriangulation.n_global_active_cells();
			std::vector<unsigned int> refineFlagsSerialCells(numberGlobalCellsSerial,0);

			dealii::BoundingBox<3> boundingBoxParallelTria=dealii::GridTools::compute_bounding_box(parallelTriangulation);

			unsigned int count=0;
			for(auto cell : serialTriangulation.active_cell_iterators())
				if(cell->is_locally_owned())
				{
					if(boundingBoxParallelTria.point_inside(cell->center()))
					{
						std::map<dealii::CellId,unsigned int>::const_iterator
							iter=cellIdToCellRefineFlagMapLocal.find(cell->id());
						if (iter!=cellIdToCellRefineFlagMapLocal.end())
							refineFlagsSerialCells[count]=iter->second;
					}
					count++;
				}

			MPI_Allreduce(MPI_IN_PLACE,
					&refineFlagsSerialCells[0],
					numberGlobalCellsSerial,
					MPI_UNSIGNED,
					MPI_SUM,
					mpi_comm);

			count=0;
			for(auto cell : serialTriangulation.active_cell_iterators())
				if(cell->is_locally_owned())
				{
					if (refineFlagsSerialCells[count]==1)
						cell->set_refine_flag();
					count++;
				}

			serialTriangulation.save_refine_flags(serialTriaCurrentRefinement);
			serialTriangulation.execute_coarsening_and_refinement();
		}


}
