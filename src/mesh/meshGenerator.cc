// ---------------------------------------------------------------------
//
// Copyright (c) 2017 The Regents of the University of Michigan and DFT-FE authors.
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
/** @file meshGenerator.cc
 *
 *  @brief Source file for meshGenerator.h
 *
 *  @author Phani Motamarri, Sambit Das, Krishnendu Ghosh
 */


#include <meshGenerator.h>
#include <dftParameters.h>
#include "meshGenUtils.cc"


//
//constructor
//
meshGeneratorClass::meshGeneratorClass(const MPI_Comm &mpi_comm_replica):
  d_parallelTriangulationUnmoved(mpi_comm_replica),
  d_parallelTriangulationUnmovedPrevious(mpi_comm_replica),
  d_parallelTriangulationMoved(mpi_comm_replica),
  mpi_communicator (mpi_comm_replica),
  d_serialTriangulationUnmoved(MPI_COMM_SELF),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::never, TimerOutput::wall_times)
{

}

//
//destructor
//
meshGeneratorClass::~meshGeneratorClass()
{

}

//
//generate adaptive mesh
//

void meshGeneratorClass::generateMesh(parallel::distributed::Triangulation<3>& parallelTriangulation)
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



      for (unsigned int i=0; i<3;i++){

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
#ifdef ENABLE_PERIODIC_BC
      meshGenUtils::markPeriodicFacesNonOrthogonal(parallelTriangulation,d_domainBoundingVectors);
#endif

      if (dftParameters::verbosity>=1)
      {
        pcout<<std::endl<< " Base uniform number of elements: "<< parallelTriangulation.n_global_active_cells();
      }

      //
      //Multilayer refinement
      //
      dealii::Point<3> origin;
      unsigned int numLevels=0;
      bool refineFlag = true;

      typename parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc;

      while(refineFlag)
	{
	  refineFlag = false;
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
		      refineFlag=true; //This sets the global refinement sweep flag
		      cell->set_refine_flag();
		    }

		}


	  //Refine
	  refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);

	  if (refineFlag)
	    {
	      if(numLevels<dftParameters::n_refinement_steps)
		{
		  if (dftParameters::verbosity>=1)
		  {
		     pcout<< "refinement in progress, level: ", numLevels<<std::endl;
		   }
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
      cell = parallelTriangulation.begin_active();
      endc = parallelTriangulation.end();
      for( ; cell != endc; ++cell)
	{
	  if(cell->is_locally_owned())
	    {
	      if(cell->minimum_vertex_distance() < minElemLength) minElemLength = cell->minimum_vertex_distance();
	    }
	}

      minElemLength = Utilities::MPI::min(minElemLength, mpi_communicator);

      //
      //print out adaptive mesh metrics
      //
      if (dftParameters::verbosity>=1)
      {
        pcout<< std::endl<<"Adaptivity summary: "<<std::endl<<" numCells: "<<parallelTriangulation.n_global_active_cells()<<", num refinement levels: "<<numLevels<<", h_min: "<<minElemLength<<std::endl;
      }
}

void meshGeneratorClass::generateMesh(parallel::distributed::Triangulation<3>& parallelTriangulation,
	                              parallel::distributed::Triangulation<3>& serialTriangulation)
{
  if(!dftParameters::meshFileName.empty())
    {
      GridIn<3> gridinParallel, gridinSerial;
      gridinParallel.attach_triangulation(parallelTriangulation);
      if (dftParameters::useSymm)
	gridinSerial.attach_triangulation(serialTriangulation);

      //
      //Read mesh in UCD format generated from Cubit
      //
      std::ifstream f1(dftParameters::meshFileName.c_str());
      std::ifstream f2(dftParameters::meshFileName.c_str());
      gridinParallel.read_ucd(f1);
      gridinSerial.read_ucd(f2);

#ifdef ENABLE_PERIODIC_BC
      meshGenUtils::markPeriodicFacesNonOrthogonal(parallelTriangulation,d_domainBoundingVectors);
      if (dftParameters::useSymm)
	meshGenUtils::markPeriodicFacesNonOrthogonal(serialTriangulation,d_domainBoundingVectors);
#endif
    }
  else
    {

      //
      //compute magnitudes of domainBounding Vectors
      //
      const double domainBoundingVectorMag1 = sqrt(d_domainBoundingVectors[0][0]*d_domainBoundingVectors[0][0] + d_domainBoundingVectors[0][1]*d_domainBoundingVectors[0][1] +  d_domainBoundingVectors[0][2]*d_domainBoundingVectors[0][2]);
      const double domainBoundingVectorMag2 = sqrt(d_domainBoundingVectors[1][0]*d_domainBoundingVectors[1][0] + d_domainBoundingVectors[1][1]*d_domainBoundingVectors[1][1] +  d_domainBoundingVectors[1][2]*d_domainBoundingVectors[1][2]);
      const double domainBoundingVectorMag3 = sqrt(d_domainBoundingVectors[2][0]*d_domainBoundingVectors[2][0] + d_domainBoundingVectors[2][1]*d_domainBoundingVectors[2][1] +  d_domainBoundingVectors[2][2]*d_domainBoundingVectors[2][2]);

      //
      //generate base level refinement
      //
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



      for (unsigned int i=0; i<3;i++){

        const double temp = numberIntervalsEachDirection[i]-std::floor(numberIntervalsEachDirection[i]);
        if(temp >= 0.5)
	  subdivisions[i] = std::ceil(numberIntervalsEachDirection[i]);
        else
	  subdivisions[i] = std::floor(numberIntervalsEachDirection[i]);
      }


      GridGenerator::subdivided_parallelepiped<3>(parallelTriangulation,
	                                          subdivisions,
				                  basisVectors);
      if (dftParameters::useSymm) {
	GridGenerator::subdivided_parallelepiped<3>(serialTriangulation,
						    subdivisions,
						    basisVectors);
      }

      //
      //Translate the main grid so that midpoint is at center
      //
      const Point<3> translation = 0.5*(vector1+vector2+vector3);
      GridTools::shift(-translation,parallelTriangulation);
      if (dftParameters::useSymm)
	GridTools::shift(-translation,serialTriangulation);

      //
      //collect periodic faces of the first level mesh to set up periodic boundary conditions later
      //
#ifdef ENABLE_PERIODIC_BC
      meshGenUtils::markPeriodicFacesNonOrthogonal(parallelTriangulation,d_domainBoundingVectors);
      if (dftParameters::useSymm)
	meshGenUtils::markPeriodicFacesNonOrthogonal(serialTriangulation,d_domainBoundingVectors);
#endif

      if (dftParameters::verbosity>=1)
      {
        pcout<<std::endl<< " Base uniform number of elements: "<< parallelTriangulation.n_global_active_cells();
      }

      //
      //Multilayer refinement
      //
      dealii::Point<3> origin;
      unsigned int numLevels=0;
      bool refineFlag = true;

      std::vector<double> centroid ;
      std::vector<int> localRefineFlag ;
      unsigned int n_cell ;

      typename parallel::distributed::Triangulation<3>::active_cell_iterator cell, endc;

      while(refineFlag)
	{
          n_cell = 0; centroid.clear(); localRefineFlag.clear();
	  refineFlag = false;
	  cell = parallelTriangulation.begin_active();
	  endc = parallelTriangulation.end();
	  //
	  for(;cell != endc; ++cell)
	      if(cell->is_locally_owned())
		{
		  dealii::Point<3> center(cell->center());
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
		  n_cell++ ;
	          Point<3> currentCentroid=cell->center();
                  centroid.push_back(currentCentroid[0]) ;
		  centroid.push_back(currentCentroid[1]) ;
		  centroid.push_back(currentCentroid[2]) ;
		  //
		  if (cellRefineFlag)
                    localRefineFlag.push_back (1) ;
		  else
		    localRefineFlag.push_back (0) ;
		  //
		  //set refine flags
		  if(cellRefineFlag)
		    {
		      refineFlag=true; //This sets the global refinement sweep flag
		      cell->set_refine_flag();
		    }

		}

	  //Refine
	  refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);

	  if (refineFlag)
	    {
	      if(numLevels<dftParameters::n_refinement_steps)
		{
		  if (dftParameters::verbosity>=1)
		  {
		     pcout<< "refinement in progress, level: ", numLevels<<std::endl;
		   }
		  parallelTriangulation.execute_coarsening_and_refinement();
		  numLevels++;
		}
	      else
		{
		  refineFlag=false;
		}
	    }

	  // Refine serial mesh
	  if(dftParameters::useSymm)
	    {
	      refineSerialMesh(n_cell, centroid, localRefineFlag, parallelTriangulation.n_global_active_cells(), serialTriangulation) ;
	      serialTriangulation.execute_coarsening_and_refinement();
	    }

        }
      //
      //compute some adaptive mesh metrics
      //
      double minElemLength = dftParameters::meshSizeOuterDomain;
      cell = parallelTriangulation.begin_active();
      endc = parallelTriangulation.end();
      for( ; cell != endc; ++cell)
	{
	  if(cell->is_locally_owned())
	    {
	      if(cell->minimum_vertex_distance() < minElemLength) minElemLength = cell->minimum_vertex_distance();
	    }
	}

      minElemLength = Utilities::MPI::min(minElemLength, mpi_communicator);

      //
      //print out adaptive mesh metrics
      //
      if (dftParameters::verbosity>=1)
      {
        pcout<< std::endl<<"Adaptivity summary: "<<std::endl<<" numCells: "<<parallelTriangulation.n_global_active_cells()<<", num refinement levels: "<<numLevels<<", h_min: "<<minElemLength<<std::endl;
      }

      const unsigned int numberGlobalCellsParallel = parallelTriangulation.n_global_active_cells();
      if (dftParameters::useSymm)
      {
         const unsigned int numberGlobalCellsSerial = serialTriangulation.n_global_active_cells();

	if (dftParameters::verbosity==2)
	{
	   pcout<<" numParallelCells: "<< numberGlobalCellsParallel<<", numSerialCells: "<< numberGlobalCellsSerial<<std::endl;
	}

	AssertThrow(numberGlobalCellsParallel==numberGlobalCellsSerial,ExcMessage("Number of cells are different for parallel and serial triangulations"));
      }

    }
}

//
//generate Mesh
//
void meshGeneratorClass::generateSerialUnmovedAndParallelMovedUnmovedMesh
                    (const std::vector<std::vector<double> > & atomLocations,
		     const std::vector<std::vector<double> > & imageAtomLocations,
		     const std::vector<std::vector<double> > & domainBoundingVectors)
{

  //
  //set the data members before generating mesh
  //
  d_atomPositions = atomLocations;
  d_imageAtomPositions = imageAtomLocations;
  d_domainBoundingVectors = domainBoundingVectors;

  //clear existing triangulation data
  d_serialTriangulationUnmoved.clear();
  d_parallelTriangulationUnmoved.clear();
  d_parallelTriangulationMoved.clear();

  //
  //generate mesh data members
  //
  generateMesh(d_parallelTriangulationUnmoved, d_serialTriangulationUnmoved);
  generateMesh(d_parallelTriangulationMoved);
}

//
//generate Mesh
//
void meshGeneratorClass::generateParallelUnmovedPreviousMesh
                    (const std::vector<std::vector<double> > & atomLocations,
		     const std::vector<std::vector<double> > & imageAtomLocations,
		     const std::vector<std::vector<double> > & domainBoundingVectors)
{

  //
  //set the data members before generating mesh
  //
  d_atomPositions = atomLocations;
  d_imageAtomPositions = imageAtomLocations;
  d_domainBoundingVectors = domainBoundingVectors;

  d_parallelTriangulationUnmovedPrevious.clear();

  generateMesh(d_parallelTriangulationUnmovedPrevious);
}

//
//get unmoved serial mesh
//
const parallel::distributed::Triangulation<3> &
meshGeneratorClass::getSerialMeshUnmoved()
{
  return d_serialTriangulationUnmoved;
}

//
//get moved parallel mesh
//
const parallel::distributed::Triangulation<3> &
meshGeneratorClass::getParallelMeshMoved()
{
  return d_parallelTriangulationMoved;
}

//
//get unmoved parallel mesh
//
const parallel::distributed::Triangulation<3> &
meshGeneratorClass::getParallelMeshUnmoved()
{
  return d_parallelTriangulationUnmoved;
}

//
//get unmoved parallel mesh
//
const parallel::distributed::Triangulation<3> &
meshGeneratorClass::getParallelMeshUnmovedPrevious()
{
  return d_parallelTriangulationUnmovedPrevious;
}

//
void meshGeneratorClass::refineSerialMesh(unsigned int n_cell,
					  std::vector<double>& centroid,
					  std::vector<int>& localRefineFlag,
					  unsigned int n_global_cell,
					  parallel::distributed::Triangulation<3>& serialTriangulation)
{

  std::vector<int> globalRefineFlag(n_global_cell) ;
  std::vector<Point<3>> centroidGlobal(n_global_cell) ;
  std::vector<double> centroidGlobalData(3*n_global_cell) ;
  std::vector<int> localSize(n_mpi_processes, 0), localSizeAllProc(n_mpi_processes, 0), mpi_offsets(n_mpi_processes, 0) ;

  for (unsigned int i=0; i<n_mpi_processes; ++i)
    {
      if (this_mpi_process==i)
	localSize[i] = n_cell ;
    }
  MPI_Allreduce(&localSize[0],
		&localSizeAllProc[0],
		n_mpi_processes,
		MPI_INT,
		MPI_SUM,
		mpi_communicator) ;


  for (unsigned int i=1; i<n_mpi_processes; ++i)
    mpi_offsets[i] = mpi_offsets[i-1] + localSizeAllProc[i-1] ;


  MPI_Allgatherv(&localRefineFlag[0],
		 n_cell,
		 MPI_INT,
		 &globalRefineFlag[0],
		 &localSizeAllProc[0],
		 &mpi_offsets[0],
		 MPI_INT,
		 mpi_communicator) ;

  for (unsigned int i=0; i<n_mpi_processes; ++i)
    localSizeAllProc[i] = 3*localSizeAllProc[i];


  mpi_offsets.resize(n_mpi_processes, 0) ;
  for (unsigned int i=1; i<n_mpi_processes; ++i)
    mpi_offsets[i] = mpi_offsets[i-1] + localSizeAllProc[i-1] ;

  MPI_Allgatherv(&centroid[0],
		 3*n_cell,
		 MPI_DOUBLE,
		 &centroidGlobalData[0],
		 &localSizeAllProc[0],
		 &mpi_offsets[0],
		 MPI_DOUBLE,
		 mpi_communicator) ;

  for ( unsigned int i = 0; i < n_global_cell ; ++i ) {
    Point<3> p1 (centroidGlobalData[3*i+0], centroidGlobalData[3*i+1], centroidGlobalData[3*i+2]) ;
    centroidGlobal[i] = p1  ;
  }

  //
  Triangulation<3>::active_cell_iterator cell = serialTriangulation.begin_active();
  Triangulation<3>::active_cell_iterator endc = serialTriangulation.end();

  for(;cell != endc; ++cell)
    {
      if(cell->is_locally_owned())
	{
	  Point<3> p1 = cell->center() ;
	  unsigned int index = 0 ;
	  for(unsigned int i = 0; i < n_global_cell ; ++i )
	    {
	      Point<3> p2 = centroidGlobal[i] ;
	      double dist = (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]) + (p1[2]-p2[2])*(p1[2]-p2[2]) ;
	      if (dist < 1.0E-10)
		{
		  index = i ;
		  break;
		}
	    }
	  if(globalRefineFlag[index]==1)
	    cell->set_refine_flag();
	}
    }


}
