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
// @author Phani Motamarri (2017)
//
#include "../../include/meshGenerator.h"
#include "../../include/dftParameters.h"

#define maxRefinementLevels 10

void markPeriodicFaces(Triangulation<3,3> &triangulation)
{

  double domainSizeX = dftParameters::domainSizeX,domainSizeY = dftParameters::domainSizeY,domainSizeZ=dftParameters::domainSizeZ;

  typename Triangulation<3,3>::active_cell_iterator cell, endc;

  //
  //mark faces
  //
  cell = triangulation.begin_active(), endc = triangulation.end();
  for(; cell!=endc; ++cell) 
    {
      for(unsigned int f=0; f < GeometryInfo<3>::faces_per_cell; ++f)
	{
	  const Point<3> face_center = cell->face(f)->center();
	  if(cell->face(f)->at_boundary())
	    {
	      if (std::abs(face_center[0]+(domainSizeX/2.0)) < 1.0e-5)
		cell->face(f)->set_boundary_id(1);
	      else if (std::abs(face_center[0]-(domainSizeX/2.0)) < 1.0e-5)
		cell->face(f)->set_boundary_id(2);
	      else if (std::abs(face_center[1]+(domainSizeY/2.0)) < 1.0e-5)
		cell->face(f)->set_boundary_id(3);
	      else if (std::abs(face_center[1]-(domainSizeY/2.0)) < 1.0e-5)
		cell->face(f)->set_boundary_id(4);
	      else if (std::abs(face_center[2]+(domainSizeZ/2.0)) < 1.0e-5)
		cell->face(f)->set_boundary_id(5);
	      else if (std::abs(face_center[2]-(domainSizeZ/2.0)) < 1.0e-5)
		cell->face(f)->set_boundary_id(6);
	    }
	}
    }

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<3>::cell_iterator> > periodicity_vector;
  for(int i = 0; i < 3; ++i)
    {
      GridTools::collect_periodic_faces(triangulation, /*b_id1*/ 2*i+1, /*b_id2*/ 2*i+2,/*direction*/ i, periodicity_vector);
    }

  triangulation.add_periodicity(periodicity_vector);
}


//
//constructor
//
meshGeneratorClass::meshGeneratorClass():
  d_parallelTriangulationUnmoved(MPI_COMM_WORLD),
  d_parallelTriangulationMoved(MPI_COMM_WORLD),
  mpi_communicator (MPI_COMM_WORLD),
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
void meshGeneratorClass::generateMesh(parallel::distributed::Triangulation<3>& parallelTriangulation, Triangulation<3,3>& serialTriangulation, types::global_dof_index & numberGlobalCells)
{
  computing_timer.enter_section("mesh");
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

#ifdef ENABLE_PERIODIC_BC
      markPeriodicFaces(parallelTriangulation);
      markPeriodicFaces(serialTriangulation);
#endif  
    }
  else
    {

      //
      //compute magnitudes of domainBounding Vectors
      //
      double domainBoundingVectorMag1 = sqrt(d_domainBoundingVectors[0][0]*d_domainBoundingVectors[0][0] + d_domainBoundingVectors[0][1]*d_domainBoundingVectors[0][1] +  d_domainBoundingVectors[0][2]*d_domainBoundingVectors[0][2]);
      double domainBoundingVectorMag2 = sqrt(d_domainBoundingVectors[1][0]*d_domainBoundingVectors[1][0] + d_domainBoundingVectors[1][1]*d_domainBoundingVectors[1][1] +  d_domainBoundingVectors[1][2]*d_domainBoundingVectors[1][2]);
      double domainBoundingVectorMag3 = sqrt(d_domainBoundingVectors[2][0]*d_domainBoundingVectors[2][0] + d_domainBoundingVectors[2][1]*d_domainBoundingVectors[2][1] +  d_domainBoundingVectors[2][2]*d_domainBoundingVectors[2][2]);
      
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


            
      for (int i=0; i<3;i++){

        double temp = numberIntervalsEachDirection[i]-std::floor(numberIntervalsEachDirection[i]);
        if((temp - std::floor(temp)) >= 0.5)
           subdivisions[i] = std::ceil(numberIntervalsEachDirection[i]);
        else
	   subdivisions[i] = std::floor(numberIntervalsEachDirection[i]);
      }

      /*subdivisions[0] = 16 ; subdivisions[1] = 16 ; subdivisions[2] = 24 ; 
      numberIntervalsEachDirection[0] = 16.0; numberIntervalsEachDirection[1] = 16.0 ; numberIntervalsEachDirection[2] = 24.0 ; */
      GridGenerator::subdivided_parallelepiped<3>(parallelTriangulation,
	                                          subdivisions,
				                  basisVectors);

       GridGenerator::subdivided_parallelepiped<3>(serialTriangulation,
	                                          subdivisions,
				                  basisVectors);

      /*GridGenerator::parallelepiped<3>(parallelTriangulation,
				       basisVectors);

      GridGenerator::parallelepiped<3>(serialTriangulation,
				       basisVectors);*/

      //
      //print basis vectors
      //
      pcout << basisVectors[0] << " "<<basisVectors[1] << " " << basisVectors[2]<<std::endl;

      //
      //Translate the main grid so that midpoint is at center
      //
      const Point<3> translation = 0.5*(vector1+vector2+vector3);
      GridTools::shift(-translation,parallelTriangulation);
      GridTools::shift(-translation,serialTriangulation);
      
      //
      //collect periodic faces of the first level mesh to set up periodic boundary conditions later
      //
#ifdef ENABLE_PERIODIC_BC
      markPeriodicFaces(parallelTriangulation);
      markPeriodicFaces(serialTriangulation);
#endif

            
      /*std::vector<double>::iterator result =  std::max_element(numberIntervalsEachDirection.begin(),
							       numberIntervalsEachDirection.end());
      double maxNumberIntervals = *result;
      double temp = log(maxNumberIntervals)/log(2);
      unsigned int baseRefinementLevel;
      if((temp - std::floor(temp)) >= 0.5)
	baseRefinementLevel = std::ceil(temp);
      else
	baseRefinementLevel = std::floor(temp);

      parallelTriangulation.refine_global(baseRefinementLevel);
      serialTriangulation.refine_global(baseRefinementLevel); */

      char buffer1[100];
      sprintf(buffer1, "\n Base uniform number of elements: %u\n", parallelTriangulation.n_global_active_cells());
      pcout << buffer1;
      
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
	    {
	      if(cell->is_locally_owned())
		{
		  dealii::Point<3> center(cell->center());
		  double currentMeshSize = cell->minimum_vertex_distance();

		  bool inInnerDomain = false;
		  //
		  //compute projection of the vector joining the center of domain and centroid of cell onto
		  //each of the domain bounding vectors
		  //
		  double projComponent_1 = (center[0]*d_domainBoundingVectors[0][0]+center[1]*d_domainBoundingVectors[0][1]+center[2]*d_domainBoundingVectors[0][2])/domainBoundingVectorMag1;
		  double projComponent_2 = (center[0]*d_domainBoundingVectors[1][0]+center[1]*d_domainBoundingVectors[1][1]+center[2]*d_domainBoundingVectors[1][2])/domainBoundingVectorMag2;
		  double projComponent_3 = (center[0]*d_domainBoundingVectors[2][0]+center[1]*d_domainBoundingVectors[2][1]+center[2]*d_domainBoundingVectors[2][2])/domainBoundingVectorMag3;

		  if((std::fabs(projComponent_1) <= dftParameters::innerDomainSizeX) && (std::fabs(projComponent_2) <= dftParameters::innerDomainSizeY) && (std::fabs(projComponent_3) <= dftParameters::innerDomainSizeZ))
		    {
		      inInnerDomain = true;
		    }

		  bool cellRefineFlag = false;

		  if(inInnerDomain && (currentMeshSize > dftParameters::meshSizeInnerDomain))
		    cellRefineFlag = true;

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

	    }

	  //Refine
	  refineFlag= Utilities::MPI::max((unsigned int) refineFlag, mpi_communicator);

	  if (refineFlag)
	    {
	      if(numLevels<maxRefinementLevels)
		{
		  char buffer2[100];
		  sprintf(buffer2, "refinement in progress, level: %u\n", numLevels);
		  pcout << buffer2;
		  parallelTriangulation.execute_coarsening_and_refinement();
		  numLevels++;
		}
	      else
		{
		  refineFlag=false;
		}
	    }
           
           // Refine serial mesh
	      numberGlobalCells = parallelTriangulation.n_global_active_cells();
	      //pcout << " check 1.1 " << std::endl ;
              refineSerialMesh(n_cell, centroid, localRefineFlag, numberGlobalCells, serialTriangulation) ;
	      //pcout << " check 1.2 " << std::endl ;
	      serialTriangulation.execute_coarsening_and_refinement();
	      
        }
      //
      //compute some adaptive mesh metrics
      //
      double minElemLength = dftParameters::meshSizeOuterDomain;
      cell = parallelTriangulation.begin_active();
      endc = parallelTriangulation.end();
      for ( ; cell != endc; ++cell)
	{
	if (cell->is_locally_owned())
	  {
	    if (cell->minimum_vertex_distance() < minElemLength) minElemLength = cell->minimum_vertex_distance();
	  }
	}
  
      minElemLength=Utilities::MPI::min(minElemLength, mpi_communicator);

      //
      //print out adaptive mesh metrics
      //
      pcout << "Refinement levels executed: " << numLevels << std::endl;
      char buffer[100];
      sprintf(buffer, "Adaptivity summary:\n numCells: %u, numLevels: %u, h_min: %5.2e\n", parallelTriangulation.n_global_active_cells(), numLevels, minElemLength);
      pcout << buffer;

      int numberGlobalCellsParallel = parallelTriangulation.n_global_active_cells();
      int numberGlobalCellsSerial = serialTriangulation.n_global_active_cells();

      sprintf(buffer, " numParallelCells: %u, numSerialCells: %u \n", numberGlobalCellsParallel, numberGlobalCellsSerial);
      pcout << buffer;

    }

  computing_timer.exit_section("mesh");

}

//
//generate Mesh
//
void meshGeneratorClass::generateSerialAndParallelMesh(std::vector<std::vector<double> > & atomLocations,
						       std::vector<std::vector<double> > & imageAtomLocations,
						       std::vector<std::vector<double> > & domainBoundingVectors)
{
  
  //
  //set the data members before generating mesh
  //
  d_atomPositions = atomLocations;
  d_imageAtomPositions = imageAtomLocations;
  d_domainBoundingVectors = domainBoundingVectors;

  types::global_dof_index numberGlobalCellsSerial,numberGlobalCellsParallel;

  //
  //generate unmoved mesh data members
  //
  //generateMesh(numberGlobalCellsSerial);

  generateMesh(d_parallelTriangulationUnmoved, d_serialTriangulationUnmoved, numberGlobalCellsParallel);
  generateMesh(d_parallelTriangulationMoved, d_serialTriangulationMoved, numberGlobalCellsParallel);

  //AssertThrow(numberGlobalCellsParallel==numberGlobalCellsSerial,ExcMessage("Number of cells are different for parallel and serial triangulations"));

  //generate moved mesh data members which are still the same meshes before
  //but will be accessed to move the meshes later
  
  /*generateMesh(d_serialTriangulationMoved,
	       numberGlobalCellsSerial);

  generateMesh(d_parallelTriangulationMoved,
	       numberGlobalCellsParallel);*/

  

}

//
//get unmoved serial mesh
//
Triangulation<3,3> &
meshGeneratorClass::getSerialMesh()
{
  return d_serialTriangulationMoved;
  //return d_serialTriangulationUnmoved;
}

//
//get moved serial mesh
//
parallel::distributed::Triangulation<3> &
meshGeneratorClass::getParallelMesh()
{
  return d_parallelTriangulationMoved;
  //return d_parallelTriangulationUnmoved;
}
//
void meshGeneratorClass::refineSerialMesh(unsigned int n_cell, std::vector<double>& centroid, std::vector<int>& localRefineFlag, unsigned int n_global_cell, Triangulation<3,3>& serialTriangulation)
{
 /*std::vector<double> centroid ;
 std::vector<bool> localRefineFlag ;
 unsigned int n_cell=0 ;
 cell = d_parallelTriangulationUnmoved.begin_active();
 endc = d_parallelTriangulationUnmoved.end();
 for(;cell != endc; ++cell)
  {
  if(cell->is_locally_owned())
    {
     n_cell++ ;
     centroid.push_back(cell->center()) ; 
     //localRefineFlag.push_back (cell->refine()) ;
    }
  }*/
  //
  std::vector<int> globalRefineFlag(n_global_cell) ;
  std::vector<Point<3>> centroidGlobal(n_global_cell) ;
  std::vector<double> centroidGlobalData(3*n_global_cell) ;
  std::vector<int> localSize(n_mpi_processes, 0), localSizeAllProc(n_mpi_processes, 0), mpi_offsets(n_mpi_processes, 0) ;
  //std::map<Point<3>, bool> centerToRefineMap ;
  //
  //pcout << "1 localSizeAllProc.size() " << localSizeAllProc.size() << std::endl ;
  for (unsigned int i=0; i<n_mpi_processes; ++i) {
      if (this_mpi_process==i)
	 localSize[i] = n_cell ;
  }
  MPI_Allreduce(&localSize[0], &localSizeAllProc[0], n_mpi_processes, MPI_INT, MPI_SUM, mpi_communicator) ;
  //
 // pcout << "2 localSizeAllProc.size() " << localSizeAllProc.size() << std::endl ;
 // pcout << " check 1.11 " <<  localSize[0] << " " << localSizeAllProc[0] << std::endl ;
  //
  for (unsigned int i=1; i<n_mpi_processes; ++i) 
      mpi_offsets[i] = mpi_offsets[i-1] + localSizeAllProc[i-1] ;
  //
  //pcout << " check 1.115 " <<  std::endl ;
  //
  MPI_Allgatherv(&localRefineFlag[0], n_cell, MPI_INT, &globalRefineFlag[0], &localSizeAllProc[0], &mpi_offsets[0], MPI_INT, mpi_communicator) ;
  //
  //MPI_Barrier(mpi_communicator) ;
  //pcout << " check 1.12 " << n_mpi_processes << std::endl ;
  //pcout << "3 localSizeAllProc.size() " << localSizeAllProc.size() << std::endl ;
  for (unsigned int i=0; i<n_mpi_processes; ++i) 
      localSizeAllProc[i] = 3*localSizeAllProc[i];
  //
  //pcout << " check 1.121 " << n_mpi_processes << std::endl ;
  mpi_offsets.resize(n_mpi_processes, 0) ;
  for (unsigned int i=1; i<n_mpi_processes; ++i) 
      mpi_offsets[i] = mpi_offsets[i-1] + localSizeAllProc[i-1] ;
  //pcout << " check 1.125 " << std::endl ;
  //MPI_Barrier(mpi_communicator) ;
  MPI_Allgatherv(&centroid[0], 3*n_cell, MPI_DOUBLE, &centroidGlobalData[0], &localSizeAllProc[0], &mpi_offsets[0], MPI_DOUBLE, mpi_communicator) ;
  //
  //pcout << " check 1.13 " << std::endl ;
 for ( unsigned int i = 0; i < n_global_cell ; ++i ) {
     Point<3> p1 (centroidGlobalData[3*i+0], centroidGlobalData[3*i+1], centroidGlobalData[3*i+2]) ;
     centroidGlobal[i] = p1  ; 
     //centerToRefineMap[p1] =  globalRefineFlag[i] ;
   }
 //
 Triangulation<3>::active_cell_iterator cell = serialTriangulation.begin_active();
 Triangulation<3>::active_cell_iterator endc = serialTriangulation.end();
 //pcout << " check 1.14 " << std::endl ;
 for(;cell != endc; ++cell)
  {
  if(cell->is_locally_owned())
    {
      //auto index  = centerToRefineMap.find(cell->center()) ;
      /*std::map<Point<3>, bool>:: iterator index ;
      for (std::map<Point<3>, bool>:: iterator iter= centerToRefineMap.begin() ; iter!= centerToRefineMap.end() ; ++iter) {
	  Point<3> p1 = iter->first ;
	  Point<3> p2 = cell->center() ;
	  if (p1==p2) {
	     index = iter ;
	     break;
	  }
      }
      if(index->second)
	 cell->set_refine_flag();  */
     Point<3> p1 = cell->center() ;
     unsigned int index = 0 ;
     for ( unsigned int i = 0; i < n_global_cell ; ++i ) {
	  Point<3> p2 = centroidGlobal[i] ;
	  if (p1==p2) {
	      index = i ;
	      break;
	  }
      }
      if (globalRefineFlag[index]==1)
	   cell->set_refine_flag();

    }
  }
  //

}

