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
/** @file triangulationManager.cc
 *
 *  @brief Source file for triangulationManager.h
 *
 *  @author Phani Motamarri, Sambit Das, Krishnendu Ghosh
 */


#include <triangulationManager.h>
#include <dftParameters.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include "meshGenUtils.cc"
#include "generateMesh.cc"
#include "restartUtils.cc"

namespace dftfe {
//
//constructor
//
triangulationManager::triangulationManager(const MPI_Comm &mpi_comm_replica,const MPI_Comm &interpoolcomm):
  d_parallelTriangulationUnmoved(mpi_comm_replica),
  d_parallelTriangulationUnmovedPrevious(mpi_comm_replica),
  d_parallelTriangulationMoved(mpi_comm_replica),
  mpi_communicator (mpi_comm_replica),
  interpoolcomm(interpoolcomm),
  d_serialTriangulationUnmoved(MPI_COMM_SELF),
  d_serialTriangulationUnmovedPrevious(MPI_COMM_SELF),
  this_mpi_process (Utilities::MPI::this_mpi_process(mpi_comm_replica)),
  n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_comm_replica)),
  pcout (std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)),
  computing_timer (pcout, TimerOutput::never, TimerOutput::wall_times)
{

}

//
//destructor
//
triangulationManager::~triangulationManager()
{

}


//
//generate Mesh
//
void triangulationManager::generateSerialUnmovedAndParallelMovedUnmovedMesh
                    (const std::vector<std::vector<double> > & atomLocations,
		     const std::vector<std::vector<double> > & imageAtomLocations,
		     const std::vector<std::vector<double> > & domainBoundingVectors,
		     const bool generateSerialTria)
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
  if (generateSerialTria)
     generateMesh(d_parallelTriangulationUnmoved, d_serialTriangulationUnmoved);
  else
     generateMesh(d_parallelTriangulationUnmoved);
  generateMesh(d_parallelTriangulationMoved);
}

//
//generate Mesh
//
void triangulationManager::generateSerialAndParallelUnmovedPreviousMesh
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
  d_serialTriangulationUnmovedPrevious.clear();

  generateMesh(d_parallelTriangulationUnmovedPrevious, d_serialTriangulationUnmovedPrevious);
}

//
//
void triangulationManager::generateCoarseMeshesForRestart
		  (const std::vector<std::vector<double> > & atomLocations,
		   const std::vector<std::vector<double> > & imageAtomLocations,
		   const std::vector<std::vector<double> > & domainBoundingVectors,
		   const bool generateSerialTria)
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
  d_parallelTriangulationUnmovedPrevious.clear();
  d_serialTriangulationUnmovedPrevious.clear();

  //
  //generate coarse meshes
  //
  if (generateSerialTria)
     generateCoarseMesh(d_serialTriangulationUnmoved);

  generateCoarseMesh(d_parallelTriangulationUnmoved);
  generateCoarseMesh(d_parallelTriangulationMoved);
  generateCoarseMesh(d_parallelTriangulationUnmovedPrevious);
  generateCoarseMesh(d_serialTriangulationUnmovedPrevious);
}

//
//get unmoved serial mesh
//
const parallel::distributed::Triangulation<3> &
triangulationManager::getSerialMeshUnmoved() const
{
  return d_serialTriangulationUnmoved;
}

//
//get moved parallel mesh
//
parallel::distributed::Triangulation<3> &
triangulationManager::getParallelMeshMoved()
{
  return d_parallelTriangulationMoved;
}

//
//get unmoved parallel mesh
//
const parallel::distributed::Triangulation<3> &
triangulationManager::getParallelMeshUnmoved() const
{
  return d_parallelTriangulationUnmoved;
}

//
//get unmoved parallel mesh
//
const parallel::distributed::Triangulation<3> &
triangulationManager::getParallelMeshUnmovedPrevious() const
{
  return d_parallelTriangulationUnmovedPrevious;
}

//
//get unmoved serial mesh
//
const parallel::distributed::Triangulation<3> &
triangulationManager::getSerialMeshUnmovedPrevious() const
{
  return d_serialTriangulationUnmovedPrevious;
}

//
void
triangulationManager::resetParallelMeshMovedToUnmoved()
{
  AssertThrow(d_parallelTriangulationUnmoved.n_global_active_cells()!=0, dftUtils::ExcInternalError());
  AssertThrow(d_parallelTriangulationUnmoved.n_global_active_cells()
	    ==d_parallelTriangulationMoved.n_global_active_cells(), dftUtils::ExcInternalError());

  std::vector<bool> vertexTouched(d_parallelTriangulationMoved.n_vertices(),
                                 false);
  typename parallel::distributed::Triangulation<3>::cell_iterator cellUnmoved, endcUnmoved, cellMoved;
  cellUnmoved = d_parallelTriangulationUnmoved.begin();
  endcUnmoved = d_parallelTriangulationUnmoved.end();
  cellMoved=d_parallelTriangulationMoved.begin();

  for (; cellUnmoved!=endcUnmoved; ++cellUnmoved, ++cellMoved)
    for (unsigned int vertexNo=0; vertexNo<dealii::GeometryInfo<3>::vertices_per_cell;++vertexNo)
     {
	 const unsigned int globalVertexNo= cellUnmoved->vertex_index(vertexNo);

	 if (vertexTouched[globalVertexNo])
	   continue;

	 cellMoved->vertex(vertexNo) = cellUnmoved->vertex(vertexNo);

	 vertexTouched[globalVertexNo] = true;
      }

}

}
