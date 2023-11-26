// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
// authors.
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


#include <constants.h>
#include <dftUtils.h>
#include <fileReaders.h>
#include <meshGenUtils.h>
#include <triangulationManager.h>

#include "generateMesh.cc"
#include "restartUtils.cc"

namespace dftfe
{
  //
  // constructor
  //
  triangulationManager::triangulationManager(
    const MPI_Comm &     mpi_comm_parent,
    const MPI_Comm &     mpi_comm_domain,
    const MPI_Comm &     interpoolcomm,
    const MPI_Comm &     interbandgroup_comm,
    const unsigned int   FEOrder,
    const dftParameters &dftParams)
    : d_parallelTriangulationUnmoved(mpi_comm_domain)
    , d_parallelTriangulationMoved(mpi_comm_domain)
    , d_mpiCommParent(mpi_comm_parent)
    , mpi_communicator(mpi_comm_domain)
    , interpoolcomm(interpoolcomm)
    , interBandGroupComm(interbandgroup_comm)
    , d_dftParams(dftParams)
    , d_serialTriangulationUnmoved(MPI_COMM_SELF)
    , d_FEOrder(FEOrder)
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0))
    , computing_timer(pcout,
                      dealii::TimerOutput::never,
                      dealii::TimerOutput::wall_times)
  {}

  //
  // destructor
  //
  triangulationManager::~triangulationManager()
  {}


  //
  // generate Mesh
  //
  void
  triangulationManager::generateSerialUnmovedAndParallelMovedUnmovedMesh(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<std::vector<double>> &imageAtomLocations,
    const std::vector<int> &                imageIds,
    const std::vector<double> &             nearestAtomDistances,
    const std::vector<std::vector<double>> &domainBoundingVectors,
    const bool                              generateSerialTria)
  {
    //
    // set the data members before generating mesh
    //
    d_atomPositions         = atomLocations;
    d_imageAtomPositions    = imageAtomLocations;
    d_imageIds              = imageIds;
    d_nearestAtomDistances  = nearestAtomDistances;
    d_domainBoundingVectors = domainBoundingVectors;

    // clear existing triangulation data
    d_serialTriangulationUnmoved.clear();
    d_parallelTriangulationUnmoved.clear();
    d_parallelTriangulationMoved.clear();
    //
    // generate mesh data members
    //
    generateMesh(d_parallelTriangulationUnmoved,
                 d_serialTriangulationUnmoved,
                 generateSerialTria);

    generateMesh(d_parallelTriangulationMoved,
                 d_serialTriangulationUnmoved,
                 false);
  }



  void
  triangulationManager::generateResetMeshes(
    const std::vector<std::vector<double>> &domainBoundingVectors,
    const bool                              generateSerialTria)
  {
    //
    // set the data members before generating mesh
    //
    d_domainBoundingVectors = domainBoundingVectors;

    // clear existing triangulation data
    d_serialTriangulationUnmoved.clear();
    d_parallelTriangulationUnmoved.clear();
    d_parallelTriangulationMoved.clear();

    //
    // generate mesh data members using cell refine flags
    //
    if (generateSerialTria)
      {
        generateCoarseMesh(d_serialTriangulationUnmoved);
        for (unsigned int i = 0; i < d_parallelTriaCurrentRefinement.size();
             ++i)
          {
            d_serialTriangulationUnmoved.load_refine_flags(
              d_serialTriaCurrentRefinement[i]);
            d_serialTriangulationUnmoved.execute_coarsening_and_refinement();
          }
      }

    generateCoarseMesh(d_parallelTriangulationUnmoved);
    generateCoarseMesh(d_parallelTriangulationMoved);
    for (unsigned int i = 0; i < d_parallelTriaCurrentRefinement.size(); ++i)
      {
        d_parallelTriangulationUnmoved.load_refine_flags(
          d_parallelTriaCurrentRefinement[i]);
        d_parallelTriangulationUnmoved.execute_coarsening_and_refinement();

        d_parallelTriangulationMoved.load_refine_flags(
          d_parallelTriaCurrentRefinement[i]);
        d_parallelTriangulationMoved.execute_coarsening_and_refinement();
      }
  }


  //
  //
  void
  triangulationManager::generateCoarseMeshesForRestart(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<std::vector<double>> &imageAtomLocations,
    const std::vector<int> &                imageIds,
    const std::vector<double> &             nearestAtomDistances,
    const std::vector<std::vector<double>> &domainBoundingVectors,
    const bool                              generateSerialTria)
  {
    //
    // set the data members before generating mesh
    //
    d_atomPositions         = atomLocations;
    d_imageAtomPositions    = imageAtomLocations;
    d_imageIds              = imageIds;
    d_nearestAtomDistances  = nearestAtomDistances;
    d_domainBoundingVectors = domainBoundingVectors;

    // clear existing triangulation data
    d_serialTriangulationUnmoved.clear();
    d_parallelTriangulationUnmoved.clear();
    d_parallelTriangulationMoved.clear();

    //
    // generate coarse meshes
    //
    if (generateSerialTria)
      generateCoarseMesh(d_serialTriangulationUnmoved);

    generateCoarseMesh(d_parallelTriangulationUnmoved);
    generateCoarseMesh(d_parallelTriangulationMoved);
  }

  //
  // get unmoved serial mesh
  //
  dealii::parallel::distributed::Triangulation<3> &
  triangulationManager::getSerialMeshUnmoved()
  {
    return d_serialTriangulationUnmoved;
  }

  //
  // get moved parallel mesh
  //
  dealii::parallel::distributed::Triangulation<3> &
  triangulationManager::getParallelMeshMoved()
  {
    return d_parallelTriangulationMoved;
  }

  //
  // get unmoved parallel mesh
  //
  dealii::parallel::distributed::Triangulation<3> &
  triangulationManager::getParallelMeshUnmoved()
  {
    return d_parallelTriangulationUnmoved;
  }

  // reset MeshB to MeshA
  void triangulationManager::resetMesh(
    dealii::parallel::distributed::Triangulation<3> &parallelTriangulationA,
    dealii::parallel::distributed::Triangulation<3> &parallelTriangulationB)
  {
    AssertThrow(parallelTriangulationA.n_global_active_cells() != 0,
                dftUtils::ExcInternalError());
    AssertThrow(parallelTriangulationA.n_global_active_cells() ==
                  parallelTriangulationB.n_global_active_cells(),
                dftUtils::ExcInternalError());

    std::vector<bool> vertexTouched(parallelTriangulationB.n_vertices(), false);
    typename dealii::parallel::distributed::Triangulation<3>::cell_iterator
      cellA,
      endcA, cellB;
    cellA = parallelTriangulationA.begin();
    endcA = parallelTriangulationA.end();
    cellB = parallelTriangulationB.begin();

    for (; cellA != endcA; ++cellA, ++cellB)
      for (unsigned int vertexNo = 0;
           vertexNo < dealii::GeometryInfo<3>::vertices_per_cell;
           ++vertexNo)
        {
          const unsigned int globalVertexNo = cellA->vertex_index(vertexNo);

          if (vertexTouched[globalVertexNo])
            continue;

          cellB->vertex(vertexNo) = cellA->vertex(vertexNo);

          vertexTouched[globalVertexNo] = true;
        }
  }

} // namespace dftfe
