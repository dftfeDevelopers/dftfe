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
#include <dftParameters.h>
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
    const MPI_Comm &   mpi_comm_replica,
    const MPI_Comm &   interpoolcomm,
    const MPI_Comm &   interbandgroup_comm,
    const unsigned int FEOrder)
    : d_parallelTriangulationUnmoved(mpi_comm_replica)
    , d_parallelTriangulationUnmovedPrevious(mpi_comm_replica)
    , d_parallelTriangulationMoved(mpi_comm_replica)
    , d_triangulationElectrostaticsRho(mpi_comm_replica)
    , d_triangulationElectrostaticsDisp(mpi_comm_replica)
    , d_triangulationElectrostaticsForce(mpi_comm_replica)
    , mpi_communicator(mpi_comm_replica)
    , interpoolcomm(interpoolcomm)
    , interBandGroupComm(interbandgroup_comm)
    , d_serialTriangulationUnmoved(MPI_COMM_SELF)
    , d_serialTriangulationUnmovedPrevious(MPI_COMM_SELF)
    , d_serialTriangulationElectrostatics(MPI_COMM_SELF)
    , d_FEOrder(FEOrder)
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_comm_replica))
    , n_mpi_processes(Utilities::MPI::n_mpi_processes(mpi_comm_replica))
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
    , computing_timer(pcout, TimerOutput::never, TimerOutput::wall_times)
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
    const bool                              generateSerialTria,
    const bool                              generateElectrostaticsTria)
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
    if (generateElectrostaticsTria)
      {
        d_triangulationElectrostaticsRho.clear();
        d_triangulationElectrostaticsDisp.clear();
        d_triangulationElectrostaticsForce.clear();
        d_serialTriangulationElectrostatics.clear();
      }
    //
    // generate mesh data members
    //
    generateMesh(d_parallelTriangulationUnmoved,
                 d_serialTriangulationUnmoved,
                 d_serialTriangulationElectrostatics,
                 d_triangulationElectrostaticsRho,
                 d_triangulationElectrostaticsDisp,
                 d_triangulationElectrostaticsForce,
                 generateElectrostaticsTria,
                 generateSerialTria);

    generateMesh(d_parallelTriangulationMoved,
                 d_serialTriangulationUnmoved,
                 d_serialTriangulationElectrostatics,
                 d_triangulationElectrostaticsRho,
                 d_triangulationElectrostaticsDisp,
                 d_triangulationElectrostaticsForce,
                 false,
                 false);
  }

  //
  // generate Mesh
  //
  void
  triangulationManager::generateSerialAndParallelUnmovedPreviousMesh(
    const std::vector<std::vector<double>> &atomLocations,
    const std::vector<std::vector<double>> &imageAtomLocations,
    const std::vector<int> &                imageIds,
    const std::vector<double> &             nearestAtomDistances,
    const std::vector<std::vector<double>> &domainBoundingVectors)
  {
    //
    // set the data members before generating mesh
    //
    d_atomPositions         = atomLocations;
    d_imageAtomPositions    = imageAtomLocations;
    d_imageIds              = imageIds;
    d_nearestAtomDistances  = nearestAtomDistances;
    d_domainBoundingVectors = domainBoundingVectors;

    d_parallelTriangulationUnmovedPrevious.clear();
    d_serialTriangulationUnmovedPrevious.clear();


    generateCoarseMesh(d_parallelTriangulationUnmovedPrevious);
    generateCoarseMesh(d_serialTriangulationUnmovedPrevious);

    for (unsigned int i = 0; i < d_parallelTriaCurrentRefinement.size(); ++i)
      {
        d_parallelTriangulationUnmovedPrevious.load_refine_flags(
          d_parallelTriaCurrentRefinement[i]);
        d_parallelTriangulationUnmovedPrevious
          .execute_coarsening_and_refinement();

        d_serialTriangulationUnmovedPrevious.load_refine_flags(
          d_serialTriaCurrentRefinement[i]);
        d_serialTriangulationUnmovedPrevious
          .execute_coarsening_and_refinement();
      }

    /*
       generateMesh(d_parallelTriangulationUnmovedPrevious,
       d_serialTriangulationUnmovedPrevious,
       d_triangulationElectrostaticsRho,
       d_triangulationElectrostaticsDisp,
       d_triangulationElectrostaticsForce,
       false);
     */
  }


  void
  triangulationManager::generateResetMeshes(
    const std::vector<std::vector<double>> &domainBoundingVectors,
    const bool                              generateSerialTria,
    const bool                              generateElectrostaticsTria)
  {
    //
    // set the data members before generating mesh
    //
    d_domainBoundingVectors = domainBoundingVectors;

    // clear existing triangulation data
    d_serialTriangulationUnmoved.clear();
    d_parallelTriangulationUnmoved.clear();
    d_parallelTriangulationMoved.clear();
    if (generateElectrostaticsTria)
      {
        d_triangulationElectrostaticsRho.clear();
        d_triangulationElectrostaticsDisp.clear();
        d_triangulationElectrostaticsForce.clear();
        d_serialTriangulationElectrostatics.clear();
      }

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


        if (generateElectrostaticsTria)
          {
            generateCoarseMesh(d_serialTriangulationElectrostatics);
            for (unsigned int i = 0; i < d_parallelTriaCurrentRefinement.size();
                 ++i)
              {
                d_serialTriangulationElectrostatics.load_refine_flags(
                  d_serialTriaCurrentRefinement[i]);
                d_serialTriangulationElectrostatics
                  .execute_coarsening_and_refinement();
              }
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

    if (generateElectrostaticsTria)
      {
        generateCoarseMesh(d_triangulationElectrostaticsRho);
        generateCoarseMesh(d_triangulationElectrostaticsDisp);
        generateCoarseMesh(d_triangulationElectrostaticsForce);
        for (unsigned int i = 0; i < d_parallelTriaCurrentRefinement.size();
             ++i)
          {
            d_triangulationElectrostaticsDisp.load_refine_flags(
              d_parallelTriaCurrentRefinement[i]);
            d_triangulationElectrostaticsDisp
              .execute_coarsening_and_refinement();

            d_triangulationElectrostaticsRho.load_refine_flags(
              d_parallelTriaCurrentRefinement[i]);
            d_triangulationElectrostaticsRho
              .execute_coarsening_and_refinement();

            d_triangulationElectrostaticsForce.load_refine_flags(
              d_parallelTriaCurrentRefinement[i]);
            d_triangulationElectrostaticsForce
              .execute_coarsening_and_refinement();
          }
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
    d_parallelTriangulationUnmovedPrevious.clear();
    d_serialTriangulationUnmovedPrevious.clear();

    //
    // generate coarse meshes
    //
    if (generateSerialTria)
      generateCoarseMesh(d_serialTriangulationUnmoved);

    generateCoarseMesh(d_parallelTriangulationUnmoved);
    generateCoarseMesh(d_parallelTriangulationMoved);
    if (dftParameters::isIonOpt || dftParameters::isCellOpt)
      {
        generateCoarseMesh(d_parallelTriangulationUnmovedPrevious);
        generateCoarseMesh(d_serialTriangulationUnmovedPrevious);
      }
  }

  //
  // get unmoved serial mesh
  //
  parallel::distributed::Triangulation<3> &
  triangulationManager::getSerialMeshUnmoved()
  {
    return d_serialTriangulationUnmoved;
  }

  //
  // get serial mesh for electrostatics
  //
  parallel::distributed::Triangulation<3> &
  triangulationManager::getSerialMeshElectrostatics()
  {
    return d_serialTriangulationElectrostatics;
  }

  //
  // get moved parallel mesh
  //
  parallel::distributed::Triangulation<3> &
  triangulationManager::getParallelMeshMoved()
  {
    return d_parallelTriangulationMoved;
  }

  //
  // get unmoved parallel mesh
  //
  parallel::distributed::Triangulation<3> &
  triangulationManager::getParallelMeshUnmoved()
  {
    return d_parallelTriangulationUnmoved;
  }

  //
  // get unmoved parallel mesh
  //
  parallel::distributed::Triangulation<3> &
  triangulationManager::getParallelMeshUnmovedPrevious()
  {
    return d_parallelTriangulationUnmovedPrevious;
  }

  //
  // get unmoved serial mesh
  //
  parallel::distributed::Triangulation<3> &
  triangulationManager::getSerialMeshUnmovedPrevious()
  {
    return d_serialTriangulationUnmovedPrevious;
  }


  //
  // get electrostatics mesh
  //
  parallel::distributed::Triangulation<3> &
  triangulationManager::getElectrostaticsMeshRho()
  {
    return d_triangulationElectrostaticsRho;
  }

  parallel::distributed::Triangulation<3> &
  triangulationManager::getElectrostaticsMeshDisp()
  {
    return d_triangulationElectrostaticsDisp;
  }

  parallel::distributed::Triangulation<3> &
  triangulationManager::getElectrostaticsMeshForce()
  {
    return d_triangulationElectrostaticsForce;
  }

  // reset MeshB to MeshA
  void triangulationManager::resetMesh(
    parallel::distributed::Triangulation<3> &parallelTriangulationA,
    parallel::distributed::Triangulation<3> &parallelTriangulationB)
  {
    AssertThrow(parallelTriangulationA.n_global_active_cells() != 0,
                dftUtils::ExcInternalError());
    AssertThrow(parallelTriangulationA.n_global_active_cells() ==
                  parallelTriangulationB.n_global_active_cells(),
                dftUtils::ExcInternalError());

    std::vector<bool> vertexTouched(parallelTriangulationB.n_vertices(), false);
    typename parallel::distributed::Triangulation<3>::cell_iterator cellA,
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
