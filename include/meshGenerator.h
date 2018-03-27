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

/** @file meshGenerator.h
 *
 *  @brief This class generates and stores adaptive finite element meshes for the real-space dft problem.
 *
 *  The class uses an adpative mesh generation strategy to generate finite element mesh for given domain
 *  based on five input parameters: BASE MESH SIZE, ATOM BALL RADIUS, MESH SIZE ATOM BALL, MESH SIZE NEAR ATOM
 *  and MAX REFINEMENT STEPS (Refer to utils/dftParameters.cc for their corresponding internal variable names).
 *  Additionaly, this class also applies periodicity to mesh. The class stores two types of meshes: moved
 *  and unmoved. They are essentially the same meshes, except that we move the nodes of the moved mesh
 *  (in the meshMovement class) such that the atoms lie on the nodes. However, once the mesh is moved, dealii
 *  has issues using that mesh for further refinement, which is why we also carry an unmoved triangulation.
 *  There are other places where we require an unmoved triangulation, for example in projection of solution
 *  fields from the previous ground state in stucture optimization.
 *
 *  @author Phani Motamarri, Sambit Das, Krishnendu Ghosh
 */

#ifndef meshGenerator_H_
#define meshGenerator_H_
#include "headers.h"

using namespace dealii;


class meshGeneratorClass
{

 public:
/** @brief Constructor.
 *
 *  @param mpi_comm_replica mpi_communicator of the current pool
 *  @param interpoolcomm mpi_communicator across pools (required to synchronize mesh generation)
 */
  meshGeneratorClass(const MPI_Comm &mpi_comm_replica,const MPI_Comm &interpoolcomm);


  /**
   * meshGeneratorClass destructor
   */
  ~meshGeneratorClass();

/** @brief generates parallel moved and unmoved meshes, and serial unmoved mesh.
 *
 *  @param atomLocations vector containing cartesian coordinates at atoms with
 *  respect to origin (center of domain).
 *  @param imageAtomLocations vector containing cartesian coordinates of image
 *  atoms with respect to origin.
 *  @param domainBoundingVectors vector of domain bounding vectors (refer to
 *  description of input parameters.
 */
  void generateSerialUnmovedAndParallelMovedUnmovedMesh
              (const std::vector<std::vector<double> > & atomLocations,
	       const std::vector<std::vector<double> > & imageAtomLocations,
	       const std::vector<std::vector<double> > & domainBoundingVectors);

/** @brief generates serial and parallel unmoved previous mesh.
 *
 *  The function is to be used a update call to update the serial and parallel unmoved previous
 *  mesh after we have used it for the field projection purposes in structure optimization.
 *
 *  @param atomLocations vector containing cartesian coordinates at atoms with
 *  respect to origin (center of domain).
 *  @param imageAtomLocations vector containing cartesian coordinates of image
 *  atoms with respect to origin.
 *  @param domainBoundingVectors vector of domain bounding vectors (refer to
 *  description of input parameters.
 */
  void generateSerialAndParallelUnmovedPreviousMesh
              (const std::vector<std::vector<double> > & atomLocations,
	       const std::vector<std::vector<double> > & imageAtomLocations,
	       const std::vector<std::vector<double> > & domainBoundingVectors);

/**
 * @brief returns constant reference to serial unmoved triangulation
 *
 */
  const parallel::distributed::Triangulation<3> & getSerialMeshUnmoved();

/**
 * @brief returns constant reference to parallel moved triangulation
 *
 */
  const parallel::distributed::Triangulation<3> & getParallelMeshMoved();

/**
 * @brief returns constant reference to parallel unmoved triangulation
 *
 */
  const parallel::distributed::Triangulation<3> & getParallelMeshUnmoved();

/**
 * @brief returns constant reference to parallel unmoved previous triangulation
 * (triangulation used in the last ground state solve during structure optimization).
 *
 */
  const parallel::distributed::Triangulation<3> & getParallelMeshUnmovedPrevious();

/**
 * @brief returns constant reference to serial unmoved previous triangulation
 * (serial version of the triangulation used in the last ground state solve during
 * structure optimization).
 *
 */
  const parallel::distributed::Triangulation<3> & getSerialMeshUnmovedPrevious();

 private:

/**
 * @brief internal function which generates a parallel and serial mesh using a adaptive refinement strategy.
 *
 */
  void generateMesh(parallel::distributed::Triangulation<3>& parallelTriangulation, parallel::distributed::Triangulation<3>& serialTriangulation);

/**
 * @brief internal function which generates a parallel mesh using a adaptive refinement strategy.
 *
 */
  void generateMesh(parallel::distributed::Triangulation<3>& parallelTriangulation);

/**
 * @brief internal function which refines the serial mesh based on refinement flags from parallel mesh.
 * This ensures that we get the same mesh in serial and parallel.
 *
 */
  void refineSerialMesh(const unsigned int n_cell,
	                const  std::vector<double>& centroid,
			const std::vector<unsigned int>& localRefineFlag,
			const unsigned int n_global_cell,
			parallel::distributed::Triangulation<3>& serialTriangulation);

  //
  //data members
  //
  parallel::distributed::Triangulation<3> d_parallelTriangulationUnmoved;
  parallel::distributed::Triangulation<3> d_parallelTriangulationUnmovedPrevious;
  parallel::distributed::Triangulation<3> d_parallelTriangulationMoved;
  parallel::distributed::Triangulation<3> d_serialTriangulationUnmoved;
  parallel::distributed::Triangulation<3> d_serialTriangulationUnmovedPrevious;

  std::vector<std::vector<double> > d_atomPositions;
  std::vector<std::vector<double> > d_imageAtomPositions;
  std::vector<std::vector<double> > d_domainBoundingVectors;

  //
  //parallel objects
  //
  const MPI_Comm mpi_communicator;
  const MPI_Comm interpoolcomm;
  const unsigned int this_mpi_process;
  const unsigned int n_mpi_processes;
  dealii::ConditionalOStream   pcout;

  //
  //compute-time logger
  //
  TimerOutput computing_timer;

};

#endif
