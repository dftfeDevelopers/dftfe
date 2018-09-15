// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018  The Regents of the University of Michigan and DFT-FE authors.
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

#ifndef triangulationManager_H_
#define triangulationManager_H_
#include "headers.h"
#include "stdafx.h"

namespace dftfe  {

  using namespace dealii;

  /**
   * @brief This class generates and stores adaptive finite element meshes for the real-space dft problem.
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
  class triangulationManager
  {

  public:
    /** @brief Constructor.
     *
     * @param mpi_comm_replica mpi_communicator of the current pool
     * @param interpool_comm mpi interpool communicator over k points
     * @param interBandGroupComm mpi interpool communicator over band groups
     */
    triangulationManager(const MPI_Comm &mpi_comm_replica,
			 const MPI_Comm &interpoolcomm,
			 const MPI_Comm &interBandGroupComm);


    /**
     * triangulationManager destructor
     */
    ~triangulationManager();

    /** @brief generates parallel moved and unmoved meshes, and serial unmoved mesh.
     *
     *  @param atomLocations vector containing cartesian coordinates at atoms with
     *  respect to origin (center of domain).
     *  @param imageAtomLocations vector containing cartesian coordinates of image
     *  atoms with respect to origin.
     *  @param domainBoundingVectors vector of domain bounding vectors (refer to
     *  description of input parameters.
     *  @param generateSerialMesh bool to toggle to generation of serial tria
     */
    void generateSerialUnmovedAndParallelMovedUnmovedMesh
      (const std::vector<std::vector<double> > & atomLocations,
       const std::vector<std::vector<double> > & imageAtomLocations,
       const std::vector<std::vector<double> > & domainBoundingVectors,
       const bool generateSerialTria);


    /** @brief generates mesh for electrostatics problem
     *
     *  @param atomLocations vector containing cartesian coordinates at atoms with
     *  respect to origin (center of domain).
     *  @param imageAtomLocations vector containing cartesian coordinates of image
     *  atoms with respect to origin.
     *  @param domainBoundingVectors vector of domain bounding vectors (refer to
     *  description of input parameters.
     */
    void generateMeshForElectrostatics(const std::vector<std::vector<double> > & atomLocations,
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


    /** @brief generates the coarse meshes for restart.
     *
     *  @param atomLocations vector containing cartesian coordinates at atoms with
     *  respect to origin (center of domain).
     *  @param imageAtomLocations vector containing cartesian coordinates of image
     *  atoms with respect to origin.
     *  @param domainBoundingVectors vector of domain bounding vectors (refer to
     *  description of input parameters.
     *  @param generateSerialMesh bool to toggle to generation of serial tria
     */
    void generateCoarseMeshesForRestart
      (const std::vector<std::vector<double> > & atomLocations,
       const std::vector<std::vector<double> > & imageAtomLocations,
       const std::vector<std::vector<double> > & domainBoundingVectors,
       const bool generateSerialTria);


    /**
     * @brief returns constant reference to triangulation to compute electrostatics
     *
     */
    void  generateSubdividedMeshWithQuadData(const dealii::MatrixFree<3,double> & matrixFreeData,
					     const ConstraintMatrix & constraints,
					     const dealii::Quadrature<3> & quadrature,				     
					     const unsigned int FEOrder,
					     const std::map<dealii::CellId,std::vector<double> > & rhoQuadValuesCoarse,				     
					     std::map<dealii::CellId,std::vector<double> > & rhoQuadValuesRefined);


    /**
     * @brief returns constant reference to serial unmoved triangulation
     *
     */
    parallel::distributed::Triangulation<3> & getSerialMeshUnmoved();

    /**
     * @brief returns reference to parallel moved triangulation
     *
     */
    parallel::distributed::Triangulation<3> & getParallelMeshMoved();

    /**
     * @brief returns constant reference to parallel unmoved triangulation
     *
     */
    parallel::distributed::Triangulation<3> & getParallelMeshUnmoved();

    /**
     * @brief returns constant reference to parallel unmoved previous triangulation
     * (triangulation used in the last ground state solve during structure optimization).
     *
     */
    parallel::distributed::Triangulation<3> & getParallelMeshUnmovedPrevious();

    /**
     * @brief returns constant reference to serial unmoved previous triangulation
     * (serial version of the triangulation used in the last ground state solve during
     * structure optimization).
     *
     */
    parallel::distributed::Triangulation<3> & getSerialMeshUnmovedPrevious();
   
      
    /**
     * @brief returns constant reference to triangulation to compute electrostatics
     *
     */
    parallel::distributed::Triangulation<3> & getElectrostaticsMesh();

    

    /**
     * @brief resets the vertices of parallel mesh moved to umoved. This is required before
     * any mesh refinemen/coarsening operations are performed.
     *
     */
    void resetParallelMeshMovedToUnmoved();

    /**
     * @brief serialize the triangulations and the associated solution vectors
     *
     *  @param [input]feOrder finite element polynomial order of the dofHandler on which solution
     *  vectors are based upon
     *  @param [input]nComponents number of components of the dofHandler on which solution
     *  vectors are based upon
     *  @param [input]solutionVectors vector of parallel distributed solution vectors to be serialized
     *  @param [input]interpoolComm This communicator is used to ensure serialization
     *  happens only in k point pool
     *  @param [input]interBandGroupComm This communicator to ensure serialization happens
     *  only in band group
     */
    void saveTriangulationsSolutionVectors
      (const unsigned int feOrder,
       const unsigned int nComponents,
       const std::vector< const dealii::parallel::distributed::Vector<double> * > & solutionVectors,
       const MPI_Comm & interpoolComm,
       const MPI_Comm &interBandGroupComm);

    /**
     * @brief de-serialize the triangulations and the associated solution vectors
     *
     *  @param [input]feOrder finite element polynomial order of the dofHandler on which solution
     *  vectors to be de-serialized are based upon
     *  @param [input]nComponents number of components of the dofHandler on which solution
     *  vectors to be de-serialized are based upon
     *  @param [output]solutionVectors vector of parallel distributed de-serialized solution vectors. The
     *  vector length must match the input vector length used in the call to saveTriangulationSolutionVectors
     */
    void loadTriangulationsSolutionVectors
      (const unsigned int feOrder,
       const unsigned int nComponents,
       std::vector< dealii::parallel::distributed::Vector<double> * > & solutionVectors);
    /**
     * @brief serialize the triangulations and the associated cell quadrature data container
     *
     *  @param [input]cellQuadDataContainerIn container of input cell quadrature data to be serialized
     *  @param [input]interpoolComm This communicator is used to ensure serialization
     *  happens only in k point pool
     *  @param [input]interBandGroupComm This communicator to ensure serialization happens
     *  only in band group
     */
    void saveTriangulationsCellQuadData
      (const std::vector<const std::map<dealii::CellId, std::vector<double> > *> & cellQuadDataContainerIn,
       const MPI_Comm & interpoolComm,
       const MPI_Comm &interBandGroupComm);

    /**
     * @brief de-serialize the triangulations and the associated cell quadrature data container
     *
     *  @param [output]cellQuadDataContainerOut container of output cell quadrature data. Must pass container
     *  of the same size used in the call to saveTriangulationsCellQuadData.
     *  @param [input]cellDataSizeContainer vector of size of the per cell quadrature data. Must match the
     *  size and the ordering used in saveTriangulationsCellQuadData
     */
    void loadTriangulationsCellQuadData
      (std::vector<std::map<dealii::CellId, std::vector<double> > > & cellQuadDataContainerOut,
       const std::vector<unsigned int>  & cellDataSizeContainer);

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
     * @brief internal function which generates a coarse mesh which is required for the load function call in
     * restarts.
     *
     */
    void generateCoarseMesh(parallel::distributed::Triangulation<3>& parallelTriangulation);

    /**
     * @brief internal function which sets refinement flags based on a custom created algorithm
     *
     */
    void refinementAlgorithmA(parallel::distributed::Triangulation<3>& parallelTriangulation,
			      std::vector<unsigned int> & locallyOwnedCellsRefineFlags,
			      std::map<dealii::CellId,unsigned int> & cellIdToCellRefineFlagMapLocal);

    /**
     * @brief internal function which refines the serial mesh based on refinement flags from parallel mesh.
     * This ensures that we get the same mesh in serial and parallel.
     *
     */
    void refineSerialMesh(const std::map<dealii::CellId,unsigned int> & cellIdToCellRefineFlagMapLocal,
			  const MPI_Comm &mpi_comm,
			  parallel::distributed::Triangulation<3>& serialTriangulation);

    /**
     * @brief internal function to serialize support triangulations. No solution data is attached to them
     */
    void saveSupportTriangulations();

    /**
     * @brief internal function to de-serialize support triangulations. No solution data is read from them
     */
    void loadSupportTriangulations();

    //
    //data members
    //
    parallel::distributed::Triangulation<3> d_parallelTriangulationUnmoved;
    parallel::distributed::Triangulation<3> d_parallelTriangulationUnmovedPrevious;
    parallel::distributed::Triangulation<3> d_parallelTriangulationMoved;
    parallel::distributed::Triangulation<3> d_triangulationElectrostatics;
    parallel::distributed::Triangulation<3> d_serialTriangulationUnmoved;
    parallel::distributed::Triangulation<3> d_serialTriangulationUnmovedPrevious;
    

    std::vector<std::vector<double> > d_atomPositions;
    std::vector<std::vector<double> > d_imageAtomPositions;
    std::vector<std::vector<double> > d_domainBoundingVectors;
    const unsigned int d_max_refinement_steps=20;

    //
    //parallel objects
    //
    const MPI_Comm mpi_communicator;
    const MPI_Comm interpoolcomm;
    const MPI_Comm interBandGroupComm;
    const unsigned int this_mpi_process;
    const unsigned int n_mpi_processes;
    dealii::ConditionalOStream   pcout;

    //
    //compute-time logger
    //
    TimerOutput computing_timer;

  };

}
#endif
