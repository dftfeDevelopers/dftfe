// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
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

#ifndef triangulationManager_H_
#define triangulationManager_H_
#include "headers.h"


namespace dftfe
{
  using namespace dealii;

  /**
   * @brief This class generates and stores adaptive finite element meshes for the real-space dft problem.
   *
   *  The class uses an adpative mesh generation strategy to generate finite
   * element mesh for given domain based on five input parameters: BASE MESH
   * SIZE, ATOM BALL RADIUS, MESH SIZE ATOM BALL, MESH SIZE NEAR ATOM and MAX
   * REFINEMENT STEPS (Refer to utils/dftParameters.cc for their corresponding
   * internal variable names). Additionaly, this class also applies periodicity
   * to mesh. The class stores two types of meshes: moved and unmoved. They are
   * essentially the same meshes, except that we move the nodes of the moved
   * mesh (in the meshMovement class) such that the atoms lie on the nodes.
   * However, once the mesh is moved, dealii has issues using that mesh for
   * further refinement, which is why we also carry an unmoved triangulation.
   *  There are other places where we require an unmoved triangulation, for
   * example in projection of solution fields from the previous ground state in
   * stucture optimization.
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
    triangulationManager(const MPI_Comm &   mpi_comm_replica,
                         const MPI_Comm &   interpoolcomm,
                         const MPI_Comm &   interBandGroupComm,
                         const unsigned int FEOrder);


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
     *  @param generateElectrostaticsTria bool to toggle to generate separate tria for electrostatics
     */
    void
    generateSerialUnmovedAndParallelMovedUnmovedMesh(
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<std::vector<double>> &imageAtomLocations,
      const std::vector<int> &                imageIds,
      const std::vector<double> &             nearestAtomDistances,
      const std::vector<std::vector<double>> &domainBoundingVectors,
      const bool                              generateSerialTria,
      const bool                              generateElectrostaticsTria);



    /** @brief generates serial and parallel unmoved previous mesh.
     *
     *  The function is to be used a update call to update the serial and
     * parallel unmoved previous mesh after we have used it for the field
     * projection purposes in structure optimization.
     *
     *  @param atomLocations vector containing cartesian coordinates at atoms with
     *  respect to origin (center of domain).
     *  @param imageAtomLocations vector containing cartesian coordinates of image
     *  atoms with respect to origin.
     *  @param domainBoundingVectors vector of domain bounding vectors (refer to
     *  description of input parameters.
     */
    void
    generateSerialAndParallelUnmovedPreviousMesh(
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<std::vector<double>> &imageAtomLocations,
      const std::vector<int> &                imageIds,
      const std::vector<double> &             nearestAtomDistances,
      const std::vector<std::vector<double>> &domainBoundingVectors);


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
    void
    generateCoarseMeshesForRestart(
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<std::vector<double>> &imageAtomLocations,
      const std::vector<int> &                imageIds,
      const std::vector<double> &             nearestAtomDistances,
      const std::vector<std::vector<double>> &domainBoundingVectors,
      const bool                              generateSerialTria);


    /**
     * @brief returns generates A-posteriori refined mesh
     *
     * @param dofHandler corresponds to starting mesh which has to refined
     * @param parallelTriangulation corresponds to starting triangulation
     * @param eigenVectorsArrayIn solution vectors used to compute errors in each cell required for refinement
     * @param FEOrder finite-element interpolating polynomial
     * @param generateElectrostaticsTria required for generating electrostatics triangulation
     */
    void
    generateAutomaticMeshApriori(
      const dealii::DoFHandler<3> &                 dofHandler,
      parallel::distributed::Triangulation<3> &     parallelTriangulation,
      const std::vector<distributedCPUVec<double>> &eigenVectorsArrayIn,
      const unsigned int                            FEOrder,
      const bool                                    generateElectrostaticsTria);


    /**
     * @brief returns constant reference to serial unmoved triangulation
     *
     */
    parallel::distributed::Triangulation<3> &
    getSerialMeshUnmoved();

    /**
     * @brief returns constant reference to serial unmoved triangulation
     *
     */
    parallel::distributed::Triangulation<3> &
    getSerialMeshElectrostatics();

    /**
     * @brief returns reference to parallel moved triangulation
     *
     */
    parallel::distributed::Triangulation<3> &
    getParallelMeshMoved();

    /**
     * @brief returns constant reference to parallel unmoved triangulation
     *
     */
    parallel::distributed::Triangulation<3> &
    getParallelMeshUnmoved();

    /**
     * @brief returns constant reference to parallel unmoved previous triangulation
     * (triangulation used in the last ground state solve during structure
     * optimization).
     *
     */
    parallel::distributed::Triangulation<3> &
    getParallelMeshUnmovedPrevious();

    /**
     * @brief returns constant reference to serial unmoved previous triangulation
     * (serial version of the triangulation used in the last ground state solve
     * during structure optimization).
     *
     */
    parallel::distributed::Triangulation<3> &
    getSerialMeshUnmovedPrevious();


    /**
     * @brief returns constant reference to triangulation to compute electrostatics
     *
     */
    parallel::distributed::Triangulation<3> &
    getElectrostaticsMeshRho();


    /**
     * @brief returns constant reference to triangulation to compute electrostatics
     *
     */
    parallel::distributed::Triangulation<3> &
    getElectrostaticsMeshDisp();


    /**
     * @brief returns constant reference to triangulation to compute electrostatics
     *
     */
    parallel::distributed::Triangulation<3> &
    getElectrostaticsMeshForce();

    /**
     * @brief resets the vertices of meshB moved to vertices of meshA.
     *
     */
    void resetMesh(
      parallel::distributed::Triangulation<3> &parallelTriangulationA,
      parallel::distributed::Triangulation<3> &parallelTriangulationB);

    /**
     * @brief generates reset meshes to the last mesh generated by auto mesh approach. This
     * is used in Gaussian update of meshes during electrostatics
     */
    void
    generateResetMeshes(
      const std::vector<std::vector<double>> &domainBoundingVectors,
      const bool                              generateSerialTria,
      const bool                              generateElectrostaticsTria);


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
    void
    saveTriangulationsSolutionVectors(
      const unsigned int                                    feOrder,
      const unsigned int                                    nComponents,
      const std::vector<const distributedCPUVec<double> *> &solutionVectors,
      const MPI_Comm &                                      interpoolComm,
      const MPI_Comm &                                      interBandGroupComm);

    /**
     * @brief de-serialize the triangulations and the associated solution vectors
     *
     *  @param [input]feOrder finite element polynomial order of the dofHandler on which solution
     *  vectors to be de-serialized are based upon
     *  @param [input]nComponents number of components of the dofHandler on which solution
     *  vectors to be de-serialized are based upon
     *  @param [output]solutionVectors vector of parallel distributed de-serialized solution vectors. The
     *  vector length must match the input vector length used in the call to
     * saveTriangulationSolutionVectors
     */
    void
    loadTriangulationsSolutionVectors(
      const unsigned int                        feOrder,
      const unsigned int                        nComponents,
      std::vector<distributedCPUVec<double> *> &solutionVectors);
    /**
     * @brief serialize the triangulations and the associated cell quadrature data container
     *
     *  @param [input]cellQuadDataContainerIn container of input cell quadrature data to be serialized
     *  @param [input]interpoolComm This communicator is used to ensure serialization
     *  happens only in k point pool
     *  @param [input]interBandGroupComm This communicator to ensure serialization happens
     *  only in band group
     */
    void
    saveTriangulationsCellQuadData(
      const std::vector<const std::map<dealii::CellId, std::vector<double>> *>
        &             cellQuadDataContainerIn,
      const MPI_Comm &interpoolComm,
      const MPI_Comm &interBandGroupComm);

    /**
     * @brief de-serialize the triangulations and the associated cell quadrature data container
     *
     *  @param [output]cellQuadDataContainerOut container of output cell quadrature data. Must pass container
     *  of the same size used in the call to saveTriangulationsCellQuadData.
     *  @param [input]cellDataSizeContainer vector of size of the per cell quadrature data. Must match the
     *  size and the ordering used in saveTriangulationsCellQuadData
     */
    void
    loadTriangulationsCellQuadData(
      std::vector<std::map<dealii::CellId, std::vector<double>>>
        &                              cellQuadDataContainerOut,
      const std::vector<unsigned int> &cellDataSizeContainer);

  private:
    /**
     * @brief internal function which generates a parallel and serial mesh using a adaptive refinement strategy.
     *
     */
    void generateMesh(
      parallel::distributed::Triangulation<3> &parallelTriangulation,
      parallel::distributed::Triangulation<3> &serialTriangulation,
      parallel::distributed::Triangulation<3>
        &serialTriangulationElectrostatics,
      parallel::distributed::Triangulation<3> &electrostaticsTriangulationRho,
      parallel::distributed::Triangulation<3> &electrostaticsTriangulationDisp,
      parallel::distributed::Triangulation<3> &electrostaticsTriangulationForce,
      const bool                               generateElectrostaticsTria,
      const bool                               generateSerialTria = false);


    /**
     * @brief internal function which generates a coarse mesh which is required for the load function call in
     * restarts.
     *
     */
    void generateCoarseMesh(
      parallel::distributed::Triangulation<3> &parallelTriangulation);

    /**
     * @brief internal function which sets refinement flags based on a custom created algorithm
     *
     * @return bool boolean flag is any local cell has refinement flag set
     */
    bool refinementAlgorithmA(
      parallel::distributed::Triangulation<3> &parallelTriangulation,
      parallel::distributed::Triangulation<3> &electrostaticsTriangulationRho,
      parallel::distributed::Triangulation<3> &electrostaticsTriangulationDisp,
      parallel::distributed::Triangulation<3> &electrostaticsTriangulationForce,
      const bool                               generateElectrostaticsTria,
      std::vector<unsigned int> &              locallyOwnedCellsRefineFlags,
      std::map<dealii::CellId, unsigned int> & cellIdToCellRefineFlagMapLocal,
      const bool   smoothenCellsOnPeriodicBoundary = false,
      const double smootheningFactor               = 2.0);

    /**
     * @brief internal function which sets refinement flags to have consistent refinement across periodic
     * boundary
     *
     * @return bool boolean flag is any local cell has refinement flag set
     */
    bool consistentPeriodicBoundaryRefinement(
      parallel::distributed::Triangulation<3> &parallelTriangulation,
      parallel::distributed::Triangulation<3> &electrostaticsTriangulationRho,
      parallel::distributed::Triangulation<3> &electrostaticsTriangulationDisp,
      parallel::distributed::Triangulation<3> &electrostaticsTriangulationForce,
      const bool                               generateElectrostaticsTria,
      std::vector<unsigned int> &              locallyOwnedCellsRefineFlags,
      std::map<dealii::CellId, unsigned int> & cellIdToCellRefineFlagMapLocal);

    /**
     * @brief check that triangulation has consistent refinement across periodic boundary including
     * for ghost cells
     *
     */
    bool checkPeriodicSurfaceRefinementConsistency(
      parallel::distributed::Triangulation<3> &parallelTriangulation);


    /**
     * @brief check that FEOrder=1 dofHandler using the triangulation has parallel consistent
     * combined hanging node and periodic constraints
     *
     */
    bool checkConstraintsConsistency(
      parallel::distributed::Triangulation<3> &parallelTriangulation);


    /**
     * @brief internal function which refines the serial mesh based on refinement flags from parallel mesh.
     * This ensures that we get the same mesh in serial and parallel.
     *
     */
    void
    refineSerialMesh(
      const std::map<dealii::CellId, unsigned int>
        &                                      cellIdToCellRefineFlagMapLocal,
      const MPI_Comm &                         mpi_comm,
      parallel::distributed::Triangulation<3> &serialTriangulation,
      const parallel::distributed::Triangulation<3> &parallelTriangulation,
      std::vector<bool> &serialTriaCurrentRefinement);

    /**
     * @brief internal function to serialize support triangulations. No solution data is attached to them
     */
    void
    saveSupportTriangulations();

    /**
     * @brief internal function to de-serialize support triangulations. No solution data is read from them
     */
    void
    loadSupportTriangulations();

    //
    // data members
    //
    parallel::distributed::Triangulation<3> d_parallelTriangulationUnmoved;
    parallel::distributed::Triangulation<3>
                                            d_parallelTriangulationUnmovedPrevious;
    parallel::distributed::Triangulation<3> d_parallelTriangulationMoved;
    parallel::distributed::Triangulation<3> d_triangulationElectrostaticsRho;
    parallel::distributed::Triangulation<3> d_triangulationElectrostaticsDisp;
    parallel::distributed::Triangulation<3> d_triangulationElectrostaticsForce;
    parallel::distributed::Triangulation<3> d_serialTriangulationUnmoved;
    parallel::distributed::Triangulation<3>
                                            d_serialTriangulationUnmovedPrevious;
    parallel::distributed::Triangulation<3> d_serialTriangulationElectrostatics;

    std::vector<std::vector<bool>> d_parallelTriaCurrentRefinement;
    std::vector<std::vector<bool>> d_serialTriaCurrentRefinement;


    std::vector<std::vector<double>> d_atomPositions;
    std::vector<std::vector<double>> d_imageAtomPositions;
    std::vector<int>                 d_imageIds;
    std::vector<double>              d_nearestAtomDistances;
    std::vector<std::vector<double>> d_domainBoundingVectors;
    const unsigned int               d_max_refinement_steps = 40;

    /// FEOrder to be used for checking parallel consistency of periodic+hanging
    /// node constraints
    const unsigned int d_FEOrder;

    //
    // parallel objects
    //
    const MPI_Comm             mpi_communicator;
    const MPI_Comm             interpoolcomm;
    const MPI_Comm             interBandGroupComm;
    const unsigned int         this_mpi_process;
    const unsigned int         n_mpi_processes;
    dealii::ConditionalOStream pcout;

    //
    // compute-time logger
    //
    TimerOutput computing_timer;
  };

} // namespace dftfe
#endif
