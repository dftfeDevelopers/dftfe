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
#include "dftParameters.h"


namespace dftfe
{
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
   *
   *  @author Phani Motamarri, Sambit Das, Krishnendu Ghosh
   */
  class triangulationManager
  {
  public:
    /** @brief Constructor.
     *
     * @param mpi_comm_parent parent mpi communicator
     * @param mpi_comm_domain domain decomposition mpi communicator
     * @param interpool_comm mpi interpool communicator over k points
     * @param interBandGroupComm mpi interpool communicator over band groups
     */
    triangulationManager(const MPI_Comm &     mpi_comm_parent,
                         const MPI_Comm &     mpi_comm_domain,
                         const MPI_Comm &     interpoolcomm,
                         const MPI_Comm &     interBandGroupComm,
                         const unsigned int   FEOrder,
                         const dftParameters &dftParams);


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
    void
    generateSerialUnmovedAndParallelMovedUnmovedMesh(
      const std::vector<std::vector<double>> &atomLocations,
      const std::vector<std::vector<double>> &imageAtomLocations,
      const std::vector<int> &                imageIds,
      const std::vector<double> &             nearestAtomDistances,
      const std::vector<std::vector<double>> &domainBoundingVectors,
      const bool                              generateSerialTria);



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
     */
    void
    generateAutomaticMeshApriori(
      const dealii::DoFHandler<3> &                    dofHandler,
      dealii::parallel::distributed::Triangulation<3> &parallelTriangulation,
      const std::vector<distributedCPUVec<double>> &   eigenVectorsArrayIn,
      const unsigned int                               FEOrder);


    /**
     * @brief returns constant reference to serial unmoved triangulation
     *
     */
    dealii::parallel::distributed::Triangulation<3> &
    getSerialMeshUnmoved();

    /**
     * @brief returns reference to parallel moved triangulation
     *
     */
    dealii::parallel::distributed::Triangulation<3> &
    getParallelMeshMoved();

    /**
     * @brief returns constant reference to parallel unmoved triangulation
     *
     */
    dealii::parallel::distributed::Triangulation<3> &
    getParallelMeshUnmoved();

    /**
     * @brief resets the vertices of meshB moved to vertices of meshA.
     *
     */
    void resetMesh(
      dealii::parallel::distributed::Triangulation<3> &parallelTriangulationA,
      dealii::parallel::distributed::Triangulation<3> &parallelTriangulationB);

    /**
     * @brief generates reset meshes to the last mesh generated by auto mesh approach. This
     * is used in Gaussian update of meshes during electrostatics
     */
    void
    generateResetMeshes(
      const std::vector<std::vector<double>> &domainBoundingVectors,
      const bool                              generateSerialTria);


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
      std::string                                           path,
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
      std::string                               path,
      const unsigned int                        feOrder,
      const unsigned int                        nComponents,
      std::vector<distributedCPUVec<double> *> &solutionVectors);

  private:
    /**
     * @brief internal function which generates a parallel and serial mesh using a adaptive refinement strategy.
     *
     */
    void generateMesh(
      dealii::parallel::distributed::Triangulation<3> &parallelTriangulation,
      dealii::parallel::distributed::Triangulation<3> &serialTriangulation,
      const bool generateSerialTria = false);


    /**
     * @brief internal function which generates a coarse mesh which is required for the load function call in
     * restarts.
     *
     */
    void generateCoarseMesh(
      dealii::parallel::distributed::Triangulation<3> &parallelTriangulation);

    /**
     * @brief internal function which sets refinement flags based on a custom created algorithm
     *
     * @return bool boolean flag is any local cell has refinement flag set
     */
    bool refinementAlgorithmA(
      dealii::parallel::distributed::Triangulation<3> &parallelTriangulation,
      std::vector<unsigned int> &             locallyOwnedCellsRefineFlags,
      std::map<dealii::CellId, unsigned int> &cellIdToCellRefineFlagMapLocal,
      const bool   smoothenCellsOnPeriodicBoundary = false,
      const double smootheningFactor               = 2.0);

    /**
     * @brief internal function which sets refinement flags to have consistent refinement across periodic
     * boundary
     *
     * @return bool boolean flag is any local cell has refinement flag set
     */
    bool consistentPeriodicBoundaryRefinement(
      dealii::parallel::distributed::Triangulation<3> &parallelTriangulation,
      std::vector<unsigned int> &             locallyOwnedCellsRefineFlags,
      std::map<dealii::CellId, unsigned int> &cellIdToCellRefineFlagMapLocal);

    /**
     * @brief check that triangulation has consistent refinement across periodic boundary including
     * for ghost cells
     *
     */
    bool checkPeriodicSurfaceRefinementConsistency(
      dealii::parallel::distributed::Triangulation<3> &parallelTriangulation);


    /**
     * @brief check that FEOrder=1 dofHandler using the triangulation has parallel consistent
     * combined hanging node and periodic constraints
     *
     */
    bool checkConstraintsConsistency(
      dealii::parallel::distributed::Triangulation<3> &parallelTriangulation);


    /**
     * @brief internal function which refines the serial mesh based on refinement flags from parallel mesh.
     * This ensures that we get the same mesh in serial and parallel.
     *
     */
    void
    refineSerialMesh(
      const std::map<dealii::CellId, unsigned int>
        &             cellIdToCellRefineFlagMapLocal,
      const MPI_Comm &mpi_comm,
      dealii::parallel::distributed::Triangulation<3> &serialTriangulation,
      const dealii::parallel::distributed::Triangulation<3>
        &                parallelTriangulation,
      std::vector<bool> &serialTriaCurrentRefinement);

    /**
     * @brief internal function to serialize support triangulations. No solution data is attached to them
     */
    void
    saveSupportTriangulations(std::string path);

    /**
     * @brief internal function to de-serialize support triangulations. No solution data is read from them
     */
    void
    loadSupportTriangulations(std::string path);

    //
    // data members
    //
    dealii::parallel::distributed::Triangulation<3>
      d_parallelTriangulationUnmoved;
    dealii::parallel::distributed::Triangulation<3>
      d_parallelTriangulationMoved;
    dealii::parallel::distributed::Triangulation<3>
      d_serialTriangulationUnmoved;

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

    const dftParameters &d_dftParams;

    //
    // parallel objects
    //
    const MPI_Comm             d_mpiCommParent;
    const MPI_Comm             mpi_communicator;
    const MPI_Comm             interpoolcomm;
    const MPI_Comm             interBandGroupComm;
    const unsigned int         this_mpi_process;
    const unsigned int         n_mpi_processes;
    dealii::ConditionalOStream pcout;

    //
    // compute-time logger
    //
    dealii::TimerOutput computing_timer;
  };

} // namespace dftfe
#endif
