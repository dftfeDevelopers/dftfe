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
// @author Sambit Das(2017)
//
//
#include <dftParameters.h>
#include <meshMovement.h>
#include <vectorUtilities.h>

namespace dftfe
{
  namespace meshMovementUtils
  {
    extern "C"
    {
      //
      // lapack Ax=b
      //
      void
      dgesv_(int *   N,
             int *   NRHS,
             double *A,
             int *   LDA,
             int *   IPIV,
             double *B,
             int *   LDB,
             int *   INFO);
    }


    std::vector<double>
    getFractionalCoordinates(const std::vector<double> &latticeVectors,
                             const Point<3> &           point,
                             const Point<3> &           corner)
    {
      //
      // recenter vertex about corner
      //
      std::vector<double> recenteredPoint(3);
      for (int i = 0; i < 3; ++i)
        recenteredPoint[i] = point[i] - corner[i];

      std::vector<double> latticeVectorsDup = latticeVectors;

      //
      // to get the fractionalCoords, solve a linear
      // system of equations
      //
      int N    = 3;
      int NRHS = 1;
      int LDA  = 3;
      int IPIV[3];
      int info;

      dgesv_(&N,
             &NRHS,
             &latticeVectorsDup[0],
             &LDA,
             &IPIV[0],
             &recenteredPoint[0],
             &LDA,
             &info);

      if (info != 0)
        {
          const std::string message(
            "LU solve in finding fractional coordinates failed.");
          Assert(false, ExcMessage(message));
        }
      return recenteredPoint;
    }

  } // namespace meshMovementUtils
  //
  // constructor
  //
  meshMovementClass::meshMovementClass(const MPI_Comm &mpi_comm_replica)
    : FEMoveMesh(FE_Q<3>(QGaussLobatto<1>(2)), 3)
    , // linear shape function
    mpi_communicator(mpi_comm_replica)
    , this_mpi_process(Utilities::MPI::this_mpi_process(mpi_comm_replica))
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
  {}

  void meshMovementClass::init(
    Triangulation<3, 3> &                   triangulation,
    Triangulation<3, 3> &                   serialTriangulation,
    const std::vector<std::vector<double>> &domainBoundingVectors)
  {
    d_domainBoundingVectors = domainBoundingVectors;
    if (triangulation.locally_owned_subdomain() ==
        numbers::invalid_subdomain_id)
      d_isParallelMesh = false;
    else
      {
        d_isParallelMesh = true;
      }

    d_dofHandlerMoveMesh.clear();
    if (d_isParallelMesh)
      {
        d_triaPtr = &(dynamic_cast<parallel::distributed::Triangulation<3> &>(
          triangulation));
        d_dofHandlerMoveMesh.initialize(*d_triaPtr, FEMoveMesh);
      }
    else
      d_dofHandlerMoveMesh.initialize(triangulation, FEMoveMesh);
    d_dofHandlerMoveMesh.distribute_dofs(FEMoveMesh);
    d_locally_owned_dofs.clear();
    d_locally_relevant_dofs.clear();
    d_locally_owned_dofs = d_dofHandlerMoveMesh.locally_owned_dofs();
    DoFTools::extract_locally_relevant_dofs(d_dofHandlerMoveMesh,
                                            d_locally_relevant_dofs);

    d_constraintsMoveMesh.clear();
    d_constraintsMoveMesh.reinit(d_locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(d_dofHandlerMoveMesh,
                                            d_constraintsMoveMesh);
    d_periodicity_vector.clear();

    // create unitVectorsXYZ
    std::vector<std::vector<double>> unitVectorsXYZ;
    unitVectorsXYZ.resize(3);

    for (int i = 0; i < 3; ++i)
      {
        unitVectorsXYZ[i].resize(3, 0.0);
        unitVectorsXYZ[i][i] = 0.0;
      }

    std::vector<Tensor<1, 3>> offsetVectors;
    // resize offset vectors
    offsetVectors.resize(3);

    for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
          {
            offsetVectors[i][j] =
              unitVectorsXYZ[i][j] - domainBoundingVectors[i][j];
          }
      }

    const std::array<int, 3> periodic = {dftParameters::periodicX,
                                         dftParameters::periodicY,
                                         dftParameters::periodicZ};

    std::vector<int> periodicDirectionVector;
    for (unsigned int d = 0; d < 3; ++d)
      {
        if (periodic[d] == 1)
          {
            periodicDirectionVector.push_back(d);
          }
      }

    for (int i = 0; i < std::accumulate(periodic.begin(), periodic.end(), 0);
         ++i)
      {
        GridTools::collect_periodic_faces(
          d_dofHandlerMoveMesh,
          /*b_id1*/ 2 * i + 1,
          /*b_id2*/ 2 * i + 2,
          /*direction*/ periodicDirectionVector[i],
          d_periodicity_vector,
          offsetVectors[periodicDirectionVector[i]]);
      }

    DoFTools::make_periodicity_constraints<DoFHandler<3>>(
      d_periodicity_vector, d_constraintsMoveMesh);
    d_constraintsMoveMesh.close();

    if (dftParameters::createConstraintsFromSerialDofhandler)
      {
        d_triaPtrSerial = &serialTriangulation;
        dealii::AffineConstraints<double> dummy;
        vectorTools::createParallelConstraintMatrixFromSerial(
          serialTriangulation,
          d_dofHandlerMoveMesh,
          mpi_communicator,
          domainBoundingVectors,
          d_constraintsMoveMesh,
          dummy);
      }
  }

  void
  meshMovementClass::initMoved(
    const std::vector<std::vector<double>> &domainBoundingVectors)
  {
    d_dofHandlerMoveMesh.distribute_dofs(FEMoveMesh);
    d_domainBoundingVectors = domainBoundingVectors;
  }

  void
  meshMovementClass::initIncrementField()
  {
    // d_incrementalDisplacement.reinit(d_locally_relevant_dofs.size());
    // d_incrementalDisplacement=0;
    IndexSet ghost_indices = d_locally_relevant_dofs;
    ghost_indices.subtract_set(d_locally_owned_dofs);

    d_incrementalDisplacement.reinit(d_locally_owned_dofs,
                                     ghost_indices,
                                     mpi_communicator);

    d_incrementalDisplacement = 0.0;

    d_incrementalDisplacement.zero_out_ghosts();
  }

  void
  meshMovementClass::finalizeIncrementField()
  {
    d_constraintsMoveMesh.distribute(d_incrementalDisplacement);
    d_incrementalDisplacement.update_ghost_values();
  }

  void
  meshMovementClass::updateTriangulationVertices()
  {
    MPI_Barrier(mpi_communicator);
    if (dftParameters::verbosity >= 4)
      pcout << "Start moving triangulation..." << std::endl;
    std::vector<bool> vertex_moved(
      d_dofHandlerMoveMesh.get_triangulation().n_vertices(), false);
    const std::vector<bool> locally_owned_vertices =
      dealii::GridTools::get_locally_owned_vertices(
        d_dofHandlerMoveMesh.get_triangulation());

    // Next move vertices on locally owned cells
    DoFHandler<3>::active_cell_iterator cell =
      d_dofHandlerMoveMesh.begin_active();
    DoFHandler<3>::active_cell_iterator endc = d_dofHandlerMoveMesh.end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            for (unsigned int vertex_no = 0;
                 vertex_no < GeometryInfo<3>::vertices_per_cell;
                 ++vertex_no)
              {
                const unsigned global_vertex_no = cell->vertex_index(vertex_no);

                if (vertex_moved[global_vertex_no] ||
                    !locally_owned_vertices[global_vertex_no])
                  continue;

                Point<3> vertexDisplacement;
                for (unsigned int d = 0; d < 3; ++d)
                  {
                    const unsigned int globalDofIndex =
                      cell->vertex_dof_index(vertex_no, d);
                    vertexDisplacement[d] =
                      d_incrementalDisplacement[globalDofIndex];
                  }

                cell->vertex(vertex_no) += vertexDisplacement;
                vertex_moved[global_vertex_no] = true;
              }
          }
      }
    if (d_isParallelMesh)
      d_triaPtr->communicate_locally_moved_vertices(locally_owned_vertices);

    d_dofHandlerMoveMesh.distribute_dofs(FEMoveMesh);
    if (dftParameters::verbosity >= 4)
      pcout << "...End moving triangulation" << std::endl;
  }


  void
  meshMovementClass::moveSubdividedMesh()
  {
    //
    // create a solution transfer object and prepare for refinement and solution
    // transfer
    //
    parallel::distributed::SolutionTransfer<3, distributedCPUVec<double>>
      solTrans(d_dofHandlerMoveMesh);
    d_triaPtr->set_all_refine_flags();
    d_triaPtr->prepare_coarsening_and_refinement();
    solTrans.prepare_for_coarsening_and_refinement(d_incrementalDisplacement);
    d_triaPtr->execute_coarsening_and_refinement();

    if (dftParameters::createConstraintsFromSerialDofhandler)
      d_triaPtrSerial->refine_global(1);

    init(*d_triaPtr, *d_triaPtrSerial, d_domainBoundingVectors);

    initIncrementField();

    solTrans.interpolate(d_incrementalDisplacement);

    finalizeIncrementField();
  }

  std::pair<bool, double>
  meshMovementClass::movedMeshCheck()
  {
    // sanity check to make sure periodic boundary conditions are maintained
    MPI_Barrier(mpi_communicator);

    // create unitVectorsXYZ
    std::vector<std::vector<double>> unitVectorsXYZ;
    unitVectorsXYZ.resize(3);

    for (int i = 0; i < 3; ++i)
      {
        unitVectorsXYZ[i].resize(3, 0.0);
        unitVectorsXYZ[i][i] = 0.0;
      }

    std::vector<Tensor<1, 3>> offsetVectors;
    // resize offset vectors
    offsetVectors.resize(3);

    for (int i = 0; i < 3; ++i)
      {
        for (int j = 0; j < 3; ++j)
          {
            offsetVectors[i][j] =
              unitVectorsXYZ[i][j] - d_domainBoundingVectors[i][j];
          }
      }
    /*
       if (dftParameters::verbosity>=4)
       pcout << "Sanity check for periodic matched faces on moved
       triangulation..." << std::endl; for(unsigned int i=0; i<
       d_periodicity_vector.size(); ++i)
       {
       if (!d_periodicity_vector[i].cell[0]->active() ||
       !d_periodicity_vector[i].cell[1]->active()) continue; if
       (d_periodicity_vector[i].cell[0]->is_artificial() ||
       d_periodicity_vector[i].cell[1]->is_artificial()) continue;

       std::vector<bool> isPeriodicFace(3);
       for(unsigned int idim=0; idim<3; ++idim){
       isPeriodicFace[idim]=GridTools::orthogonal_equality(d_periodicity_vector[i].cell[0]->face(d_periodicity_vector[i].face_idx[0]),d_periodicity_vector[i].cell[1]->face(d_periodicity_vector[i].face_idx[1]),idim,offsetVectors[idim]);
       }

       AssertThrow(isPeriodicFace[0]==true || isPeriodicFace[1]==true ||
       isPeriodicFace[2]==true,ExcMessage("Previously periodic matched face
       pairs not matching periodically for any directions after mesh
       movement"));
       }
       MPI_Barrier(mpi_communicator);
       if (dftParameters::verbosity>=4)
       pcout << "...Sanity check passed" << std::endl;

     */
    // print out mesh metrics
    typename Triangulation<3, 3>::active_cell_iterator cell, endc;
    double                                             minElemLength = 1e+6;
    double                                             maxElemLength = 0.0;
    cell = d_dofHandlerMoveMesh.get_triangulation().begin_active();
    endc = d_dofHandlerMoveMesh.get_triangulation().end();
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            if (cell->minimum_vertex_distance() < minElemLength)
              minElemLength = cell->minimum_vertex_distance();

            if (cell->minimum_vertex_distance() > maxElemLength)
              maxElemLength = cell->minimum_vertex_distance();
          }
      }
    minElemLength = Utilities::MPI::min(minElemLength, mpi_communicator);
    maxElemLength = Utilities::MPI::max(maxElemLength, mpi_communicator);

    if (dftParameters::verbosity >= 4)
      pcout << "Mesh movement quality metric, h_min: " << minElemLength
            << ", h_max: " << maxElemLength << std::endl;

    std::pair<bool, double> meshQualityMetrics;
    QGauss<3>               quadrature(2);
    FEValues<3>        fe_values(FEMoveMesh, quadrature, update_JxW_values);
    const unsigned int num_quad_points = quadrature.size();
    cell = d_dofHandlerMoveMesh.get_triangulation().begin_active();
    int    isNegativeJacobianDeterminant = 0;
    double maxJacobianRatio              = 1;
    for (; cell != endc; ++cell)
      {
        if (cell->is_locally_owned())
          {
            // Compute values for current cell.
            fe_values.reinit(cell);
            double maxJacobian = -1e+6;
            double minJacobian = 1e+6;
            for (unsigned int q_point = 0; q_point < num_quad_points; ++q_point)
              {
                double jw = fe_values.JxW(q_point);
                double j  = jw / quadrature.weight(q_point);

                if (j < 0)
                  isNegativeJacobianDeterminant = 1;
                else if (j > 1e-6)
                  {
                    if (j > maxJacobian)
                      maxJacobian = j;
                    if (j < minJacobian)
                      minJacobian = j;
                  }
              }
            if (maxJacobian / minJacobian > maxJacobianRatio)
              maxJacobianRatio = maxJacobian / minJacobian;
          }
      }
    maxJacobianRatio = Utilities::MPI::max(maxJacobianRatio, mpi_communicator);
    isNegativeJacobianDeterminant =
      Utilities::MPI::max(isNegativeJacobianDeterminant, mpi_communicator);
    bool isNegativeJacobian = isNegativeJacobianDeterminant == 1 ? true : false;
    meshQualityMetrics = std::make_pair(isNegativeJacobian, maxJacobianRatio);
    return meshQualityMetrics;
    // std::cout << "l2 norm icrement field:
    // "<<d_incrementalDisplacement.l2_norm()<<std::endl;
  }


  void
  meshMovementClass::findClosestVerticesToDestinationPoints(
    const std::vector<Point<3>> &      destinationPoints,
    std::vector<Point<3>> &            closestTriaVertexToDestPointsLocation,
    std::vector<Tensor<1, 3, double>> &dispClosestTriaVerticesToDestPoints)
  {
    closestTriaVertexToDestPointsLocation.clear();
    dispClosestTriaVerticesToDestPoints.clear();
    unsigned int        vertices_per_cell = GeometryInfo<3>::vertices_per_cell;
    std::vector<double> latticeVectorsFlattened(9, 0.0);
    for (unsigned int idim = 0; idim < 3; idim++)
      for (unsigned int jdim = 0; jdim < 3; jdim++)
        latticeVectorsFlattened[3 * idim + jdim] =
          d_domainBoundingVectors[idim][jdim];
    Point<3> corner;
    for (unsigned int idim = 0; idim < 3; idim++)
      {
        corner[idim] = 0;
        for (unsigned int jdim = 0; jdim < 3; jdim++)
          corner[idim] -= d_domainBoundingVectors[jdim][idim] / 2.0;
      }
    std::vector<double> latticeVectorsMagnitudes(3, 0.0);
    for (unsigned int idim = 0; idim < 3; idim++)
      {
        for (unsigned int jdim = 0; jdim < 3; jdim++)
          latticeVectorsMagnitudes[idim] +=
            d_domainBoundingVectors[idim][jdim] *
            d_domainBoundingVectors[idim][jdim];
        latticeVectorsMagnitudes[idim] =
          std::sqrt(latticeVectorsMagnitudes[idim]);
      }

    std::vector<bool> isPeriodic(3, false);
    isPeriodic[0] = dftParameters::periodicX;
    isPeriodic[1] = dftParameters::periodicY;
    isPeriodic[2] = dftParameters::periodicZ;

    dealii::BoundingBox<3> boundingBoxTria(
      vectorTools::createBoundingBoxTriaLocallyOwned(d_dofHandlerMoveMesh));
    ;

    for (unsigned int idest = 0; idest < destinationPoints.size(); idest++)
      {
        std::vector<bool> isDestPointOnPeriodicSurface(3, false);

        std::vector<double> destFracCoords =
          meshMovementUtils::getFractionalCoordinates(latticeVectorsFlattened,
                                                      destinationPoints[idest],
                                                      corner);
        // std::cout<< "destFracCoords: "<< destFracCoords[0] << ","
        // <<destFracCoords[1] <<"," <<destFracCoords[2]<<std::endl;
        for (unsigned int idim = 0; idim < 3; idim++)
          {
            if ((std::fabs(destFracCoords[idim] - 0.0) <
                   1e-5 / latticeVectorsMagnitudes[idim] ||
                 std::fabs(destFracCoords[idim] - 1.0) <
                   1e-5 / latticeVectorsMagnitudes[idim]) &&
                isPeriodic[idim] == true)
              isDestPointOnPeriodicSurface[idim] = true;
          }

        // pcout<<"idest: "<<idest<< "destFracCoords: "<< destFracCoords[0] <<
        // "," <<destFracCoords[1] <<"," <<destFracCoords[2]<<"
        // isDestPeriodicSurface:
        // "<<isDestPointOnPeriodicSurface[0]<<isDestPointOnPeriodicSurface[1]<<isDestPointOnPeriodicSurface[2]<<std::endl;
        double   minDistance = 1e+6;
        Point<3> closestTriaVertexLocation;

        const double                 sphereRad = 2.0;
        dealii::Tensor<1, 3, double> tempDisp;
        tempDisp[0] = sphereRad;
        tempDisp[1] = sphereRad;
        tempDisp[2] = sphereRad;
        std::pair<dealii::Point<3, double>, dealii::Point<3, double>>
          boundaryPoints;
        boundaryPoints.first  = destinationPoints[idest] - tempDisp;
        boundaryPoints.second = destinationPoints[idest] + tempDisp;
        dealii::BoundingBox<3> boundingBoxAroundPoint(boundaryPoints);

        const bool isDestPointConsidered =
          (boundingBoxTria.get_neighbor_type(boundingBoxAroundPoint) ==
           NeighborType::not_neighbors) ?
            false :
            true;

        std::vector<bool> vertex_touched(
          d_dofHandlerMoveMesh.get_triangulation().n_vertices(), false);
        DoFHandler<3>::active_cell_iterator cell = d_dofHandlerMoveMesh
                                                     .begin_active(),
                                            endc = d_dofHandlerMoveMesh.end();

        if (isDestPointConsidered)
          for (; cell != endc; ++cell)
            {
              if (cell->is_locally_owned())
                {
                  for (unsigned int i = 0; i < vertices_per_cell; ++i)
                    {
                      const unsigned global_vertex_no = cell->vertex_index(i);

                      if (vertex_touched[global_vertex_no])
                        continue;
                      vertex_touched[global_vertex_no] = true;

                      if ((d_constraintsMoveMesh.is_constrained(
                             cell->vertex_dof_index(i, 0)) &&
                           !d_constraintsMoveMesh.is_identity_constrained(
                             cell->vertex_dof_index(i, 0))) ||
                          !d_locally_owned_dofs.is_element(
                            cell->vertex_dof_index(i, 0)))
                        {
                          continue;
                        }

                      Point<3>          nodalCoor = cell->vertex(i);
                      std::vector<bool> isNodeOnPeriodicSurface(3, false);

                      bool isNodeConsidered = true;

                      if (isDestPointOnPeriodicSurface[0] ||
                          isDestPointOnPeriodicSurface[1] ||
                          isDestPointOnPeriodicSurface[2])
                        {
                          std::vector<double> nodeFracCoords =
                            meshMovementUtils::getFractionalCoordinates(
                              latticeVectorsFlattened, nodalCoor, corner);
                          for (int idim = 0; idim < 3; idim++)
                            {
                              if ((std::fabs(nodeFracCoords[idim] - 0.0) <
                                     1e-5 / latticeVectorsMagnitudes[idim] ||
                                   std::fabs(nodeFracCoords[idim] - 1.0) <
                                     1e-5 / latticeVectorsMagnitudes[idim]) &&
                                  isPeriodic[idim] == true)
                                isNodeOnPeriodicSurface[idim] = true;
                            }
                          isNodeConsidered = false;
                          // std::cout<< "nodeFracCoords: "<< nodeFracCoords[0]
                          // << "," <<nodeFracCoords[1] <<","
                          // <<nodeFracCoords[2]<<std::endl;
                          if ((isDestPointOnPeriodicSurface[0] ==
                               isNodeOnPeriodicSurface[0]) &&
                              (isDestPointOnPeriodicSurface[1] ==
                               isNodeOnPeriodicSurface[1]) &&
                              (isDestPointOnPeriodicSurface[2] ==
                               isNodeOnPeriodicSurface[2]))
                            {
                              isNodeConsidered = true;
                              // std::cout<< "nodeFracCoords: "<<
                              // nodeFracCoords[0] << "," <<nodeFracCoords[1]
                              // <<"," <<nodeFracCoords[2]<<std::endl;
                            }
                        }

                      if (!isNodeConsidered)
                        continue;

                      const double distance =
                        (nodalCoor - destinationPoints[idest]).norm();

                      if (distance < minDistance)
                        {
                          minDistance               = distance;
                          closestTriaVertexLocation = nodalCoor;
                        }
                    }
                }
            }
        const double globalMinDistance =
          Utilities::MPI::min(minDistance, mpi_communicator);

        // std::cout << "minDistance: "<< minDistance << "globalMinDistance:
        // "<<globalMinDistance << " closest vertex location: "<<
        // closestTriaVertexLocation <<std::endl;

        int minProcIdWithGlobalMinDistance = 1e+6;


        if (std::fabs(minDistance - globalMinDistance) < 1e-5)
          {
            minProcIdWithGlobalMinDistance = this_mpi_process;
          }

        minProcIdWithGlobalMinDistance =
          Utilities::MPI::min(minProcIdWithGlobalMinDistance, mpi_communicator);

        if (this_mpi_process != minProcIdWithGlobalMinDistance)
          {
            closestTriaVertexLocation[0] = 0.0;
            closestTriaVertexLocation[1] = 0.0;
            closestTriaVertexLocation[2] = 0.0;
          }

        Point<3> closestTriaVertexLocationGlobal;
        // accumulate value
        MPI_Allreduce(&(closestTriaVertexLocation[0]),
                      &(closestTriaVertexLocationGlobal[0]),
                      3,
                      MPI_DOUBLE,
                      MPI_SUM,
                      mpi_communicator);
        // important in case of k point parallelization
        MPI_Bcast(
          &(closestTriaVertexLocation[0]), 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // floating point error correction
        // if
        // ((closestTriaVertexLocationGlobal-closestTriaVertexLocation).norm()<1e-5)
        //   closestTriaVertexLocationGlobal=closestTriaVertexLocation;

        // std::cout << closestTriaVertexLocationGlobal << " disp:
        // "<<Point<3>(destinationPoints[idest]-closestTriaVertexLocationGlobal)
        // << std::endl;
        closestTriaVertexToDestPointsLocation.push_back(
          closestTriaVertexLocationGlobal);
        Tensor<1, 3, double> temp =
          destinationPoints[idest] - closestTriaVertexLocationGlobal;
        dispClosestTriaVerticesToDestPoints.push_back(temp);
      }
  }
} // namespace dftfe
