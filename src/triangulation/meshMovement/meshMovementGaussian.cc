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
// @author Sambit Das(2017) and Phani Motamarri(2019)
//
//
#include <dftParameters.h>
#include <dftUtils.h>
#include <meshMovementGaussian.h>

namespace dftfe
{
  meshMovementGaussianClass::meshMovementGaussianClass(
    const MPI_Comm &mpi_comm_replica)
    : meshMovementClass(mpi_comm_replica)
  {}

  std::pair<bool, double>
  meshMovementGaussianClass::moveMesh(
    const std::vector<Point<3>> &            controlPointLocations,
    const std::vector<Tensor<1, 3, double>> &controlPointDisplacements,
    const std::vector<double> &              gaussianWidthParameter,
    const std::vector<double> &              flatTopWidthParameter,
    const bool                               moveSubdivided)
  {
    // writeMesh("meshUnmoved.vtu");
    MPI_Barrier(mpi_communicator);
    if (dftParameters::verbosity == 2)
      pcout
        << "Computing triangulation displacement increment caused by gaussian generator displacements..."
        << std::endl;

    initIncrementField();
    computeIncrement(controlPointLocations,
                     controlPointDisplacements,
                     gaussianWidthParameter,
                     flatTopWidthParameter);
    finalizeIncrementField();
    if (dftParameters::verbosity == 2)
      pcout << "...Computed triangulation displacement increment" << std::endl;
    if (moveSubdivided)
      moveSubdividedMesh();

    updateTriangulationVertices();
    std::pair<bool, double> returnData = movedMeshCheck();
    // writeMesh("meshMoved.vtu");
    return returnData;
  }


  std::pair<bool, double>
  meshMovementGaussianClass::moveMeshTwoStep(
    const std::vector<Point<3>> &controlPointLocationsInitialMove,
    const std::vector<Point<3>> &controlPointLocationsCurrentMove,
    const std::vector<Tensor<1, 3, double>>
      &controlPointDisplacementsInitialMove,
    const std::vector<Tensor<1, 3, double>>
      &                        controlPointDisplacementsCurrentMove,
    const std::vector<double> &controllingParameterInitialMove,
    const std::vector<double> &controllingParameterCurrentMove,
    const std::vector<double> &flatTopWidthParameter,
    const bool                 moveSubdivided)
  {
    // writeMesh("meshUnmoved.vtu");
    MPI_Barrier(mpi_communicator);
    if (dftParameters::verbosity == 2)
      pcout
        << "Computing triangulation displacement increment caused by gaussian generator displacements..."
        << std::endl;

    initIncrementField();
    computeIncrementTwoStep(controlPointLocationsInitialMove,
                            controlPointLocationsCurrentMove,
                            controlPointDisplacementsInitialMove,
                            controlPointDisplacementsCurrentMove,
                            controllingParameterInitialMove,
                            controllingParameterCurrentMove,
                            flatTopWidthParameter);
    finalizeIncrementField();
    if (dftParameters::verbosity == 2)
      pcout << "...Computed triangulation displacement increment" << std::endl;
    if (moveSubdivided)
      moveSubdividedMesh();

    updateTriangulationVertices();
    std::pair<bool, double> returnData = movedMeshCheck();
    // writeMesh("meshMoved.vtu");
    return returnData;
  }


  void
  meshMovementGaussianClass::moveMeshTwoLevelElectro()
  {
    moveSubdividedMesh();
    updateTriangulationVertices();
  }

  void
  meshMovementGaussianClass::computeIncrementTwoStep(
    const std::vector<Point<3>> &controlPointLocationsInitialMove,
    const std::vector<Point<3>> &controlPointLocationsCurrentMove,
    const std::vector<Tensor<1, 3, double>>
      &controlPointDisplacementsInitialMove,
    const std::vector<Tensor<1, 3, double>>
      &                        controlPointDisplacementsCurrentMove,
    const std::vector<double> &controllingParameterInitialMove,
    const std::vector<double> &controllingParameterCurrentMove,
    const std::vector<double> &flatTopWidthParameter)
  {
    unsigned int      vertices_per_cell = GeometryInfo<3>::vertices_per_cell;
    std::vector<bool> vertex_touched(
      d_dofHandlerMoveMesh.get_triangulation().n_vertices(), false);

    std::vector<Point<3>> nodalCoordinatesUpdated(
      d_dofHandlerMoveMesh.get_triangulation().n_vertices());
    DoFHandler<3>::active_cell_iterator cell =
                                          d_dofHandlerMoveMesh.begin_active(),
                                        endc = d_dofHandlerMoveMesh.end();

    for (; cell != endc; ++cell)
      if (!cell->is_artificial())
        for (unsigned int i = 0; i < vertices_per_cell; ++i)
          {
            const unsigned global_vertex_no = cell->vertex_index(i);

            if (vertex_touched[global_vertex_no])
              continue;

            Point<3> nodalCoor = cell->vertex(i);

            if (!vertex_touched[global_vertex_no])
              nodalCoordinatesUpdated[global_vertex_no] = nodalCoor;

            vertex_touched[global_vertex_no] = true;


            int overlappedControlPointId = -1;

            // check for case where control point locations coincide with nodal
            // vertex locations
            for (unsigned int jControl = 0;
                 jControl < controlPointLocationsInitialMove.size();
                 jControl++)
              {
                const double distance =
                  (nodalCoor - controlPointLocationsInitialMove[jControl])
                    .norm();
                if (distance < 1e-5)
                  {
                    overlappedControlPointId = jControl;
                    break;
                  }
              }

            for (unsigned int iControl = 0;
                 iControl < controlPointLocationsInitialMove.size();
                 iControl++)
              {
                if (overlappedControlPointId != iControl &&
                    overlappedControlPointId != -1)
                  continue;

                const double r =
                  (nodalCoor - controlPointLocationsInitialMove[iControl])
                    .norm();
                const double gaussianWeight =
                  dftParameters::reproducible_output ?
                    std::exp(
                      -std::pow(r / controllingParameterInitialMove[iControl],
                                2)) :
                    dftUtils::getCompositeGeneratorVal(
                      flatTopWidthParameter[iControl],
                      r,
                      controllingParameterInitialMove[iControl],
                      dftParameters::gaussianOrderMoveMeshToAtoms);

                for (unsigned int idim = 0; idim < 3; idim++)
                  {
                    const unsigned int globalDofIndex =
                      cell->vertex_dof_index(i, idim);

                    if (!d_constraintsMoveMesh.is_constrained(globalDofIndex))
                      {
                        d_incrementalDisplacement[globalDofIndex] +=
                          gaussianWeight *
                          controlPointDisplacementsInitialMove[iControl][idim];

                        nodalCoordinatesUpdated[global_vertex_no][idim] +=
                          gaussianWeight *
                          controlPointDisplacementsInitialMove[iControl][idim];
                      }
                  }
              }
          }


    DoFHandler<3>::active_cell_iterator cellStep2 =
                                          d_dofHandlerMoveMesh.begin_active(),
                                        endcStep2 = d_dofHandlerMoveMesh.end();
    std::vector<bool> vertex_touchedNew(
      d_dofHandlerMoveMesh.get_triangulation().n_vertices(), false);

    for (; cellStep2 != endcStep2; ++cellStep2)
      if (!cellStep2->is_artificial())
        for (unsigned int i = 0; i < vertices_per_cell; ++i)
          {
            const unsigned global_vertex_no = cellStep2->vertex_index(i);

            if (vertex_touchedNew[global_vertex_no])
              continue;

            vertex_touchedNew[global_vertex_no] = true;
            int overlappedControlPointId        = -1;
            for (unsigned int jControl = 0;
                 jControl < controlPointLocationsCurrentMove.size();
                 jControl++)
              {
                const double distance =
                  (nodalCoordinatesUpdated[global_vertex_no] -
                   controlPointLocationsCurrentMove[jControl])
                    .norm();
                if (distance < 1e-5)
                  {
                    overlappedControlPointId = jControl;
                    break;
                  }
              }

            for (unsigned int iControl = 0;
                 iControl < controlPointLocationsCurrentMove.size();
                 iControl++)
              {
                if (overlappedControlPointId != iControl &&
                    overlappedControlPointId != -1)
                  continue;

                const double r = (nodalCoordinatesUpdated[global_vertex_no] -
                                  controlPointLocationsCurrentMove[iControl])
                                   .norm();
                const double gaussianWeight =
                  dftParameters::reproducible_output ?
                    std::exp(
                      -std::pow(r / controllingParameterCurrentMove[iControl],
                                2)) :
                    dftUtils::getCompositeGeneratorVal(
                      flatTopWidthParameter[iControl],
                      r,
                      controllingParameterCurrentMove[iControl],
                      dftParameters::gaussianOrderForce);

                for (unsigned int idim = 0; idim < 3; idim++)
                  {
                    const unsigned int globalDofIndex =
                      cellStep2->vertex_dof_index(i, idim);

                    if (!d_constraintsMoveMesh.is_constrained(globalDofIndex))
                      {
                        d_incrementalDisplacement[globalDofIndex] +=
                          gaussianWeight *
                          controlPointDisplacementsCurrentMove[iControl][idim];
                      }
                  }
              }
          }
  }


  // The triangulation nodes corresponding to control point location are
  // constrained to only their corresponding controlPointDisplacements. In other
  // words for those nodes we don't consider overlapping Gaussians
  void
  meshMovementGaussianClass::computeIncrement(
    const std::vector<Point<3>> &            controlPointLocations,
    const std::vector<Tensor<1, 3, double>> &controlPointDisplacements,
    const std::vector<double> &              gaussianWidthParameter,
    const std::vector<double> &              flatTopWidthParameter)
  {
    unsigned int      vertices_per_cell = GeometryInfo<3>::vertices_per_cell;
    std::vector<bool> vertex_touched(
      d_dofHandlerMoveMesh.get_triangulation().n_vertices(), false);
    DoFHandler<3>::active_cell_iterator cell =
                                          d_dofHandlerMoveMesh.begin_active(),
                                        endc = d_dofHandlerMoveMesh.end();
    for (; cell != endc; ++cell)
      if (!cell->is_artificial())
        for (unsigned int i = 0; i < vertices_per_cell; ++i)
          {
            const unsigned global_vertex_no = cell->vertex_index(i);

            if (vertex_touched[global_vertex_no])
              continue;
            vertex_touched[global_vertex_no] = true;
            Point<3> nodalCoor               = cell->vertex(i);

            int overlappedControlPointId = -1;
            for (unsigned int jControl = 0;
                 jControl < controlPointLocations.size();
                 jControl++)
              {
                const double distance =
                  (nodalCoor - controlPointLocations[jControl]).norm();
                if (distance < 1e-5)
                  {
                    overlappedControlPointId = jControl;
                    break;
                  }
              }

            for (unsigned int iControl = 0;
                 iControl < controlPointLocations.size();
                 iControl++)
              {
                if (overlappedControlPointId != iControl &&
                    overlappedControlPointId != -1)
                  {
                    // std::cout<< " overlappedControlPointId: "<<
                    // overlappedControlPointId << std::endl;
                    continue;
                  }
                const double r =
                  (nodalCoor - controlPointLocations[iControl]).norm();
                const double gaussianWeight =
                  dftParameters::reproducible_output ?
                    std::exp(
                      -std::pow(r / gaussianWidthParameter[iControl], 2)) :
                    dftUtils::getCompositeGeneratorVal(
                      flatTopWidthParameter[iControl],
                      r,
                      gaussianWidthParameter[iControl],
                      dftParameters::gaussianOrderMoveMeshToAtoms);
                for (unsigned int idim = 0; idim < 3; idim++)
                  {
                    const unsigned int globalDofIndex =
                      cell->vertex_dof_index(i, idim);

                    if (!d_constraintsMoveMesh.is_constrained(globalDofIndex))
                      d_incrementalDisplacement[globalDofIndex] +=
                        gaussianWeight *
                        controlPointDisplacements[iControl][idim];
                  }
              }
          }
  }

} // namespace dftfe
