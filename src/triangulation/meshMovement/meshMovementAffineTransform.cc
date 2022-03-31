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
// @author Sambit Das(2018)
//
//
#include <dftParameters.h>
#include <dftUtils.h>
#include <meshMovementAffineTransform.h>

namespace dftfe
{
  meshMovementAffineTransform::meshMovementAffineTransform(
    const MPI_Comm &mpi_comm_replica)
    : meshMovementClass(mpi_comm_replica)
  {}


  std::pair<bool, double>
  meshMovementAffineTransform::transform(
    const Tensor<2, 3, double> &deformationGradient)
  {
    d_deformationGradient = deformationGradient;
    if (dftParameters::verbosity == 2)
      pcout
        << "Computing triangulation displacement increment under affine deformation..."
        << std::endl;
    initIncrementField();
    computeIncrement();
    if (dftParameters::verbosity == 2)
      pcout << "...Computed triangulation displacement increment" << std::endl;

    dftUtils::transformDomainBoundingVectors(d_domainBoundingVectors,
                                             deformationGradient);

    updateTriangulationVertices();
    std::pair<bool, double> returnData = movedMeshCheck();
    return returnData;
  }

  std::pair<bool, double>
  meshMovementAffineTransform::moveMesh(
    const std::vector<Point<3>> &            controlPointLocations,
    const std::vector<Tensor<1, 3, double>> &controlPointDisplacements,
    const double                             controllingParameter,
    const bool                               moveSubdivided)
  {
    AssertThrow(false, dftUtils::ExcNotImplementedYet());
  }



  void
  meshMovementAffineTransform::computeIncrement()
  {
    const unsigned int vertices_per_cell = GeometryInfo<3>::vertices_per_cell;
    std::vector<bool>  vertex_touched(
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
            vertex_touched[global_vertex_no]     = true;
            const Point<3>             nodalCoor = cell->vertex(i);
            const Tensor<1, 3, double> increment =
              d_deformationGradient * nodalCoor - nodalCoor;

            for (unsigned int idim = 0; idim < 3; idim++)
              {
                const unsigned int globalDofIndex =
                  cell->vertex_dof_index(i, idim);

                d_incrementalDisplacement[globalDofIndex] = increment[idim];
              }
          }
  }

} // namespace dftfe
