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


#ifndef meshMovementGaussian_H_
#define meshMovementGaussian_H_
#include "meshMovement.h"

namespace dftfe
{
  /**
   * @brief Class to move triangulation nodes using Gaussian functions attached to control points
   *
   * @author Sambit Das
   */
  class meshMovementGaussianClass : public meshMovementClass
  {
  public:
    /** @brief Constructor
     *
     *  @param mpi_comm_parent parent mpi communicator
     *  @param mpi_comm_domain mpi communicator for domain decomposition
     */
    meshMovementGaussianClass(const MPI_Comm &mpi_comm_parent,
                              const MPI_Comm &mpi_comm_domaim);

    /** @brief Moves the triangulation corresponding to Gaussians attached to control points
     *
     *  This functions takes into account the hanging node and periodic
     * constraints when computing the nodal increment field.
     *
     *  @param controlPointLocations  vector of coordinates of control points
     *  @param controlPointDisplacements vector of displacements of control points
     *  @ controllingParameter constant in the Gaussian function:
     * exp(-(r/controllingParameter)^pow) where pow is controlled via the input
     * file parameter (.prm)
     *  @return std::pair<bool,double> mesh quality metrics
     *  pair(bool for is negative jacobian, maximum jacobian ratio)
     */
    std::pair<bool, double>
    moveMesh(const std::vector<Point<3>> &            controlPointLocations,
             const std::vector<Tensor<1, 3, double>> &controlPointDisplacements,
             const std::vector<double> &              gaussianWidthParameter,
             const std::vector<double> &              flatTopWidthParameter,
             const bool                               moveSubdivided = false);



    std::pair<bool, double>
    moveMeshTwoStep(
      const std::vector<Point<3>> &            controlPointLocations1,
      const std::vector<Point<3>> &            controlPointLocations2,
      const std::vector<Tensor<1, 3, double>> &controlPointDisplacements1,
      const std::vector<Tensor<1, 3, double>> &controlPointDisplacements2,
      const std::vector<double> &              controllingParameter1,
      const std::vector<double> &              controllingParameter2,
      const std::vector<double> &              flatTopWidthParameter,
      const bool                               moveSubdivided = false);


    void
    moveMeshTwoLevelElectro();


  private:
    /** @brief internal function which computes the nodal increment field in the local processor
     *
     */
    void
    computeIncrement(
      const std::vector<Point<3>> &            controlPointLocations,
      const std::vector<Tensor<1, 3, double>> &controlPointDisplacements,
      const std::vector<double> &              gaussianWidthParameter,
      const std::vector<double> &              flatTopWidthParameter);

    void
    computeIncrementTwoStep(
      const std::vector<Point<3>> &            controlPointLocations1,
      const std::vector<Point<3>> &            controlPointLocations2,
      const std::vector<Tensor<1, 3, double>> &controlPointDisplacements1,
      const std::vector<Tensor<1, 3, double>> &controlPointDisplacements2,
      const std::vector<double> &              gaussianWidthParameter1,
      const std::vector<double> &              gaussianWidthParameter2,
      const std::vector<double> &              flatTopWidthParameter);
  };

} // namespace dftfe
#endif
