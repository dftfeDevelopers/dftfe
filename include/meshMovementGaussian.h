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

/** @file meshMovementGaussian.h
 *  @brief Class to move triangulation nodes using Gaussian functions attached to control points
 *
 *  @author Sambit Das
 */

#ifndef meshMovementGaussian_H_
#define meshMovementGaussian_H_
#include "meshMovement.h"

class meshMovementGaussianClass : public meshMovementClass
{

public:
  /** @brief Constructor
   *
   *  @param mpi_comm_replica mpi communicator in the current pool
   */     
  meshMovementGaussianClass( MPI_Comm &mpi_comm_replica);

  /** @brief Moves the triangulation corresponding to Gaussians attached to control points
   * 
   *  This functions takes into account the hanging node and periodic constraints when 
   *  computing the nodal increment field.
   *
   *  @param controlPointLocations  vector of coordinates of control points
   *  @param controlPointDisplacements vector of displacements of control points
   *  @ controllingParameter constant in the Gaussian function: exp(-controllingParameter*r^2)
   *  @return std::pair<bool,double> mesh quality metrics 
   *  pair(bool for is negative jacobian, maximum jacobian ratio) 
   */  
  std::pair<bool,double> moveMesh(const std::vector<Point<C_DIM> > & controlPointLocations,
                                  const std::vector<Tensor<1,3,double> > & controlPointDisplacements,
                                  const double controllingParameter);
private: 
  /** @brief internal function which computes the nodal increment field in the local processor
   *
   */     
  void computeIncrement();  
  
  /// internal: storage for coordinates of the control points to which the Gaussians are attached
  std::vector<Point<C_DIM> > d_controlPointLocations;

  /// internal: storage for the displacements of each control point
  std::vector<Tensor<1,C_DIM,double> > d_controlPointDisplacements;

  /// internal: storage for the constant in the Gaussian function: exp(-d_controllingParameter*r^2)
  double d_controllingParameter;
};

#endif
