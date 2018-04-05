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

/** @file meshMovementAffineTransform.h
 *  @brief Class to update triangulation under affine transformation
 *
 *  @author Sambit Das
 */

#ifndef meshMovementAffineTransform_H_
#define meshMovementAffineTransform_H_
#include "meshMovement.h"

namespace dftfe {

    class meshMovementAffineTransform : public meshMovementClass
    {

    public:

      /** @brief Constructor
       *
       *  @param mpi_comm_replica mpi communicator in the current pool
       */
      meshMovementAffineTransform(const MPI_Comm &mpi_comm_replica);

      /** @brief Performs affine transformation of the triangulation
       *
       *  @param  deformationGradient
       *  @return std::pair<bool,double> mesh quality metrics
       *  pair(bool for is negative jacobian, maximum jacobian ratio)
       */
      std::pair<bool,double> transform(const Tensor<2,3,double> & deformationGradient);

      /// Not implemented, just present to override the pure virtual from base class
      std::pair<bool,double> moveMesh(const std::vector<Point<C_DIM> > & controlPointLocations,
				      const std::vector<Tensor<1,3,double> > & controlPointDisplacements,
				      const double controllingParameter);

    private:

      /** @brief internal function which computes the nodal increment field in the local processor
       *
       */
      void computeIncrement();

      /// storage for the deformation gradient to be applied to the triangulation
      Tensor<2,3,double> d_deformationGradient;
    };

}

#endif
