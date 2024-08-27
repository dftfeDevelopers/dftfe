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
//


#ifndef matrixFree_H_
#define matrixFree_H_
#include <MatrixFreeBase.h>
#include <linearAlgebraOperations.h>

namespace dftfe
{
  /**
   * @brief MatrixFree class template. template parameter nDofsPerDim
   * is the finite element polynomial order. nQuadPointsPerDim is the order of
   * the Gauss quadrature rule. batchSize is the size of batch tuned to hardware
   *
   * @author Gourab Panigrahi
   */
  template <int nDofsPerDim, int nQuadPointsPerDim>
  class MatrixFree : public MatrixFreeBase
  {
  public:
    /// Constructor
    MatrixFree(const MPI_Comm &mpi_comm, const int nCells);


    /**
     * @brief reinitialize data structures for total electrostatic potential solve.
     *
     * For Hartree electrostatic potential solve give an empty map to the
     * atoms parameter.
     *
     */
    void
    reinit(const dealii::MatrixFree<3, double> *matrixFreeDataPtr,
           const unsigned int &                 dofHandlerID,
           const int                            matrixFreeQuadratureID);


    /**
     * @brief Compute A matrix multipled by x.
     *
     */
    void
    computeAX(distributedDeviceVec<double> &Ax,
              distributedDeviceVec<double> &x);

    void
    computeAX(distributedDeviceVec<double> &Ax,
              distributedDeviceVec<double> &x,
              double                        coeffHelmholtz);

  private:
    const int d_nDofsPerCell, d_nQuadsPerCell, d_nCells;

    // shape function value, gradient, jacobian and map for matrixfree
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE>
                                                                        d_shapeFunction, d_jacobianFactor;
    dftfe::utils::MemoryStorage<int, dftfe::utils::MemorySpace::DEVICE> d_map;

    // Pointers to shape function value, gradient, jacobian and map for
    // matrixfree
    double *d_shapeFunctionPtr;
    double *d_jacobianFactorPtr;
    int *   d_mapPtr;

    dealii::ConditionalOStream pcout;
    const MPI_Comm             mpi_communicator;
    const int                  n_mpi_processes;
    const int                  this_mpi_process;
  };

} // namespace dftfe
#endif // matrixFree_H_
