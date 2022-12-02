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

#if defined(DFTFE_WITH_DEVICE)
#  ifndef linearSolverProblemDevice_H_
#    define linearSolverProblemDevice_H_

#    include <headers.h>
#    include <thrust/device_vector.h>
#    include <thrust/host_vector.h>

namespace dftfe
{
  /**
   * @brief Abstract class for linear solver problems to be used with the linearSolverCGDevice interface.
   *
   * @author Phani Motamarri, Sambit Das, Gourab Panigrahi
   */
  class linearSolverProblemDevice
  {
  public:
    /**
     * @brief Constructor.
     */
    linearSolverProblemDevice();

    /**
     * @brief get the reference to x field
     *
     * @return reference to x field. Assumes x field data structure is already initialized
     */
    virtual distributedDeviceVec<double> &
    getX() = 0;

    /**
     * @brief get the reference to Preconditioner
     *
     * @return reference to Preconditioner
     */
    virtual distributedDeviceVec<double> &
    getPreconditioner() = 0;

    /**
     * @brief Compute A matrix multipled by x.
     *
     */
    virtual void
    computeAX(distributedDeviceVec<double> &dst,
              distributedDeviceVec<double> &src) = 0;

    /**
     * @brief Compute right hand side vector for the problem Ax = rhs.
     *
     * @param rhs vector for the right hand side values
     */
    virtual void
    computeRhs(distributedCPUVec<double> &rhs) = 0;

    /**
     * @brief distribute x to the constrained nodes.
     *
     */
    virtual void
    setX() = 0;

    virtual void
    distributeX() = 0;

    /**
     * @brief copies x from device to host
     *
     */
    virtual void
    copyXfromDeviceToHost() = 0;

    // protected:

    /// typedef declaration needed by dealii
    typedef dealii::types::global_dof_index size_type;
  };

} // namespace dftfe
#  endif // linearSolverProblemDevice_H_
#endif
