// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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


#ifndef DFTFE_MULTIVECTORLINEARSOLVERPROBLEM_H
#define DFTFE_MULTIVECTORLINEARSOLVERPROBLEM_H

#include "headers.h"

namespace dftfe
{
  /*
   * @brief This class provides an interface for the MultivectorSolverClass
   */
  template <dftfe::utils::MemorySpace memorySpace>
  class MultiVectorLinearSolverProblem
  {
  public:
    /**
     * @brief Compute right hand side vector for the problem Ax = rhs.
     *
     * @param rhs vector for the right hand side values
     */

    virtual dftfe::linearAlgebra::MultiVector<double, memorySpace> &
    computeRhs(
      dftfe::linearAlgebra::MultiVector<double, memorySpace> &NDBCVec,
      dftfe::linearAlgebra::MultiVector<double, memorySpace> &outputVec,
      unsigned int blockSizeInput) = 0;

    /**
     * @brief Compute A matrix multipled by x.
     *
     */
    virtual void
    vmult(dftfe::linearAlgebra::MultiVector<double, memorySpace> &Ax,
          dftfe::linearAlgebra::MultiVector<double, memorySpace> &x,
          unsigned int blockSize) = 0;

    /**
     * @brief Apply the constraints to the solution vector.
     *
     */
    virtual void
    distributeX() = 0;

    /**
     * @brief Jacobi preconditioning function.
     *
     */
    virtual void
    precondition_Jacobi(
      dftfe::linearAlgebra::MultiVector<double, memorySpace> &      dst,
      const dftfe::linearAlgebra::MultiVector<double, memorySpace> &src,
      const double omega) const = 0;

    /**
     * @brief Apply square-root of the Jacobi preconditioner function.
     *
     */
    virtual void
    precondition_JacobiSqrt(
      dftfe::linearAlgebra::MultiVector<double, memorySpace> &      dst,
      const dftfe::linearAlgebra::MultiVector<double, memorySpace> &src,
      const double omega) const = 0;
  };

} // end of namespace dftfe

#endif // DFTFE_MULTIVECTORLINEARSOLVERPROBLEM_H
