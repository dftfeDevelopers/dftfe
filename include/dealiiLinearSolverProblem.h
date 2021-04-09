// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018  The Regents of the University of Michigan and DFT-FE
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

#include <headers.h>

#ifndef dealiiLinearSolverProblem_H_
#  define dealiiLinearSolverProblem_H_

namespace dftfe
{
  /**
   * @brief Abstract class for linear solve problems to be used with the dealiiLinearSolver interface.
   *
   * @author Sambit Das
   */
  class dealiiLinearSolverProblem
  {
  public:
    /**
     * @brief Constructor.
     */
    dealiiLinearSolverProblem();

    /**
     * @brief get the reference to x field
     *
     * @return reference to x field. Assumes x field data structure is already initialized
     */
    virtual distributedCPUVec<double> &
    getX() = 0;

    /**
     * @brief Compute A matrix multipled by x.
     *
     */
    virtual void
    vmult(distributedCPUVec<double> &Ax, distributedCPUVec<double> &x) = 0;

    /**
     * @brief Compute right hand side vector for the problem Ax = rhs.
     *
     * @param rhs vector for the right hand side values
     */
    virtual void
    computeRhs(distributedCPUVec<double> &rhs) = 0;

    /**
     * @brief Jacobi preconditioning function.
     *
     */
    virtual void
    precondition_Jacobi(distributedCPUVec<double> &      dst,
                        const distributedCPUVec<double> &src,
                        const double                     omega) const = 0;

    /**
     * @brief distribute x to the constrained nodes.
     *
     */
    virtual void
    distributeX() = 0;

    /// function needed by dealii to mimic SparseMatrix for Jacobi
    /// preconditioning
    virtual void
    subscribe(std::atomic<bool> *const validity,
              const std::string &      identifier = "") const = 0;

    /// function needed by dealii to mimic SparseMatrix for Jacobi
    /// preconditioning
    virtual void
    unsubscribe(std::atomic<bool> *const validity,
                const std::string &      identifier = "") const = 0;

    /// function needed by dealii to mimic SparseMatrix
    virtual bool
    operator!=(double val) const = 0;

    // protected:

    /// typedef declaration needed by dealii
    typedef dealii::types::global_dof_index size_type;
  };

} // namespace dftfe
#endif // dealiiLinearSolverProblem_H_
