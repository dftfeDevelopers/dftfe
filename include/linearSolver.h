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


#include <dealiiLinearSolverProblem.h>

#ifndef linearSolver_H_
#  define linearSolver_H_

namespace dftfe
{
  /**
   * @brief Abstract linear solver base class.
   *
   * @author Sambit Das
   */
  class linearSolver
  {
  public:
    /// Constructor
    linearSolver();

    /**
     * @brief Solve linear system, A*x=Rhs
     *
     * @param problem linearSolverProblem object (functor) to compute Rhs and A*x, and preconditioning
     * @param relTolerance Tolerance (relative) required for convergence.
     * @param maxNumberIterations Maximum number of iterations.
     * @param debugLevel Debug output level:
     *                   0 - no debug output
     *                   1 - limited debug output
     *                   2 - all debug output.
     */
    virtual void
    solve(dealiiLinearSolverProblem &problem,
          const double               relTolerance,
          const unsigned int         maxNumberIterations,
          const int                  debugLevel     = 0,
          bool                       distributeFlag = true) = 0;

  private:
  };

} // namespace dftfe
#endif
