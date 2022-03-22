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


#if defined(DFTFE_WITH_GPU)
#  ifndef linearSolverCUDA_H_
#    define linearSolverCUDA_H_

#    include <linearSolverProblemCUDA.h>

namespace dftfe
{
  /**
   * @brief Abstract linear solver base class.
   *
   * @author Sambit Das
   */
  class linearSolverCUDA
  {
  public:
    /// Constructor
    linearSolverCUDA();

    /**
     * @brief Solve linear system, A*x=Rhs
     *
     * @param problem linearSolverCUDAProblem object (functor) to compute Rhs and A*x, and preconditioning
     * @param relTolerance Tolerance (relative) required for convergence.
     * @param maxNumberIterations Maximum number of iterations.
     * @param debugLevel Debug output level:
     *                   0 - no debug output
     *                   1 - limited debug output
     *                   2 - all debug output.
     */
    virtual void
    solve(linearSolverProblemCUDA &problem,
          const double             relTolerance,
          const unsigned int       maxNumberIterations,
          const unsigned int       debugLevel = 0) = 0;

  private:
  };

} // namespace dftfe
#  endif
#endif
