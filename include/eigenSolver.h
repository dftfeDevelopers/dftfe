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


#ifndef eigenSolver_h
#define eigenSolver_h

#include "headers.h"
#include "operator.h"

namespace dftfe
{
  /**
   * @brief Base class for non-linear algebraic solver.
   *
   * @author Phani Motamarri
   */

  class eigenSolverClass
  {
  public:
    enum class ReturnValueType
    {
      SUCCESS = 0,
      FAILURE,
      MAX_ITER_REACHED
    };


  public:
    /**
     * @brief Destructor.
     */
    virtual ~eigenSolverClass() = 0;


    /**
     * @brief Solve eigen problem.
     *
     * @return Return value indicating success or failure.
     */
    virtual void
    solve(operatorDFTClass &                      operatorMatrix,
          std::vector<distributedCPUVec<double>> &eigenVectors,
          std::vector<double> &                   eigenValues,
          std::vector<double> &                   residuals) = 0;

  protected:
    /**
     * @brief Constructor.
     *
     */
    eigenSolverClass();
  };
} // namespace dftfe
#endif // dft_eigenSolver_h
