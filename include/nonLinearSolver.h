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
//
// @author Sambit Das (2018)


#ifndef dft_NonLinearSolver_h
#define dft_NonLinearSolver_h


#include "headers.h"

  //
  // forward declarations
  //
  class solverFunction;

  /**
   * @brief Base class for non-linear algebraic solver.
   */
  class nonLinearSolver {

    //
    // types
    //
  public:
    enum ReturnValueType { SUCCESS = 0,
			   FAILURE,
			   LINESEARCH_FAILED,
			   MAX_ITER_REACHED };
    
    //
    // methods
    //
  public:

    /**
     * @brief Destructor.
     */
    virtual ~nonLinearSolver() = 0;

    /**
     * @brief Solve non-linear algebraic equation.
     *
     * @return Return value indicating success or failure.
     */
     virtual ReturnValueType solve(solverFunction & function) = 0;

  protected:

    /**
     * @brief Constructor.
     *
     */
    nonLinearSolver();


  protected:
    
    /**
     * @brief Get tolerance.
     *
     * @return Value of the tolerance.
     */
    double getTolerance() const;

    /**
     * @brief Get maximum number of iterations.
     *
     * @return Maximum number of iterations.
     */
    int getMaximumNumberIterations() const;

    /**
     * @brief Get debug level.
     *
     * @return Debug level.
     */
    int getDebugLevel() const;  

    
    
    int    d_debugLevel;
    int    d_maxNumberIterations;
    double d_tolerance;

  };


#endif // dft_NonLinearSolver_h
