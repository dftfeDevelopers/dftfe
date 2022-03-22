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

#ifndef nonlinearSolverProblem_H_
#define nonlinearSolverProblem_H_
#include "headers.h"

namespace dftfe
{
  /**
   * @brief Abstract class for solver functions.
   *
   * @author Sambit Das
   */
  class nonlinearSolverProblem
  {
  public:
    /**
     * @brief Constructor.
     */
    nonlinearSolverProblem();

    /**
     * @brief Destructor.
     */
    virtual ~nonlinearSolverProblem() = 0;

    /**
     * @brief Obtain number of unknowns.
     *
     * @return Number of unknowns.
     */
    virtual unsigned int
    getNumberUnknowns() const = 0;

    /**
     * @brief Compute function value (aka energy).
     *
     *
     * @return Function value.
     */
    virtual void
    value(std::vector<double> &functionValue) = 0;


    /**
     * @brief Compute function gradient (aka forces).
     *
     * @param gradient STL vector for gradient values.
     */
    virtual void
    gradient(std::vector<double> &gradient) = 0;

    /**
     * @brief Apply preconditioner to function gradient.
     *
     * @param s STL vector for s=-M^{-1} gradient.
     * @param gradient STL vector for gradient values.
     */
    virtual void
    precondition(std::vector<double> &      s,
                 const std::vector<double> &gradient) const = 0;

    /**
     * @brief Update solution.
     *
     * @param solution Updated solution.
     */
    virtual void
    update(const std::vector<double> &solution,
           const bool                 computeForces      = true,
           const bool useSingleAtomSolutionsInitialGuess = false) = 0;

    /**
     * @brief Obtain current solution.
     *
     * @param solution Space for current solution.
     */
    virtual void
    solution(std::vector<double> &solution) = 0;

    /**
     * @brief For each unknown indicate whether that unknown should
     * be accumulated. This functionality is needed in the case of
     * parallel execution when domain decomposition is
     * employed. Unknowns residing on processor boundary should only
     * be accumulated once when dot products of vertex fields are
     * computed (e.g. residual).
     *
     * @return A vector of int values for each unknown. Value of 1
     * indicates that the unknown should be counted and 0 otherwise.
     */
    virtual std::vector<unsigned int>
    getUnknownCountFlag() const = 0;

    /**
     * @brief create checkpoint for the current state of the problem i.e problem domain and solution.
     *
     */
    virtual void
    save() = 0;
  };

} // namespace dftfe
#endif // nonlinearSolverProblem_H_
