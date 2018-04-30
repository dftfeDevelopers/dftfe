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

#ifndef CGPRPNonLinearSolver_h
#define CGPRPNonLinearSolver_h


#include "nonLinearSolver.h"

namespace dftfe {
  /**
   * @brief Concrete class implementing PRP Conjugate Gradient non-linear
   * algebraic solver.
   */
  class cgPRPNonLinearSolver : public nonLinearSolver {

    //
    // types
    //
  public:

    //
    // methods
    //
  public:

    /**
     * @brief Constructor.
     *
     * @param tolerance Tolerance (relative) required for convergence.
     * @param maxNumberIterations Maximum number of iterations.
     * @param debugLevel Debug output level:
     *                   0 - no debug output
     *                   1 - limited debug output
     *                   2 - all debug output.
     * @param lineSearchTolerance Tolereance required for line search
     * convergence.
     * @param lineSearchMaxIterations Maximum number of iterations for the
     * line search.
     * @param lineSearchDampingParameter scales the initial line search step
     */
    cgPRPNonLinearSolver(const double tolerance,
                         const unsigned int    maxNumberIterations,
                         const unsigned int    debugLevel,
		         const MPI_Comm &mpi_comm_replica,
                         const double lineSearchTolerance = 1.0e-6,
		         const unsigned int    lineSearchMaxIterations = 10,
		         const double lineSeachDampingParameter=1.0);

    /**
     * @brief Destructor.
     */
    ~cgPRPNonLinearSolver();

    /**
     * @brief Solve non-linear algebraic equation.
     *
     * @return Return value indicating success or failure.
     */
     nonLinearSolver::ReturnValueType
     solve(nonlinearSolverProblem & problem);

  private:
    /**
     * @brief Initialize direction.
     */
    void initializeDirection();

    /**
     * @brief Perform line search.
     *
     * @param problem nonlinearSolverProblem object (functor) to compute energy and
     *                 forces.
     * @param tolerance Tolerance (relative) required for convergence.
     * @param maxNumberIterations Maximum number of iterations.
     * @param debugLevel Debug output level:
     *                   0 - no debug output
     *                   1 - limited debug output
     *                   2 - all debug output.
     *
     * @return Return value indicating success or failure.
     */
    nonLinearSolver::ReturnValueType
    lineSearch(nonlinearSolverProblem & problem,
	       const double                 tolerance,
	       const unsigned int           maxNumberIterations,
	       const unsigned int           debugLevel);

    /**
     * @brief Compute delta_d and eta_p.
     *
     & @return Pair containing delta_d and eta_p.
    */
    std::pair<double, double> computeDeltaD();

    /**
     * @brief Compute eta.
     *
     * @return Value of eta.
     */
    double computeEta();

    /**
     * @brief Compute delta new and delta mid.
     */
    void computeDeltas();

    /**
     * @brief Update direction.
     */
    void updateDirection();

    /**
     * @brief Compute residual L2-norm.
     *
     * @return Value of the residual L2-norm.
     */
    double computeResidualL2Norm() const;

    /**
     * @brief Compute the total number of unknowns in all
     * processors.
     *
     * @return Number of unknowns in all processors.
     */
    unsigned int computeTotalNumberUnknowns() const;

    /**
     * @brief Update solution x -> x + \alpha direction.
     *
     * @param alpha Scalar value.
     * @param direction Direction vector.
     * @param problem nonlinearSolverProblem object.
     */
    void updateSolution(const double                alpha,
			const std::vector<double> & direction,
			nonlinearSolverProblem    & problem);

    //
    // data
    //
  private:
    std::vector<double> d_solution;
    std::vector<double> d_direction;
    std::vector<double> d_gradient;
    std::vector<double> d_s;
    std::vector<double> d_sOld;
    double              d_deltaNew;
    double              d_deltaMid;
    double              d_deltaOld;
    double              d_beta;
    unsigned int                 d_numberUnknowns;
    unsigned int                 d_iter;
    std::vector<unsigned int>    d_unknownCountFlag;
    const double              d_lineSearchTolerance;
    const unsigned int        d_lineSearchMaxIterations;
    const double              d_lineSearchDampingParameter;

    //
    // data
    //
  private:
    //parallel objects
    MPI_Comm mpi_communicator;
    const unsigned int n_mpi_processes;
    const unsigned int this_mpi_process;
    dealii::ConditionalOStream   pcout;
  };

}
#endif // CGPRPNonLinearSolver_h
