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

#ifndef LBFGSNonLinearSolver_h
#define LBFGSNonLinearSolver_h


#include "nonLinearSolver.h"
#include "linearAlgebraOperations.h"
namespace dftfe
{
  /**
   * @brief Concrete class implementing Polak-Ribiere-Polyak Conjugate Gradient non-linear
   * algebraic solver.
   *
   * @author Sambit Das
   */
  class LBFGSNonLinearSolver : public nonLinearSolver
  {
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
     * @param trustRadius_maximum Maximum trust region radius.
     * @param trustRadius_initial Initial trust region radius.
     * @param trustRadius_minimum mimimum trust region radius (will reset BFGS).
     */
    LBFGSNonLinearSolver(const bool         usePreconditioner,
                         const double       tolerance,
                         const unsigned int maxNumberIterations,
                         const int          maxNumPastSteps,
                         const unsigned int debugLevel,
                         const MPI_Comm &   mpi_comm_parent,
                         const double       trustRadius_maximum = 0.5,
                         const double       trustRadius_initial = 0.02,
                         const double       trustRadius_minimum = 1.0e-4);

    /**
     * @brief Destructor.
     */
    ~LBFGSNonLinearSolver();

    /**
     * @brief Solve non-linear problem using Polak-Ribiere-Polyak nonlinar conjugate gradient method.
     *
     * @param problem[in] nonlinearSolverProblem object.
     * @param checkpointFileName[in] if string is non-empty, creates checkpoint file
     * named checkpointFileName for every nonlinear iteration. If restart is set
     * to true, checkpointFileName must match the name of the checkpoint file.
     * Empty string will throw an error.
     * @param restart[in] boolean specifying whether this is a restart solve using the checkpoint file
     * specified by checkpointFileName.
     * @return Return value indicating success or failure.
     */
    nonLinearSolver::ReturnValueType
    solve(nonlinearSolverProblem &problem,
          const std::string       checkpointFileName = "",
          const bool              restart            = false);

  private:
    /**
     * @brief Initialize preconditioner.
     */
    void
    initializePreconditioner(nonlinearSolverProblem &problem);

    /**
     * @brief Scale preconditioner.
     */
    void
    scalePreconditioner(nonlinearSolverProblem &problem);
    /**
     * @brief Compute LBFGS step
     */
    void
    computeStep();
    /**
     * @brief Compute Update Vector
     */
    void
    computeUpdateStep();
    /**
     * @brief Update the stored history, damped LBFGS
     */
    void
    updateHistory();
    /**
     * @brief Test if the step satisfies strong Wolfe conditions
     */
    void
    checkWolfe();
    /**
     * @brief Compute trust radius for the step
     */
    void
    computeTrustRadius(nonlinearSolverProblem &problem);

    /**
     * @brief Update solution x -> x + \alpha direction.
     *
     * @param step the update step.
     * @param problem nonlinearSolverProblem object.
     *
     * @return bool true if valid update and false if increment bound exceeded
     *
     */
    bool
    updateSolution(const std::vector<double> &step,
                   nonlinearSolverProblem &   problem);

    /**
     * @brief Create checkpoint file for current state of the cg solver.
     *
     */
    void
    save(const std::string &checkpointFileName);

    /**
     * @brief Load cg solver state from checkpoint file.
     *
     */
    // void
    // load(const std::string &checkpointFileName);

    /// storage for the value and gradient of the nonlinear problem in the
    /// current bfgs step
    std::vector<double> d_gradient, d_value;

    /// storage for the value and gradient of the nonlinear problem evaluated at
    /// the end of the current bfgs step
    std::vector<double> d_gradientNew, d_valueNew;

    /// Storage for the predicted decrease
    double d_predDec;
    /// storage for the update vector computed in the current bfgs step
    std::vector<double> d_deltaX, d_deltaXNew, d_updateVector, d_preconditioner;

    /// storage for number of unknowns to be solved for in the nonlinear problem
    unsigned int d_numberUnknowns;

    /// storage for current bfgs iteration count
    unsigned int d_iter;

    /// Storage for history
    std::deque<std::vector<double>> d_deltaGq, d_deltaXq;
    std::deque<double> d_rhoq;

    const int d_maxNumPastSteps;
    int       d_numPastSteps;
    /// storage for inf norm of gradient
    double d_gradMax, d_normDeltaXnew;

    /// storage for trust region parameters
    double d_trustRadiusInitial, d_trustRadiusMax, d_trustRadiusMin,
      d_trustRadius;

    /// boolean parameter for step accepteance
    bool d_stepAccepted, d_wolfeCurvature, d_wolfeSufficientDec,
      d_wolfeSatisfied;

    /// flag for using the preconditioner
    const bool d_usePreconditioner;

    ///
    bool d_useSingleAtomSolutionsInitialGuess, d_noHistory;

    // parallel objects
    MPI_Comm                   mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#endif // BFGSNonLinearSolver_h
