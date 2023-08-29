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
   * @brief Class implementing LBFGS optimzation method.
   *
   * @author Nikhil Kodali
   */
  class LBFGSNonLinearSolver : public nonLinearSolver
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param usePreconditioner Boolean parameter specifying whether or not to use the preconditioner.
     * @param maxUpdate Maximum allowed step length in any direction.
     * @param maxNumberIterations Maximum number of iterations.
     * @param maxNumPastSteps Number of previous steps stored by the LBFGS solver.
     * @param debugLevel Debug output level:
     *                   0 - no debug output
     *                   1 - limited debug output
     *                   2 - all debug output.
     * @param mpi_comm_parent The mpi communicator used.
     */
    LBFGSNonLinearSolver(
      const bool         usePreconditioner,
      const double       maxUpdate,
      const unsigned int maxNumberIterations,
      const int          maxNumPastSteps,
      const unsigned int debugLevel,
      const MPI_Comm &   mpi_comm_parent,
      const bool         isCurvatureOnlyLineSearchStoppingCondition = false);

    /**
     * @brief Destructor.
     */
    ~LBFGSNonLinearSolver();

    /**
     * @brief Solve non-linear problem using LBFGS method.
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



    /**
     * @brief Create checkpoint file for current state of the LBFGS solver.
     *
     */
    void
    save(const std::string &checkpointFileName);


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
     * @brief Compute Hessian inverse times vector.
     */
    void
    computeHx(std::vector<double> &Hx);

    /**
     * @brief Compute LBFGS step.
     */
    void
    computeStep();

    /**
     * @brief Compute Update Vector.
     */
    void
    computeUpdateStep();

    /**
     * @brief Update the stored history, damped LBFGS.
     */
    void
    updateHistory();

    /**
     * @brief Test if the step satisfies strong Wolfe conditions.
     */
    void
    checkWolfe();

    /**
     * @brief Compute scaling factor for the step.
     */
    void
    computeStepScale(nonlinearSolverProblem &problem);

    /**
     * @brief Update solution x -> x + step.
     *
     * @param step update step vector.
     * @param problem nonlinearSolverProblem object.
     *
     * @return bool true if valid update and false if increment bound exceeded
     *
     */
    bool
    updateSolution(const std::vector<double> &step,
                   nonlinearSolverProblem &   problem);

    /**
     * @brief Load LBFGS solver state from checkpoint file.
     *
     */
    void
    load(const std::string &checkpointFileName);

    /// storage for the value and gradient of the nonlinear problem in the
    /// current bfgs step
    std::vector<double> d_gradient, d_value;

    /// storage for the value and gradient of the nonlinear problem evaluated at
    /// the end of the current bfgs step
    std::vector<double> d_gradientNew, d_valueNew;

    /// storage for the step taken in last BFGS step, step computed in the
    /// corrent BFGS step and the update vector computed in the current bfgs
    /// step.
    std::vector<double> d_deltaX, d_deltaXNew, d_updateVector;

    /// storage for the preconditioner.
    std::vector<double> d_preconditioner;

    /// storage for number of unknowns to be solved for in the nonlinear
    /// problem.
    unsigned int d_numberUnknowns;

    /// storage for current bfgs iteration count.
    unsigned int d_iter;

    /// storage for history.
    std::deque<std::vector<double>> d_deltaGq, d_deltaXq;
    std::deque<double>              d_rhoq;

    /// storage for the maximum number of past steps to be stored.
    const int d_maxNumPastSteps;

    /// storage for the number of past steps currently stored.
    int d_numPastSteps;

    /// storage for inf norm of the update step.
    double d_normDeltaXnew;

    /// storage for the maximum allowed step length.
    double d_maxStepLength;

    /// storage for backtracking line search parameter.
    double d_alpha;

    /// boolean parameter for step accepteance and Wolfe conditions.
    bool d_stepAccepted, d_wolfeCurvature, d_wolfeSufficientDec,
      d_wolfeSatisfied;

    /// flag for using the preconditioner
    const bool d_usePreconditioner;

    bool d_useSingleAtomSolutionsInitialGuess, d_noHistory;

    //
    bool d_isCurvatureOnlyLineSearchStoppingCondition;



    // parallel objects
    MPI_Comm                   mpi_communicator;
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#endif // BFGSNonLinearSolver_h
