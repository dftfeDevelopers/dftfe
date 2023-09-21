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

#ifndef BFGSNonLinearSolver_h
#define BFGSNonLinearSolver_h


#include "nonLinearSolver.h"
#include "linearAlgebraOperations.h"
namespace dftfe
{
  /**
   * @brief Class implementing a modified BFGS optimization scheme
   *
   * @author Nikhil Kodali
   */
  class BFGSNonLinearSolver : public nonLinearSolver
  {
  public:
    /**
     * @brief Constructor.
     *
     * @param usePreconditioner Boolean parameter specifying whether or not to use the preconditioner.
     * @param useRFOStep Boolean parameter specifying whether or not the RFO step is used.
     * @param maxNumberIterations Maximum number of iterations.
     * @param debugLevel Debug output level:
     *                   0 - no debug output
     *                   1 - limited debug output
     *                   2 - all debug output.
     * @param mpi_comm_parent The mpi communicator used.
     * @param trustRadius_maximum Maximum trust region radius.
     * @param trustRadius_initial Initial trust region radius.
     * @param trustRadius_minimum mimimum trust region radius (will reset BFGS).
     */
    BFGSNonLinearSolver(
      const bool         usePreconditioner,
      const bool         useRFOStep,
      const unsigned int maxNumberIterations,
      const unsigned int debugLevel,
      const MPI_Comm &   mpi_comm_parent,
      const double       trustRadius_maximum                        = 0.5,
      const double       trustRadius_initial                        = 0.02,
      const double       trustRadius_minimum                        = 1.0e-4,
      const bool         isCurvatureOnlyLineSearchStoppingCondition = false);

    /**
     * @brief Destructor.
     */
    ~BFGSNonLinearSolver();

    /**
     * @brief Solve non-linear problem using a modified BFGS method.
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
     * @brief Create checkpoint file for current state of the BFGS solver.
     *
     */
    void
    save(const std::string &checkpointFileName);

  private:
    /**
     * @brief initialize hessian, either preconditioner or identity matrix.
     */
    void
    initializeHessian(nonlinearSolverProblem &problem);

    /**
     * @brief Update Hessian according to damped BFGS rule: Procedure 18.2 of Nocedal and Wright.
     */
    void
    updateHessian();

    /**
     * @brief Scale hessian according to eqn 6.20 of Nocedal and Wright.
     */
    void
    scaleHessian();

    /**
     * @brief Check if the step satifies the Strong Wolfe conditons.
     */
    void
    checkWolfe();

    /**
     * @brief Compute step using the Rational Function Method.
     */
    void
    computeRFOStep();

    /**
     * @brief Compute the Quasi-Newton Step.
     */
    void
    computeNewtonStep();

    /**
     * @brief Compute the final update step using the trust radius and whether or not the previous step was accepted.
     */
    void
    computeStep();

    /**
     * @brief Estimate the trust radius for the next step based on the previous step and check for trust radius max/min conditons and reset BFGS if needed.
     */
    void
    computeTrustRadius(nonlinearSolverProblem &problem);


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
     * @brief Load BFGS solver state from checkpoint file.
     *
     */
    void
    load(const std::string &checkpointFileName);

    /// storage for the value and gradient of the nonlinear problem in the
    /// current bfgs step.
    std::vector<double> d_gradient, d_value;

    /// storage for the value and gradient of the nonlinear problem evaluated at
    /// the end of the current bfgs step.
    std::vector<double> d_gradientNew, d_valueNew;

    /// storage for the step taken in last BFGS step, step computed in the
    /// corrent BFGS step and the update vector computed in the current bfgs
    /// step.
    std::vector<double> d_deltaX, d_deltaXNew, d_updateVector;

    /// storage for number of unknowns to be solved for in the nonlinear
    /// problem.
    unsigned int d_numberUnknowns;

    /// storage for current bfgs iteration count
    unsigned int d_iter;

    /// storage for the S matrix in RFO framework, initialized to starting.
    /// Hessian Guess
    std::vector<double> d_Srfo;

    /// storage for the hessian in current bfgs step.
    std::vector<double> d_hessian;

    /// storage for inf norm of the update step.
    double d_normDeltaXnew;

    /// storage for trust region parameters.
    double d_trustRadiusInitial, d_trustRadiusMax, d_trustRadiusMin,
      d_trustRadius;

    /// boolean parameter for step accepteance and Wolfe conditions.
    bool d_stepAccepted, d_wolfeCurvature, d_wolfeSufficientDec,
      d_wolfeSatisfied;

    ///
    /// flag to check if hessian is scaled.
    ///
    bool d_hessianScaled;

    //
    bool d_isCurvatureOnlyLineSearchStoppingCondition;

    /// Flag to store the reset state, 0 if step is accepted, 1 if reset occured
    /// and no steps are accepted, 2 if two resets occur without step being
    /// accepted (failure of BFGS).
    int d_isReset;

    bool       d_useSingleAtomSolutionsInitialGuess;
    const bool d_useRFOStep, d_usePreconditioner;

    // parallel objects
    MPI_Comm                   mpi_communicator;
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#endif // BFGSNonLinearSolver_h
