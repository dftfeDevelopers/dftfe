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
   * @brief Concrete class implementing Polak-Ribiere-Polyak Conjugate Gradient non-linear
   * algebraic solver.
   *
   * @author Sambit Das
   */
  class BFGSNonLinearSolver : public nonLinearSolver
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
    BFGSNonLinearSolver(const double       tolerance,
                        const unsigned int maxNumberIterations,
                        const unsigned int debugLevel,
                        const MPI_Comm &   mpi_comm_parent,
                        const double       trustRadius_maximum = 0.8,
                        const unsigned int trustRadius_initial = 0.5,
                        const double       trustRadius_minimum = 1.0e-3);

    /**
     * @brief Destructor.
     */
    ~BFGSNonLinearSolver();

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
     * @brief Initialize Hessian.
     */
    void
    initializeHessian(nonlinearSolverProblem &problem);

    /**
     * @brief Update Hessian.
     */
    void
    updateHessian();

    /**
     * @brief Compute Lambda.
     */
    void
    computeLambda();
    /**
     * @brief Compute Step.
     */
    void
    computeStep();

    /**
     * @brief Compute residual L2-norm.
     *
     * @return Value of the residual L2-norm.
     */
    double
    computeResidualL2Norm() const;

    /**
     * @brief Compute the total number of unknowns in all
     * processors.
     *
     * @return Number of unknowns in all processors.
     */
    unsigned int
    computeTotalNumberUnknowns() const;

    /**
     * @brief Update solution x -> x + \alpha direction.
     *
     * @param alpha Scalar value.
     * @param direction Direction vector.
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

    /// storage for the gradient of the nonlinear problem in the current bfgs
    /// step
    std::vector<double> d_gradient;

    /// storage for the gradient of the nonlinear problem evaluated at the end
    /// of the current bfgs step
    std::vector<double> d_gradientNew;

    /// storage for the update vector computed in the current bfgs step
    std::vector<double> d_deltaX;

    /// storage for number of unknowns to be solved for in the nonlinear problem
    unsigned int d_numberUnknowns;

    /// storage for current bfgs iteration count
    unsigned int d_iter;

    /// storage for the S matrix in RFO framework, initialized to starting
    /// Hessian Guess
    std::vector<double> d_Srfo;

    /// storage for the hessian in current bfgs step
    std::vector<double> d_hessian;

    /// storage for the rfo update parameter
    double d_lambda;

    /// storage for inf norm of gradient
    double d_gradMax;


    /**
     * Storage for vector of flags (0 or 1) with size equal to the size of the
     * solution vector of the nonlinear problem. If the flag value is 1 for an
     * index in the vector, the corresponding entry in the solution vector is
     * allowed to be updated and vice-versa if flag value is 0 for an index.
     */
    std::vector<unsigned int> d_unknownCountFlag;


    /// flag which restarts bfgs if increment to the solution vector
    /// is too small
    bool d_isBFGSRestartDueToSmallRadius;

    ///
    bool d_useSingleAtomSolutionsInitialGuess;

    // parallel objects
    MPI_Comm                   mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#endif // BFGSNonLinearSolver_h
