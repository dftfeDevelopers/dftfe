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

#ifndef CGPRPNonLinearSolver_h
#define CGPRPNonLinearSolver_h


#include "nonLinearSolver.h"

namespace dftfe
{
  /**
   * @brief Concrete class implementing Polak-Ribiere-Polyak Conjugate Gradient non-linear
   * algebraic solver.
   *
   * @author Sambit Das
   */
  class cgPRPNonLinearSolver : public nonLinearSolver
  {
  public:
    /**
     * @brief Constructor.
     *
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
    cgPRPNonLinearSolver(
      const unsigned int maxNumberIterations,
      const unsigned int debugLevel,
      const MPI_Comm &   mpi_comm_parent,
      const double       lineSearchTolerance                        = 1.0e-6,
      const unsigned int lineSearchMaxIterations                    = 10,
      const double       lineSeachDampingParameter                  = 1.0,
      const double       maxIncrementSolLinf                        = 1e+6,
      const bool         isCurvatureOnlyLineSearchStoppingCondition = false);

    /**
     * @brief Destructor.
     */
    ~cgPRPNonLinearSolver();

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


    /**
     * @brief Create checkpoint file for current state of the cg solver.
     *
     */
    void
    save(const std::string &checkpointFileName);


  private:
    /**
     * @brief Initialize direction.
     */
    void
    initializeDirection();

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
    lineSearch(nonlinearSolverProblem &problem,
               const double            tolerance,
               const unsigned int      maxNumberIterations,
               const unsigned int      debugLevel,
               const std::string       checkpointFileName  = "",
               const int               startingIter        = -1,
               const bool              isCheckpointRestart = false);

    /**
     * @brief Compute delta_d and eta_p.
     *
     * @return Pair containing delta_d and eta_p.
     */
    std::pair<double, double>
    computeDeltaD();

    /**
     * @brief Compute eta.
     *
     * @return Value of eta.
     */
    double
    computeEta();

    /**
     * @brief Compute delta new and delta mid.
     */
    void
    computeDeltas();

    /**
     * @brief Update direction.
     */
    void
    updateDirection();

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
    updateSolution(const double               alpha,
                   const std::vector<double> &direction,
                   nonlinearSolverProblem &   problem);

    /**
     * @brief Load cg solver state from checkpoint file.
     *
     */
    void
    load(const std::string &checkpointFileName);

    /// storage for conjugate direction
    std::vector<double> d_conjugateDirection;

    /// storage for the gradient of the nonlinear problem in the current cg step
    std::vector<double> d_gradient;

    /// storage for the steepest descent direction of the nonlinear problem in
    /// the previous cg step
    std::vector<double> d_steepestDirectionOld;

    /// intermediate variable for beta computation
    double d_deltaNew;

    /// intermediate variable for beta computation
    double d_deltaMid;

    /// intermediate variable for beta computation
    double d_deltaOld;

    /// storage for beta- the parameter for updating the conjugate direction
    /// d_beta = (d_deltaNew - d_deltaMid)/d_deltaOld
    double d_beta;

    ///
    double d_gradMax;

    /// storage for number of unknowns to be solved for in the nonlinear problem
    unsigned int d_numberUnknowns;

    /// storage for current nonlinear cg iteration count
    unsigned int d_iter;

    /**
     * Storage for vector of flags (0 or 1) with size equal to the size of the
     * solution vector of the nonlinear problem. If the flag value is 1 for an
     * index in the vector, the corresponding entry in the solution vector is
     * allowed to be updated and vice-versa if flag value is 0 for an index.
     */
    std::vector<unsigned int> d_unknownCountFlag;

    /// line search stopping tolerance
    const double d_lineSearchTolerance;

    /// maximum number of line search iterations
    const unsigned int d_lineSearchMaxIterations;

    /// damping parameter (0,1] to be multiplied with the steepest descent
    /// direction, which controls the initial guess to the line search
    /// iteration.
    double d_lineSearchDampingParameter;

    /// flag which restarts the CG if large increment to the solution vector
    /// happens during line search
    bool d_isCGRestartDueToLargeIncrement;

    /// maximum allowed increment (measured as L_{inf}(delta x)) in solution
    /// vector beyond which CG is restarted
    double d_maxSolutionIncrementLinf;

    /// line search data
    double d_alphaChk;

    /// line search data
    double d_etaPChk;

    /// line search data
    double d_etaChk;

    /// line search data
    double d_eta;

    /// line search data
    double d_etaAlphaZeroChk;

    /// line search data
    double d_functionValueChk;

    /// line search data
    double d_functionalValueAfterAlphUpdateChk;

    /// line search iter
    int d_lineSearchRestartIterChk;

    ///
    bool d_useSingleAtomSolutionsInitialGuess;

    //
    bool d_isCurvatureOnlyLineSearchStoppingCondition;

    // parallel objects
    MPI_Comm                   mpi_communicator;
    const unsigned int         n_mpi_processes;
    const unsigned int         this_mpi_process;
    dealii::ConditionalOStream pcout;
  };

} // namespace dftfe
#endif // CGPRPNonLinearSolver_h
