//
// File:      CGNonLinearSolver.h
// Package:   dft
//
// Density Functional Theory
//
#if !defined(dft_CGNonLinearSolver_h)
#define dft_CGNonLinearSolver_h

#if defined(HAVE_CONFIG_H)
#include "dft_config.h"
#endif // HAVE_CONFIG_H

#include "NonLinearSolver.h"

#if defined(HAVE_VECTOR)
#include <vector>
#else
#error vector header file not available
#endif // HAVE_VECTOR

#if defined(HAVE_UTILITY)
#include <utility>
#else 
#error utility header file not available
#endif // HAVE_UTILITY

//
//
//

namespace dft {

  /**
   * @brief Concrete class implementing Conjugate Gradient non-linear
   * algebraic solver.
   */
  class CGNonLinearSolver : public NonLinearSolver {

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
     */
    CGNonLinearSolver(double tolerance,
                      int    maxNumberIterations,
                      int    debugLevel,
                      double lineSearchTolerance = 1.0e-6,
		      int    lineSearchMaxIterations = 6);

    /**
     * @brief Destructor.
     */
    virtual ~CGNonLinearSolver();

    /**
     * @brief Solve non-linear algebraic equation.
     *
     * @param function SolverFunction object (functor) to compute energy and
     * forces.
     * @param tolerance Tolerance (relative) required for convergence.
     * @param maxNumberIterations Maximum number of iterations.
     * @param debugLevel Debug output level:
     *                   0 - no debug output
     *                   1 - limited debug output
     *                   2 - all debug output.
     *
     * @return Return value indicating success or failure.
     */
    virtual NonLinearSolver::ReturnValueType 
    solve(SolverFunction & function,
	  double           tolerance,
	  int              maxNumberIterations,
	  int              debugLevel = 0);

  private:
    //
    // copy constructor/assignment operator
    //
    CGNonLinearSolver(const CGNonLinearSolver &); // not implemented
    CGNonLinearSolver & operator=(const CGNonLinearSolver &); // not implemented


    /**
     * @brief Initialize direction.
     */
    void initializeDirection();
      
    /**
     * @brief Perform line search.
     *
     * @param function SolverFunction object (functor) to compute energy and
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
    NonLinearSolver::ReturnValueType
    lineSearch(SolverFunction & function,
	       double                 tolerance,
	       int                    maxNumberIterations,
	       int                    debugLevel);

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
    int computeTotalNumberUnknowns() const;

    /**
     * @brief Update solution x -> x + \alpha direction.
     *
     * @param alpha Scalar value.
     * @param direction Direction vector.
     * @param function Solver function object.
     */
    void updateSolution(double                      alpha,
			const std::vector<double> & direction,
			SolverFunction            & function);

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
    int                 d_numberUnknowns;
    int                 d_iter;
    std::vector<int>    d_unknownCountFlag;
    double              d_lineSearchTolerance;
    int                 d_lineSearchMaxIterations;

    //
    // data
    //
  private:

  };

}

#endif // dft_CGNonLinearSolver_h
