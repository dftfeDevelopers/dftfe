// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE authors.
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


#include <linearSolver.h>

#ifndef dealiiLinearSolver_H_
#define dealiiLinearSolver_H_

typedef dealii::parallel::distributed::Vector<double> vectorType;

namespace dftfe {

    /**
     * @brief dealii linear solver class wrapper
     *
     * @author Sambit Das
     */
    class dealiiLinearSolver : public linearSolver
    {
       public:

	  enum solverType { CG=0, GMRES };

	  /**
	   * @brief Constructor
	   *
	   * @param mpi_comm mpi communicator
	   * @param type enum specifying the choice of the dealii linear solver
	   */
	  dealiiLinearSolver(const  MPI_Comm &mpi_comm,
		               const  solverType type);

	  /**
	   * @brief Solve linear system, A*x=Rhs
	   *
	   * @param problem linearSolverProblem object (functor) to compute Rhs and A*x, and preconditioning
           * @param relTolerance Tolerance (relative) required for convergence.
           * @param maxNumberIterations Maximum number of iterations.
	   * @param debugLevel Debug output level:
	   *                   0 - no debug output
	   *                   1 - limited debug output
	   *                   2 - all debug output.
	   */
	   void solve(dealiiLinearSolverProblem & problem,
		      const double relTolerance,
		      const unsigned int maxNumberIterations,
		      const unsigned int  debugLevel = 0);

       private:

	   /// enum denoting the choice of the dealii solver
           const solverType d_type;

           const MPI_Comm mpi_communicator;
           const unsigned int n_mpi_processes;
           const unsigned int this_mpi_process;
           dealii::ConditionalOStream   pcout;
    };

}

#endif
