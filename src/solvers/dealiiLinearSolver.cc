// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2018 The Regents of the University of Michigan and DFT-FE
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
// @author Sambit Das
//

#include <dealiiLinearSolver.h>
#include <dftParameters.h>

namespace dftfe
{
  // constructor
  dealiiLinearSolver::dealiiLinearSolver(const MPI_Comm & mpi_comm,
                                         const solverType type)
    : mpi_communicator(mpi_comm)
    , d_type(type)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm))
    , this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_comm))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0))
  {}


  // solve
  void
  dealiiLinearSolver::solve(dealiiLinearSolverProblem &problem,
                            const double               absTolerance,
                            const unsigned int         maxNumberIterations,
                            const unsigned int         debugLevel,
                            bool                       distributeFlag)
  {
    int this_process;
    MPI_Comm_rank(mpi_communicator, &this_process);
    MPI_Barrier(mpi_communicator);
    double start_time = MPI_Wtime();
    double time;

    // compute RHS
    distributedCPUVec<double> rhs;
    problem.computeRhs(rhs);

    MPI_Barrier(mpi_communicator);
    time = MPI_Wtime();

    if (dftParameters::verbosity >= 4)
      pcout << "Time for compute rhs: " << time - start_time << std::endl;

    bool conv = false; // false : converged; true : converged

    distributedCPUVec<double> &x = problem.getX();

    double res = 0.0, initial_res = 0.0;
    int    it = 0;

    try
      {
        x.update_ghost_values();

        if (d_type == CG)
          {
            // resize the vectors, but do not set the values since they'd be
            // overwritten soon anyway.
            g.reinit(x, true);
            d.reinit(x, true);
            h.reinit(x, true);

            double gh        = 0.0;
            double beta      = 0.0;
            double alpha     = 0.0;
            double old_alpha = 0.0;
            double omega     = 0.3;

            // compute residual. if vector is zero, then short-circuit the full
            // computation
            if (!x.all_zero())
              {
                problem.vmult(g, x);
                g.add(-1., rhs);
              }
            else
              g.equ(-1., rhs);

            res         = g.l2_norm();
            initial_res = res;
            if (res < absTolerance)
              conv = true;
            if (conv)
              return;

            while ((!conv) && (it < maxNumberIterations))
              {
                it++;
                old_alpha = alpha;

                if (it > 1)
                  {
                    problem.precondition_Jacobi(h, g, omega);
                    beta = gh;
                    AssertThrow(std::abs(beta) != 0.,
                                dealii::ExcMessage("Division by zero\n"));
                    gh   = g * h;
                    beta = gh / beta;
                    d.sadd(beta, -1., h);
                  }
                else
                  {
                    problem.precondition_Jacobi(h, g, omega);
                    d.equ(-1., h);
                    gh = g * h;
                  }

                problem.vmult(h, d);
                alpha = d * h;
                AssertThrow(std::abs(alpha) != 0.,
                            dealii::ExcMessage("Division by zero\n"));
                alpha = gh / alpha;

                x.add(alpha, d);
                res = std::sqrt(std::abs(g.add_and_dot(alpha, h, g)));

                if (res < absTolerance)
                  conv = true;
              }
            if (!conv)
              {
                AssertThrow(false,
                            dealii::ExcMessage(
                              "DFT-FE Error: Solver did not converge\n"));
              }
          }
        else if (d_type == GMRES)
          {
            AssertThrow(false,
                        dealii::ExcMessage("DFT-FE Error: Not implemented"));
          }

        if (distributeFlag)
          problem.distributeX();

        x.update_ghost_values();
      }
    catch (...)
      {
        AssertThrow(
          false,
          dealii::ExcMessage(
            "DFT-FE Error: Poisson solver did not converge as per set tolerances. consider increasing MAXIMUM ITERATIONS in Poisson problem parameters. In rare cases for all-electron problems this can also occur due to a known parallel constraints issue in dealii library. Try using set CONSTRAINTS FROM SERIAL DOFHANDLER=true under the Boundary conditions subsection."));
        pcout
          << "\nWarning: solver did not converge as per set tolerances. consider increasing maxLinearSolverIterations or decreasing relLinearSolverTolerance.\n";
        pcout << "Current abs. residual: " << res << std::endl;
      }

    if (debugLevel >= 2)
      {
        pcout << std::endl;
        pcout << "initial abs. residual: " << initial_res
              << " , current abs. residual: " << res << " , nsteps: " << it
              << " , abs. tolerance criterion:  " << absTolerance << "\n\n";
      }

    MPI_Barrier(mpi_communicator);
    time = MPI_Wtime() - time;

    if (dftParameters::verbosity >= 4)
      pcout << "Time for Poisson/Helmholtz problem CG iterations: " << time
            << std::endl;
  }
} // namespace dftfe
