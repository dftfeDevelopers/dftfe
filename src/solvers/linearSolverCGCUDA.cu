// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022 The Regents of the University of Michigan and DFT-FE
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
// @author Gourab Panigrahi
//

#include <linearSolverCGCUDA.h>
#include <cudaHelpers.h>

namespace dftfe
{
  // constructor
  linearSolverCGCUDA::linearSolverCGCUDA(const MPI_Comm & mpi_comm_parent,
                                         const MPI_Comm & mpi_comm_domain,
                                         const solverType type)
    : d_mpiCommParent(mpi_comm_parent)
    , mpi_communicator(mpi_comm_domain)
    , d_type(type)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_domain))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_domain))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {}


  // solve
  void
  linearSolverCGCUDA::solve(linearSolverProblemCUDA &problem,
                            const double             absTolerance,
                            const unsigned int       maxNumberIterations,
                            cublasHandle_t &         cublasHandle,
                            const unsigned int       debugLevel,
                            bool                     distributeFlag)
  {
    int this_process;
    MPI_Comm_rank(mpi_communicator, &this_process);
    MPI_Barrier(mpi_communicator);
    double start_time = MPI_Wtime();
    double time;

    // compute RHS
    distributedCPUVec<double> rhs_host;
    problem.computeRhs(rhs_host);

    distributedGPUVec<double> rhs_device;
    rhs_device.reinit(rhs_host.get_partitioner(), 1);
    cudaUtils::copyHostVecToCUDAVec<double>(rhs_host.begin(),
                                            rhs_device.begin(),
                                            rhs_device.locallyOwnedDofsSize());

    MPI_Barrier(mpi_communicator);
    time = MPI_Wtime();

    if (debugLevel >= 4)
      pcout << "Time for compute rhs_host: " << time - start_time << std::endl;

    bool conv = false;

    distributedGPUVec<double> &x = problem.getX();

    int d_xLenLocalDof = x.locallyOwnedDofsSize();

    double res = 0.0, initial_res = 0.0;
    int    it = 0;

    double start_GPU, end_GPU;

    try
      {
        x.updateGhostValues();

        if (d_type == CG)
          {
            // resize the vectors, but do not set the values since they'd be
            // overwritten soon anyway.
            gvec.reinit(x);
            dvec.reinit(x);
            hvec.reinit(x);

            gvec.zeroOutGhosts();
            dvec.zeroOutGhosts();
            hvec.zeroOutGhosts();

            double gh    = 0.0;
            double beta  = 0.0;
            double alpha = 0.0;

            MPI_Barrier(mpi_communicator);
            cudaDeviceSynchronize();
            start_GPU = MPI_Wtime();

            problem.computeAX(gvec, x);

            cudaUtils::add<double>(gvec.begin(),
                                   rhs_device.begin(),
                                   -1.,
                                   d_xLenLocalDof,
                                   cublasHandle);

            res = cudaUtils::l2_norm<double>(gvec.begin(),
                                             d_xLenLocalDof,
                                             mpi_communicator,
                                             cublasHandle);

            initial_res = res;

            if (res < absTolerance)
              conv = true;
            if (conv)
              return;

            while ((!conv) && (it < maxNumberIterations))
              {
                it++;

                if (it > 1)
                  {
                    beta = gh;
                    AssertThrow(std::abs(beta) != 0.,
                                dealii::ExcMessage("Division by zero\n"));
                    gh   = problem.cg(hvec.begin(), gvec.begin());
                    beta = gh / beta;
                    cudaUtils::sadd<double>(dvec.begin(),
                                            hvec.begin(),
                                            beta,
                                            d_xLenLocalDof);
                  }
                else
                  {
                    gh = problem.cg2(hvec.begin(), gvec.begin(), dvec.begin());
                  }

                problem.computeAX(hvec, dvec);

                alpha = cudaUtils::dot<double>(dvec.begin(),
                                               hvec.begin(),
                                               d_xLenLocalDof,
                                               mpi_communicator,
                                               cublasHandle);

                AssertThrow(std::abs(alpha) != 0.,
                            dealii::ExcMessage("Division by zero\n"));
                alpha = gh / alpha;

                res =
                  problem.cg3(hvec.begin(), gvec.begin(), dvec.begin(), alpha);

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

        x.updateGhostValues();

        if (distributeFlag)
          problem.distributeX();

        x.updateGhostValues();

        problem.copyCUDAToHost();

        MPI_Barrier(mpi_communicator);
        end_GPU = MPI_Wtime();
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
        pcout << "GPU Time: " << end_GPU - start_GPU << "\n";
      }

    MPI_Barrier(mpi_communicator);
    time = MPI_Wtime() - time;

    if (debugLevel >= 4)
      pcout << "Time for Poisson/Helmholtz problem CG iterations: " << time
            << std::endl;
  }
} // namespace dftfe
