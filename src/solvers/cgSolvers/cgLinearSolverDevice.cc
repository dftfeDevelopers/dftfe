// ---------------------------------------------------------------------
//
// Copyright (c) 2018-2020 The Regents of the University of Michigan and DFT-FE
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
// @author Phani Motamarri, Sambit Das
//

#if defined(DFTFE_WITH_DEVICE)
#  ifndef densityCalculatorDevice_H_
#    define densityCalculatorDevice_H_

#    include <cgLinearSolverDevice.h>

namespace dftfe
{
  // constructor
  cgLinearSolverDevice::cgLinearSolverDevice(const MPI_Comm &mpi_comm,
                                             cublasHandle_t  cudaBlasHandle)
    : d_cublasHandle(cudaBlasHandle)
    , mpi_communicator(mpi_comm)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm))
    , this_mpi_process(dealii::Utilities::MPI::this_mpi_process(mpi_comm))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(d_mpiCommParent) == 0))
  {}


  // solve
  void
  cgLinearSolverDevice::solve(linearSolverProblemDevice &problem,
                              const double               relTolerance,
                              const unsigned int         maxNumberIterations,
                              const int                  debugLevel)
  {
    // initialize certain variables
    double       delta_new, delta_old, delta_0, alpha, beta, residualNorm;
    double       negOne = -1.0;
    double       posOne = 1.0;
    unsigned int inc    = 1;

    // compute RHS b
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> b;

    double start_timeRhs = MPI_Wtime();
    problem.computeRhs(b);
    double end_timeRhs = MPI_Wtime() - start_timeRhs;

    if (debugLevel >= 2)
      std::cout << " Time for Poisson problem compute rhs: " << end_timeRhs
                << std::endl;

    // get size of vectors
    unsigned int localSize = b.size();


    // get access to initial guess for solving Ax=b
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> &x = problem.getX();
    // x.update_ghost_values();


    // compute Ax
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> Ax;
    Ax.resize(localSize, 0.0);
    problem.computeAX(x, Ax);


    // compute residue r = b - Ax
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> r;
    r.resize(localSize, 0.0);

    // r = b
    cublasDcopy(d_cublasHandle,
                localSize,
                b.begin(), // YArray.begin(),
                inc,
                r.begin(), // XArray.begin(),
                inc);


    // r = b - Ax i.e r - Ax
    cublasDaxpy(d_cublasHandle,
                localSize,
                &negOne,
                Ax.begin(),
                inc,
                r.begin(),
                inc);



    // precondition r
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> d;
    d.resize(localSize, 0.0);
    problem.precondition_Jacobi(r, d);



    // compute delta_new delta_new = r*d;
    cublasDdot(d_cublasHandle,
               localSize,
               r.begin(),
               inc,
               d.begin(),
               inc,
               &delta_new);



    // assign delta0 to delta_new
    delta_0 = delta_new;

    // allocate memory for q
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::DEVICE> q, s;
    q.resize(localSize, 0.0);
    s.resize(localSize, 0.0);


    unsigned int iterationNumber = 0;

    cublasDdot(d_cublasHandle,
               localSize,
               r.begin(),
               inc,
               r.begin(),
               inc,
               &residualNorm);

    if (debugLevel >= 2)
      {
        pcout
          << "Device based Linear Conjugate Gradient solver started with residual norm "
          << std::sqrt(residualNorm) << std::endl;
      }

    double start_time = MPI_Wtime();
    for (unsigned int iter = 0; iter < maxNumberIterations; ++iter)
      {
        // q = Ad
        problem.computeAX(d, q);

        // compute alpha
        double scalar;
        cublasDdot(d_cublasHandle,
                   localSize,
                   d.begin(),
                   inc,
                   q.begin(),
                   inc,
                   &scalar);

        alpha = delta_new / scalar;

        // update x; x = x + alpha*d
        cublasDaxpy(d_cublasHandle,
                    localSize,
                    &alpha,
                    d.begin(),
                    inc,
                    x.begin(),
                    inc);

        if (iter % 50 == 0)
          {
            // r = b
            cublasDcopy(d_cublasHandle,
                        localSize,
                        b.begin(), // YArray.begin(),
                        inc,
                        r.begin(), // XArray.begin(),
                        inc);

            problem.computeAX(x, Ax);

            cublasDaxpy(d_cublasHandle,
                        localSize,
                        &negOne,
                        Ax.begin(),
                        inc,
                        r.begin(),
                        inc);
          }
        else
          {
            double negAlpha = -alpha;
            cublasDaxpy(d_cublasHandle,
                        localSize,
                        &negAlpha,
                        q.begin(),
                        inc,
                        r.begin(),
                        inc);
          }

        problem.precondition_Jacobi(r, s);
        delta_old = delta_new;

        // delta_new = r*s;
        cublasDdot(d_cublasHandle,
                   localSize,
                   r.begin(),
                   inc,
                   s.begin(),
                   inc,
                   &delta_new);


        beta = delta_new / delta_old;

        // d *= beta;
        cublasDscal(d_cublasHandle,
                    localSize,
                    &beta,
                    d.begin(),
                    inc);

        // d.add(1.0,s);
        cublasDaxpy(d_cublasHandle,
                    localSize,
                    &posOne,
                    s.begin(),
                    inc,
                    d.begin(),
                    inc);

        unsigned int isBreak = 0;
        if (delta_new < relTolerance * relTolerance * delta_0)
          isBreak = 1;

        if (isBreak == 1)
          break;

        iterationNumber += 1;
      }



    // compute residual norm at end
    cublasDdot(d_cublasHandle,
               localSize,
               r.begin(),
               inc,
               r.begin(),
               inc,
               &residualNorm);

    residualNorm = std::sqrt(residualNorm);

    //
    // set error condition
    //
    unsigned int solveStatus = 1;

    if (iterationNumber == maxNumberIterations)
      solveStatus = 0;


    if (debugLevel >= 2)
      {
        if (solveStatus == 1)
          {
            pcout << "Linear Conjugate Gradient solver converged after "
                  << iterationNumber + 1 << " iterations. with residual norm "
                  << std::sqrt(residualNorm) << std::endl;
          }
        else
          {
            pcout
              << "Linear Conjugate Gradient solver failed to converge after "
              << iterationNumber << " iterations. with residual norm "
              << std::sqrt(residualNorm) << std::endl;
          }
      }


    problem.setX();

    double device_time = MPI_Wtime() - start_time;
    if (debugLevel >= 2)
      std::cout << "Time for Poisson problem iterations: " << device_time
                << std::endl;
  }

} // namespace dftfe

#  endif
#endif
