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

#if defined(DFTFE_WITH_GPU)
#  ifndef densityCalculatorCUDA_H_
#    define densityCalculatorCUDA_H_

#    include <cgLinearSolverCUDA.h>

namespace dftfe
{
  // constructor
  cgLinearSolverCUDA::cgLinearSolverCUDA(const MPI_Comm &mpi_comm,
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
  cgLinearSolverCUDA::solve(linearSolverProblemCUDA &problem,
                            const double             relTolerance,
                            const unsigned int       maxNumberIterations,
                            const int       debugLevel)
  {
    // initialize certain variables
    double       delta_new, delta_old, delta_0, alpha, beta, residualNorm;
    double       negOne = -1.0;
    double       posOne = 1.0;
    unsigned int inc    = 1;

    // compute RHS b
    thrust::device_vector<double> b;

    double start_timeRhs = MPI_Wtime();
    problem.computeRhs(b);
    double end_timeRhs = MPI_Wtime() - start_timeRhs;

    if (debugLevel >= 2)
      std::cout << " Time for Poisson problem compute rhs: " << end_timeRhs
                << std::endl;

    // get size of vectors
    unsigned int localSize = b.size();


    // get access to initial guess for solving Ax=b
    thrust::device_vector<double> &x = problem.getX();
    // x.update_ghost_values();


    // compute Ax
    thrust::device_vector<double> Ax;
    Ax.resize(localSize, 0.0);
    problem.computeAX(x, Ax);


    // compute residue r = b - Ax
    thrust::device_vector<double> r;
    r.resize(localSize, 0.0);

    // r = b
    cublasDcopy(d_cublasHandle,
                localSize,
                thrust::raw_pointer_cast(&b[0]), // YArray.begin(),
                inc,
                thrust::raw_pointer_cast(&r[0]), // XArray.begin(),
                inc);


    // r = b - Ax i.e r - Ax
    cublasDaxpy(d_cublasHandle,
                localSize,
                &negOne,
                thrust::raw_pointer_cast(&Ax[0]),
                inc,
                thrust::raw_pointer_cast(&r[0]),
                inc);



    // precondition r
    thrust::device_vector<double> d;
    d.resize(localSize, 0.0);
    problem.precondition_Jacobi(r, d);



    // compute delta_new delta_new = r*d;
    cublasDdot(d_cublasHandle,
               localSize,
               thrust::raw_pointer_cast(&r[0]),
               inc,
               thrust::raw_pointer_cast(&d[0]),
               inc,
               &delta_new);



    // assign delta0 to delta_new
    delta_0 = delta_new;

    // allocate memory for q
    thrust::device_vector<double> q, s;
    q.resize(localSize, 0.0);
    s.resize(localSize, 0.0);


    unsigned int iterationNumber = 0;

    cublasDdot(d_cublasHandle,
               localSize,
               thrust::raw_pointer_cast(&r[0]),
               inc,
               thrust::raw_pointer_cast(&r[0]),
               inc,
               &residualNorm);

    if (debugLevel >= 2)
      {
        pcout
          << "GPU based Linear Conjugate Gradient solver started with residual norm "
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
                   thrust::raw_pointer_cast(&d[0]),
                   inc,
                   thrust::raw_pointer_cast(&q[0]),
                   inc,
                   &scalar);

        alpha = delta_new / scalar;

        // update x; x = x + alpha*d
        cublasDaxpy(d_cublasHandle,
                    localSize,
                    &alpha,
                    thrust::raw_pointer_cast(&d[0]),
                    inc,
                    thrust::raw_pointer_cast(&x[0]),
                    inc);

        if (iter % 50 == 0)
          {
            // r = b
            cublasDcopy(d_cublasHandle,
                        localSize,
                        thrust::raw_pointer_cast(&b[0]), // YArray.begin(),
                        inc,
                        thrust::raw_pointer_cast(&r[0]), // XArray.begin(),
                        inc);

            problem.computeAX(x, Ax);

            cublasDaxpy(d_cublasHandle,
                        localSize,
                        &negOne,
                        thrust::raw_pointer_cast(&Ax[0]),
                        inc,
                        thrust::raw_pointer_cast(&r[0]),
                        inc);
          }
        else
          {
            double negAlpha = -alpha;
            cublasDaxpy(d_cublasHandle,
                        localSize,
                        &negAlpha,
                        thrust::raw_pointer_cast(&q[0]),
                        inc,
                        thrust::raw_pointer_cast(&r[0]),
                        inc);
          }

        problem.precondition_Jacobi(r, s);
        delta_old = delta_new;

        // delta_new = r*s;
        cublasDdot(d_cublasHandle,
                   localSize,
                   thrust::raw_pointer_cast(&r[0]),
                   inc,
                   thrust::raw_pointer_cast(&s[0]),
                   inc,
                   &delta_new);


        beta = delta_new / delta_old;

        // d *= beta;
        cublasDscal(d_cublasHandle,
                    localSize,
                    &beta,
                    thrust::raw_pointer_cast(&d[0]),
                    inc);

        // d.add(1.0,s);
        cublasDaxpy(d_cublasHandle,
                    localSize,
                    &posOne,
                    thrust::raw_pointer_cast(&s[0]),
                    inc,
                    thrust::raw_pointer_cast(&d[0]),
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
               thrust::raw_pointer_cast(&r[0]),
               inc,
               thrust::raw_pointer_cast(&r[0]),
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

    double gpu_time = MPI_Wtime() - start_time;
    if (debugLevel >= 2)
      std::cout << "Time for Poisson problem iterations: " << gpu_time
                << std::endl;
  }

} // namespace dftfe

#  endif
#endif
