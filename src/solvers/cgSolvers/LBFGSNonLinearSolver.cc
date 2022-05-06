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
// @author Nikhil Kodali

#include <LBFGSNonLinearSolver.h>
#include <fileReaders.h>
#include <nonlinearSolverProblem.h>

namespace dftfe
{
  //
  // Constructor.
  //
  LBFGSNonLinearSolver::LBFGSNonLinearSolver(
    const bool         usePreconditioner,
    const double       tolerance,
    const unsigned int maxNumberIterations,
    const int          maxNumPastSteps,
    const unsigned int debugLevel,
    const MPI_Comm &   mpi_comm_parent,
    const double       trustRadius_maximum,
    const double       trustRadius_initial,
    const double       trustRadius_minimum)
    : nonLinearSolver(debugLevel, maxNumberIterations, tolerance)
    , mpi_communicator(mpi_comm_parent)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_parent))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent))
    , d_maxNumPastSteps(maxNumPastSteps)
    , d_usePreconditioner(usePreconditioner)
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {
    d_useSingleAtomSolutionsInitialGuess = false;
    d_trustRadiusInitial                 = trustRadius_initial;
    d_trustRadiusMax                     = trustRadius_maximum;
    d_trustRadiusMin                     = trustRadius_minimum;
  }

  //
  // Destructor.
  //
  LBFGSNonLinearSolver::~LBFGSNonLinearSolver()
  {
    //
    //
    //
    return;
  }
  namespace internal
  {
    //
    // Compute L2-norm.
    //
    double
    computeL2Norm(std::vector<double> &a)
    {
      const unsigned int one = 1;
      const unsigned int n   = a.size();
      return dnrm2_(&n, a.data(), &one);
    }

    //
    // Compute Weighted L2-norm for symmetric matrix P.
    //
    double
    computePNorm(std::vector<double> &a, std::vector<double> &P)
    {
      const unsigned int one   = 1;
      const char         uplo  = 'U';
      const double       one_d = 1.0;
      const unsigned int n     = a.size();
      if (n * n != P.size())
        {
          std::cout << "DEBUG check dimensions" << std::endl;
          return -1;
        }
      std::vector<double> Pdx(n, 0.0);
      dsymv_(&uplo,
             &n,
             &one_d,
             P.data(),
             &n,
             a.data(),
             &one,
             &one_d,
             Pdx.data(),
             &one);

      return ddot_(&n, a.data(), &one, Pdx.data(), &one);
    }

    //
    // Compute Inf-norm.
    //
    double
    computeLInfNorm(std::vector<double> &vec)
    {
      double norm = 0.0;
      for (unsigned int i = 0; i < vec.size(); ++i)
        {
          norm = norm > std::abs(vec[i]) ? norm : std::abs(vec[i]);
        }
      return norm;
    }

    //
    // Compute dot product.
    //
    double
    dot(std::vector<double> &a, std::vector<double> &b)
    {
      const unsigned int one = 1;
      const unsigned int n   = a.size();
      if (n != b.size())
        {
          std::cout << "DEBUG check dimensions" << std::endl;
          return -1;
        }
      return ddot_(&n, a.data(), &one, b.data(), &one);
    }

    //
    // Compute y=alpha*x+y.
    //
    void
    axpy(double alpha, std::vector<double> &x, std::vector<double> &y)
    {
      const unsigned int one = 1;
      const unsigned int n   = x.size();
      if (n != y.size())
        {
          std::cout << "DEBUG check dimensions" << std::endl;
        }
      daxpy_(&n, &alpha, x.data(), &one, y.data(), &one);
    }

    //
    // Compute lowest n eigenvalues of symmetric matrix A and corresponding
    // eigenvectors.
    //
    void
    computeEigenSpectrum(std::vector<double>  A,
                         const int            n,
                         std::vector<double> &eigVals,
                         std::vector<double> &eigVecs)
    {
      int                 info;
      const int           one             = 1;
      const int           dimensionMatrix = std::sqrt(A.size());
      std::vector<double> eigenValues(n, 0.0);
      std::vector<double> eigenVectors(dimensionMatrix * n, 0.0);

      const int lwork = 8 * dimensionMatrix, liwork = 5 * dimensionMatrix;
      std::vector<int>    iwork(liwork, 0);
      std::vector<int>    ifail(dimensionMatrix, 0);
      const char          jobz = 'V', uplo = 'U', range = 'I', cmach = 'S';
      const double        abstol = 2 * dlamch_(&cmach);
      std::vector<double> work(lwork);
      int                 nEigVals;

      dsyevx_(&jobz,
              &range,
              &uplo,
              &dimensionMatrix,
              A.data(),
              &dimensionMatrix,
              NULL,
              NULL,
              &one,
              &n,
              &abstol,
              &nEigVals,
              eigenValues.data(),
              eigenVectors.data(),
              &dimensionMatrix,
              work.data(),
              &lwork,
              iwork.data(),
              ifail.data(),
              &info);

      //
      // free up memory associated with work
      //
      work.clear();
      iwork.clear();
      std::vector<double>().swap(work);
      std::vector<int>().swap(iwork);

      eigVals = eigenValues;
      eigVecs = eigenVectors;
    }
    //
    //  Solve Ax=b for symmetric Matrix A
    //
    void
    linearSolve(std::vector<double> A, std::vector<double> &b)
    {
      int                 info;
      const int           one   = 1;
      const int           lwork = b.size();
      std::vector<int>    ipiv(b.size(), 0);
      const char          uplo = 'U';
      std::vector<double> work(lwork);

      dsysv_(&uplo,
             &lwork,
             &one,
             A.data(),
             &lwork,
             ipiv.data(),
             b.data(),
             &lwork,
             work.data(),
             &lwork,
             &info);
    }

  } // namespace internal

  //
  // initialize preconditioner
  //
  void
  LBFGSNonLinearSolver::initializePreconditioner(
    nonlinearSolverProblem &problem)
  {
    d_preconditioner.clear();
    problem.precondition(d_preconditioner, d_gradient);
  }

  //
  // Scale preconditioner
  //
  void
  LBFGSNonLinearSolver::scalePreconditioner(nonlinearSolverProblem &problem)
  {
    std::vector<double> testDisplacment, eigenvalue;
    internal::computeEigenSpectrum(d_preconditioner,
                                   1,
                                   eigenvalue,
                                   testDisplacment);

    updateSolution(testDisplacment, problem);
    problem.gradient(d_gradientNew);
    std::vector<double> delta_g(d_numberUnknowns, 0.0);

    for (auto i = 0; i < d_numberUnknowns; ++i)
      {
        delta_g[i] = d_gradientNew[i] - d_gradient[i];
      }

    double mu = internal::dot(delta_g, testDisplacment) /
                internal::computePNorm(testDisplacment, d_preconditioner);
    pcout << "DEBUG mu " << mu << std::endl;
    if (mu > 0)
      {
        for (auto i = 0; i < d_preconditioner.size(); ++i)
          {
            d_preconditioner[i] *= mu;
          }
      }
  }

  //
  // Compute LBFGS step
  //
  void
  LBFGSNonLinearSolver::computeStep()
  {
    std::vector<double> gradient = d_gradient;
    std::vector<double> alpha(d_maxNumPastSteps, 0.0);
    for (int j = d_maxNumPastSteps - 1; j >= 0; --j)
      {
        alpha[j] = internal::dot(d_deltaXq[j], gradient) /
                   internal::dot(d_deltaGq[j], d_deltaXq[j]);
        internal::axpy(-alpha[j], d_deltaGq[j], gradient);
      }
    if (d_usePreconditioner)
      {
        internal::linearSolve(d_preconditioner, gradient);
        for (int i = 0; i < d_numberUnknowns; ++i)
          {
            gradient[i] *= -1;
          }
      }
    else if (d_iter > 0)
      {
        for (int i = 0; i < d_numberUnknowns; ++i)
          {
            gradient[i] *= -internal::dot(d_deltaXq[d_maxNumPastSteps - 1],
                                          d_deltaGq[d_maxNumPastSteps - 1]) /
                           internal::dot(d_deltaGq[d_maxNumPastSteps - 1],
                                         d_deltaGq[d_maxNumPastSteps - 1]);
          }
      }
    for (int j = 0; j < d_maxNumPastSteps; ++j)
      {
        double beta = internal::dot(d_deltaGq[j], gradient) /
                      internal::dot(d_deltaGq[j], d_deltaXq[j]);
        internal::axpy(-alpha[j] - beta, d_deltaXq[j], gradient);
      }
    d_deltaX        = gradient;
    d_normDeltaXnew = internal::computeLInfNorm(d_deltaXNew);
    pcout << "DEBUG LInf dx init " << d_normDeltaXnew << std::endl;
  }

  //
  // Compute Update Vector
  //
  void
  LBFGSNonLinearSolver::computeUpdateStep()
  {
    if (d_iter > 0)
      {
        for (auto i = 0; i < d_numberUnknowns; ++i)
          {
            d_deltaXNew[i] *= d_trustRadius / d_normDeltaXnew;
          }
      }
    pcout << "DEBUG LInf dx scaled " << internal::computeLInfNorm(d_deltaXNew)
          << std::endl;
    pcout << "DEBUG gtdx " << internal::dot(d_deltaXNew, d_gradient)
          << std::endl;
    if (d_stepAccepted)
      {
        d_updateVector = d_deltaXNew;
      }
    else
      {
        for (auto i = 0; i < d_numberUnknowns; ++i)
          {
            d_updateVector[i] = d_deltaXNew[i] - d_deltaX[i];
          }
      }
  }

  //
  // Update the stored history, damped LBFGS
  //
  void
  LBFGSNonLinearSolver::updateHistory()
  {
    double sBs =
      -internal::dot(d_deltaXNew, d_gradient) * d_trustRadius / d_normDeltaXnew;
    std::vector<double> delta_g(d_numberUnknowns, 0.0);

    for (auto i = 0; i < d_numberUnknowns; ++i)
      {
        delta_g[i] = d_gradientNew[i] - d_gradient[i];
      }
    double sy    = internal::dot(delta_g, d_deltaXNew);
    double theta = sy >= 0.2 * sBs ? 1 : 0.8 * sBs / (sBs - sy);
    pcout << "DEBUG Step BFGS theta " << theta << std::endl;
    if (theta != 1)
      {
        pcout << "DEBUG BFGS Damped " << std::endl;
      }
    std::vector<double> r(d_numberUnknowns, 0.0);
    for (auto i = 0; i < d_numberUnknowns; ++i)
      {
        r[i] = theta * delta_g[i] +
               (1.0 - theta) * d_gradient[i] * d_trustRadius / d_normDeltaXnew;
      }
    d_deltaGq.push_back(r);
    d_deltaGq.pop_front();
    d_deltaXq.push_back(d_deltaXNew);
    d_deltaGq.pop_front();
    if (d_numPastSteps < d_maxNumPastSteps)
      {
        ++d_numPastSteps;
      }
  }

  //
  // Test if the step satisfies strong Wolfe conditions
  //
  void
  LBFGSNonLinearSolver::checkWolfe()
  {
    double gtdx  = internal::dot(d_deltaXNew, d_gradient);
    double gntdx = internal::dot(d_deltaXNew, d_gradientNew);

    d_wolfeSufficientDec = (d_valueNew[0] - d_value[0]) < 0.01 * gtdx;
    d_wolfeCurvature     = std::abs(gntdx) < 0.9 * std::abs(gtdx);
    d_wolfeSatisfied     = d_wolfeSufficientDec && d_wolfeCurvature;
    pcout << "DEBUG WOLFE " << d_wolfeCurvature << " " << d_wolfeSufficientDec
          << " " << d_wolfeSatisfied << " " << std::endl;
  }

  //
  // Compute trust radius for the step
  //
  void
  LBFGSNonLinearSolver::computeTrustRadius(nonlinearSolverProblem &problem)
  {
    if (d_iter == 0)
      {
        d_trustRadius =
          d_trustRadius < d_normDeltaXnew ? d_trustRadius : d_normDeltaXnew;
      }
    else if (d_stepAccepted)
      {
        double ampfactor =
          internal::computeLInfNorm(d_deltaX) > d_trustRadius + 1e-8 ? 1.5 :
                                                                       1.1;

        ampfactor     = d_wolfeSatisfied ? 2 * ampfactor : ampfactor;
        d_trustRadius = ampfactor * d_trustRadius < d_normDeltaXnew ?
                          ampfactor * d_trustRadius :
                          d_normDeltaXnew;
        d_trustRadius =
          d_trustRadius < d_trustRadiusMax ? d_trustRadius : d_trustRadiusMax;
        if (d_trustRadius < d_trustRadiusMin)
          {
            pcout << "DEBUG reset history " << d_trustRadius << std::endl;
            if (d_usePreconditioner)
              initializePreconditioner(problem);
            d_trustRadius = d_trustRadiusInitial;
            std::fill(d_deltaGq[d_maxNumPastSteps - d_numPastSteps].begin(),
                      d_deltaGq[d_maxNumPastSteps - d_numPastSteps].end(),
                      0);
            std::fill(d_deltaXq[d_maxNumPastSteps - d_numPastSteps].begin(),
                      d_deltaXq[d_maxNumPastSteps - d_numPastSteps].end(),
                      0);
            --d_numPastSteps;
            computeStep();
            d_trustRadius =
              d_trustRadius < d_normDeltaXnew ? d_trustRadius : d_normDeltaXnew;
          }
      }
    else
      {
        double gtdx = internal::dot(d_deltaX, d_gradient);
        d_trustRadius =
          -0.5 * gtdx * d_trustRadius / ((d_valueNew[0] - d_value[0]) - gtdx);
        if (d_trustRadius < d_trustRadiusMin)
          {
            pcout << "DEBUG reset history " << d_trustRadius << std::endl;
            if (d_usePreconditioner)
              initializePreconditioner(problem);
            d_trustRadius = d_trustRadiusInitial;
            std::fill(d_deltaGq[d_maxNumPastSteps - d_numPastSteps].begin(),
                      d_deltaGq[d_maxNumPastSteps - d_numPastSteps].end(),
                      0);
            std::fill(d_deltaXq[d_maxNumPastSteps - d_numPastSteps].begin(),
                      d_deltaXq[d_maxNumPastSteps - d_numPastSteps].end(),
                      0);
            --d_numPastSteps;
            d_noHistory = d_numPastSteps == 0;
            computeStep();
            d_trustRadius =
              d_trustRadius < d_normDeltaXnew ? d_trustRadius : d_normDeltaXnew;
          }
      }

    pcout << "DEBUG Trust Radius " << d_trustRadius << std::endl;
  }


  //
  // Update solution x -> x + step.
  //
  bool
  LBFGSNonLinearSolver::updateSolution(const std::vector<double> &step,
                                       nonlinearSolverProblem &   problem)
  {
    std::vector<double> incrementVector;

    //
    // get the size of solution
    //
    const std::vector<double>::size_type solutionSize = d_numberUnknowns;
    incrementVector.resize(d_numberUnknowns);


    for (std::vector<double>::size_type i = 0; i < solutionSize; ++i)
      incrementVector[i] = step[i];

    //
    // call solver problem update
    //
    problem.update(incrementVector, true, d_useSingleAtomSolutionsInitialGuess);

    d_useSingleAtomSolutionsInitialGuess = false;
    return true;
  }


  //
  // Perform problem minimization.
  //
  nonLinearSolver::ReturnValueType
  LBFGSNonLinearSolver::solve(nonlinearSolverProblem &problem,
                              const std::string       checkpointFileName,
                              const bool              restart)
  {
    //
    // get total number of unknowns in the problem.
    //
    d_numberUnknowns = problem.getNumberUnknowns();

    //
    // allocate space for step, gradient and new gradient.
    //
    d_updateVector.resize(d_numberUnknowns);
    d_deltaX.resize(d_numberUnknowns);
    d_deltaXNew.resize(d_numberUnknowns);
    d_gradient.resize(d_numberUnknowns);
    d_gradientNew.resize(d_numberUnknowns);
    d_deltaGq.resize(d_maxNumPastSteps);
    d_deltaXq.resize(d_maxNumPastSteps);
    for (int i = 0; i < d_maxNumPastSteps; ++i)
      {
        d_deltaGq[i].resize(d_numberUnknowns);
        d_deltaXq[i].resize(d_numberUnknowns);
      }

    //
    // initialize delta new and direction
    //
    if (!restart)
      {
        d_trustRadius   = d_trustRadiusInitial;
        d_normDeltaXnew = d_trustRadiusInitial;
        d_stepAccepted  = true;
        d_numPastSteps  = 0;
        //
        // compute initial values of problem and problem gradient
        //
        pcout << "DEBUG START LBFGS " << std::endl;
        problem.gradient(d_gradient);
        problem.value(d_value);
        pcout << "DEBUG Compute g0 " << std::endl;

        if (d_usePreconditioner)
          {
            initializePreconditioner(problem);
            scalePreconditioner(problem);
          }

        pcout << "DEBUG Compute H0 " << std::endl;
      }
    else
      // NEED TO UPDATE
      {
        // load(checkpointFileName);
        MPI_Barrier(mpi_communicator);
        d_useSingleAtomSolutionsInitialGuess = true;
      }
    //
    // check for convergence
    //
    unsigned int isSuccess = 0;
    d_gradMax              = internal::computeLInfNorm(d_gradient);

    if (d_gradMax < d_tolerance)
      isSuccess = 1;

    MPI_Bcast(&(isSuccess), 1, MPI_INT, 0, mpi_communicator);
    if (isSuccess == 1)
      return SUCCESS;



    for (d_iter = 0; d_iter < d_maxNumberIterations; ++d_iter)
      {
        pcout << "BFGS Step no. " << d_iter + 1 << std::endl;
        if (d_debugLevel >= 2)
          for (unsigned int i = 0; i < d_gradient.size(); ++i)
            pcout << "d_gradient: " << d_gradient[i] << std::endl;

        // Compute the update step
        //
        pcout << "DEBUG Start Compute step " << std::endl;
        if (d_noHistory)
          {
            break;
          }
        computeStep();
        computeTrustRadius(problem);
        // figure out exit strategy

        computeUpdateStep();

        for (unsigned int i = 0; i < d_deltaXNew.size(); ++i)
          pcout << "step: " << d_deltaXNew[i] << std::endl;
        pcout << "DEBUG End Compute step " << std::endl;
        updateSolution(d_updateVector, problem);
        pcout << "DEBUG End update step " << std::endl;
        //
        // evaluate gradient
        //
        problem.gradient(d_gradientNew);
        problem.value(d_valueNew);
        // check for convergence
        //
        unsigned int isBreak = 0;

        d_gradMax = internal::computeLInfNorm(d_gradientNew);

        if (d_gradMax < d_tolerance)
          isBreak = 1;
        MPI_Bcast(&(isBreak), 1, MPI_INT, 0, mpi_communicator);
        if (isBreak == 1)
          break;

        //
        // update trust radius and hessian
        //
        checkWolfe();
        d_stepAccepted = d_wolfeSufficientDec;
        if (d_stepAccepted)
          {
            updateHistory();
            pcout << "DEBUG step accepted " << d_valueNew[0] - d_value[0]
                  << std::endl;

            d_deltaX   = d_deltaXNew;
            d_value[0] = d_valueNew[0];
            d_gradient = d_gradientNew;
          }
        else
          {
            pcout << "DEBUG step rejected " << d_valueNew[0] - d_value[0]
                  << std::endl;
            d_deltaX = d_deltaXNew;
          }
      }

    //
    // set error condition
    //
    ReturnValueType returnValue = SUCCESS;

    if (d_iter == d_maxNumberIterations)
      returnValue = MAX_ITER_REACHED;
    if (d_numPastSteps == 0)
      returnValue = FAILURE;

    //
    // final output
    //
    if (d_debugLevel >= 1)
      {
        if (returnValue == SUCCESS)
          {
            pcout << "BFGS solver converged after " << d_iter + 1
                  << " iterations." << std::endl;
          }
        else
          {
            pcout << "BFGS solver failed to converge after " << d_iter
                  << " iterations." << std::endl;
          }
      }

    //
    //
    //
    return returnValue;
  }
} // namespace dftfe
