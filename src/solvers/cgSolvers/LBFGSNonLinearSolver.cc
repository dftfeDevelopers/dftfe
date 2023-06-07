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
    const double       maxUpdate,
    const unsigned int maxNumberIterations,
    const int          maxNumPastSteps,
    const unsigned int debugLevel,
    const MPI_Comm &   mpi_comm_parent,
    const bool         isCurvatureOnlyLineSearchStoppingCondition)
    : nonLinearSolver(debugLevel, maxNumberIterations)
    , d_maxStepLength(maxUpdate)
    , mpi_communicator(mpi_comm_parent)
    , d_maxNumPastSteps(maxNumPastSteps)
    , d_usePreconditioner(usePreconditioner)
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , d_isCurvatureOnlyLineSearchStoppingCondition(
        isCurvatureOnlyLineSearchStoppingCondition)
  {
    d_useSingleAtomSolutionsInitialGuess = false;
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
  namespace internalLBFGS
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
          std::cout << "DEBUG check dimensions Pnorm" << std::endl;
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
    // Compute Weighted inner product for symmetric matrix P.
    //
    double
    computePdot(std::vector<double> &a,
                std::vector<double> &b,
                std::vector<double> &P)
    {
      const unsigned int one   = 1;
      const char         uplo  = 'U';
      const double       one_d = 1.0;
      const unsigned int n     = a.size();
      if (n * n != P.size())
        {
          std::cout << "DEBUG check dimensions Pnorm" << std::endl;
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

      return ddot_(&n, b.data(), &one, Pdx.data(), &one);
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
          std::cout << "DEBUG check dimensions dot" << std::endl;
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
          std::cout << "DEBUG check dimensions axpy" << std::endl;
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

    //
    //  compute |A|^(1/n) (geometric mean of eigenvalues) for a symmetric matrix
    //  A
    //
    double
    computeDetNormalizationFactor(std::vector<double> A)
    {
      int                 info;
      const unsigned int  dimensionMatrix = std::sqrt(A.size());
      std::vector<double> eigenValues(dimensionMatrix, 0.0);
      const unsigned int  lwork = 1 + 2 * dimensionMatrix, liwork = 1;
      std::vector<int>    iwork(liwork, 0);
      const char          jobz = 'N', uplo = 'U';
      std::vector<double> work(lwork);

      dsyevd_(&jobz,
              &uplo,
              &dimensionMatrix,
              A.data(),
              &dimensionMatrix,
              eigenValues.data(),
              &work[0],
              &lwork,
              &iwork[0],
              &liwork,
              &info);

      //
      // free up memory associated with work
      //
      work.clear();
      iwork.clear();
      std::vector<double>().swap(work);
      std::vector<int>().swap(iwork);

      double detA = 1.0;
      for (auto i = 0; i < dimensionMatrix; ++i)
        {
          detA *= std::pow(std::abs(eigenValues[i]), 1.0 / dimensionMatrix);
        }
      return detA;
    }
  } // namespace internalLBFGS

  //
  // initialize preconditioner
  //
  void
  LBFGSNonLinearSolver::initializePreconditioner(
    nonlinearSolverProblem &problem)
  {
    if (d_debugLevel >= 1)
      pcout << "Using preconditioner for LBFGS." << std::endl;
    d_preconditioner.clear();
    problem.precondition(d_preconditioner, d_gradient);
    double mu =
      1 / internalLBFGS::computeDetNormalizationFactor(d_preconditioner);
    if (d_debugLevel >= 1)
      pcout << "Scaling factor for the preconditioner. " << mu << std::endl;
    for (auto i = 0; i < d_preconditioner.size(); ++i)
      {
        d_preconditioner[i] *= mu;
      }
  }

  //
  // Compute Hessian inverse times vector.
  //
  void
  LBFGSNonLinearSolver::computeHx(std::vector<double> &Hx)
  {
    AssertThrow(
      Hx.size() == d_numberUnknowns,
      dealii::ExcMessage(std::string(
        "DFT-FE Error: data size of input to LBFGSNonLinearSolver::computeHx is incorrect.")));
    double d_scalingFactor = 1;
    if (d_numPastSteps > 0)
      {
        if (d_usePreconditioner &&
            internalLBFGS::computePdot(d_deltaXq[d_maxNumPastSteps - 1],
                                       d_deltaGq[d_maxNumPastSteps - 1],
                                       d_preconditioner) > 0)
          {
            d_scalingFactor =
              internalLBFGS::computePdot(d_deltaXq[d_maxNumPastSteps - 1],
                                         d_deltaGq[d_maxNumPastSteps - 1],
                                         d_preconditioner) /
              internalLBFGS::dot(d_deltaGq[d_maxNumPastSteps - 1],
                                 d_deltaGq[d_maxNumPastSteps - 1]);
          }
        else if (!d_usePreconditioner &&
                 internalLBFGS::dot(d_deltaXq[d_maxNumPastSteps - 1],
                                    d_deltaGq[d_maxNumPastSteps - 1]) > 0)
          {
            d_scalingFactor =
              internalLBFGS::dot(d_deltaXq[d_maxNumPastSteps - 1],
                                 d_deltaGq[d_maxNumPastSteps - 1]) /
              internalLBFGS::dot(d_deltaGq[d_maxNumPastSteps - 1],
                                 d_deltaGq[d_maxNumPastSteps - 1]);
          }
        else
          {
            AssertThrow(false,
                        dealii::ExcMessage(std::string(
                          "DFT-FE Error: LBFGS history not properly damped.")));
          }
      }
    std::vector<double> alpha(d_maxNumPastSteps, 0.0);
    for (int j = d_maxNumPastSteps - 1; j >= 0; --j)
      {
        alpha[j] = internalLBFGS::dot(d_deltaXq[j], Hx) * d_rhoq[j];
        internalLBFGS::axpy(-alpha[j], d_deltaGq[j], Hx);
      }
    if (d_usePreconditioner)
      {
        internalLBFGS::linearSolve(d_preconditioner, Hx);
        if (d_numPastSteps > 0)
          {
            for (int i = 0; i < d_numberUnknowns; ++i)
              {
                Hx[i] *= d_scalingFactor;
              }
          }
      }
    else if (d_numPastSteps > 0)
      {
        for (int i = 0; i < d_numberUnknowns; ++i)
          {
            Hx[i] *= d_scalingFactor;
          }
      }
    for (int j = 0; j < d_maxNumPastSteps; ++j)
      {
        double beta = internalLBFGS::dot(d_deltaGq[j], Hx) * d_rhoq[j];
        internalLBFGS::axpy(alpha[j] - beta, d_deltaXq[j], Hx);
      }
  }

  //
  // Compute LBFGS step
  //
  void
  LBFGSNonLinearSolver::computeStep()
  {
    std::vector<double> gradient = d_gradient;
    computeHx(gradient);
    for (int i = 0; i < d_numberUnknowns; ++i)
      {
        d_deltaXNew[i] = -gradient[i];
      }
    d_normDeltaXnew = internalLBFGS::computeLInfNorm(d_deltaXNew);
    if (d_debugLevel >= 1)
      pcout << "Computed LBFGS Step, max norm of step: " << d_normDeltaXnew
            << std::endl;
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
            d_deltaXNew[i] *= d_alpha;
          }
      }
    if (d_debugLevel >= 2)
      pcout << "Step scaled for line search, scaling coefficient: " << d_alpha
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
    std::vector<double> delta_g(d_numberUnknowns, 0.0);

    for (auto i = 0; i < d_numberUnknowns; ++i)
      {
        delta_g[i] = d_gradientNew[i] - d_gradient[i];
      }
    double sBs = -internalLBFGS::dot(d_deltaXNew, d_gradient) * d_alpha;
    double sy  = internalLBFGS::dot(delta_g, d_deltaXNew);
    std::vector<double> Hy = d_gradientNew;
    computeHx(Hy);
    for (int i = 0; i < Hy.size(); ++i)
      {
        Hy[i] += d_deltaXNew[i] / d_alpha;
      }
    double yHy   = internalLBFGS::dot(delta_g, Hy);
    double theta = 1.0;
    if ((sy / sBs < 0) || (sy / sBs < 0.2 && sy / sBs + 1e-8 < yHy / sy))
      {
        theta = 0.8 * sBs / (sBs - sy);
      }
    else if (sy / sBs > 4 && sy / sBs + 1e-8 < yHy / sy)
      {
        theta = 3 * sBs / (sy - sBs);
      }
    if (theta != 1 && d_debugLevel >= 1)
      {
        pcout << "Using damped LBFGS update with theta = " << theta
              << std::endl;
      }
    std::vector<double> r(d_numberUnknowns, 0.0);
    for (auto i = 0; i < d_numberUnknowns; ++i)
      {
        r[i] = theta * delta_g[i] - (1.0 - theta) * d_gradient[i] * d_alpha;
      }
    d_deltaGq.push_back(r);
    d_deltaGq.pop_front();
    d_deltaXq.push_back(d_deltaXNew);
    d_deltaXq.pop_front();
    d_rhoq.push_back(1.0 / internalLBFGS::dot(r, d_deltaXNew));
    d_rhoq.pop_front();
    if (d_numPastSteps < d_maxNumPastSteps)
      {
        ++d_numPastSteps;
        d_noHistory = false;
      }
  }

  //
  // Test if the step satisfies strong Wolfe conditions
  //
  void
  LBFGSNonLinearSolver::checkWolfe()
  {
    double gtdx  = internalLBFGS::dot(d_deltaXNew, d_gradient);
    double gntdx = internalLBFGS::dot(d_deltaXNew, d_gradientNew);
    if (!d_isCurvatureOnlyLineSearchStoppingCondition)
      d_wolfeSufficientDec = (d_valueNew[0] - d_value[0]) < 0.01 * gtdx;
    else
      d_wolfeSufficientDec = false;
    if (!d_isCurvatureOnlyLineSearchStoppingCondition)
      d_wolfeCurvature = std::abs(gntdx) < 0.9 * std::abs(gtdx);
    else
      d_wolfeCurvature = std::abs(gntdx) < std::abs(gtdx);
    d_wolfeSatisfied = d_wolfeSufficientDec && d_wolfeCurvature;
    if (d_debugLevel >= 1)
      {
        if (d_wolfeSatisfied && !d_isCurvatureOnlyLineSearchStoppingCondition)
          pcout << "Wolfe conditions satisfied." << std::endl;
        else if (d_wolfeSufficientDec &&
                 !d_isCurvatureOnlyLineSearchStoppingCondition)
          pcout << "Only Armijo condition satisfied." << std::endl;
        else if (d_isCurvatureOnlyLineSearchStoppingCondition &&
                 d_wolfeCurvature)
          pcout << "Curvature condition satisfied." << std::endl;
      }
  }

  //
  // Compute trust radius for the step
  //
  void
  LBFGSNonLinearSolver::computeStepScale(nonlinearSolverProblem &problem)
  {
    if (d_iter == 0 || d_stepAccepted)
      {
        d_alpha = d_normDeltaXnew > d_maxStepLength ?
                    d_maxStepLength / d_normDeltaXnew :
                    1.0;
        if (d_debugLevel >= 1 && d_normDeltaXnew > d_maxStepLength)
          pcout
            << "Step length exceeded the maximum allowed limit, scaling the step by: "
            << d_alpha << std::endl;
      }
    else
      {
        double gtdx = internalLBFGS::dot(d_deltaX, d_gradient);
        if (!d_isCurvatureOnlyLineSearchStoppingCondition)
          d_alpha =
            -0.5 * gtdx * d_alpha / ((d_valueNew[0] - d_value[0]) - gtdx);
        else
          d_alpha /= 2;
        if (d_alpha < 0.1)
          {
            if (d_debugLevel >= 1)
              pcout
                << "Removing oldest step from LBFGS history as backtracking line search failed."
                << std::endl;
            if (d_usePreconditioner)
              initializePreconditioner(problem);
            if (d_numPastSteps > 0)
              {
                d_alpha = 1.0;
                std::fill(d_deltaGq[d_maxNumPastSteps - d_numPastSteps].begin(),
                          d_deltaGq[d_maxNumPastSteps - d_numPastSteps].end(),
                          0);
                std::fill(d_deltaXq[d_maxNumPastSteps - d_numPastSteps].begin(),
                          d_deltaXq[d_maxNumPastSteps - d_numPastSteps].end(),
                          0);
                d_rhoq[d_maxNumPastSteps - d_numPastSteps] = 0.0;
                --d_numPastSteps;
                if (d_debugLevel >= 2)
                  pcout << "Number of past steps currently stored: "
                        << d_numPastSteps << std::endl;
                d_noHistory = d_numPastSteps == 0;
                computeStep();
              }
            else
              {
                d_noHistory = true;
              }
          }
      }
    if (d_debugLevel >= 1)
      pcout << "Trying step size (scaling factor): " << d_alpha << std::endl;
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
  // save checkpoint files.
  //
  void
  LBFGSNonLinearSolver::save(const std::string &checkpointFileName)
  {
    if (d_debugLevel >= 2)
      {
        pcout << "Saving LBFGS data to " << checkpointFileName << std::endl;
      }
    std::vector<std::vector<double>> data;
    for (unsigned int i = 0; i < d_deltaX.size(); ++i)
      data.push_back(std::vector<double>(1, d_deltaX[i]));
    for (unsigned int i = 0; i < d_gradient.size(); ++i)
      data.push_back(std::vector<double>(1, d_gradient[i]));
    for (unsigned int i = 0; i < d_deltaXq.size(); ++i)
      {
        for (unsigned int j = 0; j < d_deltaXq[i].size(); ++j)
          {
            data.push_back(std::vector<double>(1, d_deltaXq[i][j]));
          }
      }
    for (unsigned int i = 0; i < d_deltaGq.size(); ++i)
      {
        for (unsigned int j = 0; j < d_deltaGq[i].size(); ++j)
          {
            data.push_back(std::vector<double>(1, d_deltaGq[i][j]));
          }
      }
    for (unsigned int i = 0; i < d_deltaGq.size(); ++i)
      {
        data.push_back(std::vector<double>(1, d_rhoq[i]));
      }

    data.push_back(d_value);
    data.push_back(d_valueNew);
    data.push_back(std::vector<double>(1, d_alpha));
    data.push_back(std::vector<double>(1, d_iter));
    data.push_back(std::vector<double>(1, (double)d_stepAccepted));
    data.push_back(std::vector<double>(1, d_numPastSteps));


    dftUtils::writeDataIntoFile(data, checkpointFileName, mpi_communicator);
  }


  //
  // load from checkpoint files.
  //
  void
  LBFGSNonLinearSolver::load(const std::string &checkpointFileName)
  {
    if (d_debugLevel >= 1)
      {
        pcout << "Loading LBFGS data from " << checkpointFileName << std::endl;
      }
    std::vector<std::vector<double>> data;
    dftUtils::readFile(1, data, checkpointFileName);
    AssertThrow(
      data.size() ==
        (2 * d_numberUnknowns + 2 * d_numberUnknowns * d_maxNumPastSteps +
         d_maxNumPastSteps + 6),
      dealii::ExcMessage(std::string(
        "DFT-FE Error: data size of lbfgs solver checkpoint file is incorrect.")));

    d_deltaX.resize(d_numberUnknowns, 0.0);
    d_gradient.resize(d_numberUnknowns, 0.0);
    d_deltaGq.resize(d_maxNumPastSteps);
    d_deltaXq.resize(d_maxNumPastSteps);
    d_rhoq.resize(d_maxNumPastSteps, 0.0);
    d_value.resize(1);
    d_valueNew.resize(1);
    for (int i = 0; i < d_maxNumPastSteps; ++i)
      {
        d_deltaGq[i].resize(d_numberUnknowns);
        d_deltaXq[i].resize(d_numberUnknowns);
      }
    for (unsigned int i = 0; i < d_numberUnknowns; ++i)
      d_deltaX[i] = data[i][0];

    for (unsigned int i = 0; i < d_numberUnknowns; ++i)
      d_gradient[i] = data[i + d_numberUnknowns][0];

    for (unsigned int i = 0; i < d_deltaXq.size(); ++i)
      {
        for (unsigned int j = 0; j < d_deltaXq[i].size(); ++j)
          {
            d_deltaXq[i][j] =
              data[j + i * d_deltaXq[i].size() + 2 * d_numberUnknowns][0];
          }
      }

    for (unsigned int i = 0; i < d_deltaGq.size(); ++i)
      {
        for (unsigned int j = 0; j < d_deltaGq[i].size(); ++j)
          {
            d_deltaGq[i][j] = data[j + i * d_deltaGq[i].size() +
                                   d_numberUnknowns * d_maxNumPastSteps +
                                   2 * d_numberUnknowns][0];
          }
      }
    for (unsigned int i = 0; i < d_rhoq.size(); ++i)
      {
        d_rhoq[i] = data[i + 2 * d_numberUnknowns * d_maxNumPastSteps +
                         2 * d_numberUnknowns][0];
      }

    d_value[0] =
      data[2 * d_numberUnknowns + 2 * d_numberUnknowns * d_maxNumPastSteps +
           d_maxNumPastSteps][0];

    d_valueNew[0] =
      data[1 + 2 * d_numberUnknowns + 2 * d_numberUnknowns * d_maxNumPastSteps +
           d_maxNumPastSteps][0];

    d_alpha =
      data[2 + 2 * d_numberUnknowns + 2 * d_numberUnknowns * d_maxNumPastSteps +
           d_maxNumPastSteps][0];

    d_iter =
      (int)
        data[3 + 2 * d_numberUnknowns +
             2 * d_numberUnknowns * d_maxNumPastSteps + d_maxNumPastSteps][0] +
      1;

    d_stepAccepted =
      data[4 + 2 * d_numberUnknowns + 2 * d_numberUnknowns * d_maxNumPastSteps +
           d_maxNumPastSteps][0] == 1.0;

    d_numPastSteps =
      data[5 + 2 * d_numberUnknowns + 2 * d_numberUnknowns * d_maxNumPastSteps +
           d_maxNumPastSteps][0];

    d_noHistory = d_alpha < 0.1 && d_numPastSteps == 0;
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
    d_updateVector.resize(d_numberUnknowns, 0.0);
    d_deltaX.resize(d_numberUnknowns, 0.0);
    d_deltaXNew.resize(d_numberUnknowns, 0.0);
    d_gradient.resize(d_numberUnknowns, 0.0);
    d_gradientNew.resize(d_numberUnknowns, 0.0);
    d_deltaGq.resize(d_maxNumPastSteps);
    d_deltaXq.resize(d_maxNumPastSteps);
    d_rhoq.resize(d_maxNumPastSteps, 0.0);
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
        d_stepAccepted = true;
        d_numPastSteps = 0;
        d_iter         = 0;
        d_noHistory    = false;
        //
        // compute initial values of problem and problem gradient
        //
        problem.gradient(d_gradient);
        problem.value(d_value);

        MPI_Barrier(mpi_communicator);
        problem.save();
      }
    else
      {
        load(checkpointFileName);
        MPI_Barrier(mpi_communicator);
        d_useSingleAtomSolutionsInitialGuess = true;
      }

    if (d_usePreconditioner)
      {
        initializePreconditioner(problem);
      }

    //
    // check for convergence
    //
    unsigned int isSuccess = 0;

    if (problem.isConverged())
      isSuccess = 1;

    MPI_Bcast(&(isSuccess), 1, MPI_INT, 0, mpi_communicator);
    if (isSuccess == 1)
      return SUCCESS;



    for (d_iter = d_iter > 0 ? d_iter : 0; d_iter < d_maxNumberIterations;
         ++d_iter)
      {
        if (d_debugLevel >= 1)
          pcout << "LBFGS Step no. " << d_iter + 1 << std::endl;

        //
        // If history is all deleted then fail
        //
        if (d_noHistory)
          {
            if (d_debugLevel >= 1)
              pcout
                << "Backtracking line search failed for steepest descent step, exiting with failure."
                << std::endl;
            break;
          }

        //
        // Compute LBFGS step
        //
        computeStep();
        //
        // Backtracking line search if needed
        //
        computeStepScale(problem);
        //
        // Compute the update vector
        //
        computeUpdateStep();

        updateSolution(d_updateVector, problem);

        //
        // get gradient and value
        //
        problem.gradient(d_gradientNew);
        problem.value(d_valueNew);

        //
        // update trust radius and hessian
        //
        checkWolfe();
        d_stepAccepted = d_isCurvatureOnlyLineSearchStoppingCondition ?
                           d_wolfeCurvature :
                           d_wolfeSufficientDec;
        if (d_stepAccepted)
          {
            updateHistory();

            d_deltaX   = d_deltaXNew;
            d_value[0] = d_valueNew[0];
            d_gradient = d_gradientNew;
          }
        else
          {
            if (d_debugLevel >= 1)
              {
                if (!d_isCurvatureOnlyLineSearchStoppingCondition)
                  pcout
                    << "Step rejected as Armijo condition was not satisfied."
                    << std::endl;
                else
                  pcout
                    << "Step rejected as Curvature condition was not satisfied."
                    << std::endl;
              }

            d_deltaX = d_deltaXNew;
          }

        //
        // Save the last step
        //
        MPI_Barrier(mpi_communicator);
        problem.save();

        //
        // check for convergence
        //
        unsigned int isBreak = 0;

        if (problem.isConverged())
          isBreak = 1;
        MPI_Bcast(&(isBreak), 1, MPI_INT, 0, mpi_communicator);
        if (isBreak == 1)
          break;
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
            pcout << "LBFGS solver converged after " << d_iter + 1
                  << " iterations." << std::endl;
          }
        else if (MAX_ITER_REACHED)
          {
            pcout << "LBFGS solver failed to converge after " << d_iter
                  << " iterations." << std::endl;
          }
        else
          {
            pcout << "LBFGS solver failed to converge after " << d_iter
                  << " iterations." << std::endl;
            pcout
              << "The accuracy of computed forces seems to be inadequate, try using finer mesh and/or turning on Higher-Quad-PSP."
              << std::endl;
          }
      }

    //
    //
    //
    return returnValue;
  }
} // namespace dftfe
