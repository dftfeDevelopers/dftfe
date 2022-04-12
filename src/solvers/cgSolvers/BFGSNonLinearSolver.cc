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

#include <BFGSNonLinearSolver.h>
#include <fileReaders.h>
#include <nonlinearSolverProblem.h>

namespace dftfe
{
  //
  // Constructor.
  //
  BFGSNonLinearSolver::BFGSNonLinearSolver(
    const double       tolerance,
    const unsigned int maxNumberIterations,
    const unsigned int debugLevel,
    const MPI_Comm &   mpi_comm_parent,
    const double       trustRadius_maximum,
    const unsigned int trustRadius_initial,
    const double       trustRadius_minimum)
    : nonLinearSolver(debugLevel, maxNumberIterations, tolerance)
    , mpi_communicator(mpi_comm_parent)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_parent))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {
    d_isBFGSRestartDueToSmallRadius      = false;
    d_useSingleAtomSolutionsInitialGuess = false;
    d_trustRadiusInitial                 = trustRadius_initial;
    d_trustRadiusMax                     = trustRadius_maximum;
    d_trustRadiusMin                     = trustRadius_minimum;
  }

  //
  // Destructor.
  //
  BFGSNonLinearSolver::~BFGSNonLinearSolver()
  {
    //
    //
    //
    return;
  }

  // Computes the lowest eigenvalue of the augmented hessian.
  void
  BFGSNonLinearSolver::computeLambda()
  {
    std::vector<double> augmentedHessian((d_numberUnknowns + 1) *
                                           (d_numberUnknowns + 1),
                                         0.0);
    std::vector<double> augmentedSrfo((d_numberUnknowns + 1) *
                                        (d_numberUnknowns + 1),
                                      0.0);
    for (auto col = 0; col < d_numberUnknowns; ++col)
      {
        for (auto row = 0; row < d_numberUnknowns; ++row)
          {
            augmentedHessian[row + (d_numberUnknowns + 1) * col] =
              d_hessian[row + (d_numberUnknowns)*col];
            augmentedSrfo[row + (d_numberUnknowns + 1) * col] =
              d_Srfo[row + (d_numberUnknowns)*col];
          }
      }
    for (auto i = 0; i < d_numberUnknowns; ++i)
      {
        augmentedHessian[i + (d_numberUnknowns + 1) * d_numberUnknowns] =
          d_gradient[i];
        augmentedHessian[d_numberUnknowns + (d_numberUnknowns + 1) * i] =
          d_gradient[i];
      }
    augmentedSrfo[(d_numberUnknowns + 2) * d_numberUnknowns] = 1.0;
    /*    int                 info;
        std::vector<double> eigenValues(d_numberUnknowns + 1, 0.0);
        const unsigned int  dimensionMatrix = d_numberUnknowns + 1;
        const unsigned int  lwork           = 1 + 6 * dimensionMatrix +
                                   2 * dimensionMatrix * dimensionMatrix,
                           liwork = 3 + 5 * dimensionMatrix;
        std::vector<int>    iwork(liwork, 0);
        const char          jobz = 'N', uplo = 'U';
        std::vector<double> work(lwork);

        dsyevd_(&jobz,
                &uplo,
                &dimensionMatrix,
                augmentedHessian.data(),
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
    */
    int                 info;
    const int           one = 1;
    std::vector<double> eigenValues(1, 0.0);
    std::vector<double> eigenVector(d_numberUnknowns + 1, 0.0);
    const int           dimensionMatrix = d_numberUnknowns + 1;
    const int        lwork = 8 * dimensionMatrix, liwork = 5 * dimensionMatrix;
    std::vector<int> iwork(liwork, 0);
    std::vector<int> ifail(dimensionMatrix, 0);
    const char       jobz = 'V', uplo = 'U', range = 'I', cmach = 'S';
    const double     abstol = 2 * dlamch_(&cmach);
    std::vector<double> work(lwork);
    int                 nEigVals;

    dsygvx_(&one,
            &jobz,
            &range,
            &uplo,
            &dimensionMatrix,
            augmentedHessian.data(),
            &dimensionMatrix,
            augmentedSrfo.data(),
            &dimensionMatrix,
            NULL,
            NULL,
            &one,
            &one,
            &abstol,
            &nEigVals,
            eigenValues.data(),
            eigenVector.data(),
            &dimensionMatrix,
            work.data(),
            &lwork,
            iwork.data(),
            ifail.data(),
            &info);

    //
    // free up memory associated with work
    //
    pcout << "DEBUG lambda info " << info << " " << work[0] << std::endl;
    work.clear();
    iwork.clear();
    std::vector<double>().swap(work);
    std::vector<int>().swap(iwork);
    d_lambda = eigenValues[0];
    for (auto i = 0; i < d_numberUnknowns; ++i)
      {
        d_deltaX[i] = eigenVector[i] / eigenVector[d_numberUnknowns];
        d_deltaX[i] = d_deltaX[i] >= d_trustRadius ? d_trustRadius:d_deltaX[i];
      }
  }

  //
  // initialize hessian
  //
  void
  BFGSNonLinearSolver::initializeHessian(nonlinearSolverProblem &problem)
  {
    std::vector<double> dummyVec;
    problem.precondition(d_hessian, dummyVec);
    d_Srfo = d_hessian;
    int                 info;
    std::vector<double> eigenValues(d_numberUnknowns, 0.0);
    const unsigned int  dimensionMatrix = d_numberUnknowns;
    const unsigned int  lwork = 1 + 2 * dimensionMatrix, liwork = 1;
    std::vector<int>    iwork(liwork, 0);
    const char          jobz = 'N', uplo = 'U';
    std::vector<double> work(lwork);

    dsyevd_(&jobz,
            &uplo,
            &dimensionMatrix,
            d_Srfo.data(),
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

    double detS = 1.0;
    for (auto i = 0; i < d_numberUnknowns; ++i)
      {
        detS *= eigenValues[i];
      }
    d_Srfo = d_hessian;
    for (auto i = 0; i < d_Srfo.size(); ++i)
      {
        d_Srfo[i] /= detS;
      }
    for (unsigned int i = 0; i < d_numberUnknowns; ++i)
      {
        if (std::abs(d_gradient[i]) > d_gradMax)
          d_gradMax = std::abs(d_gradient[i]);
      }
  }

  //
  // Update Hessian.
  //
  void
  BFGSNonLinearSolver::updateHessian()
  {
    std::vector<double> delta_g(d_numberUnknowns, 0.0);

    for (auto i = 0; i < d_numberUnknowns; ++i)
      {
        delta_g[i] = d_gradientNew[i] - d_gradient[i];
      }

    const char         uplo  = 'U';
    const double       one_d = 1.0;
    const unsigned int one   = 1;
    // z=dg-Hdx, y=dg, s=dx

    std::vector<double> Hdx(d_numberUnknowns, 0.0);
    dsymv_(&uplo,
           &d_numberUnknowns,
           &one_d,
           d_hessian.data(),
           &d_numberUnknowns,
           d_deltaX.data(),
           &one,
           &one_d,
           Hdx.data(),
           &one);

    std::vector<double> dgmHdx(d_numberUnknowns, 0.0);
    for (auto i = 0; i < d_numberUnknowns; ++i)
      {
        dgmHdx[i] = delta_g[i] - Hdx[i];
      }
    double dxtHdx =
      ddot_(&d_numberUnknowns, d_deltaX.data(), &one, Hdx.data(), &one);
    double dgtdx =
      ddot_(&d_numberUnknowns, d_deltaX.data(), &one, delta_g.data(), &one);
    double ztdx =
      ddot_(&d_numberUnknowns, d_deltaX.data(), &one, dgmHdx.data(), &one);
    double dxnorm = dnrm2_(&d_numberUnknowns, d_deltaX.data(), &one);
    double dgnorm = dnrm2_(&d_numberUnknowns, delta_g.data(), &one);
    double znorm  = dnrm2_(&d_numberUnknowns, dgmHdx.data(), &one);
    if (ztdx / (znorm * dxnorm) < -0.1)
      {
        double factor = 1.0 / ztdx;
        dsyr_(&uplo,
              &d_numberUnknowns,
              &factor,
              dgmHdx.data(),
              &one,
              d_hessian.data(),
              &d_numberUnknowns);
      }
    else if (dgtdx / (dgnorm * dxnorm) > 0.1)
      {
        double factor = 1.0 / dgtdx;
        dsyr_(&uplo,
              &d_numberUnknowns,
              &factor,
              delta_g.data(),
              &one,
              d_hessian.data(),
              &d_numberUnknowns);
        factor = -1.0 / dxtHdx;
        dsyr_(&uplo,
              &d_numberUnknowns,
              &factor,
              Hdx.data(),
              &one,
              d_hessian.data(),
              &d_numberUnknowns);
      }
    else
      {
        double factor = 1.0 / dgtdx;
        dsyr2_(&uplo,
               &d_numberUnknowns,
               &factor,
               d_deltaX.data(),
               &one,
               dgmHdx.data(),
               &one,
               d_hessian.data(),
               &d_numberUnknowns);
        factor = -ztdx / (dxnorm * dxnorm);
        dsyr_(&uplo,
              &d_numberUnknowns,
              &factor,
              d_deltaX.data(),
              &one,
              d_hessian.data(),
              &d_numberUnknowns);
      }
  }

  //
  // Compute step
  //
  void
  BFGSNonLinearSolver::computeStep()
  {
    computeLambda();
    pcout << "DEBUG Lambda " << d_lambda << std::endl;
    /*std::vector<double> HpLambdaS(d_numberUnknowns * d_numberUnknowns, 0.0);

    for (auto col = 0; col < d_numberUnknowns; ++col)
      {
        for (auto row = 0; row < d_numberUnknowns; ++row)
          {
            HpLambdaS[col + (d_numberUnknowns)*row] =
              -d_hessian[col + (d_numberUnknowns)*row] +
              d_lambda * d_Srfo[col + (d_numberUnknowns)*row];
          }
      }


    int                 info;
    const int           lwork = d_numberUnknowns;
    const char          uplo  = 'U';
    const int           one   = 1;
    std::vector<double> work(d_numberUnknowns, 0.0);
    std::vector<int>    ipiv(d_numberUnknowns, 0);
    d_deltaX = d_gradient;
    pcout << "DEBUG Start Linear Solve step " << this_mpi_process << " "
          << lwork << " " << d_deltaX.size() << " " << HpLambdaS.size()
          << std::endl;
    dsysv_(&uplo,
           &lwork,
           &one,
           HpLambdaS.data(),
           &lwork,
           ipiv.data(),
           d_deltaX.data(),
           &lwork,
           work.data(),
           &lwork,
           &info);
    pcout << "DEBUG end Linear Solve step " << this_mpi_process << " " << info
          << " " << d_deltaX.size() << std::endl;*/
    // AssertThrow(false, dealii::ExcMessage(std::string(
    //                    "terminate")));
    /*work.clear();
    ipiv.clear();
    std::vector<double>().swap(work);
    std::vector<int>().swap(ipiv);*/
  }



  //
  // Compute residual L2-norm.
  //
  double
  BFGSNonLinearSolver::computeResidualL2Norm() const
  {
    // initialize norm
    //
    double norm = 0.0;

    //
    // iterate over unknowns
    //
    for (unsigned int i = 0; i < d_numberUnknowns; ++i)
      {
        const double factor   = d_unknownCountFlag[i];
        const double gradient = d_gradient[i];
        norm += factor * gradient * gradient;
      }


    //
    // take square root
    //
    norm = std::sqrt(norm);

    //
    //
    //
    return norm;
  }

  //
  // Update solution x -> x + step.
  //
  bool
  BFGSNonLinearSolver::updateSolution(const std::vector<double> &step,
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
  BFGSNonLinearSolver::solve(nonlinearSolverProblem &problem,
                             const std::string       checkpointFileName,
                             const bool              restart)
  {
    //
    // method const data
    //
    const double toleranceSqr = d_tolerance * d_tolerance;

    //
    // get total number of unknowns in the problem.
    //
    d_numberUnknowns = problem.getNumberUnknowns();

    //
    // resize d_unknownCountFlag with numberUnknown and initialize to 1
    //
    d_unknownCountFlag.resize(d_numberUnknowns, 1);

    //
    // allocate space for step, gradient and new gradient.
    //
    d_deltaX.resize(d_numberUnknowns);
    d_gradient.resize(d_numberUnknowns);
    d_gradientNew.resize(d_numberUnknowns);

    //
    // initialize delta new and direction
    //
    if (!restart)
      {
        d_trustRadius=d_trustRadiusInitial;
        //
        // compute initial values of problem and problem gradient
        //
        pcout << "DEBUG START BFGS " << std::endl;
        problem.gradient(d_gradient);
        pcout << "DEBUG Compute g0 " << std::endl;

        initializeHessian(problem);
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
    if (d_gradMax < d_tolerance)
      isSuccess = 1;

    MPI_Bcast(&(isSuccess), 1, MPI_INT, 0, mpi_communicator);
    if (isSuccess == 1)
      return SUCCESS;



    for (d_iter = 0; d_iter < d_maxNumberIterations; ++d_iter)
      {
        if (d_debugLevel >= 2)
          for (unsigned int i = 0; i < d_gradient.size(); ++i)
            pcout << "d_gradient: " << d_gradient[i] << std::endl;


        //
        // compute L2-norm of the residual (gradient)
        //
        const double residualNorm = computeResidualL2Norm();


        if (d_debugLevel >= 2)
          pcout << "BFGS Step no. | residual norm "
                   "| residual norm avg"
                << std::endl;
        else if (d_debugLevel >= 1)
          pcout << "BFGS Step no. " << d_iter + 1 << std::endl;
        //
        // output at the begining of the iteration
        //
        if (d_debugLevel >= 2)
          pcout << d_iter + 1 << " " << residualNorm << " "
                << residualNorm / d_numberUnknowns << " " << std::endl;

        //
        // Compute the update step
        //
        pcout << "DEBUG Start Compute step " << std::endl;
        computeStep();
        for (unsigned int i = 0; i < d_deltaX.size(); ++i)
          pcout << "step: " << d_deltaX[i] << std::endl;
        pcout << "DEBUG End Compute step " << std::endl;
        updateSolution(d_deltaX, problem);
        pcout << "DEBUG End update step " << std::endl;
        //
        // evaluate gradient
        //
        problem.gradient(d_gradientNew);


        //
        // check for convergence
        //
        unsigned int isBreak = 0;

        d_gradMax = 0.0;
        for (unsigned int i = 0; i < d_numberUnknowns; ++i)
          {
            if (std::abs(d_gradientNew[i]) > d_gradMax)
              d_gradMax = std::abs(d_gradientNew[i]);
          }

        if (d_gradMax < d_tolerance)
          isBreak = 1;
        MPI_Bcast(&(isSuccess), 1, MPI_INT, 0, mpi_communicator);
        if (isBreak == 1)
          break;


        //
        // update hessian
        //
        updateHessian();

        d_gradient = d_gradientNew;
      }

    //
    // set error condition
    //
    ReturnValueType returnValue = SUCCESS;

    if (d_iter == d_maxNumberIterations)
      returnValue = MAX_ITER_REACHED;

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
