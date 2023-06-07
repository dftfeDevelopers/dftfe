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
// @author Sambit Das and Phani Motamarri

#include <cgPRPNonLinearSolver.h>
#include <fileReaders.h>
#include <nonlinearSolverProblem.h>

namespace dftfe
{
  //
  // Constructor.
  //
  cgPRPNonLinearSolver::cgPRPNonLinearSolver(
    const unsigned int maxNumberIterations,
    const unsigned int debugLevel,
    const MPI_Comm &   mpi_comm_parent,
    const double       lineSearchTolerance,
    const unsigned int lineSearchMaxIterations,
    const double       lineSearchDampingParameter,
    const double       maxIncrementSolLinf,
    const bool         isCurvatureOnlyLineSearchStoppingCondition)
    : d_lineSearchTolerance(lineSearchTolerance)
    , d_lineSearchMaxIterations(lineSearchMaxIterations)
    , d_lineSearchDampingParameter(lineSearchDampingParameter)
    , d_maxSolutionIncrementLinf(maxIncrementSolLinf)
    , nonLinearSolver(debugLevel, maxNumberIterations)
    , mpi_communicator(mpi_comm_parent)
    , n_mpi_processes(dealii::Utilities::MPI::n_mpi_processes(mpi_comm_parent))
    , this_mpi_process(
        dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , d_isCurvatureOnlyLineSearchStoppingCondition(
        isCurvatureOnlyLineSearchStoppingCondition)
  {
    d_isCGRestartDueToLargeIncrement     = false;
    d_useSingleAtomSolutionsInitialGuess = false;
  }

  //
  // Destructor.
  //
  cgPRPNonLinearSolver::~cgPRPNonLinearSolver()
  {
    //
    //
    //
    return;
  }

  //
  // initialize direction
  //
  void
  cgPRPNonLinearSolver::initializeDirection()
  {
    //
    // initialize delta new
    //
    d_deltaNew = 0.0;
    d_gradMax  = 0.0;

    //
    // iterate over unknowns
    //
    for (unsigned int i = 0; i < d_numberUnknowns; ++i)
      {
        const double factor = d_unknownCountFlag[i];
        const double r      = -d_gradient[i];

        d_steepestDirectionOld[i] = r;
        d_conjugateDirection[i]   = r;
        d_deltaNew += factor * r * d_conjugateDirection[i];

        if (std::abs(d_gradient[i]) > d_gradMax)
          d_gradMax = std::abs(d_gradient[i]);
      }
    //
    //
    return;
  }

  //
  // Compute delta_d and eta_p.
  //
  std::pair<double, double>
  cgPRPNonLinearSolver::computeDeltaD()
  {
    //
    // initialize delta_d and eta_p
    //
    double deltaD = 0.0;
    double etaP   = 0.0;

    //
    // iterate over unknowns
    //
    for (unsigned int i = 0; i < d_numberUnknowns; ++i)
      {
        const double factor    = d_unknownCountFlag[i];
        const double direction = d_conjugateDirection[i];

        deltaD += factor * direction * direction;
        etaP += factor * d_gradient[i] * direction;
      }

    //
    //
    //
    return std::make_pair(deltaD, etaP);
  }

  //
  // Compute eta.
  //
  double
  cgPRPNonLinearSolver::computeEta()
  {
    //
    // initialize eta
    //
    double eta = 0.0;

    //
    // iterate over unknowns
    //
    for (unsigned int i = 0; i < d_numberUnknowns; ++i)
      {
        const double factor    = d_unknownCountFlag[i];
        const double direction = d_conjugateDirection[i];
        eta += factor * d_gradient[i] * direction;
      }

    //
    //
    return eta;
  }

  //
  // Compute delta new and delta mid.
  //
  void
  cgPRPNonLinearSolver::computeDeltas()
  {
    //
    // initialize delta new and delta mid.
    //
    d_deltaMid = 0.0;
    d_deltaNew = 0.0;

    //
    // iterate over unknowns
    //
    for (unsigned int i = 0; i < d_numberUnknowns; ++i)
      {
        const double factor = d_unknownCountFlag[i];
        const double r      = -d_gradient[i];
        const double sOld   = d_steepestDirectionOld[i];

        //
        // compute delta mid
        //
        d_deltaMid += factor * r * sOld;

        //
        // save gradient old
        //
        d_steepestDirectionOld[i] = r;

        //
        // compute delta new
        //
        d_deltaNew += factor * r * r;
      }
    //
    //
    //
    return;
  }

  //
  // Update direction.
  //
  void
  cgPRPNonLinearSolver::updateDirection()
  {
    //
    // iterate over unknowns
    //
    for (unsigned int i = 0; i < d_numberUnknowns; ++i)
      {
        d_conjugateDirection[i] *= d_beta;
        d_conjugateDirection[i] += -d_gradient[i];
      }
  }

  //
  // save checkpoint files.
  //
  void
  cgPRPNonLinearSolver::save(const std::string &checkpointFileName)
  {
    std::vector<std::vector<double>> data;
    for (unsigned int i = 0; i < d_conjugateDirection.size(); ++i)
      data.push_back(std::vector<double>(1, d_conjugateDirection[i]));

    for (unsigned int i = 0; i < d_steepestDirectionOld.size(); ++i)
      data.push_back(std::vector<double>(1, d_steepestDirectionOld[i]));

    data.push_back(std::vector<double>(1, d_alphaChk));
    data.push_back(std::vector<double>(1, d_etaPChk));
    data.push_back(std::vector<double>(1, d_etaChk));
    data.push_back(std::vector<double>(1, d_lineSearchRestartIterChk));
    data.push_back(std::vector<double>(1, d_functionValueChk));
    data.push_back(std::vector<double>(1, d_etaAlphaZeroChk));

    if (d_lineSearchRestartIterChk >= 1)
      data.push_back(
        std::vector<double>(1, d_functionalValueAfterAlphUpdateChk));

    dftUtils::writeDataIntoFile(data, checkpointFileName, mpi_communicator);
  }

  //
  // load from checkpoint files.
  //
  void
  cgPRPNonLinearSolver::load(const std::string &checkpointFileName)
  {
    std::vector<std::vector<double>> data;
    dftUtils::readFile(1, data, checkpointFileName);

    d_conjugateDirection.resize(d_numberUnknowns);
    for (unsigned int i = 0; i < d_numberUnknowns; ++i)
      d_conjugateDirection[i] = data[i][0];

    d_steepestDirectionOld.resize(d_numberUnknowns);
    d_gradient.resize(d_numberUnknowns);
    for (unsigned int i = 0; i < d_numberUnknowns; ++i)
      {
        d_steepestDirectionOld[i] = data[d_numberUnknowns + i][0];
        d_gradient[i]             = -data[d_numberUnknowns + i][0];
      }

    d_alphaChk                 = data[2 * d_numberUnknowns][0];
    d_etaPChk                  = data[2 * d_numberUnknowns + 1][0];
    d_etaChk                   = data[2 * d_numberUnknowns + 2][0];
    d_lineSearchRestartIterChk = data[2 * d_numberUnknowns + 3][0];
    d_functionValueChk         = data[2 * d_numberUnknowns + 4][0];
    d_etaAlphaZeroChk          = data[2 * d_numberUnknowns + 5][0];

    if (d_lineSearchRestartIterChk >= 1)
      d_functionalValueAfterAlphUpdateChk = data[2 * d_numberUnknowns + 6][0];

    if (d_lineSearchRestartIterChk >= 1)
      {
        AssertThrow(
          data.size() == (2 * d_numberUnknowns + 7),
          dealii::ExcMessage(std::string(
            "DFT-FE Error: data size of cg solver checkpoint file is incorrect.")));
      }
    else
      {
        AssertThrow(
          data.size() == (2 * d_numberUnknowns + 6),
          dealii::ExcMessage(std::string(
            "DFT-FE Error: data size of cg solver checkpoint file is incorrect.")));
      }
  }

  //
  // Compute residual L2-norm.
  //
  double
  cgPRPNonLinearSolver::computeResidualL2Norm() const
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
  // Compute the total number of unknowns in all processors.
  //
  unsigned int
  cgPRPNonLinearSolver::computeTotalNumberUnknowns() const
  {
    //
    // initialize total number of unknowns
    //
    unsigned int totalNumberUnknowns = 0;

    //
    // iterate over unknowns
    //
    for (unsigned int i = 0; i < d_numberUnknowns; ++i)
      {
        const unsigned int factor = d_unknownCountFlag[i];

        totalNumberUnknowns += factor;
      }

    //
    // accumulate totalnumberUnknowns
    //
    totalNumberUnknowns =
      dealii::Utilities::MPI::sum(totalNumberUnknowns, mpi_communicator);
    ;

    //
    //
    //
    return totalNumberUnknowns;
  }

  //
  // Update solution x -> x + \alpha direction.
  //
  bool
  cgPRPNonLinearSolver::updateSolution(const double               alpha,
                                       const std::vector<double> &direction,
                                       nonlinearSolverProblem &   problem)
  {
    std::vector<double> incrementVector;

    //
    // get the size of solution
    //
    const std::vector<double>::size_type solutionSize = d_numberUnknowns;
    incrementVector.resize(d_numberUnknowns);


    for (std::vector<double>::size_type i = 0; i < solutionSize; ++i)
      incrementVector[i] = alpha * direction[i];

    int isIncrementBoundExceeded = 0;
    for (std::vector<double>::size_type i = 0; i < solutionSize; ++i)
      {
        if (std::abs(incrementVector[i]) > d_maxSolutionIncrementLinf)
          isIncrementBoundExceeded = 1;
      }

    MPI_Bcast(&(isIncrementBoundExceeded), 1, MPI_INT, 0, mpi_communicator);

    if (isIncrementBoundExceeded == 1)
      {
        pcout
          << "Warning: maximum increment bound exceeded in line search update. Such situtations can also happen if the SCF iterations for the current ground-state did not converge."
          << std::endl;
        return false;
      }

    //
    // call solver problem update
    //
    problem.update(incrementVector, true, d_useSingleAtomSolutionsInitialGuess);

    d_useSingleAtomSolutionsInitialGuess = false;
    return true;
  }

  //
  // Perform line search.
  //
  nonLinearSolver::ReturnValueType
  cgPRPNonLinearSolver::lineSearch(nonlinearSolverProblem &problem,
                                   const double            tolerance,
                                   const unsigned int      maxNumberIterations,
                                   const unsigned int      debugLevel,
                                   const std::string       checkpointFileName,
                                   const int               startingIter,
                                   const bool              isCheckpointRestart)
  {
    //
    // local data
    //
    const double        toleranceSqr = tolerance * tolerance;
    std::vector<double> tempFuncValueVector;
    double              eta, etaP, etaAlphaZero, functionValue, alpha, alphaNew,
      functionalValueAfterAlphUpdate;

    //
    // constants used in Wolfe conditions
    // c1, c2 are chosen based on Page122 of the book "Numerical Optimization"
    // by Jorge Nocedal and Stephen J. Wright Also look at
    // https://en.wikipedia.org/wiki/Wolfe_conditions
    //
    const double c1 = 1e-04;
    const double c2 = 0.1;

    if (isCheckpointRestart)
      {
        // fill checkpoint data
        functionValue = d_functionValueChk;
        if (startingIter >= 1)
          functionalValueAfterAlphUpdate = d_functionalValueAfterAlphUpdateChk;
        eta          = d_etaChk;
        etaAlphaZero = d_etaAlphaZeroChk;
        etaP         = d_etaPChk;
        alpha        = d_alphaChk;
      }
    else
      {
        //
        // set the initial value of alpha
        //
        alpha = -d_lineSearchDampingParameter;

        //
        // evaluate problem gradient
        //
        problem.gradient(d_gradient);

        //
        // evaluate function value
        //
        problem.value(tempFuncValueVector);
        functionValue = tempFuncValueVector[0];

        //
        // compute delta_d and eta_p
        //
        etaP = computeEta();

        etaAlphaZero = etaP;
      }

    if (startingIter == -1)
      {
        if (debugLevel >= 2)
          pcout << "Initial guess for secant line search iteration, alpha: "
                << alpha << std::endl;

        d_functionValueChk         = functionValue;
        d_etaChk                   = etaP;
        d_etaPChk                  = etaP;
        d_alphaChk                 = alpha;
        d_etaAlphaZeroChk          = etaAlphaZero;
        d_lineSearchRestartIterChk = -1;

        MPI_Barrier(mpi_communicator);
        problem.save();

        //
        // update unknowns removing earlier update
        //
        d_isCGRestartDueToLargeIncrement = !(updateSolution(
          d_lineSearchDampingParameter, d_conjugateDirection, problem));
        if (d_isCGRestartDueToLargeIncrement)
          {
            if (debugLevel >= 1)
              pcout
                << " Secant line-search failed as maximum increment bound is exceeded. CG PRP will be restarted "
                << std::endl;

            return LINESEARCH_FAILED;
          }
      }

    //
    // begin iteration (using secant method)
    //
    for (int iter = ((startingIter >= 0) ? startingIter : 0);
         iter < maxNumberIterations;
         ++iter)
      {
        if (iter > startingIter)
          {
            //
            // evaluate problem gradient
            //
            problem.gradient(d_gradient);


            //
            // compute eta
            //
            eta = computeEta();

            //
            // swap eta and etaP to make the notation consistent to PRP
            // algorithm in "Painless Conjugate Algorithm"
            // https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf
            //
            if (iter == 0)
              {
                double temp = eta;
                eta         = etaP;
                etaP        = temp;
              }

            if (iter >= 1)
              {
                problem.value(tempFuncValueVector);
                functionalValueAfterAlphUpdate = tempFuncValueVector[0];
              }
          }

        d_functionValueChk = functionValue;
        if (iter >= 1)
          d_functionalValueAfterAlphUpdateChk = functionalValueAfterAlphUpdate;
        d_etaChk                   = eta;
        d_etaPChk                  = etaP;
        d_alphaChk                 = alpha;
        d_etaAlphaZeroChk          = etaAlphaZero;
        d_lineSearchRestartIterChk = iter;

        MPI_Barrier(mpi_communicator);
        problem.save();

        // FIXME: check whether >1 or >=1 is the correct choice
        if (iter >= 1)
          {
            int isSuccess = 0;

            d_gradMax = 0.0;
            for (unsigned int i = 0; i < d_numberUnknowns; ++i)
              {
                if (std::abs(d_gradient[i]) > d_gradMax)
                  d_gradMax = std::abs(d_gradient[i]);
              }

            double condition1 =
              (functionalValueAfterAlphUpdate - functionValue) -
              (c1 * alpha * etaAlphaZero);
            double condition2 = std::abs(eta) - c2 * std::abs(etaAlphaZero);
            if (condition1 <= 1e-08 && condition2 <= 1e-08)
              {
                if (debugLevel >= 1)
                  pcout << "Satisfied Wolfe condition " << std::endl;

                isSuccess = 1;
              }
            else if (problem.isConverged())
              {
                isSuccess = 1;
              }

            MPI_Bcast(&(isSuccess), 1, MPI_INT, 0, mpi_communicator);

            if (isSuccess == 1)
              {
                if (debugLevel >= 1)
                  pcout << " Secant line-search iterations completed "
                        << std::endl;

                return SUCCESS;
              }
          }

        //
        // update alpha
        //
        alphaNew = alpha * eta / (etaP - eta);

        //
        // output
        //
        if (debugLevel >= 2)
          pcout << "Line search iteration: " << iter
                << " alphaNew: " << alphaNew << " alpha: " << alpha
                << "  eta: " << eta << " etaP: " << etaP << std::endl;
        else if (debugLevel >= 1)
          pcout << "Line search iteration: " << iter << std::endl;

        //
        // update unknowns
        //
        if (iter == 0)
          {
            d_isCGRestartDueToLargeIncrement =
              !(updateSolution(alphaNew - d_lineSearchDampingParameter,
                               d_conjugateDirection,
                               problem));

            if (d_isCGRestartDueToLargeIncrement)
              {
                if (debugLevel >= 1)
                  pcout
                    << " Secant line-search failed as maximum increment bound is exceeded. CG PRP will be restarted "
                    << std::endl;

                return LINESEARCH_FAILED;
              }
          }
        else
          {
            d_isCGRestartDueToLargeIncrement =
              !(updateSolution(alphaNew, d_conjugateDirection, problem));
            if (d_isCGRestartDueToLargeIncrement)
              {
                if (debugLevel >= 1)
                  pcout
                    << " Secant line-search failed as maximum increment bound is exceeded. CG PRP will be restarted "
                    << std::endl;

                return LINESEARCH_FAILED;
              }
          }

        //
        // update etaP, alphaP and alpha
        //
        etaP  = eta;
        alpha = alphaNew;
      }

    if (debugLevel >= 1)
      pcout << "Maximum number of line-search iterations reached " << std::endl;

    //
    //
    //
    return MAX_ITER_REACHED;
  }



  //
  // Perform problem minimization.
  //
  nonLinearSolver::ReturnValueType
  cgPRPNonLinearSolver::solve(nonlinearSolverProblem &problem,
                              const std::string       checkpointFileName,
                              const bool              restart)
  {
    //
    // get total number of unknowns in the problem.
    //
    d_numberUnknowns = problem.getNumberUnknowns();

    //
    // resize d_unknownCountFlag with numberUnknown and initialize to 1
    //
    d_unknownCountFlag.resize(d_numberUnknowns, 1);

    //
    // allocate space for conjugate direction, gradient and old steepest
    // direction values.
    //
    d_conjugateDirection.resize(d_numberUnknowns);
    d_gradient.resize(d_numberUnknowns);
    d_steepestDirectionOld.resize(d_numberUnknowns);

    //
    // initialize delta new and direction
    //
    if (!restart)
      {
        //
        // compute initial values of problem and problem gradient
        //
        problem.gradient(d_gradient);

        initializeDirection();
      }
    else
      {
        load(checkpointFileName);
        MPI_Barrier(mpi_communicator);
        d_useSingleAtomSolutionsInitialGuess = true;

        // compute deltaNew
        d_deltaNew = 0.0;
        d_gradMax  = 0.0;
        for (unsigned int i = 0; i < d_numberUnknowns; ++i)
          {
            const double r = d_steepestDirectionOld[i];
            d_deltaNew += d_unknownCountFlag[i] * r * r;

            if (std::abs(-r) > d_gradMax)
              d_gradMax = std::abs(-r);
          }
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



    for (d_iter = 0; d_iter < d_maxNumberIterations; ++d_iter)
      {
        //
        // compute L2-norm of the residual (gradient)
        //
        const double residualNorm = computeResidualL2Norm();


        if (d_debugLevel >= 2)
          pcout << "CG Iter. no. | delta new | residual norm "
                   "| residual norm avg"
                << std::endl;
        else if (d_debugLevel >= 1)
          pcout << "CG Iter. no. " << d_iter + 1 << std::endl;
        //
        // output at the begining of the iteration
        //
        if (d_debugLevel >= 2)
          pcout << d_iter + 1 << " " << d_deltaNew << " " << residualNorm << " "
                << residualNorm / d_numberUnknowns << " " << std::endl;

        //
        // perform line search along direction
        //
        ReturnValueType lineSearchReturnValue =
          lineSearch(problem,
                     d_lineSearchTolerance,
                     d_lineSearchMaxIterations,
                     d_debugLevel,
                     checkpointFileName,
                     (restart && d_iter == 0) ? d_lineSearchRestartIterChk : -1,
                     restart && d_iter == 0);

        //
        // evaluate gradient
        //
        problem.gradient(d_gradient);

        //
        // update values of delta_new and delta_mid
        //
        d_deltaOld = d_deltaNew;
        computeDeltas();

        //
        // compute PRP beta
        //

        d_beta = (d_deltaNew - d_deltaMid) / d_deltaOld;

        if (d_debugLevel >= 2)
          pcout << " CG- d_beta: " << d_beta << std::endl;

        unsigned int isBetaZero = 0;
        if (d_beta <= 0 || d_isCGRestartDueToLargeIncrement)
          {
            if (d_debugLevel >= 2 && d_beta <= 0)
              pcout << " Negative d_beta- setting it to zero " << std::endl;
            isBetaZero                       = 1;
            d_isCGRestartDueToLargeIncrement = false;
          }
        MPI_Bcast(&(isBetaZero), 1, MPI_INT, 0, mpi_communicator);
        if (isBetaZero == 1)
          d_beta = 0;
        //
        // update direction
        //
        updateDirection();

        // save current conjugate direction and solution
        // if (!checkpointFileName.empty())
        //{
        //	save(checkpointFileName);
        //	problem.save();
        //}

        //
        // check for convergence
        //
        unsigned int isBreak = 0;

        d_gradMax = 0.0;
        for (unsigned int i = 0; i < d_numberUnknowns; ++i)
          {
            if (std::abs(d_gradient[i]) > d_gradMax)
              d_gradMax = std::abs(d_gradient[i]);
          }

        if (problem.isConverged())
          isBreak = 1;
        MPI_Bcast(&(isSuccess), 1, MPI_INT, 0, mpi_communicator);
        if (isBreak == 1)
          break;
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
            pcout << "Non-linerar Conjugate Gradient solver converged after "
                  << d_iter + 1 << " iterations." << std::endl;
          }
        else
          {
            pcout
              << "Non-linear Conjugate Gradient solver failed to converge after "
              << d_iter << " iterations." << std::endl;
          }
      }

    //
    //
    //
    return returnValue;
  }
} // namespace dftfe
