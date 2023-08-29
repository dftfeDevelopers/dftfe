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
    const bool         usePreconditioner,
    const bool         useRFOStep,
    const unsigned int maxNumberIterations,
    const unsigned int debugLevel,
    const MPI_Comm &   mpi_comm_parent,
    const double       trustRadius_maximum,
    const double       trustRadius_initial,
    const double       trustRadius_minimum,
    const bool         isCurvatureOnlyLineSearchStoppingCondition)
    : nonLinearSolver(debugLevel, maxNumberIterations)
    , mpi_communicator(mpi_comm_parent)
    , d_usePreconditioner(usePreconditioner)
    , d_useRFOStep(useRFOStep)
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
    , d_isCurvatureOnlyLineSearchStoppingCondition(
        isCurvatureOnlyLineSearchStoppingCondition)
  {
    d_isReset                            = 0;
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
  namespace internalBFGS
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
    // Compute Weighted L2-norm square a^TPa  and product Pa.
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
      double res = ddot_(&n, a.data(), &one, Pdx.data(), &one);
      a          = Pdx;
      return res;
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
    // Compute rank one symmetric update  P=P+faa^T.
    //
    void
    computeSymmetricRankOneUpdate(double &             f,
                                  std::vector<double> &a,
                                  std::vector<double> &P)
    {
      const unsigned int one  = 1;
      const char         uplo = 'U';
      const unsigned int n    = a.size();
      if (n * n != P.size())
        {
          std::cout << "DEBUG check dimensions Pnorm" << std::endl;
        }
      dsyr_(&uplo, &n, &f, a.data(), &one, P.data(), &n);
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
    // Compute lowest n generalized eigenvalues of symmetric GEP Ax=lBx and
    // corresponding eigenvectors.
    //
    void
    computeEigenSpectrumGeneralized(std::vector<double>  A,
                                    std::vector<double>  B,
                                    const int            n,
                                    std::vector<double> &eigVals,
                                    std::vector<double> &eigVecs)
    {
      int                 info;
      const int           one             = 1;
      const int           dimensionMatrix = std::sqrt(A.size());
      std::vector<double> eigenValues(n, 0.0);
      std::vector<double> eigenVectors(dimensionMatrix, 0.0);
      const int lwork = 8 * dimensionMatrix, liwork = 5 * dimensionMatrix;
      std::vector<int>    iwork(liwork, 0);
      std::vector<int>    ifail(dimensionMatrix, 0);
      const char          jobz = 'V', uplo = 'U', range = 'I', cmach = 'S';
      const double        abstol = 2 * dlamch_(&cmach);
      std::vector<double> work(lwork);
      int                 nEigVals;

      dsygvx_(&one,
              &jobz,
              &range,
              &uplo,
              &dimensionMatrix,
              A.data(),
              &dimensionMatrix,
              B.data(),
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
    //  compute |A|^(1/n) for a symmetric matrix A
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

  } // namespace internalBFGS


  //
  // initialize hessian, either preconditioner or identity matrix.
  //
  void
  BFGSNonLinearSolver::initializeHessian(nonlinearSolverProblem &problem)
  {
    d_hessian.clear();
    if (d_usePreconditioner)
      {
        if (d_debugLevel >= 1)
          pcout << "Using preconditioner for initial Hessian guess."
                << std::endl;
        problem.precondition(d_hessian, d_gradient);
      }
    else
      {
        if (d_debugLevel >= 1)
          pcout << "Using Identity matrix for initial Hessian guess."
                << std::endl;
        d_hessian.resize(d_numberUnknowns * d_numberUnknowns, 0.0);
        for (int i = 0; i < d_numberUnknowns; ++i)
          {
            d_hessian[i + i * d_numberUnknowns] = 1.0;
          }
      }
    d_Srfo      = d_hessian;
    double detS = internalBFGS::computeDetNormalizationFactor(d_Srfo);
    if (d_debugLevel >= 2)
      pcout << "Normalizing factor for Hessian matrix: " << detS << std::endl;
    for (auto i = 0; i < d_Srfo.size(); ++i)
      {
        d_Srfo[i] /= detS;
      }
    d_hessian = d_Srfo;
  }

  //
  // Update Hessian according to damped BFGS rule: Procedure 18.2 of Nocedal and
  // Wright.
  //
  void
  BFGSNonLinearSolver::updateHessian()
  {
    std::vector<double> delta_g(d_numberUnknowns, 0.0);

    for (auto i = 0; i < d_numberUnknowns; ++i)
      {
        delta_g[i] = d_gradientNew[i] - d_gradient[i];
      }

    std::vector<double> Hdx    = d_deltaXNew;
    double              dxtHdx = internalBFGS::computePNorm(Hdx, d_hessian);
    double              dgtdx  = internalBFGS::dot(d_deltaXNew, delta_g);
    if (d_stepAccepted)
      {
        double theta =
          dgtdx >= 0.2 * dxtHdx ? 1 : 0.8 * dxtHdx / (dxtHdx - dgtdx);
        if (theta != 1 && d_debugLevel >= 2)
          {
            pcout << "Using damped BFGS update with theta = " << theta
                  << std::endl;
          }
        std::vector<double> r(d_numberUnknowns, 0.0);
        for (auto i = 0; i < d_numberUnknowns; ++i)
          {
            r[i] = theta * delta_g[i] + (1.0 - theta) * Hdx[i];
          }
        double factor = 1.0 / internalBFGS::dot(d_deltaXNew, r);
        internalBFGS::computeSymmetricRankOneUpdate(factor, r, d_hessian);
        factor = -1.0 / dxtHdx;
        internalBFGS::computeSymmetricRankOneUpdate(factor, Hdx, d_hessian);
      }
  }

  //
  // Scale hessian according to eqn 6.20 of Nocedal and Wright.
  // TODO : Figure out the proper scaling of the preconditoner
  //
  void
  BFGSNonLinearSolver::scaleHessian()
  {
    std::vector<double> delta_g(d_numberUnknowns, 0.0);

    for (auto i = 0; i < d_numberUnknowns; ++i)
      {
        delta_g[i] = d_gradientNew[i] - d_gradient[i];
      }

    if (internalBFGS::dot(delta_g, d_deltaXNew) > 0)
      {
        double scalingFactor =
          internalBFGS::dot(delta_g, delta_g) /
          internalBFGS::computePdot(delta_g, d_deltaXNew, d_hessian);
        if (d_debugLevel >= 1)
          {
            pcout << "Scaling Hessian with scaling factor: " << scalingFactor
                  << std::endl;
          }
        for (auto i = 0; i < d_hessian.size(); ++i)
          {
            d_hessian[i] = d_hessian[i] * scalingFactor;
          }
        d_hessianScaled = true;
        d_trustRadius   = d_trustRadiusMax;
      }
  }

  //
  // Compute step using the Rational Function Method.
  //
  void
  BFGSNonLinearSolver::computeRFOStep()
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
    std::vector<double> eigenValues(1, 0.0);
    std::vector<double> eigenVectors(d_numberUnknowns + 1, 0.0);
    internalBFGS::computeEigenSpectrumGeneralized(
      augmentedHessian, augmentedSrfo, 1, eigenValues, eigenVectors);
    for (auto i = 0; i < d_numberUnknowns; ++i)
      {
        d_deltaXNew[i] = eigenVectors[i] / eigenVectors[d_numberUnknowns];
      }
    d_normDeltaXnew = internalBFGS::computeLInfNorm(d_deltaXNew);
    if (d_debugLevel >= 1)
      pcout << "Computed RFO Step, max norm of step: " << d_normDeltaXnew
            << std::endl;
  }


  //
  // Compute the Quasi-Newton Step.
  //
  void
  BFGSNonLinearSolver::computeNewtonStep()
  {
    for (auto i = 0; i < d_numberUnknowns; ++i)
      {
        d_deltaXNew[i] = -d_gradient[i];
      }

    internalBFGS::linearSolve(d_hessian, d_deltaXNew);

    d_normDeltaXnew = internalBFGS::computeLInfNorm(d_deltaXNew);
    if (d_debugLevel >= 1)
      pcout << "Computed Quasi-Newton Step, max norm of step: "
            << d_normDeltaXnew << std::endl;
  }

  //
  // Compute the final update step using the trust radius and whether or not the
  // previous step was accepted.
  //
  void
  BFGSNonLinearSolver::computeStep()
  {
    for (auto i = 0; i < d_numberUnknowns; ++i)
      {
        d_deltaXNew[i] *= d_trustRadius / d_normDeltaXnew;
      }
    if (d_debugLevel >= 2)
      pcout << "Step scaled to trust radius, max norm of step: "
            << internalBFGS::computeLInfNorm(d_deltaXNew) << std::endl;
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
  // Check if the step satifies the Strong Wolfe conditons.
  // TODO : Allow user to change wolfe conditon parameters?
  //
  void
  BFGSNonLinearSolver::checkWolfe()
  {
    double gtdx  = internalBFGS::dot(d_deltaXNew, d_gradient);
    double gntdx = internalBFGS::dot(d_deltaXNew, d_gradientNew);

    if (!d_isCurvatureOnlyLineSearchStoppingCondition)
      d_wolfeSufficientDec = (d_valueNew[0] - d_value[0]) < 0.01 * gtdx;
    else
      d_wolfeSufficientDec = false;
    d_wolfeCurvature = std::abs(gntdx) < 0.9 * std::abs(gtdx);
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
  // Estimate the trust radius for the next step based on the previous step.
  // Check for trust radius max/min conditons and reset BFGS if needed
  //
  void
  BFGSNonLinearSolver::computeTrustRadius(nonlinearSolverProblem &problem)
  {
    if (d_iter == 0)
      {
        d_trustRadius = d_trustRadiusMax < d_normDeltaXnew ? d_trustRadiusMax :
                                                             d_normDeltaXnew;
      }
    else if (d_stepAccepted)
      {
        double ampfactor =
          internalBFGS::computeLInfNorm(d_deltaX) > d_trustRadius + 1e-8 ? 1.5 :
                                                                           1.1;
        ampfactor     = d_wolfeSatisfied ? 2 * ampfactor : ampfactor;
        d_trustRadius = ampfactor * d_trustRadius < d_normDeltaXnew ?
                          ampfactor * d_trustRadius :
                          d_normDeltaXnew;
        d_trustRadius =
          d_trustRadius < d_trustRadiusMax ? d_trustRadius : d_trustRadiusMax;
      }
    else
      {
        double gtdx = internalBFGS::dot(d_deltaX, d_gradient);
        if (!d_isCurvatureOnlyLineSearchStoppingCondition)
          d_trustRadius =
            -0.5 * gtdx * d_trustRadius / ((d_valueNew[0] - d_value[0]) - gtdx);
        else
          d_trustRadius /= 2;
        if (d_trustRadius < d_trustRadiusMin)
          {
            if (d_debugLevel >= 1)
              pcout
                << "Resetting BFGS Hessian as trust radius is below allowed threshold."
                << std::endl;
            initializeHessian(problem);
            d_trustRadius = d_trustRadiusInitial;
            if (d_useRFOStep)
              {
                computeRFOStep();
              }
            else
              {
                computeNewtonStep();
              }
            d_trustRadius =
              d_trustRadius < d_normDeltaXnew ? d_trustRadius : d_normDeltaXnew;
            d_hessianScaled = false;
            d_isReset       = d_isReset == 1 ? 2 : 1;
          }
      }
    d_trustRadius =
      d_trustRadius < d_trustRadiusMax ? d_trustRadius : d_trustRadiusMax;
    if (d_debugLevel >= 1)
      pcout << "Estimated trust radius: " << d_trustRadius << std::endl;
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
  // save checkpoint files.
  //
  void
  BFGSNonLinearSolver::save(const std::string &checkpointFileName)
  {
    if (d_debugLevel >= 2)
      {
        pcout << "Saving BFGS data to " << checkpointFileName << std::endl;
      }
    std::vector<std::vector<double>> data;
    for (unsigned int i = 0; i < d_deltaX.size(); ++i)
      data.push_back(std::vector<double>(1, d_deltaX[i]));
    for (unsigned int i = 0; i < d_gradient.size(); ++i)
      data.push_back(std::vector<double>(1, d_gradient[i]));
    for (unsigned int i = 0; i < d_hessian.size(); ++i)
      data.push_back(std::vector<double>(1, d_hessian[i]));
    for (unsigned int i = 0; i < d_Srfo.size(); ++i)
      data.push_back(std::vector<double>(1, d_Srfo[i]));
    data.push_back(d_value);
    data.push_back(d_valueNew);
    data.push_back(std::vector<double>(1, d_trustRadius));
    data.push_back(std::vector<double>(1, d_iter));
    data.push_back(std::vector<double>(1, (double)d_stepAccepted));


    dftUtils::writeDataIntoFile(data, checkpointFileName, mpi_communicator);
  }


  //
  // load from checkpoint files.
  //
  void
  BFGSNonLinearSolver::load(const std::string &checkpointFileName)
  {
    if (d_debugLevel >= 1)
      {
        pcout << "Loading BFGS data from " << checkpointFileName << std::endl;
      }
    std::vector<std::vector<double>> data;
    dftUtils::readFile(1, data, checkpointFileName);
    AssertThrow(
      data.size() ==
        (2 * d_numberUnknowns + 2 * d_numberUnknowns * d_numberUnknowns + 5),
      dealii::ExcMessage(std::string(
        "DFT-FE Error: data size of bfgs solver checkpoint file is incorrect.")));

    d_deltaX.resize(d_numberUnknowns);
    d_gradient.resize(d_numberUnknowns);
    d_hessian.resize(d_numberUnknowns * d_numberUnknowns);
    d_Srfo.resize(d_numberUnknowns * d_numberUnknowns);
    d_value.resize(1);
    d_valueNew.resize(1);
    for (unsigned int i = 0; i < d_numberUnknowns; ++i)
      d_deltaX[i] = data[i][0];

    for (unsigned int i = 0; i < d_numberUnknowns; ++i)
      d_gradient[i] = data[i + d_numberUnknowns][0];

    for (unsigned int i = 0; i < d_numberUnknowns * d_numberUnknowns; ++i)
      d_hessian[i] = data[i + 2 * d_numberUnknowns][0];

    for (unsigned int i = 0; i < d_numberUnknowns * d_numberUnknowns; ++i)
      d_Srfo[i] =
        data[i + 2 * d_numberUnknowns + d_numberUnknowns * d_numberUnknowns][0];

    d_value[0] =
      data[2 * d_numberUnknowns + 2 * d_numberUnknowns * d_numberUnknowns][0];

    d_valueNew[0] = data[1 + 2 * d_numberUnknowns +
                         2 * d_numberUnknowns * d_numberUnknowns][0];

    d_trustRadius = data[2 + 2 * d_numberUnknowns +
                         2 * d_numberUnknowns * d_numberUnknowns][0];

    d_iter = (int)data[3 + 2 * d_numberUnknowns +
                       2 * d_numberUnknowns * d_numberUnknowns][0] +
             1;

    d_stepAccepted = data[4 + 2 * d_numberUnknowns +
                          2 * d_numberUnknowns * d_numberUnknowns][0] == 1.0;
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

    //
    // initialize delta new and direction
    //
    if (!restart)
      {
        d_trustRadius   = d_trustRadiusInitial;
        d_normDeltaXnew = d_trustRadiusInitial;
        d_stepAccepted  = true;
        d_hessianScaled = false;
        d_iter          = 0;
        //
        // compute initial values of problem and problem gradient
        //
        problem.gradient(d_gradient);
        problem.value(d_value);

        initializeHessian(problem);
        MPI_Barrier(mpi_communicator);
        problem.save();
      }
    else
      {
        load(checkpointFileName);
        MPI_Barrier(mpi_communicator);
        d_useSingleAtomSolutionsInitialGuess = true;
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
          pcout << "BFGS Step no. " << d_iter + 1 << std::endl;


        // Compute the update step
        //
        if (d_useRFOStep)
          {
            computeRFOStep();
          }
        else
          {
            computeNewtonStep();
          }
        computeTrustRadius(problem);
        MPI_Bcast(&(d_isReset), 1, MPI_INT, 0, mpi_communicator);
        if (d_isReset == 2)
          break;

        computeStep();

        updateSolution(d_updateVector, problem);
        //
        // evaluate gradient
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
            if (d_iter == 0 || !d_hessianScaled)
              {
                scaleHessian();
              }
            updateHessian();

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
        MPI_Barrier(mpi_communicator);
        problem.save();
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
    if (d_isReset == 2)
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
        else if (MAX_ITER_REACHED)
          {
            pcout << "BFGS solver failed to converge after " << d_iter
                  << " iterations." << std::endl;
          }
        else
          {
            pcout << "BFGS solver failed to converge after " << d_iter
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
