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
// @author Bikash Kanungo, Vishal Subramanian
//

#include <BFGSInverseDFTSolver.h>

namespace dftfe
{
  namespace
  {
    template <typename T>
    std::string
    to_string_with_precision(const T a_value, const int n = 6)
    {
      std::ostringstream out;
      out.precision(n);
      out << std::fixed << a_value;
      return out.str();
    }
    //
    // y = a*x + b*y
    //
    void
    vecAdd(const distributedCPUVec<double> &x,
           distributedCPUVec<double> &      y,
           const double                     a,
           const double                     b)
    {
      const unsigned int N = x.locally_owned_size();
      for (unsigned int i = 0; i < N; ++i)
        y.local_element(i) = a * x.local_element(i) + b * y.local_element(i);
    }

    void
    vecScale(distributedCPUVec<double> &x, const double a)
    {
      const unsigned int N = x.locally_owned_size();
      for (unsigned int i = 0; i < N; ++i)
        x.local_element(i) *= a;
    }

  } // namespace

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  BFGSInverseDFTSolver<FEOrder, FEOrderElectro, memorySpace>::
    BFGSInverseDFTSolver(int             numComponents,
                         double          tol,
                         double          lineSearchTol,
                         unsigned int    maxNumIter,
                         int             historySize,
                         int             numLineSearch,
                         const MPI_Comm &mpi_comm_parent)
    : d_numComponents(numComponents)
    , d_tol(tol)
    , d_lineSearchTol(lineSearchTol)
    , d_maxNumIter(maxNumIter)
    , d_historySize(historySize)
    , d_numLineSearch(numLineSearch)
    , d_k(numComponents, 0)
    , d_y(numComponents, std::list<distributedCPUVec<double>>(0))
    , d_s(numComponents, std::list<distributedCPUVec<double>>(0))
    , d_rho(numComponents, std::list<double>(0))
    , pcout(std::cout,
            (dealii::Utilities::MPI::this_mpi_process(mpi_comm_parent) == 0))
  {}

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  BFGSInverseDFTSolver<FEOrder, FEOrderElectro, memorySpace>::
    inverseJacobianTimesVec(
      const distributedCPUVec<double> &g,
      distributedCPUVec<double> &      z,
      const unsigned int               component,
      InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>
        &iDFTSolverFunction)
  {
    int                       N            = d_k[component];
    distributedCPUVec<double> q            = g;
    auto                      itReverseY   = d_y[component].rbegin();
    auto                      itReverseS   = d_s[component].rbegin();
    auto                      itReverseRho = d_rho[component].rbegin();
    double                    gamma        = 1.0;
    if (itReverseY != d_y[component].rend())
      {
        // gamma = (s_k^T*y_k)/(y_k^Ty_K)
        std::vector<double> dotProd1(1, 0.0);
        std::vector<double> dotProd2(1, 0.0);
        iDFTSolverFunction.dotProduct(*itReverseS, *itReverseY, 1, dotProd1);
        iDFTSolverFunction.dotProduct(*itReverseY, *itReverseY, 1, dotProd2);
        gamma = dotProd1[0] / dotProd2[0];
      }

    std::vector<double> alpha(N, 0.0);
    std::vector<double> beta(N, 0.0);
    for (int i = N - 1; i >= 0; --i)
      {
        std::vector<double> dotProd(1, 0.0);
        iDFTSolverFunction.dotProduct(*itReverseS, q, 1, dotProd);
        alpha[i] = (*itReverseRho) * dotProd[0];
        // q = q - alpha*y
        vecAdd(*itReverseY, q, -alpha[i], 1.0);
        ++itReverseY;
        ++itReverseS;
        ++itReverseRho;
      }

    z = q;
    vecScale(z, gamma);

    auto itY   = d_y[component].begin();
    auto itS   = d_s[component].begin();
    auto itRho = d_rho[component].begin();
    for (int i = 0; i < N; ++i)
      {
        std::vector<double> dotProd(1, 0.0);
        iDFTSolverFunction.dotProduct(*itY, z, 1, dotProd);
        beta[i] = (*itRho) * dotProd[0];
        // z = z + (alpha-beta)*s
        vecAdd(*itS, z, alpha[i] - beta[i], 1.0);
        ++itY;
        ++itS;
        ++itRho;
      }
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  BFGSInverseDFTSolver<FEOrder, FEOrderElectro, memorySpace>::fnormCP(
    const std::vector<distributedCPUVec<double>> &x,
    const std::vector<distributedCPUVec<double>> &p,
    const std::vector<double> &                   alpha,
    std::vector<double> &                         fnorms,
    InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>
      &iDFTSolverFunction)
  {
    std::vector<distributedCPUVec<double>> xnew(d_numComponents);
    for (unsigned int iComp = 0; iComp < d_numComponents; ++iComp)
      {
        xnew[iComp].reinit(x[iComp], false);
        // xnew = x + alpha*p
        xnew[iComp] = x[iComp];
        vecAdd(p[iComp], xnew[iComp], alpha[iComp], 1.0);
      }

    std::vector<distributedCPUVec<double>> g(d_numComponents);
    for (unsigned int iComp = 0; iComp < d_numComponents; ++iComp)
      {
        g[iComp].reinit(x[iComp], false);
        g[iComp] = 0.0;
      }

    std::vector<double> L(d_numComponents);
    iDFTSolverFunction.getForceVector(xnew, g, L);
    fnorms.resize(d_numComponents, 0.0);
    for (unsigned int iComp = 0; iComp < d_numComponents; ++iComp)
      {
        std::vector<double> dotProd(1, 0.0);
        iDFTSolverFunction.dotProduct(g[iComp], d_p[iComp], 1, dotProd);
        fnorms[iComp] = dotProd[0];
      }
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  BFGSInverseDFTSolver<FEOrder, FEOrderElectro, memorySpace>::solveLineSearchCP(
    std::vector<std::vector<double>> &lambda,
    std::vector<std::vector<double>> &f,
    const int                         maxIter,
    const double                      tolerance,
    InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>
      &iDFTSolverFunction)
  {
    for (unsigned int iComp = 0; iComp < d_numComponents; ++iComp)
      {
        dftfe::utils::throwException(
          lambda[iComp].size() >= 2,
          "At least two initial values are need for a secant method.");
      }

    std::vector<double> lambda0(d_numComponents);
    std::vector<double> lambda1(d_numComponents);
    std::vector<double> f0(d_numComponents);
    std::vector<double> f1(d_numComponents);
    for (unsigned int i = 0; i < maxIter; ++i)
      {
        int N = lambda[0].size();
        int M = f[0].size();
        for (unsigned int iComp = 0; iComp < d_numComponents; ++iComp)
          {
            lambda0[iComp] = lambda[iComp][N - 2];
            lambda1[iComp] = lambda[iComp][N - 1];
          }
        if (M > 0)
          {
            for (unsigned int iComp = 0; iComp < d_numComponents; ++iComp)
              f0[iComp] = f[iComp][M - 1];
          }
        else
          {
            fnormCP(d_x, d_p, lambda0, f0, iDFTSolverFunction);
            for (unsigned int iComp = 0; iComp < d_numComponents; ++iComp)
              f[iComp].push_back(f0[iComp]);
          }

        if (std::all_of(f0.begin(), f0.end(), [tolerance](double x) {
              return std::sqrt(std::fabs(x)) < tolerance;
            }))
          {
            // remove the last element (i.e., lambda1)
            for (unsigned int iComp = 0; iComp < d_numComponents; ++iComp)
              lambda[iComp].pop_back();

            std::cout << "f0 in secantCP below tolerance" << std::endl;
            break;
          }

        this->fnormCP(d_x, d_p, lambda1, f1, iDFTSolverFunction);
        for (unsigned int iComp = 0; iComp < d_numComponents; ++iComp)
          f[iComp].push_back(f1[iComp]);

        if (std::all_of(f1.begin(), f1.end(), [tolerance](double x) {
              return std::sqrt(std::fabs(x)) < tolerance;
            }))
          {
            pcout << "f1 in secantCP below tolerance" << std::endl;
            break;
          }

        //
        // TODO Fetch the tolerance from dftParams
        //
        for (unsigned int iComp = 0; iComp < d_numComponents; ++iComp)
          {
            if (fabs((f1[iComp] - f0[iComp]) / f1[iComp]) < 1e-15)
              {
                std::string message =
                  "Secant line search failed, possibly because f'(x) = 0 in the interval"
                  "The last two alphas and their function values are:\n"
                  " Alphas: (" +
                  to_string_with_precision(lambda0[iComp], 18) + "," +
                  to_string_with_precision(lambda1[iComp], 18) + ")\t" +
                  "fs: (" + to_string_with_precision(f0[iComp], 18) + "," +
                  to_string_with_precision(f1[iComp], 18) + ").";
                dftfe::utils::throwException(false, message);
              }
          }

        for (unsigned int iComp = 0; iComp < d_numComponents; ++iComp)
          {
            double s =
              (f1[iComp] - f0[iComp]) / (lambda1[iComp] - lambda0[iComp]);
            /* if the solve is going in the wrong direction, reverse it */
            if (s > 0.0)
              s = -s;
            double lambdaNext = lambda1[iComp] - f1[iComp] / s;
            // switch directions if we stepped out of bounds
            if (lambdaNext < 0.0)
              lambdaNext = lambda1[iComp] + f1[iComp] / s;

            lambda[iComp].push_back(lambdaNext);
          }
      }
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  BFGSInverseDFTSolver<FEOrder, FEOrderElectro, memorySpace>::
    solveLineSearchSecantLoss(
      std::vector<std::vector<double>> &lambda,
      std::vector<std::vector<double>> &f,
      const int                         maxIter,
      const double                      tolerance,
      InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>
        &iDFTSolverFunction)
  {
    dftfe::utils::throwException(
      false,
      "BFGSInverseDFTSolver::solveLineSearchSecantLoss() not implemented yet.");
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  BFGSInverseDFTSolver<FEOrder, FEOrderElectro, memorySpace>::
    solveLineSearchSecantForceNorm(
      std::vector<std::vector<double>> &lambda,
      std::vector<std::vector<double>> &f,
      const int                         maxIter,
      const double                      tolerance,
      InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>
        &iDFTSolverFunction)
  {
    dftfe::utils::throwException(
      false,
      "BFGSInverseDFTSolver::solveLineSearchSecantForceNorm() not implemented yet.");
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  void
  BFGSInverseDFTSolver<FEOrder, FEOrderElectro, memorySpace>::solve(
    InverseDFTSolverFunction<FEOrder, FEOrderElectro, memorySpace>
      &                                iDFTSolverFunction,
    const BFGSInverseDFTSolver::LSType lsType)
  {
    std::vector<distributedCPUVec<double>> y(d_numComponents);
    std::vector<distributedCPUVec<double>> s(d_numComponents);

    d_x = iDFTSolverFunction.getInitialGuess();

    //
    // allocate d_g, d_p, y, and s to be of the same size as d_x
    // and initialize them to zero
    //
    d_g.resize(d_numComponents);
    d_p.resize(d_numComponents);
    for (unsigned int iComp = 0; iComp < d_numComponents; ++iComp)
      {
        d_g[iComp].reinit(d_x[iComp], false);
        d_p[iComp].reinit(d_x[iComp], false);
        y[iComp].reinit(d_x[iComp], false);
        s[iComp].reinit(d_x[iComp], false);
        d_g[iComp] = 0.0;
        d_p[iComp] = 0.0;
        y[iComp]   = 0.0;
        s[iComp]   = 0.0;
      }

    std::vector<double> L(d_numComponents);
    for (unsigned int iter = 0; iter < d_maxNumIter; ++iter)
      {
        pcout << " bfgs iter = " << iter << "\n";
        if (iter == 0)
          {
            iDFTSolverFunction.getForceVector(d_x, d_g, L);
          }

        std::vector<double>              gnorm(d_numComponents);
        std::vector<std::vector<double>> lsNorm(d_numComponents);
        std::vector<std::vector<double>> lambdas(d_numComponents);
        std::vector<bool>                hasConverged(d_numComponents, false);
        for (unsigned int iComp = 0; iComp < d_numComponents; ++iComp)
          {
            std::vector<double> dotProd(1, 0.0);
            iDFTSolverFunction.dotProduct(d_g[iComp], d_g[iComp], 1, dotProd);
            const double forceNorm = std::sqrt(dotProd[0]);
            pcout << "BFGS-Inverse " << iter << " forceNorm[" << iComp
                  << "]: " << forceNorm << std::endl;

            if (forceNorm < d_tol)
              hasConverged[iComp] = true;

            inverseJacobianTimesVec(d_g[iComp],
                                    d_p[iComp],
                                    iComp,
                                    iDFTSolverFunction);
            vecScale(d_p[iComp], -1.0);

            double f0 = 0.0;
            if (lsType == BFGSInverseDFTSolver::LSType::SECANT_LOSS)
              {
                f0 = L[iComp];
              }
            else if (lsType == BFGSInverseDFTSolver::LSType::SECANT_FORCE_NORM)
              {
                f0 = forceNorm;
              }
            else if (lsType == BFGSInverseDFTSolver::LSType::CP)
              {
                std::vector<double> gDotP(1, 0.0);
                iDFTSolverFunction.dotProduct(d_g[iComp], d_p[iComp], 1, gDotP);
                f0 = gDotP[0];
              }
            else
              {
                dftfe::utils::throwException(
                  false,
                  "Invalid line search type passed to BFGSInverseDFTSolver.solve()");
              }

            lambdas[iComp].push_back(0.0);
            lambdas[iComp].push_back(1.0);
            lsNorm[iComp].push_back(f0);
          }


        for (unsigned int iComp = 0; iComp < d_numComponents; ++iComp)
          {
            pcout << " printing the loss val for iComp = " << iComp
                  << " loss = " << L[iComp] << "\n";
          }

        if (std::all_of(hasConverged.begin(), hasConverged.end(), [](bool x) {
              return x;
            }))
          {
            break;
          }


        if (lsType == BFGSInverseDFTSolver::LSType::SECANT_LOSS)
          {
            this->solveLineSearchSecantLoss(lambdas,
                                            lsNorm,
                                            d_numLineSearch,
                                            d_lineSearchTol,
                                            iDFTSolverFunction);
          }
        else if (lsType == BFGSInverseDFTSolver::LSType::SECANT_FORCE_NORM)
          {
            this->solveLineSearchSecantForceNorm(lambdas,
                                                 lsNorm,
                                                 d_numLineSearch,
                                                 d_lineSearchTol,
                                                 iDFTSolverFunction);
          }
        else
          {
            this->solveLineSearchCP(lambdas,
                                    lsNorm,
                                    d_numLineSearch,
                                    d_lineSearchTol,
                                    iDFTSolverFunction);
          }

        double optimalLambda = 1.0;
        for (unsigned int iComp = 0; iComp < d_numComponents; ++iComp)
          {
            const unsigned int numLambdas = lambdas[iComp].size();
            pcout << "Line search for component" << iComp << std::endl;
            for (unsigned int i = 0; i < numLambdas - 1; ++i)
              {
                std::cout << "Lambda: " << lambdas[iComp][i]
                          << " Norm: " << lsNorm[iComp][i] << std::endl;
              }

            //
            // If the lambda is negative set lambda to 1.0, else it will lead to
            // ascent instead of descent along the gradient
            //
            optimalLambda = (lambdas[iComp][numLambdas - 1] > 0.0) ?
                              lambdas[iComp][numLambdas - 1] :
                              1.0;
            pcout << "Optimal lambda for iComp " << iComp << " is "
                  << optimalLambda << std::endl;

            double alpha = optimalLambda;

            // s = alpha*d_p
            s[iComp] = d_p[iComp];
            vecScale(s[iComp], alpha);

            // d_x = s + d_x
            vecAdd(s[iComp], d_x[iComp], 1.0, 1.0);

            // y = g(x_{k+1}) - g(x_k)
            // first set y = -d_g (i.e. y = -g(x_k))
            y[iComp] = d_g[iComp];
            vecScale(y[iComp], -1.0);
          }


        // evaluate g(x_{k+1})
        iDFTSolverFunction.getForceVector(d_x, d_g, L);

        for (unsigned int iComp = 0; iComp < d_numComponents; ++iComp)
          {
            // add g(x_{k+1}) to y
            // to obtain y = g(x_{k+1}) - g(x_k)
            vecAdd(d_g[iComp], y[iComp], 1.0, 1.0);

            std::vector<double> curvature(1);
            iDFTSolverFunction.dotProduct(y[iComp], s[iComp], 1, curvature);
            const double rho = curvature[0];
            pcout << "Curvature condition (y^Ts) [" << iComp << "]: " << rho
                  << std::endl;

            //
            // add to history, if curvature is positive
            // TODO Use a small finite value instead
            if (rho > 0.0)
              {
                d_y[iComp].push_back(y[iComp]);
                d_s[iComp].push_back(s[iComp]);
                d_rho[iComp].push_back(1.0 / rho);
                d_k[iComp]++;
              }
          }

        for (unsigned int iComp = 0; iComp < d_numComponents; ++iComp)
          {
            if (d_k[iComp] > d_historySize)
              {
                d_k[iComp] = d_historySize;
                d_y[iComp].pop_front();
                d_s[iComp].pop_front();
                d_rho[iComp].pop_front();
              }
          }
      }
  }

  template <unsigned int              FEOrder,
            unsigned int              FEOrderElectro,
            dftfe::utils::MemorySpace memorySpace>
  std::vector<distributedCPUVec<double>>
  BFGSInverseDFTSolver<FEOrder, FEOrderElectro, memorySpace>::getSolution()
    const
  {
    return d_x;
  }

  template class BFGSInverseDFTSolver<2, 2, dftfe::utils::MemorySpace::HOST>;
  template class BFGSInverseDFTSolver<4, 4, dftfe::utils::MemorySpace::HOST>;
#ifdef DFTFE_WITH_DEVICE
  template class BFGSInverseDFTSolver<2, 2, dftfe::utils::MemorySpace::DEVICE>;
  template class BFGSInverseDFTSolver<4, 4, dftfe::utils::MemorySpace::DEVICE>;
#endif
} // namespace dftfe
