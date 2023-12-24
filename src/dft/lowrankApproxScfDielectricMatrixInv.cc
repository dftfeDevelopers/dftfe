// ---------------------------------------------------------------------
//
// Copyright (c) 2019-2020 The Regents of the University of Michigan and DFT-FE
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
#include <dft.h>
#include <linearAlgebraOperations.h>

namespace dftfe
{
  namespace internalLowrankJacInv
  {
    double
    relativeErrorEstimate(
      const std::deque<distributedCPUVec<double>> &fvcontainer,
      const distributedCPUVec<double> &            residualVec,
      const double                                 k0)
    {
      const unsigned int rank = fvcontainer.size();

      std::vector<double> mMat(rank * rank, 0.0);
      for (int j = 0; j < rank; j++)
        for (int i = 0; i < rank; i++)
          mMat[j * rank + i] = fvcontainer[i] * fvcontainer[j];

      dftfe::linearAlgebraOperations::inverse(&mMat[0], rank);

      distributedCPUVec<double> k0ResidualVec, approximationErrorVec;
      k0ResidualVec.reinit(residualVec);
      approximationErrorVec.reinit(residualVec);
      for (unsigned int idof = 0; idof < residualVec.local_size(); idof++)
        {
          k0ResidualVec.local_element(idof) =
            residualVec.local_element(idof) * k0;
          approximationErrorVec.local_element(idof) =
            k0ResidualVec.local_element(idof);
        }

      std::vector<double> innerProducts(rank, 0.0);
      for (unsigned int i = 0; i < rank; i++)
        innerProducts[i] = fvcontainer[i] * k0ResidualVec;


      for (unsigned int i = 0; i < rank; i++)
        {
          double temp = 0.0;
          for (unsigned int j = 0; j < rank; j++)
            temp += mMat[j * rank + i] * innerProducts[j];

          for (unsigned int idof = 0; idof < residualVec.local_size(); idof++)
            approximationErrorVec.local_element(idof) -=
              fvcontainer[i].local_element(idof) * temp;
        }

      return (approximationErrorVec.l2_norm() / k0ResidualVec.l2_norm());
    }

    void
    predictNextStepResidual(
      const std::deque<distributedCPUVec<double>> &fvcontainer,
      const distributedCPUVec<double> &            residualVec,
      distributedCPUVec<double> &                  predictedResidualVec,
      const double                                 k0,
      const double                                 alpha)
    {
      const unsigned int rank = fvcontainer.size();

      std::vector<double> mMat(rank * rank, 0.0);
      for (int j = 0; j < rank; j++)
        for (int i = 0; i < rank; i++)
          mMat[j * rank + i] = fvcontainer[i] * fvcontainer[j];

      dftfe::linearAlgebraOperations::inverse(&mMat[0], rank);

      distributedCPUVec<double> k0ResidualVec;
      k0ResidualVec.reinit(residualVec);
      for (unsigned int idof = 0; idof < residualVec.local_size(); idof++)
        {
          k0ResidualVec.local_element(idof) =
            residualVec.local_element(idof) * k0;
          predictedResidualVec.local_element(idof) =
            k0ResidualVec.local_element(idof);
        }

      std::vector<double> innerProducts(rank, 0.0);
      for (unsigned int i = 0; i < rank; i++)
        innerProducts[i] = fvcontainer[i] * k0ResidualVec;


      for (unsigned int i = 0; i < rank; i++)
        {
          double temp = 0.0;
          for (unsigned int j = 0; j < rank; j++)
            temp += mMat[j * rank + i] * innerProducts[j];

          for (unsigned int idof = 0; idof < residualVec.local_size(); idof++)
            predictedResidualVec.local_element(idof) -=
              alpha * fvcontainer[i].local_element(idof) * temp;
        }
    }


    void
    lowrankKernelApply(const std::deque<distributedCPUVec<double>> &fvcontainer,
                       const std::deque<distributedCPUVec<double>> &vcontainer,
                       const distributedCPUVec<double> &            x,
                       const double                                 k0,
                       distributedCPUVec<double> &                  y)
    {
      const unsigned int rank = fvcontainer.size();

      std::vector<double> mMat(rank * rank, 0.0);
      for (int j = 0; j < rank; j++)
        for (int i = 0; i < rank; i++)
          mMat[j * rank + i] = fvcontainer[i] * fvcontainer[j];

      dftfe::linearAlgebraOperations::inverse(&mMat[0], rank);

      for (unsigned int idof = 0; idof < x.local_size(); idof++)
        y.local_element(idof) = x.local_element(idof) * k0;

      std::vector<double> innerProducts(rank, 0.0);
      for (unsigned int i = 0; i < rank; i++)
        innerProducts[i] = fvcontainer[i] * y;

      y = 0;

      for (unsigned int i = 0; i < rank; i++)
        {
          double temp = 0.0;
          // FIXME: exploit symmetry of mMat
          for (unsigned int j = 0; j < rank; j++)
            temp += mMat[j * rank + i] * innerProducts[j];

          for (unsigned int idof = 0; idof < y.local_size(); idof++)
            y.local_element(idof) += vcontainer[i].local_element(idof) * temp;
        }
    }


    void
    lowrankJacInvApply(const std::deque<distributedCPUVec<double>> &fvcontainer,
                       const std::deque<distributedCPUVec<double>> &vcontainer,
                       const distributedCPUVec<double> &            x,
                       distributedCPUVec<double> &                  y)
    {
      const unsigned int rank = fvcontainer.size();

      std::vector<double> mMat(rank * rank, 0.0);
      for (int j = 0; j < rank; j++)
        for (int i = 0; i < rank; i++)
          mMat[j * rank + i] = fvcontainer[i] * fvcontainer[j];

      dftfe::linearAlgebraOperations::inverse(&mMat[0], rank);

      std::vector<double> innerProducts(rank, 0.0);
      for (unsigned int i = 0; i < rank; i++)
        innerProducts[i] = fvcontainer[i] * x;

      y = 0;

      for (unsigned int i = 0; i < rank; i++)
        {
          double temp = 0.0;
          for (unsigned int j = 0; j < rank; j++)
            temp += mMat[j * rank + i] * innerProducts[j];

          for (unsigned int idof = 0; idof < y.local_size(); idof++)
            y.local_element(idof) += vcontainer[i].local_element(idof) * temp;
        }
    }


    void
    lowrankJacApply(const std::deque<distributedCPUVec<double>> &fvcontainer,
                    const std::deque<distributedCPUVec<double>> &vcontainer,
                    const distributedCPUVec<double> &            x,
                    distributedCPUVec<double> &                  y)
    {
      const unsigned int rank = fvcontainer.size();


      std::vector<double> innerProducts(rank, 0.0);
      for (unsigned int i = 0; i < rank; i++)
        innerProducts[i] = vcontainer[i] * x;

      y = 0;
      for (unsigned int i = 0; i < rank; i++)
        for (unsigned int idof = 0; idof < y.local_size(); idof++)
          y.local_element(idof) +=
            fvcontainer[i].local_element(idof) * innerProducts[i];
    }



    double
    estimateLargestEigenvalueMagJacLowrankPower(
      const std::deque<distributedCPUVec<double>> &lowrankFvcontainer,
      const std::deque<distributedCPUVec<double>> &lowrankVcontainer,
      const distributedCPUVec<double> &            x,
      const dealii::AffineConstraints<double> &    constraintsRhoNodal)
    {
      const double tol = 1.0e-6;

      double lambdaOld     = 0.0;
      double lambdaNew     = 0.0;
      double diffLambdaAbs = 1e+6;
      //
      // generate random vector v
      //
      distributedCPUVec<double> vVector, fVector;
      vVector.reinit(x);
      fVector.reinit(x);

      vVector = 0.0, fVector = 0.0;
      // std::srand(this_mpi_process);
      const unsigned int local_size = vVector.local_size();

      // for (unsigned int i = 0; i < local_size; i++)
      //  vVector.local_element(i) = x.local_element(i);

      for (unsigned int i = 0; i < local_size; i++)
        vVector.local_element(i) = ((double)std::rand()) / ((double)RAND_MAX);

      constraintsRhoNodal.set_zero(vVector);

      vVector.update_ghost_values();

      //
      // evaluate l2 norm
      //
      vVector /= vVector.l2_norm();
      vVector.update_ghost_values();
      int iter = 0;
      while (diffLambdaAbs > tol)
        {
          fVector = 0;
          lowrankJacApply(lowrankFvcontainer,
                          lowrankVcontainer,
                          vVector,
                          fVector);
          lambdaOld = lambdaNew;
          lambdaNew = (vVector * fVector) / (vVector * vVector);

          vVector = fVector;
          vVector /= vVector.l2_norm();
          vVector.update_ghost_values();
          diffLambdaAbs = std::abs(lambdaNew - lambdaOld);
          iter++;
        }

      // std::cout << " Power iterations iter: "<< iter
      //            << std::endl;

      return std::abs(lambdaNew);
    }

    double
    estimateLargestEigenvalueMagJacInvLowrankPower(
      const std::deque<distributedCPUVec<double>> &lowrankFvcontainer,
      const std::deque<distributedCPUVec<double>> &lowrankVcontainer,
      const distributedCPUVec<double> &            x,
      const dealii::AffineConstraints<double> &    constraintsRhoNodal)
    {
      const double tol = 1.0e-6;

      double lambdaOld     = 0.0;
      double lambdaNew     = 0.0;
      double diffLambdaAbs = 1e+6;
      //
      // generate random vector v
      //
      distributedCPUVec<double> vVector, fVector;
      vVector.reinit(x);
      fVector.reinit(x);

      vVector = 0.0, fVector = 0.0;
      // std::srand(this_mpi_process);
      const unsigned int local_size = vVector.local_size();

      // for (unsigned int i = 0; i < local_size; i++)
      //   vVector.local_element(i) = x.local_element(i);

      for (unsigned int i = 0; i < local_size; i++)
        vVector.local_element(i) = ((double)std::rand()) / ((double)RAND_MAX);

      constraintsRhoNodal.set_zero(vVector);

      vVector.update_ghost_values();

      //
      // evaluate l2 norm
      //
      vVector /= vVector.l2_norm();
      vVector.update_ghost_values();

      int iter = 0;
      while (diffLambdaAbs > tol)
        {
          fVector = 0;
          lowrankJacInvApply(lowrankFvcontainer,
                             lowrankVcontainer,
                             vVector,
                             fVector);
          lambdaOld = lambdaNew;
          lambdaNew = (vVector * fVector);

          vVector = fVector;
          vVector /= vVector.l2_norm();
          vVector.update_ghost_values();
          diffLambdaAbs = std::abs(lambdaNew - lambdaOld);
          iter++;
        }

      // std::cout << " Power iterations iter: "<< iter
      //            << std::endl;

      return std::abs(lambdaNew);
    }
  } // namespace internalLowrankJacInv

  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::lowrankApproxScfDielectricMatrixInv(
    const unsigned int scfIter)
  {
    int this_process;
    MPI_Comm_rank(d_mpiCommParent, &this_process);
    MPI_Barrier(d_mpiCommParent);
    double total_time = MPI_Wtime();

    double normValue = 0.0;

    distributedCPUVec<double> residualRho;
    residualRho.reinit(d_densityInNodalValues[0]);
    residualRho = 0.0;


    // compute residual = rhoOut - rhoIn
    residualRho.add(1.0,
                    d_densityOutNodalValues[0],
                    -1.0,
                    d_densityInNodalValues[0]);

    residualRho.update_ghost_values();

    // compute l2 norm of the field residual
    normValue = rhofieldl2Norm(d_matrixFreeDataPRefined,
                               residualRho,
                               d_densityDofHandlerIndexElectro,
                               d_densityQuadratureIdElectro);

    const double predictedToActualResidualRatio =
      d_residualPredicted.l2_norm() / residualRho.l2_norm();
    if (scfIter > 1 && d_dftParamsPtr->verbosity >= 4)
      {
        pcout << "Actual residual norm value: " << normValue << std::endl;
        pcout << "Predicted residual norm value: " << d_residualNormPredicted
              << std::endl;
        pcout << "Ratio: " << predictedToActualResidualRatio << std::endl;
      }

    const double k0 = 1.0;


    distributedCPUVec<double> kernelAction;
    distributedCPUVec<double> compvec;
    distributedCPUVec<double> checkvec;
    distributedCPUVec<double> dummy;
    kernelAction.reinit(residualRho);
    checkvec.reinit(residualRho);
    compvec.reinit(residualRho);

    d_residualPredicted.reinit(residualRho);
    d_residualPredicted = 0;

    double             charge;
    const unsigned int local_size = residualRho.local_size();


    double relativeApproxError = 1.0e+6;
    if (d_rankCurrentLRD >= 1 &&
        d_dftParamsPtr->methodSubTypeLRD == "ACCUMULATED_ADAPTIVE")
      {
        relativeApproxError =
          internalLowrankJacInv::relativeErrorEstimate(d_fvcontainerVals,
                                                       residualRho,
                                                       k0);
        pcout << "Starting relative approx error accumulated: "
              << relativeApproxError << std::endl;
      }

    const double linearityRegimeFac      = d_dftParamsPtr->betaTol;
    int          rankAddedInThisScf      = 0;
    int          rankAddedBeforeClearing = 0;
    if (!(relativeApproxError < d_dftParamsPtr->adaptiveRankRelTolLRD &&
          predictedToActualResidualRatio > (1 - linearityRegimeFac) &&
          predictedToActualResidualRatio < (1 + linearityRegimeFac)))
      {
        if ((normValue < 1.0) && d_rankCurrentLRD >= 1 &&
            d_dftParamsPtr->methodSubTypeLRD == "ACCUMULATED_ADAPTIVE")
          {
            if (!d_tolReached)
              {
                if (d_dftParamsPtr->verbosity >= 4)
                  pcout
                    << " Clearing accumulation as tolerance not reached in previous step "
                    << std::endl;
                d_vcontainerVals.clear();
                d_fvcontainerVals.clear();
                d_rankCurrentLRD = 0;
              }
            else if (predictedToActualResidualRatio <
                       (1 - linearityRegimeFac) ||
                     predictedToActualResidualRatio > (1 + linearityRegimeFac))
              {
                if (d_dftParamsPtr->verbosity >= 4)
                  pcout
                    << " Clearing accumulation as outside local linear regime "
                    << ", linearity indicator: "
                    << predictedToActualResidualRatio << std::endl;
                d_vcontainerVals.clear();
                d_fvcontainerVals.clear();
                d_rankCurrentLRD = 0;
              }
          }
        else
          {
            if (d_dftParamsPtr->verbosity >= 4)
              pcout
                << " Clearing accumulation as residual is not sufficiently reduced yet "
                << std::endl;
            d_vcontainerVals.clear();
            d_fvcontainerVals.clear();
            d_rankCurrentLRD = 0;
          }

        int maxRankThisScf =
          (scfIter < 2) ? 5 : (d_rankCurrentLRD >= 1 ? 5 : 20);
        d_tolReached = false;
        while (((rankAddedInThisScf < maxRankThisScf)) ||
               ((normValue < d_dftParamsPtr->selfConsistentSolverTolerance) &&
                (d_dftParamsPtr->estimateJacCondNoFinalSCFIter)))
          {
            if (rankAddedInThisScf == 0)
              {
                d_vcontainerVals.push_back(residualRho);
                d_vcontainerVals[d_rankCurrentLRD] *= k0;
              }
            else
              d_vcontainerVals.push_back(
                d_fvcontainerVals[d_rankCurrentLRD - 1]);


            d_vcontainerVals[d_rankCurrentLRD] *=
              1.0 / d_vcontainerVals[d_rankCurrentLRD].l2_norm();


            std::deque<double> components;
            for (int jrank = 0; jrank < d_rankCurrentLRD; jrank++)
              {
                components.push_back(d_vcontainerVals[d_rankCurrentLRD] *
                                     d_vcontainerVals[jrank]);
              }


            if (d_dftParamsPtr->methodSubTypeLRD == "ACCUMULATED_ADAPTIVE" &&
                (d_rankCurrentLRD - rankAddedInThisScf) > 0)
              {
                compvec = 0;
                for (int jrank = 0;
                     jrank < (d_rankCurrentLRD - rankAddedInThisScf);
                     jrank++)
                  {
                    compvec.add(components[jrank], d_vcontainerVals[jrank]);
                  }

                checkvec = d_vcontainerVals[d_rankCurrentLRD];
                checkvec -= compvec;

                // check orthogonal complement against previous scf direction
                // functions to decide to clear or not const double
                // checkTol=0.2;
                const double normCheck = checkvec.l2_norm();
                if (normCheck < 0.01)
                  {
                    d_vcontainerVals.clear();
                    d_fvcontainerVals.clear();
                    components.clear();
                    rankAddedBeforeClearing = rankAddedInThisScf;
                    d_rankCurrentLRD        = 0;
                    rankAddedInThisScf      = 0;
                    maxRankThisScf          = 20;
                    if (d_dftParamsPtr->verbosity >= 4)
                      pcout
                        << " Clearing accumulation as current scf direction function well represented in previous scf Krylov subspace, l2norm of Orthogonal component: "
                        << normCheck << std::endl;
                    continue;
                  }
                else
                  {
                    if (d_dftParamsPtr->verbosity >= 4)
                      pcout
                        << "Orthogonal component to previous direction functions l2 norm: "
                        << normCheck << std::endl;
                  }
              }
            ////


            compvec = 0;
            for (int jrank = 0; jrank < d_rankCurrentLRD; jrank++)
              {
                compvec.add(components[jrank], d_vcontainerVals[jrank]);
              }

            d_vcontainerVals[d_rankCurrentLRD] -= compvec;

            d_vcontainerVals[d_rankCurrentLRD] *=
              1.0 / d_vcontainerVals[d_rankCurrentLRD].l2_norm();

            if (d_dftParamsPtr->verbosity >= 4)
              pcout << " Vector norm of v:  "
                    << d_vcontainerVals[d_rankCurrentLRD].l2_norm()
                    << ", for rank: " << d_rankCurrentLRD + 1 << std::endl;

            d_fvcontainerVals.push_back(residualRho);
            d_fvcontainerVals[d_rankCurrentLRD] = 0;

            d_vcontainerVals[d_rankCurrentLRD].update_ghost_values();
            charge = totalCharge(d_matrixFreeDataPRefined,
                                 d_vcontainerVals[d_rankCurrentLRD]);


            if (d_dftParamsPtr->verbosity >= 4)
              pcout << "Integral v before scaling:  " << charge << std::endl;

            d_vcontainerVals[d_rankCurrentLRD].add(-charge / d_domainVolume);

            d_vcontainerVals[d_rankCurrentLRD].update_ghost_values();
            charge = totalCharge(d_matrixFreeDataPRefined,
                                 d_vcontainerVals[d_rankCurrentLRD]);

            if (d_dftParamsPtr->verbosity >= 4)
              pcout << "Integral v after scaling:  " << charge << std::endl;

            computeOutputDensityDirectionalDerivative(
              d_vcontainerVals[d_rankCurrentLRD],
              dummy,
              dummy,
              d_fvcontainerVals[d_rankCurrentLRD],
              dummy,
              dummy);

            d_fvcontainerVals[d_rankCurrentLRD].update_ghost_values();
            charge = totalCharge(d_matrixFreeDataPRefined,
                                 d_fvcontainerVals[d_rankCurrentLRD]);


            if (d_dftParamsPtr->verbosity >= 4)
              pcout << "Integral fv before scaling:  " << charge << std::endl;

            d_fvcontainerVals[d_rankCurrentLRD].add(-charge / d_domainVolume);

            d_fvcontainerVals[d_rankCurrentLRD].update_ghost_values();
            charge = totalCharge(d_matrixFreeDataPRefined,
                                 d_fvcontainerVals[d_rankCurrentLRD]);
            if (d_dftParamsPtr->verbosity >= 4)
              pcout << "Integral fv after scaling:  " << charge << std::endl;

            if (d_dftParamsPtr->verbosity >= 4)
              pcout
                << " Vector norm of response (delta rho_min[n+delta_lambda*v1]/ delta_lambda):  "
                << d_fvcontainerVals[d_rankCurrentLRD].l2_norm()
                << " for kernel rank: " << d_rankCurrentLRD + 1 << std::endl;

            d_fvcontainerVals[d_rankCurrentLRD] -=
              d_vcontainerVals[d_rankCurrentLRD];
            d_fvcontainerVals[d_rankCurrentLRD] *= k0;
            d_rankCurrentLRD++;
            rankAddedInThisScf++;

            if (d_dftParamsPtr->methodSubTypeLRD == "ADAPTIVE" ||
                d_dftParamsPtr->methodSubTypeLRD == "ACCUMULATED_ADAPTIVE")
              {
                relativeApproxError =
                  internalLowrankJacInv::relativeErrorEstimate(
                    d_fvcontainerVals, residualRho, k0);

                if (d_dftParamsPtr->verbosity >= 4)
                  pcout << " Relative approx error:  " << relativeApproxError
                        << " for kernel rank: " << d_rankCurrentLRD
                        << std::endl;

                if ((normValue <
                     d_dftParamsPtr->selfConsistentSolverTolerance) &&
                    (d_dftParamsPtr->estimateJacCondNoFinalSCFIter))
                  {
                    if (relativeApproxError < 1.0e-5)
                      {
                        break;
                      }
                  }
                else
                  {
                    if (relativeApproxError <
                        d_dftParamsPtr->adaptiveRankRelTolLRD)
                      {
                        d_tolReached = true;
                        break;
                      }
                  }
              }
          }
      }


    if (d_dftParamsPtr->verbosity >= 4)
      pcout << " Net accumulated kernel rank:  " << d_rankCurrentLRD
            << " Accumulated or used in this scf: "
            << (rankAddedInThisScf + rankAddedBeforeClearing) << std::endl;

    internalLowrankJacInv::lowrankKernelApply(
      d_fvcontainerVals, d_vcontainerVals, residualRho, k0, kernelAction);

    if (normValue < d_dftParamsPtr->selfConsistentSolverTolerance &&
        d_dftParamsPtr->estimateJacCondNoFinalSCFIter)
      {
        const double maxAbsEigenValue =
          internalLowrankJacInv::estimateLargestEigenvalueMagJacLowrankPower(
            d_fvcontainerVals,
            d_vcontainerVals,
            residualRho,
            d_constraintsRhoNodal);
        const double minAbsEigenValue =
          1.0 /
          internalLowrankJacInv::estimateLargestEigenvalueMagJacInvLowrankPower(
            d_fvcontainerVals,
            d_vcontainerVals,
            residualRho,
            d_constraintsRhoNodal);
        pcout << " Maximum eigenvalue of low rank approx of Jacobian: "
              << maxAbsEigenValue << std::endl;
        pcout << " Minimum non-zero eigenvalue of low rank approx of Jacobian: "
              << minAbsEigenValue << std::endl;
        pcout << " Condition no of low rank approx of Jacobian: "
              << maxAbsEigenValue / minAbsEigenValue << std::endl;
      }

    // pcout << " Preconditioned simple mixing step " << std::endl;
    // preconditioned simple mixing step
    // Note for const=-1.0, it should be same as Newton step
    // For second scf iteration step (scIter==1), the rhoIn is from atomic
    // densities which casues robustness issues when used with a
    // higher mixingParameter value.
    // Suggested to use 0.1 for initial steps
    // as well as when normValue is greater than 2.0
    double const2 =
      (normValue > d_dftParamsPtr->startingNormLRDLargeDamping || scfIter < 2) ?
        -0.1 :
        -d_dftParamsPtr->mixingParameter;

    if (d_dftParamsPtr->verbosity >= 4)
      pcout << " Preconditioned mixing step, mixing constant: " << const2
            << std::endl;

    internalLowrankJacInv::predictNextStepResidual(
      d_fvcontainerVals, residualRho, d_residualPredicted, k0, -const2);

    d_residualPredicted.update_ghost_values();

    // compute l2 norm of the field residual
    d_residualNormPredicted = rhofieldl2Norm(d_matrixFreeDataPRefined,
                                             d_residualPredicted,
                                             d_densityDofHandlerIndexElectro,
                                             d_densityQuadratureIdElectro);

    d_densityInNodalValues[0].add(const2, kernelAction);

    d_densityInNodalValues[0].update_ghost_values();

    // interpolate nodal data to quadrature data
    interpolateDensityNodalDataToQuadratureDataGeneral(
      basisOperationsPtrElectroHost,
      d_densityDofHandlerIndexElectro,
      d_densityQuadratureIdElectro,
      d_densityInNodalValues[0],
      d_densityInQuadValues[0],
      d_gradDensityInQuadValues[0],
      d_gradDensityInQuadValues[0],
      d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA);

    MPI_Barrier(d_mpiCommParent);
    total_time = MPI_Wtime() - total_time;

    if (this_process == 0 && d_dftParamsPtr->verbosity >= 2)
      std::cout << "Time for low rank jac inv: " << total_time << std::endl;

    return normValue;
  }
#include "dft.inst.cc"
} // namespace dftfe
