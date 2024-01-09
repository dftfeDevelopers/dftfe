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
    frobeniusNormSpin(const distributedCPUVec<double> &vecSpin0,
                      const distributedCPUVec<double> &vecSpin1)
    {
      return std::sqrt(vecSpin0 * vecSpin0 + vecSpin1 * vecSpin1);
    }

    double
    relativeErrorEstimateSpin(
      const std::deque<distributedCPUVec<double>> &fvcontainerSpin0,
      const std::deque<distributedCPUVec<double>> &fvcontainerSpin1,
      const distributedCPUVec<double> &            residualVecSpin0,
      const distributedCPUVec<double> &            residualVecSpin1,
      const double                                 k0)
    {
      const unsigned int rank = fvcontainerSpin0.size();

      std::vector<double> mMat(rank * rank, 0.0);
      for (int j = 0; j < rank; j++)
        for (int i = 0; i < rank; i++)
          mMat[j * rank + i] = fvcontainerSpin0[i] * fvcontainerSpin0[j] +
                               fvcontainerSpin1[i] * fvcontainerSpin1[j];

      dftfe::linearAlgebraOperations::inverse(&mMat[0], rank);

      distributedCPUVec<double> k0ResidualVecSpin0, k0ResidualVecSpin1;
      distributedCPUVec<double> approximationErrorVecSpin0,
        approximationErrorVecSpin1;
      k0ResidualVecSpin0.reinit(residualVecSpin0);
      k0ResidualVecSpin1.reinit(residualVecSpin0);
      approximationErrorVecSpin0.reinit(residualVecSpin0);
      approximationErrorVecSpin1.reinit(residualVecSpin1);
      for (unsigned int idof = 0; idof < residualVecSpin0.local_size(); idof++)
        {
          k0ResidualVecSpin0.local_element(idof) =
            residualVecSpin0.local_element(idof) * k0;
          approximationErrorVecSpin0.local_element(idof) =
            k0ResidualVecSpin0.local_element(idof);

          k0ResidualVecSpin1.local_element(idof) =
            residualVecSpin1.local_element(idof) * k0;
          approximationErrorVecSpin1.local_element(idof) =
            k0ResidualVecSpin1.local_element(idof);
        }

      std::vector<double> innerProducts(rank, 0.0);
      for (unsigned int i = 0; i < rank; i++)
        innerProducts[i] = fvcontainerSpin0[i] * k0ResidualVecSpin0 +
                           fvcontainerSpin1[i] * k0ResidualVecSpin1;


      for (unsigned int i = 0; i < rank; i++)
        {
          double temp = 0.0;
          for (unsigned int j = 0; j < rank; j++)
            temp += mMat[j * rank + i] * innerProducts[j];

          for (unsigned int idof = 0; idof < residualVecSpin0.local_size();
               idof++)
            {
              approximationErrorVecSpin0.local_element(idof) -=
                fvcontainerSpin0[i].local_element(idof) * temp;
              approximationErrorVecSpin1.local_element(idof) -=
                fvcontainerSpin1[i].local_element(idof) * temp;
            }
        }

      const double frobeniusNormApproximationErrorMat =
        std::sqrt(approximationErrorVecSpin0 * approximationErrorVecSpin0 +
                  approximationErrorVecSpin1 * approximationErrorVecSpin1);
      const double frobeniusNormk0ResidualMat =
        std::sqrt(k0ResidualVecSpin0 * k0ResidualVecSpin0 +
                  k0ResidualVecSpin1 * k0ResidualVecSpin1);

      return (frobeniusNormApproximationErrorMat / frobeniusNormk0ResidualMat);
    }

    void
    lowrankKernelApplySpin(
      const std::deque<distributedCPUVec<double>> &fvcontainerSpin0,
      const std::deque<distributedCPUVec<double>> &fvcontainerSpin1,
      const std::deque<distributedCPUVec<double>> &vcontainerSpin0,
      const std::deque<distributedCPUVec<double>> &vcontainerSpin1,
      const distributedCPUVec<double> &            xSpin0,
      const distributedCPUVec<double> &            xSpin1,
      const double                                 k0,
      distributedCPUVec<double> &                  ySpin0,
      distributedCPUVec<double> &                  ySpin1)
    {
      const unsigned int rank = fvcontainerSpin0.size();

      std::vector<double> mMat(rank * rank, 0.0);
      for (int j = 0; j < rank; j++)
        for (int i = 0; i < rank; i++)
          mMat[j * rank + i] = fvcontainerSpin0[i] * fvcontainerSpin0[j] +
                               fvcontainerSpin1[i] * fvcontainerSpin1[j];

      dftfe::linearAlgebraOperations::inverse(&mMat[0], rank);

      for (unsigned int idof = 0; idof < xSpin0.local_size(); idof++)
        {
          ySpin0.local_element(idof) = xSpin0.local_element(idof) * k0;
          ySpin1.local_element(idof) = xSpin1.local_element(idof) * k0;
        }

      std::vector<double> innerProducts(rank, 0.0);
      for (unsigned int i = 0; i < rank; i++)
        {
          innerProducts[i] =
            fvcontainerSpin0[i] * ySpin0 + fvcontainerSpin1[i] * ySpin1;
        }

      ySpin0 = 0;
      ySpin1 = 0;

      for (unsigned int i = 0; i < rank; i++)
        {
          double temp = 0.0;
          for (unsigned int j = 0; j < rank; j++)
            temp += mMat[j * rank + i] * innerProducts[j];

          for (unsigned int idof = 0; idof < ySpin0.local_size(); idof++)
            {
              ySpin0.local_element(idof) +=
                vcontainerSpin0[i].local_element(idof) * temp;
              ySpin1.local_element(idof) +=
                vcontainerSpin1[i].local_element(idof) * temp;
            }
        }
    }

    void
    lowrankJacInvApplySpin(
      const std::deque<distributedCPUVec<double>> &fvcontainerSpin0,
      const std::deque<distributedCPUVec<double>> &fvcontainerSpin1,
      const std::deque<distributedCPUVec<double>> &vcontainerSpin0,
      const std::deque<distributedCPUVec<double>> &vcontainerSpin1,
      const distributedCPUVec<double> &            xSpin0,
      const distributedCPUVec<double> &            xSpin1,
      distributedCPUVec<double> &                  ySpin0,
      distributedCPUVec<double> &                  ySpin1)
    {
      const unsigned int rank = fvcontainerSpin0.size();

      std::vector<double> mMat(rank * rank, 0.0);
      for (int j = 0; j < rank; j++)
        for (int i = 0; i < rank; i++)
          mMat[j * rank + i] = fvcontainerSpin0[i] * fvcontainerSpin0[j] +
                               fvcontainerSpin1[i] * fvcontainerSpin1[j];

      dftfe::linearAlgebraOperations::inverse(&mMat[0], rank);

      for (unsigned int idof = 0; idof < xSpin0.local_size(); idof++)
        {
          ySpin0.local_element(idof) = xSpin0.local_element(idof);
          ySpin1.local_element(idof) = xSpin1.local_element(idof);
        }

      std::vector<double> innerProducts(rank, 0.0);
      for (unsigned int i = 0; i < rank; i++)
        {
          innerProducts[i] =
            fvcontainerSpin0[i] * ySpin0 + fvcontainerSpin1[i] * ySpin1;
        }

      ySpin0 = 0;
      ySpin1 = 0;

      for (unsigned int i = 0; i < rank; i++)
        {
          double temp = 0.0;
          for (unsigned int j = 0; j < rank; j++)
            temp += mMat[j * rank + i] * innerProducts[j];

          for (unsigned int idof = 0; idof < ySpin0.local_size(); idof++)
            {
              ySpin0.local_element(idof) +=
                vcontainerSpin0[i].local_element(idof) * temp;
              ySpin1.local_element(idof) +=
                vcontainerSpin1[i].local_element(idof) * temp;
            }
        }
    }


    void
    lowrankJacApplySpin(
      const std::deque<distributedCPUVec<double>> &fvSpin0container,
      const std::deque<distributedCPUVec<double>> &fvSpin1container,
      const std::deque<distributedCPUVec<double>> &vSpin0container,
      const std::deque<distributedCPUVec<double>> &vSpin1container,
      const distributedCPUVec<double> &            xSpin0,
      const distributedCPUVec<double> &            xSpin1,
      distributedCPUVec<double> &                  ySpin0,
      distributedCPUVec<double> &                  ySpin1)
    {
      const unsigned int rank = fvSpin0container.size();


      std::vector<double> innerProducts(rank, 0.0);
      for (unsigned int i = 0; i < rank; i++)
        innerProducts[i] =
          vSpin0container[i] * xSpin0 + vSpin1container[i] * xSpin1;

      ySpin0 = 0;
      ySpin1 = 0;
      for (unsigned int i = 0; i < rank; i++)
        for (unsigned int idof = 0; idof < ySpin0.local_size(); idof++)
          {
            ySpin0.local_element(idof) +=
              fvSpin0container[i].local_element(idof) * innerProducts[i];
            ySpin1.local_element(idof) +=
              fvSpin1container[i].local_element(idof) * innerProducts[i];
          }
    }


    double
    estimateLargestEigenvalueMagJacLowrankPowerSpin(
      const std::deque<distributedCPUVec<double>> &lowrankFvSpin0container,
      const std::deque<distributedCPUVec<double>> &lowrankFvSpin1container,
      const std::deque<distributedCPUVec<double>> &lowrankVSpin0container,
      const std::deque<distributedCPUVec<double>> &lowrankVSpin1container,
      const distributedCPUVec<double> &            xSpin0,
      const distributedCPUVec<double> &            xSpin1,
      const dealii::AffineConstraints<double> &    constraintsRhoNodal)
    {
      const double tol = 1.0e-6;

      double lambdaOld     = 0.0;
      double lambdaNew     = 0.0;
      double diffLambdaAbs = 1e+6;
      //
      // generate random vector v
      //
      distributedCPUVec<double> vVectorSpin0, vVectorSpin1, fVectorSpin0,
        fVectorSpin1;
      vVectorSpin0.reinit(xSpin0);
      vVectorSpin1.reinit(xSpin1);
      fVectorSpin0.reinit(xSpin0);
      fVectorSpin1.reinit(xSpin1);

      vVectorSpin0 = 0.0, fVectorSpin0 = 0.0;
      vVectorSpin1 = 0.0, fVectorSpin1 = 0.0;
      // std::srand(this_mpi_process);
      const unsigned int local_size = vVectorSpin0.local_size();

      // for (unsigned int i = 0; i < local_size; i++)
      //  vVector.local_element(i) = x.local_element(i);

      for (unsigned int i = 0; i < local_size; i++)
        {
          vVectorSpin0.local_element(i) =
            ((double)std::rand()) / ((double)RAND_MAX);
          vVectorSpin1.local_element(i) =
            ((double)std::rand()) / ((double)RAND_MAX);
        }

      constraintsRhoNodal.set_zero(vVectorSpin0);

      vVectorSpin0.update_ghost_values();

      constraintsRhoNodal.set_zero(vVectorSpin1);

      vVectorSpin1.update_ghost_values();

      //
      // evaluate l2 norm
      //
      vVectorSpin0 /=
        std::sqrt(vVectorSpin0 * vVectorSpin0 + vVectorSpin1 * vVectorSpin1);
      vVectorSpin1 /=
        std::sqrt(vVectorSpin0 * vVectorSpin0 + vVectorSpin1 * vVectorSpin1);
      vVectorSpin0.update_ghost_values();
      vVectorSpin1.update_ghost_values();
      int iter = 0;
      while (diffLambdaAbs > tol)
        {
          fVectorSpin0 = 0;
          fVectorSpin1 = 0;
          lowrankJacApplySpin(lowrankFvSpin0container,
                              lowrankFvSpin1container,
                              lowrankVSpin0container,
                              lowrankVSpin1container,
                              vVectorSpin0,
                              vVectorSpin1,
                              fVectorSpin0,
                              fVectorSpin1);
          lambdaOld = lambdaNew;
          lambdaNew =
            (vVectorSpin0 * fVectorSpin0 + vVectorSpin1 * fVectorSpin1) /
            (vVectorSpin0 * vVectorSpin0 + vVectorSpin1 * vVectorSpin1);

          vVectorSpin0 = fVectorSpin0;
          vVectorSpin1 = fVectorSpin1;
          vVectorSpin0 /= std::sqrt(vVectorSpin0 * vVectorSpin0 +
                                    vVectorSpin1 * vVectorSpin1);
          vVectorSpin1 /= std::sqrt(vVectorSpin0 * vVectorSpin0 +
                                    vVectorSpin1 * vVectorSpin1);
          vVectorSpin0.update_ghost_values();
          vVectorSpin1.update_ghost_values();

          diffLambdaAbs = std::abs(lambdaNew - lambdaOld);
          iter++;
        }

      // std::cout << " Power iterations iter: "<< iter
      //            << std::endl;

      return std::abs(lambdaNew);
    }

    double
    estimateLargestEigenvalueMagJacInvLowrankPowerSpin(
      const std::deque<distributedCPUVec<double>> &lowrankFvSpin0container,
      const std::deque<distributedCPUVec<double>> &lowrankFvSpin1container,
      const std::deque<distributedCPUVec<double>> &lowrankVSpin0container,
      const std::deque<distributedCPUVec<double>> &lowrankVSpin1container,
      const distributedCPUVec<double> &            xSpin0,
      const distributedCPUVec<double> &            xSpin1,
      const dealii::AffineConstraints<double> &    constraintsRhoNodal)
    {
      const double tol = 1.0e-6;

      double lambdaOld     = 0.0;
      double lambdaNew     = 0.0;
      double diffLambdaAbs = 1e+6;
      //
      // generate random vector v
      //
      distributedCPUVec<double> vVectorSpin0, vVectorSpin1, fVectorSpin0,
        fVectorSpin1;
      vVectorSpin0.reinit(xSpin0);
      vVectorSpin1.reinit(xSpin1);
      fVectorSpin0.reinit(xSpin0);
      fVectorSpin1.reinit(xSpin1);

      vVectorSpin0 = 0.0, fVectorSpin0 = 0.0;
      vVectorSpin1 = 0.0, fVectorSpin1 = 0.0;
      // std::srand(this_mpi_process);
      const unsigned int local_size = vVectorSpin0.local_size();

      // for (unsigned int i = 0; i < local_size; i++)
      //  vVector.local_element(i) = x.local_element(i);

      for (unsigned int i = 0; i < local_size; i++)
        {
          vVectorSpin0.local_element(i) =
            ((double)std::rand()) / ((double)RAND_MAX);
          vVectorSpin1.local_element(i) =
            ((double)std::rand()) / ((double)RAND_MAX);
        }

      constraintsRhoNodal.set_zero(vVectorSpin0);

      vVectorSpin0.update_ghost_values();

      constraintsRhoNodal.set_zero(vVectorSpin1);

      vVectorSpin1.update_ghost_values();

      //
      // evaluate l2 norm
      //
      vVectorSpin0 /=
        std::sqrt(vVectorSpin0 * vVectorSpin0 + vVectorSpin1 * vVectorSpin1);
      vVectorSpin1 /=
        std::sqrt(vVectorSpin0 * vVectorSpin0 + vVectorSpin1 * vVectorSpin1);
      vVectorSpin0.update_ghost_values();
      vVectorSpin1.update_ghost_values();
      int iter = 0;
      while (diffLambdaAbs > tol)
        {
          fVectorSpin0 = 0;
          fVectorSpin1 = 0;
          lowrankJacInvApplySpin(lowrankFvSpin0container,
                                 lowrankFvSpin1container,
                                 lowrankVSpin0container,
                                 lowrankVSpin1container,
                                 vVectorSpin0,
                                 vVectorSpin1,
                                 fVectorSpin0,
                                 fVectorSpin1);
          lambdaOld = lambdaNew;
          lambdaNew =
            (vVectorSpin0 * fVectorSpin0 + vVectorSpin1 * fVectorSpin1) /
            (vVectorSpin0 * vVectorSpin0 + vVectorSpin1 * vVectorSpin1);

          vVectorSpin0 = fVectorSpin0;
          vVectorSpin1 = fVectorSpin1;
          vVectorSpin0 /= std::sqrt(vVectorSpin0 * vVectorSpin0 +
                                    vVectorSpin1 * vVectorSpin1);
          vVectorSpin1 /= std::sqrt(vVectorSpin0 * vVectorSpin0 +
                                    vVectorSpin1 * vVectorSpin1);
          vVectorSpin0.update_ghost_values();
          vVectorSpin1.update_ghost_values();

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
  dftClass<FEOrder, FEOrderElectro>::
    lowrankApproxScfDielectricMatrixInvSpinPolarized(const unsigned int scfIter)
  {
    int this_process;
    MPI_Comm_rank(d_mpiCommParent, &this_process);
    MPI_Barrier(d_mpiCommParent);
    double total_time = MPI_Wtime();

    double normValue = 0.0;

    distributedCPUVec<double> residualRho, residualRhoSpin0, residualRhoSpin1;
    residualRho.reinit(d_densityInNodalValues[0]);
    residualRhoSpin0.reinit(d_densityInNodalValues[0]);
    residualRhoSpin1.reinit(d_densityInNodalValues[0]);

    residualRho      = 0.0;
    residualRhoSpin0 = 0.0;
    residualRhoSpin1 = 0.0;

    // compute residual = rhoOut - rhoIn
    residualRho.add(1.0,
                    d_densityOutNodalValues[0],
                    -1.0,
                    d_densityInNodalValues[0]);
    residualRhoSpin0.add(0.5,
                         d_densityOutNodalValues[0],
                         0.5,
                         d_densityOutNodalValues[1]);
    residualRhoSpin0.add(-0.5,
                         d_densityInNodalValues[0],
                         -0.5,
                         d_densityInNodalValues[1]);
    residualRhoSpin1.add(0.5,
                         d_densityOutNodalValues[0],
                         -0.5,
                         d_densityOutNodalValues[1]);
    residualRhoSpin1.add(-0.5,
                         d_densityInNodalValues[0],
                         0.5,
                         d_densityInNodalValues[1]);

    residualRho.update_ghost_values();
    residualRhoSpin0.update_ghost_values();
    residualRhoSpin1.update_ghost_values();

    // compute l2 norm of the field residual
    normValue = rhofieldl2Norm(d_matrixFreeDataPRefined,
                               residualRho,
                               d_densityDofHandlerIndexElectro,
                               d_densityQuadratureIdElectro);

    double normValueSpin0 = rhofieldl2Norm(d_matrixFreeDataPRefined,
                                           residualRhoSpin0,
                                           d_densityDofHandlerIndexElectro,
                                           d_densityQuadratureIdElectro);

    double normValueSpin1 = rhofieldl2Norm(d_matrixFreeDataPRefined,
                                           residualRhoSpin1,
                                           d_densityDofHandlerIndexElectro,
                                           d_densityQuadratureIdElectro);

    const double k0 = 1.0;

    distributedCPUVec<double> kernelActionSpin0;
    distributedCPUVec<double> kernelActionSpin1;
    distributedCPUVec<double> compvecSpin0;
    distributedCPUVec<double> compvecSpin1;
    distributedCPUVec<double> tempDensityPrimeTotalVec;
    distributedCPUVec<double> dummy;

    kernelActionSpin0.reinit(residualRho);
    kernelActionSpin1.reinit(residualRho);
    compvecSpin0.reinit(residualRho);
    compvecSpin1.reinit(residualRho);
    tempDensityPrimeTotalVec.reinit(residualRho);
    dummy.reinit(residualRho);

    kernelActionSpin0        = 0;
    kernelActionSpin1        = 0;
    compvecSpin0             = 0;
    compvecSpin1             = 0;
    tempDensityPrimeTotalVec = 0;

    double             charge;
    const unsigned int local_size = residualRho.local_size();

    const unsigned int maxRankCurrentSCF = 20;


    d_vSpin0containerVals.clear();
    d_vSpin1containerVals.clear();
    d_fvSpin0containerVals.clear();
    d_fvSpin1containerVals.clear();
    d_rankCurrentLRD = 0;

    unsigned int       rankAddedInThisScf = 0;
    const unsigned int maxRankThisScf = (scfIter < 2) ? 5 : maxRankCurrentSCF;
    while ((rankAddedInThisScf < maxRankThisScf) ||
           ((normValue < d_dftParamsPtr->selfConsistentSolverTolerance) &&
            (d_dftParamsPtr->estimateJacCondNoFinalSCFIter)))
      {
        if (rankAddedInThisScf == 0)
          {
            d_vSpin0containerVals.push_back(residualRhoSpin0);
            d_vSpin1containerVals.push_back(residualRhoSpin1);
            d_vSpin0containerVals[d_rankCurrentLRD] *= k0;
            d_vSpin1containerVals[d_rankCurrentLRD] *= k0;
          }
        else
          {
            d_vSpin0containerVals.push_back(
              d_fvSpin0containerVals[d_rankCurrentLRD - 1]);
            d_vSpin1containerVals.push_back(
              d_fvSpin1containerVals[d_rankCurrentLRD - 1]);
          }

        compvecSpin0 = 0;
        compvecSpin1 = 0;
        for (int jrank = 0; jrank < d_rankCurrentLRD; jrank++)
          {
            const double tTvj = d_vSpin0containerVals[d_rankCurrentLRD] *
                                  d_vSpin0containerVals[jrank] +
                                d_vSpin1containerVals[d_rankCurrentLRD] *
                                  d_vSpin1containerVals[jrank];
            compvecSpin0.add(tTvj, d_vSpin0containerVals[jrank]);
            compvecSpin1.add(tTvj, d_vSpin1containerVals[jrank]);
          }
        d_vSpin0containerVals[d_rankCurrentLRD] -= compvecSpin0;
        d_vSpin1containerVals[d_rankCurrentLRD] -= compvecSpin1;

        const double normvmat = internalLowrankJacInv::frobeniusNormSpin(
          d_vSpin0containerVals[d_rankCurrentLRD],
          d_vSpin1containerVals[d_rankCurrentLRD]);


        d_vSpin0containerVals[d_rankCurrentLRD] *= 1.0 / normvmat;
        d_vSpin1containerVals[d_rankCurrentLRD] *= 1.0 / normvmat;

        const double normvmatNormalized =
          internalLowrankJacInv::frobeniusNormSpin(
            d_vSpin0containerVals[d_rankCurrentLRD],
            d_vSpin1containerVals[d_rankCurrentLRD]);

        if (d_dftParamsPtr->verbosity >= 4)
          pcout << " Matrix norm of V:  " << normvmatNormalized
                << ", for rank: " << d_rankCurrentLRD + 1 << std::endl;

        d_fvSpin0containerVals.push_back(residualRhoSpin0);
        d_fvSpin0containerVals[d_rankCurrentLRD] = 0;

        d_fvSpin1containerVals.push_back(residualRhoSpin1);
        d_fvSpin1containerVals[d_rankCurrentLRD] = 0;

        for (unsigned int idof = 0;
             idof < tempDensityPrimeTotalVec.local_size();
             idof++)
          tempDensityPrimeTotalVec.local_element(idof) =
            d_vSpin0containerVals[d_rankCurrentLRD].local_element(idof) +
            d_vSpin1containerVals[d_rankCurrentLRD].local_element(idof);

        tempDensityPrimeTotalVec.update_ghost_values();
        charge =
          totalCharge(d_matrixFreeDataPRefined, tempDensityPrimeTotalVec);


        if (d_dftParamsPtr->verbosity >= 4)
          pcout << "Integral V and contraction over spin before scaling:  "
                << charge << std::endl;

        d_vSpin0containerVals[d_rankCurrentLRD].add(-charge / d_domainVolume /
                                                    2.0);
        d_vSpin1containerVals[d_rankCurrentLRD].add(-charge / d_domainVolume /
                                                    2.0);

        // d_constraintsRhoNodal.set_zero(d_vSpin0containerVals[d_rankCurrentLRD]);
        // d_constraintsRhoNodal.set_zero(d_vSpin1containerVals[d_rankCurrentLRD]);

        for (unsigned int idof = 0;
             idof < tempDensityPrimeTotalVec.local_size();
             idof++)
          tempDensityPrimeTotalVec.local_element(idof) =
            d_vSpin0containerVals[d_rankCurrentLRD].local_element(idof) +
            d_vSpin1containerVals[d_rankCurrentLRD].local_element(idof);

        tempDensityPrimeTotalVec.update_ghost_values();
        charge =
          totalCharge(d_matrixFreeDataPRefined, tempDensityPrimeTotalVec);

        if (d_dftParamsPtr->verbosity >= 4)
          pcout << "Integral V and contraction over spin after scaling:  "
                << charge << std::endl;

        computeOutputDensityDirectionalDerivative(
          tempDensityPrimeTotalVec,
          d_vSpin0containerVals[d_rankCurrentLRD],
          d_vSpin1containerVals[d_rankCurrentLRD],
          dummy,
          d_fvSpin0containerVals[d_rankCurrentLRD],
          d_fvSpin1containerVals[d_rankCurrentLRD]);

        for (unsigned int idof = 0;
             idof < tempDensityPrimeTotalVec.local_size();
             idof++)
          tempDensityPrimeTotalVec.local_element(idof) =
            d_fvSpin0containerVals[d_rankCurrentLRD].local_element(idof) +
            d_fvSpin1containerVals[d_rankCurrentLRD].local_element(idof);

        tempDensityPrimeTotalVec.update_ghost_values();
        charge =
          totalCharge(d_matrixFreeDataPRefined, tempDensityPrimeTotalVec);


        if (d_dftParamsPtr->verbosity >= 4)
          pcout << "Integral fV and contraction over spin before scaling:  "
                << charge << std::endl;

        d_fvSpin0containerVals[d_rankCurrentLRD].add(-charge / d_domainVolume /
                                                     2.0);
        d_fvSpin1containerVals[d_rankCurrentLRD].add(-charge / d_domainVolume /
                                                     2.0);

        for (unsigned int idof = 0;
             idof < tempDensityPrimeTotalVec.local_size();
             idof++)
          tempDensityPrimeTotalVec.local_element(idof) =
            d_fvSpin0containerVals[d_rankCurrentLRD].local_element(idof) +
            d_fvSpin1containerVals[d_rankCurrentLRD].local_element(idof);

        tempDensityPrimeTotalVec.update_ghost_values();
        charge =
          totalCharge(d_matrixFreeDataPRefined, tempDensityPrimeTotalVec);

        if (d_dftParamsPtr->verbosity >= 4)
          pcout << "Integral fV and contraction over spin after scaling:  "
                << charge << std::endl;

        if (d_dftParamsPtr->verbosity >= 4)
          pcout
            << " Frobenius norm of response (delta rho_min[n+delta_lambda*v1]/ delta_lambda):  "
            << internalLowrankJacInv::frobeniusNormSpin(
                 d_fvSpin0containerVals[d_rankCurrentLRD],
                 d_fvSpin1containerVals[d_rankCurrentLRD])
            << " for kernel rank: " << d_rankCurrentLRD + 1 << std::endl;

        d_fvSpin0containerVals[d_rankCurrentLRD] -=
          d_vSpin0containerVals[d_rankCurrentLRD];
        d_fvSpin1containerVals[d_rankCurrentLRD] -=
          d_vSpin1containerVals[d_rankCurrentLRD];
        d_fvSpin0containerVals[d_rankCurrentLRD] *= k0;
        d_fvSpin1containerVals[d_rankCurrentLRD] *= k0;
        d_rankCurrentLRD++;
        rankAddedInThisScf++;

        if (d_dftParamsPtr->methodSubTypeLRD == "ADAPTIVE")
          {
            const double relativeApproxError =
              internalLowrankJacInv::relativeErrorEstimateSpin(
                d_fvSpin0containerVals,
                d_fvSpin1containerVals,
                residualRhoSpin0,
                residualRhoSpin1,
                k0);

            if (d_dftParamsPtr->verbosity >= 4)
              pcout << " Relative approx error:  " << relativeApproxError
                    << " for kernel rank: " << d_rankCurrentLRD << std::endl;

            if ((normValue < d_dftParamsPtr->selfConsistentSolverTolerance) &&
                (d_dftParamsPtr->estimateJacCondNoFinalSCFIter))
              {
                if (relativeApproxError < 1.0e-5)
                  {
                    break;
                  }
              }
            else
              {
                if (relativeApproxError < d_dftParamsPtr->adaptiveRankRelTolLRD)
                  {
                    d_tolReached = true;
                    break;
                  }
              }

            /*
            if (relativeApproxError < d_dftParamsPtr->adaptiveRankRelTolLRD)
              {
                break;
              }
            */
          }
      }


    if (d_dftParamsPtr->verbosity >= 4)
      pcout << " Net accumulated kernel rank:  " << d_rankCurrentLRD
            << " Accumulated in this scf: " << rankAddedInThisScf << std::endl;

    internalLowrankJacInv::lowrankKernelApplySpin(d_fvSpin0containerVals,
                                                  d_fvSpin1containerVals,
                                                  d_vSpin0containerVals,
                                                  d_vSpin1containerVals,
                                                  residualRhoSpin0,
                                                  residualRhoSpin1,
                                                  k0,
                                                  kernelActionSpin0,
                                                  kernelActionSpin1);

    if (normValue < d_dftParamsPtr->selfConsistentSolverTolerance &&
        d_dftParamsPtr->estimateJacCondNoFinalSCFIter)
      {
        const double maxAbsEigenValue = internalLowrankJacInv::
          estimateLargestEigenvalueMagJacLowrankPowerSpin(
            d_fvSpin0containerVals,
            d_fvSpin1containerVals,
            d_vSpin0containerVals,
            d_vSpin1containerVals,
            residualRhoSpin0,
            residualRhoSpin1,
            d_constraintsRhoNodal);
        const double minAbsEigenValue =
          1.0 / internalLowrankJacInv::
                  estimateLargestEigenvalueMagJacInvLowrankPowerSpin(
                    d_fvSpin0containerVals,
                    d_fvSpin1containerVals,
                    d_vSpin0containerVals,
                    d_vSpin1containerVals,
                    residualRhoSpin0,
                    residualRhoSpin1,
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

    d_densityInNodalValues[0].add(const2,
                                  kernelActionSpin0,
                                  const2,
                                  kernelActionSpin1);
    d_densityInNodalValues[1].add(const2,
                                  kernelActionSpin0,
                                  -const2,
                                  kernelActionSpin1);
    for (unsigned int iComp = 0; iComp < d_densityInNodalValues.size(); ++iComp)
      d_densityInNodalValues[iComp].update_ghost_values();

    for (unsigned int iComp = 0; iComp < d_densityInNodalValues.size(); ++iComp)
      interpolateDensityNodalDataToQuadratureDataGeneral(
        d_basisOperationsPtrElectroHost,
        d_densityDofHandlerIndexElectro,
        d_densityQuadratureIdElectro,
        d_densityInNodalValues[iComp],
        d_densityInQuadValues[iComp],
        d_gradDensityInQuadValues[iComp],
        d_gradDensityInQuadValues[iComp],
        d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA);

    MPI_Barrier(d_mpiCommParent);
    total_time = MPI_Wtime() - total_time;

    if (this_process == 0 && d_dftParamsPtr->verbosity >= 2)
      std::cout << "Time for low rank jac inv: " << total_time << std::endl;

    if (d_dftParamsPtr->verbosity >= 4)
      pcout << " Norm of residual in spin-polarized case:  "
            << std::sqrt(normValueSpin0 * normValueSpin0 +
                         normValueSpin1 * normValueSpin1)
            << std::endl;

    return normValue;
  }
#include "dft.inst.cc"
} // namespace dftfe
