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
} // namespace internalLowrankJacInv


template <unsigned int FEOrder, unsigned int FEOrderElectro>
double
dftClass<FEOrder, FEOrderElectro>::lowrankApproxScfJacobianInvSpinPolarized(
  const unsigned int scfIter)
{
  int this_process;
  MPI_Comm_rank(d_mpiCommParent, &this_process);
  MPI_Barrier(d_mpiCommParent);
  double total_time = MPI_Wtime();

  double normValue = 0.0;

  distributedCPUVec<double> residualRho, residualRhoSpin0, residualRhoSpin1;
  residualRho.reinit(d_rhoInNodalValues);
  residualRhoSpin0.reinit(d_rhoInNodalValues);
  residualRhoSpin1.reinit(d_rhoInNodalValues);

  residualRho      = 0.0;
  residualRhoSpin0 = 0.0;
  residualRhoSpin1 = 0.0;

  // compute residual = rhoOut - rhoIn
  residualRho.add(1.0, d_rhoOutNodalValues, -1.0, d_rhoInNodalValues);
  residualRhoSpin0.add(1.0,
                       d_rhoOutSpin0NodalValues,
                       -1.0,
                       d_rhoInSpin0NodalValues);
  residualRhoSpin1.add(1.0,
                       d_rhoOutSpin1NodalValues,
                       -1.0,
                       d_rhoInSpin1NodalValues);

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

  const unsigned int maxRankCurrentSCF =
    d_dftParamsPtr->methodSubTypeLRJI == "ACCUMULATED_ADAPTIVE" ? 15 : 20;
  const unsigned int maxRankAccum = 20;

  if (d_rankCurrentLRJI >= 1 &&
      d_dftParamsPtr->methodSubTypeLRJI == "ACCUMULATED_ADAPTIVE")
    {
      const double relativeApproxError =
        internalLowrankJacInv::relativeErrorEstimateSpin(d_fvSpin0containerVals,
                                                         d_fvSpin1containerVals,
                                                         residualRhoSpin0,
                                                         residualRhoSpin1,
                                                         k0);
      if (d_rankCurrentLRJI >= maxRankAccum ||
          (relativeApproxError > d_dftParamsPtr->adaptiveRankRelTolLRJI *
                                   d_dftParamsPtr->factorAdapAccumClearLRJI) ||
          relativeApproxError > d_relativeErrorJacInvApproxPrevScfLRJI)
        {
          if (d_dftParamsPtr->verbosity >= 4)
            pcout
              << " Clearing accumulation as relative tolerance metric exceeded "
              << ", relative tolerance current scf: " << relativeApproxError
              << ", relative tolerance prev scf: "
              << d_relativeErrorJacInvApproxPrevScfLRJI << std::endl;
          d_vSpin0containerVals.clear();
          d_vSpin1containerVals.clear();
          d_fvSpin0containerVals.clear();
          d_fvSpin1containerVals.clear();
          d_rankCurrentLRJI                      = 0;
          d_relativeErrorJacInvApproxPrevScfLRJI = 100.0;
        }
      else
        d_relativeErrorJacInvApproxPrevScfLRJI = relativeApproxError;
    }
  else
    {
      d_vSpin0containerVals.clear();
      d_vSpin1containerVals.clear();
      d_fvSpin0containerVals.clear();
      d_fvSpin1containerVals.clear();
      d_rankCurrentLRJI = 0;
    }

  unsigned int       rankAddedInThisScf = 0;
  const unsigned int maxRankThisScf     = (scfIter < 2) ? 5 : maxRankCurrentSCF;
  while (((rankAddedInThisScf < maxRankThisScf) &&
          d_rankCurrentLRJI < maxRankAccum) ||
         ((normValue < d_dftParamsPtr->selfConsistentSolverTolerance) &&
          (d_dftParamsPtr->estimateJacCondNoFinalSCFIter)))
    {
      if (rankAddedInThisScf == 0)
        {
          d_vSpin0containerVals.push_back(residualRhoSpin0);
          d_vSpin1containerVals.push_back(residualRhoSpin1);
          d_vSpin0containerVals[d_rankCurrentLRJI] *= k0;
          d_vSpin1containerVals[d_rankCurrentLRJI] *= k0;
        }
      else
        {
          d_vSpin0containerVals.push_back(
            d_fvSpin0containerVals[d_rankCurrentLRJI - 1]);
          d_vSpin1containerVals.push_back(
            d_fvSpin1containerVals[d_rankCurrentLRJI - 1]);
        }

      compvecSpin0 = 0;
      compvecSpin1 = 0;
      for (int jrank = 0; jrank < d_rankCurrentLRJI; jrank++)
        {
          const double tTvj = d_vSpin0containerVals[d_rankCurrentLRJI] *
                                d_vSpin0containerVals[jrank] +
                              d_vSpin1containerVals[d_rankCurrentLRJI] *
                                d_vSpin1containerVals[jrank];
          compvecSpin0.add(tTvj, d_vSpin0containerVals[jrank]);
          compvecSpin1.add(tTvj, d_vSpin1containerVals[jrank]);
        }
      d_vSpin0containerVals[d_rankCurrentLRJI] -= compvecSpin0;
      d_vSpin1containerVals[d_rankCurrentLRJI] -= compvecSpin1;

      const double normvmat = internalLowrankJacInv::frobeniusNormSpin(
        d_vSpin0containerVals[d_rankCurrentLRJI],
        d_vSpin1containerVals[d_rankCurrentLRJI]);


      d_vSpin0containerVals[d_rankCurrentLRJI] *= 1.0 / normvmat;
      d_vSpin1containerVals[d_rankCurrentLRJI] *= 1.0 / normvmat;

      const double normvmatNormalized =
        internalLowrankJacInv::frobeniusNormSpin(
          d_vSpin0containerVals[d_rankCurrentLRJI],
          d_vSpin1containerVals[d_rankCurrentLRJI]);

      if (d_dftParamsPtr->verbosity >= 4)
        pcout << " Matrix norm of V:  " << normvmatNormalized
              << ", for rank: " << d_rankCurrentLRJI + 1 << std::endl;

      d_fvSpin0containerVals.push_back(residualRhoSpin0);
      d_fvSpin0containerVals[d_rankCurrentLRJI] = 0;

      d_fvSpin1containerVals.push_back(residualRhoSpin1);
      d_fvSpin1containerVals[d_rankCurrentLRJI] = 0;

      for (unsigned int idof = 0; idof < d_rhoInNodalValues.local_size();
           idof++)
        tempDensityPrimeTotalVec.local_element(idof) =
          d_vSpin0containerVals[d_rankCurrentLRJI].local_element(idof) +
          d_vSpin1containerVals[d_rankCurrentLRJI].local_element(idof);

      tempDensityPrimeTotalVec.update_ghost_values();
      charge = totalCharge(d_matrixFreeDataPRefined, tempDensityPrimeTotalVec);


      if (d_dftParamsPtr->verbosity >= 4)
        pcout << "Integral V and contraction over spin before scaling:  "
              << charge << std::endl;

      d_vSpin0containerVals[d_rankCurrentLRJI].add(-charge / d_domainVolume /
                                                   2.0);
      d_vSpin1containerVals[d_rankCurrentLRJI].add(-charge / d_domainVolume /
                                                   2.0);

      // d_constraintsRhoNodal.set_zero(d_vSpin0containerVals[d_rankCurrentLRJI]);
      // d_constraintsRhoNodal.set_zero(d_vSpin1containerVals[d_rankCurrentLRJI]);

      for (unsigned int idof = 0; idof < d_rhoInNodalValues.local_size();
           idof++)
        tempDensityPrimeTotalVec.local_element(idof) =
          d_vSpin0containerVals[d_rankCurrentLRJI].local_element(idof) +
          d_vSpin1containerVals[d_rankCurrentLRJI].local_element(idof);

      tempDensityPrimeTotalVec.update_ghost_values();
      charge = totalCharge(d_matrixFreeDataPRefined, tempDensityPrimeTotalVec);

      if (d_dftParamsPtr->verbosity >= 4)
        pcout << "Integral V and contraction over spin after scaling:  "
              << charge << std::endl;

      computeOutputDensityDirectionalDerivative(
        tempDensityPrimeTotalVec,
        d_vSpin0containerVals[d_rankCurrentLRJI],
        d_vSpin1containerVals[d_rankCurrentLRJI],
        dummy,
        d_fvSpin0containerVals[d_rankCurrentLRJI],
        d_fvSpin1containerVals[d_rankCurrentLRJI]);

      for (unsigned int idof = 0; idof < d_rhoInNodalValues.local_size();
           idof++)
        tempDensityPrimeTotalVec.local_element(idof) =
          d_fvSpin0containerVals[d_rankCurrentLRJI].local_element(idof) +
          d_fvSpin1containerVals[d_rankCurrentLRJI].local_element(idof);

      tempDensityPrimeTotalVec.update_ghost_values();
      charge = totalCharge(d_matrixFreeDataPRefined, tempDensityPrimeTotalVec);


      if (d_dftParamsPtr->verbosity >= 4)
        pcout << "Integral fV and contraction over spin before scaling:  "
              << charge << std::endl;

      d_fvSpin0containerVals[d_rankCurrentLRJI].add(-charge / d_domainVolume /
                                                    2.0);
      d_fvSpin1containerVals[d_rankCurrentLRJI].add(-charge / d_domainVolume /
                                                    2.0);

      for (unsigned int idof = 0; idof < d_rhoInNodalValues.local_size();
           idof++)
        tempDensityPrimeTotalVec.local_element(idof) =
          d_fvSpin0containerVals[d_rankCurrentLRJI].local_element(idof) +
          d_fvSpin1containerVals[d_rankCurrentLRJI].local_element(idof);

      tempDensityPrimeTotalVec.update_ghost_values();
      charge = totalCharge(d_matrixFreeDataPRefined, tempDensityPrimeTotalVec);

      if (d_dftParamsPtr->verbosity >= 4)
        pcout << "Integral fV and contraction over spin after scaling:  "
              << charge << std::endl;

      if (d_dftParamsPtr->verbosity >= 4)
        pcout
          << " Frobenius norm of response (delta rho_min[n+delta_lambda*v1]/ delta_lambda):  "
          << internalLowrankJacInv::frobeniusNormSpin(
               d_fvSpin0containerVals[d_rankCurrentLRJI],
               d_fvSpin1containerVals[d_rankCurrentLRJI])
          << " for kernel rank: " << d_rankCurrentLRJI + 1 << std::endl;

      d_fvSpin0containerVals[d_rankCurrentLRJI] -=
        d_vSpin0containerVals[d_rankCurrentLRJI];
      d_fvSpin1containerVals[d_rankCurrentLRJI] -=
        d_vSpin1containerVals[d_rankCurrentLRJI];
      d_fvSpin0containerVals[d_rankCurrentLRJI] *= k0;
      d_fvSpin1containerVals[d_rankCurrentLRJI] *= k0;
      d_rankCurrentLRJI++;
      rankAddedInThisScf++;

      if (d_dftParamsPtr->methodSubTypeLRJI == "ADAPTIVE" ||
          d_dftParamsPtr->methodSubTypeLRJI == "ACCUMULATED_ADAPTIVE")
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
                  << " for kernel rank: " << d_rankCurrentLRJI << std::endl;

          if (relativeApproxError < d_dftParamsPtr->adaptiveRankRelTolLRJI)
            {
              break;
            }
        }
    }


  if (d_dftParamsPtr->verbosity >= 4)
    pcout << " Net accumulated kernel rank:  " << d_rankCurrentLRJI
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


  // pcout << " Preconditioned simple mixing step " << std::endl;
  // preconditioned simple mixing step
  // Note for const=-1.0, it should be same as Newton step
  // For second scf iteration step (scIter==1), the rhoIn is from atomic
  // densities which casues robustness issues when used with a
  // higher mixingParameter value.
  // Suggested to use 0.1 for initial steps
  // as well as when normValue is greater than 2.0
  double const2 =
    (normValue > d_dftParamsPtr->startingNormLRJILargeDamping || scfIter < 2) ?
      -0.1 :
      -d_dftParamsPtr->mixingParameterLRJI;

  pcout << " Preconditioned mixing step, mixing constant: " << const2
        << std::endl;

  d_rhoInSpin0NodalValues.add(const2, kernelActionSpin0);
  d_rhoInSpin1NodalValues.add(const2, kernelActionSpin1);

  d_rhoInSpin0NodalValues.update_ghost_values();
  d_rhoInSpin1NodalValues.update_ghost_values();

  interpolateRhoSpinNodalDataToQuadratureDataGeneral(
    d_matrixFreeDataPRefined,
    d_densityDofHandlerIndexElectro,
    d_densityQuadratureIdElectro,
    d_rhoInSpin0NodalValues,
    d_rhoInSpin1NodalValues,
    *rhoInValuesSpinPolarized,
    *gradRhoInValuesSpinPolarized,
    *gradRhoInValuesSpinPolarized,
    d_dftParamsPtr->xcFamilyType == "GGA");

  // push the rhoIn to deque storing the history of nodal values
  d_rhoInSpin0NodalVals.push_back(d_rhoInSpin0NodalValues);
  d_rhoInSpin1NodalVals.push_back(d_rhoInSpin1NodalValues);

  for (unsigned int idof = 0; idof < d_rhoInNodalValues.local_size(); idof++)
    d_rhoInNodalValues.local_element(idof) =
      d_rhoInSpin0NodalValues.local_element(idof) +
      d_rhoInSpin1NodalValues.local_element(idof);

  d_rhoInNodalValues.update_ghost_values();

  // interpolate nodal data to quadrature data
  interpolateRhoNodalDataToQuadratureDataGeneral(
    d_matrixFreeDataPRefined,
    d_densityDofHandlerIndexElectro,
    d_densityQuadratureIdElectro,
    d_rhoInNodalValues,
    *rhoInValues,
    *gradRhoInValues,
    *gradRhoInValues,
    d_dftParamsPtr->xcFamilyType == "GGA");

  // push the rhoIn to deque storing the history of nodal values
  d_rhoInNodalVals.push_back(d_rhoInNodalValues);

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
