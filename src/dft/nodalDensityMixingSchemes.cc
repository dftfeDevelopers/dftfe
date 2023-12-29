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
// @author Phani Motamarri, Gourab Panigrahi
//

#include <dft.h>
#include <linearAlgebraOperations.h>

namespace dftfe
{
  // implement nodal anderson mixing scheme with Kerker
  template <unsigned int FEOrder, unsigned int FEOrderElectro>
  double
  dftClass<FEOrder, FEOrderElectro>::nodalDensity_mixing_anderson_kerker(
#ifdef DFTFE_WITH_DEVICE
    kerkerSolverProblemDevice<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>
      &                   kerkerPreconditionedResidualSolverProblemDevice,
    linearSolverCGDevice &CGSolverDevice,
#endif
    kerkerSolverProblem<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>
      &                 kerkerPreconditionedResidualSolverProblem,
    dealiiLinearSolver &CGSolver)
  {
    double normValue = 0.0;

    distributedCPUVec<double> residualRho;
    residualRho.reinit(d_rhoInNodalValues);

    residualRho = 0.0;

    // compute residual = rhoIn - rhoOut
    residualRho.add(1.0, d_rhoInNodalValues, -1.0, d_rhoOutNodalValues);

    residualRho.update_ghost_values();

    // compute l2 norm of the field residual
    normValue = rhofieldl2Norm(d_matrixFreeDataPRefined,
                               residualRho,
                               d_densityDofHandlerIndexElectro,
                               d_densityQuadratureIdElectro);


    // initialize data structures for computing mixing constants in Anderson
    // mixing scheme
    int                 N    = d_rhoOutNodalVals.size() - 1;
    int                 NRHS = 1, lda = N, ldb = N, info;
    std::vector<int>    ipiv(N);
    std::vector<double> ATotal(lda * N, 0.0), cTotal(ldb * NRHS, 0.0);


    // compute Fn = rhoOutVals[N] - rhoInVals[N]
    distributedCPUVec<double> Fn, Fnm, Fnk, FnMinusFnm, FnMinusFnk;
    Fn.reinit(d_rhoInNodalValues);
    Fnm.reinit(d_rhoInNodalValues);
    Fnk.reinit(d_rhoInNodalValues);
    FnMinusFnm.reinit(d_rhoInNodalValues);
    FnMinusFnk.reinit(d_rhoInNodalValues);

    Fn         = 0.0;
    Fnm        = 0.0;
    Fnk        = 0.0;
    FnMinusFnk = 0.0;
    FnMinusFnm = 0.0;

    Fn.add(1.0, d_rhoOutNodalVals[N], -1.0, d_rhoInNodalVals[N]);


    for (int m = 0; m < N; ++m)
      {
        Fnm = 0.0;

        Fnm.add(1.0,
                d_rhoOutNodalVals[N - 1 - m],
                -1.0,
                d_rhoInNodalVals[N - 1 - m]);

        FnMinusFnm = 0.0;

        FnMinusFnm.add(1.0, Fn, -1.0, Fnm);



        for (int k = 0; k < N; ++k)
          {
            Fnk = 0.0;

            Fnk.add(1.0,
                    d_rhoOutNodalVals[N - 1 - k],
                    -1.0,
                    d_rhoInNodalVals[N - 1 - k]);

            FnMinusFnk = 0.0;

            FnMinusFnk.add(1.0, Fn, -1.0, Fnk);

            ATotal[k * N + m] += FnMinusFnm * FnMinusFnk;
          }

        cTotal[m] += FnMinusFnm * Fn;
      }

    dgesv_(&N, &NRHS, &ATotal[0], &lda, &ipiv[0], &cTotal[0], &ldb, &info);
    if ((info > 0) && (this_mpi_process == 0))
      {
        printf(
          "Anderson Mixing: The diagonal element of the triangular factor of A,\n");
        printf(
          "U(%i,%i) is zero, so that A is singular.\nThe solution could not be computed.\n",
          info,
          info);
        exit(1);
      }

    double cn = 1.0;
    for (int i = 0; i < N; i++)
      cn -= cTotal[i];


    // do the linear combination of history of input rhos and output rhos
    distributedCPUVec<double> rhoInBar, rhoOutBar;
    rhoInBar.reinit(d_rhoInNodalValues);
    rhoOutBar.reinit(d_rhoInNodalValues);
    rhoInBar  = 0.0;
    rhoOutBar = 0.0;

    rhoInBar.add(cn, d_rhoInNodalVals[N]);

    rhoOutBar.add(cn, d_rhoOutNodalVals[N]);

    for (int i = 0; i < N; ++i)
      {
        rhoOutBar.add(cTotal[i], d_rhoOutNodalVals[N - 1 - i]);

        rhoInBar.add(cTotal[i], d_rhoInNodalVals[N - 1 - i]);
      }

    // compute difference in rhoInBar and rhoOutBar
    distributedCPUVec<double> diffRhoBar;
    diffRhoBar.reinit(d_rhoInNodalValues);
    diffRhoBar = 0.0;

    diffRhoBar.add(1.0, rhoInBar, -1.0, rhoOutBar);

    diffRhoBar.update_ghost_values();

    // create FEEval object to be used subsequently
    dealii::FEEvaluation<
      3,
      C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>(),
      C_num1DQuad<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>(),
      1,
      double>
                 fe_evalHelm(d_matrixFreeDataPRefined,
                  d_densityDofHandlerIndexElectro,
                  d_densityQuadratureIdElectro);
    unsigned int numQuadPoints = fe_evalHelm.n_q_points;
    dealii::DoFHandler<3>::active_cell_iterator subCellPtr;

    // preparation for rhs of Helmholtz solve by computing gradients of
    // (rhoInBar
    // - rhoOutBar)
    std::map<dealii::CellId, std::vector<double>> gradDensityResidualValuesMap;
    for (unsigned int cell = 0;
         cell < d_matrixFreeDataPRefined.n_cell_batches();
         ++cell)
      {
        fe_evalHelm.reinit(cell);
        fe_evalHelm.read_dof_values(diffRhoBar);
        fe_evalHelm.evaluate(false, true);

        for (unsigned int iSubCell = 0;
             iSubCell <
             d_matrixFreeDataPRefined.n_active_entries_per_cell_batch(cell);
             ++iSubCell)
          {
            subCellPtr = d_matrixFreeDataPRefined.get_cell_iterator(
              cell, iSubCell, d_densityDofHandlerIndexElectro);
            dealii::CellId subCellId = subCellPtr->id();

            gradDensityResidualValuesMap[subCellId] =
              std::vector<double>(3 * numQuadPoints);
            std::vector<double> &gradDensityResidualValues =
              gradDensityResidualValuesMap.find(subCellId)->second;

            for (unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
              {
                gradDensityResidualValues[3 * q_point + 0] =
                  fe_evalHelm.get_gradient(q_point)[0][iSubCell];
                gradDensityResidualValues[3 * q_point + 1] =
                  fe_evalHelm.get_gradient(q_point)[1][iSubCell];
                gradDensityResidualValues[3 * q_point + 2] =
                  fe_evalHelm.get_gradient(q_point)[2][iSubCell];
              }
          }
      }

    // initialize helmholtz solver function object with the quantity required
    // for computing rhs, solution vector and mixing constant

    if (d_dftParamsPtr->verbosity >= 2)
      pcout
        << "Solving Helmholtz equation for Kerker Preconditioning of nodal fields: "
        << std::endl;

    if (d_dftParamsPtr->useDevice and d_dftParamsPtr->floatingNuclearCharges and
        d_dftParamsPtr->poissonGPU)
      {
#ifdef DFTFE_WITH_DEVICE
        kerkerPreconditionedResidualSolverProblemDevice.reinit(
          d_preCondResidualVector, gradDensityResidualValuesMap);
#endif
      }
    else
      kerkerPreconditionedResidualSolverProblem.reinit(
        d_preCondResidualVector, gradDensityResidualValuesMap);

    // solve the Helmholtz system to compute preconditioned residual
    if (d_dftParamsPtr->useDevice and d_dftParamsPtr->floatingNuclearCharges and
        d_dftParamsPtr->poissonGPU)
      {
#ifdef DFTFE_WITH_DEVICE
        CGSolverDevice.solve(
          kerkerPreconditionedResidualSolverProblemDevice,
          d_dftParamsPtr->absLinearSolverToleranceHelmholtz,
          d_dftParamsPtr->maxLinearSolverIterationsHelmholtz,
          d_kohnShamDFTOperatorDevicePtr->getDeviceBlasHandle(),
          d_dftParamsPtr->verbosity,
          false);
#endif
      }
    else
      CGSolver.solve(kerkerPreconditionedResidualSolverProblem,
                     d_dftParamsPtr->absLinearSolverToleranceHelmholtz,
                     d_dftParamsPtr->maxLinearSolverIterationsHelmholtz,
                     d_dftParamsPtr->verbosity,
                     false);

    // rhoIn += mixingScalar*residual for Kerker
    d_rhoInNodalValues = 0.0;
    double const2      = -d_dftParamsPtr->mixingParameter;
    d_rhoInNodalValues.add(1.0, rhoInBar, const2, d_preCondResidualVector);

    d_rhoInNodalValues.update_ghost_values();

    // push the rhoIn to deque storing the history of nodal values
    d_rhoInNodalVals.push_back(d_rhoInNodalValues);

    // interpolate nodal values to quadrature data


    /*dealii::FEEvaluation<3,C_num1DKerkerPoly<FEOrder>(),C_num1DQuad<FEOrder>(),1,double>
      fe_evalRho(d_matrixFreeDataPRefined,0,1); numQuadPoints =
      fe_evalRho.n_q_points; for(unsigned int cell = 0; cell <
      d_matrixFreeDataPRefined.n_cell_batches(); ++cell)
      {
      fe_evalRho.reinit(cell);
      fe_evalRho.read_dof_values(d_rhoInNodalValues);
      fe_evalRho.evaluate(true,true);
      for(unsigned int iSubCell = 0; iSubCell <
      d_matrixFreeDataPRefined.n_active_entries_per_cell_batch(cell);
      ++iSubCell)
      {
      subCellPtr= d_matrixFreeDataPRefined.get_cell_iterator(cell,iSubCell);
      dealii::CellId subCellId=subCellPtr->id();
      (*rhoInValues)[subCellId] = std::vector<double>(numQuadPoints);
      std::vector<double> & tempVec = rhoInValues->find(subCellId)->second;
      for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
      {
      tempVec[q_point] = fe_evalRho.get_value(q_point)[iSubCell];
      }
      }

      if(d_excManagerPtr->getDensityBasedFamilyType() ==
      densityFamilyType::GGA)
      {
      for(unsigned int iSubCell = 0; iSubCell <
      d_matrixFreeDataPRefined.n_active_entries_per_cell_batch(cell);
      ++iSubCell)
      {
      subCellPtr= d_matrixFreeDataPRefined.get_cell_iterator(cell,iSubCell);
      dealii::CellId subCellId=subCellPtr->id();
      (*gradRhoInValues)[subCellId]=std::vector<double>(3*numQuadPoints);
      std::vector<double> & tempVec = gradRhoInValues->find(subCellId)->second;
      for(unsigned int q_point = 0; q_point < numQuadPoints; ++q_point)
      {
      tempVec[3*q_point + 0] = fe_evalRho.get_gradient(q_point)[0][iSubCell];
      tempVec[3*q_point + 1] = fe_evalRho.get_gradient(q_point)[1][iSubCell];
      tempVec[3*q_point + 2] = fe_evalRho.get_gradient(q_point)[2][iSubCell];
      }
      }
      }

      }*/

    interpolateRhoNodalDataToQuadratureDataGeneral(
      d_matrixFreeDataPRefined,
      d_densityDofHandlerIndexElectro,
      d_densityQuadratureIdElectro,
      d_rhoInNodalValues,
      *rhoInValues,
      *gradRhoInValues,
      *gradRhoInValues,
      d_excManagerPtr->getDensityBasedFamilyType() == densityFamilyType::GGA);


    return normValue;
  }
#include "dft.inst.cc"
} // namespace dftfe
