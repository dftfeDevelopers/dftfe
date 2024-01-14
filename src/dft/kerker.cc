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
  void
  dftClass<FEOrder, FEOrderElectro>::
    applyKerkerPreconditionerToTotalDensityResidual(
#ifdef DFTFE_WITH_DEVICE
      kerkerSolverProblemDevice<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>
        &                   kerkerPreconditionedResidualSolverProblemDevice,
      linearSolverCGDevice &CGSolverDevice,
#endif
      kerkerSolverProblem<C_rhoNodalPolyOrder<FEOrder, FEOrderElectro>()>
        &                 kerkerPreconditionedResidualSolverProblem,
      dealiiLinearSolver &CGSolver,
      const distributedCPUVec<double> &residualRho,
      distributedCPUVec<double> &      preCondTotalDensityResidualVector)
  {
    // create FEEval object to be used subsequently
    residualRho.update_ghost_values();

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
    // residualRho
    std::map<dealii::CellId, std::vector<double>> gradDensityResidualValuesMap;
    for (unsigned int cell = 0;
         cell < d_matrixFreeDataPRefined.n_cell_batches();
         ++cell)
      {
        fe_evalHelm.reinit(cell);
        fe_evalHelm.read_dof_values(residualRho);
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
          preCondTotalDensityResidualVector, gradDensityResidualValuesMap);
#endif
      }
    else
      kerkerPreconditionedResidualSolverProblem.reinit(
        preCondTotalDensityResidualVector, gradDensityResidualValuesMap);

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
  }
#include "dft.inst.cc"
} // namespace dftfe
