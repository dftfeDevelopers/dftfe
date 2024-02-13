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
    preCondTotalDensityResidualVector = 0.0;
    // create FEEval object to be used subsequently
    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> dummy;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST>
      residualQuadValues;
    d_densityResidualQuadValues.resize(1);
    interpolateDensityNodalDataToQuadratureDataGeneral(
      d_basisOperationsPtrElectroHost,
      d_densityDofHandlerIndexElectro,
      d_densityQuadratureIdElectro,
      residualRho,
      d_densityResidualQuadValues[0],
      dummy,
      dummy,
      false);

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
          preCondTotalDensityResidualVector, d_densityResidualQuadValues[0]);
#endif
      }
    else
      kerkerPreconditionedResidualSolverProblem.reinit(
        preCondTotalDensityResidualVector, d_densityResidualQuadValues[0]);

    // solve the Helmholtz system to compute preconditioned residual
    if (d_dftParamsPtr->useDevice and d_dftParamsPtr->floatingNuclearCharges and
        d_dftParamsPtr->poissonGPU)
      {
#ifdef DFTFE_WITH_DEVICE
        CGSolverDevice.solve(kerkerPreconditionedResidualSolverProblemDevice,
                             d_dftParamsPtr->absLinearSolverToleranceHelmholtz,
                             d_dftParamsPtr->maxLinearSolverIterationsHelmholtz,
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
    if (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_KERKER")
      preCondTotalDensityResidualVector.sadd(
        4 * M_PI * d_dftParamsPtr->kerkerParameter, 1.0, residualRho);
    else if (d_dftParamsPtr->mixingMethod == "ANDERSON_WITH_RESTA")
      {
        double kappa =
          std::sqrt(d_dftParamsPtr->restaFermiWavevector / 4.0 / M_PI);
        double beta = d_dftParamsPtr->restaScreeningLength;
        double gamma =
          kappa * beta > 1e-8 ? std::sinh(kappa * beta) / kappa / beta : 1.0;


        preCondTotalDensityResidualVector.sadd(
          kappa * kappa - kappa * kappa / gamma -
            beta * beta * kappa * kappa * kappa * kappa / 6.0 / gamma,
          1.0 - beta * beta * kappa * kappa / 6.0 / gamma,
          residualRho);
      }
  }
#include "dft.inst.cc"
} // namespace dftfe
