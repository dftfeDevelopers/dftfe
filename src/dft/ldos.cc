#include <dftfe/dft.h>
#include <dftfe/linearAlgebraOperations.h>

namespace dftfe
{
  // implement nodal anderson mixing scheme with LDOS
  template <dftfe::utils::MemorySpace memorySpace>
  void
  dftClass<memorySpace>::applyLdosPreconditionerToTotalDensityResidual(
#ifdef DFTFE_WITH_DEVICE
    ldosSolverProblemDeviceWrapperClass
                         &ldosPreconditionedResidualSolverProblemDevice,
    linearSolverCGDevice &CGSolverDevice,
#endif
    ldosSolverProblemWrapperClass &ldosPreconditionedResidualSolverProblem,
    dealiiLinearSolver            &CGSolver,
    distributedCPUVec<double>     &residualRho,
    distributedCPUVec<double>     &preCondTotalDensityResidualVector)
  {
    preCondTotalDensityResidualVector = 0.0;
    double I                          = 0.0;

    dftfe::utils::MemoryStorage<double, dftfe::utils::MemorySpace::HOST> dummy;
    d_densityResidualQuadValues.resize(1);
    d_basisOperationsPtrElectroHost->interpolate(
      residualRho,
      d_densityDofHandlerIndexElectro,
      d_densityQuadratureIdElectro,
      d_densityResidualQuadValues[0],
      dummy,
      dummy,
      false);

    if (d_dftParamsPtr->verbosity >= 2)
      pcout
        << "Solving Helmholtz equation for LDOS Preconditioning of nodal fields: "
        << std::endl;

    if (d_dftParamsPtr->useDevice and d_dftParamsPtr->poissonGPU and
        d_dftParamsPtr->floatingNuclearCharges)
      {
#ifdef DFTFE_WITH_DEVICE
        ldosPreconditionedResidualSolverProblemDevice.reinit(
          preCondTotalDensityResidualVector,
          d_densityResidualQuadValues[0],
          d_ldosQuadValuesElectro,
          d_ldosAxQuadValuesElectro,
          d_totalDOS);
#endif
      }
    else
      ldosPreconditionedResidualSolverProblem.reinit(
        preCondTotalDensityResidualVector,
        d_densityResidualQuadValues[0],
        d_ldosQuadValuesElectro,
        d_ldosAxQuadValuesElectro,
        d_totalDOS);

    if (d_dftParamsPtr->useDevice and d_dftParamsPtr->poissonGPU and
        d_dftParamsPtr->floatingNuclearCharges)
      {
#ifdef DFTFE_WITH_DEVICE
        CGSolverDevice.solve(ldosPreconditionedResidualSolverProblemDevice,
                             d_dftParamsPtr->absLinearSolverToleranceHelmholtz,
                             d_dftParamsPtr->maxLinearSolverIterationsHelmholtz,
                             d_dftParamsPtr->verbosity);

        I = 4.0 * M_PI *
            ldosPreconditionedResidualSolverProblemDevice.computeDlocIntegral(
              preCondTotalDensityResidualVector);
#endif
      }
    else
      {
        CGSolver.solve(ldosPreconditionedResidualSolverProblem,
                       d_dftParamsPtr->absLinearSolverToleranceHelmholtz,
                       d_dftParamsPtr->maxLinearSolverIterationsHelmholtz,
                       d_dftParamsPtr->verbosity,
                       false);

        I = 4.0 * M_PI *
            ldosPreconditionedResidualSolverProblem.computeDlocIntegral(
              preCondTotalDensityResidualVector);
      }
    preCondTotalDensityResidualVector.scale(d_ldosNodalValues);
    preCondTotalDensityResidualVector *= (4.0 * M_PI);
    preCondTotalDensityResidualVector.add(1.0, residualRho);
    preCondTotalDensityResidualVector.add(-(I / d_totalDOS), d_ldosNodalValues);
  }
#include "dft.inst.cc"
} // namespace dftfe
