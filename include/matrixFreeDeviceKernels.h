#ifndef poissonSolverProblemDeviceKernels_H
#define poissonSolverProblemDeviceKernels_H
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherConstants.h>


namespace dftfe
{
  template <typename Type,
            dftfe::Int M,
            dftfe::Int N,
            dftfe::Int K,
            dftfe::Int dim>
  struct matrixFreeDeviceKernels
  {
    static void
    computeAXDevicePoisson(const dftfe::Int  blocks,
                           const dftfe::Int  threads,
                           const dftfe::Int  smem,
                           Type             *V,
                           const Type       *U,
                           const Type       *P,
                           const Type       *J,
                           const dftfe::Int *map);

    static void
    computeAXDeviceHelmholtz(const dftfe::Int  blocks,
                             const dftfe::Int  threads,
                             const dftfe::Int  smem,
                             Type             *V,
                             const Type       *U,
                             const Type       *P,
                             const Type       *J,
                             const dftfe::Int *map,
                             const Type        coeffHelmholtz);

    static void
    computeAXDevicePoissonSetAttributes(const dftfe::Int smem);

    static void
    computeAXDeviceHelmholtzSetAttributes(const dftfe::Int smem);
  };

} // namespace dftfe
#endif
