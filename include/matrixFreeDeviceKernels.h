#ifndef poissonSolverProblemDeviceKernels_H
#define poissonSolverProblemDeviceKernels_H
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherConstants.h>


namespace dftfe
{
  template <typename Type, int M, int N, int K, int dim>
  struct matrixFreeDeviceKernels
  {
    static void
    computeAXDevicePoisson(const int   blocks,
                           const int   threads,
                           const int   smem,
                           Type       *V,
                           const Type *U,
                           const Type *P,
                           const Type *J,
                           const int  *map);

    static void
    computeAXDeviceHelmholtz(const int   blocks,
                             const int   threads,
                             const int   smem,
                             Type       *V,
                             const Type *U,
                             const Type *P,
                             const Type *J,
                             const int  *map,
                             const Type  coeffHelmholtz);

    static void
    computeAXDevicePoissonSetAttributes(const int smem);

    static void
    computeAXDeviceHelmholtzSetAttributes(const int smem);
  };

} // namespace dftfe
#endif
