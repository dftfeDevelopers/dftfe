#ifndef poissonSolverProblemDeviceKernels_H
#define poissonSolverProblemDeviceKernels_H
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherConstants.h>


namespace dftfe
{
  template <typename Type, int M, int N, int K, int dim>
  void
  computeAXDevicePoisson(const int   blocks,
                         const int   threads,
                         const int   smem,
                         Type *      V,
                         const Type *U,
                         const Type *P,
                         const Type *J,
                         const int * map);

  template <typename Type, int M, int N, int K, int dim>
  void
  computeAXDeviceHelmholtz(const int   blocks,
                           const int   threads,
                           const int   smem,
                           Type *      V,
                           const Type *U,
                           const Type *P,
                           const Type *J,
                           const int * map,
                           const Type  coeffHelmholtz);

} // namespace dftfe
#endif
