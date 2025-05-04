#ifndef linearSolverCGDeviceKernels_H
#define linearSolverCGDeviceKernels_H
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherConstants.h>


namespace dftfe
{ /**
   * @brief Combines precondition and dot product
   *
   */
  void
  applyPreconditionAndComputeDotProductDevice(double          *d_dvec,
                                              double          *d_devSum,
                                              const double    *d_rvec,
                                              const double    *d_jacobi,
                                              const dftfe::Int N);

  /**
   * @brief Combines precondition, sadd and dot product
   *
   */
  void
  applyPreconditionComputeDotProductAndSaddDevice(double          *d_qvec,
                                                  double          *d_devSum,
                                                  const double    *d_rvec,
                                                  const double    *d_jacobi,
                                                  const dftfe::Int N);

  /**
   * @brief Combines scaling and norm
   *
   */
  void
  scaleXRandComputeNormDevice(double          *x,
                              double          *d_rvec,
                              double          *d_devSum,
                              const double    *d_qvec,
                              const double    *d_dvec,
                              const double     alpha,
                              const dftfe::Int N);
} // namespace dftfe
#endif
