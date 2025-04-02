#ifndef constraintMatrixInfoDeviceKernels_H
#define constraintMatrixInfoDeviceKernels_H
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherConstants.h>


namespace dftfe
{
  // Declare dftUtils functions
  namespace dftUtils
  {
    template <typename ValueType>
    void
    distributeDevice(
      const unsigned int  contiguousBlockSize,
      ValueType          *xVec,
      const unsigned int *constraintLocalRowIdsUnflattened,
      const unsigned int  numConstraints,
      const unsigned int *constraintRowSizes,
      const unsigned int *constraintRowSizesAccumulated,
      const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
      const double       *constraintColumnValuesAllRowsUnflattened,
      const double       *inhomogenities);

    template <typename ValueType>
    void
    distributeSlaveToMasterAtomicAddDevice(
      const unsigned int  contiguousBlockSize,
      ValueType          *xVec,
      const unsigned int *constraintLocalRowIdsUnflattened,
      const unsigned int  numConstraints,
      const unsigned int *constraintRowSizes,
      const unsigned int *constraintRowSizesAccumulated,
      const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
      const double       *constraintColumnValuesAllRowsUnflattened);

    template <typename ValueType>
    void
    setzeroDevice(const unsigned int  contiguousBlockSize,
                  ValueType          *xVec,
                  const unsigned int *constraintLocalRowIdsUnflattened,
                  const unsigned int  numConstraints);
    void
    scaleConstraintsDevice(
      const double       *xVec,
      const unsigned int *constraintLocalRowIdsUnflattened,
      const unsigned int  numConstraints,
      const unsigned int *constraintRowSizes,
      const unsigned int *constraintRowSizesAccumulated,
      const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
      double             *constraintColumnValuesAllRowsUnflattened);
  } // namespace dftUtils
} // namespace dftfe
#endif
