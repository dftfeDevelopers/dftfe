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
      const dftfe::uInt  contiguousBlockSize,
      ValueType         *xVec,
      const dftfe::uInt *constraintLocalRowIdsUnflattened,
      const dftfe::uInt  numConstraints,
      const dftfe::uInt *constraintRowSizes,
      const dftfe::uInt *constraintRowSizesAccumulated,
      const dftfe::uInt *constraintLocalColumnIdsAllRowsUnflattened,
      const double      *constraintColumnValuesAllRowsUnflattened,
      const double      *inhomogenities);

    template <typename ValueType>
    void
    distributeSlaveToMasterAtomicAddDevice(
      const dftfe::uInt  contiguousBlockSize,
      ValueType         *xVec,
      const dftfe::uInt *constraintLocalRowIdsUnflattened,
      const dftfe::uInt  numConstraints,
      const dftfe::uInt *constraintRowSizes,
      const dftfe::uInt *constraintRowSizesAccumulated,
      const dftfe::uInt *constraintLocalColumnIdsAllRowsUnflattened,
      const double      *constraintColumnValuesAllRowsUnflattened);

    template <typename ValueType>
    void
    setzeroDevice(const dftfe::uInt  contiguousBlockSize,
                  ValueType         *xVec,
                  const dftfe::uInt *constraintLocalRowIdsUnflattened,
                  const dftfe::uInt  numConstraints);
    void
    scaleConstraintsDevice(
      const double      *xVec,
      const dftfe::uInt *constraintLocalRowIdsUnflattened,
      const dftfe::uInt  numConstraints,
      const dftfe::uInt *constraintRowSizes,
      const dftfe::uInt *constraintRowSizesAccumulated,
      const dftfe::uInt *constraintLocalColumnIdsAllRowsUnflattened,
      double            *constraintColumnValuesAllRowsUnflattened);
  } // namespace dftUtils
} // namespace dftfe
#endif
