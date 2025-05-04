#ifndef linearAlgebraOperationsDeviceKernels_H
#define linearAlgebraOperationsDeviceKernels_H
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherConstants.h>
namespace dftfe
{
  namespace linearAlgebraOperationsDevice
  {
    template <typename ValueType1, typename ValueType2>
    void
    addSubspaceRotatedBlockToX(const dftfe::uInt             BDof,
                               const dftfe::uInt             BVec,
                               const ValueType1             *rotatedXBlockSP,
                               ValueType2                   *X,
                               const dftfe::uInt             startingDofId,
                               const dftfe::uInt             startingVecId,
                               const dftfe::uInt             N,
                               dftfe::utils::deviceStream_t &streamCompute);

    template <typename ValueType1, typename ValueType2>
    void
    copyFromOverlapMatBlockToDPSPBlocks(
      const dftfe::uInt             B,
      const dftfe::uInt             D,
      const ValueType1             *overlapMatrixBlock,
      ValueType1                   *overlapMatrixBlockDP,
      ValueType2                   *overlapMatrixBlockSP,
      dftfe::utils::deviceStream_t &streamDataMove);

    template <typename ValueType1, typename ValueType2>
    void
    computeDiagQTimesX(const ValueType1 *diagValues,
                       ValueType2       *X,
                       const dftfe::uInt N,
                       const dftfe::uInt M);

    template <typename ValueType>
    void
    computeResidualDevice(const dftfe::uInt numVectors,
                          const dftfe::uInt numDofs,
                          const dftfe::uInt N,
                          const dftfe::uInt startingVecId,
                          const double     *eigenValues,
                          const ValueType  *X,
                          const ValueType  *Y,
                          double           *r);

    template <typename ValueType>
    void
    computeGeneralisedResidualDevice(const dftfe::uInt numVectors,
                                     const dftfe::uInt numDofs,
                                     const dftfe::uInt N,
                                     const dftfe::uInt startingVecId,
                                     const ValueType  *X,
                                     double           *residualSqDevice);


    template <typename ValueType>
    void
    setZero(const dftfe::uInt BVec,
            const dftfe::uInt M,
            const dftfe::uInt N,
            ValueType        *yVec,
            const dftfe::uInt startingXVecId);

  } // namespace linearAlgebraOperationsDevice
} // namespace dftfe
#endif
