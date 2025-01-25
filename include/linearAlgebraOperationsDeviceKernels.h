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
    addSubspaceRotatedBlockToX(const unsigned int            BDof,
                               const unsigned int            BVec,
                               const ValueType1 *            rotatedXBlockSP,
                               ValueType2 *                  X,
                               const unsigned int            startingDofId,
                               const unsigned int            startingVecId,
                               const unsigned int            N,
                               dftfe::utils::deviceStream_t &streamCompute);

    template <typename ValueType1, typename ValueType2>
    void
    copyFromOverlapMatBlockToDPSPBlocks(
      const unsigned int            B,
      const unsigned int            D,
      const ValueType1 *            overlapMatrixBlock,
      ValueType1 *                  overlapMatrixBlockDP,
      ValueType2 *                  overlapMatrixBlockSP,
      dftfe::utils::deviceStream_t &streamDataMove);

    template <typename ValueType>
    void
    computeDiagQTimesX(const ValueType *  diagValues,
                       ValueType *        X,
                       const unsigned int N,
                       const unsigned int M);

    template <typename ValueType>
    void
    computeResidualDevice(const unsigned int numVectors,
                          const unsigned int numDofs,
                          const unsigned int N,
                          const unsigned int startingVecId,
                          const double *     eigenValues,
                          const ValueType *  X,
                          const ValueType *  Y,
                          double *           r);
    template <typename ValueType>
    void
    setZero(const unsigned int BVec,
            const unsigned int M,
            const unsigned int N,
            ValueType *        yVec,
            const unsigned int startingXVecId);

  } // namespace linearAlgebraOperationsDevice
} // namespace dftfe
#endif
