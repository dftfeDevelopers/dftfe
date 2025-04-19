#ifndef solveVselfInBeinsDeviceKernels_H
#define solveVselfInBeinsDeviceKernels_H
#include <DeviceAPICalls.h>
#include <DeviceDataTypeOverloads.h>
#include <DeviceKernelLauncherConstants.h>

namespace dftfe
{
  namespace poissonDevice
  {
    void
    diagScale(const dftfe::uInt blockSize,
              const dftfe::uInt numContiguousBlocks,
              const double     *srcArray,
              const double     *scalingVector,
              double           *dstArray);
    void
    dotProductContributionBlocked(const dftfe::uInt numEntries,
                                  const double     *vec1,
                                  const double     *vec2,
                                  double           *vecTemp);

    void
    scaleBlocked(const dftfe::uInt blockSize,
                 const dftfe::uInt numContiguousBlocks,
                 double           *xArray,
                 const double     *scalingVector);

    void
    scale(const dftfe::uInt numEntries,
          double           *xArray,
          const double     *scalingVector);

    // y=alpha*x+y
    void
    daxpyBlocked(const dftfe::uInt blockSize,
                 const dftfe::uInt numContiguousBlocks,
                 const double     *x,
                 const double     *alpha,
                 double           *y);


    // y=-alpha*x+y
    void
    dmaxpyBlocked(const dftfe::uInt blockSize,
                  const dftfe::uInt numContiguousBlocks,
                  const double     *x,
                  const double     *alpha,
                  double           *y);

  } // namespace poissonDevice
} // namespace dftfe
#endif
