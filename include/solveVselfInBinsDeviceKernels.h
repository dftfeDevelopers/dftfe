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
    diagScale(const unsigned int blockSize,
              const unsigned int numContiguousBlocks,
              const double      *srcArray,
              const double      *scalingVector,
              double            *dstArray);
    void
    dotProductContributionBlocked(const unsigned int numEntries,
                                  const double      *vec1,
                                  const double      *vec2,
                                  double            *vecTemp);

    void
    scaleBlocked(const unsigned int blockSize,
                 const unsigned int numContiguousBlocks,
                 double            *xArray,
                 const double      *scalingVector);

    void
    scale(const unsigned int numEntries,
          double            *xArray,
          const double      *scalingVector);

    // y=alpha*x+y
    void
    daxpyBlocked(const unsigned int blockSize,
                 const unsigned int numContiguousBlocks,
                 const double      *x,
                 const double      *alpha,
                 double            *y);


    // y=-alpha*x+y
    void
    dmaxpyBlocked(const unsigned int blockSize,
                  const unsigned int numContiguousBlocks,
                  const double      *x,
                  const double      *alpha,
                  double            *y);

  } // namespace poissonDevice
} // namespace dftfe
#endif
