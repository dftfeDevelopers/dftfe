#include "solveVselfInBinsDeviceKernels.h"

namespace dftfe
{
  namespace poissonDevice
  {
    namespace
    {

      DFTFE_CREATE_KERNEL(
        void,
        diagScaleKernel,
        {
          for (dftfe::uInt index = globalThreadId;
               index < numContiguousBlocks * blockSize;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockIndex = index / blockSize;
              *(dstArray + index) =
                *(srcArray + index) * (*(scalingVector + blockIndex));
            }
        },
        const dftfe::uInt blockSize,
        const dftfe::uInt numContiguousBlocks,
        const double     *srcArray,
        const double     *scalingVector,
        double           *dstArray);



      DFTFE_CREATE_KERNEL(
        void,
        dotProductContributionBlockedKernel,
        {
          for (dftfe::uInt index = globalThreadId; index < numEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              vecTemp[index] = vec1[index] * vec2[index];
            }
        },
        const dftfe::uInt numEntries,
        const double     *vec1,
        const double     *vec2,
        double           *vecTemp);



      DFTFE_CREATE_KERNEL(
        void,
        scaleBlockedKernel,
        {
          for (dftfe::uInt index = globalThreadId;
               index < numContiguousBlocks * blockSize;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt intraBlockIndex = index % blockSize;
              *(xArray + index) *= (*(scalingVector + intraBlockIndex));
            }
        },
        const dftfe::uInt blockSize,
        const dftfe::uInt numContiguousBlocks,
        double           *xArray,
        const double     *scalingVector);



      DFTFE_CREATE_KERNEL(
        void,
        scaleKernel,
        {
          for (dftfe::uInt index = globalThreadId; index < numEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              xArray[index] *= scalingVector[index];
            }
        },
        const dftfe::uInt numEntries,
        double           *xArray,
        const double     *scalingVector);


      // y=alpha*x+y

      DFTFE_CREATE_KERNEL(
        void,
        daxpyBlockedKernel,
        {
          for (dftfe::uInt index = globalThreadId;
               index < numContiguousBlocks * blockSize;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockIndex = index / blockSize;
              const dftfe::uInt intraBlockIndex =
                index - blockIndex * blockSize;
              y[index] += alpha[intraBlockIndex] * x[index];
            }
        },
        const dftfe::uInt blockSize,
        const dftfe::uInt numContiguousBlocks,
        const double     *x,
        const double     *alpha,
        double           *y);



      // y=-alpha*x+y

      DFTFE_CREATE_KERNEL(
        void,
        dmaxpyBlockedKernel,
        {
          for (dftfe::uInt index = globalThreadId;
               index < numContiguousBlocks * blockSize;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockIndex = index / blockSize;
              const dftfe::uInt intraBlockIndex =
                index - blockIndex * blockSize;
              y[index] += -alpha[intraBlockIndex] * x[index];
            }
        },
        const dftfe::uInt blockSize,
        const dftfe::uInt numContiguousBlocks,
        const double     *x,
        const double     *alpha,
        double           *y);

    } // namespace
    void
    diagScale(const dftfe::uInt blockSize,
              const dftfe::uInt numContiguousBlocks,
              const double     *srcArray,
              const double     *scalingVector,
              double           *dstArray)
    {
      DFTFE_LAUNCH_KERNEL(diagScaleKernel,
                          (blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                            dftfe::utils::DEVICE_BLOCK_SIZE *
                            numContiguousBlocks,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::defaultStream,
                          blockSize,
                          numContiguousBlocks,
                          srcArray,
                          scalingVector,
                          dstArray);
    }
    void
    dotProductContributionBlocked(const dftfe::uInt numEntries,
                                  const double     *vec1,
                                  const double     *vec2,
                                  double           *vecTemp)
    {
      DFTFE_LAUNCH_KERNEL(dotProductContributionBlockedKernel,
                          (numEntries + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::defaultStream,
                          numEntries,
                          vec1,
                          vec2,
                          vecTemp);
    }

    void
    scaleBlocked(const dftfe::uInt blockSize,
                 const dftfe::uInt numContiguousBlocks,
                 double           *xArray,
                 const double     *scalingVector)
    {
      DFTFE_LAUNCH_KERNEL(scaleBlockedKernel,
                          (blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                            dftfe::utils::DEVICE_BLOCK_SIZE *
                            numContiguousBlocks,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::defaultStream,
                          blockSize,
                          numContiguousBlocks,
                          xArray,
                          scalingVector);
    }

    void
    scale(const dftfe::uInt numEntries,
          double           *xArray,
          const double     *scalingVector)
    {
      DFTFE_LAUNCH_KERNEL(scaleKernel,
                          (numEntries + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::defaultStream,
                          numEntries,
                          xArray,
                          scalingVector);
    }

    // y=alpha*x+y
    void
    daxpyBlocked(const dftfe::uInt blockSize,
                 const dftfe::uInt numContiguousBlocks,
                 const double     *x,
                 const double     *alpha,
                 double           *y)
    {
      DFTFE_LAUNCH_KERNEL(daxpyBlockedKernel,
                          (blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                            dftfe::utils::DEVICE_BLOCK_SIZE *
                            numContiguousBlocks,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::defaultStream,
                          blockSize,
                          numContiguousBlocks,
                          x,
                          alpha,
                          y);
    }


    // y=-alpha*x+y
    void
    dmaxpyBlocked(const dftfe::uInt blockSize,
                  const dftfe::uInt numContiguousBlocks,
                  const double     *x,
                  const double     *alpha,
                  double           *y)
    {
      DFTFE_LAUNCH_KERNEL(dmaxpyBlockedKernel,
                          (blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                            dftfe::utils::DEVICE_BLOCK_SIZE *
                            numContiguousBlocks,
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::utils::defaultStream,
                          blockSize,
                          numContiguousBlocks,
                          x,
                          alpha,
                          y);
    }
  } // namespace poissonDevice
} // namespace dftfe
