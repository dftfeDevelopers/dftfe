#include "solveVselfInBinsDeviceKernels.h"

namespace dftfe
{
  namespace poissonDevice
  {
    namespace
    {
      __global__ void
      diagScaleKernel(const unsigned int blockSize,
                      const unsigned int numContiguousBlocks,
                      const double *     srcArray,
                      const double *     scalingVector,
                      double *           dstArray)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId;
             index < numContiguousBlocks * blockSize;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex = index / blockSize;
            *(dstArray + index) =
              *(srcArray + index) * (*(scalingVector + blockIndex));
          }
      }

      __global__ void
      dotProductContributionBlockedKernel(const unsigned int numEntries,
                                          const double *     vec1,
                                          const double *     vec2,
                                          double *           vecTemp)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId; index < numEntries;
             index += blockDim.x * gridDim.x)
          {
            vecTemp[index] = vec1[index] * vec2[index];
          }
      }

      __global__ void
      scaleBlockedKernel(const unsigned int blockSize,
                         const unsigned int numContiguousBlocks,
                         double *           xArray,
                         const double *     scalingVector)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId;
             index < numContiguousBlocks * blockSize;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int intraBlockIndex = index % blockSize;
            *(xArray + index) *= (*(scalingVector + intraBlockIndex));
          }
      }

      __global__ void
      scaleKernel(const unsigned int numEntries,
                  double *           xArray,
                  const double *     scalingVector)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId; index < numEntries;
             index += blockDim.x * gridDim.x)
          {
            xArray[index] *= scalingVector[index];
          }
      }

      // y=alpha*x+y
      __global__ void
      daxpyBlockedKernel(const unsigned int blockSize,
                         const unsigned int numContiguousBlocks,
                         const double *     x,
                         const double *     alpha,
                         double *           y)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId;
             index < numContiguousBlocks * blockSize;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / blockSize;
            const unsigned int intraBlockIndex = index - blockIndex * blockSize;
            y[index] += alpha[intraBlockIndex] * x[index];
          }
      }


      // y=-alpha*x+y
      __global__ void
      dmaxpyBlockedKernel(const unsigned int blockSize,
                          const unsigned int numContiguousBlocks,
                          const double *     x,
                          const double *     alpha,
                          double *           y)
      {
        const unsigned int globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;

        for (unsigned int index = globalThreadId;
             index < numContiguousBlocks * blockSize;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / blockSize;
            const unsigned int intraBlockIndex = index - blockIndex * blockSize;
            y[index] += -alpha[intraBlockIndex] * x[index];
          }
      }
    } // namespace
    void
    diagScale(const unsigned int blockSize,
              const unsigned int numContiguousBlocks,
              const double *     srcArray,
              const double *     scalingVector,
              double *           dstArray)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      diagScaleKernel<<<(blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                          dftfe::utils::DEVICE_BLOCK_SIZE * numContiguousBlocks,
                        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        blockSize, numContiguousBlocks, srcArray, scalingVector, dstArray);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(diagScaleKernel,
                         (blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                           dftfe::utils::DEVICE_BLOCK_SIZE *
                           numContiguousBlocks,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         blockSize,
                         numContiguousBlocks,
                         srcArray,
                         scalingVector,
                         dstArray);
#endif
    }
    void
    dotProductContributionBlocked(const unsigned int numEntries,
                                  const double *     vec1,
                                  const double *     vec2,
                                  double *           vecTemp)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      dotProductContributionBlockedKernel<<<
        (numEntries + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(numEntries, vec1, vec2, vecTemp);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(dotProductContributionBlockedKernel,
                         (numEntries + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                           dftfe::utils::DEVICE_BLOCK_SIZE,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         numEntries,
                         vec1,
                         vec2,
                         vecTemp);
#endif
    }

    void
    scaleBlocked(const unsigned int blockSize,
                 const unsigned int numContiguousBlocks,
                 double *           xArray,
                 const double *     scalingVector)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      scaleBlockedKernel<<<(blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                             dftfe::utils::DEVICE_BLOCK_SIZE *
                             numContiguousBlocks,
                           dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        blockSize, numContiguousBlocks, xArray, scalingVector);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(scaleBlockedKernel,
                         (blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                           dftfe::utils::DEVICE_BLOCK_SIZE *
                           numContiguousBlocks,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         blockSize,
                         numContiguousBlocks,
                         xArray,
                         scalingVector);
#endif
    }

    void
    scale(const unsigned int numEntries,
          double *           xArray,
          const double *     scalingVector)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      scaleKernel<<<(numEntries + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                      dftfe::utils::DEVICE_BLOCK_SIZE,
                    dftfe::utils::DEVICE_BLOCK_SIZE>>>(numEntries,
                                                       xArray,
                                                       scalingVector);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(scaleKernel,
                         (numEntries + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                           dftfe::utils::DEVICE_BLOCK_SIZE,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         numEntries,
                         xArray,
                         scalingVector);
#endif
    }

    // y=alpha*x+y
    void
    daxpyBlocked(const unsigned int blockSize,
                 const unsigned int numContiguousBlocks,
                 const double *     x,
                 const double *     alpha,
                 double *           y)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      daxpyBlockedKernel<<<(blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                             dftfe::utils::DEVICE_BLOCK_SIZE *
                             numContiguousBlocks,
                           dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        blockSize, numContiguousBlocks, x, alpha, y);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(daxpyBlockedKernel,
                         (blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                           dftfe::utils::DEVICE_BLOCK_SIZE *
                           numContiguousBlocks,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         blockSize,
                         numContiguousBlocks,
                         x,
                         alpha,
                         y);
#endif
    }


    // y=-alpha*x+y
    void
    dmaxpyBlocked(const unsigned int blockSize,
                  const unsigned int numContiguousBlocks,
                  const double *     x,
                  const double *     alpha,
                  double *           y)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      dmaxpyBlockedKernel<<<
        (blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
          dftfe::utils::DEVICE_BLOCK_SIZE * numContiguousBlocks,
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        blockSize, numContiguousBlocks, x, alpha, y);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(dmaxpyBlockedKernel,
                         (blockSize + (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                           dftfe::utils::DEVICE_BLOCK_SIZE *
                           numContiguousBlocks,
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         blockSize,
                         numContiguousBlocks,
                         x,
                         alpha,
                         y);
#endif
    }
  } // namespace poissonDevice
} // namespace dftfe
