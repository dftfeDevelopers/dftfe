#include "constraintMatrixInfoDeviceKernels.h"

namespace dftfe
{
  // Declare dftUtils functions
  namespace dftUtils
  {
    namespace
    {
      __global__ void
      distributeKernel(
        const unsigned int  contiguousBlockSize,
        double *            xVec,
        const unsigned int *constraintLocalRowIdsUnflattened,
        const unsigned int  numConstraints,
        const unsigned int *constraintRowSizes,
        const unsigned int *constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened,
        const double *      inhomogenities)
      {
        const std::size_t globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const std::size_t numberEntries = numConstraints * contiguousBlockSize;

        for (std::size_t index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const std::size_t xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            xVec[xVecStartingIdRow + intraBlockIndex] =
              inhomogenities[blockIndex];
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const std::size_t xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                xVec[xVecStartingIdRow + intraBlockIndex] +=
                  constraintColumnValuesAllRowsUnflattened
                    [startingColumnNumber + i] *
                  xVec[xVecStartingIdColumn + intraBlockIndex];
              }
          }
      }


      __global__ void
      distributeKernel(
        const unsigned int  contiguousBlockSize,
        float *             xVec,
        const unsigned int *constraintLocalRowIdsUnflattened,
        const unsigned int  numConstraints,
        const unsigned int *constraintRowSizes,
        const unsigned int *constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened,
        const double *      inhomogenities)
      {
        const std::size_t globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const std::size_t numberEntries = numConstraints * contiguousBlockSize;

        for (std::size_t index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const std::size_t xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            xVec[xVecStartingIdRow + intraBlockIndex] =
              inhomogenities[blockIndex];
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const std::size_t xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                xVec[xVecStartingIdRow + intraBlockIndex] +=
                  constraintColumnValuesAllRowsUnflattened
                    [startingColumnNumber + i] *
                  xVec[xVecStartingIdColumn + intraBlockIndex];
              }
          }
      }

      __global__ void
      scaleConstraintsKernel(
        const double *      xVec,
        const unsigned int *constraintLocalRowIdsUnflattened,
        const unsigned int  numConstraints,
        const unsigned int *constraintRowSizes,
        const unsigned int *constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        double *            constraintColumnValuesAllRowsUnflattened)
      {
        const std::size_t globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const std::size_t numberEntries = numConstraints;

        for (std::size_t index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[index];
            const unsigned int numberColumns = constraintRowSizes[index];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[index];
            const std::size_t xVecStartingIdRow = constrainedRowId;
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                constraintColumnValuesAllRowsUnflattened[startingColumnNumber +
                                                         i] *=
                  xVec[constrainedColumnId];
              }
          }
      }


      __global__ void
      distributeKernel(
        const unsigned int                 contiguousBlockSize,
        dftfe::utils::deviceDoubleComplex *xVec,
        const unsigned int *               constraintLocalRowIdsUnflattened,
        const unsigned int                 numConstraints,
        const unsigned int *               constraintRowSizes,
        const unsigned int *               constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened,
        const double *      inhomogenities)
      {
        const std::size_t globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const std::size_t numberEntries = numConstraints * contiguousBlockSize;

        for (std::size_t index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const std::size_t xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            dftfe::utils::copyValue(xVec + xVecStartingIdRow + intraBlockIndex,
                                    inhomogenities[blockIndex]);
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const std::size_t xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                dftfe::utils::copyValue(
                  xVec + xVecStartingIdRow + intraBlockIndex,
                  dftfe::utils::add(
                    xVec[xVecStartingIdRow + intraBlockIndex],
                    dftfe::utils::makeComplex(
                      xVec[xVecStartingIdColumn + intraBlockIndex].x *
                        constraintColumnValuesAllRowsUnflattened
                          [startingColumnNumber + i],
                      xVec[xVecStartingIdColumn + intraBlockIndex].y *
                        constraintColumnValuesAllRowsUnflattened
                          [startingColumnNumber + i])));
              }
          }
      }


      __global__ void
      distributeKernel(
        const unsigned int                contiguousBlockSize,
        dftfe::utils::deviceFloatComplex *xVec,
        const unsigned int *              constraintLocalRowIdsUnflattened,
        const unsigned int                numConstraints,
        const unsigned int *              constraintRowSizes,
        const unsigned int *              constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened,
        const double *      inhomogenities)
      {
        const std::size_t globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const std::size_t numberEntries = numConstraints * contiguousBlockSize;

        for (std::size_t index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const std::size_t xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            dftfe::utils::copyValue(xVec + xVecStartingIdRow + intraBlockIndex,
                                    inhomogenities[blockIndex]);
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const std::size_t xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                dftfe::utils::copyValue(
                  xVec + xVecStartingIdRow + intraBlockIndex,
                  dftfe::utils::add(
                    xVec[xVecStartingIdRow + intraBlockIndex],
                    dftfe::utils::makeComplex(
                      xVec[xVecStartingIdColumn + intraBlockIndex].x *
                        constraintColumnValuesAllRowsUnflattened
                          [startingColumnNumber + i],
                      xVec[xVecStartingIdColumn + intraBlockIndex].y *
                        constraintColumnValuesAllRowsUnflattened
                          [startingColumnNumber + i])));
              }
          }
      }

      __global__ void
      distributeSlaveToMasterKernelAtomicAdd(
        const unsigned int  contiguousBlockSize,
        double *            xVec,
        const unsigned int *constraintLocalRowIdsUnflattened,
        const unsigned int  numConstraints,
        const unsigned int *constraintRowSizes,
        const unsigned int *constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened)
      {
        const std::size_t globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const std::size_t numberEntries = numConstraints * contiguousBlockSize;

        for (std::size_t index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const std::size_t xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const std::size_t xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                atomicAdd(&(xVec[xVecStartingIdColumn + intraBlockIndex]),
                          constraintColumnValuesAllRowsUnflattened
                              [startingColumnNumber + i] *
                            xVec[xVecStartingIdRow + intraBlockIndex]);
              }
            xVec[xVecStartingIdRow + intraBlockIndex] = 0.0;
          }
      }


      __global__ void
      distributeSlaveToMasterKernelAtomicAdd(
        const unsigned int                 contiguousBlockSize,
        dftfe::utils::deviceDoubleComplex *xVec,
        const unsigned int *               constraintLocalRowIdsUnflattened,
        const unsigned int                 numConstraints,
        const unsigned int *               constraintRowSizes,
        const unsigned int *               constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened)
      {
        const std::size_t globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const std::size_t numberEntries = numConstraints * contiguousBlockSize;

        for (std::size_t index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const std::size_t xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const std::size_t xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                const dftfe::utils::deviceDoubleComplex tempComplval =
                  dftfe::utils::mult(constraintColumnValuesAllRowsUnflattened
                                       [startingColumnNumber + i],
                                     xVec[xVecStartingIdRow + intraBlockIndex]);
                atomicAdd(&(xVec[xVecStartingIdColumn + intraBlockIndex].x),
                          tempComplval.x);
                atomicAdd(&(xVec[xVecStartingIdColumn + intraBlockIndex].y),
                          tempComplval.y);
              }
            xVec[xVecStartingIdRow + intraBlockIndex].x = 0.0;
            xVec[xVecStartingIdRow + intraBlockIndex].y = 0.0;
          }
      }

      __global__ void
      distributeSlaveToMasterKernelAtomicAdd(
        const unsigned int  contiguousBlockSize,
        float *             xVec,
        const unsigned int *constraintLocalRowIdsUnflattened,
        const unsigned int  numConstraints,
        const unsigned int *constraintRowSizes,
        const unsigned int *constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened)
      {
        const std::size_t globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const std::size_t numberEntries = numConstraints * contiguousBlockSize;

        for (std::size_t index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const std::size_t xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const std::size_t xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                atomicAdd(&(xVec[xVecStartingIdColumn + intraBlockIndex]),
                          constraintColumnValuesAllRowsUnflattened
                              [startingColumnNumber + i] *
                            xVec[xVecStartingIdRow + intraBlockIndex]);
              }
            xVec[xVecStartingIdRow + intraBlockIndex] = 0.0;
          }
      }


      __global__ void
      distributeSlaveToMasterKernelAtomicAdd(
        const unsigned int                contiguousBlockSize,
        dftfe::utils::deviceFloatComplex *xVec,
        const unsigned int *              constraintLocalRowIdsUnflattened,
        const unsigned int                numConstraints,
        const unsigned int *              constraintRowSizes,
        const unsigned int *              constraintRowSizesAccumulated,
        const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
        const double *      constraintColumnValuesAllRowsUnflattened)
      {
        const std::size_t globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const std::size_t numberEntries = numConstraints * contiguousBlockSize;

        for (std::size_t index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            const unsigned int constrainedRowId =
              constraintLocalRowIdsUnflattened[blockIndex];
            const unsigned int numberColumns = constraintRowSizes[blockIndex];
            const unsigned int startingColumnNumber =
              constraintRowSizesAccumulated[blockIndex];
            const std::size_t xVecStartingIdRow =
              constrainedRowId * contiguousBlockSize;
            for (unsigned int i = 0; i < numberColumns; ++i)
              {
                const unsigned int constrainedColumnId =
                  constraintLocalColumnIdsAllRowsUnflattened
                    [startingColumnNumber + i];
                const std::size_t xVecStartingIdColumn =
                  constrainedColumnId * contiguousBlockSize;
                const dftfe::utils::deviceDoubleComplex tempComplval =
                  dftfe::utils::mult(constraintColumnValuesAllRowsUnflattened
                                       [startingColumnNumber + i],
                                     xVec[xVecStartingIdRow + intraBlockIndex]);
                atomicAdd(&(xVec[xVecStartingIdColumn + intraBlockIndex].x),
                          tempComplval.x);
                atomicAdd(&(xVec[xVecStartingIdColumn + intraBlockIndex].y),
                          tempComplval.y);
              }
            xVec[xVecStartingIdRow + intraBlockIndex].x = 0.0;
            xVec[xVecStartingIdRow + intraBlockIndex].y = 0.0;
          }
      }


      __global__ void
      setzeroKernel(const unsigned int  contiguousBlockSize,
                    double *            xVec,
                    const unsigned int *constraintLocalRowIdsUnflattened,
                    const unsigned int  numConstraints)
      {
        const std::size_t globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const std::size_t numberEntries = numConstraints * contiguousBlockSize;

        for (std::size_t index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            xVec[constraintLocalRowIdsUnflattened[blockIndex] *
                   contiguousBlockSize +
                 intraBlockIndex]              = 0;
          }
      }

      __global__ void
      setzeroKernel(const unsigned int  contiguousBlockSize,
                    float *             xVec,
                    const unsigned int *constraintLocalRowIdsUnflattened,
                    const unsigned int  numConstraints)
      {
        const std::size_t globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const std::size_t numberEntries = numConstraints * contiguousBlockSize;

        for (std::size_t index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            xVec[constraintLocalRowIdsUnflattened[blockIndex] *
                   contiguousBlockSize +
                 intraBlockIndex]              = 0;
          }
      }

      __global__ void
      setzeroKernel(const unsigned int                 contiguousBlockSize,
                    dftfe::utils::deviceDoubleComplex *xVec,
                    const unsigned int *constraintLocalRowIdsUnflattened,
                    const unsigned int  numConstraints)
      {
        const std::size_t globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const std::size_t numberEntries = numConstraints * contiguousBlockSize;

        for (std::size_t index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            dftfe::utils::copyValue(
              xVec +
                constraintLocalRowIdsUnflattened[blockIndex] *
                  contiguousBlockSize +
                intraBlockIndex,
              0.0);
          }
      }


      __global__ void
      setzeroKernel(const unsigned int                contiguousBlockSize,
                    dftfe::utils::deviceFloatComplex *xVec,
                    const unsigned int *constraintLocalRowIdsUnflattened,
                    const unsigned int  numConstraints)
      {
        const std::size_t globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const std::size_t numberEntries = numConstraints * contiguousBlockSize;

        for (std::size_t index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            const unsigned int blockIndex      = index / contiguousBlockSize;
            const unsigned int intraBlockIndex = index % contiguousBlockSize;
            dftfe::utils::copyValue(
              xVec +
                constraintLocalRowIdsUnflattened[blockIndex] *
                  contiguousBlockSize +
                intraBlockIndex,
              0.0);
          }
      }
    } // namespace
    template <typename ValueType>
    void
    distributeDevice(
      const unsigned int  contiguousBlockSize,
      ValueType *         xVec,
      const unsigned int *constraintLocalRowIdsUnflattened,
      const unsigned int  numConstraints,
      const unsigned int *constraintRowSizes,
      const unsigned int *constraintRowSizesAccumulated,
      const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
      const double *      constraintColumnValuesAllRowsUnflattened,
      const double *      inhomogenities)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      distributeKernel<<<min((contiguousBlockSize * numConstraints +
                              (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                               dftfe::utils::DEVICE_BLOCK_SIZE,
                             30000),
                         dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        contiguousBlockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(xVec),
        constraintLocalRowIdsUnflattened,
        numConstraints,
        constraintRowSizes,
        constraintRowSizesAccumulated,
        constraintLocalColumnIdsAllRowsUnflattened,
        constraintColumnValuesAllRowsUnflattened,
        inhomogenities);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(distributeKernel,
                         min((contiguousBlockSize * numConstraints +
                              (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                               dftfe::utils::DEVICE_BLOCK_SIZE,
                             30000),
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         contiguousBlockSize,
                         dftfe::utils::makeDataTypeDeviceCompatible(xVec),
                         constraintLocalRowIdsUnflattened,
                         numConstraints,
                         constraintRowSizes,
                         constraintRowSizesAccumulated,
                         constraintLocalColumnIdsAllRowsUnflattened,
                         constraintColumnValuesAllRowsUnflattened,
                         inhomogenities);
#endif
    }

    template <typename ValueType>
    void
    distributeSlaveToMasterAtomicAddDevice(
      const unsigned int  contiguousBlockSize,
      ValueType *         xVec,
      const unsigned int *constraintLocalRowIdsUnflattened,
      const unsigned int  numConstraints,
      const unsigned int *constraintRowSizes,
      const unsigned int *constraintRowSizesAccumulated,
      const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
      const double *      constraintColumnValuesAllRowsUnflattened)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      distributeSlaveToMasterKernelAtomicAdd<<<
        min((contiguousBlockSize * numConstraints +
             (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
              dftfe::utils::DEVICE_BLOCK_SIZE,
            30000),
        dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        contiguousBlockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(xVec),
        constraintLocalRowIdsUnflattened,
        numConstraints,
        constraintRowSizes,
        constraintRowSizesAccumulated,
        constraintLocalColumnIdsAllRowsUnflattened,
        constraintColumnValuesAllRowsUnflattened);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(distributeSlaveToMasterKernelAtomicAdd,
                         min((contiguousBlockSize * numConstraints +
                              (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                               dftfe::utils::DEVICE_BLOCK_SIZE,
                             30000),
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         contiguousBlockSize,
                         dftfe::utils::makeDataTypeDeviceCompatible(xVec),
                         constraintLocalRowIdsUnflattened,
                         numConstraints,
                         constraintRowSizes,
                         constraintRowSizesAccumulated,
                         constraintLocalColumnIdsAllRowsUnflattened,
                         constraintColumnValuesAllRowsUnflattened);
#endif
    }
    template <typename ValueType>
    void
    setzeroDevice(const unsigned int  contiguousBlockSize,
                  ValueType *         xVec,
                  const unsigned int *constraintLocalRowIdsUnflattened,
                  const unsigned int  numConstraints)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      setzeroKernel<<<min((contiguousBlockSize * numConstraints +
                           (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                            dftfe::utils::DEVICE_BLOCK_SIZE,
                          30000),
                      dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        contiguousBlockSize,
        dftfe::utils::makeDataTypeDeviceCompatible(xVec),
        constraintLocalRowIdsUnflattened,
        numConstraints);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(setzeroKernel,
                         min((contiguousBlockSize * numConstraints +
                              (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                               dftfe::utils::DEVICE_BLOCK_SIZE,
                             30000),
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         contiguousBlockSize,
                         dftfe::utils::makeDataTypeDeviceCompatible(xVec),
                         constraintLocalRowIdsUnflattened,
                         numConstraints);
#endif
    }

    void
    scaleConstraintsDevice(
      const double *      xVec,
      const unsigned int *constraintLocalRowIdsUnflattened,
      const unsigned int  numConstraints,
      const unsigned int *constraintRowSizes,
      const unsigned int *constraintRowSizesAccumulated,
      const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
      double *            constraintColumnValuesAllRowsUnflattened)
    {
#ifdef DFTFE_WITH_DEVICE_LANG_CUDA
      scaleConstraintsKernel<<<min((numConstraints +
                                    (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                     dftfe::utils::DEVICE_BLOCK_SIZE,
                                   30000),
                               dftfe::utils::DEVICE_BLOCK_SIZE>>>(
        dftfe::utils::makeDataTypeDeviceCompatible(xVec),
        constraintLocalRowIdsUnflattened,
        numConstraints,
        constraintRowSizes,
        constraintRowSizesAccumulated,
        constraintLocalColumnIdsAllRowsUnflattened,
        constraintColumnValuesAllRowsUnflattened);
#elif DFTFE_WITH_DEVICE_LANG_HIP
      hipLaunchKernelGGL(scaleConstraintsKernel,
                         min((numConstraints +
                              (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                               dftfe::utils::DEVICE_BLOCK_SIZE,
                             30000),
                         dftfe::utils::DEVICE_BLOCK_SIZE,
                         0,
                         0,
                         dftfe::utils::makeDataTypeDeviceCompatible(xVec),
                         constraintLocalRowIdsUnflattened,
                         numConstraints,
                         constraintRowSizes,
                         constraintRowSizesAccumulated,
                         constraintLocalColumnIdsAllRowsUnflattened,
                         constraintColumnValuesAllRowsUnflattened);
#endif
    }
    template void
    distributeDevice(
      const unsigned int  contiguousBlockSize,
      double *            xVec,
      const unsigned int *constraintLocalRowIdsUnflattened,
      const unsigned int  numConstraints,
      const unsigned int *constraintRowSizes,
      const unsigned int *constraintRowSizesAccumulated,
      const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
      const double *      constraintColumnValuesAllRowsUnflattened,
      const double *      inhomogenities);
    template void
    distributeDevice(
      const unsigned int  contiguousBlockSize,
      float *             xVec,
      const unsigned int *constraintLocalRowIdsUnflattened,
      const unsigned int  numConstraints,
      const unsigned int *constraintRowSizes,
      const unsigned int *constraintRowSizesAccumulated,
      const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
      const double *      constraintColumnValuesAllRowsUnflattened,
      const double *      inhomogenities);
    template void
    distributeDevice(
      const unsigned int    contiguousBlockSize,
      std::complex<double> *xVec,
      const unsigned int *  constraintLocalRowIdsUnflattened,
      const unsigned int    numConstraints,
      const unsigned int *  constraintRowSizes,
      const unsigned int *  constraintRowSizesAccumulated,
      const unsigned int *  constraintLocalColumnIdsAllRowsUnflattened,
      const double *        constraintColumnValuesAllRowsUnflattened,
      const double *        inhomogenities);
    template void
    distributeDevice(
      const unsigned int   contiguousBlockSize,
      std::complex<float> *xVec,
      const unsigned int * constraintLocalRowIdsUnflattened,
      const unsigned int   numConstraints,
      const unsigned int * constraintRowSizes,
      const unsigned int * constraintRowSizesAccumulated,
      const unsigned int * constraintLocalColumnIdsAllRowsUnflattened,
      const double *       constraintColumnValuesAllRowsUnflattened,
      const double *       inhomogenities);
    template void
    distributeSlaveToMasterAtomicAddDevice(
      const unsigned int  contiguousBlockSize,
      double *            xVec,
      const unsigned int *constraintLocalRowIdsUnflattened,
      const unsigned int  numConstraints,
      const unsigned int *constraintRowSizes,
      const unsigned int *constraintRowSizesAccumulated,
      const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
      const double *      constraintColumnValuesAllRowsUnflattened);
    template void
    distributeSlaveToMasterAtomicAddDevice(
      const unsigned int  contiguousBlockSize,
      float *             xVec,
      const unsigned int *constraintLocalRowIdsUnflattened,
      const unsigned int  numConstraints,
      const unsigned int *constraintRowSizes,
      const unsigned int *constraintRowSizesAccumulated,
      const unsigned int *constraintLocalColumnIdsAllRowsUnflattened,
      const double *      constraintColumnValuesAllRowsUnflattened);
    template void
    distributeSlaveToMasterAtomicAddDevice(
      const unsigned int    contiguousBlockSize,
      std::complex<double> *xVec,
      const unsigned int *  constraintLocalRowIdsUnflattened,
      const unsigned int    numConstraints,
      const unsigned int *  constraintRowSizes,
      const unsigned int *  constraintRowSizesAccumulated,
      const unsigned int *  constraintLocalColumnIdsAllRowsUnflattened,
      const double *        constraintColumnValuesAllRowsUnflattened);
    template void
    distributeSlaveToMasterAtomicAddDevice(
      const unsigned int   contiguousBlockSize,
      std::complex<float> *xVec,
      const unsigned int * constraintLocalRowIdsUnflattened,
      const unsigned int   numConstraints,
      const unsigned int * constraintRowSizes,
      const unsigned int * constraintRowSizesAccumulated,
      const unsigned int * constraintLocalColumnIdsAllRowsUnflattened,
      const double *       constraintColumnValuesAllRowsUnflattened);
    template void
    setzeroDevice(const unsigned int  contiguousBlockSize,
                  double *            xVec,
                  const unsigned int *constraintLocalRowIdsUnflattened,
                  const unsigned int  numConstraints);
    template void
    setzeroDevice(const unsigned int  contiguousBlockSize,
                  float *             xVec,
                  const unsigned int *constraintLocalRowIdsUnflattened,
                  const unsigned int  numConstraints);
    template void
    setzeroDevice(const unsigned int    contiguousBlockSize,
                  std::complex<double> *xVec,
                  const unsigned int *  constraintLocalRowIdsUnflattened,
                  const unsigned int    numConstraints);
    template void
    setzeroDevice(const unsigned int   contiguousBlockSize,
                  std::complex<float> *xVec,
                  const unsigned int * constraintLocalRowIdsUnflattened,
                  const unsigned int   numConstraints);

  } // namespace dftUtils
} // namespace dftfe
