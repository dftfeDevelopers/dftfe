#include "constraintMatrixInfoDeviceKernels.h"
#include <BLASWrapper.h>

namespace dftfe
{
  // Declare dftUtils functions
  namespace dftUtils
  {
    namespace
    {

      DFTFE_CREATE_KERNEL(
        void,
        distributeKernel,
        {
          const std::size_t numberEntries =
            numConstraints * contiguousBlockSize;

          for (std::size_t index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockIndex      = index / contiguousBlockSize;
              const dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
              const dftfe::uInt constrainedRowId =
                constraintLocalRowIdsUnflattened[blockIndex];
              const dftfe::uInt numberColumns = constraintRowSizes[blockIndex];
              const dftfe::uInt startingColumnNumber =
                constraintRowSizesAccumulated[blockIndex];
              const std::size_t xVecStartingIdRow =
                constrainedRowId * contiguousBlockSize;
              xVec[xVecStartingIdRow + intraBlockIndex] =
                inhomogenities[blockIndex];
              for (dftfe::uInt i = 0; i < numberColumns; ++i)
                {
                  const dftfe::uInt constrainedColumnId =
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
        },
        const dftfe::uInt  contiguousBlockSize,
        double            *xVec,
        const dftfe::uInt *constraintLocalRowIdsUnflattened,
        const dftfe::uInt  numConstraints,
        const dftfe::uInt *constraintRowSizes,
        const dftfe::uInt *constraintRowSizesAccumulated,
        const dftfe::uInt *constraintLocalColumnIdsAllRowsUnflattened,
        const double      *constraintColumnValuesAllRowsUnflattened,
        const double      *inhomogenities);



      DFTFE_CREATE_KERNEL(
        void,
        distributeKernel,
        {
          const std::size_t numberEntries =
            numConstraints * contiguousBlockSize;

          for (std::size_t index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockIndex      = index / contiguousBlockSize;
              const dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
              const dftfe::uInt constrainedRowId =
                constraintLocalRowIdsUnflattened[blockIndex];
              const dftfe::uInt numberColumns = constraintRowSizes[blockIndex];
              const dftfe::uInt startingColumnNumber =
                constraintRowSizesAccumulated[blockIndex];
              const std::size_t xVecStartingIdRow =
                constrainedRowId * contiguousBlockSize;
              xVec[xVecStartingIdRow + intraBlockIndex] =
                inhomogenities[blockIndex];
              for (dftfe::uInt i = 0; i < numberColumns; ++i)
                {
                  const dftfe::uInt constrainedColumnId =
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
        },
        const dftfe::uInt  contiguousBlockSize,
        float             *xVec,
        const dftfe::uInt *constraintLocalRowIdsUnflattened,
        const dftfe::uInt  numConstraints,
        const dftfe::uInt *constraintRowSizes,
        const dftfe::uInt *constraintRowSizesAccumulated,
        const dftfe::uInt *constraintLocalColumnIdsAllRowsUnflattened,
        const double      *constraintColumnValuesAllRowsUnflattened,
        const double      *inhomogenities);



      DFTFE_CREATE_KERNEL(
        void,
        scaleConstraintsKernel,
        {
          const std::size_t numberEntries = numConstraints;

          for (std::size_t index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt constrainedRowId =
                constraintLocalRowIdsUnflattened[index];
              const dftfe::uInt numberColumns = constraintRowSizes[index];
              const dftfe::uInt startingColumnNumber =
                constraintRowSizesAccumulated[index];
              const std::size_t xVecStartingIdRow = constrainedRowId;
              for (dftfe::uInt i = 0; i < numberColumns; ++i)
                {
                  const dftfe::uInt constrainedColumnId =
                    constraintLocalColumnIdsAllRowsUnflattened
                      [startingColumnNumber + i];
                  constraintColumnValuesAllRowsUnflattened
                    [startingColumnNumber + i] *= xVec[constrainedColumnId];
                }
            }
        },
        const double      *xVec,
        const dftfe::uInt *constraintLocalRowIdsUnflattened,
        const dftfe::uInt  numConstraints,
        const dftfe::uInt *constraintRowSizes,
        const dftfe::uInt *constraintRowSizesAccumulated,
        const dftfe::uInt *constraintLocalColumnIdsAllRowsUnflattened,
        double            *constraintColumnValuesAllRowsUnflattened);



      DFTFE_CREATE_KERNEL(
        void,
        distributeKernel,
        {
          const std::size_t numberEntries =
            numConstraints * contiguousBlockSize;

          for (std::size_t index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockIndex      = index / contiguousBlockSize;
              const dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
              const dftfe::uInt constrainedRowId =
                constraintLocalRowIdsUnflattened[blockIndex];
              const dftfe::uInt numberColumns = constraintRowSizes[blockIndex];
              const dftfe::uInt startingColumnNumber =
                constraintRowSizesAccumulated[blockIndex];
              const std::size_t xVecStartingIdRow =
                constrainedRowId * contiguousBlockSize;
              dftfe::utils::copyValue(xVec + xVecStartingIdRow +
                                        intraBlockIndex,
                                      inhomogenities[blockIndex]);
              for (dftfe::uInt i = 0; i < numberColumns; ++i)
                {
                  const dftfe::uInt constrainedColumnId =
                    constraintLocalColumnIdsAllRowsUnflattened
                      [startingColumnNumber + i];
                  const std::size_t xVecStartingIdColumn =
                    constrainedColumnId * contiguousBlockSize;
                  dftfe::utils::copyValue(
                    xVec + xVecStartingIdRow + intraBlockIndex,
                    dftfe::utils::add(
                      xVec[xVecStartingIdRow + intraBlockIndex],
                      dftfe::utils::makeComplex(
                        dftfe::utils::realPartDevice(
                          xVec[xVecStartingIdColumn + intraBlockIndex]) *
                          constraintColumnValuesAllRowsUnflattened
                            [startingColumnNumber + i],
                        dftfe::utils::imagPartDevice(
                          xVec[xVecStartingIdColumn + intraBlockIndex]) *
                          constraintColumnValuesAllRowsUnflattened
                            [startingColumnNumber + i])));
                }
            }
        },
        const dftfe::uInt                  contiguousBlockSize,
        dftfe::utils::deviceDoubleComplex *xVec,
        const dftfe::uInt                 *constraintLocalRowIdsUnflattened,
        const dftfe::uInt                  numConstraints,
        const dftfe::uInt                 *constraintRowSizes,
        const dftfe::uInt                 *constraintRowSizesAccumulated,
        const dftfe::uInt *constraintLocalColumnIdsAllRowsUnflattened,
        const double      *constraintColumnValuesAllRowsUnflattened,
        const double      *inhomogenities);



      DFTFE_CREATE_KERNEL(
        void,
        distributeKernel,
        {
          const std::size_t numberEntries =
            numConstraints * contiguousBlockSize;

          for (std::size_t index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockIndex      = index / contiguousBlockSize;
              const dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
              const dftfe::uInt constrainedRowId =
                constraintLocalRowIdsUnflattened[blockIndex];
              const dftfe::uInt numberColumns = constraintRowSizes[blockIndex];
              const dftfe::uInt startingColumnNumber =
                constraintRowSizesAccumulated[blockIndex];
              const std::size_t xVecStartingIdRow =
                constrainedRowId * contiguousBlockSize;
              dftfe::utils::copyValue(xVec + xVecStartingIdRow +
                                        intraBlockIndex,
                                      inhomogenities[blockIndex]);
              for (dftfe::uInt i = 0; i < numberColumns; ++i)
                {
                  const dftfe::uInt constrainedColumnId =
                    constraintLocalColumnIdsAllRowsUnflattened
                      [startingColumnNumber + i];
                  const std::size_t xVecStartingIdColumn =
                    constrainedColumnId * contiguousBlockSize;
                  dftfe::utils::copyValue(
                    xVec + xVecStartingIdRow + intraBlockIndex,
                    dftfe::utils::add(
                      xVec[xVecStartingIdRow + intraBlockIndex],
                      dftfe::utils::makeComplex(
                        dftfe::utils::realPartDevice(
                          xVec[xVecStartingIdColumn + intraBlockIndex]) *
                          constraintColumnValuesAllRowsUnflattened
                            [startingColumnNumber + i],
                        dftfe::utils::imagPartDevice(
                          xVec[xVecStartingIdColumn + intraBlockIndex]) *
                          constraintColumnValuesAllRowsUnflattened
                            [startingColumnNumber + i])));
                }
            }
        },
        const dftfe::uInt                 contiguousBlockSize,
        dftfe::utils::deviceFloatComplex *xVec,
        const dftfe::uInt                *constraintLocalRowIdsUnflattened,
        const dftfe::uInt                 numConstraints,
        const dftfe::uInt                *constraintRowSizes,
        const dftfe::uInt                *constraintRowSizesAccumulated,
        const dftfe::uInt *constraintLocalColumnIdsAllRowsUnflattened,
        const double      *constraintColumnValuesAllRowsUnflattened,
        const double      *inhomogenities);



      DFTFE_CREATE_KERNEL(
        void,
        distributeSlaveToMasterKernelAtomicAdd,
        {
          const std::size_t numberEntries =
            numConstraints * contiguousBlockSize;

          for (std::size_t index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockIndex      = index / contiguousBlockSize;
              const dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
              const dftfe::uInt constrainedRowId =
                constraintLocalRowIdsUnflattened[blockIndex];
              const dftfe::uInt numberColumns = constraintRowSizes[blockIndex];
              const dftfe::uInt startingColumnNumber =
                constraintRowSizesAccumulated[blockIndex];
              const std::size_t xVecStartingIdRow =
                constrainedRowId * contiguousBlockSize;
              for (dftfe::uInt i = 0; i < numberColumns; ++i)
                {
                  const dftfe::uInt constrainedColumnId =
                    constraintLocalColumnIdsAllRowsUnflattened
                      [startingColumnNumber + i];
                  const std::size_t xVecStartingIdColumn =
                    constrainedColumnId * contiguousBlockSize;
                  dftfe::utils::atomicAddWrapper(
                    &(xVec[xVecStartingIdColumn + intraBlockIndex]),
                    constraintColumnValuesAllRowsUnflattened
                        [startingColumnNumber + i] *
                      xVec[xVecStartingIdRow + intraBlockIndex]);
                }
              xVec[xVecStartingIdRow + intraBlockIndex] = 0.0;
            }
        },
        const dftfe::uInt  contiguousBlockSize,
        double            *xVec,
        const dftfe::uInt *constraintLocalRowIdsUnflattened,
        const dftfe::uInt  numConstraints,
        const dftfe::uInt *constraintRowSizes,
        const dftfe::uInt *constraintRowSizesAccumulated,
        const dftfe::uInt *constraintLocalColumnIdsAllRowsUnflattened,
        const double      *constraintColumnValuesAllRowsUnflattened);



      DFTFE_CREATE_KERNEL(
        void,
        distributeSlaveToMasterKernelAtomicAdd,
        {
          const std::size_t numberEntries =
            numConstraints * contiguousBlockSize;

          for (std::size_t index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockIndex      = index / contiguousBlockSize;
              const dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
              const dftfe::uInt constrainedRowId =
                constraintLocalRowIdsUnflattened[blockIndex];
              const dftfe::uInt numberColumns = constraintRowSizes[blockIndex];
              const dftfe::uInt startingColumnNumber =
                constraintRowSizesAccumulated[blockIndex];
              const std::size_t xVecStartingIdRow =
                constrainedRowId * contiguousBlockSize;
              for (dftfe::uInt i = 0; i < numberColumns; ++i)
                {
                  const dftfe::uInt constrainedColumnId =
                    constraintLocalColumnIdsAllRowsUnflattened
                      [startingColumnNumber + i];
                  const std::size_t xVecStartingIdColumn =
                    constrainedColumnId * contiguousBlockSize;
                  const dftfe::utils::deviceDoubleComplex tempComplval =
                    dftfe::utils::mult(
                      constraintColumnValuesAllRowsUnflattened
                        [startingColumnNumber + i],
                      xVec[xVecStartingIdRow + intraBlockIndex]);

                  auto *add_real = reinterpret_cast<double *>(
                    &xVec[xVecStartingIdColumn + intraBlockIndex]);
                  auto *add_imag = add_real + 1;

                  dftfe::utils::atomicAddWrapper(
                    add_real, dftfe::utils::realPartDevice(tempComplval));
                  dftfe::utils::atomicAddWrapper(
                    add_imag, dftfe::utils::imagPartDevice(tempComplval));
                }
              xVec[xVecStartingIdRow + intraBlockIndex] =
                dftfe::utils::makeComplex(0.0, 0.0);
            }
        },
        const dftfe::uInt                  contiguousBlockSize,
        dftfe::utils::deviceDoubleComplex *xVec,
        const dftfe::uInt                 *constraintLocalRowIdsUnflattened,
        const dftfe::uInt                  numConstraints,
        const dftfe::uInt                 *constraintRowSizes,
        const dftfe::uInt                 *constraintRowSizesAccumulated,
        const dftfe::uInt *constraintLocalColumnIdsAllRowsUnflattened,
        const double      *constraintColumnValuesAllRowsUnflattened);



      DFTFE_CREATE_KERNEL(
        void,
        distributeSlaveToMasterKernelAtomicAdd,
        {
          const std::size_t numberEntries =
            numConstraints * contiguousBlockSize;

          for (std::size_t index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockIndex      = index / contiguousBlockSize;
              const dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
              const dftfe::uInt constrainedRowId =
                constraintLocalRowIdsUnflattened[blockIndex];
              const dftfe::uInt numberColumns = constraintRowSizes[blockIndex];
              const dftfe::uInt startingColumnNumber =
                constraintRowSizesAccumulated[blockIndex];
              const std::size_t xVecStartingIdRow =
                constrainedRowId * contiguousBlockSize;
              for (dftfe::uInt i = 0; i < numberColumns; ++i)
                {
                  const dftfe::uInt constrainedColumnId =
                    constraintLocalColumnIdsAllRowsUnflattened
                      [startingColumnNumber + i];
                  const std::size_t xVecStartingIdColumn =
                    constrainedColumnId * contiguousBlockSize;

                  dftfe::utils::atomicAddWrapper(
                    &(xVec[xVecStartingIdColumn + intraBlockIndex]),
                    constraintColumnValuesAllRowsUnflattened
                        [startingColumnNumber + i] *
                      xVec[xVecStartingIdRow + intraBlockIndex]);
                }
              xVec[xVecStartingIdRow + intraBlockIndex] = 0.0;
            }
        },
        const dftfe::uInt  contiguousBlockSize,
        float             *xVec,
        const dftfe::uInt *constraintLocalRowIdsUnflattened,
        const dftfe::uInt  numConstraints,
        const dftfe::uInt *constraintRowSizes,
        const dftfe::uInt *constraintRowSizesAccumulated,
        const dftfe::uInt *constraintLocalColumnIdsAllRowsUnflattened,
        const double      *constraintColumnValuesAllRowsUnflattened);



      DFTFE_CREATE_KERNEL(
        void,
        distributeSlaveToMasterKernelAtomicAdd,
        {
          const std::size_t numberEntries =
            numConstraints * contiguousBlockSize;

          for (std::size_t index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockIndex      = index / contiguousBlockSize;
              const dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
              const dftfe::uInt constrainedRowId =
                constraintLocalRowIdsUnflattened[blockIndex];
              const dftfe::uInt numberColumns = constraintRowSizes[blockIndex];
              const dftfe::uInt startingColumnNumber =
                constraintRowSizesAccumulated[blockIndex];
              const std::size_t xVecStartingIdRow =
                constrainedRowId * contiguousBlockSize;
              for (dftfe::uInt i = 0; i < numberColumns; ++i)
                {
                  const dftfe::uInt constrainedColumnId =
                    constraintLocalColumnIdsAllRowsUnflattened
                      [startingColumnNumber + i];
                  const std::size_t xVecStartingIdColumn =
                    constrainedColumnId * contiguousBlockSize;
                  const dftfe::utils::deviceDoubleComplex tempComplval =
                    dftfe::utils::mult(
                      constraintColumnValuesAllRowsUnflattened
                        [startingColumnNumber + i],
                      xVec[xVecStartingIdRow + intraBlockIndex]);

                  auto *add_real = reinterpret_cast<float *>(
                    &xVec[xVecStartingIdColumn + intraBlockIndex]);
                  auto *add_imag = add_real + 1;

                  dftfe::utils::atomicAddWrapper(
                    add_real, dftfe::utils::realPartDevice(tempComplval));
                  dftfe::utils::atomicAddWrapper(
                    add_imag, dftfe::utils::imagPartDevice(tempComplval));
                }
              xVec[xVecStartingIdRow + intraBlockIndex] =
                dftfe::utils::makeComplex((float)0.0, (float)0.0);
            }
        },
        const dftfe::uInt                 contiguousBlockSize,
        dftfe::utils::deviceFloatComplex *xVec,
        const dftfe::uInt                *constraintLocalRowIdsUnflattened,
        const dftfe::uInt                 numConstraints,
        const dftfe::uInt                *constraintRowSizes,
        const dftfe::uInt                *constraintRowSizesAccumulated,
        const dftfe::uInt *constraintLocalColumnIdsAllRowsUnflattened,
        const double      *constraintColumnValuesAllRowsUnflattened);



      DFTFE_CREATE_KERNEL(
        void,
        setzeroKernel,
        {
          const std::size_t numberEntries =
            numConstraints * contiguousBlockSize;

          for (std::size_t index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockIndex      = index / contiguousBlockSize;
              const dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
              xVec[constraintLocalRowIdsUnflattened[blockIndex] *
                     contiguousBlockSize +
                   intraBlockIndex]             = 0;
            }
        },
        const dftfe::uInt  contiguousBlockSize,
        double            *xVec,
        const dftfe::uInt *constraintLocalRowIdsUnflattened,
        const dftfe::uInt  numConstraints);



      DFTFE_CREATE_KERNEL(
        void,
        setzeroKernel,
        {
          const std::size_t numberEntries =
            numConstraints * contiguousBlockSize;

          for (std::size_t index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockIndex      = index / contiguousBlockSize;
              const dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
              xVec[constraintLocalRowIdsUnflattened[blockIndex] *
                     contiguousBlockSize +
                   intraBlockIndex]             = 0;
            }
        },
        const dftfe::uInt  contiguousBlockSize,
        float             *xVec,
        const dftfe::uInt *constraintLocalRowIdsUnflattened,
        const dftfe::uInt  numConstraints);



      DFTFE_CREATE_KERNEL(
        void,
        setzeroKernel,
        {
          const std::size_t numberEntries =
            numConstraints * contiguousBlockSize;

          for (std::size_t index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockIndex      = index / contiguousBlockSize;
              const dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
              dftfe::utils::copyValue(
                xVec +
                  constraintLocalRowIdsUnflattened[blockIndex] *
                    contiguousBlockSize +
                  intraBlockIndex,
                0.0);
            }
        },
        const dftfe::uInt                  contiguousBlockSize,
        dftfe::utils::deviceDoubleComplex *xVec,
        const dftfe::uInt                 *constraintLocalRowIdsUnflattened,
        const dftfe::uInt                  numConstraints);



      DFTFE_CREATE_KERNEL(
        void,
        setzeroKernel,
        {
          const std::size_t numberEntries =
            numConstraints * contiguousBlockSize;

          for (std::size_t index = globalThreadId; index < numberEntries;
               index += nThreadsPerBlock * nThreadBlock)
            {
              const dftfe::uInt blockIndex      = index / contiguousBlockSize;
              const dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
              dftfe::utils::copyValue(
                xVec +
                  constraintLocalRowIdsUnflattened[blockIndex] *
                    contiguousBlockSize +
                  intraBlockIndex,
                0.0);
            }
        },
        const dftfe::uInt                 contiguousBlockSize,
        dftfe::utils::deviceFloatComplex *xVec,
        const dftfe::uInt                *constraintLocalRowIdsUnflattened,
        const dftfe::uInt                 numConstraints);

    } // namespace
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
      const double      *inhomogenities)
    {
      DFTFE_LAUNCH_KERNEL(distributeKernel,
                          std::min((contiguousBlockSize * numConstraints +
                                    (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                     dftfe::utils::DEVICE_BLOCK_SIZE,
                                   dftfe::uInt(30000)),
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::linearAlgebra::BLASWrapper<
                            dftfe::utils::MemorySpace::DEVICE>::d_streamId,
                          contiguousBlockSize,
                          dftfe::utils::makeDataTypeDeviceCompatible(xVec),
                          constraintLocalRowIdsUnflattened,
                          numConstraints,
                          constraintRowSizes,
                          constraintRowSizesAccumulated,
                          constraintLocalColumnIdsAllRowsUnflattened,
                          constraintColumnValuesAllRowsUnflattened,
                          inhomogenities);
    }

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
      const double      *constraintColumnValuesAllRowsUnflattened)
    {
      DFTFE_LAUNCH_KERNEL(distributeSlaveToMasterKernelAtomicAdd,
                          std::min((contiguousBlockSize * numConstraints +
                                    (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                     dftfe::utils::DEVICE_BLOCK_SIZE,
                                   dftfe::uInt(30000)),
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::linearAlgebra::BLASWrapper<
                            dftfe::utils::MemorySpace::DEVICE>::d_streamId,
                          contiguousBlockSize,
                          dftfe::utils::makeDataTypeDeviceCompatible(xVec),
                          constraintLocalRowIdsUnflattened,
                          numConstraints,
                          constraintRowSizes,
                          constraintRowSizesAccumulated,
                          constraintLocalColumnIdsAllRowsUnflattened,
                          constraintColumnValuesAllRowsUnflattened);
    }
    template <typename ValueType>
    void
    setzeroDevice(const dftfe::uInt  contiguousBlockSize,
                  ValueType         *xVec,
                  const dftfe::uInt *constraintLocalRowIdsUnflattened,
                  const dftfe::uInt  numConstraints)
    {
      DFTFE_LAUNCH_KERNEL(setzeroKernel,
                          std::min((contiguousBlockSize * numConstraints +
                                    (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                     dftfe::utils::DEVICE_BLOCK_SIZE,
                                   dftfe::uInt(30000)),
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::linearAlgebra::BLASWrapper<
                            dftfe::utils::MemorySpace::DEVICE>::d_streamId,
                          contiguousBlockSize,
                          dftfe::utils::makeDataTypeDeviceCompatible(xVec),
                          constraintLocalRowIdsUnflattened,
                          numConstraints);
    }

    void
    scaleConstraintsDevice(
      const double      *xVec,
      const dftfe::uInt *constraintLocalRowIdsUnflattened,
      const dftfe::uInt  numConstraints,
      const dftfe::uInt *constraintRowSizes,
      const dftfe::uInt *constraintRowSizesAccumulated,
      const dftfe::uInt *constraintLocalColumnIdsAllRowsUnflattened,
      double            *constraintColumnValuesAllRowsUnflattened)
    {
      DFTFE_LAUNCH_KERNEL(scaleConstraintsKernel,
                          std::min((numConstraints +
                                    (dftfe::utils::DEVICE_BLOCK_SIZE - 1)) /
                                     dftfe::utils::DEVICE_BLOCK_SIZE,
                                   dftfe::uInt(30000)),
                          dftfe::utils::DEVICE_BLOCK_SIZE,
                          dftfe::linearAlgebra::BLASWrapper<
                            dftfe::utils::MemorySpace::DEVICE>::d_streamId,
                          dftfe::utils::makeDataTypeDeviceCompatible(xVec),
                          constraintLocalRowIdsUnflattened,
                          numConstraints,
                          constraintRowSizes,
                          constraintRowSizesAccumulated,
                          constraintLocalColumnIdsAllRowsUnflattened,
                          constraintColumnValuesAllRowsUnflattened);
    }
    template void
    distributeDevice(
      const dftfe::uInt  contiguousBlockSize,
      double            *xVec,
      const dftfe::uInt *constraintLocalRowIdsUnflattened,
      const dftfe::uInt  numConstraints,
      const dftfe::uInt *constraintRowSizes,
      const dftfe::uInt *constraintRowSizesAccumulated,
      const dftfe::uInt *constraintLocalColumnIdsAllRowsUnflattened,
      const double      *constraintColumnValuesAllRowsUnflattened,
      const double      *inhomogenities);
    template void
    distributeDevice(
      const dftfe::uInt  contiguousBlockSize,
      float             *xVec,
      const dftfe::uInt *constraintLocalRowIdsUnflattened,
      const dftfe::uInt  numConstraints,
      const dftfe::uInt *constraintRowSizes,
      const dftfe::uInt *constraintRowSizesAccumulated,
      const dftfe::uInt *constraintLocalColumnIdsAllRowsUnflattened,
      const double      *constraintColumnValuesAllRowsUnflattened,
      const double      *inhomogenities);
    template void
    distributeDevice(
      const dftfe::uInt     contiguousBlockSize,
      std::complex<double> *xVec,
      const dftfe::uInt    *constraintLocalRowIdsUnflattened,
      const dftfe::uInt     numConstraints,
      const dftfe::uInt    *constraintRowSizes,
      const dftfe::uInt    *constraintRowSizesAccumulated,
      const dftfe::uInt    *constraintLocalColumnIdsAllRowsUnflattened,
      const double         *constraintColumnValuesAllRowsUnflattened,
      const double         *inhomogenities);
    template void
    distributeDevice(
      const dftfe::uInt    contiguousBlockSize,
      std::complex<float> *xVec,
      const dftfe::uInt   *constraintLocalRowIdsUnflattened,
      const dftfe::uInt    numConstraints,
      const dftfe::uInt   *constraintRowSizes,
      const dftfe::uInt   *constraintRowSizesAccumulated,
      const dftfe::uInt   *constraintLocalColumnIdsAllRowsUnflattened,
      const double        *constraintColumnValuesAllRowsUnflattened,
      const double        *inhomogenities);
    template void
    distributeSlaveToMasterAtomicAddDevice(
      const dftfe::uInt  contiguousBlockSize,
      double            *xVec,
      const dftfe::uInt *constraintLocalRowIdsUnflattened,
      const dftfe::uInt  numConstraints,
      const dftfe::uInt *constraintRowSizes,
      const dftfe::uInt *constraintRowSizesAccumulated,
      const dftfe::uInt *constraintLocalColumnIdsAllRowsUnflattened,
      const double      *constraintColumnValuesAllRowsUnflattened);
    template void
    distributeSlaveToMasterAtomicAddDevice(
      const dftfe::uInt  contiguousBlockSize,
      float             *xVec,
      const dftfe::uInt *constraintLocalRowIdsUnflattened,
      const dftfe::uInt  numConstraints,
      const dftfe::uInt *constraintRowSizes,
      const dftfe::uInt *constraintRowSizesAccumulated,
      const dftfe::uInt *constraintLocalColumnIdsAllRowsUnflattened,
      const double      *constraintColumnValuesAllRowsUnflattened);
    template void
    distributeSlaveToMasterAtomicAddDevice(
      const dftfe::uInt     contiguousBlockSize,
      std::complex<double> *xVec,
      const dftfe::uInt    *constraintLocalRowIdsUnflattened,
      const dftfe::uInt     numConstraints,
      const dftfe::uInt    *constraintRowSizes,
      const dftfe::uInt    *constraintRowSizesAccumulated,
      const dftfe::uInt    *constraintLocalColumnIdsAllRowsUnflattened,
      const double         *constraintColumnValuesAllRowsUnflattened);
    template void
    distributeSlaveToMasterAtomicAddDevice(
      const dftfe::uInt    contiguousBlockSize,
      std::complex<float> *xVec,
      const dftfe::uInt   *constraintLocalRowIdsUnflattened,
      const dftfe::uInt    numConstraints,
      const dftfe::uInt   *constraintRowSizes,
      const dftfe::uInt   *constraintRowSizesAccumulated,
      const dftfe::uInt   *constraintLocalColumnIdsAllRowsUnflattened,
      const double        *constraintColumnValuesAllRowsUnflattened);
    template void
    setzeroDevice(const dftfe::uInt  contiguousBlockSize,
                  double            *xVec,
                  const dftfe::uInt *constraintLocalRowIdsUnflattened,
                  const dftfe::uInt  numConstraints);
    template void
    setzeroDevice(const dftfe::uInt  contiguousBlockSize,
                  float             *xVec,
                  const dftfe::uInt *constraintLocalRowIdsUnflattened,
                  const dftfe::uInt  numConstraints);
    template void
    setzeroDevice(const dftfe::uInt     contiguousBlockSize,
                  std::complex<double> *xVec,
                  const dftfe::uInt    *constraintLocalRowIdsUnflattened,
                  const dftfe::uInt     numConstraints);
    template void
    setzeroDevice(const dftfe::uInt    contiguousBlockSize,
                  std::complex<float> *xVec,
                  const dftfe::uInt   *constraintLocalRowIdsUnflattened,
                  const dftfe::uInt    numConstraints);

  } // namespace dftUtils
} // namespace dftfe
