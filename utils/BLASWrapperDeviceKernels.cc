// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2022  The Regents of the University of Michigan and DFT-FE
// authors.
//
// This file is part of the DFT-FE code.
//
// The DFT-FE code is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the DFT-FE distribution.
//
// ---------------------------------------------------------------------
//
namespace dftfe
{
  namespace
  {
    template <typename ValueType>
    __global__ void
    saddKernel(ValueType *            y,
               ValueType *            x,
               const ValueType        beta,
               const dftfe::size_type size)
    {
      const dftfe::size_type globalId = threadIdx.x + blockIdx.x * blockDim.x;

      for (dftfe::size_type idx = globalId; idx < size;
           idx += blockDim.x * gridDim.x)
        {
          y[idx] = beta * y[idx] - x[idx];
          x[idx] = 0;
        }
    }


    template <typename ValueTypeComplex, typename ValueTypeReal>
    __global__ void
    copyComplexArrToRealArrsDeviceKernel(const dftfe::size_type  size,
                                         const ValueTypeComplex *complexArr,
                                         ValueTypeReal *         realArr,
                                         ValueTypeReal *         imagArr)
    {
      const dftfe::size_type globalId = threadIdx.x + blockIdx.x * blockDim.x;

      for (dftfe::size_type idx = globalId; idx < size;
           idx += blockDim.x * gridDim.x)
        {
          realArr[idx] = complexArr[idx].x;
          imagArr[idx] = complexArr[idx].y;
        }
    }

    template <typename ValueTypeComplex, typename ValueTypeReal>
    __global__ void
    copyRealArrsToComplexArrDeviceKernel(const dftfe::size_type size,
                                         const ValueTypeReal *  realArr,
                                         const ValueTypeReal *  imagArr,
                                         ValueTypeComplex *     complexArr)
    {
      const dftfe::size_type globalId = threadIdx.x + blockIdx.x * blockDim.x;

      for (dftfe::size_type idx = globalId; idx < size;
           idx += blockDim.x * gridDim.x)
        {
          complexArr[idx].x = realArr[idx];
          complexArr[idx].y = imagArr[idx];
        }
    }

    template <typename ValueType1, typename ValueType2>
    __global__ void
    copyValueType1ArrToValueType2ArrDeviceKernel(
      const dftfe::size_type size,
      const ValueType1 *     valueType1Arr,
      ValueType2 *           valueType2Arr)
    {
      const dftfe::size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;

      for (dftfe::size_type index = globalThreadId; index < size;
           index += blockDim.x * gridDim.x)
        dftfe::utils::copyValue(valueType2Arr + index, valueType1Arr[index]);
    }


    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedCopyToBlockDeviceKernel(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1 *             copyFromVec,
      ValueType2 *                   copyToVec,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {
      const dftfe::size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::size_type blockIndex = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex =
            index - blockIndex * contiguousBlockSize;
          dftfe::utils::copyValue(
            copyToVec + index,
            copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex]);
        }
    }

    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedCopyToBlockScaleDeviceKernel(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1               a,
      const ValueType1 *             s,
      const ValueType2 *             copyFromVec,
      ValueType2 *                   copyToVec,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {
      const dftfe::size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::size_type blockIndex = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex =
            index - blockIndex * contiguousBlockSize;
          const ValueType1 coeff = dftfe::utils::mult(a, s[blockIndex]);
          dftfe::utils::copyValue(
            copyToVec + index,
            dftfe::utils::mult(
              copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex],
              coeff));
        }
    }


    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedCopyFromBlockDeviceKernel(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1 *             copyFromVec,
      ValueType2 *                   copyToVec,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {
      const dftfe::size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::size_type blockIndex = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex =
            index - blockIndex * contiguousBlockSize;
          dftfe::utils::copyValue(
            copyToVec + copyFromVecStartingContiguousBlockIds[blockIndex] +
              intraBlockIndex,
            copyFromVec[index]);
        }
    }


    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedCopyToBlockConstantStrideDeviceKernel(
      const dftfe::size_type blockSizeTo,
      const dftfe::size_type blockSizeFrom,
      const dftfe::size_type numBlocks,
      const dftfe::size_type startingId,
      const ValueType1 *     copyFromVec,
      ValueType2 *           copyToVec)
    {
      {
        const dftfe::size_type globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dftfe::size_type numberEntries = numBlocks * blockSizeTo;

        for (dftfe::size_type index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            dftfe::size_type blockIndex      = index / blockSizeTo;
            dftfe::size_type intraBlockIndex = index - blockIndex * blockSizeTo;
            dftfe::utils::copyValue(copyToVec + index,
                                    copyFromVec[blockIndex * blockSizeFrom +
                                                startingId + intraBlockIndex]);
          }
      }
    }

    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedCopyFromBlockConstantStrideDeviceKernel(
      const dftfe::size_type blockSizeTo,
      const dftfe::size_type blockSizeFrom,
      const dftfe::size_type numBlocks,
      const dftfe::size_type startingId,
      const ValueType1 *     copyFromVec,
      ValueType2 *           copyToVec)
    {
      {
        const dftfe::size_type globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dftfe::size_type numberEntries = numBlocks * blockSizeFrom;

        for (dftfe::size_type index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            dftfe::size_type blockIndex = index / blockSizeFrom;
            dftfe::size_type intraBlockIndex =
              index - blockIndex * blockSizeFrom;
            dftfe::utils::copyValue(copyToVec + blockIndex * blockSizeTo +
                                      startingId + intraBlockIndex,
                                    copyFromVec[index]);
          }
      }
    }

    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedCopyConstantStrideDeviceKernel(const dftfe::size_type blockSize,
                                          const dftfe::size_type strideTo,
                                          const dftfe::size_type strideFrom,
                                          const dftfe::size_type numBlocks,
                                          const dftfe::size_type startingToId,
                                          const dftfe::size_type startingFromId,
                                          const ValueType1 *     copyFromVec,
                                          ValueType2 *           copyToVec)
    {
      {
        const dftfe::size_type globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dftfe::size_type numberEntries = numBlocks * blockSize;

        for (dftfe::size_type index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            dftfe::size_type blockIndex      = index / blockSize;
            dftfe::size_type intraBlockIndex = index - blockIndex * blockSize;
            dftfe::utils::copyValue(
              copyToVec + blockIndex * strideTo + startingToId +
                intraBlockIndex,
              copyFromVec[blockIndex * strideFrom + startingFromId +
                          intraBlockIndex]);
          }
      }
    }


    // x=a*x, with inc=1
    template <typename ValueType1, typename ValueType2>
    __global__ void
    ascalDeviceKernel(const dftfe::size_type n,
                      ValueType1 *           x,
                      const ValueType2       a)
    {
      for (dftfe::size_type i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
           i += blockDim.x * gridDim.x)
        dftfe::utils::copyValue(x + i, dftfe::utils::mult(a, x[i]));
    }


    // x[iblock*blocksize+intrablockindex]=a*s[iblock]*x[iblock*blocksize+intrablockindex]
    // strided block wise
    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedBlockScaleDeviceKernel(const dftfe::size_type contiguousBlockSize,
                                  const dftfe::size_type numContiguousBlocks,
                                  const ValueType1       a,
                                  const ValueType1 *     s,
                                  ValueType2 *           x)
    {
      const dftfe::size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::size_type blockIndex = index / contiguousBlockSize;
          dftfe::utils::copyValue(
            x + index,
            dftfe::utils::mult(dftfe::utils::mult(a, s[blockIndex]), x[index]));
        }
    }

    // y=a*x+b*y, with inc=1
    template <typename ValueType1, typename ValueType2>
    __global__ void
    axpbyDeviceKernel(const dftfe::size_type n,
                      const ValueType1 *     x,
                      ValueType1 *           y,
                      const ValueType2       a,
                      const ValueType2       b)
    {
      for (dftfe::size_type i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
           i += blockDim.x * gridDim.x)
        dftfe::utils::copyValue(y + i,
                                dftfe::utils::add(dftfe::utils::mult(a, x[i]),
                                                  dftfe::utils::mult(b, y[i])));
    }

    __global__ void
    axpyStridedBlockAtomicAddDeviceKernel(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double *                 addFromVec,
      double *                       addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      const dftfe::size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          atomicAdd(&addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                              intraBlockIndex],
                    addFromVec[index]);
        }
    }

    __global__ void
    axpyStridedBlockAtomicAddDeviceKernel(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const double *                 s,
      const double *                 addFromVec,
      double *                       addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      const dftfe::size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          const double     coeff = dftfe::utils::mult(a, s[blockIndex]);
          atomicAdd(&addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                              intraBlockIndex],
                    dftfe::utils::mult(addFromVec[index], coeff));
        }
    }

    __global__ void
    axpyStridedBlockAtomicAddDeviceKernel(
      const dftfe::size_type                   contiguousBlockSize,
      const dftfe::size_type                   numContiguousBlocks,
      const double                             a,
      const double *                           s,
      const dftfe::utils::deviceDoubleComplex *addFromVec,
      dftfe::utils::deviceDoubleComplex *      addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      const dftfe::size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          const double     coeff = dftfe::utils::mult(a, s[blockIndex]);
          atomicAdd(&(addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                               intraBlockIndex]
                        .x),
                    dftfe::utils::mult(addFromVec[index].x, coeff));
          atomicAdd(&(addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                               intraBlockIndex]
                        .y),
                    dftfe::utils::mult(addFromVec[index].y, coeff));
        }
    }

    __global__ void
    axpyStridedBlockAtomicAddDeviceKernel(
      const dftfe::size_type                   contiguousBlockSize,
      const dftfe::size_type                   numContiguousBlocks,
      const dftfe::utils::deviceDoubleComplex *addFromVec,
      dftfe::utils::deviceDoubleComplex *      addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      const dftfe::size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          atomicAdd(&addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                              intraBlockIndex]
                       .x,
                    addFromVec[index].x);
          atomicAdd(&addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                              intraBlockIndex]
                       .y,
                    addFromVec[index].y);
        }
    }


    __global__ void
    axpyStridedBlockAtomicAddDeviceKernel(
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double *                 addFromVec,
      double *                       addToVecReal,
      double *                       addToVecImag,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      const dftfe::size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          atomicAdd(
            &addToVecReal[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex],
            addFromVec[index]);
          atomicAdd(
            &addToVecImag[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex],
            addFromVec[index]);
        }
    }

    __global__ void
    axpyStridedBlockAtomicAddDeviceKernel(
      const dftfe::size_type                   contiguousBlockSize,
      const dftfe::size_type                   numContiguousBlocks,
      const dftfe::utils::deviceDoubleComplex *addFromVec,
      double *                                 addToVecReal,
      double *                                 addToVecImag,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      const dftfe::size_type globalThreadId =
        blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          atomicAdd(
            &addToVecReal[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex],
            addFromVec[index].x);
          atomicAdd(
            &addToVecImag[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex],
            addFromVec[index].y);
        }
    }

  } // namespace
} // namespace dftfe
