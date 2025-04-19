// ---------------------------------------------------------------------
//
// Copyright (c) 2017-2025  The Regents of the University of Michigan and DFT-FE
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
    saddKernel(ValueType        *y,
               ValueType        *x,
               const ValueType   beta,
               const dftfe::uInt size)
    {
      const dftfe::uInt globalId = threadIdx.x + blockIdx.x * blockDim.x;

      for (dftfe::uInt idx = globalId; idx < size;
           idx += blockDim.x * gridDim.x)
        {
          y[idx] = beta * y[idx] - x[idx];
          x[idx] = 0;
        }
    }


    template <typename ValueTypeComplex, typename ValueTypeReal>
    __global__ void
    copyComplexArrToRealArrsDeviceKernel(const dftfe::uInt       size,
                                         const ValueTypeComplex *complexArr,
                                         ValueTypeReal          *realArr,
                                         ValueTypeReal          *imagArr)
    {
      const dftfe::uInt globalId = threadIdx.x + blockIdx.x * blockDim.x;

      for (dftfe::uInt idx = globalId; idx < size;
           idx += blockDim.x * gridDim.x)
        {
          realArr[idx] = complexArr[idx].x;
          imagArr[idx] = complexArr[idx].y;
        }
    }

    template <typename ValueTypeComplex, typename ValueTypeReal>
    __global__ void
    copyRealArrsToComplexArrDeviceKernel(const dftfe::uInt    size,
                                         const ValueTypeReal *realArr,
                                         const ValueTypeReal *imagArr,
                                         ValueTypeComplex    *complexArr)
    {
      const dftfe::uInt globalId = threadIdx.x + blockIdx.x * blockDim.x;

      for (dftfe::uInt idx = globalId; idx < size;
           idx += blockDim.x * gridDim.x)
        {
          complexArr[idx].x = realArr[idx];
          complexArr[idx].y = imagArr[idx];
        }
    }

    template <typename ValueType1, typename ValueType2>
    __global__ void
    copyValueType1ArrToValueType2ArrDeviceKernel(
      const dftfe::uInt size,
      const ValueType1 *valueType1Arr,
      ValueType2       *valueType2Arr)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

      for (dftfe::uInt index = globalThreadId; index < size;
           index += blockDim.x * gridDim.x)
        dftfe::utils::copyValue(valueType2Arr + index, valueType1Arr[index]);
    }
    template <typename ValueType1, typename ValueType2>
    __global__ void
    copyBlockDiagonalValueType1OffDiagonalValueType2FromValueType1ArrDeviceKernel(
      const dftfe::uInt B,
      const dftfe::uInt DRem,
      const dftfe::uInt D,
      const ValueType1 *valueType1SrcArray,
      ValueType1       *valueType1DstArray,
      ValueType2       *valueType2DstArray)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt size           = B * D;
      for (dftfe::uInt index = globalThreadId; index < size;
           index += blockDim.x * gridDim.x)
        {
          const dftfe::uInt ibdof = index / D;
          const dftfe::uInt ivec  = index % D;
          if (ivec < B)
            dftfe::utils::copyValue(valueType1DstArray + ibdof * B + ivec,
                                    valueType1SrcArray[index]);
          else
            dftfe::utils::copyValue(valueType2DstArray + (ibdof - B) +
                                      (ivec - B) * B,
                                    valueType1SrcArray[index]);
        }
    }

    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedCopyToBlockDeviceKernel(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const ValueType1  *copyFromVec,
      ValueType2        *copyToVec,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt blockIndex = index / contiguousBlockSize;
          dftfe::uInt intraBlockIndex =
            index - blockIndex * contiguousBlockSize;
          dftfe::utils::copyValue(
            copyToVec + index,
            copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex]);
        }
    }

    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedCopyToBlockDeviceKernel(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const dftfe::uInt  stratingVecId,
      const ValueType1  *copyFromVec,
      ValueType2        *copyToVec,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt blockIndex = index / contiguousBlockSize;
          dftfe::uInt intraBlockIndex =
            index - blockIndex * contiguousBlockSize;
          dftfe::utils::copyValue(
            copyToVec + index,
            copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex + stratingVecId]);
        }
    }

    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedCopyToBlockScaleDeviceKernel(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const ValueType1   a,
      const ValueType1  *s,
      const ValueType2  *copyFromVec,
      ValueType2        *copyToVec,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt blockIndex = index / contiguousBlockSize;
          dftfe::uInt intraBlockIndex =
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
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const ValueType1  *copyFromVec,
      ValueType2        *copyToVec,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt blockIndex = index / contiguousBlockSize;
          dftfe::uInt intraBlockIndex =
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
      const dftfe::uInt blockSizeTo,
      const dftfe::uInt blockSizeFrom,
      const dftfe::uInt numBlocks,
      const dftfe::uInt startingId,
      const ValueType1 *copyFromVec,
      ValueType2       *copyToVec)
    {
      {
        const dftfe::uInt globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dftfe::uInt numberEntries = numBlocks * blockSizeTo;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            dftfe::uInt blockIndex      = index / blockSizeTo;
            dftfe::uInt intraBlockIndex = index - blockIndex * blockSizeTo;
            dftfe::utils::copyValue(copyToVec + index,
                                    copyFromVec[blockIndex * blockSizeFrom +
                                                startingId + intraBlockIndex]);
          }
      }
    }

    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedCopyFromBlockConstantStrideDeviceKernel(
      const dftfe::uInt blockSizeTo,
      const dftfe::uInt blockSizeFrom,
      const dftfe::uInt numBlocks,
      const dftfe::uInt startingId,
      const ValueType1 *copyFromVec,
      ValueType2       *copyToVec)
    {
      {
        const dftfe::uInt globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dftfe::uInt numberEntries = numBlocks * blockSizeFrom;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            dftfe::uInt blockIndex      = index / blockSizeFrom;
            dftfe::uInt intraBlockIndex = index - blockIndex * blockSizeFrom;
            dftfe::utils::copyValue(copyToVec + blockIndex * blockSizeTo +
                                      startingId + intraBlockIndex,
                                    copyFromVec[index]);
          }
      }
    }

    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedCopyConstantStrideDeviceKernel(const dftfe::uInt blockSize,
                                          const dftfe::uInt strideTo,
                                          const dftfe::uInt strideFrom,
                                          const dftfe::uInt numBlocks,
                                          const dftfe::uInt startingToId,
                                          const dftfe::uInt startingFromId,
                                          const ValueType1 *copyFromVec,
                                          ValueType2       *copyToVec)
    {
      {
        const dftfe::uInt globalThreadId =
          blockIdx.x * blockDim.x + threadIdx.x;
        const dftfe::uInt numberEntries = numBlocks * blockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += blockDim.x * gridDim.x)
          {
            dftfe::uInt blockIndex      = index / blockSize;
            dftfe::uInt intraBlockIndex = index - blockIndex * blockSize;
            dftfe::utils::copyValue(
              copyToVec + blockIndex * strideTo + startingToId +
                intraBlockIndex,
              copyFromVec[blockIndex * strideFrom + startingFromId +
                          intraBlockIndex]);
          }
      }
    }

    template <typename ValueType>
    __global__ void
    addVecOverContinuousIndexKernel(const dftfe::uInt numContiguousBlocks,
                                    const dftfe::uInt contiguousBlockSize,
                                    const ValueType  *input1,
                                    const ValueType  *input2,
                                    ValueType        *output)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries  = numContiguousBlocks;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          for (dftfe::uInt iBlock = 0; iBlock < contiguousBlockSize; iBlock++)
            {
              //                    output[index] +=
              //                    input1[index*contiguousBlockSize + iBlock]*
              //                            input2[index*contiguousBlockSize +
              //                            iBlock];

              dftfe::utils::copyValue(
                output + index,
                dftfe::utils::add(
                  output[index],
                  dftfe::utils::mult(
                    input1[index * contiguousBlockSize + iBlock],
                    input2[index * contiguousBlockSize + iBlock])));
            }
        }
    }


    // x=a*x, with inc=1
    template <typename ValueType1, typename ValueType2>
    __global__ void
    ascalDeviceKernel(const dftfe::uInt n, ValueType1 *x, const ValueType2 a)
    {
      for (dftfe::uInt i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
           i += blockDim.x * gridDim.x)
        dftfe::utils::copyValue(x + i, dftfe::utils::mult(a, x[i]));
    }


    // x[iblock*blocksize+intrablockindex]=a*s[iblock]*x[iblock*blocksize+intrablockindex]
    // strided block wise
    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedBlockScaleDeviceKernel(const dftfe::uInt contiguousBlockSize,
                                  const dftfe::uInt numContiguousBlocks,
                                  const ValueType1  a,
                                  const ValueType1 *s,
                                  ValueType2       *x)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt blockIndex = index / contiguousBlockSize;
          dftfe::utils::copyValue(
            x + index,
            dftfe::utils::mult(dftfe::utils::mult(a, s[blockIndex]), x[index]));
        }
    }

    // x[iblock*blocksize+intrablockindex]=
    // beta[intrablockindex]*x[iblock*blocksize+intrablockindex] strided block
    // wise
    template <typename ValueType>
    __global__ void
    stridedBlockScaleColumnWiseKernel(const dftfe::uInt contiguousBlockSize,
                                      const dftfe::uInt numContiguousBlocks,
                                      const ValueType  *beta,
                                      ValueType        *x)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt blockIndex = index / contiguousBlockSize;
          dftfe::uInt intrablockindex =
            index - blockIndex * contiguousBlockSize;
          dftfe::utils::copyValue(x + index,
                                  dftfe::utils::mult(beta[intrablockindex],
                                                     x[index]));
        }
    }

    // y[iblock*blocksize+intrablockindex]= y[iblock*blocksize+intrablockindex]
    // +
    //                                      beta[intrablockindex]*x[iblock*blocksize+intrablockindex]
    // strided block wise
    template <typename ValueType>
    __global__ void
    stridedBlockScaleAndAddColumnWiseKernel(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const ValueType  *x,
      const ValueType  *beta,
      ValueType        *y)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt blockIndex = index / contiguousBlockSize;
          dftfe::uInt intrablockindex =
            index - blockIndex * contiguousBlockSize;
          dftfe::utils::copyValue(
            y + index,
            dftfe::utils::add(
              y[index], dftfe::utils::mult(beta[intrablockindex], x[index])));
        }
    }

    // z[iblock*blocksize+intrablockindex]=
    // alpha[intrablockindex]*x[iblock*blocksize+intrablockindex] +
    //                                      beta[intrablockindex]*y[iblock*blocksize+intrablockindex]
    // strided block wise
    template <typename ValueType>
    __global__ void
    stridedBlockScaleAndAddTwoVecColumnWiseKernel(
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const ValueType  *x,
      const ValueType  *alpha,
      const ValueType  *y,
      const ValueType  *beta,
      ValueType        *z)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt blockIndex = index / contiguousBlockSize;
          dftfe::uInt intrablockindex =
            index - blockIndex * contiguousBlockSize;
          dftfe::utils::copyValue(
            z + index,
            dftfe::utils::add(
              dftfe::utils::mult(alpha[intrablockindex], x[index]),
              dftfe::utils::mult(beta[intrablockindex], y[index])));
        }
    }

    // y=a*x+b*y, with inc=1
    template <typename ValueType1, typename ValueType2>
    __global__ void
    axpbyDeviceKernel(const dftfe::uInt n,
                      const ValueType1 *x,
                      ValueType1       *y,
                      const ValueType2  a,
                      const ValueType2  b)
    {
      for (dftfe::uInt i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
           i += blockDim.x * gridDim.x)
        dftfe::utils::copyValue(y + i,
                                dftfe::utils::add(dftfe::utils::mult(a, x[i]),
                                                  dftfe::utils::mult(b, y[i])));
    }

    __global__ void
    axpyStridedBlockAtomicAddDeviceKernel(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double      *addFromVec,
      double            *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt blockIndex      = index / contiguousBlockSize;
          dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
          atomicAdd(&addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                              intraBlockIndex],
                    addFromVec[index]);
        }
    }

    __global__ void
    axpyStridedBlockAtomicAddDeviceKernel(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double       a,
      const double      *s,
      const double      *addFromVec,
      double            *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt  blockIndex      = index / contiguousBlockSize;
          dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = dftfe::utils::mult(a, s[blockIndex]);
          atomicAdd(&addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                              intraBlockIndex],
                    dftfe::utils::mult(addFromVec[index], coeff));
        }
    }

    __global__ void
    axpyStridedBlockAtomicAddDeviceKernel(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double       a,
      const double      *addFromVec,
      double            *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt  blockIndex      = index / contiguousBlockSize;
          dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = a;
          atomicAdd(&addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                              intraBlockIndex],
                    dftfe::utils::mult(addFromVec[index], coeff));
        }
    }


    __global__ void
    axpyStridedBlockAtomicAddDeviceKernel(
      const dftfe::uInt                        contiguousBlockSize,
      const dftfe::uInt                        numContiguousBlocks,
      const double                             a,
      const double                            *s,
      const dftfe::utils::deviceDoubleComplex *addFromVec,
      dftfe::utils::deviceDoubleComplex       *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt  blockIndex      = index / contiguousBlockSize;
          dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = dftfe::utils::mult(a, s[blockIndex]);
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
      const dftfe::uInt                        contiguousBlockSize,
      const dftfe::uInt                        numContiguousBlocks,
      const double                             a,
      const dftfe::utils::deviceDoubleComplex *addFromVec,
      dftfe::utils::deviceDoubleComplex       *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt  blockIndex      = index / contiguousBlockSize;
          dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = a;
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
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double       a,
      const double      *s,
      const float       *addFromVec,
      double            *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt  blockIndex      = index / contiguousBlockSize;
          dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = dftfe::utils::mult(a, s[blockIndex]);
          atomicAdd(&addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                              intraBlockIndex],
                    dftfe::utils::mult(addFromVec[index], coeff));
        }
    }

    __global__ void
    axpyStridedBlockAtomicAddDeviceKernel(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double       a,
      const float       *addFromVec,
      double            *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt  blockIndex      = index / contiguousBlockSize;
          dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = a;
          atomicAdd(&addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                              intraBlockIndex],
                    dftfe::utils::mult(addFromVec[index], coeff));
        }
    }

    __global__ void
    axpyStridedBlockAtomicAddDeviceKernel(
      const dftfe::uInt                       contiguousBlockSize,
      const dftfe::uInt                       numContiguousBlocks,
      const double                            a,
      const double                           *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceDoubleComplex      *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt  blockIndex      = index / contiguousBlockSize;
          dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = dftfe::utils::mult(a, s[blockIndex]);
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
      const dftfe::uInt                       contiguousBlockSize,
      const dftfe::uInt                       numContiguousBlocks,
      const double                            a,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceDoubleComplex      *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt  blockIndex      = index / contiguousBlockSize;
          dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = a;
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
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double       a,
      const double      *s,
      const float       *addFromVec,
      float             *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt  blockIndex      = index / contiguousBlockSize;
          dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = dftfe::utils::mult(a, s[blockIndex]);
          atomicAdd(&addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                              intraBlockIndex],
                    dftfe::utils::mult(addFromVec[index], coeff));
        }
    }
    __global__ void
    axpyStridedBlockAtomicAddDeviceKernel(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double       a,
      const float       *addFromVec,
      float             *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt  blockIndex      = index / contiguousBlockSize;
          dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = a;
          atomicAdd(&addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                              intraBlockIndex],
                    dftfe::utils::mult(addFromVec[index], coeff));
        }
    }

    __global__ void
    axpyStridedBlockAtomicAddDeviceKernel(
      const dftfe::uInt                       contiguousBlockSize,
      const dftfe::uInt                       numContiguousBlocks,
      const double                            a,
      const double                           *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt  blockIndex      = index / contiguousBlockSize;
          dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = dftfe::utils::mult(a, s[blockIndex]);
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
      const dftfe::uInt                       contiguousBlockSize,
      const dftfe::uInt                       numContiguousBlocks,
      const double                            a,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt  blockIndex      = index / contiguousBlockSize;
          dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = a;
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
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const float        a,
      const float       *s,
      const float       *addFromVec,
      float             *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt  blockIndex      = index / contiguousBlockSize;
          dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = dftfe::utils::mult(a, s[blockIndex]);
          atomicAdd(&addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                              intraBlockIndex],
                    dftfe::utils::mult(addFromVec[index], coeff));
        }
    }

    __global__ void
    axpyStridedBlockAtomicAddDeviceKernel(
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const float        a,
      const float       *addFromVec,
      float             *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt  blockIndex      = index / contiguousBlockSize;
          dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = a;
          atomicAdd(&addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                              intraBlockIndex],
                    dftfe::utils::mult(addFromVec[index], coeff));
        }
    }

    __global__ void
    axpyStridedBlockAtomicAddDeviceKernel(
      const dftfe::uInt                       contiguousBlockSize,
      const dftfe::uInt                       numContiguousBlocks,
      const float                             a,
      const float                            *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt  blockIndex      = index / contiguousBlockSize;
          dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = dftfe::utils::mult(a, s[blockIndex]);
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
      const dftfe::uInt                       contiguousBlockSize,
      const dftfe::uInt                       numContiguousBlocks,
      const float                             a,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt  blockIndex      = index / contiguousBlockSize;
          dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = a;
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
      const dftfe::uInt                        contiguousBlockSize,
      const dftfe::uInt                        numContiguousBlocks,
      const dftfe::utils::deviceDoubleComplex *addFromVec,
      dftfe::utils::deviceDoubleComplex       *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt blockIndex      = index / contiguousBlockSize;
          dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
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
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double      *addFromVec,
      double            *addToVecReal,
      double            *addToVecImag,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt blockIndex      = index / contiguousBlockSize;
          dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
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
      const dftfe::uInt                        contiguousBlockSize,
      const dftfe::uInt                        numContiguousBlocks,
      const dftfe::utils::deviceDoubleComplex *addFromVec,
      double                                  *addToVecReal,
      double                                  *addToVecImag,
      const dftfe::uInt *addToVecStartingContiguousBlockIds)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt blockIndex      = index / contiguousBlockSize;
          dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
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

    __global__ void
    hadamardProductKernel(const dftfe::uInt vecSize,
                          const float      *xVec,
                          const float      *yVec,
                          float            *outputVec)
    {
      for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x; i < vecSize;
           i += blockDim.x * gridDim.x)
        {
          outputVec[i] = yVec[i] * xVec[i];
        }
    }

    __global__ void
    hadamardProductKernel(const dftfe::uInt vecSize,
                          const double     *xVec,
                          const double     *yVec,
                          double           *outputVec)
    {
      for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x; i < vecSize;
           i += blockDim.x * gridDim.x)
        {
          outputVec[i] = yVec[i] * xVec[i];
        }
    }

    __global__ void
    hadamardProductKernel(const dftfe::uInt                        vecSize,
                          const dftfe::utils::deviceDoubleComplex *xVec,
                          const dftfe::utils::deviceDoubleComplex *yVec,
                          dftfe::utils::deviceDoubleComplex       *outputVec)
    {
      for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x; i < vecSize;
           i += blockDim.x * gridDim.x)
        {
          outputVec[i].x = yVec[i].x * xVec[i].x - yVec[i].y * xVec[i].y;
          outputVec[i].y = yVec[i].x * xVec[i].y + yVec[i].y * xVec[i].x;
        }
    }

    __global__ void
    hadamardProductWithConjKernel(const dftfe::uInt vecSize,
                                  const float      *xVec,
                                  const float      *yVec,
                                  float            *outputVec)
    {
      for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x; i < vecSize;
           i += blockDim.x * gridDim.x)
        {
          outputVec[i] = yVec[i] * xVec[i];
        }
    }

    __global__ void
    hadamardProductWithConjKernel(const dftfe::uInt vecSize,
                                  const double     *xVec,
                                  const double     *yVec,
                                  double           *outputVec)
    {
      for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x; i < vecSize;
           i += blockDim.x * gridDim.x)
        {
          outputVec[i] = yVec[i] * xVec[i];
        }
    }

    __global__ void
    hadamardProductWithConjKernel(const dftfe::uInt vecSize,
                                  const dftfe::utils::deviceDoubleComplex *xVec,
                                  const dftfe::utils::deviceDoubleComplex *yVec,
                                  dftfe::utils::deviceDoubleComplex *outputVec)
    {
      for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x; i < vecSize;
           i += blockDim.x * gridDim.x)
        {
          outputVec[i].x = yVec[i].x * xVec[i].x + yVec[i].y * xVec[i].y;
          outputVec[i].y = yVec[i].y * xVec[i].x - yVec[i].x * xVec[i].y;
        }
    }

    template <typename ValueType0,
              typename ValueType1,
              typename ValueType2,
              typename ValueType3,
              typename ValueType4>
    __global__ void
    ApaBDDeviceKernel(const dftfe::uInt nRows,
                      const dftfe::uInt nCols,
                      const ValueType0  alpha,
                      const ValueType1 *A,
                      const ValueType2 *B,
                      const ValueType3 *D,
                      ValueType4       *C)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries  = nCols * nRows;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt      iRow   = index % nCols;
          const ValueType0 alphaD = alpha * D[iRow];
          dftfe::utils::copyValue(
            C + index,
            dftfe::utils::add(A[index], dftfe::utils::mult(B[index], alphaD)));
        }
    }

    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedBlockAxpyDeviceKernel(const dftfe::uInt contiguousBlockSize,
                                 const dftfe::uInt numContiguousBlocks,
                                 const ValueType2  a,
                                 const ValueType2 *s,
                                 const ValueType1 *addFromVec,
                                 ValueType1       *addToVec)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt      blockIndex = index / contiguousBlockSize;
          const ValueType2 coeff      = dftfe::utils::mult(a, s[blockIndex]);
          addToVec[index] =
            dftfe::utils::add(addToVec[index],
                              dftfe::utils::mult(addFromVec[index], coeff));
        }
    }

    template <typename ValueType1, typename ValueType2>
    __global__ void
    stridedBlockAxpByDeviceKernel(const dftfe::uInt contiguousBlockSize,
                                  const dftfe::uInt numContiguousBlocks,
                                  const ValueType2  a,
                                  const ValueType2  b,
                                  const ValueType2 *s,
                                  const ValueType1 *addFromVec,
                                  ValueType1       *addToVec)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt      blockIndex = index / contiguousBlockSize;
          const ValueType2 coeff      = dftfe::utils::mult(a, s[blockIndex]);
          addToVec[index] =
            dftfe::utils::add(dftfe::utils::mult(addToVec[index], b),
                              dftfe::utils::mult(addFromVec[index], coeff));
        }
    }

    template <>
    __global__ void
    stridedBlockAxpyDeviceKernel(
      const dftfe::uInt                       contiguousBlockSize,
      const dftfe::uInt                       numContiguousBlocks,
      const double                            a,
      const double                           *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt  blockIndex = index / contiguousBlockSize;
          const double coeff      = dftfe::utils::mult(a, s[blockIndex]);
          addToVec[index].x =
            dftfe::utils::add(addToVec[index].x,
                              dftfe::utils::mult(addFromVec[index].x, coeff));
          addToVec[index].y =
            dftfe::utils::add(addToVec[index].y,
                              dftfe::utils::mult(addFromVec[index].y, coeff));
        }
    }
    template <>
    __global__ void
    stridedBlockAxpByDeviceKernel(
      const dftfe::uInt                       contiguousBlockSize,
      const dftfe::uInt                       numContiguousBlocks,
      const double                            a,
      const double                            b,
      const double                           *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec)
    {
      const dftfe::uInt globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
      const dftfe::uInt numberEntries =
        numContiguousBlocks * contiguousBlockSize;

      for (dftfe::uInt index = globalThreadId; index < numberEntries;
           index += blockDim.x * gridDim.x)
        {
          dftfe::uInt  blockIndex = index / contiguousBlockSize;
          const double coeff      = dftfe::utils::mult(a, s[blockIndex]);
          addToVec[index].x =
            dftfe::utils::add(dftfe::utils::mult(addToVec[index].x, b),
                              dftfe::utils::mult(addFromVec[index].x, coeff));
          addToVec[index].y =
            dftfe::utils::add(dftfe::utils::mult(addToVec[index].y, b),
                              dftfe::utils::mult(addFromVec[index].y, coeff));
        }
    }



    __global__ void
    computeRightDiagonalScaleKernel(const double     *diagValues,
                                    double           *X,
                                    const dftfe::uInt N,
                                    const dftfe::uInt M)
    {
      const dftfe::uInt numEntries = N * M;
      for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEntries;
           i += blockDim.x * gridDim.x)
        {
          const dftfe::uInt idof = i / N;
          const dftfe::uInt ivec = i % N;

          *(X + N * idof + ivec) = *(X + N * idof + ivec) * diagValues[ivec];
        }
    }

    __global__ void
    computeRightDiagonalScaleKernel(const double *diagValues,
                                    dftfe::utils::deviceDoubleComplex *X,
                                    const dftfe::uInt                  N,
                                    const dftfe::uInt                  M)
    {
      const dftfe::uInt numEntries = N * M;
      for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEntries;
           i += blockDim.x * gridDim.x)
        {
          const dftfe::uInt idof = i / N;
          const dftfe::uInt ivec = i % N;

          *(X + N * idof + ivec) =
            dftfe::utils::mult(*(X + N * idof + ivec), diagValues[ivec]);
        }
    }

    __global__ void
    computeRightDiagonalScaleKernel(
      const dftfe::utils::deviceDoubleComplex *diagValues,
      dftfe::utils::deviceDoubleComplex       *X,
      const dftfe::uInt                        N,
      const dftfe::uInt                        M)
    {
      const dftfe::uInt numEntries = N * M;
      for (dftfe::Int i = blockIdx.x * blockDim.x + threadIdx.x; i < numEntries;
           i += blockDim.x * gridDim.x)
        {
          const dftfe::uInt idof = i / N;
          const dftfe::uInt ivec = i % N;

          *(X + N * idof + ivec) =
            dftfe::utils::mult(*(X + N * idof + ivec), diagValues[ivec]);
        }
    }



  } // namespace
} // namespace dftfe
