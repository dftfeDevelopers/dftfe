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
    DFTFE_CREATE_KERNEL(
      void,
      saddKernel,
      {
        for (dftfe::uInt idx = globalThreadId; idx < size;
             idx += nThreadsPerBlock * nThreadBlock)
          {
            y[idx] = beta * y[idx] - x[idx];
            x[idx] = 0;
          }
      },
      ValueType        *y,
      ValueType        *x,
      const ValueType   beta,
      const dftfe::uInt size);


    template <typename ValueTypeComplex, typename ValueTypeReal>
    DFTFE_CREATE_KERNEL(
      void,
      copyComplexArrToRealArrsDeviceKernel,
      {
        for (dftfe::uInt idx = globalThreadId; idx < size;
             idx += nThreadsPerBlock * nThreadBlock)
          {
            realArr[idx] = dftfe::utils::realPartDevice(complexArr[idx]);
            imagArr[idx] = dftfe::utils::imagPartDevice(complexArr[idx]);
          }
      },
      const dftfe::uInt       size,
      const ValueTypeComplex *complexArr,
      ValueTypeReal          *realArr,
      ValueTypeReal          *imagArr);

    template <typename ValueTypeComplex, typename ValueTypeReal>
    DFTFE_CREATE_KERNEL(
      void,
      copyRealArrsToComplexArrDeviceKernel,
      {
        for (dftfe::uInt idx = globalThreadId; idx < size;
             idx += nThreadsPerBlock * nThreadBlock)
          {
            complexArr[idx] =
              dftfe::utils::makeComplex(realArr[idx], imagArr[idx]);
          }
      },
      const dftfe::uInt    size,
      const ValueTypeReal *realArr,
      const ValueTypeReal *imagArr,
      ValueTypeComplex    *complexArr);


    template <typename ValueType1, typename ValueType2>
    DFTFE_CREATE_KERNEL(
      void,
      copyValueType1ArrToValueType2ArrDeviceKernel,
      {
        for (dftfe::uInt index = globalThreadId; index < size;
             index += nThreadsPerBlock * nThreadBlock)
          dftfe::utils::copyValue(valueType2Arr + index, valueType1Arr[index]);
      },
      const dftfe::uInt size,
      const ValueType1 *valueType1Arr,
      ValueType2       *valueType2Arr);

    template <typename ValueType1, typename ValueType2>
    DFTFE_CREATE_KERNEL(
      void,
      copyBlockDiagonalValueType1OffDiagonalValueType2FromValueType1ArrDeviceKernel,
      {
        const dftfe::uInt size = B * D;
        for (dftfe::uInt index = globalThreadId; index < size;
             index += nThreadsPerBlock * nThreadBlock)
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
      },
      const dftfe::uInt B,
      const dftfe::uInt DRem,
      const dftfe::uInt D,
      const ValueType1 *valueType1SrcArray,
      ValueType1       *valueType1DstArray,
      ValueType2       *valueType2DstArray);

    template <typename ValueType1, typename ValueType2>
    DFTFE_CREATE_KERNEL(
      void,
      stridedCopyToBlockDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt blockIndex = index / contiguousBlockSize;
            dftfe::uInt intraBlockIndex =
              index - blockIndex * contiguousBlockSize;
            dftfe::utils::copyValue(
              copyToVec + index,
              copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex]);
          }
      },
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const ValueType1  *copyFromVec,
      ValueType2        *copyToVec,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

    template <typename ValueType1, typename ValueType2>
    DFTFE_CREATE_KERNEL(
      void,
      stridedCopyToBlockDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt blockIndex = index / contiguousBlockSize;
            dftfe::uInt intraBlockIndex =
              index - blockIndex * contiguousBlockSize;
            dftfe::utils::copyValue(
              copyToVec + index,
              copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex + stratingVecId]);
          }
      },
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const dftfe::uInt  stratingVecId,
      const ValueType1  *copyFromVec,
      ValueType2        *copyToVec,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

    template <typename ValueType1, typename ValueType2>
    DFTFE_CREATE_KERNEL(
      void,
      stridedCopyToBlockScaleDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
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
      },
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const ValueType1   a,
      const ValueType1  *s,
      const ValueType2  *copyFromVec,
      ValueType2        *copyToVec,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds);



    template <typename ValueType1, typename ValueType2>
    DFTFE_CREATE_KERNEL(
      void,
      stridedCopyFromBlockDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt blockIndex = index / contiguousBlockSize;
            dftfe::uInt intraBlockIndex =
              index - blockIndex * contiguousBlockSize;
            dftfe::utils::copyValue(
              copyToVec + copyFromVecStartingContiguousBlockIds[blockIndex] +
                intraBlockIndex,
              copyFromVec[index]);
          }
      },
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const ValueType1  *copyFromVec,
      ValueType2        *copyToVec,
      const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

    template <typename ValueType1, typename ValueType2>
    DFTFE_CREATE_KERNEL(
      void,
      stridedCopyToBlockConstantStrideDeviceKernel,
      {
        const dftfe::uInt numberEntries = numBlocks * blockSizeTo;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt blockIndex      = index / blockSizeTo;
            dftfe::uInt intraBlockIndex = index - blockIndex * blockSizeTo;
            dftfe::utils::copyValue(copyToVec + index,
                                    copyFromVec[blockIndex * blockSizeFrom +
                                                startingId + intraBlockIndex]);
          }
      },
      const dftfe::uInt blockSizeTo,
      const dftfe::uInt blockSizeFrom,
      const dftfe::uInt numBlocks,
      const dftfe::uInt startingId,
      const ValueType1 *copyFromVec,
      ValueType2       *copyToVec);


    template <typename ValueType1, typename ValueType2>
    DFTFE_CREATE_KERNEL(
      void,
      stridedCopyFromBlockConstantStrideDeviceKernel,
      {
        const dftfe::uInt numberEntries = numBlocks * blockSizeFrom;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt blockIndex      = index / blockSizeFrom;
            dftfe::uInt intraBlockIndex = index - blockIndex * blockSizeFrom;
            dftfe::utils::copyValue(copyToVec + blockIndex * blockSizeTo +
                                      startingId + intraBlockIndex,
                                    copyFromVec[index]);
          }
      },
      const dftfe::uInt blockSizeTo,
      const dftfe::uInt blockSizeFrom,
      const dftfe::uInt numBlocks,
      const dftfe::uInt startingId,
      const ValueType1 *copyFromVec,
      ValueType2       *copyToVec);


    template <typename ValueType1, typename ValueType2>
    DFTFE_CREATE_KERNEL(
      void,
      stridedCopyConstantStrideDeviceKernel,
      {
        const dftfe::uInt numberEntries = numBlocks * blockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt blockIndex      = index / blockSize;
            dftfe::uInt intraBlockIndex = index - blockIndex * blockSize;
            dftfe::utils::copyValue(
              copyToVec + blockIndex * strideTo + startingToId +
                intraBlockIndex,
              copyFromVec[blockIndex * strideFrom + startingFromId +
                          intraBlockIndex]);
          }
      },
      const dftfe::uInt blockSize,
      const dftfe::uInt strideTo,
      const dftfe::uInt strideFrom,
      const dftfe::uInt numBlocks,
      const dftfe::uInt startingToId,
      const dftfe::uInt startingFromId,
      const ValueType1 *copyFromVec,
      ValueType2       *copyToVec);


    template <typename ValueType>
    DFTFE_CREATE_KERNEL(
      void,
      addVecOverContinuousIndexKernel,
      {
        const dftfe::uInt numberEntries = numContiguousBlocks;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            for (dftfe::uInt iBlock = 0; iBlock < contiguousBlockSize; iBlock++)
              {
                //                    output[index] +=
                //                    input1[index*contiguousBlockSize +
                //                    iBlock]*
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
      },
      const dftfe::uInt numContiguousBlocks,
      const dftfe::uInt contiguousBlockSize,
      const ValueType  *input1,
      const ValueType  *input2,
      ValueType        *output);



    // x=a*x, with inc=1
    template <typename ValueType1, typename ValueType2>
    DFTFE_CREATE_KERNEL(
      void,
      ascalDeviceKernel,
      {
        for (dftfe::uInt i = globalThreadId; i < n;
             i += nThreadsPerBlock * nThreadBlock)
          dftfe::utils::copyValue(x + i, dftfe::utils::mult(a, x[i]));
      },
      const dftfe::uInt n,
      ValueType1       *x,
      const ValueType2  a);



    // x[iblock*blocksize+intrablockindex]=a*s[iblock]*x[iblock*blocksize+intrablockindex]
    // strided block wise
    template <typename ValueType1, typename ValueType2>
    DFTFE_CREATE_KERNEL(
      void,
      stridedBlockScaleDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt blockIndex = index / contiguousBlockSize;
            dftfe::utils::copyValue(
              x + index,
              dftfe::utils::mult(dftfe::utils::mult(a, s[blockIndex]),
                                 x[index]));
          }
      },
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const ValueType1  a,
      const ValueType1 *s,
      ValueType2       *x);


    // x[iblock*blocksize+intrablockindex]=
    // beta[intrablockindex]*x[iblock*blocksize+intrablockindex] strided block
    // wise
    template <typename ValueType>
    DFTFE_CREATE_KERNEL(
      void,
      stridedBlockScaleColumnWiseKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt blockIndex = index / contiguousBlockSize;
            dftfe::uInt intrablockindex =
              index - blockIndex * contiguousBlockSize;
            dftfe::utils::copyValue(x + index,
                                    dftfe::utils::mult(beta[intrablockindex],
                                                       x[index]));
          }
      },
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const ValueType  *beta,
      ValueType        *x);


    // strided block wise
    template <typename ValueType>
    DFTFE_CREATE_KERNEL(
      void,
      stridedBlockScaleAndAddColumnWiseKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt blockIndex = index / contiguousBlockSize;
            dftfe::uInt intrablockindex =
              index - blockIndex * contiguousBlockSize;
            dftfe::utils::copyValue(
              y + index,
              dftfe::utils::add(
                y[index], dftfe::utils::mult(beta[intrablockindex], x[index])));
          }
      },
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const ValueType  *x,
      const ValueType  *beta,
      ValueType        *y);


    // strided block wise
    template <typename ValueType>
    DFTFE_CREATE_KERNEL(
      void,
      stridedBlockScaleAndAddTwoVecColumnWiseKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
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
      },
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const ValueType  *x,
      const ValueType  *alpha,
      const ValueType  *y,
      const ValueType  *beta,
      ValueType        *z);


    // y=a*x+b*y, with inc=1
    template <typename ValueType1, typename ValueType2>
    DFTFE_CREATE_KERNEL(
      void,
      axpbyDeviceKernel,
      {
        for (dftfe::uInt i = globalThreadId; i < n;
             i += nThreadsPerBlock * nThreadBlock)
          dftfe::utils::copyValue(y + i,
                                  dftfe::utils::add(dftfe::utils::mult(a, x[i]),
                                                    dftfe::utils::mult(b,
                                                                       y[i])));
      },
      const dftfe::uInt n,
      const ValueType1 *x,
      ValueType1       *y,
      const ValueType2  a,
      const ValueType2  b);



    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt blockIndex      = index / contiguousBlockSize;
            dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
            dftfe::utils::atomicAddWrapper(
              &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex],
              addFromVec[index]);
          }
      },
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double      *addFromVec,
      double            *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);



    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt  blockIndex      = index / contiguousBlockSize;
            dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
            const double coeff           = dftfe::utils::mult(a, s[blockIndex]);
            dftfe::utils::atomicAddWrapper(
              &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex],
              dftfe::utils::mult(addFromVec[index], coeff));
          }
      },
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double       a,
      const double      *s,
      const double      *addFromVec,
      double            *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);



    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt  blockIndex      = index / contiguousBlockSize;
            dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
            const double coeff           = a;
            dftfe::utils::atomicAddWrapper(
              &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex],
              dftfe::utils::mult(addFromVec[index], coeff));
          }
      },
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double       a,
      const double      *addFromVec,
      double            *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);



    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt  blockIndex      = index / contiguousBlockSize;
            dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
            const double coeff           = dftfe::utils::mult(a, s[blockIndex]);

            auto *add_real = reinterpret_cast<double *>(
              &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex]);
            auto *add_imag = add_real + 1;

            dftfe::utils::atomicAddWrapper(
              add_real,
              dftfe::utils::mult(
                dftfe::utils::realPartDevice(addFromVec[index]), coeff));
            dftfe::utils::atomicAddWrapper(
              add_imag,
              dftfe::utils::mult(
                dftfe::utils::imagPartDevice(addFromVec[index]), coeff));
          }
      },
      const dftfe::uInt                        contiguousBlockSize,
      const dftfe::uInt                        numContiguousBlocks,
      const double                             a,
      const double                            *s,
      const dftfe::utils::deviceDoubleComplex *addFromVec,
      dftfe::utils::deviceDoubleComplex       *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);



    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt  blockIndex      = index / contiguousBlockSize;
            dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
            const double coeff           = a;

            auto *add_real = reinterpret_cast<double *>(
              &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex]);
            auto *add_imag = add_real + 1;

            dftfe::utils::atomicAddWrapper(
              add_real,
              dftfe::utils::mult(
                dftfe::utils::realPartDevice(addFromVec[index]), coeff));
            dftfe::utils::atomicAddWrapper(
              add_imag,
              dftfe::utils::mult(
                dftfe::utils::imagPartDevice(addFromVec[index]), coeff));
          }
      },
      const dftfe::uInt                        contiguousBlockSize,
      const dftfe::uInt                        numContiguousBlocks,
      const double                             a,
      const dftfe::utils::deviceDoubleComplex *addFromVec,
      dftfe::utils::deviceDoubleComplex       *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);



    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt  blockIndex      = index / contiguousBlockSize;
            dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
            const double coeff           = dftfe::utils::mult(a, s[blockIndex]);
            dftfe::utils::atomicAddWrapper(
              &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex],
              dftfe::utils::mult(addFromVec[index], coeff));
          }
      },
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double       a,
      const double      *s,
      const float       *addFromVec,
      double            *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);



    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt  blockIndex      = index / contiguousBlockSize;
            dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
            const double coeff           = a;
            dftfe::utils::atomicAddWrapper(
              &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex],
              dftfe::utils::mult(addFromVec[index], coeff));
          }
      },
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double       a,
      const float       *addFromVec,
      double            *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);



    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt  blockIndex      = index / contiguousBlockSize;
            dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
            const double coeff           = dftfe::utils::mult(a, s[blockIndex]);

            auto *add_real = reinterpret_cast<double *>(
              &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex]);
            auto *add_imag = add_real + 1;

            dftfe::utils::atomicAddWrapper(
              add_real,
              dftfe::utils::mult(
                dftfe::utils::realPartDevice(addFromVec[index]), coeff));
            dftfe::utils::atomicAddWrapper(
              add_imag,
              dftfe::utils::mult(
                dftfe::utils::imagPartDevice(addFromVec[index]), coeff));
          }
      },
      const dftfe::uInt                       contiguousBlockSize,
      const dftfe::uInt                       numContiguousBlocks,
      const double                            a,
      const double                           *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceDoubleComplex      *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);


    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt  blockIndex      = index / contiguousBlockSize;
            dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
            const double coeff           = a;

            auto *add_real = reinterpret_cast<double *>(
              &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex]);
            auto *add_imag = add_real + 1;

            dftfe::utils::atomicAddWrapper(
              add_real,
              dftfe::utils::mult(
                dftfe::utils::realPartDevice(addFromVec[index]), coeff));
            dftfe::utils::atomicAddWrapper(
              add_imag,
              dftfe::utils::mult(
                dftfe::utils::imagPartDevice(addFromVec[index]), coeff));
          }
      },
      const dftfe::uInt                       contiguousBlockSize,
      const dftfe::uInt                       numContiguousBlocks,
      const double                            a,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceDoubleComplex      *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);


    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt  blockIndex      = index / contiguousBlockSize;
            dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
            const double coeff           = dftfe::utils::mult(a, s[blockIndex]);
            dftfe::utils::atomicAddWrapper(
              &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex],
              dftfe::utils::mult(addFromVec[index], coeff));
          }
      },
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double       a,
      const double      *s,
      const float       *addFromVec,
      float             *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);


    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt  blockIndex      = index / contiguousBlockSize;
            dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
            const double coeff           = a;
            dftfe::utils::atomicAddWrapper(
              &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex],
              dftfe::utils::mult(addFromVec[index], coeff));
          }
      },
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double       a,
      const float       *addFromVec,
      float             *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);



    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt  blockIndex      = index / contiguousBlockSize;
            dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
            const double coeff           = dftfe::utils::mult(a, s[blockIndex]);

            auto *add_real = reinterpret_cast<float *>(
              &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex]);
            auto *add_imag = add_real + 1;

            dftfe::utils::atomicAddWrapper(
              add_real,
              dftfe::utils::mult(
                dftfe::utils::realPartDevice(addFromVec[index]), coeff));
            dftfe::utils::atomicAddWrapper(
              add_imag,
              dftfe::utils::mult(
                dftfe::utils::imagPartDevice(addFromVec[index]), coeff));
          }
      },
      const dftfe::uInt                       contiguousBlockSize,
      const dftfe::uInt                       numContiguousBlocks,
      const double                            a,
      const double                           *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);



    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt  blockIndex      = index / contiguousBlockSize;
            dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
            const double coeff           = a;

            auto *add_real = reinterpret_cast<float *>(
              &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex]);
            auto *add_imag = add_real + 1;

            dftfe::utils::atomicAddWrapper(
              add_real,
              dftfe::utils::mult(
                dftfe::utils::realPartDevice(addFromVec[index]), coeff));
            dftfe::utils::atomicAddWrapper(
              add_imag,
              dftfe::utils::mult(
                dftfe::utils::imagPartDevice(addFromVec[index]), coeff));
          }
      },
      const dftfe::uInt                       contiguousBlockSize,
      const dftfe::uInt                       numContiguousBlocks,
      const double                            a,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);



    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt  blockIndex      = index / contiguousBlockSize;
            dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
            const double coeff           = dftfe::utils::mult(a, s[blockIndex]);
            dftfe::utils::atomicAddWrapper(
              &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex],
              dftfe::utils::mult(addFromVec[index], coeff));
          }
      },
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const float        a,
      const float       *s,
      const float       *addFromVec,
      float             *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);



    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt  blockIndex      = index / contiguousBlockSize;
            dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
            const double coeff           = a;
            dftfe::utils::atomicAddWrapper(
              &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex],
              dftfe::utils::mult(addFromVec[index], coeff));
          }
      },
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const float        a,
      const float       *addFromVec,
      float             *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);



    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt  blockIndex      = index / contiguousBlockSize;
            dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
            const double coeff           = dftfe::utils::mult(a, s[blockIndex]);

            auto *add_real = reinterpret_cast<float *>(
              &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex]);
            auto *add_imag = add_real + 1;

            dftfe::utils::atomicAddWrapper(
              add_real,
              dftfe::utils::mult(
                dftfe::utils::realPartDevice(addFromVec[index]), coeff));
            dftfe::utils::atomicAddWrapper(
              add_imag,
              dftfe::utils::mult(
                dftfe::utils::imagPartDevice(addFromVec[index]), coeff));
          }
      },
      const dftfe::uInt                       contiguousBlockSize,
      const dftfe::uInt                       numContiguousBlocks,
      const float                             a,
      const float                            *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);



    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt  blockIndex      = index / contiguousBlockSize;
            dftfe::uInt  intraBlockIndex = index % contiguousBlockSize;
            const double coeff           = a;

            auto *add_real = reinterpret_cast<float *>(
              &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex]);
            auto *add_imag = add_real + 1;

            dftfe::utils::atomicAddWrapper(
              add_real,
              dftfe::utils::mult(
                dftfe::utils::realPartDevice(addFromVec[index]), coeff));
            dftfe::utils::atomicAddWrapper(
              add_imag,
              dftfe::utils::mult(
                dftfe::utils::imagPartDevice(addFromVec[index]), coeff));
          }
      },
      const dftfe::uInt                       contiguousBlockSize,
      const dftfe::uInt                       numContiguousBlocks,
      const float                             a,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);



    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt blockIndex      = index / contiguousBlockSize;
            dftfe::uInt intraBlockIndex = index % contiguousBlockSize;

            auto *add_real = reinterpret_cast<double *>(
              &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex]);
            auto *add_imag = add_real + 1;

            dftfe::utils::atomicAddWrapper(
              add_real, dftfe::utils::realPartDevice(addFromVec[index]));
            dftfe::utils::atomicAddWrapper(
              add_imag, dftfe::utils::imagPartDevice(addFromVec[index]));
          }
      },
      const dftfe::uInt                        contiguousBlockSize,
      const dftfe::uInt                        numContiguousBlocks,
      const dftfe::utils::deviceDoubleComplex *addFromVec,
      dftfe::utils::deviceDoubleComplex       *addToVec,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);



    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt blockIndex      = index / contiguousBlockSize;
            dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
            dftfe::utils::atomicAddWrapper(
              &addToVecReal[addToVecStartingContiguousBlockIds[blockIndex] +
                            intraBlockIndex],
              addFromVec[index]);
            dftfe::utils::atomicAddWrapper(
              &addToVecImag[addToVecStartingContiguousBlockIds[blockIndex] +
                            intraBlockIndex],
              addFromVec[index]);
          }
      },
      const dftfe::uInt  contiguousBlockSize,
      const dftfe::uInt  numContiguousBlocks,
      const double      *addFromVec,
      double            *addToVecReal,
      double            *addToVecImag,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);



    DFTFE_CREATE_KERNEL(
      void,
      axpyStridedBlockAtomicAddDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt blockIndex      = index / contiguousBlockSize;
            dftfe::uInt intraBlockIndex = index % contiguousBlockSize;
            dftfe::utils::atomicAddWrapper(
              &addToVecReal[addToVecStartingContiguousBlockIds[blockIndex] +
                            intraBlockIndex],
              dftfe::utils::realPartDevice(addFromVec[index]));
            dftfe::utils::atomicAddWrapper(
              &addToVecImag[addToVecStartingContiguousBlockIds[blockIndex] +
                            intraBlockIndex],
              dftfe::utils::imagPartDevice(addFromVec[index]));
          }
      },
      const dftfe::uInt                        contiguousBlockSize,
      const dftfe::uInt                        numContiguousBlocks,
      const dftfe::utils::deviceDoubleComplex *addFromVec,
      double                                  *addToVecReal,
      double                                  *addToVecImag,
      const dftfe::uInt *addToVecStartingContiguousBlockIds);



    DFTFE_CREATE_KERNEL(
      void,
      hadamardProductKernel,
      {
        for (dftfe::Int i = globalThreadId; i < vecSize;
             i += nThreadsPerBlock * nThreadBlock)
          {
            outputVec[i] = yVec[i] * xVec[i];
          }
      },
      const dftfe::uInt vecSize,
      const float      *xVec,
      const float      *yVec,
      float            *outputVec);



    DFTFE_CREATE_KERNEL(
      void,
      hadamardProductKernel,
      {
        for (dftfe::Int i = globalThreadId; i < vecSize;
             i += nThreadsPerBlock * nThreadBlock)
          {
            outputVec[i] = yVec[i] * xVec[i];
          }
      },
      const dftfe::uInt vecSize,
      const double     *xVec,
      const double     *yVec,
      double           *outputVec);



    DFTFE_CREATE_KERNEL(
      void,
      hadamardProductKernel,
      {
        for (dftfe::Int i = globalThreadId; i < vecSize;
             i += nThreadsPerBlock * nThreadBlock)
          {
            outputVec[i] = dftfe::utils::makeComplex(
              dftfe::utils::realPartDevice(yVec[i]) *
                  dftfe::utils::realPartDevice(xVec[i]) -
                dftfe::utils::imagPartDevice(yVec[i]) *
                  dftfe::utils::imagPartDevice(xVec[i]),
              dftfe::utils::realPartDevice(yVec[i]) *
                  dftfe::utils::imagPartDevice(xVec[i]) +
                dftfe::utils::imagPartDevice(yVec[i]) *
                  dftfe::utils::realPartDevice(xVec[i]));
          }
      },
      const dftfe::uInt                        vecSize,
      const dftfe::utils::deviceDoubleComplex *xVec,
      const dftfe::utils::deviceDoubleComplex *yVec,
      dftfe::utils::deviceDoubleComplex       *outputVec);



    DFTFE_CREATE_KERNEL(
      void,
      hadamardProductWithConjKernel,
      {
        for (dftfe::Int i = globalThreadId; i < vecSize;
             i += nThreadsPerBlock * nThreadBlock)
          {
            outputVec[i] = yVec[i] * xVec[i];
          }
      },
      const dftfe::uInt vecSize,
      const float      *xVec,
      const float      *yVec,
      float            *outputVec);



    DFTFE_CREATE_KERNEL(
      void,
      hadamardProductWithConjKernel,
      {
        for (dftfe::Int i = globalThreadId; i < vecSize;
             i += nThreadsPerBlock * nThreadBlock)
          {
            outputVec[i] = yVec[i] * xVec[i];
          }
      },
      const dftfe::uInt vecSize,
      const double     *xVec,
      const double     *yVec,
      double           *outputVec);



    DFTFE_CREATE_KERNEL(
      void,
      hadamardProductWithConjKernel,
      {
        for (dftfe::Int i = globalThreadId; i < vecSize;
             i += nThreadsPerBlock * nThreadBlock)
          {
            outputVec[i] = dftfe::utils::makeComplex(
              dftfe::utils::realPartDevice(yVec[i]) *
                  dftfe::utils::realPartDevice(xVec[i]) +
                dftfe::utils::imagPartDevice(yVec[i]) *
                  dftfe::utils::imagPartDevice(xVec[i]),
              dftfe::utils::imagPartDevice(yVec[i]) *
                  dftfe::utils::realPartDevice(xVec[i]) -
                dftfe::utils::realPartDevice(yVec[i]) *
                  dftfe::utils::imagPartDevice(xVec[i]));
          }
      },
      const dftfe::uInt                        vecSize,
      const dftfe::utils::deviceDoubleComplex *xVec,
      const dftfe::utils::deviceDoubleComplex *yVec,
      dftfe::utils::deviceDoubleComplex       *outputVec);


    template <typename ValueType0,
              typename ValueType1,
              typename ValueType2,
              typename ValueType3,
              typename ValueType4>
    DFTFE_CREATE_KERNEL(
      void,
      ApaBDDeviceKernel,
      {
        const dftfe::uInt numberEntries = nCols * nRows;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt      iRow   = index % nCols;
            const ValueType0 alphaD = alpha * D[iRow];
            dftfe::utils::copyValue(
              C + index,
              dftfe::utils::add(A[index],
                                dftfe::utils::mult(B[index], alphaD)));
          }
      },
      const dftfe::uInt nRows,
      const dftfe::uInt nCols,
      const ValueType0  alpha,
      const ValueType1 *A,
      const ValueType2 *B,
      const ValueType3 *D,
      ValueType4       *C);


    template <typename ValueType1, typename ValueType2>
    DFTFE_CREATE_KERNEL(
      void,
      stridedBlockAxpyDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt      blockIndex = index / contiguousBlockSize;
            const ValueType2 coeff      = dftfe::utils::mult(a, s[blockIndex]);
            addToVec[index] =
              dftfe::utils::add(addToVec[index],
                                dftfe::utils::mult(addFromVec[index], coeff));
          }
      },
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const ValueType2  a,
      const ValueType2 *s,
      const ValueType1 *addFromVec,
      ValueType1       *addToVec);


    template <typename ValueType1, typename ValueType2>
    DFTFE_CREATE_KERNEL(
      void,
      stridedBlockAxpByDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt      blockIndex = index / contiguousBlockSize;
            const ValueType2 coeff      = dftfe::utils::mult(a, s[blockIndex]);
            addToVec[index] =
              dftfe::utils::add(dftfe::utils::mult(addToVec[index], b),
                                dftfe::utils::mult(addFromVec[index], coeff));
          }
      },
      const dftfe::uInt contiguousBlockSize,
      const dftfe::uInt numContiguousBlocks,
      const ValueType2  a,
      const ValueType2  b,
      const ValueType2 *s,
      const ValueType1 *addFromVec,
      ValueType1       *addToVec);


    template <>
    DFTFE_CREATE_KERNEL(
      void,
      stridedBlockAxpyDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt  blockIndex = index / contiguousBlockSize;
            const double coeff      = dftfe::utils::mult(a, s[blockIndex]);
            addToVec[index]         = dftfe::utils::makeComplex(
              (float)dftfe::utils::add(
                dftfe::utils::realPartDevice(addToVec[index]),
                dftfe::utils::mult(
                  dftfe::utils::realPartDevice(addFromVec[index]), coeff)),
              (float)dftfe::utils::add(
                dftfe::utils::imagPartDevice(addToVec[index]),
                dftfe::utils::mult(
                  dftfe::utils::imagPartDevice(addFromVec[index]), coeff)));
          }
      },
      const dftfe::uInt                       contiguousBlockSize,
      const dftfe::uInt                       numContiguousBlocks,
      const double                            a,
      const double                           *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec);

    template <>
    DFTFE_CREATE_KERNEL(
      void,
      stridedBlockAxpByDeviceKernel,
      {
        const dftfe::uInt numberEntries =
          numContiguousBlocks * contiguousBlockSize;

        for (dftfe::uInt index = globalThreadId; index < numberEntries;
             index += nThreadsPerBlock * nThreadBlock)
          {
            dftfe::uInt  blockIndex = index / contiguousBlockSize;
            const double coeff      = dftfe::utils::mult(a, s[blockIndex]);
            addToVec[index]         = dftfe::utils::makeComplex(
              (float)dftfe::utils::add(
                dftfe::utils::mult(
                  dftfe::utils::realPartDevice(addToVec[index]), b),
                dftfe::utils::mult(
                  dftfe::utils::realPartDevice(addFromVec[index]), coeff)),
              (float)dftfe::utils::add(
                dftfe::utils::mult(
                  dftfe::utils::imagPartDevice(addToVec[index]), b),
                dftfe::utils::mult(
                  dftfe::utils::imagPartDevice(addFromVec[index]), coeff)));
          }
      },
      const dftfe::uInt                       contiguousBlockSize,
      const dftfe::uInt                       numContiguousBlocks,
      const double                            a,
      const double                            b,
      const double                           *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec);



    DFTFE_CREATE_KERNEL(
      void,
      computeRightDiagonalScaleKernel,
      {
        const dftfe::uInt numEntries = N * M;
        for (dftfe::Int i = globalThreadId; i < numEntries;
             i += nThreadsPerBlock * nThreadBlock)
          {
            const dftfe::uInt idof = i / N;
            const dftfe::uInt ivec = i % N;

            *(X + N * idof + ivec) = *(X + N * idof + ivec) * diagValues[ivec];
          }
      },
      const double     *diagValues,
      double           *X,
      const dftfe::uInt N,
      const dftfe::uInt M);



    DFTFE_CREATE_KERNEL(
      void,
      computeRightDiagonalScaleKernel,
      {
        const dftfe::uInt numEntries = N * M;
        for (dftfe::Int i = globalThreadId; i < numEntries;
             i += nThreadsPerBlock * nThreadBlock)
          {
            const dftfe::uInt idof = i / N;
            const dftfe::uInt ivec = i % N;

            *(X + N * idof + ivec) =
              dftfe::utils::mult(*(X + N * idof + ivec), diagValues[ivec]);
          }
      },
      const double                      *diagValues,
      dftfe::utils::deviceDoubleComplex *X,
      const dftfe::uInt                  N,
      const dftfe::uInt                  M);



    DFTFE_CREATE_KERNEL(
      void,
      computeRightDiagonalScaleKernel,
      {
        const dftfe::uInt numEntries = N * M;
        for (dftfe::Int i = globalThreadId; i < numEntries;
             i += nThreadsPerBlock * nThreadBlock)
          {
            const dftfe::uInt idof = i / N;
            const dftfe::uInt ivec = i % N;

            *(X + N * idof + ivec) =
              dftfe::utils::mult(*(X + N * idof + ivec), diagValues[ivec]);
          }
      },
      const dftfe::utils::deviceDoubleComplex *diagValues,
      dftfe::utils::deviceDoubleComplex       *X,
      const dftfe::uInt                        N,
      const dftfe::uInt                        M);



  } // namespace
} // namespace dftfe
