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
    void
    saddKernel(sycl::nd_item<1>   ind,
               ValueType         *y,
               ValueType         *x,
               const ValueType    beta,
               const unsigned int size)
    {
      const unsigned int globalId     = ind.get_global_id(0);
      unsigned int       n_workgroups = ind.get_group_range(0);
      unsigned int       n_workitems  = ind.get_local_range(0);

      for (unsigned int idx = globalId; idx < size;
           idx += n_workgroups * n_workitems)
        {
          y[idx] = beta * y[idx] - x[idx];
          x[idx] = 0;
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    ascalDeviceKernel(sycl::nd_item<1>   ind,
                      const unsigned int n,
                      ValueType1        *x,
                      const ValueType2   a)
    {
      for (unsigned int i = ind.get_global_id(0); i < n;
           i += ind.get_group_range(0) * ind.get_local_range(0))
        dftfe::utils::copyValue(x + i, dftfe::utils::mult(a, x[i]));
    }


    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    copyComplexArrToRealArrsDeviceKernel(sycl::nd_item<1>        ind,
                                         const unsigned int      size,
                                         const ValueTypeComplex *complexArr,
                                         ValueTypeReal          *realArr,
                                         ValueTypeReal          *imagArr)
    {
      const unsigned int globalId     = ind.get_global_id(0);
      unsigned int       n_workgroups = ind.get_group_range(0);
      unsigned int       n_workitems  = ind.get_local_range(0);

      for (unsigned int idx = globalId; idx < size;
           idx += n_workgroups * n_workitems)
        {
          realArr[idx] = complexArr[idx].real();
          imagArr[idx] = complexArr[idx].imag();
        }
    }

    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    copyRealArrsToComplexArrDeviceKernel(sycl::nd_item<1>     ind,
                                         const unsigned int   size,
                                         const ValueTypeReal *realArr,
                                         const ValueTypeReal *imagArr,
                                         ValueTypeComplex    *complexArr)
    {
      const unsigned int globalId     = ind.get_global_id(0);
      unsigned int       n_workgroups = ind.get_group_range(0);
      unsigned int       n_workitems  = ind.get_local_range(0);

      for (unsigned int idx = globalId; idx < size;
           idx += n_workgroups * n_workitems)
        {
          complexArr[idx].real(realArr[idx]);
          complexArr[idx].imag(imagArr[idx]);
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    copyValueType1ArrToValueType2ArrDeviceKernel(
      sycl::nd_item<1>   ind,
      const unsigned int size,
      const ValueType1  *valueType1Arr,
      ValueType2        *valueType2Arr)
    {
      const unsigned int globalId     = ind.get_global_id(0);
      unsigned int       n_workgroups = ind.get_group_range(0);
      unsigned int       n_workitems  = ind.get_local_range(0);

      for (unsigned int index = globalId; index < size;
           index += n_workgroups * n_workitems)
        dftfe::utils::copyValue(valueType2Arr + index, valueType1Arr[index]);
    }

    template <typename ValueType0,
              typename ValueType1,
              typename ValueType2,
              typename ValueType3,
              typename ValueType4>
    void
    ApaBDDeviceKernel(sycl::nd_item<1>   ind,
                      const unsigned int nRows,
                      const unsigned int nCols,
                      const ValueType0   alpha,
                      const ValueType1  *A,
                      const ValueType2  *B,
                      const ValueType3  *D,
                      ValueType4        *C)
    {
      unsigned int globalThreadId = ind.get_global_id(0);
      unsigned int numberEntries  = nCols * nRows;
      unsigned int n_workgroups   = ind.get_group_range(0);
      unsigned int n_workitems    = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int     iRow   = index % nCols;
          const ValueType0 alphaD = alpha * D[iRow];
          dftfe::utils::copyValue(
            C + index,
            dftfe::utils::add(A[index], dftfe::utils::mult(B[index], alphaD)));
        }
    }


    template <typename ValueType1, typename ValueType2>
    void
    stridedCopyToBlockDeviceKernel(
      sycl::nd_item<1>    ind,
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const ValueType1   *copyFromVec,
      ValueType2         *copyToVec,
      const unsigned int *copyFromVecStartingContiguousBlockIds)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          unsigned int intraBlockIndex =
            index - blockIndex * contiguousBlockSize;
          dftfe::utils::copyValue(
            copyToVec + index,
            copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex]);
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    stridedCopyToBlockDeviceKernel(
      sycl::nd_item<1>    ind,
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const unsigned int  stratingVecId,
      const ValueType1   *copyFromVec,
      ValueType2         *copyToVec,
      const unsigned int *copyFromVecStartingContiguousBlockIds)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups = ind.get_group_range(0);
      unsigned int n_workitems  = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          unsigned int intraBlockIndex =
            index - blockIndex * contiguousBlockSize;
          dftfe::utils::copyValue(
            copyToVec + index,
            copyFromVec[copyFromVecStartingContiguousBlockIds[blockIndex] +
                        intraBlockIndex + stratingVecId]);
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    stridedCopyToBlockScaleDeviceKernel(
      sycl::nd_item<1>    ind,
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const ValueType1    a,
      const ValueType1   *s,
      const ValueType2   *copyFromVec,
      ValueType2         *copyToVec,
      const unsigned int *copyFromVecStartingContiguousBlockIds)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          unsigned int intraBlockIndex =
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
    void
    stridedCopyFromBlockDeviceKernel(
      sycl::nd_item<1>    ind,
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const ValueType1   *copyFromVec,
      ValueType2         *copyToVec,
      const unsigned int *copyFromVecStartingContiguousBlockIds)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          unsigned int intraBlockIndex =
            index - blockIndex * contiguousBlockSize;
          dftfe::utils::copyValue(
            copyToVec + copyFromVecStartingContiguousBlockIds[blockIndex] +
              intraBlockIndex,
            copyFromVec[index]);
        }
    }


    template <typename ValueType1, typename ValueType2>
    void
    stridedCopyToBlockConstantStrideDeviceKernel(
      sycl::nd_item<1>   ind,
      const unsigned int blockSizeTo,
      const unsigned int blockSizeFrom,
      const unsigned int numBlocks,
      const unsigned int startingId,
      const ValueType1  *copyFromVec,
      ValueType2        *copyToVec)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numBlocks * blockSizeTo;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / blockSizeTo;
          unsigned int intraBlockIndex = index - blockIndex * blockSizeTo;
          dftfe::utils::copyValue(copyToVec + index,
                                  copyFromVec[blockIndex * blockSizeFrom +
                                              startingId + intraBlockIndex]);
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    stridedCopyFromBlockConstantStrideDeviceKernel(
      sycl::nd_item<1>   ind,
      const unsigned int blockSizeTo,
      const unsigned int blockSizeFrom,
      const unsigned int numBlocks,
      const unsigned int startingId,
      const ValueType1  *copyFromVec,
      ValueType2        *copyToVec)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numBlocks * blockSizeFrom;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / blockSizeFrom;
          unsigned int intraBlockIndex = index - blockIndex * blockSizeFrom;
          dftfe::utils::copyValue(copyToVec + blockIndex * blockSizeTo +
                                    startingId + intraBlockIndex,
                                  copyFromVec[index]);
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    stridedCopyConstantStrideDeviceKernel(sycl::nd_item<1>   ind,
                                          const unsigned int blockSize,
                                          const unsigned int strideTo,
                                          const unsigned int strideFrom,
                                          const unsigned int numBlocks,
                                          const unsigned int startingToId,
                                          const unsigned int startingFromId,
                                          const ValueType1  *copyFromVec,
                                          ValueType2        *copyToVec)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numBlocks * blockSize;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / blockSize;
          unsigned int intraBlockIndex = index - blockIndex * blockSize;
          dftfe::utils::copyValue(
            copyToVec + blockIndex * strideTo + startingToId + intraBlockIndex,
            copyFromVec[blockIndex * strideFrom + startingFromId +
                        intraBlockIndex]);
        }
    }


    // x=a*x, with inc=1
    template <typename ValueType1, typename ValueType2>
    void
    xscalDeviceKernel(sycl::nd_item<1>   ind,
                      const unsigned int n,
                      ValueType1        *x,
                      const ValueType2   a)
    {
      unsigned int global_id    = ind.get_global_id(0);
      unsigned int n_workgroups = ind.get_group_range(0);
      unsigned int n_workitems  = ind.get_local_range(0);
      for (unsigned int i = global_id; i < n; i += n_workgroups * n_workitems)
        dftfe::utils::copyValue(x + i, dftfe::utils::mult(a, x[i]));
    }


    // x[iblock*blocksize+intrablockindex]=a*s[iblock]*x[iblock*blocksize+intrablockindex]
    // strided block wise
    template <typename ValueType1, typename ValueType2>
    void
    stridedBlockScaleDeviceKernel(sycl::nd_item<1>   ind,
                                  const unsigned int contiguousBlockSize,
                                  const unsigned int numContiguousBlocks,
                                  const ValueType1   a,
                                  const ValueType1  *s,
                                  ValueType2        *x)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          dftfe::utils::copyValue(
            x + index,
            dftfe::utils::mult(dftfe::utils::mult(a, s[blockIndex]), x[index]));
        }
    }

    // y=a*x+b*y, with inc=1
    template <typename ValueType1, typename ValueType2>
    void
    axpbyDeviceKernel(sycl::nd_item<1>   ind,
                      const unsigned int n,
                      const ValueType1  *x,
                      ValueType1        *y,
                      const ValueType2   a,
                      const ValueType2   b)
    {
      unsigned int global_id    = ind.get_global_id(0);
      unsigned int n_workgroups = ind.get_group_range(0);
      unsigned int n_workitems  = ind.get_local_range(0);
      for (unsigned int i = global_id; i < n; i += n_workgroups * n_workitems)
        dftfe::utils::copyValue(y + i,
                                dftfe::utils::add(dftfe::utils::mult(a, x[i]),
                                                  dftfe::utils::mult(b, y[i])));
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>    ind,
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const double       *addFromVec,
      double             *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          unsigned int intraBlockIndex =
            index - blockIndex * contiguousBlockSize;

          auto atomic_add =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                       intraBlockIndex]);
          atomic_add += addFromVec[index];
        }
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>    ind,
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const double        a,
      const double       *s,
      const double       *addFromVec,
      double             *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = dftfe::utils::mult(a, s[blockIndex]);

          auto atomic_add =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                       intraBlockIndex]);
          atomic_add += dftfe::utils::mult(addFromVec[index], coeff);
        }
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>            ind,
      const unsigned int          contiguousBlockSize,
      const unsigned int          numContiguousBlocks,
      const double                a,
      const double               *s,
      const std::complex<double> *addFromVec,
      std::complex<double>       *addToVec,
      const unsigned int         *addToVecStartingContiguousBlockIds)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = dftfe::utils::mult(a, s[blockIndex]);

          auto atomic_add_real =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              reinterpret_cast<double *>(
                &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex])[0]);
          atomic_add_real +=
            dftfe::utils::mult(addFromVec[index].real(), coeff);

          auto atomic_add_imag =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              reinterpret_cast<double *>(
                &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex])[1]);
          atomic_add_imag +=
            dftfe::utils::mult(addFromVec[index].imag(), coeff);
        }
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>    ind,
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const double        a,
      const double       *s,
      const float        *addFromVec,
      double             *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = dftfe::utils::mult(a, s[blockIndex]);

          auto atomic_add =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                       intraBlockIndex]);
          atomic_add += dftfe::utils::mult(addFromVec[index], coeff);
        }
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>                        ind,
      const unsigned int                      contiguousBlockSize,
      const unsigned int                      numContiguousBlocks,
      const double                            a,
      const double                           *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceDoubleComplex      *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = dftfe::utils::mult(a, s[blockIndex]);

          auto atomic_add_real =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              reinterpret_cast<double *>(
                &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex])[0]);
          atomic_add_real +=
            dftfe::utils::mult(addFromVec[index].real(), coeff);

          auto atomic_add_imag =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              reinterpret_cast<double *>(
                &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex])[1]);
          atomic_add_imag +=
            dftfe::utils::mult(addFromVec[index].imag(), coeff);
        }
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>    ind,
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const double        a,
      const double       *s,
      const float        *addFromVec,
      float              *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = dftfe::utils::mult(a, s[blockIndex]);

          auto atomic_add =
            sycl::atomic_ref<float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                       intraBlockIndex]);
          atomic_add += dftfe::utils::mult(addFromVec[index], coeff);
        }
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>                        ind,
      const unsigned int                      contiguousBlockSize,
      const unsigned int                      numContiguousBlocks,
      const double                            a,
      const double                           *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = dftfe::utils::mult(a, s[blockIndex]);

          auto atomic_add_real =
            sycl::atomic_ref<float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              reinterpret_cast<float *>(
                &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex])[0]);
          atomic_add_real +=
            dftfe::utils::mult(addFromVec[index].real(), coeff);

          auto atomic_add_imag =
            sycl::atomic_ref<float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              reinterpret_cast<float *>(
                &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex])[1]);
          atomic_add_imag +=
            dftfe::utils::mult(addFromVec[index].imag(), coeff);
        }
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>    ind,
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const float         a,
      const float        *s,
      const float        *addFromVec,
      float              *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = dftfe::utils::mult(a, s[blockIndex]);

          auto atomic_add =
            sycl::atomic_ref<float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                       intraBlockIndex]);
          atomic_add += dftfe::utils::mult(addFromVec[index], coeff);
        }
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>                        ind,
      const unsigned int                      contiguousBlockSize,
      const unsigned int                      numContiguousBlocks,
      const float                             a,
      const float                            *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = dftfe::utils::mult(a, s[blockIndex]);

          auto atomic_add_real =
            sycl::atomic_ref<float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              reinterpret_cast<float *>(
                &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex])[0]);
          atomic_add_real +=
            dftfe::utils::mult(addFromVec[index].real(), coeff);

          auto atomic_add_imag =
            sycl::atomic_ref<float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              reinterpret_cast<float *>(
                &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex])[1]);
          atomic_add_imag +=
            dftfe::utils::mult(addFromVec[index].imag(), coeff);
        }
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>            ind,
      const unsigned int          contiguousBlockSize,
      const unsigned int          numContiguousBlocks,
      const std::complex<double> *addFromVec,
      std::complex<double>       *addToVec,
      const unsigned int         *addToVecStartingContiguousBlockIds)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;

          auto atomic_add_real =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              reinterpret_cast<double *>(
                &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex])[0]);
          atomic_add_real += addFromVec[index].real();

          auto atomic_add_imag =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              reinterpret_cast<double *>(
                &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex])[1]);
          atomic_add_imag += addFromVec[index].imag();
        }
    }


    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>    ind,
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const double       *addFromVec,
      double             *addToVecReal,
      double             *addToVecImag,
      const unsigned int *addToVecStartingContiguousBlockIds)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;

          auto atomic_add_real =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              addToVecReal[addToVecStartingContiguousBlockIds[blockIndex] +
                           intraBlockIndex]);
          atomic_add_real += addFromVec[index];

          auto atomic_add_imag =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              addToVecImag[addToVecStartingContiguousBlockIds[blockIndex] +
                           intraBlockIndex]);
          atomic_add_imag += addFromVec[index];
        }
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>            ind,
      const unsigned int          contiguousBlockSize,
      const unsigned int          numContiguousBlocks,
      const std::complex<double> *addFromVec,
      double                     *addToVecReal,
      double                     *addToVecImag,
      const unsigned int         *addToVecStartingContiguousBlockIds)
    {
      unsigned int global_id     = ind.get_global_id(0);
      unsigned int numberEntries = numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups  = ind.get_group_range(0);
      unsigned int n_workitems   = ind.get_local_range(0);
      for (unsigned int index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;

          auto atomic_add_real =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              addToVecReal[addToVecStartingContiguousBlockIds[blockIndex] +
                           intraBlockIndex]);
          atomic_add_real += addFromVec[index].real();

          auto atomic_add_imag =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              addToVecImag[addToVecStartingContiguousBlockIds[blockIndex] +
                           intraBlockIndex]);
          atomic_add_imag += addFromVec[index].imag();
        }
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>    ind,
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const double        a,
      const double       *addFromVec,
      double             *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds)
    {
      unsigned int globalThreadId = ind.get_global_id(0);
      unsigned int numberEntries  = numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups   = ind.get_group_range(0);
      unsigned int n_workitems    = ind.get_local_range(0);
      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = a;

          auto atomic_add =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                       intraBlockIndex]);
          atomic_add += dftfe::utils::mult(addFromVec[index], coeff);
        }
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>                         ind,
      const unsigned int                       contiguousBlockSize,
      const unsigned int                       numContiguousBlocks,
      const double                             a,
      const dftfe::utils::deviceDoubleComplex *addFromVec,
      dftfe::utils::deviceDoubleComplex       *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups = ind.get_group_range(0);
      unsigned int n_workitems  = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = a;

          auto atomic_add_real =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              reinterpret_cast<double *>(
                &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex])[0]);
          atomic_add_real +=
            dftfe::utils::mult(addFromVec[index].real(), coeff);

          auto atomic_add_imag =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              reinterpret_cast<double *>(
                &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex])[1]);
          atomic_add_imag +=
            dftfe::utils::mult(addFromVec[index].imag(), coeff);
        }
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>    ind,
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const double        a,
      const float        *addFromVec,
      double             *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups = ind.get_group_range(0);
      unsigned int n_workitems  = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = a;

          auto atomic_add =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                       intraBlockIndex]);
          atomic_add += dftfe::utils::mult(addFromVec[index], coeff);
        }
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>                        ind,
      const unsigned int                      contiguousBlockSize,
      const unsigned int                      numContiguousBlocks,
      const double                            a,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceDoubleComplex      *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups = ind.get_group_range(0);
      unsigned int n_workitems  = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = a;

          auto atomic_add_real =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              reinterpret_cast<double *>(
                &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex])[0]);
          atomic_add_real +=
            dftfe::utils::mult(addFromVec[index].real(), coeff);

          auto atomic_add_imag =
            sycl::atomic_ref<double,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              reinterpret_cast<double *>(
                &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex])[1]);
          atomic_add_imag +=
            dftfe::utils::mult(addFromVec[index].imag(), coeff);
        }
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>    ind,
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const double        a,
      const float        *addFromVec,
      float              *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups = ind.get_group_range(0);
      unsigned int n_workitems  = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = a;

          auto atomic_add =
            sycl::atomic_ref<float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                       intraBlockIndex]);
          atomic_add += dftfe::utils::mult(addFromVec[index], coeff);
        }
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>                        ind,
      const unsigned int                      contiguousBlockSize,
      const unsigned int                      numContiguousBlocks,
      const double                            a,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups = ind.get_group_range(0);
      unsigned int n_workitems  = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;
          const double coeff           = a;

          auto atomic_add_real =
            sycl::atomic_ref<float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              reinterpret_cast<float *>(
                &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex])[0]);
          atomic_add_real +=
            dftfe::utils::mult(addFromVec[index].real(), coeff);

          auto atomic_add_imag =
            sycl::atomic_ref<float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              reinterpret_cast<float *>(
                &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex])[1]);
          atomic_add_imag +=
            dftfe::utils::mult(addFromVec[index].imag(), coeff);
        }
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>    ind,
      const unsigned int  contiguousBlockSize,
      const unsigned int  numContiguousBlocks,
      const float         a,
      const float        *addFromVec,
      float              *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups = ind.get_group_range(0);
      unsigned int n_workitems  = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;
          const float  coeff           = a;

          auto atomic_add =
            sycl::atomic_ref<float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                       intraBlockIndex]);
          atomic_add += dftfe::utils::mult(addFromVec[index], coeff);
        }
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>                        ind,
      const unsigned int                      contiguousBlockSize,
      const unsigned int                      numContiguousBlocks,
      const float                             a,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec,
      const unsigned int *addToVecStartingContiguousBlockIds)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups = ind.get_group_range(0);
      unsigned int n_workitems  = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex      = index / contiguousBlockSize;
          unsigned int intraBlockIndex = index % contiguousBlockSize;
          const float  coeff           = a;

          auto atomic_add_real =
            sycl::atomic_ref<float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              reinterpret_cast<float *>(
                &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex])[0]);
          atomic_add_real +=
            dftfe::utils::mult(addFromVec[index].real(), coeff);

          auto atomic_add_imag =
            sycl::atomic_ref<float,
                             sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>(
              reinterpret_cast<float *>(
                &addToVec[addToVecStartingContiguousBlockIds[blockIndex] +
                          intraBlockIndex])[1]);
          atomic_add_imag +=
            dftfe::utils::mult(addFromVec[index].imag(), coeff);
        }
    }

    template <typename ValueType>
    void
    addVecOverContinuousIndexKernel(sycl::nd_item<1>   ind,
                                    const unsigned int numContiguousBlocks,
                                    const unsigned int contiguousBlockSize,
                                    const ValueType   *input1,
                                    const ValueType   *input2,
                                    ValueType         *output)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numberEntries  = numContiguousBlocks;
      unsigned int       n_workgroups   = ind.get_group_range(0);
      unsigned int       n_workitems    = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          for (unsigned int iBlock = 0; iBlock < contiguousBlockSize; iBlock++)
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

    template <typename ValueType>
    void
    stridedBlockScaleColumnWiseKernel(sycl::nd_item<1>   ind,
                                      const unsigned int contiguousBlockSize,
                                      const unsigned int numContiguousBlocks,
                                      const ValueType   *beta,
                                      ValueType         *x)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups = ind.get_group_range(0);
      unsigned int n_workitems  = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          unsigned int intrablockindex =
            index - blockIndex * contiguousBlockSize;
          dftfe::utils::copyValue(x + index,
                                  dftfe::utils::mult(beta[intrablockindex],
                                                     x[index]));
        }
    }

    template <typename ValueType>
    void
    stridedBlockScaleAndAddColumnWiseKernel(
      sycl::nd_item<1>   ind,
      const unsigned int contiguousBlockSize,
      const unsigned int numContiguousBlocks,
      const ValueType   *x,
      const ValueType   *beta,
      ValueType         *y)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups = ind.get_group_range(0);
      unsigned int n_workitems  = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          unsigned int intrablockindex =
            index - blockIndex * contiguousBlockSize;
          dftfe::utils::copyValue(
            y + index,
            dftfe::utils::add(
              y[index], dftfe::utils::mult(beta[intrablockindex], x[index])));
        }
    }

    template <typename ValueType>
    void
    stridedBlockScaleAndAddTwoVecColumnWiseKernel(
      sycl::nd_item<1>   ind,
      const unsigned int contiguousBlockSize,
      const unsigned int numContiguousBlocks,
      const ValueType   *x,
      const ValueType   *alpha,
      const ValueType   *y,
      const ValueType   *beta,
      ValueType         *z)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups = ind.get_group_range(0);
      unsigned int n_workitems  = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          unsigned int intrablockindex =
            index - blockIndex * contiguousBlockSize;
          dftfe::utils::copyValue(
            z + index,
            dftfe::utils::add(
              dftfe::utils::mult(alpha[intrablockindex], x[index]),
              dftfe::utils::mult(beta[intrablockindex], y[index])));
        }
    }

    void
    hadamardProductKernel(sycl::nd_item<1>   ind,
                          const unsigned int vecSize,
                          const float       *xVec,
                          const float       *yVec,
                          float             *outputVec)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      unsigned int       n_workgroups   = ind.get_group_range(0);
      unsigned int       n_workitems    = ind.get_local_range(0);
      for (int i = globalThreadId; i < vecSize; i += n_workgroups * n_workitems)
        {
          outputVec[i] = yVec[i] * xVec[i];
        }
    }

    void
    hadamardProductKernel(sycl::nd_item<1>   ind,
                          const unsigned int vecSize,
                          const double      *xVec,
                          const double      *yVec,
                          double            *outputVec)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      unsigned int       n_workgroups   = ind.get_group_range(0);
      unsigned int       n_workitems    = ind.get_local_range(0);
      for (int i = globalThreadId; i < vecSize; i += n_workgroups * n_workitems)
        {
          outputVec[i] = yVec[i] * xVec[i];
        }
    }

    void
    hadamardProductKernel(sycl::nd_item<1>                         ind,
                          const unsigned int                       vecSize,
                          const dftfe::utils::deviceDoubleComplex *xVec,
                          const dftfe::utils::deviceDoubleComplex *yVec,
                          dftfe::utils::deviceDoubleComplex       *outputVec)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      unsigned int       n_workgroups   = ind.get_group_range(0);
      unsigned int       n_workitems    = ind.get_local_range(0);
      for (int i = globalThreadId; i < vecSize; i += n_workgroups * n_workitems)
        {
          outputVec[i].real(yVec[i].real() * xVec[i].real() -
                            yVec[i].imag() * xVec[i].imag());
          outputVec[i].imag(yVec[i].real() * xVec[i].imag() +
                            yVec[i].imag() * xVec[i].real());
        }
    }

    void
    hadamardProductWithConjKernel(sycl::nd_item<1>   ind,
                                  const unsigned int vecSize,
                                  const float       *xVec,
                                  const float       *yVec,
                                  float             *outputVec)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      unsigned int       n_workgroups   = ind.get_group_range(0);
      unsigned int       n_workitems    = ind.get_local_range(0);
      for (int i = globalThreadId; i < vecSize; i += n_workgroups * n_workitems)
        {
          outputVec[i] = yVec[i] * xVec[i];
        }
    }

    void
    hadamardProductWithConjKernel(sycl::nd_item<1>   ind,
                                  const unsigned int vecSize,
                                  const double      *xVec,
                                  const double      *yVec,
                                  double            *outputVec)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      unsigned int       n_workgroups   = ind.get_group_range(0);
      unsigned int       n_workitems    = ind.get_local_range(0);
      for (int i = globalThreadId; i < vecSize; i += n_workgroups * n_workitems)
        {
          outputVec[i] = yVec[i] * xVec[i];
        }
    }

    void
    hadamardProductWithConjKernel(sycl::nd_item<1>   ind,
                                  const unsigned int vecSize,
                                  const dftfe::utils::deviceDoubleComplex *xVec,
                                  const dftfe::utils::deviceDoubleComplex *yVec,
                                  dftfe::utils::deviceDoubleComplex *outputVec)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      unsigned int       n_workgroups   = ind.get_group_range(0);
      unsigned int       n_workitems    = ind.get_local_range(0);
      for (int i = globalThreadId; i < vecSize; i += n_workgroups * n_workitems)
        {
          outputVec[i].real(yVec[i].real() * xVec[i].real() +
                            yVec[i].imag() * xVec[i].imag());
          outputVec[i].imag(yVec[i].imag() * xVec[i].real() -
                            yVec[i].real() * xVec[i].imag());
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    stridedBlockAxpyDeviceKernel(sycl::nd_item<1>   ind,
                                 const unsigned int contiguousBlockSize,
                                 const unsigned int numContiguousBlocks,
                                 const ValueType2   a,
                                 const ValueType2  *s,
                                 const ValueType1  *addFromVec,
                                 ValueType1        *addToVec)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups = ind.get_group_range(0);
      unsigned int n_workitems  = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int     blockIndex = index / contiguousBlockSize;
          const ValueType2 coeff      = dftfe::utils::mult(a, s[blockIndex]);
          addToVec[index] =
            dftfe::utils::add(addToVec[index],
                              dftfe::utils::mult(addFromVec[index], coeff));
        }
    }

    template <>
    void
    stridedBlockAxpyDeviceKernel(
      sycl::nd_item<1>                        ind,
      const unsigned int                      contiguousBlockSize,
      const unsigned int                      numContiguousBlocks,
      const double                            a,
      const double                           *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups = ind.get_group_range(0);
      unsigned int n_workitems  = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          const double coeff      = dftfe::utils::mult(a, s[blockIndex]);
          addToVec[index].real(dftfe::utils::add(
            addToVec[index].real(),
            dftfe::utils::mult(addFromVec[index].real(), coeff)));
          addToVec[index].imag(dftfe::utils::add(
            addToVec[index].imag(),
            dftfe::utils::mult(addFromVec[index].imag(), coeff)));
        }
    }

    void
    computeRightDiagonalScaleKernel(sycl::nd_item<1>   ind,
                                    const double      *diagValues,
                                    double            *X,
                                    const unsigned int N,
                                    const unsigned int M)
    {
      const unsigned int numEntries   = N * M;
      unsigned int       n_workgroups = ind.get_group_range(0);
      unsigned int       n_workitems  = ind.get_local_range(0);
      for (int i = ind.get_global_id(0); i < numEntries;
           i += n_workgroups * n_workitems)
        {
          const unsigned int idof = i / N;
          const unsigned int ivec = i % N;

          *(X + N * idof + ivec) = *(X + N * idof + ivec) * diagValues[ivec];
        }
    }

    void
    computeRightDiagonalScaleKernel(sycl::nd_item<1> ind,
                                    const double    *diagValues,
                                    dftfe::utils::deviceDoubleComplex *X,
                                    const unsigned int                 N,
                                    const unsigned int                 M)
    {
      const unsigned int numEntries   = N * M;
      unsigned int       n_workgroups = ind.get_group_range(0);
      unsigned int       n_workitems  = ind.get_local_range(0);
      for (int i = ind.get_global_id(0); i < numEntries;
           i += n_workgroups * n_workitems)
        {
          const unsigned int idof = i / N;
          const unsigned int ivec = i % N;

          *(X + N * idof + ivec) =
            dftfe::utils::mult(*(X + N * idof + ivec), diagValues[ivec]);
        }
    }

    void
    computeRightDiagonalScaleKernel(
      sycl::nd_item<1>                         ind,
      const dftfe::utils::deviceDoubleComplex *diagValues,
      dftfe::utils::deviceDoubleComplex       *X,
      const unsigned int                       N,
      const unsigned int                       M)
    {
      const unsigned int numEntries   = N * M;
      unsigned int       n_workgroups = ind.get_group_range(0);
      unsigned int       n_workitems  = ind.get_local_range(0);
      for (int i = ind.get_global_id(0); i < numEntries;
           i += n_workgroups * n_workitems)
        {
          const unsigned int idof = i / N;
          const unsigned int ivec = i % N;

          *(X + N * idof + ivec) =
            dftfe::utils::mult(*(X + N * idof + ivec), diagValues[ivec]);
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    stridedBlockAxpByDeviceKernel(sycl::nd_item<1>   ind,
                                  const unsigned int contiguousBlockSize,
                                  const unsigned int numContiguousBlocks,
                                  const ValueType2   a,
                                  const ValueType2   b,
                                  const ValueType2  *s,
                                  const ValueType1  *addFromVec,
                                  ValueType1        *addToVec)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups = ind.get_group_range(0);
      unsigned int n_workitems  = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int     blockIndex = index / contiguousBlockSize;
          const ValueType2 coeff      = dftfe::utils::mult(a, s[blockIndex]);
          addToVec[index] =
            dftfe::utils::add(dftfe::utils::mult(addToVec[index], b),
                              dftfe::utils::mult(addFromVec[index], coeff));
        }
    }

    template <>
    void
    stridedBlockAxpByDeviceKernel(
      sycl::nd_item<1>                        ind,
      const unsigned int                      contiguousBlockSize,
      const unsigned int                      numContiguousBlocks,
      const double                            a,
      const double                            b,
      const double                           *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      unsigned int n_workgroups = ind.get_group_range(0);
      unsigned int n_workitems  = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          unsigned int blockIndex = index / contiguousBlockSize;
          const double coeff      = dftfe::utils::mult(a, s[blockIndex]);
          addToVec[index].real(dftfe::utils::add(
            dftfe::utils::mult(addToVec[index].real(), b),
            dftfe::utils::mult(addFromVec[index].real(), coeff)));
          addToVec[index].imag(dftfe::utils::add(
            dftfe::utils::mult(addToVec[index].imag(), b),
            dftfe::utils::mult(addFromVec[index].imag(), coeff)));
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    copyBlockDiagonalValueType1OffDiagonalValueType2FromValueType1ArrDeviceKernel(
      sycl::nd_item<1>   ind,
      const unsigned int B,
      const unsigned int DRem,
      const unsigned int D,
      const ValueType1  *valueType1SrcArray,
      ValueType1        *valueType1DstArray,
      ValueType2        *valueType2DstArray)
    {
      const unsigned int globalThreadId = ind.get_global_id(0);
      const unsigned int size           = B * D;
      unsigned int       n_workgroups   = ind.get_group_range(0);
      unsigned int       n_workitems    = ind.get_local_range(0);

      for (unsigned int index = globalThreadId; index < size;
           index += n_workgroups * n_workitems)
        {
          const unsigned int ibdof = index / D;
          const unsigned int ivec  = index % D;
          if (ivec < B)
            dftfe::utils::copyValue(valueType1DstArray + ibdof * B + ivec,
                                    valueType1SrcArray[index]);
          else
            dftfe::utils::copyValue(valueType2DstArray + (ibdof - B) +
                                      (ivec - B) * B,
                                    valueType1SrcArray[index]);
        }
    }

  } // namespace
} // namespace dftfe
