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
    saddKernel(sycl::nd_item<1>       ind,
               ValueType             *y,
               ValueType             *x,
               const ValueType        beta,
               const dftfe::size_type size)
    {
      const dftfe::size_type globalId     = ind.get_global_id(0);
      dftfe::size_type       n_workgroups = ind.get_group_range(0);
      dftfe::size_type       n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type idx = globalId; idx < size;
           idx += n_workgroups * n_workitems)
        {
          y[idx] = beta * y[idx] - x[idx];
          x[idx] = 0;
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    ascalDeviceKernel(sycl::nd_item<1>       ind,
                      const dftfe::size_type n,
                      ValueType1            *x,
                      const ValueType2       a)
    {
      for (dftfe::size_type i = ind.get_global_id(0); i < n;
           i += ind.get_group_range(0) * ind.get_local_range(0))
        dftfe::utils::copyValue(x + i, dftfe::utils::mult(a, x[i]));
    }


    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    copyComplexArrToRealArrsDeviceKernel(sycl::nd_item<1>        ind,
                                         const dftfe::size_type  size,
                                         const ValueTypeComplex *complexArr,
                                         ValueTypeReal          *realArr,
                                         ValueTypeReal          *imagArr)
    {
      const dftfe::size_type globalId     = ind.get_global_id(0);
      dftfe::size_type       n_workgroups = ind.get_group_range(0);
      dftfe::size_type       n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type idx = globalId; idx < size;
           idx += n_workgroups * n_workitems)
        {
          realArr[idx] = complexArr[idx].real();
          imagArr[idx] = complexArr[idx].imag();
        }
    }

    template <typename ValueTypeComplex, typename ValueTypeReal>
    void
    copyRealArrsToComplexArrDeviceKernel(sycl::nd_item<1>       ind,
                                         const dftfe::size_type size,
                                         const ValueTypeReal   *realArr,
                                         const ValueTypeReal   *imagArr,
                                         ValueTypeComplex      *complexArr)
    {
      const dftfe::size_type globalId     = ind.get_global_id(0);
      dftfe::size_type       n_workgroups = ind.get_group_range(0);
      dftfe::size_type       n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type idx = globalId; idx < size;
           idx += n_workgroups * n_workitems)
        {
          complexArr[idx].real(realArr[idx]);
          complexArr[idx].imag(imagArr[idx]);
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    copyValueType1ArrToValueType2ArrDeviceKernel(
      sycl::nd_item<1>       ind,
      const dftfe::size_type size,
      const ValueType1      *valueType1Arr,
      ValueType2            *valueType2Arr)
    {
      const dftfe::size_type globalId     = ind.get_global_id(0);
      dftfe::size_type       n_workgroups = ind.get_group_range(0);
      dftfe::size_type       n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type index = globalId; index < size;
           index += n_workgroups * n_workitems)
        dftfe::utils::copyValue(valueType2Arr + index, valueType1Arr[index]);
    }

    template <typename ValueType0,
              typename ValueType1,
              typename ValueType2,
              typename ValueType3,
              typename ValueType4>
    void
    ApaBDDeviceKernel(sycl::nd_item<1>       ind,
                      const dftfe::size_type nRows,
                      const dftfe::size_type nCols,
                      const ValueType0       alpha,
                      const ValueType1      *A,
                      const ValueType2      *B,
                      const ValueType3      *D,
                      ValueType4            *C)
    {
      dftfe::size_type globalThreadId = ind.get_global_id(0);
      dftfe::size_type numberEntries  = nCols * nRows;
      dftfe::size_type n_workgroups   = ind.get_group_range(0);
      dftfe::size_type n_workitems    = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type iRow   = index % nCols;
          const ValueType0 alphaD = alpha * D[iRow];
          dftfe::utils::copyValue(
            C + index,
            dftfe::utils::add(A[index], dftfe::utils::mult(B[index], alphaD)));
        }
    }


    template <typename ValueType1, typename ValueType2>
    void
    stridedCopyToBlockDeviceKernel(
      sycl::nd_item<1>               ind,
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1              *copyFromVec,
      ValueType2                    *copyToVec,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {
      dftfe::size_type global_id = ind.get_global_id(0);
      dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
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
    void
    stridedCopyToBlockDeviceKernel(
      sycl::nd_item<1>               ind,
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const dftfe::size_type         stratingVecId,
      const ValueType1              *copyFromVec,
      ValueType2                    *copyToVec,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex =
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
      sycl::nd_item<1>               ind,
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1               a,
      const ValueType1              *s,
      const ValueType2              *copyFromVec,
      ValueType2                    *copyToVec,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {
      dftfe::size_type global_id = ind.get_global_id(0);
      dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
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
    void
    stridedCopyFromBlockDeviceKernel(
      sycl::nd_item<1>               ind,
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const ValueType1              *copyFromVec,
      ValueType2                    *copyToVec,
      const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds)
    {
      dftfe::size_type global_id = ind.get_global_id(0);
      dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
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
    void
    stridedCopyToBlockConstantStrideDeviceKernel(
      sycl::nd_item<1>       ind,
      const dftfe::size_type blockSizeTo,
      const dftfe::size_type blockSizeFrom,
      const dftfe::size_type numBlocks,
      const dftfe::size_type startingId,
      const ValueType1      *copyFromVec,
      ValueType2            *copyToVec)
    {
      dftfe::size_type global_id     = ind.get_global_id(0);
      dftfe::size_type numberEntries = numBlocks * blockSizeTo;
      dftfe::size_type n_workgroups  = ind.get_group_range(0);
      dftfe::size_type n_workitems   = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / blockSizeTo;
          dftfe::size_type intraBlockIndex = index - blockIndex * blockSizeTo;
          dftfe::utils::copyValue(copyToVec + index,
                                  copyFromVec[blockIndex * blockSizeFrom +
                                              startingId + intraBlockIndex]);
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    stridedCopyFromBlockConstantStrideDeviceKernel(
      sycl::nd_item<1>       ind,
      const dftfe::size_type blockSizeTo,
      const dftfe::size_type blockSizeFrom,
      const dftfe::size_type numBlocks,
      const dftfe::size_type startingId,
      const ValueType1      *copyFromVec,
      ValueType2            *copyToVec)
    {
      dftfe::size_type global_id     = ind.get_global_id(0);
      dftfe::size_type numberEntries = numBlocks * blockSizeFrom;
      dftfe::size_type n_workgroups  = ind.get_group_range(0);
      dftfe::size_type n_workitems   = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / blockSizeFrom;
          dftfe::size_type intraBlockIndex = index - blockIndex * blockSizeFrom;
          dftfe::utils::copyValue(copyToVec + blockIndex * blockSizeTo +
                                    startingId + intraBlockIndex,
                                  copyFromVec[index]);
        }
    }

    template <typename ValueType1, typename ValueType2>
    void
    stridedCopyConstantStrideDeviceKernel(sycl::nd_item<1>       ind,
                                          const dftfe::size_type blockSize,
                                          const dftfe::size_type strideTo,
                                          const dftfe::size_type strideFrom,
                                          const dftfe::size_type numBlocks,
                                          const dftfe::size_type startingToId,
                                          const dftfe::size_type startingFromId,
                                          const ValueType1      *copyFromVec,
                                          ValueType2            *copyToVec)
    {
      dftfe::size_type global_id     = ind.get_global_id(0);
      dftfe::size_type numberEntries = numBlocks * blockSize;
      dftfe::size_type n_workgroups  = ind.get_group_range(0);
      dftfe::size_type n_workitems   = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / blockSize;
          dftfe::size_type intraBlockIndex = index - blockIndex * blockSize;
          dftfe::utils::copyValue(
            copyToVec + blockIndex * strideTo + startingToId + intraBlockIndex,
            copyFromVec[blockIndex * strideFrom + startingFromId +
                        intraBlockIndex]);
        }
    }


    // x=a*x, with inc=1
    template <typename ValueType1, typename ValueType2>
    void
    xscalDeviceKernel(sycl::nd_item<1>       ind,
                      const dftfe::size_type n,
                      ValueType1            *x,
                      const ValueType2       a)
    {
      dftfe::size_type global_id    = ind.get_global_id(0);
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type i = global_id; i < n;
           i += n_workgroups * n_workitems)
        dftfe::utils::copyValue(x + i, dftfe::utils::mult(a, x[i]));
    }


    // x[iblock*blocksize+intrablockindex]=a*s[iblock]*x[iblock*blocksize+intrablockindex]
    // strided block wise
    template <typename ValueType1, typename ValueType2>
    void
    stridedBlockScaleDeviceKernel(sycl::nd_item<1>       ind,
                                  const dftfe::size_type contiguousBlockSize,
                                  const dftfe::size_type numContiguousBlocks,
                                  const ValueType1       a,
                                  const ValueType1      *s,
                                  ValueType2            *x)
    {
      dftfe::size_type global_id = ind.get_global_id(0);
      dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex = index / contiguousBlockSize;
          dftfe::utils::copyValue(
            x + index,
            dftfe::utils::mult(dftfe::utils::mult(a, s[blockIndex]), x[index]));
        }
    }

    // y=a*x+b*y, with inc=1
    template <typename ValueType1, typename ValueType2>
    void
    axpbyDeviceKernel(sycl::nd_item<1>       ind,
                      const dftfe::size_type n,
                      const ValueType1      *x,
                      ValueType1            *y,
                      const ValueType2       a,
                      const ValueType2       b)
    {
      dftfe::size_type global_id    = ind.get_global_id(0);
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type i = global_id; i < n;
           i += n_workgroups * n_workitems)
        dftfe::utils::copyValue(y + i,
                                dftfe::utils::add(dftfe::utils::mult(a, x[i]),
                                                  dftfe::utils::mult(b, y[i])));
    }

    void
    axpyStridedBlockAtomicAddDeviceKernel(
      sycl::nd_item<1>               ind,
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                  *addFromVec,
      double                        *addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      dftfe::size_type global_id = ind.get_global_id(0);
      dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex =
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
      sycl::nd_item<1>               ind,
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const double                  *s,
      const double                  *addFromVec,
      double                        *addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      dftfe::size_type global_id = ind.get_global_id(0);
      dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          const double     coeff = dftfe::utils::mult(a, s[blockIndex]);

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
      sycl::nd_item<1>               ind,
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const double                  *s,
      const std::complex<double>    *addFromVec,
      std::complex<double>          *addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      dftfe::size_type global_id = ind.get_global_id(0);
      dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          const double     coeff = dftfe::utils::mult(a, s[blockIndex]);

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
      sycl::nd_item<1>               ind,
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const double                  *s,
      const float                   *addFromVec,
      double                        *addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      dftfe::size_type global_id = ind.get_global_id(0);
      dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          const double     coeff = dftfe::utils::mult(a, s[blockIndex]);

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
      const dftfe::size_type                  contiguousBlockSize,
      const dftfe::size_type                  numContiguousBlocks,
      const double                            a,
      const double                           *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceDoubleComplex      *addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      dftfe::size_type global_id = ind.get_global_id(0);
      dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          const double     coeff = dftfe::utils::mult(a, s[blockIndex]);

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
      sycl::nd_item<1>               ind,
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const double                  *s,
      const float                   *addFromVec,
      float                         *addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      dftfe::size_type global_id = ind.get_global_id(0);
      dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          const double     coeff = dftfe::utils::mult(a, s[blockIndex]);

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
      const dftfe::size_type                  contiguousBlockSize,
      const dftfe::size_type                  numContiguousBlocks,
      const double                            a,
      const double                           *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      dftfe::size_type global_id = ind.get_global_id(0);
      dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          const double     coeff = dftfe::utils::mult(a, s[blockIndex]);

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
      sycl::nd_item<1>               ind,
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const float                    a,
      const float                   *s,
      const float                   *addFromVec,
      float                         *addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      dftfe::size_type global_id = ind.get_global_id(0);
      dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          const double     coeff = dftfe::utils::mult(a, s[blockIndex]);

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
      const dftfe::size_type                  contiguousBlockSize,
      const dftfe::size_type                  numContiguousBlocks,
      const float                             a,
      const float                            *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      dftfe::size_type global_id = ind.get_global_id(0);
      dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          const double     coeff = dftfe::utils::mult(a, s[blockIndex]);

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
      sycl::nd_item<1>               ind,
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const std::complex<double>    *addFromVec,
      std::complex<double>          *addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      dftfe::size_type global_id = ind.get_global_id(0);
      dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;

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
      sycl::nd_item<1>               ind,
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                  *addFromVec,
      double                        *addToVecReal,
      double                        *addToVecImag,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      dftfe::size_type global_id = ind.get_global_id(0);
      dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;

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
      sycl::nd_item<1>               ind,
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const std::complex<double>    *addFromVec,
      double                        *addToVecReal,
      double                        *addToVecImag,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      dftfe::size_type global_id = ind.get_global_id(0);
      dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type index = global_id; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;

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
      sycl::nd_item<1>               ind,
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const double                  *addFromVec,
      double                        *addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      dftfe::size_type globalThreadId = ind.get_global_id(0);
      dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);
      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          const double     coeff           = a;

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
      const dftfe::size_type                   contiguousBlockSize,
      const dftfe::size_type                   numContiguousBlocks,
      const double                             a,
      const dftfe::utils::deviceDoubleComplex *addFromVec,
      dftfe::utils::deviceDoubleComplex       *addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          const double     coeff           = a;

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
      sycl::nd_item<1>               ind,
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const float                   *addFromVec,
      double                        *addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          const double     coeff           = a;

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
      const dftfe::size_type                  contiguousBlockSize,
      const dftfe::size_type                  numContiguousBlocks,
      const double                            a,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceDoubleComplex      *addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          const double     coeff           = a;

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
      sycl::nd_item<1>               ind,
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const double                   a,
      const float                   *addFromVec,
      float                         *addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          const double     coeff           = a;

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
      const dftfe::size_type                  contiguousBlockSize,
      const dftfe::size_type                  numContiguousBlocks,
      const double                            a,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          const double     coeff           = a;

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
      sycl::nd_item<1>               ind,
      const dftfe::size_type         contiguousBlockSize,
      const dftfe::size_type         numContiguousBlocks,
      const float                    a,
      const float                   *addFromVec,
      float                         *addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          const float      coeff           = a;

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
      const dftfe::size_type                  contiguousBlockSize,
      const dftfe::size_type                  numContiguousBlocks,
      const float                             a,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec,
      const dftfe::global_size_type *addToVecStartingContiguousBlockIds)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex      = index / contiguousBlockSize;
          dftfe::size_type intraBlockIndex = index % contiguousBlockSize;
          const float      coeff           = a;

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
    addVecOverContinuousIndexKernel(sycl::nd_item<1>       ind,
                                    const dftfe::size_type numContiguousBlocks,
                                    const dftfe::size_type contiguousBlockSize,
                                    const ValueType       *input1,
                                    const ValueType       *input2,
                                    ValueType             *output)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      const dftfe::size_type numberEntries  = numContiguousBlocks;
      dftfe::size_type       n_workgroups   = ind.get_group_range(0);
      dftfe::size_type       n_workitems    = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          for (dftfe::size_type iBlock = 0; iBlock < contiguousBlockSize;
               iBlock++)
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
    stridedBlockScaleColumnWiseKernel(
      sycl::nd_item<1>       ind,
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const ValueType       *beta,
      ValueType             *x)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex = index / contiguousBlockSize;
          dftfe::size_type intrablockindex =
            index - blockIndex * contiguousBlockSize;
          dftfe::utils::copyValue(x + index,
                                  dftfe::utils::mult(beta[intrablockindex],
                                                     x[index]));
        }
    }

    template <typename ValueType>
    void
    stridedBlockScaleAndAddColumnWiseKernel(
      sycl::nd_item<1>       ind,
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const ValueType       *x,
      const ValueType       *beta,
      ValueType             *y)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex = index / contiguousBlockSize;
          dftfe::size_type intrablockindex =
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
      sycl::nd_item<1>       ind,
      const dftfe::size_type contiguousBlockSize,
      const dftfe::size_type numContiguousBlocks,
      const ValueType       *x,
      const ValueType       *alpha,
      const ValueType       *y,
      const ValueType       *beta,
      ValueType             *z)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex = index / contiguousBlockSize;
          dftfe::size_type intrablockindex =
            index - blockIndex * contiguousBlockSize;
          dftfe::utils::copyValue(
            z + index,
            dftfe::utils::add(
              dftfe::utils::mult(alpha[intrablockindex], x[index]),
              dftfe::utils::mult(beta[intrablockindex], y[index])));
        }
    }

    void
    hadamardProductKernel(sycl::nd_item<1>       ind,
                          const dftfe::size_type vecSize,
                          const float           *xVec,
                          const float           *yVec,
                          float                 *outputVec)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      dftfe::size_type       n_workgroups   = ind.get_group_range(0);
      dftfe::size_type       n_workitems    = ind.get_local_range(0);
      for (int i = globalThreadId; i < vecSize; i += n_workgroups * n_workitems)
        {
          outputVec[i] = yVec[i] * xVec[i];
        }
    }

    void
    hadamardProductKernel(sycl::nd_item<1>       ind,
                          const dftfe::size_type vecSize,
                          const double          *xVec,
                          const double          *yVec,
                          double                *outputVec)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      dftfe::size_type       n_workgroups   = ind.get_group_range(0);
      dftfe::size_type       n_workitems    = ind.get_local_range(0);
      for (int i = globalThreadId; i < vecSize; i += n_workgroups * n_workitems)
        {
          outputVec[i] = yVec[i] * xVec[i];
        }
    }

    void
    hadamardProductKernel(sycl::nd_item<1>                         ind,
                          const dftfe::size_type                   vecSize,
                          const dftfe::utils::deviceDoubleComplex *xVec,
                          const dftfe::utils::deviceDoubleComplex *yVec,
                          dftfe::utils::deviceDoubleComplex       *outputVec)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      dftfe::size_type       n_workgroups   = ind.get_group_range(0);
      dftfe::size_type       n_workitems    = ind.get_local_range(0);
      for (int i = globalThreadId; i < vecSize; i += n_workgroups * n_workitems)
        {
          outputVec[i].real(yVec[i].real() * xVec[i].real() -
                            yVec[i].imag() * xVec[i].imag());
          outputVec[i].imag(yVec[i].real() * xVec[i].imag() +
                            yVec[i].imag() * xVec[i].real());
        }
    }

    void
    hadamardProductWithConjKernel(sycl::nd_item<1>       ind,
                                  const dftfe::size_type vecSize,
                                  const float           *xVec,
                                  const float           *yVec,
                                  float                 *outputVec)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      dftfe::size_type       n_workgroups   = ind.get_group_range(0);
      dftfe::size_type       n_workitems    = ind.get_local_range(0);
      for (int i = globalThreadId; i < vecSize; i += n_workgroups * n_workitems)
        {
          outputVec[i] = yVec[i] * xVec[i];
        }
    }

    void
    hadamardProductWithConjKernel(sycl::nd_item<1>       ind,
                                  const dftfe::size_type vecSize,
                                  const double          *xVec,
                                  const double          *yVec,
                                  double                *outputVec)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      dftfe::size_type       n_workgroups   = ind.get_group_range(0);
      dftfe::size_type       n_workitems    = ind.get_local_range(0);
      for (int i = globalThreadId; i < vecSize; i += n_workgroups * n_workitems)
        {
          outputVec[i] = yVec[i] * xVec[i];
        }
    }

    void
    hadamardProductWithConjKernel(sycl::nd_item<1>       ind,
                                  const dftfe::size_type vecSize,
                                  const dftfe::utils::deviceDoubleComplex *xVec,
                                  const dftfe::utils::deviceDoubleComplex *yVec,
                                  dftfe::utils::deviceDoubleComplex *outputVec)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      dftfe::size_type       n_workgroups   = ind.get_group_range(0);
      dftfe::size_type       n_workitems    = ind.get_local_range(0);
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
    stridedBlockAxpyDeviceKernel(sycl::nd_item<1>       ind,
                                 const dftfe::size_type contiguousBlockSize,
                                 const dftfe::size_type numContiguousBlocks,
                                 const ValueType2       a,
                                 const ValueType2      *s,
                                 const ValueType1      *addFromVec,
                                 ValueType1            *addToVec)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex = index / contiguousBlockSize;
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
      const dftfe::size_type                  contiguousBlockSize,
      const dftfe::size_type                  numContiguousBlocks,
      const double                            a,
      const double                           *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex = index / contiguousBlockSize;
          const double     coeff      = dftfe::utils::mult(a, s[blockIndex]);
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
      dftfe::size_type   n_workgroups = ind.get_group_range(0);
      dftfe::size_type   n_workitems  = ind.get_local_range(0);
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
      dftfe::size_type   n_workgroups = ind.get_group_range(0);
      dftfe::size_type   n_workitems  = ind.get_local_range(0);
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
      dftfe::size_type   n_workgroups = ind.get_group_range(0);
      dftfe::size_type   n_workitems  = ind.get_local_range(0);
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
    stridedBlockAxpByDeviceKernel(sycl::nd_item<1>       ind,
                                  const dftfe::size_type contiguousBlockSize,
                                  const dftfe::size_type numContiguousBlocks,
                                  const ValueType2       a,
                                  const ValueType2       b,
                                  const ValueType2      *s,
                                  const ValueType1      *addFromVec,
                                  ValueType1            *addToVec)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex = index / contiguousBlockSize;
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
      const dftfe::size_type                  contiguousBlockSize,
      const dftfe::size_type                  numContiguousBlocks,
      const double                            a,
      const double                            b,
      const double                           *s,
      const dftfe::utils::deviceFloatComplex *addFromVec,
      dftfe::utils::deviceFloatComplex       *addToVec)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      const dftfe::size_type numberEntries =
        numContiguousBlocks * contiguousBlockSize;
      dftfe::size_type n_workgroups = ind.get_group_range(0);
      dftfe::size_type n_workitems  = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < numberEntries;
           index += n_workgroups * n_workitems)
        {
          dftfe::size_type blockIndex = index / contiguousBlockSize;
          const double     coeff      = dftfe::utils::mult(a, s[blockIndex]);
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
      sycl::nd_item<1>       ind,
      const dftfe::size_type B,
      const dftfe::size_type DRem,
      const dftfe::size_type D,
      const ValueType1      *valueType1SrcArray,
      ValueType1            *valueType1DstArray,
      ValueType2            *valueType2DstArray)
    {
      const dftfe::size_type globalThreadId = ind.get_global_id(0);
      const dftfe::size_type size           = B * D;
      dftfe::size_type       n_workgroups   = ind.get_group_range(0);
      dftfe::size_type       n_workitems    = ind.get_local_range(0);

      for (dftfe::size_type index = globalThreadId; index < size;
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
