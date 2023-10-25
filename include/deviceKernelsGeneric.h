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

#if defined(DFTFE_WITH_DEVICE)
#  ifndef deviceKernelsGeneric_H_
#    define deviceKernelsGeneric_H_

#    include <dftfeDataTypes.h>
#    include <MemorySpaceType.h>
#    include <headers.h>
#    include <TypeConfig.h>
#    include <DeviceTypeConfig.h>

namespace dftfe
{
  namespace utils
  {
    namespace deviceKernelsGeneric
    {
      void
      setupDevice();

      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyComplexArrToRealArrsDevice(const dftfe::size_type  size,
                                     const ValueTypeComplex *complexArr,
                                     ValueTypeReal *         realArr,
                                     ValueTypeReal *         imagArr);


      template <typename ValueTypeComplex, typename ValueTypeReal>
      void
      copyRealArrsToComplexArrDevice(const dftfe::size_type size,
                                     const ValueTypeReal *  realArr,
                                     const ValueTypeReal *  imagArr,
                                     ValueTypeComplex *     complexArr);

      template <typename ValueType1, typename ValueType2>
      void
      copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                       const ValueType1 *     valueType1Arr,
                                       ValueType2 *           valueType2Arr,
                                       const deviceStream_t   streamId = 0);


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyToBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const ValueType1 *             copyFromVec,
        ValueType2 *                   copyToVecBlock,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);



      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyFromBlock(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const ValueType1 *             copyFromVecBlock,
        ValueType2 *                   copyToVec,
        const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                       const dftfe::size_type blockSizeFrom,
                                       const dftfe::size_type numBlocks,
                                       const dftfe::size_type startingId,
                                       const ValueType1 *     copyFromVec,
                                       ValueType2 *           copyToVec);


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyConstantStride(const dftfe::size_type blockSize,
                                const dftfe::size_type strideTo,
                                const dftfe::size_type strideFrom,
                                const dftfe::size_type numBlocks,
                                const dftfe::size_type startingToId,
                                const dftfe::size_type startingFromId,
                                const ValueType1 *     copyFromVec,
                                ValueType2 *           copyToVec);


      template <typename ValueType1, typename ValueType2>
      void
      stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                         const dftfe::size_type blockSizeFrom,
                                         const dftfe::size_type numBlocks,
                                         const dftfe::size_type startingId,
                                         const ValueType1 *     copyFromVec,
                                         ValueType2 *           copyToVec);


      template <typename ValueType1, typename ValueType2>
      void
      axpby(const dftfe::size_type n,
            const ValueType1 *     x,
            ValueType1 *           y,
            const ValueType2       a,
            const ValueType2       b);

      template <typename ValueType>
      void
      axpyStridedBlockAtomicAdd(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const ValueType *              addFromVec,
        ValueType *                    addToVec,
        const dftfe::global_size_type *addToVecStartingContiguousBlockIds);

      template <typename ValueType>
      void
      axpyStridedBlockAtomicAdd(
        const dftfe::size_type         contiguousBlockSize,
        const dftfe::size_type         numContiguousBlocks,
        const ValueType *              addFromVec,
        double *                       addToVecReal,
        double *                       addToVecImag,
        const dftfe::global_size_type *addToVecStartingContiguousBlockIds);


      template <typename ValueType1, typename ValueType2>
      void
      ascal(const dftfe::size_type n, ValueType1 *x, const ValueType2 a);

      template <typename ValueType1, typename ValueType2>
      void
      stridedBlockScale(const dftfe::size_type contiguousBlockSize,
                        const dftfe::size_type numContiguousBlocks,
                        const ValueType1       a,
                        const ValueType1 *     s,
                        ValueType2 *           x);

      void
      add(double *                          y,
          const double *                    x,
          const double                      alpha,
          const dftfe::size_type            size,
          dftfe::utils::deviceBlasHandle_t &deviceBlasHandle);

      double
      l2_norm(const double *                    x,
              const dftfe::size_type            size,
              const MPI_Comm &                  mpi_communicator,
              dftfe::utils::deviceBlasHandle_t &deviceBlasHandle);

      double
      dot(const double *                    x,
          const double *                    y,
          const dftfe::size_type            size,
          const MPI_Comm &                  mpi_communicator,
          dftfe::utils::deviceBlasHandle_t &deviceBlasHandle);

      template <typename ValueType>
      void
      sadd(ValueType *            y,
           ValueType *            x,
           const ValueType        beta,
           const dftfe::size_type size);

    } // namespace deviceKernelsGeneric
  }   // namespace utils
} // namespace dftfe

#  endif
#endif
