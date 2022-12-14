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
#  ifndef deviceKernelsGeneric.h
#    define deviceKernelsGeneric.h

#    include <dftfeDataTypes.h>
#    include <MemorySpaceType.h>
#    include <headers.h>
#    include <TypeConfig.h>

namespace dftfe
{
  namespace deviceKernelsGeneric
  {
    void
    setupDevice();

    template <typename NumberTypeComplex, typename NumberTypeReal>
    void
    copyComplexArrToRealArrsDevice(const dftfe::size_type   size,
                                   const NumberTypeComplex *complexArr,
                                   NumberTypeReal *         realArr,
                                   NumberTypeReal *         imagArr);


    template <typename NumberTypeComplex, typename NumberTypeReal>
    void
    copyRealArrsToComplexArrDevice(const dftfe::size_type size,
                                   const NumberTypeReal * realArr,
                                   const NumberTypeReal * imagArr,
                                   NumberTypeComplex *    complexArr);

    template <typename ValueType1, typename ValueType2>
    void
    copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                            const ValueType1 *     valueType1Arr,
                            ValueType2 *           valueType2Arr);


    template <typename ValueType>
    void
    stridedCopyToBlock(const dftfe::size_type contiguousBlockSize,
                       const dftfe::size_type numContiguousBlocks,
                       const ValueType * copyFromVec,
                       ValueType *       copyToVecBlock,
                       const dftfe::global_size_type
                         *copyFromVecStartingContiguousBlockIds);



    template <typename ValueType>
    void
    stridedCopyFromBlock(const dftfe::size_type contiguousBlockSize,
                       const dftfe::size_type numContiguousBlocks,
                       const ValueType * copyFromVecBlock,
                       ValueType *       copyToVec,
                       const dftfe::global_size_type
                         *copyFromVecStartingContiguousBlockIds);   

    template <typename ValueType>
    void
    stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                       const dftfe::size_type blockSizeFrom,
                       const dftfe::size_type numBlocks,
                       const dftfe::size_type startingId,
                       const ValueType * copyFromVec,
                       ValueType *       copyToVec);



    template <typename ValueType>
    void
    stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                       const dftfe::size_type blockSizeFrom,
                       const dftfe::size_type numBlocks,
                       const dftfe::size_type startingId,
                       const ValueType * copyFromVec,
                       ValueType *       copyToVec);    


    template <typename NumberType>
    void axpby(const dftfe::size_type     n,
               const NumberType *x,
               NumberType *      y,
               const NumberType  a,
               const NumberType  b);


    template <typename NumberType>
    void ascal(const dftfe::size_type     n,
               NumberType *      x,
               const NumberType  a);

    void
    add(double *                          y,
        const double *                    x,
        const double                      alpha,
        const int                         size,
        dftfe::utils::deviceBlasHandle_t &deviceBlasHandle);

    double
    l2_norm(const double *                    x,
            const int                         size,
            const MPI_Comm &                  mpi_communicator,
            dftfe::utils::deviceBlasHandle_t &deviceBlasHandle);

    double
    dot(const double *                    x,
        const double *                    y,
        const int                         size,
        const MPI_Comm &                  mpi_communicator,
        dftfe::utils::deviceBlasHandle_t &deviceBlasHandle);

    template <typename NumberType>
    void
    sadd(NumberType *y, NumberType *x, const NumberType beta, const int size);

  } // namespace deviceKernelsGeneric

} // namespace dftfe

#  endif
#endif
