template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleCopy(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const double                   a,
  const double *                 s,
  const double *                 copyFromVec,
  double *                       copyToVecBlock,
  const dftfe::global_size_type *addToVecStartingContiguousBlockIds);
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleCopy(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const double                   a,
  const double *                 s,
  const std::complex<double> *   copyFromVec,
  std::complex<double> *         copyToVecBlock,
  const dftfe::global_size_type *addToVecStartingContiguousBlockIds);
// for stridedBlockScale
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::size_type contiguousBlockSize,
  const dftfe::size_type numContiguousBlocks,
  const double           a,
  const double *         s,
  double *               x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::size_type contiguousBlockSize,
  const dftfe::size_type numContiguousBlocks,
  const float            a,
  const float *          s,
  float *                x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::size_type      contiguousBlockSize,
  const dftfe::size_type      numContiguousBlocks,
  const std::complex<double>  a,
  const std::complex<double> *s,
  std::complex<double> *      x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::size_type     contiguousBlockSize,
  const dftfe::size_type     numContiguousBlocks,
  const std::complex<float>  a,
  const std::complex<float> *s,
  std::complex<float> *      x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::size_type contiguousBlockSize,
  const dftfe::size_type numContiguousBlocks,
  const double           a,
  const double *         s,
  float *                x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::size_type contiguousBlockSize,
  const dftfe::size_type numContiguousBlocks,
  const float            a,
  const float *          s,
  double *               x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::size_type      contiguousBlockSize,
  const dftfe::size_type      numContiguousBlocks,
  const std::complex<double>  a,
  const std::complex<double> *s,
  std::complex<float> *       x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::size_type     contiguousBlockSize,
  const dftfe::size_type     numContiguousBlocks,
  const std::complex<float>  a,
  const std::complex<float> *s,
  std::complex<double> *     x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::size_type contiguousBlockSize,
  const dftfe::size_type numContiguousBlocks,
  const double           a,
  const double *         s,
  std::complex<double> * x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::size_type contiguousBlockSize,
  const dftfe::size_type numContiguousBlocks,
  const double           a,
  const double *         s,
  std::complex<float> *  x);
// for xscal
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
  double *               x,
  const double           a,
  const dftfe::size_type n) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
  float *                x,
  const float            a,
  const dftfe::size_type n) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
  std::complex<double> *     x,
  const std::complex<double> a,
  const dftfe::size_type     n) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
  std::complex<float> *     x,
  const std::complex<float> a,
  const dftfe::size_type    n) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
  std::complex<double> * x,
  const double           a,
  const dftfe::size_type n) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const double *                 copyFromVec,
  double *                       copyToVecBlock,
  const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const double *                 copyFromVec,
  float *                        copyToVecBlock,
  const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const float *                  copyFromVec,
  float *                        copyToVecBlock,
  const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const std::complex<double> *   copyFromVec,
  std::complex<double> *         copyToVecBlock,
  const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const std::complex<double> *   copyFromVec,
  std::complex<float> *          copyToVecBlock,
  const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const std::complex<float> *    copyFromVec,
  std::complex<float> *          copyToVecBlock,
  const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

// strided copy from block
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const double *                 copyFromVecBlock,
  double *                       copyToVec,
  const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const float *                  copyFromVecBlock,
  float *                        copyToVec,
  const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const std::complex<double> *   copyFromVecBlock,
  std::complex<double> *         copyToVec,
  const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const std::complex<float> *    copyFromVecBlock,
  std::complex<float> *          copyToVec,
  const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const double *                 copyFromVecBlock,
  float *                        copyToVec,
  const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const float *                  copyFromVecBlock,
  double *                       copyToVec,
  const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const std::complex<double> *   copyFromVecBlock,
  std::complex<float> *          copyToVec,
  const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const std::complex<float> *    copyFromVecBlock,
  std::complex<double> *         copyToVec,
  const dftfe::global_size_type *copyFromVecStartingContiguousBlockIds);


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                   const double *         valueType1Arr,
                                   std::complex<double> * valueType2Arr);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                   const double *         valueType1Arr,
                                   std::complex<float> *  valueType2Arr);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                   const double *         valueType1Arr,
                                   double *               valueType2Arr);
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2Arr(const dftfe::size_type size,
                                   const double *         valueType1Arr,
                                   float *                valueType2Arr);
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2Arr(const dftfe::size_type      size,
                                   const std::complex<double> *valueType1Arr,
                                   std::complex<float> *       valueType2Arr);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                   const dftfe::size_type blockSizeFrom,
                                   const dftfe::size_type numBlocks,
                                   const dftfe::size_type startingId,
                                   const double *         copyFromVec,
                                   double *               copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyToBlockConstantStride(const dftfe::size_type      blockSizeTo,
                                   const dftfe::size_type      blockSizeFrom,
                                   const dftfe::size_type      numBlocks,
                                   const dftfe::size_type      startingId,
                                   const std::complex<double> *copyFromVec,
                                   std::complex<double> *      copyToVec);
// axpyStridedBlockAtomicAdd
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const double *                 addFromVec,
  double *                       addToVec,
  const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const std::complex<double> *   addFromVec,
  std::complex<double> *         addToVec,
  const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const double                   a,
  const double *                 s,
  const double *                 addFromVec,
  double *                       addToVec,
  const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const double                   a,
  const double *                 s,
  const std::complex<double> *   addFromVec,
  std::complex<double> *         addToVec,
  const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const double                   a,
  const double *                 s,
  const float *                  addFromVec,
  float *                        addToVec,
  const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const double                   a,
  const double *                 s,
  const std::complex<float> *    addFromVec,
  std::complex<float> *          addToVec,
  const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const float                    a,
  const float *                  s,
  const float *                  addFromVec,
  float *                        addToVec,
  const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::size_type         contiguousBlockSize,
  const dftfe::size_type         numContiguousBlocks,
  const float                    a,
  const float *                  s,
  const std::complex<float> *    addFromVec,
  std::complex<float> *          addToVec,
  const dftfe::global_size_type *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpby(const unsigned int n,
                                                      const double       alpha,
                                                      const double *     x,
                                                      const double       beta,
                                                      double *y) const;


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpby(
  const unsigned int          n,
  const double                alpha,
  const std::complex<double> *x,
  const double                beta,
  std::complex<double> *      y) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpby(const unsigned int n,
                                                      const double       alpha,
                                                      const float *      x,
                                                      const double       beta,
                                                      float *y) const;


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpby(
  const unsigned int         n,
  const double               alpha,
  const std::complex<float> *x,
  const double               beta,
  std::complex<float> *      y) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::ApaBD(const unsigned int m,
                                                      const unsigned int n,
                                                      const double       alpha,
                                                      const double *     A,
                                                      const double *     B,
                                                      const double *     D,
                                                      double *C) const;
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::ApaBD(
  const unsigned int          m,
  const unsigned int          n,
  const double                alpha,
  const std::complex<double> *A,
  const std::complex<double> *B,
  const double *              D,
  std::complex<double> *      C) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::ApaBD(const unsigned int m,
                                                      const unsigned int n,
                                                      const double       alpha,
                                                      const float *      A,
                                                      const double *     B,
                                                      const double *     D,
                                                      float *C) const;
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::ApaBD(const unsigned int m,
                                                      const unsigned int n,
                                                      const double       alpha,
                                                      const float *      A,
                                                      const double *     B,
                                                      const double *     D,
                                                      double *C) const;
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::ApaBD(
  const unsigned int          m,
  const unsigned int          n,
  const double                alpha,
  const std::complex<float> * A,
  const std::complex<double> *B,
  const double *              D,
  std::complex<float> *       C) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::ApaBD(
  const unsigned int          m,
  const unsigned int          n,
  const double                alpha,
  const std::complex<float> * A,
  const std::complex<double> *B,
  const double *              D,
  std::complex<double> *      C) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::copyRealArrsToComplexArr(
  const dftfe::size_type size,
  const double *         realArr,
  const double *         imagArr,
  std::complex<double> * complexArr);
