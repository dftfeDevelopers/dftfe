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
                                   double *               copyToVec) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyToBlockConstantStride(const dftfe::size_type      blockSizeTo,
                                   const dftfe::size_type      blockSizeFrom,
                                   const dftfe::size_type      numBlocks,
                                   const dftfe::size_type      startingId,
                                   const std::complex<double> *copyFromVec,
                                   std::complex<double> *      copyToVec) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                   const dftfe::size_type blockSizeFrom,
                                   const dftfe::size_type numBlocks,
                                   const dftfe::size_type startingId,
                                   const float *          copyFromVec,
                                   float *                copyToVec) const;


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyToBlockConstantStride(const dftfe::size_type     blockSizeTo,
                                   const dftfe::size_type     blockSizeFrom,
                                   const dftfe::size_type     numBlocks,
                                   const dftfe::size_type     startingId,
                                   const std::complex<float> *copyFromVec,
                                   std::complex<float> *      copyToVec) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                   const dftfe::size_type blockSizeFrom,
                                   const dftfe::size_type numBlocks,
                                   const dftfe::size_type startingId,
                                   const double *         copyFromVec,
                                   float *                copyToVec) const;
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyToBlockConstantStride(const dftfe::size_type blockSizeTo,
                                   const dftfe::size_type blockSizeFrom,
                                   const dftfe::size_type numBlocks,
                                   const dftfe::size_type startingId,
                                   const float *          copyFromVec,
                                   double *               copyToVec) const;
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyToBlockConstantStride(const dftfe::size_type      blockSizeTo,
                                   const dftfe::size_type      blockSizeFrom,
                                   const dftfe::size_type      numBlocks,
                                   const dftfe::size_type      startingId,
                                   const std::complex<double> *copyFromVec,
                                   std::complex<float> *       copyToVec) const;
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyToBlockConstantStride(const dftfe::size_type     blockSizeTo,
                                   const dftfe::size_type     blockSizeFrom,
                                   const dftfe::size_type     numBlocks,
                                   const dftfe::size_type     startingId,
                                   const std::complex<float> *copyFromVec,
                                   std::complex<double> *     copyToVec) const;
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


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::hadamardProduct(
  const unsigned int m,
  const double *     X,
  const double *     Y,
  double *           output) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::hadamardProduct(
  const unsigned int m,
  const float *      X,
  const float *      Y,
  float *            output) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::hadamardProductWithConj(
  const unsigned int m,
  const double *     X,
  const double *     Y,
  double *           output) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::hadamardProductWithConj(
  const unsigned int m,
  const float *      X,
  const float *      Y,
  float *            output) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::hadamardProductWithConj(
  const unsigned int          m,
  const std::complex<double> *X,
  const std::complex<double> *Y,
  std::complex<double> *      output) const;


// stridedBlockScaleColumnWise
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleColumnWise(
  const dftfe::size_type contiguousBlockSize,
  const dftfe::size_type numContiguousBlocks,
  const double *         beta,
  double *               x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleColumnWise(
  const dftfe::size_type contiguousBlockSize,
  const dftfe::size_type numContiguousBlocks,
  const float *          beta,
  float *                x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleColumnWise(
  const dftfe::size_type     contiguousBlockSize,
  const dftfe::size_type     numContiguousBlocks,
  const std::complex<float> *beta,
  std::complex<float> *      x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleColumnWise(
  const dftfe::size_type      contiguousBlockSize,
  const dftfe::size_type      numContiguousBlocks,
  const std::complex<double> *beta,
  std::complex<double> *      x);

// for stridedBlockScaleAndAddColumnWise
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedBlockScaleAndAddColumnWise(const dftfe::size_type contiguousBlockSize,
                                    const dftfe::size_type numContiguousBlocks,
                                    const double *         x,
                                    const double *         beta,
                                    double *               y);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedBlockScaleAndAddColumnWise(const dftfe::size_type contiguousBlockSize,
                                    const dftfe::size_type numContiguousBlocks,
                                    const float *          x,
                                    const float *          beta,
                                    float *                y);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedBlockScaleAndAddColumnWise(const dftfe::size_type contiguousBlockSize,
                                    const dftfe::size_type numContiguousBlocks,
                                    const std::complex<double> *x,
                                    const std::complex<double> *beta,
                                    std::complex<double> *      y);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedBlockScaleAndAddColumnWise(const dftfe::size_type contiguousBlockSize,
                                    const dftfe::size_type numContiguousBlocks,
                                    const std::complex<float> *x,
                                    const std::complex<float> *beta,
                                    std::complex<float> *      y);

// for stridedBlockScaleAndAddTwoVecColumnWise
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedBlockScaleAndAddTwoVecColumnWise(
    const dftfe::size_type contiguousBlockSize,
    const dftfe::size_type numContiguousBlocks,
    const double *         x,
    const double *         alpha,
    const double *         y,
    const double *         beta,
    double *               z);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedBlockScaleAndAddTwoVecColumnWise(
    const dftfe::size_type contiguousBlockSize,
    const dftfe::size_type numContiguousBlocks,
    const float *          x,
    const float *          alpha,
    const float *          y,
    const float *          beta,
    float *                z);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedBlockScaleAndAddTwoVecColumnWise(
    const dftfe::size_type      contiguousBlockSize,
    const dftfe::size_type      numContiguousBlocks,
    const std::complex<double> *x,
    const std::complex<double> *alpha,
    const std::complex<double> *y,
    const std::complex<double> *beta,
    std::complex<double> *      z);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedBlockScaleAndAddTwoVecColumnWise(
    const dftfe::size_type     contiguousBlockSize,
    const dftfe::size_type     numContiguousBlocks,
    const std::complex<float> *x,
    const std::complex<float> *alpha,
    const std::complex<float> *y,
    const std::complex<float> *beta,
    std::complex<float> *      z);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::addVecOverContinuousIndex(
  const dftfe::size_type numContiguousBlocks,
  const dftfe::size_type contiguousBlockSize,
  const double *         input1,
  const double *         input2,
  double *               output);

// MultiVectorXDot
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::MultiVectorXDot(
  const unsigned int contiguousBlockSize,
  const unsigned int numContiguousBlocks,
  const double *     X,
  const double *     Y,
  const double *     onesVec,
  double *           tempVector,
  double *           tempResults,
  double *           result) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::MultiVectorXDot(
  const unsigned int contiguousBlockSize,
  const unsigned int numContiguousBlocks,
  const double *     X,
  const double *     Y,
  const double *     onesVec,
  double *           tempVector,
  double *           tempResults,
  const MPI_Comm &   mpi_communicator,
  double *           result) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::MultiVectorXDot(
  const unsigned int          contiguousBlockSize,
  const unsigned int          numContiguousBlocks,
  const std::complex<double> *X,
  const std::complex<double> *Y,
  const std::complex<double> *onesVec,
  std::complex<double> *      tempVector,
  std::complex<double> *      tempResults,
  std::complex<double> *      result) const;
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::MultiVectorXDot(
  const unsigned int          contiguousBlockSize,
  const unsigned int          numContiguousBlocks,
  const std::complex<double> *X,
  const std::complex<double> *Y,
  const std::complex<double> *onesVec,
  std::complex<double> *      tempVector,
  std::complex<double> *      tempResults,
  const MPI_Comm &            mpi_communicator,
  std::complex<double> *      result) const;

// strided copy from block constant stride
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                     const dftfe::size_type blockSizeFrom,
                                     const dftfe::size_type numBlocks,
                                     const dftfe::size_type startingId,
                                     const double *         copyFromVec,
                                     double *               copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                     const dftfe::size_type blockSizeFrom,
                                     const dftfe::size_type numBlocks,
                                     const dftfe::size_type startingId,
                                     const float *          copyFromVec,
                                     float *                copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyFromBlockConstantStride(const dftfe::size_type      blockSizeTo,
                                     const dftfe::size_type      blockSizeFrom,
                                     const dftfe::size_type      numBlocks,
                                     const dftfe::size_type      startingId,
                                     const std::complex<double> *copyFromVec,
                                     std::complex<double> *      copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyFromBlockConstantStride(const dftfe::size_type     blockSizeTo,
                                     const dftfe::size_type     blockSizeFrom,
                                     const dftfe::size_type     numBlocks,
                                     const dftfe::size_type     startingId,
                                     const std::complex<float> *copyFromVec,
                                     std::complex<float> *      copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                     const dftfe::size_type blockSizeFrom,
                                     const dftfe::size_type numBlocks,
                                     const dftfe::size_type startingId,
                                     const double *         copyFromVec,
                                     float *                copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyFromBlockConstantStride(const dftfe::size_type blockSizeTo,
                                     const dftfe::size_type blockSizeFrom,
                                     const dftfe::size_type numBlocks,
                                     const dftfe::size_type startingId,
                                     const float *          copyFromVec,
                                     double *               copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyFromBlockConstantStride(const dftfe::size_type      blockSizeTo,
                                     const dftfe::size_type      blockSizeFrom,
                                     const dftfe::size_type      numBlocks,
                                     const dftfe::size_type      startingId,
                                     const std::complex<double> *copyFromVec,
                                     std::complex<float> *       copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyFromBlockConstantStride(const dftfe::size_type     blockSizeTo,
                                     const dftfe::size_type     blockSizeFrom,
                                     const dftfe::size_type     numBlocks,
                                     const dftfe::size_type     startingId,
                                     const std::complex<float> *copyFromVec,
                                     std::complex<double> *     copyToVec);
// strided copy  constant stride
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
  const dftfe::size_type blockSize,
  const dftfe::size_type strideTo,
  const dftfe::size_type strideFrom,
  const dftfe::size_type numBlocks,
  const dftfe::size_type startingToId,
  const dftfe::size_type startingFromId,
  const double *         copyFromVec,
  double *               copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
  const dftfe::size_type blockSize,
  const dftfe::size_type strideTo,
  const dftfe::size_type strideFrom,
  const dftfe::size_type numBlocks,
  const dftfe::size_type startingToId,
  const dftfe::size_type startingFromId,
  const float *          copyFromVec,
  float *                copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
  const dftfe::size_type      blockSize,
  const dftfe::size_type      strideTo,
  const dftfe::size_type      strideFrom,
  const dftfe::size_type      numBlocks,
  const dftfe::size_type      startingToId,
  const dftfe::size_type      startingFromId,
  const std::complex<double> *copyFromVec,
  std::complex<double> *      copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
  const dftfe::size_type     blockSize,
  const dftfe::size_type     strideTo,
  const dftfe::size_type     strideFrom,
  const dftfe::size_type     numBlocks,
  const dftfe::size_type     startingToId,
  const dftfe::size_type     startingFromId,
  const std::complex<float> *copyFromVec,
  std::complex<float> *      copyToVec);


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
  const dftfe::size_type blockSize,
  const dftfe::size_type strideTo,
  const dftfe::size_type strideFrom,
  const dftfe::size_type numBlocks,
  const dftfe::size_type startingToId,
  const dftfe::size_type startingFromId,
  const double *         copyFromVec,
  float *                copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
  const dftfe::size_type blockSize,
  const dftfe::size_type strideTo,
  const dftfe::size_type strideFrom,
  const dftfe::size_type numBlocks,
  const dftfe::size_type startingToId,
  const dftfe::size_type startingFromId,
  const float *          copyFromVec,
  double *               copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
  const dftfe::size_type      blockSize,
  const dftfe::size_type      strideTo,
  const dftfe::size_type      strideFrom,
  const dftfe::size_type      numBlocks,
  const dftfe::size_type      startingToId,
  const dftfe::size_type      startingFromId,
  const std::complex<double> *copyFromVec,
  std::complex<float> *       copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
  const dftfe::size_type     blockSize,
  const dftfe::size_type     strideTo,
  const dftfe::size_type     strideFrom,
  const dftfe::size_type     numBlocks,
  const dftfe::size_type     startingToId,
  const dftfe::size_type     startingFromId,
  const std::complex<float> *copyFromVec,
  std::complex<double> *     copyToVec);


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2ArrDeviceCall(
    const dftfe::size_type             size,
    const double *                     valueType1Arr,
    std::complex<double> *             valueType2Arr,
    const dftfe::utils::deviceStream_t streamId);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2ArrDeviceCall(
    const dftfe::size_type             size,
    const double *                     valueType1Arr,
    std::complex<float> *              valueType2Arr,
    const dftfe::utils::deviceStream_t streamId);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2ArrDeviceCall(
    const dftfe::size_type             size,
    const double *                     valueType1Arr,
    double *                           valueType2Arr,
    const dftfe::utils::deviceStream_t streamId);
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2ArrDeviceCall(
    const dftfe::size_type             size,
    const double *                     valueType1Arr,
    float *                            valueType2Arr,
    const dftfe::utils::deviceStream_t streamId);
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2ArrDeviceCall(
    const dftfe::size_type             size,
    const std::complex<double> *       valueType1Arr,
    std::complex<float> *              valueType2Arr,
    const dftfe::utils::deviceStream_t streamId);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2ArrDeviceCall(
    const dftfe::size_type             size,
    const std::complex<float> *        valueType1Arr,
    std::complex<double> *             valueType2Arr,
    const dftfe::utils::deviceStream_t streamId);


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2ArrDeviceCall(
    const dftfe::size_type             size,
    const float *                      valueType1Arr,
    float *                            valueType2Arr,
    const dftfe::utils::deviceStream_t streamId);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2ArrDeviceCall(
    const dftfe::size_type             size,
    const std::complex<float> *        valueType1Arr,
    std::complex<float> *              valueType2Arr,
    const dftfe::utils::deviceStream_t streamId);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2ArrDeviceCall(
    const dftfe::size_type             size,
    const float *                      valueType1Arr,
    double *                           valueType2Arr,
    const dftfe::utils::deviceStream_t streamId);
