template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleCopy(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const double       a,
  const double      *s,
  const double      *copyFromVec,
  double            *copyToVecBlock,
  const dftfe::uInt *addToVecStartingContiguousBlockIds);
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleCopy(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const double                a,
  const double               *s,
  const std::complex<double> *copyFromVec,
  std::complex<double>       *copyToVecBlock,
  const dftfe::uInt          *addToVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleCopy(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const double       a,
  const double      *s,
  const float       *copyFromVec,
  float             *copyToVecBlock,
  const dftfe::uInt *addToVecStartingContiguousBlockIds);
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleCopy(
  const dftfe::uInt          contiguousBlockSize,
  const dftfe::uInt          numContiguousBlocks,
  const double               a,
  const double              *s,
  const std::complex<float> *copyFromVec,
  std::complex<float>       *copyToVecBlock,
  const dftfe::uInt         *addToVecStartingContiguousBlockIds);
// for stridedBlockScale
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::uInt contiguousBlockSize,
  const dftfe::uInt numContiguousBlocks,
  const double      a,
  const double     *s,
  double           *x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::uInt contiguousBlockSize,
  const dftfe::uInt numContiguousBlocks,
  const float       a,
  const float      *s,
  float            *x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const std::complex<double>  a,
  const std::complex<double> *s,
  std::complex<double>       *x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::uInt          contiguousBlockSize,
  const dftfe::uInt          numContiguousBlocks,
  const std::complex<float>  a,
  const std::complex<float> *s,
  std::complex<float>       *x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::uInt contiguousBlockSize,
  const dftfe::uInt numContiguousBlocks,
  const double      a,
  const double     *s,
  float            *x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::uInt contiguousBlockSize,
  const dftfe::uInt numContiguousBlocks,
  const float       a,
  const float      *s,
  double           *x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const std::complex<double>  a,
  const std::complex<double> *s,
  std::complex<float>        *x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::uInt          contiguousBlockSize,
  const dftfe::uInt          numContiguousBlocks,
  const std::complex<float>  a,
  const std::complex<float> *s,
  std::complex<double>      *x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::uInt     contiguousBlockSize,
  const dftfe::uInt     numContiguousBlocks,
  const double          a,
  const double         *s,
  std::complex<double> *x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScale(
  const dftfe::uInt    contiguousBlockSize,
  const dftfe::uInt    numContiguousBlocks,
  const double         a,
  const double        *s,
  std::complex<float> *x);
// for xscal
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
  double           *x,
  const double      a,
  const dftfe::uInt n) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
  float            *x,
  const float       a,
  const dftfe::uInt n) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
  std::complex<double>      *x,
  const std::complex<double> a,
  const dftfe::uInt          n) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
  std::complex<float>      *x,
  const std::complex<float> a,
  const dftfe::uInt         n) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::xscal(
  std::complex<double> *x,
  const double          a,
  const dftfe::uInt     n) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const double      *copyFromVec,
  double            *copyToVecBlock,
  const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const double      *copyFromVec,
  float             *copyToVecBlock,
  const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const float       *copyFromVec,
  float             *copyToVecBlock,
  const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const std::complex<double> *copyFromVec,
  std::complex<double>       *copyToVecBlock,
  const dftfe::uInt          *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const std::complex<double> *copyFromVec,
  std::complex<float>        *copyToVecBlock,
  const dftfe::uInt          *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
  const dftfe::uInt          contiguousBlockSize,
  const dftfe::uInt          numContiguousBlocks,
  const std::complex<float> *copyFromVec,
  std::complex<float>       *copyToVecBlock,
  const dftfe::uInt         *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const dftfe::uInt  startingVecId,
  const double      *copyFromVec,
  double            *copyToVecBlock,
  const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const dftfe::uInt  startingVecId,
  const double      *copyFromVec,
  float             *copyToVecBlock,
  const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const dftfe::uInt  startingVecId,
  const float       *copyFromVec,
  float             *copyToVecBlock,
  const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const dftfe::uInt           startingVecId,
  const std::complex<double> *copyFromVec,
  std::complex<double>       *copyToVecBlock,
  const dftfe::uInt          *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const dftfe::uInt           startingVecId,
  const std::complex<double> *copyFromVec,
  std::complex<float>        *copyToVecBlock,
  const dftfe::uInt          *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyToBlock(
  const dftfe::uInt          contiguousBlockSize,
  const dftfe::uInt          numContiguousBlocks,
  const dftfe::uInt          startingVecId,
  const std::complex<float> *copyFromVec,
  std::complex<float>       *copyToVecBlock,
  const dftfe::uInt         *copyFromVecStartingContiguousBlockIds);



// strided copy from block
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const double      *copyFromVecBlock,
  double            *copyToVec,
  const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const float       *copyFromVecBlock,
  float             *copyToVec,
  const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const std::complex<double> *copyFromVecBlock,
  std::complex<double>       *copyToVec,
  const dftfe::uInt          *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
  const dftfe::uInt          contiguousBlockSize,
  const dftfe::uInt          numContiguousBlocks,
  const std::complex<float> *copyFromVecBlock,
  std::complex<float>       *copyToVec,
  const dftfe::uInt         *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const double      *copyFromVecBlock,
  float             *copyToVec,
  const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const float       *copyFromVecBlock,
  double            *copyToVec,
  const dftfe::uInt *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const std::complex<double> *copyFromVecBlock,
  std::complex<float>        *copyToVec,
  const dftfe::uInt          *copyFromVecStartingContiguousBlockIds);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyFromBlock(
  const dftfe::uInt          contiguousBlockSize,
  const dftfe::uInt          numContiguousBlocks,
  const std::complex<float> *copyFromVecBlock,
  std::complex<double>      *copyToVec,
  const dftfe::uInt         *copyFromVecStartingContiguousBlockIds);


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2Arr(const dftfe::uInt     size,
                                   const double         *valueType1Arr,
                                   std::complex<double> *valueType2Arr);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2Arr(const dftfe::uInt           size,
                                   const std::complex<double> *valueType1Arr,
                                   std::complex<double>       *valueType2Arr);
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2Arr(const dftfe::uInt    size,
                                   const double        *valueType1Arr,
                                   std::complex<float> *valueType2Arr);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2Arr(const dftfe::uInt size,
                                   const double     *valueType1Arr,
                                   double           *valueType2Arr);
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2Arr(const dftfe::uInt size,
                                   const double     *valueType1Arr,
                                   float            *valueType2Arr);
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2Arr(const dftfe::uInt size,
                                   const float      *valueType1Arr,
                                   double           *valueType2Arr);
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2Arr(const dftfe::uInt           size,
                                   const std::complex<double> *valueType1Arr,
                                   std::complex<float>        *valueType2Arr);
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyBlockDiagonalValueType1OffDiagonalValueType2FromValueType1Arr(
    const dftfe::uInt B,
    const dftfe::uInt DRem,
    const dftfe::uInt D,
    const double     *valueType1SrcArray,
    double           *valueType1DstArray,
    float            *valueType2DstArray);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyBlockDiagonalValueType1OffDiagonalValueType2FromValueType1Arr(
    const dftfe::uInt           B,
    const dftfe::uInt           DRem,
    const dftfe::uInt           D,
    const std::complex<double> *valueType1SrcArray,
    std::complex<double>       *valueType1DstArray,
    std::complex<float>        *valueType2DstArray);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyToBlockConstantStride(const dftfe::uInt blockSizeTo,
                                   const dftfe::uInt blockSizeFrom,
                                   const dftfe::uInt numBlocks,
                                   const dftfe::uInt startingId,
                                   const double     *copyFromVec,
                                   double           *copyToVec) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyToBlockConstantStride(const dftfe::uInt           blockSizeTo,
                                   const dftfe::uInt           blockSizeFrom,
                                   const dftfe::uInt           numBlocks,
                                   const dftfe::uInt           startingId,
                                   const std::complex<double> *copyFromVec,
                                   std::complex<double>       *copyToVec) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyToBlockConstantStride(const dftfe::uInt blockSizeTo,
                                   const dftfe::uInt blockSizeFrom,
                                   const dftfe::uInt numBlocks,
                                   const dftfe::uInt startingId,
                                   const float      *copyFromVec,
                                   float            *copyToVec) const;


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyToBlockConstantStride(const dftfe::uInt          blockSizeTo,
                                   const dftfe::uInt          blockSizeFrom,
                                   const dftfe::uInt          numBlocks,
                                   const dftfe::uInt          startingId,
                                   const std::complex<float> *copyFromVec,
                                   std::complex<float>       *copyToVec) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyToBlockConstantStride(const dftfe::uInt blockSizeTo,
                                   const dftfe::uInt blockSizeFrom,
                                   const dftfe::uInt numBlocks,
                                   const dftfe::uInt startingId,
                                   const double     *copyFromVec,
                                   float            *copyToVec) const;
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyToBlockConstantStride(const dftfe::uInt blockSizeTo,
                                   const dftfe::uInt blockSizeFrom,
                                   const dftfe::uInt numBlocks,
                                   const dftfe::uInt startingId,
                                   const float      *copyFromVec,
                                   double           *copyToVec) const;
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyToBlockConstantStride(const dftfe::uInt           blockSizeTo,
                                   const dftfe::uInt           blockSizeFrom,
                                   const dftfe::uInt           numBlocks,
                                   const dftfe::uInt           startingId,
                                   const std::complex<double> *copyFromVec,
                                   std::complex<float>        *copyToVec) const;
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyToBlockConstantStride(const dftfe::uInt          blockSizeTo,
                                   const dftfe::uInt          blockSizeFrom,
                                   const dftfe::uInt          numBlocks,
                                   const dftfe::uInt          startingId,
                                   const std::complex<float> *copyFromVec,
                                   std::complex<double>      *copyToVec) const;
// axpyStridedBlockAtomicAdd
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const double      *addFromVec,
  double            *addToVec,
  const dftfe::uInt *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const std::complex<double> *addFromVec,
  std::complex<double>       *addToVec,
  const dftfe::uInt          *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const double       a,
  const double      *s,
  const double      *addFromVec,
  double            *addToVec,
  const dftfe::uInt *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const double                a,
  const double               *s,
  const std::complex<double> *addFromVec,
  std::complex<double>       *addToVec,
  const dftfe::uInt          *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const double       a,
  const double      *s,
  const float       *addFromVec,
  float             *addToVec,
  const dftfe::uInt *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::uInt          contiguousBlockSize,
  const dftfe::uInt          numContiguousBlocks,
  const double               a,
  const double              *s,
  const std::complex<float> *addFromVec,
  std::complex<float>       *addToVec,
  const dftfe::uInt         *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const float        a,
  const float       *s,
  const float       *addFromVec,
  float             *addToVec,
  const dftfe::uInt *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::uInt          contiguousBlockSize,
  const dftfe::uInt          numContiguousBlocks,
  const float                a,
  const float               *s,
  const std::complex<float> *addFromVec,
  std::complex<float>       *addToVec,
  const dftfe::uInt         *addToVecStartingContiguousBlockIds) const;


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const double       a,
  const double      *addFromVec,
  double            *addToVec,
  const dftfe::uInt *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const double                a,
  const std::complex<double> *addFromVec,
  std::complex<double>       *addToVec,
  const dftfe::uInt          *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const double       a,
  const float       *addFromVec,
  float             *addToVec,
  const dftfe::uInt *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::uInt          contiguousBlockSize,
  const dftfe::uInt          numContiguousBlocks,
  const double               a,
  const std::complex<float> *addFromVec,
  std::complex<float>       *addToVec,
  const dftfe::uInt         *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::uInt  contiguousBlockSize,
  const dftfe::uInt  numContiguousBlocks,
  const float        a,
  const float       *addFromVec,
  float             *addToVec,
  const dftfe::uInt *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpyStridedBlockAtomicAdd(
  const dftfe::uInt          contiguousBlockSize,
  const dftfe::uInt          numContiguousBlocks,
  const float                a,
  const std::complex<float> *addFromVec,
  std::complex<float>       *addToVec,
  const dftfe::uInt         *addToVecStartingContiguousBlockIds) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpby(const dftfe::uInt n,
                                                      const double      alpha,
                                                      const double     *x,
                                                      const double      beta,
                                                      double *y) const;


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpby(
  const dftfe::uInt           n,
  const double                alpha,
  const std::complex<double> *x,
  const double                beta,
  std::complex<double>       *y) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpby(const dftfe::uInt n,
                                                      const double      alpha,
                                                      const float      *x,
                                                      const double      beta,
                                                      float *y) const;


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::axpby(
  const dftfe::uInt          n,
  const double               alpha,
  const std::complex<float> *x,
  const double               beta,
  std::complex<float>       *y) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::ApaBD(const dftfe::uInt m,
                                                      const dftfe::uInt n,
                                                      const double      alpha,
                                                      const double     *A,
                                                      const double     *B,
                                                      const double     *D,
                                                      double *C) const;
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::ApaBD(
  const dftfe::uInt           m,
  const dftfe::uInt           n,
  const double                alpha,
  const std::complex<double> *A,
  const std::complex<double> *B,
  const double               *D,
  std::complex<double>       *C) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::ApaBD(const dftfe::uInt m,
                                                      const dftfe::uInt n,
                                                      const double      alpha,
                                                      const float      *A,
                                                      const double     *B,
                                                      const double     *D,
                                                      float *C) const;
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::ApaBD(const dftfe::uInt m,
                                                      const dftfe::uInt n,
                                                      const double      alpha,
                                                      const float      *A,
                                                      const double     *B,
                                                      const double     *D,
                                                      double *C) const;
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::ApaBD(
  const dftfe::uInt           m,
  const dftfe::uInt           n,
  const double                alpha,
  const std::complex<float>  *A,
  const std::complex<double> *B,
  const double               *D,
  std::complex<float>        *C) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::ApaBD(
  const dftfe::uInt           m,
  const dftfe::uInt           n,
  const double                alpha,
  const std::complex<float>  *A,
  const std::complex<double> *B,
  const double               *D,
  std::complex<double>       *C) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::copyRealArrsToComplexArr(
  const dftfe::uInt     size,
  const double         *realArr,
  const double         *imagArr,
  std::complex<double> *complexArr);


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::hadamardProduct(
  const dftfe::uInt m,
  const double     *X,
  const double     *Y,
  double           *output) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::hadamardProduct(
  const dftfe::uInt m,
  const float      *X,
  const float      *Y,
  float            *output) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::hadamardProductWithConj(
  const dftfe::uInt m,
  const double     *X,
  const double     *Y,
  double           *output) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::hadamardProductWithConj(
  const dftfe::uInt m,
  const float      *X,
  const float      *Y,
  float            *output) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::hadamardProductWithConj(
  const dftfe::uInt           m,
  const std::complex<double> *X,
  const std::complex<double> *Y,
  std::complex<double>       *output) const;


// stridedBlockScaleColumnWise
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleColumnWise(
  const dftfe::uInt contiguousBlockSize,
  const dftfe::uInt numContiguousBlocks,
  const double     *beta,
  double           *x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleColumnWise(
  const dftfe::uInt contiguousBlockSize,
  const dftfe::uInt numContiguousBlocks,
  const float      *beta,
  float            *x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleColumnWise(
  const dftfe::uInt          contiguousBlockSize,
  const dftfe::uInt          numContiguousBlocks,
  const std::complex<float> *beta,
  std::complex<float>       *x);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockScaleColumnWise(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const std::complex<double> *beta,
  std::complex<double>       *x);

// for stridedBlockScaleAndAddColumnWise
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedBlockScaleAndAddColumnWise(const dftfe::uInt contiguousBlockSize,
                                    const dftfe::uInt numContiguousBlocks,
                                    const double     *x,
                                    const double     *beta,
                                    double           *y);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedBlockScaleAndAddColumnWise(const dftfe::uInt contiguousBlockSize,
                                    const dftfe::uInt numContiguousBlocks,
                                    const float      *x,
                                    const float      *beta,
                                    float            *y);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedBlockScaleAndAddColumnWise(const dftfe::uInt contiguousBlockSize,
                                    const dftfe::uInt numContiguousBlocks,
                                    const std::complex<double> *x,
                                    const std::complex<double> *beta,
                                    std::complex<double>       *y);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedBlockScaleAndAddColumnWise(const dftfe::uInt contiguousBlockSize,
                                    const dftfe::uInt numContiguousBlocks,
                                    const std::complex<float> *x,
                                    const std::complex<float> *beta,
                                    std::complex<float>       *y);

// for stridedBlockScaleAndAddTwoVecColumnWise
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedBlockScaleAndAddTwoVecColumnWise(const dftfe::uInt contiguousBlockSize,
                                          const dftfe::uInt numContiguousBlocks,
                                          const double     *x,
                                          const double     *alpha,
                                          const double     *y,
                                          const double     *beta,
                                          double           *z);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedBlockScaleAndAddTwoVecColumnWise(const dftfe::uInt contiguousBlockSize,
                                          const dftfe::uInt numContiguousBlocks,
                                          const float      *x,
                                          const float      *alpha,
                                          const float      *y,
                                          const float      *beta,
                                          float            *z);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedBlockScaleAndAddTwoVecColumnWise(const dftfe::uInt contiguousBlockSize,
                                          const dftfe::uInt numContiguousBlocks,
                                          const std::complex<double> *x,
                                          const std::complex<double> *alpha,
                                          const std::complex<double> *y,
                                          const std::complex<double> *beta,
                                          std::complex<double>       *z);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedBlockScaleAndAddTwoVecColumnWise(const dftfe::uInt contiguousBlockSize,
                                          const dftfe::uInt numContiguousBlocks,
                                          const std::complex<float> *x,
                                          const std::complex<float> *alpha,
                                          const std::complex<float> *y,
                                          const std::complex<float> *beta,
                                          std::complex<float>       *z);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::addVecOverContinuousIndex(
  const dftfe::uInt numContiguousBlocks,
  const dftfe::uInt contiguousBlockSize,
  const double     *input1,
  const double     *input2,
  double           *output);

// MultiVectorXDot
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::MultiVectorXDot(
  const dftfe::uInt contiguousBlockSize,
  const dftfe::uInt numContiguousBlocks,
  const double     *X,
  const double     *Y,
  const double     *onesVec,
  double           *tempVector,
  double           *tempResults,
  double           *result) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::MultiVectorXDot(
  const dftfe::uInt contiguousBlockSize,
  const dftfe::uInt numContiguousBlocks,
  const double     *X,
  const double     *Y,
  const double     *onesVec,
  double           *tempVector,
  double           *tempResults,
  const MPI_Comm   &mpi_communicator,
  double           *result) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::MultiVectorXDot(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const std::complex<double> *X,
  const std::complex<double> *Y,
  const std::complex<double> *onesVec,
  std::complex<double>       *tempVector,
  std::complex<double>       *tempResults,
  std::complex<double>       *result) const;
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::MultiVectorXDot(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const std::complex<double> *X,
  const std::complex<double> *Y,
  const std::complex<double> *onesVec,
  std::complex<double>       *tempVector,
  std::complex<double>       *tempResults,
  const MPI_Comm             &mpi_communicator,
  std::complex<double>       *result) const;

// strided copy from block constant stride
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyFromBlockConstantStride(const dftfe::uInt blockSizeTo,
                                     const dftfe::uInt blockSizeFrom,
                                     const dftfe::uInt numBlocks,
                                     const dftfe::uInt startingId,
                                     const double     *copyFromVec,
                                     double           *copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyFromBlockConstantStride(const dftfe::uInt blockSizeTo,
                                     const dftfe::uInt blockSizeFrom,
                                     const dftfe::uInt numBlocks,
                                     const dftfe::uInt startingId,
                                     const float      *copyFromVec,
                                     float            *copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyFromBlockConstantStride(const dftfe::uInt           blockSizeTo,
                                     const dftfe::uInt           blockSizeFrom,
                                     const dftfe::uInt           numBlocks,
                                     const dftfe::uInt           startingId,
                                     const std::complex<double> *copyFromVec,
                                     std::complex<double>       *copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyFromBlockConstantStride(const dftfe::uInt          blockSizeTo,
                                     const dftfe::uInt          blockSizeFrom,
                                     const dftfe::uInt          numBlocks,
                                     const dftfe::uInt          startingId,
                                     const std::complex<float> *copyFromVec,
                                     std::complex<float>       *copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyFromBlockConstantStride(const dftfe::uInt blockSizeTo,
                                     const dftfe::uInt blockSizeFrom,
                                     const dftfe::uInt numBlocks,
                                     const dftfe::uInt startingId,
                                     const double     *copyFromVec,
                                     float            *copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyFromBlockConstantStride(const dftfe::uInt blockSizeTo,
                                     const dftfe::uInt blockSizeFrom,
                                     const dftfe::uInt numBlocks,
                                     const dftfe::uInt startingId,
                                     const float      *copyFromVec,
                                     double           *copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyFromBlockConstantStride(const dftfe::uInt           blockSizeTo,
                                     const dftfe::uInt           blockSizeFrom,
                                     const dftfe::uInt           numBlocks,
                                     const dftfe::uInt           startingId,
                                     const std::complex<double> *copyFromVec,
                                     std::complex<float>        *copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  stridedCopyFromBlockConstantStride(const dftfe::uInt          blockSizeTo,
                                     const dftfe::uInt          blockSizeFrom,
                                     const dftfe::uInt          numBlocks,
                                     const dftfe::uInt          startingId,
                                     const std::complex<float> *copyFromVec,
                                     std::complex<double>      *copyToVec);
// strided copy  constant stride
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
  const dftfe::uInt blockSize,
  const dftfe::uInt strideTo,
  const dftfe::uInt strideFrom,
  const dftfe::uInt numBlocks,
  const dftfe::uInt startingToId,
  const dftfe::uInt startingFromId,
  const double     *copyFromVec,
  double           *copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
  const dftfe::uInt blockSize,
  const dftfe::uInt strideTo,
  const dftfe::uInt strideFrom,
  const dftfe::uInt numBlocks,
  const dftfe::uInt startingToId,
  const dftfe::uInt startingFromId,
  const float      *copyFromVec,
  float            *copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
  const dftfe::uInt           blockSize,
  const dftfe::uInt           strideTo,
  const dftfe::uInt           strideFrom,
  const dftfe::uInt           numBlocks,
  const dftfe::uInt           startingToId,
  const dftfe::uInt           startingFromId,
  const std::complex<double> *copyFromVec,
  std::complex<double>       *copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
  const dftfe::uInt          blockSize,
  const dftfe::uInt          strideTo,
  const dftfe::uInt          strideFrom,
  const dftfe::uInt          numBlocks,
  const dftfe::uInt          startingToId,
  const dftfe::uInt          startingFromId,
  const std::complex<float> *copyFromVec,
  std::complex<float>       *copyToVec);


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
  const dftfe::uInt blockSize,
  const dftfe::uInt strideTo,
  const dftfe::uInt strideFrom,
  const dftfe::uInt numBlocks,
  const dftfe::uInt startingToId,
  const dftfe::uInt startingFromId,
  const double     *copyFromVec,
  float            *copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
  const dftfe::uInt blockSize,
  const dftfe::uInt strideTo,
  const dftfe::uInt strideFrom,
  const dftfe::uInt numBlocks,
  const dftfe::uInt startingToId,
  const dftfe::uInt startingFromId,
  const float      *copyFromVec,
  double           *copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
  const dftfe::uInt           blockSize,
  const dftfe::uInt           strideTo,
  const dftfe::uInt           strideFrom,
  const dftfe::uInt           numBlocks,
  const dftfe::uInt           startingToId,
  const dftfe::uInt           startingFromId,
  const std::complex<double> *copyFromVec,
  std::complex<float>        *copyToVec);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedCopyConstantStride(
  const dftfe::uInt          blockSize,
  const dftfe::uInt          strideTo,
  const dftfe::uInt          strideFrom,
  const dftfe::uInt          numBlocks,
  const dftfe::uInt          startingToId,
  const dftfe::uInt          startingFromId,
  const std::complex<float> *copyFromVec,
  std::complex<double>      *copyToVec);


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2ArrDeviceCall(
    const dftfe::uInt                  size,
    const double                      *valueType1Arr,
    std::complex<double>              *valueType2Arr,
    const dftfe::utils::deviceStream_t streamId);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2ArrDeviceCall(
    const dftfe::uInt                  size,
    const double                      *valueType1Arr,
    std::complex<float>               *valueType2Arr,
    const dftfe::utils::deviceStream_t streamId);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2ArrDeviceCall(
    const dftfe::uInt                  size,
    const double                      *valueType1Arr,
    double                            *valueType2Arr,
    const dftfe::utils::deviceStream_t streamId);
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2ArrDeviceCall(
    const dftfe::uInt                  size,
    const double                      *valueType1Arr,
    float                             *valueType2Arr,
    const dftfe::utils::deviceStream_t streamId);
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2ArrDeviceCall(
    const dftfe::uInt                  size,
    const std::complex<double>        *valueType1Arr,
    std::complex<float>               *valueType2Arr,
    const dftfe::utils::deviceStream_t streamId);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2ArrDeviceCall(
    const dftfe::uInt                  size,
    const std::complex<float>         *valueType1Arr,
    std::complex<double>              *valueType2Arr,
    const dftfe::utils::deviceStream_t streamId);


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2ArrDeviceCall(
    const dftfe::uInt                  size,
    const float                       *valueType1Arr,
    float                             *valueType2Arr,
    const dftfe::utils::deviceStream_t streamId);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2ArrDeviceCall(
    const dftfe::uInt                  size,
    const std::complex<float>         *valueType1Arr,
    std::complex<float>               *valueType2Arr,
    const dftfe::utils::deviceStream_t streamId);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::
  copyValueType1ArrToValueType2ArrDeviceCall(
    const dftfe::uInt                  size,
    const float                       *valueType1Arr,
    double                            *valueType2Arr,
    const dftfe::utils::deviceStream_t streamId);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockAxpy(
  const dftfe::uInt contiguousBlockSize,
  const dftfe::uInt numContiguousBlocks,
  const double     *addFromVec,
  const double     *scalingVector,
  const double      a,
  double           *addToVec) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockAxpy(
  const dftfe::uInt contiguousBlockSize,
  const dftfe::uInt numContiguousBlocks,
  const float      *addFromVec,
  const double     *scalingVector,
  const double      a,
  float            *addToVec) const;


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockAxpy(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const std::complex<double> *addFromVec,
  const std::complex<double> *scalingVector,
  const std::complex<double>  a,
  std::complex<double>       *addToVec) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockAxpy(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const std::complex<double> *addFromVec,
  const double               *scalingVector,
  const double                a,
  std::complex<double>       *addToVec) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockAxpy(
  const dftfe::uInt          contiguousBlockSize,
  const dftfe::uInt          numContiguousBlocks,
  const std::complex<float> *addFromVec,
  const double              *scalingVector,
  const double               a,
  std::complex<float>       *addToVec) const;
template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::rightDiagonalScale(
  const dftfe::uInt numberofVectors,
  const dftfe::uInt sizeOfVector,
  double           *X,
  double           *D);

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::rightDiagonalScale(
  const dftfe::uInt     numberofVectors,
  const dftfe::uInt     sizeOfVector,
  std::complex<double> *X,
  double               *D);


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockAxpBy(
  const dftfe::uInt contiguousBlockSize,
  const dftfe::uInt numContiguousBlocks,
  const double     *addFromVec,
  const double     *scalingVector,
  const double      a,
  const double      b,
  double           *addToVec) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockAxpBy(
  const dftfe::uInt contiguousBlockSize,
  const dftfe::uInt numContiguousBlocks,
  const float      *addFromVec,
  const double     *scalingVector,
  const double      a,
  const double      b,
  float            *addToVec) const;


template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockAxpBy(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const std::complex<double> *addFromVec,
  const std::complex<double> *scalingVector,
  const std::complex<double>  a,
  const std::complex<double>  b,
  std::complex<double>       *addToVec) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockAxpBy(
  const dftfe::uInt           contiguousBlockSize,
  const dftfe::uInt           numContiguousBlocks,
  const std::complex<double> *addFromVec,
  const double               *scalingVector,
  const double                a,
  const double                b,
  std::complex<double>       *addToVec) const;

template void
BLASWrapper<dftfe::utils::MemorySpace::DEVICE>::stridedBlockAxpBy(
  const dftfe::uInt          contiguousBlockSize,
  const dftfe::uInt          numContiguousBlocks,
  const std::complex<float> *addFromVec,
  const double              *scalingVector,
  const double               a,
  const double               b,
  std::complex<float>       *addToVec) const;
