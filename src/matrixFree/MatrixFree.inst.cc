constexpr bool isComplex =
  std::is_same_v<dataTypes::number, std::complex<double>>;

constexpr int batchSizeDeviceFP32    = 1;
constexpr int subBatchSizeDeviceFP32 = 1;

constexpr int batchSizeDeviceFP64    = 1;
constexpr int subBatchSizeDeviceFP64 = 1;

#ifdef DFTFE_WITH_DEVICE
template class dftfe::MatrixFree<float,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 4,
                                 4,
                                 batchSizeDeviceFP32,
                                 subBatchSizeDeviceFP32>;
template class dftfe::MatrixFree<float,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 5,
                                 5,
                                 batchSizeDeviceFP32,
                                 subBatchSizeDeviceFP32>;
template class dftfe::MatrixFree<float,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 6,
                                 6,
                                 batchSizeDeviceFP32,
                                 subBatchSizeDeviceFP32>;
template class dftfe::MatrixFree<float,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 7,
                                 7,
                                 batchSizeDeviceFP32,
                                 subBatchSizeDeviceFP32>;
template class dftfe::MatrixFree<float,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 8,
                                 8,
                                 batchSizeDeviceFP32,
                                 subBatchSizeDeviceFP32>;
template class dftfe::MatrixFree<float,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 9,
                                 9,
                                 batchSizeDeviceFP32,
                                 subBatchSizeDeviceFP32>;

template class dftfe::MatrixFree<float,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 4,
                                 6,
                                 batchSizeDeviceFP32,
                                 subBatchSizeDeviceFP32>;
template class dftfe::MatrixFree<float,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 6,
                                 8,
                                 batchSizeDeviceFP32,
                                 subBatchSizeDeviceFP32>;
template class dftfe::MatrixFree<float,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 7,
                                 9,
                                 batchSizeDeviceFP32,
                                 subBatchSizeDeviceFP32>;
template class dftfe::MatrixFree<float,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 8,
                                 10,
                                 batchSizeDeviceFP32,
                                 subBatchSizeDeviceFP32>;
template class dftfe::MatrixFree<float,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 9,
                                 11,
                                 batchSizeDeviceFP32,
                                 subBatchSizeDeviceFP32>;

template class dftfe::MatrixFree<double,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 4,
                                 4,
                                 batchSizeDeviceFP64,
                                 subBatchSizeDeviceFP64>;
template class dftfe::MatrixFree<double,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 5,
                                 5,
                                 batchSizeDeviceFP64,
                                 subBatchSizeDeviceFP64>;
template class dftfe::MatrixFree<double,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 6,
                                 6,
                                 batchSizeDeviceFP64,
                                 subBatchSizeDeviceFP64>;
template class dftfe::MatrixFree<double,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 7,
                                 7,
                                 batchSizeDeviceFP64,
                                 subBatchSizeDeviceFP64>;
template class dftfe::MatrixFree<double,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 8,
                                 8,
                                 batchSizeDeviceFP64,
                                 subBatchSizeDeviceFP64>;
template class dftfe::MatrixFree<double,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 9,
                                 9,
                                 batchSizeDeviceFP64,
                                 subBatchSizeDeviceFP64>;

template class dftfe::MatrixFree<double,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 4,
                                 6,
                                 batchSizeDeviceFP64,
                                 subBatchSizeDeviceFP64>;
template class dftfe::MatrixFree<double,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 6,
                                 8,
                                 batchSizeDeviceFP64,
                                 subBatchSizeDeviceFP64>;
template class dftfe::MatrixFree<double,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 7,
                                 9,
                                 batchSizeDeviceFP64,
                                 subBatchSizeDeviceFP64>;
template class dftfe::MatrixFree<double,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 8,
                                 10,
                                 batchSizeDeviceFP64,
                                 subBatchSizeDeviceFP64>;
template class dftfe::MatrixFree<double,
                                 dftfe::utils::MemorySpace::DEVICE,
                                 9,
                                 11,
                                 batchSizeDeviceFP64,
                                 subBatchSizeDeviceFP64>;
#endif
