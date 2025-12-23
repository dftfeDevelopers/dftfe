constexpr int batchSizeDeviceFP32 = 1;
constexpr int batchSizeDeviceFP64 = 1;

template class MatrixFreeDevice<float, 4, 4, batchSizeDeviceFP32>;
template class MatrixFreeDevice<float, 5, 5, batchSizeDeviceFP32>;
template class MatrixFreeDevice<float, 6, 6, batchSizeDeviceFP32>;
template class MatrixFreeDevice<float, 7, 7, batchSizeDeviceFP32>;
template class MatrixFreeDevice<float, 8, 8, batchSizeDeviceFP32>;
template class MatrixFreeDevice<float, 9, 9, batchSizeDeviceFP32>;
template class MatrixFreeDevice<float, 4, 6, batchSizeDeviceFP32>;
template class MatrixFreeDevice<float, 6, 8, batchSizeDeviceFP32>;
template class MatrixFreeDevice<float, 7, 9, batchSizeDeviceFP32>;
template class MatrixFreeDevice<float, 8, 10, batchSizeDeviceFP32>;
template class MatrixFreeDevice<float, 9, 11, batchSizeDeviceFP32>;

template class MatrixFreeDevice<double, 4, 4, batchSizeDeviceFP64>;
template class MatrixFreeDevice<double, 5, 5, batchSizeDeviceFP64>;
template class MatrixFreeDevice<double, 6, 6, batchSizeDeviceFP64>;
template class MatrixFreeDevice<double, 7, 7, batchSizeDeviceFP64>;
template class MatrixFreeDevice<double, 8, 8, batchSizeDeviceFP64>;
template class MatrixFreeDevice<double, 9, 9, batchSizeDeviceFP64>;
template class MatrixFreeDevice<double, 4, 6, batchSizeDeviceFP64>;
template class MatrixFreeDevice<double, 6, 8, batchSizeDeviceFP64>;
template class MatrixFreeDevice<double, 7, 9, batchSizeDeviceFP64>;
template class MatrixFreeDevice<double, 8, 10, batchSizeDeviceFP64>;
template class MatrixFreeDevice<double, 9, 11, batchSizeDeviceFP64>;
