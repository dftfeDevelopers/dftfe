constexpr int batchSizeDeviceFP64 = 1;
constexpr int batchSizeDeviceFP32 = 1;

#ifdef DFTFE_WITH_DEVICE
#  define MatrixFreeDeviceTemplates(NDOFSPERDIM)                    \
    template class MatrixFreeDevice<double,                         \
                                    dftfe::operatorList::Laplace,   \
                                    NDOFSPERDIM,                    \
                                    NDOFSPERDIM,                    \
                                    batchSizeDeviceFP64>;           \
    template class MatrixFreeDevice<double,                         \
                                    dftfe::operatorList::Helmholtz, \
                                    NDOFSPERDIM,                    \
                                    NDOFSPERDIM,                    \
                                    batchSizeDeviceFP64>;

MatrixFreeDeviceTemplates(2) MatrixFreeDeviceTemplates(3)
  MatrixFreeDeviceTemplates(4) MatrixFreeDeviceTemplates(5)
    MatrixFreeDeviceTemplates(6) MatrixFreeDeviceTemplates(7)
      MatrixFreeDeviceTemplates(8) MatrixFreeDeviceTemplates(9)
        MatrixFreeDeviceTemplates(10) MatrixFreeDeviceTemplates(11)
          MatrixFreeDeviceTemplates(12) MatrixFreeDeviceTemplates(13)
            MatrixFreeDeviceTemplates(14) MatrixFreeDeviceTemplates(15)
              MatrixFreeDeviceTemplates(16) MatrixFreeDeviceTemplates(17)

#  undef MatrixFreeDeviceTemplates
#endif
