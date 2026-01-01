constexpr int batchSizeDeviceFP64    = 1;
constexpr int subBatchSizeDeviceFP64 = 1;

#ifdef DFTFE_WITH_DEVICE
#  define MatrixFreeTemplates(T)                                        \
    template class dftfe::MatrixFree<double,                            \
                                     dftfe::utils::MemorySpace::DEVICE, \
                                     T,                                 \
                                     T,                                 \
                                     batchSizeDeviceFP64,               \
                                     subBatchSizeDeviceFP64>;

MatrixFreeTemplates(3) MatrixFreeTemplates(4) MatrixFreeTemplates(5)
  MatrixFreeTemplates(6) MatrixFreeTemplates(7) MatrixFreeTemplates(8)
    MatrixFreeTemplates(9) MatrixFreeTemplates(10) MatrixFreeTemplates(11)
      MatrixFreeTemplates(12) MatrixFreeTemplates(13) MatrixFreeTemplates(14)
        MatrixFreeTemplates(15) MatrixFreeTemplates(16) MatrixFreeTemplates(17)

#  undef MatrixFreeTemplates
#endif
