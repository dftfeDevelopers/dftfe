constexpr int batchSizeDeviceFP64    = 1;
constexpr int subBatchSizeDeviceFP64 = 1;

#ifdef DFTFE_WITH_DEVICE
#  define MatrixFreeTemplates(NDOFSPERDIM)                              \
    template class dftfe::MatrixFree<double,                            \
                                     dftfe::operatorList::Laplace,      \
                                     dftfe::utils::MemorySpace::DEVICE, \
                                     false,                             \
                                     NDOFSPERDIM,                       \
                                     NDOFSPERDIM,                       \
                                     batchSizeDeviceFP64,               \
                                     subBatchSizeDeviceFP64>;           \
    template class dftfe::MatrixFree<double,                            \
                                     dftfe::operatorList::Helmholtz,    \
                                     dftfe::utils::MemorySpace::DEVICE, \
                                     false,                             \
                                     NDOFSPERDIM,                       \
                                     NDOFSPERDIM,                       \
                                     batchSizeDeviceFP64,               \
                                     subBatchSizeDeviceFP64>;

MatrixFreeTemplates(2) MatrixFreeTemplates(3) MatrixFreeTemplates(4)
  MatrixFreeTemplates(5) MatrixFreeTemplates(6) MatrixFreeTemplates(7)
    MatrixFreeTemplates(8) MatrixFreeTemplates(9) MatrixFreeTemplates(10)
      MatrixFreeTemplates(11) MatrixFreeTemplates(12) MatrixFreeTemplates(13)
        MatrixFreeTemplates(14) MatrixFreeTemplates(15) MatrixFreeTemplates(16)
          MatrixFreeTemplates(17)

#  undef MatrixFreeTemplates
#endif
