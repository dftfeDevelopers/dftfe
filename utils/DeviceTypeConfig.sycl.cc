#include "DeviceTypeConfig.sycl.h"

namespace dftfe
{
  namespace utils
  {
    sycl::queue defaultStream{sycl::gpu_selector_v,
                              sycl::property::queue::in_order{}};
  }
} // namespace dftfe
