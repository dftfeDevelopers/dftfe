#include "DeviceTypeConfig.sycl.h"

namespace dftfe
{
  namespace utils
  {
    sycl::queue defaultStream{sycl::default_selector{},
                              sycl::property::queue::in_order{}};
  }
} // namespace dftfe
