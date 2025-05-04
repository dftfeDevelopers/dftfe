#ifndef dftfeTypeConfig_h
#define dftfeTypeConfig_h
#include <cstdint>
namespace dftfe
{
#ifdef DFTFE_WITH_64BIT_INT
  using uInt = std::uint64_t;
  using Int  = std::int64_t;
#else
  using uInt = std::uint32_t;
  using Int  = std::int32_t;
#endif
} // namespace dftfe
#endif
