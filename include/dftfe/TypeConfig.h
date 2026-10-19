#ifndef dftfeTypeConfig_h
#define dftfeTypeConfig_h
#include <dftfe/config.h>
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

// Qualifies a function compiled for both the host and the device. Keyed on
// the compiler-mode macros, which are defined only while the device compiler
// is actually compiling the translation unit. The HIP platform macros cannot
// be used here: config.h defines them for every unit of a HIP build,
// including the host-only units that the plain C++ compiler builds, which
// would then see the device attributes.
#if defined(__CUDACC__) || defined(__HIPCC__)
#  define DFTFE_HOST_DEVICE_INLINE __host__ __device__ __forceinline__
#else
#  define DFTFE_HOST_DEVICE_INLINE inline
#endif
#endif
