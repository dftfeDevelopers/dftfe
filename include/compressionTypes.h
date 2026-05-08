/*
** compressionTypes.h - GPU compressor: backend macros, type definitions,
**                      device helpers, traits, and the core scalar device
**                      functions used by the BFP encode/decode helpers.
**
** Included by compressionKernels.h and compression.h.
** Never included directly by application code.
*/

#ifndef COMPRESSION_TYPES_H
#define COMPRESSION_TYPES_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <climits>
#include <cfloat>
#include <DeviceTypeConfig.h>

/* =========================================================================
   Backend portability macros
   ========================================================================= */

#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
#  define COMPRESSION_DEVICE_FUNC __device__
#  define COMPRESSION_DEVICE_INLINE __device__ __forceinline__
#  define COMPRESSION_GLOBAL __global__
#  define COMPRESSION_RESTRICT __restrict__
#elif defined(DFTFE_WITH_DEVICE_LANG_HIP)
#  define COMPRESSION_DEVICE_FUNC __device__
#  define COMPRESSION_DEVICE_INLINE __device__ __forceinline__
#  define COMPRESSION_GLOBAL __global__
#  define COMPRESSION_RESTRICT __restrict__
#elif defined(DFTFE_WITH_DEVICE_LANG_SYCL)
#  define COMPRESSION_DEVICE_FUNC
#  define COMPRESSION_DEVICE_INLINE inline
#  define COMPRESSION_GLOBAL
#  define COMPRESSION_RESTRICT
#else
#  error "No device backend defined"
#endif

namespace compression
{

  /* =========================================================================
     Type definitions and launch parameters
     =========================================================================
   */

  typedef unsigned long long int uint64;

  static constexpr int COMPRESSION_BLOCK_SIZE = 256;

  /* =========================================================================
     Portable device helpers
     =========================================================================
   */

  /* Portable math wrappers for device code */
  COMPRESSION_DEVICE_INLINE
  double
  portable_fabs(double x)
  {
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    return sycl::fabs(x);
#else
    return fabs(x);
#endif
  }

  COMPRESSION_DEVICE_INLINE
  float
  portable_fabs(float x)
  {
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    return sycl::fabs(x);
#else
    return fabs(x);
#endif
  }

  COMPRESSION_DEVICE_INLINE
  double
  portable_frexp(double x, int *e)
  {
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    int    exp_val;
    double result = sycl::frexp(x, &exp_val);
    *e            = exp_val;
    return result;
#else
    return frexp(x, e);
#endif
  }

  COMPRESSION_DEVICE_INLINE
  float
  portable_frexp(float x, int *e)
  {
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    int   exp_val;
    float result = sycl::frexp(x, &exp_val);
    *e           = exp_val;
    return result;
#else
    return frexp(x, e);
#endif
  }

  COMPRESSION_DEVICE_INLINE
  double
  portable_ldexp(double x, int e)
  {
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    return sycl::ldexp(x, e);
#else
    return ldexp(x, e);
#endif
  }

  COMPRESSION_DEVICE_INLINE
  float
  portable_ldexp(float x, int e)
  {
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    return sycl::ldexp(x, e);
#else
    return ldexp(x, e);
#endif
  }

  template <typename T>
  COMPRESSION_DEVICE_INLINE T
  portable_max(T a, T b)
  {
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    return sycl::max(a, b);
#else
    return max(a, b);
#endif
  }

  template <typename T>
  COMPRESSION_DEVICE_INLINE T
  portable_min(T a, T b)
  {
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    return sycl::min(a, b);
#else
    return min(a, b);
#endif
  }

  COMPRESSION_DEVICE_INLINE
  int
  portable_max(int a, int b)
  {
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    return sycl::max(a, b);
#else
    return max(a, b);
#endif
  }

  COMPRESSION_DEVICE_INLINE
  int
  portable_min(int a, int b)
  {
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    return sycl::min(a, b);
#else
    return min(a, b);
#endif
  }

  COMPRESSION_DEVICE_INLINE
  unsigned int
  portable_min(unsigned int a, unsigned int b)
  {
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    return sycl::min(a, b);
#else
    return min(a, b);
#endif
  }

  /* =========================================================================
     Portable atomic helpers (for fused decompress+scatter kernels)
     =========================================================================
   */

  COMPRESSION_DEVICE_INLINE void
  portable_atomicAdd(double *addr, double val)
  {
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::atomic_ref<double, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
      ref(*addr);
    ref.fetch_add(val);
#else
    atomicAdd(addr, val);
#endif
  }

  COMPRESSION_DEVICE_INLINE void
  portable_atomicAdd(float *addr, float val)
  {
#if defined(DFTFE_WITH_DEVICE_LANG_SYCL)
    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
      ref(*addr);
    ref.fetch_add(val);
#else
    atomicAdd(addr, val);
#endif
  }

  /* =========================================================================
     Traits: scalar -> integer types and precision constants
     =========================================================================
   */

  template <typename Scalar>
  struct traits;

  template <>
  struct traits<float>
  {
    typedef int          Int;
    typedef unsigned int UInt;
    static constexpr int EBIAS  = 127;
    static constexpr int EBITS  = 8; /* exponent bits */
    static constexpr int PREC   = 32;
    static constexpr int MINEXP = -149;
  };

  template <>
  struct traits<double>
  {
    typedef long long int          Int;
    typedef unsigned long long int UInt;
    static constexpr int           EBIAS  = 1023;
    static constexpr int           EBITS  = 11;
    static constexpr int           PREC   = 64;
    static constexpr int           MINEXP = -1074;
  };

  /* =========================================================================
     Device: exponent computation
     =========================================================================
   */

  template <typename Scalar>
  COMPRESSION_DEVICE_INLINE int
  block_exponent(Scalar x)
  {
    int e = -traits<Scalar>::EBIAS;
    if (x > 0)
      {
        portable_frexp(x, &e);
        e = portable_max(e, 1 - traits<Scalar>::EBIAS);
      }
    return e;
  }

  template <typename Scalar>
  COMPRESSION_DEVICE_INLINE int
  max_exponent(const Scalar *p)
  {
    Scalar mx = 0;
    for (int i = 0; i < 4; i++)
      {
        Scalar f = portable_fabs(p[i]);
        mx       = portable_max(mx, f);
      }
    return block_exponent<Scalar>(mx);
  }

  /* =========================================================================
     Device: precision calculation
     =========================================================================
   */

  COMPRESSION_DEVICE_INLINE
  int
  calc_precision(int maxexp, int maxprec, int minexp)
  {
    return portable_min(maxprec, portable_max(0, maxexp - minexp + 8));
  }

  /* =========================================================================
     Device: pad partial block (< 4 values)
     =========================================================================
   */

  template <typename Scalar>
  COMPRESSION_DEVICE_INLINE void
  pad_block(Scalar *q, unsigned int n)
  {
    if (n == 0)
      q[0] = 0;
    if (n <= 1)
      q[1] = q[0];
    if (n <= 2)
      q[2] = q[1];
    if (n <= 3)
      q[3] = q[n <= 0 ? 0 : n - 1];
  }

} /* namespace compression */

#endif /* COMPRESSION_TYPES_H */
