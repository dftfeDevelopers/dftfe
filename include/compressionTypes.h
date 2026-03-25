/*
** compressionTypes.h - GPU compressor: backend macros, type definitions,
**                      device helpers, traits, host utilities, and the core
**                      scalar device functions (negabinary, exponent,
**                      quantisation, lifting transform, block padding).
**
** Included by compressionBlockIO.h, compressionZFP.h, compressionBFP.h,
** and compression.h.  Never included directly by application code.
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
     Type definitions
     =========================================================================
   */

  typedef unsigned long long int Word; /* must match ZFP 64-bit word */
  typedef unsigned long long int uint64;

  static constexpr unsigned int WSIZE = sizeof(Word) * CHAR_BIT; /* 64 */

  /* =========================================================================
     Portable device helpers
     =========================================================================
   */

  /* Portable read-only load (__ldg on CUDA, plain load elsewhere) */
  COMPRESSION_DEVICE_INLINE
  Word
  portable_ldg(const Word *ptr)
  {
#if defined(DFTFE_WITH_DEVICE_LANG_CUDA)
    return __ldg(ptr);
#else
    return *ptr;
#endif
  }

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
    static constexpr int EBITS  = 8; /* exponent bits (not counting cont.) */
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
     Host utilities
     =========================================================================
   */

  /* Exact compressed size in bytes */
  inline size_t
  compressed_size(size_t num_values, int bits_per_value)
  {
    const size_t maxbits    = (size_t)bits_per_value * 4; /* bits per block */
    const size_t num_blocks = (num_values + 3) / 4;
    const size_t total_bits = maxbits * num_blocks;
    const size_t num_words  = (total_bits + WSIZE - 1) / WSIZE;
    return num_words * sizeof(Word);
  }

  /* GCD for computing super-block parameters */
  inline unsigned int
  compression_gcd(unsigned int a, unsigned int b)
  {
    while (b)
      {
        unsigned int t = b;
        b              = a % b;
        a              = t;
      }
    return a;
  }

  /* =========================================================================
     Device helpers: negabinary conversion
     =========================================================================
   */

  COMPRESSION_DEVICE_INLINE
  unsigned int
  int2uint(int x)
  {
    return ((unsigned int)x + 0xaaaaaaaau) ^ 0xaaaaaaaau;
  }

  COMPRESSION_DEVICE_INLINE
  unsigned long long int
  int2uint(long long int x)
  {
    return ((unsigned long long int)x + 0xaaaaaaaaaaaaaaaaull) ^
           0xaaaaaaaaaaaaaaaaull;
  }

  COMPRESSION_DEVICE_INLINE
  int
  uint2int(unsigned int x)
  {
    return (int)((x ^ 0xaaaaaaaau) - 0xaaaaaaaau);
  }

  COMPRESSION_DEVICE_INLINE
  long long int
  uint2int(unsigned long long int x)
  {
    return (long long int)((x ^ 0xaaaaaaaaaaaaaaaaull) - 0xaaaaaaaaaaaaaaaaull);
  }

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
     Device: forward lifting transform (1D, 4-vector)
     =========================================================================
   */

  template <typename Int>
  COMPRESSION_DEVICE_INLINE void
  fwd_lift(Int *p)
  {
    Int x = p[0], y = p[1], z = p[2], w = p[3];
    x += w;
    x >>= 1;
    w -= x;
    z += y;
    z >>= 1;
    y -= z;
    x += z;
    x >>= 1;
    z -= x;
    w += y;
    w >>= 1;
    y -= w;
    w += y >> 1;
    y -= w >> 1;
    p[0] = x;
    p[1] = y;
    p[2] = z;
    p[3] = w;
  }

  /* =========================================================================
     Device: inverse lifting transform (1D, 4-vector)
     =========================================================================
   */

  template <typename Int>
  COMPRESSION_DEVICE_INLINE void
  inv_lift(Int *p)
  {
    Int x = p[0], y = p[1], z = p[2], w = p[3];
    y += w >> 1;
    w -= y >> 1;
    y += w;
    w -= y - w;
    z += x;
    x -= z - x;
    y += z;
    z -= y - z;
    w += x;
    x -= w - x;
    p[0] = x;
    p[1] = y;
    p[2] = z;
    p[3] = w;
  }

  /* =========================================================================
     Device: forward quantize (float/double -> int)
     =========================================================================
   */

  template <typename Scalar, typename Int>
  COMPRESSION_DEVICE_INLINE void
  fwd_cast(Int *iblock, const Scalar *fblock, int emax)
  {
    Scalar s = portable_ldexp((Scalar)1.0, traits<Scalar>::PREC - 2 - emax);
    for (int i = 0; i < 4; i++)
      iblock[i] = (Int)(s * fblock[i]);
  }

  /* =========================================================================
     Device: inverse dequantize (int -> float/double)
     =========================================================================
   */

  template <typename Int, typename Scalar>
  COMPRESSION_DEVICE_INLINE Scalar
  dequantize(int emax)
  {
    return portable_ldexp((Scalar)1.0, emax - (traits<Scalar>::PREC - 2));
  }

  /* =========================================================================
     Device: pad partial block (ZFP-compatible, < 4 values)
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
